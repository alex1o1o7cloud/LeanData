import Mathlib

namespace equal_roots_quadratic_l284_28492

/-- If the quadratic equation 2x^2 - 4x + m = 0 has two equal real roots, then m = 2 -/
theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, 2 * x^2 - 4 * x + m = 0 ∧ 
   ∀ y : ℝ, 2 * y^2 - 4 * y + m = 0 → y = x) → 
  m = 2 := by
  sorry

end equal_roots_quadratic_l284_28492


namespace parallel_vectors_x_value_l284_28476

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (1, 3)
  let b : ℝ × ℝ := (2, x + 2)
  parallel a b → x = 4 := by
  sorry

end parallel_vectors_x_value_l284_28476


namespace hyperbola_y_axis_condition_l284_28432

/-- Represents a conic section of the form mx^2 + ny^2 = 1 -/
structure Conic (m n : ℝ) where
  equation : ∀ (x y : ℝ), m * x^2 + n * y^2 = 1

/-- Predicate to check if a conic is a hyperbola with foci on the y-axis -/
def is_hyperbola_y_axis (m n : ℝ) : Prop :=
  m < 0 ∧ n > 0

theorem hyperbola_y_axis_condition (m n : ℝ) :
  (∃ (c : Conic m n), is_hyperbola_y_axis m n) → m * n < 0 ∧
  ∃ (m' n' : ℝ), m' * n' < 0 ∧ ¬∃ (c : Conic m' n'), is_hyperbola_y_axis m' n' :=
by sorry

end hyperbola_y_axis_condition_l284_28432


namespace range_of_a_l284_28419

-- Define the propositions p and q
def p (x : ℝ) : Prop := (2*x - 1) / (x - 1) ≤ 0

def q (x a : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) < 0

-- Define the set of x that satisfy p
def P : Set ℝ := {x | p x}

-- Define the set of x that satisfy q
def Q (a : ℝ) : Set ℝ := {x | q x a}

-- State the theorem
theorem range_of_a :
  (∀ a : ℝ, (P ⊆ Q a ∧ ¬(Q a ⊆ P))) → 
  {a : ℝ | 0 ≤ a ∧ a < 1/2} = {a : ℝ | ∃ x, q x a} :=
sorry

end range_of_a_l284_28419


namespace cube_root_unity_sum_l284_28458

def N (p q r : ℂ) : Matrix (Fin 3) (Fin 3) ℂ :=
  ![![p, q, r],
    ![q, r, p],
    ![r, p, q]]

theorem cube_root_unity_sum (p q r : ℂ) :
  N p q r ^ 3 = 1 →
  p * q * r = -1 →
  p^3 + q^3 + r^3 = -2 ∨ p^3 + q^3 + r^3 = -4 := by
sorry

end cube_root_unity_sum_l284_28458


namespace class_size_calculation_l284_28444

/-- The number of students supposed to be in Miss Smith's second period English class -/
def total_students : ℕ :=
  let tables := 6
  let students_per_table := 3
  let present_students := tables * students_per_table
  let bathroom_students := 3
  let canteen_students := 3 * bathroom_students
  let new_group_size := 4
  let new_groups := 2
  let new_students := new_groups * new_group_size
  let foreign_students := 3 + 3 + 3  -- Germany, France, Norway

  present_students + bathroom_students + canteen_students + new_students + foreign_students

theorem class_size_calculation :
  total_students = 47 := by
  sorry

end class_size_calculation_l284_28444


namespace difference_of_squares_simplification_l284_28422

theorem difference_of_squares_simplification : (365^2 - 349^2) / 16 = 714 := by
  sorry

end difference_of_squares_simplification_l284_28422


namespace catenary_properties_l284_28441

noncomputable def f (a b x : ℝ) : ℝ := a * Real.exp x + b * Real.exp (-x)

theorem catenary_properties (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∀ x, f a b x = f a b (-x) ↔ a = b) ∧
  (∀ x, f a b x = -f a b (-x) ↔ a = -b) ∧
  (a * b < 0 → ∀ x y, x < y → f a b x < f a b y ∨ ∀ x y, x < y → f a b x > f a b y) ∧
  (a * b > 0 → ∃ x, (∀ y, f a b y ≥ f a b x) ∨ (∀ y, f a b y ≤ f a b x)) :=
sorry

end catenary_properties_l284_28441


namespace solution_count_equals_r_l284_28460

def r (n : ℕ) : ℚ := (1/2 : ℚ) * (n + 1 : ℚ) + (1/4 : ℚ) * (1 + (-1)^n : ℚ)

def count_solutions (n : ℕ) : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => p.1 + 2 * p.2 = n) (Finset.product (Finset.range (n + 1)) (Finset.range (n + 1)))).card

theorem solution_count_equals_r (n : ℕ) : 
  (count_solutions n : ℚ) = r n :=
sorry

end solution_count_equals_r_l284_28460


namespace yogurt_combinations_l284_28478

theorem yogurt_combinations (num_flavors : Nat) (num_toppings : Nat) : 
  num_flavors = 6 → num_toppings = 8 → num_flavors * (num_toppings.choose 3) = 336 := by
  sorry

end yogurt_combinations_l284_28478


namespace size_relationship_l284_28448

theorem size_relationship (a b c : ℝ) 
  (ha : a = (0.2 : ℝ) ^ (1.5 : ℝ))
  (hb : b = (2 : ℝ) ^ (0.1 : ℝ))
  (hc : c = (0.2 : ℝ) ^ (1.3 : ℝ)) :
  a < c ∧ c < b :=
by sorry

end size_relationship_l284_28448


namespace ellipse_sum_theorem_l284_28443

/-- Represents an ellipse with center (h, k) and semi-axes a and b -/
structure Ellipse where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- The sum of h, k, a, and b for a specific ellipse -/
def ellipse_sum (e : Ellipse) : ℝ :=
  e.h + e.k + e.a + e.b

/-- Theorem: For an ellipse with center (-3, 5), semi-major axis 7, and semi-minor axis 4,
    the sum h + k + a + b equals 13 -/
theorem ellipse_sum_theorem (e : Ellipse)
    (center_h : e.h = -3)
    (center_k : e.k = 5)
    (semi_major : e.a = 7)
    (semi_minor : e.b = 4) :
    ellipse_sum e = 13 := by
  sorry

end ellipse_sum_theorem_l284_28443


namespace meeting_distance_l284_28403

/-- Represents the distance walked by a person -/
def distance_walked (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: Two people walking towards each other from 35 miles apart, 
    one at 2 mph and the other at 5 mph, will meet when the faster one has walked 25 miles -/
theorem meeting_distance (initial_distance : ℝ) (speed_fred : ℝ) (speed_sam : ℝ) :
  initial_distance = 35 →
  speed_fred = 2 →
  speed_sam = 5 →
  ∃ (time : ℝ), 
    distance_walked speed_fred time + distance_walked speed_sam time = initial_distance ∧
    distance_walked speed_sam time = 25 := by
  sorry

end meeting_distance_l284_28403


namespace real_roots_of_x_squared_minus_four_l284_28413

theorem real_roots_of_x_squared_minus_four (x : ℝ) : x^2 - 4 = 0 ↔ x = 2 ∨ x = -2 := by
  sorry

end real_roots_of_x_squared_minus_four_l284_28413


namespace arithmetic_sequence_common_difference_l284_28414

/-- Given an arithmetic sequence {a_n} with sum of first n terms S_n, prove the common difference is 2. -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (S : ℕ → ℝ)  -- The sum function
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)  -- Definition of S_n for arithmetic sequence
  (h2 : S 4 = 3 * S 2)  -- Given condition
  (h3 : a 7 = 15)  -- Given condition
  : ∃ d : ℝ, ∀ n, a (n + 1) - a n = d ∧ d = 2 :=
by sorry

end arithmetic_sequence_common_difference_l284_28414


namespace negative_square_power_two_l284_28485

theorem negative_square_power_two (a b : ℝ) : (-a^2 * b)^2 = a^4 * b^2 := by sorry

end negative_square_power_two_l284_28485


namespace expected_socks_theorem_l284_28412

/-- The expected number of socks taken until a pair is found -/
def expected_socks (n : ℕ) : ℝ := 2 * n

/-- Theorem: The expected number of socks taken until a pair is found is 2n -/
theorem expected_socks_theorem (n : ℕ) : expected_socks n = 2 * n := by
  sorry

end expected_socks_theorem_l284_28412


namespace increasing_function_bounds_l284_28495

theorem increasing_function_bounds (k : ℕ) (f : ℕ → ℕ) 
  (h_increasing : ∀ m n, m < n → f m < f n) 
  (h_functional : ∀ n, f (f n) = k * n) :
  ∀ n, (2 * k * n) / (k + 1) ≤ f n ∧ f n ≤ ((k + 1) * n) / 2 := by
  sorry

end increasing_function_bounds_l284_28495


namespace absolute_value_equation_solution_l284_28406

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 3| = |x - 5| :=
by
  -- The unique solution is x = 4
  use 4
  sorry

end absolute_value_equation_solution_l284_28406


namespace equilateral_triangle_symmetry_l284_28421

-- Define the shape types
inductive Shape
  | Rectangle
  | Rhombus
  | EquilateralTriangle
  | Circle

-- Define symmetry properties
def hasAxisSymmetry (s : Shape) : Prop :=
  match s with
  | Shape.Rectangle          => true
  | Shape.Rhombus            => true
  | Shape.EquilateralTriangle => true
  | Shape.Circle             => true

def hasCenterSymmetry (s : Shape) : Prop :=
  match s with
  | Shape.Rectangle          => true
  | Shape.Rhombus            => true
  | Shape.EquilateralTriangle => false
  | Shape.Circle             => true

-- Theorem statement
theorem equilateral_triangle_symmetry :
  ∃ (s : Shape), hasAxisSymmetry s ∧ ¬hasCenterSymmetry s ∧
  (∀ (t : Shape), t ≠ s → (hasAxisSymmetry t → hasCenterSymmetry t)) :=
by sorry

end equilateral_triangle_symmetry_l284_28421


namespace distribute_10_8_l284_28438

/-- The number of ways to distribute n distinct objects into k distinct groups,
    with each group containing at least one object -/
def distribute (n k : ℕ) : ℕ := sorry

/-- Stirling number of the second kind: the number of ways to partition
    a set of n elements into k non-empty subsets -/
def stirling2 (n k : ℕ) : ℕ := sorry

theorem distribute_10_8 :
  distribute 10 8 = 30240000 := by sorry

end distribute_10_8_l284_28438


namespace prime_squared_minus_one_divisible_by_thirty_l284_28468

theorem prime_squared_minus_one_divisible_by_thirty
  (p : ℕ) (hp : Nat.Prime p) (hp_ge_seven : p ≥ 7) :
  30 ∣ p^2 - 1 := by
sorry

end prime_squared_minus_one_divisible_by_thirty_l284_28468


namespace smallest_integer_sqrt_difference_l284_28461

theorem smallest_integer_sqrt_difference (n : ℕ) : 
  (∀ m : ℕ, m > 0 → m < 250001 → Real.sqrt m - Real.sqrt (m - 1) ≥ (1 : ℝ) / 1000) ∧ 
  (Real.sqrt 250001 - Real.sqrt 250000 < (1 : ℝ) / 1000) := by
  sorry

end smallest_integer_sqrt_difference_l284_28461


namespace not_necessarily_right_triangle_l284_28426

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)  -- Angles A, B, and C in degrees

-- Define the condition for option D
def angle_ratio (t : Triangle) : Prop :=
  ∃ (k : Real), t.A = 3 * k ∧ t.B = 4 * k ∧ t.C = 5 * k

-- Theorem: If the angles of a triangle satisfy the given ratio, it's not necessarily a right triangle
theorem not_necessarily_right_triangle (t : Triangle) : 
  angle_ratio t → ¬ (t.A = 90 ∨ t.B = 90 ∨ t.C = 90) :=
by
  sorry

-- Note: The proof is omitted as per the instructions

end not_necessarily_right_triangle_l284_28426


namespace track_length_track_length_is_200_l284_28435

/-- The length of a circular track given specific meeting conditions of two runners -/
theorem track_length : ℝ → Prop :=
  fun track_length =>
    ∀ (brenda_speed sally_speed : ℝ),
      brenda_speed > 0 ∧ sally_speed > 0 →
      ∃ (first_meeting_time second_meeting_time : ℝ),
        first_meeting_time > 0 ∧ second_meeting_time > first_meeting_time ∧
        brenda_speed * first_meeting_time = 120 ∧
        brenda_speed * (second_meeting_time - first_meeting_time) = 200 ∧
        (brenda_speed * first_meeting_time + sally_speed * first_meeting_time = track_length / 2) ∧
        (brenda_speed * second_meeting_time + sally_speed * second_meeting_time = 
          track_length + track_length / 2) →
        track_length = 200

/-- The track length is 200 meters -/
theorem track_length_is_200 : track_length 200 := by
  sorry

end track_length_track_length_is_200_l284_28435


namespace c_investment_is_1200_l284_28431

/-- Represents the investment and profit distribution of a business partnership --/
structure BusinessPartnership where
  investmentA : ℕ
  investmentB : ℕ
  investmentC : ℕ
  totalProfit : ℕ
  profitShareC : ℕ

/-- Calculates C's investment amount based on the given conditions --/
def calculateInvestmentC (bp : BusinessPartnership) : ℕ :=
  sorry

/-- Theorem stating that C's investment is 1200 given the specified conditions --/
theorem c_investment_is_1200 : 
  ∀ (bp : BusinessPartnership), 
  bp.investmentA = 800 ∧ 
  bp.investmentB = 1000 ∧ 
  bp.totalProfit = 1000 ∧ 
  bp.profitShareC = 400 →
  calculateInvestmentC bp = 1200 :=
sorry

end c_investment_is_1200_l284_28431


namespace gear_alignment_l284_28475

theorem gear_alignment (n : ℕ) (h1 : n = 6) :
  ∃ (rotation : Fin 32), ∀ (i : Fin n),
    (i.val + rotation : Fin 32) ∉ {j : Fin 32 | j.val < n} :=
sorry

end gear_alignment_l284_28475


namespace correct_subtraction_l284_28401

/-- Given a two-digit number XY and another number Z, prove that the correct subtraction result is 49 -/
theorem correct_subtraction (X Y Z : ℕ) : 
  X = 2 → 
  Y = 4 → 
  Z - 59 = 14 → 
  Z - (10 * X + Y) = 49 := by
sorry

end correct_subtraction_l284_28401


namespace factorization_2m_squared_minus_8_factorization_perfect_square_trinomial_l284_28499

-- Part 1
theorem factorization_2m_squared_minus_8 (m : ℝ) : 
  2 * m^2 - 8 = 2 * (m + 2) * (m - 2) := by sorry

-- Part 2
theorem factorization_perfect_square_trinomial (x y : ℝ) : 
  (x + y)^2 - 4 * (x + y) + 4 = (x + y - 2)^2 := by sorry

end factorization_2m_squared_minus_8_factorization_perfect_square_trinomial_l284_28499


namespace ratio_tr_ur_l284_28407

-- Define the square PQRS
def Square (P Q R S : ℝ × ℝ) : Prop :=
  let (px, py) := P
  let (qx, qy) := Q
  let (rx, ry) := R
  let (sx, sy) := S
  (qx - px)^2 + (qy - py)^2 = 4 ∧
  (rx - qx)^2 + (ry - qy)^2 = 4 ∧
  (sx - rx)^2 + (sy - ry)^2 = 4 ∧
  (px - sx)^2 + (py - sy)^2 = 4

-- Define the quarter circle QS
def QuarterCircle (Q S : ℝ × ℝ) : Prop :=
  let (qx, qy) := Q
  let (sx, sy) := S
  (sx - qx)^2 + (sy - qy)^2 = 4

-- Define U as the midpoint of QR
def Midpoint (U Q R : ℝ × ℝ) : Prop :=
  let (ux, uy) := U
  let (qx, qy) := Q
  let (rx, ry) := R
  ux = (qx + rx) / 2 ∧ uy = (qy + ry) / 2

-- Define T lying on SR
def PointOnLine (T S R : ℝ × ℝ) : Prop :=
  let (tx, ty) := T
  let (sx, sy) := S
  let (rx, ry) := R
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ tx = sx + t * (rx - sx) ∧ ty = sy + t * (ry - sy)

-- Define TU as tangent to the arc QS
def Tangent (T U Q S : ℝ × ℝ) : Prop :=
  let (tx, ty) := T
  let (ux, uy) := U
  let (qx, qy) := Q
  let (sx, sy) := S
  (tx - ux) * (qy - sy) = (ty - uy) * (qx - sx)

-- Theorem statement
theorem ratio_tr_ur (P Q R S T U : ℝ × ℝ) 
  (h1 : Square P Q R S)
  (h2 : QuarterCircle Q S)
  (h3 : Midpoint U Q R)
  (h4 : PointOnLine T S R)
  (h5 : Tangent T U Q S) :
  let (tx, ty) := T
  let (rx, ry) := R
  let (ux, uy) := U
  (tx - rx)^2 + (ty - ry)^2 = 16/9 * ((ux - rx)^2 + (uy - ry)^2) := by sorry

end ratio_tr_ur_l284_28407


namespace midsegment_inequality_l284_28462

/-- Midsegment theorem for triangles -/
theorem midsegment_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  let perimeter := a + b + c
  let midsegment_sum := (b + c) / 2 + (a + c) / 2 + (a + b) / 2
  midsegment_sum < perimeter ∧ midsegment_sum > 3 / 4 * perimeter :=
by sorry

end midsegment_inequality_l284_28462


namespace patanjali_speed_l284_28424

/-- Represents Patanjali's walking data over three days -/
structure WalkingData where
  speed_day1 : ℝ
  hours_day1 : ℝ
  total_distance : ℝ

/-- Conditions for Patanjali's walking problem -/
def walking_conditions (data : WalkingData) : Prop :=
  data.speed_day1 * data.hours_day1 = 18 ∧
  (data.speed_day1 + 1) * (data.hours_day1 - 1) + (data.speed_day1 + 1) * data.hours_day1 = data.total_distance - 18 ∧
  data.total_distance = 62

/-- Theorem stating that Patanjali's speed on the first day was 9 miles per hour -/
theorem patanjali_speed (data : WalkingData) 
  (h : walking_conditions data) : data.speed_day1 = 9 := by
  sorry

end patanjali_speed_l284_28424


namespace largest_n_divisible_by_three_l284_28411

def is_divisible_by_three (n : ℕ) : Prop :=
  ∃ k : ℤ, 9 * (n - 1)^3 - 3 * n^3 + 19 * n + 27 = 3 * k

theorem largest_n_divisible_by_three :
  (∀ m : ℕ, m < 50000 → is_divisible_by_three m → m ≤ 49998) ∧
  (49998 < 50000) ∧
  is_divisible_by_three 49998 :=
sorry

end largest_n_divisible_by_three_l284_28411


namespace total_suitcase_weight_is_434_l284_28442

/-- The total weight of all suitcases for a family vacation --/
def total_suitcase_weight : ℕ :=
  let siblings_suitcases := List.range 6 |>.sum
  let siblings_weight := siblings_suitcases * 10
  let parents_suitcases := 2 * 3
  let parents_weight := parents_suitcases * 12
  let grandparents_suitcases := 2 * 2
  let grandparents_weight := grandparents_suitcases * 8
  let relatives_suitcases := 8
  let relatives_weight := relatives_suitcases * 15
  siblings_weight + parents_weight + grandparents_weight + relatives_weight

theorem total_suitcase_weight_is_434 : total_suitcase_weight = 434 := by
  sorry

end total_suitcase_weight_is_434_l284_28442


namespace sixth_term_is_27_eighth_term_is_46_l284_28466

-- First sequence
def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem sixth_term_is_27 : arithmetic_sequence 2 5 6 = 27 := by sorry

-- Second sequence
def even_indexed_term (n : ℕ) : ℕ → ℕ
  | 0 => 4
  | m + 1 => 2 * even_indexed_term n m + 1

def odd_indexed_term (n : ℕ) : ℕ → ℕ
  | 0 => 2
  | m + 1 => 2 * odd_indexed_term n m + 2

def combined_sequence (n : ℕ) : ℕ :=
  if n % 2 = 0 then even_indexed_term (n / 2) (n / 2 - 1)
  else odd_indexed_term ((n + 1) / 2) ((n - 1) / 2)

theorem eighth_term_is_46 : combined_sequence 8 = 46 := by sorry

end sixth_term_is_27_eighth_term_is_46_l284_28466


namespace no_food_left_for_dog_l284_28474

theorem no_food_left_for_dog (N : ℕ) (prepared_food : ℝ) : 
  let stayed := N / 3
  let excursion := 2 * N / 3
  let lunch_portion := prepared_food / 4
  let excursion_portion := 1.5 * lunch_portion
  stayed * lunch_portion + excursion * excursion_portion = prepared_food :=
by sorry

end no_food_left_for_dog_l284_28474


namespace notebook_difference_proof_l284_28449

/-- The price of notebooks Jeremy bought -/
def jeremy_total : ℚ := 180 / 100

/-- The price of notebooks Tina bought -/
def tina_total : ℚ := 300 / 100

/-- The difference in the number of notebooks bought by Tina and Jeremy -/
def notebook_difference : ℕ := 4

/-- The price of a single notebook -/
def notebook_price : ℚ := 30 / 100

theorem notebook_difference_proof :
  ∃ (jeremy_count tina_count : ℕ),
    jeremy_count * notebook_price = jeremy_total ∧
    tina_count * notebook_price = tina_total ∧
    tina_count - jeremy_count = notebook_difference :=
by sorry

end notebook_difference_proof_l284_28449


namespace no_cube_sum_4099_l284_28494

theorem no_cube_sum_4099 : 
  ∀ a b : ℤ, a^3 + b^3 ≠ 4099 :=
by
  sorry

#check no_cube_sum_4099

end no_cube_sum_4099_l284_28494


namespace ratio_change_after_subtraction_l284_28400

theorem ratio_change_after_subtraction (a b : ℕ) (h1 : a * 5 = b * 6) (h2 : a > 5 ∧ b > 5) 
  (h3 : (a - 5) - (b - 5) = 5) : (a - 5) * 4 = (b - 5) * 5 := by
  sorry

end ratio_change_after_subtraction_l284_28400


namespace bhaskar_tour_days_l284_28416

def total_budget : ℕ := 360
def extension_days : ℕ := 4
def expense_reduction : ℕ := 3

theorem bhaskar_tour_days :
  ∃ (x : ℕ), x > 0 ∧
  (total_budget / x : ℚ) - expense_reduction = (total_budget / (x + extension_days) : ℚ) ∧
  x = 20 := by
  sorry

end bhaskar_tour_days_l284_28416


namespace find_integers_with_sum_and_lcm_l284_28457

theorem find_integers_with_sum_and_lcm : ∃ (a b : ℕ+), 
  (a + b : ℕ) = 3972 ∧ 
  Nat.lcm a b = 985928 ∧ 
  a = 1964 ∧ 
  b = 2008 := by
sorry

end find_integers_with_sum_and_lcm_l284_28457


namespace expression_evaluation_l284_28479

theorem expression_evaluation :
  let a : ℝ := 2 + Real.sqrt 3
  (a - 1 - (2 * a - 1) / (a + 1)) / ((a^2 - 4 * a + 4) / (a + 1)) = (2 * Real.sqrt 3 + 3) / 3 := by
  sorry

end expression_evaluation_l284_28479


namespace virginia_adrienne_difference_l284_28481

/-- Represents the teaching years of Virginia, Adrienne, and Dennis -/
structure TeachingYears where
  virginia : ℕ
  adrienne : ℕ
  dennis : ℕ

/-- The conditions of the teaching years problem -/
def TeachingProblem (t : TeachingYears) : Prop :=
  t.virginia + t.adrienne + t.dennis = 75 ∧
  t.dennis = 34 ∧
  ∃ (x : ℕ), t.virginia = t.adrienne + x ∧ t.virginia = t.dennis - x

/-- The theorem stating that Virginia has taught 9 more years than Adrienne -/
theorem virginia_adrienne_difference (t : TeachingYears) 
  (h : TeachingProblem t) : t.virginia - t.adrienne = 9 := by
  sorry

end virginia_adrienne_difference_l284_28481


namespace union_determines_m_l284_28482

theorem union_determines_m (A B : Set ℕ) (m : ℕ) : 
  A = {1, 3, m} → 
  B = {3, 4} → 
  A ∪ B = {1, 2, 3, 4} → 
  m = 2 := by
sorry

end union_determines_m_l284_28482


namespace circle_diameter_l284_28469

/-- The diameter of a circle is twice its radius -/
theorem circle_diameter (r : ℝ) (d : ℝ) (h : r = 7) : d = 14 ↔ d = 2 * r := by sorry

end circle_diameter_l284_28469


namespace system_solution_l284_28446

theorem system_solution (x y : ℝ) : 
  (x = 1 ∧ y = 4) → 
  (Real.sqrt (y / x) - 2 * Real.sqrt (x / y) = 1 ∧ 
   Real.sqrt (5 * x + y) + Real.sqrt (5 * x - y) = 4) := by
  sorry

end system_solution_l284_28446


namespace sum_g_equals_negative_one_l284_28447

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- State the conditions
axiom functional_equation : ∀ x y : ℝ, f (x - y) = f x * g y - g x * f y
axiom f_equality : f (-2) = f 1
axiom f_nonzero : f 1 ≠ 0

-- State the theorem to be proved
theorem sum_g_equals_negative_one : g 1 + g (-1) = -1 := by sorry

end sum_g_equals_negative_one_l284_28447


namespace double_age_in_two_years_l284_28464

/-- The number of years it takes for a man's age to be twice his son's age -/
def years_until_double_age (son_age : ℕ) (age_difference : ℕ) : ℕ :=
  let man_age := son_age + age_difference
  (man_age - 2 * son_age) / (2 - 1)

theorem double_age_in_two_years (son_age : ℕ) (age_difference : ℕ) 
  (h1 : son_age = 18) (h2 : age_difference = 20) : 
  years_until_double_age son_age age_difference = 2 := by
  sorry

end double_age_in_two_years_l284_28464


namespace arithmetic_series_sum_specific_l284_28409

def arithmetic_series_sum (a₁ aₙ d : ℚ) : ℚ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_series_sum_specific :
  arithmetic_series_sum 12 50 (1/10) = 11811 := by
  sorry

end arithmetic_series_sum_specific_l284_28409


namespace clothing_profit_l284_28418

theorem clothing_profit (price : ℝ) (profit_percent : ℝ) (loss_percent : ℝ) : 
  price = 180 ∧ 
  profit_percent = 20 ∧ 
  loss_percent = 10 → 
  (2 * price) - (price / (1 + profit_percent / 100) + price / (1 - loss_percent / 100)) = 10 := by
  sorry

end clothing_profit_l284_28418


namespace tomato_egg_soup_min_time_l284_28465

/-- Represents a cooking step with its duration -/
structure CookingStep where
  name : String
  duration : ℕ

/-- The set of cooking steps for Tomato Egg Soup -/
def tomatoEggSoupSteps : List CookingStep := [
  ⟨"A", 1⟩,
  ⟨"B", 2⟩,
  ⟨"C", 3⟩,
  ⟨"D", 1⟩,
  ⟨"E", 1⟩
]

/-- Calculates the minimum time required to complete all cooking steps -/
def minCookingTime (steps : List CookingStep) : ℕ := sorry

/-- Theorem: The minimum time to make Tomato Egg Soup is 6 minutes -/
theorem tomato_egg_soup_min_time :
  minCookingTime tomatoEggSoupSteps = 6 := by sorry

end tomato_egg_soup_min_time_l284_28465


namespace annie_initial_money_l284_28451

/-- Annie's hamburger and milkshake purchase problem -/
theorem annie_initial_money :
  let hamburger_price : ℕ := 4
  let milkshake_price : ℕ := 3
  let hamburgers_bought : ℕ := 8
  let milkshakes_bought : ℕ := 6
  let money_left : ℕ := 70
  let initial_money : ℕ := hamburger_price * hamburgers_bought + milkshake_price * milkshakes_bought + money_left
  initial_money = 120 := by sorry

end annie_initial_money_l284_28451


namespace shares_owned_problem_solution_l284_28437

/-- A function that calculates the dividend per share based on actual earnings --/
def dividend_per_share (expected_earnings : ℚ) (actual_earnings : ℚ) : ℚ :=
  let base_dividend := expected_earnings / 2
  let additional_earnings := max (actual_earnings - expected_earnings) 0
  let additional_dividend := (additional_earnings / (1/10)) * (4/100)
  base_dividend + additional_dividend

theorem shares_owned (expected_earnings actual_earnings total_dividend : ℚ) : ℚ :=
  total_dividend / (dividend_per_share expected_earnings actual_earnings)

/-- Proves the number of shares owned given the problem conditions --/
theorem problem_solution :
  let expected_earnings : ℚ := 80/100
  let actual_earnings : ℚ := 110/100
  let total_dividend : ℚ := 260
  shares_owned expected_earnings actual_earnings total_dividend = 500 := by
  sorry

end shares_owned_problem_solution_l284_28437


namespace relationship_increases_with_ratio_difference_l284_28434

-- Define the structure for a 2x2 contingency table
structure ContingencyTable :=
  (a b c d : ℕ)

-- Define the ratios
def ratio1 (t : ContingencyTable) : ℚ := t.a / (t.a + t.b)
def ratio2 (t : ContingencyTable) : ℚ := t.c / (t.c + t.d)

-- Define the difference between ratios
def ratioDifference (t : ContingencyTable) : ℚ := |ratio1 t - ratio2 t|

-- Define a measure of relationship possibility (e.g., chi-square value)
noncomputable def relationshipPossibility (t : ContingencyTable) : ℝ := sorry

-- State the theorem
theorem relationship_increases_with_ratio_difference (t : ContingencyTable) :
  ∀ (t1 t2 : ContingencyTable),
    ratioDifference t1 < ratioDifference t2 →
    relationshipPossibility t1 < relationshipPossibility t2 :=
sorry

end relationship_increases_with_ratio_difference_l284_28434


namespace tangent_y_intercept_l284_28420

/-- The curve function f(x) = x^2 + 11 -/
def f (x : ℝ) : ℝ := x^2 + 11

/-- The point P on the curve -/
def P : ℝ × ℝ := (1, 12)

/-- The slope of the tangent line at P -/
def m : ℝ := 2 * P.1

/-- The y-intercept of the tangent line -/
def b : ℝ := P.2 - m * P.1

theorem tangent_y_intercept :
  b = 10 := by sorry

end tangent_y_intercept_l284_28420


namespace notebook_price_l284_28471

theorem notebook_price (notebook_count : ℕ) (pencil_price pen_price total_spent : ℚ) : 
  notebook_count = 3 →
  pencil_price = 1.5 →
  pen_price = 1.7 →
  total_spent = 6.8 →
  ∃ (notebook_price : ℚ), 
    notebook_count * notebook_price + pencil_price + pen_price = total_spent ∧
    notebook_price = 1.2 := by
  sorry

end notebook_price_l284_28471


namespace probability_of_dime_l284_28488

/-- Represents the types of coins in the jar -/
inductive Coin
  | Dime
  | Nickel
  | Penny

/-- The value of each coin type in cents -/
def coinValue : Coin → ℚ
  | Coin.Dime => 10
  | Coin.Nickel => 5
  | Coin.Penny => 1

/-- The total value of each coin type in cents -/
def totalValue : Coin → ℚ
  | Coin.Dime => 800
  | Coin.Nickel => 700
  | Coin.Penny => 500

/-- The number of coins of each type in the jar -/
def coinCount (c : Coin) : ℚ :=
  totalValue c / coinValue c

/-- The total number of coins in the jar -/
def totalCoins : ℚ :=
  coinCount Coin.Dime + coinCount Coin.Nickel + coinCount Coin.Penny

/-- The probability of randomly selecting a dime from the jar -/
theorem probability_of_dime : 
  coinCount Coin.Dime / totalCoins = 1 / 9 := by
  sorry


end probability_of_dime_l284_28488


namespace arithmetic_mean_of_special_set_l284_28425

theorem arithmetic_mean_of_special_set (n : ℕ) (h : n > 1) :
  let set_size := 2 * n
  let special_num := 1 + 1 / n
  let regular_num := 1
  let sum := (set_size - 1) * regular_num + special_num
  sum / set_size = 1 + 1 / (2 * n^2) := by
  sorry

end arithmetic_mean_of_special_set_l284_28425


namespace meeting_point_one_third_distance_l284_28455

/-- Given two points in a 2D plane, this function calculates a point that is a fraction of the distance from the first point to the second point. -/
def intermediatePoint (x1 y1 x2 y2 t : ℝ) : ℝ × ℝ :=
  (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

/-- Theorem stating that the point (10, 5) is one-third of the way from (8, 3) to (14, 9). -/
theorem meeting_point_one_third_distance :
  intermediatePoint 8 3 14 9 (1/3) = (10, 5) := by
sorry

end meeting_point_one_third_distance_l284_28455


namespace f_zeros_f_min_max_on_interval_l284_28463

def f (x : ℝ) : ℝ := x^2 + x - 2

theorem f_zeros (x : ℝ) : f x = 0 ↔ x = 1 ∨ x = -2 := by sorry

theorem f_min_max_on_interval :
  let a : ℝ := -1
  let b : ℝ := 1
  (∀ x ∈ Set.Icc a b, f x ≥ -9/4) ∧
  (∃ x ∈ Set.Icc a b, f x = -9/4) ∧
  (∀ x ∈ Set.Icc a b, f x ≤ 0) ∧
  (∃ x ∈ Set.Icc a b, f x = 0) := by sorry

end f_zeros_f_min_max_on_interval_l284_28463


namespace problem_solution_l284_28433

theorem problem_solution (x y : ℝ) 
  (h1 : x = 51) 
  (h2 : x^3*y - 2*x^2*y + x*y = 127500) : 
  y = 1 := by
sorry

end problem_solution_l284_28433


namespace sum_of_coefficients_after_shift_l284_28453

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a quadratic function horizontally -/
def shift_left (f : QuadraticFunction) (units : ℝ) : QuadraticFunction :=
  QuadraticFunction.mk
    f.a
    (f.b + 2 * f.a * units)
    (f.a * units^2 + f.b * units + f.c)

/-- The original quadratic function y = 3x^2 + 2x - 5 -/
def original : QuadraticFunction :=
  QuadraticFunction.mk 3 2 (-5)

/-- The shifted quadratic function -/
def shifted : QuadraticFunction :=
  shift_left original 6

/-- Theorem stating that the sum of coefficients of the shifted function is 156 -/
theorem sum_of_coefficients_after_shift :
  shifted.a + shifted.b + shifted.c = 156 := by
  sorry

end sum_of_coefficients_after_shift_l284_28453


namespace nested_expression_value_l284_28497

theorem nested_expression_value : (3 * (3 * (3 * (3 * (3 * (3 + 2) + 2) + 2) + 2) + 2) + 2) = 1457 := by
  sorry

end nested_expression_value_l284_28497


namespace product_expansion_sum_l284_28480

theorem product_expansion_sum (a b c d : ℝ) : 
  (∀ x : ℝ, (4 * x^2 - 3 * x + 2) * (5 - x) = a * x^3 + b * x^2 + c * x + d) →
  9 * a + 3 * b + c + d = 26 := by
sorry

end product_expansion_sum_l284_28480


namespace both_players_is_zero_l284_28459

/-- The number of people who play both kabadi and kho kho -/
def both_players : ℕ := sorry

/-- The number of people who play kabadi (including those who play both) -/
def kabadi_players : ℕ := 10

/-- The number of people who play kho kho only -/
def kho_kho_only_players : ℕ := 15

/-- The total number of players -/
def total_players : ℕ := 25

/-- Theorem stating that the number of people playing both games is 0 -/
theorem both_players_is_zero : both_players = 0 := by
  sorry

#check both_players_is_zero

end both_players_is_zero_l284_28459


namespace factorization_equality_l284_28496

theorem factorization_equality (a : ℝ) : 
  (2 / 9) * a^2 - (4 / 3) * a + 2 = (2 / 9) * (a - 3)^2 := by
  sorry

end factorization_equality_l284_28496


namespace min_value_product_l284_28402

theorem min_value_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 8) :
  (a + 3 * b) * (b + 3 * c) * (a * c + 3) ≥ 48 := by
  sorry

end min_value_product_l284_28402


namespace problem_solution_l284_28491

def quadratic_equation (x : ℝ) : Prop := x^2 - 5*x + 1 = 0

def inequality_system (x : ℝ) : Prop :=
  x + 8 < 4*x - 1 ∧ (1/2)*x ≤ 8 - (3/2)*x

theorem problem_solution :
  (∃ x₁ x₂ : ℝ, x₁ = (5 + Real.sqrt 21) / 2 ∧
                x₂ = (5 - Real.sqrt 21) / 2 ∧
                quadratic_equation x₁ ∧
                quadratic_equation x₂) ∧
  (∀ x : ℝ, inequality_system x ↔ 3 < x ∧ x ≤ 4) :=
by sorry

end problem_solution_l284_28491


namespace smallest_n_satisfying_conditions_l284_28408

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

theorem smallest_n_satisfying_conditions :
  ∃! n : ℕ, n ≥ 10 ∧ 
            is_prime (n + 6) ∧ 
            is_perfect_square (9*n + 7) ∧
            ∀ m : ℕ, m ≥ 10 → 
                     is_prime (m + 6) → 
                     is_perfect_square (9*m + 7) → 
                     n ≤ m ∧
            n = 53 :=
sorry

end smallest_n_satisfying_conditions_l284_28408


namespace solve_fraction_equation_l284_28450

theorem solve_fraction_equation (y : ℚ) (h : (1:ℚ)/3 - (1:ℚ)/4 = 1/y) : y = 12 := by
  sorry

end solve_fraction_equation_l284_28450


namespace intersection_implies_a_values_l284_28498

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}

def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + (a^2-5) = 0}

theorem intersection_implies_a_values (a : ℝ) : A ∩ B a = {2} → a = -1 ∨ a = -3 := by
  sorry

end intersection_implies_a_values_l284_28498


namespace arithmetic_sequence_proof_l284_28486

def a (n : ℕ) := 2 * (n + 1) + 3

theorem arithmetic_sequence_proof :
  ∀ n : ℕ, a (n + 1) - a n = 2 :=
by
  sorry

end arithmetic_sequence_proof_l284_28486


namespace sum_sqrt_inequality_l284_28440

theorem sum_sqrt_inequality (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 1) : 
  Real.sqrt (x / (1 - x)) + Real.sqrt (y / (1 - y)) + Real.sqrt (z / (1 - z)) > 2 := by
  sorry

end sum_sqrt_inequality_l284_28440


namespace intersection_equality_subset_relation_l284_28439

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 5}
def B (a : ℝ) : Set ℝ := {x | -1-2*a ≤ x ∧ x ≤ a-2}

-- Theorem for part (1)
theorem intersection_equality (a : ℝ) : A ∩ B a = A ↔ a ≥ 7 := by sorry

-- Theorem for part (2)
theorem subset_relation (a : ℝ) : (∀ x, x ∈ B a → x ∈ A) ↔ a < 1/3 := by sorry

end intersection_equality_subset_relation_l284_28439


namespace cory_fruit_sequences_l284_28487

/-- The number of distinct sequences for eating fruits -/
def fruitSequences (apples oranges bananas pears : ℕ) : ℕ :=
  let total := apples + oranges + bananas + pears
  Nat.factorial total / (Nat.factorial apples * Nat.factorial oranges * Nat.factorial bananas * Nat.factorial pears)

/-- Theorem: The number of distinct sequences for eating 4 apples, 2 oranges, 1 banana, and 2 pears over 8 days is 420 -/
theorem cory_fruit_sequences :
  fruitSequences 4 2 1 2 = 420 := by
  sorry

#eval fruitSequences 4 2 1 2

end cory_fruit_sequences_l284_28487


namespace antenna_spire_height_l284_28472

/-- The height of the Empire State Building's antenna spire -/
theorem antenna_spire_height :
  let total_height : ℕ := 1454
  let top_floor_height : ℕ := 1250
  let antenna_height := total_height - top_floor_height
  antenna_height = 204 :=
by sorry

end antenna_spire_height_l284_28472


namespace line_equation_with_triangle_area_l284_28428

/-- The equation of a line passing through two points and forming a triangle -/
theorem line_equation_with_triangle_area 
  (b S : ℝ) (hb : b ≠ 0) (hS : S > 0) :
  let k := 2 * S / b
  let line_eq := fun (x y : ℝ) ↦ 2 * S * x - b^2 * y + 2 * b * S
  (∀ y, line_eq (-b) y = 0) ∧ 
  (∀ x, line_eq x k = 0) ∧
  (∃ x y, x < 0 ∧ y > 0 ∧ line_eq x y = 0) ∧
  (S = (1/2) * b * k) :=
by sorry

end line_equation_with_triangle_area_l284_28428


namespace cubic_inequality_range_l284_28489

theorem cubic_inequality_range (a : ℝ) : 
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, x^3 - a*x + 1 ≥ 0) → 
  0 ≤ a ∧ a ≤ (3 * 2^(1/3)) / 2 :=
by sorry

end cubic_inequality_range_l284_28489


namespace line_symmetry_l284_28404

/-- Given two lines l₁ and l₂ in the xy-plane, prove that if the angle bisector between them
    is y = x, and l₁ has the equation x + 2y + 3 = 0, then l₂ has the equation 2x + y + 3 = 0. -/
theorem line_symmetry (l₁ l₂ : Set (ℝ × ℝ)) : 
  (∀ p : ℝ × ℝ, p ∈ l₁ ↔ p.1 + 2 * p.2 + 3 = 0) →
  (∀ p : ℝ × ℝ, p ∈ l₂ ↔ 2 * p.1 + p.2 + 3 = 0) →
  (∀ p : ℝ × ℝ, p ∈ l₁ ∨ p ∈ l₂ → p.1 = p.2 → 
    ∃ q : ℝ × ℝ, (q ∈ l₁ ∧ q.1 + q.2 = p.1 + p.2) ∨ (q ∈ l₂ ∧ q.1 + q.2 = p.1 + p.2)) :=
by
  sorry

end line_symmetry_l284_28404


namespace josh_gummy_bears_l284_28470

theorem josh_gummy_bears (initial_candies : ℕ) : 
  (∃ (remaining_after_siblings : ℕ) (remaining_after_friend : ℕ),
    initial_candies = 3 * 10 + remaining_after_siblings ∧
    remaining_after_siblings = 2 * remaining_after_friend ∧
    remaining_after_friend = 16 + 19) →
  initial_candies = 100 := by
sorry

end josh_gummy_bears_l284_28470


namespace decimal_equivalent_of_one_fourth_squared_l284_28417

theorem decimal_equivalent_of_one_fourth_squared :
  (1 / 4 : ℚ) ^ 2 = (0.0625 : ℚ) := by sorry

end decimal_equivalent_of_one_fourth_squared_l284_28417


namespace circle_inscribed_line_intersection_l284_28456

-- Define the angle
variable (angle : Angle)

-- Define the circles
variable (ω Ω : Circle)

-- Define the line
variable (l : Line)

-- Define the points
variable (A B C D E F : Point)

-- Define the inscribed property
def inscribed (c : Circle) (α : Angle) : Prop := sorry

-- Define the intersection property
def intersects (l : Line) (c : Circle) (p q : Point) : Prop := sorry

-- Define the order of points on a line
def ordered_on_line (l : Line) (p₁ p₂ p₃ p₄ p₅ p₆ : Point) : Prop := sorry

-- Define the equality of line segments
def segment_eq (p₁ p₂ q₁ q₂ : Point) : Prop := sorry

theorem circle_inscribed_line_intersection 
  (h₁ : inscribed ω angle)
  (h₂ : inscribed Ω angle)
  (h₃ : intersects l angle A F)
  (h₄ : intersects l ω B C)
  (h₅ : intersects l Ω D E)
  (h₆ : ordered_on_line l A B C D E F)
  (h₇ : segment_eq B C D E) :
  segment_eq A B E F := by sorry

end circle_inscribed_line_intersection_l284_28456


namespace goldfinch_percentage_is_30_percent_l284_28484

def number_of_goldfinches : ℕ := 6
def number_of_sparrows : ℕ := 9
def number_of_grackles : ℕ := 5

def total_birds : ℕ := number_of_goldfinches + number_of_sparrows + number_of_grackles

def goldfinch_percentage : ℚ := (number_of_goldfinches : ℚ) / (total_birds : ℚ) * 100

theorem goldfinch_percentage_is_30_percent : goldfinch_percentage = 30 := by
  sorry

end goldfinch_percentage_is_30_percent_l284_28484


namespace intersection_when_a_is_two_empty_intersection_iff_a_in_range_l284_28477

def A (a : ℝ) : Set ℝ := {x : ℝ | |x - 1| ≤ a ∧ a > 0}

def B : Set ℝ := {x : ℝ | x > 2 ∨ x < -2}

theorem intersection_when_a_is_two :
  A 2 ∩ B = {x : ℝ | 2 < x ∧ x ≤ 3} := by sorry

theorem empty_intersection_iff_a_in_range :
  ∀ a : ℝ, A a ∩ B = ∅ ↔ 0 < a ∧ a ≤ 1 := by sorry

end intersection_when_a_is_two_empty_intersection_iff_a_in_range_l284_28477


namespace larger_number_of_pair_l284_28493

theorem larger_number_of_pair (x y : ℝ) : 
  x - y = 7 → x + y = 47 → max x y = 27 := by sorry

end larger_number_of_pair_l284_28493


namespace inverse_proportion_change_l284_28427

/-- Given positive numbers x and y that are inversely proportional, prove that when x doubles, y decreases by 50% -/
theorem inverse_proportion_change (x y x' y' k : ℝ) :
  x > 0 →
  y > 0 →
  x * y = k →
  x' = 2 * x →
  x' * y' = k →
  y' / y = 1/2 := by
  sorry

end inverse_proportion_change_l284_28427


namespace tim_took_eleven_rulers_l284_28490

/-- The number of rulers initially in the drawer -/
def initial_rulers : ℕ := 14

/-- The number of rulers left in the drawer after Tim took some out -/
def remaining_rulers : ℕ := 3

/-- The number of rulers Tim took out -/
def rulers_taken : ℕ := initial_rulers - remaining_rulers

theorem tim_took_eleven_rulers : rulers_taken = 11 := by
  sorry

end tim_took_eleven_rulers_l284_28490


namespace bacon_vs_mashed_potatoes_l284_28405

theorem bacon_vs_mashed_potatoes (mashed_potatoes bacon : ℕ) 
  (h1 : mashed_potatoes = 479) 
  (h2 : bacon = 489) : 
  bacon - mashed_potatoes = 10 := by
  sorry

end bacon_vs_mashed_potatoes_l284_28405


namespace crayons_per_child_l284_28445

theorem crayons_per_child (total_children : ℕ) (total_crayons : ℕ) 
  (h1 : total_children = 10) 
  (h2 : total_crayons = 50) : 
  total_crayons / total_children = 5 := by
  sorry

end crayons_per_child_l284_28445


namespace max_sum_digits_24hour_clock_l284_28430

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Fin 24
  minutes : Fin 60

/-- Calculates the sum of digits in a natural number -/
def sumDigits (n : ℕ) : ℕ := sorry

/-- Checks if a natural number is even -/
def isEven (n : ℕ) : Prop := ∃ k, n = 2 * k

/-- Calculates the total sum of digits in a Time24 -/
def totalSumDigits (t : Time24) : ℕ :=
  sumDigits t.hours.val + sumDigits t.minutes.val

/-- The theorem to be proved -/
theorem max_sum_digits_24hour_clock :
  ∃ (t : Time24), 
    (isEven (sumDigits t.hours.val)) ∧ 
    (∀ (t' : Time24), isEven (sumDigits t'.hours.val) → totalSumDigits t' ≤ totalSumDigits t) ∧
    totalSumDigits t = 22 := by sorry

end max_sum_digits_24hour_clock_l284_28430


namespace complement_of_M_in_U_l284_28429

def U : Set ℕ := {2011, 2012, 2013, 2014, 2015}
def M : Set ℕ := {2011, 2012, 2013}

theorem complement_of_M_in_U : U \ M = {2014, 2015} := by sorry

end complement_of_M_in_U_l284_28429


namespace teresas_current_age_l284_28473

/-- Given the ages of family members at different points in time, 
    prove Teresa's current age. -/
theorem teresas_current_age 
  (morio_current_age : ℕ)
  (morio_age_at_birth : ℕ)
  (teresa_age_at_birth : ℕ)
  (h1 : morio_current_age = 71)
  (h2 : morio_age_at_birth = 38)
  (h3 : teresa_age_at_birth = 26) :
  teresa_age_at_birth + (morio_current_age - morio_age_at_birth) = 59 :=
by sorry

end teresas_current_age_l284_28473


namespace unique_prime_pair_l284_28452

def isPrime (n : ℕ) : Prop := sorry

def nthPrime (n : ℕ) : ℕ := sorry

theorem unique_prime_pair :
  ∀ a b : ℕ, 
    a > 0 → b > 0 → 
    a - b ≥ 2 → 
    (nthPrime a - nthPrime b) ∣ (2 * (a - b)) → 
    a = 4 ∧ b = 2 := by sorry

end unique_prime_pair_l284_28452


namespace impossibility_of_three_similar_parts_l284_28467

theorem impossibility_of_three_similar_parts :
  ∀ (x : ℝ), x > 0 → ¬∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = x ∧
    (a ≤ Real.sqrt 2 * b ∧ b ≤ Real.sqrt 2 * a) ∧
    (b ≤ Real.sqrt 2 * c ∧ c ≤ Real.sqrt 2 * b) ∧
    (a ≤ Real.sqrt 2 * c ∧ c ≤ Real.sqrt 2 * a) :=
by
  sorry


end impossibility_of_three_similar_parts_l284_28467


namespace pie_selection_theorem_l284_28436

/-- Represents the types of pie packets -/
inductive PiePacket
  | CabbageCabbage
  | CherryCherry
  | CabbageCherry

/-- Represents the possible fillings of a pie -/
inductive PieFilling
  | Cabbage
  | Cherry

/-- Represents the state of a pie -/
inductive PieState
  | Whole
  | Broken

/-- Represents a strategy for selecting a pie -/
def Strategy := PiePacket → PieFilling → PieState

/-- The probability of giving a whole cherry pie given a strategy -/
def probability_whole_cherry (s : Strategy) : ℚ := sorry

/-- The simple strategy described in part (a) -/
def simple_strategy : Strategy := sorry

/-- The improved strategy described in part (b) -/
def improved_strategy : Strategy := sorry

theorem pie_selection_theorem :
  (probability_whole_cherry simple_strategy = 2/3) ∧
  (probability_whole_cherry improved_strategy > 2/3) := by
  sorry


end pie_selection_theorem_l284_28436


namespace probability_of_winning_more_than_4000_l284_28410

/-- Represents the number of boxes and keys -/
def num_boxes : ℕ := 3

/-- Represents the total number of ways to assign keys to boxes -/
def total_assignments : ℕ := Nat.factorial num_boxes

/-- Represents the number of ways to correctly assign keys to both the second and third boxes -/
def correct_assignments : ℕ := 1

/-- Theorem stating the probability of correctly assigning keys to both the second and third boxes -/
theorem probability_of_winning_more_than_4000 :
  (correct_assignments : ℚ) / total_assignments = 1 / 6 := by
  sorry

end probability_of_winning_more_than_4000_l284_28410


namespace curve_equation_l284_28483

/-- Given a curve parameterized by (x,y) = (3t + 5, 6t - 8) where t is a real number,
    prove that the equation of the line is y = 2x - 18 -/
theorem curve_equation (t : ℝ) (x y : ℝ) 
    (h1 : x = 3 * t + 5) 
    (h2 : y = 6 * t - 8) : 
  y = 2 * x - 18 := by
  sorry

end curve_equation_l284_28483


namespace trout_weight_l284_28454

theorem trout_weight (num_trout num_catfish num_bluegill : ℕ) 
                     (weight_catfish weight_bluegill total_weight : ℚ) :
  num_trout = 4 →
  num_catfish = 3 →
  num_bluegill = 5 →
  weight_catfish = 3/2 →
  weight_bluegill = 5/2 →
  total_weight = 25 →
  ∃ weight_trout : ℚ,
    weight_trout * num_trout + weight_catfish * num_catfish + weight_bluegill * num_bluegill = total_weight ∧
    weight_trout = 2 :=
by sorry

end trout_weight_l284_28454


namespace max_consecutive_integers_sum_largest_n_not_exceeding_500_l284_28415

theorem max_consecutive_integers_sum (n : ℕ) : n ≤ 31 ↔ n * (n + 1) ≤ 1000 := by sorry

theorem largest_n_not_exceeding_500 : ∃ n : ℕ, n * (n + 1) ≤ 1000 ∧ ∀ m : ℕ, m > n → m * (m + 1) > 1000 := by sorry

end max_consecutive_integers_sum_largest_n_not_exceeding_500_l284_28415


namespace modulus_of_z_l284_28423

theorem modulus_of_z (z : ℂ) (h : z * (1 - Complex.I) = -1 - Complex.I) : 
  Complex.abs z = 1 := by
  sorry

end modulus_of_z_l284_28423
