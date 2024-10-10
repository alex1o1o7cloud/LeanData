import Mathlib

namespace turnip_bag_weights_l1964_196458

def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]

def total_weight : ℕ := bag_weights.sum

theorem turnip_bag_weights (turnip_weight : ℕ) 
  (h_turnip : turnip_weight ∈ bag_weights) :
  (∃ (onion_weights carrot_weights : List ℕ),
    onion_weights ++ carrot_weights ++ [turnip_weight] = bag_weights ∧
    onion_weights.sum * 2 = carrot_weights.sum ∧
    onion_weights.sum + carrot_weights.sum + turnip_weight = total_weight) ↔
  (turnip_weight = 13 ∨ turnip_weight = 16) :=
sorry

end turnip_bag_weights_l1964_196458


namespace factorization_cubic_l1964_196478

theorem factorization_cubic (a : ℝ) : a^3 - 6*a^2 + 9*a = a*(a-3)^2 := by
  sorry

end factorization_cubic_l1964_196478


namespace timmy_candies_l1964_196417

theorem timmy_candies : ∃ x : ℕ, 
  (x / 2 - 3) / 2 - 5 = 10 ∧ x = 66 := by
  sorry

end timmy_candies_l1964_196417


namespace no_solution_exists_l1964_196457

theorem no_solution_exists : ¬∃ (a b c x : ℝ),
  (2 : ℝ)^(x * 0.15) = 5^(a * Real.sin c) ∧
  ((2 : ℝ)^(x * 0.15))^b = 32 := by
  sorry

end no_solution_exists_l1964_196457


namespace cake_eaters_l1964_196450

theorem cake_eaters (n : ℕ) (h1 : n > 0) : 
  (∃ (portions : Fin n → ℚ), 
    (∀ i, portions i > 0) ∧ 
    (∃ i, portions i = 1/11) ∧ 
    (∃ i, portions i = 1/14) ∧ 
    (∀ i, portions i ≤ 1/11) ∧ 
    (∀ i, portions i ≥ 1/14) ∧ 
    (Finset.sum Finset.univ portions = 1)) ↔ 
  (n = 12 ∨ n = 13) :=
sorry

end cake_eaters_l1964_196450


namespace expression_evaluation_l1964_196461

theorem expression_evaluation (a b c : ℝ) 
  (ha : a = 12) (hb : b = 14) (hc : c = 19) : 
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) / 
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = 45 := by
  sorry

end expression_evaluation_l1964_196461


namespace charles_stroll_distance_l1964_196471

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: Charles strolled 6 miles -/
theorem charles_stroll_distance :
  let speed : ℝ := 3
  let time : ℝ := 2
  distance speed time = 6 := by sorry

end charles_stroll_distance_l1964_196471


namespace michaels_earnings_l1964_196455

/-- Calculates earnings based on hours worked and pay rates -/
def calculate_earnings (regular_hours : ℝ) (overtime_hours : ℝ) (regular_rate : ℝ) : ℝ :=
  regular_hours * regular_rate + overtime_hours * (2 * regular_rate)

theorem michaels_earnings :
  let total_hours : ℝ := 42.857142857142854
  let regular_hours : ℝ := 40
  let overtime_hours : ℝ := total_hours - regular_hours
  let regular_rate : ℝ := 7
  calculate_earnings regular_hours overtime_hours regular_rate = 320 := by
sorry

end michaels_earnings_l1964_196455


namespace arithmetic_mean_problem_l1964_196420

theorem arithmetic_mean_problem (original_list : List ℝ) (x y z : ℝ) :
  (original_list.length = 12) →
  (original_list.sum / original_list.length = 40) →
  ((original_list.sum + x + y + z) / (original_list.length + 3) = 50) →
  (x + y = 100) →
  z = 170 := by
sorry

end arithmetic_mean_problem_l1964_196420


namespace remainder_theorem_l1964_196499

theorem remainder_theorem (P D E Q R M S C : ℕ) 
  (h1 : P = Q * D + R)
  (h2 : Q = E * M + S)
  (h3 : R < D)
  (h4 : S < E) :
  ∃ K, P = K * (D * E) + (S * D + R + C) ∧ S * D + R + C < D * E :=
sorry

end remainder_theorem_l1964_196499


namespace quadratic_inequality_condition_l1964_196477

theorem quadratic_inequality_condition (a b c : ℝ) :
  (∀ x, a * x^2 + b * x + c < 0) ↔ (b^2 - 4*a*c < 0) = False := by
sorry

end quadratic_inequality_condition_l1964_196477


namespace product_of_numbers_with_given_sum_and_lcm_l1964_196413

theorem product_of_numbers_with_given_sum_and_lcm :
  ∃ (a b : ℕ+), 
    (a + b : ℕ) = 210 ∧ 
    Nat.lcm a b = 1547 → 
    (a * b : ℕ) = 10829 := by
  sorry

end product_of_numbers_with_given_sum_and_lcm_l1964_196413


namespace theodore_stone_statues_l1964_196429

/-- The number of stone statues Theodore crafts every month -/
def stone_statues : ℕ := sorry

/-- The number of wooden statues Theodore crafts every month -/
def wooden_statues : ℕ := 20

/-- The cost of a stone statue in dollars -/
def stone_cost : ℕ := 20

/-- The cost of a wooden statue in dollars -/
def wooden_cost : ℕ := 5

/-- The tax rate as a decimal -/
def tax_rate : ℚ := 1/10

/-- Theodore's total monthly earnings after tax in dollars -/
def total_earnings : ℕ := 270

theorem theodore_stone_statues :
  stone_statues = 10 ∧
  (stone_statues * stone_cost + wooden_statues * wooden_cost) * (1 - tax_rate) = total_earnings :=
sorry

end theodore_stone_statues_l1964_196429


namespace system_solution_l1964_196418

/-- Prove that the given system of linear equations has the specified solution -/
theorem system_solution (x y : ℝ) : 
  (x = 2 ∧ y = -3) → (3 * x + y = 3 ∧ 4 * x - y = 11) :=
by sorry

end system_solution_l1964_196418


namespace sandra_betty_orange_ratio_l1964_196416

theorem sandra_betty_orange_ratio :
  ∀ (emily sandra betty : ℕ),
    emily = 7 * sandra →
    betty = 12 →
    emily = 252 →
    sandra / betty = 3 :=
by
  sorry

end sandra_betty_orange_ratio_l1964_196416


namespace f_odd_and_decreasing_l1964_196430

-- Define the function f(x) = -x^3
def f (x : ℝ) : ℝ := -x^3

-- State the theorem
theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f y < f x) :=
sorry

end f_odd_and_decreasing_l1964_196430


namespace pencil_price_theorem_l1964_196437

/-- The price of a pencil in won -/
def pencil_price : ℚ := 5000 + 20

/-- The conversion factor from won to 10,000 won units -/
def conversion_factor : ℚ := 10000

/-- The price of the pencil in units of 10,000 won -/
def pencil_price_in_units : ℚ := pencil_price / conversion_factor

theorem pencil_price_theorem : pencil_price_in_units = 0.5 := by
  sorry

end pencil_price_theorem_l1964_196437


namespace cone_lateral_surface_angle_l1964_196441

/-- Given a cone with an acute triangular cross-section and slant height 4,
    where the maximum area of cross-sections passing through the vertex is 4√3,
    prove that the central angle of the sector in the lateral surface development is π. -/
theorem cone_lateral_surface_angle (h : ℝ) (θ : ℝ) (r : ℝ) (α : ℝ) : 
  h = 4 → 
  θ < π / 2 →
  (1 / 2) * h * h * Real.sin θ = 4 * Real.sqrt 3 →
  r = 2 →
  α = 2 * π * r / h →
  α = π :=
by sorry

end cone_lateral_surface_angle_l1964_196441


namespace fraction_simplification_l1964_196434

theorem fraction_simplification : (10^9 : ℕ) / (2 * 10^5) = 5000 := by
  sorry

end fraction_simplification_l1964_196434


namespace quadratic_roots_greater_than_half_l1964_196481

theorem quadratic_roots_greater_than_half (a : ℝ) :
  (∀ x : ℝ, (2 - a) * x^2 - 3 * a * x + 2 * a = 0 → x > (1/2 : ℝ)) ↔ 16/17 < a ∧ a < 2 := by
  sorry

end quadratic_roots_greater_than_half_l1964_196481


namespace ellipse_line_intersection_properties_l1964_196440

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 3 + y^2 / 2 = 1

-- Define the line
def line (x y k : ℝ) : Prop := y = k * x + 1

-- Define the foci
def F1 : ℝ × ℝ := (-1, 0)
def F2 : ℝ × ℝ := (1, 0)

-- Define the intersection points
variable (A B : ℝ × ℝ)

-- Define the parallel condition
def parallel (F1 A F2 B : ℝ × ℝ) : Prop :=
  (A.1 - F1.1) * (B.2 - F2.2) = (A.2 - F1.2) * (B.1 - F2.1)

-- Define the perpendicular condition
def perpendicular (A F1 F2 : ℝ × ℝ) : Prop :=
  (A.1 - F1.1) * (A.1 - F2.1) + (A.2 - F1.2) * (A.2 - F2.2) = 0

-- Theorem statement
theorem ellipse_line_intersection_properties
  (k : ℝ)
  (hA : ellipse A.1 A.2 ∧ line A.1 A.2 k)
  (hB : ellipse B.1 B.2 ∧ line B.1 B.2 k) :
  ¬(parallel F1 A F2 B) ∧ ¬(perpendicular A F1 F2) := by sorry

end ellipse_line_intersection_properties_l1964_196440


namespace f_increasing_iff_a_range_l1964_196483

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 - a*x - 5 else a/x

theorem f_increasing_iff_a_range (a : ℝ) :
  (∀ x y, x < y → f a x < f a y) ↔ -3 ≤ a ∧ a ≤ -2 :=
by sorry

end f_increasing_iff_a_range_l1964_196483


namespace good_student_count_l1964_196435

/-- Represents a student in the class -/
inductive Student
| Good
| Troublemaker

/-- The total number of students in the class -/
def totalStudents : Nat := 25

/-- The number of students who made the first statement -/
def firstStatementCount : Nat := 5

/-- The number of students who made the second statement -/
def secondStatementCount : Nat := 20

/-- Checks if the first statement is true for a given number of good students -/
def firstStatementTrue (goodCount : Nat) : Prop :=
  totalStudents - goodCount > (totalStudents - 1) / 2

/-- Checks if the second statement is true for a given number of good students -/
def secondStatementTrue (goodCount : Nat) : Prop :=
  totalStudents - goodCount = 3 * (goodCount - 1)

/-- Theorem stating that the number of good students is either 5 or 7 -/
theorem good_student_count :
  ∃ (goodCount : Nat), (goodCount = 5 ∨ goodCount = 7) ∧
    (firstStatementTrue goodCount ∨ ¬firstStatementTrue goodCount) ∧
    (secondStatementTrue goodCount ∨ ¬secondStatementTrue goodCount) ∧
    goodCount ≤ totalStudents :=
  sorry

end good_student_count_l1964_196435


namespace parallelogram_area_l1964_196462

/-- The area of a parallelogram with a diagonal of length 30 meters and a perpendicular height to that diagonal of 20 meters is 600 square meters. -/
theorem parallelogram_area (d h : ℝ) (hd : d = 30) (hh : h = 20) :
  d * h = 600 := by
  sorry

end parallelogram_area_l1964_196462


namespace heather_aprons_l1964_196408

/-- The number of aprons Heather sewed before today -/
def aprons_before_today : ℕ := by sorry

/-- The total number of aprons to be sewn -/
def total_aprons : ℕ := 150

/-- The number of aprons Heather sewed today -/
def aprons_today : ℕ := 3 * aprons_before_today

/-- The number of aprons Heather will sew tomorrow -/
def aprons_tomorrow : ℕ := 49

/-- The number of remaining aprons after sewing tomorrow -/
def remaining_aprons : ℕ := aprons_tomorrow

theorem heather_aprons : 
  aprons_before_today = 13 ∧
  aprons_before_today + aprons_today + aprons_tomorrow + remaining_aprons = total_aprons := by
  sorry

end heather_aprons_l1964_196408


namespace f_derivative_at_zero_l1964_196475

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x + 1) - 3 * x

theorem f_derivative_at_zero : 
  deriv f 0 = 2 * Real.exp 1 - 3 := by sorry

end f_derivative_at_zero_l1964_196475


namespace solution_ratio_proof_l1964_196472

/-- Proves that the ratio of solutions A and B is 1:1 when mixed to form a 45% alcohol solution --/
theorem solution_ratio_proof (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) : 
  (4/9 * a + 5/11 * b) / (a + b) = 9/20 → a = b :=
by
  sorry

end solution_ratio_proof_l1964_196472


namespace intersection_theorem_l1964_196428

/-- A line passing through two points -/
structure Line1 where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ

/-- A line described by y = mx + b -/
structure Line2 where
  m : ℝ
  b : ℝ

/-- The intersection point of two lines -/
def intersection_point (l1 : Line1) (l2 : Line2) : ℝ × ℝ :=
  sorry

theorem intersection_theorem :
  let l1 : Line1 := { x1 := 0, y1 := 3, x2 := 4, y2 := 11 }
  let l2 : Line2 := { m := -1, b := 15 }
  intersection_point l1 l2 = (4, 11) := by
  sorry

end intersection_theorem_l1964_196428


namespace village_population_l1964_196494

theorem village_population (population_percentage : ℝ) (partial_population : ℕ) (total_population : ℕ) :
  population_percentage = 80 →
  partial_population = 64000 →
  (population_percentage / 100) * total_population = partial_population →
  total_population = 80000 := by
  sorry

end village_population_l1964_196494


namespace inequality_proof_l1964_196432

theorem inequality_proof (a b c d : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : 0 ≤ d)
  (h5 : a^5 + b^5 ≤ 1) (h6 : c^5 + d^5 ≤ 1) : 
  a^2 * c^3 + b^2 * d^3 ≤ 1 := by
sorry

end inequality_proof_l1964_196432


namespace quadratic_nonnegative_conditions_l1964_196489

theorem quadratic_nonnegative_conditions (a b c : ℝ) (ha : a ≠ 0)
  (hf : ∀ x : ℝ, a * x^2 + 2 * b * x + c ≥ 0) :
  a > 0 ∧ c ≥ 0 ∧ a * c - b^2 ≥ 0 := by sorry

end quadratic_nonnegative_conditions_l1964_196489


namespace probability_chords_intersect_2000_probability_chords_intersect_general_l1964_196424

/-- Given a circle with evenly spaced points, this function calculates the probability
    that chord AB intersects chord CD when five distinct points are randomly selected. -/
def probability_chords_intersect (n : ℕ) : ℚ :=
  if n < 5 then 0
  else 1 / 15

/-- Theorem stating that the probability of chord AB intersecting chord CD
    when five distinct points are randomly selected from 2000 evenly spaced
    points on a circle is 1/15. -/
theorem probability_chords_intersect_2000 :
  probability_chords_intersect 2000 = 1 / 15 := by
  sorry

/-- Theorem stating that the probability of chord AB intersecting chord CD
    is 1/15 for any number of evenly spaced points on a circle, as long as
    there are at least 5 points. -/
theorem probability_chords_intersect_general (n : ℕ) (h : n ≥ 5) :
  probability_chords_intersect n = 1 / 15 := by
  sorry

end probability_chords_intersect_2000_probability_chords_intersect_general_l1964_196424


namespace compare_powers_l1964_196427

theorem compare_powers : 
  let a : ℝ := 2^(4/3)
  let b : ℝ := 4^(2/5)
  let c : ℝ := 25^(1/3)
  b < a ∧ a < c := by sorry

end compare_powers_l1964_196427


namespace polynomial_real_root_l1964_196436

theorem polynomial_real_root (a : ℝ) :
  (∃ x : ℝ, x^4 - a*x^3 - x^2 - a*x + 1 = 0) ↔ a ≥ -1/2 := by
  sorry

end polynomial_real_root_l1964_196436


namespace smallest_common_multiple_of_6_and_15_l1964_196460

theorem smallest_common_multiple_of_6_and_15 (b : ℕ) : 
  (b % 6 = 0 ∧ b % 15 = 0) → b ≥ 30 :=
by sorry

end smallest_common_multiple_of_6_and_15_l1964_196460


namespace inscribed_square_area_l1964_196482

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 - 10*x + 21

/-- A square inscribed in the region bound by the parabola and the x-axis -/
structure InscribedSquare where
  center : ℝ
  side_half : ℝ
  lower_left_on_axis : f (center - side_half) = 0
  upper_right_on_parabola : f (center + side_half) = 2 * side_half

/-- The theorem stating the area of the inscribed square -/
theorem inscribed_square_area (s : InscribedSquare) : 
  (2 * s.side_half)^2 = 24 - 16 * Real.sqrt 5 := by
  sorry

end inscribed_square_area_l1964_196482


namespace S_equals_formula_S_2k_minus_1_is_polynomial_l1964_196468

-- Define S as a function of n and k
def S (n k : ℕ) : ℚ := sorry

-- Define S_{2k-1}(n) as a function
def S_2k_minus_1 (n k : ℕ) : ℚ := sorry

-- Theorem 1: S equals (n^k * (n+1)^k) / 2
theorem S_equals_formula (n k : ℕ) : 
  S n k = (n^k * (n+1)^k : ℚ) / 2 := by sorry

-- Theorem 2: S_{2k-1}(n) is a polynomial of degree k in (n(n+1))/2
theorem S_2k_minus_1_is_polynomial (n k : ℕ) :
  ∃ (p : Polynomial ℚ), 
    (S_2k_minus_1 n k = p.eval ((n * (n+1) : ℕ) / 2 : ℚ)) ∧ 
    (p.degree = k) := by sorry

end S_equals_formula_S_2k_minus_1_is_polynomial_l1964_196468


namespace fraction_sum_problem_l1964_196487

theorem fraction_sum_problem (x y : ℚ) (h : x / y = 2 / 7) : (x + y) / y = 9 / 7 := by
  sorry

end fraction_sum_problem_l1964_196487


namespace mikas_height_l1964_196470

/-- Proves that Mika's current height is 66 inches given the problem conditions --/
theorem mikas_height (initial_height : ℝ) : 
  initial_height > 0 →
  initial_height * 1.25 = 75 →
  initial_height * 1.1 = 66 :=
by
  sorry

#check mikas_height

end mikas_height_l1964_196470


namespace number_is_forty_l1964_196456

theorem number_is_forty (N : ℝ) (P : ℝ) : 
  (P / 100) * N = 0.25 * 16 + 2 → N = 40 := by
  sorry

end number_is_forty_l1964_196456


namespace trigonometric_identity_l1964_196423

theorem trigonometric_identity : 
  Real.sin (315 * π / 180) - Real.cos (135 * π / 180) + 2 * Real.sin (570 * π / 180) = Real.sqrt 2 + 1 := by
  sorry

end trigonometric_identity_l1964_196423


namespace maple_logs_solution_l1964_196443

/-- The number of logs each maple tree makes -/
def maple_logs : ℕ := 60

theorem maple_logs_solution : 
  ∃ (x : ℕ), x > 0 ∧ 8 * 80 + 3 * x + 4 * 100 = 1220 → x = maple_logs :=
by sorry

end maple_logs_solution_l1964_196443


namespace clothing_purchase_problem_l1964_196400

/-- The problem of determining the number of clothing pieces bought --/
theorem clothing_purchase_problem (total_spent : ℕ) (price1 price2 other_price : ℕ) :
  total_spent = 610 →
  price1 = 49 →
  price2 = 81 →
  other_price = 96 →
  ∃ (n : ℕ), total_spent = price1 + price2 + n * other_price ∧ n + 2 = 7 :=
by sorry

end clothing_purchase_problem_l1964_196400


namespace ellipse_condition_l1964_196442

def is_ellipse_equation (m : ℝ) : Prop :=
  m > 2 ∧ m < 5 ∧ m ≠ 7/2

theorem ellipse_condition (m : ℝ) :
  (2 < m ∧ m < 5) → (is_ellipse_equation m) ∧
  ∃ m', is_ellipse_equation m' ∧ ¬(2 < m' ∧ m' < 5) :=
by sorry

end ellipse_condition_l1964_196442


namespace average_difference_implies_unknown_l1964_196479

theorem average_difference_implies_unknown (x : ℝ) : 
  let set1 := [20, 40, 60]
  let set2 := [10, x, 15]
  (set1.sum / set1.length) = (set2.sum / set2.length) + 5 →
  x = 80 := by
sorry

end average_difference_implies_unknown_l1964_196479


namespace hyperbola_sequence_fixed_point_l1964_196495

/-- Definition of the hyperbola -/
def hyperbola (x y : ℝ) : Prop := y^2 - x^2 = 2

/-- Definition of the line with slope 2 passing through a point -/
def line_slope_2 (x₀ y₀ x y : ℝ) : Prop := y - y₀ = 2 * (x - x₀)

/-- Definition of the next point in the sequence -/
def next_point (x₀ x₁ : ℝ) : Prop :=
  ∃ y₁, hyperbola x₁ y₁ ∧ line_slope_2 x₀ 0 x₁ y₁

/-- Definition of the sequence of points -/
def point_sequence (x : ℕ → ℝ) : Prop :=
  ∀ n, next_point (x n) (x (n+1)) ∨ x n = 0

/-- The main theorem -/
theorem hyperbola_sequence_fixed_point :
  ∃! k : ℕ, k = (2^2048 - 2) ∧
  ∃ x : ℕ → ℝ, point_sequence x ∧ x 0 = x 2048 ∧ x 0 ≠ 0 ∧
  ∀ y : ℕ → ℝ, point_sequence y ∧ y 0 = y 2048 ∧ y 0 ≠ 0 →
    ∃! i : ℕ, i < k ∧ x 0 = y 0 :=
sorry

end hyperbola_sequence_fixed_point_l1964_196495


namespace line_through_intersection_parallel_to_l₃_l1964_196453

-- Define the lines l₁, l₂, and l₃
def l₁ (x y : ℝ) : Prop := 3 * x + 4 * y - 2 = 0
def l₂ (x y : ℝ) : Prop := 2 * x + y + 2 = 0
def l₃ (x y : ℝ) : Prop := 4 * x + 3 * y - 2 = 0

-- Define the intersection point of l₁ and l₂
def intersection_point (x y : ℝ) : Prop := l₁ x y ∧ l₂ x y

-- Define parallel lines
def parallel (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), ∀ (x y : ℝ), f x y ↔ g (x + k) y

-- Theorem statement
theorem line_through_intersection_parallel_to_l₃ :
  ∃ (a b c : ℝ), 
    (∀ (x y : ℝ), intersection_point x y → a * x + b * y + c = 0) ∧
    parallel (fun x y => a * x + b * y + c = 0) l₃ ∧
    (a = 4 ∧ b = 3 ∧ c = 2) :=
sorry

end line_through_intersection_parallel_to_l₃_l1964_196453


namespace root_in_interval_l1964_196488

theorem root_in_interval : ∃ x : ℝ, x ∈ Set.Ioo (-4 : ℝ) (-3 : ℝ) ∧ x^3 + 3*x^2 - x + 1 = 0 := by
  sorry

end root_in_interval_l1964_196488


namespace students_playing_both_sports_l1964_196402

theorem students_playing_both_sports (total : ℕ) (football : ℕ) (tennis : ℕ) (neither : ℕ) :
  total = 39 →
  football = 26 →
  tennis = 20 →
  neither = 10 →
  ∃ (both : ℕ), both = 17 ∧
    total = football + tennis - both + neither :=
by sorry

end students_playing_both_sports_l1964_196402


namespace hawk_percentage_is_25_percent_l1964_196411

/-- Represents the percentage of hawks in the bird population -/
def hawk_percentage : ℝ := sorry

/-- Represents the percentage of paddyfield-warblers in the bird population -/
def paddyfield_warbler_percentage : ℝ := sorry

/-- Represents the percentage of kingfishers in the bird population -/
def kingfisher_percentage : ℝ := sorry

/-- The percentage of non-hawks that are paddyfield-warblers -/
def paddyfield_warbler_ratio : ℝ := 0.4

/-- The ratio of kingfishers to paddyfield-warblers -/
def kingfisher_to_warbler_ratio : ℝ := 0.25

/-- The percentage of birds that are not hawks, paddyfield-warblers, or kingfishers -/
def other_birds_percentage : ℝ := 0.35

theorem hawk_percentage_is_25_percent :
  hawk_percentage = 0.25 ∧
  paddyfield_warbler_percentage = paddyfield_warbler_ratio * (1 - hawk_percentage) ∧
  kingfisher_percentage = kingfisher_to_warbler_ratio * paddyfield_warbler_percentage ∧
  hawk_percentage + paddyfield_warbler_percentage + kingfisher_percentage + other_birds_percentage = 1 :=
by sorry

end hawk_percentage_is_25_percent_l1964_196411


namespace range_of_x_when_a_is_one_range_of_a_when_not_p_implies_not_q_l1964_196452

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := (x - 3) / (x - 2) ≤ 0

-- Part 1
theorem range_of_x_when_a_is_one :
  ∀ x : ℝ, (p x 1 ∧ q x) → (2 < x ∧ x < 3) :=
sorry

-- Part 2
theorem range_of_a_when_not_p_implies_not_q :
  ∀ a : ℝ, (∀ x : ℝ, ¬(p x a) → (x ≤ 2 ∨ x > 3)) ∧
           (∃ x : ℝ, (x ≤ 2 ∨ x > 3) ∧ p x a) →
  (1 < a ∧ a ≤ 2) :=
sorry

end range_of_x_when_a_is_one_range_of_a_when_not_p_implies_not_q_l1964_196452


namespace max_correct_answers_jesse_l1964_196425

/-- Represents a math contest with given parameters -/
structure MathContest where
  total_questions : ℕ
  correct_points : ℤ
  unanswered_points : ℤ
  incorrect_points : ℤ

/-- Represents a contestant's performance in the math contest -/
structure ContestPerformance where
  contest : MathContest
  total_score : ℤ

/-- Calculates the maximum number of correctly answered questions for a given contest performance -/
def max_correct_answers (performance : ContestPerformance) : ℕ :=
  sorry

/-- The specific contest Jesse participated in -/
def jesses_contest : MathContest := {
  total_questions := 60,
  correct_points := 4,
  unanswered_points := 0,
  incorrect_points := -1
}

/-- Jesse's performance in the contest -/
def jesses_performance : ContestPerformance := {
  contest := jesses_contest,
  total_score := 112
}

theorem max_correct_answers_jesse :
  max_correct_answers jesses_performance = 34 := by
  sorry

end max_correct_answers_jesse_l1964_196425


namespace cosine_ratio_equals_negative_sqrt_three_l1964_196404

theorem cosine_ratio_equals_negative_sqrt_three : 
  (2 * Real.cos (80 * π / 180) + Real.cos (160 * π / 180)) / Real.cos (70 * π / 180) = -Real.sqrt 3 := by
  sorry

end cosine_ratio_equals_negative_sqrt_three_l1964_196404


namespace total_candies_l1964_196406

theorem total_candies (red_candies blue_candies : ℕ) 
  (h1 : red_candies = 145) 
  (h2 : blue_candies = 3264) : 
  red_candies + blue_candies = 3409 := by
  sorry

end total_candies_l1964_196406


namespace triangle_with_unit_inradius_is_right_angled_l1964_196484

/-- A triangle with integer side lengths and inradius 1 is right-angled with sides (3, 4, 5) -/
theorem triangle_with_unit_inradius_is_right_angled (a b c : ℕ) (r : ℝ) :
  r = 1 →
  (a : ℝ) + (b : ℝ) + (c : ℝ) = 2 * ((a : ℝ) * (b : ℝ) * (c : ℝ)) / ((a : ℝ) + (b : ℝ) + (c : ℝ)) →
  (a = 3 ∧ b = 4 ∧ c = 5) ∨ (a = 3 ∧ b = 5 ∧ c = 4) ∨
  (a = 4 ∧ b = 3 ∧ c = 5) ∨ (a = 4 ∧ b = 5 ∧ c = 3) ∨
  (a = 5 ∧ b = 3 ∧ c = 4) ∨ (a = 5 ∧ b = 4 ∧ c = 3) :=
by sorry

end triangle_with_unit_inradius_is_right_angled_l1964_196484


namespace pythagorean_triples_l1964_196433

theorem pythagorean_triples (n m : ℕ) : 
  (n ≥ 3 ∧ Odd n) → 
  ((n^2 - 1) / 2)^2 + n^2 = ((n^2 + 1) / 2)^2 ∧
  (m > 1) →
  (m^2 - 1)^2 + (2*m)^2 = (m^2 + 1)^2 := by
sorry

end pythagorean_triples_l1964_196433


namespace division_problem_l1964_196407

theorem division_problem : (72 : ℚ) / ((6 : ℚ) / 3) = 36 := by
  sorry

end division_problem_l1964_196407


namespace min_value_expression_l1964_196439

theorem min_value_expression (a b c : ℕ) (h1 : b > a) (h2 : a > c) (h3 : c > 0) (h4 : b ≠ 0) :
  ((a + b)^2 + (b + c)^2 + (c - a)^2 + (a - c)^2 : ℚ) / (b^2 : ℚ) ≥ 9/2 := by
  sorry

end min_value_expression_l1964_196439


namespace stratified_sampling_most_appropriate_l1964_196426

/-- Represents different sampling methods -/
inductive SamplingMethod
  | Lottery
  | RandomNumberTable
  | Systematic
  | Stratified

/-- Represents income levels -/
inductive IncomeLevel
  | High
  | Middle
  | Low

/-- Structure representing a community's income distribution -/
structure CommunityIncome where
  totalFamilies : ℕ
  highIncome : ℕ
  middleIncome : ℕ
  lowIncome : ℕ
  high_income_valid : highIncome ≤ totalFamilies
  middle_income_valid : middleIncome ≤ totalFamilies
  low_income_valid : lowIncome ≤ totalFamilies
  total_sum_valid : highIncome + middleIncome + lowIncome = totalFamilies

/-- Function to determine the most appropriate sampling method -/
def mostAppropriateSamplingMethod (community : CommunityIncome) (sampleSize : ℕ) : SamplingMethod :=
  SamplingMethod.Stratified

/-- Theorem stating that stratified sampling is most appropriate for the given community -/
theorem stratified_sampling_most_appropriate 
  (community : CommunityIncome) 
  (sampleSize : ℕ) 
  (sample_size_valid : sampleSize ≤ community.totalFamilies) :
  mostAppropriateSamplingMethod community sampleSize = SamplingMethod.Stratified :=
by
  sorry

#check stratified_sampling_most_appropriate

end stratified_sampling_most_appropriate_l1964_196426


namespace sum_of_first_53_odd_numbers_l1964_196403

theorem sum_of_first_53_odd_numbers : 
  (Finset.range 53).sum (fun n => 2 * n + 1) = 2809 := by
  sorry

end sum_of_first_53_odd_numbers_l1964_196403


namespace probability_multiple_of_three_l1964_196466

theorem probability_multiple_of_three (n : ℕ) (h : n = 21) :
  (Finset.filter (fun x => x % 3 = 0) (Finset.range n.succ)).card / n = 1 / 3 := by
  sorry

end probability_multiple_of_three_l1964_196466


namespace greg_books_multiple_l1964_196497

/-- The number of books Megan has read -/
def megan_books : ℕ := 32

/-- The number of books Kelcie has read -/
def kelcie_books : ℕ := megan_books / 4

/-- The total number of books read by all three people -/
def total_books : ℕ := 65

/-- The multiple of Kelcie's books that Greg has read -/
def greg_multiple : ℕ := 2

theorem greg_books_multiple : 
  megan_books + kelcie_books + (greg_multiple * kelcie_books + 9) = total_books :=
sorry

end greg_books_multiple_l1964_196497


namespace onions_in_basket_l1964_196467

/-- Given a basket of onions with initial count S, prove that after
    Sara adds 4, Sally removes 5, and Fred adds F onions, 
    resulting in 8 more onions than the initial count,
    Fred must have added 9 onions. -/
theorem onions_in_basket (S : ℤ) : ∃ F : ℤ, 
  S - 1 + F = S + 8 ∧ F = 9 := by
  sorry

end onions_in_basket_l1964_196467


namespace tim_found_37_shells_l1964_196474

/-- The number of seashells Sally found -/
def sally_shells : ℕ := 13

/-- The total number of seashells Tim and Sally found together -/
def total_shells : ℕ := 50

/-- The number of seashells Tim found -/
def tim_shells : ℕ := total_shells - sally_shells

theorem tim_found_37_shells : tim_shells = 37 := by
  sorry

end tim_found_37_shells_l1964_196474


namespace seating_arrangements_l1964_196431

-- Define the number of people excluding the fixed person
def n : ℕ := 4

-- Define the function to calculate the total number of permutations
def total_permutations (n : ℕ) : ℕ := n.factorial

-- Define the function to calculate the number of permutations where two specific people are adjacent
def adjacent_permutations (n : ℕ) : ℕ := 2 * (n - 1).factorial

-- Theorem statement
theorem seating_arrangements :
  total_permutations n - adjacent_permutations n = 12 :=
by sorry

end seating_arrangements_l1964_196431


namespace sequence_product_l1964_196496

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem sequence_product (a b : ℕ → ℝ) :
  (∀ n, a n ≠ 0) →
  arithmetic_sequence a →
  geometric_sequence b →
  a 4 - 2 * (a 7)^2 + 3 * a 8 = 0 →
  b 7 = a 7 →
  b 2 * b 8 * b 11 = 8 := by
sorry

end sequence_product_l1964_196496


namespace lcm_1404_972_l1964_196451

theorem lcm_1404_972 : Nat.lcm 1404 972 = 88452 := by
  sorry

end lcm_1404_972_l1964_196451


namespace halloween_decorations_l1964_196491

/-- Calculates the number of plastic skulls in Danai's Halloween decorations. -/
theorem halloween_decorations (total_decorations : ℕ) (broomsticks : ℕ) (spiderwebs : ℕ) 
  (cauldron : ℕ) (budget_left : ℕ) (left_to_put_up : ℕ) 
  (h1 : total_decorations = 83)
  (h2 : broomsticks = 4)
  (h3 : spiderwebs = 12)
  (h4 : cauldron = 1)
  (h5 : budget_left = 20)
  (h6 : left_to_put_up = 10) :
  total_decorations - (broomsticks + spiderwebs + 2 * spiderwebs + cauldron + budget_left + left_to_put_up) = 12 := by
  sorry

end halloween_decorations_l1964_196491


namespace right_triangle_hypotenuse_l1964_196409

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 6 → b = 8 → c^2 = a^2 + b^2 → c = 10 := by
  sorry

end right_triangle_hypotenuse_l1964_196409


namespace derivative_bounded_l1964_196459

open Real

/-- Given a function f: ℝ → ℝ with continuous second derivative, 
    and both f and f'' are bounded, prove that f' is also bounded. -/
theorem derivative_bounded (f : ℝ → ℝ) (hf'' : Continuous (deriv (deriv f))) 
  (hf_bdd : ∃ M, ∀ x, |f x| ≤ M) (hf''_bdd : ∃ M, ∀ x, |(deriv (deriv f)) x| ≤ M) :
  ∃ K, ∀ x, |deriv f x| ≤ K := by
  sorry

end derivative_bounded_l1964_196459


namespace republicans_count_l1964_196463

/-- Given the total number of representatives and the difference between Republicans and Democrats,
    calculate the number of Republicans. -/
def calculateRepublicans (total : ℕ) (difference : ℕ) : ℕ :=
  (total + difference) / 2

theorem republicans_count :
  calculateRepublicans 434 30 = 232 := by
  sorry

end republicans_count_l1964_196463


namespace trapezoid_shorter_base_l1964_196498

/-- A trapezoid with given properties -/
structure Trapezoid where
  longer_base : ℝ
  shorter_base : ℝ
  midpoint_segment : ℝ

/-- The property that the midpoint segment length is half the difference of base lengths -/
def midpoint_property (t : Trapezoid) : Prop :=
  t.midpoint_segment = (t.longer_base - t.shorter_base) / 2

theorem trapezoid_shorter_base (t : Trapezoid) 
  (h1 : t.longer_base = 103)
  (h2 : t.midpoint_segment = 5)
  (h3 : midpoint_property t) :
  t.shorter_base = 93 := by
  sorry

end trapezoid_shorter_base_l1964_196498


namespace power_multiplication_result_l1964_196454

theorem power_multiplication_result : 0.25^2023 * 4^2024 = 4 := by
  sorry

end power_multiplication_result_l1964_196454


namespace remainder_property_l1964_196445

/-- A polynomial of the form Dx^6 + Ex^4 + Fx^2 + 7 -/
def q (D E F : ℝ) (x : ℝ) : ℝ := D * x^6 + E * x^4 + F * x^2 + 7

/-- The remainder theorem -/
def remainder_theorem (p : ℝ → ℝ) (a : ℝ) : ℝ := p a

theorem remainder_property (D E F : ℝ) :
  remainder_theorem (q D E F) 2 = 17 →
  remainder_theorem (q D E F) (-2) = 17 := by
  sorry

end remainder_property_l1964_196445


namespace similar_cube_volume_l1964_196473

theorem similar_cube_volume (original_volume : ℝ) (scale_factor : ℝ) : 
  original_volume = 343 → scale_factor = 2 → 
  (scale_factor ^ 3) * original_volume = 2744 := by
  sorry

end similar_cube_volume_l1964_196473


namespace size_ratio_proof_l1964_196415

def anna_size : ℕ := 2

def becky_size (anna_size : ℕ) : ℕ := 3 * anna_size

def ginger_size : ℕ := 8

theorem size_ratio_proof (anna_size : ℕ) (becky_size : ℕ → ℕ) (ginger_size : ℕ)
  (h1 : anna_size = 2)
  (h2 : becky_size anna_size = 3 * anna_size)
  (h3 : ginger_size = 8)
  (h4 : ∃ k : ℕ, ginger_size = k * (becky_size anna_size - 4)) :
  ginger_size / (becky_size anna_size) = 4 / 3 := by
sorry

end size_ratio_proof_l1964_196415


namespace intersection_of_A_and_B_l1964_196421

-- Define set A
def A : Set ℝ := {y | ∃ x, y = x^2 + 2*x - 3}

-- Define set B
def B : Set ℝ := {y | ∃ x < 0, y = x + 1/x}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = Set.Icc (-4) (-2) := by sorry

end intersection_of_A_and_B_l1964_196421


namespace sufficient_not_necessary_condition_l1964_196476

theorem sufficient_not_necessary_condition :
  (∃ a : ℝ, a > 1 ∧ 1 / a < 1) ∧
  (∃ a : ℝ, ¬(a > 1) ∧ 1 / a < 1) :=
by sorry

end sufficient_not_necessary_condition_l1964_196476


namespace boat_current_rate_l1964_196401

/-- Proves that given a boat with a speed of 22 km/hr in still water,
    traveling 10.4 km downstream in 24 minutes, the rate of the current is 4 km/hr. -/
theorem boat_current_rate
  (boat_speed : ℝ)
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (h1 : boat_speed = 22)
  (h2 : downstream_distance = 10.4)
  (h3 : downstream_time = 24 / 60) :
  ∃ current_rate : ℝ,
    current_rate = 4 ∧
    downstream_distance = (boat_speed + current_rate) * downstream_time :=
by sorry

end boat_current_rate_l1964_196401


namespace cistern_wet_surface_area_l1964_196485

/-- Calculates the total wet surface area of a rectangular cistern -/
def cisternWetSurfaceArea (length width depth : ℝ) : ℝ :=
  length * width + 2 * (length * depth + width * depth)

/-- Theorem: The wet surface area of a cistern with given dimensions is 134 square meters -/
theorem cistern_wet_surface_area :
  cisternWetSurfaceArea 10 8 1.5 = 134 := by
  sorry

end cistern_wet_surface_area_l1964_196485


namespace statement_A_incorrect_l1964_196446

/-- Represents the process of meiosis and fertilization -/
structure MeiosisFertilization where
  sperm_transformation : Bool
  egg_metabolism_increase : Bool
  homologous_chromosomes_appearance : Bool
  fertilization_randomness : Bool

/-- Represents the correctness of statements about meiosis and fertilization -/
structure Statements where
  A : Bool
  B : Bool
  C : Bool
  D : Bool

/-- The given information about meiosis and fertilization -/
def given_info : MeiosisFertilization :=
  { sperm_transformation := true
  , egg_metabolism_increase := true
  , homologous_chromosomes_appearance := true
  , fertilization_randomness := true }

/-- The correctness of statements based on the given information -/
def statement_correctness (info : MeiosisFertilization) : Statements :=
  { A := false  -- Statement A is incorrect
  , B := info.sperm_transformation && info.egg_metabolism_increase
  , C := info.homologous_chromosomes_appearance
  , D := info.fertilization_randomness }

/-- Theorem stating that statement A is incorrect -/
theorem statement_A_incorrect (info : MeiosisFertilization) :
  (statement_correctness info).A = false := by
  sorry

end statement_A_incorrect_l1964_196446


namespace quadratic_roots_and_specific_case_l1964_196444

/-- The quadratic equation x^2 - (m-1)x = 3 -/
def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  x^2 - (m-1)*x = 3

theorem quadratic_roots_and_specific_case :
  (∀ m : ℝ, ∃ x y : ℝ, x ≠ y ∧ quadratic_equation m x ∧ quadratic_equation m y) ∧
  (∃ m : ℝ, quadratic_equation m 2 ∧ quadratic_equation m (-3/2) ∧ m = 5/2) :=
sorry

end quadratic_roots_and_specific_case_l1964_196444


namespace empty_set_subset_of_all_l1964_196449

theorem empty_set_subset_of_all (A : Set α) : ∅ ⊆ A := by
  sorry

end empty_set_subset_of_all_l1964_196449


namespace dan_picked_more_l1964_196469

/-- The number of apples Benny picked -/
def benny_apples : ℕ := 2

/-- The number of apples Dan picked -/
def dan_apples : ℕ := 9

/-- Theorem: Dan picked 7 more apples than Benny -/
theorem dan_picked_more : dan_apples - benny_apples = 7 := by sorry

end dan_picked_more_l1964_196469


namespace inequality_proof_l1964_196464

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≥ 2) :
  |x^12 - y^12| + 2 * x^6 * y^6 ≥ 2 := by
  sorry

end inequality_proof_l1964_196464


namespace total_spent_calculation_l1964_196419

def lunch_cost : ℝ := 50.20
def tip_rate : ℝ := 0.20

theorem total_spent_calculation :
  lunch_cost * (1 + tip_rate) = 60.24 := by
  sorry

end total_spent_calculation_l1964_196419


namespace wheel_radius_l1964_196493

/-- The radius of a wheel given its circumference and number of revolutions --/
theorem wheel_radius (distance : ℝ) (revolutions : ℕ) (h : distance = 760.57 ∧ revolutions = 500) :
  ∃ (radius : ℝ), abs (radius - 0.242) < 0.001 := by
  sorry

end wheel_radius_l1964_196493


namespace soap_brand_usage_ratio_l1964_196412

/-- Given a survey of households and their soap brand usage, prove the ratio of households
    using only brand B to those using both brands A and B. -/
theorem soap_brand_usage_ratio
  (total_households : ℕ)
  (neither_brand : ℕ)
  (only_brand_A : ℕ)
  (both_brands : ℕ)
  (h1 : total_households = 180)
  (h2 : neither_brand = 80)
  (h3 : only_brand_A = 60)
  (h4 : both_brands = 10)
  (h5 : total_households = neither_brand + only_brand_A + (total_households - neither_brand - only_brand_A - both_brands) + both_brands) :
  (total_households - neither_brand - only_brand_A - both_brands) / both_brands = 3 := by
  sorry

end soap_brand_usage_ratio_l1964_196412


namespace gain_percent_calculation_l1964_196492

theorem gain_percent_calculation (cost_price selling_price : ℝ) 
  (h1 : cost_price = 675)
  (h2 : selling_price = 1080) : 
  (selling_price - cost_price) / cost_price * 100 = 60 := by
  sorry

end gain_percent_calculation_l1964_196492


namespace expected_value_max_two_dice_rolls_expected_value_max_two_dice_rolls_eq_l1964_196480

/-- The expected value of the maximum of two independent rolls of a fair six-sided die -/
theorem expected_value_max_two_dice_rolls : ℝ :=
  let X : Fin 6 → ℝ := λ i => (i : ℝ) + 1
  let P : Fin 6 → ℝ := λ i =>
    match i with
    | 0 => 1 / 36
    | 1 => 3 / 36
    | 2 => 5 / 36
    | 3 => 7 / 36
    | 4 => 9 / 36
    | 5 => 11 / 36
  161 / 36

/-- The expected value of the maximum of two independent rolls of a fair six-sided die is 161/36 -/
theorem expected_value_max_two_dice_rolls_eq : expected_value_max_two_dice_rolls = 161 / 36 := by
  sorry

end expected_value_max_two_dice_rolls_expected_value_max_two_dice_rolls_eq_l1964_196480


namespace sqrt_equation_solution_l1964_196438

theorem sqrt_equation_solution :
  ∃! z : ℝ, Real.sqrt (10 + 3 * z) = 15 :=
by
  -- Proof goes here
  sorry

end sqrt_equation_solution_l1964_196438


namespace water_added_fourth_hour_l1964_196465

-- Define the water tank scenario
def water_tank_scenario (initial_water : ℝ) (loss_rate : ℝ) (added_third_hour : ℝ) (added_fourth_hour : ℝ) : ℝ :=
  initial_water - 4 * loss_rate + added_third_hour + added_fourth_hour

-- Theorem statement
theorem water_added_fourth_hour :
  ∃ (added_fourth_hour : ℝ),
    water_tank_scenario 40 2 1 added_fourth_hour = 36 ∧
    added_fourth_hour = 3 :=
by
  sorry


end water_added_fourth_hour_l1964_196465


namespace max_value_of_f_on_I_l1964_196490

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - 2*x - 1

-- Define the interval
def I : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}

-- Theorem statement
theorem max_value_of_f_on_I :
  ∃ (M : ℝ), M = 2 ∧ ∀ x ∈ I, f x ≤ M :=
sorry

end max_value_of_f_on_I_l1964_196490


namespace three_digit_numbers_count_l1964_196447

/-- Represents a card with two distinct numbers -/
structure Card where
  front : Nat
  back : Nat
  distinct : front ≠ back

/-- The set of cards given in the problem -/
def cards : Finset Card := sorry

/-- The number of cards -/
def num_cards : Nat := Finset.card cards

/-- The number of cards used to form a number -/
def cards_used : Nat := 3

/-- Calculates the number of different three-digit numbers that can be formed -/
def num_three_digit_numbers : Nat :=
  (num_cards.choose cards_used) * (2^cards_used) * (cards_used.factorial)

theorem three_digit_numbers_count :
  num_three_digit_numbers = 192 := by sorry

end three_digit_numbers_count_l1964_196447


namespace shaded_area_theorem_l1964_196486

/-- The area of a circle with radius r -/
def circle_area (r : ℝ) : ℝ := 3.14 * r ^ 2

/-- The theorem stating the area of the shaded region -/
theorem shaded_area_theorem :
  let large_radius : ℝ := 20
  let small_radius : ℝ := 10
  let num_small_circles : ℕ := 7
  let large_circle_area := circle_area large_radius
  let small_circle_area := circle_area small_radius
  large_circle_area - (num_small_circles : ℝ) * small_circle_area = 942 := by
  sorry

end shaded_area_theorem_l1964_196486


namespace distance_after_two_hours_l1964_196414

/-- Alice's walking speed in miles per minute -/
def alice_speed : ℚ := 1 / 20

/-- Bob's jogging speed in miles per minute -/
def bob_speed : ℚ := 3 / 40

/-- Time elapsed in minutes -/
def time_elapsed : ℚ := 120

/-- The distance between Alice and Bob after 2 hours -/
def distance_between : ℚ := alice_speed * time_elapsed + bob_speed * time_elapsed

theorem distance_after_two_hours :
  distance_between = 15 := by sorry

end distance_after_two_hours_l1964_196414


namespace rectangle_perimeter_l1964_196422

/-- The perimeter of a rectangle with a long side of 1 meter and a short side
    that is 2/8 meter shorter than the long side is 3.5 meters. -/
theorem rectangle_perimeter : 
  let long_side : ℝ := 1
  let short_side : ℝ := long_side - 2/8
  let perimeter : ℝ := 2 * long_side + 2 * short_side
  perimeter = 3.5 := by sorry

end rectangle_perimeter_l1964_196422


namespace problem_solution_l1964_196410

theorem problem_solution (m n : ℝ) 
  (h1 : (m * Real.exp m) / (4 * n^2) = (Real.log n + Real.log 2) / Real.exp m)
  (h2 : Real.exp (2 * m) = 1 / m) :
  (n = Real.exp m / 2) ∧ 
  (m + n < 7/5) ∧ 
  (1 < 2*n - m^2 ∧ 2*n - m^2 < 3/2) := by
sorry

end problem_solution_l1964_196410


namespace expected_yield_for_80kg_fertilizer_l1964_196405

/-- Represents the regression line equation for rice yield based on fertilizer amount -/
def regression_line (x : ℝ) : ℝ := 5 * x + 250

/-- Theorem stating that the expected rice yield is 650 kg when 80 kg of fertilizer is applied -/
theorem expected_yield_for_80kg_fertilizer : 
  regression_line 80 = 650 := by sorry

end expected_yield_for_80kg_fertilizer_l1964_196405


namespace pascal_triangle_prob_one_or_two_l1964_196448

/-- Represents Pascal's Triangle up to a given number of rows -/
def PascalTriangle (n : ℕ) : Type := Unit

/-- Total number of elements in the first n rows of Pascal's Triangle -/
def totalElements (pt : PascalTriangle n) : ℕ := n * (n + 1) / 2

/-- Number of elements equal to 1 in the first n rows of Pascal's Triangle -/
def countOnes (pt : PascalTriangle n) : ℕ := 1 + 2 * (n - 1)

/-- Number of elements equal to 2 in the first n rows of Pascal's Triangle -/
def countTwos (pt : PascalTriangle n) : ℕ := 2 * (n - 3)

/-- Probability of selecting 1 or 2 from the first n rows of Pascal's Triangle -/
def probOneOrTwo (pt : PascalTriangle n) : ℚ :=
  (countOnes pt + countTwos pt : ℚ) / totalElements pt

theorem pascal_triangle_prob_one_or_two :
  ∃ (pt : PascalTriangle 20), probOneOrTwo pt = 73 / 210 := by
  sorry

end pascal_triangle_prob_one_or_two_l1964_196448
