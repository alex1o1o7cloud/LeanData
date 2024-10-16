import Mathlib

namespace NUMINAMATH_CALUDE_gcd_of_2_powers_l1590_159095

theorem gcd_of_2_powers : Nat.gcd (2^1040 - 1) (2^1030 - 1) = 1023 := by sorry

end NUMINAMATH_CALUDE_gcd_of_2_powers_l1590_159095


namespace NUMINAMATH_CALUDE_triangle_point_coordinates_l1590_159094

/-- Given a triangle ABC with the following properties:
  - A has coordinates (2, 8)
  - M has coordinates (4, 11) and is the midpoint of AB
  - L has coordinates (6, 6) and BL is the angle bisector of angle ABC
  Prove that the coordinates of point C are (6, 14) -/
theorem triangle_point_coordinates (A B C M L : ℝ × ℝ) : 
  A = (2, 8) →
  M = (4, 11) →
  L = (6, 6) →
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →  -- M is midpoint of AB
  (L.1 - B.1) * (C.2 - B.2) = (L.2 - B.2) * (C.1 - B.1) →  -- BL is angle bisector
  C = (6, 14) := by sorry

end NUMINAMATH_CALUDE_triangle_point_coordinates_l1590_159094


namespace NUMINAMATH_CALUDE_distance_to_y_axis_l1590_159023

/-- The distance from point P(x, -5) to the y-axis is 10 units, given that the distance
    from P to the x-axis is half the distance from P to the y-axis. -/
theorem distance_to_y_axis (x : ℝ) : 
  let P : ℝ × ℝ := (x, -5)
  let dist_to_x_axis := |P.2|
  let dist_to_y_axis := |P.1|
  dist_to_x_axis = (1/2) * dist_to_y_axis → dist_to_y_axis = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_distance_to_y_axis_l1590_159023


namespace NUMINAMATH_CALUDE_digit2012_is_zero_l1590_159072

/-- The sequence of digits obtained by writing positive integers in order -/
def digitSequence : ℕ → ℕ :=
  sorry

/-- The 2012th digit in the sequence -/
def digit2012 : ℕ := digitSequence 2012

theorem digit2012_is_zero : digit2012 = 0 := by
  sorry

end NUMINAMATH_CALUDE_digit2012_is_zero_l1590_159072


namespace NUMINAMATH_CALUDE_central_symmetry_line_symmetry_two_lines_max_distance_l1590_159002

-- Define the curve C
def C (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 + a * p.1 * p.2 = 1}

-- Statement 1: C is centrally symmetric about the origin for all a
theorem central_symmetry (a : ℝ) : ∀ p : ℝ × ℝ, p ∈ C a → (-p.1, -p.2) ∈ C a := by sorry

-- Statement 2: C is symmetric about the lines y = x and y = -x for all a
theorem line_symmetry (a : ℝ) : 
  (∀ p : ℝ × ℝ, p ∈ C a → (p.2, p.1) ∈ C a) ∧ 
  (∀ p : ℝ × ℝ, p ∈ C a → (-p.2, -p.1) ∈ C a) := by sorry

-- Statement 3: There exist at least two distinct values of a for which C represents two lines
theorem two_lines : ∃ a₁ a₂ : ℝ, a₁ ≠ a₂ ∧ 
  (∃ l₁ l₂ m₁ m₂ : ℝ → ℝ, C a₁ = {p : ℝ × ℝ | p.2 = l₁ p.1 ∨ p.2 = l₂ p.1} ∧ 
                          C a₂ = {p : ℝ × ℝ | p.2 = m₁ p.1 ∨ p.2 = m₂ p.1}) := by sorry

-- Statement 4: When a = 1, the maximum distance between any two points on C is 2√2
theorem max_distance : 
  (∀ p q : ℝ × ℝ, p ∈ C 1 → q ∈ C 1 → Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≤ 2 * Real.sqrt 2) ∧
  (∃ p q : ℝ × ℝ, p ∈ C 1 ∧ q ∈ C 1 ∧ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 2 * Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_central_symmetry_line_symmetry_two_lines_max_distance_l1590_159002


namespace NUMINAMATH_CALUDE_student_scores_l1590_159046

theorem student_scores (math physics chemistry : ℕ) : 
  math + physics = 32 →
  (math + chemistry) / 2 = 26 →
  ∃ x : ℕ, chemistry = physics + x ∧ x = 20 := by
sorry

end NUMINAMATH_CALUDE_student_scores_l1590_159046


namespace NUMINAMATH_CALUDE_largest_value_at_negative_one_l1590_159077

/-- A monic cubic polynomial with non-negative real roots and f(0) = -64 -/
def MonicCubicPolynomial : Type := 
  {f : ℝ → ℝ // ∃ (r₁ r₂ r₃ : ℝ), (∀ x, f x = (x - r₁) * (x - r₂) * (x - r₃)) ∧ 
                                  (r₁ ≥ 0 ∧ r₂ ≥ 0 ∧ r₃ ≥ 0) ∧
                                  (f 0 = -64)}

/-- The largest possible value of f(-1) for a MonicCubicPolynomial is -125 -/
theorem largest_value_at_negative_one (f : MonicCubicPolynomial) : 
  f.val (-1) ≤ -125 := by
  sorry

end NUMINAMATH_CALUDE_largest_value_at_negative_one_l1590_159077


namespace NUMINAMATH_CALUDE_triangle_problem_l1590_159034

/-- Given a triangle ABC with tanA = 1/4, tanB = 3/5, and AB = √17,
    prove that the measure of angle C is 3π/4 and the smallest side length is √2 -/
theorem triangle_problem (A B C : Real) (AB : Real) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π →
  Real.tan A = 1/4 →
  Real.tan B = 3/5 →
  AB = Real.sqrt 17 →
  C = 3*π/4 ∧ 
  (min AB (min (AB * Real.sin A / Real.sin C) (AB * Real.sin B / Real.sin C)) = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l1590_159034


namespace NUMINAMATH_CALUDE_sticker_difference_l1590_159051

/-- Represents the distribution of stickers in boxes following an arithmetic sequence -/
structure StickerDistribution where
  total : ℕ
  boxes : ℕ
  first : ℕ
  difference : ℕ

/-- Calculates the sum of an arithmetic sequence -/
def arithmeticSum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- Theorem stating the difference between highest and lowest sticker quantities -/
theorem sticker_difference (dist : StickerDistribution)
  (h1 : dist.total = 250)
  (h2 : dist.boxes = 5)
  (h3 : dist.first = 30)
  (h4 : arithmeticSum dist.first dist.difference dist.boxes = dist.total) :
  dist.first + (dist.boxes - 1) * dist.difference - dist.first = 40 := by
  sorry

#check sticker_difference

end NUMINAMATH_CALUDE_sticker_difference_l1590_159051


namespace NUMINAMATH_CALUDE_solve_equation_l1590_159008

theorem solve_equation (n : ℕ) : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^18 → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1590_159008


namespace NUMINAMATH_CALUDE_unique_number_with_two_perfect_square_increments_l1590_159016

theorem unique_number_with_two_perfect_square_increments : 
  ∃! n : ℕ, n > 1000 ∧ 
    ∃ a b : ℕ, (n + 79 = a^2) ∧ (n + 204 = b^2) ∧ 
    n = 3765 :=
by sorry

end NUMINAMATH_CALUDE_unique_number_with_two_perfect_square_increments_l1590_159016


namespace NUMINAMATH_CALUDE_congruent_count_l1590_159031

theorem congruent_count : ∃ (n : ℕ), n = (Finset.filter (fun x => x > 0 ∧ x < 500 ∧ x % 9 = 4) (Finset.range 500)).card ∧ n = 56 := by
  sorry

end NUMINAMATH_CALUDE_congruent_count_l1590_159031


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1590_159035

theorem min_value_of_expression (b : ℝ) (h : 8 * b^2 + 7 * b + 6 = 5) :
  ∃ (m : ℝ), (∀ b', 8 * b'^2 + 7 * b' + 6 = 5 → 3 * b' + 2 ≥ m) ∧ (3 * b + 2 = m) ∧ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1590_159035


namespace NUMINAMATH_CALUDE_chord_length_implies_a_values_point_m_existence_implies_a_range_l1590_159098

-- Define the circle C
def C (a : ℝ) (x y : ℝ) : Prop := (x - a)^2 + (y - a - 1)^2 = 9

-- Define the line l
def l (x y : ℝ) : Prop := x + y - 3 = 0

-- Define the point A
def A : ℝ × ℝ := (3, 0)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Part 1
theorem chord_length_implies_a_values (a : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, C a x₁ y₁ ∧ C a x₂ y₂ ∧ l x₁ y₁ ∧ l x₂ y₂ ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = 4) →
  a = -1 ∨ a = 3 :=
sorry

-- Part 2
theorem point_m_existence_implies_a_range (a : ℝ) :
  (∃ x y : ℝ, C a x y ∧ (x - 3)^2 + y^2 = 4 * (x^2 + y^2)) →
  (-1 - 5 * Real.sqrt 2 / 2 ≤ a ∧ a ≤ -1 - Real.sqrt 2 / 2) ∨
  (-1 + Real.sqrt 2 / 2 ≤ a ∧ a ≤ -1 + 5 * Real.sqrt 2 / 2) :=
sorry

end NUMINAMATH_CALUDE_chord_length_implies_a_values_point_m_existence_implies_a_range_l1590_159098


namespace NUMINAMATH_CALUDE_stratified_sampling_result_l1590_159089

/-- Proves that the total number of students sampled is 135 given the conditions of the stratified sampling problem -/
theorem stratified_sampling_result (grade10 : ℕ) (grade11 : ℕ) (grade12 : ℕ) (sampled10 : ℕ) 
  (h1 : grade10 = 2000)
  (h2 : grade11 = 1500)
  (h3 : grade12 = 1000)
  (h4 : sampled10 = 60) :
  (grade10 + grade11 + grade12) * sampled10 / grade10 = 135 := by
  sorry

#check stratified_sampling_result

end NUMINAMATH_CALUDE_stratified_sampling_result_l1590_159089


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1590_159087

theorem complex_fraction_equality : Complex.I * 2 / (1 + Complex.I) = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1590_159087


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l1590_159020

theorem rectangle_dimensions (x : ℝ) : 
  (x - 3 > 0) →
  (3 * x + 4 > 0) →
  ((x - 3) * (3 * x + 4) = 12 * x - 9) →
  (x = (17 + 5 * Real.sqrt 13) / 6) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l1590_159020


namespace NUMINAMATH_CALUDE_express_regular_speed_ratio_l1590_159055

/-- The speed ratio of express train to regular train -/
def speed_ratio : ℝ := 2.5

/-- Regular train travel time in hours -/
def regular_time : ℝ := 10

/-- Time difference between regular and express train arrival in hours -/
def time_difference : ℝ := 3

/-- Time after departure when both trains are at same distance from Moscow -/
def distance_equality_time : ℝ := 2

/-- Minimum waiting time for express train in hours -/
def min_wait_time : ℝ := 2.5

theorem express_regular_speed_ratio 
  (wait_time : ℝ) 
  (h_wait : wait_time > min_wait_time) 
  (h_express_time : regular_time - time_difference - wait_time > 0) 
  (h_distance_equality : 
    distance_equality_time * speed_ratio = (distance_equality_time + wait_time)) :
  speed_ratio = (wait_time + distance_equality_time) / distance_equality_time :=
sorry

end NUMINAMATH_CALUDE_express_regular_speed_ratio_l1590_159055


namespace NUMINAMATH_CALUDE_car_repair_cost_l1590_159013

/-- Proves that the repair cost is approximately 13000, given the initial cost,
    selling price, and profit percentage of a car sale. -/
theorem car_repair_cost (initial_cost selling_price : ℕ) (profit_percentage : ℚ) :
  initial_cost = 42000 →
  selling_price = 66900 →
  profit_percentage = 21636363636363637 / 100000000000000 →
  ∃ (repair_cost : ℕ), 
    (repair_cost ≥ 12999 ∧ repair_cost ≤ 13001) ∧
    profit_percentage = (selling_price - (initial_cost + repair_cost)) / (initial_cost + repair_cost) :=
by sorry

end NUMINAMATH_CALUDE_car_repair_cost_l1590_159013


namespace NUMINAMATH_CALUDE_prism_volume_l1590_159054

theorem prism_volume (x y z : Real) (h : Real) :
  x = Real.sqrt 9 →
  y = Real.sqrt 9 →
  h = 6 →
  (1 / 2 : Real) * x * y * h = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l1590_159054


namespace NUMINAMATH_CALUDE_ethanol_in_fuel_tank_l1590_159036

/-- Calculates the total amount of ethanol in a fuel tank -/
def total_ethanol (tank_capacity : ℝ) (fuel_a_volume : ℝ) (fuel_a_ethanol_percent : ℝ) (fuel_b_ethanol_percent : ℝ) : ℝ :=
  let fuel_b_volume := tank_capacity - fuel_a_volume
  let ethanol_a := fuel_a_volume * fuel_a_ethanol_percent
  let ethanol_b := fuel_b_volume * fuel_b_ethanol_percent
  ethanol_a + ethanol_b

/-- The theorem states that the total amount of ethanol in the specified fuel mixture is 30 gallons -/
theorem ethanol_in_fuel_tank :
  total_ethanol 208 82 0.12 0.16 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ethanol_in_fuel_tank_l1590_159036


namespace NUMINAMATH_CALUDE_stacy_paper_completion_time_l1590_159038

/-- The number of days Stacy has to complete her paper -/
def days_to_complete : ℕ := 66 / 11

/-- The total number of pages in Stacy's paper -/
def total_pages : ℕ := 66

/-- The number of pages Stacy has to write per day -/
def pages_per_day : ℕ := 11

theorem stacy_paper_completion_time :
  days_to_complete = 6 :=
by sorry

end NUMINAMATH_CALUDE_stacy_paper_completion_time_l1590_159038


namespace NUMINAMATH_CALUDE_monotonic_quadratic_range_l1590_159093

/-- A quadratic function f(x) = x^2 + 2(a-1)x + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

/-- The function f is monotonic on the interval [-4, 4] -/
def is_monotonic_on_interval (a : ℝ) : Prop :=
  (∀ x y, -4 ≤ x ∧ x < y ∧ y ≤ 4 → f a x < f a y) ∨
  (∀ x y, -4 ≤ x ∧ x < y ∧ y ≤ 4 → f a x > f a y)

/-- If f(x) = x^2 + 2(a-1)x + 2 is monotonic on the interval [-4, 4], then a ≤ -3 or a ≥ 5 -/
theorem monotonic_quadratic_range (a : ℝ) : 
  is_monotonic_on_interval a → a ≤ -3 ∨ a ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_quadratic_range_l1590_159093


namespace NUMINAMATH_CALUDE_least_number_with_remainder_l1590_159073

theorem least_number_with_remainder (n : ℕ) : n = 282 ↔ 
  (n > 0 ∧ 
   n % 31 = 3 ∧ 
   n % 9 = 3 ∧ 
   ∀ m : ℕ, m > 0 → m % 31 = 3 → m % 9 = 3 → m ≥ n) := by
sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_l1590_159073


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l1590_159079

/-- The smallest positive integer x such that 420x is a perfect square -/
def x : ℕ := 105

/-- The smallest positive integer y such that 420y is a perfect cube -/
def y : ℕ := 22050

/-- 420 * x is a perfect square -/
axiom x_square : ∃ n : ℕ, 420 * x = n * n

/-- 420 * y is a perfect cube -/
axiom y_cube : ∃ n : ℕ, 420 * y = n * n * n

/-- x is the smallest positive integer such that 420x is a perfect square -/
axiom x_smallest : ∀ z : ℕ, z > 0 → z < x → ¬∃ n : ℕ, 420 * z = n * n

/-- y is the smallest positive integer such that 420y is a perfect cube -/
axiom y_smallest : ∀ z : ℕ, z > 0 → z < y → ¬∃ n : ℕ, 420 * z = n * n * n

theorem sum_of_x_and_y : x + y = 22155 := by sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l1590_159079


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1590_159006

/-- Two planar vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ a.1 * b.2 = k * a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (4, 1)
  let b : ℝ × ℝ := (x, 2)
  are_parallel a b → x = 8 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1590_159006


namespace NUMINAMATH_CALUDE_remaining_apples_l1590_159064

def initial_apples : ℕ := 150

def sold_to_jill (apples : ℕ) : ℕ :=
  apples - (apples * 30 / 100)

def sold_to_june (apples : ℕ) : ℕ :=
  apples - (apples * 20 / 100)

def give_to_teacher (apples : ℕ) : ℕ :=
  apples - 1

theorem remaining_apples :
  give_to_teacher (sold_to_june (sold_to_jill initial_apples)) = 83 := by
  sorry

end NUMINAMATH_CALUDE_remaining_apples_l1590_159064


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l1590_159014

/-- The discriminant of a quadratic equation ax² + bx + c -/
def discriminant (a b c : ℚ) : ℚ := b^2 - 4*a*c

/-- The coefficients of the quadratic equation 2x² + (3 - 1/2)x + 1/2 -/
def a : ℚ := 2
def b : ℚ := 3 - 1/2
def c : ℚ := 1/2

theorem quadratic_discriminant : discriminant a b c = 9/4 := by sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l1590_159014


namespace NUMINAMATH_CALUDE_sum_of_odd_powers_l1590_159015

theorem sum_of_odd_powers (x y z a : ℝ) (k : ℕ) 
  (h1 : x + y + z = a) 
  (h2 : x^3 + y^3 + z^3 = a^3) 
  (h3 : Odd k) : 
  x^k + y^k + z^k = a^k := by
  sorry

end NUMINAMATH_CALUDE_sum_of_odd_powers_l1590_159015


namespace NUMINAMATH_CALUDE_park_walking_area_l1590_159028

/-- The area available for walking in a rectangular park with a centered circular fountain -/
theorem park_walking_area (park_length park_width fountain_radius : ℝ) 
  (h1 : park_length = 50)
  (h2 : park_width = 30)
  (h3 : fountain_radius = 5) : 
  park_length * park_width - Real.pi * fountain_radius^2 = 1500 - 25 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_park_walking_area_l1590_159028


namespace NUMINAMATH_CALUDE_linear_functions_property_l1590_159007

/-- Given two linear functions f and g with specific properties, prove that A + B + 2C equals itself. -/
theorem linear_functions_property (A B C : ℝ) (h1 : A ≠ B) (h2 : A + B ≠ 0) (h3 : C ≠ 0)
  (f : ℝ → ℝ) (g : ℝ → ℝ)
  (hf : ∀ x, f x = A * x + B + C)
  (hg : ∀ x, g x = B * x + A - C)
  (h4 : ∀ x, f (g x) - g (f x) = 2 * C) :
  A + B + 2 * C = A + B + 2 * C := by
  sorry

end NUMINAMATH_CALUDE_linear_functions_property_l1590_159007


namespace NUMINAMATH_CALUDE_driver_hourly_wage_l1590_159096

/-- Calculates the hourly wage of a driver after fuel costs --/
theorem driver_hourly_wage
  (speed : ℝ)
  (time : ℝ)
  (fuel_efficiency : ℝ)
  (income_per_mile : ℝ)
  (fuel_cost_per_gallon : ℝ)
  (h1 : speed = 60)
  (h2 : time = 2)
  (h3 : fuel_efficiency = 30)
  (h4 : income_per_mile = 0.5)
  (h5 : fuel_cost_per_gallon = 2)
  : (income_per_mile * speed * time - (speed * time / fuel_efficiency) * fuel_cost_per_gallon) / time = 26 := by
  sorry

end NUMINAMATH_CALUDE_driver_hourly_wage_l1590_159096


namespace NUMINAMATH_CALUDE_max_product_constraint_l1590_159049

theorem max_product_constraint (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 4 * b = 8) :
  a * b ≤ 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 4 * b₀ = 8 ∧ a₀ * b₀ = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constraint_l1590_159049


namespace NUMINAMATH_CALUDE_roses_cut_theorem_l1590_159075

/-- The number of roses Jessica cut from her flower garden -/
def roses_cut (initial_roses final_roses : ℕ) : ℕ :=
  final_roses - initial_roses

/-- Theorem: The number of roses Jessica cut is equal to the difference between the final and initial number of roses -/
theorem roses_cut_theorem (initial_roses final_roses : ℕ) 
  (h : final_roses ≥ initial_roses) : 
  roses_cut initial_roses final_roses = final_roses - initial_roses :=
by
  sorry

#eval roses_cut 10 18  -- Should output 8

end NUMINAMATH_CALUDE_roses_cut_theorem_l1590_159075


namespace NUMINAMATH_CALUDE_closure_union_M_N_l1590_159067

-- Define the sets M and N
def M : Set ℝ := {x | (x + 3) * (x - 1) < 0}
def N : Set ℝ := {x | x ≤ -3}

-- State the theorem
theorem closure_union_M_N :
  closure (M ∪ N) = {x : ℝ | x ≥ 1} :=
sorry

end NUMINAMATH_CALUDE_closure_union_M_N_l1590_159067


namespace NUMINAMATH_CALUDE_forty_percent_relation_l1590_159018

theorem forty_percent_relation (x : ℝ) (v : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * x = v → (40/100 : ℝ) * x = 12 * v := by
  sorry

end NUMINAMATH_CALUDE_forty_percent_relation_l1590_159018


namespace NUMINAMATH_CALUDE_map_distance_conversion_l1590_159092

/-- Given a map scale where 312 inches represents 136 km,
    prove that 34 inches on the map corresponds to approximately 14.82 km in actual distance. -/
theorem map_distance_conversion (map_distance : ℝ) (actual_distance : ℝ) (ram_map_distance : ℝ)
  (h1 : map_distance = 312)
  (h2 : actual_distance = 136)
  (h3 : ram_map_distance = 34) :
  ∃ (ε : ℝ), ε > 0 ∧ abs ((actual_distance / map_distance) * ram_map_distance - 14.82) < ε :=
sorry

end NUMINAMATH_CALUDE_map_distance_conversion_l1590_159092


namespace NUMINAMATH_CALUDE_probability_odd_divisor_15_factorial_l1590_159024

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

def is_odd (n : ℕ) : Prop := n % 2 = 1

def count_divisors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (fun acc (p, e) => acc * (e + 1)) 1

def count_odd_divisors (factors : List (ℕ × ℕ)) : ℕ :=
  (factors.filter (fun (p, _) => p ≠ 2)).foldl (fun acc (_, e) => acc * (e + 1)) 1

theorem probability_odd_divisor_15_factorial :
  let f15 := factorial 15
  let factors := prime_factorization f15
  let total_divisors := count_divisors factors
  let odd_divisors := count_odd_divisors factors
  (odd_divisors : ℚ) / total_divisors = 1 / 12 := by sorry

end NUMINAMATH_CALUDE_probability_odd_divisor_15_factorial_l1590_159024


namespace NUMINAMATH_CALUDE_min_value_of_sum_l1590_159065

theorem min_value_of_sum (a b : ℝ) : 
  a > 0 → b > 0 → (1 / a + 1 / b = 1) → (∀ x y : ℝ, a > 0 ∧ b > 0 ∧ x / a + y / b = 1 → a + 4 * b ≥ 9) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l1590_159065


namespace NUMINAMATH_CALUDE_cubic_polynomial_root_l1590_159027

theorem cubic_polynomial_root (a b : ℚ) :
  (∃ (x : ℝ), x^3 + a*x + b = 0 ∧ x = 3 - Real.sqrt 5) →
  (∃ (r : ℤ), r^3 + a*r + b = 0) →
  (∃ (r : ℤ), r^3 + a*r + b = 0 ∧ r = 0) :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_root_l1590_159027


namespace NUMINAMATH_CALUDE_tan_theta_value_l1590_159056

theorem tan_theta_value (θ : Real) (h1 : 0 < θ) (h2 : θ < π/4) 
  (h3 : Real.tan θ + Real.tan (4*θ) = 0) : 
  Real.tan θ = Real.sqrt (5 - 2 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_value_l1590_159056


namespace NUMINAMATH_CALUDE_inequality_proof_l1590_159052

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_inequality : a * b + b * c + c * a ≥ 1) :
  1 / a^2 + 1 / b^2 + 1 / c^2 ≥ Real.sqrt 3 / (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1590_159052


namespace NUMINAMATH_CALUDE_sprocket_production_l1590_159053

theorem sprocket_production (machine_p machine_q machine_a : ℕ → ℕ) : 
  (∃ t_q : ℕ, 
    machine_p (t_q + 10) = 550 ∧ 
    machine_q t_q = 550 ∧ 
    (∀ t, machine_q t = (11 * machine_a t) / 10)) → 
  machine_a 1 = 5 := by
sorry

end NUMINAMATH_CALUDE_sprocket_production_l1590_159053


namespace NUMINAMATH_CALUDE_count_possible_denominators_all_denominators_divide_999_fraction_denominator_in_possible_set_seven_possible_denominators_l1590_159044

/-- Represents a three-digit number abc --/
structure ThreeDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  a_is_digit : a < 10
  b_is_digit : b < 10
  c_is_digit : c < 10
  not_all_nines : ¬(a = 9 ∧ b = 9 ∧ c = 9)
  not_all_zeros : ¬(a = 0 ∧ b = 0 ∧ c = 0)

/-- Converts a ThreeDigitNumber to its decimal value --/
def toDecimal (n : ThreeDigitNumber) : Nat :=
  100 * n.a + 10 * n.b + n.c

/-- The denominator of the fraction representation of 0.abc̅ --/
def denominator : Nat := 999

/-- The set of possible denominators for 0.abc̅ in lowest terms --/
def possibleDenominators : Finset Nat :=
  {3, 9, 27, 37, 111, 333, 999}

/-- Theorem stating that there are exactly 7 possible denominators --/
theorem count_possible_denominators :
    (possibleDenominators.card : Nat) = 7 := by sorry

/-- Theorem stating that all elements in possibleDenominators are factors of 999 --/
theorem all_denominators_divide_999 :
    ∀ d ∈ possibleDenominators, denominator % d = 0 := by sorry

/-- Theorem stating that for any ThreeDigitNumber, its fraction representation
    has a denominator in possibleDenominators --/
theorem fraction_denominator_in_possible_set (n : ThreeDigitNumber) :
    ∃ d ∈ possibleDenominators,
      (toDecimal n).gcd denominator = (denominator / d) := by sorry

/-- Main theorem proving that there are exactly 7 possible denominators --/
theorem seven_possible_denominators :
    ∃! (s : Finset Nat),
      (∀ n : ThreeDigitNumber,
        ∃ d ∈ s, (toDecimal n).gcd denominator = (denominator / d)) ∧
      s.card = 7 := by sorry

end NUMINAMATH_CALUDE_count_possible_denominators_all_denominators_divide_999_fraction_denominator_in_possible_set_seven_possible_denominators_l1590_159044


namespace NUMINAMATH_CALUDE_consecutive_integers_average_l1590_159010

theorem consecutive_integers_average (a b : ℤ) : 
  (a > 0) → 
  (b = (7 * a + 21) / 7) → 
  ((7 * b + 21) / 7 = a + 6) := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_average_l1590_159010


namespace NUMINAMATH_CALUDE_rectangle_area_difference_l1590_159071

def is_valid_rectangle (l w : ℕ) : Prop :=
  2 * l + 2 * w = 56 ∧ (l ≥ w + 5 ∨ w ≥ l + 5)

def rectangle_area (l w : ℕ) : ℕ := l * w

theorem rectangle_area_difference : 
  ∃ (l₁ w₁ l₂ w₂ : ℕ),
    is_valid_rectangle l₁ w₁ ∧
    is_valid_rectangle l₂ w₂ ∧
    ∀ (l w : ℕ),
      is_valid_rectangle l w →
      rectangle_area l w ≤ rectangle_area l₁ w₁ ∧
      rectangle_area l w ≥ rectangle_area l₂ w₂ ∧
      rectangle_area l₁ w₁ - rectangle_area l₂ w₂ = 5 :=
sorry

end NUMINAMATH_CALUDE_rectangle_area_difference_l1590_159071


namespace NUMINAMATH_CALUDE_peters_age_l1590_159086

theorem peters_age (P Q : ℝ) 
  (h1 : Q - P = P / 2)
  (h2 : P + Q = 35) :
  Q = 21 := by sorry

end NUMINAMATH_CALUDE_peters_age_l1590_159086


namespace NUMINAMATH_CALUDE_metallic_sheet_width_l1590_159063

/-- Given a rectangular metallic sheet with length 48 m, where squares of 8 m are cut from each corner
    to form a box with volume 5120 m³, the width of the original metallic sheet is 36 m. -/
theorem metallic_sheet_width :
  ∀ (w : ℝ),
    (48 - 2 * 8) * (w - 2 * 8) * 8 = 5120 →
    w = 36 :=
by sorry

end NUMINAMATH_CALUDE_metallic_sheet_width_l1590_159063


namespace NUMINAMATH_CALUDE_mold_diameter_l1590_159019

/-- The diameter of a circular mold with radius 2 inches is 4 inches. -/
theorem mold_diameter (r : ℝ) (h : r = 2) : 2 * r = 4 := by
  sorry

end NUMINAMATH_CALUDE_mold_diameter_l1590_159019


namespace NUMINAMATH_CALUDE_no_prime_solution_l1590_159037

theorem no_prime_solution : 
  ¬∃ (p : ℕ), Nat.Prime p ∧ 
  (p ^ 3 + 7) + (3 * p ^ 2 + 6) + (p ^ 2 + p + 3) + (p ^ 2 + 2 * p + 5) + 6 = 
  (p ^ 2 + 4 * p + 2) + (2 * p ^ 2 + 7 * p + 1) + (3 * p ^ 2 + 6 * p) :=
by sorry

end NUMINAMATH_CALUDE_no_prime_solution_l1590_159037


namespace NUMINAMATH_CALUDE_ratio_sum_difference_l1590_159097

theorem ratio_sum_difference (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x / y = (x + y) / (x - y) → x / y = 1 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_difference_l1590_159097


namespace NUMINAMATH_CALUDE_lowest_common_denominator_l1590_159009

theorem lowest_common_denominator (a b c : Nat) : a = 9 → b = 4 → c = 18 → Nat.lcm a (Nat.lcm b c) = 36 := by
  sorry

end NUMINAMATH_CALUDE_lowest_common_denominator_l1590_159009


namespace NUMINAMATH_CALUDE_cube_less_than_triple_l1590_159059

theorem cube_less_than_triple (x : ℤ) : x^3 < 3*x ↔ x = -3 ∨ x = -2 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_less_than_triple_l1590_159059


namespace NUMINAMATH_CALUDE_floor_ceil_sum_l1590_159011

theorem floor_ceil_sum : ⌊(-3.67 : ℝ)⌋ + ⌈(34.2 : ℝ)⌉ = 31 := by sorry

end NUMINAMATH_CALUDE_floor_ceil_sum_l1590_159011


namespace NUMINAMATH_CALUDE_yellow_balls_count_l1590_159030

theorem yellow_balls_count (Y : ℕ) : 
  (Y : ℝ) / (Y + 2) * ((Y - 1) / (Y + 1)) = 1 / 2 → Y = 5 := by
  sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l1590_159030


namespace NUMINAMATH_CALUDE_angles_with_same_terminal_side_eq_l1590_159032

/-- Given an angle α whose terminal side is the same as 8π/5, 
    this function returns the set of angles in [0, 2π] 
    whose terminal sides are the same as α/4 -/
def anglesWithSameTerminalSide (α : ℝ) : Set ℝ :=
  {x | x ∈ Set.Icc 0 (2 * Real.pi) ∧ 
       ∃ k : ℤ, α = 2 * k * Real.pi + 8 * Real.pi / 5 ∧ 
               x = (k * Real.pi / 2 + 2 * Real.pi / 5) % (2 * Real.pi)}

/-- Theorem stating that the set of angles with the same terminal side as α/4 
    is equal to the specific set of four angles -/
theorem angles_with_same_terminal_side_eq (α : ℝ) 
    (h : ∃ k : ℤ, α = 2 * k * Real.pi + 8 * Real.pi / 5) : 
  anglesWithSameTerminalSide α = {2 * Real.pi / 5, 9 * Real.pi / 10, 7 * Real.pi / 5, 19 * Real.pi / 10} := by
  sorry

end NUMINAMATH_CALUDE_angles_with_same_terminal_side_eq_l1590_159032


namespace NUMINAMATH_CALUDE_statement_independent_of_parallel_postulate_l1590_159029

-- Define a geometry
class Geometry where
  -- Define the concept of a line
  Line : Type
  -- Define the concept of a point
  Point : Type
  -- Define the concept of parallelism
  parallel : Line → Line → Prop
  -- Define the concept of intersection
  intersects : Line → Line → Prop

-- Define the statement to be proven
def statement (G : Geometry) : Prop :=
  ∀ (l₁ l₂ l₃ : G.Line),
    G.parallel l₁ l₂ → G.intersects l₃ l₁ → G.intersects l₃ l₂

-- Define the parallel postulate
def parallel_postulate (G : Geometry) : Prop :=
  ∀ (p : G.Point) (l : G.Line),
    ∃! (m : G.Line), G.parallel l m

-- Theorem: The statement is independent of the parallel postulate
theorem statement_independent_of_parallel_postulate :
  ∀ (G : Geometry),
    (statement G ↔ statement G) ∧ 
    (¬(parallel_postulate G → statement G)) ∧
    (¬(statement G → parallel_postulate G)) :=
sorry

end NUMINAMATH_CALUDE_statement_independent_of_parallel_postulate_l1590_159029


namespace NUMINAMATH_CALUDE_boy_scouts_permission_slips_l1590_159057

theorem boy_scouts_permission_slips 
  (total_scouts : ℕ) 
  (total_with_slips : ℝ) 
  (total_boys : ℝ) 
  (girl_scouts_with_slips : ℝ) 
  (h1 : total_with_slips = 0.8 * total_scouts)
  (h2 : total_boys = 0.4 * total_scouts)
  (h3 : girl_scouts_with_slips = 0.8333 * (total_scouts - total_boys)) :
  (total_with_slips - girl_scouts_with_slips) / total_boys = 0.75 := by
sorry

end NUMINAMATH_CALUDE_boy_scouts_permission_slips_l1590_159057


namespace NUMINAMATH_CALUDE_min_a_sqrt_sum_l1590_159060

theorem min_a_sqrt_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (∀ x y, x > 0 → y > 0 → Real.sqrt x + Real.sqrt y ≤ a * Real.sqrt (x + y)) →
  ∃ a_min : ℝ, a_min = Real.sqrt 2 ∧ ∀ a : ℝ, (∀ x y, x > 0 → y > 0 → Real.sqrt x + Real.sqrt y ≤ a * Real.sqrt (x + y)) → a ≥ a_min :=
by sorry

end NUMINAMATH_CALUDE_min_a_sqrt_sum_l1590_159060


namespace NUMINAMATH_CALUDE_base7_equals_base10_l1590_159081

/-- Converts a number from base 7 to base 10 -/
def base7To10 (n : ℕ) : ℕ := sorry

/-- Represents a base-10 digit (0-9) -/
def Digit := {d : ℕ // d < 10}

theorem base7_equals_base10 (c d : Digit) :
  base7To10 764 = 400 + 10 * c.val + d.val →
  (c.val * d.val) / 20 = 6 / 5 := by sorry

end NUMINAMATH_CALUDE_base7_equals_base10_l1590_159081


namespace NUMINAMATH_CALUDE_video_game_points_l1590_159040

/-- The number of points earned in a video game level --/
def points_earned (total_enemies : ℕ) (enemies_left : ℕ) (points_per_enemy : ℕ) : ℕ :=
  (total_enemies - enemies_left) * points_per_enemy

/-- Theorem: In the given scenario, the player earns 40 points --/
theorem video_game_points : points_earned 7 2 8 = 40 := by
  sorry

end NUMINAMATH_CALUDE_video_game_points_l1590_159040


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_6_l1590_159091

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  first_term : a 1 = 2
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d
  sum : ℕ → ℝ
  sum_def : ∀ n : ℕ, sum n = n * (2 * a 1 + (n - 1) * d) / 2

/-- The theorem to be proved -/
theorem arithmetic_sequence_sum_6 (seq : ArithmeticSequence) (h : seq.sum 4 = 20) :
  seq.sum 6 = 42 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_6_l1590_159091


namespace NUMINAMATH_CALUDE_sum_of_abs_coeffs_of_2x_minus_1_to_6th_l1590_159058

theorem sum_of_abs_coeffs_of_2x_minus_1_to_6th (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℤ) :
  (∀ x, (2*x - 1)^6 = a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  (|a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| = 729) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_abs_coeffs_of_2x_minus_1_to_6th_l1590_159058


namespace NUMINAMATH_CALUDE_prism_pyramid_sum_l1590_159043

/-- A shape formed by adding a pyramid to one square face of a rectangular prism -/
structure PrismPyramid where
  prism_faces : ℕ
  prism_edges : ℕ
  prism_vertices : ℕ
  pyramid_faces : ℕ
  pyramid_edges : ℕ
  pyramid_vertices : ℕ

/-- The sum of faces, edges, and vertices of the PrismPyramid -/
def total_sum (pp : PrismPyramid) : ℕ :=
  (pp.prism_faces - 1 + pp.pyramid_faces) + 
  (pp.prism_edges + pp.pyramid_edges) + 
  (pp.prism_vertices + pp.pyramid_vertices)

/-- Theorem stating that the total sum is 34 -/
theorem prism_pyramid_sum :
  ∀ (pp : PrismPyramid), 
    pp.prism_faces = 6 ∧ 
    pp.prism_edges = 12 ∧ 
    pp.prism_vertices = 8 ∧
    pp.pyramid_faces = 4 ∧
    pp.pyramid_edges = 4 ∧
    pp.pyramid_vertices = 1 →
    total_sum pp = 34 := by
  sorry

end NUMINAMATH_CALUDE_prism_pyramid_sum_l1590_159043


namespace NUMINAMATH_CALUDE_smallest_m_is_30_l1590_159078

def probability_condition (m : ℕ) : Prop :=
  (1 / 6) * ((m - 4) ^ 3 : ℚ) / (m ^ 3 : ℚ) > 3 / 5

theorem smallest_m_is_30 :
  ∀ k : ℕ, k > 0 → (probability_condition k → k ≥ 30) ∧
  probability_condition 30 := by sorry

end NUMINAMATH_CALUDE_smallest_m_is_30_l1590_159078


namespace NUMINAMATH_CALUDE_quadratic_root_sum_product_ratio_l1590_159061

theorem quadratic_root_sum_product_ratio : 
  ∀ x₁ x₂ : ℝ, x₁^2 - 2*x₁ - 8 = 0 → x₂^2 - 2*x₂ - 8 = 0 → 
  (x₁ + x₂) / (x₁ * x₂) = -1/4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_product_ratio_l1590_159061


namespace NUMINAMATH_CALUDE_height_to_hypotenuse_not_always_half_l1590_159070

theorem height_to_hypotenuse_not_always_half : ∃ (a b c h : ℝ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ h > 0 ∧
  a^2 + b^2 = c^2 ∧  -- right triangle condition
  h ≠ c / 2 ∧        -- height is not half of hypotenuse
  h * c = a * b      -- height formula
  := by sorry

end NUMINAMATH_CALUDE_height_to_hypotenuse_not_always_half_l1590_159070


namespace NUMINAMATH_CALUDE_fraction_subtraction_l1590_159022

theorem fraction_subtraction : (18 : ℚ) / 42 - 3 / 8 = 3 / 56 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l1590_159022


namespace NUMINAMATH_CALUDE_power_function_value_l1590_159090

/-- A power function is a function of the form f(x) = x^α for some real α -/
def IsPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

theorem power_function_value (f : ℝ → ℝ) (h1 : IsPowerFunction f) (h2 : f 2 / f 4 = 1 / 2) :
  f 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_value_l1590_159090


namespace NUMINAMATH_CALUDE_expression_evaluation_l1590_159076

theorem expression_evaluation :
  let a : ℚ := -1/2
  (4 - 3*a)*(1 + 2*a) - 3*a*(1 - 2*a) = 3 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1590_159076


namespace NUMINAMATH_CALUDE_lottery_distribution_l1590_159062

theorem lottery_distribution (lottery_win : ℝ) (recipients : ℕ) : 
  lottery_win = 155250 →
  recipients = 100 →
  (lottery_win / 1000) * recipients = 15525 := by
  sorry

end NUMINAMATH_CALUDE_lottery_distribution_l1590_159062


namespace NUMINAMATH_CALUDE_socks_combination_l1590_159088

/-- The number of ways to choose k items from a set of n items, where order doesn't matter -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- There are 6 socks in the drawer -/
def total_socks : ℕ := 6

/-- We need to choose 4 socks -/
def socks_to_choose : ℕ := 4

/-- The number of ways to choose 4 socks from 6 socks is 15 -/
theorem socks_combination : choose total_socks socks_to_choose = 15 := by
  sorry

end NUMINAMATH_CALUDE_socks_combination_l1590_159088


namespace NUMINAMATH_CALUDE_roots_sum_powers_l1590_159045

theorem roots_sum_powers (a b : ℝ) : 
  a + b = 6 → ab = 8 → a^2 + a^5 * b^3 + a^3 * b^5 + b^2 = 10260 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_powers_l1590_159045


namespace NUMINAMATH_CALUDE_stuffed_animals_count_l1590_159050

/-- The total number of stuffed animals for three girls -/
def total_stuffed_animals (mckenna kenley tenly : ℕ) : ℕ :=
  mckenna + kenley + tenly

/-- Theorem stating the total number of stuffed animals for the three girls -/
theorem stuffed_animals_count :
  ∃ (kenley tenly : ℕ),
    let mckenna := 34
    kenley = 2 * mckenna ∧
    tenly = kenley + 5 ∧
    total_stuffed_animals mckenna kenley tenly = 175 := by
  sorry

end NUMINAMATH_CALUDE_stuffed_animals_count_l1590_159050


namespace NUMINAMATH_CALUDE_min_value_expression_l1590_159004

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (x + 1/y) * (x + 1/y - 1024) + (y + 1/x) * (y + 1/x - 1024) ≥ -524288 := by
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1590_159004


namespace NUMINAMATH_CALUDE_every_positive_integer_appears_l1590_159047

/-- The smallest prime that doesn't divide k -/
def p (k : ℕ+) : ℕ := sorry

/-- The sequence a_n -/
def a : ℕ → ℕ+ → ℕ+
  | 0, a₀ => a₀
  | n + 1, a₀ => sorry

/-- Main theorem: every positive integer appears in the sequence -/
theorem every_positive_integer_appears (a₀ : ℕ+) :
  ∀ m : ℕ+, ∃ n : ℕ, a n a₀ = m := by sorry

end NUMINAMATH_CALUDE_every_positive_integer_appears_l1590_159047


namespace NUMINAMATH_CALUDE_monday_kids_count_l1590_159000

/-- The number of kids Julia played with on Tuesday -/
def tuesday_kids : ℕ := 18

/-- The total number of kids Julia played with on Monday and Tuesday combined -/
def monday_tuesday_total : ℕ := 33

/-- The number of kids Julia played with on Monday -/
def monday_kids : ℕ := monday_tuesday_total - tuesday_kids

theorem monday_kids_count : monday_kids = 15 := by
  sorry

end NUMINAMATH_CALUDE_monday_kids_count_l1590_159000


namespace NUMINAMATH_CALUDE_magnitude_of_z_l1590_159082

theorem magnitude_of_z (z : ℂ) (h : (Complex.I - 1) * z = (Complex.I + 1)^2) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l1590_159082


namespace NUMINAMATH_CALUDE_total_ways_is_2531_l1590_159085

/-- The number of different types of cookies -/
def num_cookie_types : ℕ := 6

/-- The number of different types of milk -/
def num_milk_types : ℕ := 4

/-- The total number of product types -/
def total_product_types : ℕ := num_cookie_types + num_milk_types

/-- The number of products they purchase collectively -/
def total_purchases : ℕ := 4

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := 
  if k > n then 0
  else (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- The number of ways Charlie and Delta can leave the store with 4 products collectively -/
def total_ways : ℕ :=
  -- Charlie 4, Delta 0
  choose total_product_types 4 +
  -- Charlie 3, Delta 1
  choose total_product_types 3 * num_cookie_types +
  -- Charlie 2, Delta 2
  choose total_product_types 2 * (choose num_cookie_types 2 + num_cookie_types) +
  -- Charlie 1, Delta 3
  total_product_types * (choose num_cookie_types 3 + num_cookie_types * (num_cookie_types - 1) + num_cookie_types) +
  -- Charlie 0, Delta 4
  (choose num_cookie_types 4 + num_cookie_types * (num_cookie_types - 1) + choose num_cookie_types 2 * 3 + num_cookie_types)

theorem total_ways_is_2531 : total_ways = 2531 := by
  sorry

end NUMINAMATH_CALUDE_total_ways_is_2531_l1590_159085


namespace NUMINAMATH_CALUDE_a_4_equals_zero_l1590_159080

def sequence_a (n : ℕ+) : ℤ := n.val^2 - 2*n.val - 8

theorem a_4_equals_zero : sequence_a 4 = 0 := by sorry

end NUMINAMATH_CALUDE_a_4_equals_zero_l1590_159080


namespace NUMINAMATH_CALUDE_profit_percentage_10_12_l1590_159017

/-- Calculates the profit percentage when selling n articles at the cost price of m articles -/
def profit_percentage (n m : ℕ) : ℚ :=
  ((m : ℚ) - (n : ℚ)) / (n : ℚ) * 100

/-- Theorem: The profit percentage when selling 10 articles at the cost price of 12 articles is 20% -/
theorem profit_percentage_10_12 : profit_percentage 10 12 = 20 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_10_12_l1590_159017


namespace NUMINAMATH_CALUDE_total_cupcakes_calculation_l1590_159048

/-- The number of cupcakes ordered for each event -/
def cupcakes_per_event : ℝ := 96.0

/-- The number of different children's events -/
def number_of_events : ℝ := 8.0

/-- The total number of cupcakes needed -/
def total_cupcakes : ℝ := cupcakes_per_event * number_of_events

theorem total_cupcakes_calculation : total_cupcakes = 768.0 := by
  sorry

end NUMINAMATH_CALUDE_total_cupcakes_calculation_l1590_159048


namespace NUMINAMATH_CALUDE_subtract_like_terms_l1590_159066

theorem subtract_like_terms (a : ℝ) : 7 * a^2 - 4 * a^2 = 3 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_subtract_like_terms_l1590_159066


namespace NUMINAMATH_CALUDE_max_robot_weight_is_270_l1590_159042

/-- Represents the weight constraints and components of a robot in the competition. -/
structure RobotWeightConstraints where
  standard_robot_weight : ℝ
  battery_weight : ℝ
  min_payload_weight : ℝ
  max_payload_weight : ℝ
  min_robot_weight_diff : ℝ

/-- Calculates the maximum weight of a robot in the competition. -/
def max_robot_weight (constraints : RobotWeightConstraints) : ℝ :=
  let min_robot_weight := constraints.standard_robot_weight + constraints.min_robot_weight_diff
  let min_total_weight := min_robot_weight + constraints.battery_weight + constraints.min_payload_weight
  2 * min_total_weight

/-- Theorem stating the maximum weight of a robot in the competition. -/
theorem max_robot_weight_is_270 (constraints : RobotWeightConstraints) 
    (h1 : constraints.standard_robot_weight = 100)
    (h2 : constraints.battery_weight = 20)
    (h3 : constraints.min_payload_weight = 10)
    (h4 : constraints.max_payload_weight = 25)
    (h5 : constraints.min_robot_weight_diff = 5) :
  max_robot_weight constraints = 270 := by
  sorry

#eval max_robot_weight { 
  standard_robot_weight := 100,
  battery_weight := 20,
  min_payload_weight := 10,
  max_payload_weight := 25,
  min_robot_weight_diff := 5
}

end NUMINAMATH_CALUDE_max_robot_weight_is_270_l1590_159042


namespace NUMINAMATH_CALUDE_max_y_value_l1590_159068

theorem max_y_value (x y : ℝ) (h : (x + y)^4 = x - y) :
  y ≤ 3 * Real.rpow 2 (1/3) / 16 := by
  sorry

end NUMINAMATH_CALUDE_max_y_value_l1590_159068


namespace NUMINAMATH_CALUDE_stair_climbing_time_l1590_159012

theorem stair_climbing_time (a₁ : ℕ) (d : ℕ) (n : ℕ) (h1 : a₁ = 30) (h2 : d = 7) (h3 : n = 8) :
  n * (2 * a₁ + (n - 1) * d) / 2 = 436 := by
  sorry

end NUMINAMATH_CALUDE_stair_climbing_time_l1590_159012


namespace NUMINAMATH_CALUDE_alternating_sequences_20_l1590_159003

/-- A function that computes the number of alternating sequences -/
def A : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => A (n + 1) + A n

/-- The number of alternating sequences for n = 20 is 10946 -/
theorem alternating_sequences_20 : A 20 = 10946 := by
  sorry

end NUMINAMATH_CALUDE_alternating_sequences_20_l1590_159003


namespace NUMINAMATH_CALUDE_mechanic_hourly_rate_l1590_159074

/-- The mechanic's hourly rate calculation -/
theorem mechanic_hourly_rate :
  let hours_per_day : ℕ := 8
  let days_worked : ℕ := 14
  let parts_cost : ℕ := 2500
  let total_cost : ℕ := 9220
  let total_hours : ℕ := hours_per_day * days_worked
  let labor_cost : ℕ := total_cost - parts_cost
  labor_cost / total_hours = 60 := by sorry

end NUMINAMATH_CALUDE_mechanic_hourly_rate_l1590_159074


namespace NUMINAMATH_CALUDE_photocopy_pages_theorem_l1590_159025

/-- The number of team members -/
def team_members : ℕ := 23

/-- The cost per page for the first 300 pages (in tenths of yuan) -/
def cost_first_300 : ℕ := 15

/-- The cost per page for additional pages beyond 300 (in tenths of yuan) -/
def cost_additional : ℕ := 10

/-- The threshold number of pages for price change -/
def threshold : ℕ := 300

/-- The ratio of total cost to single set cost -/
def cost_ratio : ℕ := 20

/-- The function to calculate the cost of photocopying a single set of materials -/
def single_set_cost (pages : ℕ) : ℕ :=
  if pages ≤ threshold then
    pages * cost_first_300
  else
    threshold * cost_first_300 + (pages - threshold) * cost_additional

/-- The function to calculate the cost of photocopying all sets of materials -/
def total_cost (pages : ℕ) : ℕ :=
  if team_members * pages ≤ threshold then
    team_members * pages * cost_first_300
  else
    threshold * cost_first_300 + (team_members * pages - threshold) * cost_additional

/-- The theorem stating that 950 pages satisfies the given conditions -/
theorem photocopy_pages_theorem :
  ∃ (pages : ℕ), pages = 950 ∧ total_cost pages = cost_ratio * single_set_cost pages :=
sorry

end NUMINAMATH_CALUDE_photocopy_pages_theorem_l1590_159025


namespace NUMINAMATH_CALUDE_quadratic_roots_l1590_159026

theorem quadratic_roots : ∃ (x₁ x₂ : ℝ), x₁ = 0 ∧ x₂ = 2 ∧ 
  (∀ x : ℝ, x^2 - 2*x = 0 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l1590_159026


namespace NUMINAMATH_CALUDE_common_tangents_l1590_159083

/-- The first curve: 9x^2 + 16y^2 = 144 -/
def curve1 (x y : ℝ) : Prop := 9 * x^2 + 16 * y^2 = 144

/-- The second curve: 7x^2 - 32y^2 = 224 -/
def curve2 (x y : ℝ) : Prop := 7 * x^2 - 32 * y^2 = 224

/-- A common tangent line: ax + by + c = 0 -/
def is_tangent (a b c : ℝ) : Prop :=
  ∀ x y : ℝ, (curve1 x y ∨ curve2 x y) → (a * x + b * y + c = 0 → 
    ∀ ε > 0, ∃ δ > 0, ∀ x' y' : ℝ, 
      ((x' - x)^2 + (y' - y)^2 < δ^2) → 
      ((curve1 x' y' ∨ curve2 x' y') → (a * x' + b * y' + c ≠ 0)))

/-- The theorem stating that the given equations are common tangents -/
theorem common_tangents : 
  (is_tangent 1 1 5 ∧ is_tangent 1 1 (-5) ∧ is_tangent 1 (-1) 5 ∧ is_tangent 1 (-1) (-5)) :=
sorry

end NUMINAMATH_CALUDE_common_tangents_l1590_159083


namespace NUMINAMATH_CALUDE_coffee_savings_l1590_159041

/-- Calculates the savings in daily coffee expenditure after a price increase and consumption reduction -/
theorem coffee_savings (original_coffees : ℕ) (original_price : ℚ) (price_increase : ℚ) : 
  let new_price := original_price * (1 + price_increase)
  let new_coffees := original_coffees / 2
  let original_spending := original_coffees * original_price
  let new_spending := new_coffees * new_price
  original_spending - new_spending = 2 :=
by
  sorry

#check coffee_savings 4 2 (1/2)

end NUMINAMATH_CALUDE_coffee_savings_l1590_159041


namespace NUMINAMATH_CALUDE_s_not_lowest_avg_l1590_159001

-- Define the set of runners
inductive Runner : Type
  | P | Q | R | S | T

-- Define a type for race results
def RaceResult := List Runner

-- Define the first race result
def firstRace : RaceResult := sorry

-- Define the second race result
def secondRace : RaceResult := [Runner.R, Runner.P, Runner.T, Runner.Q, Runner.S]

-- Function to calculate the position of a runner in a race
def position (runner : Runner) (race : RaceResult) : Nat := sorry

-- Function to calculate the average position of a runner across two races
def avgPosition (runner : Runner) (race1 race2 : RaceResult) : Rat :=
  (position runner race1 + position runner race2) / 2

-- Theorem stating that S cannot have the lowest average position
theorem s_not_lowest_avg :
  ∀ (r : Runner), r ≠ Runner.S →
    avgPosition Runner.S firstRace secondRace ≥ avgPosition r firstRace secondRace :=
  sorry

end NUMINAMATH_CALUDE_s_not_lowest_avg_l1590_159001


namespace NUMINAMATH_CALUDE_return_trip_time_l1590_159039

/-- Given a route with uphill and downhill sections, prove the return trip time -/
theorem return_trip_time (total_distance : ℝ) (uphill_speed downhill_speed : ℝ) 
  (time_ab : ℝ) (h1 : total_distance = 21) (h2 : uphill_speed = 4) 
  (h3 : downhill_speed = 6) (h4 : time_ab = 4.25) : ∃ (time_ba : ℝ), time_ba = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_return_trip_time_l1590_159039


namespace NUMINAMATH_CALUDE_area_in_three_triangles_l1590_159069

/-- Given a 6 by 8 rectangle with equilateral triangles on each side, 
    this function calculates the area of regions in exactly 3 of 4 triangles -/
def areaInThreeTriangles : ℝ := sorry

/-- The rectangle's width -/
def rectangleWidth : ℝ := 6

/-- The rectangle's length -/
def rectangleLength : ℝ := 8

/-- Theorem stating the area calculation -/
theorem area_in_three_triangles :
  areaInThreeTriangles = (288 - 154 * Real.sqrt 3) / 3 := by sorry

end NUMINAMATH_CALUDE_area_in_three_triangles_l1590_159069


namespace NUMINAMATH_CALUDE_sum_four_consecutive_divisible_by_two_l1590_159021

theorem sum_four_consecutive_divisible_by_two :
  ∀ n : ℤ, ∃ k : ℤ, (n - 1) + n + (n + 1) + (n + 2) = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_sum_four_consecutive_divisible_by_two_l1590_159021


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l1590_159099

theorem quadratic_root_problem (m : ℝ) : 
  (∃ x : ℝ, 2 * x^2 - m * x + 6 = 0 ∧ x = 1) → 
  (∃ y : ℝ, 2 * y^2 - m * y + 6 = 0 ∧ y = 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l1590_159099


namespace NUMINAMATH_CALUDE_sum_evaluation_l1590_159005

theorem sum_evaluation : 
  (4 : ℚ) / 3 + 7 / 6 + 13 / 12 + 25 / 24 + 49 / 48 + 97 / 96 - 8 = -43 / 32 := by
  sorry

end NUMINAMATH_CALUDE_sum_evaluation_l1590_159005


namespace NUMINAMATH_CALUDE_inequality_proof_l1590_159033

theorem inequality_proof (x y z : ℝ) 
  (sum_zero : x + y + z = 0)
  (abs_sum_le_one : |x| + |y| + |z| ≤ 1) :
  x + y/2 + z/3 ≤ 1/3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1590_159033


namespace NUMINAMATH_CALUDE_product_of_numbers_l1590_159084

theorem product_of_numbers (x y : ℝ) : 
  x + y = 22 → x^2 + y^2 = 460 → x * y = 40 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l1590_159084
