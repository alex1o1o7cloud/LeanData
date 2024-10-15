import Mathlib

namespace NUMINAMATH_CALUDE_chord_length_difference_l420_42021

theorem chord_length_difference (r₁ r₂ : ℝ) (hr₁ : r₁ = 26) (hr₂ : r₂ = 5) :
  let longest_chord := 2 * r₁
  let shortest_chord := 2 * Real.sqrt (r₁^2 - (r₁ - r₂)^2)
  longest_chord - shortest_chord = 52 - 2 * Real.sqrt 235 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_difference_l420_42021


namespace NUMINAMATH_CALUDE_product_digit_sum_l420_42034

theorem product_digit_sum : 
  let a := 2^20
  let b := 5^17
  let product := a * b
  (List.sum (product.digits 10)) = 8 := by sorry

end NUMINAMATH_CALUDE_product_digit_sum_l420_42034


namespace NUMINAMATH_CALUDE_binary_10101_is_21_l420_42024

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_10101_is_21 :
  binary_to_decimal [true, false, true, false, true] = 21 := by
  sorry

end NUMINAMATH_CALUDE_binary_10101_is_21_l420_42024


namespace NUMINAMATH_CALUDE_place_five_in_three_l420_42064

/-- The number of ways to place n distinct objects into k distinct containers -/
def place_objects (n k : ℕ) : ℕ := k^n

/-- Theorem: Placing 5 distinct objects into 3 distinct containers results in 3^5 ways -/
theorem place_five_in_three : place_objects 5 3 = 3^5 := by
  sorry

end NUMINAMATH_CALUDE_place_five_in_three_l420_42064


namespace NUMINAMATH_CALUDE_chebyshev_roots_l420_42015

def T : ℕ → (Real → Real)
  | 0 => λ x => 1
  | 1 => λ x => x
  | (n + 2) => λ x => 2 * x * T (n + 1) x + T n x

theorem chebyshev_roots (n : ℕ) (k : ℕ) (h : 1 ≤ k ∧ k ≤ n) :
  T n (Real.cos ((2 * k - 1 : ℝ) * Real.pi / (2 * n : ℝ))) = 0 := by
  sorry

end NUMINAMATH_CALUDE_chebyshev_roots_l420_42015


namespace NUMINAMATH_CALUDE_quadratic_solution_product_l420_42020

theorem quadratic_solution_product (a b : ℝ) : 
  (3 * a^2 + 9 * a - 21 = 0) → 
  (3 * b^2 + 9 * b - 21 = 0) → 
  (2*a - 5) * (3*b - 4) = 47 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_product_l420_42020


namespace NUMINAMATH_CALUDE_integral_cos_quadratic_l420_42081

theorem integral_cos_quadratic (f : ℝ → ℝ) :
  (∫ x in (0)..(2 * Real.pi), (1 - 8 * x^2) * Real.cos (4 * x)) = -2 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_integral_cos_quadratic_l420_42081


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l420_42070

/-- The quadratic equation (m-1)x^2 - 4x + 1 = 0 has two distinct real roots if and only if m < 5 and m ≠ 1 -/
theorem quadratic_two_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ (m - 1) * x^2 - 4 * x + 1 = 0 ∧ (m - 1) * y^2 - 4 * y + 1 = 0) ↔ 
  (m < 5 ∧ m ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l420_42070


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l420_42045

/-- A point in a 2D plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given two points, checks if they are symmetric with respect to the origin. -/
def symmetricWrtOrigin (p q : Point) : Prop :=
  q.x = -p.x ∧ q.y = -p.y

/-- The theorem stating that the point (-1, 2) is symmetric to (1, -2) with respect to the origin. -/
theorem symmetric_point_coordinates :
  let p : Point := ⟨1, -2⟩
  let q : Point := ⟨-1, 2⟩
  symmetricWrtOrigin p q := by sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l420_42045


namespace NUMINAMATH_CALUDE_calculate_expression_l420_42050

theorem calculate_expression : 
  Real.sqrt 5 * 5^(1/3) + 15 / 5 * 3 - 9^(5/2) = 5^(5/6) - 234 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l420_42050


namespace NUMINAMATH_CALUDE_unique_four_digit_square_l420_42018

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def first_two_digits_equal (n : ℕ) : Prop :=
  (n / 1000) = ((n / 100) % 10)

def last_two_digits_equal (n : ℕ) : Prop :=
  ((n / 10) % 10) = (n % 10)

theorem unique_four_digit_square :
  ∃! n : ℕ, is_four_digit n ∧
             ∃ k : ℕ, n = k^2 ∧
             first_two_digits_equal n ∧
             last_two_digits_equal n ∧
             n = 7744 :=
by sorry

end NUMINAMATH_CALUDE_unique_four_digit_square_l420_42018


namespace NUMINAMATH_CALUDE_complex_number_problem_l420_42054

theorem complex_number_problem (z : ℂ) 
  (h1 : z.re > 0) 
  (h2 : Complex.abs z = 2 * Real.sqrt 5) 
  (h3 : (Complex.I + 2) * z = Complex.I * (Complex.I * z).im) 
  (h4 : ∃ (m n : ℝ), z^2 + m*z + n = 0) : 
  z = 4 + 2*Complex.I ∧ 
  ∃ (m n : ℝ), z^2 + m*z + n = 0 ∧ m = -8 ∧ n = 20 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l420_42054


namespace NUMINAMATH_CALUDE_pentagon_angle_measure_l420_42069

theorem pentagon_angle_measure (P Q R S T : ℝ) : 
  -- Pentagon condition
  P + Q + R + S + T = 540 →
  -- Equal angles condition
  P = R ∧ P = T →
  -- Supplementary angles condition
  Q + S = 180 →
  -- Conclusion
  T = 120 := by
sorry

end NUMINAMATH_CALUDE_pentagon_angle_measure_l420_42069


namespace NUMINAMATH_CALUDE_chair_color_probability_l420_42093

/-- The probability that the last two remaining chairs are of the same color -/
def same_color_probability (black_chairs brown_chairs : ℕ) : ℚ :=
  let total_chairs := black_chairs + brown_chairs
  let black_prob := (black_chairs : ℚ) / total_chairs * ((black_chairs - 1) : ℚ) / (total_chairs - 1)
  let brown_prob := (brown_chairs : ℚ) / total_chairs * ((brown_chairs - 1) : ℚ) / (total_chairs - 1)
  black_prob + brown_prob

/-- Theorem stating that the probability of the last two chairs being the same color is 43/88 -/
theorem chair_color_probability :
  same_color_probability 15 18 = 43 / 88 := by
  sorry

end NUMINAMATH_CALUDE_chair_color_probability_l420_42093


namespace NUMINAMATH_CALUDE_keith_total_cost_l420_42062

def rabbit_toy_cost : ℚ := 6.51
def pet_food_cost : ℚ := 5.79
def cage_cost : ℚ := 12.51
def found_money : ℚ := 1.00

theorem keith_total_cost :
  rabbit_toy_cost + pet_food_cost + cage_cost - found_money = 23.81 := by
  sorry

end NUMINAMATH_CALUDE_keith_total_cost_l420_42062


namespace NUMINAMATH_CALUDE_triple_solution_l420_42087

theorem triple_solution (a b c : ℝ) : 
  a^2 + b^2 + c^2 = 1 ∧ a * (2*b - 2*a - c) ≥ 1/2 →
  ((a = 1/Real.sqrt 6 ∧ b = 2/Real.sqrt 6 ∧ c = -1/Real.sqrt 6) ∨
   (a = -1/Real.sqrt 6 ∧ b = -2/Real.sqrt 6 ∧ c = 1/Real.sqrt 6)) :=
by sorry

end NUMINAMATH_CALUDE_triple_solution_l420_42087


namespace NUMINAMATH_CALUDE_folded_rectangle_area_l420_42023

/-- Given a rectangle with dimensions 5 by 8, when folded to form a trapezoid
    where corners touch, the area of the resulting trapezoid is 55/2. -/
theorem folded_rectangle_area (rect_width : ℝ) (rect_length : ℝ) 
    (h_width : rect_width = 5)
    (h_length : rect_length = 8)
    (trapezoid_short_base : ℝ)
    (h_short_base : trapezoid_short_base = 3)
    (trapezoid_long_base : ℝ)
    (h_long_base : trapezoid_long_base = rect_length)
    (trapezoid_height : ℝ)
    (h_height : trapezoid_height = rect_width) : 
  (trapezoid_short_base + trapezoid_long_base) * trapezoid_height / 2 = 55 / 2 := by
  sorry

end NUMINAMATH_CALUDE_folded_rectangle_area_l420_42023


namespace NUMINAMATH_CALUDE_heart_ratio_equals_one_l420_42042

-- Define the ♥ operation
def heart (n m : ℕ) : ℕ := n^3 * m^3

-- Theorem statement
theorem heart_ratio_equals_one : (heart 3 2) / (heart 2 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_heart_ratio_equals_one_l420_42042


namespace NUMINAMATH_CALUDE_road_trip_distance_l420_42058

theorem road_trip_distance (first_day : ℝ) (second_day : ℝ) (third_day : ℝ) : 
  first_day = 200 →
  second_day = 3/4 * first_day →
  third_day = 1/2 * (first_day + second_day) →
  first_day + second_day + third_day = 525 := by
sorry

end NUMINAMATH_CALUDE_road_trip_distance_l420_42058


namespace NUMINAMATH_CALUDE_reciprocal_inequality_l420_42079

theorem reciprocal_inequality (a b : ℝ) (h1 : 0 < a) (h2 : a < b) : 1 / a < 1 / b := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_inequality_l420_42079


namespace NUMINAMATH_CALUDE_find_a_over_b_l420_42061

-- Define the region
def region (x y : ℝ) : Prop :=
  x ≥ 1 ∧ x + y ≤ 4 ∧ ∃ a b : ℝ, a * x + b * y + 2 ≥ 0

-- Define the objective function
def z (x y : ℝ) : ℝ := 2 * x + y

-- State the theorem
theorem find_a_over_b :
  ∃ a b : ℝ,
    (∀ x y : ℝ, region x y → z x y ≤ 7) ∧
    (∃ x y : ℝ, region x y ∧ z x y = 7) ∧
    (∀ x y : ℝ, region x y → z x y ≥ 1) ∧
    (∃ x y : ℝ, region x y ∧ z x y = 1) ∧
    a / b = -1 :=
sorry

end NUMINAMATH_CALUDE_find_a_over_b_l420_42061


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l420_42077

theorem arithmetic_expression_equality : 3^2 + 4 * 2 - 6 / 3 + 7 = 22 := by sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l420_42077


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_l420_42017

theorem quadratic_roots_difference (a b : ℝ) : 
  (2 : ℝ) ∈ {x : ℝ | x^2 - (a+1)*x + a = 0} ∧ 
  b ∈ {x : ℝ | x^2 - (a+1)*x + a = 0} → 
  a - b = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_l420_42017


namespace NUMINAMATH_CALUDE_square_side_length_difference_l420_42083

/-- Given four squares with known side length differences, prove that the total difference
    in side length from the largest to the smallest square is the sum of these differences. -/
theorem square_side_length_difference (AB CD FE : ℝ) (hAB : AB = 11) (hCD : CD = 5) (hFE : FE = 13) :
  ∃ (GH : ℝ), GH = AB + CD + FE :=
by sorry

end NUMINAMATH_CALUDE_square_side_length_difference_l420_42083


namespace NUMINAMATH_CALUDE_quadratic_roots_expression_l420_42075

theorem quadratic_roots_expression (a b : ℝ) : 
  a^2 + a - 3 = 0 → b^2 + b - 3 = 0 → 4 * b^2 - a^3 = (53 + 8 * Real.sqrt 13) / 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_expression_l420_42075


namespace NUMINAMATH_CALUDE_average_rainfall_proof_l420_42078

/-- The average rainfall for the first three days of May in a normal year -/
def average_rainfall : ℝ := 140

/-- Rainfall on the first day in cm -/
def first_day_rainfall : ℝ := 26

/-- Rainfall on the second day in cm -/
def second_day_rainfall : ℝ := 34

/-- Rainfall difference between second and third day in cm -/
def third_day_difference : ℝ := 12

/-- Difference between this year's total rainfall and average in cm -/
def rainfall_difference : ℝ := 58

theorem average_rainfall_proof :
  let third_day_rainfall := second_day_rainfall - third_day_difference
  let this_year_total := first_day_rainfall + second_day_rainfall + third_day_rainfall
  average_rainfall = this_year_total + rainfall_difference := by
  sorry

end NUMINAMATH_CALUDE_average_rainfall_proof_l420_42078


namespace NUMINAMATH_CALUDE_hamans_dropped_trays_l420_42011

theorem hamans_dropped_trays 
  (initial_trays : ℕ) 
  (additional_trays : ℕ) 
  (total_eggs_sold : ℕ) 
  (eggs_per_tray : ℕ) 
  (h1 : initial_trays = 10)
  (h2 : additional_trays = 7)
  (h3 : total_eggs_sold = 540)
  (h4 : eggs_per_tray = 30) :
  initial_trays + additional_trays + 1 - (total_eggs_sold / eggs_per_tray) = 8 :=
by sorry

end NUMINAMATH_CALUDE_hamans_dropped_trays_l420_42011


namespace NUMINAMATH_CALUDE_parallelogram_roots_l420_42053

def polynomial (a : ℝ) (z : ℂ) : ℂ :=
  z^4 - 6*z^3 + 11*a*z^2 - 3*(2*a^2 + 3*a - 3)*z + 1

def forms_parallelogram (roots : List ℂ) : Prop :=
  ∃ (w₁ w₂ : ℂ), roots = [w₁, -w₁, w₂, -w₂]

theorem parallelogram_roots (a : ℝ) :
  (∃ (roots : List ℂ), (∀ z ∈ roots, polynomial a z = 0) ∧
                       roots.length = 4 ∧
                       forms_parallelogram roots) ↔ a = 3 :=
sorry

end NUMINAMATH_CALUDE_parallelogram_roots_l420_42053


namespace NUMINAMATH_CALUDE_chromium_percentage_proof_l420_42039

/-- The percentage of chromium in the second alloy -/
def chromium_percentage_second_alloy : ℝ := 8

theorem chromium_percentage_proof (
  first_alloy_chromium_percentage : ℝ)
  (first_alloy_weight : ℝ)
  (second_alloy_weight : ℝ)
  (new_alloy_chromium_percentage : ℝ)
  (h1 : first_alloy_chromium_percentage = 15)
  (h2 : first_alloy_weight = 15)
  (h3 : second_alloy_weight = 35)
  (h4 : new_alloy_chromium_percentage = 10.1)
  : chromium_percentage_second_alloy = 8 := by
  sorry

#check chromium_percentage_proof

end NUMINAMATH_CALUDE_chromium_percentage_proof_l420_42039


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l420_42044

theorem simplify_and_evaluate (a : ℝ) (h : a = -Real.sqrt 2) :
  (a - 3) / a * 6 / (a^2 - 6*a + 9) - (2*a + 6) / (a^2 - 9) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l420_42044


namespace NUMINAMATH_CALUDE_otimes_properties_l420_42035

def otimes (a b : ℝ) : ℝ := a * (1 - b)

theorem otimes_properties : 
  (otimes 2 (-2) = 6) ∧ 
  (¬ ∀ a b, otimes a b = otimes b a) ∧ 
  (∀ a, otimes 5 a + otimes 6 a = otimes 11 a) ∧ 
  (¬ ∀ b, otimes 3 b = 3 → b = 1) := by sorry

end NUMINAMATH_CALUDE_otimes_properties_l420_42035


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l420_42047

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, a * x^2 - 4 * x + a - 3 < 0) ↔ a < 4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l420_42047


namespace NUMINAMATH_CALUDE_sum_odd_integers_less_than_100_l420_42065

theorem sum_odd_integers_less_than_100 : 
  (Finset.filter (fun n => n % 2 = 1) (Finset.range 100)).sum id = 2500 := by
  sorry

end NUMINAMATH_CALUDE_sum_odd_integers_less_than_100_l420_42065


namespace NUMINAMATH_CALUDE_items_left_in_store_l420_42073

/-- Given the number of items ordered, sold, and in the storeroom, 
    calculate the total number of items left in the whole store. -/
theorem items_left_in_store 
  (items_ordered : ℕ) 
  (items_sold : ℕ) 
  (items_in_storeroom : ℕ) 
  (h1 : items_ordered = 4458)
  (h2 : items_sold = 1561)
  (h3 : items_in_storeroom = 575) :
  items_ordered - items_sold + items_in_storeroom = 3472 :=
by sorry

end NUMINAMATH_CALUDE_items_left_in_store_l420_42073


namespace NUMINAMATH_CALUDE_negation_equivalence_l420_42043

theorem negation_equivalence : 
  (¬∃ x : ℝ, x ≤ -1 ∨ x ≥ 2) ↔ (∀ x : ℝ, -1 < x ∧ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l420_42043


namespace NUMINAMATH_CALUDE_max_area_CDFE_l420_42048

/-- A square with side length 1 -/
structure Square :=
  (A B C D : ℝ × ℝ)
  (is_square : A.1 = 0 ∧ A.2 = 0 ∧ B.1 = 1 ∧ B.2 = 0 ∧ C.1 = 1 ∧ C.2 = 1 ∧ D.1 = 0 ∧ D.2 = 1)

/-- Points E and F on sides AB and AD respectively -/
def PointsEF (s : Square) (x : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((x, 0), (0, x))

/-- Area of quadrilateral CDFE -/
def AreaCDFE (s : Square) (x : ℝ) : ℝ :=
  x * (1 - x)

/-- The maximum area of quadrilateral CDFE is 1/4 -/
theorem max_area_CDFE (s : Square) :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧
  ∀ y : ℝ, 0 ≤ y ∧ y ≤ 1 → AreaCDFE s y ≤ AreaCDFE s x ∧
  AreaCDFE s x = 1/4 :=
sorry

end NUMINAMATH_CALUDE_max_area_CDFE_l420_42048


namespace NUMINAMATH_CALUDE_probability_three_out_of_ten_l420_42022

/-- The probability of selecting at least one defective item from a set of products -/
def probability_at_least_one_defective (total : ℕ) (defective : ℕ) (selected : ℕ) : ℚ :=
  1 - (Nat.choose (total - defective) selected : ℚ) / (Nat.choose total selected : ℚ)

/-- Theorem stating the probability of selecting at least one defective item
    when 3 out of 10 items are defective and 3 items are randomly selected -/
theorem probability_three_out_of_ten :
  probability_at_least_one_defective 10 3 3 = 17 / 24 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_out_of_ten_l420_42022


namespace NUMINAMATH_CALUDE_ten_machines_four_minutes_production_l420_42089

/-- The number of bottles produced per minute by a single machine -/
def bottles_per_machine_per_minute (total_bottles : ℕ) (num_machines : ℕ) : ℕ :=
  total_bottles / num_machines

/-- The number of bottles produced per minute by a given number of machines -/
def bottles_per_minute (bottles_per_machine : ℕ) (num_machines : ℕ) : ℕ :=
  bottles_per_machine * num_machines

/-- The total number of bottles produced in a given time -/
def total_bottles (bottles_per_minute : ℕ) (minutes : ℕ) : ℕ :=
  bottles_per_minute * minutes

/-- Theorem stating that 10 machines will produce 1800 bottles in 4 minutes -/
theorem ten_machines_four_minutes_production 
  (h : bottles_per_minute (bottles_per_machine_per_minute 270 6) 10 = 450) :
  total_bottles 450 4 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_ten_machines_four_minutes_production_l420_42089


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l420_42038

/-- Given an arithmetic sequence with non-zero common difference, 
    if a_2 + a_3 = a_6, then (a_1 + a_2) / (a_3 + a_4 + a_5) = 1/3 -/
theorem arithmetic_sequence_ratio (a : ℕ → ℚ) (d : ℚ) (h1 : d ≠ 0) 
  (h2 : ∀ n, a (n + 1) = a n + d) 
  (h3 : a 2 + a 3 = a 6) : 
  (a 1 + a 2) / (a 3 + a 4 + a 5) = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l420_42038


namespace NUMINAMATH_CALUDE_find_constant_b_l420_42091

theorem find_constant_b (a b c : ℚ) : 
  (∀ x : ℚ, (4 * x^3 - 3 * x + 7/2) * (a * x^2 + b * x + c) = 
    12 * x^5 - 14 * x^4 + 18 * x^3 - (23/3) * x^2 + (14/2) * x - 3) →
  b = -7/2 := by
sorry

end NUMINAMATH_CALUDE_find_constant_b_l420_42091


namespace NUMINAMATH_CALUDE_least_three_digit_with_digit_product_12_l420_42031

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

theorem least_three_digit_with_digit_product_12 :
  ∀ n : ℕ, is_three_digit n → digit_product n = 12 → 126 ≤ n :=
sorry

end NUMINAMATH_CALUDE_least_three_digit_with_digit_product_12_l420_42031


namespace NUMINAMATH_CALUDE_circle_area_l420_42086

theorem circle_area (r : ℝ) (h : 2 * π * r = 18 * π) : π * r^2 = 81 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_l420_42086


namespace NUMINAMATH_CALUDE_right_triangle_construction_condition_l420_42067

/-- Given a right triangle ABC with leg AC = b and perimeter 2s, 
    prove that the construction is possible if and only if b < s -/
theorem right_triangle_construction_condition 
  (b s : ℝ) 
  (h_positive_b : 0 < b) 
  (h_positive_s : 0 < s) 
  (h_perimeter : ∃ (c : ℝ), b + c + (b^2 + c^2).sqrt = 2*s) :
  (∃ (c : ℝ), c > 0 ∧ b^2 + c^2 = ((2*s - b - c)^2)) ↔ b < s :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_construction_condition_l420_42067


namespace NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l420_42059

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

/-- The 15th term of the specific arithmetic sequence -/
theorem fifteenth_term_of_sequence : arithmetic_sequence (-3) 4 15 = 53 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l420_42059


namespace NUMINAMATH_CALUDE_recommended_intake_proof_l420_42049

/-- Recommended intake of added sugar for men per day (in calories) -/
def recommended_intake : ℝ := 150

/-- Calories in the soft drink -/
def soft_drink_calories : ℝ := 2500

/-- Percentage of calories from added sugar in the soft drink -/
def soft_drink_sugar_percentage : ℝ := 0.05

/-- Calories of added sugar in each candy bar -/
def candy_bar_sugar_calories : ℝ := 25

/-- Number of candy bars consumed -/
def candy_bars_consumed : ℕ := 7

/-- Percentage by which Mark exceeded the recommended intake -/
def excess_percentage : ℝ := 1

theorem recommended_intake_proof :
  let soft_drink_sugar := soft_drink_calories * soft_drink_sugar_percentage
  let candy_sugar := candy_bar_sugar_calories * candy_bars_consumed
  let total_sugar := soft_drink_sugar + candy_sugar
  total_sugar = recommended_intake * (1 + excess_percentage) :=
by sorry

end NUMINAMATH_CALUDE_recommended_intake_proof_l420_42049


namespace NUMINAMATH_CALUDE_digit_sequence_bound_l420_42002

/-- Given a positive integer N with n digits, if all its digits are distinct
    and the sum of any three consecutive digits is divisible by 5, then n ≤ 6. -/
theorem digit_sequence_bound (N : ℕ) (n : ℕ) : 
  (N ≥ 10^(n-1) ∧ N < 10^n) →  -- N is an n-digit number
  (∀ i j, i ≠ j → (N / 10^i) % 10 ≠ (N / 10^j) % 10) →  -- All digits are distinct
  (∀ i, i + 2 < n → ((N / 10^i) % 10 + (N / 10^(i+1)) % 10 + (N / 10^(i+2)) % 10) % 5 = 0) →  -- Sum of any three consecutive digits is divisible by 5
  n ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_digit_sequence_bound_l420_42002


namespace NUMINAMATH_CALUDE_even_periodic_function_monotonicity_l420_42063

def isEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def hasPeriod (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def isIncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

def isDecreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y

theorem even_periodic_function_monotonicity 
  (f : ℝ → ℝ) 
  (h_even : isEven f) 
  (h_period : hasPeriod f 2) : 
  isIncreasingOn f 0 1 ↔ isDecreasingOn f 3 4 := by
  sorry

end NUMINAMATH_CALUDE_even_periodic_function_monotonicity_l420_42063


namespace NUMINAMATH_CALUDE_discount_difference_is_399_l420_42003

def initial_price : ℝ := 8000

def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def option1_discounts : List ℝ := [0.25, 0.15, 0.05]
def option2_discounts : List ℝ := [0.35, 0.10, 0.05]

def apply_discounts (price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount price

theorem discount_difference_is_399 :
  apply_discounts initial_price option1_discounts -
  apply_discounts initial_price option2_discounts = 399 := by
  sorry

end NUMINAMATH_CALUDE_discount_difference_is_399_l420_42003


namespace NUMINAMATH_CALUDE_lemonade_pitchers_sum_l420_42007

theorem lemonade_pitchers_sum : 
  let first_intermission : ℝ := 0.25
  let second_intermission : ℝ := 0.42
  let third_intermission : ℝ := 0.25
  first_intermission + second_intermission + third_intermission = 0.92 := by
sorry

end NUMINAMATH_CALUDE_lemonade_pitchers_sum_l420_42007


namespace NUMINAMATH_CALUDE_stating_regions_in_polygon_formula_l420_42099

/-- 
Given a convex n-sided polygon where all diagonals are drawn and no three diagonals pass through a point,
this function calculates the number of regions formed inside the polygon.
-/
def regions_in_polygon (n : ℕ) : ℕ :=
  1 + (n.choose 2) - n + (n.choose 4)

/-- 
Theorem stating that the number of regions formed inside a convex n-sided polygon
with all diagonals drawn and no three diagonals passing through a point
is equal to 1 + (n choose 2) - n + (n choose 4).
-/
theorem regions_in_polygon_formula (n : ℕ) (h : n ≥ 3) :
  regions_in_polygon n = 1 + (n.choose 2) - n + (n.choose 4) :=
by sorry

end NUMINAMATH_CALUDE_stating_regions_in_polygon_formula_l420_42099


namespace NUMINAMATH_CALUDE_divisible_by_45_sum_of_digits_l420_42094

theorem divisible_by_45_sum_of_digits (a b : ℕ) : 
  (a < 10) →
  (b < 10) →
  (6 * 10000 + a * 1000 + 700 + 80 + b) % 45 = 0 →
  a + b = 6 := by sorry

end NUMINAMATH_CALUDE_divisible_by_45_sum_of_digits_l420_42094


namespace NUMINAMATH_CALUDE_z_purely_imaginary_iff_z_in_second_quadrant_iff_l420_42046

def z (m : ℝ) : ℂ := (2 * m^2 - 3 * m - 2) + (m^2 - 3 * m + 2) * Complex.I

theorem z_purely_imaginary_iff (m : ℝ) : 
  z m = Complex.I * Complex.im (z m) ↔ m = -1/2 := by sorry

theorem z_in_second_quadrant_iff (m : ℝ) :
  (Complex.re (z m) < 0 ∧ Complex.im (z m) > 0) ↔ -1/2 < m ∧ m < 1 := by sorry

end NUMINAMATH_CALUDE_z_purely_imaginary_iff_z_in_second_quadrant_iff_l420_42046


namespace NUMINAMATH_CALUDE_fourth_rectangle_area_l420_42057

/-- Represents a rectangle divided into four smaller rectangles -/
structure DividedRectangle where
  total_area : ℝ
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ
  area4 : ℝ
  vertical_halves : area1 + area4 = area2 + area3
  sum_of_areas : total_area = area1 + area2 + area3 + area4

/-- Theorem stating that given the areas of three rectangles in a divided rectangle,
    we can determine the area of the fourth rectangle -/
theorem fourth_rectangle_area
  (rect : DividedRectangle)
  (h1 : rect.area1 = 12)
  (h2 : rect.area2 = 27)
  (h3 : rect.area3 = 18) :
  rect.area4 = 27 := by
  sorry

#check fourth_rectangle_area

end NUMINAMATH_CALUDE_fourth_rectangle_area_l420_42057


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l420_42052

theorem smallest_prime_divisor_of_sum (n : ℕ) :
  (∃ p : ℕ, Nat.Prime p ∧ p ∣ (3^15 + 11^13)) →
  2 ≤ Nat.minFac (3^15 + 11^13) ∧ Nat.minFac (3^15 + 11^13) = 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l420_42052


namespace NUMINAMATH_CALUDE_tenth_term_of_arithmetic_sequence_l420_42092

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1) * d

theorem tenth_term_of_arithmetic_sequence 
  (a₁ : ℚ) (a₂₀ : ℚ) (h₁ : a₁ = 5/11) (h₂₀ : a₂₀ = 9/11)
  (h_seq : ∀ n, arithmetic_sequence a₁ ((a₂₀ - a₁) / 19) n = 
    arithmetic_sequence (5/11) ((9/11 - 5/11) / 19) n) :
  arithmetic_sequence a₁ ((a₂₀ - a₁) / 19) 10 = 1233/2309 :=
by sorry

end NUMINAMATH_CALUDE_tenth_term_of_arithmetic_sequence_l420_42092


namespace NUMINAMATH_CALUDE_stone_151_is_9_l420_42068

/-- Represents the number of stones in the arrangement. -/
def num_stones : ℕ := 12

/-- Represents the modulus for the counting pattern. -/
def counting_modulus : ℕ := 22

/-- The number we want to find the original stone for. -/
def target_count : ℕ := 151

/-- Function to determine the original stone number given a count. -/
def original_stone (count : ℕ) : ℕ :=
  (count - 1) % counting_modulus + 1

theorem stone_151_is_9 : original_stone target_count = 9 := by
  sorry

end NUMINAMATH_CALUDE_stone_151_is_9_l420_42068


namespace NUMINAMATH_CALUDE_sum_x_y_equals_two_l420_42051

-- Define the function f(t) = t^3 + 2003t
def f (t : ℝ) : ℝ := t^3 + 2003*t

-- State the theorem
theorem sum_x_y_equals_two (x y : ℝ) 
  (hx : f (x - 1) = -1) 
  (hy : f (y - 1) = 1) : 
  x + y = 2 := by
  sorry


end NUMINAMATH_CALUDE_sum_x_y_equals_two_l420_42051


namespace NUMINAMATH_CALUDE_pyramid_volume_l420_42040

/-- The volume of a pyramid with a regular hexagonal base and specific triangle areas -/
theorem pyramid_volume (base_area : ℝ) (triangle_ABG_area : ℝ) (triangle_DEG_area : ℝ)
  (h_base : base_area = 648)
  (h_ABG : triangle_ABG_area = 180)
  (h_DEG : triangle_DEG_area = 162) :
  ∃ (volume : ℝ), volume = 432 * Real.sqrt 22 := by
  sorry

#check pyramid_volume

end NUMINAMATH_CALUDE_pyramid_volume_l420_42040


namespace NUMINAMATH_CALUDE_apartment_ages_puzzle_l420_42032

def is_valid_triplet (a b c : ℕ) : Prop :=
  a * b * c = 1296 ∧ a < 100 ∧ b < 100 ∧ c < 100

def has_duplicate_sum (triplets : List (ℕ × ℕ × ℕ)) : Prop :=
  ∃ (t1 t2 : ℕ × ℕ × ℕ), t1 ∈ triplets ∧ t2 ∈ triplets ∧ t1 ≠ t2 ∧ 
    t1.1 + t1.2.1 + t1.2.2 = t2.1 + t2.2.1 + t2.2.2

theorem apartment_ages_puzzle :
  ∃! (a b c : ℕ), 
    is_valid_triplet a b c ∧
    (∀ triplets : List (ℕ × ℕ × ℕ), (∀ (x y z : ℕ), (x, y, z) ∈ triplets → is_valid_triplet x y z) →
      has_duplicate_sum triplets → (a, b, c) ∈ triplets) ∧
    a < b ∧ b < c ∧ c < 100 ∧
    a + b + c = 91 :=
by sorry

end NUMINAMATH_CALUDE_apartment_ages_puzzle_l420_42032


namespace NUMINAMATH_CALUDE_range_of_f_on_I_l420_42060

-- Define the function
def f (x : ℝ) : ℝ := x^2 + x

-- Define the interval
def I : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

-- State the theorem
theorem range_of_f_on_I :
  {y | ∃ x ∈ I, f x = y} = {y | -1/4 ≤ y ∧ y ≤ 12} := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_on_I_l420_42060


namespace NUMINAMATH_CALUDE_red_triangles_in_colored_graph_l420_42080

/-- A coloring of a complete graph is a function that assigns either red or blue to each edge. -/
def Coloring (n : ℕ) := Fin n → Fin n → Bool

/-- The set of vertices connected to a given vertex by red edges. -/
def RedNeighborhood (n : ℕ) (c : Coloring n) (v : Fin n) : Finset (Fin n) :=
  Finset.filter (fun u => c v u) (Finset.univ.erase v)

/-- A red triangle in a colored complete graph. -/
def RedTriangle (n : ℕ) (c : Coloring n) (v1 v2 v3 : Fin n) : Prop :=
  v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 ∧ c v1 v2 ∧ c v2 v3 ∧ c v1 v3

theorem red_triangles_in_colored_graph (k : ℕ) (h : k ≥ 3) :
  ∀ (c : Coloring (3*k+2)),
  (∀ v, (RedNeighborhood (3*k+2) c v).card ≥ k+2) →
  (∀ v w, ¬c v w → (RedNeighborhood (3*k+2) c v ∪ RedNeighborhood (3*k+2) c w).card ≥ 2*k+2) →
  ∃ (S : Finset (Fin (3*k+2) × Fin (3*k+2) × Fin (3*k+2))),
    S.card ≥ k+2 ∧ ∀ (t : Fin (3*k+2) × Fin (3*k+2) × Fin (3*k+2)), t ∈ S → RedTriangle (3*k+2) c t.1 t.2.1 t.2.2 :=
by sorry

end NUMINAMATH_CALUDE_red_triangles_in_colored_graph_l420_42080


namespace NUMINAMATH_CALUDE_lilac_paint_mixture_l420_42055

/-- Given a paint mixture where 70% is blue, 20% is red, and the rest is white,
    if 140 ounces of blue paint is added, then 20 ounces of white paint is added. -/
theorem lilac_paint_mixture (blue_percent : ℝ) (red_percent : ℝ) (blue_amount : ℝ) : 
  blue_percent = 0.7 →
  red_percent = 0.2 →
  blue_amount = 140 →
  let total_amount := blue_amount / blue_percent
  let white_percent := 1 - blue_percent - red_percent
  let white_amount := total_amount * white_percent
  white_amount = 20 := by
  sorry

end NUMINAMATH_CALUDE_lilac_paint_mixture_l420_42055


namespace NUMINAMATH_CALUDE_largest_n_divisibility_l420_42006

theorem largest_n_divisibility : 
  ∀ n : ℕ, n > 1221 → ¬(∃ k : ℕ, n^3 + 99 = k * (n + 11)) ∧ 
  ∃ k : ℕ, 1221^3 + 99 = k * (1221 + 11) :=
sorry

end NUMINAMATH_CALUDE_largest_n_divisibility_l420_42006


namespace NUMINAMATH_CALUDE_lg_expression_equals_two_l420_42071

-- Define lg as the base-10 logarithm
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem lg_expression_equals_two :
  (lg 5)^2 + lg 2 * lg 50 = 2 := by
  sorry

end NUMINAMATH_CALUDE_lg_expression_equals_two_l420_42071


namespace NUMINAMATH_CALUDE_product_binary1011_ternary212_eq_253_l420_42027

/-- Converts a list of digits in a given base to its decimal representation -/
def toDecimal (digits : List Nat) (base : Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * base^i) 0

/-- The binary representation of 1011 -/
def binary1011 : List Nat := [1, 0, 1, 1]

/-- The base-3 representation of 212 -/
def ternary212 : List Nat := [2, 1, 2]

theorem product_binary1011_ternary212_eq_253 :
  (toDecimal binary1011 2) * (toDecimal ternary212 3) = 253 := by
  sorry

end NUMINAMATH_CALUDE_product_binary1011_ternary212_eq_253_l420_42027


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l420_42082

/-- An isosceles triangle with congruent sides of 7 cm and perimeter of 20 cm has a base of 6 cm. -/
theorem isosceles_triangle_base_length :
  ∀ (base congruent_side : ℝ),
  congruent_side = 7 →
  base + 2 * congruent_side = 20 →
  base = 6 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l420_42082


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l420_42088

theorem triangle_abc_properties (A B C : Real) (h : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Given conditions
  A + B = 3 * C →
  2 * Real.sin (A - C) = Real.sin B →
  -- AB = 5 (implicitly used in the height calculation)
  -- Prove sin A and height h
  Real.sin A = 3 * (10 : Real).sqrt / 10 ∧ h = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l420_42088


namespace NUMINAMATH_CALUDE_dirt_bike_cost_l420_42097

/-- Proves that the cost of each dirt bike is $150 given the problem conditions -/
theorem dirt_bike_cost :
  ∀ (dirt_bike_cost : ℕ),
  (3 * dirt_bike_cost + 4 * 300 + 7 * 25 = 1825) →
  dirt_bike_cost = 150 :=
by
  sorry

#check dirt_bike_cost

end NUMINAMATH_CALUDE_dirt_bike_cost_l420_42097


namespace NUMINAMATH_CALUDE_white_balls_count_l420_42072

/-- Calculates the number of white balls in a bag given specific conditions -/
theorem white_balls_count (total : ℕ) (green yellow red purple : ℕ) (prob_not_red_purple : ℚ) : 
  total = 60 ∧ 
  green = 18 ∧ 
  yellow = 5 ∧ 
  red = 6 ∧ 
  purple = 9 ∧ 
  prob_not_red_purple = 3/4 → 
  total - (green + yellow + red + purple) = 22 := by
sorry

end NUMINAMATH_CALUDE_white_balls_count_l420_42072


namespace NUMINAMATH_CALUDE_initial_books_count_l420_42033

theorem initial_books_count (initial_books sold_books given_books remaining_books : ℕ) :
  sold_books = 11 →
  given_books = 35 →
  remaining_books = 62 →
  initial_books = sold_books + given_books + remaining_books →
  initial_books = 108 := by
  sorry

end NUMINAMATH_CALUDE_initial_books_count_l420_42033


namespace NUMINAMATH_CALUDE_profit_percentage_specific_l420_42076

/-- The profit percentage after markup and discount -/
def profit_percentage (markup : ℝ) (discount : ℝ) : ℝ :=
  let marked_price := 1 + markup
  let selling_price := marked_price * (1 - discount)
  (selling_price - 1) * 100

/-- Theorem: Given a 60% markup and a 25% discount, the profit percentage is 20% -/
theorem profit_percentage_specific : profit_percentage 0.6 0.25 = 20 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_specific_l420_42076


namespace NUMINAMATH_CALUDE_largest_among_rationals_l420_42019

theorem largest_among_rationals : 
  let numbers : List ℚ := [-2/3, -2, -1, -5]
  (∀ x ∈ numbers, x ≤ -2/3) ∧ (-2/3 ∈ numbers) := by
  sorry

end NUMINAMATH_CALUDE_largest_among_rationals_l420_42019


namespace NUMINAMATH_CALUDE_can_transport_goods_l420_42000

/-- Represents the total weight of goods in tonnes -/
def total_weight : ℝ := 13.5

/-- Represents the maximum weight of goods in a single box in tonnes -/
def max_box_weight : ℝ := 0.35

/-- Represents the number of available trucks -/
def num_trucks : ℕ := 11

/-- Represents the load capacity of each truck in tonnes -/
def truck_capacity : ℝ := 1.5

/-- Theorem stating that the given number of trucks can transport all goods in a single trip -/
theorem can_transport_goods : 
  (num_trucks : ℝ) * truck_capacity ≥ total_weight := by sorry

end NUMINAMATH_CALUDE_can_transport_goods_l420_42000


namespace NUMINAMATH_CALUDE_x_sum_less_than_2m_l420_42030

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + 5 - a / Real.exp x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.exp x * f a x

theorem x_sum_less_than_2m (a : ℝ) (m x₁ x₂ : ℝ) 
  (h1 : m ≥ 1) 
  (h2 : x₁ < m) 
  (h3 : x₂ > m) 
  (h4 : g a x₁ + g a x₂ = 2 * g a m) : 
  x₁ + x₂ < 2 * m := by
  sorry

end NUMINAMATH_CALUDE_x_sum_less_than_2m_l420_42030


namespace NUMINAMATH_CALUDE_factorization_2x_cubed_minus_8x_l420_42029

theorem factorization_2x_cubed_minus_8x (x : ℝ) : 
  2 * x^3 - 8 * x = 2 * x * (x - 2) * (x + 2) := by
sorry


end NUMINAMATH_CALUDE_factorization_2x_cubed_minus_8x_l420_42029


namespace NUMINAMATH_CALUDE_toy_spending_ratio_l420_42005

theorem toy_spending_ratio (initial_amount : ℕ) (toy_cost : ℕ) (final_amount : ℕ) :
  initial_amount = 204 →
  final_amount = 51 →
  toy_cost + (initial_amount - toy_cost) / 2 + final_amount = initial_amount →
  toy_cost * 2 = initial_amount :=
by sorry

end NUMINAMATH_CALUDE_toy_spending_ratio_l420_42005


namespace NUMINAMATH_CALUDE_manuscript_typing_cost_l420_42085

/-- The cost per page for the first typing of a manuscript -/
def first_typing_cost (total_pages : ℕ) (pages_revised_once : ℕ) (pages_revised_twice : ℕ) 
  (revision_cost : ℚ) (total_cost : ℚ) : ℚ :=
  (total_cost - revision_cost * (pages_revised_once + 2 * pages_revised_twice)) / total_pages

theorem manuscript_typing_cost :
  first_typing_cost 200 80 20 3 1360 = 5 := by
  sorry

end NUMINAMATH_CALUDE_manuscript_typing_cost_l420_42085


namespace NUMINAMATH_CALUDE_line_y_coordinate_at_x_10_l420_42013

/-- Given a line passing through points (4, 0) and (-2, -3),
    prove that the y-coordinate of the point on this line with x-coordinate 10 is 3. -/
theorem line_y_coordinate_at_x_10 :
  let m : ℚ := (0 - (-3)) / (4 - (-2))  -- Slope of the line
  let b : ℚ := 0 - m * 4                -- y-intercept of the line
  m * 10 + b = 3 := by sorry

end NUMINAMATH_CALUDE_line_y_coordinate_at_x_10_l420_42013


namespace NUMINAMATH_CALUDE_borrowing_interest_rate_l420_42036

/-- Proves that the borrowing interest rate is 4% given the conditions of the problem -/
theorem borrowing_interest_rate
  (principal : ℝ)
  (time : ℝ)
  (lending_rate : ℝ)
  (gain_per_year : ℝ)
  (h1 : principal = 5000)
  (h2 : time = 2)
  (h3 : lending_rate = 0.06)
  (h4 : gain_per_year = 100)
  : (principal * lending_rate - gain_per_year) / principal = 0.04 := by
  sorry

#eval (5000 * 0.06 - 100) / 5000  -- Should output 0.04

end NUMINAMATH_CALUDE_borrowing_interest_rate_l420_42036


namespace NUMINAMATH_CALUDE_problem_1_l420_42008

theorem problem_1 : (-5) + (-2) + 9 - (-8) = 10 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l420_42008


namespace NUMINAMATH_CALUDE_complex_magnitude_range_l420_42037

theorem complex_magnitude_range (z₁ z₂ : ℂ) 
  (h₁ : (z₁ - Complex.I) * (z₂ + Complex.I) = 1)
  (h₂ : Complex.abs z₁ = Real.sqrt 2) :
  ∃ (a b : ℝ), a = 2 - Real.sqrt 2 ∧ b = 2 + Real.sqrt 2 ∧ 
  a ≤ Complex.abs z₂ ∧ Complex.abs z₂ ≤ b :=
sorry

end NUMINAMATH_CALUDE_complex_magnitude_range_l420_42037


namespace NUMINAMATH_CALUDE_food_for_horses_l420_42012

/-- Calculates the total amount of food needed for horses over a number of days. -/
def total_food_needed (num_horses : ℕ) (num_days : ℕ) (oats_per_meal : ℕ) (oats_meals_per_day : ℕ) (grain_per_day : ℕ) : ℕ :=
  let total_oats := num_horses * num_days * oats_per_meal * oats_meals_per_day
  let total_grain := num_horses * num_days * grain_per_day
  total_oats + total_grain

/-- Theorem stating that the total food needed for 4 horses over 3 days is 132 pounds. -/
theorem food_for_horses :
  total_food_needed 4 3 4 2 3 = 132 := by
  sorry

end NUMINAMATH_CALUDE_food_for_horses_l420_42012


namespace NUMINAMATH_CALUDE_artist_cube_structure_surface_area_l420_42016

/-- Represents the cube structure described in the problem -/
structure CubeStructure where
  totalCubes : ℕ
  cubeEdgeLength : ℝ
  bottomLayerSize : ℕ
  topLayerSize : ℕ

/-- Calculates the exposed surface area of the cube structure -/
def exposedSurfaceArea (cs : CubeStructure) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem artist_cube_structure_surface_area :
  ∃ (cs : CubeStructure),
    cs.totalCubes = 16 ∧
    cs.cubeEdgeLength = 1 ∧
    cs.bottomLayerSize = 3 ∧
    cs.topLayerSize = 2 ∧
    exposedSurfaceArea cs = 49 :=
  sorry

end NUMINAMATH_CALUDE_artist_cube_structure_surface_area_l420_42016


namespace NUMINAMATH_CALUDE_easter_egg_probability_l420_42090

theorem easter_egg_probability : ∀ (total eggs : ℕ) (red_eggs : ℕ) (small_box : ℕ) (large_box : ℕ),
  total = 16 →
  red_eggs = 3 →
  small_box = 6 →
  large_box = 10 →
  small_box + large_box = total →
  (Nat.choose red_eggs 1 * Nat.choose (total - red_eggs) (small_box - 1) +
   Nat.choose red_eggs 2 * Nat.choose (total - red_eggs) (small_box - 2)) /
  Nat.choose total small_box = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_easter_egg_probability_l420_42090


namespace NUMINAMATH_CALUDE_mens_total_wages_l420_42001

/-- Proves that the men's total wages are 150 given the problem conditions --/
theorem mens_total_wages (W : ℕ) : 
  (12 : ℚ) * W = (20 : ℚ) → -- 12 men equal W women, W women equal 20 boys
  (12 : ℚ) * W * W + W * W * W + (20 : ℚ) * W = (450 : ℚ) → -- Total earnings equation
  (12 : ℚ) * ((450 : ℚ) / ((12 : ℚ) + W + (20 : ℚ))) = (150 : ℚ) -- Men's total wages
:= by sorry

end NUMINAMATH_CALUDE_mens_total_wages_l420_42001


namespace NUMINAMATH_CALUDE_accounting_majors_l420_42095

theorem accounting_majors (p q r s t u : ℕ) : 
  p * q * r * s * t * u = 51030 →
  1 < p → p < q → q < r → r < s → s < t → t < u →
  p = 2 := by sorry

end NUMINAMATH_CALUDE_accounting_majors_l420_42095


namespace NUMINAMATH_CALUDE_triple_transmission_more_accurate_main_theorem_l420_42028

/-- Probability of correctly decoding 0 in single transmission -/
def single_transmission_prob (α : ℝ) : ℝ := 1 - α

/-- Probability of correctly decoding 0 in triple transmission -/
def triple_transmission_prob (α : ℝ) : ℝ := 3 * α * (1 - α)^2 + (1 - α)^3

/-- Theorem stating that triple transmission is more accurate than single for sending 0 when 0 < α < 0.5 -/
theorem triple_transmission_more_accurate (α : ℝ) 
  (h1 : 0 < α) (h2 : α < 0.5) : 
  triple_transmission_prob α > single_transmission_prob α := by
  sorry

/-- Main theorem combining all conditions and the result -/
theorem main_theorem (α β : ℝ) 
  (h1 : 0 < α) (h2 : α < 0.5) (h3 : 0 < β) (h4 : β < 1) :
  triple_transmission_prob α > single_transmission_prob α ∧
  single_transmission_prob α = 1 - α ∧
  triple_transmission_prob α = 3 * α * (1 - α)^2 + (1 - α)^3 := by
  sorry

end NUMINAMATH_CALUDE_triple_transmission_more_accurate_main_theorem_l420_42028


namespace NUMINAMATH_CALUDE_triangle_theorem_l420_42066

-- Define the triangle ABC
def Triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Define the theorem
theorem triangle_theorem (a b c : ℝ) (A B C : ℝ) :
  Triangle a b c →
  b^2 * c * Real.cos C + c^2 * b * Real.cos B = a * b^2 + a * c^2 - a^3 →
  (A = Real.pi / 3 ∧
   (b + c = 2 → ∀ a' : ℝ, Triangle a' b c → a' ≥ 1)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_theorem_l420_42066


namespace NUMINAMATH_CALUDE_smallest_p_in_prime_sum_l420_42056

theorem smallest_p_in_prime_sum (p q r : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime r →
  p + q = r →
  1 < p →
  p < q →
  r > 10 →
  ∃ (p' : ℕ), p' = 2 ∧ (∀ (p'' : ℕ), 
    Nat.Prime p'' → 
    p'' + q = r → 
    1 < p'' → 
    p'' < q → 
    r > 10 → 
    p' ≤ p'') :=
by sorry

end NUMINAMATH_CALUDE_smallest_p_in_prime_sum_l420_42056


namespace NUMINAMATH_CALUDE_marcus_baseball_cards_l420_42074

theorem marcus_baseball_cards 
  (initial_cards : ℝ) 
  (additional_cards : ℝ) 
  (h1 : initial_cards = 210.0) 
  (h2 : additional_cards = 58.0) : 
  initial_cards + additional_cards = 268.0 := by
  sorry

end NUMINAMATH_CALUDE_marcus_baseball_cards_l420_42074


namespace NUMINAMATH_CALUDE_corner_sum_is_164_l420_42098

/-- Represents a 9x9 grid filled with numbers from 1 to 81 in row-major order -/
def Grid9x9 : Type := Fin 9 → Fin 9 → Nat

/-- The standard 9x9 grid filled with numbers 1 to 81 -/
def standardGrid : Grid9x9 :=
  λ i j => i.val * 9 + j.val + 1

/-- The sum of the corner elements in the standard 9x9 grid -/
def cornerSum (g : Grid9x9) : Nat :=
  g 0 0 + g 0 8 + g 8 0 + g 8 8

theorem corner_sum_is_164 : cornerSum standardGrid = 164 := by
  sorry

end NUMINAMATH_CALUDE_corner_sum_is_164_l420_42098


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l420_42026

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (i / (2 + i)) = ((1 : ℂ) + 2 * i) / 5 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l420_42026


namespace NUMINAMATH_CALUDE_train_passenger_count_l420_42014

def train_problem (initial_passengers : ℕ) (first_station_pickup : ℕ) (final_passengers : ℕ) : ℕ :=
  let after_first_drop := initial_passengers - (initial_passengers / 3)
  let after_first_pickup := after_first_drop + first_station_pickup
  let after_second_drop := after_first_pickup / 2
  final_passengers - after_second_drop

theorem train_passenger_count :
  train_problem 288 280 248 = 12 := by
  sorry

end NUMINAMATH_CALUDE_train_passenger_count_l420_42014


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l420_42084

theorem regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  exterior_angle = 18 → n * exterior_angle = 360 → n = 20 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l420_42084


namespace NUMINAMATH_CALUDE_special_pie_crust_flour_amount_l420_42096

/-- The amount of flour used in each special pie crust when the total flour amount remains constant but the number of crusts changes. -/
theorem special_pie_crust_flour_amount 
  (typical_crusts : ℕ) 
  (typical_flour_per_crust : ℚ) 
  (special_crusts : ℕ) 
  (h1 : typical_crusts = 50)
  (h2 : typical_flour_per_crust = 1 / 10)
  (h3 : special_crusts = 25)
  (h4 : typical_crusts * typical_flour_per_crust = special_crusts * (special_flour_per_crust : ℚ)) :
  special_flour_per_crust = 1 / 5 := by
  sorry

#check special_pie_crust_flour_amount

end NUMINAMATH_CALUDE_special_pie_crust_flour_amount_l420_42096


namespace NUMINAMATH_CALUDE_igor_sequence_uses_three_infinitely_l420_42004

/-- Represents a sequence of natural numbers where each number is obtained
    from the previous one by adding n/p, where p is a prime divisor of n. -/
def IgorSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n > 1) ∧
  (∀ n, ∃ p, Nat.Prime p ∧ p ∣ a n ∧ a (n + 1) = a n + a n / p)

/-- The theorem stating that in an infinite IgorSequence,
    the prime 3 must be used as a divisor infinitely many times. -/
theorem igor_sequence_uses_three_infinitely (a : ℕ → ℕ) (h : IgorSequence a) :
  ∀ m, ∃ n > m, ∃ p, p = 3 ∧ Nat.Prime p ∧ p ∣ a n ∧ a (n + 1) = a n + a n / p :=
sorry

end NUMINAMATH_CALUDE_igor_sequence_uses_three_infinitely_l420_42004


namespace NUMINAMATH_CALUDE_arithmetic_squares_sequence_l420_42009

theorem arithmetic_squares_sequence (k : ℤ) : 
  (∃! k : ℤ, 
    (∃ a : ℤ, 
      (49 + k = a^2) ∧ 
      (361 + k = (a + 2)^2) ∧ 
      (784 + k = (a + 4)^2))) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_squares_sequence_l420_42009


namespace NUMINAMATH_CALUDE_triangle_side_length_l420_42041

theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  a = 2 →
  c = 2 * Real.sqrt 3 →
  Real.cos A = Real.sqrt 3 / 2 →
  b < c →
  a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A →
  b = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l420_42041


namespace NUMINAMATH_CALUDE_average_difference_with_input_error_l420_42025

theorem average_difference_with_input_error (n : ℕ) (correct_value wrong_value : ℝ) : 
  n = 30 → correct_value = 75 → wrong_value = 15 → 
  (correct_value - wrong_value) / n = -2 := by
sorry

end NUMINAMATH_CALUDE_average_difference_with_input_error_l420_42025


namespace NUMINAMATH_CALUDE_spring_length_at_9kg_spring_length_conditions_l420_42010

/-- A linear function representing the relationship between mass and spring length. -/
def spring_length (x : ℝ) : ℝ := 0.5 * x + 10

/-- Theorem stating that the spring length is 14.5 cm when the mass is 9 kg. -/
theorem spring_length_at_9kg :
  spring_length 0 = 10 →
  spring_length 1 = 10.5 →
  spring_length 9 = 14.5 := by
  sorry

/-- Proof that the spring_length function satisfies the given conditions. -/
theorem spring_length_conditions :
  spring_length 0 = 10 ∧ spring_length 1 = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_spring_length_at_9kg_spring_length_conditions_l420_42010
