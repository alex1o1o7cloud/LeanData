import Mathlib

namespace NUMINAMATH_CALUDE_youngest_son_park_visits_l1438_143830

theorem youngest_son_park_visits (season_pass_cost : ℝ) (oldest_son_visits : ℕ) (youngest_son_cost_per_trip : ℝ) :
  season_pass_cost = 100 →
  oldest_son_visits = 35 →
  youngest_son_cost_per_trip = 4 →
  ∃ (youngest_son_visits : ℕ), 
    (season_pass_cost / youngest_son_visits) = youngest_son_cost_per_trip ∧
    youngest_son_visits = 25 :=
by sorry

end NUMINAMATH_CALUDE_youngest_son_park_visits_l1438_143830


namespace NUMINAMATH_CALUDE_fgh_supermarket_difference_l1438_143877

/-- Prove that the difference between FGH supermarkets in the US and Canada is 14 -/
theorem fgh_supermarket_difference : ∀ (total us : ℕ),
  total = 84 →
  us = 49 →
  us > total - us →
  us - (total - us) = 14 := by
  sorry

end NUMINAMATH_CALUDE_fgh_supermarket_difference_l1438_143877


namespace NUMINAMATH_CALUDE_unique_c_value_l1438_143817

-- Define the function f(x) = x⋅(2x+1)
def f (x : ℝ) : ℝ := x * (2 * x + 1)

-- Define the open interval (-2, 3/2)
def interval : Set ℝ := {x | -2 < x ∧ x < 3/2}

-- State the theorem
theorem unique_c_value : ∃! c : ℝ, ∀ x : ℝ, x ∈ interval ↔ f x < c :=
  sorry

end NUMINAMATH_CALUDE_unique_c_value_l1438_143817


namespace NUMINAMATH_CALUDE_last_digit_of_2008_power_last_digit_of_2008_to_2008_l1438_143828

theorem last_digit_of_2008_power (n : ℕ) : n > 0 → (2008^n) % 10 = (2008^(n % 4)) % 10 := by sorry

theorem last_digit_of_2008_to_2008 : (2008^2008) % 10 = 6 := by sorry

end NUMINAMATH_CALUDE_last_digit_of_2008_power_last_digit_of_2008_to_2008_l1438_143828


namespace NUMINAMATH_CALUDE_gcd_lcm_multiple_relation_l1438_143841

theorem gcd_lcm_multiple_relation (x y z : ℤ) (h1 : y ≠ 0) (h2 : x / y = z) : 
  Int.gcd x y = y ∧ Int.lcm x y = x := by sorry

end NUMINAMATH_CALUDE_gcd_lcm_multiple_relation_l1438_143841


namespace NUMINAMATH_CALUDE_consecutive_integers_problem_l1438_143856

theorem consecutive_integers_problem (a b c d e : ℕ) : 
  a > 0 → b > 0 → c > 0 → d > 0 → e > 0 →
  b = a + 1 → c = b + 1 → d = c + 1 → e = d + 1 →
  a < b → b < c → c < d → d < e →
  a + b = e - 1 →
  a * b = d + 1 →
  c = 4 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_problem_l1438_143856


namespace NUMINAMATH_CALUDE_sum_of_qp_values_l1438_143895

def p (x : ℝ) : ℝ := |x| - 3

def q (x : ℝ) : ℝ := -|x|

def x_values : List ℝ := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

theorem sum_of_qp_values :
  (x_values.map (λ x => q (p x))).sum = -15 := by sorry

end NUMINAMATH_CALUDE_sum_of_qp_values_l1438_143895


namespace NUMINAMATH_CALUDE_sum_of_real_and_imag_parts_of_z_l1438_143897

theorem sum_of_real_and_imag_parts_of_z (i : ℂ) (h : i^2 = -1) : 
  let z : ℂ := (1 + 2*i) / i
  (z.re + z.im : ℝ) = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_real_and_imag_parts_of_z_l1438_143897


namespace NUMINAMATH_CALUDE_travelers_checks_worth_l1438_143886

-- Define the problem parameters
def total_checks : ℕ := 30
def small_denomination : ℕ := 50
def large_denomination : ℕ := 100
def spent_checks : ℕ := 18
def remaining_average : ℕ := 75

-- Define the theorem
theorem travelers_checks_worth :
  ∀ (x y : ℕ),
    x + y = total_checks →
    x ≥ spent_checks →
    (small_denomination * (x - spent_checks) + large_denomination * y) / (total_checks - spent_checks) = remaining_average →
    small_denomination * x + large_denomination * y = 1800 :=
by
  sorry

end NUMINAMATH_CALUDE_travelers_checks_worth_l1438_143886


namespace NUMINAMATH_CALUDE_arrangements_together_count_arrangements_alternate_count_arrangements_restricted_count_l1438_143855

-- Define the number of boys and girls
def num_boys : ℕ := 2
def num_girls : ℕ := 3
def total_students : ℕ := num_boys + num_girls

-- Define the functions for each arrangement scenario
def arrangements_together : ℕ := sorry

def arrangements_alternate : ℕ := sorry

def arrangements_restricted : ℕ := sorry

-- State the theorems to be proved
theorem arrangements_together_count : arrangements_together = 24 := by sorry

theorem arrangements_alternate_count : arrangements_alternate = 12 := by sorry

theorem arrangements_restricted_count : arrangements_restricted = 60 := by sorry

end NUMINAMATH_CALUDE_arrangements_together_count_arrangements_alternate_count_arrangements_restricted_count_l1438_143855


namespace NUMINAMATH_CALUDE_sum_maximized_at_14_l1438_143883

/-- The nth term of the sequence -/
def a (n : ℕ) : ℤ := 43 - 3 * n

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ) : ℤ := n * (83 - 3 * n) / 2

/-- The theorem stating that the sum is maximized when n = 14 -/
theorem sum_maximized_at_14 :
  ∀ k : ℕ, k ≠ 0 → S 14 ≥ S k :=
sorry

end NUMINAMATH_CALUDE_sum_maximized_at_14_l1438_143883


namespace NUMINAMATH_CALUDE_greatest_divisor_remainder_l1438_143808

theorem greatest_divisor_remainder (G R1 : ℕ) : 
  G = 29 →
  1490 % G = 11 →
  (∀ d : ℕ, d > G → (1255 % d ≠ 0 ∨ 1490 % d ≠ 0)) →
  1255 % G = R1 →
  R1 = 8 := by
sorry

end NUMINAMATH_CALUDE_greatest_divisor_remainder_l1438_143808


namespace NUMINAMATH_CALUDE_derivative_f_at_specific_point_l1438_143825

-- Define the function f
def f (x : ℝ) : ℝ := x^2008

-- State the theorem
theorem derivative_f_at_specific_point :
  deriv f ((1 / 2008 : ℝ)^(1 / 2007)) = 1 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_specific_point_l1438_143825


namespace NUMINAMATH_CALUDE_drum_capacity_ratio_l1438_143820

/-- Given two drums X and Y with oil, prove that the ratio of Y's capacity to X's capacity is 2 -/
theorem drum_capacity_ratio (C_X C_Y : ℝ) : 
  C_X > 0 → C_Y > 0 → 
  (1/2 : ℝ) * C_X + (1/5 : ℝ) * C_Y = (0.45 : ℝ) * C_Y → 
  C_Y / C_X = 2 := by
sorry

end NUMINAMATH_CALUDE_drum_capacity_ratio_l1438_143820


namespace NUMINAMATH_CALUDE_fourth_buoy_adjusted_distance_l1438_143862

def buoy_distance (n : ℕ) : ℝ :=
  20 + 4 * (n - 1)

def ocean_current : ℝ := 3

theorem fourth_buoy_adjusted_distance :
  buoy_distance 4 - ocean_current = 29 := by
  sorry

end NUMINAMATH_CALUDE_fourth_buoy_adjusted_distance_l1438_143862


namespace NUMINAMATH_CALUDE_beef_weight_before_processing_l1438_143806

theorem beef_weight_before_processing (weight_after : ℝ) (percentage_lost : ℝ) 
  (h1 : weight_after = 546)
  (h2 : percentage_lost = 35) : 
  weight_after / (1 - percentage_lost / 100) = 840 := by
  sorry

end NUMINAMATH_CALUDE_beef_weight_before_processing_l1438_143806


namespace NUMINAMATH_CALUDE_equilateral_triangle_circumcircle_parallel_lines_collinear_l1438_143861

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a line in 2D space -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Represents a circle in 2D space -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- Represents a triangle in 2D space -/
structure Triangle :=
  (a : Point) (b : Point) (c : Point)

/-- Checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Checks if a point lies on a circle -/
def pointOnCircle (p : Point) (c : Circle) : Prop := sorry

/-- Checks if a line is parallel to another line -/
def isParallel (l1 l2 : Line) : Prop := sorry

/-- Finds the intersection point of two lines -/
def lineIntersection (l1 l2 : Line) : Point := sorry

/-- Checks if three points are collinear -/
def areCollinear (p1 p2 p3 : Point) : Prop := sorry

/-- Main theorem -/
theorem equilateral_triangle_circumcircle_parallel_lines_collinear 
  (abc : Triangle) 
  (circ : Circle) 
  (p : Point) 
  (h1 : isEquilateral abc) 
  (h2 : pointOnCircle p circ) 
  (l1 l2 l3 : Line) 
  (h3 : isParallel l1 (Line.mk 1 0 0))  -- Parallel to BC
  (h4 : isParallel l2 (Line.mk 0 1 0))  -- Parallel to CA
  (h5 : isParallel l3 (Line.mk 1 1 0))  -- Parallel to AB
  : 
  let m := lineIntersection l2 (Line.mk 0 1 0)  -- Intersection with CA
  let n := lineIntersection l3 (Line.mk 1 1 0)  -- Intersection with AB
  let q := lineIntersection l1 (Line.mk 1 0 0)  -- Intersection with BC
  areCollinear m n q := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_circumcircle_parallel_lines_collinear_l1438_143861


namespace NUMINAMATH_CALUDE_sum_of_distinct_remainders_for_ten_l1438_143837

theorem sum_of_distinct_remainders_for_ten : ∃ (s : Finset ℕ), 
  (∀ r ∈ s, ∃ d : ℕ, 1 ≤ d ∧ d ≤ 9 ∧ r = 10 % d) ∧ 
  (∀ d : ℕ, 1 ≤ d → d ≤ 9 → (10 % d) ∈ s) ∧
  (s.sum id = 10) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_distinct_remainders_for_ten_l1438_143837


namespace NUMINAMATH_CALUDE_neg_two_plus_one_eq_neg_one_l1438_143869

theorem neg_two_plus_one_eq_neg_one : (-2) + 1 = -1 := by
  sorry

end NUMINAMATH_CALUDE_neg_two_plus_one_eq_neg_one_l1438_143869


namespace NUMINAMATH_CALUDE_package_servings_l1438_143833

/-- The number of servings in a package of candy. -/
def servings_in_package (calories_per_serving : ℕ) (calories_in_half : ℕ) : ℕ :=
  (2 * calories_in_half) / calories_per_serving

/-- Theorem: Given a package where each serving has 120 calories and half the package contains 180 calories, 
    prove that there are 3 servings in the package. -/
theorem package_servings : servings_in_package 120 180 = 3 := by
  sorry

end NUMINAMATH_CALUDE_package_servings_l1438_143833


namespace NUMINAMATH_CALUDE_square_plus_self_divisible_by_two_l1438_143844

theorem square_plus_self_divisible_by_two (n : ℤ) : ∃ k : ℤ, n^2 + n = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_square_plus_self_divisible_by_two_l1438_143844


namespace NUMINAMATH_CALUDE_washing_machine_last_load_l1438_143860

theorem washing_machine_last_load (capacity : ℕ) (total_clothes : ℕ) : 
  capacity = 28 → total_clothes = 200 → 
  total_clothes % capacity = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_washing_machine_last_load_l1438_143860


namespace NUMINAMATH_CALUDE_maximum_of_sum_of_roots_l1438_143801

theorem maximum_of_sum_of_roots (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 13) :
  Real.sqrt (x + 27) + Real.sqrt (13 - x) + Real.sqrt x ≤ 11 ∧
  ∃ y, 0 ≤ y ∧ y ≤ 13 ∧ Real.sqrt (y + 27) + Real.sqrt (13 - y) + Real.sqrt y = 11 :=
by sorry

end NUMINAMATH_CALUDE_maximum_of_sum_of_roots_l1438_143801


namespace NUMINAMATH_CALUDE_chinese_and_math_books_same_student_probability_l1438_143809

def num_books : ℕ := 4
def num_students : ℕ := 2

def has_chinese_book : Bool := true
def has_math_book : Bool := true

def books_per_student : ℕ := num_books / num_students

theorem chinese_and_math_books_same_student_probability :
  let total_distributions := (num_books.choose books_per_student)
  let favorable_distributions := 2  -- Number of ways Chinese and Math books can be together
  (favorable_distributions : ℚ) / total_distributions = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_chinese_and_math_books_same_student_probability_l1438_143809


namespace NUMINAMATH_CALUDE_smallest_n_value_l1438_143842

theorem smallest_n_value : ∃ (n : ℕ+), 
  (∀ (m : ℕ+), m < n → ¬(∃ (r g b : ℕ+), 10*r = 16*g ∧ 16*g = 18*b ∧ 18*b = 24*m ∧ 24*m = 30*22)) ∧ 
  (∃ (r g b : ℕ+), 10*r = 16*g ∧ 16*g = 18*b ∧ 18*b = 24*n ∧ 24*n = 30*22) ∧
  n = 30 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_value_l1438_143842


namespace NUMINAMATH_CALUDE_floor_width_is_eight_meters_l1438_143887

/-- Proves that a rectangular floor with given dimensions has a width of 8 meters -/
theorem floor_width_is_eight_meters
  (floor_length : ℝ)
  (rug_area : ℝ)
  (strip_width : ℝ)
  (h1 : floor_length = 10)
  (h2 : rug_area = 24)
  (h3 : strip_width = 2)
  : ∃ (floor_width : ℝ),
    floor_width = 8 ∧
    rug_area = (floor_length - 2 * strip_width) * (floor_width - 2 * strip_width) :=
by
  sorry


end NUMINAMATH_CALUDE_floor_width_is_eight_meters_l1438_143887


namespace NUMINAMATH_CALUDE_smallest_appended_digits_for_divisibility_l1438_143894

theorem smallest_appended_digits_for_divisibility (n : ℕ) : 
  (∃ k : ℕ, 
    (n = 2014) ∧ 
    (∀ m : ℕ, m < 10 → (n * 10000 + k) % m = 0) ∧
    (∀ j : ℕ, j < 10000 → 
      (∃ m : ℕ, m < 10 ∧ (n * j + k) % m ≠ 0))) → 
  (∃ k : ℕ, k < 10000 ∧ 
    (∀ m : ℕ, m < 10 → (n * 10000 + k) % m = 0)) :=
sorry

end NUMINAMATH_CALUDE_smallest_appended_digits_for_divisibility_l1438_143894


namespace NUMINAMATH_CALUDE_product_of_D_coordinates_l1438_143854

-- Define the points
def C : ℝ × ℝ := (-2, -7)
def M : ℝ × ℝ := (4, -3)

-- Define D as a variable point
variable (D : ℝ × ℝ)

-- State the theorem
theorem product_of_D_coordinates : 
  (M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) → D.1 * D.2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_product_of_D_coordinates_l1438_143854


namespace NUMINAMATH_CALUDE_circle_radii_sum_l1438_143899

theorem circle_radii_sum : 
  ∀ s : ℝ, 
  (s > 0) →
  (s^2 - 12*s + 12 = 0) →
  (∃ t : ℝ, s^2 - 12*s + 12 = 0 ∧ s ≠ t) →
  (s + (12 - s) = 12) := by
  sorry

end NUMINAMATH_CALUDE_circle_radii_sum_l1438_143899


namespace NUMINAMATH_CALUDE_pirate_overtakes_at_six_hours_l1438_143822

/-- Represents the chase scenario between a pirate ship and a merchant vessel -/
structure ChaseScenario where
  initial_distance : ℝ
  initial_pirate_speed : ℝ
  initial_merchant_speed : ℝ
  speed_change_time : ℝ
  final_pirate_speed : ℝ
  final_merchant_speed : ℝ

/-- Calculates the time when the pirate ship overtakes the merchant vessel -/
def overtake_time (scenario : ChaseScenario) : ℝ :=
  sorry

/-- Theorem stating that the pirate ship overtakes the merchant vessel after 6 hours -/
theorem pirate_overtakes_at_six_hours (scenario : ChaseScenario) 
  (h1 : scenario.initial_distance = 15)
  (h2 : scenario.initial_pirate_speed = 14)
  (h3 : scenario.initial_merchant_speed = 10)
  (h4 : scenario.speed_change_time = 3)
  (h5 : scenario.final_pirate_speed = 12)
  (h6 : scenario.final_merchant_speed = 11) :
  overtake_time scenario = 6 :=
  sorry

end NUMINAMATH_CALUDE_pirate_overtakes_at_six_hours_l1438_143822


namespace NUMINAMATH_CALUDE_answer_key_combinations_l1438_143879

/-- Represents the number of answer choices for a multiple-choice question -/
def multipleChoiceOptions : ℕ := 4

/-- Represents the number of true-false questions -/
def trueFalseQuestions : ℕ := 5

/-- Represents the number of multiple-choice questions -/
def multipleChoiceQuestions : ℕ := 2

/-- Calculates the number of valid true-false combinations -/
def validTrueFalseCombinations : ℕ := 2^trueFalseQuestions - 2

/-- Calculates the number of multiple-choice combinations -/
def multipleChoiceCombinations : ℕ := multipleChoiceOptions^multipleChoiceQuestions

/-- Theorem: The number of ways to create an answer key for the quiz is 480 -/
theorem answer_key_combinations : 
  validTrueFalseCombinations * multipleChoiceCombinations = 480 := by
  sorry

end NUMINAMATH_CALUDE_answer_key_combinations_l1438_143879


namespace NUMINAMATH_CALUDE_circle_properties_l1438_143891

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the line equations
def line1_equation (x y : ℝ) : Prop :=
  3*x + 4*y - 6 = 0

def line2_equation (x y : ℝ) : Prop :=
  x - y - 1 = 0

-- Define the theorem
theorem circle_properties :
  -- Part 1: The circle exists when m < 5
  (∀ m : ℝ, m < 5 → ∃ x y : ℝ, circle_equation x y m) ∧
  -- Part 2: When the circle intersects line1 at M and N with |MN| = 2√3, m = 1
  (∃ m x₁ y₁ x₂ y₂ : ℝ,
    circle_equation x₁ y₁ m ∧
    circle_equation x₂ y₂ m ∧
    line1_equation x₁ y₁ ∧
    line1_equation x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 12 ∧
    m = 1) ∧
  -- Part 3: When the circle intersects line2 at A and B, there exists m = -2
  -- such that the circle with diameter AB passes through the origin
  (∃ m x₁ y₁ x₂ y₂ : ℝ,
    circle_equation x₁ y₁ m ∧
    circle_equation x₂ y₂ m ∧
    line2_equation x₁ y₁ ∧
    line2_equation x₂ y₂ ∧
    x₁ * x₂ + y₁ * y₂ = 0 ∧
    m = -2) := by
  sorry


end NUMINAMATH_CALUDE_circle_properties_l1438_143891


namespace NUMINAMATH_CALUDE_cone_in_cylinder_volume_ratio_l1438_143810

noncomputable def cone_volume (base_area : ℝ) (height : ℝ) : ℝ := 
  (1 / 3) * base_area * height

noncomputable def cylinder_volume (base_area : ℝ) (height : ℝ) : ℝ := 
  base_area * height

theorem cone_in_cylinder_volume_ratio 
  (base_area : ℝ) (height : ℝ) (h_pos : base_area > 0 ∧ height > 0) :
  let v_cone := cone_volume base_area height
  let v_cylinder := cylinder_volume base_area height
  (v_cylinder - v_cone) / v_cone = 2 := by
sorry

end NUMINAMATH_CALUDE_cone_in_cylinder_volume_ratio_l1438_143810


namespace NUMINAMATH_CALUDE_division_of_decimals_l1438_143818

theorem division_of_decimals : (0.045 : ℝ) / 0.0075 = 6 := by
  sorry

end NUMINAMATH_CALUDE_division_of_decimals_l1438_143818


namespace NUMINAMATH_CALUDE_expression_value_l1438_143846

theorem expression_value (x y z : ℤ) (hx : x = 25) (hy : y = 30) (hz : z = 10) :
  (x - (y - z)) - ((x - y) - z) = 20 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1438_143846


namespace NUMINAMATH_CALUDE_part_one_part_two_l1438_143878

-- Part 1
theorem part_one (x y : ℝ) : 
  y = Real.sqrt (x - 2) + Real.sqrt (2 - x) + 3 →
  x = 2 →
  x - y = -1 := by sorry

-- Part 2
theorem part_two (x : ℝ) :
  x = Real.sqrt 2 →
  (x / (x - 2)) / (2 + x - 4 / (2 - x)) = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1438_143878


namespace NUMINAMATH_CALUDE_max_value_of_sum_max_value_achieved_l1438_143813

theorem max_value_of_sum (x y z : ℝ) : 
  x ≥ 0 → y ≥ 0 → z ≥ 0 → x + y + z = 2 → x^2 + y^3 + z^4 ≤ 2 := by
  sorry

theorem max_value_achieved (x y z : ℝ) : 
  x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 2 ∧ x^2 + y^3 + z^4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_sum_max_value_achieved_l1438_143813


namespace NUMINAMATH_CALUDE_monitor_pixel_count_l1438_143815

/-- Calculate the total number of pixels on a monitor given its dimensions and pixel density. -/
theorem monitor_pixel_count (width : ℕ) (height : ℕ) (pixel_density : ℕ) : 
  width = 32 → height = 18 → pixel_density = 150 → 
  width * height * pixel_density * pixel_density = 12960000 := by
  sorry

end NUMINAMATH_CALUDE_monitor_pixel_count_l1438_143815


namespace NUMINAMATH_CALUDE_cube_sum_product_l1438_143870

def is_even_or_prime (n : ℕ) : Prop :=
  Even n ∨ Nat.Prime n

theorem cube_sum_product : ∃ (a b : ℕ), 
  a^3 + b^3 = 91 ∧ 
  is_even_or_prime a ∧ 
  is_even_or_prime b ∧ 
  a * b = 12 := by sorry

end NUMINAMATH_CALUDE_cube_sum_product_l1438_143870


namespace NUMINAMATH_CALUDE_initial_boarders_count_prove_initial_boarders_count_l1438_143868

/-- Proves that the initial number of boarders is 120 given the conditions of the problem -/
theorem initial_boarders_count : ℕ → ℕ → Prop :=
  fun initial_boarders initial_day_students =>
    -- Initial ratio of boarders to day students is 2:5
    (initial_boarders : ℚ) / initial_day_students = 2 / 5 →
    -- After 30 new boarders join, the ratio becomes 1:2
    ((initial_boarders : ℚ) + 30) / initial_day_students = 1 / 2 →
    -- The initial number of boarders is 120
    initial_boarders = 120

-- The proof of the theorem
theorem prove_initial_boarders_count : ∃ (initial_boarders initial_day_students : ℕ),
  initial_boarders_count initial_boarders initial_day_students :=
by
  sorry -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_initial_boarders_count_prove_initial_boarders_count_l1438_143868


namespace NUMINAMATH_CALUDE_multiplication_and_subtraction_l1438_143885

theorem multiplication_and_subtraction : 10 * (5 - 2) = 30 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_and_subtraction_l1438_143885


namespace NUMINAMATH_CALUDE_exam_results_l1438_143892

theorem exam_results (total_students : ℕ) (second_division_percent : ℚ) 
  (just_passed : ℕ) (h1 : total_students = 300) 
  (h2 : second_division_percent = 54/100) (h3 : just_passed = 57) : 
  (1 - second_division_percent - (just_passed : ℚ) / total_students) = 27/100 := by
  sorry

end NUMINAMATH_CALUDE_exam_results_l1438_143892


namespace NUMINAMATH_CALUDE_square_difference_l1438_143864

theorem square_difference (x y : ℚ) 
  (sum_eq : x + y = 9/13) 
  (diff_eq : x - y = 5/13) : 
  x^2 - y^2 = 45/169 := by
sorry

end NUMINAMATH_CALUDE_square_difference_l1438_143864


namespace NUMINAMATH_CALUDE_frank_maze_time_l1438_143850

/-- Represents the maximum additional time Frank can spend in the current maze -/
def max_additional_time (current_time : ℕ) (previous_mazes : ℕ) (average_previous : ℕ) (max_average : ℕ) : ℕ :=
  max_average * (previous_mazes + 1) - (average_previous * previous_mazes + current_time)

/-- Theorem stating the maximum additional time Frank can spend in the maze -/
theorem frank_maze_time : max_additional_time 45 4 50 60 = 55 := by
  sorry

end NUMINAMATH_CALUDE_frank_maze_time_l1438_143850


namespace NUMINAMATH_CALUDE_ellipse_equation_l1438_143898

/-- Given an ellipse with foci on the y-axis, sum of distances from any point to the foci equal to 8,
    and focal length 2√15, prove that its standard equation is (y²/16) + x² = 1 -/
theorem ellipse_equation (a b c : ℝ) (h1 : 2 * a = 8) (h2 : 2 * c = 2 * Real.sqrt 15)
    (h3 : a ^ 2 = b ^ 2 + c ^ 2) :
  ∀ (x y : ℝ), (y ^ 2 / 16 + x ^ 2 = 1) ↔ (y ^ 2 / a ^ 2 + x ^ 2 / b ^ 2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1438_143898


namespace NUMINAMATH_CALUDE_soil_cost_per_cubic_foot_l1438_143840

/-- Calculates the cost per cubic foot of soil for Bob's gardening project. -/
theorem soil_cost_per_cubic_foot
  (rose_bushes : ℕ)
  (rose_bush_cost : ℚ)
  (gardener_hourly_rate : ℚ)
  (gardener_hours_per_day : ℕ)
  (gardener_days : ℕ)
  (soil_volume : ℕ)
  (total_project_cost : ℚ)
  (h1 : rose_bushes = 20)
  (h2 : rose_bush_cost = 150)
  (h3 : gardener_hourly_rate = 30)
  (h4 : gardener_hours_per_day = 5)
  (h5 : gardener_days = 4)
  (h6 : soil_volume = 100)
  (h7 : total_project_cost = 4100) :
  (total_project_cost - (rose_bushes * rose_bush_cost + gardener_hourly_rate * gardener_hours_per_day * gardener_days)) / soil_volume = 5 := by
  sorry

end NUMINAMATH_CALUDE_soil_cost_per_cubic_foot_l1438_143840


namespace NUMINAMATH_CALUDE_min_value_quadratic_l1438_143880

theorem min_value_quadratic (x : ℝ) : x^2 + 10*x ≥ -25 ∧ ∃ y : ℝ, y^2 + 10*y = -25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l1438_143880


namespace NUMINAMATH_CALUDE_function_value_at_a_plus_one_l1438_143821

/-- Given a function f(x) = x^2 + 1, prove that f(a+1) = a^2 + 2a + 2 for any real number a. -/
theorem function_value_at_a_plus_one (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^2 + 1
  f (a + 1) = a^2 + 2*a + 2 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_a_plus_one_l1438_143821


namespace NUMINAMATH_CALUDE_chess_tournament_players_l1438_143873

/-- The number of players in the chess tournament -/
def num_players : ℕ := 45

/-- The total score of all players in the tournament -/
def total_score : ℕ := 1980

/-- Theorem stating that the number of players is correct given the total score -/
theorem chess_tournament_players :
  num_players * (num_players - 1) = total_score :=
by sorry

end NUMINAMATH_CALUDE_chess_tournament_players_l1438_143873


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1438_143853

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p | p.2 = p.1 + 1 ∧ 0 ≤ p.1 ∧ p.1 ≤ 1}
def B : Set (ℝ × ℝ) := {p | p.2 = 2 * p.1 ∧ 0 ≤ p.1 ∧ p.1 ≤ 10}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {(1, 2)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1438_143853


namespace NUMINAMATH_CALUDE_rectangle_area_12_l1438_143859

def rectangle_area (w l : ℕ+) : ℕ := w.val * l.val

def valid_rectangles : Set (ℕ+ × ℕ+) :=
  {p | rectangle_area p.1 p.2 = 12}

theorem rectangle_area_12 :
  valid_rectangles = {(1, 12), (2, 6), (3, 4), (4, 3), (6, 2), (12, 1)} := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_12_l1438_143859


namespace NUMINAMATH_CALUDE_prime_cube_plus_seven_composite_l1438_143824

theorem prime_cube_plus_seven_composite (P : ℕ) (h1 : Nat.Prime P) (h2 : Nat.Prime (P^3 + 5)) :
  ¬Nat.Prime (P^3 + 7) ∧ (P^3 + 7) > 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_cube_plus_seven_composite_l1438_143824


namespace NUMINAMATH_CALUDE_min_steps_ladder_l1438_143875

theorem min_steps_ladder (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  let n := a + b - Nat.gcd a b
  ∀ m : ℕ, (∃ (x y : ℕ), x * a - y * b = m ∧ x * a - y * b = 0) → m ≥ n :=
sorry

end NUMINAMATH_CALUDE_min_steps_ladder_l1438_143875


namespace NUMINAMATH_CALUDE_oomyapeck_eyes_eaten_l1438_143802

/-- The number of eyes Oomyapeck eats given the family size, fish per person, eyes per fish, and eyes given away --/
def eyes_eaten (family_size : ℕ) (fish_per_person : ℕ) (eyes_per_fish : ℕ) (eyes_given_away : ℕ) : ℕ :=
  family_size * fish_per_person * eyes_per_fish - eyes_given_away

/-- Theorem stating that under the given conditions, Oomyapeck eats 22 eyes --/
theorem oomyapeck_eyes_eaten :
  eyes_eaten 3 4 2 2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_oomyapeck_eyes_eaten_l1438_143802


namespace NUMINAMATH_CALUDE_sqrt_difference_inequality_l1438_143811

theorem sqrt_difference_inequality : 
  let a := Real.sqrt 2023 - Real.sqrt 2022
  let b := Real.sqrt 2022 - Real.sqrt 2021
  let c := Real.sqrt 2021 - Real.sqrt 2020
  c > b ∧ b > a := by sorry

end NUMINAMATH_CALUDE_sqrt_difference_inequality_l1438_143811


namespace NUMINAMATH_CALUDE_average_and_difference_l1438_143816

theorem average_and_difference (x : ℝ) : 
  (35 + x) / 2 = 45 → |x - 35| = 20 := by
sorry

end NUMINAMATH_CALUDE_average_and_difference_l1438_143816


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1438_143896

theorem absolute_value_inequality (x : ℝ) : 
  3 < |x + 2| ∧ |x + 2| ≤ 6 ↔ (-8 ≤ x ∧ x < -5) ∨ (1 < x ∧ x ≤ 4) := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1438_143896


namespace NUMINAMATH_CALUDE_A_equals_B_l1438_143819

/-- Set A defined as {a | a = 12m + 8n + 4l, m, n, l ∈ ℤ} -/
def A : Set ℤ := {a | ∃ m n l : ℤ, a = 12*m + 8*n + 4*l}

/-- Set B defined as {b | b = 20p + 16q + 12r, p, q, r ∈ ℤ} -/
def B : Set ℤ := {b | ∃ p q r : ℤ, b = 20*p + 16*q + 12*r}

/-- Theorem stating that A = B -/
theorem A_equals_B : A = B := by
  sorry


end NUMINAMATH_CALUDE_A_equals_B_l1438_143819


namespace NUMINAMATH_CALUDE_x_range_for_inequality_l1438_143866

theorem x_range_for_inequality (x : ℝ) : 
  (∀ m : ℝ, -2 ≤ m ∧ m ≤ 2 → 2 * x - 1 > m * (x^2 - 1)) →
  ((-1 + Real.sqrt 7) / 2 < x ∧ x < (1 + Real.sqrt 3) / 2) :=
by sorry

end NUMINAMATH_CALUDE_x_range_for_inequality_l1438_143866


namespace NUMINAMATH_CALUDE_quarters_remaining_l1438_143874

theorem quarters_remaining (initial_quarters : ℕ) (payment_dollars : ℕ) (quarters_per_dollar : ℕ) : 
  initial_quarters = 160 →
  payment_dollars = 35 →
  quarters_per_dollar = 4 →
  initial_quarters - (payment_dollars * quarters_per_dollar) = 20 :=
by sorry

end NUMINAMATH_CALUDE_quarters_remaining_l1438_143874


namespace NUMINAMATH_CALUDE_function_inequality_implies_parameter_bound_l1438_143803

/-- Given f(x) = 2ln(x) - x^2 and g(x) = xe^x - (a-1)x^2 - x - 2ln(x) for x > 0,
    if f(x) + g(x) ≥ 0 for all x > 0, then a ≤ 1 -/
theorem function_inequality_implies_parameter_bound (a : ℝ) :
  (∀ x > 0, 2 * Real.log x - x^2 + x * Real.exp x - (a - 1) * x^2 - x - 2 * Real.log x ≥ 0) →
  a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_parameter_bound_l1438_143803


namespace NUMINAMATH_CALUDE_distance_ratio_proof_l1438_143872

/-- Proves that the ratio of distances covered at different speeds is 1:1 given specific conditions -/
theorem distance_ratio_proof (total_distance : ℝ) (speed1 speed2 : ℝ) (total_time : ℝ) 
  (h1 : total_distance = 3600)
  (h2 : speed1 = 90)
  (h3 : speed2 = 180)
  (h4 : total_time = 30)
  (h5 : ∃ (d1 d2 : ℝ), d1 + d2 = total_distance ∧ d1 / speed1 + d2 / speed2 = total_time) :
  ∃ (d1 d2 : ℝ), d1 + d2 = total_distance ∧ d1 / speed1 + d2 / speed2 = total_time ∧ d1 = d2 := by
  sorry

#check distance_ratio_proof

end NUMINAMATH_CALUDE_distance_ratio_proof_l1438_143872


namespace NUMINAMATH_CALUDE_homework_problem_count_l1438_143836

theorem homework_problem_count (p t : ℕ) : 
  p > 0 → 
  t > 0 → 
  p ≥ 15 → 
  p * t = (2 * p - 10) * (t - 1) → 
  p * t = 60 := by
  sorry

end NUMINAMATH_CALUDE_homework_problem_count_l1438_143836


namespace NUMINAMATH_CALUDE_mary_uber_time_l1438_143823

/-- Mary's business trip timeline --/
def business_trip_timeline (t : ℕ) : Prop :=
  let uber_to_house := t
  let uber_to_airport := 5 * t
  let check_bag := 15
  let security := 3 * 15
  let wait_boarding := 20
  let wait_takeoff := 2 * 20
  uber_to_house + uber_to_airport + check_bag + security + wait_boarding + wait_takeoff = 180

theorem mary_uber_time : ∃ t : ℕ, business_trip_timeline t ∧ t = 10 := by
  sorry

end NUMINAMATH_CALUDE_mary_uber_time_l1438_143823


namespace NUMINAMATH_CALUDE_colin_speed_l1438_143858

/-- Given the relationships between the speeds of Colin, Brandon, Tony, Bruce, and Daniel,
    prove that Colin's speed is 8 miles per hour when Bruce's speed is 1 mile per hour. -/
theorem colin_speed (bruce tony brandon colin daniel : ℝ) : 
  bruce = 1 →
  tony = 2 * bruce →
  brandon = (1/3) * tony^2 →
  colin = 6 * brandon →
  daniel = (1/4) * colin →
  colin = 8 := by
sorry

end NUMINAMATH_CALUDE_colin_speed_l1438_143858


namespace NUMINAMATH_CALUDE_four_points_with_given_distances_l1438_143827

theorem four_points_with_given_distances : 
  ∃! (points : Finset (ℝ × ℝ)), 
    Finset.card points = 4 ∧ 
    (∀ p ∈ points, 
      (abs p.2 = 2 ∧ abs p.1 = 4)) ∧
    (∀ p : ℝ × ℝ, 
      (abs p.2 = 2 ∧ abs p.1 = 4) → p ∈ points) :=
by sorry

end NUMINAMATH_CALUDE_four_points_with_given_distances_l1438_143827


namespace NUMINAMATH_CALUDE_ratio_equality_l1438_143851

theorem ratio_equality : ∃ x : ℚ, (2 / 5 : ℚ) / (3 / 7 : ℚ) = x / (1 / 2 : ℚ) ∧ x = 7 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l1438_143851


namespace NUMINAMATH_CALUDE_part1_solution_set_part2_m_range_l1438_143848

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x - m| + |x + 3|

-- Part 1
theorem part1_solution_set :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} := by sorry

-- Part 2
theorem part2_m_range :
  {m : ℝ | ∃ x, f m x ≤ 2*m - 5} = {m : ℝ | m ≥ 8} := by sorry

end NUMINAMATH_CALUDE_part1_solution_set_part2_m_range_l1438_143848


namespace NUMINAMATH_CALUDE_percentage_not_covering_politics_l1438_143835

/-- Represents the percentage of reporters covering local politics in country X -/
def local_politics_coverage : ℝ := 10

/-- Represents the percentage of political reporters not covering local politics in country X -/
def non_local_political_coverage : ℝ := 30

/-- Represents the total number of reporters (assumed for calculation purposes) -/
def total_reporters : ℝ := 100

/-- Theorem stating that 86% of reporters do not cover politics -/
theorem percentage_not_covering_politics :
  (total_reporters - (local_politics_coverage / (100 - non_local_political_coverage) * 100)) / total_reporters * 100 = 86 := by
  sorry

end NUMINAMATH_CALUDE_percentage_not_covering_politics_l1438_143835


namespace NUMINAMATH_CALUDE_angle_identities_l1438_143847

theorem angle_identities (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.cos (α + π / 4) = Real.sqrt 3 / 3) : 
  Real.sin (α + 7 * π / 12) = (Real.sqrt 6 + 3) / 6 ∧ 
  Real.cos (2 * α + π / 6) = (2 * Real.sqrt 6 - 1) / 6 := by
sorry

end NUMINAMATH_CALUDE_angle_identities_l1438_143847


namespace NUMINAMATH_CALUDE_parabola_intersection_difference_l1438_143867

/-- The difference between the larger and smaller x-coordinates of the intersection points of two parabolas -/
theorem parabola_intersection_difference : ∃ (a c : ℝ),
  (∀ x : ℝ, 2 * x^2 - 4 * x + 4 = -x^2 - 2 * x + 4 → x = a ∨ x = c) ∧
  c ≥ a ∧
  c - a = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_difference_l1438_143867


namespace NUMINAMATH_CALUDE_min_connections_for_six_towns_l1438_143890

/-- The number of towns -/
def num_towns : ℕ := 6

/-- The formula for the number of connections in an undirected graph without loops -/
def connections (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: For 6 towns, the minimum number of connections needed is 15 -/
theorem min_connections_for_six_towns :
  connections num_towns = 15 := by sorry

end NUMINAMATH_CALUDE_min_connections_for_six_towns_l1438_143890


namespace NUMINAMATH_CALUDE_linear_equation_m_value_l1438_143881

theorem linear_equation_m_value : 
  ∃! m : ℤ, (abs m - 4 = 1) ∧ (m - 5 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_m_value_l1438_143881


namespace NUMINAMATH_CALUDE_binary_repr_25_l1438_143863

/-- The binary representation of a natural number -/
def binary_repr (n : ℕ) : List Bool :=
  if n = 0 then [false]
  else
    let rec go (m : ℕ) : List Bool :=
      if m = 0 then []
      else (m % 2 = 1) :: go (m / 2)
    go n

theorem binary_repr_25 : binary_repr 25 = [true, false, false, true, true] := by sorry

end NUMINAMATH_CALUDE_binary_repr_25_l1438_143863


namespace NUMINAMATH_CALUDE_line_touches_ellipse_l1438_143889

theorem line_touches_ellipse (a b : ℝ) (m : ℝ) (h1 : a = 3) (h2 : b = 1) :
  (∃! p : ℝ × ℝ, p.1^2 / a^2 + p.2^2 / b^2 = 1 ∧ p.2 = m * p.1 + 2) ↔ m^2 = 1/3 :=
sorry

end NUMINAMATH_CALUDE_line_touches_ellipse_l1438_143889


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1438_143876

-- Define the quadratic function f(x)
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the maximum value function h(a)
noncomputable def h (a b c : ℝ) : ℝ := 
  let x₀ := -b / (2 * a)
  f a b c x₀

-- Main theorem
theorem quadratic_function_properties
  (a b c : ℝ)
  (ha : a < 0)
  (hf : ∀ x, 1 < x ∧ x < 3 → f a b c x > -2 * x)
  (hz : ∃! x, f a b c x + 6 * a = 0)
  : 
  (a = -1/5 ∧ b = -6/5 ∧ c = -3/5) ∧
  (∀ a' b' c', h a' b' c' ≥ -2) ∧
  (∃ a₀ b₀ c₀, h a₀ b₀ c₀ = -2)
  := by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1438_143876


namespace NUMINAMATH_CALUDE_total_rent_is_105_l1438_143884

/-- Represents the rent share of a person -/
structure RentShare where
  oxen : ℕ
  months : ℕ
  share : ℚ

/-- Calculates the total rent of a pasture given the rent shares of three people -/
def calculate_total_rent (a b c : RentShare) : ℚ :=
  let total_oxen_months : ℕ := a.oxen * a.months + b.oxen * b.months + c.oxen * c.months
  let rent_per_oxen_month : ℚ := c.share / (c.oxen * c.months)
  rent_per_oxen_month * total_oxen_months

/-- Theorem: The total rent of the pasture is 105.00 given the specified conditions -/
theorem total_rent_is_105 (a b c : RentShare)
  (ha : a.oxen = 10 ∧ a.months = 7)
  (hb : b.oxen = 12 ∧ b.months = 5)
  (hc : c.oxen = 15 ∧ c.months = 3 ∧ c.share = 26.999999999999996) :
  calculate_total_rent a b c = 105 :=
by sorry

end NUMINAMATH_CALUDE_total_rent_is_105_l1438_143884


namespace NUMINAMATH_CALUDE_total_age_problem_l1438_143834

theorem total_age_problem (a b c : ℕ) : 
  b = 10 →
  a = b + 2 →
  b = 2 * c →
  a + b + c = 27 := by
sorry

end NUMINAMATH_CALUDE_total_age_problem_l1438_143834


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l1438_143843

theorem quadratic_solution_sum (x y : ℝ) : 
  x + y = 5 → x * y = 7/5 →
  ∃ (a b c d : ℝ), x = (a + b * Real.sqrt c) / d ∧ 
                    x = (a - b * Real.sqrt c) / d ∧
                    a + b + c + d = 521 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l1438_143843


namespace NUMINAMATH_CALUDE_sum_of_roots_zero_l1438_143814

theorem sum_of_roots_zero (a b c x y : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_eq1 : a^3 + a*x + y = 0)
  (h_eq2 : b^3 + b*x + y = 0)
  (h_eq3 : c^3 + c*x + y = 0) :
  a + b + c = 0 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_zero_l1438_143814


namespace NUMINAMATH_CALUDE_average_difference_l1438_143849

def even_integers_16_to_44 : List Int :=
  List.range 15 |> List.map (fun i => 16 + 2 * i)

def even_integers_14_to_56 : List Int :=
  List.range 22 |> List.map (fun i => 14 + 2 * i)

def average (l : List Int) : ℚ :=
  (l.sum : ℚ) / l.length

theorem average_difference :
  average even_integers_16_to_44 + 5 = average even_integers_14_to_56 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_l1438_143849


namespace NUMINAMATH_CALUDE_function_machine_output_l1438_143838

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 > 20 then
    step1 - 8
  else
    step1 + 10

theorem function_machine_output :
  function_machine 7 = 13 := by
  sorry

end NUMINAMATH_CALUDE_function_machine_output_l1438_143838


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_inequality_l1438_143857

theorem negation_of_proposition (P : ℝ → Prop) :
  (¬∀ x : ℝ, P x) ↔ (∃ x : ℝ, ¬P x) := by sorry

theorem negation_of_inequality :
  (¬∀ x : ℝ, x^2 + 1 ≥ 2*x) ↔ (∃ x : ℝ, x^2 + 1 < 2*x) := by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_inequality_l1438_143857


namespace NUMINAMATH_CALUDE_bookshelf_problem_l1438_143812

/-- Represents the unit price of bookshelf type A -/
def price_A : ℕ := sorry

/-- Represents the unit price of bookshelf type B -/
def price_B : ℕ := sorry

/-- Represents the maximum number of type B bookshelves that can be purchased -/
def max_B : ℕ := sorry

theorem bookshelf_problem :
  (3 * price_A + 2 * price_B = 1020) ∧
  (price_A + 3 * price_B = 900) ∧
  (∀ m : ℕ, m ≤ 20 → price_A * (20 - m) + price_B * m ≤ 4350) →
  (price_A = 180 ∧ price_B = 240 ∧ max_B = 12) :=
by sorry

end NUMINAMATH_CALUDE_bookshelf_problem_l1438_143812


namespace NUMINAMATH_CALUDE_nature_of_c_l1438_143831

theorem nature_of_c (a c : ℝ) (h : (2*a - 1) / (-3) < -(c + 1) / (-4)) :
  (c < 0 ∨ c > 0) ∧ c ≠ -1 :=
sorry

end NUMINAMATH_CALUDE_nature_of_c_l1438_143831


namespace NUMINAMATH_CALUDE_lottery_not_guaranteed_win_l1438_143807

/-- Represents the probability of not winning any ticket when buying n tickets -/
def prob_no_win (total : ℕ) (rate : ℚ) (n : ℕ) : ℚ :=
  (1 - rate) ^ n

theorem lottery_not_guaranteed_win (total : ℕ) (rate : ℚ) (n : ℕ) 
  (h_total : total = 100000)
  (h_rate : rate = 1 / 1000)
  (h_n : n = 2000) :
  prob_no_win total rate n > 0 := by
  sorry

#check lottery_not_guaranteed_win

end NUMINAMATH_CALUDE_lottery_not_guaranteed_win_l1438_143807


namespace NUMINAMATH_CALUDE_jims_pantry_flour_l1438_143893

/-- The amount of flour Jim has in the pantry -/
def flour_in_pantry : ℕ := sorry

/-- The total amount of flour Jim has in the cupboard and on the kitchen counter -/
def flour_elsewhere : ℕ := 300

/-- The amount of flour required for one loaf of bread -/
def flour_per_loaf : ℕ := 200

/-- The number of loaves Jim can bake -/
def loaves_baked : ℕ := 2

theorem jims_pantry_flour :
  flour_in_pantry = 100 :=
by sorry

end NUMINAMATH_CALUDE_jims_pantry_flour_l1438_143893


namespace NUMINAMATH_CALUDE_toy_store_revenue_ratio_l1438_143852

theorem toy_store_revenue_ratio :
  ∀ (december_revenue : ℚ),
  december_revenue > 0 →
  let november_revenue := (3 : ℚ) / 5 * december_revenue
  let january_revenue := (1 : ℚ) / 6 * november_revenue
  let average_revenue := (november_revenue + january_revenue) / 2
  december_revenue / average_revenue = 20 / 7 := by
sorry

end NUMINAMATH_CALUDE_toy_store_revenue_ratio_l1438_143852


namespace NUMINAMATH_CALUDE_pickled_vegetables_grade_C_l1438_143829

/-- Represents the number of boxes of pickled vegetables in each grade -/
structure GradeBoxes where
  A : ℕ
  B : ℕ
  C : ℕ

/-- 
Given:
- There are 420 boxes of pickled vegetables in total
- The vegetables are classified into three grades: A, B, and C
- m, n, and t are the number of boxes sampled from grades A, B, and C, respectively
- 2t = m + n

Prove that the number of boxes classified as grade C is 140
-/
theorem pickled_vegetables_grade_C (boxes : GradeBoxes) 
  (total_boxes : boxes.A + boxes.B + boxes.C = 420)
  (sample_relation : ∃ (m n t : ℕ), 2 * t = m + n) :
  boxes.C = 140 := by
  sorry

end NUMINAMATH_CALUDE_pickled_vegetables_grade_C_l1438_143829


namespace NUMINAMATH_CALUDE_pyramid_volume_in_unit_cube_l1438_143871

/-- The volume of a pyramid within a unit cube, where the pyramid's vertex is at one corner of the cube
    and its base is a triangle formed by the midpoints of three adjacent edges meeting at the opposite corner -/
theorem pyramid_volume_in_unit_cube : ∃ V : ℝ, V = Real.sqrt 3 / 24 :=
  sorry

end NUMINAMATH_CALUDE_pyramid_volume_in_unit_cube_l1438_143871


namespace NUMINAMATH_CALUDE_no_common_elements_l1438_143839

-- Define the sequence Pn(x)
def P : ℕ → (ℝ → ℝ)
  | 0 => λ x => x
  | 1 => λ x => 4 * x^3 + 3 * x
  | (n + 2) => λ x => (4 * x^2 + 2) * P (n + 1) x - P n x

-- Define the set A(m)
def A (m : ℝ) : Set ℝ := {y | ∃ n : ℕ, y = P n m}

-- Theorem statement
theorem no_common_elements (m : ℝ) (h : m > 0) :
  ∀ n k : ℕ, P n m ≠ P k (m + 4) :=
by sorry

end NUMINAMATH_CALUDE_no_common_elements_l1438_143839


namespace NUMINAMATH_CALUDE_probability_failed_math_given_failed_chinese_l1438_143882

theorem probability_failed_math_given_failed_chinese 
  (failed_math : ℝ) 
  (failed_chinese : ℝ) 
  (failed_both : ℝ) 
  (h1 : failed_math = 0.16) 
  (h2 : failed_chinese = 0.07) 
  (h3 : failed_both = 0.04) :
  failed_both / failed_chinese = 4 / 7 :=
by sorry

end NUMINAMATH_CALUDE_probability_failed_math_given_failed_chinese_l1438_143882


namespace NUMINAMATH_CALUDE_tanya_erasers_l1438_143888

/-- Given the number of erasers for Hanna, Rachel, and Tanya, prove that Tanya has 20 erasers -/
theorem tanya_erasers (h r t : ℕ) (tr : ℕ) : 
  h = 2 * r →  -- Hanna has twice as many erasers as Rachel
  r = tr / 2 - 3 →  -- Rachel has three less than one-half as many erasers as Tanya has red erasers
  tr = t / 2 →  -- Half of Tanya's erasers are red
  h = 4 →  -- Hanna has 4 erasers
  t = 20 := by sorry

end NUMINAMATH_CALUDE_tanya_erasers_l1438_143888


namespace NUMINAMATH_CALUDE_hex_count_and_sum_l1438_143826

/-- Converts a positive integer to its hexadecimal representation --/
def toHex (n : ℕ+) : List (Fin 16) := sorry

/-- Checks if a hexadecimal representation uses only digits 0-9 --/
def usesOnlyDigits (hex : List (Fin 16)) : Prop := sorry

/-- Counts numbers in [1, n] whose hexadecimal representation uses only digits 0-9 --/
def countOnlyDigits (n : ℕ+) : ℕ := sorry

/-- Computes the sum of digits of a natural number --/
def sumOfDigits (n : ℕ) : ℕ := sorry

theorem hex_count_and_sum :
  let count := countOnlyDigits 500
  count = 199 ∧ sumOfDigits count = 19 := by sorry

end NUMINAMATH_CALUDE_hex_count_and_sum_l1438_143826


namespace NUMINAMATH_CALUDE_age_difference_l1438_143804

/-- Given a man and his son, prove that the man is 35 years older than his son. -/
theorem age_difference (man_age son_age : ℕ) : 
  man_age > son_age →
  son_age = 33 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 35 := by
  sorry


end NUMINAMATH_CALUDE_age_difference_l1438_143804


namespace NUMINAMATH_CALUDE_intersection_with_complement_l1438_143832

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 2}
def B : Set Nat := {2, 3, 4}

theorem intersection_with_complement :
  A ∩ (U \ B) = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l1438_143832


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l1438_143845

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x + 2| ≤ 5} = {x : ℝ | -7 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l1438_143845


namespace NUMINAMATH_CALUDE_division_evaluation_l1438_143865

theorem division_evaluation : (180 : ℚ) / (12 + 13 * 2) = 90 / 19 := by sorry

end NUMINAMATH_CALUDE_division_evaluation_l1438_143865


namespace NUMINAMATH_CALUDE_parallelogram_area_32_18_l1438_143805

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 32 cm and height 18 cm is 576 square centimeters -/
theorem parallelogram_area_32_18 : parallelogram_area 32 18 = 576 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_32_18_l1438_143805


namespace NUMINAMATH_CALUDE_max_good_permutations_l1438_143800

/-- A sequence of points in the plane is "good" if no three points are collinear,
    the polyline is non-self-intersecting, and each triangle formed by three
    consecutive points is oriented counterclockwise. -/
def is_good_sequence (points : List (ℝ × ℝ)) : Prop :=
  sorry

/-- The number of distinct permutations of n points that form a good sequence -/
def num_good_permutations (n : ℕ) : ℕ :=
  sorry

/-- Theorem: For any integer n ≥ 3, the maximum number of distinct permutations
    of n points in the plane that form a "good" sequence is n^2 - 4n + 6. -/
theorem max_good_permutations (n : ℕ) (h : n ≥ 3) :
  num_good_permutations n = n^2 - 4*n + 6 :=
sorry

end NUMINAMATH_CALUDE_max_good_permutations_l1438_143800
