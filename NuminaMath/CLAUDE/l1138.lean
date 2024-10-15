import Mathlib

namespace NUMINAMATH_CALUDE_periodic_function_from_T_property_l1138_113820

-- Define the "T property" for a function
def has_T_property (f : ℝ → ℝ) (T : ℝ) : Prop :=
  T ≠ 0 ∧ ∀ x : ℝ, (deriv f) (x + T) = (deriv f) x

-- Main theorem
theorem periodic_function_from_T_property (f : ℝ → ℝ) (T M : ℝ) 
  (hf : Continuous f) 
  (hT : has_T_property f T) 
  (hM : ∀ x : ℝ, |f x| < M) :
  ∀ x : ℝ, f (x + T) = f x :=
sorry

end NUMINAMATH_CALUDE_periodic_function_from_T_property_l1138_113820


namespace NUMINAMATH_CALUDE_liner_and_water_speed_theorem_l1138_113893

/-- The distance between Chongqing and Shibaozhai in kilometers -/
def distance : ℝ := 270

/-- The time taken to travel downstream in hours -/
def downstream_time : ℝ := 9

/-- The time taken to travel upstream in hours -/
def upstream_time : ℝ := 13.5

/-- The speed of the liner in still water in km/h -/
def liner_speed : ℝ := 25

/-- The speed of the water flow in km/h -/
def water_speed : ℝ := 5

/-- The distance between Chongqing Port and the new dock in km -/
def new_dock_distance : ℝ := 162

theorem liner_and_water_speed_theorem :
  (downstream_time * (liner_speed + water_speed) = distance) ∧
  (upstream_time * (liner_speed - water_speed) = distance) ∧
  (new_dock_distance / (liner_speed + water_speed) = (distance - new_dock_distance) / (liner_speed - water_speed)) := by
  sorry

#check liner_and_water_speed_theorem

end NUMINAMATH_CALUDE_liner_and_water_speed_theorem_l1138_113893


namespace NUMINAMATH_CALUDE_mistake_correction_l1138_113858

theorem mistake_correction (a : ℤ) (h : 31 - a = 12) : 31 + a = 50 := by
  sorry

end NUMINAMATH_CALUDE_mistake_correction_l1138_113858


namespace NUMINAMATH_CALUDE_tree_growth_fraction_l1138_113865

/-- Represents the height of a tree over time -/
def tree_height (initial_height : ℕ) (growth_rate : ℕ) (years : ℕ) : ℕ :=
  initial_height + growth_rate * years

/-- The fraction representing the increase in height from year a to year b -/
def height_increase_fraction (initial_height : ℕ) (growth_rate : ℕ) (a b : ℕ) : ℚ :=
  (tree_height initial_height growth_rate b - tree_height initial_height growth_rate a) /
  tree_height initial_height growth_rate a

theorem tree_growth_fraction :
  height_increase_fraction 4 1 4 6 = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_tree_growth_fraction_l1138_113865


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1138_113874

-- Define sets A and B
def A : Set ℝ := {x | x > -1}
def B : Set ℝ := {x | x ≤ 5}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -1 < x ∧ x ≤ 5} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1138_113874


namespace NUMINAMATH_CALUDE_right_triangle_check_l1138_113833

def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a)

theorem right_triangle_check :
  ¬ is_right_triangle 1 2 3 ∧
  ¬ is_right_triangle 1 2 2 ∧
  ¬ is_right_triangle (Real.sqrt 2) (Real.sqrt 2) (Real.sqrt 2) ∧
  is_right_triangle 6 8 10 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_check_l1138_113833


namespace NUMINAMATH_CALUDE_largest_after_three_operations_obtainable_1999_l1138_113832

-- Define the expansion operation
def expand (a b : ℕ) : ℕ := a * b + a + b

-- Theorem for the largest number after three operations
theorem largest_after_three_operations :
  let step1 := expand 1 4
  let step2 := expand 4 step1
  let step3 := expand step1 step2
  step3 = 499 := by sorry

-- Theorem for the obtainability of 1999
theorem obtainable_1999 :
  ∃ (m n : ℕ), 2000 = 2^m * 5^n := by sorry

end NUMINAMATH_CALUDE_largest_after_three_operations_obtainable_1999_l1138_113832


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l1138_113889

theorem inequality_system_solution_set :
  let S := {x : ℝ | x - 2 ≥ -5 ∧ 3*x < x + 2}
  S = {x : ℝ | -3 ≤ x ∧ x < 1} := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l1138_113889


namespace NUMINAMATH_CALUDE_pauls_license_plate_earnings_l1138_113860

theorem pauls_license_plate_earnings 
  (total_states : ℕ) 
  (pauls_states : ℕ) 
  (total_earnings : ℚ) :
  total_states = 50 →
  pauls_states = 40 →
  total_earnings = 160 →
  (total_earnings / (pauls_states / total_states * 100 : ℚ)) = 2 :=
by sorry

end NUMINAMATH_CALUDE_pauls_license_plate_earnings_l1138_113860


namespace NUMINAMATH_CALUDE_problem_solution_l1138_113834

theorem problem_solution (p_xavier p_yvonne p_zelda : ℚ) 
  (h_xavier : p_xavier = 1/4)
  (h_yvonne : p_yvonne = 1/3)
  (h_zelda : p_zelda = 5/8) :
  p_xavier * p_yvonne * (1 - p_zelda) = 1/32 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1138_113834


namespace NUMINAMATH_CALUDE_quadratic_form_b_value_l1138_113803

theorem quadratic_form_b_value (b : ℝ) (n : ℝ) : 
  b < 0 →
  (∀ x, x^2 + b*x + 50 = (x + n)^2 + 16) →
  b = -2 * Real.sqrt 34 := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_b_value_l1138_113803


namespace NUMINAMATH_CALUDE_planes_parallel_if_perp_lines_parallel_l1138_113885

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism and perpendicularity relations
variable (parallel : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem planes_parallel_if_perp_lines_parallel
  (l m : Line) (α β : Plane)
  (h1 : perpendicular l α)
  (h2 : perpendicular m β)
  (h3 : parallel l m) :
  parallel_planes α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_perp_lines_parallel_l1138_113885


namespace NUMINAMATH_CALUDE_distinct_z_values_l1138_113818

def is_valid_number (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def reverse_digits (n : ℕ) : ℕ := 
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  100 * c + 10 * b + a

def z (x : ℕ) : ℕ := Int.natAbs (x - reverse_digits x)

theorem distinct_z_values : 
  ∃ (S : Finset ℕ), (∀ x, is_valid_number x → z x ∈ S) ∧ S.card = 10 := by
sorry

end NUMINAMATH_CALUDE_distinct_z_values_l1138_113818


namespace NUMINAMATH_CALUDE_unique_solution_condition_l1138_113821

theorem unique_solution_condition (c k : ℝ) (h_c : c ≠ 0) : 
  (∃! b : ℝ, b > 0 ∧ 
    (∃! x : ℝ, x^2 + (b + 1/b) * x + c = 0) ∧ 
    b^4 + (2 - 4*c) * b^2 + k = 0) ↔ 
  c = 1 := by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l1138_113821


namespace NUMINAMATH_CALUDE_cylinder_volume_increase_l1138_113804

theorem cylinder_volume_increase (R H : ℝ) (hR : R = 8) (hH : H = 3) :
  ∃ x : ℝ, x > 0 ∧
  ∃ C : ℝ, C > 0 ∧
  (Real.pi * (R + x)^2 * (H + x) = Real.pi * R^2 * H + C) →
  x = 16/3 := by
sorry

end NUMINAMATH_CALUDE_cylinder_volume_increase_l1138_113804


namespace NUMINAMATH_CALUDE_average_of_numbers_l1138_113814

def numbers : List ℕ := [12, 13, 14, 510, 530, 1115, 1120, 1, 1252140, 2345]

theorem average_of_numbers :
  (numbers.sum : ℚ) / numbers.length = 125790 := by sorry

end NUMINAMATH_CALUDE_average_of_numbers_l1138_113814


namespace NUMINAMATH_CALUDE_distinct_triangles_on_circle_l1138_113828

/-- The number of points marked on the circle -/
def n : ℕ := 12

/-- The number of vertices needed to form a triangle -/
def k : ℕ := 3

/-- The number of distinct triangles that can be drawn -/
def num_triangles : ℕ := Nat.choose n k

theorem distinct_triangles_on_circle :
  num_triangles = 220 := by sorry

end NUMINAMATH_CALUDE_distinct_triangles_on_circle_l1138_113828


namespace NUMINAMATH_CALUDE_tan_65_degrees_l1138_113845

/-- If tan 110° = α, then tan 65° = (α - 1) / (1 + α) -/
theorem tan_65_degrees (α : ℝ) (h : Real.tan (110 * π / 180) = α) :
  Real.tan (65 * π / 180) = (α - 1) / (α + 1) := by
  sorry

end NUMINAMATH_CALUDE_tan_65_degrees_l1138_113845


namespace NUMINAMATH_CALUDE_tim_speed_is_45_l1138_113883

/-- Represents the distance between Tim and Élan in miles -/
def initial_distance : ℝ := 150

/-- Represents Élan's initial speed in mph -/
def elan_initial_speed : ℝ := 5

/-- Represents the distance Tim travels until meeting Élan in miles -/
def tim_travel_distance : ℝ := 100

/-- Represents the number of hours until Tim and Élan meet -/
def meeting_time : ℕ := 2

/-- Represents Tim's initial speed in mph -/
def tim_initial_speed : ℝ := 45

/-- Theorem stating that given the conditions, Tim's initial speed is 45 mph -/
theorem tim_speed_is_45 :
  tim_initial_speed * (2^meeting_time - 1) = initial_distance - elan_initial_speed * (2^meeting_time - 1) :=
sorry

end NUMINAMATH_CALUDE_tim_speed_is_45_l1138_113883


namespace NUMINAMATH_CALUDE_symmetric_even_function_value_l1138_113882

/-- A function that is symmetric about x=2 -/
def SymmetricAbout2 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (4 - x) = f x

/-- An even function -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

theorem symmetric_even_function_value (f : ℝ → ℝ) 
  (h_sym : SymmetricAbout2 f) (h_even : EvenFunction f) (h_val : f 3 = 3) : 
  f (-1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_even_function_value_l1138_113882


namespace NUMINAMATH_CALUDE_sum_of_max_min_f_l1138_113875

noncomputable def f (x : ℝ) : ℝ := 1 + (Real.sin x) / (2 + Real.cos x)

theorem sum_of_max_min_f : 
  (⨆ (x : ℝ), f x) + (⨅ (x : ℝ), f x) = 2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_max_min_f_l1138_113875


namespace NUMINAMATH_CALUDE_martas_textbook_cost_l1138_113898

/-- The total cost of Marta's textbooks --/
def total_cost (sale_price : ℕ) (sale_quantity : ℕ) (online_cost : ℕ) (bookstore_multiplier : ℕ) : ℕ :=
  sale_price * sale_quantity + online_cost + bookstore_multiplier * online_cost

/-- Theorem stating the total cost of Marta's textbooks --/
theorem martas_textbook_cost :
  total_cost 10 5 40 3 = 210 := by
  sorry

end NUMINAMATH_CALUDE_martas_textbook_cost_l1138_113898


namespace NUMINAMATH_CALUDE_polynomial_symmetry_condition_l1138_113815

/-- A polynomial function of degree 4 -/
def polynomial (a b c d e : ℝ) (x : ℝ) : ℝ :=
  a * x^4 + b * x^3 + c * x^2 + d * x + e

/-- Symmetry condition for a function -/
def isSymmetric (f : ℝ → ℝ) : Prop :=
  ∃ t : ℝ, ∀ x : ℝ, f x = f (2 * t - x)

theorem polynomial_symmetry_condition
  (a b c d e : ℝ) (h : a ≠ 0) :
  isSymmetric (polynomial a b c d e) ↔ b^3 - a*b*c + 8*a^2*d = 0 :=
sorry

end NUMINAMATH_CALUDE_polynomial_symmetry_condition_l1138_113815


namespace NUMINAMATH_CALUDE_smallest_period_of_given_functions_l1138_113887

open Real

noncomputable def f1 (x : ℝ) := -cos x
noncomputable def f2 (x : ℝ) := abs (sin x)
noncomputable def f3 (x : ℝ) := cos (2 * x)
noncomputable def f4 (x : ℝ) := tan (2 * x - π / 4)

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def smallest_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  is_periodic f p ∧ p > 0 ∧ ∀ q, is_periodic f q ∧ q > 0 → p ≤ q

theorem smallest_period_of_given_functions :
  smallest_positive_period f2 π ∧
  smallest_positive_period f3 π ∧
  (∀ p, smallest_positive_period f1 p → p > π) ∧
  (∀ p, smallest_positive_period f4 p → p > π) :=
sorry

end NUMINAMATH_CALUDE_smallest_period_of_given_functions_l1138_113887


namespace NUMINAMATH_CALUDE_largest_multiple_of_seven_l1138_113899

theorem largest_multiple_of_seven (n : ℤ) : n = 147 ↔ 
  (∃ k : ℤ, n = 7 * k) ∧ 
  (-n > -150) ∧ 
  (∀ m : ℤ, (∃ j : ℤ, m = 7 * j) ∧ (-m > -150) → m ≤ n) := by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_seven_l1138_113899


namespace NUMINAMATH_CALUDE_fraction_equality_l1138_113823

theorem fraction_equality (p q r s : ℝ) 
  (h : (p - q) * (r - s) / ((q - r) * (s - p)) = 3 / 7) : 
  (p - r) * (q - s) / ((p - q) * (r - s)) = -3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1138_113823


namespace NUMINAMATH_CALUDE_gulbis_count_l1138_113807

/-- The number of gulbis in one dureum -/
def fish_per_dureum : ℕ := 20

/-- The number of dureums of gulbis -/
def num_dureums : ℕ := 156

/-- The total number of gulbis -/
def total_gulbis : ℕ := num_dureums * fish_per_dureum

theorem gulbis_count : total_gulbis = 3120 := by
  sorry

end NUMINAMATH_CALUDE_gulbis_count_l1138_113807


namespace NUMINAMATH_CALUDE_angle_I_measures_138_l1138_113806

/-- A convex pentagon with specific angle properties -/
structure ConvexPentagon where
  -- Angles in degrees
  F : ℝ
  G : ℝ
  H : ℝ
  I : ℝ
  J : ℝ
  -- Angle sum in a pentagon is 540°
  sum_eq_540 : F + G + H + I + J = 540
  -- Angles F, G, and H are congruent
  F_eq_G : F = G
  G_eq_H : G = H
  -- Angles I and J are congruent
  I_eq_J : I = J
  -- Angle F is 50° less than angle I
  F_eq_I_minus_50 : F = I - 50

/-- Theorem: In a convex pentagon with the given properties, angle I measures 138° -/
theorem angle_I_measures_138 (p : ConvexPentagon) : p.I = 138 := by
  sorry

end NUMINAMATH_CALUDE_angle_I_measures_138_l1138_113806


namespace NUMINAMATH_CALUDE_lee_cookies_l1138_113884

/-- Given that Lee can make 24 cookies with 3 cups of flour, 
    this function calculates how many cookies he can make with any number of cups. -/
def cookies_per_cups (cups : ℚ) : ℚ :=
  (24 / 3) * cups

/-- Theorem stating that Lee can make 40 cookies with 5 cups of flour. -/
theorem lee_cookies : cookies_per_cups 5 = 40 := by
  sorry

end NUMINAMATH_CALUDE_lee_cookies_l1138_113884


namespace NUMINAMATH_CALUDE_horner_method_operations_l1138_113873

-- Define the polynomial
def f (x : ℝ) : ℝ := 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x + 1

-- Define Horner's method for this specific polynomial
def horner_method (x : ℝ) : ℝ := ((((5 * x + 4) * x + 3) * x + 2) * x + 1) * x + 1

-- Theorem statement
theorem horner_method_operations :
  ∃ (mult_ops add_ops : ℕ),
    (∀ x : ℝ, f x = horner_method x) ∧
    mult_ops = 5 ∧
    add_ops = 5 :=
sorry

end NUMINAMATH_CALUDE_horner_method_operations_l1138_113873


namespace NUMINAMATH_CALUDE_bella_stamps_l1138_113878

/-- The number of stamps Bella bought -/
def total_stamps (snowflake truck rose : ℕ) : ℕ := snowflake + truck + rose

/-- Theorem stating the total number of stamps Bella bought -/
theorem bella_stamps : ∃ (snowflake truck rose : ℕ),
  snowflake = 11 ∧
  truck = snowflake + 9 ∧
  rose = truck - 13 ∧
  total_stamps snowflake truck rose = 38 := by
  sorry

end NUMINAMATH_CALUDE_bella_stamps_l1138_113878


namespace NUMINAMATH_CALUDE_expression_value_l1138_113824

theorem expression_value (a b : ℚ) (h : 4 * b = 3 + 4 * a) :
  a + (a - (a - (a - b) - b) - b) - b = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1138_113824


namespace NUMINAMATH_CALUDE_smallest_n_value_l1138_113831

/-- The number of ordered triplets (a, b, c) satisfying the conditions -/
def num_triplets : ℕ := 27000

/-- The greatest common divisor of a, b, and c -/
def gcd_value : ℕ := 91

/-- A function that counts the number of valid triplets for a given n -/
noncomputable def count_triplets (n : ℕ) : ℕ :=
  sorry

/-- The theorem stating the smallest possible value of n -/
theorem smallest_n_value :
  ∃ (n : ℕ), n = 17836000 ∧
  count_triplets n = num_triplets ∧
  (∀ m : ℕ, m < n → count_triplets m ≠ num_triplets) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_value_l1138_113831


namespace NUMINAMATH_CALUDE_one_third_greater_than_decimal_l1138_113888

theorem one_third_greater_than_decimal : 
  ∃ (ε : ℚ), ε > 0 ∧ ε = 1 / (3 * 10^9) ∧ 1/3 = 0.333333333 + ε := by
  sorry

end NUMINAMATH_CALUDE_one_third_greater_than_decimal_l1138_113888


namespace NUMINAMATH_CALUDE_ranch_problem_l1138_113813

theorem ranch_problem : ∃! (s c : ℕ), s > 0 ∧ c > 0 ∧ 35 * s + 40 * c = 1200 ∧ c > s := by
  sorry

end NUMINAMATH_CALUDE_ranch_problem_l1138_113813


namespace NUMINAMATH_CALUDE_Q_equals_two_three_four_l1138_113896

-- Define the set P
def P : Set ℕ := {1, 2}

-- Define the set Q
def Q : Set ℕ := {z | ∃ x y, x ∈ P ∧ y ∈ P ∧ z = x + y}

-- Theorem statement
theorem Q_equals_two_three_four : Q = {2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_Q_equals_two_three_four_l1138_113896


namespace NUMINAMATH_CALUDE_ball_volume_ratio_l1138_113859

theorem ball_volume_ratio :
  ∀ (x y z : ℝ),
    x > 0 → y > 0 → z > 0 →
    x = 3 * (y - x) →
    z - y = 3 * x →
    ∃ (k : ℝ), k > 0 ∧ x = 3 * k ∧ y = 4 * k ∧ z = 13 * k :=
by sorry

end NUMINAMATH_CALUDE_ball_volume_ratio_l1138_113859


namespace NUMINAMATH_CALUDE_symmetric_function_periodic_l1138_113869

/-- A function f: ℝ → ℝ satisfying certain symmetry properties -/
def symmetricFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (x + 2) = f (2 - x)) ∧ (∀ x, f (x + 7) = f (7 - x))

/-- Theorem stating that a symmetric function is periodic with period 10 -/
theorem symmetric_function_periodic (f : ℝ → ℝ) (h : symmetricFunction f) :
  ∀ x, f (x + 10) = f x := by
  sorry

end NUMINAMATH_CALUDE_symmetric_function_periodic_l1138_113869


namespace NUMINAMATH_CALUDE_function_transformation_l1138_113805

theorem function_transformation (f : ℝ → ℝ) (h : f 1 = 0) : 
  f 1 + 1 = 1 := by
sorry

end NUMINAMATH_CALUDE_function_transformation_l1138_113805


namespace NUMINAMATH_CALUDE_B_max_at_50_l1138_113849

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Definition of B_k -/
def B (k : ℕ) : ℝ := (binomial 500 k) * (0.1 ^ k)

/-- Statement: B_k is largest when k = 50 -/
theorem B_max_at_50 : ∀ k : ℕ, k ≤ 500 → B k ≤ B 50 := by sorry

end NUMINAMATH_CALUDE_B_max_at_50_l1138_113849


namespace NUMINAMATH_CALUDE_haley_magazines_l1138_113822

theorem haley_magazines 
  (num_boxes : ℕ) 
  (magazines_per_box : ℕ) 
  (h1 : num_boxes = 7)
  (h2 : magazines_per_box = 9) :
  num_boxes * magazines_per_box = 63 := by
  sorry

end NUMINAMATH_CALUDE_haley_magazines_l1138_113822


namespace NUMINAMATH_CALUDE_total_cost_special_requirement_l1138_113895

/-- The number of ways to choose 3 consecutive numbers from 01 to 10 -/
def consecutive_three_from_ten : Nat := 8

/-- The number of ways to choose 2 consecutive numbers from 11 to 20 -/
def consecutive_two_from_ten : Nat := 9

/-- The number of ways to choose 1 number from 21 to 30 -/
def one_from_ten : Nat := 10

/-- The number of ways to choose 1 number from 31 to 36 -/
def one_from_six : Nat := 6

/-- The cost of a single entry in yuan -/
def entry_cost : Nat := 2

/-- Theorem: The total cost of purchasing all possible entries meeting the special requirement is 8640 yuan -/
theorem total_cost_special_requirement : 
  consecutive_three_from_ten * consecutive_two_from_ten * one_from_ten * one_from_six * entry_cost = 8640 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_special_requirement_l1138_113895


namespace NUMINAMATH_CALUDE_coin_order_correct_l1138_113819

/-- Represents the set of coins --/
inductive Coin : Type
  | A | B | C | D | E | F

/-- Defines the covering relation between coins --/
def covers (x y : Coin) : Prop := sorry

/-- The correct order of coins from top to bottom --/
def correct_order : List Coin := [Coin.F, Coin.D, Coin.A, Coin.E, Coin.B, Coin.C]

/-- Theorem stating that the given order is correct based on the covering relations --/
theorem coin_order_correct :
  (∀ x : Coin, ¬covers x Coin.F) ∧
  (covers Coin.F Coin.D) ∧
  (covers Coin.D Coin.B) ∧ (covers Coin.D Coin.C) ∧ (covers Coin.D Coin.E) ∧
  (covers Coin.D Coin.A) ∧ (covers Coin.A Coin.B) ∧ (covers Coin.A Coin.C) ∧
  (covers Coin.D Coin.E) ∧ (covers Coin.E Coin.C) ∧
  (covers Coin.D Coin.B) ∧ (covers Coin.A Coin.B) ∧ (covers Coin.E Coin.B) ∧ (covers Coin.B Coin.C) ∧
  (∀ x : Coin, x ≠ Coin.C → covers x Coin.C) →
  correct_order = [Coin.F, Coin.D, Coin.A, Coin.E, Coin.B, Coin.C] :=
by sorry

end NUMINAMATH_CALUDE_coin_order_correct_l1138_113819


namespace NUMINAMATH_CALUDE_sum_of_angles_complex_roots_l1138_113802

theorem sum_of_angles_complex_roots (z₁ z₂ z₃ z₄ : ℂ) (r₁ r₂ r₃ r₄ : ℝ) (θ₁ θ₂ θ₃ θ₄ : ℝ) :
  z₁^4 = -16*I ∧ z₂^4 = -16*I ∧ z₃^4 = -16*I ∧ z₄^4 = -16*I ∧
  z₁ = r₁ * (Complex.cos θ₁ + Complex.I * Complex.sin θ₁) ∧
  z₂ = r₂ * (Complex.cos θ₂ + Complex.I * Complex.sin θ₂) ∧
  z₃ = r₃ * (Complex.cos θ₃ + Complex.I * Complex.sin θ₃) ∧
  z₄ = r₄ * (Complex.cos θ₄ + Complex.I * Complex.sin θ₄) ∧
  r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0 ∧ r₄ > 0 ∧
  0 ≤ θ₁ ∧ θ₁ < 2*π ∧
  0 ≤ θ₂ ∧ θ₂ < 2*π ∧
  0 ≤ θ₃ ∧ θ₃ < 2*π ∧
  0 ≤ θ₄ ∧ θ₄ < 2*π →
  θ₁ + θ₂ + θ₃ + θ₄ = (810 * π) / 180 := by sorry

end NUMINAMATH_CALUDE_sum_of_angles_complex_roots_l1138_113802


namespace NUMINAMATH_CALUDE_weight_estimation_l1138_113886

-- Define the variables and constants
variable (x y : ℝ)
variable (x_sum y_sum : ℝ)
variable (b_hat : ℝ)
variable (n : ℕ)

-- Define the conditions
def conditions (x_sum y_sum b_hat : ℝ) (n : ℕ) : Prop :=
  x_sum = 1600 ∧ y_sum = 460 ∧ b_hat = 0.85 ∧ n = 10

-- Define the regression line equation
def regression_line (x b_hat a_hat : ℝ) : ℝ :=
  b_hat * x + a_hat

-- Theorem statement
theorem weight_estimation 
  (x_sum y_sum b_hat : ℝ) (n : ℕ) 
  (h : conditions x_sum y_sum b_hat n) : 
  ∃ a_hat : ℝ, regression_line 170 b_hat a_hat = 54.5 :=
sorry

end NUMINAMATH_CALUDE_weight_estimation_l1138_113886


namespace NUMINAMATH_CALUDE_bowling_ball_weight_proof_l1138_113853

/-- The weight of one bowling ball in pounds -/
def bowling_ball_weight : ℝ := 18.75

/-- The weight of one kayak in pounds -/
def kayak_weight : ℝ := 30

theorem bowling_ball_weight_proof :
  (∀ (b k : ℝ), 8 * b = 5 * k ∧ 4 * k = 120 → b = bowling_ball_weight) :=
by sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_proof_l1138_113853


namespace NUMINAMATH_CALUDE_banana_price_reduction_l1138_113811

/-- Calculates the reduced price per dozen bananas given the original price and quantity change --/
def reduced_price_per_dozen (original_price : ℝ) (original_quantity : ℕ) : ℝ :=
  let reduced_price := 0.6 * original_price
  let new_quantity := original_quantity + 50
  let price_per_banana := 40 / new_quantity
  12 * price_per_banana

/-- Theorem stating the conditions and the result to be proved --/
theorem banana_price_reduction 
  (original_price : ℝ) 
  (original_quantity : ℕ) 
  (h1 : original_price * original_quantity = 40) 
  (h2 : 0.6 * original_price * (original_quantity + 50) = 40) :
  reduced_price_per_dozen original_price original_quantity = 3.84 :=
by sorry

#eval reduced_price_per_dozen (40 / 75) 75

end NUMINAMATH_CALUDE_banana_price_reduction_l1138_113811


namespace NUMINAMATH_CALUDE_range_of_a_l1138_113839

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, 4^x - a*2^(x+1) + a^2 - 1 ≥ 0) ↔ 
  a ∈ Set.Iic 1 ∪ Set.Ici 5 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l1138_113839


namespace NUMINAMATH_CALUDE_chocolate_bar_count_l1138_113838

/-- Represents a box containing chocolate bars -/
structure ChocolateBox where
  bars : ℕ

/-- Represents a large box containing small boxes of chocolates -/
structure LargeBox where
  smallBoxes : ℕ
  smallBoxContents : ChocolateBox

/-- Calculates the total number of chocolate bars in a large box -/
def totalChocolateBars (box : LargeBox) : ℕ :=
  box.smallBoxes * box.smallBoxContents.bars

theorem chocolate_bar_count (largeBox : LargeBox) 
    (h1 : largeBox.smallBoxes = 15)
    (h2 : largeBox.smallBoxContents.bars = 20) : 
    totalChocolateBars largeBox = 300 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_count_l1138_113838


namespace NUMINAMATH_CALUDE_charity_plates_delivered_l1138_113827

/-- The number of plates delivered by a charity given the cost of ingredients and total spent -/
theorem charity_plates_delivered (rice_cost chicken_cost total_spent : ℚ) : 
  rice_cost = 1/10 →
  chicken_cost = 4/10 →
  total_spent = 50 →
  (total_spent / (rice_cost + chicken_cost) : ℚ) = 100 := by
  sorry

end NUMINAMATH_CALUDE_charity_plates_delivered_l1138_113827


namespace NUMINAMATH_CALUDE_rectangle_formations_l1138_113868

def horizontal_lines : ℕ := 5
def vertical_lines : ℕ := 4

def choose (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

theorem rectangle_formations :
  (choose horizontal_lines 2) * (choose vertical_lines 2) = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_formations_l1138_113868


namespace NUMINAMATH_CALUDE_log_sum_equality_l1138_113897

theorem log_sum_equality (a b c : ℝ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  (1 / (1 + Real.log (c/a) / Real.log (a^2 * b))) +
  (1 / (1 + Real.log (a/b) / Real.log (b^2 * c))) +
  (1 / (1 + Real.log (b/c) / Real.log (c^2 * a))) = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equality_l1138_113897


namespace NUMINAMATH_CALUDE_saline_solution_water_calculation_l1138_113816

/-- Given a saline solution mixture, calculate the amount of water needed for a larger volume -/
theorem saline_solution_water_calculation 
  (salt_solution : ℝ) 
  (initial_water : ℝ) 
  (initial_total : ℝ) 
  (final_volume : ℝ) 
  (h1 : salt_solution = 0.05)
  (h2 : initial_water = 0.03)
  (h3 : initial_total = salt_solution + initial_water)
  (h4 : final_volume = 0.64) :
  final_volume * (initial_water / initial_total) = 0.24 := by
sorry

end NUMINAMATH_CALUDE_saline_solution_water_calculation_l1138_113816


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1138_113855

/-- Given a train of length 150 meters passing an oak tree in 9.99920006399488 seconds,
    prove that its speed is 54.00287976961843 km/hr. -/
theorem train_speed_calculation (train_length : Real) (time_to_pass : Real) :
  train_length = 150 →
  time_to_pass = 9.99920006399488 →
  (train_length / time_to_pass) * 3.6 = 54.00287976961843 := by
  sorry

#eval (150 / 9.99920006399488) * 3.6

end NUMINAMATH_CALUDE_train_speed_calculation_l1138_113855


namespace NUMINAMATH_CALUDE_girls_to_boys_ratio_l1138_113861

/-- Proves that the ratio of girls to boys is 4:5 given the class conditions -/
theorem girls_to_boys_ratio (total_students : ℕ) (boys : ℕ) (cups_per_boy : ℕ) (total_cups : ℕ) :
  total_students = 30 →
  boys = 10 →
  cups_per_boy = 5 →
  total_cups = 90 →
  (total_students - boys, boys) = (20, 10) ∧ 
  (20 : ℚ) / 10 = 4 / 5 := by
  sorry

#check girls_to_boys_ratio

end NUMINAMATH_CALUDE_girls_to_boys_ratio_l1138_113861


namespace NUMINAMATH_CALUDE_triangle_sinB_sinC_l1138_113894

theorem triangle_sinB_sinC (a b c : Real) (A B C : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  -- 2c + b = 2a * cos B
  (2 * c + b = 2 * a * Real.cos B) →
  -- Area S = 3/2 * sqrt(3)
  (1/2 * b * c * Real.sin A = 3/2 * Real.sqrt 3) →
  -- c = 2
  (c = 2) →
  -- Then sin B * sin C = 9/38
  (Real.sin B * Real.sin C = 9/38) := by
sorry

end NUMINAMATH_CALUDE_triangle_sinB_sinC_l1138_113894


namespace NUMINAMATH_CALUDE_f_properties_l1138_113852

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

def monotonic_intervals (a : ℝ) : Prop :=
  (a ≤ 0 → ∀ x y, 0 < x ∧ x < y → f a x < f a y) ∧
  (a > 0 → (∀ x y, 0 < x ∧ x < y ∧ y < 1/a → f a x < f a y) ∧
           (∀ x y, 1/a < x ∧ x < y → f a y < f a x))

def minimum_value (a : ℝ) : ℝ :=
  if a ≥ 1 then f a 2
  else if 0 < a ∧ a < 1/2 then f a 1
  else min (f a 1) (f a 2)

theorem f_properties (a : ℝ) :
  monotonic_intervals a ∧
  (a > 0 → ∀ x, x ∈ Set.Icc 1 2 → f a x ≥ minimum_value a) :=
sorry

end

end NUMINAMATH_CALUDE_f_properties_l1138_113852


namespace NUMINAMATH_CALUDE_sector_area_l1138_113877

/-- The area of a circular sector with a central angle of 60° and a radius of 10 cm is 50π/3 cm². -/
theorem sector_area (θ : Real) (r : Real) (h1 : θ = 60 * π / 180) (h2 : r = 10) :
  (θ / (2 * π)) * (π * r^2) = 50 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l1138_113877


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l1138_113879

theorem trigonometric_inequality : ∀ (a b c : ℝ),
  a = Real.sin (4/5) →
  b = Real.cos (4/5) →
  c = Real.tan (4/5) →
  c > a ∧ a > b :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l1138_113879


namespace NUMINAMATH_CALUDE_star_value_for_specific_conditions_l1138_113864

-- Define the * operation for non-zero integers
def star (a b : ℤ) : ℚ := (a : ℚ)⁻¹ + (b : ℚ)⁻¹

-- Theorem statement
theorem star_value_for_specific_conditions (a b : ℤ) 
  (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 12) (h4 : a * b = 35) :
  star a b = 12 / 35 := by
  sorry

end NUMINAMATH_CALUDE_star_value_for_specific_conditions_l1138_113864


namespace NUMINAMATH_CALUDE_joans_missed_games_l1138_113867

/-- Given that Joan's high school played 864 baseball games and she attended 395 games,
    prove that she missed 469 games. -/
theorem joans_missed_games (total_games : ℕ) (attended_games : ℕ)
  (h1 : total_games = 864)
  (h2 : attended_games = 395) :
  total_games - attended_games = 469 := by
sorry

end NUMINAMATH_CALUDE_joans_missed_games_l1138_113867


namespace NUMINAMATH_CALUDE_sin_2alpha_values_l1138_113872

theorem sin_2alpha_values (α : Real) (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : 3 * Real.cos (2 * α) = Real.sin (π / 4 - α)) :
  Real.sin (2 * α) = 1 ∨ Real.sin (2 * α) = -17/18 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_values_l1138_113872


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1138_113842

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x^2 - 2*x < 0}

def B : Set ℝ := {x | x > 1}

theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1138_113842


namespace NUMINAMATH_CALUDE_fat_thin_eating_time_l1138_113876

/-- The time it takes for two people to eat a certain amount of fruit together -/
def eating_time (rate1 rate2 amount : ℚ) : ℚ :=
  amount / (rate1 + rate2)

/-- Theorem: Mr. Fat and Mr. Thin will take 46.875 minutes to eat 5 pounds of fruit together -/
theorem fat_thin_eating_time :
  let fat_rate : ℚ := 1 / 15  -- Mr. Fat's eating rate in pounds per minute
  let thin_rate : ℚ := 1 / 25 -- Mr. Thin's eating rate in pounds per minute
  let amount : ℚ := 5         -- Amount of fruit in pounds
  eating_time fat_rate thin_rate amount = 46875 / 1000 := by
sorry

end NUMINAMATH_CALUDE_fat_thin_eating_time_l1138_113876


namespace NUMINAMATH_CALUDE_percentage_prefer_corn_l1138_113850

def kids_prefer_peas : ℕ := 6
def kids_prefer_carrots : ℕ := 9
def kids_prefer_corn : ℕ := 5

def total_kids : ℕ := kids_prefer_peas + kids_prefer_carrots + kids_prefer_corn

theorem percentage_prefer_corn : 
  (kids_prefer_corn : ℚ) / (total_kids : ℚ) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_prefer_corn_l1138_113850


namespace NUMINAMATH_CALUDE_abc_inequality_l1138_113826

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a * b * c ≥ (a + b - c) * (b + c - a) * (c + a - b) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l1138_113826


namespace NUMINAMATH_CALUDE_cube_difference_prime_factor_l1138_113863

theorem cube_difference_prime_factor (a b p : ℕ) : 
  Nat.Prime p → a^3 - b^3 = 633 * p → a = 16 ∧ b = 13 :=
by sorry

end NUMINAMATH_CALUDE_cube_difference_prime_factor_l1138_113863


namespace NUMINAMATH_CALUDE_dihydrogen_monoxide_weight_is_18_016_l1138_113856

/-- The atomic weight of hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1.008

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of hydrogen atoms in a water molecule -/
def hydrogen_count : ℕ := 2

/-- The number of oxygen atoms in a water molecule -/
def oxygen_count : ℕ := 1

/-- The molecular weight of dihydrogen monoxide (H2O) in g/mol -/
def dihydrogen_monoxide_weight : ℝ := 
  hydrogen_count * hydrogen_weight + oxygen_count * oxygen_weight

/-- Theorem: The molecular weight of dihydrogen monoxide (H2O) is 18.016 g/mol -/
theorem dihydrogen_monoxide_weight_is_18_016 : 
  dihydrogen_monoxide_weight = 18.016 := by
  sorry

end NUMINAMATH_CALUDE_dihydrogen_monoxide_weight_is_18_016_l1138_113856


namespace NUMINAMATH_CALUDE_certain_number_problem_l1138_113810

theorem certain_number_problem : ∃ x : ℤ, (3005 - 3000 + x = 2705) ∧ (x = 2700) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1138_113810


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_l1138_113854

-- Define the ellipse parameters
def a : ℝ := 2
def b : ℝ := 1

-- Define the point M
def M : ℝ × ℝ := (2, 1)

-- Define the ellipse equation
def is_on_ellipse (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define a line passing through M
def line_through_M (k : ℝ) (x y : ℝ) : Prop :=
  y - M.2 = k * (x - M.1)

-- Define the midpoint condition
def is_midpoint (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  M.1 = (x₁ + x₂) / 2 ∧ M.2 = (y₁ + y₂) / 2

-- Theorem statement
theorem ellipse_line_intersection :
  ∃ (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ),
    is_on_ellipse x₁ y₁ ∧
    is_on_ellipse x₂ y₂ ∧
    line_through_M k x₁ y₁ ∧
    line_through_M k x₂ y₂ ∧
    is_midpoint x₁ y₁ x₂ y₂ ∧
    k = -1/2 :=
  sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_l1138_113854


namespace NUMINAMATH_CALUDE_race_speed_ratio_l1138_113835

theorem race_speed_ratio (race_distance : ℕ) (head_start : ℕ) (win_margin : ℕ) :
  race_distance = 500 →
  head_start = 140 →
  win_margin = 20 →
  (race_distance - head_start : ℚ) / (race_distance - win_margin : ℚ) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_race_speed_ratio_l1138_113835


namespace NUMINAMATH_CALUDE_floor_of_7_9_l1138_113851

theorem floor_of_7_9 : ⌊(7.9 : ℝ)⌋ = 7 := by sorry

end NUMINAMATH_CALUDE_floor_of_7_9_l1138_113851


namespace NUMINAMATH_CALUDE_arctan_inequality_implies_a_nonnegative_l1138_113866

theorem arctan_inequality_implies_a_nonnegative (a : ℝ) : 
  (∀ x : ℝ, Real.arctan (Real.sqrt (x^2 + x + 13/4)) ≥ π/3 - a) → a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_arctan_inequality_implies_a_nonnegative_l1138_113866


namespace NUMINAMATH_CALUDE_total_cost_calculation_l1138_113829

def cabin_cost : ℕ := 6000
def land_cost_multiplier : ℕ := 4

theorem total_cost_calculation :
  cabin_cost + land_cost_multiplier * cabin_cost = 30000 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l1138_113829


namespace NUMINAMATH_CALUDE_gcd_2505_7350_l1138_113841

theorem gcd_2505_7350 : Nat.gcd 2505 7350 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2505_7350_l1138_113841


namespace NUMINAMATH_CALUDE_average_of_series_l1138_113837

/-- The average value of the series 0², (2z)², (4z)², (8z)² is 21z² -/
theorem average_of_series (z : ℝ) : 
  (0^2 + (2*z)^2 + (4*z)^2 + (8*z)^2) / 4 = 21 * z^2 := by
  sorry

end NUMINAMATH_CALUDE_average_of_series_l1138_113837


namespace NUMINAMATH_CALUDE_rancher_cows_count_l1138_113848

theorem rancher_cows_count (horses : ℕ) (cows : ℕ) : 
  cows = 5 * horses →
  cows + horses = 168 →
  cows = 140 := by
sorry

end NUMINAMATH_CALUDE_rancher_cows_count_l1138_113848


namespace NUMINAMATH_CALUDE_nth_root_inequality_l1138_113825

theorem nth_root_inequality (m n : ℕ) (h1 : m > n) (h2 : n ≥ 2) :
  (m : ℝ) ^ (1 / n : ℝ) - (n : ℝ) ^ (1 / m : ℝ) > 1 / (m * n : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_nth_root_inequality_l1138_113825


namespace NUMINAMATH_CALUDE_jerry_mowing_fraction_l1138_113809

def total_lawn_area : ℝ := 8
def riding_mower_rate : ℝ := 2
def push_mower_rate : ℝ := 1
def total_mowing_time : ℝ := 5

theorem jerry_mowing_fraction :
  ∃ x : ℝ,
    x ≥ 0 ∧ x ≤ 1 ∧
    (riding_mower_rate * x * total_mowing_time) +
    (push_mower_rate * (1 - x) * total_mowing_time) = total_lawn_area ∧
    x = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_jerry_mowing_fraction_l1138_113809


namespace NUMINAMATH_CALUDE_eraser_difference_l1138_113817

theorem eraser_difference (andrea_erasers : ℕ) (anya_multiplier : ℕ) : 
  andrea_erasers = 4 →
  anya_multiplier = 4 →
  anya_multiplier * andrea_erasers - andrea_erasers = 12 := by
  sorry

end NUMINAMATH_CALUDE_eraser_difference_l1138_113817


namespace NUMINAMATH_CALUDE_expression_simplification_l1138_113881

theorem expression_simplification (x : ℝ) : 
  (3*x^2 + 4*x - 5)*(x - 2) + (x - 2)*(2*x^2 - 3*x + 9) - (4*x - 7)*(x - 2)*(x - 3) = 
  x^3 + x^2 + 12*x - 36 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l1138_113881


namespace NUMINAMATH_CALUDE_count_valid_numbers_l1138_113890

/-- The set of digits that can be used to form the numbers -/
def digits : Finset Nat := {0, 1, 2, 3}

/-- A predicate that checks if a number is a four-digit even number -/
def is_valid_number (n : Nat) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ Even n

/-- A function that returns the digits of a number as a list -/
def number_to_digits (n : Nat) : List Nat :=
  sorry

/-- A predicate that checks if a number uses only the allowed digits without repetition -/
def uses_valid_digits (n : Nat) : Prop :=
  let d := number_to_digits n
  d.toFinset ⊆ digits ∧ d.length = 4 ∧ d.Nodup

/-- The set of all valid numbers according to the problem conditions -/
def valid_numbers : Finset Nat :=
  sorry

theorem count_valid_numbers : valid_numbers.card = 10 := by
  sorry

end NUMINAMATH_CALUDE_count_valid_numbers_l1138_113890


namespace NUMINAMATH_CALUDE_modulus_of_complex_l1138_113847

theorem modulus_of_complex (z : ℂ) : z = (1 + Complex.I) / Complex.I → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_l1138_113847


namespace NUMINAMATH_CALUDE_trapezoid_shorter_base_l1138_113800

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  longer_base : ℝ
  midpoint_line : ℝ
  shorter_base : ℝ

/-- The trapezoid satisfies the given conditions -/
def satisfies_conditions (t : Trapezoid) : Prop :=
  t.longer_base = 105 ∧ t.midpoint_line = 7

/-- The theorem to be proved -/
theorem trapezoid_shorter_base (t : Trapezoid) 
  (h : satisfies_conditions t) : t.shorter_base = 91 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_shorter_base_l1138_113800


namespace NUMINAMATH_CALUDE_hall_length_is_six_l1138_113846

/-- A rectangular hall with given properties --/
structure Hall where
  length : ℝ
  width : ℝ
  height : ℝ
  volume : ℝ
  area_floor_ceiling : ℝ
  area_walls : ℝ

/-- The theorem stating the conditions and the result to be proved --/
theorem hall_length_is_six (h : Hall) 
  (h_width : h.width = 6)
  (h_volume : h.volume = 108)
  (h_areas : h.area_floor_ceiling = h.area_walls)
  (h_floor_ceiling : h.area_floor_ceiling = 2 * h.length * h.width)
  (h_walls : h.area_walls = 2 * h.length * h.height + 2 * h.width * h.height)
  (h_volume_calc : h.volume = h.length * h.width * h.height) :
  h.length = 6 := by
  sorry

end NUMINAMATH_CALUDE_hall_length_is_six_l1138_113846


namespace NUMINAMATH_CALUDE_simplify_expression_l1138_113880

theorem simplify_expression (a b : ℝ) : 3*a + 2*b - 2*(a - b) = a + 4*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1138_113880


namespace NUMINAMATH_CALUDE_matches_needed_for_new_win_rate_l1138_113871

/-- Given a player who has won 19 out of 20 matches, prove that they need to win 5 more matches
    without any losses to achieve a 96% winning rate. -/
theorem matches_needed_for_new_win_rate
  (initial_matches : Nat)
  (initial_wins : Nat)
  (target_win_rate : Rat)
  (h1 : initial_matches = 20)
  (h2 : initial_wins = 19)
  (h3 : target_win_rate = 24/25) :
  ∃ (additional_wins : Nat),
    additional_wins = 5 ∧
    (initial_wins + additional_wins : Rat) / (initial_matches + additional_wins) = target_win_rate :=
by sorry

end NUMINAMATH_CALUDE_matches_needed_for_new_win_rate_l1138_113871


namespace NUMINAMATH_CALUDE_log_ratio_squared_equals_one_l1138_113836

theorem log_ratio_squared_equals_one (x y : ℝ) 
  (hx_pos : x > 0) (hy_pos : y > 0) 
  (hx_neq_one : x ≠ 1) (hy_neq_one : y ≠ 1)
  (h_log : Real.log x / Real.log 3 = Real.log 81 / Real.log y)
  (h_sum : x + y = 36) : 
  (Real.log (x / y) / Real.log 3)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_ratio_squared_equals_one_l1138_113836


namespace NUMINAMATH_CALUDE_triangle_integer_sides_altitudes_even_perimeter_l1138_113808

theorem triangle_integer_sides_altitudes_even_perimeter 
  (a b c : ℕ) 
  (ha hb hc : ℕ) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_altitudes : ha ≠ 0 ∧ hb ≠ 0 ∧ hc ≠ 0) :
  ∃ k : ℕ, a + b + c = 2 * k := by
  sorry

#check triangle_integer_sides_altitudes_even_perimeter

end NUMINAMATH_CALUDE_triangle_integer_sides_altitudes_even_perimeter_l1138_113808


namespace NUMINAMATH_CALUDE_derivative_at_one_l1138_113891

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- State the theorem
theorem derivative_at_one :
  deriv f 1 = 0 := by sorry

end NUMINAMATH_CALUDE_derivative_at_one_l1138_113891


namespace NUMINAMATH_CALUDE_parabola_tangent_theorem_l1138_113801

-- Define the parabola and points
def Parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2*p*y

structure Point where
  x : ℝ
  y : ℝ

-- Define the given conditions
def IsValidConfiguration (p : ℝ) (P A B C D Q : Point) : Prop :=
  p > 0 ∧
  Parabola p A.x A.y ∧
  Parabola p B.x B.y ∧
  C.y = 0 ∧
  D.y = 0 ∧
  Q.x = 0 ∧
  -- PA and PB are tangent to G at A and B (implied)
  -- P is outside the parabola (implied)
  -- C and D are on x-axis (y = 0)
  -- Q is on y-axis (x = 0)
  true -- Additional conditions could be added here if needed

-- Define what it means for PCQD to be a parallelogram
def IsParallelogram (P C Q D : Point) : Prop :=
  (P.x - C.x = Q.x - D.x) ∧ (P.y - C.y = Q.y - D.y)

-- Define the main theorem
theorem parabola_tangent_theorem (p : ℝ) (P A B C D Q : Point) :
  IsValidConfiguration p P A B C D Q →
  (IsParallelogram P C Q D ∧
   (IsParallelogram P C Q D ∧ (P.x - C.x)^2 + (P.y - C.y)^2 = (Q.x - D.x)^2 + (Q.y - D.y)^2 ↔ Q.y = p/2)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_tangent_theorem_l1138_113801


namespace NUMINAMATH_CALUDE_line_plane_relationship_l1138_113857

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines and between a line and a plane
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- State the theorem
theorem line_plane_relationship 
  (a b : Line) (α : Plane) 
  (h1 : parallel_lines a b) 
  (h2 : parallel_line_plane a α) : 
  parallel_line_plane b α ∨ line_in_plane b α :=
sorry

end NUMINAMATH_CALUDE_line_plane_relationship_l1138_113857


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1138_113843

theorem complex_fraction_simplification : 
  (((12^4 + 500) * (24^4 + 500) * (36^4 + 500) * (48^4 + 500) * (60^4 + 500)) / 
   ((6^4 + 500) * (18^4 + 500) * (30^4 + 500) * (42^4 + 500) * (54^4 + 500))) = -995 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1138_113843


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l1138_113892

/-- The total surface area of a right cylinder with height 10 cm and radius 3 cm is 78π cm². -/
theorem cylinder_surface_area : 
  let h : ℝ := 10  -- height in cm
  let r : ℝ := 3   -- radius in cm
  let lateral_area := 2 * Real.pi * r * h
  let base_area := Real.pi * r^2
  let total_area := lateral_area + 2 * base_area
  total_area = 78 * Real.pi := by sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l1138_113892


namespace NUMINAMATH_CALUDE_wire_cutting_l1138_113870

theorem wire_cutting (total_length : ℝ) (piece1 piece2 piece3 : ℝ) : 
  total_length = 95 →
  piece2 = 1.5 * piece1 →
  piece3 = 1.5 * piece2 →
  piece1 + piece2 + piece3 = total_length →
  piece3 = 45 := by
sorry

end NUMINAMATH_CALUDE_wire_cutting_l1138_113870


namespace NUMINAMATH_CALUDE_box_width_l1138_113812

/-- The width of a rectangular box given its filling rate, dimensions, and filling time. -/
theorem box_width
  (fill_rate : ℝ)  -- Filling rate in cubic feet per hour
  (length : ℝ)     -- Length of the box in feet
  (depth : ℝ)      -- Depth of the box in feet
  (fill_time : ℝ)  -- Time to fill the box in hours
  (h1 : fill_rate = 3)
  (h2 : length = 5)
  (h3 : depth = 3)
  (h4 : fill_time = 20) :
  ∃ (width : ℝ), width = 4 ∧ fill_rate * fill_time = length * width * depth :=
by
  sorry

end NUMINAMATH_CALUDE_box_width_l1138_113812


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1138_113844

theorem imaginary_part_of_complex_fraction : Complex.im ((3 * Complex.I + 4) / (1 + 2 * Complex.I)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1138_113844


namespace NUMINAMATH_CALUDE_curve_intersection_l1138_113830

theorem curve_intersection :
  ∃ (θ t : ℝ),
    0 ≤ θ ∧ θ ≤ π ∧
    Real.sqrt 5 * Real.cos θ = 5/6 ∧
    Real.sin θ = 2/3 ∧
    (5/4) * t = 5/6 ∧
    t = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_curve_intersection_l1138_113830


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l1138_113862

theorem quadratic_coefficient (b m : ℝ) : 
  b > 0 ∧ 
  (∀ x, x^2 + b*x + 72 = (x + m)^2 + 12) →
  b = 4 * Real.sqrt 15 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l1138_113862


namespace NUMINAMATH_CALUDE_power_of_one_seventh_l1138_113840

def is_greatest_power_of_2_factor (x : ℕ) : Prop :=
  2^x ∣ 180 ∧ ∀ k > x, ¬(2^k ∣ 180)

def is_greatest_power_of_3_factor (y : ℕ) : Prop :=
  3^y ∣ 180 ∧ ∀ k > y, ¬(3^k ∣ 180)

theorem power_of_one_seventh (x y : ℕ) 
  (h2 : is_greatest_power_of_2_factor x) 
  (h3 : is_greatest_power_of_3_factor y) : 
  (1/7 : ℚ)^(y - x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_one_seventh_l1138_113840
