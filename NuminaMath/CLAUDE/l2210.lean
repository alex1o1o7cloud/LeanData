import Mathlib

namespace NUMINAMATH_CALUDE_larger_number_in_ratio_l2210_221069

theorem larger_number_in_ratio (a b : ℚ) : 
  a / b = 8 / 3 → a + b = 143 → max a b = 104 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_in_ratio_l2210_221069


namespace NUMINAMATH_CALUDE_train_length_calculation_l2210_221001

/-- Prove that given a train traveling at a certain speed that crosses a bridge in a given time, 
    and the total length of the bridge and train is known, we can calculate the length of the train. -/
theorem train_length_calculation 
  (train_speed : ℝ) 
  (crossing_time : ℝ) 
  (total_length : ℝ) 
  (h1 : train_speed = 45) -- km/hr
  (h2 : crossing_time = 30 / 3600) -- 30 seconds converted to hours
  (h3 : total_length = 195) -- meters
  : ∃ (train_length : ℝ), train_length = 180 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l2210_221001


namespace NUMINAMATH_CALUDE_linear_function_m_value_l2210_221048

/-- Linear function passing through a point -/
def linear_function (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x - 4

/-- Theorem: For a linear function y = (m-1)x - 4 passing through (2, 4), m = 5 -/
theorem linear_function_m_value :
  ∃ (m : ℝ), linear_function m 2 = 4 ∧ m = 5 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_m_value_l2210_221048


namespace NUMINAMATH_CALUDE_exam_score_deviation_l2210_221038

/-- Given an exam with mean score 74 and standard deviation σ,
    prove that 58 is 2σ below the mean when 98 is 3σ above the mean. -/
theorem exam_score_deviation (σ : ℝ) : 
  (74 + 3 * σ = 98) → (74 - 2 * σ = 58) := by
  sorry

end NUMINAMATH_CALUDE_exam_score_deviation_l2210_221038


namespace NUMINAMATH_CALUDE_mary_seth_age_difference_l2210_221076

/-- Represents the age difference between Mary and Seth -/
def age_difference : ℝ → ℝ → ℝ := λ m s => m - s

/-- Mary's age after one year will be three times Seth's age after one year -/
def future_age_relation (m : ℝ) (s : ℝ) : Prop := m + 1 = 3 * (s + 1)

theorem mary_seth_age_difference :
  ∀ (m s : ℝ),
  m > s →
  future_age_relation m s →
  m + s = 3.5 →
  age_difference m s = 2.75 := by
sorry

end NUMINAMATH_CALUDE_mary_seth_age_difference_l2210_221076


namespace NUMINAMATH_CALUDE_expand_product_l2210_221005

theorem expand_product (x : ℝ) : (x + 6) * (x + 8) * (x - 3) = x^3 + 11*x^2 + 6*x - 144 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2210_221005


namespace NUMINAMATH_CALUDE_coefficient_x4_is_1120_l2210_221044

open BigOperators

/-- The coefficient of x^4 in the expansion of (x^2 + 2/x)^8 -/
def coefficient_x4 : ℕ :=
  (Nat.choose 8 4) * 2^4

/-- Theorem stating that the coefficient of x^4 in (x^2 + 2/x)^8 is 1120 -/
theorem coefficient_x4_is_1120 : coefficient_x4 = 1120 := by
  sorry

#eval coefficient_x4  -- This will evaluate the expression and show the result

end NUMINAMATH_CALUDE_coefficient_x4_is_1120_l2210_221044


namespace NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l2210_221061

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l2210_221061


namespace NUMINAMATH_CALUDE_triangle_properties_l2210_221012

/-- Given a triangle ABC with vertices A(0,1), B(0,-1), and C(-2,1) -/
def triangle_ABC : Set (ℝ × ℝ) := {(0, 1), (0, -1), (-2, 1)}

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The equation of a circle in the form x^2 + y^2 + Dx + Ey + F = 0 -/
structure CircleEquation where
  D : ℝ
  E : ℝ
  F : ℝ

/-- Theorem stating the equations of altitude, midline, and circumcircle -/
theorem triangle_properties (ABC : Set (ℝ × ℝ)) 
  (h : ABC = triangle_ABC) : 
  ∃ (altitude_eq : LineEquation) 
    (midline_eq : LineEquation) 
    (circumcircle_eq : CircleEquation),
  altitude_eq = ⟨1, -1, 1⟩ ∧ 
  midline_eq = ⟨1, 0, 1⟩ ∧
  circumcircle_eq = ⟨2, 0, -1⟩ := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2210_221012


namespace NUMINAMATH_CALUDE_improve_shooting_average_l2210_221040

/-- Represents a basketball player's shooting statistics -/
structure ShootingStats :=
  (initial_shots : ℕ)
  (initial_made : ℕ)
  (additional_shots : ℕ)
  (additional_made : ℕ)

/-- Calculates the shooting average as a rational number -/
def shooting_average (stats : ShootingStats) : ℚ :=
  (stats.initial_made + stats.additional_made : ℚ) / (stats.initial_shots + stats.additional_shots)

theorem improve_shooting_average 
  (stats : ShootingStats) 
  (h1 : stats.initial_shots = 40)
  (h2 : stats.initial_made = 18)
  (h3 : stats.additional_shots = 15)
  (h4 : shooting_average {initial_shots := stats.initial_shots, 
                          initial_made := stats.initial_made, 
                          additional_shots := 0, 
                          additional_made := 0} = 45/100)
  : shooting_average {initial_shots := stats.initial_shots,
                      initial_made := stats.initial_made,
                      additional_shots := stats.additional_shots,
                      additional_made := 12} = 55/100 := by
  sorry

end NUMINAMATH_CALUDE_improve_shooting_average_l2210_221040


namespace NUMINAMATH_CALUDE_prob_sum_24_four_dice_l2210_221018

/-- The number of sides on a standard die -/
def standard_die_sides : ℕ := 6

/-- The target sum we're aiming for -/
def target_sum : ℕ := 24

/-- The number of dice rolled -/
def num_dice : ℕ := 4

/-- Probability of rolling a specific number on a fair, standard six-sided die -/
def single_die_prob : ℚ := 1 / standard_die_sides

/-- Theorem: The probability of rolling a sum of 24 with four fair, standard six-sided dice is 1/1296 -/
theorem prob_sum_24_four_dice : 
  (single_die_prob ^ num_dice : ℚ) = 1 / 1296 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_24_four_dice_l2210_221018


namespace NUMINAMATH_CALUDE_kate_keyboard_cost_l2210_221051

/-- The amount Kate spent on the keyboard -/
def keyboard_cost (march_savings april_savings may_savings mouse_cost remaining : ℕ) : ℕ :=
  (march_savings + april_savings + may_savings) - (mouse_cost + remaining)

theorem kate_keyboard_cost :
  keyboard_cost 27 13 28 5 14 = 49 := by
  sorry

end NUMINAMATH_CALUDE_kate_keyboard_cost_l2210_221051


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2210_221056

def f (x : ℝ) : ℝ := 2 * (x + 1) * (x - 3)

theorem quadratic_function_properties :
  (∀ x, f x = 2 * (x + 1) * (x - 3)) ∧
  f (-1) = 0 ∧ f 3 = 0 ∧ f 1 = -8 ∧
  (∀ x ∈ Set.Icc 0 3, f x ≥ -8 ∧ f x ≤ 0) ∧
  (∀ x, f x ≥ 0 ↔ x ≤ -1 ∨ x ≥ 3) :=
by sorry

#check quadratic_function_properties

end NUMINAMATH_CALUDE_quadratic_function_properties_l2210_221056


namespace NUMINAMATH_CALUDE_fahrenheit_celsius_conversion_l2210_221097

theorem fahrenheit_celsius_conversion (C F : ℚ) : 
  C = (4/7) * (F - 32) → C = 35 → F = 93.25 := by
  sorry

end NUMINAMATH_CALUDE_fahrenheit_celsius_conversion_l2210_221097


namespace NUMINAMATH_CALUDE_valid_numbers_characterization_l2210_221081

def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧  -- two-digit number
  n % 2 = 0 ∧  -- even
  (n / 10 + n % 10) > 6 ∧  -- sum of digits greater than 6
  (n / 10) ≥ (n % 10 + 4)  -- tens digit at least 4 greater than units digit

theorem valid_numbers_characterization :
  {n : ℕ | is_valid_number n} = {70, 80, 90, 62, 72, 82, 92, 84, 94} :=
by sorry

end NUMINAMATH_CALUDE_valid_numbers_characterization_l2210_221081


namespace NUMINAMATH_CALUDE_restaurant_cooks_count_l2210_221007

/-- Proves that the number of cooks is 9 given the initial and final ratios of cooks to waiters -/
theorem restaurant_cooks_count (cooks waiters : ℕ) : 
  (cooks : ℚ) / waiters = 3 / 10 →
  cooks / (waiters + 12) = 3 / 14 →
  cooks = 9 := by
sorry

end NUMINAMATH_CALUDE_restaurant_cooks_count_l2210_221007


namespace NUMINAMATH_CALUDE_sphere_remaining_volume_l2210_221052

/-- The remaining volume of a sphere after drilling a cylindrical hole -/
theorem sphere_remaining_volume (R : ℝ) (h : R > 3) : 
  (4 / 3 * π * R^3) - (6 * π * (R^2 - 9)) - (2 * π * 3^2 * (R - 3 / 3)) = 36 * π :=
sorry

end NUMINAMATH_CALUDE_sphere_remaining_volume_l2210_221052


namespace NUMINAMATH_CALUDE_conditional_without_else_l2210_221003

-- Define the structure of conditional statements
inductive ConditionalStatement
  | ifThenElse (condition : Prop) (thenStmt : Prop) (elseStmt : Prop)
  | ifThen (condition : Prop) (thenStmt : Prop)

-- Define a property that checks if a conditional statement has an ELSE part
def hasElsePart : ConditionalStatement → Prop
  | ConditionalStatement.ifThenElse _ _ _ => true
  | ConditionalStatement.ifThen _ _ => false

-- Theorem stating that there exists a conditional statement without an ELSE part
theorem conditional_without_else : ∃ (stmt : ConditionalStatement), ¬(hasElsePart stmt) := by
  sorry


end NUMINAMATH_CALUDE_conditional_without_else_l2210_221003


namespace NUMINAMATH_CALUDE_jim_unknown_row_trees_l2210_221034

/-- Represents the production of a lemon grove over 5 years -/
structure LemonGrove where
  normal_production : ℕ  -- lemons per year for a normal tree
  increase_percent : ℕ   -- percentage increase for Jim's trees
  known_row : ℕ          -- number of trees in the known row
  total_production : ℕ   -- total lemons produced in 5 years

/-- Calculates the number of trees in the unknown row of Jim's lemon grove -/
def unknown_row_trees (grove : LemonGrove) : ℕ :=
  let jim_tree_production := grove.normal_production * (100 + grove.increase_percent) / 100
  let total_trees := grove.total_production / (jim_tree_production * 5)
  total_trees - grove.known_row

/-- Theorem stating the number of trees in the unknown row of Jim's lemon grove -/
theorem jim_unknown_row_trees :
  let grove := LemonGrove.mk 60 50 30 675000
  unknown_row_trees grove = 1470 := by
  sorry

end NUMINAMATH_CALUDE_jim_unknown_row_trees_l2210_221034


namespace NUMINAMATH_CALUDE_selling_price_with_loss_l2210_221070

def cost_price : ℝ := 1800
def loss_percentage : ℝ := 10

theorem selling_price_with_loss (cp : ℝ) (lp : ℝ) : 
  cp * (1 - lp / 100) = 1620 :=
by sorry

end NUMINAMATH_CALUDE_selling_price_with_loss_l2210_221070


namespace NUMINAMATH_CALUDE_parabola_translation_l2210_221053

-- Define the original parabola
def original_parabola (x y : ℝ) : Prop := y = x^2 + 3

-- Define the translated parabola
def translated_parabola (x y : ℝ) : Prop := y = (x + 1)^2 + 3

-- Theorem statement
theorem parabola_translation :
  ∀ x y : ℝ, original_parabola (x + 1) y ↔ translated_parabola x y :=
by sorry

end NUMINAMATH_CALUDE_parabola_translation_l2210_221053


namespace NUMINAMATH_CALUDE_smallest_d_for_3150_square_l2210_221042

/-- The smallest positive integer d such that 3150 * d is a perfect square is 14 -/
theorem smallest_d_for_3150_square : ∃ (n : ℕ), 
  (3150 * 14 = n ^ 2) ∧ 
  (∀ (d : ℕ), d > 0 ∧ d < 14 → ¬∃ (m : ℕ), 3150 * d = m ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_smallest_d_for_3150_square_l2210_221042


namespace NUMINAMATH_CALUDE_lcm_36_84_l2210_221055

theorem lcm_36_84 : Nat.lcm 36 84 = 252 := by
  sorry

end NUMINAMATH_CALUDE_lcm_36_84_l2210_221055


namespace NUMINAMATH_CALUDE_sum_in_special_base_l2210_221017

theorem sum_in_special_base (b : ℕ) (h : b > 1) :
  (b + 3) * (b + 4) * (b + 5) = 2 * b^3 + 3 * b^2 + 2 * b + 5 →
  (b + 3) + (b + 4) + (b + 5) = 4 * b + 2 :=
by sorry

end NUMINAMATH_CALUDE_sum_in_special_base_l2210_221017


namespace NUMINAMATH_CALUDE_product_abc_value_l2210_221058

theorem product_abc_value
  (h1 : b * c * d = 65)
  (h2 : c * d * e = 750)
  (h3 : d * e * f = 250)
  (h4 : (a * f) / (c * d) = 0.6666666666666666)
  : a * b * c = 130 := by
  sorry

end NUMINAMATH_CALUDE_product_abc_value_l2210_221058


namespace NUMINAMATH_CALUDE_average_of_7_12_and_M_l2210_221093

theorem average_of_7_12_and_M :
  ∃ M : ℝ, 10 < M ∧ M < 20 ∧ ((7 + 12 + M) / 3 = 11 ∨ (7 + 12 + M) / 3 = 13) := by
  sorry

end NUMINAMATH_CALUDE_average_of_7_12_and_M_l2210_221093


namespace NUMINAMATH_CALUDE_days_to_build_floor_l2210_221074

-- Define the daily pay rate for a builder
def builder_daily_pay : ℕ := 100

-- Define the total cost for the project
def total_project_cost : ℕ := 270000

-- Define the number of builders for the project
def project_builders : ℕ := 6

-- Define the number of houses in the project
def project_houses : ℕ := 5

-- Define the number of floors per house in the project
def floors_per_house : ℕ := 6

-- Theorem to prove
theorem days_to_build_floor (builders : ℕ) (days : ℕ) : 
  builders = 3 → days = 30 → 
  (builders * builder_daily_pay * days = 
   total_project_cost * builders / project_builders / 
   (project_houses * floors_per_house)) := by sorry

end NUMINAMATH_CALUDE_days_to_build_floor_l2210_221074


namespace NUMINAMATH_CALUDE_vacation_tents_l2210_221092

/-- Calculates the number of tents needed given the total number of people,
    the number of people the house can accommodate, and the capacity of each tent. -/
def tents_needed (total_people : ℕ) (house_capacity : ℕ) (tent_capacity : ℕ) : ℕ :=
  ((total_people - house_capacity) + tent_capacity - 1) / tent_capacity

theorem vacation_tents :
  tents_needed 14 4 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_vacation_tents_l2210_221092


namespace NUMINAMATH_CALUDE_arrangement_count_is_correct_l2210_221008

/-- The number of ways to arrange 5 students in a row with specific constraints -/
def arrangement_count : ℕ := 36

/-- A function that calculates the number of valid arrangements -/
def calculate_arrangements : ℕ :=
  let total_students : ℕ := 5
  let ab_pair_arrangements : ℕ := 3 * 2  -- 3! for AB pair and 2 others, 2! for AB swap
  let c_placement_options : ℕ := 3       -- C always has 3 valid positions
  ab_pair_arrangements * c_placement_options

/-- Theorem stating that the number of valid arrangements is 36 -/
theorem arrangement_count_is_correct : arrangement_count = calculate_arrangements := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_is_correct_l2210_221008


namespace NUMINAMATH_CALUDE_photographs_eighteen_hours_ago_l2210_221013

theorem photographs_eighteen_hours_ago (photos_18h_ago : ℕ) : 
  (photos_18h_ago : ℚ) + 0.8 * (photos_18h_ago : ℚ) = 180 →
  photos_18h_ago = 100 := by
sorry

end NUMINAMATH_CALUDE_photographs_eighteen_hours_ago_l2210_221013


namespace NUMINAMATH_CALUDE_chord_length_l2210_221060

/-- The length of the chord formed by the intersection of a line and a circle -/
theorem chord_length (x y : ℝ) : 
  (x - y + 2 = 0) →  -- Line equation
  ((x - 1)^2 + (y - 2)^2 = 4) →  -- Circle equation
  ∃ A B : ℝ × ℝ,  -- Points of intersection
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 14  -- Length of AB squared
  := by sorry

end NUMINAMATH_CALUDE_chord_length_l2210_221060


namespace NUMINAMATH_CALUDE_log_sum_condition_l2210_221019

theorem log_sum_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ a b, a > 1 ∧ b > 1 → Real.log a + Real.log b > 0) ∧
  (∃ a b, Real.log a + Real.log b > 0 ∧ ¬(a > 1 ∧ b > 1)) := by
  sorry

end NUMINAMATH_CALUDE_log_sum_condition_l2210_221019


namespace NUMINAMATH_CALUDE_improved_running_distance_l2210_221030

/-- Proves that a runner who can cover 40 yards in 5 seconds and improves their speed by 40% will cover 112 yards in 10 seconds -/
theorem improved_running_distance 
  (initial_distance : ℝ) 
  (initial_time : ℝ) 
  (improvement_percentage : ℝ) 
  (new_time : ℝ) :
  initial_distance = 40 ∧ 
  initial_time = 5 ∧ 
  improvement_percentage = 40 ∧ 
  new_time = 10 → 
  (initial_distance * (1 + improvement_percentage / 100) * (new_time / initial_time)) = 112 :=
by sorry

end NUMINAMATH_CALUDE_improved_running_distance_l2210_221030


namespace NUMINAMATH_CALUDE_books_sold_in_garage_sale_l2210_221065

theorem books_sold_in_garage_sale 
  (initial_books : ℕ) 
  (books_given_to_friend : ℕ) 
  (remaining_books : ℕ) 
  (h1 : initial_books = 108) 
  (h2 : books_given_to_friend = 35) 
  (h3 : remaining_books = 62) :
  initial_books - books_given_to_friend - remaining_books = 11 := by
sorry

end NUMINAMATH_CALUDE_books_sold_in_garage_sale_l2210_221065


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l2210_221084

-- Define the square
def Square (perimeter : ℝ) : Type :=
  { side : ℝ // perimeter = 4 * side }

-- Theorem statement
theorem square_area_from_perimeter (perimeter : ℝ) (h : perimeter = 80) :
  ∃ (s : Square perimeter), (s.val)^2 = 400 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l2210_221084


namespace NUMINAMATH_CALUDE_mean_exercise_days_jenkins_class_l2210_221025

/-- Represents the exercise data for a group of students -/
structure ExerciseData where
  students : List (Nat × Float)

/-- Calculates the mean number of days exercised -/
def calculateMean (data : ExerciseData) : Float :=
  let totalDays := data.students.foldl (fun acc (n, d) => acc + n.toFloat * d) 0
  let totalStudents := data.students.foldl (fun acc (n, _) => acc + n) 0
  totalDays / totalStudents.toFloat

/-- Rounds a float to the nearest hundredth -/
def roundToHundredth (x : Float) : Float :=
  (x * 100).round / 100

theorem mean_exercise_days_jenkins_class :
  let jenkinsData : ExerciseData := {
    students := [
      (2, 0.5),
      (4, 1),
      (5, 3),
      (3, 4),
      (7, 6),
      (2, 7)
    ]
  }
  roundToHundredth (calculateMean jenkinsData) = 3.83 := by
  sorry

end NUMINAMATH_CALUDE_mean_exercise_days_jenkins_class_l2210_221025


namespace NUMINAMATH_CALUDE_max_value_expression_l2210_221099

theorem max_value_expression (A M C : ℕ) (h : A + M + C = 15) :
  (∀ a m c : ℕ, a + m + c = 15 → 2*a*m*c + a*m + m*c + c*a ≤ 2*A*M*C + A*M + M*C + C*A) →
  2*A*M*C + A*M + M*C + C*A = 325 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l2210_221099


namespace NUMINAMATH_CALUDE_x_minus_q_equals_three_l2210_221046

theorem x_minus_q_equals_three (x q : ℝ) (h1 : |x - 3| = q) (h2 : x > 3) :
  x - q = 3 := by sorry

end NUMINAMATH_CALUDE_x_minus_q_equals_three_l2210_221046


namespace NUMINAMATH_CALUDE_psychology_lecture_first_probability_l2210_221006

-- Define the type for lectures
inductive Lecture
| Morality
| Psychology
| Safety

-- Define a function to calculate the number of permutations
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Define the theorem
theorem psychology_lecture_first_probability :
  let total_arrangements := factorial 3
  let favorable_arrangements := factorial 2
  (favorable_arrangements : ℚ) / total_arrangements = 1 / 3 := by
sorry


end NUMINAMATH_CALUDE_psychology_lecture_first_probability_l2210_221006


namespace NUMINAMATH_CALUDE_distance_when_in_step_l2210_221033

/-- The stride length of Jack in centimeters. -/
def jackStride : ℕ := 64

/-- The stride length of Jill in centimeters. -/
def jillStride : ℕ := 56

/-- The theorem states that the distance walked when Jack and Jill are next in step
    is equal to the least common multiple of their stride lengths. -/
theorem distance_when_in_step :
  Nat.lcm jackStride jillStride = 448 := by sorry

end NUMINAMATH_CALUDE_distance_when_in_step_l2210_221033


namespace NUMINAMATH_CALUDE_oplus_inequality_solutions_l2210_221078

def oplus (a b : ℤ) : ℤ := 1 - a * b

theorem oplus_inequality_solutions :
  (∃! (n : ℕ), ∀ (x : ℕ), oplus x 2 ≥ -3 ↔ x ≤ n) ∧
  (∃ (s : Finset ℕ), s.card = 3 ∧ ∀ (x : ℕ), x ∈ s ↔ oplus x 2 ≥ -3) :=
by sorry

end NUMINAMATH_CALUDE_oplus_inequality_solutions_l2210_221078


namespace NUMINAMATH_CALUDE_equilateral_triangle_sum_product_l2210_221029

def is_equilateral_triangle (x y z : ℂ) : Prop :=
  Complex.abs (y - x) = Complex.abs (z - y) ∧ 
  Complex.abs (z - y) = Complex.abs (x - z) ∧
  Complex.abs (x - z) = Complex.abs (y - x)

theorem equilateral_triangle_sum_product (x y z : ℂ) :
  is_equilateral_triangle x y z →
  Complex.abs (y - x) = 24 →
  Complex.abs (x + y + z) = 72 →
  Complex.abs (x * y + x * z + y * z) = 1728 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_sum_product_l2210_221029


namespace NUMINAMATH_CALUDE_garden_area_theorem_l2210_221082

/-- Represents a rectangular garden with given properties -/
structure RectangularGarden where
  length : ℝ
  width : ℝ
  perimeter_walk_count : ℕ
  length_walk_count : ℕ
  total_distance : ℝ

/-- The theorem stating the area of the garden given the conditions -/
theorem garden_area_theorem (g : RectangularGarden) 
  (h1 : g.perimeter_walk_count = 20)
  (h2 : g.length_walk_count = 50)
  (h3 : g.total_distance = 1500)
  (h4 : 2 * (g.length + g.width) = g.total_distance / g.perimeter_walk_count)
  (h5 : g.length = g.total_distance / g.length_walk_count) :
  g.length * g.width = 225 := by
  sorry

#check garden_area_theorem

end NUMINAMATH_CALUDE_garden_area_theorem_l2210_221082


namespace NUMINAMATH_CALUDE_break_even_circus_production_l2210_221026

/-- Calculates the number of sold-out performances needed to break even for a circus production -/
def break_even_performances (overhead : ℕ) (production_cost : ℕ) (revenue : ℕ) : ℕ :=
  let total_cost (x : ℕ) := overhead + production_cost * x
  let total_revenue (x : ℕ) := revenue * x
  (overhead / (revenue - production_cost) : ℕ)

/-- Proves that 9 sold-out performances are needed to break even given the specific costs and revenue -/
theorem break_even_circus_production :
  break_even_performances 81000 7000 16000 = 9 := by
  sorry

end NUMINAMATH_CALUDE_break_even_circus_production_l2210_221026


namespace NUMINAMATH_CALUDE_distinct_plants_count_l2210_221016

/-- Represents a flower bed -/
structure FlowerBed where
  plants : Finset ℕ

/-- The total number of distinct plants in three intersecting flower beds -/
def total_distinct_plants (X Y Z : FlowerBed) : ℕ :=
  (X.plants ∪ Y.plants ∪ Z.plants).card

/-- The theorem stating the total number of distinct plants in the given scenario -/
theorem distinct_plants_count (X Y Z : FlowerBed)
  (hX : X.plants.card = 600)
  (hY : Y.plants.card = 500)
  (hZ : Z.plants.card = 400)
  (hXY : (X.plants ∩ Y.plants).card = 100)
  (hYZ : (Y.plants ∩ Z.plants).card = 80)
  (hXZ : (X.plants ∩ Z.plants).card = 120)
  (hXYZ : (X.plants ∩ Y.plants ∩ Z.plants).card = 30) :
  total_distinct_plants X Y Z = 1230 := by
  sorry


end NUMINAMATH_CALUDE_distinct_plants_count_l2210_221016


namespace NUMINAMATH_CALUDE_cubic_inequality_l2210_221037

theorem cubic_inequality (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) :
  2 * a^3 - b^3 ≥ 2 * a * b^2 - a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l2210_221037


namespace NUMINAMATH_CALUDE_abs_ab_minus_cd_le_quarter_l2210_221057

theorem abs_ab_minus_cd_le_quarter 
  (a b c d : ℝ) 
  (non_neg_a : 0 ≤ a) (non_neg_b : 0 ≤ b) (non_neg_c : 0 ≤ c) (non_neg_d : 0 ≤ d)
  (sum_eq_one : a + b + c + d = 1) : 
  |a * b - c * d| ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_abs_ab_minus_cd_le_quarter_l2210_221057


namespace NUMINAMATH_CALUDE_pythagorean_triple_7_24_25_l2210_221032

theorem pythagorean_triple_7_24_25 : 
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a = 7 ∧ b = 24 ∧ c = 25 ∧ a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_7_24_25_l2210_221032


namespace NUMINAMATH_CALUDE_percentage_of_number_l2210_221096

theorem percentage_of_number (percentage : ℝ) (number : ℝ) (result : ℝ) :
  percentage = 110 ∧ number = 500 ∧ result = 550 →
  (percentage / 100) * number = result := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_number_l2210_221096


namespace NUMINAMATH_CALUDE_expected_heads_equals_55_l2210_221000

/-- The number of coins -/
def num_coins : ℕ := 80

/-- The probability of a coin landing heads on a single flip -/
def p_heads : ℚ := 1/2

/-- The probability of a coin being eligible for a second flip -/
def p_second_flip : ℚ := 1/2

/-- The probability of a coin being eligible for a third flip -/
def p_third_flip : ℚ := 1/2

/-- The expected number of heads after all flips -/
def expected_heads : ℚ := num_coins * (p_heads + p_heads * (1 - p_heads) * p_second_flip + p_heads * (1 - p_heads) * p_second_flip * (1 - p_heads) * p_third_flip)

theorem expected_heads_equals_55 : expected_heads = 55 := by
  sorry

end NUMINAMATH_CALUDE_expected_heads_equals_55_l2210_221000


namespace NUMINAMATH_CALUDE_smallest_binary_multiple_of_60_l2210_221041

def is_binary (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 0 ∨ d = 1

theorem smallest_binary_multiple_of_60 :
  ∃ (X : ℕ), X > 0 ∧ is_binary (60 * X) ∧
  (∀ (Y : ℕ), Y > 0 → is_binary (60 * Y) → X ≤ Y) ∧
  X = 185 := by
sorry

end NUMINAMATH_CALUDE_smallest_binary_multiple_of_60_l2210_221041


namespace NUMINAMATH_CALUDE_square_diff_cube_seven_six_l2210_221079

theorem square_diff_cube_seven_six : (7^2 - 6^2)^3 = 2197 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_cube_seven_six_l2210_221079


namespace NUMINAMATH_CALUDE_part_one_part_two_l2210_221054

/-- The function f(x) defined in the problem -/
def f (a c x : ℝ) : ℝ := -3 * x^2 + a * (6 - a) * x + c

/-- Part 1 of the problem -/
theorem part_one (a : ℝ) :
  f a 19 1 > 0 ↔ -2 < a ∧ a < 8 := by sorry

/-- Part 2 of the problem -/
theorem part_two (a c : ℝ) :
  (∀ x : ℝ, f a c x > 0 ↔ -1 < x ∧ x < 3) →
  ((a = 3 + Real.sqrt 3 ∨ a = 3 - Real.sqrt 3) ∧ c = 9) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2210_221054


namespace NUMINAMATH_CALUDE_equation_solution_l2210_221086

theorem equation_solution (x : ℂ) : 
  (x^2 + 4*x + 8) / (x - 3) = 2 ↔ x = -1 + (7*Real.sqrt 2/2)*I ∨ x = -1 - (7*Real.sqrt 2/2)*I :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l2210_221086


namespace NUMINAMATH_CALUDE_area_triangle_OAB_l2210_221021

/-- Given a polar coordinate system with pole O, point A(1, π/6), and point B(2, π/2),
    the area of triangle OAB is √3/2. -/
theorem area_triangle_OAB :
  let r₁ : ℝ := 1
  let θ₁ : ℝ := π / 6
  let r₂ : ℝ := 2
  let θ₂ : ℝ := π / 2
  let area := (1 / 2) * r₁ * r₂ * Real.sin (θ₂ - θ₁)
  area = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_area_triangle_OAB_l2210_221021


namespace NUMINAMATH_CALUDE_wax_remaining_after_detailing_l2210_221045

-- Define the initial amounts of wax
def waxA_initial : ℕ := 10
def waxB_initial : ℕ := 15

-- Define the amounts required for each vehicle
def waxA_car : ℕ := 4
def waxA_suv : ℕ := 6
def waxB_car : ℕ := 3
def waxB_suv : ℕ := 5

-- Define the amounts spilled
def waxA_spilled : ℕ := 3
def waxB_spilled : ℕ := 4

-- Theorem to prove
theorem wax_remaining_after_detailing :
  (waxA_initial - waxA_spilled - waxA_car) + (waxB_initial - waxB_spilled - waxB_suv) = 9 := by
  sorry

end NUMINAMATH_CALUDE_wax_remaining_after_detailing_l2210_221045


namespace NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l2210_221072

theorem cubic_roots_sum_cubes (a b c : ℝ) : 
  (3 * a^3 + 5 * a^2 - 150 * a + 7 = 0) →
  (3 * b^3 + 5 * b^2 - 150 * b + 7 = 0) →
  (3 * c^3 + 5 * c^2 - 150 * c + 7 = 0) →
  (a + b + 2)^3 + (b + c + 2)^3 + (c + a + 2)^3 = 303 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l2210_221072


namespace NUMINAMATH_CALUDE_right_triangle_integer_area_l2210_221091

theorem right_triangle_integer_area (a b : ℕ) :
  (∃ A : ℕ, A = a * b / 2) ↔ (Even a ∨ Even b) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_integer_area_l2210_221091


namespace NUMINAMATH_CALUDE_larry_gave_brother_l2210_221090

def larry_problem (initial_amount lunch_expense final_amount : ℕ) : Prop :=
  initial_amount - lunch_expense - final_amount = 2

theorem larry_gave_brother : 
  larry_problem 22 5 15 := by sorry

end NUMINAMATH_CALUDE_larry_gave_brother_l2210_221090


namespace NUMINAMATH_CALUDE_trailer_cost_is_120000_l2210_221059

/-- Represents the cost of a house in dollars -/
def house_cost : ℕ := 480000

/-- Represents the loan period in months -/
def loan_period : ℕ := 240

/-- Represents the additional monthly payment for the house compared to the trailer in dollars -/
def additional_house_payment : ℕ := 1500

/-- Calculates the cost of the trailer given the house cost, loan period, and additional house payment -/
def trailer_cost (h : ℕ) (l : ℕ) (a : ℕ) : ℕ := 
  h - l * a

/-- Theorem stating that the cost of the trailer is $120,000 -/
theorem trailer_cost_is_120000 : 
  trailer_cost house_cost loan_period additional_house_payment = 120000 := by
  sorry

end NUMINAMATH_CALUDE_trailer_cost_is_120000_l2210_221059


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2210_221094

/-- The repeating decimal 4.252525... -/
def repeating_decimal : ℚ := 4 + 25 / 99

/-- The fraction 421/99 -/
def target_fraction : ℚ := 421 / 99

/-- Theorem stating that the repeating decimal 4.252525... is equal to 421/99 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = target_fraction := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2210_221094


namespace NUMINAMATH_CALUDE_first_tap_fill_time_l2210_221087

/-- Represents the time (in hours) it takes for the first tap to fill the cistern -/
def T : ℝ := 3

/-- Represents the time (in hours) it takes for the second tap to empty the cistern -/
def empty_time : ℝ := 8

/-- Represents the time (in hours) it takes to fill the cistern when both taps are open -/
def both_open_time : ℝ := 4.8

/-- Proves that T is the correct time for the first tap to fill the cistern -/
theorem first_tap_fill_time :
  (1 / T - 1 / empty_time = 1 / both_open_time) ∧ T > 0 := by
  sorry

end NUMINAMATH_CALUDE_first_tap_fill_time_l2210_221087


namespace NUMINAMATH_CALUDE_subcubes_two_plus_painted_faces_count_l2210_221047

/-- Represents a cube with side length n --/
structure Cube (n : ℕ) where
  side_length : ℕ
  painted_faces : ℕ
  h_side : side_length = n
  h_painted : painted_faces = 6

/-- Represents a subcube of a larger cube --/
structure Subcube (n : ℕ) where
  painted_faces : ℕ
  h_painted : painted_faces ≤ 3

/-- The number of subcubes with at least two painted faces in a painted cube --/
def subcubes_with_two_plus_painted_faces (c : Cube 4) : ℕ := sorry

/-- Theorem stating that the number of 1x1x1 subcubes with at least two painted faces
    in a 4x4x4 fully painted cube is 32 --/
theorem subcubes_two_plus_painted_faces_count (c : Cube 4) :
  subcubes_with_two_plus_painted_faces c = 32 := by sorry

end NUMINAMATH_CALUDE_subcubes_two_plus_painted_faces_count_l2210_221047


namespace NUMINAMATH_CALUDE_absolute_value_non_negative_l2210_221071

theorem absolute_value_non_negative (a : ℝ) : ¬(|a| < 0) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_non_negative_l2210_221071


namespace NUMINAMATH_CALUDE_sum_factorials_6_mod_20_l2210_221024

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def sum_factorials (n : ℕ) : ℕ :=
  match n with
  | 0 => factorial 0
  | n + 1 => factorial (n + 1) + sum_factorials n

theorem sum_factorials_6_mod_20 :
  sum_factorials 6 % 20 = 13 := by sorry

end NUMINAMATH_CALUDE_sum_factorials_6_mod_20_l2210_221024


namespace NUMINAMATH_CALUDE_intersection_and_complement_when_m_eq_2_existence_of_m_for_subset_l2210_221043

-- Define sets A and B
def A : Set ℝ := {x | (x + 3) / (x - 1) ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - m*x - 2*m^2 ≤ 0}

-- Theorem for part (1)
theorem intersection_and_complement_when_m_eq_2 :
  (A ∩ B 2 = {x | -2 ≤ x ∧ x < 1}) ∧
  (Set.univ \ B 2 = {x | x < -2 ∨ x > 4}) := by sorry

-- Theorem for part (2)
theorem existence_of_m_for_subset :
  (∃ m : ℝ, m ≥ 3 ∧ A ⊆ B m) ∧
  (∀ m : ℝ, m < 3 → ¬(A ⊆ B m)) := by sorry

end NUMINAMATH_CALUDE_intersection_and_complement_when_m_eq_2_existence_of_m_for_subset_l2210_221043


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2210_221062

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, x^2 - a*x - b < 0 ↔ 2 < x ∧ x < 3) → 
  a + b = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2210_221062


namespace NUMINAMATH_CALUDE_not_all_squares_congruent_l2210_221010

-- Define a square
structure Square where
  side_length : ℝ
  angle_measure : ℝ
  is_rectangle : Prop
  similar_to_all_squares : Prop

-- Define properties of squares
axiom square_angles (s : Square) : s.angle_measure = 90

axiom square_sides_equal (s : Square) : s.side_length > 0

axiom square_is_rectangle (s : Square) : s.is_rectangle = true

axiom squares_similar (s1 s2 : Square) : s1.similar_to_all_squares ∧ s2.similar_to_all_squares

-- Theorem to prove
theorem not_all_squares_congruent : ¬∀ (s1 s2 : Square), s1 = s2 := by
  sorry


end NUMINAMATH_CALUDE_not_all_squares_congruent_l2210_221010


namespace NUMINAMATH_CALUDE_coexistent_pair_properties_l2210_221083

/-- Definition of coexistent rational number pairs -/
def is_coexistent_pair (a b : ℚ) : Prop := a - b = a * b + 1

theorem coexistent_pair_properties :
  /- (1) -/
  is_coexistent_pair 3 (1/2) ∧
  /- (2) -/
  (∀ m n : ℚ, is_coexistent_pair m n → is_coexistent_pair (-n) (-m)) ∧
  /- (3) -/
  is_coexistent_pair 4 (3/5) ∧
  /- (4) -/
  (∀ a : ℚ, is_coexistent_pair a 3 → a = -2) :=
by sorry

end NUMINAMATH_CALUDE_coexistent_pair_properties_l2210_221083


namespace NUMINAMATH_CALUDE_family_size_family_size_is_four_l2210_221031

theorem family_size (current_avg_age : ℝ) (youngest_age : ℝ) (birth_avg_age : ℝ) : ℝ :=
  let n := (youngest_age * birth_avg_age) / (current_avg_age - birth_avg_age - youngest_age)
  n

#check family_size 20 10 12.5

theorem family_size_is_four :
  family_size 20 10 12.5 = 4 := by sorry

end NUMINAMATH_CALUDE_family_size_family_size_is_four_l2210_221031


namespace NUMINAMATH_CALUDE_game_solution_l2210_221011

def game_result (x y z : ℚ) : Prop :=
  let a1 := x + y/3 + z/3
  let b1 := 2*y/3
  let c1 := 2*z/3
  let a2 := 2*a1/3
  let b2 := b1 + c1/3
  let c2 := 2*c1/3
  let a3 := 2*a2/3
  let b3 := 2*b2/3
  let c3 := c2 + b2/3 + a2/3
  x - a3 = 2 ∧ c3 - z = 2*z + 8 ∧ x + y + z < 1000

theorem game_solution :
  ∃ x y z : ℚ, game_result x y z ∧ x = 54 ∧ y = 162 ∧ z = 27 :=
by sorry

end NUMINAMATH_CALUDE_game_solution_l2210_221011


namespace NUMINAMATH_CALUDE_total_baseball_cards_l2210_221063

theorem total_baseball_cards (rob_doubles jess_doubles alex_doubles rob_total alex_total : ℕ) : 
  rob_doubles = 8 →
  jess_doubles = 40 →
  alex_doubles = 12 →
  rob_total = 24 →
  alex_total = 48 →
  rob_doubles * 3 = rob_total →
  jess_doubles = 5 * rob_doubles →
  alex_total = 2 * rob_total →
  alex_doubles * 4 = alex_total →
  rob_total + jess_doubles + alex_total = 112 := by
  sorry

end NUMINAMATH_CALUDE_total_baseball_cards_l2210_221063


namespace NUMINAMATH_CALUDE_inverse_f_at_negative_three_over_128_l2210_221023

noncomputable def f (x : ℝ) : ℝ := (x^7 - 1) / 4

theorem inverse_f_at_negative_three_over_128 :
  f⁻¹ (-3/128) = (29/32)^(1/7) := by
  sorry

end NUMINAMATH_CALUDE_inverse_f_at_negative_three_over_128_l2210_221023


namespace NUMINAMATH_CALUDE_ball_probability_l2210_221035

theorem ball_probability (m : ℕ) : 
  (3 : ℝ) / ((m : ℝ) + 3) = (1 : ℝ) / 4 → m = 9 := by
  sorry

end NUMINAMATH_CALUDE_ball_probability_l2210_221035


namespace NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l2210_221067

-- System 1
theorem system_one_solution (x y : ℝ) : 
  2 * x - y = 1 ∧ 7 * x - 3 * y = 4 → x = 1 ∧ y = 1 := by sorry

-- System 2
theorem system_two_solution (x y : ℝ) : 
  x / 2 + y / 3 = 6 ∧ x - y = -3 → x = 6 ∧ y = 9 := by sorry

end NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l2210_221067


namespace NUMINAMATH_CALUDE_tangent_slope_at_one_l2210_221039

-- Define the function f
variable (f : ℝ → ℝ)

-- State the conditions
axiom f_differentiable : Differentiable ℝ f
axiom limit_condition : ∀ ε > 0, ∃ δ > 0, ∀ x ∈ Set.Ioo (-δ) δ, 
  |((f (x + 1) - f 1) / (2 * x)) - 3| < ε

-- State the theorem
theorem tangent_slope_at_one : 
  (deriv f 1) = 6 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_one_l2210_221039


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l2210_221077

theorem z_in_first_quadrant : 
  ∃ (z : ℂ), (Complex.I + 1) * z = Complex.I^2013 ∧ 
  z.re > 0 ∧ z.im > 0 :=
by sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l2210_221077


namespace NUMINAMATH_CALUDE_Q_times_E_times_D_l2210_221022

def Q : ℂ := 3 + 4 * Complex.I
def E : ℂ := Complex.I ^ 2
def D : ℂ := 3 - 4 * Complex.I

theorem Q_times_E_times_D : Q * E * D = -25 := by
  sorry

end NUMINAMATH_CALUDE_Q_times_E_times_D_l2210_221022


namespace NUMINAMATH_CALUDE_quadratic_root_existence_l2210_221020

theorem quadratic_root_existence (a b c d : ℝ) (h : a * c = 2 * b + 2 * d) :
  (a^2 - 4*b ≥ 0) ∨ (c^2 - 4*d ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_existence_l2210_221020


namespace NUMINAMATH_CALUDE_fish_pond_estimation_l2210_221088

def fish_estimation (initial_catch : ℕ) (second_catch : ℕ) (marked_in_second : ℕ) : ℕ :=
  (initial_catch * second_catch) / marked_in_second

theorem fish_pond_estimation :
  let initial_catch := 200
  let second_catch := 200
  let marked_in_second := 8
  fish_estimation initial_catch second_catch marked_in_second = 5000 := by
  sorry

end NUMINAMATH_CALUDE_fish_pond_estimation_l2210_221088


namespace NUMINAMATH_CALUDE_tom_marble_combinations_l2210_221073

/-- Represents the number of marbles of each color -/
structure MarbleSet :=
  (red : ℕ)
  (blue : ℕ)
  (green : ℕ)
  (yellow : ℕ)

/-- Calculates the number of ways to choose 2 marbles from a given set -/
def chooseTwo (s : MarbleSet) : ℕ :=
  sorry

/-- Tom's marble set -/
def tomMarbles : MarbleSet :=
  { red := 1, blue := 1, green := 2, yellow := 3 }

theorem tom_marble_combinations :
  chooseTwo tomMarbles = 19 :=
sorry

end NUMINAMATH_CALUDE_tom_marble_combinations_l2210_221073


namespace NUMINAMATH_CALUDE_system_solution_l2210_221015

/-- Given a system of equations x + y = 2a and xy(x^2 + y^2) = 2b^4,
    this theorem states the condition for real solutions and
    provides the solutions for specific values of a and b. -/
theorem system_solution (a b : ℝ) (h : b^4 = 9375) :
  (∀ x y : ℝ, x + y = 2*a ∧ x*y*(x^2 + y^2) = 2*b^4 → a^2 ≥ b^2) ∧
  (a = 10 → ∃ x y : ℝ, (x = 15 ∧ y = 5 ∨ x = 5 ∧ y = 15) ∧
                       x + y = 2*a ∧ x*y*(x^2 + y^2) = 2*b^4) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2210_221015


namespace NUMINAMATH_CALUDE_two_valid_inequalities_l2210_221014

theorem two_valid_inequalities : 
  (∃ (f₁ f₂ f₃ : Prop), 
    (f₁ ↔ ∀ x : ℝ, Real.sqrt 5 + Real.sqrt 9 > 2 * Real.sqrt 7) ∧ 
    (f₂ ↔ ∀ a b c : ℝ, a^2 + 2*b^2 + 3*c^2 ≥ (1/6) * (a + 2*b + 3*c)^2) ∧ 
    (f₃ ↔ ∀ x : ℝ, Real.exp x ≥ x + 1) ∧ 
    (f₁ ∨ f₂ ∨ f₃) ∧ 
    (f₁ ∧ f₂ ∨ f₁ ∧ f₃ ∨ f₂ ∧ f₃) ∧ 
    ¬(f₁ ∧ f₂ ∧ f₃)) :=
by sorry

end NUMINAMATH_CALUDE_two_valid_inequalities_l2210_221014


namespace NUMINAMATH_CALUDE_bacteria_growth_without_offset_bacteria_growth_non_negative_without_offset_l2210_221068

/-- The number of bacteria after 60 minutes of doubling every minute, starting with 10 bacteria -/
def final_bacteria_count : ℕ := 10240

/-- The initial number of bacteria -/
def initial_bacteria_count : ℕ := 10

/-- The number of minutes in one hour -/
def minutes_in_hour : ℕ := 60

/-- Theorem stating that the initial number of bacteria without offset would be 0 -/
theorem bacteria_growth_without_offset :
  ∀ n : ℤ, (n + initial_bacteria_count) * 2^minutes_in_hour = final_bacteria_count → n = -initial_bacteria_count :=
by sorry

/-- Corollary stating that the non-negative initial number of bacteria without offset is 0 -/
theorem bacteria_growth_non_negative_without_offset :
  ∀ n : ℕ, (n + initial_bacteria_count) * 2^minutes_in_hour = final_bacteria_count → n = 0 :=
by sorry

end NUMINAMATH_CALUDE_bacteria_growth_without_offset_bacteria_growth_non_negative_without_offset_l2210_221068


namespace NUMINAMATH_CALUDE_square_diff_sum_l2210_221089

theorem square_diff_sum : 1010^2 - 990^2 - 1005^2 + 995^2 + 1012^2 - 988^2 = 68000 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_sum_l2210_221089


namespace NUMINAMATH_CALUDE_locus_and_max_dot_product_l2210_221002

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 8

-- Define point N
def N : ℝ × ℝ := (0, -1)

-- Define the locus C
def locus_C (x y : ℝ) : Prop := y^2 / 2 + x^2 = 1

-- Define the dot product OA · AN
def dot_product (x y : ℝ) : ℝ := -x^2 - y^2 - y

-- Theorem statement
theorem locus_and_max_dot_product :
  ∀ (x y : ℝ),
    (∃ (px py : ℝ), circle_M px py ∧
      (x - (px + 0) / 2)^2 + (y - (py + -1) / 2)^2 = ((px - 0)^2 + (py - -1)^2) / 4) →
    locus_C x y ∧
    (∀ (ax ay : ℝ), locus_C ax ay → dot_product ax ay ≤ -1/2) ∧
    (∃ (ax ay : ℝ), locus_C ax ay ∧ dot_product ax ay = -1/2) :=
by sorry


end NUMINAMATH_CALUDE_locus_and_max_dot_product_l2210_221002


namespace NUMINAMATH_CALUDE_factorial_division_l2210_221009

theorem factorial_division :
  (9 : ℕ).factorial / (4 : ℕ).factorial = 15120 :=
by
  have h1 : (9 : ℕ).factorial = 362880 := by sorry
  sorry

end NUMINAMATH_CALUDE_factorial_division_l2210_221009


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2210_221049

/-- Given that (1-i)z = 3+i, prove that z = 1 + 2i -/
theorem complex_equation_solution (z : ℂ) (h : (1 - Complex.I) * z = 3 + Complex.I) : 
  z = 1 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2210_221049


namespace NUMINAMATH_CALUDE_min_value_a_plus_b_l2210_221027

theorem min_value_a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (heq : 4 * a + b = a * b) :
  ∀ x y : ℝ, x > 0 → y > 0 → 4 * x + y = x * y → a + b ≤ x + y ∧ a + b = 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_a_plus_b_l2210_221027


namespace NUMINAMATH_CALUDE_image_of_3_4_preimages_of_1_neg6_l2210_221064

-- Define the mapping f
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 * p.2)

-- Theorem for the image of (3, 4)
theorem image_of_3_4 : f (3, 4) = (7, 12) := by sorry

-- Theorem for the pre-images of (1, -6)
theorem preimages_of_1_neg6 : 
  {p : ℝ × ℝ | f p = (1, -6)} = {(-2, 3), (3, -2)} := by sorry

end NUMINAMATH_CALUDE_image_of_3_4_preimages_of_1_neg6_l2210_221064


namespace NUMINAMATH_CALUDE_a_squared_minus_b_squared_l2210_221085

theorem a_squared_minus_b_squared (a b : ℚ) 
  (h1 : a + b = 2/3) 
  (h2 : a - b = 1/6) : 
  a^2 - b^2 = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_a_squared_minus_b_squared_l2210_221085


namespace NUMINAMATH_CALUDE_find_value_of_b_l2210_221036

/-- Given a configuration of numbers in circles with specific properties, prove the value of b. -/
theorem find_value_of_b (circle_sum : ℕ) (total_circles : ℕ) (total_sum : ℕ) 
  (overlap_sum : ℕ → ℕ → ℕ) (d_circle_sum : ℕ → ℕ) 
  (h1 : circle_sum = 21)
  (h2 : total_circles = 5)
  (h3 : total_sum = 69)
  (h4 : ∀ (b d : ℕ), overlap_sum b d = 2 + 8 + 9 + b + d)
  (h5 : ∀ (d : ℕ), d_circle_sum d = d + 5 + 9)
  (h6 : ∀ (d : ℕ), d_circle_sum d = circle_sum) :
  ∃ (b : ℕ), b = 10 := by
  sorry

end NUMINAMATH_CALUDE_find_value_of_b_l2210_221036


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2210_221075

theorem quadratic_factorization (a : ℝ) : a^2 - 6*a + 9 = (a - 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2210_221075


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2210_221095

theorem sufficient_not_necessary_condition (x : ℝ) : 
  (∀ x, x > 2 → x^2 - 3*x + 2 > 0) ∧ 
  (∃ x, x^2 - 3*x + 2 > 0 ∧ ¬(x > 2)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2210_221095


namespace NUMINAMATH_CALUDE_sequence_non_positive_l2210_221066

theorem sequence_non_positive (n : ℕ) (a : ℕ → ℝ)
  (h0 : a 0 = 0)
  (hn : a n = 0)
  (h_ineq : ∀ k, k < n → a k.pred - 2 * a k + a k.succ ≥ 0) :
  ∀ i, i ≤ n → a i ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_sequence_non_positive_l2210_221066


namespace NUMINAMATH_CALUDE_tan_addition_special_case_l2210_221004

theorem tan_addition_special_case (x : Real) (h : Real.tan x = 1/2) :
  Real.tan (x + π/3) = 7 + 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_addition_special_case_l2210_221004


namespace NUMINAMATH_CALUDE_money_left_l2210_221098

/-- Proves that if a person has 15 cents and spends 11 cents, they will have 4 cents left. -/
theorem money_left (initial_amount spent_amount : ℕ) : 
  initial_amount = 15 → spent_amount = 11 → initial_amount - spent_amount = 4 := by
  sorry

end NUMINAMATH_CALUDE_money_left_l2210_221098


namespace NUMINAMATH_CALUDE_cubic_equation_root_l2210_221080

theorem cubic_equation_root (k : ℚ) : 
  (∃ x : ℚ, 10 * k * x^3 - x - 9 = 0 ∧ x = -1) → k = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l2210_221080


namespace NUMINAMATH_CALUDE_sin_inequality_solution_set_l2210_221028

theorem sin_inequality_solution_set (a : ℝ) (θ : ℝ) (h1 : -1 < a) (h2 : a < 0) (h3 : θ = Real.arcsin a) :
  {x : ℝ | Real.sin x < a} = {x : ℝ | ∃ n : ℤ, (2 * n - 1) * π - θ < x ∧ x < 2 * n * π + θ} := by
  sorry

end NUMINAMATH_CALUDE_sin_inequality_solution_set_l2210_221028


namespace NUMINAMATH_CALUDE_increase_per_page_correct_l2210_221050

/-- The increase in drawings per page -/
def increase_per_page : ℕ := 5

/-- The number of drawings on the first page -/
def first_page_drawings : ℕ := 5

/-- The number of pages we're considering -/
def num_pages : ℕ := 5

/-- The total number of drawings on the first five pages -/
def total_drawings : ℕ := 75

/-- Theorem stating that the increase per page is correct -/
theorem increase_per_page_correct : 
  first_page_drawings + 
  (first_page_drawings + increase_per_page) + 
  (first_page_drawings + 2 * increase_per_page) + 
  (first_page_drawings + 3 * increase_per_page) + 
  (first_page_drawings + 4 * increase_per_page) = total_drawings :=
by sorry

end NUMINAMATH_CALUDE_increase_per_page_correct_l2210_221050
