import Mathlib

namespace NUMINAMATH_CALUDE_cut_cube_theorem_l98_9888

/-- Represents a cube that has been cut into smaller cubes -/
structure CutCube where
  -- The number of smaller cubes painted on exactly 2 faces
  two_face_cubes : ℕ
  -- The total number of smaller cubes created
  total_cubes : ℕ

/-- Theorem stating that a cube cut into equal smaller cubes with 12 two-face cubes results in 27 total cubes -/
theorem cut_cube_theorem (c : CutCube) (h : c.two_face_cubes = 12) : c.total_cubes = 27 := by
  sorry


end NUMINAMATH_CALUDE_cut_cube_theorem_l98_9888


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_six_sqrt_five_l98_9881

theorem sqrt_sum_equals_six_sqrt_five :
  Real.sqrt ((5 - 3 * Real.sqrt 5) ^ 2) + Real.sqrt ((5 + 3 * Real.sqrt 5) ^ 2) = 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_six_sqrt_five_l98_9881


namespace NUMINAMATH_CALUDE_f_min_at_neg_one_l98_9824

/-- The quadratic function we want to minimize -/
def f (x : ℝ) : ℝ := 3 * x^2 + 6 * x - 2

/-- The theorem stating that f is minimized at x = -1 -/
theorem f_min_at_neg_one :
  ∀ x : ℝ, f (-1) ≤ f x :=
sorry

end NUMINAMATH_CALUDE_f_min_at_neg_one_l98_9824


namespace NUMINAMATH_CALUDE_solution_set_part_i_range_of_m_part_ii_l98_9856

-- Define the function f
def f (x m : ℝ) : ℝ := |2*x| + |2*x + 3| + m

-- Part I
theorem solution_set_part_i : 
  {x : ℝ | f x (-2) ≤ 3} = {x : ℝ | -2 ≤ x ∧ x ≤ 1/2} := by sorry

-- Part II
theorem range_of_m_part_ii :
  ∀ m : ℝ, (∀ x < 0, f x m ≥ x + 2/x) → m ≥ -3 - 2*Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_part_i_range_of_m_part_ii_l98_9856


namespace NUMINAMATH_CALUDE_hidden_message_last_word_l98_9806

/-- Represents a color in the embroidery --/
inductive Color
| X | Dot | Ampersand | Colon | Star | GreaterThan | LessThan | S | Equals | Zh

/-- Represents a cell in the embroidery grid --/
structure Cell :=
  (number : ℕ)
  (color : Color)

/-- Represents the embroidery system --/
structure EmbroiderySystem :=
  (p : ℕ)
  (grid : List Cell)
  (letterMapping : Fin 33 → Fin 100)
  (colorMapping : Fin 10 → Color)

/-- Represents a decoded message --/
def DecodedMessage := List Char

/-- Function to decode the embroidery --/
def decodeEmbroidery (system : EmbroiderySystem) : DecodedMessage :=
  sorry

/-- The last word of the decoded message --/
def lastWord (message : DecodedMessage) : String :=
  sorry

/-- Theorem stating that the last word of the decoded message is "магистратура" --/
theorem hidden_message_last_word (system : EmbroiderySystem) :
  lastWord (decodeEmbroidery system) = "магистратура" :=
  sorry

end NUMINAMATH_CALUDE_hidden_message_last_word_l98_9806


namespace NUMINAMATH_CALUDE_debate_team_groups_l98_9800

/-- The number of boys on the debate team -/
def num_boys : ℕ := 28

/-- The number of girls on the debate team -/
def num_girls : ℕ := 4

/-- The minimum number of boys required in each group -/
def min_boys_per_group : ℕ := 2

/-- The minimum number of girls required in each group -/
def min_girls_per_group : ℕ := 1

/-- The maximum number of groups that can be formed -/
def max_groups : ℕ := 4

theorem debate_team_groups :
  (num_girls ≥ max_groups * min_girls_per_group) ∧
  (num_boys ≥ max_groups * min_boys_per_group) ∧
  (∀ n : ℕ, n > max_groups → 
    (num_girls < n * min_girls_per_group) ∨ 
    (num_boys < n * min_boys_per_group)) :=
sorry

end NUMINAMATH_CALUDE_debate_team_groups_l98_9800


namespace NUMINAMATH_CALUDE_power_multiplication_specific_power_multiplication_l98_9883

theorem power_multiplication (a b c : ℕ) : (10 : ℕ) ^ a * (10 : ℕ) ^ b = (10 : ℕ) ^ (a + b) := by
  sorry

theorem specific_power_multiplication : (10 : ℕ) ^ 65 * (10 : ℕ) ^ 64 = (10 : ℕ) ^ 129 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_specific_power_multiplication_l98_9883


namespace NUMINAMATH_CALUDE_shipment_average_weight_l98_9869

/-- Represents the weight distribution of boxes in a shipment. -/
structure Shipment where
  total_boxes : ℕ
  light_boxes : ℕ
  heavy_boxes : ℕ
  light_weight : ℕ
  heavy_weight : ℕ

/-- Calculates the average weight of boxes after removing some heavy boxes. -/
def new_average (s : Shipment) (removed : ℕ) : ℚ :=
  (s.light_boxes * s.light_weight + (s.heavy_boxes - removed) * s.heavy_weight) /
  (s.light_boxes + s.heavy_boxes - removed)

/-- Theorem stating the average weight of boxes in the shipment. -/
theorem shipment_average_weight (s : Shipment) :
  s.total_boxes = 20 ∧
  s.light_weight = 10 ∧
  s.heavy_weight = 20 ∧
  s.light_boxes + s.heavy_boxes = s.total_boxes ∧
  new_average s 10 = 16 →
  (s.light_boxes * s.light_weight + s.heavy_boxes * s.heavy_weight) / s.total_boxes = 39/2 := by
  sorry

#check shipment_average_weight

end NUMINAMATH_CALUDE_shipment_average_weight_l98_9869


namespace NUMINAMATH_CALUDE_unique_solution_when_a_is_three_fourths_l98_9857

/-- The equation has exactly one solution when a = 3/4 -/
theorem unique_solution_when_a_is_three_fourths (x a : ℝ) :
  (∃! x, (x^2 - a)^2 + 2*(x^2 - a) + (x - a) + 2 = 0) ↔ a = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_when_a_is_three_fourths_l98_9857


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_360_l98_9830

def sum_of_divisors (n : ℕ) : ℕ := sorry

theorem largest_prime_factor_of_sum_of_divisors_360 :
  ∃ (N : ℕ), sum_of_divisors 360 = N ∧ 
  (∀ p : ℕ, Nat.Prime p → p ∣ N → p ≤ 13) ∧
  13 ∣ N ∧ Nat.Prime 13 :=
sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_360_l98_9830


namespace NUMINAMATH_CALUDE_base4_to_decimal_conversion_l98_9808

/-- Converts a base-4 digit to its decimal value -/
def base4ToDecimal (digit : Nat) : Nat :=
  match digit with
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | _ => 0  -- Default case, should not occur in valid input

/-- Converts a list of base-4 digits to its decimal representation -/
def base4ListToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + (base4ToDecimal d) * (4 ^ (digits.length - 1 - i))) 0

theorem base4_to_decimal_conversion :
  base4ListToDecimal [0, 1, 3, 2, 0, 1, 3, 2] = 7710 := by
  sorry

#eval base4ListToDecimal [0, 1, 3, 2, 0, 1, 3, 2]

end NUMINAMATH_CALUDE_base4_to_decimal_conversion_l98_9808


namespace NUMINAMATH_CALUDE_correct_mean_calculation_l98_9843

theorem correct_mean_calculation (n : ℕ) (initial_mean : ℚ) 
  (incorrect_value1 incorrect_value2 incorrect_value3 : ℚ)
  (correct_value1 correct_value2 correct_value3 : ℚ) :
  n = 50 ∧ 
  initial_mean = 350 ∧
  incorrect_value1 = 150 ∧ correct_value1 = 180 ∧
  incorrect_value2 = 200 ∧ correct_value2 = 235 ∧
  incorrect_value3 = 270 ∧ correct_value3 = 290 →
  (n : ℚ) * initial_mean + (correct_value1 - incorrect_value1) + 
  (correct_value2 - incorrect_value2) + (correct_value3 - incorrect_value3) = n * 351.7 := by
  sorry

end NUMINAMATH_CALUDE_correct_mean_calculation_l98_9843


namespace NUMINAMATH_CALUDE_john_squat_wrap_vs_sleeves_l98_9886

/-- Given a raw squat weight, calculates the difference between
    the additional weight from wraps versus sleeves -/
def wrapVsSleevesDifference (rawSquat : ℝ) : ℝ :=
  0.25 * rawSquat - 30

theorem john_squat_wrap_vs_sleeves :
  wrapVsSleevesDifference 600 = 120 := by
  sorry

end NUMINAMATH_CALUDE_john_squat_wrap_vs_sleeves_l98_9886


namespace NUMINAMATH_CALUDE_ratio_subtraction_l98_9834

theorem ratio_subtraction (a b : ℚ) (h : a / b = 4 / 7) :
  (a - b) / b = -3 / 7 := by sorry

end NUMINAMATH_CALUDE_ratio_subtraction_l98_9834


namespace NUMINAMATH_CALUDE_solution_comparison_l98_9841

theorem solution_comparison (p q r s : ℝ) (hp : p ≠ 0) (hr : r ≠ 0) :
  (-q / p > -s / r) ↔ (s * r > q * p) :=
by sorry

end NUMINAMATH_CALUDE_solution_comparison_l98_9841


namespace NUMINAMATH_CALUDE_smallest_divisor_of_4500_l98_9878

theorem smallest_divisor_of_4500 : 
  ∀ n : ℕ, n > 0 ∧ n ∣ (4499 + 1) → n ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisor_of_4500_l98_9878


namespace NUMINAMATH_CALUDE_probability_theorem_l98_9859

def red_marbles : ℕ := 15
def blue_marbles : ℕ := 9
def green_marbles : ℕ := 6
def total_marbles : ℕ := red_marbles + blue_marbles + green_marbles

def probability_two_blue_one_red_one_green : ℚ :=
  (Nat.choose blue_marbles 2 * Nat.choose red_marbles 1 * Nat.choose green_marbles 1) /
  Nat.choose total_marbles 4

theorem probability_theorem :
  probability_two_blue_one_red_one_green = 5 / 812 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l98_9859


namespace NUMINAMATH_CALUDE_correct_senior_sample_l98_9875

/-- Represents the number of students to be selected from each grade level in a stratified sample -/
structure StratifiedSample where
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ
  seniors : ℕ

/-- Calculates the correct stratified sample given the school's demographics -/
def calculateStratifiedSample (totalStudents : ℕ) (freshmen : ℕ) (sophomoreProbability : ℚ) (sampleSize : ℕ) : StratifiedSample :=
  sorry

theorem correct_senior_sample :
  let totalStudents : ℕ := 2000
  let freshmen : ℕ := 760
  let sophomoreProbability : ℚ := 37/100
  let sampleSize : ℕ := 20
  let sample := calculateStratifiedSample totalStudents freshmen sophomoreProbability sampleSize
  sample.seniors = 5 := by sorry

end NUMINAMATH_CALUDE_correct_senior_sample_l98_9875


namespace NUMINAMATH_CALUDE_milk_delivery_proof_l98_9855

/-- The amount of milk in liters delivered to Minjeong's house in a week -/
def milk_per_week (bottles_per_day : ℕ) (liters_per_bottle : ℚ) (days_in_week : ℕ) : ℚ :=
  (bottles_per_day : ℚ) * liters_per_bottle * (days_in_week : ℚ)

/-- Proof that 4.2 liters of milk are delivered to Minjeong's house in a week -/
theorem milk_delivery_proof :
  milk_per_week 3 (2/10) 7 = 21/5 := by
  sorry

end NUMINAMATH_CALUDE_milk_delivery_proof_l98_9855


namespace NUMINAMATH_CALUDE_dot_product_NO_NM_l98_9825

-- Define the function f(x) = x^2 + 3
def f (x : ℝ) : ℝ := x^2 + 3

-- Define the theorem
theorem dot_product_NO_NM :
  ∀ x : ℝ,
  0 < x → x < 2 →
  let M : ℝ × ℝ := (x, f x)
  let N : ℝ × ℝ := (0, 1)
  let O : ℝ × ℝ := (0, 0)
  (M.1 - O.1)^2 + (M.2 - O.2)^2 = 27 →
  let NO : ℝ × ℝ := (N.1 - O.1, N.2 - O.2)
  let NM : ℝ × ℝ := (M.1 - N.1, M.2 - N.2)
  NO.1 * NM.1 + NO.2 * NM.2 = -4 :=
by sorry

end NUMINAMATH_CALUDE_dot_product_NO_NM_l98_9825


namespace NUMINAMATH_CALUDE_smallest_seven_digit_binary_l98_9812

theorem smallest_seven_digit_binary : ∀ n : ℕ, n > 0 → (
  (Nat.log 2 n + 1 = 7) ↔ n ≥ 64 ∧ ∀ m : ℕ, m > 0 ∧ m < 64 → Nat.log 2 m + 1 < 7
) := by sorry

end NUMINAMATH_CALUDE_smallest_seven_digit_binary_l98_9812


namespace NUMINAMATH_CALUDE_toms_final_amount_l98_9844

/-- Calculates the final amount of money Tom has after washing cars -/
def final_amount (initial_amount : ℝ) (supply_percentage : ℝ) (total_earnings : ℝ) (earnings_percentage : ℝ) : ℝ :=
  let amount_after_supplies := initial_amount * (1 - supply_percentage)
  let earnings := total_earnings * earnings_percentage
  amount_after_supplies + earnings

/-- Theorem stating that Tom's final amount is 114.5 dollars -/
theorem toms_final_amount :
  final_amount 74 0.15 86 0.6 = 114.5 := by
  sorry

end NUMINAMATH_CALUDE_toms_final_amount_l98_9844


namespace NUMINAMATH_CALUDE_equation_solution_l98_9802

theorem equation_solution (y : ℝ) (h : y ≠ 0) : 
  (2 / y + (3 / y) / (6 / y) = 1.5) → y = 2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l98_9802


namespace NUMINAMATH_CALUDE_sum_powers_equality_l98_9837

theorem sum_powers_equality (a b c d : ℝ) 
  (sum_eq : a + b = c + d) 
  (cube_sum_eq : a^3 + b^3 = c^3 + d^3) : 
  a^5 + b^5 = c^5 + d^5 ∧ 
  ∃ (a b c d : ℝ), (a + b = c + d) ∧ (a^3 + b^3 = c^3 + d^3) ∧ (a^4 + b^4 ≠ c^4 + d^4) := by
  sorry

end NUMINAMATH_CALUDE_sum_powers_equality_l98_9837


namespace NUMINAMATH_CALUDE_quadratic_sum_l98_9884

-- Define the quadratic function
def f (x : ℝ) : ℝ := -4 * x^2 + 20 * x - 88

-- Define the general form a(x+b)^2 + c
def g (a b c : ℝ) (x : ℝ) : ℝ := a * (x + b)^2 + c

-- Theorem statement
theorem quadratic_sum (a b c : ℝ) :
  (∀ x, f x = g a b c x) → a + b + c = -70.5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l98_9884


namespace NUMINAMATH_CALUDE_inverse_proportion_increasing_l98_9872

theorem inverse_proportion_increasing (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ < x₂ →
    (1 - 2*m) / x₁ < (1 - 2*m) / x₂) ↔
  m > 1/2 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_increasing_l98_9872


namespace NUMINAMATH_CALUDE_calculate_expression_l98_9829

theorem calculate_expression : (-1 : ℝ)^200 - (-1/2 : ℝ)^0 + (3⁻¹ : ℝ) * 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l98_9829


namespace NUMINAMATH_CALUDE_p_plus_q_equals_26_l98_9864

theorem p_plus_q_equals_26 (P Q : ℝ) :
  (∀ x : ℝ, x ≠ 3 → P / (x - 3) + Q * (x + 2) = (-2 * x^2 + 8 * x + 34) / (x - 3)) →
  P + Q = 26 := by
sorry

end NUMINAMATH_CALUDE_p_plus_q_equals_26_l98_9864


namespace NUMINAMATH_CALUDE_total_subjects_is_six_l98_9810

/-- 
Given a student's marks:
- The average mark in n subjects is 74
- The average mark in 5 subjects is 74
- The mark in the last subject is 74
Prove that the total number of subjects is 6
-/
theorem total_subjects_is_six (n : ℕ) (average_n : ℝ) (average_5 : ℝ) (last_subject : ℝ) :
  average_n = 74 →
  average_5 = 74 →
  last_subject = 74 →
  n * average_n = 5 * average_5 + last_subject →
  n = 6 := by
  sorry

end NUMINAMATH_CALUDE_total_subjects_is_six_l98_9810


namespace NUMINAMATH_CALUDE_intersection_A_B_union_A_B_range_of_a_l98_9826

-- Define the sets A, B, and C
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2 * x - 4 ≥ x - 2}
def C (a : ℝ) : Set ℝ := {x | 2 * x + a > 0}

-- Theorem statements
theorem intersection_A_B : A ∩ B = {x : ℝ | 2 ≤ x ∧ x < 3} := by sorry

theorem union_A_B : A ∪ B = {x : ℝ | x ≥ -1} := by sorry

theorem range_of_a (a : ℝ) (h : C a ∪ B = C a) : a > -4 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_A_B_range_of_a_l98_9826


namespace NUMINAMATH_CALUDE_number_equality_l98_9846

theorem number_equality (x : ℝ) (h1 : x > 0) (h2 : (2/3) * x = (49/216) * (1/x)) : x = 7/12 := by
  sorry

end NUMINAMATH_CALUDE_number_equality_l98_9846


namespace NUMINAMATH_CALUDE_monic_quadratic_with_complex_root_l98_9858

theorem monic_quadratic_with_complex_root :
  ∃ (a b : ℝ), ∀ (x : ℂ),
    (x^2 + a*x + b = 0 ↔ x = -3 - Complex.I * Real.sqrt 7 ∨ x = -3 + Complex.I * Real.sqrt 7) ∧
    (a = 6 ∧ b = 16) := by
  sorry

end NUMINAMATH_CALUDE_monic_quadratic_with_complex_root_l98_9858


namespace NUMINAMATH_CALUDE_right_triangle_sides_from_medians_l98_9861

/-- Given a right-angled triangle with medians ka and kb, prove the lengths of its sides. -/
theorem right_triangle_sides_from_medians (ka kb : ℝ) 
  (h_ka : ka = 30) (h_kb : kb = 40) : ∃ (a b c : ℝ),
  -- Definition of medians
  ka^2 = (1/4) * (2*b^2 + 2*c^2 - a^2) ∧ 
  kb^2 = (1/4) * (2*a^2 + 2*c^2 - b^2) ∧
  -- Pythagorean theorem
  a^2 + b^2 = c^2 ∧
  -- Side lengths
  a = 20 * Real.sqrt (11/3) ∧
  b = 40 / Real.sqrt 3 ∧
  c = 20 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_sides_from_medians_l98_9861


namespace NUMINAMATH_CALUDE_original_number_proof_l98_9807

theorem original_number_proof (x : ℝ) : x * 1.2 = 1800 → x = 1500 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l98_9807


namespace NUMINAMATH_CALUDE_dart_second_session_score_l98_9879

/-- Represents the points scored in each dart-throwing session -/
structure DartScores where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Checks if the given DartScores satisfy the problem conditions -/
def validScores (scores : DartScores) : Prop :=
  scores.second = 2 * scores.first ∧
  scores.third = 3 * scores.first ∧
  scores.first ≥ 8

theorem dart_second_session_score (scores : DartScores) :
  validScores scores → scores.second = 48 := by
  sorry

#check dart_second_session_score

end NUMINAMATH_CALUDE_dart_second_session_score_l98_9879


namespace NUMINAMATH_CALUDE_intersection_condition_l98_9862

open Set Real

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - m*x + 2 = 0}

theorem intersection_condition (m : ℝ) :
  A ∩ B m = B m ↔ ((-2 * sqrt 2 < m ∧ m < 2 * sqrt 2) ∨ m = 3) :=
sorry

end NUMINAMATH_CALUDE_intersection_condition_l98_9862


namespace NUMINAMATH_CALUDE_largest_multiple_under_500_l98_9828

theorem largest_multiple_under_500 : 
  ∀ n : ℕ, n > 0 ∧ 15 ∣ n ∧ n < 500 → n ≤ 495 :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_under_500_l98_9828


namespace NUMINAMATH_CALUDE_meals_left_theorem_l98_9822

/-- Calculates the number of meals left to be distributed given the initial number of meals,
    additional meals provided, and meals already distributed. -/
def meals_left_to_distribute (initial_meals : ℕ) (additional_meals : ℕ) (distributed_meals : ℕ) : ℕ :=
  initial_meals + additional_meals - distributed_meals

/-- Theorem stating that the number of meals left to distribute is correct
    given the problem conditions. -/
theorem meals_left_theorem (initial_meals additional_meals distributed_meals : ℕ)
    (h1 : initial_meals = 113)
    (h2 : additional_meals = 50)
    (h3 : distributed_meals = 85) :
    meals_left_to_distribute initial_meals additional_meals distributed_meals = 78 := by
  sorry

end NUMINAMATH_CALUDE_meals_left_theorem_l98_9822


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l98_9839

theorem expression_simplification_and_evaluation (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (1 - 1 / (x + 1)) / (x / (x^2 + 2*x + 1)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l98_9839


namespace NUMINAMATH_CALUDE_slope_of_line_l98_9836

-- Define a line with equation y = 3x + 1
def line (x : ℝ) : ℝ := 3 * x + 1

-- State the theorem: the slope of this line is 3
theorem slope_of_line :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (line x₂ - line x₁) / (x₂ - x₁) = 3) := by
sorry

end NUMINAMATH_CALUDE_slope_of_line_l98_9836


namespace NUMINAMATH_CALUDE_first_us_space_shuttle_is_columbia_l98_9866

/-- Represents a space shuttle -/
structure SpaceShuttle where
  name : String
  country : String
  year : Nat
  manned_flight_completed : Bool

/-- The world's first space shuttle developed by the United States in 1981 -/
def first_us_space_shuttle : SpaceShuttle :=
  { name := "Columbia"
  , country := "United States"
  , year := 1981
  , manned_flight_completed := true }

/-- Theorem stating that the first US space shuttle's name is Columbia -/
theorem first_us_space_shuttle_is_columbia :
  first_us_space_shuttle.name = "Columbia" :=
by sorry

end NUMINAMATH_CALUDE_first_us_space_shuttle_is_columbia_l98_9866


namespace NUMINAMATH_CALUDE_two_cyclists_problem_l98_9835

/-- Two cyclists problem -/
theorem two_cyclists_problem (north_speed : ℝ) (time : ℝ) (distance : ℝ) : 
  north_speed = 30 →
  time = 0.7142857142857143 →
  distance = 50 →
  ∃ (south_speed : ℝ), south_speed = 40 ∧ distance = (north_speed + south_speed) * time :=
by sorry

end NUMINAMATH_CALUDE_two_cyclists_problem_l98_9835


namespace NUMINAMATH_CALUDE_alexis_initial_budget_l98_9827

/-- Alexis's shopping expenses and remaining budget --/
structure ShoppingBudget where
  shirt : ℕ
  pants : ℕ
  coat : ℕ
  socks : ℕ
  belt : ℕ
  shoes : ℕ
  remaining : ℕ

/-- Calculate the initial budget given the shopping expenses and remaining amount --/
def initialBudget (s : ShoppingBudget) : ℕ :=
  s.shirt + s.pants + s.coat + s.socks + s.belt + s.shoes + s.remaining

/-- Alexis's actual shopping expenses and remaining budget --/
def alexisShopping : ShoppingBudget :=
  { shirt := 30
  , pants := 46
  , coat := 38
  , socks := 11
  , belt := 18
  , shoes := 41
  , remaining := 16 }

/-- Theorem stating that Alexis's initial budget was $200 --/
theorem alexis_initial_budget :
  initialBudget alexisShopping = 200 := by
  sorry

end NUMINAMATH_CALUDE_alexis_initial_budget_l98_9827


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_l98_9801

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_perpendicular 
  (a b : Line) (α : Plane) :
  parallel a b → perpendicular a α → perpendicular b α :=
by sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_l98_9801


namespace NUMINAMATH_CALUDE_valid_x_values_l98_9893

def is_valid_x (x : ℕ) : Prop :=
  13 ≤ x ∧ x ≤ 20 ∧
  (132 + x) % 3 = 0 ∧
  ∃ (s : ℕ), 3 * s = 132 + 3 * x

theorem valid_x_values :
  ∀ x : ℕ, is_valid_x x ↔ (x = 15 ∨ x = 18) :=
sorry

end NUMINAMATH_CALUDE_valid_x_values_l98_9893


namespace NUMINAMATH_CALUDE_clover_total_distance_l98_9894

/-- Clover's daily morning walk distance in miles -/
def morning_walk : ℝ := 1.5

/-- Clover's daily evening walk distance in miles -/
def evening_walk : ℝ := 1.5

/-- Number of days Clover walks -/
def days : ℕ := 30

/-- Theorem stating the total distance Clover walks in 30 days -/
theorem clover_total_distance : 
  (morning_walk + evening_walk) * days = 90 := by
  sorry

end NUMINAMATH_CALUDE_clover_total_distance_l98_9894


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l98_9847

def M : Set ℕ := {1, 2, 3, 6, 7}
def N : Set ℕ := {1, 2, 4, 5}

theorem intersection_of_M_and_N : M ∩ N = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l98_9847


namespace NUMINAMATH_CALUDE_garden_perimeter_l98_9873

theorem garden_perimeter :
  ∀ (length breadth perimeter : ℝ),
    length = 258 →
    breadth = 82 →
    perimeter = 2 * (length + breadth) →
    perimeter = 680 := by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_l98_9873


namespace NUMINAMATH_CALUDE_markov_equation_solution_l98_9838

theorem markov_equation_solution (m n p : ℕ) : 
  m^2 + n^2 + p^2 = m * n * p → 
  ∃ m₁ n₁ p₁ : ℕ, m = 3 * m₁ ∧ n = 3 * n₁ ∧ p = 3 * p₁ ∧ 
  m₁^2 + n₁^2 + p₁^2 = 3 * m₁ * n₁ * p₁ := by
sorry

end NUMINAMATH_CALUDE_markov_equation_solution_l98_9838


namespace NUMINAMATH_CALUDE_negation_equivalence_l98_9818

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x > 0 ∧ x^2 - 2*x - 3 ≤ 0) ↔ (∀ x : ℝ, x > 0 → x^2 - 2*x - 3 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l98_9818


namespace NUMINAMATH_CALUDE_min_y_value_l98_9811

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 14*x + 48*y) : y ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_min_y_value_l98_9811


namespace NUMINAMATH_CALUDE_greatest_abcba_div_by_11_and_3_l98_9804

def is_abcba (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), 
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    n = a * 10000 + b * 1000 + c * 100 + b * 10 + a

theorem greatest_abcba_div_by_11_and_3 : 
  (∀ n : ℕ, is_abcba n → n % 11 = 0 → n % 3 = 0 → n ≤ 96569) ∧ 
  is_abcba 96569 ∧ 
  96569 % 11 = 0 ∧ 
  96569 % 3 = 0 :=
sorry

end NUMINAMATH_CALUDE_greatest_abcba_div_by_11_and_3_l98_9804


namespace NUMINAMATH_CALUDE_young_inequality_l98_9885

theorem young_inequality (p q a b : ℝ) (hp : 0 < p) (hq : 0 < q) (hpq : 1/p + 1/q = 1) (ha : 0 < a) (hb : 0 < b) :
  a * b ≤ a^p / p + b^q / q := by
  sorry

end NUMINAMATH_CALUDE_young_inequality_l98_9885


namespace NUMINAMATH_CALUDE_money_redistribution_total_l98_9854

/-- Represents the money redistribution problem with three friends -/
def MoneyRedistribution (a j t : ℚ) : Prop :=
  -- Initial conditions
  t = 36 ∧
  -- After Amy's redistribution
  ∃ a1 j1 t1,
    t1 = 2 * t ∧
    j1 = 2 * j ∧
    a1 = a - (t + j) ∧
    -- After Jan's redistribution
    ∃ a2 j2 t2,
      t2 = 2 * t1 ∧
      a2 = 2 * a1 ∧
      j2 = 2 * j - (a1 + 72) ∧
      -- After Toy's redistribution
      ∃ a3 j3 t3,
        a3 = 2 * a2 ∧
        j3 = 2 * j2 ∧
        t3 = t2 - (a2 + j2) ∧
        t3 = 36 ∧
        a3 + j3 + t3 = 252

/-- The theorem stating that the total amount of money is 252 -/
theorem money_redistribution_total (a j t : ℚ) :
  MoneyRedistribution a j t → a + j + t = 252 := by
  sorry

end NUMINAMATH_CALUDE_money_redistribution_total_l98_9854


namespace NUMINAMATH_CALUDE_classmates_not_invited_l98_9814

/-- A simple graph representing friendships among classmates -/
structure FriendshipGraph where
  vertices : Finset Nat
  edges : Finset (Nat × Nat)
  symm : ∀ {a b}, (a, b) ∈ edges → (b, a) ∈ edges
  irrefl : ∀ a, (a, a) ∉ edges

/-- The set of vertices reachable within n steps from a given vertex -/
def reachableWithin (G : FriendshipGraph) (start : Nat) (n : Nat) : Finset Nat :=
  sorry

/-- The main theorem -/
theorem classmates_not_invited (G : FriendshipGraph) (mark : Nat) : 
  G.vertices.card = 25 →
  mark ∈ G.vertices →
  (G.vertices \ reachableWithin G mark 3).card = 5 := by
  sorry

end NUMINAMATH_CALUDE_classmates_not_invited_l98_9814


namespace NUMINAMATH_CALUDE_bcm_percentage_is_twenty_percent_l98_9845

/-- The percentage of Black Copper Marans (BCM) in a flock of chickens -/
def bcm_percentage (total_chickens : ℕ) (bcm_hen_percentage : ℚ) (bcm_hens : ℕ) : ℚ :=
  (bcm_hens : ℚ) / (bcm_hen_percentage * total_chickens)

/-- Theorem stating that the percentage of BCM in a flock of 100 chickens is 20%,
    given that 80% of BCM are hens and there are 16 BCM hens -/
theorem bcm_percentage_is_twenty_percent :
  bcm_percentage 100 (4/5) 16 = 1/5 := by
  sorry

#eval bcm_percentage 100 (4/5) 16

end NUMINAMATH_CALUDE_bcm_percentage_is_twenty_percent_l98_9845


namespace NUMINAMATH_CALUDE_expression_C_is_negative_l98_9815

-- Define the variables with their approximate values
def A : ℝ := -4.2
def B : ℝ := 2.3
def C : ℝ := -0.5
def D : ℝ := 3.4
def E : ℝ := -1.8

-- Theorem stating that the expression (D/B) * C is negative
theorem expression_C_is_negative : (D / B) * C < 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_C_is_negative_l98_9815


namespace NUMINAMATH_CALUDE_student_arrangement_count_l98_9891

/-- The number of students in the row -/
def total_students : ℕ := 4

/-- The number of students that must stand next to each other -/
def adjacent_students : ℕ := 2

/-- The number of different arrangements of students -/
def num_arrangements : ℕ := 12

/-- 
Theorem: Given 4 students standing in a row, where 2 specific students 
must stand next to each other, the number of different arrangements is 12.
-/
theorem student_arrangement_count :
  (total_students = 4) →
  (adjacent_students = 2) →
  (num_arrangements = 12) :=
by sorry

end NUMINAMATH_CALUDE_student_arrangement_count_l98_9891


namespace NUMINAMATH_CALUDE_largest_prime_factors_difference_l98_9897

theorem largest_prime_factors_difference (n : Nat) (h : n = 165033) :
  ∃ (p q : Nat), Nat.Prime p ∧ Nat.Prime q ∧ p > q ∧
  (∀ (r : Nat), Nat.Prime r ∧ r ∣ n → r ≤ p) ∧
  (∀ (r : Nat), Nat.Prime r ∧ r ∣ n ∧ r ≠ p → r ≤ q) ∧
  p - q = 140 := by
  sorry

#eval 165033

end NUMINAMATH_CALUDE_largest_prime_factors_difference_l98_9897


namespace NUMINAMATH_CALUDE_right_triangle_area_l98_9849

theorem right_triangle_area (a b c : ℝ) (h1 : a = 24) (h2 : c = 25) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 84 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l98_9849


namespace NUMINAMATH_CALUDE_chord_equation_l98_9805

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents an ellipse -/
structure Ellipse :=
  (a : ℝ)
  (b : ℝ)

/-- Checks if a point is inside an ellipse -/
def isInside (p : Point) (e : Ellipse) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) < 1

/-- Represents a line in 2D space -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- Checks if a point bisects a chord of an ellipse -/
def bisectsChord (p : Point) (e : Ellipse) : Prop :=
  sorry  -- Definition of bisecting a chord

theorem chord_equation (e : Ellipse) (m : Point) :
  e.a = 4 →
  e.b = 2 →
  m.x = 2 →
  m.y = 1 →
  isInside m e →
  bisectsChord m e →
  ∃ l : Line, l.a = 1 ∧ l.b = 2 ∧ l.c = -4 :=
sorry

end NUMINAMATH_CALUDE_chord_equation_l98_9805


namespace NUMINAMATH_CALUDE_prob_non_yellow_specific_l98_9863

/-- The probability of selecting a non-yellow jelly bean -/
def prob_non_yellow (red green yellow blue : ℕ) : ℚ :=
  (red + green + blue) / (red + green + yellow + blue)

/-- Theorem: The probability of selecting a non-yellow jelly bean from a bag
    containing 4 red, 7 green, 9 yellow, and 10 blue jelly beans is 7/10 -/
theorem prob_non_yellow_specific : prob_non_yellow 4 7 9 10 = 7/10 := by
  sorry

end NUMINAMATH_CALUDE_prob_non_yellow_specific_l98_9863


namespace NUMINAMATH_CALUDE_class_A_student_count_l98_9898

theorem class_A_student_count :
  ∀ (girls boys : ℕ),
    girls = 25 →
    girls = boys + 3 →
    girls + boys = 47 :=
by sorry

end NUMINAMATH_CALUDE_class_A_student_count_l98_9898


namespace NUMINAMATH_CALUDE_total_time_first_to_seventh_l98_9853

/-- Represents the travel times between stations in hours -/
def travel_times : List Real := [3, 2, 1.5, 4, 1, 2.5]

/-- Represents the break times at stations in minutes -/
def break_times : List Real := [45, 30, 15]

/-- Converts hours to minutes -/
def hours_to_minutes (hours : Real) : Real := hours * 60

/-- Calculates the total travel time in minutes -/
def total_travel_time : Real := (travel_times.map hours_to_minutes).sum

/-- Calculates the total break time in minutes -/
def total_break_time : Real := break_times.sum

/-- Theorem stating the total time from first to seventh station -/
theorem total_time_first_to_seventh : 
  total_travel_time + total_break_time = 930 := by sorry

end NUMINAMATH_CALUDE_total_time_first_to_seventh_l98_9853


namespace NUMINAMATH_CALUDE_expression_value_l98_9867

theorem expression_value : 
  let x : ℕ := 3
  x + x * (x ^ (x + 1)) = 246 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l98_9867


namespace NUMINAMATH_CALUDE_toothpick_grid_theorem_l98_9880

/-- Calculates the number of unique toothpicks in a rectangular grid frame. -/
def unique_toothpicks (height width : ℕ) : ℕ :=
  let horizontal_toothpicks := (height + 1) * width
  let vertical_toothpicks := height * (width + 1)
  let intersections := (height + 1) * (width + 1)
  horizontal_toothpicks + vertical_toothpicks - intersections

/-- Theorem stating that a 15x8 toothpick grid uses 119 unique toothpicks. -/
theorem toothpick_grid_theorem :
  unique_toothpicks 15 8 = 119 := by
  sorry

#eval unique_toothpicks 15 8

end NUMINAMATH_CALUDE_toothpick_grid_theorem_l98_9880


namespace NUMINAMATH_CALUDE_extended_volume_calculation_l98_9860

/-- Represents the dimensions of a rectangular parallelepiped -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of the set of points that are either inside or within one unit
    of a rectangular parallelepiped with the given dimensions -/
def extended_volume (d : Dimensions) : ℝ :=
  sorry

/-- The dimensions of the given rectangular parallelepiped -/
def given_dimensions : Dimensions :=
  { length := 2, width := 3, height := 7 }

theorem extended_volume_calculation :
  extended_volume given_dimensions = (372 + 112 * Real.pi) / 3 := by
  sorry

end NUMINAMATH_CALUDE_extended_volume_calculation_l98_9860


namespace NUMINAMATH_CALUDE_regression_line_intercept_l98_9803

/-- Represents a linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- The regression line passes through a given point -/
def passes_through (line : RegressionLine) (x y : ℝ) : Prop :=
  y = line.slope * x + line.intercept

theorem regression_line_intercept (b : ℝ) (x₀ y₀ : ℝ) :
  let line := RegressionLine.mk b ((y₀ : ℝ) - b * x₀)
  passes_through line x₀ y₀ ∧ line.slope = 1.23 ∧ x₀ = 4 ∧ y₀ = 5 →
  line.intercept = 0.08 := by
  sorry

end NUMINAMATH_CALUDE_regression_line_intercept_l98_9803


namespace NUMINAMATH_CALUDE_arithmetic_mean_property_l98_9852

def number_set : List Nat := [9, 9999, 99999999, 999999999999, 9999999999999999, 99999999999999999999]

def arithmetic_mean (xs : List Nat) : Nat :=
  xs.sum / xs.length

def has_18_digits (n : Nat) : Prop :=
  n ≥ 10^17 ∧ n < 10^18

def all_digits_distinct (n : Nat) : Prop :=
  ∀ i j, i ≠ j → (n / 10^i) % 10 ≠ (n / 10^j) % 10

def does_not_contain_4 (n : Nat) : Prop :=
  ∀ i, (n / 10^i) % 10 ≠ 4

theorem arithmetic_mean_property :
  let mean := arithmetic_mean number_set
  has_18_digits mean ∧ all_digits_distinct mean ∧ does_not_contain_4 mean :=
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_property_l98_9852


namespace NUMINAMATH_CALUDE_max_elves_theorem_l98_9877

/-- Represents the type of inhabitant -/
inductive InhabitantType
| Elf
| Dwarf

/-- Represents whether an inhabitant wears a cap -/
inductive CapStatus
| WithCap
| WithoutCap

/-- Represents the statement an inhabitant can make -/
inductive Statement
| RightIsElf
| RightHasCap

/-- Represents an inhabitant in the line -/
structure Inhabitant :=
  (type : InhabitantType)
  (capStatus : CapStatus)
  (statement : Statement)

/-- Determines if an inhabitant tells the truth based on their type and cap status -/
def tellsTruth (i : Inhabitant) : Prop :=
  match i.type, i.capStatus with
  | InhabitantType.Elf, CapStatus.WithoutCap => True
  | InhabitantType.Elf, CapStatus.WithCap => False
  | InhabitantType.Dwarf, CapStatus.WithoutCap => False
  | InhabitantType.Dwarf, CapStatus.WithCap => True

/-- Represents the line of inhabitants -/
def Line := Vector Inhabitant 60

/-- Checks if the line configuration is valid according to the problem rules -/
def isValidLine (line : Line) : Prop := sorry

/-- Counts the number of elves without caps in the line -/
def countElvesWithoutCaps (line : Line) : Nat := sorry

/-- Counts the number of elves with caps in the line -/
def countElvesWithCaps (line : Line) : Nat := sorry

/-- Main theorem: Maximum number of elves without caps is 59 and with caps is 30 -/
theorem max_elves_theorem (line : Line) (h : isValidLine line) : 
  countElvesWithoutCaps line ≤ 59 ∧ countElvesWithCaps line ≤ 30 := by sorry

end NUMINAMATH_CALUDE_max_elves_theorem_l98_9877


namespace NUMINAMATH_CALUDE_scientific_notation_of_small_number_l98_9874

theorem scientific_notation_of_small_number : 
  ∃ (a : ℝ) (n : ℤ), 0.0000001 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_small_number_l98_9874


namespace NUMINAMATH_CALUDE_square_of_binomial_constant_l98_9887

theorem square_of_binomial_constant (a : ℚ) : 
  (∃ b : ℚ, ∀ x : ℚ, 9*x^2 + 27*x + a = (3*x + b)^2) → a = 81/4 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_constant_l98_9887


namespace NUMINAMATH_CALUDE_function_point_relation_l98_9895

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the inverse function of f
variable (f_inv : ℝ → ℝ)

-- State that f_inv is indeed the inverse of f
axiom inverse_relation : ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x

-- Given condition: f^(-1)(-2) = 0
axiom condition : f_inv (-2) = 0

-- Theorem to prove
theorem function_point_relation :
  f (-5 + 5) = -2 :=
sorry

end NUMINAMATH_CALUDE_function_point_relation_l98_9895


namespace NUMINAMATH_CALUDE_inequality_and_function_property_l98_9868

def f (x : ℝ) : ℝ := |x - 1|

theorem inequality_and_function_property :
  (∀ x : ℝ, f (x - 1) + f (x + 3) ≥ 6 ↔ x ≤ -3 ∨ x ≥ 3) ∧
  (∀ a b : ℝ, |a| < 1 → |b| < 1 → a ≠ 0 → f (a * b) > |a| * f (b / a)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_function_property_l98_9868


namespace NUMINAMATH_CALUDE_square_odd_implies_odd_l98_9871

theorem square_odd_implies_odd (n : ℤ) : Odd (n^2) → Odd n := by
  sorry

end NUMINAMATH_CALUDE_square_odd_implies_odd_l98_9871


namespace NUMINAMATH_CALUDE_no_solution_exists_l98_9809

theorem no_solution_exists : ¬∃ (x y : ℤ), (x + y = 2021 ∧ (10*x + y = 2221 ∨ x + 10*y = 2221)) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l98_9809


namespace NUMINAMATH_CALUDE_mikes_training_time_l98_9813

/-- Proves that Mike trained for 1 hour per day during the first week -/
theorem mikes_training_time (x : ℝ) : 
  (7 * x + 7 * (x + 3) = 35) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_mikes_training_time_l98_9813


namespace NUMINAMATH_CALUDE_factory_production_l98_9848

/-- The total number of cars made by a factory over two days, given the production on the first day and that the second day's production is twice the first day's. -/
def total_cars (first_day_production : ℕ) : ℕ :=
  first_day_production + 2 * first_day_production

/-- Theorem stating that the total number of cars made over two days is 180,
    given that 60 cars were made on the first day. -/
theorem factory_production : total_cars 60 = 180 := by
  sorry

end NUMINAMATH_CALUDE_factory_production_l98_9848


namespace NUMINAMATH_CALUDE_coffee_purchase_problem_l98_9892

/-- Given a gift card balance, coffee price per pound, and remaining balance,
    calculate the number of pounds of coffee purchased. -/
def coffee_pounds_purchased (gift_card_balance : ℚ) (coffee_price_per_pound : ℚ) (remaining_balance : ℚ) : ℚ :=
  (gift_card_balance - remaining_balance) / coffee_price_per_pound

theorem coffee_purchase_problem :
  let gift_card_balance : ℚ := 70
  let coffee_price_per_pound : ℚ := 8.58
  let remaining_balance : ℚ := 35.68
  coffee_pounds_purchased gift_card_balance coffee_price_per_pound remaining_balance = 4 := by
  sorry

end NUMINAMATH_CALUDE_coffee_purchase_problem_l98_9892


namespace NUMINAMATH_CALUDE_grandma_olga_grandchildren_l98_9876

/-- Represents the number of grandchildren Grandma Olga has -/
def total_grandchildren (num_daughters num_sons : ℕ) 
                        (sons_per_daughter daughters_per_son : ℕ) : ℕ :=
  num_daughters * sons_per_daughter + num_sons * daughters_per_son

/-- Theorem stating that Grandma Olga has 33 grandchildren -/
theorem grandma_olga_grandchildren : 
  total_grandchildren 3 3 6 5 = 33 := by
  sorry

end NUMINAMATH_CALUDE_grandma_olga_grandchildren_l98_9876


namespace NUMINAMATH_CALUDE_min_type_a_robots_l98_9842

/-- Represents the material handling capacity of robot A in kg per hour -/
def robot_a_capacity : ℝ := 150

/-- Represents the material handling capacity of robot B in kg per hour -/
def robot_b_capacity : ℝ := 120

/-- Represents the total number of robots to be purchased -/
def total_robots : ℕ := 20

/-- Represents the minimum material handling requirement in kg per hour -/
def min_handling_requirement : ℝ := 2800

/-- Calculates the total material handling capacity given the number of type A robots -/
def total_capacity (num_a : ℕ) : ℝ :=
  (num_a : ℝ) * robot_a_capacity + ((total_robots - num_a) : ℝ) * robot_b_capacity

/-- States that the minimum number of type A robots required is 14 -/
theorem min_type_a_robots : 
  ∀ n : ℕ, n < 14 → total_capacity n < min_handling_requirement ∧
  total_capacity 14 ≥ min_handling_requirement := by sorry

end NUMINAMATH_CALUDE_min_type_a_robots_l98_9842


namespace NUMINAMATH_CALUDE_problem_solution_l98_9817

theorem problem_solution (a e : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 45 * e) : e = 49 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l98_9817


namespace NUMINAMATH_CALUDE_smallest_n_square_and_cube_l98_9820

theorem smallest_n_square_and_cube : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), 4 * n = k^2) ∧ 
  (∃ (j : ℕ), 5 * n = j^3) ∧
  (∀ (m : ℕ), m > 0 → 
    ((∃ (k : ℕ), 4 * m = k^2) ∧ (∃ (j : ℕ), 5 * m = j^3)) → 
    m ≥ 500) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_square_and_cube_l98_9820


namespace NUMINAMATH_CALUDE_logging_time_is_ten_months_l98_9870

/-- Represents the forest and logging scenario --/
structure LoggingScenario where
  forestLength : ℕ
  forestWidth : ℕ
  treesPerSquareMile : ℕ
  loggersCount : ℕ
  treesPerLoggerPerDay : ℕ
  daysPerMonth : ℕ

/-- Calculates the number of months required to cut down all trees --/
def monthsToLogForest (scenario : LoggingScenario) : ℚ :=
  let totalArea := scenario.forestLength * scenario.forestWidth
  let totalTrees := totalArea * scenario.treesPerSquareMile
  let treesPerDay := scenario.loggersCount * scenario.treesPerLoggerPerDay
  (totalTrees : ℚ) / (treesPerDay * scenario.daysPerMonth)

/-- Theorem stating that it takes 10 months to log the forest under given conditions --/
theorem logging_time_is_ten_months :
  let scenario : LoggingScenario := {
    forestLength := 4,
    forestWidth := 6,
    treesPerSquareMile := 600,
    loggersCount := 8,
    treesPerLoggerPerDay := 6,
    daysPerMonth := 30
  }
  monthsToLogForest scenario = 10 := by sorry

end NUMINAMATH_CALUDE_logging_time_is_ten_months_l98_9870


namespace NUMINAMATH_CALUDE_initial_journey_speed_l98_9833

/-- Proves that the speed of the initial journey is 63 mph given the conditions -/
theorem initial_journey_speed (d : ℝ) (v : ℝ) (h1 : v > 0) : 
  (2 * d) / (d / v + 2 * (d / v)) = 42 → v = 63 := by
  sorry

end NUMINAMATH_CALUDE_initial_journey_speed_l98_9833


namespace NUMINAMATH_CALUDE_triangle_altitude_on_square_diagonal_l98_9899

theorem triangle_altitude_on_square_diagonal (s : ℝ) (h : s > 0) :
  let square_area := s^2
  let diagonal := s * Real.sqrt 2
  let triangle_area := (1/2) * diagonal * altitude
  ∃ altitude : ℝ, 
    (square_area = triangle_area) ∧ 
    (altitude = s * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_altitude_on_square_diagonal_l98_9899


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l98_9865

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  (∀ n, S n = (n : ℝ) / 2 * (a 1 + a n)) →  -- sum formula
  (a 4 + a 8 = 4) →  -- given condition
  (S 11 + a 6 = 24) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l98_9865


namespace NUMINAMATH_CALUDE_modulus_of_z_l98_9821

theorem modulus_of_z (z : ℂ) (h : (2 * z) / (1 - z) = Complex.I) : Complex.abs z = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l98_9821


namespace NUMINAMATH_CALUDE_largest_of_five_consecutive_integers_l98_9850

theorem largest_of_five_consecutive_integers (a b c d e : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧  -- all positive
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 ∧  -- consecutive
  a * b * c * d * e = 15120 →  -- product is 15120
  e = 10 :=  -- largest is 10
by sorry

end NUMINAMATH_CALUDE_largest_of_five_consecutive_integers_l98_9850


namespace NUMINAMATH_CALUDE_xyz_sum_mod_9_l98_9889

theorem xyz_sum_mod_9 (x y z : ℕ) : 
  0 < x ∧ x < 9 ∧
  0 < y ∧ y < 9 ∧
  0 < z ∧ z < 9 ∧
  (x * y * z) % 9 = 1 ∧
  (7 * z) % 9 = 4 ∧
  (8 * y) % 9 = (5 + y) % 9 →
  (x + y + z) % 9 = 7 := by
sorry

end NUMINAMATH_CALUDE_xyz_sum_mod_9_l98_9889


namespace NUMINAMATH_CALUDE_expression_factorization_l98_9851

theorem expression_factorization (b : ℝ) :
  (8 * b^3 + 104 * b^2 - 9) - (-9 * b^3 + b^2 - 9) = b^2 * (17 * b + 103) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l98_9851


namespace NUMINAMATH_CALUDE_expression_simplification_l98_9882

theorem expression_simplification (x : ℝ) (h : x = 2) :
  (1 / (x + 1) - 1) / ((x^3 - x) / (x^2 + 2*x + 1)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l98_9882


namespace NUMINAMATH_CALUDE_madeline_class_hours_l98_9890

/-- Calculates the number of hours Madeline spends in class per week -/
def hours_in_class (hours_per_day : ℕ) (days_per_week : ℕ) 
  (homework_hours_per_day : ℕ) (sleep_hours_per_day : ℕ) 
  (work_hours_per_week : ℕ) (leftover_hours : ℕ) : ℕ :=
  hours_per_day * days_per_week - 
  (homework_hours_per_day * days_per_week + 
   sleep_hours_per_day * days_per_week + 
   work_hours_per_week + 
   leftover_hours)

theorem madeline_class_hours : 
  hours_in_class 24 7 4 8 20 46 = 18 := by
  sorry

end NUMINAMATH_CALUDE_madeline_class_hours_l98_9890


namespace NUMINAMATH_CALUDE_probability_of_no_defective_pens_l98_9819

theorem probability_of_no_defective_pens (total_pens : Nat) (defective_pens : Nat) (pens_bought : Nat) :
  total_pens = 12 →
  defective_pens = 3 →
  pens_bought = 2 →
  (1 - defective_pens / total_pens) * (1 - (defective_pens) / (total_pens - 1)) = 6/11 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_no_defective_pens_l98_9819


namespace NUMINAMATH_CALUDE_minimum_guests_l98_9831

theorem minimum_guests (total_food : ℕ) (max_per_guest : ℕ) (h1 : total_food = 327) (h2 : max_per_guest = 2) :
  ∃ (min_guests : ℕ), min_guests = 164 ∧ min_guests * max_per_guest ≥ total_food ∧ (min_guests - 1) * max_per_guest < total_food :=
by
  sorry

end NUMINAMATH_CALUDE_minimum_guests_l98_9831


namespace NUMINAMATH_CALUDE_range_of_expression_l98_9840

theorem range_of_expression (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hxy : x + y ≤ 1) :
  2/3 ≤ 4*x^2 + 4*y^2 + (1 - x - y)^2 ∧ 4*x^2 + 4*y^2 + (1 - x - y)^2 ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_expression_l98_9840


namespace NUMINAMATH_CALUDE_power_of_power_equals_729_l98_9896

theorem power_of_power_equals_729 : (3^3)^2 = 729 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_equals_729_l98_9896


namespace NUMINAMATH_CALUDE_melissa_bonus_points_l98_9823

/-- Given a player's regular points per game, number of games played, and total score,
    calculate the bonus points per game. -/
def bonusPointsPerGame (regularPointsPerGame : ℕ) (numGames : ℕ) (totalScore : ℕ) : ℕ :=
  ((totalScore - regularPointsPerGame * numGames) / numGames : ℕ)

/-- Theorem stating that for the given conditions, the bonus points per game is 82. -/
theorem melissa_bonus_points :
  bonusPointsPerGame 109 79 15089 = 82 := by
  sorry

#eval bonusPointsPerGame 109 79 15089

end NUMINAMATH_CALUDE_melissa_bonus_points_l98_9823


namespace NUMINAMATH_CALUDE_three_pair_prob_standard_deck_l98_9832

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (ranks : Nat)
  (suits : Nat)

/-- Represents a "three pair" hand in poker -/
structure ThreePair :=
  (triplet_rank : Nat)
  (triplet_suit : Nat)
  (pair_rank : Nat)
  (pair_suit : Nat)

/-- The number of ways to choose 5 cards from a deck -/
def choose_five (d : Deck) : Nat :=
  Nat.choose d.cards 5

/-- The number of valid "three pair" hands -/
def count_three_pairs (d : Deck) : Nat :=
  d.ranks * d.suits * (d.ranks - 1) * d.suits

/-- The probability of getting a "three pair" hand -/
def three_pair_probability (d : Deck) : ℚ :=
  count_three_pairs d / choose_five d

/-- Theorem: The probability of a "three pair" in a standard deck is 2,496 / 2,598,960 -/
theorem three_pair_prob_standard_deck :
  three_pair_probability (Deck.mk 52 13 4) = 2496 / 2598960 := by
  sorry

end NUMINAMATH_CALUDE_three_pair_prob_standard_deck_l98_9832


namespace NUMINAMATH_CALUDE_laptop_savings_weeks_l98_9816

theorem laptop_savings_weeks (laptop_cost : ℕ) (birthday_money : ℕ) (weekly_earnings : ℕ) 
  (h1 : laptop_cost = 800)
  (h2 : birthday_money = 140)
  (h3 : weekly_earnings = 20) :
  ∃ (weeks : ℕ), birthday_money + weekly_earnings * weeks = laptop_cost ∧ weeks = 33 := by
  sorry

end NUMINAMATH_CALUDE_laptop_savings_weeks_l98_9816
