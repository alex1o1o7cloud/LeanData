import Mathlib

namespace NUMINAMATH_CALUDE_middle_part_of_proportional_division_l180_18084

theorem middle_part_of_proportional_division (total : ℚ) (p1 p2 p3 : ℚ) :
  total = 96 →
  p1 + p2 + p3 = total →
  p2 = (1/4) * p1 →
  p3 = (1/8) * p1 →
  p2 = 17 + 21/44 :=
by sorry

end NUMINAMATH_CALUDE_middle_part_of_proportional_division_l180_18084


namespace NUMINAMATH_CALUDE_max_value_properties_l180_18065

noncomputable def f (s : ℝ) (x : ℝ) : ℝ := (Real.log s) / (1 + x) - Real.log s

theorem max_value_properties (s : ℝ) (x₀ : ℝ) 
  (h_max : ∀ x, f s x ≤ f s x₀) :
  f s x₀ = x₀ ∧ f s x₀ < (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_max_value_properties_l180_18065


namespace NUMINAMATH_CALUDE_sevenPeopleArrangementCount_l180_18089

/-- The number of ways to arrange n people in a row. -/
def arrangements (n : ℕ) : ℕ := n.factorial

/-- The number of ways to choose k items from n items. -/
def choose (n k : ℕ) : ℕ := (arrangements n) / ((arrangements k) * (arrangements (n - k)))

/-- The number of ways to arrange seven people in a row with two specific people not next to each other. -/
def sevenPeopleArrangement : ℕ :=
  (arrangements 5) * (choose 6 2)

theorem sevenPeopleArrangementCount :
  sevenPeopleArrangement = 3600 := by
  sorry

end NUMINAMATH_CALUDE_sevenPeopleArrangementCount_l180_18089


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l180_18061

/-- Given an arithmetic sequence 3, 7, 11, ..., x, y, 31, prove that x + y = 50 -/
theorem arithmetic_sequence_sum (x y : ℝ) : 
  (∃ (a : ℕ → ℝ), a 0 = 3 ∧ a 1 = 7 ∧ a 2 = 11 ∧ (∃ i j : ℕ, a i = x ∧ a (i + 1) = y ∧ a (j + 2) = 31) ∧ 
  (∀ n : ℕ, a (n + 1) - a n = a 1 - a 0)) → 
  x + y = 50 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l180_18061


namespace NUMINAMATH_CALUDE_subset_condition_l180_18023

def A : Set ℝ := {x | -2 ≤ x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x | x^2 - a*x - 4 ≤ 0}

theorem subset_condition (a : ℝ) : B a ⊆ A ↔ 0 ≤ a ∧ a < 3 := by sorry

end NUMINAMATH_CALUDE_subset_condition_l180_18023


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonal_l180_18033

/-- Given a rectangular prism with sides a, b, and c, if its surface area is 11
    and the sum of its edges is 24, then the length of its diagonal is 5. -/
theorem rectangular_prism_diagonal 
  (a b c : ℝ) 
  (h_surface : 2 * (a * b + a * c + b * c) = 11) 
  (h_edges : 4 * (a + b + c) = 24) : 
  Real.sqrt (a^2 + b^2 + c^2) = 5 := by
sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonal_l180_18033


namespace NUMINAMATH_CALUDE_university_visit_probability_l180_18099

theorem university_visit_probability : 
  let n : ℕ := 4  -- number of students
  let k : ℕ := 2  -- number of universities
  let p : ℚ := (k^n - 2) / k^n  -- probability formula
  p = 7/8 := by sorry

end NUMINAMATH_CALUDE_university_visit_probability_l180_18099


namespace NUMINAMATH_CALUDE_negative_1234_mod_9_l180_18056

theorem negative_1234_mod_9 : ∃! n : ℤ, 0 ≤ n ∧ n < 9 ∧ -1234 ≡ n [ZMOD 9] ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_negative_1234_mod_9_l180_18056


namespace NUMINAMATH_CALUDE_max_value_theorem_l180_18043

theorem max_value_theorem (x y : ℝ) (h : 2 * x^2 + x * y - y^2 = 1) :
  ∃ (M : ℝ), M = Real.sqrt 2 / 4 ∧ 
  ∀ (z : ℝ), z = (x - 2*y) / (5*x^2 - 2*x*y + 2*y^2) → z ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l180_18043


namespace NUMINAMATH_CALUDE_uncool_parents_count_l180_18037

theorem uncool_parents_count (total_students : ℕ) (cool_dads : ℕ) (cool_moms : ℕ) (both_cool : ℕ) 
  (h1 : total_students = 40)
  (h2 : cool_dads = 18)
  (h3 : cool_moms = 22)
  (h4 : both_cool = 10) :
  total_students - (cool_dads - both_cool + cool_moms - both_cool + both_cool) = 10 :=
by sorry

end NUMINAMATH_CALUDE_uncool_parents_count_l180_18037


namespace NUMINAMATH_CALUDE_marble_remainder_l180_18082

theorem marble_remainder (r p : ℕ) : 
  r % 8 = 5 → p % 8 = 6 → (r + p) % 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_marble_remainder_l180_18082


namespace NUMINAMATH_CALUDE_least_integer_absolute_value_l180_18022

theorem least_integer_absolute_value (x : ℤ) : 
  (∀ y : ℤ, y < x → |3 * y + 4| > 18) ∧ |3 * x + 4| ≤ 18 → x = -7 :=
sorry

end NUMINAMATH_CALUDE_least_integer_absolute_value_l180_18022


namespace NUMINAMATH_CALUDE_smallest_top_block_l180_18021

/-- Represents the pyramid structure -/
structure Pyramid :=
  (layer1 : Fin 15 → ℕ)
  (layer2 : Fin 10 → ℕ)
  (layer3 : Fin 6 → ℕ)
  (layer4 : ℕ)

/-- The rule for assigning numbers to upper blocks -/
def upper_block_rule (a b c : ℕ) : ℕ := 2 * (a + b + c)

/-- The pyramid satisfies the numbering rules -/
def valid_pyramid (p : Pyramid) : Prop :=
  (∀ i : Fin 15, p.layer1 i ∈ Finset.range 16) ∧
  (∀ i : Fin 10, ∃ a b c : Fin 15, p.layer2 i = upper_block_rule (p.layer1 a) (p.layer1 b) (p.layer1 c)) ∧
  (∀ i : Fin 6, ∃ a b c : Fin 10, p.layer3 i = upper_block_rule (p.layer2 a) (p.layer2 b) (p.layer2 c)) ∧
  (∃ a b c : Fin 6, p.layer4 = upper_block_rule (p.layer3 a) (p.layer3 b) (p.layer3 c))

/-- The theorem stating the smallest possible number for the top block -/
theorem smallest_top_block (p : Pyramid) (h : valid_pyramid p) : p.layer4 ≥ 48 := by
  sorry

end NUMINAMATH_CALUDE_smallest_top_block_l180_18021


namespace NUMINAMATH_CALUDE_trig_identity_proof_l180_18066

theorem trig_identity_proof : 
  1 / Real.cos (70 * π / 180) - Real.sqrt 3 / Real.sin (70 * π / 180) = 1 / (Real.cos (20 * π / 180))^2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l180_18066


namespace NUMINAMATH_CALUDE_arithmetic_sequence_part1_arithmetic_sequence_part2_l180_18032

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  d : ℤ
  seq_def : ∀ n : ℕ, a n = a 1 + (n - 1) * d

/-- Part 1 of the problem -/
theorem arithmetic_sequence_part1 (seq : ArithmeticSequence) 
  (h1 : seq.a 5 = -1) (h2 : seq.a 8 = 2) : 
  seq.a 1 = -5 ∧ seq.d = 1 := by
  sorry

/-- Part 2 of the problem -/
theorem arithmetic_sequence_part2 (seq : ArithmeticSequence) 
  (h1 : seq.a 1 + seq.a 6 = 12) (h2 : seq.a 4 = 7) :
  seq.a 9 = 17 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_part1_arithmetic_sequence_part2_l180_18032


namespace NUMINAMATH_CALUDE_staplers_remaining_l180_18070

/-- The number of staplers left after stapling reports -/
def staplers_left (initial_staplers : ℕ) (dozens_stapled : ℕ) : ℕ :=
  initial_staplers - dozens_stapled * 12

/-- Theorem: Given 50 initial staplers and 3 dozen reports stapled, 14 staplers are left -/
theorem staplers_remaining : staplers_left 50 3 = 14 := by
  sorry

end NUMINAMATH_CALUDE_staplers_remaining_l180_18070


namespace NUMINAMATH_CALUDE_total_nut_weight_l180_18018

/-- Represents the weight of nuts in kilograms -/
structure NutWeight where
  almonds : Float
  pecans : Float
  walnuts : Float
  cashews : Float
  pistachios : Float
  brazilNuts : Float
  macadamiaNuts : Float
  hazelnuts : Float

/-- Conversion rate from ounces to kilograms -/
def ounceToKgRate : Float := 0.0283495

/-- Weights of nuts bought by the chef -/
def chefNuts : NutWeight where
  almonds := 0.14
  pecans := 0.38
  walnuts := 0.22
  cashews := 0.47
  pistachios := 0.29
  brazilNuts := 6 * ounceToKgRate
  macadamiaNuts := 4.5 * ounceToKgRate
  hazelnuts := 7.3 * ounceToKgRate

/-- Theorem stating the total weight of nuts bought by the chef -/
theorem total_nut_weight : 
  chefNuts.almonds + chefNuts.pecans + chefNuts.walnuts + chefNuts.cashews + 
  chefNuts.pistachios + chefNuts.brazilNuts + chefNuts.macadamiaNuts + 
  chefNuts.hazelnuts = 2.1128216 := by
  sorry

end NUMINAMATH_CALUDE_total_nut_weight_l180_18018


namespace NUMINAMATH_CALUDE_billy_homework_questions_l180_18097

/-- Represents the number of questions solved in each hour -/
structure HourlyQuestions where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Theorem: Given the conditions, Billy solved 132 questions in the third hour -/
theorem billy_homework_questions (q : HourlyQuestions) : 
  q.third = 2 * q.second ∧ 
  q.third = 3 * q.first ∧ 
  q.first + q.second + q.third = 242 → 
  q.third = 132 := by
  sorry

end NUMINAMATH_CALUDE_billy_homework_questions_l180_18097


namespace NUMINAMATH_CALUDE_earth_inhabitable_fraction_l180_18017

theorem earth_inhabitable_fraction :
  let land_fraction : ℚ := 2/3
  let inhabitable_land_fraction : ℚ := 3/4
  (land_fraction * inhabitable_land_fraction : ℚ) = 1/2 := by sorry

end NUMINAMATH_CALUDE_earth_inhabitable_fraction_l180_18017


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_factorial_sum_l180_18000

theorem greatest_prime_factor_of_factorial_sum : 
  ∃ (p : ℕ), Nat.Prime p ∧ 
  p ∣ (Nat.factorial 15 + Nat.factorial 17) ∧ 
  ∀ (q : ℕ), Nat.Prime q → q ∣ (Nat.factorial 15 + Nat.factorial 17) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_factorial_sum_l180_18000


namespace NUMINAMATH_CALUDE_model_comparison_theorem_l180_18053

/-- A model for fitting data -/
structure Model where
  /-- The sum of squared residuals for this model -/
  sumSquaredResiduals : ℝ
  /-- Whether the residual points are uniformly distributed in a horizontal band -/
  uniformResiduals : Prop

/-- Compares the fitting effects of two models -/
def betterFit (m1 m2 : Model) : Prop :=
  m1.sumSquaredResiduals < m2.sumSquaredResiduals

/-- Indicates whether a model is appropriate based on its residual plot -/
def appropriateModel (m : Model) : Prop :=
  m.uniformResiduals

theorem model_comparison_theorem :
  ∀ (m1 m2 : Model),
    (betterFit m1 m2 → m1.sumSquaredResiduals < m2.sumSquaredResiduals) ∧
    (appropriateModel m1 ↔ m1.uniformResiduals) :=
by sorry

end NUMINAMATH_CALUDE_model_comparison_theorem_l180_18053


namespace NUMINAMATH_CALUDE_repeating_56_equals_fraction_l180_18030

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (a b : ℕ) : ℚ :=
  (10 * a + b : ℚ) / 99

/-- The repeating decimal 0.56̄ -/
def repeating_56 : ℚ := RepeatingDecimal 5 6

theorem repeating_56_equals_fraction : repeating_56 = 56 / 99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_56_equals_fraction_l180_18030


namespace NUMINAMATH_CALUDE_monday_loaves_l180_18024

/-- Represents the number of loaves baked on a given day -/
def loaves : Fin 6 → ℕ
  | 0 => 5  -- Wednesday
  | 1 => 7  -- Thursday
  | 2 => 10 -- Friday
  | 3 => 14 -- Saturday
  | 4 => 19 -- Sunday
  | 5 => 25 -- Monday (to be proven)

/-- The pattern of increase in loaves from one day to the next -/
def increase (n : Fin 5) : ℕ := loaves (n + 1) - loaves n

/-- The theorem stating that the number of loaves baked on Monday is 25 -/
theorem monday_loaves :
  (∀ n : Fin 4, increase (n + 1) = increase n + 1) →
  loaves 5 = 25 := by
  sorry


end NUMINAMATH_CALUDE_monday_loaves_l180_18024


namespace NUMINAMATH_CALUDE_cube_nested_square_root_l180_18038

theorem cube_nested_square_root : (Real.sqrt (2 + Real.sqrt (2 + Real.sqrt 2)))^3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_cube_nested_square_root_l180_18038


namespace NUMINAMATH_CALUDE_sum_bounds_l180_18013

theorem sum_bounds (a b c : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) 
  (h_eq : a^2 + b^2 + c^2 + a*b + 2/3*a*c + 4/3*b*c = 1) : 
  1 ≤ a + b + c ∧ a + b + c ≤ Real.sqrt 345 / 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_bounds_l180_18013


namespace NUMINAMATH_CALUDE_jane_calculation_l180_18049

theorem jane_calculation (x y z : ℝ) 
  (h1 : x - 2 * (y - 3 * z) = 25)
  (h2 : x - 2 * y - 3 * z = 7) :
  x - 2 * y = 13 := by
sorry

end NUMINAMATH_CALUDE_jane_calculation_l180_18049


namespace NUMINAMATH_CALUDE_work_left_fraction_l180_18079

theorem work_left_fraction (a_days b_days work_days : ℚ) 
  (ha : a_days = 20)
  (hb : b_days = 30)
  (hw : work_days = 4) : 
  1 - work_days * (1 / a_days + 1 / b_days) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_work_left_fraction_l180_18079


namespace NUMINAMATH_CALUDE_last_three_digits_of_2005_power_l180_18090

theorem last_three_digits_of_2005_power (A : ℕ) :
  A = 2005^2005 →
  A % 1000 = 125 := by
sorry

end NUMINAMATH_CALUDE_last_three_digits_of_2005_power_l180_18090


namespace NUMINAMATH_CALUDE_find_S_l180_18074

theorem find_S : ∃ S : ℚ, (1/4 : ℚ) * (1/6 : ℚ) * S = (1/5 : ℚ) * (1/8 : ℚ) * 160 ∧ S = 96 := by
  sorry

end NUMINAMATH_CALUDE_find_S_l180_18074


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_leg_length_l180_18047

/-- An isosceles trapezoid circumscribed around a circle with area S and acute base angle π/6 has leg length √(2S) -/
theorem isosceles_trapezoid_leg_length (S : ℝ) (h_pos : S > 0) :
  ∃ (x : ℝ),
    x > 0 ∧
    x = Real.sqrt (2 * S) ∧
    ∃ (a b h : ℝ),
      a > 0 ∧ b > 0 ∧ h > 0 ∧
      a + b = 2 * x ∧
      h = x * Real.sin (π / 6) ∧
      S = (a + b) * h / 2 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_leg_length_l180_18047


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l180_18085

theorem expression_simplification_and_evaluation :
  let a : ℤ := -1
  let b : ℤ := -2
  let original_expression := (2*a + b)*(b - 2*a) - (a - 3*b)^2
  let simplified_expression := -5*a^2 + 6*a*b - 8*b^2
  original_expression = simplified_expression ∧ simplified_expression = -25 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l180_18085


namespace NUMINAMATH_CALUDE_dice_sum_product_l180_18069

theorem dice_sum_product (a b c d : ℕ) : 
  1 ≤ a ∧ a ≤ 6 ∧
  1 ≤ b ∧ b ≤ 6 ∧
  1 ≤ c ∧ c ≤ 6 ∧
  1 ≤ d ∧ d ≤ 6 ∧
  a * b * c * d = 180 →
  a + b + c + d ≠ 19 := by
sorry

end NUMINAMATH_CALUDE_dice_sum_product_l180_18069


namespace NUMINAMATH_CALUDE_solution_difference_l180_18031

theorem solution_difference (x : ℝ) : 
  (∃ x₁ x₂ : ℝ, (x₁ ^ 3 = 7 - x₁^2 / 4) ∧ (x₂ ^ 3 = 7 - x₂^2 / 4) ∧ x₁ ≠ x₂) → 
  (∃ x₁ x₂ : ℝ, (x₁ ^ 3 = 7 - x₁^2 / 4) ∧ (x₂ ^ 3 = 7 - x₂^2 / 4) ∧ x₁ ≠ x₂ ∧ |x₁ - x₂| = 4 * Real.sqrt 6) :=
by
  sorry

end NUMINAMATH_CALUDE_solution_difference_l180_18031


namespace NUMINAMATH_CALUDE_scientific_notation_of_400000_l180_18020

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

/-- The number we want to represent in scientific notation -/
def number : ℝ := 400000

/-- The expected scientific notation representation -/
def expected : ScientificNotation :=
  { coefficient := 4
  , exponent := 5
  , is_valid := by sorry }

theorem scientific_notation_of_400000 :
  toScientificNotation number = expected := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_400000_l180_18020


namespace NUMINAMATH_CALUDE_tan_22_5_degree_decomposition_l180_18008

theorem tan_22_5_degree_decomposition :
  ∃ (a b c : ℕ+), 
    (a.val ≥ b.val ∧ b.val ≥ c.val) ∧
    (Real.tan (22.5 * π / 180) = Real.sqrt a.val - 1 + Real.sqrt b.val - Real.sqrt c.val) ∧
    (a.val + b.val + c.val = 12) := by sorry

end NUMINAMATH_CALUDE_tan_22_5_degree_decomposition_l180_18008


namespace NUMINAMATH_CALUDE_star_properties_l180_18088

/-- Custom multiplication operation for rational numbers -/
def star (x y : ℚ) : ℚ := x * y + 1

/-- Theorem stating the properties of the star operation -/
theorem star_properties :
  (star 2 3 = 7) ∧
  (star (star 1 4) (-1/2) = -3/2) ∧
  (∀ a b c : ℚ, star a (b + c) + 1 = star a b + star a c) := by
  sorry

end NUMINAMATH_CALUDE_star_properties_l180_18088


namespace NUMINAMATH_CALUDE_floor_ceil_sum_l180_18009

theorem floor_ceil_sum : ⌊(-3.01 : ℝ)⌋ + ⌈(24.99 : ℝ)⌉ = 21 := by sorry

end NUMINAMATH_CALUDE_floor_ceil_sum_l180_18009


namespace NUMINAMATH_CALUDE_plan_A_rate_correct_l180_18016

/-- The per-minute charge after the first 5 minutes under plan A -/
def plan_A_rate : ℝ := 0.06

/-- The fixed charge for the first 5 minutes under plan A -/
def plan_A_fixed_charge : ℝ := 0.60

/-- The per-minute charge under plan B -/
def plan_B_rate : ℝ := 0.08

/-- The duration at which both plans cost the same -/
def equal_cost_duration : ℝ := 14.999999999999996

theorem plan_A_rate_correct :
  plan_A_rate * (equal_cost_duration - 5) + plan_A_fixed_charge =
  plan_B_rate * equal_cost_duration := by sorry

end NUMINAMATH_CALUDE_plan_A_rate_correct_l180_18016


namespace NUMINAMATH_CALUDE_coefficient_x4_in_expansion_l180_18052

theorem coefficient_x4_in_expansion : 
  let expansion := (fun x => (1 - x)^5 * (2*x + 1))
  ∃ (a b c d e f : ℚ), 
    ∀ x, expansion x = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f ∧ b = -15 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x4_in_expansion_l180_18052


namespace NUMINAMATH_CALUDE_exists_good_number_in_interval_l180_18080

/-- A function that checks if a natural number is a "good number" (all digits ≤ 5) -/
def is_good_number (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d ≤ 5

/-- The main theorem: For any natural number x, there exists a "good number" y in [x, 9/5x) -/
theorem exists_good_number_in_interval (x : ℕ) : 
  ∃ y : ℕ, x ≤ y ∧ y < (9 * x) / 5 ∧ is_good_number y :=
sorry

end NUMINAMATH_CALUDE_exists_good_number_in_interval_l180_18080


namespace NUMINAMATH_CALUDE_line_point_distance_l180_18087

/-- Given a point M(a,b) on the line 4x - 3y + c = 0, 
    if the minimum value of (a-1)² + (b-1)² is 4, 
    then c = -11 or c = 9 -/
theorem line_point_distance (a b c : ℝ) : 
  (4 * a - 3 * b + c = 0) → 
  (∃ (d : ℝ), d = 4 ∧ ∀ (x y : ℝ), 4 * x - 3 * y + c = 0 → (x - 1)^2 + (y - 1)^2 ≥ d) →
  (c = -11 ∨ c = 9) :=
sorry

end NUMINAMATH_CALUDE_line_point_distance_l180_18087


namespace NUMINAMATH_CALUDE_payment_per_task_l180_18034

/-- Calculates the payment per task for a contractor given their work schedule and total earnings -/
theorem payment_per_task (hours_per_task : ℝ) (hours_per_day : ℝ) (days_per_week : ℕ) (total_earnings : ℝ) :
  hours_per_task = 2 →
  hours_per_day = 10 →
  days_per_week = 5 →
  total_earnings = 1400 →
  (total_earnings / (days_per_week * (hours_per_day / hours_per_task))) = 56 := by
sorry

end NUMINAMATH_CALUDE_payment_per_task_l180_18034


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l180_18028

-- Define the sets A and B
def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {x | 3 - 2*x > 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | x < 3/2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l180_18028


namespace NUMINAMATH_CALUDE_scooter_gain_percent_l180_18035

/-- Calculate the gain percent on a scooter sale -/
theorem scooter_gain_percent
  (purchase_price : ℝ)
  (repair_costs : ℝ)
  (selling_price : ℝ)
  (h1 : purchase_price = 900)
  (h2 : repair_costs = 300)
  (h3 : selling_price = 1320) :
  (selling_price - (purchase_price + repair_costs)) / (purchase_price + repair_costs) * 100 = 10 :=
by sorry

end NUMINAMATH_CALUDE_scooter_gain_percent_l180_18035


namespace NUMINAMATH_CALUDE_cubic_root_product_l180_18001

theorem cubic_root_product (a b c : ℝ) : 
  (a^3 - 15*a^2 + 22*a - 8 = 0) ∧ 
  (b^3 - 15*b^2 + 22*b - 8 = 0) ∧ 
  (c^3 - 15*c^2 + 22*c - 8 = 0) → 
  (2+a)*(2+b)*(2+c) = 120 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_product_l180_18001


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l180_18055

/-- Given a quadratic inequality ax^2 + (a-1)x - 1 > 0 with solution set (-1, -1/2), prove that a = -2 -/
theorem quadratic_inequality_solution (a : ℝ) : 
  (∀ x : ℝ, ax^2 + (a-1)*x - 1 > 0 ↔ -1 < x ∧ x < -1/2) → 
  a = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l180_18055


namespace NUMINAMATH_CALUDE_composite_division_l180_18077

def first_four_composites : List Nat := [4, 6, 8, 9]
def next_four_composites : List Nat := [10, 12, 14, 15]

theorem composite_division :
  (first_four_composites.prod : ℚ) / (next_four_composites.prod : ℚ) = 12 / 175 := by
  sorry

end NUMINAMATH_CALUDE_composite_division_l180_18077


namespace NUMINAMATH_CALUDE_harry_book_count_l180_18051

/-- The number of books Harry has -/
def harry_books : ℕ := 50

/-- The number of books Flora has -/
def flora_books : ℕ := 2 * harry_books

/-- The number of books Gary has -/
def gary_books : ℕ := harry_books / 2

/-- The total number of books -/
def total_books : ℕ := 175

theorem harry_book_count : 
  harry_books + flora_books + gary_books = total_books ∧ harry_books = 50 := by
  sorry

end NUMINAMATH_CALUDE_harry_book_count_l180_18051


namespace NUMINAMATH_CALUDE_dog_tricks_conversion_l180_18067

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

theorem dog_tricks_conversion :
  base7ToBase10 [3, 5, 6] = 332 := by
  sorry

end NUMINAMATH_CALUDE_dog_tricks_conversion_l180_18067


namespace NUMINAMATH_CALUDE_exists_natural_sqrt_nested_root_l180_18094

theorem exists_natural_sqrt_nested_root : ∃ n : ℕ, n > 1 ∧ ∃ m : ℕ, (n : ℝ)^(7/8) = m := by
  sorry

end NUMINAMATH_CALUDE_exists_natural_sqrt_nested_root_l180_18094


namespace NUMINAMATH_CALUDE_probability_of_winning_pair_l180_18086

/-- A card in the deck -/
structure Card where
  color : Bool  -- True for red, False for green
  label : Fin 5 -- Labels A, B, C, D, E represented as 0, 1, 2, 3, 4

/-- The deck of cards -/
def deck : Finset Card := sorry

/-- A pair of cards is winning if they have the same color or the same label -/
def is_winning_pair (c1 c2 : Card) : Bool :=
  c1.color = c2.color ∨ c1.label = c2.label

/-- The set of all possible pairs of cards -/
def all_pairs : Finset (Card × Card) := sorry

/-- The set of winning pairs -/
def winning_pairs : Finset (Card × Card) := sorry

/-- The probability of drawing a winning pair -/
theorem probability_of_winning_pair :
  (winning_pairs.card : ℚ) / all_pairs.card = 35 / 66 := by sorry

end NUMINAMATH_CALUDE_probability_of_winning_pair_l180_18086


namespace NUMINAMATH_CALUDE_sqrt_180_simplification_l180_18050

theorem sqrt_180_simplification : Real.sqrt 180 = 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_180_simplification_l180_18050


namespace NUMINAMATH_CALUDE_set_intersection_union_l180_18040

theorem set_intersection_union (M N P : Set ℕ) : 
  M = {1} → N = {1, 2} → P = {1, 2, 3} → (M ∪ N) ∩ P = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_union_l180_18040


namespace NUMINAMATH_CALUDE_alternative_increase_is_nineteen_cents_l180_18003

/-- Represents the fine structure for overdue books in a library --/
structure OverdueFine where
  initial_fine : ℚ
  standard_increase : ℚ
  alternative_increase : ℚ
  fifth_day_fine : ℚ

/-- Calculates the fine for a given number of days overdue --/
def calculate_fine (f : OverdueFine) (days : ℕ) : ℚ :=
  f.initial_fine + (days - 1) * min f.standard_increase f.alternative_increase

/-- Theorem stating that the alternative increase is $0.19 --/
theorem alternative_increase_is_nineteen_cents 
  (f : OverdueFine) 
  (h1 : f.initial_fine = 7/100)
  (h2 : f.standard_increase = 30/100)
  (h3 : f.fifth_day_fine = 86/100) : 
  f.alternative_increase = 19/100 := by
  sorry

#eval let f : OverdueFine := {
  initial_fine := 7/100,
  standard_increase := 30/100,
  alternative_increase := 19/100,
  fifth_day_fine := 86/100
}
calculate_fine f 5

end NUMINAMATH_CALUDE_alternative_increase_is_nineteen_cents_l180_18003


namespace NUMINAMATH_CALUDE_inscribed_hemisphere_volume_l180_18058

/-- Given a cone with height 4 cm and slant height 5 cm, the volume of an inscribed hemisphere
    whose base lies on the base of the cone is (1152/125)π cm³. -/
theorem inscribed_hemisphere_volume (h : ℝ) (l : ℝ) (r : ℝ) :
  h = 4 →
  l = 5 →
  l^2 = h^2 + r^2 →
  (∃ x, x > 0 ∧ x < h ∧ r^2 + (l - x)^2 = h^2 ∧ x^2 + r^2 = r^2) →
  (2/3) * π * ((12/5)^3) = (1152/125) * π :=
by sorry

end NUMINAMATH_CALUDE_inscribed_hemisphere_volume_l180_18058


namespace NUMINAMATH_CALUDE_percentage_division_problem_l180_18012

theorem percentage_division_problem : (168 / 100 * 1265) / 6 = 354.2 := by
  sorry

end NUMINAMATH_CALUDE_percentage_division_problem_l180_18012


namespace NUMINAMATH_CALUDE_vanessa_score_l180_18048

theorem vanessa_score (team_score : ℕ) (other_players : ℕ) (other_avg : ℚ) : 
  team_score = 48 → 
  other_players = 6 → 
  other_avg = 3.5 → 
  team_score - (other_players : ℚ) * other_avg = 27 := by
  sorry

end NUMINAMATH_CALUDE_vanessa_score_l180_18048


namespace NUMINAMATH_CALUDE_locus_is_hyperbola_l180_18054

/-- Given the coordinates of point P(x, y) satisfying the following conditions,
    prove that the locus of P is a hyperbola. -/
theorem locus_is_hyperbola
  (a c : ℝ)
  (x y θ₁ θ₂ : ℝ)
  (h1 : (x - a) * Real.cos θ₁ + y * Real.sin θ₁ = a)
  (h2 : (x - a) * Real.cos θ₂ + y * Real.sin θ₂ = a)
  (h3 : Real.tan (θ₁ / 2) - Real.tan (θ₂ / 2) = 2 * c)
  (h4 : c > 1) :
  ∃ (k m n : ℝ), y^2 = k * x^2 + m * x + n ∧ k > 1 :=
sorry

end NUMINAMATH_CALUDE_locus_is_hyperbola_l180_18054


namespace NUMINAMATH_CALUDE_smallest_sum_of_four_odds_divisible_by_five_l180_18081

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

def consecutive_odds (a b c d : ℕ) : Prop :=
  is_odd a ∧ is_odd b ∧ is_odd c ∧ is_odd d ∧
  b = a + 2 ∧ c = b + 2 ∧ d = c + 2

def not_divisible_by_three (n : ℕ) : Prop := n % 3 ≠ 0

theorem smallest_sum_of_four_odds_divisible_by_five :
  ∃ a b c d : ℕ,
    consecutive_odds a b c d ∧
    not_divisible_by_three a ∧
    not_divisible_by_three b ∧
    not_divisible_by_three c ∧
    not_divisible_by_three d ∧
    (a + b + c + d) % 5 = 0 ∧
    a + b + c + d = 40 ∧
    (∀ w x y z : ℕ,
      consecutive_odds w x y z →
      not_divisible_by_three w →
      not_divisible_by_three x →
      not_divisible_by_three y →
      not_divisible_by_three z →
      (w + x + y + z) % 5 = 0 →
      w + x + y + z ≥ 40) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_four_odds_divisible_by_five_l180_18081


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l180_18093

theorem reciprocal_of_negative_2023 :
  ∃ x : ℚ, x * (-2023) = 1 ∧ x = -1/2023 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l180_18093


namespace NUMINAMATH_CALUDE_largest_s_value_largest_s_value_is_121_l180_18046

/-- The largest possible value of s for regular polygons Q1 (r-gon) and Q2 (s-gon) 
    satisfying the given conditions -/
theorem largest_s_value : ℕ :=
  let r : ℕ → ℕ := fun s => 120 * s / (122 - s)
  let interior_angle : ℕ → ℚ := fun n => (n - 2 : ℚ) * 180 / n
  let s_max := 121
  have h1 : ∀ s : ℕ, s ≥ 3 → s ≤ s_max → 
    (interior_angle (r s)) / (interior_angle s) = 61 / 60 := by sorry
  have h2 : ∀ s : ℕ, s > s_max → ¬(∃ r : ℕ, r ≥ s ∧ 
    (interior_angle r) / (interior_angle s) = 61 / 60) := by sorry
  s_max

/-- Proof that the largest possible value of s is indeed 121 -/
theorem largest_s_value_is_121 : largest_s_value = 121 := by sorry

end NUMINAMATH_CALUDE_largest_s_value_largest_s_value_is_121_l180_18046


namespace NUMINAMATH_CALUDE_unfilled_holes_l180_18026

theorem unfilled_holes (total : ℕ) (filled_percentage : ℚ) : 
  total = 8 → filled_percentage = 75 / 100 → total - (filled_percentage * total).floor = 2 := by
sorry

end NUMINAMATH_CALUDE_unfilled_holes_l180_18026


namespace NUMINAMATH_CALUDE_handshakes_in_specific_tournament_l180_18068

/-- Represents a tennis tournament with teams of women -/
structure WomensTennisTournament where
  total_teams : Nat
  women_per_team : Nat
  participating_teams : Nat

/-- Calculates the number of handshakes in the tournament -/
def calculate_handshakes (tournament : WomensTennisTournament) : Nat :=
  let total_women := tournament.participating_teams * tournament.women_per_team
  let handshakes_per_woman := (tournament.participating_teams - 1) * tournament.women_per_team
  (total_women * handshakes_per_woman) / 2

/-- Theorem stating the number of handshakes in the specific tournament scenario -/
theorem handshakes_in_specific_tournament :
  let tournament := WomensTennisTournament.mk 4 2 3
  calculate_handshakes tournament = 12 := by
  sorry

end NUMINAMATH_CALUDE_handshakes_in_specific_tournament_l180_18068


namespace NUMINAMATH_CALUDE_sarahs_age_l180_18062

theorem sarahs_age :
  ∀ (s : ℚ), -- Sarah's age
  (∃ (g : ℚ), -- Grandmother's age
    g = 10 * s ∧ -- Grandmother is ten times Sarah's age
    g - s = 60) -- Grandmother was 60 when Sarah was born
  → s = 20 / 3 := by
sorry

end NUMINAMATH_CALUDE_sarahs_age_l180_18062


namespace NUMINAMATH_CALUDE_roots_sum_theorem_l180_18045

theorem roots_sum_theorem (p q : ℝ) : 
  p^2 - 5*p + 6 = 0 → q^2 - 5*q + 6 = 0 → p^3 + p^5*q + p*q^5 + q^3 = 617 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_theorem_l180_18045


namespace NUMINAMATH_CALUDE_required_speed_for_average_l180_18002

/-- Proves the required speed for the last part of a journey to achieve a desired average speed --/
theorem required_speed_for_average 
  (total_time : ℝ) 
  (initial_time : ℝ) 
  (initial_speed : ℝ) 
  (desired_avg_speed : ℝ) 
  (h1 : total_time = 5) 
  (h2 : initial_time = 3) 
  (h3 : initial_speed = 60) 
  (h4 : desired_avg_speed = 70) : 
  (desired_avg_speed * total_time - initial_speed * initial_time) / (total_time - initial_time) = 85 := by
  sorry

#check required_speed_for_average

end NUMINAMATH_CALUDE_required_speed_for_average_l180_18002


namespace NUMINAMATH_CALUDE_kay_age_is_32_l180_18078

-- Define Kay's age and the number of siblings
def kay_age : ℕ := sorry
def num_siblings : ℕ := 14

-- Define the ages of the youngest and oldest siblings
def youngest_sibling_age : ℕ := kay_age / 2 - 5
def oldest_sibling_age : ℕ := 4 * youngest_sibling_age

-- State the theorem
theorem kay_age_is_32 :
  num_siblings = 14 ∧
  youngest_sibling_age = kay_age / 2 - 5 ∧
  oldest_sibling_age = 4 * youngest_sibling_age ∧
  oldest_sibling_age = 44 →
  kay_age = 32 :=
by sorry

end NUMINAMATH_CALUDE_kay_age_is_32_l180_18078


namespace NUMINAMATH_CALUDE_hcf_problem_l180_18075

theorem hcf_problem (a b : ℕ+) (h1 : a * b = 45276) (h2 : Nat.lcm a b = 2058) :
  Nat.gcd a b = 22 := by sorry

end NUMINAMATH_CALUDE_hcf_problem_l180_18075


namespace NUMINAMATH_CALUDE_divisibility_by_eight_l180_18014

theorem divisibility_by_eight (a b c d : ℤ) :
  8 ∣ (1000 * a + 100 * b + 10 * c + d) ↔ 8 ∣ (4 * b + 2 * c + d) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_eight_l180_18014


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l180_18036

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by sorry

theorem negation_of_quadratic_equation :
  (¬ ∃ x : ℝ, x^2 - x - 1 = 0) ↔ (∀ x : ℝ, x^2 - x - 1 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l180_18036


namespace NUMINAMATH_CALUDE_truck_wheels_l180_18060

/-- Toll calculation function -/
def toll (x : ℕ) : ℚ :=
  0.5 + 0.5 * (x - 2)

/-- Number of wheels on the front axle -/
def frontWheels : ℕ := 2

/-- Number of wheels on each non-front axle -/
def otherWheels : ℕ := 4

/-- Theorem stating the total number of wheels on the truck -/
theorem truck_wheels (x : ℕ) (h1 : toll x = 2) (h2 : x > 0) : 
  frontWheels + (x - 1) * otherWheels = 18 := by
  sorry

#check truck_wheels

end NUMINAMATH_CALUDE_truck_wheels_l180_18060


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_squared_divisible_by_72_l180_18025

theorem largest_divisor_of_n_squared_divisible_by_72 (n : ℕ) (hn : n > 0) (h_div : 72 ∣ n^2) :
  ∃ m : ℕ, m = 12 ∧ m ∣ n ∧ ∀ k : ℕ, k ∣ n → k ≤ m :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_squared_divisible_by_72_l180_18025


namespace NUMINAMATH_CALUDE_floor_sqrt_72_l180_18071

theorem floor_sqrt_72 : ⌊Real.sqrt 72⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_72_l180_18071


namespace NUMINAMATH_CALUDE_equal_number_of_boys_and_girls_l180_18073

theorem equal_number_of_boys_and_girls 
  (m : ℕ) (d : ℕ) (M : ℝ) (D : ℝ) 
  (h1 : M / m ≠ D / d) 
  (h2 : (M / m + D / d) / 2 = (M + D) / (m + d)) : 
  m = d := by sorry

end NUMINAMATH_CALUDE_equal_number_of_boys_and_girls_l180_18073


namespace NUMINAMATH_CALUDE_truck_count_l180_18095

theorem truck_count (tanks trucks : ℕ) : 
  tanks = 5 * trucks →
  tanks + trucks = 140 →
  trucks = 23 := by
sorry

end NUMINAMATH_CALUDE_truck_count_l180_18095


namespace NUMINAMATH_CALUDE_unique_determination_of_polynomial_minimality_of_points_l180_18076

/-- A polynomial of degree 2017 with integer coefficients and leading coefficient 1 -/
def IntPolynomial2017 : Type := 
  {p : Polynomial ℤ // p.degree = 2017 ∧ p.leadingCoeff = 1}

/-- The minimum number of points needed to uniquely determine the polynomial -/
def minPointsForUniqueness : ℕ := 2017

theorem unique_determination_of_polynomial (p q : IntPolynomial2017) 
  (points : Fin minPointsForUniqueness → ℤ) :
  (∀ i : Fin minPointsForUniqueness, p.val.eval (points i) = q.val.eval (points i)) →
  p = q :=
sorry

theorem minimality_of_points :
  ∀ k : ℕ, k < minPointsForUniqueness →
  ∃ (p q : IntPolynomial2017) (points : Fin k → ℤ),
    (∀ i : Fin k, p.val.eval (points i) = q.val.eval (points i)) ∧
    p ≠ q :=
sorry

end NUMINAMATH_CALUDE_unique_determination_of_polynomial_minimality_of_points_l180_18076


namespace NUMINAMATH_CALUDE_smallest_interesting_number_l180_18098

/-- A natural number is interesting if 2n is a perfect square and 15n is a perfect cube. -/
def is_interesting (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 2 * n = a ^ 2 ∧ 15 * n = b ^ 3

/-- The smallest interesting number is 1800. -/
theorem smallest_interesting_number : 
  (is_interesting 1800 ∧ ∀ m < 1800, ¬ is_interesting m) :=
sorry

end NUMINAMATH_CALUDE_smallest_interesting_number_l180_18098


namespace NUMINAMATH_CALUDE_complex_number_problem_l180_18005

theorem complex_number_problem (a : ℝ) (z : ℂ) (h1 : z = a + I) 
  (h2 : (Complex.I * 2 + 1) * z ∈ {w : ℂ | w.re = 0 ∧ w.im ≠ 0}) :
  z = 2 + I ∧ Complex.abs (z / (2 - I)) = 1 := by sorry

end NUMINAMATH_CALUDE_complex_number_problem_l180_18005


namespace NUMINAMATH_CALUDE_n_has_21_digits_l180_18064

/-- The smallest positive integer satisfying the given conditions -/
def n : ℕ := sorry

/-- n is divisible by 30 -/
axiom n_div_30 : 30 ∣ n

/-- n^2 is a perfect fourth power -/
axiom n_sq_fourth_power : ∃ k : ℕ, n^2 = k^4

/-- n^3 is a perfect fifth power -/
axiom n_cube_fifth_power : ∃ k : ℕ, n^3 = k^5

/-- n is the smallest positive integer satisfying the conditions -/
axiom n_smallest : ∀ m : ℕ, m > 0 → (30 ∣ m) → (∃ k : ℕ, m^2 = k^4) → (∃ k : ℕ, m^3 = k^5) → n ≤ m

/-- The number of digits in a natural number -/
def num_digits (x : ℕ) : ℕ := sorry

/-- The main theorem: n has 21 digits -/
theorem n_has_21_digits : num_digits n = 21 := by sorry

end NUMINAMATH_CALUDE_n_has_21_digits_l180_18064


namespace NUMINAMATH_CALUDE_sqrt_division_equality_l180_18039

theorem sqrt_division_equality : Real.sqrt 2 / Real.sqrt 3 = Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_division_equality_l180_18039


namespace NUMINAMATH_CALUDE_min_value_a_l180_18029

theorem min_value_a (a : ℝ) : 
  (∀ x ∈ Set.Ioo 0 (1/2), x^2 + a*x + 1 ≥ 0) → a ≥ -5/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_a_l180_18029


namespace NUMINAMATH_CALUDE_min_lateral_surface_area_cone_l180_18019

/-- Given a cone with volume 4π/3, its minimum lateral surface area is 2√3π. -/
theorem min_lateral_surface_area_cone (r h : ℝ) (h_volume : (1/3) * π * r^2 * h = (4/3) * π) :
  ∃ (S_min : ℝ), S_min = 2 * Real.sqrt 3 * π ∧ 
  ∀ (S : ℝ), S = π * r * Real.sqrt (r^2 + h^2) → S ≥ S_min :=
sorry

end NUMINAMATH_CALUDE_min_lateral_surface_area_cone_l180_18019


namespace NUMINAMATH_CALUDE_flu_transmission_rate_l180_18063

/-- 
Given two rounds of flu transmission where a total of 121 people are infected,
prove that on average, one person infects 10 others in each round.
-/
theorem flu_transmission_rate : 
  ∃ x : ℕ, 
    (1 + x + x * (1 + x) = 121) ∧ 
    (x = 10) := by
  sorry

end NUMINAMATH_CALUDE_flu_transmission_rate_l180_18063


namespace NUMINAMATH_CALUDE_pig_farm_area_l180_18015

/-- Represents a rectangular pig farm with specific properties -/
structure PigFarm where
  short_side : ℝ
  long_side : ℝ
  fence_length : ℝ
  area : ℝ

/-- Creates a PigFarm given the length of the shorter side -/
def make_pig_farm (x : ℝ) : PigFarm :=
  { short_side := x
  , long_side := 2 * x
  , fence_length := 4 * x
  , area := 2 * x * x
  }

/-- Theorem stating the area of the pig farm with given conditions -/
theorem pig_farm_area :
  ∃ (farm : PigFarm), farm.fence_length = 150 ∧ farm.area = 2812.5 := by
  sorry


end NUMINAMATH_CALUDE_pig_farm_area_l180_18015


namespace NUMINAMATH_CALUDE_sequence_property_l180_18091

theorem sequence_property (a : ℕ → ℝ) 
  (h : ∀ m : ℕ, m > 1 → a (m + 1) * a (m - 1) = a m ^ 2 - a 1 ^ 2) :
  ∀ m n : ℕ, m > n ∧ n > 1 → a (m + n) * a (m - n) = a m ^ 2 - a n ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_sequence_property_l180_18091


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_on_circle_l180_18092

theorem sum_of_x_and_y_on_circle (x y : ℝ) (h : x^2 + y^2 = 16*x - 8*y - 60) : x + y = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_on_circle_l180_18092


namespace NUMINAMATH_CALUDE_rat_count_proof_l180_18011

def total_rats (kenia_rats : ℕ) (hunter_rats : ℕ) (elodie_rats : ℕ) : ℕ :=
  kenia_rats + hunter_rats + elodie_rats

theorem rat_count_proof (kenia_rats : ℕ) (hunter_rats : ℕ) (elodie_rats : ℕ) :
  kenia_rats = 3 * (hunter_rats + elodie_rats) →
  elodie_rats = 30 →
  elodie_rats = hunter_rats + 10 →
  total_rats kenia_rats hunter_rats elodie_rats = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_rat_count_proof_l180_18011


namespace NUMINAMATH_CALUDE_function_equality_condition_l180_18072

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 2*x
def g (a x : ℝ) : ℝ := a*x - 1

-- Define the domain interval
def I : Set ℝ := Set.Icc (-1) 2

-- Define the theorem
theorem function_equality_condition (a : ℝ) : 
  (∀ x₁ ∈ I, ∃ x₂ ∈ I, f x₁ = g a x₂) ↔ 
  (a ≤ -4 ∨ a ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_function_equality_condition_l180_18072


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l180_18083

theorem quadratic_equation_roots (p q : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁^2 + p*x₁ + q = 0 ∧ x₂^2 + p*x₂ + q = 0 ∧ 
   x₁ - x₂ = 5 ∧ x₁^3 - x₂^3 = 35) →
  ((p = 1 ∧ q = -6) ∨ (p = -1 ∧ q = -6)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l180_18083


namespace NUMINAMATH_CALUDE_expression_evaluation_l180_18027

theorem expression_evaluation : 
  0.064^(-1/3) - (-7/9)^0 + ((-2)^3)^(1/3) - 16^(-0.75) = -5/8 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l180_18027


namespace NUMINAMATH_CALUDE_integer_fraction_characterization_l180_18042

theorem integer_fraction_characterization (a b : ℕ+) :
  (∃ k : ℕ+, (a.val ^ 2 : ℚ) / (2 * a.val * b.val ^ 2 - b.val ^ 3 + 1) = k.val) ↔
  (∃ l : ℕ+, (a = 2 * l ∧ b = 1) ∨ 
             (a = l ∧ b = 2 * l) ∨ 
             (a = 8 * l.val ^ 4 - l ∧ b = 2 * l)) :=
sorry

end NUMINAMATH_CALUDE_integer_fraction_characterization_l180_18042


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l180_18004

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 0}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -2 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l180_18004


namespace NUMINAMATH_CALUDE_parabola_point_D_l180_18010

/-- A parabola passing through three given points -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  point_A : a * (-0.8)^2 + b * (-0.8) + c = 4.132
  point_B : a * 1.2^2 + b * 1.2 + c = -1.948
  point_C : a * 2.8^2 + b * 2.8 + c = -3.932

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def y_coordinate (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem parabola_point_D (p : Parabola) : y_coordinate p 1.8 = -2.992 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_D_l180_18010


namespace NUMINAMATH_CALUDE_food_distribution_l180_18059

theorem food_distribution (initial_men : ℕ) (initial_days : ℕ) (additional_men : ℕ) (days_before_increase : ℕ) : 
  initial_men = 760 →
  initial_days = 22 →
  additional_men = 40 →
  days_before_increase = 2 →
  (initial_men * initial_days - initial_men * days_before_increase) / (initial_men + additional_men) = 19 :=
by sorry

end NUMINAMATH_CALUDE_food_distribution_l180_18059


namespace NUMINAMATH_CALUDE_complex_power_2006_l180_18057

def i : ℂ := Complex.I

theorem complex_power_2006 : ((1 + i) / (1 - i)) ^ 2006 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_2006_l180_18057


namespace NUMINAMATH_CALUDE_point_not_in_region_l180_18007

def planar_region (x y : ℝ) : Prop := 2 * x + 3 * y < 6

theorem point_not_in_region :
  ¬ (planar_region 0 2) ∧
  (planar_region 0 0) ∧
  (planar_region 1 1) ∧
  (planar_region 2 0) :=
by sorry

end NUMINAMATH_CALUDE_point_not_in_region_l180_18007


namespace NUMINAMATH_CALUDE_butterfly_development_time_l180_18041

/-- The time (in days) a butterfly spends in a cocoon -/
def cocoon_time : ℕ := 30

/-- The time (in days) a butterfly spends as a larva -/
def larva_time : ℕ := 3 * cocoon_time

/-- The total time (in days) from butterfly egg to butterfly -/
def total_time : ℕ := larva_time + cocoon_time

theorem butterfly_development_time : total_time = 120 := by
  sorry

end NUMINAMATH_CALUDE_butterfly_development_time_l180_18041


namespace NUMINAMATH_CALUDE_coke_to_sprite_ratio_l180_18006

/-- Represents the ratio of ingredients in a drink -/
structure DrinkRatio where
  coke : ℚ
  sprite : ℚ
  mountainDew : ℚ

/-- Represents the composition of a drink -/
structure Drink where
  ratio : DrinkRatio
  cokeAmount : ℚ
  totalAmount : ℚ

/-- Theorem: Given a drink with the specified ratio and amounts, prove that the ratio of Coke to Sprite is 2:1 -/
theorem coke_to_sprite_ratio 
  (drink : Drink) 
  (h1 : drink.ratio.sprite = 1)
  (h2 : drink.ratio.mountainDew = 3)
  (h3 : drink.cokeAmount = 6)
  (h4 : drink.totalAmount = 18) :
  drink.ratio.coke / drink.ratio.sprite = 2 := by
  sorry


end NUMINAMATH_CALUDE_coke_to_sprite_ratio_l180_18006


namespace NUMINAMATH_CALUDE_mermaid_seashell_age_l180_18096

/-- Converts a base-9 number to base-10 --/
def base9_to_base10 (hundreds : Nat) (tens : Nat) (ones : Nat) : Nat :=
  hundreds * 9^2 + tens * 9^1 + ones * 9^0

/-- The mermaid's seashell collection age conversion theorem --/
theorem mermaid_seashell_age :
  base9_to_base10 3 6 2 = 299 := by
  sorry

end NUMINAMATH_CALUDE_mermaid_seashell_age_l180_18096


namespace NUMINAMATH_CALUDE_joan_seashells_l180_18044

theorem joan_seashells (sam_shells : ℕ) (total_shells : ℕ) (joan_shells : ℕ) 
  (h1 : sam_shells = 35)
  (h2 : total_shells = 53)
  (h3 : total_shells = sam_shells + joan_shells) :
  joan_shells = 18 := by
  sorry

end NUMINAMATH_CALUDE_joan_seashells_l180_18044
