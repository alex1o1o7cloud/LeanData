import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_calculation_l620_62085

theorem arithmetic_calculation : 8 - 7 + 6 * 5 + 4 - 3 * 2 + 1 - 0 = 30 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l620_62085


namespace NUMINAMATH_CALUDE_prob_different_cinemas_value_l620_62018

/-- The number of cinemas in the city -/
def num_cinemas : ℕ := 10

/-- The number of boys going to the cinema -/
def num_boys : ℕ := 7

/-- The probability of 7 boys choosing different cinemas out of 10 cinemas -/
def prob_different_cinemas : ℚ :=
  (num_cinemas.factorial / (num_cinemas - num_boys).factorial) / num_cinemas ^ num_boys

theorem prob_different_cinemas_value : 
  prob_different_cinemas = 15120 / 250000 :=
sorry

end NUMINAMATH_CALUDE_prob_different_cinemas_value_l620_62018


namespace NUMINAMATH_CALUDE_factorization_equality_l620_62065

theorem factorization_equality (a : ℝ) : 2*a^2 + 4*a + 2 = 2*(a+1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l620_62065


namespace NUMINAMATH_CALUDE_bandit_gem_distribution_theorem_l620_62058

/-- Represents the distribution of precious stones for a bandit -/
structure GemDistribution where
  rubies : ℕ
  sapphires : ℕ
  emeralds : ℕ
  sum_is_100 : rubies + sapphires + emeralds = 100

/-- The proposition to be proven -/
theorem bandit_gem_distribution_theorem (bandits : Finset GemDistribution) 
    (h : bandits.card = 102) :
  (∃ b1 b2 : GemDistribution, b1 ∈ bandits ∧ b2 ∈ bandits ∧ b1 ≠ b2 ∧
    b1.rubies = b2.rubies ∧ b1.sapphires = b2.sapphires ∧ b1.emeralds = b2.emeralds) ∨
  (∃ b1 b2 : GemDistribution, b1 ∈ bandits ∧ b2 ∈ bandits ∧ b1 ≠ b2 ∧
    b1.rubies ≠ b2.rubies ∧ b1.sapphires ≠ b2.sapphires ∧ b1.emeralds ≠ b2.emeralds) :=
by
  sorry

end NUMINAMATH_CALUDE_bandit_gem_distribution_theorem_l620_62058


namespace NUMINAMATH_CALUDE_perpendicular_line_through_circle_l620_62092

/-- Given a circle C and a line l in polar coordinates, 
    this theorem proves the equation of a line passing through C 
    and perpendicular to l. -/
theorem perpendicular_line_through_circle 
  (C : ℝ → ℝ) 
  (l : ℝ → ℝ → ℝ) 
  (h_C : ∀ θ, C θ = 2 * Real.cos θ) 
  (h_l : ∀ ρ θ, l ρ θ = ρ * Real.cos θ - ρ * Real.sin θ - 4) :
  ∃ f : ℝ → ℝ → ℝ, 
    (∀ ρ θ, f ρ θ = ρ * (Real.cos θ + Real.sin θ) - 1) ∧
    (∃ θ₀, C θ₀ = f (C θ₀) θ₀) ∧
    (∀ ρ₁ θ₁ ρ₂ θ₂, 
      l ρ₁ θ₁ = 0 → l ρ₂ θ₂ = 0 → f ρ₁ θ₁ = 0 → f ρ₂ θ₂ = 0 →
      (ρ₁ * Real.cos θ₁ - ρ₂ * Real.cos θ₂) * (ρ₁ * Real.sin θ₁ - ρ₂ * Real.sin θ₂) = 
      -(ρ₁ * Real.cos θ₁ - ρ₂ * Real.cos θ₂) * (ρ₁ * Real.sin θ₁ - ρ₂ * Real.sin θ₂)) :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_circle_l620_62092


namespace NUMINAMATH_CALUDE_cats_problem_l620_62031

/-- Given an initial number of cats and the number of female and male kittens,
    calculate the total number of cats. -/
def total_cats (initial : ℕ) (female_kittens : ℕ) (male_kittens : ℕ) : ℕ :=
  initial + female_kittens + male_kittens

/-- Theorem stating that given 2 initial cats, 3 female kittens, and 2 male kittens,
    the total number of cats is 7. -/
theorem cats_problem : total_cats 2 3 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_cats_problem_l620_62031


namespace NUMINAMATH_CALUDE_angle_measure_proof_l620_62019

theorem angle_measure_proof (x : ℝ) : 
  (180 - x = 3 * (90 - x)) → x = 45 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l620_62019


namespace NUMINAMATH_CALUDE_japanese_students_fraction_l620_62029

theorem japanese_students_fraction (j : ℕ) (s : ℕ) : 
  s = 2 * j →
  (3 * s / 8 + j / 4) / (j + s) = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_japanese_students_fraction_l620_62029


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l620_62012

theorem arithmetic_geometric_mean_inequality
  (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  (a + b + c) / 3 ≥ (a * b * c) ^ (1/3) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l620_62012


namespace NUMINAMATH_CALUDE_triangle_cosine_theorem_l620_62098

theorem triangle_cosine_theorem (A B C : ℝ) (h_triangle : A + B + C = π) 
  (h_cos_A : Real.cos A = 4/5) (h_cos_B : Real.cos B = 7/25) : 
  Real.cos C = 44/125 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_theorem_l620_62098


namespace NUMINAMATH_CALUDE_remaining_cooking_times_l620_62042

/-- Calculates the remaining cooking time in seconds for a food item -/
def remainingCookingTime (recommendedTime actualTime : ℕ) : ℕ :=
  (recommendedTime - actualTime) * 60

/-- Represents the cooking times for different food items -/
structure CookingTimes where
  frenchFries : ℕ
  chickenNuggets : ℕ
  mozzarellaSticks : ℕ

/-- Theorem stating the remaining cooking times for each food item -/
theorem remaining_cooking_times 
  (recommended : CookingTimes) 
  (actual : CookingTimes) : 
  remainingCookingTime recommended.frenchFries actual.frenchFries = 600 ∧
  remainingCookingTime recommended.chickenNuggets actual.chickenNuggets = 780 ∧
  remainingCookingTime recommended.mozzarellaSticks actual.mozzarellaSticks = 300 :=
by
  sorry

#check remaining_cooking_times (CookingTimes.mk 12 18 8) (CookingTimes.mk 2 5 3)

end NUMINAMATH_CALUDE_remaining_cooking_times_l620_62042


namespace NUMINAMATH_CALUDE_island_distance_l620_62081

theorem island_distance (n : ℝ) : 
  let a := 8*n
  let b := 5*n
  let c := 7*n
  let α := 60 * π / 180
  a^2 + b^2 - 2*a*b*Real.cos α = c^2 := by sorry

end NUMINAMATH_CALUDE_island_distance_l620_62081


namespace NUMINAMATH_CALUDE_farm_field_area_l620_62094

/-- The area of a farm field given specific ploughing conditions --/
theorem farm_field_area (planned_rate : ℝ) (actual_rate : ℝ) (extra_days : ℕ) (area_left : ℝ) : 
  planned_rate = 120 →
  actual_rate = 85 →
  extra_days = 2 →
  area_left = 40 →
  ∃ (planned_days : ℝ), 
    planned_rate * planned_days = actual_rate * (planned_days + extra_days) + area_left ∧
    planned_rate * planned_days = 720 := by
  sorry

end NUMINAMATH_CALUDE_farm_field_area_l620_62094


namespace NUMINAMATH_CALUDE_derivative_f_at_one_l620_62017

noncomputable def f (x : ℝ) : ℝ := (Real.log x + 2^x) / x^2

theorem derivative_f_at_one :
  deriv f 1 = 2 * Real.log 2 - 3 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_one_l620_62017


namespace NUMINAMATH_CALUDE_chef_cherry_pies_l620_62045

/-- Given a chef with an initial number of cherries, some used cherries, and a fixed number of cherries required per pie, 
    this function calculates the maximum number of additional pies that can be made with the remaining cherries. -/
def max_additional_pies (initial_cherries used_cherries cherries_per_pie : ℕ) : ℕ :=
  (initial_cherries - used_cherries) / cherries_per_pie

/-- Theorem stating that for the given values, the maximum number of additional pies is 4. -/
theorem chef_cherry_pies : max_additional_pies 500 350 35 = 4 := by
  sorry

end NUMINAMATH_CALUDE_chef_cherry_pies_l620_62045


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_l620_62014

theorem largest_divisor_of_n (n : ℕ) (h1 : n > 0) (h2 : 450 ∣ n^2) : 
  ∀ d : ℕ, d > 0 ∧ d ∣ n → d ≤ 30 :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_l620_62014


namespace NUMINAMATH_CALUDE_log_expression_equals_one_l620_62022

noncomputable def a : ℝ := Real.log 5 / Real.log 6
noncomputable def b : ℝ := Real.log 3 / Real.log 10
noncomputable def c : ℝ := Real.log 2 / Real.log 15

theorem log_expression_equals_one :
  (1 - 2 * a * b * c) / (a * b + b * c + c * a) = 1 := by sorry

end NUMINAMATH_CALUDE_log_expression_equals_one_l620_62022


namespace NUMINAMATH_CALUDE_curve_transformation_l620_62091

theorem curve_transformation (x : ℝ) : 
  Real.sin (2 * x) = Real.sin (2 * (x + π / 8) + π / 4) := by sorry

end NUMINAMATH_CALUDE_curve_transformation_l620_62091


namespace NUMINAMATH_CALUDE_fraction_sum_l620_62011

theorem fraction_sum : (2 : ℚ) / 5 + 3 / 8 + 1 = 71 / 40 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_l620_62011


namespace NUMINAMATH_CALUDE_expression_simplification_l620_62036

theorem expression_simplification :
  1 / (1 / ((Real.sqrt 2 + 1)^2) + 1 / ((Real.sqrt 5 - 2)^3)) = 
  1 / (41 + 17 * Real.sqrt 5 - 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l620_62036


namespace NUMINAMATH_CALUDE_cycle_gain_percent_l620_62015

theorem cycle_gain_percent (cost_price selling_price : ℝ) (h1 : cost_price = 1500) (h2 : selling_price = 1620) :
  (selling_price - cost_price) / cost_price * 100 = 8 := by
  sorry

end NUMINAMATH_CALUDE_cycle_gain_percent_l620_62015


namespace NUMINAMATH_CALUDE_max_children_in_class_l620_62001

/-- The maximum number of children in a class given the distribution of items -/
theorem max_children_in_class (total_apples total_cookies total_chocolates : ℕ)
  (leftover_apples leftover_cookies leftover_chocolates : ℕ)
  (h1 : total_apples = 55)
  (h2 : total_cookies = 114)
  (h3 : total_chocolates = 83)
  (h4 : leftover_apples = 3)
  (h5 : leftover_cookies = 10)
  (h6 : leftover_chocolates = 5) :
  Nat.gcd (total_apples - leftover_apples)
    (Nat.gcd (total_cookies - leftover_cookies) (total_chocolates - leftover_chocolates)) = 26 := by
  sorry

end NUMINAMATH_CALUDE_max_children_in_class_l620_62001


namespace NUMINAMATH_CALUDE_glucose_solution_volume_l620_62027

/-- Given a glucose solution with a concentration of 15 grams per 100 cubic centimeters,
    prove that a volume containing 9.75 grams of glucose is 65 cubic centimeters. -/
theorem glucose_solution_volume 
  (concentration : ℝ) 
  (volume : ℝ) 
  (glucose_mass : ℝ) 
  (h1 : concentration = 15 / 100) 
  (h2 : glucose_mass = 9.75) 
  (h3 : concentration * volume = glucose_mass) : 
  volume = 65 := by
sorry

end NUMINAMATH_CALUDE_glucose_solution_volume_l620_62027


namespace NUMINAMATH_CALUDE_smallest_solution_quadratic_l620_62087

theorem smallest_solution_quadratic (x : ℝ) : 
  (2 * x^2 + 30 * x - 84 = x * (x + 15)) → x ≥ -28 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_quadratic_l620_62087


namespace NUMINAMATH_CALUDE_savings_ratio_l620_62060

theorem savings_ratio (initial_savings : ℝ) (may_savings : ℝ) :
  initial_savings = 10 →
  may_savings = 160 →
  ∃ (r : ℝ), r > 0 ∧ may_savings = initial_savings * r^4 →
  r = 2 := by
sorry

end NUMINAMATH_CALUDE_savings_ratio_l620_62060


namespace NUMINAMATH_CALUDE_view_characteristics_l620_62056

/-- Represents the view described in the problem -/
structure View where
  endless_progress : Bool
  unlimited_capacity : Bool
  unlimited_resources : Bool

/-- Represents the characteristics of the view -/
structure ViewCharacteristics where
  emphasizes_subjective_initiative : Bool
  ignores_objective_conditions : Bool

/-- Theorem stating that the given view unilaterally emphasizes subjective initiative
    while ignoring objective conditions and laws -/
theorem view_characteristics (v : View) 
  (h1 : v.endless_progress = true)
  (h2 : v.unlimited_capacity = true)
  (h3 : v.unlimited_resources = true) :
  ∃ (c : ViewCharacteristics), 
    c.emphasizes_subjective_initiative ∧ c.ignores_objective_conditions := by
  sorry


end NUMINAMATH_CALUDE_view_characteristics_l620_62056


namespace NUMINAMATH_CALUDE_min_value_4x_minus_y_l620_62053

theorem min_value_4x_minus_y (x y : ℝ) 
  (h1 : x - y ≥ 0) 
  (h2 : x + y - 4 ≥ 0) 
  (h3 : x ≤ 4) : 
  ∃ (m : ℝ), m = 6 ∧ ∀ (x' y' : ℝ), 
    x' - y' ≥ 0 → x' + y' - 4 ≥ 0 → x' ≤ 4 → 4 * x' - y' ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_4x_minus_y_l620_62053


namespace NUMINAMATH_CALUDE_bottom_right_is_one_l620_62071

/-- Represents a 3x3 grid --/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Check if two positions are adjacent --/
def adjacent (p q : Fin 3 × Fin 3) : Prop :=
  (p.1 = q.1 ∧ (p.2.val + 1 = q.2.val ∨ q.2.val + 1 = p.2.val)) ∨
  (p.2 = q.2 ∧ (p.1.val + 1 = q.1.val ∨ q.1.val + 1 = p.1.val))

/-- Check if two numbers are consecutive --/
def consecutive (m n : Fin 9) : Prop :=
  m.val + 1 = n.val ∨ n.val + 1 = m.val

/-- The theorem to prove --/
theorem bottom_right_is_one (g : Grid) :
  (∀ i j k l : Fin 3, g i j ≠ g k l → (i, j) ≠ (k, l)) →
  (∀ i j k l : Fin 3, consecutive (g i j) (g k l) → adjacent (i, j) (k, l)) →
  (g 0 0).val + (g 0 2).val + (g 2 0).val + (g 2 2).val = 24 →
  (g 1 1).val + (g 0 1).val + (g 1 0).val + (g 1 2).val + (g 2 1).val = 25 →
  (g 2 2).val = 1 := by
  sorry

end NUMINAMATH_CALUDE_bottom_right_is_one_l620_62071


namespace NUMINAMATH_CALUDE_line_properties_l620_62049

def line_vector (t : ℝ) : ℝ × ℝ × ℝ := sorry

theorem line_properties :
  (∃ line_vector : ℝ → ℝ × ℝ × ℝ,
    (line_vector (-2) = (2, 4, 7)) ∧
    (line_vector 1 = (-1, 0, -3))) →
  (∃ line_vector : ℝ → ℝ × ℝ × ℝ,
    (line_vector (-2) = (2, 4, 7)) ∧
    (line_vector 1 = (-1, 0, -3)) ∧
    (line_vector (-1) = (1, 8, 5)) ∧
    (¬ ∃ t : ℝ, line_vector t = (3, 604, -6))) :=
by sorry

end NUMINAMATH_CALUDE_line_properties_l620_62049


namespace NUMINAMATH_CALUDE_unique_three_digit_sum_27_l620_62093

/-- A three-digit number is a natural number between 100 and 999 inclusive. -/
def ThreeDigitNumber (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- The sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

/-- The main theorem: there is exactly one three-digit number whose digits sum to 27 -/
theorem unique_three_digit_sum_27 : ∃! n : ℕ, ThreeDigitNumber n ∧ sumOfDigits n = 27 := by
  sorry


end NUMINAMATH_CALUDE_unique_three_digit_sum_27_l620_62093


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l620_62059

theorem arithmetic_mean_problem (x : ℕ) : 
  let numbers := [3, 117, 915, 138, 1917, 2114, x]
  (numbers.sum % 7 = 7) →
  (numbers.sum / numbers.length : ℚ) = 745 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l620_62059


namespace NUMINAMATH_CALUDE_exactlyOneAndTwoBlackMutuallyExclusiveNotContradictory_l620_62008

/-- Represents the outcome of drawing two balls from a bag -/
inductive DrawOutcome
| OneBOne  -- One black, one red
| TwoB     -- Two black
| TwoR     -- Two red

/-- The probability space for drawing two balls from a bag with 2 red and 3 black balls -/
def drawProbSpace : Type := DrawOutcome

/-- The event "Exactly one black ball is drawn" -/
def exactlyOneBlack (outcome : drawProbSpace) : Prop :=
  outcome = DrawOutcome.OneBOne

/-- The event "Exactly two black balls are drawn" -/
def exactlyTwoBlack (outcome : drawProbSpace) : Prop :=
  outcome = DrawOutcome.TwoB

/-- Two events are mutually exclusive if they cannot occur simultaneously -/
def mutuallyExclusive (e1 e2 : drawProbSpace → Prop) : Prop :=
  ∀ (outcome : drawProbSpace), ¬(e1 outcome ∧ e2 outcome)

/-- Two events are contradictory if exactly one of them must occur -/
def contradictory (e1 e2 : drawProbSpace → Prop) : Prop :=
  ∀ (outcome : drawProbSpace), e1 outcome ↔ ¬(e2 outcome)

theorem exactlyOneAndTwoBlackMutuallyExclusiveNotContradictory :
  mutuallyExclusive exactlyOneBlack exactlyTwoBlack ∧
  ¬(contradictory exactlyOneBlack exactlyTwoBlack) := by
  sorry

end NUMINAMATH_CALUDE_exactlyOneAndTwoBlackMutuallyExclusiveNotContradictory_l620_62008


namespace NUMINAMATH_CALUDE_cistern_emptied_in_8_minutes_l620_62064

/-- Given a pipe that can empty 2/3 of a cistern in 10 minutes,
    this function calculates the part of the cistern that will be empty in t minutes. -/
def cisternEmptied (t : ℚ) : ℚ :=
  (2/3) * (t / 10)

/-- Theorem stating that the part of the cistern emptied in 8 minutes is 8/15. -/
theorem cistern_emptied_in_8_minutes :
  cisternEmptied 8 = 8/15 := by
  sorry

end NUMINAMATH_CALUDE_cistern_emptied_in_8_minutes_l620_62064


namespace NUMINAMATH_CALUDE_possible_values_of_a_l620_62099

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then Real.log x + a / x
  else if x < 0 then -(Real.log (-x) + a / (-x))
  else 0

-- Define the theorem
theorem possible_values_of_a (a : ℝ) :
  (∀ x, f a (-x) = -(f a x)) →  -- f is odd
  (∃ x₁ x₂ x₃ x₄ : ℝ,
    x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄ ∧  -- x₁ < x₂ < x₃ < x₄
    x₁ + x₄ = 0 ∧  -- x₁ + x₄ = 0
    ∃ r : ℝ,  -- Existence of common ratio r for geometric sequence
      f a x₂ = r * f a x₁ ∧
      f a x₃ = r * f a x₂ ∧
      f a x₄ = r * f a x₃ ∧
    ∃ d : ℝ,  -- Existence of common difference d for arithmetic sequence
      x₂ = x₁ + d ∧
      x₃ = x₂ + d ∧
      x₄ = x₃ + d) →
  a ≤ Real.sqrt 3 / (2 * Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l620_62099


namespace NUMINAMATH_CALUDE_triangle_arithmetic_angle_sequence_l620_62052

theorem triangle_arithmetic_angle_sequence (A B C : Real) : 
  -- The angles form a triangle
  A + B + C = Real.pi →
  -- The angles form an arithmetic sequence
  A + C = 2 * B →
  -- Prove that sin B = √3/2
  Real.sin B = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_arithmetic_angle_sequence_l620_62052


namespace NUMINAMATH_CALUDE_ordering_abc_l620_62088

theorem ordering_abc :
  let a : ℝ := 31/32
  let b : ℝ := Real.cos (1/4)
  let c : ℝ := 4 * Real.sin (1/4)
  c > b ∧ b > a := by sorry

end NUMINAMATH_CALUDE_ordering_abc_l620_62088


namespace NUMINAMATH_CALUDE_initial_dolphins_count_l620_62013

/-- The initial number of dolphins in the ocean -/
def initial_dolphins : ℕ := 65

/-- The number of dolphins joining from the river -/
def joining_dolphins : ℕ := 3 * initial_dolphins

/-- The total number of dolphins after joining -/
def total_dolphins : ℕ := 260

theorem initial_dolphins_count : initial_dolphins = 65 :=
  by sorry

end NUMINAMATH_CALUDE_initial_dolphins_count_l620_62013


namespace NUMINAMATH_CALUDE_files_deleted_l620_62048

/-- Given the initial number of files and the number of files left after deletion,
    prove that the number of files deleted is 14. -/
theorem files_deleted (initial_files : ℕ) (files_left : ℕ) 
  (h1 : initial_files = 21) 
  (h2 : files_left = 7) : 
  initial_files - files_left = 14 := by
  sorry


end NUMINAMATH_CALUDE_files_deleted_l620_62048


namespace NUMINAMATH_CALUDE_max_sum_reciprocal_ninth_l620_62016

theorem max_sum_reciprocal_ninth (a b : ℕ+) (h : (a : ℚ)⁻¹ + (b : ℚ)⁻¹ = (9 : ℚ)⁻¹) :
  (a : ℕ) + b ≤ 100 ∧ ∃ (a' b' : ℕ+), (a' : ℚ)⁻¹ + (b' : ℚ)⁻¹ = (9 : ℚ)⁻¹ ∧ (a' : ℕ) + b' = 100 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_reciprocal_ninth_l620_62016


namespace NUMINAMATH_CALUDE_day_after_53_from_monday_is_friday_l620_62003

/-- Days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday
  | DayOfWeek.Sunday => DayOfWeek.Monday

/-- Function to get the day of the week after a given number of days -/
def dayAfter (start : DayOfWeek) (days : Nat) : DayOfWeek :=
  match days with
  | 0 => start
  | n + 1 => nextDay (dayAfter start n)

theorem day_after_53_from_monday_is_friday :
  dayAfter DayOfWeek.Monday 53 = DayOfWeek.Friday := by
  sorry

end NUMINAMATH_CALUDE_day_after_53_from_monday_is_friday_l620_62003


namespace NUMINAMATH_CALUDE_square_garden_perimeter_l620_62054

theorem square_garden_perimeter (area : Real) (perimeter : Real) :
  area = 90.25 →
  area = 2 * perimeter + 14.25 →
  perimeter = 38 := by
  sorry

end NUMINAMATH_CALUDE_square_garden_perimeter_l620_62054


namespace NUMINAMATH_CALUDE_park_perimeter_l620_62034

/-- Given a square park with a road inside, proves that the perimeter is 600 meters -/
theorem park_perimeter (s : ℝ) : 
  s > 0 →  -- The side length is positive
  s^2 - (s - 6)^2 = 1764 →  -- The area of the road is 1764 sq meters
  4 * s = 600 :=  -- The perimeter is 600 meters
by
  sorry

end NUMINAMATH_CALUDE_park_perimeter_l620_62034


namespace NUMINAMATH_CALUDE_triangle_side_length_l620_62078

/-- Checks if three lengths can form a valid triangle --/
def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem: In a triangle with sides 5, 8, and a, the only valid value for a is 9 --/
theorem triangle_side_length : ∃! a : ℝ, is_valid_triangle 5 8 a ∧ a > 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l620_62078


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l620_62037

theorem sqrt_product_simplification (y : ℝ) (h : y ≥ 0) :
  Real.sqrt (48 * y) * Real.sqrt (18 * y) * Real.sqrt (50 * y) = 120 * y * Real.sqrt (3 * y) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l620_62037


namespace NUMINAMATH_CALUDE_sum_of_digits_l620_62044

theorem sum_of_digits (a b c d : Nat) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 →
  b + c = 10 →
  c + d = 1 →
  a + d = 2 →
  a + b + c + d = 13 := by
sorry

end NUMINAMATH_CALUDE_sum_of_digits_l620_62044


namespace NUMINAMATH_CALUDE_gondor_thursday_laptops_l620_62046

/-- Represents the earnings and repair counts for Gondor --/
structure GondorEarnings where
  phone_repair_price : ℕ
  laptop_repair_price : ℕ
  monday_phones : ℕ
  tuesday_phones : ℕ
  wednesday_laptops : ℕ
  total_earnings : ℕ

/-- Calculates the number of laptops repaired on Thursday --/
def thursday_laptops (g : GondorEarnings) : ℕ :=
  let mon_tue_wed_earnings := g.phone_repair_price * (g.monday_phones + g.tuesday_phones) + 
                              g.laptop_repair_price * g.wednesday_laptops
  let thursday_earnings := g.total_earnings - mon_tue_wed_earnings
  thursday_earnings / g.laptop_repair_price

/-- Theorem stating that Gondor repaired 4 laptops on Thursday --/
theorem gondor_thursday_laptops :
  let g : GondorEarnings := {
    phone_repair_price := 10,
    laptop_repair_price := 20,
    monday_phones := 3,
    tuesday_phones := 5,
    wednesday_laptops := 2,
    total_earnings := 200
  }
  thursday_laptops g = 4 := by sorry

end NUMINAMATH_CALUDE_gondor_thursday_laptops_l620_62046


namespace NUMINAMATH_CALUDE_subtract_decimals_l620_62050

theorem subtract_decimals : 3.79 - 2.15 = 1.64 := by
  sorry

end NUMINAMATH_CALUDE_subtract_decimals_l620_62050


namespace NUMINAMATH_CALUDE_part_1_part_2_l620_62075

-- Define the sets M, N, and H
def M : Set ℝ := {x | 1 < x ∧ x < 3}
def N : Set ℝ := {x | x^2 - 6*x + 8 ≤ 0}
def H (a : ℝ) : Set ℝ := {x | |x - a| ≤ 2}

-- Define the custom set operation ∆
def triangleOp (A B : Set ℝ) : Set ℝ := A ∩ (Set.univ \ B)

-- Theorem for part (1)
theorem part_1 :
  triangleOp M N = {x | 1 < x ∧ x < 2} ∧
  triangleOp N M = {x | 3 ≤ x ∧ x ≤ 4} := by sorry

-- Theorem for part (2)
theorem part_2 (a : ℝ) :
  triangleOp (triangleOp N M) (H a) =
    if a ≥ 4 ∨ a ≤ -1 then
      {x | 1 < x ∧ x < 2}
    else if 3 < a ∧ a < 4 then
      {x | 1 < x ∧ x < a - 2}
    else if -1 < a ∧ a < 0 then
      {x | a + 2 < x ∧ x < 2}
    else
      ∅ := by sorry

end NUMINAMATH_CALUDE_part_1_part_2_l620_62075


namespace NUMINAMATH_CALUDE_arithmetic_sum_10_to_100_l620_62089

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) : ℕ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

/-- Theorem: The sum of the arithmetic series from 10 to 100 with common difference 1 is 5005 -/
theorem arithmetic_sum_10_to_100 :
  arithmetic_sum 10 100 1 = 5005 := by
  sorry

#eval arithmetic_sum 10 100 1

end NUMINAMATH_CALUDE_arithmetic_sum_10_to_100_l620_62089


namespace NUMINAMATH_CALUDE_function_graph_point_l620_62004

theorem function_graph_point (f : ℝ → ℝ) (h : f 8 = 5) :
  let g := fun x => (f (3 * x) / 3 + 3) / 3
  g (8/3) = 14/9 ∧ 8/3 + 14/9 = 38/9 := by
sorry

end NUMINAMATH_CALUDE_function_graph_point_l620_62004


namespace NUMINAMATH_CALUDE_texas_california_plate_difference_l620_62028

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible California license plates -/
def california_plates : ℕ := num_letters^3 * num_digits^3

/-- The number of possible Texas license plates -/
def texas_plates : ℕ := num_letters^3 * num_digits^4

/-- The difference in the number of possible license plates between Texas and California -/
def plate_difference : ℕ := texas_plates - california_plates

theorem texas_california_plate_difference :
  plate_difference = 158184000 := by
  sorry

end NUMINAMATH_CALUDE_texas_california_plate_difference_l620_62028


namespace NUMINAMATH_CALUDE_factor_polynomial_l620_62090

theorem factor_polynomial (x : ℝ) : 54 * x^5 - 135 * x^9 = 27 * x^5 * (2 - 5 * x^4) := by sorry

end NUMINAMATH_CALUDE_factor_polynomial_l620_62090


namespace NUMINAMATH_CALUDE_bird_feeding_problem_l620_62026

/-- Given the following conditions:
    - There are 6 baby birds
    - Papa bird caught 9 worms
    - Mama bird caught 13 worms
    - 2 worms were stolen from Mama bird
    - Mama bird needs to catch 34 more worms
    - The worms are needed for 3 days
    Prove that each baby bird needs 3 worms per day. -/
theorem bird_feeding_problem (
  num_babies : ℕ)
  (papa_worms : ℕ)
  (mama_worms : ℕ)
  (stolen_worms : ℕ)
  (additional_worms : ℕ)
  (num_days : ℕ)
  (h1 : num_babies = 6)
  (h2 : papa_worms = 9)
  (h3 : mama_worms = 13)
  (h4 : stolen_worms = 2)
  (h5 : additional_worms = 34)
  (h6 : num_days = 3) :
  (papa_worms + mama_worms - stolen_worms + additional_worms) / (num_babies * num_days) = 3 := by
  sorry

#eval (9 + 13 - 2 + 34) / (6 * 3)  -- This should output 3

end NUMINAMATH_CALUDE_bird_feeding_problem_l620_62026


namespace NUMINAMATH_CALUDE_student_group_assignments_l620_62002

theorem student_group_assignments (n : ℕ) (k : ℕ) :
  n = 5 → k = 2 → (2 : ℕ) ^ n = 32 := by
  sorry

end NUMINAMATH_CALUDE_student_group_assignments_l620_62002


namespace NUMINAMATH_CALUDE_aartis_work_completion_time_l620_62063

/-- If Aarti can complete three times a piece of work in 15 days, 
    then she can complete one piece of work in 5 days. -/
theorem aartis_work_completion_time :
  ∀ (work_completion_time : ℝ),
  (3 * work_completion_time = 15) →
  work_completion_time = 5 :=
by sorry

end NUMINAMATH_CALUDE_aartis_work_completion_time_l620_62063


namespace NUMINAMATH_CALUDE_complex_equality_l620_62035

theorem complex_equality (x y z : ℝ) (α β γ : ℂ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hα : Complex.abs α = 1) (hβ : Complex.abs β = 1) (hγ : Complex.abs γ = 1)
  (hxyz : x + y + z = 0) (hαβγ : α * x + β * y + γ * z = 0) :
  α = β ∧ β = γ := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_l620_62035


namespace NUMINAMATH_CALUDE_yard_length_l620_62010

theorem yard_length (num_trees : ℕ) (distance : ℝ) :
  num_trees = 11 →
  distance = 18 →
  (num_trees - 1) * distance = 180 :=
by sorry

end NUMINAMATH_CALUDE_yard_length_l620_62010


namespace NUMINAMATH_CALUDE_equation_system_solution_l620_62068

/-- Given a system of equations with constants m and n, prove that the solution for a and b is (4, 1) -/
theorem equation_system_solution (m n : ℝ) : 
  (∃ x y : ℝ, -2*m*x + 5*y = 15 ∧ x + 7*n*y = 14 ∧ x = 5 ∧ y = 2) →
  (∃ a b : ℝ, -2*m*(a+b) + 5*(a-2*b) = 15 ∧ (a+b) + 7*n*(a-2*b) = 14 ∧ a = 4 ∧ b = 1) :=
by sorry

end NUMINAMATH_CALUDE_equation_system_solution_l620_62068


namespace NUMINAMATH_CALUDE_range_of_x_l620_62005

def p (x : ℝ) : Prop := (x + 2) * (x - 2) ≤ 0
def q (x : ℝ) : Prop := x^2 - 3*x - 4 ≤ 0

theorem range_of_x : 
  (∀ x : ℝ, ¬(p x ∧ q x)) → 
  (∀ x : ℝ, p x ∨ q x) → 
  {x : ℝ | p x ∨ q x} = {x : ℝ | -2 ≤ x ∧ x < -1} ∪ {x : ℝ | 2 < x ∧ x ≤ 4} :=
sorry

end NUMINAMATH_CALUDE_range_of_x_l620_62005


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a5_zero_l620_62047

/-- An arithmetic sequence with common difference d -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_a5_zero
  (a : ℕ → ℝ) (d : ℝ)
  (h_arith : arithmetic_sequence a d)
  (h_d_nonzero : d ≠ 0)
  (h_condition : a 3 + a 9 = a 10 - a 8) :
  a 5 = 0 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a5_zero_l620_62047


namespace NUMINAMATH_CALUDE_system_sampling_theorem_l620_62066

/-- Represents a system sampling method -/
structure SystemSampling where
  total_students : ℕ
  sample_size : ℕ
  common_difference : ℕ

/-- Checks if a list of numbers forms a valid system sample -/
def is_valid_sample (s : SystemSampling) (sample : List ℕ) : Prop :=
  sample.length = s.sample_size ∧
  ∀ i j, i < j → j < s.sample_size →
    sample[j]! - sample[i]! = s.common_difference * (j - i)

theorem system_sampling_theorem (s : SystemSampling)
  (h_total : s.total_students = 160)
  (h_size : s.sample_size = 5)
  (h_diff : s.common_difference = 32)
  (h_known : ∃ (sample : List ℕ), is_valid_sample s sample ∧ 
    40 ∈ sample ∧ 72 ∈ sample ∧ 136 ∈ sample) :
  ∃ (full_sample : List ℕ), is_valid_sample s full_sample ∧
    40 ∈ full_sample ∧ 72 ∈ full_sample ∧ 136 ∈ full_sample ∧
    8 ∈ full_sample ∧ 104 ∈ full_sample :=
sorry

end NUMINAMATH_CALUDE_system_sampling_theorem_l620_62066


namespace NUMINAMATH_CALUDE_subset_condition_iff_m_geq_three_l620_62096

theorem subset_condition_iff_m_geq_three (m : ℝ) : 
  (∀ x : ℝ, x^2 - x ≤ 0 → x^2 - 4*x + m ≥ 0) ↔ m ≥ 3 := by sorry

end NUMINAMATH_CALUDE_subset_condition_iff_m_geq_three_l620_62096


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l620_62041

theorem complex_fraction_equality : 
  (((4.625 - 13/18 * 9/26) / (9/4) + 2.5 / 1.25 / 6.75) / (1 + 53/68)) /
  ((1/2 - 0.375) / 0.125 + (5/6 - 7/12) / (0.358 - 1.4796 / 13.7)) = 17/27 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l620_62041


namespace NUMINAMATH_CALUDE_distance_is_100_miles_l620_62074

/-- Represents the fuel efficiency of a car in miles per gallon. -/
def miles_per_gallon : ℝ := 20

/-- Represents the amount of gas needed to reach Grandma's house in gallons. -/
def gallons_needed : ℝ := 5

/-- Calculates the distance to Grandma's house in miles. -/
def distance_to_grandma : ℝ := miles_per_gallon * gallons_needed

/-- Theorem stating that the distance to Grandma's house is 100 miles. -/
theorem distance_is_100_miles : distance_to_grandma = 100 :=
  sorry

end NUMINAMATH_CALUDE_distance_is_100_miles_l620_62074


namespace NUMINAMATH_CALUDE_money_distribution_l620_62080

theorem money_distribution (w x y z : ℝ) (h1 : w = 375) 
  (h2 : x = 6 * w) (h3 : y = 2 * w) (h4 : z = 4 * w) : 
  x - y = 1500 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l620_62080


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l620_62039

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + 3 * b = 1) :
  (2 / a + 3 / b) ≥ 25 := by sorry

theorem min_value_achieved (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + 3 * b = 1) :
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 2 * a₀ + 3 * b₀ = 1 ∧ (2 / a₀ + 3 / b₀) = 25 := by sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l620_62039


namespace NUMINAMATH_CALUDE_equation_solution_l620_62079

theorem equation_solution : 
  ∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = 3/5 ∧ 
  (∀ x : ℝ, (x - 3)^2 + 4*x*(x - 3) = 0 ↔ (x = x₁ ∨ x = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l620_62079


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l620_62061

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 1 + a 4 + a 7 = 39) →
  (a 2 + a 5 + a 8 = 33) →
  (a 3 + a 6 + a 9 = 27) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l620_62061


namespace NUMINAMATH_CALUDE_solve_linear_equation_l620_62067

theorem solve_linear_equation :
  ∀ x : ℚ, 3 * x + 4 = -6 * x - 11 → x = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l620_62067


namespace NUMINAMATH_CALUDE_floor_sqrt_50_squared_l620_62072

theorem floor_sqrt_50_squared : ⌊Real.sqrt 50⌋^2 = 49 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_50_squared_l620_62072


namespace NUMINAMATH_CALUDE_fraction_transformation_l620_62057

theorem fraction_transformation (x : ℚ) : (3 - 2*x) / (5 + x) = 1/2 → x = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_transformation_l620_62057


namespace NUMINAMATH_CALUDE_sqrt_four_squared_l620_62043

theorem sqrt_four_squared : (Real.sqrt 4)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_squared_l620_62043


namespace NUMINAMATH_CALUDE_f_strictly_decreasing_iff_a_in_range_l620_62006

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a^x else (a - 4) * x + 3 * a

theorem f_strictly_decreasing_iff_a_in_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) * (x₁ - x₂) < 0) ↔
  0 < a ∧ a ≤ 1/3 :=
sorry

end NUMINAMATH_CALUDE_f_strictly_decreasing_iff_a_in_range_l620_62006


namespace NUMINAMATH_CALUDE_score_difference_is_3_4_l620_62082

-- Define the score distribution
def score_distribution : List (ℝ × ℝ) := [
  (60, 0.15),
  (75, 0.20),
  (88, 0.25),
  (92, 0.10),
  (98, 0.30)
]

-- Define the mean score
def mean_score : ℝ := (score_distribution.map (λ (score, freq) => score * freq)).sum

-- Define the median score
def median_score : ℝ := 88

-- Theorem statement
theorem score_difference_is_3_4 :
  |median_score - mean_score| = 3.4 := by sorry

end NUMINAMATH_CALUDE_score_difference_is_3_4_l620_62082


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_l620_62095

theorem ceiling_floor_sum : ⌈(7 : ℚ) / 3⌉ + ⌊-(7 : ℚ) / 3⌋ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_l620_62095


namespace NUMINAMATH_CALUDE_factor_expression_l620_62025

theorem factor_expression (x : ℝ) : 3*x*(x-5) - 2*(x-5) + 4*x*(x-5) = (x-5)*(7*x-2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l620_62025


namespace NUMINAMATH_CALUDE_triangle_hypotenuse_l620_62032

theorem triangle_hypotenuse (x y : ℝ) (h : ℝ) : 
  (1/3 : ℝ) * π * y^2 * x = 1200 * π →
  (1/3 : ℝ) * π * x^2 * (2*x) = 3840 * π →
  x^2 + y^2 = h^2 →
  h = 2 * Real.sqrt 131 := by
sorry

end NUMINAMATH_CALUDE_triangle_hypotenuse_l620_62032


namespace NUMINAMATH_CALUDE_travel_time_difference_l620_62083

/-- Proves that the difference in travel time between two cars is 2 hours -/
theorem travel_time_difference (distance : ℝ) (speed_r speed_p : ℝ) : 
  distance = 600 ∧ 
  speed_r = 50 ∧ 
  speed_p = speed_r + 10 →
  distance / speed_r - distance / speed_p = 2 := by
sorry


end NUMINAMATH_CALUDE_travel_time_difference_l620_62083


namespace NUMINAMATH_CALUDE_sqrt_calculations_l620_62021

theorem sqrt_calculations :
  (∀ (a b c : ℝ), 
    a = 4 * Real.sqrt (1/2) ∧ 
    b = Real.sqrt 32 ∧ 
    c = Real.sqrt 8 →
    a + b - c = 4 * Real.sqrt 2) ∧
  (∀ (d e f g : ℝ),
    d = Real.sqrt 6 ∧
    e = Real.sqrt 3 ∧
    f = Real.sqrt 12 ∧
    g = Real.sqrt 3 →
    d * e + f / g = 3 * Real.sqrt 2 + 2) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_calculations_l620_62021


namespace NUMINAMATH_CALUDE_small_forward_duration_l620_62069

/-- Represents the duration of footage for each player in seconds. -/
structure PlayerFootage where
  pointGuard : ℕ
  shootingGuard : ℕ
  smallForward : ℕ
  powerForward : ℕ
  center : ℕ

/-- Calculates the total duration of all players' footage in seconds. -/
def totalDuration (pf : PlayerFootage) : ℕ :=
  pf.pointGuard + pf.shootingGuard + pf.smallForward + pf.powerForward + pf.center

/-- The number of players in the team. -/
def numPlayers : ℕ := 5

/-- The average duration per player in seconds. -/
def avgDurationPerPlayer : ℕ := 120 -- 2 minutes = 120 seconds

theorem small_forward_duration (pf : PlayerFootage) 
    (h1 : pf.pointGuard = 130)
    (h2 : pf.shootingGuard = 145)
    (h3 : pf.powerForward = 60)
    (h4 : pf.center = 180)
    (h5 : totalDuration pf = numPlayers * avgDurationPerPlayer) :
    pf.smallForward = 85 := by
  sorry

end NUMINAMATH_CALUDE_small_forward_duration_l620_62069


namespace NUMINAMATH_CALUDE_crossing_time_for_49_explorers_l620_62024

/-- The minimum time required for all explorers to cross a river -/
def minimum_crossing_time (
  num_explorers : ℕ
  ) (boat_capacity : ℕ
  ) (crossing_time : ℕ
  ) : ℕ :=
  -- The actual calculation would go here
  45

/-- Theorem stating that for 49 explorers, a boat capacity of 7, and a crossing time of 3 minutes,
    the minimum time to cross is 45 minutes -/
theorem crossing_time_for_49_explorers :
  minimum_crossing_time 49 7 3 = 45 := by
  sorry

end NUMINAMATH_CALUDE_crossing_time_for_49_explorers_l620_62024


namespace NUMINAMATH_CALUDE_decagon_diagonals_l620_62055

/-- The number of sides in a decagon -/
def decagon_sides : ℕ := 10

/-- Formula for the number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The number of diagonals in a regular decagon is 35 -/
theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l620_62055


namespace NUMINAMATH_CALUDE_distribute_five_into_three_l620_62033

/-- The number of ways to distribute n distinct objects into k distinct containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- Theorem: The number of ways to distribute 5 distinct objects into 3 distinct containers is 3^5 -/
theorem distribute_five_into_three : distribute 5 3 = 3^5 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_into_three_l620_62033


namespace NUMINAMATH_CALUDE_power_zero_of_sum_one_l620_62030

theorem power_zero_of_sum_one (a : ℝ) (h : a ≠ -1) : (a + 1)^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_of_sum_one_l620_62030


namespace NUMINAMATH_CALUDE_tangent_problem_l620_62009

theorem tangent_problem (α : Real) 
  (h : Real.tan (π/4 + α) = 1/2) : 
  (Real.tan α = -1/3) ∧ 
  ((Real.sin (2*α) - Real.cos α ^ 2) / (2 + Real.cos (2*α)) = (2 * Real.tan α - 1) / (3 + Real.tan α ^ 2)) := by
  sorry

end NUMINAMATH_CALUDE_tangent_problem_l620_62009


namespace NUMINAMATH_CALUDE_sum_243_62_base5_l620_62076

/-- Converts a natural number to its base 5 representation --/
def toBase5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: toBase5 (n / 5)

/-- Adds two numbers in base 5 representation --/
def addBase5 (a b : List ℕ) : List ℕ :=
  sorry  -- Implementation details omitted

/-- Theorem: The sum of 243 and 62 in base 5 is 2170₅ --/
theorem sum_243_62_base5 :
  addBase5 (toBase5 243) (toBase5 62) = [0, 7, 1, 2] :=
sorry

end NUMINAMATH_CALUDE_sum_243_62_base5_l620_62076


namespace NUMINAMATH_CALUDE_leap_year_date_statistics_l620_62007

/-- Represents the data for dates in a leap year -/
structure LeapYearData where
  dates : Fin 31 → ℕ
  sum_of_values : ℕ
  total_count : ℕ

/-- The mean of the data -/
def mean (data : LeapYearData) : ℚ :=
  data.sum_of_values / data.total_count

/-- The median of the data -/
def median (data : LeapYearData) : ℕ := 16

/-- The median of the modes -/
def median_of_modes (data : LeapYearData) : ℕ := 15

theorem leap_year_date_statistics (data : LeapYearData) 
  (h1 : ∀ i : Fin 29, data.dates i = 12)
  (h2 : data.dates 30 = 11)
  (h3 : data.dates 31 = 7)
  (h4 : data.sum_of_values = 5767)
  (h5 : data.total_count = 366) :
  median_of_modes data < mean data ∧ mean data < median data := by
  sorry


end NUMINAMATH_CALUDE_leap_year_date_statistics_l620_62007


namespace NUMINAMATH_CALUDE_band_to_orchestra_ratio_l620_62077

theorem band_to_orchestra_ratio : 
  ∀ (orchestra_students band_students choir_boys choir_girls total_students : ℕ),
    orchestra_students = 20 →
    choir_boys = 12 →
    choir_girls = 16 →
    total_students = 88 →
    total_students = orchestra_students + band_students + choir_boys + choir_girls →
    band_students = 2 * orchestra_students :=
by
  sorry

end NUMINAMATH_CALUDE_band_to_orchestra_ratio_l620_62077


namespace NUMINAMATH_CALUDE_grocery_shop_sales_l620_62038

theorem grocery_shop_sales (sales1 sales2 sales3 sales4 sales6 average_sale : ℕ)
  (h1 : sales1 = 6735)
  (h2 : sales2 = 6927)
  (h3 : sales3 = 6855)
  (h4 : sales4 = 7230)
  (h5 : sales6 = 4691)
  (h6 : average_sale = 6500) :
  ∃ sales5 : ℕ, sales5 = 6562 ∧
  (sales1 + sales2 + sales3 + sales4 + sales5 + sales6) / 6 = average_sale := by
  sorry

end NUMINAMATH_CALUDE_grocery_shop_sales_l620_62038


namespace NUMINAMATH_CALUDE_jacks_speed_l620_62070

/-- Prove Jack's speed given the conditions of the problem -/
theorem jacks_speed (initial_distance : ℝ) (christina_speed : ℝ) (lindy_speed : ℝ) (lindy_distance : ℝ) :
  initial_distance = 360 →
  christina_speed = 7 →
  lindy_speed = 12 →
  lindy_distance = 360 →
  ∃ (jack_speed : ℝ), jack_speed = 5 := by
  sorry


end NUMINAMATH_CALUDE_jacks_speed_l620_62070


namespace NUMINAMATH_CALUDE_fiftieth_term_is_248_l620_62023

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

theorem fiftieth_term_is_248 :
  arithmeticSequenceTerm 3 5 50 = 248 := by
  sorry

end NUMINAMATH_CALUDE_fiftieth_term_is_248_l620_62023


namespace NUMINAMATH_CALUDE_class_mean_calculation_l620_62073

theorem class_mean_calculation (total_students : ℕ) (group1_students : ℕ) (group2_students : ℕ)
  (group1_mean : ℚ) (group2_mean : ℚ) :
  total_students = group1_students + group2_students →
  group1_students = 24 →
  group2_students = 6 →
  group1_mean = 80 / 100 →
  group2_mean = 85 / 100 →
  (group1_students * group1_mean + group2_students * group2_mean) / total_students = 81 / 100 :=
by sorry

end NUMINAMATH_CALUDE_class_mean_calculation_l620_62073


namespace NUMINAMATH_CALUDE_vector_parallel_perpendicular_l620_62084

/-- Two vectors are parallel if their corresponding components are proportional -/
def IsParallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

/-- Two vectors are perpendicular if their dot product is zero -/
def IsPerpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- Given vectors a and b, prove the values of m for parallel and perpendicular cases -/
theorem vector_parallel_perpendicular (m : ℝ) :
  let a : ℝ × ℝ := (m, 1)
  let b : ℝ × ℝ := (1, 2)
  (IsParallel a b → m = 1/2) ∧
  (IsPerpendicular a b → m = -2) := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_perpendicular_l620_62084


namespace NUMINAMATH_CALUDE_distinct_pairs_from_twelve_l620_62020

theorem distinct_pairs_from_twelve (n : ℕ) : n = 12 → (n.choose 2 = 66) := by
  sorry

end NUMINAMATH_CALUDE_distinct_pairs_from_twelve_l620_62020


namespace NUMINAMATH_CALUDE_yuna_has_biggest_number_l620_62000

-- Define the type for students
inductive Student : Type
  | Yoongi : Student
  | Jungkook : Student
  | Yuna : Student
  | Yoojung : Student

-- Define a function that assigns numbers to students
def studentNumber : Student → Nat
  | Student.Yoongi => 7
  | Student.Jungkook => 6
  | Student.Yuna => 9
  | Student.Yoojung => 8

-- Theorem statement
theorem yuna_has_biggest_number :
  (∀ s : Student, studentNumber s ≤ studentNumber Student.Yuna) ∧
  studentNumber Student.Yuna = 9 := by
  sorry

end NUMINAMATH_CALUDE_yuna_has_biggest_number_l620_62000


namespace NUMINAMATH_CALUDE_gcd_of_polynomial_and_linear_l620_62097

theorem gcd_of_polynomial_and_linear (a : ℤ) (h : ∃ k : ℤ, a = 360 * k) :
  Int.gcd (a^2 + 6*a + 8) (a + 4) = 4 := by sorry

end NUMINAMATH_CALUDE_gcd_of_polynomial_and_linear_l620_62097


namespace NUMINAMATH_CALUDE_wind_velocity_problem_l620_62040

/-- Represents the relationship between pressure, area, and velocity -/
def pressure_relation (k : ℝ) (A : ℝ) (V : ℝ) : ℝ := k * A * V^3

theorem wind_velocity_problem (k : ℝ) :
  (pressure_relation k 1 8 = 1) →
  (pressure_relation k 9 12 = 27) :=
by sorry

end NUMINAMATH_CALUDE_wind_velocity_problem_l620_62040


namespace NUMINAMATH_CALUDE_expression_simplification_l620_62062

theorem expression_simplification (a b : ℝ) 
  (h1 : a ≠ b/2) (h2 : a ≠ -b/2) (h3 : a ≠ -b) (h4 : a ≠ 0) (h5 : b ≠ 0) :
  (((a - b) / (2*a - b) - (a^2 + b^2 + a) / (2*a^2 + a*b - b^2)) / 
   ((4*b^4 + 4*a*b^2 + a^2) / (2*b^2 + a))) * (b^2 + b + a*b + a) = 
  (b + 1) / (b - 2*a) := by
sorry


end NUMINAMATH_CALUDE_expression_simplification_l620_62062


namespace NUMINAMATH_CALUDE_only_345_is_pythagorean_triple_l620_62051

/-- Checks if three numbers form a Pythagorean triple -/
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

/-- Proof that (3, 4, 5) is the only Pythagorean triple among the given sets -/
theorem only_345_is_pythagorean_triple :
  (¬ isPythagoreanTriple 1 2 3) ∧
  (isPythagoreanTriple 3 4 5) ∧
  (¬ isPythagoreanTriple 4 5 6) ∧
  (¬ isPythagoreanTriple 7 8 9) :=
sorry

end NUMINAMATH_CALUDE_only_345_is_pythagorean_triple_l620_62051


namespace NUMINAMATH_CALUDE_square_root_possible_value_l620_62086

theorem square_root_possible_value (a : ℝ) : 
  (a = -1 ∨ a = -6 ∨ a = 3 ∨ a = -7) → 
  (∃ x : ℝ, x^2 = a) ↔ a = 3 :=
by sorry

end NUMINAMATH_CALUDE_square_root_possible_value_l620_62086
