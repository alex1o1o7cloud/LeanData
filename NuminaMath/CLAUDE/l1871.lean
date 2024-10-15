import Mathlib

namespace NUMINAMATH_CALUDE_function_and_inequality_problem_l1871_187196

/-- Given a function f(x) = b * a^x with the specified properties, 
    prove that f(x) = 3 * 2^x and find the maximum value of m. -/
theorem function_and_inequality_problem 
  (f : ℝ → ℝ) 
  (a b : ℝ) 
  (h1 : ∀ x, f x = b * a^x)
  (h2 : a > 0)
  (h3 : a ≠ 1)
  (h4 : f 1 = 6)
  (h5 : f 3 = 24) :
  (∀ x, f x = 3 * 2^x) ∧ 
  (∀ m, (∀ x ≤ 1, (1/a)^x + (1/b)^x - m ≥ 0) ↔ m ≤ 5/6) :=
by sorry

end NUMINAMATH_CALUDE_function_and_inequality_problem_l1871_187196


namespace NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l1871_187187

theorem greatest_divisor_four_consecutive_integers (n : ℕ+) :
  ∃ (k : ℕ), k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) ∧
  ∀ (m : ℕ), m ∣ (n * (n + 1) * (n + 2) * (n + 3)) → m ≤ k :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l1871_187187


namespace NUMINAMATH_CALUDE_tan_half_angle_fourth_quadrant_l1871_187112

theorem tan_half_angle_fourth_quadrant (α : Real) :
  (α ∈ Set.Icc (3 * Real.pi / 2) (2 * Real.pi)) →  -- α is in the fourth quadrant
  (Real.sin α + Real.cos α = 1 / 5) →               -- given condition
  Real.tan (α / 2) = -1 / 3 := by                   -- conclusion to prove
sorry


end NUMINAMATH_CALUDE_tan_half_angle_fourth_quadrant_l1871_187112


namespace NUMINAMATH_CALUDE_special_function_unique_l1871_187181

/-- A function satisfying the given conditions -/
def SpecialFunction (g : ℝ → ℝ) : Prop :=
  g 1 = 2 ∧ ∀ x y : ℝ, g (x^2 - y^2) = (x - y) * (g x + g y)

/-- Theorem stating that any function satisfying the conditions must be g(x) = 2x -/
theorem special_function_unique (g : ℝ → ℝ) (h : SpecialFunction g) :
    ∀ x : ℝ, g x = 2 * x := by
  sorry

end NUMINAMATH_CALUDE_special_function_unique_l1871_187181


namespace NUMINAMATH_CALUDE_men_complete_nine_units_l1871_187138

/-- The number of men in the committee -/
def num_men : ℕ := 250

/-- The number of women in the committee -/
def num_women : ℕ := 150

/-- The number of units completed per day when all men and women work -/
def total_units : ℕ := 12

/-- The number of units completed per day when only women work -/
def women_units : ℕ := 3

/-- The number of units completed per day by men -/
def men_units : ℕ := total_units - women_units

theorem men_complete_nine_units : men_units = 9 := by
  sorry

end NUMINAMATH_CALUDE_men_complete_nine_units_l1871_187138


namespace NUMINAMATH_CALUDE_sampling_inspection_correct_for_yeast_l1871_187142

/-- Represents a biological experimental technique --/
inductive BiologicalTechnique
| YeastCounting
| SoilAnimalRichness
| OnionRootMitosis
| FatIdentification

/-- Represents a method used in biological experiments --/
inductive ExperimentalMethod
| SamplingInspection
| MarkRecapture
| RinsingForDye
| HydrochloricAcidWashing

/-- Function that returns the correct method for a given technique --/
def correct_method (technique : BiologicalTechnique) : ExperimentalMethod :=
  match technique with
  | BiologicalTechnique.YeastCounting => ExperimentalMethod.SamplingInspection
  | _ => ExperimentalMethod.SamplingInspection  -- Placeholder for other techniques

/-- Theorem stating that the sampling inspection method is correct for yeast counting --/
theorem sampling_inspection_correct_for_yeast :
  correct_method BiologicalTechnique.YeastCounting = ExperimentalMethod.SamplingInspection :=
by sorry

end NUMINAMATH_CALUDE_sampling_inspection_correct_for_yeast_l1871_187142


namespace NUMINAMATH_CALUDE_inverse_of_B_squared_l1871_187197

theorem inverse_of_B_squared (B : Matrix (Fin 3) (Fin 3) ℝ) 
  (h : B⁻¹ = ![![2, -3, 0], ![0, -1, 0], ![0, 0, 5]]) : 
  (B^2)⁻¹ = ![![4, -3, 0], ![0, 1, 0], ![0, 0, 25]] := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_B_squared_l1871_187197


namespace NUMINAMATH_CALUDE_f_derivative_at_one_l1871_187172

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x)

theorem f_derivative_at_one :
  ∀ x : ℝ, x ≠ 0 → f (1 / x) = x / (1 + x) →
  HasDerivAt f (-1/4) 1 :=
by sorry

end NUMINAMATH_CALUDE_f_derivative_at_one_l1871_187172


namespace NUMINAMATH_CALUDE_largest_812_double_l1871_187192

/-- Converts a natural number to its base-8 representation as a list of digits --/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Interprets a list of digits as a base-12 number --/
def fromBase12 (digits : List ℕ) : ℕ :=
  sorry

/-- Checks if a number is an 8-12 double --/
def is812Double (n : ℕ) : Prop :=
  fromBase12 (toBase8 n) = 3 * n

theorem largest_812_double :
  ∀ n : ℕ, n > 3 → ¬(is812Double n) :=
sorry

end NUMINAMATH_CALUDE_largest_812_double_l1871_187192


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l1871_187126

theorem quadratic_inequality_condition (x : ℝ) : 
  2 * x^2 - 5 * x - 3 ≥ 0 ↔ x ≤ -1/2 ∨ x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l1871_187126


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1871_187189

theorem min_value_of_expression (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 6) :
  (x^2 + y^2) / (x + y) + (x^2 + z^2) / (x + z) + (y^2 + z^2) / (y + z) ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1871_187189


namespace NUMINAMATH_CALUDE_base_n_representation_of_b_l1871_187122

theorem base_n_representation_of_b (n : ℕ) (a b : ℤ) (x y : ℚ) : 
  n > 9 →
  x^2 - a*x + b = 0 →
  y^2 - a*y + b = 0 →
  (x = n ∨ y = n) →
  2*x - y = 6 →
  a = 2*n + 7 →
  b = 14 :=
by sorry

end NUMINAMATH_CALUDE_base_n_representation_of_b_l1871_187122


namespace NUMINAMATH_CALUDE_inequality_condition_l1871_187143

theorem inequality_condition (p : ℝ) :
  (∀ x₁ x₂ x₃ : ℝ, x₁^2 + x₂^2 + x₃^2 ≥ p * (x₁ * x₂ + x₂ * x₃)) ↔ p ≤ Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_condition_l1871_187143


namespace NUMINAMATH_CALUDE_hundredth_digit_is_two_l1871_187105

/-- The decimal representation of 7/26 has a repeating cycle of 9 digits -/
def decimal_cycle : Fin 9 → Nat
| 0 => 2
| 1 => 6
| 2 => 9
| 3 => 2
| 4 => 3
| 5 => 0
| 6 => 7
| 7 => 6
| 8 => 9

/-- The 100th digit after the decimal point in the decimal representation of 7/26 -/
def hundredth_digit : Nat :=
  decimal_cycle (100 % 9)

theorem hundredth_digit_is_two : hundredth_digit = 2 := by
  sorry

end NUMINAMATH_CALUDE_hundredth_digit_is_two_l1871_187105


namespace NUMINAMATH_CALUDE_expression_evaluation_l1871_187174

/-- Given x = y + z and y > z > 0, prove that ((x+y)^z + (x+z)^y) / (y^z + z^y) = 2^y + 2^z -/
theorem expression_evaluation (x y z : ℝ) 
  (h1 : x = y + z) 
  (h2 : y > z) 
  (h3 : z > 0) : 
  ((x + y)^z + (x + z)^y) / (y^z + z^y) = 2^y + 2^z :=
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1871_187174


namespace NUMINAMATH_CALUDE_average_of_xyz_l1871_187185

theorem average_of_xyz (x y z : ℝ) (h : (5 / 4) * (x + y + z) = 20) :
  (x + y + z) / 3 = 16 / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_of_xyz_l1871_187185


namespace NUMINAMATH_CALUDE_fill_jug_completely_l1871_187153

/-- The capacity of the jug in milliliters -/
def jug_capacity : ℕ := 800

/-- The capacity of a small container in milliliters -/
def container_capacity : ℕ := 48

/-- The minimum number of small containers needed to fill the jug completely -/
def min_containers : ℕ := 17

theorem fill_jug_completely :
  min_containers = (jug_capacity + container_capacity - 1) / container_capacity ∧
  min_containers * container_capacity ≥ jug_capacity ∧
  (min_containers - 1) * container_capacity < jug_capacity := by
  sorry

end NUMINAMATH_CALUDE_fill_jug_completely_l1871_187153


namespace NUMINAMATH_CALUDE_inequality_range_l1871_187169

theorem inequality_range (m : ℝ) : 
  (∀ x y : ℝ, 3 * x^2 + y^2 ≥ m * x * (x + y)) → 
  m ∈ Set.Icc (-6) 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l1871_187169


namespace NUMINAMATH_CALUDE_combined_years_is_75_l1871_187114

/-- The combined total of years taught by Virginia, Adrienne, and Dennis -/
def combinedYears (adrienneYears virginiaYears dennisYears : ℕ) : ℕ :=
  adrienneYears + virginiaYears + dennisYears

/-- Theorem stating the combined total of years taught is 75 -/
theorem combined_years_is_75 :
  ∀ (adrienneYears virginiaYears dennisYears : ℕ),
    virginiaYears = adrienneYears + 9 →
    virginiaYears = dennisYears - 9 →
    dennisYears = 34 →
    combinedYears adrienneYears virginiaYears dennisYears = 75 := by
  sorry


end NUMINAMATH_CALUDE_combined_years_is_75_l1871_187114


namespace NUMINAMATH_CALUDE_root_sum_fraction_values_l1871_187145

theorem root_sum_fraction_values (α β γ : ℝ) : 
  (α^3 - α^2 - 2*α + 1 = 0) →
  (β^3 - β^2 - 2*β + 1 = 0) →
  (γ^3 - γ^2 - 2*γ + 1 = 0) →
  (α ≠ 0 ∧ β ≠ 0 ∧ γ ≠ 0) →
  (α/β + β/γ + γ/α = 3 ∨ α/β + β/γ + γ/α = -4) := by
sorry

end NUMINAMATH_CALUDE_root_sum_fraction_values_l1871_187145


namespace NUMINAMATH_CALUDE_square_areas_sum_l1871_187100

theorem square_areas_sum (a b c : ℕ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 5) :
  a^2 + b^2 = c^2 :=
by sorry

end NUMINAMATH_CALUDE_square_areas_sum_l1871_187100


namespace NUMINAMATH_CALUDE_major_premise_identification_l1871_187149

theorem major_premise_identification (α : ℝ) (m : ℝ) 
  (h1 : ∀ x : ℝ, |Real.sin x| ≤ 1)
  (h2 : m = Real.sin α)
  (h3 : |m| ≤ 1) :
  (∀ x : ℝ, |Real.sin x| ≤ 1) = (|Real.sin x| ≤ 1) := by
sorry

end NUMINAMATH_CALUDE_major_premise_identification_l1871_187149


namespace NUMINAMATH_CALUDE_root_difference_quadratic_l1871_187165

theorem root_difference_quadratic (p : ℝ) : 
  let r := (2*p + Real.sqrt (9 : ℝ))
  let s := (2*p - Real.sqrt (9 : ℝ))
  r - s = 6 := by
sorry

end NUMINAMATH_CALUDE_root_difference_quadratic_l1871_187165


namespace NUMINAMATH_CALUDE_second_column_halving_matrix_l1871_187199

def halve_second_column (N : Matrix (Fin 2) (Fin 2) ℝ) (M : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  ∀ i j, (N * M) i j = if j = 1 then (1/2 : ℝ) * M i j else M i j

theorem second_column_halving_matrix :
  ∃ N : Matrix (Fin 2) (Fin 2) ℝ, 
    (N 0 0 = 1 ∧ N 0 1 = 0 ∧ N 1 0 = 0 ∧ N 1 1 = 1/2) ∧
    ∀ M : Matrix (Fin 2) (Fin 2) ℝ, halve_second_column N M :=
by
  sorry

end NUMINAMATH_CALUDE_second_column_halving_matrix_l1871_187199


namespace NUMINAMATH_CALUDE_roberto_salary_l1871_187135

/-- Calculates the final salary after two consecutive percentage increases -/
def final_salary (starting_salary : ℝ) (first_increase : ℝ) (second_increase : ℝ) : ℝ :=
  starting_salary * (1 + first_increase) * (1 + second_increase)

/-- Theorem: Roberto's final salary calculation -/
theorem roberto_salary : 
  final_salary 80000 0.4 0.2 = 134400 := by
  sorry

#eval final_salary 80000 0.4 0.2

end NUMINAMATH_CALUDE_roberto_salary_l1871_187135


namespace NUMINAMATH_CALUDE_min_omega_value_l1871_187125

open Real

theorem min_omega_value (ω φ : ℝ) (f : ℝ → ℝ) : 
  ω > 0 → 
  abs φ < π / 2 →
  (∀ x, f x = sin (ω * x + φ)) →
  f 0 = 1 / 2 →
  (∀ x, f x ≤ f (π / 12)) →
  (∀ ω' > 0, (∀ x, sin (ω' * x + φ) ≤ sin (ω' * π / 12 + φ)) → ω' ≥ ω) →
  ω = 4 := by
sorry

end NUMINAMATH_CALUDE_min_omega_value_l1871_187125


namespace NUMINAMATH_CALUDE_largest_multiple_under_500_l1871_187117

theorem largest_multiple_under_500 : 
  ∀ n : ℕ, n > 0 ∧ 15 ∣ n ∧ n < 500 → n ≤ 495 :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_under_500_l1871_187117


namespace NUMINAMATH_CALUDE_inequality_and_equality_conditions_l1871_187104

theorem inequality_and_equality_conditions (x y : ℝ) (h : (x + 1) * (y + 2) = 8) :
  (x * y - 10)^2 ≥ 64 ∧ 
  (∃ (a b : ℝ), (a * b - 10)^2 = 64 ∧ (a + 1) * (b + 2) = 8) ∧
  (∀ (a b : ℝ), (a * b - 10)^2 = 64 ∧ (a + 1) * (b + 2) = 8 → (a = 1 ∧ b = 2) ∨ (a = -3 ∧ b = -6)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_conditions_l1871_187104


namespace NUMINAMATH_CALUDE_x_value_l1871_187158

theorem x_value : ∃ x : ℝ, (2*x - 3*x + 5*x - x = 120) ∧ (x = 40) := by sorry

end NUMINAMATH_CALUDE_x_value_l1871_187158


namespace NUMINAMATH_CALUDE_tiling_ways_2x12_l1871_187180

/-- The number of ways to tile a 2 × n rectangle with 1 × 2 dominoes -/
def tiling_ways : ℕ → ℕ
  | 0 => 0  -- Added for completeness
  | 1 => 1
  | 2 => 2
  | n+3 => tiling_ways (n+2) + tiling_ways (n+1)

/-- Theorem: The number of ways to tile a 2 × 12 rectangle with 1 × 2 dominoes is 233 -/
theorem tiling_ways_2x12 : tiling_ways 12 = 233 := by
  sorry


end NUMINAMATH_CALUDE_tiling_ways_2x12_l1871_187180


namespace NUMINAMATH_CALUDE_segment_polynomial_l1871_187164

/-- Given a line segment AB with point T, prove that x^2 - 6√2x + 16 has roots equal to AT and TB lengths -/
theorem segment_polynomial (AB : ℝ) (T : ℝ) (h1 : 0 < T ∧ T < AB) 
  (h2 : AB - T = (1/2) * T) (h3 : T * (AB - T) = 16) :
  ∃ (AT TB : ℝ), AT = T ∧ TB = AB - T ∧ 
  (∀ x : ℝ, x^2 - 6 * Real.sqrt 2 * x + 16 = 0 ↔ (x = AT ∨ x = TB)) :=
sorry

end NUMINAMATH_CALUDE_segment_polynomial_l1871_187164


namespace NUMINAMATH_CALUDE_discounted_soda_price_70_cans_l1871_187146

/-- Calculate the price of discounted soda cans -/
def discounted_soda_price (regular_price : ℚ) (discount_percent : ℚ) (num_cans : ℕ) : ℚ :=
  let discounted_price := regular_price * (1 - discount_percent)
  let full_cases := num_cans / 24
  let remaining_cans := num_cans % 24
  discounted_price * (↑full_cases * 24 + ↑remaining_cans)

/-- The price of 70 cans of soda with a regular price of $0.55 and 25% discount in 24-can cases is $28.875 -/
theorem discounted_soda_price_70_cans :
  discounted_soda_price (55/100) (25/100) 70 = 28875/1000 :=
sorry

end NUMINAMATH_CALUDE_discounted_soda_price_70_cans_l1871_187146


namespace NUMINAMATH_CALUDE_kingdom_animal_percentage_l1871_187139

/-- Represents the number of cats in the kingdom -/
def num_cats : ℕ := 25

/-- Represents the number of hogs in the kingdom -/
def num_hogs : ℕ := 75

/-- The relationship between hogs and cats -/
axiom hogs_cats_relation : num_hogs = 3 * num_cats

/-- The percentage we're looking for -/
def percentage : ℚ := 50

theorem kingdom_animal_percentage :
  (percentage / 100) * (num_cats - 5 : ℚ) = 10 :=
sorry

end NUMINAMATH_CALUDE_kingdom_animal_percentage_l1871_187139


namespace NUMINAMATH_CALUDE_min_value_of_x2_plus_y2_l1871_187107

theorem min_value_of_x2_plus_y2 (x y : ℝ) : 
  (x - 1)^2 + y^2 = 16 → ∃ (m : ℝ), (∀ (a b : ℝ), (a - 1)^2 + b^2 = 16 → x^2 + y^2 ≤ a^2 + b^2) ∧ m = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_x2_plus_y2_l1871_187107


namespace NUMINAMATH_CALUDE_total_cost_of_materials_l1871_187176

/-- The total cost of materials for a construction company -/
theorem total_cost_of_materials
  (gravel_quantity : ℝ)
  (gravel_price : ℝ)
  (sand_quantity : ℝ)
  (sand_price : ℝ)
  (h1 : gravel_quantity = 5.91)
  (h2 : gravel_price = 30.50)
  (h3 : sand_quantity = 8.11)
  (h4 : sand_price = 40.50) :
  gravel_quantity * gravel_price + sand_quantity * sand_price = 508.71 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_of_materials_l1871_187176


namespace NUMINAMATH_CALUDE_leadership_selection_l1871_187119

/-- The number of ways to choose a president, vice-president, and committee from a group --/
def choose_leadership (total : ℕ) (committee_size : ℕ) : ℕ :=
  total * (total - 1) * (Nat.choose (total - 2) committee_size)

/-- The problem statement --/
theorem leadership_selection :
  choose_leadership 10 3 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_leadership_selection_l1871_187119


namespace NUMINAMATH_CALUDE_impossible_circle_assignment_l1871_187128

-- Define the type for circles
def Circle := Fin 6

-- Define the connection relation between circles
def connected : Circle → Circle → Prop := sorry

-- Define the divisibility relation
def divides (a b : ℕ) : Prop := ∃ k, b = a * k

-- Main theorem
theorem impossible_circle_assignment :
  ¬ ∃ (f : Circle → ℕ),
    (∀ i j : Circle, connected i j → (divides (f i) (f j) ∨ divides (f j) (f i))) ∧
    (∀ i j : Circle, ¬ connected i j → ¬ divides (f i) (f j) ∧ ¬ divides (f j) (f i)) :=
by sorry


end NUMINAMATH_CALUDE_impossible_circle_assignment_l1871_187128


namespace NUMINAMATH_CALUDE_marble_distribution_l1871_187121

theorem marble_distribution (n : ℕ) (hn : n = 450) :
  (Finset.filter (fun m : ℕ => m > 1 ∧ n / m > 1) (Finset.range (n + 1))).card = 16 := by
  sorry

end NUMINAMATH_CALUDE_marble_distribution_l1871_187121


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1871_187171

/-- Given a geometric sequence {aₙ} where aₙ ∈ ℝ, if a₃ and a₁₁ are the two roots of the equation 3x² - 25x + 27 = 0, then a₇ = 3. -/
theorem geometric_sequence_problem (a : ℕ → ℝ) (h_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_roots : 3 * (a 3)^2 - 25 * (a 3) + 27 = 0 ∧ 3 * (a 11)^2 - 25 * (a 11) + 27 = 0) :
  a 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1871_187171


namespace NUMINAMATH_CALUDE_no_prime_root_sum_29_l1871_187147

/-- A quadratic equation x^2 - 29x + k = 0 with prime roots -/
def has_prime_roots (k : ℤ) : Prop :=
  ∃ p q : ℕ, 
    Nat.Prime p ∧ 
    Nat.Prime q ∧ 
    (p : ℤ) + (q : ℤ) = 29 ∧ 
    (p : ℤ) * (q : ℤ) = k

/-- There are no integer values of k such that x^2 - 29x + k = 0 has two prime roots -/
theorem no_prime_root_sum_29 : ¬∃ k : ℤ, has_prime_roots k := by
  sorry

end NUMINAMATH_CALUDE_no_prime_root_sum_29_l1871_187147


namespace NUMINAMATH_CALUDE_jack_afternoon_emails_l1871_187123

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 4

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := 8

/-- The total number of emails Jack received in the afternoon and evening -/
def afternoon_evening_emails : ℕ := 13

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := afternoon_evening_emails - evening_emails

theorem jack_afternoon_emails :
  afternoon_emails = 5 := by sorry

end NUMINAMATH_CALUDE_jack_afternoon_emails_l1871_187123


namespace NUMINAMATH_CALUDE_runners_meeting_time_l1871_187134

/-- Represents a runner with their start time (in minutes after 7:00 AM) and lap duration -/
structure Runner where
  startTime : ℕ
  lapDuration : ℕ

/-- The earliest time (in minutes after 7:00 AM) when all runners meet at the starting point -/
def earliestMeetingTime (runners : List Runner) : ℕ :=
  sorry

/-- The problem statement -/
theorem runners_meeting_time :
  let kevin := Runner.mk 45 5
  let laura := Runner.mk 50 8
  let neil := Runner.mk 55 10
  let runners := [kevin, laura, neil]
  earliestMeetingTime runners = 95
  := by sorry

end NUMINAMATH_CALUDE_runners_meeting_time_l1871_187134


namespace NUMINAMATH_CALUDE_kyle_earnings_theorem_l1871_187151

/-- Calculates the money Kyle makes from selling his remaining baked goods --/
def kyle_earnings (initial_cookies : ℕ) (initial_brownies : ℕ) 
                  (kyle_cookies_eaten : ℕ) (kyle_brownies_eaten : ℕ)
                  (mom_cookies_eaten : ℕ) (mom_brownies_eaten : ℕ)
                  (cookie_price : ℚ) (brownie_price : ℚ) : ℚ :=
  let remaining_cookies := initial_cookies - (kyle_cookies_eaten + mom_cookies_eaten)
  let remaining_brownies := initial_brownies - (kyle_brownies_eaten + mom_brownies_eaten)
  (remaining_cookies : ℚ) * cookie_price + (remaining_brownies : ℚ) * brownie_price

/-- Theorem stating Kyle's earnings from selling all remaining baked goods --/
theorem kyle_earnings_theorem : 
  kyle_earnings 60 32 2 2 1 2 1 (3/2) = 99 := by
  sorry

end NUMINAMATH_CALUDE_kyle_earnings_theorem_l1871_187151


namespace NUMINAMATH_CALUDE_inequality_proof_l1871_187108

theorem inequality_proof (a b c d e p q : ℝ) 
  (hp_pos : 0 < p) 
  (hq_geq_p : p ≤ q)
  (ha : p ≤ a ∧ a ≤ q)
  (hb : p ≤ b ∧ b ≤ q)
  (hc : p ≤ c ∧ c ≤ q)
  (hd : p ≤ d ∧ d ≤ q)
  (he : p ≤ e ∧ e ≤ q) :
  (a + b + c + d + e) * (1/a + 1/b + 1/c + 1/d + 1/e) ≤ 25 + 6 * (Real.sqrt (p/q) - Real.sqrt (q/p))^2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1871_187108


namespace NUMINAMATH_CALUDE_red_ball_packs_l1871_187129

theorem red_ball_packs (total_balls : ℕ) (yellow_packs green_packs : ℕ) (balls_per_pack : ℕ) :
  total_balls = 399 →
  yellow_packs = 10 →
  green_packs = 8 →
  balls_per_pack = 19 →
  ∃ red_packs : ℕ, red_packs = 3 ∧ 
    total_balls = (red_packs + yellow_packs + green_packs) * balls_per_pack :=
by sorry

end NUMINAMATH_CALUDE_red_ball_packs_l1871_187129


namespace NUMINAMATH_CALUDE_complementary_angles_adjustment_l1871_187188

-- Define the ratio of the two complementary angles
def angle_ratio : ℚ := 3 / 7

-- Define the increase percentage for the smaller angle
def small_angle_increase : ℚ := 1 / 5

-- Function to calculate the decrease percentage for the larger angle
def large_angle_decrease (ratio : ℚ) (increase : ℚ) : ℚ :=
  1 - (90 - 90 * ratio / (1 + ratio) * (1 + increase)) / (90 * ratio / (1 + ratio))

-- Theorem statement
theorem complementary_angles_adjustment :
  large_angle_decrease angle_ratio small_angle_increase = 43 / 500 := by
  sorry

#eval large_angle_decrease angle_ratio small_angle_increase

end NUMINAMATH_CALUDE_complementary_angles_adjustment_l1871_187188


namespace NUMINAMATH_CALUDE_problem_solution_l1871_187193

theorem problem_solution (x : ℝ) (h : x - 1/x = 5) : x^2 - 1/x^2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1871_187193


namespace NUMINAMATH_CALUDE_non_negative_sequence_l1871_187113

theorem non_negative_sequence (a : Fin 100 → ℝ) 
  (h1 : ∀ i : Fin 98, a i - 2 * a (i + 1) + a (i + 2) ≤ 0)
  (h2 : a 0 = a 99)
  (h3 : a 0 ≥ 0) : 
  ∀ i : Fin 100, a i ≥ 0 := by
sorry

end NUMINAMATH_CALUDE_non_negative_sequence_l1871_187113


namespace NUMINAMATH_CALUDE_applicants_age_standard_deviation_l1871_187198

/-- The standard deviation of applicants' ages given specific conditions -/
theorem applicants_age_standard_deviation 
  (average_age : ℝ)
  (max_different_ages : ℕ)
  (h_average : average_age = 30)
  (h_max_ages : max_different_ages = 15)
  (h_range : max_different_ages = 2 * standard_deviation)
  (standard_deviation : ℝ) :
  standard_deviation = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_applicants_age_standard_deviation_l1871_187198


namespace NUMINAMATH_CALUDE_moon_permutations_l1871_187124

def word_length : ℕ := 4
def repeated_letter_count : ℕ := 2

theorem moon_permutations :
  (word_length.factorial) / (repeated_letter_count.factorial) = 12 := by
  sorry

end NUMINAMATH_CALUDE_moon_permutations_l1871_187124


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1871_187101

def A : Set ℕ := {1, 3, 5, 7, 9}
def B : Set ℕ := {0, 3, 6, 9, 12}

theorem intersection_of_A_and_B : A ∩ B = {3, 9} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1871_187101


namespace NUMINAMATH_CALUDE_weight_loss_challenge_l1871_187130

theorem weight_loss_challenge (initial_weight : ℝ) (x : ℝ) : 
  x > 0 →
  (initial_weight * (1 - x / 100 + 2 / 100)) / initial_weight = 1 - 11.26 / 100 →
  x = 13.26 :=
by sorry

end NUMINAMATH_CALUDE_weight_loss_challenge_l1871_187130


namespace NUMINAMATH_CALUDE_first_group_size_l1871_187168

/-- The number of persons in the first group -/
def P : ℕ := 42

/-- The number of days the first group works -/
def days_first : ℕ := 12

/-- The number of hours per day the first group works -/
def hours_first : ℕ := 5

/-- The number of persons in the second group -/
def persons_second : ℕ := 30

/-- The number of days the second group works -/
def days_second : ℕ := 14

/-- The number of hours per day the second group works -/
def hours_second : ℕ := 6

/-- Theorem stating that P is the correct number of persons in the first group -/
theorem first_group_size :
  P * days_first * hours_first = persons_second * days_second * hours_second :=
by sorry

end NUMINAMATH_CALUDE_first_group_size_l1871_187168


namespace NUMINAMATH_CALUDE_integer_part_sqrt_seven_l1871_187157

theorem integer_part_sqrt_seven : ⌊Real.sqrt 7⌋ = 2 := by sorry

end NUMINAMATH_CALUDE_integer_part_sqrt_seven_l1871_187157


namespace NUMINAMATH_CALUDE_tan_sum_angle_l1871_187116

theorem tan_sum_angle (α β : Real) : 
  α ∈ Set.Ioo 0 (Real.pi / 2) →
  β ∈ Set.Ioo 0 (Real.pi / 2) →
  2 * Real.tan α = Real.sin (2 * β) / (Real.sin β + Real.sin β ^ 2) →
  Real.tan (2 * α + β + Real.pi / 3) = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_angle_l1871_187116


namespace NUMINAMATH_CALUDE_mason_father_age_l1871_187140

/-- Mason's age -/
def mason_age : ℕ := 20

/-- Sydney's age -/
def sydney_age : ℕ := mason_age + 6

/-- Mason's father's age -/
def father_age : ℕ := sydney_age + 6

theorem mason_father_age : father_age = 26 := by
  sorry

end NUMINAMATH_CALUDE_mason_father_age_l1871_187140


namespace NUMINAMATH_CALUDE_multiply_by_one_seventh_squared_l1871_187177

theorem multiply_by_one_seventh_squared (x : ℝ) : x * (1/7)^2 = 7^3 ↔ x = 16807 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_one_seventh_squared_l1871_187177


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l1871_187186

theorem simplify_trig_expression (α β : ℝ) : 
  Real.sqrt ((1 - Real.sin α * Real.sin β)^2 - (Real.cos α * Real.cos β)^2) = 
  |Real.sin α - Real.sin β| := by
sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l1871_187186


namespace NUMINAMATH_CALUDE_average_of_distinct_t_is_22_3_l1871_187179

/-- Given a polynomial x^2 - 6x + t with only positive integer roots,
    this function returns the average of all distinct possible values of t. -/
def averageOfDistinctT : ℚ :=
  22 / 3

/-- The polynomial x^2 - 6x + t has only positive integer roots. -/
axiom has_positive_integer_roots (t : ℤ) : 
  ∃ (r₁ r₂ : ℕ+), r₁.val * r₁.val - 6 * r₁.val + t = 0 ∧ 
                  r₂.val * r₂.val - 6 * r₂.val + t = 0

/-- The main theorem stating that the average of all distinct possible values of t
    for the polynomial x^2 - 6x + t with only positive integer roots is 22/3. -/
theorem average_of_distinct_t_is_22_3 :
  averageOfDistinctT = 22 / 3 :=
sorry

end NUMINAMATH_CALUDE_average_of_distinct_t_is_22_3_l1871_187179


namespace NUMINAMATH_CALUDE_privateer_overtakes_at_1730_l1871_187144

/-- Represents the chase between a privateer and a merchantman -/
structure ChaseScenario where
  initial_distance : ℝ
  initial_time : ℕ  -- represented in minutes since midnight
  initial_privateer_speed : ℝ
  initial_merchantman_speed : ℝ
  initial_chase_duration : ℝ
  new_speed_ratio : ℚ

/-- Calculates the time when the privateer overtakes the merchantman -/
def overtake_time (scenario : ChaseScenario) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem privateer_overtakes_at_1730 : 
  let scenario : ChaseScenario := {
    initial_distance := 10,
    initial_time := 11 * 60 + 45,  -- 11:45 a.m. in minutes
    initial_privateer_speed := 11,
    initial_merchantman_speed := 8,
    initial_chase_duration := 2,
    new_speed_ratio := 17 / 15
  }
  overtake_time scenario = 17 * 60 + 30  -- 5:30 p.m. in minutes
:= by sorry


end NUMINAMATH_CALUDE_privateer_overtakes_at_1730_l1871_187144


namespace NUMINAMATH_CALUDE_medical_team_selection_l1871_187170

theorem medical_team_selection (male_doctors female_doctors team_size : ℕ) 
  (h1 : male_doctors = 6)
  (h2 : female_doctors = 3)
  (h3 : team_size = 5) :
  (Nat.choose (male_doctors + female_doctors) team_size) - 
  (Nat.choose male_doctors team_size) = 120 := by
  sorry

end NUMINAMATH_CALUDE_medical_team_selection_l1871_187170


namespace NUMINAMATH_CALUDE_largest_B_for_divisibility_by_4_l1871_187110

def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

def is_single_digit (n : ℕ) : Prop := n ≤ 9

def seven_digit_number (B X : ℕ) : ℕ := 4000000 + 100000 * B + 6000 + 792 * 10 + X

theorem largest_B_for_divisibility_by_4 :
  ∃ (B : ℕ), is_single_digit B ∧
  (∃ (X : ℕ), is_single_digit X ∧ is_divisible_by_4 (seven_digit_number B X)) ∧
  ∀ (B' : ℕ), is_single_digit B' →
    (∃ (X : ℕ), is_single_digit X ∧ is_divisible_by_4 (seven_digit_number B' X)) →
    B' ≤ B :=
by sorry

end NUMINAMATH_CALUDE_largest_B_for_divisibility_by_4_l1871_187110


namespace NUMINAMATH_CALUDE_decreasing_function_on_positive_reals_l1871_187191

/-- The function f(x) = 9 - x² is decreasing on the interval (0, +∞) -/
theorem decreasing_function_on_positive_reals :
  ∀ x y : ℝ, 0 < x → 0 < y → x < y → (9 - x^2 : ℝ) > (9 - y^2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_decreasing_function_on_positive_reals_l1871_187191


namespace NUMINAMATH_CALUDE_sum_transformation_l1871_187195

theorem sum_transformation (xs : List ℝ) 
  (h1 : xs.sum = 40)
  (h2 : (xs.map (λ x => 1 - x)).sum = 20) :
  (xs.map (λ x => 1 + x)).sum = 100 := by
  sorry

end NUMINAMATH_CALUDE_sum_transformation_l1871_187195


namespace NUMINAMATH_CALUDE_problem_solution_l1871_187150

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 1

-- Define the theorem
theorem problem_solution :
  -- Part 1
  (∀ x ∈ Set.Icc 1 3, g 1 x ∈ Set.Icc 0 4) ∧
  (∀ a : ℝ, (∀ x ∈ Set.Icc 1 3, g a x ∈ Set.Icc 0 4) → a = 1) ∧
  
  -- Part 2
  (∀ k : ℝ, (∀ x : ℝ, x ≥ 1 → g 1 (2^x) - k * 4^x ≥ 0) ↔ k ≤ 1/4) ∧
  
  -- Part 3
  (∀ k : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    (g 1 (|2^x₁ - 1|) / |2^x₁ - 1| + k * (2 / |2^x₁ - 1|) - 3*k = 0) ∧
    (g 1 (|2^x₂ - 1|) / |2^x₂ - 1| + k * (2 / |2^x₂ - 1|) - 3*k = 0) ∧
    (g 1 (|2^x₃ - 1|) / |2^x₃ - 1| + k * (2 / |2^x₃ - 1|) - 3*k = 0)) ↔
   k > 0) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1871_187150


namespace NUMINAMATH_CALUDE_inequalities_hold_l1871_187178

theorem inequalities_hold (a b : ℝ) (h : a > b) :
  (∀ c : ℝ, c ≠ 0 → a / c^2 > b / c^2) ∧
  (∀ c : ℝ, a * |c| ≥ b * |c|) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_hold_l1871_187178


namespace NUMINAMATH_CALUDE_linear_equation_exponent_l1871_187115

/-- If x^(m+1) - 2 = 1 is a linear equation with respect to x, then m = 0 -/
theorem linear_equation_exponent (m : ℕ) : 
  (∀ x, ∃ a b : ℝ, x^(m+1) - 2 = a*x + b) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_exponent_l1871_187115


namespace NUMINAMATH_CALUDE_calculate_second_discount_other_discount_percentage_l1871_187190

/-- Given an article with a list price and two successive discounts, 
    calculate the second discount percentage. -/
theorem calculate_second_discount 
  (list_price : ℝ) 
  (final_price : ℝ) 
  (first_discount : ℝ) : ℝ :=
  let price_after_first_discount := list_price * (1 - first_discount / 100)
  let second_discount := (price_after_first_discount - final_price) / price_after_first_discount * 100
  second_discount

/-- Prove that for an article with a list price of 70 units, 
    after applying two successive discounts, one of which is 10%, 
    resulting in a final price of 56.16 units, 
    the other discount percentage is approximately 10.857%. -/
theorem other_discount_percentage : 
  let result := calculate_second_discount 70 56.16 10
  ∃ ε > 0, abs (result - 10.857) < ε :=
sorry

end NUMINAMATH_CALUDE_calculate_second_discount_other_discount_percentage_l1871_187190


namespace NUMINAMATH_CALUDE_min_value_of_max_absolute_min_value_achievable_l1871_187183

theorem min_value_of_max_absolute (a b : ℝ) : 
  max (max (|a + b|) (|a - b|)) (|1 - b|) ≥ (1/2 : ℝ) := by sorry

theorem min_value_achievable : 
  ∃ (a b : ℝ), max (max (|a + b|) (|a - b|)) (|1 - b|) = (1/2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_min_value_of_max_absolute_min_value_achievable_l1871_187183


namespace NUMINAMATH_CALUDE_remainder_problem_l1871_187141

theorem remainder_problem (n : ℤ) (h : n % 7 = 2) : (5 * n + 9) % 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1871_187141


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1871_187167

theorem arithmetic_calculation : ((55 * 45 - 37 * 43) - (3 * 221 + 1)) / 22 = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1871_187167


namespace NUMINAMATH_CALUDE_chocolate_difference_l1871_187136

theorem chocolate_difference (robert_chocolates nickel_chocolates : ℕ) 
  (h1 : robert_chocolates = 12) 
  (h2 : nickel_chocolates = 3) : 
  robert_chocolates - nickel_chocolates = 9 := by
sorry

end NUMINAMATH_CALUDE_chocolate_difference_l1871_187136


namespace NUMINAMATH_CALUDE_remaining_drawings_l1871_187161

-- Define the given parameters
def total_markers : ℕ := 12
def drawings_per_marker : ℚ := 3/2
def drawings_already_made : ℕ := 8

-- State the theorem
theorem remaining_drawings : 
  ⌊(total_markers : ℚ) * drawings_per_marker⌋ - drawings_already_made = 10 := by
  sorry

end NUMINAMATH_CALUDE_remaining_drawings_l1871_187161


namespace NUMINAMATH_CALUDE_minimum_percentage_owning_95_percent_l1871_187159

/-- Represents the distribution of wealth in a population -/
structure WealthDistribution where
  totalPeople : ℝ
  totalWealth : ℝ
  wealthFunction : ℝ → ℝ
  -- wealthFunction x represents the amount of wealth owned by the top x fraction of people
  wealthMonotone : ∀ x y, 0 ≤ x → x ≤ y → y ≤ 1 → wealthFunction x ≤ wealthFunction y
  wealthBounds : wealthFunction 0 = 0 ∧ wealthFunction 1 = totalWealth

/-- The theorem stating the minimum percentage of people owning 95% of wealth -/
theorem minimum_percentage_owning_95_percent
  (dist : WealthDistribution)
  (h_10_percent : dist.wealthFunction 0.1 ≥ 0.9 * dist.totalWealth) :
  ∃ x : ℝ, x ≤ 0.55 ∧ dist.wealthFunction x ≥ 0.95 * dist.totalWealth := by
  sorry


end NUMINAMATH_CALUDE_minimum_percentage_owning_95_percent_l1871_187159


namespace NUMINAMATH_CALUDE_max_defective_items_l1871_187133

theorem max_defective_items 
  (N M n : ℕ) 
  (h1 : M ≤ N) 
  (h2 : n ≤ N) : 
  ∃ X : ℕ, X ≤ min M n ∧ 
  ∀ Y : ℕ, Y ≤ M ∧ Y ≤ n → Y ≤ X :=
sorry

end NUMINAMATH_CALUDE_max_defective_items_l1871_187133


namespace NUMINAMATH_CALUDE_units_digit_47_pow_25_l1871_187127

theorem units_digit_47_pow_25 : ∃ n : ℕ, 47^25 ≡ 7 [ZMOD 10] :=
sorry

end NUMINAMATH_CALUDE_units_digit_47_pow_25_l1871_187127


namespace NUMINAMATH_CALUDE_blue_highlighters_count_l1871_187182

def total_highlighters : ℕ := 33
def pink_highlighters : ℕ := 10
def yellow_highlighters : ℕ := 15

theorem blue_highlighters_count :
  total_highlighters - (pink_highlighters + yellow_highlighters) = 8 :=
by sorry

end NUMINAMATH_CALUDE_blue_highlighters_count_l1871_187182


namespace NUMINAMATH_CALUDE_parabola_equation_part1_parabola_equation_part2_l1871_187175

-- Part 1
theorem parabola_equation_part1 (a b c : ℝ) (h : a ≠ 0) :
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →
  (1, 10) = (- b / (2 * a), a * (- b / (2 * a))^2 + b * (- b / (2 * a)) + c) →
  (-1, -2) = (-1, a * (-1)^2 + b * (-1) + c) →
  (∀ x y : ℝ, y = -3 * (x - 1)^2 + 10) := by sorry

-- Part 2
theorem parabola_equation_part2 (a b c : ℝ) (h : a ≠ 0) :
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →
  (0 = a * (-1)^2 + b * (-1) + c) →
  (0 = a * 3^2 + b * 3 + c) →
  (3 = c) →
  (∀ x y : ℝ, y = -x^2 + 2 * x + 3) := by sorry

end NUMINAMATH_CALUDE_parabola_equation_part1_parabola_equation_part2_l1871_187175


namespace NUMINAMATH_CALUDE_water_fraction_in_first_container_l1871_187154

/-- Represents the amount of liquid in a container -/
structure Container where
  juice : ℚ
  water : ℚ

/-- The problem setup and operations -/
def liquidTransfer : Prop :=
  ∃ (initial1 initial2 after_first_transfer after_second_transfer final1 : Container),
    -- Initial setup
    initial1.juice = 5 ∧ initial1.water = 0 ∧
    initial2.juice = 0 ∧ initial2.water = 5 ∧
    
    -- First transfer (1/3 of juice from container 1 to 2)
    after_first_transfer.juice = initial1.juice - (initial1.juice / 3) ∧
    after_first_transfer.water = initial1.water ∧
    
    -- Second transfer (1/4 of mixture from container 2 back to 1)
    final1.juice = after_first_transfer.juice + 
      ((initial2.water + (initial1.juice / 3)) / 4) * ((initial1.juice / 3) / (initial2.water + (initial1.juice / 3))) ∧
    final1.water = ((initial2.water + (initial1.juice / 3)) / 4) * (initial2.water / (initial2.water + (initial1.juice / 3))) ∧
    
    -- Final result
    final1.water / (final1.juice + final1.water) = 3 / 13

/-- The main theorem to prove -/
theorem water_fraction_in_first_container : liquidTransfer := by
  sorry

end NUMINAMATH_CALUDE_water_fraction_in_first_container_l1871_187154


namespace NUMINAMATH_CALUDE_min_product_of_reciprocal_sum_l1871_187160

theorem min_product_of_reciprocal_sum (x y : ℕ+) : 
  (1 : ℚ) / x + 1 / (3 * y) = 1 / 9 → 
  ∃ (a b : ℕ+), (1 : ℚ) / a + 1 / (3 * b) = 1 / 9 ∧ a * b = 108 ∧ 
  ∀ (c d : ℕ+), (1 : ℚ) / c + 1 / (3 * d) = 1 / 9 → c * d ≥ 108 := by
sorry

end NUMINAMATH_CALUDE_min_product_of_reciprocal_sum_l1871_187160


namespace NUMINAMATH_CALUDE_sum_f_equals_1326_l1871_187137

/-- A lattice point is a point with integer coordinates -/
def is_lattice_point (p : ℤ × ℤ) : Prop := True

/-- f(n) is the number of lattice points on the segment from (0,0) to (n, n+3), excluding endpoints -/
def f (n : ℕ) : ℕ := 
  if n % 3 = 0 then 2 else 0

/-- The sum of f(n) from 1 to 1990 -/
def sum_f : ℕ := (Finset.range 1990).sum f

theorem sum_f_equals_1326 : sum_f = 1326 := by
  sorry

end NUMINAMATH_CALUDE_sum_f_equals_1326_l1871_187137


namespace NUMINAMATH_CALUDE_vertical_angles_are_congruent_l1871_187163

/-- Two angles are vertical if they are formed by two intersecting lines and are not adjacent. -/
def are_vertical_angles (α β : Angle) : Prop := sorry

/-- Two angles are congruent if they have the same measure. -/
def are_congruent (α β : Angle) : Prop := sorry

/-- If two angles are vertical angles, then these two angles are congruent. -/
theorem vertical_angles_are_congruent (α β : Angle) : 
  are_vertical_angles α β → are_congruent α β := by sorry

end NUMINAMATH_CALUDE_vertical_angles_are_congruent_l1871_187163


namespace NUMINAMATH_CALUDE_solution_form_l1871_187194

theorem solution_form (a b c d : ℝ) 
  (h1 : a + b + c = d) 
  (h2 : 1/a + 1/b + 1/c = 1/d) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) : 
  (c = -a ∧ d = b) ∨ (c = -b ∧ d = a) := by
sorry

end NUMINAMATH_CALUDE_solution_form_l1871_187194


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l1871_187103

theorem inverse_variation_problem (x y : ℝ) (k : ℝ) (h1 : x = k / (y ^ 2)) 
  (h2 : 1 = k / (2 ^ 2)) (h3 : 0.1111111111111111 = k / (y ^ 2)) : y = 6 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l1871_187103


namespace NUMINAMATH_CALUDE_john_jane_difference_l1871_187152

/-- The width of the streets in Perfectville -/
def street_width : ℕ := 30

/-- The side length of a block in Perfectville -/
def block_side_length : ℕ := 500

/-- The side length of John's path -/
def john_path_side : ℕ := block_side_length + 2 * street_width

/-- The perimeter of Jane's path -/
def jane_perimeter : ℕ := 4 * block_side_length

/-- The perimeter of John's path -/
def john_perimeter : ℕ := 4 * john_path_side

theorem john_jane_difference :
  john_perimeter - jane_perimeter = 240 := by sorry

end NUMINAMATH_CALUDE_john_jane_difference_l1871_187152


namespace NUMINAMATH_CALUDE_dog_journey_distance_l1871_187162

/-- 
Given a journey where:
- The total time is 2 hours
- The first half of the distance is traveled at 10 km/h
- The second half of the distance is traveled at 5 km/h
Prove that the total distance traveled is 40/3 km
-/
theorem dog_journey_distance : 
  ∀ (total_distance : ℝ),
  (total_distance / 20 + total_distance / 10 = 2) →
  total_distance = 40 / 3 := by
  sorry

end NUMINAMATH_CALUDE_dog_journey_distance_l1871_187162


namespace NUMINAMATH_CALUDE_truck_capacity_l1871_187106

theorem truck_capacity (x y : ℝ) 
  (h1 : 2 * x + 3 * y = 15.5) 
  (h2 : 5 * x + 6 * y = 35) : 
  3 * x + 5 * y = 24.5 := by
sorry

end NUMINAMATH_CALUDE_truck_capacity_l1871_187106


namespace NUMINAMATH_CALUDE_ellipse_properties_and_max_area_l1871_187166

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  ecc : ℝ
  h_a_pos : a > 0
  h_b_pos : b > 0
  h_a_gt_b : a > b
  h_ecc : ecc = 2 * Real.sqrt 2 / 3
  h_vertex : b = 1

/-- A line intersecting the ellipse -/
structure IntersectingLine (E : Ellipse) where
  k : ℝ
  m : ℝ

/-- Triangle formed by intersection points and vertex -/
def triangle_area (E : Ellipse) (l : IntersectingLine E) : ℝ := sorry

/-- Theorem stating the properties of the ellipse and maximum triangle area -/
theorem ellipse_properties_and_max_area (E : Ellipse) :
  E.a = 3 ∧
  (∃ (l : IntersectingLine E), ∀ (l' : IntersectingLine E),
    triangle_area E l ≥ triangle_area E l') ∧
  (∃ (l : IntersectingLine E), triangle_area E l = 27/8) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_and_max_area_l1871_187166


namespace NUMINAMATH_CALUDE_N2O3_molecular_weight_N2O3_is_limiting_reactant_l1871_187184

-- Define atomic weights
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00

-- Define the molecular formula of N2O3
def N2O3_formula : Nat × Nat := (2, 3)

-- Define the number of moles for each reactant
def moles_N2O3 : ℝ := 3
def moles_SO2 : ℝ := 4

-- Define the function to calculate molecular weight
def molecular_weight (n_atoms_N n_atoms_O : Nat) : ℝ :=
  (n_atoms_N : ℝ) * atomic_weight_N + (n_atoms_O : ℝ) * atomic_weight_O

-- Define the function to determine the limiting reactant
def is_limiting_reactant (moles_A moles_B : ℝ) (coeff_A coeff_B : Nat) : Prop :=
  moles_A / (coeff_A : ℝ) < moles_B / (coeff_B : ℝ)

-- Theorem statements
theorem N2O3_molecular_weight :
  molecular_weight N2O3_formula.1 N2O3_formula.2 = 76.02 := by sorry

theorem N2O3_is_limiting_reactant :
  is_limiting_reactant moles_N2O3 moles_SO2 1 1 := by sorry

end NUMINAMATH_CALUDE_N2O3_molecular_weight_N2O3_is_limiting_reactant_l1871_187184


namespace NUMINAMATH_CALUDE_min_length_AB_l1871_187120

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 4)^2 = 2

-- Define the line l
def line_l (x y : ℝ) : Prop := 2*x - y - 3 = 0

-- Define the chord MN
def chord_MN (M N : ℝ × ℝ) : Prop := 
  circle_C M.1 M.2 ∧ circle_C N.1 N.2

-- Define the perpendicularity condition
def perpendicular_CM_CN (C M N : ℝ × ℝ) : Prop := 
  (M.1 - C.1) * (N.1 - C.1) + (M.2 - C.2) * (N.2 - C.2) = 0

-- Define the midpoint condition
def midpoint_P (P M N : ℝ × ℝ) : Prop := 
  P.1 = (M.1 + N.1) / 2 ∧ P.2 = (M.2 + N.2) / 2

-- Define the angle condition
def angle_APB_geq_pi_div_2 (A P B : ℝ × ℝ) : Prop :=
  (P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) ≤ 0

-- Main theorem
theorem min_length_AB : 
  ∀ (M N P A B : ℝ × ℝ),
    chord_MN M N →
    perpendicular_CM_CN (2, 4) M N →
    midpoint_P P M N →
    line_l A.1 A.2 →
    line_l B.1 B.2 →
    angle_APB_geq_pi_div_2 A P B →
    (A.1 - B.1)^2 + (A.2 - B.2)^2 ≥ ((6 * Real.sqrt 5) / 5 + 2)^2 :=
sorry

end NUMINAMATH_CALUDE_min_length_AB_l1871_187120


namespace NUMINAMATH_CALUDE_berkeley_b_count_l1871_187155

def abraham_total : ℕ := 20
def abraham_b : ℕ := 12
def berkeley_total : ℕ := 30

theorem berkeley_b_count : ℕ := by
  -- Define berkeley_b as the number of students in Mrs. Berkeley's class who received a 'B'
  -- Prove that berkeley_b = 18 given the conditions
  sorry

#check berkeley_b_count

end NUMINAMATH_CALUDE_berkeley_b_count_l1871_187155


namespace NUMINAMATH_CALUDE_system_solution_unique_l1871_187132

theorem system_solution_unique :
  ∃! (x y : ℝ), x + y = 15 ∧ x - y = 5 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l1871_187132


namespace NUMINAMATH_CALUDE_negative_fraction_range_l1871_187109

theorem negative_fraction_range (x : ℝ) : (-5 : ℝ) / (2 - x) < 0 → x < 2 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_range_l1871_187109


namespace NUMINAMATH_CALUDE_factorization_equality_l1871_187148

theorem factorization_equality (x y : ℝ) : 
  (x + y)^2 + 4*(x - y)^2 - 4*(x^2 - y^2) = (x - 3*y)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_equality_l1871_187148


namespace NUMINAMATH_CALUDE_invoice_problem_l1871_187173

theorem invoice_problem (x y : ℚ) : 
  (0.3 < x) ∧ (x < 0.4) ∧ 
  (7000 < y) ∧ (y < 8000) ∧ 
  (y * 100 - (y.floor * 100) = 65) ∧
  (237 * x = y) →
  (x = 0.31245 ∧ y = 7400.65) := by
sorry

end NUMINAMATH_CALUDE_invoice_problem_l1871_187173


namespace NUMINAMATH_CALUDE_sandras_sweets_l1871_187131

theorem sandras_sweets (saved : ℚ) (mother_gave : ℚ) (father_gave : ℚ)
  (candy_cost : ℚ) (jelly_bean_cost : ℚ) (candy_count : ℕ) (jelly_bean_count : ℕ)
  (remaining : ℚ) :
  saved = 10 →
  mother_gave = 4 →
  candy_cost = 1/2 →
  jelly_bean_cost = 1/5 →
  candy_count = 14 →
  jelly_bean_count = 20 →
  remaining = 11 →
  saved + mother_gave + father_gave = 
    candy_cost * candy_count + jelly_bean_cost * jelly_bean_count + remaining →
  father_gave / mother_gave = 2 := by
sorry

#eval (8 : ℚ) / (4 : ℚ) -- Expected output: 2

end NUMINAMATH_CALUDE_sandras_sweets_l1871_187131


namespace NUMINAMATH_CALUDE_cone_cylinder_volume_ratio_l1871_187111

/-- The ratio of the volume of a cone to the volume of a cylinder with specified dimensions -/
theorem cone_cylinder_volume_ratio :
  let cone_height : ℝ := 10
  let cylinder_height : ℝ := 30
  let radius : ℝ := 5
  let cone_volume := (1/3) * π * radius^2 * cone_height
  let cylinder_volume := π * radius^2 * cylinder_height
  cone_volume / cylinder_volume = 2/9 := by
sorry

end NUMINAMATH_CALUDE_cone_cylinder_volume_ratio_l1871_187111


namespace NUMINAMATH_CALUDE_trigonometric_and_algebraic_identities_l1871_187156

theorem trigonometric_and_algebraic_identities :
  (2 * Real.sin (45 * π / 180) ^ 2 + Real.tan (60 * π / 180) * Real.tan (30 * π / 180) - Real.cos (60 * π / 180) = 3 / 2) ∧
  (Real.sqrt 12 - 2 * Real.cos (30 * π / 180) + (3 - Real.pi) ^ 0 + |1 - Real.sqrt 3| = 2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_and_algebraic_identities_l1871_187156


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_97_l1871_187118

theorem gcd_of_powers_of_97 : 
  Nat.Prime 97 → Nat.gcd (97^7 + 1) (97^7 + 97^3 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_97_l1871_187118


namespace NUMINAMATH_CALUDE_prob_same_color_is_31_105_l1871_187102

def blue_marbles : ℕ := 4
def yellow_marbles : ℕ := 5
def black_marbles : ℕ := 6
def total_marbles : ℕ := blue_marbles + yellow_marbles + black_marbles

def prob_same_color : ℚ :=
  (blue_marbles * (blue_marbles - 1) + yellow_marbles * (yellow_marbles - 1) + black_marbles * (black_marbles - 1)) /
  (total_marbles * (total_marbles - 1))

theorem prob_same_color_is_31_105 : prob_same_color = 31 / 105 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_color_is_31_105_l1871_187102
