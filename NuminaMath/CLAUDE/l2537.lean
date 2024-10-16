import Mathlib

namespace NUMINAMATH_CALUDE_six_digit_divisibility_l2537_253712

theorem six_digit_divisibility (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) :
  ∃ (k : ℕ), 100100 * a + 10010 * b + 1001 * c = 7 * 11 * 13 * k := by
  sorry

end NUMINAMATH_CALUDE_six_digit_divisibility_l2537_253712


namespace NUMINAMATH_CALUDE_angle_A_measure_l2537_253731

theorem angle_A_measure (A B C : ℝ) (a b c : ℝ) : 
  a = Real.sqrt 7 →
  b = 3 →
  Real.sqrt 7 * Real.sin B + Real.sin A = 2 * Real.sqrt 3 →
  A = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_measure_l2537_253731


namespace NUMINAMATH_CALUDE_expected_sixes_is_half_l2537_253741

-- Define the number of dice
def num_dice : ℕ := 3

-- Define the probability of rolling a 6 on one die
def prob_six : ℚ := 1 / 6

-- Define the expected number of 6's
def expected_sixes : ℚ := num_dice * prob_six

-- Theorem statement
theorem expected_sixes_is_half : expected_sixes = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expected_sixes_is_half_l2537_253741


namespace NUMINAMATH_CALUDE_x_equals_y_l2537_253728

theorem x_equals_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (eq1 : x = 2 + 1 / y) (eq2 : y = 2 + 1 / x) : y = x := by
  sorry

end NUMINAMATH_CALUDE_x_equals_y_l2537_253728


namespace NUMINAMATH_CALUDE_intensity_reduction_target_met_emissions_decrease_from_2030_l2537_253790

-- Define the initial carbon emission intensity in 2005
def initial_intensity : ℝ := 3

-- Define the annual decrease in carbon emission intensity
def annual_decrease : ℝ := 0.08

-- Define the target reduction percentage
def target_reduction : ℝ := 0.4

-- Define the initial GDP in 2005 (in million yuan)
def initial_gdp : ℝ := 1

-- Define the annual GDP growth rate
def gdp_growth_rate : ℝ := 0.08

-- Function to calculate carbon emission intensity for a given year
def carbon_intensity (year : ℕ) : ℝ :=
  initial_intensity - (year - 2005) * annual_decrease

-- Function to calculate GDP for a given year
def gdp (year : ℕ) : ℝ :=
  initial_gdp * (1 + gdp_growth_rate) ^ (year - 2005)

-- Function to calculate total carbon emissions for a given year
def carbon_emissions (year : ℕ) : ℝ :=
  (carbon_intensity year) * (gdp year) * 1000

-- Theorem 1: The carbon emission intensity in 2020 meets the 40% reduction target
theorem intensity_reduction_target_met : carbon_intensity 2020 = initial_intensity * (1 - target_reduction) :=
  sorry

-- Theorem 2: Carbon dioxide emissions start to decrease from 2030
theorem emissions_decrease_from_2030 : 
  ∀ (year : ℕ), year ≥ 2030 → carbon_emissions (year + 1) < carbon_emissions year :=
  sorry

end NUMINAMATH_CALUDE_intensity_reduction_target_met_emissions_decrease_from_2030_l2537_253790


namespace NUMINAMATH_CALUDE_circle_area_increase_l2537_253774

theorem circle_area_increase (r : ℝ) (hr : r > 0) : 
  let new_radius := 2.5 * r
  let original_area := π * r^2
  let new_area := π * new_radius^2
  (new_area - original_area) / original_area = 5.25 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_increase_l2537_253774


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2537_253703

theorem fraction_to_decimal :
  (5 : ℚ) / 16 = (3125 : ℚ) / 10000 :=
by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2537_253703


namespace NUMINAMATH_CALUDE_coefficients_of_given_equation_l2537_253747

/-- Represents a quadratic equation of the form ax² + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The given quadratic equation 2x² + 3x - 1 = 0 -/
def givenEquation : QuadraticEquation :=
  { a := 2, b := 3, c := -1 }

theorem coefficients_of_given_equation :
  givenEquation.a = 2 ∧ givenEquation.b = 3 ∧ givenEquation.c = -1 := by
  sorry

end NUMINAMATH_CALUDE_coefficients_of_given_equation_l2537_253747


namespace NUMINAMATH_CALUDE_sin_150_degrees_l2537_253771

theorem sin_150_degrees : Real.sin (150 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_150_degrees_l2537_253771


namespace NUMINAMATH_CALUDE_tyrone_gave_fifteen_marbles_l2537_253754

/-- Represents the marble redistribution problem between Tyrone and Eric -/
def marble_redistribution (x : ℕ) : Prop :=
  let tyrone_initial : ℕ := 150
  let eric_initial : ℕ := 30
  let tyrone_final : ℕ := tyrone_initial - x
  let eric_final : ℕ := eric_initial + x
  (tyrone_final = 3 * eric_final) ∧ (x > 0) ∧ (x < tyrone_initial)

/-- The theorem stating that Tyrone gave 15 marbles to Eric -/
theorem tyrone_gave_fifteen_marbles : 
  ∃ (x : ℕ), marble_redistribution x ∧ x = 15 := by
sorry

end NUMINAMATH_CALUDE_tyrone_gave_fifteen_marbles_l2537_253754


namespace NUMINAMATH_CALUDE_total_spent_equals_42_33_l2537_253780

/-- The total amount Joan spent on clothing -/
def total_spent : ℚ := 15 + 14.82 + 12.51

/-- Theorem stating that the total amount spent is equal to $42.33 -/
theorem total_spent_equals_42_33 : total_spent = 42.33 := by sorry

end NUMINAMATH_CALUDE_total_spent_equals_42_33_l2537_253780


namespace NUMINAMATH_CALUDE_square_root_calculation_l2537_253736

theorem square_root_calculation : 2 * (Real.sqrt 50625)^2 = 101250 := by
  sorry

end NUMINAMATH_CALUDE_square_root_calculation_l2537_253736


namespace NUMINAMATH_CALUDE_bicycle_profit_percentage_l2537_253791

/-- Represents the profit percentage as a rational number between 0 and 1 -/
def ProfitPercentage : Type := { r : ℚ // 0 ≤ r ∧ r ≤ 1 }

/-- Calculate the selling price given the cost price and profit percentage -/
def sellingPrice (costPrice : ℚ) (profitPercentage : ProfitPercentage) : ℚ :=
  costPrice * (1 + profitPercentage.val)

theorem bicycle_profit_percentage 
  (costPriceA : ℚ)
  (profitPercentageA : ProfitPercentage)
  (finalSellingPrice : ℚ) :
  costPriceA = 120 →
  profitPercentageA.val = 1/2 →
  finalSellingPrice = 225 →
  let sellingPriceA := sellingPrice costPriceA profitPercentageA
  let profitB := finalSellingPrice - sellingPriceA
  let profitPercentageB : ProfitPercentage := ⟨profitB / sellingPriceA, sorry⟩
  profitPercentageB.val = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_profit_percentage_l2537_253791


namespace NUMINAMATH_CALUDE_necklace_bead_count_l2537_253778

/-- Proves that the total number of beads in a necklace is 40 -/
theorem necklace_bead_count :
  let amethyst_count : ℕ := 7
  let amber_count : ℕ := 2 * amethyst_count
  let turquoise_count : ℕ := 19
  let total_count : ℕ := amethyst_count + amber_count + turquoise_count
  total_count = 40 := by sorry

end NUMINAMATH_CALUDE_necklace_bead_count_l2537_253778


namespace NUMINAMATH_CALUDE_existence_of_unequal_indices_l2537_253711

theorem existence_of_unequal_indices (a b c : ℕ → ℕ) : 
  ∃ m n : ℕ, m ≠ n ∧ a m ≥ a n ∧ b m ≥ b n ∧ c m ≥ c n := by
  sorry

end NUMINAMATH_CALUDE_existence_of_unequal_indices_l2537_253711


namespace NUMINAMATH_CALUDE_cross_product_zero_implies_values_l2537_253726

def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2.1 * b.2.2 - a.2.2 * b.2.1,
   a.2.2 * b.1 - a.1 * b.2.2,
   a.1 * b.2.1 - a.2.1 * b.1)

theorem cross_product_zero_implies_values (x y : ℝ) :
  cross_product (3, x, -9) (4, 6, y) = (0, 0, 0) →
  x = 9/2 ∧ y = -12 := by
  sorry

end NUMINAMATH_CALUDE_cross_product_zero_implies_values_l2537_253726


namespace NUMINAMATH_CALUDE_sunset_increase_calculation_l2537_253749

/-- The daily increase in sunset time, given initial and final sunset times over a period. -/
def daily_sunset_increase (initial_time final_time : ℕ) (days : ℕ) : ℚ :=
  (final_time - initial_time) / days

/-- Theorem stating that the daily sunset increase is 1.2 minutes under given conditions. -/
theorem sunset_increase_calculation :
  let initial_time := 18 * 60  -- 6:00 PM in minutes since midnight
  let final_time := 18 * 60 + 48  -- 6:48 PM in minutes since midnight
  let days := 40
  daily_sunset_increase initial_time final_time days = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_sunset_increase_calculation_l2537_253749


namespace NUMINAMATH_CALUDE_expression_simplification_l2537_253727

theorem expression_simplification (x y : ℝ) :
  x * (4 * x^3 - 3 * x^2 + 2 * y) - 6 * (x^3 - 3 * x^2 + 2 * x + 8) =
  4 * x^4 - 9 * x^3 + 18 * x^2 + 2 * x * y - 12 * x - 48 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2537_253727


namespace NUMINAMATH_CALUDE_inequality_proof_l2537_253769

theorem inequality_proof (x y z : ℝ) 
  (h_nonneg_x : 0 ≤ x) (h_nonneg_y : 0 ≤ y) (h_nonneg_z : 0 ≤ z)
  (h_sum : x + y + z = 1) :
  2 ≤ (1 - x^2)^2 + (1 - y^2)^2 + (1 - z^2)^2 ∧ 
  (1 - x^2)^2 + (1 - y^2)^2 + (1 - z^2)^2 ≤ (1 + x) * (1 + y) * (1 + z) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2537_253769


namespace NUMINAMATH_CALUDE_garbage_ratio_proof_l2537_253744

def garbage_problem (collection_days_per_week : ℕ) 
                    (avg_collection_per_day : ℕ) 
                    (weeks_without_collection : ℕ) 
                    (total_accumulated : ℕ) : Prop :=
  let weekly_collection := collection_days_per_week * avg_collection_per_day
  let total_normal_collection := weekly_collection * weeks_without_collection
  let first_week_garbage := weekly_collection
  let second_week_garbage := total_accumulated - first_week_garbage
  (2 : ℚ) * second_week_garbage = first_week_garbage

theorem garbage_ratio_proof : 
  garbage_problem 3 200 2 900 := by
  sorry

#check garbage_ratio_proof

end NUMINAMATH_CALUDE_garbage_ratio_proof_l2537_253744


namespace NUMINAMATH_CALUDE_male_rabbits_count_l2537_253767

theorem male_rabbits_count (white : ℕ) (black : ℕ) (female : ℕ) 
  (h1 : white = 12) 
  (h2 : black = 9) 
  (h3 : female = 8) : 
  white + black - female = 13 := by
  sorry

end NUMINAMATH_CALUDE_male_rabbits_count_l2537_253767


namespace NUMINAMATH_CALUDE_probability_at_least_one_even_is_65_81_l2537_253707

def valid_digits : Finset ℕ := {0, 3, 5, 7, 8, 9}
def code_length : ℕ := 4

def probability_at_least_one_even : ℚ :=
  1 - (Finset.filter (λ x => ¬ Even x) valid_digits).card ^ code_length /
      valid_digits.card ^ code_length

theorem probability_at_least_one_even_is_65_81 :
  probability_at_least_one_even = 65 / 81 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_even_is_65_81_l2537_253707


namespace NUMINAMATH_CALUDE_jeffrey_mailbox_steps_l2537_253742

/-- Represents Jeffrey's walking pattern -/
structure WalkingPattern where
  forward : ℕ
  backward : ℕ

/-- Calculates the total steps taken given a walking pattern and distance -/
def totalSteps (pattern : WalkingPattern) (distance : ℕ) : ℕ :=
  distance * (pattern.forward + pattern.backward) / (pattern.forward - pattern.backward)

/-- Theorem: Jeffrey takes 330 steps to reach the mailbox -/
theorem jeffrey_mailbox_steps :
  let pattern : WalkingPattern := { forward := 3, backward := 2 }
  let distance : ℕ := 66
  totalSteps pattern distance = 330 := by
  sorry

#eval totalSteps { forward := 3, backward := 2 } 66

end NUMINAMATH_CALUDE_jeffrey_mailbox_steps_l2537_253742


namespace NUMINAMATH_CALUDE_tangerines_oranges_percentage_l2537_253789

/-- Represents the quantities of fruits in Tina's bag -/
structure FruitBag where
  apples : ℕ
  oranges : ℕ
  tangerines : ℕ
  grapes : ℕ
  kiwis : ℕ

/-- Calculates the total number of fruits in the bag -/
def totalFruits (bag : FruitBag) : ℕ :=
  bag.apples + bag.oranges + bag.tangerines + bag.grapes + bag.kiwis

/-- Calculates the number of tangerines and oranges in the bag -/
def tangerinesAndOranges (bag : FruitBag) : ℕ :=
  bag.tangerines + bag.oranges

/-- Theorem stating that the percentage of tangerines and oranges in the remaining fruits is 47.5% -/
theorem tangerines_oranges_percentage (initialBag : FruitBag)
    (h1 : initialBag.apples = 9)
    (h2 : initialBag.oranges = 5)
    (h3 : initialBag.tangerines = 17)
    (h4 : initialBag.grapes = 12)
    (h5 : initialBag.kiwis = 7) :
    let finalBag : FruitBag := {
      apples := initialBag.apples,
      oranges := initialBag.oranges - 2 + 3,
      tangerines := initialBag.tangerines - 10 + 6,
      grapes := initialBag.grapes - 4,
      kiwis := initialBag.kiwis - 3
    }
    (tangerinesAndOranges finalBag : ℚ) / (totalFruits finalBag : ℚ) * 100 = 47.5 := by
  sorry

end NUMINAMATH_CALUDE_tangerines_oranges_percentage_l2537_253789


namespace NUMINAMATH_CALUDE_sum_of_cubes_equation_l2537_253779

theorem sum_of_cubes_equation (x y : ℝ) : 
  x^3 + 21*x*y + y^3 = 343 → (x + y = 7 ∨ x + y = -14) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_equation_l2537_253779


namespace NUMINAMATH_CALUDE_derivative_y_l2537_253738

def y (x : ℝ) : ℝ := x^2 - 5*x + 4

theorem derivative_y (x : ℝ) : 
  deriv y x = 2*x - 5 := by sorry

end NUMINAMATH_CALUDE_derivative_y_l2537_253738


namespace NUMINAMATH_CALUDE_student_professor_ratio_l2537_253718

def total_people : ℕ := 40000
def num_students : ℕ := 37500

theorem student_professor_ratio :
  let num_professors := total_people - num_students
  num_students / num_professors = 15 := by
  sorry

end NUMINAMATH_CALUDE_student_professor_ratio_l2537_253718


namespace NUMINAMATH_CALUDE_inverse_of_3_mod_229_l2537_253721

theorem inverse_of_3_mod_229 : ∃ x : ℕ, x < 229 ∧ (3 * x) % 229 = 1 :=
  by
    use 153
    sorry

end NUMINAMATH_CALUDE_inverse_of_3_mod_229_l2537_253721


namespace NUMINAMATH_CALUDE_no_linear_function_satisfies_inequality_l2537_253722

theorem no_linear_function_satisfies_inequality :
  ¬ ∃ (a b : ℝ), ∀ x ∈ Set.Icc 0 (2 * Real.pi),
    (a * x + b)^2 - Real.cos x * (a * x + b) < (1/4) * Real.sin x^2 := by
  sorry

end NUMINAMATH_CALUDE_no_linear_function_satisfies_inequality_l2537_253722


namespace NUMINAMATH_CALUDE_characterize_valid_functions_l2537_253713

def is_valid_function (f : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, f (n + 1) > (f n + f (f n)) / 2

theorem characterize_valid_functions :
  ∀ f : ℕ → ℕ, is_valid_function f →
  ∃ b : ℕ, (∀ n < b, f n = n) ∧ (∀ n ≥ b, f n = n + 1) :=
sorry

end NUMINAMATH_CALUDE_characterize_valid_functions_l2537_253713


namespace NUMINAMATH_CALUDE_pure_imaginary_iff_m_eq_3_second_quadrant_iff_m_between_1_and_3_l2537_253777

-- Define the complex number z as a function of real m
def z (m : ℝ) : ℂ := (m^2 - 2*m - 3 : ℝ) + (m^2 - 1 : ℝ) * Complex.I

-- Part 1: z is a pure imaginary number iff m = 3
theorem pure_imaginary_iff_m_eq_3 :
  ∀ m : ℝ, (z m).re = 0 ↔ m = 3 :=
sorry

-- Part 2: z is in the second quadrant iff 1 < m < 3
theorem second_quadrant_iff_m_between_1_and_3 :
  ∀ m : ℝ, ((z m).re < 0 ∧ (z m).im > 0) ↔ (1 < m ∧ m < 3) :=
sorry

end NUMINAMATH_CALUDE_pure_imaginary_iff_m_eq_3_second_quadrant_iff_m_between_1_and_3_l2537_253777


namespace NUMINAMATH_CALUDE_square_difference_given_sum_and_product_l2537_253724

theorem square_difference_given_sum_and_product (x y : ℝ) 
  (h1 : x^2 + y^2 = 10) (h2 : x * y = 2) : (x - y)^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_given_sum_and_product_l2537_253724


namespace NUMINAMATH_CALUDE_exists_negative_greater_than_neg_half_l2537_253798

theorem exists_negative_greater_than_neg_half : ∃ x : ℚ, -1/2 < x ∧ x < 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_negative_greater_than_neg_half_l2537_253798


namespace NUMINAMATH_CALUDE_Z_set_eq_roster_l2537_253799

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the set we want to prove
def Z_set : Set ℂ := {z | ∃ n : ℤ, z = i^n + i^(-n)}

-- The theorem to prove
theorem Z_set_eq_roster : Z_set = {0, 2, -2} := by sorry

end NUMINAMATH_CALUDE_Z_set_eq_roster_l2537_253799


namespace NUMINAMATH_CALUDE_max_value_at_two_l2537_253786

-- Define the function f
def f (c : ℝ) (x : ℝ) : ℝ := x * (x - c)^2

-- State the theorem
theorem max_value_at_two (c : ℝ) :
  (∀ x : ℝ, f c x ≤ f c 2) → c = 6 :=
by sorry

end NUMINAMATH_CALUDE_max_value_at_two_l2537_253786


namespace NUMINAMATH_CALUDE_combined_girls_average_is_89_l2537_253725

/-- Represents a high school with average test scores -/
structure School where
  boyAvg : ℝ
  girlAvg : ℝ
  combinedAvg : ℝ

/-- Calculates the combined average score for girls given two schools and the combined boys' average -/
def combinedGirlsAverage (lincoln : School) (monroe : School) (combinedBoysAvg : ℝ) : ℝ :=
  sorry

theorem combined_girls_average_is_89 (lincoln : School) (monroe : School) (combinedBoysAvg : ℝ) :
  lincoln.boyAvg = 75 ∧
  lincoln.girlAvg = 78 ∧
  lincoln.combinedAvg = 76 ∧
  monroe.boyAvg = 85 ∧
  monroe.girlAvg = 92 ∧
  monroe.combinedAvg = 88 ∧
  combinedBoysAvg = 82 →
  combinedGirlsAverage lincoln monroe combinedBoysAvg = 89 := by
  sorry

end NUMINAMATH_CALUDE_combined_girls_average_is_89_l2537_253725


namespace NUMINAMATH_CALUDE_digital_root_of_2_pow_100_l2537_253719

/-- The digital root of a natural number is the single digit obtained by repeatedly summing its digits. -/
def digital_root (n : ℕ) : ℕ := sorry

/-- Theorem: The digital root of 2^100 is 7. -/
theorem digital_root_of_2_pow_100 : digital_root (2^100) = 7 := by sorry

end NUMINAMATH_CALUDE_digital_root_of_2_pow_100_l2537_253719


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2537_253708

theorem arithmetic_sequence_middle_term (a₁ a₃ z : ℤ) : 
  a₁ = 2^3 → a₃ = 2^5 → z = (a₁ + a₃) / 2 → z = 20 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2537_253708


namespace NUMINAMATH_CALUDE_max_reciprocal_eccentricity_sum_l2537_253737

theorem max_reciprocal_eccentricity_sum (e₁ e₂ : ℝ) : 
  e₁ > 0 → e₂ > 0 → 
  (∃ b c : ℝ, b > 0 ∧ c > b ∧ 
    e₁ = c / Real.sqrt (c^2 + (2*b)^2) ∧ 
    e₂ = c / Real.sqrt (c^2 - b^2)) → 
  1/e₁^2 + 4/e₂^2 = 5 → 
  1/e₁ + 1/e₂ ≤ 5/2 :=
by sorry

end NUMINAMATH_CALUDE_max_reciprocal_eccentricity_sum_l2537_253737


namespace NUMINAMATH_CALUDE_common_solutions_exist_l2537_253797

theorem common_solutions_exist (y : ℝ) : 
  (∃ x : ℝ, x^2 + y^2 - 16 = 0 ∧ x^2 - 3*y - 12 = 0) ↔ (y = -4 ∨ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_common_solutions_exist_l2537_253797


namespace NUMINAMATH_CALUDE_fifth_term_of_sequence_l2537_253748

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem fifth_term_of_sequence (a₁ a₂ a₃ : ℕ) (h1 : a₁ = 3) (h2 : a₂ = 7) (h3 : a₃ = 11) :
  arithmetic_sequence a₁ (a₂ - a₁) 5 = 19 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_of_sequence_l2537_253748


namespace NUMINAMATH_CALUDE_highlighter_spend_l2537_253773

def total_money : ℕ := 100
def heaven_spend : ℕ := 30
def eraser_price : ℕ := 4
def eraser_count : ℕ := 10

theorem highlighter_spend :
  total_money - heaven_spend - (eraser_price * eraser_count) = 30 := by
  sorry

end NUMINAMATH_CALUDE_highlighter_spend_l2537_253773


namespace NUMINAMATH_CALUDE_prob_ace_hearts_king_spades_l2537_253787

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of cards of each suit in a standard deck -/
def CardsPerSuit : ℕ := 13

/-- The probability of drawing a specific card from a standard deck -/
def ProbFirstCard : ℚ := 1 / StandardDeck

/-- The probability of drawing a specific card from the remaining deck after one card is drawn -/
def ProbSecondCard : ℚ := 1 / (StandardDeck - 1)

/-- The probability of drawing two specific cards in order from a standard deck -/
def ProbTwoSpecificCards : ℚ := ProbFirstCard * ProbSecondCard

theorem prob_ace_hearts_king_spades : 
  ProbTwoSpecificCards = 1 / 2652 := by sorry

end NUMINAMATH_CALUDE_prob_ace_hearts_king_spades_l2537_253787


namespace NUMINAMATH_CALUDE_circle_area_ratio_concentric_circles_area_ratio_l2537_253740

theorem circle_area_ratio : 
  ∀ (r₁ r₂ : ℝ), r₁ > 0 → r₂ > r₁ → 
  (π * r₂^2 - π * r₁^2) / (π * r₁^2) = (r₂^2 / r₁^2) - 1 :=
by sorry

theorem concentric_circles_area_ratio :
  let d₁ : ℝ := 2  -- diameter of smaller circle
  let d₂ : ℝ := 6  -- diameter of larger circle
  let r₁ : ℝ := d₁ / 2  -- radius of smaller circle
  let r₂ : ℝ := d₂ / 2  -- radius of larger circle
  (π * r₂^2 - π * r₁^2) / (π * r₁^2) = 8 :=
by sorry

end NUMINAMATH_CALUDE_circle_area_ratio_concentric_circles_area_ratio_l2537_253740


namespace NUMINAMATH_CALUDE_negation_of_exponential_inequality_l2537_253709

theorem negation_of_exponential_inequality :
  (¬ ∀ x : ℝ, x ≤ 0 → Real.exp x ≤ 1) ↔ (∃ x₀ : ℝ, x₀ ≤ 0 ∧ Real.exp x₀ > 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_exponential_inequality_l2537_253709


namespace NUMINAMATH_CALUDE_h1n1_diameter_scientific_notation_l2537_253717

theorem h1n1_diameter_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 0.000000081 = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ n = -8 :=
sorry

end NUMINAMATH_CALUDE_h1n1_diameter_scientific_notation_l2537_253717


namespace NUMINAMATH_CALUDE_prove_last_score_l2537_253716

def scores : List ℤ := [50, 55, 60, 85, 90, 100]

def is_integer_average (sublist : List ℤ) : Prop :=
  ∃ n : ℤ, (sublist.sum : ℚ) / sublist.length = n

def last_score_is_60 : Prop :=
  ∀ perm : List ℤ, perm.length = 6 →
    perm.toFinset = scores.toFinset →
    (∀ k : ℕ, k ≤ 5 → is_integer_average (perm.take k)) →
    perm.reverse.head? = some 60

theorem prove_last_score : last_score_is_60 := by
  sorry

end NUMINAMATH_CALUDE_prove_last_score_l2537_253716


namespace NUMINAMATH_CALUDE_certain_number_problem_l2537_253700

theorem certain_number_problem (x : ℝ) : 
  (15 - 2 + x) / 2 * 8 = 77 → x = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2537_253700


namespace NUMINAMATH_CALUDE_chlorine_cost_l2537_253720

-- Define the pool dimensions
def pool_length : ℝ := 10
def pool_width : ℝ := 8
def pool_depth : ℝ := 6

-- Define the chlorine requirement
def cubic_feet_per_quart : ℝ := 120

-- Define the cost of chlorine
def cost_per_quart : ℝ := 3

-- Theorem statement
theorem chlorine_cost : 
  let pool_volume : ℝ := pool_length * pool_width * pool_depth
  let quarts_needed : ℝ := pool_volume / cubic_feet_per_quart
  let total_cost : ℝ := quarts_needed * cost_per_quart
  total_cost = 12 := by sorry

end NUMINAMATH_CALUDE_chlorine_cost_l2537_253720


namespace NUMINAMATH_CALUDE_elena_bouquet_petals_l2537_253758

/-- Represents the number of flowers of each type in Elena's garden -/
structure FlowerCounts where
  lilies : ℕ
  tulips : ℕ
  roses : ℕ
  daisies : ℕ

/-- Represents the number of petals for each type of flower -/
structure PetalCounts where
  lily_petals : ℕ
  tulip_petals : ℕ
  rose_petals : ℕ
  daisy_petals : ℕ

/-- Calculates the number of flowers to take for the bouquet -/
def bouquet_flowers (garden : FlowerCounts) : FlowerCounts :=
  let min_count := min (garden.lilies / 2) (min (garden.tulips / 2) (min (garden.roses / 2) (garden.daisies / 2)))
  { lilies := min_count
    tulips := min_count
    roses := min_count
    daisies := min_count }

/-- Calculates the total number of petals in the bouquet -/
def total_petals (flowers : FlowerCounts) (petals : PetalCounts) : ℕ :=
  flowers.lilies * petals.lily_petals +
  flowers.tulips * petals.tulip_petals +
  flowers.roses * petals.rose_petals +
  flowers.daisies * petals.daisy_petals

/-- Elena's garden and petal counts -/
def elena_garden : FlowerCounts := { lilies := 8, tulips := 5, roses := 4, daisies := 3 }
def elena_petals : PetalCounts := { lily_petals := 6, tulip_petals := 3, rose_petals := 5, daisy_petals := 12 }

theorem elena_bouquet_petals :
  total_petals (bouquet_flowers elena_garden) elena_petals = 52 := by
  sorry


end NUMINAMATH_CALUDE_elena_bouquet_petals_l2537_253758


namespace NUMINAMATH_CALUDE_kiwis_to_add_for_orange_percentage_l2537_253715

/-- Proves that adding 7 kiwis to a box with 24 oranges, 30 kiwis, 15 apples, and 20 bananas
    will make oranges exactly 25% of the total fruits -/
theorem kiwis_to_add_for_orange_percentage (oranges kiwis apples bananas : ℕ) 
    (h1 : oranges = 24) 
    (h2 : kiwis = 30) 
    (h3 : apples = 15) 
    (h4 : bananas = 20) : 
    let total := oranges + kiwis + apples + bananas + 7
    (oranges : ℚ) / total = 1/4 := by sorry

end NUMINAMATH_CALUDE_kiwis_to_add_for_orange_percentage_l2537_253715


namespace NUMINAMATH_CALUDE_equation_roots_l2537_253759

theorem equation_roots : ∃ (x y : ℝ), x < 0 ∧ y = 0 ∧
  3^x + x^2 + 2*x - 1 = 0 ∧
  3^y + y^2 + 2*y - 1 = 0 ∧
  ∀ (z : ℝ), (3^z + z^2 + 2*z - 1 = 0) → (z = x ∨ z = y) :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_l2537_253759


namespace NUMINAMATH_CALUDE_matchstick_20th_stage_l2537_253765

/-- Arithmetic sequence with first term 3 and common difference 3 -/
def matchstick_sequence (n : ℕ) : ℕ := 3 + (n - 1) * 3

/-- The 20th term of the matchstick sequence is 60 -/
theorem matchstick_20th_stage : matchstick_sequence 20 = 60 := by
  sorry

end NUMINAMATH_CALUDE_matchstick_20th_stage_l2537_253765


namespace NUMINAMATH_CALUDE_germs_left_percentage_l2537_253788

/-- Represents the effectiveness of four sanitizer sprays and their overlaps -/
structure SanitizerSprays where
  /-- Kill rates for each spray -/
  spray1 : ℝ
  spray2 : ℝ
  spray3 : ℝ
  spray4 : ℝ
  /-- Two-way overlaps between sprays -/
  overlap12 : ℝ
  overlap23 : ℝ
  overlap34 : ℝ
  overlap13 : ℝ
  overlap14 : ℝ
  overlap24 : ℝ
  /-- Three-way overlaps between sprays -/
  overlap123 : ℝ
  overlap234 : ℝ

/-- Calculates the percentage of germs left after applying all sprays -/
def germsLeft (s : SanitizerSprays) : ℝ :=
  100 - (s.spray1 + s.spray2 + s.spray3 + s.spray4 - 
         (s.overlap12 + s.overlap23 + s.overlap34 + s.overlap13 + s.overlap14 + s.overlap24) -
         (s.overlap123 + s.overlap234))

/-- Theorem stating that for the given spray effectiveness and overlaps, 13.8% of germs are left -/
theorem germs_left_percentage (s : SanitizerSprays) 
  (h1 : s.spray1 = 50) (h2 : s.spray2 = 35) (h3 : s.spray3 = 20) (h4 : s.spray4 = 10)
  (h5 : s.overlap12 = 10) (h6 : s.overlap23 = 7) (h7 : s.overlap34 = 5)
  (h8 : s.overlap13 = 3) (h9 : s.overlap14 = 2) (h10 : s.overlap24 = 1)
  (h11 : s.overlap123 = 0.5) (h12 : s.overlap234 = 0.3) :
  germsLeft s = 13.8 := by
  sorry


end NUMINAMATH_CALUDE_germs_left_percentage_l2537_253788


namespace NUMINAMATH_CALUDE_cubic_function_two_zeros_l2537_253745

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := 3 * x^3 - x + a

-- State the theorem
theorem cubic_function_two_zeros (a : ℝ) (h : a > 0) :
  (∃! x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0) ↔ a = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_two_zeros_l2537_253745


namespace NUMINAMATH_CALUDE_area_BEIH_l2537_253730

/-- Given a 2×2 square ABCD with B at (0,0), E is the midpoint of AB, 
    F is the midpoint of BC, I is the intersection of AF and DE, 
    and H is the intersection of BD and AF. -/
def square_setup (A B C D E F H I : ℝ × ℝ) : Prop :=
  B = (0, 0) ∧ 
  C = (2, 0) ∧ 
  D = (2, 2) ∧ 
  A = (0, 2) ∧
  E = (0, 1) ∧
  F = (1, 0) ∧
  H.1 = H.2 ∧ -- H is on the diagonal BD
  I.2 = -2 * I.1 + 2 ∧ -- I is on line AF
  I.2 = (1/2) * I.1 + 1 -- I is on line DE

/-- The area of quadrilateral BEIH is 7/15 -/
theorem area_BEIH (A B C D E F H I : ℝ × ℝ) 
  (h : square_setup A B C D E F H I) : 
  let area := (1/2) * abs ((E.1 * I.2 + I.1 * H.2 + H.1 * B.2 + B.1 * E.2) - 
                           (E.2 * I.1 + I.2 * H.1 + H.2 * B.1 + B.2 * E.1))
  area = 7/15 := by
sorry

end NUMINAMATH_CALUDE_area_BEIH_l2537_253730


namespace NUMINAMATH_CALUDE_slope_product_on_hyperbola_l2537_253704

noncomputable def hyperbola (x y : ℝ) : Prop :=
  x^2 - (2 * y^2) / (Real.sqrt 5 + 1) = 1

theorem slope_product_on_hyperbola 
  (M N P : ℝ × ℝ) 
  (hM : hyperbola M.1 M.2) 
  (hN : hyperbola N.1 N.2) 
  (hP : hyperbola P.1 P.2) 
  (hMN : N = (-M.1, -M.2)) :
  let k_PM := (P.2 - M.2) / (P.1 - M.1)
  let k_PN := (P.2 - N.2) / (P.1 - N.1)
  k_PM * k_PN = (Real.sqrt 5 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_slope_product_on_hyperbola_l2537_253704


namespace NUMINAMATH_CALUDE_circle_radius_with_secant_l2537_253784

/-- Represents a circle with an external point and a secant --/
structure CircleWithSecant where
  -- Radius of the circle
  r : ℝ
  -- Distance from external point P to center
  distPC : ℝ
  -- Length of external segment PQ
  lenPQ : ℝ
  -- Length of segment QR
  lenQR : ℝ
  -- Condition: P is outside the circle
  h_outside : distPC > r
  -- Condition: PQ is external segment
  h_external : lenPQ < distPC

/-- The radius of the circle given the specified conditions --/
theorem circle_radius_with_secant (c : CircleWithSecant)
    (h_distPC : c.distPC = 17)
    (h_lenPQ : c.lenPQ = 12)
    (h_lenQR : c.lenQR = 8) :
    c.r = 7 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_with_secant_l2537_253784


namespace NUMINAMATH_CALUDE_page_lines_increase_l2537_253732

/-- 
Given an original number of lines L in a page, 
if increasing the number of lines by 80 results in a 50% increase, 
then the new total number of lines is 240.
-/
theorem page_lines_increase (L : ℕ) : 
  (L + 80 = L + L / 2) → (L + 80 = 240) := by
  sorry

end NUMINAMATH_CALUDE_page_lines_increase_l2537_253732


namespace NUMINAMATH_CALUDE_approximate_0_9915_l2537_253733

theorem approximate_0_9915 : 
  ∃ (x : ℚ), (x = 0.956) ∧ 
  (∀ (y : ℚ), abs (y - 0.9915) < abs (x - 0.9915) → abs (y - 0.9915) ≥ 0.0005) :=
by sorry

end NUMINAMATH_CALUDE_approximate_0_9915_l2537_253733


namespace NUMINAMATH_CALUDE_fraction_equality_l2537_253768

theorem fraction_equality (a b c : ℝ) 
  (h1 : a / b = 20) 
  (h2 : b / c = 10) : 
  (a + b) / (b + c) = 210 / 11 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2537_253768


namespace NUMINAMATH_CALUDE_inserted_numbers_sum_l2537_253701

theorem inserted_numbers_sum : 
  ∀ (a b : ℝ), 
    (∃ (d : ℝ), a = 4 + d ∧ b = 4 + 2*d) →  -- arithmetic progression condition
    (∃ (r : ℝ), b = a*r ∧ 16 = b*r) →       -- geometric progression condition
    a + b = 6 * Real.sqrt 3 + 8 := by
sorry

end NUMINAMATH_CALUDE_inserted_numbers_sum_l2537_253701


namespace NUMINAMATH_CALUDE_largest_square_and_rectangle_in_right_triangle_l2537_253751

/-- Given a right triangle ABC with legs AC = a and CB = b, prove:
    1. The side length of the largest square (with vertex C) that lies entirely within the triangle ABC is ab/(a+b)
    2. The dimensions of the largest rectangle (with vertex C) that lies entirely within the triangle ABC are a/2 and b/2 -/
theorem largest_square_and_rectangle_in_right_triangle 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let square_side := a * b / (a + b)
  let rectangle_width := a / 2
  let rectangle_height := b / 2
  (∀ s, s > 0 → s * s ≤ square_side * square_side) ∧
  (∀ w h, w > 0 → h > 0 → w * h ≤ rectangle_width * rectangle_height) := by
  sorry

end NUMINAMATH_CALUDE_largest_square_and_rectangle_in_right_triangle_l2537_253751


namespace NUMINAMATH_CALUDE_min_students_with_blue_eyes_and_backpack_l2537_253763

theorem min_students_with_blue_eyes_and_backpack
  (total_students : ℕ)
  (blue_eyes : ℕ)
  (backpack : ℕ)
  (h1 : total_students = 25)
  (h2 : blue_eyes = 15)
  (h3 : backpack = 18)
  : ∃ (both : ℕ), both ≥ 7 ∧ both ≤ min blue_eyes backpack :=
by
  sorry

end NUMINAMATH_CALUDE_min_students_with_blue_eyes_and_backpack_l2537_253763


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l2537_253752

theorem right_triangle_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (right_triangle : a^2 + b^2 = c^2) (b_larger : b > a) (tan_condition : b/a < 2) :
  (a^2 / (b^2 + c^2)) + (b^2 / (a^2 + c^2)) > 4/9 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l2537_253752


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2537_253714

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x < 0}
def B : Set ℝ := {x | |x| > 1}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2537_253714


namespace NUMINAMATH_CALUDE_quadratic_condition_for_x_greater_than_two_l2537_253776

theorem quadratic_condition_for_x_greater_than_two :
  (∀ x : ℝ, x > 2 → x^2 - 3*x + 2 > 0) ∧
  (∃ x : ℝ, x^2 - 3*x + 2 > 0 ∧ x ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_condition_for_x_greater_than_two_l2537_253776


namespace NUMINAMATH_CALUDE_picture_hanging_l2537_253756

theorem picture_hanging (board_width : ℕ) (picture_width : ℕ) (num_pictures : ℕ) :
  board_width = 320 ∧ picture_width = 30 ∧ num_pictures = 6 →
  (board_width - num_pictures * picture_width) / (num_pictures + 1) = 20 :=
by sorry

end NUMINAMATH_CALUDE_picture_hanging_l2537_253756


namespace NUMINAMATH_CALUDE_fixed_point_parabola_l2537_253735

theorem fixed_point_parabola (k : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 3 * x^2 + k * x - 2 * k
  f 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_parabola_l2537_253735


namespace NUMINAMATH_CALUDE_painter_rooms_problem_l2537_253794

theorem painter_rooms_problem (time_per_room : ℕ) (rooms_painted : ℕ) (time_remaining : ℕ) :
  time_per_room = 8 →
  rooms_painted = 8 →
  time_remaining = 16 →
  rooms_painted + (time_remaining / time_per_room) = 10 :=
by sorry

end NUMINAMATH_CALUDE_painter_rooms_problem_l2537_253794


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2537_253770

-- Define the sets M and N
def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {y | ∃ x, y = 2^x}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x | 0 < x ∧ x ≤ 2} :=
sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2537_253770


namespace NUMINAMATH_CALUDE_line_symmetry_l2537_253729

/-- Given a point (x, y) on a line, returns the x-coordinate of its symmetric point with respect to x = 1 -/
def symmetric_x (x : ℝ) : ℝ := 2 - x

/-- The original line -/
def original_line (x y : ℝ) : Prop := x - 2*y + 1 = 0

/-- The symmetric line -/
def symmetric_line (x y : ℝ) : Prop := x + 2*y - 3 = 0

/-- Theorem stating that the symmetric_line is indeed symmetric to the original_line with respect to x = 1 -/
theorem line_symmetry :
  ∀ x y : ℝ, original_line x y → symmetric_line (symmetric_x x) y :=
sorry

end NUMINAMATH_CALUDE_line_symmetry_l2537_253729


namespace NUMINAMATH_CALUDE_text_ratio_is_five_to_one_l2537_253705

/-- Represents the number of texts in each category --/
structure TextCounts where
  grocery : ℕ
  notResponding : ℕ
  police : ℕ

/-- The conditions of the problem --/
def textProblemConditions (t : TextCounts) : Prop :=
  t.grocery = 5 ∧
  t.police = (t.grocery + t.notResponding) / 10 ∧
  t.grocery + t.notResponding + t.police = 33

/-- The theorem to be proved --/
theorem text_ratio_is_five_to_one (t : TextCounts) :
  textProblemConditions t →
  t.notResponding / t.grocery = 5 := by
sorry

end NUMINAMATH_CALUDE_text_ratio_is_five_to_one_l2537_253705


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2537_253766

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Given two vectors a and b, where a = (1, 2) and b = (2x, -3),
    if a is parallel to b, then x = 3 -/
theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (2 * x, -3)
  parallel a b → x = 3 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2537_253766


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2537_253772

theorem trigonometric_identity (α : Real) (m : Real) (h : Real.tan α = m) :
  Real.sin (π/4 + α)^2 - Real.sin (π/6 - α)^2 - Real.cos (5*π/12) * Real.sin (5*π/12 - 2*α) = 2*m / (1 + m^2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2537_253772


namespace NUMINAMATH_CALUDE_first_pumpkin_weight_l2537_253760

/-- The weight of the first pumpkin given the total weight of two pumpkins and the weight of the second pumpkin -/
theorem first_pumpkin_weight (total_weight second_weight : ℝ) 
  (h1 : total_weight = 12.7)
  (h2 : second_weight = 8.7) : 
  total_weight - second_weight = 4 := by
  sorry

end NUMINAMATH_CALUDE_first_pumpkin_weight_l2537_253760


namespace NUMINAMATH_CALUDE_tim_soda_cans_l2537_253746

/-- The number of soda cans Tim has at the end of the scenario -/
def final_soda_cans (initial : ℕ) (taken : ℕ) : ℕ :=
  let remaining := initial - taken
  remaining + remaining / 2

/-- Theorem stating that Tim ends up with 24 cans of soda -/
theorem tim_soda_cans : final_soda_cans 22 6 = 24 := by
  sorry

end NUMINAMATH_CALUDE_tim_soda_cans_l2537_253746


namespace NUMINAMATH_CALUDE_sum_of_binary_digits_222_l2537_253723

/-- The sum of the digits in the binary representation of a natural number -/
def sum_of_binary_digits (n : ℕ) : ℕ :=
  (n.digits 2).sum

/-- Theorem: The sum of the digits in the binary representation of 222 is 6 -/
theorem sum_of_binary_digits_222 :
  sum_of_binary_digits 222 = 6 := by
  sorry

#eval sum_of_binary_digits 222  -- This should output 6

end NUMINAMATH_CALUDE_sum_of_binary_digits_222_l2537_253723


namespace NUMINAMATH_CALUDE_aziz_parents_move_year_l2537_253762

/-- The year Aziz's parents moved to America -/
def year_parents_moved (current_year : ℕ) (aziz_age : ℕ) (years_before_birth : ℕ) : ℕ :=
  current_year - aziz_age - years_before_birth

/-- Proof that Aziz's parents moved to America in 1982 -/
theorem aziz_parents_move_year :
  year_parents_moved 2021 36 3 = 1982 := by
  sorry

end NUMINAMATH_CALUDE_aziz_parents_move_year_l2537_253762


namespace NUMINAMATH_CALUDE_college_girls_count_l2537_253785

theorem college_girls_count (boys girls : ℕ) : 
  (boys : ℚ) / girls = 8 / 5 →
  boys + girls = 780 →
  girls = 300 := by
sorry

end NUMINAMATH_CALUDE_college_girls_count_l2537_253785


namespace NUMINAMATH_CALUDE_ellipse_m_range_collinearity_AGN_l2537_253792

-- Define the curve C
def C (m : ℝ) (x y : ℝ) : Prop := (5 - m) * x^2 + (m - 2) * y^2 = 8

-- Define the condition for C to be an ellipse with foci on x-axis
def is_ellipse_x_foci (m : ℝ) : Prop :=
  (8 / (5 - m) > 8 / (m - 2)) ∧ (8 / (5 - m) > 0) ∧ (8 / (m - 2) > 0)

-- Define the line y = kx + 4
def line_k (k : ℝ) (x y : ℝ) : Prop := y = k * x + 4

-- Define the line y = 1
def line_one (x y : ℝ) : Prop := y = 1

-- Theorem for part 1
theorem ellipse_m_range (m : ℝ) :
  is_ellipse_x_foci m → (7/2 < m) ∧ (m < 5) := by sorry

-- Theorem for part 2
theorem collinearity_AGN (k : ℝ) (xA yA xB yB xM yM xN yN xG : ℝ) :
  C 4 0 yA ∧ C 4 0 yB ∧ yA > yB ∧
  C 4 xM yM ∧ C 4 xN yN ∧
  line_k k xM yM ∧ line_k k xN yN ∧
  line_one xG 1 ∧
  (yM - yB) / (xM - xB) = (1 - yB) / (xG - xB) →
  ∃ (t : ℝ), xG = t * xA + (1 - t) * xN ∧ 1 = t * yA + (1 - t) * yN := by sorry

end NUMINAMATH_CALUDE_ellipse_m_range_collinearity_AGN_l2537_253792


namespace NUMINAMATH_CALUDE_concentric_circles_area_l2537_253706

theorem concentric_circles_area (r : ℝ) (h : r > 0) : 
  2 * π * r + 2 * π * (2 * r) = 36 * π → 
  π * (2 * r)^2 - π * r^2 = 108 * π := by
sorry

end NUMINAMATH_CALUDE_concentric_circles_area_l2537_253706


namespace NUMINAMATH_CALUDE_towel_packs_l2537_253795

theorem towel_packs (towels_per_pack : ℕ) (total_towels : ℕ) (num_packs : ℕ) :
  towels_per_pack = 3 →
  total_towels = 27 →
  num_packs * towels_per_pack = total_towels →
  num_packs = 9 := by
  sorry

end NUMINAMATH_CALUDE_towel_packs_l2537_253795


namespace NUMINAMATH_CALUDE_bulbs_needed_l2537_253796

/-- Represents the number of bulbs required for each type of ceiling light. -/
structure BulbRequirement where
  small : Nat
  medium : Nat
  large : Nat

/-- Represents the number of each type of ceiling light. -/
structure CeilingLights where
  small : Nat
  medium : Nat
  large : Nat

/-- Calculates the total number of bulbs needed given the requirements and quantities. -/
def totalBulbs (req : BulbRequirement) (lights : CeilingLights) : Nat :=
  req.small * lights.small + req.medium * lights.medium + req.large * lights.large

/-- The main theorem stating the total number of bulbs needed. -/
theorem bulbs_needed :
  ∀ (req : BulbRequirement) (lights : CeilingLights),
    req.small = 1 ∧ req.medium = 2 ∧ req.large = 3 ∧
    lights.medium = 12 ∧
    lights.large = 2 * lights.medium ∧
    lights.small = lights.medium + 10 →
    totalBulbs req lights = 118 := by
  sorry


end NUMINAMATH_CALUDE_bulbs_needed_l2537_253796


namespace NUMINAMATH_CALUDE_investment_result_approx_17607_l2537_253739

/-- Calculates the final amount of an investment after tax and compound interest --/
def investment_after_tax (initial_investment : ℝ) (interest_rate : ℝ) (tax_rate : ℝ) (years : ℕ) : ℝ :=
  let compound_factor := 1 + interest_rate * (1 - tax_rate)
  initial_investment * compound_factor ^ years

/-- Theorem stating that the investment result is approximately $17,607 --/
theorem investment_result_approx_17607 :
  ∃ ε > 0, |investment_after_tax 15000 0.05 0.10 4 - 17607| < ε :=
sorry

end NUMINAMATH_CALUDE_investment_result_approx_17607_l2537_253739


namespace NUMINAMATH_CALUDE_successful_pair_existence_l2537_253775

/-- A pair of natural numbers is successful if their arithmetic mean and geometric mean are both natural numbers. -/
def IsSuccessfulPair (a b : ℕ) : Prop :=
  ∃ m g : ℕ, 2 * m = a + b ∧ g * g = a * b

theorem successful_pair_existence (m n k : ℕ) (h1 : m > n) (h2 : n > 0) (h3 : m > k) (h4 : k > 0)
  (h5 : IsSuccessfulPair (m + n) (m - n)) (h6 : m^2 - n^2 = k^2) :
  ∃ (a b : ℕ), a ≠ b ∧ IsSuccessfulPair a b ∧ 2 * m = a + b ∧ (a ≠ m + n ∨ b ≠ m - n) := by
  sorry

end NUMINAMATH_CALUDE_successful_pair_existence_l2537_253775


namespace NUMINAMATH_CALUDE_sqrt_x_minus_9_real_l2537_253702

theorem sqrt_x_minus_9_real (x : ℝ) : (∃ y : ℝ, y^2 = x - 9) ↔ x ≥ 9 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_9_real_l2537_253702


namespace NUMINAMATH_CALUDE_not_divisible_by_seven_l2537_253743

theorem not_divisible_by_seven (k : ℕ) : ¬(7 ∣ (2^(2*k - 1) + 2^k + 1)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_seven_l2537_253743


namespace NUMINAMATH_CALUDE_sqrt_two_irrational_l2537_253710

-- Define what it means for a real number to be rational
def IsRational (x : ℝ) : Prop :=
  ∃ (n d : ℤ), d ≠ 0 ∧ x = n / d

-- Define what it means for a real number to be irrational
def IsIrrational (x : ℝ) : Prop := ¬(IsRational x)

-- Theorem statement
theorem sqrt_two_irrational : IsIrrational (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_irrational_l2537_253710


namespace NUMINAMATH_CALUDE_trapezium_other_side_length_l2537_253793

theorem trapezium_other_side_length (a b h : ℝ) (area : ℝ) : 
  a = 20 → h = 15 → area = 285 → area = (a + b) * h / 2 → b = 18 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_other_side_length_l2537_253793


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_range_l2537_253755

open Real

theorem function_inequality_implies_a_range (a : ℝ) (h_a : a > 0) :
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 1 3 → x₂ ∈ Set.Icc 1 3 → x₁ ≠ x₂ →
    |x₁ + a * log x₁ - (x₂ + a * log x₂)| < |1 / x₁ - 1 / x₂|) →
  a < 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_range_l2537_253755


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l2537_253761

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sixth_term
  (a : ℕ → ℝ)
  (h_geo : is_geometric_sequence a)
  (h_roots : a 3 * a 7 = 256)
  (h_a4 : a 4 = 8) :
  a 6 = 32 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l2537_253761


namespace NUMINAMATH_CALUDE_fraction_equality_l2537_253734

theorem fraction_equality (x y : ℝ) : 
  (5 + 2*x) / (7 + 3*x + y) = (3 + 4*x) / (4 + 2*x + y) ↔ 
  8*x^2 + 19*x + 5*x*y = -1 - 5*y :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_l2537_253734


namespace NUMINAMATH_CALUDE_smallest_n_for_sqrt_difference_l2537_253764

theorem smallest_n_for_sqrt_difference (n : ℕ) : 
  (n > 0) → 
  (∀ m : ℕ, m > 0 → m < 626 → Real.sqrt m - Real.sqrt (m - 1) ≥ 0.02) → 
  (Real.sqrt 626 - Real.sqrt 625 < 0.02) → 
  (626 = n) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_sqrt_difference_l2537_253764


namespace NUMINAMATH_CALUDE_percentage_problem_l2537_253782

/-- Given a number N and a percentage P, this theorem proves that P is 20%
    when N is 580 and P% of N equals 30% of 120 plus 80. -/
theorem percentage_problem (N : ℝ) (P : ℝ) : 
  N = 580 → 
  (P / 100) * N = (30 / 100) * 120 + 80 → 
  P = 20 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l2537_253782


namespace NUMINAMATH_CALUDE_multiple_of_seven_l2537_253783

theorem multiple_of_seven : (2222^5555 + 5555^2222) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_seven_l2537_253783


namespace NUMINAMATH_CALUDE_magician_marbles_left_l2537_253753

theorem magician_marbles_left (red_initial : Nat) (blue_initial : Nat) 
  (red_taken : Nat) (blue_taken_multiplier : Nat) : 
  red_initial = 20 → 
  blue_initial = 30 → 
  red_taken = 3 → 
  blue_taken_multiplier = 4 → 
  (red_initial - red_taken) + (blue_initial - (blue_taken_multiplier * red_taken)) = 35 := by
  sorry

end NUMINAMATH_CALUDE_magician_marbles_left_l2537_253753


namespace NUMINAMATH_CALUDE_late_students_total_time_l2537_253757

theorem late_students_total_time (charlize_late : ℕ) 
  (h1 : charlize_late = 20)
  (ana_late : ℕ) 
  (h2 : ana_late = charlize_late + 5)
  (ben_late : ℕ) 
  (h3 : ben_late = charlize_late - 15)
  (clara_late : ℕ) 
  (h4 : clara_late = 2 * charlize_late)
  (daniel_late : ℕ) 
  (h5 : daniel_late = clara_late - 10) :
  charlize_late + ana_late + ben_late + clara_late + daniel_late = 120 := by
  sorry

end NUMINAMATH_CALUDE_late_students_total_time_l2537_253757


namespace NUMINAMATH_CALUDE_special_polynomial_value_l2537_253781

/-- A polynomial of degree n satisfying the given condition -/
def SpecialPolynomial (n : ℕ) : (ℕ → ℚ) := fun k => 1 / (Nat.choose (n+1) k)

/-- The theorem stating the value of p(n+1) for the special polynomial -/
theorem special_polynomial_value (n : ℕ) :
  let p := SpecialPolynomial n
  p (n+1) = if Even n then 1 else 0 := by
  sorry

end NUMINAMATH_CALUDE_special_polynomial_value_l2537_253781


namespace NUMINAMATH_CALUDE_food_company_inspection_l2537_253750

theorem food_company_inspection (large_companies medium_companies total_inspected medium_inspected : ℕ) 
  (h1 : large_companies = 4)
  (h2 : medium_companies = 20)
  (h3 : total_inspected = 40)
  (h4 : medium_inspected = 5) :
  ∃ (small_companies : ℕ), 
    small_companies = 136 ∧ 
    total_inspected = large_companies + medium_inspected + (total_inspected - large_companies - medium_inspected) :=
by sorry

end NUMINAMATH_CALUDE_food_company_inspection_l2537_253750
