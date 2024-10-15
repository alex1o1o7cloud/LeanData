import Mathlib

namespace NUMINAMATH_CALUDE_rectangular_solid_edge_sum_l3721_372172

theorem rectangular_solid_edge_sum (a r : ℝ) : 
  a > 0 ∧ r > 0 →
  (a / r) * a * (a * r) = 512 →
  2 * ((a^2 / r) + (a^2 * r) + a^2) = 320 →
  4 * (a / r + a + a * r) = 56 + 12 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_edge_sum_l3721_372172


namespace NUMINAMATH_CALUDE_random_number_table_sampling_sequence_l3721_372174

/-- Represents the steps in the sampling process -/
inductive SamplingStep
  | AssignNumbers
  | ObtainSamples
  | SelectStartingNumber

/-- Represents a sequence of sampling steps -/
def SamplingSequence := List SamplingStep

/-- The correct sampling sequence -/
def correctSequence : SamplingSequence :=
  [SamplingStep.AssignNumbers, SamplingStep.SelectStartingNumber, SamplingStep.ObtainSamples]

/-- Checks if a given sequence is valid for random number table sampling -/
def isValidSequence (seq : SamplingSequence) : Prop :=
  seq = correctSequence

theorem random_number_table_sampling_sequence :
  isValidSequence correctSequence :=
sorry

end NUMINAMATH_CALUDE_random_number_table_sampling_sequence_l3721_372174


namespace NUMINAMATH_CALUDE_last_triangle_perimeter_l3721_372178

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  positive : 0 < a ∧ 0 < b ∧ 0 < c
  inequality : a < b + c ∧ b < a + c ∧ c < a + b

/-- Constructs the next triangle in the sequence if it exists -/
def nextTriangle (T : Triangle) : Option Triangle := sorry

/-- The sequence of triangles starting from T₁ -/
def triangleSequence : ℕ → Option Triangle
  | 0 => some ⟨20, 21, 29, sorry, sorry⟩
  | n + 1 => match triangleSequence n with
    | none => none
    | some T => nextTriangle T

/-- The perimeter of a triangle -/
def perimeter (T : Triangle) : ℝ := T.a + T.b + T.c

/-- Finds the last valid triangle in the sequence -/
def lastTriangle : Option Triangle := sorry

theorem last_triangle_perimeter :
  ∀ T, lastTriangle = some T → perimeter T = 35 := by sorry

end NUMINAMATH_CALUDE_last_triangle_perimeter_l3721_372178


namespace NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_500_l3721_372129

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℕ, (n : ℝ) > Real.sqrt 500 ∧ ∀ m : ℕ, (m : ℝ) > Real.sqrt 500 → m ≥ n := by
  sorry

end NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_500_l3721_372129


namespace NUMINAMATH_CALUDE_franks_breakfast_shopping_l3721_372110

/-- Frank's breakfast shopping problem -/
theorem franks_breakfast_shopping
  (num_buns : ℕ)
  (num_milk_bottles : ℕ)
  (milk_price : ℚ)
  (egg_price_multiplier : ℕ)
  (total_paid : ℚ)
  (h_num_buns : num_buns = 10)
  (h_num_milk_bottles : num_milk_bottles = 2)
  (h_milk_price : milk_price = 2)
  (h_egg_price : egg_price_multiplier = 3)
  (h_total_paid : total_paid = 11)
  : (total_paid - (num_milk_bottles * milk_price + egg_price_multiplier * milk_price)) / num_buns = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_franks_breakfast_shopping_l3721_372110


namespace NUMINAMATH_CALUDE_a_can_be_any_real_l3721_372192

theorem a_can_be_any_real : ∀ (a b c d e : ℝ), 
  bd ≠ 0 → e ≠ 0 → (a / b + e < -(c / d)) → 
  (∃ (a_pos a_neg a_zero : ℝ), 
    (a_pos > 0 ∧ a_pos / b + e < -(c / d)) ∧
    (a_neg < 0 ∧ a_neg / b + e < -(c / d)) ∧
    (a_zero = 0 ∧ a_zero / b + e < -(c / d))) :=
by sorry

end NUMINAMATH_CALUDE_a_can_be_any_real_l3721_372192


namespace NUMINAMATH_CALUDE_total_tuition_correct_l3721_372185

/-- The total tuition fee that Bran needs to pay -/
def total_tuition : ℝ := 90

/-- Bran's monthly earnings from his part-time job -/
def monthly_earnings : ℝ := 15

/-- The percentage of tuition covered by Bran's scholarship -/
def scholarship_percentage : ℝ := 0.3

/-- The number of months Bran has to pay his tuition -/
def payment_period : ℕ := 3

/-- The amount Bran still needs to pay after scholarship and earnings -/
def remaining_payment : ℝ := 18

/-- Theorem stating that the total tuition is correct given the conditions -/
theorem total_tuition_correct :
  (1 - scholarship_percentage) * total_tuition - 
  (monthly_earnings * payment_period) = remaining_payment :=
by sorry

end NUMINAMATH_CALUDE_total_tuition_correct_l3721_372185


namespace NUMINAMATH_CALUDE_counterexample_fifth_power_l3721_372112

theorem counterexample_fifth_power : 144^5 + 121^5 + 95^5 + 30^5 = 159^5 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_fifth_power_l3721_372112


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3721_372113

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  x₁ = -3 ∧ x₂ = 5 ∧ 
  (x₁^2 - 2*x₁ - 15 = 0) ∧ 
  (x₂^2 - 2*x₂ - 15 = 0) ∧
  (∀ x : ℝ, x^2 - 2*x - 15 = 0 → x = x₁ ∨ x = x₂) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3721_372113


namespace NUMINAMATH_CALUDE_linear_function_not_in_fourth_quadrant_l3721_372135

-- Define the linear function
def f (k : ℝ) (x : ℝ) : ℝ := (k - 2) * x + k

-- Theorem statement
theorem linear_function_not_in_fourth_quadrant (k : ℝ) (h1 : k ≠ 2) :
  (∀ x > 0, f k x ≥ 0) → k > 2 := by
  sorry


end NUMINAMATH_CALUDE_linear_function_not_in_fourth_quadrant_l3721_372135


namespace NUMINAMATH_CALUDE_decimal_to_percentage_l3721_372101

theorem decimal_to_percentage (x : ℝ) : x = 5.02 → (x * 100 : ℝ) = 502 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_percentage_l3721_372101


namespace NUMINAMATH_CALUDE_limit_a_over_3n_l3721_372109

def S (n : ℕ) : ℝ := -3 * (n ^ 2 : ℝ) + 2 * n + 1

def a (n : ℕ) : ℝ := S (n + 1) - S n

theorem limit_a_over_3n :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n / (3 * (n + 1)) + 2| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_a_over_3n_l3721_372109


namespace NUMINAMATH_CALUDE_complement_B_intersect_A_l3721_372121

open Set

universe u

def U : Set ℝ := univ
def A : Set ℝ := {x | |x| < 1}
def B : Set ℝ := {x | x > -1/2}

theorem complement_B_intersect_A :
  (U \ B) ∩ A = {x : ℝ | -1 < x ∧ x ≤ -1/2} := by sorry

end NUMINAMATH_CALUDE_complement_B_intersect_A_l3721_372121


namespace NUMINAMATH_CALUDE_abs_eq_neg_implies_nonpositive_l3721_372160

theorem abs_eq_neg_implies_nonpositive (a : ℝ) : |a| = -a → a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_abs_eq_neg_implies_nonpositive_l3721_372160


namespace NUMINAMATH_CALUDE_correct_stratified_sampling_l3721_372128

/-- Represents the number of students sampled from a grade -/
structure SampledStudents :=
  (freshmen : ℕ)
  (sophomores : ℕ)
  (juniors : ℕ)

/-- Calculates the stratified sample size for a grade -/
def stratifiedSampleSize (gradeTotal : ℕ) (totalStudents : ℕ) (sampleSize : ℕ) : ℕ :=
  (gradeTotal * sampleSize) / totalStudents

/-- Theorem stating the correct stratified sampling for the given school -/
theorem correct_stratified_sampling :
  let totalStudents : ℕ := 2700
  let freshmenTotal : ℕ := 900
  let sophomoresTotal : ℕ := 1200
  let juniorsTotal : ℕ := 600
  let sampleSize : ℕ := 135
  let result : SampledStudents := {
    freshmen := stratifiedSampleSize freshmenTotal totalStudents sampleSize,
    sophomores := stratifiedSampleSize sophomoresTotal totalStudents sampleSize,
    juniors := stratifiedSampleSize juniorsTotal totalStudents sampleSize
  }
  result.freshmen = 45 ∧ result.sophomores = 60 ∧ result.juniors = 30 :=
by sorry

end NUMINAMATH_CALUDE_correct_stratified_sampling_l3721_372128


namespace NUMINAMATH_CALUDE_relative_errors_equal_l3721_372122

theorem relative_errors_equal (length1 length2 error1 error2 : ℝ) 
  (h1 : length1 = 25)
  (h2 : length2 = 150)
  (h3 : error1 = 0.05)
  (h4 : error2 = 0.3) : 
  (error1 / length1) = (error2 / length2) := by
  sorry

end NUMINAMATH_CALUDE_relative_errors_equal_l3721_372122


namespace NUMINAMATH_CALUDE_dog_cat_sum_l3721_372164

/-- Represents a three-digit number composed of digits D, O, and G -/
def DOG (D O G : Nat) : Nat := 100 * D + 10 * O + G

/-- Represents a three-digit number composed of digits C, A, and T -/
def CAT (C A T : Nat) : Nat := 100 * C + 10 * A + T

/-- Theorem stating that if DOG + CAT = 1000 for different digits, then the sum of all digits is 28 -/
theorem dog_cat_sum (D O G C A T : Nat) 
  (h1 : D ≠ O ∧ D ≠ G ∧ D ≠ C ∧ D ≠ A ∧ D ≠ T ∧ 
        O ≠ G ∧ O ≠ C ∧ O ≠ A ∧ O ≠ T ∧ 
        G ≠ C ∧ G ≠ A ∧ G ≠ T ∧ 
        C ≠ A ∧ C ≠ T ∧ 
        A ≠ T)
  (h2 : D < 10 ∧ O < 10 ∧ G < 10 ∧ C < 10 ∧ A < 10 ∧ T < 10)
  (h3 : DOG D O G + CAT C A T = 1000) :
  D + O + G + C + A + T = 28 := by
  sorry

end NUMINAMATH_CALUDE_dog_cat_sum_l3721_372164


namespace NUMINAMATH_CALUDE_quadratic_root_property_l3721_372181

theorem quadratic_root_property (α β : ℝ) : 
  α^2 - 3*α - 4 = 0 → β^2 - 3*β - 4 = 0 → α^2 + α*β - 3*α = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l3721_372181


namespace NUMINAMATH_CALUDE_vector_subtraction_l3721_372166

def a : Fin 2 → ℝ := ![3, 2]
def b : Fin 2 → ℝ := ![0, -1]

theorem vector_subtraction :
  (3 • b - a) = ![(-3 : ℝ), -5] := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_l3721_372166


namespace NUMINAMATH_CALUDE_tileIV_in_rectangleD_l3721_372150

-- Define the structure for a tile
structure Tile where
  top : ℕ
  right : ℕ
  bottom : ℕ
  left : ℕ

-- Define the tiles
def tileI : Tile := ⟨3, 1, 4, 2⟩
def tileII : Tile := ⟨2, 3, 1, 5⟩
def tileIII : Tile := ⟨4, 0, 3, 1⟩
def tileIV : Tile := ⟨5, 4, 2, 0⟩

-- Define the set of all tiles
def allTiles : Set Tile := {tileI, tileII, tileIII, tileIV}

-- Define a function to check if two tiles can be adjacent
def canBeAdjacent (t1 t2 : Tile) : Bool :=
  (t1.right = t2.left) ∨ (t1.left = t2.right) ∨ (t1.top = t2.bottom) ∨ (t1.bottom = t2.top)

-- Theorem: Tile IV must be placed in Rectangle D
theorem tileIV_in_rectangleD :
  ∀ (t : Tile), t ∈ allTiles → t ≠ tileIV →
    ∃ (t' : Tile), t' ∈ allTiles ∧ t' ≠ t ∧ t' ≠ tileIV ∧ canBeAdjacent t t' = true →
      ¬∃ (t'' : Tile), t'' ∈ allTiles ∧ t'' ≠ tileIV ∧ canBeAdjacent tileIV t'' = true :=
sorry

end NUMINAMATH_CALUDE_tileIV_in_rectangleD_l3721_372150


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3721_372198

theorem complex_number_in_first_quadrant :
  let z : ℂ := (3 + Complex.I) / (1 - Complex.I)
  (z.re > 0) ∧ (z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3721_372198


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3721_372144

theorem polynomial_coefficient_sum (a₄ a₃ a₂ a₁ a₀ : ℝ) :
  (∀ x : ℝ, (x - 1)^4 = a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₄ - a₃ + a₂ - a₁ = 15 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3721_372144


namespace NUMINAMATH_CALUDE_fantasia_license_plates_l3721_372156

/-- Represents the number of available letters in the alphabet. -/
def num_letters : ℕ := 26

/-- Represents the number of available digits. -/
def num_digits : ℕ := 10

/-- Calculates the number of valid license plates in Fantasia. -/
def count_license_plates : ℕ :=
  num_letters * num_letters * num_letters * num_digits * (num_digits - 1) * (num_digits - 2)

/-- Theorem stating that the number of valid license plates in Fantasia is 15,818,400. -/
theorem fantasia_license_plates :
  count_license_plates = 15818400 :=
by sorry

end NUMINAMATH_CALUDE_fantasia_license_plates_l3721_372156


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3721_372108

theorem sufficient_not_necessary_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ a b, a > 0 → b > 0 → a + b = 2 → a * b ≤ 1) ∧
  (∃ a b, a > 0 ∧ b > 0 ∧ a * b ≤ 1 ∧ a + b ≠ 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3721_372108


namespace NUMINAMATH_CALUDE_symmetry_implies_p_plus_r_zero_l3721_372154

/-- Represents a curve of the form y = (px + 2q) / (rx + 2s) -/
structure Curve where
  p : ℝ
  q : ℝ
  r : ℝ
  s : ℝ
  p_nonzero : p ≠ 0
  q_nonzero : q ≠ 0
  r_nonzero : r ≠ 0
  s_nonzero : s ≠ 0

/-- The property of y = 2x being an axis of symmetry for the curve -/
def is_axis_of_symmetry (c : Curve) : Prop :=
  ∀ x y : ℝ, y = (c.p * x + 2 * c.q) / (c.r * x + 2 * c.s) →
    y = (c.p * (y / 2) + 2 * c.q) / (c.r * (y / 2) + 2 * c.s)

/-- The main theorem stating that if y = 2x is an axis of symmetry, then p + r = 0 -/
theorem symmetry_implies_p_plus_r_zero (c : Curve) :
  is_axis_of_symmetry c → c.p + c.r = 0 := by sorry

end NUMINAMATH_CALUDE_symmetry_implies_p_plus_r_zero_l3721_372154


namespace NUMINAMATH_CALUDE_forty_second_card_l3721_372153

def card_sequence : ℕ → ℕ :=
  fun n => (n - 1) % 13 + 1

theorem forty_second_card :
  card_sequence 42 = 3 := by
  sorry

end NUMINAMATH_CALUDE_forty_second_card_l3721_372153


namespace NUMINAMATH_CALUDE_quadratic_extrema_l3721_372176

-- Define the function
def f (x : ℝ) : ℝ := (x - 3)^2 - 1

-- Define the domain
def domain : Set ℝ := {x | 1 ≤ x ∧ x ≤ 4}

theorem quadratic_extrema :
  ∃ (min max : ℝ), 
    (∀ x ∈ domain, f x ≥ min) ∧
    (∃ x ∈ domain, f x = min) ∧
    (∀ x ∈ domain, f x ≤ max) ∧
    (∃ x ∈ domain, f x = max) ∧
    min = -1 ∧ max = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_extrema_l3721_372176


namespace NUMINAMATH_CALUDE_jan_2022_is_saturday_l3721_372133

/-- Enumeration of days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to advance a day by n days -/
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDay (nextDay d) n

/-- Theorem: If January 2021 has exactly five Fridays, five Saturdays, and five Sundays,
    then January 1, 2022 falls on a Saturday -/
theorem jan_2022_is_saturday
  (h : ∃ (first_day : DayOfWeek),
       (advanceDay first_day 0 = DayOfWeek.Friday ∧
        advanceDay first_day 1 = DayOfWeek.Saturday ∧
        advanceDay first_day 2 = DayOfWeek.Sunday) ∧
       (∀ (n : Nat), n < 31 → 
        (advanceDay first_day n = DayOfWeek.Friday ∨
         advanceDay first_day n = DayOfWeek.Saturday ∨
         advanceDay first_day n = DayOfWeek.Sunday) →
        (advanceDay first_day (n + 7) = advanceDay first_day n))) :
  advanceDay DayOfWeek.Friday 365 = DayOfWeek.Saturday := by
  sorry


end NUMINAMATH_CALUDE_jan_2022_is_saturday_l3721_372133


namespace NUMINAMATH_CALUDE_complex_equation_sum_l3721_372158

theorem complex_equation_sum (a b : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : (a + 2 * i) / i = b + i) : a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l3721_372158


namespace NUMINAMATH_CALUDE_chili_beans_cans_l3721_372180

-- Define the ratio of tomato soup cans to chili beans cans
def soup_to_beans_ratio : ℚ := 1 / 2

-- Define the total number of cans
def total_cans : ℕ := 12

-- Theorem to prove
theorem chili_beans_cans (t c : ℕ) 
  (h1 : t + c = total_cans) 
  (h2 : c = 2 * t) : c = 8 := by
  sorry

end NUMINAMATH_CALUDE_chili_beans_cans_l3721_372180


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3721_372130

-- Define sets A and B
def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | x < 3}

-- Theorem stating the intersection of A and B
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 1 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3721_372130


namespace NUMINAMATH_CALUDE_bacteria_growth_proof_l3721_372175

/-- The increase in bacteria population given initial and final counts -/
def bacteria_increase (initial final : ℕ) : ℕ :=
  final - initial

/-- Theorem stating the increase in bacteria population for the given scenario -/
theorem bacteria_growth_proof (initial final : ℕ) 
  (h1 : initial = 600) 
  (h2 : final = 8917) : 
  bacteria_increase initial final = 8317 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_growth_proof_l3721_372175


namespace NUMINAMATH_CALUDE_tan_pi_minus_alpha_eq_two_implies_ratio_eq_one_fourth_l3721_372137

theorem tan_pi_minus_alpha_eq_two_implies_ratio_eq_one_fourth
  (α : ℝ) (h : Real.tan (π - α) = 2) :
  (Real.sin (π/2 + α) + Real.sin (π - α)) /
  (Real.cos (3*π/2 + α) + 2 * Real.cos (π + α)) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_pi_minus_alpha_eq_two_implies_ratio_eq_one_fourth_l3721_372137


namespace NUMINAMATH_CALUDE_emerie_quarters_l3721_372187

/-- Represents the number of coins of a specific type --/
structure CoinCount where
  dimes : Nat
  nickels : Nat
  quarters : Nat

/-- The total number of coins --/
def totalCoins (c : CoinCount) : Nat :=
  c.dimes + c.nickels + c.quarters

/-- Emerie's coin count --/
def emerie : CoinCount :=
  { dimes := 7, nickels := 5, quarters := 0 }

/-- Zain's coin count --/
def zain (e : CoinCount) : CoinCount :=
  { dimes := e.dimes + 10, nickels := e.nickels + 10, quarters := e.quarters + 10 }

theorem emerie_quarters : 
  totalCoins (zain emerie) = 48 → emerie.quarters = 6 := by
  sorry

end NUMINAMATH_CALUDE_emerie_quarters_l3721_372187


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l3721_372173

theorem largest_constant_inequality (C : ℝ) :
  (∀ x y z : ℝ, x^2 + y^2 + z^2 + 1 ≥ C*(x + y + z)) ↔ C ≤ 2 / Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l3721_372173


namespace NUMINAMATH_CALUDE_megan_water_consumption_l3721_372118

/-- The number of glasses of water Megan drinks in a given time period -/
def glasses_of_water (minutes : ℕ) : ℕ :=
  minutes / 20

theorem megan_water_consumption : glasses_of_water 220 = 11 := by
  sorry

end NUMINAMATH_CALUDE_megan_water_consumption_l3721_372118


namespace NUMINAMATH_CALUDE_foreign_exchange_earnings_equation_l3721_372146

/-- Represents the monthly decline rate as a real number between 0 and 1 -/
def monthly_decline_rate : ℝ := sorry

/-- Initial foreign exchange earnings in July (in millions of USD) -/
def initial_earnings : ℝ := 200

/-- Foreign exchange earnings in September (in millions of USD) -/
def final_earnings : ℝ := 98

/-- The number of months between July and September -/
def months_elapsed : ℕ := 2

theorem foreign_exchange_earnings_equation :
  initial_earnings * (1 - monthly_decline_rate) ^ months_elapsed = final_earnings :=
sorry

end NUMINAMATH_CALUDE_foreign_exchange_earnings_equation_l3721_372146


namespace NUMINAMATH_CALUDE_sum_of_cubes_zero_l3721_372165

theorem sum_of_cubes_zero (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a / (b - c)^3 + b / (c - a)^3 + c / (a - b)^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_zero_l3721_372165


namespace NUMINAMATH_CALUDE_square_difference_601_597_l3721_372138

theorem square_difference_601_597 : (601 : ℤ)^2 - (597 : ℤ)^2 = 4792 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_601_597_l3721_372138


namespace NUMINAMATH_CALUDE_village_population_l3721_372111

theorem village_population (P : ℕ) : 
  (P : ℝ) * (1 - 0.05) * (1 - 0.15) = 3294 → P = 4080 := by
sorry

end NUMINAMATH_CALUDE_village_population_l3721_372111


namespace NUMINAMATH_CALUDE_exponential_equation_solution_l3721_372152

theorem exponential_equation_solution :
  ∃ y : ℝ, (20 : ℝ)^y * 200^(3*y) = 8000^7 ∧ y = 3 :=
by sorry

end NUMINAMATH_CALUDE_exponential_equation_solution_l3721_372152


namespace NUMINAMATH_CALUDE_max_value_quadratic_l3721_372147

theorem max_value_quadratic (x : ℝ) :
  let f : ℝ → ℝ := fun x => 10 * x - 2 * x^2
  ∃ (max_val : ℝ), max_val = 12.5 ∧ ∀ y : ℝ, f y ≤ max_val :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l3721_372147


namespace NUMINAMATH_CALUDE_no_all_power_of_five_l3721_372161

theorem no_all_power_of_five : ¬∃ (a : Fin 2018 → ℕ), ∀ i : Fin 2018, 
  ∃ k : ℕ, (a i)^2018 + a (i.succ) = 5^k := by
  sorry

end NUMINAMATH_CALUDE_no_all_power_of_five_l3721_372161


namespace NUMINAMATH_CALUDE_no_solution_sqrt_equation_l3721_372139

theorem no_solution_sqrt_equation :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 3 → Real.sqrt (x + 1) + Real.sqrt (3 - x) < 17 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_sqrt_equation_l3721_372139


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3721_372188

theorem complex_equation_solution (a : ℂ) :
  (1 + a * Complex.I) / (2 + Complex.I) = 1 + 2 * Complex.I → a = 5 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3721_372188


namespace NUMINAMATH_CALUDE_intersection_M_complement_N_l3721_372107

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x^2 - 2*x - 8 ≤ 0}
def N : Set ℝ := {x : ℝ | Real.exp (Real.log 2 * (1 - x)) > 1}

-- Define the theorem
theorem intersection_M_complement_N :
  M ∩ (Set.univ \ N) = Set.Icc (-2) 1 :=
sorry

end NUMINAMATH_CALUDE_intersection_M_complement_N_l3721_372107


namespace NUMINAMATH_CALUDE_quadratic_root_property_l3721_372141

theorem quadratic_root_property (m : ℝ) : m^2 - m - 3 = 0 → 2023 - m^2 + m = 2020 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l3721_372141


namespace NUMINAMATH_CALUDE_f_neg_two_l3721_372190

def f (x : ℝ) : ℝ := x^2 + 3*x - 5

theorem f_neg_two : f (-2) = -7 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_two_l3721_372190


namespace NUMINAMATH_CALUDE_rational_equation_result_l3721_372179

theorem rational_equation_result (x y : ℚ) 
  (h : |x + 2017| + (y - 2017)^2 = 0) : 
  (x / y)^2017 = -1 := by
  sorry

end NUMINAMATH_CALUDE_rational_equation_result_l3721_372179


namespace NUMINAMATH_CALUDE_positive_roots_l3721_372114

theorem positive_roots (x y z : ℝ) 
  (sum_pos : x + y + z > 0) 
  (sum_prod_pos : x*y + y*z + z*x > 0) 
  (prod_pos : x*y*z > 0) : 
  x > 0 ∧ y > 0 ∧ z > 0 := by
sorry

end NUMINAMATH_CALUDE_positive_roots_l3721_372114


namespace NUMINAMATH_CALUDE_pencil_count_l3721_372104

/-- The number of pencils Cindi bought -/
def cindi_pencils : ℕ := 75

/-- The number of pencils Marcia bought -/
def marcia_pencils : ℕ := 112

/-- The number of pencils Donna bought -/
def donna_pencils : ℕ := 448

/-- The number of pencils Bob bought -/
def bob_pencils : ℕ := cindi_pencils + 20

/-- The total number of pencils bought by Donna, Marcia, and Bob -/
def total_pencils : ℕ := donna_pencils + marcia_pencils + bob_pencils

theorem pencil_count : total_pencils = 655 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l3721_372104


namespace NUMINAMATH_CALUDE_difference_of_squares_l3721_372182

theorem difference_of_squares (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3721_372182


namespace NUMINAMATH_CALUDE_smarties_remainder_l3721_372103

theorem smarties_remainder (m : ℕ) (h : m % 11 = 5) : (2 * m) % 11 = 10 := by
  sorry

end NUMINAMATH_CALUDE_smarties_remainder_l3721_372103


namespace NUMINAMATH_CALUDE_sequence_sum_l3721_372119

theorem sequence_sum (A B C D E F G H I J : ℤ) : 
  E = 8 ∧ 
  A + B + C = 27 ∧ 
  B + C + D = 27 ∧ 
  C + D + E = 27 ∧ 
  D + E + F = 27 ∧ 
  E + F + G = 27 ∧ 
  F + G + H = 27 ∧ 
  G + H + I = 27 ∧ 
  H + I + J = 27 
  → A + J = -27 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l3721_372119


namespace NUMINAMATH_CALUDE_statement_A_statement_B_statement_C_statement_D_l3721_372194

-- Define the curve C
def C (m n x y : ℝ) : Prop := m * x^2 + n * y^2 = 1

-- Statement A
theorem statement_A (m n : ℝ) (h1 : n > m) (h2 : m > 0) :
  ¬ (∃ a b : ℝ, a > b ∧ a > 0 ∧ b > 0 ∧
    ∀ x y : ℝ, C m n x y ↔ (x^2 / a^2 + y^2 / b^2 = 1) ∧
    (∃ c : ℝ, c > 0 ∧ a^2 = b^2 + c^2)) :=
sorry

-- Statement B
theorem statement_B (n : ℝ) (h : n > 0) :
  ∃ y1 y2 : ℝ, y1 ≠ y2 ∧ ∀ x : ℝ, C 0 n x y1 ∧ C 0 n x y2 :=
sorry

-- Statement C
theorem statement_C (m n : ℝ) (h : m * n < 0) :
  ∃ k : ℝ, k > 0 ∧ ∀ x y : ℝ, C m n x y →
    (y - k * x) * (y + k * x) ≤ 0 ∧ k^2 = -m/n :=
sorry

-- Statement D
theorem statement_D (n : ℝ) (h : n > 0) :
  ¬ (∀ x y : ℝ, C n n x y ↔ x^2 + y^2 = n) :=
sorry

end NUMINAMATH_CALUDE_statement_A_statement_B_statement_C_statement_D_l3721_372194


namespace NUMINAMATH_CALUDE_unique_pair_solution_l3721_372169

theorem unique_pair_solution : 
  ∃! (p n : ℕ), 
    n > p ∧ 
    p.Prime ∧ 
    (∃ k : ℕ, k > 0 ∧ n^(n - p) = k^n) ∧ 
    p = 2 ∧ 
    n = 4 := by
sorry

end NUMINAMATH_CALUDE_unique_pair_solution_l3721_372169


namespace NUMINAMATH_CALUDE_vector_sum_zero_l3721_372171

variable {V : Type*} [AddCommGroup V]

def vector (A B : V) : V := B - A

theorem vector_sum_zero (M B O A C D : V) : 
  (vector M B + vector B O + vector O M = 0) ∧
  (vector O B + vector O C + vector B O + vector C O = 0) ∧
  (vector A B - vector A C + vector B D - vector C D = 0) := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_zero_l3721_372171


namespace NUMINAMATH_CALUDE_sphere_to_cone_height_l3721_372163

/-- Given a sphere with diameter 6 cm and a cone with base diameter 12 cm,
    if their volumes are equal, then the height of the cone is 3 cm. -/
theorem sphere_to_cone_height (sphere_diameter : ℝ) (cone_base_diameter : ℝ) (cone_height : ℝ) :
  sphere_diameter = 6 →
  cone_base_diameter = 12 →
  (4 / 3) * Real.pi * (sphere_diameter / 2) ^ 3 = (1 / 3) * Real.pi * (cone_base_diameter / 2) ^ 2 * cone_height →
  cone_height = 3 := by
  sorry

#check sphere_to_cone_height

end NUMINAMATH_CALUDE_sphere_to_cone_height_l3721_372163


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_sup_ratio_l3721_372143

/-- A quadrilateral inscribed in a unit circle with two parallel sides -/
structure InscribedQuadrilateral where
  /-- The difference between the lengths of the parallel sides -/
  d : ℝ
  /-- The distance from the intersection of the diagonals to the center of the circle -/
  h : ℝ
  /-- The difference d is positive -/
  d_pos : d > 0
  /-- The quadrilateral is inscribed in a unit circle -/
  h_bound : h ≤ 1

/-- The supremum of d/h for inscribed quadrilaterals is 2 -/
theorem inscribed_quadrilateral_sup_ratio :
  ∀ ε > 0, ∃ q : InscribedQuadrilateral, q.d / q.h > 2 - ε ∧ ∀ q' : InscribedQuadrilateral, q'.d / q'.h ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_sup_ratio_l3721_372143


namespace NUMINAMATH_CALUDE_sum_with_divisibility_conditions_l3721_372149

theorem sum_with_divisibility_conditions : 
  ∃ (a b : ℕ), a + b = 316 ∧ a % 13 = 0 ∧ b % 11 = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_with_divisibility_conditions_l3721_372149


namespace NUMINAMATH_CALUDE_f_at_two_l3721_372126

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom monotonic_increasing : Monotone f
axiom functional_equation : ∀ x : ℝ, f (f x - Real.exp x) = Real.exp 1 + 1

-- State the theorem
theorem f_at_two (f : ℝ → ℝ) (h1 : Monotone f) (h2 : ∀ x : ℝ, f (f x - Real.exp x) = Real.exp 1 + 1) :
  f 2 = Real.exp 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_f_at_two_l3721_372126


namespace NUMINAMATH_CALUDE_base_thirteen_unique_l3721_372145

/-- Converts a list of digits in base b to its decimal representation -/
def toDecimal (digits : List Nat) (b : Nat) : Nat :=
  digits.foldl (fun acc d => acc * b + d) 0

/-- Theorem stating that 13 is the unique base for which the equation holds -/
theorem base_thirteen_unique :
  ∃! b : Nat, b > 1 ∧ 
    toDecimal [5, 3, 2, 4] b + toDecimal [6, 4, 7, 3] b = toDecimal [1, 2, 5, 3, 2] b :=
by sorry

end NUMINAMATH_CALUDE_base_thirteen_unique_l3721_372145


namespace NUMINAMATH_CALUDE_closest_fraction_l3721_372106

def fractions : List ℚ := [1/4, 1/5, 1/6, 1/7, 1/8]
def team_gamma_fraction : ℚ := 13/80

theorem closest_fraction :
  ∀ f ∈ fractions, |team_gamma_fraction - 1/6| ≤ |team_gamma_fraction - f| :=
by sorry

end NUMINAMATH_CALUDE_closest_fraction_l3721_372106


namespace NUMINAMATH_CALUDE_common_difference_is_five_l3721_372116

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- The common difference of an arithmetic sequence -/
def commonDifference (seq : ArithmeticSequence) : ℝ :=
  seq.a 2 - seq.a 1

/-- Theorem: The common difference is 5 given the conditions -/
theorem common_difference_is_five (seq : ArithmeticSequence)
  (h1 : seq.S 17 = 255)
  (h2 : seq.a 10 = 20) :
  commonDifference seq = 5 := by
  sorry

#check common_difference_is_five

end NUMINAMATH_CALUDE_common_difference_is_five_l3721_372116


namespace NUMINAMATH_CALUDE_a_spending_percentage_l3721_372102

/-- Proves that A spends 95% of his salary given the conditions of the problem -/
theorem a_spending_percentage (total_salary : ℝ) (a_salary : ℝ) (b_spending_percentage : ℝ) :
  total_salary = 6000 →
  a_salary = 4500 →
  b_spending_percentage = 0.85 →
  let b_salary := total_salary - a_salary
  let a_savings := a_salary * (1 - (95 / 100))
  let b_savings := b_salary * (1 - b_spending_percentage)
  a_savings = b_savings :=
by sorry

end NUMINAMATH_CALUDE_a_spending_percentage_l3721_372102


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3721_372170

theorem sufficient_not_necessary (a : ℝ) :
  (∀ a, a > 3 → ∃ x : ℝ, x^2 + a*x + 1 < 0) ∧
  (∃ a, (∃ x : ℝ, x^2 + a*x + 1 < 0) ∧ ¬(a > 3)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3721_372170


namespace NUMINAMATH_CALUDE_trigonometric_system_solution_l3721_372132

theorem trigonometric_system_solution (x y : ℝ) :
  (Real.sin x + Real.cos y = 0) ∧ (Real.sin x ^ 2 + Real.cos y ^ 2 = 1/2) →
  (∃ (k n : ℤ), 
    ((x = (-1)^(k+1) * Real.pi/6 + Real.pi * k ∧ y = Real.pi/3 + 2*Real.pi*n) ∨
     (x = (-1)^(k+1) * Real.pi/6 + Real.pi * k ∧ y = -Real.pi/3 + 2*Real.pi*n)) ∨
    ((x = (-1)^k * Real.pi/6 + Real.pi * k ∧ y = 2*Real.pi/3 + 2*Real.pi*n) ∨
     (x = (-1)^k * Real.pi/6 + Real.pi * k ∧ y = -2*Real.pi/3 + 2*Real.pi*n))) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_system_solution_l3721_372132


namespace NUMINAMATH_CALUDE_x_minus_y_equals_three_l3721_372183

theorem x_minus_y_equals_three (x y : ℝ) 
  (h1 : x + y = 8) 
  (h2 : x^2 - y^2 = 24) : 
  x - y = 3 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_three_l3721_372183


namespace NUMINAMATH_CALUDE_repeating_decimal_difference_l3721_372105

theorem repeating_decimal_difference : 
  let x : ℚ := 72 / 99  -- $0.\overline{72}$ as a fraction
  let y : ℚ := 72 / 100 -- $0.72$ as a fraction
  x - y = 2 / 275 := by
sorry

end NUMINAMATH_CALUDE_repeating_decimal_difference_l3721_372105


namespace NUMINAMATH_CALUDE_petyas_addition_mistake_l3721_372177

theorem petyas_addition_mistake :
  ∃ (x y : ℕ) (c : Fin 10),
    x + y = 12345 ∧
    (10 * x + c.val) + y = 44444 ∧
    x = 3566 ∧
    y = 8779 := by
  sorry

end NUMINAMATH_CALUDE_petyas_addition_mistake_l3721_372177


namespace NUMINAMATH_CALUDE_hotel_room_encoding_l3721_372100

theorem hotel_room_encoding (x : ℕ) : 
  1 ≤ x ∧ x ≤ 30 ∧ x % 5 = 1 ∧ x % 7 = 6 → x = 13 := by
  sorry

end NUMINAMATH_CALUDE_hotel_room_encoding_l3721_372100


namespace NUMINAMATH_CALUDE_farmers_animal_purchase_l3721_372120

/-- The farmer's animal purchase problem -/
theorem farmers_animal_purchase
  (total : ℕ) (goat_pig_sheep : ℕ) (cow_pig_sheep : ℕ) (goat_pig : ℕ)
  (h1 : total = 1325)
  (h2 : goat_pig_sheep = 425)
  (h3 : cow_pig_sheep = 1225)
  (h4 : goat_pig = 275) :
  ∃ (cow goat sheep pig : ℕ),
    cow + goat + sheep + pig = total ∧
    goat + sheep + pig = goat_pig_sheep ∧
    cow + sheep + pig = cow_pig_sheep ∧
    goat + pig = goat_pig ∧
    cow = 900 ∧ goat = 100 ∧ sheep = 150 ∧ pig = 175 := by
  sorry


end NUMINAMATH_CALUDE_farmers_animal_purchase_l3721_372120


namespace NUMINAMATH_CALUDE_toys_per_rabbit_l3721_372136

-- Define the number of rabbits
def num_rabbits : ℕ := 16

-- Define the number of toys bought on Monday
def monday_toys : ℕ := 6

-- Define the number of toys bought on Wednesday
def wednesday_toys : ℕ := 2 * monday_toys

-- Define the number of toys bought on Friday
def friday_toys : ℕ := 4 * monday_toys

-- Define the number of toys bought on Saturday
def saturday_toys : ℕ := wednesday_toys / 2

-- Define the total number of toys
def total_toys : ℕ := monday_toys + wednesday_toys + friday_toys + saturday_toys

-- Theorem statement
theorem toys_per_rabbit : total_toys / num_rabbits = 3 := by
  sorry

end NUMINAMATH_CALUDE_toys_per_rabbit_l3721_372136


namespace NUMINAMATH_CALUDE_muffin_count_arthur_muffins_l3721_372167

/-- The total number of muffins Arthur wants to have -/
def total_muffins (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem stating that the total number of muffins is the sum of initial and additional muffins -/
theorem muffin_count (initial : ℕ) (additional : ℕ) :
  total_muffins initial additional = initial + additional :=
by sorry

/-- Theorem proving the specific case in the problem -/
theorem arthur_muffins :
  total_muffins 35 48 = 83 :=
by sorry

end NUMINAMATH_CALUDE_muffin_count_arthur_muffins_l3721_372167


namespace NUMINAMATH_CALUDE_greatest_integer_radius_l3721_372148

theorem greatest_integer_radius (A : ℝ) (h1 : 50 * Real.pi < A) (h2 : A < 75 * Real.pi) :
  ∃ (r : ℕ), r * r * Real.pi = A ∧ r ≤ 8 ∧ ∀ (s : ℕ), s * s * Real.pi = A → s ≤ r :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_radius_l3721_372148


namespace NUMINAMATH_CALUDE_histogram_classes_l3721_372134

def max_value : ℝ := 169
def min_value : ℝ := 143
def class_interval : ℝ := 3

theorem histogram_classes : 
  ∃ (n : ℕ), n = ⌈(max_value - min_value) / class_interval⌉ ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_histogram_classes_l3721_372134


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3721_372189

def set_A : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + 3}
def set_B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 3 * p.1 - 1}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {(2, 5)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3721_372189


namespace NUMINAMATH_CALUDE_probability_of_third_draw_l3721_372168

/-- Represents the outcome of a single draw -/
inductive Ball : Type
| Hui : Ball
| Zhou : Ball
| Mei : Ball
| Li : Ball

/-- Represents the result of three draws -/
structure ThreeDraw :=
  (first : Ball)
  (second : Ball)
  (third : Ball)

/-- Checks if a ThreeDraw result meets the conditions -/
def isValidDraw (draw : ThreeDraw) : Prop :=
  ((draw.first = Ball.Hui ∨ draw.first = Ball.Zhou) ∧
   (draw.second ≠ Ball.Hui ∧ draw.second ≠ Ball.Zhou)) ∨
  ((draw.first ≠ Ball.Hui ∧ draw.first ≠ Ball.Zhou) ∧
   (draw.second = Ball.Hui ∨ draw.second = Ball.Zhou)) ∧
  (draw.third = Ball.Hui ∨ draw.third = Ball.Zhou)

/-- The total number of trials in the experiment -/
def totalTrials : Nat := 16

/-- The number of successful outcomes in the experiment -/
def successfulTrials : Nat := 2

/-- Theorem stating the probability of drawing both "惠" and "州" exactly on the third draw -/
theorem probability_of_third_draw :
  (successfulTrials : ℚ) / totalTrials = 1 / 8 :=
sorry

end NUMINAMATH_CALUDE_probability_of_third_draw_l3721_372168


namespace NUMINAMATH_CALUDE_orange_grape_ratio_l3721_372159

/-- Given the number of orange and grape sweets, and the number of sweets per tray,
    calculate the ratio of orange to grape sweets in each tray. -/
def sweetRatio (orange : Nat) (grape : Nat) (perTray : Nat) : Rat :=
  (orange / perTray) / (grape / perTray)

/-- Theorem stating that for 36 orange sweets and 44 grape sweets,
    when divided into trays of 4, the ratio is 9/11. -/
theorem orange_grape_ratio :
  sweetRatio 36 44 4 = 9 / 11 := by
  sorry

end NUMINAMATH_CALUDE_orange_grape_ratio_l3721_372159


namespace NUMINAMATH_CALUDE_logarithm_equation_l3721_372184

theorem logarithm_equation : 
  (((1 - Real.log 3 / Real.log 6) ^ 2 + 
    (Real.log 2 / Real.log 6) * (Real.log 18 / Real.log 6)) / 
   (Real.log 4 / Real.log 6)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_equation_l3721_372184


namespace NUMINAMATH_CALUDE_specific_polyhedron_volume_l3721_372140

/-- A polyhedron formed by folding a flat figure -/
structure Polyhedron where
  /-- The number of equilateral triangles in the flat figure -/
  num_triangles : ℕ
  /-- The number of squares in the flat figure -/
  num_squares : ℕ
  /-- The side length of the squares -/
  square_side : ℝ
  /-- The number of regular hexagons in the flat figure -/
  num_hexagons : ℕ

/-- Calculate the volume of the polyhedron -/
def calculate_volume (p : Polyhedron) : ℝ :=
  sorry

/-- The theorem stating the volume of the specific polyhedron -/
theorem specific_polyhedron_volume :
  let p : Polyhedron := {
    num_triangles := 3,
    num_squares := 3,
    square_side := 2,
    num_hexagons := 1
  }
  calculate_volume p = 11 :=
sorry

end NUMINAMATH_CALUDE_specific_polyhedron_volume_l3721_372140


namespace NUMINAMATH_CALUDE_square_pattern_l3721_372151

theorem square_pattern (n : ℕ) : (n - 1) * (n + 1) + 1 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_square_pattern_l3721_372151


namespace NUMINAMATH_CALUDE_fence_perimeter_is_262_l3721_372193

/-- Calculates the outer perimeter of a rectangular fence with given specifications -/
def calculate_fence_perimeter (total_posts : ℕ) (post_width : ℚ) (post_spacing : ℕ) 
  (aspect_ratio : ℚ) : ℚ :=
  let width_posts := total_posts / (3 : ℕ)
  let length_posts := 2 * width_posts
  let width := (width_posts - 1) * post_spacing + width_posts * post_width
  let length := (length_posts - 1) * post_spacing + length_posts * post_width
  2 * (width + length)

/-- The outer perimeter of the fence with given specifications is 262 feet -/
theorem fence_perimeter_is_262 : 
  calculate_fence_perimeter 32 (1/2) 6 2 = 262 := by
  sorry

end NUMINAMATH_CALUDE_fence_perimeter_is_262_l3721_372193


namespace NUMINAMATH_CALUDE_sum_base4_equals_l3721_372155

/-- Converts a base 4 number represented as a list of digits to a natural number -/
def base4ToNat (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 4 * acc + d) 0

/-- Converts a natural number to its base 4 representation as a list of digits -/
def natToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem sum_base4_equals :
  base4ToNat [2, 1, 2] + base4ToNat [1, 0, 3] + base4ToNat [3, 2, 1] =
  base4ToNat [1, 0, 1, 2] := by
  sorry

end NUMINAMATH_CALUDE_sum_base4_equals_l3721_372155


namespace NUMINAMATH_CALUDE_total_rowing_campers_l3721_372127

def morning_rowing : ℕ := 13
def afternoon_rowing : ℕ := 21
def morning_hiking : ℕ := 59

theorem total_rowing_campers :
  morning_rowing + afternoon_rowing = 34 := by sorry

end NUMINAMATH_CALUDE_total_rowing_campers_l3721_372127


namespace NUMINAMATH_CALUDE_unique_pair_satisfying_conditions_l3721_372197

theorem unique_pair_satisfying_conditions :
  ∃! (a b : ℕ), 
    b > a ∧ 
    a > 1 ∧ 
    a ≤ 20 ∧ 
    b ≤ 20 ∧
    (∀ (x y : ℕ), y > x ∧ x > 1 ∧ x ≤ 20 ∧ y ≤ 20 ∧ x + y = a + b →
      ∃ (p q r s : ℕ), p ≠ r ∧ q ≠ s ∧ q > p ∧ p > 1 ∧ s > r ∧ r > 1 ∧ x * y = p * q ∧ x * y = r * s) ∧
    (∀ (p q : ℕ), q > p ∧ p > 1 ∧ a * b = p * q → a = p ∧ b = q) :=
sorry

end NUMINAMATH_CALUDE_unique_pair_satisfying_conditions_l3721_372197


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3721_372157

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a₁ : ℝ  -- First term
  d : ℝ   -- Common difference

/-- Sum of first n terms of an arithmetic sequence -/
def sumFirstNTerms (seq : ArithmeticSequence) (n : ℕ) : ℝ := sorry

/-- Condition for symmetry of intersection points -/
def symmetricIntersectionPoints (seq : ArithmeticSequence) : Prop := sorry

theorem arithmetic_sequence_sum (seq : ArithmeticSequence) (n : ℕ) :
  symmetricIntersectionPoints seq →
  sumFirstNTerms seq n = -n^2 + 2*n := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3721_372157


namespace NUMINAMATH_CALUDE_roots_of_equation_l3721_372125

theorem roots_of_equation : ∀ x : ℝ, 
  (x^2 - 5*x + 6)*(x - 1)*(x - 6) = 0 ↔ x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_equation_l3721_372125


namespace NUMINAMATH_CALUDE_holiday_approval_count_l3721_372115

theorem holiday_approval_count (total : ℕ) (oppose_percent : ℚ) (indifferent_percent : ℚ) 
  (h_total : total = 600)
  (h_oppose : oppose_percent = 6 / 100)
  (h_indifferent : indifferent_percent = 14 / 100) :
  ↑total * (1 - oppose_percent - indifferent_percent) = 480 :=
by sorry

end NUMINAMATH_CALUDE_holiday_approval_count_l3721_372115


namespace NUMINAMATH_CALUDE_common_tangent_range_l3721_372186

/-- The range of a for which y = ln x and y = ax² have a common tangent line -/
theorem common_tangent_range (a : ℝ) : 
  (a > 0 ∧ ∃ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧ 
    (1 / x₁ = 2 * a * x₂) ∧ 
    (Real.log x₁ - 1 = -a * x₂^2)) ↔ 
  a ≥ 1 / (2 * Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_common_tangent_range_l3721_372186


namespace NUMINAMATH_CALUDE_ryan_chinese_hours_l3721_372196

/-- Ryan's daily study schedule -/
structure StudySchedule where
  english_hours : ℕ
  chinese_hours : ℕ
  english_more_than_chinese : ℕ

/-- Ryan's actual study schedule satisfying the given conditions -/
def ryan_schedule : StudySchedule :=
  { english_hours := 6,
    chinese_hours := 2,
    english_more_than_chinese := 4 }

/-- Theorem stating that Ryan's schedule satisfies the given conditions -/
theorem ryan_chinese_hours :
  ryan_schedule.english_hours = ryan_schedule.chinese_hours + ryan_schedule.english_more_than_chinese :=
by sorry

end NUMINAMATH_CALUDE_ryan_chinese_hours_l3721_372196


namespace NUMINAMATH_CALUDE_parallel_tangents_condition_l3721_372195

-- Define the two curves
def curve1 (x : ℝ) : ℝ := x^2 - 1
def curve2 (x : ℝ) : ℝ := 1 - x^3

-- Define the derivatives of the curves
def curve1_derivative (x : ℝ) : ℝ := 2 * x
def curve2_derivative (x : ℝ) : ℝ := -3 * x^2

-- Theorem statement
theorem parallel_tangents_condition (x₀ : ℝ) :
  curve1_derivative x₀ = curve2_derivative x₀ ↔ x₀ = 0 ∨ x₀ = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_tangents_condition_l3721_372195


namespace NUMINAMATH_CALUDE_xy_value_l3721_372124

theorem xy_value (x y : ℝ) : 
  |x - y + 1| + (y + 5)^2010 = 0 → x * y = 30 := by sorry

end NUMINAMATH_CALUDE_xy_value_l3721_372124


namespace NUMINAMATH_CALUDE_marble_exchange_problem_l3721_372142

/-- Represents the marble exchange problem with Woong, Youngsoo, and Hyogeun --/
theorem marble_exchange_problem (W Y H : ℕ) : 
  (W + 2 = 20) →  -- Woong's final marbles
  (Y - 5 = 20) →  -- Youngsoo's final marbles
  (H + 3 = 20) →  -- Hyogeun's final marbles
  W = 18 :=        -- Woong's initial marbles
by sorry

end NUMINAMATH_CALUDE_marble_exchange_problem_l3721_372142


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3721_372162

theorem arithmetic_calculation : 2 + 3 * 4 - 5 + 6 / 3 = 11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3721_372162


namespace NUMINAMATH_CALUDE_order_of_products_l3721_372117

theorem order_of_products (m n : ℝ) (hm : m < 0) (hn : -1 < n ∧ n < 0) :
  m < m * n^2 ∧ m * n^2 < m * n := by sorry

end NUMINAMATH_CALUDE_order_of_products_l3721_372117


namespace NUMINAMATH_CALUDE_frogs_on_lily_pads_l3721_372131

/-- Given the total number of frogs in a pond, the number of frogs on logs, and the number of baby frogs on a rock,
    calculate the number of frogs on lily pads. -/
theorem frogs_on_lily_pads (total : ℕ) (on_logs : ℕ) (on_rock : ℕ) 
    (h1 : total = 32) 
    (h2 : on_logs = 3) 
    (h3 : on_rock = 24) : 
  total - on_logs - on_rock = 5 := by
  sorry

end NUMINAMATH_CALUDE_frogs_on_lily_pads_l3721_372131


namespace NUMINAMATH_CALUDE_volume_of_specific_pyramid_l3721_372123

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (side1 : ℝ) (side2 : ℝ) (side3 : ℝ) (side4 : ℝ)
  (shorter_diagonal : ℝ)

/-- Represents a pyramid -/
structure Pyramid :=
  (base : Quadrilateral)
  (lateral_face_angle : ℝ)

/-- Calculate the volume of a pyramid -/
def pyramid_volume (p : Pyramid) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem volume_of_specific_pyramid :
  let base := Quadrilateral.mk 5 5 10 10 (4 * Real.sqrt 5)
  let pyr := Pyramid.mk base (π / 4)  -- 45° in radians
  pyramid_volume pyr = 500 / 9 := by sorry

end NUMINAMATH_CALUDE_volume_of_specific_pyramid_l3721_372123


namespace NUMINAMATH_CALUDE_digit_40000_is_1_l3721_372199

/-- The sequence of digits formed by concatenating natural numbers -/
def digit_sequence : ℕ → ℕ := sorry

/-- The 40,000th digit in the sequence -/
def digit_40000 : ℕ := digit_sequence 40000

/-- Theorem: The 40,000th digit in the sequence is 1 -/
theorem digit_40000_is_1 : digit_40000 = 1 := by sorry

end NUMINAMATH_CALUDE_digit_40000_is_1_l3721_372199


namespace NUMINAMATH_CALUDE_car_wash_goal_remaining_l3721_372191

def car_wash_fundraiser (goal : ℕ) (high_donors : ℕ) (high_donation : ℕ) (low_donors : ℕ) (low_donation : ℕ) : ℕ :=
  goal - (high_donors * high_donation + low_donors * low_donation)

theorem car_wash_goal_remaining :
  car_wash_fundraiser 150 3 10 15 5 = 45 := by sorry

end NUMINAMATH_CALUDE_car_wash_goal_remaining_l3721_372191
