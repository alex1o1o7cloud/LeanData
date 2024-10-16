import Mathlib

namespace NUMINAMATH_CALUDE_min_sum_squares_min_sum_squares_attained_l1852_185238

theorem min_sum_squares (x₁ x₂ x₃ : ℝ) (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0) 
  (h_sum : 2*x₁ + 4*x₂ + 6*x₃ = 120) : 
  x₁^2 + x₂^2 + x₃^2 ≥ 350 := by
  sorry

theorem min_sum_squares_attained (x₁ x₂ x₃ : ℝ) (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0) 
  (h_sum : 2*x₁ + 4*x₂ + 6*x₃ = 120) : 
  ∃ y₁ y₂ y₃ : ℝ, y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0 ∧ 2*y₁ + 4*y₂ + 6*y₃ = 120 ∧ 
  y₁^2 + y₂^2 + y₃^2 = 350 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_min_sum_squares_attained_l1852_185238


namespace NUMINAMATH_CALUDE_absolute_value_comparison_l1852_185274

theorem absolute_value_comparison (a b : ℚ) : 
  |a| = 2/3 ∧ |b| = 3/5 → 
  ((a = 2/3 ∨ a = -2/3) ∧ 
   (b = 3/5 ∨ b = -3/5) ∧ 
   (a = 2/3 → a > b) ∧ 
   (a = -2/3 → a < b)) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_comparison_l1852_185274


namespace NUMINAMATH_CALUDE_race_distance_difference_l1852_185212

theorem race_distance_difference (race_distance : ℝ) (a_time b_time : ℝ) 
  (h1 : race_distance = 120)
  (h2 : a_time = 36)
  (h3 : b_time = 45) : 
  race_distance - (race_distance / b_time * a_time) = 24 := by
  sorry

end NUMINAMATH_CALUDE_race_distance_difference_l1852_185212


namespace NUMINAMATH_CALUDE_expand_product_l1852_185214

theorem expand_product (x : ℝ) : (x + 3) * (x - 8) = x^2 - 5*x - 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1852_185214


namespace NUMINAMATH_CALUDE_bus_problem_l1852_185256

/-- The number of students remaining on a bus after a given number of stops,
    where half the students get off at each stop. -/
def studentsRemaining (initial : ℕ) (stops : ℕ) : ℚ :=
  initial / (2 ^ stops)

/-- Theorem stating that if a bus starts with 48 students and half of the remaining
    students get off at each of three stops, then 6 students will remain after the third stop. -/
theorem bus_problem : studentsRemaining 48 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_bus_problem_l1852_185256


namespace NUMINAMATH_CALUDE_annulus_area_l1852_185281

/-- The area of an annulus formed by two concentric circles. -/
theorem annulus_area (r s x : ℝ) (hr : r > 0) (hs : s > 0) (hrs : r > s) :
  let P := Real.sqrt (r^2 - s^2)
  x^2 = r^2 - s^2 →
  π * (r^2 - s^2) = π * x^2 := by sorry

end NUMINAMATH_CALUDE_annulus_area_l1852_185281


namespace NUMINAMATH_CALUDE_rhombus_area_l1852_185239

/-- The area of a rhombus with diagonals satisfying a specific equation --/
theorem rhombus_area (a b : ℝ) (h : (a - 1)^2 + Real.sqrt (b - 4) = 0) :
  (1/2 : ℝ) * a * b = 2 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l1852_185239


namespace NUMINAMATH_CALUDE_function_properties_l1852_185219

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x + φ)

theorem function_properties
  (ω φ : ℝ)
  (h_ω : ω > 0)
  (h_φ : -π / 2 ≤ φ ∧ φ < π / 2)
  (h_sym : ∀ x, f ω φ (2 * π / 3 - x) = f ω φ (2 * π / 3 + x))
  (h_period : ∀ x, f ω φ (x + π) = f ω φ x) :
  ω = 2 ∧
  φ = -π / 6 ∧
  (∀ x ∈ Set.Icc 0 (π / 2), f ω φ x ≤ Real.sqrt 3) ∧
  (∀ x ∈ Set.Icc 0 (π / 2), f ω φ x ≥ -Real.sqrt 3 / 2) ∧
  (∃ x ∈ Set.Icc 0 (π / 2), f ω φ x = Real.sqrt 3) ∧
  (∃ x ∈ Set.Icc 0 (π / 2), f ω φ x = -Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1852_185219


namespace NUMINAMATH_CALUDE_submarine_invention_uses_analogy_l1852_185213

/-- Represents the type of reasoning used in an invention process. -/
inductive ReasoningType
  | Analogy
  | Deduction
  | Induction

/-- Represents an invention process. -/
structure Invention where
  name : String
  inspiration : String
  reasoning : ReasoningType

/-- The submarine invention process. -/
def submarineInvention : Invention :=
  { name := "submarine",
    inspiration := "fish shape",
    reasoning := ReasoningType.Analogy }

/-- Theorem stating that the reasoning used in inventing submarines by imitating
    the shape of fish is analogy. -/
theorem submarine_invention_uses_analogy :
  submarineInvention.reasoning = ReasoningType.Analogy := by
  sorry

end NUMINAMATH_CALUDE_submarine_invention_uses_analogy_l1852_185213


namespace NUMINAMATH_CALUDE_A_n_is_integer_l1852_185258

theorem A_n_is_integer (a b n : ℕ) (h1 : a > b) (h2 : b > 0) 
  (θ : Real) (h3 : 0 < θ) (h4 : θ < Real.pi / 2) 
  (h5 : Real.sin θ = (2 * a * b : ℝ) / ((a^2 + b^2) : ℝ)) :
  ∃ k : ℤ, ((a^2 + b^2 : ℕ)^n : ℝ) * Real.sin (n * θ) = k := by
  sorry

#check A_n_is_integer

end NUMINAMATH_CALUDE_A_n_is_integer_l1852_185258


namespace NUMINAMATH_CALUDE_cos_300_degrees_l1852_185225

theorem cos_300_degrees : Real.cos (300 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_300_degrees_l1852_185225


namespace NUMINAMATH_CALUDE_factorization_x_squared_minus_nine_l1852_185276

theorem factorization_x_squared_minus_nine (x : ℝ) : x^2 - 9 = (x - 3) * (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x_squared_minus_nine_l1852_185276


namespace NUMINAMATH_CALUDE_rhombus_area_l1852_185233

/-- The area of a rhombus with specific properties -/
theorem rhombus_area (s : ℝ) (d₁ d₂ : ℝ) (h_side : s = Real.sqrt 130) 
  (h_diag_diff : d₂ = d₁ + 4) (h_perp : d₁ * d₂ = 4 * s^2) : d₁ * d₂ / 2 = 126 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l1852_185233


namespace NUMINAMATH_CALUDE_max_x_in_grid_l1852_185291

/-- Represents a 5x5 grid with X placements -/
def Grid := Fin 5 → Fin 5 → Bool

/-- Checks if three X's are in a row (horizontally, vertically, or diagonally) -/
def has_three_in_row (g : Grid) : Prop := sorry

/-- Checks if each row has at least one X -/
def each_row_has_x (g : Grid) : Prop := sorry

/-- Counts the number of X's in the grid -/
def count_x (g : Grid) : Nat := sorry

/-- Theorem: The maximum number of X's in a 5x5 grid without three in a row and at least one X per row is 10 -/
theorem max_x_in_grid : 
  ∀ g : Grid, 
  ¬has_three_in_row g → 
  each_row_has_x g → 
  count_x g ≤ 10 ∧ 
  ∃ g' : Grid, ¬has_three_in_row g' ∧ each_row_has_x g' ∧ count_x g' = 10 := by
  sorry

end NUMINAMATH_CALUDE_max_x_in_grid_l1852_185291


namespace NUMINAMATH_CALUDE_harmonic_sum_number_bounds_harmonic_number_digit_sum_even_l1852_185230

/-- Represents a three-digit natural number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ units ≤ 9

/-- Checks if a number is a sum number -/
def is_sum_number (n : ThreeDigitNumber) : Prop :=
  n.hundreds = n.tens + n.units

/-- Checks if a number is a harmonic number -/
def is_harmonic_number (n : ThreeDigitNumber) : Prop :=
  n.hundreds = n.tens^2 - n.units^2

/-- Checks if a number is a harmonic sum number -/
def is_harmonic_sum_number (n : ThreeDigitNumber) : Prop :=
  is_sum_number n ∧ is_harmonic_number n

/-- Converts a ThreeDigitNumber to its numeric value -/
def to_nat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.units

theorem harmonic_sum_number_bounds (n : ThreeDigitNumber) :
  is_harmonic_sum_number n → 110 ≤ to_nat n ∧ to_nat n ≤ 954 := by
  sorry

theorem harmonic_number_digit_sum_even (n : ThreeDigitNumber) :
  is_harmonic_number n → Even (n.hundreds + n.tens + n.units) := by
  sorry

end NUMINAMATH_CALUDE_harmonic_sum_number_bounds_harmonic_number_digit_sum_even_l1852_185230


namespace NUMINAMATH_CALUDE_system_of_equations_sum_l1852_185284

theorem system_of_equations_sum (a b c x y z : ℝ) 
  (eq1 : 17 * x + b * y + c * z = 0)
  (eq2 : a * x + 29 * y + c * z = 0)
  (eq3 : a * x + b * y + 53 * z = 0)
  (ha : a ≠ 17)
  (hx : x ≠ 0) :
  a / (a - 17) + b / (b - 29) + c / (c - 53) = 1 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_sum_l1852_185284


namespace NUMINAMATH_CALUDE_power_of_seven_mod_hundred_l1852_185255

theorem power_of_seven_mod_hundred : ∃ (n : ℕ), n > 0 ∧ 7^n % 100 = 1 ∧ ∀ (k : ℕ), 0 < k → k < n → 7^k % 100 ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_power_of_seven_mod_hundred_l1852_185255


namespace NUMINAMATH_CALUDE_min_sum_sides_triangle_l1852_185211

theorem min_sum_sides_triangle (a b c : ℝ) (A B C : ℝ) :
  (a > 0) → (b > 0) → (c > 0) →
  (A > 0) → (B > 0) → (C > 0) →
  ((a + b)^2 - c^2 = 4) →
  (C = Real.pi / 3) →
  (c^2 = a^2 + b^2 - 2*a*b*(Real.cos C)) →
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = a + b ∧ x * y = 4 / 3) →
  (a + b ≥ 4 * Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_sides_triangle_l1852_185211


namespace NUMINAMATH_CALUDE_odd_even_properties_l1852_185295

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem odd_even_properties (f g : ℝ → ℝ) (h1 : is_odd f) (h2 : is_even g) :
  (∀ x, (|f x| + g x) = (|f (-x)| + g (-x))) ∧
  (∀ x, f x * |g x| = -(f (-x) * |g (-x)|)) :=
sorry

end NUMINAMATH_CALUDE_odd_even_properties_l1852_185295


namespace NUMINAMATH_CALUDE_problem_1_solution_l1852_185221

theorem problem_1_solution (x : ℝ) : 
  (2 / (x - 3) = 1 / x) ↔ (x = -3) :=
sorry

end NUMINAMATH_CALUDE_problem_1_solution_l1852_185221


namespace NUMINAMATH_CALUDE_dogs_not_doing_anything_l1852_185278

def total_dogs : ℕ := 264
def running_dogs : ℕ := 40
def playing_dogs : ℕ := 66
def barking_dogs : ℕ := 44
def digging_dogs : ℕ := 26
def agility_dogs : ℕ := 12

theorem dogs_not_doing_anything : 
  total_dogs - (running_dogs + playing_dogs + barking_dogs + digging_dogs + agility_dogs) = 76 := by
  sorry

end NUMINAMATH_CALUDE_dogs_not_doing_anything_l1852_185278


namespace NUMINAMATH_CALUDE_cosine_sine_inequality_l1852_185254

theorem cosine_sine_inequality (x : ℝ) : 
  (1 / 4 : ℝ) ≤ (Real.cos x)^6 + (Real.sin x)^6 ∧ (Real.cos x)^6 + (Real.sin x)^6 ≤ 1 :=
by
  sorry

#check cosine_sine_inequality

end NUMINAMATH_CALUDE_cosine_sine_inequality_l1852_185254


namespace NUMINAMATH_CALUDE_smallest_number_l1852_185289

theorem smallest_number (a b c d : ℝ) : 
  a = -2024 → b = -2022 → c = -2022.5 → d = 0 →
  (a < -2023 ∧ b > -2023 ∧ c > -2023 ∧ d > -2023) := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l1852_185289


namespace NUMINAMATH_CALUDE_log_product_equals_one_l1852_185243

theorem log_product_equals_one : 
  Real.log 3 / Real.log 2 * (Real.log 4 / Real.log 9) = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equals_one_l1852_185243


namespace NUMINAMATH_CALUDE_eldest_age_l1852_185231

theorem eldest_age (a b c d : ℕ) : 
  (∃ (x : ℕ), a = 5 * x ∧ b = 7 * x ∧ c = 8 * x ∧ d = 9 * x) →  -- ages are in ratio 5:7:8:9
  (a - 10) + (b - 10) + (c - 10) + (d - 10) = 107 →             -- sum of ages 10 years ago
  d = 45                                                        -- present age of eldest
  := by sorry

end NUMINAMATH_CALUDE_eldest_age_l1852_185231


namespace NUMINAMATH_CALUDE_largest_α_is_173_l1852_185282

/-- A triangle with angles satisfying specific conditions -/
structure SpecialTriangle where
  α : ℕ
  β : ℕ
  γ : ℕ
  angle_sum : α + β + γ = 180
  angle_order : α > β ∧ β > γ
  α_obtuse : α > 90
  α_prime : Nat.Prime α
  β_prime : Nat.Prime β

/-- The largest possible value of α in a SpecialTriangle is 173 -/
theorem largest_α_is_173 : ∀ t : SpecialTriangle, t.α ≤ 173 ∧ ∃ t' : SpecialTriangle, t'.α = 173 :=
  sorry

end NUMINAMATH_CALUDE_largest_α_is_173_l1852_185282


namespace NUMINAMATH_CALUDE_initial_workers_correct_l1852_185298

/-- Represents the initial number of workers employed by the contractor -/
def initial_workers : ℕ := 360

/-- Represents the total number of days to complete the wall -/
def total_days : ℕ := 50

/-- Represents the number of days after which progress is measured -/
def days_passed : ℕ := 25

/-- Represents the percentage of work completed after 'days_passed' -/
def work_completed : ℚ := 2/5

/-- Represents the additional workers needed to complete the work on time -/
def additional_workers : ℕ := 90

/-- Theorem stating that the initial number of workers is correct given the conditions -/
theorem initial_workers_correct :
  initial_workers * (total_days : ℚ) = (initial_workers + additional_workers) * 
    (total_days * work_completed) :=
by sorry

end NUMINAMATH_CALUDE_initial_workers_correct_l1852_185298


namespace NUMINAMATH_CALUDE_lunch_cost_proof_l1852_185227

theorem lunch_cost_proof (adam_cost rick_cost jose_cost total_cost : ℚ) : 
  adam_cost = (2 : ℚ) / (3 : ℚ) * rick_cost →
  rick_cost = jose_cost →
  jose_cost = 45 →
  total_cost = adam_cost + rick_cost + jose_cost →
  total_cost = 120 := by
sorry

end NUMINAMATH_CALUDE_lunch_cost_proof_l1852_185227


namespace NUMINAMATH_CALUDE_competition_result_l1852_185267

def math_competition (sammy_score : ℕ) (opponent_score : ℕ) : Prop :=
  let gab_score := 2 * sammy_score
  let cher_score := 2 * gab_score
  let total_score := sammy_score + gab_score + cher_score
  total_score - opponent_score = 55

theorem competition_result : math_competition 20 85 := by
  sorry

end NUMINAMATH_CALUDE_competition_result_l1852_185267


namespace NUMINAMATH_CALUDE_apples_in_baskets_l1852_185297

theorem apples_in_baskets (total_apples : ℕ) (num_baskets : ℕ) (removed_apples : ℕ) 
  (h1 : total_apples = 64)
  (h2 : num_baskets = 4)
  (h3 : removed_apples = 3)
  : (total_apples / num_baskets) - removed_apples = 13 := by
  sorry

end NUMINAMATH_CALUDE_apples_in_baskets_l1852_185297


namespace NUMINAMATH_CALUDE_securities_stamp_duty_difference_l1852_185290

/-- The securities transaction stamp duty problem -/
theorem securities_stamp_duty_difference :
  let old_rate : ℚ := 3 / 1000
  let new_rate : ℚ := 1 / 1000
  let purchase_value : ℚ := 100000
  (purchase_value * old_rate - purchase_value * new_rate) = 200 := by
  sorry

end NUMINAMATH_CALUDE_securities_stamp_duty_difference_l1852_185290


namespace NUMINAMATH_CALUDE_sum_of_digits_cube_n_nines_l1852_185215

/-- The sum of digits function for natural numbers -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The function that returns a number composed of n nines -/
def n_nines (n : ℕ) : ℕ := 10^n - 1

theorem sum_of_digits_cube_n_nines (n : ℕ) :
  sum_of_digits ((n_nines n)^3) = 18 * n := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_cube_n_nines_l1852_185215


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_power_l1852_185279

theorem imaginary_part_of_complex_power (i : ℂ) (h : i * i = -1) :
  let z := (1 + i) / (1 - i)
  Complex.im (z ^ 2023) = -Complex.im i :=
by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_power_l1852_185279


namespace NUMINAMATH_CALUDE_function_existence_and_properties_l1852_185280

/-- A function satisfying the given equation -/
def SatisfiesEquation (f : ℤ → ℤ → ℤ) : Prop :=
  ∀ n m : ℤ, f n m = (1/4) * (f (n-1) m + f (n+1) m + f n (m-1) + f n (m+1))

/-- The function is non-constant -/
def IsNonConstant (f : ℤ → ℤ → ℤ) : Prop :=
  ∃ n₁ m₁ n₂ m₂ : ℤ, f n₁ m₁ ≠ f n₂ m₂

/-- The function takes values both greater and less than any integer -/
def SpansAllIntegers (f : ℤ → ℤ → ℤ) : Prop :=
  ∀ k : ℤ, (∃ n₁ m₁ : ℤ, f n₁ m₁ > k) ∧ (∃ n₂ m₂ : ℤ, f n₂ m₂ < k)

/-- The main theorem -/
theorem function_existence_and_properties :
  ∃ f : ℤ → ℤ → ℤ, SatisfiesEquation f ∧ IsNonConstant f ∧ SpansAllIntegers f := by
  sorry

end NUMINAMATH_CALUDE_function_existence_and_properties_l1852_185280


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1852_185235

/-- An arithmetic sequence with sum of first n terms S_n = -2n^2 + 15n -/
def S (n : ℕ+) : ℤ := -2 * n.val ^ 2 + 15 * n.val

/-- The general term of the arithmetic sequence -/
def a (n : ℕ+) : ℤ := 17 - 4 * n.val

theorem arithmetic_sequence_properties :
  ∀ n : ℕ+,
  -- The general term of the sequence is a_n = 17 - 4n
  (∀ k : ℕ+, S k - S (k - 1) = a k) ∧
  -- S_n achieves its maximum value when n = 4
  (∀ k : ℕ+, S k ≤ S 4) ∧
  -- The maximum value of S_n is 28
  S 4 = 28 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1852_185235


namespace NUMINAMATH_CALUDE_floor_abs_negative_l1852_185265

theorem floor_abs_negative : ⌊|(-45.7 : ℝ)|⌋ = 45 := by sorry

end NUMINAMATH_CALUDE_floor_abs_negative_l1852_185265


namespace NUMINAMATH_CALUDE_eccentricity_ratio_range_l1852_185232

theorem eccentricity_ratio_range (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let e₁ := Real.sqrt (a^2 - b^2) / a
  let e₂ := Real.sqrt (a^2 - b^2) / b
  (e₁ * e₂ < 1) →
  (∃ x, x > Real.sqrt 2 ∧ x < (1 + Real.sqrt 5) / 2 ∧ e₂ / e₁ = x) ∧
  (∀ y, y ≤ Real.sqrt 2 ∨ y ≥ (1 + Real.sqrt 5) / 2 → e₂ / e₁ ≠ y) :=
by sorry


end NUMINAMATH_CALUDE_eccentricity_ratio_range_l1852_185232


namespace NUMINAMATH_CALUDE_russia_us_size_ratio_l1852_185206

theorem russia_us_size_ratio :
  ∀ (us canada russia : ℝ),
    us > 0 →
    canada = 1.5 * us →
    russia = (4/3) * canada →
    russia / us = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_russia_us_size_ratio_l1852_185206


namespace NUMINAMATH_CALUDE_justin_reading_theorem_l1852_185268

/-- Calculates the total number of pages Justin reads in a week -/
def totalPagesRead (firstDayPages : ℕ) (remainingDays : ℕ) : ℕ :=
  firstDayPages + remainingDays * (2 * firstDayPages)

/-- Proves that Justin reads 130 pages in a week -/
theorem justin_reading_theorem :
  totalPagesRead 10 6 = 130 :=
by sorry

end NUMINAMATH_CALUDE_justin_reading_theorem_l1852_185268


namespace NUMINAMATH_CALUDE_northern_shoe_capital_relocation_l1852_185203

structure XionganNewArea where
  green_ecological : Bool
  innovation_driven : Bool
  coordinated_development : Bool
  open_development : Bool

structure AnxinCounty where
  santai_town : Bool
  traditional_shoemaking : Bool
  northern_shoe_capital : Bool
  nationwide_market : Bool
  adequate_transportation : Bool

def industrial_structure_adjustment (county : AnxinCounty) (new_area : XionganNewArea) : Bool :=
  county.traditional_shoemaking ∧ 
  (new_area.green_ecological ∧ new_area.innovation_driven ∧ 
   new_area.coordinated_development ∧ new_area.open_development)

def relocation_reason (county : AnxinCounty) (new_area : XionganNewArea) : String :=
  if industrial_structure_adjustment county new_area then
    "Industrial structure adjustment"
  else
    "Other reasons"

theorem northern_shoe_capital_relocation 
  (anxin : AnxinCounty) 
  (xiong_an : XionganNewArea) 
  (h1 : anxin.santai_town = true)
  (h2 : anxin.traditional_shoemaking = true)
  (h3 : anxin.northern_shoe_capital = true)
  (h4 : anxin.nationwide_market = true)
  (h5 : anxin.adequate_transportation = true)
  (h6 : xiong_an.green_ecological = true)
  (h7 : xiong_an.innovation_driven = true)
  (h8 : xiong_an.coordinated_development = true)
  (h9 : xiong_an.open_development = true) :
  relocation_reason anxin xiong_an = "Industrial structure adjustment" := by
  sorry

#check northern_shoe_capital_relocation

end NUMINAMATH_CALUDE_northern_shoe_capital_relocation_l1852_185203


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_plus_one_l1852_185241

theorem sum_of_reciprocals_plus_one (a b c : ℂ) : 
  (a^3 - a + 1 = 0) → (b^3 - b + 1 = 0) → (c^3 - c + 1 = 0) → 
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_plus_one_l1852_185241


namespace NUMINAMATH_CALUDE_peg_arrangement_count_l1852_185204

/-- The number of ways to arrange colored pegs on a triangular board. -/
def arrangeColoredPegs (yellow red green blue orange : Nat) : Nat :=
  Nat.factorial yellow * Nat.factorial red * Nat.factorial green * Nat.factorial blue * Nat.factorial orange

/-- Theorem stating the number of arrangements for the given peg counts. -/
theorem peg_arrangement_count :
  arrangeColoredPegs 6 5 4 3 2 = 12441600 := by
  sorry

end NUMINAMATH_CALUDE_peg_arrangement_count_l1852_185204


namespace NUMINAMATH_CALUDE_a_representation_l1852_185200

theorem a_representation (a : ℤ) (x y : ℤ) (h : 3 * a = x^2 + 2 * y^2) :
  ∃ (u v : ℤ), a = u^2 + 2 * v^2 := by
sorry

end NUMINAMATH_CALUDE_a_representation_l1852_185200


namespace NUMINAMATH_CALUDE_shortest_path_length_l1852_185240

/-- A regular octahedron with edge length 1 -/
structure RegularOctahedron where
  /-- The edge length of the octahedron is 1 -/
  edge_length : ℝ
  edge_length_eq : edge_length = 1

/-- A path on the surface of an octahedron -/
structure SurfacePath (o : RegularOctahedron) where
  /-- The length of the path -/
  length : ℝ
  /-- The path starts at a vertex -/
  starts_at_vertex : Bool
  /-- The path ends at the opposite vertex -/
  ends_at_opposite_vertex : Bool

/-- The theorem stating that the shortest path between opposite vertices has length 2 -/
theorem shortest_path_length (o : RegularOctahedron) : 
  ∃ (p : SurfacePath o), p.length = 2 ∧ 
  ∀ (q : SurfacePath o), q.starts_at_vertex ∧ q.ends_at_opposite_vertex → q.length ≥ p.length :=
sorry

end NUMINAMATH_CALUDE_shortest_path_length_l1852_185240


namespace NUMINAMATH_CALUDE_cannot_determine_start_month_l1852_185266

/-- Represents a month of the year -/
inductive Month
| January | February | March | April | May | June
| July | August | September | October | November | December

/-- Represents Nolan's GRE preparation period -/
structure PreparationPeriod where
  start_month : Month
  end_month : Month
  end_day : Nat

/-- The given information about Nolan's GRE preparation -/
def nolans_preparation : PreparationPeriod :=
  { end_month := Month.August,
    end_day := 3,
    start_month := sorry }  -- We don't know the start month

/-- Theorem stating that we cannot determine Nolan's start month -/
theorem cannot_determine_start_month :
  ∀ m : Month, ∃ p : PreparationPeriod,
    p.end_month = nolans_preparation.end_month ∧
    p.end_day = nolans_preparation.end_day ∧
    p.start_month = m :=
sorry

end NUMINAMATH_CALUDE_cannot_determine_start_month_l1852_185266


namespace NUMINAMATH_CALUDE_eighth_diagram_shaded_fraction_l1852_185285

/-- The number of shaded triangles in the nth diagram (n ≥ 1) -/
def shaded (n : ℕ) : ℕ := (n - 1) * n / 2

/-- The total number of small triangles in the nth diagram -/
def total (n : ℕ) : ℕ := n ^ 2

/-- The fraction of shaded triangles in the nth diagram -/
def shaded_fraction (n : ℕ) : ℚ := shaded n / total n

theorem eighth_diagram_shaded_fraction :
  shaded_fraction 8 = 7 / 16 := by sorry

end NUMINAMATH_CALUDE_eighth_diagram_shaded_fraction_l1852_185285


namespace NUMINAMATH_CALUDE_sum_of_four_numbers_l1852_185242

theorem sum_of_four_numbers : 3456 + 4563 + 5634 + 6345 = 19998 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_numbers_l1852_185242


namespace NUMINAMATH_CALUDE_equal_passengers_after_changes_l1852_185234

/-- Represents the number of passengers in a bus --/
structure BusPassengers where
  men : ℕ
  women : ℕ

/-- Calculates the total number of passengers --/
def BusPassengers.total (p : BusPassengers) : ℕ := p.men + p.women

/-- Represents the changes in passengers at a city --/
structure PassengerChanges where
  menLeaving : ℕ
  womenEntering : ℕ

/-- Applies changes to the passenger count --/
def applyChanges (p : BusPassengers) (c : PassengerChanges) : BusPassengers :=
  { men := p.men - c.menLeaving,
    women := p.women + c.womenEntering }

theorem equal_passengers_after_changes 
  (initialPassengers : BusPassengers)
  (changes : PassengerChanges) :
  initialPassengers.total = 72 →
  initialPassengers.women = initialPassengers.men / 2 →
  changes.menLeaving = 16 →
  changes.womenEntering = 8 →
  let finalPassengers := applyChanges initialPassengers changes
  finalPassengers.men = finalPassengers.women :=
by sorry

end NUMINAMATH_CALUDE_equal_passengers_after_changes_l1852_185234


namespace NUMINAMATH_CALUDE_log_sum_equals_two_l1852_185259

theorem log_sum_equals_two : Real.log 4 + Real.log 25 = 2 * Real.log 10 := by sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l1852_185259


namespace NUMINAMATH_CALUDE_weekly_social_media_time_l1852_185210

/-- Charlotte's daily phone usage in hours -/
def daily_phone_usage : ℕ := 16

/-- The fraction of phone time spent on social media -/
def social_media_fraction : ℚ := 1/2

/-- Number of days in a week -/
def days_in_week : ℕ := 7

/-- Theorem: Charlotte spends 56 hours on social media in a week -/
theorem weekly_social_media_time : 
  (daily_phone_usage * social_media_fraction * days_in_week : ℚ) = 56 := by
sorry

end NUMINAMATH_CALUDE_weekly_social_media_time_l1852_185210


namespace NUMINAMATH_CALUDE_owen_final_turtles_l1852_185260

def turtles_problem (owen_initial johanna_initial owen_after_month johanna_after_month owen_final : ℕ) : Prop :=
  (johanna_initial = owen_initial - 5) ∧
  (owen_after_month = 2 * owen_initial) ∧
  (johanna_after_month = johanna_initial / 2) ∧
  (owen_final = owen_after_month + johanna_after_month)

theorem owen_final_turtles :
  ∃ (owen_initial johanna_initial owen_after_month johanna_after_month owen_final : ℕ),
    turtles_problem owen_initial johanna_initial owen_after_month johanna_after_month owen_final ∧
    owen_initial = 21 ∧
    owen_final = 50 :=
by sorry

end NUMINAMATH_CALUDE_owen_final_turtles_l1852_185260


namespace NUMINAMATH_CALUDE_triangle_area_with_median_l1852_185249

/-- Given a triangle DEF with side lengths and median, calculate its area using Heron's formula -/
theorem triangle_area_with_median (DE DF DM : ℝ) (h1 : DE = 8) (h2 : DF = 17) (h3 : DM = 11) :
  ∃ (EF : ℝ), let a := DE
               let b := DF
               let c := EF
               let s := (a + b + c) / 2
               (s * (s - a) * (s - b) * (s - c)).sqrt = DM * EF / 2 := by
  sorry

#check triangle_area_with_median

end NUMINAMATH_CALUDE_triangle_area_with_median_l1852_185249


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1852_185224

/-- The polynomial with unknown coefficients a and b -/
def P (x a b : ℝ) : ℝ := 3 * x^4 + a * x^3 + 48 * x^2 + b * x + 12

/-- The given factor of the polynomial -/
def F (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 2

/-- Theorem stating that the polynomial P has the factor F when a = -26.5 and b = -40 -/
theorem polynomial_factorization (x : ℝ) : 
  ∃ (Q : ℝ → ℝ), P x (-26.5) (-40) = F x * Q x := by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1852_185224


namespace NUMINAMATH_CALUDE_ned_lost_lives_l1852_185217

/-- Proves that Ned lost 13 lives in a video game -/
theorem ned_lost_lives (initial_lives current_lives : ℕ) 
  (h1 : initial_lives = 83) 
  (h2 : current_lives = 70) : 
  initial_lives - current_lives = 13 := by
  sorry

end NUMINAMATH_CALUDE_ned_lost_lives_l1852_185217


namespace NUMINAMATH_CALUDE_five_touching_circles_exist_l1852_185253

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_positive : radius > 0

/-- Two circles touch if the distance between their centers is equal to the sum or difference of their radii --/
def circles_touch (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius + c2.radius)^2 ∨
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius - c2.radius)^2

/-- Theorem: There exists a configuration of five circles such that any two of them touch each other --/
theorem five_touching_circles_exist : ∃ (c1 c2 c3 c4 c5 : Circle),
  circles_touch c1 c2 ∧ circles_touch c1 c3 ∧ circles_touch c1 c4 ∧ circles_touch c1 c5 ∧
  circles_touch c2 c3 ∧ circles_touch c2 c4 ∧ circles_touch c2 c5 ∧
  circles_touch c3 c4 ∧ circles_touch c3 c5 ∧
  circles_touch c4 c5 :=
sorry

end NUMINAMATH_CALUDE_five_touching_circles_exist_l1852_185253


namespace NUMINAMATH_CALUDE_cookies_left_l1852_185283

/-- The number of cookies in a dozen -/
def cookies_per_dozen : ℕ := 12

/-- The number of dozens of cookies Meena bakes -/
def dozens_baked : ℕ := 5

/-- The number of dozens of cookies Mr. Stone buys -/
def dozens_sold_to_stone : ℕ := 2

/-- The number of cookies Brock buys -/
def cookies_sold_to_brock : ℕ := 7

/-- Calculates the total number of cookies Meena bakes -/
def total_cookies_baked : ℕ := dozens_baked * cookies_per_dozen

/-- Calculates the number of cookies sold to Mr. Stone -/
def cookies_sold_to_stone : ℕ := dozens_sold_to_stone * cookies_per_dozen

/-- Calculates the number of cookies sold to Katy -/
def cookies_sold_to_katy : ℕ := 2 * cookies_sold_to_brock

/-- Calculates the total number of cookies sold -/
def total_cookies_sold : ℕ := cookies_sold_to_stone + cookies_sold_to_brock + cookies_sold_to_katy

/-- Theorem: Meena has 15 cookies left after selling to Mr. Stone, Brock, and Katy -/
theorem cookies_left : total_cookies_baked - total_cookies_sold = 15 := by
  sorry

end NUMINAMATH_CALUDE_cookies_left_l1852_185283


namespace NUMINAMATH_CALUDE_smallest_binary_multiple_of_48_squared_l1852_185288

def is_binary_number (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 0 ∨ d = 1

def target_number : ℕ := 11111111100000000

theorem smallest_binary_multiple_of_48_squared :
  (target_number % (48^2) = 0) ∧
  is_binary_number target_number ∧
  ∀ m : ℕ, m < target_number →
    ¬(m % (48^2) = 0 ∧ is_binary_number m) :=
by sorry

#eval target_number % (48^2)  -- Should output 0
#eval target_number.digits 10  -- Should output [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]

end NUMINAMATH_CALUDE_smallest_binary_multiple_of_48_squared_l1852_185288


namespace NUMINAMATH_CALUDE_natalia_crates_l1852_185216

/-- Calculates the number of crates needed for a given number of items and crate capacity -/
def crates_needed (items : ℕ) (capacity : ℕ) : ℕ :=
  (items + capacity - 1) / capacity

/-- The total number of crates needed for Natalia's items -/
def total_crates : ℕ :=
  crates_needed 145 12 + crates_needed 271 8 + crates_needed 419 10 + crates_needed 209 14

theorem natalia_crates :
  total_crates = 104 := by
  sorry

end NUMINAMATH_CALUDE_natalia_crates_l1852_185216


namespace NUMINAMATH_CALUDE_manuscript_fee_proof_l1852_185247

/-- Calculates the tax payable for manuscript income not exceeding 4000 yuan -/
def tax_payable (income : ℝ) : ℝ := (income - 800) * 0.2 * 0.7

/-- The manuscript fee before tax deduction -/
def manuscript_fee : ℝ := 2800

theorem manuscript_fee_proof :
  manuscript_fee ≤ 4000 ∧
  tax_payable manuscript_fee = 280 :=
sorry

end NUMINAMATH_CALUDE_manuscript_fee_proof_l1852_185247


namespace NUMINAMATH_CALUDE_no_valid_solution_l1852_185222

-- Define the equation
def equation (x : ℝ) : Prop :=
  (36 - x) - (14 - x) = 2 * ((36 - x) - (18 - x))

-- Theorem stating that there is no valid solution
theorem no_valid_solution : ¬∃ (x : ℝ), x ≥ 0 ∧ equation x :=
sorry

end NUMINAMATH_CALUDE_no_valid_solution_l1852_185222


namespace NUMINAMATH_CALUDE_quadratic_root_sum_squares_l1852_185201

theorem quadratic_root_sum_squares (h : ℝ) : 
  (∃ x y : ℝ, 2 * x^2 + 4 * h * x + 6 = 0 ∧ 
               2 * y^2 + 4 * h * y + 6 = 0 ∧ 
               x^2 + y^2 = 34) → 
  |h| = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_squares_l1852_185201


namespace NUMINAMATH_CALUDE_intersection_A_B_l1852_185286

def A : Set ℝ := {x | x + 2 = 0}
def B : Set ℝ := {x | x^2 - 4 = 0}

theorem intersection_A_B : A ∩ B = {-2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1852_185286


namespace NUMINAMATH_CALUDE_three_real_roots_l1852_185245

/-- The polynomial f(x) = x^3 - 6x^2 + 9x - 2 -/
def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 9*x - 2

/-- Theorem: The equation f(x) = 0 has exactly three real roots -/
theorem three_real_roots : ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, f x = 0 := by sorry

end NUMINAMATH_CALUDE_three_real_roots_l1852_185245


namespace NUMINAMATH_CALUDE_fraction_value_l1852_185226

theorem fraction_value : (550 + 50) / (5^2 + 5) = 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l1852_185226


namespace NUMINAMATH_CALUDE_equation_solutions_l1852_185263

-- Define the equation
def equation (x : ℂ) : Prop :=
  (x - 4)^4 + (x - 6)^4 = 16

-- Define the set of solutions
def solution_set : Set ℂ :=
  {5 + Complex.I * Real.sqrt 7, 5 - Complex.I * Real.sqrt 7, 6, 4}

-- Theorem statement
theorem equation_solutions :
  ∀ x : ℂ, equation x ↔ x ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1852_185263


namespace NUMINAMATH_CALUDE_max_wrappers_l1852_185264

theorem max_wrappers (andy_wrappers : ℕ) (total_wrappers : ℕ) (max_wrappers : ℕ) : 
  andy_wrappers = 34 → total_wrappers = 49 → max_wrappers = total_wrappers - andy_wrappers →
  max_wrappers = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_max_wrappers_l1852_185264


namespace NUMINAMATH_CALUDE_solution_range_l1852_185248

-- Define the solution set A
def A : Set ℝ := {x | x^2 ≤ 5*x - 4}

-- Define the solution set M as a function of a
def M (a : ℝ) : Set ℝ := {x | (x - a) * (x - 2) ≤ 0}

-- State the theorem
theorem solution_range : 
  {a : ℝ | M a ⊆ A} = {a : ℝ | 1 ≤ a ∧ a ≤ 4} := by sorry

end NUMINAMATH_CALUDE_solution_range_l1852_185248


namespace NUMINAMATH_CALUDE_quadratic_translation_l1852_185287

/-- Given a quadratic function f and its translated version g, 
    prove that f has the form -2x^2+1 -/
theorem quadratic_translation (f g : ℝ → ℝ) :
  (∀ x, g x = -2*x^2 + 4*x + 1) →
  (∀ x, g x = f (x - 1) + 2) →
  (∀ x, f x = -2*x^2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_translation_l1852_185287


namespace NUMINAMATH_CALUDE_computer_desk_prices_l1852_185277

theorem computer_desk_prices :
  ∃ (x y : ℝ),
    (10 * x + 200 * y = 90000) ∧
    (12 * x + 120 * y = 90000) ∧
    (x = 6000) ∧
    (y = 150) := by
  sorry

end NUMINAMATH_CALUDE_computer_desk_prices_l1852_185277


namespace NUMINAMATH_CALUDE_at_least_two_babies_speak_l1852_185261

def probability_baby_speaks : ℚ := 1 / 5

def number_of_babies : ℕ := 7

theorem at_least_two_babies_speak :
  let p := probability_baby_speaks
  let n := number_of_babies
  (1 : ℚ) - (1 - p)^n - n * p * (1 - p)^(n-1) = 50477 / 78125 :=
by sorry

end NUMINAMATH_CALUDE_at_least_two_babies_speak_l1852_185261


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l1852_185262

theorem cyclic_sum_inequality (r : ℝ) (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  x^r * (x - y) * (x - z) + y^r * (y - z) * (y - x) + z^r * (z - x) * (z - y) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l1852_185262


namespace NUMINAMATH_CALUDE_correct_life_insights_l1852_185220

/- Define the types of connections -/
inductive ConnectionType
  | Objective
  | Diverse
  | Inevitable
  | Conditional

/- Define the actions related to connections -/
inductive ConnectionAction
  | CannotAdjust
  | EstablishNew
  | EliminateAccidental
  | GraspConditions

/- Define a proposition that represents an insight about connections -/
structure ConnectionInsight where
  type : ConnectionType
  action : ConnectionAction

/- Define the function that determines if an insight is correct -/
def isCorrectInsight (insight : ConnectionInsight) : Prop :=
  (insight.type = ConnectionType.Diverse ∧ insight.action = ConnectionAction.EstablishNew) ∨
  (insight.type = ConnectionType.Conditional ∧ insight.action = ConnectionAction.GraspConditions)

/- The theorem to prove -/
theorem correct_life_insights :
  ∀ (insight : ConnectionInsight),
    isCorrectInsight insight ↔
      (insight.type = ConnectionType.Diverse ∧ insight.action = ConnectionAction.EstablishNew) ∨
      (insight.type = ConnectionType.Conditional ∧ insight.action = ConnectionAction.GraspConditions) :=
by sorry


end NUMINAMATH_CALUDE_correct_life_insights_l1852_185220


namespace NUMINAMATH_CALUDE_pizza_order_problem_l1852_185236

theorem pizza_order_problem (slices_per_pizza : ℕ) (james_fraction : ℚ) (james_slices : ℕ) :
  slices_per_pizza = 6 →
  james_fraction = 2 / 3 →
  james_slices = 8 →
  (james_slices : ℚ) / james_fraction / slices_per_pizza = 2 :=
by sorry

end NUMINAMATH_CALUDE_pizza_order_problem_l1852_185236


namespace NUMINAMATH_CALUDE_correct_calculation_result_l1852_185292

theorem correct_calculation_result (x : ℝ) (h : 5 * x = 30) : 8 * x = 48 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_result_l1852_185292


namespace NUMINAMATH_CALUDE_parabola_parameter_is_two_l1852_185237

/-- Proves that given a hyperbola and a parabola with specific properties, 
    the parameter of the parabola is 2. -/
theorem parabola_parameter_is_two 
  (a b p : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hp : p > 0) 
  (hyperbola : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (eccentricity : Real.sqrt (1 + b^2 / a^2) = 2)
  (parabola : ∀ x y, y^2 = 2 * p * x)
  (triangle_area : 1/4 * p^2 * b / a = Real.sqrt 3) :
  p = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_parameter_is_two_l1852_185237


namespace NUMINAMATH_CALUDE_distinct_values_in_sequence_l1852_185270

def is_valid_f (f : ℕ → ℕ) : Prop :=
  f 1 = 1 ∧
  (∀ a b : ℕ, 0 < a → 0 < b → a ≤ b → f a ≤ f b) ∧
  (∀ a : ℕ, 0 < a → f (2 * a) = f a + 1)

theorem distinct_values_in_sequence (f : ℕ → ℕ) (hf : is_valid_f f) :
  Finset.card (Finset.image f (Finset.range 2015)) = 11 := by
  sorry

end NUMINAMATH_CALUDE_distinct_values_in_sequence_l1852_185270


namespace NUMINAMATH_CALUDE_drug_use_percentage_is_four_percent_l1852_185251

/-- Warner's Random Response Technique for surveying athletes --/
structure WarnerSurvey where
  total_athletes : ℕ
  yes_answers : ℕ
  prob_odd_roll : ℚ
  prob_even_birthday : ℚ

/-- Calculate the percentage of athletes who have used performance-enhancing drugs --/
def calculate_drug_use_percentage (survey : WarnerSurvey) : ℚ :=
  2 * (survey.yes_answers / survey.total_athletes - 1/4)

/-- Theorem stating that the drug use percentage is 4% for the given survey --/
theorem drug_use_percentage_is_four_percent (survey : WarnerSurvey) 
  (h1 : survey.total_athletes = 200)
  (h2 : survey.yes_answers = 54)
  (h3 : survey.prob_odd_roll = 1/2)
  (h4 : survey.prob_even_birthday = 1/2) :
  calculate_drug_use_percentage survey = 4/100 := by
  sorry

end NUMINAMATH_CALUDE_drug_use_percentage_is_four_percent_l1852_185251


namespace NUMINAMATH_CALUDE_jame_annual_earnings_difference_l1852_185257

/-- Calculates the difference in annual earnings between Jame's new job and old job -/
def annual_earnings_difference (
  new_hourly_rate : ℕ) 
  (new_weekly_hours : ℕ)
  (old_hourly_rate : ℕ)
  (old_weekly_hours : ℕ)
  (weeks_per_year : ℕ) : ℕ :=
  ((new_hourly_rate * new_weekly_hours) - (old_hourly_rate * old_weekly_hours)) * weeks_per_year

/-- Proves that the difference in annual earnings between Jame's new job and old job is $20,800 -/
theorem jame_annual_earnings_difference :
  annual_earnings_difference 20 40 16 25 52 = 20800 := by
  sorry

end NUMINAMATH_CALUDE_jame_annual_earnings_difference_l1852_185257


namespace NUMINAMATH_CALUDE_missing_number_value_l1852_185228

theorem missing_number_value (a b some_number : ℕ) : 
  a = 105 → 
  b = 147 → 
  a^3 = 21 * 25 * some_number * b → 
  some_number = 3 := by
sorry

end NUMINAMATH_CALUDE_missing_number_value_l1852_185228


namespace NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l1852_185209

/-- Proposition p: For all real x, x^2 - 4x + 2m ≥ 0 -/
def proposition_p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 4*x + 2*m ≥ 0

/-- m ≥ 3 is a sufficient condition for proposition p -/
theorem sufficient_condition (m : ℝ) :
  m ≥ 3 → proposition_p m :=
sorry

/-- m ≥ 3 is not a necessary condition for proposition p -/
theorem not_necessary_condition :
  ∃ m : ℝ, m < 3 ∧ proposition_p m :=
sorry

/-- m ≥ 3 is a sufficient but not necessary condition for proposition p -/
theorem sufficient_but_not_necessary :
  (∀ m : ℝ, m ≥ 3 → proposition_p m) ∧
  (∃ m : ℝ, m < 3 ∧ proposition_p m) :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l1852_185209


namespace NUMINAMATH_CALUDE_sphere_wedge_volume_l1852_185229

/-- Given a sphere with circumference 18π inches, cut into 8 congruent wedges,
    the volume of one wedge is 121.5π cubic inches. -/
theorem sphere_wedge_volume (circumference : ℝ) (num_wedges : ℕ) :
  circumference = 18 * Real.pi →
  num_wedges = 8 →
  (1 / num_wedges : ℝ) * (4 / 3 * Real.pi * (circumference / (2 * Real.pi))^3) = 121.5 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_wedge_volume_l1852_185229


namespace NUMINAMATH_CALUDE_kramer_packing_theorem_l1852_185252

/-- The number of boxes Kramer can pack per minute -/
def boxes_per_minute : ℕ := 10

/-- The number of cases Kramer can pack in 2 hours -/
def cases_in_two_hours : ℕ := 240

/-- The number of minutes in 2 hours -/
def minutes_in_two_hours : ℕ := 2 * 60

/-- The number of boxes of cigarettes in one case -/
def boxes_per_case : ℕ := (boxes_per_minute * minutes_in_two_hours) / cases_in_two_hours

theorem kramer_packing_theorem : boxes_per_case = 5 := by
  sorry

end NUMINAMATH_CALUDE_kramer_packing_theorem_l1852_185252


namespace NUMINAMATH_CALUDE_shiela_neighbors_l1852_185207

theorem shiela_neighbors (total_drawings : ℕ) (drawings_per_neighbor : ℕ) 
  (h1 : total_drawings = 54)
  (h2 : drawings_per_neighbor = 9)
  (h3 : total_drawings % drawings_per_neighbor = 0) :
  total_drawings / drawings_per_neighbor = 6 := by
  sorry

end NUMINAMATH_CALUDE_shiela_neighbors_l1852_185207


namespace NUMINAMATH_CALUDE_joe_journey_time_l1852_185293

/-- Represents Joe's journey from home to the store -/
structure JoeJourney where
  walk_speed : ℝ
  run_speed : ℝ
  walk_time : ℝ
  total_distance : ℝ

/-- Theorem: Joe's total journey time is 15 minutes -/
theorem joe_journey_time (j : JoeJourney) 
  (h1 : j.run_speed = 2 * j.walk_speed)
  (h2 : j.walk_time = 10)
  (h3 : j.total_distance = 2 * (j.walk_speed * j.walk_time)) : 
  j.walk_time + (j.total_distance / 2) / j.run_speed = 15 := by
  sorry

#check joe_journey_time

end NUMINAMATH_CALUDE_joe_journey_time_l1852_185293


namespace NUMINAMATH_CALUDE_geometric_series_sum_l1852_185205

theorem geometric_series_sum (a r : ℝ) (n : ℕ) (h : r ≠ 1) :
  let S := (a * (1 - r^n)) / (1 - r)
  a = -1 → r = -3 → n = 10 → S = 14762 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l1852_185205


namespace NUMINAMATH_CALUDE_value_added_to_fraction_l1852_185244

theorem value_added_to_fraction (x y : ℝ) : 
  x = 8 → 0.75 * x + y = 8 → y = 2 := by sorry

end NUMINAMATH_CALUDE_value_added_to_fraction_l1852_185244


namespace NUMINAMATH_CALUDE_series_value_l1852_185272

def series_term (n : ℕ) : ℤ := n * (n + 1) - (n + 1) * (n + 2)

def series_sum : ℕ → ℤ
  | 0 => 0
  | n + 1 => series_sum n + series_term (n + 1)

theorem series_value : series_sum 2000 = 2004002 := by
  sorry

end NUMINAMATH_CALUDE_series_value_l1852_185272


namespace NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l1852_185218

/-- The surface area of a sphere circumscribing a regular square pyramid -/
theorem circumscribed_sphere_surface_area
  (base_edge : ℝ)
  (lateral_edge : ℝ)
  (h_base : base_edge = 2)
  (h_lateral : lateral_edge = Real.sqrt 3)
  : (4 : ℝ) * Real.pi * ((3 : ℝ) / 2) ^ 2 = 9 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l1852_185218


namespace NUMINAMATH_CALUDE_square_area_from_rectangles_l1852_185271

/-- The area of a square formed by three identical rectangles -/
theorem square_area_from_rectangles (width : ℝ) (h1 : width = 4) : 
  let length := 3 * width
  let square_side := length + width
  square_side ^ 2 = 256 := by sorry

end NUMINAMATH_CALUDE_square_area_from_rectangles_l1852_185271


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l1852_185223

/-- Proves that a train with given length and speed takes the calculated time to cross a bridge of given length -/
theorem train_bridge_crossing_time 
  (train_length : Real) 
  (train_speed_kmh : Real) 
  (bridge_length : Real) : 
  train_length = 110 → 
  train_speed_kmh = 72 → 
  bridge_length = 142 → 
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let crossing_time := total_distance / train_speed_ms
  crossing_time = 12.6 := by
  sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l1852_185223


namespace NUMINAMATH_CALUDE_median_of_special_list_l1852_185208

/-- Represents the special list where each number n from 1 to 100 appears n times -/
def special_list : List ℕ := sorry

/-- The length of the special list -/
def list_length : ℕ := (List.range 100).sum + 100

/-- The median of a list is the average of the middle two elements when the list has even length -/
def median (l : List ℕ) : ℚ := sorry

theorem median_of_special_list : median special_list = 71 := by sorry

end NUMINAMATH_CALUDE_median_of_special_list_l1852_185208


namespace NUMINAMATH_CALUDE_mary_eggs_l1852_185296

/-- Given that Mary starts with 27 eggs and finds 4 more, prove that she ends up with 31 eggs. -/
theorem mary_eggs :
  let initial_eggs : ℕ := 27
  let found_eggs : ℕ := 4
  let final_eggs : ℕ := initial_eggs + found_eggs
  final_eggs = 31 := by
sorry

end NUMINAMATH_CALUDE_mary_eggs_l1852_185296


namespace NUMINAMATH_CALUDE_all_hop_sequences_eventually_periodic_l1852_185299

/-- The biggest positive prime number that divides n -/
def f (n : ℕ) : ℕ := sorry

/-- The smallest positive prime number that divides n -/
def g (n : ℕ) : ℕ := sorry

/-- The next position after hopping from n -/
def hop (n : ℕ) : ℕ := f n + g n

/-- A sequence is eventually periodic if it reaches a cycle after some point -/
def EventuallyPeriodic (seq : ℕ → ℕ) : Prop :=
  ∃ (start cycle : ℕ), ∀ n ≥ start, seq (n + cycle) = seq n

/-- The sequence of hops starting from k -/
def hopSequence (k : ℕ) : ℕ → ℕ
  | 0 => k
  | n + 1 => hop (hopSequence k n)

theorem all_hop_sequences_eventually_periodic :
  ∀ k > 1, EventuallyPeriodic (hopSequence k) := by sorry

end NUMINAMATH_CALUDE_all_hop_sequences_eventually_periodic_l1852_185299


namespace NUMINAMATH_CALUDE_cody_caramel_boxes_l1852_185250

-- Define the given conditions
def chocolate_boxes : ℕ := 7
def pieces_per_box : ℕ := 8
def total_pieces : ℕ := 80

-- Define the function to calculate the number of caramel boxes
def caramel_boxes : ℕ :=
  (total_pieces - chocolate_boxes * pieces_per_box) / pieces_per_box

-- Theorem statement
theorem cody_caramel_boxes :
  caramel_boxes = 3 := by
  sorry

end NUMINAMATH_CALUDE_cody_caramel_boxes_l1852_185250


namespace NUMINAMATH_CALUDE_soccer_ball_max_height_l1852_185294

/-- The height of the soccer ball as a function of time -/
def h (t : ℝ) : ℝ := -20 * t^2 + 40 * t + 20

/-- Theorem stating that the maximum height reached by the soccer ball is 40 feet -/
theorem soccer_ball_max_height :
  ∃ (max : ℝ), max = 40 ∧ ∀ (t : ℝ), h t ≤ max :=
sorry

end NUMINAMATH_CALUDE_soccer_ball_max_height_l1852_185294


namespace NUMINAMATH_CALUDE_second_discount_percentage_l1852_185202

theorem second_discount_percentage
  (original_price : ℝ)
  (first_discount_percent : ℝ)
  (final_price : ℝ)
  (h1 : original_price = 175)
  (h2 : first_discount_percent = 20)
  (h3 : final_price = 133)
  : ∃ (second_discount_percent : ℝ),
    final_price = original_price * (1 - first_discount_percent / 100) * (1 - second_discount_percent / 100) ∧
    second_discount_percent = 5 :=
sorry

end NUMINAMATH_CALUDE_second_discount_percentage_l1852_185202


namespace NUMINAMATH_CALUDE_sin_pi_plus_2alpha_l1852_185246

theorem sin_pi_plus_2alpha (α : ℝ) (h : Real.sin (α - π/4) = 3/5) :
  Real.sin (π + 2*α) = -7/25 := by sorry

end NUMINAMATH_CALUDE_sin_pi_plus_2alpha_l1852_185246


namespace NUMINAMATH_CALUDE_count_odd_rank_subsets_l1852_185269

/-- The number of cards in the deck -/
def total_cards : ℕ := 8056

/-- The number of ranks in the deck -/
def total_ranks : ℕ := 2014

/-- The number of suits per rank -/
def suits_per_rank : ℕ := 4

/-- The number of subsets with cards from an odd number of distinct ranks -/
def odd_rank_subsets : ℕ := (16^total_ranks - 14^total_ranks) / 2

/-- Theorem stating the number of subsets with cards from an odd number of distinct ranks -/
theorem count_odd_rank_subsets :
  total_cards = total_ranks * suits_per_rank →
  odd_rank_subsets = (16^total_ranks - 14^total_ranks) / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_count_odd_rank_subsets_l1852_185269


namespace NUMINAMATH_CALUDE_work_done_by_force_l1852_185273

/-- Work done by a force on a particle -/
theorem work_done_by_force (F S : ℝ × ℝ) : 
  F = (-1, -2) → S = (3, 4) → F.1 * S.1 + F.2 * S.2 = -11 := by sorry

end NUMINAMATH_CALUDE_work_done_by_force_l1852_185273


namespace NUMINAMATH_CALUDE_triangle_geometric_sequence_cosine_l1852_185275

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if a, b, c form a geometric sequence and c = 2a, then cos B = 1/√2 -/
theorem triangle_geometric_sequence_cosine (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Ensure positive side lengths
  A > 0 ∧ B > 0 ∧ C > 0 →  -- Ensure positive angles
  A + B + C = π →  -- Sum of angles in a triangle
  c^2 = a^2 + b^2 - 2*a*b*Real.cos C →  -- Law of cosines
  (∃ r : ℝ, b = a*r ∧ c = b*r) →  -- Geometric sequence condition
  c = 2*a →  -- Given condition
  Real.cos B = 1 / Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_geometric_sequence_cosine_l1852_185275
