import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l522_52235

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : Real.sqrt x + Real.sqrt y + Real.sqrt z = 1) :
  (x^2 + y*z) / Real.sqrt (2*x^2*(y+z)) +
  (y^2 + z*x) / Real.sqrt (2*y^2*(z+x)) +
  (z^2 + x*y) / Real.sqrt (2*z^2*(x+y)) ≥ 1 := by
    sorry

end NUMINAMATH_CALUDE_inequality_proof_l522_52235


namespace NUMINAMATH_CALUDE_sufficient_condition_for_positive_quadratic_l522_52265

theorem sufficient_condition_for_positive_quadratic (m : ℝ) :
  m > 1 → ∀ x : ℝ, x^2 - 2*x + m > 0 := by sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_positive_quadratic_l522_52265


namespace NUMINAMATH_CALUDE_fraction_equality_implies_zero_l522_52293

theorem fraction_equality_implies_zero (x : ℝ) : 
  (4 + x) / (6 + x) = (2 + x) / (3 + x) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_zero_l522_52293


namespace NUMINAMATH_CALUDE_quadratic_inequality_l522_52204

/-- The quadratic function f(x) = -2(x+1)^2 + k -/
def f (k : ℝ) (x : ℝ) : ℝ := -2 * (x + 1)^2 + k

/-- Theorem stating the inequality between f(2), f(-3), and f(-0.5) -/
theorem quadratic_inequality (k : ℝ) : f k 2 < f k (-3) ∧ f k (-3) < f k (-0.5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l522_52204


namespace NUMINAMATH_CALUDE_dividend_calculation_l522_52297

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h_divisor : divisor = 15)
  (h_quotient : quotient = 8)
  (h_remainder : remainder = 5) :
  divisor * quotient + remainder = 125 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l522_52297


namespace NUMINAMATH_CALUDE_lcm_of_36_and_220_l522_52292

theorem lcm_of_36_and_220 : Nat.lcm 36 220 = 1980 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_36_and_220_l522_52292


namespace NUMINAMATH_CALUDE_oil_bill_ratio_change_l522_52298

theorem oil_bill_ratio_change 
  (january_bill : ℝ) 
  (february_bill : ℝ) 
  (initial_ratio : ℚ) 
  (added_amount : ℝ) :
  january_bill = 59.99999999999997 →
  initial_ratio = 3 / 2 →
  february_bill / january_bill = initial_ratio →
  (february_bill + added_amount) / january_bill = 5 / 3 →
  added_amount = 10 := by
sorry

end NUMINAMATH_CALUDE_oil_bill_ratio_change_l522_52298


namespace NUMINAMATH_CALUDE_tv_price_change_l522_52212

theorem tv_price_change (initial_price : ℝ) (x : ℝ) 
  (h1 : initial_price > 0) 
  (h2 : x > 0) : 
  (initial_price * 0.8 * (1 + x / 100) = initial_price * 1.12) → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_tv_price_change_l522_52212


namespace NUMINAMATH_CALUDE_intersection_perpendicular_line_l522_52205

-- Define the lines
def line1 (x y : ℝ) : Prop := 3 * x + y = 0
def line2 (x y : ℝ) : Prop := x + y - 2 = 0
def line3 (x y : ℝ) : Prop := 2 * x + y + 3 = 0

-- Define the result line
def result_line (x y : ℝ) : Prop := x - 2 * y + 7 = 0

-- Theorem statement
theorem intersection_perpendicular_line :
  ∃ (x₀ y₀ : ℝ),
    (line1 x₀ y₀ ∧ line2 x₀ y₀) ∧
    (∀ (x y : ℝ), result_line x y → 
      ((x - x₀) * 2 + (y - y₀) * 1 = 0)) ∧
    result_line x₀ y₀ :=
sorry

end NUMINAMATH_CALUDE_intersection_perpendicular_line_l522_52205


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l522_52266

theorem sufficient_not_necessary (x : ℝ) : 
  (∀ x, x > 1 → x^2 + x - 2 > 0) ∧ 
  (∃ x, x^2 + x - 2 > 0 ∧ x ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l522_52266


namespace NUMINAMATH_CALUDE_min_total_books_l522_52231

/-- Represents the number of books for each subject in the library. -/
structure LibraryBooks where
  physics : ℕ
  chemistry : ℕ
  biology : ℕ
  mathematics : ℕ
  history : ℕ

/-- Defines the conditions for the library books problem. -/
def LibraryBooksProblem (books : LibraryBooks) : Prop :=
  books.physics * 2 = books.chemistry * 3 ∧
  books.chemistry * 3 = books.biology * 4 ∧
  books.biology * 6 = books.mathematics * 5 ∧
  books.mathematics * 8 = books.history * 7 ∧
  books.mathematics ≥ 1000 ∧
  books.physics + books.chemistry + books.biology + books.mathematics + books.history > 10000

/-- Theorem stating the minimum possible total number of books in the library. -/
theorem min_total_books (books : LibraryBooks) (h : LibraryBooksProblem books) :
  books.physics + books.chemistry + books.biology + books.mathematics + books.history = 10050 :=
by
  sorry


end NUMINAMATH_CALUDE_min_total_books_l522_52231


namespace NUMINAMATH_CALUDE_red_pencils_count_l522_52215

/-- Given a box of pencils with blue, red, and green colors, prove that the number of red pencils is 6 --/
theorem red_pencils_count (B R G : ℕ) : 
  B + R + G = 20 →  -- Total number of pencils
  B = 6 * G →       -- Blue pencils are 6 times green pencils
  R < B →           -- Fewer red pencils than blue ones
  R = 6 :=
by sorry

end NUMINAMATH_CALUDE_red_pencils_count_l522_52215


namespace NUMINAMATH_CALUDE_ratio_average_l522_52273

theorem ratio_average (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  b / a = 5 / 4 →
  c / a = 6 / 4 →
  c = 24 →
  (a + b + c) / 3 = 20 := by
sorry

end NUMINAMATH_CALUDE_ratio_average_l522_52273


namespace NUMINAMATH_CALUDE_line_symmetry_l522_52243

/-- Given two lines l₁ and l, prove that l₂ is symmetric to l₁ with respect to l -/
theorem line_symmetry (x y : ℝ) : 
  let l₁ : ℝ → ℝ → Prop := λ x y ↦ 2 * x + y - 4 = 0
  let l : ℝ → ℝ → Prop := λ x y ↦ 3 * x + 4 * y - 1 = 0
  let l₂ : ℝ → ℝ → Prop := λ x y ↦ 2 * x + y - 6 = 0
  (∀ x y, l₁ x y ↔ l₂ x y) ∧ 
  (∀ x₁ y₁ x₂ y₂, l₁ x₁ y₁ → l₂ x₂ y₂ → 
    ∃ x₀ y₀, l x₀ y₀ ∧ 
    (x₀ - x₁)^2 + (y₀ - y₁)^2 = (x₀ - x₂)^2 + (y₀ - y₂)^2) :=
by sorry

end NUMINAMATH_CALUDE_line_symmetry_l522_52243


namespace NUMINAMATH_CALUDE_factorial_ratio_equals_two_l522_52261

theorem factorial_ratio_equals_two : (Nat.factorial 10 * Nat.factorial 4 * Nat.factorial 3) / (Nat.factorial 9 * Nat.factorial 6) = 2 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_equals_two_l522_52261


namespace NUMINAMATH_CALUDE_conference_duration_theorem_l522_52240

/-- The duration of the conference in minutes -/
def conference_duration (first_session_hours : ℕ) (first_session_minutes : ℕ) 
  (second_session_hours : ℕ) (second_session_minutes : ℕ) : ℕ :=
  (first_session_hours * 60 + first_session_minutes) + 
  (second_session_hours * 60 + second_session_minutes)

/-- Theorem stating the total duration of the conference -/
theorem conference_duration_theorem : 
  conference_duration 8 15 3 40 = 715 := by sorry

end NUMINAMATH_CALUDE_conference_duration_theorem_l522_52240


namespace NUMINAMATH_CALUDE_equation_solution_l522_52236

theorem equation_solution :
  ∃! x : ℝ, (x^2 + 4*x + 5) / (x + 3) = x + 7 ∧ x ≠ -3 :=
by
  use (-8/3)
  sorry

end NUMINAMATH_CALUDE_equation_solution_l522_52236


namespace NUMINAMATH_CALUDE_half_percent_of_150_in_paise_l522_52233

/-- Converts rupees to paise -/
def rupees_to_paise (r : ℚ) : ℚ := 100 * r

/-- Calculates the percentage of a given value -/
def percentage_of (p : ℚ) (v : ℚ) : ℚ := (p / 100) * v

theorem half_percent_of_150_in_paise : 
  rupees_to_paise (percentage_of 0.5 150) = 75 := by
  sorry

end NUMINAMATH_CALUDE_half_percent_of_150_in_paise_l522_52233


namespace NUMINAMATH_CALUDE_range_of_a_l522_52247

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 - 2*a*x + 2 < 0) → a ∈ Set.Icc (-Real.sqrt 2) (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l522_52247


namespace NUMINAMATH_CALUDE_armstrong_made_quote_l522_52218

-- Define the type for astronauts
inductive Astronaut : Type
| Apollo : Astronaut
| MichaelCollins : Astronaut
| Armstrong : Astronaut
| Aldrin : Astronaut

-- Define the famous quote
def famous_quote : String := "That's one small step for man, one giant leap for mankind."

-- Define the property of making the quote on the Moon
def made_quote_on_moon (a : Astronaut) : Prop := 
  a = Astronaut.Armstrong ∧ ∃ (quote : String), quote = famous_quote

-- Theorem stating that Armstrong made the famous quote on the Moon
theorem armstrong_made_quote : 
  ∃ (a : Astronaut), made_quote_on_moon a :=
sorry

end NUMINAMATH_CALUDE_armstrong_made_quote_l522_52218


namespace NUMINAMATH_CALUDE_first_platform_length_is_150_l522_52241

/-- The length of a train in meters -/
def train_length : ℝ := 150

/-- The length of the second platform in meters -/
def second_platform_length : ℝ := 250

/-- The time taken to cross the first platform in seconds -/
def time_first_platform : ℝ := 15

/-- The time taken to cross the second platform in seconds -/
def time_second_platform : ℝ := 20

/-- The length of the first platform in meters -/
def first_platform_length : ℝ := 150

theorem first_platform_length_is_150 :
  (train_length + first_platform_length) / time_first_platform =
  (train_length + second_platform_length) / time_second_platform :=
sorry

end NUMINAMATH_CALUDE_first_platform_length_is_150_l522_52241


namespace NUMINAMATH_CALUDE_farmer_pumpkin_seeds_per_row_l522_52239

/-- Represents the farmer's planting scenario -/
structure FarmerPlanting where
  bean_seedlings : ℕ
  bean_per_row : ℕ
  pumpkin_seeds : ℕ
  radishes : ℕ
  radish_per_row : ℕ
  rows_per_bed : ℕ
  plant_beds : ℕ

/-- Calculates the number of pumpkin seeds per row -/
def pumpkin_seeds_per_row (fp : FarmerPlanting) : ℕ :=
  fp.pumpkin_seeds / (fp.plant_beds * fp.rows_per_bed - fp.bean_seedlings / fp.bean_per_row - fp.radishes / fp.radish_per_row)

/-- Theorem stating that given the specific planting scenario, the farmer plants 7 pumpkin seeds per row -/
theorem farmer_pumpkin_seeds_per_row :
  let fp : FarmerPlanting := {
    bean_seedlings := 64,
    bean_per_row := 8,
    pumpkin_seeds := 84,
    radishes := 48,
    radish_per_row := 6,
    rows_per_bed := 2,
    plant_beds := 14
  }
  pumpkin_seeds_per_row fp = 7 := by
  sorry

end NUMINAMATH_CALUDE_farmer_pumpkin_seeds_per_row_l522_52239


namespace NUMINAMATH_CALUDE_smallest_multiple_of_3_to_7_l522_52264

theorem smallest_multiple_of_3_to_7 : 
  ∃ (N : ℕ), N > 0 ∧ 
    (∀ (k : ℕ), k > 0 ∧ k < N → 
      ¬(3 ∣ k ∧ 4 ∣ k ∧ 5 ∣ k ∧ 6 ∣ k ∧ 7 ∣ k)) ∧
    (3 ∣ N ∧ 4 ∣ N ∧ 5 ∣ N ∧ 6 ∣ N ∧ 7 ∣ N) ∧
    N = 420 :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_3_to_7_l522_52264


namespace NUMINAMATH_CALUDE_probability_exactly_two_ones_equals_fraction_l522_52272

def num_dice : ℕ := 12
def num_sides : ℕ := 6
def target_outcome : ℕ := 1
def target_count : ℕ := 2

def probability_exactly_two_ones : ℚ :=
  (num_dice.choose target_count : ℚ) * 
  (1 / num_sides) ^ target_count * 
  ((num_sides - 1) / num_sides) ^ (num_dice - target_count)

theorem probability_exactly_two_ones_equals_fraction :
  probability_exactly_two_ones = (66 * 5^10 : ℚ) / (36 * 6^10) := by
  sorry

end NUMINAMATH_CALUDE_probability_exactly_two_ones_equals_fraction_l522_52272


namespace NUMINAMATH_CALUDE_f_properties_l522_52286

noncomputable def f (x : ℝ) := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + Real.cos (2 * x) + 3

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ 
    ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∃ (M : ℝ), M = 5 ∧ ∀ (x : ℝ), 0 ≤ x ∧ x ≤ π/4 → f x ≤ M) ∧
  f (π/6) = 5 := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l522_52286


namespace NUMINAMATH_CALUDE_molecular_weight_Al2S3_l522_52263

/-- The atomic weight of Aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of Sulfur in g/mol -/
def atomic_weight_S : ℝ := 32.06

/-- The number of Aluminum atoms in Al2S3 -/
def num_Al_atoms : ℕ := 2

/-- The number of Sulfur atoms in Al2S3 -/
def num_S_atoms : ℕ := 3

/-- The number of moles of Al2S3 -/
def num_moles : ℝ := 10

/-- Theorem: The molecular weight of 10 moles of Al2S3 is 1501.4 grams -/
theorem molecular_weight_Al2S3 : 
  (num_Al_atoms : ℝ) * atomic_weight_Al * num_moles + 
  (num_S_atoms : ℝ) * atomic_weight_S * num_moles = 1501.4 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_Al2S3_l522_52263


namespace NUMINAMATH_CALUDE_add_9999_seconds_to_1645_l522_52283

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat
  deriving Repr

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

theorem add_9999_seconds_to_1645 :
  let initial_time : Time := ⟨16, 45, 0⟩
  let seconds_to_add : Nat := 9999
  let final_time : Time := addSeconds initial_time seconds_to_add
  final_time = ⟨19, 31, 39⟩ := by sorry

end NUMINAMATH_CALUDE_add_9999_seconds_to_1645_l522_52283


namespace NUMINAMATH_CALUDE_product_of_powers_of_ten_l522_52229

theorem product_of_powers_of_ten : (10^0.4) * (10^0.6) * (10^0.3) * (10^0.2) * (10^0.5) = 100 := by
  sorry

end NUMINAMATH_CALUDE_product_of_powers_of_ten_l522_52229


namespace NUMINAMATH_CALUDE_H_triple_2_l522_52287

/-- The function H defined as H(x) = 2x - 1 for all real x -/
def H (x : ℝ) : ℝ := 2 * x - 1

/-- Theorem stating that H(H(H(2))) = 9 -/
theorem H_triple_2 : H (H (H 2)) = 9 := by
  sorry

end NUMINAMATH_CALUDE_H_triple_2_l522_52287


namespace NUMINAMATH_CALUDE_polygon_internal_angle_sum_l522_52256

theorem polygon_internal_angle_sum (n : ℕ) (h : n > 2) :
  let external_angle : ℚ := 40
  let internal_angle_sum : ℚ := (n - 2) * 180
  external_angle * n = 360 → internal_angle_sum = 1260 := by
sorry

end NUMINAMATH_CALUDE_polygon_internal_angle_sum_l522_52256


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l522_52211

theorem simplify_and_evaluate (x y : ℚ) (hx : x = 1/2) (hy : y = -1) :
  (x - 3*y)^2 - (x - y)*(x + 2*y) = 29/2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l522_52211


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l522_52216

theorem quadratic_inequality_solution (a : ℝ) : 
  (a > 0 ∧ ∃ x : ℝ, x^2 - 8*x + a < 0) ↔ (0 < a ∧ a < 16) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l522_52216


namespace NUMINAMATH_CALUDE_cube_congruence_l522_52253

theorem cube_congruence (a b : ℕ) : a ≡ b [MOD 1000] → a^3 ≡ b^3 [MOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_cube_congruence_l522_52253


namespace NUMINAMATH_CALUDE_middle_term_is_plus_minus_six_l522_52222

/-- The coefficient of the middle term in the expansion of (a ± 3b)² -/
def middle_term_coefficient (a b : ℝ) : Set ℝ :=
  {x : ℝ | ∃ (sign : ℝ) (h : sign = 1 ∨ sign = -1), 
    (a + sign * 3 * b)^2 = a^2 + x * a * b + 9 * b^2}

/-- Theorem stating that the coefficient of the middle term is either 6 or -6 -/
theorem middle_term_is_plus_minus_six (a b : ℝ) : 
  middle_term_coefficient a b = {6, -6} := by
sorry

end NUMINAMATH_CALUDE_middle_term_is_plus_minus_six_l522_52222


namespace NUMINAMATH_CALUDE_g_composition_of_three_l522_52259

def g (x : ℝ) : ℝ := 3 * x + 2

theorem g_composition_of_three : g (g (g 3)) = 107 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_of_three_l522_52259


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l522_52250

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  q ≠ 1 →
  (∀ n, a n > 0) →
  (a 3 + a 6 = 2 * a 5) →
  (a 3 + a 4) / (a 4 + a 5) = (-1 + Real.sqrt 5) / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l522_52250


namespace NUMINAMATH_CALUDE_savings_percentage_is_10_percent_l522_52278

def basic_salary : ℝ := 240
def sales : ℝ := 2500
def commission_rate : ℝ := 0.02
def savings : ℝ := 29

def commission : ℝ := sales * commission_rate
def total_earnings : ℝ := basic_salary + commission

theorem savings_percentage_is_10_percent :
  (savings / total_earnings) * 100 = 10 := by sorry

end NUMINAMATH_CALUDE_savings_percentage_is_10_percent_l522_52278


namespace NUMINAMATH_CALUDE_negation_of_proposition_l522_52282

theorem negation_of_proposition (p : Prop) :
  (p ↔ ∃ x, x < 1 ∧ x^2 ≤ 1) →
  (¬p ↔ ∀ x, x < 1 → x^2 > 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l522_52282


namespace NUMINAMATH_CALUDE_average_of_solutions_is_zero_l522_52230

theorem average_of_solutions_is_zero :
  let f : ℝ → ℝ := fun x => Real.sqrt (3 * x^2 + 4)
  let solutions := {x : ℝ | f x = Real.sqrt 28}
  ∃ (x₁ x₂ : ℝ), x₁ ∈ solutions ∧ x₂ ∈ solutions ∧ x₁ ≠ x₂ ∧
    (x₁ + x₂) / 2 = 0 ∧
    ∀ (x : ℝ), x ∈ solutions → (x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_average_of_solutions_is_zero_l522_52230


namespace NUMINAMATH_CALUDE_tom_marbles_groups_l522_52227

/-- Represents the colors of marbles --/
inductive MarbleColor
  | Red
  | Green
  | Blue
  | Yellow

/-- Represents Tom's collection of marbles --/
structure MarbleCollection where
  red : Nat
  green : Nat
  blue : Nat
  yellow : Nat

/-- Calculates the number of different groups of two marbles that can be chosen --/
def countDifferentGroups (collection : MarbleCollection) : Nat :=
  sorry

/-- Theorem stating that Tom's specific collection results in 12 different groups --/
theorem tom_marbles_groups :
  let toms_collection : MarbleCollection := {
    red := 1,
    green := 1,
    blue := 2,
    yellow := 3
  }
  countDifferentGroups toms_collection = 12 := by
  sorry

end NUMINAMATH_CALUDE_tom_marbles_groups_l522_52227


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l522_52255

def f (x : ℝ) := x^3 - 3*x - 3

theorem root_exists_in_interval :
  ∃ x ∈ Set.Ioo 2 3, f x = 0 :=
sorry

end NUMINAMATH_CALUDE_root_exists_in_interval_l522_52255


namespace NUMINAMATH_CALUDE_twenty_first_term_is_4641_l522_52257

/-- The sequence where each term is the sum of consecutive integers, 
    and the number of integers in each group increases by 1 -/
def sequence_term (n : ℕ) : ℕ :=
  let first_num := 1 + (n * (n - 1)) / 2
  let last_num := first_num + n - 1
  n * (first_num + last_num) / 2

/-- The 21st term of the sequence is 4641 -/
theorem twenty_first_term_is_4641 : sequence_term 21 = 4641 := by
  sorry

end NUMINAMATH_CALUDE_twenty_first_term_is_4641_l522_52257


namespace NUMINAMATH_CALUDE_prime_pair_sum_10_product_21_l522_52209

theorem prime_pair_sum_10_product_21 : 
  ∃! (p q : ℕ), Prime p ∧ Prime q ∧ p + q = 10 ∧ p * q = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_prime_pair_sum_10_product_21_l522_52209


namespace NUMINAMATH_CALUDE_square_root_of_one_fourth_l522_52271

theorem square_root_of_one_fourth :
  {x : ℚ | x^2 = (1 : ℚ) / 4} = {(1 : ℚ) / 2, -(1 : ℚ) / 2} := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_one_fourth_l522_52271


namespace NUMINAMATH_CALUDE_equilateral_triangles_with_squares_l522_52248

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Represents a square -/
structure Square :=
  (A : Point)
  (B : Point)
  (C : Point)
  (D : Point)

/-- Checks if a triangle is equilateral -/
def is_equilateral (t : Triangle) : Prop :=
  sorry

/-- Constructs a square externally on a side of a triangle -/
def construct_external_square (t : Triangle) (side : Fin 3) : Square :=
  sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ :=
  sorry

/-- The main theorem -/
theorem equilateral_triangles_with_squares
  (ABC BCD : Triangle)
  (ABEF : Square)
  (CDGH : Square)
  (h1 : is_equilateral ABC)
  (h2 : is_equilateral BCD)
  (h3 : ABEF = construct_external_square ABC 0)
  (h4 : CDGH = construct_external_square BCD 1)
  : distance ABEF.C CDGH.C / distance ABC.B ABC.C = 3 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangles_with_squares_l522_52248


namespace NUMINAMATH_CALUDE_art_gallery_display_ratio_l522_52220

theorem art_gallery_display_ratio :
  let total_pieces : ℕ := 2700
  let sculptures_not_displayed : ℕ := 1200
  let paintings_not_displayed : ℕ := sculptures_not_displayed / 3
  let pieces_not_displayed : ℕ := sculptures_not_displayed + paintings_not_displayed
  let pieces_displayed : ℕ := total_pieces - pieces_not_displayed
  let sculptures_displayed : ℕ := pieces_displayed / 6
  pieces_displayed / total_pieces = 11 / 27 :=
by
  sorry

end NUMINAMATH_CALUDE_art_gallery_display_ratio_l522_52220


namespace NUMINAMATH_CALUDE_josh_candies_left_to_share_l522_52280

/-- The number of candies left to be shared with others after Josh distributes and eats some. -/
def candies_left_to_share (initial_candies : ℕ) (siblings : ℕ) (candies_per_sibling : ℕ) (candies_to_eat : ℕ) : ℕ :=
  let remaining_after_siblings := initial_candies - siblings * candies_per_sibling
  let remaining_after_friend := remaining_after_siblings / 2
  remaining_after_friend - candies_to_eat

/-- Theorem stating that given Josh's initial conditions, there are 19 candies left to share. -/
theorem josh_candies_left_to_share :
  candies_left_to_share 100 3 10 16 = 19 := by
  sorry


end NUMINAMATH_CALUDE_josh_candies_left_to_share_l522_52280


namespace NUMINAMATH_CALUDE_teal_color_survey_l522_52242

theorem teal_color_survey (total : ℕ) (more_green : ℕ) (both : ℕ) (neither : ℕ) :
  total = 150 →
  more_green = 90 →
  both = 40 →
  neither = 25 →
  ∃ (more_blue : ℕ), more_blue = 75 ∧ 
    more_blue + (more_green - both) + neither = total :=
by sorry

end NUMINAMATH_CALUDE_teal_color_survey_l522_52242


namespace NUMINAMATH_CALUDE_quadratic_inequality_l522_52277

/-- The quadratic function f(x) = ax^2 + bx + c -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The solution set of ax^2 + bx + c > 0 is {x | -2 < x < 4} -/
def solution_set (a b c : ℝ) : Set ℝ := {x | -2 < x ∧ x < 4}

theorem quadratic_inequality (a b c : ℝ) 
  (h : ∀ x, x ∈ solution_set a b c ↔ f a b c x > 0) :
  f a b c 2 > f a b c (-1) ∧ f a b c (-1) > f a b c 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l522_52277


namespace NUMINAMATH_CALUDE_count_integer_points_l522_52260

def point_A : ℤ × ℤ := (2, 3)
def point_B : ℤ × ℤ := (150, 903)

def is_between (p q r : ℤ × ℤ) : Prop :=
  (p.1 < q.1 ∧ q.1 < r.1) ∨ (r.1 < q.1 ∧ q.1 < p.1)

def on_line (p q r : ℤ × ℤ) : Prop :=
  (r.2 - p.2) * (q.1 - p.1) = (q.2 - p.2) * (r.1 - p.1)

def integer_points_between : Prop :=
  ∃ (S : Finset (ℤ × ℤ)),
    S.card = 4 ∧
    (∀ p ∈ S, is_between point_A p point_B ∧ on_line point_A point_B p) ∧
    (∀ p : ℤ × ℤ, is_between point_A p point_B ∧ on_line point_A point_B p → p ∈ S)

theorem count_integer_points : integer_points_between := by
  sorry

end NUMINAMATH_CALUDE_count_integer_points_l522_52260


namespace NUMINAMATH_CALUDE_matrix_sum_equality_l522_52262

def A : Matrix (Fin 2) (Fin 2) ℤ := !![4, -3; 0, 5]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![-6, 8; 7, -10]

theorem matrix_sum_equality : A + B = !![(-2 : ℤ), 5; 7, -5] := by sorry

end NUMINAMATH_CALUDE_matrix_sum_equality_l522_52262


namespace NUMINAMATH_CALUDE_max_diagonal_sum_l522_52202

/-- A rhombus with side length 5 and diagonals d1 and d2 -/
structure Rhombus where
  side_length : ℝ
  d1 : ℝ
  d2 : ℝ
  side_is_5 : side_length = 5
  d1_le_6 : d1 ≤ 6
  d2_ge_6 : d2 ≥ 6

/-- The maximum sum of diagonals in a rhombus with given constraints is 14 -/
theorem max_diagonal_sum (r : Rhombus) : (r.d1 + r.d2 ≤ 14) ∧ (∃ (s : Rhombus), s.d1 + s.d2 = 14) :=
  sorry

end NUMINAMATH_CALUDE_max_diagonal_sum_l522_52202


namespace NUMINAMATH_CALUDE_two_sevens_numeral_l522_52217

/-- Given two sevens in a numeral with a difference of 69930 between their place values,
    prove that the numeral is 7700070. -/
theorem two_sevens_numeral (A B : ℕ) : 
  A - B = 69930 →
  A = 10 * B →
  A = 77700 ∧ B = 7770 ∧ 7700070 = 7 * A + 7 * B :=
by sorry

end NUMINAMATH_CALUDE_two_sevens_numeral_l522_52217


namespace NUMINAMATH_CALUDE_min_value_fourth_root_plus_reciprocal_l522_52296

theorem min_value_fourth_root_plus_reciprocal (x : ℝ) (hx : x > 0) :
  2 * x^(1/4) + 1/x ≥ 3 ∧ (2 * x^(1/4) + 1/x = 3 ↔ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_fourth_root_plus_reciprocal_l522_52296


namespace NUMINAMATH_CALUDE_circle_area_increase_l522_52288

theorem circle_area_increase (r : ℝ) (h : r > 0) :
  let new_radius := 1.5 * r
  let original_area := π * r^2
  let new_area := π * new_radius^2
  (new_area - original_area) / original_area = 1.25 := by
sorry

end NUMINAMATH_CALUDE_circle_area_increase_l522_52288


namespace NUMINAMATH_CALUDE_sum_234_78_base5_l522_52210

/-- Converts a natural number from base 10 to base 5 -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 5 to a natural number in base 10 -/
def fromBase5 (digits : List ℕ) : ℕ :=
  sorry

theorem sum_234_78_base5 : 
  toBase5 (234 + 78) = [2, 2, 2, 2] := by sorry

end NUMINAMATH_CALUDE_sum_234_78_base5_l522_52210


namespace NUMINAMATH_CALUDE_hoopit_students_count_l522_52206

/-- Represents the number of toes on each hand for Hoopits -/
def hoopit_toes_per_hand : ℕ := 3

/-- Represents the number of hands for Hoopits -/
def hoopit_hands : ℕ := 4

/-- Represents the number of toes on each hand for Neglarts -/
def neglart_toes_per_hand : ℕ := 2

/-- Represents the number of hands for Neglarts -/
def neglart_hands : ℕ := 5

/-- Represents the number of Neglart students on the bus -/
def neglart_students : ℕ := 8

/-- Represents the total number of toes on the bus -/
def total_toes : ℕ := 164

/-- Theorem stating that the number of Hoopit students on the bus is 7 -/
theorem hoopit_students_count : 
  ∃ (h : ℕ), h * (hoopit_toes_per_hand * hoopit_hands) + 
             neglart_students * (neglart_toes_per_hand * neglart_hands) = total_toes ∧ 
             h = 7 := by
  sorry

end NUMINAMATH_CALUDE_hoopit_students_count_l522_52206


namespace NUMINAMATH_CALUDE_hotel_charge_comparison_l522_52254

/-- 
Given three hotels P, R, and G, where:
- The charge for a single room at hotel P is 40% less than hotel R
- The charge for a single room at hotel P is 10% less than hotel G

Prove that the charge for a single room at hotel R is 50% greater than 
the charge for a single room at hotel G.
-/
theorem hotel_charge_comparison (P R G : ℝ) 
  (h1 : P = R * (1 - 0.4))
  (h2 : P = G * (1 - 0.1)) :
  R = G * 1.5 := by sorry

end NUMINAMATH_CALUDE_hotel_charge_comparison_l522_52254


namespace NUMINAMATH_CALUDE_arun_weight_average_l522_52285

-- Define Arun's weight as a real number
def arun_weight : ℝ := sorry

-- Define the conditions on Arun's weight
def condition1 : Prop := 61 < arun_weight ∧ arun_weight < 72
def condition2 : Prop := 60 < arun_weight ∧ arun_weight < 70
def condition3 : Prop := arun_weight ≤ 64
def condition4 : Prop := 62 < arun_weight ∧ arun_weight < 73
def condition5 : Prop := 59 < arun_weight ∧ arun_weight < 68

-- Theorem stating that the average of possible weights is 63.5
theorem arun_weight_average :
  condition1 ∧ condition2 ∧ condition3 ∧ condition4 ∧ condition5 →
  (63 + 64) / 2 = 63.5 :=
by sorry

end NUMINAMATH_CALUDE_arun_weight_average_l522_52285


namespace NUMINAMATH_CALUDE_square_side_length_range_l522_52258

theorem square_side_length_range (area : ℝ) (h : area = 15) :
  ∃ x : ℝ, x > 3 ∧ x < 4 ∧ x^2 = area := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_range_l522_52258


namespace NUMINAMATH_CALUDE_distinct_cube_paintings_eq_30_l522_52238

/-- The number of faces on a cube -/
def num_faces : ℕ := 6

/-- The number of available colors -/
def num_colors : ℕ := 6

/-- The number of rotational symmetries of a cube -/
def num_rotations : ℕ := 24

/-- The number of distinct ways to paint a cube -/
def distinct_cube_paintings : ℕ := (num_colors.factorial) / num_rotations

theorem distinct_cube_paintings_eq_30 : distinct_cube_paintings = 30 := by
  sorry

end NUMINAMATH_CALUDE_distinct_cube_paintings_eq_30_l522_52238


namespace NUMINAMATH_CALUDE_rectangle_area_equals_perimeter_l522_52207

theorem rectangle_area_equals_perimeter (a b : ℕ) : 
  a ≠ b →  -- non-square condition
  a * b = 2 * (a + b) →  -- area equals perimeter condition
  2 * (a + b) = 18 :=  -- conclusion: perimeter is 18
by sorry

end NUMINAMATH_CALUDE_rectangle_area_equals_perimeter_l522_52207


namespace NUMINAMATH_CALUDE_female_workers_l522_52251

/-- Represents the number of workers in a company --/
structure Company where
  male : ℕ
  female : ℕ
  male_no_plan : ℕ
  female_no_plan : ℕ

/-- The conditions of the company --/
def company_conditions (c : Company) : Prop :=
  c.male = 112 ∧
  c.male_no_plan = (40 * c.male) / 100 ∧
  c.female_no_plan = (25 * c.female) / 100 ∧
  (30 * (c.male_no_plan + c.female_no_plan)) / 100 = c.male_no_plan ∧
  (60 * (c.male - c.male_no_plan + c.female - c.female_no_plan)) / 100 = (c.male - c.male_no_plan)

/-- The theorem to be proved --/
theorem female_workers (c : Company) : company_conditions c → c.female = 420 := by
  sorry

end NUMINAMATH_CALUDE_female_workers_l522_52251


namespace NUMINAMATH_CALUDE_coin_problem_l522_52208

/-- Given a total of 12 coins consisting of quarters and nickels with a total value of 220 cents, 
    prove that the number of nickels is 4. -/
theorem coin_problem (q n : ℕ) : 
  q + n = 12 → 
  25 * q + 5 * n = 220 → 
  n = 4 := by
  sorry

end NUMINAMATH_CALUDE_coin_problem_l522_52208


namespace NUMINAMATH_CALUDE_time_addition_theorem_l522_52295

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds a duration to a given time and returns the result on a 12-hour clock -/
def addTime (initial : Time) (dHours dMinutes dSeconds : Nat) : Time :=
  sorry

/-- The sum of the components of a Time -/
def timeSum (t : Time) : Nat :=
  t.hours + t.minutes + t.seconds

theorem time_addition_theorem :
  let initialTime : Time := ⟨3, 0, 0⟩
  let finalTime := addTime initialTime 315 78 30
  timeSum finalTime = 55 := by sorry

end NUMINAMATH_CALUDE_time_addition_theorem_l522_52295


namespace NUMINAMATH_CALUDE_ellipse_equation_from_conditions_l522_52275

/-- An ellipse with foci on the y-axis -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < b ∧ b < a
  h_c : c^2 = a^2 - b^2

/-- The equation of the ellipse -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The condition that a point on the short axis and the two foci form an equilateral triangle -/
def equilateral_triangle_condition (e : Ellipse) : Prop :=
  e.a = 2 * e.c

/-- The condition that the shortest distance from the foci to the endpoints of the major axis is √3 -/
def shortest_distance_condition (e : Ellipse) : Prop :=
  e.a - e.c = Real.sqrt 3

theorem ellipse_equation_from_conditions (e : Ellipse)
  (h_triangle : equilateral_triangle_condition e)
  (h_distance : shortest_distance_condition e) :
  ∀ x y : ℝ, ellipse_equation e x y ↔ x^2 / 12 + y^2 / 9 = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_from_conditions_l522_52275


namespace NUMINAMATH_CALUDE_roses_cut_equals_ten_l522_52281

def initial_roses : ℕ := 6
def final_roses : ℕ := 16

theorem roses_cut_equals_ten : 
  final_roses - initial_roses = 10 := by sorry

end NUMINAMATH_CALUDE_roses_cut_equals_ten_l522_52281


namespace NUMINAMATH_CALUDE_positive_integer_solutions_l522_52276

theorem positive_integer_solutions :
  ∀ (a b c x y z : ℕ+),
    (a + b + c = x * y * z ∧ x + y + z = a * b * c) ↔
    ((x = 3 ∧ y = 2 ∧ z = 1 ∧ a = 3 ∧ b = 2 ∧ c = 1) ∨
     (x = 3 ∧ y = 3 ∧ z = 1 ∧ a = 5 ∧ b = 2 ∧ c = 1) ∨
     (x = 5 ∧ y = 2 ∧ z = 1 ∧ a = 3 ∧ b = 3 ∧ c = 1)) :=
by sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_l522_52276


namespace NUMINAMATH_CALUDE_expression_evaluation_l522_52245

theorem expression_evaluation :
  let x : ℚ := 1/2
  (2*x - 1)^2 - (3*x + 1)*(3*x - 1) + 5*x*(x - 1) = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l522_52245


namespace NUMINAMATH_CALUDE_rectangular_plot_length_l522_52237

theorem rectangular_plot_length
  (width : ℝ)
  (num_poles : ℕ)
  (pole_spacing : ℝ)
  (h1 : width = 50)
  (h2 : num_poles = 14)
  (h3 : pole_spacing = 20)
  : ∃ (length : ℝ), length = 80 ∧ 2 * (length + width) = (num_poles - 1) * pole_spacing :=
by sorry

end NUMINAMATH_CALUDE_rectangular_plot_length_l522_52237


namespace NUMINAMATH_CALUDE_bianca_recycling_points_l522_52226

/-- Calculates the points earned by Bianca for recycling bottles and cans --/
def points_earned (aluminum_points plastic_points glass_points : ℕ)
                  (aluminum_bags plastic_bags glass_bags : ℕ)
                  (aluminum_not_recycled plastic_not_recycled glass_not_recycled : ℕ) : ℕ :=
  (aluminum_points * (aluminum_bags - aluminum_not_recycled)) +
  (plastic_points * (plastic_bags - plastic_not_recycled)) +
  (glass_points * (glass_bags - glass_not_recycled))

theorem bianca_recycling_points :
  points_earned 5 8 10 10 5 5 3 2 1 = 99 := by
  sorry

end NUMINAMATH_CALUDE_bianca_recycling_points_l522_52226


namespace NUMINAMATH_CALUDE_positive_integer_solutions_count_l522_52267

theorem positive_integer_solutions_count : ∃ (n : ℕ), n = 10 ∧ 
  n = (Finset.filter (λ (x : ℕ × ℕ × ℕ) => 
    x.1 + x.2.1 + x.2.2 = 6 ∧ x.1 > 0 ∧ x.2.1 > 0 ∧ x.2.2 > 0) 
    (Finset.product (Finset.range 7) (Finset.product (Finset.range 7) (Finset.range 7)))).card :=
by
  sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_count_l522_52267


namespace NUMINAMATH_CALUDE_photo_arrangements_3_3_1_l522_52223

/-- The number of possible arrangements for a group photo -/
def photo_arrangements (num_boys num_girls : ℕ) : ℕ :=
  let adjacent_choices := num_boys * num_girls
  let remaining_boys := num_boys - 1
  let remaining_girls := num_girls - 1
  let remaining_arrangements := (remaining_boys * (remaining_boys - 1)) * 
                                (remaining_girls * (remaining_girls - 1) * 
                                 (remaining_boys + remaining_girls) * 
                                 (remaining_boys + remaining_girls - 1))
  2 * adjacent_choices * remaining_arrangements

/-- Theorem stating the number of arrangements for 3 boys, 3 girls, and 1 teacher -/
theorem photo_arrangements_3_3_1 :
  photo_arrangements 3 3 = 432 := by
  sorry

#eval photo_arrangements 3 3

end NUMINAMATH_CALUDE_photo_arrangements_3_3_1_l522_52223


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l522_52299

theorem arctan_equation_solution :
  ∃ x : ℝ, Real.arctan (2 / x) + Real.arctan (3 / x^3) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l522_52299


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l522_52270

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence satisfying the condition,
    prove that the sum of specific terms equals 2502.5. -/
theorem arithmetic_sequence_sum
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_condition : a 3 + a 4 + a 10 + a 11 = 2002) :
  a 1 + a 5 + a 7 + a 9 + a 13 = 2502.5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l522_52270


namespace NUMINAMATH_CALUDE_det_dilation_matrix_det_dilation_matrix_7_l522_52291

def dilation_matrix (scale_factor : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![scale_factor, 0],
    ![0, scale_factor]]

theorem det_dilation_matrix (scale_factor : ℝ) :
  Matrix.det (dilation_matrix scale_factor) = scale_factor ^ 2 := by
  sorry

theorem det_dilation_matrix_7 :
  Matrix.det (dilation_matrix 7) = 49 := by
  sorry

end NUMINAMATH_CALUDE_det_dilation_matrix_det_dilation_matrix_7_l522_52291


namespace NUMINAMATH_CALUDE_min_gold_chips_l522_52269

/-- Represents a box of chips with gold, silver, and bronze chips. -/
structure ChipBox where
  gold : ℕ
  silver : ℕ
  bronze : ℕ

/-- Checks if a ChipBox satisfies the given conditions. -/
def isValidChipBox (box : ChipBox) : Prop :=
  box.bronze ≥ 2 * box.silver ∧
  box.bronze ≤ box.gold / 4 ∧
  box.silver + box.bronze ≥ 75

/-- Theorem stating the minimum number of gold chips in a valid ChipBox. -/
theorem min_gold_chips (box : ChipBox) :
  isValidChipBox box → box.gold ≥ 200 := by
  sorry

#check min_gold_chips

end NUMINAMATH_CALUDE_min_gold_chips_l522_52269


namespace NUMINAMATH_CALUDE_integer_operation_proof_l522_52234

theorem integer_operation_proof (n : ℤ) : 5 * (n - 2) = 85 → n = 19 := by
  sorry

end NUMINAMATH_CALUDE_integer_operation_proof_l522_52234


namespace NUMINAMATH_CALUDE_probability_a_and_b_selected_l522_52203

-- Define the total number of students
def total_students : ℕ := 5

-- Define the number of students to be selected
def selected_students : ℕ := 3

-- Define the number of ways to select 3 students from 5
def total_ways : ℕ := Nat.choose total_students selected_students

-- Define the number of ways to select A, B, and one other student
def favorable_ways : ℕ := Nat.choose (total_students - 2) (selected_students - 2)

-- Theorem to prove
theorem probability_a_and_b_selected :
  (favorable_ways : ℚ) / total_ways = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_a_and_b_selected_l522_52203


namespace NUMINAMATH_CALUDE_perfect_cube_condition_l522_52221

/-- A polynomial x^3 + px^2 + qx + n is a perfect cube if and only if q = p^2 / 3 and n = p^3 / 27 -/
theorem perfect_cube_condition (p q n : ℝ) :
  (∃ a : ℝ, ∀ x : ℝ, x^3 + p*x^2 + q*x + n = (x + a)^3) ↔ 
  (q = p^2 / 3 ∧ n = p^3 / 27) :=
by sorry

end NUMINAMATH_CALUDE_perfect_cube_condition_l522_52221


namespace NUMINAMATH_CALUDE_carol_cupcakes_l522_52284

/-- Calculates the total number of cupcakes Carol has after selling some and making more. -/
def total_cupcakes (initial : ℕ) (sold : ℕ) (new_made : ℕ) : ℕ :=
  initial - sold + new_made

/-- Proves that Carol has 49 cupcakes in total given the initial conditions. -/
theorem carol_cupcakes : total_cupcakes 30 9 28 = 49 := by
  sorry

end NUMINAMATH_CALUDE_carol_cupcakes_l522_52284


namespace NUMINAMATH_CALUDE_smallest_slope_tangent_line_l522_52290

/-- The equation of the curve -/
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x - 10

/-- The derivative of the curve -/
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x + 6

theorem smallest_slope_tangent_line :
  ∃ (x₀ y₀ : ℝ), 
    (∀ x : ℝ, f' x₀ ≤ f' x) ∧ 
    y₀ = f x₀ ∧
    (3 : ℝ) * x - y - 11 = 0 :=
sorry

end NUMINAMATH_CALUDE_smallest_slope_tangent_line_l522_52290


namespace NUMINAMATH_CALUDE_max_value_theorem_l522_52249

theorem max_value_theorem (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 20) :
  Real.sqrt (x + 16) + Real.sqrt (20 - x) + 2 * Real.sqrt x ≤ (16 * Real.sqrt 3 + 2 * Real.sqrt 33) / 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l522_52249


namespace NUMINAMATH_CALUDE_cos_max_value_l522_52228

open Real

theorem cos_max_value (x : ℝ) :
  let f := fun x => 3 - 2 * cos (x + π / 4)
  (∀ x, f x ≤ 5) ∧
  (∃ k : ℤ, f (2 * k * π + 3 * π / 4) = 5) :=
sorry

end NUMINAMATH_CALUDE_cos_max_value_l522_52228


namespace NUMINAMATH_CALUDE_x₀_value_l522_52214

noncomputable section

variables (a b : ℝ) (x₀ : ℝ)

def f (x : ℝ) := a * x^2 + b

theorem x₀_value (ha : a ≠ 0) (hx₀ : x₀ > 0) 
  (h_integral : ∫ x in (0)..(2), f a b x = 2 * f a b x₀) : 
  x₀ = 2 * Real.sqrt 3 / 3 := by
  sorry

end

end NUMINAMATH_CALUDE_x₀_value_l522_52214


namespace NUMINAMATH_CALUDE_planted_area_fraction_l522_52244

theorem planted_area_fraction (a b c x : ℝ) (h1 : a = 5) (h2 : b = 12) (h3 : c^2 = a^2 + b^2) 
  (h4 : x^2 - 7*x + 9 = 0) (h5 : x > 0) (h6 : x < a) (h7 : x < b) :
  (a*b/2 - x^2) / (a*b/2) = 30/30 - ((7 - Real.sqrt 13)/2)^2 / 30 := by
  sorry

end NUMINAMATH_CALUDE_planted_area_fraction_l522_52244


namespace NUMINAMATH_CALUDE_team_selection_count_l522_52246

theorem team_selection_count (n : ℕ) (k : ℕ) (h1 : n = 17) (h2 : k = 4) :
  Nat.choose n k = 2380 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_count_l522_52246


namespace NUMINAMATH_CALUDE_area_MNKP_l522_52294

theorem area_MNKP (S_ABCD : ℝ) (h : S_ABCD = 84 * Real.sqrt 3) :
  ∃ S_MNKP : ℝ, S_MNKP = 42 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_area_MNKP_l522_52294


namespace NUMINAMATH_CALUDE_triangle_circumradius_l522_52252

/-- The circumradius of a triangle with sides 12, 10, and 7 is 6 units. -/
theorem triangle_circumradius (a b c : ℝ) (h_a : a = 12) (h_b : b = 10) (h_c : c = 7) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let R := (a * b * c) / (4 * area)
  R = 6 := by sorry

end NUMINAMATH_CALUDE_triangle_circumradius_l522_52252


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l522_52224

theorem quadratic_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 6*x + m = 0 ∧ y^2 - 6*y + m = 0) → m < 9 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l522_52224


namespace NUMINAMATH_CALUDE_cal_anthony_ratio_l522_52200

/-- Represents the number of transactions handled by each person --/
structure Transactions where
  mabel : ℕ
  anthony : ℕ
  cal : ℕ
  jade : ℕ

/-- The given conditions of the problem --/
def problem_conditions (t : Transactions) : Prop :=
  t.mabel = 90 ∧
  t.anthony = t.mabel + t.mabel / 10 ∧
  t.jade = 80 ∧
  t.jade = t.cal + 14

/-- The theorem to be proved --/
theorem cal_anthony_ratio (t : Transactions) 
  (h : problem_conditions t) : 
  (t.cal : ℚ) / t.anthony = 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_cal_anthony_ratio_l522_52200


namespace NUMINAMATH_CALUDE_second_part_sum_l522_52268

/-- Calculates the interest on a principal amount for a given rate and time. -/
def interest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time / 100

/-- Proves that given the conditions, the second part of the sum is 1672. -/
theorem second_part_sum (total : ℚ) (first_part : ℚ) (second_part : ℚ) 
  (h1 : total = 2717)
  (h2 : first_part + second_part = total)
  (h3 : interest first_part 3 8 = interest second_part 5 3) :
  second_part = 1672 := by
  sorry

end NUMINAMATH_CALUDE_second_part_sum_l522_52268


namespace NUMINAMATH_CALUDE_even_red_faces_count_l522_52232

/-- Represents a rectangular block with given dimensions -/
structure Block where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Counts the number of cubes with an even number of red faces in a painted block -/
def countEvenRedFaces (b : Block) : ℕ :=
  sorry

/-- The main theorem stating that a 6x4x2 block has 24 cubes with an even number of red faces -/
theorem even_red_faces_count (b : Block) (h1 : b.length = 6) (h2 : b.width = 4) (h3 : b.height = 2) :
  countEvenRedFaces b = 24 := by
  sorry

#check even_red_faces_count

end NUMINAMATH_CALUDE_even_red_faces_count_l522_52232


namespace NUMINAMATH_CALUDE_number_of_ferries_divisible_by_four_l522_52289

/-- Represents a ferry route between two points across a lake. -/
structure FerryRoute where
  /-- Time interval between ferry departures -/
  departureInterval : ℕ
  /-- Time taken to cross the lake -/
  crossingTime : ℕ
  /-- Number of ferries arriving during docking time -/
  arrivingFerries : ℕ

/-- Theorem stating that the number of ferries on a route with given conditions is divisible by 4 -/
theorem number_of_ferries_divisible_by_four (route : FerryRoute) 
  (h1 : route.crossingTime = route.arrivingFerries * route.departureInterval)
  (h2 : route.crossingTime > 0) : 
  ∃ (n : ℕ), (4 * route.crossingTime) / route.departureInterval = 4 * n := by
  sorry


end NUMINAMATH_CALUDE_number_of_ferries_divisible_by_four_l522_52289


namespace NUMINAMATH_CALUDE_smallest_value_complex_sum_l522_52279

theorem smallest_value_complex_sum (x y z : ℕ) (θ : ℂ) 
  (hxyz : x < y ∧ y < z)
  (hθ4 : θ^4 = 1)
  (hθ_neq_1 : θ ≠ 1) :
  ∃ (w : ℕ), w > 0 ∧ ∀ (a b c : ℕ) (ϕ : ℂ),
    a < b ∧ b < c → ϕ^4 = 1 → ϕ ≠ 1 →
    Complex.abs (↑x + ↑y * θ + ↑z * θ^3) ≤ Complex.abs (↑a + ↑b * ϕ + ↑c * ϕ^3) ∧
    Complex.abs (↑x + ↑y * θ + ↑z * θ^3) = Real.sqrt (↑w) :=
sorry

end NUMINAMATH_CALUDE_smallest_value_complex_sum_l522_52279


namespace NUMINAMATH_CALUDE_oliver_candy_boxes_l522_52274

def candy_problem (morning_boxes afternoon_multiplier given_away : ℕ) : ℕ :=
  morning_boxes + afternoon_multiplier * morning_boxes - given_away

theorem oliver_candy_boxes :
  candy_problem 8 3 10 = 22 := by
  sorry

end NUMINAMATH_CALUDE_oliver_candy_boxes_l522_52274


namespace NUMINAMATH_CALUDE_initial_population_l522_52219

theorem initial_population (P : ℝ) : 
  (P * (1 - 0.1)^2 = 8100) → P = 10000 := by
  sorry

end NUMINAMATH_CALUDE_initial_population_l522_52219


namespace NUMINAMATH_CALUDE_modulo_equivalence_solution_l522_52213

theorem modulo_equivalence_solution :
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 15 ∧ n ≡ 12345 [ZMOD 11] ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_modulo_equivalence_solution_l522_52213


namespace NUMINAMATH_CALUDE_fraction_inequality_l522_52201

theorem fraction_inequality (a b : ℝ) (ha : a ≠ 0) (ha1 : a + 1 ≠ 0) :
  ¬(∀ a b, b / a = (b + 1) / (a + 1)) :=
sorry

end NUMINAMATH_CALUDE_fraction_inequality_l522_52201


namespace NUMINAMATH_CALUDE_mean_equality_implies_y_equals_three_l522_52225

theorem mean_equality_implies_y_equals_three :
  let mean1 := (7 + 11 + 19) / 3
  let mean2 := (16 + 18 + y) / 3
  mean1 = mean2 →
  y = 3 := by
sorry

end NUMINAMATH_CALUDE_mean_equality_implies_y_equals_three_l522_52225
