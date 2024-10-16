import Mathlib

namespace NUMINAMATH_CALUDE_room_area_calculation_l2142_214249

/-- The area of a rectangular room with width 8 feet and length 1.5 feet is 12 square feet. -/
theorem room_area_calculation (width length area : ℝ) : 
  width = 8 → length = 1.5 → area = width * length → area = 12 := by
sorry

end NUMINAMATH_CALUDE_room_area_calculation_l2142_214249


namespace NUMINAMATH_CALUDE_suitcase_profit_l2142_214290

/-- Calculates the total profit and profit per suitcase for a store selling suitcases. -/
theorem suitcase_profit (num_suitcases : ℕ) (purchase_price : ℕ) (total_revenue : ℕ) :
  num_suitcases = 60 →
  purchase_price = 100 →
  total_revenue = 8100 →
  (total_revenue - num_suitcases * purchase_price = 2100) ∧
  ((total_revenue - num_suitcases * purchase_price) / num_suitcases = 35) := by
  sorry

#check suitcase_profit

end NUMINAMATH_CALUDE_suitcase_profit_l2142_214290


namespace NUMINAMATH_CALUDE_expand_product_l2142_214213

theorem expand_product (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2142_214213


namespace NUMINAMATH_CALUDE_only_D_greater_than_one_l2142_214234

theorem only_D_greater_than_one : 
  (0 / 0.16 ≤ 1) ∧ (1 * 0.16 ≤ 1) ∧ (1 / 1.6 ≤ 1) ∧ (1 * 1.6 > 1) := by
  sorry

end NUMINAMATH_CALUDE_only_D_greater_than_one_l2142_214234


namespace NUMINAMATH_CALUDE_polynomial_degree_l2142_214263

/-- The degree of the polynomial resulting from the expansion of 
    (3x^5 + 2x^3 - x + 7)(4x^11 - 6x^8 + 5x^5 - 15) - (x^2 + 3)^8 is 16 -/
theorem polynomial_degree : ℕ := by
  sorry

end NUMINAMATH_CALUDE_polynomial_degree_l2142_214263


namespace NUMINAMATH_CALUDE_john_annual_oil_change_cost_l2142_214242

/-- Calculates the annual cost of oil changes for a driver with given conditions. -/
def annual_oil_change_cost (miles_per_month : ℕ) (miles_per_oil_change : ℕ) (free_changes_per_year : ℕ) (cost_per_change : ℕ) : ℕ :=
  let changes_per_year := 12 * miles_per_month / miles_per_oil_change
  let paid_changes := changes_per_year - free_changes_per_year
  paid_changes * cost_per_change

/-- Theorem stating that John pays $150 a year for oil changes given the specified conditions. -/
theorem john_annual_oil_change_cost : 
  annual_oil_change_cost 1000 3000 1 50 = 150 := by
  sorry

end NUMINAMATH_CALUDE_john_annual_oil_change_cost_l2142_214242


namespace NUMINAMATH_CALUDE_harmonic_numbers_theorem_l2142_214237

/-- Definition of harmonic numbers -/
def are_harmonic (a b c : ℝ) : Prop :=
  1/b - 1/a = 1/c - 1/b

/-- Theorem: For harmonic numbers x, 5, 3 where x > 5, x = 15 -/
theorem harmonic_numbers_theorem (x : ℝ) 
  (h1 : are_harmonic x 5 3)
  (h2 : x > 5) : 
  x = 15 := by
  sorry

end NUMINAMATH_CALUDE_harmonic_numbers_theorem_l2142_214237


namespace NUMINAMATH_CALUDE_number_problem_l2142_214252

theorem number_problem (a b : ℤ) : 
  a + b = 72 → 
  a = b + 12 → 
  a = 42 → 
  b = 30 := by
sorry

end NUMINAMATH_CALUDE_number_problem_l2142_214252


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_divisibility_l2142_214292

theorem arithmetic_sequence_sum_divisibility :
  ∀ (a d : ℕ+), ∃ (k : ℕ+), (12 * a + 66 * d : ℕ) = 6 * k ∧
  ∀ (m : ℕ+), m < 6 → ∃ (a' d' : ℕ+), ¬(∃ (k' : ℕ+), (12 * a' + 66 * d' : ℕ) = m * k') :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_divisibility_l2142_214292


namespace NUMINAMATH_CALUDE_missing_number_implies_next_prime_l2142_214299

theorem missing_number_implies_next_prime (n : ℕ) : n > 3 →
  (∀ r s : ℕ, r ≥ 3 ∧ s ≥ 3 → n ≠ r * s - (r + s)) →
  Nat.Prime (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_missing_number_implies_next_prime_l2142_214299


namespace NUMINAMATH_CALUDE_no_simultaneous_age_ratio_l2142_214220

theorem no_simultaneous_age_ratio : ¬∃ (x : ℝ), x ≥ 0 ∧ 
  (85 + x = 3.5 * (15 + x)) ∧ (55 + x = 2 * (15 + x)) := by
  sorry

end NUMINAMATH_CALUDE_no_simultaneous_age_ratio_l2142_214220


namespace NUMINAMATH_CALUDE_expression_simplification_l2142_214224

theorem expression_simplification (a b : ℝ) 
  (h1 : a + b = 4) 
  (h2 : a - b = 2) : 
  (a^2 - 6*a*b + 9*b^2) / (a^2 - 2*a*b) / 
  ((5*b^2) / (a - 2*b) - a - 2*b) - 1/a = -1/3 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2142_214224


namespace NUMINAMATH_CALUDE_smallest_number_problem_l2142_214256

theorem smallest_number_problem (a b c : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  (a + b + c) / 3 = 30 →
  b = 31 →
  a ≤ b ∧ b ≤ c →
  c = b + 6 →
  a = 22 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_problem_l2142_214256


namespace NUMINAMATH_CALUDE_fraction_sum_and_product_l2142_214233

theorem fraction_sum_and_product (x y : ℚ) :
  x + y = 13/14 ∧ x * y = 3/28 →
  (x = 3/7 ∧ y = 1/4) ∨ (x = 1/4 ∧ y = 3/7) := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_and_product_l2142_214233


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l2142_214264

theorem simplest_quadratic_radical :
  let a := Real.sqrt 8
  let b := Real.sqrt 7
  let c := Real.sqrt 12
  let d := Real.sqrt (1/3)
  (∃ (x y : ℝ), a = x * Real.sqrt y ∧ x ≠ 1) ∧
  (∃ (x y : ℝ), c = x * Real.sqrt y ∧ x ≠ 1) ∧
  (∃ (x y : ℝ), d = x * Real.sqrt y ∧ x ≠ 1) ∧
  (∀ (x y : ℝ), b = x * Real.sqrt y → x = 1) :=
by sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l2142_214264


namespace NUMINAMATH_CALUDE_unique_prime_triple_l2142_214226

theorem unique_prime_triple : ∃! (p q r : ℕ), 
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
  Nat.Prime (4 * q - 1) ∧
  (p + q : ℚ) / (p + r) = r - p ∧
  p = 2 ∧ q = 3 ∧ r = 3 := by
sorry

end NUMINAMATH_CALUDE_unique_prime_triple_l2142_214226


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2142_214212

theorem arithmetic_sequence_problem (a b c d e : ℕ) :
  a < 10 ∧
  b = 12 ∧
  e = 33 ∧
  a < b ∧ b < c ∧ c < d ∧ d < e ∧
  e < 100 ∧
  b - a = c - b ∧
  c - b = d - c ∧
  d - c = e - d →
  a = 5 ∧ b = 12 ∧ c = 19 ∧ d = 26 ∧ e = 33 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2142_214212


namespace NUMINAMATH_CALUDE_largest_n_divisible_by_seven_l2142_214286

def expression (n : ℕ) : ℤ :=
  9 * (n - 3)^7 - 2 * n^3 + 15 * n - 33

theorem largest_n_divisible_by_seven :
  ∃ (n : ℕ), n = 149998 ∧
  n < 150000 ∧
  expression n % 7 = 0 ∧
  ∀ (m : ℕ), m < 150000 → m > n → expression m % 7 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_largest_n_divisible_by_seven_l2142_214286


namespace NUMINAMATH_CALUDE_cards_per_page_l2142_214248

theorem cards_per_page (new_cards old_cards pages : ℕ) 
  (h1 : new_cards = 3) 
  (h2 : old_cards = 13) 
  (h3 : pages = 2) : 
  (new_cards + old_cards) / pages = 8 :=
by sorry

end NUMINAMATH_CALUDE_cards_per_page_l2142_214248


namespace NUMINAMATH_CALUDE_beach_probability_l2142_214209

def beach_scenario (total_sunglasses : ℕ) (total_caps : ℕ) (prob_cap_given_sunglasses : ℚ) : Prop :=
  ∃ (both : ℕ),
    total_sunglasses = 60 ∧
    total_caps = 40 ∧
    prob_cap_given_sunglasses = 1/3 ∧
    both ≤ total_sunglasses ∧
    both ≤ total_caps ∧
    (both : ℚ) / total_sunglasses = prob_cap_given_sunglasses

theorem beach_probability (total_sunglasses total_caps : ℕ) (prob_cap_given_sunglasses : ℚ) :
  beach_scenario total_sunglasses total_caps prob_cap_given_sunglasses →
  (∃ (both : ℕ), (both : ℚ) / total_caps = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_beach_probability_l2142_214209


namespace NUMINAMATH_CALUDE_number_2018_in_equation_31_l2142_214241

def first_term (n : ℕ) : ℕ := 2 * n^2

theorem number_2018_in_equation_31 :
  ∃ k : ℕ, k ≥ first_term 31 ∧ k ≤ first_term 32 ∧ k = 2018 :=
by sorry

end NUMINAMATH_CALUDE_number_2018_in_equation_31_l2142_214241


namespace NUMINAMATH_CALUDE_inverse_inequality_for_negatives_l2142_214277

theorem inverse_inequality_for_negatives (a b : ℝ) : 0 > a → a > b → (1 / a) < (1 / b) := by
  sorry

end NUMINAMATH_CALUDE_inverse_inequality_for_negatives_l2142_214277


namespace NUMINAMATH_CALUDE_g_behavior_at_infinity_l2142_214204

-- Define the function g(x)
def g (x : ℝ) : ℝ := -3 * x^3 + 5 * x + 1

-- State the theorem
theorem g_behavior_at_infinity :
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → g x < M) ∧
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x < -N → g x > M) := by
  sorry

end NUMINAMATH_CALUDE_g_behavior_at_infinity_l2142_214204


namespace NUMINAMATH_CALUDE_complementary_probability_l2142_214272

theorem complementary_probability (P_snow : ℚ) (h : P_snow = 2/5) :
  1 - P_snow = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_complementary_probability_l2142_214272


namespace NUMINAMATH_CALUDE_max_n_value_l2142_214273

theorem max_n_value (a b c : ℝ) (n : ℕ) (h1 : a > b) (h2 : b > c)
  (h3 : ∀ (a b c : ℝ), a > b → b > c → (a - b)⁻¹ + (b - c)⁻¹ ≥ n^2 * (a - c)⁻¹) :
  n ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_n_value_l2142_214273


namespace NUMINAMATH_CALUDE_find_divisor_l2142_214231

theorem find_divisor (dividend quotient remainder divisor : ℕ) : 
  dividend = 13787 →
  quotient = 89 →
  remainder = 14 →
  dividend = divisor * quotient + remainder →
  divisor = 155 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l2142_214231


namespace NUMINAMATH_CALUDE_solve_complex_equation_l2142_214229

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the equation
def equation (w : ℂ) : Prop :=
  2 + 3 * i * w = 4 - 2 * i * w

-- State the theorem
theorem solve_complex_equation :
  ∃ w : ℂ, equation w ∧ w = -2 * i / 5 :=
by sorry

end NUMINAMATH_CALUDE_solve_complex_equation_l2142_214229


namespace NUMINAMATH_CALUDE_exchange_result_l2142_214214

/-- The number of bills after exchanging 2 $100 bills as described -/
def total_bills : ℕ :=
  let initial_hundred_bills : ℕ := 2
  let fifty_bills : ℕ := 2  -- From exchanging one $100 bill
  let ten_bills : ℕ := 50 / 10  -- From exchanging half of the remaining $100 bill
  let five_bills : ℕ := 50 / 5  -- From exchanging the other half of the remaining $100 bill
  fifty_bills + ten_bills + five_bills

/-- Theorem stating that the total number of bills after the exchange is 17 -/
theorem exchange_result : total_bills = 17 := by
  sorry

end NUMINAMATH_CALUDE_exchange_result_l2142_214214


namespace NUMINAMATH_CALUDE_circle_symmetry_l2142_214287

-- Define the original circle
def original_circle (x y : ℝ) : Prop := (x - 2)^2 + (y + 1)^2 = 4

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := y = x + 1

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop := (x + 2)^2 + (y - 3)^2 = 4

-- Theorem statement
theorem circle_symmetry :
  ∀ (x y : ℝ), 
  (∃ (x₀ y₀ : ℝ), original_circle x₀ y₀ ∧ 
   symmetry_line ((x + x₀) / 2) ((y + y₀) / 2) ∧
   (y - y₀) = -(x - x₀)) →
  symmetric_circle x y :=
by sorry

end NUMINAMATH_CALUDE_circle_symmetry_l2142_214287


namespace NUMINAMATH_CALUDE_simple_interest_problem_l2142_214228

/-- Given a sum P put at simple interest for 3 years, if increasing the interest rate
    by 1% results in an additional Rs. 75 interest, then P = 2500. -/
theorem simple_interest_problem (P : ℝ) (R : ℝ) : 
  P * (R + 1) * 3 / 100 - P * R * 3 / 100 = 75 → P = 2500 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l2142_214228


namespace NUMINAMATH_CALUDE_squirrel_mushroom_collection_l2142_214260

/-- Represents the number of mushrooms in each clearing --/
def MushroomSequence : Type := List Nat

/-- The total number of mushrooms collected by the squirrel --/
def TotalMushrooms : Nat := 60

/-- The number of clearings visited by the squirrel --/
def NumberOfClearings : Nat := 10

/-- Checks if a given sequence is valid according to the problem conditions --/
def IsValidSequence (seq : MushroomSequence) : Prop :=
  seq.length = NumberOfClearings ∧
  seq.sum = TotalMushrooms ∧
  seq.all (· > 0)

/-- The correct sequence of mushrooms collected in each clearing --/
def CorrectSequence : MushroomSequence := [5, 2, 11, 8, 2, 12, 3, 7, 2, 8]

/-- Theorem stating that the CorrectSequence is a valid solution to the problem --/
theorem squirrel_mushroom_collection :
  IsValidSequence CorrectSequence :=
sorry

end NUMINAMATH_CALUDE_squirrel_mushroom_collection_l2142_214260


namespace NUMINAMATH_CALUDE_parallel_line_plane_false_l2142_214222

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relation between lines and between a line and a plane
variable (parallel_line : Line → Line → Prop)
variable (parallel_plane : Line → Plane → Prop)

-- Define the "contained in" relation between a line and a plane
variable (contained_in : Line → Plane → Prop)

-- State the theorem to be proven false
theorem parallel_line_plane_false :
  ¬(∀ (l : Line) (p : Plane), parallel_plane l p →
    ∀ (m : Line), contained_in m p → parallel_line l m) :=
sorry

end NUMINAMATH_CALUDE_parallel_line_plane_false_l2142_214222


namespace NUMINAMATH_CALUDE_product_of_22nd_and_23rd_multiples_l2142_214206

/-- The sequence of multiples of 3 greater than 0 and less than 100 -/
def multiples_of_3 : List Nat :=
  (List.range 33).map (fun n => (n + 1) * 3)

/-- The 22nd element in the sequence -/
def element_22 : Nat := multiples_of_3[21]

/-- The 23rd element in the sequence -/
def element_23 : Nat := multiples_of_3[22]

theorem product_of_22nd_and_23rd_multiples :
  element_22 * element_23 = 4554 :=
by sorry

end NUMINAMATH_CALUDE_product_of_22nd_and_23rd_multiples_l2142_214206


namespace NUMINAMATH_CALUDE_return_flight_time_l2142_214239

/-- Represents the flight scenario between two cities -/
structure FlightScenario where
  d : ℝ  -- distance between cities
  p : ℝ  -- plane's speed in still air
  w : ℝ  -- wind speed
  against_wind_time : ℝ -- time for flight against wind
  still_air_time : ℝ  -- time for flight in still air

/-- The conditions of the flight scenario -/
def flight_conditions (scenario : FlightScenario) : Prop :=
  scenario.against_wind_time = 120 ∧
  scenario.d = scenario.against_wind_time * (scenario.p - scenario.w) ∧
  scenario.d / (scenario.p + scenario.w) = scenario.still_air_time - 10

/-- The theorem stating that under the given conditions, the return flight time is 110 minutes -/
theorem return_flight_time (scenario : FlightScenario) 
  (h : flight_conditions scenario) : 
  scenario.d / (scenario.p + scenario.w) = 110 := by
  sorry


end NUMINAMATH_CALUDE_return_flight_time_l2142_214239


namespace NUMINAMATH_CALUDE_surrounding_decagon_theorem_l2142_214215

/-- The number of sides of the surrounding polygons when a regular m-sided polygon
    is surrounded by m regular n-sided polygons without gaps or overlaps. -/
def surrounding_polygon_sides (m : ℕ) : ℕ :=
  if m = 4 then 8 else
  if m = 10 then
    let interior_angle_m := (180 * (m - 2)) / m
    let n := (720 / (360 - interior_angle_m) : ℕ)
    n
  else 0

/-- Theorem stating that when a regular 10-sided polygon is surrounded by 10 regular n-sided polygons
    without gaps or overlaps, n must equal 5. -/
theorem surrounding_decagon_theorem :
  surrounding_polygon_sides 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_surrounding_decagon_theorem_l2142_214215


namespace NUMINAMATH_CALUDE_unique_divisor_square_sum_l2142_214293

theorem unique_divisor_square_sum (p n : ℕ) (hp : p.Prime) (hp2 : p > 2) (hn : n > 0) :
  ∃! d : ℕ, d > 0 ∧ d ∣ (p * n^2) ∧ ∃ k : ℕ, n^2 + d = k^2 :=
by sorry

end NUMINAMATH_CALUDE_unique_divisor_square_sum_l2142_214293


namespace NUMINAMATH_CALUDE_consecutive_numbers_sum_l2142_214216

theorem consecutive_numbers_sum (n : ℕ) : 
  (n + (n + 1) + (n + 2) = 60) → 
  ((n + 2) + (n + 3) + (n + 4) = 66) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_sum_l2142_214216


namespace NUMINAMATH_CALUDE_fraction_decomposition_l2142_214217

theorem fraction_decomposition (x : ℝ) (h1 : x ≠ 7) (h2 : x ≠ -6) :
  (3 * x + 5) / (x^2 - x - 42) = 2 / (x - 7) + 1 / (x + 6) := by
  sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l2142_214217


namespace NUMINAMATH_CALUDE_triangle_inequality_satisfied_l2142_214279

theorem triangle_inequality_satisfied (a b c : ℝ) (ha : a = 25) (hb : b = 24) (hc : c = 7) :
  a + b > c ∧ a + c > b ∧ b + c > a :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_satisfied_l2142_214279


namespace NUMINAMATH_CALUDE_sin_shift_l2142_214271

theorem sin_shift (x : ℝ) : 
  Real.sin (2 * x - π / 3) = Real.sin (2 * (x - π / 6)) := by
  sorry

end NUMINAMATH_CALUDE_sin_shift_l2142_214271


namespace NUMINAMATH_CALUDE_factor_3x_squared_minus_75_l2142_214291

theorem factor_3x_squared_minus_75 (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) := by
  sorry

end NUMINAMATH_CALUDE_factor_3x_squared_minus_75_l2142_214291


namespace NUMINAMATH_CALUDE_one_fourths_in_three_eighths_l2142_214218

theorem one_fourths_in_three_eighths (x : ℚ) : x = 3/8 → (x / (1/4)) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_one_fourths_in_three_eighths_l2142_214218


namespace NUMINAMATH_CALUDE_album_time_calculation_l2142_214240

/-- Calculates the total time to finish all songs in an album -/
def total_album_time (initial_songs : ℕ) (song_duration : ℕ) (added_songs : ℕ) : ℕ :=
  (initial_songs + added_songs) * song_duration

/-- Theorem: Given an initial album of 25 songs, each 3 minutes long, and adding 10 more songs
    of the same duration, the total time to finish all songs in the album is 105 minutes. -/
theorem album_time_calculation :
  total_album_time 25 3 10 = 105 := by
  sorry

end NUMINAMATH_CALUDE_album_time_calculation_l2142_214240


namespace NUMINAMATH_CALUDE_games_missed_l2142_214253

/-- Given a total number of soccer games and the number of games Jessica attended,
    calculate the number of games Jessica missed. -/
theorem games_missed (total_games attended_games : ℕ) : 
  total_games = 6 → attended_games = 2 → total_games - attended_games = 4 := by
  sorry

end NUMINAMATH_CALUDE_games_missed_l2142_214253


namespace NUMINAMATH_CALUDE_find_C_value_l2142_214235

theorem find_C_value (D : ℝ) (h1 : 4 * C - 2 * D - 3 = 26) (h2 : D = 3) : C = 8.75 := by
  sorry

end NUMINAMATH_CALUDE_find_C_value_l2142_214235


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2142_214225

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of specific terms in an arithmetic sequence -/
theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ) 
  (h : arithmetic_sequence a) 
  (h_sum : a 3 + a 7 = 37) : 
  a 2 + a 4 + a 6 + a 8 = 74 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2142_214225


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_abs_coeff_l2142_214232

theorem binomial_expansion_sum_abs_coeff : 
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ),
  (∀ x : ℝ, (1 - x)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  |a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| = 64 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_abs_coeff_l2142_214232


namespace NUMINAMATH_CALUDE_ellipse_m_value_l2142_214246

/-- Represents an ellipse with equation x²/(m-2) + y²/(10-m) = 1 -/
structure Ellipse (m : ℝ) where
  eq : ∀ (x y : ℝ), x^2 / (m - 2) + y^2 / (10 - m) = 1

/-- Represents the focal distance of an ellipse -/
def focalDistance (e : Ellipse m) := 4

/-- Represents that the foci of the ellipse are on the x-axis -/
def fociOnXAxis (e : Ellipse m) := True

theorem ellipse_m_value (m : ℝ) (e : Ellipse m) 
  (h1 : focalDistance e = 4) (h2 : fociOnXAxis e) : m = 8 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_m_value_l2142_214246


namespace NUMINAMATH_CALUDE_invalid_votes_percentage_l2142_214251

theorem invalid_votes_percentage 
  (total_votes : ℕ) 
  (candidate_a_percentage : ℚ) 
  (candidate_a_votes : ℕ) 
  (h1 : total_votes = 560000)
  (h2 : candidate_a_percentage = 75 / 100)
  (h3 : candidate_a_votes = 357000) :
  (total_votes - (candidate_a_votes / candidate_a_percentage)) / total_votes = 15 / 100 := by
sorry

end NUMINAMATH_CALUDE_invalid_votes_percentage_l2142_214251


namespace NUMINAMATH_CALUDE_fibonacci_like_sequence_a8_l2142_214297

def fibonacci_like_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (n + 2) = a (n + 1) + a n

theorem fibonacci_like_sequence_a8 (a : ℕ → ℕ) :
  fibonacci_like_sequence a →
  (∀ n : ℕ, n ≥ 1 → a (n + 1) > a n) →
  (∀ n : ℕ, a n > 0) →
  a 7 = 240 →
  a 8 = 386 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_like_sequence_a8_l2142_214297


namespace NUMINAMATH_CALUDE_expand_expression_l2142_214203

theorem expand_expression (x : ℝ) : 3 * (x - 7) * (x + 10) + 5 * x = 3 * x^2 + 14 * x - 210 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2142_214203


namespace NUMINAMATH_CALUDE_inequality_proof_l2142_214247

theorem inequality_proof (x a : ℝ) (f : ℝ → ℝ) 
  (h1 : f = λ x => x^2 - x + 1) 
  (h2 : |x - a| < 1) : 
  |f x - f a| < 2 * (|a| + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2142_214247


namespace NUMINAMATH_CALUDE_linear_functions_intersection_l2142_214254

theorem linear_functions_intersection (a b c d : ℝ) (h : a ≠ b) :
  (∃ x y : ℝ, y = a * x + a ∧ y = b * x + b ∧ y = c * x + d) → c = d := by
  sorry

end NUMINAMATH_CALUDE_linear_functions_intersection_l2142_214254


namespace NUMINAMATH_CALUDE_min_value_of_function_l2142_214250

theorem min_value_of_function (x : ℝ) (h : x > -1) :
  (x^2 + 7*x + 10) / (x + 1) ≥ 9 ∧
  ∃ y : ℝ, y > -1 ∧ (y^2 + 7*y + 10) / (y + 1) = 9 :=
by
  sorry

#check min_value_of_function

end NUMINAMATH_CALUDE_min_value_of_function_l2142_214250


namespace NUMINAMATH_CALUDE_candy_distribution_l2142_214288

theorem candy_distribution (total_candies : ℕ) (total_children : ℕ) (lollipops_per_boy : ℕ) :
  total_candies = 90 →
  total_children = 40 →
  lollipops_per_boy = 3 →
  ∃ (num_boys num_girls : ℕ) (candy_canes_per_girl : ℕ),
    num_boys + num_girls = total_children ∧
    num_boys * lollipops_per_boy = total_candies / 3 ∧
    num_girls * candy_canes_per_girl = total_candies * 2 / 3 ∧
    candy_canes_per_girl = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l2142_214288


namespace NUMINAMATH_CALUDE_dans_balloons_l2142_214245

theorem dans_balloons (sam_initial : Real) (fred_given : Real) (total : Real) : Real :=
  let sam_remaining := sam_initial - fred_given
  let dan_balloons := total - sam_remaining
  dan_balloons

#check dans_balloons 46.0 10.0 52.0

end NUMINAMATH_CALUDE_dans_balloons_l2142_214245


namespace NUMINAMATH_CALUDE_initial_gum_pieces_l2142_214282

theorem initial_gum_pieces (x : ℕ) : x + 16 + 20 = 61 → x = 25 := by
  sorry

end NUMINAMATH_CALUDE_initial_gum_pieces_l2142_214282


namespace NUMINAMATH_CALUDE_asterisk_replacement_l2142_214283

theorem asterisk_replacement : ∃ x : ℝ, (x / 20) * (x / 180) = 1 ∧ x = 60 := by
  sorry

end NUMINAMATH_CALUDE_asterisk_replacement_l2142_214283


namespace NUMINAMATH_CALUDE_quadratic_solution_l2142_214275

theorem quadratic_solution (x : ℝ) : x^2 - 4*x + 3 = 0 ∧ x ≥ 0 → x = 1 ∨ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l2142_214275


namespace NUMINAMATH_CALUDE_f_3_equals_18_l2142_214210

def f : ℕ → ℕ
  | 0     => 3
  | (n+1) => (n+1) * f n

theorem f_3_equals_18 : f 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_f_3_equals_18_l2142_214210


namespace NUMINAMATH_CALUDE_inequality_solution_l2142_214270

/-- Given an inequality ax^2 - 3x + 2 < 0 with solution set {x | 1 < x < b}, prove a + b = 3 -/
theorem inequality_solution (a b : ℝ) 
  (h : ∀ x, ax^2 - 3*x + 2 < 0 ↔ 1 < x ∧ x < b) : 
  a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2142_214270


namespace NUMINAMATH_CALUDE_dans_remaining_money_l2142_214255

/-- Proves that if Dan has $3 and spends $1 on a candy bar, the amount of money left is $2. -/
theorem dans_remaining_money (initial_amount : ℕ) (candy_bar_cost : ℕ) (remaining_amount : ℕ) :
  initial_amount = 3 →
  candy_bar_cost = 1 →
  remaining_amount = initial_amount - candy_bar_cost →
  remaining_amount = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_dans_remaining_money_l2142_214255


namespace NUMINAMATH_CALUDE_pasta_sauce_cost_l2142_214281

/-- The cost of pasta sauce given grocery shopping conditions -/
theorem pasta_sauce_cost 
  (mustard_oil_quantity : ℝ) 
  (mustard_oil_price : ℝ) 
  (pasta_quantity : ℝ) 
  (pasta_price : ℝ) 
  (pasta_sauce_quantity : ℝ) 
  (initial_money : ℝ) 
  (money_left : ℝ) 
  (h1 : mustard_oil_quantity = 2) 
  (h2 : mustard_oil_price = 13) 
  (h3 : pasta_quantity = 3) 
  (h4 : pasta_price = 4) 
  (h5 : pasta_sauce_quantity = 1) 
  (h6 : initial_money = 50) 
  (h7 : money_left = 7) : 
  (initial_money - money_left - (mustard_oil_quantity * mustard_oil_price + pasta_quantity * pasta_price)) / pasta_sauce_quantity = 5 := by
sorry

end NUMINAMATH_CALUDE_pasta_sauce_cost_l2142_214281


namespace NUMINAMATH_CALUDE_march_first_is_monday_l2142_214284

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the day of the week for a given date in March
def marchDayOfWeek (date : Nat) : DayOfWeek := sorry

-- State the theorem
theorem march_first_is_monday : 
  marchDayOfWeek 8 = DayOfWeek.Monday → marchDayOfWeek 1 = DayOfWeek.Monday := by
  sorry

end NUMINAMATH_CALUDE_march_first_is_monday_l2142_214284


namespace NUMINAMATH_CALUDE_expression_equality_l2142_214257

theorem expression_equality : 
  (Real.sqrt 12) / 2 + |Real.sqrt 3 - 2| - Real.tan (π / 3) = 2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2142_214257


namespace NUMINAMATH_CALUDE_solve_equation_l2142_214266

theorem solve_equation (x : ℝ) (h : x - 3*x + 4*x = 140) : x = 70 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2142_214266


namespace NUMINAMATH_CALUDE_x_value_proof_l2142_214280

def star_operation (a b : ℝ) : ℝ := a * b + a + b

theorem x_value_proof :
  ∀ x : ℝ, star_operation 3 x = 27 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l2142_214280


namespace NUMINAMATH_CALUDE_rectangle_area_l2142_214294

theorem rectangle_area (L B : ℝ) (h1 : L - B = 23) (h2 : 2 * (L + B) = 266) : L * B = 4290 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2142_214294


namespace NUMINAMATH_CALUDE_quadratic_inequality_minimum_l2142_214230

theorem quadratic_inequality_minimum (a b : ℝ) (h1 : Set.Icc 1 4 = {x : ℝ | x^2 - 5*a*x + b ≤ 0}) :
  let t (x y : ℝ) := a/x + b/y
  ∀ x y : ℝ, x > 0 → y > 0 → x + y = 2 → t x y ≥ 9/2 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_minimum_l2142_214230


namespace NUMINAMATH_CALUDE_inequality_of_means_l2142_214258

theorem inequality_of_means (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  (a + b + c) / 3 > (a * b * c) ^ (1/3) ∧ (a * b * c) ^ (1/3) > 3 * a * b * c / (a * b + b * c + c * a) :=
sorry

end NUMINAMATH_CALUDE_inequality_of_means_l2142_214258


namespace NUMINAMATH_CALUDE_solve_equation_l2142_214261

theorem solve_equation (y : ℝ) (h : (9 / y^2) = (y / 36)) : y = (324 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2142_214261


namespace NUMINAMATH_CALUDE_f_2019_value_l2142_214207

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  f 1 = 1/4 ∧ ∀ x y : ℝ, 4 * f x * f y = f (x + y) + f (x - y)

/-- The main theorem stating that f(2019) = -1/2 for any function satisfying the conditions -/
theorem f_2019_value (f : ℝ → ℝ) (hf : special_function f) : f 2019 = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_2019_value_l2142_214207


namespace NUMINAMATH_CALUDE_swimming_speed_in_still_water_l2142_214298

/-- Given a person swimming against a current, calculates their swimming speed in still water. -/
theorem swimming_speed_in_still_water 
  (current_speed : ℝ) 
  (distance_against_current : ℝ) 
  (time_against_current : ℝ) 
  (h1 : current_speed = 10)
  (h2 : distance_against_current = 8)
  (h3 : time_against_current = 4) :
  distance_against_current = (swimming_speed - current_speed) * time_against_current →
  swimming_speed = 12 :=
by
  sorry

#check swimming_speed_in_still_water

end NUMINAMATH_CALUDE_swimming_speed_in_still_water_l2142_214298


namespace NUMINAMATH_CALUDE_arithmetic_sequence_11th_term_l2142_214221

/-- An arithmetic sequence with a₁ = 1 and aₙ₊₂ - aₙ = 6 -/
def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 2) - a n = 6

/-- The 11th term of the arithmetic sequence is 31 -/
theorem arithmetic_sequence_11th_term
  (a : ℕ → ℝ) (h : arithmeticSequence a) : a 11 = 31 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_11th_term_l2142_214221


namespace NUMINAMATH_CALUDE_second_number_value_l2142_214296

theorem second_number_value (a b : ℝ) 
  (eq1 : a * (a - 6) = 7)
  (eq2 : b * (b - 6) = 7)
  (neq : a ≠ b)
  (sum : a + b = 6) :
  b = 7 := by
  sorry

end NUMINAMATH_CALUDE_second_number_value_l2142_214296


namespace NUMINAMATH_CALUDE_cosine_range_theorem_l2142_214238

theorem cosine_range_theorem (x : ℝ) :
  x ∈ Set.Icc 0 (2 * Real.pi) →
  x ∈ {x | Real.cos x ≤ 1/2} ↔ x ∈ Set.Icc (Real.pi/3) (5*Real.pi/3) := by
sorry

end NUMINAMATH_CALUDE_cosine_range_theorem_l2142_214238


namespace NUMINAMATH_CALUDE_triangle_theorem_l2142_214227

noncomputable def triangle_proof (a b c : ℝ) (A B C : ℝ) : Prop :=
  -- Triangle ABC is acute
  0 < A ∧ A < Real.pi/2 ∧
  0 < B ∧ B < Real.pi/2 ∧
  0 < C ∧ C < Real.pi/2 ∧
  -- Sum of angles is π
  A + B + C = Real.pi ∧
  -- Given conditions
  a = 2*b * Real.sin A ∧
  a = 3 * Real.sqrt 3 ∧
  c = 5 →
  -- Conclusions
  B = Real.pi/6 ∧  -- 30° in radians
  (1/2 * a * c * Real.sin B = 15 * Real.sqrt 3 / 4) ∧
  b = Real.sqrt 7

theorem triangle_theorem :
  ∀ a b c A B C, triangle_proof a b c A B C :=
sorry

end NUMINAMATH_CALUDE_triangle_theorem_l2142_214227


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l2142_214219

/-- Given vectors a and b in ℝ², prove that if a is perpendicular to (a - b), then the x-coordinate of b is 1/2. -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (h1 : a = (2, 3)) (h2 : b.2 = 4) :
  (a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) = 0) → b.1 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l2142_214219


namespace NUMINAMATH_CALUDE_sqrt_72_equals_6_sqrt_2_l2142_214244

theorem sqrt_72_equals_6_sqrt_2 : Real.sqrt 72 = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_72_equals_6_sqrt_2_l2142_214244


namespace NUMINAMATH_CALUDE_parallel_lines_reasoning_is_deductive_l2142_214259

-- Define the types of reasoning
inductive ReasoningType
  | Deductive
  | Analogical

-- Define the characteristics of different types of reasoning
def isDeductive (r : ReasoningType) : Prop :=
  r = ReasoningType.Deductive

def isGeneralToSpecific (r : ReasoningType) : Prop :=
  r = ReasoningType.Deductive

-- Define the geometric concept
def sameSideInteriorAngles (a b : ℝ) : Prop :=
  a + b = 180

-- Theorem statement
theorem parallel_lines_reasoning_is_deductive :
  ∀ (A B : ℝ) (r : ReasoningType),
    sameSideInteriorAngles A B →
    isGeneralToSpecific r →
    isDeductive r :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_reasoning_is_deductive_l2142_214259


namespace NUMINAMATH_CALUDE_complex_number_location_l2142_214262

theorem complex_number_location (z : ℂ) (h : (1 - Complex.I) * z = Complex.I ^ 2013) :
  (z.re < 0) ∧ (z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l2142_214262


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2142_214243

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- Definition of geometric sequence
  a 2 = 8 →                     -- Given condition
  a 5 = 64 →                    -- Given condition
  q = 2 :=                      -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2142_214243


namespace NUMINAMATH_CALUDE_beverly_bottle_caps_l2142_214289

/-- The number of bottle caps in Beverly's collection -/
def total_bottle_caps (small_box_caps : ℕ) (large_box_caps : ℕ) 
                      (small_boxes : ℕ) (large_boxes : ℕ) 
                      (individual_caps : ℕ) : ℕ :=
  small_box_caps * small_boxes + large_box_caps * large_boxes + individual_caps

/-- Theorem stating the total number of bottle caps in Beverly's collection -/
theorem beverly_bottle_caps : 
  total_bottle_caps 35 75 7 3 23 = 493 := by
  sorry

end NUMINAMATH_CALUDE_beverly_bottle_caps_l2142_214289


namespace NUMINAMATH_CALUDE_horner_v3_value_hex_210_to_decimal_l2142_214265

-- Define the polynomial and Horner's method
def f (x : ℤ) : ℤ := 3*x^6 + 5*x^5 + 6*x^4 + 79*x^3 - 8*x^2 + 35*x + 12

def horner_step (v : ℤ) (x : ℤ) (a : ℤ) : ℤ := v * x + a

def horner_v3 (x : ℤ) : ℤ :=
  let v0 := 3
  let v1 := horner_step v0 x 5
  let v2 := horner_step v1 x 6
  horner_step v2 x 79

-- Theorem for the first part of the problem
theorem horner_v3_value : horner_v3 (-4) = -57 := by sorry

-- Define hexadecimal to decimal conversion
def hex_to_decimal (d2 d1 d0 : ℕ) : ℕ := d2 * 6^2 + d1 * 6^1 + d0 * 6^0

-- Theorem for the second part of the problem
theorem hex_210_to_decimal : hex_to_decimal 2 1 0 = 78 := by sorry

end NUMINAMATH_CALUDE_horner_v3_value_hex_210_to_decimal_l2142_214265


namespace NUMINAMATH_CALUDE_buddy_fraction_l2142_214274

theorem buddy_fraction (s n : ℕ) (hs : s > 0) (hn : n > 0) : 
  (s : ℚ) / 3 = (n : ℚ) / 4 →
  ((s : ℚ) / 3 + (n : ℚ) / 4) / ((s : ℚ) + (n : ℚ)) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_buddy_fraction_l2142_214274


namespace NUMINAMATH_CALUDE_paintings_distribution_l2142_214211

theorem paintings_distribution (total_paintings : ℕ) (num_rooms : ℕ) (paintings_per_room : ℕ) :
  total_paintings = 32 →
  num_rooms = 4 →
  paintings_per_room = total_paintings / num_rooms →
  paintings_per_room = 8 := by
  sorry

end NUMINAMATH_CALUDE_paintings_distribution_l2142_214211


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_neg_one_and_five_l2142_214223

theorem arithmetic_mean_of_neg_one_and_five (x y : ℝ) : 
  x = -1 → y = 5 → (x + y) / 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_neg_one_and_five_l2142_214223


namespace NUMINAMATH_CALUDE_total_sum_lent_l2142_214201

/-- Proves that the total sum lent is 2665 given the specified conditions --/
theorem total_sum_lent (first_part second_part : ℕ) : 
  second_part = 1640 →
  (first_part * 8 * 3) = (second_part * 3 * 5) →
  first_part + second_part = 2665 := by
  sorry

#check total_sum_lent

end NUMINAMATH_CALUDE_total_sum_lent_l2142_214201


namespace NUMINAMATH_CALUDE_solve_equation_l2142_214267

theorem solve_equation (x : ℤ) (h : 9873 + x = 13200) : x = 3327 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2142_214267


namespace NUMINAMATH_CALUDE_power_equation_solution_l2142_214205

theorem power_equation_solution (y : ℕ) : 8^5 + 8^5 + 2 * 8^5 = 2^y → y = 17 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l2142_214205


namespace NUMINAMATH_CALUDE_garden_area_is_855_l2142_214236

/-- Represents a rectangular garden with fence posts -/
structure Garden where
  posts : ℕ
  post_distance : ℝ
  longer_side_post_ratio : ℕ

/-- Calculates the area of the garden given the specifications -/
def garden_area (g : Garden) : ℝ :=
  let shorter_side_posts := (g.posts / 2) / (g.longer_side_post_ratio + 1)
  let longer_side_posts := g.longer_side_post_ratio * shorter_side_posts
  let shorter_side_length := (shorter_side_posts - 1) * g.post_distance
  let longer_side_length := (longer_side_posts - 1) * g.post_distance
  shorter_side_length * longer_side_length

/-- Theorem stating that the garden with given specifications has an area of 855 square yards -/
theorem garden_area_is_855 (g : Garden) 
    (h1 : g.posts = 24)
    (h2 : g.post_distance = 6)
    (h3 : g.longer_side_post_ratio = 3) : 
  garden_area g = 855 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_is_855_l2142_214236


namespace NUMINAMATH_CALUDE_quadratic_rewrite_product_l2142_214268

theorem quadratic_rewrite_product (p q r : ℤ) : 
  (∀ x, 4 * x^2 - 20 * x - 32 = (p * x + q)^2 + r) → p * q = -10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_product_l2142_214268


namespace NUMINAMATH_CALUDE_tom_fruit_purchase_l2142_214200

/-- Given Tom's purchase of apples and mangoes, prove the amount of mangoes bought -/
theorem tom_fruit_purchase (apple_kg : ℕ) (apple_rate : ℕ) (mango_rate : ℕ) (total_paid : ℕ) 
  (h1 : apple_kg = 8)
  (h2 : apple_rate = 70)
  (h3 : mango_rate = 70)
  (h4 : total_paid = 1190) :
  (total_paid - apple_kg * apple_rate) / mango_rate = 9 := by
  sorry

end NUMINAMATH_CALUDE_tom_fruit_purchase_l2142_214200


namespace NUMINAMATH_CALUDE_ramon_twice_loui_age_l2142_214278

/-- The age of Loui today -/
def loui_age : ℕ := 23

/-- The age of Ramon today -/
def ramon_age : ℕ := 26

/-- The number of years until Ramon is twice as old as Loui is today -/
def years_until_double : ℕ := 20

/-- Theorem stating that in 'years_until_double' years, Ramon will be twice as old as Loui is today -/
theorem ramon_twice_loui_age : 
  ramon_age + years_until_double = 2 * loui_age := by
  sorry

end NUMINAMATH_CALUDE_ramon_twice_loui_age_l2142_214278


namespace NUMINAMATH_CALUDE_expression_equality_l2142_214285

theorem expression_equality : (3 + 2)^127 + 3 * (2^126 + 3^126) = 5^127 + 3 * 2^126 + 3 * 3^126 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2142_214285


namespace NUMINAMATH_CALUDE_angle_value_l2142_214208

theorem angle_value (a : ℝ) : 3 * a + 150 = 360 → a = 70 := by
  sorry

end NUMINAMATH_CALUDE_angle_value_l2142_214208


namespace NUMINAMATH_CALUDE_expenditure_ratio_l2142_214276

/-- Represents a person with income and expenditure -/
structure Person where
  income : ℕ
  expenditure : ℕ

/-- The problem setup -/
def problem_setup : Prop := ∃ (p1 p2 : Person),
  -- The ratio of incomes is 5:4
  p1.income * 4 = p2.income * 5 ∧
  -- Each person saves 2200
  p1.income - p1.expenditure = 2200 ∧
  p2.income - p2.expenditure = 2200 ∧
  -- P1's income is 5500
  p1.income = 5500

/-- The theorem to prove -/
theorem expenditure_ratio (h : problem_setup) :
  ∃ (p1 p2 : Person), p1.expenditure * 2 = p2.expenditure * 3 :=
sorry

end NUMINAMATH_CALUDE_expenditure_ratio_l2142_214276


namespace NUMINAMATH_CALUDE_right_triangle_area_l2142_214202

theorem right_triangle_area (a b : ℝ) (h : Real.sqrt (a - 5) + (b - 4)^2 = 0) :
  let area := (1/2) * a * b
  area = 6 ∨ area = 10 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2142_214202


namespace NUMINAMATH_CALUDE_quadratic_coefficients_l2142_214269

/-- A quadratic function with vertex (h, k) and y-intercept c can be represented as f(x) = a(x - h)² + k,
    where a ≠ 0 and f(0) = c. -/
def quadratic_function (a h k c : ℝ) (ha : a ≠ 0) (f : ℝ → ℝ) :=
  ∀ x, f x = a * (x - h)^2 + k ∧ f 0 = c

theorem quadratic_coefficients (f : ℝ → ℝ) (a h k c : ℝ) (ha : a ≠ 0) :
  quadratic_function a h k c ha f →
  h = 2 ∧ k = -1 ∧ c = 11 →
  a = 3 ∧ 
  ∃ b, ∀ x, f x = 3 * x^2 + b * x + 11 ∧ b = -12 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_coefficients_l2142_214269


namespace NUMINAMATH_CALUDE_harry_tomato_packets_l2142_214295

/-- Represents the number of packets of tomato seeds Harry bought -/
def tomato_packets : ℕ := sorry

/-- The price of a packet of pumpkin seeds in dollars -/
def pumpkin_price : ℚ := 5/2

/-- The price of a packet of tomato seeds in dollars -/
def tomato_price : ℚ := 3/2

/-- The price of a packet of chili pepper seeds in dollars -/
def chili_price : ℚ := 9/10

/-- The number of packets of pumpkin seeds Harry bought -/
def pumpkin_bought : ℕ := 3

/-- The number of packets of chili pepper seeds Harry bought -/
def chili_bought : ℕ := 5

/-- The total amount Harry spent in dollars -/
def total_spent : ℚ := 18

theorem harry_tomato_packets : 
  pumpkin_price * pumpkin_bought + tomato_price * tomato_packets + chili_price * chili_bought = total_spent ∧ 
  tomato_packets = 4 := by sorry

end NUMINAMATH_CALUDE_harry_tomato_packets_l2142_214295
