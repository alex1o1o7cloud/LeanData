import Mathlib

namespace NUMINAMATH_CALUDE_repeating_decimal_fraction_sum_l3952_395294

theorem repeating_decimal_fraction_sum (a b : ℕ+) : 
  (a.val : ℚ) / b.val = 35 / 99 → 
  Nat.gcd a.val b.val = 1 → 
  a.val + b.val = 134 := by
sorry

end NUMINAMATH_CALUDE_repeating_decimal_fraction_sum_l3952_395294


namespace NUMINAMATH_CALUDE_quadratic_root_meaningful_l3952_395289

theorem quadratic_root_meaningful (x : ℝ) : 
  (∃ (y : ℝ), y = 2 / Real.sqrt (3 + x)) ↔ x > -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_meaningful_l3952_395289


namespace NUMINAMATH_CALUDE_first_car_years_earlier_l3952_395297

-- Define the manufacture years of the cars
def first_car_year : ℕ := 1970
def third_car_year : ℕ := 2000

-- Define the time difference between the second and third cars
def years_between_second_and_third : ℕ := 20

-- Define the manufacture year of the second car
def second_car_year : ℕ := third_car_year - years_between_second_and_third

-- Theorem to prove
theorem first_car_years_earlier (h : second_car_year = third_car_year - years_between_second_and_third) :
  second_car_year - first_car_year = 10 := by
  sorry

end NUMINAMATH_CALUDE_first_car_years_earlier_l3952_395297


namespace NUMINAMATH_CALUDE_paint_remaining_l3952_395259

theorem paint_remaining (num_statues : ℕ) (paint_per_statue : ℚ) (h1 : num_statues = 3) (h2 : paint_per_statue = 1/6) : 
  num_statues * paint_per_statue = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_paint_remaining_l3952_395259


namespace NUMINAMATH_CALUDE_m_range_for_z_in_third_quadrant_l3952_395298

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := m * (3 + Complex.I) - (2 + Complex.I)

-- Define the condition for a point to be in the third quadrant
def in_third_quadrant (w : ℂ) : Prop := w.re < 0 ∧ w.im < 0

-- State the theorem
theorem m_range_for_z_in_third_quadrant :
  ∀ m : ℝ, in_third_quadrant (z m) ↔ m < 2/3 := by
  sorry

end NUMINAMATH_CALUDE_m_range_for_z_in_third_quadrant_l3952_395298


namespace NUMINAMATH_CALUDE_lcm_from_hcf_and_product_l3952_395246

theorem lcm_from_hcf_and_product (a b : ℕ+) :
  Nat.gcd a b = 20 →
  a * b = 2560 →
  Nat.lcm a b = 128 := by
sorry

end NUMINAMATH_CALUDE_lcm_from_hcf_and_product_l3952_395246


namespace NUMINAMATH_CALUDE_max_brownies_144_l3952_395209

/-- Represents the dimensions of a rectangular pan -/
structure PanDimensions where
  m : ℕ
  n : ℕ

/-- Calculates the number of interior pieces in the pan -/
def interiorPieces (d : PanDimensions) : ℕ := (d.m - 2) * (d.n - 2)

/-- Calculates the number of perimeter pieces in the pan -/
def perimeterPieces (d : PanDimensions) : ℕ := 2 * d.m + 2 * d.n - 4

/-- Represents the condition that interior pieces are twice the perimeter pieces -/
def interiorTwicePerimeter (d : PanDimensions) : Prop :=
  interiorPieces d = 2 * perimeterPieces d

/-- The theorem stating that the maximum number of brownies is 144 -/
theorem max_brownies_144 :
  ∃ (d : PanDimensions), interiorTwicePerimeter d ∧
  (∀ (d' : PanDimensions), interiorTwicePerimeter d' → d.m * d.n ≥ d'.m * d'.n) ∧
  d.m * d.n = 144 := by
  sorry

end NUMINAMATH_CALUDE_max_brownies_144_l3952_395209


namespace NUMINAMATH_CALUDE_A_power_15_minus_3_power_14_is_zero_l3952_395202

def A : Matrix (Fin 2) (Fin 2) ℝ := !![4, 5; 0, 3]

theorem A_power_15_minus_3_power_14_is_zero :
  A^15 - 3 • A^14 = 0 := by sorry

end NUMINAMATH_CALUDE_A_power_15_minus_3_power_14_is_zero_l3952_395202


namespace NUMINAMATH_CALUDE_cube_equation_solution_l3952_395215

theorem cube_equation_solution (a b : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 15 * b) : b = 147 := by
  sorry

end NUMINAMATH_CALUDE_cube_equation_solution_l3952_395215


namespace NUMINAMATH_CALUDE_chess_tournament_matches_l3952_395217

/-- The number of matches in a chess tournament -/
def tournament_matches (n : ℕ) (matches_per_pair : ℕ) : ℕ :=
  matches_per_pair * n * (n - 1) / 2

/-- Theorem: In a chess tournament with 150 players, where each player plays 3 matches
    against every other player, the total number of matches is 33,750 -/
theorem chess_tournament_matches :
  tournament_matches 150 3 = 33750 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_matches_l3952_395217


namespace NUMINAMATH_CALUDE_tangent_parallel_at_minus_one_minus_four_l3952_395229

-- Define the function
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Theorem statement
theorem tangent_parallel_at_minus_one_minus_four :
  let P₀ : ℝ × ℝ := (-1, -4)
  (f' (P₀.1) = 4) ∧ (f P₀.1 = P₀.2) := by
  sorry

end NUMINAMATH_CALUDE_tangent_parallel_at_minus_one_minus_four_l3952_395229


namespace NUMINAMATH_CALUDE_smaller_tetrahedron_volume_ratio_l3952_395262

-- Define a regular tetrahedron
structure RegularTetrahedron where
  edge_length : ℝ
  is_positive : edge_length > 0

-- Define the division of edges
def divide_edges (t : RegularTetrahedron) : ℕ := 3

-- Define the smaller tetrahedron
structure SmallerTetrahedron (t : RegularTetrahedron) where
  division_points : divide_edges t = 3

-- Define the volume ratio
def volume_ratio (t : RegularTetrahedron) (s : SmallerTetrahedron t) : ℚ := 1 / 27

-- Theorem statement
theorem smaller_tetrahedron_volume_ratio 
  (t : RegularTetrahedron) 
  (s : SmallerTetrahedron t) : 
  volume_ratio t s = 1 / 27 := by
  sorry


end NUMINAMATH_CALUDE_smaller_tetrahedron_volume_ratio_l3952_395262


namespace NUMINAMATH_CALUDE_smallest_consecutive_digit_sum_divisible_by_7_l3952_395207

-- Define a function to calculate the digit sum of a natural number
def digitSum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digitSum (n / 10)

-- Define a predicate for consecutive numbers with digit sums divisible by 7
def consecutiveDigitSumDivisibleBy7 (n : ℕ) : Prop :=
  (digitSum n) % 7 = 0 ∧ (digitSum (n + 1)) % 7 = 0

-- Theorem statement
theorem smallest_consecutive_digit_sum_divisible_by_7 :
  ∀ n : ℕ, n < 69999 → ¬(consecutiveDigitSumDivisibleBy7 n) ∧
  consecutiveDigitSumDivisibleBy7 69999 :=
by sorry

end NUMINAMATH_CALUDE_smallest_consecutive_digit_sum_divisible_by_7_l3952_395207


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3952_395278

/-- Given an arithmetic sequence {a_n} where a_5 + a_7 = 2, 
    prove that a_4 + 2a_6 + a_8 = 4 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) : 
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) → -- arithmetic sequence condition
  (a 5 + a 7 = 2) →                                -- given condition
  (a 4 + 2 * a 6 + a 8 = 4) :=                     -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3952_395278


namespace NUMINAMATH_CALUDE_honey_balance_l3952_395291

/-- The initial amount of honey produced by a bee colony -/
def initial_honey : ℝ := 0.36

/-- The amount of honey eaten by bears -/
def eaten_honey : ℝ := 0.05

/-- The amount of honey that remains -/
def remaining_honey : ℝ := 0.31

/-- Theorem stating that the initial amount of honey is equal to the sum of eaten and remaining honey -/
theorem honey_balance : initial_honey = eaten_honey + remaining_honey := by
  sorry

end NUMINAMATH_CALUDE_honey_balance_l3952_395291


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3952_395212

theorem complex_fraction_equality : (5 / 2 / (1 / 2) * (5 / 2)) / (5 / 2 * (1 / 2) / (5 / 2)) = 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3952_395212


namespace NUMINAMATH_CALUDE_largest_inexpressible_integer_l3952_395269

/-- 
Given positive integers a, b, and c with no pairwise common divisors greater than 1,
the largest integer that cannot be expressed as xbc + yca + zab 
(where x, y, z are non-negative integers) is 2abc - ab - bc - ca.
-/
theorem largest_inexpressible_integer 
  (a b c : ℕ+) 
  (h_coprime : ∀ (p : ℕ+), p ∣ a → p ∣ b → p = 1) 
  (h_coprime' : ∀ (p : ℕ+), p ∣ b → p ∣ c → p = 1) 
  (h_coprime'' : ∀ (p : ℕ+), p ∣ a → p ∣ c → p = 1) :
  ∀ n : ℤ, n > 2*a*b*c - a*b - b*c - c*a → 
  ∃ (x y z : ℕ), n = x*b*c + y*c*a + z*a*b :=
by sorry

end NUMINAMATH_CALUDE_largest_inexpressible_integer_l3952_395269


namespace NUMINAMATH_CALUDE_donkeys_and_boys_l3952_395299

theorem donkeys_and_boys (b d : ℕ) : 
  (d = b - 1) →  -- Condition 1: When each boy sits on a donkey, one boy is left
  (b / 2 = d - 1) →  -- Condition 2: When two boys sit on each donkey, one donkey is left
  (b = 4 ∧ d = 3) :=  -- Conclusion: There are 4 boys and 3 donkeys
by sorry

end NUMINAMATH_CALUDE_donkeys_and_boys_l3952_395299


namespace NUMINAMATH_CALUDE_abs_greater_than_two_necessary_not_sufficient_l3952_395288

theorem abs_greater_than_two_necessary_not_sufficient :
  (∀ x : ℝ, x < -2 → |x| > 2) ∧
  ¬(∀ x : ℝ, |x| > 2 → x < -2) :=
by sorry

end NUMINAMATH_CALUDE_abs_greater_than_two_necessary_not_sufficient_l3952_395288


namespace NUMINAMATH_CALUDE_golden_ratio_equation_l3952_395235

theorem golden_ratio_equation : 
  let x : ℝ := (Real.sqrt 5 + 1) / 2
  let y : ℝ := (Real.sqrt 5 - 1) / 2
  x^3 * y + 2 * x^2 * y^2 + x * y^3 = 5 := by sorry

end NUMINAMATH_CALUDE_golden_ratio_equation_l3952_395235


namespace NUMINAMATH_CALUDE_lcm_18_24_l3952_395227

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_24_l3952_395227


namespace NUMINAMATH_CALUDE_last_three_digits_l3952_395234

/-- A function that generates the list of positive integers with first digit 2 in increasing order -/
def digit2List : ℕ → ℕ 
| 0 => 2
| (n + 1) => 
  let prev := digit2List n
  if prev < 10 then 20
  else if prev % 10 = 9 then prev + 11
  else prev + 1

/-- The 998th digit in the digit2List -/
def digit998 : ℕ := sorry

/-- The 999th digit in the digit2List -/
def digit999 : ℕ := sorry

/-- The 1000th digit in the digit2List -/
def digit1000 : ℕ := sorry

/-- Theorem stating that the 998th, 999th, and 1000th digits form the number 216 -/
theorem last_three_digits : 
  digit998 * 100 + digit999 * 10 + digit1000 = 216 := by sorry

end NUMINAMATH_CALUDE_last_three_digits_l3952_395234


namespace NUMINAMATH_CALUDE_simplify_expression_l3952_395222

theorem simplify_expression : (6^6 * 12^6 * 6^12 * 12^12 : ℕ) = 72^18 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3952_395222


namespace NUMINAMATH_CALUDE_jimmy_remaining_cards_l3952_395231

/-- Calculates the number of cards Jimmy has left after giving cards to Bob and Mary. -/
def cards_left (initial_cards : ℕ) (cards_to_bob : ℕ) : ℕ :=
  initial_cards - cards_to_bob - (2 * cards_to_bob)

/-- Theorem stating that Jimmy has 9 cards left after giving cards to Bob and Mary. -/
theorem jimmy_remaining_cards :
  cards_left 18 3 = 9 := by
  sorry

#eval cards_left 18 3

end NUMINAMATH_CALUDE_jimmy_remaining_cards_l3952_395231


namespace NUMINAMATH_CALUDE_bryan_total_books_l3952_395272

/-- The number of bookshelves Bryan has -/
def num_bookshelves : ℕ := 15

/-- The number of books in each bookshelf -/
def books_per_shelf : ℕ := 78

/-- The total number of books Bryan has -/
def total_books : ℕ := num_bookshelves * books_per_shelf

/-- Theorem stating that Bryan has 1170 books in total -/
theorem bryan_total_books : total_books = 1170 := by
  sorry

end NUMINAMATH_CALUDE_bryan_total_books_l3952_395272


namespace NUMINAMATH_CALUDE_dans_cards_correct_l3952_395275

/-- The number of Pokemon cards Sally initially had -/
def initial_cards : ℕ := 27

/-- The number of Pokemon cards Sally lost -/
def lost_cards : ℕ := 20

/-- The number of Pokemon cards Sally has now -/
def final_cards : ℕ := 48

/-- The number of Pokemon cards Dan gave Sally -/
def dans_cards : ℕ := 41

theorem dans_cards_correct : 
  initial_cards + dans_cards - lost_cards = final_cards :=
by sorry

end NUMINAMATH_CALUDE_dans_cards_correct_l3952_395275


namespace NUMINAMATH_CALUDE_extreme_values_and_tangent_line_l3952_395270

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 5*x - 4

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 8*x + 5

theorem extreme_values_and_tangent_line :
  -- Local minimum at x = 5/3
  (∃ δ₁ > 0, ∀ x ∈ Set.Ioo (5/3 - δ₁) (5/3 + δ₁), f x ≥ f (5/3)) ∧
  f (5/3) = -58/27 ∧
  -- Local maximum at x = 1
  (∃ δ₂ > 0, ∀ x ∈ Set.Ioo (1 - δ₂) (1 + δ₂), f x ≤ f 1) ∧
  f 1 = -2 ∧
  -- Tangent line equation at x = 2
  (∀ x : ℝ, f' 2 * (x - 2) + f 2 = x - 4) := by
  sorry

end NUMINAMATH_CALUDE_extreme_values_and_tangent_line_l3952_395270


namespace NUMINAMATH_CALUDE_increasing_interval_of_sine_function_l3952_395257

open Real

theorem increasing_interval_of_sine_function 
  (f : ℝ → ℝ) (g : ℝ → ℝ) (ω : ℝ) :
  (ω > 0) →
  (∀ x, f x = 2 * sin (ω * x + π / 4)) →
  (∀ x, g x = 2 * cos (2 * x - π / 4)) →
  (∀ x, f (x + π / ω) = f x) →
  (∀ x, g (x + π) = g x) →
  (Set.Icc 0 (π / 8) : Set ℝ) = {x | x ∈ Set.Icc 0 π ∧ ∀ y ∈ Set.Icc 0 x, f y ≤ f x} :=
sorry

end NUMINAMATH_CALUDE_increasing_interval_of_sine_function_l3952_395257


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l3952_395221

/-- The distance from the focus to the directrix of the parabola y = 1/2 * x^2 is 1 -/
theorem parabola_focus_directrix_distance : 
  let p : ℝ → ℝ := fun x ↦ (1/2) * x^2
  ∃ f d : ℝ, 
    (∀ x, p x = (1/4) * (x^2 + 1)) ∧  -- Standard form of parabola
    (f = 1/2) ∧                       -- y-coordinate of focus
    (d = -1/2) ∧                      -- y-coordinate of directrix
    (f - d = 1) :=                    -- Distance between focus and directrix
by sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l3952_395221


namespace NUMINAMATH_CALUDE_temperature_proof_l3952_395283

-- Define the temperatures for each day
def monday : ℝ := sorry
def tuesday : ℝ := sorry
def wednesday : ℝ := sorry
def thursday : ℝ := sorry
def friday : ℝ := 31

-- Define the conditions
theorem temperature_proof :
  (monday + tuesday + wednesday + thursday) / 4 = 48 →
  (tuesday + wednesday + thursday + friday) / 4 = 46 →
  friday = 31 →
  monday = 39 :=
by sorry

end NUMINAMATH_CALUDE_temperature_proof_l3952_395283


namespace NUMINAMATH_CALUDE_jen_current_age_l3952_395287

/-- Jen's age when her son was born -/
def jen_age_at_birth : ℕ := 25

/-- Relationship between Jen's age and her son's age -/
def jen_age_relation (son_age : ℕ) : ℕ := 3 * son_age - 7

/-- Theorem stating Jen's current age -/
theorem jen_current_age :
  ∃ (son_age : ℕ), jen_age_at_birth + son_age = jen_age_relation son_age ∧
                   jen_age_at_birth + son_age = 41 := by
  sorry

end NUMINAMATH_CALUDE_jen_current_age_l3952_395287


namespace NUMINAMATH_CALUDE_topsoil_cost_l3952_395248

/-- The cost of topsoil in dollars per cubic foot -/
def cost_per_cubic_foot : ℝ := 7

/-- The volume of topsoil in cubic yards -/
def volume_in_cubic_yards : ℝ := 8

/-- The conversion factor from cubic yards to cubic feet -/
def cubic_yards_to_cubic_feet : ℝ := 27

/-- The total cost of topsoil in dollars -/
def total_cost : ℝ := 1512

theorem topsoil_cost : 
  cost_per_cubic_foot * volume_in_cubic_yards * cubic_yards_to_cubic_feet = total_cost := by
  sorry

end NUMINAMATH_CALUDE_topsoil_cost_l3952_395248


namespace NUMINAMATH_CALUDE_equation_solution_l3952_395274

theorem equation_solution (x : ℚ) : 9 / (5 + 3 / x) = 1 → x = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3952_395274


namespace NUMINAMATH_CALUDE_log_8_4_equals_twice_log_8_2_l3952_395241

-- Define log_8 as a function
noncomputable def log_8 (x : ℝ) : ℝ := Real.log x / Real.log 8

-- State the theorem
theorem log_8_4_equals_twice_log_8_2 :
  ∃ (ε : ℝ), ε ≥ 0 ∧ ε < 0.00005 ∧ 
  ∃ (δ : ℝ), δ ≥ 0 ∧ δ < 0.00005 ∧
  |log_8 2 - 0.2525| ≤ ε →
  |log_8 4 - 2 * log_8 2| ≤ δ :=
sorry

end NUMINAMATH_CALUDE_log_8_4_equals_twice_log_8_2_l3952_395241


namespace NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l3952_395251

/-- A geometric sequence of positive integers -/
def GeometricSequence (a : ℕ → ℕ) : Prop :=
  ∃ r : ℕ, r > 1 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem sixth_term_of_geometric_sequence
  (a : ℕ → ℕ)
  (h_geometric : GeometricSequence a)
  (h_first : a 1 = 3)
  (h_fifth : a 5 = 243) :
  a 6 = 729 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l3952_395251


namespace NUMINAMATH_CALUDE_hyperbola_range_l3952_395282

/-- The equation represents a hyperbola -/
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m + 2) + y^2 / (m + 1) = 1 ∧ (m + 2) * (m + 1) < 0

/-- The range of m for which the equation represents a hyperbola -/
theorem hyperbola_range :
  ∀ m : ℝ, is_hyperbola m ↔ -2 < m ∧ m < -1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_range_l3952_395282


namespace NUMINAMATH_CALUDE_min_production_to_meet_demand_l3952_395205

/-- Total market demand function -/
def f (x : ℕ) : ℕ := x * (x + 1) * (35 - 2 * x)

/-- Monthly demand function -/
def g (x : ℕ) : ℤ := f x - f (x - 1)

/-- The range of valid month numbers -/
def valid_months : Set ℕ := {x | 1 ≤ x ∧ x ≤ 12}

theorem min_production_to_meet_demand :
  ∃ (a : ℕ), (∀ x ∈ valid_months, (g x : ℝ) ≤ a) ∧
  (∀ b : ℕ, (∀ x ∈ valid_months, (g x : ℝ) ≤ b) → a ≤ b) ∧
  a = 171 := by
  sorry

end NUMINAMATH_CALUDE_min_production_to_meet_demand_l3952_395205


namespace NUMINAMATH_CALUDE_kit_prices_correct_l3952_395261

/-- The price of kit B in yuan -/
def price_B : ℝ := 150

/-- The price of kit A in yuan -/
def price_A : ℝ := 180

/-- The relationship between the prices of kit A and kit B -/
def price_relation : Prop := price_A = 1.2 * price_B

/-- The equation representing the difference in quantities purchased -/
def quantity_difference : Prop :=
  (9900 / price_A) - (7500 / price_B) = 5

theorem kit_prices_correct :
  price_relation ∧ quantity_difference → price_A = 180 ∧ price_B = 150 := by
  sorry

end NUMINAMATH_CALUDE_kit_prices_correct_l3952_395261


namespace NUMINAMATH_CALUDE_kim_cherry_difference_l3952_395258

/-- The number of questions Nicole answered correctly -/
def nicole_correct : ℕ := 22

/-- The number of questions Cherry answered correctly -/
def cherry_correct : ℕ := 17

/-- The number of questions Kim answered correctly -/
def kim_correct : ℕ := nicole_correct + 3

theorem kim_cherry_difference : kim_correct - cherry_correct = 8 := by
  sorry

end NUMINAMATH_CALUDE_kim_cherry_difference_l3952_395258


namespace NUMINAMATH_CALUDE_charlyn_visible_area_l3952_395292

/-- The area of the region visible to Charlyn during her walk around a square -/
def visible_area (square_side : ℝ) (visibility_range : ℝ) : ℝ :=
  let inner_square_side := square_side - 2 * visibility_range
  let inner_area := inner_square_side ^ 2
  let outer_rectangles_area := 4 * (square_side * visibility_range)
  let corner_squares_area := 4 * (visibility_range ^ 2)
  (square_side ^ 2 - inner_area) + outer_rectangles_area + corner_squares_area

/-- Theorem stating that the visible area for Charlyn's walk is 160 km² -/
theorem charlyn_visible_area :
  visible_area 10 2 = 160 := by
  sorry

#eval visible_area 10 2

end NUMINAMATH_CALUDE_charlyn_visible_area_l3952_395292


namespace NUMINAMATH_CALUDE_rectangle_length_from_square_wire_l3952_395239

/-- Given a square with side length 20 cm and a rectangle with width 14 cm made from the same total wire length, the length of the rectangle is 26 cm. -/
theorem rectangle_length_from_square_wire (square_side : ℝ) (rect_width : ℝ) (rect_length : ℝ) : 
  square_side = 20 →
  rect_width = 14 →
  4 * square_side = 2 * (rect_width + rect_length) →
  rect_length = 26 := by
sorry

end NUMINAMATH_CALUDE_rectangle_length_from_square_wire_l3952_395239


namespace NUMINAMATH_CALUDE_total_blue_balloons_l3952_395233

/-- The number of blue balloons Joan and Melanie have in total is 81, 
    given that Joan has 40 and Melanie has 41. -/
theorem total_blue_balloons (joan_balloons melanie_balloons : ℕ) 
  (h1 : joan_balloons = 40) 
  (h2 : melanie_balloons = 41) : 
  joan_balloons + melanie_balloons = 81 := by
  sorry

end NUMINAMATH_CALUDE_total_blue_balloons_l3952_395233


namespace NUMINAMATH_CALUDE_line_inclination_l3952_395271

def line_equation (x y : ℝ) : Prop := y = x + 1

def angle_of_inclination (θ : ℝ) : Prop := θ = Real.arctan 1

theorem line_inclination :
  ∀ x y θ : ℝ, line_equation x y → angle_of_inclination θ → θ * (180 / Real.pi) = 45 :=
by sorry

end NUMINAMATH_CALUDE_line_inclination_l3952_395271


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcf_six_exists_138_unique_greatest_l3952_395220

theorem greatest_integer_with_gcf_six (n : ℕ) : n < 150 ∧ Nat.gcd n 18 = 6 → n ≤ 138 :=
by sorry

theorem exists_138 : 138 < 150 ∧ Nat.gcd 138 18 = 6 :=
by sorry

theorem unique_greatest : ∀ m : ℕ, m > 138 → m < 150 → Nat.gcd m 18 ≠ 6 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcf_six_exists_138_unique_greatest_l3952_395220


namespace NUMINAMATH_CALUDE_max_qpn_value_l3952_395245

/-- Represents a two-digit number with equal digits -/
def TwoDigitEqualDigits (n : Nat) : Prop :=
  n ≥ 11 ∧ n ≤ 99 ∧ n % 11 = 0

/-- Represents a one-digit number -/
def OneDigit (n : Nat) : Prop :=
  n ≥ 1 ∧ n ≤ 9

/-- Represents a three-digit number -/
def ThreeDigits (n : Nat) : Prop :=
  n ≥ 100 ∧ n ≤ 999

theorem max_qpn_value (nn n qpn : Nat) 
  (h1 : TwoDigitEqualDigits nn)
  (h2 : OneDigit n)
  (h3 : ThreeDigits qpn)
  (h4 : nn * n = qpn) :
  qpn ≤ 396 :=
sorry

end NUMINAMATH_CALUDE_max_qpn_value_l3952_395245


namespace NUMINAMATH_CALUDE_slope_angle_of_line_PQ_l3952_395260

theorem slope_angle_of_line_PQ 
  (a b c : ℝ) 
  (hab : a ≠ b) 
  (hbc : b ≠ c) 
  (hac : a ≠ c) 
  (P : ℝ × ℝ) 
  (hP : P = (b, b + c)) 
  (Q : ℝ × ℝ) 
  (hQ : Q = (a, c + a)) : 
  Real.arctan ((Q.2 - P.2) / (Q.1 - P.1)) = π / 4 := by
  sorry

#check slope_angle_of_line_PQ

end NUMINAMATH_CALUDE_slope_angle_of_line_PQ_l3952_395260


namespace NUMINAMATH_CALUDE_tshirt_jersey_cost_difference_l3952_395265

/-- The amount the Razorback shop makes off each t-shirt -/
def tshirt_profit : ℕ := 192

/-- The amount the Razorback shop makes off each jersey -/
def jersey_profit : ℕ := 34

/-- The difference in cost between a t-shirt and a jersey -/
def cost_difference : ℕ := tshirt_profit - jersey_profit

theorem tshirt_jersey_cost_difference :
  cost_difference = 158 :=
sorry

end NUMINAMATH_CALUDE_tshirt_jersey_cost_difference_l3952_395265


namespace NUMINAMATH_CALUDE_tournament_games_count_l3952_395295

/-- Calculates the total number of games played in a tournament given the ratio of outcomes and the number of games won. -/
def total_games (ratio_won ratio_lost ratio_tied : ℕ) (games_won : ℕ) : ℕ :=
  let games_per_ratio := games_won / ratio_won
  let games_lost := ratio_lost * games_per_ratio
  let games_tied := ratio_tied * games_per_ratio
  games_won + games_lost + games_tied

/-- Theorem stating that given a ratio of 7:4:5 for games won:lost:tied and 42 games won, the total number of games played is 96. -/
theorem tournament_games_count :
  total_games 7 4 5 42 = 96 := by
  sorry

end NUMINAMATH_CALUDE_tournament_games_count_l3952_395295


namespace NUMINAMATH_CALUDE_time_to_work_l3952_395240

def round_trip_time : ℝ := 2
def speed_to_work : ℝ := 80
def speed_to_home : ℝ := 120

theorem time_to_work :
  let distance := (round_trip_time * speed_to_work * speed_to_home) / (speed_to_work + speed_to_home)
  let time_to_work := distance / speed_to_work
  time_to_work * 60 = 72 := by sorry

end NUMINAMATH_CALUDE_time_to_work_l3952_395240


namespace NUMINAMATH_CALUDE_abc_product_l3952_395208

theorem abc_product (a b c : ℝ) 
  (eq1 : b + c = 16) 
  (eq2 : c + a = 17) 
  (eq3 : a + b = 18) : 
  a * b * c = 606.375 := by
sorry

end NUMINAMATH_CALUDE_abc_product_l3952_395208


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l3952_395200

theorem quadratic_roots_product (x : ℝ) : 
  (x - 4) * (2 * x + 10) = x^2 - 15 * x + 56 → 
  ∃ a b c : ℝ, a * x^2 + b * x + c = 0 ∧ (c / a) + 6 = -90 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l3952_395200


namespace NUMINAMATH_CALUDE_unique_rectangle_exists_restore_coordinate_system_l3952_395290

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle in 2D space -/
structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Checks if four points form a rectangle -/
def isRectangle (r : Rectangle) : Prop :=
  let AB := (r.B.x - r.A.x)^2 + (r.B.y - r.A.y)^2
  let BC := (r.C.x - r.B.x)^2 + (r.C.y - r.B.y)^2
  let CD := (r.D.x - r.C.x)^2 + (r.D.y - r.C.y)^2
  let DA := (r.A.x - r.D.x)^2 + (r.A.y - r.D.y)^2
  AB = CD ∧ BC = DA ∧ 
  (r.B.x - r.A.x) * (r.C.x - r.B.x) + (r.B.y - r.A.y) * (r.C.y - r.B.y) = 0

/-- Theorem: Given two points A and B, there exists a unique rectangle with A and B as diagonal endpoints -/
theorem unique_rectangle_exists (A B : Point) : 
  ∃! (r : Rectangle), r.A = A ∧ r.B = B ∧ isRectangle r := by
  sorry

/-- Main theorem: Given points A(1,2) and B(3,1), a unique rectangle can be constructed 
    with A and B as diagonal endpoints, which is sufficient to restore the coordinate system -/
theorem restore_coordinate_system : 
  let A : Point := ⟨1, 2⟩
  let B : Point := ⟨3, 1⟩
  ∃! (r : Rectangle), r.A = A ∧ r.B = B ∧ isRectangle r := by
  sorry

end NUMINAMATH_CALUDE_unique_rectangle_exists_restore_coordinate_system_l3952_395290


namespace NUMINAMATH_CALUDE_sequence_sum_l3952_395266

theorem sequence_sum (a b c d : ℕ) 
  (h1 : b - a = d - c) 
  (h2 : d - a = 24) 
  (h3 : b - a = (d - c) + 2) 
  (h4 : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) : 
  a + b + c + d = 54 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l3952_395266


namespace NUMINAMATH_CALUDE_perpendicular_lines_triangle_area_l3952_395238

/-- Two perpendicular lines intersecting at (8,6) with y-intercepts differing by 14 form a triangle with area 56 -/
theorem perpendicular_lines_triangle_area :
  ∀ (m₁ m₂ b₁ b₂ : ℝ),
  m₁ * m₂ = -1 →                         -- perpendicular lines
  8 * m₁ + b₁ = 6 →                      -- line 1 passes through (8,6)
  8 * m₂ + b₂ = 6 →                      -- line 2 passes through (8,6)
  b₁ - b₂ = 14 →                         -- difference between y-intercepts
  (1/2) * 8 * |b₁ - b₂| = 56 :=          -- area of triangle
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_triangle_area_l3952_395238


namespace NUMINAMATH_CALUDE_f₁_eq_f₂_l3952_395249

/-- Function f₁ that always returns 1 -/
def f₁ : ℝ → ℝ := λ _ => 1

/-- Function f₂ that returns x^0 -/
def f₂ : ℝ → ℝ := λ x => x^0

/-- Theorem stating that f₁ and f₂ are the same function -/
theorem f₁_eq_f₂ : f₁ = f₂ := by sorry

end NUMINAMATH_CALUDE_f₁_eq_f₂_l3952_395249


namespace NUMINAMATH_CALUDE_toys_in_box_time_l3952_395206

/-- The time in minutes required to put all toys in the box -/
def time_to_put_toys_in_box (total_toys : ℕ) (mom_puts_in : ℕ) (mia_takes_out : ℕ) (cycle_time : ℕ) : ℕ :=
  let net_gain_per_cycle := mom_puts_in - mia_takes_out
  let cycles_after_first_minute := (total_toys - 2 * mom_puts_in) / net_gain_per_cycle
  let total_seconds := (cycles_after_first_minute + 2) * cycle_time
  total_seconds / 60

/-- Theorem stating that under the given conditions, it takes 22 minutes to put all toys in the box -/
theorem toys_in_box_time : time_to_put_toys_in_box 45 4 3 30 = 22 := by
  sorry

end NUMINAMATH_CALUDE_toys_in_box_time_l3952_395206


namespace NUMINAMATH_CALUDE_prob_two_red_cards_l3952_395250

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (suits : ℕ)
  (cards_per_suit : ℕ)
  (red_suits : ℕ)
  (black_suits : ℕ)

/-- Defines a standard 52-card deck -/
def standard_deck : Deck :=
  { total_cards := 52,
    suits := 4,
    cards_per_suit := 13,
    red_suits := 2,
    black_suits := 2 }

/-- The probability of drawing a red card from the deck -/
def prob_red_card (d : Deck) : ℚ :=
  (d.red_suits * d.cards_per_suit) / d.total_cards

/-- Theorem: The probability of drawing two red cards in succession with replacement is 1/4 -/
theorem prob_two_red_cards (d : Deck) (h : d = standard_deck) :
  (prob_red_card d) * (prob_red_card d) = 1 / 4 := by
  sorry


end NUMINAMATH_CALUDE_prob_two_red_cards_l3952_395250


namespace NUMINAMATH_CALUDE_smaller_root_of_equation_l3952_395273

theorem smaller_root_of_equation :
  let f (x : ℚ) := (x - 7/8)^2 + (x - 1/4) * (x - 7/8)
  ∃ (r : ℚ), f r = 0 ∧ r < 9/16 ∧ f (9/16) = 0 :=
by sorry

end NUMINAMATH_CALUDE_smaller_root_of_equation_l3952_395273


namespace NUMINAMATH_CALUDE_train_b_start_time_l3952_395224

/-- The time when trains meet, in hours after midnight -/
def meeting_time : ℝ := 12

/-- The time when train A starts, in hours after midnight -/
def train_a_start : ℝ := 8

/-- The distance between city A and city B in kilometers -/
def total_distance : ℝ := 465

/-- The speed of train A in km/hr -/
def train_a_speed : ℝ := 60

/-- The speed of train B in km/hr -/
def train_b_speed : ℝ := 75

/-- The theorem stating that the train from city B starts at 9 a.m. -/
theorem train_b_start_time :
  ∃ (t : ℝ),
    t = 9 ∧
    (meeting_time - train_a_start) * train_a_speed +
      (meeting_time - t) * train_b_speed = total_distance :=
by sorry

end NUMINAMATH_CALUDE_train_b_start_time_l3952_395224


namespace NUMINAMATH_CALUDE_isosceles_triangle_condition_l3952_395276

theorem isosceles_triangle_condition (A B C : ℝ) (h_triangle : A + B + C = π) 
  (h_condition : 2 * Real.cos B * Real.sin A = Real.sin C) : A = B := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_condition_l3952_395276


namespace NUMINAMATH_CALUDE_min_value_of_f_l3952_395237

noncomputable section

def f (x : ℝ) : ℝ := (1/2) * x - Real.sin x

theorem min_value_of_f :
  ∃ (min : ℝ), min = π/6 - Real.sqrt 3/2 ∧
  ∀ x ∈ Set.Ioo 0 π, f x ≥ min :=
sorry

end

end NUMINAMATH_CALUDE_min_value_of_f_l3952_395237


namespace NUMINAMATH_CALUDE_a_formula_l3952_395254

noncomputable section

/-- The function f(x) = x / sqrt(1 + x^2) -/
def f (x : ℝ) : ℝ := x / Real.sqrt (1 + x^2)

/-- The sequence a_n defined recursively -/
def a (x : ℝ) : ℕ → ℝ
  | 0 => f x
  | n + 1 => f (a x n)

/-- The theorem stating the general formula for a_n -/
theorem a_formula (x : ℝ) (h : x > 0) (n : ℕ) :
  a x n = x / Real.sqrt (1 + n * x^2) := by
  sorry

end

end NUMINAMATH_CALUDE_a_formula_l3952_395254


namespace NUMINAMATH_CALUDE_exist_a_with_two_subsets_a_eq_one_implies_A_eq_two_thirds_a_eq_neg_one_eighth_implies_A_eq_four_thirds_l3952_395214

/-- The set A defined by the quadratic equation (a-1)x^2 + 3x - 2 = 0 -/
def A (a : ℝ) : Set ℝ := {x : ℝ | (a - 1) * x^2 + 3 * x - 2 = 0}

/-- The theorem stating the existence of 'a' values for which A has exactly two subsets -/
theorem exist_a_with_two_subsets :
  ∃ (a₁ a₂ : ℝ), a₁ ≠ a₂ ∧ 
  (∀ (S : Set ℝ), S ⊆ A a₁ → (S = ∅ ∨ S = A a₁)) ∧
  (∀ (S : Set ℝ), S ⊆ A a₂ → (S = ∅ ∨ S = A a₂)) ∧
  a₁ = 1 ∧ a₂ = -1/8 := by
  sorry

/-- The theorem stating that for a = 1, A = {2/3} -/
theorem a_eq_one_implies_A_eq_two_thirds :
  A 1 = {2/3} := by
  sorry

/-- The theorem stating that for a = -1/8, A = {4/3} -/
theorem a_eq_neg_one_eighth_implies_A_eq_four_thirds :
  A (-1/8) = {4/3} := by
  sorry

end NUMINAMATH_CALUDE_exist_a_with_two_subsets_a_eq_one_implies_A_eq_two_thirds_a_eq_neg_one_eighth_implies_A_eq_four_thirds_l3952_395214


namespace NUMINAMATH_CALUDE_profit_division_time_l3952_395285

/-- Represents the partnership problem with given conditions -/
def PartnershipProblem (initial_ratio_p initial_ratio_q initial_ratio_r : ℚ)
  (withdrawal_time : ℕ) (withdrawal_fraction : ℚ)
  (total_profit r_profit : ℚ) : Prop :=
  -- Initial ratio of shares
  initial_ratio_p + initial_ratio_q + initial_ratio_r = 1 ∧
  -- p withdraws half of the capital after two months
  withdrawal_time = 2 ∧
  withdrawal_fraction = 1/2 ∧
  -- Given total profit and r's share
  total_profit > 0 ∧
  r_profit > 0 ∧
  r_profit < total_profit

/-- Theorem stating the number of months after which the profit was divided -/
theorem profit_division_time (initial_ratio_p initial_ratio_q initial_ratio_r : ℚ)
  (withdrawal_time : ℕ) (withdrawal_fraction : ℚ)
  (total_profit r_profit : ℚ) :
  PartnershipProblem initial_ratio_p initial_ratio_q initial_ratio_r
    withdrawal_time withdrawal_fraction total_profit r_profit →
  ∃ (n : ℕ), n = 12 := by
  sorry

end NUMINAMATH_CALUDE_profit_division_time_l3952_395285


namespace NUMINAMATH_CALUDE_bacon_suggestion_count_l3952_395277

theorem bacon_suggestion_count (mashed_potatoes : ℕ) (bacon : ℕ) : 
  mashed_potatoes = 479 → 
  bacon = mashed_potatoes + 10 → 
  bacon = 489 := by
sorry

end NUMINAMATH_CALUDE_bacon_suggestion_count_l3952_395277


namespace NUMINAMATH_CALUDE_equation_condition_for_x_equals_4_l3952_395244

theorem equation_condition_for_x_equals_4 :
  (∃ x : ℝ, x^2 - 3*x - 4 = 0) ∧
  (∀ x : ℝ, x = 4 → x^2 - 3*x - 4 = 0) ∧
  (∃ x : ℝ, x^2 - 3*x - 4 = 0 ∧ x ≠ 4) :=
by sorry

end NUMINAMATH_CALUDE_equation_condition_for_x_equals_4_l3952_395244


namespace NUMINAMATH_CALUDE_als_initial_investment_l3952_395225

theorem als_initial_investment (a b c : ℝ) : 
  a + b + c = 2000 →
  3*a + 2*b + 2*c = 3500 →
  a = 500 := by
sorry

end NUMINAMATH_CALUDE_als_initial_investment_l3952_395225


namespace NUMINAMATH_CALUDE_family_gathering_handshakes_l3952_395284

/-- The number of handshakes at a family gathering --/
def total_handshakes (twin_sets : ℕ) (triplet_sets : ℕ) : ℕ :=
  let twin_count := twin_sets * 2
  let triplet_count := triplet_sets * 3
  let twin_handshakes := (twin_count * (twin_count - 2)) / 2
  let triplet_handshakes := (triplet_count * (triplet_count - 3)) / 2
  let twin_triplet_handshakes := twin_count * (triplet_count / 3) + triplet_count * (twin_count / 4)
  twin_handshakes + triplet_handshakes + twin_triplet_handshakes

/-- Theorem stating the total number of handshakes at the family gathering --/
theorem family_gathering_handshakes :
  total_handshakes 10 7 = 614 := by
  sorry

end NUMINAMATH_CALUDE_family_gathering_handshakes_l3952_395284


namespace NUMINAMATH_CALUDE_abhinav_bhupathi_total_money_l3952_395213

/-- The problem of calculating the total amount of money Abhinav and Bhupathi have together. -/
theorem abhinav_bhupathi_total_money (abhinav_amount bhupathi_amount : ℚ) : 
  (4 : ℚ) / 15 * abhinav_amount = (2 : ℚ) / 5 * bhupathi_amount →
  bhupathi_amount = 484 →
  abhinav_amount + bhupathi_amount = 1210 := by
  sorry

#check abhinav_bhupathi_total_money

end NUMINAMATH_CALUDE_abhinav_bhupathi_total_money_l3952_395213


namespace NUMINAMATH_CALUDE_plants_original_cost_l3952_395243

/-- Given a discount and the amount spent on plants, calculate the original cost. -/
def original_cost (discount : ℚ) (amount_spent : ℚ) : ℚ :=
  discount + amount_spent

/-- Theorem stating that given the specific discount and amount spent, the original cost is $467.00 -/
theorem plants_original_cost :
  let discount : ℚ := 399
  let amount_spent : ℚ := 68
  original_cost discount amount_spent = 467 := by
sorry

end NUMINAMATH_CALUDE_plants_original_cost_l3952_395243


namespace NUMINAMATH_CALUDE_circle_equation_l3952_395293

/-- A circle with center on the x-axis, radius √2, passing through (-2, 1) -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  passes_through : ℝ × ℝ
  center_on_x_axis : center.2 = 0
  radius_is_sqrt_2 : radius = Real.sqrt 2
  passes_through_point : passes_through = (-2, 1)

/-- The equation of the circle is either (x+1)² + y² = 2 or (x+3)² + y² = 2 -/
theorem circle_equation (c : Circle) :
  (∀ x y : ℝ, (x + 1)^2 + y^2 = 2 ∨ (x + 3)^2 + y^2 = 2 ↔
    (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l3952_395293


namespace NUMINAMATH_CALUDE_shell_distribution_l3952_395242

theorem shell_distribution (jillian savannah clayton friends_share : ℕ) : 
  jillian = 29 →
  savannah = 17 →
  clayton = 8 →
  friends_share = 27 →
  (jillian + savannah + clayton) / friends_share = 2 :=
by sorry

end NUMINAMATH_CALUDE_shell_distribution_l3952_395242


namespace NUMINAMATH_CALUDE_max_q_plus_2r_l3952_395219

theorem max_q_plus_2r (q r : ℕ+) (h : 1230 = 28 * q + r) : 
  (∀ q' r' : ℕ+, 1230 = 28 * q' + r' → q' + 2 * r' ≤ q + 2 * r) ∧ q + 2 * r = 95 := by
  sorry

end NUMINAMATH_CALUDE_max_q_plus_2r_l3952_395219


namespace NUMINAMATH_CALUDE_dusty_paid_hundred_l3952_395201

/-- Represents the cost and quantity of cake slices --/
structure CakeOrder where
  single_layer_cost : ℕ
  double_layer_cost : ℕ
  single_layer_quantity : ℕ
  double_layer_quantity : ℕ

/-- Calculates the total cost of the cake order --/
def total_cost (order : CakeOrder) : ℕ :=
  order.single_layer_cost * order.single_layer_quantity +
  order.double_layer_cost * order.double_layer_quantity

/-- Represents Dusty's cake purchase and change received --/
structure DustysPurchase where
  order : CakeOrder
  change_received : ℕ

/-- Theorem: Given Dusty's cake purchase and change received, prove that he paid $100 --/
theorem dusty_paid_hundred (purchase : DustysPurchase)
  (h1 : purchase.order.single_layer_cost = 4)
  (h2 : purchase.order.double_layer_cost = 7)
  (h3 : purchase.order.single_layer_quantity = 7)
  (h4 : purchase.order.double_layer_quantity = 5)
  (h5 : purchase.change_received = 37) :
  total_cost purchase.order + purchase.change_received = 100 := by
  sorry


end NUMINAMATH_CALUDE_dusty_paid_hundred_l3952_395201


namespace NUMINAMATH_CALUDE_cookout_bun_packs_l3952_395226

/-- Calculate the number of bun packs needed for a cookout --/
theorem cookout_bun_packs 
  (total_friends : ℕ) 
  (burgers_per_guest : ℕ) 
  (non_meat_eaters : ℕ) 
  (no_bread_eaters : ℕ) 
  (gluten_free_friends : ℕ) 
  (nut_allergy_friends : ℕ)
  (regular_buns_per_pack : ℕ) 
  (gluten_free_buns_per_pack : ℕ) 
  (nut_free_buns_per_pack : ℕ)
  (h1 : total_friends = 35)
  (h2 : burgers_per_guest = 3)
  (h3 : non_meat_eaters = 7)
  (h4 : no_bread_eaters = 4)
  (h5 : gluten_free_friends = 3)
  (h6 : nut_allergy_friends = 1)
  (h7 : regular_buns_per_pack = 15)
  (h8 : gluten_free_buns_per_pack = 6)
  (h9 : nut_free_buns_per_pack = 5) :
  (((total_friends - non_meat_eaters) * burgers_per_guest - no_bread_eaters * burgers_per_guest + regular_buns_per_pack - 1) / regular_buns_per_pack = 5) ∧ 
  ((gluten_free_friends * burgers_per_guest + gluten_free_buns_per_pack - 1) / gluten_free_buns_per_pack = 2) ∧
  ((nut_allergy_friends * burgers_per_guest + nut_free_buns_per_pack - 1) / nut_free_buns_per_pack = 1) :=
by sorry

end NUMINAMATH_CALUDE_cookout_bun_packs_l3952_395226


namespace NUMINAMATH_CALUDE_solution_to_equation_l3952_395232

theorem solution_to_equation (y : ℝ) (h1 : y ≠ 3) (h2 : y ≠ 3/2) :
  (y^2 - 11*y + 24)/(y - 3) + (2*y^2 + 7*y - 18)/(2*y - 3) = -10 ↔ y = -4 :=
by sorry

end NUMINAMATH_CALUDE_solution_to_equation_l3952_395232


namespace NUMINAMATH_CALUDE_arccos_sqrt2_over_2_l3952_395223

theorem arccos_sqrt2_over_2 : Real.arccos (Real.sqrt 2 / 2) = π / 4 := by sorry

end NUMINAMATH_CALUDE_arccos_sqrt2_over_2_l3952_395223


namespace NUMINAMATH_CALUDE_opposite_direction_time_calculation_l3952_395204

/-- Given two people moving in opposite directions from the same starting point,
    calculate the time taken to reach a specific distance between them. -/
theorem opposite_direction_time_calculation 
  (speed1 : ℝ) (speed2 : ℝ) (distance : ℝ) 
  (h1 : speed1 = 2) 
  (h2 : speed2 = 3) 
  (h3 : distance = 20) : 
  distance / (speed1 + speed2) = 4 := by
  sorry

#check opposite_direction_time_calculation

end NUMINAMATH_CALUDE_opposite_direction_time_calculation_l3952_395204


namespace NUMINAMATH_CALUDE_modified_fibonacci_series_sum_l3952_395228

/-- Modified Fibonacci sequence -/
def F : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | (n + 2) => F (n + 1) + F n

/-- The sum of the series F_n / 5^n from n = 0 to infinity -/
noncomputable def seriesSum : ℝ := ∑' n, (F n : ℝ) / 5^n

theorem modified_fibonacci_series_sum : seriesSum = 35 / 18 := by
  sorry

end NUMINAMATH_CALUDE_modified_fibonacci_series_sum_l3952_395228


namespace NUMINAMATH_CALUDE_nonnegative_solutions_count_l3952_395263

theorem nonnegative_solutions_count : ∃! (x : ℝ), x ≥ 0 ∧ x^2 + 6*x = 18 := by
  sorry

end NUMINAMATH_CALUDE_nonnegative_solutions_count_l3952_395263


namespace NUMINAMATH_CALUDE_square_value_theorem_l3952_395210

theorem square_value_theorem (a b : ℝ) (h : a > b) :
  ∃ square : ℝ, (-2*a - 1 < -2*b + square) ∧ (square = 0) := by
sorry

end NUMINAMATH_CALUDE_square_value_theorem_l3952_395210


namespace NUMINAMATH_CALUDE_square_b_minus_d_l3952_395252

theorem square_b_minus_d (a b c d : ℝ) 
  (eq1 : a - b - c + d = 13) 
  (eq2 : a + b - c - d = 9) : 
  (b - d)^2 = 4 := by sorry

end NUMINAMATH_CALUDE_square_b_minus_d_l3952_395252


namespace NUMINAMATH_CALUDE_symmetry_yoz_plane_l3952_395203

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The yoz plane in 3D space -/
def yozPlane : Set Point3D := {p : Point3D | p.x = 0}

/-- Symmetry with respect to the yoz plane -/
def symmetricPointYOZ (p : Point3D) : Point3D :=
  ⟨-p.x, p.y, p.z⟩

theorem symmetry_yoz_plane :
  let p : Point3D := ⟨2, 3, 5⟩
  symmetricPointYOZ p = ⟨-2, 3, 5⟩ := by
  sorry

end NUMINAMATH_CALUDE_symmetry_yoz_plane_l3952_395203


namespace NUMINAMATH_CALUDE_ac_squared_gt_bc_squared_implies_a_gt_b_l3952_395268

theorem ac_squared_gt_bc_squared_implies_a_gt_b (a b c : ℝ) :
  a * c^2 > b * c^2 → a > b :=
by sorry

end NUMINAMATH_CALUDE_ac_squared_gt_bc_squared_implies_a_gt_b_l3952_395268


namespace NUMINAMATH_CALUDE_stating_not_always_triangle_from_parallelogram_l3952_395281

/-- A stick represents a line segment with a positive length. -/
structure Stick :=
  (length : ℝ)
  (positive : length > 0)

/-- A parallelogram composed of four equal sticks. -/
structure Parallelogram :=
  (stick : Stick)

/-- Represents a potential triangle formed from the parallelogram's sticks. -/
structure PotentialTriangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (side3 : ℝ)

/-- Checks if a triangle can be formed given three side lengths. -/
def isValidTriangle (t : PotentialTriangle) : Prop :=
  t.side1 + t.side2 > t.side3 ∧
  t.side2 + t.side3 > t.side1 ∧
  t.side1 + t.side3 > t.side2

/-- 
Theorem stating that it's not always possible to form a triangle 
from a parallelogram's sticks.
-/
theorem not_always_triangle_from_parallelogram :
  ∃ p : Parallelogram, ¬∃ t : PotentialTriangle, 
    (t.side1 = p.stick.length ∧ t.side2 = p.stick.length ∧ t.side3 = 2 * p.stick.length) ∧
    isValidTriangle t :=
sorry

end NUMINAMATH_CALUDE_stating_not_always_triangle_from_parallelogram_l3952_395281


namespace NUMINAMATH_CALUDE_complex_square_equality_l3952_395230

theorem complex_square_equality (a b : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (a + b * Complex.I)^2 = (3 : ℂ) + 4 * Complex.I →
  a^2 + b^2 = 5 ∧ a * b = 2 := by
sorry

end NUMINAMATH_CALUDE_complex_square_equality_l3952_395230


namespace NUMINAMATH_CALUDE_milk_consumption_l3952_395211

theorem milk_consumption (bottle_milk : ℚ) (pour_fraction : ℚ) (drink_fraction : ℚ) :
  bottle_milk = 3/4 →
  pour_fraction = 1/2 →
  drink_fraction = 1/3 →
  drink_fraction * (pour_fraction * bottle_milk) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_milk_consumption_l3952_395211


namespace NUMINAMATH_CALUDE_quadrilateral_offset_l3952_395253

theorem quadrilateral_offset (diagonal : ℝ) (offset1 : ℝ) (area : ℝ) :
  diagonal = 30 →
  offset1 = 9 →
  area = 225 →
  ∃ offset2 : ℝ, 
    offset2 = 6 ∧ 
    area = (1/2) * diagonal * offset1 + (1/2) * diagonal * offset2 :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_offset_l3952_395253


namespace NUMINAMATH_CALUDE_greatest_length_of_rope_pieces_l3952_395236

theorem greatest_length_of_rope_pieces : Nat.gcd 28 (Nat.gcd 42 70) = 7 := by sorry

end NUMINAMATH_CALUDE_greatest_length_of_rope_pieces_l3952_395236


namespace NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l3952_395255

theorem square_sum_given_difference_and_product (x y : ℝ) 
  (h1 : x - y = 20) (h2 : x * y = 9) : x^2 + y^2 = 418 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l3952_395255


namespace NUMINAMATH_CALUDE_money_distribution_l3952_395280

theorem money_distribution (A B C : ℕ) 
  (h1 : A + B + C = 500) 
  (h2 : B + C = 330) 
  (h3 : C = 30) : 
  A + C = 200 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l3952_395280


namespace NUMINAMATH_CALUDE_cell_count_after_ten_days_l3952_395256

/-- Represents the cell division process over 10 days -/
def cellDivision (initialCells : ℕ) (firstSplitFactor : ℕ) (laterSplitFactor : ℕ) (totalDays : ℕ) : ℕ :=
  let afterFirstTwoDays := initialCells * firstSplitFactor
  let remainingDivisions := (totalDays - 2) / 2
  afterFirstTwoDays * laterSplitFactor ^ remainingDivisions

/-- Theorem stating the number of cells after 10 days -/
theorem cell_count_after_ten_days :
  cellDivision 5 3 2 10 = 240 := by
  sorry

#eval cellDivision 5 3 2 10

end NUMINAMATH_CALUDE_cell_count_after_ten_days_l3952_395256


namespace NUMINAMATH_CALUDE_expression_equality_l3952_395216

theorem expression_equality : Real.sqrt 8 ^ (1/3) - |2 - Real.sqrt 3| + (1/2)^0 - Real.sqrt 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3952_395216


namespace NUMINAMATH_CALUDE_weight_estimate_error_l3952_395279

/-- The weight of a disk with precise 1-meter diameter in kg -/
def precise_disk_weight : ℝ := 100

/-- The radius of a disk in meters -/
def disk_radius : ℝ := 0.5

/-- The standard deviation of the disk radius in meters -/
def radius_std_dev : ℝ := 0.01

/-- The number of disks in the stack -/
def num_disks : ℕ := 100

/-- The expected weight of a single disk with variable radius -/
noncomputable def expected_disk_weight : ℝ := sorry

/-- The error in the weight estimate -/
theorem weight_estimate_error :
  num_disks * expected_disk_weight - (num_disks : ℝ) * precise_disk_weight = 4 := by sorry

end NUMINAMATH_CALUDE_weight_estimate_error_l3952_395279


namespace NUMINAMATH_CALUDE_fruit_trees_space_l3952_395247

/-- The total space needed for fruit trees in Quinton's yard -/
theorem fruit_trees_space (apple_width peach_width : ℕ) 
  (apple_space peach_space : ℕ) (apple_count peach_count : ℕ) : 
  apple_width = 10 → 
  peach_width = 12 → 
  apple_space = 12 → 
  peach_space = 15 → 
  apple_count = 2 → 
  peach_count = 2 → 
  (apple_count * apple_width + apple_space) + 
  (peach_count * peach_width + peach_space) = 71 := by
sorry

end NUMINAMATH_CALUDE_fruit_trees_space_l3952_395247


namespace NUMINAMATH_CALUDE_bake_sale_donation_ratio_is_one_to_one_l3952_395264

/-- Represents the financial details of Andrew's bake sale fundraiser. -/
structure BakeSale where
  total_earnings : ℕ
  ingredient_cost : ℕ
  personal_donation : ℕ
  total_homeless_donation : ℕ

/-- Calculates the ratio of homeless shelter donation to food bank donation. -/
def donation_ratio (sale : BakeSale) : ℚ :=
  let available_for_donation := sale.total_earnings - sale.ingredient_cost
  let homeless_donation := sale.total_homeless_donation - sale.personal_donation
  let food_bank_donation := available_for_donation - homeless_donation
  homeless_donation / food_bank_donation

/-- Theorem stating that the donation ratio is 1:1 for the given bake sale. -/
theorem bake_sale_donation_ratio_is_one_to_one 
  (sale : BakeSale) 
  (h1 : sale.total_earnings = 400)
  (h2 : sale.ingredient_cost = 100)
  (h3 : sale.personal_donation = 10)
  (h4 : sale.total_homeless_donation = 160) : 
  donation_ratio sale = 1 := by
  sorry

end NUMINAMATH_CALUDE_bake_sale_donation_ratio_is_one_to_one_l3952_395264


namespace NUMINAMATH_CALUDE_geometric_series_terms_l3952_395286

theorem geometric_series_terms (r : ℝ) (sum : ℝ) (h_r : r = 1/4) (h_sum : sum = 40) :
  let a := sum * (1 - r)
  (a * r = 7.5) ∧ (a * r^2 = 1.875) := by
sorry

end NUMINAMATH_CALUDE_geometric_series_terms_l3952_395286


namespace NUMINAMATH_CALUDE_s_square_sum_l3952_395218

/-- The sequence s_n is defined by the power series expansion of 1 / (1 - 2x - x^2) -/
noncomputable def s : ℕ → ℝ := sorry

/-- The power series expansion of 1 / (1 - 2x - x^2) -/
axiom power_series_expansion (x : ℝ) (h : x ≠ 0) : 
  (1 : ℝ) / (1 - 2*x - x^2) = ∑' (n : ℕ), s n * x^n

/-- The main theorem: s_n^2 + s_{n+1}^2 = s_{2n+2} for all non-negative integers n -/
theorem s_square_sum (n : ℕ) : (s n)^2 + (s (n+1))^2 = s (2*n+2) := by sorry

end NUMINAMATH_CALUDE_s_square_sum_l3952_395218


namespace NUMINAMATH_CALUDE_negation_of_existence_inequality_l3952_395267

theorem negation_of_existence_inequality : 
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_inequality_l3952_395267


namespace NUMINAMATH_CALUDE_expected_elderly_in_sample_l3952_395296

/-- Calculates the expected number of elderly individuals in a stratified sample -/
def expectedElderlyInSample (totalPopulation : ℕ) (elderlyPopulation : ℕ) (sampleSize : ℕ) : ℕ :=
  (elderlyPopulation * sampleSize) / totalPopulation

/-- Theorem: Expected number of elderly individuals in the sample -/
theorem expected_elderly_in_sample :
  expectedElderlyInSample 165 22 15 = 2 := by
  sorry

end NUMINAMATH_CALUDE_expected_elderly_in_sample_l3952_395296
