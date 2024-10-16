import Mathlib

namespace NUMINAMATH_CALUDE_keaton_orange_earnings_l2512_251252

/-- Represents Keaton's farm earnings -/
structure FarmEarnings where
  months_between_orange_harvests : ℕ
  months_between_apple_harvests : ℕ
  apple_harvest_earnings : ℕ
  total_annual_earnings : ℕ

/-- Calculates the earnings from each orange harvest -/
def orange_harvest_earnings (farm : FarmEarnings) : ℕ :=
  let orange_harvests_per_year := 12 / farm.months_between_orange_harvests
  let apple_harvests_per_year := 12 / farm.months_between_apple_harvests
  let annual_apple_earnings := apple_harvests_per_year * farm.apple_harvest_earnings
  let annual_orange_earnings := farm.total_annual_earnings - annual_apple_earnings
  annual_orange_earnings / orange_harvests_per_year

/-- Theorem stating that Keaton's orange harvest earnings are $50 -/
theorem keaton_orange_earnings :
  let farm := FarmEarnings.mk 2 3 30 420
  orange_harvest_earnings farm = 50 := by
  sorry

end NUMINAMATH_CALUDE_keaton_orange_earnings_l2512_251252


namespace NUMINAMATH_CALUDE_zero_not_in_range_of_g_l2512_251254

noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then Int.ceil (2 / (x + 3))
  else if x < -3 then Int.floor (2 / (x + 3))
  else 0  -- Arbitrary value for x = -3, as g is not defined there

theorem zero_not_in_range_of_g :
  ∀ x : ℝ, x ≠ -3 → g x ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_zero_not_in_range_of_g_l2512_251254


namespace NUMINAMATH_CALUDE_election_majority_l2512_251261

theorem election_majority (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 400 →
  winning_percentage = 70 / 100 →
  (winning_percentage * total_votes : ℚ).floor - ((1 - winning_percentage) * total_votes : ℚ).floor = 160 := by
sorry

end NUMINAMATH_CALUDE_election_majority_l2512_251261


namespace NUMINAMATH_CALUDE_parabola_directrix_l2512_251223

/-- Given a parabola with equation y = 4x^2 - 6, its directrix has equation y = -97/16 -/
theorem parabola_directrix (x y : ℝ) :
  y = 4 * x^2 - 6 → ∃ (k : ℝ), k = -97/16 ∧ (∀ (x₀ y₀ : ℝ), y₀ = 4 * x₀^2 - 6 → y₀ - k = (x₀ - 0)^2 + (y₀ - (k + 1/4))^2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2512_251223


namespace NUMINAMATH_CALUDE_journey_results_correct_l2512_251224

/-- Truck's journey between Town A and Village B -/
structure TruckJourney where
  uphill_distance : ℝ
  downhill_distance : ℝ
  flat_distance : ℝ
  round_trip_time_diff : ℝ
  uphill_speed_ratio : ℝ
  downhill_speed_ratio : ℝ
  flat_speed_ratio : ℝ

/-- Calculated speeds and times for the journey -/
structure JourneyResults where
  uphill_speed : ℝ
  downhill_speed : ℝ
  flat_speed : ℝ
  time_a_to_b : ℝ
  time_b_to_a : ℝ

/-- Theorem stating the correctness of the calculated results -/
theorem journey_results_correct (j : TruckJourney)
  (res : JourneyResults)
  (h1 : j.uphill_distance = 20)
  (h2 : j.downhill_distance = 14)
  (h3 : j.flat_distance = 5)
  (h4 : j.round_trip_time_diff = 1/6)
  (h5 : j.uphill_speed_ratio = 3)
  (h6 : j.downhill_speed_ratio = 6)
  (h7 : j.flat_speed_ratio = 5)
  (h8 : res.uphill_speed = 18)
  (h9 : res.downhill_speed = 36)
  (h10 : res.flat_speed = 30)
  (h11 : res.time_a_to_b = 5/3)
  (h12 : res.time_b_to_a = 3/2) :
  (j.uphill_distance / res.uphill_speed +
   j.downhill_distance / res.downhill_speed +
   j.flat_distance / res.flat_speed) -
  (j.uphill_distance / res.downhill_speed +
   j.downhill_distance / res.uphill_speed +
   j.flat_distance / res.flat_speed) = j.round_trip_time_diff ∧
  res.time_a_to_b =
    j.uphill_distance / res.uphill_speed +
    j.downhill_distance / res.downhill_speed +
    j.flat_distance / res.flat_speed ∧
  res.time_b_to_a =
    j.uphill_distance / res.downhill_speed +
    j.downhill_distance / res.uphill_speed +
    j.flat_distance / res.flat_speed ∧
  res.uphill_speed / res.downhill_speed = j.uphill_speed_ratio / j.downhill_speed_ratio ∧
  res.downhill_speed / res.flat_speed = j.downhill_speed_ratio / j.flat_speed_ratio :=
by sorry


end NUMINAMATH_CALUDE_journey_results_correct_l2512_251224


namespace NUMINAMATH_CALUDE_bicycle_price_increase_l2512_251208

theorem bicycle_price_increase (initial_price : ℝ) (first_increase : ℝ) (second_increase : ℝ) :
  initial_price = 220 →
  first_increase = 0.08 →
  second_increase = 0.10 →
  let price_after_first := initial_price * (1 + first_increase)
  let final_price := price_after_first * (1 + second_increase)
  final_price = 261.36 := by
sorry

end NUMINAMATH_CALUDE_bicycle_price_increase_l2512_251208


namespace NUMINAMATH_CALUDE_f_is_even_and_has_zero_point_l2512_251258

-- Define the function f(x) = x^2 - 1
def f (x : ℝ) : ℝ := x^2 - 1

-- Theorem stating that f is an even function and has a zero point
theorem f_is_even_and_has_zero_point :
  (∀ x : ℝ, f (-x) = f x) ∧ (∃ x : ℝ, f x = 0) := by
  sorry

end NUMINAMATH_CALUDE_f_is_even_and_has_zero_point_l2512_251258


namespace NUMINAMATH_CALUDE_palindrome_difference_l2512_251266

def is_palindrome (n : ℕ) : Prop :=
  ∃ (d : List ℕ), n = d.foldl (λ acc x => acc * 10 + x) 0 ∧ d = d.reverse

def has_9_digits (n : ℕ) : Prop :=
  999999999 ≥ n ∧ n ≥ 100000000

def starts_with_nonzero (n : ℕ) : Prop :=
  n ≥ 100000000

def consecutive_palindromes (m n : ℕ) : Prop :=
  is_palindrome m ∧ is_palindrome n ∧ n > m ∧
  ∀ k, m < k ∧ k < n → ¬is_palindrome k

theorem palindrome_difference (m n : ℕ) :
  has_9_digits m ∧ has_9_digits n ∧
  starts_with_nonzero m ∧ starts_with_nonzero n ∧
  consecutive_palindromes m n →
  n - m = 100000011 := by sorry

end NUMINAMATH_CALUDE_palindrome_difference_l2512_251266


namespace NUMINAMATH_CALUDE_jasons_books_l2512_251213

/-- Given that Keith has 20 books and together with Jason they have 41 books,
    prove that Jason has 21 books. -/
theorem jasons_books (keith_books : ℕ) (total_books : ℕ) (h1 : keith_books = 20) (h2 : total_books = 41) :
  total_books - keith_books = 21 := by
  sorry

end NUMINAMATH_CALUDE_jasons_books_l2512_251213


namespace NUMINAMATH_CALUDE_quadratic_polynomial_satisfies_conditions_l2512_251226

theorem quadratic_polynomial_satisfies_conditions :
  ∃ (q : ℝ → ℝ),
    (∀ x, q x = 2.5 * x^2 - 5.5 * x + 13) ∧
    q (-1) = 10 ∧
    q 2 = 1 ∧
    q 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_satisfies_conditions_l2512_251226


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l2512_251294

def A : Set ℝ := {x | x - 1 ≥ 0}
def B : Set ℝ := {x | |x| ≤ 2}

theorem set_intersection_theorem : A ∩ B = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l2512_251294


namespace NUMINAMATH_CALUDE_sophomore_count_l2512_251225

theorem sophomore_count (total : ℕ) (sophomore_percent : ℚ) (senior_percent : ℚ)
  (h_total : total = 50)
  (h_sophomore_percent : sophomore_percent = 1/5)
  (h_senior_percent : senior_percent = 1/4)
  (h_team_equal : ∃ (team_size : ℕ), 
    sophomore_percent * (total - seniors) = ↑team_size ∧
    senior_percent * seniors = ↑team_size)
  (seniors : ℕ) :
  total - seniors = 22 :=
sorry

end NUMINAMATH_CALUDE_sophomore_count_l2512_251225


namespace NUMINAMATH_CALUDE_divide_c_by_a_l2512_251291

theorem divide_c_by_a (a b c : ℝ) (h1 : a * b = 3) (h2 : b * c = 8/5) : c / a = 8/15 := by
  sorry

end NUMINAMATH_CALUDE_divide_c_by_a_l2512_251291


namespace NUMINAMATH_CALUDE_adjacent_sum_theorem_l2512_251215

/-- Represents a 3x3 table with numbers from 1 to 9 -/
def Table := Fin 3 → Fin 3 → Fin 9

/-- Checks if a table contains each number from 1 to 9 exactly once -/
def isValidTable (t : Table) : Prop :=
  ∀ n : Fin 9, ∃! (i j : Fin 3), t i j = n

/-- Checks if the table has 1, 2, 3, and 4 in the correct positions -/
def hasCorrectCorners (t : Table) : Prop :=
  t 0 0 = 0 ∧ t 2 0 = 1 ∧ t 0 2 = 2 ∧ t 2 2 = 3

/-- Returns the sum of adjacent numbers to the given position -/
def adjacentSum (t : Table) (i j : Fin 3) : Nat :=
  (if i > 0 then (t (i-1) j).val + 1 else 0) +
  (if i < 2 then (t (i+1) j).val + 1 else 0) +
  (if j > 0 then (t i (j-1)).val + 1 else 0) +
  (if j < 2 then (t i (j+1)).val + 1 else 0)

/-- The main theorem to prove -/
theorem adjacent_sum_theorem (t : Table) 
  (valid : isValidTable t) 
  (corners : hasCorrectCorners t) 
  (sum_5 : ∃ i j : Fin 3, t i j = 4 ∧ adjacentSum t i j = 9) :
  ∃ i j : Fin 3, t i j = 5 ∧ adjacentSum t i j = 29 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_sum_theorem_l2512_251215


namespace NUMINAMATH_CALUDE_competition_scores_l2512_251233

theorem competition_scores (n k : ℕ) : n ≥ 2 ∧ k ≥ 1 →
  (k * n * (n + 1) = 52 * n * (n - 1)) ↔ 
  ((n = 25 ∧ k = 2) ∨ (n = 12 ∧ k = 4) ∨ (n = 3 ∧ k = 13)) := by
sorry

end NUMINAMATH_CALUDE_competition_scores_l2512_251233


namespace NUMINAMATH_CALUDE_brady_dwayne_earnings_difference_l2512_251230

/-- Given that Dwayne makes $1,500 in a year and Brady and Dwayne's combined earnings are $3,450 in a year,
    prove that Brady makes $450 more than Dwayne in a year. -/
theorem brady_dwayne_earnings_difference :
  let dwayne_earnings : ℕ := 1500
  let combined_earnings : ℕ := 3450
  let brady_earnings : ℕ := combined_earnings - dwayne_earnings
  brady_earnings - dwayne_earnings = 450 := by
sorry

end NUMINAMATH_CALUDE_brady_dwayne_earnings_difference_l2512_251230


namespace NUMINAMATH_CALUDE_nail_polish_count_l2512_251216

theorem nail_polish_count (kim heidi karen : ℕ) : 
  kim = 12 →
  heidi = kim + 5 →
  karen = kim - 4 →
  heidi + karen = 25 := by sorry

end NUMINAMATH_CALUDE_nail_polish_count_l2512_251216


namespace NUMINAMATH_CALUDE_heather_final_blocks_l2512_251289

/-- Calculates the final number of blocks Heather has after receiving blocks from Jose. -/
def final_blocks (heather_start : Float) (jose_shared : Float) : Float :=
  heather_start + jose_shared

/-- Theorem stating that Heather ends with 127.0 blocks given the initial conditions. -/
theorem heather_final_blocks :
  final_blocks 86.0 41.0 = 127.0 := by
  sorry

end NUMINAMATH_CALUDE_heather_final_blocks_l2512_251289


namespace NUMINAMATH_CALUDE_intersection_equation_l2512_251255

theorem intersection_equation (a b : ℝ) (hb : b ≠ 0) :
  ∃ m n : ℤ, (m : ℝ)^3 - a*(m : ℝ)^2 - b*(m : ℝ) = a*(m : ℝ) + b ∧
             (m : ℝ)^3 - a*(m : ℝ)^2 - b*(m : ℝ) = (n : ℝ) →
  2*a - b + 8 = 0 := by
sorry

end NUMINAMATH_CALUDE_intersection_equation_l2512_251255


namespace NUMINAMATH_CALUDE_constant_function_m_values_l2512_251206

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m * x - |x + 1|

-- State the theorem
theorem constant_function_m_values
  (m : ℝ)
  (h_exists : ∃ (a b : ℝ), -2 ≤ a ∧ a < b ∧
    ∀ x, x ∈ Set.Icc a b → ∃ c, f m x = c) :
  m = 1 ∨ m = -1 := by
sorry

end NUMINAMATH_CALUDE_constant_function_m_values_l2512_251206


namespace NUMINAMATH_CALUDE_number_division_problem_l2512_251272

theorem number_division_problem (x y : ℝ) : 
  (x - 5) / y = 7 → (x - 24) / 10 = 3 → y = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l2512_251272


namespace NUMINAMATH_CALUDE_min_sum_of_distances_l2512_251241

/-- Represents a five-digit number -/
def FiveDigitNumber := Fin 100000

/-- The distance between two five-digit numbers -/
def distance (a b : FiveDigitNumber) : Nat :=
  sorry

/-- A permutation of all five-digit numbers -/
def Permutation := Equiv.Perm FiveDigitNumber

/-- The sum of distances between consecutive numbers in a permutation -/
def sumOfDistances (p : Permutation) : Nat :=
  sorry

/-- The minimum possible sum of distances between consecutive five-digit numbers -/
theorem min_sum_of_distances :
  ∃ (p : Permutation), sumOfDistances p = 101105 ∧
  ∀ (q : Permutation), sumOfDistances q ≥ 101105 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_distances_l2512_251241


namespace NUMINAMATH_CALUDE_apple_buying_difference_l2512_251283

theorem apple_buying_difference :
  ∀ (w : ℕ),
  (2 * 30 + 3 * w = 210) →
  (30 < w) →
  (w - 30 = 20) :=
by
  sorry

end NUMINAMATH_CALUDE_apple_buying_difference_l2512_251283


namespace NUMINAMATH_CALUDE_bert_pencil_usage_l2512_251271

/-- The number of words Bert writes to use up a pencil -/
def words_per_pencil (puzzles_per_day : ℕ) (days_per_pencil : ℕ) (words_per_puzzle : ℕ) : ℕ :=
  puzzles_per_day * days_per_pencil * words_per_puzzle

/-- Theorem stating that Bert writes 1050 words to use up a pencil -/
theorem bert_pencil_usage :
  words_per_pencil 1 14 75 = 1050 := by
  sorry

end NUMINAMATH_CALUDE_bert_pencil_usage_l2512_251271


namespace NUMINAMATH_CALUDE_intersection_max_k_l2512_251269

theorem intersection_max_k : 
  let f : ℝ → ℝ := fun x => Real.log x / x
  ∃ k_max : ℝ, k_max = 1 / Real.exp 1 ∧ 
    (∀ k : ℝ, (∃ x : ℝ, x > 0 ∧ k * x = Real.log x) → k ≤ k_max) :=
by sorry

end NUMINAMATH_CALUDE_intersection_max_k_l2512_251269


namespace NUMINAMATH_CALUDE_inequality_proof_l2512_251257

theorem inequality_proof (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) 
  (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) : 
  Real.sqrt (a * (1 - b) * (1 - c)) + Real.sqrt (b * (1 - a) * (1 - c)) + 
  Real.sqrt (c * (1 - a) * (1 - b)) ≤ 1 + Real.sqrt (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2512_251257


namespace NUMINAMATH_CALUDE_determinant_solution_l2512_251201

/-- Given a ≠ 0 and b ≠ 0, the solution to the determinant equation is (3b^2 + ab) / (a + b) -/
theorem determinant_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let x := (3 * b^2 + a * b) / (a + b)
  (x + a) * ((x + b) * (2 * x) - x * (2 * x)) -
  x * (x * (2 * x) - x * (2 * x + a + b)) +
  x * (x * (2 * x) - (x + b) * (2 * x + a + b)) = 0 := by
  sorry

#check determinant_solution

end NUMINAMATH_CALUDE_determinant_solution_l2512_251201


namespace NUMINAMATH_CALUDE_train_length_calculation_l2512_251288

/-- Calculates the length of a train given its speed and time to cross a post -/
theorem train_length_calculation (speed_km_hr : ℝ) (time_seconds : ℝ) : 
  speed_km_hr = 40 → time_seconds = 25.2 → 
  ∃ (length_meters : ℝ), abs (length_meters - 280.392) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l2512_251288


namespace NUMINAMATH_CALUDE_mothers_carrots_count_l2512_251243

/-- The number of carrots Nancy picked -/
def nancys_carrots : ℕ := 38

/-- The number of good carrots -/
def good_carrots : ℕ := 71

/-- The number of bad carrots -/
def bad_carrots : ℕ := 14

/-- The number of carrots Nancy's mother picked -/
def mothers_carrots : ℕ := (good_carrots + bad_carrots) - nancys_carrots

theorem mothers_carrots_count : mothers_carrots = 47 := by
  sorry

end NUMINAMATH_CALUDE_mothers_carrots_count_l2512_251243


namespace NUMINAMATH_CALUDE_min_socks_for_fifteen_pairs_l2512_251285

/-- Represents the number of socks of each color in the room -/
structure SockCollection where
  red : Nat
  green : Nat
  blue : Nat
  yellow : Nat
  black : Nat

/-- The minimum number of socks needed to guarantee a certain number of pairs -/
def minSocksForPairs (socks : SockCollection) (pairs : Nat) : Nat :=
  5 + 5 * 2 * (pairs - 1) + 1

/-- Theorem stating the minimum number of socks needed for 15 pairs -/
theorem min_socks_for_fifteen_pairs (socks : SockCollection)
    (h1 : socks.red = 120)
    (h2 : socks.green = 100)
    (h3 : socks.blue = 70)
    (h4 : socks.yellow = 50)
    (h5 : socks.black = 30) :
    minSocksForPairs socks 15 = 146 := by
  sorry

#eval minSocksForPairs { red := 120, green := 100, blue := 70, yellow := 50, black := 30 } 15

end NUMINAMATH_CALUDE_min_socks_for_fifteen_pairs_l2512_251285


namespace NUMINAMATH_CALUDE_rotated_angle_measure_l2512_251249

/-- Given an angle of 60 degrees rotated 600 degrees clockwise, 
    the resulting acute angle measure is 60 degrees. -/
theorem rotated_angle_measure (initial_angle rotation : ℝ) : 
  initial_angle = 60 → 
  rotation = 600 → 
  (initial_angle - (rotation % 360)) % 360 = 60 := by
  sorry

end NUMINAMATH_CALUDE_rotated_angle_measure_l2512_251249


namespace NUMINAMATH_CALUDE_function_value_at_four_l2512_251250

/-- Given a function f: ℝ → ℝ satisfying f(x) + 2f(1 - x) = 3x^2 for all x,
    prove that f(4) = 2 -/
theorem function_value_at_four (f : ℝ → ℝ) 
    (h : ∀ x : ℝ, f x + 2 * f (1 - x) = 3 * x^2) : 
    f 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_four_l2512_251250


namespace NUMINAMATH_CALUDE_sum_of_fourth_and_fifth_terms_l2512_251205

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r ^ (n - 1)

theorem sum_of_fourth_and_fifth_terms :
  let a₁ : ℝ := 4096
  let r : ℝ := 1/4
  (geometric_sequence a₁ r 4) + (geometric_sequence a₁ r 5) = 80 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fourth_and_fifth_terms_l2512_251205


namespace NUMINAMATH_CALUDE_sum_of_cubes_equation_l2512_251268

theorem sum_of_cubes_equation (x y : ℝ) :
  x^3 + 21*x*y + y^3 = 343 → x + y = 7 ∨ x + y = -14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_equation_l2512_251268


namespace NUMINAMATH_CALUDE_M_intersect_N_empty_l2512_251280

/-- Set M in the complex plane --/
def M : Set ℂ :=
  {z | ∃ t : ℝ, t ≠ -1 ∧ t ≠ 0 ∧ z = t / (1 + t) + Complex.I * (1 + t) / t}

/-- Set N in the complex plane --/
def N : Set ℂ :=
  {z | ∃ t : ℝ, |t| ≤ 1 ∧ z = Real.sqrt 2 * (Complex.cos (Real.arcsin t) + Complex.I * Complex.cos (Real.arccos t))}

/-- The intersection of sets M and N is empty --/
theorem M_intersect_N_empty : M ∩ N = ∅ := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_empty_l2512_251280


namespace NUMINAMATH_CALUDE_sand_delivery_theorem_l2512_251212

/-- The amount of sand remaining after a truck's journey -/
def sand_remaining (initial : Real) (loss : Real) : Real :=
  initial - loss

/-- The total amount of sand from all trucks -/
def total_sand (truck1 : Real) (truck2 : Real) (truck3 : Real) : Real :=
  truck1 + truck2 + truck3

theorem sand_delivery_theorem :
  let truck1_initial : Real := 4.1
  let truck1_loss : Real := 2.4
  let truck2_initial : Real := 5.7
  let truck2_loss : Real := 3.6
  let truck3_initial : Real := 8.2
  let truck3_loss : Real := 1.9
  total_sand
    (sand_remaining truck1_initial truck1_loss)
    (sand_remaining truck2_initial truck2_loss)
    (sand_remaining truck3_initial truck3_loss) = 10.1 := by
  sorry

end NUMINAMATH_CALUDE_sand_delivery_theorem_l2512_251212


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2512_251217

theorem fraction_to_decimal : 19 / (2^2 * 5^3) = 0.095 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2512_251217


namespace NUMINAMATH_CALUDE_smallest_n_not_prime_l2512_251207

theorem smallest_n_not_prime (n : ℕ) : 
  (∀ k < n, Nat.Prime (2^k + 1)) ∧ ¬(Nat.Prime (2^n + 1)) ↔ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_not_prime_l2512_251207


namespace NUMINAMATH_CALUDE_episode_length_l2512_251214

/-- Given a TV mini series with 6 episodes and a total watching time of 5 hours,
    prove that the length of each episode is 50 minutes. -/
theorem episode_length (num_episodes : ℕ) (total_time : ℕ) : 
  num_episodes = 6 → total_time = 5 * 60 → total_time / num_episodes = 50 := by
  sorry

end NUMINAMATH_CALUDE_episode_length_l2512_251214


namespace NUMINAMATH_CALUDE_final_number_not_zero_l2512_251277

/-- Represents the operation of replacing two numbers with their sum or difference -/
inductive Operation
  | Sum : ℕ → ℕ → Operation
  | Difference : ℕ → ℕ → Operation

/-- The type representing the state of the blackboard -/
def Blackboard := List ℕ

/-- Applies an operation to the blackboard -/
def applyOperation (board : Blackboard) (op : Operation) : Blackboard :=
  match op with
  | Operation.Sum a b => sorry
  | Operation.Difference a b => sorry

/-- Represents a sequence of operations -/
def OperationSequence := List Operation

/-- Applies a sequence of operations to the initial blackboard -/
def applyOperations (initialBoard : Blackboard) (ops : OperationSequence) : Blackboard :=
  ops.foldl applyOperation initialBoard

/-- The initial state of the blackboard -/
def initialBoard : Blackboard := List.range 1974

theorem final_number_not_zero (ops : OperationSequence) :
  (applyOperations initialBoard ops).length = 1 →
  (applyOperations initialBoard ops).head? ≠ some 0 := by
  sorry

end NUMINAMATH_CALUDE_final_number_not_zero_l2512_251277


namespace NUMINAMATH_CALUDE_symmetric_point_example_l2512_251295

/-- Given a point P in a Cartesian coordinate system, this function returns its symmetric point with respect to the origin. -/
def symmetricPoint (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

/-- Theorem stating that the symmetric point of (2, -3) with respect to the origin is (-2, 3). -/
theorem symmetric_point_example : symmetricPoint (2, -3) = (-2, 3) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_example_l2512_251295


namespace NUMINAMATH_CALUDE_set_operations_l2512_251222

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def A : Set Nat := {1, 3, 5, 7}
def B : Set Nat := {3, 5}

theorem set_operations :
  (A ∪ B = {1, 3, 5, 7}) ∧
  ((U \ A) ∪ B = {2, 3, 4, 5, 6}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l2512_251222


namespace NUMINAMATH_CALUDE_digit_sum_to_100_l2512_251299

def digits : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9]

def insert_operators (ds : List Nat) : List (Option Bool) :=
  [none, some true, some true, some false, some false, some false, some false, some true, some false]

def evaluate (ds : List Nat) (ops : List (Option Bool)) : Int :=
  match ds, ops with
  | [], _ => 0
  | d :: ds', none :: ops' => d * 100 + evaluate ds' ops'
  | d :: ds', some true :: ops' => d + evaluate ds' ops'
  | d :: ds', some false :: ops' => -d + evaluate ds' ops'
  | _, _ => 0

theorem digit_sum_to_100 :
  ∃ (ops : List (Option Bool)), evaluate digits ops = 100 :=
sorry

end NUMINAMATH_CALUDE_digit_sum_to_100_l2512_251299


namespace NUMINAMATH_CALUDE_waiter_tip_calculation_l2512_251270

theorem waiter_tip_calculation (total_customers : ℕ) (non_tipping_customers : ℕ) (total_tip : ℚ) :
  total_customers = 7 →
  non_tipping_customers = 5 →
  total_tip = 6 →
  (total_tip / (total_customers - non_tipping_customers) : ℚ) = 3 :=
by sorry

end NUMINAMATH_CALUDE_waiter_tip_calculation_l2512_251270


namespace NUMINAMATH_CALUDE_sports_books_count_l2512_251282

theorem sports_books_count (total_books : ℕ) (school_books : ℕ) (sports_books : ℕ) 
  (h1 : total_books = 344)
  (h2 : school_books = 136)
  (h3 : total_books = school_books + sports_books) :
  sports_books = 208 := by
sorry

end NUMINAMATH_CALUDE_sports_books_count_l2512_251282


namespace NUMINAMATH_CALUDE_parallelogram_base_l2512_251265

/-- The area of a parallelogram -/
def area_parallelogram (base height : ℝ) : ℝ := base * height

/-- Theorem: Given a parallelogram with height 36 cm and area 1728 cm², its base is 48 cm -/
theorem parallelogram_base (height area : ℝ) (h1 : height = 36) (h2 : area = 1728) :
  ∃ base : ℝ, area_parallelogram base height = area ∧ base = 48 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_l2512_251265


namespace NUMINAMATH_CALUDE_other_number_is_twenty_l2512_251202

theorem other_number_is_twenty (a b : ℤ) : 
  3 * a + 4 * b = 140 → (a = 20 ∨ b = 20) → (a = 20 ∧ b = 20) :=
by sorry

end NUMINAMATH_CALUDE_other_number_is_twenty_l2512_251202


namespace NUMINAMATH_CALUDE_log_sum_simplification_l2512_251237

theorem log_sum_simplification :
  1 / (Real.log 3 / Real.log 12 + 2) +
  1 / (Real.log 2 / Real.log 8 + 2) +
  1 / (Real.log 3 / Real.log 9 + 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_simplification_l2512_251237


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2512_251274

theorem quadratic_inequality_solution (x : ℝ) :
  -10 * x^2 + 6 * x + 8 < 0 ↔ -0.64335 < x ∧ x < 1.24335 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2512_251274


namespace NUMINAMATH_CALUDE_face_mask_profit_l2512_251287

/-- Calculates the total profit from selling face masks given the specified conditions. -/
theorem face_mask_profit :
  let num_boxes : ℕ := 3
  let discount_rate : ℚ := 1/5
  let original_price : ℚ := 8
  let masks_per_box : List ℕ := [25, 30, 35]
  let selling_price : ℚ := 3/5

  let discounted_price := original_price * (1 - discount_rate)
  let total_cost := num_boxes * discounted_price
  let total_masks := masks_per_box.sum
  let total_revenue := total_masks * selling_price
  let profit := total_revenue - total_cost

  profit = 348/10 :=
by sorry

end NUMINAMATH_CALUDE_face_mask_profit_l2512_251287


namespace NUMINAMATH_CALUDE_fourth_root_equation_solutions_l2512_251264

theorem fourth_root_equation_solutions :
  {x : ℝ | x > 0 ∧ (x^(1/4) = 20 / (9 - x^(1/4)))} = {256, 625} := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solutions_l2512_251264


namespace NUMINAMATH_CALUDE_ratio_equality_l2512_251220

theorem ratio_equality (x y z : ℝ) (h : x / 3 = y / 4 ∧ y / 4 = z / 5) :
  (2 * x + y - z) / (3 * x - 2 * y + z) = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l2512_251220


namespace NUMINAMATH_CALUDE_ninth_grade_maximizes_profit_l2512_251246

/-- Represents the profit function for a product with different quality grades. -/
def profit_function (k : ℕ) : ℝ :=
  let profit_per_piece := 8 + 2 * (k - 1)
  let pieces_produced := 60 - 3 * (k - 1)
  (profit_per_piece * pieces_produced : ℝ)

/-- Theorem stating that the 9th quality grade maximizes the profit. -/
theorem ninth_grade_maximizes_profit :
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → profit_function k ≤ profit_function 9 := by
  sorry

#check ninth_grade_maximizes_profit

end NUMINAMATH_CALUDE_ninth_grade_maximizes_profit_l2512_251246


namespace NUMINAMATH_CALUDE_betty_picked_15_oranges_l2512_251260

def orange_problem (betty_oranges : ℕ) : Prop :=
  let bill_oranges : ℕ := 12
  let frank_oranges : ℕ := 3 * (betty_oranges + bill_oranges)
  let seeds_planted : ℕ := 2 * frank_oranges
  let trees_grown : ℕ := seeds_planted
  let oranges_per_tree : ℕ := 5
  let total_oranges : ℕ := trees_grown * oranges_per_tree
  total_oranges = 810

theorem betty_picked_15_oranges :
  ∃ (betty_oranges : ℕ), orange_problem betty_oranges ∧ betty_oranges = 15 :=
by sorry

end NUMINAMATH_CALUDE_betty_picked_15_oranges_l2512_251260


namespace NUMINAMATH_CALUDE_cucumbers_for_apples_l2512_251210

-- Define the cost relationships
def apple_banana_ratio : ℚ := 10 / 5
def banana_cucumber_ratio : ℚ := 3 / 4

-- Define the number of apples we're interested in
def apples_of_interest : ℚ := 20

-- Theorem to prove
theorem cucumbers_for_apples :
  let bananas_for_apples : ℚ := apples_of_interest / apple_banana_ratio
  let cucumbers_for_bananas : ℚ := bananas_for_apples * (1 / banana_cucumber_ratio)
  cucumbers_for_bananas = 40 / 3 :=
by sorry

end NUMINAMATH_CALUDE_cucumbers_for_apples_l2512_251210


namespace NUMINAMATH_CALUDE_two_face_cubes_5x5x5_l2512_251296

/-- The number of unit cubes with exactly two faces on the surface of a 5x5x5 cube -/
def two_face_cubes (n : ℕ) : ℕ := 12 * (n - 2)

/-- Theorem stating that the number of unit cubes with exactly two faces
    on the surface of a 5x5x5 cube is 36 -/
theorem two_face_cubes_5x5x5 :
  two_face_cubes 5 = 36 := by
  sorry

end NUMINAMATH_CALUDE_two_face_cubes_5x5x5_l2512_251296


namespace NUMINAMATH_CALUDE_stock_price_calculation_l2512_251235

/-- Calculates the price of a stock given investment details -/
theorem stock_price_calculation 
  (investment : ℝ) 
  (dividend_rate : ℝ) 
  (annual_income : ℝ) 
  (face_value : ℝ) 
  (h1 : investment = 6800)
  (h2 : dividend_rate = 0.20)
  (h3 : annual_income = 1000)
  (h4 : face_value = 100) : 
  (investment / (annual_income / dividend_rate)) * face_value = 136 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_calculation_l2512_251235


namespace NUMINAMATH_CALUDE_garage_to_other_rooms_ratio_l2512_251279

/-- Given the number of bulbs needed for other rooms and the total number of bulbs Sean has,
    prove that the ratio of garage bulbs to other room bulbs is 1:2. -/
theorem garage_to_other_rooms_ratio
  (other_rooms_bulbs : ℕ)
  (total_packs : ℕ)
  (bulbs_per_pack : ℕ)
  (h1 : other_rooms_bulbs = 8)
  (h2 : total_packs = 6)
  (h3 : bulbs_per_pack = 2) :
  (total_packs * bulbs_per_pack - other_rooms_bulbs) / other_rooms_bulbs = 1 / 2 := by
  sorry

#check garage_to_other_rooms_ratio

end NUMINAMATH_CALUDE_garage_to_other_rooms_ratio_l2512_251279


namespace NUMINAMATH_CALUDE_total_pages_called_l2512_251211

def pages_last_week : ℝ := 10.2
def pages_this_week : ℝ := 8.6

theorem total_pages_called :
  pages_last_week + pages_this_week = 18.8 := by
  sorry

end NUMINAMATH_CALUDE_total_pages_called_l2512_251211


namespace NUMINAMATH_CALUDE_rectangle_area_l2512_251247

theorem rectangle_area (b : ℝ) : 
  let square_area : ℝ := 2500
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := (2 / 5) * circle_radius
  let rectangle_breadth : ℝ := b
  let rectangle_area : ℝ := rectangle_length * rectangle_breadth
  rectangle_area = 20 * b := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l2512_251247


namespace NUMINAMATH_CALUDE_problem_solution_l2512_251219

theorem problem_solution :
  ∀ (m x : ℝ),
    (m = 1 → (((x - 3*m) * (x - m) < 0 ∧ |x - 3| ≤ 1) ↔ (2 ≤ x ∧ x < 3))) ∧
    (m > 0 → ((∀ x, |x - 3| ≤ 1 → (x - 3*m) * (x - m) < 0) ∧
              (∃ x, (x - 3*m) * (x - m) < 0 ∧ |x - 3| > 1)) ↔
             (4/3 < m ∧ m < 2)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2512_251219


namespace NUMINAMATH_CALUDE_tan_30_squared_plus_sin_45_squared_l2512_251284

theorem tan_30_squared_plus_sin_45_squared : 
  (Real.tan (30 * π / 180))^2 + (Real.sin (45 * π / 180))^2 = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_tan_30_squared_plus_sin_45_squared_l2512_251284


namespace NUMINAMATH_CALUDE_tens_digit_of_1047_pow_1024_minus_1049_l2512_251245

theorem tens_digit_of_1047_pow_1024_minus_1049 : ∃ n : ℕ, (1047^1024 - 1049) % 100 = 32 ∧ n * 10 + 3 = (1047^1024 - 1049) / 10 % 10 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_1047_pow_1024_minus_1049_l2512_251245


namespace NUMINAMATH_CALUDE_circle_cut_and_reform_l2512_251286

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

-- Define a point inside a circle
def PointInside (c : Circle) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 < c.radius^2

-- Define the theorem
theorem circle_cut_and_reform (c : Circle) (a : ℝ × ℝ) (h : PointInside c a) :
  ∃ (part1 part2 : Set (ℝ × ℝ)), 
    (part1 ∪ part2 = {p | PointInside c p}) ∧
    (∃ (new_circle : Circle), new_circle.center = a ∧
      part1 ∪ part2 = {p | PointInside new_circle p}) :=
sorry

end NUMINAMATH_CALUDE_circle_cut_and_reform_l2512_251286


namespace NUMINAMATH_CALUDE_students_play_both_calculation_l2512_251209

/-- Represents the number of students who play both football and cricket -/
def students_play_both (total students_football students_cricket students_neither : ℕ) : ℕ :=
  students_football + students_cricket - (total - students_neither)

theorem students_play_both_calculation :
  students_play_both 450 325 175 50 = 100 := by
  sorry

end NUMINAMATH_CALUDE_students_play_both_calculation_l2512_251209


namespace NUMINAMATH_CALUDE_nicole_bought_23_candies_l2512_251251

def nicole_candies (x : ℕ) : Prop :=
  ∃ (y : ℕ), 
    (2 * x) / 3 = y + 5 + 10 ∧ 
    y ≥ 0 ∧
    x > 0

theorem nicole_bought_23_candies : 
  ∃ (x : ℕ), nicole_candies x ∧ x = 23 := by sorry

end NUMINAMATH_CALUDE_nicole_bought_23_candies_l2512_251251


namespace NUMINAMATH_CALUDE_lily_shopping_exceeds_budget_l2512_251234

/-- Proves that the total cost of items exceeds Lily's initial amount --/
theorem lily_shopping_exceeds_budget :
  let initial_amount : ℝ := 70
  let celery_price : ℝ := 8 * (1 - 0.2)
  let cereal_price : ℝ := 14
  let bread_price : ℝ := 10 * (1 - 0.05)
  let milk_price : ℝ := 12 * (1 - 0.15)
  let potato_price : ℝ := 2 * 8
  let cookie_price : ℝ := 15
  let tax_rate : ℝ := 0.07
  let total_cost : ℝ := (celery_price + cereal_price + bread_price + milk_price + potato_price + cookie_price) * (1 + tax_rate)
  total_cost > initial_amount := by sorry

end NUMINAMATH_CALUDE_lily_shopping_exceeds_budget_l2512_251234


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l2512_251276

theorem largest_integer_with_remainder (n : ℕ) : 
  n < 61 ∧ n % 6 = 5 ∧ ∀ m : ℕ, m < 61 ∧ m % 6 = 5 → m ≤ n → n = 59 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l2512_251276


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l2512_251248

theorem inequality_and_equality_condition (a b : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b < 2) :
  (1 / (1 + a^2) + 1 / (1 + b^2) ≤ 2 / (1 + a*b)) ∧
  (1 / (1 + a^2) + 1 / (1 + b^2) = 2 / (1 + a*b) ↔ a = b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l2512_251248


namespace NUMINAMATH_CALUDE_gcd_21n_plus_4_14n_plus_3_gcd_factorial_plus_one_gcd_F_m_F_n_l2512_251236

-- Problem 1
theorem gcd_21n_plus_4_14n_plus_3 (n : ℕ+) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by sorry

-- Problem 2
theorem gcd_factorial_plus_one (n : ℕ) : Nat.gcd (Nat.factorial n + 1) (Nat.factorial (n + 1) + 1) = 1 := by sorry

-- Problem 3
def F (k : ℕ) : ℕ := 2^(2^k) + 1

theorem gcd_F_m_F_n (m n : ℕ) (h : m ≠ n) : Nat.gcd (F m) (F n) = 1 := by sorry

end NUMINAMATH_CALUDE_gcd_21n_plus_4_14n_plus_3_gcd_factorial_plus_one_gcd_F_m_F_n_l2512_251236


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2512_251275

-- Define the sets P and Q
def P : Set ℝ := {x | x < 1}
def Q : Set ℝ := {x | x^2 < 4}

-- State the theorem
theorem intersection_of_P_and_Q : P ∩ Q = {x : ℝ | -2 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2512_251275


namespace NUMINAMATH_CALUDE_min_value_x_plus_two_over_x_l2512_251244

theorem min_value_x_plus_two_over_x (x : ℝ) (h : x > 0) :
  x + 2 / x ≥ 2 * Real.sqrt 2 ∧
  (x + 2 / x = 2 * Real.sqrt 2 ↔ x = Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_two_over_x_l2512_251244


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2512_251232

theorem quadratic_equation_solution :
  ∀ y : ℂ, 4 + 3 * y^2 = 0.7 * y - 40 ↔ y = (0.1167 : ℝ) + (3.8273 : ℝ) * I ∨ y = (0.1167 : ℝ) - (3.8273 : ℝ) * I :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2512_251232


namespace NUMINAMATH_CALUDE_season_length_l2512_251227

def games_per_month : ℕ := 7
def games_in_season : ℕ := 14

theorem season_length :
  games_in_season / games_per_month = 2 :=
sorry

end NUMINAMATH_CALUDE_season_length_l2512_251227


namespace NUMINAMATH_CALUDE_equation_solution_l2512_251200

theorem equation_solution : ∃ x : ℝ, (9 / (x + 3 / 0.75) = 1) ∧ (x = 5) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2512_251200


namespace NUMINAMATH_CALUDE_factorization_cubic_quadratic_l2512_251267

theorem factorization_cubic_quadratic (a : ℝ) : a^3 - 2*a^2 = a^2*(a - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_cubic_quadratic_l2512_251267


namespace NUMINAMATH_CALUDE_chocolate_bars_count_l2512_251262

theorem chocolate_bars_count (small_boxes : ℕ) (bars_per_box : ℕ) 
  (h1 : small_boxes = 21) 
  (h2 : bars_per_box = 25) : 
  small_boxes * bars_per_box = 525 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_count_l2512_251262


namespace NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l2512_251240

theorem greatest_value_quadratic_inequality :
  ∃ (a_max : ℝ), a_max = 8 ∧
  (∀ a : ℝ, a^2 - 12*a + 32 ≤ 0 → a ≤ a_max) ∧
  (a_max^2 - 12*a_max + 32 ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l2512_251240


namespace NUMINAMATH_CALUDE_inscribed_sphere_radius_tetrahedron_l2512_251290

/-- Given a tetrahedron with volume V, face areas S₁, S₂, S₃, S₄, and an inscribed sphere of radius R,
    prove that R = 3V / (S₁ + S₂ + S₃ + S₄) -/
theorem inscribed_sphere_radius_tetrahedron (V : ℝ) (S₁ S₂ S₃ S₄ : ℝ) (R : ℝ) 
    (h_volume : V > 0)
    (h_areas : S₁ > 0 ∧ S₂ > 0 ∧ S₃ > 0 ∧ S₄ > 0)
    (h_inscribed : R > 0) :
  R = 3 * V / (S₁ + S₂ + S₃ + S₄) :=
sorry

end NUMINAMATH_CALUDE_inscribed_sphere_radius_tetrahedron_l2512_251290


namespace NUMINAMATH_CALUDE_fraction_sum_division_l2512_251297

theorem fraction_sum_division (a b c d e f g h : ℚ) :
  a = 3/7 →
  b = 5/8 →
  c = 5/12 →
  d = 2/9 →
  e = a + b →
  f = c + d →
  g = e / f →
  g = 531/322 :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_division_l2512_251297


namespace NUMINAMATH_CALUDE_thirteen_power_division_l2512_251238

theorem thirteen_power_division : (13 : ℕ) ^ 8 / (13 : ℕ) ^ 5 = 2197 := by sorry

end NUMINAMATH_CALUDE_thirteen_power_division_l2512_251238


namespace NUMINAMATH_CALUDE_prob_two_defective_of_six_l2512_251242

/-- The probability of finding exactly two defective components in two tests -/
def probability_two_defective (total : ℕ) (defective : ℕ) : ℚ :=
  if total < 2 ∨ defective ≠ 2 then 0
  else 2 / (total * (total - 1))

/-- Theorem stating the probability of finding two defective components
    out of six total components, where two are defective -/
theorem prob_two_defective_of_six :
  probability_two_defective 6 2 = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_defective_of_six_l2512_251242


namespace NUMINAMATH_CALUDE_reciprocal_inequality_l2512_251273

theorem reciprocal_inequality (a b : ℝ) (h1 : a > 0) (h2 : 0 > b) : 1 / a > 1 / b := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_inequality_l2512_251273


namespace NUMINAMATH_CALUDE_combination_number_identity_l2512_251228

theorem combination_number_identity (n r : ℕ) (h1 : n > r) (h2 : r ≥ 1) :
  Nat.choose n r = (n / r) * Nat.choose (n - 1) (r - 1) := by
  sorry

end NUMINAMATH_CALUDE_combination_number_identity_l2512_251228


namespace NUMINAMATH_CALUDE_no_integer_roots_l2512_251253

theorem no_integer_roots : ∀ x : ℤ, x^3 - 5*x^2 - 11*x + 35 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_roots_l2512_251253


namespace NUMINAMATH_CALUDE_collinear_points_m_value_l2512_251218

/-- Given three points A, B, and C in a 2D plane, this function checks if they are collinear -/
def are_collinear (A B C : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (x₃, y₃) := C
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- Theorem stating that if A(0,2), B(3,0), and C(m,1-m) are collinear, then m = -9 -/
theorem collinear_points_m_value :
  ∀ m : ℝ, are_collinear (0, 2) (3, 0) (m, 1 - m) → m = -9 :=
by
  sorry

end NUMINAMATH_CALUDE_collinear_points_m_value_l2512_251218


namespace NUMINAMATH_CALUDE_extreme_value_implies_a_equals_negative_one_l2512_251231

/-- Given a function f(x) = x^3 - ax^2 - x + a, where a is a real number,
    if f(x) has an extreme value at x = -1, then a = -1 -/
theorem extreme_value_implies_a_equals_negative_one (a : ℝ) :
  let f := λ x : ℝ => x^3 - a*x^2 - x + a
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-1 - ε) (-1 + ε), f (-1) ≥ f x) ∨
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-1 - ε) (-1 + ε), f (-1) ≤ f x) →
  a = -1 :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_implies_a_equals_negative_one_l2512_251231


namespace NUMINAMATH_CALUDE_tangent_sum_identity_l2512_251293

theorem tangent_sum_identity (α β γ : Real) 
  (h_sum : α + β + γ = Real.pi / 2)
  (h_α : ∃ (t : Real), Real.tan α = t)
  (h_β : ∃ (t : Real), Real.tan β = t)
  (h_γ : ∃ (t : Real), Real.tan γ = t) :
  Real.tan α * Real.tan β + Real.tan β * Real.tan γ + Real.tan γ * Real.tan α = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_identity_l2512_251293


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2512_251259

def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | -2 ≤ x ∧ x < 1}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -1 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2512_251259


namespace NUMINAMATH_CALUDE_final_sum_theorem_l2512_251203

def num_participants : ℕ := 43

def calculator_operation (n : ℕ) (initial_value : ℤ) : ℤ :=
  match initial_value with
  | 2 => 2^(2^n)
  | 1 => 1
  | -1 => (-1)^n
  | _ => initial_value

theorem final_sum_theorem :
  calculator_operation num_participants 2 +
  calculator_operation num_participants 1 +
  calculator_operation num_participants (-1) = 2^(2^num_participants) := by
  sorry

end NUMINAMATH_CALUDE_final_sum_theorem_l2512_251203


namespace NUMINAMATH_CALUDE_decreasing_f_implies_a_geq_2_l2512_251229

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x - 3

-- State the theorem
theorem decreasing_f_implies_a_geq_2 :
  ∀ a : ℝ, (∀ x y : ℝ, -8 < x ∧ x < y ∧ y < 2 → f a x > f a y) → a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_f_implies_a_geq_2_l2512_251229


namespace NUMINAMATH_CALUDE_final_price_correct_l2512_251281

/-- The final selling price of an item after two discounts -/
def final_price (m : ℝ) : ℝ :=
  0.8 * m - 10

/-- Theorem stating the correctness of the final price calculation -/
theorem final_price_correct (m : ℝ) :
  let first_discount := 0.2
  let second_discount := 10
  let price_after_first := m * (1 - first_discount)
  let final_price := price_after_first - second_discount
  final_price = 0.8 * m - 10 :=
by sorry

end NUMINAMATH_CALUDE_final_price_correct_l2512_251281


namespace NUMINAMATH_CALUDE_unique_two_digit_number_l2512_251292

/-- A function that reverses a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- Theorem stating that 73 is the unique two-digit number satisfying the given condition -/
theorem unique_two_digit_number : 
  ∃! n : ℕ, 10 ≤ n ∧ n < 100 ∧ n = 2 * (reverse_digits n) - 1 :=
by
  sorry

#check unique_two_digit_number

end NUMINAMATH_CALUDE_unique_two_digit_number_l2512_251292


namespace NUMINAMATH_CALUDE_tarantulas_in_egg_sac_l2512_251298

/-- The number of legs a tarantula has -/
def tarantula_legs : ℕ := 8

/-- The total number of baby tarantula legs in the egg sacs -/
def total_legs : ℕ := 32000

/-- The number of egg sacs containing the baby tarantulas -/
def num_egg_sacs : ℕ := 4

/-- The number of tarantulas in one egg sac -/
def tarantulas_per_sac : ℕ := total_legs / (tarantula_legs * num_egg_sacs)

theorem tarantulas_in_egg_sac : tarantulas_per_sac = 1000 := by
  sorry

end NUMINAMATH_CALUDE_tarantulas_in_egg_sac_l2512_251298


namespace NUMINAMATH_CALUDE_square_of_five_times_sqrt_three_l2512_251204

theorem square_of_five_times_sqrt_three : (5 * Real.sqrt 3) ^ 2 = 75 := by
  sorry

end NUMINAMATH_CALUDE_square_of_five_times_sqrt_three_l2512_251204


namespace NUMINAMATH_CALUDE_curve_self_intersection_l2512_251263

-- Define the parametric equations
def x (t : ℝ) : ℝ := t^2 + 3
def y (t : ℝ) : ℝ := t^3 - 6*t + 4

-- Theorem statement
theorem curve_self_intersection :
  ∃ (a b : ℝ), a ≠ b ∧ x a = x b ∧ y a = y b ∧ x a = 9 ∧ y a = 4 := by
  sorry

end NUMINAMATH_CALUDE_curve_self_intersection_l2512_251263


namespace NUMINAMATH_CALUDE_first_round_score_l2512_251256

def card_values : List ℕ := [2, 4, 7, 13]

theorem first_round_score (total_score : ℕ) (last_round_score : ℕ) 
  (h1 : total_score = 16)
  (h2 : last_round_score = 2)
  (h3 : card_values.sum = 26)
  (h4 : ∃ (n : ℕ), n * card_values.sum = 16 + 17 + 21 + 24)
  : ∃ (first_round_score : ℕ), 
    first_round_score ∈ card_values ∧ 
    ∃ (second_round_score : ℕ), 
      second_round_score ∈ card_values ∧ 
      first_round_score + second_round_score + last_round_score = total_score ∧
      first_round_score = 7 :=
by
  sorry

#check first_round_score

end NUMINAMATH_CALUDE_first_round_score_l2512_251256


namespace NUMINAMATH_CALUDE_factorization_equality_l2512_251221

theorem factorization_equality (x : ℝ) : 8*x - 2*x^2 = 2*x*(4 - x) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2512_251221


namespace NUMINAMATH_CALUDE_bill_donut_order_ways_l2512_251278

/-- The number of ways to distribute identical items into distinct groups -/
def distribute_items (items : ℕ) (groups : ℕ) : ℕ :=
  Nat.choose (items + groups - 1) (groups - 1)

/-- The number of ways Bill can fulfill his donut order -/
theorem bill_donut_order_ways : distribute_items 3 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_bill_donut_order_ways_l2512_251278


namespace NUMINAMATH_CALUDE_ratio_x_y_l2512_251239

theorem ratio_x_y (x y : ℝ) (h : (0.6 * 500 : ℝ) = 0.5 * x ∧ (0.6 * 500 : ℝ) = 0.4 * y) : 
  x / y = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_y_l2512_251239
