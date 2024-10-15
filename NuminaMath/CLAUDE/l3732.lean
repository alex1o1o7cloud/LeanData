import Mathlib

namespace NUMINAMATH_CALUDE_yellow_ball_fraction_l3732_373252

theorem yellow_ball_fraction (total : ℕ) (green blue white yellow : ℕ) : 
  (green : ℚ) / total = 1 / 4 →
  (blue : ℚ) / total = 1 / 8 →
  white = 26 →
  blue = 6 →
  total = green + blue + white + yellow →
  (yellow : ℚ) / total = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_yellow_ball_fraction_l3732_373252


namespace NUMINAMATH_CALUDE_game_result_l3732_373268

def g (n : Nat) : Nat :=
  if n % 6 = 0 then 8
  else if n % 3 = 0 then 4
  else if n % 2 = 0 then 3
  else 1

def cora_rolls : List Nat := [5, 4, 3, 6, 2, 1]
def dana_rolls : List Nat := [6, 3, 4, 3, 5, 3]

def total_points (rolls : List Nat) : Nat :=
  (rolls.map g).sum

theorem game_result : (total_points cora_rolls) * (total_points dana_rolls) = 480 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l3732_373268


namespace NUMINAMATH_CALUDE_scientific_notation_proof_l3732_373219

def number_to_express : ℝ := 460000000

theorem scientific_notation_proof :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ number_to_express = a * (10 : ℝ) ^ n ∧ a = 4.6 ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_proof_l3732_373219


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l3732_373222

theorem min_value_theorem (x : ℝ) : (x^2 + 9) / Real.sqrt (x^2 + 5) ≥ 5 := by
  sorry

theorem min_value_achievable : ∃ x : ℝ, (x^2 + 9) / Real.sqrt (x^2 + 5) = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l3732_373222


namespace NUMINAMATH_CALUDE_unique_three_digit_factorial_sum_l3732_373217

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def digits_factorial_sum (n : ℕ) : ℕ :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  factorial d1 + factorial d2 + factorial d3

theorem unique_three_digit_factorial_sum :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧
  digits_factorial_sum n = n ∧
  (n / 100 = 3 ∨ (n / 10) % 10 = 3 ∨ n % 10 = 3) ∧
  n = 145 :=
sorry

end NUMINAMATH_CALUDE_unique_three_digit_factorial_sum_l3732_373217


namespace NUMINAMATH_CALUDE_value_added_to_number_l3732_373258

theorem value_added_to_number (sum number value : ℕ) : 
  sum = number + value → number = 81 → sum = 96 → value = 15 := by
  sorry

end NUMINAMATH_CALUDE_value_added_to_number_l3732_373258


namespace NUMINAMATH_CALUDE_composite_quotient_l3732_373298

theorem composite_quotient (n : ℕ) (h1 : n ≥ 4) (h2 : n ∣ 2^n - 2) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ (2^n - 2) / n = a * b := by
  sorry

end NUMINAMATH_CALUDE_composite_quotient_l3732_373298


namespace NUMINAMATH_CALUDE_card_row_theorem_l3732_373261

/-- Represents a row of nine cards --/
def CardRow := Fin 9 → ℕ

/-- Checks if three consecutive cards are in increasing order --/
def increasing_three (row : CardRow) (i : Fin 7) : Prop :=
  row i < row (i + 1) ∧ row (i + 1) < row (i + 2)

/-- Checks if three consecutive cards are in decreasing order --/
def decreasing_three (row : CardRow) (i : Fin 7) : Prop :=
  row i > row (i + 1) ∧ row (i + 1) > row (i + 2)

/-- The main theorem --/
theorem card_row_theorem (row : CardRow) : 
  (∀ i : Fin 9, row i ∈ Finset.range 10) →  -- Cards are numbered 1 to 9
  (∀ i j : Fin 9, i ≠ j → row i ≠ row j) →  -- All numbers are different
  (∀ i : Fin 7, ¬increasing_three row i) →  -- No three consecutive increasing
  (∀ i : Fin 7, ¬decreasing_three row i) →  -- No three consecutive decreasing
  row 0 = 1 →                               -- Given visible cards
  row 1 = 6 →
  row 2 = 3 →
  row 3 = 4 →
  row 6 = 8 →
  row 7 = 7 →
  row 4 = 5 ∧ row 5 = 2 ∧ row 8 = 9         -- Conclusion: A = 5, B = 2, C = 9
:= by sorry


end NUMINAMATH_CALUDE_card_row_theorem_l3732_373261


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3732_373208

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + Complex.I) = 4) : 
  z.im = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3732_373208


namespace NUMINAMATH_CALUDE_prop_a_prop_b_prop_d_l3732_373201

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Proposition A
theorem prop_a (t : Triangle) (h : t.A > t.B) : Real.sin t.A > Real.sin t.B := by sorry

-- Proposition B
theorem prop_b (t : Triangle) (h : t.A < π/2 ∧ t.B < π/2 ∧ t.C < π/2) : Real.sin t.A > Real.cos t.B := by sorry

-- Proposition D
theorem prop_d (t : Triangle) (h1 : t.B = π/3) (h2 : t.b^2 = t.a * t.c) : t.A = π/3 ∧ t.B = π/3 ∧ t.C = π/3 := by sorry

end NUMINAMATH_CALUDE_prop_a_prop_b_prop_d_l3732_373201


namespace NUMINAMATH_CALUDE_nancy_carrots_l3732_373209

/-- The number of carrots Nancy picked the next day -/
def carrots_picked_next_day (initial_carrots : ℕ) (thrown_out : ℕ) (total_carrots : ℕ) : ℕ :=
  total_carrots - (initial_carrots - thrown_out)

/-- Proof that Nancy picked 21 carrots the next day -/
theorem nancy_carrots :
  carrots_picked_next_day 12 2 31 = 21 := by
  sorry

end NUMINAMATH_CALUDE_nancy_carrots_l3732_373209


namespace NUMINAMATH_CALUDE_chess_tournament_games_per_pair_l3732_373262

/-- Represents a chess tournament with a given number of players and total games. -/
structure ChessTournament where
  num_players : ℕ
  total_games : ℕ

/-- Calculates the number of times each player plays against each opponent in a chess tournament. -/
def games_per_pair (tournament : ChessTournament) : ℚ :=
  (2 * tournament.total_games : ℚ) / (tournament.num_players * (tournament.num_players - 1))

/-- Theorem stating that in a chess tournament with 18 players and 306 total games,
    each player plays against each opponent exactly 2 times. -/
theorem chess_tournament_games_per_pair :
  let tournament := ChessTournament.mk 18 306
  games_per_pair tournament = 2 := by
  sorry


end NUMINAMATH_CALUDE_chess_tournament_games_per_pair_l3732_373262


namespace NUMINAMATH_CALUDE_smallest_factor_b_l3732_373265

theorem smallest_factor_b : 
  ∀ b : ℕ+, 
    (∃ (p q : ℤ), (∀ x : ℝ, x^2 + b * x + 2016 = (x + p) * (x + q))) →
    b ≥ 92 :=
by sorry

end NUMINAMATH_CALUDE_smallest_factor_b_l3732_373265


namespace NUMINAMATH_CALUDE_fraction_equality_l3732_373283

theorem fraction_equality (x y : ℝ) : 
  (5 + 2*x) / (7 + 3*x + y) = (3 + 4*x) / (4 + 2*x + y) ↔ 
  8*x^2 + 19*x + 5*x*y = -1 - 5*y :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_l3732_373283


namespace NUMINAMATH_CALUDE_lcm_gcf_ratio_252_675_l3732_373237

theorem lcm_gcf_ratio_252_675 : 
  Nat.lcm 252 675 / Nat.gcd 252 675 = 2100 := by sorry

end NUMINAMATH_CALUDE_lcm_gcf_ratio_252_675_l3732_373237


namespace NUMINAMATH_CALUDE_no_snuggly_numbers_l3732_373280

/-- A two-digit positive integer is 'snuggly' if it equals the sum of its nonzero tens digit, 
    the cube of its units digit, and 5. -/
def is_snuggly (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ ∃ a b : ℕ, 
    n = 10 * a + b ∧ 
    1 ≤ a ∧ a ≤ 9 ∧ 
    0 ≤ b ∧ b ≤ 9 ∧
    n = a + b^3 + 5

theorem no_snuggly_numbers : ¬∃ n : ℕ, is_snuggly n :=
sorry

end NUMINAMATH_CALUDE_no_snuggly_numbers_l3732_373280


namespace NUMINAMATH_CALUDE_q_transformation_l3732_373200

theorem q_transformation (w d z z' : ℝ) (hw : w > 0) (hd : d > 0) (hz : z ≠ 0) :
  let q := 5 * w / (4 * d * z^2)
  let q' := 5 * (4 * w) / (4 * (2 * d) * z'^2)
  q' / q = 2/9 ↔ z' = 3 * Real.sqrt 2 * z := by
sorry

end NUMINAMATH_CALUDE_q_transformation_l3732_373200


namespace NUMINAMATH_CALUDE_units_digit_of_3_to_1987_l3732_373277

theorem units_digit_of_3_to_1987 : 3^1987 % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_3_to_1987_l3732_373277


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3732_373238

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, (x < 1 ∨ x > 5) → x^2 - 2*(a-2)*x + a > 0) ↔ 
  (1 < a ∧ a ≤ 5) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3732_373238


namespace NUMINAMATH_CALUDE_sum_of_nineteen_terms_l3732_373226

/-- Given an arithmetic sequence {a_n}, S_n represents the sum of its first n terms -/
def S (n : ℕ) : ℝ := sorry

/-- {a_n} is an arithmetic sequence -/
def isArithmeticSequence (a : ℕ → ℝ) : Prop := sorry

/-- The second, ninth, and nineteenth terms of the sequence sum to 6 -/
axiom sum_condition (a : ℕ → ℝ) : a 2 + a 9 + a 19 = 6

theorem sum_of_nineteen_terms (a : ℕ → ℝ) (h : isArithmeticSequence a) : 
  S 19 = 38 := by sorry

end NUMINAMATH_CALUDE_sum_of_nineteen_terms_l3732_373226


namespace NUMINAMATH_CALUDE_factor_x9_minus_512_l3732_373223

theorem factor_x9_minus_512 (x : ℝ) : 
  x^9 - 512 = (x - 2) * (x^2 + 2*x + 4) * (x^6 + 8*x^3 + 64) := by
  sorry

end NUMINAMATH_CALUDE_factor_x9_minus_512_l3732_373223


namespace NUMINAMATH_CALUDE_arlene_hike_distance_l3732_373236

/-- Calculates the total distance hiked given the hiking time and average pace. -/
def total_distance (time : ℝ) (pace : ℝ) : ℝ := time * pace

/-- Proves that Arlene hiked 24 miles on Saturday. -/
theorem arlene_hike_distance :
  let time : ℝ := 6 -- hours
  let pace : ℝ := 4 -- miles per hour
  total_distance time pace = 24 := by
  sorry

end NUMINAMATH_CALUDE_arlene_hike_distance_l3732_373236


namespace NUMINAMATH_CALUDE_function_max_min_difference_l3732_373225

theorem function_max_min_difference (a : ℝ) (h1 : a > 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x-1)
  (∀ x ∈ Set.Icc 2 3, f x ≤ f 3) ∧
  (∀ x ∈ Set.Icc 2 3, f 2 ≤ f x) ∧
  (f 3 - f 2 = a / 2) →
  a = 3/2 := by sorry

end NUMINAMATH_CALUDE_function_max_min_difference_l3732_373225


namespace NUMINAMATH_CALUDE_simplify_expression_l3732_373253

theorem simplify_expression (x : ℝ) : (3*x)^3 - (4*x^2)*(2*x^3) = 27*x^3 - 8*x^5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3732_373253


namespace NUMINAMATH_CALUDE_fixed_point_parabola_l3732_373284

theorem fixed_point_parabola (k : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 3 * x^2 + k * x - 2 * k
  f 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_parabola_l3732_373284


namespace NUMINAMATH_CALUDE_max_packing_ge_min_covering_l3732_373270

/-- Represents a polygon in 2D space -/
structure Polygon

/-- The largest number of non-overlapping circles with diameter 1 whose centers lie inside the polygon -/
def max_packing (M : Polygon) : ℕ :=
  sorry

/-- The smallest number of circles with radius 1 needed to cover the entire polygon -/
def min_covering (M : Polygon) : ℕ :=
  sorry

/-- Theorem stating that the maximum packing is greater than or equal to the minimum covering -/
theorem max_packing_ge_min_covering (M : Polygon) : max_packing M ≥ min_covering M :=
  sorry

end NUMINAMATH_CALUDE_max_packing_ge_min_covering_l3732_373270


namespace NUMINAMATH_CALUDE_chromosome_stability_l3732_373211

-- Define the number of chromosomes in somatic cells
def somaticChromosomes : ℕ := 46

-- Define the process of meiosis
def meiosis (n : ℕ) : ℕ := n / 2

-- Define the process of fertilization
def fertilization (n : ℕ) : ℕ := n * 2

-- Theorem: Meiosis and fertilization maintain chromosome stability across generations
theorem chromosome_stability :
  ∀ (generation : ℕ),
    fertilization (meiosis somaticChromosomes) = somaticChromosomes :=
by sorry

end NUMINAMATH_CALUDE_chromosome_stability_l3732_373211


namespace NUMINAMATH_CALUDE_scientific_notation_of_given_number_l3732_373203

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ coefficient ∧ coefficient < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

/-- The given number -/
def givenNumber : ℝ := 0.0000046

/-- Theorem: The scientific notation of 0.0000046 is 4.6 × 10^(-6) -/
theorem scientific_notation_of_given_number :
  toScientificNotation givenNumber = ScientificNotation.mk 4.6 (-6) sorry := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_given_number_l3732_373203


namespace NUMINAMATH_CALUDE_range_of_a_l3732_373295

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, StrictMono (fun x => (3 - 2*a)^x)
def q (a : ℝ) : Prop := ∀ x : ℝ, 0 < x^2 + 2*a*x + 4

-- Define the theorem
theorem range_of_a (a : ℝ) 
  (h1 : p a ∨ q a) 
  (h2 : ¬(p a ∧ q a)) : 
  a ≤ -2 ∨ (1 ≤ a ∧ a < 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3732_373295


namespace NUMINAMATH_CALUDE_division_problem_l3732_373250

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) 
  (h1 : dividend = 12)
  (h2 : divisor = 17)
  (h3 : remainder = 10)
  (h4 : dividend = divisor * quotient + remainder) :
  quotient = 0 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3732_373250


namespace NUMINAMATH_CALUDE_temperature_difference_l3732_373249

def january_temp : ℝ := -3
def march_temp : ℝ := 2

theorem temperature_difference : march_temp - january_temp = 5 := by
  sorry

end NUMINAMATH_CALUDE_temperature_difference_l3732_373249


namespace NUMINAMATH_CALUDE_alice_stool_height_l3732_373276

/-- The height of the ceiling above the floor in centimeters -/
def ceiling_height : ℝ := 300

/-- The distance of the light bulb below the ceiling in centimeters -/
def light_bulb_below_ceiling : ℝ := 15

/-- Alice's height in centimeters -/
def alice_height : ℝ := 150

/-- The distance Alice can reach above her head in centimeters -/
def alice_reach : ℝ := 40

/-- The minimum height of the stool Alice needs in centimeters -/
def stool_height : ℝ := 95

theorem alice_stool_height : 
  ceiling_height - light_bulb_below_ceiling = alice_height + alice_reach + stool_height := by
  sorry

end NUMINAMATH_CALUDE_alice_stool_height_l3732_373276


namespace NUMINAMATH_CALUDE_flower_problem_solution_l3732_373216

/-- Given initial flowers and minimum flowers per bouquet, 
    calculate additional flowers needed and number of bouquets -/
def flower_arrangement (initial_flowers : ℕ) (min_per_bouquet : ℕ) : 
  {additional_flowers : ℕ // ∃ (num_bouquets : ℕ), 
    num_bouquets * min_per_bouquet = initial_flowers + additional_flowers ∧
    num_bouquets * min_per_bouquet > initial_flowers ∧
    ∀ (k : ℕ), k * min_per_bouquet > initial_flowers → 
      k * min_per_bouquet ≥ num_bouquets * min_per_bouquet} :=
sorry

theorem flower_problem_solution : 
  (flower_arrangement 1273 89).val = 62 ∧ 
  ∃ (num_bouquets : ℕ), num_bouquets = 15 ∧
    num_bouquets * 89 = 1273 + (flower_arrangement 1273 89).val :=
sorry

end NUMINAMATH_CALUDE_flower_problem_solution_l3732_373216


namespace NUMINAMATH_CALUDE_game_time_calculation_l3732_373294

/-- Calculates the total time before playing a game given download, installation, and tutorial times. -/
def totalGameTime (downloadTime : ℕ) : ℕ :=
  let installTime := downloadTime / 2
  let combinedTime := downloadTime + installTime
  let tutorialTime := 3 * combinedTime
  combinedTime + tutorialTime

/-- Theorem stating that for a download time of 10 minutes, the total time before playing is 60 minutes. -/
theorem game_time_calculation :
  totalGameTime 10 = 60 := by
  sorry

end NUMINAMATH_CALUDE_game_time_calculation_l3732_373294


namespace NUMINAMATH_CALUDE_circle_area_tripled_l3732_373248

theorem circle_area_tripled (r n : ℝ) (h : r > 0) (h_n : n > 0) :
  π * (r + n)^2 = 3 * π * r^2 → r = n * (1 + Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_tripled_l3732_373248


namespace NUMINAMATH_CALUDE_prime_triplet_equation_l3732_373299

theorem prime_triplet_equation (p q r : ℕ) : 
  Prime p ∧ Prime q ∧ Prime r →
  (p : ℚ) / q - 4 / (r + 1) = 1 →
  ((p = 7 ∧ q = 3 ∧ r = 2) ∨ (p = 5 ∧ q = 3 ∧ r = 5)) := by
  sorry

end NUMINAMATH_CALUDE_prime_triplet_equation_l3732_373299


namespace NUMINAMATH_CALUDE_flowers_per_pot_l3732_373235

theorem flowers_per_pot (total_pots : ℕ) (total_flowers : ℕ) (h1 : total_pots = 544) (h2 : total_flowers = 17408) :
  total_flowers / total_pots = 32 := by
  sorry

end NUMINAMATH_CALUDE_flowers_per_pot_l3732_373235


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3732_373228

theorem triangle_perimeter : ∀ (a b c : ℝ),
  a = 4 ∧ b = 6 ∧ c^2 - 6*c + 8 = 0 ∧
  a + b > c ∧ a + c > b ∧ b + c > a →
  a + b + c = 14 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3732_373228


namespace NUMINAMATH_CALUDE_min_candies_count_l3732_373271

theorem min_candies_count (c : ℕ) : 
  c % 6 = 5 → 
  c % 8 = 7 → 
  c % 9 = 6 → 
  c % 11 = 0 → 
  (∀ n : ℕ, n < c → 
    (n % 6 = 5 ∧ n % 8 = 7 ∧ n % 9 = 6 ∧ n % 11 = 0) → False) → 
  c = 359 := by
sorry

end NUMINAMATH_CALUDE_min_candies_count_l3732_373271


namespace NUMINAMATH_CALUDE_parabola_line_intersection_triangle_area_l3732_373279

/-- Given a parabola y = x^2 + 2 and a line y = r, if the triangle formed by the vertex of the parabola
    and the two intersections of the line and parabola has an area A such that 10 ≤ A ≤ 50,
    then 10^(2/3) + 2 ≤ r ≤ 50^(2/3) + 2. -/
theorem parabola_line_intersection_triangle_area (r : ℝ) : 
  let parabola := fun x : ℝ => x^2 + 2
  let line := fun _ : ℝ => r
  let vertex := (0, parabola 0)
  let intersections := {x : ℝ | parabola x = line x}
  let triangle_area := (r - 2)^(3/2) / 2
  10 ≤ triangle_area ∧ triangle_area ≤ 50 → 10^(2/3) + 2 ≤ r ∧ r ≤ 50^(2/3) + 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_triangle_area_l3732_373279


namespace NUMINAMATH_CALUDE_tiles_remaining_l3732_373281

theorem tiles_remaining (initial_tiles : ℕ) : 
  initial_tiles = 2022 → 
  (initial_tiles - initial_tiles / 6 - (initial_tiles - initial_tiles / 6) / 5 - 
   (initial_tiles - initial_tiles / 6 - (initial_tiles - initial_tiles / 6) / 5) / 4) = 1011 := by
  sorry

end NUMINAMATH_CALUDE_tiles_remaining_l3732_373281


namespace NUMINAMATH_CALUDE_bicycle_distance_l3732_373266

theorem bicycle_distance (motorcycle_speed : ℝ) (bicycle_speed_ratio : ℝ) (time_minutes : ℝ) :
  motorcycle_speed = 90 →
  bicycle_speed_ratio = 2 / 3 →
  time_minutes = 15 →
  (bicycle_speed_ratio * motorcycle_speed) * (time_minutes / 60) = 15 := by
sorry

end NUMINAMATH_CALUDE_bicycle_distance_l3732_373266


namespace NUMINAMATH_CALUDE_complex_sum_zero_l3732_373206

noncomputable def w : ℂ := Complex.exp (Complex.I * (3 * Real.pi / 8))

theorem complex_sum_zero :
  w / (1 + w^3) + w^2 / (1 + w^5) + w^3 / (1 + w^7) = 0 := by sorry

end NUMINAMATH_CALUDE_complex_sum_zero_l3732_373206


namespace NUMINAMATH_CALUDE_trig_expression_equals_one_l3732_373247

theorem trig_expression_equals_one :
  let x : Real := 40 * π / 180
  let y : Real := 50 * π / 180
  (Real.sqrt (1 - 2 * Real.sin x * Real.cos x)) / (Real.cos x - Real.sqrt (1 - Real.sin y ^ 2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_one_l3732_373247


namespace NUMINAMATH_CALUDE_apple_selling_price_l3732_373292

/-- The selling price of an apple, given its cost price and loss ratio. -/
def selling_price (cost_price : ℚ) (loss_ratio : ℚ) : ℚ :=
  cost_price * (1 - loss_ratio)

/-- Theorem stating that the selling price of an apple is 15,
    given a cost price of 18 and a loss ratio of 1/6. -/
theorem apple_selling_price :
  selling_price 18 (1/6) = 15 := by
  sorry

end NUMINAMATH_CALUDE_apple_selling_price_l3732_373292


namespace NUMINAMATH_CALUDE_tap_b_fills_12_liters_l3732_373244

/-- Represents the water flow problem with two taps filling a bucket. -/
structure WaterFlow where
  bucket_volume : ℝ
  tap_a_rate : ℝ
  fill_time_both : ℝ

/-- The amount of water tap B fills in 20 minutes. -/
def tap_b_fill_20min (w : WaterFlow) : ℝ :=
  2 * (w.bucket_volume - w.tap_a_rate * w.fill_time_both)

/-- Theorem stating that tap B fills 12 liters in 20 minutes under given conditions. -/
theorem tap_b_fills_12_liters (w : WaterFlow)
  (h1 : w.bucket_volume = 36)
  (h2 : w.tap_a_rate = 3)
  (h3 : w.fill_time_both = 10) :
  tap_b_fill_20min w = 12 := by
  sorry

end NUMINAMATH_CALUDE_tap_b_fills_12_liters_l3732_373244


namespace NUMINAMATH_CALUDE_coworker_lunch_pizzas_l3732_373210

/-- Calculates the number of pizzas needed for a group lunch -/
def pizzas_ordered (coworkers : ℕ) (slices_per_pizza : ℕ) (slices_per_person : ℕ) : ℕ :=
  (coworkers * slices_per_person) / slices_per_pizza

/-- Proves that 12 coworkers each getting 2 slices from pizzas with 8 slices each requires 3 pizzas -/
theorem coworker_lunch_pizzas :
  pizzas_ordered 12 8 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_coworker_lunch_pizzas_l3732_373210


namespace NUMINAMATH_CALUDE_inscribed_trapezoid_intersection_l3732_373267

/-- A trapezoid inscribed in the parabola y = x^2 -/
structure InscribedTrapezoid where
  /-- Left x-coordinate of the upper base -/
  a : ℝ
  /-- Left x-coordinate of the lower base -/
  b : ℝ
  /-- The product of the lengths of the bases is k -/
  base_product : (2 * a) * (2 * b) = k
  /-- k is positive -/
  k_pos : k > 0

/-- The theorem stating that all inscribed trapezoids with base product k 
    have lateral sides intersecting at (0, -k/4) -/
theorem inscribed_trapezoid_intersection 
  (k : ℝ) (trap : InscribedTrapezoid) : 
  ∃ (x y : ℝ), x = 0 ∧ y = -k/4 ∧ 
  (∀ (t : ℝ), 
    ((t - trap.a) * (trap.b^2 - trap.a^2) = (trap.b - trap.a) * (t^2 - trap.a^2) ↔ 
     (t = x ∧ t^2 = y)) ∧
    ((t + trap.a) * (trap.b^2 - trap.a^2) = (trap.b + trap.a) * (t^2 - trap.a^2) ↔ 
     (t = x ∧ t^2 = y))) :=
by sorry


end NUMINAMATH_CALUDE_inscribed_trapezoid_intersection_l3732_373267


namespace NUMINAMATH_CALUDE_simplify_expression_l3732_373229

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) :
  (25 * x^3) * (8 * x^2) * (1 / (4 * x)^3) = (25 / 8) * x^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3732_373229


namespace NUMINAMATH_CALUDE_square_difference_l3732_373205

theorem square_difference (a b : ℝ) (h1 : a + b = 8) (h2 : a - b = 4) : a^2 - b^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l3732_373205


namespace NUMINAMATH_CALUDE_laptop_price_l3732_373204

theorem laptop_price (upfront_percentage : ℚ) (upfront_payment : ℚ) 
  (h1 : upfront_percentage = 20 / 100)
  (h2 : upfront_payment = 240) : 
  upfront_payment / upfront_percentage = 1200 := by
sorry

end NUMINAMATH_CALUDE_laptop_price_l3732_373204


namespace NUMINAMATH_CALUDE_angle_between_diagonals_l3732_373286

/-- 
Given a quadrilateral with area A, and diagonals d₁ and d₂, 
the angle α between the diagonals satisfies the equation:
A = (1/2) * d₁ * d₂ * sin(α)
-/
def quadrilateral_area_diagonals (A d₁ d₂ α : ℝ) : Prop :=
  A = (1/2) * d₁ * d₂ * Real.sin α

theorem angle_between_diagonals (A d₁ d₂ α : ℝ) 
  (h_area : A = 3)
  (h_diag1 : d₁ = 6)
  (h_diag2 : d₂ = 2)
  (h_quad : quadrilateral_area_diagonals A d₁ d₂ α) :
  α = π / 6 := by
  sorry

#check angle_between_diagonals

end NUMINAMATH_CALUDE_angle_between_diagonals_l3732_373286


namespace NUMINAMATH_CALUDE_rectangular_parallelepiped_surface_area_l3732_373243

theorem rectangular_parallelepiped_surface_area
  (m n : ℤ)
  (h_m_lt_n : m < n)
  (x y z : ℤ)
  (h_x : x = n * (n - m))
  (h_y : y = m * n)
  (h_z : z = m * (n - m)) :
  2 * (x + y) * z = 2 * x * y := by
  sorry

end NUMINAMATH_CALUDE_rectangular_parallelepiped_surface_area_l3732_373243


namespace NUMINAMATH_CALUDE_total_earnings_l3732_373207

def wednesday_amount : ℚ := 1832
def sunday_amount : ℚ := 3162.5

theorem total_earnings : wednesday_amount + sunday_amount = 4994.5 := by
  sorry

end NUMINAMATH_CALUDE_total_earnings_l3732_373207


namespace NUMINAMATH_CALUDE_correct_cube_root_l3732_373293

-- Define the cube root function for real numbers
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Theorem statement
theorem correct_cube_root : cubeRoot (-125) = -5 := by
  sorry

end NUMINAMATH_CALUDE_correct_cube_root_l3732_373293


namespace NUMINAMATH_CALUDE_rs_length_l3732_373259

/-- Right-angled triangle PQR with perpendiculars to PQ at P and QR at R meeting at S -/
structure SpecialTriangle where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  S : ℝ × ℝ
  pq_length : dist P Q = 6
  qr_length : dist Q R = 8
  right_angle : (Q.1 - P.1) * (R.1 - Q.1) + (Q.2 - P.2) * (R.2 - Q.2) = 0
  perp_at_p : (S.1 - P.1) * (Q.1 - P.1) + (S.2 - P.2) * (Q.2 - P.2) = 0
  perp_at_r : (S.1 - R.1) * (Q.1 - R.1) + (S.2 - R.2) * (Q.2 - R.2) = 0

/-- The length of RS in the special triangle is 8 -/
theorem rs_length (t : SpecialTriangle) : dist t.R t.S = 8 := by
  sorry

end NUMINAMATH_CALUDE_rs_length_l3732_373259


namespace NUMINAMATH_CALUDE_smallest_prime_factors_difference_l3732_373296

def number : Nat := 96043

theorem smallest_prime_factors_difference (p q : Nat) : 
  Prime p ∧ Prime q ∧ p ∣ number ∧ q ∣ number ∧
  (∀ r, Prime r → r ∣ number → r ≥ p) ∧
  (∀ r, Prime r → r ∣ number → r = p ∨ r ≥ q) →
  q - p = 4 := by sorry

end NUMINAMATH_CALUDE_smallest_prime_factors_difference_l3732_373296


namespace NUMINAMATH_CALUDE_max_x5_value_l3732_373272

theorem max_x5_value (x1 x2 x3 x4 x5 : ℕ+) 
  (h : x1 + x2 + x3 + x4 + x5 ≤ x1 * x2 * x3 * x4 * x5) :
  x5 ≤ 5 ∧ ∃ (a b c d : ℕ+), a + b + c + d + 5 ≤ a * b * c * d * 5 := by
  sorry

end NUMINAMATH_CALUDE_max_x5_value_l3732_373272


namespace NUMINAMATH_CALUDE_mean_study_hours_thompson_class_l3732_373256

theorem mean_study_hours_thompson_class : 
  let study_hours := [0, 2, 4, 6, 8, 10, 12]
  let student_counts := [3, 6, 8, 5, 4, 2, 2]
  let total_students := 30
  let total_hours := (List.zip study_hours student_counts).map (fun (h, c) => h * c) |>.sum
  (total_hours : ℚ) / total_students = 5 := by
  sorry

end NUMINAMATH_CALUDE_mean_study_hours_thompson_class_l3732_373256


namespace NUMINAMATH_CALUDE_simple_interest_problem_l3732_373202

theorem simple_interest_problem (r : ℝ) (n : ℝ) :
  (400 * r * n) / 100 + 200 = (400 * (r + 5) * n) / 100 →
  n = 10 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l3732_373202


namespace NUMINAMATH_CALUDE_positive_difference_l3732_373287

theorem positive_difference (a b c d : ℝ) (h1 : a < b) (h2 : b < 0) (h3 : 0 < c) (h4 : c < d) :
  d - c - b - a > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_difference_l3732_373287


namespace NUMINAMATH_CALUDE_product_xyz_equals_negative_one_l3732_373260

theorem product_xyz_equals_negative_one (x y z : ℝ) 
  (h1 : x + 1/y = 3) (h2 : y + 1/z = 3) : x * y * z = -1 := by
  sorry

end NUMINAMATH_CALUDE_product_xyz_equals_negative_one_l3732_373260


namespace NUMINAMATH_CALUDE_power_product_square_l3732_373213

theorem power_product_square (x y : ℝ) : (x * y^2)^2 = x^2 * y^4 := by
  sorry

end NUMINAMATH_CALUDE_power_product_square_l3732_373213


namespace NUMINAMATH_CALUDE_max_knights_between_knights_theorem_l3732_373240

/-- Represents a seating arrangement of knights and samurais around a round table. -/
structure SeatingArrangement where
  total_knights : Nat
  total_samurais : Nat
  knights_with_samurai_right : Nat

/-- Calculates the maximum number of knights that could be seated next to two other knights. -/
def max_knights_between_knights (arrangement : SeatingArrangement) : Nat :=
  arrangement.total_knights - (arrangement.knights_with_samurai_right + 1)

/-- Theorem stating the maximum number of knights seated next to two other knights
    for the given arrangement. -/
theorem max_knights_between_knights_theorem (arrangement : SeatingArrangement) 
  (h1 : arrangement.total_knights = 40)
  (h2 : arrangement.total_samurais = 10)
  (h3 : arrangement.knights_with_samurai_right = 7) :
  max_knights_between_knights arrangement = 32 := by
  sorry

#eval max_knights_between_knights { total_knights := 40, total_samurais := 10, knights_with_samurai_right := 7 }

end NUMINAMATH_CALUDE_max_knights_between_knights_theorem_l3732_373240


namespace NUMINAMATH_CALUDE_unique_triple_solution_l3732_373278

theorem unique_triple_solution (p q : Nat) (n : Nat) (h_p : Nat.Prime p) (h_q : Nat.Prime q)
    (h_n : n > 1) (h_p_odd : Odd p) (h_q_odd : Odd q)
    (h_cong1 : q^(n+2) ≡ 3^(n+2) [MOD p^n])
    (h_cong2 : p^(n+2) ≡ 3^(n+2) [MOD q^n]) :
    p = 3 ∧ q = 3 := by
  sorry

#check unique_triple_solution

end NUMINAMATH_CALUDE_unique_triple_solution_l3732_373278


namespace NUMINAMATH_CALUDE_gcf_of_84_112_210_l3732_373274

theorem gcf_of_84_112_210 : Nat.gcd 84 (Nat.gcd 112 210) = 14 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_84_112_210_l3732_373274


namespace NUMINAMATH_CALUDE_rose_crystal_beads_l3732_373273

/-- The number of beads in each bracelet -/
def beads_per_bracelet : ℕ := 8

/-- The number of metal beads Nancy has -/
def nancy_metal_beads : ℕ := 40

/-- The number of pearl beads Nancy has more than metal beads -/
def nancy_extra_pearl_beads : ℕ := 20

/-- The number of bracelets they can make -/
def total_bracelets : ℕ := 20

/-- The relation between Rose's crystal and stone beads -/
def rose_stone_to_crystal_ratio : ℕ := 2

/-- Theorem: Rose has 20 crystal beads -/
theorem rose_crystal_beads :
  ∃ (crystal_beads : ℕ),
    crystal_beads = 20 ∧
    crystal_beads * (rose_stone_to_crystal_ratio + 1) =
      total_bracelets * beads_per_bracelet -
      (nancy_metal_beads + nancy_metal_beads + nancy_extra_pearl_beads) :=
by sorry

end NUMINAMATH_CALUDE_rose_crystal_beads_l3732_373273


namespace NUMINAMATH_CALUDE_partnership_profit_share_l3732_373254

/-- Given a partnership where A invests 3 times as much as B and 2/3 of what C invests,
    and the total profit is 55000, prove that C's share of the profit is (9/17) * 55000. -/
theorem partnership_profit_share (a b c : ℝ) (total_profit : ℝ) : 
  a = 3 * b ∧ a = (2/3) * c ∧ total_profit = 55000 → 
  c * total_profit / (a + b + c) = (9/17) * 55000 := by
sorry

end NUMINAMATH_CALUDE_partnership_profit_share_l3732_373254


namespace NUMINAMATH_CALUDE_quadratic_complete_square_l3732_373255

theorem quadratic_complete_square (c : ℝ) (h1 : c > 0) :
  (∃ n : ℝ, ∀ x : ℝ, x^2 + c*x + 20 = (x + n)^2 + 12) →
  c = 4 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_complete_square_l3732_373255


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3732_373245

/-- The equation of the tangent line to the circle (x-1)^2 + y^2 = 5 at the point (2, 2) is x + 2y - 6 = 0 -/
theorem tangent_line_equation (x y : ℝ) : 
  (∀ x y, (x - 1)^2 + y^2 = 5) →  -- Circle equation
  (2 - 1)^2 + 2^2 = 5 →           -- Point (2, 2) lies on the circle
  x + 2*y - 6 = 0                 -- Equation of the tangent line
    ↔ 
  ((x - 1)^2 + y^2 = 5 → (x - 2) + 2*(y - 2) = 0) -- Tangent line property
  :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3732_373245


namespace NUMINAMATH_CALUDE_range_of_m_l3732_373227

-- Define proposition p
def p (m : ℝ) : Prop :=
  ∃ a b : ℝ, a > b ∧ a^2 = m/2 ∧ b^2 = m/2 - 1

-- Define proposition q
def q (m : ℝ) : Prop :=
  ∀ x : ℝ, 4*x^2 - 4*m*x + 4*m - 3 ≥ 0

-- Theorem statement
theorem range_of_m :
  ∃ m_min m_max : ℝ,
    m_min = 1 ∧ m_max = 2 ∧
    ∀ m : ℝ, (¬(p m) ∧ q m) ↔ m_min ≤ m ∧ m ≤ m_max :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3732_373227


namespace NUMINAMATH_CALUDE_cube_gt_of_gt_l3732_373290

theorem cube_gt_of_gt (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_gt_of_gt_l3732_373290


namespace NUMINAMATH_CALUDE_inequality_proof_l3732_373234

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  1/a + 1/b + 4/c + 16/d ≥ 64/(a + b + c + d) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3732_373234


namespace NUMINAMATH_CALUDE_negation_of_forall_geq_one_l3732_373220

theorem negation_of_forall_geq_one :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ 1) ↔ (∃ x : ℝ, x^2 + 1 < 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_geq_one_l3732_373220


namespace NUMINAMATH_CALUDE_smallest_common_factor_l3732_373285

theorem smallest_common_factor (n : ℕ) : n = 85 ↔ 
  (n > 0 ∧ 
   ∃ (k : ℕ), k > 1 ∧ k ∣ (11*n - 4) ∧ k ∣ (8*n + 6) ∧
   ∀ (m : ℕ), m < n → 
     (∀ (j : ℕ), j > 1 → ¬(j ∣ (11*m - 4) ∧ j ∣ (8*m + 6)))) := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_factor_l3732_373285


namespace NUMINAMATH_CALUDE_range_of_a_for_sqrt_function_l3732_373297

theorem range_of_a_for_sqrt_function (a : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = Real.sqrt (2^x - a)) → a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_sqrt_function_l3732_373297


namespace NUMINAMATH_CALUDE_james_carrot_sticks_l3732_373282

/-- The number of carrot sticks James ate before dinner -/
def before_dinner : ℕ := 22

/-- The number of carrot sticks James ate after dinner -/
def after_dinner : ℕ := 15

/-- The total number of carrot sticks James ate -/
def total_carrot_sticks : ℕ := before_dinner + after_dinner

theorem james_carrot_sticks : total_carrot_sticks = 37 := by
  sorry

end NUMINAMATH_CALUDE_james_carrot_sticks_l3732_373282


namespace NUMINAMATH_CALUDE_children_attending_show_l3732_373218

/-- Proves that the number of children attending the show is 3 --/
theorem children_attending_show :
  let adult_ticket_price : ℕ := 12
  let child_ticket_price : ℕ := 10
  let num_adults : ℕ := 3
  let total_cost : ℕ := 66
  let num_children : ℕ := (total_cost - num_adults * adult_ticket_price) / child_ticket_price
  num_children = 3 := by
sorry


end NUMINAMATH_CALUDE_children_attending_show_l3732_373218


namespace NUMINAMATH_CALUDE_typing_service_problem_l3732_373246

/-- The typing service problem -/
theorem typing_service_problem 
  (total_pages : ℕ) 
  (twice_revised_pages : ℕ) 
  (initial_cost : ℕ) 
  (revision_cost : ℕ) 
  (total_cost : ℕ) 
  (h1 : total_pages = 100)
  (h2 : twice_revised_pages = 20)
  (h3 : initial_cost = 5)
  (h4 : revision_cost = 3)
  (h5 : total_cost = 710) :
  ∃ (once_revised_pages : ℕ),
    once_revised_pages = 30 ∧
    total_cost = 
      initial_cost * total_pages + 
      revision_cost * once_revised_pages + 
      2 * revision_cost * twice_revised_pages :=
by
  sorry


end NUMINAMATH_CALUDE_typing_service_problem_l3732_373246


namespace NUMINAMATH_CALUDE_peaches_eaten_l3732_373257

/-- Represents the state of peaches in a bowl --/
structure PeachBowl where
  total : ℕ
  ripe : ℕ
  unripe : ℕ

/-- Calculates the state of peaches after a given number of days --/
def ripenPeaches (initial : PeachBowl) (days : ℕ) (ripeningRate : ℕ) : PeachBowl :=
  { total := initial.total,
    ripe := min initial.total (initial.ripe + days * ripeningRate),
    unripe := max 0 (initial.total - (initial.ripe + days * ripeningRate)) }

/-- Theorem stating the number of peaches eaten --/
theorem peaches_eaten 
  (initial : PeachBowl)
  (ripeningRate : ℕ)
  (days : ℕ)
  (finalState : PeachBowl)
  (h1 : initial.total = 18)
  (h2 : initial.ripe = 4)
  (h3 : ripeningRate = 2)
  (h4 : days = 5)
  (h5 : finalState.ripe = finalState.unripe + 7)
  (h6 : finalState.total + 3 = (ripenPeaches initial days ripeningRate).total) :
  3 = initial.total - finalState.total :=
by
  sorry


end NUMINAMATH_CALUDE_peaches_eaten_l3732_373257


namespace NUMINAMATH_CALUDE_gcd_45_75_l3732_373215

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_45_75_l3732_373215


namespace NUMINAMATH_CALUDE_exists_column_with_n_colors_l3732_373233

/-- Represents a color in the grid -/
structure Color where
  id : Nat

/-- Represents a cell in the grid -/
structure Cell where
  row : Nat
  col : Nat
  color : Color

/-- Represents the grid -/
structure Grid where
  size : Nat
  cells : List Cell

/-- Checks if a subgrid contains all n^2 colors -/
def containsAllColors (g : Grid) (n : Nat) (startRow : Nat) (startCol : Nat) : Prop :=
  ∀ c : Color, ∃ cell ∈ g.cells, 
    cell.row ≥ startRow ∧ cell.row < startRow + n ∧
    cell.col ≥ startCol ∧ cell.col < startCol + n ∧
    cell.color = c

/-- Checks if a row contains n distinct colors -/
def rowHasNColors (g : Grid) (n : Nat) (row : Nat) : Prop :=
  ∃ colors : List Color, colors.length = n ∧
    (∀ c ∈ colors, ∃ cell ∈ g.cells, cell.row = row ∧ cell.color = c) ∧
    (∀ cell ∈ g.cells, cell.row = row → cell.color ∈ colors)

/-- Checks if a column contains exactly n distinct colors -/
def columnHasExactlyNColors (g : Grid) (n : Nat) (col : Nat) : Prop :=
  ∃ colors : List Color, colors.length = n ∧
    (∀ c ∈ colors, ∃ cell ∈ g.cells, cell.col = col ∧ cell.color = c) ∧
    (∀ cell ∈ g.cells, cell.col = col → cell.color ∈ colors)

/-- The main theorem -/
theorem exists_column_with_n_colors (g : Grid) (n : Nat) :
  (∃ m : Nat, g.size = m * n) →
  (∀ i j : Nat, i < g.size - n + 1 → j < g.size - n + 1 → containsAllColors g n i j) →
  (∃ row : Nat, row < g.size ∧ rowHasNColors g n row) →
  (∃ col : Nat, col < g.size ∧ columnHasExactlyNColors g n col) :=
by sorry

end NUMINAMATH_CALUDE_exists_column_with_n_colors_l3732_373233


namespace NUMINAMATH_CALUDE_f_max_value_f_solution_set_max_ab_plus_bc_l3732_373231

-- Define the function f
def f (x : ℝ) : ℝ := |x - 3| - 2 * |x + 1|

-- Theorem for the maximum value of f
theorem f_max_value : ∃ m : ℝ, m = 4 ∧ ∀ x : ℝ, f x ≤ m :=
sorry

-- Theorem for the solution set of f(x) < 1
theorem f_solution_set : ∀ x : ℝ, f x < 1 ↔ x < -4 ∨ x > 0 :=
sorry

-- Theorem for the maximum value of ab + bc
theorem max_ab_plus_bc :
  ∀ a b c : ℝ, a > 0 → b > 0 → a^2 + 2*b^2 + c^2 = 4 →
  ∃ max : ℝ, max = 2 ∧ a*b + b*c ≤ max :=
sorry

end NUMINAMATH_CALUDE_f_max_value_f_solution_set_max_ab_plus_bc_l3732_373231


namespace NUMINAMATH_CALUDE_books_bought_at_yard_sale_l3732_373269

theorem books_bought_at_yard_sale 
  (initial_books : ℕ) 
  (final_books : ℕ) 
  (h1 : initial_books = 35)
  (h2 : final_books = 56) :
  final_books - initial_books = 21 :=
by sorry

end NUMINAMATH_CALUDE_books_bought_at_yard_sale_l3732_373269


namespace NUMINAMATH_CALUDE_no_valid_A_l3732_373212

theorem no_valid_A : ¬∃ (A : ℕ), A < 10 ∧ 75 % A = 0 ∧ (5361000 + 100 * A + 4) % 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_A_l3732_373212


namespace NUMINAMATH_CALUDE_cupcake_packages_l3732_373214

theorem cupcake_packages (initial_cupcakes : ℕ) (eaten_cupcakes : ℕ) (cupcakes_per_package : ℕ) :
  initial_cupcakes = 18 →
  eaten_cupcakes = 8 →
  cupcakes_per_package = 2 →
  (initial_cupcakes - eaten_cupcakes) / cupcakes_per_package = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_cupcake_packages_l3732_373214


namespace NUMINAMATH_CALUDE_max_product_xyz_l3732_373251

theorem max_product_xyz (x y z : ℝ) 
  (hx : x ≥ 20) (hy : y ≥ 40) (hz : z ≥ 1675) (hsum : x + y + z = 2015) : 
  x * y * z ≤ 721480000 / 27 := by
  sorry

end NUMINAMATH_CALUDE_max_product_xyz_l3732_373251


namespace NUMINAMATH_CALUDE_log_50_between_consecutive_integers_l3732_373241

theorem log_50_between_consecutive_integers : 
  ∃ (m n : ℤ), m + 1 = n ∧ (m : ℝ) < Real.log 50 / Real.log 10 ∧ Real.log 50 / Real.log 10 < n ∧ m + n = 3 := by
sorry

end NUMINAMATH_CALUDE_log_50_between_consecutive_integers_l3732_373241


namespace NUMINAMATH_CALUDE_degree_of_monomial_l3732_373289

/-- The degree of a monomial is the sum of the exponents of its variables -/
def monomialDegree (coefficient : ℤ) (xExponent yExponent : ℕ) : ℕ :=
  xExponent + yExponent

/-- The monomial -3x^5y^2 has degree 7 -/
theorem degree_of_monomial :
  monomialDegree (-3) 5 2 = 7 := by sorry

end NUMINAMATH_CALUDE_degree_of_monomial_l3732_373289


namespace NUMINAMATH_CALUDE_inequality_proof_l3732_373232

theorem inequality_proof (m n : ℕ) (h : m < n) :
  m^2 + Real.sqrt (m^2 + m) < n^2 - Real.sqrt (n^2 - n) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3732_373232


namespace NUMINAMATH_CALUDE_weekend_pie_revenue_l3732_373242

structure PieSlice where
  name : String
  slices_per_pie : ℕ
  price_per_slice : ℕ
  customers : ℕ

def apple_pie : PieSlice := {
  name := "Apple",
  slices_per_pie := 8,
  price_per_slice := 3,
  customers := 88
}

def peach_pie : PieSlice := {
  name := "Peach",
  slices_per_pie := 6,
  price_per_slice := 4,
  customers := 78
}

def cherry_pie : PieSlice := {
  name := "Cherry",
  slices_per_pie := 10,
  price_per_slice := 5,
  customers := 45
}

def revenue (pie : PieSlice) : ℕ :=
  pie.customers * pie.price_per_slice

def total_revenue (pies : List PieSlice) : ℕ :=
  pies.foldl (fun acc pie => acc + revenue pie) 0

theorem weekend_pie_revenue :
  total_revenue [apple_pie, peach_pie, cherry_pie] = 801 := by
  sorry

end NUMINAMATH_CALUDE_weekend_pie_revenue_l3732_373242


namespace NUMINAMATH_CALUDE_train_crossing_time_l3732_373275

/-- Calculates the time for a train to cross a signal pole given its length, 
    the platform length, and the time to cross the platform. -/
theorem train_crossing_time 
  (train_length : ℝ) 
  (platform_length : ℝ) 
  (platform_crossing_time : ℝ) 
  (h1 : train_length = 300)
  (h2 : platform_length = 550.0000000000001)
  (h3 : platform_crossing_time = 51) :
  (train_length / ((train_length + platform_length) / platform_crossing_time)) = 18 := by
sorry

end NUMINAMATH_CALUDE_train_crossing_time_l3732_373275


namespace NUMINAMATH_CALUDE_intersection_A_B_range_of_m_when_A_subset_C_l3732_373239

-- Define sets A, B, and C
def A : Set ℝ := {x | x^2 - 5*x - 6 < 0}
def B : Set ℝ := {x | 6*x^2 - 5*x + 1 ≥ 0}
def C (m : ℝ) : Set ℝ := {x | (x - m)*(x - m - 9) < 0}

-- Theorem for the intersection of A and B
theorem intersection_A_B :
  A ∩ B = {x | -1 < x ∧ x ≤ 1/3 ∨ 1/2 ≤ x ∧ x < 6} :=
sorry

-- Theorem for the range of m when A is a subset of C
theorem range_of_m_when_A_subset_C :
  (∀ m : ℝ, A ⊆ C m → -3 ≤ m ∧ m ≤ -1) ∧
  (∀ m : ℝ, -3 ≤ m ∧ m ≤ -1 → A ⊆ C m) :=
sorry

end NUMINAMATH_CALUDE_intersection_A_B_range_of_m_when_A_subset_C_l3732_373239


namespace NUMINAMATH_CALUDE_simplify_expression_l3732_373230

theorem simplify_expression (x : ℝ) : 1 - (2 * (1 - (1 + (1 - (3 - x))))) = -3 + 2*x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3732_373230


namespace NUMINAMATH_CALUDE_chocolate_sales_l3732_373263

theorem chocolate_sales (cost_price selling_price : ℝ) (n : ℕ) : 
  44 * cost_price = n * selling_price →
  selling_price = cost_price * (1 + 5/6) →
  n = 24 := by
sorry

end NUMINAMATH_CALUDE_chocolate_sales_l3732_373263


namespace NUMINAMATH_CALUDE_bridge_building_time_l3732_373224

/-- Represents the time taken to build a bridge given a number of workers -/
def build_time (workers : ℕ) : ℝ := sorry

/-- The constant representing the total work required -/
def total_work : ℝ := 18 * 6

theorem bridge_building_time :
  (build_time 18 = 6) →
  (∀ w₁ w₂ : ℕ, w₁ * build_time w₁ = w₂ * build_time w₂) →
  build_time 30 = 3.6 := by sorry

end NUMINAMATH_CALUDE_bridge_building_time_l3732_373224


namespace NUMINAMATH_CALUDE_existence_of_unequal_indices_l3732_373288

theorem existence_of_unequal_indices (a b c : ℕ → ℕ) : 
  ∃ m n : ℕ, m ≠ n ∧ a m ≥ a n ∧ b m ≥ b n ∧ c m ≥ c n := by
  sorry

end NUMINAMATH_CALUDE_existence_of_unequal_indices_l3732_373288


namespace NUMINAMATH_CALUDE_root_existence_iff_a_ge_three_l3732_373221

/-- The function f(x) = ln x + x + 2/x - a has a root for some x > 0 if and only if a ≥ 3 -/
theorem root_existence_iff_a_ge_three (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ Real.log x + x + 2 / x - a = 0) ↔ a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_root_existence_iff_a_ge_three_l3732_373221


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3732_373264

def A : Set ℝ := {x | x^2 - 2*x ≤ 0}
def B : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3732_373264


namespace NUMINAMATH_CALUDE_bird_watching_percentage_difference_l3732_373291

def gabrielle_robins : ℕ := 5
def gabrielle_cardinals : ℕ := 4
def gabrielle_blue_jays : ℕ := 3

def chase_robins : ℕ := 2
def chase_blue_jays : ℕ := 3
def chase_cardinals : ℕ := 5

def gabrielle_total : ℕ := gabrielle_robins + gabrielle_cardinals + gabrielle_blue_jays
def chase_total : ℕ := chase_robins + chase_blue_jays + chase_cardinals

theorem bird_watching_percentage_difference :
  (gabrielle_total - chase_total : ℚ) / chase_total * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_bird_watching_percentage_difference_l3732_373291
