import Mathlib

namespace trigonometric_inequality_l3457_345705

theorem trigonometric_inequality (x : ℝ) : 
  (9.276 * Real.sin (2 * x) * Real.sin (3 * x) - Real.cos (2 * x) * Real.cos (3 * x) > Real.sin (10 * x)) ↔ 
  (∃ n : ℤ, ((-Real.pi / 10 + 2 * Real.pi * n / 5 < x ∧ x < -Real.pi / 30 + 2 * Real.pi * n) ∨ 
             (Real.pi / 10 + 2 * Real.pi * n / 5 < x ∧ x < 7 * Real.pi / 30 + 2 * Real.pi * n))) :=
by sorry

end trigonometric_inequality_l3457_345705


namespace string_average_length_l3457_345703

theorem string_average_length :
  let string1 : ℚ := 4
  let string2 : ℚ := 5
  let string3 : ℚ := 7
  let num_strings : ℕ := 3
  (string1 + string2 + string3) / num_strings = 16 / 3 := by
  sorry

end string_average_length_l3457_345703


namespace thirty_six_is_triangular_and_square_l3457_345731

/-- Definition of triangular numbers -/
def is_triangular (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * (k + 1) / 2

/-- Definition of square numbers -/
def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2

/-- Theorem: 36 is both a triangular number and a square number -/
theorem thirty_six_is_triangular_and_square :
  is_triangular 36 ∧ is_square 36 :=
sorry

end thirty_six_is_triangular_and_square_l3457_345731


namespace tic_tac_toe_winning_probability_l3457_345759

/-- A tic-tac-toe board is a 3x3 grid. -/
def TicTacToeBoard := Fin 3 → Fin 3 → Bool

/-- A winning position is a line (row, column, or diagonal) on the board. -/
def WinningPosition : Type := List (Fin 3 × Fin 3)

/-- The set of all winning positions on a tic-tac-toe board. -/
def allWinningPositions : List WinningPosition :=
  -- 3 horizontal lines
  [[(0,0), (0,1), (0,2)], [(1,0), (1,1), (1,2)], [(2,0), (2,1), (2,2)]] ++
  -- 3 vertical lines
  [[(0,0), (1,0), (2,0)], [(0,1), (1,1), (2,1)], [(0,2), (1,2), (2,2)]] ++
  -- 2 diagonal lines
  [[(0,0), (1,1), (2,2)], [(0,2), (1,1), (2,0)]]

/-- The number of ways to arrange 3 noughts on a 3x3 board. -/
def totalArrangements : ℕ := 84

/-- The probability of three noughts being in a winning position. -/
def winningProbability : ℚ := 2 / 21

theorem tic_tac_toe_winning_probability :
  (List.length allWinningPositions : ℚ) / totalArrangements = winningProbability := by
  sorry

end tic_tac_toe_winning_probability_l3457_345759


namespace line_intersects_circle_l3457_345788

/-- 
Given a > 0, prove that the line x + a²y - a = 0 intersects 
the circle (x - a)² + (y - 1/a)² = 1
-/
theorem line_intersects_circle (a : ℝ) (h : a > 0) : 
  ∃ (x y : ℝ), (x + a^2 * y - a = 0) ∧ ((x - a)^2 + (y - 1/a)^2 = 1) :=
by sorry

end line_intersects_circle_l3457_345788


namespace impossible_division_l3457_345773

/-- Represents a chess-like board with alternating colors -/
def Board := Fin 8 → Fin 8 → Bool

/-- Represents an L-shaped piece on the board -/
structure LPiece :=
  (x : Fin 8) (y : Fin 8)

/-- Checks if an L-piece is valid (within bounds and not in the cut-out corner) -/
def isValidPiece (b : Board) (p : LPiece) : Prop :=
  p.x < 6 ∧ p.y < 6 ∧ ¬(p.x = 0 ∧ p.y = 0)

/-- Counts the number of squares of each color covered by an L-piece -/
def colorCount (b : Board) (p : LPiece) : Nat × Nat :=
  let trueCount := (b p.x p.y).toNat + (b p.x (p.y + 1)).toNat + (b (p.x + 1) p.y).toNat + (b (p.x + 1) (p.y + 1)).toNat
  (trueCount, 4 - trueCount)

/-- The main theorem stating that it's impossible to divide the board as required -/
theorem impossible_division (b : Board) : ¬ ∃ (pieces : List LPiece),
  pieces.length = 15 ∧ 
  (∀ p ∈ pieces, isValidPiece b p) ∧
  (∀ p ∈ pieces, (colorCount b p).1 = 3 ∨ (colorCount b p).2 = 3) ∧
  (pieces.map (λ p => (colorCount b p).1)).sum = 30 :=
sorry

end impossible_division_l3457_345773


namespace cubic_sum_minus_product_l3457_345717

theorem cubic_sum_minus_product (x y z : ℝ) 
  (h1 : x + y + z = 10) 
  (h2 : x*y + x*z + y*z = 30) : 
  x^3 + y^3 + z^3 - 3*x*y*z = 100 := by
  sorry

end cubic_sum_minus_product_l3457_345717


namespace solution_set_f_less_than_4_range_of_a_for_f_geq_abs_a_plus_1_l3457_345782

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| + 2 * |x - 1|

-- Theorem for part I
theorem solution_set_f_less_than_4 :
  {x : ℝ | f x < 4} = {x : ℝ | -1 < x ∧ x < 5/3} := by sorry

-- Theorem for part II
theorem range_of_a_for_f_geq_abs_a_plus_1 :
  (∀ x : ℝ, f x ≥ |a + 1|) ↔ -3 ≤ a ∧ a ≤ 1 := by sorry

end solution_set_f_less_than_4_range_of_a_for_f_geq_abs_a_plus_1_l3457_345782


namespace salary_sum_l3457_345728

/-- Given 5 individuals with an average salary and one known salary, 
    prove the sum of the other four salaries. -/
theorem salary_sum (average_salary : ℕ) (known_salary : ℕ) : 
  average_salary = 8800 → known_salary = 8000 → 
  (5 * average_salary) - known_salary = 36000 := by
  sorry

end salary_sum_l3457_345728


namespace oranges_from_joyce_calculation_l3457_345779

/-- Represents the number of oranges Clarence has initially. -/
def initial_oranges : ℕ := 5

/-- Represents the total number of oranges Clarence has after receiving some from Joyce. -/
def total_oranges : ℕ := 8

/-- Represents the number of oranges Joyce gave to Clarence. -/
def oranges_from_joyce : ℕ := total_oranges - initial_oranges

/-- Proves that the number of oranges Joyce gave to Clarence is equal to the difference
    between Clarence's total oranges after receiving from Joyce and Clarence's initial oranges. -/
theorem oranges_from_joyce_calculation :
  oranges_from_joyce = total_oranges - initial_oranges :=
by sorry

end oranges_from_joyce_calculation_l3457_345779


namespace middle_integer_of_three_consecutive_l3457_345775

/-- Given three consecutive integers whose sum is 360, the middle integer is 120. -/
theorem middle_integer_of_three_consecutive (n : ℤ) : 
  (n - 1) + n + (n + 1) = 360 → n = 120 := by
  sorry

end middle_integer_of_three_consecutive_l3457_345775


namespace shooting_test_probability_l3457_345776

/-- The probability of a successful shot -/
def p : ℚ := 2/3

/-- The number of successful shots required to pass -/
def required_successes : ℕ := 3

/-- The maximum number of shots allowed -/
def max_shots : ℕ := 5

/-- The probability of passing the shooting test -/
def pass_probability : ℚ := 64/81

theorem shooting_test_probability :
  (p^required_successes) +
  (Nat.choose 4 required_successes * p^required_successes * (1-p)) +
  (Nat.choose 5 required_successes * p^required_successes * (1-p)^2) = pass_probability :=
sorry

end shooting_test_probability_l3457_345776


namespace words_with_vowels_count_l3457_345780

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def vowels : Finset Char := {'A', 'E'}
def consonants : Finset Char := alphabet \ vowels
def word_length : Nat := 5

def total_words : Nat := alphabet.card ^ word_length
def consonant_words : Nat := consonants.card ^ word_length
def words_with_vowels : Nat := total_words - consonant_words

theorem words_with_vowels_count : words_with_vowels = 6752 := by sorry

end words_with_vowels_count_l3457_345780


namespace stream_speed_proof_l3457_345741

/-- Proves that the speed of a stream is 5 km/hr given the conditions of a boat's travel --/
theorem stream_speed_proof (boat_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  boat_speed = 25 →
  downstream_distance = 120 →
  downstream_time = 4 →
  ∃ stream_speed : ℝ, stream_speed = 5 ∧ downstream_distance = (boat_speed + stream_speed) * downstream_time :=
by
  sorry

#check stream_speed_proof

end stream_speed_proof_l3457_345741


namespace simplified_fraction_sum_l3457_345704

theorem simplified_fraction_sum (c d : ℕ+) : 
  (c : ℚ) / d = 0.375 ∧ 
  ∀ (a b : ℕ+), (a : ℚ) / b = 0.375 → c ≤ a ∧ d ≤ b → 
  c + d = 11 := by sorry

end simplified_fraction_sum_l3457_345704


namespace farm_corn_cobs_l3457_345753

/-- Calculates the total number of corn cobs grown on a farm with two fields -/
def total_corn_cobs (field1_rows : ℕ) (field2_rows : ℕ) (cobs_per_row : ℕ) : ℕ :=
  (field1_rows * cobs_per_row) + (field2_rows * cobs_per_row)

/-- Theorem stating that the total number of corn cobs on the farm is 116 -/
theorem farm_corn_cobs : total_corn_cobs 13 16 4 = 116 := by
  sorry

end farm_corn_cobs_l3457_345753


namespace checkerboard_fraction_sum_l3457_345784

/-- The number of squares in a n×n grid -/
def squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- The number of rectangles in a (n+1)×(n+1) grid -/
def rectangles (n : ℕ) : ℕ := (n * (n + 1) / 2)^2

theorem checkerboard_fraction_sum :
  let s := squares 7
  let r := rectangles 7
  let g := Nat.gcd s r
  (s / g) + (r / g) = 33 := by sorry

end checkerboard_fraction_sum_l3457_345784


namespace prism_volume_l3457_345721

/-- The volume of a right rectangular prism with specific face areas and dimension ratio -/
theorem prism_volume (a b c : ℝ) : 
  a * b = 64 → 
  b * c = 81 → 
  a * c = 72 → 
  b = 2 * a → 
  |a * b * c - 1629| < 1 := by
sorry

end prism_volume_l3457_345721


namespace different_colors_probability_l3457_345734

structure Box where
  red : ℕ
  black : ℕ
  white : ℕ
  yellow : ℕ

def boxA : Box := { red := 3, black := 3, white := 3, yellow := 0 }
def boxB : Box := { red := 0, black := 2, white := 2, yellow := 2 }

def totalBalls (box : Box) : ℕ := box.red + box.black + box.white + box.yellow

def probabilityDifferentColors (boxA boxB : Box) : ℚ :=
  let totalA := totalBalls boxA
  let totalB := totalBalls boxB
  let sameColor := boxA.black * boxB.black + boxA.white * boxB.white
  (totalA * totalB - sameColor) / (totalA * totalB)

theorem different_colors_probability :
  probabilityDifferentColors boxA boxB = 2 / 9 := by
  sorry

end different_colors_probability_l3457_345734


namespace lisas_teaspoons_l3457_345789

theorem lisas_teaspoons (num_children : ℕ) (baby_spoons_per_child : ℕ) (decorative_spoons : ℕ) 
  (large_spoons : ℕ) (total_spoons : ℕ) :
  num_children = 4 →
  baby_spoons_per_child = 3 →
  decorative_spoons = 2 →
  large_spoons = 10 →
  total_spoons = 39 →
  total_spoons - (num_children * baby_spoons_per_child + decorative_spoons + large_spoons) = 15 := by
  sorry

end lisas_teaspoons_l3457_345789


namespace cube_root_of_negative_eight_l3457_345774

theorem cube_root_of_negative_eight (x : ℝ) : x^3 = -8 → x = -2 := by
  sorry

end cube_root_of_negative_eight_l3457_345774


namespace imaginary_part_of_z_l3457_345724

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) : 
  Complex.im (i * (i - 3)) = -3 := by sorry

end imaginary_part_of_z_l3457_345724


namespace sam_watermelons_l3457_345754

/-- The number of watermelons Sam grew initially -/
def initial_watermelons : ℕ := 4

/-- The number of additional watermelons Sam grew -/
def additional_watermelons : ℕ := 3

/-- The total number of watermelons Sam has -/
def total_watermelons : ℕ := initial_watermelons + additional_watermelons

theorem sam_watermelons : total_watermelons = 7 := by
  sorry

end sam_watermelons_l3457_345754


namespace minimum_distance_to_exponential_curve_l3457_345783

open Real

theorem minimum_distance_to_exponential_curve (a : ℝ) :
  (∃ x₀ : ℝ, (x₀ - a)^2 + (exp x₀ - a)^2 ≤ 1/2) → a = 1/2 := by
  sorry

end minimum_distance_to_exponential_curve_l3457_345783


namespace antifreeze_solution_l3457_345794

def antifreeze_problem (x : ℝ) : Prop :=
  let solution1_percent : ℝ := 10
  let total_volume : ℝ := 20
  let target_percent : ℝ := 15
  let volume_each : ℝ := 7.5
  (solution1_percent * volume_each + x * volume_each) / total_volume = target_percent

theorem antifreeze_solution : 
  ∃ x : ℝ, antifreeze_problem x ∧ x = 30 := by
sorry

end antifreeze_solution_l3457_345794


namespace vasya_fish_count_l3457_345710

/-- Represents the number of fish Vasya caught -/
def total_fish : ℕ := 10

/-- Represents the weight of the three largest fish as a fraction of the total catch -/
def largest_fish_weight_fraction : ℚ := 35 / 100

/-- Represents the weight of the three smallest fish as a fraction of the remaining catch -/
def smallest_fish_weight_fraction : ℚ := 5 / 13

/-- Represents the number of largest fish -/
def num_largest_fish : ℕ := 3

/-- Represents the number of smallest fish -/
def num_smallest_fish : ℕ := 3

theorem vasya_fish_count :
  ∃ (x : ℕ),
    total_fish = num_largest_fish + x + num_smallest_fish ∧
    (1 - largest_fish_weight_fraction) * smallest_fish_weight_fraction = 
      (25 : ℚ) / 100 ∧
    (35 : ℚ) / 3 ≤ (40 : ℚ) / x ∧
    (40 : ℚ) / x ≤ (25 : ℚ) / 3 :=
sorry

end vasya_fish_count_l3457_345710


namespace function_inequality_condition_l3457_345730

theorem function_inequality_condition (k : ℝ) : 
  (∀ (a x₁ x₂ : ℝ), 1 ≤ a ∧ a ≤ 2 ∧ 2 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 4 →
    |x₁ + a / x₁ - 4| - |x₂ + a / x₂ - 4| < k * (x₁ - x₂)) ↔
  k ≤ 6 - 4 * Real.sqrt 3 :=
by sorry

end function_inequality_condition_l3457_345730


namespace locus_is_ellipse_l3457_345723

/-- A complex number z tracing a circle centered at the origin with radius 3 -/
def z_on_circle (z : ℂ) : Prop := Complex.abs z = 3

/-- The locus of points (x, y) satisfying x + yi = z + 1/z -/
def locus (z : ℂ) (x y : ℝ) : Prop := x + y * Complex.I = z + 1 / z

/-- The equation of an ellipse in standard form -/
def is_ellipse (x y : ℝ) : Prop := ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ x^2 / a^2 + y^2 / b^2 = 1

theorem locus_is_ellipse :
  ∀ z : ℂ, z_on_circle z →
  ∀ x y : ℝ, locus z x y →
  is_ellipse x y :=
sorry

end locus_is_ellipse_l3457_345723


namespace tricycle_count_l3457_345743

theorem tricycle_count (num_bicycles : ℕ) (bicycle_wheels : ℕ) (tricycle_wheels : ℕ) (total_wheels : ℕ) :
  num_bicycles = 16 →
  bicycle_wheels = 2 →
  tricycle_wheels = 3 →
  total_wheels = 53 →
  ∃ num_tricycles : ℕ, num_bicycles * bicycle_wheels + num_tricycles * tricycle_wheels = total_wheels ∧ num_tricycles = 7 :=
by sorry

end tricycle_count_l3457_345743


namespace sandwich_meat_cost_l3457_345718

/-- The cost of a pack of sandwich meat given the following conditions:
  * 1 loaf of bread, 2 packs of sandwich meat, and 2 packs of sliced cheese make 10 sandwiches
  * Bread costs $4.00
  * Cheese costs $4.00 per pack
  * There's a $1.00 off coupon for one pack of cheese
  * There's a $1.00 off coupon for one pack of meat
  * Each sandwich costs $2.00
-/
theorem sandwich_meat_cost :
  let bread_cost : ℚ := 4
  let cheese_cost : ℚ := 4
  let cheese_discount : ℚ := 1
  let meat_discount : ℚ := 1
  let sandwich_cost : ℚ := 2
  let sandwich_count : ℕ := 10
  let total_cost : ℚ := sandwich_cost * sandwich_count
  let cheese_total : ℚ := 2 * cheese_cost - cheese_discount
  ∃ meat_cost : ℚ,
    bread_cost + cheese_total + 2 * meat_cost - meat_discount = total_cost ∧
    meat_cost = 5 :=
by sorry

end sandwich_meat_cost_l3457_345718


namespace sufficient_not_necessary_l3457_345744

theorem sufficient_not_necessary :
  (∀ x y : ℝ, x > 1 ∧ y > 1 → x + y > 2) ∧
  (∃ x y : ℝ, x + y > 2 ∧ ¬(x > 1 ∧ y > 1)) :=
by sorry

end sufficient_not_necessary_l3457_345744


namespace investment_loss_calculation_l3457_345756

/-- Represents the capital and loss of two investors -/
structure InvestmentScenario where
  capital_ratio : ℚ  -- Ratio of smaller capital to larger capital
  larger_loss : ℚ    -- Loss of the investor with larger capital
  total_loss : ℚ     -- Total loss of both investors

/-- Theorem stating the relationship between capital ratio, larger investor's loss, and total loss -/
theorem investment_loss_calculation (scenario : InvestmentScenario) 
  (h1 : scenario.capital_ratio = 1 / 9)
  (h2 : scenario.larger_loss = 1080) :
  scenario.total_loss = 1200 := by
  sorry

end investment_loss_calculation_l3457_345756


namespace triangle_height_l3457_345796

theorem triangle_height (area : ℝ) (base : ℝ) (height : ℝ) :
  area = 46 →
  base = 10 →
  area = (base * height) / 2 →
  height = 9.2 := by
sorry

end triangle_height_l3457_345796


namespace a_zero_sufficient_not_necessary_for_ab_zero_l3457_345781

theorem a_zero_sufficient_not_necessary_for_ab_zero :
  (∃ a b : ℝ, a = 0 → a * b = 0) ∧
  (∃ a b : ℝ, a * b = 0 ∧ a ≠ 0) :=
by sorry

end a_zero_sufficient_not_necessary_for_ab_zero_l3457_345781


namespace geometric_sequence_fifth_term_l3457_345777

/-- A geometric sequence of positive integers with first term 3 and fourth term 243 has fifth term 243. -/
theorem geometric_sequence_fifth_term :
  ∀ (a : ℕ → ℕ),
  (∀ n : ℕ, a (n + 1) / a n = a 2 / a 1) →  -- Geometric sequence condition
  (∀ n : ℕ, a n > 0) →  -- Positive integer condition
  a 1 = 3 →  -- First term is 3
  a 4 = 243 →  -- Fourth term is 243
  a 5 = 243 :=  -- Fifth term is 243
by
  sorry

end geometric_sequence_fifth_term_l3457_345777


namespace sin_arctan_x_equals_x_l3457_345711

theorem sin_arctan_x_equals_x (x : ℝ) :
  x > 0 →
  Real.sin (Real.arctan x) = x →
  x^4 = (3 - Real.sqrt 5) / 2 := by
sorry

end sin_arctan_x_equals_x_l3457_345711


namespace smallest_n_divisible_by_68_l3457_345786

theorem smallest_n_divisible_by_68 :
  ∃ (n : ℕ), n^2 + 14*n + 13 ≡ 0 [MOD 68] ∧
  (∀ (m : ℕ), m < n → ¬(m^2 + 14*m + 13 ≡ 0 [MOD 68])) ∧
  n = 21 := by
  sorry

end smallest_n_divisible_by_68_l3457_345786


namespace percentage_error_calculation_l3457_345713

theorem percentage_error_calculation (N : ℝ) (h : N > 0) : 
  let correct := N * 5
  let incorrect := N / 10
  let absolute_error := |correct - incorrect|
  let percentage_error := (absolute_error / correct) * 100
  percentage_error = 98 := by
sorry

end percentage_error_calculation_l3457_345713


namespace candy_bar_earnings_difference_l3457_345768

/-- The problem of calculating the difference in earnings between Tina and Marvin from selling candy bars. -/
theorem candy_bar_earnings_difference :
  let candy_bar_price : ℕ := 2
  let marvins_sales : ℕ := 35
  let tinas_sales : ℕ := 3 * marvins_sales
  let marvins_earnings : ℕ := candy_bar_price * marvins_sales
  let tinas_earnings : ℕ := candy_bar_price * tinas_sales
  tinas_earnings - marvins_earnings = 140 := by
sorry

end candy_bar_earnings_difference_l3457_345768


namespace sqrt_twelve_minus_abs_one_minus_sqrt_three_plus_seven_plus_pi_pow_zero_l3457_345771

theorem sqrt_twelve_minus_abs_one_minus_sqrt_three_plus_seven_plus_pi_pow_zero :
  Real.sqrt 12 - |1 - Real.sqrt 3| + (7 + Real.pi)^0 = Real.sqrt 3 + 2 := by
  sorry

end sqrt_twelve_minus_abs_one_minus_sqrt_three_plus_seven_plus_pi_pow_zero_l3457_345771


namespace lena_tennis_win_probability_l3457_345732

theorem lena_tennis_win_probability :
  ∀ (p_lose : ℚ),
  p_lose = 3/7 →
  (∀ (p_win : ℚ), p_win + p_lose = 1 → p_win = 4/7) :=
by sorry

end lena_tennis_win_probability_l3457_345732


namespace chromatic_number_le_max_degree_plus_one_l3457_345709

/-- A graph is represented by its vertex set and an adjacency relation -/
structure Graph (V : Type*) where
  adj : V → V → Prop

/-- The degree of a vertex in a graph -/
def degree (G : Graph V) (v : V) : ℕ := sorry

/-- The maximum degree of a graph -/
def maxDegree (G : Graph V) : ℕ := sorry

/-- A coloring of a graph is a function from vertices to colors -/
def isColoring (G : Graph V) (f : V → ℕ) : Prop :=
  ∀ u v : V, G.adj u v → f u ≠ f v

/-- The chromatic number of a graph -/
def chromaticNumber (G : Graph V) : ℕ := sorry

/-- Theorem: The chromatic number of a graph is at most one more than its maximum degree -/
theorem chromatic_number_le_max_degree_plus_one (V : Type*) (G : Graph V) :
  chromaticNumber G ≤ maxDegree G + 1 := by sorry

end chromatic_number_le_max_degree_plus_one_l3457_345709


namespace mary_final_cards_l3457_345798

/-- Calculates the final number of baseball cards Mary has after a series of transactions -/
def final_card_count (initial_cards torn_cards fred_cards bought_cards lost_cards lisa_trade_in lisa_trade_out alex_trade_in alex_trade_out : ℕ) : ℕ :=
  initial_cards - torn_cards + fred_cards + bought_cards - lost_cards - lisa_trade_in + lisa_trade_out - alex_trade_in + alex_trade_out

/-- Theorem stating that Mary ends up with 70 baseball cards -/
theorem mary_final_cards : 
  final_card_count 18 8 26 40 5 3 4 7 5 = 70 := by
  sorry

end mary_final_cards_l3457_345798


namespace success_arrangements_l3457_345715

def word_length : ℕ := 7
def s_count : ℕ := 3
def c_count : ℕ := 2
def u_count : ℕ := 1
def e_count : ℕ := 1

theorem success_arrangements : 
  (word_length.factorial) / (s_count.factorial * c_count.factorial * u_count.factorial * e_count.factorial) = 420 :=
sorry

end success_arrangements_l3457_345715


namespace greatest_integer_radius_l3457_345700

theorem greatest_integer_radius (A : ℝ) (h : A < 80 * Real.pi) :
  ∃ (r : ℕ), r * r * Real.pi = A ∧ r ≤ 8 ∧ ∀ (s : ℕ), s * s * Real.pi = A → s ≤ r :=
sorry

end greatest_integer_radius_l3457_345700


namespace expression_evaluation_l3457_345748

theorem expression_evaluation : 
  202.2 * 89.8 - 20.22 * 186 + 2.022 * 3570 - 0.2022 * 16900 = 18198 := by
  sorry

end expression_evaluation_l3457_345748


namespace constant_term_expansion_l3457_345739

def constant_term (n : ℕ) : ℕ :=
  Nat.choose n 0 + 
  Nat.choose n 2 * Nat.choose 2 1 + 
  Nat.choose n 4 * Nat.choose 4 2 + 
  Nat.choose n 6 * Nat.choose 6 3

theorem constant_term_expansion : constant_term 6 = 141 := by
  sorry

end constant_term_expansion_l3457_345739


namespace chocolate_bar_count_l3457_345772

theorem chocolate_bar_count (milk_chocolate dark_chocolate white_chocolate : ℕ) 
  (h1 : milk_chocolate = 25)
  (h2 : dark_chocolate = 25)
  (h3 : white_chocolate = 25)
  (h4 : ∃ (total : ℕ), total > 0 ∧ 
    milk_chocolate = total / 4 ∧ 
    dark_chocolate = total / 4 ∧ 
    white_chocolate = total / 4) :
  ∃ (almond_chocolate : ℕ), almond_chocolate = 25 :=
sorry

end chocolate_bar_count_l3457_345772


namespace total_pencils_l3457_345736

theorem total_pencils (jessica_pencils sandy_pencils jason_pencils : ℕ) 
  (h1 : jessica_pencils = 8)
  (h2 : sandy_pencils = 8)
  (h3 : jason_pencils = 8) :
  jessica_pencils + sandy_pencils + jason_pencils = 24 := by
  sorry

end total_pencils_l3457_345736


namespace quadratic_root_values_l3457_345727

theorem quadratic_root_values (b c : ℝ) : 
  (Complex.I * Real.sqrt 2 + 1) ^ 2 + b * (Complex.I * Real.sqrt 2 + 1) + c = 0 → 
  b = -2 ∧ c = 3 := by
  sorry

end quadratic_root_values_l3457_345727


namespace relationship_abc_l3457_345740

noncomputable def a : ℝ := (1.1 : ℝ) ^ (0.1 : ℝ)
noncomputable def b : ℝ := Real.log 2
noncomputable def c : ℝ := Real.log (Real.sqrt 3 / 3) / Real.log (1/3)

theorem relationship_abc : a > b ∧ b > c := by sorry

end relationship_abc_l3457_345740


namespace marble_probability_l3457_345766

/-- Given a box of 100 marbles with specified probabilities for white and green marbles,
    prove that the probability of drawing either a red or blue marble is 11/20. -/
theorem marble_probability (total : ℕ) (p_white p_green : ℚ) 
    (h_total : total = 100)
    (h_white : p_white = 1 / 4)
    (h_green : p_green = 1 / 5) :
    (total - (p_white * total + p_green * total)) / total = 11 / 20 := by
  sorry

end marble_probability_l3457_345766


namespace coefficient_x_cubed_expansion_l3457_345750

theorem coefficient_x_cubed_expansion : 
  let expansion := (1 + X : Polynomial ℤ)^5 - (1 + X : Polynomial ℤ)^6
  expansion.coeff 3 = -10 := by
  sorry

end coefficient_x_cubed_expansion_l3457_345750


namespace consecutive_even_integers_sum_l3457_345799

theorem consecutive_even_integers_sum (y : ℤ) : 
  y % 2 = 0 ∧ 
  (y + 2) % 2 = 0 ∧ 
  y = 2 * (y + 2) → 
  y + (y + 2) = -6 := by
  sorry

end consecutive_even_integers_sum_l3457_345799


namespace rational_number_ordering_l3457_345701

theorem rational_number_ordering (a b c : ℚ) 
  (h1 : a - b > 0) (h2 : b - c > 0) : c < b ∧ b < a := by
  sorry

end rational_number_ordering_l3457_345701


namespace fence_length_15m_l3457_345716

/-- The length of a fence surrounding a square swimming pool -/
def fence_length (side_length : ℝ) : ℝ := 4 * side_length

/-- Theorem: The length of a fence surrounding a square swimming pool with side length 15 meters is 60 meters -/
theorem fence_length_15m : fence_length 15 = 60 := by
  sorry

end fence_length_15m_l3457_345716


namespace line_intersects_parabola_vertex_l3457_345760

theorem line_intersects_parabola_vertex (b : ℝ) : 
  (∃! (x y : ℝ), y = x + b ∧ y = x^2 + 2*b^2 ∧ x = 0) ↔ (b = 0 ∨ b = 1/2) :=
sorry

end line_intersects_parabola_vertex_l3457_345760


namespace quadratic_always_positive_inequality_implication_existence_of_divisible_number_l3457_345767

-- Problem 1
theorem quadratic_always_positive : ∀ x : ℝ, x^2 - 8*x + 17 > 0 := by sorry

-- Problem 2
theorem inequality_implication : ∀ x : ℝ, (x+2)^2 - (x-3)^2 ≥ 0 → x ≥ 1/2 := by sorry

-- Problem 3
theorem existence_of_divisible_number : ∃ n : ℕ, 11 ∣ (6*n^2 - 7) := by sorry

end quadratic_always_positive_inequality_implication_existence_of_divisible_number_l3457_345767


namespace kim_integer_problem_l3457_345733

theorem kim_integer_problem (x y : ℤ) : 
  3 * x + 2 * y = 145 → (x = 35 ∨ y = 35) → (x = 20 ∨ y = 20) :=
by sorry

end kim_integer_problem_l3457_345733


namespace peter_green_notebooks_l3457_345719

/-- Represents the number of green notebooks Peter bought -/
def green_notebooks (total notebooks : ℕ) (black_notebooks pink_notebooks : ℕ) 
  (total_cost black_cost pink_cost : ℕ) : ℕ :=
  total - black_notebooks - pink_notebooks

/-- Theorem stating that Peter bought 2 green notebooks -/
theorem peter_green_notebooks : 
  green_notebooks 4 1 1 45 15 10 = 2 := by
  sorry

end peter_green_notebooks_l3457_345719


namespace gcd_g_50_52_eq_one_l3457_345747

/-- The function g(x) = x^2 - 3x + 2023 -/
def g (x : ℤ) : ℤ := x^2 - 3*x + 2023

/-- Theorem: The greatest common divisor of g(50) and g(52) is 1 -/
theorem gcd_g_50_52_eq_one : Int.gcd (g 50) (g 52) = 1 := by
  sorry

end gcd_g_50_52_eq_one_l3457_345747


namespace smallest_four_digit_unique_divisible_by_digits_with_five_l3457_345726

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def has_unique_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 4 ∧ digits.toFinset.card = 4

def divisible_by_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d ≠ 0 → n % d = 0

def includes_digit_five (n : ℕ) : Prop :=
  5 ∈ n.digits 10

theorem smallest_four_digit_unique_divisible_by_digits_with_five :
  ∀ n : ℕ, is_four_digit n →
           has_unique_digits n →
           divisible_by_digits n →
           includes_digit_five n →
           1560 ≤ n :=
sorry

end smallest_four_digit_unique_divisible_by_digits_with_five_l3457_345726


namespace equation_condition_l3457_345763

theorem equation_condition (a b c : ℤ) :
  a * (a - b) + b * (b - c) + c * (c - a) = 0 ↔ 
  (a = c ∧ b - 2 = c ∧ a = 0 ∧ b = 0 ∧ c = 0) := by
sorry

end equation_condition_l3457_345763


namespace roller_coaster_runs_l3457_345725

def people_in_line : ℕ := 1532
def num_cars : ℕ := 8
def seats_per_car : ℕ := 3

def capacity_per_ride : ℕ := num_cars * seats_per_car

theorem roller_coaster_runs : 
  ∃ (runs : ℕ), runs * capacity_per_ride ≥ people_in_line ∧ 
  ∀ (k : ℕ), k * capacity_per_ride ≥ people_in_line → k ≥ runs :=
by sorry

end roller_coaster_runs_l3457_345725


namespace sticks_form_triangle_l3457_345746

-- Define the lengths of the sticks
def a : ℝ := 3
def b : ℝ := 4
def c : ℝ := 5

-- Define the triangle inequality theorem
def triangle_inequality (x y z : ℝ) : Prop :=
  x + y > z ∧ x + z > y ∧ y + z > x

-- Theorem statement
theorem sticks_form_triangle : triangle_inequality a b c := by
  sorry

end sticks_form_triangle_l3457_345746


namespace triangle_satisfies_equation_l3457_345722

/-- Converts a number from base 5 to base 10 -/
def base5To10 (d1 d2 : ℕ) : ℕ := 5 * d1 + d2

/-- Converts a number from base 12 to base 10 -/
def base12To10 (d1 d2 : ℕ) : ℕ := 12 * d1 + d2

/-- The digit satisfying the equation in base 5 and base 12 -/
def triangle : ℕ := 2

theorem triangle_satisfies_equation :
  base5To10 5 triangle = base12To10 triangle 3 ∧ triangle < 10 := by sorry

end triangle_satisfies_equation_l3457_345722


namespace right_triangle_perimeter_l3457_345714

theorem right_triangle_perimeter (a b c r : ℝ) : 
  a > 0 → b > 0 → c > 0 → r > 0 →
  c = 10 → r = 1 →
  a^2 + b^2 = c^2 →
  (a + b - c) * r = a * b / 2 →
  a + b + c = 24 :=
sorry

end right_triangle_perimeter_l3457_345714


namespace customized_notebook_combinations_l3457_345707

/-- The number of different notebook designs available. -/
def notebook_designs : ℕ := 12

/-- The number of different pen types available. -/
def pen_types : ℕ := 3

/-- The number of different sticker varieties available. -/
def sticker_varieties : ℕ := 5

/-- The total number of possible combinations for a customized notebook package. -/
def total_combinations : ℕ := notebook_designs * pen_types * sticker_varieties

/-- Theorem stating that the total number of combinations is 180. -/
theorem customized_notebook_combinations :
  total_combinations = 180 := by sorry

end customized_notebook_combinations_l3457_345707


namespace marco_score_percentage_l3457_345769

/-- Proves that Marco scored 10% less than the average test score -/
theorem marco_score_percentage (average_score : ℝ) (margaret_score : ℝ) (marco_score : ℝ) :
  average_score = 90 →
  margaret_score = 86 →
  margaret_score = marco_score + 5 →
  (average_score - marco_score) / average_score = 0.1 := by
  sorry

end marco_score_percentage_l3457_345769


namespace emma_ball_lists_l3457_345751

/-- The number of balls in the bin -/
def n : ℕ := 24

/-- The number of draws -/
def k : ℕ := 4

/-- The number of possible lists when drawing with replacement from n balls, k times -/
def num_lists (n k : ℕ) : ℕ := n^k

theorem emma_ball_lists : num_lists n k = 331776 := by
  sorry

end emma_ball_lists_l3457_345751


namespace problem_solution_l3457_345793

theorem problem_solution (x y z : ℝ) 
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h_xyz : x * y * z = 1)
  (h_x_z : x + 1 / z = 7)
  (h_y_x : y + 1 / x = 20) :
  z + 1 / y = 29 / 139 := by
sorry

end problem_solution_l3457_345793


namespace min_pencils_divisible_by_3_and_4_l3457_345752

theorem min_pencils_divisible_by_3_and_4 : 
  ∃ n : ℕ, n > 0 ∧ n % 3 = 0 ∧ n % 4 = 0 ∧ ∀ m : ℕ, m > 0 → m % 3 = 0 → m % 4 = 0 → n ≤ m :=
by
  -- Proof goes here
  sorry

end min_pencils_divisible_by_3_and_4_l3457_345752


namespace line_equation_from_conditions_l3457_345764

/-- Vector in R² -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Line in R² -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Point in R² -/
structure Point2D where
  x : ℝ
  y : ℝ

def vector_add (v w : Vector2D) : Vector2D :=
  ⟨v.x + w.x, v.y + w.y⟩

def vector_scale (k : ℝ) (v : Vector2D) : Vector2D :=
  ⟨k * v.x, k * v.y⟩

def is_perpendicular (v : Vector2D) (l : Line2D) : Prop :=
  v.x * l.a + v.y * l.b = 0

def point_on_line (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

theorem line_equation_from_conditions 
  (a b : Vector2D)
  (A : Point2D)
  (l : Line2D)
  (h1 : a = ⟨6, 2⟩)
  (h2 : b = ⟨-4, 1/2⟩)
  (h3 : A = ⟨3, -1⟩)
  (h4 : is_perpendicular (vector_add a (vector_scale 2 b)) l)
  (h5 : point_on_line A l) :
  l = ⟨2, -3, -9⟩ :=
sorry

end line_equation_from_conditions_l3457_345764


namespace polynomial_simplification_l3457_345765

theorem polynomial_simplification (x : ℝ) : 
  (2 * x^6 + 3 * x^5 + x^4 + x^3 + x + 10) - (x^6 + 4 * x^5 + 2 * x^4 - x^3 + 12) = 
  x^6 - x^5 - x^4 + 2 * x^3 + x - 2 := by
  sorry

end polynomial_simplification_l3457_345765


namespace pet_store_puppies_l3457_345735

theorem pet_store_puppies (initial_kittens : ℕ) (sold_puppies sold_kittens remaining_pets : ℕ) 
  (h1 : initial_kittens = 6)
  (h2 : sold_puppies = 2)
  (h3 : sold_kittens = 3)
  (h4 : remaining_pets = 8) :
  ∃ initial_puppies : ℕ, 
    initial_puppies - sold_puppies + initial_kittens - sold_kittens = remaining_pets ∧ 
    initial_puppies = 7 := by
  sorry

end pet_store_puppies_l3457_345735


namespace managers_salary_correct_managers_salary_l3457_345758

theorem managers_salary (num_employees : ℕ) (avg_salary : ℝ) (avg_increase : ℝ) : ℝ :=
  let total_salary := num_employees * avg_salary
  let new_avg_salary := avg_salary + avg_increase
  let new_total_salary := (num_employees + 1) * new_avg_salary
  new_total_salary - total_salary

theorem correct_managers_salary :
  managers_salary 50 2000 250 = 14750 := by sorry

end managers_salary_correct_managers_salary_l3457_345758


namespace stratified_sampling_middle_schools_l3457_345790

theorem stratified_sampling_middle_schools 
  (total_schools : ℕ) 
  (middle_schools : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_schools = 700) 
  (h2 : middle_schools = 200) 
  (h3 : sample_size = 70) :
  (sample_size : ℚ) * (middle_schools : ℚ) / (total_schools : ℚ) = 20 := by
sorry

end stratified_sampling_middle_schools_l3457_345790


namespace sales_volume_correct_profit_at_95_max_profit_at_110_l3457_345738

/-- Represents the weekly sales volume as a function of selling price -/
def sales_volume (x : ℝ) : ℝ := -10 * x + 1500

/-- Represents the weekly profit as a function of selling price -/
def profit (x : ℝ) : ℝ := (x - 80) * sales_volume x

/-- The cost price of each shirt -/
def cost_price : ℝ := 80

/-- The minimum allowed selling price -/
def min_price : ℝ := 90

/-- The maximum allowed selling price -/
def max_price : ℝ := 110

theorem sales_volume_correct :
  sales_volume 90 = 600 ∧ 
  ∀ x, sales_volume (x + 1) = sales_volume x - 10 := by sorry

theorem profit_at_95 : profit 95 = 8250 := by sorry

theorem max_profit_at_110 : 
  profit 110 = 12000 ∧
  ∀ x, min_price ≤ x ∧ x ≤ max_price → profit x ≤ 12000 := by sorry

end sales_volume_correct_profit_at_95_max_profit_at_110_l3457_345738


namespace complement_A_intersect_B_l3457_345712

-- Define set A
def A : Set ℝ := {x | x^2 + x - 2 > 0}

-- Define set B
def B : Set ℝ := {y | ∃ x > 0, y = Real.log x / Real.log 2}

-- Define the complement of A in ℝ
def complement_A : Set ℝ := {x | x ∉ A}

-- Theorem statement
theorem complement_A_intersect_B : complement_A ∩ B = Set.Icc (-2) 1 := by
  sorry

end complement_A_intersect_B_l3457_345712


namespace min_sum_squares_l3457_345762

theorem min_sum_squares (a b c d e f g h : ℤ) : 
  a ∈ ({-7, -5, -3, -1, 0, 2, 4, 5} : Set ℤ) →
  b ∈ ({-7, -5, -3, -1, 0, 2, 4, 5} : Set ℤ) →
  c ∈ ({-7, -5, -3, -1, 0, 2, 4, 5} : Set ℤ) →
  d ∈ ({-7, -5, -3, -1, 0, 2, 4, 5} : Set ℤ) →
  e ∈ ({-7, -5, -3, -1, 0, 2, 4, 5} : Set ℤ) →
  f ∈ ({-7, -5, -3, -1, 0, 2, 4, 5} : Set ℤ) →
  g ∈ ({-7, -5, -3, -1, 0, 2, 4, 5} : Set ℤ) →
  h ∈ ({-7, -5, -3, -1, 0, 2, 4, 5} : Set ℤ) →
  a ≠ b → a ≠ c → a ≠ d → a ≠ e → a ≠ f → a ≠ g → a ≠ h →
  b ≠ c → b ≠ d → b ≠ e → b ≠ f → b ≠ g → b ≠ h →
  c ≠ d → c ≠ e → c ≠ f → c ≠ g → c ≠ h →
  d ≠ e → d ≠ f → d ≠ g → d ≠ h →
  e ≠ f → e ≠ g → e ≠ h →
  f ≠ g → f ≠ h →
  g ≠ h →
  (a + b + c + d)^2 + (e + f + g + h)^2 ≥ 34 := by
sorry

end min_sum_squares_l3457_345762


namespace square_perimeter_relation_l3457_345785

theorem square_perimeter_relation (perimeter_C : ℝ) (area_ratio : ℝ) : 
  perimeter_C = 40 →
  area_ratio = 1/3 →
  let side_C := perimeter_C / 4
  let area_C := side_C ^ 2
  let area_D := area_ratio * area_C
  let side_D := Real.sqrt area_D
  let perimeter_D := 4 * side_D
  perimeter_D = (40 * Real.sqrt 3) / 3 := by
sorry

end square_perimeter_relation_l3457_345785


namespace quadratic_equation_solution_l3457_345755

theorem quadratic_equation_solution (x c d : ℝ) : 
  x^2 + 14*x = 92 → 
  (∃ c d : ℕ+, x = Real.sqrt c - d) →
  (∃ c d : ℕ+, x = Real.sqrt c - d ∧ c + d = 148) :=
by sorry

end quadratic_equation_solution_l3457_345755


namespace jillian_oranges_l3457_345787

/-- Given that Jillian divides oranges into pieces for her friends, 
    this theorem proves the number of oranges she had. -/
theorem jillian_oranges 
  (pieces_per_orange : ℕ) 
  (pieces_per_friend : ℕ) 
  (num_friends : ℕ) 
  (h1 : pieces_per_orange = 10) 
  (h2 : pieces_per_friend = 4) 
  (h3 : num_friends = 200) : 
  (num_friends * pieces_per_friend) / pieces_per_orange = 80 := by
  sorry

end jillian_oranges_l3457_345787


namespace polyhedron_edge_intersection_l3457_345792

/-- A polyhedron with a specified number of edges. -/
structure Polyhedron where
  edges : ℕ

/-- A plane that can intersect edges of a polyhedron. -/
structure IntersectingPlane where
  intersected_edges : ℕ

/-- Theorem about the maximum number of intersected edges for different types of polyhedra. -/
theorem polyhedron_edge_intersection
  (p : Polyhedron)
  (h : p.edges = 100) :
  ∃ (convex_max non_convex_max : ℕ),
    -- For a convex polyhedron, the maximum number of intersected edges is 66
    (∀ (plane : IntersectingPlane), plane.intersected_edges ≤ convex_max) ∧
    convex_max = 66 ∧
    -- For a non-convex polyhedron, there exists a configuration where 96 edges can be intersected
    (∃ (plane : IntersectingPlane), plane.intersected_edges = non_convex_max) ∧
    non_convex_max = 96 ∧
    -- For any polyhedron, it's impossible to intersect all 100 edges
    (∀ (plane : IntersectingPlane), plane.intersected_edges < p.edges) :=
by
  sorry

end polyhedron_edge_intersection_l3457_345792


namespace problem_solution_l3457_345778

theorem problem_solution : 18 * ((150 / 3) + (40 / 5) + (16 / 32) + 2) = 1089 := by
  sorry

end problem_solution_l3457_345778


namespace xy_value_l3457_345706

theorem xy_value (x y : ℝ) (sum_eq : x + y = 10) (sum_cubes_eq : x^3 + y^3 = 370) : x * y = 21 := by
  sorry

end xy_value_l3457_345706


namespace sin_alpha_value_l3457_345761

theorem sin_alpha_value (α : Real) 
  (h : Real.cos (α - π / 2) = 1 / 3) : 
  Real.sin α = 1 / 3 := by
sorry

end sin_alpha_value_l3457_345761


namespace power_two_gt_two_n_plus_one_power_two_le_two_n_plus_one_for_small_n_smallest_n_for_inequality_l3457_345729

theorem power_two_gt_two_n_plus_one (n : ℕ) : n ≥ 3 → 2^n > 2*n + 1 :=
  sorry

theorem power_two_le_two_n_plus_one_for_small_n :
  (2^1 ≤ 2*1 + 1) ∧ (2^2 ≤ 2*2 + 1) :=
  sorry

theorem smallest_n_for_inequality : ∀ n : ℕ, n ≥ 3 ↔ 2^n > 2*n + 1 :=
  sorry

end power_two_gt_two_n_plus_one_power_two_le_two_n_plus_one_for_small_n_smallest_n_for_inequality_l3457_345729


namespace captains_age_and_crew_size_l3457_345702

theorem captains_age_and_crew_size (l k : ℕ) : 
  l * (l - 1) = k * (l - 2) + 15 → 
  ((l = 1 ∧ k = 15) ∨ (l = 15 ∧ k = 15)) := by
sorry

end captains_age_and_crew_size_l3457_345702


namespace min_abs_GB_is_392_l3457_345757

-- Define the Revolution polynomial
def Revolution (U S A : ℤ) (x : ℤ) : ℤ := x^3 + U*x^2 + S*x + A

-- State the theorem
theorem min_abs_GB_is_392 
  (U S A G B : ℤ) 
  (h1 : U + S + A + 1 = 1773)
  (h2 : ∀ x, Revolution U S A x = 0 ↔ x = G ∨ x = B)
  (h3 : G ≠ B)
  (h4 : G ≠ 0)
  (h5 : B ≠ 0) :
  ∃ (G' B' : ℤ), G' * B' = 392 ∧ 
    ∀ (G'' B'' : ℤ), G'' ≠ 0 ∧ B'' ≠ 0 ∧ 
      (∀ x, Revolution U S A x = 0 ↔ x = G'' ∨ x = B'') → 
      abs (G'' * B'') ≥ 392 :=
sorry

end min_abs_GB_is_392_l3457_345757


namespace find_a_l3457_345770

theorem find_a : ∃ a : ℝ, (2 * 1 - a * (-1) = 3) ∧ a = 1 := by sorry

end find_a_l3457_345770


namespace data_mode_is_neg_one_l3457_345708

def data : List Int := [-1, 0, 2, -1, 3]

def mode (l : List α) [DecidableEq α] : Option α :=
  l.foldl (λ acc x => 
    match acc with
    | none => some x
    | some y => if l.count x > l.count y then some x else some y
  ) none

theorem data_mode_is_neg_one : mode data = some (-1) := by
  sorry

end data_mode_is_neg_one_l3457_345708


namespace linear_function_properties_l3457_345795

/-- A linear function passing through points (3,5) and (-4,-9) -/
def f (x : ℝ) : ℝ := 2 * x - 1

theorem linear_function_properties :
  (∃ k b : ℝ, ∀ x, f x = k * x + b) ∧
  f 3 = 5 ∧
  f (-4) = -9 ∧
  f 0 = -1 ∧
  f (1/2) = 0 ∧
  (1/2 * |f 0|) / 2 = 1/4 ∧
  (∀ a : ℝ, f a = 2 → a = 3/2) :=
sorry

end linear_function_properties_l3457_345795


namespace drawing_red_ball_certain_l3457_345742

/-- A bag containing only red balls -/
structure RedBallBag where
  num_balls : ℕ
  all_red : True

/-- The probability of drawing a red ball from a bag of red balls -/
def prob_draw_red (bag : RedBallBag) : ℝ :=
  1

/-- An event is certain if its probability is 1 -/
def is_certain_event (p : ℝ) : Prop :=
  p = 1

/-- Theorem: Drawing a red ball from a bag containing only 5 red balls is a certain event -/
theorem drawing_red_ball_certain (bag : RedBallBag) (h : bag.num_balls = 5) :
    is_certain_event (prob_draw_red bag) := by
  sorry

end drawing_red_ball_certain_l3457_345742


namespace red_candles_count_l3457_345797

/-- Given the ratio of red candles to blue candles and the number of blue candles,
    calculate the number of red candles. -/
theorem red_candles_count (blue_candles : ℕ) (ratio_red : ℕ) (ratio_blue : ℕ) 
    (h1 : blue_candles = 27) 
    (h2 : ratio_red = 5) 
    (h3 : ratio_blue = 3) : ℕ :=
  45

#check red_candles_count

end red_candles_count_l3457_345797


namespace inequality_proof_l3457_345791

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (b + c) / (2 * a) + (c + a) / (2 * b) + (a + b) / (2 * c) ≥
  2 * a / (b + c) + 2 * b / (c + a) + 2 * c / (a + b) := by
  sorry

end inequality_proof_l3457_345791


namespace min_value_theorem_l3457_345720

theorem min_value_theorem (a : ℝ) (x₁ x₂ : ℝ) 
  (h_a : a > 0)
  (h_sol : ∀ x, -x^2 + 6*a*x - 3*a^2 ≥ 0 ↔ x₁ ≤ x ∧ x ≤ x₂) :
  ∃ (m : ℝ), m = 2 * Real.sqrt 6 ∧ 
    ∀ y₁ y₂, (∀ x, -x^2 + 6*a*x - 3*a^2 ≥ 0 ↔ y₁ ≤ x ∧ x ≤ y₂) → 
      y₁ + y₂ + 3*a / (y₁ * y₂) ≥ m :=
by sorry

end min_value_theorem_l3457_345720


namespace factor_calculation_l3457_345745

theorem factor_calculation (x : ℝ) (h : x = 36) : 
  ∃ f : ℝ, ((x + 10) * f / 2) - 2 = 88 / 2 ∧ f = 2 := by
  sorry

end factor_calculation_l3457_345745


namespace point_C_coordinates_l3457_345737

/-- Triangle ABC with right angle at B and altitude from A to D on BC -/
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  h_A : A = (7, 1)
  h_B : B = (5, -3)
  h_D : D = (5, 1)
  h_right_angle : (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0
  h_altitude : (A.1 - D.1) * (C.1 - B.1) + (A.2 - D.2) * (C.2 - B.2) = 0

/-- The coordinates of point C in the given right triangle -/
theorem point_C_coordinates (t : RightTriangle) : t.C = (5, 5) := by
  sorry

end point_C_coordinates_l3457_345737


namespace cyclic_inequality_sqrt_l3457_345749

theorem cyclic_inequality_sqrt (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (Real.sqrt (3 * x * (x + y) * (y + z)) +
   Real.sqrt (3 * y * (y + z) * (z + x)) +
   Real.sqrt (3 * z * (z + x) * (x + y))) ≤
  Real.sqrt (4 * (x + y + z)^3) := by
  sorry

end cyclic_inequality_sqrt_l3457_345749
