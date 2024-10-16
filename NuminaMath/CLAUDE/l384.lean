import Mathlib

namespace NUMINAMATH_CALUDE_derivative_x_ln_x_l384_38454

/-- The derivative of x * ln(x) is 1 + ln(x) -/
theorem derivative_x_ln_x (x : ℝ) (h : x > 0) : 
  deriv (fun x => x * Real.log x) x = 1 + Real.log x := by
  sorry

end NUMINAMATH_CALUDE_derivative_x_ln_x_l384_38454


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_division_l384_38436

/-- An isosceles right triangle -/
structure IsoscelesRightTriangle where
  side : ℝ
  side_positive : side > 0

/-- An isosceles trapezoid -/
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  leg : ℝ
  base1_positive : base1 > 0
  base2_positive : base2 > 0
  leg_positive : leg > 0

/-- A division of an isosceles right triangle into trapezoids -/
def TriangleDivision (t : IsoscelesRightTriangle) := 
  List IsoscelesTrapezoid

theorem isosceles_right_triangle_division (t : IsoscelesRightTriangle) :
  ∃ (d : TriangleDivision t), d.length = 7 :=
sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_division_l384_38436


namespace NUMINAMATH_CALUDE_remainder_987670_div_128_l384_38404

theorem remainder_987670_div_128 : 987670 % 128 = 22 := by
  sorry

end NUMINAMATH_CALUDE_remainder_987670_div_128_l384_38404


namespace NUMINAMATH_CALUDE_work_ratio_l384_38413

/-- Given that A can finish a work in 18 days and A and B working together can finish 1/6 of the work in a day, 
    prove that the ratio of time taken by B to A to finish the work is 1:2 -/
theorem work_ratio (a_time : ℝ) (combined_rate : ℝ) 
  (ha : a_time = 18)
  (hc : combined_rate = 1/6) : 
  (a_time / 2) / a_time = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_work_ratio_l384_38413


namespace NUMINAMATH_CALUDE_power_division_rule_l384_38448

theorem power_division_rule (m : ℝ) (h : m ≠ 0) : m^7 / m = m^6 := by
  sorry

end NUMINAMATH_CALUDE_power_division_rule_l384_38448


namespace NUMINAMATH_CALUDE_symmetry_of_point_l384_38444

def is_symmetric_wrt_origin (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = -q.2

theorem symmetry_of_point :
  let p : ℝ × ℝ := (-2, 3)
  let q : ℝ × ℝ := (2, -3)
  is_symmetric_wrt_origin p q :=
by sorry

end NUMINAMATH_CALUDE_symmetry_of_point_l384_38444


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l384_38407

theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geom : ∃ (q : ℝ), ∀ (n : ℕ), a (n + 1) = a n * q)
    (h_sum1 : a 3 + a 6 = 6) (h_sum2 : a 5 + a 8 = 9) :
  a 7 + a 10 = 27 / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l384_38407


namespace NUMINAMATH_CALUDE_parallel_transitivity_l384_38493

-- Define a type for lines in a plane
structure Line where
  -- You can add more specific properties here if needed
  mk :: 

-- Define what it means for two lines to be parallel
def parallel (l1 l2 : Line) : Prop :=
  -- The definition of parallel lines
  sorry

-- State the theorem
theorem parallel_transitivity (l1 l2 l3 : Line) : 
  parallel l1 l3 → parallel l2 l3 → parallel l1 l2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_transitivity_l384_38493


namespace NUMINAMATH_CALUDE_smallest_n_value_l384_38473

def count_factors_of_five (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625)

theorem smallest_n_value (a b c m n : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 2023 →
  11 ∣ a →
  a * b * c ≠ 0 →
  ∃ (m : ℕ), m % 10 ≠ 0 ∧ a.factorial * b.factorial * c.factorial = m * (10 ^ n) →
  (∀ k, k < n → ¬∃ (l : ℕ), l % 10 ≠ 0 ∧ a.factorial * b.factorial * c.factorial = l * (10 ^ k)) →
  n = 497 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_value_l384_38473


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l384_38405

theorem arithmetic_calculations :
  (15 + (-6) + 3 - (-4) = 16) ∧
  (8 - 2^3 / (4/9) * (-2/3)^2 = 0) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l384_38405


namespace NUMINAMATH_CALUDE_parabola_intersection_l384_38496

theorem parabola_intersection :
  let f (x : ℝ) := 3 * x^2 - 4 * x + 2
  let g (x : ℝ) := 9 * x^2 + 6 * x + 2
  ∀ x y : ℝ, f x = g x ∧ y = f x ↔ (x = 0 ∧ y = 2) ∨ (x = -5/3 ∧ y = 17) :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l384_38496


namespace NUMINAMATH_CALUDE_smallest_root_of_quadratic_l384_38486

theorem smallest_root_of_quadratic (x : ℝ) :
  (10 * x^2 - 48 * x + 44 = 0) →
  (∀ y : ℝ, 10 * y^2 - 48 * y + 44 = 0 → x ≤ y) →
  x = 1.234 := by
sorry

end NUMINAMATH_CALUDE_smallest_root_of_quadratic_l384_38486


namespace NUMINAMATH_CALUDE_same_color_sock_pairs_l384_38452

def white_socks : ℕ := 5
def brown_socks : ℕ := 5
def blue_socks : ℕ := 4
def black_socks : ℕ := 2

def total_socks : ℕ := white_socks + brown_socks + blue_socks + black_socks

def choose_pair (n : ℕ) : ℕ := n.choose 2

theorem same_color_sock_pairs :
  choose_pair white_socks + choose_pair brown_socks + choose_pair blue_socks + choose_pair black_socks = 27 := by
  sorry

end NUMINAMATH_CALUDE_same_color_sock_pairs_l384_38452


namespace NUMINAMATH_CALUDE_last_four_digits_of_5_pow_2018_l384_38485

def last_four_digits (n : ℕ) : ℕ := n % 10000

def cycle : List ℕ := [3125, 5625, 8125, 0625]

theorem last_four_digits_of_5_pow_2018 :
  last_four_digits (5^2018) = 5625 := by
  sorry

end NUMINAMATH_CALUDE_last_four_digits_of_5_pow_2018_l384_38485


namespace NUMINAMATH_CALUDE_complex_number_with_prime_modulus_exists_l384_38472

theorem complex_number_with_prime_modulus_exists : ∃ (z : ℂ), 
  z^2 = (3 + Complex.I) * z - 24 + 15 * Complex.I ∧ 
  ∃ (p : ℕ), Nat.Prime p ∧ (z.re^2 + z.im^2 : ℝ) = p :=
sorry

end NUMINAMATH_CALUDE_complex_number_with_prime_modulus_exists_l384_38472


namespace NUMINAMATH_CALUDE_min_value_implies_a_eq_6_l384_38499

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1/3 then 3 - Real.sin (a * x) else a * x + Real.log x / Real.log 3

-- State the theorem
theorem min_value_implies_a_eq_6 (a : ℝ) (h1 : a > 0) :
  (∀ x, f a x ≥ 1) ∧ (∃ x, f a x = 1) → a = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_implies_a_eq_6_l384_38499


namespace NUMINAMATH_CALUDE_poker_hand_probabilities_l384_38456

-- Define the total number of possible 5-card hands
def total_hands : ℕ := 2598960

-- Define the number of ways to get each hand type
def pair_ways : ℕ := 1098240
def two_pair_ways : ℕ := 123552
def three_of_a_kind_ways : ℕ := 54912
def straight_ways : ℕ := 10000
def flush_ways : ℕ := 5108
def full_house_ways : ℕ := 3744
def four_of_a_kind_ways : ℕ := 624
def straight_flush_ways : ℕ := 40

-- Define the probability of each hand type
def prob_pair : ℚ := pair_ways / total_hands
def prob_two_pair : ℚ := two_pair_ways / total_hands
def prob_three_of_a_kind : ℚ := three_of_a_kind_ways / total_hands
def prob_straight : ℚ := straight_ways / total_hands
def prob_flush : ℚ := flush_ways / total_hands
def prob_full_house : ℚ := full_house_ways / total_hands
def prob_four_of_a_kind : ℚ := four_of_a_kind_ways / total_hands
def prob_straight_flush : ℚ := straight_flush_ways / total_hands

-- Theorem stating the probabilities of different poker hands
theorem poker_hand_probabilities :
  (prob_pair = 1098240 / 2598960) ∧
  (prob_two_pair = 123552 / 2598960) ∧
  (prob_three_of_a_kind = 54912 / 2598960) ∧
  (prob_straight = 10000 / 2598960) ∧
  (prob_flush = 5108 / 2598960) ∧
  (prob_full_house = 3744 / 2598960) ∧
  (prob_four_of_a_kind = 624 / 2598960) ∧
  (prob_straight_flush = 40 / 2598960) :=
by sorry

end NUMINAMATH_CALUDE_poker_hand_probabilities_l384_38456


namespace NUMINAMATH_CALUDE_unique_solution_exponential_equation_l384_38422

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (2 : ℝ) ^ (4 * x + 2) * (4 : ℝ) ^ (2 * x + 1) = (8 : ℝ) ^ (3 * x + 4) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_equation_l384_38422


namespace NUMINAMATH_CALUDE_least_number_of_cans_l384_38476

def maaza_volume : ℕ := 60
def pepsi_volume : ℕ := 144
def sprite_volume : ℕ := 368

def can_volume : ℕ := Nat.gcd maaza_volume (Nat.gcd pepsi_volume sprite_volume)

def maaza_cans : ℕ := maaza_volume / can_volume
def pepsi_cans : ℕ := pepsi_volume / can_volume
def sprite_cans : ℕ := sprite_volume / can_volume

def total_cans : ℕ := maaza_cans + pepsi_cans + sprite_cans

theorem least_number_of_cans : total_cans = 143 := by
  sorry

end NUMINAMATH_CALUDE_least_number_of_cans_l384_38476


namespace NUMINAMATH_CALUDE_least_distance_eight_girls_circle_l384_38427

/-- The least total distance traveled by 8 girls on a circle -/
theorem least_distance_eight_girls_circle (r : ℝ) (h : r = 50) :
  let n : ℕ := 8  -- number of girls
  let angle := 2 * Real.pi / n  -- angle between adjacent girls
  let non_adjacent_angle := 3 * angle  -- angle to third girl (non-adjacent)
  let single_path := r * Real.sqrt (2 + Real.sqrt 2)  -- distance to one non-adjacent girl and back
  let total_distance := n * 4 * single_path  -- total distance for all girls
  total_distance = 1600 * Real.sqrt (2 + Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_least_distance_eight_girls_circle_l384_38427


namespace NUMINAMATH_CALUDE_matching_probability_abe_bob_l384_38446

/-- Represents the number of jelly beans of each color for a person -/
structure JellyBeans where
  green : ℕ
  red : ℕ
  yellow : ℕ

/-- Calculates the total number of jelly beans -/
def JellyBeans.total (jb : JellyBeans) : ℕ := jb.green + jb.red + jb.yellow

/-- Represents the jelly beans held by Abe -/
def abe : JellyBeans := { green := 1, red := 1, yellow := 0 }

/-- Represents the jelly beans held by Bob -/
def bob : JellyBeans := { green := 1, red := 2, yellow := 1 }

/-- Calculates the probability of two people showing the same color jelly bean -/
def matchingProbability (person1 person2 : JellyBeans) : ℚ :=
  let greenProb := (person1.green : ℚ) / person1.total * (person2.green : ℚ) / person2.total
  let redProb := (person1.red : ℚ) / person1.total * (person2.red : ℚ) / person2.total
  greenProb + redProb

theorem matching_probability_abe_bob :
  matchingProbability abe bob = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_matching_probability_abe_bob_l384_38446


namespace NUMINAMATH_CALUDE_f_derivative_positive_at_midpoint_l384_38402

open Real

/-- The function f(x) = x^2 + 2x - a(ln x + x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*x - a*(log x + x)

/-- The derivative of f(x) -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := 2*x + 2 - a*(1/x + 1)

theorem f_derivative_positive_at_midpoint (a c : ℝ) (x₁ x₂ : ℝ) 
  (hx₁ : f a x₁ = c) (hx₂ : f a x₂ = c) (hne : x₁ ≠ x₂) :
  f_derivative a ((x₁ + x₂) / 2) > 0 :=
sorry

end NUMINAMATH_CALUDE_f_derivative_positive_at_midpoint_l384_38402


namespace NUMINAMATH_CALUDE_geometric_series_sum_l384_38466

theorem geometric_series_sum (a r : ℚ) (n : ℕ) (h : r ≠ 1) :
  let series_sum := a * (1 - r^n) / (1 - r)
  (a = 1/4) → (r = -1/4) → (n = 6) → series_sum = 4095/5120 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l384_38466


namespace NUMINAMATH_CALUDE_number_of_trucks_l384_38453

theorem number_of_trucks (total_packages : ℕ) (packages_per_truck : ℕ) (h1 : total_packages = 490) (h2 : packages_per_truck = 70) :
  total_packages / packages_per_truck = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_of_trucks_l384_38453


namespace NUMINAMATH_CALUDE_increase_by_percentage_increase_75_by_150_percent_l384_38421

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) :
  initial + (initial * (percentage / 100)) = initial * (1 + percentage / 100) :=
by sorry

theorem increase_75_by_150_percent :
  75 + (75 * (150 / 100)) = 187.5 :=
by sorry

end NUMINAMATH_CALUDE_increase_by_percentage_increase_75_by_150_percent_l384_38421


namespace NUMINAMATH_CALUDE_notebook_statements_l384_38475

theorem notebook_statements :
  ∃! n : Fin 40, (∀ m : Fin 40, (m.val + 1 = n.val) ↔ (m = n)) ∧ n.val = 39 :=
sorry

end NUMINAMATH_CALUDE_notebook_statements_l384_38475


namespace NUMINAMATH_CALUDE_eventually_divisible_by_large_power_of_two_l384_38480

/-- Represents the state of the board at any given minute -/
structure BoardState where
  numbers : Finset ℕ
  odd_count : ℕ
  minute : ℕ

/-- The initial state of the board -/
def initial_board : BoardState :=
  { numbers := Finset.empty,  -- We don't know the specific numbers, so we use an empty set
    odd_count := 33,
    minute := 0 }

/-- The next state of the board after one minute -/
def next_board_state (state : BoardState) : BoardState :=
  { numbers := state.numbers,  -- We don't update the specific numbers
    odd_count := state.odd_count,
    minute := state.minute + 1 }

/-- Predicate to check if a number is divisible by 2^10000000 -/
def is_divisible_by_large_power_of_two (n : ℕ) : Prop :=
  ∃ k, n = k * (2^10000000)

/-- The main theorem to prove -/
theorem eventually_divisible_by_large_power_of_two :
  ∃ (n : ℕ) (state : BoardState), 
    state.minute = n ∧ 
    ∃ (m : ℕ), m ∈ state.numbers ∧ is_divisible_by_large_power_of_two m :=
  sorry

end NUMINAMATH_CALUDE_eventually_divisible_by_large_power_of_two_l384_38480


namespace NUMINAMATH_CALUDE_third_player_games_l384_38429

/-- Represents a table tennis game with three players. -/
structure TableTennisGame where
  totalGames : ℕ
  player1Games : ℕ
  player2Games : ℕ
  player3Games : ℕ

/-- The rules of the game ensure that the total number of games is equal to the maximum number of games played by any player. -/
axiom total_games_rule (game : TableTennisGame) : game.totalGames = max game.player1Games (max game.player2Games game.player3Games)

/-- The sum of games played by all players is twice the total number of games. -/
axiom sum_of_games_rule (game : TableTennisGame) : game.player1Games + game.player2Games + game.player3Games = 2 * game.totalGames

/-- Theorem: In a three-player table tennis game where the loser gives up their spot, 
    if the first player plays 10 games and the second player plays 21 games, 
    then the third player must play 11 games. -/
theorem third_player_games (game : TableTennisGame) 
    (h1 : game.player1Games = 10) 
    (h2 : game.player2Games = 21) : 
  game.player3Games = 11 := by
  sorry

end NUMINAMATH_CALUDE_third_player_games_l384_38429


namespace NUMINAMATH_CALUDE_fountain_position_l384_38440

/-- Two towers with a fountain between them -/
structure TowerSetup where
  tower1_height : ℝ
  tower2_height : ℝ
  distance_between_towers : ℝ
  fountain_distance : ℝ

/-- The setup satisfies the problem conditions -/
def valid_setup (s : TowerSetup) : Prop :=
  s.tower1_height = 30 ∧
  s.tower2_height = 40 ∧
  s.distance_between_towers = 50 ∧
  0 < s.fountain_distance ∧
  s.fountain_distance < s.distance_between_towers

/-- The birds' flight paths are equal -/
def equal_flight_paths (s : TowerSetup) : Prop :=
  s.tower1_height^2 + s.fountain_distance^2 =
  s.tower2_height^2 + (s.distance_between_towers - s.fountain_distance)^2

theorem fountain_position (s : TowerSetup) 
  (h1 : valid_setup s) (h2 : equal_flight_paths s) :
  s.fountain_distance = 32 ∧ 
  s.distance_between_towers - s.fountain_distance = 18 := by
  sorry

end NUMINAMATH_CALUDE_fountain_position_l384_38440


namespace NUMINAMATH_CALUDE_perpendicular_lines_k_values_l384_38406

-- Define the coefficients of the lines
def a (k : ℝ) := k - 3
def b (k : ℝ) := 5 - k
def m (k : ℝ) := 2 * (k - 3)
def n : ℝ := -2

-- Define the perpendicularity condition
def perpendicular (k : ℝ) : Prop := a k * m k + b k * n = 0

-- Theorem statement
theorem perpendicular_lines_k_values :
  ∀ k : ℝ, perpendicular k → k = 1 ∨ k = 4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_k_values_l384_38406


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l384_38464

theorem inequality_and_equality_condition (a b c : ℝ) (h : a^2 + b^2 + c^2 = 3) :
  (a^2 / (2 + b + c^2) + b^2 / (2 + c + a^2) + c^2 / (2 + a + b^2) ≥ (a + b + c)^2 / 12) ∧
  ((a^2 / (2 + b + c^2) + b^2 / (2 + c + a^2) + c^2 / (2 + a + b^2) = (a + b + c)^2 / 12) ↔ 
   (a = 1 ∧ b = 1 ∧ c = 1)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l384_38464


namespace NUMINAMATH_CALUDE_probability_two_blue_balls_l384_38430

/-- The probability of drawing two blue balls consecutively from an urn -/
theorem probability_two_blue_balls (total_balls : Nat) (blue_balls : Nat) (red_balls : Nat) :
  total_balls = blue_balls + red_balls →
  blue_balls = 6 →
  red_balls = 4 →
  (blue_balls : ℚ) / total_balls * (blue_balls - 1) / (total_balls - 1) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_blue_balls_l384_38430


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l384_38403

theorem unique_solution_for_equation (x r p n : ℕ+) : 
  (x ^ r.val - 1 = p ^ n.val) ∧ 
  (Nat.Prime p.val) ∧ 
  (r.val ≥ 2) ∧ 
  (n.val ≥ 2) → 
  (x = 3 ∧ r = 2 ∧ p = 2 ∧ n = 3) := by
sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l384_38403


namespace NUMINAMATH_CALUDE_inequality_proof_l384_38463

theorem inequality_proof (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b = (a + 1/a^3)/2) 
  (hc : c = (b + 1/b^3)/2) 
  (hb_lt_1 : b < 1) : 
  1 < c ∧ c < a :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l384_38463


namespace NUMINAMATH_CALUDE_expand_product_l384_38401

theorem expand_product (x : ℝ) : 2 * (x - 3) * (x + 6) = 2 * x^2 + 6 * x - 36 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l384_38401


namespace NUMINAMATH_CALUDE_first_part_speed_l384_38495

/-- Proves that given a 50 km trip with two equal parts, where the second part is traveled at 33 km/h, 
    and the average speed of the entire trip is 44.00000000000001 km/h, 
    the speed of the first part of the trip is 66 km/h. -/
theorem first_part_speed (total_distance : ℝ) (first_part_distance : ℝ) (second_part_speed : ℝ) (average_speed : ℝ) :
  total_distance = 50 →
  first_part_distance = 25 →
  second_part_speed = 33 →
  average_speed = 44.00000000000001 →
  (total_distance / (first_part_distance / (total_distance - first_part_distance) * second_part_speed + first_part_distance / second_part_speed)) = average_speed →
  (total_distance - first_part_distance) / second_part_speed + first_part_distance / ((total_distance - first_part_distance) * second_part_speed / first_part_distance) = total_distance / average_speed →
  (total_distance - first_part_distance) * second_part_speed / first_part_distance = 66 :=
by sorry

end NUMINAMATH_CALUDE_first_part_speed_l384_38495


namespace NUMINAMATH_CALUDE_sin_sum_equality_l384_38416

theorem sin_sum_equality : 
  Real.sin (7 * π / 30) + Real.sin (11 * π / 30) = 
  Real.sin (π / 30) + Real.sin (13 * π / 30) + 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_equality_l384_38416


namespace NUMINAMATH_CALUDE_two_successive_discounts_l384_38474

theorem two_successive_discounts (list_price : ℝ) (final_price : ℝ) (first_discount : ℝ) (second_discount : ℝ) :
  list_price = 70 →
  final_price = 59.22 →
  first_discount = 10 →
  (list_price - (first_discount / 100) * list_price) * (1 - second_discount / 100) = final_price →
  second_discount = 6 := by
sorry

end NUMINAMATH_CALUDE_two_successive_discounts_l384_38474


namespace NUMINAMATH_CALUDE_cheese_division_theorem_l384_38457

/-- Represents the weight of cheese pieces -/
structure CheesePair :=
  (larger : ℕ)
  (smaller : ℕ)

/-- Divides cheese by taking a piece equal to the smaller portion from the larger -/
def divide_cheese (pair : CheesePair) : CheesePair :=
  CheesePair.mk pair.smaller (pair.larger - pair.smaller)

/-- The initial weight of the cheese -/
def initial_weight : ℕ := 850

/-- The final weight of each piece of cheese -/
def final_piece_weight : ℕ := 25

theorem cheese_division_theorem :
  let final_state := CheesePair.mk final_piece_weight final_piece_weight
  let third_division := divide_cheese (divide_cheese (divide_cheese (CheesePair.mk initial_weight 0)))
  third_division = final_state :=
sorry

end NUMINAMATH_CALUDE_cheese_division_theorem_l384_38457


namespace NUMINAMATH_CALUDE_parabola_circle_tangency_l384_38410

-- Define the parabola C
def parabola (x y : ℝ) : Prop := y^2 = x

-- Define the line l
def line_l (x : ℝ) : Prop := x = 1

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define a point on the parabola
def point_on_parabola (A : ℝ × ℝ) : Prop := parabola A.1 A.2

-- Define a line tangent to circle M
def tangent_to_circle_M (A B : ℝ × ℝ) : Prop :=
  ∃ (k m : ℝ), ∀ (x y : ℝ), y = k * x + m → 
    ((x - A.1) * (B.2 - A.2) = (y - A.2) * (B.1 - A.1)) →
    (2 * k + 2 * m - 4)^2 / (k^2 + 1) = 1

theorem parabola_circle_tangency 
  (A₁ A₂ A₃ : ℝ × ℝ) 
  (h₁ : point_on_parabola A₁) 
  (h₂ : point_on_parabola A₂) 
  (h₃ : point_on_parabola A₃) 
  (h₄ : tangent_to_circle_M A₁ A₂) 
  (h₅ : tangent_to_circle_M A₁ A₃) :
  tangent_to_circle_M A₂ A₃ :=
sorry

end NUMINAMATH_CALUDE_parabola_circle_tangency_l384_38410


namespace NUMINAMATH_CALUDE_linear_function_condition_l384_38411

/-- A linear function f(x) = ax + b satisfies f(x)f(y) + f(x+y-xy) ≤ 0 for all x, y ∈ [0, 1]
    if and only if -1 ≤ b ≤ 0 and -(b + 1) ≤ a ≤ -b -/
theorem linear_function_condition (a b : ℝ) :
  (∀ x y : ℝ, x ∈ Set.Icc 0 1 → y ∈ Set.Icc 0 1 →
    (a * x + b) * (a * y + b) + (a * (x + y - x * y) + b) ≤ 0) ↔
  (b ∈ Set.Icc (-1) 0 ∧ a ∈ Set.Icc (-(b + 1)) (-b)) :=
sorry

end NUMINAMATH_CALUDE_linear_function_condition_l384_38411


namespace NUMINAMATH_CALUDE_josie_shopping_shortfall_l384_38426

def gift_amount : ℕ := 80
def cassette_price : ℕ := 15
def num_cassettes : ℕ := 3
def headphone_price : ℕ := 40
def vinyl_price : ℕ := 12

theorem josie_shopping_shortfall :
  gift_amount < cassette_price * num_cassettes + headphone_price + vinyl_price ∧
  cassette_price * num_cassettes + headphone_price + vinyl_price - gift_amount = 17 :=
by sorry

end NUMINAMATH_CALUDE_josie_shopping_shortfall_l384_38426


namespace NUMINAMATH_CALUDE_milk_replacement_theorem_l384_38467

-- Define the percentage of milk replaced by water in each operation
def replacement_percentage : ℝ → Prop := λ x =>
  -- Define the function that calculates the remaining milk percentage after three operations
  let remaining_milk := (1 - x/100)^3
  -- The remaining milk percentage should be 51.2%
  remaining_milk = 0.512

-- Theorem statement
theorem milk_replacement_theorem : 
  ∃ x : ℝ, replacement_percentage x ∧ x = 20 :=
sorry

end NUMINAMATH_CALUDE_milk_replacement_theorem_l384_38467


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l384_38458

theorem quadratic_root_difference (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (∀ x y : ℝ, x^2 + p*x + q = 0 ∧ y^2 + p*y + q = 0 ∧ x ≠ y → |x - y| = 2) →
  p = 2 * Real.sqrt (q + 1) := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l384_38458


namespace NUMINAMATH_CALUDE_sum_fifth_sixth_l384_38439

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q
  sum_first_two : a 1 + a 2 = 3
  sum_third_fourth : a 3 + a 4 = 12

/-- The sum of the fifth and sixth terms equals 48 -/
theorem sum_fifth_sixth (seq : GeometricSequence) : seq.a 5 + seq.a 6 = 48 := by
  sorry

end NUMINAMATH_CALUDE_sum_fifth_sixth_l384_38439


namespace NUMINAMATH_CALUDE_dress_price_ratio_l384_38400

theorem dress_price_ratio (marked_price : ℝ) (marked_price_pos : marked_price > 0) : 
  let discount_rate : ℝ := 1/4
  let selling_price : ℝ := marked_price * (1 - discount_rate)
  let cost_price : ℝ := 2/3 * selling_price
  cost_price / marked_price = 1/2 := by
sorry

end NUMINAMATH_CALUDE_dress_price_ratio_l384_38400


namespace NUMINAMATH_CALUDE_steve_long_letter_time_l384_38414

/-- Represents the writing habits of Steve --/
structure WritingHabits where
  days_between_letters : ℕ
  minutes_per_regular_letter : ℕ
  minutes_per_page : ℕ
  long_letter_time_factor : ℕ
  total_pages_per_month : ℕ
  days_in_month : ℕ

/-- Calculates the time spent on the long letter at the end of the month --/
def long_letter_time (habits : WritingHabits) : ℕ :=
  let regular_letters := habits.days_in_month / habits.days_between_letters
  let pages_per_regular_letter := habits.minutes_per_regular_letter / habits.minutes_per_page
  let regular_letter_pages := regular_letters * pages_per_regular_letter
  let long_letter_pages := habits.total_pages_per_month - regular_letter_pages
  long_letter_pages * (habits.minutes_per_page * habits.long_letter_time_factor)

/-- Theorem stating that Steve spends 80 minutes writing the long letter --/
theorem steve_long_letter_time :
  ∃ (habits : WritingHabits),
    habits.days_between_letters = 3 ∧
    habits.minutes_per_regular_letter = 20 ∧
    habits.minutes_per_page = 10 ∧
    habits.long_letter_time_factor = 2 ∧
    habits.total_pages_per_month = 24 ∧
    habits.days_in_month = 30 ∧
    long_letter_time habits = 80 := by
  sorry

end NUMINAMATH_CALUDE_steve_long_letter_time_l384_38414


namespace NUMINAMATH_CALUDE_inscribing_square_area_l384_38445

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x - 6*y + 24 = 0

-- Define the square that inscribes the circle
structure InscribingSquare :=
  (side_length : ℝ)
  (parallel_to_x_axis : Prop)
  (inscribes_circle : Prop)

-- Theorem statement
theorem inscribing_square_area
  (square : InscribingSquare)
  (h_circle : ∀ x y, circle_equation x y ↔ (x - 4)^2 + (y - 3)^2 = 1) :
  square.side_length^2 = 4 :=
sorry

end NUMINAMATH_CALUDE_inscribing_square_area_l384_38445


namespace NUMINAMATH_CALUDE_product_of_fractions_l384_38449

theorem product_of_fractions : (3 : ℚ) / 8 * 2 / 5 * 1 / 4 = 3 / 80 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l384_38449


namespace NUMINAMATH_CALUDE_binary_operation_theorem_l384_38434

def binary_to_decimal (b : List Bool) : Nat :=
  b.reverse.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : Nat) (acc : List Bool) : List Bool :=
    if m = 0 then acc
    else aux (m / 2) ((m % 2 = 1) :: acc)
  aux n []

def binary_add_subtract (a b c d : List Bool) : List Bool :=
  let sum := binary_to_decimal a + binary_to_decimal b - binary_to_decimal c + binary_to_decimal d
  decimal_to_binary sum

theorem binary_operation_theorem :
  binary_add_subtract [true, true, false, true] [true, true, true] [true, false, true, false] [true, false, false, true] =
  [true, false, false, true, true] := by sorry

end NUMINAMATH_CALUDE_binary_operation_theorem_l384_38434


namespace NUMINAMATH_CALUDE_ratio_and_equation_solution_l384_38417

theorem ratio_and_equation_solution :
  ∀ (x y z b : ℤ),
  (∃ (k : ℤ), x = 3 * k ∧ y = 4 * k ∧ z = 7 * k) →
  y = 15 * b - 5 →
  (b = 3 → (∃ (k : ℤ), x = 3 * k ∧ y = 4 * k ∧ z = 7 * k) ∧ y = 15 * b - 5) :=
by sorry

end NUMINAMATH_CALUDE_ratio_and_equation_solution_l384_38417


namespace NUMINAMATH_CALUDE_zero_intersection_area_l384_38447

-- Define the square pyramid
structure SquarePyramid where
  base_side : ℝ
  slant_edge : ℝ

-- Define the plane passing through midpoints
structure IntersectingPlane where
  pyramid : SquarePyramid

-- Define the intersection area
def intersection_area (plane : IntersectingPlane) : ℝ := sorry

-- Theorem statement
theorem zero_intersection_area 
  (pyramid : SquarePyramid) 
  (h1 : pyramid.base_side = 6) 
  (h2 : pyramid.slant_edge = 5) :
  intersection_area { pyramid := pyramid } = 0 := by sorry

end NUMINAMATH_CALUDE_zero_intersection_area_l384_38447


namespace NUMINAMATH_CALUDE_sum_of_solutions_eq_eight_l384_38408

theorem sum_of_solutions_eq_eight : 
  ∃ (N₁ N₂ : ℝ), N₁ * (N₁ - 8) = -7 ∧ N₂ * (N₂ - 8) = -7 ∧ N₁ + N₂ = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_eq_eight_l384_38408


namespace NUMINAMATH_CALUDE_inequality_proof_l384_38438

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / Real.sqrt (3 * a + 2 * b + c) +
  b / Real.sqrt (3 * b + 2 * c + a) +
  c / Real.sqrt (3 * c + 2 * a + b) ≤
  (1 / Real.sqrt 2) * Real.sqrt (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l384_38438


namespace NUMINAMATH_CALUDE_no_50_cell_crossing_l384_38465

/-- The maximum number of cells a straight line can cross on an m × n grid -/
def maxCrossedCells (m n : ℕ) : ℕ := m + n - Nat.gcd m n

/-- Theorem: On a 20 × 30 grid, it's impossible to draw a straight line that crosses 50 cells -/
theorem no_50_cell_crossing :
  maxCrossedCells 20 30 < 50 := by
  sorry

end NUMINAMATH_CALUDE_no_50_cell_crossing_l384_38465


namespace NUMINAMATH_CALUDE_no_geometric_mean_opposite_signs_l384_38468

/-- The geometric mean of two real numbers does not exist if they have opposite signs -/
theorem no_geometric_mean_opposite_signs (a b : ℝ) (ha : a > 0) (hb : b < 0) :
  ¬∃ (x : ℝ), x^2 = a * b :=
by sorry

end NUMINAMATH_CALUDE_no_geometric_mean_opposite_signs_l384_38468


namespace NUMINAMATH_CALUDE_exactly_one_female_probability_l384_38487

def total_students : ℕ := 50
def male_students : ℕ := 30
def female_students : ℕ := 20
def group_size : ℕ := 5

def male_in_group : ℕ := male_students * group_size / total_students
def female_in_group : ℕ := female_students * group_size / total_students

theorem exactly_one_female_probability : 
  (male_in_group * female_in_group * 2) / (group_size * (group_size - 1)) = 3/5 :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_female_probability_l384_38487


namespace NUMINAMATH_CALUDE_expedition_ratio_l384_38462

/-- Proves the ratio of weeks spent on the last expedition to the second expedition -/
theorem expedition_ratio : 
  ∀ (first second last : ℕ) (total : ℕ),
  first = 3 →
  second = first + 2 →
  total = 7 * (first + second + last) →
  total = 126 →
  last = 2 * second :=
by
  sorry

end NUMINAMATH_CALUDE_expedition_ratio_l384_38462


namespace NUMINAMATH_CALUDE_fraction_nonzero_digits_l384_38441

def fraction := 800 / (2^5 * 5^11)

def count_nonzero_decimal_digits (x : ℚ) : ℕ := sorry

theorem fraction_nonzero_digits :
  count_nonzero_decimal_digits fraction = 3 := by sorry

end NUMINAMATH_CALUDE_fraction_nonzero_digits_l384_38441


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_angle_range_l384_38498

/-- The range of inclination angles for which a line intersects an ellipse at two distinct points -/
theorem line_ellipse_intersection_angle_range 
  (A : ℝ × ℝ) 
  (l : ℝ → ℝ × ℝ) 
  (α : ℝ) 
  (ellipse : ℝ × ℝ → Prop) : 
  A = (-2, 0) →
  (∀ t, l t = (-2 + t * Real.cos α, t * Real.sin α)) →
  (ellipse (x, y) ↔ x^2 / 2 + y^2 = 1) →
  (∃ B C, B ≠ C ∧ ellipse B ∧ ellipse C ∧ ∃ t₁ t₂, l t₁ = B ∧ l t₂ = C) ↔
  (0 ≤ α ∧ α < Real.arcsin (Real.sqrt 3 / 3)) ∨ 
  (Real.pi - Real.arcsin (Real.sqrt 3 / 3) < α ∧ α < Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_angle_range_l384_38498


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l384_38433

theorem decimal_to_fraction :
  (0.36 : ℚ) = 9 / 25 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l384_38433


namespace NUMINAMATH_CALUDE_bankers_interest_rate_l384_38479

/-- Proves that given a time period of 3 years, a banker's gain of 270,
    and a banker's discount of 1020, the rate of interest per annum is 12%. -/
theorem bankers_interest_rate 
  (time : ℕ) (bankers_gain : ℚ) (bankers_discount : ℚ) :
  time = 3 → 
  bankers_gain = 270 → 
  bankers_discount = 1020 → 
  ∃ (rate : ℚ), rate = 12 ∧ 
    bankers_gain = bankers_discount - (bankers_discount / (1 + rate / 100 * time)) :=
by sorry

end NUMINAMATH_CALUDE_bankers_interest_rate_l384_38479


namespace NUMINAMATH_CALUDE_air_conditioner_energy_savings_l384_38431

/-- Represents the monthly energy savings in kWh for an air conditioner type -/
structure EnergySavings where
  savings : ℝ

/-- Represents the two types of air conditioners -/
inductive AirConditionerType
  | A
  | B

/-- The energy savings after raising temperature and cleaning for both air conditioner types -/
def energy_savings_after_measures (x y : EnergySavings) : ℝ :=
  x.savings + 1.1 * y.savings

/-- The theorem to be proved -/
theorem air_conditioner_energy_savings 
  (savings_A savings_B : EnergySavings) :
  savings_A.savings - savings_B.savings = 27 ∧
  energy_savings_after_measures savings_A savings_B = 405 →
  savings_A.savings = 207 ∧ savings_B.savings = 180 := by
  sorry

end NUMINAMATH_CALUDE_air_conditioner_energy_savings_l384_38431


namespace NUMINAMATH_CALUDE_percentage_failed_hindi_l384_38489

theorem percentage_failed_hindi (failed_english : ℝ) (failed_both : ℝ) (passed_both : ℝ) :
  failed_english = 50 →
  failed_both = 25 →
  passed_both = 50 →
  ∃ failed_hindi : ℝ, failed_hindi = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_failed_hindi_l384_38489


namespace NUMINAMATH_CALUDE_quadrilateral_area_l384_38425

/-- Pick's Theorem for quadrilaterals -/
def area_by_picks_theorem (interior_points : ℕ) (boundary_points : ℕ) : ℚ :=
  interior_points + boundary_points / 2 - 1

/-- The quadrilateral in the problem -/
structure Quadrilateral where
  interior_points : ℕ
  boundary_points : ℕ

/-- The specific quadrilateral from the problem -/
def problem_quadrilateral : Quadrilateral where
  interior_points := 12
  boundary_points := 6

theorem quadrilateral_area :
  area_by_picks_theorem problem_quadrilateral.interior_points problem_quadrilateral.boundary_points = 14 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l384_38425


namespace NUMINAMATH_CALUDE_power_equation_implies_m_equals_one_l384_38420

theorem power_equation_implies_m_equals_one (s m : ℕ) :
  (2^16) * (25^s) = 5 * (10^m) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_implies_m_equals_one_l384_38420


namespace NUMINAMATH_CALUDE_f_symmetry_l384_38409

/-- Given a function f(x) = x³ + 2x, prove that f(a) + f(-a) = 0 for any real number a -/
theorem f_symmetry (a : ℝ) : let f (x : ℝ) := x^3 + 2*x; f a + f (-a) = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_symmetry_l384_38409


namespace NUMINAMATH_CALUDE_smallest_sum_of_factors_l384_38477

theorem smallest_sum_of_factors (r s t : ℕ+) (h : r * s * t = 1230) :
  ∃ (r' s' t' : ℕ+), r' * s' * t' = 1230 ∧ r' + s' + t' = 52 ∧
  ∀ (x y z : ℕ+), x * y * z = 1230 → r' + s' + t' ≤ x + y + z :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_factors_l384_38477


namespace NUMINAMATH_CALUDE_special_triangle_sides_l384_38450

/-- A triangle with specific properties -/
structure SpecialTriangle where
  -- Sides of the triangle
  a : ℕ+
  b : ℕ+
  c : ℕ+
  -- The perimeter is a natural number (implied by sides being natural numbers)
  perimeter_nat : (a + b + c : ℕ) > 0
  -- The circumradius is 65/8
  circumradius_eq : (a * b * c : ℚ) / (4 * (a + b + c : ℚ)) = 65 / 8
  -- The inradius is 4
  inradius_eq : (a * b * c : ℚ) / ((a + b + c : ℚ) * (a + b + c - 2 * min a (min b c))) = 4

/-- The sides of the special triangle are (13, 14, 15) -/
theorem special_triangle_sides (t : SpecialTriangle) : t.a = 13 ∧ t.b = 14 ∧ t.c = 15 := by
  sorry


end NUMINAMATH_CALUDE_special_triangle_sides_l384_38450


namespace NUMINAMATH_CALUDE_discount_percentage_l384_38483

theorem discount_percentage
  (CP : ℝ) -- Cost Price
  (MP : ℝ) -- Marked Price
  (SP : ℝ) -- Selling Price
  (MP_condition : MP = CP * 1.5) -- Marked Price is 50% above Cost Price
  (SP_condition : SP = CP * 0.99) -- Selling Price results in 1% loss on Cost Price
  : (MP - SP) / MP * 100 = 34 := by
sorry

end NUMINAMATH_CALUDE_discount_percentage_l384_38483


namespace NUMINAMATH_CALUDE_g_value_at_half_l384_38424

/-- Given a function g : ℝ → ℝ satisfying the equation
    g(x) - 3g(1/x) = 4^x + e^x for all x ≠ 0,
    prove that g(1/2) = (3e^2 - 13√e + 82) / 8 -/
theorem g_value_at_half (g : ℝ → ℝ) 
    (h : ∀ x : ℝ, x ≠ 0 → g x - 3 * g (1/x) = 4^x + Real.exp x) : 
  g (1/2) = (3 * Real.exp 2 - 13 * Real.sqrt (Real.exp 1) + 82) / 8 := by
sorry

end NUMINAMATH_CALUDE_g_value_at_half_l384_38424


namespace NUMINAMATH_CALUDE_x_convergence_l384_38497

def x : ℕ → ℚ
  | 0 => 7
  | n + 1 => (x n ^ 2 + 6 * x n + 5) / (x n + 7)

theorem x_convergence :
  ∃ m : ℕ, 130 ≤ m ∧ m ≤ 240 ∧ x m ≤ 5 + 1 / 2^21 ∧
  ∀ k : ℕ, k < m → x k > 5 + 1 / 2^21 :=
sorry

end NUMINAMATH_CALUDE_x_convergence_l384_38497


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l384_38490

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 1}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l384_38490


namespace NUMINAMATH_CALUDE_decimal_124_to_base_5_has_three_consecutive_digits_l384_38488

/-- Convert a decimal number to base 5 --/
def to_base_5 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

/-- Check if a list of digits has three consecutive identical digits --/
def has_three_consecutive_digits (digits : List ℕ) : Prop :=
  ∃ i, i + 2 < digits.length ∧
       digits[i]! = digits[i+1]! ∧
       digits[i+1]! = digits[i+2]!

/-- The main theorem --/
theorem decimal_124_to_base_5_has_three_consecutive_digits :
  has_three_consecutive_digits (to_base_5 124) :=
sorry

end NUMINAMATH_CALUDE_decimal_124_to_base_5_has_three_consecutive_digits_l384_38488


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l384_38419

theorem rectangle_perimeter (length width : ℝ) : 
  length / width = 4 / 3 →
  length * width = 972 →
  2 * (length + width) = 126 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l384_38419


namespace NUMINAMATH_CALUDE_functional_equation_solution_l384_38437

theorem functional_equation_solution (f g : ℚ → ℚ) 
  (h1 : ∀ x y : ℚ, f (g x - g y) = f (g x) - y)
  (h2 : ∀ x y : ℚ, g (f x - f y) = g (f x) - y) :
  ∃ c : ℚ, c ≠ 0 ∧ (∀ x : ℚ, f x = c * x) ∧ (∀ x : ℚ, g x = x / c) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l384_38437


namespace NUMINAMATH_CALUDE_limit_implies_a_equals_one_l384_38428

theorem limit_implies_a_equals_one (a : ℝ) :
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |((a * n - 2) / (n + 1)) - 1| < ε) →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_limit_implies_a_equals_one_l384_38428


namespace NUMINAMATH_CALUDE_line_intersection_l384_38469

theorem line_intersection (x y : ℚ) :
  (y = 3 * x) ∧ (y - 5 = -6 * x) ↔ (x = 5/9 ∧ y = 5/3) :=
by sorry

end NUMINAMATH_CALUDE_line_intersection_l384_38469


namespace NUMINAMATH_CALUDE_toothpicks_250th_stage_l384_38491

/-- The nth term of an arithmetic sequence -/
def arithmeticSequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

/-- The number of toothpicks in the nth stage of the pattern -/
def toothpicks (n : ℕ) : ℕ :=
  arithmeticSequence 4 3 n

theorem toothpicks_250th_stage :
  toothpicks 250 = 751 := by sorry

end NUMINAMATH_CALUDE_toothpicks_250th_stage_l384_38491


namespace NUMINAMATH_CALUDE_equality_condition_l384_38478

theorem equality_condition (a b c : ℝ) : 
  a = b + c + 2 → (a + b * c = (a + b) * (a + c) ↔ a = 0 ∨ a = 1) := by
  sorry

end NUMINAMATH_CALUDE_equality_condition_l384_38478


namespace NUMINAMATH_CALUDE_probability_jack_or_queen_l384_38435

theorem probability_jack_or_queen (total_cards : ℕ) (jack_queen_count : ℕ) 
  (h1 : total_cards = 104) 
  (h2 : jack_queen_count = 16) : 
  (jack_queen_count : ℚ) / total_cards = 2 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_jack_or_queen_l384_38435


namespace NUMINAMATH_CALUDE_hyperbola_point_coordinate_l384_38494

theorem hyperbola_point_coordinate :
  ∀ x : ℝ,
  (Real.sqrt ((x - 5)^2 + 4^2) - Real.sqrt ((x + 5)^2 + 4^2) = 6) →
  x = -3 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_point_coordinate_l384_38494


namespace NUMINAMATH_CALUDE_system_solution_set_l384_38423

-- Define the system of linear inequalities
def system (x : ℝ) : Prop := x - 2 > 1 ∧ x < 4

-- Define the solution set
def solution_set : Set ℝ := {x | 3 < x ∧ x < 4}

-- Theorem stating that the solution set is correct
theorem system_solution_set : {x : ℝ | system x} = solution_set := by sorry

end NUMINAMATH_CALUDE_system_solution_set_l384_38423


namespace NUMINAMATH_CALUDE_system_solution_range_l384_38459

theorem system_solution_range (x y k : ℝ) : 
  (2 * x + y = k + 1) →
  (x + 2 * y = 2) →
  (x + y < 0) →
  (k < -3) := by
sorry

end NUMINAMATH_CALUDE_system_solution_range_l384_38459


namespace NUMINAMATH_CALUDE_disprove_tangent_line_circle_l384_38443

theorem disprove_tangent_line_circle : ∃ a b : ℝ, a^2 + b^2 ≠ 0 ∧ a^2 + b^2 ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_disprove_tangent_line_circle_l384_38443


namespace NUMINAMATH_CALUDE_complex_squared_plus_2i_l384_38471

theorem complex_squared_plus_2i (i : ℂ) : i^2 = -1 → (1 + i)^2 + 2*i = 4*i := by
  sorry

end NUMINAMATH_CALUDE_complex_squared_plus_2i_l384_38471


namespace NUMINAMATH_CALUDE_intersection_point_solution_l384_38461

-- Define the lines
def line1 (x y : ℝ) : Prop := y = -x + 4
def line2 (x y m : ℝ) : Prop := y = 2*x + m

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := x + y - 4 = 0
def equation2 (x y m : ℝ) : Prop := 2*x - y + m = 0

-- Theorem statement
theorem intersection_point_solution (m n : ℝ) :
  (line1 3 n ∧ line2 3 n m) →
  (equation1 3 1 ∧ equation2 3 1 m) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_solution_l384_38461


namespace NUMINAMATH_CALUDE_quadratic_roots_and_equation_l384_38412

theorem quadratic_roots_and_equation (x₁ x₂ a : ℝ) : 
  (x₁^2 + 4*x₁ - 3 = 0) →
  (x₂^2 + 4*x₂ - 3 = 0) →
  (2*x₁*(x₂^2 + 3*x₂ - 3) + a = 2) →
  (a = -4) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_and_equation_l384_38412


namespace NUMINAMATH_CALUDE_consecutive_odd_sum_remainder_l384_38442

theorem consecutive_odd_sum_remainder (n : ℕ) :
  let sum := (List.range 7).map (λ i => 12157 + 2 * i) |>.sum
  sum % 14 = 7 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_sum_remainder_l384_38442


namespace NUMINAMATH_CALUDE_yard_sale_books_l384_38455

def books_bought (initial_books final_books : ℕ) : ℕ :=
  final_books - initial_books

theorem yard_sale_books : books_bought 35 56 = 21 := by
  sorry

end NUMINAMATH_CALUDE_yard_sale_books_l384_38455


namespace NUMINAMATH_CALUDE_total_jumps_equals_1085_l384_38460

def ronald_jumps : ℕ := 157

def rupert_jumps : ℕ := 3 * ronald_jumps + 23

def rebecca_initial_jumps : ℕ := 47
def rebecca_common_difference : ℕ := 5
def rebecca_sequences : ℕ := 7

def rebecca_last_jumps : ℕ := rebecca_initial_jumps + (rebecca_sequences - 1) * rebecca_common_difference

def rebecca_total_jumps : ℕ := rebecca_sequences * (rebecca_initial_jumps + rebecca_last_jumps) / 2

def total_jumps : ℕ := ronald_jumps + rupert_jumps + rebecca_total_jumps

theorem total_jumps_equals_1085 : total_jumps = 1085 := by
  sorry

end NUMINAMATH_CALUDE_total_jumps_equals_1085_l384_38460


namespace NUMINAMATH_CALUDE_house_wall_planks_l384_38418

theorem house_wall_planks (total_planks small_planks : ℕ) 
  (h1 : total_planks = 29)
  (h2 : small_planks = 17) :
  total_planks - small_planks = 12 := by
  sorry

end NUMINAMATH_CALUDE_house_wall_planks_l384_38418


namespace NUMINAMATH_CALUDE_sqrt_one_sixty_four_l384_38451

theorem sqrt_one_sixty_four : Real.sqrt (1 / 64) = 1 / 8 := by sorry

end NUMINAMATH_CALUDE_sqrt_one_sixty_four_l384_38451


namespace NUMINAMATH_CALUDE_f_even_and_decreasing_l384_38481

def f (x : ℝ) : ℝ := -x^2

theorem f_even_and_decreasing :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f y < f x) := by
  sorry

end NUMINAMATH_CALUDE_f_even_and_decreasing_l384_38481


namespace NUMINAMATH_CALUDE_special_square_smallest_area_l384_38484

/-- A square with specific properties -/
structure SpecialSquare where
  /-- Two vertices lie on the line y = 2x + 3 -/
  vertices_on_line : ℝ → ℝ → Prop
  /-- Two vertices lie on the parabola y = -x^2 + 4x + 5 -/
  vertices_on_parabola : ℝ → ℝ → Prop
  /-- One vertex lies on the origin (0, 0) -/
  vertex_on_origin : Prop

/-- The smallest possible area of a SpecialSquare -/
def smallest_area (s : SpecialSquare) : ℝ := 580

/-- Theorem stating the smallest possible area of a SpecialSquare -/
theorem special_square_smallest_area (s : SpecialSquare) :
  smallest_area s = 580 := by sorry

end NUMINAMATH_CALUDE_special_square_smallest_area_l384_38484


namespace NUMINAMATH_CALUDE_weight_loss_duration_l384_38482

/-- Calculates the number of months required to reach a target weight given initial weight, weight loss per month, and target weight. -/
def months_to_reach_weight (initial_weight : ℕ) (weight_loss_per_month : ℕ) (target_weight : ℕ) : ℕ :=
  (initial_weight - target_weight) / weight_loss_per_month

/-- Proves that it takes 12 months to reduce weight from 250 pounds to 154 pounds, losing 8 pounds per month. -/
theorem weight_loss_duration :
  months_to_reach_weight 250 8 154 = 12 := by
  sorry

end NUMINAMATH_CALUDE_weight_loss_duration_l384_38482


namespace NUMINAMATH_CALUDE_difference_of_squares_representation_l384_38492

theorem difference_of_squares_representation (n : ℕ) : 
  n = 2^4035 → 
  (∃ (count : ℕ), count = 2018 ∧ 
    (∃ (S : Finset (ℕ × ℕ)), 
      S.card = count ∧
      ∀ (pair : ℕ × ℕ), pair ∈ S ↔ 
        (∃ (a b : ℕ), pair = (a, b) ∧ n = a^2 - b^2))) :=
by sorry

end NUMINAMATH_CALUDE_difference_of_squares_representation_l384_38492


namespace NUMINAMATH_CALUDE_modulo_residue_problem_l384_38470

theorem modulo_residue_problem : (348 + 8 * 58 + 9 * 195 + 6 * 29) % 19 = 5 := by
  sorry

end NUMINAMATH_CALUDE_modulo_residue_problem_l384_38470


namespace NUMINAMATH_CALUDE_wax_calculation_l384_38415

theorem wax_calculation (total_wax : ℕ) (additional_wax : ℕ) (possessed_wax : ℕ) : 
  total_wax = 353 → additional_wax = 22 → possessed_wax = total_wax - additional_wax → possessed_wax = 331 := by
  sorry

end NUMINAMATH_CALUDE_wax_calculation_l384_38415


namespace NUMINAMATH_CALUDE_unique_f_exists_and_power_of_two_property_l384_38432

def is_valid_f (f : ℕ+ → ℕ+) : Prop :=
  f 1 = 1 ∧ f 2 = 1 ∧ ∀ n ≥ 3, f n = f (f (n-1)) + f (n - f (n-1))

theorem unique_f_exists_and_power_of_two_property :
  ∃! f : ℕ+ → ℕ+, is_valid_f f ∧ ∀ m : ℕ, m ≥ 1 → f (2^m) = 2^(m-1) :=
sorry

end NUMINAMATH_CALUDE_unique_f_exists_and_power_of_two_property_l384_38432
