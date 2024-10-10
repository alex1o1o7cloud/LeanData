import Mathlib

namespace smallest_square_tiling_l380_38043

/-- The smallest square perfectly tiled by 3x4 rectangles -/
def smallest_tiled_square : ℕ := 12

/-- The number of 3x4 rectangles needed to tile the smallest square -/
def num_rectangles : ℕ := 9

/-- The area of a 3x4 rectangle -/
def rectangle_area : ℕ := 3 * 4

theorem smallest_square_tiling :
  (smallest_tiled_square * smallest_tiled_square) % rectangle_area = 0 ∧
  num_rectangles * rectangle_area = smallest_tiled_square * smallest_tiled_square ∧
  ∀ n : ℕ, n < smallest_tiled_square → (n * n) % rectangle_area ≠ 0 := by
  sorry

#check smallest_square_tiling

end smallest_square_tiling_l380_38043


namespace problem_statement_l380_38001

theorem problem_statement (a b : ℝ) (h : 2*a + b + 1 = 0) : 1 + 4*a + 2*b = -1 := by
  sorry

end problem_statement_l380_38001


namespace no_real_solutions_l380_38090

theorem no_real_solutions :
  ¬∃ (x : ℝ), x + 2 * Real.sqrt (x - 1) = 6 := by
  sorry

end no_real_solutions_l380_38090


namespace rain_probability_l380_38080

theorem rain_probability (monday_rain : ℝ) (tuesday_rain : ℝ) (no_rain : ℝ)
  (h1 : monday_rain = 0.7)
  (h2 : tuesday_rain = 0.5)
  (h3 : no_rain = 0.2) :
  monday_rain + tuesday_rain - (1 - no_rain) = 0.4 := by
  sorry

end rain_probability_l380_38080


namespace monotone_increasing_condition_l380_38048

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 2

-- State the theorem
theorem monotone_increasing_condition (a : ℝ) :
  (∀ x ≥ 2, Monotone (f a)) → a ≥ -2 :=
by sorry

end monotone_increasing_condition_l380_38048


namespace problem_solution_l380_38023

def f (x : ℝ) : ℝ := |2*x - 1| + |2*x + 2|

theorem problem_solution :
  (∃ (M : ℝ), (∀ x, f x ≥ M) ∧ (∃ x, f x = M) ∧ M = 3) ∧
  (∀ x, f x < 3 + |2*x + 2| ↔ -1 < x ∧ x < 2) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a^2 + 2*b^2 = 3 → 2*a + b ≤ 3*Real.sqrt 6 / 2) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^2 + 2*b^2 = 3 ∧ 2*a + b = 3*Real.sqrt 6 / 2) :=
by sorry

end problem_solution_l380_38023


namespace algebraic_expression_value_l380_38027

theorem algebraic_expression_value (a b c : ℤ) 
  (h1 : a - b = 3) 
  (h2 : b + c = -5) : 
  a * c - b * c + a^2 - a * b = -6 := by sorry

end algebraic_expression_value_l380_38027


namespace gcd_of_128_144_480_l380_38000

theorem gcd_of_128_144_480 : Nat.gcd 128 (Nat.gcd 144 480) = 16 := by
  sorry

end gcd_of_128_144_480_l380_38000


namespace x_squared_minus_four_y_squared_plus_one_equals_negative_three_l380_38021

theorem x_squared_minus_four_y_squared_plus_one_equals_negative_three 
  (x y : ℝ) (h1 : x + 2*y = 4) (h2 : x - 2*y = -1) : 
  x^2 - 4*y^2 + 1 = -3 := by
  sorry

end x_squared_minus_four_y_squared_plus_one_equals_negative_three_l380_38021


namespace remainder_777_pow_777_mod_13_l380_38067

theorem remainder_777_pow_777_mod_13 : 777^777 % 13 = 1 := by
  sorry

end remainder_777_pow_777_mod_13_l380_38067


namespace song_time_is_125_minutes_l380_38069

/-- Represents the duration of a radio show in minutes -/
def total_show_time : ℕ := 3 * 60

/-- Represents the duration of a single talking segment in minutes -/
def talking_segment_duration : ℕ := 10

/-- Represents the duration of a single ad break in minutes -/
def ad_break_duration : ℕ := 5

/-- Represents the number of talking segments in the show -/
def num_talking_segments : ℕ := 3

/-- Represents the number of ad breaks in the show -/
def num_ad_breaks : ℕ := 5

/-- Calculates the total time spent on talking segments -/
def total_talking_time : ℕ := talking_segment_duration * num_talking_segments

/-- Calculates the total time spent on ad breaks -/
def total_ad_time : ℕ := ad_break_duration * num_ad_breaks

/-- Calculates the total time spent on non-song content -/
def total_non_song_time : ℕ := total_talking_time + total_ad_time

/-- Theorem: The remaining time for songs in the radio show is 125 minutes -/
theorem song_time_is_125_minutes : 
  total_show_time - total_non_song_time = 125 := by sorry

end song_time_is_125_minutes_l380_38069


namespace triangle_area_l380_38012

theorem triangle_area (a b c : ℝ) (h_right_angle : a^2 + b^2 = c^2)
  (h_angle : a / c = 1 / 2) (h_hypotenuse : c = 40) :
  (1 / 2) * a * b = 200 * Real.sqrt 3 := by
  sorry

end triangle_area_l380_38012


namespace ratio_S4_a3_l380_38009

/-- Given a sequence {a_n} where S_n is the sum of its first n terms -/
def S (n : ℕ) : ℚ := 2^n - 1

/-- The n-th term of the sequence -/
def a (n : ℕ) : ℚ := S n - S (n-1)

/-- The theorem to prove -/
theorem ratio_S4_a3 : S 4 / a 3 = 15/4 := by sorry

end ratio_S4_a3_l380_38009


namespace food_additives_percentage_l380_38064

/-- Represents the budget allocation for a research category -/
structure BudgetAllocation where
  percentage : ℝ
  degrees : ℝ

/-- Represents the total budget and its allocations -/
structure Budget where
  total_degrees : ℝ
  microphotonics : BudgetAllocation
  home_electronics : BudgetAllocation
  genetically_modified_microorganisms : BudgetAllocation
  industrial_lubricants : BudgetAllocation
  basic_astrophysics : BudgetAllocation
  food_additives : BudgetAllocation

/-- The Megatech Corporation's research and development budget -/
def megatech_budget : Budget := {
  total_degrees := 360
  microphotonics := { percentage := 14, degrees := 0 }
  home_electronics := { percentage := 24, degrees := 0 }
  genetically_modified_microorganisms := { percentage := 19, degrees := 0 }
  industrial_lubricants := { percentage := 8, degrees := 0 }
  basic_astrophysics := { percentage := 0, degrees := 72 }
  food_additives := { percentage := 0, degrees := 0 }
}

/-- Theorem: The percentage of the budget allocated to food additives is 15% -/
theorem food_additives_percentage : megatech_budget.food_additives.percentage = 15 := by
  sorry


end food_additives_percentage_l380_38064


namespace a_completion_time_l380_38037

def job_completion_time (a b c : ℝ) : Prop :=
  (1 / b = 8) ∧ 
  (1 / c = 12) ∧ 
  (2340 / (1 / a + 1 / b + 1 / c) = 780 / (1 / b))

theorem a_completion_time (a b c : ℝ) : 
  job_completion_time a b c → 1 / a = 6 := by
  sorry

end a_completion_time_l380_38037


namespace third_shot_probability_l380_38014

-- Define the probability of hitting the target in one shot
def hit_probability : ℝ := 0.9

-- Define the number of shots
def num_shots : ℕ := 4

-- Define the event of hitting the target on the nth shot
def hit_on_nth_shot (n : ℕ) : ℝ := hit_probability

-- Theorem statement
theorem third_shot_probability :
  hit_on_nth_shot 3 = hit_probability :=
by sorry

end third_shot_probability_l380_38014


namespace greatest_3digit_base7_divisible_by_7_l380_38038

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (a b c : Nat) : Nat :=
  a * 7^2 + b * 7 + c

/-- Checks if a number is a valid 3-digit base 7 number --/
def isValidBase7 (a b c : Nat) : Prop :=
  a > 0 ∧ a < 7 ∧ b < 7 ∧ c < 7

/-- The proposed solution in base 7 --/
def solution : (Nat × Nat × Nat) := (6, 6, 0)

theorem greatest_3digit_base7_divisible_by_7 :
  let (a, b, c) := solution
  isValidBase7 a b c ∧
  base7ToBase10 a b c % 7 = 0 ∧
  ∀ x y z, isValidBase7 x y z → 
    base7ToBase10 x y z % 7 = 0 → 
    base7ToBase10 x y z ≤ base7ToBase10 a b c :=
by sorry

end greatest_3digit_base7_divisible_by_7_l380_38038


namespace painting_difference_l380_38062

/-- Represents a 5x5x5 cube -/
structure Cube :=
  (size : Nat)
  (h_size : size = 5)

/-- Counts the number of unit cubes with at least one painted face when two opposite faces and one additional face are painted -/
def count_painted_opposite_plus_one (c : Cube) : Nat :=
  c.size * c.size + (c.size - 2) * c.size + c.size * c.size

/-- Counts the number of unit cubes with at least one painted face when three adjacent faces sharing one vertex are painted -/
def count_painted_adjacent (c : Cube) : Nat :=
  (c.size - 1) * 9 + c.size * c.size

/-- The difference between the two painting configurations is 4 -/
theorem painting_difference (c : Cube) : 
  count_painted_opposite_plus_one c - count_painted_adjacent c = 4 := by
  sorry


end painting_difference_l380_38062


namespace digital_earth_technologies_l380_38074

-- Define the set of all possible technologies
def AllTechnologies : Set String :=
  {"Sustainable development", "Global positioning technology", "Geographic information system",
   "Global positioning system", "Virtual technology", "Network technology"}

-- Define the digital Earth as a complex computer technology system
structure DigitalEarth where
  technologies : Set String
  complex : Bool
  integrates_various_tech : Bool

-- Define the supporting technologies for the digital Earth
def SupportingTechnologies (de : DigitalEarth) : Set String := de.technologies

-- Theorem statement
theorem digital_earth_technologies (de : DigitalEarth) 
  (h1 : de.complex = true) 
  (h2 : de.integrates_various_tech = true) : 
  SupportingTechnologies de = AllTechnologies := by
  sorry

end digital_earth_technologies_l380_38074


namespace f_properties_l380_38096

noncomputable def f (t : ℝ) (x : ℝ) : ℝ := x^3 - (3*(t+1)/2)*x^2 + 3*t*x + 1

theorem f_properties (t : ℝ) (h : t > 0) :
  (∃ (max : ℝ), t = 2 → ∀ x, f t x ≤ max ∧ ∃ y, f t y = max) ∧
  (∃ (a b : ℝ), a < b ∧ a > 0 ∧ b ≤ 1/3 ∧
    (∀ t', a < t' ∧ t' ≤ b →
      ∃ x₀, 0 < x₀ ∧ x₀ < 2 ∧ ∀ x, 0 ≤ x ∧ x ≤ 2 → f t' x₀ ≤ f t' x)) ∧
  (∃ (a b : ℝ), a < b ∧ a > 0 ∧ b ≤ 1/3 ∧
    (∀ t', a < t' ∧ t' ≤ b →
      ∀ x, x ≥ 0 → f t' x ≤ x * Real.exp x + 1)) :=
by sorry

end f_properties_l380_38096


namespace exists_winning_strategy_for_first_player_l380_38006

/-- Represents the state of the orange game -/
structure GameState :=
  (oranges : ℕ)
  (player_turn : Bool)

/-- Defines a valid move in the game -/
def valid_move (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 5

/-- Applies a move to the game state -/
def apply_move (state : GameState) (move : ℕ) : GameState :=
  { oranges := state.oranges - move
  , player_turn := ¬state.player_turn }

/-- Determines if a game state is winning for the current player -/
def is_winning_state (state : GameState) : Prop :=
  state.oranges = 0

/-- Defines a winning strategy for the game -/
def winning_strategy (strategy : GameState → ℕ) : Prop :=
  ∀ (state : GameState),
    valid_move (strategy state) ∧
    (is_winning_state (apply_move state (strategy state)) ∨
     ∀ (opponent_move : ℕ),
       valid_move opponent_move →
       ¬is_winning_state (apply_move (apply_move state (strategy state)) opponent_move))

/-- Theorem stating that there exists a winning strategy for the first player in the 100-orange game -/
theorem exists_winning_strategy_for_first_player :
  ∃ (strategy : GameState → ℕ),
    winning_strategy strategy ∧
    strategy { oranges := 100, player_turn := true } = 4 :=
sorry

end exists_winning_strategy_for_first_player_l380_38006


namespace probability_ella_zoe_same_team_l380_38033

/-- The number of cards in the deck -/
def deck_size : ℕ := 52

/-- The card number chosen by Ella -/
def b : ℕ := 11

/-- The probability that Ella and Zoe are on the same team -/
def p (b : ℕ) : ℚ :=
  let remaining_cards := deck_size - 2
  let total_combinations := remaining_cards.choose 2
  let lower_team_combinations := (b - 1).choose 2
  let higher_team_combinations := (deck_size - b - 11).choose 2
  (lower_team_combinations + higher_team_combinations : ℚ) / total_combinations

theorem probability_ella_zoe_same_team :
  p b = 857 / 1225 :=
sorry

end probability_ella_zoe_same_team_l380_38033


namespace product_in_unit_interval_sufficient_not_necessary_l380_38051

theorem product_in_unit_interval_sufficient_not_necessary (a b : ℝ) :
  (((0 ≤ a ∧ a ≤ 1) ∧ (0 ≤ b ∧ b ≤ 1)) → (0 ≤ a * b ∧ a * b ≤ 1)) ∧
  ¬(((0 ≤ a * b ∧ a * b ≤ 1) → ((0 ≤ a ∧ a ≤ 1) ∧ (0 ≤ b ∧ b ≤ 1)))) :=
by sorry

end product_in_unit_interval_sufficient_not_necessary_l380_38051


namespace polar_to_rectangular_l380_38059

/-- Conversion from polar to rectangular coordinates -/
theorem polar_to_rectangular (r θ : ℝ) :
  r > 0 →
  θ = 11 * π / 6 →
  (r * Real.cos θ, r * Real.sin θ) = (5 * Real.sqrt 3, -5) := by
  sorry

end polar_to_rectangular_l380_38059


namespace cinnamon_distribution_exists_l380_38047

/-- Represents the number of cinnamon swirls eaten by each person -/
structure CinnamonDistribution where
  jane : ℕ
  siblings : Fin 2 → ℕ
  cousins : Fin 5 → ℕ

/-- Theorem stating the existence of a valid cinnamon swirl distribution -/
theorem cinnamon_distribution_exists : ∃ (d : CinnamonDistribution), 
  -- Each person eats a different number of pieces
  (∀ (i j : Fin 2), i ≠ j → d.siblings i ≠ d.siblings j) ∧ 
  (∀ (i j : Fin 5), i ≠ j → d.cousins i ≠ d.cousins j) ∧
  (∀ (i : Fin 2) (j : Fin 5), d.siblings i ≠ d.cousins j) ∧
  (∀ (i : Fin 2), d.jane ≠ d.siblings i) ∧
  (∀ (j : Fin 5), d.jane ≠ d.cousins j) ∧
  -- Jane eats 1 fewer piece than her youngest sibling
  (∃ (i : Fin 2), d.jane + 1 = d.siblings i ∧ ∀ (j : Fin 2), d.siblings j ≥ d.siblings i) ∧
  -- Jane's youngest sibling eats 2 pieces more than one of her cousins
  (∃ (i : Fin 2) (j : Fin 5), d.siblings i = d.cousins j + 2 ∧ ∀ (k : Fin 2), d.siblings k ≥ d.siblings i) ∧
  -- The sum of all pieces eaten equals 50
  d.jane + (Finset.sum (Finset.univ : Finset (Fin 2)) d.siblings) + (Finset.sum (Finset.univ : Finset (Fin 5)) d.cousins) = 50 :=
sorry

end cinnamon_distribution_exists_l380_38047


namespace shaded_area_square_with_triangles_l380_38005

/-- The area of the shaded region in a square with two unshaded triangles -/
theorem shaded_area_square_with_triangles : 
  let square_side : ℝ := 50
  let triangle1_base : ℝ := 20
  let triangle1_height : ℝ := 20
  let triangle2_base : ℝ := 20
  let triangle2_height : ℝ := 20
  let square_area := square_side * square_side
  let triangle1_area := (1/2) * triangle1_base * triangle1_height
  let triangle2_area := (1/2) * triangle2_base * triangle2_height
  let total_triangle_area := triangle1_area + triangle2_area
  let shaded_area := square_area - total_triangle_area
  shaded_area = 2100 := by sorry

end shaded_area_square_with_triangles_l380_38005


namespace orange_box_problem_l380_38060

theorem orange_box_problem (box1_capacity box2_capacity : ℕ) 
  (box1_fill_fraction : ℚ) (total_oranges : ℕ) :
  box1_capacity = 80 →
  box2_capacity = 50 →
  box1_fill_fraction = 3/4 →
  total_oranges = 90 →
  ∃ (box2_fill_fraction : ℚ),
    box2_fill_fraction = 3/5 ∧
    (box1_capacity : ℚ) * box1_fill_fraction + (box2_capacity : ℚ) * box2_fill_fraction = total_oranges := by
  sorry

end orange_box_problem_l380_38060


namespace cube_cutting_l380_38007

theorem cube_cutting (n : ℕ) : n > 0 → 6 * (n - 2)^2 = 54 → n^3 = 125 := by
  sorry

end cube_cutting_l380_38007


namespace average_speed_two_part_trip_l380_38036

/-- Calculates the average speed of a two-part trip -/
theorem average_speed_two_part_trip
  (total_distance : ℝ)
  (distance1 : ℝ)
  (speed1 : ℝ)
  (distance2 : ℝ)
  (speed2 : ℝ)
  (h1 : total_distance = distance1 + distance2)
  (h2 : distance1 = 35)
  (h3 : distance2 = 35)
  (h4 : speed1 = 48)
  (h5 : speed2 = 24)
  (h6 : total_distance = 70) :
  ∃ (avg_speed : ℝ), abs (avg_speed - 32) < 0.1 ∧
  avg_speed = total_distance / (distance1 / speed1 + distance2 / speed2) := by
  sorry


end average_speed_two_part_trip_l380_38036


namespace union_of_M_and_N_l380_38025

-- Define set M
def M : Set ℝ := {x : ℝ | -3 < x ∧ x ≤ 5}

-- Define set N
def N : Set ℝ := {x : ℝ | x < -5 ∨ x > 5}

-- Theorem statement
theorem union_of_M_and_N : M ∪ N = {x : ℝ | x < -5 ∨ x > -3} := by
  sorry

end union_of_M_and_N_l380_38025


namespace distinct_circular_arrangements_l380_38079

/-- The number of distinct circular arrangements of girls and boys -/
def circularArrangements (girls boys : ℕ) : ℕ :=
  (Nat.factorial 16 * Nat.factorial 25) / Nat.factorial 9

/-- Theorem stating the number of distinct circular arrangements -/
theorem distinct_circular_arrangements :
  circularArrangements 8 25 = (Nat.factorial 16 * Nat.factorial 25) / Nat.factorial 9 :=
by sorry

end distinct_circular_arrangements_l380_38079


namespace orange_juice_fraction_l380_38018

theorem orange_juice_fraction : 
  let pitcher_capacity : ℚ := 600
  let pitcher1_fraction : ℚ := 1/3
  let pitcher2_fraction : ℚ := 2/5
  let orange_juice1 : ℚ := pitcher_capacity * pitcher1_fraction
  let orange_juice2 : ℚ := pitcher_capacity * pitcher2_fraction
  let total_orange_juice : ℚ := orange_juice1 + orange_juice2
  let total_mixture : ℚ := pitcher_capacity * 2
  total_orange_juice / total_mixture = 11/30 := by
sorry


end orange_juice_fraction_l380_38018


namespace bouquet_calculation_l380_38034

def max_bouquets (total_flowers : ℕ) (flowers_per_bouquet : ℕ) (wilted_flowers : ℕ) : ℕ :=
  (total_flowers - wilted_flowers) / flowers_per_bouquet

theorem bouquet_calculation (total_flowers : ℕ) (flowers_per_bouquet : ℕ) (wilted_flowers : ℕ) 
  (h1 : total_flowers = 53)
  (h2 : flowers_per_bouquet = 7)
  (h3 : wilted_flowers = 18) :
  max_bouquets total_flowers flowers_per_bouquet wilted_flowers = 5 := by
  sorry

end bouquet_calculation_l380_38034


namespace fish_count_difference_l380_38020

theorem fish_count_difference (n G S R : ℕ) : 
  n > 0 → 
  n = G + S + R → 
  n - G = (2 * n) / 3 - 1 → 
  n - R = (2 * n) / 3 + 4 → 
  S = G + 2 :=
by sorry

end fish_count_difference_l380_38020


namespace sequence_with_least_period_l380_38089

theorem sequence_with_least_period (p : ℕ) (h : p ≥ 2) :
  ∃ (x : ℕ → ℝ), 
    (∀ n, x (n + p) = x n) ∧ 
    (∀ n, x (n + 1) = x n - 1 / x n) ∧
    (∀ k, k < p → ¬(∀ n, x (n + k) = x n)) := by
  sorry

end sequence_with_least_period_l380_38089


namespace complex_polynomial_roots_l380_38070

theorem complex_polynomial_roots (c : ℂ) : 
  (∃ (P : ℂ → ℂ), P = (fun x ↦ (x^2 - 2*x + 2) * (x^2 - c*x + 4) * (x^2 - 4*x + 8)) ∧ 
   (∃ (r1 r2 r3 r4 : ℂ), r1 ≠ r2 ∧ r1 ≠ r3 ∧ r1 ≠ r4 ∧ r2 ≠ r3 ∧ r2 ≠ r4 ∧ r3 ≠ r4 ∧
    ∀ x, P x = 0 ↔ x = r1 ∨ x = r2 ∨ x = r3 ∨ x = r4)) →
  Complex.abs c = Real.sqrt 10 := by
sorry

end complex_polynomial_roots_l380_38070


namespace q_share_approx_l380_38085

/-- Represents a partner in the partnership -/
inductive Partner
| P
| Q
| R

/-- Calculates the share ratio for a given partner -/
def shareRatio (partner : Partner) : Rat :=
  match partner with
  | Partner.P => 1/2
  | Partner.Q => 1/3
  | Partner.R => 1/4

/-- Calculates the investment duration for a given partner in months -/
def investmentDuration (partner : Partner) : ℕ :=
  match partner with
  | Partner.P => 2
  | _ => 12

/-- Calculates the capital ratio after p's withdrawal -/
def capitalRatioAfterWithdrawal (partner : Partner) : Rat :=
  match partner with
  | Partner.P => 1/4
  | _ => shareRatio partner

/-- The total profit in Rs -/
def totalProfit : ℕ := 378

/-- The total duration of the partnership in months -/
def totalDuration : ℕ := 12

/-- Calculates the investment parts for a given partner -/
def investmentParts (partner : Partner) : Rat :=
  (shareRatio partner * investmentDuration partner) +
  (capitalRatioAfterWithdrawal partner * (totalDuration - investmentDuration partner))

/-- Theorem stating that Q's share of the profit is approximately 123.36 Rs -/
theorem q_share_approx (ε : ℝ) (h : ε > 0) :
  ∃ (q_share : ℝ), abs (q_share - 123.36) < ε ∧
  q_share = (investmentParts Partner.Q / (investmentParts Partner.P + investmentParts Partner.Q + investmentParts Partner.R)) * totalProfit :=
sorry

end q_share_approx_l380_38085


namespace first_pipe_rate_correct_l380_38013

/-- The rate at which the first pipe pumps water (in gallons per hour) -/
def first_pipe_rate : ℝ := 48

/-- The rate at which the second pipe pumps water (in gallons per hour) -/
def second_pipe_rate : ℝ := 192

/-- The capacity of the well in gallons -/
def well_capacity : ℝ := 1200

/-- The time it takes to fill the well in hours -/
def fill_time : ℝ := 5

theorem first_pipe_rate_correct : 
  first_pipe_rate * fill_time + second_pipe_rate * fill_time = well_capacity := by
  sorry

end first_pipe_rate_correct_l380_38013


namespace oil_leak_calculation_l380_38081

theorem oil_leak_calculation (total_leak : ℕ) (initial_leak : ℕ) (h1 : total_leak = 11687) (h2 : initial_leak = 6522) :
  total_leak - initial_leak = 5165 := by
  sorry

end oil_leak_calculation_l380_38081


namespace range_of_a_l380_38031

/-- A function f is monotonically increasing -/
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- The logarithm function with base a -/
noncomputable def log_base (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

/-- Proposition p: log_a x is monotonically increasing for x > 0 -/
def p (a : ℝ) : Prop :=
  MonotonicallyIncreasing (fun x => log_base a x)

/-- Proposition q: x^2 + ax + 1 > 0 for all real x -/
def q (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + a*x + 1 > 0

/-- The theorem stating the range of a given the conditions -/
theorem range_of_a (a : ℝ) : 
  ((p a ∨ q a) ∧ ¬(p a ∧ q a)) → a ∈ Set.Ioc (-2) 1 ∪ Set.Ici 2 :=
sorry

end range_of_a_l380_38031


namespace vector_expression_not_equal_AD_l380_38042

/-- Given vectors in a plane or space, prove that the expression
    (MB + AD) - BM is not equal to AD. -/
theorem vector_expression_not_equal_AD
  (A B C D M O : EuclideanSpace ℝ (Fin n)) :
  (M - B + (A - D)) - (B - M) ≠ A - D := by sorry

end vector_expression_not_equal_AD_l380_38042


namespace max_value_of_f_l380_38008

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

-- State the theorem
theorem max_value_of_f :
  ∃ (m : ℝ), m = 8 ∧ ∀ (x : ℝ), x ∈ Set.Icc (-1) 4 → f x ≤ m :=
by sorry

end max_value_of_f_l380_38008


namespace same_color_probability_l380_38010

/-- The probability of drawing two balls of the same color from a box of 6 balls -/
theorem same_color_probability : ℝ := by
  -- Define the number of balls of each color
  let red_balls : ℕ := 3
  let yellow_balls : ℕ := 2
  let blue_balls : ℕ := 1

  -- Define the total number of balls
  let total_balls : ℕ := red_balls + yellow_balls + blue_balls

  -- Define the probability of drawing two balls of the same color
  let prob : ℝ := 4 / 15

  -- Proof goes here
  sorry

end same_color_probability_l380_38010


namespace melanie_dimes_given_to_dad_l380_38078

theorem melanie_dimes_given_to_dad (initial_dimes : ℕ) (dimes_from_mother : ℕ) (final_dimes : ℕ) :
  initial_dimes = 7 →
  dimes_from_mother = 4 →
  final_dimes = 3 →
  initial_dimes + dimes_from_mother - final_dimes = 8 := by
sorry

end melanie_dimes_given_to_dad_l380_38078


namespace election_votes_calculation_l380_38004

theorem election_votes_calculation (total_votes : ℕ) : 
  (80 : ℚ) / 100 * ((100 : ℚ) - 15) / 100 * total_votes = 380800 →
  total_votes = 560000 := by
sorry

end election_votes_calculation_l380_38004


namespace weighted_average_constants_l380_38099

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define points A, B, C, P, Q
variable (A B C P Q : V)

-- Define the conditions
variable (hAPC : ∃ (k : ℝ), P - C = k • (A - C) ∧ k = 4/5)
variable (hBQC : ∃ (k : ℝ), Q - C = k • (B - C) ∧ k = 1/5)

-- Define r and s
variable (r s : ℝ)

-- Define the weighted average conditions
variable (hP : P = r • A + (1 - r) • C)
variable (hQ : Q = s • B + (1 - s) • C)

-- State the theorem
theorem weighted_average_constants : r = 1/5 ∧ s = 4/5 := by sorry

end weighted_average_constants_l380_38099


namespace min_coins_for_alex_l380_38055

/-- The minimum number of additional coins needed for distribution. -/
def min_additional_coins (num_friends : ℕ) (initial_coins : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_coins

/-- Theorem stating the minimum number of additional coins needed for the given scenario. -/
theorem min_coins_for_alex : min_additional_coins 15 63 = 57 := by
  sorry

end min_coins_for_alex_l380_38055


namespace sufficient_not_necessary_l380_38093

theorem sufficient_not_necessary : 
  (∀ x : ℝ, 1 < x ∧ x < 2 → x < 2) ∧ 
  (∃ x : ℝ, x < 2 ∧ ¬(1 < x ∧ x < 2)) := by
  sorry

end sufficient_not_necessary_l380_38093


namespace least_common_multiple_first_ten_l380_38061

theorem least_common_multiple_first_ten : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ 10 → k > 0 → n % k = 0) ∧ 
  (∀ m : ℕ, m < n → ∃ k : ℕ, k ≤ 10 ∧ k > 0 ∧ m % k ≠ 0) :=
by
  use 2520
  sorry

end least_common_multiple_first_ten_l380_38061


namespace abc_divides_sum_power_seven_l380_38082

theorem abc_divides_sum_power_seven (a b c : ℕ+) 
  (hab : a ∣ b^2) (hbc : b ∣ c^2) (hca : c ∣ a^2) : 
  (a * b * c) ∣ (a + b + c)^7 := by
  sorry

end abc_divides_sum_power_seven_l380_38082


namespace mark_vaccine_waiting_time_l380_38044

/-- Calculates the total waiting time in minutes for Mark's vaccine appointments and effectiveness periods -/
def total_waiting_time : ℕ :=
  let first_vaccine_wait := 4
  let second_vaccine_wait := 20
  let secondary_first_dose_wait := 30 + 10
  let secondary_second_dose_wait := 14 + 3
  let effectiveness_wait := 21
  let total_days := first_vaccine_wait + second_vaccine_wait + secondary_first_dose_wait + 
                    secondary_second_dose_wait + effectiveness_wait
  total_days * 24 * 60

theorem mark_vaccine_waiting_time :
  total_waiting_time = 146880 := by
  sorry

end mark_vaccine_waiting_time_l380_38044


namespace larger_number_proof_l380_38030

/-- Given two positive integers with HCF 23 and LCM factors 11 and 12, the larger is 276 -/
theorem larger_number_proof (a b : ℕ+) : 
  (Nat.gcd a.val b.val = 23) → 
  (∃ (k : ℕ+), Nat.lcm a.val b.val = 23 * 11 * 12 * k.val) → 
  max a b = 276 := by
sorry

end larger_number_proof_l380_38030


namespace cistern_theorem_l380_38057

/-- Represents the cistern problem -/
def cistern_problem (capacity : ℝ) (leak_time : ℝ) (tap_rate : ℝ) : Prop :=
  let leak_rate : ℝ := capacity / leak_time
  let net_rate : ℝ := leak_rate - tap_rate
  let emptying_time : ℝ := capacity / net_rate
  emptying_time = 24

/-- The theorem statement for the cistern problem -/
theorem cistern_theorem :
  cistern_problem 480 20 4 := by sorry

end cistern_theorem_l380_38057


namespace midpoint_value_part1_midpoint_value_part2_l380_38086

-- Definition of midpoint value
def is_midpoint_value (a b : ℝ) : Prop := a^2 - b > 0

-- Part 1
theorem midpoint_value_part1 : 
  is_midpoint_value 4 3 ∧ ∀ x, x^2 - 8*x + 3 = 0 ↔ x^2 - 2*4*x + 3 = 0 :=
sorry

-- Part 2
theorem midpoint_value_part2 (m n : ℝ) : 
  (is_midpoint_value 3 n ∧ 
   ∀ x, x^2 - m*x + n = 0 ↔ x^2 - 2*3*x + n = 0 ∧
   (n^2 - m*n + n = 0)) →
  (n = 0 ∨ n = 5) :=
sorry

end midpoint_value_part1_midpoint_value_part2_l380_38086


namespace inequality_solution_l380_38071

theorem inequality_solution (x : ℝ) : 
  3 - 2 / (3 * x + 2) < 5 ↔ x < -1 ∨ x > -2/3 :=
by sorry

end inequality_solution_l380_38071


namespace cylinder_lateral_surface_area_l380_38045

/-- The lateral surface area of a cylinder with a square axial cross-section -/
theorem cylinder_lateral_surface_area (s : ℝ) (h : s = 10) :
  let circumference := s * Real.pi
  let height := s
  height * circumference = 100 * Real.pi := by
  sorry

end cylinder_lateral_surface_area_l380_38045


namespace original_rectangle_area_l380_38058

theorem original_rectangle_area (new_area : ℝ) (h1 : new_area = 32) : ∃ original_area : ℝ,
  (original_area * 4 = new_area) ∧ original_area = 8 := by
  sorry

end original_rectangle_area_l380_38058


namespace distance_to_hypotenuse_l380_38003

/-- A right triangle with specific properties -/
structure RightTriangle where
  /-- The length of one leg of the triangle -/
  leg1 : ℝ
  /-- The length of the other leg of the triangle -/
  leg2 : ℝ
  /-- The distance from the intersection point of the medians to one leg -/
  dist1 : ℝ
  /-- The distance from the intersection point of the medians to the other leg -/
  dist2 : ℝ
  /-- Ensure the triangle is not degenerate -/
  leg1_pos : 0 < leg1
  leg2_pos : 0 < leg2
  /-- The distances from the intersection point to the legs are positive -/
  dist1_pos : 0 < dist1
  dist2_pos : 0 < dist2
  /-- The given distances from the intersection point to the legs -/
  dist1_eq : dist1 = 3
  dist2_eq : dist2 = 4

/-- The theorem to be proved -/
theorem distance_to_hypotenuse (t : RightTriangle) : 
  let hypotenuse := Real.sqrt (t.leg1^2 + t.leg2^2)
  let area := t.leg1 * t.leg2 / 2
  area / hypotenuse = 12/5 := by sorry

end distance_to_hypotenuse_l380_38003


namespace max_abs_quadratic_on_interval_l380_38068

/-- The function f(x) = |x^2 - 2x - t| with maximum value 2 on [0, 3] implies t = 1 -/
theorem max_abs_quadratic_on_interval (t : ℝ) : 
  (∀ x ∈ Set.Icc 0 3, |x^2 - 2*x - t| ≤ 2) ∧ 
  (∃ x ∈ Set.Icc 0 3, |x^2 - 2*x - t| = 2) →
  t = 1 := by
  sorry

end max_abs_quadratic_on_interval_l380_38068


namespace expression_evaluation_l380_38052

theorem expression_evaluation : 3 - (-1) + 4 - 5 + (-6) - (-7) + 8 - 9 = 3 := by
  sorry

end expression_evaluation_l380_38052


namespace power_calculation_l380_38066

theorem power_calculation : 2^300 + 9^3 / 9^2 - 3^4 = 2^300 - 72 := by
  sorry

end power_calculation_l380_38066


namespace formula_describes_relationship_l380_38097

/-- The formula y = 80 - 10x describes the relationship between x and y for a given set of points -/
theorem formula_describes_relationship : ∀ (x y : ℝ), 
  ((x = 0 ∧ y = 80) ∨ 
   (x = 1 ∧ y = 70) ∨ 
   (x = 2 ∧ y = 60) ∨ 
   (x = 3 ∧ y = 50) ∨ 
   (x = 4 ∧ y = 40)) → 
  y = 80 - 10 * x := by
sorry

end formula_describes_relationship_l380_38097


namespace fruits_per_slice_l380_38032

/-- Represents the number of fruits per dozen -/
def dozenSize : ℕ := 12

/-- Represents the number of dozens of Granny Smith apples -/
def grannySmithDozens : ℕ := 4

/-- Represents the number of dozens of Fuji apples -/
def fujiDozens : ℕ := 2

/-- Represents the number of dozens of Bartlett pears -/
def bartlettDozens : ℕ := 3

/-- Represents the number of Granny Smith apple pies -/
def grannySmithPies : ℕ := 4

/-- Represents the number of slices per Granny Smith apple pie -/
def grannySmithSlices : ℕ := 6

/-- Represents the number of Fuji apple pies -/
def fujiPies : ℕ := 3

/-- Represents the number of slices per Fuji apple pie -/
def fujiSlices : ℕ := 8

/-- Represents the number of pear tarts -/
def pearTarts : ℕ := 2

/-- Represents the number of slices per pear tart -/
def pearSlices : ℕ := 10

/-- Theorem stating the number of fruits per slice for each type of pie/tart -/
theorem fruits_per_slice :
  (grannySmithDozens * dozenSize) / (grannySmithPies * grannySmithSlices) = 2 ∧
  (fujiDozens * dozenSize) / (fujiPies * fujiSlices) = 1 ∧
  (bartlettDozens * dozenSize : ℚ) / (pearTarts * pearSlices) = 1.8 := by
  sorry

end fruits_per_slice_l380_38032


namespace six_people_lineup_permutations_l380_38091

theorem six_people_lineup_permutations : Nat.factorial 6 = 720 := by
  sorry

end six_people_lineup_permutations_l380_38091


namespace find_y_value_l380_38050

theorem find_y_value (x y : ℝ) (h1 : 1.5 * x = 0.75 * y) (h2 : x = 24) : y = 48 := by
  sorry

end find_y_value_l380_38050


namespace solution_set_inequality_l380_38065

theorem solution_set_inequality (x : ℝ) : 
  (((2 * x - 1) / (x + 2)) > 1) ↔ (x < -2 ∨ x > 3) := by sorry

end solution_set_inequality_l380_38065


namespace twice_x_minus_one_negative_l380_38063

theorem twice_x_minus_one_negative (x : ℝ) : (2 * x - 1 < 0) ↔ (∃ y, y = 2 * x - 1 ∧ y < 0) := by
  sorry

end twice_x_minus_one_negative_l380_38063


namespace smallest_dual_palindrome_l380_38039

/-- Check if a number is a palindrome in a given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Convert a number from one base to another -/
def convertBase (n : ℕ) (fromBase toBase : ℕ) : ℕ := sorry

theorem smallest_dual_palindrome : 
  ∀ k : ℕ, k > 30 → 
    (isPalindrome k 2 ∧ isPalindrome k 6) → 
    k ≥ 55 ∧ 
    isPalindrome 55 2 ∧ 
    isPalindrome 55 6 := by
  sorry

end smallest_dual_palindrome_l380_38039


namespace lisa_pencils_count_l380_38076

/-- The number of pencils Gloria has initially -/
def gloria_initial : ℕ := 2

/-- The total number of pencils after Lisa gives hers to Gloria -/
def total_pencils : ℕ := 101

/-- The number of pencils Lisa has initially -/
def lisa_initial : ℕ := total_pencils - gloria_initial

theorem lisa_pencils_count : lisa_initial = 99 := by
  sorry

end lisa_pencils_count_l380_38076


namespace edward_scored_seven_l380_38098

/-- Given the total points scored and the friend's score, calculate Edward's score. -/
def edward_score (total : ℕ) (friend_score : ℕ) : ℕ :=
  total - friend_score

/-- Theorem: Edward's score is 7 points when the total is 13 and his friend scored 6. -/
theorem edward_scored_seven :
  edward_score 13 6 = 7 := by
  sorry

end edward_scored_seven_l380_38098


namespace prob_exactly_two_of_three_l380_38072

/-- The probability of exactly two out of three independent events occurring, 
    given their individual probabilities -/
theorem prob_exactly_two_of_three (p_A p_B p_C : ℝ) 
  (h_A : p_A = 1/5) (h_B : p_B = 1/4) (h_C : p_C = 1/3) :
  p_A * p_B * (1 - p_C) + p_A * (1 - p_B) * p_C + (1 - p_A) * p_B * p_C = 3/20 := by
  sorry

end prob_exactly_two_of_three_l380_38072


namespace correct_operation_is_multiplication_by_two_l380_38035

theorem correct_operation_is_multiplication_by_two (N : ℝ) (x : ℝ) :
  (N / 10 = (5 / 100) * (N * x)) → x = 2 := by
  sorry

end correct_operation_is_multiplication_by_two_l380_38035


namespace gcd_g_x_is_20_l380_38083

def g (x : ℤ) : ℤ := (3*x + 4)*(8*x + 5)*(15*x + 11)*(x + 17)

theorem gcd_g_x_is_20 (x : ℤ) (h : 34560 ∣ x) : 
  Nat.gcd (Int.natAbs (g x)) (Int.natAbs x) = 20 := by
sorry

end gcd_g_x_is_20_l380_38083


namespace smallest_prime_factor_of_3087_l380_38088

theorem smallest_prime_factor_of_3087 : Nat.minFac 3087 = 3 := by
  sorry

end smallest_prime_factor_of_3087_l380_38088


namespace book_sale_revenue_l380_38011

theorem book_sale_revenue (total_books : ℕ) (sold_fraction : ℚ) (price_per_book : ℚ) (remaining_books : ℕ) : 
  sold_fraction = 2/3 →
  price_per_book = 2 →
  remaining_books = 36 →
  (1 - sold_fraction) * total_books = remaining_books →
  sold_fraction * total_books * price_per_book = 144 :=
by
  sorry

end book_sale_revenue_l380_38011


namespace coefficient_x3y7_value_l380_38073

/-- The coefficient of x^3 * y^7 in the expansion of (x + 1/x - y)^10 -/
def coefficient_x3y7 : ℤ :=
  let n : ℕ := 10
  let k : ℕ := 7
  let m : ℕ := 3
  (-1)^k * (n.choose k) * (m.choose 0)

theorem coefficient_x3y7_value : coefficient_x3y7 = -120 := by
  sorry

end coefficient_x3y7_value_l380_38073


namespace roberts_spending_l380_38002

theorem roberts_spending (total : ℝ) : 
  total = 100 + 125 + 0.1 * total → total = 250 :=
by sorry

end roberts_spending_l380_38002


namespace train_length_proof_l380_38054

/-- Proves that given a train moving at 55 km/hr and a man moving at 7 km/hr in the opposite direction,
    if it takes 10.45077684107852 seconds for the train to pass the man, then the length of the train is 180 meters. -/
theorem train_length_proof (train_speed : ℝ) (man_speed : ℝ) (passing_time : ℝ) :
  train_speed = 55 →
  man_speed = 7 →
  passing_time = 10.45077684107852 →
  (train_speed + man_speed) * (5 / 18) * passing_time = 180 := by
sorry

end train_length_proof_l380_38054


namespace license_plate_theorem_l380_38017

def letter_count : Nat := 26
def digit_count : Nat := 10
def letter_positions : Nat := 5
def digit_positions : Nat := 3

def license_plate_combinations : Nat :=
  letter_count * (Nat.choose (letter_count - 1) (letter_positions - 2)) *
  (Nat.choose letter_positions 2) * (Nat.factorial (letter_positions - 2)) *
  digit_count * (digit_count - 1) * (digit_count - 2)

theorem license_plate_theorem :
  license_plate_combinations = 2594880000 := by sorry

end license_plate_theorem_l380_38017


namespace candies_remaining_l380_38026

def vasya_eat (n : ℕ) : ℕ := n - (1 + (n - 9) / 7)

def petya_eat (n : ℕ) : ℕ := n - (1 + (n - 7) / 9)

theorem candies_remaining (initial_candies : ℕ) : 
  initial_candies = 1000 → petya_eat (vasya_eat initial_candies) = 761 := by
  sorry

end candies_remaining_l380_38026


namespace u_closed_form_l380_38016

def u : ℕ → ℤ
  | 0 => 1
  | 1 => 4
  | (n + 2) => 5 * u (n + 1) - 6 * u n

theorem u_closed_form (n : ℕ) : u n = 2 * 3^n - 2^n := by
  sorry

end u_closed_form_l380_38016


namespace care_package_weight_l380_38046

theorem care_package_weight (initial_weight : ℝ) (brownies_factor : ℝ) (additional_jelly_beans : ℝ) (gummy_worms_factor : ℝ) :
  initial_weight = 2 ∧
  brownies_factor = 3 ∧
  additional_jelly_beans = 2 ∧
  gummy_worms_factor = 2 →
  (((initial_weight * brownies_factor + additional_jelly_beans) * gummy_worms_factor) : ℝ) = 16 := by
  sorry

end care_package_weight_l380_38046


namespace marble_difference_l380_38077

theorem marble_difference : ∀ (total_marbles : ℕ),
  -- Conditions
  (total_marbles > 0) →  -- Ensure there are marbles
  (∃ (blue1 green1 blue2 green2 : ℕ),
    -- Jar 1 ratio
    7 * green1 = 3 * blue1 ∧
    -- Jar 2 ratio
    5 * green2 = 4 * blue2 ∧
    -- Same total in each jar
    blue1 + green1 = blue2 + green2 ∧
    -- Total green marbles
    green1 + green2 = 140 ∧
    -- Total marbles in each jar
    blue1 + green1 = total_marbles) →
  -- Conclusion
  ∃ (blue1 blue2 : ℕ), blue1 - blue2 = 27 := by
  sorry

end marble_difference_l380_38077


namespace seven_points_triangle_l380_38094

/-- A point in a 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- The angle between three points --/
def angle (p1 p2 p3 : Point) : ℝ := sorry

/-- A set of seven points on a plane --/
def SevenPoints : Type := Fin 7 → Point

theorem seven_points_triangle (points : SevenPoints) :
  ∃ i j k : Fin 7, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    (angle (points i) (points j) (points k) > 2 * π / 3 ∨
     angle (points j) (points k) (points i) > 2 * π / 3 ∨
     angle (points k) (points i) (points j) > 2 * π / 3) :=
  sorry

end seven_points_triangle_l380_38094


namespace cubic_root_product_l380_38022

theorem cubic_root_product : 
  let f : ℝ → ℝ := λ x => 3 * x^3 - 9 * x^2 + 5 * x - 10
  ∀ a b c : ℝ, f a = 0 → f b = 0 → f c = 0 → a * b * c = 10 / 3 :=
by
  sorry

end cubic_root_product_l380_38022


namespace sock_selection_theorem_l380_38075

theorem sock_selection_theorem : 
  (Finset.univ.filter (fun x : Finset (Fin 8) => x.card = 4)).card = 70 := by
  sorry

end sock_selection_theorem_l380_38075


namespace volleyball_lineup_combinations_l380_38092

theorem volleyball_lineup_combinations (total_players : ℕ) 
  (starting_lineup_size : ℕ) (required_players : ℕ) : 
  total_players = 15 → 
  starting_lineup_size = 7 → 
  required_players = 3 → 
  Nat.choose (total_players - required_players) (starting_lineup_size - required_players) = 495 := by
  sorry

end volleyball_lineup_combinations_l380_38092


namespace triangle_medians_and_area_sum_l380_38084

theorem triangle_medians_and_area_sum (a b c : ℝ) (h1 : a = 8) (h2 : b = 15) (h3 : c = 17) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let medians_sum := 3 / 4 * (a^2 + b^2 + c^2)
  medians_sum + area^2 = 4033.5 := by
sorry

end triangle_medians_and_area_sum_l380_38084


namespace min_abs_z_l380_38024

theorem min_abs_z (z : ℂ) (h : Complex.abs (z - 2) + Complex.abs (z - 7*I) = 10) :
  Complex.abs z ≥ 1.4 := by
  sorry

end min_abs_z_l380_38024


namespace dividend_calculation_l380_38049

theorem dividend_calculation (quotient divisor remainder : ℕ) : 
  quotient = 15000 → 
  divisor = 82675 → 
  remainder = 57801 → 
  quotient * divisor + remainder = 1240182801 := by
sorry

end dividend_calculation_l380_38049


namespace sum_of_19th_powers_zero_l380_38087

theorem sum_of_19th_powers_zero (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (sum_cubes_zero : a^3 + b^3 + c^3 = 0) : 
  a^19 + b^19 + c^19 = 0 := by
sorry

end sum_of_19th_powers_zero_l380_38087


namespace profit_percentage_l380_38041

theorem profit_percentage (C S : ℝ) (h : 19 * C = 16 * S) : 
  (S - C) / C * 100 = 18.75 := by
  sorry

end profit_percentage_l380_38041


namespace complex_squared_i_positive_l380_38029

theorem complex_squared_i_positive (a : ℝ) :
  (∃ (x : ℝ), x > 0 ∧ (Complex.I * (a + Complex.I)^2 = x)) → a = -1 := by
  sorry

end complex_squared_i_positive_l380_38029


namespace brown_rabbit_hop_distance_l380_38095

/-- Proves that given a white rabbit hopping 15 meters per minute and a total distance of 135 meters
    hopped by both rabbits in 5 minutes, the brown rabbit hops 12 meters per minute. -/
theorem brown_rabbit_hop_distance
  (white_rabbit_speed : ℝ)
  (total_distance : ℝ)
  (time : ℝ)
  (h1 : white_rabbit_speed = 15)
  (h2 : total_distance = 135)
  (h3 : time = 5) :
  (total_distance - white_rabbit_speed * time) / time = 12 := by
  sorry

#check brown_rabbit_hop_distance

end brown_rabbit_hop_distance_l380_38095


namespace partial_fraction_decomposition_sum_l380_38028

-- Define the polynomial
def p (x : ℝ) : ℝ := x^3 - 3*x^2 - 4*x + 12

-- State the theorem
theorem partial_fraction_decomposition_sum (a b c D E F : ℝ) : 
  -- a, b, c are distinct roots of p
  p a = 0 → p b = 0 → p c = 0 → a ≠ b → b ≠ c → a ≠ c →
  -- Partial fraction decomposition holds
  (∀ s : ℝ, s ≠ a → s ≠ b → s ≠ c → 
    1 / (s^3 - 3*s^2 - 4*s + 12) = D / (s - a) + E / (s - b) + F / (s - c)) →
  -- Conclusion
  1 / D + 1 / E + 1 / F + a * b * c = 4 :=
by
  sorry

end partial_fraction_decomposition_sum_l380_38028


namespace geometric_sequence_general_term_l380_38015

theorem geometric_sequence_general_term 
  (a : ℕ → ℝ) -- a is the sequence
  (S : ℕ → ℝ) -- S is the sum function
  (h1 : ∀ n, S n = 3^n - 1) -- Given condition
  (h2 : ∀ n, S n = S (n-1) + a n) -- Property of sum of sequences
  : ∀ n, a n = 2 * 3^(n-1) := by sorry

end geometric_sequence_general_term_l380_38015


namespace exponential_inequality_l380_38040

theorem exponential_inequality (a b c : ℝ) : 
  a^b > a^c ∧ a^c > 1 ∧ b < c → b < c ∧ c < 0 ∧ 0 < a ∧ a < 1 := by
  sorry

end exponential_inequality_l380_38040


namespace function_monotonicity_l380_38053

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def periodic_two (f : ℝ → ℝ) : Prop := ∀ x, f x = f (2 - x)

def decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem function_monotonicity (f : ℝ → ℝ) 
  (h1 : is_even f)
  (h2 : periodic_two f)
  (h3 : decreasing_on f 1 2) :
  increasing_on f (-2) (-1) ∧ decreasing_on f 3 4 := by
  sorry

end function_monotonicity_l380_38053


namespace derivative_value_implies_coefficient_l380_38019

theorem derivative_value_implies_coefficient (f' : ℝ → ℝ) (a : ℝ) :
  (∀ x, f' x = 2 * x^3 + a * x^2 + x) →
  f' 1 = 9 →
  a = 6 := by
sorry

end derivative_value_implies_coefficient_l380_38019


namespace train_bridge_crossing_time_l380_38056

/-- The time taken for a train to cross a bridge -/
theorem train_bridge_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (bridge_length : ℝ)
  (h1 : train_length = 110)
  (h2 : train_speed_kmh = 36)
  (h3 : bridge_length = 132) :
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 24.2 := by
  sorry

end train_bridge_crossing_time_l380_38056
