import Mathlib

namespace linear_equation_solution_l2974_297492

theorem linear_equation_solution (a b : ℝ) :
  (3 : ℝ) * a + (-2 : ℝ) * b = -1 → 3 * a - 2 * b + 2024 = 2023 := by
  sorry

end linear_equation_solution_l2974_297492


namespace cement_mixture_weight_l2974_297499

/-- Given a cement mixture composed of sand, water, and gravel, where:
    - 1/4 of the mixture is sand (by weight)
    - 2/5 of the mixture is water (by weight)
    - 14 pounds of the mixture is gravel
    Prove that the total weight of the mixture is 40 pounds. -/
theorem cement_mixture_weight :
  ∀ (total_weight : ℝ),
  (1/4 : ℝ) * total_weight +     -- Weight of sand
  (2/5 : ℝ) * total_weight +     -- Weight of water
  14 = total_weight →            -- Weight of gravel
  total_weight = 40 :=
by
  sorry


end cement_mixture_weight_l2974_297499


namespace track_length_is_600_l2974_297436

/-- Represents a circular running track with two runners -/
structure CircularTrack where
  length : ℝ
  brenda_speed : ℝ
  sally_speed : ℝ

/-- The conditions of the problem -/
def problem_conditions (track : CircularTrack) : Prop :=
  ∃ (first_meeting_time second_meeting_time : ℝ),
    -- Brenda runs 120 meters before first meeting
    track.brenda_speed * first_meeting_time = 120 ∧
    -- Sally runs (length/2 - 120) meters before first meeting
    track.sally_speed * first_meeting_time = track.length / 2 - 120 ∧
    -- Sally runs an additional 180 meters between meetings
    track.sally_speed * (second_meeting_time - first_meeting_time) = 180 ∧
    -- Brenda's position at second meeting
    track.brenda_speed * second_meeting_time =
      track.length - (track.length / 2 - 120 + 180)

/-- The theorem to be proven -/
theorem track_length_is_600 (track : CircularTrack) :
  problem_conditions track → track.length = 600 := by
  sorry


end track_length_is_600_l2974_297436


namespace inequality_proof_l2974_297469

theorem inequality_proof (x y z : ℝ) 
  (h1 : x ≥ 1) (h2 : y ≥ 1) (h3 : z ≥ 1)
  (h4 : 1 / (x^2 - 1) + 1 / (y^2 - 1) + 1 / (z^2 - 1) = 1) : 
  1 / (x + 1) + 1 / (y + 1) + 1 / (z + 1) ≤ 1 := by
sorry

end inequality_proof_l2974_297469


namespace farmer_water_capacity_l2974_297421

/-- Calculates the total water capacity for a farmer's trucks -/
def total_water_capacity (num_trucks : ℕ) (tanks_per_truck : ℕ) (tank_capacity : ℕ) : ℕ :=
  num_trucks * tanks_per_truck * tank_capacity

/-- Theorem: The farmer can carry 1350 liters of water -/
theorem farmer_water_capacity :
  total_water_capacity 3 3 150 = 1350 := by
  sorry

end farmer_water_capacity_l2974_297421


namespace point_quadrant_relation_l2974_297428

/-- If point M(1+a, 2b-1) is in the third quadrant, then point N(a-1, 1-2b) is in the second quadrant. -/
theorem point_quadrant_relation (a b : ℝ) : 
  (1 + a < 0 ∧ 2*b - 1 < 0) → (a - 1 < 0 ∧ 1 - 2*b > 0) :=
by sorry

end point_quadrant_relation_l2974_297428


namespace equation_solutions_l2974_297407

theorem equation_solutions : ∀ x : ℝ,
  (x^2 - 3*x = 4 ↔ x = 4 ∨ x = -1) ∧
  (x*(x-2) + x - 2 = 0 ↔ x = 2 ∨ x = -1) := by
  sorry

end equation_solutions_l2974_297407


namespace complementary_angles_difference_l2974_297412

theorem complementary_angles_difference (x y : ℝ) : 
  x + y = 90 →  -- complementary angles sum to 90°
  x / y = 3 / 5 →  -- ratio of angles is 3:5
  |x - y| = 22.5 :=  -- positive difference is 22.5°
by sorry

end complementary_angles_difference_l2974_297412


namespace complement_P_equals_two_l2974_297491

def U : Set Int := {-1, 0, 1, 2}

def P : Set Int := {x : Int | x^2 < 2}

theorem complement_P_equals_two : 
  {x ∈ U | x ∉ P} = {2} := by sorry

end complement_P_equals_two_l2974_297491


namespace min_containers_for_85_units_l2974_297418

/-- Represents the possible container sizes for snacks -/
inductive ContainerSize
  | small : ContainerSize  -- 5 units
  | medium : ContainerSize -- 10 units
  | large : ContainerSize  -- 20 units

/-- Returns the number of units in a given container size -/
def containerUnits (size : ContainerSize) : Nat :=
  match size with
  | .small => 5
  | .medium => 10
  | .large => 20

/-- Represents a combination of containers -/
structure ContainerCombination where
  small : Nat
  medium : Nat
  large : Nat

/-- Calculates the total number of units in a combination of containers -/
def totalUnits (combo : ContainerCombination) : Nat :=
  combo.small * containerUnits ContainerSize.small +
  combo.medium * containerUnits ContainerSize.medium +
  combo.large * containerUnits ContainerSize.large

/-- Calculates the total number of containers in a combination -/
def totalContainers (combo : ContainerCombination) : Nat :=
  combo.small + combo.medium + combo.large

/-- Theorem: The minimum number of containers to get exactly 85 units is 5 -/
theorem min_containers_for_85_units :
  ∃ (combo : ContainerCombination),
    totalUnits combo = 85 ∧
    totalContainers combo = 5 ∧
    (∀ (other : ContainerCombination),
      totalUnits other = 85 → totalContainers other ≥ 5) := by
  sorry

end min_containers_for_85_units_l2974_297418


namespace frank_candy_purchase_l2974_297486

/-- The number of candies Frank can buy with his arcade tickets -/
def candies_bought (whack_a_mole_tickets : ℕ) (skee_ball_tickets : ℕ) (candy_cost : ℕ) : ℕ :=
  (whack_a_mole_tickets + skee_ball_tickets) / candy_cost

/-- Theorem: Frank can buy 7 candies with his arcade tickets -/
theorem frank_candy_purchase :
  candies_bought 33 9 6 = 7 := by
  sorry

end frank_candy_purchase_l2974_297486


namespace complex_division_result_l2974_297433

theorem complex_division_result : (5 + Complex.I) / (1 - Complex.I) = 2 + 3 * Complex.I := by
  sorry

end complex_division_result_l2974_297433


namespace jim_investment_is_36000_l2974_297430

/-- Represents the investment of three individuals in a business. -/
structure Investment where
  john : ℕ
  james : ℕ
  jim : ℕ

/-- Calculates Jim's investment given the ratio and total investment. -/
def calculate_jim_investment (ratio : Investment) (total : ℕ) : ℕ :=
  let total_parts := ratio.john + ratio.james + ratio.jim
  let jim_parts := ratio.jim
  (total * jim_parts) / total_parts

/-- Theorem stating that Jim's investment is $36,000 given the conditions. -/
theorem jim_investment_is_36000 :
  let ratio : Investment := ⟨4, 7, 9⟩
  let total_investment : ℕ := 80000
  calculate_jim_investment ratio total_investment = 36000 := by
  sorry

end jim_investment_is_36000_l2974_297430


namespace yuan_david_age_difference_l2974_297439

theorem yuan_david_age_difference : 
  ∀ (yuan_age david_age : ℕ),
    david_age = 7 →
    yuan_age = 2 * david_age →
    yuan_age - david_age = 7 :=
by
  sorry

end yuan_david_age_difference_l2974_297439


namespace fraction_difference_l2974_297431

theorem fraction_difference (m n : ℝ) (h1 : m^2 - n^2 = m*n) (h2 : m*n ≠ 0) :
  n/m - m/n = -1 := by
  sorry

end fraction_difference_l2974_297431


namespace cost_of_paints_l2974_297481

def cost_paintbrush : ℚ := 2.40
def cost_easel : ℚ := 6.50
def rose_has : ℚ := 7.10
def rose_needs : ℚ := 11.00

theorem cost_of_paints :
  let total_cost := rose_has + rose_needs
  let cost_paints := total_cost - (cost_paintbrush + cost_easel)
  cost_paints = 9.20 := by sorry

end cost_of_paints_l2974_297481


namespace discount_percentage_proof_l2974_297494

theorem discount_percentage_proof (jacket_price shirt_price : ℝ) 
  (jacket_discount shirt_discount : ℝ) : 
  jacket_price = 100 →
  shirt_price = 50 →
  jacket_discount = 0.3 →
  shirt_discount = 0.6 →
  (jacket_price * jacket_discount + shirt_price * shirt_discount) / (jacket_price + shirt_price) = 0.4 := by
sorry

end discount_percentage_proof_l2974_297494


namespace max_sections_five_lines_l2974_297497

/-- The maximum number of sections a rectangle can be divided into by n line segments --/
def max_sections (n : ℕ) : ℕ :=
  if n = 0 then 1 else (n * (n + 1)) / 2 + 1

/-- Theorem: The maximum number of sections a rectangle can be divided into by 5 line segments is 16 --/
theorem max_sections_five_lines :
  max_sections 5 = 16 := by
  sorry

end max_sections_five_lines_l2974_297497


namespace G_properties_l2974_297485

/-- The curve G defined by x³ + y³ - 6xy = 0 for x > 0 and y > 0 -/
def G : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 > 0 ∧ p.2 > 0 ∧ p.1^3 + p.2^3 - 6*p.1*p.2 = 0}

/-- The line y = x -/
def line_y_eq_x : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = p.2}

/-- The line x + y - 6 = 0 -/
def line_x_plus_y_eq_6 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 = 6}

theorem G_properties :
  (∀ p : ℝ × ℝ, p ∈ G → (p.2, p.1) ∈ G) ∧ 
  (∃! p : ℝ × ℝ, p ∈ G ∩ line_x_plus_y_eq_6) ∧
  (∀ p : ℝ × ℝ, p ∈ G → Real.sqrt (p.1^2 + p.2^2) ≤ 3 * Real.sqrt 2) ∧
  (∃ p : ℝ × ℝ, p ∈ G ∧ Real.sqrt (p.1^2 + p.2^2) = 3 * Real.sqrt 2) :=
by sorry

end G_properties_l2974_297485


namespace inequality_holds_iff_m_in_interval_l2974_297480

theorem inequality_holds_iff_m_in_interval :
  ∀ m : ℝ, (∀ x : ℝ, -6 < (2 * x^2 + m * x - 4) / (x^2 - x + 1) ∧ 
    (2 * x^2 + m * x - 4) / (x^2 - x + 1) < 4) ↔ 
  -2 < m ∧ m < 4 := by
sorry

end inequality_holds_iff_m_in_interval_l2974_297480


namespace reflection_result_l2974_297454

/-- Reflects a point (x, y) across the line x = k -/
def reflect_point (x y k : ℝ) : ℝ × ℝ := (2 * k - x, y)

/-- Reflects a line y = mx + c across x = k -/
def reflect_line (m c k : ℝ) : ℝ × ℝ := 
  let point := reflect_point k (m * k + c) k
  (-m, 2 * m * k + c - m * point.1)

theorem reflection_result : 
  let original_slope : ℝ := -2
  let original_intercept : ℝ := 7
  let reflection_line : ℝ := 3
  let (a, b) := reflect_line original_slope original_intercept reflection_line
  2 * a + b = -1 := by sorry

end reflection_result_l2974_297454


namespace domain_of_g_l2974_297452

-- Define the original function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def dom_f : Set ℝ := Set.Icc (-1) 4

-- Define the new function g
def g (x : ℝ) : ℝ := f (2 * x - 1)

-- State the theorem
theorem domain_of_g :
  {x : ℝ | g x ∈ Set.range f} = Set.Icc 0 (5/2) := by sorry

end domain_of_g_l2974_297452


namespace simplify_fraction_l2974_297409

theorem simplify_fraction (a : ℝ) (h : a ≠ 0) : (a + 1) / a - 1 / a = 1 := by
  sorry

end simplify_fraction_l2974_297409


namespace seventh_observation_value_l2974_297459

theorem seventh_observation_value 
  (n : ℕ) 
  (initial_avg : ℝ) 
  (decrease : ℝ) 
  (h1 : n = 6) 
  (h2 : initial_avg = 12) 
  (h3 : decrease = 1) : 
  let new_avg := initial_avg - decrease
  let new_obs := (n + 1) * new_avg - n * initial_avg
  new_obs = 5 := by sorry

end seventh_observation_value_l2974_297459


namespace quadratic_rewrite_ratio_l2974_297453

theorem quadratic_rewrite_ratio (j : ℝ) :
  let original := 8 * j^2 - 6 * j + 16
  ∃ (c p q : ℝ), 
    (∀ j, original = c * (j + p)^2 + q) ∧
    q / p = -119 / 3 :=
by sorry

end quadratic_rewrite_ratio_l2974_297453


namespace vitamin_a_daily_serving_l2974_297451

/-- The amount of Vitamin A in each pill (in mg) -/
def vitamin_a_per_pill : ℕ := 50

/-- The number of pills needed for the weekly recommended amount -/
def pills_per_week : ℕ := 28

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The recommended daily serving of Vitamin A (in mg) -/
def recommended_daily_serving : ℕ := (vitamin_a_per_pill * pills_per_week) / days_per_week

theorem vitamin_a_daily_serving :
  recommended_daily_serving = 200 := by
  sorry

end vitamin_a_daily_serving_l2974_297451


namespace power_division_l2974_297449

theorem power_division (m : ℝ) : m^10 / m^5 = m^5 := by
  sorry

end power_division_l2974_297449


namespace flippers_win_probability_l2974_297444

theorem flippers_win_probability :
  let n : ℕ := 6  -- Total number of games
  let k : ℕ := 4  -- Number of games to win
  let p : ℚ := 3/5  -- Probability of winning a single game
  Nat.choose n k * p^k * (1-p)^(n-k) = 4860/15625 := by
sorry

end flippers_win_probability_l2974_297444


namespace max_height_particle_from_wheel_l2974_297404

/-- The maximum height reached by a particle thrown off a rolling wheel -/
theorem max_height_particle_from_wheel
  (r : ℝ) -- radius of the wheel
  (ω : ℝ) -- angular velocity of the wheel
  (g : ℝ) -- acceleration due to gravity
  (h_pos : r > 0) -- radius is positive
  (ω_pos : ω > 0) -- angular velocity is positive
  (g_pos : g > 0) -- gravity is positive
  (h_ω : ω > Real.sqrt (g / r)) -- condition on angular velocity
  : ∃ (h : ℝ), h = (r * ω + g / ω)^2 / (2 * g) ∧
    ∀ (h' : ℝ), h' ≤ h :=
by sorry

end max_height_particle_from_wheel_l2974_297404


namespace sqrt_equation_solution_l2974_297476

theorem sqrt_equation_solution :
  ∃! z : ℝ, Real.sqrt (10 + 3 * z) = 8 := by
  sorry

end sqrt_equation_solution_l2974_297476


namespace four_square_games_l2974_297406

/-- The number of players in the four-square league -/
def total_players : ℕ := 12

/-- The number of players in each game -/
def players_per_game : ℕ := 6

/-- The number of times two specific players play together -/
def games_together : ℕ := 210

/-- The total number of possible game combinations -/
def total_combinations : ℕ := Nat.choose total_players players_per_game

theorem four_square_games (player1 player2 : Fin total_players) 
  (h_distinct : player1 ≠ player2) :
  (Nat.choose (total_players - 2) (players_per_game - 2) = games_together) ∧
  (total_combinations = Nat.choose total_players players_per_game) ∧
  (2 * games_together = players_per_game * (total_combinations / total_players)) :=
sorry

end four_square_games_l2974_297406


namespace smallest_dual_base_palindrome_l2974_297479

/-- Checks if a number is a palindrome in a given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a number from base 10 to another base -/
def toBase (n : ℕ) (base : ℕ) : List ℕ := sorry

theorem smallest_dual_base_palindrome :
  ∀ k : ℕ,
    k > 8 ∧ 
    isPalindrome k 3 ∧ 
    isPalindrome k 5 →
    k ≥ 26 :=
by sorry

end smallest_dual_base_palindrome_l2974_297479


namespace smallest_common_factor_thirty_three_satisfies_smallest_n_is_33_l2974_297484

theorem smallest_common_factor (n : ℕ) : n > 0 ∧ ∃ (k : ℕ), k > 1 ∧ k ∣ (8*n - 3) ∧ k ∣ (6*n + 5) → n ≥ 33 :=
sorry

theorem thirty_three_satisfies : ∃ (k : ℕ), k > 1 ∧ k ∣ (8*33 - 3) ∧ k ∣ (6*33 + 5) :=
sorry

theorem smallest_n_is_33 : (∃ (n : ℕ), n > 0 ∧ ∃ (k : ℕ), k > 1 ∧ k ∣ (8*n - 3) ∧ k ∣ (6*n + 5)) ∧
  (∀ (m : ℕ), m > 0 ∧ ∃ (k : ℕ), k > 1 ∧ k ∣ (8*m - 3) ∧ k ∣ (6*m + 5) → m ≥ 33) :=
sorry

end smallest_common_factor_thirty_three_satisfies_smallest_n_is_33_l2974_297484


namespace fractional_part_equality_l2974_297422

/-- Given k = 2 + √3, prove that k^n - ⌊k^n⌋ = 1 - 1/k^n for any natural number n. -/
theorem fractional_part_equality (n : ℕ) : 
  let k : ℝ := 2 + Real.sqrt 3
  (k^n : ℝ) - ⌊k^n⌋ = 1 - 1 / (k^n) := by
  sorry

end fractional_part_equality_l2974_297422


namespace digits_of_2_pow_100_l2974_297410

theorem digits_of_2_pow_100 (h : ∃ n : ℕ, 10^(n-1) ≤ 2^200 ∧ 2^200 < 10^n ∧ n = 61) :
  ∃ m : ℕ, 10^(m-1) ≤ 2^100 ∧ 2^100 < 10^m ∧ m = 31 :=
by sorry

end digits_of_2_pow_100_l2974_297410


namespace min_value_a_plus_2b_min_value_a_plus_2b_exact_min_value_a_plus_2b_equality_l2974_297473

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / (a + 2) + 1 / (b + 2) = 1 / 3) : 
  ∀ x y, x > 0 → y > 0 → 1 / (x + 2) + 1 / (y + 2) = 1 / 3 → a + 2 * b ≤ x + 2 * y :=
by sorry

theorem min_value_a_plus_2b_exact (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / (a + 2) + 1 / (b + 2) = 1 / 3) : 
  a + 2 * b ≥ 3 + 6 * Real.sqrt 2 :=
by sorry

theorem min_value_a_plus_2b_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / (a + 2) + 1 / (b + 2) = 1 / 3) : 
  (a + 2 * b = 3 + 6 * Real.sqrt 2) ↔ 
  (a = 1 + 3 * Real.sqrt 2 ∧ b = 1 + 3 * Real.sqrt 2 / 2) :=
by sorry

end min_value_a_plus_2b_min_value_a_plus_2b_exact_min_value_a_plus_2b_equality_l2974_297473


namespace total_profit_is_29_20_l2974_297440

/-- Represents the profit calculation for candied fruits --/
def candied_fruit_profit (num_apples num_grapes num_oranges : ℕ)
  (apple_price apple_cost grape_price grape_cost orange_price orange_cost : ℚ) : ℚ :=
  let apple_profit := num_apples * (apple_price - apple_cost)
  let grape_profit := num_grapes * (grape_price - grape_cost)
  let orange_profit := num_oranges * (orange_price - orange_cost)
  apple_profit + grape_profit + orange_profit

/-- Theorem stating that the total profit is $29.20 given the problem conditions --/
theorem total_profit_is_29_20 :
  candied_fruit_profit 15 12 10 2 1.2 1.5 0.9 2.5 1.5 = 29.2 := by
  sorry

end total_profit_is_29_20_l2974_297440


namespace watch_cost_price_l2974_297493

theorem watch_cost_price (CP : ℝ) : 
  (0.90 * CP = CP - 0.10 * CP) →
  (1.05 * CP = CP + 0.05 * CP) →
  (1.05 * CP - 0.90 * CP = 180) →
  CP = 1200 :=
by sorry

end watch_cost_price_l2974_297493


namespace binomial_seven_four_l2974_297438

theorem binomial_seven_four : Nat.choose 7 4 = 35 := by
  sorry

end binomial_seven_four_l2974_297438


namespace rescue_net_sag_l2974_297432

/-- The sag of an elastic rescue net for two different jumpers -/
theorem rescue_net_sag 
  (m₁ m₂ x₁ h₁ h₂ : ℝ) 
  (hm₁ : m₁ = 78.75)
  (hm₂ : m₂ = 45)
  (hx₁ : x₁ = 1)
  (hh₁ : h₁ = 15)
  (hh₂ : h₂ = 29)
  (x₂ : ℝ) :
  28 * x₂^2 - x₂ - 29 = 0 ↔ 
  m₂ * (h₂ + x₂) / (m₁ * (h₁ + x₁)) = x₂^2 / x₁^2 := by
sorry


end rescue_net_sag_l2974_297432


namespace sunday_occurs_five_times_in_january_l2974_297411

/-- Represents the days of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents a month -/
structure Month where
  days : ℕ
  first_day : DayOfWeek

/-- December of year M -/
def december : Month := {
  days := 31,
  first_day := DayOfWeek.Thursday  -- This is arbitrary, as we don't know the exact first day
}

/-- January of year M+1 -/
def january : Month := {
  days := 31,
  first_day := sorry  -- We don't know the exact first day, it depends on December
}

/-- Count occurrences of a specific day in a month -/
def count_day_occurrences (m : Month) (d : DayOfWeek) : ℕ := sorry

/-- The main theorem to prove -/
theorem sunday_occurs_five_times_in_january :
  (count_day_occurrences december DayOfWeek.Thursday = 5) →
  (count_day_occurrences january DayOfWeek.Sunday = 5) :=
sorry

end sunday_occurs_five_times_in_january_l2974_297411


namespace smallest_palindromic_n_string_l2974_297498

/-- An n-string is a string of digits formed by writing the numbers 1, 2, ..., n in some order -/
def nString (n : ℕ) := List ℕ

/-- A palindromic string reads the same forwards and backwards -/
def isPalindromic (s : List ℕ) : Prop :=
  s = s.reverse

/-- The smallest n > 1 such that there exists a palindromic n-string -/
def smallestPalindromicN : ℕ := 19

theorem smallest_palindromic_n_string : 
  (∀ k : ℕ, 1 < k → k < smallestPalindromicN → 
    ¬∃ s : nString k, isPalindromic s) ∧
  (∃ s : nString smallestPalindromicN, isPalindromic s) := by
  sorry

end smallest_palindromic_n_string_l2974_297498


namespace odd_function_extension_l2974_297408

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_extension 
  (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_neg : ∀ x < 0, f x = x^2 - 3*x - 1) : 
  ∀ x > 0, f x = -x^2 - 3*x + 1 :=
by sorry

end odd_function_extension_l2974_297408


namespace greatest_possible_median_l2974_297419

theorem greatest_possible_median (k m r s t : ℕ) : 
  k > 0 → m > 0 → r > 0 → s > 0 → t > 0 →
  (k + m + r + s + t) / 5 = 16 →
  k < m → m < r → r < s → s < t →
  t = 42 →
  r ≤ 32 ∧ ∃ (k' m' r' s' : ℕ), 
    k' > 0 ∧ m' > 0 ∧ r' > 0 ∧ s' > 0 ∧
    (k' + m' + r' + s' + 42) / 5 = 16 ∧
    k' < m' ∧ m' < r' ∧ r' < s' ∧ s' < 42 ∧
    r' = 32 :=
by sorry

end greatest_possible_median_l2974_297419


namespace sum_f_at_one_equals_exp_e_l2974_297442

noncomputable def f : ℕ → (ℝ → ℝ)
| 0 => fun x => Real.exp x
| (n + 1) => fun x => x * (deriv (f n)) x

theorem sum_f_at_one_equals_exp_e :
  (∑' n, (f n 1) / n.factorial) = Real.exp (Real.exp 1) := by sorry

end sum_f_at_one_equals_exp_e_l2974_297442


namespace alice_bob_meet_after_5_turns_l2974_297478

/-- Represents the number of points on the circle -/
def num_points : ℕ := 15

/-- Represents Alice's clockwise movement per turn -/
def alice_move : ℕ := 4

/-- Represents Bob's counterclockwise movement per turn -/
def bob_move : ℕ := 8

/-- Calculates the position after a given number of moves -/
def position_after_moves (start : ℕ) (move : ℕ) (turns : ℕ) : ℕ :=
  (start + move * turns) % num_points

/-- Theorem stating that Alice and Bob meet after 5 turns -/
theorem alice_bob_meet_after_5_turns :
  ∃ (meeting_point : ℕ),
    position_after_moves num_points alice_move 5 = meeting_point ∧
    position_after_moves num_points (num_points - bob_move) 5 = meeting_point :=
by sorry

end alice_bob_meet_after_5_turns_l2974_297478


namespace quadratic_inequality_implication_l2974_297471

theorem quadratic_inequality_implication (y : ℝ) :
  y^2 - 7*y + 12 < 0 → 42 < y^2 + 7*y + 12 ∧ y^2 + 7*y + 12 < 56 := by
  sorry

end quadratic_inequality_implication_l2974_297471


namespace distinctly_marked_fraction_l2974_297457

/-- Proves that the fraction of a 15 by 24 rectangular region that is distinctly marked is 1/6,
    given that one-third of the rectangle is shaded and half of the shaded area is distinctly marked. -/
theorem distinctly_marked_fraction (length width : ℕ) (shaded_fraction marked_fraction : ℚ) :
  length = 15 →
  width = 24 →
  shaded_fraction = 1/3 →
  marked_fraction = 1/2 →
  (shaded_fraction * marked_fraction : ℚ) = 1/6 :=
by sorry

end distinctly_marked_fraction_l2974_297457


namespace goldbach_2024_l2974_297472

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem goldbach_2024 :
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 2024 :=
sorry

end goldbach_2024_l2974_297472


namespace larger_solution_quadratic_l2974_297464

theorem larger_solution_quadratic (x : ℝ) : 
  x^2 - 7*x - 18 = 0 → x ≤ 9 ∧ (∃ y : ℝ, y ≠ x ∧ y^2 - 7*y - 18 = 0) := by
  sorry

end larger_solution_quadratic_l2974_297464


namespace interview_probability_implies_total_workers_l2974_297468

/-- The number of workers excluding Jack and Jill -/
def other_workers : ℕ := 6

/-- The probability of selecting both Jack and Jill for the interview -/
def probability : ℚ := 1 / 28

/-- The number of workers to be selected for the interview -/
def selected_workers : ℕ := 2

/-- The total number of workers -/
def total_workers : ℕ := other_workers + 2

theorem interview_probability_implies_total_workers :
  (probability = (1 : ℚ) / (total_workers.choose selected_workers)) →
  total_workers = 8 := by
  sorry

end interview_probability_implies_total_workers_l2974_297468


namespace quadratic_inequality_problem_l2974_297467

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) := a * x^2 + 5 * x - 2

-- Define the solution set condition
def solution_set_condition (a : ℝ) : Prop :=
  ∀ x, f a x > 0 ↔ (1/2 < x ∧ x < 2)

-- Theorem statement
theorem quadratic_inequality_problem (a : ℝ) (h : solution_set_condition a) :
  a = -2 ∧ 
  (∀ x, a * x^2 + 5 * x + a^2 - 1 > 0 ↔ (-1/2 < x ∧ x < 3)) :=
sorry

end quadratic_inequality_problem_l2974_297467


namespace remaining_money_l2974_297463

def savings : ℕ := 5555 -- in base 8
def ticket_cost : ℕ := 1200 -- in base 10

def base_8_to_10 (n : ℕ) : ℕ :=
  (n / 1000) * 512 + ((n / 100) % 10) * 64 + ((n / 10) % 10) * 8 + (n % 10)

theorem remaining_money :
  base_8_to_10 savings - ticket_cost = 1725 := by sorry

end remaining_money_l2974_297463


namespace negation_equivalence_l2974_297441

theorem negation_equivalence :
  (¬ ∀ a : ℝ, a ∈ Set.Icc 0 1 → a^4 + a^2 > 1) ↔
  (∃ a : ℝ, a ∈ Set.Icc 0 1 ∧ a^4 + a^2 ≤ 1) :=
by sorry

end negation_equivalence_l2974_297441


namespace circle_segment_ratio_l2974_297434

theorem circle_segment_ratio : 
  ∀ (r : ℝ) (S₁ S₂ : ℝ), 
  r > 0 → 
  S₁ = (1 / 12) * r^2 * (4 * π - 3 * Real.sqrt 3) →
  S₂ = (1 / 12) * r^2 * (8 * π + 3 * Real.sqrt 3) →
  S₁ / S₂ = (4 * π - 3 * Real.sqrt 3) / (8 * π + 3 * Real.sqrt 3) := by
sorry


end circle_segment_ratio_l2974_297434


namespace square_area_on_line_and_parabola_l2974_297466

/-- A square with one side on y = x + 4 and two vertices on y² = x has area 18 or 50 -/
theorem square_area_on_line_and_parabola :
  ∀ (A B C D : ℝ × ℝ),
    (∃ (y₁ y₂ : ℝ),
      A.2 = A.1 + 4 ∧
      B.2 = B.1 + 4 ∧
      C = (y₁^2, y₁) ∧
      D = (y₂^2, y₂) ∧
      (B.1 - A.1)^2 + (B.2 - A.2)^2 = (C.1 - B.1)^2 + (C.2 - B.2)^2 ∧
      (C.1 - B.1)^2 + (C.2 - B.2)^2 = (D.1 - C.1)^2 + (D.2 - C.2)^2 ∧
      (D.1 - C.1)^2 + (D.2 - C.2)^2 = (A.1 - D.1)^2 + (A.2 - D.2)^2) →
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 18) ∨ ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 50) :=
by sorry


end square_area_on_line_and_parabola_l2974_297466


namespace sum_of_squared_differences_zero_l2974_297488

theorem sum_of_squared_differences_zero (x y z : ℝ) :
  (x - 3)^2 + (y - 4)^2 + (z - 5)^2 = 0 → x + y + z = 12 := by
  sorry

end sum_of_squared_differences_zero_l2974_297488


namespace two_thousand_five_is_334th_term_l2974_297445

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem two_thousand_five_is_334th_term :
  arithmetic_sequence 7 6 334 = 2005 :=
by sorry

end two_thousand_five_is_334th_term_l2974_297445


namespace ali_seashells_l2974_297450

/-- Proves that Ali started with 180 seashells given the conditions of the problem -/
theorem ali_seashells : 
  ∀ S : ℕ, 
  (S - 40 - 30) / 2 = 55 → 
  S = 180 := by
sorry

end ali_seashells_l2974_297450


namespace boys_to_girls_ratio_l2974_297455

/-- Given a college with 416 total students and 160 girls, the ratio of boys to girls is 8:5 -/
theorem boys_to_girls_ratio (total_students : ℕ) (girls : ℕ) : 
  total_students = 416 → girls = 160 → 
  (total_students - girls) / girls = 8 / 5 := by
  sorry

end boys_to_girls_ratio_l2974_297455


namespace inequality_proof_l2974_297423

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x * y * z + x * y + y * z + z * x = x + y + z + 1) :
  (1 / 3 : ℝ) * (Real.sqrt ((1 + x^2) / (1 + x)) + Real.sqrt ((1 + y^2) / (1 + y)) + Real.sqrt ((1 + z^2) / (1 + z)))
  ≤ ((x + y + z) / 3) ^ (5 / 8) := by
  sorry

end inequality_proof_l2974_297423


namespace multiplication_exponent_rule_l2974_297400

theorem multiplication_exponent_rule (a : ℝ) (h : a ≠ 0) : a * a^2 = a^3 := by
  sorry

end multiplication_exponent_rule_l2974_297400


namespace cos_alpha_on_unit_circle_l2974_297426

theorem cos_alpha_on_unit_circle (α : Real) :
  let P : ℝ × ℝ := (-Real.sqrt 3 / 2, -1 / 2)
  (P.1^2 + P.2^2 = 1) →  -- Point P is on the unit circle
  (∃ t : ℝ, t > 0 ∧ t * (Real.cos α) = P.1 ∧ t * (Real.sin α) = P.2) →  -- P is on the terminal side of α
  Real.cos α = -Real.sqrt 3 / 2 := by
  sorry

end cos_alpha_on_unit_circle_l2974_297426


namespace spade_sum_equals_six_l2974_297465

-- Define the ♠ operation
def spade (a b : ℝ) : ℝ := |a - b|

-- Theorem statement
theorem spade_sum_equals_six : 
  (spade 2 3) + (spade 5 10) = 6 := by
  sorry

end spade_sum_equals_six_l2974_297465


namespace matrix_equation_l2974_297420

def A : Matrix (Fin 2) (Fin 2) ℤ := !![3, -5; 2, -3]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![21, -34; 13, -21]
def N : Matrix (Fin 2) (Fin 2) ℤ := !![5, 3; 3, 2]

theorem matrix_equation : N * A = B := by sorry

end matrix_equation_l2974_297420


namespace emily_gardens_l2974_297414

theorem emily_gardens (total_seeds : ℕ) (big_garden_seeds : ℕ) (seeds_per_small_garden : ℕ) 
  (h1 : total_seeds = 41)
  (h2 : big_garden_seeds = 29)
  (h3 : seeds_per_small_garden = 4) :
  (total_seeds - big_garden_seeds) / seeds_per_small_garden = 3 :=
by sorry

end emily_gardens_l2974_297414


namespace diagonals_in_150_degree_polygon_l2974_297417

/-- A polygon where all interior angles are 150 degrees -/
structure RegularPolygon where
  interior_angle : ℝ
  interior_angle_eq : interior_angle = 150

/-- The number of diagonals from one vertex in a RegularPolygon -/
def diagonals_from_vertex (p : RegularPolygon) : ℕ :=
  9

/-- Theorem: In a polygon where all interior angles are 150°, 
    the number of diagonals that can be drawn from one vertex is 9 -/
theorem diagonals_in_150_degree_polygon (p : RegularPolygon) :
  diagonals_from_vertex p = 9 := by
  sorry

end diagonals_in_150_degree_polygon_l2974_297417


namespace hyperbola_equation_l2974_297487

/-- Represents a hyperbola with focus on the x-axis -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b
  h_relation : a^2 + b^2 = c^2

/-- The standard equation of a hyperbola -/
def standardEquation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- The equation of an asymptote of a hyperbola -/
def asymptoteEquation (h : Hyperbola) (x y : ℝ) : Prop :=
  y = (h.b / h.a) * x

theorem hyperbola_equation (h : Hyperbola) 
  (h_asymptote : asymptoteEquation h x y ↔ y = 2 * x)
  (h_focus : h.c = Real.sqrt 5) :
  standardEquation h x y ↔ x^2 - y^2 / 4 = 1 := by
  sorry

end hyperbola_equation_l2974_297487


namespace mama_bird_stolen_worms_l2974_297461

/-- The number of worms stolen from Mama bird -/
def stolen_worms : ℕ := by sorry

theorem mama_bird_stolen_worms :
  let babies : ℕ := 6
  let worms_per_baby_per_day : ℕ := 3
  let days : ℕ := 3
  let papa_worms : ℕ := 9
  let mama_worms : ℕ := 13
  let additional_worms_needed : ℕ := 34
  
  stolen_worms = 2 := by sorry

end mama_bird_stolen_worms_l2974_297461


namespace rectangular_prism_diagonal_l2974_297402

theorem rectangular_prism_diagonal (a b c : ℝ) (ha : a = 12) (hb : b = 15) (hc : c = 8) :
  Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 433 := by
  sorry

end rectangular_prism_diagonal_l2974_297402


namespace quadratic_inequality_l2974_297448

theorem quadratic_inequality (x : ℝ) : x^2 - 6*x - 16 > 0 ↔ x < -2 ∨ x > 8 := by sorry

end quadratic_inequality_l2974_297448


namespace hyeyoung_walk_distance_l2974_297490

/-- Given a promenade of length 6 km, prove that walking to its halfway point is 3 km. -/
theorem hyeyoung_walk_distance (promenade_length : ℝ) (hyeyoung_distance : ℝ) 
  (h1 : promenade_length = 6)
  (h2 : hyeyoung_distance = promenade_length / 2) :
  hyeyoung_distance = 3 := by
  sorry

end hyeyoung_walk_distance_l2974_297490


namespace smallest_n_for_f_greater_than_15_l2974_297424

-- Define the function f
def f (n : ℕ+) : ℕ := sorry

-- Theorem statement
theorem smallest_n_for_f_greater_than_15 :
  (∀ k : ℕ+, k < 4 → f k ≤ 15) ∧ f 4 > 15 := by sorry

end smallest_n_for_f_greater_than_15_l2974_297424


namespace acute_angle_m_range_l2974_297496

def a : ℝ × ℝ := (1, 2)
def b (m : ℝ) : ℝ × ℝ := (4, m)

def angle_is_acute (v w : ℝ × ℝ) : Prop :=
  0 < v.1 * w.1 + v.2 * w.2 ∧ 
  (v.1 * w.1 + v.2 * w.2)^2 < (v.1^2 + v.2^2) * (w.1^2 + w.2^2)

theorem acute_angle_m_range :
  ∀ m : ℝ, angle_is_acute a (b m) → m ∈ Set.Ioo (-2) 8 ∪ Set.Ioi 8 :=
by sorry

end acute_angle_m_range_l2974_297496


namespace victoria_wins_l2974_297482

/-- Represents a player in the game -/
inductive Player : Type
| Harry : Player
| Victoria : Player

/-- Represents a line segment on the grid -/
inductive Segment : Type
| EastWest : Segment
| NorthSouth : Segment

/-- Represents the state of the game -/
structure GameState :=
(turn : Player)
(harry_score : Nat)
(victoria_score : Nat)
(moves : List Segment)

/-- Represents a strategy for a player -/
def Strategy := GameState → Segment

/-- Determines if a move is valid for a given player -/
def valid_move (player : Player) (segment : Segment) : Bool :=
  match player, segment with
  | Player.Harry, Segment.EastWest => true
  | Player.Victoria, Segment.NorthSouth => true
  | _, _ => false

/-- Determines if a move completes a square -/
def completes_square (state : GameState) (segment : Segment) : Bool :=
  sorry -- Implementation details omitted

/-- Applies a move to the game state -/
def apply_move (state : GameState) (segment : Segment) : GameState :=
  sorry -- Implementation details omitted

/-- Determines if the game is over -/
def game_over (state : GameState) : Bool :=
  sorry -- Implementation details omitted

/-- Determines the winner of the game -/
def winner (state : GameState) : Option Player :=
  sorry -- Implementation details omitted

/-- Victoria's winning strategy -/
def victoria_strategy : Strategy :=
  sorry -- Implementation details omitted

/-- Theorem stating that Victoria has a winning strategy -/
theorem victoria_wins :
  ∀ (harry_strategy : Strategy),
  ∃ (final_state : GameState),
  (game_over final_state = true) ∧
  (winner final_state = some Player.Victoria) :=
sorry

end victoria_wins_l2974_297482


namespace complex_calculation_l2974_297435

theorem complex_calculation : 
  let z : ℂ := 1 + I
  z^2 - 2/z = -1 + 3*I := by
  sorry

end complex_calculation_l2974_297435


namespace fifth_term_of_special_arithmetic_sequence_l2974_297474

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- The main theorem -/
theorem fifth_term_of_special_arithmetic_sequence (seq : ArithmeticSequence) 
    (h1 : seq.a 1 = 2)
    (h2 : 3 * seq.S 3 = seq.S 2 + seq.S 4) : 
  seq.a 5 = -10 := by
  sorry

end fifth_term_of_special_arithmetic_sequence_l2974_297474


namespace max_payment_is_31_l2974_297401

/-- Represents a four-digit number of the form 20** -/
def FourDigitNumber := {n : ℕ | 2000 ≤ n ∧ n ≤ 2099}

/-- Payments for divisibility -/
def payments : List ℕ := [1, 3, 5, 7, 9, 11]

/-- Divisors to check -/
def divisors : List ℕ := [1, 3, 5, 7, 9, 11]

/-- Calculate the payment for a given number -/
def calculatePayment (n : FourDigitNumber) : ℕ :=
  (List.zip divisors payments).foldl
    (fun acc (d, p) => if n % d = 0 then acc + p else acc)
    0

/-- The maximum payment possible -/
def maxPayment : ℕ := 31

theorem max_payment_is_31 :
  ∃ (n : FourDigitNumber), calculatePayment n = maxPayment ∧
  ∀ (m : FourDigitNumber), calculatePayment m ≤ maxPayment :=
sorry

end max_payment_is_31_l2974_297401


namespace min_sum_product_72_l2974_297405

theorem min_sum_product_72 (a b : ℤ) (h : a * b = 72) :
  ∀ x y : ℤ, x * y = 72 → a + b ≤ x + y ∧ ∃ a₀ b₀ : ℤ, a₀ * b₀ = 72 ∧ a₀ + b₀ = -17 :=
by sorry

end min_sum_product_72_l2974_297405


namespace birds_and_storks_count_l2974_297489

/-- Given initial birds, storks, and additional birds, calculates the total number of birds and storks -/
def total_birds_and_storks (initial_birds : ℕ) (storks : ℕ) (additional_birds : ℕ) : ℕ :=
  initial_birds + additional_birds + storks

/-- Proves that with 3 initial birds, 2 storks, and 5 additional birds, the total is 10 -/
theorem birds_and_storks_count : total_birds_and_storks 3 2 5 = 10 := by
  sorry

end birds_and_storks_count_l2974_297489


namespace equation_solution_l2974_297429

theorem equation_solution (n k l m : ℕ) :
  l > 1 →
  (1 + n^k)^l = 1 + n^m →
  n = 2 ∧ k = 1 ∧ l = 2 ∧ m = 3 := by
  sorry

end equation_solution_l2974_297429


namespace locus_perpendicular_tangents_l2974_297413

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := 4 * x^2 + 5 * y^2 = 20

/-- Tangent line to the ellipse at point (a, b) -/
def tangent_line (x y a b : ℝ) : Prop :=
  ellipse a b ∧ 4 * a * x + 5 * b * y = 20

/-- Two lines are perpendicular -/
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 * x2 + y1 * y2 = 0

/-- The theorem: locus of points with perpendicular tangents -/
theorem locus_perpendicular_tangents (x y : ℝ) :
  (∃ a1 b1 a2 b2 : ℝ,
    tangent_line x y a1 b1 ∧
    tangent_line x y a2 b2 ∧
    perpendicular (x - a1) (y - b1) (x - a2) (y - b2)) →
  x^2 + y^2 = 9 := by
  sorry

end locus_perpendicular_tangents_l2974_297413


namespace probability_one_blue_is_9_22_l2974_297462

/-- Represents the number of jellybeans of each color in the bowl -/
structure JellyBeanBowl where
  red : ℕ
  blue : ℕ
  white : ℕ

/-- Calculates the probability of picking exactly one blue jellybean -/
def probability_one_blue (bowl : JellyBeanBowl) : ℚ :=
  let total := bowl.red + bowl.blue + bowl.white
  let favorable := bowl.blue * (total - bowl.blue).choose 2
  favorable / total.choose 3

/-- The main theorem stating the probability of picking exactly one blue jellybean -/
theorem probability_one_blue_is_9_22 :
  probability_one_blue ⟨5, 2, 5⟩ = 9/22 := by
  sorry

#eval probability_one_blue ⟨5, 2, 5⟩

end probability_one_blue_is_9_22_l2974_297462


namespace total_power_cost_l2974_297416

def refrigerator_cost (water_heater_cost : ℝ) : ℝ := 3 * water_heater_cost

def electric_oven_cost : ℝ := 500

theorem total_power_cost (water_heater_cost : ℝ) 
  (h1 : electric_oven_cost = 2 * water_heater_cost) :
  water_heater_cost + refrigerator_cost water_heater_cost + electric_oven_cost = 1500 := by
  sorry

end total_power_cost_l2974_297416


namespace defect_selection_probability_l2974_297443

/-- Given a set of tubes with defects, calculate the probability of selecting specific defect types --/
theorem defect_selection_probability
  (total_tubes : ℕ)
  (type_a_defects : ℕ)
  (type_b_defects : ℕ)
  (h1 : total_tubes = 50)
  (h2 : type_a_defects = 5)
  (h3 : type_b_defects = 3)
  : ℚ :=
  3 / 490

#check defect_selection_probability

end defect_selection_probability_l2974_297443


namespace num_rna_molecules_l2974_297427

/-- Represents the number of possible bases for each position in an RNA molecule -/
def num_bases : ℕ := 4

/-- Represents the length of the RNA molecule -/
def rna_length : ℕ := 100

/-- Theorem stating that the number of unique RNA molecules is 4^100 -/
theorem num_rna_molecules : (num_bases : ℕ) ^ rna_length = 4 ^ 100 := by
  sorry

end num_rna_molecules_l2974_297427


namespace only_cone_cannot_have_quadrilateral_cross_section_l2974_297456

-- Define the types of solids
inductive Solid
  | Cylinder
  | Cone
  | FrustumOfCone
  | Prism

-- Define a function that checks if a solid can have a quadrilateral cross-section
def canHaveQuadrilateralCrossSection (s : Solid) : Prop :=
  match s with
  | Solid.Cylinder => true
  | Solid.Cone => false
  | Solid.FrustumOfCone => true
  | Solid.Prism => true

-- Theorem stating that only a Cone cannot have a quadrilateral cross-section
theorem only_cone_cannot_have_quadrilateral_cross_section :
  ∀ s : Solid, ¬(canHaveQuadrilateralCrossSection s) ↔ s = Solid.Cone :=
by
  sorry


end only_cone_cannot_have_quadrilateral_cross_section_l2974_297456


namespace max_CP_value_l2974_297437

-- Define the equilateral triangle ABC
def EquilateralTriangle (A B C : ℝ × ℝ) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

-- Define the point P
def P : ℝ × ℝ := sorry

-- Define the distances
def AP : ℝ := 2
def BP : ℝ := 3

-- Theorem statement
theorem max_CP_value 
  (A B C : ℝ × ℝ) 
  (h_equilateral : EquilateralTriangle A B C) 
  (h_AP : dist A P = AP) 
  (h_BP : dist B P = BP) :
  ∀ P', dist C P' ≤ 5 :=
sorry

end max_CP_value_l2974_297437


namespace polynomial_factorization_l2974_297495

theorem polynomial_factorization (x : ℤ) :
  4 * (x + 4) * (x + 7) * (x + 9) * (x + 11) - 5 * x^2 = (2 * x^2 + 72 * x + 126)^2 := by
  sorry

end polynomial_factorization_l2974_297495


namespace problem_solution_l2974_297403

theorem problem_solution (r s : ℝ) 
  (h1 : 1 < r) 
  (h2 : r < s) 
  (h3 : 1/r + 1/s = 3/4) 
  (h4 : r*s = 8) : 
  s = 4 := by sorry

end problem_solution_l2974_297403


namespace circle_diameter_from_area_l2974_297483

theorem circle_diameter_from_area :
  ∀ (r : ℝ), r > 0 → π * r^2 = 150 * π → 2 * r = 10 * Real.sqrt 6 := by
  sorry

end circle_diameter_from_area_l2974_297483


namespace largest_prime_factor_of_9973_l2974_297458

theorem largest_prime_factor_of_9973 :
  ∃ (p : ℕ), p.Prime ∧ p ∣ 9973 ∧ p = 103 ∧ ∀ (q : ℕ), q.Prime → q ∣ 9973 → q ≤ p :=
by
  sorry

end largest_prime_factor_of_9973_l2974_297458


namespace fruit_spending_sum_l2974_297446

/-- The total amount Mary spent on fruits after discounts -/
def total_spent : ℝ := 52.09

/-- The amount Mary paid for berries -/
def berries_price : ℝ := 11.08

/-- The amount Mary paid for apples -/
def apples_price : ℝ := 14.33

/-- The amount Mary paid for peaches -/
def peaches_price : ℝ := 9.31

/-- The amount Mary paid for grapes -/
def grapes_price : ℝ := 7.50

/-- The amount Mary paid for bananas -/
def bananas_price : ℝ := 5.25

/-- The amount Mary paid for pineapples -/
def pineapples_price : ℝ := 4.62

/-- Theorem stating that the sum of individual fruit prices equals the total spent -/
theorem fruit_spending_sum :
  berries_price + apples_price + peaches_price + grapes_price + bananas_price + pineapples_price = total_spent :=
by sorry

end fruit_spending_sum_l2974_297446


namespace sara_trout_count_l2974_297460

theorem sara_trout_count (melanie_trout : ℕ) (sara_trout : ℕ) : 
  melanie_trout = 10 → 
  melanie_trout = 2 * sara_trout → 
  sara_trout = 5 := by
sorry

end sara_trout_count_l2974_297460


namespace sum_not_equal_product_l2974_297477

theorem sum_not_equal_product : (0 + 1 + 2 + 3) ≠ (0 * 1 * 2 * 3) := by
  sorry

end sum_not_equal_product_l2974_297477


namespace arithmetic_geometric_intersection_l2974_297475

/-- An arithmetic sequence of integers -/
def ArithmeticSequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def GeometricSequence (b : ℕ → ℤ) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ ∀ n : ℕ, b (n + 1) = r * b n

/-- The theorem statement -/
theorem arithmetic_geometric_intersection (a : ℕ → ℤ) (d : ℤ) (n₁ : ℕ) :
  d ≠ 0 →
  ArithmeticSequence a d →
  a 5 = 6 →
  5 < n₁ →
  GeometricSequence (fun n ↦ if n = 1 then a 3 else if n = 2 then a 5 else a (n₁ + n - 3)) →
  (∃ k : ℕ, k ≤ 7 ∧
    ∀ n : ℕ, n ≤ 2015 →
      (∃ m : ℕ, m ≤ k ∧ a n = if m = 1 then a 3 else if m = 2 then a 5 else a (n₁ + m - 3))) ∧
  (∀ k : ℕ, k > 7 →
    ¬∀ n : ℕ, n ≤ 2015 →
      (∃ m : ℕ, m ≤ k ∧ a n = if m = 1 then a 3 else if m = 2 then a 5 else a (n₁ + m - 3))) :=
by
  sorry


end arithmetic_geometric_intersection_l2974_297475


namespace solution_set_f_range_of_a_l2974_297425

-- Part 1
def f (x : ℝ) : ℝ := |2 * x + 1| + |2 * x - 3|

theorem solution_set_f (x : ℝ) : f x = 4 ↔ -1/2 ≤ x ∧ x ≤ 3/2 := by sorry

-- Part 2
def f' (a x : ℝ) : ℝ := |x + a| + |x - 1|
def g (x : ℝ) : ℝ := |x - 2| + 1

theorem range_of_a (a : ℝ) :
  (∀ x₁, ∃ x₂, g x₂ = f' a x₁) → a ≤ -2 ∨ a ≥ 0 := by sorry

end solution_set_f_range_of_a_l2974_297425


namespace equal_marked_cells_exist_l2974_297447

/-- Represents an L-shaped triomino -/
structure Triomino where
  cells : Fin 3 → (Fin 2010 × Fin 2010)

/-- Represents a marking of cells in the grid -/
def Marking := Fin 2010 → Fin 2010 → Bool

/-- Checks if a marking is valid (one cell per triomino) -/
def isValidMarking (grid : List Triomino) (m : Marking) : Prop := sorry

/-- Counts marked cells in a given row -/
def countMarkedInRow (m : Marking) (row : Fin 2010) : Nat := sorry

/-- Counts marked cells in a given column -/
def countMarkedInColumn (m : Marking) (col : Fin 2010) : Nat := sorry

/-- Main theorem statement -/
theorem equal_marked_cells_exist (grid : List Triomino) 
  (h : grid.length = 2010 * 2010 / 3) : 
  ∃ m : Marking, 
    isValidMarking grid m ∧ 
    (∀ r₁ r₂ : Fin 2010, countMarkedInRow m r₁ = countMarkedInRow m r₂) ∧
    (∀ c₁ c₂ : Fin 2010, countMarkedInColumn m c₁ = countMarkedInColumn m c₂) := by
  sorry

end equal_marked_cells_exist_l2974_297447


namespace sergio_income_l2974_297470

/-- Represents the total income from fruit sales -/
def total_income (mango_production : ℕ) (price_per_kg : ℕ) : ℕ :=
  let apple_production := 2 * mango_production
  let orange_production := mango_production + 200
  (apple_production + orange_production + mango_production) * price_per_kg

/-- Proves that Mr. Sergio's total income is $90000 given the conditions -/
theorem sergio_income : total_income 400 50 = 90000 := by
  sorry

end sergio_income_l2974_297470


namespace percent_of_percent_l2974_297415

theorem percent_of_percent (y : ℝ) : (0.3 * 0.6 * y) = (0.18 * y) := by sorry

end percent_of_percent_l2974_297415
