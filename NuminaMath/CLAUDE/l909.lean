import Mathlib

namespace NUMINAMATH_CALUDE_value_of_a_minus_b_l909_90960

theorem value_of_a_minus_b (a b : ℝ) : 
  ({x : ℝ | |x - a| < b} = {x : ℝ | 2 < x ∧ x < 4}) → a - b = 2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_minus_b_l909_90960


namespace NUMINAMATH_CALUDE_value_of_z_l909_90998

theorem value_of_z (x y z : ℝ) : 
  y = 3 * x - 5 → 
  z = 3 * x + 3 → 
  y = 1 → 
  z = 9 := by
sorry

end NUMINAMATH_CALUDE_value_of_z_l909_90998


namespace NUMINAMATH_CALUDE_sum_of_binary_numbers_l909_90900

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def num1 : List Bool := [true, true, true, true, true, true, true, true, true]
def num2 : List Bool := [true, false, false, false, false, false, true]

theorem sum_of_binary_numbers :
  binary_to_decimal num1 + binary_to_decimal num2 = 576 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_binary_numbers_l909_90900


namespace NUMINAMATH_CALUDE_orange_juice_distribution_l909_90917

theorem orange_juice_distribution (pitcher_capacity : ℝ) (h : pitcher_capacity > 0) :
  let juice_amount : ℝ := (5 / 8) * pitcher_capacity
  let num_cups : ℕ := 4
  let juice_per_cup : ℝ := juice_amount / num_cups
  (juice_per_cup / pitcher_capacity) * 100 = 15.625 := by
sorry

end NUMINAMATH_CALUDE_orange_juice_distribution_l909_90917


namespace NUMINAMATH_CALUDE_ten_day_search_cost_l909_90971

/-- Tom's charging scheme for item search -/
def search_cost (days : ℕ) : ℕ :=
  let initial_rate := 100
  let discounted_rate := 60
  let initial_period := 5
  if days ≤ initial_period then
    days * initial_rate
  else
    initial_period * initial_rate + (days - initial_period) * discounted_rate

/-- The theorem stating the total cost for a 10-day search -/
theorem ten_day_search_cost : search_cost 10 = 800 := by
  sorry

end NUMINAMATH_CALUDE_ten_day_search_cost_l909_90971


namespace NUMINAMATH_CALUDE_game_strategies_l909_90905

def game_state (n : ℕ) : Prop := n > 0

def player_A_move (n m : ℕ) : Prop := n ≤ m ∧ m ≤ n^2

def player_B_move (n m : ℕ) : Prop := ∃ (p : ℕ) (k : ℕ), Nat.Prime p ∧ n = m * p^k

def A_wins (n : ℕ) : Prop := n = 1990

def B_wins (n : ℕ) : Prop := n = 1

def A_has_winning_strategy (n₀ : ℕ) : Prop := game_state n₀ ∧ n₀ ≥ 8

def B_has_winning_strategy (n₀ : ℕ) : Prop := game_state n₀ ∧ n₀ ≤ 5

def no_guaranteed_winning_strategy (n₀ : ℕ) : Prop := game_state n₀ ∧ (n₀ = 6 ∨ n₀ = 7)

theorem game_strategies :
  ∀ n₀ : ℕ, game_state n₀ →
    (A_has_winning_strategy n₀ ↔ n₀ ≥ 8) ∧
    (B_has_winning_strategy n₀ ↔ n₀ ≤ 5) ∧
    (no_guaranteed_winning_strategy n₀ ↔ (n₀ = 6 ∨ n₀ = 7)) :=
  sorry

end NUMINAMATH_CALUDE_game_strategies_l909_90905


namespace NUMINAMATH_CALUDE_prime_comparison_l909_90918

theorem prime_comparison (x y : ℕ) (hx : Prime x) (hy : Prime y) 
  (hlcm : Nat.lcm x y = 10) (heq : 2 * x + y = 12) : x > y := by
  sorry

end NUMINAMATH_CALUDE_prime_comparison_l909_90918


namespace NUMINAMATH_CALUDE_problem_statement_l909_90915

theorem problem_statement (x y z : ℝ) 
  (h1 : x + y - 6 = 0) 
  (h2 : z^2 + 9 = x*y) : 
  x^2 + (1/3)*y^2 = 12 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l909_90915


namespace NUMINAMATH_CALUDE_equilibrium_instability_l909_90988

/-- The system of differential equations -/
def system (x y : ℝ) : ℝ × ℝ :=
  (y^3 + x^5, x^3 + y^5)

/-- The Lyapunov function -/
def v (x y : ℝ) : ℝ :=
  x^4 - y^4

/-- The time derivative of the Lyapunov function -/
def dv_dt (x y : ℝ) : ℝ :=
  4 * (x^8 - y^8)

/-- Theorem stating the instability of the equilibrium point (0, 0) -/
theorem equilibrium_instability :
  ∃ (ε : ℝ), ε > 0 ∧
  ∀ (δ : ℝ), δ > 0 →
  ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 < δ^2 ∧
  ∃ (t : ℝ), t > 0 ∧
  let (x, y) := system x₀ y₀
  x^2 + y^2 > ε^2 :=
sorry

end NUMINAMATH_CALUDE_equilibrium_instability_l909_90988


namespace NUMINAMATH_CALUDE_star_value_proof_l909_90965

def star (a b : ℤ) : ℤ := a^2 + 2*a*b + b^2

theorem star_value_proof (a b : ℤ) (h : 4 ∣ (a + b)) : 
  a = 3 → b = 5 → star a b = 64 := by
  sorry

end NUMINAMATH_CALUDE_star_value_proof_l909_90965


namespace NUMINAMATH_CALUDE_min_value_when_a_is_1_range_of_a_for_bounded_f_l909_90923

/-- The function f(x) defined as |2x-a| - |x+3| --/
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| - |x + 3|

/-- Theorem stating the minimum value of f(x) when a = 1 --/
theorem min_value_when_a_is_1 :
  ∃ (m : ℝ), m = -7/2 ∧ ∀ (x : ℝ), f 1 x ≥ m := by sorry

/-- Theorem stating the range of a for which f(x) ≤ 4 when x ∈ [0,3] --/
theorem range_of_a_for_bounded_f :
  ∀ (a : ℝ), (∀ (x : ℝ), x ∈ Set.Icc 0 3 → f a x ≤ 4) ↔ a ∈ Set.Icc (-4) 7 := by sorry

end NUMINAMATH_CALUDE_min_value_when_a_is_1_range_of_a_for_bounded_f_l909_90923


namespace NUMINAMATH_CALUDE_vertices_form_parabola_l909_90935

/-- A parabola in the family of parabolas described by y = x^2 + 2ax + a for all real a -/
structure Parabola where
  a : ℝ

/-- The vertex of a parabola in the family -/
def vertex (p : Parabola) : ℝ × ℝ :=
  (-p.a, p.a - p.a^2)

/-- The set of all vertices of parabolas in the family -/
def vertex_set : Set (ℝ × ℝ) :=
  {v | ∃ p : Parabola, v = vertex p}

/-- The equation of the curve on which the vertices lie -/
def vertex_curve (x y : ℝ) : Prop :=
  y = -x^2 - x

theorem vertices_form_parabola :
  ∀ v ∈ vertex_set, vertex_curve v.1 v.2 := by
  sorry

#check vertices_form_parabola

end NUMINAMATH_CALUDE_vertices_form_parabola_l909_90935


namespace NUMINAMATH_CALUDE_walmart_cards_sent_eq_two_l909_90931

/-- Represents the gift card scenario --/
structure GiftCardScenario where
  bestBuyCards : ℕ
  bestBuyValue : ℕ
  walmartCards : ℕ
  walmartValue : ℕ
  sentBestBuy : ℕ
  remainingValue : ℕ

/-- Calculates the number of Walmart gift cards sent --/
def walmartsCardsSent (s : GiftCardScenario) : ℕ :=
  let totalInitialValue := s.bestBuyCards * s.bestBuyValue + s.walmartCards * s.walmartValue
  let sentValue := totalInitialValue - s.remainingValue
  let sentWalmartValue := sentValue - s.sentBestBuy * s.bestBuyValue
  sentWalmartValue / s.walmartValue

/-- Theorem stating the number of Walmart gift cards sent --/
theorem walmart_cards_sent_eq_two (s : GiftCardScenario) 
  (h1 : s.bestBuyCards = 6)
  (h2 : s.bestBuyValue = 500)
  (h3 : s.walmartCards = 9)
  (h4 : s.walmartValue = 200)
  (h5 : s.sentBestBuy = 1)
  (h6 : s.remainingValue = 3900) :
  walmartsCardsSent s = 2 := by
  sorry


end NUMINAMATH_CALUDE_walmart_cards_sent_eq_two_l909_90931


namespace NUMINAMATH_CALUDE_stone_heap_theorem_l909_90948

/-- 
Given k ≥ 3 heaps of stones with 1, 2, ..., k stones respectively,
after merging heaps, the final number of stones p is given by
p = (k + 1) * (3k - 1) / 8.
This function returns p given k.
-/
def final_stones (k : ℕ) : ℚ :=
  (k + 1) * (3 * k - 1) / 8

/-- 
This theorem states that for k ≥ 3, the final number of stones p
is a perfect square if and only if both 2k + 2 and 3k + 1 are perfect squares,
and that the least k satisfying this condition is 161.
-/
theorem stone_heap_theorem (k : ℕ) (h : k ≥ 3) :
  (∃ n : ℕ, final_stones k = n^2) ↔ 
  (∃ x y : ℕ, 2*k + 2 = x^2 ∧ 3*k + 1 = y^2) ∧
  (∀ m : ℕ, m < 161 → ¬(∃ x y : ℕ, 2*m + 2 = x^2 ∧ 3*m + 1 = y^2)) :=
sorry

end NUMINAMATH_CALUDE_stone_heap_theorem_l909_90948


namespace NUMINAMATH_CALUDE_expand_expression_l909_90991

theorem expand_expression (y : ℝ) : 5 * (y + 6) * (y - 3) = 5 * y^2 + 15 * y - 90 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l909_90991


namespace NUMINAMATH_CALUDE_average_of_three_angles_l909_90955

/-- Given that the average of α and β is 105°, prove that the average of α, β, and γ is 80°. -/
theorem average_of_three_angles (α β γ : ℝ) :
  (α + β) / 2 = 105 → (α + β + γ) / 3 = 80 :=
by sorry

end NUMINAMATH_CALUDE_average_of_three_angles_l909_90955


namespace NUMINAMATH_CALUDE_sequence_sum_l909_90913

-- Define the sequence a_n
def a : ℕ → ℕ
  | 0 => 1  -- First term is 1
  | n + 1 => 
    let k := (n + 1).sqrt  -- k-th group
    if (k * k ≤ n + 1) ∧ (n + 1 < (k + 1) * (k + 1)) then
      if n + 1 = k * k then 1 else 2
    else a n  -- This case should never happen, but Lean needs it for totality

-- Define the sum S_n
def S (n : ℕ) : ℕ := (List.range n).map a |>.sum

-- Theorem statement
theorem sequence_sum :
  S 20 = 36 ∧ S 2017 = 3989 := by sorry

end NUMINAMATH_CALUDE_sequence_sum_l909_90913


namespace NUMINAMATH_CALUDE_johns_trip_cost_l909_90912

/-- Calculates the total cost of a car rental trip -/
def total_trip_cost (rental_cost : ℚ) (gas_price : ℚ) (gas_needed : ℚ) (mileage_cost : ℚ) (distance : ℚ) : ℚ :=
  rental_cost + gas_price * gas_needed + mileage_cost * distance

/-- Theorem stating that the total cost of John's trip is $338 -/
theorem johns_trip_cost : 
  total_trip_cost 150 3.5 8 0.5 320 = 338 := by
  sorry

end NUMINAMATH_CALUDE_johns_trip_cost_l909_90912


namespace NUMINAMATH_CALUDE_second_player_wins_l909_90920

/-- Represents the game state with the number of diamonds and the current player. -/
structure GameState :=
  (diamonds : ℕ)
  (currentPlayer : Bool)

/-- Defines a valid move in the game. -/
def validMove (s : GameState) (newPiles : ℕ × ℕ) : Prop :=
  s.diamonds = newPiles.1 + newPiles.2 ∧ newPiles.1 > 0 ∧ newPiles.2 > 0

/-- Defines the game over condition. -/
def gameOver (s : GameState) : Prop :=
  s.diamonds = 1

/-- Defines the winning condition for a player. -/
def wins (player : Bool) (s : GameState) : Prop :=
  gameOver s ∧ s.currentPlayer ≠ player

/-- Theorem: The second player has a winning strategy in a game with 2017 diamonds. -/
theorem second_player_wins :
  ∃ (strategy : GameState → ℕ × ℕ),
    ∀ (firstPlayerMoves : GameState → ℕ × ℕ),
      wins false (GameState.mk 2017 true) := by
        sorry

#check second_player_wins

end NUMINAMATH_CALUDE_second_player_wins_l909_90920


namespace NUMINAMATH_CALUDE_absolute_value_sum_difference_l909_90999

theorem absolute_value_sum_difference (a b : ℝ) 
  (ha : |a| = 4) (hb : |b| = 3) : 
  ((a * b < 0 → |a + b| = 1) ∧ (a * b > 0 → |a - b| = 1)) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_difference_l909_90999


namespace NUMINAMATH_CALUDE_first_volume_pages_l909_90901

/-- Given a two-volume book with a total of 999 digits used for page numbers,
    where the first volume has 9 more pages than the second volume,
    prove that the number of pages in the first volume is 207. -/
theorem first_volume_pages (total_digits : ℕ) (page_difference : ℕ) 
  (h1 : total_digits = 999)
  (h2 : page_difference = 9) :
  ∃ (first_volume second_volume : ℕ),
    first_volume = second_volume + page_difference ∧
    first_volume = 207 :=
by sorry

end NUMINAMATH_CALUDE_first_volume_pages_l909_90901


namespace NUMINAMATH_CALUDE_total_pens_count_l909_90975

theorem total_pens_count (red_pens : ℕ) (black_pens : ℕ) (blue_pens : ℕ) 
  (h1 : red_pens = 8)
  (h2 : black_pens = red_pens + 10)
  (h3 : blue_pens = red_pens + 7) :
  red_pens + black_pens + blue_pens = 41 := by
  sorry

end NUMINAMATH_CALUDE_total_pens_count_l909_90975


namespace NUMINAMATH_CALUDE_two_negative_solutions_iff_b_in_range_l909_90926

/-- The equation 9^x + |3^x + b| = 5 has exactly two negative real number solutions if and only if b is in the open interval (-5.25, -5) -/
theorem two_negative_solutions_iff_b_in_range (b : ℝ) : 
  (∃ (x y : ℝ), x < 0 ∧ y < 0 ∧ x ≠ y ∧ 
    (9^x + |3^x + b| = 5) ∧ 
    (9^y + |3^y + b| = 5) ∧
    (∀ z : ℝ, z < 0 → z ≠ x → z ≠ y → 9^z + |3^z + b| ≠ 5)) ↔ 
  -5.25 < b ∧ b < -5 :=
sorry

end NUMINAMATH_CALUDE_two_negative_solutions_iff_b_in_range_l909_90926


namespace NUMINAMATH_CALUDE_g_has_three_zeros_l909_90942

/-- A function g(x) with a parameter n -/
def g (n : ℕ) (x : ℝ) : ℝ := 2 * x^n + 10 * x^2 - 2 * x - 1

/-- Theorem stating that g(x) has exactly 3 real zeros when n > 3 and n is odd -/
theorem g_has_three_zeros (n : ℕ) (hn : n > 3) (hodd : Odd n) :
  ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, g n x = 0 :=
sorry

end NUMINAMATH_CALUDE_g_has_three_zeros_l909_90942


namespace NUMINAMATH_CALUDE_bryans_total_amount_l909_90994

/-- The total amount received from selling precious stones -/
def total_amount (num_stones : ℕ) (price_per_stone : ℕ) : ℕ :=
  num_stones * price_per_stone

/-- Theorem: Bryan's total amount from selling 8 stones at 1785 dollars each is 14280 dollars -/
theorem bryans_total_amount :
  total_amount 8 1785 = 14280 := by
  sorry

end NUMINAMATH_CALUDE_bryans_total_amount_l909_90994


namespace NUMINAMATH_CALUDE_savings_account_interest_rate_l909_90921

theorem savings_account_interest_rate (initial_deposit : ℝ) (first_year_balance : ℝ) (total_increase_percentage : ℝ) :
  initial_deposit = 1000 →
  first_year_balance = 1100 →
  total_increase_percentage = 32 →
  let total_amount := initial_deposit * (1 + total_increase_percentage / 100)
  let second_year_increase := total_amount - first_year_balance
  let second_year_increase_percentage := (second_year_increase / first_year_balance) * 100
  second_year_increase_percentage = 20 := by
sorry

end NUMINAMATH_CALUDE_savings_account_interest_rate_l909_90921


namespace NUMINAMATH_CALUDE_smallest_factorization_coefficient_l909_90966

theorem smallest_factorization_coefficient (b : ℕ+) (p q : ℤ) : 
  (∀ x, (x^2 : ℤ) + b * x + 1760 = (x + p) * (x + q)) →
  (∀ b' : ℕ+, b' < b → 
    ¬∃ p' q' : ℤ, ∀ x, (x^2 : ℤ) + b' * x + 1760 = (x + p') * (x + q')) →
  b = 108 := by
sorry

end NUMINAMATH_CALUDE_smallest_factorization_coefficient_l909_90966


namespace NUMINAMATH_CALUDE_rabbits_ate_27_watermelons_l909_90969

/-- The number of watermelons eaten by rabbits, given initial and remaining counts. -/
def watermelons_eaten (initial : ℕ) (remaining : ℕ) : ℕ := initial - remaining

/-- Theorem stating that 27 watermelons were eaten by rabbits. -/
theorem rabbits_ate_27_watermelons : 
  watermelons_eaten 35 8 = 27 := by sorry

end NUMINAMATH_CALUDE_rabbits_ate_27_watermelons_l909_90969


namespace NUMINAMATH_CALUDE_beads_per_necklace_l909_90982

def total_beads : ℕ := 52
def necklaces_made : ℕ := 26

theorem beads_per_necklace : 
  total_beads / necklaces_made = 2 := by sorry

end NUMINAMATH_CALUDE_beads_per_necklace_l909_90982


namespace NUMINAMATH_CALUDE_subset_equality_l909_90932

theorem subset_equality (h : ℕ) (X S : Set ℕ) : h ≥ 3 →
  X = {n : ℕ | n ≥ 2 * h} →
  S ⊆ X →
  S.Nonempty →
  (∀ a b : ℕ, a ≥ h → b ≥ h → (a + b) ∈ S → (a * b) ∈ S) →
  (∀ a b : ℕ, a ≥ h → b ≥ h → (a * b) ∈ S → (a + b) ∈ S) →
  S = X :=
by sorry

end NUMINAMATH_CALUDE_subset_equality_l909_90932


namespace NUMINAMATH_CALUDE_hannah_adblock_efficiency_l909_90952

/-- The percentage of ads not blocked by Hannah's AdBlock -/
def ads_not_blocked : ℝ := sorry

/-- The percentage of not blocked ads that are interesting -/
def interesting_not_blocked_ratio : ℝ := 0.20

/-- The percentage of all ads that are not interesting and not blocked -/
def not_interesting_not_blocked_ratio : ℝ := 0.16

theorem hannah_adblock_efficiency :
  ads_not_blocked = 0.20 :=
sorry

end NUMINAMATH_CALUDE_hannah_adblock_efficiency_l909_90952


namespace NUMINAMATH_CALUDE_gcd_lcm_42_30_l909_90980

theorem gcd_lcm_42_30 :
  (Nat.gcd 42 30 = 6) ∧ (Nat.lcm 42 30 = 210) := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_42_30_l909_90980


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l909_90906

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x^2 - 2*x < 0}

def B : Set ℝ := {x | x > 1}

theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l909_90906


namespace NUMINAMATH_CALUDE_equation_solution_l909_90956

theorem equation_solution :
  ∃ (x : ℝ), x ≠ -1 ∧ x ≠ 1 ∧ (x - 1) / (x + 1) - 3 / (x^2 - 1) = 1 ∧ x = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l909_90956


namespace NUMINAMATH_CALUDE_potassium_bromate_weight_l909_90949

/-- The atomic weight of Potassium in g/mol -/
def atomic_weight_K : ℝ := 39.10

/-- The atomic weight of Bromine in g/mol -/
def atomic_weight_Br : ℝ := 79.90

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of Potassium atoms in Potassium Bromate -/
def num_K : ℕ := 1

/-- The number of Bromine atoms in Potassium Bromate -/
def num_Br : ℕ := 1

/-- The number of Oxygen atoms in Potassium Bromate -/
def num_O : ℕ := 3

/-- The molecular weight of Potassium Bromate in g/mol -/
def molecular_weight_KBrO3 : ℝ :=
  num_K * atomic_weight_K + num_Br * atomic_weight_Br + num_O * atomic_weight_O

/-- Theorem stating that the molecular weight of Potassium Bromate is 167.00 g/mol -/
theorem potassium_bromate_weight : molecular_weight_KBrO3 = 167.00 := by
  sorry

end NUMINAMATH_CALUDE_potassium_bromate_weight_l909_90949


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l909_90983

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 2 - 1) :
  (1 + 1 / (a - 1)) / (a / (a^2 - 1)) = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l909_90983


namespace NUMINAMATH_CALUDE_industrial_machine_shirts_l909_90928

/-- The number of shirts made by an industrial machine yesterday -/
def shirts_yesterday (x : ℕ) : Prop :=
  let shirts_per_minute : ℕ := 8
  let total_minutes : ℕ := 2
  let shirts_today : ℕ := 3
  let total_shirts : ℕ := shirts_per_minute * total_minutes
  x = total_shirts - shirts_today

theorem industrial_machine_shirts : shirts_yesterday 13 := by
  sorry

end NUMINAMATH_CALUDE_industrial_machine_shirts_l909_90928


namespace NUMINAMATH_CALUDE_shaded_area_closest_to_21_l909_90940

/-- The area of the shaded region in a 4 × 6 rectangle with a circle of diameter 2 removed is closest to 21 -/
theorem shaded_area_closest_to_21 (rectangle_width : ℝ) (rectangle_height : ℝ) (circle_diameter : ℝ) :
  rectangle_width = 4 →
  rectangle_height = 6 →
  circle_diameter = 2 →
  ∃ (shaded_area : ℝ), 
    shaded_area = rectangle_width * rectangle_height - Real.pi * (circle_diameter / 2)^2 ∧
    abs (shaded_area - 21) = (Int.floor shaded_area - 21).natAbs.min (Int.ceil shaded_area - 21).natAbs := by
  sorry


end NUMINAMATH_CALUDE_shaded_area_closest_to_21_l909_90940


namespace NUMINAMATH_CALUDE_partnership_profit_theorem_l909_90902

/-- Represents the profit distribution in a partnership --/
structure Partnership where
  investment_ratio : ℝ  -- Ratio of A's investment to B's investment
  time_ratio : ℝ        -- Ratio of A's investment time to B's investment time
  b_profit : ℝ          -- B's profit

/-- Calculates the total profit of the partnership --/
def total_profit (p : Partnership) : ℝ :=
  let a_profit := p.b_profit * p.investment_ratio * p.time_ratio
  a_profit + p.b_profit

/-- Theorem stating that under the given conditions, the total profit is 28000 --/
theorem partnership_profit_theorem (p : Partnership) 
  (h1 : p.investment_ratio = 3)
  (h2 : p.time_ratio = 2)
  (h3 : p.b_profit = 4000) :
  total_profit p = 28000 := by
  sorry

#eval total_profit { investment_ratio := 3, time_ratio := 2, b_profit := 4000 }

end NUMINAMATH_CALUDE_partnership_profit_theorem_l909_90902


namespace NUMINAMATH_CALUDE_function_periodicity_l909_90996

/-- A function satisfying the given functional equation is periodic with period 4k -/
theorem function_periodicity (f : ℝ → ℝ) (k : ℝ) (hk : k ≠ 0) 
  (h : ∀ x, f (x + k) * (1 - f x) = 1 + f x) : 
  ∀ x, f (x + 4 * k) = f x := by
  sorry

end NUMINAMATH_CALUDE_function_periodicity_l909_90996


namespace NUMINAMATH_CALUDE_intersection_sum_l909_90943

/-- Two circles with centers on the line x + y = 0 intersect at points M(m, 1) and N(-1, n) -/
def circles_intersection (m n : ℝ) : Prop :=
  ∃ (c₁ c₂ : ℝ × ℝ), 
    (c₁.1 + c₁.2 = 0) ∧ 
    (c₂.1 + c₂.2 = 0) ∧ 
    ((m - c₁.1)^2 + (1 - c₁.2)^2 = (-1 - c₁.1)^2 + (n - c₁.2)^2) ∧
    ((m - c₂.1)^2 + (1 - c₂.2)^2 = (-1 - c₂.1)^2 + (n - c₂.2)^2)

/-- The theorem to be proved -/
theorem intersection_sum (m n : ℝ) (h : circles_intersection m n) : m + n = 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l909_90943


namespace NUMINAMATH_CALUDE_lcm_18_30_l909_90914

theorem lcm_18_30 : Nat.lcm 18 30 = 90 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_30_l909_90914


namespace NUMINAMATH_CALUDE_unique_prime_six_digit_number_l909_90963

def is_prime (n : Nat) : Prop := n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

def six_digit_number (B A : Nat) : Nat := 3000000 + B * 10000 + 1200 + A

theorem unique_prime_six_digit_number :
  ∃! (B A : Nat), B < 10 ∧ A < 10 ∧ 
    is_prime (six_digit_number B A) ∧
    B + A = 9 := by sorry

end NUMINAMATH_CALUDE_unique_prime_six_digit_number_l909_90963


namespace NUMINAMATH_CALUDE_arrangement_of_digits_and_blanks_l909_90922

theorem arrangement_of_digits_and_blanks : 
  let n : ℕ := 6  -- total number of boxes
  let k : ℕ := 4  -- number of distinct digits
  let b : ℕ := 2  -- number of blank spaces
  n! / b! = 360 := by
sorry

end NUMINAMATH_CALUDE_arrangement_of_digits_and_blanks_l909_90922


namespace NUMINAMATH_CALUDE_vertex_y_coordinate_is_zero_l909_90967

-- Define a trinomial function
def trinomial (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the condition that (f(x))^3 - f(x) = 0 has three real roots
def has_three_real_roots (f : ℝ → ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), (∀ x, f x = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
  (f x₁)^3 - f x₁ = 0 ∧ (f x₂)^3 - f x₂ = 0 ∧ (f x₃)^3 - f x₃ = 0

-- Theorem statement
theorem vertex_y_coordinate_is_zero 
  (a b c : ℝ) 
  (h : has_three_real_roots (trinomial a b c)) :
  let f := trinomial a b c
  let vertex_y := f (- b / (2 * a))
  vertex_y = 0 := by
sorry

end NUMINAMATH_CALUDE_vertex_y_coordinate_is_zero_l909_90967


namespace NUMINAMATH_CALUDE_max_intersections_three_lines_one_circle_l909_90910

-- Define a type for geometric figures
inductive Figure
| Line : Figure
| Circle : Figure

-- Define a function to count maximum intersections between two figures
def maxIntersections (f1 f2 : Figure) : ℕ :=
  match f1, f2 with
  | Figure.Line, Figure.Line => 1
  | Figure.Line, Figure.Circle => 2
  | Figure.Circle, Figure.Line => 2
  | Figure.Circle, Figure.Circle => 0

-- Theorem statement
theorem max_intersections_three_lines_one_circle :
  ∃ (l1 l2 l3 : Figure) (c : Figure),
    l1 = Figure.Line ∧ l2 = Figure.Line ∧ l3 = Figure.Line ∧ c = Figure.Circle ∧
    l1 ≠ l2 ∧ l2 ≠ l3 ∧ l1 ≠ l3 ∧
    (maxIntersections l1 l2 + maxIntersections l2 l3 + maxIntersections l1 l3 +
     maxIntersections l1 c + maxIntersections l2 c + maxIntersections l3 c) = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_max_intersections_three_lines_one_circle_l909_90910


namespace NUMINAMATH_CALUDE_correct_seniority_ranking_l909_90937

-- Define the type for colleagues
inductive Colleague : Type
  | Ella : Colleague
  | Mark : Colleague
  | Nora : Colleague

-- Define the seniority relation
def moreSeniorThan : Colleague → Colleague → Prop := sorry

-- Axioms for the problem conditions
axiom different_seniorities :
  ∀ (a b : Colleague), a ≠ b → (moreSeniorThan a b ∨ moreSeniorThan b a)

axiom exactly_one_true :
  (moreSeniorThan Colleague.Mark Colleague.Ella ∧ moreSeniorThan Colleague.Mark Colleague.Nora) ∨
  (¬moreSeniorThan Colleague.Ella Colleague.Mark ∨ ¬moreSeniorThan Colleague.Ella Colleague.Nora) ∨
  (¬moreSeniorThan Colleague.Mark Colleague.Nora ∨ moreSeniorThan Colleague.Nora Colleague.Mark)

-- The theorem to prove
theorem correct_seniority_ranking :
  moreSeniorThan Colleague.Ella Colleague.Nora ∧
  moreSeniorThan Colleague.Nora Colleague.Mark :=
by sorry

end NUMINAMATH_CALUDE_correct_seniority_ranking_l909_90937


namespace NUMINAMATH_CALUDE_expression_simplification_l909_90925

theorem expression_simplification (a b : ℝ) (h1 : a = 2) (h2 : b = -1) :
  2 * (-a^2 + 2*a*b) - 3 * (a*b - a^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l909_90925


namespace NUMINAMATH_CALUDE_clocks_chime_together_l909_90972

def clock1_interval : ℕ := 15
def clock2_interval : ℕ := 25

theorem clocks_chime_together : Nat.lcm clock1_interval clock2_interval = 75 := by
  sorry

end NUMINAMATH_CALUDE_clocks_chime_together_l909_90972


namespace NUMINAMATH_CALUDE_bingbing_correct_qianqian_incorrect_l909_90984

-- Define the basic parameters of the problem
def downstream_time : ℝ := 2
def upstream_time : ℝ := 2.5
def water_speed : ℝ := 3

-- Define Bingbing's equation
def bingbing_equation (x : ℝ) : Prop :=
  2 * (x + water_speed) = upstream_time * (x - water_speed)

-- Define Qianqian's equation
def qianqian_equation (x : ℝ) : Prop :=
  x / downstream_time - x / upstream_time = water_speed * downstream_time

-- Theorem stating that Bingbing's equation correctly models the problem
theorem bingbing_correct :
  ∃ (x : ℝ), bingbing_equation x ∧ x > 0 ∧ 
  (x * downstream_time = x * upstream_time) :=
sorry

-- Theorem stating that Qianqian's equation does not correctly model the problem
theorem qianqian_incorrect :
  ¬(∃ (x : ℝ), qianqian_equation x ∧ 
  (x * downstream_time = x * upstream_time ∨ x > 0)) :=
sorry

end NUMINAMATH_CALUDE_bingbing_correct_qianqian_incorrect_l909_90984


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l909_90916

theorem sum_of_a_and_b (a b : ℕ+) (h : a.val^2 - b.val^4 = 2009) : a.val + b.val = 47 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l909_90916


namespace NUMINAMATH_CALUDE_other_lateral_side_length_l909_90968

/-- A trapezoid with the property that a line through the midpoint of one lateral side
    divides it into two quadrilaterals, each with an inscribed circle -/
structure SpecialTrapezoid where
  /-- Length of one base -/
  a : ℝ
  /-- Length of the other base -/
  b : ℝ
  /-- The trapezoid has the special property -/
  has_special_property : Bool

/-- The length of the other lateral side in a special trapezoid -/
def other_lateral_side (t : SpecialTrapezoid) : ℝ :=
  t.a + t.b

theorem other_lateral_side_length (t : SpecialTrapezoid) 
  (h : t.has_special_property = true) : 
  other_lateral_side t = t.a + t.b := by
  sorry

end NUMINAMATH_CALUDE_other_lateral_side_length_l909_90968


namespace NUMINAMATH_CALUDE_age_puzzle_solution_l909_90987

/-- Represents a person in the age puzzle -/
structure Person where
  name : String
  age : Nat

/-- The conditions of the age puzzle -/
def AgePuzzle (tamara lena marina : Person) : Prop :=
  tamara.age = lena.age - 2 ∧
  tamara.age = marina.age + 1 ∧
  lena.age = marina.age + 3 ∧
  marina.age < tamara.age

/-- The theorem stating the unique solution to the age puzzle -/
theorem age_puzzle_solution :
  ∃! (tamara lena marina : Person),
    tamara.name = "Tamara" ∧
    lena.name = "Lena" ∧
    marina.name = "Marina" ∧
    AgePuzzle tamara lena marina ∧
    tamara.age = 23 ∧
    lena.age = 25 ∧
    marina.age = 22 := by
  sorry

end NUMINAMATH_CALUDE_age_puzzle_solution_l909_90987


namespace NUMINAMATH_CALUDE_problem_statement_l909_90959

theorem problem_statement (x : ℚ) (h : 5 * x - 3 = 15 * x + 21) : 
  3 * (2 * x + 5) = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l909_90959


namespace NUMINAMATH_CALUDE_paint_distribution_321_60_l909_90962

/-- Given a paint mixture with a ratio of red:white:blue and a total number of cans,
    calculate the number of cans for each color. -/
def paint_distribution (red white blue total : ℕ) : ℕ × ℕ × ℕ :=
  let sum := red + white + blue
  let red_cans := total * red / sum
  let white_cans := total * white / sum
  let blue_cans := total * blue / sum
  (red_cans, white_cans, blue_cans)

/-- Prove that for a 3:2:1 ratio and 60 total cans, we get 30 red, 20 white, and 10 blue cans. -/
theorem paint_distribution_321_60 :
  paint_distribution 3 2 1 60 = (30, 20, 10) := by
  sorry

end NUMINAMATH_CALUDE_paint_distribution_321_60_l909_90962


namespace NUMINAMATH_CALUDE_complement_of_A_l909_90930

-- Define the set A
def A : Set ℝ := {x | x^2 + 3*x ≥ 0} ∪ {x | 2*x > 1}

-- State the theorem
theorem complement_of_A : 
  {x : ℝ | x ∉ A} = {x : ℝ | -3 < x ∧ x < 0} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l909_90930


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l909_90941

theorem complex_fraction_sum (x y : ℂ) 
  (h : (x + y) / (x - y) - 3 * (x - y) / (x + y) = 2) :
  (x^6 + y^6) / (x^6 - y^6) + (x^6 - y^6) / (x^6 + y^6) = 8320 / 4095 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l909_90941


namespace NUMINAMATH_CALUDE_three_fifths_of_ten_times_seven_minus_three_l909_90946

theorem three_fifths_of_ten_times_seven_minus_three (x : ℚ) : x = 40.2 → x = (3 / 5) * ((10 * 7) - 3) := by
  sorry

end NUMINAMATH_CALUDE_three_fifths_of_ten_times_seven_minus_three_l909_90946


namespace NUMINAMATH_CALUDE_expansion_properties_l909_90976

theorem expansion_properties (x : ℝ) : 
  let expansion := (x + 1) * (x + 2)^4
  ∃ (a b c d e f : ℝ), 
    expansion = a*x^5 + b*x^4 + c*x^3 + 56*x^2 + d*x + e ∧
    a + b + c + 56 + d + e = 162 :=
by sorry

end NUMINAMATH_CALUDE_expansion_properties_l909_90976


namespace NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l909_90939

theorem rectangle_circle_area_ratio 
  (l w r : ℝ) 
  (h1 : 2 * l + 2 * w = 2 * Real.pi * r) 
  (h2 : l = 2 * w) : 
  (l * w) / (Real.pi * r^2) = 2 * Real.pi / 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l909_90939


namespace NUMINAMATH_CALUDE_eighteen_men_handshakes_l909_90909

/-- The maximum number of handshakes among n men without cyclic handshakes -/
def maxHandshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: The maximum number of handshakes among 18 men without cyclic handshakes is 153 -/
theorem eighteen_men_handshakes :
  maxHandshakes 18 = 153 := by
  sorry

end NUMINAMATH_CALUDE_eighteen_men_handshakes_l909_90909


namespace NUMINAMATH_CALUDE_number_exceeds_fraction_l909_90911

theorem number_exceeds_fraction (N : ℚ) (F : ℚ) : 
  N = 56 → N = F + 35 → F = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeds_fraction_l909_90911


namespace NUMINAMATH_CALUDE_fraction_simplification_l909_90933

theorem fraction_simplification (x y : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x - 1/y ≠ 0) :
  (y - 1/x) / (x - 1/y) = y/x :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l909_90933


namespace NUMINAMATH_CALUDE_gcd_2028_2295_l909_90908

theorem gcd_2028_2295 : Nat.gcd 2028 2295 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2028_2295_l909_90908


namespace NUMINAMATH_CALUDE_line_equations_correct_l909_90985

/-- Triangle ABC with vertices A(0,4), B(-2,6), and C(-8,0) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- Given triangle ABC, compute the equation of line AB -/
def lineAB (t : Triangle) : LineEquation :=
  { a := 1, b := 1, c := -4 }

/-- Given triangle ABC, compute the midpoint D of side AC -/
def midpointD (t : Triangle) : ℝ × ℝ :=
  ((-4 : ℝ), (2 : ℝ))

/-- Given triangle ABC, compute the equation of line BD where D is the midpoint of AC -/
def lineBD (t : Triangle) : LineEquation :=
  { a := 2, b := -1, c := 10 }

/-- Theorem stating that for the given triangle, the computed line equations are correct -/
theorem line_equations_correct (t : Triangle) 
    (h : t.A = (0, 4) ∧ t.B = (-2, 6) ∧ t.C = (-8, 0)) : 
  (lineAB t = { a := 1, b := 1, c := -4 }) ∧ 
  (lineBD t = { a := 2, b := -1, c := 10 }) := by
  sorry

end NUMINAMATH_CALUDE_line_equations_correct_l909_90985


namespace NUMINAMATH_CALUDE_smallest_slope_tangent_line_l909_90979

/-- The curve function -/
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x - 10

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x + 6

/-- Theorem: The equation of the tangent line with the smallest slope -/
theorem smallest_slope_tangent_line :
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, y = f x → (∀ m : ℝ, ∃ x₀ y₀ : ℝ, y₀ = f x₀ ∧ m = f' x₀ → m ≥ f' (-1)) ∧ 
    a*x + b*y + c = 0 ∧ 
    -14 = f (-1) ∧ 
    3 = f' (-1) ∧
    a = 3 ∧ b = -1 ∧ c = -11) :=
sorry

end NUMINAMATH_CALUDE_smallest_slope_tangent_line_l909_90979


namespace NUMINAMATH_CALUDE_perpendicular_lines_l909_90951

theorem perpendicular_lines (a : ℝ) : 
  (∀ x y : ℝ, x + 2*y + 3 = 0 ∧ 4*x - a*y + 5 = 0) →
  ((-(1:ℝ)/2) * (4/a) = -1) →
  a = 2 := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l909_90951


namespace NUMINAMATH_CALUDE_chocolate_gum_pricing_l909_90978

theorem chocolate_gum_pricing (c g : ℝ) 
  (h : (2 * c > 5 * g ∧ 3 * c ≤ 8 * g) ∨ (2 * c ≤ 5 * g ∧ 3 * c > 8 * g)) :
  7 * c < 19 * g := by
  sorry

end NUMINAMATH_CALUDE_chocolate_gum_pricing_l909_90978


namespace NUMINAMATH_CALUDE_max_increase_year_1998_l909_90986

def sales : Fin 11 → ℝ
  | 0 => 3.0
  | 1 => 4.5
  | 2 => 5.1
  | 3 => 7.0
  | 4 => 8.5
  | 5 => 9.7
  | 6 => 10.7
  | 7 => 12.0
  | 8 => 13.2
  | 9 => 13.7
  | 10 => 7.5

def year_increase (i : Fin 10) : ℝ :=
  sales (i.succ) - sales i

theorem max_increase_year_1998 :
  ∃ i : Fin 10, (i.val + 1995 = 1998) ∧
    ∀ j : Fin 10, year_increase j ≤ year_increase i :=
by sorry

end NUMINAMATH_CALUDE_max_increase_year_1998_l909_90986


namespace NUMINAMATH_CALUDE_probability_two_green_marbles_l909_90989

/-- The probability of drawing two green marbles without replacement from a jar containing 5 red, 3 green, and 7 white marbles is 1/35. -/
theorem probability_two_green_marbles (red green white : ℕ) 
  (h_red : red = 5) 
  (h_green : green = 3) 
  (h_white : white = 7) : 
  (green / (red + green + white)) * ((green - 1) / (red + green + white - 1)) = 1 / 35 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_green_marbles_l909_90989


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l909_90992

theorem sqrt_equation_solution (x : ℚ) : 
  (Real.sqrt (4 * x + 6) / Real.sqrt (8 * x + 6) = Real.sqrt 6 / 2) → x = -3/8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l909_90992


namespace NUMINAMATH_CALUDE_hyperbola_m_value_l909_90947

/-- Represents a hyperbola with parameter m -/
structure Hyperbola (m : ℝ) where
  eq : ∀ x y : ℝ, 3 * m * x^2 - m * y^2 = 3

/-- The distance from the center to a focus of the hyperbola -/
def focal_distance (h : Hyperbola m) : ℝ := 2

theorem hyperbola_m_value (h : Hyperbola m) 
  (focus : focal_distance h = 2) : m = -1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_m_value_l909_90947


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_parallel_planes_l909_90977

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (line_parallel : Line → Plane → Prop)
variable (line_perpendicular : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_from_parallel_planes 
  (α β : Plane) (l m : Line) :
  parallel α β → perpendicular l α → line_parallel m β → 
  line_perpendicular l m :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_parallel_planes_l909_90977


namespace NUMINAMATH_CALUDE_f_m_plus_one_positive_l909_90924

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + x + a

theorem f_m_plus_one_positive (a m : ℝ) (ha : a > 0) (hm : f a m < 0) : f a (m + 1) > 0 := by
  sorry

end NUMINAMATH_CALUDE_f_m_plus_one_positive_l909_90924


namespace NUMINAMATH_CALUDE_video_game_spending_l909_90990

/-- The total amount spent on video games is the sum of the costs of individual games -/
theorem video_game_spending (basketball_cost racing_cost : ℚ) :
  basketball_cost = 5.2 →
  racing_cost = 4.23 →
  basketball_cost + racing_cost = 9.43 := by
  sorry

end NUMINAMATH_CALUDE_video_game_spending_l909_90990


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l909_90929

theorem inequality_and_equality_condition (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
  (sum_eq_one : a + b + c + d = 1) :
  (b * c * d / (1 - a)^2) + (a * c * d / (1 - b)^2) + 
  (a * b * d / (1 - c)^2) + (a * b * c / (1 - d)^2) ≤ 1/9 ∧
  ((b * c * d / (1 - a)^2) + (a * c * d / (1 - b)^2) + 
   (a * b * d / (1 - c)^2) + (a * b * c / (1 - d)^2) = 1/9 ↔ 
   a = 1/4 ∧ b = 1/4 ∧ c = 1/4 ∧ d = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l909_90929


namespace NUMINAMATH_CALUDE_spoiled_fish_fraction_l909_90954

theorem spoiled_fish_fraction (initial_stock sold_fish new_stock final_stock : ℕ) : 
  initial_stock = 200 →
  sold_fish = 50 →
  new_stock = 200 →
  final_stock = 300 →
  (final_stock - new_stock) / (initial_stock - sold_fish) = 2/3 := by
sorry

end NUMINAMATH_CALUDE_spoiled_fish_fraction_l909_90954


namespace NUMINAMATH_CALUDE_stratified_sampling_male_athletes_l909_90958

theorem stratified_sampling_male_athletes :
  let total_athletes : ℕ := 28 + 21
  let male_athletes : ℕ := 28
  let sample_size : ℕ := 14
  let selected_male_athletes : ℕ := (male_athletes * sample_size) / total_athletes
  selected_male_athletes = 8 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_male_athletes_l909_90958


namespace NUMINAMATH_CALUDE_three_planes_divide_space_l909_90938

-- Define a plane in 3D space
def Plane : Type := ℝ × ℝ × ℝ → Prop

-- Define a function to check if three planes intersect pairwise
def intersect_pairwise (p1 p2 p3 : Plane) : Prop := sorry

-- Define a function to check if three lines are mutually parallel
def mutually_parallel_intersections (p1 p2 p3 : Plane) : Prop := sorry

-- Define a function to count the number of parts the space is divided into
def count_parts (p1 p2 p3 : Plane) : ℕ := sorry

-- Theorem statement
theorem three_planes_divide_space :
  ∀ (p1 p2 p3 : Plane),
    intersect_pairwise p1 p2 p3 →
    mutually_parallel_intersections p1 p2 p3 →
    count_parts p1 p2 p3 = 7 :=
by sorry

end NUMINAMATH_CALUDE_three_planes_divide_space_l909_90938


namespace NUMINAMATH_CALUDE_soda_price_ratio_l909_90997

/-- The ratio of unit prices between two soda brands -/
theorem soda_price_ratio (v : ℝ) (p : ℝ) (h1 : v > 0) (h2 : p > 0) : 
  (0.85 * p) / (1.25 * v) / (p / v) = 17 / 25 := by
  sorry

end NUMINAMATH_CALUDE_soda_price_ratio_l909_90997


namespace NUMINAMATH_CALUDE_complex_absolute_value_l909_90904

theorem complex_absolute_value (i : ℂ) (z : ℂ) : 
  i^2 = -1 → z = (2*i)/(1+i) → Complex.abs (z - 2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l909_90904


namespace NUMINAMATH_CALUDE_tan_sum_product_equals_one_l909_90907

theorem tan_sum_product_equals_one :
  Real.tan (22 * π / 180) + Real.tan (23 * π / 180) + Real.tan (22 * π / 180) * Real.tan (23 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_product_equals_one_l909_90907


namespace NUMINAMATH_CALUDE_mixed_number_comparison_l909_90995

theorem mixed_number_comparison : (-2 - 1/3 : ℚ) < -2.3 := by
  sorry

end NUMINAMATH_CALUDE_mixed_number_comparison_l909_90995


namespace NUMINAMATH_CALUDE_sine_graph_shift_l909_90953

theorem sine_graph_shift (x : ℝ) :
  Real.sin (2 * (x + π / 4) + π / 6) = Real.sin (2 * x + 2 * π / 3) := by
  sorry

#check sine_graph_shift

end NUMINAMATH_CALUDE_sine_graph_shift_l909_90953


namespace NUMINAMATH_CALUDE_max_slope_no_lattice_points_l909_90944

theorem max_slope_no_lattice_points :
  ∃ (a : ℚ), a = 17/51 ∧
  (∀ (m : ℚ) (x y : ℤ), 1/3 < m → m < a → 1 ≤ x → x ≤ 50 →
    y = m * x + 3 → ¬(∃ (x' y' : ℤ), x' = x ∧ y' = y)) ∧
  (∀ (a' : ℚ), a < a' →
    ∃ (m : ℚ) (x y : ℤ), 1/3 < m → m < a' → 1 ≤ x → x ≤ 50 →
      y = m * x + 3 ∧ (∃ (x' y' : ℤ), x' = x ∧ y' = y)) :=
by sorry

end NUMINAMATH_CALUDE_max_slope_no_lattice_points_l909_90944


namespace NUMINAMATH_CALUDE_sample_size_theorem_l909_90974

/-- Represents a population of students -/
structure Population where
  size : Nat

/-- Represents a sample of students -/
structure Sample where
  size : Nat
  population : Population

/-- Theorem: Given a population of 5000 students and a selection of 250 students,
    the 250 students form a sample of the population with a sample size of 250. -/
theorem sample_size_theorem (pop : Population) (sam : Sample) 
    (h1 : pop.size = 5000) (h2 : sam.size = 250) (h3 : sam.population = pop) : 
    sam.size = 250 ∧ sam.population = pop := by
  sorry

#check sample_size_theorem

end NUMINAMATH_CALUDE_sample_size_theorem_l909_90974


namespace NUMINAMATH_CALUDE_pool_filling_solution_l909_90981

/-- Represents the time taken to fill a pool given two pumps with specific properties -/
def pool_filling_time (pool_volume : ℝ) : Prop :=
  ∃ (t1 t2 : ℝ),
    t1 > 0 ∧ t2 > 0 ∧
    -- First pump fills 8 hours faster than second pump
    t2 - t1 = 8 ∧
    -- Second pump initially runs for twice the time of both pumps together
    2 * (1 / (1/t1 + 1/t2)) * (1/t2) +
    -- Then both pumps run for 1.5 hours
    1.5 * (1/t1 + 1/t2) = 1 ∧
    -- Times for each pump to fill separately
    t1 = 4 ∧ t2 = 12

/-- Theorem stating the existence of a solution for the pool filling problem -/
theorem pool_filling_solution (pool_volume : ℝ) (h : pool_volume > 0) :
  pool_filling_time pool_volume :=
sorry

end NUMINAMATH_CALUDE_pool_filling_solution_l909_90981


namespace NUMINAMATH_CALUDE_angies_age_l909_90945

theorem angies_age :
  ∀ (A : ℕ), (2 * A + 4 = 20) → A = 8 := by
  sorry

end NUMINAMATH_CALUDE_angies_age_l909_90945


namespace NUMINAMATH_CALUDE_rodney_ian_money_difference_l909_90961

def rodney_ian_difference (jessica_money : ℕ) (jessica_rodney_diff : ℕ) : ℕ :=
  let rodney_money := jessica_money - jessica_rodney_diff
  let ian_money := jessica_money / 2
  rodney_money - ian_money

theorem rodney_ian_money_difference :
  rodney_ian_difference 100 15 = 35 :=
by sorry

end NUMINAMATH_CALUDE_rodney_ian_money_difference_l909_90961


namespace NUMINAMATH_CALUDE_disease_cases_1975_l909_90927

/-- Calculates the number of disease cases in a given year, assuming a linear decrease -/
def diseaseCases (initialYear finalYear : ℕ) (initialCases finalCases : ℕ) (targetYear : ℕ) : ℕ :=
  let totalYears := finalYear - initialYear
  let totalDecrease := initialCases - finalCases
  let annualDecrease := totalDecrease / totalYears
  let yearsPassed := targetYear - initialYear
  initialCases - (annualDecrease * yearsPassed)

theorem disease_cases_1975 :
  diseaseCases 1950 2000 500000 1000 1975 = 250500 := by
  sorry

end NUMINAMATH_CALUDE_disease_cases_1975_l909_90927


namespace NUMINAMATH_CALUDE_cake_problem_l909_90936

theorem cake_problem (cube_edge : ℝ) (M : ℝ × ℝ) (N : ℝ × ℝ) :
  cube_edge = 2 →
  M = (2, 1) →
  N = (4/5, 2/5) →
  let volume := cube_edge * (1/2 * N.1 * N.2)
  let icing_area := (1/2 * N.1 * N.2) + (cube_edge * cube_edge)
  volume + icing_area = 32/5 := by
sorry

end NUMINAMATH_CALUDE_cake_problem_l909_90936


namespace NUMINAMATH_CALUDE_T_is_three_rays_with_common_endpoint_l909_90919

-- Define the set T
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (5 = x + 3 ∧ y - 2 < 5) ∨
               (5 = y - 2 ∧ x + 3 < 5) ∨
               (x + 3 = y - 2 ∧ 5 < x + 3)}

-- Define a ray
def Ray (start : ℝ × ℝ) (dir : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, t ≥ 0 ∧ p = (start.1 + t * dir.1, start.2 + t * dir.2)}

-- Theorem statement
theorem T_is_three_rays_with_common_endpoint :
  ∃ (start : ℝ × ℝ) (dir1 dir2 dir3 : ℝ × ℝ),
    T = Ray start dir1 ∪ Ray start dir2 ∪ Ray start dir3 ∧
    dir1 ≠ dir2 ∧ dir1 ≠ dir3 ∧ dir2 ≠ dir3 :=
  sorry

end NUMINAMATH_CALUDE_T_is_three_rays_with_common_endpoint_l909_90919


namespace NUMINAMATH_CALUDE_plywood_cut_theorem_l909_90957

theorem plywood_cut_theorem :
  ∃ (a b c d : Set (ℝ × ℝ)),
    -- The original square has area 625 cm²
    (∀ (x y : ℝ), (x, y) ∈ a ∪ b ∪ c ∪ d → 0 ≤ x ∧ x ≤ 25 ∧ 0 ≤ y ∧ y ≤ 25) ∧
    -- The four parts are disjoint
    (a ∩ b = ∅ ∧ a ∩ c = ∅ ∧ a ∩ d = ∅ ∧ b ∩ c = ∅ ∧ b ∩ d = ∅ ∧ c ∩ d = ∅) ∧
    -- The four parts cover the entire original square
    (∀ (x y : ℝ), 0 ≤ x ∧ x ≤ 25 ∧ 0 ≤ y ∧ y ≤ 25 → (x, y) ∈ a ∪ b ∪ c ∪ d) ∧
    -- The parts can be rearranged into two squares
    (∃ (s₁ s₂ : Set (ℝ × ℝ)),
      -- First square has side length 24 cm
      (∀ (x y : ℝ), (x, y) ∈ s₁ → 0 ≤ x ∧ x ≤ 24 ∧ 0 ≤ y ∧ y ≤ 24) ∧
      -- Second square has side length 7 cm
      (∀ (x y : ℝ), (x, y) ∈ s₂ → 0 ≤ x ∧ x ≤ 7 ∧ 0 ≤ y ∧ y ≤ 7) ∧
      -- The rearranged squares cover the same area as the original parts
      (∀ (x y : ℝ), (x, y) ∈ s₁ ∪ s₂ ↔ (x, y) ∈ a ∪ b ∪ c ∪ d)) :=
by
  sorry


end NUMINAMATH_CALUDE_plywood_cut_theorem_l909_90957


namespace NUMINAMATH_CALUDE_bruce_savings_l909_90950

-- Define the given amounts and rates
def aunt_money : ℝ := 87.32
def grandfather_money : ℝ := 152.68
def savings_rate : ℝ := 0.35
def interest_rate : ℝ := 0.025

-- Define the function to calculate the amount after one year
def amount_after_one_year (aunt_money grandfather_money savings_rate interest_rate : ℝ) : ℝ :=
  let total_money := aunt_money + grandfather_money
  let saved_amount := total_money * savings_rate
  let interest := saved_amount * interest_rate
  saved_amount + interest

-- Theorem statement
theorem bruce_savings : 
  amount_after_one_year aunt_money grandfather_money savings_rate interest_rate = 86.10 := by
  sorry

end NUMINAMATH_CALUDE_bruce_savings_l909_90950


namespace NUMINAMATH_CALUDE_line_through_P_perpendicular_to_given_line_l909_90903

-- Define the point P
def P : ℝ × ℝ := (4, -1)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := 3 * x - 4 * y + 6 = 0

-- Define the equation of the line we're looking for
def target_line (x y : ℝ) : Prop := 4 * x + 3 * y - 13 = 0

-- Theorem statement
theorem line_through_P_perpendicular_to_given_line :
  ∃ (m b : ℝ), 
    (∀ (x y : ℝ), m * x + y + b = 0 ↔ target_line x y) ∧
    (m * P.1 + P.2 + b = 0) ∧
    (m * 4 + 3 = 0) :=
sorry

end NUMINAMATH_CALUDE_line_through_P_perpendicular_to_given_line_l909_90903


namespace NUMINAMATH_CALUDE_marble_combinations_l909_90934

-- Define the number of marbles
def total_marbles : ℕ := 9

-- Define the number of marbles to choose
def chosen_marbles : ℕ := 4

-- Define the combination function
def combination (n k : ℕ) : ℕ := Nat.choose n k

-- Theorem statement
theorem marble_combinations : combination total_marbles chosen_marbles = 126 := by
  sorry

end NUMINAMATH_CALUDE_marble_combinations_l909_90934


namespace NUMINAMATH_CALUDE_peach_problem_l909_90993

theorem peach_problem (martine benjy gabrielle : ℕ) : 
  martine = 2 * benjy + 6 →
  benjy = gabrielle / 3 →
  martine = 16 →
  gabrielle = 15 := by
sorry

end NUMINAMATH_CALUDE_peach_problem_l909_90993


namespace NUMINAMATH_CALUDE_circle_tangent_line_segment_l909_90964

/-- Given two circles in a plane with radii r₁ and r₂, centered at O₁ and O₂ respectively,
    touching a line at points M₁ and M₂, and lying on the same side of the line,
    if the ratio of M₁M₂ to O₁O₂ is k, then M₁M₂ can be calculated. -/
theorem circle_tangent_line_segment (r₁ r₂ : ℝ) (k : ℝ) (h₁ : r₁ = 12) (h₂ : r₂ = 7) (h₃ : k = 2 * Real.sqrt 5 / 5) :
  let M₁M₂ := r₁ - r₂
  M₁M₂ * (Real.sqrt (1 - k^2) / k) = 10 :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_line_segment_l909_90964


namespace NUMINAMATH_CALUDE_other_root_is_one_l909_90973

/-- The integer part of √6 -/
def int_part_sqrt_6 : ℤ := 2

/-- The equation x^2 - 3x - m = 0 -/
def equation (x m : ℝ) : Prop := x^2 - 3*x - m = 0

/-- One root of the equation is the integer part of √6 -/
axiom root_is_int_part_sqrt_6 (m : ℝ) : equation (int_part_sqrt_6 : ℝ) m

theorem other_root_is_one (m : ℝ) : 
  ∃ (x : ℝ), x ≠ int_part_sqrt_6 ∧ equation x m ∧ x = 1 :=
sorry

end NUMINAMATH_CALUDE_other_root_is_one_l909_90973


namespace NUMINAMATH_CALUDE_circle_area_ratio_l909_90970

theorem circle_area_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) : 
  (30 / 360 : ℝ) * (2 * π * r₁) = (24 / 360 : ℝ) * (2 * π * r₂) →
  (π * r₁^2) / (π * r₂^2) = 16 / 25 := by
sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l909_90970
