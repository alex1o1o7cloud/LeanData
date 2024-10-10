import Mathlib

namespace cost_to_fly_D_to_E_l2831_283191

/-- Represents a city in the triangle --/
inductive City
| D
| E
| F

/-- Calculates the cost of flying between two cities --/
def flyCost (distance : ℝ) : ℝ :=
  120 + 0.12 * distance

/-- The triangle formed by the cities --/
structure Triangle where
  DE : ℝ
  DF : ℝ
  isRightAngled : True

/-- The problem setup --/
structure TripProblem where
  cities : Triangle
  flyFromDToE : True

theorem cost_to_fly_D_to_E (problem : TripProblem) : 
  flyCost problem.cities.DE = 660 :=
sorry

end cost_to_fly_D_to_E_l2831_283191


namespace fixed_amount_more_economical_l2831_283135

theorem fixed_amount_more_economical (p₁ p₂ : ℝ) (h₁ : p₁ > 0) (h₂ : p₂ > 0) :
  2 / (1 / p₁ + 1 / p₂) ≤ (p₁ + p₂) / 2 := by
  sorry

#check fixed_amount_more_economical

end fixed_amount_more_economical_l2831_283135


namespace zhaoqing_population_l2831_283185

theorem zhaoqing_population (total_population : ℝ) 
  (h1 : 3.06 = 0.8 * total_population - 0.18) 
  (h2 : 3.06 = agricultural_population) : 
  total_population = 4.05 := by
sorry

end zhaoqing_population_l2831_283185


namespace constant_term_proof_l2831_283113

/-- The constant term in the expansion of (√x + 1/(3x))^10 -/
def constant_term : ℕ := 210

/-- The index of the term with the maximum coefficient -/
def max_coeff_index : ℕ := 6

theorem constant_term_proof (h : max_coeff_index = 6) : constant_term = 210 := by
  sorry

end constant_term_proof_l2831_283113


namespace functional_equation_implies_linear_scaling_l2831_283121

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x^3 + y^3) = (x + y) * (f x^2 - f x * f y + f y^2)

/-- The main theorem to be proved -/
theorem functional_equation_implies_linear_scaling
  (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∀ x, f (1996 * x) = 1996 * f x := by
  sorry

end functional_equation_implies_linear_scaling_l2831_283121


namespace remainder_theorem_l2831_283195

theorem remainder_theorem : 7 * 10^20 + 1^20 ≡ 8 [ZMOD 9] := by sorry

end remainder_theorem_l2831_283195


namespace perfect_squares_divisibility_l2831_283163

theorem perfect_squares_divisibility (a b : ℕ+) 
  (h : ∃ S : Set (ℕ+ × ℕ+), Set.Infinite S ∧ 
    ∀ (m n : ℕ+), (m, n) ∈ S → 
      ∃ (k l : ℕ+), (m : ℕ)^2 + (a : ℕ) * (n : ℕ) + (b : ℕ) = (k : ℕ)^2 ∧
                    (n : ℕ)^2 + (a : ℕ) * (m : ℕ) + (b : ℕ) = (l : ℕ)^2) : 
  (a : ℕ) ∣ 2 * (b : ℕ) := by
  sorry

end perfect_squares_divisibility_l2831_283163


namespace quadratic_transformation_l2831_283188

/-- Transformation of a quadratic equation under a linear substitution -/
theorem quadratic_transformation (A B C D E F α β γ β' γ' : ℝ) :
  let Δ := A * C - B^2
  let x := λ x' y' : ℝ => α * x' + β * y' + γ
  let y := λ x' y' : ℝ => x' + β' * y' + γ'
  let original_eq := λ x y : ℝ => A * x^2 + 2 * B * x * y + C * y^2 + 2 * D * x + 2 * E * y + F
  ∃ a b : ℝ, 
    (Δ > 0 → ∀ x' y' : ℝ, original_eq (x x' y') (y x' y') = 0 ↔ x'^2 / a^2 + y'^2 / b^2 = 1) ∧
    (Δ < 0 → ∀ x' y' : ℝ, original_eq (x x' y') (y x' y') = 0 ↔ x'^2 / a^2 - y'^2 / b^2 = 1) :=
by sorry

end quadratic_transformation_l2831_283188


namespace least_positive_integer_congruence_l2831_283136

theorem least_positive_integer_congruence :
  ∃ (x : ℕ), x > 0 ∧ (x + 4609 : ℤ) ≡ 2104 [ZMOD 12] ∧
  ∀ (y : ℕ), y > 0 → (y + 4609 : ℤ) ≡ 2104 [ZMOD 12] → x ≤ y :=
by sorry

end least_positive_integer_congruence_l2831_283136


namespace bus_miss_time_l2831_283117

theorem bus_miss_time (usual_time : ℝ) (h : usual_time = 12) :
  let slower_time := (5 / 4) * usual_time
  slower_time - usual_time = 3 :=
by sorry

end bus_miss_time_l2831_283117


namespace intersection_of_A_and_B_l2831_283142

def A : Set ℕ := {0, 1, 2, 3, 4, 5}
def B : Set ℕ := {x | x^2 < 10}

theorem intersection_of_A_and_B : A ∩ B = {0, 1, 2, 3} := by sorry

end intersection_of_A_and_B_l2831_283142


namespace tv_price_reduction_l2831_283131

theorem tv_price_reduction (x : ℝ) : 
  (1 - x / 100) * 1.80 = 1.44000000000000014 → x = 20 := by
  sorry

end tv_price_reduction_l2831_283131


namespace line_points_k_value_l2831_283189

theorem line_points_k_value (m n k : ℝ) : 
  (m = 2*n + 5) →                   -- First point (m, n) satisfies the line equation
  (m + 5 = 2*(n + k) + 5) →         -- Second point (m + 5, n + k) satisfies the line equation
  k = 5/2 := by                     -- Conclusion: k = 2.5
sorry


end line_points_k_value_l2831_283189


namespace player_B_wins_l2831_283192

/-- Represents the state of the pizza game -/
structure GameState :=
  (pizzeria1 : Nat)
  (pizzeria2 : Nat)

/-- Represents a player's move -/
inductive Move
  | EatFromOne (pizzeria : Nat) (amount : Nat)
  | EatFromBoth

/-- Defines the rules of the game -/
def isValidMove (state : GameState) (move : Move) : Prop :=
  match move with
  | Move.EatFromOne 1 amount => amount > 0 ∧ amount ≤ state.pizzeria1
  | Move.EatFromOne 2 amount => amount > 0 ∧ amount ≤ state.pizzeria2
  | Move.EatFromBoth => state.pizzeria1 > 0 ∧ state.pizzeria2 > 0
  | _ => False

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.EatFromOne 1 amount => ⟨state.pizzeria1 - amount, state.pizzeria2⟩
  | Move.EatFromOne 2 amount => ⟨state.pizzeria1, state.pizzeria2 - amount⟩
  | Move.EatFromBoth => ⟨state.pizzeria1 - 1, state.pizzeria2 - 1⟩
  | _ => state

/-- Defines a winning strategy for player B -/
def hasWinningStrategy (player : Nat) : Prop :=
  ∀ (state : GameState),
    (state.pizzeria1 = 2010 ∧ state.pizzeria2 = 2010) →
    ∃ (strategy : GameState → Move),
      (∀ (s : GameState), isValidMove s (strategy s)) ∧
      (player = 2 → ∃ (n : Nat), state.pizzeria1 + state.pizzeria2 = n ∧ n % 2 = 1)

/-- The main theorem: Player B (second player) has a winning strategy -/
theorem player_B_wins : hasWinningStrategy 2 := by
  sorry

end player_B_wins_l2831_283192


namespace notebook_increase_correct_l2831_283118

/-- Calculates the increase in Jimin's notebook count -/
def notebook_increase (initial : ℕ) (father_bought : ℕ) (mother_bought : ℕ) : ℕ :=
  father_bought + mother_bought

theorem notebook_increase_correct (initial : ℕ) (father_bought : ℕ) (mother_bought : ℕ) :
  notebook_increase initial father_bought mother_bought = father_bought + mother_bought :=
by sorry

end notebook_increase_correct_l2831_283118


namespace double_price_increase_l2831_283196

theorem double_price_increase (original_price : ℝ) (h : original_price > 0) :
  (original_price * (1 + 0.06) * (1 + 0.06)) = (original_price * (1 + 0.1236)) := by
  sorry

end double_price_increase_l2831_283196


namespace both_not_land_l2831_283133

-- Define the propositions
variable (p q : Prop)

-- p represents "A lands within the designated area"
-- q represents "B lands within the designated area"

-- Theorem: "Both trainees did not land within the designated area" 
-- is equivalent to (¬p) ∧ (¬q)
theorem both_not_land (p q : Prop) : 
  (¬p ∧ ¬q) ↔ ¬(p ∨ q) :=
sorry

end both_not_land_l2831_283133


namespace like_terms_imply_sum_of_exponents_l2831_283190

/-- Two terms are considered like terms if their variables and corresponding exponents match -/
def are_like_terms (term1 term2 : ℕ → ℕ → ℚ) : Prop :=
  ∀ (a b : ℕ), ∃ (c : ℚ), term1 a b = c * term2 a b ∨ term2 a b = c * term1 a b

/-- The first term in our problem -/
def term1 (m : ℕ) (a b : ℕ) : ℚ := 2 * (a ^ m) * (b ^ 3)

/-- The second term in our problem -/
def term2 (n : ℕ) (a b : ℕ) : ℚ := -3 * a * (b ^ n)

theorem like_terms_imply_sum_of_exponents (m n : ℕ) :
  are_like_terms (term1 m) (term2 n) → m + n = 4 :=
by
  sorry

end like_terms_imply_sum_of_exponents_l2831_283190


namespace sum_of_powers_l2831_283172

theorem sum_of_powers : -1^2008 + (-1)^2009 + 1^2010 - 1^2011 = -2 := by
  sorry

end sum_of_powers_l2831_283172


namespace quotient_divisible_by_five_l2831_283144

theorem quotient_divisible_by_five : ∃ k : ℤ, 4^1993 + 6^1993 = 5 * k := by
  sorry

end quotient_divisible_by_five_l2831_283144


namespace not_right_triangle_not_triangle_l2831_283134

theorem not_right_triangle (a b c : ℝ) (ha : a = 1) (hb : b = 1) (hc : c = 2) :
  ¬(a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2) :=
by sorry

theorem not_triangle (a b c : ℝ) (ha : a = 1) (hb : b = 1) (hc : c = 2) :
  ¬(a + b > c ∧ b + c > a ∧ c + a > b) :=
by sorry

end not_right_triangle_not_triangle_l2831_283134


namespace candles_remaining_l2831_283107

/-- Calculates the number of candles remaining after three people use them according to specific rules. -/
theorem candles_remaining (total : ℕ) (alyssa_fraction : ℚ) (chelsea_fraction : ℚ) (bianca_fraction : ℚ) : 
  total = 60 ∧ 
  alyssa_fraction = 1/2 ∧ 
  chelsea_fraction = 7/10 ∧ 
  bianca_fraction = 4/5 →
  ↑total - (alyssa_fraction * ↑total + 
    chelsea_fraction * (↑total - alyssa_fraction * ↑total) + 
    ⌊bianca_fraction * (↑total - alyssa_fraction * ↑total - chelsea_fraction * (↑total - alyssa_fraction * ↑total))⌋) = 2 := by
  sorry

#check candles_remaining

end candles_remaining_l2831_283107


namespace correct_calculation_l2831_283116

theorem correct_calculation (x : ℤ) (h : x - 6 = 51) : 6 * x = 342 := by
  sorry

end correct_calculation_l2831_283116


namespace shaded_area_circles_l2831_283151

theorem shaded_area_circles (R : ℝ) (h : R = 10) : 
  let large_circle_area := π * R^2
  let small_circle_radius := R / 2
  let small_circle_area := π * small_circle_radius^2
  let shaded_area := large_circle_area - 2 * small_circle_area
  shaded_area = 50 * π := by sorry

end shaded_area_circles_l2831_283151


namespace total_markers_l2831_283186

theorem total_markers :
  let red_markers : ℕ := 41
  let blue_markers : ℕ := 64
  let green_markers : ℕ := 35
  let black_markers : ℕ := 78
  let yellow_markers : ℕ := 102
  red_markers + blue_markers + green_markers + black_markers + yellow_markers = 320 :=
by
  sorry

end total_markers_l2831_283186


namespace chess_game_draw_probability_l2831_283106

theorem chess_game_draw_probability 
  (p_jian_win : ℝ) 
  (p_gu_not_win : ℝ) 
  (h1 : p_jian_win = 0.4) 
  (h2 : p_gu_not_win = 0.6) : 
  p_gu_not_win - p_jian_win = 0.2 := by
sorry

end chess_game_draw_probability_l2831_283106


namespace rectangular_box_area_volume_relation_l2831_283166

/-- A rectangular box with dimensions x, y, and z -/
structure RectangularBox where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Properties of the rectangular box -/
def RectangularBox.properties (box : RectangularBox) : Prop :=
  let top_area := box.x * box.y
  let side_area := box.y * box.z
  let volume := box.x * box.y * box.z
  (side_area * volume ^ 2 = box.z ^ 3 * volume)

/-- Theorem: The product of the side area and square of volume equals z³V -/
theorem rectangular_box_area_volume_relation (box : RectangularBox) :
  box.properties :=
by
  sorry

#check rectangular_box_area_volume_relation

end rectangular_box_area_volume_relation_l2831_283166


namespace probability_two_red_marbles_l2831_283115

/-- The probability of drawing two red marbles without replacement from a jar containing
    2 red marbles, 3 green marbles, and 10 white marbles is 1/105. -/
theorem probability_two_red_marbles :
  let red_marbles : ℕ := 2
  let green_marbles : ℕ := 3
  let white_marbles : ℕ := 10
  let total_marbles : ℕ := red_marbles + green_marbles + white_marbles
  (red_marbles : ℚ) / total_marbles * (red_marbles - 1) / (total_marbles - 1) = 1 / 105 :=
by sorry

end probability_two_red_marbles_l2831_283115


namespace quadratic_root_implies_m_l2831_283104

theorem quadratic_root_implies_m (m : ℝ) : 
  (3^2 : ℝ) - m*3 - 6 = 0 → m = 1 := by
sorry

end quadratic_root_implies_m_l2831_283104


namespace tan_neg_585_deg_l2831_283199

theorem tan_neg_585_deg : Real.tan (-585 * π / 180) = -1 := by sorry

end tan_neg_585_deg_l2831_283199


namespace ohara_triple_49_64_l2831_283108

/-- Definition of an O'Hara triple -/
def is_ohara_triple (a b y : ℕ) : Prop :=
  Real.sqrt a + Real.sqrt b = y

/-- Theorem: If (49, 64, y) is an O'Hara triple, then y = 15 -/
theorem ohara_triple_49_64 (y : ℕ) :
  is_ohara_triple 49 64 y → y = 15 := by
  sorry

end ohara_triple_49_64_l2831_283108


namespace sequence_product_l2831_283155

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

/-- The main theorem -/
theorem sequence_product (a b : ℕ → ℝ) :
  (∀ n, a n ≠ 0) →
  arithmetic_sequence a →
  2 * a 3 - (a 7)^2 + 2 * a 11 = 0 →
  geometric_sequence b →
  b 7 = a 7 →
  b 6 * b 8 = 16 := by
  sorry

end sequence_product_l2831_283155


namespace smallest_prime_after_six_nonprimes_l2831_283129

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def consecutive_nonprimes (start : ℕ) (count : ℕ) : Prop :=
  ∀ k : ℕ, k ≥ start ∧ k < start + count → ¬(is_prime k)

theorem smallest_prime_after_six_nonprimes :
  ∃ n : ℕ, 
    (consecutive_nonprimes (n - 6) 6) ∧ 
    (is_prime n) ∧ 
    (∀ m : ℕ, m < n → ¬(consecutive_nonprimes (m - 6) 6 ∧ is_prime m)) ∧
    n = 53 :=
sorry

end smallest_prime_after_six_nonprimes_l2831_283129


namespace sin_pi_half_equals_one_l2831_283123

theorem sin_pi_half_equals_one : 
  let f : ℝ → ℝ := fun x ↦ Real.sin (x / 2 + π / 4)
  f (π / 2) = 1 := by
sorry

end sin_pi_half_equals_one_l2831_283123


namespace pi_minus_three_zero_plus_half_inverse_equals_three_l2831_283125

theorem pi_minus_three_zero_plus_half_inverse_equals_three :
  (Real.pi - 3) ^ (0 : ℕ) + (1 / 2) ^ (-1 : ℤ) = 3 := by
  sorry

end pi_minus_three_zero_plus_half_inverse_equals_three_l2831_283125


namespace solve_for_a_l2831_283197

theorem solve_for_a : ∃ a : ℝ, (3 + 2 * a = -1) ∧ (a = -2) := by sorry

end solve_for_a_l2831_283197


namespace total_jumps_calculation_total_jumps_is_4411_l2831_283178

/-- Calculate the total number of jumps made by Rupert and Ronald throughout the week. -/
theorem total_jumps_calculation : ℕ := by
  -- Define the number of jumps for Ronald on Monday
  let ronald_monday : ℕ := 157

  -- Define Rupert's jumps on Monday relative to Ronald's
  let rupert_monday : ℕ := ronald_monday + 86

  -- Define Ronald's jumps on Tuesday
  let ronald_tuesday : ℕ := 193

  -- Define Rupert's jumps on Tuesday
  let rupert_tuesday : ℕ := rupert_monday - 35

  -- Define the constant decrease rate from Thursday to Sunday
  let daily_decrease : ℕ := 20

  -- Calculate total jumps
  let total_jumps : ℕ := 
    -- Monday
    ronald_monday + rupert_monday +
    -- Tuesday
    ronald_tuesday + rupert_tuesday +
    -- Wednesday (doubled from Tuesday)
    2 * ronald_tuesday + 2 * rupert_tuesday +
    -- Thursday to Sunday (4 days with constant decrease)
    (2 * ronald_tuesday - daily_decrease) + (2 * rupert_tuesday - daily_decrease) +
    (2 * ronald_tuesday - 2 * daily_decrease) + (2 * rupert_tuesday - 2 * daily_decrease) +
    (2 * ronald_tuesday - 3 * daily_decrease) + (2 * rupert_tuesday - 3 * daily_decrease) +
    (2 * ronald_tuesday - 4 * daily_decrease) + (2 * rupert_tuesday - 4 * daily_decrease)

  exact total_jumps

/-- Prove that the total number of jumps is 4411 -/
theorem total_jumps_is_4411 : total_jumps_calculation = 4411 := by
  sorry

end total_jumps_calculation_total_jumps_is_4411_l2831_283178


namespace chord_length_concentric_circles_l2831_283150

theorem chord_length_concentric_circles (R r : ℝ) (h : R^2 - r^2 = 15) :
  ∃ c : ℝ, c = 2 * Real.sqrt 15 ∧ c^2 / 4 + r^2 = R^2 := by
  sorry

end chord_length_concentric_circles_l2831_283150


namespace interior_angles_sum_increase_l2831_283119

/-- The sum of interior angles of a convex polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- Theorem: If the sum of interior angles of a convex polygon with n sides is 2340°,
    then the sum of interior angles of a convex polygon with (n + 4) sides is 3060°. -/
theorem interior_angles_sum_increase (n : ℕ) :
  sum_interior_angles n = 2340 → sum_interior_angles (n + 4) = 3060 := by
  sorry

end interior_angles_sum_increase_l2831_283119


namespace team_total_score_l2831_283183

/-- Represents a basketball player with their score -/
structure Player where
  name : String
  score : ℕ

/-- The school basketball team -/
def team : List Player := [
  { name := "Daniel", score := 7 },
  { name := "Ramon", score := 8 },
  { name := "Ian", score := 2 },
  { name := "Bernardo", score := 11 },
  { name := "Tiago", score := 6 },
  { name := "Pedro", score := 12 },
  { name := "Ed", score := 1 },
  { name := "André", score := 7 }
]

/-- The total score of the team is the sum of individual player scores -/
def totalScore (team : List Player) : ℕ :=
  team.map (·.score) |>.sum

/-- Theorem: The total score of the team is 54 -/
theorem team_total_score : totalScore team = 54 := by
  sorry

end team_total_score_l2831_283183


namespace crabs_count_proof_l2831_283132

/-- The number of crabs on the first day -/
def crabs_day1 : ℕ := 72

/-- The number of oysters on the first day -/
def oysters_day1 : ℕ := 50

/-- The total count of oysters and crabs over two days -/
def total_count : ℕ := 195

theorem crabs_count_proof :
  crabs_day1 = 72 ∧
  oysters_day1 = 50 ∧
  (oysters_day1 + crabs_day1 + oysters_day1 / 2 + crabs_day1 * 2 / 3 = total_count) :=
sorry

end crabs_count_proof_l2831_283132


namespace existence_of_six_numbers_l2831_283179

theorem existence_of_six_numbers : ∃ (a b c d e f : ℕ),
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧
  (a + b + c + d + e + f : ℚ) / ((1 : ℚ)/a + 1/b + 1/c + 1/d + 1/e + 1/f) = 2012 :=
sorry

end existence_of_six_numbers_l2831_283179


namespace quadratic_prime_roots_l2831_283105

theorem quadratic_prime_roots (k : ℕ) : 
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ 
   p + q = 99 ∧ p * q = k ∧
   ∀ x : ℝ, x^2 - 99*x + k = 0 ↔ (x = p ∨ x = q)) →
  k = 194 :=
sorry

end quadratic_prime_roots_l2831_283105


namespace sum_of_coefficients_cubic_expansion_l2831_283184

theorem sum_of_coefficients_cubic_expansion :
  ∃ (a b c d e : ℝ), 
    (∀ x, 27 * x^3 + 64 = (a*x + b) * (c*x^2 + d*x + e)) ∧
    (a + b + c + d + e = 20) := by
  sorry

end sum_of_coefficients_cubic_expansion_l2831_283184


namespace factors_of_60_l2831_283159

-- Define the number of positive factors function
def num_positive_factors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

-- State the theorem
theorem factors_of_60 : num_positive_factors 60 = 12 := by
  sorry

end factors_of_60_l2831_283159


namespace simplify_radical_expression_l2831_283171

theorem simplify_radical_expression :
  Real.sqrt (13 + Real.sqrt 48) - Real.sqrt (5 - (2 * Real.sqrt 3 + 1)) + 2 * Real.sqrt (3 + (Real.sqrt 3 - 1)) = Real.sqrt 6 + Real.sqrt 2 := by
  sorry

end simplify_radical_expression_l2831_283171


namespace polyhedron_formula_l2831_283154

/-- Represents a convex polyhedron with specific face configuration -/
structure Polyhedron where
  faces : ℕ
  triangles : ℕ
  pentagons : ℕ
  hexagons : ℕ
  T : ℕ
  P : ℕ
  H : ℕ
  faces_sum : faces = triangles + pentagons + hexagons
  faces_types : faces = 32 ∧ triangles = 10 ∧ pentagons = 8 ∧ hexagons = 14

/-- Calculates the number of edges in the polyhedron -/
def edges (poly : Polyhedron) : ℕ :=
  (3 * poly.triangles + 5 * poly.pentagons + 6 * poly.hexagons) / 2

/-- Calculates the number of vertices in the polyhedron using Euler's formula -/
def vertices (poly : Polyhedron) : ℕ :=
  edges poly - poly.faces + 2

/-- Theorem stating that for the given polyhedron, 100P + 10T + V = 249 -/
theorem polyhedron_formula (poly : Polyhedron) : 100 * poly.P + 10 * poly.T + vertices poly = 249 := by
  sorry

end polyhedron_formula_l2831_283154


namespace all_digits_satisfy_inequality_l2831_283162

theorem all_digits_satisfy_inequality :
  ∀ A : ℕ, A ≤ 9 → 27 * 10 * A + 2708 - 1203 > 1022 := by
  sorry

end all_digits_satisfy_inequality_l2831_283162


namespace triangle_inequality_theorem_l2831_283152

def triangle_inequality (f g : ℝ → ℝ) (A B : ℝ) : Prop :=
  f (Real.cos A) * g (Real.sin B) > f (Real.sin B) * g (Real.cos A)

theorem triangle_inequality_theorem 
  (f g : ℝ → ℝ) 
  (hf : Differentiable ℝ f) 
  (hg : Differentiable ℝ g)
  (hg_pos : ∀ x, g x > 0)
  (h_deriv : ∀ x, (deriv f x) * (g x) - (f x) * (deriv g x) > 0)
  (A B C : ℝ)
  (h_obtuse : C > Real.pi / 2)
  (h_triangle : A + B + C = Real.pi) :
  triangle_inequality f g A B :=
sorry

end triangle_inequality_theorem_l2831_283152


namespace hash_example_l2831_283156

def hash (a b c d : ℝ) : ℝ := b^2 - 4*a*c + d^2

theorem hash_example : hash 2 3 1 4 = 17 := by sorry

end hash_example_l2831_283156


namespace inner_triangle_side_length_l2831_283137

theorem inner_triangle_side_length 
  (outer_side : ℝ) 
  (inner_side : ℝ) 
  (small_side : ℝ) 
  (h_outer : outer_side = 6) 
  (h_small : small_side = 1) 
  (h_parallel : inner_triangles_parallel_to_outer)
  (h_vertex_outer : inner_triangles_vertex_on_outer_side)
  (h_vertex_inner : inner_triangles_vertex_on_other_inner)
  (h_congruent : inner_triangles_congruent)
  : inner_side = 1/3 := by
  sorry

end inner_triangle_side_length_l2831_283137


namespace lucas_change_l2831_283128

/-- Represents the shopping scenario and calculates the change --/
def calculate_change (initial_amount : ℝ) 
  (avocado_costs : List ℝ) 
  (water_cost : ℝ) 
  (water_quantity : ℕ) 
  (apple_cost : ℝ) 
  (apple_quantity : ℕ) : ℝ :=
  let total_cost := (avocado_costs.sum + water_cost * water_quantity + apple_cost * apple_quantity)
  initial_amount - total_cost

/-- Theorem stating that Lucas brings home $6.75 in change --/
theorem lucas_change : 
  calculate_change 20 [1.50, 2.25, 3.00] 1.75 2 0.75 4 = 6.75 := by
  sorry

#eval calculate_change 20 [1.50, 2.25, 3.00] 1.75 2 0.75 4

end lucas_change_l2831_283128


namespace sum_of_digits_of_special_palindrome_l2831_283176

/-- A function that checks if a natural number is a three-digit palindrome -/
def isThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10) ∧ (n / 10 % 10 = n / 10 % 10)

/-- A function that calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  (n / 100) + (n / 10 % 10) + (n % 10)

/-- Theorem stating that if x is a three-digit palindrome and x + 50 is also a three-digit palindrome,
    then the sum of digits of x is 19 -/
theorem sum_of_digits_of_special_palindrome (x : ℕ) :
  isThreeDigitPalindrome x ∧ isThreeDigitPalindrome (x + 50) → sumOfDigits x = 19 :=
by sorry

end sum_of_digits_of_special_palindrome_l2831_283176


namespace subset_implies_lower_bound_l2831_283193

theorem subset_implies_lower_bound (a : ℝ) : 
  let M := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
  let N := {x : ℝ | x ≤ a}
  M ⊆ N → a ≥ 2 := by
  sorry

end subset_implies_lower_bound_l2831_283193


namespace total_profit_is_4650_l2831_283164

/-- Given the capitals of three individuals P, Q, and R, and the profit share of R,
    calculate the total profit. -/
def calculate_total_profit (Cp Cq Cr R_share : ℚ) : ℚ :=
  let total_ratio := (10 : ℚ) / 4 + 10 / 6 + 1
  R_share * total_ratio / (1 : ℚ)

/-- Theorem stating that under given conditions, the total profit is 4650. -/
theorem total_profit_is_4650 (Cp Cq Cr : ℚ) (h1 : 4 * Cp = 6 * Cq) (h2 : 6 * Cq = 10 * Cr) 
    (h3 : calculate_total_profit Cp Cq Cr 900 = 4650) : 
  calculate_total_profit Cp Cq Cr 900 = 4650 := by
  sorry

#eval calculate_total_profit 1 1 1 900

end total_profit_is_4650_l2831_283164


namespace function_non_positive_on_interval_l2831_283167

theorem function_non_positive_on_interval (a : ℝ) :
  (∃ x ∈ Set.Icc 0 1, a^2 * x - 2*a + 1 ≤ 0) ↔ a ≥ 1/2 := by sorry

end function_non_positive_on_interval_l2831_283167


namespace present_age_of_B_present_age_of_B_exists_l2831_283139

/-- Proves that given the conditions in the problem, B's current age is 37 years. -/
theorem present_age_of_B : ℕ → ℕ → Prop :=
  fun a b =>
    (a + 10 = 2 * (b - 10)) →  -- A will be twice as old as B was 10 years ago, in 10 years
    (a = b + 7) →              -- A is now 7 years older than B
    (b = 37)                   -- B's current age is 37

/-- The theorem holds for some values of a and b. -/
theorem present_age_of_B_exists : ∃ a b, present_age_of_B a b :=
  sorry

end present_age_of_B_present_age_of_B_exists_l2831_283139


namespace smallest_multiplier_for_450_cube_l2831_283112

/-- Given a positive integer n, returns true if n is a perfect cube, false otherwise -/
def isPerfectCube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

/-- The smallest positive integer that, when multiplied by 450, results in a perfect cube -/
def smallestMultiplier : ℕ := 60

theorem smallest_multiplier_for_450_cube :
  (isPerfectCube (450 * smallestMultiplier)) ∧
  (∀ n : ℕ, 0 < n → n < smallestMultiplier → ¬(isPerfectCube (450 * n))) :=
sorry

end smallest_multiplier_for_450_cube_l2831_283112


namespace constant_phi_is_cone_l2831_283173

/-- Spherical coordinates -/
structure SphericalCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- A cone in 3D space -/
def Cone : Set SphericalCoord := sorry

/-- The shape described by φ = c in spherical coordinates -/
def ConstantPhiShape (c : ℝ) : Set SphericalCoord :=
  {p : SphericalCoord | p.φ = c}

/-- Theorem stating that the shape described by φ = c is a cone -/
theorem constant_phi_is_cone (c : ℝ) :
  ConstantPhiShape c = Cone := by sorry

end constant_phi_is_cone_l2831_283173


namespace uncles_age_l2831_283198

theorem uncles_age (bud_age uncle_age : ℕ) : 
  bud_age = 8 → 
  3 * bud_age = uncle_age → 
  uncle_age = 24 := by
sorry

end uncles_age_l2831_283198


namespace prob_four_ones_in_five_rolls_l2831_283146

/-- The probability of rolling a 1 on a fair six-sided die -/
def prob_one : ℚ := 1 / 6

/-- The probability of not rolling a 1 on a fair six-sided die -/
def prob_not_one : ℚ := 5 / 6

/-- The number of rolls -/
def num_rolls : ℕ := 5

/-- The number of times we want to roll a 1 -/
def target_ones : ℕ := 4

/-- The probability of rolling exactly four 1s in five rolls of a fair six-sided die -/
theorem prob_four_ones_in_five_rolls : 
  (num_rolls.choose target_ones : ℚ) * prob_one ^ target_ones * prob_not_one ^ (num_rolls - target_ones) = 25 / 7776 := by
  sorry

end prob_four_ones_in_five_rolls_l2831_283146


namespace simplify_trig_expression_l2831_283141

theorem simplify_trig_expression :
  (Real.sqrt (1 - 2 * Real.sin (40 * π / 180) * Real.cos (40 * π / 180))) /
  (Real.cos (40 * π / 180) - Real.sqrt (1 - Real.sin (50 * π / 180) ^ 2)) = 1 := by
  sorry

end simplify_trig_expression_l2831_283141


namespace no_sum_of_squared_digits_greater_than_2008_l2831_283160

def sum_of_squared_digits (n : ℕ) : ℕ :=
  let digits := Nat.digits 10 n
  List.sum (List.map (λ d => d * d) digits)

theorem no_sum_of_squared_digits_greater_than_2008 :
  ∀ n : ℕ, n > 2008 → n ≠ sum_of_squared_digits n :=
sorry

end no_sum_of_squared_digits_greater_than_2008_l2831_283160


namespace plane_equation_correct_l2831_283138

/-- A plane in 3D Cartesian coordinates with intercepts a, b, and c on the x, y, and z axes respectively. -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  h₁ : a ≠ 0
  h₂ : b ≠ 0
  h₃ : c ≠ 0

/-- The equation of a plane in 3D Cartesian coordinates with given intercepts. -/
def planeEquation (p : Plane3D) (x y z : ℝ) : Prop :=
  x / p.a + y / p.b + z / p.c = 1

/-- Theorem stating that the equation x/a + y/b + z/c = 1 represents a plane
    with intercepts a, b, and c on the x, y, and z axes respectively. -/
theorem plane_equation_correct (p : Plane3D) :
  ∀ x y z : ℝ, planeEquation p x y z ↔ 
    (x = p.a ∧ y = 0 ∧ z = 0) ∨
    (x = 0 ∧ y = p.b ∧ z = 0) ∨
    (x = 0 ∧ y = 0 ∧ z = p.c) :=
  sorry

end plane_equation_correct_l2831_283138


namespace card_arrangement_possible_l2831_283101

def initial_sequence : List ℕ := [7, 8, 9, 4, 5, 6, 1, 2, 3]
def final_sequence : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

def reverse_sublist (l : List α) (start finish : ℕ) : List α :=
  (l.take start) ++ (l.drop start |>.take (finish - start + 1) |>.reverse) ++ (l.drop (finish + 1))

def can_transform (l : List ℕ) : Prop :=
  ∃ (s1 f1 s2 f2 s3 f3 : ℕ),
    reverse_sublist (reverse_sublist (reverse_sublist l s1 f1) s2 f2) s3 f3 = final_sequence

theorem card_arrangement_possible :
  can_transform initial_sequence :=
sorry

end card_arrangement_possible_l2831_283101


namespace min_sphere_surface_area_l2831_283174

/-- Given a rectangular parallelepiped with volume 12, height 4, and all vertices on the surface of a sphere,
    prove that the minimum surface area of the sphere is 22π. -/
theorem min_sphere_surface_area (a b c : ℝ) (h_volume : a * b * c = 12) (h_height : c = 4)
  (h_on_sphere : ∃ (r : ℝ), a^2 + b^2 + c^2 = 4 * r^2) :
  ∃ (S : ℝ), S = 22 * Real.pi ∧ ∀ (r : ℝ), (a^2 + b^2 + c^2 = 4 * r^2) → 4 * Real.pi * r^2 ≥ S := by
  sorry

end min_sphere_surface_area_l2831_283174


namespace horner_method_first_step_l2831_283169

def f (x : ℝ) : ℝ := 0.5 * x^5 + 4 * x^4 - 3 * x^2 + x - 1

def horner_first_step (a₅ a₄ : ℝ) (x : ℝ) : ℝ := a₅ * x + a₄

theorem horner_method_first_step :
  horner_first_step 0.5 4 3 = 5.5 :=
sorry

end horner_method_first_step_l2831_283169


namespace open_box_volume_is_5760_l2831_283175

/-- Calculate the volume of an open box formed by cutting squares from a rectangular sheet. -/
def openBoxVolume (sheetLength sheetWidth cutSize : ℝ) : ℝ :=
  (sheetLength - 2 * cutSize) * (sheetWidth - 2 * cutSize) * cutSize

/-- Theorem: The volume of the open box is 5760 m³ -/
theorem open_box_volume_is_5760 :
  openBoxVolume 52 36 8 = 5760 := by
  sorry

end open_box_volume_is_5760_l2831_283175


namespace magic_square_sum_l2831_283170

/-- Represents a 3x3 magic square with some known and unknown values -/
structure MagicSquare where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  f : ℕ
  sum : ℕ
  sum_eq_row1 : sum = 30 + e + 18
  sum_eq_row2 : sum = 15 + c + d
  sum_eq_row3 : sum = a + 27 + b
  sum_eq_col1 : sum = 30 + 15 + a
  sum_eq_col2 : sum = e + c + 27
  sum_eq_col3 : sum = 18 + d + b
  sum_eq_diag1 : sum = 30 + c + b
  sum_eq_diag2 : sum = a + c + 18

theorem magic_square_sum (ms : MagicSquare) : ms.d + ms.e = 47 := by
  sorry

end magic_square_sum_l2831_283170


namespace triangle_angle_C_l2831_283122

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_angles : A + B + C = Real.pi

-- Define the conditions of the problem
def problem_conditions (t : Triangle) : Prop :=
  5 * Real.sin t.A + 3 * Real.cos t.B = 7 ∧
  3 * Real.sin t.B + 5 * Real.cos t.A = 3

-- Theorem statement
theorem triangle_angle_C (t : Triangle) :
  problem_conditions t → Real.sin t.C = 4/5 := by sorry

end triangle_angle_C_l2831_283122


namespace first_discount_percentage_l2831_283165

theorem first_discount_percentage (original_price : ℝ) (final_price : ℝ) (second_discount : ℝ) :
  original_price = 32 →
  final_price = 18 →
  second_discount = 0.25 →
  ∃ (first_discount : ℝ),
    final_price = original_price * (1 - first_discount) * (1 - second_discount) ∧
    first_discount = 0.25 := by
  sorry

end first_discount_percentage_l2831_283165


namespace smallest_n_for_gcd_lcm_condition_l2831_283148

theorem smallest_n_for_gcd_lcm_condition : ∃ (n : ℕ), 
  (∃ (a b : ℕ), Nat.gcd a b = 999 ∧ Nat.lcm a b = n.factorial) ∧ 
  (∀ (m : ℕ), m < n → ¬∃ (a b : ℕ), Nat.gcd a b = 999 ∧ Nat.lcm a b = m.factorial) :=
by sorry

end smallest_n_for_gcd_lcm_condition_l2831_283148


namespace function_extremum_l2831_283187

/-- The function f(x) with parameters a and b -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f(x) with respect to x -/
def f_deriv (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem function_extremum (a b : ℝ) :
  f a b 1 = 10 ∧ f_deriv a b 1 = 0 →
  ∀ x, f a b x = x^3 + 4*x^2 - 11*x + 16 :=
by sorry

end function_extremum_l2831_283187


namespace quadratic_equation_solution_l2831_283158

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ = 2 + Real.sqrt 10 ∧ x₂ = 2 - Real.sqrt 10) ∧ 
  (x₁^2 - 4*x₁ - 6 = 0 ∧ x₂^2 - 4*x₂ - 6 = 0) := by
  sorry

end quadratic_equation_solution_l2831_283158


namespace candy_parade_total_l2831_283194

/-- The total number of candy pieces caught by Tabitha and her friends at the Christmas parade -/
theorem candy_parade_total (tabitha stan : ℕ) (julie carlos : ℕ) 
    (h1 : tabitha = 22)
    (h2 : stan = 13)
    (h3 : julie = tabitha / 2)
    (h4 : carlos = 2 * stan) :
  tabitha + stan + julie + carlos = 72 := by
  sorry

end candy_parade_total_l2831_283194


namespace smallest_prime_with_30_divisors_l2831_283127

/-- A function that counts the number of positive divisors of a natural number -/
def count_divisors (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number is prime -/
def is_prime (n : ℕ) : Prop := sorry

/-- The expression p^3 + 4p^2 + 4p -/
def f (p : ℕ) : ℕ := p^3 + 4*p^2 + 4*p

theorem smallest_prime_with_30_divisors :
  ∀ p : ℕ, is_prime p → (∀ q < p, is_prime q → count_divisors (f q) ≠ 30) →
  count_divisors (f p) = 30 → p = 43 :=
sorry

end smallest_prime_with_30_divisors_l2831_283127


namespace ralph_received_eight_cards_l2831_283147

/-- The number of cards Ralph's father gave him -/
def cards_from_father (initial_cards final_cards : ℕ) : ℕ :=
  final_cards - initial_cards

/-- Proof that Ralph's father gave him 8 cards -/
theorem ralph_received_eight_cards :
  let initial_cards : ℕ := 4
  let final_cards : ℕ := 12
  cards_from_father initial_cards final_cards = 8 := by
  sorry

end ralph_received_eight_cards_l2831_283147


namespace octopus_puzzle_l2831_283140

structure Octopus where
  color : String
  legs : Nat
  statement : Bool

def isLying (o : Octopus) : Bool :=
  (o.legs = 7 ∧ ¬o.statement) ∨ (o.legs = 8 ∧ o.statement)

def totalLegs (os : List Octopus) : Nat :=
  os.foldl (fun acc o => acc + o.legs) 0

theorem octopus_puzzle :
  ∃ (green blue red : Octopus),
    [green, blue, red].all (fun o => o.legs = 7 ∨ o.legs = 8) ∧
    isLying green ∧
    ¬isLying blue ∧
    isLying red ∧
    green.statement = (totalLegs [green, blue, red] = 21) ∧
    blue.statement = ¬green.statement ∧
    red.statement = (¬green.statement ∧ ¬blue.statement) ∧
    green.legs = 7 ∧
    blue.legs = 8 ∧
    red.legs = 7 :=
  sorry

#check octopus_puzzle

end octopus_puzzle_l2831_283140


namespace fraction_reduction_divisibility_l2831_283114

theorem fraction_reduction_divisibility
  (a b c d n : ℕ)
  (h1 : (a * n + b) % 2017 = 0)
  (h2 : (c * n + d) % 2017 = 0) :
  (a * d - b * c) % 2017 = 0 :=
by sorry

end fraction_reduction_divisibility_l2831_283114


namespace consecutive_integer_products_sum_l2831_283100

theorem consecutive_integer_products_sum : 
  ∃ (a b c x y z w : ℕ), 
    (b = a + 1) ∧ 
    (c = b + 1) ∧ 
    (y = x + 1) ∧ 
    (z = y + 1) ∧ 
    (w = z + 1) ∧ 
    (a * b * c = 924) ∧ 
    (x * y * z * w = 924) ∧ 
    (a + b + c + x + y + z + w = 75) := by
  sorry

end consecutive_integer_products_sum_l2831_283100


namespace cricket_run_rate_theorem_l2831_283120

/-- Represents a cricket game scenario -/
structure CricketGame where
  total_overs : ℕ
  first_overs : ℕ
  first_run_rate : ℚ
  target : ℕ

/-- Calculates the required run rate for the remaining overs -/
def required_run_rate (game : CricketGame) : ℚ :=
  let remaining_overs := game.total_overs - game.first_overs
  let runs_scored := game.first_run_rate * game.first_overs
  let runs_needed := game.target - runs_scored
  runs_needed / remaining_overs

/-- Theorem stating the required run rate for the given scenario -/
theorem cricket_run_rate_theorem (game : CricketGame) 
  (h1 : game.total_overs = 50)
  (h2 : game.first_overs = 10)
  (h3 : game.first_run_rate = 4.8)
  (h4 : game.target = 282) :
  required_run_rate game = 5.85 := by
  sorry

#eval required_run_rate { total_overs := 50, first_overs := 10, first_run_rate := 4.8, target := 282 }

end cricket_run_rate_theorem_l2831_283120


namespace interview_probability_l2831_283103

theorem interview_probability (total_students : ℕ) (french_students : ℕ) (spanish_students : ℕ) (german_students : ℕ)
  (h1 : total_students = 30)
  (h2 : french_students = 22)
  (h3 : spanish_students = 25)
  (h4 : german_students = 5)
  (h5 : french_students ≤ spanish_students)
  (h6 : spanish_students ≤ total_students)
  (h7 : german_students ≤ total_students) :
  let non_french_spanish : ℕ := total_students - spanish_students
  let total_combinations : ℕ := total_students.choose 2
  let non_informative_combinations : ℕ := (non_french_spanish + (spanish_students - french_students)).choose 2
  (1 : ℚ) - (non_informative_combinations : ℚ) / (total_combinations : ℚ) = 407 / 435 := by
    sorry

end interview_probability_l2831_283103


namespace weeks_to_save_shirt_l2831_283182

/-- Calculates the minimum number of whole weeks needed to save for a shirt -/
def min_weeks_to_save (shirt_cost : ℚ) (initial_savings : ℚ) (weekly_savings : ℚ) : ℕ :=
  Nat.ceil ((shirt_cost - initial_savings) / weekly_savings)

/-- Theorem stating that 34 weeks are needed to save for the shirt under given conditions -/
theorem weeks_to_save_shirt : min_weeks_to_save 15 5 0.3 = 34 := by
  sorry

end weeks_to_save_shirt_l2831_283182


namespace factorization_difference_l2831_283126

theorem factorization_difference (a b : ℤ) : 
  (∀ y : ℝ, 3 * y^2 - y - 18 = (3 * y + a) * (y + b)) → 
  a - b = -11 := by
sorry

end factorization_difference_l2831_283126


namespace complex_modulus_problem_l2831_283181

theorem complex_modulus_problem (z : ℂ) (h : z * (1 + Complex.I) = 2 - Complex.I) : 
  Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end complex_modulus_problem_l2831_283181


namespace imaginary_part_of_z_l2831_283111

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 4 + 2 * Complex.I) :
  z.im = -1 := by
  sorry

end imaginary_part_of_z_l2831_283111


namespace symmetric_points_sum_power_l2831_283153

/-- Two points are symmetric with respect to the x-axis if their x-coordinates are equal
    and their y-coordinates are negatives of each other -/
def symmetric_wrt_x_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = p2.1 ∧ p1.2 = -p2.2

theorem symmetric_points_sum_power (a b : ℝ) :
  symmetric_wrt_x_axis (a - 1, 5) (2, b - 1) →
  (a + b) ^ 2023 = -1 := by
  sorry

end symmetric_points_sum_power_l2831_283153


namespace lawn_mowing_earnings_l2831_283168

theorem lawn_mowing_earnings 
  (lawns_mowed : ℕ) 
  (initial_savings : ℕ) 
  (total_after_mowing : ℕ) 
  (h1 : lawns_mowed = 5)
  (h2 : initial_savings = 7)
  (h3 : total_after_mowing = 47) :
  (total_after_mowing - initial_savings) / lawns_mowed = 8 := by
  sorry

end lawn_mowing_earnings_l2831_283168


namespace orange_count_in_second_group_l2831_283149

def apple_cost : ℚ := 21/100

theorem orange_count_in_second_group 
  (first_group : 6 * apple_cost + 3 * orange_cost = 177/100)
  (second_group : 2 * apple_cost + x * orange_cost = 127/100)
  (orange_cost : ℚ) (x : ℚ) : x = 5 := by
  sorry

end orange_count_in_second_group_l2831_283149


namespace points_difference_l2831_283109

theorem points_difference (zach_points ben_points : ℕ) 
  (h1 : zach_points = 42) 
  (h2 : ben_points = 21) : 
  zach_points - ben_points = 21 := by
sorry

end points_difference_l2831_283109


namespace sum_of_digits_up_to_billion_l2831_283177

/-- Sum of digits function for a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Sum of digits of all numbers from 1 to n -/
def sumOfDigitsUpTo (n : ℕ) : ℕ := sorry

/-- The main theorem: sum of digits of all numbers from 1 to 1 billion -/
theorem sum_of_digits_up_to_billion :
  sumOfDigitsUpTo 1000000000 = 40500000001 := by sorry

end sum_of_digits_up_to_billion_l2831_283177


namespace periodic_function_value_l2831_283110

/-- Given a function f(x) = a * sin(π * x + α) + b * cos(π * x + β) + 4,
    where a, b, α, β are non-zero real numbers, and f(2012) = 6,
    prove that f(2013) = 2 -/
theorem periodic_function_value (a b α β : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hα : α ≠ 0) (hβ : β ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * Real.sin (π * x + α) + b * Real.cos (π * x + β) + 4
  (f 2012 = 6) → (f 2013 = 2) := by
  sorry

end periodic_function_value_l2831_283110


namespace factor_expression_l2831_283124

theorem factor_expression (x : ℝ) : 270 * x^3 - 90 * x^2 + 18 * x = 18 * x * (15 * x^2 - 5 * x + 1) := by
  sorry

end factor_expression_l2831_283124


namespace prob_sum_three_l2831_283161

/-- Represents a ball with a number label -/
inductive Ball : Type
| one : Ball
| two : Ball

/-- Represents the result of two draws -/
structure TwoDraws where
  first : Ball
  second : Ball

/-- The set of all possible outcomes from two draws -/
def allOutcomes : Finset TwoDraws :=
  sorry

/-- The set of favorable outcomes (sum of drawn numbers is 3) -/
def favorableOutcomes : Finset TwoDraws :=
  sorry

/-- The probability of an event is the number of favorable outcomes
    divided by the total number of outcomes -/
def probability (event : Finset TwoDraws) : ℚ :=
  (event.card : ℚ) / (allOutcomes.card : ℚ)

/-- The main theorem: the probability of drawing two balls with sum 3 is 1/2 -/
theorem prob_sum_three : probability favorableOutcomes = 1/2 :=
  sorry

end prob_sum_three_l2831_283161


namespace a_share_of_profit_is_correct_l2831_283145

/-- Calculates the share of profit for an investor in a partnership business. -/
def calculate_share_of_profit (investment_a investment_b investment_c total_profit : ℚ) : ℚ :=
  let total_investment := investment_a + investment_b + investment_c
  (investment_a / total_investment) * total_profit

/-- Theorem stating that A's share of the profit is correctly calculated. -/
theorem a_share_of_profit_is_correct (investment_a investment_b investment_c total_profit : ℚ) :
  calculate_share_of_profit investment_a investment_b investment_c total_profit =
  3750 / 1 :=
by sorry

end a_share_of_profit_is_correct_l2831_283145


namespace min_value_theorem_l2831_283157

open Real

theorem min_value_theorem (x y : ℝ) (hx : x > 1) (hy : y > 1) (h : x * y - 2 * x - y + 1 = 0) :
  ∃ (m : ℝ), m = 15 ∧ ∀ (a b : ℝ), a > 1 → b > 1 → a * b - 2 * a - b + 1 = 0 → (3/2) * a^2 + b^2 ≥ m :=
by sorry

end min_value_theorem_l2831_283157


namespace term_without_x_in_special_expansion_l2831_283143

/-- Given a binomial expansion of (x³ + 1/x²)^n where n is such that only
    the coefficient of the sixth term is maximum, the term without x is 210 -/
theorem term_without_x_in_special_expansion :
  ∃ n : ℕ,
    (∀ k : ℕ, k ≠ 5 → Nat.choose n k ≤ Nat.choose n 5) ∧
    (∃ r : ℕ, Nat.choose n r = 210 ∧ 3 * n = 5 * r) :=
sorry

end term_without_x_in_special_expansion_l2831_283143


namespace complement_of_union_l2831_283180

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 2, 3, 4}
def B : Set Nat := {1, 3, 5}

theorem complement_of_union :
  (U \ (A ∪ B)) = {6} := by sorry

end complement_of_union_l2831_283180


namespace sqrt_two_divided_by_sqrt_two_minus_one_l2831_283130

theorem sqrt_two_divided_by_sqrt_two_minus_one :
  Real.sqrt 2 / (Real.sqrt 2 - 1) = 2 + Real.sqrt 2 := by
  sorry

end sqrt_two_divided_by_sqrt_two_minus_one_l2831_283130


namespace selection_theorem_l2831_283102

def male_teachers : ℕ := 5
def female_teachers : ℕ := 3
def total_selection : ℕ := 3

def select_with_both_genders (m f s : ℕ) : ℕ :=
  Nat.choose m 2 * Nat.choose f 1 + Nat.choose m 1 * Nat.choose f 2

theorem selection_theorem :
  select_with_both_genders male_teachers female_teachers total_selection = 45 := by
  sorry

end selection_theorem_l2831_283102
