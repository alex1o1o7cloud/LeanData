import Mathlib

namespace layoffs_payment_l2280_228078

theorem layoffs_payment :
  let total_employees := 450
  let salary_2000_employees := 150
  let salary_2500_employees := 200
  let salary_3000_employees := 100
  let first_round_2000_layoffs := 0.20 * salary_2000_employees
  let first_round_2500_layoffs := 0.25 * salary_2500_employees
  let first_round_3000_layoffs := 0.15 * salary_3000_employees
  let remaining_2000_after_first_round := salary_2000_employees - first_round_2000_layoffs
  let remaining_2500_after_first_round := salary_2500_employees - first_round_2500_layoffs
  let remaining_3000_after_first_round := salary_3000_employees - first_round_3000_layoffs
  let second_round_2000_layoffs := 0.10 * remaining_2000_after_first_round
  let second_round_2500_layoffs := 0.15 * remaining_2500_after_first_round
  let second_round_3000_layoffs := 0.05 * remaining_3000_after_first_round
  let remaining_2000_after_second_round := remaining_2000_after_first_round - second_round_2000_layoffs
  let remaining_2500_after_second_round := remaining_2500_after_first_round - second_round_2500_layoffs
  let remaining_3000_after_second_round := remaining_3000_after_first_round - second_round_3000_layoffs
  let total_payment := remaining_2000_after_second_round * 2000 + remaining_2500_after_second_round * 2500 + remaining_3000_after_second_round * 3000
  total_payment = 776500 := sorry

end layoffs_payment_l2280_228078


namespace find_point_D_l2280_228075

structure Point :=
  (x : ℤ)
  (y : ℤ)

def translation_rule (A C : Point) : Point :=
{
  x := C.x - A.x,
  y := C.y - A.y
}

def translate (P delta : Point) : Point :=
{
  x := P.x + delta.x,
  y := P.y + delta.y
}

def A := Point.mk (-1) 4
def C := Point.mk 1 2
def B := Point.mk 2 1
def D := Point.mk 4 (-1)
def translation_delta : Point := translation_rule A C

theorem find_point_D : translate B translation_delta = D :=
by
  sorry

end find_point_D_l2280_228075


namespace diana_total_cost_l2280_228045

noncomputable def shopping_total_cost := 
  let t_shirt_price := 10
  let sweater_price := 25
  let jacket_price := 100
  let jeans_price := 40
  let shoes_price := 70 

  let t_shirt_discount := 0.20
  let sweater_discount := 0.10
  let jacket_discount := 0.15
  let jeans_discount := 0.05
  let shoes_discount := 0.25

  let clothes_tax := 0.06
  let shoes_tax := 0.09

  let t_shirt_qty := 8
  let sweater_qty := 5
  let jacket_qty := 3
  let jeans_qty := 6
  let shoes_qty := 4

  let t_shirt_total := t_shirt_qty * t_shirt_price 
  let sweater_total := sweater_qty * sweater_price 
  let jacket_total := jacket_qty * jacket_price 
  let jeans_total := jeans_qty * jeans_price 
  let shoes_total := shoes_qty * shoes_price 

  let t_shirt_discounted := t_shirt_total * (1 - t_shirt_discount)
  let sweater_discounted := sweater_total * (1 - sweater_discount)
  let jacket_discounted := jacket_total * (1 - jacket_discount)
  let jeans_discounted := jeans_total * (1 - jeans_discount)
  let shoes_discounted := shoes_total * (1 - shoes_discount)

  let t_shirt_final := t_shirt_discounted * (1 + clothes_tax)
  let sweater_final := sweater_discounted * (1 + clothes_tax)
  let jacket_final := jacket_discounted * (1 + clothes_tax)
  let jeans_final := jeans_discounted * (1 + clothes_tax)
  let shoes_final := shoes_discounted * (1 + shoes_tax)

  t_shirt_final + sweater_final + jacket_final + jeans_final + shoes_final

theorem diana_total_cost : shopping_total_cost = 927.97 :=
by sorry

end diana_total_cost_l2280_228045


namespace simplify_fraction_l2280_228062

variables {a b c x y z : ℝ}

theorem simplify_fraction :
  (cx * (a^2 * x^2 + 3 * a^2 * y^2 + c^2 * z^2) + bz * (a^2 * x^2 + 3 * c^2 * x^2 + c^2 * y^2)) / (cx + bz) =
  a^2 * x^2 + c^2 * y^2 + c^2 * z^2 :=
sorry

end simplify_fraction_l2280_228062


namespace triangle_inequality_l2280_228020

theorem triangle_inequality (S R r : ℝ) (h : S^2 = 2 * R^2 + 8 * R * r + 3 * r^2) : 
  R / r = 2 ∨ R / r ≥ Real.sqrt 2 + 1 := 
by 
  sorry

end triangle_inequality_l2280_228020


namespace tenth_term_arithmetic_sequence_l2280_228035

theorem tenth_term_arithmetic_sequence :
  ∀ (a₁ a₃₀ : ℕ) (d : ℕ) (n : ℕ), a₁ = 3 → a₃₀ = 89 → n = 10 → 
  (a₃₀ - a₁) / 29 = d → a₁ + (n - 1) * d = 30 :=
by
  intros a₁ a₃₀ d n h₁ h₃₀ hn hd
  sorry

end tenth_term_arithmetic_sequence_l2280_228035


namespace divisible_by_55_l2280_228006

theorem divisible_by_55 (n : ℤ) : 
  (55 ∣ (n^2 + 3 * n + 1)) ↔ (n % 55 = 46 ∨ n % 55 = 6) := 
by 
  sorry

end divisible_by_55_l2280_228006


namespace sin2alpha_div_1_plus_cos2alpha_eq_3_l2280_228067

theorem sin2alpha_div_1_plus_cos2alpha_eq_3 (α : ℝ) (h : Real.tan α = 3) : 
  (Real.sin (2 * α)) / (1 + Real.cos (2 * α)) = 3 := 
  sorry

end sin2alpha_div_1_plus_cos2alpha_eq_3_l2280_228067


namespace half_of_animals_get_sick_l2280_228012

theorem half_of_animals_get_sick : 
  let chickens := 26
  let piglets := 40
  let goats := 34
  let total_animals := chickens + piglets + goats
  let sick_animals := total_animals / 2
  sick_animals = 50 :=
by
  sorry

end half_of_animals_get_sick_l2280_228012


namespace average_percent_increase_per_year_l2280_228049

def initial_population : ℕ := 175000
def final_population : ℕ := 262500
def years : ℕ := 10

theorem average_percent_increase_per_year :
  ( ( ( ( final_population - initial_population ) / years : ℝ ) / initial_population ) * 100 ) = 5 := by
  sorry

end average_percent_increase_per_year_l2280_228049


namespace value_of_a_minus_b_l2280_228064

theorem value_of_a_minus_b (a b : ℤ) 
  (h₁ : |a| = 7) 
  (h₂ : |b| = 5) 
  (h₃ : a < b) : 
  a - b = -12 ∨ a - b = -2 := 
sorry

end value_of_a_minus_b_l2280_228064


namespace bike_price_l2280_228052

-- Definitions of the conditions
def maria_savings : ℕ := 120
def mother_offer : ℕ := 250
def amount_needed : ℕ := 230

-- Theorem statement
theorem bike_price (maria_savings mother_offer amount_needed : ℕ) : 
  maria_savings + mother_offer + amount_needed = 600 := 
by
  -- Sorry is used here to skip the actual proof steps
  sorry

end bike_price_l2280_228052


namespace work_problem_l2280_228007

theorem work_problem (W : ℝ) (d : ℝ) :
  (1 / 40) * d * W + (28 / 35) * W = W → d = 8 :=
by
  intro h
  sorry

end work_problem_l2280_228007


namespace proof_problem_l2280_228059

axiom is_line (m : Type) : Prop
axiom is_plane (α : Type) : Prop
axiom is_subset_of_plane (m : Type) (β : Type) : Prop
axiom is_perpendicular (a : Type) (b : Type) : Prop
axiom is_parallel (a : Type) (b : Type) : Prop

theorem proof_problem
  (m n : Type) 
  (α β : Type)
  (h1 : is_line m)
  (h2 : is_line n)
  (h3 : is_plane α)
  (h4 : is_plane β)
  (h_prop2 : is_parallel α β → is_subset_of_plane m α → is_parallel m β)
  (h_prop3 : is_perpendicular n α → is_perpendicular n β → is_perpendicular m α → is_perpendicular m β)
  : (is_subset_of_plane m β → is_perpendicular α β → ¬ (is_perpendicular m α)) ∧ 
    (is_parallel m α → is_parallel m β → ¬ (is_parallel α β)) :=
sorry

end proof_problem_l2280_228059


namespace nickels_used_for_notebook_l2280_228057

def notebook_cost_dollars : ℚ := 1.30
def dollar_to_cents_conversion : ℤ := 100
def nickel_value_cents : ℤ := 5

theorem nickels_used_for_notebook : 
  (notebook_cost_dollars * dollar_to_cents_conversion) / nickel_value_cents = 26 := 
by 
  sorry

end nickels_used_for_notebook_l2280_228057


namespace rectangle_perimeter_given_square_l2280_228051

-- Defining the problem conditions
def square_side_length (p : ℕ) : ℕ := p / 4

def rectangle_perimeter (s : ℕ) : ℕ := 2 * (s + (s / 2))

-- Stating the theorem: Given the perimeter of the square is 80, prove the perimeter of one of the rectangles is 60
theorem rectangle_perimeter_given_square (p : ℕ) (h : p = 80) : rectangle_perimeter (square_side_length p) = 60 :=
by
  sorry

end rectangle_perimeter_given_square_l2280_228051


namespace marble_problem_l2280_228034

theorem marble_problem : Nat.lcm (Nat.lcm (Nat.lcm 2 3) 5) 7 = 210 := by
  sorry

end marble_problem_l2280_228034


namespace ratio_divisor_to_remainder_l2280_228001

theorem ratio_divisor_to_remainder (R D Q : ℕ) (hR : R = 46) (hD : D = 10 * Q) (hdvd : 5290 = D * Q + R) :
  D / R = 5 :=
by
  sorry

end ratio_divisor_to_remainder_l2280_228001


namespace probability_star_top_card_is_one_fifth_l2280_228036

-- Define the total number of cards in the deck
def total_cards : ℕ := 65

-- Define the number of star cards in the deck
def star_cards : ℕ := 13

-- Define the probability calculation
def probability_star_top_card : ℚ := star_cards / total_cards

-- State the theorem regarding the probability
theorem probability_star_top_card_is_one_fifth :
  probability_star_top_card = 1 / 5 :=
by
  sorry

end probability_star_top_card_is_one_fifth_l2280_228036


namespace dealer_gross_profit_l2280_228026

theorem dealer_gross_profit (P S G : ℝ) (hP : P = 150) (markup : S = P + 0.5 * S) :
  G = S - P → G = 150 :=
by
  sorry

end dealer_gross_profit_l2280_228026


namespace polynomial_divisibility_l2280_228095

theorem polynomial_divisibility (P : Polynomial ℝ) (n : ℕ) (h_pos : 0 < n) :
  ∃ Q : Polynomial ℝ, (P * P + Q * Q) % (X * X + 1)^n = 0 :=
sorry

end polynomial_divisibility_l2280_228095


namespace seq_property_l2280_228050

theorem seq_property (m : ℤ) (h1 : |m| ≥ 2)
  (a : ℕ → ℤ)
  (h2 : ¬ (a 1 = 0 ∧ a 2 = 0))
  (h3 : ∀ n : ℕ, n > 0 → a (n + 2) = a (n + 1) - m * a n)
  (r s : ℕ)
  (h4 : r > s ∧ s ≥ 2)
  (h5 : a r = a 1 ∧ a s = a 1) :
  r - s ≥ |m| :=
by
  sorry

end seq_property_l2280_228050


namespace star_of_15_star_eq_neg_15_l2280_228037

def y_star (y : ℤ) : ℤ := 10 - y
def star_y (y : ℤ) : ℤ := y - 10

theorem star_of_15_star_eq_neg_15 : star_y (y_star 15) = -15 :=
by {
  -- applying given definitions;
  sorry
}

end star_of_15_star_eq_neg_15_l2280_228037


namespace bail_rate_l2280_228005

theorem bail_rate 
  (distance_to_shore : ℝ) 
  (shore_speed : ℝ) 
  (leak_rate : ℝ) 
  (boat_capacity : ℝ) 
  (time_to_shore_min : ℝ) 
  (net_water_intake : ℝ)
  (r : ℝ) :
  distance_to_shore = 2 →
  shore_speed = 3 →
  leak_rate = 12 →
  boat_capacity = 40 →
  time_to_shore_min = 40 →
  net_water_intake = leak_rate - r →
  net_water_intake * (time_to_shore_min) ≤ boat_capacity →
  r ≥ 11 :=
by
  intros h_dist h_speed h_leak h_cap h_time h_net h_ineq
  sorry

end bail_rate_l2280_228005


namespace max_m_value_l2280_228099

theorem max_m_value (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) : 
  (∀ a b : ℝ, a > 0 → b > 0 → ((2 / a) + (1 / b) ≥ (m / (2 * a + b)))) → m ≤ 9 :=
sorry

end max_m_value_l2280_228099


namespace milk_quality_check_l2280_228063

/-
Suppose there is a collection of 850 bags of milk numbered from 001 to 850. 
From this collection, 50 bags are randomly selected for testing by reading numbers 
from a random number table. Starting from the 3rd line and the 1st group of numbers, 
continuing to the right, we need to find the next 4 bag numbers after the sequence 
614, 593, 379, 242.
-/

def random_numbers : List Nat := [
  78226, 85384, 40527, 48987, 60602, 16085, 29971, 61279,
  43021, 92980, 27768, 26916, 27783, 84572, 78483, 39820,
  61459, 39073, 79242, 20372, 21048, 87088, 34600, 74636,
  63171, 58247, 12907, 50303, 28814, 40422, 97895, 61421,
  42372, 53183, 51546, 90385, 12120, 64042, 51320, 22983
]

noncomputable def next_valid_numbers (nums : List Nat) (start_idx : Nat) : List Nat :=
  nums.drop start_idx |>.filter (λ n => n ≤ 850) |>.take 4

theorem milk_quality_check :
  next_valid_numbers random_numbers 18 = [203, 722, 104, 88] :=
sorry

end milk_quality_check_l2280_228063


namespace find_a12_l2280_228000

namespace ArithmeticSequence

variable {α : Type*} [LinearOrderedField α]

def arithmetic_sequence (a : ℕ → α) :=
  ∃ d : α, ∀ n : ℕ, a n = a 0 + n * d

theorem find_a12 {a : ℕ → α} (h1 : a 4 = 1) (h2 : a 7 + a 9 = 16) :
  a 12 = 15 := 
sorry

end ArithmeticSequence

end find_a12_l2280_228000


namespace fraction_of_number_l2280_228027

theorem fraction_of_number (x : ℕ) (f : ℚ) (h1 : x = 140) (h2 : 0.65 * x = f * x - 21) : f = 0.8 :=
sorry

end fraction_of_number_l2280_228027


namespace chess_player_max_consecutive_win_prob_l2280_228096

theorem chess_player_max_consecutive_win_prob
  {p1 p2 p3 : ℝ} 
  (h1 : 0 < p1)
  (h2 : p1 < p2)
  (h3 : p2 < p3) :
  ∀ pA pB pC : ℝ, pC = (2 * p3 * (p1 + p2) - 4 * p1 * p2 * p3) 
                  → pB = (2 * p2 * (p1 + p3) - 4 * p1 * p2 * p3) 
                  → pA = (2 * p1 * (p2 + p3) - 4 * p1 * p2 * p3) 
                  → pC > pB ∧ pC > pA := 
by
  sorry

end chess_player_max_consecutive_win_prob_l2280_228096


namespace proof_inequality_l2280_228086

noncomputable def inequality_proof (α : ℝ) (a b : ℝ) (m : ℕ) : Prop :=
  (0 < α) → (α < Real.pi / 2) →
  (m ≥ 1) →
  (0 < a) → (0 < b) →
  (a / (Real.cos α)^m + b / (Real.sin α)^m ≥ (a^((2:ℝ)/(m+2)) + b^((2:ℝ)/(m+2)))^((m+2)/2))

-- Statement of the proof problem
theorem proof_inequality (α : ℝ) (a b : ℝ) (m : ℕ) 
  (h1 : 0 < α) (h2 : α < Real.pi / 2) (h3 : 1 ≤ m) (h4 : 0 < a) (h5 : 0 < b) : 
  a / (Real.cos α)^m + b / (Real.sin α)^m ≥ 
    (a^((2:ℝ)/(m+2)) + b^((2:ℝ)/(m+2)))^((m+2)/2) :=
by
  sorry

end proof_inequality_l2280_228086


namespace ratio_JL_JM_l2280_228028

theorem ratio_JL_JM (s w h : ℝ) (shared_area_25 : 0.25 * s^2 = 0.4 * w * h) (jm_eq_s : h = s) :
  w / h = 5 / 8 :=
by
  -- Proof will go here
  sorry

end ratio_JL_JM_l2280_228028


namespace base_eight_to_base_ten_l2280_228068

theorem base_eight_to_base_ten (a b c : ℕ) (ha : a = 1) (hb : b = 5) (hc : c = 7) :
  a * 8^2 + b * 8^1 + c * 8^0 = 111 :=
by 
  -- rest of the proof will go here
  sorry

end base_eight_to_base_ten_l2280_228068


namespace weights_problem_l2280_228091

theorem weights_problem
  (weights : Fin 10 → ℝ)
  (h1 : ∀ (i j k l a b c : Fin 10), i ≠ j → i ≠ k → i ≠ l → i ≠ a → i ≠ b → i ≠ c →
    j ≠ k → j ≠ l → j ≠ a → j ≠ b → j ≠ c →
    k ≠ l → k ≠ a → k ≠ b → k ≠ c → 
    l ≠ a → l ≠ b → l ≠ c →
    a ≠ b → a ≠ c →
    b ≠ c →
    weights i + weights j + weights k + weights l > weights a + weights b + weights c)
  (h2 : ∀ (i j : Fin 9), weights i ≤ weights (i + 1)) :
  ∀ (i j k a b : Fin 10), i ≠ j → i ≠ k → i ≠ a → i ≠ b → j ≠ k → j ≠ a → j ≠ b → k ≠ a → k ≠ b → a ≠ b → 
    weights i + weights j + weights k > weights a + weights b := 
sorry

end weights_problem_l2280_228091


namespace dima_is_mistaken_l2280_228046

theorem dima_is_mistaken :
  (∃ n : Nat, n > 0 ∧ ∀ n, 3 * n = 4 * n) → False :=
by
  intros h
  obtain ⟨n, hn1, hn2⟩ := h
  have hn := (hn2 n)
  linarith

end dima_is_mistaken_l2280_228046


namespace find_a_plus_b_l2280_228016

theorem find_a_plus_b (a b : ℕ) (positive_a : 0 < a) (positive_b : 0 < b)
  (condition : ∀ (n : ℕ), (n > 0) → (∃ m n : ℕ, n = m * a + n * b) ∨ (∃ k l : ℕ, n = 2009 + k * a + l * b))
  (not_expressible : ∃ m n : ℕ, 1776 = m * a + n * b): a + b = 133 :=
sorry

end find_a_plus_b_l2280_228016


namespace solution_1_solution_2_l2280_228031

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 3)

theorem solution_1 :
  (∀ x : ℝ, f x = 2 * Real.sin (2 * x + Real.pi / 3)) :=
by sorry

theorem solution_2 (x0 : ℝ) (hx0 : x0 ∈ Set.Icc (Real.pi / 2) Real.pi) :
  f (x0 / 2) = -3 / 8 → 
  Real.cos (x0 + Real.pi / 6) = - Real.sqrt 741 / 32 - 3 / 32 :=
by sorry

end solution_1_solution_2_l2280_228031


namespace sum_of_nonneg_reals_l2280_228025

theorem sum_of_nonneg_reals (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0)
  (h4 : x^2 + y^2 + z^2 = 52) (h5 : x * y + y * z + z * x = 24) :
  x + y + z = 10 :=
sorry

end sum_of_nonneg_reals_l2280_228025


namespace no_integer_pair_2006_l2280_228002

theorem no_integer_pair_2006 : ∀ (x y : ℤ), x^2 - y^2 ≠ 2006 := by
  sorry

end no_integer_pair_2006_l2280_228002


namespace solve_system_l2280_228033

open Real

-- Define the system of equations as hypotheses
def eqn1 (x y z : ℝ) : Prop := x + y + 2 - 4 * x * y = 0
def eqn2 (x y z : ℝ) : Prop := y + z + 2 - 4 * y * z = 0
def eqn3 (x y z : ℝ) : Prop := z + x + 2 - 4 * z * x = 0

-- State the theorem
theorem solve_system (x y z : ℝ) :
  (eqn1 x y z ∧ eqn2 x y z ∧ eqn3 x y z) ↔ 
  ((x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = -1/2 ∧ y = -1/2 ∧ z = -1/2)) :=
by 
  sorry

end solve_system_l2280_228033


namespace number_of_ways_to_choose_marbles_l2280_228043

theorem number_of_ways_to_choose_marbles 
  (total_marbles : ℕ) 
  (red_count green_count blue_count : ℕ) 
  (total_choice chosen_rgb_count remaining_choice : ℕ) 
  (h_total_marbles : total_marbles = 15) 
  (h_red_count : red_count = 2) 
  (h_green_count : green_count = 2) 
  (h_blue_count : blue_count = 2) 
  (h_total_choice : total_choice = 5) 
  (h_chosen_rgb_count : chosen_rgb_count = 2) 
  (h_remaining_choice : remaining_choice = 3) :
  ∃ (num_ways : ℕ), num_ways = 3300 :=
sorry

end number_of_ways_to_choose_marbles_l2280_228043


namespace marcus_baseball_cards_l2280_228018

-- Define the number of baseball cards Carter has
def carter_cards : ℕ := 152

-- Define the number of additional cards Marcus has compared to Carter
def additional_cards : ℕ := 58

-- Define the number of baseball cards Marcus has
def marcus_cards : ℕ := carter_cards + additional_cards

-- The proof statement asserting Marcus' total number of baseball cards
theorem marcus_baseball_cards : marcus_cards = 210 :=
by {
  -- This is where the proof steps would go, but we are skipping with sorry
  sorry
}

end marcus_baseball_cards_l2280_228018


namespace problem1_problem2_l2280_228058

variable {a b : ℝ}

theorem problem1 (h : a ≠ b) : 
  ((b / (a - b)) - (a / (a - b))) = -1 := 
by
  sorry

theorem problem2 (h1 : a ≠ b) (h2 : a ≠ 0) (h3 : b ≠ 0) : 
  ((a^2 - a * b)/(a^2) / ((a / b) - (b / a))) = (b / (a + b)) := 
by
  sorry

end problem1_problem2_l2280_228058


namespace number_of_truthful_dwarfs_l2280_228048

/-- Each of the 10 dwarfs either always tells the truth or always lies. 
It is known that each of them likes exactly one type of ice cream: vanilla, chocolate, or fruit. 
Prove the number of truthful dwarfs. -/
theorem number_of_truthful_dwarfs (x y : ℕ) (h1 : x + y = 10) (h2 : x + 2 * y = 16) : x = 4 :=
by sorry

end number_of_truthful_dwarfs_l2280_228048


namespace negation_P_l2280_228021

-- Define the original proposition P
def P (a b : ℝ) : Prop := (a^2 + b^2 = 0) → (a = 0 ∧ b = 0)

-- State the negation of P
theorem negation_P : ∀ (a b : ℝ), (a^2 + b^2 ≠ 0) → (a ≠ 0 ∨ b ≠ 0) :=
by
  sorry

end negation_P_l2280_228021


namespace botanical_garden_path_length_l2280_228023

theorem botanical_garden_path_length
  (scale : ℝ)
  (path_length_map : ℝ)
  (path_length_real : ℝ)
  (h_scale : scale = 500)
  (h_path_length_map : path_length_map = 6.5)
  (h_path_length_real : path_length_real = path_length_map * scale) :
  path_length_real = 3250 :=
by
  sorry

end botanical_garden_path_length_l2280_228023


namespace parabola_symmetry_l2280_228054

-- Define the function f as explained in the problem
def f (x : ℝ) (b c : ℝ) : ℝ := x^2 + b*x + c

-- Lean theorem to prove the inequality based on given conditions
theorem parabola_symmetry (b c : ℝ) (h : ∀ t : ℝ, f (3 + t) b c = f (3 - t) b c) :
  f 3 b c < f 1 b c ∧ f 1 b c < f 6 b c :=
by
  sorry

end parabola_symmetry_l2280_228054


namespace sum_of_coeffs_l2280_228053

theorem sum_of_coeffs (a0 a1 a2 a3 a4 a5 : ℤ)
  (h1 : (1 - 2 * (0 : ℤ))^5 = a0)
  (h2 : (1 - 2 * (1 : ℤ))^5 = a0 + a1 + a2 + a3 + a4 + a5) :
  a1 + a2 + a3 + a4 + a5 = -2 := by
  sorry

end sum_of_coeffs_l2280_228053


namespace sqrt_product_eq_225_l2280_228076

theorem sqrt_product_eq_225 : (Real.sqrt (5 * 3) * Real.sqrt (3 ^ 3 * 5 ^ 3) = 225) :=
by
  sorry

end sqrt_product_eq_225_l2280_228076


namespace function_increasing_interval_l2280_228056

theorem function_increasing_interval :
  (∀ x ∈ Set.Icc (0 : ℝ) (Real.pi),
  (2 * Real.sin ((Real.pi / 6) - 2 * x) : ℝ)
  ≤ 2 * Real.sin ((Real.pi / 6) - 2 * x + 1)) ↔ (x ∈ Set.Icc (Real.pi / 3) (5 * Real.pi / 6)) :=
sorry

end function_increasing_interval_l2280_228056


namespace sam_morning_run_distance_l2280_228087

variable (n : ℕ) (x : ℝ)

theorem sam_morning_run_distance (h : x + 2 * n * x + 12 = 18) : x = 6 / (1 + 2 * n) :=
by
  sorry

end sam_morning_run_distance_l2280_228087


namespace minimum_trucks_needed_l2280_228061

theorem minimum_trucks_needed 
  (total_weight : ℕ) (box_weight: ℕ) (truck_capacity: ℕ) (min_trucks: ℕ)
  (h_total_weight : total_weight = 10)
  (h_box_weight_le : ∀ (w : ℕ), w <= box_weight → w <= 1)
  (h_truck_capacity : truck_capacity = 3)
  (h_min_trucks : min_trucks = 5) : 
  min_trucks >= (total_weight / truck_capacity) :=
sorry

end minimum_trucks_needed_l2280_228061


namespace coprime_composite_lcm_l2280_228030

theorem coprime_composite_lcm (a b : ℕ) (ha : a > 1) (hb : b > 1) (hcoprime : Nat.gcd a b = 1) (hlcm : Nat.lcm a b = 120) : 
  Nat.gcd a b = 1 ∧ min a b = 8 := 
by 
  sorry

end coprime_composite_lcm_l2280_228030


namespace distinct_parallel_lines_l2280_228040

theorem distinct_parallel_lines (k : ℝ) :
  (∃ (L1 L2 : ℝ × ℝ → Prop), 
    (∀ x y, L1 (x, y) ↔ x - 2 * y - 3 = 0) ∧ 
    (∀ x y, L2 (x, y) ↔ 18 * x - k^2 * y - 9 * k = 0)) → 
  (∃ slope1 slope2, 
    slope1 = 1/2 ∧ 
    slope2 = 18 / k^2 ∧
    (slope1 = slope2) ∧
    (¬ (∀ x y, x - 2 * y - 3 = 18 * x - k^2 * y - 9 * k))) → 
  k = -6 :=
by 
  sorry

end distinct_parallel_lines_l2280_228040


namespace triangle_integral_y_difference_l2280_228090

theorem triangle_integral_y_difference :
  ∀ (y : ℕ), (3 ≤ y ∧ y ≤ 15) → (∃ y_min y_max : ℕ, y_min = 3 ∧ y_max = 15 ∧ (y_max - y_min = 12)) :=
by
  intro y
  intro h
  -- skipped proof
  sorry

end triangle_integral_y_difference_l2280_228090


namespace valid_integer_values_n_l2280_228083

def is_integer (x : ℝ) : Prop := ∃ k : ℤ, x = k

theorem valid_integer_values_n : ∃ (n_values : ℕ), n_values = 3 ∧
  ∀ n : ℤ, is_integer (3200 * (2 / 5) ^ (2 * n)) ↔ n = 0 ∨ n = 1 ∨ n = 2 :=
by
  sorry

end valid_integer_values_n_l2280_228083


namespace geometric_sequence_ratio_l2280_228072

theorem geometric_sequence_ratio
  (a1 r : ℝ) (h_r : r ≠ 1)
  (h : (1 - r^6) / (1 - r^3) = 1 / 2) :
  (1 - r^9) / (1 - r^3) = 3 / 4 :=
  sorry

end geometric_sequence_ratio_l2280_228072


namespace lily_pads_half_lake_l2280_228013

theorem lily_pads_half_lake (n : ℕ) (h : n = 39) :
  (n - 1) = 38 :=
by
  sorry

end lily_pads_half_lake_l2280_228013


namespace theater_earnings_l2280_228019

theorem theater_earnings :
  let matinee_price := 5
  let evening_price := 7
  let opening_night_price := 10
  let popcorn_price := 10
  let matinee_customers := 32
  let evening_customers := 40
  let opening_night_customers := 58
  let half_of_customers_that_bought_popcorn := 
    (matinee_customers + evening_customers + opening_night_customers) / 2
  let total_earnings := 
    (matinee_price * matinee_customers) + 
    (evening_price * evening_customers) + 
    (opening_night_price * opening_night_customers) + 
    (popcorn_price * half_of_customers_that_bought_popcorn)
  total_earnings = 1670 :=
by
  sorry

end theater_earnings_l2280_228019


namespace alyssa_went_to_13_games_last_year_l2280_228038

theorem alyssa_went_to_13_games_last_year :
  ∀ (X : ℕ), (11 + X + 15 = 39) → X = 13 :=
by
  intros X h
  sorry

end alyssa_went_to_13_games_last_year_l2280_228038


namespace determine_c_l2280_228079

theorem determine_c {f : ℝ → ℝ} (c : ℝ) (h : ∀ x, f x = 2 / (3 * x + c))
  (hf_inv : ∀ x, (f⁻¹ x) = (3 - 6 * x) / x) : c = 18 :=
by sorry

end determine_c_l2280_228079


namespace retirement_fund_increment_l2280_228098

theorem retirement_fund_increment (k y : ℝ) (h1 : k * Real.sqrt (y + 3) = k * Real.sqrt y + 15)
  (h2 : k * Real.sqrt (y + 5) = k * Real.sqrt y + 27) : k * Real.sqrt y = 810 := by
  sorry

end retirement_fund_increment_l2280_228098


namespace determine_k_l2280_228084

theorem determine_k (k : ℝ) (h : 2 - 2^2 = k * (2)^2 + 1) : k = -3/4 :=
by
  sorry

end determine_k_l2280_228084


namespace ineq_five_times_x_minus_six_gt_one_l2280_228081

variable {x : ℝ}

theorem ineq_five_times_x_minus_six_gt_one (x : ℝ) : 5 * x - 6 > 1 :=
sorry

end ineq_five_times_x_minus_six_gt_one_l2280_228081


namespace apps_difference_l2280_228070

variable (initial_apps : ℕ) (added_apps : ℕ) (apps_left : ℕ)
variable (total_apps : ℕ := initial_apps + added_apps)
variable (deleted_apps : ℕ := total_apps - apps_left)
variable (difference : ℕ := added_apps - deleted_apps)

theorem apps_difference (h1 : initial_apps = 115) (h2 : added_apps = 235) (h3 : apps_left = 178) : 
  difference = 63 := by
  sorry

end apps_difference_l2280_228070


namespace grains_of_rice_in_teaspoon_is_10_l2280_228024

noncomputable def grains_of_rice_per_teaspoon : ℕ :=
  let grains_per_cup := 480
  let tablespoons_per_half_cup := 8
  let teaspoons_per_tablespoon := 3
  grains_per_cup / (2 * tablespoons_per_half_cup * teaspoons_per_tablespoon)

theorem grains_of_rice_in_teaspoon_is_10 : grains_of_rice_per_teaspoon = 10 :=
by
  sorry

end grains_of_rice_in_teaspoon_is_10_l2280_228024


namespace sum_of_x_and_y_l2280_228014

-- Define the given angles
def angle_A : ℝ := 34
def angle_B : ℝ := 74
def angle_C : ℝ := 32

-- State the theorem
theorem sum_of_x_and_y (x y : ℝ) :
  (680 - x - y) = 720 → (x + y = 40) :=
by
  intro h
  sorry

end sum_of_x_and_y_l2280_228014


namespace general_term_formula_for_b_n_sum_of_first_n_terms_of_c_n_l2280_228093

def is_geometric_sequence (a : ℕ → ℤ) : Prop :=
  ∃ q : ℤ, ∀ n : ℕ, a (n + 1) = a n * q

def is_arithmetic_sequence (b : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d

def c_sequence (a b : ℕ → ℤ) (n : ℕ) : ℤ := a n - b n

def sum_c_sequence (c : ℕ → ℤ) (n : ℕ) : ℤ :=
  (Finset.range n).sum c

theorem general_term_formula_for_b_n (a b : ℕ → ℤ) (n : ℕ) 
  (h1 : is_geometric_sequence a)
  (h2 : is_arithmetic_sequence b)
  (h3 : a 1 = b 1)
  (h4 : a 2 = 3)
  (h5 : a 3 = 9)
  (h6 : a 4 = b 14) :
  b n = 2 * n - 1 :=
sorry

theorem sum_of_first_n_terms_of_c_n (a b : ℕ → ℤ) (n : ℕ)
  (h1 : is_geometric_sequence a)
  (h2 : is_arithmetic_sequence b)
  (h3 : a 1 = b 1)
  (h4 : a 2 = 3)
  (h5 : a 3 = 9)
  (h6 : a 4 = b 14)
  (h7 : ∀ n : ℕ, c_sequence a b n = a n - b n) :
  sum_c_sequence (c_sequence a b) n = (3 ^ n) / 2 - n ^ 2 - 1 / 2 :=
sorry

end general_term_formula_for_b_n_sum_of_first_n_terms_of_c_n_l2280_228093


namespace minimize_reciprocals_l2280_228082

theorem minimize_reciprocals (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 4 * b = 30) :
  (a = 10 ∧ b = 5) → ∀ x y : ℕ, (x > 0) → (y > 0) → (x + 4 * y = 30) → (1 / (x : ℝ) + 1 / (y : ℝ) ≥ 1 / 10 + 1 / 5) := 
by {
  sorry
}

end minimize_reciprocals_l2280_228082


namespace triangle_side_length_l2280_228094

noncomputable def sine (x : ℝ) : ℝ := Real.sin (x * Real.pi / 180) -- Define sine function explicitly (degrees to radians)

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ)
  (hA : A = 30) (hC : C = 45) (ha : a = 4) :
  c = 4 * Real.sqrt 2 := by
  sorry

end triangle_side_length_l2280_228094


namespace bicycle_spokes_count_l2280_228077

theorem bicycle_spokes_count (bicycles wheels spokes : ℕ) 
       (h1 : bicycles = 4) 
       (h2 : wheels = 2) 
       (h3 : spokes = 10) : 
       bicycles * (wheels * spokes) = 80 :=
by
  sorry

end bicycle_spokes_count_l2280_228077


namespace largest_shaded_area_figure_C_l2280_228088

noncomputable def area_of_square (s : ℝ) : ℝ := s^2
noncomputable def area_of_circle (r : ℝ) : ℝ := Real.pi * r^2
noncomputable def shaded_area_of_figure_A : ℝ := 4 - Real.pi
noncomputable def shaded_area_of_figure_B : ℝ := 4 - Real.pi
noncomputable def shaded_area_of_figure_C : ℝ := Real.pi - 2

theorem largest_shaded_area_figure_C : shaded_area_of_figure_C > shaded_area_of_figure_A ∧ shaded_area_of_figure_C > shaded_area_of_figure_B := by
  sorry

end largest_shaded_area_figure_C_l2280_228088


namespace sqrt_square_sub_sqrt2_l2280_228003

theorem sqrt_square_sub_sqrt2 (h : 1 < Real.sqrt 2) : Real.sqrt ((1 - Real.sqrt 2) ^ 2) = Real.sqrt 2 - 1 :=
by 
  sorry

end sqrt_square_sub_sqrt2_l2280_228003


namespace probability_of_johns_8th_roll_l2280_228066

noncomputable def probability_johns_8th_roll_is_last : ℚ :=
  (7/8)^6 * (1/8)

theorem probability_of_johns_8th_roll :
  probability_johns_8th_roll_is_last = 117649 / 2097152 := by
  sorry

end probability_of_johns_8th_roll_l2280_228066


namespace sum_first_9_terms_l2280_228015

variable (a b : ℕ → ℝ)
variable (S : ℕ → ℝ)

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop := ∀ m n k l, m + n = k + l → a m * a n = a k * a l
def geometric_prop (a : ℕ → ℝ) : Prop := a 3 * a 7 = 2 * a 5
def arithmetic_b5_eq_a5 (a b : ℕ → ℝ) : Prop := b 5 = a 5

-- The Sum Sn of an arithmetic sequence up to the nth terms
def arithmetic_sum (b : ℕ → ℝ) (S : ℕ → ℝ) : Prop := ∀ n, S n = (n / 2) * (b 1 + b n)

-- Question statement: proving the required sum
theorem sum_first_9_terms (a b : ℕ → ℝ) (S : ℕ → ℝ) 
  (hg : is_geometric_sequence a) 
  (hp : geometric_prop a) 
  (hb : arithmetic_b5_eq_a5 a b) 
  (arith_sum: arithmetic_sum b S) :
  S 9 = 18 :=
  sorry

end sum_first_9_terms_l2280_228015


namespace math_problem_l2280_228008

theorem math_problem 
  (x y : ℝ) 
  (h : x^2 + y^2 - x * y = 1) 
  : (-2 ≤ x + y) ∧ (x^2 + y^2 ≤ 2) :=
by
  sorry

end math_problem_l2280_228008


namespace airplane_altitude_l2280_228071

theorem airplane_altitude (d_Alice_Bob : ℝ) (angle_Alice : ℝ) (angle_Bob : ℝ) (altitude : ℝ) : 
  d_Alice_Bob = 8 ∧ angle_Alice = 45 ∧ angle_Bob = 30 → altitude = 16 / 3 :=
by
  intros h
  rcases h with ⟨h1, ⟨h2, h3⟩⟩
  -- you may insert the proof here if needed
  sorry

end airplane_altitude_l2280_228071


namespace yuna_has_biggest_number_l2280_228065

-- Define the numbers assigned to each student
def Yoongi_num : ℕ := 7
def Jungkook_num : ℕ := 6
def Yuna_num : ℕ := 9
def Yoojung_num : ℕ := 8

-- State the main theorem that Yuna has the biggest number
theorem yuna_has_biggest_number : 
  (Yuna_num = 9) ∧ (Yuna_num > Yoongi_num) ∧ (Yuna_num > Jungkook_num) ∧ (Yuna_num > Yoojung_num) :=
sorry

end yuna_has_biggest_number_l2280_228065


namespace total_number_of_people_l2280_228022

theorem total_number_of_people (c a : ℕ) (h1 : c = 2 * a) (h2 : c = 28) : c + a = 42 :=
by
  sorry

end total_number_of_people_l2280_228022


namespace problem_a_correct_answer_l2280_228009

def initial_digit_eq_six (n : ℕ) : Prop :=
∃ k a : ℕ, n = 6 * 10^k + a ∧ a = n / 25

theorem problem_a_correct_answer :
  ∀ n : ℕ, initial_digit_eq_six n ↔ ∃ m : ℕ, n = 625 * 10^m :=
by
  sorry

end problem_a_correct_answer_l2280_228009


namespace correct_calculation_result_l2280_228073

theorem correct_calculation_result (x : ℤ) (h : x - 32 = 25) : x - 23 = 34 :=
by
  sorry

end correct_calculation_result_l2280_228073


namespace kim_average_round_correct_answers_l2280_228044

theorem kim_average_round_correct_answers (x : ℕ) :
  (6 * 2) + (x * 3) + (4 * 5) = 38 → x = 2 :=
by
  intros h
  sorry

end kim_average_round_correct_answers_l2280_228044


namespace perfect_match_of_products_l2280_228092

theorem perfect_match_of_products
  (x : ℕ)  -- number of workers assigned to produce nuts
  (h1 : 22 - x ≥ 0)  -- ensuring non-negative number of workers for screws
  (h2 : 1200 * (22 - x) = 2 * 2000 * x) :  -- the condition for perfect matching
  (2 * 1200 * (22 - x) = 2000 * x) :=  -- the correct equation
by sorry

end perfect_match_of_products_l2280_228092


namespace library_books_l2280_228010

theorem library_books (N x y : ℕ) (h1 : x = N / 17) (h2 : y = x + 2000)
    (h3 : y = (N - 2 * 2000) / 15 + (14 * (N - 2000) / 17)): 
  N = 544000 := 
sorry

end library_books_l2280_228010


namespace find_T5_l2280_228074

variables (a b x y : ℝ)

def T (n : ℕ) : ℝ := a * x^n + b * y^n

theorem find_T5
  (h1 : T a b x y 1 = 3)
  (h2 : T a b x y 2 = 7)
  (h3 : T a b x y 3 = 6)
  (h4 : T a b x y 4 = 42) :
  T a b x y 5 = -360 :=
sorry

end find_T5_l2280_228074


namespace betsy_sewing_l2280_228029

-- Definitions of conditions
def total_squares : ℕ := 16 + 16
def sewn_percentage : ℝ := 0.25
def sewn_squares : ℝ := sewn_percentage * total_squares
def squares_left : ℝ := total_squares - sewn_squares

-- Proof that Betsy needs to sew 24 more squares
theorem betsy_sewing : squares_left = 24 := by
  -- Sorry placeholder for the actual proof
  sorry

end betsy_sewing_l2280_228029


namespace prob_heart_club_spade_l2280_228032

-- Definitions based on the conditions
def total_cards : ℕ := 52
def cards_per_suit : ℕ := 13

-- Definitions based on the question
def prob_first_heart : ℚ := cards_per_suit / total_cards
def prob_second_club : ℚ := cards_per_suit / (total_cards - 1)
def prob_third_spade : ℚ := cards_per_suit / (total_cards - 2)

-- The main proof statement to be proved
theorem prob_heart_club_spade :
  prob_first_heart * prob_second_club * prob_third_spade = 169 / 10200 :=
by
  sorry

end prob_heart_club_spade_l2280_228032


namespace prob_two_correct_prob_at_least_two_correct_prob_all_incorrect_l2280_228055

noncomputable def total_outcomes := 24
noncomputable def outcomes_two_correct := 6
noncomputable def outcomes_at_least_two_correct := 7
noncomputable def outcomes_all_incorrect := 9

theorem prob_two_correct : (outcomes_two_correct : ℚ) / total_outcomes = 1 / 4 := by
  sorry

theorem prob_at_least_two_correct : (outcomes_at_least_two_correct : ℚ) / total_outcomes = 7 / 24 := by
  sorry

theorem prob_all_incorrect : (outcomes_all_incorrect : ℚ) / total_outcomes = 3 / 8 := by
  sorry

end prob_two_correct_prob_at_least_two_correct_prob_all_incorrect_l2280_228055


namespace solution_set_no_pos_ab_l2280_228060

def f (x : ℝ) : ℝ := abs (x - 1) + abs (x - 2)

theorem solution_set :
  {x : ℝ | f x ≤ x + 1} = {x : ℝ | 2 / 3 ≤ x ∧ x ≤ 4} :=
by sorry

theorem no_pos_ab :
  ¬ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2 * a + b = 1 ∧ 1 / a + 2 / b = 4 :=
by sorry

end solution_set_no_pos_ab_l2280_228060


namespace molecular_weight_constant_l2280_228011

-- Given condition
def molecular_weight (compound : Type) : ℝ := 260

-- Proof problem statement (no proof yet)
theorem molecular_weight_constant (compound : Type) : molecular_weight compound = 260 :=
by
  sorry

end molecular_weight_constant_l2280_228011


namespace find_value_correct_l2280_228089

-- Definitions for the given conditions
def equation1 (a b : ℚ) : Prop := 3 * a - b = 8
def equation2 (a b : ℚ) : Prop := 4 * b + 7 * a = 13

-- Definition for the question
def find_value (a b : ℚ) : ℚ := 2 * a + b

-- Statement of the proof
theorem find_value_correct (a b : ℚ) (h1 : equation1 a b) (h2 : equation2 a b) : find_value a b = 73 / 19 := 
by 
  sorry

end find_value_correct_l2280_228089


namespace probability_even_sum_of_selected_envelopes_l2280_228047

theorem probability_even_sum_of_selected_envelopes :
  let face_values := [5, 6, 8, 10]
  let possible_sum_is_even (s : ℕ) : Prop := s % 2 = 0
  let num_combinations := Nat.choose 4 2
  let favorable_combinations := 3
  (favorable_combinations / num_combinations : ℚ) = 1 / 2 :=
by
  sorry

end probability_even_sum_of_selected_envelopes_l2280_228047


namespace power_sum_eq_l2280_228017

theorem power_sum_eq : (-2)^2011 + (-2)^2012 = 2^2011 := by
  sorry

end power_sum_eq_l2280_228017


namespace k_squared_geq_25_div_3_l2280_228069

open Real

theorem k_squared_geq_25_div_3 
  (a₁ a₂ a₃ a₄ a₅ k : ℝ)
  (h₁₂ : abs (a₁ - a₂) ≥ 1) (h₁₃ : abs (a₁ - a₃) ≥ 1) (h₁₄ : abs (a₁ - a₄) ≥ 1) (h₁₅ : abs (a₁ - a₅) ≥ 1)
  (h₂₃ : abs (a₂ - a₃) ≥ 1) (h₂₄ : abs (a₂ - a₄) ≥ 1) (h₂₅ : abs (a₂ - a₅) ≥ 1)
  (h₃₄ : abs (a₃ - a₄) ≥ 1) (h₃₅ : abs (a₃ - a₅) ≥ 1)
  (h₄₅ : abs (a₄ - a₅) ≥ 1)
  (eq1 : a₁ + a₂ + a₃ + a₄ + a₅ = 2 * k)
  (eq2 : a₁^2 + a₂^2 + a₃^2 + a₄^2 + a₅^2 = 2 * k^2) :
  k^2 ≥ 25 / 3 :=
by
  sorry

end k_squared_geq_25_div_3_l2280_228069


namespace remainder_when_dividing_polynomial_by_x_minus_3_l2280_228042

noncomputable def P (x : ℤ) : ℤ := 
  2 * x^8 - 3 * x^7 + 4 * x^6 - x^4 + 6 * x^3 - 5 * x^2 + 18 * x - 20

theorem remainder_when_dividing_polynomial_by_x_minus_3 :
  P 3 = 17547 :=
by
  sorry

end remainder_when_dividing_polynomial_by_x_minus_3_l2280_228042


namespace polynomial_not_33_l2280_228004

theorem polynomial_not_33 (x y : ℤ) : x^5 + 3 * x^4 * y - 5 * x^3 * y^2 - 15 * x^2 * y^3 + 4 * x * y^4 + 12 * y^5 ≠ 33 := 
sorry

end polynomial_not_33_l2280_228004


namespace discriminant_of_quadratic_equation_l2280_228039

noncomputable def discriminant (a b c : ℤ) : ℤ :=
  b^2 - 4 * a * c

theorem discriminant_of_quadratic_equation : discriminant 5 (-11) (-18) = 481 := by
  sorry

end discriminant_of_quadratic_equation_l2280_228039


namespace simplify_fraction_l2280_228080

theorem simplify_fraction (a b c d : ℕ) (h₁ : a = 2) (h₂ : b = 462) (h₃ : c = 29) (h₄ : d = 42) :
  (a : ℚ) / (b : ℚ) + (c : ℚ) / (d : ℚ) = 107 / 154 :=
by {
  sorry
}

end simplify_fraction_l2280_228080


namespace butterfly_flutters_total_distance_l2280_228041

-- Define the conditions
def start_pos : ℤ := 0
def first_move : ℤ := 4
def second_move : ℤ := -3
def third_move : ℤ := 7

-- Define a function that calculates the total distance
def total_distance (xs : List ℤ) : ℤ :=
  List.sum (List.map (fun ⟨x, y⟩ => abs (y - x)) (xs.zip xs.tail))

-- Create the butterfly's path
def path : List ℤ := [start_pos, first_move, second_move, third_move]

-- Define the proposition that we need to prove
theorem butterfly_flutters_total_distance : total_distance path = 21 := sorry

end butterfly_flutters_total_distance_l2280_228041


namespace min_distance_of_complex_numbers_l2280_228085

open Complex

theorem min_distance_of_complex_numbers
  (z w : ℂ)
  (h₁ : abs (z + 1 + 3 * Complex.I) = 1)
  (h₂ : abs (w - 7 - 8 * Complex.I) = 3) :
  ∃ d, d = Real.sqrt 185 - 4 ∧ ∀ Z W : ℂ, abs (Z + 1 + 3 * Complex.I) = 1 → abs (W - 7 - 8 * Complex.I) = 3 → abs (Z - W) ≥ d :=
sorry

end min_distance_of_complex_numbers_l2280_228085


namespace fifth_selected_ID_is_01_l2280_228097

noncomputable def populationIDs : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

noncomputable def randomNumberTable : List (List ℕ) :=
  [[78, 16, 65, 72,  8, 2, 63, 14,  7, 2, 43, 69, 97, 28,  1, 98],
   [32,  4, 92, 34, 49, 35, 82,  0, 36, 23, 48, 69, 69, 38, 74, 81]]

noncomputable def selectedIDs (table : List (List ℕ)) : List ℕ :=
  [8, 2, 14, 7, 1]  -- Derived from the selection method

theorem fifth_selected_ID_is_01 : (selectedIDs randomNumberTable).get! 4 = 1 := by
  sorry

end fifth_selected_ID_is_01_l2280_228097
