import Mathlib

namespace no_primes_divisible_by_60_l168_168787

theorem no_primes_divisible_by_60 (p : ℕ) (prime_p : Nat.Prime p) : ¬ (60 ∣ p) :=
by
  sorry

end no_primes_divisible_by_60_l168_168787


namespace max_covered_squares_l168_168103

-- Definitions representing the conditions
def checkerboard_squares : ℕ := 1 -- side length of each square on the checkerboard
def card_side_len : ℕ := 2 -- side length of the card

-- Theorem statement representing the question and answer
theorem max_covered_squares : ∀ n, 
  (∃ board_side squared_len, 
    checkerboard_squares = 1 ∧ card_side_len = 2 ∧
    (board_side = checkerboard_squares ∧ squared_len = card_side_len) ∧
    n ≤ 16) →
  n = 16 :=
  sorry

end max_covered_squares_l168_168103


namespace simplify_expression_l168_168996

theorem simplify_expression (x y z : ℝ) : ((x + y) - (z - y)) - ((x + z) - (y + z)) = 3 * y - z := by
  sorry

end simplify_expression_l168_168996


namespace remainder_when_eight_n_plus_five_divided_by_eleven_l168_168217

theorem remainder_when_eight_n_plus_five_divided_by_eleven
  (n : ℤ) (h : n % 11 = 4) : (8 * n + 5) % 11 = 4 := 
  sorry

end remainder_when_eight_n_plus_five_divided_by_eleven_l168_168217


namespace average_speed_round_trip_l168_168593

theorem average_speed_round_trip (D : ℝ) (hD : D > 0) :
  let time_uphill := D / 5
  let time_downhill := D / 100
  let total_distance := 2 * D
  let total_time := time_uphill + time_downhill
  let average_speed := total_distance / total_time
  average_speed = 200 / 21 :=
by
  sorry

end average_speed_round_trip_l168_168593


namespace radius_of_sector_l168_168653

theorem radius_of_sector (l : ℝ) (α : ℝ) (R : ℝ) (h1 : l = 2 * π / 3) (h2 : α = π / 3) : R = 2 := by
  have : l = |α| * R := by sorry
  rw [h1, h2] at this
  sorry

end radius_of_sector_l168_168653


namespace half_guests_want_two_burgers_l168_168435

theorem half_guests_want_two_burgers 
  (total_guests : ℕ) (half_guests : ℕ)
  (time_per_side : ℕ) (time_per_burger : ℕ)
  (grill_capacity : ℕ) (total_time : ℕ)
  (guests_one_burger : ℕ) (total_burgers : ℕ) : 
  total_guests = 30 →
  time_per_side = 4 →
  time_per_burger = 8 →
  grill_capacity = 5 →
  total_time = 72 →
  guests_one_burger = 15 →
  total_burgers = 45 →
  half_guests * 2 = total_burgers - guests_one_burger →
  half_guests = 15 :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end half_guests_want_two_burgers_l168_168435


namespace price_decrease_is_50_percent_l168_168072

-- Original price is 50 yuan
def original_price : ℝ := 50

-- Price after 100% increase
def increased_price : ℝ := original_price * (1 + 1)

-- Required percentage decrease to return to original price
def required_percentage_decrease (x : ℝ) : ℝ := increased_price * (1 - x)

theorem price_decrease_is_50_percent : required_percentage_decrease 0.5 = 50 :=
  by 
    sorry

end price_decrease_is_50_percent_l168_168072


namespace largest_possible_red_socks_l168_168929

theorem largest_possible_red_socks (t r g : ℕ) (h1 : t = r + g) (h2 : t ≤ 3000)
    (h3 : (r * (r - 1) + g * (g - 1)) * 5 = 3 * t * (t - 1)) :
    r ≤ 1199 :=
sorry

end largest_possible_red_socks_l168_168929


namespace percent_increase_of_income_l168_168239

theorem percent_increase_of_income (original_income new_income : ℝ) 
  (h1 : original_income = 120) (h2 : new_income = 180) :
  ((new_income - original_income) / original_income) * 100 = 50 := 
by 
  rw [h1, h2]
  norm_num

end percent_increase_of_income_l168_168239


namespace base5_div_l168_168348

-- Definitions for base 5 numbers
def n1 : ℕ := (2 * 125) + (4 * 25) + (3 * 5) + 4  -- 2434_5 in base 10 is 369
def n2 : ℕ := (1 * 25) + (3 * 5) + 2              -- 132_5 in base 10 is 42
def d  : ℕ := (2 * 5) + 1                          -- 21_5 in base 10 is 11

theorem base5_div (res : ℕ) : res = (122 : ℕ) → (n1 + n2) / d = res :=
by sorry

end base5_div_l168_168348


namespace John_spending_l168_168569

theorem John_spending
  (X : ℝ)
  (h1 : (1/2) * X + (1/3) * X + (1/10) * X + 10 = X) :
  X = 150 :=
by
  sorry

end John_spending_l168_168569


namespace minimum_value_l168_168786

noncomputable def function_y (x : ℝ) : ℝ := 3 * x^2 - 12 * x + 1450

theorem minimum_value : ∀ x : ℝ, function_y x ≥ 1438 :=
by 
  intro x
  sorry

end minimum_value_l168_168786


namespace alexis_suit_coat_expense_l168_168458

theorem alexis_suit_coat_expense :
  let budget := 200
  let shirt_cost := 30
  let pants_cost := 46
  let socks_cost := 11
  let belt_cost := 18
  let shoes_cost := 41
  let leftover := 16
  let other_expenses := shirt_cost + pants_cost + socks_cost + belt_cost + shoes_cost
  budget - leftover - other_expenses = 38 := 
by
  let budget := 200
  let shirt_cost := 30
  let pants_cost := 46
  let socks_cost := 11
  let belt_cost := 18
  let shoes_cost := 41
  let leftover := 16
  let other_expenses := shirt_cost + pants_cost + socks_cost + belt_cost + shoes_cost
  sorry

end alexis_suit_coat_expense_l168_168458


namespace problem1_problem2_problem3_l168_168438

-- Problem 1
theorem problem1 : 13 + (-7) - (-9) + 5 * (-2) = 5 :=
by 
  sorry

-- Problem 2
theorem problem2 : abs (-7 / 2) * (12 / 7) / (4 / 3) / (3 ^ 2) = 1 / 2 :=
by 
  sorry

-- Problem 3
theorem problem3 : -1^4 - (1 / 6) * (2 - (-3)^2) = 1 / 6 :=
by 
  sorry

end problem1_problem2_problem3_l168_168438


namespace total_fruits_consumed_l168_168730

def starting_cherries : ℝ := 16.5
def remaining_cherries : ℝ := 6.3

def starting_strawberries : ℝ := 10.7
def remaining_strawberries : ℝ := 8.4

def starting_blueberries : ℝ := 20.2
def remaining_blueberries : ℝ := 15.5

theorem total_fruits_consumed 
  (sc : ℝ := starting_cherries)
  (rc : ℝ := remaining_cherries)
  (ss : ℝ := starting_strawberries)
  (rs : ℝ := remaining_strawberries)
  (sb : ℝ := starting_blueberries)
  (rb : ℝ := remaining_blueberries) :
  (sc - rc) + (ss - rs) + (sb - rb) = 17.2 := by
  sorry

end total_fruits_consumed_l168_168730


namespace find_x_given_scores_l168_168628

theorem find_x_given_scores : 
  ∃ x : ℝ, (9.1 + 9.3 + x + 9.2 + 9.4) / 5 = 9.3 ∧ x = 9.5 :=
by {
  sorry
}

end find_x_given_scores_l168_168628


namespace fundraising_part1_fundraising_part2_l168_168732

-- Problem 1
theorem fundraising_part1 (x y : ℕ) 
(h1 : x + y = 60) 
(h2 : 100 * x + 80 * y = 5600) :
x = 40 ∧ y = 20 := 
by 
  sorry

-- Problem 2
theorem fundraising_part2 (a : ℕ) 
(h1 : 100 * a + 80 * (80 - a) ≤ 6890) :
a ≤ 24 := 
by 
  sorry

end fundraising_part1_fundraising_part2_l168_168732


namespace flowers_count_l168_168181

theorem flowers_count (lilies : ℕ) (sunflowers : ℕ) (daisies : ℕ) (total_flowers : ℕ) (roses : ℕ)
  (h1 : lilies = 40) (h2 : sunflowers = 40) (h3 : daisies = 40) (h4 : total_flowers = 160) :
  lilies + sunflowers + daisies + roses = 160 → roses = 40 := 
by
  sorry

end flowers_count_l168_168181


namespace problem_solution_l168_168288

def p (x : ℝ) : ℝ := x^2 - 4*x + 3
def tilde_p (x : ℝ) : ℝ := p (p x)

-- Proof problem: Prove tilde_p 2 = -4 
theorem problem_solution : tilde_p 2 = -4 := sorry

end problem_solution_l168_168288


namespace jordan_weight_after_exercise_l168_168824

theorem jordan_weight_after_exercise :
  let initial_weight := 250
  let weight_loss_4_weeks := 3 * 4
  let weight_loss_8_weeks := 2 * 8
  let total_weight_loss := weight_loss_4_weeks + weight_loss_8_weeks
  let current_weight := initial_weight - total_weight_loss
  current_weight = 222 :=
by
  let initial_weight := 250
  let weight_loss_4_weeks := 3 * 4
  let weight_loss_8_weeks := 2 * 8
  let total_weight_loss := weight_loss_4_weeks + weight_loss_8_weeks
  let current_weight := initial_weight - total_weight_loss
  show current_weight = 222
  sorry

end jordan_weight_after_exercise_l168_168824


namespace sheila_initial_savings_l168_168762

noncomputable def initial_savings (monthly_savings : ℕ) (years : ℕ) (family_addition : ℕ) (total_amount : ℕ) : ℕ :=
  total_amount - (monthly_savings * 12 * years + family_addition)

def sheila_initial_savings_proof : Prop :=
  initial_savings 276 4 7000 23248 = 3000

theorem sheila_initial_savings : sheila_initial_savings_proof :=
  by
    -- Proof goes here
    sorry

end sheila_initial_savings_l168_168762


namespace slices_per_person_l168_168672

theorem slices_per_person (total_slices : ℕ) (total_people : ℕ) (h_slices : total_slices = 12) (h_people : total_people = 3) :
  total_slices / total_people = 4 :=
by
  sorry

end slices_per_person_l168_168672


namespace evaluate_expression_l168_168714

theorem evaluate_expression (x y z : ℝ) (hxy : x > y ∧ y > 1) (hz : z > 0) :
  (x^y * y^(x+z)) / (y^(y+z) * x^x) = (x / y)^(y - x) :=
by
  sorry

end evaluate_expression_l168_168714


namespace negation_P_l168_168818

-- Define the proposition P
def P (m : ℤ) : Prop := ∃ x : ℤ, 2 * x^2 + x + m ≤ 0

-- Define the negation of the proposition P
theorem negation_P (m : ℤ) : ¬P m ↔ ∀ x : ℤ, 2 * x^2 + x + m > 0 :=
by
  sorry

end negation_P_l168_168818


namespace monkey_reaches_tree_top_in_hours_l168_168269

-- Definitions based on conditions
def height_of_tree : ℕ := 22
def hop_per_hour : ℕ := 3
def slip_per_hour : ℕ := 2
def effective_climb_per_hour : ℕ := hop_per_hour - slip_per_hour

-- The theorem we want to prove
theorem monkey_reaches_tree_top_in_hours
  (height_of_tree hop_per_hour slip_per_hour : ℕ)
  (h1 : height_of_tree = 22)
  (h2 : hop_per_hour = 3)
  (h3 : slip_per_hour = 2) :
  ∃ t : ℕ, t = 22 ∧ effective_climb_per_hour * (t - 1) + hop_per_hour = height_of_tree := by
  sorry

end monkey_reaches_tree_top_in_hours_l168_168269


namespace find_number_l168_168537

theorem find_number (x : ℝ) (h : 0.5 * x = 0.1667 * x + 10) : x = 30 :=
sorry

end find_number_l168_168537


namespace planes_divide_space_l168_168505

-- Definition of a triangular prism
def triangular_prism (V : Type) (P : Set (Set V)) : Prop :=
  ∃ (A B C D E F : V),
    P = {{A, B, C}, {D, E, F}, {A, B, D, E}, {B, C, E, F}, {C, A, F, D}}

-- The condition: planes containing the faces of a triangular prism
def planes_containing_faces (V : Type) (P : Set (Set V)) : Prop :=
  triangular_prism V P

-- Proof statement: The planes containing the faces of a triangular prism divide the space into 21 parts
theorem planes_divide_space (V : Type) (P : Set (Set V))
  (h : planes_containing_faces V P) :
  ∃ parts : ℕ, parts = 21 := by
  sorry

end planes_divide_space_l168_168505


namespace evaluate_expression_l168_168652

theorem evaluate_expression : 
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) = 5^128 - 4^128 :=
by sorry

end evaluate_expression_l168_168652


namespace lana_average_speed_l168_168220

theorem lana_average_speed (initial_reading : ℕ) (final_reading : ℕ) (time_first_day : ℕ) (time_second_day : ℕ) :
  initial_reading = 1991 → 
  final_reading = 2332 → 
  time_first_day = 5 → 
  time_second_day = 7 → 
  (final_reading - initial_reading) / (time_first_day + time_second_day : ℝ) = 28.4 :=
by
  intros h_init h_final h_first h_second
  rw [h_init, h_final, h_first, h_second]
  norm_num
  sorry

end lana_average_speed_l168_168220


namespace even_function_f3_l168_168650

theorem even_function_f3 (a : ℝ) (h : ∀ x : ℝ, (x + 2) * (x - a) = (-x + 2) * (-x - a)) : (3 + 2) * (3 - a) = 5 := by
  sorry

end even_function_f3_l168_168650


namespace std_deviation_calc_l168_168417

theorem std_deviation_calc 
  (μ : ℝ) (σ : ℝ) (V : ℝ) (k : ℝ)
  (hμ : μ = 14.0)
  (hσ : σ = 1.5)
  (hV : V = 11)
  (hk : k = (μ - V) / σ) :
  k = 2 := by
  sorry

end std_deviation_calc_l168_168417


namespace subset_123_12_false_l168_168922

-- Definitions derived from conditions
def is_int (x : ℤ) := true
def subset_123_12 (A B : Set ℕ) := A = {1, 2, 3} ∧ B = {1, 2}
def intersection_empty {A B : Set ℕ} (hA : A = {1, 2}) (hB : B = ∅) := (A ∩ B = ∅)
def union_nat_real {A B : Set ℝ} (hA : Set.univ ⊆ A) (hB : Set.univ ⊆ B) := (A ∪ B)

-- The mathematically equivalent proof problem
theorem subset_123_12_false (A B : Set ℕ) (hA : A = {1, 2, 3}) (hB : B = {1, 2}):
  ¬ (A ⊆ B) :=
by
  sorry

end subset_123_12_false_l168_168922


namespace aram_fraction_of_fine_l168_168249

theorem aram_fraction_of_fine (F : ℝ) (H1 : Joe_paid = (1/4)*F + 3)
  (H2 : Peter_paid = (1/3)*F - 3)
  (H3 : Aram_paid = (1/2)*F - 4)
  (H4 : Joe_paid + Peter_paid + Aram_paid = F) : 
  Aram_paid / F = 5 / 12 := 
sorry

end aram_fraction_of_fine_l168_168249


namespace container_volume_ratio_l168_168928

theorem container_volume_ratio
  (A B : ℚ)
  (H1 : 3/5 * A + 1/4 * B = 4/5 * B)
  (H2 : 3/5 * A = (4/5 * B - 1/4 * B)) :
  A / B = 11 / 12 :=
by
  sorry

end container_volume_ratio_l168_168928


namespace max_area_rectangle_l168_168976

theorem max_area_rectangle (P : ℕ) (hP : P = 40) : ∃ A : ℕ, A = 100 ∧ ∀ x y : ℕ, 2 * (x + y) = P → x * y ≤ A := by
  sorry

end max_area_rectangle_l168_168976


namespace problem_l168_168905

noncomputable def M (x y z : ℝ) : ℝ :=
  (Real.sqrt (x^2 + x * y + y^2) * Real.sqrt (y^2 + y * z + z^2)) +
  (Real.sqrt (y^2 + y * z + z^2) * Real.sqrt (z^2 + z * x + x^2)) +
  (Real.sqrt (z^2 + z * x + x^2) * Real.sqrt (x^2 + x * y + y^2))

theorem problem (x y z : ℝ) (α β : ℝ) 
  (h1 : ∀ x y z, α * (x * y + y * z + z * x) ≤ M x y z)
  (h2 : ∀ x y z, M x y z ≤ β * (x^2 + y^2 + z^2)) :
  (∀ α, α ≤ 3) ∧ (∀ β, β ≥ 3) :=
sorry

end problem_l168_168905


namespace product_of_solutions_of_abs_equation_l168_168298

theorem product_of_solutions_of_abs_equation :
  (∃ x₁ x₂ : ℚ, |5 * x₁ - 2| + 7 = 52 ∧ |5 * x₂ - 2| + 7 = 52 ∧ x₁ ≠ x₂ ∧ (x₁ * x₂ = -2021 / 25)) :=
sorry

end product_of_solutions_of_abs_equation_l168_168298


namespace value_of_b_l168_168209

def g (x : ℝ) : ℝ := 5 * x - 6

theorem value_of_b (b : ℝ) : g b = 0 ↔ b = 6 / 5 :=
by sorry

end value_of_b_l168_168209


namespace power_inequality_l168_168964

theorem power_inequality (a b c : ℕ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hcb : c ≥ b) : 
  a^b * (a + b)^c > c^b * a^c := 
sorry

end power_inequality_l168_168964


namespace sugar_initial_weight_l168_168880

theorem sugar_initial_weight (packs : ℕ) (pack_weight : ℕ) (leftover : ℕ) (used_percentage : ℝ)
  (h1 : packs = 30)
  (h2 : pack_weight = 350)
  (h3 : leftover = 50)
  (h4 : used_percentage = 0.60) : 
  (packs * pack_weight + leftover) = 10550 :=
by 
  sorry

end sugar_initial_weight_l168_168880


namespace termite_ridden_not_collapsing_l168_168310

theorem termite_ridden_not_collapsing
  (total_homes : ℕ)
  (termite_ridden_fraction : ℚ)
  (collapsing_fraction_of_termite_ridden : ℚ)
  (h1 : termite_ridden_fraction = 1/3)
  (h2 : collapsing_fraction_of_termite_ridden = 1/4) :
  (termite_ridden_fraction - (termite_ridden_fraction * collapsing_fraction_of_termite_ridden)) = 1/4 := 
by {
  sorry
}

end termite_ridden_not_collapsing_l168_168310


namespace perp_lines_value_of_m_parallel_lines_value_of_m_l168_168474

theorem perp_lines_value_of_m (m : ℝ) : 
  (∀ x y : ℝ, x + m * y + 6 = 0) ∧ (∀ x y : ℝ, (m - 2) * x + 3 * y + 2 * m = 0) →
  (m ≠ 0) →
  (∀ x y : ℝ, (x + m * y + 6 = 0) → (∃ x' y' : ℝ, (m - 2) * x' + 3 * y' + 2 * m = 0) → 
  (∀ x y x' y' : ℝ, -((1 : ℝ) / m) * ((m - 2) / 3) = -1)) → 
  m = 1 / 2 := 
sorry

theorem parallel_lines_value_of_m (m : ℝ) : 
  (∀ x y : ℝ, x + m * y + 6 = 0) ∧ (∀ x y : ℝ, (m - 2) * x + 3 * y + 2 * m = 0) →
  (m ≠ 0) →
  (∀ x y : ℝ, (x + m * y + 6 = 0) → (∃ x' y' : ℝ, (m - 2) * x' + 3 * y' + 2 * m = 0) → 
  (∀ x y x' y' : ℝ, -((1 : ℝ) / m) = ((m - 2) / 3))) → 
  m = -1 := 
sorry

end perp_lines_value_of_m_parallel_lines_value_of_m_l168_168474


namespace roshini_sweets_cost_correct_l168_168154

noncomputable def roshini_sweet_cost_before_discounts_and_tax : ℝ := 10.54

theorem roshini_sweets_cost_correct (R F1 F2 F3 : ℝ) (h1 : R + F1 + F2 + F3 = 10.54)
    (h2 : R * 0.9 = (10.50 - 9.20) / 1.08)
    (h3 : F1 + F2 + F3 = 3.40 + 4.30 + 1.50) :
    R + F1 + F2 + F3 = roshini_sweet_cost_before_discounts_and_tax :=
by
  sorry

end roshini_sweets_cost_correct_l168_168154


namespace marge_funds_for_fun_l168_168852

-- Definitions based on given conditions
def lottery_amount : ℕ := 12006
def taxes_paid : ℕ := lottery_amount / 2
def remaining_after_taxes : ℕ := lottery_amount - taxes_paid
def student_loans_paid : ℕ := remaining_after_taxes / 3
def remaining_after_loans : ℕ := remaining_after_taxes - student_loans_paid
def savings : ℕ := 1000
def remaining_after_savings : ℕ := remaining_after_loans - savings
def stock_market_investment : ℕ := savings / 5
def remaining_after_investment : ℕ := remaining_after_savings - stock_market_investment

-- The proof goal
theorem marge_funds_for_fun : remaining_after_investment = 2802 :=
sorry

end marge_funds_for_fun_l168_168852


namespace sum_of_integers_from_neg15_to_5_l168_168513

-- defining the conditions
def first_term : ℤ := -15
def last_term : ℤ := 5

-- sum of integers from first_term to last_term
def sum_arithmetic_series (a l : ℤ) : ℤ :=
  let n := l - a + 1
  (n * (a + l)) / 2

-- the statement we need to prove
theorem sum_of_integers_from_neg15_to_5 : sum_arithmetic_series first_term last_term = -105 := by
  sorry

end sum_of_integers_from_neg15_to_5_l168_168513


namespace find_D_double_prime_l168_168966

def reflectY (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

def translateUp1 (p : ℝ × ℝ) : ℝ × ℝ := (p.1, p.2 + 1)

def reflectYeqX (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

def translateDown1 (p : ℝ × ℝ) : ℝ × ℝ := (p.1, p.2 - 1)

def D'' (D : ℝ × ℝ) : ℝ × ℝ :=
  translateDown1 (reflectYeqX (translateUp1 (reflectY D)))

theorem find_D_double_prime :
  let D := (5, 0)
  D'' D = (-1, 4) :=
by
  sorry

end find_D_double_prime_l168_168966


namespace original_square_perimeter_l168_168369

-- Define the problem statement
theorem original_square_perimeter (P_perimeter : ℕ) (hP : P_perimeter = 56) : 
  ∃ sq_perimeter : ℕ, sq_perimeter = 32 := 
by 
  sorry

end original_square_perimeter_l168_168369


namespace five_digit_numbers_last_two_different_l168_168541

def total_five_digit_numbers : ℕ := 90000

def five_digit_numbers_last_two_same : ℕ := 9000

theorem five_digit_numbers_last_two_different :
  (total_five_digit_numbers - five_digit_numbers_last_two_same) = 81000 := 
by 
  sorry

end five_digit_numbers_last_two_different_l168_168541


namespace sum_of_first_six_primes_mod_seventh_prime_l168_168462

open Int

theorem sum_of_first_six_primes_mod_seventh_prime :
  ((2 + 3 + 5 + 7 + 11 + 13) % 17) = 7 := 
  sorry

end sum_of_first_six_primes_mod_seventh_prime_l168_168462


namespace teacher_total_score_l168_168149

/-- Conditions -/
def written_test_score : ℝ := 80
def interview_score : ℝ := 60
def written_test_weight : ℝ := 0.6
def interview_weight : ℝ := 0.4

/-- Prove the total score -/
theorem teacher_total_score :
  written_test_score * written_test_weight + interview_score * interview_weight = 72 :=
by
  sorry

end teacher_total_score_l168_168149


namespace problem_statement_l168_168606

noncomputable def f (x : ℝ) : ℝ := x / Real.cos x

theorem problem_statement (x1 x2 x3 : ℝ) (h1 : abs x1 < Real.pi / 2)
                         (h2 : abs x2 < Real.pi / 2) (h3 : abs x3 < Real.pi / 2)
                         (c1 : f x1 + f x2 ≥ 0) (c2 : f x2 + f x3 ≥ 0) (c3 : f x3 + f x1 ≥ 0) :
  f (x1 + x2 + x3) ≥ 0 :=
sorry

end problem_statement_l168_168606


namespace linda_color_choices_l168_168546

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def combination (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem linda_color_choices : combination 8 3 = 56 :=
  by sorry

end linda_color_choices_l168_168546


namespace four_divides_sum_of_squares_iff_even_l168_168141

theorem four_divides_sum_of_squares_iff_even (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (4 ∣ (a^2 + b^2 + c^2)) ↔ (Even a ∧ Even b ∧ Even c) :=
by
  sorry

end four_divides_sum_of_squares_iff_even_l168_168141


namespace wheat_acres_l168_168155

theorem wheat_acres (x y : ℤ) 
  (h1 : x + y = 4500) 
  (h2 : 42 * x + 35 * y = 165200) : 
  y = 3400 :=
sorry

end wheat_acres_l168_168155


namespace range_of_a_l168_168164

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x + 3| - |x + 2| ≥ Real.log a / Real.log 2) ↔ (0 < a ∧ a ≤ 2) :=
by
  sorry

end range_of_a_l168_168164


namespace find_base_k_l168_168763

theorem find_base_k (k : ℕ) (hk : 0 < k) (h : 7/51 = (2 * k + 3) / (k^2 - 1)) : k = 16 :=
sorry

end find_base_k_l168_168763


namespace arithmetic_expression_value_l168_168059

theorem arithmetic_expression_value : 4 * (8 - 3 + 2) - 7 = 21 := by
  sorry

end arithmetic_expression_value_l168_168059


namespace max_area_isosceles_triangle_l168_168969

theorem max_area_isosceles_triangle (b : ℝ) (h : ℝ) (area : ℝ) 
  (h_cond : h^2 = 1 - b^2 / 4)
  (area_def : area = 1 / 2 * b * h) : 
  area ≤ 2 * Real.sqrt 2 / 3 := 
sorry

end max_area_isosceles_triangle_l168_168969


namespace solution_set_x2_f_x_positive_l168_168273

noncomputable def f : ℝ → ℝ := sorry
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_at_2 : f 2 = 0
axiom derivative_condition : ∀ x, x > 0 → ((x * (deriv f x) - f x) / x^2) > 0

theorem solution_set_x2_f_x_positive :
  {x : ℝ | x^2 * f x > 0} = {x : ℝ | -2 < x ∧ x < 0} ∪ {x : ℝ | x > 2} :=
sorry

end solution_set_x2_f_x_positive_l168_168273


namespace correct_growth_rate_equation_l168_168347

-- Define the conditions
def packages_first_day := 200
def packages_third_day := 242

-- Define the average daily growth rate
variable (x : ℝ)

-- State the theorem to prove
theorem correct_growth_rate_equation :
  packages_first_day * (1 + x)^2 = packages_third_day :=
by
  sorry

end correct_growth_rate_equation_l168_168347


namespace interest_received_l168_168817

theorem interest_received
  (total_investment : ℝ)
  (part_invested_6 : ℝ)
  (rate_6 : ℝ)
  (rate_9 : ℝ) :
  part_invested_6 = 7200 →
  rate_6 = 0.06 →
  rate_9 = 0.09 →
  total_investment = 10000 →
  (total_investment - part_invested_6) * rate_9 + part_invested_6 * rate_6 = 684 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end interest_received_l168_168817


namespace volume_and_surface_area_of_prism_l168_168256

theorem volume_and_surface_area_of_prism 
  (a b c : ℝ)
  (h1 : a * b = 24)
  (h2 : b * c = 18)
  (h3 : c * a = 12) :
  (a * b * c = 72) ∧ (2 * (a * b + b * c + c * a) = 108) := by
  sorry

end volume_and_surface_area_of_prism_l168_168256


namespace smallest_positive_integer_cube_ends_in_632_l168_168355

theorem smallest_positive_integer_cube_ends_in_632 :
  ∃ n : ℕ, (n > 0) ∧ (n^3 % 1000 = 632) ∧ ∀ m : ℕ, (m > 0) ∧ (m^3 % 1000 = 632) → n ≤ m := 
sorry

end smallest_positive_integer_cube_ends_in_632_l168_168355


namespace puppies_per_cage_calculation_l168_168158

noncomputable def initial_puppies : ℝ := 18.0
noncomputable def additional_puppies : ℝ := 3.0
noncomputable def total_puppies : ℝ := initial_puppies + additional_puppies
noncomputable def total_cages : ℝ := 4.2
noncomputable def puppies_per_cage : ℝ := total_puppies / total_cages

theorem puppies_per_cage_calculation :
  puppies_per_cage = 5.0 :=
by
  sorry

end puppies_per_cage_calculation_l168_168158


namespace hyperbola_center_l168_168067

theorem hyperbola_center (x y : ℝ) :
  (∃ h k, h = 2 ∧ k = -1 ∧ 
    (∀ x y, (3 * y + 3)^2 / 7^2 - (4 * x - 8)^2 / 6^2 = 1 ↔ 
      (y - (-1))^2 / ((7 / 3)^2) - (x - 2)^2 / ((3 / 2)^2) = 1)) :=
by sorry

end hyperbola_center_l168_168067


namespace prob_first_red_light_third_intersection_l168_168078

noncomputable def red_light_at_third_intersection (p : ℝ) (h : p = 2/3) : ℝ :=
(1 - p) * (1 - (1/2)) * (1/2)

theorem prob_first_red_light_third_intersection (h : 2/3 = (2/3 : ℝ)) :
  red_light_at_third_intersection (2/3) h = 1/12 := sorry

end prob_first_red_light_third_intersection_l168_168078


namespace area_of_regular_octagon_l168_168904

/-- The perimeters of a square and a regular octagon are equal.
    The area of the square is 16.
    Prove that the area of the regular octagon is 8 + 8 * sqrt 2. -/
theorem area_of_regular_octagon (a b : ℝ) (h1 : 4 * a = 8 * b) (h2 : a^2 = 16) :
  2 * (1 + Real.sqrt 2) * b^2 = 8 + 8 * Real.sqrt 2 :=
by
  sorry

end area_of_regular_octagon_l168_168904


namespace solve_remainder_l168_168831

theorem solve_remainder (y : ℤ) 
  (hc1 : y + 4 ≡ 9 [ZMOD 3^3])
  (hc2 : y + 4 ≡ 16 [ZMOD 5^3])
  (hc3 : y + 4 ≡ 36 [ZMOD 7^3]) : 
  y ≡ 32 [ZMOD 105] :=
by
  sorry

end solve_remainder_l168_168831


namespace jimmy_earnings_l168_168303

theorem jimmy_earnings : 
  let price15 := 15
  let price20 := 20
  let discount := 5
  let sale_price15 := price15 - discount
  let sale_price20 := price20 - discount
  let num_low_worth := 4
  let num_high_worth := 1
  num_low_worth * sale_price15 + num_high_worth * sale_price20 = 55 :=
by
  sorry

end jimmy_earnings_l168_168303


namespace find_f_4_l168_168943

noncomputable def f (x : ℕ) (a b c : ℕ) : ℕ := 2 * a * x + b * x + c

theorem find_f_4
  (a b c : ℕ)
  (f1 : f 1 a b c = 10)
  (f2 : f 2 a b c = 20) :
  f 4 a b c = 40 :=
sorry

end find_f_4_l168_168943


namespace Madelyn_daily_pizza_expense_l168_168609

theorem Madelyn_daily_pizza_expense (total_expense : ℕ) (days_in_may : ℕ) 
  (h1 : total_expense = 465) (h2 : days_in_may = 31) : 
  total_expense / days_in_may = 15 := 
by
  sorry

end Madelyn_daily_pizza_expense_l168_168609


namespace fewest_tiles_needed_l168_168867

theorem fewest_tiles_needed 
  (tile_len : ℝ) (tile_wid : ℝ) (region_len : ℝ) (region_wid : ℝ)
  (h_tile_dims : tile_len = 2 ∧ tile_wid = 3)
  (h_region_dims : region_len = 48 ∧ region_wid = 72) :
  (region_len * region_wid) / (tile_len * tile_wid) = 576 :=
by {
  sorry
}

end fewest_tiles_needed_l168_168867


namespace al_sandwich_combinations_l168_168050

def types_of_bread : ℕ := 5
def types_of_meat : ℕ := 6
def types_of_cheese : ℕ := 5

def restricted_turkey_swiss_combinations : ℕ := 5
def restricted_white_chicken_combinations : ℕ := 5
def restricted_rye_turkey_combinations : ℕ := 5

def total_sandwich_combinations : ℕ := types_of_bread * types_of_meat * types_of_cheese

def valid_sandwich_combinations : ℕ :=
  total_sandwich_combinations - restricted_turkey_swiss_combinations
  - restricted_white_chicken_combinations - restricted_rye_turkey_combinations

theorem al_sandwich_combinations : valid_sandwich_combinations = 135 := 
  by
  sorry

end al_sandwich_combinations_l168_168050


namespace inverse_proportion_value_scientific_notation_l168_168825

-- Statement to prove for Question 1:
theorem inverse_proportion_value (m : ℤ) (x : ℝ) :
  (m - 2) * x ^ (m ^ 2 - 5) = 0 ↔ m = -2 := by
  sorry

-- Statement to prove for Question 2:
theorem scientific_notation : -0.00000032 = -3.2 * 10 ^ (-7) := by
  sorry

end inverse_proportion_value_scientific_notation_l168_168825


namespace total_feet_l168_168642

theorem total_feet (H C F : ℕ) (h1 : H + C = 48) (h2 : H = 28) :
  F = 2 * H + 4 * C → F = 136 :=
by
  -- substitute H = 28 and perform the calculations
  sorry

end total_feet_l168_168642


namespace grandfather_age_l168_168696

theorem grandfather_age :
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ 10 * a + b = a + b^2 ∧ 10 * a + b = 89 :=
by
  sorry

end grandfather_age_l168_168696


namespace baking_powder_now_l168_168389

def baking_powder_yesterday : ℝ := 0.4
def baking_powder_used : ℝ := 0.1

theorem baking_powder_now : 
  baking_powder_yesterday - baking_powder_used = 0.3 :=
by
  sorry

end baking_powder_now_l168_168389


namespace trig_identity_proofs_l168_168162

theorem trig_identity_proofs (α : ℝ) 
  (h : Real.sin α + Real.cos α = 1 / 5) :
  (Real.sin α - Real.cos α = 7 / 5 ∨ Real.sin α - Real.cos α = -7 / 5) ∧
  (Real.sin α ^ 3 + Real.cos α ^ 3 = 37 / 125) :=
by
  sorry

end trig_identity_proofs_l168_168162


namespace eq1_solution_eq2_no_solution_l168_168557

theorem eq1_solution (x : ℝ) (h : x ≠ 0 ∧ x ≠ 2) :
  (2/x + 1/(x*(x-2)) = 5/(2*x)) ↔ x = 4 :=
by sorry

theorem eq2_no_solution (x : ℝ) (h : x ≠ 2) :
  (5*x - 4)/ (x - 2) = (4*x + 10) / (3*x - 6) - 1 ↔ false :=
by sorry

end eq1_solution_eq2_no_solution_l168_168557


namespace range_of_x_l168_168014

def p (x : ℝ) : Prop := x^2 - 5 * x + 6 ≥ 0
def q (x : ℝ) : Prop := 0 < x ∧ x < 4

theorem range_of_x (x : ℝ) (hpq : p x ∨ q x) (hnq : ¬ q x) : x ≤ 0 ∨ x ≥ 4 :=
by sorry

end range_of_x_l168_168014


namespace find_printer_price_l168_168491

variable (C P M : ℝ)

theorem find_printer_price
  (h1 : C + P + M = 3000)
  (h2 : P = (1/4) * (C + P + M + 800)) :
  P = 950 :=
sorry

end find_printer_price_l168_168491


namespace correct_yeast_population_change_statement_l168_168371

def yeast_produces_CO2 (aerobic : Bool) : Bool := 
  True

def yeast_unicellular_fungus : Bool := 
  True

def boiling_glucose_solution_purpose : Bool := 
  True

def yeast_facultative_anaerobe : Bool := 
  True

theorem correct_yeast_population_change_statement : 
  (∀ (aerobic : Bool), yeast_produces_CO2 aerobic) →
  yeast_unicellular_fungus →
  boiling_glucose_solution_purpose →
  yeast_facultative_anaerobe →
  "D is correct" = "D is correct" :=
by
  intros
  exact rfl

end correct_yeast_population_change_statement_l168_168371


namespace hyperbola_property_l168_168340

def hyperbola := {x : ℝ // ∃ y : ℝ, x^2 - y^2 / 8 = 1}

def is_on_left_branch (M : hyperbola) : Prop :=
  M.1 < 0

def focus1 : ℝ := -3
def focus2 : ℝ := 3

def distance (a b : ℝ) : ℝ := abs (a - b)

theorem hyperbola_property (M : hyperbola) (hM : is_on_left_branch M) :
  distance M.1 focus1 + distance focus1 focus2 - distance M.1 focus2 = 4 :=
  sorry

end hyperbola_property_l168_168340


namespace solve_for_x_l168_168251

theorem solve_for_x : ∃ x : ℝ, 2 * ((x - 1) - (2 * x + 1)) = 6 ∧ x = -5 := by
  use -5
  sorry

end solve_for_x_l168_168251


namespace fewer_popsicle_sticks_l168_168492

theorem fewer_popsicle_sticks :
  let boys := 10
  let girls := 12
  let sticks_per_boy := 15
  let sticks_per_girl := 12
  let boys_total := boys * sticks_per_boy
  let girls_total := girls * sticks_per_girl
  boys_total - girls_total = 6 := 
by
  let boys := 10
  let girls := 12
  let sticks_per_boy := 15
  let sticks_per_girl := 12
  let boys_total := boys * sticks_per_boy
  let girls_total := girls * sticks_per_girl
  show boys_total - girls_total = 6
  sorry

end fewer_popsicle_sticks_l168_168492


namespace Emily_money_made_l168_168097

def price_per_bar : ℕ := 4
def total_bars : ℕ := 8
def bars_sold : ℕ := total_bars - 3
def money_made : ℕ := bars_sold * price_per_bar

theorem Emily_money_made : money_made = 20 :=
by
  sorry

end Emily_money_made_l168_168097


namespace karen_total_nuts_l168_168759

variable (x y : ℝ)
variable (hx : x = 0.25)
variable (hy : y = 0.25)

theorem karen_total_nuts : x + y = 0.50 := by
  rw [hx, hy]
  norm_num

end karen_total_nuts_l168_168759


namespace yura_picture_dimensions_l168_168226

theorem yura_picture_dimensions (a b : ℕ) (h : (a + 2) * (b + 2) - a * b = a * b) :
  (a = 3 ∧ b = 10) ∨ (a = 10 ∧ b = 3) ∨ (a = 4 ∧ b = 6) ∨ (a = 6 ∧ b = 4) :=
by
  -- Place your proof here
  sorry

end yura_picture_dimensions_l168_168226


namespace action_figure_collection_complete_l168_168717

theorem action_figure_collection_complete (act_figures : ℕ) (cost_per_fig : ℕ) (extra_money_needed : ℕ) (total_collection : ℕ) 
    (h1 : act_figures = 7) 
    (h2 : cost_per_fig = 8) 
    (h3 : extra_money_needed = 72) : 
    total_collection = 16 :=
by
  sorry

end action_figure_collection_complete_l168_168717


namespace bread_calories_l168_168453

theorem bread_calories (total_calories : Nat) (pb_calories : Nat) (pb_servings : Nat) (bread_pieces : Nat) (bread_calories : Nat)
  (h1 : total_calories = 500)
  (h2 : pb_calories = 200)
  (h3 : pb_servings = 2)
  (h4 : bread_pieces = 1)
  (h5 : total_calories = pb_servings * pb_calories + bread_pieces * bread_calories) : 
  bread_calories = 100 :=
by
  sorry

end bread_calories_l168_168453


namespace teal_more_green_count_l168_168105

open Set

-- Define the survey data structure
def Survey : Type := {p : ℕ // p ≤ 150}

def people_surveyed : ℕ := 150
def more_blue (s : Survey) : Prop := sorry
def more_green (s : Survey) : Prop := sorry

-- Define the given conditions
def count_more_blue : ℕ := 90
def count_more_both : ℕ := 40
def count_neither : ℕ := 20

-- Define the proof statement
theorem teal_more_green_count :
  (count_more_both + (people_surveyed - (count_neither + (count_more_blue - count_more_both)))) = 80 :=
by {
  -- Sorry is used as a placeholder for the proof
  sorry
}

end teal_more_green_count_l168_168105


namespace inequality_lemma_l168_168575

theorem inequality_lemma (a b c d : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1) (hd : 0 < d ∧ d < 1) :
  1 + a * b + b * c + c * d + d * a + a * c + b * d > a + b + c + d :=
by 
  sorry

end inequality_lemma_l168_168575


namespace flag_blue_area_l168_168731

theorem flag_blue_area (A C₁ C₃ : ℝ) (h₀ : A = 1.0) (h₁ : C₁ + C₃ = 0.36 * A) :
  C₃ = 0.02 * A := by
  sorry

end flag_blue_area_l168_168731


namespace time_released_rope_first_time_l168_168131

theorem time_released_rope_first_time :
  ∀ (rate_ascent : ℕ) (rate_descent : ℕ) (time_first_ascent : ℕ) (time_second_ascent : ℕ) (highest_elevation : ℕ)
    (total_elevation_gained : ℕ) (elevation_difference : ℕ) (time_descent : ℕ),
  rate_ascent = 50 →
  rate_descent = 10 →
  time_first_ascent = 15 →
  time_second_ascent = 15 →
  highest_elevation = 1400 →
  total_elevation_gained = (rate_ascent * time_first_ascent) + (rate_ascent * time_second_ascent) →
  elevation_difference = total_elevation_gained - highest_elevation →
  time_descent = elevation_difference / rate_descent →
  time_descent = 10 :=
by
  intros rate_ascent rate_descent time_first_ascent time_second_ascent highest_elevation total_elevation_gained elevation_difference time_descent
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end time_released_rope_first_time_l168_168131


namespace september_first_2021_was_wednesday_l168_168711

-- Defining the main theorem based on the conditions and the question
theorem september_first_2021_was_wednesday
  (doubledCapitalOnWeekdays : ∀ day : Nat, day = 0 % 7 → True)
  (sevenFiftyPercOnWeekends : ∀ day : Nat, day = 5 % 7 → True)
  (millionaireOnLastDayOfYear: ∀ day : Nat, day = 364 % 7 → True)
  : 1 % 7 = 3 % 7 := 
sorry

end september_first_2021_was_wednesday_l168_168711


namespace reservoir_full_percentage_after_storm_l168_168405

theorem reservoir_full_percentage_after_storm 
  (original_contents water_added : ℤ) 
  (percentage_full_before_storm: ℚ) 
  (total_capacity new_contents : ℚ) 
  (H1 : original_contents = 220 * 10^9) 
  (H2 : water_added = 110 * 10^9) 
  (H3 : percentage_full_before_storm = 0.40)
  (H4 : total_capacity = original_contents / percentage_full_before_storm)
  (H5 : new_contents = original_contents + water_added) :
  (new_contents / total_capacity) = 0.60 := 
by 
  sorry

end reservoir_full_percentage_after_storm_l168_168405


namespace emily_curtains_purchase_l168_168617

theorem emily_curtains_purchase 
    (c : ℕ) 
    (curtain_cost : ℕ := 30)
    (print_count : ℕ := 9)
    (print_cost_per_unit : ℕ := 15)
    (installation_cost : ℕ := 50)
    (total_cost : ℕ := 245) :
    (curtain_cost * c + print_count * print_cost_per_unit + installation_cost = total_cost) → c = 2 :=
by
  sorry

end emily_curtains_purchase_l168_168617


namespace min_value_a_3b_9c_l168_168333

theorem min_value_a_3b_9c (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 27) : 
  a + 3 * b + 9 * c ≥ 27 := 
sorry

end min_value_a_3b_9c_l168_168333


namespace paper_cut_count_incorrect_l168_168483

theorem paper_cut_count_incorrect (n : ℕ) (h : n = 1961) : 
  ∀ i, (∃ k, i = 7 ∨ i = 7 + 6 * k) → i % 6 = 1 → n ≠ i :=
by
  sorry

end paper_cut_count_incorrect_l168_168483


namespace sum_abc_is_eight_l168_168849

theorem sum_abc_is_eight (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : (a + b + c)^3 - a^3 - b^3 - c^3 = 294) : a + b + c = 8 :=
by
  sorry

end sum_abc_is_eight_l168_168849


namespace right_triangle_condition_l168_168956

def fib (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 4
  | 2 => 4
  | n + 3 => fib (n + 2) + fib (n + 1)

theorem right_triangle_condition (n : ℕ) : 
  ∃ a b c, a = fib n * fib (n + 4) ∧ 
           b = fib (n + 1) * fib (n + 3) ∧ 
           c = 2 * fib (n + 2) ∧
           a * a + b * b = c * c :=
by sorry

end right_triangle_condition_l168_168956


namespace cards_per_deck_l168_168448

theorem cards_per_deck (decks : ℕ) (cards_per_layer : ℕ) (layers : ℕ) 
  (h_decks : decks = 16) 
  (h_cards_per_layer : cards_per_layer = 26) 
  (h_layers : layers = 32) 
  (total_cards_used : ℕ := cards_per_layer * layers) 
  (number_of_cards_per_deck : ℕ := total_cards_used / decks) :
  number_of_cards_per_deck = 52 :=
by 
  sorry

end cards_per_deck_l168_168448


namespace largest_num_of_hcf_and_lcm_factors_l168_168477

theorem largest_num_of_hcf_and_lcm_factors (hcf : ℕ) (f1 f2 : ℕ) (hcf_eq : hcf = 23) (f1_eq : f1 = 13) (f2_eq : f2 = 14) : 
    hcf * max f1 f2 = 322 :=
by
  -- use the conditions to find the largest number
  rw [hcf_eq, f1_eq, f2_eq]
  sorry

end largest_num_of_hcf_and_lcm_factors_l168_168477


namespace quadratic_ratio_l168_168352

theorem quadratic_ratio (b c : ℤ) (h : ∀ x : ℤ, x^2 + 1400 * x + 1400 = (x + b) ^ 2 + c) : c / b = -698 :=
sorry

end quadratic_ratio_l168_168352


namespace largest_determinable_1986_l168_168878

-- Define main problem with conditions
def largest_determinable_cards (total : ℕ) (select : ℕ) : ℕ :=
  total - 27

-- Statement we need to prove
theorem largest_determinable_1986 :
  largest_determinable_cards 2013 10 = 1986 :=
by
  sorry

end largest_determinable_1986_l168_168878


namespace find_replacement_percentage_l168_168110

noncomputable def final_percentage_replacement_alcohol_solution (a₁ p₁ p₂ x : ℝ) : Prop :=
  let d := 0.4 -- gallons
  let final_solution := 1 -- gallon
  let initial_pure_alcohol := a₁ * p₁ / 100
  let remaining_pure_alcohol := initial_pure_alcohol - (d * p₁ / 100)
  let added_pure_alcohol := d * x / 100
  remaining_pure_alcohol + added_pure_alcohol = final_solution * p₂ / 100

theorem find_replacement_percentage :
  final_percentage_replacement_alcohol_solution 1 75 65 50 :=
by
  sorry

end find_replacement_percentage_l168_168110


namespace chinese_money_plant_sales_l168_168948

/-- 
Consider a scenario where a plant supplier sells 20 pieces of orchids for $50 each 
and some pieces of potted Chinese money plant for $25 each. He paid his two workers $40 each 
and bought new pots worth $150. The plant supplier had $1145 left from his earnings. 
Prove that the number of pieces of potted Chinese money plants sold by the supplier is 15.
-/
theorem chinese_money_plant_sales (earnings_orchids earnings_per_orchid: ℤ)
  (num_orchids: ℤ)
  (earnings_plants earnings_per_plant: ℤ)
  (worker_wage num_workers: ℤ)
  (new_pots_cost remaining_money: ℤ)
  (earnings: ℤ)
  (P : earnings_orchids = num_orchids * earnings_per_orchid)
  (Q : earnings = earnings_orchids + earnings_plants)
  (R : earnings - (worker_wage * num_workers + new_pots_cost) = remaining_money)
  (conditions: earnings_per_orchid = 50 ∧ num_orchids = 20 ∧ earnings_per_plant = 25 ∧ worker_wage = 40 ∧ num_workers = 2 ∧ new_pots_cost = 150 ∧ remaining_money = 1145):
  earnings_plants / earnings_per_plant = 15 := 
by
  sorry

end chinese_money_plant_sales_l168_168948


namespace point_of_tangency_l168_168573

noncomputable def parabola1 (x : ℝ) : ℝ := 2 * x^2 + 10 * x + 14
noncomputable def parabola2 (y : ℝ) : ℝ := 4 * y^2 + 16 * y + 68

theorem point_of_tangency : 
  ∃ (x y : ℝ), parabola1 x = y ∧ parabola2 y = x ∧ x = -9/4 ∧ y = -15/8 :=
by
  -- The proof will show that the point of tangency is (-9/4, -15/8)
  sorry

end point_of_tangency_l168_168573


namespace exists_two_integers_with_difference_divisible_by_2022_l168_168498

theorem exists_two_integers_with_difference_divisible_by_2022 (a : Fin 2023 → ℤ) : 
  ∃ i j : Fin 2023, i ≠ j ∧ (a i - a j) % 2022 = 0 := by
  sorry

end exists_two_integers_with_difference_divisible_by_2022_l168_168498


namespace arithmetic_geometric_sequence_l168_168704

-- Define the arithmetic sequence
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + n * d

-- Define the first term, common difference and positions of terms in geometric sequence
def a1 : ℤ := -8
def d : ℤ := 2
def a3 := arithmetic_sequence a1 d 2
def a4 := arithmetic_sequence a1 d 3

-- Conditions for the terms forming a geometric sequence
def geometric_condition (a b c : ℤ) : Prop :=
  b^2 = a * c

-- Statement to prove
theorem arithmetic_geometric_sequence :
  geometric_condition a1 a3 a4 → a1 = -8 :=
by
  intro h
  -- Proof can be filled in here
  sorry

end arithmetic_geometric_sequence_l168_168704


namespace min_value_expression_l168_168018

theorem min_value_expression (y : ℝ) (hy : y > 0) : 9 * y + 1 / y^6 ≥ 10 :=
by
  sorry

end min_value_expression_l168_168018


namespace female_managers_count_l168_168484

-- Definitions for the problem statement

def total_female_employees : ℕ := 500
def fraction_of_managers : ℚ := 2 / 5
def fraction_of_male_managers : ℚ := 2 / 5

-- Problem parameters
variable (E M FM : ℕ) -- E: total employees, M: male employees, FM: female managers

-- Conditions
def total_employees_eq : Prop := E = M + total_female_employees
def total_managers_eq : Prop := fraction_of_managers * E = fraction_of_male_managers * M + FM

-- The statement we want to prove
theorem female_managers_count (h1 : total_employees_eq E M) (h2 : total_managers_eq E M FM) : FM = 200 :=
by
  -- to be proven
  sorry

end female_managers_count_l168_168484


namespace find_sum_of_a_and_b_l168_168829

theorem find_sum_of_a_and_b (a b : ℝ) 
  (h1 : ∀ x : ℝ, (abs (x^2 - 2 * a * x + b) = 8) → (x = a ∨ x = a + 4 ∨ x = a - 4))
  (h2 : a^2 + (a - 4)^2 = (a + 4)^2) :
  a + b = 264 :=
by
  sorry

end find_sum_of_a_and_b_l168_168829


namespace growth_comparison_l168_168622

theorem growth_comparison (x : ℝ) (h : ℝ) (hx : x > 0) : 
  (0 < x ∧ x < 1 / 2 → (x + h) - x > ((x + h)^2 - x^2)) ∧
  (x > 1 / 2 → ((x + h)^2 - x^2) > (x + h) - x) :=
by
  sorry

end growth_comparison_l168_168622


namespace problem1_problem2_problem3_l168_168627

section problem

variable (m : ℝ)

-- Proposition p: The equation x^2 - 4mx + 1 = 0 has real solutions
def p : Prop := (16 * m^2 - 4) ≥ 0

-- Proposition q: There exists some x₀ ∈ ℝ such that mx₀^2 - 2x₀ - 1 > 0
def q : Prop := ∃ (x₀ : ℝ), (m * x₀^2 - 2 * x₀ - 1) > 0

-- Solution to (1): If p is true, the range of values for m
theorem problem1 (hp : p m) : m ≥ 1/2 ∨ m ≤ -1/2 := sorry

-- Solution to (2): If q is true, the range of values for m
theorem problem2 (hq : q m) : m > -1 := sorry

-- Solution to (3): If both p and q are false but either p or q is true,
-- find the range of values for m
theorem problem3 (hnp : ¬p m) (hnq : ¬q m) (hpq : p m ∨ q m) : -1 < m ∧ m < 1/2 := sorry

end problem

end problem1_problem2_problem3_l168_168627


namespace hypotenuse_length_l168_168442

theorem hypotenuse_length (a b c : ℕ) (h₀ : a = 15) (h₁ : b = 36) (h₂ : a^2 + b^2 = c^2) : c = 39 :=
by
  -- Proof is omitted
  sorry

end hypotenuse_length_l168_168442


namespace line_through_vertex_has_two_a_values_l168_168616

-- Definitions for the line and parabola as conditions
def line_eq (a x : ℝ) : ℝ := 2 * x + a
def parabola_eq (a x : ℝ) : ℝ := x^2 + 2 * a^2

-- The proof problem
theorem line_through_vertex_has_two_a_values :
  (∃ a1 a2 : ℝ, (a1 ≠ a2) ∧ (line_eq a1 0 = parabola_eq a1 0) ∧ (line_eq a2 0 = parabola_eq a2 0)) ∧
  (∀ a : ℝ, line_eq a 0 = parabola_eq a 0 → (a = 0 ∨ a = 1/2)) :=
sorry

end line_through_vertex_has_two_a_values_l168_168616


namespace fixed_point_of_function_l168_168493

theorem fixed_point_of_function (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  ∃ x y : ℝ, y = a^(x-1) + 1 ∧ (x, y) = (1, 2) :=
by 
  sorry

end fixed_point_of_function_l168_168493


namespace find_coordinates_of_M_l168_168358

def point_in_second_quadrant (P : ℝ × ℝ) : Prop :=
  P.1 < 0 ∧ P.2 > 0

def distance_to_x_axis (P : ℝ × ℝ) (d : ℝ) : Prop :=
  abs (P.2) = d

def distance_to_y_axis (P : ℝ × ℝ) (d : ℝ) : Prop :=
  abs (P.1) = d

theorem find_coordinates_of_M :
  ∃ M : ℝ × ℝ, point_in_second_quadrant M ∧ distance_to_x_axis M 5 ∧ distance_to_y_axis M 3 ∧ M = (-3, 5) :=
by
  sorry

end find_coordinates_of_M_l168_168358


namespace exists_two_digit_number_N_l168_168715

-- Statement of the problem
theorem exists_two_digit_number_N : 
  ∃ (N : ℕ), (∃ (a b : ℕ), N = 10 * a + b ∧ N = a * b + 2 * (a + b) ∧ 10 ≤ N ∧ N < 100) :=
by
  sorry

end exists_two_digit_number_N_l168_168715


namespace shirley_ends_with_106_l168_168169

-- Define the initial number of eggs and the number bought
def initialEggs : Nat := 98
def additionalEggs : Nat := 8

-- Define the final count as the sum of initial eggs and additional eggs
def finalEggCount : Nat := initialEggs + additionalEggs

-- State the theorem with the correct answer
theorem shirley_ends_with_106 :
  finalEggCount = 106 :=
by
  sorry

end shirley_ends_with_106_l168_168169


namespace tan_ratio_given_sin_equation_l168_168760

theorem tan_ratio_given_sin_equation (α β : ℝ) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.sin (2*α + β) = (3/2) * Real.sin β) : 
  Real.tan (α + β) / Real.tan α = 5 :=
by
  -- Proof goes here
  sorry

end tan_ratio_given_sin_equation_l168_168760


namespace minimize_expression_l168_168528

theorem minimize_expression (a : ℝ) : ∃ c : ℝ, 0 ≤ c ∧ c ≤ a ∧ (∀ x : ℝ, 0 ≤ x ∧ x ≤ a → (x^2 + 3 * (a-x)^2) ≥ ((3*a/4)^2 + 3 * (a-3*a/4)^2)) :=
by
  sorry

end minimize_expression_l168_168528


namespace price_decrease_percentage_l168_168757

variables (P Q : ℝ)
variables (Q' R R' : ℝ)

-- Condition: the number sold increased by 60%
def quantity_increase_condition : Prop :=
  Q' = Q * (1 + 0.60)

-- Condition: the total revenue increased by 28.000000000000025%
def revenue_increase_condition : Prop :=
  R' = R * (1 + 0.28000000000000025)

-- Definition: the original revenue R
def original_revenue : Prop :=
  R = P * Q

-- The new price P' after decreasing by x%
variables (P' : ℝ) (x : ℝ)
def new_price_condition : Prop :=
  P' = P * (1 - x / 100)

-- The new revenue R'
def new_revenue : Prop :=
  R' = P' * Q'

-- The proof problem
theorem price_decrease_percentage (P Q Q' R R' P' x : ℝ)
  (h1 : quantity_increase_condition Q Q')
  (h2 : revenue_increase_condition R R')
  (h3 : original_revenue P Q R)
  (h4 : new_price_condition P P' x)
  (h5 : new_revenue P' Q' R') :
  x = 20 :=
sorry

end price_decrease_percentage_l168_168757


namespace negation_exists_l168_168306

theorem negation_exists :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ x) ↔ ∃ x : ℝ, x^2 + 1 < x :=
sorry

end negation_exists_l168_168306


namespace determine_remaining_sides_l168_168489

variables (A B C D E : Type)

def cyclic_quadrilateral (A B C D : Type) : Prop := sorry

def known_sides (AB CD : ℝ) : Prop := AB > 0 ∧ CD > 0

def known_ratio (m n : ℝ) : Prop := m > 0 ∧ n > 0

theorem determine_remaining_sides
  {A B C D : Type}
  (h_cyclic : cyclic_quadrilateral A B C D)
  (AB CD : ℝ) (h_sides : known_sides AB CD)
  (m n : ℝ) (h_ratio : known_ratio m n) :
  ∃ (BC AD : ℝ), BC / AD = m / n ∧ BC > 0 ∧ AD > 0 :=
sorry

end determine_remaining_sides_l168_168489


namespace parabola_directrix_l168_168777

theorem parabola_directrix (vertex_origin : ∀ (x y : ℝ), x = 0 ∧ y = 0)
    (directrix : ∀ (y : ℝ), y = 4) : ∃ p, x^2 = -2 * p * y ∧ p = 8 ∧ x^2 = -16 * y := 
sorry

end parabola_directrix_l168_168777


namespace convert_to_polar_coordinates_l168_168504

open Real

noncomputable def polar_coordinates (x y : ℝ) : ℝ × ℝ :=
  let r := sqrt (x^2 + y^2)
  let θ := if y < 0 then 2 * π - arctan (abs y / abs x) else arctan (abs y / abs x)
  (r, θ)

theorem convert_to_polar_coordinates : 
  polar_coordinates 3 (-3) = (3 * sqrt 2, 7 * π / 4) :=
by
  sorry

end convert_to_polar_coordinates_l168_168504


namespace remainder_when_divided_by_63_l168_168705

theorem remainder_when_divided_by_63 (x : ℤ) (h1 : ∃ q : ℤ, x = 63 * q + r ∧ 0 ≤ r ∧ r < 63) (h2 : ∃ k : ℤ, x = 9 * k + 2) :
  ∃ r : ℤ, 0 ≤ r ∧ r < 63 ∧ r = 7 :=
by
  sorry

end remainder_when_divided_by_63_l168_168705


namespace additional_tickets_won_l168_168895

-- Definitions from the problem
def initial_tickets : ℕ := 49
def spent_tickets : ℕ := 25
def final_tickets : ℕ := 30

-- The main statement we need to prove
theorem additional_tickets_won (initial_tickets : ℕ) (spent_tickets : ℕ) (final_tickets : ℕ) : 
  final_tickets - (initial_tickets - spent_tickets) = 6 :=
by
  sorry

end additional_tickets_won_l168_168895


namespace smallest_perimeter_of_scalene_triangle_with_conditions_l168_168845

def is_odd_prime (n : ℕ) : Prop :=
  Nat.Prime n ∧ n % 2 = 1

-- Define a scalene triangle
structure ScaleneTriangle :=
  (a b c : ℕ)
  (a_ne_b : a ≠ b)
  (a_ne_c : a ≠ c)
  (b_ne_c : b ≠ c)
  (triangle_inequality1 : a + b > c)
  (triangle_inequality2 : a + c > b)
  (triangle_inequality3 : b + c > a)

-- Define the problem conditions
def problem_conditions (a b c : ℕ) : Prop :=
  is_odd_prime a ∧ is_odd_prime b ∧ is_odd_prime c ∧
  a < b ∧ b < c ∧
  Nat.Prime (a + b + c) ∧
  (∃ (t : ScaleneTriangle), t.a = a ∧ t.b = b ∧ t.c = c)

-- Define the proposition
theorem smallest_perimeter_of_scalene_triangle_with_conditions :
  ∃ (a b c : ℕ), problem_conditions a b c ∧ a + b + c = 23 :=
sorry

end smallest_perimeter_of_scalene_triangle_with_conditions_l168_168845


namespace min_value_of_sum_l168_168574

noncomputable def min_value_x_3y (x y : ℝ) : ℝ :=
  x + 3 * y

theorem min_value_of_sum (x y : ℝ) 
  (hx_pos : x > 0) (hy_pos : y > 0) 
  (cond : 1 / (x + 1) + 1 / (y + 1) = 1 / 2) :
  x + 3 * y ≥ 4 + 4 * Real.sqrt 3 :=
  sorry

end min_value_of_sum_l168_168574


namespace shaded_area_square_semicircles_l168_168983

theorem shaded_area_square_semicircles :
  let side_length := 2
  let radius_circle := side_length * Real.sqrt 2 / 2
  let area_circle := Real.pi * radius_circle^2
  let area_square := side_length^2
  let area_semicircle := Real.pi * (side_length / 2)^2 / 2
  let total_area_semicircles := 4 * area_semicircle
  let shaded_area := total_area_semicircles - area_circle
  shaded_area = 4 :=
by
  sorry

end shaded_area_square_semicircles_l168_168983


namespace molecular_weight_compound_l168_168897

def atomic_weight_H : ℝ := 1.008
def atomic_weight_C : ℝ := 12.011
def atomic_weight_O : ℝ := 15.999
def atomic_weight_N : ℝ := 14.007
def atomic_weight_Cl : ℝ := 35.453

def molecular_weight (nH nC nO nN nCl : ℕ) : ℝ :=
  nH * atomic_weight_H + nC * atomic_weight_C + nO * atomic_weight_O + nN * atomic_weight_N + nCl * atomic_weight_Cl

theorem molecular_weight_compound :
  molecular_weight 4 2 3 1 2 = 160.964 := by
  sorry

end molecular_weight_compound_l168_168897


namespace intersection_points_of_graph_and_line_l168_168023

theorem intersection_points_of_graph_and_line (f : ℝ → ℝ) :
  (∀ x : ℝ, f x ≠ my_special_value) → (∀ x₁ x₂ : ℝ, f x₁ = f x₂ → x₁ = x₂) →
  ∃! x : ℝ, x = 1 ∧ ∃ y : ℝ, y = f x :=
by
  sorry

end intersection_points_of_graph_and_line_l168_168023


namespace circle_tangent_locus_l168_168726

theorem circle_tangent_locus (a b : ℝ) :
  (∃ r : ℝ, (a ^ 2 + b ^ 2 = (r + 1) ^ 2) ∧ ((a - 3) ^ 2 + b ^ 2 = (5 - r) ^ 2)) →
  3 * a ^ 2 + 4 * b ^ 2 - 14 * a - 49 = 0 := by
  sorry

end circle_tangent_locus_l168_168726


namespace estimate_larger_than_difference_l168_168026

theorem estimate_larger_than_difference
  (u v δ γ : ℝ)
  (huv : u > v)
  (hδ : δ > 0)
  (hγ : γ > 0)
  (hδγ : δ > γ) : (u + δ) - (v - γ) > u - v := by
  sorry

end estimate_larger_than_difference_l168_168026


namespace decimal_subtraction_l168_168122

theorem decimal_subtraction (a b : ℝ) (h1 : a = 3.79) (h2 : b = 2.15) : a - b = 1.64 := by
  rw [h1, h2]
  -- This follows from the correct calculation rule
  sorry

end decimal_subtraction_l168_168122


namespace area_of_EFGH_l168_168444

def shorter_side := 6
def ratio := 2
def longer_side := shorter_side * ratio
def width := 2 * longer_side
def length := shorter_side

theorem area_of_EFGH : length * width = 144 := by
  sorry

end area_of_EFGH_l168_168444


namespace solve_for_x_l168_168673

def equation (x : ℝ) (y : ℝ) : Prop := 5 * y^2 + y + 10 = 2 * (9 * x^2 + y + 6)

def y_condition (x : ℝ) : ℝ := 3 * x

theorem solve_for_x (x : ℝ) :
  equation x (y_condition x) ↔ (x = 1/3 ∨ x = -2/9) := by
  sorry

end solve_for_x_l168_168673


namespace original_number_without_10s_digit_l168_168182

theorem original_number_without_10s_digit (h : ℕ) (n : ℕ) 
  (h_eq_1 : h = 1) 
  (n_eq : n = 2 * 1000 + h * 100 + 84) 
  (div_by_6: n % 6 = 0) : n = 2184 → 284 = 284 :=
by
  sorry

end original_number_without_10s_digit_l168_168182


namespace car_distances_equal_600_l168_168668

-- Define the variables
def time_R (t : ℝ) := t
def speed_R := 50
def time_P (t : ℝ) := t - 2
def speed_P := speed_R + 10
def distance (t : ℝ) := speed_R * time_R t

-- The Lean theorem statement
theorem car_distances_equal_600 (t : ℝ) (h : time_R t = t) (h1 : speed_R = 50) 
  (h2 : time_P t = t - 2) (h3 : speed_P = speed_R + 10) :
  distance t = 600 :=
by
  -- We would provide the proof here, but for now we use sorry to indicate the proof is omitted.
  sorry

end car_distances_equal_600_l168_168668


namespace required_cement_l168_168368

def total_material : ℝ := 0.67
def sand : ℝ := 0.17
def dirt : ℝ := 0.33
def cement : ℝ := 0.17

theorem required_cement : cement = total_material - (sand + dirt) := 
by
  sorry

end required_cement_l168_168368


namespace sum_is_composite_l168_168224

theorem sum_is_composite (a b c d : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h : a * b = c * d) : 
  ∃ x y : ℕ, 1 < x ∧ 1 < y ∧ x * y = a + b + c + d :=
sorry

end sum_is_composite_l168_168224


namespace product_173_240_l168_168241

theorem product_173_240 :
  ∃ n : ℕ, n = 3460 ∧ n * 12 = 173 * 240 ∧ 173 * 240 = 41520 :=
by
  sorry

end product_173_240_l168_168241


namespace exists_positive_integer_special_N_l168_168412

theorem exists_positive_integer_special_N : 
  ∃ (N : ℕ), 
    (∃ (m : ℕ), N = 1990 * (m + 995)) ∧ 
    (∀ (n : ℕ), (∃ (m : ℕ), 2 * N = (n + 1) * (2 * m + n)) ↔ (3980 = 2 * 1990)) := by
  sorry

end exists_positive_integer_special_N_l168_168412


namespace magnitude_of_z_l168_168834

open Complex

theorem magnitude_of_z {z : ℂ} (h : z * (1 + I) = 1 - I) : abs z = 1 :=
sorry

end magnitude_of_z_l168_168834


namespace find_expression_value_l168_168662

theorem find_expression_value (x y : ℚ) (h₁ : 3 * x + y = 6) (h₂ : x + 3 * y = 8) :
  9 * x ^ 2 + 15 * x * y + 9 * y ^ 2 = 1629 / 16 := 
sorry

end find_expression_value_l168_168662


namespace Rose_has_20_crystal_beads_l168_168398

noncomputable def num_crystal_beads (metal_beads_Nancy : ℕ) (pearl_beads_more_than_metal : ℕ) (beads_per_bracelet : ℕ)
    (total_bracelets : ℕ) (stone_to_crystal_ratio : ℕ) : ℕ :=
  let pearl_beads_Nancy := metal_beads_Nancy + pearl_beads_more_than_metal
  let total_beads_Nancy := metal_beads_Nancy + pearl_beads_Nancy
  let beads_needed := beads_per_bracelet * total_bracelets
  let beads_Rose := beads_needed - total_beads_Nancy
  beads_Rose / stone_to_crystal_ratio.succ

theorem Rose_has_20_crystal_beads :
  num_crystal_beads 40 20 8 20 2 = 20 :=
by
  sorry

end Rose_has_20_crystal_beads_l168_168398


namespace john_total_distance_l168_168230

theorem john_total_distance : 
  let daily_distance := 1700
  let days_run := 6
  daily_distance * days_run = 10200 :=
by
  sorry

end john_total_distance_l168_168230


namespace roots_equal_of_quadratic_eq_zero_l168_168373

theorem roots_equal_of_quadratic_eq_zero (a : ℝ) :
  (∃ x : ℝ, (x^2 - a*x + 1) = 0 ∧ (∀ y : ℝ, (y^2 - a*y + 1) = 0 → y = x)) → (a = 2 ∨ a = -2) :=
by
  sorry

end roots_equal_of_quadratic_eq_zero_l168_168373


namespace scientific_notation_86560_l168_168386

theorem scientific_notation_86560 : ∃ a n, (86560 : ℝ) = a * 10 ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 8.656 ∧ n = 4 :=
by {
  sorry
}

end scientific_notation_86560_l168_168386


namespace yellow_faces_of_cube_l168_168746

theorem yellow_faces_of_cube (n : ℕ) (h : 6 * n^2 = (1 / 3) * (6 * n^3)) : n = 3 :=
by {
  sorry
}

end yellow_faces_of_cube_l168_168746


namespace equation_one_solution_equation_two_no_solution_l168_168656

theorem equation_one_solution (x : ℝ) (hx1 : x ≠ 3) : (2 * x + 9) / (3 - x) = (4 * x - 7) / (x - 3) ↔ x = -1 / 3 := 
by 
    sorry

theorem equation_two_no_solution (x : ℝ) (hx2 : x ≠ 1) (hx3 : x ≠ -1) : 
    (x + 1) / (x - 1) - 4 / (x ^ 2 - 1) = 1 → False := 
by 
    sorry

end equation_one_solution_equation_two_no_solution_l168_168656


namespace solve_quadratic_l168_168599

theorem solve_quadratic : ∀ x : ℝ, x ^ 2 - 6 * x + 8 = 0 ↔ x = 2 ∨ x = 4 := by
  sorry

end solve_quadratic_l168_168599


namespace sum_distinct_x2_y2_z2_l168_168530

/-
Given positive integers x, y, and z such that
x + y + z = 30 and gcd(x, y) + gcd(y, z) + gcd(z, x) = 10,
prove that the sum of all possible distinct values of x^2 + y^2 + z^2 is 404.
-/
theorem sum_distinct_x2_y2_z2 (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 30) 
  (h_gcd : Nat.gcd x y + Nat.gcd y z + Nat.gcd z x = 10) : 
  x^2 + y^2 + z^2 = 404 :=
sorry

end sum_distinct_x2_y2_z2_l168_168530


namespace circle_radius_five_l168_168713

theorem circle_radius_five (c : ℝ) : (∃ x y : ℝ, x^2 + 10 * x + y^2 + 8 * y + c = 0) ∧ 
                                     ((x + 5)^2 + (y + 4)^2 = 25) → c = 16 :=
by
  sorry

end circle_radius_five_l168_168713


namespace percentage_seeds_germinated_l168_168815

/-- There were 300 seeds planted in the first plot and 200 seeds planted in the second plot. 
    30% of the seeds in the first plot germinated and 32% of the total seeds germinated.
    Prove that 35% of the seeds in the second plot germinated. -/
theorem percentage_seeds_germinated 
  (s1 s2 : ℕ) (p1 p2 t : ℚ)
  (h1 : s1 = 300) 
  (h2 : s2 = 200) 
  (h3 : p1 = 30) 
  (h4 : t = 32) 
  (h5 : 0.30 * s1 + p2 * s2 = 0.32 * (s1 + s2)) :
  p2 = 35 :=
by 
  -- Proof goes here
  sorry

end percentage_seeds_germinated_l168_168815


namespace inequality_a4_b4_c4_geq_l168_168977

theorem inequality_a4_b4_c4_geq (a b c : ℝ) : 
  a^4 + b^4 + c^4 ≥ a^2 * b^2 + b^2 * c^2 + c^2 * a^2 := 
by
  sorry

end inequality_a4_b4_c4_geq_l168_168977


namespace probability_diff_topics_l168_168811

theorem probability_diff_topics
  (num_topics : ℕ)
  (num_combinations : ℕ)
  (num_different_combinations : ℕ)
  (h1 : num_topics = 6)
  (h2 : num_combinations = num_topics * num_topics)
  (h3 : num_combinations = 36)
  (h4 : num_different_combinations = num_topics * (num_topics - 1))
  (h5 : num_different_combinations = 30) :
  (num_different_combinations / num_combinations) = 5 / 6 := 
by 
  sorry

end probability_diff_topics_l168_168811


namespace find_ordered_pair_l168_168658

theorem find_ordered_pair:
  ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ 10 * m * n = 45 - 5 * m - 3 * n ∧ (m, n) = (1, 11) :=
by
  sorry

end find_ordered_pair_l168_168658


namespace find_p_of_abs_sum_roots_eq_five_l168_168720

theorem find_p_of_abs_sum_roots_eq_five (p : ℝ) : 
  (∃ x y : ℝ, x + y = -p ∧ x * y = -6 ∧ |x| + |y| = 5) → (p = 1 ∨ p = -1) := by
  sorry

end find_p_of_abs_sum_roots_eq_five_l168_168720


namespace max_principals_in_10_years_l168_168065

theorem max_principals_in_10_years (term_length : ℕ) (period_length : ℕ) (max_principals : ℕ)
  (term_length_eq : term_length = 4) (period_length_eq : period_length = 10) :
  max_principals = 4 :=
by
  sorry

end max_principals_in_10_years_l168_168065


namespace smallest_integer_solution_l168_168015

theorem smallest_integer_solution (x : ℤ) (h : 10 - 5 * x < -18) : x = 6 :=
sorry

end smallest_integer_solution_l168_168015


namespace range_of_values_l168_168633

theorem range_of_values (x : ℝ) : (x^2 - 5 * x + 6 < 0) ↔ (2 < x ∧ x < 3) :=
sorry

end range_of_values_l168_168633


namespace increasing_interval_of_f_maximum_value_of_f_l168_168989

open Real

def f (x : ℝ) : ℝ := x^2 - 2 * x

-- Consider x in the interval [-2, 4]
def domain_x (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 4

theorem increasing_interval_of_f :
  ∃a b : ℝ, (a, b) = (1, 4) ∧ ∀ x y : ℝ, domain_x x → domain_x y → a ≤ x → x < y → y ≤ b → f x < f y := sorry

theorem maximum_value_of_f :
  ∃ M : ℝ, M = 8 ∧ ∀ x : ℝ, domain_x x → f x ≤ M := sorry

end increasing_interval_of_f_maximum_value_of_f_l168_168989


namespace simplify_expression_1_simplify_expression_2_l168_168531

-- Problem 1
theorem simplify_expression_1 (a b : ℝ) : 4 * (a + b) + 2 * (a + b) - (a + b) = 5 * a + 5 * b :=
  sorry

-- Problem 2
theorem simplify_expression_2 (m : ℝ) : (3 * m / 2) - (5 * m / 2 - 1) + 3 * (4 - m) = -4 * m + 13 :=
  sorry

end simplify_expression_1_simplify_expression_2_l168_168531


namespace fish_lifespan_is_12_l168_168847

def hamster_lifespan : ℝ := 2.5
def dog_lifespan : ℝ := 4 * hamster_lifespan
def fish_lifespan : ℝ := dog_lifespan + 2

theorem fish_lifespan_is_12 : fish_lifespan = 12 := by
  sorry

end fish_lifespan_is_12_l168_168847


namespace reciprocal_sum_fractions_l168_168409

theorem reciprocal_sum_fractions:
  let a := (3: ℚ) / 4
  let b := (5: ℚ) / 6
  let c := (1: ℚ) / 2
  (a + b + c)⁻¹ = 12 / 25 :=
by
  sorry

end reciprocal_sum_fractions_l168_168409


namespace number_of_poly_lines_l168_168069

def nonSelfIntersectingPolyLines (n : ℕ) : ℕ :=
  if n = 2 then 1
  else if n ≥ 3 then n * 2^(n - 3)
  else 0

theorem number_of_poly_lines (n : ℕ) (h : n > 1) :
  nonSelfIntersectingPolyLines n =
  if n = 2 then 1 else n * 2^(n - 3) :=
by sorry

end number_of_poly_lines_l168_168069


namespace range_of_m_l168_168710

variable {x y m : ℝ}

theorem range_of_m (hx : 0 < x) (hy : 0 < y) (h_eq : 1/x + 4/y = 1) (h_ineq : ∃ x y, x + y/4 < m^2 - 3*m) : m < -1 ∨ m > 4 :=
sorry

end range_of_m_l168_168710


namespace shaded_area_is_correct_l168_168982

noncomputable def square_shaded_area (side : ℝ) (beta : ℝ) (cos_beta : ℝ) : ℝ :=
  if (0 < beta) ∧ (beta < 90) ∧ (cos_beta = 3 / 5) ∧ (side = 2) then 3 / 10 
  else 0

theorem shaded_area_is_correct :
  square_shaded_area 2 beta (3 / 5) = 3 / 10 :=
by
  sorry

end shaded_area_is_correct_l168_168982


namespace positive_integer_solutions_l168_168244

theorem positive_integer_solutions :
  ∀ (a b c : ℕ), (8 * a - 5 * b)^2 + (3 * b - 2 * c)^2 + (3 * c - 7 * a)^2 = 2 → 
  (a = 3 ∧ b = 5 ∧ c = 7) ∨ (a = 12 ∧ b = 19 ∧ c = 28) :=
by
  sorry

end positive_integer_solutions_l168_168244


namespace fred_games_last_year_l168_168708

def total_games : Nat := 47
def games_this_year : Nat := 36

def games_last_year (total games games this year : Nat) : Nat := total_games - games_this_year

theorem fred_games_last_year : games_last_year total_games games_this_year = 11 :=
by
  sorry

end fred_games_last_year_l168_168708


namespace polygon_sides_l168_168043

theorem polygon_sides (n : ℕ) (h : (n-3) * 180 < 2008 ∧ 2008 < (n-1) * 180) : 
  n = 14 :=
sorry

end polygon_sides_l168_168043


namespace sum_possible_values_l168_168372

theorem sum_possible_values (x y : ℝ) (h : x * y - x / y^3 - y / x^3 = 4) :
  (x - 2) * (y - 2) = 4 ∨ (x - 2) * (y - 2) = 0 → (4 + 0 = 4) :=
by
  sorry

end sum_possible_values_l168_168372


namespace person_a_catch_up_person_b_5_times_l168_168246

theorem person_a_catch_up_person_b_5_times :
  ∀ (num_flags laps_a laps_b : ℕ),
  num_flags = 2015 →
  laps_a = 23 →
  laps_b = 13 →
  (∃ t : ℕ, ∃ n : ℕ, 10 * t = num_flags * n ∧
             23 * t / 10 = k * num_flags ∧
             n % 2 = 0) →
  n = 10 →
  10 / (2 * 1) = 5 :=
by sorry

end person_a_catch_up_person_b_5_times_l168_168246


namespace if_a_eq_b_then_ac_eq_bc_l168_168694

theorem if_a_eq_b_then_ac_eq_bc (a b c : ℝ) : a = b → ac = bc :=
sorry

end if_a_eq_b_then_ac_eq_bc_l168_168694


namespace problem_statement_l168_168302

def atOp (a b : ℝ) := a * b ^ (1 / 2)

theorem problem_statement : atOp ((2 * 3) ^ 2) ((3 * 5) ^ 2 / 9) = 180 := by
  sorry

end problem_statement_l168_168302


namespace distance_inequality_l168_168901

theorem distance_inequality (a : ℝ) (h : |a - 1| < 3) : -2 < a ∧ a < 4 :=
sorry

end distance_inequality_l168_168901


namespace xyz_equivalence_l168_168020

theorem xyz_equivalence (x y z a b : ℝ) (h₁ : 4^x = a) (h₂: 2^y = b) (h₃ : 8^z = a * b) : 3 * z = 2 * x + y :=
by
  -- Here, we leave the proof as an exercise
  sorry

end xyz_equivalence_l168_168020


namespace gain_percent_calculation_l168_168725

variable (CP SP : ℝ)
variable (gain gain_percent : ℝ)

theorem gain_percent_calculation
  (h₁ : CP = 900) 
  (h₂ : SP = 1180)
  (h₃ : gain = SP - CP)
  (h₄ : gain_percent = (gain / CP) * 100) :
  gain_percent = 31.11 := by
sorry

end gain_percent_calculation_l168_168725


namespace farmer_plough_rate_l168_168307

-- Define the problem statement and the required proof 

theorem farmer_plough_rate :
  ∀ (x y : ℕ),
  90 * x = 3780 ∧ y * (x + 2) = 3740 → y = 85 :=
by
  sorry

end farmer_plough_rate_l168_168307


namespace avg_price_two_returned_theorem_l168_168086

-- Defining the initial conditions given in the problem
def avg_price_of_five (price: ℕ) (packets: ℕ) : Prop :=
  packets = 5 ∧ price = 20

def avg_price_of_three_remaining (price: ℕ) (packets: ℕ) : Prop :=
  packets = 3 ∧ price = 12
  
def cost_of_packets (price packets: ℕ) := price * packets

noncomputable def avg_price_two_returned (total_initial_cost total_remaining_cost: ℕ) :=
  (total_initial_cost - total_remaining_cost) / 2

-- The Lean 4 proof statement
theorem avg_price_two_returned_theorem (p1 p2 p3 p4: ℕ):
  avg_price_of_five p1 5 →
  avg_price_of_three_remaining p2 3 →
  cost_of_packets p1 5 = 100 →
  cost_of_packets p2 3 = 36 →
  avg_price_two_returned 100 36 = 32 :=
by
  intros h1 h2 h3 h4
  sorry

end avg_price_two_returned_theorem_l168_168086


namespace range_of_a_l168_168007

noncomputable def f (x a : ℝ) : ℝ := (Real.sin x)^2 + a * Real.cos x + a

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x a ≤ 1) → a ≤ 0 :=
by
  sorry

end range_of_a_l168_168007


namespace QR_value_l168_168234

-- Given conditions for the problem
def QP : ℝ := 15
def sinQ : ℝ := 0.4

-- Define QR based on the given conditions
noncomputable def QR : ℝ := QP / sinQ

-- The theorem to prove that QR = 37.5
theorem QR_value : QR = 37.5 := 
by
  unfold QR QP sinQ
  sorry

end QR_value_l168_168234


namespace abs_function_le_two_l168_168379

theorem abs_function_le_two {x : ℝ} (h : |x| ≤ 2) : |3 * x - x^3| ≤ 2 :=
sorry

end abs_function_le_two_l168_168379


namespace min_cookies_satisfy_conditions_l168_168316

theorem min_cookies_satisfy_conditions : ∃ (b : ℕ), b ≡ 5 [MOD 6] ∧ b ≡ 7 [MOD 8] ∧ b ≡ 8 [MOD 9] ∧ ∀ (b' : ℕ), (b' ≡ 5 [MOD 6] ∧ b' ≡ 7 [MOD 8] ∧ b' ≡ 8 [MOD 9]) → b ≤ b' := 
sorry

end min_cookies_satisfy_conditions_l168_168316


namespace balls_in_box_l168_168032

def num_blue : Nat := 6
def num_red : Nat := 4
def num_green : Nat := 3 * num_blue
def num_yellow : Nat := 2 * num_red
def num_total : Nat := num_blue + num_red + num_green + num_yellow

theorem balls_in_box : num_total = 36 := by
  sorry

end balls_in_box_l168_168032


namespace sum_abc_eq_ten_l168_168039

theorem sum_abc_eq_ten (a b c : ℝ) (h : (a - 5)^2 + (b - 3)^2 + (c - 2)^2 = 0) : a + b + c = 10 :=
by
  sorry

end sum_abc_eq_ten_l168_168039


namespace odd_powers_sum_divisible_by_p_l168_168534

theorem odd_powers_sum_divisible_by_p
  (p : ℕ)
  (hp_prime : Prime p)
  (hp_gt_3 : 3 < p)
  (a b c d : ℕ)
  (h_sum : (a + b + c + d) % p = 0)
  (h_cube_sum : (a^3 + b^3 + c^3 + d^3) % p = 0)
  (n : ℕ)
  (hn_odd : n % 2 = 1 ) :
  (a^n + b^n + c^n + d^n) % p = 0 :=
sorry

end odd_powers_sum_divisible_by_p_l168_168534


namespace choir_membership_l168_168058

theorem choir_membership (n : ℕ) (h1 : n % 7 = 4) (h2 : n % 8 = 3) (h3 : n ≥ 100) (h4 : n ≤ 200) :
  n = 123 ∨ n = 179 :=
by
  sorry

end choir_membership_l168_168058


namespace cylinder_surface_area_l168_168087

theorem cylinder_surface_area (h : ℝ) (r : ℝ) (h_eq : h = 12) (r_eq : r = 4) : 
  2 * π * r * (r + h) = 128 * π := 
by
  rw [h_eq, r_eq]
  sorry

end cylinder_surface_area_l168_168087


namespace perpendicular_lines_a_value_l168_168993

theorem perpendicular_lines_a_value (a : ℝ) :
  (a * (a + 2) = -1) → a = -1 :=
by
  intro h
  sorry

end perpendicular_lines_a_value_l168_168993


namespace discount_rate_on_pony_jeans_is_15_l168_168485

noncomputable def discountProblem : Prop :=
  ∃ (F P : ℝ),
    (15 * 3 * F / 100 + 18 * 2 * P / 100 = 8.55) ∧ 
    (F + P = 22) ∧ 
    (P = 15)

theorem discount_rate_on_pony_jeans_is_15 : discountProblem :=
sorry

end discount_rate_on_pony_jeans_is_15_l168_168485


namespace students_not_pass_l168_168703

theorem students_not_pass (total_students : ℕ) (percentage_passed : ℕ) (students_passed : ℕ) (students_not_passed : ℕ) :
  total_students = 804 →
  percentage_passed = 75 →
  students_passed = total_students * percentage_passed / 100 →
  students_not_passed = total_students - students_passed →
  students_not_passed = 201 :=
by
  intros h1 h2 h3 h4
  sorry

end students_not_pass_l168_168703


namespace sum_of_squares_neq_fourth_powers_l168_168021

theorem sum_of_squares_neq_fourth_powers (m n : ℕ) : 
  m^2 + (m + 1)^2 ≠ n^4 + (n + 1)^4 :=
by 
  sorry

end sum_of_squares_neq_fourth_powers_l168_168021


namespace systematic_sampling_first_group_draw_l168_168216

noncomputable def index_drawn_from_group (x n : ℕ) : ℕ := x + 8 * (n - 1)

theorem systematic_sampling_first_group_draw (k : ℕ) (fifteenth_group : index_drawn_from_group k 15 = 116) :
  index_drawn_from_group k 1 = 4 := 
sorry

end systematic_sampling_first_group_draw_l168_168216


namespace simplify_fraction_l168_168056

theorem simplify_fraction (x : ℝ) (h : x ≠ -1) : (x + 1) / (x^2 + 2 * x + 1) = 1 / (x + 1) :=
by
  sorry

end simplify_fraction_l168_168056


namespace c_left_before_completion_l168_168481

def a_one_day_work : ℚ := 1 / 24
def b_one_day_work : ℚ := 1 / 30
def c_one_day_work : ℚ := 1 / 40
def total_work_completed (days : ℚ) : Prop := days = 11

theorem c_left_before_completion (days_left : ℚ) (h : total_work_completed 11) :
  (11 - days_left) * (a_one_day_work + b_one_day_work + c_one_day_work) +
  (days_left * (a_one_day_work + b_one_day_work)) = 1 :=
sorry

end c_left_before_completion_l168_168481


namespace students_per_group_l168_168404

theorem students_per_group (total_students not_picked groups : ℕ) 
    (h1 : total_students = 64) 
    (h2 : not_picked = 36) 
    (h3 : groups = 4) : (total_students - not_picked) / groups = 7 :=
by
  sorry

end students_per_group_l168_168404


namespace int_solutions_to_inequalities_l168_168697

theorem int_solutions_to_inequalities :
  { x : ℤ | -5 * x ≥ 3 * x + 15 } ∩
  { x : ℤ | -3 * x ≤ 9 } ∩
  { x : ℤ | 7 * x ≤ -14 } = { -3, -2 } :=
by {
  sorry
}

end int_solutions_to_inequalities_l168_168697


namespace McKenna_stuffed_animals_count_l168_168430

def stuffed_animals (M K T : ℕ) : Prop :=
  M + K + T = 175 ∧ K = 2 * M ∧ T = K + 5

theorem McKenna_stuffed_animals_count (M K T : ℕ) (h : stuffed_animals M K T) : M = 34 :=
by
  sorry

end McKenna_stuffed_animals_count_l168_168430


namespace donuts_left_for_coworkers_l168_168454

theorem donuts_left_for_coworkers :
  ∀ (total_donuts gluten_free regular gluten_free_chocolate gluten_free_plain regular_chocolate regular_plain consumed_gluten_free consumed_regular afternoon_gluten_free_chocolate afternoon_gluten_free_plain afternoon_regular_chocolate afternoon_regular_plain left_gluten_free_chocolate left_gluten_free_plain left_regular_chocolate left_regular_plain),
  total_donuts = 30 →
  gluten_free = 12 →
  regular = 18 →
  gluten_free_chocolate = 6 →
  gluten_free_plain = 6 →
  regular_chocolate = 11 →
  regular_plain = 7 →
  consumed_gluten_free = 1 →
  consumed_regular = 1 →
  afternoon_gluten_free_chocolate = 2 →
  afternoon_gluten_free_plain = 1 →
  afternoon_regular_chocolate = 2 →
  afternoon_regular_plain = 1 →
  left_gluten_free_chocolate = gluten_free_chocolate - consumed_gluten_free * 0.5 - afternoon_gluten_free_chocolate →
  left_gluten_free_plain = gluten_free_plain - consumed_gluten_free * 0.5 - afternoon_gluten_free_plain →
  left_regular_chocolate = regular_chocolate - consumed_regular * 1 - afternoon_regular_chocolate →
  left_regular_plain = regular_plain - consumed_regular * 0 - afternoon_regular_plain →
  left_gluten_free_chocolate + left_gluten_free_plain + left_regular_chocolate + left_regular_plain = 23 :=
by
  intros
  sorry

end donuts_left_for_coworkers_l168_168454


namespace simplify_expression_l168_168914

theorem simplify_expression : (Real.cos (18 * Real.pi / 180) * Real.cos (42 * Real.pi / 180) - 
                              Real.cos (72 * Real.pi / 180) * Real.sin (42 * Real.pi / 180) = 1 / 2) :=
by
  sorry

end simplify_expression_l168_168914


namespace find_m_l168_168464

theorem find_m (m : ℕ) (h : m * (m - 1) * (m - 2) * (m - 3) * (m - 4) = 2 * m * (m - 1) * (m - 2)) : m = 5 :=
sorry

end find_m_l168_168464


namespace fraction_not_covered_correct_l168_168121

def area_floor : ℕ := 64
def width_rug : ℕ := 2
def length_rug : ℕ := 7
def area_rug := width_rug * length_rug
def area_not_covered := area_floor - area_rug
def fraction_not_covered := (area_not_covered : ℚ) / area_floor

theorem fraction_not_covered_correct :
  fraction_not_covered = 25 / 32 :=
by
  -- Proof goes here
  sorry

end fraction_not_covered_correct_l168_168121


namespace exist_m_squared_plus_9_mod_2_pow_n_minus_1_l168_168663

theorem exist_m_squared_plus_9_mod_2_pow_n_minus_1 (n : ℕ) (hn : n > 0) :
  (∃ m : ℤ, (m^2 + 9) % (2^n - 1) = 0) ↔ ∃ k : ℕ, n = 2^k :=
by
  sorry

end exist_m_squared_plus_9_mod_2_pow_n_minus_1_l168_168663


namespace proper_subset_of_A_l168_168134

def A : Set ℝ := {x | x^2 < 5 * x}

theorem proper_subset_of_A :
  (∀ x, x ∈ Set.Ioc 1 5 → x ∈ A ∧ ∀ y, y ∈ A → y ∉ Set.Ioc 1 5 → ¬(Set.Ioc 1 5 = A)) :=
sorry

end proper_subset_of_A_l168_168134


namespace evaluate_three_star_twostar_one_l168_168467

def operator_star (a b : ℕ) : ℕ :=
  a^b - b^a

theorem evaluate_three_star_twostar_one : operator_star 3 (operator_star 2 1) = 2 := 
  by
    sorry

end evaluate_three_star_twostar_one_l168_168467


namespace number_of_decks_bought_l168_168055

theorem number_of_decks_bought :
  ∃ T : ℕ, (8 * T + 5 * 8 = 64) ∧ T = 3 :=
by
  sorry

end number_of_decks_bought_l168_168055


namespace loci_of_square_view_l168_168686

-- Definitions based on the conditions in a)
def square (A B C D : Point) : Prop := -- Formalize what it means to be a square
sorry

def region1 (P : Point) (A B : Point) : Prop := -- Formalize the definition of region 1
sorry

def region2 (P : Point) (B C : Point) : Prop := -- Formalize the definition of region 2
sorry

-- Additional region definitions (3 through 9)
-- ...

def visible_side (P A B : Point) : Prop := -- Definition of a visible side from a point
sorry

def visible_diagonal (P A C : Point) : Prop := -- Definition of a visible diagonal from a point
sorry

def loci_of_angles (angle : ℝ) : Set Point := -- Definition of loci for a given angle
sorry

-- Main problem statement with the question and conditions as hypotheses
theorem loci_of_square_view (A B C D P : Point) (angle : ℝ) :
    square A B C D →
    (∀ P, (visible_side P A B ∨ visible_side P B C ∨ visible_side P C D ∨ visible_side P D A → 
             P ∈ loci_of_angles angle) ∧ 
         ((region1 P A B ∨ region2 P B C) → visible_diagonal P A C)) →
    -- Additional conditions here
    True :=
-- Prove that the loci is as described in the solution
sorry

end loci_of_square_view_l168_168686


namespace find_f_at_75_l168_168295

variables (f : ℝ → ℝ) (h₀ : ∀ x, f (x + 2) = -f x)
variables (h₁ : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x)
variables (h₂ : ∀ x, f (-x) = -f x)

theorem find_f_at_75 : f 7.5 = -0.5 := by
  sorry

end find_f_at_75_l168_168295


namespace neg_p_l168_168009

theorem neg_p :
  (¬ (∀ x : ℝ, x^2 + 1 > 0)) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) :=
by
  sorry

end neg_p_l168_168009


namespace p_necessary_for_q_l168_168287

variable (x : ℝ)

def p := (x - 3) * (|x| + 1) < 0
def q := |1 - x| < 2

theorem p_necessary_for_q : (∀ x, q x → p x) ∧ (∃ x, q x) ∧ (∃ x, ¬(p x ∧ q x)) := by
  sorry

end p_necessary_for_q_l168_168287


namespace evaluate_expression_l168_168553

theorem evaluate_expression : (3^3)^4 = 531441 :=
by sorry

end evaluate_expression_l168_168553


namespace solve_expression_l168_168120

noncomputable def expression : ℝ := 5 * 1.6 - 2 * 1.4 / 1.3

theorem solve_expression : expression = 5.8462 := 
by 
  sorry

end solve_expression_l168_168120


namespace complement_union_eq_zero_or_negative_l168_168600

def U : Set ℝ := Set.univ

def P : Set ℝ := { x | x > 1 }

def Q : Set ℝ := { x | x * (x - 2) < 0 }

theorem complement_union_eq_zero_or_negative :
  (U \ (P ∪ Q)) = { x | x ≤ 0 } := by
  sorry

end complement_union_eq_zero_or_negative_l168_168600


namespace circle_passing_through_points_l168_168363

theorem circle_passing_through_points :
  ∃ D E F : ℝ, 
    (∀ x y : ℝ, 
      (x = 0 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = 4 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = -1 ∧ y = 1 → x^2 + y^2 + D * x + E * y + F = 0)) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 → x^2 + y^2 - 4 * x - 6 * y = 0) :=
sorry

end circle_passing_through_points_l168_168363


namespace find_subtracted_number_l168_168463

variable (initial_number : Real)
variable (sum : Real := initial_number + 5)
variable (product : Real := sum * 7)
variable (quotient : Real := product / 5)
variable (remainder : Real := 33)

theorem find_subtracted_number 
  (initial_number_eq : initial_number = 22.142857142857142)
  : quotient - remainder = 5 := by
  sorry

end find_subtracted_number_l168_168463


namespace m_coins_can_collect_k_rubles_l168_168297

theorem m_coins_can_collect_k_rubles
  (a1 a2 a3 a4 a5 a6 a7 m k : ℕ)
  (h1 : a1 + 2 * a2 + 5 * a3 + 10 * a4 + 20 * a5 + 50 * a6 + 100 * a7 = m)
  (h2 : a1 + a2 + a3 + a4 + a5 + a6 + a7 = k) :
  ∃ (b1 b2 b3 b4 b5 b6 b7 : ℕ), 
    100 * (b1 + 2 * b2 + 5 * b3 + 10 * b4 + 20 * b5 + 50 * b6 + 100 * b7) = 100 * k ∧ 
    b1 + b2 + b3 + b4 + b5 + b6 + b7 = m := 
sorry

end m_coins_can_collect_k_rubles_l168_168297


namespace simplify_expression_l168_168376

variable (x : ℝ)

theorem simplify_expression : 3 * x + 4 * x^3 + 2 - (7 - 3 * x - 4 * x^3) = 8 * x^3 + 6 * x - 5 := 
by 
  sorry

end simplify_expression_l168_168376


namespace find_locus_of_p_l168_168598

noncomputable def locus_of_point_p (a b : ℝ) : Set (ℝ × ℝ) :=
{p | (p.snd = 0 ∧ -a < p.fst ∧ p.fst < a) ∨ (p.fst = a^2 / Real.sqrt (a^2 + b^2))}

theorem find_locus_of_p (a b : ℝ) (P : ℝ × ℝ) :
  (∃ (x0 y0: ℝ),
      P = (x0, y0) ∧
      ( ∃ (x1 y1 x2 y2 : ℝ),
        (x0 ≠ 0 ∨ y0 ≠ 0) ∧
        (x1 ≠ x2 ∨ y1 ≠ y2) ∧
        (y0 = 0 ∨ (b^2 * x0 = -a^2 * (x0 - Real.sqrt (a^2 + b^2)))) ∧
        ((y0 = 0 ∧ -a < x0 ∧ x0 < a) ∨ x0 = a^2 / Real.sqrt (a^2 + b^2)))) ↔ 
  P ∈ locus_of_point_p a b :=
sorry

end find_locus_of_p_l168_168598


namespace find_CP_A_l168_168548

noncomputable def CP_A : Float := 173.41
def SP_B (CP_A : Float) : Float := 1.20 * CP_A
def SP_C (SP_B : Float) : Float := 1.25 * SP_B
def TC_C (SP_C : Float) : Float := 1.15 * SP_C
def SP_D1 (TC_C : Float) : Float := 1.30 * TC_C
def SP_D2 (SP_D1 : Float) : Float := 0.90 * SP_D1
def SP_D2_actual : Float := 350

theorem find_CP_A : 
  (SP_D2 (SP_D1 (TC_C (SP_C (SP_B CP_A))))) = SP_D2_actual → 
  CP_A = 173.41 := sorry

end find_CP_A_l168_168548


namespace triangle_inequality_l168_168419

theorem triangle_inequality (a b c : ℝ) (h : a^2 = b^2 + c^2) : 
  (b - c)^2 * (a^2 + 4 * b * c)^2 ≤ 2 * a^6 :=
by
  sorry

end triangle_inequality_l168_168419


namespace greatest_sum_of_consecutive_integers_product_less_500_l168_168328

theorem greatest_sum_of_consecutive_integers_product_less_500 :
  ∃ n : ℤ, n * (n + 1) < 500 ∧ (n + (n + 1)) = 43 :=
by
  sorry

end greatest_sum_of_consecutive_integers_product_less_500_l168_168328


namespace y_intercept_of_line_l168_168423

theorem y_intercept_of_line : 
  ∃ y : ℝ, ∀ x : ℝ, (3 * x - 4 * y = 12) ∧ x = 0 → y = -3 := by
  -- proof skipped
  sorry

end y_intercept_of_line_l168_168423


namespace a1964_eq_neg1_l168_168525

def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ a 2 = 1 ∧ a 3 = -1 ∧ ∀ n ≥ 4, a n = a (n-1) * a (n-3)

theorem a1964_eq_neg1 (a : ℕ → ℤ) (h : seq a) : a 1964 = -1 :=
  by sorry

end a1964_eq_neg1_l168_168525


namespace domain_of_f_l168_168461

noncomputable def f (x : ℝ) : ℝ := (2 * x + 3) / (Real.sqrt (x - 7))

theorem domain_of_f :
  {x : ℝ | ∃ y : ℝ, f y = x} = Set.Ioi 7 := by
  sorry

end domain_of_f_l168_168461


namespace exists_infinite_subset_with_gcd_l168_168414

/-- A set of natural numbers where each number is a product of at most 1987 primes -/
def is_bounded_product_set (A : Set ℕ) (k : ℕ) : Prop :=
  ∀ a ∈ A, ∃ (S : Finset ℕ), (∀ p ∈ S, Nat.Prime p) ∧ a = S.prod id ∧ S.card ≤ k

/-- Prove the existence of an infinite subset and a common gcd for any pair of its elements -/
theorem exists_infinite_subset_with_gcd (A : Set ℕ) (k : ℕ) (hk : k = 1987)
  (hA : is_bounded_product_set A k) (h_inf : Set.Infinite A) :
  ∃ (B : Set ℕ) (b : ℕ), Set.Subset B A ∧ Set.Infinite B ∧ ∀ (x y : ℕ), x ∈ B → y ∈ B → x ≠ y → Nat.gcd x y = b := 
sorry

end exists_infinite_subset_with_gcd_l168_168414


namespace find_greatest_K_l168_168921

theorem find_greatest_K {u v w K : ℝ} (hu : u > 0) (hv : v > 0) (hw : w > 0) (hu2_gt_4vw : u^2 > 4 * v * w) :
  (u^2 - 4 * v * w)^2 > K * (2 * v^2 - u * w) * (2 * w^2 - u * v) ↔ K ≤ 16 := 
sorry

end find_greatest_K_l168_168921


namespace probability_none_hit_l168_168511

theorem probability_none_hit (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) :
  (1 - p)^5 = (1 - p) * (1 - p) * (1 - p) * (1 - p) * (1 - p) :=
by sorry

end probability_none_hit_l168_168511


namespace factorization_correct_l168_168283

noncomputable def factorize_poly (m n : ℕ) : ℕ := 2 * m * n ^ 2 - 12 * m * n + 18 * m

theorem factorization_correct (m n : ℕ) :
  factorize_poly m n = 2 * m * (n - 3) ^ 2 :=
by
  sorry

end factorization_correct_l168_168283


namespace quadratic_equal_roots_relation_l168_168802

theorem quadratic_equal_roots_relation (a b c : ℝ) (h₁ : b ≠ c) 
  (h₂ : ∀ x : ℝ, (b - c) * x^2 + (a - b) * x + (c - a) = 0 → 
          (a - b)^2 - 4 * (b - c) * (c - a) = 0) : 
  c = (a + b) / 2 := sorry

end quadratic_equal_roots_relation_l168_168802


namespace bus_ride_difference_l168_168776

theorem bus_ride_difference (vince_bus_length zachary_bus_length : Real)
    (h_vince : vince_bus_length = 0.62)
    (h_zachary : zachary_bus_length = 0.5) :
    vince_bus_length - zachary_bus_length = 0.12 :=
by
  sorry

end bus_ride_difference_l168_168776


namespace polynomial_degrees_l168_168647

-- Define the degree requirement for the polynomial.
def polynomial_deg_condition (m n : ℕ) : Prop :=
  2 + m = 5 ∧ n - 2 = 0 ∧ 2 + 2 = 5

theorem polynomial_degrees (m n : ℕ) (h : polynomial_deg_condition m n) : m - n = 1 :=
by
  have h1 : 2 + m = 5 := h.1
  have h2 : n - 2 = 0 := h.2.1
  have h3 := h.2.2
  have : m = 3 := by linarith
  have : n = 2 := by linarith
  linarith

end polynomial_degrees_l168_168647


namespace arithmetic_sequence_a3_l168_168144

theorem arithmetic_sequence_a3 (a : ℕ → ℕ) (h1 : a 6 = 6) (h2 : a 9 = 9) : a 3 = 3 :=
by
  -- proof goes here
  sorry

end arithmetic_sequence_a3_l168_168144


namespace solve_for_x_l168_168166

theorem solve_for_x (x y : ℝ) (h₁ : x - y = 8) (h₂ : x + y = 16) (h₃ : x * y = 48) : x = 12 :=
sorry

end solve_for_x_l168_168166


namespace order_of_abc_l168_168519

noncomputable def a : ℝ := 2017^0
noncomputable def b : ℝ := 2015 * 2017 - 2016^2
noncomputable def c : ℝ := ((-2/3)^2016) * ((3/2)^2017)

theorem order_of_abc : b < a ∧ a < c := by
  -- proof omitted
  sorry

end order_of_abc_l168_168519


namespace triangle_circumscribed_circle_diameter_l168_168063

noncomputable def circumscribed_circle_diameter (a : ℝ) (A : ℝ) : ℝ :=
  a / Real.sin A

theorem triangle_circumscribed_circle_diameter :
  let a := 16
  let A := Real.pi / 4   -- 45 degrees in radians
  circumscribed_circle_diameter a A = 16 * Real.sqrt 2 :=
by
  sorry

end triangle_circumscribed_circle_diameter_l168_168063


namespace find_number_l168_168156

theorem find_number (x : ℤ) (h : x + x^2 + 15 = 96) : x = -9 :=
sorry

end find_number_l168_168156


namespace no_values_satisfy_equation_l168_168968

-- Define the sum of the digits function S
noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- Define the sum of digits of the sum of the digits function S(S(n))
noncomputable def sum_of_sum_of_digits (n : ℕ) : ℕ :=
  sum_of_digits (sum_of_digits n)

-- Theorem statement about the number of n satisfying n + S(n) + S(S(n)) = 2099
theorem no_values_satisfy_equation :
  (∃ n : ℕ, n > 0 ∧ n + sum_of_digits n + sum_of_sum_of_digits n = 2099) ↔ False := sorry

end no_values_satisfy_equation_l168_168968


namespace collinear_vectors_l168_168507

open Real

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (OA OB OP : V) (m n : ℝ)

-- Given conditions
def non_collinear (OA OB : V) : Prop :=
  ∀ (t : ℝ), OA ≠ t • OB

def collinear_points (P A B : V) : Prop :=
  ∃ (t : ℝ), P - A = t • (B - A)

def linear_combination (OP OA OB : V) (m n : ℝ) : Prop :=
  OP = m • OA + n • OB

-- The theorem statement
theorem collinear_vectors (noncol : non_collinear OA OB)
  (collinearPAB : collinear_points OP OA OB)
  (lin_comb : linear_combination OP OA OB m n) :
  m = 2 ∧ n = -1 := by
sorry

end collinear_vectors_l168_168507


namespace Carly_fourth_week_running_distance_l168_168500

theorem Carly_fourth_week_running_distance :
  let week1_distance_per_day := 2
  let week2_distance_per_day := (week1_distance_per_day * 2) + 3
  let week3_distance_per_day := week2_distance_per_day * (9 / 7)
  let week4_intended_distance_per_day := week3_distance_per_day * 0.9
  let week4_actual_distance_per_day := week4_intended_distance_per_day * 0.5
  let week4_days_run := 5 -- due to 2 rest days
  (week4_actual_distance_per_day * week4_days_run) = 20.25 := 
by 
    -- We use sorry here to skip the proof
    sorry

end Carly_fourth_week_running_distance_l168_168500


namespace proof_sum_q_p_x_l168_168645

def p (x : ℝ) : ℝ := |x| - 3
def q (x : ℝ) : ℝ := -|x|

-- define the list of x values
def x_values : List ℝ := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

-- define q_p_x to apply q to p of each x
def q_p_x : List ℝ := x_values.map (λ x => q (p x))

-- define the sum of q(p(x)) for given x values
def sum_q_p_x : ℝ := q_p_x.sum

theorem proof_sum_q_p_x : sum_q_p_x = -15 := by
  -- steps of solution
  sorry

end proof_sum_q_p_x_l168_168645


namespace remainder_21_l168_168422

theorem remainder_21 (y : ℤ) (k : ℤ) (h : y = 288 * k + 45) : y % 24 = 21 := 
  sorry

end remainder_21_l168_168422


namespace compound_proposition_C_l168_168367

open Real

def p : Prop := ∃ x : ℝ, x - 2 > log x 
def q : Prop := ∀ x : ℝ, sin x < x

theorem compound_proposition_C : p ∧ ¬q :=
by sorry

end compound_proposition_C_l168_168367


namespace bridesmaids_count_l168_168482

theorem bridesmaids_count
  (hours_per_dress : ℕ)
  (hours_per_week : ℕ)
  (weeks : ℕ)
  (total_hours : ℕ)
  (dresses : ℕ) :
  hours_per_dress = 12 →
  hours_per_week = 4 →
  weeks = 15 →
  total_hours = hours_per_week * weeks →
  dresses = total_hours / hours_per_dress →
  dresses = 5 := by
  sorry

end bridesmaids_count_l168_168482


namespace triangle_side_lengths_l168_168315

open Real

theorem triangle_side_lengths (a b c : ℕ) (R : ℝ)
    (h1 : a * a + 4 * d * d = 2500)
    (h2 : b * b + 4 * e * e = 2500)
    (h3 : R = 12.5)
    (h4 : (2:ℝ) * d ≤ a)
    (h5 : (2:ℝ) * e ≤ b)
    (h6 : a > b)
    (h7 : a ≠ b)
    (h8 : 2 * R = 25) :
    (a, b, c) = (15, 7, 20) := by
  sorry

end triangle_side_lengths_l168_168315


namespace minimum_value_of_product_l168_168729

theorem minimum_value_of_product (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_sum : (a / b + b / c + c / a) + (b / a + c / b + a / c) = 9) :
  (a / b + b / c + c / a) * (b / a + c / b + a / c) = 30 := 
sorry

end minimum_value_of_product_l168_168729


namespace probability_standard_bulb_l168_168873

structure FactoryConditions :=
  (P_H1 : ℝ)
  (P_H2 : ℝ)
  (P_H3 : ℝ)
  (P_A_H1 : ℝ)
  (P_A_H2 : ℝ)
  (P_A_H3 : ℝ)

theorem probability_standard_bulb (conditions : FactoryConditions) : 
  conditions.P_H1 = 0.45 → 
  conditions.P_H2 = 0.40 → 
  conditions.P_H3 = 0.15 →
  conditions.P_A_H1 = 0.70 → 
  conditions.P_A_H2 = 0.80 → 
  conditions.P_A_H3 = 0.81 → 
  (conditions.P_H1 * conditions.P_A_H1 + 
   conditions.P_H2 * conditions.P_A_H2 + 
   conditions.P_H3 * conditions.P_A_H3) = 0.7565 :=
by 
  intros h1 h2 h3 a_h1 a_h2 a_h3 
  sorry

end probability_standard_bulb_l168_168873


namespace div_by_3kp1_iff_div_by_3k_l168_168848

theorem div_by_3kp1_iff_div_by_3k (m n k : ℕ) (h1 : m > n) :
  (3 ^ (k + 1)) ∣ (4 ^ m - 4 ^ n) ↔ (3 ^ k) ∣ (m - n) := 
sorry

end div_by_3kp1_iff_div_by_3k_l168_168848


namespace exists_p_q_for_integer_roots_l168_168775

theorem exists_p_q_for_integer_roots : 
  ∃ (p q : ℤ), ∀ k (hk : k ∈ (Finset.range 10)), 
    ∃ (r1 r2 : ℤ), (r1 + r2 = -(p + k)) ∧ (r1 * r2 = (q + k)) :=
sorry

end exists_p_q_for_integer_roots_l168_168775


namespace probability_of_5_distinct_dice_rolls_is_5_over_54_l168_168175

def count_distinct_dice_rolls : ℕ :=
  6 * 5 * 4 * 3 * 2

def total_dice_rolls : ℕ :=
  6 ^ 5

def probability_of_distinct_rolls : ℚ :=
  count_distinct_dice_rolls / total_dice_rolls

theorem probability_of_5_distinct_dice_rolls_is_5_over_54 : 
  probability_of_distinct_rolls = 5 / 54 :=
by
  sorry

end probability_of_5_distinct_dice_rolls_is_5_over_54_l168_168175


namespace sequence_difference_l168_168068

theorem sequence_difference :
  ∃ a : ℕ → ℕ,
    a 1 = 1 ∧ a 2 = 1 ∧
    (∀ n ≥ 1, (a (n + 2) : ℚ) / a (n + 1) - (a (n + 1) : ℚ) / a n = 1) ∧
    a 6 - a 5 = 96 :=
sorry

end sequence_difference_l168_168068


namespace vertex_on_x_axis_iff_t_eq_neg_4_l168_168088

theorem vertex_on_x_axis_iff_t_eq_neg_4 (t : ℝ) :
  (∃ x : ℝ, (4 + t) = 0) ↔ t = -4 :=
by
  sorry

end vertex_on_x_axis_iff_t_eq_neg_4_l168_168088


namespace value_of_k_l168_168022

theorem value_of_k (x y k : ℤ) (h1 : 2 * x - y = 5 * k + 6) (h2 : 4 * x + 7 * y = k) (h3 : x + y = 2024)
: k = 2023 :=
by
  sorry

end value_of_k_l168_168022


namespace diagonal_length_l168_168781

noncomputable def convertHectaresToSquareMeters (hectares : ℝ) : ℝ :=
  hectares * 10000

noncomputable def sideLength (areaSqMeters : ℝ) : ℝ :=
  Real.sqrt areaSqMeters

noncomputable def diagonal (side : ℝ) : ℝ :=
  side * Real.sqrt 2

theorem diagonal_length (area : ℝ) (h : area = 1 / 2) :
  let areaSqMeters := convertHectaresToSquareMeters area
  let side := sideLength areaSqMeters
  let diag := diagonal side
  abs (diag - 100) < 1 :=
by
  sorry

end diagonal_length_l168_168781


namespace simplify_and_rationalize_l168_168396

theorem simplify_and_rationalize : (1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5) :=
by sorry

end simplify_and_rationalize_l168_168396


namespace number_of_members_l168_168019

def cost_knee_pads : ℤ := 6
def cost_jersey : ℤ := cost_knee_pads + 7
def total_cost_per_member : ℤ := 2 * (cost_knee_pads + cost_jersey)
def total_expenditure : ℤ := 3120

theorem number_of_members (n : ℤ) (h : n * total_cost_per_member = total_expenditure) : n = 82 :=
sorry

end number_of_members_l168_168019


namespace theater_loss_l168_168274

/-- 
A movie theater has a total capacity of 50 people and charges $8 per ticket.
On a Tuesday night, they only sold 24 tickets. 
Prove that the revenue lost by not selling out is $208.
-/
theorem theater_loss 
  (capacity : ℕ) 
  (price : ℕ) 
  (sold_tickets : ℕ) 
  (h_cap : capacity = 50) 
  (h_price : price = 8) 
  (h_sold : sold_tickets = 24) : 
  capacity * price - sold_tickets * price = 208 :=
by
  sorry

end theater_loss_l168_168274


namespace granola_bars_distribution_l168_168233

theorem granola_bars_distribution
  (total_bars : ℕ)
  (eaten_bars : ℕ)
  (num_children : ℕ)
  (remaining_bars := total_bars - eaten_bars)
  (bars_per_child := remaining_bars / num_children) :
  total_bars = 200 → eaten_bars = 80 → num_children = 6 → bars_per_child = 20 :=
by
  intros h1 h2 h3
  sorry

end granola_bars_distribution_l168_168233


namespace log_base2_probability_l168_168077

theorem log_base2_probability (n : ℕ) (h1 : 100 ≤ n ∧ n ≤ 999) (h2 : ∃ k : ℕ, n = 2^k) : 
  ∃ p : ℚ, p = 1/300 :=
  sorry

end log_base2_probability_l168_168077


namespace calc_expression_l168_168300

theorem calc_expression (a : ℝ) : 4 * a * a^3 - a^4 = 3 * a^4 := by
  sorry

end calc_expression_l168_168300


namespace points_per_member_l168_168988

def numMembersTotal := 12
def numMembersAbsent := 4
def totalPoints := 64

theorem points_per_member (h : numMembersTotal - numMembersAbsent = 12 - 4) :
  (totalPoints / (numMembersTotal - numMembersAbsent)) = 8 := 
  sorry

end points_per_member_l168_168988


namespace intersection_sets_l168_168180

-- Define the sets A and B as given in the problem conditions
def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {0, 2, 4}

-- Lean theorem statement for proving the intersection of sets A and B is {0, 2}
theorem intersection_sets : A ∩ B = {0, 2} := 
by
  sorry

end intersection_sets_l168_168180


namespace simplify_polynomial_l168_168364

variable (x : ℝ)

theorem simplify_polynomial : 
  (2 * x^4 + 3 * x^3 - 5 * x + 6) + (-6 * x^4 - 2 * x^3 + 3 * x^2 + 5 * x - 4) = 
  -4 * x^4 + x^3 + 3 * x^2 + 2 :=
by
  sorry

end simplify_polynomial_l168_168364


namespace function_increasing_iff_l168_168755

noncomputable def f (x a : ℝ) : ℝ := (1 / 3) * x^3 - a * x

theorem function_increasing_iff (a : ℝ) :
  (∀ x : ℝ, 0 < x^2 - a) ↔ a ≤ 0 :=
by
  sorry

end function_increasing_iff_l168_168755


namespace cost_of_parts_per_tire_repair_is_5_l168_168747

-- Define the given conditions
def charge_per_tire_repair : ℤ := 20
def num_tire_repairs : ℤ := 300
def charge_per_complex_repair : ℤ := 300
def num_complex_repairs : ℤ := 2
def cost_per_complex_repair_parts : ℤ := 50
def retail_shop_profit : ℤ := 2000
def fixed_expenses : ℤ := 4000
def total_profit : ℤ := 3000

-- Define the calculation for total revenue
def total_revenue : ℤ := 
    (charge_per_tire_repair * num_tire_repairs) + 
    (charge_per_complex_repair * num_complex_repairs) + 
    retail_shop_profit

-- Define the calculation for total expenses
def total_expenses : ℤ := total_revenue - total_profit

-- Define the calculation for parts cost of tire repairs
def parts_cost_tire_repairs : ℤ := 
    total_expenses - (cost_per_complex_repair_parts * num_complex_repairs) - fixed_expenses

def cost_per_tire_repair : ℤ := parts_cost_tire_repairs / num_tire_repairs

-- The statement to be proved
theorem cost_of_parts_per_tire_repair_is_5 : cost_per_tire_repair = 5 := by
    sorry

end cost_of_parts_per_tire_repair_is_5_l168_168747


namespace simplify_evaluate_expression_l168_168125

theorem simplify_evaluate_expression (a b : ℚ) (h1 : a = -2) (h2 : b = 1/5) :
    2 * a * b^2 - (6 * a^3 * b + 2 * (a * b^2 - (1/2) * a^3 * b)) = 8 := 
by
  sorry

end simplify_evaluate_expression_l168_168125


namespace petya_friends_l168_168451

theorem petya_friends : ∃ x : ℕ, (5 * x + 8 = 6 * x - 11) → x = 19 :=
by
  use 19
  intro h
  sorry

end petya_friends_l168_168451


namespace parallel_lines_m_l168_168380

theorem parallel_lines_m (m : ℝ) :
  (∀ (x y : ℝ), 3 * m * x + (m + 2) * y + 1 = 0) ∧
  (∀ (x y : ℝ), (m - 2) * x + (m + 2) * y + 2 = 0) ∧
  (∀ (x₁ y₁ x₂ y₂ : ℝ), (3 * m) / (m + 2) = (m - 2) / (m + 2)) →
  (m = -1 ∨ m = -2) :=
sorry

end parallel_lines_m_l168_168380


namespace cubed_expression_value_l168_168550

open Real

theorem cubed_expression_value (a b c : ℝ)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : a + b + 2 * c = 0) :
  (a^3 + b^3 + 2 * c^3) / (a * b * c) = -3 * (a^2 - a * b + b^2) / (2 * a * b) :=
  sorry

end cubed_expression_value_l168_168550


namespace wire_length_before_cut_l168_168466

theorem wire_length_before_cut (S : ℝ) (L : ℝ) (h1 : S = 4) (h2 : S = (2/5) * L) : S + L = 14 :=
by 
  sorry

end wire_length_before_cut_l168_168466


namespace greatest_possible_x_l168_168561

-- Define the numbers and the lcm condition
def num1 := 12
def num2 := 18
def lcm_val := 108

-- Function to calculate the lcm of three numbers
def lcm3 (a b c : ℕ) := Nat.lcm (Nat.lcm a b) c

-- Proposition stating the problem condition
theorem greatest_possible_x (x : ℕ) (h : lcm3 x num1 num2 = lcm_val) : x ≤ lcm_val := sorry

end greatest_possible_x_l168_168561


namespace number_of_members_l168_168420

theorem number_of_members (n : ℕ) (h : n^2 = 9801) : n = 99 :=
sorry

end number_of_members_l168_168420


namespace weight_of_substance_l168_168866

variable (k W1 W2 : ℝ)

theorem weight_of_substance (h1 : ∃ (k : ℝ), ∀ (V W : ℝ), V = k * W)
  (h2 : 48 = k * W1) (h3 : 36 = k * 84) : 
  (∃ (W2 : ℝ), 48 = (36 / 84) * W2) → W2 = 112 := 
by
  sorry

end weight_of_substance_l168_168866


namespace initial_boys_l168_168345

theorem initial_boys (p : ℝ) (initial_boys : ℝ) (final_boys : ℝ) (final_groupsize : ℝ) : 
  (initial_boys = 0.35 * p) ->
  (final_boys = 0.35 * p - 1) ->
  (final_groupsize = p + 3) ->
  (final_boys / final_groupsize = 0.3) ->
  initial_boys = 13 := 
by
  sorry

end initial_boys_l168_168345


namespace proof_problem_l168_168994

noncomputable def f (x : ℝ) := Real.tan (x + (Real.pi / 4))

theorem proof_problem :
  (- (3 * Real.pi) / 4 < 1 - Real.pi ∧ 1 - Real.pi < -1 ∧ -1 < 0 ∧ 0 < Real.pi / 4) →
  f 0 > f (-1) ∧ f (-1) > f 1 := by
  sorry

end proof_problem_l168_168994


namespace find_rate_percent_l168_168510

def P : ℝ := 800
def SI : ℝ := 200
def T : ℝ := 4

theorem find_rate_percent (R : ℝ) :
  SI = P * R * T / 100 → R = 6.25 :=
by
  sorry

end find_rate_percent_l168_168510


namespace incorrect_statement_l168_168544

theorem incorrect_statement :
  ¬ (∀ (l1 l2 l3 : ℝ → ℝ → Prop), 
      (∀ (x y : ℝ), l3 x y → l1 x y) ∧ 
      (∀ (x y : ℝ), l3 x y → l2 x y) → 
      (∀ (x y : ℝ), l1 x y → l2 x y)) :=
by sorry

end incorrect_statement_l168_168544


namespace num_enemies_left_l168_168115

-- Definitions of conditions
def points_per_enemy : Nat := 5
def total_enemies : Nat := 8
def earned_points : Nat := 10

-- Theorem statement to prove the number of undefeated enemies
theorem num_enemies_left (points_per_enemy total_enemies earned_points : Nat) : 
    (earned_points / points_per_enemy) <= total_enemies →
    total_enemies - (earned_points / points_per_enemy) = 6 := by
  sorry

end num_enemies_left_l168_168115


namespace circumference_proportionality_l168_168661

theorem circumference_proportionality (r : ℝ) (C : ℝ) (k : ℝ) (π : ℝ)
  (h1 : C = k * r)
  (h2 : C = 2 * π * r) :
  k = 2 * π :=
sorry

end circumference_proportionality_l168_168661


namespace total_seeds_gray_sections_combined_l168_168276

noncomputable def total_seeds_first_circle : ℕ := 87
noncomputable def seeds_white_first_circle : ℕ := 68
noncomputable def total_seeds_second_circle : ℕ := 110
noncomputable def seeds_white_second_circle : ℕ := 68

theorem total_seeds_gray_sections_combined :
  (total_seeds_first_circle - seeds_white_first_circle) +
  (total_seeds_second_circle - seeds_white_second_circle) = 61 :=
by
  sorry

end total_seeds_gray_sections_combined_l168_168276


namespace measure_of_angle_C_l168_168250

-- Definitions of the angles
def angles (A B C : ℝ) : Prop :=
  -- Conditions: measure of angle A is 1/4 of measure of angle B
  A = (1 / 4) * B ∧
  -- Lines p and q are parallel so alternate interior angles are equal
  C = A ∧
  -- Since angles B and C are supplementary
  B + C = 180

-- The problem in Lean 4 statement: Prove that C = 36 given the conditions
theorem measure_of_angle_C (A B C : ℝ) (h : angles A B C) : C = 36 := sorry

end measure_of_angle_C_l168_168250


namespace n_in_S_implies_n_squared_in_S_l168_168327

-- Definition of the set S
def S : Set ℕ := {n | ∃ a b c d e f : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧ 
                      n - 1 = a^2 + b^2 ∧ n = c^2 + d^2 ∧ n + 1 = e^2 + f^2}

-- The proof goal
theorem n_in_S_implies_n_squared_in_S (n : ℕ) (h : n ∈ S) : n^2 ∈ S :=
by
  sorry

end n_in_S_implies_n_squared_in_S_l168_168327


namespace point_in_fourth_quadrant_l168_168910

def point : ℝ × ℝ := (1, -2)

def is_fourth_quadrant (p: ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 < 0

theorem point_in_fourth_quadrant : is_fourth_quadrant point :=
by
  sorry

end point_in_fourth_quadrant_l168_168910


namespace roots_of_quadratic_l168_168706

theorem roots_of_quadratic (x1 x2 : ℝ) (h : ∀ x, x^2 - 3 * x - 2 = 0 → x = x1 ∨ x = x2) :
  x1 * x2 + x1 + x2 = 1 :=
sorry

end roots_of_quadratic_l168_168706


namespace parabola_focus_distance_l168_168854

theorem parabola_focus_distance (M : ℝ × ℝ) (h1 : (M.2)^2 = 4 * M.1) (h2 : dist M (1, 0) = 4) : M.1 = 3 :=
sorry

end parabola_focus_distance_l168_168854


namespace simplification_at_negative_two_l168_168330

noncomputable def simplify_expression (x : ℚ) : ℚ :=
  ((x^2 - 4*x + 4) / (x^2 - 1)) / ((x^2 - 2*x) / (x + 1)) + (1 / (x - 1))

theorem simplification_at_negative_two :
  ∀ x : ℚ, -2 ≤ x ∧ x ≤ 2 ∧ x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 → simplify_expression (-2) = -1 :=
by simp [simplify_expression]; sorry

end simplification_at_negative_two_l168_168330


namespace hyperbola_asymptote_solution_l168_168193

theorem hyperbola_asymptote_solution (b : ℝ) (hb : b > 0)
  (h_asym : ∀ x y, (∀ y : ℝ, y = (1 / 2) * x ∨ y = - (1 / 2) * x) → (x^2 / 4 - y^2 / b^2 = 1)) :
  b = 1 :=
sorry

end hyperbola_asymptote_solution_l168_168193


namespace muffin_machine_completion_time_l168_168146

theorem muffin_machine_completion_time :
  let start_time := 9 * 60 -- minutes
  let partial_completion_time := (12 * 60) + 15 -- minutes
  let partial_duration := partial_completion_time - start_time
  let fraction_of_day := 1 / 4
  let total_duration := partial_duration / fraction_of_day
  start_time + total_duration = (22 * 60) := -- 10:00 PM in minutes
by
  sorry

end muffin_machine_completion_time_l168_168146


namespace value_of_M_l168_168290

theorem value_of_M :
  let row_seq := [25, 25 + (8 - 25) / 3, 25 + 2 * (8 - 25) / 3, 8, 8 + (8 - 25) / 3, 8 + 2 * (8 - 25) / 3, -9]
  let col_seq1 := [25, 25 - 4, 25 - 8]
  let col_seq2 := [16, 20, 20 + 4]
  let col_seq3 := [-9, -9 - 11/4, -9 - 2 * 11/4, -20]
  let M := -9 - (-11/4)
  M = -6.25 :=
by
  sorry

end value_of_M_l168_168290


namespace solution_set_l168_168654

def solve_inequalities (x : ℝ) : Prop :=
  (3 * x - 2) / (x - 6) ≤ 1 ∧ 2 * x ^ 2 - x - 1 > 0

theorem solution_set : { x : ℝ | solve_inequalities x } = { x : ℝ | (-2 ≤ x ∧ x < 1/2) ∨ (1 < x ∧ x < 6) } :=
by sorry

end solution_set_l168_168654


namespace loss_percentage_l168_168160

theorem loss_percentage (C S : ℕ) (H1 : C = 750) (H2 : S = 600) : (C - S) * 100 / C = 20 := by
  sorry

end loss_percentage_l168_168160


namespace algebraic_expression_value_l168_168346

-- Definitions based on the conditions
variable {a : ℝ}
axiom root_equation : 2 * a^2 + 3 * a - 4 = 0

-- Definition of the problem: Proving that 2a^2 + 3a equals 4.
theorem algebraic_expression_value : 2 * a^2 + 3 * a = 4 :=
by 
  have h : 2 * a^2 + 3 * a - 4 = 0 := root_equation
  have h' : 2 * a^2 + 3 * a = 4 := by sorry
  exact h'

end algebraic_expression_value_l168_168346


namespace find_x_when_parallel_l168_168898

-- Given vectors
def a : ℝ × ℝ := (-2, 4)
def b (x : ℝ) : ℝ × ℝ := (x, -2)

-- Conditional statement: parallel vectors
def parallel_vectors (u v : ℝ × ℝ) : Prop := 
  u.1 * v.2 = u.2 * v.1

-- Proof statement
theorem find_x_when_parallel (x : ℝ) (h : parallel_vectors a (b x)) : x = 1 := 
  sorry

end find_x_when_parallel_l168_168898


namespace infinite_zeros_in_S_l168_168835

-- Define the sequence a_n
def a (n : ℕ) : ℤ :=
  if n % 4 = 0 then -↑(n + 1) else
  if n % 4 = 1 then ↑n else
  if n % 4 = 2 then ↑n else
  -↑(n + 1)

-- Define the sequence S_k as partial sum of a_n
def S : ℕ → ℤ
| 0       => a 0
| (n + 1) => S n + a (n + 1)

-- Proposition: S_k contains infinitely many zeros
theorem infinite_zeros_in_S : ∀ n : ℕ, ∃ m > n, S m = 0 := sorry

end infinite_zeros_in_S_l168_168835


namespace polygon_sides_and_diagonals_l168_168318

theorem polygon_sides_and_diagonals (n : ℕ) :
  (180 * (n - 2) = 3 * 360 + 180) → n = 9 ∧ (n - 3 = 6) :=
by
  intro h_sum_angles
  -- This is where you would provide the proof.
  sorry

end polygon_sides_and_diagonals_l168_168318


namespace quadratic_solution_factoring_solution_l168_168630

-- Define the first problem: Solve 2x^2 - 6x - 5 = 0
theorem quadratic_solution (x : ℝ) : 2 * x^2 - 6 * x - 5 = 0 ↔ x = (3 + Real.sqrt 19) / 2 ∨ x = (3 - Real.sqrt 19) / 2 :=
by
  sorry

-- Define the second problem: Solve 3x(4-x) = 2(x-4)
theorem factoring_solution (x : ℝ) : 3 * x * (4 - x) = 2 * (x - 4) ↔ x = 4 ∨ x = -2 / 3 :=
by
  sorry

end quadratic_solution_factoring_solution_l168_168630


namespace correct_interval_for_monotonic_decrease_l168_168329

noncomputable def f (x : ℝ) : ℝ := |Real.tan (1 / 2 * x - Real.pi / 6)|

theorem correct_interval_for_monotonic_decrease :
  ∀ k : ℤ, ∃ I : Set ℝ,
    I = Set.Ioc (2 * k * Real.pi - 2 * Real.pi / 3) (2 * k * Real.pi + Real.pi / 3) ∧
    ∀ x y, x ∈ I → y ∈ I → x < y → f y < f x :=
sorry

end correct_interval_for_monotonic_decrease_l168_168329


namespace pages_for_ten_dollars_l168_168594

theorem pages_for_ten_dollars (p c pages_per_cent : ℕ) (dollars cents : ℕ) (h1 : p = 5) (h2 : c = 10) (h3 : pages_per_cent = p / c) (h4 : dollars = 10) (h5 : cents = 100 * dollars) :
  (cents * pages_per_cent) = 500 :=
by
  sorry

end pages_for_ten_dollars_l168_168594


namespace base7_difference_l168_168152

theorem base7_difference (a b : ℕ) (h₁ : a = 12100) (h₂ : b = 3666) :
  ∃ c, c = 1111 ∧ (a - b = c) := by
sorry

end base7_difference_l168_168152


namespace part1_part2_l168_168814

-- Part (1)
theorem part1 (a : ℕ → ℕ) (d : ℕ) (S_3 T_3 : ℕ) (h₁ : 3 * a 2 = 3 * a 1 + a 3) (h₂ : S_3 + T_3 = 21) :
  (∀ n, a n = 3 * n) :=
sorry

-- Part (2)
theorem part2 (b : ℕ → ℕ) (d : ℕ) (S_99 T_99 : ℕ) (h₁ : ∀ m n : ℕ, b (m + n) - b m = d * n)
  (h₂ : S_99 - T_99 = 99) : d = 51 / 50 :=
sorry

end part1_part2_l168_168814


namespace min_value_of_expression_l168_168090

open Real

noncomputable def condition (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ (1 / (x + 2) + 1 / (y + 2) = 1 / 4)

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1 / (x + 2) + 1 / (y + 2) = 1 / 4) :
  2 * x + 3 * y = 5 + 4 * sqrt 3 :=
sorry

end min_value_of_expression_l168_168090


namespace eliminate_x3_term_l168_168520

noncomputable def polynomial (n : ℝ) : Polynomial ℝ :=
  (Polynomial.X ^ 2 + Polynomial.C n * Polynomial.X + Polynomial.C 3) *
  (Polynomial.X ^ 2 - Polynomial.C 3 * Polynomial.X)

theorem eliminate_x3_term (n : ℝ) : (polynomial n).coeff 3 = 0 ↔ n = 3 :=
by
  -- sorry to skip the proof for now as it's not required
  sorry

end eliminate_x3_term_l168_168520


namespace sufficient_but_not_necessary_l168_168044

variable {a b : ℝ}

theorem sufficient_but_not_necessary (h : b < a ∧ a < 0) : 1 / a < 1 / b :=
by
  sorry

end sufficient_but_not_necessary_l168_168044


namespace total_students_l168_168918

-- Define n as total number of students
variable (n : ℕ)

-- Define conditions
variable (h1 : 550 ≤ n)
variable (h2 : (n / 10) + 10 ≤ n)

-- Define the proof statement
theorem total_students (h : (550 * 10 + 5) = n ∧ 
                        550 * 10 / n + 10 = 45 + n) : 
                        n = 1000 := by
  sorry

end total_students_l168_168918


namespace pieces_to_cut_l168_168540

-- Define the conditions
def rodLength : ℝ := 42.5  -- Length of the rod
def pieceLength : ℝ := 0.85  -- Length of each piece

-- Define the theorem that needs to be proven
theorem pieces_to_cut (h1 : rodLength = 42.5) (h2 : pieceLength = 0.85) : 
  (rodLength / pieceLength) = 50 := 
  by sorry

end pieces_to_cut_l168_168540


namespace intersect_empty_range_of_a_union_subsets_range_of_a_l168_168351

variable {x a : ℝ}

def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | (x - 6) * (x + 2) > 0}

theorem intersect_empty_range_of_a (h : A a ∩ B = ∅) : -2 ≤ a ∧ a ≤ 3 :=
by
  sorry

theorem union_subsets_range_of_a (h : A a ∪ B = B) : a < -5 ∨ a > 6 :=
by
  sorry

end intersect_empty_range_of_a_union_subsets_range_of_a_l168_168351


namespace physics_kit_prices_l168_168554

theorem physics_kit_prices :
  ∃ (price_A price_B : ℝ), price_A = 180 ∧ price_B = 150 ∧
    price_A = 1.2 * price_B ∧
    9900 / price_A = 7500 / price_B + 5 :=
by
  use 180, 150
  sorry

end physics_kit_prices_l168_168554


namespace quotient_is_20_l168_168460

theorem quotient_is_20 (D d r Q : ℕ) (hD : D = 725) (hd : d = 36) (hr : r = 5) (h : D = d * Q + r) :
  Q = 20 :=
by sorry

end quotient_is_20_l168_168460


namespace fraction_of_draws_is_two_ninths_l168_168267

-- Define the fraction of games that Ben wins and Tom wins
def BenWins : ℚ := 4 / 9
def TomWins : ℚ := 1 / 3

-- Definition of the fraction of games ending in a draw
def fraction_of_draws (BenWins TomWins : ℚ) : ℚ :=
  1 - (BenWins + TomWins)

-- The theorem to be proved
theorem fraction_of_draws_is_two_ninths : fraction_of_draws BenWins TomWins = 2 / 9 :=
by
  sorry

end fraction_of_draws_is_two_ninths_l168_168267


namespace legoland_kangaroos_l168_168161

theorem legoland_kangaroos :
  ∃ (K R : ℕ), R = 5 * K ∧ K + R = 216 ∧ R = 180 := by
  sorry

end legoland_kangaroos_l168_168161


namespace stratified_sampling_distribution_l168_168924

/-- A high school has a total of 2700 students, among which there are 900 freshmen, 
1200 sophomores, and 600 juniors. Using stratified sampling, a sample of 135 students 
is drawn. Prove that the sample contains 45 freshmen, 60 sophomores, and 30 juniors --/
theorem stratified_sampling_distribution :
  let total_students := 2700
  let freshmen := 900
  let sophomores := 1200
  let juniors := 600
  let sample_size := 135
  (sample_size * freshmen / total_students = 45) ∧ 
  (sample_size * sophomores / total_students = 60) ∧ 
  (sample_size * juniors / total_students = 30) :=
by
  sorry

end stratified_sampling_distribution_l168_168924


namespace find_working_hours_for_y_l168_168932

theorem find_working_hours_for_y (Wx Wy Wz Ww : ℝ) (h1 : Wx = 1/8)
  (h2 : Wy + Wz = 1/6) (h3 : Wx + Wz = 1/4) (h4 : Wx + Wy + Ww = 1/5)
  (h5 : Wx + Ww + Wz = 1/3) : 1 / Wy = 24 :=
by
  -- Given the conditions
  -- Wx = 1/8
  -- Wy + Wz = 1/6
  -- Wx + Wz = 1/4
  -- Wx + Wy + Ww = 1/5
  -- Wx + Ww + Wz = 1/3
  -- We need to prove that 1 / Wy = 24
  sorry

end find_working_hours_for_y_l168_168932


namespace minimum_value_expr_l168_168644

theorem minimum_value_expr (a : ℝ) (h₀ : 0 < a) (h₁ : a < 2) : 
  ∃ (m : ℝ), m = (4 / a + 1 / (2 - a)) ∧ m = 9 / 2 :=
by
  sorry

end minimum_value_expr_l168_168644


namespace f_n_2_l168_168486

def f (m n : ℕ) : ℝ :=
if h : m = 1 ∧ n = 1 then 1 else
if h : n > m then 0 else 
sorry -- This would be calculated based on the recursive definition

lemma f_2_2 : f 2 2 = 2 :=
sorry

theorem f_n_2 (n : ℕ) (hn : n ≥ 1) : f n 2 = 2^(n - 1) :=
sorry

end f_n_2_l168_168486


namespace greatest_value_q_minus_r_l168_168850

theorem greatest_value_q_minus_r {x y : ℕ} (hx : x < 10) (hy : y < 10) (hqr : 9 * (x - y) < 70) :
  9 * (x - y) = 63 :=
sorry

end greatest_value_q_minus_r_l168_168850


namespace train_a_distance_traveled_l168_168925

variable (distance : ℝ) (speedA : ℝ) (speedB : ℝ) (relative_speed : ℝ) (time_to_meet : ℝ) 

axiom condition1 : distance = 450
axiom condition2 : speedA = 50
axiom condition3 : speedB = 50
axiom condition4 : relative_speed = speedA + speedB
axiom condition5 : time_to_meet = distance / relative_speed

theorem train_a_distance_traveled : (50 * time_to_meet) = 225 := by
  sorry

end train_a_distance_traveled_l168_168925


namespace solution_correct_l168_168634

-- Conditions of the problem
variable (f : ℝ → ℝ)
variable (h_f_domain : ∀ (x : ℝ), 0 < x → 0 < f x)
variable (h_f_eq : ∀ (x y : ℝ), 0 < x → 0 < y → f x * f (y * f x) = f (x + y))

-- Correct answer to be proven
theorem solution_correct :
  ∃ b : ℝ, 0 ≤ b ∧ ∀ t : ℝ, 0 < t → f t = 1 / (1 + b * t) :=
sorry

end solution_correct_l168_168634


namespace positive_solution_for_y_l168_168308

theorem positive_solution_for_y (x y z : ℝ) 
  (h1 : x * y = 4 - x - 2 * y)
  (h2 : y * z = 8 - 3 * y - 2 * z)
  (h3 : x * z = 40 - 5 * x - 2 * z) : y = 2 := 
sorry

end positive_solution_for_y_l168_168308


namespace circle_condition_l168_168739

theorem circle_condition (m : ℝ) : (∃ x y : ℝ, x^2 + y^2 + 4*x - 2*y + 5*m = 0) →
  (m < 1) :=
by
  sorry

end circle_condition_l168_168739


namespace largest_four_digit_number_divisible_by_six_l168_168305

theorem largest_four_digit_number_divisible_by_six : 
  ∃ n : ℕ, (1000 ≤ n ∧ n ≤ 9999) ∧ (n % 2 = 0) ∧ (n % 3 = 0) ∧ 
  (∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999) ∧ (m % 2 = 0) ∧ (m % 3 = 0) → m ≤ n) ∧ n = 9960 := 
by { sorry }

end largest_four_digit_number_divisible_by_six_l168_168305


namespace proof_expectation_red_balls_drawn_l168_168952

noncomputable def expectation_red_balls_drawn : Prop :=
  let total_ways := Nat.choose 5 2
  let ways_2_red := Nat.choose 3 2
  let ways_1_red_1_yellow := Nat.choose 3 1 * Nat.choose 2 1
  let p_X_eq_2 := (ways_2_red : ℝ) / total_ways
  let p_X_eq_1 := (ways_1_red_1_yellow : ℝ) / total_ways
  let expectation := 2 * p_X_eq_2 + 1 * p_X_eq_1
  expectation = 1.2

theorem proof_expectation_red_balls_drawn :
  expectation_red_balls_drawn :=
by
  sorry

end proof_expectation_red_balls_drawn_l168_168952


namespace probability_all_same_flips_l168_168951

noncomputable def four_same_flips_probability : ℚ := 
  (∑' n : ℕ, if n > 0 then (1/2)^(4*n) else 0)

theorem probability_all_same_flips : 
  four_same_flips_probability = 1 / 15 := 
sorry

end probability_all_same_flips_l168_168951


namespace unique_n_l168_168407

theorem unique_n (n : ℕ) (h_pos : 0 < n) :
  (∀ x y : ℕ, (xy + 1) % n = 0 → (x + y) % n = 0) ↔ n = 2 :=
by
  sorry

end unique_n_l168_168407


namespace calc_nabla_l168_168588

noncomputable def op_nabla (a b : ℚ) : ℚ := (a + b) / (1 + a * b)

theorem calc_nabla : (op_nabla (op_nabla 2 3) 4) = 11 / 9 :=
by
  unfold op_nabla
  sorry

end calc_nabla_l168_168588


namespace ralph_fewer_pictures_l168_168377

-- Define the number of wild animal pictures Ralph and Derrick have.
def ralph_pictures : ℕ := 26
def derrick_pictures : ℕ := 34

-- The main theorem stating that Ralph has 8 fewer pictures than Derrick.
theorem ralph_fewer_pictures : derrick_pictures - ralph_pictures = 8 := by
  -- The proof is omitted, denoted by 'sorry'.
  sorry

end ralph_fewer_pictures_l168_168377


namespace percent_of_part_is_20_l168_168040

theorem percent_of_part_is_20 {Part Whole : ℝ} (hPart : Part = 14) (hWhole : Whole = 70) : (Part / Whole) * 100 = 20 :=
by
  rw [hPart, hWhole]
  have h : (14 : ℝ) / 70 = 0.2 := by norm_num
  rw [h]
  norm_num

end percent_of_part_is_20_l168_168040


namespace find_EQ_length_l168_168734

theorem find_EQ_length (a b c d : ℕ) (parallel : Prop) (circle_tangent : Prop) :
  a = 105 ∧ b = 45 ∧ c = 21 ∧ d = 80 ∧ parallel ∧ circle_tangent → (∃ x : ℚ, x = 336 / 5) :=
by
  sorry

end find_EQ_length_l168_168734


namespace border_area_correct_l168_168292

-- Definition of the dimensions of the photograph
def photo_height := 8
def photo_width := 10
def frame_border := 3

-- Definition of the areas of the photograph and the framed area
def photo_area := photo_height * photo_width
def frame_height := photo_height + 2 * frame_border
def frame_width := photo_width + 2 * frame_border
def frame_area := frame_height * frame_width

-- Theorem stating that the area of the border is 144 square inches
theorem border_area_correct : (frame_area - photo_area) = 144 := 
by
  sorry

end border_area_correct_l168_168292


namespace remainder_m_n_mod_1000_l168_168079

noncomputable def m : ℕ :=
  Nat.card {p : ℕ × ℕ × ℕ // 4 * p.1 + 3 * p.2 + 2 * p.2.1 = 2009 ∧ 0 < p.1 ∧ 0 < p.2 ∧ 0 < p.2.1}

noncomputable def n : ℕ :=
  Nat.card {p : ℕ × ℕ × ℕ // 4 * p.1 + 3 * p.2 + 2 * p.2.1 = 2000 ∧ 0 < p.1 ∧ 0 < p.2 ∧ 0 < p.2.1}

theorem remainder_m_n_mod_1000 : (m - n) % 1000 = 0 :=
by
  sorry

end remainder_m_n_mod_1000_l168_168079


namespace last_operation_ends_at_eleven_am_l168_168674

-- Definitions based on conditions
def operation_duration : ℕ := 45 -- duration of each operation in minutes
def start_time : ℕ := 8 * 60 -- start time of the first operation in minutes since midnight
def interval : ℕ := 15 -- interval between operations in minutes
def total_operations : ℕ := 10 -- total number of operations

-- Compute the start time of the last operation (10th operation)
def start_time_last_operation : ℕ := start_time + interval * (total_operations - 1)

-- Compute the end time of the last operation
def end_time_last_operation : ℕ := start_time_last_operation + operation_duration

-- End time of the last operation expected to be 11:00 a.m. in minutes since midnight
def expected_end_time : ℕ := 11 * 60 

theorem last_operation_ends_at_eleven_am : 
  end_time_last_operation = expected_end_time := by
  sorry

end last_operation_ends_at_eleven_am_l168_168674


namespace max_value_of_f_l168_168971

noncomputable def f (x : ℝ) : ℝ := min (2^x) (min (x + 2) (10 - x))

theorem max_value_of_f : ∃ M, (∀ x ≥ 0, f x ≤ M) ∧ (∃ x ≥ 0, f x = M) ∧ M = 6 :=
by
  sorry

end max_value_of_f_l168_168971


namespace arithmetic_seq_sum_x_y_l168_168202

theorem arithmetic_seq_sum_x_y :
  ∃ (x y : ℕ), (∀ n : ℕ, n > 0 → a_n = 3 + (n - 1) * 5) ∧ x + 33 = 33 ∧ x = 28 → x + y = 61 :=
by
  sorry

end arithmetic_seq_sum_x_y_l168_168202


namespace paint_fraction_second_week_l168_168571

theorem paint_fraction_second_week
  (total_paint : ℕ)
  (first_week_fraction : ℚ)
  (total_used : ℕ)
  (paint_first_week : ℕ)
  (remaining_paint : ℕ)
  (paint_second_week : ℕ)
  (fraction_second_week : ℚ) :
  total_paint = 360 →
  first_week_fraction = 1/4 →
  total_used = 225 →
  paint_first_week = first_week_fraction * total_paint →
  remaining_paint = total_paint - paint_first_week →
  paint_second_week = total_used - paint_first_week →
  fraction_second_week = paint_second_week / remaining_paint →
  fraction_second_week = 1/2 :=
by
  sorry

end paint_fraction_second_week_l168_168571


namespace students_taking_either_but_not_both_l168_168174

-- Definitions to encapsulate the conditions
def students_taking_both : ℕ := 15
def students_taking_mathematics : ℕ := 30
def students_taking_history_only : ℕ := 12

-- The goal is to prove the number of students taking mathematics or history but not both
theorem students_taking_either_but_not_both
  (hb : students_taking_both = 15)
  (hm : students_taking_mathematics = 30)
  (hh : students_taking_history_only = 12) : 
  students_taking_mathematics - students_taking_both + students_taking_history_only = 27 :=
by
  sorry

end students_taking_either_but_not_both_l168_168174


namespace train_speed_l168_168440

theorem train_speed (length : ℕ) (cross_time : ℕ) (speed : ℝ)
    (h1 : length = 250)
    (h2 : cross_time = 3)
    (h3 : speed = (length / cross_time : ℝ) * 3.6) :
    speed = 300 := 
sorry

end train_speed_l168_168440


namespace max_value_of_f_f_is_increasing_on_intervals_l168_168923

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.cos x ^ 2 + Real.sqrt 3 * Real.sin (2 * x)

theorem max_value_of_f :
  ∃ (k : ℤ), ∀ (x : ℝ), x = k * Real.pi + Real.pi / 6 → f x = 3 :=
sorry

theorem f_is_increasing_on_intervals :
  ∀ (k : ℤ), ∀ (x y : ℝ), k * Real.pi - Real.pi / 3 ≤ x →
                x ≤ y → y ≤ k * Real.pi + Real.pi / 6 →
                f x ≤ f y :=
sorry

end max_value_of_f_f_is_increasing_on_intervals_l168_168923


namespace odd_and_monotonic_l168_168197

-- Definitions based on the conditions identified
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = - f x
def is_monotonic_increasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x ≤ f y
def f (x : ℝ) : ℝ := x ^ 3

-- Theorem statement without the proof
theorem odd_and_monotonic :
  is_odd f ∧ is_monotonic_increasing f :=
sorry

end odd_and_monotonic_l168_168197


namespace alice_has_ball_after_two_turns_l168_168106

noncomputable def prob_alice_has_ball_after_two_turns : ℚ :=
  let p_A_B := (3 : ℚ) / 5 -- Probability Alice tosses to Bob
  let p_B_A := (1 : ℚ) / 3 -- Probability Bob tosses to Alice
  let p_A_A := (2 : ℚ) / 5 -- Probability Alice keeps the ball
  (p_A_B * p_B_A) + (p_A_A * p_A_A)

theorem alice_has_ball_after_two_turns :
  prob_alice_has_ball_after_two_turns = 9 / 25 :=
by
  -- skipping the proof
  sorry

end alice_has_ball_after_two_turns_l168_168106


namespace unique_solution_7x_eq_3y_plus_4_l168_168185

theorem unique_solution_7x_eq_3y_plus_4 (x y : ℕ) (hx : 1 ≤ x) (hy : 1 ≤ y) :
    7^x = 3^y + 4 ↔ (x = 1 ∧ y = 1) :=
by
  sorry

end unique_solution_7x_eq_3y_plus_4_l168_168185


namespace gcd_of_8_and_12_l168_168294

theorem gcd_of_8_and_12 :
  let a := 8
  let b := 12
  let lcm_ab := 24
  Nat.lcm a b = lcm_ab → Nat.gcd a b = 4 :=
by
  intros
  sorry

end gcd_of_8_and_12_l168_168294


namespace cost_of_tree_planting_l168_168138

theorem cost_of_tree_planting 
  (initial_temp final_temp : ℝ) (temp_drop_per_tree cost_per_tree : ℝ) 
  (h_initial: initial_temp = 80) (h_final: final_temp = 78.2) 
  (h_temp_drop_per_tree: temp_drop_per_tree = 0.1) 
  (h_cost_per_tree: cost_per_tree = 6) : 
  (final_temp - initial_temp) / temp_drop_per_tree * cost_per_tree = 108 := 
by
  sorry

end cost_of_tree_planting_l168_168138


namespace cost_of_bananas_and_cantaloupe_l168_168903

-- Define variables representing the prices
variables (a b c d : ℝ)

-- Define the given conditions as hypotheses
def conditions : Prop :=
  a + b + c + d = 33 ∧
  d = 3 * a ∧
  c = a + 2 * b

-- State the main theorem
theorem cost_of_bananas_and_cantaloupe (h : conditions a b c d) : b + c = 13 :=
by {
  sorry
}

end cost_of_bananas_and_cantaloupe_l168_168903


namespace other_root_of_quadratic_l168_168436

theorem other_root_of_quadratic (m : ℝ) (x2 : ℝ) : (x^2 + m * x + 6 = 0) → (x + 2) * (x + x2) = 0 → x2 = -3 :=
by
  sorry

end other_root_of_quadratic_l168_168436


namespace max_value_x_minus_y_l168_168469

theorem max_value_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end max_value_x_minus_y_l168_168469


namespace find_f_at_2_l168_168188

noncomputable def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 1

theorem find_f_at_2 (a b : ℝ) 
  (h1 : 3 + 2 * a + b = 0) 
  (h2 : 1 + a + b + 1 = -2) : 
  f a b 2 = 3 := 
by
  dsimp [f]
  sorry

end find_f_at_2_l168_168188


namespace perspective_square_area_l168_168957

theorem perspective_square_area (a b : ℝ) (ha : a = 4 ∨ b = 4) : 
  a * a = 16 ∨ (2 * b) * (2 * b) = 64 :=
by 
sorry

end perspective_square_area_l168_168957


namespace quadratic_point_comparison_l168_168231

theorem quadratic_point_comparison (c y1 y2 y3 : ℝ) 
  (h1 : y1 = -(-2:ℝ)^2 + c)
  (h2 : y2 = -(1:ℝ)^2 + c)
  (h3 : y3 = -(3:ℝ)^2 + c) : y2 > y1 ∧ y1 > y3 := 
by
  sorry

end quadratic_point_comparison_l168_168231


namespace trajectory_of_P_distance_EF_l168_168045

section Exercise

-- Define the curve C in polar coordinates
def curve_C (ρ' θ: ℝ) : Prop :=
  ρ' * Real.cos (θ + Real.pi / 4) = 1

-- Define the relationship between OP and OQ
def product_OP_OQ (ρ ρ' : ℝ) : Prop :=
  ρ * ρ' = Real.sqrt 2

-- Define the trajectory of point P (C1) as the goal
theorem trajectory_of_P (ρ θ: ℝ) (hC: curve_C ρ' θ) (hPQ: product_OP_OQ ρ ρ') :
  ρ = Real.cos θ - Real.sin θ :=
sorry

-- Define the coordinates and the curve C2
def curve_C2 (x y t: ℝ) : Prop :=
  x = 0.5 - Real.sqrt 2 / 2 * t ∧ y = Real.sqrt 2 / 2 * t

-- Define the line l in Cartesian coordinates that needs to be converted to polar
def line_l (x y: ℝ) : Prop :=
  y = -Real.sqrt 3 * x

-- Define the distance |EF| to be proved
theorem distance_EF (θ ρ_1 ρ_2: ℝ) (hx: curve_C2 (0.5 - Real.sqrt 2 / 2 * t) (Real.sqrt 2 / 2 * t) t)
  (hE: θ = 2 * Real.pi / 3 ∨ θ = -Real.pi / 3)
  (hρ1: ρ_1 = Real.cos (-Real.pi / 3) - Real.sin (-Real.pi / 3))
  (hρ2: ρ_2 = 0.5 * (Real.sqrt 3 + 1)) :
  |ρ_1 + ρ_2| = Real.sqrt 3 + 1 :=
sorry

end Exercise

end trajectory_of_P_distance_EF_l168_168045


namespace arianna_sleep_hours_l168_168913

-- Defining the given conditions
def total_hours_in_a_day : ℕ := 24
def hours_at_work : ℕ := 6
def hours_in_class : ℕ := 3
def hours_at_gym : ℕ := 2
def hours_on_chores : ℕ := 5

-- Formulating the total hours spent on activities
def total_hours_on_activities := hours_at_work + hours_in_class + hours_at_gym + hours_on_chores

-- Proving Arianna's sleep hours
theorem arianna_sleep_hours : total_hours_in_a_day - total_hours_on_activities = 8 :=
by
  -- Direct proof placeholder, to be filled in with actual proof steps or tactic
  sorry

end arianna_sleep_hours_l168_168913


namespace charley_initial_pencils_l168_168542

theorem charley_initial_pencils (P : ℕ) (lost_initially : P - 6 = (P - 1/3 * (P - 6) - 6)) (current_pencils : P - 1/3 * (P - 6) - 6 = 16) : P = 30 := 
sorry

end charley_initial_pencils_l168_168542


namespace total_net_loss_l168_168779

theorem total_net_loss 
  (P_x P_y : ℝ)
  (h1 : 1.2 * P_x = 25000)
  (h2 : 0.8 * P_y = 25000) :
  (25000 - P_x) - (P_y - 25000) = -2083.33 :=
by 
  sorry

end total_net_loss_l168_168779


namespace lara_yesterday_more_than_sarah_l168_168418

variable (yesterdaySarah todaySarah todayLara : ℕ)
variable (cansDifference : ℕ)

axiom yesterdaySarah_eq : yesterdaySarah = 50
axiom todaySarah_eq : todaySarah = 40
axiom todayLara_eq : todayLara = 70
axiom cansDifference_eq : cansDifference = 20

theorem lara_yesterday_more_than_sarah :
  let totalCansYesterday := yesterdaySarah + todaySarah + cansDifference
  let laraYesterday := totalCansYesterday - yesterdaySarah
  laraYesterday - yesterdaySarah = 30 :=
by
  sorry

end lara_yesterday_more_than_sarah_l168_168418


namespace find_t_when_perpendicular_l168_168861

variable {t : ℝ}

def vector_m (t : ℝ) : ℝ × ℝ := (t + 1, 1)
def vector_n (t : ℝ) : ℝ × ℝ := (t + 2, 2)
def add_vectors (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)
def sub_vectors (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def dot_product (a b : ℝ × ℝ) : ℝ := (a.1 * b.1) + (a.2 * b.2)

theorem find_t_when_perpendicular : 
  (dot_product (add_vectors (vector_m t) (vector_n t)) (sub_vectors (vector_m t) (vector_n t)) = 0) ↔ t = -3 := by
  sorry

end find_t_when_perpendicular_l168_168861


namespace cost_of_tissues_l168_168552
-- Import the entire Mathlib library

-- Define the context and the assertion without computing the proof details
theorem cost_of_tissues
  (n_tp : ℕ) -- Number of toilet paper rolls
  (c_tp : ℝ) -- Cost per toilet paper roll
  (n_pt : ℕ) -- Number of paper towels rolls
  (c_pt : ℝ) -- Cost per paper towel roll
  (n_t : ℕ) -- Number of tissue boxes
  (T : ℝ) -- Total cost of all items
  (H_tp : n_tp = 10) -- Given: 10 rolls of toilet paper
  (H_c_tp : c_tp = 1.5) -- Given: $1.50 per roll of toilet paper
  (H_pt : n_pt = 7) -- Given: 7 rolls of paper towels
  (H_c_pt : c_pt = 2) -- Given: $2 per roll of paper towel
  (H_t : n_t = 3) -- Given: 3 boxes of tissues
  (H_T : T = 35) -- Given: total cost is $35
  : (T - (n_tp * c_tp + n_pt * c_pt)) / n_t = 2 := -- Conclusion: the cost of one box of tissues is $2
by {
  sorry -- Proof details to be supplied here
}

end cost_of_tissues_l168_168552


namespace find_multiple_l168_168517

-- Definitions based on the problem's conditions
def n_drunk_drivers : ℕ := 6
def total_students : ℕ := 45
def num_speeders (M : ℕ) : ℕ := M * n_drunk_drivers - 3

-- The theorem that we need to prove
theorem find_multiple (M : ℕ) (h1: total_students = n_drunk_drivers + num_speeders M) : M = 7 :=
by
  sorry

end find_multiple_l168_168517


namespace expected_value_of_white_balls_l168_168526

-- Definitions for problem conditions
def totalBalls : ℕ := 6
def whiteBalls : ℕ := 2
def redBalls : ℕ := 4
def ballsDrawn : ℕ := 2

-- Probability calculations
def P_X_0 : ℚ := (Nat.choose 4 2) / (Nat.choose totalBalls ballsDrawn)
def P_X_1 : ℚ := ((Nat.choose whiteBalls 1) * (Nat.choose redBalls 1)) / (Nat.choose totalBalls ballsDrawn)
def P_X_2 : ℚ := (Nat.choose whiteBalls 2) / (Nat.choose totalBalls ballsDrawn)

-- Expected value calculation
def expectedValue : ℚ := (0 * P_X_0) + (1 * P_X_1) + (2 * P_X_2)

theorem expected_value_of_white_balls :
  expectedValue = 2 / 3 :=
by
  sorry

end expected_value_of_white_balls_l168_168526


namespace peter_fish_caught_l168_168721

theorem peter_fish_caught (n : ℕ) (h : 3 * n = n + 24) : n = 12 :=
sorry

end peter_fish_caught_l168_168721


namespace final_withdrawal_amount_july_2005_l168_168408

-- Define the conditions given in the problem
variables (a r : ℝ) (n : ℕ)

-- Define the recursive formula for deposits
def deposit_amount (n : ℕ) : ℝ :=
  if n = 0 then a else (deposit_amount (n - 1)) * (1 + r) + a

-- The problem statement translated to Lean
theorem final_withdrawal_amount_july_2005 :
  deposit_amount a r 5 = a / r * ((1 + r) ^ 6 - (1 + r)) :=
sorry

end final_withdrawal_amount_july_2005_l168_168408


namespace problem_1_problem_2_l168_168252

def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

theorem problem_1 : {x : ℝ | f x ≥ 1} = {x : ℝ | x ≥ 1} :=
  sorry

theorem problem_2 (m : ℝ) : (∃ x : ℝ, f x ≥ x^2 - x + m) → m ≤ 5/4 :=
  sorry

end problem_1_problem_2_l168_168252


namespace solve_x_l168_168771

-- Define the structure of the pyramid
def pyramid (x : ℕ) : Prop :=
  let level1 := [x + 4, 12, 15, 18]
  let level2 := [x + 16, 27, 33]
  let level3 := [x + 43, 60]
  let top := x + 103
  top = 120

theorem solve_x : ∃ x : ℕ, pyramid x → x = 17 :=
by
  -- Proof omitted
  sorry

end solve_x_l168_168771


namespace total_students_class_is_63_l168_168643

def num_tables : ℕ := 6
def students_per_table : ℕ := 3
def girls_bathroom : ℕ := 4
def times_canteen : ℕ := 4
def group1_students : ℕ := 4
def group2_students : ℕ := 5
def group3_students : ℕ := 6
def germany_students : ℕ := 2
def france_students : ℕ := 4
def norway_students : ℕ := 3
def italy_students : ℕ := 1

def total_students_in_class : ℕ :=
  (num_tables * students_per_table) +
  girls_bathroom +
  (times_canteen * girls_bathroom) +
  (group1_students + group2_students + group3_students) +
  (germany_students + france_students + norway_students + italy_students)

theorem total_students_class_is_63 : total_students_in_class = 63 :=
  by
    sorry

end total_students_class_is_63_l168_168643


namespace green_light_probability_l168_168304

def red_duration : ℕ := 30
def green_duration : ℕ := 25
def yellow_duration : ℕ := 5

def total_cycle : ℕ := red_duration + green_duration + yellow_duration
def green_probability : ℚ := green_duration / total_cycle

theorem green_light_probability :
  green_probability = 5 / 12 := by
  sorry

end green_light_probability_l168_168304


namespace simplify_expression_l168_168549

variable (x : ℝ)

theorem simplify_expression : 
  2 * x^3 - (7 * x^2 - 9 * x) - 2 * (x^3 - 3 * x^2 + 4 * x) = -x^2 + x := 
by
  sorry

end simplify_expression_l168_168549


namespace max_f_value_l168_168108

noncomputable def f (x : ℝ) : ℝ := min (3 * x + 1) (min (- (4 / 3) * x + 3) ((1 / 3) * x + 9))

theorem max_f_value : ∃ x : ℝ, f x = 31 / 13 :=
by 
  sorry

end max_f_value_l168_168108


namespace baseball_cards_given_l168_168719

theorem baseball_cards_given
  (initial_cards : ℕ)
  (maria_take : ℕ)
  (peter_cards : ℕ)
  (paul_triples : ℕ)
  (final_cards : ℕ)
  (h1 : initial_cards = 15)
  (h2 : maria_take = (initial_cards + 1) / 2)
  (h3 : final_cards = 3 * (initial_cards - maria_take - peter_cards))
  (h4 : final_cards = 18) :
  peter_cards = 1 := 
sorry

end baseball_cards_given_l168_168719


namespace boat_stream_speed_l168_168618

/-- A boat can travel with a speed of 22 km/hr in still water. 
If the speed of the stream is unknown, the boat takes 7 hours 
to go 189 km downstream. What is the speed of the stream?
-/
theorem boat_stream_speed (v : ℝ) : (22 + v) * 7 = 189 → v = 5 :=
by
  intro h
  sorry

end boat_stream_speed_l168_168618


namespace Marie_speed_l168_168659

theorem Marie_speed (distance time : ℕ) (h1 : distance = 372) (h2 : time = 31) : distance / time = 12 :=
by
  have h3 : distance = 372 := h1
  have h4 : time = 31 := h2
  sorry

end Marie_speed_l168_168659


namespace carol_name_tag_l168_168772

theorem carol_name_tag (a b c : ℕ) (ha : Prime a ∧ a ≥ 10 ∧ a < 100) (hb : Prime b ∧ b ≥ 10 ∧ b < 100) (hc : Prime c ∧ c ≥ 10 ∧ c < 100) 
  (h1 : b + c = 14) (h2 : a + c = 20) (h3 : a + b = 18) : c = 11 := 
by 
  sorry

end carol_name_tag_l168_168772


namespace part1_solution_set_part2_min_value_l168_168975

-- Part 1
noncomputable def f (x : ℝ) : ℝ := 2 * |x + 1| + |3 * x|

theorem part1_solution_set :
  {x : ℝ | f x ≥ 3 * |x| + 1} = {x : ℝ | x ≥ -1/2} ∪ {x : ℝ | x ≤ -3/2} :=
by
  sorry

-- Part 2
noncomputable def f_min (x a b : ℝ) : ℝ := 2 * |x + a| + |3 * x - b|

theorem part2_min_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : ∃ x, f_min x a b = 2) :
  3 * a + b = 3 :=
by
  sorry

end part1_solution_set_part2_min_value_l168_168975


namespace compare_M_N_l168_168397

variable (a : ℝ)

def M : ℝ := 2 * a * (a - 2) + 7
def N : ℝ := (a - 2) * (a - 3)

theorem compare_M_N : M a > N a :=
by
  sorry

end compare_M_N_l168_168397


namespace exists_int_x_l168_168139

theorem exists_int_x (K M N : ℤ) (h1 : K ≠ 0) (h2 : M ≠ 0) (h3 : N ≠ 0) (h_coprime : Int.gcd K M = 1) :
  ∃ x : ℤ, K ∣ (M * x + N) :=
by
  sorry

end exists_int_x_l168_168139


namespace three_digit_numbers_not_multiple_of_3_5_7_l168_168625

theorem three_digit_numbers_not_multiple_of_3_5_7 : 
  let total_three_digit_numbers := 900
  let multiples_of_3 := (999 - 100) / 3 + 1
  let multiples_of_5 := (995 - 100) / 5 + 1
  let multiples_of_7 := (994 - 105) / 7 + 1
  let multiples_of_15 := (990 - 105) / 15 + 1
  let multiples_of_21 := (987 - 105) / 21 + 1
  let multiples_of_35 := (980 - 105) / 35 + 1
  let multiples_of_105 := (945 - 105) / 105 + 1
  let total_multiples := multiples_of_3 + multiples_of_5 + multiples_of_7 - multiples_of_15 - multiples_of_21 - multiples_of_35 + multiples_of_105
  let non_multiples_total := total_three_digit_numbers - total_multiples
  non_multiples_total = 412 :=
by
  sorry

end three_digit_numbers_not_multiple_of_3_5_7_l168_168625


namespace find_m_if_f_even_l168_168107

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + (m - 2) * x + (m^2 - 7 * m + 12)

theorem find_m_if_f_even :
  (∀ x : ℝ, f m (-x) = f m x) → m = 2 :=
by 
  intro h
  sorry

end find_m_if_f_even_l168_168107


namespace mod11_residue_l168_168114

theorem mod11_residue :
  (305 % 11 = 8) →
  (44 % 11 = 0) →
  (176 % 11 = 0) →
  (18 % 11 = 7) →
  (305 + 7 * 44 + 9 * 176 + 6 * 18) % 11 = 6 :=
by
  intros h1 h2 h3 h4
  sorry

end mod11_residue_l168_168114


namespace Kira_was_away_for_8_hours_l168_168863

theorem Kira_was_away_for_8_hours
  (kibble_rate: ℕ)
  (initial_kibble: ℕ)
  (remaining_kibble: ℕ)
  (hours_per_pound: ℕ) 
  (kibble_eaten: ℕ)
  (kira_was_away: ℕ)
  (h1: kibble_rate = 1)
  (h2: initial_kibble = 3)
  (h3: remaining_kibble = 1)
  (h4: hours_per_pound = 4)
  (h5: kibble_eaten = initial_kibble - remaining_kibble)
  (h6: kira_was_away = hours_per_pound * kibble_eaten) : 
  kira_was_away = 8 :=
by
  sorry

end Kira_was_away_for_8_hours_l168_168863


namespace monthly_rent_l168_168470

theorem monthly_rent (cost : ℕ) (maintenance_percentage : ℚ) (annual_taxes : ℕ) (desired_return_rate : ℚ) (monthly_rent : ℚ) :
  cost = 20000 ∧
  maintenance_percentage = 0.10 ∧
  annual_taxes = 460 ∧
  desired_return_rate = 0.06 →
  monthly_rent = 153.70 := 
sorry

end monthly_rent_l168_168470


namespace power_mod_7_l168_168946

theorem power_mod_7 {a : ℤ} (h : a = 3) : (a ^ 123) % 7 = 6 := by
  sorry

end power_mod_7_l168_168946


namespace min_quadratic_expr_l168_168496

noncomputable def quadratic_expr (x : ℝ) := x^2 + 10 * x + 3

theorem min_quadratic_expr : ∃ x : ℝ, quadratic_expr x = -22 :=
by
  use -5
  simp [quadratic_expr]
  sorry

end min_quadratic_expr_l168_168496


namespace distinct_after_removal_l168_168035

variable (n : ℕ)
variable (subsets : Fin n → Finset (Fin n))

theorem distinct_after_removal :
  ∃ k : Fin n, ∀ i j : Fin n, i ≠ j → (subsets i \ {k}) ≠ (subsets j \ {k}) := by
  sorry

end distinct_after_removal_l168_168035


namespace whole_numbers_between_sqrt_18_and_sqrt_98_l168_168008

theorem whole_numbers_between_sqrt_18_and_sqrt_98 :
  let lower_bound := Real.sqrt 18
  let upper_bound := Real.sqrt 98
  let smallest_whole_num := 5
  let largest_whole_num := 9
  (largest_whole_num - smallest_whole_num + 1) = 5 :=
by
  -- Introduce variables
  let lower_bound := Real.sqrt 18
  let upper_bound := Real.sqrt 98
  let smallest_whole_num := 5
  let largest_whole_num := 9
  -- Sorry indicates the proof steps are skipped
  sorry

end whole_numbers_between_sqrt_18_and_sqrt_98_l168_168008


namespace weight_of_3_moles_HClO2_correct_l168_168001

def atomic_weight_H : ℝ := 1.008
def atomic_weight_Cl : ℝ := 35.453
def atomic_weight_O : ℝ := 15.999

def molecular_weight_HClO2 : ℝ := (1 * atomic_weight_H) + (1 * atomic_weight_Cl) + (2 * atomic_weight_O)
def weight_of_3_moles_HClO2 : ℝ := 3 * molecular_weight_HClO2

theorem weight_of_3_moles_HClO2_correct : weight_of_3_moles_HClO2 = 205.377 := by
  sorry

end weight_of_3_moles_HClO2_correct_l168_168001


namespace integral_result_l168_168587

theorem integral_result (b : ℝ) (h : ∫ x in e..b, (2 / x) = 6) : b = Real.exp 4 :=
sorry

end integral_result_l168_168587


namespace part1_part2_l168_168392

noncomputable def f (a x : ℝ) : ℝ := a * x^2 + (2 - a) * x + a

-- Part 1
theorem part1 (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 1) ↔ a ≥ 2 * Real.sqrt 3 / 3 := sorry

-- Part 2
theorem part2 (a x : ℝ) : 
  (f a x < a + 2) ↔ 
    (a = 0 ∧ x < 1) ∨ 
    (a > 0 ∧ -2 / a < x ∧ x < 1) ∨ 
    (-2 < a ∧ a < 0 ∧ (x < 1 ∨ x > -2 / a)) ∨ 
    (a = -2) ∨ 
    (a < -2 ∧ (x < -2 / a ∨ x > 1)) := sorry

end part1_part2_l168_168392


namespace tetrahedron_inequality_l168_168702

theorem tetrahedron_inequality
  (a b c d h_a h_b h_c h_d V : ℝ)
  (ha : V = 1/3 * a * h_a)
  (hb : V = 1/3 * b * h_b)
  (hc : V = 1/3 * c * h_c)
  (hd : V = 1/3 * d * h_d) :
  (a + b + c + d) * (h_a + h_b + h_c + h_d) >= 48 * V := 
  by sorry

end tetrahedron_inequality_l168_168702


namespace ship_speed_in_still_water_l168_168953

theorem ship_speed_in_still_water
  (x y : ℝ)
  (h1: x + y = 32)
  (h2: x - y = 28)
  (h3: x > y) : 
  x = 30 := 
sorry

end ship_speed_in_still_water_l168_168953


namespace total_spent_l168_168480

-- Define the number of books and magazines Lynne bought
def num_books_cats : ℕ := 7
def num_books_solar_system : ℕ := 2
def num_magazines : ℕ := 3

-- Define the costs
def cost_per_book : ℕ := 7
def cost_per_magazine : ℕ := 4

-- Calculate the total cost and assert that it equals to $75
theorem total_spent :
  (num_books_cats * cost_per_book) + 
  (num_books_solar_system * cost_per_book) + 
  (num_magazines * cost_per_magazine) = 75 := 
sorry

end total_spent_l168_168480


namespace option_d_correct_l168_168225

theorem option_d_correct (x y : ℝ) : -4 * x * y + 3 * x * y = -1 * x * y := 
by {
  sorry
}

end option_d_correct_l168_168225


namespace only_solutions_mod_n_l168_168488

theorem only_solutions_mod_n (n : ℕ) : (∀ k : ℤ, ∃ a : ℤ, (a^3 + a - k) % (n : ℤ) = 0) ↔ (∃ k : ℕ, n = 3 ^ k) := 
sorry

end only_solutions_mod_n_l168_168488


namespace total_salaries_proof_l168_168657

def total_salaries (A_salary B_salary : ℝ) :=
  A_salary + B_salary

theorem total_salaries_proof : ∀ A_salary B_salary : ℝ,
  A_salary = 3000 →
  (0.05 * A_salary = 0.15 * B_salary) →
  total_salaries A_salary B_salary = 4000 :=
by
  intros A_salary B_salary h1 h2
  rw [h1] at h2
  sorry

end total_salaries_proof_l168_168657


namespace inscribed_sphere_volume_l168_168503

theorem inscribed_sphere_volume (edge_length : ℝ) (h_edge : edge_length = 12) : 
  ∃ (V : ℝ), V = 288 * Real.pi :=
by
  sorry

end inscribed_sphere_volume_l168_168503


namespace value_of_m2_plus_3n2_l168_168570

noncomputable def real_numbers_with_condition (m n : ℝ) : Prop :=
  (m^2 + 3*n^2)^2 - 4*(m^2 + 3*n^2) - 12 = 0

theorem value_of_m2_plus_3n2 (m n : ℝ) (h : real_numbers_with_condition m n) : m^2 + 3*n^2 = 6 :=
by
  sorry

end value_of_m2_plus_3n2_l168_168570


namespace constant_function_of_zero_derivative_l168_168199

theorem constant_function_of_zero_derivative
  {f : ℝ → ℝ}
  (h : ∀ x : ℝ, deriv f x = 0) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c :=
sorry

end constant_function_of_zero_derivative_l168_168199


namespace opinion_change_difference_l168_168580

variables (initial_enjoy final_enjoy initial_not_enjoy final_not_enjoy : ℕ)
variables (n : ℕ) -- number of students in the class

-- Given conditions
def initial_conditions :=
  initial_enjoy = 40 * n / 100 ∧ initial_not_enjoy = 60 * n / 100

def final_conditions :=
  final_enjoy = 80 * n / 100 ∧ final_not_enjoy = 20 * n / 100

-- The theorem to prove
theorem opinion_change_difference :
  initial_conditions n initial_enjoy initial_not_enjoy →
  final_conditions n final_enjoy final_not_enjoy →
  (40 ≤ initial_enjoy + 20 ∧ 40 ≤ initial_not_enjoy + 20 ∧
  max_change = 60 ∧ min_change = 40 → max_change - min_change = 20) := 
  sorry

end opinion_change_difference_l168_168580


namespace tens_digit_6_pow_18_l168_168790

/--
To find the tens digit of \(6^{18}\), we look at the powers of 6 and determine their tens digits. 
We note the pattern in tens digits (3, 1, 9, 7, 6) which repeats every 5 powers. 
Since \(6^{18}\) corresponds to the 3rd position in the repeating cycle, we claim the tens digit is 1.
--/
theorem tens_digit_6_pow_18 : (6^18 / 10) % 10 = 1 :=
by sorry

end tens_digit_6_pow_18_l168_168790


namespace chastity_lollipops_l168_168669

theorem chastity_lollipops (initial_money lollipop_cost gummy_cost left_money total_gummies total_spent lollipops : ℝ)
  (h1 : initial_money = 15)
  (h2 : lollipop_cost = 1.50)
  (h3 : gummy_cost = 2)
  (h4 : left_money = 5)
  (h5 : total_gummies = 2)
  (h6 : total_spent = initial_money - left_money)
  (h7 : total_spent = 10)
  (h8 : total_gummies * gummy_cost = 4)
  (h9 : total_spent - (total_gummies * gummy_cost) = 6)
  (h10 : lollipops = (total_spent - (total_gummies * gummy_cost)) / lollipop_cost) :
  lollipops = 4 := 
sorry

end chastity_lollipops_l168_168669


namespace koala_fiber_absorption_l168_168070

theorem koala_fiber_absorption (x : ℝ) (hx : 0.30 * x = 12) : x = 40 :=
by
  sorry

end koala_fiber_absorption_l168_168070


namespace johns_percentage_increase_l168_168832

def original_amount : ℕ := 60
def new_amount : ℕ := 84

def percentage_increase (original new : ℕ) := ((new - original : ℕ) * 100) / original 

theorem johns_percentage_increase : percentage_increase original_amount new_amount = 40 :=
by
  sorry

end johns_percentage_increase_l168_168832


namespace find_h_parallel_line_l168_168142

theorem find_h_parallel_line:
  ∃ h : ℚ, (3 * (h : ℚ) - 2 * (24 : ℚ) = 7) → (h = 47 / 3) :=
by
  sorry

end find_h_parallel_line_l168_168142


namespace first_student_time_l168_168324

-- Define the conditions
def num_students := 4
def avg_last_three := 35
def avg_all := 30
def total_time_all := num_students * avg_all
def total_time_last_three := (num_students - 1) * avg_last_three

-- State the theorem
theorem first_student_time : (total_time_all - total_time_last_three) = 15 :=
by
  -- Proof is skipped
  sorry

end first_student_time_l168_168324


namespace minimum_value_am_bn_l168_168539

-- Definitions and conditions
variables {a b m n : ℝ}
variables (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < m) (h₃ : 0 < n)
variables (h₄ : a + b = 1) (h₅ : m * n = 2)

-- Statement of the proof problem
theorem minimum_value_am_bn :
  ∃ c, (∀ a b m n : ℝ, 0 < a → 0 < b → 0 < m → 0 < n → a + b = 1 → m * n = 2 → (am * bn) * (bm * an) ≥ c) ∧ c = 2 :=
sorry

end minimum_value_am_bn_l168_168539


namespace sandy_initial_amount_l168_168384

theorem sandy_initial_amount 
  (cost_shirt : ℝ) (cost_jacket : ℝ) (found_money : ℝ)
  (h1 : cost_shirt = 12.14) (h2 : cost_jacket = 9.28) (h3 : found_money = 7.43) : 
  (cost_shirt + cost_jacket + found_money = 28.85) :=
by
  rw [h1, h2, h3]
  norm_num

end sandy_initial_amount_l168_168384


namespace employee_saves_86_25_l168_168806

def initial_purchase_price : ℝ := 500
def markup_rate : ℝ := 0.15
def employee_discount_rate : ℝ := 0.15

def retail_price : ℝ := initial_purchase_price * (1 + markup_rate)
def employee_discount_amount : ℝ := retail_price * employee_discount_rate
def employee_savings : ℝ := retail_price - (retail_price - employee_discount_amount)

theorem employee_saves_86_25 :
  employee_savings = 86.25 := 
sorry

end employee_saves_86_25_l168_168806


namespace simplify_expression_l168_168770

theorem simplify_expression (a : ℤ) (ha : a = -2) : 
  3 * a^2 + (a^2 + (5 * a^2 - 2 * a) - 3 * (a^2 - 3 * a)) = 10 := 
by 
  sorry

end simplify_expression_l168_168770


namespace calculate_m_l168_168081

theorem calculate_m (m : ℕ) : 9^4 = 3^m → m = 8 :=
by
  sorry

end calculate_m_l168_168081


namespace op_op_k_l168_168677

def op (x y : ℝ) : ℝ := x^3 + x - y

theorem op_op_k (k : ℝ) : op k (op k k) = k := sorry

end op_op_k_l168_168677


namespace union_of_sets_l168_168958

-- Define the sets A and B
def A : Set ℝ := { x | x^2 - x - 2 < 0 }
def B : Set ℝ := { x | 1 < x ∧ x < 4 }

-- Define the set representing the union's result
def C : Set ℝ := { x | -1 < x ∧ x < 4 }

-- The theorem statement
theorem union_of_sets : ∀ x : ℝ, (x ∈ (A ∪ B) ↔ x ∈ C) :=
by
  sorry

end union_of_sets_l168_168958


namespace pascal_triangle_eighth_row_l168_168945

def sum_interior_numbers (n : ℕ) : ℕ :=
  2^(n-1) - 2

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose (n-1) (k-1) 

theorem pascal_triangle_eighth_row:
  sum_interior_numbers 8 = 126 ∧ binomial_coefficient 8 3 = 21 :=
by
  sorry

end pascal_triangle_eighth_row_l168_168945


namespace rubles_greater_than_seven_l168_168624

theorem rubles_greater_than_seven (x : ℕ) (h : x > 7) : ∃ a b : ℕ, x = 3 * a + 5 * b :=
sorry

end rubles_greater_than_seven_l168_168624


namespace hours_worked_on_saturday_l168_168877

-- Definitions from the problem conditions
def hourly_wage : ℝ := 15
def hours_friday : ℝ := 10
def hours_sunday : ℝ := 14
def total_earnings : ℝ := 450

-- Define number of hours worked on Saturday as a variable
variable (hours_saturday : ℝ)

-- Total earnings can be expressed as the sum of individual day earnings
def total_earnings_eq : Prop := 
  total_earnings = (hours_friday * hourly_wage) + (hours_sunday * hourly_wage) + (hours_saturday * hourly_wage)

-- Prove that the hours worked on Saturday is 6
theorem hours_worked_on_saturday :
  total_earnings_eq hours_saturday →
  hours_saturday = 6 := by
  sorry

end hours_worked_on_saturday_l168_168877


namespace simplify_expression_l168_168263

theorem simplify_expression : 
  (6^8 - 4^7) * (2^3 - (-2)^3) ^ 10 = 1663232 * 16 ^ 10 := 
by {
  sorry
}

end simplify_expression_l168_168263


namespace value_of_x_l168_168439

theorem value_of_x : ∀ x : ℝ, (x^2 - 4) / (x - 2) = 0 → x ≠ 2 → x = -2 := by
  intros x h1 h2
  sorry

end value_of_x_l168_168439


namespace normalize_equation1_normalize_equation2_l168_168362

-- Define the first equation
def equation1 (x y : ℝ) := 2 * x - 3 * y - 10 = 0

-- Define the normalized form of the first equation
def normalized_equation1 (x y : ℝ) := (2 / Real.sqrt 13) * x - (3 / Real.sqrt 13) * y - (10 / Real.sqrt 13) = 0

-- Prove that the normalized form of the first equation is correct
theorem normalize_equation1 (x y : ℝ) (h : equation1 x y) : normalized_equation1 x y := 
sorry

-- Define the second equation
def equation2 (x y : ℝ) := 3 * x + 4 * y = 0

-- Define the normalized form of the second equation
def normalized_equation2 (x y : ℝ) := (3 / 5) * x + (4 / 5) * y = 0

-- Prove that the normalized form of the second equation is correct
theorem normalize_equation2 (x y : ℝ) (h : equation2 x y) : normalized_equation2 x y := 
sorry

end normalize_equation1_normalize_equation2_l168_168362


namespace toys_total_is_240_l168_168247

def number_of_toys_elder : Nat := 60
def number_of_toys_younger (toys_elder : Nat) : Nat := 3 * toys_elder
def total_number_of_toys (toys_elder toys_younger : Nat) : Nat := toys_elder + toys_younger

theorem toys_total_is_240 : total_number_of_toys number_of_toys_elder (number_of_toys_younger number_of_toys_elder) = 240 :=
by
  sorry

end toys_total_is_240_l168_168247


namespace arcsin_cos_eq_neg_pi_div_six_l168_168678

theorem arcsin_cos_eq_neg_pi_div_six :
  Real.arcsin (Real.cos (2 * Real.pi / 3)) = -Real.pi / 6 :=
by
  sorry

end arcsin_cos_eq_neg_pi_div_six_l168_168678


namespace calculation_result_l168_168104

def initial_number : ℕ := 15
def subtracted_value : ℕ := 2
def added_value : ℕ := 4
def divisor : ℕ := 1
def second_divisor : ℕ := 2
def multiplier : ℕ := 8

theorem calculation_result : 
  (initial_number - subtracted_value + (added_value / divisor : ℕ)) / second_divisor * multiplier = 68 :=
by
  sorry

end calculation_result_l168_168104


namespace number_of_eggs_l168_168133

-- Define the conditions as assumptions
variables (marbles : ℕ) (eggs : ℕ)
variables (eggs_A eggs_B eggs_C : ℕ)
variables (marbles_A marbles_B marbles_C : ℕ)

-- Conditions from the problem
axiom eggs_total : marbles = 4
axiom marbles_total : eggs = 15
axiom eggs_groups : eggs_A ≠ eggs_B ∧ eggs_B ≠ eggs_C ∧ eggs_A ≠ eggs_C
axiom marbles_diff1 : marbles_B - marbles_A = eggs_B
axiom marbles_diff2 : marbles_C - marbles_B = eggs_C

-- Prove that the number of eggs in each group is as specified in the answer
theorem number_of_eggs :
  eggs_A = 12 ∧ eggs_B = 1 ∧ eggs_C = 2 :=
by {
  sorry
}

end number_of_eggs_l168_168133


namespace rowing_trip_time_l168_168338

theorem rowing_trip_time
  (v_0 : ℝ) -- Rowing speed in still water
  (v_c : ℝ) -- Velocity of current
  (d : ℝ) -- Distance to the place
  (h_v0 : v_0 = 10) -- Given condition that rowing speed is 10 kmph
  (h_vc : v_c = 2) -- Given condition that current speed is 2 kmph
  (h_d : d = 144) -- Given condition that distance is 144 km :
  : (d / (v_0 - v_c) + d / (v_0 + v_c)) = 30 := -- Proving the total round trip time is 30 hours
by
  sorry

end rowing_trip_time_l168_168338


namespace range_of_m_l168_168612

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (x < 3) ↔ (x / 3 < 1 - (x - 3) / 6 ∧ x < m)) → m ≥ 3 :=
by
  sorry

end range_of_m_l168_168612


namespace interval_length_l168_168265

theorem interval_length (a b m h : ℝ) (h_eq : h = m / |a - b|) : |a - b| = m / h := 
by 
  sorry

end interval_length_l168_168265


namespace first_player_wins_l168_168240

def winning_strategy (m n : ℕ) : Prop :=
  if m = 1 ∧ n = 1 then false else true

theorem first_player_wins (m n : ℕ) :
  winning_strategy m n :=
by
  sorry

end first_player_wins_l168_168240


namespace king_arthur_actual_weight_l168_168991

theorem king_arthur_actual_weight (K H E : ℤ) 
  (h1 : K + E = 19) 
  (h2 : H + E = 101) 
  (h3 : K + H + E = 114) : K = 13 := 
by 
  -- Introduction for proof to be skipped
  sorry

end king_arthur_actual_weight_l168_168991


namespace find_value_of_a20_l168_168687

variable {α : Type*} [LinearOrder α] [Field α]

def arithmetic_sequence (a d : α) (n : ℕ) : α :=
  a + (n - 1) * d

def arithmetic_sum (a d : α) (n : ℕ) : α :=
  n * a + (n * (n - 1) / 2) * d

theorem find_value_of_a20 
  (a d : ℝ) 
  (h1 : arithmetic_sequence a d 3 + arithmetic_sequence a d 5 = 4)
  (h2 : arithmetic_sum a d 15 = 60) :
  arithmetic_sequence a d 20 = 10 := 
sorry

end find_value_of_a20_l168_168687


namespace solve_for_x_l168_168344

theorem solve_for_x (x : ℝ) (h : (7 * x + 3) / (x + 5) - 5 / (x + 5) = 2 / (x + 5)) : x = 4 / 7 :=
by
  sorry

end solve_for_x_l168_168344


namespace find_angle_B_l168_168218

-- Define the parallel lines and angles
variables (l m : ℝ) -- Representing the lines as real numbers for simplicity
variables (A C B : ℝ) -- Representing the angles as real numbers

-- The conditions
def parallel_lines (l m : ℝ) : Prop := l = m
def angle_A (A : ℝ) : Prop := A = 100
def angle_C (C : ℝ) : Prop := C = 60

-- The theorem stating that, given the conditions, the angle B is 120 degrees
theorem find_angle_B (l m : ℝ) (A C B : ℝ) 
  (h1 : parallel_lines l m) 
  (h2 : angle_A A) 
  (h3 : angle_C C) : B = 120 :=
sorry

end find_angle_B_l168_168218


namespace symmetric_angle_of_inclination_l168_168381

theorem symmetric_angle_of_inclination (α₁ : ℝ) (h : 0 ≤ α₁ ∧ α₁ < π) : 
  (∃ β₁ : ℝ, (α₁ = 0 ∧ β₁ = 0) ∨ (0 < α₁ ∧ α₁ < π ∧ β₁ = π - α₁)) :=
by
  sorry

end symmetric_angle_of_inclination_l168_168381


namespace polynomial_multiplication_identity_l168_168626

-- Statement of the problem
theorem polynomial_multiplication_identity (x : ℝ) : 
  (25 * x^3) * (12 * x^2) * (1 / (5 * x)^3) = (12 / 5) * x^2 :=
by
  sorry

end polynomial_multiplication_identity_l168_168626


namespace product_of_05_and_2_3_is_1_3_l168_168736

theorem product_of_05_and_2_3_is_1_3 : (0.5 * (2 / 3) = 1 / 3) :=
by sorry

end product_of_05_and_2_3_is_1_3_l168_168736


namespace larger_number_l168_168579

theorem larger_number (x y : ℕ) (h1 : x + y = 47) (h2 : x - y = 3) : max x y = 25 :=
sorry

end larger_number_l168_168579


namespace value_of_x_when_y_is_six_l168_168289

theorem value_of_x_when_y_is_six 
  (k : ℝ) -- The constant of variation
  (h1 : ∀ y : ℝ, x = k / y^2) -- The inverse relationship
  (h2 : y = 2)
  (h3 : x = 1)
  : x = 1 / 9 :=
by
  sorry

end value_of_x_when_y_is_six_l168_168289


namespace functional_equation_solution_l168_168279

theorem functional_equation_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (x + f (x + y)) + f (x * y) = x + f (x + y) + y * f x) :
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = 2 - x) :=
sorry

end functional_equation_solution_l168_168279


namespace number_of_students_from_second_department_is_17_l168_168516

noncomputable def students_selected_from_second_department 
  (total_students : ℕ)
  (num_departments : ℕ)
  (students_per_department : List (ℕ × ℕ))
  (sample_size : ℕ)
  (starting_number : ℕ) : ℕ :=
-- This function will compute the number of students selected from the second department.
sorry

theorem number_of_students_from_second_department_is_17 : 
  students_selected_from_second_department 600 3 
    [(1, 300), (301, 495), (496, 600)] 50 3 = 17 :=
-- Proof is left as an exercise.
sorry

end number_of_students_from_second_department_is_17_l168_168516


namespace bc_ad_divisible_by_u_l168_168907

theorem bc_ad_divisible_by_u 
  (a b c d u : ℤ) 
  (h1 : u ∣ a * c)
  (h2 : u ∣ b * c + a * d)
  (h3 : u ∣ b * d) : 
  u ∣ b * c ∧ u ∣ a * d :=
by
  sorry

end bc_ad_divisible_by_u_l168_168907


namespace parallel_lines_slope_condition_l168_168783

-- Define the first line equation and the slope
def line1 (x : ℝ) : ℝ := 6 * x + 5
def slope1 : ℝ := 6

-- Define the second line equation and the slope
def line2 (x c : ℝ) : ℝ := (3 * c) * x - 7
def slope2 (c : ℝ) : ℝ := 3 * c

-- Theorem stating that if the lines are parallel, the value of c is 2
theorem parallel_lines_slope_condition (c : ℝ) : 
  (slope1 = slope2 c) → c = 2 := 
  by
    sorry -- Proof

end parallel_lines_slope_condition_l168_168783


namespace least_faces_combined_l168_168441

theorem least_faces_combined (a b : ℕ) (h1 : a ≥ 6) (h2 : b ≥ 6)
  (h3 : (∃ k : ℕ, k * a * b = 20) → (∃ m : ℕ, 2 * m = 10 * (k + 10))) 
  (h4 : (∃ n : ℕ, n = (a * b) / 10)) (h5 : ∃ l : ℕ, l = 5) : a + b = 20 :=
by
  sorry

end least_faces_combined_l168_168441


namespace less_than_n_repetitions_l168_168092

variable {n : ℕ} (a : Fin n.succ → ℕ)

def is_repetition (a : Fin n.succ → ℕ) (k l p : ℕ) : Prop :=
  p ≤ (l - k) / 2 ∧
  (∀ i : ℕ, k + 1 ≤ i ∧ i ≤ l - p → a ⟨i, sorry⟩ = a ⟨i + p, sorry⟩) ∧
  (k > 0 → a ⟨k, sorry⟩ ≠ a ⟨k + p, sorry⟩) ∧
  (l < n → a ⟨l - p + 1, sorry⟩ ≠ a ⟨l + 1, sorry⟩)

theorem less_than_n_repetitions (a : Fin n.succ → ℕ) :
  ∃ r : ℕ, r < n ∧ ∀ k l : ℕ, is_repetition a k l r → r < n :=
sorry

end less_than_n_repetitions_l168_168092


namespace symmetric_poly_roots_identity_l168_168235

variable (a b c : ℝ)

theorem symmetric_poly_roots_identity (h1 : a + b + c = 6) (h2 : ab + bc + ca = 5) (h3 : abc = 1) :
  (1 / a^3) + (1 / b^3) + (1 / c^3) = 38 :=
by
  sorry

end symmetric_poly_roots_identity_l168_168235


namespace intersection_correct_l168_168555

def setA : Set ℝ := { x | x - 1 ≤ 0 }
def setB : Set ℝ := { x | x^2 - 4 * x ≤ 0 }
def expected_intersection : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

theorem intersection_correct : (setA ∩ setB) = expected_intersection :=
sorry

end intersection_correct_l168_168555


namespace increase_in_average_l168_168753

theorem increase_in_average (A : ℤ) (avg_after_12 : ℤ) (score_12th_inning : ℤ) (A : ℤ) : 
  score_12th_inning = 75 → avg_after_12 = 64 → (11 * A + score_12th_inning = 768) → (avg_after_12 - A = 1) :=
by
  intros h_score h_avg h_total
  sorry

end increase_in_average_l168_168753


namespace rectangle_integer_sides_noncongruent_count_l168_168547

theorem rectangle_integer_sides_noncongruent_count (h w : ℕ) :
  (2 * (w + h) = 72 ∧ w ≠ h) ∨ ((w = h) ∧ 2 * (w + h) = 72) →
  (∃ (count : ℕ), count = 18) :=
by
  sorry

end rectangle_integer_sides_noncongruent_count_l168_168547


namespace length_of_side_of_largest_square_l168_168243

-- Definitions based on the conditions
def string_length : ℕ := 24

-- The main theorem corresponding to the problem statement.
theorem length_of_side_of_largest_square (h: string_length = 24) : 24 / 4 = 6 :=
by
  sorry

end length_of_side_of_largest_square_l168_168243


namespace arithmetic_sequence_sum_l168_168082

-- Define that a sequence is arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

theorem arithmetic_sequence_sum (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_cond : a 2 + a 10 = 16) : a 4 + a 6 + a 8 = 24 :=
by
  sorry

end arithmetic_sequence_sum_l168_168082


namespace parabola_directrix_eq_neg2_l168_168434

-- Definitions based on conditions
def ellipse_focus (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ x = 2 ∧ y = 0

def parabola_directrix (p x y : ℝ) : Prop :=
  y^2 = 2 * p * x ∧ ∃ x, x = -p / 2

theorem parabola_directrix_eq_neg2 (p : ℝ) (hp : p > 0) :
  (∀ (x y : ℝ), ellipse_focus 9 5 x y → parabola_directrix p x y) →
  (∃ x y : ℝ, parabola_directrix p x y → x = -2) :=
sorry

end parabola_directrix_eq_neg2_l168_168434


namespace smallest_positive_integer_n_l168_168936

theorem smallest_positive_integer_n :
  ∃ n: ℕ, (n > 0) ∧ (∀ k: ℕ, 1 ≤ k ∧ k ≤ n → (∃ d: ℕ, d ∣ (n^2 - 2 * n) ∧ d ∣ k) ∧ (k ∣ (n^2 - 2 * n) → k = d)) ∧ n = 5 :=
by
  sorry

end smallest_positive_integer_n_l168_168936


namespace jenny_profit_l168_168062

-- Definitions for the conditions
def cost_per_pan : ℝ := 10.0
def pans_sold : ℕ := 20
def selling_price_per_pan : ℝ := 25.0

-- Definition for the profit calculation based on the given conditions
def total_revenue : ℝ := pans_sold * selling_price_per_pan
def total_cost : ℝ := pans_sold * cost_per_pan
def profit : ℝ := total_revenue - total_cost

-- The actual theorem statement
theorem jenny_profit : profit = 300.0 := by
  sorry

end jenny_profit_l168_168062


namespace mrs_smith_class_boys_girls_ratio_l168_168495

theorem mrs_smith_class_boys_girls_ratio (total_students boys girls : ℕ) (h1 : boys / girls = 3 / 4) (h2 : boys + girls = 42) : girls = boys + 6 :=
by
  sorry

end mrs_smith_class_boys_girls_ratio_l168_168495


namespace absolute_value_simplification_l168_168764

theorem absolute_value_simplification (x : ℝ) (h : x > 3) : 
  |x - Real.sqrt ((x - 3)^2)| = 3 := 
by 
  sorry

end absolute_value_simplification_l168_168764


namespace positive_inequality_l168_168388

open Real

/-- Given positive real numbers x, y, z such that xyz ≥ 1, prove that
    (x^5 - x^2) / (x^5 + y^2 + z^2) + (y^5 - y^2) / (y^5 + x^2 + z^2) + (z^5 - z^2) / (z^5 + x^2 + y^2) ≥ 0.
-/
theorem positive_inequality (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h : x * y * z ≥ 1) : 
  (x^5 - x^2) / (x^5 + y^2 + z^2) + 
  (y^5 - y^2) / (y^5 + x^2 + z^2) + 
  (z^5 - z^2) / (z^5 + x^2 + y^2) ≥ 0 :=
by
  sorry

end positive_inequality_l168_168388


namespace lattice_point_in_PQE_l168_168117

-- Define points and their integer coordinates
structure Point :=
  (x : ℤ)
  (y : ℤ)

-- Define a convex quadrilateral with integer coordinates
structure ConvexQuadrilateral :=
  (P : Point)
  (Q : Point)
  (R : Point)
  (S : Point)

-- Define the intersection point of diagonals as another point
def diagIntersection (quad: ConvexQuadrilateral) : Point := sorry

-- Define the condition for the sum of angles at P and Q being less than 180 degrees
def sumAnglesLessThan180 (quad : ConvexQuadrilateral) : Prop := sorry

-- Define a function to check if a point is a lattice point
def isLatticePoint (p : Point) : Prop := true  -- Since all points are lattice points by definition

-- Define the proof problem
theorem lattice_point_in_PQE (quad : ConvexQuadrilateral) (E : Point) :
  sumAnglesLessThan180 quad →
  ∃ p : Point, p ≠ quad.P ∧ p ≠ quad.Q ∧ isLatticePoint p ∧ sorry := sorry -- (prove the point is in PQE)

end lattice_point_in_PQE_l168_168117


namespace students_divided_into_groups_l168_168888

theorem students_divided_into_groups (total_students : ℕ) (not_picked : ℕ) (students_per_group : ℕ) (n_groups : ℕ) 
  (h1 : total_students = 64) 
  (h2 : not_picked = 36) 
  (h3 : students_per_group = 7) 
  (h4 : total_students - not_picked = 28) 
  (h5 : 28 / students_per_group = 4) :
  n_groups = 4 :=
by
  sorry

end students_divided_into_groups_l168_168888


namespace Diane_net_loss_l168_168949

variable (x y a b: ℝ)

axiom h1 : x * a = 65
axiom h2 : y * b = 150

theorem Diane_net_loss : (y * b) - (x * a) = 50 := by
  sorry

end Diane_net_loss_l168_168949


namespace combined_mpg_rate_l168_168524

-- Conditions of the problem
def ray_mpg : ℝ := 48
def tom_mpg : ℝ := 24
def ray_distance (s : ℝ) : ℝ := 2 * s
def tom_distance (s : ℝ) : ℝ := s

-- Theorem to prove the combined rate of miles per gallon
theorem combined_mpg_rate (s : ℝ) (h : s > 0) : 
  let total_distance := tom_distance s + ray_distance s
  let ray_gas_usage := ray_distance s / ray_mpg
  let tom_gas_usage := tom_distance s / tom_mpg
  let total_gas_usage := ray_gas_usage + tom_gas_usage
  total_distance / total_gas_usage = 36 := 
by
  sorry

end combined_mpg_rate_l168_168524


namespace find_number_of_non_officers_l168_168366

theorem find_number_of_non_officers
  (avg_salary_all : ℝ)
  (avg_salary_officers : ℝ)
  (avg_salary_non_officers : ℝ)
  (num_officers : ℕ) :
  avg_salary_all = 120 ∧
  avg_salary_officers = 450 ∧
  avg_salary_non_officers = 110 ∧
  num_officers = 15 →
  ∃ N : ℕ, (120 * (15 + N) = 450 * 15 + 110 * N) ∧ N = 495 :=
by
  sorry

end find_number_of_non_officers_l168_168366


namespace solution_set_l168_168844

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ) -- Function for the derivative of f

axiom f_deriv : ∀ x, f' x = (deriv f) x

axiom f_condition1 : ∀ x, f x > 1 - f' x
axiom f_condition2 : f 0 = 0
  
theorem solution_set (x : ℝ) : (e^x * f x > e^x - 1) ↔ (x > 0) := 
  sorry

end solution_set_l168_168844


namespace maximize_mice_two_kittens_different_versions_JPPF_JPPF_combinations_JPPF_two_males_one_female_l168_168533

-- Defining productivity functions for male and female kittens
def male_productivity (K : ℝ) : ℝ := 80 - 4 * K
def female_productivity (K : ℝ) : ℝ := 16 - 0.25 * K

-- Condition (a): Maximizing number of mice caught by 2 kittens
theorem maximize_mice_two_kittens : 
  ∃ (male1 male2 : ℝ) (K_m1 K_m2 : ℝ), 
    (male1 = male_productivity K_m1) ∧ 
    (male2 = male_productivity K_m2) ∧
    (K_m1 = 0) ∧ (K_m2 = 0) ∧
    (male1 + male2 = 160) := 
sorry

-- Condition (b): Different versions of JPPF
theorem different_versions_JPPF : 
  ∃ (v1 v2 v3 : Unit), 
    (v1 ≠ v2) ∧ (v2 ≠ v3) ∧ (v1 ≠ v3) :=
sorry

-- Condition (c): Analytical form of JPPF for each combination
theorem JPPF_combinations :
  ∃ (M K1 K2 : ℝ),
    (M = 160 - 4 * K1 ∧ K1 ≤ 40) ∨
    (M = 32 - 0.5 * K2 ∧ K2 ≤ 64) ∨
    (M = 96 - 0.25 * K2 ∧ K2 ≤ 64) ∨
    (M = 336 - 4 * K2 ∧ 64 < K2 ∧ K2 ≤ 84) :=
sorry

-- Condition (d): Analytical form for 2 males and 1 female
theorem JPPF_two_males_one_female :
  ∃ (M K : ℝ), 
    (0 < K ∧ K ≤ 64 ∧ M = 176 - 0.25 * K) ∨
    (64 < K ∧ K ≤ 164 ∧ M = 416 - 4 * K) :=
sorry

end maximize_mice_two_kittens_different_versions_JPPF_JPPF_combinations_JPPF_two_males_one_female_l168_168533


namespace quadratic_two_distinct_real_roots_l168_168171

theorem quadratic_two_distinct_real_roots (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2 * x₁ + 4 * c = 0 ∧ x₂^2 + 2 * x₂ + 4 * c = 0) ↔ c < 1 / 4 :=
sorry

end quadratic_two_distinct_real_roots_l168_168171


namespace primes_unique_l168_168967

-- Let's define that p, q, r are prime numbers, and define the main conditions.
def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem primes_unique (p q r : ℕ) 
  (hp : is_prime p) 
  (hq : is_prime q)
  (hr : is_prime r)
  (div1 : (p^4 - 1) % (q * r) = 0)
  (div2 : (q^4 - 1) % (p * r) = 0)
  (div3 : (r^4 - 1) % (p * q) = 0) :
  (p = 2 ∧ q = 3 ∧ r = 5) ∨ (p = 2 ∧ q = 5 ∧ r = 3) ∨ (p = 3 ∧ q = 2 ∧ r = 5) ∨ 
  (p = 3 ∧ q = 5 ∧ r = 2) ∨ (p = 5 ∧ q = 2 ∧ r = 3) ∨ (p = 5 ∧ q = 3 ∧ r = 2) :=
by sorry

end primes_unique_l168_168967


namespace right_triangle_perimeter_l168_168916

theorem right_triangle_perimeter (area : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) 
  (h_area : area = 120)
  (h_a : a = 24)
  (h_area_eq : area = (1/2) * a * b)
  (h_c : c^2 = a^2 + b^2) :
  a + b + c = 60 :=
by
  sorry

end right_triangle_perimeter_l168_168916


namespace S2_side_length_656_l168_168685

noncomputable def S1_S2_S3_side_lengths (l1 l2 a b c : ℕ) (total_length : ℕ) : Prop :=
  l1 + l2 + a + b + c = total_length

theorem S2_side_length_656 :
  ∃ (l1 l2 a c : ℕ), S1_S2_S3_side_lengths l1 l2 a 656 c 3322 :=
by
  sorry

end S2_side_length_656_l168_168685


namespace carlos_finishes_first_l168_168973

theorem carlos_finishes_first
  (a : ℝ) -- Andy's lawn area
  (r : ℝ) -- Andy's mowing rate
  (hBeth_lawn : ∀ (b : ℝ), b = a / 3) -- Beth's lawn area
  (hCarlos_lawn : ∀ (c : ℝ), c = a / 4) -- Carlos' lawn area
  (hCarlos_Beth_rate : ∀ (rc rb : ℝ), rc = r / 2 ∧ rb = r / 2) -- Carlos' and Beth's mowing rate
  : (∃ (ta tb tc : ℝ), ta = a / r ∧ tb = (2 * a) / (3 * r) ∧ tc = a / (2 * r) ∧ tc < tb ∧ tc < ta) :=
-- Prove that the mowing times are such that Carlos finishes first
sorry

end carlos_finishes_first_l168_168973


namespace decreased_area_of_equilateral_triangle_l168_168222

theorem decreased_area_of_equilateral_triangle 
    (A : ℝ) (hA : A = 100 * Real.sqrt 3) 
    (decrease : ℝ) (hdecrease : decrease = 6) :
    let s := Real.sqrt (4 * A / Real.sqrt 3)
    let s' := s - decrease
    let A' := (s' ^ 2 * Real.sqrt 3) / 4
    A - A' = 51 * Real.sqrt 3 :=
by
  sorry

end decreased_area_of_equilateral_triangle_l168_168222


namespace total_doctors_and_nurses_l168_168476

theorem total_doctors_and_nurses
    (ratio_doctors_nurses : ℕ -> ℕ -> Prop)
    (num_nurses : ℕ)
    (h₁ : ratio_doctors_nurses 2 3)
    (h₂ : num_nurses = 150) :
    ∃ num_doctors total_doctors_nurses, 
    (total_doctors_nurses = num_doctors + num_nurses) 
    ∧ (num_doctors / num_nurses = 2 / 3) 
    ∧ total_doctors_nurses = 250 := 
by
  sorry

end total_doctors_and_nurses_l168_168476


namespace value_of_expression_l168_168395

theorem value_of_expression 
  (triangle square : ℝ) 
  (h1 : 3 + triangle = 5) 
  (h2 : triangle + square = 7) : 
  triangle + triangle + triangle + square + square = 16 :=
by
  sorry

end value_of_expression_l168_168395


namespace hot_dog_cost_l168_168640

variable {Real : Type} [LinearOrderedField Real]

-- Define the cost of a hamburger and a hot dog
variables (h d : Real)

-- Arthur's buying conditions
def condition1 := 3 * h + 4 * d = 10
def condition2 := 2 * h + 3 * d = 7

-- Problem statement: Proving that the cost of a hot dog is 1 dollar
theorem hot_dog_cost
    (h d : Real)
    (hc1 : condition1 h d)
    (hc2 : condition2 h d) : 
    d = 1 :=
sorry

end hot_dog_cost_l168_168640


namespace blue_faces_ratio_l168_168497

-- Define the cube dimensions and derived properties
def cube_dimension : ℕ := 13
def original_cube_surface_area : ℕ := 6 * (cube_dimension ^ 2)
def total_number_of_mini_cubes : ℕ := cube_dimension ^ 3
def total_mini_cube_faces : ℕ := 6 * total_number_of_mini_cubes
def red_faces : ℕ := original_cube_surface_area
def blue_faces : ℕ := total_mini_cube_faces - red_faces
def blue_to_red_ratio : ℕ := blue_faces / red_faces

-- The statement to be proved
theorem blue_faces_ratio :
  blue_to_red_ratio = 12 := by
  sorry

end blue_faces_ratio_l168_168497


namespace paper_string_area_l168_168512

theorem paper_string_area (side len overlap : ℝ) (n : ℕ) (h_side : side = 30) 
                          (h_len : len = 30) (h_overlap : overlap = 7) (h_n : n = 6) :
  let area_one_sheet := side * len
  let effective_len := side - overlap
  let total_length := len + effective_len * (n - 1)
  let width := side
  let area := total_length * width
  area = 4350 := 
by
  sorry

end paper_string_area_l168_168512


namespace ceil_sqrt_225_eq_15_l168_168213

theorem ceil_sqrt_225_eq_15 : ⌈ Real.sqrt 225 ⌉ = 15 :=
by
  sorry

end ceil_sqrt_225_eq_15_l168_168213


namespace point_on_y_axis_l168_168733

theorem point_on_y_axis (y : ℝ) :
  let A := (1, 0, 2)
  let B := (1, -3, 1)
  let M := (0, y, 0)
  dist A M = dist B M → y = -1 :=
by sorry

end point_on_y_axis_l168_168733


namespace trapezium_second_side_length_l168_168177

theorem trapezium_second_side_length
  (side1 : ℝ)
  (height : ℝ)
  (area : ℝ) 
  (h1 : side1 = 20) 
  (h2 : height = 13) 
  (h3 : area = 247) : 
  ∃ side2 : ℝ, 0 ≤ side2 ∧ ∀ side2, area = 1 / 2 * (side1 + side2) * height → side2 = 18 :=
by
  use 18
  sorry

end trapezium_second_side_length_l168_168177


namespace remainder_when_divided_by_10_l168_168535

theorem remainder_when_divided_by_10 :
  (4219 * 2675 * 394082 * 5001) % 10 = 0 :=
sorry

end remainder_when_divided_by_10_l168_168535


namespace at_least_one_A_or_B_selected_prob_l168_168791

theorem at_least_one_A_or_B_selected_prob :
  let students := ['A', 'B', 'C', 'D']
  let total_pairs := 6
  let complementary_event_prob := 1 / total_pairs
  let at_least_one_A_or_B_prob := 1 - complementary_event_prob
  at_least_one_A_or_B_prob = 5 / 6 :=
by
  let students := ['A', 'B', 'C', 'D']
  let total_pairs := 6
  let complementary_event_prob := 1 / total_pairs
  let at_least_one_A_or_B_prob := 1 - complementary_event_prob
  sorry

end at_least_one_A_or_B_selected_prob_l168_168791


namespace option_d_correct_l168_168320

theorem option_d_correct (a b : ℝ) : (a - b)^2 = (b - a)^2 := 
by {
  sorry
}

end option_d_correct_l168_168320


namespace odd_function_iff_l168_168037

variable {α : Type*} [LinearOrderedField α]

noncomputable def f (a b x : α) : α := x * abs (x + a) + b

theorem odd_function_iff (a b : α) : 
  (∀ x : α, f a b (-x) = -f a b x) ↔ (a^2 + b^2 = 0) :=
by
  sorry

end odd_function_iff_l168_168037


namespace transform_equation_l168_168911

theorem transform_equation (x : ℝ) :
  x^2 + 4 * x + 1 = 0 → (x + 2)^2 = 3 :=
by
  intro h
  sorry

end transform_equation_l168_168911


namespace min_moves_to_balance_stacks_l168_168210

theorem min_moves_to_balance_stacks :
  let stack1 := 9
  let stack2 := 7
  let stack3 := 5
  let stack4 := 10
  let target := 8
  let total_coins := stack1 + stack2 + stack3 + stack4
  total_coins = 31 →
  ∃ moves, moves = 11 ∧
    (stack1 + 3 * moves = target) ∧
    (stack2 + 3 * moves = target) ∧
    (stack3 + 3 * moves = target) ∧
    (stack4 + 3 * moves = target) :=
sorry

end min_moves_to_balance_stacks_l168_168210


namespace initial_pencils_correct_l168_168374

variable (initial_pencils : ℕ)
variable (pencils_added : ℕ := 45)
variable (total_pencils : ℕ := 72)

theorem initial_pencils_correct (h : total_pencils = initial_pencils + pencils_added) : initial_pencils = 27 := by
  sorry

end initial_pencils_correct_l168_168374


namespace sum_squares_and_products_of_nonneg_reals_l168_168248

theorem sum_squares_and_products_of_nonneg_reals {x y z : ℝ} 
  (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) 
  (h1 : x^2 + y^2 + z^2 = 52) 
  (h2 : x*y + y*z + z*x = 27) : 
  x + y + z = Real.sqrt 106 := 
by 
  sorry

end sum_squares_and_products_of_nonneg_reals_l168_168248


namespace pirate_coins_l168_168619

def coins_remain (k : ℕ) (x : ℕ) : ℕ :=
  if k = 0 then x else coins_remain (k - 1) x * (15 - k) / 15

theorem pirate_coins (x : ℕ) :
  (∀ k < 15, (k + 1) * coins_remain k x % 15 = 0) → 
  coins_remain 14 x = 8442 :=
sorry

end pirate_coins_l168_168619


namespace max_abc_value_l168_168597

theorem max_abc_value (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a * b + b * c = 518) (h2 : a * b - a * c = 360) : 
  a * b * c ≤ 1008 :=
sorry

end max_abc_value_l168_168597


namespace hall_length_width_difference_l168_168354

theorem hall_length_width_difference : 
  ∃ L W : ℝ, W = (1 / 2) * L ∧ L * W = 450 ∧ L - W = 15 :=
sorry

end hall_length_width_difference_l168_168354


namespace points_earned_l168_168808

def each_enemy_points : ℕ := 3
def total_enemies : ℕ := 6
def defeated_enemies : ℕ := total_enemies - 2

theorem points_earned : defeated_enemies * each_enemy_points = 12 :=
by
  -- proof goes here
  sorry

end points_earned_l168_168808


namespace solve_coin_problem_l168_168641

def coin_problem : Prop :=
  ∃ (x y z : ℕ), 
  1 * x + 2 * y + 5 * z = 71 ∧ 
  x = y ∧ 
  x + y + z = 31 ∧ 
  x = 12 ∧ 
  y = 12 ∧ 
  z = 7

theorem solve_coin_problem : coin_problem :=
  sorry

end solve_coin_problem_l168_168641


namespace ratio_of_candies_l168_168094

theorem ratio_of_candies (emily_candies jennifer_candies bob_candies : ℕ)
  (h1 : emily_candies = 6)
  (h2 : bob_candies = 4)
  (h3 : jennifer_candies = 2 * emily_candies) : 
  jennifer_candies / bob_candies = 3 := 
by
  sorry

end ratio_of_candies_l168_168094


namespace emily_necklaces_l168_168465

theorem emily_necklaces (total_beads : ℕ) (beads_per_necklace : ℕ) (necklaces_made : ℕ) 
  (h1 : total_beads = 52)
  (h2 : beads_per_necklace = 2)
  (h3 : necklaces_made = total_beads / beads_per_necklace) :
  necklaces_made = 26 :=
by
  rw [h1, h2] at h3
  exact h3

end emily_necklaces_l168_168465


namespace combined_weight_of_Meg_and_Chris_cats_l168_168940

-- Definitions based on the conditions
def ratio (M A C : ℕ) : Prop := 13 * A = 21 * M ∧ 13 * C = 28 * M 
def half_anne (M A : ℕ) : Prop := M = 20 + A / 2
def total_weight (M A C T : ℕ) : Prop := T = M + A + C

-- Theorem statement
theorem combined_weight_of_Meg_and_Chris_cats (M A C T : ℕ) 
  (h1 : ratio M A C) 
  (h2 : half_anne M A) 
  (h3 : total_weight M A C T) : 
  M + C = 328 := 
sorry

end combined_weight_of_Meg_and_Chris_cats_l168_168940


namespace value_of_y_l168_168833

variable (x y : ℤ)

-- Define the conditions
def condition1 : Prop := 3 * (x^2 + x + 1) = y - 6
def condition2 : Prop := x = -3

-- Theorem to prove
theorem value_of_y (h1 : condition1 x y) (h2 : condition2 x) : y = 27 := by
  sorry

end value_of_y_l168_168833


namespace product_of_positive_solutions_l168_168589

theorem product_of_positive_solutions :
  ∃ n : ℕ, ∃ p : ℕ, Prime p ∧ (n^2 - 41*n + 408 = p) ∧ (∀ m : ℕ, (Prime p ∧ (m^2 - 41*m + 408 = p)) → m = n) ∧ (n = 406) := 
sorry

end product_of_positive_solutions_l168_168589


namespace negation_equivalence_l168_168157

theorem negation_equivalence {Triangle : Type} (has_circumcircle : Triangle → Prop) :
  ¬ (∃ (t : Triangle), ¬ has_circumcircle t) ↔ (∀ (t : Triangle), has_circumcircle t) :=
by
  sorry

end negation_equivalence_l168_168157


namespace negation_false_l168_168788

theorem negation_false (a b : ℝ) : ¬ ((a ≤ 1 ∨ b ≤ 1) → a + b ≤ 2) :=
sorry

end negation_false_l168_168788


namespace number_of_new_students_l168_168262

variable (O N : ℕ)
variable (H1 : 48 * O + 32 * N = 44 * 160)
variable (H2 : O + N = 160)

theorem number_of_new_students : N = 40 := sorry

end number_of_new_students_l168_168262


namespace linear_combination_of_matrices_l168_168853

variable (A B : Matrix (Fin 3) (Fin 3) ℤ) 

def matrixA : Matrix (Fin 3) (Fin 3) ℤ := 
  ![
    ![2, -4, 0],
    ![-1, 5, 1],
    ![0, 3, -7]
  ]

def matrixB : Matrix (Fin 3) (Fin 3) ℤ := 
  ![
    ![4, -1, -2],
    ![0, -3, 5],
    ![2, 0, -4]
  ]

theorem linear_combination_of_matrices :
  3 • matrixA - 2 • matrixB = 
  ![
    ![-2, -10, 4],
    ![-3, 21, -7],
    ![-4, 9, -13]
  ] :=
sorry

end linear_combination_of_matrices_l168_168853


namespace gcd_min_val_l168_168821

theorem gcd_min_val (p q r : ℕ) (hpq : Nat.gcd p q = 210) (hpr : Nat.gcd p r = 1155) : ∃ (g : ℕ), g = Nat.gcd q r ∧ g = 105 :=
by
  sorry

end gcd_min_val_l168_168821


namespace certain_time_in_seconds_l168_168041

theorem certain_time_in_seconds
  (ratio : ℕ) (minutes : ℕ) (time_in_minutes : ℕ) (seconds_in_a_minute : ℕ)
  (h_ratio : ratio = 8)
  (h_minutes : minutes = 4)
  (h_time : time_in_minutes = minutes)
  (h_conversion : seconds_in_a_minute = 60) :
  time_in_minutes * seconds_in_a_minute = 240 :=
by
  sorry

end certain_time_in_seconds_l168_168041


namespace arithmetic_mean_is_12_l168_168939

/-- The arithmetic mean of the numbers 3, 11, 7, 9, 15, 13, 8, 19, 17, 21, 14, and 7 is equal to 12 -/
theorem arithmetic_mean_is_12 : 
  let numbers := [3, 11, 7, 9, 15, 13, 8, 19, 17, 21, 14, 7]
  let sum := numbers.foldl (· + ·) 0
  let count := numbers.length
  (sum / count) = 12 :=
by
  sorry

end arithmetic_mean_is_12_l168_168939


namespace cooks_in_restaurant_l168_168291

theorem cooks_in_restaurant
  (C W : ℕ) 
  (h1 : C * 8 = 3 * W) 
  (h2 : C * 4 = (W + 12)) :
  C = 9 :=
by
  sorry

end cooks_in_restaurant_l168_168291


namespace factorization_of_expression_l168_168116

theorem factorization_of_expression (a b c : ℝ) : 
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / 
  ((a - b)^3 + (b - c)^3 + (c - a)^3) = 
  (a^2 + ab + b^2) * (b^2 + bc + c^2) * (c^2 + ca + a^2) :=
by
  sorry

end factorization_of_expression_l168_168116


namespace sum_of_possible_values_l168_168735

theorem sum_of_possible_values (N : ℝ) (h : N * (N - 8) = 4) : ∃ S : ℝ, S = 8 :=
sorry

end sum_of_possible_values_l168_168735


namespace investment_amount_l168_168391

theorem investment_amount (x y : ℝ) (hx : x ≤ 11000) (hy : 0.07 * x + 0.12 * y ≥ 2450) : x + y = 25000 := 
sorry

end investment_amount_l168_168391


namespace selecting_elements_l168_168875

theorem selecting_elements (P Q S : ℕ) (a : ℕ) 
    (h1 : P = Nat.choose 17 (2 * a - 1))
    (h2 : Q = Nat.choose 17 (2 * a))
    (h3 : S = Nat.choose 18 12) :
    P + Q = S → (a = 3 ∨ a = 6) :=
by
  sorry

end selecting_elements_l168_168875


namespace sets_are_equal_l168_168064

def X : Set ℝ := {x | ∃ n : ℤ, x = (2 * n + 1) * Real.pi}
def Y : Set ℝ := {y | ∃ k : ℤ, y = (4 * k + 1) * Real.pi ∨ y = (4 * k - 1) * Real.pi}

theorem sets_are_equal : X = Y :=
by sorry

end sets_are_equal_l168_168064


namespace surface_area_of_given_cylinder_l168_168984

noncomputable def surface_area_of_cylinder (length width : ℝ) : ℝ :=
  let r := (length / (2 * Real.pi))
  let h := width
  2 * Real.pi * r^2 + 2 * Real.pi * r * h

theorem surface_area_of_given_cylinder : 
  surface_area_of_cylinder (4 * Real.pi) 2 = 16 * Real.pi :=
by
  -- Proof will be filled here
  sorry

end surface_area_of_given_cylinder_l168_168984


namespace cost_of_tax_free_items_l168_168560

-- Definitions based on the conditions.
def total_spending : ℝ := 20
def sales_tax_percentage : ℝ := 0.30
def tax_rate : ℝ := 0.06

-- Derived calculations for intermediate variables for clarity
def taxable_items_cost : ℝ := total_spending * (1 - sales_tax_percentage)
def sales_tax_paid : ℝ := taxable_items_cost * tax_rate
def tax_free_items_cost : ℝ := total_spending - taxable_items_cost

-- Lean 4 statement for the problem
theorem cost_of_tax_free_items :
  tax_free_items_cost = 6 := by
    -- The proof would go here, but we are skipping it.
    sorry

end cost_of_tax_free_items_l168_168560


namespace perpendicular_line_equation_l168_168938

theorem perpendicular_line_equation :
  (∀ (x y : ℝ), 2 * x + 3 * y + 1 = 0 → x - 3 * y + 4 = 0 →
  ∃ (l : ℝ) (m : ℝ), m = 4 / 3 ∧ y = m * x + l → y = 4 / 3 * x + 1 / 9) 
  ∧ (∀ (x y : ℝ), 3 * x + 4 * y - 7 = 0 → -3 / 4 * 4 / 3 = -1) :=
by 
  sorry

end perpendicular_line_equation_l168_168938


namespace bottles_left_on_shelf_l168_168326

variable (initial_bottles : ℕ)
variable (bottles_jason : ℕ)
variable (bottles_harry : ℕ)

theorem bottles_left_on_shelf (h₁ : initial_bottles = 35) (h₂ : bottles_jason = 5) (h₃ : bottles_harry = bottles_jason + 6) :
  initial_bottles - (bottles_jason + bottles_harry) = 24 := by
  sorry

end bottles_left_on_shelf_l168_168326


namespace correct_operation_l168_168876

variable (a b : ℝ)

theorem correct_operation : 
  ¬ ((a - b) ^ 2 = a ^ 2 - b ^ 2) ∧
  ¬ ((a^3) ^ 2 = a ^ 5) ∧
  (a ^ 5 / a ^ 3 = a ^ 2) ∧
  ¬ (a ^ 3 + a ^ 2 = a ^ 5) :=
by
  sorry

end correct_operation_l168_168876


namespace function_C_is_even_l168_168585

theorem function_C_is_even : ∀ x : ℝ, 2 * (-x)^2 - 1 = 2 * x^2 - 1 :=
by
  intro x
  sorry

end function_C_is_even_l168_168585


namespace chord_line_equation_l168_168343

theorem chord_line_equation (x y : ℝ) 
  (ellipse : ∀ (x y : ℝ), x^2 / 36 + y^2 / 9 = 1)
  (bisect_point : x / 2 = 4 ∧ y / 2 = 2) : 
  x + 2 * y - 8 = 0 :=
sorry

end chord_line_equation_l168_168343


namespace sufficient_but_not_necessary_condition_l168_168255

theorem sufficient_but_not_necessary_condition
  (a b : ℝ) (h : a > b + 1) : (a > b) ∧ ¬ (∀ (a b : ℝ), a > b → a > b + 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l168_168255


namespace transformed_expression_value_l168_168798

theorem transformed_expression_value :
  (240 / 80) * 60 / 40 + 10 = 14.5 :=
by
  sorry

end transformed_expression_value_l168_168798


namespace composite_product_division_l168_168183

-- Define the first 12 positive composite integers
def first_six_composites := [4, 6, 8, 9, 10, 12]
def next_six_composites := [14, 15, 16, 18, 20, 21]

-- Define the product of a list of integers
def product (l : List ℕ) : ℕ :=
  l.foldl (· * ·) 1

-- Define the target problem statement
theorem composite_product_division :
  (product first_six_composites : ℚ) / (product next_six_composites : ℚ) = 1 / 49 := by
  sorry

end composite_product_division_l168_168183


namespace cookies_prepared_l168_168693

theorem cookies_prepared (n_people : ℕ) (cookies_per_person : ℕ) (total_cookies : ℕ) 
  (h1 : n_people = 25) (h2 : cookies_per_person = 45) : total_cookies = 1125 :=
by
  sorry

end cookies_prepared_l168_168693


namespace special_even_diff_regular_l168_168595

def first_n_even_sum (n : ℕ) : ℕ := 2 * (n * (n + 1) / 2)

def special_even_sum (n : ℕ) : ℕ :=
  let sum_cubes := (n * (n + 1) / 2) ^ 2
  let sum_squares := n * (n + 1) * (2 * n + 1) / 6
  2 * (sum_cubes + sum_squares)

theorem special_even_diff_regular : 
  let n := 100
  special_even_sum n - first_n_even_sum n = 51403900 :=
by
  sorry

end special_even_diff_regular_l168_168595


namespace pascal_triangle_ratio_l168_168884

theorem pascal_triangle_ratio (n r : ℕ) 
  (h1 : (3 * r + 3 = 2 * n - 2 * r))
  (h2 : (4 * r + 8 = 3 * n - 3 * r - 3)) : 
  n = 34 :=
sorry

end pascal_triangle_ratio_l168_168884


namespace find_x_l168_168756

-- Define the conditions as given in the problem
def angle1 (x : ℝ) : ℝ := 6 * x
def angle2 (x : ℝ) : ℝ := 3 * x
def angle3 (x : ℝ) : ℝ := x
def angle4 (x : ℝ) : ℝ := 5 * x
def sum_of_angles (x : ℝ) : ℝ := angle1 x + angle2 x + angle3 x + angle4 x

-- State the problem: prove that x equals 24 given the sum of angles is 360 degrees
theorem find_x (x : ℝ) (h : sum_of_angles x = 360) : x = 24 :=
by
  sorry

end find_x_l168_168756


namespace page_cost_in_cents_l168_168145

theorem page_cost_in_cents (notebooks pages_per_notebook total_cost : ℕ)
  (h_notebooks : notebooks = 2)
  (h_pages_per_notebook : pages_per_notebook = 50)
  (h_total_cost : total_cost = 5 * 100) :
  (total_cost / (notebooks * pages_per_notebook)) = 5 :=
by
  sorry

end page_cost_in_cents_l168_168145


namespace triangle_equilateral_l168_168906

noncomputable def point := (ℝ × ℝ)

noncomputable def D : point := (0, 0)
noncomputable def E : point := (2, 0)
noncomputable def F : point := (1, Real.sqrt 3)

noncomputable def dist (p1 p2 : point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def D' (l : ℝ) : point :=
  let ED := dist E D
  (D.1 + l * ED * (Real.sqrt 3), D.2 + l * ED)

noncomputable def E' (l : ℝ) : point :=
  let DF := dist D F
  (E.1 + l * DF * (Real.sqrt 3), E.2 + l * DF)

noncomputable def F' (l : ℝ) : point :=
  let DE := dist D E
  (F.1 - 2 * l * DE, F.2 + (Real.sqrt 3 - l * DE))

theorem triangle_equilateral (l : ℝ) (h : l = 1 / Real.sqrt 3) :
  let DD' := dist D (D' l)
  let EE' := dist E (E' l)
  let FF' := dist F (F' l)
  dist (D' l) (E' l) = dist (E' l) (F' l) ∧ dist (E' l) (F' l) = dist (F' l) (D' l) ∧ dist (F' l) (D' l) = dist (D' l) (E' l) := sorry

end triangle_equilateral_l168_168906


namespace alexis_pants_l168_168605

theorem alexis_pants (P D : ℕ) (A_p : ℕ)
  (h1 : P + D = 13)
  (h2 : 3 * D = 18)
  (h3 : A_p = 3 * P) : A_p = 21 :=
  sorry

end alexis_pants_l168_168605


namespace find_phi_l168_168635

theorem find_phi :
  ∀ φ : ℝ, 0 < φ ∧ φ < 90 → 
    (∃θ : ℝ, θ = 144 ∧ θ = 2 * φ ∧ (144 - θ) = 72) → φ = 81 :=
by
  intros φ h1 h2
  sorry

end find_phi_l168_168635


namespace factor_expression_l168_168682

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l168_168682


namespace determine_constants_l168_168126

theorem determine_constants (a b c d : ℝ) 
  (periodic : (2 * (2 * Real.pi / b) = 4 * Real.pi))
  (vert_shift : d = 3)
  (max_val : (d + a = 8))
  (min_val : (d - a = -2)) :
  a = 5 ∧ b = 1 :=
by
  sorry

end determine_constants_l168_168126


namespace sum_of_primes_146_sum_of_primes_99_l168_168215

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Prove that there exist tuples of (p1, p2, p3) such that their sum is 146
theorem sum_of_primes_146 :
  ∃ (p1 p2 p3 : ℕ), is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ p1 + p2 + p3 = 146 ∧ p1 ≤ p2 ∧ p2 ≤ p3 :=
sorry

-- Prove that there exist tuples of (p1, p2, p3) such that their sum is 99
theorem sum_of_primes_99 :
  ∃ (p1 p2 p3 : ℕ), is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ p1 + p2 + p3 = 99 ∧ p1 ≤ p2 ∧ p2 ≤ p3 :=
sorry

end sum_of_primes_146_sum_of_primes_99_l168_168215


namespace polynomial_roots_geometric_progression_q_l168_168227

theorem polynomial_roots_geometric_progression_q :
    ∃ (a r : ℝ), (a ≠ 0) ∧ (r ≠ 0) ∧
    (a + a * r + a * r ^ 2 + a * r ^ 3 = 0) ∧
    (a ^ 4 * r ^ 6 = 16) ∧
    (a ^ 2 + (a * r) ^ 2 + (a * r ^ 2) ^ 2 + (a * r ^ 3) ^ 2 = 16) :=
by
    sorry

end polynomial_roots_geometric_progression_q_l168_168227


namespace probability_of_fx_leq_zero_is_3_over_10_l168_168578

noncomputable def fx (x : ℝ) : ℝ := -x + 2

def in_interval (x : ℝ) (a b : ℝ) : Prop := a ≤ x ∧ x ≤ b

def probability_fx_leq_zero : ℚ :=
  let interval_start := -5
  let interval_end := 5
  let fx_leq_zero_start := 2
  let fx_leq_zero_end := 5
  (fx_leq_zero_end - fx_leq_zero_start) / (interval_end - interval_start)

theorem probability_of_fx_leq_zero_is_3_over_10 :
  probability_fx_leq_zero = 3 / 10 :=
sorry

end probability_of_fx_leq_zero_is_3_over_10_l168_168578


namespace days_in_week_l168_168350

theorem days_in_week {F D : ℕ} (h1 : F = 3 + 11) (h2 : F = 2 * D) : D = 7 :=
by
  sorry

end days_in_week_l168_168350


namespace practice_minutes_l168_168403

def month_total_days : ℕ := (2 * 6) + (2 * 7)

def piano_daily_minutes : ℕ := 25

def violin_daily_minutes := piano_daily_minutes * 3

def flute_daily_minutes := violin_daily_minutes / 2

theorem practice_minutes (piano_total : ℕ) (violin_total : ℕ) (flute_total : ℕ) :
  (26 * piano_daily_minutes = 650) ∧ 
  (20 * violin_daily_minutes = 1500) ∧ 
  (16 * flute_daily_minutes = 600) := by
  sorry

end practice_minutes_l168_168403


namespace cost_of_flowers_l168_168076

theorem cost_of_flowers 
  (interval : ℕ) (perimeter : ℕ) (cost_per_flower : ℕ)
  (h_interval : interval = 30)
  (h_perimeter : perimeter = 1500)
  (h_cost : cost_per_flower = 5000) :
  (perimeter / interval) * cost_per_flower = 250000 :=
by
  sorry

end cost_of_flowers_l168_168076


namespace lost_card_number_l168_168908

theorem lost_card_number (n x : ℕ) (sum_n : ℕ) (h_sum : sum_n = n * (n + 1) / 2) (h_remaining_sum : sum_n - x = 101) : x = 4 :=
sorry

end lost_card_number_l168_168908


namespace sum_of_b_values_l168_168321

theorem sum_of_b_values (b1 b2 : ℝ) : 
  (∀ x : ℝ, (9 * x^2 + (b1 + 15) * x + 16 = 0 ∨ 9 * x^2 + (b2 + 15) * x + 16 = 0) ∧ 
           (b1 + 15)^2 - 4 * 9 * 16 = 0 ∧ 
           (b2 + 15)^2 - 4 * 9 * 16 = 0) → 
  (b1 + b2) = -30 := 
sorry

end sum_of_b_values_l168_168321


namespace smallest_power_of_7_not_palindrome_l168_168985

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

theorem smallest_power_of_7_not_palindrome : ∃ n : ℕ, n > 0 ∧ 7^n = 2401 ∧ ¬is_palindrome (7^n) ∧ (∀ m : ℕ, m > 0 ∧ ¬is_palindrome (7^m) → 7^n ≤ 7^m) :=
by
  sorry

end smallest_power_of_7_not_palindrome_l168_168985


namespace find_circle_center_l168_168959

def circle_center_condition (x y : ℝ) : Prop :=
  (3 * x - 4 * y = 24 ∨ 3 * x - 4 * y = -12) ∧ 3 * x + 2 * y = 0

theorem find_circle_center :
  ∃ (x y : ℝ), circle_center_condition x y ∧ (x, y) = (2/3, -1) :=
by
  sorry

end find_circle_center_l168_168959


namespace b_amount_l168_168883

-- Define the conditions
def total_amount (a b : ℝ) : Prop := a + b = 1210
def fraction_condition (a b : ℝ) : Prop := (1/3) * a = (1/4) * b

-- Define the main theorem to prove B's amount
theorem b_amount (a b : ℝ) (h1 : total_amount a b) (h2 : fraction_condition a b) : b = 691.43 :=
sorry

end b_amount_l168_168883


namespace solve_a_for_pure_imaginary_l168_168812

theorem solve_a_for_pure_imaginary (a : ℝ) : (1 - a^2 = 0) ∧ (2 * a ≠ 0) → (a = 1 ∨ a = -1) :=
by
  sorry

end solve_a_for_pure_imaginary_l168_168812


namespace fraction_a_over_b_l168_168785

theorem fraction_a_over_b (x y a b : ℝ) (hb : b ≠ 0) (h1 : 4 * x - 2 * y = a) (h2 : 9 * y - 18 * x = b) :
  a / b = -2 / 9 :=
by
  sorry

end fraction_a_over_b_l168_168785


namespace roots_quadratic_reciprocal_l168_168332

theorem roots_quadratic_reciprocal (x1 x2 : ℝ) (h1 : x1 + x2 = -8) (h2 : x1 * x2 = 4) :
  (1 / x1) + (1 / x2) = -2 :=
sorry

end roots_quadratic_reciprocal_l168_168332


namespace ludek_unique_stamps_l168_168487

theorem ludek_unique_stamps (K M L : ℕ) (k_m_shared k_l_shared m_l_shared : ℕ)
  (hk : K + M = 101)
  (hl : K + L = 115)
  (hm : M + L = 110)
  (k_m_shared := 5)
  (k_l_shared := 12)
  (m_l_shared := 7) :
  L - k_l_shared - m_l_shared = 43 :=
by
  sorry

end ludek_unique_stamps_l168_168487


namespace fraction_diff_equals_7_over_12_l168_168872

noncomputable def fraction_diff : ℚ :=
  (2 + 4 + 6) / (1 + 3 + 5) - (1 + 3 + 5) / (2 + 4 + 6)

theorem fraction_diff_equals_7_over_12 : fraction_diff = 7 / 12 := by
  sorry

end fraction_diff_equals_7_over_12_l168_168872


namespace quadratic_at_most_two_roots_l168_168445

theorem quadratic_at_most_two_roots (a b c x1 x2 x3 : ℝ) (ha : a ≠ 0) 
(h1 : a * x1^2 + b * x1 + c = 0)
(h2 : a * x2^2 + b * x2 + c = 0)
(h3 : a * x3^2 + b * x3 + c = 0)
(h_distinct : x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) : 
false :=
sorry

end quadratic_at_most_two_roots_l168_168445


namespace check_range_a_l168_168750

open Set

def A : Set ℝ := {x | x < -1/2 ∨ x > 1}
def B (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x - 1 ≤ 0}

theorem check_range_a :
  (∃! x₁ x₂ : ℤ, x₁ ≠ x₂ ∧ (x₁ : ℝ) ∈ A ∩ B a ∧ (x₂ : ℝ) ∈ A ∩ B a) →
  a ∈ Icc (4/3 : ℝ) (15/8 : ℝ) :=
sorry

end check_range_a_l168_168750


namespace none_of_these_l168_168425

def y_values_match (f : ℕ → ℕ) : Prop :=
  f 0 = 200 ∧ f 1 = 140 ∧ f 2 = 80 ∧ f 3 = 20 ∧ f 4 = 0

theorem none_of_these :
  ¬ (∃ f : ℕ → ℕ, 
    (∀ x, f x = 200 - 15 * x ∨ 
    f x = 200 - 20 * x + 5 * x^2 ∨ 
    f x = 200 - 30 * x + 10 * x^2 ∨ 
    f x = 150 - 50 * x) ∧ 
    y_values_match f) :=
by sorry

end none_of_these_l168_168425


namespace profit_share_of_B_l168_168823

theorem profit_share_of_B (P : ℝ) (A_share B_share C_share : ℝ) :
  let A_initial := 8000
  let B_initial := 10000
  let C_initial := 12000
  let total_capital := A_initial + B_initial + C_initial
  let investment_ratio_A := A_initial / total_capital
  let investment_ratio_B := B_initial / total_capital
  let investment_ratio_C := C_initial / total_capital
  let total_profit := 4200
  let diff_AC := 560
  A_share = (investment_ratio_A * total_profit) →
  B_share = (investment_ratio_B * total_profit) →
  C_share = (investment_ratio_C * total_profit) →
  C_share - A_share = diff_AC →
  B_share = 1400 :=
by
  intros
  sorry

end profit_share_of_B_l168_168823


namespace solve_gcd_problem_l168_168101

def gcd_problem : Prop :=
  gcd 153 119 = 17

theorem solve_gcd_problem : gcd_problem :=
  by
    sorry

end solve_gcd_problem_l168_168101


namespace jia_can_formulate_quadratic_yi_cannot_formulate_quadratic_bing_cannot_formulate_quadratic_ding_cannot_formulate_quadratic_l168_168522

theorem jia_can_formulate_quadratic :
  ∃ (p q x₁ x₂ : ℤ), 
    x₁ % 3 = 1 ∧ x₂ % 3 = 1 ∧ p % 3 = 1 ∧ q % 3 = 1 ∧ 
    (p = -(x₁ + x₂)) ∧ (q = x₁ * x₂) :=
  sorry

theorem yi_cannot_formulate_quadratic :
  ¬ ∃ (p q x₁ x₂ : ℤ), 
    x₁ % 3 = 2 ∧ x₂ % 3 = 2 ∧ p % 3 = 2 ∧ q % 3 = 2 ∧ 
    (p = -(x₁ + x₂)) ∧ (q = x₁ * x₂) :=
  sorry

theorem bing_cannot_formulate_quadratic :
  ¬ ∃ (p q x₁ x₂ : ℤ), 
    x₁ % 3 = 2 ∧ x₂ % 3 = 2 ∧ p % 3 = 1 ∧ q % 3 = 1 ∧ 
    (p = -(x₁ + x₂)) ∧ (q = x₁ * x₂) :=
  sorry

theorem ding_cannot_formulate_quadratic :
  ¬ ∃ (p q x₁ x₂ : ℤ), 
    x₁ % 3 = 1 ∧ x₂ % 3 = 1 ∧ p % 3 = 2 ∧ q % 3 = 2 ∧ 
    (p = -(x₁ + x₂)) ∧ (q = x₁ * x₂) :=
  sorry

end jia_can_formulate_quadratic_yi_cannot_formulate_quadratic_bing_cannot_formulate_quadratic_ding_cannot_formulate_quadratic_l168_168522


namespace minimize_dot_product_l168_168944

def vector := ℝ × ℝ

def OA : vector := (2, 2)
def OB : vector := (4, 1)

def dot_product (u v : vector) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def AP (P : vector) : vector :=
  (P.1 - OA.1, P.2 - OA.2)

def BP (P : vector) : vector :=
  (P.1 - OB.1, P.2 - OB.2)

def is_on_x_axis (P : vector) : Prop :=
  P.2 = 0

theorem minimize_dot_product :
  ∃ (P : vector), is_on_x_axis P ∧ dot_product (AP P) (BP P) = ( (P.1 - 3) ^ 2 + 1) ∧ P = (3, 0) :=
by
  sorry

end minimize_dot_product_l168_168944


namespace geometric_sequence_sum_5_l168_168523

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ i j : ℕ, ∃ r : ℝ, a (i + 1) = a i * r ∧ a (j + 1) = a j * r

theorem geometric_sequence_sum_5
  (a : ℕ → ℝ)
  (h : geometric_sequence a)
  (h_pos : ∀ n : ℕ, a n > 0)
  (h_eq : a 2 * a 6 + 2 * a 4 * a 5 + (a 5) ^ 2 = 25) :
  a 4 + a 5 = 5 := by
  sorry

end geometric_sequence_sum_5_l168_168523


namespace sweet_tray_GCD_l168_168468

/-!
Tim has a bag of 36 orange-flavoured sweets and Peter has a bag of 44 grape-flavoured sweets.
They have to divide up the sweets into small trays with equal number of sweets;
each tray containing either orange-flavoured or grape-flavoured sweets only.
The largest possible number of sweets in each tray without any remainder is 4.
-/

theorem sweet_tray_GCD :
  Nat.gcd 36 44 = 4 :=
by
  sorry

end sweet_tray_GCD_l168_168468


namespace value_of_f_inv_sum_l168_168965

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def f_inv (y : ℝ) : ℝ := sorry

axiom f_inv_is_inverse : ∀ x : ℝ, f (f_inv x) = x ∧ f_inv (f x) = x
axiom f_condition : ∀ x : ℝ, f x + f (-x) = 2

theorem value_of_f_inv_sum (x : ℝ) : f_inv (2008 - x) + f_inv (x - 2006) = 0 :=
sorry

end value_of_f_inv_sum_l168_168965


namespace eq_has_one_integral_root_l168_168601

theorem eq_has_one_integral_root :
  ∀ x : ℝ, (x - (9 / (x - 5)) = 4 - (9 / (x-5))) → x = 4 := by
  intros x h
  sorry

end eq_has_one_integral_root_l168_168601


namespace find_k_l168_168342

theorem find_k (k n m : ℕ) (hk : k > 0) (hn : n > 0) (hm : m > 0) 
  (h : (1 / (n ^ 2 : ℝ) + 1 / (m ^ 2 : ℝ)) = (k : ℝ) / (n ^ 2 + m ^ 2)) : k = 4 :=
sorry

end find_k_l168_168342


namespace min_rows_512_l168_168870

theorem min_rows_512 (n : ℕ) (table : ℕ → ℕ → ℕ) 
  (H : ∀ A (i j : ℕ), i < 10 → j < 10 → i ≠ j → ∃ B, B < n ∧ (table B i ≠ table A i) ∧ (table B j ≠ table A j) ∧ ∀ k, k ≠ i ∧ k ≠ j → table B k = table A k) : 
  n ≥ 512 :=
sorry

end min_rows_512_l168_168870


namespace solve_for_x_l168_168582

-- Step d: Lean 4 statement
theorem solve_for_x : 
  (∃ x : ℚ, (x + 7) / (x - 4) = (x - 5) / (x + 2)) → (∃ x : ℚ, x = 1 / 3) :=
sorry

end solve_for_x_l168_168582


namespace minimum_value_of_quadratic_l168_168675

def quadratic_polynomial (x : ℝ) : ℝ := 2 * x^2 - 16 * x + 22

theorem minimum_value_of_quadratic : ∃ x : ℝ, quadratic_polynomial x = -10 :=
by 
  use 4
  { sorry }

end minimum_value_of_quadratic_l168_168675


namespace ranking_l168_168130

variables (score : string → ℝ)
variables (Hannah Cassie Bridget David : string)

-- Conditions based on the problem statement
axiom Hannah_shows_her_test_to_everyone : ∀ x, x ≠ Hannah → x = Cassie ∨ x = Bridget ∨ x = David
axiom David_shows_his_test_only_to_Bridget : ∀ x, x ≠ Bridget → x ≠ David
axiom Cassie_does_not_show_her_test : ∀ x, x = Hannah ∨ x = Bridget ∨ x = David → x ≠ Cassie

-- Statements based on what Cassie and Bridget claim
axiom Cassie_statement : score Cassie > min (score Hannah) (score Bridget)
axiom Bridget_statement : score David > score Bridget

-- Final ranking to be proved
theorem ranking : score David > score Bridget ∧ score Bridget > score Cassie ∧ score Cassie > score Hannah := sorry

end ranking_l168_168130


namespace mosel_fills_315_boxes_per_week_l168_168170

-- Definitions for the conditions given in the problem.
def hens : ℕ := 270
def eggs_per_hen_per_day : ℕ := 1
def boxes_capacity : ℕ := 6
def days_per_week : ℕ := 7

-- Objective: Prove that the number of boxes filled each week is 315
theorem mosel_fills_315_boxes_per_week :
  let eggs_per_day := hens * eggs_per_hen_per_day
  let boxes_per_day := eggs_per_day / boxes_capacity
  let boxes_per_week := boxes_per_day * days_per_week
  boxes_per_week = 315 := by
  sorry

end mosel_fills_315_boxes_per_week_l168_168170


namespace total_vegetables_l168_168111

theorem total_vegetables (b k r : ℕ) (broccoli_weight_kg : ℝ) (broccoli_weight_g : ℝ) 
  (kohlrabi_mult : ℕ) (radish_mult : ℕ) :
  broccoli_weight_kg = 5 ∧ 
  broccoli_weight_g = 0.25 ∧ 
  kohlrabi_mult = 4 ∧ 
  radish_mult = 3 ∧ 
  b = broccoli_weight_kg / broccoli_weight_g ∧ 
  k = kohlrabi_mult * b ∧ 
  r = radish_mult * k →
  b + k + r = 340 := 
by
  sorry

end total_vegetables_l168_168111


namespace square_of_second_arm_l168_168153

theorem square_of_second_arm (a b c : ℝ) (h₁ : c = a + 2) (h₂ : a^2 + b^2 = c^2) : b^2 = 4 * a + 4 :=
sorry

end square_of_second_arm_l168_168153


namespace candidate_percentage_l168_168836

theorem candidate_percentage (P : ℝ) (h : (P / 100) * 7800 + 2340 = 7800) : P = 70 :=
sorry

end candidate_percentage_l168_168836


namespace selling_price_l168_168800

/-- 
Prove that the selling price (S) of an article with a cost price (C) of 180 sold at a 15% profit (P) is 207.
-/
theorem selling_price (C P S : ℝ) (hC : C = 180) (hP : P = 15) (hS : S = 207) :
  S = C + (P / 100 * C) :=
by
  -- here we rely on sorry to skip the proof details
  sorry

end selling_price_l168_168800


namespace inequality_solution_set_range_of_a_l168_168581

section
variable {x a : ℝ}

def f (x a : ℝ) := |2 * x - 5 * a| + |2 * x + 1|
def g (x : ℝ) := |x - 1| + 3

theorem inequality_solution_set :
  {x : ℝ | |g x| < 8} = {x : ℝ | -4 < x ∧ x < 6} :=
sorry

theorem range_of_a (h : ∀ x₁ : ℝ, ∃ x₂ : ℝ, f x₁ a = g x₂) :
  a ≥ 0.4 ∨ a ≤ -0.8 :=
sorry
end

end inequality_solution_set_range_of_a_l168_168581


namespace least_pos_int_for_multiple_of_5_l168_168184

theorem least_pos_int_for_multiple_of_5 (n : ℕ) (h1 : n = 725) : ∃ x : ℕ, x > 0 ∧ (725 + x) % 5 = 0 ∧ x = 5 :=
by
  sorry

end least_pos_int_for_multiple_of_5_l168_168184


namespace length_of_rod_l168_168309

theorem length_of_rod (w1 w2 l1 l2 : ℝ) (h_uniform : ∀ m n, m * w1 = n * w2) (h1 : w1 = 42.75) (h2 : l1 = 11.25) : 
  l2 = 6 := 
  by
  have wpm := w1 / l1
  have h3 : 22.8 / wpm = l2 := by sorry
  rw [h1, h2] at *
  simp at *
  sorry

end length_of_rod_l168_168309


namespace find_e_l168_168864

-- Define values for a, b, c, d
def a := 2
def b := 3
def c := 4
def d := 5

-- State the problem
theorem find_e (e : ℚ) : a + b + c + d + e = a + (b + (c - (d * e))) → e = -5/6 :=
by
  sorry

end find_e_l168_168864


namespace jade_driving_hours_per_day_l168_168490

variable (Jade Krista : ℕ)
variable (days driving_hours total_hours : ℕ)

theorem jade_driving_hours_per_day :
  (days = 3) →
  (Krista = 6) →
  (total_hours = 42) →
  (total_hours = days * Jade + days * Krista) →
  Jade = 8 :=
by
  intros h_days h_krista h_total_hours h_equation
  sorry

end jade_driving_hours_per_day_l168_168490


namespace necessary_but_not_sufficient_condition_l168_168499

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  ( (2*x - 1)*x = 0 → x = 0 ) ∧ ( x = 0 → (2*x - 1)*x = 0 ) :=
by
  sorry

end necessary_but_not_sufficient_condition_l168_168499


namespace solve_for_x_l168_168738

theorem solve_for_x (x : ℝ) : 5 * x - 3 * x = 405 - 9 * (x + 4) → x = 369 / 11 := by
  sorry

end solve_for_x_l168_168738


namespace minimum_s_value_l168_168828

theorem minimum_s_value (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_cond : 3 * x^2 + 2 * y^2 + z^2 = 1) :
  ∃ (s : ℝ), s = 8 * Real.sqrt 6 ∧ ∀ (x' y' z' : ℝ), (0 < x' ∧ 0 < y' ∧ 0 < z' ∧ 3 * x'^2 + 2 * y'^2 + z'^2 = 1) → 
      s ≤ (1 + z') / (x' * y' * z') :=
sorry

end minimum_s_value_l168_168828


namespace proof_l168_168992

open Set

-- Universal set U
def U : Set ℕ := {x | x ∈ Finset.range 7}

-- Set A
def A : Set ℕ := {1, 3, 5}

-- Set B
def B : Set ℕ := {4, 5, 6}

-- Complement of A in U
def CU (s : Set ℕ) : Set ℕ := U \ s

-- Proof statement
theorem proof : (CU A) ∩ B = {4, 6} :=
by
  sorry

end proof_l168_168992


namespace range_of_c_l168_168803

def P (c : ℝ) : Prop := ∀ x1 x2 : ℝ, x1 < x2 → (c ^ x1) > (c ^ x2)
def q (c : ℝ) : Prop := ∀ x : ℝ, x > (1 / 2) → (2 * c * x - c) > 0

theorem range_of_c (c : ℝ) (h1 : c > 0) (h2 : c ≠ 1)
  (h3 : ¬ (P c ∧ q c)) (h4 : (P c ∨ q c)) :
  (1 / 2) < c ∧ c < 1 :=
by
  sorry

end range_of_c_l168_168803


namespace smallest_integer_cubing_y_eq_350_l168_168048

def y : ℕ := 2^3 * 3^5 * 4^5 * 5^4 * 6^3 * 7^5 * 8^2

theorem smallest_integer_cubing_y_eq_350 : ∃ z : ℕ, z * y = (2^23) * (3^9) * (5^6) * (7^6) → z = 350 :=
by
  sorry

end smallest_integer_cubing_y_eq_350_l168_168048


namespace inequality_solution_set_l168_168196

theorem inequality_solution_set (x : ℝ) :
  (3 * x + 1) / (1 - 2 * x) ≥ 0 ↔ -1 / 3 ≤ x ∧ x < 1 / 2 :=
by
  sorry

end inequality_solution_set_l168_168196


namespace closest_fraction_l168_168807

theorem closest_fraction (won : ℚ) (options : List ℚ) (closest : ℚ) 
  (h_won : won = 25 / 120) 
  (h_options : options = [1 / 4, 1 / 5, 1 / 6, 1 / 7, 1 / 8]) 
  (h_closest : closest = 1 / 5) :
  ∃ x ∈ options, abs (won - x) = abs (won - closest) := 
sorry

end closest_fraction_l168_168807


namespace Gerald_charge_per_chore_l168_168066

noncomputable def charge_per_chore (E SE SP C : ℕ) : ℕ :=
  let total_expenditure := E * SE
  let monthly_saving_goal := total_expenditure / SP
  monthly_saving_goal / C

theorem Gerald_charge_per_chore :
  charge_per_chore 100 4 8 5 = 10 :=
by
  sorry

end Gerald_charge_per_chore_l168_168066


namespace converse_proposition_l168_168238

-- Define the propositions p and q
variables (p q : Prop)

-- State the problem as a theorem
theorem converse_proposition (p q : Prop) : (q → p) ↔ ¬p → ¬q ∧ ¬q → ¬p ∧ (p → q) := 
by 
  sorry

end converse_proposition_l168_168238


namespace four_digit_numbers_count_l168_168521

open Nat

def is_valid_digit (n : ℕ) : Prop :=
  n ≥ 0 ∧ n ≤ 9

def four_diff_digits (a b c d : ℕ) : Prop :=
  is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧ is_valid_digit d ∧ (a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d)

def is_multiple_of_5 (n : ℕ) : Prop :=
  n % 5 = 0

def is_multiple_of_3 (n : ℕ) : Prop :=
  n % 3 = 0

def leading_digit_not_zero (a : ℕ) : Prop :=
  a ≠ 0

def largest_digit_seven (a b c d : ℕ) : Prop :=
  a = 7 ∨ b = 7 ∨ c = 7 ∨ d = 7

theorem four_digit_numbers_count :
  ∃ n, n = 45 ∧
  ∀ (a b c d : ℕ),
    four_diff_digits a b c d ∧
    leading_digit_not_zero a ∧
    is_multiple_of_5 (a * 1000 + b * 100 + c * 10 + d) ∧
    is_multiple_of_3 (a * 1000 + b * 100 + c * 10 + d) ∧
    largest_digit_seven a b c d →
    n = 45 :=
sorry

end four_digit_numbers_count_l168_168521


namespace tank_filling_time_l168_168277

noncomputable def fill_time (R1 R2 R3 : ℚ) : ℚ :=
  1 / (R1 + R2 + R3)

theorem tank_filling_time :
  let R1 := 1 / 18
  let R2 := 1 / 30
  let R3 := -1 / 45
  fill_time R1 R2 R3 = 15 :=
by
  intros
  unfold fill_time
  sorry

end tank_filling_time_l168_168277


namespace find_smallest_positive_angle_l168_168745

noncomputable def sin_deg (x : ℝ) : ℝ := Real.sin (x * Real.pi / 180)
noncomputable def cos_deg (x : ℝ) : ℝ := Real.cos (x * Real.pi / 180)

theorem find_smallest_positive_angle :
  ∃ φ > 0, cos_deg φ = sin_deg 45 + cos_deg 37 - sin_deg 23 - cos_deg 11 ∧ φ = 53 := 
by
  sorry

end find_smallest_positive_angle_l168_168745


namespace trig_identity_l168_168176

theorem trig_identity (α a : ℝ) (h1 : 0 < α) (h2 : α < π / 2)
    (h3 : (Real.tan α) + (1 / (Real.tan α)) = a) : 
    (1 / Real.sin α) + (1 / Real.cos α) = Real.sqrt (a^2 + 2 * a) :=
by
  sorry

end trig_identity_l168_168176


namespace point_on_or_outside_circle_l168_168716

theorem point_on_or_outside_circle (a : ℝ) : 
  let P := (a, 2 - a)
  let r := 2
  let center := (0, 0)
  let distance_square := (P.1 - center.1)^2 + (P.2 - center.2)^2
  distance_square >= r := 
by
  sorry

end point_on_or_outside_circle_l168_168716


namespace average_number_of_problems_per_day_l168_168335

theorem average_number_of_problems_per_day (P D : ℕ) (hP : P = 161) (hD : D = 7) : (P / D) = 23 :=
  by sorry

end average_number_of_problems_per_day_l168_168335


namespace num_true_statements_l168_168728

theorem num_true_statements :
  (if (2 : ℝ) = 2 then (2 : ℝ)^2 - 4 = 0 else false) ∧
  ((∀ (x : ℝ), x^2 - 4 = 0 → x = 2) ∨ (∃ (x : ℝ), x^2 - 4 = 0 ∧ x ≠ 2)) ∧
  ((∀ (x : ℝ), x ≠ 2 → x^2 - 4 ≠ 0) ∨ (∃ (x : ℝ), x ≠ 2 ∧ x^2 - 4 = 0)) ∧
  ((∀ (x : ℝ), x^2 - 4 ≠ 0 → x ≠ 2) ∨ (∃ (x : ℝ), x^2 - 4 ≠ 0 ∧ x = 2)) :=
sorry

end num_true_statements_l168_168728


namespace mutually_exclusive_not_complementary_l168_168471

-- Define the people
inductive Person
| A 
| B 
| C

open Person

-- Define the colors
inductive Color
| Red
| Yellow
| Blue

open Color

-- Event A: Person A gets the Red card
def event_a (assignment: Person → Color) : Prop := assignment A = Red

-- Event B: Person B gets the Red card
def event_b (assignment: Person → Color) : Prop := assignment B = Red

-- Definition of mutually exclusive events
def mutually_exclusive (P Q: Prop): Prop := P → ¬Q

-- Definition of complementary events
def complementary (P Q: Prop): Prop := P ↔ ¬Q

theorem mutually_exclusive_not_complementary :
  ∀ (assignment: Person → Color),
  mutually_exclusive (event_a assignment) (event_b assignment) ∧ ¬complementary (event_a assignment) (event_b assignment) :=
by
  sorry

end mutually_exclusive_not_complementary_l168_168471


namespace bananas_in_each_group_l168_168317

theorem bananas_in_each_group (total_bananas groups : ℕ) (h1 : total_bananas = 392) (h2 : groups = 196) :
    total_bananas / groups = 2 :=
by
  sorry

end bananas_in_each_group_l168_168317


namespace find_abc_l168_168784

theorem find_abc (a b c : ℤ) 
  (h₁ : a^4 - 2 * b^2 = a)
  (h₂ : b^4 - 2 * c^2 = b)
  (h₃ : c^4 - 2 * a^2 = c)
  (h₄ : a + b + c = -3) : 
  a = -1 ∧ b = -1 ∧ c = -1 := 
sorry

end find_abc_l168_168784


namespace find_x_l168_168042

theorem find_x (x : ℝ) (h : 0.40 * x = (1/3) * x + 110) : x = 1650 :=
sorry

end find_x_l168_168042


namespace intersection_correct_l168_168741

variable (A B : Set ℝ)  -- Define variables A and B as sets of real numbers

-- Define set A as {x | -3 ≤ x < 4}
def setA : Set ℝ := {x | -3 ≤ x ∧ x < 4}

-- Define set B as {x | -2 ≤ x ≤ 5}
def setB : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

-- The goal is to prove the intersection of A and B is {x | -2 ≤ x < 4}
theorem intersection_correct : setA ∩ setB = {x : ℝ | -2 ≤ x ∧ x < 4} := sorry

end intersection_correct_l168_168741


namespace ratio_b_a_l168_168400

theorem ratio_b_a (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a ≠ b) (h4 : a + b > 2 * a) (h5 : 2 * a > a) 
  (h6 : a + b > b) (h7 : a + 2 * a = b) : 
  b = a * Real.sqrt 2 :=
by
  sorry

end ratio_b_a_l168_168400


namespace value_of_h_h_2_is_353_l168_168211

def h (x : ℕ) : ℕ := 3 * x^2 - x + 1

theorem value_of_h_h_2_is_353 : h (h 2) = 353 := 
by
  sorry

end value_of_h_h_2_is_353_l168_168211


namespace bottle_caps_left_l168_168934

theorem bottle_caps_left {init_caps given_away_rebecca given_away_michael left_caps : ℝ} 
  (h1 : init_caps = 143.6)
  (h2 : given_away_rebecca = 89.2)
  (h3 : given_away_michael = 16.7)
  (h4 : left_caps = init_caps - (given_away_rebecca + given_away_michael)) :
  left_caps = 37.7 := by
  sorry

end bottle_caps_left_l168_168934


namespace area_of_rectangle_l168_168671

theorem area_of_rectangle
  (a : ℝ) (b : ℝ) (c : ℝ)
  (h_a : a = 16)
  (h_c : c = 17)
  (h_diag : a^2 + b^2 = c^2) :
  abs (a * b - 91.9136) < 0.0001 :=
by
  sorry

end area_of_rectangle_l168_168671


namespace problem_1_1_and_2_problem_1_2_l168_168862

section Sequence

variables (a : ℕ → ℕ)
variable (S : ℕ → ℕ)

-- Conditions
axiom a_1 : a 1 = 3
axiom a_n_recurr : ∀ n ≥ 2, a n = 2 * a (n - 1) + (n - 2)

-- Prove that {a_n + n} is a geometric sequence and find the general term formula for {a_n}
theorem problem_1_1_and_2 :
  (∀ n ≥ 2, (a (n - 1) + (n - 1) ≠ 0)) ∧ ((a 1 + 1) * 2^(n - 1) = a n + n) ∧
  (∀ n, a n = 2^(n + 1) - n) :=
sorry

-- Find the sum of the first n terms, S_n, of the sequence {a_n}
theorem problem_1_2 (n : ℕ) : S n = 2^(n + 2) - 4 - (n^2 + n) / 2 :=
sorry

end Sequence

end problem_1_1_and_2_problem_1_2_l168_168862


namespace Congcong_CO2_emissions_l168_168572

-- Definitions based on conditions
def CO2_emissions (t: ℝ) : ℝ := t * 0.91 -- Condition 1: CO2 emissions calculation

def Congcong_water_usage : ℝ := 6 -- Condition 2: Congcong's water usage (6 tons)

-- Statement we want to prove
theorem Congcong_CO2_emissions : CO2_emissions Congcong_water_usage = 5.46 :=
by 
  sorry

end Congcong_CO2_emissions_l168_168572


namespace new_individuals_weight_l168_168151

variables (W : ℝ) (A B C : ℝ)

-- Conditions
def original_twelve_people_weight : ℝ := W
def weight_leaving_1 : ℝ := 64
def weight_leaving_2 : ℝ := 75
def weight_leaving_3 : ℝ := 81
def average_increase : ℝ := 3.6
def total_weight_increase : ℝ := 12 * average_increase
def weight_leaving_sum : ℝ := weight_leaving_1 + weight_leaving_2 + weight_leaving_3

-- Equation derived from the problem conditions
def new_individuals_weight_sum : ℝ := weight_leaving_sum + total_weight_increase

-- Theorem to prove
theorem new_individuals_weight :
  A + B + C = 263.2 :=
by
  sorry

end new_individuals_weight_l168_168151


namespace range_of_a_l168_168740

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x + 1| + |x - a| ≥ 5) ↔ (a ≤ -6 ∨ a ≥ 4) :=
by
  sorry

end range_of_a_l168_168740


namespace zeros_of_f_l168_168254

def f (x : ℝ) : ℝ := x^3 - 2 * x^2 - x + 2

theorem zeros_of_f : ∀ x, f x = 0 ↔ x = -1 ∨ x = 1 ∨ x = 2 :=
by {
  sorry
}

end zeros_of_f_l168_168254


namespace arithmetic_geom_sequence_ratio_l168_168795

theorem arithmetic_geom_sequence_ratio (a : ℕ → ℝ) (d a1 : ℝ) (h1 : d ≠ 0) 
(h2 : ∀ n, a (n+1) = a n + d)
(h3 : (a 0 + 2 * d)^2 = a 0 * (a 0 + 8 * d)):
  (a 0 + a 2 + a 8) / (a 1 + a 3 + a 9) = 13 / 16 := 
by sorry

end arithmetic_geom_sequence_ratio_l168_168795


namespace max_value_of_sin2A_tan2B_l168_168236

-- Definitions for the trigonometric functions and angles in triangle ABC
variables {A B C : ℝ}

-- Condition: sin^2 A + sin^2 B = sin^2 C - sqrt 2 * sin A * sin B
def condition (A B C : ℝ) : Prop :=
  (Real.sin A) ^ 2 + (Real.sin B) ^ 2 = (Real.sin C) ^ 2 - Real.sqrt 2 * (Real.sin A) * (Real.sin B)

-- Question: Find the maximum value of sin 2A * tan^2 B
noncomputable def target (A B : ℝ) : ℝ :=
  Real.sin (2 * A) * (Real.tan B) ^ 2

-- The proof statement
theorem max_value_of_sin2A_tan2B (h : condition A B C) : ∃ (max_val : ℝ), max_val = 3 - 2 * Real.sqrt 2 ∧ ∀ (x : ℝ), target A x ≤ max_val := 
sorry

end max_value_of_sin2A_tan2B_l168_168236


namespace JoggerDifference_l168_168909

theorem JoggerDifference (tyson_joggers alexander_joggers christopher_joggers : ℕ)
  (h1 : christopher_joggers = 20 * tyson_joggers)
  (h2 : christopher_joggers = 80)
  (h3 : alexander_joggers = tyson_joggers + 22) :
  christopher_joggers - alexander_joggers = 54 := by
  sorry

end JoggerDifference_l168_168909


namespace jeans_to_tshirt_ratio_l168_168856

noncomputable def socks_price := 5
noncomputable def tshirt_price := socks_price + 10
noncomputable def jeans_price := 30

theorem jeans_to_tshirt_ratio :
  jeans_price / tshirt_price = (2 : ℝ) :=
by sorry

end jeans_to_tshirt_ratio_l168_168856


namespace baker_usual_pastries_l168_168813

variable (P : ℕ)

theorem baker_usual_pastries
  (h1 : 2 * 14 + 4 * 25 - (2 * P + 4 * 10) = 48) : P = 20 :=
by
  sorry

end baker_usual_pastries_l168_168813


namespace units_digit_same_units_and_tens_digit_same_l168_168312

theorem units_digit_same (n : ℕ) : 
  (∃ a : ℕ, a ∈ [0, 1, 5, 6] ∧ n % 10 = a ∧ n^2 % 10 = a) := 
sorry

theorem units_and_tens_digit_same (n : ℕ) : 
  n ∈ [0, 1, 25, 76] ↔ (n % 100 = n^2 % 100) := 
sorry

end units_digit_same_units_and_tens_digit_same_l168_168312


namespace weekly_earnings_l168_168264

def phone_repair_cost : ℕ := 11
def laptop_repair_cost : ℕ := 15
def computer_repair_cost : ℕ := 18

def phone_repairs : ℕ := 5
def laptop_repairs : ℕ := 2
def computer_repairs : ℕ := 2

def total_earnings : ℕ := 
  phone_repairs * phone_repair_cost + 
  laptop_repairs * laptop_repair_cost + 
  computer_repairs * computer_repair_cost

theorem weekly_earnings : total_earnings = 121 := by
  sorry

end weekly_earnings_l168_168264


namespace suitable_for_experimental_method_is_meters_run_l168_168025

-- Define the options as a type
inductive ExperimentalOption
| recommending_class_monitor_candidates
| surveying_classmates_birthdays
| meters_run_in_10_seconds
| avian_influenza_occurrences_world

-- Define a function that checks if an option is suitable for the experimental method
def is_suitable_for_experimental_method (option: ExperimentalOption) : Prop :=
  option = ExperimentalOption.meters_run_in_10_seconds

-- The theorem stating which option is suitable for the experimental method
theorem suitable_for_experimental_method_is_meters_run :
  is_suitable_for_experimental_method ExperimentalOption.meters_run_in_10_seconds :=
by
  sorry

end suitable_for_experimental_method_is_meters_run_l168_168025


namespace sum_three_consecutive_divisible_by_three_l168_168194

theorem sum_three_consecutive_divisible_by_three (n : ℤ) : 3 ∣ ((n - 1) + n + (n + 1)) :=
by
  sorry  -- Proof goes here

end sum_three_consecutive_divisible_by_three_l168_168194


namespace Victor_can_carry_7_trays_at_a_time_l168_168280

-- Define the conditions
def trays_from_first_table : Nat := 23
def trays_from_second_table : Nat := 5
def number_of_trips : Nat := 4

-- Define the total number of trays
def total_trays : Nat := trays_from_first_table + trays_from_second_table

-- Prove that the number of trays Victor can carry at a time is 7
theorem Victor_can_carry_7_trays_at_a_time :
  total_trays / number_of_trips = 7 :=
by
  sorry

end Victor_can_carry_7_trays_at_a_time_l168_168280


namespace smallest_solution_l168_168421

noncomputable def equation (x : ℝ) : Prop :=
  (1 / (x - 1)) + (1 / (x - 5)) = 4 / (x - 4)

theorem smallest_solution : 
  ∃ x : ℝ, equation x ∧ x ≠ 1 ∧ x ≠ 5 ∧ x ≠ 4 ∧ x = (5 - Real.sqrt 33) / 2 :=
by
  sorry

end smallest_solution_l168_168421


namespace gcd_lcm_sum_8_12_l168_168148

-- Define the problem statement
theorem gcd_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end gcd_lcm_sum_8_12_l168_168148


namespace max_sum_of_lengths_l168_168060

theorem max_sum_of_lengths (x y : ℕ) (hx : 1 < x) (hy : 1 < y) (hxy : x + 3 * y < 5000) :
  ∃ a b : ℕ, x = 2^a ∧ y = 2^b ∧ a + b = 20 := sorry

end max_sum_of_lengths_l168_168060


namespace base_n_multiple_of_5_l168_168093

-- Define the polynomial f(n)
def f (n : ℕ) : ℕ := 4 + n + 3 * n^2 + 5 * n^3 + n^4 + 4 * n^5

-- The main theorem to be proven
theorem base_n_multiple_of_5 (n : ℕ) (h1 : 2 ≤ n) (h2 : n ≤ 100) : 
  f n % 5 ≠ 0 :=
by sorry

end base_n_multiple_of_5_l168_168093


namespace least_positive_integer_condition_l168_168664

theorem least_positive_integer_condition (n : ℕ) :
  (∀ d ∈ [2, 3, 4, 5, 6, 7, 8, 9, 11], n % d = 1) → n = 10396 := 
by
  sorry

end least_positive_integer_condition_l168_168664


namespace wire_length_approx_is_correct_l168_168804

noncomputable def S : ℝ := 5.999999999999998
noncomputable def L : ℝ := (5 / 2) * S
noncomputable def W : ℝ := S + L

theorem wire_length_approx_is_correct : abs (W - 21) < 1e-16 := by
  sorry

end wire_length_approx_is_correct_l168_168804


namespace smallest_integer_x_l168_168885

theorem smallest_integer_x (x : ℤ) (h : x < 3 * x - 12) : x ≥ 7 :=
sorry

end smallest_integer_x_l168_168885


namespace red_lights_l168_168646

theorem red_lights (total_lights yellow_lights blue_lights red_lights : ℕ)
  (h1 : total_lights = 95)
  (h2 : yellow_lights = 37)
  (h3 : blue_lights = 32)
  (h4 : red_lights = total_lights - (yellow_lights + blue_lights)) :
  red_lights = 26 := by
  sorry

end red_lights_l168_168646


namespace find_n_l168_168406

theorem find_n (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 10) (h3 : n % 11 = 99999 % 11) : n = 9 :=
sorry

end find_n_l168_168406


namespace line_circle_no_intersection_l168_168382

theorem line_circle_no_intersection : 
  ¬ ∃ (x y : ℝ), 3 * x + 4 * y = 12 ∧ x^2 + y^2 = 4 := by
  sorry

end line_circle_no_intersection_l168_168382


namespace swimmer_path_min_time_l168_168278

theorem swimmer_path_min_time (k : ℝ) :
  (k > Real.sqrt 2 → ∀ x y : ℝ, x = 0 ∧ y = 0 ∧ t = 2/k) ∧
  (k < Real.sqrt 2 → x = 1 ∧ y = 1 ∧ t = Real.sqrt 2) ∧
  (k = Real.sqrt 2 → ∀ x y : ℝ, x = y ∧ t = (1 / Real.sqrt 2) + Real.sqrt 2 + (1 / Real.sqrt 2)) :=
by sorry

end swimmer_path_min_time_l168_168278


namespace find_x_l168_168695

def vector := (ℝ × ℝ)

def collinear (u v : vector) : Prop :=
  ∃ k : ℝ, u = (k * v.1, k * v.2)

def a : vector := (1, 2)
def b (x : ℝ) : vector := (x, 1)
def a_minus_b (x : ℝ) : vector := ((1 - x), 1)

theorem find_x (x : ℝ) (h : collinear a (a_minus_b x)) : x = 1/2 :=
by
  sorry

end find_x_l168_168695


namespace max_radius_approx_l168_168266

open Real

def angle_constraint (θ : ℝ) : Prop :=
  π / 4 ≤ θ ∧ θ ≤ 3 * π / 4

def wire_constraint (r θ : ℝ) : Prop :=
  16 = r * (2 + θ)

noncomputable def max_radius (θ : ℝ) : ℝ :=
  16 / (2 + θ)

theorem max_radius_approx :
  ∃ r θ, angle_constraint θ ∧ wire_constraint r θ ∧ abs (r - 3.673) < 0.001 :=
by
  sorry

end max_radius_approx_l168_168266


namespace product_of_terms_geometric_sequence_l168_168752

variable {a : ℕ → ℝ}
variable {q : ℝ}
noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem product_of_terms_geometric_sequence
  (ha: geometric_sequence a q)
  (h3_4: a 3 * a 4 = 6) :
  a 2 * a 5 = 6 :=
by
  sorry

end product_of_terms_geometric_sequence_l168_168752


namespace seq_sum_11_l168_168749

noncomputable def S (n : ℕ) : ℕ := sorry

noncomputable def a (n : ℕ) : ℕ := sorry

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

theorem seq_sum_11 :
  (∀ n : ℕ, S n = (n * (a 1 + a n)) / 2) ∧
  (is_arithmetic_sequence a) ∧
  (3 * (a 2 + a 4) + 2 * (a 6 + a 9 + a 12) = 12) →
  S 11 = 11 :=
by
  sorry

end seq_sum_11_l168_168749


namespace percentage_increase_l168_168002

variable (P N N' : ℝ)
variable (h : P * 0.90 * N' = P * N * 1.035)

theorem percentage_increase :
  ((N' - N) / N) * 100 = 15 :=
by
  -- By given condition, we have the equation:
  -- P * 0.90 * N' = P * N * 1.035
  sorry

end percentage_increase_l168_168002


namespace min_frac_a_n_over_n_l168_168857

open Nat

def a : ℕ → ℕ
| 0     => 60
| (n+1) => a n + 2 * n

theorem min_frac_a_n_over_n : ∃ n : ℕ, n > 0 ∧ (a n / n = (29 / 2) ∧ ∀ m : ℕ, m > 0 → a m / m ≥ (29 / 2)) :=
by
  sorry

end min_frac_a_n_over_n_l168_168857


namespace angle_sum_x_y_l168_168902

def angle_A := 36
def angle_B := 80
def angle_C := 24

def target_sum : ℕ := 140

theorem angle_sum_x_y (angle_A angle_B angle_C : ℕ) (x y : ℕ) : 
  angle_A = 36 → angle_B = 80 → angle_C = 24 → x + y = 140 := by 
  intros _ _ _
  sorry

end angle_sum_x_y_l168_168902


namespace problem_statement_l168_168012

noncomputable def g : ℝ → ℝ := sorry

theorem problem_statement 
  (h : ∀ x y : ℝ, g (g x + y) = g (x + y) + x * g y - x * y^2 - x + 2) :
  ∃ (m t : ℕ), (m = 1) ∧ (t = 3) ∧ (m * t = 3) :=
sorry

end problem_statement_l168_168012


namespace one_serving_weight_l168_168173

-- Outline the main variables
def chicken_weight_pounds : ℝ := 4.5
def stuffing_weight_ounces : ℝ := 24
def num_servings : ℝ := 12
def conversion_factor : ℝ := 16 -- 1 pound = 16 ounces

-- Define the weights in ounces
def chicken_weight_ounces : ℝ := chicken_weight_pounds * conversion_factor

-- Total weight in ounces for all servings
def total_weight_ounces : ℝ := chicken_weight_ounces + stuffing_weight_ounces

-- Prove one serving weight in ounces
theorem one_serving_weight : total_weight_ounces / num_servings = 8 := by
  sorry

end one_serving_weight_l168_168173


namespace find_y_l168_168429

theorem find_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (hrem : x % y = 11.52) (hdiv : x / y = 96.12) : y = 96 := 
sorry

end find_y_l168_168429


namespace number_of_rowers_l168_168935

theorem number_of_rowers (total_coaches : ℕ) (votes_per_coach : ℕ) (votes_per_rower : ℕ) 
  (htotal_coaches : total_coaches = 36) (hvotes_per_coach : votes_per_coach = 5) 
  (hvotes_per_rower : votes_per_rower = 3) : 
  (total_coaches * votes_per_coach) / votes_per_rower = 60 :=
by 
  sorry

end number_of_rowers_l168_168935


namespace smallest_six_digit_divisible_by_111_l168_168927

theorem smallest_six_digit_divisible_by_111 : ∃ n : ℕ, 100000 ≤ n ∧ n < 1000000 ∧ n % 111 = 0 ∧ n = 100011 :=
by {
  sorry
}

end smallest_six_digit_divisible_by_111_l168_168927


namespace find_a_range_l168_168614

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 - a*x - 5 else (a + 1) / x

theorem find_a_range (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) → 
  - (7 / 2) ≤ a ∧ a ≤ -2 :=
by
  sorry

end find_a_range_l168_168614


namespace measure_of_angle_A_l168_168450

theorem measure_of_angle_A
    (A B : ℝ)
    (h1 : A + B = 90)
    (h2 : A = 3 * B) :
    A = 67.5 :=
by
  sorry

end measure_of_angle_A_l168_168450


namespace cost_of_small_bonsai_l168_168514

variable (cost_small_bonsai cost_big_bonsai : ℝ)

theorem cost_of_small_bonsai : 
  cost_big_bonsai = 20 → 
  3 * cost_small_bonsai + 5 * cost_big_bonsai = 190 → 
  cost_small_bonsai = 30 := 
by
  intros h1 h2 
  sorry

end cost_of_small_bonsai_l168_168514


namespace num_positive_int_values_l168_168387

theorem num_positive_int_values (N : ℕ) :
  (∃ m : ℕ, N = m ∧ m > 0 ∧ 48 % (m + 3) = 0) ↔ (N < 7) :=
sorry

end num_positive_int_values_l168_168387


namespace ineq_medians_triangle_l168_168890

theorem ineq_medians_triangle (a b c s_a s_b s_c : ℝ)
  (h_mediana : s_a = 1 / 2 * Real.sqrt (2 * b^2 + 2 * c^2 - a^2))
  (h_medianb : s_b = 1 / 2 * Real.sqrt (2 * a^2 + 2 * c^2 - b^2))
  (h_medianc : s_c = 1 / 2 * Real.sqrt (2 * a^2 + 2 * b^2 - c^2))
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  a + b + c > s_a + s_b + s_c ∧ s_a + s_b + s_c > (3 / 4) * (a + b + c) := 
sorry

end ineq_medians_triangle_l168_168890


namespace otimes_example_l168_168124

def otimes (a b : ℤ) : ℤ := a^2 - a * b

theorem otimes_example : otimes 4 (otimes 2 (-5)) = -40 := by
  sorry

end otimes_example_l168_168124


namespace factorization_of_x4_plus_81_l168_168098

theorem factorization_of_x4_plus_81 :
  ∀ x : ℝ, x^4 + 81 = (x^2 - 3 * x + 4.5) * (x^2 + 3 * x + 4.5) :=
by
  intros x
  sorry

end factorization_of_x4_plus_81_l168_168098


namespace room_breadth_is_five_l168_168334

theorem room_breadth_is_five 
  (length : ℝ)
  (height : ℝ)
  (bricks_per_square_meter : ℝ)
  (total_bricks : ℝ)
  (H_length : length = 4)
  (H_height : height = 2)
  (H_bricks_per_square_meter : bricks_per_square_meter = 17)
  (H_total_bricks : total_bricks = 340) 
  : ∃ (breadth : ℝ), breadth = 5 :=
by
  -- we leave the proof as sorry for now
  sorry

end room_breadth_is_five_l168_168334


namespace average_of_last_three_numbers_l168_168707

theorem average_of_last_three_numbers (nums : List ℝ) (h_len : nums.length = 6) 
    (h_avg6 : nums.sum / 6 = 60) (h_avg3 : (nums.take 3).sum / 3 = 55) : 
    ((nums.drop 3).sum) / 3 = 65 := 
sorry

end average_of_last_three_numbers_l168_168707


namespace smallest_lcm_4_digit_integers_l168_168195

theorem smallest_lcm_4_digit_integers (k l : ℕ) (h1 : 1000 ≤ k ∧ k ≤ 9999) (h2 : 1000 ≤ l ∧ l ≤ 9999) (h3 : Nat.gcd k l = 11) : Nat.lcm k l = 92092 :=
by
  sorry

end smallest_lcm_4_digit_integers_l168_168195


namespace percentage_of_360_l168_168053

theorem percentage_of_360 (percentage : ℝ) : 
  (percentage / 100) * 360 = 93.6 → percentage = 26 := 
by
  intro h
  -- proof missing
  sorry

end percentage_of_360_l168_168053


namespace no_solutions_triples_l168_168089

theorem no_solutions_triples (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a! + b^3 ≠ 18 + c^3 :=
by
  sorry

end no_solutions_triples_l168_168089


namespace find_triplets_l168_168590

theorem find_triplets (a b p : ℕ) (ha : 0 < a) (hb : 0 < b) (hp : Nat.Prime p) (h_eq : (a + b)^p = p^a + p^b) : (a = 1 ∧ b = 1 ∧ p = 2) :=
by
  sorry

end find_triplets_l168_168590


namespace senya_mistakes_in_OCTAHEDRON_l168_168568

noncomputable def mistakes_in_word (word : String) : Nat :=
  if word = "TETRAHEDRON" then 5
  else if word = "DODECAHEDRON" then 6
  else if word = "ICOSAHEDRON" then 7
  else if word = "OCTAHEDRON" then 5 
  else 0

theorem senya_mistakes_in_OCTAHEDRON : mistakes_in_word "OCTAHEDRON" = 5 := by
  sorry

end senya_mistakes_in_OCTAHEDRON_l168_168568


namespace packed_lunch_needs_l168_168987

-- Definitions based on conditions
def students_A : ℕ := 10
def students_B : ℕ := 15
def students_C : ℕ := 20

def total_students : ℕ := students_A + students_B + students_C

def slices_per_sandwich : ℕ := 4
def sandwiches_per_student : ℕ := 2
def bread_slices_per_student : ℕ := sandwiches_per_student * slices_per_sandwich
def total_bread_slices : ℕ := total_students * bread_slices_per_student

def bags_of_chips_per_student : ℕ := 1
def total_bags_of_chips : ℕ := total_students * bags_of_chips_per_student

def apples_per_student : ℕ := 3
def total_apples : ℕ := total_students * apples_per_student

def granola_bars_per_student : ℕ := 1
def total_granola_bars : ℕ := total_students * granola_bars_per_student

-- Proof goals
theorem packed_lunch_needs :
  total_bread_slices = 360 ∧
  total_bags_of_chips = 45 ∧
  total_apples = 135 ∧
  total_granola_bars = 45 :=
by
  sorry

end packed_lunch_needs_l168_168987


namespace solution_set_of_inequality_l168_168699

theorem solution_set_of_inequality : {x : ℝ | -2 < x ∧ x < 1} = {x : ℝ | -x^2 - x + 2 > 0} :=
by
  sorry

end solution_set_of_inequality_l168_168699


namespace find_slant_height_l168_168894

-- Definitions of the given conditions
variable (r1 r2 L A1 A2 : ℝ)
variable (π : ℝ := Real.pi)

-- The conditions as given in the problem
def conditions : Prop := 
  r1 = 3 ∧ r2 = 4 ∧ 
  (π * L * (r1 + r2) = A1 + A2) ∧ 
  (A1 = π * r1^2) ∧ 
  (A2 = π * r2^2)

-- The theorem stating the question and the correct answer
theorem find_slant_height (h : conditions r1 r2 L A1 A2) : 
  L = 5 := 
sorry

end find_slant_height_l168_168894


namespace fencing_cost_l168_168637

theorem fencing_cost (w : ℝ) (h : ℝ) (p : ℝ) (cost_per_meter : ℝ) 
  (hw : h = w + 10) (perimeter : p = 220) (cost_rate : cost_per_meter = 6.5) : 
  ((p * cost_per_meter) = 1430) := by 
  sorry

end fencing_cost_l168_168637


namespace proposition_B_l168_168583

-- Definitions of the conditions
def line (α : Type) := α
def plane (α : Type) := α
def is_within {α : Type} (a : line α) (p : plane α) : Prop := sorry
def is_perpendicular {α : Type} (a : line α) (p : plane α) : Prop := sorry
def planes_are_perpendicular {α : Type} (p₁ p₂ : plane α) : Prop := sorry
def is_prism (poly : Type) : Prop := sorry

-- Propositions
def p {α : Type} (a : line α) (α₁ α₂ : plane α) : Prop :=
  is_within a α₁ ∧ is_perpendicular a α₂ → planes_are_perpendicular α₁ α₂

def q (poly : Type) : Prop := 
  (∃ (face1 face2 : poly), face1 ≠ face2 ∧ sorry) ∧ sorry

-- Proposition B
theorem proposition_B {α : Type} (a : line α) (α₁ α₂ : plane α) (poly : Type) :
  (p a α₁ α₂) ∧ ¬(q poly) :=
by {
  -- Skipping proof
  sorry
}

end proposition_B_l168_168583


namespace find_repair_charge_l168_168011

theorem find_repair_charge
    (cost_oil_change : ℕ)
    (cost_car_wash : ℕ)
    (num_oil_changes : ℕ)
    (num_repairs : ℕ)
    (num_car_washes : ℕ)
    (total_earnings : ℕ)
    (R : ℕ) :
    (cost_oil_change = 20) →
    (cost_car_wash = 5) →
    (num_oil_changes = 5) →
    (num_repairs = 10) →
    (num_car_washes = 15) →
    (total_earnings = 475) →
    5 * cost_oil_change + 10 * R + 15 * cost_car_wash = total_earnings →
    R = 30 :=
by
    intros h1 h2 h3 h4 h5 h6 h7
    sorry

end find_repair_charge_l168_168011


namespace baseball_card_decrease_l168_168851

theorem baseball_card_decrease (x : ℝ) :
  (0 < x) ∧ (x < 100) ∧ (100 - x) * 0.9 = 45 → x = 50 :=
by
  intros h
  sorry

end baseball_card_decrease_l168_168851


namespace calc_expression_l168_168446

theorem calc_expression (x y z : ℚ) (h1 : x = 1 / 3) (h2 : y = 2 / 3) (h3 : z = x * y) :
  3 * x^2 * y^5 * z^3 = 768 / 1594323 :=
by
  sorry

end calc_expression_l168_168446


namespace length_PR_in_triangle_l168_168681

/-- In any triangle PQR, given:
  PQ = 7, QR = 10, median PS = 5,
  the length of PR must be sqrt(149). -/
theorem length_PR_in_triangle (PQ QR PS : ℝ) (PQ_eq : PQ = 7) (QR_eq : QR = 10) (PS_eq : PS = 5) : 
  ∃ (PR : ℝ), PR = Real.sqrt 149 := 
sorry

end length_PR_in_triangle_l168_168681


namespace total_share_proof_l168_168722

variable (P R : ℕ) -- Parker's share and Richie's share
variable (total_share : ℕ) -- Total share

-- Define conditions
def ratio_condition : Prop := (P : ℕ) / 2 = (R : ℕ) / 3
def parker_share_condition : Prop := P = 50

-- Prove the total share is 125
theorem total_share_proof 
  (h1 : ratio_condition P R)
  (h2 : parker_share_condition P) : 
  total_share = 125 :=
sorry

end total_share_proof_l168_168722


namespace num_and_sum_of_divisors_of_36_l168_168887

noncomputable def num_divisors_and_sum (n : ℕ) : ℕ × ℕ :=
  let divisors := (List.range (n + 1)).filter (λ x => n % x = 0)
  (divisors.length, divisors.sum)

theorem num_and_sum_of_divisors_of_36 : num_divisors_and_sum 36 = (9, 91) := by
  sorry

end num_and_sum_of_divisors_of_36_l168_168887


namespace final_price_after_increase_and_decrease_l168_168095

variable (P : ℝ)

theorem final_price_after_increase_and_decrease (h : P > 0) : 
  let increased_price := P * 1.15
  let final_price := increased_price * 0.85
  final_price = P * 0.9775 :=
by
  sorry

end final_price_after_increase_and_decrease_l168_168095


namespace cube_surface_area_l168_168413

theorem cube_surface_area (V : ℝ) (hV : V = 64) : ∃ S : ℝ, S = 96 := 
by
  sorry

end cube_surface_area_l168_168413


namespace number_divisible_by_5_l168_168551

theorem number_divisible_by_5 (A B C : ℕ) :
  (∃ (k1 k2 k3 k4 k5 k6 : ℕ), 3*10^6 + 10^5 + 7*10^4 + A*10^3 + B*10^2 + 4*10 + C = k1 ∧ 5 * k1 = 0 ∧
                          5 * k2 + 10 = 5 * k2 ∧ 5 * k3 + 5 = 5 * k3 ∧ 
                          5 * k4 + 3 = 5 * k4 ∧ 5 * k5 + 1 = 5 * k5 ∧ 
                          5 * k6 + 7 = 5 * k6) → C = 5 :=
by
  sorry

end number_divisible_by_5_l168_168551


namespace calculate_result_l168_168501

theorem calculate_result : (-3 : ℝ)^(2022) * (1 / 3 : ℝ)^(2023) = 1 / 3 := 
by sorry

end calculate_result_l168_168501


namespace prime_k_for_equiangular_polygons_l168_168990

-- Definitions for conditions in Lean 4
def is_equiangular_polygon (n : ℕ) (angle : ℕ) : Prop :=
  angle = 180 - 360 / n

def is_prime (k : ℕ) : Prop :=
  Nat.Prime k

def valid_angle (x : ℕ) (k : ℕ) : Prop :=
  x < 180 / k

-- The main statement
theorem prime_k_for_equiangular_polygons (n1 n2 x k : ℕ) :
  is_equiangular_polygon n1 x →
  is_equiangular_polygon n2 (k * x) →
  1 < k →
  is_prime k →
  k = 3 :=
by sorry -- proof is not required

end prime_k_for_equiangular_polygons_l168_168990


namespace wendy_packages_chocolates_l168_168999

variable (packages_per_5min : Nat := 2)
variable (dozen_size : Nat := 12)
variable (minutes_in_hour : Nat := 60)
variable (hours : Nat := 4)

theorem wendy_packages_chocolates (h1 : packages_per_5min = 2) 
                                 (h2 : dozen_size = 12) 
                                 (h3 : minutes_in_hour = 60) 
                                 (h4 : hours = 4) : 
    let chocolates_per_5min := packages_per_5min * dozen_size
    let intervals_per_hour := minutes_in_hour / 5
    let chocolates_per_hour := chocolates_per_5min * intervals_per_hour
    let chocolates_in_4hours := chocolates_per_hour * hours
    chocolates_in_4hours = 1152 := 
by
  let chocolates_per_5min := packages_per_5min * dozen_size
  let intervals_per_hour := minutes_in_hour / 5
  let chocolates_per_hour := chocolates_per_5min * intervals_per_hour
  let chocolates_in_4hours := chocolates_per_hour * hours
  sorry

end wendy_packages_chocolates_l168_168999


namespace range_of_a_l168_168604

theorem range_of_a (a : ℝ) (h : ∀ t : ℝ, 0 < t → t ≤ 2 → (t / (t^2 + 9) ≤ a ∧ a ≤ (t + 2) / t^2)) : 
  (2 / 13) ≤ a ∧ a ≤ 1 :=
sorry

end range_of_a_l168_168604


namespace students_neither_math_nor_physics_l168_168591

theorem students_neither_math_nor_physics :
  let total_students := 150
  let students_math := 80
  let students_physics := 60
  let students_both := 20
  total_students - (students_math - students_both + students_physics - students_both + students_both) = 30 :=
by
  sorry

end students_neither_math_nor_physics_l168_168591


namespace seeds_in_big_garden_is_correct_l168_168150

def total_seeds : ℕ := 41
def small_gardens : ℕ := 3
def seeds_per_small_garden : ℕ := 4

def seeds_in_small_gardens : ℕ := small_gardens * seeds_per_small_garden
def seeds_in_big_garden : ℕ := total_seeds - seeds_in_small_gardens

theorem seeds_in_big_garden_is_correct : seeds_in_big_garden = 29 := by
  -- proof goes here
  sorry

end seeds_in_big_garden_is_correct_l168_168150


namespace hyperbola_equation_l168_168508

theorem hyperbola_equation (a b c : ℝ)
  (ha : a > 0) (hb : b > 0)
  (eccentricity : c = 2 * a)
  (distance_foci_asymptote : b = 1)
  (hyperbola_eq : c^2 = a^2 + b^2) :
  (3 * x^2 - y^2 = 1) :=
by
  sorry

end hyperbola_equation_l168_168508


namespace cats_joined_l168_168084

theorem cats_joined (c : ℕ) (h : 1 + c + 2 * c + 6 * c = 37) : c = 4 :=
sorry

end cats_joined_l168_168084


namespace angle_quadrant_l168_168997

theorem angle_quadrant (α : ℝ) (h1 : Real.sin α > 0) (h2 : Real.cos α < 0) : 
  0 < (π - α) ∧ (π - α) < π  :=
by
  sorry

end angle_quadrant_l168_168997


namespace find_sum_of_numbers_l168_168163

theorem find_sum_of_numbers 
  (a b : ℕ)
  (h₁ : a.gcd b = 5)
  (h₂ : a * b / a.gcd b = 120)
  (h₃ : (1 : ℚ) / a + 1 / b = 0.09166666666666666) :
  a + b = 55 := 
sorry

end find_sum_of_numbers_l168_168163


namespace scientific_notation_of_360_billion_l168_168796

def number_in_scientific_notation (n : ℕ) : String :=
  match n with
  | 360000000000 => "3.6 × 10^11"
  | _ => "Unknown"

theorem scientific_notation_of_360_billion : 
  number_in_scientific_notation 360000000000 = "3.6 × 10^11" :=
by
  -- insert proof steps here
  sorry

end scientific_notation_of_360_billion_l168_168796


namespace algebraic_expression_transformation_l168_168899

theorem algebraic_expression_transformation (a b : ℝ) (h : ∀ x : ℝ, x^2 - 6*x + b = (x - a)^2 - 1) : b - a = 5 :=
by
  sorry

end algebraic_expression_transformation_l168_168899


namespace problem_1_problem_2_l168_168311

open Real

noncomputable def f (x : ℝ) : ℝ := 3^x
noncomputable def g (x : ℝ) : ℝ := log x / log 3

theorem problem_1 : g 4 + g 8 - g (32 / 9) = 2 := 
by
  sorry

theorem problem_2 (x : ℝ) (h : 0 < x ∧ x < 1) : g (x / (1 - x)) < 1 ↔ 0 < x ∧ x < 3 / 4 :=
by
  sorry

end problem_1_problem_2_l168_168311


namespace real_life_distance_between_cities_l168_168383

variable (map_distance : ℕ)
variable (scale : ℕ)

theorem real_life_distance_between_cities (h1 : map_distance = 45) (h2 : scale = 10) :
  map_distance * scale = 450 :=
sorry

end real_life_distance_between_cities_l168_168383


namespace towel_bleach_percentage_decrease_l168_168558

-- Define the problem
theorem towel_bleach_percentage_decrease (L B : ℝ) (x : ℝ) (h_length : 0 < L) (h_breadth : 0 < B) 
  (h1 : 0.64 * L * B = 0.8 * L * (1 - x / 100) * B) :
  x = 20 :=
by
  -- The actual proof is not needed, providing "sorry" as a placeholder for the proof.
  sorry

end towel_bleach_percentage_decrease_l168_168558


namespace sum_of_terms_l168_168003

-- Defining the arithmetic progression
def arithmetic_progression (a d : ℕ) (n : ℕ) : ℕ :=
  a + (n - 1) * d

-- Given conditions
theorem sum_of_terms (a d : ℕ) (h : (a + 3 * d) + (a + 11 * d) = 20) :
  12 * (a + 11 * d) / 2 = 60 :=
by
  sorry

end sum_of_terms_l168_168003


namespace sin_alpha_minus_3pi_l168_168727

theorem sin_alpha_minus_3pi (α : ℝ) (h : Real.sin α = 3/5) : Real.sin (α - 3 * Real.pi) = -3/5 :=
by
  sorry

end sin_alpha_minus_3pi_l168_168727


namespace johns_final_weight_is_200_l168_168167

-- Define the initial weight, percentage of weight loss, and weight gain
def initial_weight : ℝ := 220
def weight_loss_percentage : ℝ := 0.10
def weight_gain : ℝ := 2

-- Define a function to calculate the final weight
def final_weight (initial_weight : ℝ) (weight_loss_percentage : ℝ) (weight_gain : ℝ) : ℝ := 
  let weight_lost := initial_weight * weight_loss_percentage
  let weight_after_loss := initial_weight - weight_lost
  weight_after_loss + weight_gain

-- The proof problem is to show that the final weight is 200 pounds
theorem johns_final_weight_is_200 :
  final_weight initial_weight weight_loss_percentage weight_gain = 200 := 
by
  sorry

end johns_final_weight_is_200_l168_168167


namespace domain_transform_l168_168638

variable (f : ℝ → ℝ)

theorem domain_transform (h : ∀ x, -1 ≤ x ∧ x ≤ 4 → ∃ y, f y = x) :
  ∀ x, 0 ≤ x ∧ x ≤ 5 / 2 → ∃ y, f y = 2 * x - 1 :=
sorry

end domain_transform_l168_168638


namespace fraction_to_decimal_l168_168842

theorem fraction_to_decimal : (5 / 50) = 0.10 := 
by
  sorry

end fraction_to_decimal_l168_168842


namespace rational_sum_zero_cube_nonzero_fifth_power_zero_l168_168393

theorem rational_sum_zero_cube_nonzero_fifth_power_zero
  (a b c : ℚ) 
  (h_sum : a + b + c = 0)
  (h_cube_nonzero : a^3 + b^3 + c^3 ≠ 0) 
  : a^5 + b^5 + c^5 = 0 :=
sorry

end rational_sum_zero_cube_nonzero_fifth_power_zero_l168_168393


namespace probZ_eq_1_4_l168_168610

noncomputable def probX : ℚ := 1/4
noncomputable def probY : ℚ := 1/3
noncomputable def probW : ℚ := 1/6

theorem probZ_eq_1_4 :
  let probZ : ℚ := 1 - (probX + probY + probW)
  probZ = 1/4 :=
by
  sorry

end probZ_eq_1_4_l168_168610


namespace quiz_show_prob_l168_168758

-- Definitions extracted from the problem conditions
def n : ℕ := 4 -- Number of questions
def p_correct : ℚ := 1 / 4 -- Probability of guessing a question correctly
def p_incorrect : ℚ := 3 / 4 -- Probability of guessing a question incorrectly

-- We need to prove that the probability of answering at least 3 out of 4 questions correctly 
-- by guessing randomly is 13/256.
theorem quiz_show_prob :
  (Nat.choose n 3 * (p_correct ^ 3) * (p_incorrect ^ 1) +
   Nat.choose n 4 * (p_correct ^ 4)) = 13 / 256 :=
by sorry

end quiz_show_prob_l168_168758


namespace min_value_one_over_x_plus_one_over_y_l168_168506

theorem min_value_one_over_x_plus_one_over_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x * y = 4) : 
  (1 / x + 1 / y) ≥ 1 :=
by
  sorry -- Proof goes here

end min_value_one_over_x_plus_one_over_y_l168_168506


namespace sum_a_n_eq_2014_l168_168284

def f (n : ℕ) : ℤ :=
  if n % 2 = 1 then (n : ℤ)^2 else - (n : ℤ)^2

def a (n : ℕ) : ℤ :=
  f n + f (n + 1)

theorem sum_a_n_eq_2014 : (Finset.range 2014).sum a = 2014 :=
by
  sorry

end sum_a_n_eq_2014_l168_168284


namespace arrangement_count_l168_168774

def numArrangements : Nat := 15000

theorem arrangement_count (students events : ℕ) (nA nB : ℕ) 
  (A_ne_B : nA ≠ nB) 
  (all_students : students = 7) 
  (all_events : events = 5) 
  (one_event_per_student : ∀ (e : ℕ), e < events → ∃ s, s < students ∧ (∀ (s' : ℕ), s' < students → s' ≠ s → e ≠ s')) :
  numArrangements = 15000 := 
sorry

end arrangement_count_l168_168774


namespace obtuse_triangle_contradiction_l168_168515

theorem obtuse_triangle_contradiction (A B C : ℝ) (h1 : A + B + C = 180) (h2 : 0 < A) (h3 : 0 < B) (h4 : 0 < C) : 
  (A > 90 ∧ B > 90) → false :=
by
  sorry

end obtuse_triangle_contradiction_l168_168515


namespace range_of_t_l168_168532
noncomputable def f (x : ℝ) (t : ℝ) : ℝ := Real.exp (2 * x) - t
noncomputable def g (x : ℝ) (t : ℝ) : ℝ := t * Real.exp x - 1

theorem range_of_t (t : ℝ) :
  (∀ x : ℝ, f x t ≥ g x t) ↔ t ≤ 2 * Real.sqrt 2 - 2 :=
by sorry

end range_of_t_l168_168532


namespace max_min_values_l168_168118

noncomputable def f (x : ℝ) : ℝ := x^3 - 12 * x + 8

theorem max_min_values :
  ∃ x_max x_min : ℝ, x_max ∈ Set.Icc (-3 : ℝ) 3 ∧ x_min ∈ Set.Icc (-3 : ℝ) 3 ∧
    (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x ≤ f x_max) ∧ (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x_min ≤ f x) ∧
    f (-2) = 24 ∧ f 2 = -6 := sorry

end max_min_values_l168_168118


namespace car_travel_time_l168_168272

-- Definitions
def speed : ℝ := 50
def miles_per_gallon : ℝ := 30
def tank_capacity : ℝ := 15
def fraction_used : ℝ := 0.5555555555555556

-- Theorem statement
theorem car_travel_time : (fraction_used * tank_capacity * miles_per_gallon / speed) = 5 :=
sorry

end car_travel_time_l168_168272


namespace cyclist_go_south_speed_l168_168978

noncomputable def speed_of_cyclist_go_south (v : ℝ) : Prop :=
  let north_speed := 10 -- speed of cyclist going north in kmph
  let time := 2 -- time in hours
  let distance := 50 -- distance apart in km
  (north_speed + v) * time = distance

theorem cyclist_go_south_speed (v : ℝ) : speed_of_cyclist_go_south v → v = 15 :=
by
  intro h
  -- Proof part is skipped
  sorry

end cyclist_go_south_speed_l168_168978


namespace find_value_of_expression_l168_168172

theorem find_value_of_expression (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = -1) : 2 * a + 2 * b - 3 * (a * b) = 9 :=
by
  sorry

end find_value_of_expression_l168_168172


namespace lowest_possible_price_l168_168691

-- Definitions based on the provided conditions
def regular_discount_range : Set Real := {x | 0.10 ≤ x ∧ x ≤ 0.30}
def additional_discount : Real := 0.20
def retail_price : Real := 35.00

-- Problem statement transformed into Lean
theorem lowest_possible_price :
  ∃ d ∈ regular_discount_range, (retail_price * (1 - d)) * (1 - additional_discount) = 19.60 :=
by
  sorry

end lowest_possible_price_l168_168691


namespace smallest_three_digit_number_exists_l168_168228

def is_valid_permutation_sum (x y z : ℕ) : Prop :=
  let perms := [100*x + 10*y + z, 100*x + 10*z + y, 100*y + 10*x + z, 100*z + 10*x + y, 100*y + 10*z + x, 100*z + 10*y + x]
  perms.sum = 2220

theorem smallest_three_digit_number_exists : ∃ (x y z : ℕ), x < y ∧ y < z ∧ x + y + z = 10 ∧ is_valid_permutation_sum x y z ∧ 100 * x + 10 * y + z = 127 :=
by {
  -- proof goal and steps would go here if we were to complete the proof
  sorry
}

end smallest_three_digit_number_exists_l168_168228


namespace gerald_remaining_pfennigs_l168_168701

-- Definitions of Gerald's initial money and the costs of items
def farthings : Nat := 54
def groats : Nat := 8
def florins : Nat := 17
def meat_pie_cost : Nat := 120
def sausage_roll_cost : Nat := 75

-- Conversion rates
def farthings_to_pfennigs (f : Nat) : Nat := f / 6
def groats_to_pfennigs (g : Nat) : Nat := g * 4
def florins_to_pfennigs (f : Nat) : Nat := f * 40

-- Total pfennigs Gerald has
def total_pfennigs : Nat :=
  farthings_to_pfennigs farthings + groats_to_pfennigs groats + florins_to_pfennigs florins

-- Total cost of both items
def total_cost : Nat := meat_pie_cost + sausage_roll_cost

-- Gerald's remaining pfennigs after purchase
def remaining_pfennigs : Nat := total_pfennigs - total_cost

theorem gerald_remaining_pfennigs :
  remaining_pfennigs = 526 :=
by
  sorry

end gerald_remaining_pfennigs_l168_168701


namespace total_pens_l168_168024

theorem total_pens (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l168_168024


namespace find_concentration_of_second_mixture_l168_168670

noncomputable def concentration_of_second_mixture (total_volume : ℝ) (final_percent : ℝ) (pure_antifreeze : ℝ) (pure_antifreeze_amount : ℝ) : ℝ :=
  let remaining_volume := total_volume - pure_antifreeze_amount
  let final_pure_amount := final_percent * total_volume
  let required_pure_antifreeze := final_pure_amount - pure_antifreeze
  (required_pure_antifreeze / remaining_volume) * 100

theorem find_concentration_of_second_mixture :
  concentration_of_second_mixture 55 0.20 6.11 6.11 = 10 :=
by
  simp [concentration_of_second_mixture]
  sorry

end find_concentration_of_second_mixture_l168_168670


namespace domain_of_k_l168_168061

noncomputable def domain_of_h := Set.Icc (-10 : ℝ) 6

def h (x : ℝ) : Prop := x ∈ domain_of_h
def k (x : ℝ) : Prop := h (-3 * x + 1)

theorem domain_of_k : ∀ x : ℝ, k x ↔ x ∈ Set.Icc (-5/3) (11/3) :=
by
  intro x
  change (-3 * x + 1 ∈ Set.Icc (-10 : ℝ) 6) ↔ (x ∈ Set.Icc (-5/3 : ℝ) (11/3))
  sorry

end domain_of_k_l168_168061


namespace max_sequence_value_l168_168567

theorem max_sequence_value : 
  ∃ n ∈ (Set.univ : Set ℤ), (∀ m ∈ (Set.univ : Set ℤ), -m^2 + 15 * m + 3 ≤ -n^2 + 15 * n + 3) ∧ (-n^2 + 15 * n + 3 = 59) :=
by
  sorry

end max_sequence_value_l168_168567


namespace total_height_geometric_solid_l168_168189

-- Definitions corresponding to conditions
def radius_cylinder1 : ℝ := 1
def radius_cylinder2 : ℝ := 3
def height_water_surface_figure2 : ℝ := 20
def height_water_surface_figure3 : ℝ := 28

-- The total height of the geometric solid is 29 cm
theorem total_height_geometric_solid :
  ∃ height_total : ℝ,
    (height_water_surface_figure2 + height_total - height_water_surface_figure3) = 29 :=
sorry

end total_height_geometric_solid_l168_168189


namespace largest_of_seven_consecutive_integers_l168_168301

theorem largest_of_seven_consecutive_integers (a : ℕ) (h : a > 0) (sum_eq_77 : (a + (a + 1) + (a + 2) + (a + 3) + (a + 4) + (a + 5) + (a + 6) = 77)) :
  a + 6 = 14 :=
by
  sorry

end largest_of_seven_consecutive_integers_l168_168301


namespace min_value_expression_l168_168494

theorem min_value_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a * b - a - 2 * b = 0) :
  ∃ p : ℝ, p = (a^2/4 - 2/a + b^2 - 1/b) ∧ p = 7 :=
by sorry

end min_value_expression_l168_168494


namespace original_grape_jelly_beans_l168_168620

namespace JellyBeans

-- Definition of the problem conditions
variables (g c : ℕ)
axiom h1 : g = 3 * c
axiom h2 : g - 15 = 5 * (c - 5)

-- Proof goal statement
theorem original_grape_jelly_beans : g = 15 :=
by
  sorry

end JellyBeans

end original_grape_jelly_beans_l168_168620


namespace find_m_l168_168200

theorem find_m (m x1 x2 : ℝ) 
  (h1 : x1 * x1 - 2 * (m + 1) * x1 + m^2 + 2 = 0)
  (h2 : x2 * x2 - 2 * (m + 1) * x2 + m^2 + 2 = 0)
  (h3 : (x1 + 1) * (x2 + 1) = 8) : 
  m = 1 :=
sorry

end find_m_l168_168200


namespace incorrect_operation_l168_168385

variable (a : ℕ)

-- Conditions
def condition1 := 4 * a ^ 2 - a ^ 2 = 3 * a ^ 2
def condition2 := a ^ 3 * a ^ 6 = a ^ 9
def condition3 := (a ^ 2) ^ 3 = a ^ 5
def condition4 := (2 * a ^ 2) ^ 2 = 4 * a ^ 4

-- Theorem to prove
theorem incorrect_operation : (a ^ 2) ^ 3 ≠ a ^ 5 := 
by
  sorry

end incorrect_operation_l168_168385


namespace find_sum_u_v_l168_168370

theorem find_sum_u_v (u v : ℤ) (huv : 0 < v ∧ v < u) (pentagon_area : u^2 + 3 * u * v = 451) : u + v = 21 :=
by 
  sorry

end find_sum_u_v_l168_168370


namespace items_per_friend_l168_168052

theorem items_per_friend (pencils : ℕ) (erasers : ℕ) (friends : ℕ) 
    (pencils_eq : pencils = 35) 
    (erasers_eq : erasers = 5) 
    (friends_eq : friends = 5) : 
    (pencils + erasers) / friends = 8 := 
by
  sorry

end items_per_friend_l168_168052


namespace shaded_area_is_correct_l168_168113

-- Define the basic constants and areas
def grid_length : ℝ := 15
def grid_height : ℝ := 5
def total_grid_area : ℝ := grid_length * grid_height

def large_triangle_base : ℝ := 15
def large_triangle_height : ℝ := 3
def large_triangle_area : ℝ := 0.5 * large_triangle_base * large_triangle_height

def small_triangle_base : ℝ := 3
def small_triangle_height : ℝ := 4
def small_triangle_area : ℝ := 0.5 * small_triangle_base * small_triangle_height

-- Define the total shaded area
def shaded_area : ℝ := total_grid_area - large_triangle_area + small_triangle_area

-- Theorem stating that the shaded area is 58.5 square units
theorem shaded_area_is_correct : shaded_area = 58.5 := 
by 
  -- proof will be provided here
  sorry

end shaded_area_is_correct_l168_168113


namespace find_larger_number_l168_168323

theorem find_larger_number (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 15) : L = 1635 := 
by 
  sorry

end find_larger_number_l168_168323


namespace sufficient_not_necessary_condition_l168_168955

theorem sufficient_not_necessary_condition (x k : ℝ) (p : x ≥ k) (q : (2 - x) / (x + 1) < 0) :
  (∀ x, x ≥ k → ((2 - x) / (x + 1) < 0)) ∧ (∃ x, (2 - x) / (x + 1) < 0 ∧ x < k) → k > 2 := by
  sorry

end sufficient_not_necessary_condition_l168_168955


namespace polynomial_min_k_eq_l168_168074

theorem polynomial_min_k_eq {k : ℝ} :
  (∃ k : ℝ, ∀ x y : ℝ, 9 * x^2 - 12 * k * x * y + (2 * k^2 + 3) * y^2 - 6 * x - 9 * y + 12 >= 0)
  ↔ k = (Real.sqrt 3) / 4 :=
sorry

end polynomial_min_k_eq_l168_168074


namespace student_scores_l168_168826

def weighted_average (math history science geography : ℝ) : ℝ :=
  (math * 0.30) + (history * 0.30) + (science * 0.20) + (geography * 0.20)

theorem student_scores :
  ∀ (math history science geography : ℝ),
    math = 74 →
    history = 81 →
    science = geography + 5 →
    science ≥ 75 →
    weighted_average math history science geography = 80 →
    science = 86.25 ∧ geography = 81.25 :=
by
  intros math history science geography h_math h_history h_science h_min_sci h_avg
  sorry

end student_scores_l168_168826


namespace bart_pages_bought_l168_168712

theorem bart_pages_bought (total_money : ℝ) (price_per_notepad : ℝ) (pages_per_notepad : ℕ)
  (h1 : total_money = 10) (h2 : price_per_notepad = 1.25) (h3 : pages_per_notepad = 60) :
  total_money / price_per_notepad * pages_per_notepad = 480 :=
by
  sorry

end bart_pages_bought_l168_168712


namespace discount_offered_is_5_percent_l168_168051

noncomputable def cost_price : ℝ := 100

noncomputable def selling_price_with_discount : ℝ := cost_price * 1.216

noncomputable def selling_price_without_discount : ℝ := cost_price * 1.28

noncomputable def discount : ℝ := selling_price_without_discount - selling_price_with_discount

noncomputable def discount_percentage : ℝ := (discount / selling_price_without_discount) * 100

theorem discount_offered_is_5_percent : discount_percentage = 5 :=
by 
  sorry

end discount_offered_is_5_percent_l168_168051


namespace sufficient_but_not_necessary_condition_l168_168509

-- The conditions of the problem
variables (a b : ℝ)

-- The proposition to be proved
theorem sufficient_but_not_necessary_condition (h : a + b = 1) : 4 * a * b ≤ 1 :=
by
  sorry

end sufficient_but_not_necessary_condition_l168_168509


namespace cube_volume_l168_168941

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 864) : s^3 = 1728 := 
by
  sorry

end cube_volume_l168_168941


namespace carpet_needed_correct_l168_168013

def length_room : ℕ := 15
def width_room : ℕ := 9
def length_closet : ℕ := 3
def width_closet : ℕ := 2

def area_room : ℕ := length_room * width_room
def area_closet : ℕ := length_closet * width_closet
def area_to_carpet : ℕ := area_room - area_closet
def sq_ft_to_sq_yd (sqft: ℕ) : ℕ := (sqft + 8) / 9  -- Adding 8 to ensure proper rounding up

def carpet_needed : ℕ := sq_ft_to_sq_yd area_to_carpet

theorem carpet_needed_correct :
  carpet_needed = 15 := by
  sorry

end carpet_needed_correct_l168_168013


namespace fraction_Cal_to_Anthony_l168_168268

-- definitions for Mabel, Anthony, Cal, and Jade's transactions
def Mabel_transactions : ℕ := 90
def Anthony_transactions : ℕ := Mabel_transactions + (Mabel_transactions / 10)
def Jade_transactions : ℕ := 85
def Cal_transactions : ℕ := Jade_transactions - 19

-- goal: prove the fraction Cal handled compared to Anthony is 2/3
theorem fraction_Cal_to_Anthony : (Cal_transactions : ℚ) / (Anthony_transactions : ℚ) = 2 / 3 :=
by
  sorry

end fraction_Cal_to_Anthony_l168_168268


namespace total_work_completion_days_l168_168127

theorem total_work_completion_days :
  let Amit_work_rate := 1 / 15
  let Ananthu_work_rate := 1 / 90
  let Chandra_work_rate := 1 / 45

  let Amit_days_worked_alone := 3
  let Ananthu_days_worked_alone := 6
  
  let work_by_Amit := Amit_days_worked_alone * Amit_work_rate
  let work_by_Ananthu := Ananthu_days_worked_alone * Ananthu_work_rate
  
  let initial_work_done := work_by_Amit + work_by_Ananthu
  let remaining_work := 1 - initial_work_done

  let combined_work_rate := Amit_work_rate + Ananthu_work_rate + Chandra_work_rate
  let days_all_worked_together := remaining_work / combined_work_rate

  Amit_days_worked_alone + Ananthu_days_worked_alone + days_all_worked_together = 17 :=
by
  sorry

end total_work_completion_days_l168_168127


namespace arithmetic_sequence_S9_l168_168337

theorem arithmetic_sequence_S9 :
  ∀ {a : ℕ → ℤ} {S : ℕ → ℤ},
  (∀ n : ℕ, S n = (n * (2 * a 1 + (n - 1) * d)) / 2) →
  a 2 = 3 →
  S 4 = 16 →
  S 9 = 81 :=
by
  intro a S h_S h_a2 h_S4
  sorry

end arithmetic_sequence_S9_l168_168337


namespace clara_loses_q_minus_p_l168_168325

def clara_heads_prob : ℚ := 2 / 3
def clara_tails_prob : ℚ := 1 / 3

def ethan_heads_prob : ℚ := 1 / 4
def ethan_tails_prob : ℚ := 3 / 4

def lose_prob_clara : ℚ := clara_heads_prob
def both_tails_prob : ℚ := clara_tails_prob * ethan_tails_prob

noncomputable def total_prob_clara_loses : ℚ :=
  lose_prob_clara + ∑' n : ℕ, (both_tails_prob ^ n) * lose_prob_clara

theorem clara_loses_q_minus_p :
  ∃ (p q : ℕ), Nat.gcd p q = 1 ∧ total_prob_clara_loses = p / q ∧ (q - p = 1) :=
sorry

end clara_loses_q_minus_p_l168_168325


namespace probability_of_white_balls_from_both_boxes_l168_168165

theorem probability_of_white_balls_from_both_boxes :
  let P_white_A := 3 / (3 + 2)
  let P_white_B := 2 / (2 + 3)
  P_white_A * P_white_B = 6 / 25 :=
by
  sorry

end probability_of_white_balls_from_both_boxes_l168_168165


namespace final_selling_price_l168_168232

def actual_price : ℝ := 9941.52
def discount1 : ℝ := 0.20
def discount2 : ℝ := 0.10
def discount3 : ℝ := 0.05

noncomputable def final_price (P : ℝ) : ℝ :=
  P * (1 - discount1) * (1 - discount2) * (1 - discount3)

theorem final_selling_price :
  final_price actual_price = 6800.00 :=
by
  sorry

end final_selling_price_l168_168232


namespace solve_inequality_l168_168054

theorem solve_inequality : 
  {x : ℝ | (1 / (x^2 + 1)) > (4 / x) + (21 / 10)} = {x : ℝ | -2 < x ∧ x < 0} :=
by
  sorry

end solve_inequality_l168_168054


namespace river_current_speed_l168_168203

def motorboat_speed_still_water : ℝ := 20
def distance_between_points : ℝ := 60
def total_trip_time : ℝ := 6.25

theorem river_current_speed : ∃ v_T : ℝ, v_T = 4 ∧ 
  (distance_between_points / (motorboat_speed_still_water + v_T)) + 
  (distance_between_points / (motorboat_speed_still_water - v_T)) = total_trip_time := 
sorry

end river_current_speed_l168_168203


namespace vertex_of_parabola_l168_168527

theorem vertex_of_parabola :
  (∃ (h k : ℤ), ∀ (x : ℝ), y = (x - h)^2 + k) → (h = 2 ∧ k = -3) := by
  sorry

end vertex_of_parabola_l168_168527


namespace find_first_term_of_sequence_l168_168518

theorem find_first_term_of_sequence (a : ℕ → ℝ)
  (h_rec : ∀ n, a (n + 1) = 1 / (1 - a n))
  (h_a8 : a 8 = 2) :
  a 1 = 1 / 2 :=
sorry

end find_first_term_of_sequence_l168_168518


namespace inequality_part_1_inequality_part_2_l168_168433

theorem inequality_part_1 (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 1) :
  (a^2 + b^2) / (2 * c) + (b^2 + c^2) / (2 * a) + (c^2 + a^2) / (2 * b) ≥ 1 := by
sorry

theorem inequality_part_2 (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 1) :
  (a^2 / (b + c)) + (b^2 / (a + c)) + (c^2 / (a + b)) ≥ 1 / 2 := by
sorry

end inequality_part_1_inequality_part_2_l168_168433


namespace original_deck_size_l168_168621

-- Define the conditions
def boys_kept_away (remaining_cards kept_away_cards : ℕ) : Prop :=
  remaining_cards + kept_away_cards = 52

-- Define the problem
theorem original_deck_size (remaining_cards : ℕ) (kept_away_cards := 2) :
  boys_kept_away remaining_cards kept_away_cards → remaining_cards + kept_away_cards = 52 :=
by
  intro h
  exact h

end original_deck_size_l168_168621


namespace sum_of_two_primes_l168_168773

theorem sum_of_two_primes (k : ℕ) (n : ℕ) (h : n = 1 + 10 * k) :
  (n = 1 ∨ ∃ p1 p2 : ℕ, Nat.Prime p1 ∧ Nat.Prime p2 ∧ n = p1 + p2) :=
by
  sorry

end sum_of_two_primes_l168_168773


namespace solution_l168_168159

theorem solution 
  (a b c : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_condition : (a^2 / (b - c)^2) + (b^2 / (c - a)^2) + (c^2 / (a - b)^2) = 0) :
  (a^3 / (b - c)^3) + (b^3 / (c - a)^3) + (c^3 / (a - b)^3) = 0 := 
sorry 

end solution_l168_168159


namespace sum_of_squares_correct_l168_168479

-- Define the three incorrect entries
def incorrect_entry_1 : Nat := 52
def incorrect_entry_2 : Nat := 81
def incorrect_entry_3 : Nat := 111

-- Define the sum of the squares of these entries
def sum_of_squares : Nat := incorrect_entry_1 ^ 2 + incorrect_entry_2 ^ 2 + incorrect_entry_3 ^ 2

-- State that this sum of squares equals 21586
theorem sum_of_squares_correct : sum_of_squares = 21586 := by
  sorry

end sum_of_squares_correct_l168_168479


namespace determine_weights_of_balls_l168_168632

theorem determine_weights_of_balls (A B C D E m1 m2 m3 m4 m5 m6 m7 m8 m9 : ℝ)
  (h1 : m1 = A)
  (h2 : m2 = B)
  (h3 : m3 = C)
  (h4 : m4 = A + D)
  (h5 : m5 = A + E)
  (h6 : m6 = B + D)
  (h7 : m7 = B + E)
  (h8 : m8 = C + D)
  (h9 : m9 = C + E) :
  ∃ (A' B' C' D' E' : ℝ), 
    ((A' = A ∨ B' = B ∨ C' = C ∨ D' = D ∨ E' = E) ∧
     (A' ≠ B' ∧ A' ≠ C' ∧ A' ≠ D' ∧ A' ≠ E' ∧
      B' ≠ C' ∧ B' ≠ D' ∧ B' ≠ E' ∧
      C' ≠ D' ∧ C' ≠ E' ∧
      D' ≠ E')) :=
sorry

end determine_weights_of_balls_l168_168632


namespace circle_equation_tangent_to_line_l168_168789

def circle_center : (ℝ × ℝ) := (3, -1)
def tangent_line (x y : ℝ) : Prop := 3 * x + 4 * y = 0

/-- The equation of the circle with center at (3, -1) and tangent to the line 3x + 4y = 0 is (x - 3)^2 + (y + 1)^2 = 1 -/
theorem circle_equation_tangent_to_line : 
  ∃ r, ∀ x y: ℝ, ((x - 3)^2 + (y + 1)^2 = r^2) ∧ (∀ (cx cy: ℝ), cx = 3 → cy = -1 → (tangent_line cx cy → r = 1)) :=
by
  sorry

end circle_equation_tangent_to_line_l168_168789


namespace Jungkook_blue_balls_unchanged_l168_168545

variable (initialRedBalls : ℕ) (initialBlueBalls : ℕ) (initialYellowBalls : ℕ)
variable (newYellowBallGifted: ℕ)

-- Define the initial conditions
def Jungkook_balls := initialRedBalls = 5 ∧ initialBlueBalls = 4 ∧ initialYellowBalls = 3 ∧ newYellowBallGifted = 1

-- State the theorem to prove
theorem Jungkook_blue_balls_unchanged (h : Jungkook_balls initRed initBlue initYellow newYellowGift): initialBlueBalls = 4 := 
by
sorry

end Jungkook_blue_balls_unchanged_l168_168545


namespace class_size_l168_168285

theorem class_size :
  ∃ (N : ℕ), (20 ≤ N) ∧ (N ≤ 30) ∧ (∃ (x : ℕ), N = 3 * x + 1) ∧ (∃ (y : ℕ), N = 4 * y + 1) ∧ (N = 25) :=
by { sorry }

end class_size_l168_168285


namespace intersect_sets_l168_168566

   variable (P : Set ℕ) (Q : Set ℕ)

   -- Definitions based on given conditions
   def P_def : Set ℕ := {1, 3, 5}
   def Q_def : Set ℕ := {x | 2 ≤ x ∧ x ≤ 5}

   -- Theorem statement in Lean 4
   theorem intersect_sets :
     P = P_def → Q = Q_def → P ∩ Q = {3, 5} :=
   by
     sorry
   
end intersect_sets_l168_168566


namespace right_angled_triangle_max_area_l168_168129

theorem right_angled_triangle_max_area (a b : ℝ) (h : a + b = 4) : (1 / 2) * a * b ≤ 2 :=
by 
  sorry

end right_angled_triangle_max_area_l168_168129


namespace michael_has_16_blocks_l168_168005

-- Define the conditions
def number_of_boxes : ℕ := 8
def blocks_per_box : ℕ := 2

-- Define the expected total number of blocks
def total_blocks : ℕ := 16

-- State the theorem
theorem michael_has_16_blocks (n_boxes blocks_per_b : ℕ) :
  n_boxes = number_of_boxes → 
  blocks_per_b = blocks_per_box → 
  n_boxes * blocks_per_b = total_blocks :=
by intros h1 h2; rw [h1, h2]; sorry

end michael_has_16_blocks_l168_168005


namespace problem_solved_by_at_least_one_student_l168_168655

theorem problem_solved_by_at_least_one_student (P_A P_B : ℝ) 
  (hA : P_A = 0.8) 
  (hB : P_B = 0.9) :
  (1 - (1 - P_A) * (1 - P_B) = 0.98) :=
by
  have pAwrong := 1 - P_A
  have pBwrong := 1 - P_B
  have both_wrong := pAwrong * pBwrong
  have one_right := 1 - both_wrong
  sorry

end problem_solved_by_at_least_one_student_l168_168655


namespace madeline_water_intake_l168_168036

-- Declare necessary data and conditions
def bottle_A : ℕ := 8
def bottle_B : ℕ := 12
def bottle_C : ℕ := 16

def goal_yoga : ℕ := 15
def goal_work : ℕ := 35
def goal_jog : ℕ := 20
def goal_evening : ℕ := 30

def intake_yoga : ℕ := 2 * bottle_A
def intake_work : ℕ := 3 * bottle_B
def intake_jog : ℕ := 2 * bottle_C
def intake_evening : ℕ := 2 * bottle_A + 2 * bottle_C

def total_intake : ℕ := intake_yoga + intake_work + intake_jog + intake_evening
def goal_total : ℕ := 100

-- Statement of the proof problem
theorem madeline_water_intake : total_intake = 132 ∧ total_intake - goal_total = 32 :=
by
  -- Calculation parts go here (not needed per instruction)
  sorry

end madeline_water_intake_l168_168036


namespace necessary_but_not_sufficient_l168_168660

theorem necessary_but_not_sufficient (x : ℝ) :
  (x < 2 → (x^2 - x - 2 >= 0) ∨ (x >= -1 ∧ x < 2)) ∧ ((-1 < x ∧ x < 2) → x < 2) :=
by
  sorry

end necessary_but_not_sufficient_l168_168660


namespace avg_cards_removed_until_prime_l168_168563

theorem avg_cards_removed_until_prime:
  let prime_count := 13
  let cards_count := 42
  let non_prime_count := cards_count - prime_count
  let groups_count := prime_count + 1
  let avg_non_prime_per_group := (non_prime_count: ℚ) / (groups_count: ℚ)
  (groups_count: ℚ) > 0 →
  avg_non_prime_per_group + 1 = (43: ℚ) / (14: ℚ) :=
by
  sorry

end avg_cards_removed_until_prime_l168_168563


namespace jason_initial_cards_l168_168896

theorem jason_initial_cards (a : ℕ) (b : ℕ) (x : ℕ) : 
  a = 224 → 
  b = 452 → 
  x = a + b → 
  x = 676 := 
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end jason_initial_cards_l168_168896


namespace find_x_l168_168136

noncomputable def log_base_2 (x : ℝ) : ℝ := Real.log x / Real.log 2
noncomputable def log_base_3 (x : ℝ) : ℝ := Real.log x / Real.log 3
noncomputable def log_base_5 (x : ℝ) : ℝ := Real.log x / Real.log 5
noncomputable def log_base_4 (x : ℝ) : ℝ := Real.log x / Real.log 4

noncomputable def right_triangular_prism_area (x : ℝ) : ℝ := 
  let a := log_base_2 x
  let b := log_base_3 x
  let c := log_base_5 x
  (1/2) * a * b + 3 * c * (a + b + c)

noncomputable def right_triangular_prism_volume (x : ℝ) : ℝ := 
  let a := log_base_2 x
  let b := log_base_3 x
  let c := log_base_5 x
  (1/2) * a * b * c

noncomputable def rectangular_prism_area (x : ℝ) : ℝ := 
  let a := log_base_2 x
  let b := log_base_4 x
  2 * (a * b + a * a + b * a)

noncomputable def rectangular_prism_volume (x : ℝ) : ℝ := 
  let a := log_base_2 x
  let b := log_base_4 x
  a * b * a

theorem find_x (x : ℝ) (h : right_triangular_prism_area x + rectangular_prism_area x = right_triangular_prism_volume x + rectangular_prism_volume x) :
  x = 1152 := by
sorry

end find_x_l168_168136


namespace find_oxygen_weight_l168_168980

-- Definitions of given conditions
def molecular_weight : ℝ := 68
def weight_hydrogen : ℝ := 1
def weight_chlorine : ℝ := 35.5

-- Definition of unknown atomic weight of oxygen
def weight_oxygen : ℝ := 15.75

-- Mathematical statement to prove
theorem find_oxygen_weight :
  weight_hydrogen + weight_chlorine + 2 * weight_oxygen = molecular_weight := by
sorry

end find_oxygen_weight_l168_168980


namespace function_decreasing_interval_l168_168648

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

theorem function_decreasing_interval :
  ∃ I : Set ℝ, I = (Set.Ioo 0 2) ∧ ∀ x ∈ I, deriv f x < 0 :=
by
  sorry

end function_decreasing_interval_l168_168648


namespace smallest_perimeter_is_23_l168_168631

def is_odd_prime (n : ℕ) : Prop := Nat.Prime n ∧ n % 2 = 1

def are_consecutive_odd_primes (a b c : ℕ) : Prop :=
  is_odd_prime a ∧ is_odd_prime b ∧ is_odd_prime c ∧ b = a + 2 ∧ c = b + 2

def satisfies_triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem smallest_perimeter_is_23 : 
  ∃ (a b c : ℕ), are_consecutive_odd_primes a b c ∧ satisfies_triangle_inequality a b c ∧ is_prime (a + b + c) ∧ (a + b + c) = 23 :=
by
  sorry

end smallest_perimeter_is_23_l168_168631


namespace joey_hourly_wage_l168_168809

def sneakers_cost : ℕ := 92
def mowing_earnings (lawns : ℕ) (rate : ℕ) : ℕ := lawns * rate
def selling_earnings (figures : ℕ) (rate : ℕ) : ℕ := figures * rate
def total_additional_earnings (mowing : ℕ) (selling : ℕ) : ℕ := mowing + selling
def remaining_amount (total_cost : ℕ) (earned : ℕ) : ℕ := total_cost - earned
def hourly_wage (remaining : ℕ) (hours : ℕ) : ℕ := remaining / hours

theorem joey_hourly_wage :
  let total_mowing := mowing_earnings 3 8
  let total_selling := selling_earnings 2 9
  let total_earned := total_additional_earnings total_mowing total_selling
  let remaining := remaining_amount sneakers_cost total_earned
  hourly_wage remaining 10 = 5 :=
by
  sorry

end joey_hourly_wage_l168_168809


namespace hyperbola_b_value_l168_168748

theorem hyperbola_b_value (b : ℝ) (h₁ : b > 0) 
  (h₂ : ∃ x y, x^2 - (y^2 / b^2) = 1 ∧ (∀ (c : ℝ), c = Real.sqrt (1 + b^2) → c / 1 = 2)) : b = Real.sqrt 3 :=
by { sorry }

end hyperbola_b_value_l168_168748


namespace probability_at_least_one_hit_l168_168336

variable (P₁ P₂ : ℝ)

theorem probability_at_least_one_hit (h₁ : 0 ≤ P₁ ∧ P₁ ≤ 1) (h₂ : 0 ≤ P₂ ∧ P₂ ≤ 1) :
  1 - (1 - P₁) * (1 - P₂) = P₁ + P₂ - P₁ * P₂ :=
by
  sorry

end probability_at_least_one_hit_l168_168336


namespace rectangle_perimeter_l168_168431

-- Definitions and assumptions
variables (outer_square_area inner_square_area : ℝ) (rectangles_identical : Prop)

-- Given conditions
def outer_square_area_condition : Prop := outer_square_area = 9
def inner_square_area_condition : Prop := inner_square_area = 1
def rectangles_identical_condition : Prop := rectangles_identical

-- The main theorem to prove
theorem rectangle_perimeter (h_outer : outer_square_area_condition outer_square_area)
                            (h_inner : inner_square_area_condition inner_square_area)
                            (h_rectangles : rectangles_identical_condition rectangles_identical) :
  ∃ perimeter : ℝ, perimeter = 6 :=
by
  sorry

end rectangle_perimeter_l168_168431


namespace James_present_age_l168_168797

variable (D J : ℕ)

theorem James_present_age 
  (h1 : D / J = 6 / 5)
  (h2 : D + 4 = 28) :
  J = 20 := 
by
  sorry

end James_present_age_l168_168797


namespace find_A_l168_168613

theorem find_A :
  ∃ A B C : ℝ, 
  (1 : ℝ) / (x^3 - 7 * x^2 + 11 * x + 15) = 
  A / (x - 5) + B / (x + 3) + C / ((x + 3)^2) → 
  A = 1 / 64 := 
by 
  sorry

end find_A_l168_168613


namespace paco_cookies_l168_168963

theorem paco_cookies :
  let initial_cookies := 25
  let ate_cookies := 5
  let remaining_cookies_after_eating := initial_cookies - ate_cookies
  let gave_away_cookies := 4
  let remaining_cookies_after_giving := remaining_cookies_after_eating - gave_away_cookies
  let bought_cookies := 3
  let final_cookies := remaining_cookies_after_giving + bought_cookies
  let combined_bought_and_gave_away := gave_away_cookies + bought_cookies
  (ate_cookies - combined_bought_and_gave_away) = -2 :=
by sorry

end paco_cookies_l168_168963


namespace compare_y_values_l168_168033

theorem compare_y_values :
  let y₁ := 2 / (-2)
  let y₂ := 2 / (-1)
  y₁ > y₂ := by sorry

end compare_y_values_l168_168033


namespace sum_of_cubes_of_consecutive_integers_l168_168871

theorem sum_of_cubes_of_consecutive_integers :
  ∃ (a b c d : ℕ), a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ (a^2 + b^2 + c^2 + d^2 = 9340) ∧ (a^3 + b^3 + c^3 + d^3 = 457064) :=
by
  sorry

end sum_of_cubes_of_consecutive_integers_l168_168871


namespace quadratic_function_expression_rational_function_expression_l168_168452

-- Problem 1:
theorem quadratic_function_expression (f : ℝ → ℝ) :
  (∀ x, f (x + 1) - f x = 3 * x) ∧ (f 0 = 1) → (∀ x, f x = (3 / 2) * x^2 - (3 / 2) * x + 1) :=
by
  sorry

-- Problem 2:
theorem rational_function_expression (f : ℝ → ℝ) : 
  (∀ x, x ≠ 0 → 3 * f (1 / x) + f x = x) → 
  (∀ x, x ≠ 0 → f x = 3 / (8 * x) - x / 8) :=
by
  sorry

end quadratic_function_expression_rational_function_expression_l168_168452


namespace sqrt_fraction_addition_l168_168349

theorem sqrt_fraction_addition :
  (Real.sqrt ((25 : ℝ) / 36 + 16 / 9)) = Real.sqrt 89 / 6 := by
  sorry

end sqrt_fraction_addition_l168_168349


namespace harry_items_left_l168_168961

def sea_stars : ℕ := 34
def seashells : ℕ := 21
def snails : ℕ := 29
def lost_items : ℕ := 25

def total_items : ℕ := sea_stars + seashells + snails
def remaining_items : ℕ := total_items - lost_items

theorem harry_items_left : remaining_items = 59 := by
  -- proof skipped
  sorry

end harry_items_left_l168_168961


namespace evaluate_expression_l168_168639

-- Defining the primary condition
def condition (x : ℝ) : Prop := x > 3

-- Definition of the expression we need to evaluate
def expression (x : ℝ) : ℝ := abs (1 - abs (x - 3))

-- Stating the theorem
theorem evaluate_expression (x : ℝ) (h : condition x) : expression x = abs (4 - x) := 
by 
  -- Since the problem only asks for the statement, the proof is left as sorry.
  sorry

end evaluate_expression_l168_168639


namespace last_digit_x4_plus_inv_x4_l168_168538

theorem last_digit_x4_plus_inv_x4 (x : ℝ) (h : x^2 - 13 * x + 1 = 0) : (x^4 + (1 / x)^4) % 10 = 7 := 
by
  sorry

end last_digit_x4_plus_inv_x4_l168_168538


namespace circle_cartesian_line_circle_intersect_l168_168793

noncomputable def L_parametric (t : ℝ) : ℝ × ℝ :=
  (t, 1 + 2 * t)

noncomputable def C_polar (θ : ℝ) : ℝ :=
  2 * Real.sqrt 2 * Real.sin (θ + Real.pi / 4)

def L_cartesian (x y : ℝ) : Prop :=
  y = 2 * x + 1

def C_cartesian (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 2

theorem circle_cartesian :
  ∀ x y : ℝ, C_polar x = y ↔ C_cartesian x y :=
sorry

theorem line_circle_intersect (x y : ℝ) :
  L_cartesian x y → C_cartesian x y → True :=
sorry

end circle_cartesian_line_circle_intersect_l168_168793


namespace problem_sol_l168_168629

open Complex

theorem problem_sol (a b : ℝ) (i : ℂ) (h : i^2 = -1) (h_eq : (a + i) / i = 1 + b * i) : a + b = 0 :=
sorry

end problem_sol_l168_168629


namespace symmetrical_character_l168_168427

def symmetrical (char : String) : Prop :=
  -- Define a predicate symmetrical which checks if a given character
  -- is a symmetrical figure somehow. This needs to be implemented
  -- properly based on the graphical property of the character.
  sorry 

theorem symmetrical_character :
  ∀ (c : String), (c = "幸" → symmetrical c) ∧ 
                  (c = "福" → ¬ symmetrical c) ∧ 
                  (c = "惠" → ¬ symmetrical c) ∧ 
                  (c = "州" → ¬ symmetrical c) :=
by
  sorry

end symmetrical_character_l168_168427


namespace weight_of_second_new_player_l168_168083

theorem weight_of_second_new_player 
  (total_weight_seven_players : ℕ)
  (average_weight_seven_players : ℕ)
  (total_players_with_new_players : ℕ)
  (average_weight_with_new_players : ℕ)
  (weight_first_new_player : ℕ)
  (W : ℕ) :
  total_weight_seven_players = 7 * average_weight_seven_players →
  total_players_with_new_players = 9 →
  average_weight_with_new_players = 106 →
  weight_first_new_player = 110 →
  (total_weight_seven_players + weight_first_new_player + W) / total_players_with_new_players = average_weight_with_new_players →
  W = 60 := 
by sorry

end weight_of_second_new_player_l168_168083


namespace qin_jiushao_algorithm_correct_operations_l168_168147

def qin_jiushao_algorithm_operations (f : ℝ → ℝ) (x : ℝ) : ℕ × ℕ := sorry

def f (x : ℝ) : ℝ := 4 * x^5 - x^2 + 2
def x : ℝ := 3

theorem qin_jiushao_algorithm_correct_operations :
  qin_jiushao_algorithm_operations f x = (5, 2) :=
sorry

end qin_jiushao_algorithm_correct_operations_l168_168147


namespace final_range_a_l168_168998

open Real

noncomputable def f (a x : ℝ) : ℝ := log x + x^2 - a * x

lemma increasing_function_range_a (a : ℝ) (h : ∀ x : ℝ, x > 0 → deriv (f a) x ≥ 0) :
  a ≤ 2 * sqrt 2 :=
sorry

lemma condition_range_a (a : ℝ) (h1 : ∀ x : ℝ, x > 0 → deriv (f a) x ≥ 0)
  (h2 : ∀ x : ℝ, 0 < x ∧ x ≤ 1 → f a x ≤ 1/2 * (3 * x^2 + 1 / x^2 - 6 * x)) :
  2 ≤ a :=
sorry

theorem final_range_a (a : ℝ)
  (h1 : ∀ x : ℝ, x > 0 → deriv (f a) x ≥ 0)
  (h2 : ∀ x : ℝ, 0 < x ∧ x ≤ 1 → f a x ≤ 1/2 * (3 * x^2 + 1 / x^2 - 6 * x)) :
  2 ≤ a ∧ a ≤ 2 * sqrt 2 :=
sorry

end final_range_a_l168_168998


namespace determine_n_l168_168135

theorem determine_n (x n : ℝ) : 
  (∃ c d : ℝ, G = (c * x + d) ^ 2) ∧ (G = (8 * x^2 + 24 * x + 3 * n) / 8) → n = 6 :=
by {
  sorry
}

end determine_n_l168_168135


namespace odd_function_sum_zero_l168_168071

def odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g x

theorem odd_function_sum_zero (g : ℝ → ℝ) (a : ℝ) (h_odd : odd_function g) : 
  g a + g (-a) = 0 :=
by 
  sorry

end odd_function_sum_zero_l168_168071


namespace taxi_fare_distance_l168_168361

theorem taxi_fare_distance (x : ℕ) (h₁ : 8 + 2 * (x - 3) = 20) : x = 9 :=
by {
  sorry
}

end taxi_fare_distance_l168_168361


namespace find_s_l168_168684

theorem find_s (s : ℝ) (t : ℝ) (h1 : t = 4) (h2 : t = 12 * s^2 + 2 * s) : s = 0.5 ∨ s = -2 / 3 :=
by
  sorry

end find_s_l168_168684


namespace cannot_determine_number_of_pens_l168_168556

theorem cannot_determine_number_of_pens 
  (P : ℚ) -- marked price of one pen
  (N : ℕ) -- number of pens = 46
  (discount : ℚ := 0.01) -- 1% discount
  (profit_percent : ℚ := 11.91304347826087) -- given profit percent
  : ¬ ∃ (N : ℕ), 
        profit_percent = ((N * P * (1 - discount) - N * P) / (N * P)) * 100 :=
by
  sorry

end cannot_determine_number_of_pens_l168_168556


namespace strictly_decreasing_interval_l168_168281

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - Real.log x

theorem strictly_decreasing_interval :
  ∀ x y : ℝ, (0 < x ∧ x < 1) ∧ (0 < y ∧ y < 1) ∧ y < x → f y < f x :=
by
  sorry

end strictly_decreasing_interval_l168_168281


namespace verka_digit_sets_l168_168109

-- Define the main conditions as:
def is_three_digit_number (a b c : ℕ) : Prop :=
  let num1 := 100 * a + 10 * b + c
  let num2 := 100 * a + 10 * c + b
  let num3 := 100 * b + 10 * a + c
  let num4 := 100 * b + 10 * c + a
  let num5 := 100 * c + 10 * a + b
  let num6 := 100 * c + 10 * b + a
  num1 + num2 + num3 + num4 + num5 + num6 = 1221

-- Prove the main theorem
theorem verka_digit_sets :
  ∃ (a b c : ℕ), is_three_digit_number a a c ∧
                 ((a, c) = (1, 9) ∨ (a, c) = (2, 7) ∨ (a, c) = (3, 5) ∨ (a, c) = (4, 3) ∨ (a, c) = (5, 1)) :=
by sorry

end verka_digit_sets_l168_168109


namespace solve_for_x_l168_168411

theorem solve_for_x (x : ℕ) (h1 : x > 0) (h2 : x % 6 = 0) (h3 : x^2 > 144) (h4 : x < 30) : x = 18 ∨ x = 24 :=
by
  sorry

end solve_for_x_l168_168411


namespace angela_action_figures_l168_168205

theorem angela_action_figures (n s r g : ℕ) (hn : n = 24) (hs : s = n * 1 / 4) (hr : r = n - s) (hg : g = r * 1 / 3) :
  r - g = 12 :=
sorry

end angela_action_figures_l168_168205


namespace unique_ordered_pair_satisfies_equation_l168_168843

theorem unique_ordered_pair_satisfies_equation :
  ∃! (m n : ℕ), 0 < m ∧ 0 < n ∧ (6 / m + 3 / n + 1 / (m * n) = 1) :=
by
  sorry

end unique_ordered_pair_satisfies_equation_l168_168843


namespace speed_of_second_train_l168_168132

-- Definitions of conditions
def distance_train1 : ℝ := 200
def speed_train1 : ℝ := 50
def distance_train2 : ℝ := 240
def time_train1_and_train2 : ℝ := 4

-- Statement of the problem
theorem speed_of_second_train : (distance_train2 / time_train1_and_train2) = 60 := by
  sorry

end speed_of_second_train_l168_168132


namespace seq_formula_l168_168143

theorem seq_formula (a : ℕ → ℕ) (h₀ : a 1 = 2) (h₁ : ∀ n : ℕ, 0 < n → a (n + 1) = 2 * a n - 1) :
  ∀ n : ℕ, 0 < n → a n = 2 ^ (n - 1) + 1 := 
by 
  sorry

end seq_formula_l168_168143


namespace cost_of_paving_l168_168375

def length : ℝ := 5.5
def width : ℝ := 4
def rate_per_sq_meter : ℝ := 850

theorem cost_of_paving :
  rate_per_sq_meter * (length * width) = 18700 :=
by
  sorry

end cost_of_paving_l168_168375


namespace final_result_is_8_l168_168792

theorem final_result_is_8 (n : ℕ) (h1 : n = 2976) (h2 : (n / 12) - 240 = 8) : (n / 12) - 240 = 8 :=
by {
  -- Proof steps would go here
  sorry
}

end final_result_is_8_l168_168792


namespace files_remaining_on_flash_drive_l168_168472

def initial_music_files : ℕ := 32
def initial_video_files : ℕ := 96
def deleted_files : ℕ := 60

def total_initial_files : ℕ := initial_music_files + initial_video_files

theorem files_remaining_on_flash_drive 
  (h : total_initial_files = 128) : (total_initial_files - deleted_files) = 68 := by
  sorry

end files_remaining_on_flash_drive_l168_168472


namespace ratio_of_areas_l168_168915

theorem ratio_of_areas (T A B : ℝ) (hT : T = 900) (hB : B = 405) (hSum : A + B = T) :
  (A - B) / ((A + B) / 2) = 1 / 5 :=
by
  sorry

end ratio_of_areas_l168_168915


namespace least_four_digit_with_factors_3_5_7_l168_168838

open Nat

-- Definitions for the conditions
def has_factors (n : ℕ) (factors : List ℕ) : Prop :=
  ∀ f ∈ factors, f ∣ n

-- Main theorem statement
theorem least_four_digit_with_factors_3_5_7
  (n : ℕ) 
  (h1 : 1000 ≤ n) 
  (h2 : n < 10000)
  (h3 : has_factors n [3, 5, 7]) :
  n = 1050 :=
sorry

end least_four_digit_with_factors_3_5_7_l168_168838


namespace proof_problem_l168_168356

def h (x : ℝ) : ℝ := x^2 - 3 * x + 7
def k (x : ℝ) : ℝ := 2 * x + 4

theorem proof_problem : h (k 3) - k (h 3) = 59 := by
  sorry

end proof_problem_l168_168356


namespace bob_corn_calc_l168_168769

noncomputable def bob_corn_left (initial_bushels : ℕ) (ears_per_bushel : ℕ) (bushels_taken_by_terry : ℕ) (bushels_taken_by_jerry : ℕ) (bushels_taken_by_linda : ℕ) (ears_taken_by_stacy : ℕ) : ℕ :=
  let initial_ears := initial_bushels * ears_per_bushel
  let ears_given_away := (bushels_taken_by_terry + bushels_taken_by_jerry + bushels_taken_by_linda) * ears_per_bushel + ears_taken_by_stacy
  initial_ears - ears_given_away

theorem bob_corn_calc :
  bob_corn_left 50 14 8 3 12 21 = 357 :=
by
  sorry

end bob_corn_calc_l168_168769


namespace fractional_eq_solution_range_l168_168219

theorem fractional_eq_solution_range (m : ℝ) : 
  (∃ x : ℝ, (2 * x - m) / (x + 1) = 3 ∧ x > 0) ↔ m < -3 :=
by
  sorry

end fractional_eq_solution_range_l168_168219


namespace max_area_triangle_l168_168564

theorem max_area_triangle (A B C : ℝ) (a b c : ℝ) (h1 : Real.sqrt 2 * Real.sin A = Real.sqrt 3 * Real.cos A) (h2 : a = Real.sqrt 3) :
  ∃ (max_area : ℝ), max_area = (3 * Real.sqrt 3) / (8 * Real.sqrt 5) := 
sorry

end max_area_triangle_l168_168564


namespace sharp_triple_72_l168_168882

-- Definition of the transformation function
def sharp (N : ℝ) : ℝ := 0.4 * N + 3

-- Theorem statement
theorem sharp_triple_72 : sharp (sharp (sharp 72)) = 9.288 := by
  sorry

end sharp_triple_72_l168_168882


namespace find_numbers_between_70_and_80_with_gcd_6_l168_168651

theorem find_numbers_between_70_and_80_with_gcd_6 :
  ∃ n, 70 ≤ n ∧ n ≤ 80 ∧ Nat.gcd n 30 = 6 ∧ (n = 72 ∨ n = 78) :=
by
  sorry

end find_numbers_between_70_and_80_with_gcd_6_l168_168651


namespace ratio_wx_l168_168794

theorem ratio_wx (w x y : ℚ) (h1 : w / y = 3 / 4) (h2 : (x + y) / y = 13 / 4) : w / x = 1 / 3 :=
  sorry

end ratio_wx_l168_168794


namespace rectangle_width_l168_168168

theorem rectangle_width (w l : ℕ) (h1 : l = 2 * w) (h2 : 2 * (w + l) = w * l) : w = 3 :=
by sorry

end rectangle_width_l168_168168


namespace f_properties_l168_168223

noncomputable def f : ℚ × ℚ → ℚ := sorry

theorem f_properties :
  (∀ (x y z : ℚ), f (x*y, z) = f (x, z) * f (y, z)) →
  (∀ (x y z : ℚ), f (z, x*y) = f (z, x) * f (z, y)) →
  (∀ (x : ℚ), f (x, 1 - x) = 1) →
  (∀ (x : ℚ), f (x, x) = 1) ∧
  (∀ (x : ℚ), f (x, -x) = 1) ∧
  (∀ (x y : ℚ), f (x, y) * f (y, x) = 1) :=
by
  intros h1 h2 h3
  -- Proof goes here
  sorry

end f_properties_l168_168223


namespace Kiera_envelopes_l168_168892

theorem Kiera_envelopes (blue yellow green : ℕ) (total_envelopes : ℕ) 
  (cond1 : blue = 14) 
  (cond2 : total_envelopes = 46) 
  (cond3 : green = 3 * yellow) 
  (cond4 : total_envelopes = blue + yellow + green) : yellow = 6 - 8 := 
by sorry

end Kiera_envelopes_l168_168892


namespace cuboid_surface_area_l168_168378

noncomputable def total_surface_area (x y z : ℝ) : ℝ :=
  2 * (x * y + y * z + z * x)

theorem cuboid_surface_area (x y z : ℝ) (h1 : x + y + z = 40) (h2 : x^2 + y^2 + z^2 = 625) :
  total_surface_area x y z = 975 :=
sorry

end cuboid_surface_area_l168_168378


namespace work_problem_l168_168827

theorem work_problem (P Q R W t_q : ℝ) (h1 : P = Q + R) 
    (h2 : (P + Q) * 10 = W) 
    (h3 : R * 35 = W) 
    (h4 : Q * t_q = W) : 
    t_q = 28 := 
by
    sorry

end work_problem_l168_168827


namespace find_a_l168_168229

def f : ℝ → ℝ := sorry

theorem find_a (x a : ℝ) 
  (h1 : ∀ x, f ((1/2)*x - 1) = 2*x - 5)
  (h2 : f a = 6) : 
  a = 7/4 := 
by 
  sorry

end find_a_l168_168229


namespace part_I_part_II_l168_168365

variable {a b : ℝ}

theorem part_I (h1 : a * b ≠ 0) (h2 : a * b > 0) :
  b / a + a / b ≥ 2 :=
sorry

theorem part_II (h1 : a * b ≠ 0) (h3 : a * b < 0) :
  abs (b / a + a / b) ≥ 2 :=
sorry

end part_I_part_II_l168_168365


namespace ratio_of_c_and_d_l168_168608

theorem ratio_of_c_and_d
  (x y c d : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hd : d ≠ 0) 
  (h1 : 8 * x - 6 * y = c)
  (h2 : 9 * y - 12 * x = d) :
  c / d = -2 / 3 := 
  sorry

end ratio_of_c_and_d_l168_168608


namespace tesseract_hyper_volume_l168_168046

theorem tesseract_hyper_volume
  (a b c d : ℝ)
  (h1 : a * b * c = 72)
  (h2 : b * c * d = 75)
  (h3 : c * d * a = 48)
  (h4 : d * a * b = 50) :
  a * b * c * d = 3600 :=
sorry

end tesseract_hyper_volume_l168_168046


namespace people_in_room_eq_33_l168_168075

variable (people chairs : ℕ)

def chairs_empty := 5
def chairs_total := 5 * 5
def chairs_occupied := (4 * chairs_total) / 5
def people_seated := 3 * people / 5

theorem people_in_room_eq_33 : 
    (people_seated = chairs_occupied ∧ chairs_total - chairs_occupied = chairs_empty)
    → people = 33 :=
by
  sorry

end people_in_room_eq_33_l168_168075


namespace find_letters_with_dot_but_no_straight_line_l168_168584

-- Define the problem statement and conditions
def DL : ℕ := 16
def L : ℕ := 30
def Total_letters : ℕ := 50

-- Define the function that calculates the number of letters with a dot but no straight line
def letters_with_dot_but_no_straight_line (DL L Total_letters : ℕ) : ℕ := Total_letters - (L + DL)

-- State the theorem to be proved
theorem find_letters_with_dot_but_no_straight_line : letters_with_dot_but_no_straight_line DL L Total_letters = 4 :=
by
  sorry

end find_letters_with_dot_but_no_straight_line_l168_168584


namespace max_integer_value_l168_168455

theorem max_integer_value (x : ℝ) : 
  ∃ M : ℤ, ∀ y : ℝ, (M = ⌊ 1 + 10 / (4 * y^2 + 12 * y + 9) ⌋ ∧ M ≤ 11) := 
sorry

end max_integer_value_l168_168455


namespace cost_difference_l168_168047

def cost_per_copy_X : ℝ := 1.25
def cost_per_copy_Y : ℝ := 2.75
def num_copies : ℕ := 80

theorem cost_difference :
  num_copies * cost_per_copy_Y - num_copies * cost_per_copy_X = 120 := sorry

end cost_difference_l168_168047


namespace max_f_value_l168_168201

noncomputable def f (x : ℝ) : ℝ := 9 * Real.sin x + 12 * Real.cos x

theorem max_f_value : ∃ x : ℝ, f x = 15 :=
by
  sorry

end max_f_value_l168_168201


namespace sin_cos_identity_l168_168855

theorem sin_cos_identity (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : 
  Real.sin x * Real.cos x = 4 / 17 := by
  sorry

end sin_cos_identity_l168_168855


namespace neg_p_implies_neg_q_l168_168253

variables {x : ℝ}

def condition_p (x : ℝ) : Prop := |x + 1| > 2
def condition_q (x : ℝ) : Prop := 5 * x - 6 > x^2
def neg_p (x : ℝ) : Prop := |x + 1| ≤ 2
def neg_q (x : ℝ) : Prop := x ≤ 2 ∨ x ≥ 3

theorem neg_p_implies_neg_q : (∀ x, neg_p x → neg_q x) :=
by 
  -- Proof is skipped according to the instructions
  sorry

end neg_p_implies_neg_q_l168_168253


namespace find_m_for_given_slope_l168_168410

theorem find_m_for_given_slope (m : ℝ) :
  (∃ (P Q : ℝ × ℝ),
    P = (-2, m) ∧ Q = (m, 4) ∧
    (Q.2 - P.2) / (Q.1 - P.1) = 1) → m = 1 :=
by
  sorry

end find_m_for_given_slope_l168_168410


namespace percentage_decrease_increase_l168_168874

theorem percentage_decrease_increase (x : ℝ) : 
  (1 - x / 100) * (1 + x / 100) = 0.75 ↔ x = 50 :=
by
  sorry

end percentage_decrease_increase_l168_168874


namespace triangle_solution_condition_l168_168778

-- Definitions of segments
variables {A B D E : Type}
variables (c f g : Real)

-- Allow noncomputable definitions for geometric constraints
noncomputable def triangle_construction (c f g : Real) : String :=
  if c > f then "more than one solution"
  else if c = f then "exactly one solution"
  else "no solution"

-- The proof problem statement
theorem triangle_solution_condition (c f g : Real) :
  (c > f → triangle_construction c f g = "more than one solution") ∧
  (c = f → triangle_construction c f g = "exactly one solution") ∧
  (c < f → triangle_construction c f g = "no solution") :=
by
  sorry

end triangle_solution_condition_l168_168778


namespace triangle_area_l168_168198

theorem triangle_area {a b : ℝ} (h₁ : a = 3) (h₂ : b = 4) (h₃ : Real.sin (C : ℝ) = 1/2) :
  let area := (1 / 2) * a * b * (Real.sin C) 
  area = 3 := 
by
  rw [h₁, h₂, h₃]
  simp [Real.sin, mul_assoc]
  sorry

end triangle_area_l168_168198


namespace find_a4_l168_168919

variable {a_n : ℕ → ℕ}
variable {S : ℕ → ℕ}

def is_arithmetic_sequence (a_n : ℕ → ℕ) :=
  ∃ d : ℕ, ∀ n : ℕ, a_n (n + 1) = a_n n + d

def sum_first_n_terms (S : ℕ → ℕ) (a_n : ℕ → ℕ) :=
  ∀ n : ℕ, S n = (n * (a_n 1 + a_n n)) / 2

theorem find_a4 (h : S 7 = 35) (hs : sum_first_n_terms S a_n) (ha : is_arithmetic_sequence a_n) : a_n 4 = 5 := 
  by sorry

end find_a4_l168_168919


namespace binary_sum_to_decimal_l168_168742

theorem binary_sum_to_decimal :
  let bin1 := "1101011"
  let bin2 := "1010110"
  let dec1 := 64 + 32 + 0 + 8 + 0 + 2 + 1 -- decimal value of "1101011"
  let dec2 := 64 + 0 + 16 + 0 + 4 + 2 + 0 -- decimal value of "1010110"
  dec1 + dec2 = 193 := by
  sorry

end binary_sum_to_decimal_l168_168742


namespace no_b_satisfies_143b_square_of_integer_l168_168299

theorem no_b_satisfies_143b_square_of_integer :
  ∀ b : ℤ, b > 4 → ¬ ∃ k : ℤ, b^2 + 4 * b + 3 = k^2 :=
by
  intro b hb
  by_contra h
  obtain ⟨k, hk⟩ := h
  have : b^2 + 4 * b + 3 = k ^ 2 := hk
  sorry

end no_b_satisfies_143b_square_of_integer_l168_168299


namespace fraction_of_dark_tiles_is_correct_l168_168666

def num_tiles_in_block : ℕ := 64
def num_dark_tiles : ℕ := 18
def expected_fraction_dark_tiles : ℚ := 9 / 32

theorem fraction_of_dark_tiles_is_correct :
  (num_dark_tiles : ℚ) / num_tiles_in_block = expected_fraction_dark_tiles := by
sorry

end fraction_of_dark_tiles_is_correct_l168_168666


namespace problem1_problem2_problem3_l168_168805

-- Problem 1
theorem problem1 (A B : Set α) (x : α) : x ∈ A ∪ B → x ∈ A ∨ x ∈ B :=
by sorry

-- Problem 2
theorem problem2 (A B : Set α) (x : α) : x ∈ A ∩ B → x ∈ A ∧ x ∈ B :=
by sorry

-- Problem 3
theorem problem3 (a b : ℝ) : a > 0 ∧ b > 0 → a * b > 0 :=
by sorry

end problem1_problem2_problem3_l168_168805


namespace total_cases_is_8_l168_168690

def num_blue_cards : Nat := 3
def num_yellow_cards : Nat := 5

def total_cases : Nat := num_blue_cards + num_yellow_cards

theorem total_cases_is_8 : total_cases = 8 := by
  sorry

end total_cases_is_8_l168_168690


namespace intersection_complement_l168_168891

def A : Set ℝ := { x | x^2 ≤ 4 * x }
def B : Set ℝ := { x | ∃ y, y = Real.sqrt (x - 3) }

theorem intersection_complement (x : ℝ) : 
  x ∈ A ∩ (Set.univ \ B) ↔ x ∈ Set.Ico 0 3 := 
sorry

end intersection_complement_l168_168891


namespace operation_result_l168_168186

def star (a b c : ℝ) : ℝ := (a + b + c) ^ 2

theorem operation_result (x : ℝ) : star (x - 1) (1 - x) 1 = 1 := 
by
  sorry

end operation_result_l168_168186


namespace remainder_prod_mod_10_l168_168102

theorem remainder_prod_mod_10 :
  (2457 * 7963 * 92324) % 10 = 4 :=
  sorry

end remainder_prod_mod_10_l168_168102


namespace donut_combinations_l168_168676

-- Define the problem statement where Bill needs to purchase 10 donuts,
-- with at least one of each of the 5 kinds, and calculate the combinations.

def count_donut_combinations : ℕ :=
  Nat.choose 9 4

theorem donut_combinations :
  count_donut_combinations = 126 :=
by
  -- Proof can be filled in here
  sorry

end donut_combinations_l168_168676


namespace total_students_in_high_school_l168_168889

-- Definitions based on the problem conditions
def freshman_students : ℕ := 400
def sample_students : ℕ := 45
def sophomore_sample_students : ℕ := 15
def senior_sample_students : ℕ := 10

-- The theorem to be proved
theorem total_students_in_high_school : (sample_students = 45) → (freshman_students = 400) → (sophomore_sample_students = 15) → (senior_sample_students = 10) → ∃ total_students : ℕ, total_students = 900 :=
by
  sorry

end total_students_in_high_school_l168_168889


namespace eccentricity_of_hyperbola_l168_168920

variables (a b c e : ℝ)
variable (h1 : a > 0)
variable (h2 : b > 0)
variable (h3 : c = Real.sqrt (a^2 + b^2))
variable (h4 : 3 * -(a^2 / c) + c = a^2 * c / (b^2 - a^2) + c)
variable (h5 : e = c / a)

theorem eccentricity_of_hyperbola : e = Real.sqrt 3 :=
by {
  sorry
}

end eccentricity_of_hyperbola_l168_168920


namespace largest_integer_b_l168_168960

theorem largest_integer_b (b : ℤ) : (b^2 < 60) → b ≤ 7 :=
by sorry

end largest_integer_b_l168_168960


namespace time_for_b_alone_l168_168917

theorem time_for_b_alone (A B : ℝ) (h1 : A + B = 1 / 16) (h2 : A = 1 / 24) : B = 1 / 48 :=
by
  sorry

end time_for_b_alone_l168_168917


namespace trajectory_of_center_line_passes_fixed_point_l168_168258

-- Define the conditions
def pointA : ℝ × ℝ := (4, 0)
def chord_length : ℝ := 8
def pointB : ℝ × ℝ := (-3, 0)
def not_perpendicular_to_x_axis (t : ℝ) : Prop := t ≠ 0
def trajectory_eq (x y : ℝ) : Prop := y^2 = 8 * x
def line_eq (t m y x : ℝ) : Prop := x = t * y + m
def x_axis_angle_bisector (y1 x1 y2 x2 : ℝ) : Prop := (y1 / (x1 + 3)) + (y2 / (x2 + 3)) = 0

-- Prove the trajectory of the center of the moving circle is \( y^2 = 8x \)
theorem trajectory_of_center (x y : ℝ) 
  (H1: (x-4)^2 + y^2 = 4^2 + x^2) 
  (H2: trajectory_eq x y) : 
  trajectory_eq x y := sorry

-- Prove the line passes through the fixed point (3, 0)
theorem line_passes_fixed_point (t m y1 x1 y2 x2 : ℝ) 
  (Ht: not_perpendicular_to_x_axis t)
  (Hsys: ∀ y x, line_eq t m y x → trajectory_eq x y)
  (Hangle: x_axis_angle_bisector y1 x1 y2 x2) : 
  (m = 3) ∧ ∃ y, line_eq t 3 y 3 := sorry

end trajectory_of_center_line_passes_fixed_point_l168_168258


namespace range_of_a_l168_168688

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x + 1| + |x - 2| > a) → a < 3 :=
by
  sorry

end range_of_a_l168_168688


namespace interest_rate_calculation_l168_168709

theorem interest_rate_calculation
  (P : ℕ) 
  (I : ℕ) 
  (T : ℕ) 
  (R : ℕ) 
  (principal : P = 9200) 
  (time : T = 3) 
  (interest_diff : P - 5888 = I) 
  (interest_formula : I = P * R * T / 100) 
  : R = 12 :=
sorry

end interest_rate_calculation_l168_168709


namespace x_eq_1_is_iff_x2_minus_2x_plus_1_eq_0_l168_168261

theorem x_eq_1_is_iff_x2_minus_2x_plus_1_eq_0 :
  (x = 1 ↔ x^2 - 2 * x + 1 = 0) :=
sorry

end x_eq_1_is_iff_x2_minus_2x_plus_1_eq_0_l168_168261


namespace adult_ticket_cost_l168_168038

def num_total_tickets : ℕ := 510
def cost_senior_ticket : ℕ := 15
def total_receipts : ℤ := 8748
def num_senior_tickets : ℕ := 327
def num_adult_tickets : ℕ := num_total_tickets - num_senior_tickets
def revenue_senior : ℤ := num_senior_tickets * cost_senior_ticket
def revenue_adult (cost_adult_ticket : ℤ) : ℤ := num_adult_tickets * cost_adult_ticket

theorem adult_ticket_cost : 
  ∃ (cost_adult_ticket : ℤ), 
    revenue_adult cost_adult_ticket + revenue_senior = total_receipts ∧ 
    cost_adult_ticket = 21 :=
by
  sorry

end adult_ticket_cost_l168_168038


namespace vector_addition_AC_l168_168119

def vector := (ℝ × ℝ)

def AB : vector := (0, 1)
def BC : vector := (1, 0)

def AC (AB BC : vector) : vector := (AB.1 + BC.1, AB.2 + BC.2) 

theorem vector_addition_AC (AB BC : vector) (h1 : AB = (0, 1)) (h2 : BC = (1, 0)) : 
  AC AB BC = (1, 1) :=
by
  sorry

end vector_addition_AC_l168_168119


namespace describe_T_l168_168665

def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (common : ℝ), 
    (common = 5 ∧ p.1 + 3 = common ∧ p.2 - 6 ≤ common) ∨
    (common = 5 ∧ p.2 - 6 = common ∧ p.1 + 3 ≤ common) ∨
    (common = p.1 + 3 ∧ common = p.2 - 6 ∧ common ≤ 5)}

theorem describe_T :
  T = {(2, y) | y ≤ 11} ∪ { (x, 11) | x ≤ 2} ∪ { (x, x + 9) | x ≤ 2} :=
by
  sorry

end describe_T_l168_168665


namespace find_AC_l168_168447

noncomputable def isTriangle (A B C : Type) : Type := sorry

-- Define angles and lengths.
variables (A B C : Type)
variables (angle_A angle_B : ℝ)
variables (BC AC : ℝ)

-- Assume the given conditions.
axiom angle_A_60 : angle_A = 60 * Real.pi / 180
axiom angle_B_45 : angle_B = 45 * Real.pi / 180
axiom BC_12 : BC = 12

-- Statement to prove.
theorem find_AC 
  (h_triangle : isTriangle A B C)
  (h_angle_A : angle_A = 60 * Real.pi / 180)
  (h_angle_B : angle_B = 45 * Real.pi / 180)
  (h_BC : BC = 12) :
  ∃ AC : ℝ, AC = 8 * Real.sqrt 3 / 3 :=
sorry

end find_AC_l168_168447


namespace tangent_condition_sum_f_l168_168816

def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x + 1

theorem tangent_condition (a : ℝ) (h : f a 1 = f a 1) (m : ℝ) : 
    (3 * a + 1 = (7 - (f a 1)) / 2) := 
    sorry

theorem sum_f (a : ℝ) (h : a = 3/7) : 
    f a (-4) + f a (-3) + f a (-2) + f a (-1) + f a 0 + 
    f a 1 + f a 2 + f a 3 + f a 4 = 9 := 
    sorry

end tangent_condition_sum_f_l168_168816


namespace smallest_perimeter_of_triangle_with_area_sqrt3_l168_168443

open Real

-- Define an equilateral triangle with given area
def equilateral_triangle (a : ℝ) : Prop :=
  ∃ s: ℝ, s > 0 ∧ a = (sqrt 3 / 4) * s^2

-- Problem statement: Prove the smallest perimeter of such a triangle is 6.
theorem smallest_perimeter_of_triangle_with_area_sqrt3 : 
  equilateral_triangle (sqrt 3) → ∃ s: ℝ, s > 0 ∧ 3 * s = 6 :=
by 
  sorry

end smallest_perimeter_of_triangle_with_area_sqrt3_l168_168443


namespace fuel_A_volume_l168_168245

-- Let V_A and V_B be defined as the volumes of fuel A and B respectively.
def V_A : ℝ := sorry
def V_B : ℝ := sorry

-- Given conditions:
axiom h1 : V_A + V_B = 214
axiom h2 : 0.12 * V_A + 0.16 * V_B = 30

-- Prove that the volume of fuel A added, V_A, is 106 gallons.
theorem fuel_A_volume : V_A = 106 := 
by
  sorry

end fuel_A_volume_l168_168245


namespace hyperbola_eccentricity_l168_168886

noncomputable def eccentricity (a b : ℝ) : ℝ := 
  let e := (1 + (b^2) / (a^2)).sqrt
  e

theorem hyperbola_eccentricity 
  (a b : ℝ) 
  (h1 : a + b = 5)
  (h2 : a * b = 6)
  (h3 : a > b) :
  eccentricity a b = Real.sqrt 13 / 3 :=
sorry

end hyperbola_eccentricity_l168_168886


namespace find_n_l168_168192

theorem find_n
  (n : ℤ)
  (h : n + (n + 1) + (n + 2) + (n + 3) = 30) :
  n = 6 :=
by
  sorry

end find_n_l168_168192


namespace balls_in_boxes_l168_168680

theorem balls_in_boxes : (2^7 = 128) := 
by
  -- number of balls
  let n : ℕ := 7
  -- number of boxes
  let b : ℕ := 2
  have h : b ^ n = 128 := by sorry
  exact h

end balls_in_boxes_l168_168680


namespace smallest_n_for_divisibility_property_l168_168529

theorem smallest_n_for_divisibility_property (k : ℕ) : ∃ n : ℕ, n = k + 2 ∧ ∀ (S : Finset ℤ), 
  S.card = n → 
  ∃ (a b : ℤ), a ∈ S ∧ b ∈ S ∧ (a ≠ b ∧ (a + b) % (2 * k + 1) = 0 ∨ (a - b) % (2 * k + 1) = 0) :=
by
sorry

end smallest_n_for_divisibility_property_l168_168529


namespace calc_expr_solve_fractional_eq_l168_168017

-- Problem 1: Calculate the expression
theorem calc_expr : (-2)^2 - (64:ℝ)^(1/3) + (-3)^0 - (1/3)^0 = 0 := 
by 
  -- provide intermediate steps (not necessary as per the question requirements)
  sorry

-- Problem 2: Solve the fractional equation
theorem solve_fractional_eq (x : ℝ) (h : x ≠ -1) : 
  (x / (x + 1) = 5 / (2 * x + 2) - 1) ↔ x = 3 / 4 := 
by 
  -- provide intermediate steps (not necessary as per the question requirements)
  sorry

end calc_expr_solve_fractional_eq_l168_168017


namespace polynomial_degree_bound_l168_168723

theorem polynomial_degree_bound (m n k : ℕ) (P : Polynomial ℤ) 
  (hm_pos : 0 < m)
  (hn_pos : 0 < n)
  (hk_pos : 2 ≤ k)
  (hP_odd : ∀ i, P.coeff i % 2 = 1) 
  (h_div : (X - 1) ^ m ∣ P)
  (hm_bound : m ≥ 2 ^ k) :
  n ≥ 2 ^ (k + 1) - 1 := sorry

end polynomial_degree_bound_l168_168723


namespace division_expression_l168_168737

theorem division_expression :
  (240 : ℚ) / (12 + 12 * 2 - 3) = 240 / 33 := by
  sorry

end division_expression_l168_168737


namespace T_perimeter_is_20_l168_168881

-- Define the perimeter of a rectangle given its length and width
def perimeter_rectangle (length width : ℝ) : ℝ :=
  2 * length + 2 * width

-- Given conditions
def rect1_length : ℝ := 1
def rect1_width : ℝ := 4
def rect2_length : ℝ := 2
def rect2_width : ℝ := 5
def overlap_height : ℝ := 1

-- Calculate the perimeter of each rectangle
def perimeter_rect1 : ℝ := perimeter_rectangle rect1_length rect1_width
def perimeter_rect2 : ℝ := perimeter_rectangle rect2_length rect2_width

-- Calculate the overlap adjustment
def overlap_adjustment : ℝ := 2 * overlap_height

-- The total perimeter of the T shape
def perimeter_T : ℝ := perimeter_rect1 + perimeter_rect2 - overlap_adjustment

-- The proof statement that we need to show
theorem T_perimeter_is_20 : perimeter_T = 20 := by
  sorry

end T_perimeter_is_20_l168_168881


namespace minimum_games_for_80_percent_l168_168237

theorem minimum_games_for_80_percent :
  ∃ N : ℕ, ( ∀ N' : ℕ, (1 + N') / (5 + N') * 100 < 80 → N < N') ∧ (1 + N) / (5 + N) * 100 ≥ 80 :=
sorry

end minimum_games_for_80_percent_l168_168237


namespace parabola_num_xintercepts_l168_168586

-- Defining the equation of the parabola
def parabola (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 5

-- The main theorem to state: the number of x-intercepts for the parabola is 2.
theorem parabola_num_xintercepts : ∃ (a b : ℝ), parabola a = 0 ∧ parabola b = 0 ∧ a ≠ b :=
by
  sorry

end parabola_num_xintercepts_l168_168586


namespace bus_speed_including_stoppages_l168_168611

-- Definitions based on conditions
def speed_excluding_stoppages : ℝ := 50 -- kmph
def stoppage_time_per_hour : ℝ := 18 -- minutes

-- Lean statement of the problem
theorem bus_speed_including_stoppages :
  (speed_excluding_stoppages * (1 - stoppage_time_per_hour / 60)) = 35 := by
  sorry

end bus_speed_including_stoppages_l168_168611


namespace sum_possible_m_continuous_l168_168214

noncomputable def g (x m : ℝ) : ℝ :=
if x < m then x^2 + 4 * x + 3 else 3 * x + 9

theorem sum_possible_m_continuous :
  let m₁ := -3
  let m₂ := 2
  m₁ + m₂ = -1 :=
by
  sorry

end sum_possible_m_continuous_l168_168214


namespace cube_volume_l168_168178

theorem cube_volume (S : ℝ) (hS : S = 294) : ∃ V : ℝ, V = 343 := by
  sorry

end cube_volume_l168_168178


namespace total_pumpkin_weight_l168_168259

-- Conditions
def weight_first_pumpkin : ℝ := 4
def weight_second_pumpkin : ℝ := 8.7

-- Statement
theorem total_pumpkin_weight :
  weight_first_pumpkin + weight_second_pumpkin = 12.7 :=
by
  -- Proof can be done manually or via some automation here
  sorry

end total_pumpkin_weight_l168_168259


namespace insufficient_info_for_pumpkins_l168_168859

variable (jason_watermelons : ℕ) (sandy_watermelons : ℕ) (total_watermelons : ℕ)

theorem insufficient_info_for_pumpkins (h1 : jason_watermelons = 37)
  (h2 : sandy_watermelons = 11)
  (h3 : jason_watermelons + sandy_watermelons = total_watermelons)
  (h4 : total_watermelons = 48) : 
  ¬∃ (jason_pumpkins : ℕ), true
:= by
  sorry

end insufficient_info_for_pumpkins_l168_168859


namespace one_third_of_flour_l168_168390

-- Definition of the problem conditions
def initial_flour : ℚ := 5 + 2 / 3
def portion : ℚ := 1 / 3

-- Definition of the theorem to prove
theorem one_third_of_flour : portion * initial_flour = 1 + 8 / 9 :=
by {
  -- Placeholder proof
  sorry
}

end one_third_of_flour_l168_168390


namespace complement_union_complement_intersection_complementA_intersect_B_l168_168459

def setA (x : ℝ) : Prop := 3 ≤ x ∧ x < 7
def setB (x : ℝ) : Prop := 2 < x ∧ x < 10

theorem complement_union (x : ℝ) : ¬(setA x ∨ setB x) ↔ x ≤ 2 ∨ x ≥ 10 := sorry

theorem complement_intersection (x : ℝ) : ¬(setA x ∧ setB x) ↔ x < 3 ∨ x ≥ 7 := sorry

theorem complementA_intersect_B (x : ℝ) : (¬setA x ∧ setB x) ↔ (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10) := sorry

end complement_union_complement_intersection_complementA_intersect_B_l168_168459


namespace johnny_future_years_l168_168353

theorem johnny_future_years (x : ℕ) (h1 : 8 + x = 2 * (8 - 3)) : x = 2 :=
by
  sorry

end johnny_future_years_l168_168353


namespace geo_seq_product_l168_168879

theorem geo_seq_product (a : ℕ → ℝ) (r : ℝ) (h_pos : ∀ n, 0 < a n) 
  (h_geo : ∀ n, a (n + 1) = a n * r) 
  (h_roots : a 1 ^ 2 - 10 * a 1 + 16 = 0) 
  (h_root19 : a 19 ^ 2 - 10 * a 19 + 16 = 0) : 
  a 8 * a 10 * a 12 = 64 :=
by
  sorry

end geo_seq_product_l168_168879


namespace a_not_multiple_of_5_l168_168912

theorem a_not_multiple_of_5 (a : ℤ) (h : a % 5 ≠ 0) : (a^4 + 4) % 5 = 0 :=
sorry

end a_not_multiple_of_5_l168_168912


namespace jaden_toy_cars_l168_168319

theorem jaden_toy_cars :
  let initial : Nat := 14
  let bought : Nat := 28
  let birthday : Nat := 12
  let to_sister : Nat := 8
  let to_friend : Nat := 3
  initial + bought + birthday - to_sister - to_friend = 43 :=
by
  let initial : Nat := 14
  let bought : Nat := 28
  let birthday : Nat := 12
  let to_sister : Nat := 8
  let to_friend : Nat := 3
  show initial + bought + birthday - to_sister - to_friend = 43
  sorry

end jaden_toy_cars_l168_168319


namespace directrix_of_parabola_l168_168073

open Real

noncomputable def parabola_directrix (a : ℝ) : ℝ := -a / 4

theorem directrix_of_parabola (a : ℝ) (h : a = 4) : parabola_directrix a = -4 :=
by
  sorry

end directrix_of_parabola_l168_168073


namespace geometric_seq_increasing_condition_not_sufficient_nor_necessary_l168_168128

-- Definitions based on conditions
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) := ∀ n : ℕ, a (n + 1) = q * a n
def monotonically_increasing (a : ℕ → ℝ) := ∀ n : ℕ, a n ≤ a (n + 1)
def common_ratio_gt_one (q : ℝ) := q > 1

-- Proof statement of the problem
theorem geometric_seq_increasing_condition_not_sufficient_nor_necessary 
    (a : ℕ → ℝ) (q : ℝ) 
    (h1 : geometric_sequence a q) : 
    ¬(common_ratio_gt_one q ↔ monotonically_increasing a) :=
sorry

end geometric_seq_increasing_condition_not_sufficient_nor_necessary_l168_168128


namespace f_10_l168_168970

noncomputable def f : ℕ → ℕ
| 0       => 1
| (n + 1) => 2 * f n

theorem f_10 : f 10 = 2^10 :=
by
  -- This would be filled in with the necessary proof steps to show f(10) = 2^10
  sorry

end f_10_l168_168970


namespace find_primes_l168_168562

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ (m : ℕ), m ∣ n → m = 1 ∨ m = n

def divides (a b : ℕ) : Prop := ∃ k, b = k * a

/- Define the three conditions -/
def condition1 (p q r : ℕ) : Prop := divides p (1 + q ^ r)
def condition2 (p q r : ℕ) : Prop := divides q (1 + r ^ p)
def condition3 (p q r : ℕ) : Prop := divides r (1 + p ^ q)

def satisfies_conditions (p q r : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ is_prime r ∧ condition1 p q r ∧ condition2 p q r ∧ condition3 p q r

theorem find_primes (p q r : ℕ) :
  satisfies_conditions p q r ↔ (p = 2 ∧ q = 5 ∧ r = 3) ∨ (p = 5 ∧ q = 3 ∧ r = 2) ∨ (p = 3 ∧ q = 2 ∧ r = 5) :=
by
  sorry

end find_primes_l168_168562


namespace average_age_union_l168_168096

open Real

variables {a b c d A B C D : ℝ}

theorem average_age_union (h1 : A / a = 40)
                         (h2 : B / b = 30)
                         (h3 : C / c = 45)
                         (h4 : D / d = 35)
                         (h5 : (A + B) / (a + b) = 37)
                         (h6 : (A + C) / (a + c) = 42)
                         (h7 : (A + D) / (a + d) = 39)
                         (h8 : (B + C) / (b + c) = 40)
                         (h9 : (B + D) / (b + d) = 37)
                         (h10 : (C + D) / (c + d) = 43) : 
  (A + B + C + D) / (a + b + c + d) = 44.5 := 
sorry

end average_age_union_l168_168096


namespace sum_series_eq_11_div_18_l168_168100

theorem sum_series_eq_11_div_18 :
  (∑' n : ℕ, if n = 0 then 0 else 1 / (n * (n + 3))) = 11 / 18 :=
by
  sorry

end sum_series_eq_11_div_18_l168_168100


namespace min_value_of_a_l168_168700

theorem min_value_of_a (a b c : ℝ) (h₁ : a > 0) (h₂ : ∃ p q : ℝ, 0 < p ∧ p < 2 ∧ 0 < q ∧ q < 2 ∧ 
  ∀ x, ax^2 + bx + c = a * (x - p) * (x - q)) (h₃ : 25 * a + 10 * b + 4 * c ≥ 4) (h₄ : c ≥ 1) : 
  a ≥ 16 / 25 :=
sorry

end min_value_of_a_l168_168700


namespace decagon_diagonals_l168_168692

-- Define the number of sides of the polygon
def n : ℕ := 10

-- Calculate the number of diagonals in an n-sided polygon
def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- State the theorem that the number of diagonals in a decagon is 35
theorem decagon_diagonals : number_of_diagonals n = 35 := by
  sorry

end decagon_diagonals_l168_168692


namespace repeating_decimal_sum_to_fraction_l168_168744

def repeating_decimal_123 : ℚ := 123 / 999
def repeating_decimal_0045 : ℚ := 45 / 9999
def repeating_decimal_000678 : ℚ := 678 / 999999

theorem repeating_decimal_sum_to_fraction :
  repeating_decimal_123 + repeating_decimal_0045 + repeating_decimal_000678 = 128178 / 998001000 :=
by
  sorry

end repeating_decimal_sum_to_fraction_l168_168744


namespace difference_of_sides_l168_168947

-- Definitions based on conditions
def smaller_square_side (s : ℝ) := s
def larger_square_side (S s : ℝ) (h : (S^2 : ℝ) = 4 * s^2) := S

-- Theorem statement based on the proof problem
theorem difference_of_sides (s S : ℝ) (h : (S^2 : ℝ) = 4 * s^2) : S - s = s := 
by
  sorry

end difference_of_sides_l168_168947


namespace find_multiple_l168_168986

theorem find_multiple (x k : ℕ) (hx : x > 0) (h_eq : x + 17 = k * (1/x)) (h_x : x = 3) : k = 60 :=
by
  sorry

end find_multiple_l168_168986


namespace sector_perimeter_l168_168208

noncomputable def radius : ℝ := 2
noncomputable def central_angle_deg : ℝ := 120
noncomputable def expected_perimeter : ℝ := (4 / 3) * Real.pi + 4

theorem sector_perimeter (r : ℝ) (θ : ℝ) (h_r : r = radius) (h_θ : θ = central_angle_deg) :
    let arc_length := θ / 360 * 2 * Real.pi * r
    let perimeter := arc_length + 2 * r
    perimeter = expected_perimeter :=
by
  -- Skip the proof
  sorry

end sector_perimeter_l168_168208


namespace ancient_chinese_problem_l168_168602

theorem ancient_chinese_problem (x y : ℤ) 
  (h1 : y = 8 * x - 3) 
  (h2 : y = 7 * x + 4) : 
  (y = 8 * x - 3) ∧ (y = 7 * x + 4) :=
by
  exact ⟨h1, h2⟩

end ancient_chinese_problem_l168_168602


namespace diamonds_in_G_10_l168_168402

-- Define the sequence rule for diamonds in Gn
def diamonds_in_G (n : ℕ) : ℕ :=
  if n = 1 then 2
  else if n = 2 then 10
  else 2 + 2 * n^2 + 2 * n - 4

-- The main theorem to prove that the number of diamonds in G₁₀ is 218
theorem diamonds_in_G_10 : diamonds_in_G 10 = 218 := by
  sorry

end diamonds_in_G_10_l168_168402


namespace obtuse_dihedral_angles_l168_168926

theorem obtuse_dihedral_angles (AOB BOC COA : ℝ) (h1 : AOB > 90) (h2 : BOC > 90) (h3 : COA > 90) :
  ∃ α β γ : ℝ, α > 90 ∧ β > 90 ∧ γ > 90 :=
sorry

end obtuse_dihedral_angles_l168_168926


namespace total_glass_area_l168_168341

theorem total_glass_area 
  (len₁ len₂ len₃ wid₁ wid₂ wid₃ : ℕ)
  (h₁ : len₁ = 30) (h₂ : wid₁ = 12)
  (h₃ : len₂ = 30) (h₄ : wid₂ = 12)
  (h₅ : len₃ = 20) (h₆ : wid₃ = 12) :
  (len₁ * wid₁ + len₂ * wid₂ + len₃ * wid₃) = 960 := 
by
  sorry

end total_glass_area_l168_168341


namespace solution_of_az_eq_b_l168_168426

theorem solution_of_az_eq_b (a b z x y : ℝ) :
  (∃! x, 4 + 3 * a * x = 2 * a - 7) →
  (¬ ∃ y, 2 + y = (b + 1) * y) →
  az = b →
  z = 0 :=
by
  intros h1 h2 h3
  -- proof starts here
  sorry

end solution_of_az_eq_b_l168_168426


namespace ellipse_focus_distance_l168_168004

theorem ellipse_focus_distance (m : ℝ) (a b c : ℝ)
  (ellipse_eq : ∀ x y : ℝ, x^2 / m + y^2 / 16 = 1)
  (focus_distance : ∀ P : ℝ × ℝ, ∃ F1 F2 : ℝ × ℝ, dist P F1 = 3 ∧ dist P F2 = 7) :
  m = 25 := 
  sorry

end ellipse_focus_distance_l168_168004


namespace number_of_sets_count_number_of_sets_l168_168698

theorem number_of_sets (P : Set ℕ) :
  ({1, 2} ⊆ P ∧ P ⊆ {1, 2, 3, 4}) → (P = {1, 2} ∨ P = {1, 2, 3} ∨ P = {1, 2, 4}) :=
sorry

theorem count_number_of_sets :
  ∃ (Ps : Finset (Set ℕ)), 
  (∀ P ∈ Ps, {1, 2} ⊆ P ∧ P ⊆ {1, 2, 3, 4}) ∧ Ps.card = 3 :=
sorry

end number_of_sets_count_number_of_sets_l168_168698


namespace four_digit_numbers_with_product_exceeds_10_l168_168286

theorem four_digit_numbers_with_product_exceeds_10 : 
  (∃ count : ℕ, 
    count = (6 * 58 * 10) ∧
    count = 3480) := 
by {
  sorry
}

end four_digit_numbers_with_product_exceeds_10_l168_168286


namespace polynomial_difference_square_l168_168603

theorem polynomial_difference_square (a : Fin 11 → ℝ) (x : ℝ) (sqrt2 : ℝ)
  (h_eq : (sqrt2 - x)^10 = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + 
          a 6 * x^6 + a 7 * x^7 + a 8 * x^8 + a 9 * x^9 + a 10 * x^10) : 
  ((a 0 + a 2 + a 4 + a 6 + a 8 + a 10)^2 - (a 1 + a 3 + a 5 + a 7 + a 9)^2 = 1) :=
by
  sorry

end polynomial_difference_square_l168_168603


namespace livestock_allocation_l168_168607

theorem livestock_allocation :
  ∃ (x y z : ℕ), x + y + z = 100 ∧ 20 * x + 6 * y + z = 200 ∧ x = 5 ∧ y = 1 ∧ z = 94 :=
by
  sorry

end livestock_allocation_l168_168607


namespace complement_P_inter_Q_l168_168822

def P : Set ℝ := {x | x^2 - 2 * x ≥ 0}
def Q : Set ℝ := {x | 1 < x ∧ x ≤ 2}
def complement_P : Set ℝ := {x | 0 < x ∧ x < 2}

theorem complement_P_inter_Q : (complement_P ∩ Q) = {x | 1 < x ∧ x < 2} := by
  sorry

end complement_P_inter_Q_l168_168822


namespace total_tissues_brought_l168_168810

def number_of_students (group1 group2 group3 : Nat) : Nat :=
  group1 + group2 + group3

def number_of_tissues_per_student (tissues_per_box : Nat) (total_students : Nat) : Nat :=
  tissues_per_box * total_students

theorem total_tissues_brought :
  let group1 := 9
  let group2 := 10
  let group3 := 11
  let tissues_per_box := 40
  let total_students := number_of_students group1 group2 group3
  number_of_tissues_per_student tissues_per_box total_students = 1200 :=
by
  sorry

end total_tissues_brought_l168_168810


namespace prime_mod_30_not_composite_l168_168667

theorem prime_mod_30_not_composite (p : ℕ) (h_prime : Prime p) (h_gt_30 : p > 30) : 
  ¬ ∃ (x : ℕ), (x > 1 ∧ ∃ (a b : ℕ), x = a * b ∧ a > 1 ∧ b > 1) ∧ (0 < x ∧ x < 30 ∧ ∃ (k : ℕ), p = 30 * k + x) :=
by
  sorry

end prime_mod_30_not_composite_l168_168667


namespace shop_owner_percentage_profit_l168_168846

theorem shop_owner_percentage_profit :
  let cost_price_per_kg := 100
  let buy_cheat_percent := 18.5 / 100
  let sell_cheat_percent := 22.3 / 100
  let amount_bought := 1 / (1 + buy_cheat_percent)
  let amount_sold := 1 - sell_cheat_percent
  let effective_cost_price := cost_price_per_kg * amount_sold / amount_bought
  let selling_price := cost_price_per_kg
  let profit := selling_price - effective_cost_price
  let percentage_profit := (profit / effective_cost_price) * 100
  percentage_profit = 52.52 :=
by
  sorry

end shop_owner_percentage_profit_l168_168846


namespace part_a_part_b_l168_168257

def N := 10^40

def is_divisor (a b : ℕ) : Prop := b % a = 0

def is_perfect_square (a : ℕ) : Prop := ∃ m : ℕ, m * m = a

def is_perfect_cube (a : ℕ) : Prop := ∃ m : ℕ, m * m * m = a

def is_perfect_power (a : ℕ) : Prop := ∃ (m n : ℕ), n > 1 ∧ a = m^n

def num_divisors_not_square_or_cube (n : ℕ) : ℕ := sorry -- This should calculate the number of divisors that are neither perfect squares nor perfect cubes

def num_divisors_not_in_form_m_n (n : ℕ) : ℕ := sorry -- This should calculate the number of divisors that cannot be represented in the form m^n where n > 1

theorem part_a : num_divisors_not_square_or_cube N = 1093 := by
  sorry

theorem part_b : num_divisors_not_in_form_m_n N = 981 := by
  sorry

end part_a_part_b_l168_168257


namespace distance_to_grandma_l168_168034

-- Definitions based on the conditions
def miles_per_gallon : ℕ := 20
def gallons_needed : ℕ := 5

-- The theorem statement to prove the distance is 100 miles
theorem distance_to_grandma : miles_per_gallon * gallons_needed = 100 := by
  sorry

end distance_to_grandma_l168_168034


namespace two_primes_equal_l168_168931

theorem two_primes_equal
  (a b c : ℕ)
  (p q r : ℕ)
  (hp : p = b^c + a ∧ Nat.Prime p)
  (hq : q = a^b + c ∧ Nat.Prime q)
  (hr : r = c^a + b ∧ Nat.Prime r) :
  p = q ∨ q = r ∨ r = p := 
sorry

end two_primes_equal_l168_168931


namespace complement_supplement_angle_l168_168360

theorem complement_supplement_angle (α : ℝ) : 
  ( 180 - α) = 3 * ( 90 - α ) → α = 45 :=
by 
  sorry

end complement_supplement_angle_l168_168360


namespace relationship_between_D_and_A_l168_168428

variables (A B C D : Prop)

def sufficient_not_necessary (P Q : Prop) : Prop := (P → Q) ∧ ¬ (Q → P)
def necessary_not_sufficient (P Q : Prop) : Prop := (Q → P) ∧ ¬ (P → Q)
def necessary_and_sufficient (P Q : Prop) : Prop := (P ↔ Q)

-- Conditions
axiom h1 : sufficient_not_necessary A B
axiom h2 : necessary_not_sufficient C B
axiom h3 : necessary_and_sufficient D C

-- Proof Goal
theorem relationship_between_D_and_A : necessary_not_sufficient D A :=
by
  sorry

end relationship_between_D_and_A_l168_168428


namespace period_sine_transformed_l168_168754

theorem period_sine_transformed (x : ℝ) : 
  let y := 3 * Real.sin ((x / 3) + (Real.pi / 4))
  ∃ p : ℝ, (∀ x : ℝ, y = 3 * Real.sin ((x + p) / 3 + (Real.pi / 4)) ↔ y = 3 * Real.sin ((x / 3) + (Real.pi / 4))) ∧ p = 6 * Real.pi :=
sorry

end period_sine_transformed_l168_168754


namespace fraction_value_l168_168457

theorem fraction_value (m n : ℤ) (h : (m - 8) * (m - 8) + abs (n + 6) = 0) : n / m = -(3 / 4) :=
by sorry

end fraction_value_l168_168457


namespace gcd_linear_combination_l168_168837

theorem gcd_linear_combination (a b : ℤ) (h : Int.gcd a b = 1) : 
    Int.gcd (11 * a + 2 * b) (18 * a + 5 * b) = 1 := 
by
  sorry

end gcd_linear_combination_l168_168837


namespace total_points_l168_168858

theorem total_points (Jon Jack Tom : ℕ) (h1 : Jon = 3) (h2 : Jack = Jon + 5) (h3 : Tom = Jon + Jack - 4) : Jon + Jack + Tom = 18 := by
  sorry

end total_points_l168_168858


namespace negation_example_l168_168221

theorem negation_example (h : ∀ x ∈ Set.Icc (-1 : ℝ) 1, x^2 + 3 * x - 1 > 0) :
  ∃ x ∈ Set.Icc (-1 : ℝ) 1, x^2 + 3 * x - 1 ≤ 0 :=
sorry

end negation_example_l168_168221


namespace find_k_l168_168841

-- Define the problem's conditions and constants
variables (S x y : ℝ)

-- Define the main theorem to prove k = 8 given the conditions
theorem find_k (h1 : 0.75 * x + ((S - 0.75 * x) * x) / (x + y) - (S * x) / (x + y) = 18) :
  (x * y / 3) / (x + y) = 8 := by 
  sorry

end find_k_l168_168841


namespace investment_duration_l168_168415

theorem investment_duration 
  (P SI R : ℕ) (T : ℕ) 
  (hP : P = 800) 
  (hSI : SI = 128) 
  (hR : R = 4) 
  (h : SI = P * R * T / 100) 
  : T = 4 :=
by 
  rw [hP, hSI, hR] at h
  sorry

end investment_duration_l168_168415


namespace solveRealInequality_l168_168830

theorem solveRealInequality (x : ℝ) (hx : 0 < x) : x * Real.sqrt (18 - x) + Real.sqrt (18 * x - x^3) ≥ 18 → x = 3 :=
by
  sorry -- proof to be filled in

end solveRealInequality_l168_168830


namespace current_price_after_adjustment_l168_168559

variable (x : ℝ) -- Define x, the original price per unit

theorem current_price_after_adjustment (x : ℝ) : (x + 10) * 0.75 = ((x + 10) * 0.75) :=
by
  sorry

end current_price_after_adjustment_l168_168559


namespace sum_first_eight_terms_geometric_sequence_l168_168006

noncomputable def sum_of_geometric_sequence (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem sum_first_eight_terms_geometric_sequence :
  sum_of_geometric_sequence (1/2) (1/3) 8 = 9840 / 6561 :=
by
  sorry

end sum_first_eight_terms_geometric_sequence_l168_168006


namespace five_digit_number_l168_168839

open Nat

noncomputable def problem_statement : Prop :=
  ∃ A B C D E F : ℕ,
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
    C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
    D ≠ E ∧ D ≠ F ∧
    E ≠ F ∧
    A + B + C + D + E + F = 25 ∧
    (A, B, C, D, E, F) = (3, 4, 2, 1, 6, 9)

theorem five_digit_number : problem_statement := 
  sorry

end five_digit_number_l168_168839


namespace john_spends_6_dollars_l168_168424

-- Let treats_per_day, cost_per_treat, and days_in_month be defined by the conditions of the problem.
def treats_per_day : ℕ := 2
def cost_per_treat : ℝ := 0.1
def days_in_month : ℕ := 30

-- The total expenditure should be defined as the number of treats multiplied by their cost.
def total_number_of_treats := treats_per_day * days_in_month
def total_expenditure := total_number_of_treats * cost_per_treat

-- The statement to be proven: John spends $6 on the treats.
theorem john_spends_6_dollars :
  total_expenditure = 6 :=
sorry

end john_spends_6_dollars_l168_168424


namespace constant_S13_l168_168683

noncomputable def S (a d : ℝ) (n : ℕ) : ℝ := n * (2 * a + (n - 1) * d) / 2

theorem constant_S13 (a d p : ℝ) 
  (h : a + a + 3 * d + a + 7 * d = p) : 
  S a d 13 = 13 * p / 18 :=
by
  unfold S
  sorry

end constant_S13_l168_168683


namespace mollys_present_age_l168_168743

theorem mollys_present_age (x : ℤ) (h : x + 18 = 5 * (x - 6)) : x = 12 := by
  sorry

end mollys_present_age_l168_168743


namespace complement_is_correct_l168_168394

def U : Set ℝ := {x | x^2 ≤ 4}
def A : Set ℝ := {x | abs (x + 1) ≤ 1}
def complement_U_A : Set ℝ := U \ A

theorem complement_is_correct :
  complement_U_A = {x | 0 < x ∧ x ≤ 2} := by
  sorry

end complement_is_correct_l168_168394


namespace irreducible_fraction_eq_l168_168933

theorem irreducible_fraction_eq (p q : ℕ) (h1 : p > 0) (h2 : q > 0) (h3 : Nat.gcd p q = 1) (h4 : q % 2 = 1) :
  ∃ n k : ℕ, n > 0 ∧ k > 0 ∧ (p : ℚ) / q = (n : ℚ) / (2 ^ k - 1) :=
by
  sorry

end irreducible_fraction_eq_l168_168933


namespace greatest_x_for_lcm_l168_168010

theorem greatest_x_for_lcm (x : ℕ) (h_lcm : Nat.lcm (Nat.lcm x 15) 21 = 105) : x ≤ 105 :=
by
  sorry

end greatest_x_for_lcm_l168_168010


namespace proof_a_squared_plus_b_squared_l168_168751

theorem proof_a_squared_plus_b_squared (a b : ℝ) (h1 : (a + b) ^ 2 = 4) (h2 : a * b = 1) : a ^ 2 + b ^ 2 = 2 := 
by 
  sorry

end proof_a_squared_plus_b_squared_l168_168751


namespace chase_travel_time_l168_168282

-- Definitions of speeds
def chase_speed (C : ℝ) := C
def cameron_speed (C : ℝ) := 2 * C
def danielle_speed (C : ℝ) := 6 * (cameron_speed C)

-- Time taken by Danielle to cover distance
def time_taken_by_danielle (C : ℝ) := 30  
def distance_travelled (C : ℝ) := (time_taken_by_danielle C) * (danielle_speed C)  -- 180C

-- Speeds on specific stretches
def cameron_bike_speed (C : ℝ) := 0.75 * (cameron_speed C)
def chase_scooter_speed (C : ℝ) := 1.25 * (chase_speed C)

-- Prove the time Chase takes to travel the same distance D
theorem chase_travel_time (C : ℝ) : 
  (distance_travelled C) / (chase_speed C) = 180 := sorry

end chase_travel_time_l168_168282


namespace total_repair_cost_l168_168140

theorem total_repair_cost :
  let rate1 := 60
  let hours1 := 8
  let days1 := 14
  let rate2 := 75
  let hours2 := 6
  let days2 := 10
  let parts_cost := 3200
  let first_mechanic_cost := rate1 * hours1 * days1
  let second_mechanic_cost := rate2 * hours2 * days2
  let total_cost := first_mechanic_cost + second_mechanic_cost + parts_cost
  total_cost = 14420 := by
  sorry

end total_repair_cost_l168_168140


namespace temperature_reaches_90_at_17_l168_168296

def temperature (t : ℝ) : ℝ := -t^2 + 14 * t + 40

theorem temperature_reaches_90_at_17 :
  ∃ t : ℝ, temperature t = 90 ∧ t = 17 :=
by
  exists 17
  dsimp [temperature]
  norm_num
  sorry

end temperature_reaches_90_at_17_l168_168296


namespace parallel_vectors_l168_168204

theorem parallel_vectors (m : ℝ) (a b : ℝ × ℝ) (h₁ : a = (2, 3)) (h₂ : b = (-1, 2)) :
  (m * a.1 + b.1) * (-1) - 4 * (m * a.2 + b.2) = 0 → m = -1 / 2 :=
by
  intro h
  rw [h₁, h₂] at h
  simp at h
  sorry

end parallel_vectors_l168_168204


namespace intersection_cardinality_l168_168801

variable {a b : ℝ}
variable {f : ℝ → ℝ}

theorem intersection_cardinality {a b : ℝ} {f : ℝ → ℝ} :
  (∃! y, (0, y) ∈ ({ (x, y) | y = f x ∧ a ≤ x ∧ x ≤ b } ∩ { (x, y) | x = 0 })) ∨
  ¬ (∃ y, (0, y) ∈ { (x, y) | y = f x ∧ a ≤ x ∧ x ≤ b }) :=
by
  sorry

end intersection_cardinality_l168_168801


namespace jail_time_ratio_l168_168000

def arrests (days : ℕ) (cities : ℕ) (arrests_per_day : ℕ) : ℕ := days * cities * arrests_per_day
def jail_days_before_trial (total_arrests : ℕ) (days_before_trial : ℕ) : ℕ := total_arrests * days_before_trial
def weeks_from_days (days : ℕ) : ℕ := days / 7
def time_after_trial (total_jail_time_weeks : ℕ) (weeks_before_trial : ℕ) : ℕ := total_jail_time_weeks - weeks_before_trial
def total_possible_jail_time (total_arrests : ℕ) (sentence_weeks : ℕ) : ℕ := total_arrests * sentence_weeks
def ratio (after_trial_weeks : ℕ) (total_possible_weeks : ℕ) : ℚ := after_trial_weeks / total_possible_weeks

theorem jail_time_ratio 
    (days : ℕ := 30) 
    (cities : ℕ := 21)
    (arrests_per_day : ℕ := 10)
    (days_before_trial : ℕ := 4)
    (total_jail_time_weeks : ℕ := 9900)
    (sentence_weeks : ℕ := 2) :
    ratio 
      (time_after_trial 
        total_jail_time_weeks 
        (weeks_from_days 
          (jail_days_before_trial 
            (arrests days cities arrests_per_day) 
            days_before_trial))) 
      (total_possible_jail_time 
        (arrests days cities arrests_per_day) 
        sentence_weeks) = 1/2 := 
by
  -- We leave the proof as an exercise
  sorry

end jail_time_ratio_l168_168000


namespace true_proposition_is_b_l168_168207

open Real

theorem true_proposition_is_b :
  (∃ n : ℝ, ∀ m : ℝ, m * n = m) ∧
  (¬ ∀ n : ℝ, n^2 ≥ n) ∧
  (¬ ∀ n : ℝ, ∃ m : ℝ, m^2 < n) ∧
  (¬ ∀ n : ℝ, n^2 < n) :=
  by
    sorry

end true_proposition_is_b_l168_168207


namespace find_p_l168_168782

theorem find_p (p : ℕ) : 64^5 = 8^p → p = 10 :=
by
  intro h
  sorry

end find_p_l168_168782


namespace sally_total_spent_l168_168623

-- Define the prices paid by Sally for peaches after the coupon and for cherries
def P_peaches : ℝ := 12.32
def C_cherries : ℝ := 11.54

-- State the problem to prove that the total amount Sally spent is 23.86
theorem sally_total_spent : P_peaches + C_cherries = 23.86 := by
  sorry

end sally_total_spent_l168_168623


namespace total_weight_correct_total_money_earned_correct_l168_168860

variable (records : List Int) (std_weight : Int)

-- Conditions
def deviation_sum (records : List Int) : Int := records.foldl (· + ·) 0

def batch_weight (std_weight : Int) (n : Int) (deviation_sum : Int) : Int :=
  deviation_sum + std_weight * n

def first_day_sales (total_weight : Int) (price_per_kg : Int) : Int :=
  price_per_kg * (total_weight / 2)

def second_day_sales (total_weight : Int) (first_day_sales_weight : Int) (discounted_price_per_kg : Int) : Int :=
  discounted_price_per_kg * (total_weight - first_day_sales_weight)

def total_earnings (first_day_sales : Int) (second_day_sales : Int) : Int :=
  first_day_sales + second_day_sales

-- Proof statements
theorem total_weight_correct : 
  deviation_sum records = 4 ∧ std_weight = 30 ∧ records.length = 8 → 
  batch_weight std_weight records.length (deviation_sum records) = 244 :=
by
  intro h
  sorry

theorem total_money_earned_correct :
  first_day_sales (batch_weight std_weight records.length (deviation_sum records)) 10 = 1220 ∧
  second_day_sales (batch_weight std_weight records.length (deviation_sum records)) (batch_weight std_weight records.length (deviation_sum records) / 2) (10 * 9 / 10) = 1098 →
  total_earnings 1220 1098 = 2318 :=
by
  intro h
  sorry

end total_weight_correct_total_money_earned_correct_l168_168860


namespace p_necessary_condition_q_l168_168190

variable (a b : ℝ) (p : ab = 0) (q : a^2 + b^2 ≠ 0)

theorem p_necessary_condition_q : (∀ a b : ℝ, (ab = 0) → (a^2 + b^2 ≠ 0)) ∧ (∃ a b : ℝ, (a^2 + b^2 ≠ 0) ∧ ¬ (ab = 0)) := sorry

end p_necessary_condition_q_l168_168190


namespace find_other_number_l168_168592

theorem find_other_number (a b : ℕ) (h₁ : Nat.lcm a b = 3780) (h₂ : Nat.gcd a b = 18) (h₃ : a = 180) : b = 378 := by
  sorry

end find_other_number_l168_168592


namespace rectangle_other_side_l168_168962

theorem rectangle_other_side (A x y : ℝ) (hA : A = 1 / 8) (hx : x = 1 / 2) (hArea : A = x * y) :
    y = 1 / 4 := 
  sorry

end rectangle_other_side_l168_168962


namespace eliana_refill_l168_168767

theorem eliana_refill (total_spent cost_per_refill : ℕ) (h1 : total_spent = 63) (h2 : cost_per_refill = 21) : (total_spent / cost_per_refill) = 3 :=
sorry

end eliana_refill_l168_168767


namespace resulting_solution_percentage_l168_168475

theorem resulting_solution_percentage :
  ∀ (C_init R C_replace : ℚ), 
  C_init = 0.85 → 
  R = 0.6923076923076923 → 
  C_replace = 0.2 → 
  (C_init * (1 - R) + C_replace * R) = 0.4 :=
by
  intros C_init R C_replace hC_init hR hC_replace
  -- Omitted proof here
  sorry

end resulting_solution_percentage_l168_168475


namespace third_divisor_l168_168091

theorem third_divisor (x : ℕ) (h1 : x - 16 = 136) (h2 : ∃ y, y = x - 16) (h3 : 4 ∣ x) (h4 : 6 ∣ x) (h5 : 10 ∣ x) : 19 ∣ x := 
by
  sorry

end third_divisor_l168_168091


namespace vasya_most_points_anya_least_possible_l168_168565

theorem vasya_most_points_anya_least_possible :
  ∃ (A B V : ℕ) (A_score B_score V_score : ℕ),
  A > B ∧ B > V ∧
  A_score = 9 ∧ B_score = 10 ∧ V_score = 11 ∧
  (∃ (words_common_AB words_common_AV words_only_B words_only_V : ℕ),
  words_common_AB = 6 ∧ words_common_AV = 3 ∧ words_only_B = 2 ∧ words_only_V = 4 ∧
  A = words_common_AB + words_common_AV ∧
  B = words_only_B + words_common_AB ∧
  V = words_only_V + words_common_AV ∧
  A_score = words_common_AB + words_common_AV ∧
  B_score = 2 * words_only_B + words_common_AB ∧
  V_score = 2 * words_only_V + words_common_AV) :=
sorry

end vasya_most_points_anya_least_possible_l168_168565


namespace repeating_decimals_for_n_div_18_l168_168137

theorem repeating_decimals_for_n_div_18 :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 20 → (¬ (∃ m : ℕ, m * 18 = n * (2^k * 5^l) ∧ 0 < k ∧ 0 < l)) :=
by
  sorry

end repeating_decimals_for_n_div_18_l168_168137


namespace simplify_expression_l168_168865

theorem simplify_expression :
  15 * (18 / 5) * (-42 / 45) = -50.4 :=
by
  sorry

end simplify_expression_l168_168865


namespace arithmetic_sequence_a4_eq_1_l168_168979

theorem arithmetic_sequence_a4_eq_1 
  (a : ℕ → ℝ)
  (h_arith_seq : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h_pos : ∀ n : ℕ, a n > 0)
  (h_eq : a 2 ^ 2 + 2 * a 2 * a 6 + a 6 ^ 2 - 4 = 0) : 
  a 4 = 1 :=
sorry

end arithmetic_sequence_a4_eq_1_l168_168979


namespace no_infinite_lines_satisfying_conditions_l168_168187

theorem no_infinite_lines_satisfying_conditions :
  ¬ ∃ (l : ℕ → ℝ → ℝ → Prop)
      (k : ℕ → ℝ)
      (a b : ℕ → ℝ),
    (∀ n, l n 1 1) ∧
    (∀ n, k (n + 1) = a n - b n) ∧
    (∀ n, k n * k (n + 1) ≥ 0) := 
sorry

end no_infinite_lines_satisfying_conditions_l168_168187


namespace car_y_speed_l168_168780

noncomputable def carY_average_speed (vX : ℝ) (tY : ℝ) (d : ℝ) : ℝ :=
  d / tY

theorem car_y_speed (vX : ℝ := 35) (tY_min : ℝ := 72) (dX_after_Y : ℝ := 245) :
  carY_average_speed vX (dX_after_Y / vX) dX_after_Y = 35 := 
by
  sorry

end car_y_speed_l168_168780


namespace solve_quadratic_1_solve_quadratic_2_solve_quadratic_3_solve_quadratic_4_solve_quadratic_5_solve_quadratic_6_l168_168322

-- Problem 1: 5x² = 40x
theorem solve_quadratic_1 (x : ℝ) : 5 * x^2 = 40 * x ↔ (x = 0 ∨ x = 8) :=
by sorry

-- Problem 2: 25/9 x² = 100
theorem solve_quadratic_2 (x : ℝ) : (25 / 9) * x^2 = 100 ↔ (x = 6 ∨ x = -6) :=
by sorry

-- Problem 3: 10x = x² + 21
theorem solve_quadratic_3 (x : ℝ) : 10 * x = x^2 + 21 ↔ (x = 7 ∨ x = 3) :=
by sorry

-- Problem 4: x² = 12x + 288
theorem solve_quadratic_4 (x : ℝ) : x^2 = 12 * x + 288 ↔ (x = 24 ∨ x = -12) :=
by sorry

-- Problem 5: x² + 20 1/4 = 11 1/4 x
theorem solve_quadratic_5 (x : ℝ) : x^2 + 81 / 4 = 45 / 4 * x ↔ (x = 9 / 4 ∨ x = 9) :=
by sorry

-- Problem 6: 1/12 x² + 7/12 x = 19
theorem solve_quadratic_6 (x : ℝ) : (1 / 12) * x^2 + (7 / 12) * x = 19 ↔ (x = 12 ∨ x = -19) :=
by sorry

end solve_quadratic_1_solve_quadratic_2_solve_quadratic_3_solve_quadratic_4_solve_quadratic_5_solve_quadratic_6_l168_168322


namespace travel_time_l168_168275

-- Given conditions
def distance_per_hour : ℤ := 27
def distance_to_sfl : ℤ := 81

-- Theorem statement to prove
theorem travel_time (dph : ℤ) (dts : ℤ) (h1 : dph = distance_per_hour) (h2 : dts = distance_to_sfl) : 
  dts / dph = 3 := 
by
  -- immediately helps execute the Lean statement
  sorry

end travel_time_l168_168275


namespace tangent_line_equation_l168_168123

theorem tangent_line_equation (y : ℝ → ℝ) (x : ℝ) (dy_dx : ℝ → ℝ) (tangent_eq : ℝ → ℝ → Prop):
  (∀ x, y x = x^2 + Real.log x) →
  (∀ x, dy_dx x = (deriv y) x) →
  (dy_dx 1 = 3) →
  (tangent_eq x (y x) ↔ (3 * x - y x - 2 = 0)) →
  tangent_eq 1 (y 1) :=
by
  intros y_def dy_dx_def slope_at_1 tangent_line_char
  sorry

end tangent_line_equation_l168_168123


namespace correct_multiplier_l168_168112

theorem correct_multiplier (x : ℕ) 
  (h1 : 137 * 34 + 1233 = 137 * x) : 
  x = 43 := 
by 
  sorry

end correct_multiplier_l168_168112


namespace first_tap_time_l168_168942

-- Define the variables and conditions
variables (T : ℝ)
-- The cistern can be emptied by the second tap in 9 hours
-- Both taps together fill the cistern in 7.2 hours.
def first_tap_fills_cistern_in_time (T : ℝ) :=
  (1 / T) - (1 / 9) = 1 / 7.2

theorem first_tap_time :
  first_tap_fills_cistern_in_time 4 :=
by
  -- now we can use the definition to show the proof
  unfold first_tap_fills_cistern_in_time
  -- directly substitute and show
  sorry

end first_tap_time_l168_168942


namespace initial_crayons_l168_168099

theorem initial_crayons {C : ℕ} (h : C + 12 = 53) : C = 41 :=
by
  -- This is where the proof would go, but we use sorry to skip it.
  sorry

end initial_crayons_l168_168099


namespace simplify_fraction_l168_168399

variable (y b : ℚ)

theorem simplify_fraction : 
  (y+2) / 4 + (5 - 4*y + b) / 3 = (-13*y + 4*b + 26) / 12 := 
by
  sorry

end simplify_fraction_l168_168399


namespace sharks_win_percentage_at_least_ninety_percent_l168_168030

theorem sharks_win_percentage_at_least_ninety_percent (N : ℕ) :
  let initial_games := 3
  let initial_shark_wins := 2
  let total_games := initial_games + N
  let total_shark_wins := initial_shark_wins + N
  total_shark_wins * 10 ≥ total_games * 9 ↔ N ≥ 7 :=
by
  intros
  sorry

end sharks_win_percentage_at_least_ninety_percent_l168_168030


namespace credit_extended_by_automobile_finance_companies_l168_168314

def percentage_of_automobile_installment_credit : ℝ := 0.36
def total_consumer_installment_credit : ℝ := 416.66667
def fraction_extended_by_finance_companies : ℝ := 0.5

theorem credit_extended_by_automobile_finance_companies :
  fraction_extended_by_finance_companies * (percentage_of_automobile_installment_credit * total_consumer_installment_credit) = 75 :=
by
  sorry

end credit_extended_by_automobile_finance_companies_l168_168314


namespace trigonometric_identity_tangent_line_l168_168502

theorem trigonometric_identity_tangent_line 
  (α : ℝ) 
  (h_tan : Real.tan α = 4) 
  : Real.cos α ^ 2 - Real.sin (2 * α) = - 7 / 17 := 
by sorry

end trigonometric_identity_tangent_line_l168_168502


namespace apples_in_basket_l168_168359

noncomputable def total_apples (good_cond: ℕ) (good_ratio: ℝ) := (good_cond : ℝ) / good_ratio

theorem apples_in_basket : total_apples 66 0.88 = 75 :=
by
  sorry

end apples_in_basket_l168_168359


namespace base_of_second_term_l168_168449

theorem base_of_second_term (h : ℕ) (a b c : ℕ) (H1 : h > 0) 
  (H2 : 225 ∣ h) (H3 : 216 ∣ h) 
  (H4 : h = (2^a) * (some_number^b) * (5^c)) 
  (H5 : a + b + c = 8) : some_number = 3 :=
by
  sorry

end base_of_second_term_l168_168449


namespace remainder_sum_first_six_primes_div_seventh_prime_l168_168819

-- Define the first six prime numbers
def firstSixPrimes : List ℕ := [2, 3, 5, 7, 11, 13]

-- Define the sum of the first six prime numbers
def sumOfFirstSixPrimes : ℕ := firstSixPrimes.sum

-- Define the seventh prime number
def seventhPrime : ℕ := 17

-- Proof statement that the remainder of the division is 7
theorem remainder_sum_first_six_primes_div_seventh_prime :
  (sumOfFirstSixPrimes % seventhPrime) = 7 :=
by
  sorry

end remainder_sum_first_six_primes_div_seventh_prime_l168_168819


namespace value_of_abs_m_minus_n_l168_168636

theorem value_of_abs_m_minus_n  (m n : ℝ) (h_eq : ∀ x, (x^2 - 2 * x + m) * (x^2 - 2 * x + n) = 0)
  (h_arith_seq : ∀ x₁ x₂ x₃ x₄ : ℝ, x₁ + x₂ = 2 ∧ x₃ + x₄ = 2 ∧ x₁ = 1 / 4 ∧ x₂ = 3 / 4 ∧ x₃ = 5 / 4 ∧ x₄ = 7 / 4) :
  |m - n| = 1 / 2 :=
by
  sorry

end value_of_abs_m_minus_n_l168_168636


namespace fraction_identity_l168_168768

-- Definitions for conditions
variables (a b : ℚ)

-- The main statement to prove
theorem fraction_identity (h : a/b = 2/5) : (a + b) / b = 7 / 5 :=
by
  sorry

end fraction_identity_l168_168768


namespace total_rowing_and_hiking_l168_168974

def total_campers : ℕ := 80
def morning_rowing : ℕ := 41
def morning_hiking : ℕ := 4
def morning_swimming : ℕ := 15
def afternoon_rowing : ℕ := 26
def afternoon_hiking : ℕ := 8
def afternoon_swimming : ℕ := total_campers - afternoon_rowing - afternoon_hiking - (total_campers - morning_rowing - morning_hiking - morning_swimming)

theorem total_rowing_and_hiking : 
  (morning_rowing + afternoon_rowing) + (morning_hiking + afternoon_hiking) = 79 :=
by
  sorry

end total_rowing_and_hiking_l168_168974


namespace kingfisher_catch_difference_l168_168950

def pelicanFish : Nat := 13
def fishermanFish (K : Nat) : Nat := 3 * (pelicanFish + K)
def fishermanConditionFish : Nat := pelicanFish + 86

theorem kingfisher_catch_difference (K : Nat) (h1 : K > pelicanFish)
  (h2 : fishermanFish K = fishermanConditionFish) :
  K - pelicanFish = 7 := by
  sorry

end kingfisher_catch_difference_l168_168950


namespace calculate_two_times_square_root_squared_l168_168937

theorem calculate_two_times_square_root_squared : 2 * (Real.sqrt 50625) ^ 2 = 101250 := by
  sorry

end calculate_two_times_square_root_squared_l168_168937


namespace problem1_problem2_l168_168016

-- Statement for Question (1)
theorem problem1 (x : ℝ) (h : |x - 1| + x ≥ x + 2) : x ≤ -1 ∨ x ≥ 3 :=
  sorry

-- Statement for Question (2)
theorem problem2 (a : ℝ) (h : ∀ x : ℝ, |x - a| + x ≤ 3 * x → x ≥ 2) : a = 6 :=
  sorry

end problem1_problem2_l168_168016


namespace abcd_hife_value_l168_168085

theorem abcd_hife_value (a b c d e f g h i : ℝ) 
  (h1 : a / b = 1 / 3) 
  (h2 : b / c = 2) 
  (h3 : c / d = 1 / 2) 
  (h4 : d / e = 3) 
  (h5 : e / f = 1 / 10) 
  (h6 : f / g = 3 / 4) 
  (h7 : g / h = 1 / 5) 
  (h8 : h / i = 5) : 
  abcd / hife = 17.28 := sorry

end abcd_hife_value_l168_168085


namespace boat_problem_l168_168893

theorem boat_problem (x n : ℕ) (h1 : n = 7 * x + 5) (h2 : n = 8 * x - 2) :
  n = 54 ∧ x = 7 := by
sorry

end boat_problem_l168_168893


namespace find_nth_term_of_arithmetic_seq_l168_168972

def is_arithmetic_seq (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def is_geometric_progression (a1 a2 a5 : ℝ) :=
  a1 * a5 = a2^2

theorem find_nth_term_of_arithmetic_seq (a : ℕ → ℝ) (d : ℝ) (h_arith : is_arithmetic_seq a d)
    (h_a1 : a 1 = 1) (h_nonzero : d ≠ 0) (h_geom : is_geometric_progression (a 1) (a 2) (a 5)) : 
    ∀ n, a n = 2 * n - 1 :=
by
  sorry

end find_nth_term_of_arithmetic_seq_l168_168972


namespace quadratic_cubic_expression_l168_168689

theorem quadratic_cubic_expression
  (r s : ℝ)
  (h_eq : ∀ x : ℝ, 3 * x^2 - 4 * x - 12 = 0 → x = r ∨ x = s) :
  (9 * r^3 - 9 * s^3) / (r - s) = 52 :=
by 
  sorry

end quadratic_cubic_expression_l168_168689


namespace monthly_earnings_l168_168766

-- Defining the initial conditions and known information
def current_worth : ℝ := 90
def months : ℕ := 5

-- Let I be the initial investment, and E be the earnings per month.

noncomputable def initial_investment (I : ℝ) := I * 3 = current_worth
noncomputable def earned_twice_initial (E : ℝ) (I : ℝ) := E * months = 2 * I

-- Proving the monthly earnings
theorem monthly_earnings (I E : ℝ) (h1 : initial_investment I) (h2 : earned_twice_initial E I) : E = 12 :=
sorry

end monthly_earnings_l168_168766


namespace sin_double_angle_l168_168206

theorem sin_double_angle (α : ℝ) (h1 : α ∈ Set.Ioo (π / 2) π) (h2 : Real.sin α = 4 / 5) : Real.sin (2 * α) = -24 / 25 :=
by
  sorry

end sin_double_angle_l168_168206


namespace multiple_of_C_share_l168_168028

noncomputable def find_multiple (A B C : ℕ) (total : ℕ) (mult : ℕ) (h1 : 4 * A = mult * C) (h2 : 5 * B = mult * C) (h3 : A + B + C = total) : ℕ :=
  mult

theorem multiple_of_C_share (A B : ℕ) (h1 : 4 * A = 10 * 160) (h2 : 5 * B = 10 * 160) (h3 : A + B + 160 = 880) : find_multiple A B 160 880 10 h1 h2 h3 = 10 :=
by
  sorry

end multiple_of_C_share_l168_168028


namespace find_x_l168_168869

theorem find_x (x : ℕ) (h : x * 5^4 = 75625) : x = 121 :=
by
  sorry

end find_x_l168_168869


namespace mingi_math_test_total_pages_l168_168765

theorem mingi_math_test_total_pages (first_page last_page : Nat) (h_first_page : first_page = 8) (h_last_page : last_page = 21) : first_page <= last_page -> ((last_page - first_page + 1) = 14) :=
by
  sorry

end mingi_math_test_total_pages_l168_168765


namespace boys_in_fifth_grade_l168_168437

theorem boys_in_fifth_grade (T S : ℕ) (percent_boys_soccer : ℝ) (girls_not_playing_soccer : ℕ) 
    (hT : T = 420) (hS : S = 250) (h_percent : percent_boys_soccer = 0.86) 
    (h_girls_not_playing_soccer : girls_not_playing_soccer = 65) : 
    ∃ B : ℕ, B = 320 :=
by
  -- We don't need to provide the proof details here
  sorry

end boys_in_fifth_grade_l168_168437


namespace percentage_discount_l168_168179

theorem percentage_discount (P D: ℝ) 
  (sale_price: P * (100 - D) / 100 = 78.2)
  (final_price_increase: 78.2 * 1.25 = P - 5.75):
  D = 24.44 :=
by
  sorry

end percentage_discount_l168_168179


namespace combined_share_b_d_l168_168543

-- Definitions for the amounts shared between the children
def total_amount : ℝ := 15800
def share_a_plus_c : ℝ := 7022.222222222222

-- The goal is to prove that the combined share of B and D is 8777.777777777778
theorem combined_share_b_d :
  ∃ B D : ℝ, (B + D = total_amount - share_a_plus_c) :=
by
  sorry

end combined_share_b_d_l168_168543


namespace problem1_problem2_l168_168339

-- Problem 1: Prove the expression
theorem problem1 (a b : ℝ) : 
  2 * a * (a - 2 * b) - (2 * a - b) ^ 2 = -2 * a ^ 2 - b ^ 2 := 
sorry

-- Problem 2: Prove the solution to the equation
theorem problem2 (x : ℝ) (h : (x - 1) ^ 3 - 3 = 3 / 8) : 
  x = 5 / 2 := 
sorry

end problem1_problem2_l168_168339


namespace cost_per_minute_l168_168270

theorem cost_per_minute (monthly_fee cost total_bill : ℝ) (minutes : ℕ) :
  monthly_fee = 2 ∧ total_bill = 23.36 ∧ minutes = 178 → 
  cost = (total_bill - monthly_fee) / minutes → 
  cost = 0.12 :=
by
  intros h1 h2
  sorry

end cost_per_minute_l168_168270


namespace minimal_functions_l168_168416

open Int

theorem minimal_functions (f : ℤ → ℤ) (c : ℤ) :
  (∀ x, f (x + 2017) = f x) ∧
  (∀ x y, (f (f x + f y + 1) - f (f x + f y)) % 2017 = c) →
  (c = 1 ∨ c = 2016 ∨ c = 1008 ∨ c = 1009) :=
by
  sorry

end minimal_functions_l168_168416


namespace part1_part2_l168_168596

theorem part1 (x y : ℝ) (h1 : (1, 0) = (x, y)) (h2 : (0, 2) = (x, y)): 
    ∃ k b : ℝ, k = -2 ∧ b = 2 ∧ y = k * x + b := 
by 
  sorry

theorem part2 (m n : ℝ) (h : n = -2 * m + 2) (hm : -2 < m ∧ m ≤ 3):
    -4 ≤ n ∧ n < 6 := 
by 
  sorry

end part1_part2_l168_168596


namespace min_xy_min_x_plus_y_l168_168930

theorem min_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y - x * y = 0) : xy ≥ 64 := 
sorry

theorem min_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y - x * y = 0) : x + y ≥ 18 := 
sorry

end min_xy_min_x_plus_y_l168_168930


namespace f_neg_two_l168_168995

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^5 + b * x - c / x + 2

theorem f_neg_two (a b c : ℝ) (h : f a b c 2 = 4) : f a b c (-2) = 0 :=
sorry

end f_neg_two_l168_168995


namespace average_speed_car_l168_168271

theorem average_speed_car (speed_first_hour ground_speed_headwind speed_second_hour : ℝ) (time_first_hour time_second_hour : ℝ) (h1 : speed_first_hour = 90) (h2 : ground_speed_headwind = 10) (h3 : speed_second_hour = 55) (h4 : time_first_hour = 1) (h5 : time_second_hour = 1) : 
(speed_first_hour + ground_speed_headwind) * time_first_hour + speed_second_hour * time_second_hour / (time_first_hour + time_second_hour) = 77.5 :=
sorry

end average_speed_car_l168_168271


namespace arithmetic_seq_a11_l168_168761

theorem arithmetic_seq_a11 (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : S 21 = 105) : a 11 = 5 :=
sorry

end arithmetic_seq_a11_l168_168761


namespace new_trailers_added_l168_168049

theorem new_trailers_added :
  let initial_trailers := 25
  let initial_average_age := 15
  let years_passed := 3
  let current_average_age := 12
  let total_initial_age := initial_trailers * (initial_average_age + years_passed)
  ∀ n : Nat, 
    ((25 * 18) + (n * 3) = (25 + n) * 12) →
    n = 17 := 
by
  intros
  sorry

end new_trailers_added_l168_168049


namespace max_sum_of_abcd_l168_168031

noncomputable def abcd_product (a b c d : ℕ) : ℕ := a * b * c * d

theorem max_sum_of_abcd (a b c d : ℕ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
    (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d) (h7 : abcd_product a b c d = 1995) : 
    a + b + c + d ≤ 142 :=
sorry

end max_sum_of_abcd_l168_168031


namespace rectangle_dimensions_l168_168820

theorem rectangle_dimensions (x : ℝ) (h : 4 * x * x = 120) : x = Real.sqrt 30 ∧ 4 * x = 4 * Real.sqrt 30 :=
by
  sorry

end rectangle_dimensions_l168_168820


namespace num_students_in_section_A_l168_168242

def avg_weight (total_weight : ℕ) (total_students : ℕ) : ℕ :=
  total_weight / total_students

variables (x : ℕ) -- number of students in section A
variables (weight_A : ℕ := 40 * x) -- total weight of section A
variables (students_B : ℕ := 20)
variables (weight_B : ℕ := 20 * 35) -- total weight of section B
variables (total_weight : ℕ := weight_A + weight_B) -- total weight of the whole class
variables (total_students : ℕ := x + students_B) -- total number of students in the class
variables (avg_weight_class : ℕ := avg_weight total_weight total_students)

theorem num_students_in_section_A :
  avg_weight_class = 38 → x = 30 :=
by
-- The proof will go here
sorry

end num_students_in_section_A_l168_168242


namespace high_card_point_value_l168_168401

theorem high_card_point_value :
  ∀ (H L : ℕ), 
  (L = 1) →
  ∀ (high low total_points : ℕ), 
  (total_points = 5) →
  (high + (L + L + L) = total_points) →
  high = 2 :=
by
  intros
  sorry

end high_card_point_value_l168_168401


namespace lock_code_difference_l168_168027

theorem lock_code_difference :
  ∃ A B C D, A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
             (A = 4 ∧ B = 2 * C ∧ C = D) ∨
             (A = 9 ∧ B = 3 * C ∧ C = D) ∧
             (A * 100 + B * 10 + C - (D * 100 + (2 * D) * 10 + D)) = 541 :=
sorry

end lock_code_difference_l168_168027


namespace denote_loss_of_300_dollars_l168_168536

-- Define the concept of financial transactions
def denote_gain (amount : Int) : Int := amount
def denote_loss (amount : Int) : Int := -amount

-- The condition given in the problem
def earn_500_dollars_is_500 := denote_gain 500 = 500

-- The assertion we need to prove
theorem denote_loss_of_300_dollars : denote_loss 300 = -300 := 
by 
  sorry

end denote_loss_of_300_dollars_l168_168536


namespace min_value_expr_l168_168432

open Real

theorem min_value_expr (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x + 1/y) * (x + 1/y - 2020) + (y + 1/x) * (y + 1/x - 2020) ≥ -2040200 :=
by
  sorry

end min_value_expr_l168_168432


namespace pesto_calculation_l168_168260

def basil_needed_per_pesto : ℕ := 4
def basil_harvest_per_week : ℕ := 16
def weeks : ℕ := 8
def total_basil_harvested : ℕ := basil_harvest_per_week * weeks
def total_pesto_possible : ℕ := total_basil_harvested / basil_needed_per_pesto

theorem pesto_calculation :
  total_pesto_possible = 32 :=
by
  sorry

end pesto_calculation_l168_168260


namespace eval_at_d_eq_4_l168_168649

theorem eval_at_d_eq_4 : ((4: ℕ) ^ 4 - (4: ℕ) * ((4: ℕ) - 2) ^ 4) ^ 4 = 136048896 :=
by
  sorry

end eval_at_d_eq_4_l168_168649


namespace general_term_formula_is_not_element_l168_168577

theorem general_term_formula (a : ℕ → ℤ) (h1 : a 1 = 2) (h17 : a 17 = 66) :
  (∀ n, a n = 4 * n - 2) :=
by
  sorry

theorem is_not_element (a : ℕ → ℤ) (h : ∀ n, a n = 4 * n - 2) :
  ¬ (∃ n : ℕ, a n = 88) :=
by
  sorry

end general_term_formula_is_not_element_l168_168577


namespace A_wins_if_N_is_perfect_square_l168_168679

noncomputable def player_A_can_always_win (N : ℕ) : Prop :=
  ∀ (B_moves : ℕ → ℕ), ∃ (A_moves : ℕ → ℕ), A_moves 0 = N ∧
  (∀ n, B_moves n = 0 ∨ (A_moves n ∣ B_moves (n + 1) ∨ B_moves (n + 1) ∣ A_moves n))

theorem A_wins_if_N_is_perfect_square :
  ∀ N : ℕ, player_A_can_always_win N ↔ ∃ n : ℕ, N = n * n := sorry

end A_wins_if_N_is_perfect_square_l168_168679


namespace total_time_l168_168478

def time_to_eat_cereal (rate1 rate2 rate3 : ℚ) (amount : ℚ) : ℚ :=
  let combined_rate := rate1 + rate2 + rate3
  amount / combined_rate

theorem total_time (rate1 rate2 rate3 : ℚ) (amount : ℚ) 
  (h1 : rate1 = 1 / 15)
  (h2 : rate2 = 1 / 20)
  (h3 : rate3 = 1 / 30)
  (h4 : amount = 4) : 
  time_to_eat_cereal rate1 rate2 rate3 amount = 80 / 3 := 
by 
  rw [time_to_eat_cereal, h1, h2, h3, h4]
  sorry

end total_time_l168_168478


namespace chocolates_problem_l168_168313

theorem chocolates_problem 
  (N : ℕ)
  (h1 : ∃ R C : ℕ, N = R * C)
  (h2 : ∃ r1, r1 = 3 * C - 1)
  (h3 : ∃ c1, c1 = R * 5 - 1)
  (h4 : ∃ r2 c2 : ℕ, r2 = 3 ∧ c2 = 5 ∧ r2 * c2 = r1 - 1)
  (h5 : N = 3 * (3 * C - 1))
  (h6 : N / 3 = (12 * 5) / 3) : 
  N = 60 ∧ (N - (3 * C - 1)) = 25 :=
by 
  sorry

end chocolates_problem_l168_168313


namespace tubs_from_usual_vendor_l168_168080

def total_tubs_needed : Nat := 100
def tubs_in_storage : Nat := 20
def fraction_from_new_vendor : Rat := 1 / 4

theorem tubs_from_usual_vendor :
  let remaining_tubs := total_tubs_needed - tubs_in_storage
  let tubs_from_new_vendor := remaining_tubs * fraction_from_new_vendor
  let tubs_from_usual_vendor := remaining_tubs - tubs_from_new_vendor
  tubs_from_usual_vendor = 60 :=
by
  intro remaining_tubs tubs_from_new_vendor
  exact sorry

end tubs_from_usual_vendor_l168_168080


namespace math_problem_l168_168473

theorem math_problem :
  let numerator := (15^4 + 400) * (30^4 + 400) * (45^4 + 400) * (60^4 + 400) * (75^4 + 400)
  let denominator := (5^4 + 400) * (20^4 + 400) * (35^4 + 400) * (50^4 + 400) * (65^4 + 400)
  numerator / denominator = 301 :=
by 
  sorry

end math_problem_l168_168473


namespace valid_k_l168_168293

theorem valid_k (k : ℕ) (h_pos : k ≥ 1) (h : 10^k - 1 = 9 * k^2) : k = 1 := by
  sorry

end valid_k_l168_168293


namespace polynomial_expansion_sum_constants_l168_168331

theorem polynomial_expansion_sum_constants :
  ∃ (A B C D : ℤ), ((x - 3) * (4 * x ^ 2 + 2 * x - 7) = A * x ^ 3 + B * x ^ 2 + C * x + D) → A + B + C + D = 2 := 
by
  sorry

end polynomial_expansion_sum_constants_l168_168331


namespace uncle_jerry_total_tomatoes_l168_168456

def day1_tomatoes : ℕ := 120
def day2_tomatoes : ℕ := day1_tomatoes + 50
def day3_tomatoes : ℕ := 2 * day2_tomatoes
def total_tomatoes : ℕ := day1_tomatoes + day2_tomatoes + day3_tomatoes

theorem uncle_jerry_total_tomatoes : total_tomatoes = 630 := by
  sorry

end uncle_jerry_total_tomatoes_l168_168456


namespace smallest_sum_l168_168212

theorem smallest_sum (a b : ℕ) (h1 : 3^8 * 5^2 = a^b) (h2 : 0 < a) (h3 : 0 < b) : a + b = 407 :=
sorry

end smallest_sum_l168_168212


namespace atomic_weight_O_l168_168981

-- We define the atomic weights of sodium and chlorine
def atomic_weight_Na : ℝ := 22.99
def atomic_weight_Cl : ℝ := 35.45

-- We define the molecular weight of the compound
def molecular_weight_compound : ℝ := 74.0

-- We want to prove that the atomic weight of oxygen (O) is 15.56 given the above conditions
theorem atomic_weight_O : 
  (molecular_weight_compound = atomic_weight_Na + atomic_weight_Cl + w -> w = 15.56) :=
by
  sorry

end atomic_weight_O_l168_168981


namespace square_root_condition_l168_168868

-- Define the condition under which the square root of an expression is defined
def is_square_root_defined (x : ℝ) : Prop := (x + 3) ≥ 0

-- Prove that the condition for the square root of x + 3 to be defined is x ≥ -3
theorem square_root_condition (x : ℝ) : is_square_root_defined x ↔ x ≥ -3 := 
sorry

end square_root_condition_l168_168868


namespace power_function_solution_l168_168799

theorem power_function_solution (f : ℝ → ℝ) (alpha : ℝ)
  (h₀ : ∀ x, f x = x ^ alpha)
  (h₁ : f (1 / 8) = 2) :
  f (-1 / 8) = -2 :=
sorry

end power_function_solution_l168_168799


namespace jimin_shared_fruits_total_l168_168615

-- Define the quantities given in the conditions
def persimmons : ℕ := 2
def apples : ℕ := 7

-- State the theorem to be proved
theorem jimin_shared_fruits_total : persimmons + apples = 9 := by
  sorry

end jimin_shared_fruits_total_l168_168615


namespace find_point_M_l168_168191

def parabola (x y : ℝ) := x^2 = 4 * y
def focus_dist (M : ℝ × ℝ) := dist M (0, 1) = 2
def point_on_parabola (M : ℝ × ℝ) := parabola M.1 M.2

theorem find_point_M (M : ℝ × ℝ) (h1 : point_on_parabola M) (h2 : focus_dist M) :
  M = (2, 1) ∨ M = (-2, 1) := by
  sorry

end find_point_M_l168_168191


namespace number_of_solid_figures_is_4_l168_168357

def is_solid_figure (shape : String) : Bool :=
  shape = "cone" ∨ shape = "cuboid" ∨ shape = "sphere" ∨ shape = "triangular prism"

def shapes : List String :=
  ["circle", "square", "cone", "cuboid", "line segment", "sphere", "triangular prism", "right-angled triangle"]

def number_of_solid_figures : Nat :=
  (shapes.filter is_solid_figure).length

theorem number_of_solid_figures_is_4 : number_of_solid_figures = 4 :=
  by sorry

end number_of_solid_figures_is_4_l168_168357


namespace function_single_intersection_l168_168900

theorem function_single_intersection (a : ℝ) : 
  (∃ x : ℝ, ax^2 - x + 1 = 0 ∧ ∀ y : ℝ, (ax^2 - x + 1 = 0 → y = x)) ↔ (a = 0 ∨ a = 1/4) :=
sorry

end function_single_intersection_l168_168900


namespace juanitas_dessert_cost_l168_168954

theorem juanitas_dessert_cost :
  let brownie_cost := 2.50
  let ice_cream_cost := 1.00
  let syrup_cost := 0.50
  let nuts_cost := 1.50
  let num_scoops_ice_cream := 2
  let num_syrups := 2
  let total_cost := brownie_cost + num_scoops_ice_cream * ice_cream_cost + num_syrups * syrup_cost + nuts_cost
  total_cost = 7.00 :=
by
  sorry

end juanitas_dessert_cost_l168_168954


namespace geometric_sequence_sum_l168_168718

theorem geometric_sequence_sum (k : ℕ) (h1 : a_1 = 1) (h2 : a_k = 243) (h3 : q = 3) : S_k = 364 := 
by 
  -- Sorry is used here to skip the proof
  sorry

end geometric_sequence_sum_l168_168718


namespace roof_problem_l168_168724

theorem roof_problem (w l : ℝ) (h1 : l = 4 * w) (h2 : l * w = 900) : l - w = 45 := 
by
  sorry

end roof_problem_l168_168724


namespace johns_videos_weekly_minutes_l168_168840

theorem johns_videos_weekly_minutes (daily_minutes weekly_minutes : ℕ) (short_video_length long_factor: ℕ) (short_videos_per_day long_videos_per_day days : ℕ)
  (h1 : daily_minutes = short_videos_per_day * short_video_length + long_videos_per_day * (long_factor * short_video_length))
  (h2 : weekly_minutes = daily_minutes * days)
  (h_short_videos_per_day : short_videos_per_day = 2)
  (h_long_videos_per_day : long_videos_per_day = 1)
  (h_short_video_length : short_video_length = 2)
  (h_long_factor : long_factor = 6)
  (h_weekly_minutes : weekly_minutes = 112):
  days = 7 :=
by
  sorry

end johns_videos_weekly_minutes_l168_168840


namespace probability_third_draw_first_class_expected_value_first_class_in_10_draws_l168_168576

-- Define the problem with products
structure Products where
  total : ℕ
  first_class : ℕ
  second_class : ℕ

-- Given products configuration
def products : Products := { total := 5, first_class := 3, second_class := 2 }

-- Probability calculation without replacement
-- Define the event of drawing
def draw_without_replacement (p : Products) (draws : ℕ) (desired_event : ℕ -> Bool) : ℚ := 
  if draws = 3 ∧ desired_event 3 ∧ ¬ desired_event 1 ∧ ¬ desired_event 2 then
    (2 / 5) * ((1 : ℚ) / 4) * (3 / 3)
  else 
    0

-- Define desired_event for the specific problem
def desired_event (n : ℕ) : Bool := 
  match n with
  | 3 => true
  | _ => false

-- The first problem's proof statement
theorem probability_third_draw_first_class : draw_without_replacement products 3 desired_event = 1 / 10 := sorry

-- Expected value calculation with replacement
-- Binomial distribution to find expected value
def expected_value_with_replacement (p : Products) (draws : ℕ) : ℚ :=
  draws * (p.first_class / p.total)

-- The second problem's proof statement
theorem expected_value_first_class_in_10_draws : expected_value_with_replacement products 10 = 6 := sorry

end probability_third_draw_first_class_expected_value_first_class_in_10_draws_l168_168576


namespace branches_and_ornaments_l168_168029

def numberOfBranchesAndOrnaments (b t : ℕ) : Prop :=
  (b = t - 1) ∧ (2 * b = t - 1)

theorem branches_and_ornaments : ∃ (b t : ℕ), numberOfBranchesAndOrnaments b t ∧ b = 3 ∧ t = 4 :=
by
  sorry

end branches_and_ornaments_l168_168029


namespace jenny_questions_wrong_l168_168057

variable (j k l m : ℕ)

theorem jenny_questions_wrong
  (h1 : j + k = l + m)
  (h2 : j + m = k + l + 6)
  (h3 : l = 7) : j = 10 := by
  sorry

end jenny_questions_wrong_l168_168057
