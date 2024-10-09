import Mathlib

namespace find_weight_of_a_l1410_141041

-- Define the weights
variables (a b c d e : ℝ)

-- Given conditions
def condition1 := (a + b + c) / 3 = 50
def condition2 := (a + b + c + d) / 4 = 53
def condition3 := (b + c + d + e) / 4 = 51
def condition4 := e = d + 3

-- Proof goal
theorem find_weight_of_a : condition1 a b c → condition2 a b c d → condition3 b c d e → condition4 d e → a = 73 :=
by
  intros h1 h2 h3 h4
  sorry

end find_weight_of_a_l1410_141041


namespace cyclist_C_speed_l1410_141028

variables (c d : ℕ) -- Speeds of cyclists C and D in mph
variables (d_eq : d = c + 6) -- Cyclist D travels 6 mph faster than cyclist C
variables (h1 : 80 = 65 + 15) -- Total distance from X to Y and back to the meet point
variables (same_time : 65 / c = 95 / d) -- Equating the travel times of both cyclists

theorem cyclist_C_speed : c = 13 :=
by
  sorry -- Proof is omitted

end cyclist_C_speed_l1410_141028


namespace complex_identity_l1410_141069

theorem complex_identity (α β : ℝ) (h : Complex.exp (Complex.I * α) + Complex.exp (Complex.I * β) = Complex.mk (-1 / 3) (5 / 8)) :
  Complex.exp (-Complex.I * α) + Complex.exp (-Complex.I * β) = Complex.mk (-1 / 3) (-5 / 8) :=
by
  sorry

end complex_identity_l1410_141069


namespace evaluate_expression_l1410_141053

theorem evaluate_expression (a : ℕ) (h : a = 2) : (7 * a ^ 2 - 10 * a + 3) * (3 * a - 4) = 22 :=
by
  -- Here would be the proof which is omitted as per instructions
  sorry

end evaluate_expression_l1410_141053


namespace range_of_a_l1410_141023

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h_def : ∀ x, f x = x^2 - 2 * a * x + a^2 - 1) 
(h_sol : ∀ x, f (f x) ≥ 0) : a ≤ -2 :=
sorry

end range_of_a_l1410_141023


namespace largest_integer_x_l1410_141001

theorem largest_integer_x (x : ℤ) : 
  (0.2 : ℝ) < (x : ℝ) / 7 ∧ (x : ℝ) / 7 < (7 : ℝ) / 12 → x = 4 :=
sorry

end largest_integer_x_l1410_141001


namespace tan3theta_l1410_141034

theorem tan3theta (theta : ℝ) (h : Real.tan theta = 3) : Real.tan (3 * theta) = 9 / 13 := 
by
  sorry

end tan3theta_l1410_141034


namespace problem_solution_l1410_141091

variable (α : ℝ)
variable (h : Real.cos α = 1 / 5)

theorem problem_solution : Real.cos (2 * α - 2017 * Real.pi) = 23 / 25 := by
  sorry

end problem_solution_l1410_141091


namespace average_weight_of_24_boys_l1410_141013

theorem average_weight_of_24_boys (A : ℝ) : 
  (24 * A + 8 * 45.15) / 32 = 48.975 → A = 50.25 :=
by
  intro h
  sorry

end average_weight_of_24_boys_l1410_141013


namespace comparison_of_neg_square_roots_l1410_141030

noncomputable def compare_square_roots : Prop :=
  -2 * Real.sqrt 11 > -3 * Real.sqrt 5

theorem comparison_of_neg_square_roots : compare_square_roots :=
by
  -- Omitting the proof details
  sorry

end comparison_of_neg_square_roots_l1410_141030


namespace games_bought_l1410_141047

def initial_money : ℕ := 35
def spent_money : ℕ := 7
def cost_per_game : ℕ := 4

theorem games_bought : (initial_money - spent_money) / cost_per_game = 7 := by
  sorry

end games_bought_l1410_141047


namespace unknown_number_value_l1410_141005

theorem unknown_number_value (a : ℕ) (n : ℕ) 
  (h1 : a = 105) 
  (h2 : a ^ 3 = 21 * n * 45 * 49) : 
  n = 75 :=
sorry

end unknown_number_value_l1410_141005


namespace find_min_value_c_l1410_141031

theorem find_min_value_c (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 2010) :
  (∃ x y : ℤ, 3 * x + y = 3005 ∧ y = abs (x - a) + abs (x - 2 * b) + abs (x - c) ∧
   (∀ x' y' : ℤ, 3 * x' + y' = 3005 → y' = abs (x' - a) + abs (x' - 2 * b) + abs (x' - c) → x = x' ∧ y = y')) →
  c ≥ 1014 :=
by
  sorry

end find_min_value_c_l1410_141031


namespace necessary_and_sufficient_conditions_l1410_141061

-- Definitions for sets A and B
def U : Set (ℝ × ℝ) := {p | true}

def A (m : ℝ) : Set (ℝ × ℝ) := {p | 2 * p.1 - p.2 + m > 0}

def B (n : ℝ) : Set (ℝ × ℝ) := {p | p.1 + p.2 - n ≤ 0}

-- Given point P(2, 3)
def P : ℝ × ℝ := (2, 3)

-- Complement of B
def B_complement (n : ℝ) : Set (ℝ × ℝ) := {p | p.1 + p.2 - n > 0}

-- Intersection of A and complement of B
def A_inter_B_complement (m n : ℝ) : Set (ℝ × ℝ) := A m ∩ B_complement n

-- Theorem stating the necessary and sufficient conditions for P to belong to A ∩ (complement of B)
theorem necessary_and_sufficient_conditions (m n : ℝ) : 
  P ∈ A_inter_B_complement m n ↔ m > -1 ∧ n < 5 :=
sorry

end necessary_and_sufficient_conditions_l1410_141061


namespace range_of_f_l1410_141080

noncomputable def f (x : ℝ) : ℝ := - (2 / (x - 1))

theorem range_of_f :
  {y : ℝ | ∃ x : ℝ, (0 ≤ x ∧ x < 1 ∨ 1 < x ∧ x ≤ 2) ∧ f x = y} = 
  {y : ℝ | y ≤ -2 ∨ 2 ≤ y} :=
by
  sorry

end range_of_f_l1410_141080


namespace find_x_l1410_141037

theorem find_x (p q r s x : ℚ) (hpq : p ≠ q) (hq0 : q ≠ 0) 
    (h : (p + x) / (q - x) = r / s) 
    (hp : p = 3) (hq : q = 5) (hr : r = 7) (hs : s = 9) : 
    x = 1/2 :=
by {
  sorry
}

end find_x_l1410_141037


namespace smallest_four_digit_in_pascals_triangle_l1410_141004

theorem smallest_four_digit_in_pascals_triangle : ∃ n, ∃ k, k ≤ n ∧ 1000 ≤ Nat.choose n k :=
sorry

end smallest_four_digit_in_pascals_triangle_l1410_141004


namespace Mike_onions_grew_l1410_141064

-- Define the data:
variables (nancy_onions dan_onions total_onions mike_onions : ℕ)

-- Conditions:
axiom Nancy_onions_grew : nancy_onions = 2
axiom Dan_onions_grew : dan_onions = 9
axiom Total_onions_grew : total_onions = 15

-- Theorem to prove:
theorem Mike_onions_grew (h : total_onions = nancy_onions + dan_onions + mike_onions) : mike_onions = 4 :=
by
  -- The proof is not provided, so we use sorry:
  sorry

end Mike_onions_grew_l1410_141064


namespace multiplication_by_9_l1410_141084

theorem multiplication_by_9 (n : ℕ) (h1 : n < 10) : 9 * n = 10 * (n - 1) + (10 - n) := 
sorry

end multiplication_by_9_l1410_141084


namespace find_p_minus_q_l1410_141066

theorem find_p_minus_q (p q : ℝ) (h : ∀ x, x^2 - 6 * x + q = 0 ↔ (x - p)^2 = 7) : p - q = 1 :=
sorry

end find_p_minus_q_l1410_141066


namespace find_x_l1410_141074

theorem find_x (x : ℝ) (h_pos : x > 0) (h_eq : x * (⌊x⌋) = 132) : x = 12 := sorry

end find_x_l1410_141074


namespace total_recruits_211_l1410_141071

theorem total_recruits_211 (P N D : ℕ) (total : ℕ) 
  (h1 : P = 50) 
  (h2 : N = 100) 
  (h3 : D = 170) 
  (h4 : ∃ (x y : ℕ), (x = 4 * y ∨ y = 4 * x) ∧ 
                      ((x, P) = (y, N) ∨ (x, N) = (y, D) ∨ (x, P) = (y, D))) :
  total = 211 :=
by
  sorry

end total_recruits_211_l1410_141071


namespace perfect_square_of_factorials_l1410_141078

open Nat

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem perfect_square_of_factorials :
  let E1 := factorial 98 * factorial 99
  let E2 := factorial 98 * factorial 100
  let E3 := factorial 99 * factorial 100
  let E4 := factorial 99 * factorial 101
  let E5 := factorial 100 * factorial 101
  is_perfect_square E3 :=
by
  -- definition of E1, E2, E3, E4, E5 as expressions given conditions
  let E1 := factorial 98 * factorial 99
  let E2 := factorial 98 * factorial 100
  let E3 := factorial 99 * factorial 100
  let E4 := factorial 99 * factorial 101
  let E5 := factorial 100 * factorial 101
  
  -- specify that E3 is the perfect square
  show is_perfect_square E3

  sorry

end perfect_square_of_factorials_l1410_141078


namespace inequality_solution_set_l1410_141090

theorem inequality_solution_set (x : ℝ) : (x^2 + x) / (2*x - 1) ≤ 1 ↔ x < 1 / 2 := 
sorry

end inequality_solution_set_l1410_141090


namespace solve_for_a_l1410_141024

theorem solve_for_a (x : ℤ) (a : ℤ) (h : 3 * x + 2 * a + 1 = 2) (hx : x = -1) : a = 2 :=
by
  sorry

end solve_for_a_l1410_141024


namespace marbles_difference_l1410_141095

theorem marbles_difference : 10 - 8 = 2 :=
by
  sorry

end marbles_difference_l1410_141095


namespace find_cost_price_l1410_141012

-- Conditions
def initial_cost_price (C : ℝ) : Prop :=
  let SP := 1.07 * C
  let NCP := 0.92 * C
  let NSP := SP - 3
  NSP = 1.0304 * C

-- The problem is to prove the initial cost price C given the conditions
theorem find_cost_price (C : ℝ) (h : initial_cost_price C) : C = 75.7575 := 
  sorry

end find_cost_price_l1410_141012


namespace common_difference_of_arithmetic_seq_l1410_141007

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a n = a 0 + n * d

theorem common_difference_of_arithmetic_seq :
  ∀ (a : ℕ → ℝ) (d : ℝ),
  arithmetic_sequence a d →
  (a 4 + a 8 = 10) →
  (a 10 = 6) →
  d = 1 / 4 :=
by
  intros a d h_seq h1 h2
  sorry

end common_difference_of_arithmetic_seq_l1410_141007


namespace dress_cost_l1410_141087

theorem dress_cost (x : ℝ) 
  (h1 : 30 * x = 10 + x) 
  (h2 : 3 * ((10 + x) / 30) = x) : 
  x = 10 / 9 :=
by
  sorry

end dress_cost_l1410_141087


namespace calculate_students_l1410_141062

noncomputable def handshakes (m n : ℕ) : ℕ :=
  1/2 * (4 * 3 + 5 * (2 * (m - 2) + 2 * (n - 2)) + 8 * (m - 2) * (n - 2))

theorem calculate_students (m n : ℕ) (h_m : 3 ≤ m) (h_n : 3 ≤ n) (h_handshakes : handshakes m n = 1020) : m * n = 140 :=
by
  sorry

end calculate_students_l1410_141062


namespace larger_pie_flour_amount_l1410_141085

variable (p1 : ℕ) (f1 : ℚ) (p2 : ℕ) (f2 : ℚ)

def prepared_pie_crusts (p1 p2 : ℕ) (f1 : ℚ) (f2 : ℚ) : Prop :=
  p1 * f1 = p2 * f2

theorem larger_pie_flour_amount (h : prepared_pie_crusts 40 25 (1/8) f2) : f2 = 1/5 :=
by
  sorry

end larger_pie_flour_amount_l1410_141085


namespace sector_area_proof_l1410_141079

-- Define the sector with its characteristics
structure sector :=
  (r : ℝ)            -- radius
  (theta : ℝ)        -- central angle

-- Given conditions
def sector_example : sector := {r := 1, theta := 2}

-- Definition of perimeter for a sector
def perimeter (sec : sector) : ℝ :=
  2 * sec.r + sec.theta * sec.r

-- Definition of area for a sector
def area (sec : sector) : ℝ :=
  0.5 * sec.r * (sec.theta * sec.r)

-- Theorem statement based on the problem statement
theorem sector_area_proof (sec : sector) (h1 : perimeter sec = 4) (h2 : sec.theta = 2) : area sec = 1 := 
  sorry

end sector_area_proof_l1410_141079


namespace find_b_l1410_141096

theorem find_b (b : ℝ) (h : ∃ x : ℝ, x^2 + b*x - 35 = 0 ∧ x = -5) : b = -2 :=
by
  sorry

end find_b_l1410_141096


namespace find_scalars_l1410_141019

def M : Matrix (Fin 2) (Fin 2) ℤ := ![![2, 7], ![-3, -1]]
def M_squared : Matrix (Fin 2) (Fin 2) ℤ := ![![-17, 7], ![-3, -20]]
def I : Matrix (Fin 2) (Fin 2) ℤ := 1

theorem find_scalars :
  ∃ p q : ℤ, M_squared = p • M + q • I ∧ (p, q) = (1, -19) := sorry

end find_scalars_l1410_141019


namespace total_number_of_sweets_l1410_141065

theorem total_number_of_sweets (num_crates : ℕ) (sweets_per_crate : ℕ) (total_sweets : ℕ) 
  (h1 : num_crates = 4) (h2 : sweets_per_crate = 16) : total_sweets = 64 := by
  sorry

end total_number_of_sweets_l1410_141065


namespace initial_minutes_planA_equivalence_l1410_141097

-- Conditions translated into Lean:
variable (x : ℝ)

-- Definitions for costs
def planA_cost_12 : ℝ := 0.60 + 0.06 * (12 - x)
def planB_cost_12 : ℝ := 0.08 * 12

-- Theorem we want to prove
theorem initial_minutes_planA_equivalence :
  (planA_cost_12 x = planB_cost_12) → x = 6 :=
by
  intro h
  -- complete proof is skipped with sorry
  sorry

end initial_minutes_planA_equivalence_l1410_141097


namespace sufficient_but_not_necessary_condition_l1410_141042

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a = 1 / 8) → (∀ x : ℝ, x > 0 → 2 * x + a / x ≥ 1) :=
by sorry

end sufficient_but_not_necessary_condition_l1410_141042


namespace fraction_product_sum_l1410_141027

theorem fraction_product_sum :
  (1/3) * (5/6) * (3/7) + (1/4) * (1/8) = 101/672 :=
by
  sorry

end fraction_product_sum_l1410_141027


namespace smallest_five_digit_in_pascals_triangle_l1410_141092

/-- In Pascal's triangle, the smallest five-digit number is 10000. -/
theorem smallest_five_digit_in_pascals_triangle : 
  ∃ (n k : ℕ), (10000 = Nat.choose n k) ∧ (∀ m l : ℕ, Nat.choose m l < 10000) → (n > m) := 
sorry

end smallest_five_digit_in_pascals_triangle_l1410_141092


namespace sqrt_x_minus_1_meaningful_l1410_141002

theorem sqrt_x_minus_1_meaningful (x : ℝ) : (x - 1 ≥ 0) ↔ (x ≥ 1) := by
  sorry

end sqrt_x_minus_1_meaningful_l1410_141002


namespace cost_of_one_pack_of_gummy_bears_l1410_141082

theorem cost_of_one_pack_of_gummy_bears
    (num_chocolate_bars : ℕ)
    (num_gummy_bears : ℕ)
    (num_chocolate_chips : ℕ)
    (total_cost : ℕ)
    (cost_per_chocolate_bar : ℕ)
    (cost_per_chocolate_chip : ℕ)
    (cost_of_one_gummy_bear_pack : ℕ)
    (h1 : num_chocolate_bars = 10)
    (h2 : num_gummy_bears = 10)
    (h3 : num_chocolate_chips = 20)
    (h4 : total_cost = 150)
    (h5 : cost_per_chocolate_bar = 3)
    (h6 : cost_per_chocolate_chip = 5)
    (h7 : num_chocolate_bars * cost_per_chocolate_bar +
          num_gummy_bears * cost_of_one_gummy_bear_pack +
          num_chocolate_chips * cost_per_chocolate_chip = total_cost) :
    cost_of_one_gummy_bear_pack = 2 := by
  sorry

end cost_of_one_pack_of_gummy_bears_l1410_141082


namespace part1_part2_l1410_141032

theorem part1 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (hA : A + B + C = π) 
  (ha : a = 2) 
  (hcosC : Real.cos C = -1 / 4) 
  (hsinA_sinB : Real.sin A = 2 * Real.sin B) : b = 1 ∧ c = Real.sqrt 6 := 
  sorry

theorem part2
  (A B C : ℝ) 
  (a b c : ℝ) 
  (hA : A + B + C = π) 
  (ha : a = 2) 
  (hcosC : Real.cos C = -1 / 4)
  (hcosA_minus_pi_div_4 : Real.cos (A - π / 4) = 4 / 5) : c = 5 * Real.sqrt 30 / 2 := 
  sorry

end part1_part2_l1410_141032


namespace factor_expression_l1410_141022

theorem factor_expression (x : ℝ) : 
  72 * x^2 + 108 * x + 36 = 36 * (2 * x^2 + 3 * x + 1) :=
sorry

end factor_expression_l1410_141022


namespace find_certain_number_l1410_141075

theorem find_certain_number (D S X : ℕ): 
  D = 20 → 
  S = 55 → 
  X + (D - S) = 3 * D - 90 →
  X = 5 := 
by
  sorry

end find_certain_number_l1410_141075


namespace find_a_plus_d_l1410_141006

theorem find_a_plus_d (a b c d : ℝ) (h₁ : ab + bc + ca + db = 42) (h₂ : b + c = 6) : a + d = 7 := 
sorry

end find_a_plus_d_l1410_141006


namespace prime_factor_condition_l1410_141020

def seq (n : ℕ) : ℕ :=
  match n with
  | 0 => 2
  | 1 => 1
  | n + 2 => seq (n + 1) + seq n

theorem prime_factor_condition (p k : ℕ) (hp : Nat.Prime p) (h : p ∣ seq (2 * k) - 2) :
  p ∣ seq (2 * k - 1) - 1 :=
sorry

end prime_factor_condition_l1410_141020


namespace remainder_of_expression_l1410_141010

theorem remainder_of_expression :
  (7 * 10^20 + 2^20) % 11 = 8 := 
by {
  -- Prove the expression step by step
  -- sorry
  sorry
}

end remainder_of_expression_l1410_141010


namespace topsoil_cost_l1410_141016

theorem topsoil_cost (cost_per_cubic_foot : ℕ) (cubic_yard_to_cubic_foot : ℕ) (volume_in_cubic_yards : ℕ) :
  cost_per_cubic_foot = 8 →
  cubic_yard_to_cubic_foot = 27 →
  volume_in_cubic_yards = 3 →
  volume_in_cubic_yards * cubic_yard_to_cubic_foot * cost_per_cubic_foot = 648 :=
by
  intros h1 h2 h3
  sorry

end topsoil_cost_l1410_141016


namespace sum_of_all_four_is_zero_l1410_141008

variables {a b c d : ℤ}

theorem sum_of_all_four_is_zero (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_sum_rows : a + b = c + d) 
  (h_product_columns : a * c = b * d) :
  a + b + c + d = 0 := 
sorry

end sum_of_all_four_is_zero_l1410_141008


namespace dave_won_tickets_l1410_141088

theorem dave_won_tickets (initial_tickets spent_tickets final_tickets won_tickets : ℕ) 
  (h1 : initial_tickets = 25) 
  (h2 : spent_tickets = 22) 
  (h3 : final_tickets = 18) 
  (h4 : won_tickets = final_tickets - (initial_tickets - spent_tickets)) :
  won_tickets = 15 := 
by 
  sorry

end dave_won_tickets_l1410_141088


namespace distance_from_P_to_origin_l1410_141052

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem distance_from_P_to_origin :
  distance (-1) 2 0 0 = Real.sqrt 5 :=
by
  sorry

end distance_from_P_to_origin_l1410_141052


namespace theta1_gt_theta2_l1410_141054

theorem theta1_gt_theta2 (a : ℝ) (b : ℝ) (θ1 θ2 : ℝ)
  (h_range_θ1 : 0 ≤ θ1 ∧ θ1 ≤ π) (h_range_θ2 : 0 ≤ θ2 ∧ θ2 ≤ π)
  (x1 x2 : ℝ) (hx1 : x1 = a * Real.cos θ1) (hx2 : x2 = a * Real.cos θ2)
  (h_less : x1 < x2) : θ1 > θ2 :=
by
  sorry

end theta1_gt_theta2_l1410_141054


namespace find_first_number_l1410_141094

theorem find_first_number : ∃ x : ℕ, x + 7314 = 3362 + 13500 ∧ x = 9548 :=
by
  -- This is where the proof would go
  sorry

end find_first_number_l1410_141094


namespace line_tangent_to_circle_l1410_141051

theorem line_tangent_to_circle (r : ℝ) :
  (∀ (x y : ℝ), (x + y = 4) → (x - 2)^2 + (y + 1)^2 = r) → r = 9 / 2 :=
sorry

end line_tangent_to_circle_l1410_141051


namespace wendy_total_sales_correct_l1410_141048

noncomputable def wendy_total_sales : ℝ :=
  let morning_apples := 40 * 1.50
  let morning_oranges := 30 * 1
  let morning_bananas := 10 * 0.75
  let afternoon_apples := 50 * 1.35
  let afternoon_oranges := 40 * 0.90
  let afternoon_bananas := 20 * 0.675
  let unsold_bananas := 20 * 0.375
  let unsold_oranges := 10 * 0.50
  let total_morning := morning_apples + morning_oranges + morning_bananas
  let total_afternoon := afternoon_apples + afternoon_oranges + afternoon_bananas
  let total_day_sales := total_morning + total_afternoon
  let total_unsold_sales := unsold_bananas + unsold_oranges
  total_day_sales + total_unsold_sales

theorem wendy_total_sales_correct :
  wendy_total_sales = 227 := by
  unfold wendy_total_sales
  sorry

end wendy_total_sales_correct_l1410_141048


namespace find_a1_l1410_141035

variable (a : ℕ → ℕ)
variable (q : ℕ)
variable (h_q_pos : 0 < q)
variable (h_a2a6 : a 2 * a 6 = 8 * a 4)
variable (h_a2 : a 2 = 2)

theorem find_a1 :
  a 1 = 1 :=
by
  sorry

end find_a1_l1410_141035


namespace ways_to_go_home_via_library_l1410_141089

def ways_from_school_to_library := 2
def ways_from_library_to_home := 3

theorem ways_to_go_home_via_library : 
  ways_from_school_to_library * ways_from_library_to_home = 6 :=
by 
  sorry

end ways_to_go_home_via_library_l1410_141089


namespace distribute_pencils_l1410_141009

variables {initial_pencils : ℕ} {num_containers : ℕ} {additional_pencils : ℕ}

theorem distribute_pencils (h₁ : initial_pencils = 150) (h₂ : num_containers = 5)
                           (h₃ : additional_pencils = 30) :
  (initial_pencils + additional_pencils) / num_containers = 36 :=
by sorry

end distribute_pencils_l1410_141009


namespace remainder_of_a_sq_plus_five_mod_seven_l1410_141068

theorem remainder_of_a_sq_plus_five_mod_seven (a : ℕ) (h : a % 7 = 4) : (a^2 + 5) % 7 = 0 := 
by 
  sorry

end remainder_of_a_sq_plus_five_mod_seven_l1410_141068


namespace div_36_of_n_ge_5_l1410_141000

noncomputable def n := Nat

theorem div_36_of_n_ge_5 (n : ℕ) (hn : n ≥ 5) (h2 : ¬ (n % 2 = 0)) (h3 : ¬ (n % 3 = 0)) : 36 ∣ (n^2 - 1) :=
by
  sorry

end div_36_of_n_ge_5_l1410_141000


namespace unique_real_solution_for_cubic_l1410_141044

theorem unique_real_solution_for_cubic {b : ℝ} :
  (∀ x : ℝ, (x^3 - b * x^2 - 3 * b * x + b^2 - 4 = 0) → ∃! x : ℝ, (x^3 - b * x^2 - 3 * b * x + b^2 - 4 = 0)) ↔ b > 3 :=
sorry

end unique_real_solution_for_cubic_l1410_141044


namespace problem_l1410_141058

open Real

theorem problem (x y : ℝ) (h_posx : 0 < x) (h_posy : 0 < y) (h_cond : x + y^(2016) ≥ 1) : 
  x^(2016) + y > 1 - 1/100 :=
by sorry

end problem_l1410_141058


namespace total_cost_l1410_141057

variable (a b : ℝ)

theorem total_cost (ha : a ≥ 0) (hb : b ≥ 0) : 3 * a + 4 * b = 3 * a + 4 * b :=
by sorry

end total_cost_l1410_141057


namespace part1_optimal_strategy_part2_optimal_strategy_l1410_141011

noncomputable def R (x1 x2 : ℝ) : ℝ := -2 * x1^2 - x2^2 + 13 * x1 + 11 * x2 - 28

theorem part1_optimal_strategy :
  ∃ x1 x2 : ℝ, x1 + x2 = 5 ∧ x1 = 2 ∧ x2 = 3 ∧
    ∀ y1 y2, y1 + y2 = 5 → (R y1 y2 - (y1 + y2) ≤ R x1 x2 - (x1 + x2)) := 
by
  sorry

theorem part2_optimal_strategy :
  ∃ x1 x2 : ℝ, x1 = 3 ∧ x2 = 5 ∧
    ∀ y1 y2, (R y1 y2 - (y1 + y2) ≤ R x1 x2 - (x1 + x2)) := 
by
  sorry

end part1_optimal_strategy_part2_optimal_strategy_l1410_141011


namespace complete_the_square_l1410_141093

theorem complete_the_square (y : ℤ) : y^2 + 14 * y + 60 = (y + 7)^2 + 11 :=
by
  sorry

end complete_the_square_l1410_141093


namespace proof_problem_l1410_141059

noncomputable def f (a x : ℝ) : ℝ := (a / (a^2 - 1)) * (Real.exp (Real.log a * x) - Real.exp (-Real.log a * x))

theorem proof_problem (
  a : ℝ
) (h1 : a > 1) :
  (∀ x, f a x = (a / (a^2 - 1)) * (Real.exp (Real.log a * x) - Real.exp (-Real.log a * x))) ∧
  (∀ x, f a (-x) = -f a x) ∧
  (∀ x1 x2, x1 < x2 → f a x1 < f a x2) ∧
  (∀ m, -1 < 1 - m ∧ 1 - m < m^2 - 1 ∧ m^2 - 1 < 1 → 1 < m ∧ m < Real.sqrt 2)
  :=
sorry

end proof_problem_l1410_141059


namespace smallest_discount_l1410_141086

theorem smallest_discount (n : ℕ) (h1 : (1 - 0.12) * (1 - 0.18) = 0.88 * 0.82)
  (h2 : (1 - 0.08) * (1 - 0.08) * (1 - 0.08) = 0.92 * 0.92 * 0.92)
  (h3 : (1 - 0.20) * (1 - 0.10) = 0.80 * 0.90) :
  (29 > 27.84 ∧ 29 > 22.1312 ∧ 29 > 28) :=
by {
  sorry
}

end smallest_discount_l1410_141086


namespace problem_statement_l1410_141083

noncomputable def f (x : ℚ) : ℚ := (x^2 - x - 6) / (x^3 - 2 * x^2 - x + 2)

def a : ℕ := 1  -- number of holes
def b : ℕ := 2  -- number of vertical asymptotes
def c : ℕ := 1  -- number of horizontal asymptotes
def d : ℕ := 0  -- number of oblique asymptotes

theorem problem_statement : a + 2 * b + 3 * c + 4 * d = 8 :=
by
  sorry

end problem_statement_l1410_141083


namespace people_in_circle_l1410_141021

theorem people_in_circle (n : ℕ) (h : ∃ k : ℕ, k * 2 + 7 = 18) : n = 22 :=
by
  sorry

end people_in_circle_l1410_141021


namespace larger_number_is_299_l1410_141039

theorem larger_number_is_299 (A B : ℕ) 
  (HCF_AB : Nat.gcd A B = 23) 
  (LCM_12_13 : Nat.lcm A B = 23 * 12 * 13) : 
  max A B = 299 := 
sorry

end larger_number_is_299_l1410_141039


namespace remainder_when_divided_by_x_plus_2_l1410_141077

variable (D E F : ℝ)

def q (x : ℝ) := D * x^4 + E * x^2 + F * x + 7

theorem remainder_when_divided_by_x_plus_2 :
  q D E F (-2) = 21 - 2 * F :=
by
  have hq2 : q D E F 2 = 21 := sorry
  sorry

end remainder_when_divided_by_x_plus_2_l1410_141077


namespace number_of_possible_values_l1410_141017

theorem number_of_possible_values (x : ℕ) (h1 : x > 6) (h2 : x + 4 > 0) :
  ∃ (n : ℕ), n = 24 := 
sorry

end number_of_possible_values_l1410_141017


namespace find_k_l1410_141040

theorem find_k (k : ℝ) :
  (∀ x : ℝ, x^2 + k * x + 12 = 0 → ∃ y : ℝ, y = x + 3 ∧ y^2 - k * y + 12 = 0) →
  k = 3 :=
sorry

end find_k_l1410_141040


namespace total_carrots_l1410_141036

theorem total_carrots (sally_carrots fred_carrots : ℕ) (h1 : sally_carrots = 6) (h2 : fred_carrots = 4) : sally_carrots + fred_carrots = 10 := by
  sorry

end total_carrots_l1410_141036


namespace probability_snow_at_least_once_l1410_141063

-- Defining the probability of no snow on the first five days
def no_snow_first_five_days : ℚ := (4 / 5) ^ 5

-- Defining the probability of no snow on the next five days
def no_snow_next_five_days : ℚ := (2 / 3) ^ 5

-- Total probability of no snow during the first ten days
def no_snow_first_ten_days : ℚ := no_snow_first_five_days * no_snow_next_five_days

-- Probability of snow at least once during the first ten days
def snow_at_least_once_first_ten_days : ℚ := 1 - no_snow_first_ten_days

-- Desired proof statement
theorem probability_snow_at_least_once :
  snow_at_least_once_first_ten_days = 726607 / 759375 := by
  sorry

end probability_snow_at_least_once_l1410_141063


namespace correct_first_coupon_day_l1410_141018

def is_redemption_valid (start_day : ℕ) (interval : ℕ) (num_coupons : ℕ) (closed_day : ℕ) : Prop :=
  ∀ n : ℕ, n < num_coupons → (start_day + n * interval) % 7 ≠ closed_day

def wednesday : ℕ := 3  -- Assuming Sunday = 0, Monday = 1, ..., Saturday = 6

theorem correct_first_coupon_day : 
  is_redemption_valid wednesday 10 6 0 :=
by {
  -- Proof goes here
  sorry
}

end correct_first_coupon_day_l1410_141018


namespace least_number_of_square_tiles_l1410_141026

theorem least_number_of_square_tiles (length : ℕ) (breadth : ℕ) (gcd : ℕ) (area_room : ℕ) (area_tile : ℕ) (num_tiles : ℕ) :
  length = 544 → breadth = 374 → gcd = Nat.gcd length breadth → gcd = 2 →
  area_room = length * breadth → area_tile = gcd * gcd →
  num_tiles = area_room / area_tile → num_tiles = 50864 :=
by
  sorry

end least_number_of_square_tiles_l1410_141026


namespace max_value_of_z_l1410_141098

theorem max_value_of_z (x y z : ℝ) (h_add : x + y + z = 5) (h_mult : x * y + y * z + z * x = 3) : z ≤ 13 / 3 :=
sorry

end max_value_of_z_l1410_141098


namespace range_of_third_side_l1410_141067

theorem range_of_third_side (y : ℝ) : (2 < y) ↔ (y < 8) :=
by sorry

end range_of_third_side_l1410_141067


namespace complex_ratio_max_min_diff_l1410_141043

noncomputable def max_minus_min_complex_ratio (z w : ℂ) : ℝ :=
max (1 : ℝ) (0 : ℝ) - min (1 : ℝ) (0 : ℝ)

theorem complex_ratio_max_min_diff (z w : ℂ) (hz : z ≠ 0) (hw : w ≠ 0) : 
  max_minus_min_complex_ratio z w = 1 :=
by sorry

end complex_ratio_max_min_diff_l1410_141043


namespace product_y_coordinates_l1410_141038

theorem product_y_coordinates : 
  ∀ y : ℝ, (∀ P : ℝ × ℝ, P.1 = -1 ∧ (P.1 - 4)^2 + (P.2 - 3)^2 = 64 → P = (-1, y)) →
  ((3 + Real.sqrt 39) * (3 - Real.sqrt 39) = -30) :=
by
  intros y h
  sorry

end product_y_coordinates_l1410_141038


namespace min_n_consecutive_integers_sum_of_digits_is_multiple_of_8_l1410_141081

theorem min_n_consecutive_integers_sum_of_digits_is_multiple_of_8 
: ∃ n : ℕ, (∀ (nums : Fin n.succ → ℕ), 
              (∀ i j, i < j → nums i < nums j → nums j = nums i + 1) →
              ∃ i, (nums i) % 8 = 0) ∧ n = 15 := 
sorry

end min_n_consecutive_integers_sum_of_digits_is_multiple_of_8_l1410_141081


namespace even_function_a_value_l1410_141050

theorem even_function_a_value (a : ℝ) : 
  (∀ x : ℝ, (a + 1) * x^2 + (a - 2) * x + a^2 - a - 2 = (a + 1) * x^2 - (a - 2) * x + a^2 - a - 2) → a = 2 := 
by sorry

end even_function_a_value_l1410_141050


namespace simplify_expression_l1410_141033

theorem simplify_expression : 
  1.5 * (Real.sqrt 1 + Real.sqrt (1+3) + Real.sqrt (1+3+5) + Real.sqrt (1+3+5+7) + Real.sqrt (1+3+5+7+9)) = 22.5 :=
by
  sorry

end simplify_expression_l1410_141033


namespace cyclists_meet_time_l1410_141046

/-- 
  Two cyclists start on a circular track from a given point but in opposite directions with speeds of 7 m/s and 8 m/s.
  The circumference of the circle is 180 meters.
  After what time will they meet at the starting point? 
-/
theorem cyclists_meet_time :
  let speed1 := 7 -- m/s
  let speed2 := 8 -- m/s
  let circumference := 180 -- meters
  (circumference / (speed1 + speed2) = 12) :=
by
  let speed1 := 7 -- m/s
  let speed2 := 8 -- m/s
  let circumference := 180 -- meters
  sorry

end cyclists_meet_time_l1410_141046


namespace rectangle_length_l1410_141056

theorem rectangle_length (sq_side_len rect_width : ℕ) (sq_area : ℕ) (rect_len : ℕ) 
    (h1 : sq_side_len = 6) 
    (h2 : rect_width = 4) 
    (h3 : sq_area = sq_side_len * sq_side_len) 
    (h4 : sq_area = rect_width * rect_len) :
    rect_len = 9 := 
by 
  sorry

end rectangle_length_l1410_141056


namespace total_apples_packed_correct_l1410_141055

-- Define the daily production of apples under normal conditions
def apples_per_box := 40
def boxes_per_day := 50
def days_per_week := 7
def apples_per_day := apples_per_box * boxes_per_day

-- Define the change in daily production for the next week
def fewer_apples := 500
def apples_per_day_next_week := apples_per_day - fewer_apples

-- Define the weekly production in normal and next conditions
def apples_first_week := apples_per_day * days_per_week
def apples_second_week := apples_per_day_next_week * days_per_week

-- Define the total apples packed in two weeks
def total_apples_packed := apples_first_week + apples_second_week

-- Prove the total apples packed is 24500
theorem total_apples_packed_correct : total_apples_packed = 24500 := by
  sorry

end total_apples_packed_correct_l1410_141055


namespace socks_problem_l1410_141029

/-
  Theorem: Given x + y + z = 15, 2x + 4y + 5z = 36, and x, y, z ≥ 1, 
  the number of $2 socks Jack bought is x = 4.
-/

theorem socks_problem
  (x y z : ℕ)
  (h1 : x + y + z = 15)
  (h2 : 2 * x + 4 * y + 5 * z = 36)
  (h3 : 1 ≤ x)
  (h4 : 1 ≤ y)
  (h5 : 1 ≤ z) :
  x = 4 :=
  sorry

end socks_problem_l1410_141029


namespace ratio_Cheryl_C_to_Cyrus_Y_l1410_141014

noncomputable def Cheryl_C : ℕ := 126
noncomputable def Madeline_M : ℕ := 63
noncomputable def Total_pencils : ℕ := 231
noncomputable def Cyrus_Y : ℕ := Total_pencils - Cheryl_C - Madeline_M

theorem ratio_Cheryl_C_to_Cyrus_Y : 
  Cheryl_C = 2 * Madeline_M → 
  Madeline_M + Cheryl_C + Cyrus_Y = Total_pencils → 
  Cheryl_C / Cyrus_Y = 3 :=
by
  intros h1 h2
  sorry

end ratio_Cheryl_C_to_Cyrus_Y_l1410_141014


namespace inverse_proposition_l1410_141045

theorem inverse_proposition :
  (∀ x : ℝ, x < 0 → x^2 > 0) → (∀ y : ℝ, y^2 > 0 → y < 0) :=
by
  sorry

end inverse_proposition_l1410_141045


namespace range_of_a_l1410_141073

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 + 2 * x - a > 0) → a < -1 :=
by
  sorry

end range_of_a_l1410_141073


namespace largest_fraction_l1410_141076

theorem largest_fraction :
  let A := 2 / 5
  let B := 3 / 7
  let C := 4 / 9
  let D := 5 / 11
  let E := 6 / 13
  E > A ∧ E > B ∧ E > C ∧ E > D :=
by
  let A := 2 / 5
  let B := 3 / 7
  let C := 4 / 9
  let D := 5 / 11
  let E := 6 / 13
  sorry

end largest_fraction_l1410_141076


namespace solve_eq1_solve_eq2_l1410_141025

theorem solve_eq1 {x : ℝ} : 2 * x^2 - 1 = 49 ↔ x = 5 ∨ x = -5 := 
  sorry

theorem solve_eq2 {x : ℝ} : (x + 3)^3 = 64 ↔ x = 1 := 
  sorry

end solve_eq1_solve_eq2_l1410_141025


namespace find_p_q_l1410_141099

theorem find_p_q : 
  (∀ x : ℝ, (x - 2) * (x + 1) ∣ (x ^ 5 - x ^ 4 + x ^ 3 - p * x ^ 2 + q * x - 8)) → (p = -1 ∧ q = -10) :=
by
  sorry

end find_p_q_l1410_141099


namespace proof_problem_l1410_141015

variable {a b c d e f : ℝ}

theorem proof_problem :
  (a * b * c = 130) →
  (b * c * d = 65) →
  (d * e * f = 250) →
  (a * f / (c * d) = 0.5) →
  (c * d * e = 1000) :=
by
  intros h1 h2 h3 h4
  sorry

end proof_problem_l1410_141015


namespace arith_sign_change_geo_sign_change_l1410_141049

-- Definitions for sequences
def arith_sequence (a₁ d : ℝ) : ℕ → ℝ
| 0 => a₁
| (n + 1) => arith_sequence a₁ d n + d

def geo_sequence (a₁ r : ℝ) : ℕ → ℝ
| 0 => a₁
| (n + 1) => geo_sequence a₁ r n * r

-- Problem statement
theorem arith_sign_change :
  ∀ (a₁ d : ℝ), (∃ N : ℕ, arith_sequence a₁ d N = 0) ∨ (∀ n m : ℕ, (arith_sequence a₁ d n) * (arith_sequence a₁ d m) ≥ 0) :=
sorry

theorem geo_sign_change :
  ∀ (a₁ r : ℝ), r < 0 → ∀ n : ℕ, (geo_sequence a₁ r n) * (geo_sequence a₁ r (n + 1)) < 0 :=
sorry

end arith_sign_change_geo_sign_change_l1410_141049


namespace line_parameterization_l1410_141070

theorem line_parameterization (r k : ℝ) (t : ℝ) :
  (∀ x y : ℝ, (x, y) = (r + 3 * t, 2 + k * t) → (y = 2 * x - 5) ) ∧
  (t = 0 → r = 7 / 2) ∧
  (t = 1 → k = 6) :=
by
  sorry

end line_parameterization_l1410_141070


namespace stuart_initial_marbles_l1410_141003

theorem stuart_initial_marbles
    (betty_marbles : ℕ)
    (stuart_marbles_after_given : ℕ)
    (percentage_given : ℚ)
    (betty_gave : ℕ):
    betty_marbles = 60 →
    stuart_marbles_after_given = 80 →
    percentage_given = 0.40 →
    betty_gave = percentage_given * betty_marbles →
    stuart_marbles_after_given = stuart_initial + betty_gave →
    stuart_initial = 56 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end stuart_initial_marbles_l1410_141003


namespace find_general_term_l1410_141072

variable (a : ℕ → ℝ) (a1 : a 1 = 1)

def isGeometricSequence (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = a n * q

def isArithmeticSequence (u v w : ℝ) :=
  2 * v = u + w

theorem find_general_term (h1 : a 1 = 1)
  (h2 : (isGeometricSequence a (1 / 2)))
  (h3 : isArithmeticSequence (1 / a 1) (1 / a 3) (1 / a 4 - 1)) :
  ∀ n, a n = (1 / 2) ^ (n - 1) :=
sorry

end find_general_term_l1410_141072


namespace compute_x_y_sum_l1410_141060

theorem compute_x_y_sum (x y : ℝ) (hx : x > 1) (hy : y > 1)
  (h : (Real.log x / Real.log 2)^4 + (Real.log y / Real.log 3)^4 + 8 = 8 * (Real.log x / Real.log 2) * (Real.log y / Real.log 3)) :
  x^Real.sqrt 2 + y^Real.sqrt 2 = 13 :=
by
  sorry

end compute_x_y_sum_l1410_141060
