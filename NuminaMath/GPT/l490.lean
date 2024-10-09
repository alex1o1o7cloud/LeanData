import Mathlib

namespace remainder_of_large_number_l490_49086

noncomputable def X (k : ℕ) : ℕ :=
  match k with
  | 0 => 1
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | 4 => 16
  | 5 => 32
  | 6 => 64
  | 7 => 128
  | 8 => 256
  | 9 => 512
  | 10 => 1024
  | 11 => 2048
  | 12 => 4096
  | 13 => 8192
  | _ => 0

noncomputable def concatenate_X (k : ℕ) : ℕ :=
  if k = 5 then 
    100020004000800160032
  else if k = 11 then 
    100020004000800160032006401280256051210242048
  else if k = 13 then 
    10002000400080016003200640128025605121024204840968192
  else 
    0

theorem remainder_of_large_number :
  (concatenate_X 13) % (concatenate_X 5) = 40968192 :=
by
  sorry

end remainder_of_large_number_l490_49086


namespace geometric_sequence_ninth_tenth_term_sum_l490_49092

noncomputable def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q^n

theorem geometric_sequence_ninth_tenth_term_sum (a₁ q : ℝ)
  (h1 : a₁ + a₁ * q = 2)
  (h5 : a₁ * q^4 + a₁ * q^5 = 4) :
  geometric_sequence a₁ q 8 + geometric_sequence a₁ q 9 = 8 :=
by
  sorry

end geometric_sequence_ninth_tenth_term_sum_l490_49092


namespace company_workers_count_l490_49024

-- Definitions
def num_supervisors := 13
def team_leads_per_supervisor := 3
def workers_per_team_lead := 10

-- Hypothesis
def team_leads := num_supervisors * team_leads_per_supervisor
def workers := team_leads * workers_per_team_lead

-- Theorem to prove
theorem company_workers_count : workers = 390 :=
by
  sorry

end company_workers_count_l490_49024


namespace intersection_A_B_l490_49063

open Set

def isInSetA (x : ℕ) : Prop := ∃ n : ℕ, x = 3 * n + 2
def A : Set ℕ := { x | isInSetA x }
def B : Set ℕ := {6, 8, 10, 12, 14}

theorem intersection_A_B :
  A ∩ B = {8, 14} :=
sorry

end intersection_A_B_l490_49063


namespace initial_workers_l490_49002

theorem initial_workers (M : ℝ) :
  let totalLength : ℝ := 15
  let totalDays : ℝ := 300
  let completedLength : ℝ := 2.5
  let completedDays : ℝ := 100
  let remainingLength : ℝ := totalLength - completedLength
  let remainingDays : ℝ := totalDays - completedDays
  let extraMen : ℝ := 60
  let rateWithM : ℝ := completedLength / completedDays
  let newRate : ℝ := remainingLength / remainingDays
  let newM : ℝ := M + extraMen
  (rateWithM * M = newRate * newM) → M = 100 :=
by
  intros h
  sorry

end initial_workers_l490_49002


namespace cylinder_volume_ratio_l490_49097

theorem cylinder_volume_ratio
  (h : ℝ)     -- height of cylinder B (radius of cylinder A)
  (r : ℝ)     -- radius of cylinder B (height of cylinder A)
  (VA : ℝ)    -- volume of cylinder A
  (VB : ℝ)    -- volume of cylinder B
  (cond1 : r = h / 3)
  (cond2 : VB = 3 * VA)
  (cond3 : VB = N * Real.pi * h^3) :
  N = 1 / 3 := 
sorry

end cylinder_volume_ratio_l490_49097


namespace find_fake_coin_l490_49046

def coin_value (n : Nat) : Nat :=
  match n with
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 5
  | _ => 0

def coin_weight (n : Nat) : Nat :=
  match n with
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 5
  | _ => 0

def is_fake (weight : Nat) : Prop :=
  weight ≠ coin_weight 1 ∧ weight ≠ coin_weight 2 ∧ weight ≠ coin_weight 3 ∧ weight ≠ coin_weight 4

theorem find_fake_coin :
  ∃ (n : Nat) (w : Nat), (is_fake w) → ∃! (m : Nat), m ≠ w ∧ (m = coin_weight 1 ∨ m = coin_weight 2 ∨ m = coin_weight 3 ∨ m = coin_weight 4) := 
sorry

end find_fake_coin_l490_49046


namespace cost_of_gas_per_gallon_l490_49093

-- Definitions based on the conditions
def hours_driven_1 : ℕ := 2
def speed_1 : ℕ := 60
def hours_driven_2 : ℕ := 3
def speed_2 : ℕ := 50
def mileage_per_gallon : ℕ := 30
def total_gas_cost : ℕ := 18

-- An assumption to simplify handling dollars and gallons
noncomputable def cost_per_gallon : ℕ := total_gas_cost / (speed_1 * hours_driven_1 + speed_2 * hours_driven_2) * mileage_per_gallon

theorem cost_of_gas_per_gallon :
  cost_per_gallon = 2 := by
sorry

end cost_of_gas_per_gallon_l490_49093


namespace final_customer_boxes_l490_49068

theorem final_customer_boxes (f1 f2 f3 f4 goal left boxes_first : ℕ) 
  (h1 : boxes_first = 5) 
  (h2 : f2 = 4 * boxes_first) 
  (h3 : f3 = f2 / 2) 
  (h4 : f4 = 3 * f3)
  (h5 : goal = 150) 
  (h6 : left = 75) 
  (h7 : goal - left = f1 + f2 + f3 + f4) : 
  (goal - left - (f1 + f2 + f3 + f4) = 10) := 
sorry

end final_customer_boxes_l490_49068


namespace negation_of_universal_prop_l490_49088

theorem negation_of_universal_prop : 
  (¬ (∀ (x : ℝ), x ^ 2 ≥ 0)) ↔ (∃ (x : ℝ), x ^ 2 < 0) :=
by sorry

end negation_of_universal_prop_l490_49088


namespace sum_of_numbers_l490_49007

theorem sum_of_numbers : (4.75 + 0.303 + 0.432) = 5.485 := 
by  
  sorry

end sum_of_numbers_l490_49007


namespace range_of_a_l490_49028

def p (a : ℝ) : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

theorem range_of_a (a : ℝ) (h : p a ∧ q a) : a ≤ -2 ∨ a = 1 := 
sorry

end range_of_a_l490_49028


namespace g_is_correct_l490_49085

noncomputable def g : ℝ → ℝ := sorry

axiom g_0 : g 0 = 2

axiom g_functional_eq : ∀ x y : ℝ, g (x * y) = g (x^2 + y^2) + 2 * (x - y)^2

theorem g_is_correct : ∀ x : ℝ, g x = 2 - 2 * x := 
by 
  sorry

end g_is_correct_l490_49085


namespace ratio_proof_l490_49032

theorem ratio_proof (a b c : ℝ) (ha : b / a = 3) (hb : c / b = 4) :
    (a + 2 * b) / (b + 2 * c) = 7 / 27 := by
  sorry

end ratio_proof_l490_49032


namespace probability_selecting_A_l490_49056

theorem probability_selecting_A :
  let total_people := 4
  let favorable_outcomes := 1
  let probability := favorable_outcomes / total_people
  probability = 1 / 4 :=
by
  sorry

end probability_selecting_A_l490_49056


namespace find_y_when_x4_l490_49008

theorem find_y_when_x4 : 
  (∀ x y : ℚ, 5 * y + 3 = 344 / (x ^ 3)) ∧ (5 * (8:ℚ) + 3 = 344 / (2 ^ 3)) → 
  (∃ y : ℚ, 5 * y + 3 = 344 / (4 ^ 3) ∧ y = 19 / 40) := 
by
  sorry

end find_y_when_x4_l490_49008


namespace online_store_commission_l490_49029

theorem online_store_commission (cost : ℝ) (desired_profit_pct : ℝ) (online_price : ℝ) (commission_pct : ℝ) :
  cost = 19 →
  desired_profit_pct = 0.20 →
  online_price = 28.5 →
  commission_pct = 25 :=
by
  sorry

end online_store_commission_l490_49029


namespace valid_division_l490_49030

theorem valid_division (A B C E F G H K : ℕ) (hA : A = 7) (hB : B = 1) (hC : C = 2)
    (hE : E = 6) (hF : F = 8) (hG : G = 5) (hH : H = 4) (hK : K = 9) :
    (A * 10 + B) / ((C * 100 + A * 10 + B) / 100 + E + B * F * D) = 71 / 271 :=
by {
  sorry
}

end valid_division_l490_49030


namespace Sarah_books_in_8_hours_l490_49078

theorem Sarah_books_in_8_hours (pages_per_hour: ℕ) (pages_per_book: ℕ) (hours_available: ℕ) 
  (h_pages_per_hour: pages_per_hour = 120) (h_pages_per_book: pages_per_book = 360) (h_hours_available: hours_available = 8) :
  hours_available * pages_per_hour / pages_per_book = 2 := by
  sorry

end Sarah_books_in_8_hours_l490_49078


namespace find_a2019_l490_49051

-- Arithmetic sequence
def a (n : ℕ) : ℤ := sorry -- to be defined later

-- Given conditions
def sum_first_five_terms (a: ℕ → ℤ) : Prop := a 1 + a 2 + a 3 + a 4 + a 5 = 15
def term_six (a: ℕ → ℤ) : Prop := a 6 = 6

-- Question (statement to be proved)
def term_2019 (a: ℕ → ℤ) : Prop := a 2019 = 2019

-- Main theorem to be proved
theorem find_a2019 (a: ℕ → ℤ) 
  (h1 : sum_first_five_terms a)
  (h2 : term_six a) : 
  term_2019 a := 
by
  sorry

end find_a2019_l490_49051


namespace inverse_value_l490_49049

def f (x : ℤ) : ℤ := 5 * x ^ 3 - 3

theorem inverse_value : ∀ y, (f y) = 4 → y = 317 :=
by
  intros
  sorry

end inverse_value_l490_49049


namespace quadratic_roots_l490_49095

-- Define the given conditions of the equation
def eqn (z : ℂ) : Prop := z^2 + 2 * z + (3 - 4 * Complex.I) = 0

-- State the theorem to prove that the roots of the equation are 2i and -2 + 2i.
theorem quadratic_roots :
  ∃ z1 z2 : ℂ, (z1 = 2 * Complex.I ∧ z2 = -2 + 2 * Complex.I) ∧ 
  (∀ z : ℂ, eqn z → z = z1 ∨ z = z2) :=
by
  sorry

end quadratic_roots_l490_49095


namespace sqrt_2x_plus_y_eq_4_l490_49005

theorem sqrt_2x_plus_y_eq_4 (x y : ℝ) 
  (h1 : (3 * x + 1) = 4) 
  (h2 : (2 * y - 1) = 27) : 
  Real.sqrt (2 * x + y) = 4 := 
by 
  sorry

end sqrt_2x_plus_y_eq_4_l490_49005


namespace sarah_jim_ratio_l490_49098

theorem sarah_jim_ratio
  (Tim_toads : ℕ)
  (hTim : Tim_toads = 30)
  (Jim_toads : ℕ)
  (hJim : Jim_toads = Tim_toads + 20)
  (Sarah_toads : ℕ)
  (hSarah : Sarah_toads = 100) :
  Sarah_toads / Jim_toads = 2 :=
by
  sorry

end sarah_jim_ratio_l490_49098


namespace range_of_a_l490_49038

theorem range_of_a (f : ℝ → ℝ) (a : ℝ)
  (h_even : ∀ x, f x = f (-x))
  (h_mono_inc : ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y)
  (h_ineq : f (a - 3) < f 4) : -1 < a ∧ a < 7 :=
by
  sorry

end range_of_a_l490_49038


namespace complex_div_l490_49062

theorem complex_div (i : ℂ) (hi : i^2 = -1) : (1 + i) / i = 1 - i := by
  sorry

end complex_div_l490_49062


namespace total_spent_snacks_l490_49079

-- Define the costs and discounts
def cost_pizza : ℕ := 10
def boxes_robert_orders : ℕ := 5
def pizza_discount : ℝ := 0.15
def cost_soft_drink : ℝ := 1.50
def soft_drinks_robert : ℕ := 10
def cost_hamburger : ℕ := 3
def hamburgers_teddy_orders : ℕ := 6
def hamburger_discount : ℝ := 0.10
def soft_drinks_teddy : ℕ := 10

-- Calculate total costs
def total_cost_robert : ℝ := 
  let cost_pizza_total := (boxes_robert_orders * cost_pizza) * (1 - pizza_discount)
  let cost_soft_drinks_total := soft_drinks_robert * cost_soft_drink
  cost_pizza_total + cost_soft_drinks_total

def total_cost_teddy : ℝ :=
  let cost_hamburger_total := (hamburgers_teddy_orders * cost_hamburger) * (1 - hamburger_discount)
  let cost_soft_drinks_total := soft_drinks_teddy * cost_soft_drink
  cost_hamburger_total + cost_soft_drinks_total

-- The final theorem to prove the total spending
theorem total_spent_snacks : 
  total_cost_robert + total_cost_teddy = 88.70 := by
  sorry

end total_spent_snacks_l490_49079


namespace Anton_thought_number_is_729_l490_49070

theorem Anton_thought_number_is_729 :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧
  ((n / 100 = 1 ∧ (n / 10) % 10 = 0 ∧ n % 10 = 9) ∨
   (n / 100 = 7 ∧ (n / 10) % 10 = 0 ∧ n % 10 = 4) ∨
   (n / 100 = 1 ∧ (n / 10) % 10 = 2 ∧ n % 10 = 4)) → n = 729 :=
by sorry

end Anton_thought_number_is_729_l490_49070


namespace negation_equiv_l490_49015

theorem negation_equiv (p : Prop) : 
  (p = (∃ x₀ : ℝ, x₀^2 + 2*x₀ + 2 ≤ 0)) → 
  (¬ p = (∀ x : ℝ, x^2 + 2*x + 2 > 0)) :=
by
  sorry

end negation_equiv_l490_49015


namespace find_number_l490_49042

theorem find_number (x : ℕ) (h : x * 99999 = 65818408915) : x = 658185 :=
sorry

end find_number_l490_49042


namespace Eiffel_Tower_model_scale_l490_49034

theorem Eiffel_Tower_model_scale
  (h_tower : ℝ := 324)
  (h_model_cm : ℝ := 18) :
  (h_tower / (h_model_cm / 100)) / 100 = 18 :=
by
  sorry

end Eiffel_Tower_model_scale_l490_49034


namespace last_three_digits_of_5_pow_9000_l490_49061

theorem last_three_digits_of_5_pow_9000 (h : 5^300 ≡ 1 [MOD 800]) : 5^9000 ≡ 1 [MOD 800] :=
by
  -- The proof is omitted here according to the instruction
  sorry

end last_three_digits_of_5_pow_9000_l490_49061


namespace solve_abs_inequality_l490_49084

theorem solve_abs_inequality (x : ℝ) : abs ((7 - x) / 4) < 3 → 2 < x ∧ x < 19 :=
by 
  sorry

end solve_abs_inequality_l490_49084


namespace greg_rolls_probability_l490_49001

noncomputable def probability_of_more_ones_than_twos_and_threes_combined : ℚ :=
  (3046.5 : ℚ) / 7776

theorem greg_rolls_probability :
  probability_of_more_ones_than_twos_and_threes_combined = (3046.5 : ℚ) / 7776 := 
by 
  sorry

end greg_rolls_probability_l490_49001


namespace find_n_l490_49077

-- Definitions of the conditions
variables (x n : ℝ)
variable (h1 : (x / 4) * n + 10 - 12 = 48)
variable (h2 : x = 40)

-- Theorem statement
theorem find_n (x n : ℝ) (h1 : (x / 4) * n + 10 - 12 = 48) (h2 : x = 40) : n = 5 :=
by
  sorry

end find_n_l490_49077


namespace original_number_l490_49080

theorem original_number (N y x : ℕ) 
  (h1: N + y = 54321)
  (h2: N = 10 * y + x)
  (h3: 11 * y + x = 54321)
  (h4: x = 54321 % 11)
  (hy: y = 4938) : 
  N = 49383 := 
  by 
  sorry

end original_number_l490_49080


namespace greatest_three_digit_multiple_of_17_l490_49016

def is_multiple_of (n : ℕ) (k : ℕ) : Prop := ∃ m : ℕ, n = k * m

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∀ n : ℕ, is_three_digit n ∧ is_multiple_of n 17 → n = 986 := by
  sorry

end greatest_three_digit_multiple_of_17_l490_49016


namespace goblins_return_l490_49075

theorem goblins_return (n : ℕ) (f : Fin n → Fin n) (h1 : ∀ a, ∃! b, f a = b) (h2 : ∀ b, ∃! a, f a = b) : 
  ∃ k : ℕ, ∀ x : Fin n, (f^[k]) x = x := 
sorry

end goblins_return_l490_49075


namespace e_is_dq_sequence_l490_49053

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d a₀, ∀ n, a n = a₀ + n * d

def is_geometric_sequence (b : ℕ → ℕ) : Prop :=
  ∃ q b₀, q > 0 ∧ ∀ n, b n = b₀ * q^n

def is_dq_sequence (c : ℕ → ℕ) : Prop :=
  ∃ a b, is_arithmetic_sequence a ∧ is_geometric_sequence b ∧ ∀ n, c n = a n + b n

def e (n : ℕ) : ℕ :=
  n + 2^n

theorem e_is_dq_sequence : is_dq_sequence e :=
  sorry

end e_is_dq_sequence_l490_49053


namespace assignment_statement_correct_l490_49094

def meaning_of_assignment_statement (N : ℕ) := N + 1

theorem assignment_statement_correct :
  meaning_of_assignment_statement N = N + 1 :=
sorry

end assignment_statement_correct_l490_49094


namespace solution_set_of_fraction_inequality_l490_49057

theorem solution_set_of_fraction_inequality (a b x : ℝ) (h1: ∀ x, ax - b > 0 ↔ x ∈ Set.Iio 1) (h2: a < 0) (h3: a - b = 0) :
  ∀ x, (a * x + b) / (x - 2) > 0 ↔ x ∈ Set.Ioo (-1 : ℝ) 2 := 
sorry

end solution_set_of_fraction_inequality_l490_49057


namespace equal_lead_concentration_l490_49073

theorem equal_lead_concentration (x : ℝ) (h1 : 0 < x) (h2 : x < 6) (h3 : x < 12) 
: (x / 6 = (12 - x) / 12) → x = 4 := by
  sorry

end equal_lead_concentration_l490_49073


namespace prove_value_of_custom_ops_l490_49052

-- Define custom operations to match problem statement
def custom_op1 (x : ℤ) : ℤ := 7 - x
def custom_op2 (x : ℤ) : ℤ := x - 10

-- The main proof statement
theorem prove_value_of_custom_ops : custom_op2 (custom_op1 12) = -15 :=
by sorry

end prove_value_of_custom_ops_l490_49052


namespace solve_inequality_l490_49096

theorem solve_inequality (x : ℝ) (h : 3 - (1 / (3 * x + 4)) < 5) : 
  x ∈ { x : ℝ | x < -11/6 } ∨ x ∈ { x : ℝ | x > -4/3 } :=
by
  sorry

end solve_inequality_l490_49096


namespace ceil_minus_floor_eq_one_imp_ceil_minus_x_l490_49037

variable {x : ℝ}

theorem ceil_minus_floor_eq_one_imp_ceil_minus_x (H : ⌈x⌉ - ⌊x⌋ = 1) : ∃ (n : ℤ) (f : ℝ), (x = n + f) ∧ (0 < f) ∧ (f < 1) ∧ (⌈x⌉ - x = 1 - f) := sorry

end ceil_minus_floor_eq_one_imp_ceil_minus_x_l490_49037


namespace fraction_to_decimal_l490_49099

theorem fraction_to_decimal : (22 / 8 : ℝ) = 2.75 := 
sorry

end fraction_to_decimal_l490_49099


namespace hours_per_day_initial_l490_49043

-- Definition of the problem and conditions
def initial_men : ℕ := 75
def depth1 : ℕ := 50
def additional_men : ℕ := 65
def total_men : ℕ := initial_men + additional_men
def depth2 : ℕ := 70
def hours_per_day2 : ℕ := 6
def work1 (H : ℝ) := initial_men * H * depth1
def work2 := total_men * hours_per_day2 * depth2

-- Statement to prove
theorem hours_per_day_initial (H : ℝ) (h1 : work1 H = work2) : H = 15.68 :=
by
  sorry

end hours_per_day_initial_l490_49043


namespace sum_of_fully_paintable_numbers_l490_49035

def is_fully_paintable (h t u : ℕ) : Prop :=
  (∀ n : ℕ, (∀ k1 : ℕ, n ≠ 1 + k1 * h) ∧ (∀ k2 : ℕ, n ≠ 3 + k2 * t) ∧ (∀ k3 : ℕ, n ≠ 2 + k3 * u)) → False

theorem sum_of_fully_paintable_numbers :  ∃ L : List ℕ, (∀ x ∈ L, ∃ (h t u : ℕ), is_fully_paintable h t u ∧ 100 * h + 10 * t + u = x) ∧ L.sum = 944 :=
sorry

end sum_of_fully_paintable_numbers_l490_49035


namespace product_of_distances_is_one_l490_49020

theorem product_of_distances_is_one (k : ℝ) (x1 x2 : ℝ)
  (h1 : x1^2 - k*x1 - 1 = 0)
  (h2 : x2^2 - k*x2 - 1 = 0)
  (h3 : x1 ≠ x2) :
  (|x1| * |x2| = 1) :=
by
  -- Proof goes here
  sorry

end product_of_distances_is_one_l490_49020


namespace part1_part2_l490_49071

-- Define the conditions for p and q
def p (x : ℝ) : Prop := (x + 2) * (x - 6) <= 0
def q (x m : ℝ) : Prop := (2 - m <= x) ∧ (x <= 2 + m)

-- Proof statement for part (1)
theorem part1 (m: ℝ) : 
  (∀ x : ℝ, p x → q x m) → 4 <= m :=
sorry

-- Proof statement for part (2)
theorem part2 (x : ℝ) (m : ℝ) : 
  (m = 5) → (p x ∨ q x m) ∧ ¬(p x ∧ q x m) → x ∈ Set.Ico (-3) (-2) ∪ Set.Ioc 6 7 :=
sorry

end part1_part2_l490_49071


namespace inequality_correctness_l490_49087

theorem inequality_correctness (a b c : ℝ) (h : c^2 > 0) : (a * c^2 > b * c^2) ↔ (a > b) := by 
sorry

end inequality_correctness_l490_49087


namespace largest_number_l490_49054

theorem largest_number
  (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ) (e : ℝ)
  (ha : a = 0.883) (hb : b = 0.8839) (hc : c = 0.88) (hd : d = 0.839) (he : e = 0.889) :
  e > a ∧ e > b ∧ e > c ∧ e > d :=
by {
  sorry
}

end largest_number_l490_49054


namespace find_x_l490_49036

theorem find_x (x : ℝ) (h : 128/x + 75/x + 57/x = 6.5) : x = 40 :=
by
  sorry

end find_x_l490_49036


namespace units_digit_of_8_pow_47_l490_49022

theorem units_digit_of_8_pow_47 : (8 ^ 47) % 10 = 2 := by
  sorry

end units_digit_of_8_pow_47_l490_49022


namespace parabola_sum_vertex_point_l490_49090

theorem parabola_sum_vertex_point
  (a b c : ℝ)
  (h_vertex : ∀ y : ℝ, y = -6 → x = a * (y + 6)^2 + 8)
  (h_point : x = a * ((-4) + 6)^2 + 8)
  (ha : a = 0.5)
  (hb : b = 6)
  (hc : c = 26) :
  a + b + c = 32.5 :=
by
  sorry

end parabola_sum_vertex_point_l490_49090


namespace form_x2_sub_2y2_l490_49026

theorem form_x2_sub_2y2 (x y : ℤ) (hx : x % 2 = 1) : (x^2 - 2*y^2) % 8 = 1 ∨ (x^2 - 2*y^2) % 8 = -1 := 
sorry

end form_x2_sub_2y2_l490_49026


namespace ernie_circles_l490_49010

theorem ernie_circles (boxes_per_circle_ali boxes_per_circle_ernie total_boxes circles_ali : ℕ) 
  (h1 : boxes_per_circle_ali = 8)
  (h2 : boxes_per_circle_ernie = 10)
  (h3 : total_boxes = 80)
  (h4 : circles_ali = 5) : 
  (total_boxes - circles_ali * boxes_per_circle_ali) / boxes_per_circle_ernie = 4 :=
by
  sorry

end ernie_circles_l490_49010


namespace subtract_mult_equal_l490_49081

theorem subtract_mult_equal :
  2000000000000 - 1111111111111 * 1 = 888888888889 :=
by
  sorry

end subtract_mult_equal_l490_49081


namespace min_volume_for_cone_l490_49072

noncomputable def min_cone_volume (V1 : ℝ) : Prop :=
  ∀ V2 : ℝ, (V1 = 1) → 
    V2 ≥ (4 / 3)

-- The statement without proof
theorem min_volume_for_cone : 
  min_cone_volume 1 :=
sorry

end min_volume_for_cone_l490_49072


namespace euclid1976_partb_problem2_l490_49018

theorem euclid1976_partb_problem2
  (x y : ℝ)
  (geo_prog : y^2 = 2 * x)
  (arith_prog : 2 / y = 1 / x + 9 / x^2) :
  x * y = 27 / 2 := by 
  sorry

end euclid1976_partb_problem2_l490_49018


namespace cube_root_solutions_l490_49019

theorem cube_root_solutions (p : ℕ) (hp : p > 3) :
    (∃ (k : ℤ) (h1 : k^2 ≡ -3 [ZMOD p]), ∀ x, x^3 ≡ 1 [ZMOD p] → 
        (x = 1 ∨ (x^2 + x + 1 ≡ 0 [ZMOD p])) )
    ∨ 
    (∀ x, x^3 ≡ 1 [ZMOD p] → x = 1) := 
sorry

end cube_root_solutions_l490_49019


namespace angle_F_measure_l490_49050

-- Define angle B
def angle_B := 120

-- Define angle C being supplementary to angle B on a straight line
def angle_C := 180 - angle_B

-- Define angle D
def angle_D := 45

-- Define angle E
def angle_E := 30

-- Define the vertically opposite angle F to angle C
def angle_F := angle_C

theorem angle_F_measure : angle_F = 60 :=
by
  -- Provide a proof by specifying sorry to indicate the proof is not complete
  sorry

end angle_F_measure_l490_49050


namespace gcd_1020_multiple_38962_l490_49055

-- Define that x is a multiple of 38962
def multiple_of (x n : ℤ) : Prop := ∃ k : ℤ, x = k * n

-- The main theorem statement
theorem gcd_1020_multiple_38962 (x : ℤ) (h : multiple_of x 38962) : Int.gcd 1020 x = 6 := 
sorry

end gcd_1020_multiple_38962_l490_49055


namespace prime_check_for_d1_prime_check_for_d2_l490_49083

-- Define d1 and d2
def d1 : ℕ := 9^4 - 9^3 + 9^2 - 9 + 1
def d2 : ℕ := 9^4 - 9^2 + 1

-- Prime checking function
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Stating the conditions and proofs
theorem prime_check_for_d1 : ¬ is_prime d1 :=
by {
  -- condition: ten 8's in base nine is divisible by d1 (5905) is not used here directly
  sorry
}

theorem prime_check_for_d2 : is_prime d2 :=
by {
  -- condition: twelve 8's in base nine is divisible by d2 (6481) is not used here directly
  sorry
}

end prime_check_for_d1_prime_check_for_d2_l490_49083


namespace combination_sum_eq_l490_49067

theorem combination_sum_eq :
  ∀ (n : ℕ), (2 * n ≥ 10 - 2 * n) ∧ (3 + n ≥ 2 * n) →
  Nat.choose (2 * n) (10 - 2 * n) + Nat.choose (3 + n) (2 * n) = 16 :=
by
  intro n h
  cases' h with h1 h2
  sorry

end combination_sum_eq_l490_49067


namespace hotel_rolls_l490_49025

theorem hotel_rolls (m n : ℕ) (rel_prime : Nat.gcd m n = 1) : 
  let num_nut_rolls := 3
  let num_cheese_rolls := 3
  let num_fruit_rolls := 3
  let total_rolls := 9
  let num_guests := 3
  let rolls_per_guest := 3
  let probability_first_guest := (3 / 9) * (3 / 8) * (3 / 7)
  let probability_second_guest := (2 / 6) * (2 / 5) * (2 / 4)
  let probability_third_guest := 1
  let overall_probability := probability_first_guest * probability_second_guest * probability_third_guest
  overall_probability = (9 / 70) → m = 9 ∧ n = 70 → m + n = 79 :=
by
  intros
  sorry

end hotel_rolls_l490_49025


namespace eagles_min_additional_wins_l490_49023

theorem eagles_min_additional_wins {N : ℕ} (eagles_initial_wins falcons_initial_wins : ℕ) (initial_games : ℕ)
  (total_games_won_fraction : ℚ) (required_fraction : ℚ) :
  eagles_initial_wins = 3 →
  falcons_initial_wins = 4 →
  initial_games = eagles_initial_wins + falcons_initial_wins →
  total_games_won_fraction = (3 + N) / (7 + N) →
  required_fraction = 9 / 10 →
  total_games_won_fraction = required_fraction →
  N = 33 :=
by
  sorry

end eagles_min_additional_wins_l490_49023


namespace investor_more_money_in_A_l490_49003

noncomputable def investment_difference 
  (investment_A : ℝ) 
  (investment_B : ℝ) 
  (yield_A : ℝ) 
  (yield_B : ℝ) :
  ℝ :=
investment_A * (1 + yield_A) - investment_B * (1 + yield_B)

theorem investor_more_money_in_A
  (investment_A : ℝ) 
  (investment_B : ℝ) 
  (yield_A : ℝ) 
  (yield_B : ℝ)
  (hA : investment_A = 300)
  (hB : investment_B = 200)
  (hYA : yield_A = 0.3)
  (hYB : yield_B = 0.5)
  :
  investment_difference investment_A investment_B yield_A yield_B = 90 := 
by
  sorry

end investor_more_money_in_A_l490_49003


namespace harry_pencils_remaining_l490_49066

def num_pencils_anna : ℕ := 50
def num_pencils_harry_initial := 2 * num_pencils_anna
def num_pencils_lost_harry := 19

def pencils_left_harry (pencils_anna : ℕ) (pencils_harry_initial : ℕ) (pencils_lost : ℕ) : ℕ :=
  pencils_harry_initial - pencils_lost

theorem harry_pencils_remaining : pencils_left_harry num_pencils_anna num_pencils_harry_initial num_pencils_lost_harry = 81 :=
by
  sorry

end harry_pencils_remaining_l490_49066


namespace find_Pete_original_number_l490_49059

noncomputable def PeteOriginalNumber (x : ℝ) : Prop :=
  5 * (3 * x + 15) = 200

theorem find_Pete_original_number : ∃ x : ℝ, PeteOriginalNumber x ∧ x = 25 / 3 :=
by
  sorry

end find_Pete_original_number_l490_49059


namespace shaded_region_area_l490_49089

-- Definitions of known conditions
def grid_section_1_area : ℕ := 3 * 3
def grid_section_2_area : ℕ := 4 * 5
def grid_section_3_area : ℕ := 5 * 6

def total_grid_area : ℕ := grid_section_1_area + grid_section_2_area + grid_section_3_area

def base_of_unshaded_triangle : ℕ := 15
def height_of_unshaded_triangle : ℕ := 6

def unshaded_triangle_area : ℕ := (base_of_unshaded_triangle * height_of_unshaded_triangle) / 2

-- Statement of the problem
theorem shaded_region_area : (total_grid_area - unshaded_triangle_area) = 14 :=
by
  -- Placeholder for the proof
  sorry

end shaded_region_area_l490_49089


namespace proof_problem_l490_49039

theorem proof_problem (p q : Prop) (hnpq : ¬ (p ∧ q)) (hnp : ¬ p) : ¬ p :=
by
  exact hnp

end proof_problem_l490_49039


namespace cube_negative_iff_l490_49047

theorem cube_negative_iff (x : ℝ) : x < 0 ↔ x^3 < 0 :=
sorry

end cube_negative_iff_l490_49047


namespace John_gave_the_store_20_dollars_l490_49045

def slurpee_cost : ℕ := 2
def change_received : ℕ := 8
def slurpees_bought : ℕ := 6
def total_money_given : ℕ := slurpee_cost * slurpees_bought + change_received

theorem John_gave_the_store_20_dollars : total_money_given = 20 := 
by 
  sorry

end John_gave_the_store_20_dollars_l490_49045


namespace angle_bisector_eqn_l490_49012

-- Define the vertices A, B, and C
def A : (ℝ × ℝ) := (4, 3)
def B : (ℝ × ℝ) := (-4, -1)
def C : (ℝ × ℝ) := (9, -7)

-- State the theorem with conditions and the given answer
theorem angle_bisector_eqn (A B C : (ℝ × ℝ)) (hA : A = (4, 3)) (hB : B = (-4, -1)) (hC : C = (9, -7)) :
  ∃ b c, (3:ℝ) * (3:ℝ) - b * (3:ℝ) + c = 0 ∧ b + c = -6 := 
by 
  use -1, -5
  simp
  sorry

end angle_bisector_eqn_l490_49012


namespace rachel_minutes_before_bed_l490_49091

-- Define the conditions in the Lean Lean.
def minutes_spent_solving_before_bed (m : ℕ) : Prop :=
  let problems_solved_before_bed := 5 * m
  let problems_finished_at_lunch := 16
  let total_problems_solved := 76
  problems_solved_before_bed + problems_finished_at_lunch = total_problems_solved

-- The statement we want to prove
theorem rachel_minutes_before_bed : ∃ m : ℕ, minutes_spent_solving_before_bed m ∧ m = 12 :=
sorry

end rachel_minutes_before_bed_l490_49091


namespace chi_squared_confidence_l490_49000

theorem chi_squared_confidence (K_squared : ℝ) :
  (99.5 / 100 : ℝ) = 0.995 → (K_squared ≥ 7.879) :=
sorry

end chi_squared_confidence_l490_49000


namespace generic_packages_needed_eq_2_l490_49041

-- Define parameters
def tees_per_generic_package : ℕ := 12
def tees_per_aero_package : ℕ := 2
def members_foursome : ℕ := 4
def tees_needed_per_member : ℕ := 20
def aero_packages_purchased : ℕ := 28

-- Calculate total tees needed and total tees obtained from aero packages
def total_tees_needed : ℕ := members_foursome * tees_needed_per_member
def aero_tees_obtained : ℕ := aero_packages_purchased * tees_per_aero_package
def generic_tees_needed : ℕ := total_tees_needed - aero_tees_obtained

-- Prove the number of generic packages needed is 2
theorem generic_packages_needed_eq_2 : 
  generic_tees_needed / tees_per_generic_package = 2 :=
  sorry

end generic_packages_needed_eq_2_l490_49041


namespace victor_earnings_l490_49009

variable (wage hours_mon hours_tue : ℕ)

def hourly_wage : ℕ := 6
def hours_worked_monday : ℕ := 5
def hours_worked_tuesday : ℕ := 5

theorem victor_earnings :
  (hours_worked_monday + hours_worked_tuesday) * hourly_wage = 60 :=
by
  sorry

end victor_earnings_l490_49009


namespace number_of_boys_l490_49013

variable {total_marbles : ℕ} (marbles_per_boy : ℕ := 10)
variable (H_total_marbles : total_marbles = 20)

theorem number_of_boys (total_marbles_marbs_eq_20 : total_marbles = 20) (marbles_per_boy_eq_10 : marbles_per_boy = 10) :
  total_marbles / marbles_per_boy = 2 :=
by {
  sorry
}

end number_of_boys_l490_49013


namespace greatest_AB_CBA_div_by_11_l490_49060

noncomputable def AB_CBA_max_value (A B C : ℕ) : ℕ := 10001 * A + 1010 * B + 100 * C + 10 * B + A

theorem greatest_AB_CBA_div_by_11 :
  ∃ (A B C : ℕ), A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ 
  2 * A - 2 * B + C % 11 = 0 ∧ 
  ∀ (A' B' C' : ℕ),
    A' ≠ B' ∧ B' ≠ C' ∧ C' ≠ A' ∧ 
    2 * A' - 2 * B' + C' % 11 = 0 → 
    AB_CBA_max_value A B C ≥ AB_CBA_max_value A' B' C' :=
  by sorry

end greatest_AB_CBA_div_by_11_l490_49060


namespace quadratic_solution_l490_49004

theorem quadratic_solution :
  (∀ x : ℝ, (x^2 - x - 1 = 0) ↔ (x = (1 + Real.sqrt 5) / 2 ∨ x = -(1 + Real.sqrt 5) / 2)) :=
by
  intro x
  rw [sub_eq_neg_add, sub_eq_neg_add]
  sorry

end quadratic_solution_l490_49004


namespace line_equation_l490_49033

variable (x y : ℝ)

theorem line_equation (x1 y1 m : ℝ) (h : x1 = -2 ∧ y1 = 3 ∧ m = 2) :
    -2 * x + y = 1 := by
  sorry

end line_equation_l490_49033


namespace bernie_savings_l490_49048

-- Defining conditions
def chocolates_per_week : ℕ := 2
def weeks : ℕ := 3
def chocolates_total : ℕ := chocolates_per_week * weeks
def local_store_cost_per_chocolate : ℕ := 3
def different_store_cost_per_chocolate : ℕ := 2

-- Defining the costs in both stores
def local_store_total_cost : ℕ := chocolates_total * local_store_cost_per_chocolate
def different_store_total_cost : ℕ := chocolates_total * different_store_cost_per_chocolate

-- The statement we want to prove
theorem bernie_savings : local_store_total_cost - different_store_total_cost = 6 :=
by
  sorry

end bernie_savings_l490_49048


namespace train_cross_bridge_time_l490_49065

def train_length : ℕ := 170
def train_speed_kmph : ℕ := 45
def bridge_length : ℕ := 205

def total_distance : ℕ := train_length + bridge_length
def train_speed_mps : ℕ := (train_speed_kmph * 1000) / 3600

theorem train_cross_bridge_time : (total_distance / train_speed_mps) = 30 := 
sorry

end train_cross_bridge_time_l490_49065


namespace total_students_l490_49031

-- Condition 1: 20% of students are below 8 years of age.
-- Condition 2: The number of students of 8 years of age is 72.
-- Condition 3: The number of students above 8 years of age is 2/3 of the number of students of 8 years of age.

variable {T : ℝ} -- Total number of students

axiom cond1 : 0.20 * T = (T - (72 + (2 / 3) * 72))
axiom cond2 : 72 = 72
axiom cond3 : (T - 72 - (2 / 3) * 72) = 0

theorem total_students : T = 150 := by
  -- Proof goes here
  sorry

end total_students_l490_49031


namespace find_bc_l490_49076

theorem find_bc (A : ℝ) (a : ℝ) (area : ℝ) (b c : ℝ) :
  A = 60 * (π / 180) → a = Real.sqrt 7 → area = (3 * Real.sqrt 3) / 2 →
  ((b = 3 ∧ c = 2) ∨ (b = 2 ∧ c = 3)) :=
by
  intros hA ha harea
  -- From the given area condition, derive bc = 6
  have h1 : b * c = 6 := sorry
  -- From the given conditions, derive b + c = 5
  have h2 : b + c = 5 := sorry
  -- Solve the system of equations to find possible values for b and c
  -- Using x² - S⋅x + P = 0 where x are roots, S = b + c, P = b⋅c
  have h3 : (b = 3 ∧ c = 2) ∨ (b = 2 ∧ c = 3) := sorry
  exact h3

end find_bc_l490_49076


namespace find_n_l490_49044

theorem find_n (n : ℕ) (h1 : ∃ k : ℕ, 12 - n = k * k) : n = 11 := 
by sorry

end find_n_l490_49044


namespace marla_errand_total_time_l490_49011

theorem marla_errand_total_time :
  let drive_time := 20
  let school_time := 70
  let total_time := 2 * drive_time + school_time
  total_time = 110 :=
by
  let drive_time := 20
  let school_time := 70
  let total_time := 2 * drive_time + school_time
  show total_time = 110
  sorry

end marla_errand_total_time_l490_49011


namespace cos_7theta_l490_49014

theorem cos_7theta (θ : ℝ) (h : Real.cos θ = 1/3) : Real.cos (7 * θ) = 49 / 2187 := 
  sorry

end cos_7theta_l490_49014


namespace fraction_always_defined_l490_49058

theorem fraction_always_defined (y : ℝ) : (y^2 + 1) ≠ 0 := 
by
  -- proof is not required
  sorry

end fraction_always_defined_l490_49058


namespace quadratic_eqn_a_range_l490_49017

variable {a : ℝ}

theorem quadratic_eqn_a_range (a : ℝ) : (∃ x : ℝ, (a - 3) * x^2 - 4 * x + 1 = 0) ↔ a ≠ 3 :=
by sorry

end quadratic_eqn_a_range_l490_49017


namespace pebbles_difference_l490_49082

def candy_pebbles : Nat := 4
def lance_pebbles : Nat := 3 * candy_pebbles

theorem pebbles_difference {candy_pebbles lance_pebbles : Nat} (h1 : candy_pebbles = 4) (h2 : lance_pebbles = 3 * candy_pebbles) : lance_pebbles - candy_pebbles = 8 := by
  sorry

end pebbles_difference_l490_49082


namespace ratio_of_areas_l490_49027

-- Definitions and conditions
variables (s r : ℝ)
variables (h1 : 4 * s = 4 * π * r)

-- Statement to prove
theorem ratio_of_areas (h1 : 4 * s = 4 * π * r) : s^2 / (π * r^2) = π := by
  sorry

end ratio_of_areas_l490_49027


namespace difference_of_squares_l490_49040

theorem difference_of_squares (a b c : ℤ) (h₁ : a < b) (h₂ : b < c) (h₃ : a % 2 = 0) (h₄ : b % 2 = 0) (h₅ : c % 2 = 0) (h₆ : a + b + c = 1992) :
  c^2 - a^2 = 5312 :=
by
  sorry

end difference_of_squares_l490_49040


namespace h_in_terms_of_f_l490_49074

-- Definitions based on conditions in a)
def reflect_y_axis (f : ℝ → ℝ) (x : ℝ) := f (-x)
def shift_left (f : ℝ → ℝ) (x : ℝ) (c : ℝ) := f (x + c)

-- Express h(x) in terms of f(x) based on conditions
theorem h_in_terms_of_f (f : ℝ → ℝ) (x : ℝ) :
  reflect_y_axis (shift_left f 2) x = f (-x - 2) :=
by
  sorry

end h_in_terms_of_f_l490_49074


namespace joann_third_day_lollipops_l490_49021

theorem joann_third_day_lollipops
  (a b c d e : ℕ)
  (h1 : b = a + 6)
  (h2 : c = b + 6)
  (h3 : d = c + 6)
  (h4 : e = d + 6)
  (h5 : a + b + c + d + e = 100) :
  c = 20 :=
by
  sorry

end joann_third_day_lollipops_l490_49021


namespace sum_of_squares_eq_expansion_l490_49069

theorem sum_of_squares_eq_expansion (a b : ℝ) : a^2 + b^2 + 2*a*b = (a + b)^2 :=
sorry

end sum_of_squares_eq_expansion_l490_49069


namespace ratio_of_part_to_whole_l490_49006

theorem ratio_of_part_to_whole : 
  (1 / 4) * (2 / 5) * P = 15 → 
  (40 / 100) * N = 180 → 
  P / N = 1 / 6 := 
by
  intros h1 h2
  sorry

end ratio_of_part_to_whole_l490_49006


namespace rectangle_is_possible_l490_49064

def possibleToFormRectangle (stick_lengths : List ℕ) : Prop :=
  ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ (a + b) * 2 = List.sum stick_lengths

noncomputable def sticks : List ℕ := List.range' 1 99

theorem rectangle_is_possible : possibleToFormRectangle sticks :=
sorry

end rectangle_is_possible_l490_49064
