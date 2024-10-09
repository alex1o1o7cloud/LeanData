import Mathlib

namespace bobby_candy_left_l1859_185978

theorem bobby_candy_left (initial_candies := 21) (first_eaten := 5) (second_eaten := 9) : 
  initial_candies - first_eaten - second_eaten = 7 :=
by
  -- Proof goes here
  sorry

end bobby_candy_left_l1859_185978


namespace compute_abc_l1859_185924

theorem compute_abc (a b c : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h_sum : a + b + c = 30) (h_frac : (1 : ℚ) / a + 1 / b + 1 / c + 450 / (a * b * c) = 1) : a * b * c = 1920 :=
by sorry

end compute_abc_l1859_185924


namespace solution_is_consecutive_even_integers_l1859_185908

def consecutive_even_integers_solution_exists : Prop :=
  ∃ (x y z w : ℕ), (x + y + z + w = 68) ∧ 
                   (y = x + 2) ∧ (z = x + 4) ∧ (w = x + 6) ∧
                   (x % 2 = 0) ∧ (y % 2 = 0) ∧ (z % 2 = 0) ∧ (w % 2 = 0)

theorem solution_is_consecutive_even_integers : consecutive_even_integers_solution_exists :=
sorry

end solution_is_consecutive_even_integers_l1859_185908


namespace cost_of_each_teddy_bear_is_15_l1859_185900

-- Definitions
variable (number_of_toys_cost_10 : ℕ := 28)
variable (cost_per_toy : ℕ := 10)
variable (number_of_teddy_bears : ℕ := 20)
variable (total_amount_in_wallet : ℕ := 580)

-- Theorem statement
theorem cost_of_each_teddy_bear_is_15 :
  (total_amount_in_wallet - (number_of_toys_cost_10 * cost_per_toy)) / number_of_teddy_bears = 15 :=
by
  -- proof goes here
  sorry

end cost_of_each_teddy_bear_is_15_l1859_185900


namespace chef_cherries_l1859_185980

theorem chef_cherries :
  ∀ (total_cherries used_cherries remaining_cherries : ℕ),
    total_cherries = 77 →
    used_cherries = 60 →
    remaining_cherries = total_cherries - used_cherries →
    remaining_cherries = 17 :=
by
  sorry

end chef_cherries_l1859_185980


namespace lemons_left_l1859_185925

/--
Prove that Cristine has 9 lemons left, given that she initially bought 12 lemons and gave away 1/4 of them.
-/
theorem lemons_left {initial_lemons : ℕ} (h1 : initial_lemons = 12) (fraction_given : ℚ) (h2 : fraction_given = 1 / 4) : initial_lemons - initial_lemons * fraction_given = 9 := by
  sorry

end lemons_left_l1859_185925


namespace remainder_142_to_14_l1859_185945

theorem remainder_142_to_14 (N k : ℤ) 
  (h : N = 142 * k + 110) : N % 14 = 8 :=
sorry

end remainder_142_to_14_l1859_185945


namespace general_term_b_l1859_185962

noncomputable def S (n : ℕ) : ℚ := sorry -- Define the sum of the first n terms sequence S_n
noncomputable def a (n : ℕ) : ℚ := sorry -- Define the sequence a_n
noncomputable def b (n : ℕ) : ℤ := Int.log 3 (|a n|) -- Define the sequence b_n using log base 3

-- Theorem stating the general formula for the sequence b_n
theorem general_term_b (n : ℕ) (h : 0 < n) :
  b n = -n :=
sorry -- We skip the proof, focusing on statement declaration

end general_term_b_l1859_185962


namespace absent_children_l1859_185955

-- Definitions
def total_children := 840
def bananas_per_child_present := 4
def bananas_per_child_if_all_present := 2
def total_bananas_if_all_present := total_children * bananas_per_child_if_all_present

-- The theorem to prove
theorem absent_children (A : ℕ) (P : ℕ) :
  P = total_children - A →
  total_bananas_if_all_present = P * bananas_per_child_present →
  A = 420 :=
by
  sorry

end absent_children_l1859_185955


namespace line_slope_l1859_185929

theorem line_slope : 
  (∀ (x y : ℝ), (x / 4 - y / 3 = -2) → (y = -3/4 * x - 6)) ∧ (∀ (x : ℝ), ∃ y : ℝ, (x / 4 - y / 3 = -2)) :=
by
  sorry

end line_slope_l1859_185929


namespace largest_multiple_of_seven_smaller_than_neg_85_l1859_185930

theorem largest_multiple_of_seven_smaller_than_neg_85 
  : ∃ k : ℤ, (k * 7 < -85) ∧ (∀ m : ℤ, (m * 7 < -85) → (m * 7 ≤ k * 7)) ∧ (k = -13) 
  := sorry

end largest_multiple_of_seven_smaller_than_neg_85_l1859_185930


namespace exists_real_m_l1859_185931

noncomputable def f (a : ℝ) (x : ℝ) := 4 * x + a * x ^ 2 - (2 / 3) * x ^ 3
noncomputable def g (x : ℝ) := 2 * x + (1 / 3) * x ^ 3

theorem exists_real_m (a : ℝ) (t : ℝ) (x1 x2 : ℝ) :
  (-1 : ℝ) ≤ a ∧ a ≤ 1 →
  (-1 : ℝ) ≤ t ∧ t ≤ 1 →
  f a x1 = g x1 ∧ f a x2 = g x2 →
  x1 ≠ 0 ∧ x2 ≠ 0 →
  x1 ≠ x2 →
  ∃ m : ℝ, (m ≥ 2 ∨ m ≤ -2) ∧ m^2 + t * m + 1 ≥ |x1 - x2| :=
sorry

end exists_real_m_l1859_185931


namespace smallest_possible_value_of_n_l1859_185928

theorem smallest_possible_value_of_n (n : ℕ) (h : lcm 60 n / gcd 60 n = 45) : n = 1080 :=
by
  sorry

end smallest_possible_value_of_n_l1859_185928


namespace smallest_collection_l1859_185965

def Yoongi_collected : ℕ := 4
def Jungkook_collected : ℕ := 6 * 3
def Yuna_collected : ℕ := 5

theorem smallest_collection : Yoongi_collected = 4 ∧ Yoongi_collected ≤ Jungkook_collected ∧ Yoongi_collected ≤ Yuna_collected := by
  sorry

end smallest_collection_l1859_185965


namespace all_items_weight_is_8040_l1859_185902

def weight_of_all_items : Real :=
  let num_tables := 15
  let settings_per_table := 8
  let backup_percentage := 0.25

  let weight_fork := 3.5
  let weight_knife := 4.0
  let weight_spoon := 4.5
  let weight_large_plate := 14.0
  let weight_small_plate := 10.0
  let weight_wine_glass := 7.0
  let weight_water_glass := 9.0
  let weight_table_decoration := 16.0

  let total_settings := (num_tables * settings_per_table) * (1 + backup_percentage)
  let weight_per_setting := (weight_fork + weight_knife + weight_spoon) + (weight_large_plate + weight_small_plate) + (weight_wine_glass + weight_water_glass)
  let total_weight_decorations := num_tables * weight_table_decoration

  let total_weight := total_settings * weight_per_setting + total_weight_decorations
  total_weight

theorem all_items_weight_is_8040 :
  weight_of_all_items = 8040 := sorry

end all_items_weight_is_8040_l1859_185902


namespace total_value_of_bills_in_cash_drawer_l1859_185910

-- Definitions based on conditions
def total_bills := 54
def five_dollar_bills := 20
def twenty_dollar_bills := total_bills - five_dollar_bills
def value_of_five_dollar_bills := 5
def value_of_twenty_dollar_bills := 20
def total_value_of_five_dollar_bills := five_dollar_bills * value_of_five_dollar_bills
def total_value_of_twenty_dollar_bills := twenty_dollar_bills * value_of_twenty_dollar_bills

-- Statement to prove
theorem total_value_of_bills_in_cash_drawer :
  total_value_of_five_dollar_bills + total_value_of_twenty_dollar_bills = 780 :=
by
  -- Proof goes here
  sorry

end total_value_of_bills_in_cash_drawer_l1859_185910


namespace total_dolls_count_l1859_185983

-- Define the conditions
def big_box_dolls : Nat := 7
def small_box_dolls : Nat := 4
def num_big_boxes : Nat := 5
def num_small_boxes : Nat := 9

-- State the theorem that needs to be proved
theorem total_dolls_count : 
  big_box_dolls * num_big_boxes + small_box_dolls * num_small_boxes = 71 := 
by
  sorry

end total_dolls_count_l1859_185983


namespace golf_tournament_percentage_increase_l1859_185939

theorem golf_tournament_percentage_increase:
  let electricity_bill := 800
  let cell_phone_expenses := electricity_bill + 400
  let golf_tournament_cost := 1440
  (golf_tournament_cost - cell_phone_expenses) / cell_phone_expenses * 100 = 20 :=
by
  sorry

end golf_tournament_percentage_increase_l1859_185939


namespace rented_apartment_years_l1859_185918

-- Given conditions
def months_in_year := 12
def payment_first_3_years_per_month := 300
def payment_remaining_years_per_month := 350
def total_paid := 19200
def first_period_years := 3

-- Define the total payment calculation
def total_payment (additional_years: ℕ): ℕ :=
  (first_period_years * months_in_year * payment_first_3_years_per_month) + 
  (additional_years * months_in_year * payment_remaining_years_per_month)

-- Main theorem statement
theorem rented_apartment_years (additional_years: ℕ) :
  total_payment additional_years = total_paid → (first_period_years + additional_years) = 5 :=
by
  intros h
  -- This skips the proof
  sorry

end rented_apartment_years_l1859_185918


namespace quadratic_root_q_value_l1859_185963

theorem quadratic_root_q_value
  (p q : ℝ)
  (h1 : ∃ r : ℝ, r = -3 ∧ 3 * r^2 + p * r + q = 0)
  (h2 : ∃ s : ℝ, -3 + s = -2) :
  q = -9 :=
sorry

end quadratic_root_q_value_l1859_185963


namespace two_times_x_equals_two_l1859_185932

theorem two_times_x_equals_two (x : ℝ) (h : x = 1) : 2 * x = 2 := by
  sorry

end two_times_x_equals_two_l1859_185932


namespace new_person_weight_increase_avg_l1859_185935

theorem new_person_weight_increase_avg
  (W : ℝ) -- total weight of the original 20 people
  (new_person_weight : ℝ) -- weight of the new person
  (h1 : (W - 80 + new_person_weight) = W + 20 * 15) -- condition given in the problem
  : new_person_weight = 380 := 
sorry

end new_person_weight_increase_avg_l1859_185935


namespace mary_money_left_l1859_185905

def drink_price (p : ℕ) : ℕ := p
def medium_pizza_price (p : ℕ) : ℕ := 2 * p
def large_pizza_price (p : ℕ) : ℕ := 3 * p
def drinks_cost (n : ℕ) (p : ℕ) : ℕ := n * drink_price p
def medium_pizzas_cost (n : ℕ) (p : ℕ) : ℕ := n * medium_pizza_price p
def large_pizza_cost (n : ℕ) (p : ℕ) : ℕ := n * large_pizza_price p
def total_cost (p : ℕ) : ℕ := drinks_cost 5 p + medium_pizzas_cost 2 p + large_pizza_cost 1 p
def money_left (initial_money : ℕ) (p : ℕ) : ℕ := initial_money - total_cost p

theorem mary_money_left (p : ℕ) : money_left 50 p = 50 - 12 * p := sorry

end mary_money_left_l1859_185905


namespace find_number_of_sides_l1859_185961

theorem find_number_of_sides (n : ℕ) (h : n - (n * (n - 3)) / 2 = 3) : n = 3 := 
sorry

end find_number_of_sides_l1859_185961


namespace min_n_of_inequality_l1859_185997

theorem min_n_of_inequality : 
  ∀ (n : ℕ), (1 ≤ n) → (1 / n - 1 / (n + 1) < 1 / 10) → (n = 3 ∨ ∃ (k : ℕ), k ≥ 3 ∧ n = k) :=
by
  sorry

end min_n_of_inequality_l1859_185997


namespace a_seq_formula_T_seq_sum_l1859_185927

-- Definition of the sequence \( \{a_n\} \)
def a_seq (n : ℕ) (p : ℤ) : ℤ := 2 * n + 5

-- Condition: Sum of the first n terms \( s_n = n^2 + pn \)
def s_seq (n : ℕ) (p : ℤ) : ℤ := n^2 + p * n

-- Condition: \( \{a_2, a_5, a_{10}\} \) form a geometric sequence
def is_geometric (a2 a5 a10 : ℤ) : Prop :=
  a2 * a10 = a5 * a5

-- Definition of the sequence \( \{b_n\} \)
def b_seq (n : ℕ) (p : ℤ) : ℚ := 1 + 5 / (a_seq n p * a_seq (n + 1) p)

-- Function to find the sum of first n terms of \( \{b_n\} \)
def T_seq (n : ℕ) (p : ℤ) : ℚ :=
  n + 5 * (1 / (7 : ℚ) - 1 / (2 * n + 7 : ℚ)) + n / (14 * n + 49 : ℚ)

theorem a_seq_formula (p : ℤ) : ∀ n, a_seq n p = 2 * n + 5 :=
by
  sorry

theorem T_seq_sum (p : ℤ) : ∀ n, T_seq n p = (14 * n^2 + 54 * n) / (14 * n + 49) :=
by
  sorry

end a_seq_formula_T_seq_sum_l1859_185927


namespace correct_reasoning_methods_l1859_185981

-- Definitions based on conditions
def reasoning_1 : String := "Inductive reasoning"
def reasoning_2 : String := "Deductive reasoning"
def reasoning_3 : String := "Analogical reasoning"

-- Proposition stating that the correct answer is D
theorem correct_reasoning_methods :
  (reasoning_1 = "Inductive reasoning") ∧
  (reasoning_2 = "Deductive reasoning") ∧
  (reasoning_3 = "Analogical reasoning") ↔
  (choice = "D") :=
by sorry

end correct_reasoning_methods_l1859_185981


namespace gcf_7fact_8fact_l1859_185958

-- Definitions based on the conditions
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

noncomputable def greatest_common_divisor (a b : ℕ) : ℕ :=
  Nat.gcd a b

-- Theorem statement
theorem gcf_7fact_8fact : greatest_common_divisor (factorial 7) (factorial 8) = 5040 := by
  sorry

end gcf_7fact_8fact_l1859_185958


namespace a_plus_c_eq_neg800_l1859_185916

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b
noncomputable def g (x : ℝ) (c d : ℝ) : ℝ := x^2 + c * x + d

theorem a_plus_c_eq_neg800 (a b c d : ℝ) (h1 : g (-a / 2) c d = 0)
  (h2 : f (-c / 2) a b = 0) (h3 : ∀ x, f x a b ≥ f (-a / 2) a b)
  (h4 : ∀ x, g x c d ≥ g (-c / 2) c d) (h5 : f (-a / 2) a b = g (-c / 2) c d)
  (h6 : f 200 a b = -200) (h7 : g 200 c d = -200) :
  a + c = -800 := sorry

end a_plus_c_eq_neg800_l1859_185916


namespace exists_a_b_not_multiple_p_l1859_185912

theorem exists_a_b_not_multiple_p (p : ℕ) (hp : Nat.Prime p) :
  ∃ a b : ℤ, ∀ m : ℤ, ¬ (m^3 + 2017 * a * m + b) ∣ (p : ℤ) :=
sorry

end exists_a_b_not_multiple_p_l1859_185912


namespace sum_of_digits_in_base_7_l1859_185953

theorem sum_of_digits_in_base_7 (A B C : ℕ) (hA : A > 0) (hB : B > 0) (hC : C > 0) (hA7 : A < 7) (hB7 : B < 7) (hC7 : C < 7)
  (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C) 
  (h_eqn : A * 49 + B * 7 + C + (B * 7 + C) = A * 49 + C * 7 + A) : 
  (A + B + C) = 14 := by
  sorry

end sum_of_digits_in_base_7_l1859_185953


namespace remainder_sum_mod9_l1859_185984

def a1 := 8243
def a2 := 8244
def a3 := 8245
def a4 := 8246

theorem remainder_sum_mod9 : ((a1 + a2 + a3 + a4) % 9) = 7 :=
by
  sorry

end remainder_sum_mod9_l1859_185984


namespace spadesuit_eval_l1859_185936

def spadesuit (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spadesuit_eval : spadesuit 2 (spadesuit 3 (spadesuit 1 2)) = 4 := 
by
  sorry

end spadesuit_eval_l1859_185936


namespace problem1_l1859_185917

theorem problem1 (x : ℝ) : abs (2 * x - 3) < 1 ↔ 1 < x ∧ x < 2 := sorry

end problem1_l1859_185917


namespace kids_in_group_l1859_185903

open Nat

theorem kids_in_group (A K : ℕ) (h1 : A + K = 11) (h2 : 8 * A = 72) : K = 2 := by
  sorry

end kids_in_group_l1859_185903


namespace max_det_bound_l1859_185975

noncomputable def max_det_estimate : ℕ := 327680 * 2^16

theorem max_det_bound (M : Matrix (Fin 17) (Fin 17) ℤ)
  (h : ∀ i j, M i j = 1 ∨ M i j = -1) :
  abs (Matrix.det M) ≤ max_det_estimate :=
sorry

end max_det_bound_l1859_185975


namespace sum_of_c_and_d_l1859_185946

theorem sum_of_c_and_d (c d : ℝ) :
  (∀ x : ℝ, x ≠ 2 → x ≠ -3 → (x - 2) * (x + 3) = x^2 + c * x + d) →
  c + d = -5 :=
by
  intros h
  sorry

end sum_of_c_and_d_l1859_185946


namespace junk_mail_per_house_l1859_185904

theorem junk_mail_per_house (total_junk_mail : ℕ) (houses_per_block : ℕ) 
  (h1 : total_junk_mail = 14) (h2 : houses_per_block = 7) : 
  (total_junk_mail / houses_per_block) = 2 :=
by 
  sorry

end junk_mail_per_house_l1859_185904


namespace multiples_of_7_between_15_and_200_l1859_185940

theorem multiples_of_7_between_15_and_200 : ∃ n : ℕ, n = 26 ∧ ∃ (a₁ a_n d : ℕ), 
  a₁ = 21 ∧ a_n = 196 ∧ d = 7 ∧ (a₁ + (n - 1) * d = a_n) := 
by
  sorry

end multiples_of_7_between_15_and_200_l1859_185940


namespace fruit_vendor_total_l1859_185922

theorem fruit_vendor_total (lemons_dozen avocados_dozen : ℝ) (dozen_size : ℝ) 
  (lemons : ℝ) (avocados : ℝ) (total_fruits : ℝ) 
  (h1 : lemons_dozen = 2.5) (h2 : avocados_dozen = 5) 
  (h3 : dozen_size = 12) (h4 : lemons = lemons_dozen * dozen_size) 
  (h5 : avocados = avocados_dozen * dozen_size) 
  (h6 : total_fruits = lemons + avocados) : 
  total_fruits = 90 := 
sorry

end fruit_vendor_total_l1859_185922


namespace find_functional_equation_solutions_l1859_185964

theorem find_functional_equation_solutions :
  (∀ f : ℝ → ℝ, (∀ x y : ℝ, x > 0 → y > 0 → f x * f (y * f x) = f (x + y)) →
    (∃ a > 0, ∀ x > 0, f x = 1 / (1 + a * x) ∨ ∀ x > 0, f x = 1)) :=
by
  sorry

end find_functional_equation_solutions_l1859_185964


namespace common_ratio_of_geometric_sequence_l1859_185950

variable (a : ℕ → ℝ) -- The geometric sequence {a_n}
variable (q : ℝ)     -- The common ratio

-- Conditions
axiom h1 : a 2 = 18
axiom h2 : a 4 = 8

theorem common_ratio_of_geometric_sequence :
  (∀ n : ℕ, a (n + 1) = a n * q) ∧ q^2 = 4/9 → q = 2/3 ∨ q = -2/3 := by
  sorry

end common_ratio_of_geometric_sequence_l1859_185950


namespace log2_of_fraction_l1859_185991

theorem log2_of_fraction : Real.logb 2 0.03125 = -5 := by
  sorry

end log2_of_fraction_l1859_185991


namespace parallel_lines_l1859_185974

theorem parallel_lines (m : ℝ) :
    (∀ x y : ℝ, x + (m+1) * y - 1 = 0 → mx + 2 * y - 1 = 0 → (m = 1 → False)) → m = -2 :=
by
  sorry

end parallel_lines_l1859_185974


namespace principal_equivalence_l1859_185982

-- Define the conditions
def SI : ℝ := 4020.75
def R : ℝ := 9
def T : ℝ := 5

-- Define the principal calculation
noncomputable def P := SI / (R * T / 100)

-- Prove that the principal P equals 8935
theorem principal_equivalence : P = 8935 := by
  sorry

end principal_equivalence_l1859_185982


namespace geom_mean_4_16_l1859_185942

theorem geom_mean_4_16 (x : ℝ) (h : x^2 = 4 * 16) : x = 8 ∨ x = -8 :=
by
  sorry

end geom_mean_4_16_l1859_185942


namespace focus_of_parabola_l1859_185993

theorem focus_of_parabola (f d : ℝ) :
  (∀ x : ℝ, x^2 + (4*x^2 - f)^2 = (4*x^2 - d)^2) → 8*f + 8*d = 1 → f^2 = d^2 → f = 1/16 :=
by
  intro hEq hCoeff hSq
  sorry

end focus_of_parabola_l1859_185993


namespace part1_part2_l1859_185959

def A : Set ℝ := {x | (x + 4) * (x - 2) > 0}
def B : Set ℝ := {y | ∃ x : ℝ, y = (x - 1)^2 + 1}
def C (a : ℝ) : Set ℝ := {x | -4 ≤ x ∧ x ≤ a}

theorem part1 : A ∩ B = {x : ℝ | x > 2} := 
by sorry

theorem part2 (a : ℝ) (h : (C a \ A) ⊆ C a) : 2 ≤ a :=
by sorry

end part1_part2_l1859_185959


namespace seats_in_16th_row_l1859_185944

def arithmetic_sequence (a d n : ℕ) : ℕ := a + (n - 1) * d

theorem seats_in_16th_row : arithmetic_sequence 5 2 16 = 35 := by
  sorry

end seats_in_16th_row_l1859_185944


namespace evaluate_f_of_composed_g_l1859_185956

def f (x : ℤ) : ℤ := 3 * x - 4
def g (x : ℤ) : ℤ := x + 2

theorem evaluate_f_of_composed_g :
  f (2 + g 3) = 17 :=
by
  sorry

end evaluate_f_of_composed_g_l1859_185956


namespace system_inequalities_1_system_inequalities_2_l1859_185971

theorem system_inequalities_1 (x: ℝ):
  (4 * (x + 1) ≤ 7 * x + 10) → (x - 5 < (x - 8)/3) → (-2 ≤ x ∧ x < 7 / 2) :=
by
  intros h1 h2
  sorry

theorem system_inequalities_2 (x: ℝ):
  (x - 3 * (x - 2) ≥ 4) → ((2 * x - 1) / 5 ≥ (x + 1) / 2) → (x ≤ -7) :=
by
  intros h1 h2
  sorry

end system_inequalities_1_system_inequalities_2_l1859_185971


namespace overall_gain_is_10_percent_l1859_185995

noncomputable def total_cost_price : ℝ := 700 + 500 + 300
noncomputable def total_gain : ℝ := 70 + 50 + 30
noncomputable def overall_gain_percentage : ℝ := (total_gain / total_cost_price) * 100

theorem overall_gain_is_10_percent :
  overall_gain_percentage = 10 :=
by
  sorry

end overall_gain_is_10_percent_l1859_185995


namespace sufficient_but_not_necessary_condition_l1859_185943

theorem sufficient_but_not_necessary_condition :
  (∀ x : ℝ, 0 < x → x < 4 → x^2 - 4 * x < 0) ∧ ¬ (∀ x : ℝ, x^2 - 4 * x < 0 → 0 < x ∧ x < 5) :=
sorry

end sufficient_but_not_necessary_condition_l1859_185943


namespace find_fourth_term_in_sequence_l1859_185947

theorem find_fourth_term_in_sequence (x: ℤ) (h1: 86 - 8 = 78) (h2: 2 - 86 = -84) (h3: x - 2 = -90) (h4: -12 - x = 76):
  x = -88 :=
sorry

end find_fourth_term_in_sequence_l1859_185947


namespace Marie_finish_time_l1859_185977

def Time := Nat × Nat -- Represents time as (hours, minutes)

def start_time : Time := (9, 0)
def finish_two_tasks_time : Time := (11, 20)
def total_tasks : Nat := 4

def minutes_since_start (t : Time) : Nat :=
  let (h, m) := t
  (h - 9) * 60 + m

def calculate_finish_time (start: Time) (two_tasks_finish: Time) (total_tasks: Nat) : Time :=
  let duration_two_tasks := minutes_since_start two_tasks_finish
  let duration_each_task := duration_two_tasks / 2
  let total_time := duration_each_task * total_tasks
  let total_minutes_after_start := total_time + minutes_since_start start
  let finish_hour := 9 + total_minutes_after_start / 60
  let finish_minute := total_minutes_after_start % 60
  (finish_hour, finish_minute)

theorem Marie_finish_time :
  calculate_finish_time start_time finish_two_tasks_time total_tasks = (13, 40) :=
by
  sorry

end Marie_finish_time_l1859_185977


namespace inequality_proof_equality_condition_l1859_185979

variable (a b c : ℝ)
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / c) + (c / b) ≥ (4 * a) / (a + b) := 
by
  sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / c) + (c / b) = (4 * a) / (a + b) ↔ a = b ∧ b = c :=
by
  sorry

end inequality_proof_equality_condition_l1859_185979


namespace equal_playing_time_l1859_185999

-- Given conditions
def total_minutes : Nat := 120
def number_of_children : Nat := 6
def children_playing_at_a_time : Nat := 2

-- Proof problem statement
theorem equal_playing_time :
  (children_playing_at_a_time * total_minutes) / number_of_children = 40 :=
by
  sorry

end equal_playing_time_l1859_185999


namespace sum_of_three_numbers_l1859_185994

theorem sum_of_three_numbers :
  ∃ (S1 S2 S3 : ℕ), 
    S2 = 72 ∧
    S1 = 2 * S2 ∧
    S3 = S1 / 3 ∧
    S1 + S2 + S3 = 264 := 
by
  sorry

end sum_of_three_numbers_l1859_185994


namespace common_ratio_of_geometric_series_l1859_185941

theorem common_ratio_of_geometric_series (a r : ℝ) (r_pos : 0 < r) (r_lt_one : r < 1) 
(h : (a / (1 - r)) = 81 * (a * r^4 / (1 - r))) : r = 1 / 3 :=
by
  have h_simplified : r^4 = 1 / 81 :=
    by
      sorry
  have r_value : r = (1 / 3) := by
      sorry
  exact r_value

end common_ratio_of_geometric_series_l1859_185941


namespace third_consecutive_even_sum_52_l1859_185957

theorem third_consecutive_even_sum_52
  (x : ℤ)
  (h : x + (x + 2) + (x + 4) + (x + 6) = 52) :
  x + 4 = 14 :=
by
  sorry

end third_consecutive_even_sum_52_l1859_185957


namespace tan_sub_eq_one_third_l1859_185926

theorem tan_sub_eq_one_third (α β : Real) (hα : Real.tan α = 3) (hβ : Real.tan β = 4/3) : 
  Real.tan (α - β) = 1/3 := by
  sorry

end tan_sub_eq_one_third_l1859_185926


namespace volume_tetrahedron_PXYZ_l1859_185921

noncomputable def volume_of_tetrahedron_PXYZ (x y z : ℝ) : ℝ :=
  (1 / 6) * x * y * z

theorem volume_tetrahedron_PXYZ :
  ∃ (x y z : ℝ), (x^2 + y^2 = 49) ∧ (y^2 + z^2 = 64) ∧ (z^2 + x^2 = 81) ∧
  volume_of_tetrahedron_PXYZ (Real.sqrt x) (Real.sqrt y) (Real.sqrt z) = 4 * Real.sqrt 11 := 
by {
  sorry
}

end volume_tetrahedron_PXYZ_l1859_185921


namespace percentage_of_boys_l1859_185919

theorem percentage_of_boys (total_students : ℕ) (ratio_boys_to_girls : ℕ) (ratio_girls_to_boys : ℕ) 
  (h_ratio : ratio_boys_to_girls = 3 ∧ ratio_girls_to_boys = 4 ∧ total_students = 42) : 
  (18 / 42) * 100 = 42.857 := 
by 
  sorry

end percentage_of_boys_l1859_185919


namespace no_real_pairs_arithmetic_prog_l1859_185985

theorem no_real_pairs_arithmetic_prog :
  ¬ ∃ a b : ℝ, (a = (1 / 2) * (8 + b)) ∧ (a + a * b = 2 * b) := by
sorry

end no_real_pairs_arithmetic_prog_l1859_185985


namespace complex_expression_equality_l1859_185938

theorem complex_expression_equality (i : ℂ) (h : i^2 = -1) : (1 + i)^16 - (1 - i)^16 = 0 := by
  sorry

end complex_expression_equality_l1859_185938


namespace cost_price_percentage_of_marked_price_l1859_185987

theorem cost_price_percentage_of_marked_price (MP CP : ℝ) (discount gain_percent : ℝ) 
  (h_discount : discount = 0.12) (h_gain_percent : gain_percent = 0.375) 
  (h_SP_def : SP = MP * (1 - discount))
  (h_SP_gain : SP = CP * (1 + gain_percent)) :
  CP / MP = 0.64 :=
by
  sorry

end cost_price_percentage_of_marked_price_l1859_185987


namespace solve_system_l1859_185949

theorem solve_system (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x * y = 4 * z) (h2 : x / y = 81) (h3 : x * z = 36) :
  x = 36 ∧ y = 4 / 9 ∧ z = 1 :=
by
  sorry

end solve_system_l1859_185949


namespace evaluate_expression_l1859_185934

theorem evaluate_expression (x y : ℝ) (h1 : x * y = -2) (h2 : x + y = 4) : x^2 * y + x * y^2 = -8 :=
by
  sorry

end evaluate_expression_l1859_185934


namespace square_area_l1859_185914

def edge1 (x : ℝ) := 5 * x - 18
def edge2 (x : ℝ) := 27 - 4 * x
def x_val : ℝ := 5

theorem square_area : edge1 x_val = edge2 x_val → (edge1 x_val) ^ 2 = 49 :=
by
  intro h
  -- Proof required here
  sorry

end square_area_l1859_185914


namespace negation_example_l1859_185951

theorem negation_example :
  (¬ (∀ a : ℕ, a > 0 → 2^a ≥ a^2)) ↔ (∃ a : ℕ, a > 0 ∧ 2^a < a^2) :=
by sorry

end negation_example_l1859_185951


namespace min_value_of_reciprocal_squares_l1859_185968

variable (a b : ℝ)

-- Define the two circle equations
def circle1 (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * a * x + a^2 - 4 = 0

def circle2 (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * b * y - 1 + 4 * b^2 = 0

-- The condition that the two circles are externally tangent and have three common tangents
def externallyTangent (a b : ℝ) : Prop :=
  -- From the derivation in the solution, we must have:
  (a^2 + 4 * b^2 = 9)

-- Ensure a and b are non-zero
def nonzero (a b : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0

-- State the main theorem to prove
theorem min_value_of_reciprocal_squares (h1 : externallyTangent a b) (h2 : nonzero a b) :
  (1 / a^2) + (1 / b^2) = 1 := 
sorry

end min_value_of_reciprocal_squares_l1859_185968


namespace eq_neg2_multi_l1859_185970

theorem eq_neg2_multi {m n : ℝ} (h : m = n) : -2 * m = -2 * n :=
by sorry

end eq_neg2_multi_l1859_185970


namespace fraction_to_decimal_l1859_185954

theorem fraction_to_decimal : (5 : ℝ) / 16 = 0.3125 := by
  sorry

end fraction_to_decimal_l1859_185954


namespace polynomial_simplification_l1859_185998

theorem polynomial_simplification (w : ℝ) : 
  3 * w + 4 - 6 * w - 5 + 7 * w + 8 - 9 * w - 10 + 2 * w ^ 2 = 2 * w ^ 2 - 5 * w - 3 :=
by
  sorry

end polynomial_simplification_l1859_185998


namespace identify_counterfeit_bag_l1859_185901

theorem identify_counterfeit_bag (n : ℕ) (w W : ℕ) (H : ∃ k : ℕ, k ≤ n ∧ W = w * (n * (n + 1) / 2) - k) : 
  ∃ bag_num, bag_num = w * (n * (n + 1) / 2) - W := by
  sorry

end identify_counterfeit_bag_l1859_185901


namespace outer_squares_equal_three_times_inner_squares_l1859_185920

theorem outer_squares_equal_three_times_inner_squares
  (a b c m_a m_b m_c : ℝ) 
  (h : m_a^2 + m_b^2 + m_c^2 = 3 / 4 * (a^2 + b^2 + c^2)) :
  a^2 + b^2 + c^2 = 3 * (m_a^2 + m_b^2 + m_c^2) := 
by 
  sorry

end outer_squares_equal_three_times_inner_squares_l1859_185920


namespace part_I_part_II_l1859_185913

noncomputable def f (x : ℝ) : ℝ := (Real.log (1 + x)) - (2 * x) / (x + 2)
noncomputable def g (x : ℝ) : ℝ := f x - (4 / (x + 2))

theorem part_I (x : ℝ) (h₀ : 0 < x) : f x > 0 := sorry

theorem part_II (a : ℝ) (h : ∀ x, g x < x + a) : -2 < a := sorry

end part_I_part_II_l1859_185913


namespace spokes_ratio_l1859_185969

theorem spokes_ratio (B : ℕ) (front_spokes : ℕ) (total_spokes : ℕ) 
  (h1 : front_spokes = 20) 
  (h2 : total_spokes = 60) 
  (h3 : front_spokes + B = total_spokes) : 
  B / front_spokes = 2 :=
by 
  sorry

end spokes_ratio_l1859_185969


namespace hyperbola_properties_l1859_185973

-- Define the conditions and the final statements we need to prove
theorem hyperbola_properties (a : ℝ) (ha : a > 2) (E : ℝ → ℝ → Prop)
  (hE : ∀ x y, E x y ↔ (x^2 / a^2 - y^2 / (a^2 - 4) = 1))
  (e : ℝ) (he : e = (Real.sqrt (a^2 + (a^2 - 4))) / a) :
  (∃ E' : ℝ → ℝ → Prop,
   ∀ x y, E' x y ↔ (x^2 / 9 - y^2 / 5 = 1)) ∧
  (∃ foci line: ℝ → ℝ → Prop,
   (∀ P : ℝ × ℝ, (E P.1 P.2) →
    (∃ Q : ℝ × ℝ, (P.1 - Q.1) * (P.1 + (Real.sqrt (2*a^2-4))) = 0 ∧ Q.2=0 ∧ 
     line (P.1) (P.2) ↔ P.1 - P.2 = 2))) :=
by
  sorry

end hyperbola_properties_l1859_185973


namespace line_segment_no_intersection_l1859_185967

theorem line_segment_no_intersection (a : ℝ) :
  (¬ ∃ t : ℝ, (0 ≤ t ∧ t ≤ 1 ∧ (1 - t) * (3 : ℝ) + t * (1 : ℝ) = 2 ∧ (1 - t) * (1 : ℝ) + t * (2 : ℝ) = (2 - (1 - t) * (3 : ℝ)) / a)) ->
  (a < -1 ∨ a > 0.5) :=
by
  sorry

end line_segment_no_intersection_l1859_185967


namespace power_mod_equiv_l1859_185986

theorem power_mod_equiv : 7^150 % 12 = 1 := 
  by
  sorry

end power_mod_equiv_l1859_185986


namespace smaller_number_is_17_l1859_185915

theorem smaller_number_is_17 (x y : ℕ) (h1 : x * y = 323) (h2 : x - y = 2) : y = 17 :=
sorry

end smaller_number_is_17_l1859_185915


namespace Simplify_division_l1859_185976

theorem Simplify_division :
  (5 * 10^9) / (2 * 10^5 * 5) = 5000 := sorry

end Simplify_division_l1859_185976


namespace neg_p_necessary_not_sufficient_neg_q_l1859_185923

def p (x : ℝ) := abs x < 1
def q (x : ℝ) := x^2 + x - 6 < 0

theorem neg_p_necessary_not_sufficient_neg_q :
  (¬ (∃ x, p x)) → (¬ (∃ x, q x)) ∧ ¬ ((¬ (∃ x, p x)) → (¬ (∃ x, q x))) :=
by
  sorry

end neg_p_necessary_not_sufficient_neg_q_l1859_185923


namespace factor_expression_l1859_185937

theorem factor_expression (x : ℝ) : 45 * x^3 + 135 * x^2 = 45 * x^2 * (x + 3) :=
  by
    sorry

end factor_expression_l1859_185937


namespace prove_root_property_l1859_185906

-- Define the quadratic equation and its roots
theorem prove_root_property :
  let r := -4 + Real.sqrt 226
  let s := -4 - Real.sqrt 226
  (r + 4) * (s + 4) = -226 :=
by
  -- the proof steps go here (omitted)
  sorry

end prove_root_property_l1859_185906


namespace stratified_sampling_group_l1859_185992

-- Definitions of conditions
def female_students : ℕ := 24
def male_students : ℕ := 36
def selected_females : ℕ := 8
def selected_males : ℕ := 12

-- Total number of ways to select the group
def total_combinations : ℕ := Nat.choose female_students selected_females * Nat.choose male_students selected_males

-- Proof of the problem
theorem stratified_sampling_group :
  (total_combinations = Nat.choose 24 8 * Nat.choose 36 12) :=
by
  sorry

end stratified_sampling_group_l1859_185992


namespace expand_expression_l1859_185933

theorem expand_expression : 
  ∀ (x : ℝ), (7 * x^3 - 5 * x + 2) * 4 * x^2 = 28 * x^5 - 20 * x^3 + 8 * x^2 :=
by
  intros x
  sorry

end expand_expression_l1859_185933


namespace quilt_percentage_shaded_l1859_185948

theorem quilt_percentage_shaded :
  ∀ (total_squares full_shaded half_shaded quarter_shaded : ℕ),
    total_squares = 25 →
    full_shaded = 4 →
    half_shaded = 8 →
    quarter_shaded = 4 →
    ((full_shaded + half_shaded * 1 / 2 + quarter_shaded * 1 / 2) / total_squares * 100 = 40) :=
by
  intros
  sorry

end quilt_percentage_shaded_l1859_185948


namespace minimum_components_needed_l1859_185952

-- Define the parameters of the problem
def production_cost_per_component : ℝ := 80
def shipping_cost_per_component : ℝ := 7
def fixed_monthly_cost : ℝ := 16500
def selling_price_per_component : ℝ := 198.33

-- Define the total cost as a function of the number of components
def total_cost (x : ℝ) : ℝ :=
  fixed_monthly_cost + (production_cost_per_component + shipping_cost_per_component) * x

-- Define the revenue as a function of the number of components
def revenue (x : ℝ) : ℝ :=
  selling_price_per_component * x

-- Define the theorem to be proved
theorem minimum_components_needed (x : ℝ) : x = 149 ↔ total_cost x ≤ revenue x := sorry

end minimum_components_needed_l1859_185952


namespace trader_loss_percentage_l1859_185966

def profit_loss_percentage (SP1 SP2 CP1 CP2 : ℚ) : ℚ :=
  ((SP1 + SP2) - (CP1 + CP2)) / (CP1 + CP2) * 100

theorem trader_loss_percentage :
  let SP1 := 325475
  let SP2 := 325475
  let CP1 := SP1 / (1 + 0.10)
  let CP2 := SP2 / (1 - 0.10)
  profit_loss_percentage SP1 SP2 CP1 CP2 = -1 := by
  sorry

end trader_loss_percentage_l1859_185966


namespace sin_seventeen_pi_over_four_l1859_185989

theorem sin_seventeen_pi_over_four : Real.sin (17 * Real.pi / 4) = Real.sqrt 2 / 2 := sorry

end sin_seventeen_pi_over_four_l1859_185989


namespace proof1_proof2_l1859_185988

variable (a : ℝ) (m n : ℝ)
axiom am_eq_two : a^m = 2
axiom an_eq_three : a^n = 3

theorem proof1 : a^(4 * m + 3 * n) = 432 := by
  sorry

theorem proof2 : a^(5 * m - 2 * n) = 32 / 9 := by
  sorry

end proof1_proof2_l1859_185988


namespace object_speed_approx_l1859_185960

theorem object_speed_approx :
  ∃ (speed : ℝ), abs (speed - 27.27) < 0.01 ∧
  (∀ (d : ℝ) (t : ℝ)
    (m : ℝ), 
    d = 80 ∧ t = 2 ∧ m = 5280 →
    speed = (d / m) / (t / 3600)) :=
by 
  sorry

end object_speed_approx_l1859_185960


namespace correct_option_d_l1859_185907

-- Definitions
variable (f : ℝ → ℝ)
variable (hf_even : ∀ x : ℝ, f x = f (-x))
variable (hf_inc : ∀ x y : ℝ, -1 ≤ x → x ≤ 0 → -1 ≤ y → y ≤ 0 → x ≤ y → f x ≤ f y)

-- Theorem statement
theorem correct_option_d :
  f (Real.sin (Real.pi / 12)) > f (Real.tan (Real.pi / 12)) :=
sorry

end correct_option_d_l1859_185907


namespace exists_close_pair_in_interval_l1859_185909

theorem exists_close_pair_in_interval (x1 x2 x3 : ℝ) (h1 : 0 ≤ x1 ∧ x1 < 1) (h2 : 0 ≤ x2 ∧ x2 < 1) (h3 : 0 ≤ x3 ∧ x3 < 1) :
  ∃ a b, (a = x1 ∨ a = x2 ∨ a = x3) ∧ (b = x1 ∨ b = x2 ∨ b = x3) ∧ a ≠ b ∧ |b - a| < 1 / 2 :=
sorry

end exists_close_pair_in_interval_l1859_185909


namespace arc_length_l1859_185911

-- Define the conditions
def radius (r : ℝ) := 2 * r + 2 * r = 8
def central_angle (θ : ℝ) := θ = 2 -- Given the central angle

-- Define the length of the arc
def length_of_arc (l r : ℝ) := l = r * 2

-- Theorem stating that given the sector conditions, the length of the arc is 4 cm
theorem arc_length (r l : ℝ) (h1 : central_angle 2) (h2 : radius r) (h3 : length_of_arc l r) : l = 4 :=
by
  sorry

end arc_length_l1859_185911


namespace rectangular_plot_width_l1859_185972

/-- Theorem: The width of a rectangular plot where the length is thrice its width and the area is 432 sq meters is 12 meters. -/
theorem rectangular_plot_width (w l : ℝ) (h₁ : l = 3 * w) (h₂ : l * w = 432) : w = 12 :=
by
  sorry

end rectangular_plot_width_l1859_185972


namespace total_rubber_bands_l1859_185990

theorem total_rubber_bands (harper_bands : ℕ) (brother_bands: ℕ):
  harper_bands = 15 →
  brother_bands = harper_bands - 6 →
  harper_bands + brother_bands = 24 :=
by
  intros h1 h2
  sorry

end total_rubber_bands_l1859_185990


namespace smallest_area_of_ellipse_l1859_185996

theorem smallest_area_of_ellipse 
    (a b : ℝ)
    (h1 : ∀ x y, (x - 2)^2 + y^2 < 4 → (x / a)^2 + (y / b)^2 < 1)
    (h2 : ∀ x y, (x + 2)^2 + y^2 < 4 → (x / a)^2 + (y / b)^2 < 1) :
    π * a * b = π :=
sorry

end smallest_area_of_ellipse_l1859_185996
