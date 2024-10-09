import Mathlib

namespace geometric_sequence_product_l720_72052

theorem geometric_sequence_product :
  ∃ a : ℕ → ℝ, 
    a 1 = 1 ∧ 
    a 5 = 16 ∧ 
    (∀ n, a (n + 1) = a n * r) ∧
    ∃ r : ℝ, 
      a 2 * a 3 * a 4 = 64 :=
by
  sorry

end geometric_sequence_product_l720_72052


namespace factor_expression_l720_72065

theorem factor_expression (x : ℝ) :
  (12 * x ^ 5 + 33 * x ^ 3 + 10) - (3 * x ^ 5 - 4 * x ^ 3 - 1) = x ^ 3 * (9 * x ^ 2 + 37) + 11 :=
by {
  -- Provide the skeleton for the proof using simplification
  sorry
}

end factor_expression_l720_72065


namespace units_digit_Fermat_5_l720_72054

def Fermat_number (n: ℕ) : ℕ :=
  2 ^ (2 ^ n) + 1

theorem units_digit_Fermat_5 : (Fermat_number 5) % 10 = 7 := by
  sorry

end units_digit_Fermat_5_l720_72054


namespace sixth_student_stickers_l720_72031

-- Define the given conditions.
def first_student_stickers := 29
def increment := 6

-- Define the number of stickers given to each subsequent student.
def stickers (n : ℕ) : ℕ :=
  first_student_stickers + n * increment

-- Theorem statement: the 6th student will receive 59 stickers.
theorem sixth_student_stickers : stickers 5 = 59 :=
by
  sorry

end sixth_student_stickers_l720_72031


namespace min_value_expression_l720_72080

theorem min_value_expression (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 8) :
  (a + 3 * b) * (b + 3 * c) * (a * c + 2) ≥ 64 :=
by
  sorry

end min_value_expression_l720_72080


namespace find_fruit_cost_l720_72012

-- Define the conditions
def muffin_cost : ℝ := 2
def francis_muffin_count : ℕ := 2
def francis_fruit_count : ℕ := 2
def kiera_muffin_count : ℕ := 2
def kiera_fruit_count : ℕ := 1
def total_cost : ℝ := 17

-- Define the cost of each fruit cup
variable (F : ℝ)

-- The statement to be proved
theorem find_fruit_cost (h : francis_muffin_count * muffin_cost 
                + francis_fruit_count * F 
                + kiera_muffin_count * muffin_cost 
                + kiera_fruit_count * F = total_cost) : 
                F = 1.80 :=
by {
  sorry
}

end find_fruit_cost_l720_72012


namespace horner_evaluation_at_2_l720_72097

noncomputable def f : ℕ → ℕ :=
  fun x => (((2 * x + 3) * x + 0) * x + 5) * x - 4

theorem horner_evaluation_at_2 : f 2 = 14 :=
  by
    sorry

end horner_evaluation_at_2_l720_72097


namespace taller_tree_height_l720_72094

variable (T S : ℝ)

theorem taller_tree_height (h1 : T - S = 20)
  (h2 : T - 10 = 3 * (S - 10)) : T = 40 :=
sorry

end taller_tree_height_l720_72094


namespace number_of_poison_frogs_l720_72096

theorem number_of_poison_frogs
  (total_frogs : ℕ) (tree_frogs : ℕ) (wood_frogs : ℕ) (poison_frogs : ℕ)
  (h₁ : total_frogs = 78)
  (h₂ : tree_frogs = 55)
  (h₃ : wood_frogs = 13)
  (h₄ : total_frogs = tree_frogs + wood_frogs + poison_frogs) :
  poison_frogs = 10 :=
by sorry

end number_of_poison_frogs_l720_72096


namespace sample_size_stratified_sampling_l720_72078

theorem sample_size_stratified_sampling 
  (teachers : ℕ) (male_students : ℕ) (female_students : ℕ) 
  (n : ℕ) (females_drawn : ℕ) 
  (total_people : ℕ := teachers + male_students + female_students) 
  (females_total : ℕ := female_students) 
  (proportion_drawn : ℚ := (females_drawn : ℚ) / females_total) :
  teachers = 200 → 
  male_students = 1200 → 
  female_students = 1000 → 
  females_drawn = 80 → 
  proportion_drawn = ((n : ℚ) / total_people) → 
  n = 192 :=
by
  sorry

end sample_size_stratified_sampling_l720_72078


namespace max_value_of_b_l720_72086

theorem max_value_of_b {m b : ℚ} (x : ℤ) 
  (line_eq : ∀ x : ℤ, 0 < x ∧ x ≤ 200 → 
    ¬ ∃ (y : ℤ), y = m * x + 3)
  (m_range : 1/3 < m ∧ m < b) :
  b = 69/208 :=
by
  sorry

end max_value_of_b_l720_72086


namespace min_Box_value_l720_72099

/-- The conditions are given as:
  1. (ax + b)(bx + a) = 24x^2 + Box * x + 24
  2. a, b, Box are distinct integers
  The task is to find the minimum possible value of Box.
-/
theorem min_Box_value :
  ∃ (a b Box : ℤ), a ≠ b ∧ a ≠ Box ∧ b ≠ Box ∧ (∀ x : ℤ, (a * x + b) * (b * x + a) = 24 * x^2 + Box * x + 24) ∧ Box = 52 := sorry

end min_Box_value_l720_72099


namespace find_initial_mean_l720_72003

/-- 
  The mean of 50 observations is M.
  One observation was wrongly taken as 23 but should have been 30.
  The corrected mean is 36.5.
  Prove that the initial mean M was 36.36.
-/
theorem find_initial_mean (M : ℝ) (h : 50 * 36.36 + 7 = 50 * 36.5) : 
  (500 * 36.36 - 7) = 1818 :=
sorry

end find_initial_mean_l720_72003


namespace simplify_expr_l720_72076

-- Define variables and conditions
variables (x y a b c : ℝ)

-- State the theorem
theorem simplify_expr : 
  (2 - y) * 24 * (x - y + 2 * (a - 2 - 3 * c) * a - 2 * b + c) = 
  2 + 4 * b^2 - a * b - c^2 :=
sorry

end simplify_expr_l720_72076


namespace mcgregor_books_finished_l720_72068

def total_books := 89
def floyd_books := 32
def books_left := 23

theorem mcgregor_books_finished : ∀ mg_books : Nat, mg_books = total_books - floyd_books - books_left → mg_books = 34 := 
by
  intro mg_books
  sorry

end mcgregor_books_finished_l720_72068


namespace composite_19_8n_plus_17_l720_72059

theorem composite_19_8n_plus_17 (n : ℕ) (h : n > 0) : ¬ Nat.Prime (19 * 8^n + 17) := 
by 
  sorry

end composite_19_8n_plus_17_l720_72059


namespace find_y_when_x_eq_4_l720_72066

theorem find_y_when_x_eq_4 (x y : ℝ) (k : ℝ) :
  (8 * y = k / x^3) →
  (y = 25) →
  (x = 2) →
  (exists y', x = 4 → y' = 25/8) :=
by
  sorry

end find_y_when_x_eq_4_l720_72066


namespace equalize_costs_l720_72060

theorem equalize_costs (X Y Z : ℝ) (hXY : X < Y) (hYZ : Y < Z) : (Y + Z - 2 * X) / 3 = (X + Y + Z) / 3 - X := by
  sorry

end equalize_costs_l720_72060


namespace molecular_weight_of_NH4I_correct_l720_72091

-- Define the atomic weights as given conditions
def atomic_weight_N : ℝ := 14.01
def atomic_weight_H : ℝ := 1.01
def atomic_weight_I : ℝ := 126.90

-- Define the calculation of the molecular weight of NH4I
def molecular_weight_NH4I : ℝ :=
  atomic_weight_N + 4 * atomic_weight_H + atomic_weight_I

-- Theorem stating the molecular weight of NH4I is 144.95 g/mol
theorem molecular_weight_of_NH4I_correct : molecular_weight_NH4I = 144.95 :=
by
  sorry

end molecular_weight_of_NH4I_correct_l720_72091


namespace geom_seq_a3_value_l720_72049

theorem geom_seq_a3_value (a_n : ℕ → ℝ) (h1 : ∃ r : ℝ, ∀ n : ℕ, a_n (n+1) = a_n (1) * r^n) 
                          (h2 : a_n (2) * a_n (4) = 2 * a_n (3) - 1) :
  a_n (3) = 1 :=
sorry

end geom_seq_a3_value_l720_72049


namespace total_birds_in_tree_l720_72088

theorem total_birds_in_tree (bluebirds cardinals swallows : ℕ) 
  (h1 : swallows = 2) 
  (h2 : swallows = bluebirds / 2) 
  (h3 : cardinals = 3 * bluebirds) : 
  swallows + bluebirds + cardinals = 18 := 
by 
  sorry

end total_birds_in_tree_l720_72088


namespace ratio_m_over_n_l720_72018

theorem ratio_m_over_n : 
  ∀ (m n : ℕ) (a b : ℝ),
  let α := (3 : ℝ) / 4
  let β := (19 : ℝ) / 20
  (a = α * b) →
  (a = β * (a * m + b * n) / (m + n)) →
  (n ≠ 0) →
  m / n = 8 / 9 :=
by
  intros m n a b α β hα hβ hn
  sorry

end ratio_m_over_n_l720_72018


namespace cosine_sum_formula_l720_72051

theorem cosine_sum_formula
  (α : Real) 
  (h1 : Real.sin (Real.pi - α) = 4 / 5) 
  (h2 : 0 < α ∧ α < Real.pi / 2) : 
  Real.cos (α + Real.pi / 4) = -Real.sqrt 2 / 10 := 
by 
  sorry

end cosine_sum_formula_l720_72051


namespace probability_of_rolling_two_exactly_four_times_in_five_rolls_l720_72042

theorem probability_of_rolling_two_exactly_four_times_in_five_rolls :
  let p := (1 / 6)
  let q := (5 / 6)
  let n := 5
  let k := 4
  let probability := (n.choose k) * p^k * q^(n-k)
  probability = (25 / 7776) :=
by
  let p := (1 / 6)
  let q := (5 / 6)
  let n := 5
  let k := 4
  let probability := (n.choose k) * p^k * q^(n - k)
  have h : probability = (25 / 7776) := sorry
  exact h

end probability_of_rolling_two_exactly_four_times_in_five_rolls_l720_72042


namespace travis_total_cost_l720_72044

namespace TravelCost

def cost_first_leg : ℝ := 1500
def discount_first_leg : ℝ := 0.25
def fees_first_leg : ℝ := 100

def cost_second_leg : ℝ := 800
def discount_second_leg : ℝ := 0.20
def fees_second_leg : ℝ := 75

def cost_third_leg : ℝ := 1200
def discount_third_leg : ℝ := 0.35
def fees_third_leg : ℝ := 120

def discounted_cost (cost : ℝ) (discount : ℝ) : ℝ :=
  cost - (cost * discount)

def total_leg_cost (cost : ℝ) (discount : ℝ) (fees : ℝ) : ℝ :=
  (discounted_cost cost discount) + fees

def total_journey_cost : ℝ :=
  total_leg_cost cost_first_leg discount_first_leg fees_first_leg + 
  total_leg_cost cost_second_leg discount_second_leg fees_second_leg + 
  total_leg_cost cost_third_leg discount_third_leg fees_third_leg

theorem travis_total_cost : total_journey_cost = 2840 := by
  sorry

end TravelCost

end travis_total_cost_l720_72044


namespace slope_negative_l720_72017

theorem slope_negative (k b m n : ℝ) (h₁ : k ≠ 0) (h₂ : m < n) 
  (ha : m = k * 1 + b) (hb : n = k * -1 + b) : k < 0 :=
by
  sorry

end slope_negative_l720_72017


namespace find_the_number_l720_72057

theorem find_the_number :
  ∃ x : ℕ, 72519 * x = 724827405 ∧ x = 10005 :=
by
  sorry

end find_the_number_l720_72057


namespace water_fee_20_water_fee_55_l720_72029

-- Define the water charge method as a function
def water_fee (a : ℕ) : ℝ :=
  if a ≤ 15 then 2 * a else 2.5 * a - 7.5

-- Prove the specific cases
theorem water_fee_20 :
  water_fee 20 = 42.5 :=
by sorry

theorem water_fee_55 :
  (∃ a : ℕ, water_fee a = 55) ↔ (a = 25) :=
by sorry

end water_fee_20_water_fee_55_l720_72029


namespace y_value_l720_72085

theorem y_value (x y : ℤ) (h1 : x^2 = y + 7) (h2 : x = -6) : y = 29 :=
by
  sorry

end y_value_l720_72085


namespace base7_and_base13_addition_l720_72023

def base7_to_nat (a b c : ℕ) : ℕ := a * 49 + b * 7 + c

def base13_to_nat (a b c : ℕ) : ℕ := a * 169 + b * 13 + c

theorem base7_and_base13_addition (a b c d e f : ℕ) :
  a = 5 → b = 3 → c = 6 → d = 4 → e = 12 → f = 5 →
  base7_to_nat a b c + base13_to_nat d e f = 1109 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  unfold base7_to_nat base13_to_nat
  sorry

end base7_and_base13_addition_l720_72023


namespace diane_postage_problem_l720_72043

-- Definition of stamps
def stamps : List (ℕ × ℕ) := [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)]

-- Define a function to compute the number of arrangements that sums to a target value
def arrangements_sum_to (target : ℕ) (stamps : List (ℕ × ℕ)) : ℕ :=
  sorry -- Implementation detail is skipped

-- The main theorem to prove
theorem diane_postage_problem :
  arrangements_sum_to 15 stamps = 271 :=
by sorry

end diane_postage_problem_l720_72043


namespace subtracted_number_l720_72038

def least_sum_is (x y z : ℤ) (a : ℤ) : Prop :=
  (x - a) * (y - 5) * (z - 2) = 1000 ∧ x + y + z = 7

theorem subtracted_number (x y z a : ℤ) (h : least_sum_is x y z a) : a = 30 :=
sorry

end subtracted_number_l720_72038


namespace distance_between_D_and_E_l720_72025

theorem distance_between_D_and_E 
  (A B C D E P : Type)
  (d_AB : ℕ) (d_BC : ℕ) (d_AC : ℕ) (d_PC : ℕ) 
  (AD_parallel_BC : Prop) (AB_parallel_CE : Prop) 
  (distance_DE : ℕ) :
  d_AB = 15 →
  d_BC = 18 → 
  d_AC = 21 → 
  d_PC = 7 → 
  AD_parallel_BC →
  AB_parallel_CE →
  distance_DE = 15 :=
by
  sorry

end distance_between_D_and_E_l720_72025


namespace probability_same_spot_l720_72015

theorem probability_same_spot :
  let students := ["A", "B"]
  let spots := ["Spot 1", "Spot 2"]
  let total_outcomes := [(("A", "Spot 1"), ("B", "Spot 1")),
                         (("A", "Spot 1"), ("B", "Spot 2")),
                         (("A", "Spot 2"), ("B", "Spot 1")),
                         (("A", "Spot 2"), ("B", "Spot 2"))]
  let favorable_outcomes := [(("A", "Spot 1"), ("B", "Spot 1")),
                             (("A", "Spot 2"), ("B", "Spot 2"))]
  ∀ (students : List String) (spots : List String)
    (total_outcomes favorable_outcomes : List ((String × String) × (String × String))),
  (students = ["A", "B"]) →
  (spots = ["Spot 1", "Spot 2"]) →
  (total_outcomes = [(("A", "Spot 1"), ("B", "Spot 1")),
                     (("A", "Spot 1"), ("B", "Spot 2")),
                     (("A", "Spot 2"), ("B", "Spot 1")),
                     (("A", "Spot 2"), ("B", "Spot 2"))]) →
  (favorable_outcomes = [(("A", "Spot 1"), ("B", "Spot 1")),
                         (("A", "Spot 2"), ("B", "Spot 2"))]) →
  favorable_outcomes.length / total_outcomes.length = 1 / 2 := 
by
  intros
  sorry

end probability_same_spot_l720_72015


namespace marble_ratio_l720_72020

theorem marble_ratio (W L M : ℕ) (h1 : W = 16) (h2 : L = W + W / 4) (h3 : W + L + M = 60) :
  M / (W + L) = 2 / 3 := 
sorry

end marble_ratio_l720_72020


namespace arithmetic_identity_l720_72077

theorem arithmetic_identity : Real.sqrt 16 + ((1/2) ^ (-2:ℤ)) = 8 := 
by 
  sorry

end arithmetic_identity_l720_72077


namespace gas_price_increase_l720_72034

theorem gas_price_increase (P C : ℝ) (x : ℝ) 
  (h1 : P * C = P * (1 + x) * 1.10 * C * (1 - 0.27272727272727)) :
  x = 0.25 :=
by
  -- The proof will be filled here
  sorry

end gas_price_increase_l720_72034


namespace compare_f_values_l720_72050

variable (a : ℝ) (f : ℝ → ℝ) (m n : ℝ)

theorem compare_f_values (h_a : 0 < a ∧ a < 1)
    (h_f : ∀ x > 0, f (Real.logb a x) = a * (x^2 - 1) / (x * (a^2 - 1)))
    (h_mn : m > n ∧ n > 0 ∧ m > 0) :
    f (1 / n) > f (1 / m) := by 
  sorry

end compare_f_values_l720_72050


namespace part_I_part_II_l720_72007

noncomputable def f (x a : ℝ) : ℝ := |x - a| - |2 * x - 1|

theorem part_I (a : ℝ) (x : ℝ) (h : a = 2) :
    f x a + 3 ≥ 0 ↔ -4 ≤ x ∧ x ≤ 2 := 
by
    -- problem restatement
    sorry

theorem part_II (a : ℝ) (h : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → f x a ≤ 3) :
    -3 ≤ a ∧ a ≤ 5 := 
by
    -- problem restatement
    sorry

end part_I_part_II_l720_72007


namespace not_possible_100_odd_sequence_l720_72000

def is_square_mod_8 (n : ℤ) : Prop :=
  n % 8 = 0 ∨ n % 8 = 1 ∨ n % 8 = 4

def sum_consecutive_is_square_mod_8 (seq : List ℤ) (k : ℕ) : Prop :=
  ∀ i : ℕ, i + k ≤ seq.length →
  is_square_mod_8 (seq.drop i |>.take k |>.sum)

def valid_odd_sequence (seq : List ℤ) : Prop :=
  seq.length = 100 ∧
  (∀ n ∈ seq, n % 2 = 1) ∧
  sum_consecutive_is_square_mod_8 seq 5 ∧
  sum_consecutive_is_square_mod_8 seq 9

theorem not_possible_100_odd_sequence :
  ¬∃ seq : List ℤ, valid_odd_sequence seq :=
by
  sorry

end not_possible_100_odd_sequence_l720_72000


namespace problem1_solution_problem2_solution_l720_72013

noncomputable def problem1 (a b : ℝ) (A B : ℝ) (h1 : b * Real.cos A - a * Real.sin B = 0) : Real := 
  A

noncomputable def problem2 (a b c : ℝ) (A : ℝ) (area : ℝ) (h1 : b = Real.sqrt 2) (h2 : A = Real.pi / 4) (h3 : area = 1) : Real :=
  a

theorem problem1_solution (a b : ℝ) (A B : ℝ) (h1 : b * Real.cos A - a * Real.sin B = 0) :
  problem1 a b A B h1 = Real.pi / 4 :=
sorry

theorem problem2_solution (a b c : ℝ) (A : ℝ) (area : ℝ) (h1 : b = Real.sqrt 2) (h2 : A = Real.pi / 4) (h3 : area = 1) :
  problem2 a b c A area h1 h2 h3 = Real.sqrt 2 :=
sorry

end problem1_solution_problem2_solution_l720_72013


namespace sum_series_equals_half_l720_72004

theorem sum_series_equals_half :
  ∑' n, 1 / (n * (n+1) * (n+2)) = 1 / 2 :=
sorry

end sum_series_equals_half_l720_72004


namespace price_reduction_correct_l720_72036

theorem price_reduction_correct :
  ∃ x : ℝ, (0.3 - x) * (500 + 4000 * x) = 180 ∧ x = 0.1 :=
by
  sorry

end price_reduction_correct_l720_72036


namespace student_selection_problem_l720_72006

noncomputable def total_selections : ℕ :=
  let C := Nat.choose
  let A := Nat.factorial
  (C 3 1 * C 3 2 + C 3 2 * C 3 1 + C 3 3) * A 3

theorem student_selection_problem :
  total_selections = 114 :=
by
  sorry

end student_selection_problem_l720_72006


namespace scientific_notation_of_216000_l720_72087

theorem scientific_notation_of_216000 :
  216000 = 2.16 * 10^5 :=
sorry

end scientific_notation_of_216000_l720_72087


namespace debts_equal_in_25_days_l720_72045

-- Define the initial debts and the interest rates
def Darren_initial_debt : ℝ := 200
def Darren_interest_rate : ℝ := 0.08
def Fergie_initial_debt : ℝ := 300
def Fergie_interest_rate : ℝ := 0.04

-- Define the debts as a function of days passed t
def Darren_debt (t : ℝ) : ℝ := Darren_initial_debt * (1 + Darren_interest_rate * t)
def Fergie_debt (t : ℝ) : ℝ := Fergie_initial_debt * (1 + Fergie_interest_rate * t)

-- Prove that Darren and Fergie will owe the same amount in 25 days
theorem debts_equal_in_25_days : ∃ t, Darren_debt t = Fergie_debt t ∧ t = 25 := by
  sorry

end debts_equal_in_25_days_l720_72045


namespace candy_total_l720_72047

theorem candy_total (r b : ℕ) (hr : r = 145) (hb : b = 3264) : r + b = 3409 := by
  -- We can use Lean's rewrite tactic to handle the equalities, but since proof is skipped,
  -- it's not necessary to write out detailed tactics here.
  sorry

end candy_total_l720_72047


namespace shaded_area_equals_l720_72041

noncomputable def area_shaded_figure (R : ℝ) : ℝ :=
  let α := (60 : ℝ) * (Real.pi / 180)
  (2 * Real.pi * R^2) / 3

theorem shaded_area_equals : ∀ R : ℝ, area_shaded_figure R = (2 * Real.pi * R^2) / 3 := sorry

end shaded_area_equals_l720_72041


namespace broccoli_difference_l720_72067

theorem broccoli_difference (A : ℕ) (s : ℕ) (s' : ℕ)
  (h1 : A = 1600)
  (h2 : s = Nat.sqrt A)
  (h3 : s' < s)
  (h4 : (s')^2 < A)
  (h5 : A - (s')^2 = 79) :
  (1600 - (s')^2) = 79 :=
by
  sorry

end broccoli_difference_l720_72067


namespace factorize_x2_minus_2x_plus_1_l720_72032

theorem factorize_x2_minus_2x_plus_1 :
  ∀ (x : ℝ), x^2 - 2 * x + 1 = (x - 1)^2 :=
by
  intro x
  linarith

end factorize_x2_minus_2x_plus_1_l720_72032


namespace nuts_in_tree_l720_72075

def num_squirrels := 4
def num_nuts := 2

theorem nuts_in_tree :
  ∀ (S N : ℕ), S = num_squirrels → S - N = 2 → N = num_nuts :=
by
  intros S N hS hDiff
  sorry

end nuts_in_tree_l720_72075


namespace pencils_in_all_l720_72009

/-- Eugene's initial number of pencils -/
def initial_pencils : ℕ := 51

/-- Pencils Eugene gets from Joyce -/
def additional_pencils : ℕ := 6

/-- Total number of pencils Eugene has in all -/
def total_pencils : ℕ :=
  initial_pencils + additional_pencils

/-- Proof that Eugene has 57 pencils in all -/
theorem pencils_in_all : total_pencils = 57 := by
  sorry

end pencils_in_all_l720_72009


namespace distance_from_center_to_tangent_chord_l720_72053

theorem distance_from_center_to_tangent_chord
  (R a m x : ℝ)
  (h1 : m^2 = 4 * R^2)
  (h2 : 16 * R^2 * x^4 - 16 * R^2 * x^2 * (a^2 + R^2) + 16 * a^4 * R^4 - a^2 * (4 * R^2 - m^2)^2 = 0) :
  x = R :=
sorry

end distance_from_center_to_tangent_chord_l720_72053


namespace range_of_a_l720_72019

def A (a : ℝ) := ({-1, 0, a} : Set ℝ)
def B := {x : ℝ | 0 < x ∧ x < 1}

theorem range_of_a (a : ℝ) (h : A a ∩ B ≠ ∅) : 0 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l720_72019


namespace greatest_integer_sum_l720_72058

def floor (x : ℚ) : ℤ := ⌊x⌋

theorem greatest_integer_sum :
  floor (2017 * 3 / 11) + 
  floor (2017 * 4 / 11) + 
  floor (2017 * 5 / 11) + 
  floor (2017 * 6 / 11) + 
  floor (2017 * 7 / 11) + 
  floor (2017 * 8 / 11) = 6048 :=
  by sorry

end greatest_integer_sum_l720_72058


namespace range_of_a_l720_72022

theorem range_of_a (a : ℝ) :
  ((∀ x : ℝ, a * x^2 + a * x - 1 < 0) ↔ (-4 < a ∧ a ≤ 0)) :=
sorry

end range_of_a_l720_72022


namespace isosceles_triangle_perimeter_l720_72095

-- Definitions and conditions
-- Define the lengths of the three sides of the triangle
def a : ℕ := 3
def b : ℕ := 8

-- Define that the triangle is isosceles
def is_isosceles_triangle := 
  (a = a) ∨ (b = b) ∨ (a = b)

-- Perimeter of the triangle
def perimeter (x y z : ℕ) := x + y + z

-- The theorem we need to prove
theorem isosceles_triangle_perimeter : is_isosceles_triangle → (a + b + b = 19) :=
by
  intro h
  sorry

end isosceles_triangle_perimeter_l720_72095


namespace arithmetic_sequence_a₄_l720_72027

open Int

noncomputable def S (a₁ d n : ℤ) : ℤ :=
  n * a₁ + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_a₄ {a₁ d : ℤ}
  (h₁ : S a₁ d 5 = 15) (h₂ : S a₁ d 9 = 63) :
  a₁ + 3 * d = 5 :=
  sorry

end arithmetic_sequence_a₄_l720_72027


namespace mass_fraction_K2SO4_l720_72079

theorem mass_fraction_K2SO4 :
  (2.61 * 100 / 160) = 1.63 :=
by
  -- Proof details are not required as per instructions
  sorry

end mass_fraction_K2SO4_l720_72079


namespace least_n_l720_72064

theorem least_n (n : ℕ) (h : 0 < n ∧ (1 / n - 1 / (n + 1) < 1 / 15)) : n ≥ 4 :=
by sorry

end least_n_l720_72064


namespace correct_factorization_l720_72062

theorem correct_factorization :
  ∀ (x : ℝ), -x^2 + 2*x - 1 = - (x - 1)^2 :=
by
  intro x
  sorry

end correct_factorization_l720_72062


namespace initial_markup_percentage_l720_72030

theorem initial_markup_percentage
  (cost_price : ℝ := 100)
  (profit_percentage : ℝ := 14)
  (discount_percentage : ℝ := 5)
  (selling_price : ℝ := cost_price * (1 + profit_percentage / 100))
  (x : ℝ := 20) :
  (cost_price + cost_price * x / 100) * (1 - discount_percentage / 100) = selling_price := by
  sorry

end initial_markup_percentage_l720_72030


namespace paint_fraction_l720_72039

variable (T C : ℕ) (h : T = 60) (t : ℕ) (partial_t : ℚ)

theorem paint_fraction (hT : T = 60) (ht : t = 12) : partial_t = t / T := by
  rw [ht, hT]
  norm_num
  sorry

end paint_fraction_l720_72039


namespace sum_of_z_values_l720_72026

def f (x : ℚ) : ℚ := x^2 + x + 1

theorem sum_of_z_values : ∃ z₁ z₂ : ℚ, f (4 * z₁) = 12 ∧ f (4 * z₂) = 12 ∧ (z₁ + z₂ = - 1 / 12) :=
by
  sorry

end sum_of_z_values_l720_72026


namespace number_99_in_column_4_l720_72040

-- Definition of the arrangement rule
def column_of (num : ℕ) : ℕ :=
  ((num % 10) + 4) / 2 % 5 + 1

theorem number_99_in_column_4 : 
  column_of 99 = 4 :=
by
  sorry

end number_99_in_column_4_l720_72040


namespace non_degenerate_ellipse_condition_l720_72055

theorem non_degenerate_ellipse_condition (k : ℝ) :
  (∃ x y : ℝ, 9 * x^2 + y^2 - 18 * x - 2 * y = k) ↔ k > -10 :=
sorry

end non_degenerate_ellipse_condition_l720_72055


namespace calculation_correct_l720_72092

noncomputable def calc_expression : Float :=
  20.17 * 69 + 201.7 * 1.3 - 8.2 * 1.7

theorem calculation_correct : calc_expression = 1640 := 
  by 
    sorry

end calculation_correct_l720_72092


namespace option_one_cost_option_two_cost_cost_effectiveness_l720_72005

-- Definition of costs based on conditions
def price_of_suit : ℕ := 500
def price_of_tie : ℕ := 60
def discount_option_one (x : ℕ) : ℕ := 60 * x + 8800
def discount_option_two (x : ℕ) : ℕ := 54 * x + 9000

-- Theorem statements
theorem option_one_cost (x : ℕ) (hx : x > 20) : discount_option_one x = 60 * x + 8800 :=
by sorry

theorem option_two_cost (x : ℕ) (hx : x > 20) : discount_option_two x = 54 * x + 9000 :=
by sorry

theorem cost_effectiveness (x : ℕ) (hx : x = 30) : discount_option_one x < discount_option_two x :=
by sorry

end option_one_cost_option_two_cost_cost_effectiveness_l720_72005


namespace find_x_l720_72046

theorem find_x (x : ℝ) (a b : ℝ) (h₀ : a * b = 4 * a - 2 * b)
  (h₁ : 3 * (6 * x) = -2) :
  x = 17 / 2 :=
by
  sorry

end find_x_l720_72046


namespace scientific_notation_of_population_l720_72002

theorem scientific_notation_of_population (population : Real) (h_pop : population = 6.8e6) :
    ∃ a n, (1 ≤ |a| ∧ |a| < 10) ∧ (population = a * 10^n) ∧ (a = 6.8) ∧ (n = 6) :=
by
  sorry

end scientific_notation_of_population_l720_72002


namespace remaining_budget_l720_72001

def charge_cost : ℝ := 3.5
def num_charges : ℝ := 4
def total_budget : ℝ := 20

theorem remaining_budget : total_budget - (num_charges * charge_cost) = 6 := 
by 
  sorry

end remaining_budget_l720_72001


namespace part_a_part_b_l720_72033

-- Define the predicate ensuring that among any three consecutive symbols, there is at least one zero
def valid_sequence (s : List Char) : Prop :=
  ∀ (i : Nat), i + 2 < s.length → (s.get! i = '0' ∨ s.get! (i + 1) = '0' ∨ s.get! (i + 2) = '0')

-- Count the valid sequences given the number of 'X's and 'O's
noncomputable def count_valid_sequences (n_zeros n_crosses : Nat) : Nat :=
  sorry -- Implementation of the combinatorial counting

-- Part (a): n = 29
theorem part_a : count_valid_sequences 14 29 = 15 := by
  sorry

-- Part (b): n = 28
theorem part_b : count_valid_sequences 14 28 = 120 := by
  sorry

end part_a_part_b_l720_72033


namespace average_age_of_5_people_l720_72072

theorem average_age_of_5_people (avg_age_18 : ℕ) (avg_age_9 : ℕ) (age_15th : ℕ) (total_persons: ℕ) (persons_9: ℕ) (remaining_persons: ℕ) : 
  avg_age_18 = 15 ∧ 
  avg_age_9 = 16 ∧ 
  age_15th = 56 ∧ 
  total_persons = 18 ∧ 
  persons_9 = 9 ∧ 
  remaining_persons = 5 → 
  (avg_age_18 * total_persons - avg_age_9 * persons_9 - age_15th) / remaining_persons = 14 := 
sorry

end average_age_of_5_people_l720_72072


namespace mark_gig_schedule_l720_72083

theorem mark_gig_schedule 
  (every_other_day : ∀ weeks, ∃ gigs, gigs = weeks * 7 / 2) 
  (songs_per_gig : 2 * 5 + 10 = 20) 
  (total_minutes : ∃ gigs, 280 = gigs * 20) : 
  ∃ weeks, weeks = 4 := 
by 
  sorry

end mark_gig_schedule_l720_72083


namespace equivalent_function_l720_72021

theorem equivalent_function :
  (∀ x : ℝ, (76 * x ^ 6) ^ 7 = |x|) :=
by
  sorry

end equivalent_function_l720_72021


namespace smallest_positive_debt_l720_72071

noncomputable def pigs_value : ℤ := 300
noncomputable def goats_value : ℤ := 210

theorem smallest_positive_debt : ∃ D p g : ℤ, (D = pigs_value * p + goats_value * g) ∧ D > 0 ∧ ∀ D' p' g' : ℤ, (D' = pigs_value * p' + goats_value * g' ∧ D' > 0) → D ≤ D' :=
by
  sorry

end smallest_positive_debt_l720_72071


namespace nth_term_sequence_l720_72070

def sequence (n : ℕ) : ℕ :=
  if n = 0 then 1
  else (2 ^ n) - 1

theorem nth_term_sequence (n : ℕ) : 
  sequence n = 2 ^ n - 1 :=
by
  sorry

end nth_term_sequence_l720_72070


namespace part1_x1_part1_x0_part1_xneg2_general_inequality_l720_72035

-- Prove inequality for specific values of x
theorem part1_x1 : - (1/2 : ℝ) * (1: ℝ)^2 + 2 * (1: ℝ) < -(1: ℝ) + 5 := by
  sorry

theorem part1_x0 : - (1/2 : ℝ) * (0: ℝ)^2 + 2 * (0: ℝ) < -(0: ℝ) + 5 := by
  sorry

theorem part1_xneg2 : - (1/2 : ℝ) * (-2: ℝ)^2 + 2 * (-2: ℝ) < -(-2: ℝ) + 5 := by
  sorry

-- Prove general inequality for all real x
theorem general_inequality (x : ℝ) : - (1/2 : ℝ) * x^2 + 2 * x < -x + 5 := by
  sorry

end part1_x1_part1_x0_part1_xneg2_general_inequality_l720_72035


namespace intersection_of_medians_x_coord_l720_72016

def parabola (x : ℝ) : ℝ := x^2 - 4 * x - 1

theorem intersection_of_medians_x_coord (x_a x_b : ℝ) (y : ℝ) :
  (parabola x_a = y) ∧ (parabola x_b = y) ∧ (parabola 5 = parabola 5) → 
  (2 : ℝ) < ((5 + 4) / 3) :=
sorry

end intersection_of_medians_x_coord_l720_72016


namespace num_winners_is_4_l720_72024

variables (A B C D : Prop)

-- Conditions
axiom h1 : A → B
axiom h2 : B → (C ∨ ¬ A)
axiom h3 : ¬ D → (A ∧ ¬ C)
axiom h4 : D → A

-- Assumptions
axiom hA : A
axiom hD : D

-- Statement to prove
theorem num_winners_is_4 : A ∧ B ∧ C ∧ D :=
by {
  sorry
}

end num_winners_is_4_l720_72024


namespace number_of_sophomores_l720_72081

-- Definition of the conditions
variables (J S P j s p : ℕ)

-- Condition: Equal number of students in debate team
def DebateTeam_Equal : Prop := j = s ∧ s = p

-- Condition: Total number of students
def TotalStudents : Prop := J + S + P = 45

-- Condition: Percentage relationships
def PercentRelations_J : Prop := j = J / 5
def PercentRelations_S : Prop := s = 3 * S / 20
def PercentRelations_P : Prop := p = P / 10

-- The main theorem to prove
theorem number_of_sophomores : DebateTeam_Equal j s p 
                               → TotalStudents J S P 
                               → PercentRelations_J J j 
                               → PercentRelations_S S s 
                               → PercentRelations_P P p 
                               → P = 21 :=
by 
  sorry

end number_of_sophomores_l720_72081


namespace john_extra_hours_l720_72011

theorem john_extra_hours (daily_earnings : ℕ) (hours_worked : ℕ) (bonus : ℕ) (hourly_wage : ℕ) (total_earnings_with_bonus : ℕ) (total_hours_with_bonus : ℕ) : 
  daily_earnings = 80 ∧ 
  hours_worked = 8 ∧ 
  bonus = 20 ∧ 
  hourly_wage = 10 ∧ 
  total_earnings_with_bonus = daily_earnings + bonus ∧
  total_hours_with_bonus = total_earnings_with_bonus / hourly_wage → 
  total_hours_with_bonus - hours_worked = 2 := 
by 
  sorry

end john_extra_hours_l720_72011


namespace cristina_running_pace_4point2_l720_72098

theorem cristina_running_pace_4point2 :
  ∀ (nicky_pace head_start time_after_start cristina_pace : ℝ),
    nicky_pace = 3 →
    head_start = 12 →
    time_after_start = 30 →
    cristina_pace = 4.2 →
    (time_after_start = head_start + 30 →
    cristina_pace * time_after_start = nicky_pace * (head_start + 30)) :=
by
  sorry

end cristina_running_pace_4point2_l720_72098


namespace simplified_expression_at_one_l720_72090

noncomputable def original_expression (a : ℚ) : ℚ :=
  (2 * a + 2) / a / (4 / (a ^ 2)) - a / (a + 1)

theorem simplified_expression_at_one : original_expression 1 = 1 / 2 := by
  sorry

end simplified_expression_at_one_l720_72090


namespace total_wait_days_l720_72069

-- Definitions based on the conditions
def days_first_appointment := 4
def days_second_appointment := 20
def days_vaccine_effective := 2 * 7  -- 2 weeks converted to days

-- Theorem stating the total wait time
theorem total_wait_days : days_first_appointment + days_second_appointment + days_vaccine_effective = 38 := by
  sorry

end total_wait_days_l720_72069


namespace negation_example_l720_72028

theorem negation_example (p : ∀ n : ℕ, n^2 < 2^n) : 
  ¬ (∀ n : ℕ, n^2 < 2^n) ↔ ∃ n : ℕ, n^2 ≥ 2^n :=
by sorry

end negation_example_l720_72028


namespace difference_of_two_numbers_l720_72037

theorem difference_of_two_numbers 
(x y : ℝ) 
(h1 : x + y = 20) 
(h2 : x^2 - y^2 = 160) : 
  x - y = 8 := 
by 
  sorry

end difference_of_two_numbers_l720_72037


namespace eval_ff_ff_3_l720_72010

def f (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else 3 * x + 1

theorem eval_ff_ff_3 : f (f (f (f 3))) = 8 :=
  sorry

end eval_ff_ff_3_l720_72010


namespace a_can_complete_in_6_days_l720_72048

noncomputable def rate_b : ℚ := 1/8
noncomputable def rate_c : ℚ := 1/12
noncomputable def earnings_total : ℚ := 2340
noncomputable def earnings_b : ℚ := 780.0000000000001

theorem a_can_complete_in_6_days :
  ∃ (rate_a : ℚ), 
    (1 / rate_a) = 6 ∧
    rate_a + rate_b + rate_c = 3 * rate_b ∧
    earnings_b = (rate_b / (rate_a + rate_b + rate_c)) * earnings_total := sorry

end a_can_complete_in_6_days_l720_72048


namespace small_cubes_with_two_faces_painted_l720_72073

-- Statement of the problem
theorem small_cubes_with_two_faces_painted
  (remaining_cubes : ℕ)
  (edges_with_two_painted_faces : ℕ)
  (number_of_edges : ℕ) :
  remaining_cubes = 60 → edges_with_two_painted_faces = 2 → number_of_edges = 12 →
  (remaining_cubes - (4 * (edges_with_two_painted_faces - 1) * (number_of_edges))) = 28 :=
by
  sorry

end small_cubes_with_two_faces_painted_l720_72073


namespace solve_equation_l720_72061

theorem solve_equation (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 1) :
  (3 / (x - 2) = 2 / (x - 1)) ↔ (x = -1) :=
sorry

end solve_equation_l720_72061


namespace annies_classmates_count_l720_72008

theorem annies_classmates_count (spent : ℝ) (cost_per_candy : ℝ) (candies_left : ℕ) (candies_per_classmate : ℕ) (expected_classmates : ℕ):
  spent = 8 ∧ cost_per_candy = 0.1 ∧ candies_left = 12 ∧ candies_per_classmate = 2 ∧ expected_classmates = 34 →
  (spent / cost_per_candy) - candies_left = (expected_classmates * candies_per_classmate) := 
by
  intros h
  sorry

end annies_classmates_count_l720_72008


namespace divisor_value_l720_72014

theorem divisor_value :
  ∃ D : ℕ, 
    (242 % D = 11) ∧
    (698 % D = 18) ∧
    (365 % D = 15) ∧
    (527 % D = 13) ∧
    ((242 + 698 + 365 + 527) % D = 9) ∧
    (D = 48) :=
sorry

end divisor_value_l720_72014


namespace present_age_of_B_l720_72082

theorem present_age_of_B (A B : ℕ) (h1 : A + 20 = 2 * (B - 20)) (h2 : A = B + 10) : B = 70 :=
by
  sorry

end present_age_of_B_l720_72082


namespace obtuse_triangle_has_two_acute_angles_l720_72084

-- Definition of an obtuse triangle
def is_obtuse_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180 ∧ (A > 90 ∨ B > 90 ∨ C > 90)

-- A theorem to prove that an obtuse triangle has exactly 2 acute angles 
theorem obtuse_triangle_has_two_acute_angles (A B C : ℝ) (h : is_obtuse_triangle A B C) : 
  (A > 0 ∧ A < 90 → B > 0 ∧ B < 90 → C > 0 ∧ C < 90) ∧
  (A > 0 ∧ A < 90 ∧ B > 0 ∧ B < 90) ∨
  (A > 0 ∧ A < 90 ∧ C > 0 ∧ C < 90) ∨
  (B > 0 ∧ B < 90 ∧ C > 0 ∧ C < 90) :=
sorry

end obtuse_triangle_has_two_acute_angles_l720_72084


namespace inequality_holds_l720_72074

theorem inequality_holds (m : ℝ) (h : 0 ≤ m ∧ m < 12) :
  ∀ x : ℝ, 3 * m * x ^ 2 + m * x + 1 > 0 :=
sorry

end inequality_holds_l720_72074


namespace total_cost_of_two_rackets_l720_72056

axiom racket_full_price : ℕ
axiom price_of_first_racket : racket_full_price = 60
axiom price_of_second_racket : racket_full_price / 2 = 30

theorem total_cost_of_two_rackets : 60 + 30 = 90 :=
sorry

end total_cost_of_two_rackets_l720_72056


namespace find_z_l720_72093

theorem find_z (y z : ℝ) (k : ℝ) 
  (h1 : y = 3) (h2 : z = 16) (h3 : y ^ 2 * (z ^ (1 / 4)) = k)
  (h4 : k = 18) (h5 : y = 6) : z = 1 / 16 := by
  sorry

end find_z_l720_72093


namespace solve_for_x_l720_72063

theorem solve_for_x (x : ℚ) (h : 5 * x + 9 * x = 420 - 10 * (x - 4)) : 
  x = 115 / 6 :=
by
  sorry

end solve_for_x_l720_72063


namespace range_of_a_l720_72089

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, (-2 < x - 1 ∧ x - 1 < 3) ∧ (x - a > 0)) ↔ (a ≤ -1) :=
sorry

end range_of_a_l720_72089
