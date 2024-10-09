import Mathlib

namespace train_speed_conversion_l1500_150030

/-- Define a function to convert kmph to m/s --/
def kmph_to_ms (speed_kmph : ℕ) : ℕ :=
  (speed_kmph * 1000) / 3600

/-- Theorem stating that 72 kmph is equivalent to 20 m/s --/
theorem train_speed_conversion : kmph_to_ms 72 = 20 :=
by
  sorry

end train_speed_conversion_l1500_150030


namespace lemonade_percentage_correct_l1500_150002
noncomputable def lemonade_percentage (first_lemonade first_carbon second_carbon mixture_carbon first_portion : ℝ) : ℝ :=
  100 - second_carbon

theorem lemonade_percentage_correct :
  let first_lemonade := 20
  let first_carbon := 80
  let second_carbon := 55
  let mixture_carbon := 60
  let first_portion := 19.99999999999997
  lemonade_percentage first_lemonade first_carbon second_carbon mixture_carbon first_portion = 45 :=
by
  -- Proof to be completed.
  sorry

end lemonade_percentage_correct_l1500_150002


namespace Kato_finishes_first_l1500_150038

-- Define constants and variables from the problem conditions
def Kato_total_pages : ℕ := 10
def Kato_lines_per_page : ℕ := 20
def Gizi_lines_per_page : ℕ := 30
def conversion_ratio : ℚ := 3 / 4
def initial_pages_written_by_Kato : ℕ := 4
def initial_additional_lines_by_Kato : ℚ := 2.5
def Kato_to_Gizi_writing_ratio : ℚ := 3 / 4

-- Calculate total lines in Kato's manuscript
def Kato_total_lines : ℕ := Kato_total_pages * Kato_lines_per_page

-- Convert Kato's lines to Gizi's format
def Kato_lines_in_Gizi_format : ℚ := Kato_total_lines * conversion_ratio

-- Calculate total pages Gizi needs to type
def Gizi_total_pages : ℚ := Kato_lines_in_Gizi_format / Gizi_lines_per_page

-- Calculate initial lines by Kato before Gizi starts typing
def initial_lines_by_Kato : ℚ := initial_pages_written_by_Kato * Kato_lines_per_page + initial_additional_lines_by_Kato

-- Lines Kato writes for every page Gizi types including setup time consideration
def additional_lines_by_Kato_per_Gizi_page : ℚ := Gizi_lines_per_page * Kato_to_Gizi_writing_ratio + initial_additional_lines_by_Kato / Gizi_total_pages

-- Calculate total lines Kato writes while Gizi finishes 5 pages
def final_lines_by_Kato : ℚ := additional_lines_by_Kato_per_Gizi_page * Gizi_total_pages

-- Remaining lines after initial setup for Kato
def remaining_lines_by_Kato_after_initial : ℚ := Kato_total_lines - initial_lines_by_Kato

-- Final proof statement
theorem Kato_finishes_first : final_lines_by_Kato ≥ remaining_lines_by_Kato_after_initial :=
by sorry

end Kato_finishes_first_l1500_150038


namespace none_of_these_l1500_150081

theorem none_of_these (s x y : ℝ) (hs : s > 1) (hx2y_ne_zero : x^2 * y ≠ 0) (hineq : x * s^2 > y * s^2) :
  ¬ (-x > -y) ∧ ¬ (-x > y) ∧ ¬ (1 > -y / x) ∧ ¬ (1 < y / x) :=
by
  sorry

end none_of_these_l1500_150081


namespace probability_slope_le_one_l1500_150096

noncomputable def point := (ℝ × ℝ)

def Q_in_unit_square (Q : point) : Prop :=
  0 ≤ Q.1 ∧ Q.1 ≤ 1 ∧ 0 ≤ Q.2 ∧ Q.2 ≤ 1

def slope_le_one (Q : point) : Prop :=
  (Q.2 - (1/4)) / (Q.1 - (3/4)) ≤ 1

theorem probability_slope_le_one :
  ∃ p q : ℕ, Q_in_unit_square Q → slope_le_one Q →
  p.gcd q = 1 ∧ (p + q = 11) :=
sorry

end probability_slope_le_one_l1500_150096


namespace minimum_value_of_expression_l1500_150090

variable (a b c d : ℝ)

-- The given conditions:
def cond1 : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0
def cond2 : Prop := a^2 + b^2 = 4
def cond3 : Prop := c * d = 1

-- The minimum value:
def expression_value : ℝ := (a^2 * c^2 + b^2 * d^2) * (b^2 * c^2 + a^2 * d^2)

theorem minimum_value_of_expression :
  cond1 a b c d → cond2 a b → cond3 c d → expression_value a b c d ≥ 16 :=
by
  sorry

end minimum_value_of_expression_l1500_150090


namespace find_k_l1500_150047

def vector_a : ℝ × ℝ := (2, 1)
def vector_b (k : ℝ) : ℝ × ℝ := (k, 3)

def vec_add_2b (k : ℝ) : ℝ × ℝ := (2 + 2 * k, 7)
def vec_sub_b (k : ℝ) : ℝ × ℝ := (4 - k, -1)

def vectors_not_parallel (k : ℝ) : Prop :=
  (vec_add_2b k).fst * (vec_sub_b k).snd ≠ (vec_add_2b k).snd * (vec_sub_b k).fst

theorem find_k (k : ℝ) (h : vectors_not_parallel k) : k ≠ 6 :=
by
  sorry

end find_k_l1500_150047


namespace C_is_necessary_but_not_sufficient_for_A_l1500_150042

-- Define C, B, A to be logical propositions
variables (A B C : Prop)

-- The conditions given
axiom h1 : A → B
axiom h2 : ¬ (B → A)
axiom h3 : B ↔ C

-- The conclusion: Prove that C is a necessary but not sufficient condition for A
theorem C_is_necessary_but_not_sufficient_for_A : (A → C) ∧ ¬ (C → A) :=
by
  sorry

end C_is_necessary_but_not_sufficient_for_A_l1500_150042


namespace markup_rate_l1500_150014

theorem markup_rate (S : ℝ) (C : ℝ) (hS : S = 8) (h1 : 0.20 * S = 0.10 * S + (S - C)) :
  ((S - C) / C) * 100 = 42.857 :=
by
  -- Assume given conditions and reasoning to conclude the proof
  sorry

end markup_rate_l1500_150014


namespace bales_stacked_correct_l1500_150082

-- Given conditions
def initial_bales : ℕ := 28
def final_bales : ℕ := 82

-- Define the stacking function
def bales_stacked (initial final : ℕ) : ℕ := final - initial

-- Theorem statement we need to prove
theorem bales_stacked_correct : bales_stacked initial_bales final_bales = 54 := by
  sorry

end bales_stacked_correct_l1500_150082


namespace total_length_of_ropes_l1500_150070

theorem total_length_of_ropes (L : ℝ) 
  (h1 : (L - 12 = 4 * (L - 42))) : 
  2 * L = 104 := 
by
  sorry

end total_length_of_ropes_l1500_150070


namespace odd_function_increasing_ln_x_condition_l1500_150073

theorem odd_function_increasing_ln_x_condition 
  {f : ℝ → ℝ} 
  (h_odd : ∀ x, f (-x) = -f x) 
  (h_increasing : ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y) 
  {x : ℝ} 
  (h_f_ln_x : f (Real.log x) < 0) : 
  0 < x ∧ x < 1 := 
sorry

end odd_function_increasing_ln_x_condition_l1500_150073


namespace find_b_l1500_150086

theorem find_b (b : ℝ) : (∃ c : ℝ, (16 : ℝ) * x^2 + 40 * x + b = (4 * x + c)^2) → b = 25 :=
by
  sorry

end find_b_l1500_150086


namespace same_gender_probability_l1500_150046

-- Define the total number of teachers in School A and their gender distribution.
def schoolA_teachers : Nat := 3
def schoolA_males : Nat := 2
def schoolA_females : Nat := 1

-- Define the total number of teachers in School B and their gender distribution.
def schoolB_teachers : Nat := 3
def schoolB_males : Nat := 1
def schoolB_females : Nat := 2

-- Calculate the probability of selecting two teachers of the same gender.
theorem same_gender_probability :
  (schoolA_males * schoolB_males + schoolA_females * schoolB_females) / (schoolA_teachers * schoolB_teachers) = 4 / 9 :=
by
  sorry

end same_gender_probability_l1500_150046


namespace mod_equiv_inverse_sum_l1500_150025

theorem mod_equiv_inverse_sum :
  (3^15 + 3^14 + 3^13 + 3^12) % 17 = 5 :=
by sorry

end mod_equiv_inverse_sum_l1500_150025


namespace second_quadrant_necessary_not_sufficient_l1500_150066

open Classical

-- Definitions
def isSecondQuadrant (α : ℝ) : Prop := 90 < α ∧ α < 180
def isObtuseAngle (α : ℝ) : Prop := 90 < α ∧ α < 180 ∨ 180 < α ∧ α < 270

-- The theorem statement
theorem second_quadrant_necessary_not_sufficient (α : ℝ) :
  (isSecondQuadrant α → isObtuseAngle α) ∧ ¬(isSecondQuadrant α ↔ isObtuseAngle α) :=
by
  sorry

end second_quadrant_necessary_not_sufficient_l1500_150066


namespace number_of_salads_bought_l1500_150012

variable (hot_dogs_cost : ℝ := 5 * 1.50)
variable (initial_money : ℝ := 2 * 10)
variable (change_given_back : ℝ := 5)
variable (total_spent : ℝ := initial_money - change_given_back)
variable (salad_cost : ℝ := 2.50)

theorem number_of_salads_bought : (total_spent - hot_dogs_cost) / salad_cost = 3 := 
by 
  sorry

end number_of_salads_bought_l1500_150012


namespace pentagon_area_l1500_150023

open Real

/-- The area of a pentagon with sides 18, 25, 30, 28, and 25 units is 950 square units -/
theorem pentagon_area (a b c d e : ℝ) (h₁ : a = 18) (h₂ : b = 25) (h₃ : c = 30) (h₄ : d = 28) (h₅ : e = 25) : 
  ∃ (area : ℝ), area = 950 :=
by {
  sorry
}

end pentagon_area_l1500_150023


namespace average_percentage_decrease_l1500_150048

theorem average_percentage_decrease : 
  ∃ x : ℝ, (0 < x) ∧ (x < 1) ∧ ((2000 * (1 - x)^2 = 1280) ↔ (x = 0.18)) :=
by
  sorry

end average_percentage_decrease_l1500_150048


namespace translate_upwards_l1500_150085

theorem translate_upwards (x : ℝ) : (2 * x^2) + 2 = 2 * x^2 + 2 := by
  sorry

end translate_upwards_l1500_150085


namespace smallest_sum_of_50_consecutive_integers_with_product_zero_and_positive_sum_l1500_150080

theorem smallest_sum_of_50_consecutive_integers_with_product_zero_and_positive_sum :
  ∃ (a : ℤ), (∃ (l : List ℤ), l.length = 50 ∧ List.prod l = 0 ∧ 0 < List.sum l ∧ List.sum l = 25) :=
by
  sorry

end smallest_sum_of_50_consecutive_integers_with_product_zero_and_positive_sum_l1500_150080


namespace triangle_area_problem_l1500_150005

theorem triangle_area_problem (c d : ℝ) (hc : c > 0) (hd : d > 0) 
  (h_area : (∃ t : ℝ, t > 0 ∧ (2 * c * t + 3 * d * (12 / (2 * c)) = 12) ∧ (∃ s : ℝ, s > 0 ∧ 2 * c * (12 / (3 * d)) + 3 * d * s = 12)) ∧ 
    ((1 / 2) * (12 / (2 * c)) * (12 / (3 * d)) = 12)) : c * d = 1 := 
by 
  sorry

end triangle_area_problem_l1500_150005


namespace paul_collected_total_cans_l1500_150029

theorem paul_collected_total_cans :
  let saturday_bags := 10
  let sunday_bags := 5
  let saturday_cans_per_bag := 12
  let sunday_cans_per_bag := 15
  let saturday_total_cans := saturday_bags * saturday_cans_per_bag
  let sunday_total_cans := sunday_bags * sunday_cans_per_bag
  let total_cans := saturday_total_cans + sunday_total_cans
  total_cans = 195 := 
by
  sorry

end paul_collected_total_cans_l1500_150029


namespace base_500_in_base_has_six_digits_l1500_150035

theorem base_500_in_base_has_six_digits (b : ℕ) : b^5 ≤ 500 ∧ 500 < b^6 ↔ b = 3 := 
by
  sorry

end base_500_in_base_has_six_digits_l1500_150035


namespace y_not_multiple_of_7_y_not_multiple_of_14_y_not_multiple_of_21_y_not_multiple_of_28_l1500_150020

def y : ℕ := 42 + 98 + 210 + 333 + 175 + 28

theorem y_not_multiple_of_7 : ¬ (7 ∣ y) := sorry
theorem y_not_multiple_of_14 : ¬ (14 ∣ y) := sorry
theorem y_not_multiple_of_21 : ¬ (21 ∣ y) := sorry
theorem y_not_multiple_of_28 : ¬ (28 ∣ y) := sorry

end y_not_multiple_of_7_y_not_multiple_of_14_y_not_multiple_of_21_y_not_multiple_of_28_l1500_150020


namespace divisor_is_seventeen_l1500_150095

theorem divisor_is_seventeen (D x : ℕ) (h1 : D = 7 * x) (h2 : D + x = 136) : x = 17 :=
by
  sorry

end divisor_is_seventeen_l1500_150095


namespace triangle_neg3_4_l1500_150040

def triangle (a b : ℚ) : ℚ := -a + b

theorem triangle_neg3_4 : triangle (-3) 4 = 7 := 
by 
  sorry

end triangle_neg3_4_l1500_150040


namespace total_bronze_needed_l1500_150015

theorem total_bronze_needed (w1 w2 w3 : ℕ) (h1 : w1 = 50) (h2 : w2 = 2 * w1) (h3 : w3 = 4 * w2) : w1 + w2 + w3 = 550 :=
by
  -- We'll complete the proof later
  sorry

end total_bronze_needed_l1500_150015


namespace book_distribution_ways_l1500_150062

theorem book_distribution_ways : 
  ∃ n : ℕ, n = 7 ∧ ∀ k : ℕ, 1 ≤ k ∧ k ≤ 7 →
  ∃ l : ℕ, l + (8 - l) = 8 ∧ 1 ≤ l ∧ 1 ≤ 8 - l :=
by
  -- We will provide a proof here.
  sorry

end book_distribution_ways_l1500_150062


namespace repeating_decimals_sum_l1500_150097

def repeating_decimal1 : ℚ := (2 : ℚ) / 9  -- 0.\overline{2}
def repeating_decimal2 : ℚ := (3 : ℚ) / 99 -- 0.\overline{03}

theorem repeating_decimals_sum : repeating_decimal1 + repeating_decimal2 = (25 : ℚ) / 99 :=
by
  sorry

end repeating_decimals_sum_l1500_150097


namespace delores_remaining_money_l1500_150057

variable (delores_money : ℕ := 450)
variable (computer_price : ℕ := 1000)
variable (computer_discount : ℝ := 0.30)
variable (printer_price : ℕ := 100)
variable (printer_tax_rate : ℝ := 0.15)
variable (table_price_euros : ℕ := 200)
variable (exchange_rate : ℝ := 1.2)

def computer_sale_price : ℝ := computer_price * (1 - computer_discount)
def printer_total_cost : ℝ := printer_price * (1 + printer_tax_rate)
def table_cost_dollars : ℝ := table_price_euros * exchange_rate
def total_cost : ℝ := computer_sale_price + printer_total_cost + table_cost_dollars
def remaining_money : ℝ := delores_money - total_cost

theorem delores_remaining_money : remaining_money = -605 := by
  sorry

end delores_remaining_money_l1500_150057


namespace trig_sum_identity_l1500_150094

theorem trig_sum_identity :
  Real.sin (20 * Real.pi / 180) + Real.sin (40 * Real.pi / 180) +
  Real.sin (60 * Real.pi / 180) - Real.sin (80 * Real.pi / 180) = Real.sqrt 3 / 2 := 
sorry

end trig_sum_identity_l1500_150094


namespace balloons_problem_l1500_150083

theorem balloons_problem :
  ∃ (b y : ℕ), y = 3414 ∧ b + y = 8590 ∧ b - y = 1762 := 
by
  sorry

end balloons_problem_l1500_150083


namespace smallest_b_for_quadratic_factorization_l1500_150028

theorem smallest_b_for_quadratic_factorization : ∃ (b : ℕ), 
  (∀ r s : ℤ, (r * s = 4032) ∧ (r + s = b) → b ≥ 127) ∧ 
  (∃ r s : ℤ, (r * s = 4032) ∧ (r + s = b) ∧ (b = 127))
:= sorry

end smallest_b_for_quadratic_factorization_l1500_150028


namespace triangle_inequality_l1500_150007

variables {a b c h : ℝ}
variable {n : ℕ}

theorem triangle_inequality
  (h_triangle : a^2 + b^2 = c^2)
  (h_height : a * b = c * h)
  (h_cond : a + b < c + h)
  (h_pos_n : n > 0) :
  a^n + b^n < c^n + h^n :=
sorry

end triangle_inequality_l1500_150007


namespace trajectory_of_P_l1500_150063

-- Definitions for points and distance
structure Point where
  x : ℝ
  y : ℝ

noncomputable def dist (P Q : Point) : ℝ :=
  Real.sqrt ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2)

-- Fixed points F1 and F2
variable (F1 F2 : Point)
-- Distance condition
axiom dist_F1F2 : dist F1 F2 = 8

-- Moving point P satisfying the condition
variable (P : Point)
axiom dist_PF1_PF2 : dist P F1 + dist P F2 = 8

-- Proof goal: P lies on the line segment F1F2
theorem trajectory_of_P : ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ P = ⟨(1 - t) * F1.x + t * F2.x, (1 - t) * F1.y + t * F2.y⟩ :=
  sorry

end trajectory_of_P_l1500_150063


namespace part1_part2_l1500_150067

-- Part (I)
theorem part1 (a : ℝ) :
  (∀ x : ℝ, 3 * x - abs (-2 * x + 1) ≥ a ↔ 2 ≤ x) → a = 3 :=
by
  sorry

-- Part (II)
theorem part2 (a : ℝ) :
  (∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → (x - abs (x - a) ≤ 1)) → (a ≤ 1 ∨ 3 ≤ a) :=
by
  sorry

end part1_part2_l1500_150067


namespace number_of_performance_orders_l1500_150061

-- Define the options for the programs
def programs : List String := ["A", "B", "C", "D", "E", "F", "G", "H"]

-- Define a function to count valid performance orders under given conditions
def countPerformanceOrders (progs : List String) : ℕ :=
  sorry  -- This is where the logic to count performance orders goes

-- The theorem to assert the total number of performance orders
theorem number_of_performance_orders : countPerformanceOrders programs = 2860 :=
by
  sorry  -- Proof of the theorem

end number_of_performance_orders_l1500_150061


namespace sqrt_product_simplification_l1500_150064

variable (p : ℝ)

theorem sqrt_product_simplification (hp : 0 ≤ p) :
  (Real.sqrt (42 * p) * Real.sqrt (7 * p) * Real.sqrt (14 * p)) = 42 * p * Real.sqrt (7 * p) :=
sorry

end sqrt_product_simplification_l1500_150064


namespace distinct_sums_l1500_150041

theorem distinct_sums (n : ℕ) (a : Fin n → ℕ) (h_distinct : Function.Injective a) :
  ∃ S : Finset ℕ, S.card ≥ n * (n + 1) / 2 :=
by
  sorry

end distinct_sums_l1500_150041


namespace simplify_proof_l1500_150068

noncomputable def simplify_expression (x : ℝ) (hx : x ≠ 0) (hx1 : x ≠ 1) (hx_1 : x ≠ -1) : ℝ :=
  (1 - 1/x) / ((1 - x^2) / x)

theorem simplify_proof (x : ℝ) (hx : x ≠ 0) (hx1 : x ≠ 1) (hx_1 : x ≠ -1) : 
  simplify_expression x hx hx1 hx_1 = -1 / (1 + x) := by 
  sorry

end simplify_proof_l1500_150068


namespace odd_positive_int_divides_3pow_n_plus_1_l1500_150054

theorem odd_positive_int_divides_3pow_n_plus_1 (n : ℕ) (hn_odd : n % 2 = 1) (hn_pos : n > 0) : 
  n ∣ (3^n + 1) ↔ n = 1 := 
by
  sorry

end odd_positive_int_divides_3pow_n_plus_1_l1500_150054


namespace percent_relation_l1500_150093

theorem percent_relation (x y z w : ℝ) (h1 : x = 1.25 * y) (h2 : y = 0.40 * z) (h3 : z = 1.10 * w) :
  (x / w) * 100 = 55 := by sorry

end percent_relation_l1500_150093


namespace people_with_fewer_than_seven_cards_l1500_150022

theorem people_with_fewer_than_seven_cards (total_cards : ℕ) (num_people : ℕ) (cards_per_person : ℕ) (extra_cards : ℕ)
  (h1 : total_cards = 52) (h2 : num_people = 8) (h3 : total_cards = num_people * cards_per_person + extra_cards) (h4 : extra_cards < num_people) :
  ∃ fewer_than_seven : ℕ, num_people - extra_cards = fewer_than_seven :=
by
  have remainder := (52 % 8)
  have cards_per_person := (52 / 8)
  have number_fewer_than_seven := num_people - remainder
  existsi number_fewer_than_seven
  sorry

end people_with_fewer_than_seven_cards_l1500_150022


namespace second_less_than_first_third_less_than_first_l1500_150049

variable (X : ℝ)

def first_number : ℝ := 0.70 * X
def second_number : ℝ := 0.63 * X
def third_number : ℝ := 0.59 * X

theorem second_less_than_first : 
  ((first_number X - second_number X) / first_number X * 100) = 10 :=
by
  sorry

theorem third_less_than_first : 
  ((third_number X - first_number X) / first_number X * 100) = -15.71 :=
by
  sorry

end second_less_than_first_third_less_than_first_l1500_150049


namespace line_divides_circle_l1500_150065

theorem line_divides_circle (k m : ℝ) :
  (∀ x y : ℝ, y = x - 1 → x^2 + y^2 + k*x + m*y - 4 = 0 → m - k = 2) :=
sorry

end line_divides_circle_l1500_150065


namespace find_number_l1500_150027

theorem find_number (x : ℝ) (h : 3034 - x / 200.4 = 3029) : x = 1002 :=
sorry

end find_number_l1500_150027


namespace apples_total_l1500_150092

theorem apples_total (Benny_picked Dan_picked : ℕ) (hB : Benny_picked = 2) (hD : Dan_picked = 9) : Benny_picked + Dan_picked = 11 :=
by
  -- Definitions
  sorry

end apples_total_l1500_150092


namespace cos_double_angle_identity_l1500_150037

theorem cos_double_angle_identity (α : ℝ) (h : Real.sin (Real.pi / 6 - α) = 1 / 3) :
  Real.cos (2 * Real.pi / 3 + 2 * α) = -7 / 9 :=
by
  sorry

end cos_double_angle_identity_l1500_150037


namespace gcd_poly_l1500_150024

theorem gcd_poly {b : ℕ} (h : 1116 ∣ b) : Nat.gcd (b^2 + 11 * b + 36) (b + 6) = 6 :=
by
  sorry

end gcd_poly_l1500_150024


namespace arithmetic_sequence_fifth_term_l1500_150019

theorem arithmetic_sequence_fifth_term:
  ∀ (a₁ aₙ : ℕ) (n : ℕ),
    n = 20 → a₁ = 2 → aₙ = 59 →
    ∃ d a₅, d = (59 - 2) / (20 - 1) ∧ a₅ = 2 + (5 - 1) * d ∧ a₅ = 14 :=
by
  sorry

end arithmetic_sequence_fifth_term_l1500_150019


namespace solve_for_x_l1500_150050

theorem solve_for_x :
  ∀ x : ℝ, 4 * x + 9 * x = 360 - 9 * (x - 4) → x = 18 :=
by
  intros x h
  sorry

end solve_for_x_l1500_150050


namespace mark_total_flowers_l1500_150087

theorem mark_total_flowers (yellow purple green total : ℕ) 
  (hyellow : yellow = 10)
  (hpurple : purple = yellow + (yellow * 80 / 100))
  (hgreen : green = (yellow + purple) * 25 / 100)
  (htotal : total = yellow + purple + green) : 
  total = 35 :=
by
  sorry

end mark_total_flowers_l1500_150087


namespace apples_to_cucumbers_l1500_150084

theorem apples_to_cucumbers (a b c : ℕ) 
    (h₁ : 10 * a = 5 * b) 
    (h₂ : 3 * b = 4 * c) : 
    (24 * a) = 16 * c := 
by
  sorry

end apples_to_cucumbers_l1500_150084


namespace cost_price_of_radio_l1500_150072

-- Definitions for conditions
def selling_price := 1245
def loss_percentage := 17

-- Prove that the cost price is Rs. 1500 given the conditions
theorem cost_price_of_radio : 
  ∃ C, (C - 1245) * 100 / C = 17 ∧ C = 1500 := 
sorry

end cost_price_of_radio_l1500_150072


namespace cost_of_shoes_l1500_150071

   theorem cost_of_shoes (initial_budget remaining_budget : ℝ) (H_initial : initial_budget = 999) (H_remaining : remaining_budget = 834) : 
   initial_budget - remaining_budget = 165 := by
     sorry
   
end cost_of_shoes_l1500_150071


namespace no_pos_integers_exist_l1500_150075

theorem no_pos_integers_exist (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  ¬ (3 * (a * b + b * c + c * a) ∣ a^2 + b^2 + c^2) :=
sorry

end no_pos_integers_exist_l1500_150075


namespace tan_five_pi_over_four_l1500_150076

-- Define the question to prove
theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 :=
by
  sorry

end tan_five_pi_over_four_l1500_150076


namespace find_x_l1500_150088

theorem find_x (x : ℝ) : (x / 18) * (36 / 72) = 1 → x = 36 :=
by
  intro h
  sorry

end find_x_l1500_150088


namespace find_a_l1500_150010

theorem find_a (x a : ℝ) (A B : ℝ × ℝ) (C : ℝ × ℝ) :
  A = (7, 1) ∧ B = (1, 4) ∧ C = (x, a * x) ∧ 
  (x - 7, a * x - 1) = (2 * (1 - x), 2 * (4 - a * x)) → 
  a = 1 :=
sorry

end find_a_l1500_150010


namespace unique_real_solution_bound_l1500_150091

theorem unique_real_solution_bound (b : ℝ) :
  (∀ x : ℝ, x^3 - b*x^2 - 4*b*x + b^2 - 4 = 0 → ∃! y : ℝ, y = x) → b < 1 :=
by
  sorry

end unique_real_solution_bound_l1500_150091


namespace gcd_lcm_product_l1500_150079

theorem gcd_lcm_product (a b : ℕ) (ha : a = 24) (hb : b = 36) : 
  Nat.gcd a b * Nat.lcm a b = 864 :=
by
  rw [ha, hb]
  -- This theorem proves that the product of the GCD and LCM of 24 and 36 equals 864.

  sorry -- Proof will go here

end gcd_lcm_product_l1500_150079


namespace percentage_difference_l1500_150099

theorem percentage_difference : (70 / 100 : ℝ) * 100 - (60 / 100 : ℝ) * 80 = 22 := by
  sorry

end percentage_difference_l1500_150099


namespace Jerry_age_l1500_150045

theorem Jerry_age (M J : ℕ) (h1 : M = 2 * J - 6) (h2 : M = 22) : J = 14 :=
by
  sorry

end Jerry_age_l1500_150045


namespace arithmetic_geometric_sum_l1500_150055

theorem arithmetic_geometric_sum (a : ℕ → ℚ) (S : ℕ → ℚ) (d : ℚ) (n : ℕ) (h1 : a 1 = 2)
  (h2 : a 3 = a 1 + 2 * d) (h3 : a 5 = a 1 + 4 * d) (h4 : (a 3) ^ 2 = a 1 * a 5)
  (h5 : d ≠ 0) : S n = (n^2 + 7 * n) / 4 := sorry

end arithmetic_geometric_sum_l1500_150055


namespace value_of_e_l1500_150077

theorem value_of_e
  (a b c d e : ℤ)
  (h1 : b = a + 2)
  (h2 : c = a + 4)
  (h3 : d = a + 6)
  (h4 : e = a + 8)
  (h5 : a + c = 146) :
  e = 79 :=
  by sorry

end value_of_e_l1500_150077


namespace base9_number_perfect_square_l1500_150004

theorem base9_number_perfect_square (a b d : ℕ) (h1 : a ≠ 0) (h2 : 0 ≤ d ∧ d ≤ 8) (n : ℕ) 
  (h3 : n = 729 * a + 81 * b + 45 + d) (h4 : ∃ k : ℕ, k * k = n) : d = 0 := 
sorry

end base9_number_perfect_square_l1500_150004


namespace matrix_eq_sum_35_l1500_150016

theorem matrix_eq_sum_35 (a b c d : ℤ) (h1 : 2 * a = 14 * a - 15 * b)
  (h2 : 2 * b = 9 * a - 10 * b)
  (h3 : 3 * c = 14 * c - 15 * d)
  (h4 : 3 * d = 9 * c - 10 * d) :
  a + b + c + d = 35 :=
sorry

end matrix_eq_sum_35_l1500_150016


namespace milk_water_ratio_l1500_150011

theorem milk_water_ratio (total_volume : ℕ) (initial_milk_ratio : ℕ) (initial_water_ratio : ℕ) (added_water : ℕ)
  (h₁ : total_volume = 45) (h₂ : initial_milk_ratio = 4) (h₃ : initial_water_ratio = 1) (h₄ : added_water = 9) :
  (36 : ℕ) / (18 : ℕ) = 2 :=
by sorry

end milk_water_ratio_l1500_150011


namespace tan_product_l1500_150098

theorem tan_product :
  (Real.tan (Real.pi / 8)) * (Real.tan (3 * Real.pi / 8)) * (Real.tan (5 * Real.pi / 8)) = 1 :=
sorry

end tan_product_l1500_150098


namespace nail_insertion_l1500_150006

theorem nail_insertion (k : ℝ) (h1 : 0 < k) (h2 : k < 1) : 
  (4/7) + (4/7) * k + (4/7) * k^2 = 1 :=
by sorry

end nail_insertion_l1500_150006


namespace sand_needed_l1500_150051

def area_rectangular_patch : ℕ := 6 * 7
def area_square_patch : ℕ := 5 * 5
def sand_per_square_inch : ℕ := 3

theorem sand_needed : area_rectangular_patch + area_square_patch * sand_per_square_inch = 201 := sorry

end sand_needed_l1500_150051


namespace work_problem_l1500_150039

/--
Given:
1. A and B together can finish the work in 16 days.
2. B alone can finish the work in 48 days.
To Prove:
A alone can finish the work in 24 days.
-/
theorem work_problem (a b : ℕ)
  (h1 : a + b = 16)
  (h2 : b = 48) :
  a = 24 := 
sorry

end work_problem_l1500_150039


namespace number_of_people_in_first_group_l1500_150069

-- Define variables representing the work done by one person in one day (W) and the number of people in the first group (P)
variable (W : ℕ) (P : ℕ)

-- Conditions from the problem
-- Some people can do 3 times a particular work in 3 days
def condition1 : Prop := P * 3 * W = 3 * W

-- It takes 6 people 3 days to do 6 times of that particular work
def condition2 : Prop := 6 * 3 * W = 6 * W

-- The statement to prove
theorem number_of_people_in_first_group 
  (h1 : condition1 W P) 
  (h2 : condition2 W) : P = 3 :=
by
  sorry

end number_of_people_in_first_group_l1500_150069


namespace entry_exit_options_l1500_150003

theorem entry_exit_options :
  let south_gates := 4
  let north_gates := 3
  let total_gates := south_gates + north_gates
  (total_gates * total_gates = 49) :=
by {
  let south_gates := 4
  let north_gates := 3
  let total_gates := south_gates + north_gates
  show total_gates * total_gates = 49
  sorry
}

end entry_exit_options_l1500_150003


namespace average_weight_l1500_150033

theorem average_weight :
  ∀ (A B C : ℝ),
    (A + B = 84) → 
    (B + C = 86) → 
    (B = 35) → 
    (A + B + C) / 3 = 45 :=
by
  intros A B C hab hbc hb
  -- proof omitted
  sorry

end average_weight_l1500_150033


namespace bn_is_arithmetic_seq_an_general_term_l1500_150032

def seq_an (a : ℕ → ℝ) : Prop :=
a 1 = 2 ∧ ∀ n, (a (n + 1) - 1) * (a n - 1) = 3 * (a n - a (n + 1))

def seq_bn (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
∀ n, b n = 1 / (a n - 1)

theorem bn_is_arithmetic_seq (a : ℕ → ℝ) (b : ℕ → ℝ) (h1 : seq_an a) (h2 : seq_bn a b) : 
∀ n, b (n + 1) - b n = 1 / 3 :=
sorry

theorem an_general_term (a : ℕ → ℝ) (b : ℕ → ℝ) (h1 : seq_an a) (h2 : seq_bn a b) : 
∀ n, a n = (n + 5) / (n + 2) :=
sorry

end bn_is_arithmetic_seq_an_general_term_l1500_150032


namespace garden_perimeter_is_64_l1500_150060

-- Define the playground dimensions and its area 
def playground_length := 16
def playground_width := 12
def playground_area := playground_length * playground_width

-- Define the garden width and its area being the same as the playground's area
def garden_width := 8
def garden_area := playground_area

-- Calculate the garden's length
def garden_length := garden_area / garden_width

-- Calculate the perimeter of the garden
def garden_perimeter := 2 * (garden_length + garden_width)

theorem garden_perimeter_is_64 :
  garden_perimeter = 64 := 
sorry

end garden_perimeter_is_64_l1500_150060


namespace solution_set_of_inequalities_l1500_150013

-- Define the conditions of the inequality system
def inequality1 (x : ℝ) : Prop := x - 2 ≥ -5
def inequality2 (x : ℝ) : Prop := 3 * x < x + 2

-- The statement to prove the solution set of the inequalities
theorem solution_set_of_inequalities :
  { x : ℝ | inequality1 x ∧ inequality2 x } = { x : ℝ | -3 ≤ x ∧ x < 1 } :=
  sorry

end solution_set_of_inequalities_l1500_150013


namespace closest_perfect_square_multiple_of_4_l1500_150021

theorem closest_perfect_square_multiple_of_4 (n : ℕ) (h1 : ∃ k : ℕ, k^2 = n) (h2 : n % 4 = 0) : n = 324 := by
  -- Define 350 as the target
  let target := 350

  -- Conditions
  have cond1 : ∃ k : ℕ, k^2 = n := h1
  
  have cond2 : n % 4 = 0 := h2

  -- Check possible values meeting conditions
  by_cases h : n = 324
  { exact h }
  
  -- Exclude non-multiples of 4 and perfect squares further away from 350
  sorry

end closest_perfect_square_multiple_of_4_l1500_150021


namespace find_interest_rate_l1500_150044

-- Define the given conditions
variables (P A t n CI : ℝ) (r : ℝ)

-- Suppose given conditions
variables (hP : P = 1200)
variables (hCI : CI = 240)
variables (hA : A = P + CI)
variables (ht : t = 1)
variables (hn : n = 1)

-- Define the statement to prove 
theorem find_interest_rate : (A = P * (1 + r / n)^(n * t)) → (r = 0.2) :=
by
  sorry

end find_interest_rate_l1500_150044


namespace smallest_N_l1500_150074

-- Definitions for the problem conditions
def is_rectangular_block (a b c : ℕ) (N : ℕ) : Prop :=
  N = a * b * c ∧ 143 = (a - 1) * (b - 1) * (c - 1)

-- Theorem to prove the smallest possible value of N
theorem smallest_N : ∃ a b c : ℕ, is_rectangular_block a b c 336 :=
by
  sorry

end smallest_N_l1500_150074


namespace matrix_equation_l1500_150026

open Matrix

-- Define matrix B
def B : Matrix (Fin 2) (Fin 2) (ℤ) :=
  ![![1, -2], 
    ![-3, 5]]

-- The proof problem statement in Lean 4
theorem matrix_equation (r s : ℤ) (I : Matrix (Fin 2) (Fin 2) (ℤ))  [DecidableEq (ℤ)] [Fintype (Fin 2)] : 
  I = 1 ∧ B ^ 6 = r • B + s • I ↔ r = 2999 ∧ s = 2520 := by {
    sorry
}

end matrix_equation_l1500_150026


namespace fifth_inequality_nth_inequality_solve_given_inequality_l1500_150017

theorem fifth_inequality :
  ∀ x, 1 < x ∧ x < 2 → (x + 2 / x < 3) →
  ∀ x, 3 < x ∧ x < 4 → (x + 12 / x < 7) →
  ∀ x, 5 < x ∧ x < 6 → (x + 30 / x < 11) →
  (x + 90 / x < 19) := by
  sorry

theorem nth_inequality (n : ℕ) :
  ∀ x, (2 * n - 1 < x ∧ x < 2 * n) →
  (x + 2 * n * (2 * n - 1) / x < 4 * n - 1) := by
  sorry

theorem solve_given_inequality (a : ℕ) (x : ℝ) (h_a_pos: 0 < a) :
  x + 12 * a / (x + 1) < 4 * a + 2 →
  (2 < x ∧ x < 4 * a - 1) := by
  sorry

end fifth_inequality_nth_inequality_solve_given_inequality_l1500_150017


namespace area_of_cos_integral_l1500_150008

theorem area_of_cos_integral : 
  (∫ x in (0:ℝ)..(3 * Real.pi / 2), |Real.cos x|) = 3 :=
by
  sorry

end area_of_cos_integral_l1500_150008


namespace min_S_min_S_values_range_of_c_l1500_150053

-- Part 1
theorem min_S (a b c : ℝ) (h : a + b + c = 1) : 
  2 * a^2 + 3 * b^2 + c^2 ≥ (6 / 11) :=
sorry

-- Part 1, finding exact values of a, b, c where minimum is reached
theorem min_S_values (a b c : ℝ) (h : a + b + c = 1) :
  2 * a^2 + 3 * b^2 + c^2 = (6 / 11) ↔ a = (3 / 11) ∧ b = (2 / 11) ∧ c = (6 / 11) :=
sorry
  
-- Part 2
theorem range_of_c (a b c : ℝ) (h1 : 2 * a^2 + 3 * b^2 + c^2 = 1) : 
  (1 / 11) ≤ c ∧ c ≤ 1 :=
sorry

end min_S_min_S_values_range_of_c_l1500_150053


namespace integer_solutions_count_l1500_150089

theorem integer_solutions_count :
  ∃ (s : Finset ℤ), s.card = 6 ∧ ∀ x ∈ s, 4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 6 :=
by
  sorry

end integer_solutions_count_l1500_150089


namespace shorter_leg_of_right_triangle_l1500_150058

theorem shorter_leg_of_right_triangle (a b c : ℕ) (h₁ : a^2 + b^2 = c^2) (h₂ : c = 65) (h₃ : a ≤ b) : a = 25 :=
sorry

end shorter_leg_of_right_triangle_l1500_150058


namespace geometric_sequence_Sn_geometric_sequence_Sn_l1500_150018

noncomputable def Sn (a1 q : ℝ) (n : ℕ) : ℝ :=
  if q = 1/3 then (27/2) - (1/2) * 3^(n - 3)
  else if q = 3 then (3^n - 1) / 2
  else 0

theorem geometric_sequence_Sn (a1 : ℝ) (n : ℕ) (h1 : a1 * (1/3) = 3)
  (h2 : a1 + a1 * (1/3)^2 = 10) : 
  Sn a1 (1/3) n = (27/2) - (1/2) * 3^(n - 3) :=
by
  sorry

theorem geometric_sequence_Sn' (a1 : ℝ) (n : ℕ) (h1 : a1 * 3 = 3) 
  (h2 : a1 + a1 * 3^2 = 10) : 
  Sn a1 3 n = (3^n - 1) / 2 :=
by
  sorry

end geometric_sequence_Sn_geometric_sequence_Sn_l1500_150018


namespace total_pixels_correct_l1500_150056

-- Define the monitor's dimensions and pixel density as given conditions
def width_inches : ℕ := 21
def height_inches : ℕ := 12
def pixels_per_inch : ℕ := 100

-- Define the width and height in pixels based on the given conditions
def width_pixels : ℕ := width_inches * pixels_per_inch
def height_pixels : ℕ := height_inches * pixels_per_inch

-- State the objective: proving the total number of pixels on the monitor
theorem total_pixels_correct : width_pixels * height_pixels = 2520000 := by
  sorry

end total_pixels_correct_l1500_150056


namespace gain_percent_is_33_33_l1500_150059
noncomputable def gain_percent_calculation (C S : ℝ) := ((S - C) / C) * 100

theorem gain_percent_is_33_33
  (C S : ℝ)
  (h : 75 * C = 56.25 * S) :
  gain_percent_calculation C S = 33.33 := by
  sorry

end gain_percent_is_33_33_l1500_150059


namespace edward_made_in_summer_l1500_150000

def edward_made_in_spring := 2
def cost_of_supplies := 5
def money_left_over := 24

theorem edward_made_in_summer : edward_made_in_spring + x - cost_of_supplies = money_left_over → x = 27 :=
by
  intros h
  sorry

end edward_made_in_summer_l1500_150000


namespace find_n_for_positive_root_l1500_150036

theorem find_n_for_positive_root :
  ∃ x : ℝ, x > 0 ∧ (∃ n : ℝ, (n / (x - 1) + 2 / (1 - x) = 1)) ↔ n = 2 :=
by
  sorry

end find_n_for_positive_root_l1500_150036


namespace find_sum_l1500_150031

noncomputable def principal_sum (P R : ℝ) := 
  let I := (P * R * 10) / 100
  let new_I := (P * (R + 5) * 10) / 100
  I + 600 = new_I

theorem find_sum (P R : ℝ) (h : principal_sum P R) : P = 1200 := 
  sorry

end find_sum_l1500_150031


namespace largest_multiple_of_15_less_than_neg_150_l1500_150009

theorem largest_multiple_of_15_less_than_neg_150 : ∃ m : ℤ, m % 15 = 0 ∧ m < -150 ∧ (∀ n : ℤ, n % 15 = 0 ∧ n < -150 → n ≤ m) ∧ m = -165 := sorry

end largest_multiple_of_15_less_than_neg_150_l1500_150009


namespace average_increase_by_3_l1500_150078

def initial_average_before_inning_17 (A : ℝ) : Prop :=
  16 * A + 85 = 17 * 37

theorem average_increase_by_3 (A : ℝ) (h : initial_average_before_inning_17 A) :
  37 - A = 3 :=
by
  sorry

end average_increase_by_3_l1500_150078


namespace time_to_cross_bridge_l1500_150001

noncomputable def train_crossing_time
  (length_train : ℕ)
  (length_bridge : ℕ)
  (speed_kmph : ℕ)
  (conversion_factor : ℚ) : ℚ :=
  (length_train + length_bridge) / (speed_kmph * conversion_factor)

theorem time_to_cross_bridge :
  train_crossing_time 135 240 45 (5 / 18) = 30 := by
  sorry

end time_to_cross_bridge_l1500_150001


namespace number_of_dots_in_120_circles_l1500_150052

theorem number_of_dots_in_120_circles :
  ∃ n : ℕ, (n = 14) ∧ (∀ m : ℕ, m * (m + 1) / 2 + m ≤ 120 → m ≤ n) :=
by
  sorry

end number_of_dots_in_120_circles_l1500_150052


namespace find_other_integer_l1500_150034

theorem find_other_integer (x y : ℕ) (h1 : 1 ≤ x ∧ x ≤ 9) (h2 : 1 ≤ y ∧ y ≤ 9) (h3 : 7 * x + y = 68) : y = 5 :=
by
  sorry

end find_other_integer_l1500_150034


namespace total_distance_covered_l1500_150043

theorem total_distance_covered (up_speed down_speed up_time down_time : ℕ) (H1 : up_speed = 30) (H2 : down_speed = 50) (H3 : up_time = 5) (H4 : down_time = 5) :
  (up_speed * up_time + down_speed * down_time) = 400 := 
by
  sorry

end total_distance_covered_l1500_150043
