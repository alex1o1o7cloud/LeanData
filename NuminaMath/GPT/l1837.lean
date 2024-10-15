import Mathlib

namespace NUMINAMATH_GPT_probability_red_or_white_is_7_over_10_l1837_183726

/-
A bag consists of 20 marbles, of which 6 are blue, 9 are red, and the remainder are white.
If Lisa is to select a marble from the bag at random, prove that the probability that the
marble will be red or white is 7/10.
-/
def num_marbles : ℕ := 20
def num_blue : ℕ := 6
def num_red : ℕ := 9
def num_white : ℕ := num_marbles - (num_blue + num_red)

def probability_red_or_white : ℚ :=
  (num_red + num_white) / num_marbles

theorem probability_red_or_white_is_7_over_10 :
  probability_red_or_white = 7 / 10 := 
sorry

end NUMINAMATH_GPT_probability_red_or_white_is_7_over_10_l1837_183726


namespace NUMINAMATH_GPT_molly_age_condition_l1837_183718

-- Definitions
def S : ℕ := 38 - 6
def M : ℕ := 24

-- The proof problem
theorem molly_age_condition :
  (S / M = 4 / 3) → (S = 32) → (M = 24) :=
by
  intro h_ratio h_S
  sorry

end NUMINAMATH_GPT_molly_age_condition_l1837_183718


namespace NUMINAMATH_GPT_find_k_l1837_183794

theorem find_k : ∃ k : ℚ, (k = (k + 4) / 4) ∧ k = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1837_183794


namespace NUMINAMATH_GPT_snowfall_total_l1837_183737

theorem snowfall_total (snowfall_wed snowfall_thu snowfall_fri : ℝ)
  (h_wed : snowfall_wed = 0.33)
  (h_thu : snowfall_thu = 0.33)
  (h_fri : snowfall_fri = 0.22) :
  snowfall_wed + snowfall_thu + snowfall_fri = 0.88 :=
by
  rw [h_wed, h_thu, h_fri]
  norm_num

end NUMINAMATH_GPT_snowfall_total_l1837_183737


namespace NUMINAMATH_GPT_greatest_prime_factor_of_factorial_sum_l1837_183761

theorem greatest_prime_factor_of_factorial_sum : 
  ∃ p, Prime p ∧ p > 11 ∧ (∀ q, Prime q ∧ q > 11 → q ≤ 61) ∧ p = 61 :=
by
  sorry

end NUMINAMATH_GPT_greatest_prime_factor_of_factorial_sum_l1837_183761


namespace NUMINAMATH_GPT_calculate_selling_price_l1837_183770

-- Define the conditions
def purchase_price : ℝ := 900
def repair_cost : ℝ := 300
def gain_percentage : ℝ := 0.10

-- Define the total cost
def total_cost : ℝ := purchase_price + repair_cost

-- Define the gain
def gain : ℝ := gain_percentage * total_cost

-- Define the selling price
def selling_price : ℝ := total_cost + gain

-- The theorem to prove
theorem calculate_selling_price : selling_price = 1320 := by
  sorry

end NUMINAMATH_GPT_calculate_selling_price_l1837_183770


namespace NUMINAMATH_GPT_calculate_correctly_l1837_183763

theorem calculate_correctly (n : ℕ) (h1 : n - 21 = 52) : n - 40 = 33 := 
by 
  sorry

end NUMINAMATH_GPT_calculate_correctly_l1837_183763


namespace NUMINAMATH_GPT_wedding_cost_l1837_183780

theorem wedding_cost (venue_cost food_drink_cost guests_john : ℕ) 
  (guest_increment decorations_base decorations_per_guest transport_couple transport_per_guest entertainment_cost surchage_rate discount_thresh : ℕ) (discount_rate : ℕ) :
  let guests_wife := guests_john + (guests_john * guest_increment / 100)
  let venue_total := venue_cost + (venue_cost * surchage_rate / 100)
  let food_drink_total := if guests_wife > discount_thresh then (food_drink_cost * guests_wife) * (100 - discount_rate) / 100 else food_drink_cost * guests_wife
  let decorations_total := decorations_base + (decorations_per_guest * guests_wife)
  let transport_total := transport_couple + (transport_per_guest * guests_wife)
  (venue_total + food_drink_total + decorations_total + transport_total + entertainment_cost = 56200) :=
by {
  -- Constants given in the conditions
  let venue_cost := 10000
  let food_drink_cost := 500
  let guests_john := 50
  let guest_increment := 60
  let decorations_base := 2500
  let decorations_per_guest := 10
  let transport_couple := 200
  let transport_per_guest := 15
  let entertainment_cost := 4000
  let surchage_rate := 15
  let discount_thresh := 75
  let discount_rate := 10
  sorry
}

end NUMINAMATH_GPT_wedding_cost_l1837_183780


namespace NUMINAMATH_GPT_parabola_ratio_l1837_183777

noncomputable def AF_over_BF (p : ℝ) (h_p : p > 0) : ℝ :=
  let AF := 4 * p
  let x := (4 / 7) * p -- derived from solving the equation in the solution
  AF / x

theorem parabola_ratio (p : ℝ) (h_p : p > 0) : AF_over_BF p h_p = 7 :=
  sorry

end NUMINAMATH_GPT_parabola_ratio_l1837_183777


namespace NUMINAMATH_GPT_banana_price_reduction_l1837_183799

theorem banana_price_reduction (P_r : ℝ) (P : ℝ) (n : ℝ) (m : ℝ) (h1 : P_r = 3) (h2 : n = 40) (h3 : m = 64) 
  (h4 : 160 = (n / P_r) * 12) 
  (h5 : 96 = 160 - m) 
  (h6 : (40 / 8) = P) :
  (P - P_r) / P * 100 = 40 :=
by
  sorry

end NUMINAMATH_GPT_banana_price_reduction_l1837_183799


namespace NUMINAMATH_GPT_binary_101_is_5_l1837_183711

-- Define the function to convert a binary number to a decimal number
def binary_to_decimal : List Nat → Nat :=
  List.foldl (λ acc x => acc * 2 + x) 0

-- Convert the binary number 101₂ (which is [1, 0, 1] in list form) to decimal
theorem binary_101_is_5 : binary_to_decimal [1, 0, 1] = 5 := 
by 
  sorry

end NUMINAMATH_GPT_binary_101_is_5_l1837_183711


namespace NUMINAMATH_GPT_point_in_second_quadrant_l1837_183795

theorem point_in_second_quadrant (m n : ℝ)
  (h_translation : ∃ A' : ℝ × ℝ, A' = (m+2, n+3) ∧ (A'.1 < 0) ∧ (A'.2 > 0)) :
  m < -2 ∧ n > -3 :=
by
  sorry

end NUMINAMATH_GPT_point_in_second_quadrant_l1837_183795


namespace NUMINAMATH_GPT_maximize_ab2c3_l1837_183704

def positive_numbers (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 

def sum_constant (a b c A : ℝ) : Prop :=
  a + b + c = A

noncomputable def maximize_expression (a b c : ℝ) : ℝ :=
  a * b^2 * c^3

theorem maximize_ab2c3 (a b c A : ℝ) (h1 : positive_numbers a b c)
  (h2 : sum_constant a b c A) : 
  maximize_expression a b c ≤ maximize_expression (A / 6) (A / 3) (A / 2) :=
sorry

end NUMINAMATH_GPT_maximize_ab2c3_l1837_183704


namespace NUMINAMATH_GPT_color_of_85th_bead_l1837_183796

def bead_pattern : List String := ["red", "red", "orange", "yellow", "yellow", "yellow", "green", "blue", "blue"]

def bead_color (n : ℕ) : String :=
  bead_pattern.get! (n % bead_pattern.length)

theorem color_of_85th_bead : bead_color 84 = "yellow" := 
by
  sorry

end NUMINAMATH_GPT_color_of_85th_bead_l1837_183796


namespace NUMINAMATH_GPT_feet_of_wood_required_l1837_183774

def rung_length_in_inches : ℤ := 18
def spacing_between_rungs_in_inches : ℤ := 6
def height_to_climb_in_feet : ℤ := 50

def feet_per_rung := rung_length_in_inches / 12
def rungs_per_foot := 12 / spacing_between_rungs_in_inches
def total_rungs := height_to_climb_in_feet * rungs_per_foot
def total_feet_of_wood := total_rungs * feet_per_rung

theorem feet_of_wood_required :
  total_feet_of_wood = 150 :=
by
  sorry

end NUMINAMATH_GPT_feet_of_wood_required_l1837_183774


namespace NUMINAMATH_GPT_largest_K_inequality_l1837_183731

noncomputable def largest_K : ℝ := 18

theorem largest_K_inequality (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) 
(h_cond : a * b + b * c + c * a = a * b * c) :
( (a^a * (b^2 + c^2)) / ((a^a - 1)^2) + (b^b * (c^2 + a^2)) / ((b^b - 1)^2) + (c^c * (a^2 + b^2)) / ((c^c - 1)^2) )
≥ largest_K * ((a + b + c) / (a * b * c - 1)) ^ 2 :=
sorry

end NUMINAMATH_GPT_largest_K_inequality_l1837_183731


namespace NUMINAMATH_GPT_part_I_part_II_l1837_183797

variable (x : ℝ)

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

-- Define complement of B in real numbers
def neg_RB : Set ℝ := {x | -1 ≤ x ∧ x ≤ 5}

-- Part I: Statement for a = -2
theorem part_I (a : ℝ) (h : a = -2) : A a ∩ neg_RB = {x | -1 ≤ x ∧ x ≤ 1} := by
  sorry

-- Part II: Statement for A ∪ B = B
theorem part_II (a : ℝ) (h : ∀ x, A a x -> B x) : a < -4 ∨ a > 5 := by
  sorry

end NUMINAMATH_GPT_part_I_part_II_l1837_183797


namespace NUMINAMATH_GPT_count_letters_with_both_l1837_183776

theorem count_letters_with_both (a b c x : ℕ) 
  (h₁ : a = 24) 
  (h₂ : b = 7) 
  (h₃ : c = 40) 
  (H : a + b + x = c) : 
  x = 9 :=
by {
  -- Proof here
  sorry
}

end NUMINAMATH_GPT_count_letters_with_both_l1837_183776


namespace NUMINAMATH_GPT_behavior_of_f_in_interval_l1837_183788

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + 3 * m * x + 3

-- Define the property of even function
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

-- The theorem statement
theorem behavior_of_f_in_interval (m : ℝ) (hf_even : is_even_function (f m)) :
  m = 0 → (∀ x : ℝ, -4 < x ∧ x < 0 → f 0 x < f 0 (-x)) ∧ (∀ x : ℝ, 0 < x ∧ x < 2 → f 0 (-x) > f 0 x) :=
by 
  sorry

end NUMINAMATH_GPT_behavior_of_f_in_interval_l1837_183788


namespace NUMINAMATH_GPT_range_of_a_l1837_183715

noncomputable def p (a : ℝ) := ∀ x : ℝ, x^2 + a ≥ 0
noncomputable def q (a : ℝ) := ∃ x : ℝ, x^2 + (2 + a) * x + 1 = 0

theorem range_of_a (a : ℝ) : (p a ∧ q a) → (a ≥ 0) := by
  sorry

end NUMINAMATH_GPT_range_of_a_l1837_183715


namespace NUMINAMATH_GPT_rectangle_y_value_l1837_183746

theorem rectangle_y_value 
  (y : ℝ)
  (A : (0, 0) = E ∧ (0, 5) = F ∧ (y, 5) = G ∧ (y, 0) = H)
  (area : 5 * y = 35)
  (y_pos : y > 0) :
  y = 7 :=
sorry

end NUMINAMATH_GPT_rectangle_y_value_l1837_183746


namespace NUMINAMATH_GPT_dog_treats_cost_l1837_183750

theorem dog_treats_cost
  (treats_per_day : ℕ)
  (cost_per_treat : ℚ)
  (days_in_month : ℕ)
  (H1 : treats_per_day = 2)
  (H2 : cost_per_treat = 0.1)
  (H3 : days_in_month = 30) :
  treats_per_day * days_in_month * cost_per_treat = 6 :=
by sorry

end NUMINAMATH_GPT_dog_treats_cost_l1837_183750


namespace NUMINAMATH_GPT_sales_volume_relation_maximize_profit_l1837_183705

-- Definition of the conditions given in the problem
def cost_price : ℝ := 40
def min_selling_price : ℝ := 45
def initial_selling_price : ℝ := 45
def initial_sales_volume : ℝ := 700
def sales_decrease_rate : ℝ := 20

-- Lean statement for part 1
theorem sales_volume_relation (x : ℝ) : 
  (45 ≤ x) →
  (y = 700 - 20 * (x - 45)) → 
  y = -20 * x + 1600 := sorry

-- Lean statement for part 2
theorem maximize_profit (x : ℝ) :
  (45 ≤ x) →
  (P = (x - 40) * (-20 * x + 1600)) →
  ∃ max_x max_P, max_x = 60 ∧ max_P = 8000 := sorry

end NUMINAMATH_GPT_sales_volume_relation_maximize_profit_l1837_183705


namespace NUMINAMATH_GPT_c_minus_a_equals_90_l1837_183720

variable (a b c : ℝ)

def average_a_b (a b : ℝ) : Prop := (a + b) / 2 = 45
def average_b_c (b c : ℝ) : Prop := (b + c) / 2 = 90

theorem c_minus_a_equals_90
  (h1 : average_a_b a b)
  (h2 : average_b_c b c) :
  c - a = 90 :=
  sorry

end NUMINAMATH_GPT_c_minus_a_equals_90_l1837_183720


namespace NUMINAMATH_GPT_sum_of_a_b_c_d_e_l1837_183716

theorem sum_of_a_b_c_d_e (a b c d e : ℤ) (h1 : (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = 120)
  (h2 : a ≠ b) (h3 : a ≠ c) (h4 : a ≠ d) (h5 : a ≠ e) (h6 : b ≠ c) (h7 : b ≠ d) (h8 : b ≠ e) 
  (h9 : c ≠ d) (h10 : c ≠ e) (h11 : d ≠ e) : a + b + c + d + e = 33 := by
  sorry

end NUMINAMATH_GPT_sum_of_a_b_c_d_e_l1837_183716


namespace NUMINAMATH_GPT_average_rate_decrease_price_reduction_l1837_183728

-- Define the initial and final factory prices
def initial_price : ℝ := 200
def final_price : ℝ := 162

-- Define the function representing the average rate of decrease
def average_rate_of_decrease (x : ℝ) : Prop :=
  initial_price * (1 - x) * (1 - x) = final_price

-- Theorem stating the average rate of decrease (proving x = 0.1)
theorem average_rate_decrease : ∃ x : ℝ, average_rate_of_decrease x ∧ x = 0.1 :=
by
  use 0.1
  sorry

-- Define the selling price without reduction, sold without reduction, increase in pieces sold, and profit
def selling_price : ℝ := 200
def sold_without_reduction : ℕ := 20
def increase_pcs_per_5yuan_reduction : ℕ := 10
def profit : ℝ := 1150

-- Define the function representing the price reduction determination
def price_reduction_correct (m : ℝ) : Prop :=
  (38 - m) * (sold_without_reduction + 2 * m / 5) = profit

-- Theorem stating the price reduction (proving m = 15)
theorem price_reduction : ∃ m : ℝ, price_reduction_correct m ∧ m = 15 :=
by
  use 15
  sorry

end NUMINAMATH_GPT_average_rate_decrease_price_reduction_l1837_183728


namespace NUMINAMATH_GPT_big_al_ate_40_bananas_on_june_7_l1837_183742

-- Given conditions
def bananas_eaten_on_day (initial_bananas : ℕ) (day : ℕ) : ℕ :=
  initial_bananas + 4 * (day - 1)

def total_bananas_eaten (initial_bananas : ℕ) : ℕ :=
  bananas_eaten_on_day initial_bananas 1 +
  bananas_eaten_on_day initial_bananas 2 +
  bananas_eaten_on_day initial_bananas 3 +
  bananas_eaten_on_day initial_bananas 4 +
  bananas_eaten_on_day initial_bananas 5 +
  bananas_eaten_on_day initial_bananas 6 +
  bananas_eaten_on_day initial_bananas 7

noncomputable def final_bananas_on_june_7 (initial_bananas : ℕ) : ℕ :=
  bananas_eaten_on_day initial_bananas 7

-- Theorem to be proved
theorem big_al_ate_40_bananas_on_june_7 :
  ∃ initial_bananas, total_bananas_eaten initial_bananas = 196 ∧ final_bananas_on_june_7 initial_bananas = 40 :=
sorry

end NUMINAMATH_GPT_big_al_ate_40_bananas_on_june_7_l1837_183742


namespace NUMINAMATH_GPT_base_n_representation_l1837_183713

theorem base_n_representation (n : ℕ) (b : ℕ) (h₀ : 8 < n) (h₁ : ∃ b, (n : ℤ)^2 - (n+8) * (n : ℤ) + b = 0) : 
  b = 8 * n :=
by
  sorry

end NUMINAMATH_GPT_base_n_representation_l1837_183713


namespace NUMINAMATH_GPT_negation_prop_l1837_183707

variable {U : Type} (A B : Set U)
variable (x : U)

theorem negation_prop (h : x ∈ A ∩ B) : (x ∉ A ∩ B) → (x ∉ A ∧ x ∉ B) :=
sorry

end NUMINAMATH_GPT_negation_prop_l1837_183707


namespace NUMINAMATH_GPT_domain_g_eq_l1837_183727

noncomputable def domain_f : Set ℝ := {x | -8 ≤ x ∧ x ≤ 4}

noncomputable def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f (-2 * x)

theorem domain_g_eq (f : ℝ → ℝ) (h : ∀ x, x ∈ domain_f → f x ∈ domain_f) :
  {x | x ∈ [-2, 4]} = {x | -2 ≤ x ∧ x ≤ 4} :=
by {
  sorry
}

end NUMINAMATH_GPT_domain_g_eq_l1837_183727


namespace NUMINAMATH_GPT_initial_discount_l1837_183712

theorem initial_discount (P D : ℝ) 
  (h1 : P - 71.4 = 5.25)
  (h2 : P * (1 - D) * 1.25 = 71.4) : 
  D = 0.255 :=
by {
  sorry
}

end NUMINAMATH_GPT_initial_discount_l1837_183712


namespace NUMINAMATH_GPT_tamara_is_68_inch_l1837_183723

-- Defining the conditions
variables (K T : ℕ)

-- Condition 1: Tamara's height in terms of Kim's height
def tamara_height := T = 3 * K - 4

-- Condition 2: Combined height of Tamara and Kim
def combined_height := T + K = 92

-- Statement to prove: Tamara's height is 68 inches
theorem tamara_is_68_inch (h1 : tamara_height T K) (h2 : combined_height T K) : T = 68 :=
by
  sorry

end NUMINAMATH_GPT_tamara_is_68_inch_l1837_183723


namespace NUMINAMATH_GPT_trigonometric_ratio_sum_l1837_183798

open Real

theorem trigonometric_ratio_sum (x y : ℝ) 
  (h₁ : sin x / sin y = 2) 
  (h₂ : cos x / cos y = 1 / 3) :
  sin (2 * x) / sin (2 * y) + cos (2 * x) / cos (2 * y) = 41 / 57 := 
by
  sorry

end NUMINAMATH_GPT_trigonometric_ratio_sum_l1837_183798


namespace NUMINAMATH_GPT_solve_for_q_l1837_183730

theorem solve_for_q (p q : ℚ) (h1 : 5 * p + 6 * q = 10) (h2 : 6 * p + 5 * q = 17) : q = -25 / 11 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_q_l1837_183730


namespace NUMINAMATH_GPT_find_m_l1837_183767

def circle1 (x y m : ℝ) : Prop := (x + 2)^2 + (y - m)^2 = 9
def circle2 (x y m : ℝ) : Prop := (x - m)^2 + (y + 1)^2 = 4

theorem find_m (m : ℝ) : 
  ∃ x1 y1 x2 y2 : ℝ, 
    circle1 x1 y1 m ∧ 
    circle2 x2 y2 m ∧ 
    (m + 2)^2 + (-1 - m)^2 = 25 → 
    m = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1837_183767


namespace NUMINAMATH_GPT_soda_cans_ratio_l1837_183710

theorem soda_cans_ratio
  (initial_cans : ℕ := 22)
  (cans_taken : ℕ := 6)
  (final_cans : ℕ := 24)
  (x : ℚ := 1 / 2)
  (cans_left : ℕ := 16)
  (cans_bought : ℕ := 16 * 1 / 2) :
  (cans_bought / cans_left : ℚ) = 1 / 2 :=
sorry

end NUMINAMATH_GPT_soda_cans_ratio_l1837_183710


namespace NUMINAMATH_GPT_kombucha_bottles_l1837_183722

theorem kombucha_bottles (b_m : ℕ) (c : ℝ) (r : ℝ) (m : ℕ)
  (hb : b_m = 15) (hc : c = 3.00) (hr : r = 0.10) (hm : m = 12) :
  (b_m * m * r) / c = 6 := by
  sorry

end NUMINAMATH_GPT_kombucha_bottles_l1837_183722


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1837_183725

theorem arithmetic_sequence_sum :
  3 * (75 + 77 + 79 + 81 + 83) = 1185 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1837_183725


namespace NUMINAMATH_GPT_correct_option_B_l1837_183772

theorem correct_option_B (x y a b : ℝ) :
  (3 * x + 2 * x^2 ≠ 5 * x) →
  (-y^2 * x + x * y^2 = 0) →
  (-a * b - a * b ≠ 0) →
  (3 * a^3 * b^2 - 2 * a^3 * b^2 ≠ 1) →
  (-y^2 * x + x * y^2 = 0) :=
by
  intros hA hB hC hD
  exact hB

end NUMINAMATH_GPT_correct_option_B_l1837_183772


namespace NUMINAMATH_GPT_kite_ratio_equality_l1837_183706

-- Definitions for points, lines, and conditions in the geometric setup
variables {Point : Type*} [MetricSpace Point]

-- Assuming A, B, C, D, P, E, F, G, H, I, J are points
variable (A B C D P E F G H I J : Point)

-- Conditions based on the problem
variables (AB_eq_AD : dist A B = dist A D)
          (BC_eq_CD : dist B C = dist C D)
          (on_BD : P ∈ line B D)
          (line_PE_inter_AD : E ∈ line P E ∧ E ∈ line A D)
          (line_PF_inter_BC : F ∈ line P F ∧ F ∈ line B C)
          (line_PG_inter_AB : G ∈ line P G ∧ G ∈ line A B)
          (line_PH_inter_CD : H ∈ line P H ∧ H ∈ line C D)
          (GF_inter_BD_at_I : I ∈ line G F ∧ I ∈ line B D)
          (EH_inter_BD_at_J : J ∈ line E H ∧ J ∈ line B D)

-- The statement to prove
theorem kite_ratio_equality :
  dist P I / dist P B = dist P J / dist P D := sorry

end NUMINAMATH_GPT_kite_ratio_equality_l1837_183706


namespace NUMINAMATH_GPT_eq_op_op_op_92_l1837_183754

noncomputable def opN (N : ℝ) : ℝ := 0.75 * N + 2

theorem eq_op_op_op_92 : opN (opN (opN 92)) = 43.4375 :=
by
  sorry

end NUMINAMATH_GPT_eq_op_op_op_92_l1837_183754


namespace NUMINAMATH_GPT_max_not_sum_S_l1837_183759

def S : Set ℕ := {n | ∃ k : ℕ, n = 10^k + 1000}

theorem max_not_sum_S : ∀ x : ℕ, (∀ y ∈ S, ∃ m : ℕ, x ≠ m * y) ↔ x = 34999 := by
  sorry

end NUMINAMATH_GPT_max_not_sum_S_l1837_183759


namespace NUMINAMATH_GPT_remainder_83_pow_89_times_5_mod_11_l1837_183778

theorem remainder_83_pow_89_times_5_mod_11 : 
  (83^89 * 5) % 11 = 10 := 
by
  have h1 : 83 % 11 = 6 := by sorry
  have h2 : 6^10 % 11 = 1 := by sorry
  have h3 : 89 = 8 * 10 + 9 := by sorry
  sorry

end NUMINAMATH_GPT_remainder_83_pow_89_times_5_mod_11_l1837_183778


namespace NUMINAMATH_GPT_total_trips_correct_l1837_183735

-- Define Timothy's movie trips in 2009
def timothy_2009_trips : ℕ := 24

-- Define Timothy's movie trips in 2010
def timothy_2010_trips : ℕ := timothy_2009_trips + 7

-- Define Theresa's movie trips in 2009
def theresa_2009_trips : ℕ := timothy_2009_trips / 2

-- Define Theresa's movie trips in 2010
def theresa_2010_trips : ℕ := timothy_2010_trips * 2

-- Define the total number of trips for Timothy and Theresa in 2009 and 2010
def total_trips : ℕ := (timothy_2009_trips + timothy_2010_trips) + (theresa_2009_trips + theresa_2010_trips)

-- Prove the total number of trips is 129
theorem total_trips_correct : total_trips = 129 :=
by
  sorry

end NUMINAMATH_GPT_total_trips_correct_l1837_183735


namespace NUMINAMATH_GPT_digit_d_is_six_l1837_183782

theorem digit_d_is_six (d : ℕ) (h_even : d % 2 = 0) (h_digits_sum : 7 + 4 + 8 + 2 + d % 9 = 0) : d = 6 :=
by 
  sorry

end NUMINAMATH_GPT_digit_d_is_six_l1837_183782


namespace NUMINAMATH_GPT_mold_growth_problem_l1837_183771

/-- Given the conditions:
    - Initial mold spores: 50 at 9:00 a.m.
    - Colony doubles in size every 10 minutes.
    - Time elapsed: 70 minutes from 9:00 a.m. to 10:10 a.m.,

    Prove that the number of mold spores at 10:10 a.m. is 6400 -/
theorem mold_growth_problem : 
  let initial_mold_spores := 50
  let doubling_period_minutes := 10
  let elapsed_minutes := 70
  let doublings := elapsed_minutes / doubling_period_minutes
  let final_population := initial_mold_spores * (2 ^ doublings)
  final_population = 6400 :=
by 
  let initial_mold_spores := 50
  let doubling_period_minutes := 10
  let elapsed_minutes := 70
  let doublings := elapsed_minutes / doubling_period_minutes
  let final_population := initial_mold_spores * (2 ^ doublings)
  sorry

end NUMINAMATH_GPT_mold_growth_problem_l1837_183771


namespace NUMINAMATH_GPT_length_of_AB_l1837_183708

-- Definitions based on given conditions:
variables (AB BC CD DE AE AC : ℕ)
variables (h1 : BC = 3 * CD) (h2 : DE = 8) (h3 : AC = 11) (h4 : AE = 21)

-- The theorem stating the length of AB given the conditions.
theorem length_of_AB (AB BC CD DE AE AC : ℕ)
  (h1 : BC = 3 * CD) (h2 : DE = 8) (h3 : AC = 11) (h4 : AE = 21) : AB = 5 := by
  sorry

end NUMINAMATH_GPT_length_of_AB_l1837_183708


namespace NUMINAMATH_GPT_greatest_value_of_b_l1837_183753

theorem greatest_value_of_b (b : ℝ) : -b^2 + 8 * b - 15 ≥ 0 → b ≤ 5 := sorry

end NUMINAMATH_GPT_greatest_value_of_b_l1837_183753


namespace NUMINAMATH_GPT_range_of_f_l1837_183749

noncomputable def f (x : ℝ) : ℝ :=
  (Real.arccos x)^4 + (Real.arcsin x)^4

theorem range_of_f :
  ∀ y, (∃ x, x ∈ Set.Icc (-1:ℝ) 1 ∧ f x = y) ↔ y ∈ Set.Icc 0 (Real.pi^4 / 8) :=
sorry

end NUMINAMATH_GPT_range_of_f_l1837_183749


namespace NUMINAMATH_GPT_age_ratios_l1837_183775

variable (A B : ℕ)

-- Given conditions
theorem age_ratios :
  (A / B = 2 / 1) → (A - 4 = B + 4) → ((A + 4) / (B - 4) = 5 / 1) :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_age_ratios_l1837_183775


namespace NUMINAMATH_GPT_evaluate_f_difference_l1837_183700

def f (x : ℝ) : ℝ := x^4 + x^2 + 3*x^3 + 5*x

theorem evaluate_f_difference : f 5 - f (-5) = 800 := by
  sorry

end NUMINAMATH_GPT_evaluate_f_difference_l1837_183700


namespace NUMINAMATH_GPT_M_minus_N_positive_l1837_183766

variable (a b : ℝ)

def M : ℝ := 10 * a^2 + b^2 - 7 * a + 8
def N : ℝ := a^2 + b^2 + 5 * a + 1

theorem M_minus_N_positive : M a b - N a b ≥ 3 := by
  sorry

end NUMINAMATH_GPT_M_minus_N_positive_l1837_183766


namespace NUMINAMATH_GPT_johns_average_speed_l1837_183736

def continuous_driving_duration (start_time end_time : ℝ) (distance : ℝ) : Prop :=
start_time = 10.5 ∧ end_time = 14.75 ∧ distance = 190

theorem johns_average_speed
  (start_time end_time : ℝ) 
  (distance : ℝ)
  (h : continuous_driving_duration start_time end_time distance) :
  (distance / (end_time - start_time) = 44.7) :=
by
  sorry

end NUMINAMATH_GPT_johns_average_speed_l1837_183736


namespace NUMINAMATH_GPT_arithmetic_sequence_ratio_l1837_183714

theorem arithmetic_sequence_ratio (x y a₁ a₂ a₃ b₁ b₂ b₃ b₄ : ℝ) (h₁ : x ≠ y)
    (h₂ : a₁ = x + d) (h₃ : a₂ = x + 2 * d) (h₄ : a₃ = x + 3 * d) (h₅ : y = x + 4 * d)
    (h₆ : b₁ = x - d') (h₇ : b₂ = x + d') (h₈ : b₃ = x + 2 * d') (h₉ : y = x + 3 * d') (h₁₀ : b₄ = x + 4 * d') :
    (b₄ - b₃) / (a₂ - a₁) = 8 / 3 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_ratio_l1837_183714


namespace NUMINAMATH_GPT_area_of_square_A_l1837_183738

noncomputable def square_areas (a b : ℕ) : Prop :=
  (b ^ 2 = 81) ∧ (a = b + 4)

theorem area_of_square_A : ∃ a b : ℕ, square_areas a b → a ^ 2 = 169 :=
by
  sorry

end NUMINAMATH_GPT_area_of_square_A_l1837_183738


namespace NUMINAMATH_GPT_pyramid_partition_volumes_l1837_183702

noncomputable def pyramid_partition_ratios (S A B C D P Q V1 V2 : ℝ) : Prop :=
  let P := ((S + B) / 2 : ℝ)
  let Q := ((S + D) / 2 : ℝ)
  (V1 < V2) → 
  (V2 / V1 = 5)

theorem pyramid_partition_volumes
  (S A B C D P Q : ℝ)
  (V1 V2 : ℝ)
  (hP : P = (S + B) / 2)
  (hQ : Q = (S + D) / 2)
  (hV1 : V1 < V2)
  : V2 / V1 = 5 := 
sorry

end NUMINAMATH_GPT_pyramid_partition_volumes_l1837_183702


namespace NUMINAMATH_GPT_unique_9_tuple_satisfying_condition_l1837_183719

theorem unique_9_tuple_satisfying_condition :
  ∃! (a : Fin 9 → ℕ), 
    (∀ i j k : Fin 9, i < j ∧ j < k →
      ∃ l : Fin 9, l ≠ i ∧ l ≠ j ∧ l ≠ k ∧ a i + a j + a k + a l = 100) :=
sorry

end NUMINAMATH_GPT_unique_9_tuple_satisfying_condition_l1837_183719


namespace NUMINAMATH_GPT_total_amount_due_is_correct_l1837_183747

-- Define the initial conditions
def initial_amount : ℝ := 350
def first_year_interest_rate : ℝ := 0.03
def second_and_third_years_interest_rate : ℝ := 0.05

-- Define the total amount calculation after three years.
def total_amount_after_three_years (P : ℝ) (r1 : ℝ) (r2 : ℝ) : ℝ :=
  let first_year_amount := P * (1 + r1)
  let second_year_amount := first_year_amount * (1 + r2)
  let third_year_amount := second_year_amount * (1 + r2)
  third_year_amount

theorem total_amount_due_is_correct : 
  total_amount_after_three_years initial_amount first_year_interest_rate second_and_third_years_interest_rate = 397.45 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_due_is_correct_l1837_183747


namespace NUMINAMATH_GPT_point_on_line_l1837_183739

theorem point_on_line (m n k : ℝ) (h1 : m = 2 * n + 5) (h2 : m + 4 = 2 * (n + k) + 5) : k = 2 := by
  sorry

end NUMINAMATH_GPT_point_on_line_l1837_183739


namespace NUMINAMATH_GPT_total_population_l1837_183732

-- Define the conditions
variables (T G Td Lb : ℝ)

-- Given conditions and the result
def conditions : Prop :=
  G = 1 / 2 * T ∧
  Td = 0.60 * G ∧
  Lb = 16000 ∧
  T = Td + G + Lb

-- Problem statement: Prove that the total population T is 80000
theorem total_population (h : conditions T G Td Lb) : T = 80000 :=
by
  sorry

end NUMINAMATH_GPT_total_population_l1837_183732


namespace NUMINAMATH_GPT_servant_leaves_after_nine_months_l1837_183748

-- Definitions based on conditions
def yearly_salary : ℕ := 90 + 90
def monthly_salary : ℕ := yearly_salary / 12
def amount_received : ℕ := 45 + 90

-- The theorem to prove
theorem servant_leaves_after_nine_months :
    amount_received / monthly_salary = 9 :=
by
  -- Using the provided conditions, we establish the equality we need.
  sorry

end NUMINAMATH_GPT_servant_leaves_after_nine_months_l1837_183748


namespace NUMINAMATH_GPT_frood_least_throw_points_more_than_eat_l1837_183757

theorem frood_least_throw_points_more_than_eat (n : ℕ) : n^2 > 12 * n ↔ n ≥ 13 :=
sorry

end NUMINAMATH_GPT_frood_least_throw_points_more_than_eat_l1837_183757


namespace NUMINAMATH_GPT_gcd_f_50_51_l1837_183789

-- Define f(x)
def f (x : ℤ) : ℤ := x^3 - x^2 + 2 * x + 2000

-- State the problem: Prove gcd(f(50), f(51)) = 8
theorem gcd_f_50_51 : Int.gcd (f 50) (f 51) = 8 := by
  sorry

end NUMINAMATH_GPT_gcd_f_50_51_l1837_183789


namespace NUMINAMATH_GPT_blue_water_bottles_initial_count_l1837_183724

theorem blue_water_bottles_initial_count
    (red : ℕ) (black : ℕ) (taken_out : ℕ) (left : ℕ) (initial_blue : ℕ) :
    red = 2 →
    black = 3 →
    taken_out = 5 →
    left = 4 →
    initial_blue + red + black = taken_out + left →
    initial_blue = 4 := by
  intros
  sorry

end NUMINAMATH_GPT_blue_water_bottles_initial_count_l1837_183724


namespace NUMINAMATH_GPT_fibonacci_series_sum_l1837_183733

def fibonacci (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else fibonacci (n-1) + fibonacci (n-2)

noncomputable def sum_fibonacci_fraction : ℚ :=
  ∑' (n : ℕ), (fibonacci n : ℚ) / (5^n : ℚ)

theorem fibonacci_series_sum : sum_fibonacci_fraction = 5 / 19 := by
  sorry

end NUMINAMATH_GPT_fibonacci_series_sum_l1837_183733


namespace NUMINAMATH_GPT_gcd_546_210_l1837_183768

theorem gcd_546_210 : Nat.gcd 546 210 = 42 := by
  sorry -- Proof is required to solve

end NUMINAMATH_GPT_gcd_546_210_l1837_183768


namespace NUMINAMATH_GPT_integer_roots_of_poly_l1837_183773

-- Define the polynomial
def poly (x : ℤ) (b1 b2 : ℤ) : ℤ :=
  x^3 + b2 * x ^ 2 + b1 * x + 18

-- The list of possible integer roots
def possible_integer_roots := [-18, -9, -6, -3, -2, -1, 1, 2, 3, 6, 9, 18]

-- Statement of the theorem
theorem integer_roots_of_poly (b1 b2 : ℤ) :
  ∀ x : ℤ, poly x b1 b2 = 0 → x ∈ possible_integer_roots :=
sorry

end NUMINAMATH_GPT_integer_roots_of_poly_l1837_183773


namespace NUMINAMATH_GPT_total_cookies_is_58_l1837_183760

noncomputable def total_cookies : ℝ :=
  let M : ℝ := 5
  let T : ℝ := 2 * M
  let W : ℝ := T + 0.4 * T
  let Th : ℝ := W - 0.25 * W
  let F : ℝ := Th - 0.25 * Th
  let Sa : ℝ := F - 0.25 * F
  let Su : ℝ := Sa - 0.25 * Sa
  M + T + W + Th + F + Sa + Su

theorem total_cookies_is_58 : total_cookies = 58 :=
by
  sorry

end NUMINAMATH_GPT_total_cookies_is_58_l1837_183760


namespace NUMINAMATH_GPT_shoes_cost_l1837_183785

theorem shoes_cost (S : ℝ) : 
  let suit := 430
  let discount := 100
  let total_paid := 520
  suit + S - discount = total_paid -> 
  S = 190 :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_shoes_cost_l1837_183785


namespace NUMINAMATH_GPT_large_square_pattern_l1837_183765

theorem large_square_pattern :
  999999^2 = 1000000 * 999998 + 1 :=
by sorry

end NUMINAMATH_GPT_large_square_pattern_l1837_183765


namespace NUMINAMATH_GPT_parabolas_intersect_on_circle_l1837_183734

theorem parabolas_intersect_on_circle :
  let parabola1 (x y : ℝ) := y = (x - 2)^2
  let parabola2 (x y : ℝ) := x + 6 = (y + 1)^2
  ∃ (cx cy r : ℝ), ∀ (x y : ℝ), (parabola1 x y ∧ parabola2 x y) → (x - cx)^2 + (y - cy)^2 = r^2 ∧ r^2 = 33/2 :=
by
  sorry

end NUMINAMATH_GPT_parabolas_intersect_on_circle_l1837_183734


namespace NUMINAMATH_GPT_investment_value_l1837_183781

-- Define the compound interest calculation
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

-- Given values
def P : ℝ := 8000
def r : ℝ := 0.05
def n : ℕ := 7

-- The theorem statement in Lean 4
theorem investment_value :
  round (compound_interest P r n) = 11257 :=
by
  sorry

end NUMINAMATH_GPT_investment_value_l1837_183781


namespace NUMINAMATH_GPT_final_result_is_110_l1837_183729

theorem final_result_is_110 (x : ℕ) (h1 : x = 155) : (x * 2 - 200) = 110 :=
by
  -- placeholder for the solution proof
  sorry

end NUMINAMATH_GPT_final_result_is_110_l1837_183729


namespace NUMINAMATH_GPT_contrapositive_of_given_condition_l1837_183755

-- Definitions
variable (P Q : Prop)

-- Given condition: If Jane answered all questions correctly, she will get a prize
axiom h : P → Q

-- Statement to be proven: If Jane did not get a prize, she answered at least one question incorrectly
theorem contrapositive_of_given_condition : ¬ Q → ¬ P := by
  sorry

end NUMINAMATH_GPT_contrapositive_of_given_condition_l1837_183755


namespace NUMINAMATH_GPT_original_deck_card_count_l1837_183793

theorem original_deck_card_count (r b : ℕ) 
  (h1 : r / (r + b) = 1 / 4)
  (h2 : r / (r + b + 6) = 1 / 6) : r + b = 12 :=
sorry

end NUMINAMATH_GPT_original_deck_card_count_l1837_183793


namespace NUMINAMATH_GPT_rectangular_prism_edges_vertices_faces_sum_l1837_183784

theorem rectangular_prism_edges_vertices_faces_sum (a b c : ℕ) (h1: a = 2) (h2: b = 3) (h3: c = 4) : 
  12 + 8 + 6 = 26 :=
by
  sorry

end NUMINAMATH_GPT_rectangular_prism_edges_vertices_faces_sum_l1837_183784


namespace NUMINAMATH_GPT_ratio_of_volumes_of_tetrahedrons_l1837_183762

theorem ratio_of_volumes_of_tetrahedrons (a b : ℝ) (h : a / b = 1 / 2) : (a^3) / (b^3) = 1 / 8 :=
by
-- proof goes here
sorry

end NUMINAMATH_GPT_ratio_of_volumes_of_tetrahedrons_l1837_183762


namespace NUMINAMATH_GPT_base_square_eq_l1837_183769

theorem base_square_eq (b : ℕ) (h : (3*b + 3)^2 = b^3 + 2*b^2 + 3*b) : b = 9 :=
sorry

end NUMINAMATH_GPT_base_square_eq_l1837_183769


namespace NUMINAMATH_GPT_parabola_chord_solution_l1837_183792

noncomputable def parabola_chord : Prop :=
  ∃ x_A x_B : ℝ, (140 = 5 * x_B^2 + 2 * x_A^2) ∧ 
  ((x_A = -5 * Real.sqrt 2 ∧ x_B = 2 * Real.sqrt 2) ∨ 
   (x_A = 5 * Real.sqrt 2 ∧ x_B = -2 * Real.sqrt 2))

theorem parabola_chord_solution : parabola_chord := 
sorry

end NUMINAMATH_GPT_parabola_chord_solution_l1837_183792


namespace NUMINAMATH_GPT_where_they_meet_l1837_183783

/-- Define the conditions under which Petya and Vasya are walking. -/
structure WalkingCondition (n : ℕ) where
  lampposts : ℕ
  start_p : ℕ
  start_v : ℕ
  position_p : ℕ
  position_v : ℕ

/-- Initial conditions based on the problem statement. -/
def initialCondition : WalkingCondition 100 := {
  lampposts := 100,
  start_p := 1,
  start_v := 100,
  position_p := 22,
  position_v := 88
}

/-- Prove Petya and Vasya will meet at the 64th lamppost. -/
theorem where_they_meet (cond : WalkingCondition 100) : 64 ∈ { x | x = 64 } :=
  -- The formal proof would go here.
  sorry

end NUMINAMATH_GPT_where_they_meet_l1837_183783


namespace NUMINAMATH_GPT_income_of_sixth_member_l1837_183745

def income_member1 : ℝ := 11000
def income_member2 : ℝ := 15000
def income_member3 : ℝ := 10000
def income_member4 : ℝ := 9000
def income_member5 : ℝ := 13000
def number_of_members : ℕ := 6
def average_income : ℝ := 12000
def total_income_of_five_members := income_member1 + income_member2 + income_member3 + income_member4 + income_member5

theorem income_of_sixth_member :
  6 * average_income - total_income_of_five_members = 14000 := by
  sorry

end NUMINAMATH_GPT_income_of_sixth_member_l1837_183745


namespace NUMINAMATH_GPT_integer_root_count_l1837_183758

theorem integer_root_count (b : ℝ) :
  (∃ r s : ℤ, r + s = b ∧ r * s = 8 * b) ↔
  b = -9 ∨ b = 0 ∨ b = 9 :=
sorry

end NUMINAMATH_GPT_integer_root_count_l1837_183758


namespace NUMINAMATH_GPT_solve_for_x_l1837_183791

theorem solve_for_x (x : ℝ) (h : 4 * x + 45 ≠ 0) :
  (8 * x^2 + 80 * x + 4) / (4 * x + 45) = 2 * x + 3 → x = -131 / 22 := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_x_l1837_183791


namespace NUMINAMATH_GPT_my_age_is_five_times_son_age_l1837_183741

theorem my_age_is_five_times_son_age (son_age_next : ℕ) (my_age : ℕ) (h1 : son_age_next = 8) (h2 : my_age = 5 * (son_age_next - 1)) : my_age = 35 :=
by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_my_age_is_five_times_son_age_l1837_183741


namespace NUMINAMATH_GPT_infinite_primes_divide_f_l1837_183703

def non_constant_function (f : ℕ → ℕ) : Prop :=
  ∃ a b : ℕ, a ≠ b ∧ f a ≠ f b

def divisibility_condition (f : ℕ → ℕ) : Prop :=
  ∀ a b : ℕ, a ≠ b → (a - b) ∣ (f a - f b)

theorem infinite_primes_divide_f (f : ℕ → ℕ) 
  (h_non_const : non_constant_function f)
  (h_div : divisibility_condition f) :
  ∃ᶠ p in Filter.atTop, ∃ c : ℕ, p ∣ f c := sorry

end NUMINAMATH_GPT_infinite_primes_divide_f_l1837_183703


namespace NUMINAMATH_GPT_ferry_journey_time_difference_l1837_183743

/-
  Problem statement:
  Prove that the journey of ferry Q is 1 hour longer than the journey of ferry P,
  given the following conditions:
  1. Ferry P travels for 3 hours at 6 kilometers per hour.
  2. Ferry Q takes a route that is two times longer than ferry P.
  3. Ferry P is slower than ferry Q by 3 kilometers per hour.
-/

theorem ferry_journey_time_difference :
  let speed_P := 6
  let time_P := 3
  let distance_P := speed_P * time_P
  let distance_Q := 2 * distance_P
  let speed_diff := 3
  let speed_Q := speed_P + speed_diff
  let time_Q := distance_Q / speed_Q
  time_Q - time_P = 1 :=
by
  sorry

end NUMINAMATH_GPT_ferry_journey_time_difference_l1837_183743


namespace NUMINAMATH_GPT_problem_l1837_183779

variable (p q : Prop)

theorem problem (h : ¬ (¬ p ∨ ¬ q)) : (p ∧ q) ∧ (p ∨ q) :=
by
  sorry

end NUMINAMATH_GPT_problem_l1837_183779


namespace NUMINAMATH_GPT_find_linear_in_two_variables_l1837_183756

def is_linear_in_two_variables (eq : String) : Bool :=
  eq = "x=y+1"

theorem find_linear_in_two_variables :
  (is_linear_in_two_variables "4xy=2" = false) ∧
  (is_linear_in_two_variables "1-x=7" = false) ∧
  (is_linear_in_two_variables "x^2+2y=-2" = false) ∧
  (is_linear_in_two_variables "x=y+1" = true) :=
by
  sorry

end NUMINAMATH_GPT_find_linear_in_two_variables_l1837_183756


namespace NUMINAMATH_GPT_value_of_a_plus_b_l1837_183787

theorem value_of_a_plus_b :
  ∀ (a b x y : ℝ), x = 3 → y = -2 → 
  a * x + b * y = 2 → b * x + a * y = -3 → 
  a + b = -1 := 
by
  intros a b x y hx hy h1 h2
  subst hx
  subst hy
  sorry

end NUMINAMATH_GPT_value_of_a_plus_b_l1837_183787


namespace NUMINAMATH_GPT_smallest_c_for_inverse_l1837_183717

noncomputable def g (x : ℝ) : ℝ := (x + 3)^2 - 6

theorem smallest_c_for_inverse : 
  ∃ (c : ℝ), (∀ x1 x2, x1 ≥ c → x2 ≥ c → g x1 = g x2 → x1 = x2) ∧ 
            (∀ c', c' < c → ∃ x1 x2, x1 ≥ c' → x2 ≥ c' → g x1 = g x2 ∧ x1 ≠ x2) ∧ 
            c = -3 :=
by 
  sorry

end NUMINAMATH_GPT_smallest_c_for_inverse_l1837_183717


namespace NUMINAMATH_GPT_base_number_is_2_l1837_183764

open Real

noncomputable def valid_x (x : ℝ) (n : ℕ) := sqrt (x^n) = 64

theorem base_number_is_2 (x : ℝ) (n : ℕ) (h : valid_x x n) (hn : n = 12) : x = 2 := 
by 
  sorry

end NUMINAMATH_GPT_base_number_is_2_l1837_183764


namespace NUMINAMATH_GPT_trays_from_second_table_l1837_183701

def trays_per_trip : ℕ := 4
def trips : ℕ := 9
def trays_from_first_table : ℕ := 20

theorem trays_from_second_table :
  trays_per_trip * trips - trays_from_first_table = 16 :=
by
  sorry

end NUMINAMATH_GPT_trays_from_second_table_l1837_183701


namespace NUMINAMATH_GPT_ratio_of_triangle_areas_bcx_acx_l1837_183721

theorem ratio_of_triangle_areas_bcx_acx
  (BC AC : ℕ) (hBC : BC = 36) (hAC : AC = 45)
  (is_angle_bisector_CX : ∀ BX AX : ℕ, BX / AX = BC / AC) :
  (∃ BX AX : ℕ, BX / AX = 4 / 5) :=
by
  have h_ratio := is_angle_bisector_CX 36 45
  rw [hBC, hAC] at h_ratio
  exact ⟨4, 5, h_ratio⟩

end NUMINAMATH_GPT_ratio_of_triangle_areas_bcx_acx_l1837_183721


namespace NUMINAMATH_GPT_remainder_67pow67_add_67_div_68_l1837_183740

-- Lean statement starting with the question and conditions translated to Lean

theorem remainder_67pow67_add_67_div_68 : 
  (67 ^ 67 + 67) % 68 = 66 := 
by
  -- Condition: 67 ≡ -1 mod 68
  have h : 67 % 68 = -1 % 68 := by norm_num
  sorry

end NUMINAMATH_GPT_remainder_67pow67_add_67_div_68_l1837_183740


namespace NUMINAMATH_GPT_combined_weight_l1837_183709

-- Define the conditions
variables (Ron_weight Roger_weight Rodney_weight : ℕ)

-- Define the conditions as Lean propositions
def conditions : Prop :=
  Rodney_weight = 2 * Roger_weight ∧ 
  Roger_weight = 4 * Ron_weight - 7 ∧ 
  Rodney_weight = 146

-- Define the proof goal
def proof_goal : Prop :=
  Rodney_weight + Roger_weight + Ron_weight = 239

theorem combined_weight (Ron_weight Roger_weight Rodney_weight : ℕ) (h : conditions Ron_weight Roger_weight Rodney_weight) : 
  proof_goal Ron_weight Roger_weight Rodney_weight :=
sorry

end NUMINAMATH_GPT_combined_weight_l1837_183709


namespace NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l1837_183752

-- Defining the necessary variables as real numbers for the proof
variables (x y : ℝ)

-- Prove the first expression simplification
theorem simplify_expr1 : 
  (x + 2 * y) * (x - 2 * y) - x * (x + 3 * y) = -4 * y^2 - 3 * x * y :=
  sorry

-- Prove the second expression simplification
theorem simplify_expr2 : 
  (x - 1 - 3 / (x + 1)) / ((x^2 - 4 * x + 4) / (x + 1)) = (x + 2) / (x - 2) :=
  sorry

end NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l1837_183752


namespace NUMINAMATH_GPT_trapezoid_circumscribed_radius_l1837_183744

theorem trapezoid_circumscribed_radius 
  (a b : ℝ) 
  (height : ℝ)
  (ratio_ab : a / b = 5 / 12)
  (height_eq_midsegment : height = 17) :
  ∃ r : ℝ, r = 13 :=
by
  -- Assuming conditions directly as given
  have h1 : a / b = 5 / 12 := ratio_ab
  have h2 : height = 17 := height_eq_midsegment
  -- The rest of the proof goes here
  sorry

end NUMINAMATH_GPT_trapezoid_circumscribed_radius_l1837_183744


namespace NUMINAMATH_GPT_min_value_expression_min_value_is_7_l1837_183751

theorem min_value_expression (x : ℝ) (hx : x > 0) : 
  6 * x + 1 / (x^6) ≥ 7 :=
sorry

theorem min_value_is_7 : 
  6 * 1 + 1 / (1^6) = 7 :=
by norm_num

end NUMINAMATH_GPT_min_value_expression_min_value_is_7_l1837_183751


namespace NUMINAMATH_GPT_zero_extreme_points_l1837_183786

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 3*x

theorem zero_extreme_points : ∀ x : ℝ, 
  ∃! (y : ℝ), deriv f y = 0 → y = x :=
by
  sorry

end NUMINAMATH_GPT_zero_extreme_points_l1837_183786


namespace NUMINAMATH_GPT_range_of_a_neg_p_true_l1837_183790

theorem range_of_a_neg_p_true :
  (∀ x : ℝ, x ∈ Set.Ioo (-2:ℝ) 0 → x^2 + (2*a - 1)*x + a ≠ 0) →
  ∀ a : ℝ, a ∈ Set.Icc 0 ((2 + Real.sqrt 3) / 2) :=
sorry

end NUMINAMATH_GPT_range_of_a_neg_p_true_l1837_183790
