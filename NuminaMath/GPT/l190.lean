import Mathlib

namespace NUMINAMATH_GPT_diagonal_BD_l190_19042

variables {A B C D : Point}
variables {AB BC BE : ℝ}
variables {parallelogram : ABCD A B C D}

-- Conditions
def side_AB : AB = 3 := sorry
def side_BC : BC = 5 := sorry
def intersection_BE : BE = 9 := sorry

-- Goal 
theorem diagonal_BD : ∀ (BD : ℝ), BD = 34 / 9 :=
by sorry

end NUMINAMATH_GPT_diagonal_BD_l190_19042


namespace NUMINAMATH_GPT_valentina_burger_length_l190_19073

-- Definitions and conditions
def share : ℕ := 6
def total_length (share : ℕ) : ℕ := 2 * share

-- Proof statement
theorem valentina_burger_length : total_length share = 12 := by
  sorry

end NUMINAMATH_GPT_valentina_burger_length_l190_19073


namespace NUMINAMATH_GPT_calculate_fg1_l190_19090

def f (x : ℝ) : ℝ := 4 - 3 * x
def g (x : ℝ) : ℝ := x^3 + 1

theorem calculate_fg1 : f (g 1) = -2 :=
by
  sorry

end NUMINAMATH_GPT_calculate_fg1_l190_19090


namespace NUMINAMATH_GPT_kanul_total_amount_l190_19082

theorem kanul_total_amount (T : ℝ) (R : ℝ) (M : ℝ) (C : ℝ)
  (hR : R = 80000)
  (hM : M = 30000)
  (hC : C = 0.2 * T)
  (hT : T = R + M + C) : T = 137500 :=
by {
  sorry
}

end NUMINAMATH_GPT_kanul_total_amount_l190_19082


namespace NUMINAMATH_GPT_quadratic_real_roots_l190_19050

theorem quadratic_real_roots (m : ℝ) : 
  (∃ x : ℝ, (m - 3) * x^2 - 2 * x + 1 = 0) →
  (m ≤ 4 ∧ m ≠ 3) :=
sorry

end NUMINAMATH_GPT_quadratic_real_roots_l190_19050


namespace NUMINAMATH_GPT_paco_initial_cookies_l190_19044

theorem paco_initial_cookies (x : ℕ) (h : x - 2 + 36 = 2 + 34) : x = 2 :=
by
-- proof steps will be filled in here
sorry

end NUMINAMATH_GPT_paco_initial_cookies_l190_19044


namespace NUMINAMATH_GPT_fill_pipe_half_time_l190_19062

theorem fill_pipe_half_time (T : ℝ) (hT : 0 < T) :
  ∀ t : ℝ, t = T / 2 :=
by
  sorry

end NUMINAMATH_GPT_fill_pipe_half_time_l190_19062


namespace NUMINAMATH_GPT_p_is_necessary_not_sufficient_for_q_l190_19072

  variable (x : ℝ)

  def p := |x| ≤ 2
  def q := 0 ≤ x ∧ x ≤ 2

  theorem p_is_necessary_not_sufficient_for_q : (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬ q x) :=
  by
    sorry
  
end NUMINAMATH_GPT_p_is_necessary_not_sufficient_for_q_l190_19072


namespace NUMINAMATH_GPT_area_diff_l190_19022

-- Defining the side lengths of squares
def side_length_small_square : ℕ := 4
def side_length_large_square : ℕ := 10

-- Calculating the areas
def area_small_square : ℕ := side_length_small_square ^ 2
def area_large_square : ℕ := side_length_large_square ^ 2

-- Theorem statement
theorem area_diff (a_small a_large : ℕ) (h1 : a_small = side_length_small_square ^ 2) (h2 : a_large = side_length_large_square ^ 2) : 
  a_large - a_small = 84 :=
by
  sorry

end NUMINAMATH_GPT_area_diff_l190_19022


namespace NUMINAMATH_GPT_find_s_l190_19067

theorem find_s : ∃ s : ℝ, ⌊s⌋ + s = 18.3 ∧ s = 9.3 :=
by
  sorry

end NUMINAMATH_GPT_find_s_l190_19067


namespace NUMINAMATH_GPT_number_of_students_in_third_group_l190_19057

-- Definitions based on given conditions
def students_group1 : ℕ := 9
def students_group2 : ℕ := 10
def tissues_per_box : ℕ := 40
def total_tissues : ℕ := 1200

-- Define the number of students in the third group as a variable
variable {x : ℕ}

-- Prove that the number of students in the third group is 11
theorem number_of_students_in_third_group (h : 360 + 400 + 40 * x = 1200) : x = 11 :=
by sorry

end NUMINAMATH_GPT_number_of_students_in_third_group_l190_19057


namespace NUMINAMATH_GPT_correct_answer_l190_19029

def total_contestants : Nat := 56
def selected_contestants : Nat := 14

theorem correct_answer :
  (total_contestants = 56) →
  (selected_contestants = 14) →
  (selected_contestants = 14) :=
by
  intro h_total h_selected
  exact h_selected

end NUMINAMATH_GPT_correct_answer_l190_19029


namespace NUMINAMATH_GPT_lisa_eggs_l190_19060

theorem lisa_eggs :
  ∃ x : ℕ, (5 * 52) * (4 * x + 3 + 2) = 3380 ∧ x = 2 :=
by
  sorry

end NUMINAMATH_GPT_lisa_eggs_l190_19060


namespace NUMINAMATH_GPT_martha_total_cost_l190_19056

-- Definitions for the conditions
def amount_cheese_needed : ℝ := 1.5 -- in kg
def amount_meat_needed : ℝ := 0.5 -- in kg
def cost_cheese_per_kg : ℝ := 6.0 -- in dollars per kg
def cost_meat_per_kg : ℝ := 8.0 -- in dollars per kg

-- Total cost that needs to be calculated
def total_cost : ℝ :=
  (amount_cheese_needed * cost_cheese_per_kg) +
  (amount_meat_needed * cost_meat_per_kg)

-- Statement of the theorem
theorem martha_total_cost : total_cost = 13 := by
  sorry

end NUMINAMATH_GPT_martha_total_cost_l190_19056


namespace NUMINAMATH_GPT_mandy_total_shirts_l190_19093

-- Condition definitions
def black_packs : ℕ := 6
def black_shirts_per_pack : ℕ := 7
def yellow_packs : ℕ := 8
def yellow_shirts_per_pack : ℕ := 4

theorem mandy_total_shirts : 
  (black_packs * black_shirts_per_pack + yellow_packs * yellow_shirts_per_pack) = 74 :=
by
  sorry

end NUMINAMATH_GPT_mandy_total_shirts_l190_19093


namespace NUMINAMATH_GPT_geric_initial_bills_l190_19017

theorem geric_initial_bills (G K J : ℕ) 
  (h1: G = 2 * K)
  (h2: K = J - 2)
  (h3: J - 3 = 7) : G = 16 := 
  by 
  sorry

end NUMINAMATH_GPT_geric_initial_bills_l190_19017


namespace NUMINAMATH_GPT_smallest_possible_a_l190_19064

theorem smallest_possible_a (a b c : ℝ) 
  (h1 : (∀ x, y = a * x ^ 2 + b * x + c ↔ y = a * (x + 1/3) ^ 2 + 5/9))
  (h2 : a > 0)
  (h3 : ∃ n : ℤ, a + b + c = n) : 
  a = 1/4 :=
sorry

end NUMINAMATH_GPT_smallest_possible_a_l190_19064


namespace NUMINAMATH_GPT_student_arrangement_l190_19092

theorem student_arrangement :
  let total_arrangements := 720
  let ab_violations := 240
  let cd_violations := 240
  let ab_cd_overlap := 96
  let restricted_arrangements := ab_violations + cd_violations - ab_cd_overlap
  let valid_arrangements := total_arrangements - restricted_arrangements
  valid_arrangements = 336 :=
by
  let total_arrangements := 720
  let ab_violations := 240
  let cd_violations := 240
  let ab_cd_overlap := 96
  let restricted_arrangements := ab_violations + cd_violations - ab_cd_overlap
  let valid_arrangements := total_arrangements - restricted_arrangements
  exact sorry

end NUMINAMATH_GPT_student_arrangement_l190_19092


namespace NUMINAMATH_GPT_remaining_amount_to_be_paid_is_1080_l190_19049

noncomputable def deposit : ℕ := 120
noncomputable def total_price : ℕ := 10 * deposit
noncomputable def remaining_amount : ℕ := total_price - deposit

theorem remaining_amount_to_be_paid_is_1080 :
  remaining_amount = 1080 :=
by
  sorry

end NUMINAMATH_GPT_remaining_amount_to_be_paid_is_1080_l190_19049


namespace NUMINAMATH_GPT_not_necessarily_divisor_sixty_four_l190_19098

theorem not_necessarily_divisor_sixty_four (k : ℤ) (h : (k * (k + 1) * (k + 2)) % 8 = 0) :
  ¬ ((k * (k + 1) * (k + 2)) % 64 = 0) := 
sorry

end NUMINAMATH_GPT_not_necessarily_divisor_sixty_four_l190_19098


namespace NUMINAMATH_GPT_valid_digit_distribution_l190_19080

theorem valid_digit_distribution (n : ℕ) : 
  (∃ (d1 d2 d5 others : ℕ), 
    d1 = n / 2 ∧
    d2 = n / 5 ∧
    d5 = n / 5 ∧
    others = n / 10 ∧
    d1 + d2 + d5 + others = n) :=
by
  sorry

end NUMINAMATH_GPT_valid_digit_distribution_l190_19080


namespace NUMINAMATH_GPT_calculate_new_shipment_bears_l190_19039

theorem calculate_new_shipment_bears 
  (initial_bears : ℕ)
  (shelves : ℕ)
  (bears_per_shelf : ℕ)
  (total_bears_on_shelves : ℕ) 
  (h_total_bears_on_shelves : total_bears_on_shelves = shelves * bears_per_shelf)
  : initial_bears = 6 → shelves = 4 → bears_per_shelf = 6 → total_bears_on_shelves - initial_bears = 18 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3] at *
  simp at *
  sorry

end NUMINAMATH_GPT_calculate_new_shipment_bears_l190_19039


namespace NUMINAMATH_GPT_fish_total_count_l190_19012

theorem fish_total_count :
  let num_fishermen : ℕ := 20
  let fish_caught_per_fisherman : ℕ := 400
  let fish_caught_by_twentieth_fisherman : ℕ := 2400
  (19 * fish_caught_per_fisherman + fish_caught_by_twentieth_fisherman) = 10000 :=
by
  sorry

end NUMINAMATH_GPT_fish_total_count_l190_19012


namespace NUMINAMATH_GPT_a_10_equals_1024_l190_19066

-- Define the sequence a_n and its properties
variable {a : ℕ → ℕ}
variable (h_prop : ∀ p q : ℕ, p > 0 → q > 0 → a (p + q) = a p * a q)
variable (h_a2 : a 2 = 4)

-- Prove the statement that a_10 = 1024 given the above conditions.
theorem a_10_equals_1024 : a 10 = 1024 :=
sorry

end NUMINAMATH_GPT_a_10_equals_1024_l190_19066


namespace NUMINAMATH_GPT_vector_parallel_cos_sin_l190_19046

theorem vector_parallel_cos_sin (θ : ℝ) (a b : ℝ × ℝ) (ha : a = (Real.cos θ, Real.sin θ)) (hb : b = (1, -2)) :
  ∀ (h : ∃ k : ℝ, a = (k * 1, k * (-2))), 
  (Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) = 3 := 
by
  sorry

end NUMINAMATH_GPT_vector_parallel_cos_sin_l190_19046


namespace NUMINAMATH_GPT_sum_of_solutions_l190_19009

theorem sum_of_solutions (y x : ℝ) (h1 : y = 7) (h2 : x^2 + y^2 = 100) : 
  x + -x = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_solutions_l190_19009


namespace NUMINAMATH_GPT_abs_inequality_solution_set_l190_19015

theorem abs_inequality_solution_set {x : ℝ} : (|x - 2| < 1) ↔ (1 < x ∧ x < 3) :=
sorry

end NUMINAMATH_GPT_abs_inequality_solution_set_l190_19015


namespace NUMINAMATH_GPT_average_messages_correct_l190_19020

-- Definitions for the conditions
def messages_monday := 220
def messages_tuesday := 1 / 2 * messages_monday
def messages_wednesday := 50
def messages_thursday := 50
def messages_friday := 50

-- Definition for the total and average messages
def total_messages := messages_monday + messages_tuesday + messages_wednesday + messages_thursday + messages_friday
def average_messages := total_messages / 5

-- Statement to prove
theorem average_messages_correct : average_messages = 96 := 
by sorry

end NUMINAMATH_GPT_average_messages_correct_l190_19020


namespace NUMINAMATH_GPT_bianca_birthday_money_l190_19001

-- Define the conditions
def num_friends : ℕ := 5
def money_per_friend : ℕ := 6

-- State the proof problem
theorem bianca_birthday_money : num_friends * money_per_friend = 30 :=
by
  sorry

end NUMINAMATH_GPT_bianca_birthday_money_l190_19001


namespace NUMINAMATH_GPT_fraction_to_decimal_l190_19031

theorem fraction_to_decimal : (5 / 8 : ℝ) = 0.625 := 
  by sorry

end NUMINAMATH_GPT_fraction_to_decimal_l190_19031


namespace NUMINAMATH_GPT_common_ratio_l190_19095

-- Definitions for the geometric sequence
variables {a_n : ℕ → ℝ} {S_n q : ℝ}

-- Conditions provided in the problem
def condition1 (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) : Prop :=
  S_n 3 = a_n 1 + a_n 2 + a_n 3

def condition2 (a_n : ℕ → ℝ) (S_n : ℝ) : Prop :=
  3 * (a_n 1 + a_n 2 + a_n 3) = a_n 4 - 2

def condition3 (a_n : ℕ → ℝ) (S_n : ℝ) : Prop :=
  3 * (a_n 1 + a_n 2) = a_n 3 - 2

-- The theorem we want to prove
theorem common_ratio (a_n : ℕ → ℝ) (q : ℝ) :
  condition2 a_n S_n ∧ condition3 a_n S_n → q = 4 :=
by
  sorry

end NUMINAMATH_GPT_common_ratio_l190_19095


namespace NUMINAMATH_GPT_selling_price_is_correct_l190_19036

def wholesale_cost : ℝ := 24.35
def gross_profit_percentage : ℝ := 0.15

def gross_profit : ℝ := gross_profit_percentage * wholesale_cost
def selling_price : ℝ := wholesale_cost + gross_profit

theorem selling_price_is_correct :
  selling_price = 28.00 :=
by
  sorry

end NUMINAMATH_GPT_selling_price_is_correct_l190_19036


namespace NUMINAMATH_GPT_evaluate_expression_at_three_l190_19087

theorem evaluate_expression_at_three : 
  (3^2 + 3 * (3^6) = 2196) :=
by
  sorry -- This is where the proof would go

end NUMINAMATH_GPT_evaluate_expression_at_three_l190_19087


namespace NUMINAMATH_GPT_angles_sum_l190_19061

def points_on_circle (A B C R S O : Type) : Prop := sorry

def arc_measure (B R S : Type) (m1 m2 : ℝ) : Prop := sorry

def angle_T (A C B S : Type) (T : ℝ) : Prop := sorry

def angle_U (O C B S : Type) (U : ℝ) : Prop := sorry

theorem angles_sum
  (A B C R S O : Type)
  (h1 : points_on_circle A B C R S O)
  (h2 : arc_measure B R S 48 54)
  (h3 : angle_T A C B S 78)
  (h4 : angle_U O C B S 27) :
  78 + 27 = 105 :=
by sorry

end NUMINAMATH_GPT_angles_sum_l190_19061


namespace NUMINAMATH_GPT_sum_sqrt_inequality_l190_19034

theorem sum_sqrt_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  (3 / 2) * (a + b + c) ≥ (Real.sqrt (a^2 + b * c) + Real.sqrt (b^2 + c * a) + Real.sqrt (c^2 + a * b)) :=
by
  sorry

end NUMINAMATH_GPT_sum_sqrt_inequality_l190_19034


namespace NUMINAMATH_GPT_bookshop_inventory_l190_19026

theorem bookshop_inventory
  (initial_inventory : ℕ := 743)
  (saturday_sales_instore : ℕ := 37)
  (saturday_sales_online : ℕ := 128)
  (sunday_sales_instore : ℕ := 2 * saturday_sales_instore)
  (sunday_sales_online : ℕ := saturday_sales_online + 34)
  (new_shipment : ℕ := 160) :
  (initial_inventory - (saturday_sales_instore + saturday_sales_online + sunday_sales_instore + sunday_sales_online) + new_shipment = 502) :=
by
  sorry

end NUMINAMATH_GPT_bookshop_inventory_l190_19026


namespace NUMINAMATH_GPT_ordered_triple_unique_l190_19091

theorem ordered_triple_unique (a b c : ℝ) (h2 : a > 2) (h3 : b > 2) (h4 : c > 2)
    (h : (a + 3)^2 / (b + c - 3) + (b + 5)^2 / (c + a - 5) + (c + 7)^2 / (a + b - 7) = 49) :
    a = 7 ∧ b = 5 ∧ c = 3 :=
sorry

end NUMINAMATH_GPT_ordered_triple_unique_l190_19091


namespace NUMINAMATH_GPT_xy_value_l190_19043

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 12) : x * y = 12 :=
by sorry

end NUMINAMATH_GPT_xy_value_l190_19043


namespace NUMINAMATH_GPT_trains_meeting_time_l190_19070

noncomputable def kmph_to_mps (speed : ℕ) : ℕ := speed * 1000 / 3600

noncomputable def time_to_meet (L1 L2 D S1 S2 : ℕ) : ℕ := 
  let S1_mps := kmph_to_mps S1
  let S2_mps := kmph_to_mps S2
  let relative_speed := S1_mps + S2_mps
  let total_distance := L1 + L2 + D
  total_distance / relative_speed

theorem trains_meeting_time : time_to_meet 210 120 160 74 92 = 10620 / 1000 :=
by
  sorry

end NUMINAMATH_GPT_trains_meeting_time_l190_19070


namespace NUMINAMATH_GPT_count_zero_expressions_l190_19085

/-- Given four specific vector expressions, prove that exactly two of them evaluate to the zero vector. --/
theorem count_zero_expressions
(AB BC CA MB BO OM AC BD CD OA OC CO : ℝ × ℝ)
(H1 : AB + BC + CA = 0)
(H2 : AB + (MB + BO + OM) ≠ 0)
(H3 : AB - AC + BD - CD = 0)
(H4 : OA + OC + BO + CO ≠ 0) :
  (∃ count, count = 2 ∧
      ((AB + BC + CA = 0) → count = count + 1) ∧
      ((AB + (MB + BO + OM) = 0) → count = count + 1) ∧
      ((AB - AC + BD - CD = 0) → count = count + 1) ∧
      ((OA + OC + BO + CO = 0) → count = count + 1)) :=
sorry

end NUMINAMATH_GPT_count_zero_expressions_l190_19085


namespace NUMINAMATH_GPT_interval_of_monotonic_increase_l190_19054

noncomputable def f (x : ℝ) : ℝ := Real.logb (1/2) (6 + x - x^2)

theorem interval_of_monotonic_increase :
  {x : ℝ | -2 < x ∧ x < 3} → x ∈ Set.Ioc (1/2) 3 :=
by
  sorry

end NUMINAMATH_GPT_interval_of_monotonic_increase_l190_19054


namespace NUMINAMATH_GPT_quadratic_polynomial_correct_l190_19038

noncomputable def q (x : ℝ) : ℝ := (11/10) * x^2 - (21/10) * x + 5

theorem quadratic_polynomial_correct :
  (q (-1) = 4) ∧ (q 2 = 1) ∧ (q 4 = 10) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_quadratic_polynomial_correct_l190_19038


namespace NUMINAMATH_GPT_problem_solution_l190_19045

noncomputable def g (x : ℝ) (P : ℝ) (Q : ℝ) (R : ℝ) : ℝ := x^2 / (P * x^2 + Q * x + R)

theorem problem_solution (P Q R : ℤ) 
  (h1 : ∀ x > 5, g x P Q R > 0.5)
  (h2 : P * (-3)^2 + Q * (-3) + R = 0)
  (h3 : P * 4^2 + Q * 4 + R = 0)
  (h4 : ∃ y : ℝ, y = 1 / P ∧ ∀ x : ℝ, abs (g x P Q R - y) < ε):
  P + Q + R = -24 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l190_19045


namespace NUMINAMATH_GPT_field_fence_length_l190_19086

theorem field_fence_length (L : ℝ) (A : ℝ) (W : ℝ) (fencing : ℝ) (hL : L = 20) (hA : A = 210) (hW : A = L * W) : 
  fencing = 2 * W + L → fencing = 41 :=
by
  rw [hL, hA] at hW
  sorry

end NUMINAMATH_GPT_field_fence_length_l190_19086


namespace NUMINAMATH_GPT_warehouse_painted_area_l190_19059

theorem warehouse_painted_area :
  let length := 8
  let width := 6
  let height := 3.5
  let door_width := 1
  let door_height := 2
  let front_back_area := 2 * (length * height)
  let left_right_area := 2 * (width * height)
  let total_wall_area := front_back_area + left_right_area
  let door_area := door_width * door_height
  let painted_area := total_wall_area - door_area
  painted_area = 96 :=
by
  -- Sorry to skip the actual proof steps
  sorry

end NUMINAMATH_GPT_warehouse_painted_area_l190_19059


namespace NUMINAMATH_GPT_factorize_expr_l190_19032

theorem factorize_expr (a : ℝ) : a^2 - 8 * a = a * (a - 8) :=
sorry

end NUMINAMATH_GPT_factorize_expr_l190_19032


namespace NUMINAMATH_GPT_ratio_of_common_differences_l190_19099

variable (a b d1 d2 : ℝ)

theorem ratio_of_common_differences
  (h1 : a + 4 * d1 = b)
  (h2 : a + 5 * d2 = b) :
  d1 / d2 = 5 / 4 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_common_differences_l190_19099


namespace NUMINAMATH_GPT_find_x_in_triangle_l190_19008

theorem find_x_in_triangle (y z : ℝ) (cos_Y_minus_Z : ℝ) (h1 : y = 7) (h2 : z = 6) (h3 : cos_Y_minus_Z = 1 / 2) : 
    ∃ x : ℝ, x = Real.sqrt 73 :=
by
  existsi Real.sqrt 73
  sorry

end NUMINAMATH_GPT_find_x_in_triangle_l190_19008


namespace NUMINAMATH_GPT_deductive_reasoning_l190_19089

theorem deductive_reasoning (
  deductive_reasoning_form : Prop
): ¬(deductive_reasoning_form → true → correct_conclusion) :=
by sorry

end NUMINAMATH_GPT_deductive_reasoning_l190_19089


namespace NUMINAMATH_GPT_cost_equivalence_l190_19088

theorem cost_equivalence (b a p : ℕ) (h1 : 4 * b = 3 * a) (h2 : 9 * a = 6 * p) : 24 * b = 12 * p :=
  sorry

end NUMINAMATH_GPT_cost_equivalence_l190_19088


namespace NUMINAMATH_GPT_find_f_neg_one_l190_19078

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem find_f_neg_one (f : ℝ → ℝ) (h_odd : is_odd f)
(h_pos : ∀ x, 0 < x → f x = x^2 + 1/x) : f (-1) = -2 := 
sorry

end NUMINAMATH_GPT_find_f_neg_one_l190_19078


namespace NUMINAMATH_GPT_units_digit_base_6_l190_19005

theorem units_digit_base_6 (n m : ℕ) (h₁ : n = 312) (h₂ : m = 67) : (312 * 67) % 6 = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_units_digit_base_6_l190_19005


namespace NUMINAMATH_GPT_cost_equality_store_comparison_for_10_l190_19033

-- price definitions
def teapot_price := 30
def teacup_price := 5
def teapot_count := 5

-- store A and B promotional conditions
def storeA_cost (x : Nat) : Real := 5 * x + 125
def storeB_cost (x : Nat) : Real := 4.5 * x + 135

theorem cost_equality (x : Nat) (h : x > 5) :
  storeA_cost x = storeB_cost x → x = 20 := by
  sorry

theorem store_comparison_for_10 (x : Nat) (h : x = 10) :
  storeA_cost x < storeB_cost x := by
  sorry

end NUMINAMATH_GPT_cost_equality_store_comparison_for_10_l190_19033


namespace NUMINAMATH_GPT_find_r_l190_19041

variable (m r : ℝ)

theorem find_r (h1 : 5 = m * 3^r) (h2 : 45 = m * 9^(2 * r)) : r = 2 / 3 := by
  sorry

end NUMINAMATH_GPT_find_r_l190_19041


namespace NUMINAMATH_GPT_parabola_tangent_line_l190_19013

theorem parabola_tangent_line (a b : ℝ) :
  (∀ x : ℝ, ax^2 + b * x + 2 = 2 * x + 3 → a = -1 ∧ b = 4) :=
sorry

end NUMINAMATH_GPT_parabola_tangent_line_l190_19013


namespace NUMINAMATH_GPT_problem_statement_l190_19068

-- Definitions of sets S and P
def S : Set ℝ := {x | x^2 - 3 * x - 10 < 0}
def P (a : ℝ) : Set ℝ := {x | a + 1 < x ∧ x < 2 * a + 15}

-- Proof statement
theorem problem_statement (a : ℝ) : 
  (S = {x | -2 < x ∧ x < 5}) ∧ (S ⊆ P a → a ∈ Set.Icc (-5 : ℝ) (-3 : ℝ)) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l190_19068


namespace NUMINAMATH_GPT_ghost_enter_exit_ways_l190_19000

theorem ghost_enter_exit_ways : 
  (∃ (enter_win : ℕ) (exit_win : ℕ), enter_win ≠ exit_win ∧ 1 ≤ enter_win ∧ enter_win ≤ 8 ∧ 1 ≤ exit_win ∧ exit_win ≤ 8) →
  ∃ (ways : ℕ), ways = 8 * 7 :=
by
  sorry

end NUMINAMATH_GPT_ghost_enter_exit_ways_l190_19000


namespace NUMINAMATH_GPT_circle_eq_center_tangent_l190_19004

theorem circle_eq_center_tangent (x y : ℝ) : 
  let center := (5, 4)
  let radius := 4
  (x - center.1) ^ 2 + (y - center.2) ^ 2 = radius ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_circle_eq_center_tangent_l190_19004


namespace NUMINAMATH_GPT_infinite_series_sum_eq_33_div_8_l190_19051

noncomputable def infinite_series_sum: ℝ :=
  ∑' n: ℕ, n^3 / (3^n : ℝ)

theorem infinite_series_sum_eq_33_div_8:
  infinite_series_sum = 33 / 8 :=
sorry

end NUMINAMATH_GPT_infinite_series_sum_eq_33_div_8_l190_19051


namespace NUMINAMATH_GPT_rectangle_area_same_width_l190_19084

theorem rectangle_area_same_width
  (square_area : ℝ) (area_eq : square_area = 36)
  (rect_width_eq_side : ℝ → ℝ → Prop) (width_eq : ∀ s, rect_width_eq_side s s)
  (rect_length_eq_3_times_width : ℝ → ℝ → Prop) (length_eq : ∀ w, rect_length_eq_3_times_width w (3 * w)) :
  (∃ s l w, s = 6 ∧ w = s ∧ l = 3 * w ∧ square_area = s * s ∧ rect_width_eq_side w s ∧ rect_length_eq_3_times_width w l ∧ w * l = 108) :=
by {
  sorry
}

end NUMINAMATH_GPT_rectangle_area_same_width_l190_19084


namespace NUMINAMATH_GPT_smallest_n_for_geometric_sequence_divisibility_l190_19047

theorem smallest_n_for_geometric_sequence_divisibility :
  ∃ n : ℕ, (∀ m : ℕ, m < n → ¬ (2 * 10 ^ 6 ∣ (30 ^ (m - 1) * (5 / 6)))) ∧ (2 * 10 ^ 6 ∣ (30 ^ (n - 1) * (5 / 6))) ∧ n = 8 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_for_geometric_sequence_divisibility_l190_19047


namespace NUMINAMATH_GPT_solve_cubic_diophantine_l190_19007

theorem solve_cubic_diophantine :
  (∃ x y z : ℤ, x^3 + y^3 + z^3 - 3 * x * y * z = 2003) ↔ 
  (x = 667 ∧ y = 668 ∧ z = 668) ∨ 
  (x = 668 ∧ y = 667 ∧ z = 668) ∨ 
  (x = 668 ∧ y = 668 ∧ z = 667) :=
sorry

end NUMINAMATH_GPT_solve_cubic_diophantine_l190_19007


namespace NUMINAMATH_GPT_number_of_solutions_l190_19002

theorem number_of_solutions :
  ∃ (sols : Finset ℝ), 
    (∀ x, x ∈ sols → 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ 2 * (Real.sin x)^3 - 5 * (Real.sin x)^2 + 2 * Real.sin x = 0) 
    ∧ Finset.card sols = 5 := 
by
  sorry

end NUMINAMATH_GPT_number_of_solutions_l190_19002


namespace NUMINAMATH_GPT_product_of_roots_l190_19097

theorem product_of_roots :
  ∀ a b c : ℚ, (a ≠ 0) → a = 24 → b = 60 → c = -600 → (c / a) = -25 :=
sorry

end NUMINAMATH_GPT_product_of_roots_l190_19097


namespace NUMINAMATH_GPT_factorize_quadratic_trinomial_l190_19071

theorem factorize_quadratic_trinomial (t : ℝ) : t^2 - 10 * t + 25 = (t - 5)^2 :=
by
  sorry

end NUMINAMATH_GPT_factorize_quadratic_trinomial_l190_19071


namespace NUMINAMATH_GPT_find_y_of_rectangle_area_l190_19055

theorem find_y_of_rectangle_area (y : ℝ) (h1 : y > 0) 
(h2 : (0, 0) = (0, 0)) (h3 : (0, 6) = (0, 6)) 
(h4 : (y, 6) = (y, 6)) (h5 : (y, 0) = (y, 0)) 
(h6 : 6 * y = 42) : y = 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_y_of_rectangle_area_l190_19055


namespace NUMINAMATH_GPT_solve_equation_l190_19021

-- Definitions for the variables and the main equation
def equation (x y z : ℤ) : Prop :=
  5 * x^2 + y^2 + 3 * z^2 - 2 * y * z = 30

-- The statement that needs to be proved
theorem solve_equation (x y z : ℤ) :
  equation x y z ↔ (x, y, z) = (1, 5, 0) ∨ (x, y, z) = (1, -5, 0) ∨ (x, y, z) = (-1, 5, 0) ∨ (x, y, z) = (-1, -5, 0) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l190_19021


namespace NUMINAMATH_GPT_speed_of_train_A_l190_19048

noncomputable def train_speed_A (V_B : ℝ) (T_A T_B : ℝ) : ℝ :=
  (T_B / T_A) * V_B

theorem speed_of_train_A : train_speed_A 165 9 4 = 73.33 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_train_A_l190_19048


namespace NUMINAMATH_GPT_temperature_equivalence_l190_19081

theorem temperature_equivalence (x : ℝ) (h : x = (9 / 5) * x + 32) : x = -40 :=
sorry

end NUMINAMATH_GPT_temperature_equivalence_l190_19081


namespace NUMINAMATH_GPT_minimum_a_l190_19035

noncomputable def f (x : ℝ) : ℝ := abs x + abs (x - 1)
noncomputable def g (x a : ℝ) : ℝ := f x - a

theorem minimum_a (a : ℝ) : (∃ x : ℝ, g x a = 0) ↔ (a ≥ 1) :=
by sorry

end NUMINAMATH_GPT_minimum_a_l190_19035


namespace NUMINAMATH_GPT_angies_monthly_salary_l190_19083

theorem angies_monthly_salary 
    (necessities_expense : ℕ)
    (taxes_expense : ℕ)
    (left_over : ℕ)
    (monthly_salary : ℕ) :
  necessities_expense = 42 → 
  taxes_expense = 20 → 
  left_over = 18 → 
  monthly_salary = necessities_expense + taxes_expense + left_over → 
  monthly_salary = 80 :=
by 
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end NUMINAMATH_GPT_angies_monthly_salary_l190_19083


namespace NUMINAMATH_GPT_sum_of_digits_of_smallest_number_l190_19077

noncomputable def sum_of_digits (n : Nat) : Nat :=
  n.digits 10 |>.foldl (· + ·) 0

theorem sum_of_digits_of_smallest_number :
  (n : Nat) → (h1 : (Nat.ceil (n / 2) - Nat.ceil (n / 3) = 15)) → 
  sum_of_digits n = 9 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_of_smallest_number_l190_19077


namespace NUMINAMATH_GPT_career_preference_degrees_l190_19025

theorem career_preference_degrees (boys girls : ℕ) (ratio_boys_to_girls : boys / gcd boys girls = 2 ∧ girls / gcd boys girls = 3) 
  (boys_preference : ℕ) (girls_preference : ℕ) 
  (h1 : boys_preference = boys / 3)
  (h2 : girls_preference = 2 * girls / 3) : 
  (boys_preference + girls_preference) / (boys + girls) * 360 = 192 :=
by
  sorry

end NUMINAMATH_GPT_career_preference_degrees_l190_19025


namespace NUMINAMATH_GPT_area_increase_300_percent_l190_19074

noncomputable def percentage_increase_of_area (d : ℝ) : ℝ :=
  let d' := 2 * d
  let r := d / 2
  let r' := d' / 2
  let A := Real.pi * r^2
  let A' := Real.pi * (r')^2
  100 * (A' - A) / A

theorem area_increase_300_percent (d : ℝ) : percentage_increase_of_area d = 300 :=
by
  sorry

end NUMINAMATH_GPT_area_increase_300_percent_l190_19074


namespace NUMINAMATH_GPT_phase_shift_cosine_l190_19058

theorem phase_shift_cosine (x : ℝ) : 2 * x + (Real.pi / 2) = 0 → x = - (Real.pi / 4) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_phase_shift_cosine_l190_19058


namespace NUMINAMATH_GPT_n_minus_m_l190_19096

variable (m n : ℕ)

def is_congruent_to_5_mod_13 (x : ℕ) : Prop := x % 13 = 5
def is_smallest_three_digit_integer_congruent_to_5_mod_13 (x : ℕ) : Prop :=
  is_congruent_to_5_mod_13 x ∧ x ≥ 100 ∧ ∀ y, is_congruent_to_5_mod_13 y → y ≥ 100 → x ≤ y

def is_smallest_four_digit_integer_congruent_to_5_mod_13 (x : ℕ) : Prop :=
  is_congruent_to_5_mod_13 x ∧ x ≥ 1000 ∧ ∀ y, is_congruent_to_5_mod_13 y → y ≥ 1000 → x ≤ y

theorem n_minus_m
  (h₁ : is_smallest_three_digit_integer_congruent_to_5_mod_13 m)
  (h₂ : is_smallest_four_digit_integer_congruent_to_5_mod_13 n) :
  n - m = 897 := sorry

end NUMINAMATH_GPT_n_minus_m_l190_19096


namespace NUMINAMATH_GPT_sin_alpha_through_point_l190_19076

theorem sin_alpha_through_point (α : ℝ) (x y : ℝ) (h : x = -1 ∧ y = 2) (r : ℝ) (h_r : r = Real.sqrt (x^2 + y^2)) :
  Real.sin α = 2 * Real.sqrt 5 / 5 :=
by
  sorry

end NUMINAMATH_GPT_sin_alpha_through_point_l190_19076


namespace NUMINAMATH_GPT_triangle_equilateral_of_equal_angle_ratios_l190_19053

theorem triangle_equilateral_of_equal_angle_ratios
  (a b c : ℝ)
  (h₁ : a + b + c = 180)
  (h₂ : a = b)
  (h₃ : b = c) :
  a = 60 ∧ b = 60 ∧ c = 60 :=
by
  sorry

end NUMINAMATH_GPT_triangle_equilateral_of_equal_angle_ratios_l190_19053


namespace NUMINAMATH_GPT_overlap_per_connection_is_4_cm_l190_19016

-- Condition 1: There are 24 tape measures.
def number_of_tape_measures : Nat := 24

-- Condition 2: Each tape measure is 28 cm long.
def length_of_one_tape_measure : Nat := 28

-- Condition 3: The total length of all connected tape measures is 580 cm.
def total_length_with_overlaps : Nat := 580

-- The question to prove: The overlap per connection is 4 cm.
theorem overlap_per_connection_is_4_cm 
  (n : Nat) (length_one : Nat) (total_length : Nat) 
  (h_n : n = number_of_tape_measures)
  (h_length_one : length_one = length_of_one_tape_measure)
  (h_total_length : total_length = total_length_with_overlaps) :
  ((n * length_one - total_length) / (n - 1)) = 4 := 
by 
  sorry

end NUMINAMATH_GPT_overlap_per_connection_is_4_cm_l190_19016


namespace NUMINAMATH_GPT_find_real_number_x_l190_19027

theorem find_real_number_x 
    (x : ℝ) 
    (i : ℂ) 
    (h_imaginary_unit : i*i = -1) 
    (h_equation : (1 - 2*i)*(x + i) = 4 - 3*i) : 
    x = 2 := 
by
  sorry

end NUMINAMATH_GPT_find_real_number_x_l190_19027


namespace NUMINAMATH_GPT_dropped_score_l190_19065

variable (A B C D : ℕ)

theorem dropped_score (h1 : A + B + C + D = 180) (h2 : A + B + C = 150) : D = 30 := by
  sorry

end NUMINAMATH_GPT_dropped_score_l190_19065


namespace NUMINAMATH_GPT_regular_polygon_sides_l190_19063

noncomputable def interiorAngle (n : ℕ) : ℝ :=
  if n ≥ 3 then (180 * (n - 2) / n) else 0

noncomputable def exteriorAngle (n : ℕ) : ℝ :=
  180 - interiorAngle n

theorem regular_polygon_sides (n : ℕ) (h : interiorAngle n = 160) : n = 18 :=
by sorry

end NUMINAMATH_GPT_regular_polygon_sides_l190_19063


namespace NUMINAMATH_GPT_Amy_initial_cupcakes_l190_19018

def initialCupcakes (packages : ℕ) (cupcakesPerPackage : ℕ) (eaten : ℕ) : ℕ :=
  packages * cupcakesPerPackage + eaten

theorem Amy_initial_cupcakes :
  let packages := 9
  let cupcakesPerPackage := 5
  let eaten := 5
  initialCupcakes packages cupcakesPerPackage eaten = 50 :=
by
  sorry

end NUMINAMATH_GPT_Amy_initial_cupcakes_l190_19018


namespace NUMINAMATH_GPT_score_difference_l190_19011

-- Definitions of the given conditions
def Layla_points : ℕ := 70
def Total_points : ℕ := 112

-- The statement to be proven
theorem score_difference : (Layla_points - (Total_points - Layla_points)) = 28 :=
by sorry

end NUMINAMATH_GPT_score_difference_l190_19011


namespace NUMINAMATH_GPT_average_trees_planted_l190_19030

def A := 225
def B := A + 48
def C := A - 24
def total_trees := A + B + C
def average := total_trees / 3

theorem average_trees_planted :
  average = 233 := by
  sorry

end NUMINAMATH_GPT_average_trees_planted_l190_19030


namespace NUMINAMATH_GPT_choose_blue_pair_l190_19075

/-- In a drawer, there are 12 distinguishable socks: 5 white, 3 brown, and 4 blue socks.
    Prove that the number of ways to choose a pair of socks such that both socks are blue is 6. -/
theorem choose_blue_pair (total_socks white_socks brown_socks blue_socks : ℕ)
  (h_total : total_socks = 12) (h_white : white_socks = 5) (h_brown : brown_socks = 3) (h_blue : blue_socks = 4) :
  (blue_socks.choose 2) = 6 :=
by
  sorry

end NUMINAMATH_GPT_choose_blue_pair_l190_19075


namespace NUMINAMATH_GPT_range_m_graph_in_quadrants_l190_19069

theorem range_m_graph_in_quadrants (m : ℝ) :
  (∀ x : ℝ, x ≠ 0 → ((x > 0 → (m + 2) / x > 0) ∧ (x < 0 → (m + 2) / x < 0))) ↔ m > -2 :=
by 
  sorry

end NUMINAMATH_GPT_range_m_graph_in_quadrants_l190_19069


namespace NUMINAMATH_GPT_arithmetic_sequence_a2a3_l190_19023

noncomputable def arithmetic_sequence_sum (a : Nat → ℝ) (d : ℝ) : Prop :=
  ∀ n : Nat, a (n + 1) = a n + d

theorem arithmetic_sequence_a2a3 
  (a : Nat → ℝ) (d : ℝ) 
  (arith_seq : arithmetic_sequence_sum a d)
  (H : a 1 + a 2 + a 3 + a 4 = 30) : 
  a 2 + a 3 = 15 :=
by 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_a2a3_l190_19023


namespace NUMINAMATH_GPT_number_of_weavers_l190_19079

theorem number_of_weavers (W : ℕ) 
  (h1 : ∀ t : ℕ, t = 4 → 4 = W * (1 * t)) 
  (h2 : ∀ t : ℕ, t = 16 → 64 = 16 * (1 / (W:ℝ) * t)) : 
  W = 4 := 
by {
  sorry
}

end NUMINAMATH_GPT_number_of_weavers_l190_19079


namespace NUMINAMATH_GPT_cinema_chairs_l190_19037

theorem cinema_chairs (chairs_between : ℕ) (h : chairs_between = 30) :
  chairs_between + 2 = 32 := by
  sorry

end NUMINAMATH_GPT_cinema_chairs_l190_19037


namespace NUMINAMATH_GPT_intersection_M_S_l190_19010

def M := {x : ℕ | 0 < x ∧ x < 4 }

def S : Set ℕ := {2, 3, 5}

theorem intersection_M_S : (M ∩ S) = {2, 3} := by
  sorry

end NUMINAMATH_GPT_intersection_M_S_l190_19010


namespace NUMINAMATH_GPT_javier_first_throw_distance_l190_19052

noncomputable def javelin_first_throw_initial_distance (x : Real) : Real :=
  let throw1_adjusted := 2 * x * 0.95 - 2
  let throw2_adjusted := x * 0.92 - 4
  let throw3_adjusted := 4 * x - 1
  if (throw1_adjusted + throw2_adjusted + throw3_adjusted = 1050) then
    2 * x
  else
    0

theorem javier_first_throw_distance : ∃ x : Real, javelin_first_throw_initial_distance x = 310 :=
by
  sorry

end NUMINAMATH_GPT_javier_first_throw_distance_l190_19052


namespace NUMINAMATH_GPT_solution_set_of_inequality_l190_19006

theorem solution_set_of_inequality (x : ℝ) : (x-50)*(60-x) > 0 ↔ 50 < x ∧ x < 60 :=
by sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l190_19006


namespace NUMINAMATH_GPT_common_ratio_is_two_l190_19028

-- Geometric sequence definition
noncomputable def common_ratio (n : ℕ) (a : ℕ → ℝ) : ℝ :=
a 2 / a 1

-- The sequence has 10 terms
def ten_terms (a : ℕ → ℝ) : Prop :=
∀ n, 1 ≤ n ∧ n ≤ 10

-- The product of the odd terms is 2
def product_of_odd_terms (a : ℕ → ℝ) : Prop :=
(a 1) * (a 3) * (a 5) * (a 7) * (a 9) = 2

-- The product of the even terms is 64
def product_of_even_terms (a : ℕ → ℝ) : Prop :=
(a 2) * (a 4) * (a 6) * (a 8) * (a 10) = 64

-- The problem statement to prove that the common ratio q is 2
theorem common_ratio_is_two (a : ℕ → ℝ) (q : ℝ) (h1 : ten_terms a) 
(h2 : product_of_odd_terms a) (h3 : product_of_even_terms a) : q = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_common_ratio_is_two_l190_19028


namespace NUMINAMATH_GPT_probability_x_gt_9y_in_rectangle_l190_19024

theorem probability_x_gt_9y_in_rectangle :
  let a := 1007
  let b := 1008
  let area_triangle := (a * a / 18 : ℚ)
  let area_rectangle := (a * b : ℚ)
  area_triangle / area_rectangle = (1 : ℚ) / 18 :=
by
  sorry

end NUMINAMATH_GPT_probability_x_gt_9y_in_rectangle_l190_19024


namespace NUMINAMATH_GPT_graph_passes_through_point_l190_19003

theorem graph_passes_through_point (a : ℝ) (h : a < 0) : (0, 0) ∈ {p : ℝ × ℝ | ∃ x : ℝ, p = (x, (1 - a)^x - 1)} :=
by
  sorry

end NUMINAMATH_GPT_graph_passes_through_point_l190_19003


namespace NUMINAMATH_GPT_find_f_2008_l190_19094

noncomputable def f (x : ℝ) : ℝ := Real.cos x

noncomputable def f_n (n : ℕ) : (ℝ → ℝ) :=
match n with
| 0     => f
| (n+1) => (deriv (f_n n))

theorem find_f_2008 (x : ℝ) : (f_n 2008) x = Real.cos x := by
  sorry

end NUMINAMATH_GPT_find_f_2008_l190_19094


namespace NUMINAMATH_GPT_largest_digit_B_divisible_by_4_l190_19014

theorem largest_digit_B_divisible_by_4 :
  ∃ B : ℕ, B = 9 ∧ ∀ k : ℕ, (k ≤ 9 → (∃ n : ℕ, 4 * n = 10 * B + 792 % 100)) :=
by
  sorry

end NUMINAMATH_GPT_largest_digit_B_divisible_by_4_l190_19014


namespace NUMINAMATH_GPT_winning_majority_vote_l190_19019

def total_votes : ℕ := 600

def winning_percentage : ℝ := 0.70

def losing_percentage : ℝ := 0.30

theorem winning_majority_vote : (0.70 * (total_votes : ℝ) - 0.30 * (total_votes : ℝ)) = 240 := 
by
  sorry

end NUMINAMATH_GPT_winning_majority_vote_l190_19019


namespace NUMINAMATH_GPT_sum_of_cubes_zero_l190_19040

variables {a b c : ℝ}

theorem sum_of_cubes_zero (h₁ : a + b + c = 0) (h₂ : a^2 + b^2 + c^2 = a^4 + b^4 + c^4) : a^3 + b^3 + c^3 = 0 :=
sorry

end NUMINAMATH_GPT_sum_of_cubes_zero_l190_19040
