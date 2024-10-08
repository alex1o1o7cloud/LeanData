import Mathlib

namespace sum_of_positive_integer_factors_of_24_l174_174935

-- Define the number 24
def n : ℕ := 24

-- Define the list of positive factors of 24
def pos_factors_of_24 : List ℕ := [1, 2, 4, 8, 3, 6, 12, 24]

-- Define the sum of the factors
def sum_of_factors : ℕ := pos_factors_of_24.sum

-- The theorem statement
theorem sum_of_positive_integer_factors_of_24 : sum_of_factors = 60 := by
  sorry

end sum_of_positive_integer_factors_of_24_l174_174935


namespace eq_x2_inv_x2_and_x8_inv_x8_l174_174226

theorem eq_x2_inv_x2_and_x8_inv_x8 (x : ℝ) 
  (h : 47 = x^4 + 1 / x^4) : 
  (x^2 + 1 / x^2 = 7) ∧ (x^8 + 1 / x^8 = -433) :=
by
  sorry

end eq_x2_inv_x2_and_x8_inv_x8_l174_174226


namespace perimeter_equal_l174_174398

theorem perimeter_equal (x : ℕ) (hx : x = 4)
    (side_square : ℕ := x + 2) 
    (side_triangle : ℕ := 2 * x) 
    (perimeter_square : ℕ := 4 * side_square)
    (perimeter_triangle : ℕ := 3 * side_triangle) :
    perimeter_square = perimeter_triangle :=
by
    -- Given x = 4
    -- Calculate side lengths
    -- side_square = x + 2 = 4 + 2 = 6
    -- side_triangle = 2 * x = 2 * 4 = 8
    -- Calculate perimeters
    -- perimeter_square = 4 * side_square = 4 * 6 = 24
    -- perimeter_triangle = 3 * side_triangle = 3 * 8 = 24
    -- Therefore, perimeter_square = perimeter_triangle = 24
    sorry

end perimeter_equal_l174_174398


namespace triangle_area_range_l174_174027

theorem triangle_area_range (A B C : ℝ) (a b c : ℝ) 
  (h1 : a * Real.sin B = Real.sqrt 3 * b * Real.cos A)
  (h2 : a = 3) :
  0 < (1 / 2) * b * c * Real.sin A ∧ 
  (1 / 2) * b * c * Real.sin A ≤ (9 * Real.sqrt 3) / 4 := 
  sorry

end triangle_area_range_l174_174027


namespace possible_values_of_polynomial_l174_174948

theorem possible_values_of_polynomial (x : ℝ) (h : x^2 - 7 * x + 12 < 0) : 
48 < x^2 + 7 * x + 12 ∧ x^2 + 7 * x + 12 < 64 :=
sorry

end possible_values_of_polynomial_l174_174948


namespace number_of_12_digit_numbers_with_consecutive_digits_same_l174_174568

theorem number_of_12_digit_numbers_with_consecutive_digits_same : 
  let total := (2 : ℕ) ^ 12
  let excluded := 2
  total - excluded = 4094 :=
by
  let total := (2 : ℕ) ^ 12
  let excluded := 2
  have h : total = 4096 := by norm_num
  have h' : total - excluded = 4094 := by norm_num
  exact h'

end number_of_12_digit_numbers_with_consecutive_digits_same_l174_174568


namespace total_students_in_class_l174_174186

theorem total_students_in_class (B G : ℕ) (h1 : G = 160) (h2 : 5 * G = 8 * B) : B + G = 260 :=
by
  -- Proof steps would go here
  sorry

end total_students_in_class_l174_174186


namespace value_of_f_10_l174_174613

def f (n : ℕ) : ℕ := n^2 - n + 17

theorem value_of_f_10 : f 10 = 107 := by
  sorry

end value_of_f_10_l174_174613


namespace Wendy_earned_45_points_l174_174660

-- Definitions for the conditions
def points_per_bag : Nat := 5
def total_bags : Nat := 11
def unrecycled_bags : Nat := 2

-- The variable for recycled bags and total points earned
def recycled_bags := total_bags - unrecycled_bags
def total_points := recycled_bags * points_per_bag

theorem Wendy_earned_45_points : total_points = 45 :=
by
  -- Proof goes here
  sorry

end Wendy_earned_45_points_l174_174660


namespace parabola_line_intersection_l174_174752

/-- 
Given a parabola \( y^2 = 2x \), a line passing through the focus of 
the parabola intersects the parabola at points \( A \) and \( B \) where 
the sum of the x-coordinates of \( A \) and \( B \) is equal to 2. 
Prove that such a line exists and there are exactly 3 such lines.
--/
theorem parabola_line_intersection :
  ∃ l₁ l₂ l₃ : (ℝ × ℝ) → (ℝ × ℝ), 
    (∀ p, l₁ p = l₂ p ∧ l₁ p = l₃ p → false) ∧
    ∀ (A B : ℝ × ℝ), 
      (A.2 ^ 2 = 2 * A.1) ∧ 
      (B.2 ^ 2 = 2 * B.1) ∧ 
      (A.1 + B.1 = 2) →
      (∃ k : ℝ, 
        ∀ (x : ℝ), 
          ((A.2 = k * (A.1 - 1)) ∧ (B.2 = k * (B.1 - 1))) ∧ 
          (k * (A.1 - 1) = k * (B.1 - 1)) ∧ 
          (k ≠ 0)) :=
sorry

end parabola_line_intersection_l174_174752


namespace problem_statement_l174_174799

variable {R : Type*} [LinearOrderedField R]

def is_even_function (f : R → R) : Prop := ∀ x : R, f x = f (-x)

theorem problem_statement (f : R → R)
  (h1 : is_even_function f)
  (h2 : ∀ x1 x2 : R, x1 ≤ -1 → x2 ≤ -1 → (x2 - x1) * (f x2 - f x1) < 0) :
  f (-1) < f (-3 / 2) ∧ f (-3 / 2) < f 2 :=
sorry

end problem_statement_l174_174799


namespace new_rope_length_l174_174188

-- Define the given constants and conditions
def rope_length_initial : ℝ := 12
def additional_area : ℝ := 1511.7142857142858
noncomputable def pi_approx : ℝ := Real.pi

-- Define the proof statement
theorem new_rope_length :
  let r2 := Real.sqrt ((additional_area / pi_approx) + rope_length_initial ^ 2)
  r2 = 25 :=
by
  -- Placeholder for the proof
  sorry

end new_rope_length_l174_174188


namespace find_divisor_l174_174113

theorem find_divisor (D N : ℕ) (h₁ : N = 265) (h₂ : N / D + 8 = 61) : D = 5 :=
by
  sorry

end find_divisor_l174_174113


namespace xiao_ming_should_choose_store_A_l174_174683

def storeB_cost (x : ℕ) : ℝ := 0.85 * x

def storeA_cost (x : ℕ) : ℝ :=
  if x ≤ 10 then x
  else 0.7 * x + 3

theorem xiao_ming_should_choose_store_A (x : ℕ) (h : x = 22) :
  storeA_cost x < storeB_cost x := by
  sorry

end xiao_ming_should_choose_store_A_l174_174683


namespace final_center_coordinates_l174_174887

-- Definition of the initial condition: the center of Circle U
def center_initial : ℝ × ℝ := (3, -4)

-- Definition of the reflection function across the y-axis
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

-- Definition of the translation function to translate a point 5 units up
def translate_up_5 (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, p.2 + 5)

-- Defining the final coordinates after reflection and translation
def center_final : ℝ × ℝ :=
  translate_up_5 (reflect_y_axis center_initial)

-- Problem statement: Prove that the final center coordinates are (-3, 1)
theorem final_center_coordinates :
  center_final = (-3, 1) :=
by {
  -- Skipping the proof itself, but the theorem statement should be equivalent
  sorry
}

end final_center_coordinates_l174_174887


namespace sin_cos_identity_l174_174001

theorem sin_cos_identity (a : ℝ) (h : Real.sin (π - a) = -2 * Real.sin (π / 2 + a)) : 
  Real.sin a * Real.cos a = -2 / 5 :=
by
  sorry

end sin_cos_identity_l174_174001


namespace can_lid_boxes_count_l174_174396

theorem can_lid_boxes_count 
  (x y : ℕ) 
  (h1 : 3 * x + y + 14 = 75) : 
  x = 20 ∧ y = 1 :=
by 
  sorry

end can_lid_boxes_count_l174_174396


namespace exist_positive_real_x_l174_174024

theorem exist_positive_real_x (x : ℝ) (hx1 : 0 < x) (hx2 : Nat.floor x * x = 90) : x = 10 := 
sorry

end exist_positive_real_x_l174_174024


namespace find_a3_a4_a5_l174_174467

open Real

variables {a : ℕ → ℝ} (q : ℝ)

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

noncomputable def a_1 : ℝ := 3

def sum_of_first_three (a : ℕ → ℝ) : Prop :=
  a 0 + a 1 + a 2 = 21

def all_terms_positive (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, 0 < a n

theorem find_a3_a4_a5 (h1 : is_geometric_sequence a) (h2 : a 0 = a_1) (h3 : sum_of_first_three a) (h4 : all_terms_positive a) :
  a 2 + a 3 + a 4 = 84 :=
sorry

end find_a3_a4_a5_l174_174467


namespace minimum_value_of_expression_l174_174723

theorem minimum_value_of_expression (x : ℝ) (hx : x > 0) :
  2 * x + 1 / x^6 ≥ 3 :=
sorry

end minimum_value_of_expression_l174_174723


namespace calculate_heartsuit_ratio_l174_174858

def heartsuit (n m : ℕ) : ℕ := n^2 * m^3

theorem calculate_heartsuit_ratio :
  (heartsuit 3 5) / (heartsuit 5 3) = 5 / 3 :=
by sorry

end calculate_heartsuit_ratio_l174_174858


namespace loss_percentage_l174_174633

theorem loss_percentage (cost_price selling_price : ℝ) (h_cost : cost_price = 1500) (h_sell : selling_price = 1260) : 
  (cost_price - selling_price) / cost_price * 100 = 16 := 
by
  sorry

end loss_percentage_l174_174633


namespace real_solutions_l174_174446

theorem real_solutions (x : ℝ) :
  (1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) + 1 / ((x - 5) * (x - 6)) = 1 / 12) ↔ (x = 12 ∨ x = -4) :=
by
  sorry

end real_solutions_l174_174446


namespace confidence_k_squared_l174_174975

-- Define the condition for 95% confidence relation between events A and B
def confidence_95 (A B : Prop) : Prop := 
  -- Placeholder for the actual definition, assume 95% confidence implies a specific condition
  True

-- Define the data value and critical value condition
def K_squared : ℝ := sorry  -- Placeholder for the actual K² value

theorem confidence_k_squared (A B : Prop) (h : confidence_95 A B) : K_squared > 3.841 := 
by
  sorry  -- Proof is not required, only the statement

end confidence_k_squared_l174_174975


namespace percentage_reduction_in_price_l174_174814

noncomputable def original_price_per_mango : ℝ := 416.67 / 125

noncomputable def original_num_mangoes : ℝ := 360 / original_price_per_mango

def additional_mangoes : ℝ := 12

noncomputable def new_num_mangoes : ℝ := original_num_mangoes + additional_mangoes

noncomputable def new_price_per_mango : ℝ := 360 / new_num_mangoes

noncomputable def percentage_reduction : ℝ := (original_price_per_mango - new_price_per_mango) / original_price_per_mango * 100

theorem percentage_reduction_in_price : percentage_reduction = 10 := by
  sorry

end percentage_reduction_in_price_l174_174814


namespace base_conversion_problem_l174_174982

theorem base_conversion_problem (b : ℕ) (h : b^2 + b + 3 = 34) : b = 6 :=
sorry

end base_conversion_problem_l174_174982


namespace nina_weeks_to_afford_game_l174_174173

noncomputable def game_cost : ℝ := 50
noncomputable def sales_tax_rate : ℝ := 0.10
noncomputable def weekly_allowance : ℝ := 10
noncomputable def saving_rate : ℝ := 0.5

noncomputable def total_cost : ℝ := game_cost + (game_cost * sales_tax_rate)
noncomputable def savings_per_week : ℝ := weekly_allowance * saving_rate
noncomputable def weeks_needed : ℝ := total_cost / savings_per_week

theorem nina_weeks_to_afford_game : weeks_needed = 11 := by
  sorry

end nina_weeks_to_afford_game_l174_174173


namespace combined_age_of_four_siblings_l174_174836

theorem combined_age_of_four_siblings :
  let aaron_age := 15
  let sister_age := 3 * aaron_age
  let henry_age := 4 * sister_age
  let alice_age := aaron_age - 2
  aaron_age + sister_age + henry_age + alice_age = 253 :=
by
  let aaron_age := 15
  let sister_age := 3 * aaron_age
  let henry_age := 4 * sister_age
  let alice_age := aaron_age - 2
  have h1 : aaron_age + sister_age + henry_age + alice_age = 15 + 3 * 15 + 4 * (3 * 15) + (15 - 2) := by sorry
  have h2 : 15 + 3 * 15 + 4 * (3 * 15) + (15 - 2) = 253 := by sorry
  exact h1.trans h2

end combined_age_of_four_siblings_l174_174836


namespace symmetric_line_equation_l174_174788

theorem symmetric_line_equation (x y : ℝ) :
  (2 : ℝ) * (2 - x) + (3 : ℝ) * (-2 - y) - 6 = 0 → 2 * x + 3 * y + 8 = 0 :=
by
  sorry

end symmetric_line_equation_l174_174788


namespace percentage_l_75_m_l174_174550

theorem percentage_l_75_m
  (j k l m : ℝ)
  (x : ℝ)
  (h1 : 1.25 * j = 0.25 * k)
  (h2 : 1.5 * k = 0.5 * l)
  (h3 : (x / 100) * l = 0.75 * m)
  (h4 : 0.2 * m = 7 * j) :
  x = 175 :=
by
  sorry

end percentage_l_75_m_l174_174550


namespace cost_of_green_pill_l174_174223

-- Let the cost of a green pill be g and the cost of a pink pill be p
variables (g p : ℕ)
-- Beth takes two green pills and one pink pill each day
-- A green pill costs twice as much as a pink pill
-- The total cost for the pills over three weeks (21 days) is $945

theorem cost_of_green_pill : 
  (2 * g + p) * 21 = 945 ∧ g = 2 * p → g = 18 :=
by
  sorry

end cost_of_green_pill_l174_174223


namespace courtyard_paving_l174_174941

noncomputable def length_of_brick (L : ℕ) := L = 12

theorem courtyard_paving  (courtyard_length : ℕ) (courtyard_width : ℕ) 
                           (brick_width : ℕ) (total_bricks : ℕ) 
                           (H1 : courtyard_length = 18) (H2 : courtyard_width = 12) 
                           (H3 : brick_width = 6) (H4 : total_bricks = 30000) 
                           : length_of_brick 12 := 
by 
  sorry

end courtyard_paving_l174_174941


namespace intersection_A_B_l174_174297

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | 0 ≤ x ∧ x < 5 / 2}
def C : Set ℤ := {x | x ∈ A ∧ (x : ℝ) ∈ B}

theorem intersection_A_B : C = {0, 1, 2} := 
by
  sorry

end intersection_A_B_l174_174297


namespace smallest_n_l174_174820

theorem smallest_n (n : ℕ) (h1 : 1826 % 26 = 6) (h2 : 5 * n % 26 = 6) : n = 20 :=
sorry

end smallest_n_l174_174820


namespace problem1_problem2_l174_174839

-- Problem 1
theorem problem1 (a b c d : ℝ) (hab : a * b > 0) (hbc_ad : b * c - a * d > 0) : (c / a) - (d / b) > 0 := sorry

-- Problem 2
theorem problem2 (a b c d : ℝ) (ha_gt_b : a > b) (hc_gt_d : c > d) : a - d > b - c := sorry

end problem1_problem2_l174_174839


namespace arith_seq_a4a6_equals_4_l174_174485

variable (a : ℕ → ℝ) (d : ℝ)
variable (h2 : a 2 = a 1 + d)
variable (h4 : a 4 = a 1 + 3 * d)
variable (h6 : a 6 = a 1 + 5 * d)
variable (h8 : a 8 = a 1 + 7 * d)
variable (h10 : a 10 = a 1 + 9 * d)
variable (condition : (a 2)^2 + 2 * a 2 * a 8 + a 6 * a 10 = 16)

theorem arith_seq_a4a6_equals_4 : a 4 * a 6 = 4 := by
  sorry

end arith_seq_a4a6_equals_4_l174_174485


namespace find_angle_B_l174_174255

noncomputable def angle_B (a b c : ℝ) (B C : ℝ) : Prop :=
b = 2 * Real.sqrt 3 ∧ c = 2 ∧ C = Real.pi / 6 ∧
(Real.sin B = (b * Real.sin C) / c ∧ b > c → (B = Real.pi / 3 ∨ B = 2 * Real.pi / 3))

theorem find_angle_B :
  ∃ (B : ℝ), angle_B 1 (2 * Real.sqrt 3) 2 B (Real.pi / 6) :=
by
  sorry

end find_angle_B_l174_174255


namespace speed_of_slower_train_is_36_l174_174274

-- Definitions used in the conditions
def length_of_train := 25 -- meters
def combined_length_of_trains := 2 * length_of_train -- meters
def time_to_pass := 18 -- seconds
def speed_of_faster_train := 46 -- km/hr
def conversion_factor := 1000 / 3600 -- to convert from km/hr to m/s

-- Prove that speed of the slower train is 36 km/hr
theorem speed_of_slower_train_is_36 :
  ∃ v : ℕ, v = 36 ∧ ((combined_length_of_trains : ℝ) = ((speed_of_faster_train - v) * conversion_factor * time_to_pass)) :=
sorry

end speed_of_slower_train_is_36_l174_174274


namespace front_crawl_speed_l174_174468
   
   def swim_condition := 
     ∃ F : ℝ, -- Speed of front crawl in yards per minute
     (∃ t₁ t₂ d₁ d₂ : ℝ, -- t₁ is time for front crawl, t₂ is time for breaststroke, d₁ and d₂ are distances
               t₁ = 8 ∧
               t₂ = 4 ∧
               d₁ = t₁ * F ∧
               d₂ = t₂ * 35 ∧
               d₁ + d₂ = 500 ∧
               t₁ + t₂ = 12) ∧
     F = 45
   
   theorem front_crawl_speed : swim_condition :=
     by
       sorry -- Proof goes here, with given conditions satisfying F = 45
   
end front_crawl_speed_l174_174468


namespace solve_system1_solve_system2_l174_174680

-- Definition for System (1)
theorem solve_system1 (x y : ℤ) (h1 : x - 2 * y = 0) (h2 : 3 * x - y = 5) : x = 2 ∧ y = 1 := 
by
  sorry

-- Definition for System (2)
theorem solve_system2 (x y : ℤ) 
  (h1 : 3 * (x - 1) - 4 * (y + 1) = -1) 
  (h2 : (x / 2) + (y / 3) = -2) : x = -2 ∧ y = -3 := 
by
  sorry

end solve_system1_solve_system2_l174_174680


namespace zachary_needs_more_money_l174_174309

def cost_in_usd_football (euro_to_usd : ℝ) (football_cost_eur : ℝ) : ℝ :=
  football_cost_eur * euro_to_usd

def cost_in_usd_shorts (gbp_to_usd : ℝ) (shorts_cost_gbp : ℝ) (pairs : ℕ) : ℝ :=
  shorts_cost_gbp * pairs * gbp_to_usd

def cost_in_usd_shoes (shoes_cost_usd : ℝ) : ℝ :=
  shoes_cost_usd

def cost_in_usd_socks (jpy_to_usd : ℝ) (socks_cost_jpy : ℝ) (pairs : ℕ) : ℝ :=
  socks_cost_jpy * pairs * jpy_to_usd

def cost_in_usd_water_bottle (krw_to_usd : ℝ) (water_bottle_cost_krw : ℝ) : ℝ :=
  water_bottle_cost_krw * krw_to_usd

def total_cost_before_discount (cost_football_usd cost_shorts_usd cost_shoes_usd
                                cost_socks_usd cost_water_bottle_usd : ℝ) : ℝ :=
  cost_football_usd + cost_shorts_usd + cost_shoes_usd + cost_socks_usd + cost_water_bottle_usd

def discounted_total_cost (total_cost : ℝ) (discount : ℝ) : ℝ :=
  total_cost * (1 - discount)

def additional_money_needed (discounted_total_cost current_money : ℝ) : ℝ :=
  discounted_total_cost - current_money

theorem zachary_needs_more_money (euro_to_usd : ℝ) (gbp_to_usd : ℝ) (jpy_to_usd : ℝ) (krw_to_usd : ℝ)
  (football_cost_eur : ℝ) (shorts_cost_gbp : ℝ) (pairs_shorts : ℕ) (shoes_cost_usd : ℝ)
  (socks_cost_jpy : ℝ) (pairs_socks : ℕ) (water_bottle_cost_krw : ℝ) (current_money_usd : ℝ)
  (discount : ℝ) : additional_money_needed 
      (discounted_total_cost
          (total_cost_before_discount
            (cost_in_usd_football euro_to_usd football_cost_eur)
            (cost_in_usd_shorts gbp_to_usd shorts_cost_gbp pairs_shorts)
            (cost_in_usd_shoes shoes_cost_usd)
            (cost_in_usd_socks jpy_to_usd socks_cost_jpy pairs_socks)
            (cost_in_usd_water_bottle krw_to_usd water_bottle_cost_krw)) 
          discount) 
      current_money_usd = 7.127214 := 
sorry

end zachary_needs_more_money_l174_174309


namespace ondra_homework_problems_l174_174552

theorem ondra_homework_problems (a b c d : ℤ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (-a) * (-b) ≠ -a - b ∧ 
  (-c) * (-d) = -182 * (1 / (-c - d)) →
  ((a = 2 ∧ b = 2) 
  ∨ (c = 1 ∧ d = 13) 
  ∨ (c = 13 ∧ d = 1)) :=
sorry

end ondra_homework_problems_l174_174552


namespace meaningful_range_fraction_l174_174538

theorem meaningful_range_fraction (x : ℝ) : 
  ¬ (x = 3) ↔ (∃ y, y = x / (x - 3)) :=
sorry

end meaningful_range_fraction_l174_174538


namespace sequence_sum_S6_l174_174608

theorem sequence_sum_S6 (a_n : ℕ → ℕ) (S_n : ℕ → ℕ) (h : ∀ n, S_n n = 2 * a_n n - 3) :
  S_n 6 = 189 :=
by
  sorry

end sequence_sum_S6_l174_174608


namespace fraction_of_garden_occupied_by_triangle_beds_l174_174081

theorem fraction_of_garden_occupied_by_triangle_beds :
  ∀ (rect_height rect_width trapezoid_short_base trapezoid_long_base : ℝ) 
    (num_triangles : ℕ) 
    (triangle_leg_length : ℝ)
    (total_area_triangles : ℝ)
    (total_garden_area : ℝ)
    (fraction : ℝ),
  rect_height = 10 → rect_width = 30 →
  trapezoid_short_base = 20 → trapezoid_long_base = 30 → num_triangles = 3 →
  triangle_leg_length = 10 / 3 →
  total_area_triangles = 3 * (1 / 2 * (triangle_leg_length ^ 2)) →
  total_garden_area = rect_height * rect_width →
  fraction = total_area_triangles / total_garden_area →
  fraction = 1 / 18 := by
  intros rect_height rect_width trapezoid_short_base trapezoid_long_base
         num_triangles triangle_leg_length total_area_triangles
         total_garden_area fraction
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end fraction_of_garden_occupied_by_triangle_beds_l174_174081


namespace percentage_proof_l174_174953

theorem percentage_proof (a : ℝ) (paise : ℝ) (x : ℝ) (h1: paise = 85) (h2: a = 170) : 
  (x/100) * a = paise ↔ x = 50 := 
by
  -- The setup includes:
  -- paise = 85
  -- a = 170
  -- We prove that x% of 170 equals 85 if and only if x = 50.
  sorry

end percentage_proof_l174_174953


namespace cost_comparison_compare_cost_when_x_30_more_cost_effective_30_l174_174455

-- Definitions for the conditions in the problem
def cost_of_suit : ℕ := 1000
def cost_of_tie : ℕ := 200

-- Definitions for Option 1 and Option 2 calculations
def option1_total_cost (x : ℕ) (h : x > 20) : ℕ := 200 * x + 16000
def option2_total_cost (x : ℕ) (h : x > 20) : ℕ := 180 * x + 18000

-- Case x=30 for comparison
def x : ℕ := 30
def option1_cost_when_x_30 : ℕ := 200 * x + 16000
def option2_cost_when_x_30 : ℕ := 180 * x + 18000

-- More cost-effective plan when x=30
def more_cost_effective_plan_for_x_30 : ℕ := 21800

theorem cost_comparison (x : ℕ) (h1 : x > 20) :
  option1_total_cost x h1 = 200 * x + 16000 ∧
  option2_total_cost x h1 = 180 * x + 18000 := 
by
  sorry

theorem compare_cost_when_x_30 :
  option1_cost_when_x_30 = 22000 ∧
  option2_cost_when_x_30 = 23400 ∧
  option1_cost_when_x_30 < option2_cost_when_x_30 := 
by
  sorry

theorem more_cost_effective_30 :
  more_cost_effective_plan_for_x_30 = 21800 := 
by
  sorry

end cost_comparison_compare_cost_when_x_30_more_cost_effective_30_l174_174455


namespace solve_custom_eq_l174_174364

-- Define the custom operation a * b = ab + a + b, we will use ∗ instead of * to avoid confusion with multiplication

def custom_op (a b : Nat) : Nat := a * b + a + b

-- State the problem in Lean 4
theorem solve_custom_eq (x : Nat) : custom_op 3 x = 27 → x = 6 :=
by
  sorry

end solve_custom_eq_l174_174364


namespace arithmetic_geometric_sequence_l174_174410

theorem arithmetic_geometric_sequence (a : ℕ → ℤ) (h1 : ∀ n, a (n + 1) = a n + 3)
    (h2 : (a 1 + 3) * (a 1 + 21) = (a 1 + 9) ^ 2) : a 3 = 12 :=
by 
  sorry

end arithmetic_geometric_sequence_l174_174410


namespace total_food_per_day_l174_174753

theorem total_food_per_day :
  let num_puppies := 4
  let num_dogs := 3
  let dog_meal_weight := 4
  let dog_meals_per_day := 3
  let dog_food_per_day := dog_meal_weight * dog_meals_per_day
  let total_dog_food_per_day := dog_food_per_day * num_dogs
  let puppy_meal_weight := dog_meal_weight / 2
  let puppy_meals_per_day := dog_meals_per_day * 3
  let puppy_food_per_day := puppy_meal_weight * puppy_meals_per_day
  let total_puppy_food_per_day := puppy_food_per_day * num_puppies
  total_dog_food_per_day + total_puppy_food_per_day = 108 :=
by
  sorry

end total_food_per_day_l174_174753


namespace not_always_possible_repaint_all_white_l174_174620

-- Define the conditions and the problem
def equilateral_triangle_division (n: ℕ) : Prop := 
  ∀ m, m > 1 → m = n^2

def line_parallel_repaint (triangles : List ℕ) : Prop :=
  -- Definition of how the repaint operation affects the triangle colors
  sorry

theorem not_always_possible_repaint_all_white (n : ℕ) (h: equilateral_triangle_division n) :
  ¬∀ triangles, line_parallel_repaint triangles → (∀ t ∈ triangles, t = 0) := 
sorry

end not_always_possible_repaint_all_white_l174_174620


namespace combined_total_years_l174_174240

theorem combined_total_years (A : ℕ) (V : ℕ) (D : ℕ)
(h1 : V = A + 9)
(h2 : V = D - 9)
(h3 : D = 34) : A + V + D = 75 :=
by sorry

end combined_total_years_l174_174240


namespace sin_square_eq_c_div_a2_plus_b2_l174_174893

theorem sin_square_eq_c_div_a2_plus_b2 
  (a b c : ℝ) (α β : ℝ)
  (h1 : a * Real.cos α + b * Real.sin α = c)
  (h2 : a * Real.cos β + b * Real.sin β = 0)
  (not_all_zero : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) : 
  Real.sin (α - β) ^ 2 = c ^ 2 / (a ^ 2 + b ^ 2) :=
by
  sorry

end sin_square_eq_c_div_a2_plus_b2_l174_174893


namespace simplify_polynomial_l174_174826

theorem simplify_polynomial :
  (3 * y - 2) * (6 * y ^ 12 + 3 * y ^ 11 + 6 * y ^ 10 + 3 * y ^ 9) =
  18 * y ^ 13 - 3 * y ^ 12 + 12 * y ^ 11 - 3 * y ^ 10 - 6 * y ^ 9 :=
by
  sorry

end simplify_polynomial_l174_174826


namespace find_x_when_y_is_minus_21_l174_174715

variable (x y k : ℝ)

theorem find_x_when_y_is_minus_21
  (h1 : x * y = k)
  (h2 : x + y = 35)
  (h3 : y = 3 * x)
  (h4 : y = -21) :
  x = -10.9375 := by
  sorry

end find_x_when_y_is_minus_21_l174_174715


namespace range_of_a_l174_174959

-- Defining the function f
noncomputable def f (x a : ℝ) : ℝ :=
  (Real.exp x) * (2 * x - 1) - a * x + a

-- Main statement
theorem range_of_a (a : ℝ)
  (h1 : a < 1)
  (h2 : ∃ x0 x1 : ℤ, x0 ≠ x1 ∧ f x0 a ≤ 0 ∧ f x1 a ≤ 0) :
  (5 / (3 * Real.exp 2)) < a ∧ a ≤ (3 / (2 * Real.exp 1)) :=
sorry

end range_of_a_l174_174959


namespace inequality_nonempty_solution_set_l174_174251

theorem inequality_nonempty_solution_set (a : ℝ) : 
  (∃ x : ℝ, |x-3| + |x-4| < a) ↔ a > 1 :=
by
  sorry

end inequality_nonempty_solution_set_l174_174251


namespace find_some_number_l174_174416

-- Definitions of symbol replacements
def replacement_minus (a b : Nat) := a + b
def replacement_plus (a b : Nat) := a * b
def replacement_times (a b : Nat) := a / b
def replacement_div (a b : Nat) := a - b

-- The transformed equation using the replacements
def transformed_equation (some_number : Nat) :=
  replacement_minus
    some_number
    (replacement_div
      (replacement_plus 9 (replacement_times 8 3))
      25) = 5

theorem find_some_number : ∃ some_number : Nat, transformed_equation some_number ∧ some_number = 6 :=
by
  exists 6
  unfold transformed_equation
  unfold replacement_minus replacement_plus replacement_times replacement_div
  sorry

end find_some_number_l174_174416


namespace inequality_solution_l174_174091

theorem inequality_solution : { x : ℝ | (x - 1) / (x + 3) < 0 } = { x : ℝ | -3 < x ∧ x < 1 } :=
sorry

end inequality_solution_l174_174091


namespace no_six_coins_sum_70_cents_l174_174769

theorem no_six_coins_sum_70_cents :
  ¬ ∃ (p n d q : ℕ), p + n + d + q = 6 ∧ p + 5 * n + 10 * d + 25 * q = 70 :=
by
  sorry

end no_six_coins_sum_70_cents_l174_174769


namespace vector_dot_product_l174_174302

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (-1, 2)

-- Define the operation to calculate (a + 2b)
def two_b : ℝ × ℝ := (2 * b.1, 2 * b.2)
def a_plus_2b : ℝ × ℝ := (a.1 + two_b.1, a.2 + two_b.2)

-- Define the dot product function
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- State the theorem
theorem vector_dot_product : dot_product a_plus_2b b = 14 := by
  sorry

end vector_dot_product_l174_174302


namespace total_feet_is_correct_l174_174275

-- definitions according to conditions
def number_of_heads := 46
def number_of_hens := 24
def number_of_cows := number_of_heads - number_of_hens
def hen_feet := 2
def cow_feet := 4
def total_hen_feet := number_of_hens * hen_feet
def total_cow_feet := number_of_cows * cow_feet
def total_feet := total_hen_feet + total_cow_feet

-- proof statement with sorry
theorem total_feet_is_correct : total_feet = 136 :=
by
  sorry

end total_feet_is_correct_l174_174275


namespace gcd_A_B_l174_174404

theorem gcd_A_B (a b : ℕ) (h1 : Nat.gcd a b = 1) (h2 : a > 0) (h3 : b > 0) : 
  Nat.gcd (8 * a + 3 * b) (3 * a + 2 * b) ≠ 1 → Nat.gcd (8 * a + 3 * b) (3 * a + 2 * b) = 7 :=
by
  sorry

end gcd_A_B_l174_174404


namespace range_of_k_l174_174936

theorem range_of_k (k : ℝ) :
  (∃ x : ℝ, (k - 1) * x^2 + 2 * x - 1 = 0) ↔ (k ≥ 0 ∧ k ≠ 1) :=
by
  sorry

end range_of_k_l174_174936


namespace num_values_of_a_l174_174434

def A : Set ℝ := {-1, 1, 3}
def B (a : ℝ) : Set ℝ := {1, a^2 - 2 * a}

theorem num_values_of_a : ∃v : Finset ℝ, (∀ a ∈ v, B a ⊆ A) ∧ v.card = 3 :=
by
  sorry

end num_values_of_a_l174_174434


namespace age_difference_l174_174411

-- Defining the current age of the son
def S : ℕ := 26

-- Defining the current age of the man
def M : ℕ := 54

-- Defining the condition that in two years, the man's age is twice the son's age
def condition : Prop := (M + 2) = 2 * (S + 2)

-- The theorem that states how much older the man is than the son
theorem age_difference : condition → M - S = 28 := by
  sorry

end age_difference_l174_174411


namespace y_coordinate_of_point_on_line_l174_174803

theorem y_coordinate_of_point_on_line (x y : ℝ) (h1 : -4 = x) (h2 : ∃ m b : ℝ, y = m * x + b ∧ y = 3 ∧ x = 10 ∧ m * 4 + b = 0) : y = -4 :=
sorry

end y_coordinate_of_point_on_line_l174_174803


namespace max_lg_value_l174_174291

noncomputable def max_lg_product (x y : ℝ) (hx: x > 1) (hy: y > 1) (hxy: Real.log x / Real.log 10 + Real.log y / Real.log 10 = 4) : ℝ :=
  4

theorem max_lg_value (x y : ℝ) (hx : x > 1) (hy : y > 1) (hxy : Real.log x / Real.log 10 + Real.log y / Real.log 10 = 4) :
  max_lg_product x y hx hy hxy = 4 := 
by
  unfold max_lg_product
  sorry

end max_lg_value_l174_174291


namespace four_digit_number_count_l174_174324

-- Define the start and end of four-digit numbers
def fourDigitStart : Nat := 1000
def fourDigitEnd : Nat := 9999

-- Main theorem: Number of four-digit numbers
theorem four_digit_number_count : fourDigitEnd - fourDigitStart + 1 = 9000 := by
  sorry  -- Proof here

end four_digit_number_count_l174_174324


namespace remainder_of_1998_to_10_mod_10k_l174_174692

theorem remainder_of_1998_to_10_mod_10k : 
  let x := 1998
  let y := 10^4
  x^10 % y = 1024 := 
by
  let x := 1998
  let y := 10^4
  sorry

end remainder_of_1998_to_10_mod_10k_l174_174692


namespace jeff_average_skips_is_14_l174_174107

-- Definitions of the given conditions in the problem
def sam_skips_per_round : ℕ := 16
def rounds : ℕ := 4

-- Number of skips by Jeff in each round based on the conditions
def jeff_first_round_skips : ℕ := sam_skips_per_round - 1
def jeff_second_round_skips : ℕ := sam_skips_per_round - 3
def jeff_third_round_skips : ℕ := sam_skips_per_round + 4
def jeff_fourth_round_skips : ℕ := sam_skips_per_round / 2

-- Total skips by Jeff in all rounds
def jeff_total_skips : ℕ := jeff_first_round_skips + 
                           jeff_second_round_skips + 
                           jeff_third_round_skips + 
                           jeff_fourth_round_skips

-- Average skips per round by Jeff
def jeff_average_skips : ℕ := jeff_total_skips / rounds

-- Theorem statement
theorem jeff_average_skips_is_14 : jeff_average_skips = 14 := 
by 
    sorry

end jeff_average_skips_is_14_l174_174107


namespace middle_tree_distance_l174_174559

theorem middle_tree_distance (d : ℕ) (b : ℕ) (c : ℕ) 
  (h_b : b = 84) (h_c : c = 91) 
  (h_right_triangle : d^2 + b^2 = c^2) : 
  d = 35 :=
by
  sorry

end middle_tree_distance_l174_174559


namespace pirates_coins_l174_174735

noncomputable def coins (x : ℕ) (k : ℕ) : ℕ :=
  match k with
  | 0     => x
  | k + 1 => (coins x k) - (coins x k * (k + 2) / 15)

theorem pirates_coins (x : ℕ) (H : x = 2^15 * 3^8 * 5^14) :
  ∃ n : ℕ, n = coins x 14 :=
sorry

end pirates_coins_l174_174735


namespace max_log_value_l174_174601

noncomputable def max_log_product (a b : ℝ) : ℝ :=
  if h : a > 0 ∧ b > 0 ∧ a * b = 8 then (Real.logb 2 a) * (Real.logb 2 (2 * b)) else 0

theorem max_log_value (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a * b = 8) :
  max_log_product a b ≤ 4 :=
sorry

end max_log_value_l174_174601


namespace negation_of_existence_l174_174636

theorem negation_of_existence :
  (¬ ∃ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 > 0) :=
by
  sorry

end negation_of_existence_l174_174636


namespace solve_for_A_l174_174819

theorem solve_for_A (A B : ℕ) (h1 : 4 * 10 + A + 10 * B + 3 = 68) (h2 : 10 ≤ 4 * 10 + A) (h3 : 4 * 10 + A < 100) (h4 : 10 ≤ 10 * B + 3) (h5 : 10 * B + 3 < 100) (h6 : A < 10) (h7 : B < 10) : A = 5 := 
by
  sorry

end solve_for_A_l174_174819


namespace cylinder_volume_l174_174098

theorem cylinder_volume (short_side long_side : ℝ) (h_short_side : short_side = 12) (h_long_side : long_side = 18) : 
  ∀ (r h : ℝ) (h_radius : r = short_side / 2) (h_height : h = long_side), 
    volume = π * r^2 * h := 
by
  sorry

end cylinder_volume_l174_174098


namespace initial_number_of_men_l174_174180

theorem initial_number_of_men (M : ℕ) (h1 : ∃ food : ℕ, food = M * 22) (h2 : ∀ food, food = (M * 20)) (h3 : ∃ food : ℕ, food = ((M + 40) * 19)) : M = 760 := by
  sorry

end initial_number_of_men_l174_174180


namespace john_expenditure_l174_174553

theorem john_expenditure (X : ℝ) (h : (1/2) * X + (1/3) * X + (1/10) * X + 8 = X) : X = 120 :=
by
  sorry

end john_expenditure_l174_174553


namespace range_of_a_for_p_range_of_a_for_p_and_q_l174_174177

variable (a : ℝ)

/-- For any x ∈ ℝ, ax^2 - x + 3 > 0 if and only if a > 1/12 -/
def condition_p : Prop := ∀ x : ℝ, a * x^2 - x + 3 > 0

/-- There exists x ∈ [1, 2] such that 2^x * a ≥ 1 -/
def condition_q : Prop := ∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ a * 2^x ≥ 1

/-- Theorem (1): The range of values for a such that condition_p holds true is (1/12, +∞) -/
theorem range_of_a_for_p (h : condition_p a) : a > 1/12 :=
sorry

/-- Theorem (2): The range of values for a such that condition_p and condition_q have different truth values is (1/12, 1/4) -/
theorem range_of_a_for_p_and_q (h₁ : condition_p a) (h₂ : ¬condition_q a) : 1/12 < a ∧ a < 1/4 :=
sorry

end range_of_a_for_p_range_of_a_for_p_and_q_l174_174177


namespace expenditure_ratio_l174_174534

variable (P1 P2 : Type)
variable (I1 I2 E1 E2 : ℝ)
variable (R_incomes : I1 / I2 = 5 / 4)
variable (S1 S2 : ℝ)
variable (S_equal : S1 = S2)
variable (I1_fixed : I1 = 4000)
variable (Savings : S1 = 1600)

theorem expenditure_ratio :
  (I1 - E1 = 1600) → 
  (I2 * 4 / 5 - E2 = 1600) →
  I2 = 3200 →
  E1 / E2 = 3 / 2 :=
by
  intro P1_savings P2_savings I2_calc
  -- proof steps go here
  sorry

end expenditure_ratio_l174_174534


namespace smallest_integer_to_make_multiple_of_five_l174_174967

/-- The smallest positive integer that can be added to 725 to make it a multiple of 5 is 5. -/
theorem smallest_integer_to_make_multiple_of_five : 
  ∃ k : ℕ, k > 0 ∧ (725 + k) % 5 = 0 ∧ ∀ m : ℕ, m > 0 ∧ (725 + m) % 5 = 0 → k ≤ m :=
sorry

end smallest_integer_to_make_multiple_of_five_l174_174967


namespace tree_drops_leaves_on_fifth_day_l174_174447

def initial_leaves := 340
def daily_drop_fraction := 1 / 10

noncomputable def leaves_after_day (n: ℕ) : ℕ :=
  match n with
  | 0 => initial_leaves
  | 1 => initial_leaves - Nat.floor (initial_leaves * daily_drop_fraction)
  | 2 => leaves_after_day 1 - Nat.floor (leaves_after_day 1 * daily_drop_fraction)
  | 3 => leaves_after_day 2 - Nat.floor (leaves_after_day 2 * daily_drop_fraction)
  | 4 => leaves_after_day 3 - Nat.floor (leaves_after_day 3 * daily_drop_fraction)
  | _ => 0  -- beyond the 4th day

theorem tree_drops_leaves_on_fifth_day : leaves_after_day 4 = 225 := by
  -- We'll skip the detailed proof here, focusing on the statement
  sorry

end tree_drops_leaves_on_fifth_day_l174_174447


namespace scientific_notation_correct_l174_174281

-- The given number
def given_number : ℕ := 9000000000

-- The correct answer in scientific notation
def correct_sci_not : ℕ := 9 * (10 ^ 9)

-- The theorem to prove
theorem scientific_notation_correct :
  given_number = correct_sci_not :=
by
  sorry

end scientific_notation_correct_l174_174281


namespace fraction_of_surface_area_is_red_l174_174305

structure Cube :=
  (edge_length : ℕ)
  (small_cubes : ℕ)
  (num_red_cubes : ℕ)
  (num_blue_cubes : ℕ)
  (blue_cube_edge_length : ℕ)
  (red_outer_layer : ℕ)

def surface_area (c : Cube) : ℕ := 6 * (c.edge_length * c.edge_length)

theorem fraction_of_surface_area_is_red (c : Cube) 
  (h_edge_length : c.edge_length = 4)
  (h_small_cubes : c.small_cubes = 64)
  (h_num_red_cubes : c.num_red_cubes = 40)
  (h_num_blue_cubes : c.num_blue_cubes = 24)
  (h_blue_cube_edge_length : c.blue_cube_edge_length = 2)
  (h_red_outer_layer : c.red_outer_layer = 1)
  : (surface_area c) / (surface_area c) = 1 := 
by
  sorry

end fraction_of_surface_area_is_red_l174_174305


namespace find_real_root_a_l174_174406

theorem find_real_root_a (a b c : ℂ) (ha : a.im = 0) (h1 : a + b + c = 5) (h2 : a * b + b * c + c * a = 7) (h3 : a * b * c = 3) : a = 1 :=
sorry

end find_real_root_a_l174_174406


namespace exists_c_d_in_set_of_13_reals_l174_174067

theorem exists_c_d_in_set_of_13_reals (a : Fin 13 → ℝ) :
  ∃ (c d : ℝ), c ∈ Set.range a ∧ d ∈ Set.range a ∧ 0 < (c - d) / (1 + c * d) ∧ (c - d) / (1 + c * d) < 2 - Real.sqrt 3 := 
by
  sorry

end exists_c_d_in_set_of_13_reals_l174_174067


namespace final_hair_length_l174_174449

theorem final_hair_length (x y z : ℕ) (hx : x = 16) (hy : y = 11) (hz : z = 12) : 
  (x - y) + z = 17 :=
by
  sorry

end final_hair_length_l174_174449


namespace toads_max_l174_174137

theorem toads_max (n : ℕ) (h₁ : n ≥ 3) : 
  ∃ k : ℕ, k = ⌈ (n : ℝ) / 2 ⌉ ∧ ∀ (labels : Fin n → Fin n) (jumps : Fin n → ℕ), 
  (∀ i, jumps (labels i) = labels i) → ¬ ∃ f : Fin k → Fin n, ∀ i₁ i₂, i₁ ≠ i₂ → f i₁ ≠ f i₂ :=
sorry

end toads_max_l174_174137


namespace triangle_sum_l174_174908

def triangle (a b c : ℕ) : ℤ := a * b - c

theorem triangle_sum :
  triangle 2 3 5 + triangle 1 4 7 = -2 :=
by
  -- This is where the proof would go
  sorry

end triangle_sum_l174_174908


namespace find_cos_value_l174_174117

open Real

noncomputable def cos_value (α : ℝ) : ℝ :=
  cos (2 * π / 3 + 2 * α)

theorem find_cos_value (α : ℝ) (h : sin (π / 6 - α) = 1 / 4) :
  cos_value α = -7 / 8 :=
sorry

end find_cos_value_l174_174117


namespace problem_statement_l174_174937

def class_of_rem (k : ℕ) : Set ℤ := {n | ∃ m : ℤ, n = 4 * m + k}

theorem problem_statement : (2013 ∈ class_of_rem 1) ∧ 
                            (-2 ∈ class_of_rem 2) ∧ 
                            (∀ x : ℤ, x ∈ class_of_rem 0 ∨ x ∈ class_of_rem 1 ∨ x ∈ class_of_rem 2 ∨ x ∈ class_of_rem 3) ∧ 
                            (∀ a b : ℤ, (∃ k : ℕ, (a ∈ class_of_rem k ∧ b ∈ class_of_rem k)) ↔ (a - b) ∈ class_of_rem 0) :=
by
  -- each of the statements should hold true
  sorry

end problem_statement_l174_174937


namespace total_balls_l174_174384

theorem total_balls (black_balls : ℕ) (prob_pick_black : ℚ) (total_balls : ℕ) :
  black_balls = 4 → prob_pick_black = 1 / 3 → total_balls = 12 :=
by
  intros h1 h2
  sorry

end total_balls_l174_174384


namespace greatest_number_of_cool_cells_l174_174097

noncomputable def greatest_cool_cells (n : ℕ) (grid : Matrix (Fin n) (Fin n) ℝ) : ℕ :=
n^2 - 2 * n + 1

theorem greatest_number_of_cool_cells (n : ℕ) (grid : Matrix (Fin n) (Fin n) ℝ) (h : 0 < n) :
  ∃ m, m = (n - 1)^2 ∧ m = greatest_cool_cells n grid :=
sorry

end greatest_number_of_cool_cells_l174_174097


namespace weight_of_square_piece_l174_174285

open Real

theorem weight_of_square_piece 
  (uniform_density : Prop)
  (side_length_triangle side_length_square : ℝ)
  (weight_triangle : ℝ)
  (ht : side_length_triangle = 6)
  (hs : side_length_square = 6)
  (wt : weight_triangle = 48) :
  ∃ weight_square : ℝ, weight_square = 27.7 :=
by
  sorry

end weight_of_square_piece_l174_174285


namespace qudrilateral_diagonal_length_l174_174208

theorem qudrilateral_diagonal_length (A h1 h2 d : ℝ) 
  (h_area : A = 140) (h_offsets : h1 = 8) (h_offsets2 : h2 = 2) 
  (h_formula : A = 1 / 2 * d * (h1 + h2)) : 
  d = 28 :=
by
  sorry

end qudrilateral_diagonal_length_l174_174208


namespace dime_quarter_problem_l174_174368

theorem dime_quarter_problem :
  15 * 25 + 10 * 10 = 25 * 25 + 35 * 10 :=
by
  sorry

end dime_quarter_problem_l174_174368


namespace cos_third_quadrant_l174_174815

theorem cos_third_quadrant (B : ℝ) (hB : -π < B ∧ B < -π / 2) (sin_B : Real.sin B = 5 / 13) :
  Real.cos B = -12 / 13 :=
by
  sorry

end cos_third_quadrant_l174_174815


namespace arrange_digits_l174_174430

theorem arrange_digits (A B C D E F : ℕ)
  (h1 : A ≠ B) (h2 : A ≠ C) (h3 : A ≠ D) (h4 : A ≠ E) (h5 : A ≠ F)
  (h6 : B ≠ C) (h7 : B ≠ D) (h8 : B ≠ E) (h9 : B ≠ F)
  (h10 : C ≠ D) (h11 : C ≠ E) (h12 : C ≠ F)
  (h13 : D ≠ E) (h14 : D ≠ F) (h15 : E ≠ F)
  (range_A : 1 ≤ A ∧ A ≤ 6) (range_B : 1 ≤ B ∧ B ≤ 6) (range_C : 1 ≤ C ∧ C ≤ 6)
  (range_D : 1 ≤ D ∧ D ≤ 6) (range_E : 1 ≤ E ∧ E ≤ 6) (range_F : 1 ≤ F ∧ F ≤ 6)
  (sum_line1 : A + D + E = 15) (sum_line2 : A + C + 9 = 15) 
  (sum_line3 : B + D + 9 = 15) (sum_line4 : 7 + C + E = 15) 
  (sum_line5 : 9 + C + A = 15) (sum_line6 : A + 8 + F = 15) 
  (sum_line7 : 7 + D + F = 15) : 
  (A = 4) ∧ (B = 1) ∧ (C = 2) ∧ (D = 5) ∧ (E = 6) ∧ (F = 3) :=
sorry

end arrange_digits_l174_174430


namespace mistake_position_is_34_l174_174197

def arithmetic_sequence_sum (n : ℕ) (a_1 : ℕ) (d : ℕ) : ℕ :=
  n * (2 * a_1 + (n - 1) * d) / 2

def modified_sequence_sum (n : ℕ) (a_1 : ℕ) (d : ℕ) (mistake_index : ℕ) : ℕ :=
  let correct_sum := arithmetic_sequence_sum n a_1 d
  correct_sum - 2 * d

theorem mistake_position_is_34 :
  ∃ mistake_index : ℕ, mistake_index = 34 ∧ 
    modified_sequence_sum 37 1 3 mistake_index = 2011 :=
by
  sorry

end mistake_position_is_34_l174_174197


namespace billboard_shorter_side_length_l174_174210

theorem billboard_shorter_side_length
  (L W : ℝ)
  (h1 : L * W = 91)
  (h2 : 2 * L + 2 * W = 40) :
  L = 7 ∨ W = 7 :=
by sorry

end billboard_shorter_side_length_l174_174210


namespace functional_equation_solution_l174_174247

theorem functional_equation_solution :
  ∀ (f : ℝ → ℝ), (∀ x y : ℝ, f (⌊x⌋ * y) = f x * ⌊f y⌋) → (∀ x : ℝ, f x = 0) :=
by
  sorry

end functional_equation_solution_l174_174247


namespace intercept_sum_mod_7_l174_174532

theorem intercept_sum_mod_7 :
  ∃ (x_0 y_0 : ℤ), (2 * x_0 ≡ 3 * y_0 + 1 [ZMOD 7]) ∧ (0 ≤ x_0) ∧ (x_0 < 7) ∧ (0 ≤ y_0) ∧ (y_0 < 7) ∧ (x_0 + y_0 = 6) :=
by
  sorry

end intercept_sum_mod_7_l174_174532


namespace ball_in_boxes_l174_174204

theorem ball_in_boxes (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) :
  k^n = 243 :=
by
  sorry

end ball_in_boxes_l174_174204


namespace purely_imaginary_value_of_m_third_quadrant_value_of_m_l174_174148

theorem purely_imaginary_value_of_m (m : ℝ) :
  (2 * m^2 - 3 * m - 2 = 0) ∧ (m^2 - 2 * m ≠ 0) → m = -1/2 :=
by
  sorry

theorem third_quadrant_value_of_m (m : ℝ) :
  (2 * m^2 - 3 * m - 2 < 0) ∧ (m^2 - 2 * m < 0) → 0 < m ∧ m < 2 :=
by
  sorry

end purely_imaginary_value_of_m_third_quadrant_value_of_m_l174_174148


namespace ax_by_power5_l174_174415

-- Define the real numbers a, b, x, and y
variables (a b x y : ℝ)

-- Define the conditions as assumptions
axiom axiom1 : a * x + b * y = 3
axiom axiom2 : a * x^2 + b * y^2 = 7
axiom axiom3 : a * x^3 + b * y^3 = 16
axiom axiom4 : a * x^4 + b * y^4 = 42

-- State the theorem to prove ax^5 + by^5 = 20
theorem ax_by_power5 : a * x^5 + b * y^5 = 20 :=
  sorry

end ax_by_power5_l174_174415


namespace mobile_purchase_price_l174_174576

theorem mobile_purchase_price (M : ℝ) 
  (P_grinder : ℝ := 15000)
  (L_grinder : ℝ := 0.05 * P_grinder)
  (SP_grinder : ℝ := P_grinder - L_grinder)
  (SP_mobile : ℝ := 1.1 * M)
  (P_overall : ℝ := P_grinder + M)
  (SP_overall : ℝ := SP_grinder + SP_mobile)
  (profit : ℝ := 50)
  (h : SP_overall = P_overall + profit) :
  M = 8000 :=
by 
  sorry

end mobile_purchase_price_l174_174576


namespace sheela_monthly_income_eq_l174_174872

-- Defining the conditions
def sheela_deposit : ℝ := 4500
def percentage_of_income : ℝ := 0.28

-- Define Sheela's monthly income as I
variable (I : ℝ)

-- The theorem to prove
theorem sheela_monthly_income_eq : (percentage_of_income * I = sheela_deposit) → (I = 16071.43) :=
by
  sorry

end sheela_monthly_income_eq_l174_174872


namespace weekly_rental_fee_percentage_l174_174049

theorem weekly_rental_fee_percentage
  (camera_value : ℕ)
  (rental_period_weeks : ℕ)
  (friend_percentage : ℚ)
  (john_paid : ℕ)
  (percentage : ℚ)
  (total_rental_fee : ℚ)
  (weekly_rental_fee : ℚ)
  (P : ℚ)
  (camera_value_pos : camera_value = 5000)
  (rental_period_weeks_pos : rental_period_weeks = 4)
  (friend_percentage_pos : friend_percentage = 0.40)
  (john_paid_pos : john_paid = 1200)
  (percentage_pos : percentage = 1 - friend_percentage)
  (total_rental_fee_calc : total_rental_fee = john_paid / percentage)
  (weekly_rental_fee_calc : weekly_rental_fee = total_rental_fee / rental_period_weeks)
  (weekly_rental_fee_equation : weekly_rental_fee = P * camera_value)
  (P_calc : P = weekly_rental_fee / camera_value) :
  P * 100 = 10 := 
by 
  sorry

end weekly_rental_fee_percentage_l174_174049


namespace infinite_series_sum_l174_174951

theorem infinite_series_sum :
  ∑' (n : ℕ), (1 / (1 + 3^n : ℝ) - 1 / (1 + 3^(n+1) : ℝ)) = 1/2 := 
sorry

end infinite_series_sum_l174_174951


namespace sandy_remaining_puppies_l174_174939

-- Definitions from the problem
def initial_puppies : ℕ := 8
def given_away_puppies : ℕ := 4

-- Theorem statement
theorem sandy_remaining_puppies : initial_puppies - given_away_puppies = 4 := by
  sorry

end sandy_remaining_puppies_l174_174939


namespace probability_B_winning_l174_174375

def P_A : ℝ := 0.2
def P_D : ℝ := 0.5
def P_B : ℝ := 1 - (P_A + P_D)

theorem probability_B_winning : P_B = 0.3 :=
by
  -- Proof steps go here
  sorry

end probability_B_winning_l174_174375


namespace Adam_total_balls_l174_174043

def number_of_red_balls := 20
def number_of_blue_balls := 10
def number_of_orange_balls := 5
def number_of_pink_balls := 3 * number_of_orange_balls

def total_number_of_balls := 
  number_of_red_balls + number_of_blue_balls + number_of_pink_balls + number_of_orange_balls

theorem Adam_total_balls : total_number_of_balls = 50 := by
  sorry

end Adam_total_balls_l174_174043


namespace problem_statement_l174_174257

def op (x y : ℕ) : ℕ := x^2 + 2*y

theorem problem_statement (a : ℕ) : op a (op a a) = 3*a^2 + 4*a := 
by sorry

end problem_statement_l174_174257


namespace smallest_digit_divisibility_l174_174706

theorem smallest_digit_divisibility : 
  ∃ d : ℕ, (d < 10) ∧ (∃ k1 k2 : ℤ, 5 + 2 + 8 + d + 7 + 4 = 9 * k1 ∧ 5 + 2 + 8 + d + 7 + 4 = 3 * k2) ∧ (∀ d' : ℕ, (d' < 10) ∧ 
  (∃ k1 k2 : ℤ, 5 + 2 + 8 + d' + 7 + 4 = 9 * k1 ∧ 5 + 2 + 8 + d' + 7 + 4 = 3 * k2) → d ≤ d') :=
by
  sorry

end smallest_digit_divisibility_l174_174706


namespace max_value_of_expression_max_value_achieved_l174_174590

theorem max_value_of_expression (x y z : ℝ) (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) :
    8 * x + 3 * y + 10 * z ≤ Real.sqrt 173 :=
sorry

theorem max_value_achieved (x y z : ℝ) (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1)
    (hx : x = Real.sqrt 173 / 30)
    (hy : y = Real.sqrt 173 / 20)
    (hz : z = Real.sqrt 173 / 50) :
    8 * x + 3 * y + 10 * z = Real.sqrt 173 :=
sorry

end max_value_of_expression_max_value_achieved_l174_174590


namespace ratio_of_areas_l174_174063

-- Definitions of perimeter in Lean terms
def P_A : ℕ := 16
def P_B : ℕ := 32

-- Ratio of the area of region A to region C
theorem ratio_of_areas (s_A s_C : ℕ) (h₀ : 4 * s_A = P_A)
  (h₁ : 4 * s_C = 12) : s_A^2 / s_C^2 = 1 / 9 :=
by 
  sorry

end ratio_of_areas_l174_174063


namespace route_Y_quicker_than_route_X_l174_174392

theorem route_Y_quicker_than_route_X :
    let dist_X := 9  -- distance of Route X in miles
    let speed_X := 45  -- speed of Route X in miles per hour
    let dist_Y := 8  -- total distance of Route Y in miles
    let normal_dist_Y := 6.5  -- normal speed distance of Route Y in miles
    let construction_dist_Y := 1.5  -- construction zone distance of Route Y in miles
    let normal_speed_Y := 50  -- normal speed of Route Y in miles per hour
    let construction_speed_Y := 25  -- construction zone speed of Route Y in miles per hour
    let time_X := (dist_X / speed_X) * 60  -- time for Route X in minutes
    let time_Y1 := (normal_dist_Y / normal_speed_Y) * 60  -- time for normal speed segment of Route Y in minutes
    let time_Y2 := (construction_dist_Y / construction_speed_Y) * 60  -- time for construction zone segment of Route Y in minutes
    let time_Y := time_Y1 + time_Y2  -- total time for Route Y in minutes
    time_X - time_Y = 0.6 :=  -- the difference in time between Route X and Route Y in minutes
by
  sorry

end route_Y_quicker_than_route_X_l174_174392


namespace probability_at_least_one_tree_survives_l174_174345

noncomputable def prob_at_least_one_survives (survival_rate_A survival_rate_B : ℚ) (n_A n_B : ℕ) : ℚ :=
  1 - ((1 - survival_rate_A)^(n_A) * (1 - survival_rate_B)^(n_B))

theorem probability_at_least_one_tree_survives :
  prob_at_least_one_survives (5/6) (4/5) 2 2 = 899 / 900 :=
by
  sorry

end probability_at_least_one_tree_survives_l174_174345


namespace total_players_count_l174_174649

theorem total_players_count (M W : ℕ) (h1 : W = M + 4) (h2 : (M : ℚ) / W = 5 / 9) : M + W = 14 :=
sorry

end total_players_count_l174_174649


namespace find_d10_bills_l174_174523

variable (V : Int) (d10 d20 : Int)

-- Given conditions
def spent_money (d10 d20 : Int) : Int := 10 * d10 + 20 * d20

axiom spent_amount : spent_money d10 d20 = 80
axiom more_20_bills : d20 = d10 + 1

-- Question to prove
theorem find_d10_bills : d10 = 2 :=
by {
  -- We mark the theorem to be proven
  sorry
}

end find_d10_bills_l174_174523


namespace find_C_coordinates_l174_174541

noncomputable def maximize_angle (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) (x : ℝ) : Prop :=
  ∀ C : ℝ × ℝ, C = (x, 0) → x = Real.sqrt (a * b)

theorem find_C_coordinates (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  maximize_angle a b ha hb hab (Real.sqrt (a * b)) :=
by  sorry

end find_C_coordinates_l174_174541


namespace acute_angled_triangle_range_l174_174339

theorem acute_angled_triangle_range (x : ℝ) (h : (x^2 + 6)^2 < (x^2 + 4)^2 + (4 * x)^2) : x > (Real.sqrt 15) / 3 := sorry

end acute_angled_triangle_range_l174_174339


namespace sam_total_pennies_l174_174316

def a : ℕ := 98
def b : ℕ := 93

theorem sam_total_pennies : a + b = 191 :=
by
  sorry

end sam_total_pennies_l174_174316


namespace smallest_d_for_inverse_domain_l174_174997

noncomputable def g (x : ℝ) : ℝ := 2 * (x + 1)^2 - 7

theorem smallest_d_for_inverse_domain : ∃ d : ℝ, (∀ x1 x2 : ℝ, x1 ≥ d → x2 ≥ d → g x1 = g x2 → x1 = x2) ∧ d = -1 :=
by
  use -1
  constructor
  · sorry
  · rfl

end smallest_d_for_inverse_domain_l174_174997


namespace min_Sn_l174_174690

variable {a : ℕ → ℤ}

def arithmetic_sequence (a : ℕ → ℤ) (a₄ : ℤ) (d : ℤ) : Prop :=
  a 4 = a₄ ∧ ∀ n : ℕ, n > 0 → a n = a 1 + (n - 1) * d

def Sn (a : ℕ → ℤ) (n : ℕ) :=
  n / 2 * (2 * a 1 + (n - 1) * 3)

theorem min_Sn (a : ℕ → ℤ) (h1 : arithmetic_sequence a (-15) 3) :
  ∃ n : ℕ, (Sn a n = -108) :=
sorry

end min_Sn_l174_174690


namespace function_monotonically_increasing_range_l174_174088

theorem function_monotonically_increasing_range (a : ℝ) :
  (∀ x y : ℝ, x ≤ 1 ∧ y ≤ 1 ∧ x ≤ y → ((4 - a / 2) * x + 2) ≤ ((4 - a / 2) * y + 2)) ∧
  (∀ x y : ℝ, x > 1 ∧ y > 1 ∧ x ≤ y → a^x ≤ a^y) ∧
  (∀ x : ℝ, if x = 1 then a^1 ≥ (4 - a / 2) * 1 + 2 else true) ↔
  4 ≤ a ∧ a < 8 :=
sorry

end function_monotonically_increasing_range_l174_174088


namespace smaller_of_x_and_y_l174_174342

theorem smaller_of_x_and_y 
  (x y a b c d : ℝ) 
  (h1 : 0 < a) 
  (h2 : a < b + 1) 
  (h3 : x + y = c) 
  (h4 : x - y = d) 
  (h5 : x / y = a / (b + 1)) :
  min x y = (ac/(a + b + 1)) := 
by
  sorry

end smaller_of_x_and_y_l174_174342


namespace students_watching_l174_174010

theorem students_watching (b g : ℕ) (h : b + g = 33) : (2 / 3 : ℚ) * b + (2 / 3 : ℚ) * g = 22 := by
  sorry

end students_watching_l174_174010


namespace green_duck_percentage_l174_174567

noncomputable def smaller_pond_ducks : ℕ := 45
noncomputable def larger_pond_ducks : ℕ := 55
noncomputable def green_percentage_small_pond : ℝ := 0.20
noncomputable def green_percentage_large_pond : ℝ := 0.40

theorem green_duck_percentage :
  let total_ducks := smaller_pond_ducks + larger_pond_ducks
  let green_ducks_smaller := green_percentage_small_pond * (smaller_pond_ducks : ℝ)
  let green_ducks_larger := green_percentage_large_pond * (larger_pond_ducks : ℝ)
  let total_green_ducks := green_ducks_smaller + green_ducks_larger
  (total_green_ducks / total_ducks) * 100 = 31 :=
by {
  -- The proof is omitted.
  sorry
}

end green_duck_percentage_l174_174567


namespace find_nabla_l174_174300

theorem find_nabla : ∀ (nabla : ℤ), 5 * (-4) = nabla + 2 → nabla = -22 :=
by
  intros nabla h
  sorry

end find_nabla_l174_174300


namespace largest_even_number_l174_174642

theorem largest_even_number (n : ℤ) 
    (h1 : (n-6) % 2 = 0) 
    (h2 : (n+6) = 3 * (n-6)) :
    (n + 6) = 18 :=
by
  sorry

end largest_even_number_l174_174642


namespace zoe_bought_8_roses_l174_174379

-- Define the conditions
def each_flower_costs : ℕ := 3
def roses_bought (R : ℕ) : Prop := true
def daisies_bought : ℕ := 2
def total_spent : ℕ := 30

-- The main theorem to prove
theorem zoe_bought_8_roses (R : ℕ) (h1 : total_spent = 30) 
  (h2 : 3 * R + 3 * daisies_bought = total_spent) : R = 8 := by
  sorry

end zoe_bought_8_roses_l174_174379


namespace find_Y_l174_174894

theorem find_Y 
  (a b c d X Y : ℕ)
  (h1 : a + b + c + d = 40)
  (h2 : X + Y + c + b = 40)
  (h3 : a + b + X = 30)
  (h4 : c + d + Y = 30)
  (h5 : X = 9) 
  : Y = 11 := 
by 
  sorry

end find_Y_l174_174894


namespace value_of_expression_l174_174306

theorem value_of_expression (m : ℝ) 
  (h : m^2 - 2 * m - 1 = 0) : 3 * m^2 - 6 * m + 2020 = 2023 := 
by 
  /- Proof is omitted -/
  sorry

end value_of_expression_l174_174306


namespace calculate_paint_area_l174_174387

def barn_length : ℕ := 12
def barn_width : ℕ := 15
def barn_height : ℕ := 6
def window_length : ℕ := 2
def window_width : ℕ := 2

def area_to_paint : ℕ := 796

theorem calculate_paint_area 
    (b_len : ℕ := barn_length) 
    (b_wid : ℕ := barn_width) 
    (b_hei : ℕ := barn_height) 
    (win_len : ℕ := window_length) 
    (win_wid : ℕ := window_width) : 
    b_len = 12 → 
    b_wid = 15 → 
    b_hei = 6 → 
    win_len = 2 → 
    win_wid = 2 →
    area_to_paint = 796 :=
by
  -- Here, the proof would be provided.
  -- This line is a placeholder (sorry) indicating that the proof is yet to be constructed.
  sorry

end calculate_paint_area_l174_174387


namespace perimeter_of_T_shaped_figure_l174_174792

theorem perimeter_of_T_shaped_figure :
  let a := 3    -- width of the horizontal rectangle
  let b := 5    -- height of the horizontal rectangle
  let c := 2    -- width of the vertical rectangle
  let d := 4    -- height of the vertical rectangle
  let overlap := 1 -- overlap length
  2 * a + 2 * b + 2 * c + 2 * d - 2 * overlap = 26 := by
  sorry

end perimeter_of_T_shaped_figure_l174_174792


namespace find_value_of_f2_l174_174414

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x - 1)

theorem find_value_of_f2 : f 2 = 101 / 99 :=
  sorry

end find_value_of_f2_l174_174414


namespace contrapositive_inequality_l174_174737

theorem contrapositive_inequality {x y : ℝ} (h : x^2 ≤ y^2) : x ≤ y :=
  sorry

end contrapositive_inequality_l174_174737


namespace arithmetic_sequence_common_difference_l174_174312

theorem arithmetic_sequence_common_difference
  (a : ℤ)
  (a_n : ℤ)
  (S_n : ℤ)
  (n : ℤ)
  (d : ℚ)
  (h1 : a = 3)
  (h2 : a_n = 34)
  (h3 : S_n = 222)
  (h4 : S_n = n * (a + a_n) / 2)
  (h5 : a_n = a + (n - 1) * d) :
  d = 31 / 11 :=
by
  sorry

end arithmetic_sequence_common_difference_l174_174312


namespace mod_inverse_13_1728_l174_174470

theorem mod_inverse_13_1728 :
  (13 * 133) % 1728 = 1 := by
  sorry

end mod_inverse_13_1728_l174_174470


namespace books_shelved_in_fiction_section_l174_174128

def calculate_books_shelved_in_fiction_section (total_books : ℕ) (remaining_books : ℕ) (books_shelved_in_history : ℕ) (books_shelved_in_children : ℕ) (books_added_back : ℕ) : ℕ :=
  let total_shelved := total_books - remaining_books
  let adjusted_books_shelved_in_children := books_shelved_in_children - books_added_back
  let total_shelved_in_history_and_children := books_shelved_in_history + adjusted_books_shelved_in_children
  total_shelved - total_shelved_in_history_and_children

theorem books_shelved_in_fiction_section:
  calculate_books_shelved_in_fiction_section 51 16 12 8 4 = 19 :=
by 
  -- Definition of the function gives the output directly so proof is trivial.
  rfl

end books_shelved_in_fiction_section_l174_174128


namespace other_liquid_cost_l174_174489

-- Definitions based on conditions
def total_fuel_gallons : ℕ := 12
def fuel_price_per_gallon : ℝ := 8
def oil_price_per_gallon : ℝ := 15
def fuel_cost : ℝ := total_fuel_gallons * fuel_price_per_gallon
def other_liquid_price_per_gallon (x : ℝ) : Prop :=
  (7 * x + 5 * oil_price_per_gallon = fuel_cost) ∨
  (7 * oil_price_per_gallon + 5 * x = fuel_cost)

-- Question: The cost of the other liquid per gallon
theorem other_liquid_cost :
  ∃ x, other_liquid_price_per_gallon x ∧ x = 3 :=
sorry

end other_liquid_cost_l174_174489


namespace count_divisible_by_3_in_range_l174_174966

theorem count_divisible_by_3_in_range (a b : ℤ) :
  a = 252 → b = 549 → (∃ n : ℕ, (a ≤ 3 * n ∧ 3 * n ≤ b) ∧ (b - a) / 3 = (100 : ℝ)) :=
by
  intros ha hb
  have h1 : ∃ k : ℕ, a = 3 * k := by sorry
  have h2 : ∃ m : ℕ, b = 3 * m := by sorry
  sorry

end count_divisible_by_3_in_range_l174_174966


namespace smallest_whole_number_larger_than_triangle_perimeter_l174_174469

theorem smallest_whole_number_larger_than_triangle_perimeter :
  (∀ s : ℝ, 16 < s ∧ s < 30 → ∃ n : ℕ, n = 60) :=
by
  sorry

end smallest_whole_number_larger_than_triangle_perimeter_l174_174469


namespace minimum_value_of_expression_l174_174038

theorem minimum_value_of_expression (x : ℝ) (hx : x > 0) : 6 * x + 1 / x ^ 6 ≥ 7 :=
sorry

end minimum_value_of_expression_l174_174038


namespace integer_solutions_l174_174603

-- Define the polynomial equation as a predicate
def polynomial (n : ℤ) : Prop := n^5 - 2 * n^4 - 7 * n^2 - 7 * n + 3 = 0

-- The theorem statement
theorem integer_solutions :
  {n : ℤ | polynomial n} = {-1, 3} :=
by 
  sorry

end integer_solutions_l174_174603


namespace Cody_total_bill_l174_174938

-- Definitions for the problem
def cost_per_child : ℝ := 7.5
def cost_per_adult : ℝ := 12.0

variables (A C : ℕ)

-- Conditions
def condition1 : Prop := C = A + 8
def condition2 : Prop := A + C = 12

-- Total bill
def total_cost := (A * cost_per_adult) + (C * cost_per_child)

-- The proof statement
theorem Cody_total_bill (h1 : condition1 A C) (h2 : condition2 A C) : total_cost A C = 99.0 := by
  sorry

end Cody_total_bill_l174_174938


namespace minimum_omega_l174_174002

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x)
noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x)
noncomputable def h (ω : ℝ) (x : ℝ) : ℝ := f ω x + g ω x

theorem minimum_omega (ω : ℝ) (m : ℝ) 
  (h1 : 0 < ω)
  (h2 : ∀ x : ℝ, h ω m ≤ h ω x ∧ h ω x ≤ h ω (m + 1)) :
  ω = π :=
by
  sorry

end minimum_omega_l174_174002


namespace total_distance_travelled_eight_boys_on_circle_l174_174190

noncomputable def distance_travelled_by_boys (radius : ℝ) : ℝ :=
  let n := 8
  let angle := 2 * Real.pi / n
  let distance_to_non_adjacent := 2 * radius * Real.sin (2 * angle / 2)
  n * (100 + 3 * distance_to_non_adjacent)

theorem total_distance_travelled_eight_boys_on_circle :
  distance_travelled_by_boys 50 = 800 + 1200 * Real.sqrt 2 :=
  by
    sorry

end total_distance_travelled_eight_boys_on_circle_l174_174190


namespace find_n_l174_174674

theorem find_n : ∀ (n x : ℝ), (3639 + n - x = 3054) → (x = 596.95) → (n = 11.95) :=
by
  intros n x h1 h2
  sorry

end find_n_l174_174674


namespace inequality_solution_l174_174751

theorem inequality_solution :
  { x : ℝ | 0 < x ∧ x ≤ 7/3 ∨ 3 ≤ x } = { x : ℝ | (0 < x ∧ x ≤ 7/3) ∨ 3 ≤ x } :=
sorry

end inequality_solution_l174_174751


namespace segments_have_common_point_l174_174981

-- Define the predicate that checks if two segments intersect
def segments_intersect (seg1 seg2 : ℝ × ℝ) : Prop :=
  let (a1, b1) := seg1
  let (a2, b2) := seg2
  max a1 a2 ≤ min b1 b2

-- Define the main theorem
theorem segments_have_common_point (segments : Fin 2019 → ℝ × ℝ)
  (h_intersect : ∀ (i j : Fin 2019), i ≠ j → segments_intersect (segments i) (segments j)) :
  ∃ p : ℝ, ∀ i : Fin 2019, (segments i).1 ≤ p ∧ p ≤ (segments i).2 :=
by
  sorry

end segments_have_common_point_l174_174981


namespace nineteen_power_six_l174_174513

theorem nineteen_power_six :
    19^11 / 19^5 = 47045881 := by
  sorry

end nineteen_power_six_l174_174513


namespace symmetric_line_condition_l174_174139

theorem symmetric_line_condition (x y : ℝ) :
  (∀ x y : ℝ, x - 2 * y - 3 = 0 → -y + 2 * x - 3 = 0) →
  (∀ x y : ℝ, x + y = 0 → ∃ a b c : ℝ, 2 * x - y - 3 = 0) :=
sorry

end symmetric_line_condition_l174_174139


namespace fourth_proportional_segment_l174_174922

theorem fourth_proportional_segment 
  (a b c : ℝ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) 
  : ∃ x : ℝ, x = (b * c) / a := 
by
  sorry

end fourth_proportional_segment_l174_174922


namespace solution_to_inequality_system_l174_174136

theorem solution_to_inequality_system :
  (∀ x : ℝ, 2 * (x - 1) < x + 2 → (x + 1) / 2 < x → 1 < x ∧ x < 4) :=
by
  intros x h1 h2
  sorry

end solution_to_inequality_system_l174_174136


namespace total_students_in_school_l174_174435

theorem total_students_in_school : 
  let C1 := 32
  let C2 := C1 - 2
  let C3 := C2 - 2
  let C4 := C3 - 2
  let C5 := C4 - 2
  C1 + C2 + C3 + C4 + C5 = 140 :=
by
  let C1 := 32
  let C2 := C1 - 2
  let C3 := C2 - 2
  let C4 := C3 - 2
  let C5 := C4 - 2
  sorry

end total_students_in_school_l174_174435


namespace sum_of_three_squares_l174_174511

theorem sum_of_three_squares (s t : ℤ) (h1 : 3 * s + 2 * t = 27)
                             (h2 : 2 * s + 3 * t = 23) (h3 : s + 2 * t = 13) :
  3 * s = 21 :=
sorry

end sum_of_three_squares_l174_174511


namespace rectangular_prism_lateral_edge_length_l174_174388

-- Definition of the problem conditions
def is_rectangular_prism (v : ℕ) : Prop := v = 8
def sum_lateral_edges (l : ℕ) : ℕ := 4 * l

-- Theorem stating the problem to prove
theorem rectangular_prism_lateral_edge_length :
  ∀ (v l : ℕ), is_rectangular_prism v → sum_lateral_edges l = 56 → l = 14 :=
by
  intros v l h1 h2
  sorry

end rectangular_prism_lateral_edge_length_l174_174388


namespace angle_bisector_b_c_sum_l174_174147

theorem angle_bisector_b_c_sum (A B C : ℝ × ℝ)
  (hA : A = (4, -3))
  (hB : B = (-6, 21))
  (hC : C = (10, 7)) :
  ∃ b c : ℝ, (3 * x + b * y + c = 0) ∧ (b + c = correct_answer) :=
by
  sorry

end angle_bisector_b_c_sum_l174_174147


namespace solve_problem_l174_174916

noncomputable def problem_expression : ℝ :=
  4^(1/2) + Real.log (3^2) / Real.log 3

theorem solve_problem : problem_expression = 4 := by
  sorry

end solve_problem_l174_174916


namespace solve_for_x_l174_174536

theorem solve_for_x (x : ℝ) : 2 * x + 3 * x = 600 - (4 * x + 6 * x) → x = 40 :=
by
  intro h
  sorry

end solve_for_x_l174_174536


namespace trihedral_angle_plane_angles_acute_l174_174169

open Real

-- Define what it means for an angle to be acute
def is_acute (θ : ℝ) : Prop :=
  0 < θ ∧ θ < π / 2

-- Define the given conditions
variable {A B C α β γ : ℝ}
variable (hA : is_acute A)
variable (hB : is_acute B)
variable (hC : is_acute C)

-- State the problem: if dihedral angles are acute, then plane angles are also acute
theorem trihedral_angle_plane_angles_acute :
  is_acute A → is_acute B → is_acute C → is_acute α ∧ is_acute β ∧ is_acute γ :=
sorry

end trihedral_angle_plane_angles_acute_l174_174169


namespace dino_remaining_balance_is_4650_l174_174749

def gigA_hours : Nat := 20
def gigA_rate : Nat := 10

def gigB_hours : Nat := 30
def gigB_rate : Nat := 20

def gigC_hours : Nat := 5
def gigC_rate : Nat := 40

def gigD_hours : Nat := 15
def gigD_rate : Nat := 25

def gigE_hours : Nat := 10
def gigE_rate : Nat := 30

def january_expense : Nat := 500
def february_expense : Nat := 550
def march_expense : Nat := 520
def april_expense : Nat := 480

theorem dino_remaining_balance_is_4650 :
  let gigA_earnings := gigA_hours * gigA_rate
  let gigB_earnings := gigB_hours * gigB_rate
  let gigC_earnings := gigC_hours * gigC_rate
  let gigD_earnings := gigD_hours * gigD_rate
  let gigE_earnings := gigE_hours * gigE_rate

  let total_monthly_earnings := gigA_earnings + gigB_earnings + gigC_earnings + gigD_earnings + gigE_earnings

  let total_expenses := january_expense + february_expense + march_expense + april_expense

  let total_earnings_four_months := total_monthly_earnings * 4

  total_earnings_four_months - total_expenses = 4650 :=
by {
  sorry
}

end dino_remaining_balance_is_4650_l174_174749


namespace equal_costs_l174_174074

noncomputable def cost_scheme_1 (x : ℕ) : ℝ := 350 + 5 * x

noncomputable def cost_scheme_2 (x : ℕ) : ℝ := 360 + 4.5 * x

theorem equal_costs (x : ℕ) : cost_scheme_1 x = cost_scheme_2 x ↔ x = 20 := by
  sorry

end equal_costs_l174_174074


namespace parallelogram_properties_l174_174881

noncomputable def perimeter (x y : ℤ) : ℝ :=
  2 * (5 + Real.sqrt ((x - 7) ^ 2 + (y - 3) ^ 2))

noncomputable def area (x y : ℤ) : ℝ :=
  5 * abs (y - 3)

theorem parallelogram_properties (x y : ℤ) (hx : x = 7) (hy : y = 7) :
  (perimeter x y + area x y) = 38 :=
by
  simp [perimeter, area, hx, hy]
  sorry

end parallelogram_properties_l174_174881


namespace large_doll_cost_is_8_l174_174606

-- Define the cost of the large monkey doll
def cost_large_doll : ℝ := 8

-- Define the total amount spent
def total_spent : ℝ := 320

-- Define the price difference between large and small dolls
def price_difference : ℝ := 4

-- Define the count difference between buying small dolls and large dolls
def count_difference : ℝ := 40

theorem large_doll_cost_is_8 
    (h1 : total_spent = 320)
    (h2 : ∀ L, L - price_difference = 4)
    (h3 : ∀ L, (total_spent / (L - 4)) = (total_spent / L) + count_difference) :
    cost_large_doll = 8 := 
by 
  sorry

end large_doll_cost_is_8_l174_174606


namespace kim_boxes_on_tuesday_l174_174969

theorem kim_boxes_on_tuesday
  (sold_on_thursday : ℕ)
  (sold_on_wednesday : ℕ)
  (sold_on_tuesday : ℕ)
  (h1 : sold_on_thursday = 1200)
  (h2 : sold_on_wednesday = 2 * sold_on_thursday)
  (h3 : sold_on_tuesday = 2 * sold_on_wednesday) :
  sold_on_tuesday = 4800 :=
sorry

end kim_boxes_on_tuesday_l174_174969


namespace chess_group_players_l174_174120

theorem chess_group_players (n : ℕ) (h : n * (n - 1) / 2 = 1225) : n = 50 :=
sorry

end chess_group_players_l174_174120


namespace amount_distributed_l174_174659

theorem amount_distributed (A : ℝ) (h : A / 20 = A / 25 + 120) : A = 12000 :=
by
  sorry

end amount_distributed_l174_174659


namespace intersection_point_in_AB_l174_174296

def A (p : ℝ × ℝ) : Prop := p.snd = 2 * p.fst - 1
def B (p : ℝ × ℝ) : Prop := p.snd = p.fst + 3

theorem intersection_point_in_AB : (4, 7) ∈ {p : ℝ × ℝ | A p} ∩ {p : ℝ × ℝ | B p} :=
by
  sorry

end intersection_point_in_AB_l174_174296


namespace regression_decrease_by_three_l174_174348

-- Define the regression equation
def regression_equation (x : ℝ) : ℝ := 2 - 3 * x

-- Prove that when the explanatory variable increases by 1 unit, the predicted variable decreases by 3 units
theorem regression_decrease_by_three : ∀ x : ℝ, regression_equation (x + 1) = regression_equation x - 3 :=
by
  intro x
  unfold regression_equation
  sorry

end regression_decrease_by_three_l174_174348


namespace square_units_digit_l174_174600

theorem square_units_digit (n : ℕ) (h : (n^2 / 10) % 10 = 7) : n^2 % 10 = 6 := 
sorry

end square_units_digit_l174_174600


namespace ratio_of_running_speed_l174_174069

theorem ratio_of_running_speed (distance : ℝ) (time_jack : ℝ) (time_jill : ℝ) 
  (h_distance_eq : distance = 42) (h_time_jack_eq : time_jack = 6) 
  (h_time_jill_eq : time_jill = 4.2) :
  (distance / time_jack) / (distance / time_jill) = 7 / 10 := by 
  sorry

end ratio_of_running_speed_l174_174069


namespace B_completes_work_in_18_days_l174_174510

variable {A B : ℝ}
variable (x : ℝ)

-- Conditions provided
def A_works_twice_as_fast_as_B (h1 : A = 2 * B) : Prop := true
def together_finish_work_in_6_days (h2 : B = 1 / x ∧ A = 2 / x ∧ 1 / 6 = (A + B)) : Prop := true

-- Theorem to prove: It takes B 18 days to complete the work independently
theorem B_completes_work_in_18_days (h1 : A = 2 * B) (h2 : B = 1 / x ∧ A = 2 / x ∧ 1 / 6 = (A + B)) : x = 18 := by
  sorry

end B_completes_work_in_18_days_l174_174510


namespace increasing_interval_a_geq_neg2_l174_174481

theorem increasing_interval_a_geq_neg2
  (f : ℝ → ℝ)
  (h : ∀ x, f x = x^2 + 2 * (a - 2) * x + 5)
  (h_inc : ∀ x > 4, f (x + 1) > f x) :
  a ≥ -2 :=
sorry

end increasing_interval_a_geq_neg2_l174_174481


namespace ratio_of_full_boxes_l174_174864

theorem ratio_of_full_boxes 
  (F H : ℕ)
  (boxes_count_eq : F + H = 20)
  (parsnips_count_eq : 20 * F + 10 * H = 350) :
  F / (F + H) = 3 / 4 := 
by
  -- proof will be placed here
  sorry

end ratio_of_full_boxes_l174_174864


namespace trapezoid_diagonals_l174_174544

theorem trapezoid_diagonals (a c b d e f : ℝ) (h1 : a ≠ c):
  e^2 = a * c + (a * d^2 - c * b^2) / (a - c) ∧ 
  f^2 = a * c + (a * b^2 - c * d^2) / (a - c) := 
by
  sorry

end trapezoid_diagonals_l174_174544


namespace solve_system_l174_174931

def system_of_equations_solution : Prop :=
  ∃ (x y : ℚ), 4 * x - 7 * y = -9 ∧ 5 * x + 3 * y = -11 ∧ (x, y) = (-(104 : ℚ) / 47, (1 : ℚ) / 47)

theorem solve_system : system_of_equations_solution :=
sorry

end solve_system_l174_174931


namespace cos_seven_pi_over_four_l174_174238

theorem cos_seven_pi_over_four :
  Real.cos (7 * Real.pi / 4) = Real.sqrt 2 / 2 :=
by
  sorry

end cos_seven_pi_over_four_l174_174238


namespace intersection_A_B_l174_174766

def setA : Set ℤ := { x | x < -3 }
def setB : Set ℤ := {-5, -4, -3, 1}

theorem intersection_A_B : setA ∩ setB = {-5, -4} := by
  sorry

end intersection_A_B_l174_174766


namespace problem_l174_174358

-- Definitions
def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {2, 3}
def complement_U (S : Set ℕ) : Set ℕ := U \ S

theorem problem : ((complement_U M) ∩ (complement_U N)) = {5} :=
by
  sorry

end problem_l174_174358


namespace candy_bar_price_l174_174164

theorem candy_bar_price (total_money bread_cost candy_bar_price remaining_money : ℝ) 
    (h1 : total_money = 32)
    (h2 : bread_cost = 3)
    (h3 : remaining_money = 18)
    (h4 : total_money - bread_cost - candy_bar_price - (1 / 3) * (total_money - bread_cost - candy_bar_price) = remaining_money) :
    candy_bar_price = 1.33 := 
sorry

end candy_bar_price_l174_174164


namespace angle_is_30_degrees_l174_174313

theorem angle_is_30_degrees (A : ℝ) (h_acute : A > 0 ∧ A < π / 2) (h_sin : Real.sin A = 1/2) : A = π / 6 := 
by 
  sorry

end angle_is_30_degrees_l174_174313


namespace no_intersection_points_l174_174604

theorem no_intersection_points :
  ∀ x y : ℝ, y = abs (3 * x + 6) ∧ y = -2 * abs (2 * x - 1) → false :=
by
  intros x y h
  cases h
  sorry

end no_intersection_points_l174_174604


namespace abs_neg_six_l174_174085

theorem abs_neg_six : abs (-6) = 6 :=
by
  -- Proof goes here
  sorry

end abs_neg_six_l174_174085


namespace mod_product_l174_174646

theorem mod_product (n : ℕ) (h1 : 0 ≤ n) (h2 : n < 50) : 
  173 * 927 % 50 = n := 
  by
    sorry

end mod_product_l174_174646


namespace sum_of_legs_equal_l174_174656

theorem sum_of_legs_equal
  (a b c d e f g h : ℝ)
  (x y : ℝ)
  (h_similar_shaded1 : a = a * x ∧ b = a * y)
  (h_similar_shaded2 : c = c * x ∧ d = c * y)
  (h_similar_shaded3 : e = e * x ∧ f = e * y)
  (h_similar_shaded4 : g = g * x ∧ h = g * y)
  (h_similar_unshaded1 : h = h * x ∧ a = h * y)
  (h_similar_unshaded2 : b = b * x ∧ c = b * y)
  (h_similar_unshaded3 : d = d * x ∧ e = d * y)
  (h_similar_unshaded4 : f = f * x ∧ g = f * y)
  (x_non_zero : x ≠ 0) (y_non_zero : y ≠ 0) : 
  (a * y + b + c * x) + (c * y + d + e * x) + (e * y + f + g * x) + (g * y + h + a * x) 
  = (h * x + a + b * y) + (b * x + c + d * y) + (d * x + e + f * y) + (f * x + g + h * y) :=
sorry

end sum_of_legs_equal_l174_174656


namespace find_value_l174_174681

theorem find_value
  (y1 y2 y3 y4 y5 : ℝ)
  (h1 : y1 + 4 * y2 + 9 * y3 + 16 * y4 + 25 * y5 = 3)
  (h2 : 4 * y1 + 9 * y2 + 16 * y3 + 25 * y4 + 36 * y5 = 20)
  (h3 : 9 * y1 + 16 * y2 + 25 * y3 + 36 * y4 + 49 * y5 = 150) :
  16 * y1 + 25 * y2 + 36 * y3 + 49 * y4 + 64 * y5 = 336 :=
by
  sorry

end find_value_l174_174681


namespace height_of_the_carton_l174_174677

noncomputable def carton_height : ℕ :=
  let carton_length := 25
  let carton_width := 42
  let soap_box_length := 7
  let soap_box_width := 6
  let soap_box_height := 10
  let max_soap_boxes := 150
  let boxes_per_row := carton_length / soap_box_length
  let boxes_per_column := carton_width / soap_box_width
  let boxes_per_layer := boxes_per_row * boxes_per_column
  let layers := max_soap_boxes / boxes_per_layer
  layers * soap_box_height

theorem height_of_the_carton :
  carton_height = 70 :=
by
  -- The computation and necessary assumptions for proving the height are encapsulated above.
  sorry

end height_of_the_carton_l174_174677


namespace xy_sum_of_squares_l174_174052

theorem xy_sum_of_squares (x y : ℝ) (h1 : x - y = 18) (h2 : x + y = 22) : x^2 + y^2 = 404 := by
  sorry

end xy_sum_of_squares_l174_174052


namespace opposite_of_2023_l174_174808

def opposite (n : ℤ) := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l174_174808


namespace number_of_trees_l174_174432

theorem number_of_trees (n : ℕ) (diff : ℕ) (count1 : ℕ) (count2 : ℕ) (timur1 : ℕ) (alexander1 : ℕ) (timur2 : ℕ) (alexander2 : ℕ) : 
  diff = alexander1 - timur1 ∧
  count1 = timur2 + (alexander2 - timur1) ∧
  n = count1 + diff →
  n = 118 :=
by
  sorry

end number_of_trees_l174_174432


namespace problem_statement_l174_174366

variable (p q : ℝ)

def condition := p ^ 2 / q ^ 3 = 4 / 5

theorem problem_statement (hpq : condition p q) : 11 / 7 + (2 * q ^ 3 - p ^ 2) / (2 * q ^ 3 + p ^ 2) = 2 :=
sorry

end problem_statement_l174_174366


namespace equation_solution_l174_174110

def solve_equation (x : ℝ) : Prop :=
  ((3 * x + 6) / (x^2 + 5 * x - 6) = (4 - x) / (x - 2)) ↔ 
  x = -3 ∨ x = (1 + Real.sqrt 17) / 2 ∨ x = (1 - Real.sqrt 17) / 2

theorem equation_solution (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -6) (h3 : x ≠ 2) : solve_equation x :=
by
  sorry

end equation_solution_l174_174110


namespace part_a_part_b_l174_174228

open Nat

theorem part_a (n: ℕ) (h_pos: 0 < n) : (2^n - 1) % 7 = 0 ↔ ∃ k : ℕ, k > 0 ∧ n = 3 * k :=
sorry

theorem part_b (n: ℕ) (h_pos: 0 < n) : (2^n + 1) % 7 ≠ 0 :=
sorry

end part_a_part_b_l174_174228


namespace combined_molecular_weight_l174_174459

def atomic_weight_C : ℝ := 12.01
def atomic_weight_Cl : ℝ := 35.45
def atomic_weight_S : ℝ := 32.07
def atomic_weight_F : ℝ := 19.00

def molecular_weight_CCl4 : ℝ := atomic_weight_C + 4 * atomic_weight_Cl
def molecular_weight_SF6 : ℝ := atomic_weight_S + 6 * atomic_weight_F

def weight_moles_CCl4 (moles : ℝ) : ℝ := moles * molecular_weight_CCl4
def weight_moles_SF6 (moles : ℝ) : ℝ := moles * molecular_weight_SF6

theorem combined_molecular_weight : weight_moles_CCl4 9 + weight_moles_SF6 5 = 2114.64 := by
  sorry

end combined_molecular_weight_l174_174459


namespace largest_angle_of_consecutive_integer_angles_of_hexagon_l174_174504

theorem largest_angle_of_consecutive_integer_angles_of_hexagon 
  (angles : Fin 6 → ℝ)
  (h_consecutive : ∃ (x : ℝ), angles = ![
    x - 3, x - 2, x - 1, x, x + 1, x + 2 ])
  (h_sum : (angles 0 + angles 1 + angles 2 + angles 3 + angles 4 + angles 5) = 720) :
  (angles 5 = 122.5) :=
by
  sorry

end largest_angle_of_consecutive_integer_angles_of_hexagon_l174_174504


namespace range_of_a_l174_174876

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x - 1| + |x - 2| ≤ a) ↔ a ≥ 1 :=
sorry

end range_of_a_l174_174876


namespace quadratic_function_min_value_l174_174838

theorem quadratic_function_min_value :
  ∃ x, ∀ y, 5 * x^2 - 15 * x + 2 ≤ 5 * y^2 - 15 * y + 2 ∧ (5 * x^2 - 15 * x + 2 = -9.25) :=
by
  sorry

end quadratic_function_min_value_l174_174838


namespace solve_quadratic_equation1_solve_quadratic_equation2_l174_174870

theorem solve_quadratic_equation1 (x : ℝ) : x^2 - 4 * x - 1 = 0 ↔ x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5 := 
by sorry

theorem solve_quadratic_equation2 (x : ℝ) : (x + 3) * (x - 3) = 3 * (x + 3) ↔ x = -3 ∨ x = 6 :=
by sorry

end solve_quadratic_equation1_solve_quadratic_equation2_l174_174870


namespace correctness_of_propositions_l174_174663

-- Definitions of the conditions
def residual_is_random_error (e : ℝ) : Prop := ∃ (y : ℝ) (y_hat : ℝ), e = y - y_hat
def data_constraints (a b c d : ℕ) : Prop := a ≥ 5 ∧ b ≥ 5 ∧ c ≥ 5 ∧ d ≥ 5
def histogram_judgement : Prop := ∀ (H : Type) (rel : H → H → Prop), ¬(H ≠ H) ∨ (∀ x y : H, rel x y ↔ true)

-- The mathematical equivalence proof problem
theorem correctness_of_propositions (e : ℝ) (a b c d : ℕ) : 
  (residual_is_random_error e → false) ∧
  (data_constraints a b c d → true) ∧
  (histogram_judgement → true) :=
by
  sorry

end correctness_of_propositions_l174_174663


namespace sum_of_squares_l174_174652

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 21) (h2 : x * y = 43) : x^2 + y^2 = 355 :=
sorry

end sum_of_squares_l174_174652


namespace find_n_for_primes_l174_174850

def A_n (n : ℕ) : ℕ := 1 + 7 * (10^n - 1) / 9
def B_n (n : ℕ) : ℕ := 3 + 7 * (10^n - 1) / 9

theorem find_n_for_primes (n : ℕ) :
  (∀ n, n > 0 → (Nat.Prime (A_n n) ∧ Nat.Prime (B_n n)) ↔ n = 1) :=
sorry

end find_n_for_primes_l174_174850


namespace surface_area_is_correct_volume_is_approximately_correct_l174_174413

noncomputable def surface_area_of_CXYZ (height : ℝ) (side_length : ℝ) : ℝ :=
  let area_CZX_CZY := 48
  let area_CXY := 9 * Real.sqrt 3
  let area_XYZ := 9 * Real.sqrt 15
  2 * area_CZX_CZY + area_CXY + area_XYZ

theorem surface_area_is_correct (height : ℝ) (side_length : ℝ) (h : height = 24) (s : side_length = 18) :
  surface_area_of_CXYZ height side_length = 96 + 9 * Real.sqrt 3 + 9 * Real.sqrt 15 :=
by
  sorry

noncomputable def volume_of_CXYZ (height : ℝ ) (side_length : ℝ) : ℝ :=
  -- Placeholder for the volume calculation approximation method.
  486

theorem volume_is_approximately_correct
  (height : ℝ) (side_length : ℝ) (h : height = 24) (s : side_length = 18) :
  volume_of_CXYZ height side_length = 486 :=
by
  sorry

end surface_area_is_correct_volume_is_approximately_correct_l174_174413


namespace parity_equivalence_l174_174143

def p_q_parity_condition (p q : ℕ) : Prop :=
  (p^3 - q^3) % 2 = 0 ↔ (p + q) % 2 = 0

theorem parity_equivalence (p q : ℕ) : p_q_parity_condition p q :=
by sorry

end parity_equivalence_l174_174143


namespace dust_storm_acres_l174_174852

def total_acres : ℕ := 64013
def untouched_acres : ℕ := 522
def dust_storm_covered : ℕ := total_acres - untouched_acres

theorem dust_storm_acres :
  dust_storm_covered = 63491 := by
  sorry

end dust_storm_acres_l174_174852


namespace max_sum_of_ten_consecutive_in_hundred_l174_174765

theorem max_sum_of_ten_consecutive_in_hundred :
  ∀ (s : Fin 100 → ℕ), (∀ i : Fin 100, 1 ≤ s i ∧ s i ≤ 100) → 
  (∃ i : Fin 91, (s i + s (i + 1) + s (i + 2) + s (i + 3) +
  s (i + 4) + s (i + 5) + s (i + 6) + s (i + 7) + s (i + 8) + s (i + 9)) ≥ 505) :=
by
  intro s hs
  sorry

end max_sum_of_ten_consecutive_in_hundred_l174_174765


namespace prime_sum_eq_14_l174_174977

theorem prime_sum_eq_14 (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h : q^5 - 2 * p^2 = 1) : p + q = 14 := 
sorry

end prime_sum_eq_14_l174_174977


namespace least_five_digit_integer_congruent_3_mod_17_l174_174004

theorem least_five_digit_integer_congruent_3_mod_17 : 
  ∃ n, n ≥ 10000 ∧ n % 17 = 3 ∧ ∀ m, (m ≥ 10000 ∧ m % 17 = 3) → n ≤ m := 
sorry

end least_five_digit_integer_congruent_3_mod_17_l174_174004


namespace linear_function_passing_origin_l174_174290

theorem linear_function_passing_origin (m : ℝ) :
  (∃ (y x : ℝ), y = -2 * x + (m - 5) ∧ y = 0 ∧ x = 0) → m = 5 :=
by
  sorry

end linear_function_passing_origin_l174_174290


namespace projectile_reaches_100_feet_l174_174960

noncomputable def projectile_height (t : ℝ) : ℝ :=
  -16 * t^2 + 80 * t

theorem projectile_reaches_100_feet :
  ∃ t : ℝ, t = 2.5 ∧ projectile_height t = 100 :=
by
  use 2.5
  sorry

end projectile_reaches_100_feet_l174_174960


namespace taxable_income_l174_174372

theorem taxable_income (tax_paid : ℚ) (state_tax_rate : ℚ) (months_resident : ℚ) (total_months : ℚ) (T : ℚ) :
  tax_paid = 1275 ∧ state_tax_rate = 0.04 ∧ months_resident = 9 ∧ total_months = 12 → 
  T = 42500 :=
by
  intros h
  sorry

end taxable_income_l174_174372


namespace negation_of_universal_l174_174337

theorem negation_of_universal:
  ¬(∀ x : ℕ, x^2 > 1) ↔ ∃ x : ℕ, x^2 ≤ 1 :=
by sorry

end negation_of_universal_l174_174337


namespace reciprocal_of_2016_is_1_div_2016_l174_174879

theorem reciprocal_of_2016_is_1_div_2016 : (2016 * (1 / 2016) = 1) :=
by
  sorry

end reciprocal_of_2016_is_1_div_2016_l174_174879


namespace total_inflation_time_l174_174500

theorem total_inflation_time (time_per_ball : ℕ) (alexia_balls : ℕ) (extra_balls : ℕ) : 
  time_per_ball = 20 → alexia_balls = 20 → extra_balls = 5 →
  (alexia_balls * time_per_ball) + ((alexia_balls + extra_balls) * time_per_ball) = 900 :=
by 
  intros h1 h2 h3
  sorry

end total_inflation_time_l174_174500


namespace seeds_distributed_equally_l174_174964

theorem seeds_distributed_equally (S G n seeds_per_small_garden : ℕ) 
  (hS : S = 42) 
  (hG : G = 36) 
  (hn : n = 3) 
  (h_seeds : seeds_per_small_garden = (S - G) / n) : 
  seeds_per_small_garden = 2 := by
  rw [hS, hG, hn] at h_seeds
  simp at h_seeds
  exact h_seeds

end seeds_distributed_equally_l174_174964


namespace total_plums_l174_174174

def alyssa_plums : Nat := 17
def jason_plums : Nat := 10

theorem total_plums : alyssa_plums + jason_plums = 27 := 
by
  -- proof goes here
  sorry

end total_plums_l174_174174


namespace inequality_holds_l174_174639

theorem inequality_holds (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by
  sorry

end inequality_holds_l174_174639


namespace volleyball_team_math_count_l174_174615

theorem volleyball_team_math_count (total_players taking_physics taking_both : ℕ) 
  (h1 : total_players = 30) 
  (h2 : taking_physics = 15) 
  (h3 : taking_both = 6) 
  (h4 : total_players = 30 ∧ total_players = (taking_physics + (total_players - taking_physics - taking_both))) 
  : (total_players - (taking_physics - taking_both) + taking_both) = 21 := 
by
  sorry

end volleyball_team_math_count_l174_174615


namespace top_quality_soccer_balls_l174_174209

theorem top_quality_soccer_balls (N : ℕ) (f : ℝ) (hN : N = 10000) (hf : f = 0.975) : N * f = 9750 := by
  sorry

end top_quality_soccer_balls_l174_174209


namespace escalator_length_l174_174242

theorem escalator_length :
  ∃ L : ℝ, L = 150 ∧ 
    (∀ t : ℝ, t = 10 → ∀ v_p : ℝ, v_p = 3 → ∀ v_e : ℝ, v_e = 12 → L = (v_p + v_e) * t) :=
by sorry

end escalator_length_l174_174242


namespace set_subset_l174_174802

-- Define the sets M and N
def M := {x : ℝ | abs x ≤ 1}
def N := {y : ℝ | ∃ x : ℝ, y = 2^x ∧ x ≤ 0}

-- The mathematical statement to be proved
theorem set_subset : N ⊆ M := sorry

end set_subset_l174_174802


namespace min_value_g_geq_6_min_value_g_eq_6_l174_174156

noncomputable def g (x : ℝ) : ℝ :=
  x + (x / (x^2 + 2)) + (x * (x + 5) / (x^2 + 3)) + (3 * (x + 3) / (x * (x^2 + 3)))

theorem min_value_g_geq_6 : ∀ x > 0, g x ≥ 6 :=
by
  sorry

theorem min_value_g_eq_6 : ∃ x > 0, g x = 6 :=
by
  sorry

end min_value_g_geq_6_min_value_g_eq_6_l174_174156


namespace mismatching_socks_l174_174877

theorem mismatching_socks (total_socks : ℕ) (pairs : ℕ) (socks_per_pair : ℕ) 
  (h1 : total_socks = 25) (h2 : pairs = 4) (h3 : socks_per_pair = 2) : 
  total_socks - (socks_per_pair * pairs) = 17 :=
by
  sorry

end mismatching_socks_l174_174877


namespace anthony_solve_l174_174570

def completing_square (a b c : ℤ) : ℤ :=
  let d := Int.sqrt a
  let e := b / (2 * d)
  let f := (d * e * e - c)
  d + e + f

theorem anthony_solve (d e f : ℤ) (h_d_pos : d > 0)
  (h_eqn : 25 * d * d + 30 * d * e - 72 = 0)
  (h_form : (d * x + e)^2 = f) : 
  d + e + f = 89 :=
by
  have d : ℤ := 5
  have e : ℤ := 3
  have f : ℤ := 81
  sorry

end anthony_solve_l174_174570


namespace oliver_final_amount_l174_174569

def initial_amount : ℤ := 33
def spent : ℤ := 4
def received : ℤ := 32

def final_amount (initial_amount spent received : ℤ) : ℤ :=
  initial_amount - spent + received

theorem oliver_final_amount : final_amount initial_amount spent received = 61 := 
by sorry

end oliver_final_amount_l174_174569


namespace find_a_l174_174841

theorem find_a (a : ℝ) (h_pos : 0 < a) (h_neq_one : a ≠ 1)
  (h_diff : |a^2 - a| = 6) : a = 3 :=
sorry

end find_a_l174_174841


namespace total_students_in_class_l174_174022

theorem total_students_in_class 
  (avg_age_all : ℝ)
  (num_students1 : ℕ) (avg_age1 : ℝ)
  (num_students2 : ℕ) (avg_age2 : ℝ)
  (age_student17 : ℕ)
  (total_students : ℕ) :
  avg_age_all = 17 →
  num_students1 = 5 →
  avg_age1 = 14 →
  num_students2 = 9 →
  avg_age2 = 16 →
  age_student17 = 75 →
  total_students = num_students1 + num_students2 + 1 →
  total_students = 17 :=
by
  intro h_avg_all h_num1 h_avg1 h_num2 h_avg2 h_age17 h_total
  -- Additional proof steps would go here
  sorry

end total_students_in_class_l174_174022


namespace math_proof_problem_l174_174697

open Nat

noncomputable def number_of_pairs := 
  let N := 20^19
  let num_divisors := (38 + 1) * (19 + 1)
  let total_pairs := num_divisors * num_divisors
  let ab_dividing_pairs := 780 * 210
  total_pairs - ab_dividing_pairs

theorem math_proof_problem : number_of_pairs = 444600 := 
  by exact sorry

end math_proof_problem_l174_174697


namespace slope_of_line_l174_174393

theorem slope_of_line : 
  ∀ (x1 y1 x2 y2 : ℝ), 
  x1 = 1 → y1 = 3 → x2 = 6 → y2 = -7 → 
  (x1 ≠ x2) → ((y2 - y1) / (x2 - x1) = -2) :=
by
  intros x1 y1 x2 y2 hx1 hy1 hx2 hy2 hx1_ne_x2
  rw [hx1, hy1, hx2, hy2]
  sorry

end slope_of_line_l174_174393


namespace circles_radius_difference_l174_174041

variable (s : ℝ)

theorem circles_radius_difference (h : (π * (2*s)^2) / (π * s^2) = 4) : (2 * s - s) = s :=
by
  sorry

end circles_radius_difference_l174_174041


namespace equation_holds_iff_b_eq_c_l174_174374

theorem equation_holds_iff_b_eq_c (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) (h4 : a ≠ b) (h5 : a ≠ c) (h6 : b ≠ c) :
  (10 * a + b + 1) * (10 * a + c) = 100 * a * a + 100 * a + b + c ↔ b = c :=
by sorry

end equation_holds_iff_b_eq_c_l174_174374


namespace largest_five_digit_congruent_to_18_mod_25_l174_174851

theorem largest_five_digit_congruent_to_18_mod_25 : 
  ∃ (x : ℕ), x < 100000 ∧ 10000 ≤ x ∧ x % 25 = 18 ∧ x = 99993 :=
by
  sorry

end largest_five_digit_congruent_to_18_mod_25_l174_174851


namespace domain_of_f_l174_174829

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.log (x + 1)) + Real.sqrt (4 - x^2)

theorem domain_of_f : 
  {x : ℝ | x > -1 ∧ x ≤ 2 ∧ x ≠ 0 ∧ 4 - x^2 ≥ 0} = {x : ℝ | (-1 < x ∧ x < 0) ∨ (0 < x ∧ x ≤ 2)} :=
by
  sorry

end domain_of_f_l174_174829


namespace ratio_of_football_to_hockey_l174_174224

variables (B F H s : ℕ)

-- Definitions from conditions
def condition1 : Prop := B = F - 50
def condition2 : Prop := F = s * H
def condition3 : Prop := H = 200
def condition4 : Prop := B + F + H = 1750

-- Proof statement
theorem ratio_of_football_to_hockey (B F H s : ℕ) 
  (h1 : condition1 B F)
  (h2 : condition2 F s H)
  (h3 : condition3 H)
  (h4 : condition4 B F H) : F / H = 4 :=
sorry

end ratio_of_football_to_hockey_l174_174224


namespace rational_solutions_k_l174_174168

theorem rational_solutions_k (k : ℕ) (h : k > 0) : (∃ x : ℚ, 2 * (k : ℚ) * x^2 + 36 * x + 3 * (k : ℚ) = 0) → k = 6 :=
by
  -- proof to be written
  sorry

end rational_solutions_k_l174_174168


namespace math_problem_proof_l174_174362

-- Define the fractions involved
def frac1 : ℚ := -49
def frac2 : ℚ := 4 / 7
def frac3 : ℚ := -8 / 7

-- The original expression
def original_expr : ℚ :=
  frac1 * frac2 - frac2 / frac3

-- Declare the theorem to be proved
theorem math_problem_proof : original_expr = -27.5 :=
by
  sorry

end math_problem_proof_l174_174362


namespace no_nat_nums_satisfy_gcd_lcm_condition_l174_174359

theorem no_nat_nums_satisfy_gcd_lcm_condition :
  ¬ ∃ (x y : ℕ), Nat.gcd x y + Nat.lcm x y = x + y + 2021 := 
sorry

end no_nat_nums_satisfy_gcd_lcm_condition_l174_174359


namespace arithmetic_series_sum_correct_l174_174142

-- Define the parameters of the arithmetic series
def a : ℤ := -53
def l : ℤ := 3
def d : ℤ := 2

-- Define the number of terms in the series
def n : ℕ := 29

-- The expected sum of the series
def expected_sum : ℤ := -725

-- Define the nth term formula
noncomputable def nth_term (a d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

-- Define the sum of the arithmetic series
noncomputable def arithmetic_series_sum (a l : ℤ) (n : ℕ) : ℤ :=
  (n * (a + l)) / 2

-- Statement of the proof problem
theorem arithmetic_series_sum_correct :
  arithmetic_series_sum a l n = expected_sum := by
  sorry

end arithmetic_series_sum_correct_l174_174142


namespace initial_average_weight_l174_174763

theorem initial_average_weight
  (A : ℚ) -- Define A as a rational number since we are dealing with division 
  (h1 : 6 * A + 133 = 7 * 151) : -- Condition from the problem translated into an equation
  A = 154 := -- Statement we need to prove
by
  sorry -- Placeholder for the proof

end initial_average_weight_l174_174763


namespace find_f2_l174_174704

variable (a b : ℝ)

def f (x : ℝ) : ℝ := a * x^3 + b * x + 1

theorem find_f2 (h : f a b (-2) = 3) : f a b 2 = -1 :=
by
  sorry

end find_f2_l174_174704


namespace sequence_general_term_l174_174402

noncomputable def a₁ : ℕ → ℚ := sorry

variable (S : ℕ → ℚ)

axiom h₀ : a₁ 1 = -1
axiom h₁ : ∀ n : ℕ, a₁ (n + 1) = S n * S (n + 1)

theorem sequence_general_term (n : ℕ) : S n = -1 / n := by
  sorry

end sequence_general_term_l174_174402


namespace identity_of_polynomials_l174_174352

theorem identity_of_polynomials (a b : ℝ) : 
  (2 * x + a)^3 = 
  5 * x^3 + (3 * x + b) * (x^2 - x - 1) - 10 * x^2 + 10 * x 
  → a = -1 ∧ b = 1 := 
by 
  sorry

end identity_of_polynomials_l174_174352


namespace find_OP_l174_174637

variable (a b c d e f : ℝ)
variable (P : ℝ)

-- Given conditions
axiom AP_PD_ratio : (a - P) / (P - d) = 2 / 3
axiom BP_PC_ratio : (b - P) / (P - c) = 3 / 4

-- Conclusion to prove
theorem find_OP : P = (3 * a + 2 * d) / 5 :=
by
  sorry

end find_OP_l174_174637


namespace arcsin_double_angle_identity_l174_174295

open Real

theorem arcsin_double_angle_identity (x θ : ℝ) (h₁ : -1 ≤ x) (h₂ : x ≤ 1) (h₃ : arcsin x = θ) (h₄ : -π / 2 ≤ θ) (h₅ : θ ≤ -π / 4) :
    arcsin (2 * x * sqrt (1 - x^2)) = -(π + 2 * θ) := by
  sorry

end arcsin_double_angle_identity_l174_174295


namespace largest_digit_to_correct_sum_l174_174976

theorem largest_digit_to_correct_sum :
  (725 + 864 + 991 = 2570) → (∃ (d : ℕ), d = 9 ∧ 
  (∃ (n1 : ℕ), n1 ∈ [702, 710, 711, 721, 715] ∧ 
  ∃ (n2 : ℕ), n2 ∈ [806, 805, 814, 854, 864] ∧ 
  ∃ (n3 : ℕ), n3 ∈ [918, 921, 931, 941, 981, 991] ∧ 
  n1 + n2 + n3 = n1 + n2 + n3 - 10))
    → d = 9 :=
by
  sorry

end largest_digit_to_correct_sum_l174_174976


namespace Xiaobing_jumps_189_ropes_per_minute_l174_174702

-- Define conditions and variables
variable (x : ℕ) -- The number of ropes Xiaohan jumps per minute

-- Conditions:
-- 1. Xiaobing jumps x + 21 ropes per minute
-- 2. Time taken for Xiaobing to jump 135 ropes is the same as the time taken for Xiaohan to jump 120 ropes

theorem Xiaobing_jumps_189_ropes_per_minute (h : 135 * x = 120 * (x + 21)) :
    x + 21 = 189 :=
by
  sorry -- Proof is not required as per instructions

end Xiaobing_jumps_189_ropes_per_minute_l174_174702


namespace simplify_expression_l174_174517

theorem simplify_expression (x : ℝ) : 
  (2 * x - 3 * (2 + x) + 4 * (2 - x) - 5 * (2 + 3 * x)) = -20 * x - 8 :=
by
  sorry

end simplify_expression_l174_174517


namespace unique_element_set_l174_174385

theorem unique_element_set (a : ℝ) : 
  (∃! x, (a - 1) * x^2 + 3 * x - 2 = 0) ↔ (a = 1 ∨ a = -1 / 8) :=
by sorry

end unique_element_set_l174_174385


namespace total_packs_of_groceries_l174_174462

-- Definitions for the conditions
def packs_of_cookies : ℕ := 2
def packs_of_cake : ℕ := 12

-- Theorem stating the total packs of groceries
theorem total_packs_of_groceries : packs_of_cookies + packs_of_cake = 14 :=
by sorry

end total_packs_of_groceries_l174_174462


namespace exists_N_minimal_l174_174328

-- Assuming m and n are positive and coprime
variables (m n : ℕ)
variables (h_pos_m : 0 < m) (h_pos_n : 0 < n)
variables (h_coprime : Nat.gcd m n = 1)

-- Statement of the mathematical problem
theorem exists_N_minimal :
  ∃ N : ℕ, (∀ k : ℕ, k ≥ N → ∃ a b : ℕ, k = a * m + b * n) ∧
           (N = m * n - m - n + 1) := 
  sorry

end exists_N_minimal_l174_174328


namespace value_of_M_l174_174336

theorem value_of_M
  (M : ℝ)
  (h : 25 / 100 * M = 55 / 100 * 4500) :
  M = 9900 :=
sorry

end value_of_M_l174_174336


namespace fraction_of_number_is_three_quarters_l174_174438

theorem fraction_of_number_is_three_quarters 
  (f : ℚ) 
  (h1 : 76 ≠ 0) 
  (h2 : f * 76 = 76 - 19) : 
  f = 3 / 4 :=
by
  sorry

end fraction_of_number_is_three_quarters_l174_174438


namespace chess_tournament_time_spent_l174_174232

theorem chess_tournament_time_spent (games : ℕ) (moves_per_game : ℕ)
  (opening_moves : ℕ) (middle_moves : ℕ) (endgame_moves : ℕ)
  (polly_opening_time : ℝ) (peter_opening_time : ℝ)
  (polly_middle_time : ℝ) (peter_middle_time : ℝ)
  (polly_endgame_time : ℝ) (peter_endgame_time : ℝ)
  (total_time_hours : ℝ) :
  games = 4 →
  moves_per_game = 38 →
  opening_moves = 12 →
  middle_moves = 18 →
  endgame_moves = 8 →
  polly_opening_time = 35 →
  peter_opening_time = 45 →
  polly_middle_time = 30 →
  peter_middle_time = 45 →
  polly_endgame_time = 40 →
  peter_endgame_time = 60 →
  total_time_hours = (4 * ((12 * 35 + 18 * 30 + 8 * 40) + (12 * 45 + 18 * 45 + 8 * 60))) / 3600 :=
sorry

end chess_tournament_time_spent_l174_174232


namespace factorize_expression_l174_174040

-- Variables used in the expression
variables (m n : ℤ)

-- The expression to be factored
def expr := 4 * m^3 * n - 16 * m * n^3

-- The desired factorized form of the expression
def factored := 4 * m * n * (m + 2 * n) * (m - 2 * n)

-- The proof problem statement
theorem factorize_expression : expr m n = factored m n :=
by sorry

end factorize_expression_l174_174040


namespace original_cost_of_article_l174_174791

theorem original_cost_of_article (x: ℝ) (h: 0.76 * x = 320) : x = 421.05 :=
sorry

end original_cost_of_article_l174_174791


namespace jake_third_test_marks_l174_174856

theorem jake_third_test_marks 
  (avg_marks : ℕ)
  (marks_test1 : ℕ)
  (marks_test2 : ℕ)
  (marks_test3 : ℕ)
  (marks_test4 : ℕ)
  (h_avg : avg_marks = 75)
  (h_test1 : marks_test1 = 80)
  (h_test2 : marks_test2 = marks_test1 + 10)
  (h_test3_eq_test4 : marks_test3 = marks_test4)
  (h_total : avg_marks * 4 = marks_test1 + marks_test2 + marks_test3 + marks_test4) : 
  marks_test3 = 65 :=
sorry

end jake_third_test_marks_l174_174856


namespace platform_length_l174_174638

theorem platform_length
    (train_length : ℕ)
    (time_to_cross_tree : ℕ)
    (speed : ℕ)
    (time_to_pass_platform : ℕ)
    (platform_length : ℕ) :
    train_length = 1200 →
    time_to_cross_tree = 120 →
    speed = train_length / time_to_cross_tree →
    time_to_pass_platform = 150 →
    speed * time_to_pass_platform = train_length + platform_length →
    platform_length = 300 :=
by
  intros h_train_length h_time_to_cross_tree h_speed h_time_to_pass_platform h_pass_platform_eq
  sorry

end platform_length_l174_174638


namespace find_length_AD_l174_174132

-- Given data and conditions
def triangle_ABC (A B C D : Type) : Prop := sorry
def angle_bisector_AD (A B C D : Type) : Prop := sorry
def length_BD : ℝ := 40
def length_BC : ℝ := 45
def length_AC : ℝ := 36

-- Prove that AD = 320 units
theorem find_length_AD (A B C D : Type)
  (h1 : triangle_ABC A B C D)
  (h2 : angle_bisector_AD A B C D)
  (h3 : length_BD = 40)
  (h4 : length_BC = 45)
  (h5 : length_AC = 36) :
  ∃ x : ℝ, x = 320 :=
sorry

end find_length_AD_l174_174132


namespace green_or_yellow_probability_l174_174874

-- Given the number of marbles of each color
def green_marbles : ℕ := 4
def yellow_marbles : ℕ := 3
def white_marbles : ℕ := 6

-- The total number of marbles
def total_marbles : ℕ := green_marbles + yellow_marbles + white_marbles

-- The number of favorable outcomes (green or yellow marbles)
def favorable_marbles : ℕ := green_marbles + yellow_marbles

-- The probability of drawing a green or yellow marble as a fraction
def probability_of_green_or_yellow : Rat := favorable_marbles / total_marbles

theorem green_or_yellow_probability :
  probability_of_green_or_yellow = 7 / 13 :=
by
  sorry

end green_or_yellow_probability_l174_174874


namespace inequality_inequality_l174_174729

theorem inequality_inequality
  (x y z : ℝ) :
  (x^2 / (x^2 + 2 * y * z) + y^2 / (y^2 + 2 * z * x) + z^2 / (z^2 + 2 * x * y) ≥ 1) :=
by sorry

end inequality_inequality_l174_174729


namespace proposition_1_proposition_2_proposition_3_proposition_4_l174_174520

axiom p1 : Prop
axiom p2 : Prop
axiom p3 : Prop
axiom p4 : Prop

axiom p1_true : p1 = true
axiom p2_false : p2 = false
axiom p3_false : p3 = false
axiom p4_true : p4 = true

theorem proposition_1 : (p1 ∧ p4) = true := by sorry
theorem proposition_2 : (p1 ∧ p2) = false := by sorry
theorem proposition_3 : (¬p2 ∨ p3) = true := by sorry
theorem proposition_4 : (¬p3 ∨ ¬p4) = true := by sorry

end proposition_1_proposition_2_proposition_3_proposition_4_l174_174520


namespace least_m_plus_n_l174_174584

theorem least_m_plus_n (m n : ℕ) (h1 : Nat.gcd (m + n) 231 = 1) 
                                  (h2 : m^m ∣ n^n) 
                                  (h3 : ¬ m ∣ n)
                                  : m + n = 75 :=
sorry

end least_m_plus_n_l174_174584


namespace tan_2theta_l174_174361

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x + Real.cos x

theorem tan_2theta (θ : ℝ) (h : ∀ x, f x ≤ f θ) : Real.tan (2 * θ) = -4 / 3 := 
by 
  sorry

end tan_2theta_l174_174361


namespace quadratic_roots_range_l174_174394

theorem quadratic_roots_range (k : ℝ) : 
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (k * x₁^2 - 4 * x₁ + 1 = 0) ∧ (k * x₂^2 - 4 * x₂ + 1 = 0)) 
  ↔ (k < 4 ∧ k ≠ 0) := 
by
  sorry

end quadratic_roots_range_l174_174394


namespace exists_parallel_line_l174_174422

variable (P : ℝ × ℝ)
variable (g : ℝ × ℝ)
variable (in_first_quadrant : 0 < P.1 ∧ 0 < P.2)
variable (parallel_to_second_projection_plane : ∃ c : ℝ, g = (c, 0))

theorem exists_parallel_line (P : ℝ × ℝ) (g : ℝ × ℝ) (in_first_quadrant : 0 < P.1 ∧ 0 < P.2)
  (parallel_to_second_projection_plane : ∃ c : ℝ, g = (c, 0)) :
  ∃ a : ℝ × ℝ, (∃ d : ℝ, g = (d, 0)) ∧ (a = P) :=
sorry

end exists_parallel_line_l174_174422


namespace andrew_total_homeless_shelter_donation_l174_174940

-- Given constants and conditions
def bake_sale_total : ℕ := 400
def ingredients_cost : ℕ := 100
def piggy_bank_donation : ℕ := 10

-- Intermediate calculated values
def remaining_total : ℕ := bake_sale_total - ingredients_cost
def shelter_donation_from_bake_sale : ℕ := remaining_total / 2

-- Final goal statement
theorem andrew_total_homeless_shelter_donation :
  shelter_donation_from_bake_sale + piggy_bank_donation = 160 :=
by
  -- Proof to be provided.
  sorry

end andrew_total_homeless_shelter_donation_l174_174940


namespace fertilizer_needed_l174_174330

def p_flats := 4
def p_per_flat := 8
def p_ounces := 8

def r_flats := 3
def r_per_flat := 6
def r_ounces := 3

def s_flats := 5
def s_per_flat := 10
def s_ounces := 6

def o_flats := 2
def o_per_flat := 4
def o_ounces := 4

def vf_quantity := 2
def vf_ounces := 2

def total_fertilizer : ℕ := 
  p_flats * p_per_flat * p_ounces +
  r_flats * r_per_flat * r_ounces +
  s_flats * s_per_flat * s_ounces +
  o_flats * o_per_flat * o_ounces +
  vf_quantity * vf_ounces

theorem fertilizer_needed : total_fertilizer = 646 := by
  -- proof goes here
  sorry

end fertilizer_needed_l174_174330


namespace houses_built_during_boom_l174_174899

theorem houses_built_during_boom :
  let original_houses := 20817
  let current_houses := 118558
  let houses_built := current_houses - original_houses
  houses_built = 97741 := by
  sorry

end houses_built_during_boom_l174_174899


namespace a_plus_b_values_l174_174942

theorem a_plus_b_values (a b : ℝ) (h1 : abs a = 5) (h2 : abs b = 3) (h3 : abs (a - b) = b - a) : a + b = -2 ∨ a + b = -8 :=
sorry

end a_plus_b_values_l174_174942


namespace minimum_value_expression_l174_174813

theorem minimum_value_expression 
  (x1 x2 x3 x4 x5 : ℝ) 
  (h_pos : 0 < x1 ∧ 0 < x2 ∧ 0 < x3 ∧ 0 < x4 ∧ 0 < x5) 
  (h_cond : x1^3 + x2^3 + x3^3 + x4^3 + x5^3 = 1) : 
  ∃ y, y = (3 * Real.sqrt 3) / 2 ∧ 
  (y = (x1 / (1 - x1^2) + x2 / (1 - x2^2) + x3 / (1 - x3^2) + x4 / (1 - x4^2) + x5 / (1 - x5^2))) :=
sorry

end minimum_value_expression_l174_174813


namespace seating_arrangement_l174_174913

theorem seating_arrangement (M : ℕ) (h1 : 8 * M = 12 * M) : M = 3 :=
by
  sorry

end seating_arrangement_l174_174913


namespace g_value_l174_174239

noncomputable def g (x : ℝ) : ℝ := sorry

theorem g_value (h : ∀ x ≠ 0, g x - 3 * g (1 / x) = 3^x) :
  g 3 = -(27 + 3 * (3:ℝ)^(1/3)) / 8 :=
sorry

end g_value_l174_174239


namespace wade_customers_l174_174956

theorem wade_customers (F : ℕ) (h1 : 2 * F + 6 * F + 72 = 296) : F = 28 := 
by 
  sorry

end wade_customers_l174_174956


namespace union_of_M_and_N_l174_174278

def M : Set ℝ := {x | x^2 - 6 * x + 5 = 0}
def N : Set ℝ := {x | x^2 - 5 * x = 0}

theorem union_of_M_and_N : M ∪ N = {0, 1, 5} := by
  sorry

end union_of_M_and_N_l174_174278


namespace problem_solution_l174_174726

noncomputable def corrected_angles 
  (x1_star x2_star x3_star : ℝ) 
  (σ : ℝ) 
  (h_sum : x1_star + x2_star + x3_star - 180.0 = 0)  
  (h_var : σ^2 = (0.1)^2) : ℝ × ℝ × ℝ :=
  let Δ := 2.0 / 3.0 * 0.667
  let Δx1 := Δ * (σ^2 / 2)
  let Δx2 := Δ * (σ^2 / 2)
  let Δx3 := Δ * (σ^2 / 2)
  let corrected_x1 := x1_star - Δx1
  let corrected_x2 := x2_star - Δx2
  let corrected_x3 := x3_star - Δx3
  (corrected_x1, corrected_x2, corrected_x3)

theorem problem_solution :
  corrected_angles 31 62 89 (0.1) sorry sorry = (30.0 + 40 / 60, 61.0 + 40 / 60, 88 + 20 / 60) := 
  sorry

end problem_solution_l174_174726


namespace vertex_of_quadratic_l174_174155

theorem vertex_of_quadratic (x : ℝ) : 
  (y : ℝ) = -2 * (x + 1) ^ 2 + 3 →
  (∃ vertex_x vertex_y : ℝ, vertex_x = -1 ∧ vertex_y = 3 ∧ y = -2 * (vertex_x + 1) ^ 2 + vertex_y) :=
by
  intro h
  exists -1, 3
  simp [h]
  sorry

end vertex_of_quadratic_l174_174155


namespace union_correct_l174_174018

variable (x : ℝ)
def A := {x | -2 < x ∧ x < 1}
def B := {x | 0 < x ∧ x < 3}
def unionSet := {x | -2 < x ∧ x < 3}

theorem union_correct : ( {x | -2 < x ∧ x < 1} ∪ {x | 0 < x ∧ x < 3} ) = {x | -2 < x ∧ x < 3} := by
  sorry

end union_correct_l174_174018


namespace division_number_l174_174287

-- Definitions from conditions
def D : Nat := 3
def Q : Nat := 4
def R : Nat := 3

-- Theorem statement
theorem division_number : ∃ N : Nat, N = D * Q + R ∧ N = 15 :=
by
  sorry

end division_number_l174_174287


namespace largest_divisor_of_n_l174_174187

theorem largest_divisor_of_n (n : ℕ) (h1 : 0 < n) (h2 : 127 ∣ n^3) : 127 ∣ n :=
sorry

end largest_divisor_of_n_l174_174187


namespace min_f_when_a_neg3_range_of_a_l174_174807

open Real

noncomputable def f (x a : ℝ) : ℝ := abs (x - 1) + abs (x - a)

-- First statement: Minimum value of f(x) when a = -3
theorem min_f_when_a_neg3 : (∀ x : ℝ, f x (-3) ≥ 4) ∧ (∃ x : ℝ,  f x (-3) = 4) := by
  sorry

-- Second statement: Range of a given the condition
theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x a ≤ 2 * a + 2 * abs (x - 1)) ↔ a ≥ 1/3 := by
  sorry

end min_f_when_a_neg3_range_of_a_l174_174807


namespace pascals_triangle_ratio_456_l174_174325

theorem pascals_triangle_ratio_456 (n : ℕ) :
  (∃ r : ℕ,
    (n.choose r * 5 = (n.choose (r + 1)) * 4) ∧
    ((n.choose (r + 1)) * 6 = (n.choose (r + 2)) * 5)) →
  n = 98 :=
sorry

end pascals_triangle_ratio_456_l174_174325


namespace polynomial_root_recip_squares_l174_174885

theorem polynomial_root_recip_squares (a b c : ℝ) 
  (h1 : a + b + c = 6) (h2 : a * b + b * c + c * a = 11) (h3 : a * b * c = 6):
  1 / a^2 + 1 / b^2 + 1 / c^2 = 49 / 36 :=
sorry

end polynomial_root_recip_squares_l174_174885


namespace airport_exchange_rate_frac_l174_174586

variable (euros_received : ℕ) (euros : ℕ) (official_exchange_rate : ℕ) (dollars_received : ℕ)

theorem airport_exchange_rate_frac (h1 : euros = 70) (h2 : official_exchange_rate = 5) (h3 : dollars_received = 10) :
  (euros_received * dollars_received) = (euros * official_exchange_rate) →
  euros_received = 5 / 7 :=
  sorry

end airport_exchange_rate_frac_l174_174586


namespace mean_of_two_fractions_l174_174699

theorem mean_of_two_fractions :
  ( (2 : ℚ) / 3 + (4 : ℚ) / 9 ) / 2 = 5 / 9 :=
by
  sorry

end mean_of_two_fractions_l174_174699


namespace max_value_log_div_x_l174_174847

noncomputable def func (x : ℝ) := (Real.log x) / x

theorem max_value_log_div_x : ∃ x > 0, func x = 1 / Real.exp 1 ∧ 
(∀ t > 0, t ≠ x → func t ≤ func x) :=
sorry

end max_value_log_div_x_l174_174847


namespace value_of_f_g_5_l174_174705

def g (x : ℕ) : ℕ := 4 * x - 5
def f (x : ℕ) : ℕ := 6 * x + 11

theorem value_of_f_g_5 : f (g 5) = 101 := by
  sorry

end value_of_f_g_5_l174_174705


namespace total_hours_worked_l174_174691

-- Define the number of hours worked on Saturday
def hours_saturday : ℕ := 6

-- Define the number of hours worked on Sunday
def hours_sunday : ℕ := 4

-- Define the total number of hours worked on both days
def total_hours : ℕ := hours_saturday + hours_sunday

-- The theorem to prove the total number of hours worked on Saturday and Sunday
theorem total_hours_worked : total_hours = 10 := by
  sorry

end total_hours_worked_l174_174691


namespace visitors_on_monday_l174_174611

theorem visitors_on_monday (M : ℕ) (h : M + 2 * M + 100 = 250) : M = 50 :=
by
  sorry

end visitors_on_monday_l174_174611


namespace total_legs_l174_174353

-- Define the conditions
def chickens : Nat := 7
def sheep : Nat := 5
def legs_chicken : Nat := 2
def legs_sheep : Nat := 4

-- State the problem as a theorem
theorem total_legs :
  chickens * legs_chicken + sheep * legs_sheep = 34 :=
by
  sorry -- Proof not provided

end total_legs_l174_174353


namespace production_cost_per_performance_l174_174594

def overhead_cost := 81000
def income_per_performance := 16000
def performances_needed := 9

theorem production_cost_per_performance :
  ∃ P, 9 * income_per_performance = overhead_cost + 9 * P ∧ P = 7000 :=
by
  sorry

end production_cost_per_performance_l174_174594


namespace weight_of_hollow_golden_sphere_l174_174162

theorem weight_of_hollow_golden_sphere : 
  let diameter := 12
  let thickness := 0.3
  let pi := (3 : Real)
  let outer_radius := diameter / 2
  let inner_radius := (outer_radius - thickness)
  let outer_volume := (4 / 3) * pi * outer_radius^3
  let inner_volume := (4 / 3) * pi * inner_radius^3
  let gold_volume := outer_volume - inner_volume
  let weight_per_cubic_inch := 1
  let weight := gold_volume * weight_per_cubic_inch
  weight = 123.23 :=
by
  sorry

end weight_of_hollow_golden_sphere_l174_174162


namespace integer_solutions_count_l174_174304

theorem integer_solutions_count : 
  ∃ (s : Finset ℤ), 
    (∀ x ∈ s, 2 * x + 1 > -3 ∧ -x + 3 ≥ 0) ∧ 
    s.card = 5 := 
by 
  sorry

end integer_solutions_count_l174_174304


namespace pickup_carries_10_bags_per_trip_l174_174146

def total_weight : ℕ := 10000
def weight_one_bag : ℕ := 50
def number_of_trips : ℕ := 20
def total_bags : ℕ := total_weight / weight_one_bag
def bags_per_trip : ℕ := total_bags / number_of_trips

theorem pickup_carries_10_bags_per_trip : bags_per_trip = 10 := by
  sorry

end pickup_carries_10_bags_per_trip_l174_174146


namespace curved_surface_area_cone_l174_174883

variable (a α β : ℝ) (l := a * Real.sin α) (r := a * Real.cos β)

theorem curved_surface_area_cone :
  π * r * l = π * a^2 * Real.sin α * Real.cos β := by
  sorry

end curved_surface_area_cone_l174_174883


namespace find_multiple_l174_174237

theorem find_multiple (m : ℤ) (h : 38 + m * 43 = 124) : m = 2 := by
    sorry

end find_multiple_l174_174237


namespace prove_perpendicular_planes_l174_174236

-- Defining the non-coincident lines m and n
variables {m n : Set Point} {α β : Set Point}

-- Lines and plane relationship definitions
def parallel (x y : Set Point) : Prop := sorry
def perpendicular (x y : Set Point) : Prop := sorry
def subset (x y : Set Point) : Prop := sorry

-- Given conditions
axiom h1 : parallel m n
axiom h2 : subset m α
axiom h3 : perpendicular n β

-- Prove that α is perpendicular to β
theorem prove_perpendicular_planes :
  perpendicular α β :=
  sorry

end prove_perpendicular_planes_l174_174236


namespace gcd_of_45_75_90_l174_174983

def gcd_three_numbers (a b c : ℕ) : ℕ :=
  Nat.gcd (Nat.gcd a b) c

theorem gcd_of_45_75_90 : gcd_three_numbers 45 75 90 = 15 := by
  sorry

end gcd_of_45_75_90_l174_174983


namespace leon_required_score_l174_174381

noncomputable def leon_scores : List ℕ := [72, 68, 75, 81, 79]

theorem leon_required_score (n : ℕ) :
  (List.sum leon_scores + n) / (List.length leon_scores + 1) ≥ 80 ↔ n ≥ 105 :=
by sorry

end leon_required_score_l174_174381


namespace reduced_price_proof_l174_174519

noncomputable def reduced_price (P: ℝ) := 0.88 * P

theorem reduced_price_proof :
  ∃ R P : ℝ, R = reduced_price P ∧ 1200 / R = 1200 / P + 6 ∧ R = 24 :=
by
  sorry

end reduced_price_proof_l174_174519


namespace equal_sharing_of_chicken_wings_l174_174873

theorem equal_sharing_of_chicken_wings 
  (initial_wings : ℕ) (additional_wings : ℕ) (number_of_friends : ℕ)
  (total_wings : ℕ) (wings_per_person : ℕ)
  (h_initial : initial_wings = 8)
  (h_additional : additional_wings = 10)
  (h_number : number_of_friends = 3)
  (h_total : total_wings = initial_wings + additional_wings)
  (h_division : wings_per_person = total_wings / number_of_friends) :
  wings_per_person = 6 := 
  by
  sorry

end equal_sharing_of_chicken_wings_l174_174873


namespace bag_contains_n_black_balls_l174_174522

theorem bag_contains_n_black_balls (n : ℕ) : (5 / (n + 5) = 1 / 3) → n = 10 := by
  sorry

end bag_contains_n_black_balls_l174_174522


namespace factorize_expression_l174_174784

theorem factorize_expression (x y : ℝ) :
  9 * x^2 - y^2 - 4 * y - 4 = (3 * x + y + 2) * (3 * x - y - 2) :=
by
  sorry

end factorize_expression_l174_174784


namespace percentage_water_mixture_l174_174897

theorem percentage_water_mixture 
  (volume_A : ℝ) (volume_B : ℝ) (volume_C : ℝ)
  (ratio_A : ℝ := 5) (ratio_B : ℝ := 3) (ratio_C : ℝ := 2)
  (percentage_water_A : ℝ := 0.20) (percentage_water_B : ℝ := 0.35) (percentage_water_C : ℝ := 0.50) :
  (volume_A = ratio_A) → (volume_B = ratio_B) → (volume_C = ratio_C) → 
  ((percentage_water_A * volume_A + percentage_water_B * volume_B + percentage_water_C * volume_C) /
   (ratio_A + ratio_B + ratio_C)) * 100 = 30.5 := 
by 
  intros hA hB hC
  -- Proof steps would go here
  sorry

end percentage_water_mixture_l174_174897


namespace victors_friend_decks_l174_174003

theorem victors_friend_decks:
  ∀ (deck_cost : ℕ) (victor_decks : ℕ) (total_spent : ℕ)
  (friend_decks : ℕ),
  deck_cost = 8 →
  victor_decks = 6 →
  total_spent = 64 →
  (victor_decks * deck_cost + friend_decks * deck_cost = total_spent) →
  friend_decks = 2 :=
by
  intros deck_cost victor_decks total_spent friend_decks hc hv ht heq
  sorry

end victors_friend_decks_l174_174003


namespace melissa_work_hours_l174_174451

theorem melissa_work_hours (total_fabric : ℕ) (fabric_per_dress : ℕ) (hours_per_dress : ℕ) (total_num_dresses : ℕ) (total_hours : ℕ) 
  (h1 : total_fabric = 56) (h2 : fabric_per_dress = 4) (h3 : hours_per_dress = 3) : 
  total_hours = (total_fabric / fabric_per_dress) * hours_per_dress := by
  sorry

end melissa_work_hours_l174_174451


namespace diego_payment_l174_174776

theorem diego_payment (d : ℤ) (celina : ℤ) (total : ℤ) (h₁ : celina = 1000 + 4 * d) (h₂ : total = celina + d) (h₃ : total = 50000) : d = 9800 :=
sorry

end diego_payment_l174_174776


namespace sugar_for_recipe_l174_174056

theorem sugar_for_recipe (sugar_frosting sugar_cake : ℝ) (h1 : sugar_frosting = 0.6) (h2 : sugar_cake = 0.2) :
  sugar_frosting + sugar_cake = 0.8 :=
by
  sorry

end sugar_for_recipe_l174_174056


namespace sufficient_and_necessary_condition_l174_174487

noncomputable def f (x : ℝ) : ℝ := x + Real.sin x

theorem sufficient_and_necessary_condition (a b : ℝ) : (a + b > 0) ↔ (f a + f b > 0) :=
by sorry

end sufficient_and_necessary_condition_l174_174487


namespace percent_of_percent_l174_174609

theorem percent_of_percent (y : ℝ) : 0.3 * (0.6 * y) = 0.18 * y :=
sorry

end percent_of_percent_l174_174609


namespace veronica_initial_marbles_l174_174622

variable {D M P V : ℕ}

theorem veronica_initial_marbles (hD : D = 14) (hM : M = 20) (hP : P = 19)
  (h_total : D + M + P + V = 60) : V = 7 :=
by
  sorry

end veronica_initial_marbles_l174_174622


namespace max_additional_hours_l174_174230

/-- Define the additional hours of studying given the investments in dorms, food, and parties -/
def additional_hours (a b c : ℝ) : ℝ :=
  5 * a + 3 * b + (11 * c - c^2)

/-- Define the total investment constraint -/
def investment_constraint (a b c : ℝ) : Prop :=
  a + b + c = 5

/-- Prove the maximal additional hours of studying -/
theorem max_additional_hours : ∃ (a b c : ℝ), investment_constraint a b c ∧ additional_hours a b c = 34 :=
by
  sorry

end max_additional_hours_l174_174230


namespace nonincreasing_7_digit_integers_l174_174216

theorem nonincreasing_7_digit_integers : 
  ∃ n : ℕ, n = 11439 ∧ (∀ x : ℕ, (10^6 ≤ x ∧ x < 10^7) → 
    (∀ i j : ℕ, 1 ≤ i ∧ i < j ∧ j ≤ 7 → (x / 10^(7 - i) % 10) ≥ (x / 10^(7 - j) % 10))) :=
by
  sorry

end nonincreasing_7_digit_integers_l174_174216


namespace find_extrema_l174_174192

noncomputable def f (x : ℝ) : ℝ := x^3 + (-3/2) * x^2 + (-3) * x + 1
noncomputable def f' (x : ℝ) : ℝ := 3 * x^2 + 2 * (-3/2) * x + (-3)
noncomputable def g (x : ℝ) : ℝ := f' x * Real.exp x

theorem find_extrema :
  (a = -3/2 ∧ b = -3 ∧ f' (1) = (3 * (1:ℝ)^2 - 3/2 * (1:ℝ) - 3) ) ∧
  (g 1 = -3 * Real.exp 1 ∧ g (-2) = 15 * Real.exp (-2)) := 
by
  -- Sorry for skipping the proof
  sorry

end find_extrema_l174_174192


namespace suitableTempForPreservingBoth_l174_174376

-- Definitions for the temperature ranges of types A and B vegetables
def suitableTempRangeA := {t : ℝ | 3 ≤ t ∧ t ≤ 8}
def suitableTempRangeB := {t : ℝ | 5 ≤ t ∧ t ≤ 10}

-- The intersection of the suitable temperature ranges
def suitableTempRangeForBoth := {t : ℝ | 5 ≤ t ∧ t ≤ 8}

-- The theorem statement we need to prove
theorem suitableTempForPreservingBoth :
  suitableTempRangeForBoth = suitableTempRangeA ∩ suitableTempRangeB :=
sorry

end suitableTempForPreservingBoth_l174_174376


namespace proof_problem_l174_174464

noncomputable def p : ℝ := -5 / 3
noncomputable def q : ℝ := -1

def A (p : ℝ) : Set ℝ := {x | 2 * x^2 + 3 * p * x + 2 = 0}
def B (q : ℝ) : Set ℝ := {x | 2 * x^2 + x + q = 0}

theorem proof_problem (h : (A p ∩ B q) = {1 / 2}) :
    p = -5 / 3 ∧ q = -1 ∧ (A p ∪ B q) = {-1, 1 / 2, 2} := by
  sorry

end proof_problem_l174_174464


namespace regular_polygon_sides_l174_174111

-- Define the main theorem statement
theorem regular_polygon_sides (n : ℕ) : 
  (n > 2) ∧ 
  ((n - 2) * 180 / n - 360 / n = 90) → 
  n = 8 := by
  sorry

end regular_polygon_sides_l174_174111


namespace cosine_of_3pi_over_2_l174_174901

theorem cosine_of_3pi_over_2 : Real.cos (3 * Real.pi / 2) = 0 := by
  sorry

end cosine_of_3pi_over_2_l174_174901


namespace Joe_first_lift_weight_l174_174822

variable (F S : ℝ)

theorem Joe_first_lift_weight (h1 : F + S = 600) (h2 : 2 * F = S + 300) : F = 300 := 
sorry

end Joe_first_lift_weight_l174_174822


namespace land_area_decreases_l174_174701

theorem land_area_decreases (a : ℕ) (h : a > 4) : (a * a) > ((a + 4) * (a - 4)) :=
by
  sorry

end land_area_decreases_l174_174701


namespace books_sold_on_monday_l174_174673

def InitialStock : ℕ := 800
def BooksNotSold : ℕ := 600
def BooksSoldTuesday : ℕ := 10
def BooksSoldWednesday : ℕ := 20
def BooksSoldThursday : ℕ := 44
def BooksSoldFriday : ℕ := 66

def TotalBooksSold : ℕ := InitialStock - BooksNotSold
def BooksSoldAfterMonday : ℕ := BooksSoldTuesday + BooksSoldWednesday + BooksSoldThursday + BooksSoldFriday

theorem books_sold_on_monday : 
  TotalBooksSold - BooksSoldAfterMonday = 60 := by
  sorry

end books_sold_on_monday_l174_174673


namespace power_multiplication_l174_174425

variable (a : ℝ)

theorem power_multiplication : (-a)^3 * a^2 = -a^5 := 
sorry

end power_multiplication_l174_174425


namespace smallest_x_satisfying_equation_l174_174602

theorem smallest_x_satisfying_equation :
  ∀ x : ℝ, (2 * x ^ 2 + 24 * x - 60 = x * (x + 13)) → x = -15 ∨ x = 4 ∧ ∃ y : ℝ, y = -15 ∨ y = 4 ∧ y ≤ x :=
by
  sorry

end smallest_x_satisfying_equation_l174_174602


namespace meghan_total_money_l174_174075

theorem meghan_total_money :
  let num_100_bills := 2
  let num_50_bills := 5
  let num_10_bills := 10
  let value_100_bills := num_100_bills * 100
  let value_50_bills := num_50_bills * 50
  let value_10_bills := num_10_bills * 10
  let total_money := value_100_bills + value_50_bills + value_10_bills
  total_money = 550 := by sorry

end meghan_total_money_l174_174075


namespace sum_of_roots_eq_14_l174_174032

-- Define the condition as a hypothesis
def equation (x : ℝ) := (x - 7) ^ 2 = 16

-- Define the statement that needs to be proved
theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, equation x → x = 11 ∨ x = 3) → 11 + 3 = 14 :=
by 
sorry

end sum_of_roots_eq_14_l174_174032


namespace drum_y_capacity_filled_l174_174259

-- Definitions of the initial conditions
def capacity_of_drum_X (C : ℝ) (half_full_x : ℝ) := half_full_x = 1 / 2 * C
def capacity_of_drum_Y (C : ℝ) (two_c_y : ℝ) := two_c_y = 2 * C
def oil_in_drum_X (C : ℝ) (half_full_x : ℝ) := half_full_x = 1 / 2 * C
def oil_in_drum_Y (C : ℝ) (four_fifth_c_y : ℝ) := four_fifth_c_y = 4 / 5 * C

-- Theorem to prove the capacity filled in drum Y after pouring all oil from X
theorem drum_y_capacity_filled {C : ℝ} (hx : 1/2 * C = 1 / 2 * C) (hy : 2 * C = 2 * C) (ox : 1/2 * C = 1 / 2 * C) (oy : 4/5 * 2 * C = 4 / 5 * C) :
  ( (1/2 * C + 4/5 * C) / (2 * C) ) = 13 / 20 :=
by
  sorry

end drum_y_capacity_filled_l174_174259


namespace proper_subset_count_of_set_l174_174461

theorem proper_subset_count_of_set (s : Finset ℕ) (h : s = {1, 2, 3}) : s.powerset.card - 1 = 7 := by
  sorry

end proper_subset_count_of_set_l174_174461


namespace inradius_circumradius_inequality_l174_174321

variable {R r a b c : ℝ}

def inradius (ABC : Triangle) := r
def circumradius (ABC : Triangle) := R
def side_a (ABC : Triangle) := a
def side_b (ABC : Triangle) := b
def side_c (ABC : Triangle) := c

theorem inradius_circumradius_inequality (ABC : Triangle) :
  R / (2 * r) ≥ (64 * a^2 * b^2 * c^2 / ((4 * a^2 - (b - c)^2) * (4 * b^2 - (c - a)^2) * (4 * c^2 - (a - b)^2)))^2 :=
sorry

end inradius_circumradius_inequality_l174_174321


namespace number_of_boys_l174_174583

theorem number_of_boys (x : ℕ) (y : ℕ) (h1 : x + y = 8) (h2 : y > x) : x = 1 ∨ x = 2 ∨ x = 3 :=
by
  sorry

end number_of_boys_l174_174583


namespace conic_sections_are_parabolas_l174_174745

theorem conic_sections_are_parabolas (x y : ℝ) :
  y^6 - 9*x^6 = 3*y^3 - 1 → ∃ k : ℝ, (y^3 - 1 = k * 3 * x^3 ∨ y^3 = -k * 3 * x^3 + 1) := by
  sorry

end conic_sections_are_parabolas_l174_174745


namespace find_x_l174_174101

-- Declaration for the custom operation
def star (a b : ℝ) : ℝ := a * b + 3 * b - 2 * a

-- Theorem statement
theorem find_x (x : ℝ) (h : star 3 x = 23) : x = 29 / 6 :=
by {
    sorry -- The proof steps are to be filled here.
}

end find_x_l174_174101


namespace probability_one_die_shows_4_given_sum_7_l174_174360

def outcomes_with_sum_7 : List (ℕ × ℕ) := [(1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1)]

def outcome_has_4 (outcome : ℕ × ℕ) : Bool :=
  outcome.fst = 4 ∨ outcome.snd = 4

def favorable_outcomes : List (ℕ × ℕ) :=
  outcomes_with_sum_7.filter outcome_has_4

theorem probability_one_die_shows_4_given_sum_7 :
  (favorable_outcomes.length : ℚ) / (outcomes_with_sum_7.length : ℚ) = 1 / 3 := sorry

end probability_one_die_shows_4_given_sum_7_l174_174360


namespace loan_amounts_l174_174167

theorem loan_amounts (x y : ℝ) (h1 : x + y = 50) (h2 : 0.1 * x + 0.08 * y = 4.4) : x = 20 ∧ y = 30 := by
  sorry

end loan_amounts_l174_174167


namespace monthly_rent_calculation_l174_174171

noncomputable def monthly_rent (purchase_cost : ℕ) (maintenance_pct : ℝ) (annual_taxes : ℕ) (target_roi : ℝ) : ℝ :=
  let annual_return := target_roi * (purchase_cost : ℝ)
  let total_annual_requirement := annual_return + (annual_taxes : ℝ)
  let monthly_requirement := total_annual_requirement / 12
  let actual_rent := monthly_requirement / (1 - maintenance_pct)
  actual_rent

theorem monthly_rent_calculation :
  monthly_rent 12000 0.15 400 0.06 = 109.80 :=
by
  sorry

end monthly_rent_calculation_l174_174171


namespace white_sox_wins_l174_174578

theorem white_sox_wins 
  (total_games : ℕ) 
  (games_won : ℕ) 
  (games_lost : ℕ)
  (win_loss_difference : ℤ) 
  (total_games_condition : total_games = 162) 
  (lost_games_condition : games_lost = 63) 
  (win_loss_diff_condition : (games_won : ℤ) - games_lost = win_loss_difference) 
  (win_loss_difference_value : win_loss_difference = 36) 
  : games_won = 99 :=
by
  sorry

end white_sox_wins_l174_174578


namespace find_x_intercept_l174_174954

variables (a x y : ℝ)
def l1 (a x y : ℝ) : Prop := (a + 2) * x + 3 * y = 5
def l2 (a x y : ℝ) : Prop := (a - 1) * x + 2 * y = 6
def are_parallel (a : ℝ) : Prop := (- (a + 2) / 3) = (- (a - 1) / 2)
def x_intercept_of_l1 (a x : ℝ) : Prop := l1 a x 0

theorem find_x_intercept (h : are_parallel a) : x_intercept_of_l1 7 (5 / 9) := 
sorry

end find_x_intercept_l174_174954


namespace burger_cost_proof_l174_174294

variable {burger_cost fries_cost salad_cost total_cost : ℕ}
variable {quantity_of_fries : ℕ}

theorem burger_cost_proof (h_fries_cost : fries_cost = 2)
    (h_salad_cost : salad_cost = 3 * fries_cost)
    (h_quantity_of_fries : quantity_of_fries = 2)
    (h_total_cost : total_cost = 15)
    (h_equation : burger_cost + (quantity_of_fries * fries_cost) + salad_cost = total_cost) :
    burger_cost = 5 :=
by 
  sorry

end burger_cost_proof_l174_174294


namespace maximum_ratio_is_2_plus_2_sqrt2_l174_174152

noncomputable def C1_polar_eq (θ : ℝ) : Prop :=
  ∀ ρ : ℝ, ρ * (Real.cos θ + Real.sin θ) = 1

noncomputable def C2_polar_eq (θ : ℝ) : Prop :=
  ∀ ρ : ℝ, ρ = 4 * Real.cos θ

theorem maximum_ratio_is_2_plus_2_sqrt2 (α : ℝ) (hα : 0 ≤ α ∧ α ≤ Real.pi / 2) :
  ∃ ρA ρB : ℝ, (ρA = 1 / (Real.cos α + Real.sin α)) ∧ (ρB = 4 * Real.cos α) ∧ 
  (4 * Real.cos α * (Real.cos α + Real.sin α) = 2 + 2 * Real.sqrt 2) :=
sorry

end maximum_ratio_is_2_plus_2_sqrt2_l174_174152


namespace find_a_l174_174598

theorem find_a (a : ℝ) :
  (∃ b : ℝ, 4 * b + 3 = 7 ∧ 5 * (-b) - 1 = 2 * (-b) + a) → a = -4 :=
by
  sorry

end find_a_l174_174598


namespace triangle_height_relationship_l174_174346

theorem triangle_height_relationship
  (b : ℝ) (h1 h2 h3 : ℝ)
  (area1 area2 area3 : ℝ)
  (h_equal_angle : area1 / area2 = 16 / 25)
  (h_diff_angle : area1 / area3 = 4 / 9) :
  4 * h2 = 5 * h1 ∧ 6 * h2 = 5 * h3 := by
    sorry

end triangle_height_relationship_l174_174346


namespace inequality_proof_l174_174834

variable (x y z : ℝ)

theorem inequality_proof (h : x + y + z = x * y + y * z + z * x) :
  x / (x^2 + 1) + y / (y^2 + 1) + z / (z^2 + 1) ≥ -1/2 :=
sorry

end inequality_proof_l174_174834


namespace yellow_curved_given_curved_l174_174950

variable (P_green : ℝ) (P_yellow : ℝ) (P_straight : ℝ) (P_curved : ℝ)
variable (P_red_given_straight : ℝ) 

-- Given conditions
variables (h1 : P_green = 3 / 4) 
          (h2 : P_yellow = 1 / 4) 
          (h3 : P_straight = 1 / 2) 
          (h4 : P_curved = 1 / 2)
          (h5 : P_red_given_straight = 1 / 3)

-- To be proven
theorem yellow_curved_given_curved : (P_yellow * P_curved) / P_curved = 1 / 4 :=
by
sorry

end yellow_curved_given_curved_l174_174950


namespace daily_wage_c_l174_174062

theorem daily_wage_c (a_days b_days c_days total_earnings : ℕ)
  (ratio_a_b ratio_b_c : ℚ)
  (a_wage b_wage c_wage : ℚ) :
  a_days = 6 →
  b_days = 9 →
  c_days = 4 →
  total_earnings = 1480 →
  ratio_a_b = 3 / 4 →
  ratio_b_c = 4 / 5 →
  b_wage = ratio_a_b * a_wage → 
  c_wage = ratio_b_c * b_wage → 
  a_days * a_wage + b_days * b_wage + c_days * c_wage = total_earnings →
  c_wage = 100 / 3 :=
by
  intros
  sorry

end daily_wage_c_l174_174062


namespace remainder_when_divided_by_6_l174_174126

theorem remainder_when_divided_by_6 (n : ℕ) (h₁ : n = 482157)
  (odd_n : n % 2 ≠ 0) (div_by_3 : n % 3 = 0) : n % 6 = 3 :=
by
  -- Proof goes here
  sorry

end remainder_when_divided_by_6_l174_174126


namespace option_D_forms_triangle_l174_174741

theorem option_D_forms_triangle (a b c : ℝ) (ha : a = 6) (hb : b = 8) (hc : c = 9) : 
  a + b > c ∧ a + c > b ∧ b + c > a := by
  sorry

end option_D_forms_triangle_l174_174741


namespace evaluate_expression_l174_174401

theorem evaluate_expression : 150 * (150 - 5) - (150 * 150 - 7) = -743 :=
by
  sorry

end evaluate_expression_l174_174401


namespace sum_of_valid_b_values_l174_174086

/-- Given a quadratic equation 3x² + 7x + b = 0, where b is a positive integer,
and the requirement that the equation must have rational roots, the sum of all
possible positive integer values of b is 6. -/
theorem sum_of_valid_b_values : 
  ∃ (b_values : List ℕ), 
    (∀ b ∈ b_values, 0 < b ∧ ∃ n : ℤ, 49 - 12 * b = n^2) ∧ b_values.sum = 6 :=
by sorry

end sum_of_valid_b_values_l174_174086


namespace find_f_minus_1_l174_174721

noncomputable def f : ℝ → ℝ := sorry

axiom additivity : ∀ x y : ℝ, f (x + y) = f x + f y
axiom f_at_2 : f 2 = 4

theorem find_f_minus_1 : f (-1) = -2 := 
by 
  sorry

end find_f_minus_1_l174_174721


namespace ratio_simplified_l174_174496

variable (a b c : ℕ)
variable (n m p : ℕ) (h1 : n > 0) (h2 : m > 0) (h3 : p > 0)

theorem ratio_simplified (h_ratio : a^n = 3 * c^p ∧ b^m = 4 * c^p ∧ c^p = 7 * c^p) :
  (a^n + b^m + c^p) / c^p = 2 := sorry

end ratio_simplified_l174_174496


namespace percentage_fewer_than_50000_l174_174093

def percentage_lt_20000 : ℝ := 35
def percentage_20000_to_49999 : ℝ := 45
def percentage_lt_50000 : ℝ := 80

theorem percentage_fewer_than_50000 :
  percentage_lt_20000 + percentage_20000_to_49999 = percentage_lt_50000 := 
by
  sorry

end percentage_fewer_than_50000_l174_174093


namespace stock_percent_change_l174_174957

variable (x : ℝ)

theorem stock_percent_change (h1 : ∀ x, 0.75 * x = x * 0.75)
                             (h2 : ∀ x, 1.05 * x = 0.75 * x + 0.3 * 0.75 * x):
    ((1.05 * x - x) / x) * 100 = 5 :=
by
  sorry

end stock_percent_change_l174_174957


namespace max_handshakes_25_people_l174_174261

-- Define the number of people attending the conference.
def num_people : ℕ := 25

-- Define the combinatorial formula to calculate the maximum number of handshakes.
def max_handshakes (n : ℕ) : ℕ := n.choose 2

-- State the theorem that we need to prove.
theorem max_handshakes_25_people : max_handshakes num_people = 300 :=
by
  -- Proof will be filled in later
  sorry

end max_handshakes_25_people_l174_174261


namespace max_area_rectangle_l174_174635

-- Define the conditions using Lean
def is_rectangle (length width : ℕ) : Prop :=
  2 * (length + width) = 34

-- Define the problem as a theorem in Lean
theorem max_area_rectangle : ∃ (length width : ℕ), is_rectangle length width ∧ length * width = 72 :=
by
  sorry

end max_area_rectangle_l174_174635


namespace always_possible_to_rotate_disks_l174_174154

def labels_are_distinct (a : Fin 20 → ℕ) : Prop :=
  ∀ i j : Fin 20, i ≠ j → a i ≠ a j

def opposite_position (i : Fin 20) (r : Fin 20) : Fin 20 :=
  (i + r) % 20

def no_identical_numbers_opposite (a b : Fin 20 → ℕ) (r : Fin 20) : Prop :=
  ∀ i : Fin 20, a i ≠ b (opposite_position i r)

theorem always_possible_to_rotate_disks (a b : Fin 20 → ℕ) :
  labels_are_distinct a →
  labels_are_distinct b →
  ∃ r : Fin 20, no_identical_numbers_opposite a b r :=
sorry

end always_possible_to_rotate_disks_l174_174154


namespace hyperbola_center_l174_174882

theorem hyperbola_center (x y : ℝ) :
    9 * x^2 - 54 * x - 36 * y^2 + 360 * y - 864 = 0 → (x, y) = (3, 5) :=
by
  sorry

end hyperbola_center_l174_174882


namespace sequence_item_l174_174962

theorem sequence_item (n : ℕ) (a_n : ℕ → Rat) (h : a_n n = 2 / (n^2 + n)) : a_n n = 1 / 15 → n = 5 := by
  sorry

end sequence_item_l174_174962


namespace students_like_basketball_or_cricket_or_both_l174_174708

theorem students_like_basketball_or_cricket_or_both :
  let basketball_lovers := 9
  let cricket_lovers := 8
  let both_lovers := 6
  basketball_lovers + cricket_lovers - both_lovers = 11 :=
by
  sorry

end students_like_basketball_or_cricket_or_both_l174_174708


namespace min_k_value_l174_174034

variable (p q r s k : ℕ)

/-- Prove the smallest value of k for which p, q, r, and s are positive integers and 
    satisfy the given equations is 77
-/
theorem min_k_value (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s)
  (eq1 : p + 2 * q + 3 * r + 4 * s = k)
  (eq2 : 4 * p = 3 * q)
  (eq3 : 4 * p = 2 * r)
  (eq4 : 4 * p = s) : k = 77 :=
sorry

end min_k_value_l174_174034


namespace price_of_case_bulk_is_12_l174_174493

noncomputable def price_per_can_grocery_store : ℚ := 6 / 12
noncomputable def price_per_can_bulk : ℚ := price_per_can_grocery_store - 0.25
def cans_per_case_bulk : ℕ := 48
noncomputable def price_per_case_bulk : ℚ := price_per_can_bulk * cans_per_case_bulk

theorem price_of_case_bulk_is_12 : price_per_case_bulk = 12 :=
by
  sorry

end price_of_case_bulk_is_12_l174_174493


namespace intersection_of_A_and_B_l174_174739

def A : Set ℝ := { x | -1 ≤ x ∧ x < 3 }
def B : Set ℝ := { y | 2 ≤ y ∧ y ≤ 5 }

theorem intersection_of_A_and_B :
  A ∩ B = { x | 2 ≤ x ∧ x < 3 } :=
sorry

end intersection_of_A_and_B_l174_174739


namespace problem_A_problem_B_problem_C_problem_D_problem_E_l174_174525

-- Definitions and assumptions based on the problem statement
def eqI (x y z : ℕ) := x + y + z = 45
def eqII (x y z w : ℕ) := x + y + z + w = 50
def consecutive_odd_integers (x y z : ℕ) := y = x + 2 ∧ z = x + 4
def multiples_of_five (x y z w : ℕ) := (∃ a b c d : ℕ, x = 5 * a ∧ y = 5 * b ∧ z = 5 * c ∧ w = 5 * d)
def consecutive_integers (x y z w : ℕ) := y = x + 1 ∧ z = x + 2 ∧ w = x + 3
def prime_integers (x y z : ℕ) := Prime x ∧ Prime y ∧ Prime z

-- Lean theorem statements
theorem problem_A : ∃ x y z : ℕ, eqI x y z ∧ consecutive_odd_integers x y z := 
sorry

theorem problem_B : ¬ (∃ x y z : ℕ, eqI x y z ∧ prime_integers x y z) := 
sorry

theorem problem_C : ¬ (∃ x y z w : ℕ, eqII x y z w ∧ consecutive_odd_integers x y z) :=
sorry

theorem problem_D : ∃ x y z w : ℕ, eqII x y z w ∧ multiples_of_five x y z w := 
sorry

theorem problem_E : ∃ x y z w : ℕ, eqII x y z w ∧ consecutive_integers x y z w := 
sorry

end problem_A_problem_B_problem_C_problem_D_problem_E_l174_174525


namespace parabola_focus_directrix_l174_174794

-- Definitions and conditions
def parabola (y a x : ℝ) : Prop := y^2 = a * x
def distance_from_focus_to_directrix (d : ℝ) : Prop := d = 2

-- Statement of the problem
theorem parabola_focus_directrix {a : ℝ} (h : parabola y a x) (h2 : distance_from_focus_to_directrix d) : 
  a = 4 ∨ a = -4 :=
sorry

end parabola_focus_directrix_l174_174794


namespace smallest_n_divisibility_l174_174428

theorem smallest_n_divisibility (n : ℕ) (h : n = 5) : 
  ∃ k, (1 ≤ k ∧ k ≤ n + 1) ∧ (n^2 - n) % k = 0 ∧ ∃ i, (1 ≤ i ∧ i ≤ n + 1) ∧ (n^2 - n) % i ≠ 0 :=
by
  sorry

end smallest_n_divisibility_l174_174428


namespace insulation_cost_l174_174266

def tank_length : ℕ := 4
def tank_width : ℕ := 5
def tank_height : ℕ := 2
def cost_per_sqft : ℕ := 20

def surface_area (L W H : ℕ) : ℕ := 2 * (L * W + L * H + W * H)
def total_cost (SA cost_per_sqft : ℕ) : ℕ := SA * cost_per_sqft

theorem insulation_cost : 
  total_cost (surface_area tank_length tank_width tank_height) cost_per_sqft = 1520 :=
by
  sorry

end insulation_cost_l174_174266


namespace largest_q_value_l174_174700

theorem largest_q_value : ∃ q, q >= 1 ∧ q^4 - q^3 - q - 1 ≤ 0 ∧ (∀ r, r >= 1 ∧ r^4 - r^3 - r - 1 ≤ 0 → r ≤ q) ∧ q = (Real.sqrt 5 + 1) / 2 := 
sorry

end largest_q_value_l174_174700


namespace solution_to_system_of_equations_l174_174301

def augmented_matrix_system_solution (x y : ℝ) : Prop :=
  (x + 3 * y = 5) ∧ (2 * x + 4 * y = 6)

theorem solution_to_system_of_equations :
  ∃! (x y : ℝ), augmented_matrix_system_solution x y ∧ x = -1 ∧ y = 2 :=
by {
  sorry
}

end solution_to_system_of_equations_l174_174301


namespace students_per_configuration_l174_174712

theorem students_per_configuration (students_per_column : ℕ → ℕ) :
  students_per_column 1 = 15 ∧
  students_per_column 2 = 1 ∧
  students_per_column 3 = 1 ∧
  students_per_column 4 = 6 ∧
  ∀ i j, (i ≠ j ∧ i ≤ 12 ∧ j ≤ 12) → students_per_column i ≠ students_per_column j →
  (∃ n, 13 ≤ n ∧ ∀ k, k < 13 → students_per_column k * n = 60) :=
by
  sorry

end students_per_configuration_l174_174712


namespace grade_assignment_ways_l174_174821

/-- Define the number of students and the number of grade choices -/
def num_students : ℕ := 15
def num_grades : ℕ := 4

/-- Define the total number of ways to assign grades -/
def total_ways : ℕ := num_grades ^ num_students

/-- Prove that the total number of ways to assign grades is 4^15 -/
theorem grade_assignment_ways : total_ways = 1073741824 := by
  -- proof here
  sorry

end grade_assignment_ways_l174_174821


namespace loaves_count_l174_174508

theorem loaves_count 
  (init_loaves : ℕ)
  (sold_percent : ℕ) 
  (bulk_purchase : ℕ)
  (bulk_discount_percent : ℕ)
  (evening_purchase : ℕ)
  (evening_discount_percent : ℕ)
  (final_loaves : ℕ)
  (h1 : init_loaves = 2355)
  (h2 : sold_percent = 30)
  (h3 : bulk_purchase = 750)
  (h4 : bulk_discount_percent = 20)
  (h5 : evening_purchase = 489)
  (h6 : evening_discount_percent = 15)
  (h7 : final_loaves = 2888) :
  let mid_morning_sold := init_loaves * sold_percent / 100
  let loaves_after_sale := init_loaves - mid_morning_sold
  let bulk_discount_loaves := bulk_purchase * bulk_discount_percent / 100
  let loaves_after_bulk_purchase := loaves_after_sale + bulk_purchase
  let evening_discount_loaves := evening_purchase * evening_discount_percent / 100
  let loaves_after_evening_purchase := loaves_after_bulk_purchase + evening_purchase
  loaves_after_evening_purchase = final_loaves :=
by
  sorry

end loaves_count_l174_174508


namespace bob_age_is_eleven_l174_174005

/-- 
Susan, Arthur, Tom, and Bob are siblings. Arthur is 2 years older than Susan, 
Tom is 3 years younger than Bob. Susan is 15 years old, 
and the total age of all four family members is 51 years. 
This theorem states that Bob is 11 years old.
-/

theorem bob_age_is_eleven
  (S A T B : ℕ)
  (h1 : A = S + 2)
  (h2 : T = B - 3)
  (h3 : S = 15)
  (h4 : S + A + T + B = 51) : 
  B = 11 :=
  sorry

end bob_age_is_eleven_l174_174005


namespace other_factor_of_product_l174_174817

def product_has_factors (n : ℕ) : Prop :=
  ∃ a b c d e f : ℕ, n = (2^a) * (3^b) * (5^c) * (7^d) * (11^e) * (13^f) ∧ a ≥ 4 ∧ b ≥ 3

def smallest_w (x : ℕ) : ℕ :=
  if h : x = 1452 then 468 else 1

theorem other_factor_of_product (w : ℕ) : 
  (product_has_factors (1452 * w)) → (w = 468) :=
by
  sorry

end other_factor_of_product_l174_174817


namespace fractions_order_l174_174506

theorem fractions_order :
  let frac1 := (21 : ℚ) / (17 : ℚ)
  let frac2 := (23 : ℚ) / (19 : ℚ)
  let frac3 := (25 : ℚ) / (21 : ℚ)
  frac3 < frac2 ∧ frac2 < frac1 :=
by sorry

end fractions_order_l174_174506


namespace find_positive_solutions_l174_174193

theorem find_positive_solutions (x₁ x₂ x₃ x₄ x₅ : ℝ) (h_pos : 0 < x₁ ∧ 0 < x₂ ∧ 0 < x₃ ∧ 0 < x₄ ∧ 0 < x₅)
    (h1 : x₁ + x₂ = x₃^2)
    (h2 : x₂ + x₃ = x₄^2)
    (h3 : x₃ + x₄ = x₅^2)
    (h4 : x₄ + x₅ = x₁^2)
    (h5 : x₅ + x₁ = x₂^2) :
    x₁ = 2 ∧ x₂ = 2 ∧ x₃ = 2 ∧ x₄ = 2 ∧ x₅ = 2 := 
    by {
        -- Proof goes here
        sorry
    }

end find_positive_solutions_l174_174193


namespace cost_of_first_20_kgs_l174_174256

theorem cost_of_first_20_kgs 
  (l m n : ℕ) 
  (hl1 : 30 * l +  3 * m = 333) 
  (hl2 : 30 * l +  6 * m = 366) 
  (hl3 : 30 * l + 15 * m = 465) 
  (hl4 : 30 * l + 20 * m = 525) 
  : 20 * l = 200 :=
by
  sorry

end cost_of_first_20_kgs_l174_174256


namespace find_x_for_prime_power_l174_174679

theorem find_x_for_prime_power (x : ℤ) :
  (∃ p k : ℕ, Nat.Prime p ∧ k > 0 ∧ (2 * x * x + x - 6 = p ^ k)) → (x = -3 ∨ x = 2 ∨ x = 5) := by
  sorry

end find_x_for_prime_power_l174_174679


namespace factorize_m_square_minus_16_l174_174515

-- Define the expression
def expr (m : ℝ) : ℝ := m^2 - 16

-- Define the factorized form
def factorized_expr (m : ℝ) : ℝ := (m + 4) * (m - 4)

-- State the theorem
theorem factorize_m_square_minus_16 (m : ℝ) : expr m = factorized_expr m :=
by
  sorry

end factorize_m_square_minus_16_l174_174515


namespace eval_expression_l174_174760

theorem eval_expression :
  72 + (120 / 15) + (18 * 19) - 250 - (360 / 6) = 112 :=
by sorry

end eval_expression_l174_174760


namespace code_word_MEET_l174_174925

def translate_GREAT_TIME : String → ℕ 
| "G" => 0
| "R" => 1
| "E" => 2
| "A" => 3
| "T" => 4
| "I" => 5
| "M" => 6
| _   => 0 -- Default case for simplicity, not strictly necessary

theorem code_word_MEET : translate_GREAT_TIME "M" = 6 ∧ translate_GREAT_TIME "E" = 2 ∧ translate_GREAT_TIME "T" = 4 →
  let MEET : ℕ := (translate_GREAT_TIME "M" * 1000) + 
                  (translate_GREAT_TIME "E" * 100) + 
                  (translate_GREAT_TIME "E" * 10) + 
                  (translate_GREAT_TIME "T")
  MEET = 6224 :=
sorry

end code_word_MEET_l174_174925


namespace probability_at_least_three_prime_dice_l174_174912

-- Definitions from the conditions
def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11

def p := 5 / 12
def q := 7 / 12
def binomial (n k : ℕ) := Nat.choose n k

-- The probability of at least three primes
theorem probability_at_least_three_prime_dice :
  (binomial 5 3 * p ^ 3 * q ^ 2) +
  (binomial 5 4 * p ^ 4 * q ^ 1) +
  (binomial 5 5 * p ^ 5 * q ^ 0) = 40625 / 622080 :=
by
  sorry

end probability_at_least_three_prime_dice_l174_174912


namespace max_value_set_x_graph_transformation_l174_174644

noncomputable def function_y (x : ℝ) : ℝ := 2 * Real.sin (2 * x + (Real.pi / 6)) + 2

theorem max_value_set_x :
  ∃ k : ℤ, ∀ x : ℝ, x = k * Real.pi + Real.pi / 6 → function_y x = 4 :=
by
  sorry

theorem graph_transformation :
  ∀ x : ℝ, ∃ y : ℝ, (y = Real.sin x → y = 2 * Real.sin (2 * x + (Real.pi / 6)) + 2) :=
by
  sorry

end max_value_set_x_graph_transformation_l174_174644


namespace gasoline_tank_capacity_l174_174090

theorem gasoline_tank_capacity
  (initial_fill : ℝ) (final_fill : ℝ) (gallons_used : ℝ) (x : ℝ)
  (h1 : initial_fill = 3 / 4)
  (h2 : final_fill = 1 / 3)
  (h3 : gallons_used = 18)
  (h4 : initial_fill * x - final_fill * x = gallons_used) :
  x = 43 :=
by
  -- Skipping the proof
  sorry

end gasoline_tank_capacity_l174_174090


namespace smallest_sum_is_4_9_l174_174728

theorem smallest_sum_is_4_9 :
  min
    (min
      (min
        (min (1/3 + 1/4) (1/3 + 1/5))
        (min (1/3 + 1/6) (1/3 + 1/7)))
      (1/3 + 1/9)) = 4/9 :=
  by sorry

end smallest_sum_is_4_9_l174_174728


namespace shortest_handspan_is_Doyoon_l174_174440

def Sangwon_handspan_cm : ℝ := 19.8
def Doyoon_handspan_cm : ℝ := 18.9
def Changhyeok_handspan_cm : ℝ := 19.3

theorem shortest_handspan_is_Doyoon :
  Doyoon_handspan_cm < Sangwon_handspan_cm ∧ Doyoon_handspan_cm < Changhyeok_handspan_cm :=
by
  sorry

end shortest_handspan_is_Doyoon_l174_174440


namespace largest_number_l174_174183

theorem largest_number (A B C D E : ℝ) (hA : A = 0.998) (hB : B = 0.9899) (hC : C = 0.9) (hD : D = 0.9989) (hE : E = 0.8999) :
  D > A ∧ D > B ∧ D > C ∧ D > E :=
by
  sorry

end largest_number_l174_174183


namespace exam_cutoff_mark_l174_174806

theorem exam_cutoff_mark
  (num_students : ℕ)
  (absent_percentage : ℝ)
  (fail_percentage : ℝ)
  (fail_mark_diff : ℝ)
  (just_pass_percentage : ℝ)
  (remaining_avg_mark : ℝ)
  (class_avg_mark : ℝ)
  (absent_students : ℕ)
  (fail_students : ℕ)
  (just_pass_students : ℕ)
  (remaining_students : ℕ)
  (total_marks : ℝ)
  (P : ℝ) :
  absent_percentage = 0.2 →
  fail_percentage = 0.3 →
  fail_mark_diff = 20 →
  just_pass_percentage = 0.1 →
  remaining_avg_mark = 65 →
  class_avg_mark = 36 →
  absent_students = (num_students * absent_percentage) →
  fail_students = (num_students * fail_percentage) →
  just_pass_students = (num_students * just_pass_percentage) →
  remaining_students = num_students - absent_students - fail_students - just_pass_students →
  total_marks = (absent_students * 0) + (fail_students * (P - fail_mark_diff)) + (just_pass_students * P) + (remaining_students * remaining_avg_mark) →
  class_avg_mark = total_marks / num_students →
  P = 40 :=
by
  intros
  sorry

end exam_cutoff_mark_l174_174806


namespace marguerites_fraction_l174_174303

variable (x r b s : ℕ)

theorem marguerites_fraction
  (h1 : r = 5 * (x - r))
  (h2 : b = (x - b) / 5)
  (h3 : r + b + s = x) : s = 0 := by sorry

end marguerites_fraction_l174_174303


namespace total_length_of_ribbon_l174_174198

-- Define the conditions
def length_per_piece : ℕ := 73
def number_of_pieces : ℕ := 51

-- The theorem to prove
theorem total_length_of_ribbon : length_per_piece * number_of_pieces = 3723 :=
by
  sorry

end total_length_of_ribbon_l174_174198


namespace solve_for_x_l174_174234

theorem solve_for_x (x : ℝ) (h : 5 * x + 3 = 10 * x - 22) : x = 5 :=
sorry

end solve_for_x_l174_174234


namespace find_altitude_to_hypotenuse_l174_174380

-- define the conditions
def area : ℝ := 540
def hypotenuse : ℝ := 36
def altitude : ℝ := 30

-- define the problem statement
theorem find_altitude_to_hypotenuse (A : ℝ) (c : ℝ) (h : ℝ) 
  (h_area : A = 540) (h_hypotenuse : c = 36) : h = 30 :=
by
  -- skipping the proof
  sorry

end find_altitude_to_hypotenuse_l174_174380


namespace cold_brew_cost_l174_174051

theorem cold_brew_cost :
  let drip_coffee_cost := 2.25
  let espresso_cost := 3.50
  let latte_cost := 4.00
  let vanilla_syrup_cost := 0.50
  let cappuccino_cost := 3.50
  let total_order_cost := 25.00
  let drip_coffee_total := 2 * drip_coffee_cost
  let lattes_total := 2 * latte_cost
  let known_costs := drip_coffee_total + espresso_cost + lattes_total + vanilla_syrup_cost + cappuccino_cost
  total_order_cost - known_costs = 5.00 →
  5.00 / 2 = 2.50 := by sorry

end cold_brew_cost_l174_174051


namespace probability_of_pulling_blue_ball_l174_174292

def given_conditions (total_balls : ℕ) (initial_blue_balls : ℕ) (blue_balls_removed : ℕ) :=
  total_balls = 15 ∧ initial_blue_balls = 7 ∧ blue_balls_removed = 3

theorem probability_of_pulling_blue_ball
  (total_balls : ℕ) (initial_blue_balls : ℕ) (blue_balls_removed : ℕ)
  (hc : given_conditions total_balls initial_blue_balls blue_balls_removed) :
  ((initial_blue_balls - blue_balls_removed) / (total_balls - blue_balls_removed) : ℚ) = 1 / 3 :=
by
  sorry

end probability_of_pulling_blue_ball_l174_174292


namespace mole_can_sustain_l174_174727

noncomputable def mole_winter_sustainability : Prop :=
  ∃ (grain millet : ℕ), 
    grain = 8 ∧ 
    millet = 0 ∧ 
    ∀ (month : ℕ), 1 ≤ month ∧ month ≤ 3 → 
      ((grain ≥ 3 ∧ (grain - 3) + millet <= 12) ∨ 
      (grain ≥ 1 ∧ millet ≥ 3 ∧ (grain - 1) + (millet - 3) <= 12)) ∧
      ((∃ grain_exchanged millet_gained : ℕ, 
         grain_exchanged ≤ grain ∧
         millet_gained = 2 * grain_exchanged ∧
         grain - grain_exchanged + millet_gained <= 12 ∧
         grain = grain - grain_exchanged) → 
      (grain = 0 ∧ millet = 0))

theorem mole_can_sustain : mole_winter_sustainability := 
sorry 

end mole_can_sustain_l174_174727


namespace probability_not_losing_l174_174676

theorem probability_not_losing (P_winning P_drawing : ℚ)
  (h_winning : P_winning = 1/3)
  (h_drawing : P_drawing = 1/4) :
  P_winning + P_drawing = 7/12 := 
by
  sorry

end probability_not_losing_l174_174676


namespace division_of_powers_of_ten_l174_174031

theorem division_of_powers_of_ten : 10^8 / (2 * 10^6) = 50 := by 
  sorry

end division_of_powers_of_ten_l174_174031


namespace cauchy_schwarz_inequality_l174_174476

theorem cauchy_schwarz_inequality (a b x y : ℝ) :
  (a^2 + b^2) * (x^2 + y^2) ≥ (a * x + b * y)^2 :=
by
  sorry

end cauchy_schwarz_inequality_l174_174476


namespace edge_length_of_cube_l174_174688

/--
Given:
1. A cuboid with base width of 70 cm, base length of 40 cm, and height of 150 cm.
2. A cube-shaped cabinet whose volume is 204,000 cm³ smaller than that of the cuboid.

Prove that one edge of the cube-shaped cabinet is 60 cm.
-/
theorem edge_length_of_cube (W L H V_diff : ℝ) (cuboid_vol : ℝ) (cube_vol : ℝ) (edge : ℝ) :
  W = 70 ∧ L = 40 ∧ H = 150 ∧ V_diff = 204000 ∧ 
  cuboid_vol = W * L * H ∧ cube_vol = cuboid_vol - V_diff ∧ edge ^ 3 = cube_vol -> 
  edge = 60 :=
by
  sorry

end edge_length_of_cube_l174_174688


namespace adjacent_irreducible_rationals_condition_l174_174767

theorem adjacent_irreducible_rationals_condition 
  (a b c d : ℕ) 
  (hab_cop : Nat.gcd a b = 1) (hcd_cop : Nat.gcd c d = 1) 
  (h_ab_prod : a * b < 1988) (h_cd_prod : c * d < 1988) 
  (adj : ∀ p q r s, (Nat.gcd p q = 1) → (Nat.gcd r s = 1) → 
                  (p * q < 1988) → (r * s < 1988) →
                  (p / q < r / s) → (p * s - q * r = 1)) : 
  b * c - a * d = 1 :=
sorry

end adjacent_irreducible_rationals_condition_l174_174767


namespace lines_perpendicular_l174_174716

structure Vec3 :=
(x : ℝ) 
(y : ℝ) 
(z : ℝ)

def line1_dir (x : ℝ) : Vec3 := ⟨x, -1, 2⟩
def line2_dir : Vec3 := ⟨2, 1, 4⟩

def dot_product (v1 v2 : Vec3) : ℝ :=
v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

theorem lines_perpendicular (x : ℝ) :
  dot_product (line1_dir x) line2_dir = 0 ↔ x = -7 / 2 :=
by sorry

end lines_perpendicular_l174_174716


namespace difference_in_zits_l174_174645

variable (avgZitsSwanson : ℕ := 5)
variable (avgZitsJones : ℕ := 6)
variable (numKidsSwanson : ℕ := 25)
variable (numKidsJones : ℕ := 32)
variable (totalZitsSwanson : ℕ := avgZitsSwanson * numKidsSwanson)
variable (totalZitsJones : ℕ := avgZitsJones * numKidsJones)

theorem difference_in_zits :
  totalZitsJones - totalZitsSwanson = 67 := by
  sorry

end difference_in_zits_l174_174645


namespace positive_difference_of_two_numbers_l174_174588

variable {x y : ℝ}

theorem positive_difference_of_two_numbers (h₁ : x + y = 8) (h₂ : x^2 - y^2 = 24) : |x - y| = 3 :=
by
  sorry

end positive_difference_of_two_numbers_l174_174588


namespace min_abs_sum_is_5_l174_174911

noncomputable def min_abs_sum (x : ℝ) : ℝ :=
  |x + 1| + |x + 3| + |x + 6|

theorem min_abs_sum_is_5 : ∃ x : ℝ, (∀ y : ℝ, min_abs_sum y ≥ min_abs_sum x) ∧ min_abs_sum x = 5 :=
by
  use -3
  sorry

end min_abs_sum_is_5_l174_174911


namespace compare_negatives_l174_174383

theorem compare_negatives : -3 < -2 := 
by { sorry }

end compare_negatives_l174_174383


namespace garden_width_l174_174733

theorem garden_width (w : ℝ) (h : ℝ) 
  (h1 : w * h ≥ 150)
  (h2 : h = w + 20)
  (h3 : 2 * (w + h) ≤ 70) :
  w = -10 + 5 * Real.sqrt 10 :=
by sorry

end garden_width_l174_174733


namespace maximum_triangle_area_l174_174233

-- Define the maximum area of a triangle given two sides.
theorem maximum_triangle_area (a b : ℝ) (h_a : a = 1984) (h_b : b = 2016) :
  ∃ (max_area : ℝ), max_area = 1998912 :=
by
  sorry

end maximum_triangle_area_l174_174233


namespace hyperbola_eccentricity_l174_174529

variable (a b c e : ℝ)
variable (a_pos : a > 0)
variable (b_pos : b > 0)
variable (hyperbola_eq : c = Real.sqrt (a^2 + b^2))
variable (y_B : ℝ)
variable (slope_eq : 3 = (y_B - 0) / (c - a))
variable (y_B_on_hyperbola : y_B = b^2 / a)

theorem hyperbola_eccentricity (h : a > 0) (h' : b > 0) (c_def : c = Real.sqrt (a^2 + b^2))
    (slope_cond : 3 = (y_B - 0) / (c - a)) (y_B_cond : y_B = b^2 / a) :
    e = 2 :=
sorry

end hyperbola_eccentricity_l174_174529


namespace simplify_fraction_l174_174593

theorem simplify_fraction : (81000 ^ 3) / (27000 ^ 3) = 27 := by
  sorry

end simplify_fraction_l174_174593


namespace Jean_had_41_candies_at_first_l174_174127

-- Let total_candies be the initial number of candies Jean had
variable (total_candies : ℕ)
-- Jean gave 18 pieces to a friend
def given_away := 18
-- Jean ate 7 pieces
def eaten := 7
-- Jean has 16 pieces left now
def remaining := 16

-- Calculate the total number of candies initially
def candy_initial (total_candies given_away eaten remaining : ℕ) : Prop :=
  total_candies = remaining + (given_away + eaten)

-- Prove that Jean had 41 pieces of candy initially
theorem Jean_had_41_candies_at_first : candy_initial 41 given_away eaten remaining :=
by
  -- Skipping the proof for now
  sorry

end Jean_had_41_candies_at_first_l174_174127


namespace smallest_n_for_divisibility_by_ten_million_l174_174561

theorem smallest_n_for_divisibility_by_ten_million 
  (a₁ a₂ : ℝ) 
  (a₁_eq : a₁ = 5 / 6) 
  (a₂_eq : a₂ = 30) 
  (n : ℕ) 
  (T : ℕ → ℝ) 
  (T_def : ∀ (k : ℕ), T k = a₁ * (36 ^ (k - 1))) :
  (∃ n, T n = T 9 ∧ (∃ m : ℤ, T n = m * 10^7)) := 
sorry

end smallest_n_for_divisibility_by_ten_million_l174_174561


namespace tom_and_elizabeth_climb_ratio_l174_174106

theorem tom_and_elizabeth_climb_ratio :
  let elizabeth_time := 30
  let tom_time_hours := 2
  let tom_time_minutes := tom_time_hours * 60
  (tom_time_minutes / elizabeth_time) = 4 :=
by sorry

end tom_and_elizabeth_climb_ratio_l174_174106


namespace fixed_points_and_zeros_no_fixed_points_range_b_l174_174986

def f (b c x : ℝ) : ℝ := x^2 + b * x + c

theorem fixed_points_and_zeros (b c : ℝ) (h1 : f b c (-3) = -3) (h2 : f b c 2 = 2) :
  ∃ x1 x2 : ℝ, f b c x1 = 0 ∧ f b c x2 = 0 ∧ x1 = -1 + Real.sqrt 7 ∧ x2 = -1 - Real.sqrt 7 :=
sorry

theorem no_fixed_points_range_b {b : ℝ} (h : ∀ x : ℝ, f b (b^2 / 4) x ≠ x) : 
  b > 1 / 3 ∨ b < -1 :=
sorry

end fixed_points_and_zeros_no_fixed_points_range_b_l174_174986


namespace second_frog_hops_eq_18_l174_174011

-- Define the given conditions
variables (x : ℕ) (h3 : ℕ)

def second_frog_hops := 2 * h3
def first_frog_hops := 4 * second_frog_hops
def total_hops := h3 + second_frog_hops + first_frog_hops

-- The proof goal
theorem second_frog_hops_eq_18 (H : total_hops = 99) : second_frog_hops = 18 :=
by
  sorry

end second_frog_hops_eq_18_l174_174011


namespace intersection_A_B_l174_174163

-- Define the sets A and B
def A : Set ℤ := {1, 3, 5, 7}
def B : Set ℤ := {x | 2 ≤ x ∧ x ≤ 5}

-- The goal is to prove that A ∩ B = {3, 5}
theorem intersection_A_B : A ∩ B = {3, 5} :=
  sorry

end intersection_A_B_l174_174163


namespace chips_left_uneaten_l174_174103

theorem chips_left_uneaten 
    (chips_per_cookie : ℕ)
    (cookies_per_dozen : ℕ)
    (dozens_of_cookies : ℕ)
    (cookies_eaten_ratio : ℕ) 
    (h_chips : chips_per_cookie = 7)
    (h_cookies_dozen : cookies_per_dozen = 12)
    (h_dozens : dozens_of_cookies = 4)
    (h_eaten_ratio : cookies_eaten_ratio = 2) : 
  (cookies_per_dozen * dozens_of_cookies / cookies_eaten_ratio) * chips_per_cookie = 168 :=
by 
  sorry

end chips_left_uneaten_l174_174103


namespace sum_a3_a4_a5_a6_l174_174926

theorem sum_a3_a4_a5_a6 (S : ℕ → ℕ) (h : ∀ n, S n = n^2 + 2 * n) : S 6 - S 2 = 40 :=
by
  sorry

end sum_a3_a4_a5_a6_l174_174926


namespace gasoline_price_percent_increase_l174_174565

theorem gasoline_price_percent_increase 
  (highest_price : ℕ) (lowest_price : ℕ) 
  (h_highest : highest_price = 17) 
  (h_lowest : lowest_price = 10) : 
  (highest_price - lowest_price) * 100 / lowest_price = 70 := 
by 
  sorry

end gasoline_price_percent_increase_l174_174565


namespace carterHas152Cards_l174_174555

-- Define the number of baseball cards Marcus has.
def marcusCards : Nat := 210

-- Define the number of baseball cards Carter has.
def carterCards : Nat := marcusCards - 58

-- Theorem to prove Carter's baseball cards total 152 given the conditions.
theorem carterHas152Cards (h1 : marcusCards = 210) (h2 : marcusCards = carterCards + 58) : carterCards = 152 :=
by
  -- Proof omitted for this exercise
  sorry

end carterHas152Cards_l174_174555


namespace exists_integers_m_n_l174_174629

theorem exists_integers_m_n (a b c p q r : ℝ) (h_a : a ≠ 0) (h_p : p ≠ 0) :
  ∃ (m n : ℤ), ∀ (x : ℝ), (a * x^2 + b * x + c = m * (p * x^2 + q * x + r) + n) := sorry

end exists_integers_m_n_l174_174629


namespace age_of_25th_student_l174_174184

variable (total_students : ℕ) (total_average : ℕ)
variable (group1_students : ℕ) (group1_average : ℕ)
variable (group2_students : ℕ) (group2_average : ℕ)

theorem age_of_25th_student 
  (h1 : total_students = 25) 
  (h2 : total_average = 25)
  (h3 : group1_students = 10)
  (h4 : group1_average = 22)
  (h5 : group2_students = 14)
  (h6 : group2_average = 28) : 
  (total_students * total_average) =
  (group1_students * group1_average) + (group2_students * group2_average) + 13 :=
by sorry

end age_of_25th_student_l174_174184


namespace observations_count_l174_174055

theorem observations_count (n : ℕ) 
  (original_mean : ℚ) (wrong_value_corrected : ℚ) (corrected_mean : ℚ)
  (h1 : original_mean = 36)
  (h2 : wrong_value_corrected = 1)
  (h3 : corrected_mean = 36.02) :
  n = 50 :=
by
  sorry

end observations_count_l174_174055


namespace parking_savings_l174_174442

theorem parking_savings (weekly_cost : ℕ) (monthly_cost : ℕ) (weeks_in_year : ℕ) (months_in_year : ℕ)
  (h_weekly_cost : weekly_cost = 10)
  (h_monthly_cost : monthly_cost = 42)
  (h_weeks_in_year : weeks_in_year = 52)
  (h_months_in_year : months_in_year = 12) :
  weekly_cost * weeks_in_year - monthly_cost * months_in_year = 16 := 
by
  sorry

end parking_savings_l174_174442


namespace tan_sin_cos_l174_174610

theorem tan_sin_cos (θ : ℝ) (h : Real.tan θ = 1 / 2) : 
  Real.sin (2 * θ) - 2 * Real.cos θ ^ 2 = - 4 / 5 := by 
  sorry

end tan_sin_cos_l174_174610


namespace matthew_egg_rolls_l174_174439

theorem matthew_egg_rolls (A P M : ℕ) 
  (h1 : M = 3 * P) 
  (h2 : P = A / 2) 
  (h3 : A = 4) : 
  M = 6 :=
by
  sorry

end matthew_egg_rolls_l174_174439


namespace remainder_when_divided_by_6_l174_174533

theorem remainder_when_divided_by_6 (n : ℕ) (h1 : Nat.Prime (n + 3)) (h2 : Nat.Prime (n + 7)) : n % 6 = 4 :=
  sorry

end remainder_when_divided_by_6_l174_174533


namespace abs_diff_inequality_l174_174685

theorem abs_diff_inequality (m : ℝ) : (∃ x : ℝ, |x + 2| - |x + 3| > m) ↔ m < -1 :=
sorry

end abs_diff_inequality_l174_174685


namespace polynomial_expansion_l174_174787

theorem polynomial_expansion (x : ℝ) :
  (5 * x^2 + 3 * x - 7) * (4 * x^3) = 20 * x^5 + 12 * x^4 - 28 * x^3 :=
by 
  sorry

end polynomial_expansion_l174_174787


namespace relationship_between_x_and_y_l174_174859

variables (x y : ℝ)

theorem relationship_between_x_and_y (h1 : x + y > 2 * x) (h2 : x - y < 2 * y) : y > x := 
sorry

end relationship_between_x_and_y_l174_174859


namespace possible_values_of_c_l174_174846

theorem possible_values_of_c (a b c : ℕ) (n : ℕ) (h₀ : a ≠ 0) (h₁ : n = 729 * a + 81 * b + 36 + c) (h₂ : ∃ k, n = k^3) :
  c = 1 ∨ c = 8 :=
sorry

end possible_values_of_c_l174_174846


namespace triangle_area_l174_174933

theorem triangle_area (x : ℝ) (h1 : 6 * x = 6) (h2 : 8 * x = 8) (h3 : 10 * x = 2 * 5) : 
  1 / 2 * 6 * 8 = 24 := 
sorry

end triangle_area_l174_174933


namespace total_fence_poles_l174_174971

def num_poles_per_side : ℕ := 27
def sides_of_square : ℕ := 4
def corners_of_square : ℕ := 4

theorem total_fence_poles : 
  (num_poles_per_side * sides_of_square) - corners_of_square = 104 :=
  sorry

end total_fence_poles_l174_174971


namespace inequality_solution_l174_174119

theorem inequality_solution (x : ℝ) : 
  (x < 2 ∨ x = 3) ↔ (x - 3) / ((x - 2) * (x - 3)) ≤ 0 := 
by {
  sorry
}

end inequality_solution_l174_174119


namespace determinant_eq_sum_of_products_l174_174344

theorem determinant_eq_sum_of_products (x y z : ℝ) :
  Matrix.det (Matrix.of ![![1, x + z, y], ![1, x + y + z, y + z], ![1, x + z, x + y + z]]) = x * y + y * z + z * x :=
by
  sorry

end determinant_eq_sum_of_products_l174_174344


namespace parabola_focus_l174_174654

theorem parabola_focus (x y : ℝ) (h : y = 4 * x^2) : (0, 1 / 16) = (0, 1 / 16) :=
by
  sorry

end parabola_focus_l174_174654


namespace sandy_initial_payment_l174_174990

variable (P : ℝ) 

theorem sandy_initial_payment
  (h1 : (1.2 : ℝ) * (P + 200) = 1200) :
  P = 800 :=
by
  -- Proof goes here
  sorry

end sandy_initial_payment_l174_174990


namespace fifteen_percent_minus_70_l174_174450

theorem fifteen_percent_minus_70 (a : ℝ) : 0.15 * a - 70 = (15 / 100) * a - 70 :=
by sorry

end fifteen_percent_minus_70_l174_174450


namespace find_percentage_l174_174703

theorem find_percentage (P : ℝ) : 
  (∀ x : ℝ, x = 0.40 * 800 → x = P / 100 * 650 + 190) → P = 20 := 
by
  intro h
  sorry

end find_percentage_l174_174703


namespace functional_expression_and_range_l174_174270

-- We define the main problem conditions and prove the required statements based on those conditions
theorem functional_expression_and_range (x y : ℝ) (h1 : ∃ k : ℝ, (y + 2) = k * (4 - x) ∧ k ≠ 0)
                                        (h2 : x = 3 → y = 1) :
                                        (y = -3 * x + 10) ∧ ( -2 < y ∧ y < 1 → 3 < x ∧ x < 4) :=
by
  sorry

end functional_expression_and_range_l174_174270


namespace parabola_shifted_left_and_down_l174_174928

-- Define the original parabolic equation
def original_parabola (x : ℝ) : ℝ := 2 * x ^ 2 - 1

-- Define the transformed parabolic equation
def transformed_parabola (x : ℝ) : ℝ := 2 * (x + 1) ^ 2 - 3

-- Theorem statement
theorem parabola_shifted_left_and_down :
  ∀ x : ℝ, transformed_parabola x = 2 * (x + 1) ^ 2 - 3 :=
by 
  -- Proof Left as an exercise.
  sorry

end parabola_shifted_left_and_down_l174_174928


namespace positive_divisors_multiple_of_5_l174_174627

theorem positive_divisors_multiple_of_5 (a b c : ℕ) (h_a : 0 ≤ a ∧ a ≤ 2) (h_b : 0 ≤ b ∧ b ≤ 3) (h_c : 1 ≤ c ∧ c ≤ 2) :
  (a * b * c = 3 * 4 * 2) :=
sorry

end positive_divisors_multiple_of_5_l174_174627


namespace combined_flock_size_after_5_years_l174_174268

noncomputable def initial_flock_size : ℕ := 100
noncomputable def ducks_killed_per_year : ℕ := 20
noncomputable def ducks_born_per_year : ℕ := 30
noncomputable def years_passed : ℕ := 5
noncomputable def other_flock_size : ℕ := 150

theorem combined_flock_size_after_5_years
  (init_size : ℕ := initial_flock_size)
  (killed_per_year : ℕ := ducks_killed_per_year)
  (born_per_year : ℕ := ducks_born_per_year)
  (years : ℕ := years_passed)
  (other_size : ℕ := other_flock_size) :
  init_size + (years * (born_per_year - killed_per_year)) + other_size = 300 := by
  -- The formal proof would go here.
  sorry

end combined_flock_size_after_5_years_l174_174268


namespace part_a_total_time_part_b_average_time_part_c_probability_l174_174682

theorem part_a_total_time :
  ∃ (total_combinations: ℕ) (time_per_attempt: ℕ) (total_time: ℕ),
    total_combinations = Nat.choose 10 3 ∧ 
    time_per_attempt = 2 ∧ 
    total_time = total_combinations * time_per_attempt / 60 ∧ 
    total_time = 4 := sorry

theorem part_b_average_time :
  ∃ (total_combinations: ℕ) (avg_attempts: ℚ) (time_per_attempt: ℕ) (avg_time: ℚ),
    total_combinations = Nat.choose 10 3 ∧ 
    avg_attempts = (1 + total_combinations) / 2 ∧ 
    time_per_attempt = 2 ∧ 
    avg_time = (avg_attempts * time_per_attempt) / 60 ∧ 
    avg_time = 2 + 1 / 60 := sorry

theorem part_c_probability :
  ∃ (total_combinations: ℕ) (time_limit: ℕ) (attempt_in_time: ℕ) (probability: ℚ),
    total_combinations = Nat.choose 10 3 ∧ 
    time_limit = 60 ∧ 
    attempt_in_time = time_limit / 2 ∧ 
    probability = (attempt_in_time - 1) / total_combinations ∧ 
    probability = 29 / 120 := sorry

end part_a_total_time_part_b_average_time_part_c_probability_l174_174682


namespace speed_of_student_B_l174_174785

open Function
open Real

theorem speed_of_student_B (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) (b_speed : ℝ) :
  distance = 12 ∧ speed_ratio = 1.2 ∧ time_difference = 1 / 6 → b_speed = 12 :=
by
  intro h
  have h1 := h.1
  have h2 := (h.2).1
  have h3 := (h.2).2
  sorry

end speed_of_student_B_l174_174785


namespace carnations_count_l174_174215

theorem carnations_count (c : ℕ) : 
  (9 * 6 = 54) ∧ (47 ≤ c + 47) ∧ (c + 47 = 54) → c = 7 := 
by
  sorry

end carnations_count_l174_174215


namespace cannot_be_square_difference_l174_174400

def square_difference_formula (a b : ℝ) : ℝ := a^2 - b^2

def expression_A (x : ℝ) : ℝ := (x + 1) * (x - 1)
def expression_B (x : ℝ) : ℝ := (-x + 1) * (-x - 1)
def expression_C (x : ℝ) : ℝ := (x + 1) * (-x + 1)
def expression_D (x : ℝ) : ℝ := (x + 1) * (1 + x)

theorem cannot_be_square_difference (x : ℝ) : 
  ¬ (∃ a b, (x + 1) * (1 + x) = square_difference_formula a b) := 
sorry

end cannot_be_square_difference_l174_174400


namespace sheep_count_l174_174556

theorem sheep_count {c s : ℕ} 
  (h1 : c + s = 20)
  (h2 : 2 * c + 4 * s = 60) : s = 10 :=
sorry

end sheep_count_l174_174556


namespace player_holds_seven_black_cards_l174_174078

theorem player_holds_seven_black_cards
    (total_cards : ℕ := 13)
    (num_red_cards : ℕ := 6)
    (S D H C : ℕ)
    (h1 : D = 2 * S)
    (h2 : H = 2 * D)
    (h3 : C = 6)
    (h4 : S + D + H + C = total_cards) :
    S + C = 7 := 
by
  sorry

end player_holds_seven_black_cards_l174_174078


namespace find_A_l174_174084

theorem find_A (A B C : ℕ) (h1 : A = B * C + 8) (h2 : A + B + C = 2994) : A = 8 ∨ A = 2864 :=
by
  sorry

end find_A_l174_174084


namespace num_factors_x_l174_174558

theorem num_factors_x (x : ℕ) (h : 2011^(2011^2012) = x^x) : ∃ n : ℕ, n = 2012 ∧  ∀ d : ℕ, d ∣ x -> d ≤ n :=
sorry

end num_factors_x_l174_174558


namespace c_share_l174_174547

theorem c_share (a b c d e : ℝ) (k : ℝ)
  (h1 : a + b + c + d + e = 1010)
  (h2 : a - 25 = 4 * k)
  (h3 : b - 10 = 3 * k)
  (h4 : c - 15 = 6 * k)
  (h5 : d - 20 = 2 * k)
  (h6 : e - 30 = 5 * k) :
  c = 288 :=
by
  -- proof with necessary steps
  sorry

end c_share_l174_174547


namespace cost_of_each_candy_bar_l174_174260

theorem cost_of_each_candy_bar
  (p_chips : ℝ)
  (total_cost : ℝ)
  (num_students : ℕ)
  (num_chips_per_student : ℕ)
  (num_candy_bars_per_student : ℕ)
  (h1 : p_chips = 0.50)
  (h2 : total_cost = 15)
  (h3 : num_students = 5)
  (h4 : num_chips_per_student = 2)
  (h5 : num_candy_bars_per_student = 1) :
  ∃ C : ℝ, C = 2 := 
by 
  sorry

end cost_of_each_candy_bar_l174_174260


namespace painting_cost_in_cny_l174_174624

theorem painting_cost_in_cny (usd_to_nad : ℝ) (usd_to_cny : ℝ) (painting_cost_nad : ℝ) :
  usd_to_nad = 8 → usd_to_cny = 7 → painting_cost_nad = 160 →
  painting_cost_nad / usd_to_nad * usd_to_cny = 140 :=
by
  intros
  sorry

end painting_cost_in_cny_l174_174624


namespace jar_last_days_l174_174016

theorem jar_last_days :
  let serving_size := 0.5 -- each serving is 0.5 ounces
  let daily_servings := 3  -- James uses 3 servings every day
  let quart_ounces := 32   -- 1 quart = 32 ounces
  let jar_size := quart_ounces - 2 -- container is 2 ounces less than 1 quart
  let daily_consumption := daily_servings * serving_size
  let number_of_days := jar_size / daily_consumption
  number_of_days = 20 := by
  sorry

end jar_last_days_l174_174016


namespace questions_left_blank_l174_174965

-- Definitions based on the conditions
def total_questions : Nat := 60
def word_problems : Nat := 20
def add_subtract_problems : Nat := 25
def algebra_problems : Nat := 10
def geometry_problems : Nat := 5
def total_time : Nat := 90

def time_per_word_problem : Nat := 2
def time_per_add_subtract_problem : Float := 1.5
def time_per_algebra_problem : Nat := 3
def time_per_geometry_problem : Nat := 4

def word_problems_answered : Nat := 15
def add_subtract_problems_answered : Nat := 22
def algebra_problems_answered : Nat := 8
def geometry_problems_answered : Nat := 3

-- The final goal is to prove that Steve left 12 questions blank
theorem questions_left_blank :
  total_questions - (word_problems_answered + add_subtract_problems_answered + algebra_problems_answered + geometry_problems_answered) = 12 :=
by
  sorry

end questions_left_blank_l174_174965


namespace min_value_of_f_l174_174175

noncomputable def f (x : ℝ) : ℝ := (4 * x^2 - 2 * x + 16) / (2 * x - 1)

theorem min_value_of_f :
  ∃ x : ℝ, x ≥ 1 ∧ f x = 9 ∧ (∀ y : ℝ, y ≥ 1 → f y ≥ 9) :=
by { sorry }

end min_value_of_f_l174_174175


namespace divisor_problem_l174_174738

theorem divisor_problem (n : ℕ) (hn_pos : 0 < n) (h72 : Nat.totient n = 72) (h5n : Nat.totient (5 * n) = 96) : ∃ k : ℕ, (n = 5^k * m ∧ Nat.gcd m 5 = 1) ∧ k = 2 :=
by
  sorry

end divisor_problem_l174_174738


namespace original_savings_l174_174582

variable (A B : ℕ)

-- A's savings are 5 times that of B's savings
def cond1 : Prop := A = 5 * B

-- If A withdraws 60 yuan and B deposits 60 yuan, then B's savings will be twice that of A's savings
def cond2 : Prop := (B + 60) = 2 * (A - 60)

-- Prove the original savings of A and B
theorem original_savings (h1 : cond1 A B) (h2 : cond2 A B) : A = 100 ∧ B = 20 := by
  sorry

end original_savings_l174_174582


namespace age_difference_l174_174390

variable (a b c : ℕ)

theorem age_difference (h : a + b = b + c + 13) : a - c = 13 :=
by
  sorry

end age_difference_l174_174390


namespace larger_cuboid_length_is_16_l174_174711

def volume (l w h : ℝ) : ℝ := l * w * h

def cuboid_length_proof : Prop :=
  ∀ (length_large : ℝ), 
  (volume 5 4 3 * 32 = volume length_large 10 12) → 
  length_large = 16

theorem larger_cuboid_length_is_16 : cuboid_length_proof :=
by
  intros length_large eq_volume
  sorry

end larger_cuboid_length_is_16_l174_174711


namespace find_unknown_number_l174_174596

theorem find_unknown_number (x : ℕ) :
  (x + 30 + 50) / 3 = ((20 + 40 + 6) / 3 + 8) → x = 10 := by
    sorry

end find_unknown_number_l174_174596


namespace line_common_chord_eq_l174_174718

theorem line_common_chord_eq (a b : ℝ) :
  (∀ (x1 x2 y1 y2 : ℝ), x1^2 + y1^2 = 1 → (x2 - a)^2 + (y2 - b)^2 = 1 → 
    2 * a * x2 + 2 * b * y2 - 3 = 0) :=
sorry

end line_common_chord_eq_l174_174718


namespace amin_probability_four_attempts_before_three_hits_amin_probability_not_qualified_stops_after_two_consecutive_misses_l174_174065

/-- Prove that the probability Amin makes 4 attempts before hitting 3 times (given the probability of each hit is 1/2) is 3/16. -/
theorem amin_probability_four_attempts_before_three_hits (p_hit : ℚ := 1 / 2) : 
  (∃ (P : ℚ), P = 3/16) :=
sorry

/-- Prove that the probability Amin stops shooting after missing two consecutive shots and not qualifying as level B or A player is 25/32, given the probability of each hit is 1/2. -/
theorem amin_probability_not_qualified_stops_after_two_consecutive_misses (p_hit : ℚ := 1 / 2) : 
  (∃ (P : ℚ), P = 25/32) :=
sorry

end amin_probability_four_attempts_before_three_hits_amin_probability_not_qualified_stops_after_two_consecutive_misses_l174_174065


namespace greatest_num_fruit_in_each_basket_l174_174077

theorem greatest_num_fruit_in_each_basket : 
  let oranges := 15
  let peaches := 9
  let pears := 18
  let gcd := Nat.gcd (Nat.gcd oranges peaches) pears
  gcd = 3 :=
by
  sorry

end greatest_num_fruit_in_each_basket_l174_174077


namespace largest_of_seven_consecutive_l174_174497

theorem largest_of_seven_consecutive (n : ℕ) 
  (h1: n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 3010) :
  n + 6 = 433 :=
by 
  sorry

end largest_of_seven_consecutive_l174_174497


namespace necessary_and_sufficient_condition_l174_174477

noncomputable def f (a x : ℝ) : ℝ := a * x - x^2

theorem necessary_and_sufficient_condition (a : ℝ) (h : 0 < a) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f a x ≤ 1) ↔ (0 < a ∧ a ≤ 2) := by
  sorry

end necessary_and_sufficient_condition_l174_174477


namespace least_positive_integer_satisfying_conditions_l174_174444

theorem least_positive_integer_satisfying_conditions :
  ∃ b : ℕ, b > 0 ∧ (b % 7 = 6) ∧ (b % 11 = 10) ∧ (b % 13 = 12) ∧ b = 1000 :=
by
  sorry

end least_positive_integer_satisfying_conditions_l174_174444


namespace sum_possible_x_coordinates_l174_174112

-- Define the vertices of the parallelogram
def A := (1, 2)
def B := (3, 8)
def C := (4, 1)

-- Definition of what it means to be a fourth vertex that forms a parallelogram
def is_fourth_vertex (D : ℤ × ℤ) : Prop :=
  (D = (6, 7)) ∨ (D = (2, -5)) ∨ (D = (0, 9))

-- The sum of possible x-coordinates for the fourth vertex
def sum_x_coordinates : ℤ :=
  6 + 2 + 0

theorem sum_possible_x_coordinates :
  (∃ D, is_fourth_vertex D) → sum_x_coordinates = 8 :=
by
  -- Sorry is used to skip the detailed proof steps
  sorry

end sum_possible_x_coordinates_l174_174112


namespace monica_book_ratio_theorem_l174_174825

/-
Given:
1. Monica read 16 books last year.
2. This year, she read some multiple of the number of books she read last year.
3. Next year, she will read 69 books.
4. Next year, she wants to read 5 more than twice the number of books she read this year.

Prove:
The ratio of the number of books she read this year to the number of books she read last year is 2.
-/

noncomputable def monica_book_ratio_proof : Prop :=
  let last_year_books := 16
  let next_year_books := 69
  ∃ (x : ℕ), (∃ (n : ℕ), x = last_year_books * n) ∧ (2 * x + 5 = next_year_books) ∧ (x / last_year_books = 2)

theorem monica_book_ratio_theorem : monica_book_ratio_proof :=
  by
    sorry

end monica_book_ratio_theorem_l174_174825


namespace evaluate_expression_l174_174543

theorem evaluate_expression : 
  let a := 3 
  let b := 2 
  (a^2 + b)^2 - (a^2 - b)^2 + 2*a*b = 78 := 
by
  let a := 3
  let b := 2
  sorry

end evaluate_expression_l174_174543


namespace total_students_l174_174389

def Varsity_students : ℕ := 1300
def Northwest_students : ℕ := 1400
def Central_students : ℕ := 1800
def Greenbriar_students : ℕ := 1650

theorem total_students : Varsity_students + Northwest_students + Central_students + Greenbriar_students = 6150 :=
by
  -- Proof is omitted, so we use sorry.
  sorry

end total_students_l174_174389


namespace pencil_case_cost_l174_174748

-- Defining given conditions
def initial_amount : ℕ := 10
def toy_truck_cost : ℕ := 3
def remaining_amount : ℕ := 5
def total_spent : ℕ := initial_amount - remaining_amount

-- Proof statement
theorem pencil_case_cost : total_spent - toy_truck_cost = 2 :=
by
  sorry

end pencil_case_cost_l174_174748


namespace trains_cross_time_l174_174774

theorem trains_cross_time
  (length_each_train : ℝ)
  (speed_each_train_kmh : ℝ)
  (relative_speed_m_s : ℝ)
  (total_distance : ℝ)
  (conversion_factor : ℝ) :
  length_each_train = 120 →
  speed_each_train_kmh = 27 →
  conversion_factor = 1000 / 3600 →
  relative_speed_m_s = speed_each_train_kmh * conversion_factor →
  total_distance = 2 * length_each_train →
  total_distance / relative_speed_m_s = 16 :=
by
  sorry

end trains_cross_time_l174_174774


namespace nina_homework_total_l174_174742

def ruby_math_homework : ℕ := 6

def ruby_reading_homework : ℕ := 2

def nina_math_homework : ℕ := ruby_math_homework * 4 + ruby_math_homework

def nina_reading_homework : ℕ := ruby_reading_homework * 8 + ruby_reading_homework

def nina_total_homework : ℕ := nina_math_homework + nina_reading_homework

theorem nina_homework_total :
  nina_total_homework = 48 :=
by
  unfold nina_total_homework
  unfold nina_math_homework
  unfold nina_reading_homework
  unfold ruby_math_homework
  unfold ruby_reading_homework
  sorry

end nina_homework_total_l174_174742


namespace width_of_field_l174_174947

noncomputable def field_width 
  (field_length : ℝ) 
  (rope_length : ℝ)
  (grazing_area : ℝ) : ℝ :=
if field_length > 2 * rope_length 
then rope_length
else grazing_area

theorem width_of_field 
  (field_length : ℝ := 45)
  (rope_length : ℝ := 22)
  (grazing_area : ℝ := 380.132711084365) : field_width field_length rope_length grazing_area = rope_length :=
by 
  sorry

end width_of_field_l174_174947


namespace solution_to_functional_equation_l174_174129

noncomputable def find_functions (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x = f (x / 2) + (x / 2) * (deriv f x)

theorem solution_to_functional_equation :
  ∀ (f : ℝ → ℝ), (∀ x : ℝ, f x = f (x / 2) + (x / 2) * (deriv f x)) ↔ (∃ c b : ℝ, ∀ x : ℝ, f x = c * x + b) :=
by {
  sorry
}

end solution_to_functional_equation_l174_174129


namespace four_digit_sum_10_divisible_by_9_is_0_l174_174617

theorem four_digit_sum_10_divisible_by_9_is_0 : 
  ∀ (N : ℕ), (1000 * ((N / 1000) % 10) + 100 * ((N / 100) % 10) + 10 * ((N / 10) % 10) + (N % 10) = 10) ∧ (N % 9 = 0) → false :=
by
  sorry

end four_digit_sum_10_divisible_by_9_is_0_l174_174617


namespace smallest_root_of_g_l174_174105

noncomputable def g (x : ℝ) : ℝ := 21 * x^4 - 20 * x^2 + 3

theorem smallest_root_of_g : ∀ x : ℝ, g x = 0 → x = - Real.sqrt (3 / 7) :=
by
  sorry

end smallest_root_of_g_l174_174105


namespace multiplication_schemes_correct_l174_174801

theorem multiplication_schemes_correct :
  ∃ A B C D E F G H I K L M N P : ℕ,
    A = 7 ∧ B = 7 ∧ C = 4 ∧ D = 4 ∧ E = 3 ∧ F = 0 ∧ G = 8 ∧ H = 3 ∧ I = 3 ∧ K = 8 ∧ L = 8 ∧ M = 0 ∧ N = 7 ∧ P = 7 ∧
    (A * 10 + B) * (C * 10 + D) * (A * 10 + B) = E * 100 + F * 10 + G ∧
    (C * 10 + G) * (K * 10 + L) = A * 100 + M * 10 + C ∧
    E * 100 + F * 10 + G / (H * 1000 + I * 100 + G * 10 + G) = (E * 100 + F * 10 + G) / (H * 1000 + I * 100 + G * 10 + G) ∧
    (A * 100 + M * 10 + C) / (N * 1000 + P * 100 + C * 10 + C) = (A * 100 + M * 10 + C) / (N * 1000 + P * 100 + C * 10 + C) :=
sorry

end multiplication_schemes_correct_l174_174801


namespace equation_of_parametrized_curve_l174_174172

theorem equation_of_parametrized_curve :
  ∀ t : ℝ, let x := 3 * t + 6 
           let y := 5 * t - 8 
           ∃ (m b : ℝ), y = m * x + b ∧ m = 5 / 3 ∧ b = -18 :=
by
  sorry

end equation_of_parametrized_curve_l174_174172


namespace largest_pack_size_of_markers_l174_174258

theorem largest_pack_size_of_markers (markers_John markers_Alex : ℕ) (h_John : markers_John = 36) (h_Alex : markers_Alex = 60) : 
  ∃ (n : ℕ), (∀ (x : ℕ), (∀ (y : ℕ), (x * n = markers_John ∧ y * n = markers_Alex) → n ≤ 12) ∧ (12 * x = markers_John ∨ 12 * y = markers_Alex)) :=
by 
  sorry

end largest_pack_size_of_markers_l174_174258


namespace congruence_theorem_l174_174486

def triangle_congruent_SSA (a b : ℝ) (gamma : ℝ) :=
  b * b = a * a + (-2 * a * 5 * Real.cos gamma) + 25

theorem congruence_theorem : triangle_congruent_SSA 3 5 (150 * Real.pi / 180) :=
by
  -- Proof is omitted, based on the problem's instruction.
  sorry

end congruence_theorem_l174_174486


namespace max_non_overlapping_triangles_l174_174557

variable (L : ℝ) (n : ℕ)
def equilateral_triangle (L : ℝ) := true   -- Placeholder definition for equilateral triangle 
def non_overlapping_interior := true        -- Placeholder definition for non-overlapping condition
def unit_triangle_orientation_shift := true -- Placeholder for orientation condition

theorem max_non_overlapping_triangles (L_pos : 0 < L)
                                    (h1 : equilateral_triangle L)
                                    (h2 : ∀ i, i < n → non_overlapping_interior)
                                    (h3 : ∀ i, i < n → unit_triangle_orientation_shift) :
                                    n ≤ (2 : ℝ) / 3 * L^2 := 
by 
  sorry

end max_non_overlapping_triangles_l174_174557


namespace find_number_l174_174314

theorem find_number (x : ℝ) : 
  (72 = 0.70 * x + 30) -> x = 60 :=
by
  sorry

end find_number_l174_174314


namespace number_of_repeating_decimals_l174_174647

open Nat

theorem number_of_repeating_decimals :
  ∀ n : ℕ, (1 ≤ n ∧ n ≤ 15) → (¬ ∃ k : ℕ, k * 18 = n) :=
by
  intros n h
  sorry

end number_of_repeating_decimals_l174_174647


namespace correct_solution_to_equation_l174_174671

theorem correct_solution_to_equation :
  ∃ x m : ℚ, (m = 3 ∧ x = 14 / 23 → 7 * (2 - 2 * x) = 3 * (3 * x - m) + 63) ∧ (∃ x : ℚ, (∃ m : ℚ, m = 3) ∧ (7 * (2 - 2 * x) - (3 * (3 * x - 3) + 63) = 0)) →
  x = 2 := 
sorry

end correct_solution_to_equation_l174_174671


namespace angle_bisector_segment_rel_l174_174653

variable (a b c : ℝ) -- The sides of the triangle
variable (u v : ℝ)   -- The segments into which fa divides side a
variable (fa : ℝ)    -- The length of the angle bisector

-- Statement setting up the given conditions and the proof we need
theorem angle_bisector_segment_rel : 
  (u : ℝ) = a * c / (b + c) → 
  (v : ℝ) = a * b / (b + c) → 
  (fa : ℝ) = 2 * (Real.sqrt (b * s * (s - a) * c)) / (b + c) → 
  fa^2 = b * c - u * v :=
sorry

end angle_bisector_segment_rel_l174_174653


namespace fraction_students_say_dislike_but_actually_like_is_25_percent_l174_174772

variable (total_students : Nat) (students_like_dancing : Nat) (students_dislike_dancing : Nat) 
         (students_like_dancing_but_say_dislike : Nat) (students_dislike_dancing_and_say_dislike : Nat) 
         (total_say_dislike : Nat)

def fraction_of_students_who_say_dislike_but_actually_like (total_students students_like_dancing students_dislike_dancing 
         students_like_dancing_but_say_dislike students_dislike_dancing_and_say_dislike total_say_dislike : Nat) : Nat :=
    (students_like_dancing_but_say_dislike * 100) / total_say_dislike

theorem fraction_students_say_dislike_but_actually_like_is_25_percent
  (h1 : total_students = 100)
  (h2 : students_like_dancing = 60)
  (h3 : students_dislike_dancing = 40)
  (h4 : students_like_dancing_but_say_dislike = 12)
  (h5 : students_dislike_dancing_and_say_dislike = 36)
  (h6 : total_say_dislike = 48) :
  fraction_of_students_who_say_dislike_but_actually_like total_students students_like_dancing students_dislike_dancing 
    students_like_dancing_but_say_dislike students_dislike_dancing_and_say_dislike total_say_dislike = 25 :=
by sorry

end fraction_students_say_dislike_but_actually_like_is_25_percent_l174_174772


namespace geometric_sequence_condition_l174_174768

-- Definitions based on conditions
def S (n : ℕ) (m : ℤ) : ℤ := 3^(n + 1) + m
def a1 (m : ℤ) : ℤ := S 1 m
def a_n (n : ℕ) : ℤ := if n = 1 then a1 (-3) else 2 * 3^n

-- The proof statement
theorem geometric_sequence_condition (m : ℤ) (h1 : a1 m = 3^2 + m) (h2 : ∀ n, n ≥ 2 → a_n n = 2 * 3^n) :
  m = -3 :=
sorry

end geometric_sequence_condition_l174_174768


namespace locus_of_Q_is_circle_l174_174498

variables {A B C P Q : ℝ}

def point_on_segment (A B C : ℝ) : Prop := C > A ∧ C < B

def variable_point_on_circle (A B P : ℝ) : Prop := (P - A) * (P - B) = 0

def ratio_condition (C P Q A B : ℝ) : Prop := (P - C) / (C - Q) = (A - C) / (C - B)

def locus_of_Q_circle (A B C P Q : ℝ) : Prop := ∃ B', (C > A ∧ C < B) → (P - A) * (P - B) = 0 → (P - C) / (C - Q) = (A - C) / (C - B) → (Q - B') * (Q - B) = 0

theorem locus_of_Q_is_circle (A B C P Q : ℝ) :
  point_on_segment A B C →
  variable_point_on_circle A B P →
  ratio_condition C P Q A B →
  locus_of_Q_circle A B C P Q :=
by
  sorry

end locus_of_Q_is_circle_l174_174498


namespace range_of_a_l174_174351

   noncomputable section

   variable {f : ℝ → ℝ}

   /-- The requried theorem based on the given conditions and the correct answer -/
   theorem range_of_a (even_f : ∀ x, f (-x) = f x)
                      (increasing_f : ∀ x y, x ≤ y → y ≤ 0 → f x ≤ f y)
                      (h : f a ≤ f 2) : a ≤ -2 ∨ a ≥ 2 :=
   sorry
   
end range_of_a_l174_174351


namespace find_b_for_integer_a_l174_174667

theorem find_b_for_integer_a (a : ℤ) (b : ℝ) (h1 : 0 ≤ b) (h2 : b < 1) (h3 : (a:ℝ)^2 = 2 * b * (a + b)) :
  b = 0 ∨ b = (-1 + Real.sqrt 3) / 2 :=
sorry

end find_b_for_integer_a_l174_174667


namespace asymptotes_of_hyperbola_l174_174861

-- Definitions for the hyperbola and the asymptotes
def hyperbola_equation (x y : ℝ) : Prop := x^2 - 2 * y^2 = 1
def asymptote_equation (x y : ℝ) : Prop := y = (Real.sqrt 2 / 2) * x ∨ y = - (Real.sqrt 2 / 2) * x

-- The theorem to prove
theorem asymptotes_of_hyperbola (x y : ℝ) (h : hyperbola_equation x y) :
  asymptote_equation x y :=
sorry

end asymptotes_of_hyperbola_l174_174861


namespace parabola_focus_l174_174775

theorem parabola_focus (x y : ℝ) (h : y^2 = 8 * x) : (x, y) = (2, 0) :=
sorry

end parabola_focus_l174_174775


namespace arithmetic_seq_infinitely_many_squares_l174_174030

theorem arithmetic_seq_infinitely_many_squares 
  (a d : ℕ) 
  (h : ∃ (n y : ℕ), a + n * d = y^2) : 
  ∃ (m : ℕ), ∀ k : ℕ, ∃ n' y' : ℕ, a + n' * d = y'^2 :=
by sorry

end arithmetic_seq_infinitely_many_squares_l174_174030


namespace range_of_k_l174_174130

theorem range_of_k (k : ℝ) :
  (∀ x : ℤ, ((x^2 - x - 2 > 0) ∧ (2*x^2 + (2*k + 5)*x + 5*k < 0)) ↔ (x = -2)) -> 
  (-3 ≤ k ∧ k < 2) :=
by 
  sorry

end range_of_k_l174_174130


namespace empty_seats_l174_174930

theorem empty_seats (total_seats : ℕ) (people_watching : ℕ) (h_total_seats : total_seats = 750) (h_people_watching : people_watching = 532) : 
  total_seats - people_watching = 218 :=
by
  sorry

end empty_seats_l174_174930


namespace cost_of_expensive_feed_l174_174658

open Lean Real

theorem cost_of_expensive_feed (total_feed : Real)
                              (total_cost_per_pound : Real) 
                              (cheap_feed_weight : Real)
                              (cheap_cost_per_pound : Real)
                              (expensive_feed_weight : Real)
                              (expensive_cost_per_pound : Real):
  total_feed = 35 ∧ 
  total_cost_per_pound = 0.36 ∧ 
  cheap_feed_weight = 17 ∧ 
  cheap_cost_per_pound = 0.18 ∧ 
  expensive_feed_weight = total_feed - cheap_feed_weight →
  total_feed * total_cost_per_pound - cheap_feed_weight * cheap_cost_per_pound = expensive_feed_weight * expensive_cost_per_pound →
  expensive_cost_per_pound = 0.53 :=
by {
  sorry
}

end cost_of_expensive_feed_l174_174658


namespace proportional_function_l174_174731

theorem proportional_function (k m : ℝ) (f : ℝ → ℝ) :
  (∀ x, f x = k * x) →
  f 2 = -4 →
  (∀ x, f x + m = -2 * x + m) →
  f 2 = -4 ∧ (f 1 + m = 1) →
  k = -2 ∧ m = 3 := 
by
  intros h1 h2 h3 h4
  sorry

end proportional_function_l174_174731


namespace find_2a_plus_b_l174_174417

theorem find_2a_plus_b (a b : ℝ) (h1 : 3 * a + 2 * b = 18) (h2 : 5 * a + 4 * b = 31) :
  2 * a + b = 11.5 :=
sorry

end find_2a_plus_b_l174_174417


namespace sequence_term_value_l174_174678

theorem sequence_term_value :
  ∃ (a : ℕ → ℚ), a 1 = 2 ∧ (∀ n, a (n + 1) = a n + 1 / 2) ∧ a 101 = 52 :=
by
  sorry

end sequence_term_value_l174_174678


namespace intersection_of_A_and_B_l174_174980

def setA : Set ℝ := {x | abs (x - 1) < 2}

def setB : Set ℝ := {x | x^2 + x - 2 > 0}

theorem intersection_of_A_and_B :
  (setA ∩ setB) = {x | 1 < x ∧ x < 3} :=
sorry

end intersection_of_A_and_B_l174_174980


namespace arithmetic_mean_geometric_mean_l174_174816

theorem arithmetic_mean_geometric_mean (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : 
  (a + b) / 2 ≥ Real.sqrt (a * b) :=
sorry

end arithmetic_mean_geometric_mean_l174_174816


namespace surface_area_increase_l174_174066

def cube_dimensions : ℝ × ℝ × ℝ := (10, 10, 10)

def number_of_cuts := 3

def initial_surface_area (length : ℝ) (width : ℝ) (height : ℝ) : ℝ :=
  6 * (length * width)

def increase_in_surface_area (cuts : ℕ) (length : ℝ) (width : ℝ) : ℝ :=
  cuts * 2 * (length * width)

theorem surface_area_increase : 
  initial_surface_area 10 10 10 + increase_in_surface_area 3 10 10 = 
  initial_surface_area 10 10 10 + 600 :=
by
  sorry

end surface_area_increase_l174_174066


namespace problem1_problem2_problem3_problem4_l174_174116

-- Proof statement for problem 1
theorem problem1 : (1 : ℤ) * (-8) + 10 + 2 + (-1) = 3 := sorry

-- Proof statement for problem 2
theorem problem2 : (-21.6 : ℝ) - (-3) - |(-7.4)| + (-2 / 5) = -26.4 := sorry

-- Proof statement for problem 3
theorem problem3 : (-12 / 5) / (-1 / 10) * (-5 / 6) * (-0.4 : ℝ) = 8 := sorry

-- Proof statement for problem 4
theorem problem4 : ((5 / 8) - (1 / 6) + (7 / 12)) * (-24 : ℝ) = -25 := sorry

end problem1_problem2_problem3_problem4_l174_174116


namespace range_m_l174_174218

noncomputable def even_function (f : ℝ → ℝ) : Prop := 
  ∀ x, f x = f (-x)

noncomputable def decreasing_on_non_neg (f : ℝ → ℝ) : Prop := 
  ∀ ⦃x y⦄, 0 ≤ x → x ≤ y → f y ≤ f x

theorem range_m (f : ℝ → ℝ)
  (h_even : even_function f)
  (h_dec : decreasing_on_non_neg f) :
  ∀ m, f (1 - m) < f m → m < 1 / 2 :=
by
  sorry

end range_m_l174_174218


namespace quad_area_FDBG_l174_174793

open Real

noncomputable def area_quad_FDBG (AB AC area_ABC : ℝ) : ℝ :=
  let AD := AB / 2
  let AE := AC / 2
  let area_ADE := area_ABC / 4
  let x := 2 * area_ABC / (AB * AC)
  let sin_A := x
  let hyp_ratio := sin_A / (area_ABC / AC)
  let factor := hyp_ratio / 2
  let area_AFG := factor * area_ADE
  area_ABC - area_ADE - 2 * area_AFG

theorem quad_area_FDBG (AB AC area_ABC : ℝ) (hAB : AB = 60) (hAC : AC = 15) (harea : area_ABC = 180) :
  area_quad_FDBG AB AC area_ABC = 117 := by
  sorry

end quad_area_FDBG_l174_174793


namespace abc_inequality_l174_174020

theorem abc_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  a * b + b * c + c * a ≥ a * Real.sqrt (b * c) + b * Real.sqrt (a * c) + c * Real.sqrt (a * b) :=
sorry

end abc_inequality_l174_174020


namespace expected_value_of_biased_die_l174_174354

-- Definitions for probabilities
def prob1 : ℚ := 1 / 15
def prob2 : ℚ := 1 / 15
def prob3 : ℚ := 1 / 15
def prob4 : ℚ := 1 / 15
def prob5 : ℚ := 1 / 5
def prob6 : ℚ := 3 / 5

-- Definition for expected value
def expected_value : ℚ := (prob1 * 1) + (prob2 * 2) + (prob3 * 3) + (prob4 * 4) + (prob5 * 5) + (prob6 * 6)

theorem expected_value_of_biased_die : expected_value = 16 / 3 :=
by sorry

end expected_value_of_biased_die_l174_174354


namespace senior_students_in_sample_l174_174161

theorem senior_students_in_sample 
  (total_students : ℕ) (total_seniors : ℕ) (sample_size : ℕ) 
  (h1 : total_students = 2000)
  (h2 : total_seniors = 500)
  (h3 : sample_size = 200) : 
  (total_seniors * sample_size / total_students = 50) :=
by {
  sorry
}

end senior_students_in_sample_l174_174161


namespace betty_height_in_feet_l174_174478

theorem betty_height_in_feet (dog_height carter_height betty_height : ℕ) (h1 : dog_height = 24) 
  (h2 : carter_height = 2 * dog_height) (h3 : betty_height = carter_height - 12) : betty_height / 12 = 3 :=
by
  sorry

end betty_height_in_feet_l174_174478


namespace find_integer_m_l174_174315

theorem find_integer_m 
  (m : ℤ)
  (h1 : 30 ≤ m ∧ m ≤ 80)
  (h2 : ∃ k : ℤ, m = 6 * k)
  (h3 : m % 8 = 2)
  (h4 : m % 5 = 2) : 
  m = 42 := 
sorry

end find_integer_m_l174_174315


namespace fill_cistern_time_l174_174987

theorem fill_cistern_time (R1 R2 R3 : ℝ) (H1 : R1 = 1/10) (H2 : R2 = 1/12) (H3 : R3 = 1/40) : 
  (1 / (R1 + R2 - R3)) = (120 / 19) :=
by
  sorry

end fill_cistern_time_l174_174987


namespace greatest_divisor_of_product_of_5_consecutive_multiples_of_4_l174_174855

theorem greatest_divisor_of_product_of_5_consecutive_multiples_of_4 :
  let n1 := 4
  let n2 := 8
  let n3 := 12
  let n4 := 16
  let n5 := 20
  let spf1 := 2 -- smallest prime factor of 4
  let spf2 := 2 -- smallest prime factor of 8
  let spf3 := 2 -- smallest prime factor of 12
  let spf4 := 2 -- smallest prime factor of 16
  let spf5 := 2 -- smallest prime factor of 20
  let p1 := n1^spf1
  let p2 := n2^spf2
  let p3 := n3^spf3
  let p4 := n4^spf4
  let p5 := n5^spf5
  let product := p1 * p2 * p3 * p4 * p5
  product % (2^24) = 0 :=
by 
  sorry

end greatest_divisor_of_product_of_5_consecutive_multiples_of_4_l174_174855


namespace find_possible_values_of_n_l174_174993

theorem find_possible_values_of_n (n : ℕ) :
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ n + 1 ∧ 
    (2*n*(2*n + 1))/2 - (n*k + (n*(n-1))/2) = 1615) ↔ (n = 34 ∨ n = 38) :=
by
  sorry

end find_possible_values_of_n_l174_174993


namespace cupcake_cookie_price_ratio_l174_174214

theorem cupcake_cookie_price_ratio
  (c k : ℚ)
  (h1 : 5 * c + 3 * k = 23)
  (h2 : 4 * c + 4 * k = 21) :
  k / c = 13 / 29 :=
  sorry

end cupcake_cookie_price_ratio_l174_174214


namespace max_gcd_of_15n_plus_4_and_9n_plus_2_eq_2_l174_174473

theorem max_gcd_of_15n_plus_4_and_9n_plus_2_eq_2 (n : ℕ) (hn : n > 0) :
  ∃ m, m = Nat.gcd (15 * n + 4) (9 * n + 2) ∧ m ≤ 2 :=
by
  sorry

end max_gcd_of_15n_plus_4_and_9n_plus_2_eq_2_l174_174473


namespace reflection_of_C_over_y_eq_x_l174_174626

def point_reflection_over_yx := ∀ (A B C : (ℝ × ℝ)), 
  A = (6, 2) → 
  B = (2, 5) → 
  C = (2, 2) → 
  (reflect_y_eq_x C) = (2, 2)
where reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

theorem reflection_of_C_over_y_eq_x :
  point_reflection_over_yx :=
by 
  sorry

end reflection_of_C_over_y_eq_x_l174_174626


namespace buckets_needed_l174_174329

variable {C : ℝ} (hC : C > 0)

theorem buckets_needed (h : 42 * C = 42 * C) : 
  (42 * C) / ((2 / 5) * C) = 105 :=
by
  sorry

end buckets_needed_l174_174329


namespace mb_less_than_neg_one_point_five_l174_174157

theorem mb_less_than_neg_one_point_five (m b : ℚ) (h1 : m = 3/4) (h2 : b = -2) : m * b < -1.5 :=
by {
  -- sorry skips the proof
  sorry
}

end mb_less_than_neg_one_point_five_l174_174157


namespace josh_money_left_l174_174546

def initial_amount : ℝ := 20
def cost_hat : ℝ := 10
def cost_pencil : ℝ := 2
def number_of_cookies : ℝ := 4
def cost_per_cookie : ℝ := 1.25

theorem josh_money_left : initial_amount - cost_hat - cost_pencil - (number_of_cookies * cost_per_cookie) = 3 := by
  sorry

end josh_money_left_l174_174546


namespace chameleons_all_white_l174_174921

theorem chameleons_all_white :
  ∀ (a b c : ℕ), a = 800 → b = 1000 → c = 1220 → 
  (a + b + c = 3020) → (a % 3 = 2) → (b % 3 = 1) → (c % 3 = 2) →
    ∃ k : ℕ, (k = 3020 ∧ (k % 3 = 1)) ∧ 
    (if k = b then a = 0 ∧ c = 0 else false) :=
by
  sorry

end chameleons_all_white_l174_174921


namespace solve_for_y_l174_174831

theorem solve_for_y (y : ℚ) (h : 1/3 + 1/y = 7/9) : y = 9/4 :=
by
  sorry

end solve_for_y_l174_174831


namespace concentric_circle_area_ratio_l174_174418

theorem concentric_circle_area_ratio (r R : ℝ) (h_ratio : (π * R^2) / (π * r^2) = 16 / 3) :
  R - r = 1.309 * r :=
by
  sorry

end concentric_circle_area_ratio_l174_174418


namespace diff_of_squares_value_l174_174796

theorem diff_of_squares_value :
  535^2 - 465^2 = 70000 :=
by sorry

end diff_of_squares_value_l174_174796


namespace average_last_30_l174_174009

theorem average_last_30 (avg_first_40 : ℝ) 
  (avg_all_70 : ℝ) 
  (sum_first_40 : ℝ := 40 * avg_first_40)
  (sum_all_70 : ℝ := 70 * avg_all_70) 
  (total_results: ℕ := 70):
  (30 : ℝ) * (40: ℝ) + (30: ℝ) * (40: ℝ) = 70 * 34.285714285714285 :=
by
  sorry

end average_last_30_l174_174009


namespace unique_two_digit_integer_l174_174507

theorem unique_two_digit_integer (t : ℕ) (h : 11 * t % 100 = 36) (ht : 10 ≤ t ∧ t ≤ 99) : t = 76 :=
by
  sorry

end unique_two_digit_integer_l174_174507


namespace find_number_l174_174949

-- Define the conditions
variables (y : ℝ) (Some_number : ℝ) (x : ℝ)

-- State the given equation
def equation := 19 * (x + y) + Some_number = 19 * (-x + y) - 21

-- State the proposition to prove
theorem find_number (h : equation 1 y Some_number) : Some_number = -59 :=
sorry

end find_number_l174_174949


namespace son_age_l174_174743

theorem son_age {x : ℕ} {father son : ℕ} 
  (h1 : father = 4 * son)
  (h2 : (son - 10) + (father - 10) = 60)
  (h3 : son = x)
  : x = 16 := 
sorry

end son_age_l174_174743


namespace find_smaller_integer_l174_174150

theorem find_smaller_integer : ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ y = x + 8 ∧ x * y = 80 ∧ x = 2 :=
by
  sorry

end find_smaller_integer_l174_174150


namespace cuboid_surface_area_two_cubes_l174_174512

noncomputable def cuboid_surface_area (b : ℝ) : ℝ :=
  let l := 2 * b
  let w := b
  let h := b
  2 * (l * w + l * h + w * h)

theorem cuboid_surface_area_two_cubes (b : ℝ) : cuboid_surface_area b = 10 * b^2 := by
  sorry

end cuboid_surface_area_two_cubes_l174_174512


namespace find_four_digit_number_l174_174338

theorem find_four_digit_number : ∃ N : ℕ, 999 < N ∧ N < 10000 ∧ (∃ a : ℕ, a^2 = N) ∧ 
  (∃ b : ℕ, b^3 = N % 1000) ∧ (∃ c : ℕ, c^4 = N % 100) ∧ N = 9216 := 
by
  sorry

end find_four_digit_number_l174_174338


namespace amount_left_for_gas_and_maintenance_l174_174999

def monthly_income : ℤ := 3200
def rent : ℤ := 1250
def utilities : ℤ := 150
def retirement_savings : ℤ := 400
def groceries_eating_out : ℤ := 300
def insurance : ℤ := 200
def miscellaneous : ℤ := 200
def car_payment : ℤ := 350

def total_expenses : ℤ :=
  rent + utilities + retirement_savings + groceries_eating_out + insurance + miscellaneous + car_payment

theorem amount_left_for_gas_and_maintenance : monthly_income - total_expenses = 350 :=
by
  -- Proof is omitted
  sorry

end amount_left_for_gas_and_maintenance_l174_174999


namespace radius_squared_l174_174231

theorem radius_squared (r : ℝ) (AB_len CD_len BP_len : ℝ) (angle_APD : ℝ) (r_squared : ℝ) :
  AB_len = 10 →
  CD_len = 7 →
  BP_len = 8 →
  angle_APD = 60 →
  r_squared = r^2 →
  r_squared = 73 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  sorry

end radius_squared_l174_174231


namespace solve_eq1_solve_eq2_l174_174979

-- Define the problem for equation (1)
theorem solve_eq1 (x : Real) : (x - 1)^2 = 2 ↔ (x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2) :=
by 
  sorry

-- Define the problem for equation (2)
theorem solve_eq2 (x : Real) : x^2 - 6 * x - 7 = 0 ↔ (x = -1 ∨ x = 7) :=
by 
  sorry

end solve_eq1_solve_eq2_l174_174979


namespace systematic_sampling_first_group_l174_174643

theorem systematic_sampling_first_group (x : ℕ) (n : ℕ) (k : ℕ) (total_students : ℕ) (sampled_students : ℕ) 
  (interval : ℕ) (group_num : ℕ) (group_val : ℕ) 
  (h1 : total_students = 1000) (h2 : sampled_students = 40) (h3 : interval = total_students / sampled_students)
  (h4 : interval = 25) (h5 : group_num = 18) 
  (h6 : group_val = 443) (h7 : group_val = x + (group_num - 1) * interval) : 
  x = 18 := 
by 
  sorry

end systematic_sampling_first_group_l174_174643


namespace joe_out_of_money_after_one_month_worst_case_l174_174824

-- Define the initial amount Joe has
def initial_amount : ℝ := 240

-- Define Joe's monthly subscription cost
def subscription_cost : ℝ := 15

-- Define the range of prices for buying games
def min_game_cost : ℝ := 40
def max_game_cost : ℝ := 60

-- Define the range of prices for selling games
def min_resale_price : ℝ := 20
def max_resale_price : ℝ := 40

-- Define the maximum number of games Joe can purchase per month
def max_games_per_month : ℕ := 3

-- Prove that Joe will be out of money after 1 month in the worst-case scenario
theorem joe_out_of_money_after_one_month_worst_case :
  initial_amount - 
  (max_games_per_month * max_game_cost - max_games_per_month * min_resale_price + subscription_cost) < 0 :=
by
  sorry

end joe_out_of_money_after_one_month_worst_case_l174_174824


namespace cos_value_proof_l174_174955

variable (α : Real)
variable (h1 : -Real.pi / 2 < α ∧ α < 0)
variable (h2 : Real.sin (α + Real.pi / 3) + Real.sin α = -(4 * Real.sqrt 3) / 5)

theorem cos_value_proof : Real.cos (α + 2 * Real.pi / 3) = 4 / 5 :=
by
  sorry

end cos_value_proof_l174_174955


namespace ice_cubes_total_l174_174023

theorem ice_cubes_total (initial_cubes made_cubes : ℕ) (h_initial : initial_cubes = 2) (h_made : made_cubes = 7) : initial_cubes + made_cubes = 9 :=
by
  sorry

end ice_cubes_total_l174_174023


namespace calculate_final_amount_l174_174896

def initial_amount : ℝ := 7500
def first_year_rate : ℝ := 0.20
def second_year_rate : ℝ := 0.25

def first_year_interest (p : ℝ) (r : ℝ) : ℝ := p * r
def amount_after_first_year (p : ℝ) (i : ℝ) : ℝ := p + i

def second_year_interest (p : ℝ) (r : ℝ) : ℝ := p * r
def amount_after_second_year (p : ℝ) (i : ℝ) : ℝ := p + i

theorem calculate_final_amount :
  let initial : ℝ := initial_amount
  let interest1 : ℝ := first_year_interest initial first_year_rate
  let amount1 : ℝ := amount_after_first_year initial interest1
  let interest2 : ℝ := second_year_interest amount1 second_year_rate
  let final_amount : ℝ := amount_after_second_year amount1 interest2
  final_amount = 11250 := by
  sorry

end calculate_final_amount_l174_174896


namespace point_of_tangency_l174_174958

theorem point_of_tangency (x y : ℝ) (h : (y = x^3 + x - 2)) (slope : 4 = 3 * x^2 + 1) : (x, y) = (-1, -4) := 
sorry

end point_of_tangency_l174_174958


namespace find_a_l174_174229

theorem find_a (a : ℝ) (h : (2:ℝ)^2 + 2 * a - 3 * a = 0) : a = 4 :=
sorry

end find_a_l174_174229


namespace product_of_two_equal_numbers_l174_174220

-- Definitions and conditions
def arithmetic_mean (xs : List ℚ) : ℚ :=
  xs.sum / xs.length

-- Theorem stating the product of the two equal numbers
theorem product_of_two_equal_numbers (a b c : ℚ) (x : ℚ) :
  arithmetic_mean [a, b, c, x, x] = 20 → a = 22 → b = 18 → c = 32 → x * x = 196 :=
by
  intros h_mean h_a h_b h_c
  sorry

end product_of_two_equal_numbers_l174_174220


namespace simplify_sqrt_450_l174_174200

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_450_l174_174200


namespace students_sign_up_ways_l174_174248

theorem students_sign_up_ways :
  let students := 4
  let choices_per_student := 3
  (choices_per_student ^ students) = 3^4 :=
by
  sorry

end students_sign_up_ways_l174_174248


namespace kate_hair_length_l174_174991

theorem kate_hair_length :
  ∀ (logans_hair : ℕ) (emilys_hair : ℕ) (kates_hair : ℕ),
  logans_hair = 20 →
  emilys_hair = logans_hair + 6 →
  kates_hair = emilys_hair / 2 →
  kates_hair = 13 :=
by
  intros logans_hair emilys_hair kates_hair
  sorry

end kate_hair_length_l174_174991


namespace intersection_points_relation_l174_174293

noncomputable def num_intersections (k : ℕ) : ℕ :=
  k * (k - 1) / 2

theorem intersection_points_relation (k : ℕ) :
  num_intersections (k + 1) = num_intersections k + k := by
sorry

end intersection_points_relation_l174_174293


namespace valid_x_values_l174_174433

noncomputable def valid_triangle_sides (x : ℕ) : Prop :=
  8 + 11 > x + 3 ∧ 8 + (x + 3) > 11 ∧ 11 + (x + 3) > 8

theorem valid_x_values :
  {x : ℕ | valid_triangle_sides x ∧ x > 0} = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15} :=
by
  sorry

end valid_x_values_l174_174433


namespace gcd_459_357_l174_174123

theorem gcd_459_357 : Nat.gcd 459 357 = 51 :=
by
  sorry

end gcd_459_357_l174_174123


namespace work_completion_l174_174205

theorem work_completion (Rp Rq Dp W : ℕ) 
  (h1 : Rq = W / 12) 
  (h2 : W = 4*Rp + 6*(Rp + Rq)) 
  (h3 : Rp = W / Dp) 
  : Dp = 20 :=
by
  sorry

end work_completion_l174_174205


namespace problem_equivalent_proof_l174_174915

variable {a : ℕ → ℝ}
variable {q : ℝ}

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

def seq_condition1 (a : ℕ → ℝ) (q : ℝ) :=
  a 2 * a 4 * a 5 = a 3 * a 6

def seq_condition2 (a : ℕ → ℝ) :=
  a 9 * a 10 = -8

-- The problem to prove
theorem problem_equivalent_proof :
  is_geometric_sequence a q →
  seq_condition1 a q →
  seq_condition2 a →
  a 7 = -2 :=
by
  sorry

end problem_equivalent_proof_l174_174915


namespace compute_fraction_l174_174880

def x : ℚ := 2 / 3
def y : ℚ := 3 / 2
def z : ℚ := 1 / 3

theorem compute_fraction :
  (1 / 3) * x^7 * y^5 * z^4 = 11 / 600 :=
by
  sorry

end compute_fraction_l174_174880


namespace election_winner_votes_l174_174115

theorem election_winner_votes :
  ∃ V W : ℝ, (V = (71.42857142857143 / 100) * V + 3000 + 5000) ∧
            (W = (71.42857142857143 / 100) * V) ∧
            W = 20000 := by
  sorry

end election_winner_votes_l174_174115


namespace volume_and_surface_area_of_inscribed_sphere_l174_174109

theorem volume_and_surface_area_of_inscribed_sphere (edge_length : ℝ) (h_edge : edge_length = 10) :
    let r := edge_length / 2
    let V := (4 / 3) * π * r^3
    let A := 4 * π * r^2
    V = (500 / 3) * π ∧ A = 100 * π := 
by
  sorry

end volume_and_surface_area_of_inscribed_sphere_l174_174109


namespace probability_different_suits_correct_l174_174670

-- Definitions based on conditions
def cards_in_deck : ℕ := 52
def cards_picked : ℕ := 3
def first_card_suit_not_matter : Prop := True
def second_card_different_suit : Prop := True
def third_card_different_suit : Prop := True

-- Definition of the probability function
def probability_different_suits (cards_total : ℕ) (cards_picked : ℕ) : Rat :=
  let first_card_prob := 1
  let second_card_prob := 39 / 51
  let third_card_prob := 26 / 50
  first_card_prob * second_card_prob * third_card_prob

-- The theorem statement to prove the probability each card is of a different suit
theorem probability_different_suits_correct :
  probability_different_suits cards_in_deck cards_picked = 169 / 425 :=
by
  -- Proof should be written here
  sorry

end probability_different_suits_correct_l174_174670


namespace eccentricity_range_of_hyperbola_l174_174978

open Real

noncomputable def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (x^2 / a^2 - y^2 / b^2 = 1)

noncomputable def eccentricity_range :=
  ∀ (a b c : ℝ), 
    ∃ (e : ℝ),
      hyperbola a b (-c) 0 ∧ -- condition for point F
      (a + b > 0) ∧ -- additional conditions due to hyperbola properties
      (1 < e ∧ e < 2)
      
theorem eccentricity_range_of_hyperbola :
  eccentricity_range :=
by
  sorry

end eccentricity_range_of_hyperbola_l174_174978


namespace hypotenuse_longer_side_difference_l174_174628

theorem hypotenuse_longer_side_difference
  (x : ℝ)
  (h1 : 17^2 = x^2 + (x - 7)^2)
  (h2 : x = 15)
  : 17 - x = 2 := by
  sorry

end hypotenuse_longer_side_difference_l174_174628


namespace trees_falling_count_l174_174491

/-- Definition of the conditions of the problem. --/
def initial_mahogany_trees : ℕ := 50
def initial_narra_trees : ℕ := 30
def trees_on_farm_after_typhoon : ℕ := 88

/-- The mathematical proof problem statement in Lean 4:
Prove the total number of trees that fell during the typhoon (N + M) is equal to 5,
given the conditions.
--/
theorem trees_falling_count (M N : ℕ) 
  (h1 : M = N + 1)
  (h2 : (initial_mahogany_trees - M + 3 * M) + (initial_narra_trees - N + 2 * N) = trees_on_farm_after_typhoon) :
  N + M = 5 := sorry

end trees_falling_count_l174_174491


namespace meaningful_fraction_l174_174006

theorem meaningful_fraction (x : ℝ) : (x ≠ 5) ↔ (∃ y : ℝ, y = 1 / (x - 5)) :=
by
  sorry

end meaningful_fraction_l174_174006


namespace cat_food_finished_on_sunday_l174_174160

def cat_morning_consumption : ℚ := 1 / 2
def cat_evening_consumption : ℚ := 1 / 3
def total_food : ℚ := 10
def daily_consumption : ℚ := cat_morning_consumption + cat_evening_consumption
def days_to_finish_food (total_food daily_consumption : ℚ) : ℚ :=
  total_food / daily_consumption

theorem cat_food_finished_on_sunday :
  days_to_finish_food total_food daily_consumption = 7 := 
sorry

end cat_food_finished_on_sunday_l174_174160


namespace largest_three_digit_multiple_of_9_with_digits_sum_27_l174_174457

theorem largest_three_digit_multiple_of_9_with_digits_sum_27 :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ (n % 9 = 0) ∧ ((n / 100) + ((n % 100) / 10) + (n % 10) = 27) ∧ n = 999 :=
by
  sorry

end largest_three_digit_multiple_of_9_with_digits_sum_27_l174_174457


namespace evaluate_custom_operation_l174_174934

def custom_operation (x y : ℕ) : ℕ := 2 * x - 4 * y

theorem evaluate_custom_operation :
  custom_operation 7 3 = 2 :=
by
  sorry

end evaluate_custom_operation_l174_174934


namespace expected_interval_proof_l174_174273

noncomputable def expected_interval_between_trains : ℝ := 3

theorem expected_interval_proof
  (northern_route_time southern_route_time : ℝ)
  (counter_clockwise_delay : ℝ)
  (home_to_work_less_than_work_to_home : ℝ) :
  northern_route_time = 17 →
  southern_route_time = 11 →
  counter_clockwise_delay = 75 / 60 →
  home_to_work_less_than_work_to_home = 1 →
  expected_interval_between_trains = 3 :=
by
  intros
  sorry

end expected_interval_proof_l174_174273


namespace sin_half_alpha_l174_174079

theorem sin_half_alpha (α : ℝ) (hα : 0 < α ∧ α < π / 2) (hcos : Real.cos α = (1 + Real.sqrt 5) / 4) : 
  Real.sin (α / 2) = (Real.sqrt 5 - 1) / 4 := 
by 
  sorry

end sin_half_alpha_l174_174079


namespace green_more_than_blue_l174_174924

theorem green_more_than_blue (B Y G : Nat) (h1 : B + Y + G = 108) (h2 : B * 7 = Y * 3) (h3 : B * 8 = G * 3) : G - B = 30 := by
  sorry

end green_more_than_blue_l174_174924


namespace least_sum_of_exponents_l174_174929

theorem least_sum_of_exponents (a b c : ℕ) (ha : 2^a ∣ 520) (hb : 2^b ∣ 520) (hc : 2^c ∣ 520) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  a + b + c = 12 :=
by
  sorry

end least_sum_of_exponents_l174_174929


namespace product_469157_9999_l174_174397

theorem product_469157_9999 : 469157 * 9999 = 4690872843 := by
  -- computation and its proof would go here
  sorry

end product_469157_9999_l174_174397


namespace problem_maximum_marks_l174_174252

theorem problem_maximum_marks (M : ℝ) (h : 0.92 * M = 184) : M = 200 :=
sorry

end problem_maximum_marks_l174_174252


namespace proportion_fourth_number_l174_174503

theorem proportion_fourth_number (x y : ℝ) (h₀ : 0.75 * y = 5 * x) (h₁ : x = 1.65) : y = 11 :=
by
  sorry

end proportion_fourth_number_l174_174503


namespace trapezium_distance_l174_174072

theorem trapezium_distance (a b area : ℝ) (h : ℝ) :
  a = 20 ∧ b = 18 ∧ area = 266 ∧
  area = (1/2) * (a + b) * h -> h = 14 :=
by
  sorry

end trapezium_distance_l174_174072


namespace transform_into_product_l174_174891

theorem transform_into_product : 447 * (Real.sin (75 * Real.pi / 180) + Real.sin (15 * Real.pi / 180)) = 447 * Real.sqrt 6 / 2 := by
  sorry

end transform_into_product_l174_174891


namespace sum_odd_divisors_90_eq_78_l174_174675

-- Noncomputable is used because we might need arithmetic operations that are not computable in Lean
noncomputable def sum_of_odd_divisors_of_90 : Nat :=
  1 + 3 + 5 + 9 + 15 + 45

theorem sum_odd_divisors_90_eq_78 : sum_of_odd_divisors_of_90 = 78 := 
  by 
    -- The sum is directly given; we don't need to compute it here
    sorry

end sum_odd_divisors_90_eq_78_l174_174675


namespace negation_exists_to_forall_l174_174284

theorem negation_exists_to_forall :
  (¬ ∃ x : ℝ, x^2 + 2*x - 3 > 0) ↔ (∀ x : ℝ, x^2 + 2*x - 3 ≤ 0) :=
by
  sorry

end negation_exists_to_forall_l174_174284


namespace Sasha_earnings_proof_l174_174028

def Monday_hours : ℕ := 90  -- 1.5 hours * 60 minutes/hour
def Tuesday_minutes : ℕ := 75  -- 1 hour * 60 minutes/hour + 15 minutes
def Wednesday_minutes : ℕ := 115  -- 11:10 AM - 9:15 AM
def Thursday_minutes : ℕ := 45

def total_minutes_worked : ℕ := Monday_hours + Tuesday_minutes + Wednesday_minutes + Thursday_minutes

def hourly_rate : ℚ := 4.50
def total_hours : ℚ := total_minutes_worked / 60

def weekly_earnings : ℚ := total_hours * hourly_rate

theorem Sasha_earnings_proof : weekly_earnings = 24 := by
  sorry

end Sasha_earnings_proof_l174_174028


namespace problem_statement_l174_174307

-- Define the repeating decimal and the required gcd condition
def repeating_decimal_value := (356 : ℚ) / 999
def gcd_condition (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Define the main theorem stating the required sum
theorem problem_statement (a b : ℕ) 
                          (h_a : a = 356) 
                          (h_b : b = 999) 
                          (h_gcd : gcd_condition a b) : 
    a + b = 1355 := by
  sorry

end problem_statement_l174_174307


namespace number_division_l174_174071

theorem number_division (x : ℚ) (h : x / 6 = 1 / 10) : (x / (3 / 25)) = 5 :=
by {
  sorry
}

end number_division_l174_174071


namespace find_certain_number_l174_174974

theorem find_certain_number (x : ℝ) : 136 - 0.35 * x = 31 -> x = 300 :=
by
  intro h
  sorry

end find_certain_number_l174_174974


namespace arcsin_one_eq_pi_div_two_l174_174269

theorem arcsin_one_eq_pi_div_two : Real.arcsin 1 = Real.pi / 2 :=
by
  -- proof steps here
  sorry

end arcsin_one_eq_pi_div_two_l174_174269


namespace find_c_l174_174419

-- Defining the variables and conditions given in the problem
variables (a b c : ℝ)

-- Conditions
def vertex_condition : Prop := (2, -3) = (a * (-3)^2 + b * (-3) + c, -3)
def point_condition : Prop := (7, -1) = (a * (-1)^2 + b * (-1) + c, -1)

-- Problem Statement
theorem find_c 
  (h_vertex : vertex_condition a b c)
  (h_point : point_condition a b c) :
  c = 53 / 4 :=
sorry

end find_c_l174_174419


namespace intersection_of_A_and_B_l174_174424

open Finset

def A : Finset ℤ := {-2, -1, 0, 1, 2}
def B : Finset ℤ := {1, 2, 3}

theorem intersection_of_A_and_B :
  A ∩ B = {1, 2} :=
by sorry

end intersection_of_A_and_B_l174_174424


namespace dhoni_toys_average_cost_l174_174732

theorem dhoni_toys_average_cost (A : ℝ) (h1 : ∃ x1 x2 x3 x4 x5, (x1 + x2 + x3 + x4 + x5) / 5 = A)
  (h2 : 5 * A = 5 * A)
  (h3 : ∃ x6, x6 = 16)
  (h4 : (5 * A + 16) / 6 = 11) : A = 10 :=
by
  sorry

end dhoni_toys_average_cost_l174_174732


namespace positive_difference_l174_174122

theorem positive_difference (a b : ℝ) (h₁ : a + b = 10) (h₂ : a^2 - b^2 = 40) : |a - b| = 4 :=
by
  sorry

end positive_difference_l174_174122


namespace binary_to_decimal_conversion_l174_174972

theorem binary_to_decimal_conversion : (1 * 2^5 + 1 * 2^4 + 0 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0) = 51 :=
by 
  sorry

end binary_to_decimal_conversion_l174_174972


namespace find_c_l174_174012

def is_midpoint (p1 p2 mid : ℝ × ℝ) : Prop :=
(mid.1 = (p1.1 + p2.1) / 2) ∧ (mid.2 = (p1.2 + p2.2) / 2)

def is_perpendicular_bisector (line : ℝ → ℝ → ℝ) (p1 p2 : ℝ × ℝ) : Prop := 
∃ mid : ℝ × ℝ, 
is_midpoint p1 p2 mid ∧ line mid.1 mid.2 = 0

theorem find_c (c : ℝ) : 
is_perpendicular_bisector (λ x y => 3 * x - y - c) (2, 4) (6, 8) → c = 6 :=
by
  sorry

end find_c_l174_174012


namespace beads_per_bracelet_is_10_l174_174650

-- Definitions of given conditions
def num_necklaces_Monday : ℕ := 10
def num_necklaces_Tuesday : ℕ := 2
def num_necklaces : ℕ := num_necklaces_Monday + num_necklaces_Tuesday

def beads_per_necklace : ℕ := 20
def beads_necklaces : ℕ := num_necklaces * beads_per_necklace

def num_earrings : ℕ := 7
def beads_per_earring : ℕ := 5
def beads_earrings : ℕ := num_earrings * beads_per_earring

def total_beads_used : ℕ := 325
def beads_used_for_necklaces_and_earrings : ℕ := beads_necklaces + beads_earrings
def beads_remaining_for_bracelets : ℕ := total_beads_used - beads_used_for_necklaces_and_earrings

def num_bracelets : ℕ := 5
def beads_per_bracelet : ℕ := beads_remaining_for_bracelets / num_bracelets

-- Theorem statement to prove
theorem beads_per_bracelet_is_10 : beads_per_bracelet = 10 := by
  sorry

end beads_per_bracelet_is_10_l174_174650


namespace solution_to_exponential_equation_l174_174036

theorem solution_to_exponential_equation :
  ∃ x : ℕ, (8^12 + 8^12 + 8^12 = 2^x) ∧ x = 38 :=
by
  sorry

end solution_to_exponential_equation_l174_174036


namespace triplet_solution_l174_174530

theorem triplet_solution (a b c : ℝ)
  (h1 : a^2 + b = c^2)
  (h2 : b^2 + c = a^2)
  (h3 : c^2 + a = b^2) :
  (a = 0 ∧ b = 0 ∧ c = 0) ∨
  (a = 0 ∧ b = 1 ∧ c = -1) ∨
  (a = -1 ∧ b = 0 ∧ c = 1) ∨
  (a = 1 ∧ b = -1 ∧ c = 0) :=
sorry

end triplet_solution_l174_174530


namespace value_of_f_at_2019_l174_174903

variable (f : ℝ → ℝ)
variable (h_even : ∀ x : ℝ, f x = f (-x))
variable (h_positive : ∀ x : ℝ, f x > 0)
variable (h_functional : ∀ x : ℝ, f (x + 2) = 1 / (f x))

theorem value_of_f_at_2019 : f 2019 = 1 :=
by
  sorry

end value_of_f_at_2019_l174_174903


namespace fraction_of_number_l174_174319

theorem fraction_of_number (F : ℚ) (h : 0.5 * F * 120 = 36) : F = 3 / 5 :=
by
  sorry

end fraction_of_number_l174_174319


namespace number_of_zeros_is_one_l174_174448

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 3 * x

theorem number_of_zeros_is_one : 
  ∃! x : ℝ, f x = 0 :=
sorry

end number_of_zeros_is_one_l174_174448


namespace correct_propositions_l174_174875

def Line := Type
def Plane := Type

variables (m n: Line) (α β γ: Plane)

-- Conditions from the problem statement
axiom perp (x: Line) (y: Plane): Prop -- x ⊥ y
axiom parallel (x: Line) (y: Plane): Prop -- x ∥ y
axiom perp_planes (x: Plane) (y: Plane): Prop -- x ⊥ y
axiom parallel_planes (x: Plane) (y: Plane): Prop -- x ∥ y

-- Given the conditions
axiom h1: perp m α
axiom h2: parallel n α
axiom h3: perp_planes α γ
axiom h4: perp_planes β γ
axiom h5: parallel_planes α β
axiom h6: parallel_planes β γ
axiom h7: parallel m α
axiom h8: parallel n α
axiom h9: perp m n
axiom h10: perp m γ

-- Lean statement for the problem: Prove that Propositions ① and ④ are correct.
theorem correct_propositions : (perp m n) ∧ (perp m γ) :=
by sorry -- Proof steps are not required.

end correct_propositions_l174_174875


namespace x_percent_more_than_y_l174_174811

theorem x_percent_more_than_y (z : ℝ) (hz : z ≠ 0) (y : ℝ) (x : ℝ)
  (h1 : y = 0.70 * z) (h2 : x = 0.84 * z) :
  x = y + 0.20 * y :=
by
  -- proof goes here
  sorry

end x_percent_more_than_y_l174_174811


namespace decompose_expression_l174_174946

-- Define the variables a and b as real numbers
variables (a b : ℝ)

-- State the theorem corresponding to the proof problem
theorem decompose_expression : 9 * a^2 * b - b = b * (3 * a + 1) * (3 * a - 1) :=
by
  sorry

end decompose_expression_l174_174946


namespace quadratic_inequality_solution_l174_174631

noncomputable def solve_inequality (a b : ℝ) : Prop :=
  (∀ x : ℝ, (x > -1/2 ∧ x < 1/3) → (a * x^2 + b * x + 2 > 0)) →
  (a = -12) ∧ (b = -2)

theorem quadratic_inequality_solution :
   solve_inequality (-12) (-2) :=
by
  intro h
  sorry

end quadratic_inequality_solution_l174_174631


namespace probability_N_14_mod_5_is_1_l174_174860

theorem probability_N_14_mod_5_is_1 :
  let total := 1950
  let favorable := 2
  let outcomes := 5
  (favorable / outcomes) = (2 / 5) := by
  sorry

end probability_N_14_mod_5_is_1_l174_174860


namespace min_value_frac_sum_l174_174405

theorem min_value_frac_sum (m n : ℝ) (h1 : 2 * m + n = 2) (h2 : m * n > 0) : 
  ∃ c : ℝ, c = 4 ∧ (∀ m n, 2 * m + n = 2 → m * n > 0 → (1 / m + 2 / n) ≥ c) :=
sorry

end min_value_frac_sum_l174_174405


namespace sum_of_squares_and_cubes_l174_174343

theorem sum_of_squares_and_cubes (a b : ℤ) (h : ∃ k : ℤ, a^2 - 4*b = k^2) :
  ∃ x1 x2 : ℤ, a^2 - 2*b = x1^2 + x2^2 ∧ 3*a*b - a^3 = x1^3 + x2^3 :=
by
  sorry

end sum_of_squares_and_cubes_l174_174343


namespace abc_order_l174_174853

noncomputable def a : ℝ := Real.log (3 / 2) - 3 / 2
noncomputable def b : ℝ := Real.log Real.pi - Real.pi
noncomputable def c : ℝ := Real.log 3 - 3

theorem abc_order : a > c ∧ c > b := by
  have h₁: a = Real.log (3 / 2) - 3 / 2 := rfl
  have h₂: b = Real.log Real.pi - Real.pi := rfl
  have h₃: c = Real.log 3 - 3 := rfl
  sorry

end abc_order_l174_174853


namespace light_travel_50_years_l174_174403

theorem light_travel_50_years :
  let one_year_distance := 9460800000000 -- distance light travels in one year
  let fifty_years_distance := 50 * one_year_distance
  let scientific_notation_distance := 473.04 * 10^12
  fifty_years_distance = scientific_notation_distance :=
by
  sorry

end light_travel_50_years_l174_174403


namespace complex_number_quadrant_l174_174809

open Complex

theorem complex_number_quadrant 
  (a b c d : ℤ) : 
  (a + b * Complex.I) * (c - d * Complex.I) = (a*c + b*d) + (a*d + b*c) * Complex.I → 
  (0 < (a*c + b*d) ∧ 0 < (a*d + b*c)) → 
  True := 
by
  intro h_mul h_coord
  sorry


end complex_number_quadrant_l174_174809


namespace equation1_sol_equation2_sol_equation3_sol_l174_174253

theorem equation1_sol (x : ℝ) : 9 * x^2 - (x - 1)^2 = 0 ↔ (x = -0.5 ∨ x = 0.25) :=
sorry

theorem equation2_sol (x : ℝ) : (x * (x - 3) = 10) ↔ (x = 5 ∨ x = -2) :=
sorry

theorem equation3_sol (x : ℝ) : (x + 3)^2 = 2 * x + 5 ↔ (x = -2) :=
sorry

end equation1_sol_equation2_sol_equation3_sol_l174_174253


namespace contingency_fund_l174_174426

theorem contingency_fund:
  let d := 240
  let cp := d * (1.0 / 3)
  let lc := d * (1.0 / 2)
  let r := d - cp - lc
  let lp := r * (1.0 / 4)
  let cf := r - lp
  cf = 30 := 
by
  sorry

end contingency_fund_l174_174426


namespace pie_chart_probability_l174_174463

theorem pie_chart_probability
  (P_W P_X P_Z : ℚ)
  (h_W : P_W = 1/4)
  (h_X : P_X = 1/3)
  (h_Z : P_Z = 1/6) :
  1 - P_W - P_X - P_Z = 1/4 :=
by
  -- The detailed proof steps are omitted as per the requirement.
  sorry

end pie_chart_probability_l174_174463


namespace square_diagonal_cut_l174_174474

/--
Given a square with side length 10,
prove that cutting along the diagonal results in two 
right-angled isosceles triangles with dimensions 10, 10, 10*sqrt(2).
-/
theorem square_diagonal_cut (side_length : ℕ) (triangle_side1 triangle_side2 hypotenuse : ℝ) 
  (h_side : side_length = 10)
  (h_triangle_side1 : triangle_side1 = 10) 
  (h_triangle_side2 : triangle_side2 = 10)
  (h_hypotenuse : hypotenuse = 10 * Real.sqrt 2) : 
  triangle_side1 = side_length ∧ triangle_side2 = side_length ∧ hypotenuse = side_length * Real.sqrt 2 :=
by
  sorry

end square_diagonal_cut_l174_174474


namespace principal_amount_l174_174944

theorem principal_amount
  (SI : ℝ) (R : ℝ) (T : ℝ)
  (h1 : SI = 155) (h2 : R = 4.783950617283951) (h3 : T = 4) :
  SI * 100 / (R * T) = 810.13 := 
  by 
    -- proof omitted
    sorry

end principal_amount_l174_174944


namespace radius_circle_B_l174_174045

theorem radius_circle_B (rA rB rD : ℝ) 
  (hA : rA = 2) (hD : rD = 2 * rA) (h_tangent : (rA + rB) ^ 2 = rD ^ 2) : 
  rB = 2 :=
by
  sorry

end radius_circle_B_l174_174045


namespace sin_B_triangle_area_l174_174265

variable {a b c : ℝ}
variable {A B C : ℝ}

theorem sin_B (hC : C = 3 / 4 * Real.pi) (hSinA : Real.sin A = Real.sqrt 5 / 5) :
  Real.sin B = Real.sqrt 10 / 10 := by
  sorry

theorem triangle_area (hC : C = 3 / 4 * Real.pi) (hSinA : Real.sin A = Real.sqrt 5 / 5)
  (hDiff : c - a = 5 - Real.sqrt 10) (hSinB : Real.sin B = Real.sqrt 10 / 10) :
  1 / 2 * a * c * Real.sin B = 5 / 2 := by
  sorry

end sin_B_triangle_area_l174_174265


namespace prob_6_higher_than_3_after_10_shuffles_l174_174454

def p_k (k : Nat) : ℚ := (3^k - 2^k) / (2 * 3^k)

theorem prob_6_higher_than_3_after_10_shuffles :
  p_k 10 = (3^10 - 2^10) / (2 * 3^10) :=
by
  sorry

end prob_6_higher_than_3_after_10_shuffles_l174_174454


namespace complement_union_l174_174573

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {2, 4, 5}
def B : Set ℕ := {3, 4, 5}

theorem complement_union :
  ((U \ A) ∪ B) = {1, 3, 4, 5, 6} :=
by
  sorry

end complement_union_l174_174573


namespace alpha_cubed_plus_5beta_plus_10_l174_174206

noncomputable def α: ℝ := sorry
noncomputable def β: ℝ := sorry

-- Given conditions
axiom roots_eq : ∀ x : ℝ, x^2 + 2 * x - 1 = 0 → (x = α ∨ x = β)
axiom sum_eq : α + β = -2
axiom prod_eq : α * β = -1

-- The theorem stating the desired result
theorem alpha_cubed_plus_5beta_plus_10 :
  α^3 + 5 * β + 10 = -2 :=
sorry

end alpha_cubed_plus_5beta_plus_10_l174_174206


namespace interior_edges_sum_l174_174249

theorem interior_edges_sum (frame_width area outer_length : ℝ) (h1 : frame_width = 2) (h2 : area = 30)
  (h3 : outer_length = 7) : 
  2 * (outer_length - 2 * frame_width) + 2 * ((area / outer_length - 4)) = 7 := 
by
  sorry

end interior_edges_sum_l174_174249


namespace complex_number_calculation_l174_174000

theorem complex_number_calculation (i : ℂ) (h : i * i = -1) : i^7 - 2/i = i := 
by 
  sorry

end complex_number_calculation_l174_174000


namespace find_y_l174_174963

theorem find_y 
  (α : Real)
  (P : Real × Real)
  (P_coord : P = (-Real.sqrt 3, y))
  (sin_alpha : Real.sin α = Real.sqrt 13 / 13) :
  P.2 = 1 / 2 :=
by
  sorry

end find_y_l174_174963


namespace cos_6theta_l174_174554

theorem cos_6theta (θ : ℝ) (h : Real.cos θ = 1/4) : Real.cos (6 * θ) = -3224/4096 := 
by
  sorry

end cos_6theta_l174_174554


namespace find_a_c_area_A_90_area_B_90_l174_174574

variable (a b c : ℝ)
variable (C : ℝ)

def triangle_condition1 := a + 1/a = 4 * Real.cos C
def triangle_condition2 := b = 1
def sin_C := Real.sin C = Real.sqrt 21 / 7

-- Proof problem for (1)
theorem find_a_c (h1 : triangle_condition1 a C) (h2 : triangle_condition2 b) (h3 : sin_C C) :
  (a = Real.sqrt 7 ∧ c = 2) ∨ (a = Real.sqrt 7 / 7 ∧ c = 2 * Real.sqrt 7 / 7) :=
sorry

-- Conditions for (2) when A=90°
def right_triangle_A := C = Real.pi / 2

-- Proof problem for (2) when A=90°
theorem area_A_90 (h1 : triangle_condition1 a C) (h2 : triangle_condition2 b) (h4 : right_triangle_A C) :
  ((a = Real.sqrt 3) → area = Real.sqrt 2 / 2) :=
sorry

-- Conditions for (2) when B=90°
def right_triangle_B := b = 1 ∧ C = Real.pi / 2

-- Proof problem for (2) when B=90°
theorem area_B_90 (h1 : triangle_condition1 a C) (h2 : triangle_condition2 b) (h5 : right_triangle_B b C) :
  ((a = Real.sqrt 3 / 3) → area = Real.sqrt 2 / 6) :=
sorry

end find_a_c_area_A_90_area_B_90_l174_174574


namespace infinitely_many_not_sum_of_three_fourth_powers_l174_174781

theorem infinitely_many_not_sum_of_three_fourth_powers : ∀ n : ℕ, n > 0 → n ≡ 5 [MOD 16] → ¬(∃ a b c : ℤ, n = a^4 + b^4 + c^4) :=
by sorry

end infinitely_many_not_sum_of_three_fourth_powers_l174_174781


namespace peanuts_in_box_l174_174250

theorem peanuts_in_box (initial_peanuts : ℕ) (added_peanuts : ℕ) (h1 : initial_peanuts = 4) (h2 : added_peanuts = 2) : initial_peanuts + added_peanuts = 6 := by
  sorry

end peanuts_in_box_l174_174250


namespace MarksScore_l174_174902

theorem MarksScore (h_highest : ℕ) (h_range : ℕ) (h_relation : h_highest - h_least = h_range) (h_mark_twice : Mark = 2 * h_least) : Mark = 46 :=
by
    let h_highest := 98
    let h_range := 75
    let h_least := h_highest - h_range
    let Mark := 2 * h_least
    have := h_relation
    have := h_mark_twice
    sorry

end MarksScore_l174_174902


namespace quadratic_inequality_l174_174992

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_inequality
  (a b c : ℝ)
  (h_pos : 0 < a)
  (h_roots : ∀ x : ℝ, a * x^2 + b * x + c ≠ 0)
  (x : ℝ) :
  f a b c x + f a b c (x - 1) - f a b c (x + 1) > -4 * a :=
  sorry

end quadratic_inequality_l174_174992


namespace round_2741836_to_nearest_integer_l174_174047

theorem round_2741836_to_nearest_integer :
  (2741836.4928375).round = 2741836 := 
by
  -- Explanation that 0.4928375 < 0.5 leading to rounding down
  sorry

end round_2741836_to_nearest_integer_l174_174047


namespace percentage_orange_juice_in_blend_l174_174612

theorem percentage_orange_juice_in_blend :
  let pear_juice_per_pear := 10 / 2
  let orange_juice_per_orange := 8 / 2
  let pear_juice := 2 * pear_juice_per_pear
  let orange_juice := 3 * orange_juice_per_orange
  let total_juice := pear_juice + orange_juice
  (orange_juice / total_juice) = (6 / 11) := 
by
  sorry

end percentage_orange_juice_in_blend_l174_174612


namespace min_square_sum_l174_174262

theorem min_square_sum (a b m n : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 15 * a + 16 * b = m * m) (h4 : 16 * a - 15 * b = n * n) : 481 ≤ min (m * m) (n * n) :=
sorry

end min_square_sum_l174_174262


namespace find_d_l174_174282

theorem find_d (d : ℝ) (h : 3 * (2 - (π / 2)) = 6 + d * π) : d = -3 / 2 :=
by
  sorry

end find_d_l174_174282


namespace chord_length_intercepted_l174_174094

theorem chord_length_intercepted 
  (line_eq : ∀ x y : ℝ, 3 * x - 4 * y = 0)
  (circle_eq : ∀ x y : ℝ, (x - 1)^2 + (y - 2)^2 = 2) : 
  ∃ l : ℝ, l = 2 :=
by 
  sorry

end chord_length_intercepted_l174_174094


namespace stock_price_no_return_l174_174827

/-- Define the increase and decrease factors. --/
def increase_factor := 117 / 100
def decrease_factor := 83 / 100

/-- Define the proof that the stock price cannot return to its initial value after any number of 
    increases and decreases. --/
theorem stock_price_no_return 
  (P0 : ℝ) (k l : ℕ) : 
  P0 * (increase_factor ^ k) * (decrease_factor ^ l) ≠ P0 :=
by
  sorry

end stock_price_no_return_l174_174827


namespace find_c1_minus_c2_l174_174580

theorem find_c1_minus_c2 (c1 c2 : ℝ) (h1 : 2 * 3 + 3 * 5 = c1) (h2 : 5 = c2) : c1 - c2 = 16 := by
  sorry

end find_c1_minus_c2_l174_174580


namespace cubed_identity_l174_174657

variable (x : ℝ)

theorem cubed_identity (h : x + 1/x = 7) : x^3 + 1/x^3 = 322 := 
by
  sorry

end cubed_identity_l174_174657


namespace average_sleep_per_day_l174_174341

-- Define a structure for time duration
structure TimeDuration where
  hours : ℕ
  minutes : ℕ

-- Define instances for each day
def mondayNight : TimeDuration := ⟨8, 15⟩
def mondayNap : TimeDuration := ⟨0, 30⟩
def tuesdayNight : TimeDuration := ⟨7, 45⟩
def tuesdayNap : TimeDuration := ⟨0, 45⟩
def wednesdayNight : TimeDuration := ⟨8, 10⟩
def wednesdayNap : TimeDuration := ⟨0, 50⟩
def thursdayNight : TimeDuration := ⟨10, 25⟩
def thursdayNap : TimeDuration := ⟨0, 20⟩
def fridayNight : TimeDuration := ⟨7, 50⟩
def fridayNap : TimeDuration := ⟨0, 40⟩

-- Function to convert TimeDuration to total minutes
def totalMinutes (td : TimeDuration) : ℕ :=
  td.hours * 60 + td.minutes

-- Define the total sleep time for each day
def mondayTotal := totalMinutes mondayNight + totalMinutes mondayNap
def tuesdayTotal := totalMinutes tuesdayNight + totalMinutes tuesdayNap
def wednesdayTotal := totalMinutes wednesdayNight + totalMinutes wednesdayNap
def thursdayTotal := totalMinutes thursdayNight + totalMinutes thursdayNap
def fridayTotal := totalMinutes fridayNight + totalMinutes fridayNap

-- Sum of all sleep times
def totalSleep := mondayTotal + tuesdayTotal + wednesdayTotal + thursdayTotal + fridayTotal
-- Average sleep in minutes per day
def averageSleep := totalSleep / 5
-- Convert average sleep in total minutes back to hours and minutes
def averageHours := averageSleep / 60
def averageMinutes := averageSleep % 60

theorem average_sleep_per_day :
  averageHours = 9 ∧ averageMinutes = 6 := by
  sorry

end average_sleep_per_day_l174_174341


namespace quotient_is_8_l174_174719

def dividend : ℕ := 64
def divisor : ℕ := 8
def quotient := dividend / divisor

theorem quotient_is_8 : quotient = 8 := 
by 
  show quotient = 8 
  sorry

end quotient_is_8_l174_174719


namespace cylinder_ratio_l174_174968

theorem cylinder_ratio
  (V : ℝ) (R H : ℝ)
  (hV : V = 1000)
  (hVolume : π * R^2 * H = V) :
  H / R = 1 :=
by
  sorry

end cylinder_ratio_l174_174968


namespace smallest_int_rel_prime_150_l174_174914

theorem smallest_int_rel_prime_150 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 150 = 1 ∧ ∀ y : ℕ, (y > 1 ∧ Nat.gcd y 150 = 1) → x ≤ y :=
by
  sorry

end smallest_int_rel_prime_150_l174_174914


namespace repeating_fraction_equality_l174_174614

theorem repeating_fraction_equality : (0.5656565656 : ℚ) = 56 / 99 :=
by
  sorry

end repeating_fraction_equality_l174_174614


namespace negation_of_exists_proposition_l174_174453

theorem negation_of_exists_proposition :
  (¬ ∃ x : ℝ, x^2 - x + 1 = 0) ↔ ∀ x : ℝ, x^2 - x + 1 ≠ 0 :=
sorry

end negation_of_exists_proposition_l174_174453


namespace marbles_solid_color_non_yellow_l174_174707

theorem marbles_solid_color_non_yellow (total_marble solid_colored solid_yellow : ℝ)
    (h1: solid_colored = 0.90 * total_marble)
    (h2: solid_yellow = 0.05 * total_marble) :
    (solid_colored - solid_yellow) / total_marble = 0.85 := by
  -- sorry is used to skip the proof
  sorry

end marbles_solid_color_non_yellow_l174_174707


namespace trigonometric_identity_simplification_l174_174516

theorem trigonometric_identity_simplification :
  (Real.sin (15 * Real.pi / 180) * Real.cos (75 * Real.pi / 180) + Real.cos (15 * Real.pi / 180) * Real.sin (105 * Real.pi / 180) = 1) :=
by sorry

end trigonometric_identity_simplification_l174_174516


namespace no_cube_sum_of_three_consecutive_squares_l174_174621

theorem no_cube_sum_of_three_consecutive_squares :
  ¬∃ x y : ℤ, x^3 = (y-1)^2 + y^2 + (y+1)^2 :=
by
  sorry

end no_cube_sum_of_three_consecutive_squares_l174_174621


namespace calculate_fraction_l174_174923

theorem calculate_fraction : (1 / (1 + 1 / (4 + 1 / 5))) = (21 / 26) :=
by
  sorry

end calculate_fraction_l174_174923


namespace faster_cow_days_to_eat_one_bag_l174_174597

-- Conditions as assumptions
def num_cows : ℕ := 60
def num_husks : ℕ := 150
def num_days : ℕ := 80
def faster_cows : ℕ := 20
def normal_cows : ℕ := num_cows - faster_cows
def faster_rate : ℝ := 1.3

-- The question translated to Lean 4 statement
theorem faster_cow_days_to_eat_one_bag :
  (faster_cows * faster_rate + normal_cows) / num_cows * (num_husks / num_days) = 1 / 27.08 :=
sorry

end faster_cow_days_to_eat_one_bag_l174_174597


namespace reflect_parabola_y_axis_l174_174619

theorem reflect_parabola_y_axis (x y : ℝ) :
  (y = 2 * (x - 1)^2 - 4) → (y = 2 * (-x - 1)^2 - 4) :=
sorry

end reflect_parabola_y_axis_l174_174619


namespace solve_inverse_function_l174_174245

-- Define the given functions
def f (x : ℝ) : ℝ := x^3 + 2*x^2 + 3*x + 1
def g (x : ℝ) : ℝ := x^4 - x^3 + 4*x^2 + 8*x + 8
def h (x : ℝ) : ℝ := x + 1

-- State the mathematical equivalent proof problem
theorem solve_inverse_function (x : ℝ) :
  f ⁻¹' {g x} = {y | h y = x + 1} ↔
  (x = (3 + Real.sqrt 5) / 2) ∨ (x = (3 - Real.sqrt 5) / 2) :=
sorry -- Proof is omitted

end solve_inverse_function_l174_174245


namespace problem_part1_problem_part2_l174_174181

open Complex

noncomputable def E1 := ((1 + I)^2 / (1 + 2 * I)) + ((1 - I)^2 / (2 - I))

theorem problem_part1 : E1 = (6 / 5) - (2 / 5) * I :=
by
  sorry

theorem problem_part2 (x y : ℝ) (h1 : (x / 2) + (y / 5) = 1) (h2 : (x / 2) + (2 * y / 5) = 3) : x = -2 ∧ y = 10 :=
by
  sorry

end problem_part1_problem_part2_l174_174181


namespace functional_equation_l174_174429

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation (f_add : ∀ x y : ℝ, f (x + y) = f x + f y) (f_two : f 2 = 4) : f 1 = 2 :=
sorry

end functional_equation_l174_174429


namespace emily_pen_selections_is_3150_l174_174357

open Function

noncomputable def emily_pen_selections : ℕ :=
  (Nat.choose 10 4) * (Nat.choose 6 2)

theorem emily_pen_selections_is_3150 : emily_pen_selections = 3150 :=
by
  sorry

end emily_pen_selections_is_3150_l174_174357


namespace triangle_ABC_BC_length_l174_174709

theorem triangle_ABC_BC_length 
  (A B C D : ℝ)
  (AB AD DC AC BD BC : ℝ)
  (h1 : BD = 20)
  (h2 : AC = 69)
  (h3 : AB = 29)
  (h4 : BD^2 + DC^2 = BC^2)
  (h5 : AD^2 + BD^2 = AB^2)
  (h6 : AC = AD + DC) : 
  BC = 52 := 
by
  sorry

end triangle_ABC_BC_length_l174_174709


namespace x_plus_2y_equals_5_l174_174033

theorem x_plus_2y_equals_5 (x y : ℝ) (h1 : 2 * x + y = 6) (h2 : (x + y) / 3 = 1.222222222222222) : x + 2 * y = 5 := 
by sorry

end x_plus_2y_equals_5_l174_174033


namespace problem_statement_l174_174961

theorem problem_statement (x y : ℝ) 
  (hA : A = (x + y) * (y - 3 * x))
  (hB : B = (x - y)^4 / (x - y)^2)
  (hCond : 2 * y + A = B - 6) :
  y = 2 * x^2 - 3 ∧ (y + 3)^2 - 2 * x * (x * y - 3) - 6 * x * (x + 1) = 0 :=
by
  sorry

end problem_statement_l174_174961


namespace find_f_pi_six_value_l174_174805

noncomputable def f (x : ℝ) (f'₀ : ℝ) : ℝ := f'₀ * Real.sin x + Real.cos x

theorem find_f_pi_six_value (f'₀ : ℝ) (h : f'₀ = 2 + Real.sqrt 3) : f (π / 6) f'₀ = 1 + Real.sqrt 3 := 
by
  -- condition from the problem
  let f₀ := f (π / 6) f'₀
  -- final goal to prove
  sorry

end find_f_pi_six_value_l174_174805


namespace exact_time_now_l174_174029

noncomputable def minute_hand_position (t : ℝ) : ℝ := 6 * (t + 4)
noncomputable def hour_hand_position (t : ℝ) : ℝ := 0.5 * (t - 2) + 270
noncomputable def is_opposite (x y : ℝ) : Prop := |x - y| = 180

theorem exact_time_now (t : ℝ) (h1 : 0 ≤ t) (h2 : t < 60)
  (h3 : is_opposite (minute_hand_position t) (hour_hand_position t)) :
  t = 591/50 :=
by
  sorry

end exact_time_now_l174_174029


namespace range_of_f_l174_174096

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 2

theorem range_of_f : ∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 → f x ∈ Set.Icc (-18 : ℝ) (2 : ℝ) :=
sorry

end range_of_f_l174_174096


namespace leaves_count_l174_174907

theorem leaves_count {m n L : ℕ} (h1 : m + n = 10) (h2 : L = 5 * m + 2 * n) :
  ¬(L = 45 ∨ L = 39 ∨ L = 37 ∨ L = 31) :=
by
  sorry

end leaves_count_l174_174907


namespace lcm_of_18_and_36_l174_174890

theorem lcm_of_18_and_36 : Nat.lcm 18 36 = 36 := 
by 
  sorry

end lcm_of_18_and_36_l174_174890


namespace geometric_ratio_l174_174762

noncomputable def S (n : ℕ) : ℝ := sorry  -- Let's assume S is a function that returns the sum of the first n terms of the geometric sequence.

-- Conditions
axiom S_10_eq_S_5 : S 10 = 2 * S 5

-- Definition to be proved
theorem geometric_ratio :
  (S 5 + S 10 + S 15) / (S 10 - S 5) = -9 / 2 :=
sorry

end geometric_ratio_l174_174762


namespace expression_never_prime_l174_174108

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem expression_never_prime (p : ℕ) (hp : is_prime p) : ¬ is_prime (p^2 + 20) := sorry

end expression_never_prime_l174_174108


namespace jacob_fraction_of_phoebe_age_l174_174452

-- Definitions
def Rehana_current_age := 25
def Rehana_future_age (years : Nat) := Rehana_current_age + years
def Phoebe_future_age (years : Nat) := (Rehana_future_age years) / 3
def Phoebe_current_age := Phoebe_future_age 5 - 5
def Jacob_age := 3
def fraction_of_Phoebe_age := Jacob_age / Phoebe_current_age

-- Theorem statement
theorem jacob_fraction_of_phoebe_age :
  fraction_of_Phoebe_age = 3 / 5 :=
  sorry

end jacob_fraction_of_phoebe_age_l174_174452


namespace radii_difference_of_concentric_circles_l174_174225

theorem radii_difference_of_concentric_circles 
  (r : ℝ) 
  (h_area_ratio : (π * (2 * r)^2) / (π * r^2) = 4) : 
  (2 * r) - r = r :=
by
  sorry

end radii_difference_of_concentric_circles_l174_174225


namespace total_volume_of_four_cubes_is_500_l174_174804

-- Definition of the edge length of each cube
def edge_length : ℝ := 5

-- Definition of the volume of one cube
def volume_of_cube (s : ℝ) : ℝ := s^3

-- Definition of the number of cubes
def number_of_cubes : ℕ := 4

-- Definition of the total volume
def total_volume (n : ℕ) (v : ℝ) : ℝ := n * v

-- The proposition we want to prove
theorem total_volume_of_four_cubes_is_500 :
  total_volume number_of_cubes (volume_of_cube edge_length) = 500 :=
by
  sorry

end total_volume_of_four_cubes_is_500_l174_174804


namespace same_sum_sufficient_days_l174_174333

variable {S Wb Wc : ℝ}
variable (h1 : S = 12 * Wb)
variable (h2 : S = 24 * Wc)

theorem same_sum_sufficient_days : ∃ D : ℝ, D = 8 ∧ S = D * (Wb + Wc) :=
by
  use 8
  sorry

end same_sum_sufficient_days_l174_174333


namespace largest_value_l174_174217

def X := (2010 / 2009) + (2010 / 2011)
def Y := (2010 / 2011) + (2012 / 2011)
def Z := (2011 / 2010) + (2011 / 2012)

theorem largest_value : X > Y ∧ X > Z := 
by
  sorry

end largest_value_l174_174217


namespace tan_alpha_trigonometric_expression_l174_174479

variable (α : ℝ)
variable (h1 : Real.sin (Real.pi + α) = 3 / 5)
variable (h2 : Real.pi < α ∧ α < 3 * Real.pi / 2)

theorem tan_alpha (h1 : Real.sin (Real.pi + α) = 3 / 5) (h2 : Real.pi < α ∧ α < 3 * Real.pi / 2) : Real.tan α = 3 / 4 := 
sorry

theorem trigonometric_expression (h1 : Real.sin (Real.pi + α) = 3 / 5) (h2 : Real.pi < α ∧ α < 3 * Real.pi / 2) : 
  (Real.sin ((Real.pi + α) / 2) - Real.cos ((Real.pi + α) / 2)) / 
  (Real.sin ((Real.pi - α) / 2) - Real.cos ((Real.pi - α) / 2)) = -1 / 2 := 
sorry

end tan_alpha_trigonometric_expression_l174_174479


namespace smallest_three_digit_in_pascals_triangle_l174_174203

theorem smallest_three_digit_in_pascals_triangle : ∃ k n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ ∀ m, ((m <= n) ∧ (m >= 100)) → m ≥ n :=
by
  sorry

end smallest_three_digit_in_pascals_triangle_l174_174203


namespace algebraic_notation_3m_minus_n_squared_l174_174862

theorem algebraic_notation_3m_minus_n_squared (m n : ℝ) : 
  (3 * m - n)^2 = (3 * m - n) ^ 2 :=
by sorry

end algebraic_notation_3m_minus_n_squared_l174_174862


namespace circle_center_radius_l174_174527

theorem circle_center_radius :
  ∃ (h k r : ℝ), (∀ x y : ℝ, (x + 1)^2 + (y - 1)^2 = 4 → (x - h)^2 + (y - k)^2 = r^2) ∧
    h = -1 ∧ k = 1 ∧ r = 2 :=
by
  sorry

end circle_center_radius_l174_174527


namespace quadratic_inequality_l174_174686

theorem quadratic_inequality (t x₁ x₂ : ℝ) (α β : ℝ)
  (ht : (2 * x₁^2 - t * x₁ - 2 = 0) ∧ (2 * x₂^2 - t * x₂ - 2 = 0))
  (hx : α ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ β)
  (hαβ : α < β)
  (roots : α + β = t / 2 ∧ α * β = -1) :
  4*x₁*x₂ - t*(x₁ + x₂) - 4 < 0 := 
sorry

end quadratic_inequality_l174_174686


namespace expression_evaluation_l174_174755

theorem expression_evaluation (a b c : ℤ) 
  (h1 : c = a + 8) 
  (h2 : b = a + 4) 
  (h3 : a = 5) 
  (h4 : a + 2 ≠ 0) 
  (h5 : b - 3 ≠ 0) 
  (h6 : c + 7 ≠ 0) : 
  (a + 3) / (a + 2) * (b - 2) / (b - 3) * (c + 10) / (c + 7) = 23/15 :=
by
  sorry

end expression_evaluation_l174_174755


namespace minimum_discount_percentage_l174_174587

theorem minimum_discount_percentage (cost_price marked_price : ℝ) (profit_margin : ℝ) (discount : ℝ) :
  cost_price = 400 ∧ marked_price = 600 ∧ profit_margin = 0.05 ∧ 
  (marked_price * (1 - discount / 100) - cost_price) / cost_price ≥ profit_margin → discount ≤ 30 := 
by
  intros h
  rcases h with ⟨hc, hm, hp, hineq⟩
  sorry

end minimum_discount_percentage_l174_174587


namespace distance_formula_example_l174_174932

variable (x1 y1 x2 y2 : ℝ)

theorem distance_formula_example : dist (3, -1) (-4, 3) = Real.sqrt 65 :=
by
  let x1 := 3
  let y1 := -1
  let x2 := -4
  let y2 := 3
  sorry

end distance_formula_example_l174_174932


namespace coloring_time_saved_percentage_l174_174370

variable (n : ℕ := 10) -- number of pictures
variable (draw_time : ℝ := 2) -- time to draw each picture in hours
variable (total_time : ℝ := 34) -- total time spent on drawing and coloring in hours

/-- 
  Prove the percentage of time saved on coloring each picture compared to drawing 
  given the specified conditions.
-/
theorem coloring_time_saved_percentage (n : ℕ) (draw_time total_time : ℝ) 
  (h1 : draw_time > 0)
  (draw_total_time : draw_time * n = 20)
  (total_picture_time : draw_time * n + coloring_total_time = total_time) :
  (draw_time - (coloring_total_time / n)) / draw_time * 100 = 30 := 
by
  sorry

end coloring_time_saved_percentage_l174_174370


namespace remainder_mod_41_l174_174655

theorem remainder_mod_41 (M : ℤ) (hM1 : M = 1234567891011123940) : M % 41 = 0 :=
by
  sorry

end remainder_mod_41_l174_174655


namespace intersection_A_B_l174_174340

def setA : Set (ℝ × ℝ) := {p | ∃ (x: ℝ), p = (x, x^2)}
def setB : Set (ℝ × ℝ) := {p | ∃ (x: ℝ), p = (x, Real.sqrt x)}

theorem intersection_A_B :
  (setA ∩ setB) = {(0, 0), (1, 1)} := by
  sorry

end intersection_A_B_l174_174340


namespace angle_x_in_triangle_l174_174764

theorem angle_x_in_triangle :
  ∀ (x : ℝ), x + 2 * x + 50 = 180 → x = 130 / 3 :=
by
  intro x h
  sorry

end angle_x_in_triangle_l174_174764


namespace domain_of_f_l174_174918

noncomputable def f (x : ℝ) : ℝ := 1 / Real.log x + Real.sqrt (2 - x)

theorem domain_of_f :
  { x : ℝ | 0 < x ∧ x ≤ 2 ∧ x ≠ 1 } = { x : ℝ | (0 < x ∧ x < 1) ∨ (1 < x ∧ x ≤ 2) } :=
by
  sorry

end domain_of_f_l174_174918


namespace find_value_of_m_l174_174199

theorem find_value_of_m :
  (∃ y : ℝ, y = 20 - (0.5 * -6.7)) →
  (m : ℝ) = 3 * -6.7 + (20 - (0.5 * -6.7)) :=
by {
  sorry
}

end find_value_of_m_l174_174199


namespace sin_cos_bounds_l174_174182

theorem sin_cos_bounds (w x y z : ℝ)
  (hw : -Real.pi / 2 ≤ w ∧ w ≤ Real.pi / 2)
  (hx : -Real.pi / 2 ≤ x ∧ x ≤ Real.pi / 2)
  (hy : -Real.pi / 2 ≤ y ∧ y ≤ Real.pi / 2)
  (hz : -Real.pi / 2 ≤ z ∧ z ≤ Real.pi / 2)
  (h₁ : Real.sin w + Real.sin x + Real.sin y + Real.sin z = 1)
  (h₂ : Real.cos (2 * w) + Real.cos (2 * x) + Real.cos (2 * y) + Real.cos (2 * z) ≥ 10 / 3) :
  0 ≤ w ∧ w ≤ Real.pi / 6 ∧ 0 ≤ x ∧ x ≤ Real.pi / 6 ∧ 0 ≤ y ∧ y ≤ Real.pi / 6 ∧ 0 ≤ z ∧ z ≤ Real.pi / 6 :=
by
  sorry

end sin_cos_bounds_l174_174182


namespace inverse_variation_l174_174373

theorem inverse_variation (a b : ℝ) (k : ℝ) (h1 : a * b = k) (h2 : 800 * 0.5 = k) (h3 : a = 1600) : b = 0.25 :=
by 
  sorry

end inverse_variation_l174_174373


namespace three_divides_two_pow_n_plus_one_l174_174395

theorem three_divides_two_pow_n_plus_one (n : ℕ) (hn : n > 0) : 
  (3 ∣ 2^n + 1) ↔ Odd n := 
sorry

end three_divides_two_pow_n_plus_one_l174_174395


namespace adam_action_figures_per_shelf_l174_174810

-- Define the number of shelves and the total number of action figures
def shelves : ℕ := 4
def total_action_figures : ℕ := 44

-- Define the number of action figures per shelf
def action_figures_per_shelf : ℕ := total_action_figures / shelves

-- State the theorem to be proven
theorem adam_action_figures_per_shelf : action_figures_per_shelf = 11 :=
by sorry

end adam_action_figures_per_shelf_l174_174810


namespace new_time_between_maintenance_checks_l174_174125

-- Definitions based on the conditions
def original_time : ℝ := 25
def percentage_increase : ℝ := 0.20

-- Statement to be proved
theorem new_time_between_maintenance_checks : original_time * (1 + percentage_increase) = 30 := by
  sorry

end new_time_between_maintenance_checks_l174_174125


namespace log_expression_value_l174_174837

theorem log_expression_value
  (h₁ : x + (Real.log 32 / Real.log 8) = 1.6666666666666667)
  (h₂ : Real.log 32 / Real.log 8 = 1.6666666666666667) :
  x = 0 :=
by
  sorry

end log_expression_value_l174_174837


namespace derivative_at_one_l174_174909

noncomputable def f (x : ℝ) : ℝ := x^2 + Real.exp x

theorem derivative_at_one : deriv f 1 = 2 + Real.exp 1 := by
  sorry

end derivative_at_one_l174_174909


namespace problem_correctness_l174_174714

theorem problem_correctness (a b x y m : ℝ) 
  (h1 : a + b = 0) 
  (h2 : x * y = 1) 
  (h3 : |m| = 2) : 
  (m = 2 ∨ m = -2) ∧ (m^2 + (a + b) / 2 + (- (x * y)) ^ 2023 = 3) := 
by
  sorry

end problem_correctness_l174_174714


namespace work_problem_l174_174668

theorem work_problem (W : ℝ) (A B C : ℝ)
  (h1 : B + C = W / 24)
  (h2 : C + A = W / 12)
  (h3 : C = W / 32) : A + B = W / 16 := 
by
  sorry

end work_problem_l174_174668


namespace prime_sq_mod_12_l174_174053

theorem prime_sq_mod_12 (p : ℕ) (hp_prime : Nat.Prime p) (hp_gt_3 : p > 3) : (p * p) % 12 = 1 := by
  sorry

end prime_sq_mod_12_l174_174053


namespace find_k_l174_174625

theorem find_k (a b k : ℝ) (h1 : 2^a = k) (h2 : 3^b = k) (hk : k ≠ 1) (h3 : 2 * a + b = a * b) : 
  k = 18 :=
sorry

end find_k_l174_174625


namespace value_of_polynomial_l174_174458

theorem value_of_polynomial (x y : ℝ) (h : x - y = 5) : (x - y)^2 + 2 * (x - y) - 10 = 25 :=
by sorry

end value_of_polynomial_l174_174458


namespace Kishore_education_expense_l174_174616

theorem Kishore_education_expense
  (rent milk groceries petrol misc saved : ℝ) -- expenses
  (total_saved_salary : ℝ) -- percentage of saved salary
  (saving_amount : ℝ) -- actual saving
  (total_salary total_expense_children_education : ℝ) -- total salary and expense on children's education
  (H1 : rent = 5000)
  (H2 : milk = 1500)
  (H3 : groceries = 4500)
  (H4 : petrol = 2000)
  (H5 : misc = 3940)
  (H6 : total_saved_salary = 0.10)
  (H7 : saving_amount = 2160)
  (H8 : total_salary = saving_amount / total_saved_salary)
  (H9 : total_expense_children_education = total_salary - (rent + milk + groceries + petrol + misc) - saving_amount) :
  total_expense_children_education = 2600 :=
by 
  simp only [H1, H2, H3, H4, H5, H6, H7] at *
  norm_num at *
  sorry

end Kishore_education_expense_l174_174616


namespace remainder_of_4521_l174_174998

theorem remainder_of_4521 (h1 : ∃ d : ℕ, d = 88)
  (h2 : 3815 % 88 = 31) : 4521 % 88 = 33 :=
sorry

end remainder_of_4521_l174_174998


namespace rectangle_area_l174_174246

theorem rectangle_area (ABCD : Type*) (small_square : ℕ) (shaded_squares : ℕ) (side_length : ℕ) 
  (shaded_area : ℕ) (width : ℕ) (height : ℕ)
  (H1 : shaded_squares = 3) 
  (H2 : side_length = 2)
  (H3 : shaded_area = side_length * side_length)
  (H4 : width = 6)
  (H5 : height = 4)
  : (width * height) = 24 :=
by
  sorry

end rectangle_area_l174_174246


namespace naomi_number_of_ways_to_1000_l174_174412

-- Define the initial condition and operations

def start : ℕ := 2

def add1 (n : ℕ) : ℕ := n + 1

def square (n : ℕ) : ℕ := n * n

-- Define a proposition that counts the number of ways to reach 1000 from 2 using these operations
def count_ways (start target : ℕ) : ℕ := sorry  -- We'll need a complex function to literally count the paths, but we'll abstract this here.

-- Theorem stating the number of ways to reach 1000
theorem naomi_number_of_ways_to_1000 : count_ways start 1000 = 128 := 
sorry

end naomi_number_of_ways_to_1000_l174_174412


namespace find_a_l174_174832

-- Definition of the curve y = x^3 + ax + 1
def curve (x a : ℝ) : ℝ := x^3 + a * x + 1

-- Definition of the tangent line y = 2x + 1
def tangent_line (x : ℝ) : ℝ := 2 * x + 1

-- The slope of the tangent line is 2
def slope_of_tangent_line (x : ℝ) (a : ℝ) : ℝ := 3 * x^2 + a

theorem find_a (a : ℝ) : 
  (∃ x₀, curve x₀ a = tangent_line x₀) ∧ (∃ x₀, slope_of_tangent_line x₀ a = 2) → a = 2 :=
by
  sorry

end find_a_l174_174832


namespace parabola_vertex_l174_174327

theorem parabola_vertex:
  ∃ x y: ℝ, y^2 + 8 * y + 2 * x + 1 = 0 ∧ (x, y) = (7.5, -4) := sorry

end parabola_vertex_l174_174327


namespace cost_per_mile_sunshine_is_018_l174_174460

theorem cost_per_mile_sunshine_is_018 :
  ∀ (x : ℝ) (daily_rate_sunshine daily_rate_city cost_per_mile_city : ℝ),
  daily_rate_sunshine = 17.99 →
  daily_rate_city = 18.95 →
  cost_per_mile_city = 0.16 →
  (daily_rate_sunshine + 48 * x = daily_rate_city + cost_per_mile_city * 48) →
  x = 0.18 :=
by
  intros x daily_rate_sunshine daily_rate_city cost_per_mile_city
  intros h1 h2 h3 h4
  sorry

end cost_per_mile_sunshine_is_018_l174_174460


namespace proof_problem_statement_l174_174141

noncomputable def proof_problem (x y: ℝ) : Prop :=
  x ≥ 1 ∧ y ≥ 1 ∧ (∀ n : ℕ, n > 0 → (⌊x / y⌋ : ℝ) = ⌊↑n * x⌋ / ⌊↑n * y⌋) →
  (x = y ∨ (∃ k : ℤ, k ≠ 0 ∧ (x = k * y ∨ y = k * x)))

-- The formal statement of the problem
theorem proof_problem_statement (x y : ℝ) :
  proof_problem x y := by
  sorry

end proof_problem_statement_l174_174141


namespace comparison_of_exponential_and_power_l174_174518

theorem comparison_of_exponential_and_power :
  let a := 2 ^ 0.6
  let b := 0.6 ^ 2
  a > b :=
by
  let a := 2 ^ 0.6
  let b := 0.6 ^ 2
  sorry

end comparison_of_exponential_and_power_l174_174518


namespace james_balloons_l174_174386

-- Definitions
def amy_balloons : ℕ := 513
def extra_balloons_james_has : ℕ := 709

-- Statement of the problem
theorem james_balloons : amy_balloons + extra_balloons_james_has = 1222 :=
by
  -- Placeholder for the actual proof
  sorry

end james_balloons_l174_174386


namespace total_amount_earned_is_90_l174_174102

variable (W : ℕ)

-- Define conditions
def work_capacity_condition : Prop :=
  5 = W ∧ W = 8

-- Define wage per man in Rs.
def wage_per_man : ℕ := 6

-- Define total amount earned by 5 men
def total_earned_by_5_men : ℕ := 5 * wage_per_man

-- Define total amount for the problem
def total_earned (W : ℕ) : ℕ :=
  3 * total_earned_by_5_men

-- The final proof statement
theorem total_amount_earned_is_90 (W : ℕ) (h : work_capacity_condition W) : total_earned W = 90 := by
  sorry

end total_amount_earned_is_90_l174_174102


namespace other_endpoint_of_diameter_l174_174900

theorem other_endpoint_of_diameter (center endpoint : ℝ × ℝ) (hc : center = (1, 2)) (he : endpoint = (4, 6)) :
    ∃ other_endpoint : ℝ × ℝ, other_endpoint = (-2, -2) :=
by
  sorry

end other_endpoint_of_diameter_l174_174900


namespace xiao_zhang_return_distance_xiao_zhang_no_refuel_needed_l174_174490

def total_distance : ℕ :=
  15 - 3 + 16 - 11 + 10 - 12 + 4 - 15 + 16 - 18

def fuel_consumption_per_km : ℝ := 0.6
def initial_fuel : ℝ := 72.2

theorem xiao_zhang_return_distance :
  total_distance = 2 := by
  sorry

theorem xiao_zhang_no_refuel_needed :
  (initial_fuel - fuel_consumption_per_km * (|15| + |3| + |16| + |11| + |10| + |12| + |4| + |15| + |16| + |18|)) >= 0 := by
  sorry

end xiao_zhang_return_distance_xiao_zhang_no_refuel_needed_l174_174490


namespace kernel_red_given_popped_l174_174263

def prob_red_given_popped (P_red : ℚ) (P_green : ℚ) 
                           (P_popped_given_red : ℚ) (P_popped_given_green : ℚ) : ℚ :=
  let P_red_popped := P_red * P_popped_given_red
  let P_green_popped := P_green * P_popped_given_green
  let P_popped := P_red_popped + P_green_popped
  P_red_popped / P_popped

theorem kernel_red_given_popped : prob_red_given_popped (3/4) (1/4) (3/5) (3/4) = 12/17 :=
by
  sorry

end kernel_red_given_popped_l174_174263


namespace range_of_m_l174_174299

noncomputable def A (x : ℝ) : ℝ := x^2 - (3/2) * x + 1

def in_interval (x : ℝ) : Prop := (3/4 ≤ x) ∧ (x ≤ 2)

def B (y : ℝ) (m : ℝ) : Prop := y ≥ 1 - m^2

theorem range_of_m (m : ℝ) :
  (∀ x, in_interval x → B (A x) m) ↔ (m ≤ - (3/4) ∨ m ≥ (3/4)) := 
sorry

end range_of_m_l174_174299


namespace egg_rolls_total_l174_174995

theorem egg_rolls_total (omar_egg_rolls karen_egg_rolls lily_egg_rolls : ℕ) :
  omar_egg_rolls = 219 → karen_egg_rolls = 229 → lily_egg_rolls = 275 → 
  omar_egg_rolls + karen_egg_rolls + lily_egg_rolls = 723 := 
by
  intros h1 h2 h3
  sorry

end egg_rolls_total_l174_174995


namespace problem_condition_relationship_l174_174076

theorem problem_condition_relationship (x : ℝ) :
  (x^2 - x - 2 > 0) → (|x - 1| > 1) := 
sorry

end problem_condition_relationship_l174_174076


namespace abs_tan_45_eq_sqrt3_factor_4x2_36_l174_174869

theorem abs_tan_45_eq_sqrt3 : abs (1 - Real.sqrt 3) + Real.tan (Real.pi / 4) = Real.sqrt 3 := 
by 
  sorry

theorem factor_4x2_36 (x : ℝ) : 4 * x ^ 2 - 36 = 4 * (x + 3) * (x - 3) := 
by 
  sorry

end abs_tan_45_eq_sqrt3_factor_4x2_36_l174_174869


namespace circle_equation_l174_174149

theorem circle_equation :
  ∃ r : ℝ, ∀ x y : ℝ,
  ((x - 2) * (x - 2) + (y - 1) * (y - 1) = r * r) ∧
  ((5 - 2) * (5 - 2) + (-2 - 1) * (-2 - 1) = r * r) ∧
  (5 + 2 * -2 - 5 + r * r = 0) :=
sorry

end circle_equation_l174_174149


namespace bus_ticket_problem_l174_174298

variables (x y : ℕ)

theorem bus_ticket_problem (h1 : x + y = 99) (h2 : 2 * x + 3 * y = 280) : x = 17 ∧ y = 82 :=
by
  sorry

end bus_ticket_problem_l174_174298


namespace tan_subtraction_l174_174640

variable {α β : ℝ}

theorem tan_subtraction (h1 : Real.tan α = 2) (h2 : Real.tan β = -3) : Real.tan (α - β) = -1 := by
  sorry

end tan_subtraction_l174_174640


namespace rest_area_milepost_l174_174973

theorem rest_area_milepost 
  (milepost_fifth_exit : ℕ) 
  (milepost_fifteenth_exit : ℕ) 
  (rest_area_milepost : ℕ)
  (h1 : milepost_fifth_exit = 50)
  (h2 : milepost_fifteenth_exit = 350)
  (h3 : rest_area_milepost = (milepost_fifth_exit + (milepost_fifteenth_exit - milepost_fifth_exit) / 2)) :
  rest_area_milepost = 200 := 
by
  intros
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end rest_area_milepost_l174_174973


namespace remainder_7_pow_93_mod_12_l174_174483

theorem remainder_7_pow_93_mod_12 : 7 ^ 93 % 12 = 7 := 
by
  -- the sequence repeats every two terms: 7, 1, 7, 1, ...
  sorry

end remainder_7_pow_93_mod_12_l174_174483


namespace gasoline_needed_l174_174334

variable (distance_trip : ℕ) (fuel_per_trip_distance : ℕ) (trip_distance : ℕ) (fuel_needed : ℕ)

theorem gasoline_needed (h1 : distance_trip = 140)
                       (h2 : fuel_per_trip_distance = 10)
                       (h3 : trip_distance = 70)
                       (h4 : fuel_needed = 20) :
  (fuel_per_trip_distance * (distance_trip / trip_distance)) = fuel_needed :=
by sorry

end gasoline_needed_l174_174334


namespace hemisphere_surface_area_ratio_l174_174025

theorem hemisphere_surface_area_ratio 
  (r : ℝ) (sphere_surface_area : ℝ) (hemisphere_surface_area : ℝ) 
  (eq1 : sphere_surface_area = 4 * π * r^2) 
  (eq2 : hemisphere_surface_area = 3 * π * r^2) : 
  hemisphere_surface_area / sphere_surface_area = 3 / 4 :=
by sorry

end hemisphere_surface_area_ratio_l174_174025


namespace quadratic_real_equal_roots_l174_174548

theorem quadratic_real_equal_roots (m : ℝ) :
  (3*x^2 + (2 - m)*x + 5 = 0 → (3 : ℕ) * x^2 + ((2 : ℕ) - m) * x + (5 : ℕ) = 0) →
  ∃ m₁ m₂ : ℝ, m₁ = 2 - 2 * Real.sqrt 15 ∧ m₂ = 2 + 2 * Real.sqrt 15 ∧ 
    (∀ x : ℝ, (3 * x^2 + (2 - m₁) * x + 5 = 0) ∧ (3 * x^2 + (2 - m₂) * x + 5 = 0)) :=
sorry

end quadratic_real_equal_roots_l174_174548


namespace pyramid_apex_angle_l174_174089

theorem pyramid_apex_angle (A B C D E O : Type) 
  (square_base : Π (P Q : Type), Prop) 
  (isosceles_triangle : Π (R S T : Type), Prop)
  (AEB_angle : Π (X Y Z : Type), Prop) 
  (angle_AOB : ℝ)
  (angle_AEB : ℝ)
  (square_base_conditions : square_base A B ∧ square_base B C ∧ square_base C D ∧ square_base D A)
  (isosceles_triangle_conditions : isosceles_triangle A E B ∧ isosceles_triangle B E C ∧ isosceles_triangle C E D ∧ isosceles_triangle D E A)
  (center : O)
  (diagonals_intersect_at_right_angle : angle_AOB = 90)
  (measured_angle_at_apex : angle_AEB = 100) :
False :=
sorry

end pyramid_apex_angle_l174_174089


namespace assign_grades_l174_174114

-- Definitions based on the conditions:
def num_students : ℕ := 12
def num_grades : ℕ := 4

-- Statement of the theorem
theorem assign_grades : num_grades ^ num_students = 16777216 := by
  sorry

end assign_grades_l174_174114


namespace number_of_friends_l174_174420

theorem number_of_friends (total_bottle_caps : ℕ) (bottle_caps_per_friend : ℕ) (h1 : total_bottle_caps = 18) (h2 : bottle_caps_per_friend = 3) :
  total_bottle_caps / bottle_caps_per_friend = 6 :=
by
  sorry

end number_of_friends_l174_174420


namespace quadratic_roots_sum_l174_174158

theorem quadratic_roots_sum (x₁ x₂ m : ℝ) 
  (eq1 : x₁^2 - (2 * m - 2) * x₁ + (m^2 - 2 * m) = 0) 
  (eq2 : x₂^2 - (2 * m - 2) * x₂ + (m^2 - 2 * m) = 0)
  (h : x₁ + x₂ = 10) : m = 6 :=
sorry

end quadratic_roots_sum_l174_174158


namespace travel_time_by_raft_l174_174501

variable (U V : ℝ) -- U: speed of the steamboat, V: speed of the river current
variable (S : ℝ) -- S: distance between cities A and B

-- Conditions
variable (h1 : S = 12 * U - 15 * V) -- Distance calculation, city B to city A
variable (h2 : S = 8 * U + 10 * V)  -- Distance calculation, city A to city B
variable (T : ℝ) -- Time taken on a raft

-- Proof problem
theorem travel_time_by_raft : T = 60 :=
by
  sorry


end travel_time_by_raft_l174_174501


namespace at_least_two_of_three_equations_have_solutions_l174_174618

theorem at_least_two_of_three_equations_have_solutions
  (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  ∃ x : ℝ, (x - a) * (x - b) = x - c ∨ (x - b) * (x - c) = x - a ∨ (x - c) * (x - a) = x - b := 
sorry

end at_least_two_of_three_equations_have_solutions_l174_174618


namespace Tanika_total_boxes_sold_l174_174332

theorem Tanika_total_boxes_sold:
  let friday_boxes := 60
  let saturday_boxes := friday_boxes + 0.5 * friday_boxes
  let sunday_boxes := saturday_boxes - 0.3 * saturday_boxes
  friday_boxes + saturday_boxes + sunday_boxes = 213 :=
by
  sorry

end Tanika_total_boxes_sold_l174_174332


namespace solve_for_s_l174_174153

theorem solve_for_s (s : ℝ) (t : ℝ) (h1 : t = 8 * s^2) (h2 : t = 4.8) : s = Real.sqrt 0.6 ∨ s = -Real.sqrt 0.6 := by
  sorry

end solve_for_s_l174_174153


namespace find_min_value_l174_174734

noncomputable def problem (x y : ℝ) : Prop :=
  (3^(-x) * y^4 - 2 * y^2 + 3^x ≤ 0) ∧
  (27^x + y^4 - 3^x - 1 = 0)

theorem find_min_value :
  ∃ x y : ℝ, problem x y ∧ 
  (∀ (x' y' : ℝ), problem x' y' → (x^3 + y^3) ≤ (x'^3 + y'^3)) ∧ (x^3 + y^3 = -1) := 
sorry

end find_min_value_l174_174734


namespace contradiction_proof_l174_174099

theorem contradiction_proof (a b c d : ℝ) (h1 : a + b = 1) (h2 : c + d = 1) (h3 : ac + bd > 1) : ¬ (0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) :=
by 
  sorry

end contradiction_proof_l174_174099


namespace abs_inequality_solution_set_l174_174605

-- Define the main problem as a Lean theorem statement
theorem abs_inequality_solution_set (x : ℝ) : 
  (|x - 5| + |x + 3| ≥ 10 ↔ (x ≤ -4 ∨ x ≥ 6)) :=
by {
  sorry
}

end abs_inequality_solution_set_l174_174605


namespace minimum_value_l174_174399

noncomputable def expr (x y : ℝ) := x^2 + x * y + y^2 - 3 * y

theorem minimum_value :
  ∃ x y : ℝ, expr x y = -3 ∧
  ∀ x' y' : ℝ, expr x' y' ≥ -3 :=
sorry

end minimum_value_l174_174399


namespace tank_capacity_l174_174042

theorem tank_capacity (C : ℕ) 
  (leak_rate : C / 4 = C / 4)               -- Condition: Leak rate is C/4 litres per hour
  (inlet_rate : 6 * 60 = 360)                -- Condition: Inlet rate is 360 litres per hour
  (net_emptying_rate : C / 12 = (360 - C / 4))  -- Condition: Net emptying rate for 12 hours
  : C = 1080 := 
by 
  -- Conditions imply that C = 1080 
  sorry

end tank_capacity_l174_174042


namespace average_price_per_book_l174_174207

theorem average_price_per_book
  (spent1 spent2 spent3 spent4 : ℝ) (books1 books2 books3 books4 : ℕ)
  (h1 : spent1 = 1080) (h2 : spent2 = 840) (h3 : spent3 = 765) (h4 : spent4 = 630)
  (hb1 : books1 = 65) (hb2 : books2 = 55) (hb3 : books3 = 45) (hb4 : books4 = 35) :
  (spent1 + spent2 + spent3 + spent4) / (books1 + books2 + books3 + books4) = 16.575 :=
by {
  sorry
}

end average_price_per_book_l174_174207


namespace factorize_expression_l174_174797

theorem factorize_expression (x : ℝ) : 2 * x ^ 2 - 50 = 2 * (x + 5) * (x - 5) := 
  sorry

end factorize_expression_l174_174797


namespace domain_f_domain_g_intersection_M_N_l174_174070

namespace MathProof

open Set

def M : Set ℝ := { x | -2 < x ∧ x < 4 }
def N : Set ℝ := { x | x < 1 ∨ x ≥ 3 }

theorem domain_f :
  (M = { x : ℝ | -2 < x ∧ x < 4 }) := by
  sorry

theorem domain_g :
  (N = { x : ℝ | x < 1 ∨ x ≥ 3 }) := by
  sorry

theorem intersection_M_N : 
  (M ∩ N = { x : ℝ | (-2 < x ∧ x < 1) ∨ (3 ≤ x ∧ x < 4) }) := by
  sorry

end MathProof

end domain_f_domain_g_intersection_M_N_l174_174070


namespace find_n_l174_174445

theorem find_n : ∃ n : ℕ, n < 200 ∧ ∃ k : ℕ, n^2 + (n + 1)^2 = k^2 ∧ (n = 3 ∨ n = 20 ∨ n = 119) := 
by
  sorry

end find_n_l174_174445


namespace initial_blue_balls_l174_174562

theorem initial_blue_balls (B : ℕ) (h1 : 25 - 5 = 20) (h2 : (B - 5) / 20 = 1 / 5) : B = 9 :=
by
  sorry

end initial_blue_balls_l174_174562


namespace delivery_driver_stops_l174_174095

theorem delivery_driver_stops (total_boxes : ℕ) (boxes_per_stop : ℕ) (stops : ℕ) :
  total_boxes = 27 → boxes_per_stop = 9 → stops = total_boxes / boxes_per_stop → stops = 3 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end delivery_driver_stops_l174_174095


namespace number_of_donuts_correct_l174_174267

noncomputable def number_of_donuts_in_each_box :=
  let x : ℕ := 12
  let total_boxes : ℕ := 4
  let donuts_given_to_mom : ℕ := x
  let donuts_given_to_sister : ℕ := 6
  let donuts_left : ℕ := 30
  x

theorem number_of_donuts_correct :
  ∀ (x : ℕ),
  (total_boxes * x - donuts_given_to_mom - donuts_given_to_sister = donuts_left) → x = 12 :=
by
  sorry

end number_of_donuts_correct_l174_174267


namespace quadratic_real_solutions_l174_174322

theorem quadratic_real_solutions (m : ℝ) :
  (∃ x : ℝ, (m - 3) * x^2 + 4 * x + 1 = 0) ↔ (m ≤ 7 ∧ m ≠ 3) :=
by
  sorry

end quadratic_real_solutions_l174_174322


namespace poplar_more_than_pine_l174_174100

theorem poplar_more_than_pine (pine poplar : ℕ) (h1 : pine = 180) (h2 : poplar = 4 * pine) : poplar - pine = 540 :=
by
  -- Proof will be filled here
  sorry

end poplar_more_than_pine_l174_174100


namespace min_value_a_l174_174666

theorem min_value_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - x - 6 > 0 → x > a) ∧
  ¬ (∀ x : ℝ, x > a → x^2 - x - 6 > 0) ↔ a = 3 :=
sorry

end min_value_a_l174_174666


namespace evaluate_expression_l174_174019

theorem evaluate_expression (x : ℤ) (h : x = 4) : 3 * x + 5 = 17 :=
by
  sorry

end evaluate_expression_l174_174019


namespace abcd_inequality_l174_174756

theorem abcd_inequality (a b c d : ℝ) :
  (a * c + b * d)^2 ≤ (a^2 + b^2) * (c^2 + d^2) :=
sorry

end abcd_inequality_l174_174756


namespace find_m_n_l174_174798

theorem find_m_n (m n : ℕ) (h1 : m ≥ 0) (h2 : n ≥ 0) (h3 : 3^m - 7^n = 2) : m = 2 ∧ n = 1 := 
sorry

end find_m_n_l174_174798


namespace box_internal_volume_in_cubic_feet_l174_174140

def box_length := 26 -- inches
def box_width := 26 -- inches
def box_height := 14 -- inches
def wall_thickness := 1 -- inch

def external_volume := box_length * box_width * box_height -- cubic inches
def internal_length := box_length - 2 * wall_thickness
def internal_width := box_width - 2 * wall_thickness
def internal_height := box_height - 2 * wall_thickness
def internal_volume := internal_length * internal_width * internal_height -- cubic inches

def cubic_inches_to_cubic_feet (v : ℕ) : ℕ := v / 1728

theorem box_internal_volume_in_cubic_feet : cubic_inches_to_cubic_feet internal_volume = 4 := by
  sorry

end box_internal_volume_in_cubic_feet_l174_174140


namespace cos_75_eq_sqrt6_sub_sqrt2_div_4_l174_174356

theorem cos_75_eq_sqrt6_sub_sqrt2_div_4 :
  Real.cos (75 * Real.pi / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := sorry

end cos_75_eq_sqrt6_sub_sqrt2_div_4_l174_174356


namespace books_still_to_read_l174_174812

-- Define the given conditions
def total_books : ℕ := 22
def books_read : ℕ := 12

-- State the theorem to be proven
theorem books_still_to_read : total_books - books_read = 10 := 
by
  -- skipping the proof
  sorry

end books_still_to_read_l174_174812


namespace value_a_plus_c_l174_174895

noncomputable def f (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c
noncomputable def g (a b c : ℝ) (x : ℝ) := c * x^2 + b * x + a

theorem value_a_plus_c (a b c : ℝ) (h : ∀ x : ℝ, f a b c (g a b c x) = x) : a + c = -1 :=
by
  sorry

end value_a_plus_c_l174_174895


namespace monotonic_intervals_max_min_values_on_interval_l174_174057

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.exp x

theorem monotonic_intervals :
  (∀ x > -2, 0 < (x + 2) * Real.exp x) ∧ (∀ x < -2, (x + 2) * Real.exp x < 0) :=
by
  sorry

theorem max_min_values_on_interval :
  let a := -4
  let b := 0
  let f_a := (-4 + 1) * Real.exp (-4)
  let f_b := (0 + 1) * Real.exp 0
  let f_c := (-2 + 1) * Real.exp (-2)
  (f b = 1) ∧ (f_c = -1 / Real.exp 2) ∧ (f_a < f_b) ∧ (f_a < f_c) ∧ (f_c < f_b) :=
by
  sorry

end monotonic_intervals_max_min_values_on_interval_l174_174057


namespace surface_area_of_sphere_l174_174407

theorem surface_area_of_sphere (V : ℝ) (hV : V = 72 * π) : 
  ∃ A : ℝ, A = 36 * π * (2^(2/3)) := by 
  sorry

end surface_area_of_sphere_l174_174407


namespace convert_to_scientific_notation_l174_174545

theorem convert_to_scientific_notation :
  (26.62 * 10^9) = 2.662 * 10^9 :=
by
  sorry

end convert_to_scientific_notation_l174_174545


namespace correct_calculation_l174_174840

-- Define the base type for exponents
variables (a : ℝ)

theorem correct_calculation :
  (a^3 * a^5 = a^8) ∧ 
  ¬((a^3)^2 = a^5) ∧ 
  ¬(a^5 + a^2 = a^7) ∧ 
  ¬(a^6 / a^2 = a^3) :=
by
  sorry

end correct_calculation_l174_174840


namespace condition_on_a_l174_174844

theorem condition_on_a (a : ℝ) : 
  (∀ x : ℝ, (5 * x - 3 < 3 * x + 5) → (x < a)) ↔ (a ≥ 4) :=
by
  sorry

end condition_on_a_l174_174844


namespace red_balls_removal_l174_174630

theorem red_balls_removal (total_balls : ℕ) (red_balls : ℕ) (blue_balls : ℕ) (x : ℕ) :
  total_balls = 600 →
  red_balls = 420 →
  blue_balls = 180 →
  (red_balls - x) / (total_balls - x : ℚ) = 3 / 5 ↔ x = 150 :=
by 
  intros;
  sorry

end red_balls_removal_l174_174630


namespace range_of_m_l174_174581

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x^2 + m * x + 1 ≥ 0) ↔ (-2 ≤ m ∧ m ≤ 2) :=
by
  sorry

end range_of_m_l174_174581


namespace integer_solutions_count_l174_174842

theorem integer_solutions_count : ∃ (s : Finset ℤ), (∀ x ∈ s, x^2 - x - 2 ≤ 0) ∧ (Finset.card s = 4) :=
by
  sorry

end integer_solutions_count_l174_174842


namespace candy_division_l174_174771

theorem candy_division (pieces_of_candy : Nat) (students : Nat) 
  (h1 : pieces_of_candy = 344) (h2 : students = 43) : pieces_of_candy / students = 8 := by
  sorry

end candy_division_l174_174771


namespace sqrt_sixteen_is_four_l174_174227

theorem sqrt_sixteen_is_four : Real.sqrt 16 = 4 := 
by 
  sorry

end sqrt_sixteen_is_four_l174_174227


namespace prove_length_square_qp_l174_174747

noncomputable def length_square_qp (r1 r2 d : ℝ) (x : ℝ) : Prop :=
  r1 = 10 ∧ r2 = 8 ∧ d = 15 ∧ (2*r1*x - (x^2 + r2^2 - d^2) = 0) → x^2 = 164

theorem prove_length_square_qp : length_square_qp 10 8 15 x :=
sorry

end prove_length_square_qp_l174_174747


namespace sequence_proofs_l174_174818

theorem sequence_proofs (a b : ℕ → ℝ) :
  a 1 = 1 ∧ b 1 = 0 ∧ 
  (∀ n, 4 * a (n + 1) = 3 * a n - b n + 4) ∧ 
  (∀ n, 4 * b (n + 1) = 3 * b n - a n - 4) → 
  (∀ n, a n + b n = (1 / 2) ^ (n - 1)) ∧ 
  (∀ n, a n - b n = 2 * n - 1) ∧ 
  (∀ n, a n = (1 / 2) ^ n + n - 1 / 2 ∧ b n = (1 / 2) ^ n - n + 1 / 2) :=
sorry

end sequence_proofs_l174_174818


namespace quadratic_polynomial_inequality_l174_174254

variable {a b c : ℝ}

theorem quadratic_polynomial_inequality (h1 : ∀ x : ℝ, a * x^2 + b * x + c < 0)
    (h2 : a < 0)
    (h3 : b^2 - 4 * a * c < 0) :
    b / a < c / a + 1 := 
by 
  sorry

end quadratic_polynomial_inequality_l174_174254


namespace profitable_year_exists_option2_more_economical_l174_174135

noncomputable def total_expenses (x : ℕ) : ℝ := 2 * (x:ℝ)^2 + 10 * x  

noncomputable def annual_income (x : ℕ) : ℝ := 50 * x  

def year_profitable (x : ℕ) : Prop := annual_income x > total_expenses x + 98 / 1000

theorem profitable_year_exists : ∃ x : ℕ, year_profitable x ∧ x = 3 := sorry

noncomputable def total_profit (x : ℕ) : ℝ := 
  50 * x - 2 * (x:ℝ)^2 + 10 * x - 98 / 1000 + if x = 10 then 8 else if x = 7 then 26 else 0

theorem option2_more_economical : 
  total_profit 10 = 110 ∧ total_profit 7 = 110 ∧ 7 < 10 :=
sorry

end profitable_year_exists_option2_more_economical_l174_174135


namespace remainder_of_polynomial_l174_174015

-- Define the polynomial
def P (x : ℝ) : ℝ := x^4 - 4 * x^2 + 7 * x - 8

-- State the theorem
theorem remainder_of_polynomial (x : ℝ) : P 3 = 50 := sorry

end remainder_of_polynomial_l174_174015


namespace correct_calculation_l174_174540

-- Definitions for each condition
def conditionA (a b : ℝ) : Prop := (a - b) * (-a - b) = a^2 - b^2
def conditionB (a : ℝ) : Prop := 2 * a^3 + 3 * a^3 = 5 * a^6
def conditionC (x y : ℝ) : Prop := 6 * x^3 * y^2 / (3 * x) = 2 * x^2 * y^2
def conditionD (x : ℝ) : Prop := (-2 * x^2)^3 = -6 * x^6

-- The proof problem
theorem correct_calculation (a b x y : ℝ) :
  ¬ conditionA a b ∧ ¬ conditionB a ∧ conditionC x y ∧ ¬ conditionD x := 
sorry

end correct_calculation_l174_174540


namespace solve_for_x_l174_174423

theorem solve_for_x (y : ℝ) (x : ℝ) (h1 : y = 432) (h2 : 12^2 * x^4 / 432 = y) : x = 6 := by
  sorry

end solve_for_x_l174_174423


namespace least_positive_integer_l174_174868

theorem least_positive_integer (a : ℕ) :
  (a % 2 = 1) ∧ (a % 3 = 2) ∧ (a % 4 = 3) ∧ (a % 5 = 4) → a = 59 :=
by
  sorry

end least_positive_integer_l174_174868


namespace min_value_of_expression_l174_174121

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1) : x^2 + (1 / 4) * y^2 ≥ 1 / 8 :=
sorry

end min_value_of_expression_l174_174121


namespace integer_roots_of_quadratic_l174_174427

theorem integer_roots_of_quadratic (a : ℚ) :
  (∃ x₁ x₂ : ℤ, 
    a * x₁ * x₁ + (a + 1) * x₁ + (a - 1) = 0 ∧ 
    a * x₂ * x₂ + (a + 1) * x₂ + (a - 1) = 0 ∧ 
    x₁ ≠ x₂) ↔ 
      a = 0 ∨ a = -1/7 ∨ a = 1 :=
by
  sorry

end integer_roots_of_quadratic_l174_174427


namespace GCD_is_six_l174_174176

-- Define the numbers
def a : ℕ := 36
def b : ℕ := 60
def c : ℕ := 90

-- Define the GCD using Lean's gcd function
def GCD_abc : ℕ := Nat.gcd (Nat.gcd a b) c

-- State the theorem that GCD of 36, 60, and 90 is 6
theorem GCD_is_six : GCD_abc = 6 := by
  sorry -- Proof skipped

end GCD_is_six_l174_174176


namespace imaginary_part_of_l174_174335

theorem imaginary_part_of (i : ℂ) (h : i.im = 1) : (1 + i) ^ 5 = -14 - 4 * i := by
  sorry

end imaginary_part_of_l174_174335


namespace overlapping_area_of_rectangular_strips_l174_174695

theorem overlapping_area_of_rectangular_strips (theta : ℝ) (h_theta : theta ≠ 0) :
  let width := 2
  let diag_1 := width
  let diag_2 := width / Real.sin theta
  let area := (diag_1 * diag_2) / 2
  area = 2 / Real.sin theta :=
by
  let width := 2
  let diag_1 := width
  let diag_2 := width / Real.sin theta
  let area := (diag_1 * diag_2) / 2
  sorry

end overlapping_area_of_rectangular_strips_l174_174695


namespace find_x_and_C_l174_174037

def A (x : ℝ) : Set ℝ := {1, 3, x^2}
def B (x : ℝ) : Set ℝ := {1, 2 - x}

theorem find_x_and_C (x : ℝ) (C : Set ℝ) :
  B x ⊆ A x → B (-2) ∪ C = A (-2) → x = -2 ∧ C = {3} :=
by
  sorry

end find_x_and_C_l174_174037


namespace total_books_l174_174466

theorem total_books (shelves_mystery shelves_picture : ℕ) (books_per_shelf : ℕ) 
    (h_mystery : shelves_mystery = 5) (h_picture : shelves_picture = 4) (h_books_per_shelf : books_per_shelf = 6) : 
    shelves_mystery * books_per_shelf + shelves_picture * books_per_shelf = 54 := 
by 
  sorry

end total_books_l174_174466


namespace fill_tank_with_reduced_bucket_capacity_l174_174828

theorem fill_tank_with_reduced_bucket_capacity (C : ℝ) :
    let original_buckets := 200
    let original_capacity := C
    let new_capacity := (4 / 5) * original_capacity
    let new_buckets := 250
    (original_buckets * original_capacity) = ((new_buckets) * new_capacity) :=
by
    sorry

end fill_tank_with_reduced_bucket_capacity_l174_174828


namespace division_remainder_l174_174857

-- Define the conditions
def dividend : ℝ := 9087.42
def divisor : ℝ := 417.35
def quotient : ℝ := 21

-- Define the expected remainder
def expected_remainder : ℝ := 323.07

-- Statement of the problem
theorem division_remainder : dividend - divisor * quotient = expected_remainder :=
by
  sorry

end division_remainder_l174_174857


namespace remainder_of_n_mod_1000_l174_174068

-- Definition of the set S
def S : Set ℕ := { n | 1 ≤ n ∧ n ≤ 15 }

-- Define the number of sets of three non-empty disjoint subsets of S
def num_sets_of_three_non_empty_disjoint_subsets (S : Set ℕ) : ℕ :=
  let total_partitions := 4^15
  let single_empty_partition := 3 * 3^15
  let double_empty_partition := 3 * 2^15
  let all_empty_partition := 1
  total_partitions - single_empty_partition + double_empty_partition - all_empty_partition

-- Compute the result of the number modulo 1000
def result := (num_sets_of_three_non_empty_disjoint_subsets S) % 1000

-- Theorem that states the remainder when n is divided by 1000
theorem remainder_of_n_mod_1000 : result = 406 := by
  sorry

end remainder_of_n_mod_1000_l174_174068


namespace Dhoni_spending_difference_l174_174391

-- Definitions
def RentPercent := 20
def LeftOverPercent := 61
def TotalSpendPercent := 100 - LeftOverPercent
def DishwasherPercent := TotalSpendPercent - RentPercent

-- Theorem statement
theorem Dhoni_spending_difference :
  DishwasherPercent = RentPercent - 1 := 
by
  sorry

end Dhoni_spending_difference_l174_174391


namespace min_value_of_polynomial_l174_174770

theorem min_value_of_polynomial : ∃ x : ℝ, (x^2 + x + 1) = 3 / 4 :=
by {
  -- Solution steps are omitted
  sorry
}

end min_value_of_polynomial_l174_174770


namespace total_trucks_l174_174505

-- Define the number of trucks Namjoon has
def trucks_namjoon : ℕ := 3

-- Define the number of trucks Taehyung has
def trucks_taehyung : ℕ := 2

-- Prove that together, Namjoon and Taehyung have 5 trucks
theorem total_trucks : trucks_namjoon + trucks_taehyung = 5 := by 
  sorry

end total_trucks_l174_174505


namespace relation_between_incircle_radius_perimeter_area_l174_174421

theorem relation_between_incircle_radius_perimeter_area (r p S : ℝ) (h : S = (1 / 2) * r * p) : S = (1 / 2) * r * p :=
by {
  sorry
}

end relation_between_incircle_radius_perimeter_area_l174_174421


namespace isosceles_triangle_perimeter_l174_174465

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 6 ∨ a = 9) (h2 : b = 6 ∨ b = 9) (h : a ≠ b) : (a * 2 + b = 21 ∨ a * 2 + b = 24) :=
by
  sorry

end isosceles_triangle_perimeter_l174_174465


namespace speed_of_man_in_still_water_correct_l174_174710

def upstream_speed : ℝ := 25 -- Upstream speed in kmph
def downstream_speed : ℝ := 39 -- Downstream speed in kmph
def speed_in_still_water : ℝ := 32 -- The speed of the man in still water

theorem speed_of_man_in_still_water_correct :
  (upstream_speed + downstream_speed) / 2 = speed_in_still_water :=
by
  sorry

end speed_of_man_in_still_water_correct_l174_174710


namespace granger_cisco_combined_spots_l174_174235

theorem granger_cisco_combined_spots :
  let R := 46
  let C := (R / 2) - 5
  let G := 5 * C
  G + C = 108 := by 
  let R := 46
  let C := (R / 2) - 5
  let G := 5 * C
  sorry

end granger_cisco_combined_spots_l174_174235


namespace min_points_to_guarantee_win_l174_174280

theorem min_points_to_guarantee_win (P Q R S: ℕ) (bonus: ℕ) :
    (P = 6 ∨ P = 4 ∨ P = 2) ∧ (Q = 6 ∨ Q = 4 ∨ Q = 2) ∧ 
    (R = 6 ∨ R = 4 ∨ R = 2) ∧ (S = 6 ∨ S = 4 ∨ S = 2) →
    (bonus = 3 ↔ ((P = 6 ∧ Q = 4 ∧ R = 2) ∨ (P = 6 ∧ Q = 2 ∧ R = 4) ∨ 
                   (P = 4 ∧ Q = 6 ∧ R = 2) ∨ (P = 4 ∧ Q = 2 ∧ R = 6) ∨ 
                   (P = 2 ∧ Q = 6 ∧ R = 4) ∨ (P = 2 ∧ Q = 4 ∧ R = 6))) →
    (P + Q + R + S + bonus ≥ 24) :=
by sorry

end min_points_to_guarantee_win_l174_174280


namespace max_possible_number_under_operations_l174_174651

theorem max_possible_number_under_operations :
  ∀ x : ℕ, x < 17 →
    ∀ n : ℕ, (∃ k : ℕ, k < n ∧ (x + 17 * k) % 19 = 0) →
    ∃ m : ℕ, m = (304 : ℕ) :=
sorry

end max_possible_number_under_operations_l174_174651


namespace total_games_l174_174134

variable (Ken_games Dave_games Jerry_games : ℕ)

-- The conditions from the problem.
def condition1 : Prop := Ken_games = Dave_games + 5
def condition2 : Prop := Dave_games = Jerry_games + 3
def condition3 : Prop := Jerry_games = 7

-- The final statement to prove
theorem total_games (h1 : condition1 Ken_games Dave_games) 
                    (h2 : condition2 Dave_games Jerry_games) 
                    (h3 : condition3 Jerry_games) : 
  Ken_games + Dave_games + Jerry_games = 32 :=
by
  sorry

end total_games_l174_174134


namespace quadratic_root_l174_174661

theorem quadratic_root (a b c : ℝ) (h : 9 * a - 3 * b + c = 0) : 
  a * (-3)^2 + b * (-3) + c = 0 :=
by
  sorry

end quadratic_root_l174_174661


namespace relationship_among_log_exp_powers_l174_174717

theorem relationship_among_log_exp_powers :
  let a := Real.log 0.3 / Real.log 2
  let b := Real.exp (0.3 * Real.log 2)
  let c := Real.exp (0.2 * Real.log 0.3)
  a < c ∧ c < b :=
by
  sorry

end relationship_among_log_exp_powers_l174_174717


namespace amount_in_paise_l174_174083

theorem amount_in_paise (a : ℝ) (h_a : a = 170) (percentage_value : ℝ) (h_percentage : percentage_value = 0.5 / 100) : 
  (percentage_value * a * 100) = 85 := 
by
  sorry

end amount_in_paise_l174_174083


namespace fraction_at_x_eq_4571_div_39_l174_174575

def numerator (x : ℕ) : ℕ := x^6 - 16 * x^3 + x^2 + 64
def denominator (x : ℕ) : ℕ := x^3 - 8

theorem fraction_at_x_eq_4571_div_39 : numerator 5 / denominator 5 = 4571 / 39 :=
by
  sorry

end fraction_at_x_eq_4571_div_39_l174_174575


namespace inclination_angle_of_line_l174_174222

theorem inclination_angle_of_line 
  (l : ℝ) (h : l = Real.tan (-π / 6)) : 
  ∀ θ, θ = Real.pi / 2 :=
by
  -- Placeholder proof
  sorry

end inclination_angle_of_line_l174_174222


namespace f_neg1_l174_174310

noncomputable def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f (-x) = f x
axiom symmetry_about_x2 : ∀ x : ℝ, f (2 + x) = f (2 - x)
axiom f3_value : f 3 = 3

theorem f_neg1 : f (-1) = 3 := by
  sorry

end f_neg1_l174_174310


namespace find_integers_for_perfect_square_l174_174905

def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, n = m * m

theorem find_integers_for_perfect_square :
  {x : ℤ | is_perfect_square (x^4 + x^3 + x^2 + x + 1)} = {-1, 0, 3} :=
by
  sorry

end find_integers_for_perfect_square_l174_174905


namespace smallest_three_digit_plus_one_multiple_l174_174046

theorem smallest_three_digit_plus_one_multiple (x : ℕ) : 
  (421 = x) →
  (x ≥ 100 ∧ x < 1000) ∧ 
  ∃ k : ℕ, x = k * Nat.lcm (Nat.lcm 3 4) * Nat.lcm 5 7 + 1 :=
by
  sorry

end smallest_three_digit_plus_one_multiple_l174_174046


namespace simplify_polynomial_l174_174790

theorem simplify_polynomial (x : ℝ) :
  (2 * x^6 + x^5 + 3 * x^4 + x^3 + 5) - (x^6 + 2 * x^5 + x^4 - x^3 + 7) = 
  x^6 - x^5 + 2 * x^4 + 2 * x^3 - 2 :=
by
  sorry

end simplify_polynomial_l174_174790


namespace distance_BC_400m_l174_174480

-- Define the hypotheses
variables
  (starting_from_same_time : Prop) -- Sam and Nik start from points A and B respectively at the same time
  (constant_speeds : Prop) -- They travel towards each other at constant speeds along the same route
  (meeting_point_C : Prop) -- They meet at point C, which is 600 m away from starting point A
  (speed_Sam : ℕ) (speed_Sam_value : speed_Sam = 50) -- The speed of Sam is 50 meters per minute
  (time_Sam : ℕ) (time_Sam_value : time_Sam = 20) -- It took Sam 20 minutes to cover the distance between A and B

-- Define the statement to be proven
theorem distance_BC_400m
  (d_AB : ℕ) (d_AB_value : d_AB = speed_Sam * time_Sam)
  (d_AC : ℕ) (d_AC_value : d_AC = 600)
  (d_BC : ℕ) (d_BC_value : d_BC = d_AB - d_AC) :
  d_BC = 400 := by
  sorry

end distance_BC_400m_l174_174480


namespace mary_peter_lucy_chestnuts_l174_174693

noncomputable def mary_picked : ℕ := 12
noncomputable def peter_picked : ℕ := mary_picked / 2
noncomputable def lucy_picked : ℕ := peter_picked + 2
noncomputable def total_picked : ℕ := mary_picked + peter_picked + lucy_picked

theorem mary_peter_lucy_chestnuts : total_picked = 26 := by
  sorry

end mary_peter_lucy_chestnuts_l174_174693


namespace no_real_solutions_iff_k_gt_4_l174_174350

theorem no_real_solutions_iff_k_gt_4 (k : ℝ) :
  (∀ x : ℝ, x^2 - 4 * x + k ≠ 0) ↔ k > 4 :=
sorry

end no_real_solutions_iff_k_gt_4_l174_174350


namespace a_1994_is_7_l174_174092

def f (m : ℕ) : ℕ := m % 10

def a (n : ℕ) : ℕ := f (2^(n + 1) - 1)

theorem a_1994_is_7 : a 1994 = 7 :=
by
  sorry

end a_1994_is_7_l174_174092


namespace min_value_fraction_l174_174892

theorem min_value_fraction (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_ln : Real.log (a + b) = 0) :
  (2 / a + 3 / b) = 5 + 2 * Real.sqrt 6 :=
by
  sorry

end min_value_fraction_l174_174892


namespace find_number_l174_174377

-- Definitions based on conditions
def condition (x : ℝ) : Prop := (x - 5) / 3 = 4

-- The target theorem to prove
theorem find_number (x : ℝ) (h : condition x) : x = 17 :=
sorry

end find_number_l174_174377


namespace counting_numbers_dividing_56_greater_than_2_l174_174044

theorem counting_numbers_dividing_56_greater_than_2 :
  (∃ (A : Finset ℕ), A = {n ∈ (Finset.range 57) | n > 2 ∧ 56 % n = 0} ∧ A.card = 5) :=
sorry

end counting_numbers_dividing_56_greater_than_2_l174_174044


namespace cost_price_of_book_l174_174758

theorem cost_price_of_book
  (C : ℝ)
  (h : 1.09 * C - 0.91 * C = 9) :
  C = 50 :=
sorry

end cost_price_of_book_l174_174758


namespace value_of_f_neg_4_l174_174371

noncomputable def f : ℝ → ℝ := λ x => if x ≥ 0 then Real.sqrt x else - (Real.sqrt (-x))

-- Definition that f is an odd function
def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

theorem value_of_f_neg_4 :
  isOddFunction f ∧ (∀ x, x ≥ 0 → f x = Real.sqrt x) → f (-4) = -2 := 
by
  sorry

end value_of_f_neg_4_l174_174371


namespace gcd_1734_816_1343_l174_174761

theorem gcd_1734_816_1343 : Int.gcd (Int.gcd 1734 816) 1343 = 17 :=
by
  sorry

end gcd_1734_816_1343_l174_174761


namespace GIMPS_meaning_l174_174475

/--
  Curtis Cooper's team discovered the largest prime number known as \( 2^{74,207,281} - 1 \), which is a Mersenne prime.
  GIMPS stands for "Great Internet Mersenne Prime Search."

  Prove that GIMPS means "Great Internet Mersenne Prime Search".
-/
theorem GIMPS_meaning : GIMPS = "Great Internet Mersenne Prime Search" :=
  sorry

end GIMPS_meaning_l174_174475


namespace children_got_off_l174_174277

theorem children_got_off {x : ℕ} 
  (initial_children : ℕ := 22)
  (children_got_on : ℕ := 40)
  (children_left : ℕ := 2)
  (equation : initial_children + children_got_on - x = children_left) :
  x = 60 :=
sorry

end children_got_off_l174_174277


namespace ratio_snakes_to_lions_is_S_per_100_l174_174456

variables {S G : ℕ}

/-- Giraffe count in Safari National Park is 10 fewer than snakes -/
def safari_giraffes_minus_ten (S G : ℕ) : Prop := G = S - 10

/-- The number of lions in Safari National Park -/
def safari_lions : ℕ := 100

/-- The ratio of number of snakes to number of lions in Safari National Park -/
def ratio_snakes_to_lions (S : ℕ) : ℕ := S / safari_lions

/-- Prove the ratio of the number of snakes to the number of lions in Safari National Park -/
theorem ratio_snakes_to_lions_is_S_per_100 :
  ∀ S G, safari_giraffes_minus_ten S G → (ratio_snakes_to_lions S = S / 100) :=
by
  intros S G h
  sorry

end ratio_snakes_to_lions_is_S_per_100_l174_174456


namespace yellow_sweets_l174_174524

-- Definitions
def green_sweets : Nat := 212
def blue_sweets : Nat := 310
def sweets_per_person : Nat := 256
def people : Nat := 4

-- Proof problem statement
theorem yellow_sweets : green_sweets + blue_sweets + x = sweets_per_person * people → x = 502 := by
  sorry

end yellow_sweets_l174_174524


namespace detectives_sons_ages_l174_174698

theorem detectives_sons_ages (x y : ℕ) (h1 : x < 5) (h2 : y < 5) (h3 : x * y = 4) (h4 : (∃ x₁ y₁ : ℕ, (x₁ * y₁ = 4 ∧ x₁ < 5 ∧ y₁ < 5) ∧ x₁ ≠ x ∨ y₁ ≠ y)) :
  (x = 1 ∨ x = 4) ∧ (y = 1 ∨ y = 4) :=
by
  sorry

end detectives_sons_ages_l174_174698


namespace prove_smallest_solution_l174_174369

noncomputable def smallest_solution : ℝ :=
  if h : 0 ≤ (3 - Real.sqrt 17) / 2 then min ((3 - Real.sqrt 17) / 2) 1
  else (3 - Real.sqrt 17) / 2  -- Assumption as sqrt(17) > 3, so (3 - sqrt(17))/2 < 0

theorem prove_smallest_solution :
  ∃ x : ℝ, (x * |x| = 3 * x - 2) ∧ 
           (∀ y : ℝ, (y * |y| = 3 * y - 2) → x ≤ y) ∧
           x = (3 - Real.sqrt 17) / 2 :=
sorry

end prove_smallest_solution_l174_174369


namespace shortest_wire_length_l174_174347

theorem shortest_wire_length
  (d1 d2 : ℝ) (h_d1 : d1 = 10) (h_d2 : d2 = 30) :
  let r1 := d1 / 2
  let r2 := d2 / 2
  let straight_sections := 2 * (r2 - r1)
  let curved_sections := 2 * Real.pi * r1 + 2 * Real.pi * r2
  let total_wire_length := straight_sections + curved_sections
  total_wire_length = 20 + 40 * Real.pi :=
by
  sorry

end shortest_wire_length_l174_174347


namespace framed_painting_ratio_l174_174145

def painting_width := 20
def painting_height := 30

def smaller_dimension := painting_width + 2 * 5
def larger_dimension := painting_height + 4 * 5

noncomputable def ratio := (smaller_dimension : ℚ) / (larger_dimension : ℚ)

theorem framed_painting_ratio :
  ratio = 3 / 5 :=
by
  sorry

end framed_painting_ratio_l174_174145


namespace neither_chemistry_nor_biology_l174_174665

variable (club_size chemistry_students biology_students both_students neither_students : ℕ)

def students_in_club : Prop :=
  club_size = 75

def students_taking_chemistry : Prop :=
  chemistry_students = 40

def students_taking_biology : Prop :=
  biology_students = 35

def students_taking_both : Prop :=
  both_students = 25

theorem neither_chemistry_nor_biology :
  students_in_club club_size ∧ 
  students_taking_chemistry chemistry_students ∧
  students_taking_biology biology_students ∧
  students_taking_both both_students →
  neither_students = 75 - ((chemistry_students - both_students) + (biology_students - both_students) + both_students) :=
by
  intros
  sorry

end neither_chemistry_nor_biology_l174_174665


namespace math_problem_proof_l174_174243

-- Define the system of equations
structure equations :=
  (x y m : ℝ)
  (eq1 : x + 2*y - 6 = 0)
  (eq2 : x - 2*y + m*x + 5 = 0)

-- Define the problem conditions and prove the required solutions in Lean 4
theorem math_problem_proof :
  -- Part 1: Positive integer solutions for x + 2y - 6 = 0
  (∀ x y : ℕ, x + 2*y = 6 → (x, y) = (2, 2) ∨ (x, y) = (4, 1)) ∧
  -- Part 2: Given x + y = 0, find m
  (∀ x y : ℝ, x + y = 0 → x + 2*y - 6 = 0 → x - 2*y - (13/6)*x + 5 = 0) ∧
  -- Part 3: Fixed solution for x - 2y + mx + 5 = 0
  (∀ m : ℝ, 0 - 2*2.5 + m*0 + 5 = 0) :=
sorry

end math_problem_proof_l174_174243


namespace triangle_angle_properties_l174_174730

theorem triangle_angle_properties
  (a b : ℕ)
  (h₁ : a = 45)
  (h₂ : b = 70) :
  ∃ (c : ℕ), a + b + c = 180 ∧ c = 65 ∧ max (max a b) c = 70 := by
  sorry

end triangle_angle_properties_l174_174730


namespace length_first_train_l174_174521

noncomputable def length_second_train : ℝ := 200
noncomputable def speed_first_train_kmh : ℝ := 42
noncomputable def speed_second_train_kmh : ℝ := 30
noncomputable def time_seconds : ℝ := 14.998800095992321

noncomputable def speed_first_train_ms : ℝ := speed_first_train_kmh * 1000 / 3600
noncomputable def speed_second_train_ms : ℝ := speed_second_train_kmh * 1000 / 3600

noncomputable def relative_speed : ℝ := speed_first_train_ms + speed_second_train_ms
noncomputable def combined_length : ℝ := relative_speed * time_seconds

theorem length_first_train : combined_length - length_second_train = 99.9760019198464 :=
by
  sorry

end length_first_train_l174_174521


namespace suitable_graph_for_air_composition_is_pie_chart_l174_174017

/-- The most suitable type of graph to visually represent the percentage 
of each component in the air is a pie chart, based on the given conditions. -/
theorem suitable_graph_for_air_composition_is_pie_chart 
  (bar_graph : Prop)
  (line_graph : Prop)
  (pie_chart : Prop)
  (histogram : Prop)
  (H1 : bar_graph → comparing_quantities)
  (H2 : line_graph → display_data_over_time)
  (H3 : pie_chart → show_proportions_of_whole)
  (H4 : histogram → show_distribution_of_dataset) 
  : suitable_graph_to_represent_percentage = pie_chart :=
sorry

end suitable_graph_for_air_composition_is_pie_chart_l174_174017


namespace find_ticket_price_l174_174871

theorem find_ticket_price
  (P : ℝ) -- The original price of each ticket
  (h1 : 10 * 0.6 * P + 20 * 0.85 * P + 26 * P = 980) :
  P = 20 :=
sorry

end find_ticket_price_l174_174871


namespace tiles_difference_eighth_sixth_l174_174996

-- Define the side length of the nth square
def side_length (n : ℕ) : ℕ := n

-- Define the number of tiles given the side length
def number_of_tiles (n : ℕ) : ℕ := n * n

-- State the theorem about the difference in tiles between the 8th and 6th squares
theorem tiles_difference_eighth_sixth :
  number_of_tiles (side_length 8) - number_of_tiles (side_length 6) = 28 :=
by
  -- skipping the proof
  sorry

end tiles_difference_eighth_sixth_l174_174996


namespace halfway_between_l174_174927

theorem halfway_between (a b : ℚ) (h1 : a = 1/12) (h2 : b = 1/15) : (a + b) / 2 = 3 / 40 := by
  -- proofs go here
  sorry

end halfway_between_l174_174927


namespace andy_coats_l174_174833

-- Define the initial number of minks Andy buys
def initial_minks : ℕ := 30

-- Define the number of babies each mink has
def babies_per_mink : ℕ := 6

-- Define the total initial minks including babies
def total_initial_minks : ℕ := initial_minks * babies_per_mink + initial_minks

-- Define the number of minks set free by activists
def minks_set_free : ℕ := total_initial_minks / 2

-- Define the number of minks remaining after half are set free
def remaining_minks : ℕ := total_initial_minks - minks_set_free

-- Define the number of mink skins needed for one coat
def mink_skins_per_coat : ℕ := 15

-- Define the number of coats Andy can make
def coats_andy_can_make : ℕ := remaining_minks / mink_skins_per_coat

-- The theorem to prove the number of coats Andy can make
theorem andy_coats : coats_andy_can_make = 7 := by
  sorry

end andy_coats_l174_174833


namespace hannahs_son_cuts_three_strands_per_minute_l174_174577

variable (x : ℕ)

theorem hannahs_son_cuts_three_strands_per_minute
  (total_strands : ℕ)
  (hannah_rate : ℕ)
  (total_time : ℕ)
  (total_strands_cut : ℕ := hannah_rate * total_time)
  (son_rate := (total_strands - total_strands_cut) / total_time)
  (hannah_rate := 8)
  (total_time := 2)
  (total_strands := 22) :
  son_rate = 3 := 
by
  sorry

end hannahs_son_cuts_three_strands_per_minute_l174_174577


namespace least_xy_value_l174_174563

theorem least_xy_value (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
  (h : 1 / x + 1 / (3 * y) = 1 / 6) : x * y = 96 :=
by sorry

end least_xy_value_l174_174563


namespace ratio_of_side_lengths_l174_174539

theorem ratio_of_side_lengths (w1 w2 : ℝ) (s1 s2 : ℝ)
  (h1 : w1 = 8) (h2 : w2 = 64)
  (v1 : w1 = s1 ^ 3)
  (v2 : w2 = s2 ^ 3) : 
  s2 / s1 = 2 := by
  sorry

end ratio_of_side_lengths_l174_174539


namespace averagePricePerBook_l174_174013

-- Define the prices and quantities from the first store
def firstStoreFictionBooks : ℕ := 25
def firstStoreFictionPrice : ℝ := 20
def firstStoreNonFictionBooks : ℕ := 15
def firstStoreNonFictionPrice : ℝ := 30
def firstStoreChildrenBooks : ℕ := 20
def firstStoreChildrenPrice : ℝ := 8

-- Define the prices and quantities from the second store
def secondStoreFictionBooks : ℕ := 10
def secondStoreFictionPrice : ℝ := 18
def secondStoreNonFictionBooks : ℕ := 20
def secondStoreNonFictionPrice : ℝ := 25
def secondStoreChildrenBooks : ℕ := 30
def secondStoreChildrenPrice : ℝ := 5

-- Definition of total books from first and second store
def totalBooks : ℕ :=
  firstStoreFictionBooks + firstStoreNonFictionBooks + firstStoreChildrenBooks +
  secondStoreFictionBooks + secondStoreNonFictionBooks + secondStoreChildrenBooks

-- Definition of the total cost from first and second store
def totalCost : ℝ :=
  (firstStoreFictionBooks * firstStoreFictionPrice) +
  (firstStoreNonFictionBooks * firstStoreNonFictionPrice) +
  (firstStoreChildrenBooks * firstStoreChildrenPrice) +
  (secondStoreFictionBooks * secondStoreFictionPrice) +
  (secondStoreNonFictionBooks * secondStoreNonFictionPrice) +
  (secondStoreChildrenBooks * secondStoreChildrenPrice)

-- Theorem: average price per book
theorem averagePricePerBook : (totalCost / totalBooks : ℝ) = 16.17 := by
  sorry

end averagePricePerBook_l174_174013


namespace factorization_identity_l174_174326

-- We are asked to prove the mathematical equality under given conditions.
theorem factorization_identity (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 8)^2 :=
by
  sorry

end factorization_identity_l174_174326


namespace unique_ones_digits_divisible_by_8_l174_174276

/-- Carla likes numbers that are divisible by 8.
    We want to show that there are 5 unique ones digits for such numbers. -/
theorem unique_ones_digits_divisible_by_8 : 
  (Finset.card 
    (Finset.image (fun n => n % 10) 
                  (Finset.filter (fun n => n % 8 = 0) (Finset.range 100)))) = 5 := 
by
  sorry

end unique_ones_digits_divisible_by_8_l174_174276


namespace sin_cos_product_l174_174531

open Real

theorem sin_cos_product (θ : ℝ) (h : sin θ + cos θ = 3 / 4) : sin θ * cos θ = -7 / 32 := 
  by 
    sorry

end sin_cos_product_l174_174531


namespace part1_part2_part3_l174_174542

section CircleLine

-- Given: Circle C with equation x^2 + y^2 - 2x - 2y + 1 = 0
-- Tangent to line l intersecting the x-axis at A and the y-axis at B
variable (a b : ℝ) (ha : a > 2) (hb : b > 2)

-- Ⅰ. Prove that (a - 2)(b - 2) = 2
theorem part1 : (a - 2) * (b - 2) = 2 :=
sorry

-- Ⅱ. Find the equation of the trajectory of the midpoint of segment AB
theorem part2 (x y : ℝ) (hx : x > 1) (hy : y > 1) : (x - 1) * (y - 1) = 1 :=
sorry

-- Ⅲ. Find the minimum value of the area of triangle AOB
theorem part3 : ∃ (area : ℝ), area = 6 :=
sorry

end CircleLine

end part1_part2_part3_l174_174542


namespace general_term_sequence_sum_of_cn_l174_174082

theorem general_term_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) (hS : S 2 = 3)
  (hS_eq : ∀ n, 2 * S n = n + n * a n) :
  ∀ n, a n = n :=
by
  sorry

theorem sum_of_cn (S : ℕ → ℕ) (a : ℕ → ℕ) (c : ℕ → ℕ) (T : ℕ → ℕ)
  (hS : S 2 = 3)
  (hS_eq : ∀ n, 2 * S n = n + n * a n)
  (ha : ∀ n, a n = n)
  (hc_odd : ∀ n, c (2 * n - 1) = a (2 * n))
  (hc_even : ∀ n, c (2 * n) = 3 * 2^(a (2 * n - 1)) + 1) :
  ∀ n, T (2 * n) = 2^(2 * n + 1) + n^2 + 2 * n - 2 :=
by
  sorry

end general_term_sequence_sum_of_cn_l174_174082


namespace lychees_remaining_l174_174662
-- Definitions of the given conditions
def initial_lychees : ℕ := 500
def sold_lychees : ℕ := initial_lychees / 2
def home_lychees : ℕ := initial_lychees - sold_lychees
def eaten_lychees : ℕ := (3 * home_lychees) / 5

-- Statement to prove
theorem lychees_remaining : home_lychees - eaten_lychees = 100 := by
  sorry

end lychees_remaining_l174_174662


namespace second_child_birth_year_l174_174318

theorem second_child_birth_year (first_child_birth : ℕ)
  (second_child_birth : ℕ)
  (third_child_birth : ℕ)
  (fourth_child_birth : ℕ)
  (first_child_years_ago : first_child_birth = 15)
  (third_child_on_second_child_fourth_birthday : third_child_birth = second_child_birth + 4)
  (fourth_child_two_years_after_third : fourth_child_birth = third_child_birth + 2)
  (fourth_child_age : fourth_child_birth = 8) :
  second_child_birth = first_child_birth - 14 := 
by
  sorry

end second_child_birth_year_l174_174318


namespace one_million_div_one_fourth_l174_174488

theorem one_million_div_one_fourth : (1000000 : ℝ) / (1 / 4) = 4000000 := by
  sorry

end one_million_div_one_fourth_l174_174488


namespace exists_f_m_eq_n_plus_2017_l174_174131

theorem exists_f_m_eq_n_plus_2017 (m : ℕ) (h : m > 0) :
  (∃ f : ℤ → ℤ, ∀ n : ℤ, (f^[m] n = n + 2017)) ↔ (m = 1 ∨ m = 2017) :=
by
  sorry

end exists_f_m_eq_n_plus_2017_l174_174131


namespace find_number_l174_174641

theorem find_number (x : ℝ) : 
  0.05 * x = 0.20 * 650 + 190 → x = 6400 :=
by
  intro h
  sorry

end find_number_l174_174641


namespace total_pages_l174_174482

-- Definitions based on conditions
def math_pages : ℕ := 10
def extra_reading_pages : ℕ := 3
def reading_pages : ℕ := math_pages + extra_reading_pages

-- Statement of the proof problem
theorem total_pages : math_pages + reading_pages = 23 := by 
  sorry

end total_pages_l174_174482


namespace new_average_weight_l174_174059

theorem new_average_weight 
  (average_weight_19 : ℕ → ℝ)
  (weight_new_student : ℕ → ℝ)
  (new_student_count : ℕ)
  (old_student_count : ℕ)
  (h1 : average_weight_19 old_student_count = 15.0)
  (h2 : weight_new_student new_student_count = 11.0)
  : (average_weight_19 (old_student_count + new_student_count) = 14.8) :=
by
  sorry

end new_average_weight_l174_174059


namespace workshops_participation_l174_174535

variable (x y z a b c d : ℕ)
variable (A B C : Finset ℕ)

theorem workshops_participation:
  (A.card = 15) →
  (B.card = 14) →
  (C.card = 11) →
  (25 = x + y + z + a + b + c + d) →
  (12 = a + b + c + d) →
  (A.card = x + a + c + d) →
  (B.card = y + a + b + d) →
  (C.card = z + b + c + d) →
  d = 0 :=
by
  intro hA hB hC hTotal hAtLeastTwo hAkA hBkA hCkA
  -- The proof will go here
  -- Parsing these inputs shall lead to establishing d = 0
  sorry

end workshops_participation_l174_174535


namespace men_with_all_attributes_le_l174_174408

theorem men_with_all_attributes_le (total men_with_tv men_with_radio men_with_ac: ℕ) (married_men: ℕ) 
(h_total: total = 100) 
(h_married_men: married_men = 84) 
(h_men_with_tv: men_with_tv = 75) 
(h_men_with_radio: men_with_radio = 85) 
(h_men_with_ac: men_with_ac = 70) : 
  ∃ x, x ≤ men_with_ac ∧ x ≤ married_men ∧ x ≤ men_with_tv ∧ x ≤ men_with_radio ∧ (x ≤ total) := 
sorry

end men_with_all_attributes_le_l174_174408


namespace books_per_bookshelf_l174_174471

theorem books_per_bookshelf (total_books bookshelves : ℕ) (h_total_books : total_books = 34) (h_bookshelves : bookshelves = 2) : total_books / bookshelves = 17 :=
by
  sorry

end books_per_bookshelf_l174_174471


namespace sum_of_money_l174_174782

theorem sum_of_money (A B C : ℝ) (hB : B = 0.65 * A) (hC : C = 0.40 * A) (hC_val : C = 56) :
  A + B + C = 287 :=
by {
  sorry
}

end sum_of_money_l174_174782


namespace opposite_sides_range_l174_174985

theorem opposite_sides_range (a : ℝ) :
  (3 * (-3) - 2 * (-1) - a) * (3 * 4 - 2 * (-6) - a) < 0 ↔ -7 < a ∧ a < 24 :=
by
  simp
  sorry

end opposite_sides_range_l174_174985


namespace find_overall_mean_score_l174_174073

variable (M N E : ℝ)
variable (m n e : ℝ)

theorem find_overall_mean_score :
  M = 85 → N = 75 → E = 65 →
  m / n = 4 / 5 → n / e = 3 / 2 →
  ((85 * m) + (75 * n) + (65 * e)) / (m + n + e) = 82 :=
by
  sorry

end find_overall_mean_score_l174_174073


namespace fraction_zero_solution_l174_174133

theorem fraction_zero_solution (x : ℝ) (h : (x - 1) / (2 - x) = 0) : x = 1 :=
sorry

end fraction_zero_solution_l174_174133


namespace F_double_prime_coordinates_correct_l174_174843

structure Point where
  x : Int
  y : Int

def reflect_over_y_axis (p : Point) : Point :=
  { x := -p.x, y := p.y }

def reflect_over_x_axis (p : Point) : Point :=
  { x := p.x, y := -p.y }

def F : Point := { x := 6, y := -4 }

def F' : Point := reflect_over_y_axis F

def F'' : Point := reflect_over_x_axis F'

theorem F_double_prime_coordinates_correct : F'' = { x := -6, y := 4 } :=
  sorry

end F_double_prime_coordinates_correct_l174_174843


namespace tucker_boxes_l174_174780

def tissues_per_box := 160
def used_tissues := 210
def left_tissues := 270

def total_tissues := used_tissues + left_tissues

theorem tucker_boxes : total_tissues = tissues_per_box * 3 :=
by
  sorry

end tucker_boxes_l174_174780


namespace max_value_expression_l174_174221

theorem max_value_expression (a b c d : ℤ) (hb_pos : b > 0)
  (h1 : a + b = c) (h2 : b + c = d) (h3 : c + d = a) : 
  a - 2 * b + 3 * c - 4 * d = -7 := 
sorry

end max_value_expression_l174_174221


namespace find_eccentricity_l174_174537

noncomputable def ellipse_eccentricity (m : ℝ) (c : ℝ) (a : ℝ) : ℝ :=
  c / a

theorem find_eccentricity
  (m : ℝ) (c := Real.sqrt 2) (a := 3 * Real.sqrt 2 / 2)
  (h1 : 2 * m^2 - (m + 1) = 2)
  (h2 : m > 0) :
  ellipse_eccentricity m c a = 2 / 3 :=
by sorry

end find_eccentricity_l174_174537


namespace sequence_term_1000_l174_174599

theorem sequence_term_1000 (a : ℕ → ℤ) 
  (h1 : a 1 = 2010) 
  (h2 : a 2 = 2011) 
  (h3 : ∀ n, 1 ≤ n → a n + a (n + 1) + a (n + 2) = 2 * n) : 
  a 1000 = 2676 :=
sorry

end sequence_term_1000_l174_174599


namespace fraction_of_income_from_tips_l174_174952

variable (S T I : ℝ)

/- Definition of the conditions -/
def tips_condition : Prop := T = (3 / 4) * S
def income_condition : Prop := I = S + T

/- The proof problem statement, asserting the desired result -/
theorem fraction_of_income_from_tips (h1 : tips_condition S T) (h2 : income_condition S T I) : T / I = 3 / 7 := by
  sorry

end fraction_of_income_from_tips_l174_174952


namespace missing_fraction_is_two_l174_174054

theorem missing_fraction_is_two :
  (1/2) + (-5/6) + (1/5) + (1/4) + (-9/20) + (-5/6) + 2 = 0.8333333333333334 := by
  sorry

end missing_fraction_is_two_l174_174054


namespace possible_values_of_a_plus_b_l174_174484

theorem possible_values_of_a_plus_b (a b : ℤ)
  (h1 : ∃ α : ℝ, 0 ≤ α ∧ α < 2 * Real.pi ∧ (∃ (sinα cosα : ℝ), sinα = Real.sin α ∧ cosα = Real.cos α ∧ (sinα + cosα = -a) ∧ (sinα * cosα = 2 * b^2))) :
  a + b = 1 ∨ a + b = -1 := 
sorry

end possible_values_of_a_plus_b_l174_174484


namespace measure_of_angle_F_l174_174549

theorem measure_of_angle_F (angle_D angle_E angle_F : ℝ) (h1 : angle_D = 80)
  (h2 : angle_E = 4 * angle_F + 10)
  (h3 : angle_D + angle_E + angle_F = 180) : angle_F = 18 := 
by
  sorry

end measure_of_angle_F_l174_174549


namespace expression_equals_one_l174_174884

noncomputable def compute_expression : ℝ :=
  (1 - 1 / Real.cos (30 * Real.pi / 180)) *
  (1 + 1 / Real.sin (60 * Real.pi / 180)) *
  (1 - 1 / Real.sin (30 * Real.pi / 180)) *
  (1 + 1 / Real.cos (60 * Real.pi / 180))

theorem expression_equals_one : compute_expression = 1 :=
by
  sorry

end expression_equals_one_l174_174884


namespace first_discount_percentage_l174_174725

theorem first_discount_percentage (d : ℝ) (h : d > 0) :
  (∃ x : ℝ, (0 < x) ∧ (x < 100) ∧ 0.6 * d = (d * (1 - x / 100)) * 0.8) → x = 25 :=
by
  sorry

end first_discount_percentage_l174_174725


namespace recycling_drive_l174_174830

theorem recycling_drive (S : ℕ) 
  (h1 : ∀ (n : ℕ), n = 280 * S) -- Each section collected 280 kilos in two weeks
  (h2 : ∀ (t : ℕ), t = 2000 - 320) -- After the third week, they needed 320 kilos more to reach their target of 2000 kilos
  : S = 3 :=
by
  sorry

end recycling_drive_l174_174830


namespace problem_quadratic_has_real_root_l174_174286

theorem problem_quadratic_has_real_root (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end problem_quadratic_has_real_root_l174_174286


namespace product_of_real_roots_eq_one_l174_174592

theorem product_of_real_roots_eq_one:
  ∀ x : ℝ, (x ^ (Real.log x / Real.log 5) = 25) → (∀ x1 x2 : ℝ, (x1 ^ (Real.log x1 / Real.log 5) = 25) → (x2 ^ (Real.log x2 / Real.log 5) = 25) → x1 * x2 = 1) :=
by
  sorry

end product_of_real_roots_eq_one_l174_174592


namespace parabola_equation_1_parabola_equation_2_l174_174865

noncomputable def parabola_vertex_focus (vertex focus : ℝ × ℝ) : Prop :=
  ∃ p : ℝ, (focus.1 = p / 2 ∧ focus.2 = 0) ∧ (∀ x y : ℝ, y^2 = 2 * p * x ↔ y^2 = 24 * x)

noncomputable def standard_parabola_through_point (point : ℝ × ℝ) : Prop :=
  ∃ p : ℝ, ( ( point.1^2 = 2 * p * point.2 ∧ point.2 ≠ 0 ∧ point.1 ≠ 0) ∧ (∀ x y : ℝ, x^2 = 2 * p * y ↔ x^2 = y / 2) ) ∨
           ( ( point.2^2 = 2 * p * point.1 ∧ point.1 ≠ 0 ∧ point.2 ≠ 0) ∧ (∀ x y : ℝ, y^2 = 2 * p * x ↔ y^2 = 4 * x) )

theorem parabola_equation_1 : parabola_vertex_focus (0, 0) (6, 0) := 
  sorry

theorem parabola_equation_2 : standard_parabola_through_point (1, 2) := 
  sorry

end parabola_equation_1_parabola_equation_2_l174_174865


namespace least_number_to_add_l174_174495

theorem least_number_to_add (k n : ℕ) (h : k = 1015) (m : n = 25) : 
  ∃ x : ℕ, (k + x) % n = 0 ∧ x = 10 := by
  sorry

end least_number_to_add_l174_174495


namespace sum_of_areas_of_sixteen_disks_l174_174035

theorem sum_of_areas_of_sixteen_disks :
  let r := 1 - (2:ℝ).sqrt
  let area_one_disk := r^2 * Real.pi
  let total_area := 16 * area_one_disk
  total_area = Real.pi * (48 - 32 * (2:ℝ).sqrt) :=
by
  sorry

end sum_of_areas_of_sixteen_disks_l174_174035


namespace cost_of_candy_l174_174786

theorem cost_of_candy (initial_amount pencil_cost remaining_after_candy : ℕ) 
  (h1 : initial_amount = 43) 
  (h2 : pencil_cost = 20) 
  (h3 : remaining_after_candy = 18) :
  ∃ candy_cost : ℕ, candy_cost = initial_amount - pencil_cost - remaining_after_candy :=
by
  sorry

end cost_of_candy_l174_174786


namespace conic_section_is_hyperbola_l174_174494

-- Definitions for the conditions in the problem
def conic_section_equation (x y : ℝ) := (x - 4) ^ 2 = 5 * (y + 2) ^ 2 - 45

-- The theorem that we need to prove
theorem conic_section_is_hyperbola : ∀ x y : ℝ, (conic_section_equation x y) → "H" = "H" :=
by
  intro x y h
  sorry

end conic_section_is_hyperbola_l174_174494


namespace sum_of_fifth_powers_divisibility_l174_174355

theorem sum_of_fifth_powers_divisibility (a b c d e : ℤ) :
  (a^5 + b^5 + c^5 + d^5 + e^5) % 25 = 0 → (a % 5 = 0) ∨ (b % 5 = 0) ∨ (c % 5 = 0) ∨ (d % 5 = 0) ∨ (e % 5 = 0) :=
by
  sorry

end sum_of_fifth_powers_divisibility_l174_174355


namespace width_of_rectangular_prism_l174_174777

theorem width_of_rectangular_prism (l h d : ℕ) (w : ℤ) 
  (hl : l = 3) (hh : h = 12) (hd : d = 13) 
  (diag_eq : d = Int.sqrt (l^2 + w^2 + h^2)) : 
  w = 4 := by
  sorry

end width_of_rectangular_prism_l174_174777


namespace prob_neither_prime_nor_composite_l174_174087

theorem prob_neither_prime_nor_composite :
  (1 / 95 : ℚ) = 1 / 95 := by
  sorry

end prob_neither_prime_nor_composite_l174_174087


namespace complex_division_l174_174823

def imaginary_unit := Complex.I

theorem complex_division :
  (1 - 3 * imaginary_unit) / (2 + imaginary_unit) = -1 / 5 - 7 / 5 * imaginary_unit := by
  sorry

end complex_division_l174_174823


namespace fraction_sent_for_production_twice_l174_174740

variable {x : ℝ} (hx : x > 0)

theorem fraction_sent_for_production_twice :
  let initial_sulfur := (1.5 / 100 : ℝ)
  let first_sulfur_addition := (0.5 / 100 : ℝ)
  let second_sulfur_addition := (2 / 100 : ℝ) 
  (initial_sulfur - initial_sulfur * x + first_sulfur_addition * x -
    ((initial_sulfur - initial_sulfur * x + first_sulfur_addition * x) * x) + 
    second_sulfur_addition * x = initial_sulfur) → x = 1 / 2 :=
sorry

end fraction_sent_for_production_twice_l174_174740


namespace inequality_solution_non_negative_integer_solutions_l174_174441

theorem inequality_solution (x : ℝ) :
  (x - 2) / 2 ≤ (7 - x) / 3 → x ≤ 4 :=
by
  sorry

theorem non_negative_integer_solutions :
  { n : ℤ | n ≥ 0 ∧ n ≤ 4 } = {0, 1, 2, 3, 4} :=
by
  sorry

end inequality_solution_non_negative_integer_solutions_l174_174441


namespace sweet_cookies_more_than_salty_l174_174349

-- Definitions for the given conditions
def sweet_cookies_ate : Nat := 32
def salty_cookies_ate : Nat := 23

-- The statement to prove
theorem sweet_cookies_more_than_salty :
  sweet_cookies_ate - salty_cookies_ate = 9 := by
  sorry

end sweet_cookies_more_than_salty_l174_174349


namespace positive_integers_square_of_sum_of_digits_l174_174589

-- Define the sum of the digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the main theorem
theorem positive_integers_square_of_sum_of_digits :
  ∀ (n : ℕ), (n > 0) → (n = sum_of_digits n ^ 2) → (n = 1 ∨ n = 81) :=
by
  sorry

end positive_integers_square_of_sum_of_digits_l174_174589


namespace john_must_study_4_5_hours_l174_174436

-- Let "study_time" be the amount of time John needs to study for the second exam.

noncomputable def study_time_for_avg_score (hours1 score1 target_avg total_exams : ℝ) (direct_relation : Prop) :=
  2 * target_avg - score1 / (score1 / hours1)

theorem john_must_study_4_5_hours :
  study_time_for_avg_score 3 60 75 2 (60 / 3 = 90 / study_time_for_avg_score 3 60 75 2 (60 / 3 = 90 / study_time_for_avg_score 3 60 75 2 (sorry))) = 4.5 :=
sorry

end john_must_study_4_5_hours_l174_174436


namespace integer_root_of_polynomial_l174_174672

/-- Prove that -6 is a root of the polynomial equation x^3 + bx + c = 0,
    where b and c are rational numbers and 3 - sqrt(5) is a root
 -/
theorem integer_root_of_polynomial (b c : ℚ)
  (h : ∀ x : ℝ, (x^3 + (b : ℝ)*x + (c : ℝ) = 0) → x = (3 - Real.sqrt 5) ∨ x = (3 + Real.sqrt 5) ∨ x = -6) :
  ∃ x : ℤ, x^3 + b*x + c = 0 ∧ x = -6 :=
by
  sorry

end integer_root_of_polynomial_l174_174672


namespace difference_in_average_speed_l174_174789

theorem difference_in_average_speed 
  (distance : ℕ) 
  (time_diff : ℕ) 
  (speed_B : ℕ) 
  (time_B : ℕ) 
  (time_A : ℕ) 
  (speed_A : ℕ)
  (h1 : distance = 300)
  (h2 : time_diff = 3)
  (h3 : speed_B = 20)
  (h4 : time_B = distance / speed_B)
  (h5 : time_A = time_B - time_diff)
  (h6 : speed_A = distance / time_A) 
  : speed_A - speed_B = 5 := 
sorry

end difference_in_average_speed_l174_174789


namespace avg_starting_with_d_l174_174917

-- Define c and d as positive integers
variables (c d : ℤ) (hc : c > 0) (hd : d > 0)

-- Define d as the average of the seven consecutive integers starting with c
def avg_starting_with_c (c : ℤ) : ℤ := (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) / 7

-- Define the condition that d is the average of the seven consecutive integers starting with c
axiom d_is_avg_starting_with_c : d = avg_starting_with_c c

-- Prove that the average of the seven consecutive integers starting with d equals c + 6
theorem avg_starting_with_d (c d : ℤ) (hc : c > 0) (hd : d > 0) (h : d = avg_starting_with_c c) :
  ((d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6)) / 7) = c + 6 := by
  sorry

end avg_starting_with_d_l174_174917


namespace movie_marathon_first_movie_length_l174_174060

theorem movie_marathon_first_movie_length 
  (x : ℝ)
  (h2 : 1.5 * x = second_movie)
  (h3 : second_movie + x - 1 = last_movie)
  (h4 : (x + second_movie + last_movie) = 9)
  (h5 : last_movie = 2.5 * x - 1) :
  x = 2 :=
by
  sorry

end movie_marathon_first_movie_length_l174_174060


namespace average_increase_l174_174984

variable (A : ℕ) -- The batsman's average before the 17th inning
variable (runs_in_17th_inning : ℕ := 86) -- Runs made in the 17th inning
variable (new_average : ℕ := 38) -- The average after the 17th inning
variable (total_runs_16_innings : ℕ := 16 * A) -- Total runs after 16 innings
variable (total_runs_after_17_innings : ℕ := total_runs_16_innings + runs_in_17th_inning) -- Total runs after 17 innings
variable (total_runs_should_be : ℕ := 17 * new_average) -- Theoretical total runs after 17 innings

theorem average_increase :
  total_runs_after_17_innings = total_runs_should_be → (new_average - A) = 3 :=
by
  sorry

end average_increase_l174_174984


namespace smallest_number_divisible_by_15_and_36_l174_174212

theorem smallest_number_divisible_by_15_and_36 : 
  ∃ x, (∀ y, (y % 15 = 0 ∧ y % 36 = 0) → y ≥ x) ∧ x = 180 :=
by
  sorry

end smallest_number_divisible_by_15_and_36_l174_174212


namespace index_card_area_l174_174528

theorem index_card_area (a b : ℕ) (new_area : ℕ) (reduce_length reduce_width : ℕ)
  (original_length : a = 3) (original_width : b = 7)
  (reduced_area_condition : a * (b - reduce_width) = new_area)
  (reduce_width_2 : reduce_width = 2) 
  (new_area_correct : new_area = 15) :
  (a - reduce_length) * b = 7 := by
  sorry

end index_card_area_l174_174528


namespace area_rectangle_l174_174185

theorem area_rectangle 
    (x y : ℝ)
    (h1 : 5 * x + 4 * y = 10)
    (h2 : 3 * x = 2 * y) :
    5 * (x * y) = 3000 / 121 :=
by
  sorry

end area_rectangle_l174_174185


namespace b_joined_after_x_months_l174_174138

-- Establish the given conditions as hypotheses
theorem b_joined_after_x_months
  (a_start_capital : ℝ)
  (b_start_capital : ℝ)
  (profit_ratio : ℝ)
  (months_in_year : ℝ)
  (a_capital_time : ℝ)
  (b_capital_time : ℝ)
  (a_profit_ratio : ℝ)
  (b_profit_ratio : ℝ)
  (x : ℝ)
  (h1 : a_start_capital = 3500)
  (h2 : b_start_capital = 9000)
  (h3 : profit_ratio = 2 / 3)
  (h4 : months_in_year = 12)
  (h5 : a_capital_time = 12)
  (h6 : b_capital_time = 12 - x)
  (h7 : a_profit_ratio = 2)
  (h8 : b_profit_ratio = 3)
  (h_ratio : (a_start_capital * a_capital_time) / (b_start_capital * b_capital_time) = profit_ratio) :
  x = 5 :=
by
  sorry

end b_joined_after_x_months_l174_174138


namespace leftover_grass_seed_coverage_l174_174560

/-
Question: How many extra square feet could the leftover grass seed cover after Drew reseeds his lawn?

Conditions:
- One bag of grass seed covers 420 square feet of lawn.
- The lawn consists of a rectangular section and a triangular section.
- Rectangular section:
    - Length: 32 feet
    - Width: 45 feet
- Triangular section:
    - Base: 25 feet
    - Height: 20 feet
- Triangular section requires 1.5 times the standard coverage rate.
- Drew bought seven bags of seed.

Answer: The leftover grass seed coverage is 1125 square feet.
-/

theorem leftover_grass_seed_coverage
  (bag_coverage : ℕ := 420)
  (rect_length : ℕ := 32)
  (rect_width : ℕ := 45)
  (tri_base : ℕ := 25)
  (tri_height : ℕ := 20)
  (coverage_multiplier : ℕ := 15)  -- Using 15 instead of 1.5 for integer math
  (bags_bought : ℕ := 7) :
  (bags_bought * bag_coverage - 
    (rect_length * rect_width + tri_base * tri_height * coverage_multiplier / 20) = 1125) :=
  by {
    -- Placeholder for proof steps
    sorry
  }

end leftover_grass_seed_coverage_l174_174560


namespace sector_field_area_l174_174970

/-- Given a sector field with a circumference of 30 steps and a diameter of 16 steps, prove that its area is 120 square steps. --/
theorem sector_field_area (C : ℝ) (d : ℝ) (A : ℝ) : 
  C = 30 → d = 16 → A = 120 :=
by
  sorry

end sector_field_area_l174_174970


namespace natural_numbers_satisfying_condition_l174_174104

open Nat

theorem natural_numbers_satisfying_condition (r : ℕ) :
  ∃ k : Set ℕ, k = { k | ∃ s t : ℕ, k = 2^(r + s) * t ∧ 2 ∣ t ∧ 2 ∣ s } :=
by
  sorry

end natural_numbers_satisfying_condition_l174_174104


namespace above_265_is_234_l174_174021

namespace PyramidArray

-- Definition of the pyramid structure and identifying important properties
def is_number_in_pyramid (n : ℕ) : Prop :=
  ∃ k : ℕ, (k^2 - (k - 1)^2) / 2 ≥ n ∧ (k^2 - (k - 1)^2) / 2 < n + (2 * k - 1)

def row_start (k : ℕ) : ℕ :=
  (k - 1)^2 + 1

def row_end (k : ℕ) : ℕ :=
  k^2

def number_above (n : ℕ) (r : ℕ) : ℕ :=
  row_start r + ((n - row_start (r + 1)) % (2 * (r + 1) - 1))

theorem above_265_is_234 : 
  (number_above 265 16) = 234 := 
sorry

end PyramidArray

end above_265_is_234_l174_174021


namespace even_function_l174_174189

theorem even_function (f : ℝ → ℝ) (not_zero : ∃ x, f x ≠ 0) 
  (h : ∀ a b : ℝ, f (a + b) + f (a - b) = 2 * f a + 2 * f b) : 
  ∀ x : ℝ, f (-x) = f x := 
sorry

end even_function_l174_174189


namespace point_in_fourth_quadrant_l174_174736

def is_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

theorem point_in_fourth_quadrant :
  is_fourth_quadrant 2 (-3) :=
by
  sorry

end point_in_fourth_quadrant_l174_174736


namespace number_of_arrangements_l174_174264

theorem number_of_arrangements (V T : ℕ) (hV : V = 3) (hT : T = 4) :
  ∃ n : ℕ, n = 36 :=
by
  sorry

end number_of_arrangements_l174_174264


namespace no_four_nat_satisfy_l174_174779

theorem no_four_nat_satisfy:
  ∀ (x y z t : ℕ), 3 * x^4 + 5 * y^4 + 7 * z^4 ≠ 11 * t^4 :=
by
  sorry

end no_four_nat_satisfy_l174_174779


namespace miss_davis_sticks_left_l174_174080

theorem miss_davis_sticks_left (initial_sticks groups_per_class sticks_per_group : ℕ) 
(h1 : initial_sticks = 170) (h2 : groups_per_class = 10) (h3 : sticks_per_group = 15) : 
initial_sticks - (groups_per_class * sticks_per_group) = 20 :=
by sorry

end miss_davis_sticks_left_l174_174080


namespace convex_quadrilateral_inequality_l174_174050

variable (a b c d : ℝ) -- lengths of sides of quadrilateral
variable (S : ℝ) -- Area of the quadrilateral

-- Given condition: a, b, c, d are lengths of the sides of a convex quadrilateral
def is_convex_quadrilateral (a b c d : ℝ) (S : ℝ) : Prop :=
  S ≤ (a^2 + b^2 + c^2 + d^2) / 4

theorem convex_quadrilateral_inequality (a b c d : ℝ) (S : ℝ) 
  (h : is_convex_quadrilateral a b c d S) : 
  S ≤ (a^2 + b^2 + c^2 + d^2) / 4 := 
by
  sorry

end convex_quadrilateral_inequality_l174_174050


namespace original_rent_of_increased_friend_l174_174170

theorem original_rent_of_increased_friend (avg_rent : ℝ) (new_avg_rent : ℝ) (num_friends : ℝ) (rent_increase_pct : ℝ)
  (total_old_rent : ℝ) (total_new_rent : ℝ) (increase_amount : ℝ) (R : ℝ) :
  avg_rent = 800 ∧ new_avg_rent = 850 ∧ num_friends = 4 ∧ rent_increase_pct = 0.16 ∧
  total_old_rent = num_friends * avg_rent ∧ total_new_rent = num_friends * new_avg_rent ∧
  increase_amount = total_new_rent - total_old_rent ∧ increase_amount = rent_increase_pct * R →
  R = 1250 :=
by
  sorry

end original_rent_of_increased_friend_l174_174170


namespace team_leads_per_supervisor_l174_174064

def num_workers : ℕ := 390
def num_supervisors : ℕ := 13
def leads_per_worker_ratio : ℕ := 10

theorem team_leads_per_supervisor : (num_workers / leads_per_worker_ratio) / num_supervisors = 3 :=
by
  sorry

end team_leads_per_supervisor_l174_174064


namespace increasing_interval_iff_l174_174288

noncomputable def f (a x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 3 * x

def is_increasing (a : ℝ) : Prop :=
  ∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 → f a x₁ < f a x₂

theorem increasing_interval_iff (a : ℝ) (h : a ≠ 0) :
  is_increasing a ↔ a ∈ Set.Ioo (-(5/4)) 0 ∪ Set.Ioi 0 :=
sorry

end increasing_interval_iff_l174_174288


namespace dice_impossible_divisible_by_10_l174_174585

theorem dice_impossible_divisible_by_10 :
  ¬ ∃ n ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ), n % 10 = 0 :=
by
  sorry

end dice_impossible_divisible_by_10_l174_174585


namespace minimum_value_l174_174634

-- Given conditions
variables (a b c d : ℝ)
variables (h_a : a > 0) (h_b : b = 0) (h_a_eq : a = 1)

-- Define the function
def f (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- The statement to prove
theorem minimum_value (h_c : c = 0) : ∃ x : ℝ, f a b c d x = d :=
by
  -- Given the conditions a=1, b=0, and c=0, we need to show that the minimum value is d
  sorry

end minimum_value_l174_174634


namespace units_digit_quotient_l174_174845

theorem units_digit_quotient (n : ℕ) :
  (2^1993 + 3^1993) % 5 = 0 →
  ((2^1993 + 3^1993) / 5) % 10 = 3 := by
  sorry

end units_digit_quotient_l174_174845


namespace count_common_divisors_l174_174571

theorem count_common_divisors : 
  (Nat.divisors 60 ∩ Nat.divisors 90 ∩ Nat.divisors 30).card = 8 :=
by
  sorry

end count_common_divisors_l174_174571


namespace total_cost_is_correct_l174_174165

-- Define the price of pizzas
def pizza_price : ℕ := 5

-- Define the count of triple cheese and meat lovers pizzas
def triple_cheese_pizzas : ℕ := 10
def meat_lovers_pizzas : ℕ := 9

-- Define the special offers
def buy1get1free (count : ℕ) : ℕ := count / 2 + count % 2
def buy2get1free (count : ℕ) : ℕ := (count / 3) * 2 + count % 3

-- Define the cost calculations using the special offers
def cost_triple_cheese : ℕ := buy1get1free triple_cheese_pizzas * pizza_price
def cost_meat_lovers : ℕ := buy2get1free meat_lovers_pizzas * pizza_price

-- Define the total cost calculation
def total_cost : ℕ := cost_triple_cheese + cost_meat_lovers

-- The theorem we need to prove
theorem total_cost_is_correct :
  total_cost = 55 := by
  sorry

end total_cost_is_correct_l174_174165


namespace cost_per_mile_l174_174648

theorem cost_per_mile (x : ℝ) (daily_fee : ℝ) (daily_budget : ℝ) (max_miles : ℝ)
  (h1 : daily_fee = 50)
  (h2 : daily_budget = 88)
  (h3 : max_miles = 190)
  (h4 : daily_budget = daily_fee + x * max_miles) :
  x = 0.20 :=
by
  sorry

end cost_per_mile_l174_174648


namespace solve_equation_l174_174687

theorem solve_equation :
  ∀ (x : ℝ), x * (3 * x + 6) = 7 * (3 * x + 6) → (x = 7 ∨ x = -2) :=
by
  intro x
  sorry

end solve_equation_l174_174687


namespace range_a_part1_range_a_part2_l174_174757

def A (x : ℝ) : Prop := x^2 - 3*x + 2 ≤ 0
def B (x a : ℝ) : Prop := x = x^2 - 4*x + a
def C (x a : ℝ) : Prop := x^2 - a*x - 4 ≤ 0

def p (a : ℝ) : Prop := ∃ x : ℝ, A x ∧ B x a
def q (a : ℝ) : Prop := ∀ x : ℝ, A x → C x a

theorem range_a_part1 : ¬(p a) → a > 6 := sorry

theorem range_a_part2 : p a ∧ q a → 0 ≤ a ∧ a ≤ 6 := sorry

end range_a_part1_range_a_part2_l174_174757


namespace top_card_is_king_l174_174713

noncomputable def num_cards := 52
noncomputable def num_kings := 4
noncomputable def probability_king := num_kings / num_cards

theorem top_card_is_king :
  probability_king = 1 / 13 := by
  sorry

end top_card_is_king_l174_174713


namespace determine_values_l174_174551

theorem determine_values (x y : ℝ) (h1 : x - y = 25) (h2 : x * y = 36) : (x^2 + y^2 = 697) ∧ (x + y = Real.sqrt 769) :=
by
  -- Proof goes here
  sorry

end determine_values_l174_174551


namespace usb_drive_total_capacity_l174_174195

-- Define the conditions as α = total capacity, β = busy space (50%), γ = available space (50%)
variable (α : ℕ) -- Total capacity of the USB drive in gigabytes
variable (β γ : ℕ) -- Busy space and available space in gigabytes
variable (h1 : β = α / 2) -- 50% of total capacity is busy
variable (h2 : γ = 8)  -- 8 gigabytes are still available

-- Define the problem as a theorem that these conditions imply the total capacity
theorem usb_drive_total_capacity (h : γ = α / 2) : α = 16 :=
by
  -- defer the proof
  sorry

end usb_drive_total_capacity_l174_174195


namespace sales_discount_percentage_l174_174166

theorem sales_discount_percentage :
  ∀ (P N : ℝ) (D : ℝ),
  (N * 1.12 * (P * (1 - D / 100)) = P * N * (1 + 0.008)) → D = 10 :=
by
  intros P N D h
  sorry

end sales_discount_percentage_l174_174166


namespace cos_arcsin_eq_tan_arcsin_eq_l174_174898

open Real

theorem cos_arcsin_eq (h : arcsin (3 / 5) = θ) : cos (arcsin (3 / 5)) = 4 / 5 := by
  sorry

theorem tan_arcsin_eq (h : arcsin (3 / 5) = θ) : tan (arcsin (3 / 5)) = 3 / 4 := by
  sorry

end cos_arcsin_eq_tan_arcsin_eq_l174_174898


namespace quadratic_inequality_iff_l174_174213

noncomputable def quadratic_inequality_solution (x : ℝ) : Prop := x^2 + 4*x - 96 > abs x

theorem quadratic_inequality_iff (x : ℝ) : quadratic_inequality_solution x ↔ x < -12 ∨ x > 8 := by
  sorry

end quadratic_inequality_iff_l174_174213


namespace range_of_a_bisection_method_solution_l174_174835

noncomputable def f (a x : ℝ) : ℝ := a * x^3 - 2 * a * x + 3 * a - 4

theorem range_of_a (a : ℝ) (h : ∃ x : ℝ, -1 < x ∧ x < 1 ∧ f a x = 0) :
  (12 * (27 - 4 * Real.sqrt 6) / 211 ≤ a) ∧ (a ≤ 12 * (27 + 4 * Real.sqrt 6) / 211) :=
sorry

theorem bisection_method_solution (h : ∃ x : ℝ, -1 < x ∧ x < 1 ∧ f (32 / 17) x = 0) :
  ∃ x : ℝ, (0 < x) ∧ (x < 1) ∧ (|f (32 / 17) x| < 0.1) :=
sorry

end range_of_a_bisection_method_solution_l174_174835


namespace exponent_sum_l174_174689

theorem exponent_sum : (-2:ℝ) ^ 4 + (-2:ℝ) ^ (3 / 2) + (-2:ℝ) ^ 1 + 2 ^ 1 + 2 ^ (3 / 2) + 2 ^ 4 = 32 := by
  sorry

end exponent_sum_l174_174689


namespace gcd_of_12012_and_18018_l174_174219

theorem gcd_of_12012_and_18018 : Int.gcd 12012 18018 = 6006 := 
by
  -- Here we are assuming the factorization given in the conditions
  have h₁ : 12012 = 12 * 1001 := sorry
  have h₂ : 18018 = 18 * 1001 := sorry
  have gcd_12_18 : Int.gcd 12 18 = 6 := sorry
  -- This sorry will be replaced by the actual proof involving the above conditions to conclude the stated theorem
  sorry

end gcd_of_12012_and_18018_l174_174219


namespace integrality_condition_l174_174443

noncomputable def binom (n k : ℕ) : ℕ := 
  n.choose k

theorem integrality_condition (n k : ℕ) (h : 1 ≤ k) (h1 : k < n) (h2 : (k + 1) ∣ (n^2 - 3*k^2 - 2)) : 
  ∃ m : ℕ, m = (n^2 - 3*k^2 - 2) / (k + 1) ∧ (m * binom n k) % 1 = 0 :=
sorry

end integrality_condition_l174_174443


namespace find_n_eq_5_l174_174607

variable {a_n b_n : ℕ → ℤ}

def a (n : ℕ) : ℤ := 2 + 3 * (n - 1)
def b (n : ℕ) : ℤ := -2 + 4 * (n - 1)

theorem find_n_eq_5 :
  ∃ n : ℕ, a n = b n ∧ n = 5 :=
by
  sorry

end find_n_eq_5_l174_174607


namespace average_score_l174_174492

theorem average_score (T : ℝ) (M F : ℝ) (avgM avgF : ℝ) 
  (h1 : M = 0.4 * T) 
  (h2 : M + F = T) 
  (h3 : avgM = 75) 
  (h4 : avgF = 80) : 
  (75 * M + 80 * F) / T = 78 := 
  by 
  sorry

end average_score_l174_174492


namespace expression_not_equal_l174_174906

variable (a b c : ℝ)

theorem expression_not_equal :
  (a - (b - c)) ≠ (a - b - c) :=
by sorry

end expression_not_equal_l174_174906


namespace subcommittee_count_l174_174367

def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem subcommittee_count : 
  let R := 10
  let D := 4
  let subR := 4
  let subD := 2
  binomial R subR * binomial D subD = 1260 := 
by
  sorry

end subcommittee_count_l174_174367


namespace complex_unit_circle_sum_l174_174564

theorem complex_unit_circle_sum :
  let z1 := (1 + Complex.I * Real.sqrt 3) / 2
  let z2 := (1 - Complex.I * Real.sqrt 3) / 2
  (z1 ^ 8 + z2 ^ 8 = -1) :=
by
  sorry

end complex_unit_circle_sum_l174_174564


namespace two_numbers_max_product_l174_174363

theorem two_numbers_max_product :
  ∃ x y : ℝ, x - y = 4 ∧ x + y = 35 ∧ ∀ z w : ℝ, z - w = 4 → z + w = 35 → z * w ≤ x * y :=
by
  sorry

end two_numbers_max_product_l174_174363


namespace exists_no_minimum_value_has_zeros_for_any_a_not_monotonically_increasing_when_a_ge_1_exists_m_for_3_distinct_roots_l174_174848

noncomputable def f (x a : ℝ) : ℝ :=
if x > a then (x - 1)^3 else abs (x - 1)

theorem exists_no_minimum_value :
  ∃ a : ℝ, ¬ ∃ m : ℝ, ∀ x : ℝ, f x a ≥ m :=
sorry

theorem has_zeros_for_any_a (a : ℝ) : ∃ x : ℝ, f x a = 0 :=
sorry

theorem not_monotonically_increasing_when_a_ge_1 (a : ℝ) (h : a ≥ 1) :
  ¬ ∀ x y : ℝ, 1 < x → x < y → y < a → f x a ≤ f y a :=
sorry

theorem exists_m_for_3_distinct_roots (a : ℝ) (h : 1 < a ∧ a < 2) :
  ∃ m : ℝ, ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 a = m ∧ f x2 a = m ∧ f x3 a = m :=
sorry

end exists_no_minimum_value_has_zeros_for_any_a_not_monotonically_increasing_when_a_ge_1_exists_m_for_3_distinct_roots_l174_174848


namespace melissa_games_l174_174863

noncomputable def total_points_scored := 91
noncomputable def points_per_game := 7
noncomputable def number_of_games_played := total_points_scored / points_per_game

theorem melissa_games : number_of_games_played = 13 :=
by 
  sorry

end melissa_games_l174_174863


namespace points_distance_within_rectangle_l174_174994

theorem points_distance_within_rectangle :
  ∀ (points : Fin 6 → (ℝ × ℝ)), (∀ i, 0 ≤ (points i).1 ∧ (points i).1 ≤ 3 ∧ 0 ≤ (points i).2 ∧ (points i).2 ≤ 4) →
  ∃ (i j : Fin 6), i ≠ j ∧ dist (points i) (points j) ≤ Real.sqrt 2 :=
by
  sorry

end points_distance_within_rectangle_l174_174994


namespace matthew_crackers_left_l174_174008

-- Definition of the conditions:
def initial_crackers := 23
def friends := 2
def crackers_eaten_per_friend := 6

-- Calculate the number of crackers Matthew has left:
def crackers_left (total_crackers : ℕ) (num_friends : ℕ) (eaten_per_friend : ℕ) : ℕ :=
  let crackers_given := (total_crackers - total_crackers % num_friends)
  let kept_by_matthew := total_crackers % num_friends
  let remaining_with_friends := (crackers_given / num_friends - eaten_per_friend) * num_friends
  kept_by_matthew + remaining_with_friends
  
-- Theorem to prove:
theorem matthew_crackers_left : crackers_left initial_crackers friends crackers_eaten_per_friend = 11 := by
  sorry

end matthew_crackers_left_l174_174008


namespace power_function_is_odd_l174_174886

theorem power_function_is_odd (m : ℝ) (x : ℝ) (h : (m^2 - m - 1) * (-x)^m = -(m^2 - m - 1) * x^m) : m = -1 :=
sorry

end power_function_is_odd_l174_174886


namespace trapezoid_area_l174_174194

variable (a b : ℝ) (h1 : a > b)

theorem trapezoid_area (h2 : ∃ (angle1 angle2 : ℝ), angle1 = 30 ∧ angle2 = 45) : 
  (1/4) * ((a^2 - b^2) * (Real.sqrt 3 - 1)) = 
    ((1/2) * (a + b) * ((b - a) * (Real.sqrt 3 - 1) / 2)) := 
sorry

end trapezoid_area_l174_174194


namespace joe_spends_50_per_month_l174_174271

variable (X : ℕ) -- amount Joe spends per month

theorem joe_spends_50_per_month :
  let initial_amount := 240
  let resale_value := 30
  let months := 12
  let final_amount := 0 -- this means he runs out of money
  (initial_amount = months * X - months * resale_value) →
  X = 50 := 
by
  intros
  sorry

end joe_spends_50_per_month_l174_174271


namespace average_mb_per_hour_of_music_l174_174943

/--
Given a digital music library:
- It contains 14 days of music.
- The first 7 days use 10,000 megabytes of disk space.
- The next 7 days use 14,000 megabytes of disk space.
- Each day has 24 hours.

Prove that the average megabytes per hour of music in this library is 71 megabytes.
-/
theorem average_mb_per_hour_of_music
  (days_total : ℕ) 
  (days_first : ℕ) 
  (days_second : ℕ) 
  (mb_first : ℕ) 
  (mb_second : ℕ) 
  (hours_per_day : ℕ) 
  (total_mb : ℕ) 
  (total_hours : ℕ) :
  days_total = 14 →
  days_first = 7 →
  days_second = 7 →
  mb_first = 10000 →
  mb_second = 14000 →
  hours_per_day = 24 →
  total_mb = mb_first + mb_second →
  total_hours = days_total * hours_per_day →
  total_mb / total_hours = 71 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end average_mb_per_hour_of_music_l174_174943


namespace find_y_when_x_is_1_l174_174867

theorem find_y_when_x_is_1 (t : ℝ) (x : ℝ) (y : ℝ) 
  (h1 : x = 3 - 2 * t) 
  (h2 : y = 5 * t + 6) 
  (h3 : x = 1) : 
  y = 11 :=
by
  sorry

end find_y_when_x_is_1_l174_174867


namespace flowers_in_each_basket_l174_174920

-- Definitions based on the conditions
def initial_flowers (d1 d2 : Nat) : Nat := d1 + d2
def grown_flowers (initial growth : Nat) : Nat := initial + growth
def remaining_flowers (grown dead : Nat) : Nat := grown - dead
def flowers_per_basket (remaining baskets : Nat) : Nat := remaining / baskets

-- Given conditions in Lean 4
theorem flowers_in_each_basket 
    (daughters_flowers : Nat) 
    (growth : Nat) 
    (dead : Nat) 
    (baskets : Nat) 
    (h_daughters : daughters_flowers = 5 + 5) 
    (h_growth : growth = 20) 
    (h_dead : dead = 10) 
    (h_baskets : baskets = 5) : 
    flowers_per_basket (remaining_flowers (grown_flowers (initial_flowers 5 5) growth) dead) baskets = 4 := 
sorry

end flowers_in_each_basket_l174_174920


namespace math_proof_l174_174323

noncomputable def f (ω x : ℝ) := Real.sin (ω * x) - Real.sqrt 3 * Real.cos (ω * x)

theorem math_proof (h1 : ∀ x, f ω x = f ω (x + π)) (h2 : 0 < ω) :
  (ω = 2) ∧ (f 2 (-5 * Real.pi / 6) = 0) ∧ ¬∀ x : ℝ, x ∈ Set.Ioo (Real.pi / 3) (11 * Real.pi / 12) → 
  (∃ x₁ x₂ : ℝ, f 2 x₁ < f 2 x₂) ∧ (∀ x : ℝ, f 2 (x - Real.pi / 3) ≠ Real.cos (2 * x - Real.pi / 6)) := 
by
  sorry

end math_proof_l174_174323


namespace vendor_throws_away_8_percent_l174_174595

theorem vendor_throws_away_8_percent (total_apples: ℕ) (h₁ : total_apples > 0) :
    let apples_after_first_day := total_apples * 40 / 100
    let thrown_away_first_day := apples_after_first_day * 10 / 100
    let apples_after_second_day := (apples_after_first_day - thrown_away_first_day) * 30 / 100
    let thrown_away_second_day := apples_after_second_day * 20 / 100
    let apples_after_third_day := (apples_after_second_day - thrown_away_second_day) * 60 / 100
    let thrown_away_third_day := apples_after_third_day * 30 / 100
    total_apples > 0 → (8 : ℕ) * total_apples = (thrown_away_first_day + thrown_away_second_day + thrown_away_third_day) * 100 := 
by
    -- Placeholder proof
    sorry

end vendor_throws_away_8_percent_l174_174595


namespace geometric_series_product_l174_174514

theorem geometric_series_product (y : ℝ) :
  (∑'n : ℕ, (1 / 3 : ℝ) ^ n) * (∑'n : ℕ, (- 1 / 3 : ℝ) ^ n)
  = ∑'n : ℕ, (y⁻¹ : ℝ) ^ n ↔ y = 9 :=
by
  sorry

end geometric_series_product_l174_174514


namespace pump_leak_drain_time_l174_174910

theorem pump_leak_drain_time {P L : ℝ} (hP : P = 0.25) (hPL : P - L = 0.05) : (1 / L) = 5 :=
by sorry

end pump_leak_drain_time_l174_174910


namespace convex_quadrilateral_division_l174_174472

-- Definitions for convex quadrilateral and some basic geometric objects.
structure Point :=
  (x : ℝ)
  (y : ℝ)

structure Quadrilateral :=
  (A B C D : Point)
  (convex : ∀ (X Y Z : Point), (X ≠ Y) ∧ (Y ≠ Z) ∧ (Z ≠ X))

-- Definitions for lines and midpoints.
def is_midpoint (M X Y : Point) : Prop :=
  M.x = (X.x + Y.x) / 2 ∧ M.y = (X.y + Y.y) / 2

-- Preliminary to determining equal area division.
def equal_area_division (Q : Quadrilateral) (L : Point → Point → Prop) : Prop :=
  ∃ F,
    is_midpoint F Q.A Q.B ∧
    -- Assuming some way to relate area with F and L
    L Q.D F ∧
    -- Placeholder for equality of areas (details depend on how we calculate area)
    sorry

-- Problem statement in Lean 4
theorem convex_quadrilateral_division (Q : Quadrilateral) :
  ∃ L, equal_area_division Q L :=
by
  -- Proof will be constructed here based on steps in the solution
  sorry

end convex_quadrilateral_division_l174_174472


namespace hyperbola_eccentricity_l174_174773

theorem hyperbola_eccentricity 
  (a b : ℝ) (h1 : 2 * (1 : ℝ) + 1 = 0) (h2 : 0 < a) (h3 : 0 < b) 
  (h4 : b = 2 * a) : 
  (∃ e : ℝ, e = (Real.sqrt 5)) 
:= 
  sorry

end hyperbola_eccentricity_l174_174773


namespace geometric_figure_area_l174_174159

theorem geometric_figure_area :
  (∀ (z : ℂ),
     (0 < (z.re / 20)) ∧ ((z.re / 20) < 1) ∧ 
     (0 < (z.im / 20)) ∧ ((z.im / 20) < 1) ∧ 
     (0 < (20 / z.re)) ∧ ((20 / z.re) < 1) ∧ 
     (0 < (20 / z.im)) ∧ ((20 / z.im) < 1)) →
     (∃ (area : ℝ), area = 400 - 50 * Real.pi) :=
by
  sorry

end geometric_figure_area_l174_174159


namespace mary_earnings_max_hours_l174_174202

noncomputable def earnings (hours : ℕ) : ℝ :=
  if hours <= 40 then 
    hours * 10
  else if hours <= 60 then 
    (40 * 10) + ((hours - 40) * 13)
  else 
    (40 * 10) + (20 * 13) + ((hours - 60) * 16)

theorem mary_earnings_max_hours : 
  earnings 70 = 820 :=
by
  sorry

end mary_earnings_max_hours_l174_174202


namespace number_of_triangles_l174_174317

-- Defining the problem conditions
def ten_points : Finset (ℕ) := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- The main theorem to prove
theorem number_of_triangles : (ten_points.card.choose 3) = 120 :=
by
  sorry

end number_of_triangles_l174_174317


namespace intersection_of_sets_l174_174854

-- Defining set M
def M : Set ℝ := { x | x^2 + x - 2 < 0 }

-- Defining set N
def N : Set ℝ := { x | 0 < x ∧ x ≤ 2 }

-- Theorem stating the solution
theorem intersection_of_sets : M ∩ N = { x : ℝ | 0 < x ∧ x ≤ 1 } :=
by
  sorry

end intersection_of_sets_l174_174854


namespace fixed_point_of_exponential_function_l174_174308

-- The function definition and conditions are given as hypotheses
theorem fixed_point_of_exponential_function
  (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  ∃ P : ℝ × ℝ, (∀ x : ℝ, (x = 1) → P = (x, a^(x-1) - 2)) → P = (1, -1) :=
by
  sorry

end fixed_point_of_exponential_function_l174_174308


namespace total_cost_price_l174_174499

theorem total_cost_price (SP1 SP2 SP3 : ℝ) (P1 P2 P3 : ℝ) 
  (h1 : SP1 = 120) (h2 : SP2 = 150) (h3 : SP3 = 200)
  (h4 : P1 = 0.20) (h5 : P2 = 0.25) (h6 : P3 = 0.10) : (SP1 / (1 + P1) + SP2 / (1 + P2) + SP3 / (1 + P3) = 401.82) :=
by
  sorry

end total_cost_price_l174_174499


namespace certain_amount_of_seconds_l174_174889

theorem certain_amount_of_seconds (X : ℕ)
    (cond1 : 12 / X = 16 / 480) :
    X = 360 :=
by
  sorry

end certain_amount_of_seconds_l174_174889


namespace sequence_periodic_l174_174572

theorem sequence_periodic (a : ℕ → ℚ) (h1 : a 1 = 4 / 5)
  (h2 : ∀ n, 0 ≤ a n ∧ a n ≤ 1 → 
    (a (n + 1) = if a n ≤ 1 / 2 then 2 * a n else 2 * a n - 1)) :
  a 2017 = 4 / 5 :=
sorry

end sequence_periodic_l174_174572


namespace correct_system_of_equations_l174_174058

theorem correct_system_of_equations (x y : ℕ) (h1 : x + y = 145) (h2 : 10 * x + 12 * y = 1580) :
  (x + y = 145) ∧ (10 * x + 12 * y = 1580) :=
by
  sorry

end correct_system_of_equations_l174_174058


namespace remaining_fruits_correct_l174_174241

-- The definitions for the number of fruits in terms of the number of plums
def apples := 180
def plums := apples / 3
def pears := 2 * plums
def cherries := 4 * apples

-- Damien's portion of each type of fruit picked
def apples_picked := (3/5) * apples
def plums_picked := (2/3) * plums
def pears_picked := (3/4) * pears
def cherries_picked := (7/10) * cherries

-- The remaining number of fruits
def apples_remaining := apples - apples_picked
def plums_remaining := plums - plums_picked
def pears_remaining := pears - pears_picked
def cherries_remaining := cherries - cherries_picked

-- The total remaining number of fruits
def total_remaining_fruits := apples_remaining + plums_remaining + pears_remaining + cherries_remaining

theorem remaining_fruits_correct :
  total_remaining_fruits = 338 :=
by {
  -- The conditions ensure that the imported libraries are broad
  sorry
}

end remaining_fruits_correct_l174_174241


namespace bob_favorite_number_is_correct_l174_174722

def bob_favorite_number : ℕ :=
  99

theorem bob_favorite_number_is_correct :
  50 < bob_favorite_number ∧
  bob_favorite_number < 100 ∧
  bob_favorite_number % 11 = 0 ∧
  bob_favorite_number % 2 ≠ 0 ∧
  (bob_favorite_number / 10 + bob_favorite_number % 10) % 3 = 0 :=
by
  sorry

end bob_favorite_number_is_correct_l174_174722


namespace function_evaluation_l174_174118

theorem function_evaluation (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x + 1) = 2 * x ^ 2 + 1) : 
  ∀ x : ℝ, f x = 2 * x ^ 2 - 4 * x + 3 :=
sorry

end function_evaluation_l174_174118


namespace tabby_average_speed_l174_174849

noncomputable def overall_average_speed : ℝ := 
  let swimming_speed : ℝ := 1
  let cycling_speed : ℝ := 18
  let running_speed : ℝ := 6
  let time_swimming : ℝ := 2
  let time_cycling : ℝ := 3
  let time_running : ℝ := 2
  let distance_swimming := swimming_speed * time_swimming
  let distance_cycling := cycling_speed * time_cycling
  let distance_running := running_speed * time_running
  let total_distance := distance_swimming + distance_cycling + distance_running
  let total_time := time_swimming + time_cycling + time_running
  total_distance / total_time

theorem tabby_average_speed : overall_average_speed = 9.71 := sorry

end tabby_average_speed_l174_174849


namespace transform_polynomial_l174_174144

open Real

variable {x y : ℝ}

theorem transform_polynomial 
  (h1 : y = x + 1 / x) 
  (h2 : x^4 + x^3 - 4 * x^2 + x + 1 = 0) : 
  x^2 * (y^2 + y - 6) = 0 := 
sorry

end transform_polynomial_l174_174144


namespace compute_star_difference_l174_174320

def star (x y : ℤ) : ℤ := x^2 * y - 3 * x

theorem compute_star_difference : (star 6 3) - (star 3 6) = 45 := by
  sorry

end compute_star_difference_l174_174320


namespace find_m_values_l174_174878

noncomputable def lines_cannot_form_triangle (m : ℝ) : Prop :=
  (4 * m - 1 = 0) ∨ (6 * m + 1 = 0) ∨ (m^2 + m / 3 - 2 / 3 = 0)

theorem find_m_values :
  { m : ℝ | lines_cannot_form_triangle m } = {4, -1 / 6, -1, 2 / 3} :=
by
  sorry

end find_m_values_l174_174878


namespace find_b_l174_174888

theorem find_b (a b : ℝ) (h1 : (-6) * a^2 = 3 * (4 * a + b))
  (h2 : a = 1) : b = -6 :=
by 
  sorry

end find_b_l174_174888


namespace calculate_exponent_product_l174_174007

theorem calculate_exponent_product :
  (2^0.5) * (2^0.3) * (2^0.2) * (2^0.1) * (2^0.9) = 4 :=
by
  sorry

end calculate_exponent_product_l174_174007


namespace jack_finishes_in_16_days_l174_174904

noncomputable def pages_in_book : ℕ := 285
noncomputable def weekday_reading_rate : ℕ := 23
noncomputable def weekend_reading_rate : ℕ := 35
noncomputable def weekdays_per_week : ℕ := 5
noncomputable def weekends_per_week : ℕ := 2
noncomputable def weekday_skipped : ℕ := 1
noncomputable def weekend_skipped : ℕ := 1

noncomputable def pages_per_week : ℕ :=
  (weekdays_per_week - weekday_skipped) * weekday_reading_rate + 
  (weekends_per_week - weekend_skipped) * weekend_reading_rate

noncomputable def weeks_needed : ℕ :=
  pages_in_book / pages_per_week

noncomputable def pages_left_after_weeks : ℕ :=
  pages_in_book % pages_per_week

noncomputable def extra_days_needed (pages_left : ℕ) : ℕ :=
  if pages_left > weekend_reading_rate then 2
  else if pages_left > weekday_reading_rate then 2
  else 1

noncomputable def total_days_needed : ℕ :=
  weeks_needed * 7 + extra_days_needed (pages_left_after_weeks)

theorem jack_finishes_in_16_days : total_days_needed = 16 := by
  sorry

end jack_finishes_in_16_days_l174_174904


namespace expand_and_simplify_l174_174744

theorem expand_and_simplify (x y : ℝ) : 
  (x + 6) * (x + 8 + y) = x^2 + 14 * x + x * y + 48 + 6 * y :=
by sorry

end expand_and_simplify_l174_174744


namespace dog_cat_food_difference_l174_174919

theorem dog_cat_food_difference :
  let dogFood := 600
  let catFood := 327
  dogFood - catFood = 273 :=
by
  let dogFood := 600
  let catFood := 327
  show dogFood - catFood = 273
  sorry

end dog_cat_food_difference_l174_174919


namespace inequality_least_n_l174_174526

theorem inequality_least_n (n : ℕ) (h : (1 : ℝ) / n - (1 : ℝ) / (n + 2) < 1 / 15) : n = 5 :=
sorry

end inequality_least_n_l174_174526


namespace min_value_of_quadratic_l174_174988

theorem min_value_of_quadratic (x : ℝ) : ∃ m : ℝ, (∀ x, x^2 + 10 * x ≥ m) ∧ m = -25 := by
  sorry

end min_value_of_quadratic_l174_174988


namespace number_of_roses_now_l174_174800

-- Given Conditions
def initial_roses : Nat := 7
def initial_orchids : Nat := 12
def current_orchids : Nat := 20
def orchids_more_than_roses : Nat := 9

-- Question to Prove: 
theorem number_of_roses_now :
  ∃ (R : Nat), (current_orchids = R + orchids_more_than_roses) ∧ (R = 11) :=
by {
  sorry
}

end number_of_roses_now_l174_174800


namespace sum_of_digits_of_N_l174_174759

theorem sum_of_digits_of_N :
  (∃ N : ℕ, 3 * N * (N + 1) / 2 = 3825 ∧ (N.digits 10).sum = 5) :=
by
  sorry

end sum_of_digits_of_N_l174_174759


namespace range_x_range_a_l174_174431

variable {x a : ℝ}
def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := (x - 3) * (2 - x) ≥ 0

-- (1) If a = 1, find the range of x for which p ∧ q is true.
theorem range_x (h : a = 1) : 2 ≤ x ∧ x < 3 ↔ p 1 x ∧ q x := sorry

-- (2) If ¬p is a necessary but not sufficient condition for ¬q, find the range of real number a.
theorem range_a : (¬p a x → ¬q x) → (∃ a : ℝ, 1 < a ∧ a < 2) := sorry

end range_x_range_a_l174_174431


namespace mean_equality_l174_174378

theorem mean_equality (x : ℤ) (h : (8 + 10 + 24) / 3 = (16 + x + 18) / 3) : x = 8 := by 
sorry

end mean_equality_l174_174378


namespace probability_of_BEI3_is_zero_l174_174048

def isVowelOrDigit (s : Char) : Prop :=
  (s ∈ ['A', 'E', 'I', 'O', 'U']) ∨ (s.isDigit)

def isNonVowel (s : Char) : Prop :=
  s ∈ ['B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z']

def isHexDigit (s : Char) : Prop :=
  s.isDigit ∨ s ∈ ['A', 'B', 'C', 'D', 'E', 'F']

noncomputable def numPossiblePlates : Nat :=
  13 * 21 * 20 * 16

theorem probability_of_BEI3_is_zero :
    ∃ (totalPlates : Nat), 
    (totalPlates = numPossiblePlates) ∧
    ¬(isVowelOrDigit 'B') →
    (1 : ℚ) / (totalPlates : ℚ) = 0 :=
by
  sorry

end probability_of_BEI3_is_zero_l174_174048


namespace binom_sub_floor_divisible_by_prime_l174_174201

theorem binom_sub_floor_divisible_by_prime (p n : ℕ) (hp : Nat.Prime p) (hn : n ≥ p) :
  p ∣ (Nat.choose n p - (n / p)) :=
sorry

end binom_sub_floor_divisible_by_prime_l174_174201


namespace original_weight_of_marble_l174_174211

theorem original_weight_of_marble (W : ℝ) (h1 : W * 0.75 * 0.85 * 0.90 = 109.0125) : W = 190 :=
by
  sorry

end original_weight_of_marble_l174_174211


namespace find_number_of_elements_l174_174684

theorem find_number_of_elements (n S : ℕ) (h1: (S + 26) / n = 15) (h2: (S + 36) / n = 16) : n = 10 := by
  sorry

end find_number_of_elements_l174_174684


namespace sin_330_degree_l174_174566

theorem sin_330_degree : Real.sin (330 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end sin_330_degree_l174_174566


namespace painting_colors_area_l174_174623

theorem painting_colors_area
  (B G Y : ℕ)
  (h_total_blue : B + (1 / 3 : ℝ) * G = 38)
  (h_total_yellow : Y + (2 / 3 : ℝ) * G = 38)
  (h_grass_sky_relation : G = B + 6) :
  B = 27 ∧ G = 33 ∧ Y = 16 :=
by
  sorry

end painting_colors_area_l174_174623


namespace isosceles_triangle_perimeter_l174_174061

/-- 
  Given an isosceles triangle with two sides of length 6 and the third side of length 2,
  prove that the perimeter of the triangle is 14.
-/
theorem isosceles_triangle_perimeter (a b c : ℕ) (h1 : a = 6) (h2 : b = 6) (h3 : c = 2) 
  (triangle_ineq1 : a + b > c) (triangle_ineq2 : a + c > b) (triangle_ineq3 : b + c > a) :
  a + b + c = 14 :=
  sorry

end isosceles_triangle_perimeter_l174_174061


namespace time_for_one_mile_l174_174311

theorem time_for_one_mile (d v : ℝ) (mile_in_feet : ℝ) (num_circles : ℕ) 
  (circle_circumference : ℝ) (distance_in_miles : ℝ) (time : ℝ) :
  d = 50 ∧ v = 10 ∧ mile_in_feet = 5280 ∧ num_circles = 106 ∧ 
  circle_circumference = 50 * Real.pi ∧ 
  distance_in_miles = (106 * 50 * Real.pi) / 5280 ∧ 
  time = distance_in_miles / v →
  time = Real.pi / 10 :=
by {
  sorry
}

end time_for_one_mile_l174_174311


namespace sue_necklace_total_beads_l174_174632

theorem sue_necklace_total_beads :
  ∀ (purple blue green : ℕ),
  purple = 7 →
  blue = 2 * purple →
  green = blue + 11 →
  (purple + blue + green = 46) :=
by
  intros purple blue green h1 h2 h3
  rw [h1, h2, h3]
  sorry

end sue_necklace_total_beads_l174_174632


namespace rate_of_interest_is_20_l174_174331

-- Definitions of the given conditions
def principal := 400
def simple_interest := 160
def time := 2

-- Definition of the rate of interest based on the given formula
def rate_of_interest (P SI T : ℕ) : ℕ := (SI * 100) / (P * T)

-- Theorem stating that the rate of interest is 20% given the conditions
theorem rate_of_interest_is_20 :
  rate_of_interest principal simple_interest time = 20 := by
  sorry

end rate_of_interest_is_20_l174_174331


namespace question_1_question_2_l174_174669

def f (n : ℕ) : ℕ := (2 * n + 7) * 3^n + 9

theorem question_1 :
  f 1 * f 2 * f 3 = 36 * 108 * 360 := by
  sorry

theorem question_2 :
  ∃ m ≥ 2, ∀ n : ℕ, n > 0 → f n % m = 0 ∧ m = 36 := by
  sorry

end question_1_question_2_l174_174669


namespace Kyle_papers_delivered_each_week_proof_l174_174750

-- Definitions based on identified conditions
def k_m := 100        -- Number of papers delivered from Monday to Saturday
def d_m := 6          -- Number of days from Monday to Saturday
def k_s1 := 90        -- Number of regular customers on Sunday
def k_s2 := 30        -- Number of Sunday-only customers

-- Total number of papers delivered in a week
def total_papers_week := (k_m * d_m) + (k_s1 + k_s2)

theorem Kyle_papers_delivered_each_week_proof :
  total_papers_week = 720 :=
by
  sorry

end Kyle_papers_delivered_each_week_proof_l174_174750


namespace determine_d_l174_174178

theorem determine_d (a b c d : ℝ) (h : a^2 + b^2 + c^2 + 2 = d + (a + b + c - d)^(1/3)) : d = 1/2 := by
  sorry

end determine_d_l174_174178


namespace razorback_tshirt_profit_l174_174746

theorem razorback_tshirt_profit :
  let profit_per_tshirt := 9
  let cost_per_tshirt := 4
  let num_tshirts_sold := 245
  let discount := 0.2
  let selling_price := profit_per_tshirt + cost_per_tshirt
  let discount_amount := discount * selling_price
  let discounted_price := selling_price - discount_amount
  let total_revenue := discounted_price * num_tshirts_sold
  let total_production_cost := cost_per_tshirt * num_tshirts_sold
  let total_profit := total_revenue - total_production_cost
  total_profit = 1568 :=
by
  sorry

end razorback_tshirt_profit_l174_174746


namespace kirill_height_l174_174724

theorem kirill_height (K B : ℕ) (h1 : K = B - 14) (h2 : K + B = 112) : K = 49 :=
by
  sorry

end kirill_height_l174_174724


namespace triangle_angle_sum_acute_l174_174664

theorem triangle_angle_sum_acute (x : ℝ) (h1 : 60 + 70 + x = 180) (h2 : x ≠ 60 ∧ x ≠ 70) :
  x = 50 ∧ (60 < 90 ∧ 70 < 90 ∧ x < 90) := by
  sorry

end triangle_angle_sum_acute_l174_174664


namespace is_correct_functional_expression_l174_174437

variable (x : ℝ)

def is_isosceles_triangle (x : ℝ) (y : ℝ) : Prop :=
  2*x + y = 20

theorem is_correct_functional_expression (h1 : 5 < x) (h2 : x < 10) : 
  ∃ y, y = 20 - 2*x :=
by
  sorry

end is_correct_functional_expression_l174_174437


namespace find_abc_integers_l174_174382

theorem find_abc_integers (a b c : ℕ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c) 
(h4 : (a - 1) * (b - 1) * (c - 1) ∣ a * b * c - 1) : (a = 3 ∧ b = 5 ∧ c = 15) ∨ 
(a = 2 ∧ b = 4 ∧ c = 8) :=
sorry

end find_abc_integers_l174_174382


namespace unique_not_in_range_l174_174778

open Real

noncomputable def f (a b c d x : ℝ) : ℝ := (a * x + b) / (c * x + d)

theorem unique_not_in_range (a b c d : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) (h₄ : d ≠ 0)
  (h₅ : f a b c d 10 = 10) (h₆ : f a b c d 50 = 50) 
  (h₇ : ∀ x, x ≠ -d / c → f a b c d (f a b c d x) = x) :
  ∃! x, ¬ ∃ y, f a b c d y = x :=
  sorry

end unique_not_in_range_l174_174778


namespace edward_score_l174_174283

theorem edward_score (total_points : ℕ) (friend_points : ℕ) 
  (h1 : total_points = 13) (h2 : friend_points = 6) : 
  ∃ edward_points : ℕ, edward_points = 7 :=
by
  sorry

end edward_score_l174_174283


namespace balloons_remaining_proof_l174_174509

-- The initial number of balloons the clown has
def initial_balloons : ℕ := 3 * 12

-- The number of boys who buy balloons
def boys : ℕ := 3

-- The number of girls who buy balloons
def girls : ℕ := 12

-- The total number of children buying balloons
def total_children : ℕ := boys + girls

-- The remaining number of balloons after sales
def remaining_balloons (initial : ℕ) (sold : ℕ) : ℕ := initial - sold

-- Problem statement: Proof that the remaining balloons are 21 given the conditions
theorem balloons_remaining_proof : remaining_balloons initial_balloons total_children = 21 := sorry

end balloons_remaining_proof_l174_174509


namespace sum_of_integers_l174_174945

theorem sum_of_integers (x y : ℤ) (h_pos : 0 < y) (h_gt : x > y) (h_diff : x - y = 14) (h_prod : x * y = 48) : x + y = 20 :=
sorry

end sum_of_integers_l174_174945


namespace no_a_b_exist_no_a_b_c_exist_l174_174694

-- Part (a):
theorem no_a_b_exist (a b : ℕ) (h0 : 0 < a) (h1 : 0 < b) :
  ¬ (∀ n : ℕ, 0 < n → ∃ k : ℕ, a * 2^n + b * 5^n = k^2) :=
sorry

-- Part (b):
theorem no_a_b_c_exist (a b c : ℕ) (h0 : 0 < a) (h1 : 0 < b) (h2 : 0 < c) :
  ¬ (∀ n : ℕ, 0 < n → ∃ k : ℕ, a * 2^n + b * 5^n + c = k^2) :=
sorry

end no_a_b_exist_no_a_b_c_exist_l174_174694


namespace freddy_talk_time_dad_l174_174795

-- Conditions
def localRate : ℝ := 0.05
def internationalRate : ℝ := 0.25
def talkTimeBrother : ℕ := 31
def totalCost : ℝ := 10.0

-- Goal: Prove the duration of Freddy's local call to his dad is 45 minutes
theorem freddy_talk_time_dad : 
  ∃ (talkTimeDad : ℕ), 
    talkTimeDad = 45 ∧
    totalCost = (talkTimeBrother : ℝ) * internationalRate + (talkTimeDad : ℝ) * localRate := 
by
  sorry

end freddy_talk_time_dad_l174_174795


namespace largest_six_consecutive_composites_less_than_40_l174_174409

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) := ¬ is_prime n ∧ n > 1

theorem largest_six_consecutive_composites_less_than_40 :
  ∃ (seq : ℕ → ℕ) (i : ℕ),
    (∀ j : ℕ, j < 6 → is_composite (seq (i + j))) ∧ 
    (seq i < 40) ∧ 
    (seq (i+1) < 40) ∧ 
    (seq (i+2) < 40) ∧ 
    (seq (i+3) < 40) ∧ 
    (seq (i+4) < 40) ∧ 
    (seq (i+5) < 40) ∧ 
    seq (i+5) = 30 
:= sorry

end largest_six_consecutive_composites_less_than_40_l174_174409


namespace geometric_sequence_sum_S6_l174_174365

theorem geometric_sequence_sum_S6 (S : ℕ → ℝ) (S_2_eq_4 : S 2 = 4) (S_4_eq_16 : S 4 = 16) :
  S 6 = 52 :=
sorry

end geometric_sequence_sum_S6_l174_174365


namespace count_right_triangles_with_given_conditions_l174_174124

-- Define the type of our points as a pair of integers
def Point := (ℤ × ℤ)

-- Define the orthocenter being a specific point
def isOrthocenter (P : Point) := P = (-1, 7)

-- Define that a given triangle has a right angle at the origin
def rightAngledAtOrigin (O A B : Point) :=
  O = (0, 0) ∧
  (A.fst = 0 ∨ A.snd = 0) ∧
  (B.fst = 0 ∨ B.snd = 0) ∧
  (A.fst ≠ 0 ∨ A.snd ≠ 0) ∧
  (B.fst ≠ 0 ∨ B.snd ≠ 0)

-- Define that the points are lattice points
def areLatticePoints (O A B : Point) :=
  ∃ t k : ℤ, (A = (3 * t, 4 * t) ∧ B = (-4 * k, 3 * k)) ∨
            (B = (3 * t, 4 * t) ∧ A = (-4 * k, 3 * k))

-- Define the number of right triangles given the constraints
def numberOfRightTriangles : ℕ := 2

-- Statement of the problem
theorem count_right_triangles_with_given_conditions :
  ∃ (O A B : Point),
    rightAngledAtOrigin O A B ∧
    isOrthocenter (-1, 7) ∧
    areLatticePoints O A B ∧
    numberOfRightTriangles = 2 :=
  sorry

end count_right_triangles_with_given_conditions_l174_174124


namespace calculation_1_calculation_2_calculation_3_calculation_4_l174_174272

theorem calculation_1 : -3 - (-4) = 1 :=
by sorry

theorem calculation_2 : -1/3 + (-4/3) = -5/3 :=
by sorry

theorem calculation_3 : (-2) * (-3) * (-5) = -30 :=
by sorry

theorem calculation_4 : 15 / 4 * (-1/4) = -15/16 :=
by sorry

end calculation_1_calculation_2_calculation_3_calculation_4_l174_174272


namespace average_price_of_remaining_packets_l174_174014

variables (initial_avg_price : ℕ) (initial_packets : ℕ) (returned_packets : ℕ) (returned_avg_price : ℕ)

def total_initial_cost := initial_avg_price * initial_packets
def total_returned_cost := returned_avg_price * returned_packets
def remaining_packets := initial_packets - returned_packets
def total_remaining_cost := total_initial_cost initial_avg_price initial_packets - total_returned_cost returned_avg_price returned_packets
def remaining_avg_price := total_remaining_cost initial_avg_price initial_packets returned_avg_price returned_packets / remaining_packets initial_packets returned_packets

theorem average_price_of_remaining_packets :
  initial_avg_price = 20 →
  initial_packets = 5 →
  returned_packets = 2 →
  returned_avg_price = 32 →
  remaining_avg_price initial_avg_price initial_packets returned_avg_price returned_packets = 12
:=
by
  intros h1 h2 h3 h4
  rw [remaining_avg_price, total_remaining_cost, total_initial_cost, total_returned_cost]
  norm_num [h1, h2, h3, h4]
  sorry

end average_price_of_remaining_packets_l174_174014


namespace geo_seq_b_formula_b_n_sum_T_n_l174_174244

-- Define the sequence a_n 
def a (n : ℕ) : ℕ :=
  if n = 0 then 1 else sorry -- Definition based on provided conditions

-- Define the partial sum S_n
def S (n : ℕ) : ℕ :=
  if n = 0 then 1 else 4 * a (n-1) + 2 -- Given condition S_{n+1} = 4a_n + 2

-- Condition for b_n
def b (n : ℕ) : ℕ :=
  a (n+1) - 2 * a n

-- Definition for c_n
def c (n : ℕ) := (b n) / 3

-- Define the sequence terms for c_n based sequence
def T (n : ℕ) : ℝ :=
  sorry -- Needs explicit definition from given sequence part

-- Proof statements
theorem geo_seq_b : ∀ n : ℕ, b (n + 1) = 2 * b n :=
  sorry

theorem formula_b_n : ∀ n : ℕ, b n = 3 * 2^(n-1) :=
  sorry

theorem sum_T_n : ∀ n : ℕ, T n = n / (n + 1) :=
  sorry

end geo_seq_b_formula_b_n_sum_T_n_l174_174244


namespace horses_put_by_c_l174_174720

theorem horses_put_by_c (a_horses a_months b_horses b_months c_months total_cost b_cost : ℕ) (x : ℕ) 
  (h1 : a_horses = 12) 
  (h2 : a_months = 8) 
  (h3 : b_horses = 16) 
  (h4 : b_months = 9) 
  (h5 : c_months = 6) 
  (h6 : total_cost = 870) 
  (h7 : b_cost = 360) 
  (h8 : 144 / (96 + 144 + 6 * x) = 360 / 870) : 
  x = 18 := 
by 
  sorry

end horses_put_by_c_l174_174720


namespace sean_divided_by_julie_is_2_l174_174866

-- Define the sum of the first n natural numbers
def sum_natural (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define Sean's sum as twice the sum of the first 300 natural numbers
def sean_sum : ℕ := 2 * sum_natural 300

-- Define Julie's sum as the sum of the first 300 natural numbers
def julie_sum : ℕ := sum_natural 300

-- Prove that Sean's sum divided by Julie's sum is 2
theorem sean_divided_by_julie_is_2 : sean_sum / julie_sum = 2 := by
  sorry

end sean_divided_by_julie_is_2_l174_174866


namespace find_x_l174_174754

theorem find_x (x : ℝ) (h : ∑' n : ℕ, (n + 1) * x ^ n = 9) : x = 2 / 3 :=
sorry

end find_x_l174_174754


namespace wellington_population_l174_174696

theorem wellington_population 
  (W P L : ℕ)
  (h1 : P = 7 * W)
  (h2 : P = L + 800)
  (h3 : P + L = 11800) : 
  W = 900 :=
by
  sorry

end wellington_population_l174_174696


namespace youtube_more_than_tiktok_l174_174989

-- Definitions for followers in different social media platforms
def instagram_followers : ℕ := 240
def facebook_followers : ℕ := 500
def total_followers : ℕ := 3840

-- Number of followers on Twitter is half the sum of followers on Instagram and Facebook
def twitter_followers : ℕ := (instagram_followers + facebook_followers) / 2

-- Number of followers on TikTok is 3 times the followers on Twitter
def tiktok_followers : ℕ := 3 * twitter_followers

-- Calculate the number of followers on all social media except YouTube
def other_followers : ℕ := instagram_followers + facebook_followers + twitter_followers + tiktok_followers

-- Number of followers on YouTube
def youtube_followers : ℕ := total_followers - other_followers

-- Prove the number of followers on YouTube is greater than TikTok by a certain amount
theorem youtube_more_than_tiktok : youtube_followers - tiktok_followers = 510 := by
  -- Sorry is a placeholder for the proof
  sorry

end youtube_more_than_tiktok_l174_174989


namespace stratified_sampling_second_grade_l174_174191

theorem stratified_sampling_second_grade (r1 r2 r3 : ℕ) (total_sample : ℕ) (total_ratio : ℕ):
  r1 = 3 ∧ r2 = 3 ∧ r3 = 4 ∧ total_sample = 50 ∧ total_ratio = r1 + r2 + r3 →
  (r2 * total_sample) / total_ratio = 15 :=
by
  sorry

end stratified_sampling_second_grade_l174_174191


namespace simplify_sqrt_expression_l174_174039

theorem simplify_sqrt_expression :
  (3 * (Real.sqrt (4 * 3)) - 2 * (Real.sqrt (1 / 3)) +
     Real.sqrt (16 * 3)) / (2 * Real.sqrt 3) = 14 / 3 := by
sorry

end simplify_sqrt_expression_l174_174039


namespace general_term_of_sequence_l174_174783

def A := {n : ℕ | ∃ k : ℕ, k + 1 = n }
def B := {m : ℕ | ∃ k : ℕ, 3 * k - 1 = m }

theorem general_term_of_sequence (k : ℕ) : 
  ∃ a_k : ℕ, a_k ∈ A ∩ B ∧ a_k = 9 * k^2 - 9 * k + 2 :=
sorry

end general_term_of_sequence_l174_174783


namespace petya_vasya_common_result_l174_174279

theorem petya_vasya_common_result (a b : ℝ) (h1 : b ≠ 0) (h2 : a/b = (a + b)/(2 * a)) (h3 : a/b ≠ 1) : 
  a/b = -1/2 :=
by 
  sorry

end petya_vasya_common_result_l174_174279


namespace sum_arithmetic_sequence_terms_l174_174026

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, a (n + m) = a n + m * (a 1 - a 0)

theorem sum_arithmetic_sequence_terms (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a) 
  (h₅ : a 5 = 8) :
  a 2 + a 4 + a 5 + a 9 = 32 :=
by
  sorry

end sum_arithmetic_sequence_terms_l174_174026


namespace number_of_possible_A2_eq_one_l174_174502

noncomputable def unique_possible_A2 (A : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  (A^4 = 0) → (A^2 = 0)

theorem number_of_possible_A2_eq_one (A : Matrix (Fin 2) (Fin 2) ℝ) :
  unique_possible_A2 A :=
by 
  sorry

end number_of_possible_A2_eq_one_l174_174502


namespace problem1_solution_set_problem2_range_of_a_l174_174289

-- Define the functions
def f (x a : ℝ) : ℝ := |2 * x - 1| + |2 * x + a|
def g (x : ℝ) : ℝ := x + 3

-- Problem 1: Proving the solution set when a = -2
theorem problem1_solution_set (x : ℝ) : (f x (-2) < g x) ↔ (0 < x ∧ x < 2) :=
  sorry

-- Problem 2: Proving the range of a
theorem problem2_range_of_a (a : ℝ) : 
  (a > -1) ∧ (∀ x, (x ∈ Set.Icc (-a/2) (1/2) → f x a ≤ g x)) ↔ a ∈ Set.Ioo (-1) (4/3) ∨ a = 4/3 :=
  sorry

end problem1_solution_set_problem2_range_of_a_l174_174289


namespace unique_function_l174_174196

theorem unique_function (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f (x + 1) ≥ f x + 1) 
  (h2 : ∀ x y : ℝ, f (x * y) ≥ f x * f y) : 
  ∀ x : ℝ, f x = x := 
sorry

end unique_function_l174_174196


namespace compute_fraction_l174_174151

noncomputable def distinct_and_sum_zero (w x y z : ℝ) : Prop :=
w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ w + x + y + z = 0

theorem compute_fraction (w x y z : ℝ) (h : distinct_and_sum_zero w x y z) :
  (w * y + x * z) / (w^2 + x^2 + y^2 + z^2) = -1 / 2 :=
sorry

end compute_fraction_l174_174151


namespace find_value_of_X_l174_174591

theorem find_value_of_X :
  let X_initial := 5
  let S_initial := 0
  let X_increment := 3
  let target_sum := 15000
  let X := X_initial + X_increment * 56
  2 * target_sum ≥ 3 * 57 * 57 + 7 * 57 →
  X = 173 :=
by
  sorry

end find_value_of_X_l174_174591


namespace axis_of_symmetry_l174_174579

-- Define the given parabola
def parabola (x : ℝ) : ℝ := x^2 - 2 * x + 5

-- Define the statement that we need to prove
theorem axis_of_symmetry : (∃ (a : ℝ), ∀ x, parabola (x) = (x - a) ^ 2 + 4) ∧ 
                           (∃ (b : ℝ), b = 1) :=
by
  sorry

end axis_of_symmetry_l174_174579


namespace smallest_cookie_packages_l174_174179

/-- The smallest number of cookie packages Zoey can buy in order to buy an equal number of cookie
and milk packages. -/
theorem smallest_cookie_packages (n : ℕ) (h1 : ∃ k : ℕ, 5 * k = 7 * n) : n = 7 :=
sorry

end smallest_cookie_packages_l174_174179
