import Mathlib

namespace problem_b_is_proposition_l1457_145794

def is_proposition (s : String) : Prop :=
  s = "sin 45° = 1" ∨ s = "x^2 + 2x - 1 > 0"

theorem problem_b_is_proposition : is_proposition "sin 45° = 1" :=
by
  -- insert proof steps to establish that "sin 45° = 1" is a proposition
  sorry

end problem_b_is_proposition_l1457_145794


namespace triangles_with_two_colors_l1457_145756

theorem triangles_with_two_colors {n : ℕ} 
  (h1 : ∀ (p : Finset ℝ) (hn : p.card = n) 
      (e : p → p → Prop), 
      (∀ (x y : p), e x y → e x y = red ∨ e x y = yellow ∨ e x y = green) /\
      (∀ (a b c : p), 
        (e a b = red ∨ e a b = yellow ∨ e a b = green) ∧ 
        (e b c = red ∨ e b c = yellow ∨ e b c = green) ∧ 
        (e a c = red ∨ e a c = yellow ∨ e a c = green) → 
        (e a b ≠ e b c ∨ e b c ≠ e a c ∨ e a b ≠ e a c))) :
  n < 13 := 
sorry

end triangles_with_two_colors_l1457_145756


namespace quadratic_intersection_y_axis_l1457_145720

theorem quadratic_intersection_y_axis :
  (∃ y, y = 3 * (0: ℝ)^2 - 4 * (0: ℝ) + 5 ∧ (0, y) = (0, 5)) :=
by
  sorry

end quadratic_intersection_y_axis_l1457_145720


namespace total_apples_collected_l1457_145706

-- Definitions based on conditions
def number_of_green_apples : ℕ := 124
def number_of_red_apples : ℕ := 3 * number_of_green_apples

-- Proof statement
theorem total_apples_collected : number_of_red_apples + number_of_green_apples = 496 := by
  sorry

end total_apples_collected_l1457_145706


namespace johnson_and_martinez_tied_at_may_l1457_145766

def home_runs_johnson (m : String) : ℕ :=
  if m = "January" then 2 else
  if m = "February" then 12 else
  if m = "March" then 20 else
  if m = "April" then 15 else
  if m = "May" then 9 else 0

def home_runs_martinez (m : String) : ℕ :=
  if m = "January" then 5 else
  if m = "February" then 9 else
  if m = "March" then 15 else
  if m = "April" then 20 else
  if m = "May" then 9 else 0

def cumulative_home_runs (player_home_runs : String → ℕ) (months : List String) : ℕ :=
  months.foldl (λ acc m => acc + player_home_runs m) 0

def months_up_to_may : List String :=
  ["January", "February", "March", "April", "May"]

theorem johnson_and_martinez_tied_at_may :
  cumulative_home_runs home_runs_johnson months_up_to_may
  = cumulative_home_runs home_runs_martinez months_up_to_may :=
by
    sorry

end johnson_and_martinez_tied_at_may_l1457_145766


namespace sequence_properties_l1457_145775

/-- Theorem setup:
Assume a sequence {a_n} with a_1 = 1 and a_{n+1} = 2a_n / (a_n + 2)
Also, define b_n = 1 / a_n
-/
theorem sequence_properties 
  (a : ℕ → ℝ)
  (h₀ : a 1 = 1)
  (h_rec : ∀ n : ℕ, a (n + 1) = 2 * a n / (a n + 2)) :
  -- Prove that {b_n} (b n = 1 / a n) is arithmetic with common difference 1/2
  (∃ b : ℕ → ℝ, (∀ n : ℕ, b n = 1 / a n) ∧ (∀ n : ℕ, b (n + 1) = b n + 1 / 2)) ∧ 
  -- Prove the general formula for a_n
  (∀ n : ℕ, a (n + 1) = 2 / (n + 1)) := 
sorry


end sequence_properties_l1457_145775


namespace find_g4_l1457_145711

variables (g : ℝ → ℝ)

-- Given conditions
axiom condition1 : ∀ x : ℝ, g x + 3 * g (2 - x) = 2 * x^2 + x - 1
axiom condition2 : g 4 + 3 * g (-2) = 35
axiom condition3 : g (-2) + 3 * g 4 = 5

theorem find_g4 : g 4 = -5 / 2 :=
by
  sorry

end find_g4_l1457_145711


namespace gcd_m_n_is_one_l1457_145741

def m : ℕ := 122^2 + 234^2 + 344^2

def n : ℕ := 123^2 + 235^2 + 343^2

theorem gcd_m_n_is_one : Nat.gcd m n = 1 :=
by
  sorry

end gcd_m_n_is_one_l1457_145741


namespace part_a_part_b_l1457_145700

variable {A : Type} [Ring A] (h : ∀ x : A, x + x^2 + x^3 = x^4 + x^5 + x^6)

-- Part (a)
theorem part_a (x : A) (n : Nat) (hn : n ≥ 2) (hx : x^n = 0) : x = 0 :=
sorry

-- Part (b)
theorem part_b (x : A) : x^4 = x :=
by
  have h : ∀ x : A, x + x^2 + x^3 = x^4 + x^5 + x^6 := h
  sorry

end part_a_part_b_l1457_145700


namespace Uncle_Fyodor_age_l1457_145798

variable (age : ℕ)

-- Conditions from the problem
def Sharik_statement : Prop := age > 11
def Matroskin_statement : Prop := age > 10

-- The theorem stating the problem to be proved
theorem Uncle_Fyodor_age
  (H : (Sharik_statement age ∧ ¬Matroskin_statement age) ∨ (¬Sharik_statement age ∧ Matroskin_statement age)) :
  age = 11 :=
by
  sorry

end Uncle_Fyodor_age_l1457_145798


namespace total_children_l1457_145771

-- Definitions for the conditions in the problem
def boys : ℕ := 19
def girls : ℕ := 41

-- Theorem stating the total number of children is 60
theorem total_children : boys + girls = 60 :=
by
  -- calculation done to show steps, but not necessary for the final statement
  sorry

end total_children_l1457_145771


namespace probability_of_red_light_l1457_145754

-- Definitions based on the conditions
def red_duration : ℕ := 30
def yellow_duration : ℕ := 5
def green_duration : ℕ := 40

def total_cycle_time : ℕ := red_duration + yellow_duration + green_duration

-- Statement of the problem to prove the probability of seeing red light
theorem probability_of_red_light : (red_duration : ℚ) / total_cycle_time = 2 / 5 := 
by sorry

end probability_of_red_light_l1457_145754


namespace max_value_of_quadratic_at_2_l1457_145793

def f (x : ℝ) : ℝ := -2 * x^2 + 8 * x + 16

theorem max_value_of_quadratic_at_2 : ∃ (x : ℝ), x = 2 ∧ ∀ y : ℝ, f y ≤ f x :=
by
  use 2
  sorry

end max_value_of_quadratic_at_2_l1457_145793


namespace exists_k_tastrophic_function_l1457_145791

noncomputable def k_tastrophic (f : ℕ+ → ℕ+) (k : ℕ) (n : ℕ+) : Prop :=
(f^[k] n) = n^k

theorem exists_k_tastrophic_function (k : ℕ) (h : k > 1) : ∃ f : ℕ+ → ℕ+, ∀ n : ℕ+, k_tastrophic f k n :=
by sorry

end exists_k_tastrophic_function_l1457_145791


namespace total_marbles_l1457_145751

theorem total_marbles (r b g : ℕ) (total : ℕ) 
  (h_ratio : 2 * g = 4 * b) 
  (h_blue_marbles : b = 36) 
  (h_total_formula : total = r + b + g) 
  : total = 108 :=
by
  sorry

end total_marbles_l1457_145751


namespace percentage_difference_l1457_145726

theorem percentage_difference (X : ℝ) (h1 : first_num = 0.70 * X) (h2 : second_num = 0.63 * X) :
  (first_num - second_num) / first_num * 100 = 10 := by
  sorry

end percentage_difference_l1457_145726


namespace complex_subtraction_l1457_145710

theorem complex_subtraction (z1 z2 : ℂ) (h1 : z1 = 2 + 3 * I) (h2 : z2 = 3 + I) :
  z1 - z2 = -1 + 2 * I := 
by
  sorry

end complex_subtraction_l1457_145710


namespace max_of_x_l1457_145753

theorem max_of_x (x y z : ℝ) (h1 : x + y + z = 7) (h2 : xy + xz + yz = 10) : x ≤ 3 := by
  sorry

end max_of_x_l1457_145753


namespace half_angle_in_second_and_fourth_quadrants_l1457_145714

theorem half_angle_in_second_and_fourth_quadrants
  (k : ℤ) (α : ℝ) (h : 2 * k * π + π < α ∧ α < 2 * k * π + 3 * π / 2) :
  (∃ m : ℤ, m * π + π / 2 < α / 2 ∧ α / 2 < m * π + 3 * π / 4) :=
by sorry

end half_angle_in_second_and_fourth_quadrants_l1457_145714


namespace solve_equation_l1457_145732

theorem solve_equation (x : ℝ) (h : x ≠ 1) : -x^2 = (2 * x + 4) / (x - 1) → (x = -2 ∨ x = 1) :=
by
  sorry

end solve_equation_l1457_145732


namespace graph_of_equation_is_shifted_hyperbola_l1457_145786

-- Definitions
def given_equation (x y : ℝ) : Prop := x^2 - 4*y^2 - 2*x = 0

-- Theorem statement
theorem graph_of_equation_is_shifted_hyperbola :
  ∀ x y : ℝ, given_equation x y = ((x - 1)^2 = 1 + 4*y^2) :=
by
  sorry

end graph_of_equation_is_shifted_hyperbola_l1457_145786


namespace digit_swap_division_l1457_145777

theorem digit_swap_division (ab ba : ℕ) (k1 k2 : ℤ) (a b : ℕ) :
  (ab = 10 * a + b) ∧ (ba = 10 * b + a) →
  (ab % 7 = 1) ∧ (ba % 7 = 1) →
  ∃ n, n = 4 :=
by
  sorry

end digit_swap_division_l1457_145777


namespace isosceles_triangle_perimeter_l1457_145783

noncomputable def perimeter_of_isosceles_triangle : ℝ :=
  let BC := 10
  let height := 6
  let half_base := BC / 2
  let side := Real.sqrt (height^2 + half_base^2)
  let perimeter := 2 * side + BC
  perimeter

theorem isosceles_triangle_perimeter :
  let BC := 10
  let height := 6
  perimeter_of_isosceles_triangle = 2 * Real.sqrt (height^2 + (BC / 2)^2) + BC := by
  sorry

end isosceles_triangle_perimeter_l1457_145783


namespace tan_add_sin_l1457_145759

noncomputable def tan (x : ℝ) : ℝ := Real.sin x / Real.cos x

theorem tan_add_sin (h1 : tan (Real.pi / 6) = Real.sin (Real.pi / 6) / Real.cos (Real.pi / 6))
  (h2 : Real.sin (Real.pi / 6) = 1 / 2)
  (h3 : Real.cos (Real.pi / 6) = Real.sqrt 3 / 2) :
  tan (Real.pi / 6) + 4 * Real.sin (Real.pi / 6) = (Real.sqrt 3 / 3) + 2 := 
sorry

end tan_add_sin_l1457_145759


namespace quadratic_expression_value_l1457_145735

theorem quadratic_expression_value (x1 x2 : ℝ)
    (h1: x1^2 + 5 * x1 + 1 = 0)
    (h2: x2^2 + 5 * x2 + 1 = 0) :
    ( (x1 * Real.sqrt 6 / (1 + x2))^2 + (x2 * Real.sqrt 6 / (1 + x1))^2 ) = 220 := 
sorry

end quadratic_expression_value_l1457_145735


namespace linda_original_amount_l1457_145723

-- Define the original amount of money Lucy and Linda have
variables (L : ℕ) (lucy_initial : ℕ := 20)

-- Condition: If Lucy gives Linda $5, they have the same amount of money.
def condition := (lucy_initial - 5) = (L + 5)

-- Theorem: The original amount of money that Linda had
theorem linda_original_amount (h : condition L) : L = 10 := 
sorry

end linda_original_amount_l1457_145723


namespace grocer_display_proof_l1457_145752

-- Define the arithmetic sequence conditions
def num_cans_in_display (n : ℕ) : Prop :=
  let a := 1
  let d := 2
  (n * n = 225) 

-- Prove the total weight is 1125 kg
def total_weight_supported (weight_per_can : ℕ) (total_cans : ℕ) : Prop :=
  (total_cans * weight_per_can = 1125)

-- State the main theorem combining the two proofs.
theorem grocer_display_proof (n weight_per_can total_cans : ℕ) :
  num_cans_in_display n → total_weight_supported weight_per_can total_cans → 
  n = 15 ∧ total_cans * weight_per_can = 1125 :=
by {
  sorry
}

end grocer_display_proof_l1457_145752


namespace ornamental_rings_remaining_l1457_145729

-- Definitions based on conditions
variable (initial_stock : ℕ) (final_stock : ℕ)

-- Condition 1
def condition1 := initial_stock + 200 = 3 * initial_stock

-- Condition 2
def condition2 := final_stock = (200 + initial_stock) * 1 / 4 - (200 + initial_stock) / 4 + 300 - 150

-- Theorem statement to prove the final stock is 225
theorem ornamental_rings_remaining
  (h1 : condition1 initial_stock)
  (h2 : condition2 initial_stock final_stock) :
  final_stock = 225 :=
sorry

end ornamental_rings_remaining_l1457_145729


namespace percent_not_filler_l1457_145760

theorem percent_not_filler (total_weight filler_weight : ℕ) (h1 : total_weight = 180) (h2 : filler_weight = 45) : 
  ((total_weight - filler_weight) * 100 / total_weight = 75) :=
by 
  sorry

end percent_not_filler_l1457_145760


namespace triangle_altitude_from_equal_area_l1457_145745

variable (x : ℝ)

theorem triangle_altitude_from_equal_area (h : x^2 = (1 / 2) * x * altitude) :
  altitude = 2 * x := by
  sorry

end triangle_altitude_from_equal_area_l1457_145745


namespace area_increase_by_nine_l1457_145748

theorem area_increase_by_nine (a : ℝ) :
  let original_area := a^2;
  let extended_side_length := 3 * a;
  let extended_area := extended_side_length^2;
  extended_area / original_area = 9 :=
by
  let original_area := a^2;
  let extended_side_length := 3 * a;
  let extended_area := (extended_side_length)^2;
  sorry

end area_increase_by_nine_l1457_145748


namespace chosen_number_l1457_145770

theorem chosen_number (x: ℤ) (h: 2 * x - 152 = 102) : x = 127 :=
by
  sorry

end chosen_number_l1457_145770


namespace solve_inequality_l1457_145768

theorem solve_inequality (x : ℝ) : (1 + x) / 3 < x / 2 → x > 2 := 
by {
  sorry
}

end solve_inequality_l1457_145768


namespace exists_xy_nat_divisible_l1457_145761

theorem exists_xy_nat_divisible (n : ℕ) : ∃ x y : ℤ, (x^2 + y^2 - 2018) % n = 0 :=
by
  use 43, 13
  sorry

end exists_xy_nat_divisible_l1457_145761


namespace solve_for_n_l1457_145788

theorem solve_for_n (n : ℤ) : (3 : ℝ)^(2 * n + 2) = 1 / 9 ↔ n = -2 := by
  sorry

end solve_for_n_l1457_145788


namespace highest_temperature_l1457_145769

theorem highest_temperature
  (initial_temp : ℝ := 60)
  (final_temp : ℝ := 170)
  (heating_rate : ℝ := 5)
  (cooling_rate : ℝ := 7)
  (total_time : ℝ := 46) :
  ∃ T : ℝ, (T - initial_temp) / heating_rate + (T - final_temp) / cooling_rate = total_time ∧ T = 240 :=
by
  sorry

end highest_temperature_l1457_145769


namespace area_to_paint_l1457_145713

def height_of_wall : ℝ := 10
def length_of_wall : ℝ := 15
def window_height : ℝ := 3
def window_length : ℝ := 3
def door_height : ℝ := 1
def door_length : ℝ := 7

theorem area_to_paint : 
  let total_wall_area := height_of_wall * length_of_wall
  let window_area := window_height * window_length
  let door_area := door_height * door_length
  let area_to_paint := total_wall_area - window_area - door_area
  area_to_paint = 134 := 
by 
  sorry

end area_to_paint_l1457_145713


namespace find_roots_combination_l1457_145746

theorem find_roots_combination 
  (α β : ℝ)
  (hα : α^2 - 3 * α + 1 = 0)
  (hβ : β^2 - 3 * β + 1 = 0) :
  7 * α^3 + 10 * β^4 = 697 := by
  sorry

end find_roots_combination_l1457_145746


namespace circle_areas_sum_l1457_145734

theorem circle_areas_sum {r s t : ℝ}
  (h1 : r + s = 5)
  (h2 : s + t = 12)
  (h3 : r + t = 13) :
  (Real.pi * r ^ 2 + Real.pi * s ^ 2 + Real.pi * t ^ 2) = 81 * Real.pi :=
by sorry

end circle_areas_sum_l1457_145734


namespace acute_triangle_l1457_145747

variable (A B C : ℝ)

def is_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180 ∧ 0 < A ∧ 0 < B ∧ 0 < C

def each_angle_less_than_sum_of_others (A B C : ℝ) : Prop :=
  A < B + C ∧ B < A + C ∧ C < A + B

theorem acute_triangle (A B C : ℝ) 
  (h1 : is_triangle A B C) 
  (h2 : each_angle_less_than_sum_of_others A B C) : 
  A < 90 ∧ B < 90 ∧ C < 90 := 
sorry

end acute_triangle_l1457_145747


namespace sale_in_2nd_month_l1457_145797

-- Defining the variables for the sales in the months
def sale_in_1st_month : ℝ := 6435
def sale_in_3rd_month : ℝ := 7230
def sale_in_4th_month : ℝ := 6562
def sale_in_5th_month : ℝ := 6855
def required_sale_in_6th_month : ℝ := 5591
def required_average_sale : ℝ := 6600
def number_of_months : ℝ := 6
def total_sales_needed : ℝ := required_average_sale * number_of_months

-- Proof statement
theorem sale_in_2nd_month : sale_in_1st_month + x + sale_in_3rd_month + sale_in_4th_month + sale_in_5th_month + required_sale_in_6th_month = total_sales_needed → x = 6927 :=
by
  sorry

end sale_in_2nd_month_l1457_145797


namespace rectangle_area_function_relationship_l1457_145785

theorem rectangle_area_function_relationship (x y : ℝ) (h : x * y = 12) : y = 12 / x :=
by
  sorry

end rectangle_area_function_relationship_l1457_145785


namespace problem_statement_l1457_145739

noncomputable def f (x : ℝ) : ℝ := if x ≥ 1 then Real.log x / Real.log 2 else sorry

theorem problem_statement : f (1 / 2) < f (1 / 3) ∧ f (1 / 3) < f 2 :=
by
  -- Definitions based on given conditions
  have h1 : ∀ x : ℝ, f (2 - x) = f x := sorry
  have h2 : ∀ x : ℝ, 1 ≤ x → f x = Real.log x / Real.log 2 := sorry
  -- Proof of the statement based on h1 and h2
  sorry

end problem_statement_l1457_145739


namespace smallest_non_factor_product_of_48_l1457_145701

theorem smallest_non_factor_product_of_48 :
  ∃ (x y : ℕ), x ≠ y ∧ x * y ≤ 48 ∧ (x ∣ 48) ∧ (y ∣ 48) ∧ ¬ (x * y ∣ 48) ∧ x * y = 18 :=
by
  sorry

end smallest_non_factor_product_of_48_l1457_145701


namespace bees_population_reduction_l1457_145743

theorem bees_population_reduction :
  ∀ (initial_population loss_per_day : ℕ),
  initial_population = 80000 → 
  loss_per_day = 1200 → 
  ∃ days : ℕ, initial_population - days * loss_per_day = initial_population / 4 ∧ days = 50 :=
by
  intros initial_population loss_per_day h_initial h_loss
  use 50
  sorry

end bees_population_reduction_l1457_145743


namespace milk_cost_is_3_l1457_145780

def Banana_cost : ℝ := 2
def Sales_tax_rate : ℝ := 0.20
def Total_spent : ℝ := 6

theorem milk_cost_is_3 (Milk_cost : ℝ) :
  Total_spent = (Milk_cost + Banana_cost) + Sales_tax_rate * (Milk_cost + Banana_cost) → 
  Milk_cost = 3 :=
by
  simp [Banana_cost, Sales_tax_rate, Total_spent]
  sorry

end milk_cost_is_3_l1457_145780


namespace buffet_dishes_l1457_145707

-- To facilitate the whole proof context, but skipping proof parts with 'sorry'

-- Oliver will eat if there is no mango in the dishes

variables (D : ℕ) -- Total number of dishes

-- Conditions:
variables (h1 : 3 <= D) -- there are at least 3 dishes with mango salsa
variables (h2 : 1 ≤ D / 6) -- one-sixth of dishes have fresh mango
variables (h3 : 1 ≤ D) -- there's at least one dish with mango jelly
variables (h4 : D / 6 ≥ 2) -- Oliver can pick out the mangoes from 2 of dishes with fresh mango
variables (h5 : D - (3 + (D / 6 - 2) + 1) = 28) -- there are 28 dishes Oliver can eat

theorem buffet_dishes : D = 36 :=
by
  sorry -- Skip the actual proof

end buffet_dishes_l1457_145707


namespace solution_set_inequality_l1457_145773

theorem solution_set_inequality (a b : ℝ)
  (h : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 3 → x^2 + a * x + b ≤ 0) :
  a * b = 6 :=
by {
  sorry
}

end solution_set_inequality_l1457_145773


namespace solution_set_of_inequality_l1457_145795

theorem solution_set_of_inequality (x : ℝ) (h : x ≠ 0) : 
  (x + 1) / x ≤ 3 ↔ x ∈ Set.Iio (0 : ℝ) ∪ Set.Ici (1 / 2) :=
by sorry

end solution_set_of_inequality_l1457_145795


namespace correlation_index_l1457_145790

variable (height_variation_weight_explained : ℝ)
variable (random_errors_contribution : ℝ)

def R_squared : ℝ := height_variation_weight_explained

theorem correlation_index (h1 : height_variation_weight_explained = 0.64) (h2 : random_errors_contribution = 0.36) : R_squared height_variation_weight_explained = 0.64 :=
by
  exact h1  -- Placeholder for actual proof, since only statement is required

end correlation_index_l1457_145790


namespace correct_negation_of_p_l1457_145750

open Real

def proposition_p (x : ℝ) := x > 0 → sin x ≥ -1

theorem correct_negation_of_p :
  ¬ (∀ x, proposition_p x) ↔ (∃ x, x > 0 ∧ sin x < -1) :=
by
  sorry

end correct_negation_of_p_l1457_145750


namespace fraction_identity_l1457_145778

theorem fraction_identity (x y z v : ℝ) (hy : y ≠ 0) (hv : v ≠ 0)
    (h : x / y + z / v = 1) : x / y - z / v = (x / y) ^ 2 - (z / v) ^ 2 := by
  sorry

end fraction_identity_l1457_145778


namespace berries_difference_l1457_145724

theorem berries_difference (total_berries : ℕ) (dima_rate : ℕ) (sergey_rate : ℕ)
  (sergey_berries_picked : ℕ) (dima_berries_picked : ℕ)
  (dima_basket : ℕ) (sergey_basket : ℕ) :
  total_berries = 900 →
  sergey_rate = 2 * dima_rate →
  sergey_berries_picked = 2 * (total_berries / 3) →
  dima_berries_picked = total_berries / 3 →
  sergey_basket = sergey_berries_picked / 2 →
  dima_basket = (2 * dima_berries_picked) / 3 →
  sergey_basket > dima_basket ∧ sergey_basket - dima_basket = 100 :=
by
  intro h_total h_rate h_sergey_picked h_dima_picked h_sergey_basket h_dima_basket
  sorry

end berries_difference_l1457_145724


namespace calculation_of_expression_l1457_145740

theorem calculation_of_expression :
  (1.99 ^ 2 - 1.98 * 1.99 + 0.99 ^ 2) = 1 := 
by sorry

end calculation_of_expression_l1457_145740


namespace average_weight_of_whole_class_l1457_145703

theorem average_weight_of_whole_class :
  let students_A := 50
  let students_B := 50
  let avg_weight_A := 60
  let avg_weight_B := 80
  let total_students := students_A + students_B
  let total_weight_A := students_A * avg_weight_A
  let total_weight_B := students_B * avg_weight_B
  let total_weight := total_weight_A + total_weight_B
  let avg_weight := total_weight / total_students
  avg_weight = 70 := 
by 
  sorry

end average_weight_of_whole_class_l1457_145703


namespace sides_of_polygon_l1457_145796

theorem sides_of_polygon (n : ℕ) : 
  (n - 2) * 180 = 3 * 360 - 180 → n = 7 :=
by
  sorry

end sides_of_polygon_l1457_145796


namespace face_value_of_share_l1457_145721

theorem face_value_of_share (FV : ℝ) (dividend_percent : ℝ) (interest_percent : ℝ) (market_value : ℝ) :
  dividend_percent = 0.09 → 
  interest_percent = 0.12 →
  market_value = 33 →
  (0.09 * FV = 0.12 * 33) → FV = 44 :=
by
  intros
  sorry

end face_value_of_share_l1457_145721


namespace anusha_receives_84_l1457_145774

-- Define the conditions as given in the problem
def anusha_amount (A : ℕ) (B : ℕ) (E : ℕ) : Prop :=
  12 * A = 8 * B ∧ 12 * A = 6 * E ∧ A + B + E = 378

-- Lean statement to prove the amount Anusha gets is 84
theorem anusha_receives_84 (A B E : ℕ) (h : anusha_amount A B E) : A = 84 :=
sorry

end anusha_receives_84_l1457_145774


namespace michael_earnings_l1457_145799

-- Define variables for pay rates and hours.
def regular_pay_rate : ℝ := 7.00
def overtime_multiplier : ℝ := 2
def regular_hours : ℝ := 40
def overtime_hours (total_hours : ℝ) : ℝ := total_hours - regular_hours

-- Define the earnings functions.
def regular_earnings (hourly_rate : ℝ) (hours : ℝ) : ℝ := hourly_rate * hours
def overtime_earnings (hourly_rate : ℝ) (multiplier : ℝ) (hours : ℝ) : ℝ := hourly_rate * multiplier * hours

-- Total earnings calculation.
def total_earnings (total_hours : ℝ) : ℝ := 
regular_earnings regular_pay_rate regular_hours + 
overtime_earnings regular_pay_rate overtime_multiplier (overtime_hours total_hours)

-- The theorem to prove the correct earnings for 42.857142857142854 hours worked.
theorem michael_earnings : total_earnings 42.857142857142854 = 320 := by
  sorry

end michael_earnings_l1457_145799


namespace triangle_square_ratio_l1457_145718

theorem triangle_square_ratio (s_t s_s : ℕ) (h : 3 * s_t = 4 * s_s) : (s_t : ℚ) / s_s = 4 / 3 := by
  sorry

end triangle_square_ratio_l1457_145718


namespace rowing_upstream_speed_l1457_145765

-- Definitions based on conditions
def V_m : ℝ := 45 -- speed of the man in still water
def V_downstream : ℝ := 53 -- speed of the man rowing downstream
def V_s : ℝ := V_downstream - V_m -- speed of the stream
def V_upstream : ℝ := V_m - V_s -- speed of the man rowing upstream

-- The goal is to prove that the speed of the man rowing upstream is 37 kmph
theorem rowing_upstream_speed :
  V_upstream = 37 := by
  sorry

end rowing_upstream_speed_l1457_145765


namespace cupric_cyanide_formation_l1457_145757

/--
Given:
1 mole of CuSO₄ 
2 moles of HCN

Prove:
The number of moles of Cu(CN)₂ formed is 0.
-/
theorem cupric_cyanide_formation (CuSO₄ HCN : ℕ) (h₁ : CuSO₄ = 1) (h₂ : HCN = 2) : 0 = 0 :=
by
  -- Proof goes here
  sorry

end cupric_cyanide_formation_l1457_145757


namespace locus_of_tangency_centers_l1457_145787

def locus_of_centers (a b : ℝ) : Prop := 8 * a ^ 2 + 9 * b ^ 2 - 16 * a - 64 = 0

theorem locus_of_tangency_centers (a b : ℝ)
  (hx1 : ∃ x y : ℝ, x ^ 2 + y ^ 2 = 1) 
  (hx2 : ∃ x y : ℝ, (x - 2) ^ 2 + y ^ 2 = 25) 
  (hcent : ∃ r : ℝ, a^2 + b^2 = (r + 1)^2 ∧ (a - 2)^2 + b^2 = (5 - r)^2) : 
  locus_of_centers a b :=
sorry

end locus_of_tangency_centers_l1457_145787


namespace ratio_calculation_l1457_145709

theorem ratio_calculation (A B C : ℚ)
  (h_ratio : (A / B = 3 / 2) ∧ (B / C = 2 / 5)) :
  (4 * A + 3 * B) / (5 * C - 2 * B) = 15 / 23 := by
  sorry

end ratio_calculation_l1457_145709


namespace cat_total_birds_caught_l1457_145744

theorem cat_total_birds_caught (day_birds night_birds : ℕ) 
  (h1 : day_birds = 8) 
  (h2 : night_birds = 2 * day_birds) :
  day_birds + night_birds = 24 :=
sorry

end cat_total_birds_caught_l1457_145744


namespace part1_part2_l1457_145755

open Set

def f (x : ℝ) : ℝ := abs (x + 2) - abs (2 * x - 1)

def M : Set ℝ := { x | f x > 0 }

theorem part1 :
  M = { x | - (1 / 3 : ℝ) < x ∧ x < 3 } :=
sorry

theorem part2 :
  ∀ (x y : ℝ), x ∈ M → y ∈ M → abs (x + y + x * y) < 15 :=
sorry

end part1_part2_l1457_145755


namespace trajectory_center_of_C_number_of_lines_l_l1457_145782

noncomputable def trajectory_equation : Prop :=
  ∃ (a b : ℝ), a = 4 ∧ b^2 = 12 ∧ (∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1)

noncomputable def line_count : Prop :=
  ∀ (k m : ℤ), 
  ∃ (num_lines : ℕ), 
  (∀ (x : ℝ), (3 + 4 * k^2) * x^2 + 8 * k * m * x + 4 * m^2 - 48 = 0 → num_lines = 9 ∨ num_lines = 0) ∧
  (∀ (x : ℝ), (3 - k^2) * x^2 - 2 * k * m * x - m^2 - 12 = 0 → num_lines = 9 ∨ num_lines = 0)

theorem trajectory_center_of_C :
  trajectory_equation :=
sorry

theorem number_of_lines_l :
  line_count :=
sorry

end trajectory_center_of_C_number_of_lines_l_l1457_145782


namespace rachel_picked_total_apples_l1457_145789

-- Define the conditions
def num_trees : ℕ := 4
def apples_per_tree_picked : ℕ := 7
def apples_remaining : ℕ := 29

-- Define the total apples picked
def total_apples_picked : ℕ := num_trees * apples_per_tree_picked

-- Formal statement of the goal
theorem rachel_picked_total_apples : total_apples_picked = 28 := 
by
  sorry

end rachel_picked_total_apples_l1457_145789


namespace sufficient_but_not_necessary_condition_l1457_145719

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x^2 - 2*x < 0) → (|x - 2| < 2) ∧ ¬(|x - 2| < 2) → (x^2 - 2*x < 0 ↔ |x-2| < 2) :=
sorry

end sufficient_but_not_necessary_condition_l1457_145719


namespace solve_equation_l1457_145733

-- Defining the original equation as a Lean function
def equation (x : ℝ) : Prop :=
  (1 / (x + 8) + 1 / (x + 5) = 1 / (x + 11) + 1 / (x + 2))

theorem solve_equation :
  ∃ x : ℝ, equation x ∧ x = -13 / 2 :=
by
  -- Equation specification and transformations
  sorry

end solve_equation_l1457_145733


namespace polynomial_remainder_l1457_145749

theorem polynomial_remainder (a b : ℝ) (h : ∀ x : ℝ, (x^3 - 2*x^2 + a*x + b) % ((x - 1)*(x - 2)) = 2*x + 1) : 
  a = 1 ∧ b = 3 := 
sorry

end polynomial_remainder_l1457_145749


namespace line_through_two_quadrants_l1457_145727

theorem line_through_two_quadrants (k : ℝ) (h : k ≠ 0) :
  (∀ x : ℝ, (x > 0 → k * x > 0) ∧ (x < 0 → k * x < 0)) → k > 0 :=
sorry

end line_through_two_quadrants_l1457_145727


namespace greatest_common_divisor_of_120_and_m_l1457_145781

theorem greatest_common_divisor_of_120_and_m (m : ℕ) (h : (∀ d, d ∣ 120 ∧ d ∣ m → d = 1 ∨ d = 2 ∨ d = 4)) : gcd 120 m = 4 :=
by
  sorry

end greatest_common_divisor_of_120_and_m_l1457_145781


namespace choir_members_correct_l1457_145725

noncomputable def choir_membership : ℕ :=
  let n := 226
  n

theorem choir_members_correct (n : ℕ) (h1 : n % 10 = 6) (h2 : n % 11 = 6) (h3 : 200 ≤ n) (h4 : n ≤ 300) : n = 226 :=
by
  sorry

end choir_members_correct_l1457_145725


namespace complement_of_A_with_respect_to_U_l1457_145737

open Set

-- Definitions
def U : Set ℤ := {-1, 1, 3}
def A : Set ℤ := {-1}

-- Theorem statement
theorem complement_of_A_with_respect_to_U :
  (U \ A) = {1, 3} :=
by
  sorry

end complement_of_A_with_respect_to_U_l1457_145737


namespace angle_C_eq_pi_div_3_find_ab_values_l1457_145716

noncomputable def find_angle_C (A B C : ℝ) (a b c : ℝ) : ℝ :=
  if c * Real.cos B + b * Real.cos C = 2 * a * Real.cos C then C else 0

noncomputable def find_sides_ab (A B C : ℝ) (c S : ℝ) : Set (ℝ × ℝ) :=
  if C = Real.pi / 3 ∧ c = 2 * Real.sqrt 3 ∧ S = 2 * Real.sqrt 3 then
    { (a, b) | a^4 - 20 * a^2 + 64 = 0 ∧ b = 8 / a } else
    ∅

theorem angle_C_eq_pi_div_3 (A B C : ℝ) (a b c : ℝ) :
  (c * Real.cos B + b * Real.cos C = 2 * a * Real.cos C)
  ↔ (C = Real.pi / 3) :=
sorry

theorem find_ab_values (A B C : ℝ) (c S a b : ℝ) :
  (C = Real.pi / 3) ∧ (c = 2 * Real.sqrt 3) ∧ (S = 2 * Real.sqrt 3) ∧ (a^4 - 20 * a^2 + 64 = 0) ∧ (b = 8 / a)
  ↔ ((a, b) = (2, 4) ∨ (a, b) = (4, 2)) :=
sorry

end angle_C_eq_pi_div_3_find_ab_values_l1457_145716


namespace sum_of_distances_from_circumcenter_to_sides_l1457_145705

theorem sum_of_distances_from_circumcenter_to_sides :
  let r1 := 3
  let r2 := 5
  let r3 := 7
  let a := r1 + r2
  let b := r1 + r3
  let c := r2 + r3
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let r_incircle := area / s
  r_incircle = Real.sqrt 7 →
  let sum_distances := (7 / 4) + (7 / (3 * Real.sqrt 6)) + (7 / (Real.sqrt 30))
  sum_distances = (7 / 4) + (7 / (3 * Real.sqrt 6)) + (7 / (Real.sqrt 30))
:= sorry

end sum_of_distances_from_circumcenter_to_sides_l1457_145705


namespace ribbon_per_gift_l1457_145767

-- Definitions for the conditions in the problem
def total_ribbon_used : ℚ := 4/15
def num_gifts: ℕ := 5

-- Statement to prove
theorem ribbon_per_gift : total_ribbon_used / num_gifts = 4 / 75 :=
by
  sorry

end ribbon_per_gift_l1457_145767


namespace zoes_apartment_number_units_digit_is_1_l1457_145758

-- Defining the conditions as the initial problem does
def is_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def is_divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

def has_digit_two (n : ℕ) : Prop :=
  n / 10 = 2 ∨ n % 10 = 2

def three_out_of_four (n : ℕ) : Prop :=
  (is_square n ∧ is_odd n ∧ is_divisible_by_3 n ∧ ¬ has_digit_two n) ∨
  (is_square n ∧ is_odd n ∧ ¬ is_divisible_by_3 n ∧ has_digit_two n) ∨
  (is_square n ∧ ¬ is_odd n ∧ is_divisible_by_3 n ∧ has_digit_two n) ∨
  (¬ is_square n ∧ is_odd n ∧ is_divisible_by_3 n ∧ has_digit_two n)

theorem zoes_apartment_number_units_digit_is_1 : ∃ n : ℕ, is_two_digit_number n ∧ three_out_of_four n ∧ n % 10 = 1 :=
by
  sorry

end zoes_apartment_number_units_digit_is_1_l1457_145758


namespace unique_n_value_l1457_145728

theorem unique_n_value (n : ℕ) (d : ℕ → ℕ) (h1 : 1 = d 1) (h2 : ∀ i, d i ≤ n) (h3 : ∀ i j, i < j → d i < d j) 
                       (h4 : d (n - 1) = n) (h5 : ∃ k, k ≥ 4 ∧ ∀ i ≤ k, d i ∣ n)
                       (h6 : ∃ d1 d2 d3 d4, d 1 = d1 ∧ d 2 = d2 ∧ d 3 = d3 ∧ d 4 = d4 ∧ n = d1^2 + d2^2 + d3^2 + d4^2) : 
                       n = 130 := sorry

end unique_n_value_l1457_145728


namespace total_number_of_animals_l1457_145738

-- Define the data and conditions
def total_legs : ℕ := 38
def chickens : ℕ := 5
def chicken_legs : ℕ := 2
def sheep_legs : ℕ := 4

-- Define the proof problem
theorem total_number_of_animals (h1 : total_legs = 38) 
                                (h2 : chickens = 5) 
                                (h3 : chicken_legs = 2) 
                                (h4 : sheep_legs = 4) : 
  (∃ sheep : ℕ, chickens + sheep = 12) :=
by 
  sorry

end total_number_of_animals_l1457_145738


namespace complement_and_intersection_l1457_145792

open Set

def U : Set ℤ := {-2, -1, 0, 1, 2}
def A : Set ℤ := {-2, -1, 0}
def B : Set ℤ := {0, 1, 2}

theorem complement_and_intersection :
  ((U \ A) ∩ B) = {1, 2} := 
by
  sorry

end complement_and_intersection_l1457_145792


namespace average_and_variance_of_original_data_l1457_145776

theorem average_and_variance_of_original_data (μ σ_sq : ℝ)
  (h1 : 2 * μ - 80 = 1.2)
  (h2 : 4 * σ_sq = 4.4) :
  μ = 40.6 ∧ σ_sq = 1.1 :=
by
  sorry

end average_and_variance_of_original_data_l1457_145776


namespace cat_food_inequality_l1457_145712

variable (B S : ℝ)
variable (h1 : B > S)
variable (h2 : B < 2 * S)
variable (h3 : B + 2 * S = 2 * (B + 2 * S) / 2)

theorem cat_food_inequality : 4 * B + 4 * S < 3 * (B + 2 * S) := by
  sorry

end cat_food_inequality_l1457_145712


namespace find_x_l1457_145702

theorem find_x (x : ℝ) (h: 0.8 * 90 = 70 / 100 * x + 30) : x = 60 :=
by
  sorry

end find_x_l1457_145702


namespace student_total_marks_l1457_145784

theorem student_total_marks (total_questions correct_answers incorrect_answer_score correct_answer_score : ℕ)
    (h1 : total_questions = 60)
    (h2 : correct_answers = 38)
    (h3 : correct_answer_score = 4)
    (h4 : incorrect_answer_score = 1)
    (incorrect_answers := total_questions - correct_answers) 
    : (correct_answers * correct_answer_score - incorrect_answers * incorrect_answer_score) = 130 :=
by
  -- proof to be provided here
  sorry

end student_total_marks_l1457_145784


namespace increasing_sequence_a_range_l1457_145708

theorem increasing_sequence_a_range (f : ℕ → ℝ) (a : ℝ)
  (h1 : ∀ n, f n = if n ≤ 7 then (3 - a) * n - 3 else a ^ (n - 6))
  (h2 : ∀ n : ℕ, f n < f (n + 1)) :
  2 < a ∧ a < 3 :=
sorry

end increasing_sequence_a_range_l1457_145708


namespace solve_eq1_solve_eq2_l1457_145722

theorem solve_eq1 (x : ℝ) : (x^2 - 2 * x - 8 = 0) ↔ (x = 4 ∨ x = -2) :=
sorry

theorem solve_eq2 (x : ℝ) : (2 * x^2 - 4 * x + 1 = 0) ↔ (x = (2 + Real.sqrt 2) / 2 ∨ x = (2 - Real.sqrt 2) / 2) :=
sorry

end solve_eq1_solve_eq2_l1457_145722


namespace sum_of_two_integers_l1457_145762

theorem sum_of_two_integers (x y : ℝ) (h₁ : x^2 + y^2 = 130) (h₂ : x * y = 45) : x + y = 2 * Real.sqrt 55 :=
sorry

end sum_of_two_integers_l1457_145762


namespace emails_difference_l1457_145763

theorem emails_difference
  (emails_morning : ℕ)
  (emails_afternoon : ℕ)
  (h_morning : emails_morning = 10)
  (h_afternoon : emails_afternoon = 3)
  : emails_morning - emails_afternoon = 7 := by
  sorry

end emails_difference_l1457_145763


namespace find_pure_imaginary_solutions_l1457_145731

noncomputable def poly_eq_zero (x : ℂ) : Prop :=
  x^4 - 6 * x^3 + 13 * x^2 - 42 * x - 72 = 0

noncomputable def is_imaginary (x : ℂ) : Prop :=
  x.im ≠ 0 ∧ x.re = 0

theorem find_pure_imaginary_solutions :
  ∀ x : ℂ, poly_eq_zero x ∧ is_imaginary x ↔ (x = Complex.I * Real.sqrt 7 ∨ x = -Complex.I * Real.sqrt 7) :=
by sorry

end find_pure_imaginary_solutions_l1457_145731


namespace lcm_24_36_40_l1457_145717

-- Define the natural numbers 24, 36, and 40
def n1 : ℕ := 24
def n2 : ℕ := 36
def n3 : ℕ := 40

-- Define the prime factorization of each number
def factors_n1 := [2^3, 3^1] -- 24 = 2^3 * 3^1
def factors_n2 := [2^2, 3^2] -- 36 = 2^2 * 3^2
def factors_n3 := [2^3, 5^1] -- 40 = 2^3 * 5^1

-- Prove that the LCM of n1, n2, n3 is 360
theorem lcm_24_36_40 : Nat.lcm (Nat.lcm n1 n2) n3 = 360 := sorry

end lcm_24_36_40_l1457_145717


namespace more_ducks_than_four_times_chickens_l1457_145742

def number_of_chickens (C : ℕ) : Prop :=
  185 = 150 + C

def number_of_ducks (C : ℕ) (MoreDucks : ℕ) : Prop :=
  150 = 4 * C + MoreDucks

theorem more_ducks_than_four_times_chickens (C MoreDucks : ℕ) (h1 : number_of_chickens C) (h2 : number_of_ducks C MoreDucks) : MoreDucks = 10 := by
  sorry

end more_ducks_than_four_times_chickens_l1457_145742


namespace seq_periodic_l1457_145704

def seq (n : ℕ) : ℚ :=
  if n = 1 then 1/4
  else ite (n > 1) (1 - (1 / (seq (n-1)))) 0 -- handle invalid cases with a default zero

theorem seq_periodic {n : ℕ} (h : seq 1 = 1/4) (h2 : ∀ k ≥ 2, seq k = 1 - (1 / (seq (k-1)))) :
  seq 2014 = 1/4 :=
sorry

end seq_periodic_l1457_145704


namespace original_avg_expenditure_correct_l1457_145779

variables (A B C a b c X Y Z : ℝ)
variables (hA : A > 0) (hB : B > 0) (hC : C > 0) (ha : a > 0) (hb : b > 0) (hc : c > 0)

theorem original_avg_expenditure_correct
    (h_orig_exp : (A * X + B * Y + C * Z) / (A + B + C) - 1 
    = ((A + a) * X + (B + b) * Y + (C + c) * Z + 42) / 42):
    True := 
sorry

end original_avg_expenditure_correct_l1457_145779


namespace Hallie_earnings_l1457_145772

theorem Hallie_earnings :
  let w := 10
  let hM := 7
  let tM := 18
  let hT := 5
  let tT := 12
  let hW := 7
  let tW := 20
  let mondayEarnings := hM * w + tM
  let tuesdayEarnings := hT * w + tT
  let wednesdayEarnings := hW * w + tW
  let totalEarnings := mondayEarnings + tuesdayEarnings + wednesdayEarnings
  totalEarnings = 240 := by {
    let w := 10
    let hM := 7
    let tM := 18
    let hT := 5
    let tT := 12
    let hW := 7
    let tW := 20
    let mondayEarnings := hM * w + tM
    let tuesdayEarnings := hT * w + tT
    let wednesdayEarnings := hW * w + tW
    let totalEarnings := mondayEarnings + tuesdayEarnings + wednesdayEarnings
    sorry
  }

end Hallie_earnings_l1457_145772


namespace A_is_false_l1457_145730

variables {a b : ℝ}

-- Condition: Proposition B - The sum of the roots of the equation is 2
axiom sum_of_roots : ∀ (x1 x2 : ℝ), x1 + x2 = -a

-- Condition: Proposition C - x = 3 is a root of the equation
axiom root3 : ∃ (x1 x2 : ℝ), (x1 = 3 ∨ x2 = 3)

-- Condition: Proposition D - The two roots have opposite signs
axiom opposite_sign_roots : ∀ (x1 x2 : ℝ), x1 * x2 < 0

-- Prove: Proposition A is false
theorem A_is_false : ¬ (∃ x1 x2 : ℝ, x1 = 1 ∨ x2 = 1) :=
by
  sorry

end A_is_false_l1457_145730


namespace overall_gain_loss_percent_zero_l1457_145764

theorem overall_gain_loss_percent_zero (CP_A CP_B CP_C SP_A SP_B SP_C : ℝ)
  (h1 : CP_A = 600) (h2 : CP_B = 700) (h3 : CP_C = 800)
  (h4 : SP_A = 450) (h5 : SP_B = 750) (h6 : SP_C = 900) :
  ((SP_A + SP_B + SP_C) - (CP_A + CP_B + CP_C)) / (CP_A + CP_B + CP_C) * 100 = 0 :=
by
  sorry

end overall_gain_loss_percent_zero_l1457_145764


namespace field_area_l1457_145736

theorem field_area
  (L : ℕ) (W : ℕ) (A : ℕ)
  (h₁ : L = 20)
  (h₂ : 2 * W + L = 100)
  (h₃ : A = L * W) :
  A = 800 := by
  sorry

end field_area_l1457_145736


namespace peter_speed_l1457_145715

theorem peter_speed (P : ℝ) (h1 : P >= 0) (h2 : 1.5 * P + 1.5 * (P + 3) = 19.5) : P = 5 := by
  sorry

end peter_speed_l1457_145715
