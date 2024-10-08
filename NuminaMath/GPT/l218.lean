import Mathlib

namespace age_ratio_l218_218287

variable (R D : ℕ)

theorem age_ratio (h1 : D = 24) (h2 : R + 6 = 38) : R / D = 4 / 3 := by
  sorry

end age_ratio_l218_218287


namespace Bill_tossed_objects_l218_218022

theorem Bill_tossed_objects (Ted_sticks Ted_rocks Bill_sticks Bill_rocks : ℕ)
  (h1 : Bill_sticks = Ted_sticks + 6)
  (h2 : Ted_rocks = 2 * Bill_rocks)
  (h3 : Ted_sticks = 10)
  (h4 : Ted_rocks = 10) :
  Bill_sticks + Bill_rocks = 21 :=
by
  sorry

end Bill_tossed_objects_l218_218022


namespace c_work_time_l218_218389

theorem c_work_time (A B C : ℝ) 
  (h1 : A + B = 1/10) 
  (h2 : B + C = 1/5) 
  (h3 : C + A = 1/15) : 
  C = 1/12 :=
by
  -- Proof will go here
  sorry

end c_work_time_l218_218389


namespace number_of_terms_in_arithmetic_sequence_l218_218134

-- Definitions derived directly from the conditions
def first_term : ℕ := 2
def common_difference : ℕ := 4
def last_term : ℕ := 2010

-- Lean statement for the proof problem
theorem number_of_terms_in_arithmetic_sequence :
  ∃ n : ℕ, last_term = first_term + (n - 1) * common_difference ∧ n = 503 :=
by
  sorry

end number_of_terms_in_arithmetic_sequence_l218_218134


namespace first_car_distance_l218_218885

-- Definitions for conditions
variable (x : ℝ) -- distance the first car ran before taking the right turn
def distance_apart_initial := 150 -- initial distance between the cars
def distance_first_car_main_road := 2 * x -- total distance first car ran on the main road
def distance_second_car := 62 -- distance the second car ran due to breakdown
def distance_between_cars := 38 -- distance between the cars after running 

-- Proof (statement only, no solution steps)
theorem first_car_distance (hx : distance_apart_initial = distance_first_car_main_road + distance_second_car + distance_between_cars) : 
  x = 25 :=
by
  unfold distance_apart_initial distance_first_car_main_road distance_second_car distance_between_cars at hx
  -- Implementation placeholder
  sorry

end first_car_distance_l218_218885


namespace geometric_sequence_t_value_l218_218252

theorem geometric_sequence_t_value (S : ℕ → ℝ) (a : ℕ → ℝ) (t : ℝ) :
  (∀ n, S n = t * 5^n - 2) → 
  (∀ n ≥ 1, a (n + 1) = S (n + 1) - S n) → 
  (a 1 ≠ 0) → -- Ensure the sequence is non-trivial.
  (∀ n, a (n + 1) / a n = 5) → 
  t = 5 := 
by 
  intros h1 h2 h3 h4
  sorry

end geometric_sequence_t_value_l218_218252


namespace shirley_eggs_start_l218_218212

theorem shirley_eggs_start (eggs_end : ℕ) (eggs_bought : ℕ) (eggs_start : ℕ) (h_end : eggs_end = 106) (h_bought : eggs_bought = 8) :
  eggs_start = eggs_end - eggs_bought → eggs_start = 98 :=
by
  intros h_start
  rw [h_end, h_bought] at h_start
  exact h_start

end shirley_eggs_start_l218_218212


namespace Connie_savings_l218_218780

theorem Connie_savings (cost_of_watch : ℕ) (extra_needed : ℕ) (saved_amount : ℕ) : 
  cost_of_watch = 55 → 
  extra_needed = 16 → 
  saved_amount = cost_of_watch - extra_needed → 
  saved_amount = 39 := 
by
  intro h1 h2 h3
  rw [h1, h2] at h3
  exact h3


end Connie_savings_l218_218780


namespace blue_tint_percentage_in_new_mixture_l218_218414

-- Define the conditions given in the problem
def original_volume : ℝ := 40
def blue_tint_percentage : ℝ := 0.20
def added_blue_tint_volume : ℝ := 8

-- Calculate the original blue tint volume
def original_blue_tint_volume := blue_tint_percentage * original_volume

-- Calculate the new blue tint volume after adding more blue tint
def new_blue_tint_volume := original_blue_tint_volume + added_blue_tint_volume

-- Calculate the new total volume of the mixture
def new_total_volume := original_volume + added_blue_tint_volume

-- Define the expected result in percentage
def expected_blue_tint_percentage : ℝ := 33.3333

-- Statement to prove
theorem blue_tint_percentage_in_new_mixture :
  (new_blue_tint_volume / new_total_volume) * 100 = expected_blue_tint_percentage :=
sorry

end blue_tint_percentage_in_new_mixture_l218_218414


namespace domain_range_of_p_l218_218111

variable (h : ℝ → ℝ)
variable (h_domain : ∀ x, -1 ≤ x ∧ x ≤ 3)
variable (h_range : ∀ x, 0 ≤ h x ∧ h x ≤ 2)

def p (x : ℝ) : ℝ := 2 - h (x - 1)

theorem domain_range_of_p :
  (∀ x, 0 ≤ x ∧ x ≤ 4) ∧ (∀ y, 0 ≤ y ∧ y ≤ 2) :=
by
  -- Proof to show that the domain of p(x) is [0, 4] and the range is [0, 2]
  sorry

end domain_range_of_p_l218_218111


namespace employee_y_payment_l218_218196

theorem employee_y_payment (X Y : ℝ) (h1 : X + Y = 590) (h2 : X = 1.2 * Y) : Y = 268.18 := by
  sorry

end employee_y_payment_l218_218196


namespace correct_transformation_l218_218529

theorem correct_transformation (a b c : ℝ) (h : c ≠ 0) (h1 : a / c = b / c) : a = b :=
by 
  -- Actual proof would go here, but we use sorry for the scaffold.
  sorry

end correct_transformation_l218_218529


namespace seventh_term_value_l218_218222

open Nat

noncomputable def a : ℤ := sorry
noncomputable def d : ℤ := sorry
noncomputable def n : ℤ := sorry

-- Conditions as definitions
def sum_first_five : Prop := 5 * a + 10 * d = 34
def sum_last_five : Prop := 5 * a + 5 * (n - 1) * d = 146
def sum_all_terms : Prop := (n * (2 * a + (n - 1) * d)) / 2 = 234

-- Theorem statement
theorem seventh_term_value :
  sum_first_five ∧ sum_last_five ∧ sum_all_terms → a + 6 * d = 18 :=
by
  sorry

end seventh_term_value_l218_218222


namespace find_a_value_l218_218918

def line1 (a : ℝ) (x y : ℝ) : ℝ := a * x + (a + 2) * y + 1
def line2 (a : ℝ) (x y : ℝ) : ℝ := a * x - y + 2

-- Define what it means for two lines to be not parallel
def not_parallel (a : ℝ) : Prop :=
  ∀ x y : ℝ, (line1 a x y ≠ 0 ∧ line2 a x y ≠ 0)

theorem find_a_value (a : ℝ) (h : not_parallel a) : a = 0 ∨ a = -3 :=
  sorry

end find_a_value_l218_218918


namespace value_of_m_l218_218258

theorem value_of_m (m : ℝ) : (∀ x : ℝ, (x^2 + 2 * m * x + m > 3 / 16)) ↔ (1 / 4 < m ∧ m < 3 / 4) :=
by sorry

end value_of_m_l218_218258


namespace binar_operation_correct_l218_218688

theorem binar_operation_correct : 
  let a := 13  -- 1101_2 in decimal
  let b := 15  -- 1111_2 in decimal
  let c := 9   -- 1001_2 in decimal
  let d := 2   -- 10_2 in decimal
  a + b - c * d = 10 ↔ "1010" = "1010" := 
by 
  intros
  simp
  sorry

end binar_operation_correct_l218_218688


namespace find_b_plus_c_l218_218733

-- Definitions based on the given conditions.
variables {A : ℝ} {a b c : ℝ}

-- The conditions in the problem
theorem find_b_plus_c
  (h_cosA : Real.cos A = 1 / 3)
  (h_a : a = Real.sqrt 3)
  (h_bc : b * c = 3 / 2) :
  b + c = Real.sqrt 7 :=
sorry

end find_b_plus_c_l218_218733


namespace calculate_radius_l218_218158

noncomputable def radius_of_wheel (D : ℝ) (N : ℕ) (π : ℝ) : ℝ :=
  D / (2 * π * N)

theorem calculate_radius : 
  radius_of_wheel 4224 3000 Real.pi = 0.224 :=
by
  sorry

end calculate_radius_l218_218158


namespace vacation_costs_l218_218604

variable (Anne_paid Beth_paid Carlos_paid : ℕ) (a b : ℕ)

theorem vacation_costs (hAnne : Anne_paid = 120) (hBeth : Beth_paid = 180) (hCarlos : Carlos_paid = 150)
  (h_a : a = 30) (h_b : b = 30) :
  a - b = 0 := sorry

end vacation_costs_l218_218604


namespace expected_adjacent_black_l218_218696

noncomputable def ExpectedBlackPairs :=
  let totalCards := 104
  let blackCards := 52
  let totalPairs := 103
  let probAdjacentBlack := (blackCards - 1) / (totalPairs)
  blackCards * probAdjacentBlack

theorem expected_adjacent_black :
  ExpectedBlackPairs = 2601 / 103 :=
by
  sorry

end expected_adjacent_black_l218_218696


namespace Bill_initial_money_l218_218309

theorem Bill_initial_money (joint_money : ℕ) (pizza_cost : ℕ) (num_pizzas : ℕ) (final_bill_amount : ℕ) (initial_joint_money_eq : joint_money = 42) (pizza_cost_eq : pizza_cost = 11) (num_pizzas_eq : num_pizzas = 3) (final_bill_amount_eq : final_bill_amount = 39) :
  ∃ b : ℕ, b = 30 :=
by
  sorry

end Bill_initial_money_l218_218309


namespace susie_initial_amount_l218_218708

-- Definitions for conditions:
def initial_amount (X : ℝ) : Prop :=
  X + 0.20 * X = 240

-- Main theorem to prove:
theorem susie_initial_amount (X : ℝ) (h : initial_amount X) : X = 200 :=
by 
  -- structured proof will go here
  sorry

end susie_initial_amount_l218_218708


namespace functions_are_even_l218_218062

noncomputable def f_A (x : ℝ) : ℝ := -|x| + 2
noncomputable def f_B (x : ℝ) : ℝ := x^2 - 3
noncomputable def f_C (x : ℝ) : ℝ := Real.sqrt (1 - x^2)

theorem functions_are_even :
  (∀ x : ℝ, f_A x = f_A (-x)) ∧
  (∀ x : ℝ, f_B x = f_B (-x)) ∧
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f_C x = f_C (-x)) :=
by
  sorry

end functions_are_even_l218_218062


namespace domain_of_function_l218_218615

theorem domain_of_function :
  {x : ℝ | 2 - x ≥ 0} = {x : ℝ | x ≤ 2} :=
by
  sorry

end domain_of_function_l218_218615


namespace find_x_l218_218695

noncomputable def x : ℝ := 20

def condition1 (x : ℝ) : Prop := x > 0
def condition2 (x : ℝ) : Prop := x / 100 * 150 - 20 = 10

theorem find_x (x : ℝ) : condition1 x ∧ condition2 x ↔ x = 20 :=
by
  sorry

end find_x_l218_218695


namespace solution_of_fractional_inequality_l218_218010

noncomputable def solution_set_of_inequality : Set ℝ :=
  {x : ℝ | -3 < x ∨ x > 1/2 }

theorem solution_of_fractional_inequality :
  {x : ℝ | (2 * x - 1) / (x + 3) > 0} = solution_set_of_inequality :=
by
  sorry

end solution_of_fractional_inequality_l218_218010


namespace xyz_distinct_real_squares_l218_218995

theorem xyz_distinct_real_squares (x y z : ℝ) 
  (h1 : x^2 = 2 + y)
  (h2 : y^2 = 2 + z)
  (h3 : z^2 = 2 + x) 
  (h4 : x ≠ y ∧ y ≠ z ∧ z ≠ x) :
  x^2 + y^2 + z^2 = 5 ∨ x^2 + y^2 + z^2 = 6 ∨ x^2 + y^2 + z^2 = 9 :=
by 
  sorry

end xyz_distinct_real_squares_l218_218995


namespace find_m_l218_218931

theorem find_m (x y m : ℤ) (h1 : x = 1) (h2 : y = -1) (h3 : 2 * x + m + y = 0) : m = -1 := by
  -- Proof can be completed here
  sorry

end find_m_l218_218931


namespace proof_problem_l218_218327

-- Define the conditions for the problem

def is_factor (a b : ℕ) : Prop :=
  ∃ n : ℕ, b = a * n

def is_divisor (a b : ℕ) : Prop :=
  b % a = 0

-- Statement that needs to be proven
theorem proof_problem :
  is_factor 5 65 ∧ ¬(is_divisor 19 361 ∧ ¬is_divisor 19 190) ∧ ¬(¬is_divisor 36 144 ∨ ¬is_divisor 36 73) ∧ ¬(is_divisor 14 28 ∧ ¬is_divisor 14 56) ∧ is_factor 9 144 :=
by sorry

end proof_problem_l218_218327


namespace intersection_of_A_and_B_l218_218035

def setA : Set ℝ := { x | x^2 - 3 * x - 4 < 0 }
def setB : Set ℝ := {-4, 1, 3, 5}
def resultSet : Set ℝ := {1, 3}

theorem intersection_of_A_and_B :
  setA ∩ setB = resultSet := 
by
  sorry

end intersection_of_A_and_B_l218_218035


namespace cannot_determine_exact_insect_l218_218591

-- Defining the conditions as premises
def insect_legs : ℕ := 6

def total_legs_two_insects (legs_per_insect : ℕ) (num_insects : ℕ) : ℕ :=
  legs_per_insect * num_insects

-- Statement: Proving that given just the number of legs, we cannot determine the exact type of insect
theorem cannot_determine_exact_insect (legs : ℕ) (num_insects : ℕ) (h1 : legs = 6) (h2 : num_insects = 2) (h3 : total_legs_two_insects legs num_insects = 12) :
  ∃ insect_type, insect_type :=
by
  sorry

end cannot_determine_exact_insect_l218_218591


namespace cylinder_in_sphere_volume_difference_is_correct_l218_218489

noncomputable def volume_difference (base_radius_cylinder : ℝ) (radius_sphere : ℝ) : ℝ :=
  let height_cylinder := Real.sqrt (radius_sphere^2 - base_radius_cylinder^2)
  let volume_sphere := (4 / 3) * Real.pi * radius_sphere^3
  let volume_cylinder := Real.pi * base_radius_cylinder^2 * height_cylinder
  volume_sphere - volume_cylinder

theorem cylinder_in_sphere_volume_difference_is_correct :
  volume_difference 4 7 = (1372 - 48 * Real.sqrt 33) / 3 * Real.pi :=
by
  sorry

end cylinder_in_sphere_volume_difference_is_correct_l218_218489


namespace rotated_line_l1_l218_218634

-- Define the original line equation and the point around which the line is rotated
def line_l (x y : ℝ) : Prop := x - y + 1 = 0
def point_A : ℝ × ℝ := (2, 3)

-- Define the line equation that needs to be proven
def line_l1 (x y : ℝ) : Prop := x + y - 5 = 0

-- The theorem stating that after a 90-degree rotation of line l around point A, the new line is equation l1
theorem rotated_line_l1 : 
  ∀ (x y : ℝ), 
  (∃ (k : ℝ), k = 1 ∧ ∀ (x y), line_l x y ∧ ∀ (x y), line_l1 x y) ∧ 
  ∀ (a b : ℝ), (a, b) = point_A → 
  x + y - 5 = 0 := 
by
  sorry

end rotated_line_l1_l218_218634


namespace total_distance_after_fourth_bounce_l218_218501

noncomputable def total_distance_traveled (initial_height : ℝ) (bounce_ratio : ℝ) (num_bounces : ℕ) : ℝ :=
  let fall_distances := (List.range (num_bounces + 1)).map (λ n => initial_height * bounce_ratio^n)
  let rise_distances := (List.range num_bounces).map (λ n => initial_height * bounce_ratio^(n+1))
  fall_distances.sum + rise_distances.sum

theorem total_distance_after_fourth_bounce :
  total_distance_traveled 25 (5/6 : ℝ) 4 = 154.42 :=
by
  sorry

end total_distance_after_fourth_bounce_l218_218501


namespace total_seats_in_theater_l218_218588

theorem total_seats_in_theater 
    (n : ℕ) 
    (a1 : ℕ)
    (an : ℕ)
    (d : ℕ)
    (h1 : a1 = 12)
    (h2 : d = 2)
    (h3 : an = 48)
    (h4 : an = a1 + (n - 1) * d) :
    (n = 19) →
    (2 * (a1 + an) * n / 2 = 570) :=
by
  intros
  sorry

end total_seats_in_theater_l218_218588


namespace intersection_of_lines_l218_218723

theorem intersection_of_lines
    (x y : ℚ) 
    (h1 : y = 3 * x - 1)
    (h2 : y + 4 = -6 * x) :
    x = -1 / 3 ∧ y = -2 := 
sorry

end intersection_of_lines_l218_218723


namespace find_a_value_l218_218670

theorem find_a_value 
  (A : Set ℤ := {-1, 0, 1})
  (a : ℤ) 
  (B : Set ℤ := {a, a^2}) 
  (h_union : A ∪ B = A) : 
  a = -1 :=
sorry

end find_a_value_l218_218670


namespace Julie_work_hours_per_week_l218_218714

variable (hours_per_week_summer : ℕ) (weeks_summer : ℕ)
variable (earnings_summer : ℕ)
variable (weeks_school_year : ℕ)
variable (earnings_school_year : ℕ)

theorem Julie_work_hours_per_week :
  hours_per_week_summer = 40 →
  weeks_summer = 10 →
  earnings_summer = 4000 →
  weeks_school_year = 40 →
  earnings_school_year = 4000 →
  (∀ rate_per_hour, rate_per_hour = earnings_summer / (hours_per_week_summer * weeks_summer) →
  (earnings_school_year / (weeks_school_year * rate_per_hour) = 10)) :=
by intros h1 h2 h3 h4 h5 rate_per_hour hr; sorry

end Julie_work_hours_per_week_l218_218714


namespace part_I_part_II_l218_218363

noncomputable def f (a x : ℝ) : ℝ := |a * x - 1| + |x + 2|

theorem part_I (h₁ : ∀ x : ℝ, f 1 x ≤ 5 ↔ -3 ≤ x ∧ x ≤ 2) : True :=
by sorry

theorem part_II (h₂ : ∃ a : ℝ, a > 0 ∧ (∀ x, f a x ≥ 2) ∧ (∀ b : ℝ, b > 0 ∧ (∀ x, f b x ≥ 2) → a ≤ b) ) : True :=
by sorry

end part_I_part_II_l218_218363


namespace find_side_length_l218_218031

theorem find_side_length (a b c : ℝ) (A : ℝ) 
  (h1 : Real.cos A = 7 / 8) 
  (h2 : c - a = 2) 
  (h3 : b = 3) : 
  a = 2 := by
  sorry

end find_side_length_l218_218031


namespace sum_of_interior_angles_l218_218702

theorem sum_of_interior_angles (n : ℕ) (interior_angle : ℝ) :
  (interior_angle = 144) → (180 - 144) * n = 360 → n = 10 → (n - 2) * 180 = 1440 :=
by
  intros h1 h2 h3
  sorry

end sum_of_interior_angles_l218_218702


namespace circles_and_squares_intersection_l218_218017

def circles_and_squares_intersection_count : Nat :=
  let radius := (1 : ℚ) / 8
  let square_side := (1 : ℚ) / 4
  let slope := (1 : ℚ) / 3
  let line (x : ℚ) : ℚ := slope * x
  let num_segments := 243
  let intersections_per_segment := 4
  num_segments * intersections_per_segment

theorem circles_and_squares_intersection : 
  circles_and_squares_intersection_count = 972 :=
by
  sorry

end circles_and_squares_intersection_l218_218017


namespace radius_of_circle_proof_l218_218968

noncomputable def radius_of_circle (x y : ℝ) (h1 : x = Real.pi * r ^ 2) (h2 : y = 2 * Real.pi * r) (h3 : x + y = 100 * Real.pi) : ℝ :=
  r

theorem radius_of_circle_proof (r x y : ℝ) (h1 : x = Real.pi * r ^ 2) (h2 : y = 2 * Real.pi * r) (h3 : x + y = 100 * Real.pi) : r = 10 :=
by
  sorry

end radius_of_circle_proof_l218_218968


namespace stratified_sampling_correct_l218_218437

-- Definitions based on the conditions
def total_employees : ℕ := 300
def over_40 : ℕ := 50
def between_30_and_40 : ℕ := 150
def under_30 : ℕ := 100
def sample_size : ℕ := 30
def stratified_ratio : ℕ := 1 / 10  -- sample_size / total_employees

-- Function to compute the number of individuals sampled from each age group
def sampled_from_age_group (group_size : ℕ) : ℕ :=
  group_size * stratified_ratio

-- Mathematical properties to be proved
theorem stratified_sampling_correct :
  sampled_from_age_group over_40 = 5 ∧ 
  sampled_from_age_group between_30_and_40 = 15 ∧ 
  sampled_from_age_group under_30 = 10 := by
  sorry

end stratified_sampling_correct_l218_218437


namespace sqrt_nested_l218_218247

theorem sqrt_nested (x : ℝ) (hx : 0 ≤ x) : Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x))) = x ^ (15 / 16) := by
  sorry

end sqrt_nested_l218_218247


namespace store_revenue_after_sale_l218_218558

/--
A store has 2000 items, each normally selling for $50. 
They offer an 80% discount and manage to sell 90% of the items. 
The store owes $15,000 to creditors. Prove that the store has $3,000 left after the sale.
-/
theorem store_revenue_after_sale :
  let items := 2000
  let retail_price := 50
  let discount := 0.8
  let sale_percentage := 0.9
  let debt := 15000
  let items_sold := items * sale_percentage
  let discount_amount := retail_price * discount
  let sale_price_per_item := retail_price - discount_amount
  let total_revenue := items_sold * sale_price_per_item
  let money_left := total_revenue - debt
  money_left = 3000 :=
by
  sorry

end store_revenue_after_sale_l218_218558


namespace contractor_laborers_l218_218168

theorem contractor_laborers (x : ℕ) (h : 9 * x = 15 * (x - 6)) : x = 15 :=
by
  sorry

end contractor_laborers_l218_218168


namespace inequality_preservation_l218_218386

theorem inequality_preservation (a b : ℝ) (h : a > b) : (1/3 : ℝ) * a - 1 > (1/3 : ℝ) * b - 1 := 
by sorry

end inequality_preservation_l218_218386


namespace rectangular_solid_edges_sum_l218_218409

theorem rectangular_solid_edges_sum
  (b s : ℝ)
  (h_vol : (b / s) * b * (b * s) = 432)
  (h_sa : 2 * ((b ^ 2 / s) + b ^ 2 * s + b ^ 2) = 432)
  (h_gp : 0 < s ∧ s ≠ 1) :
  4 * (b / s + b + b * s) = 144 := 
by
  sorry

end rectangular_solid_edges_sum_l218_218409


namespace carter_cheesecakes_l218_218094

theorem carter_cheesecakes (C : ℕ) (nm : ℕ) (nr : ℕ) (increase : ℕ) (this_week_cakes : ℕ) (usual_cakes : ℕ) :
  nm = 5 → nr = 8 → increase = 38 → 
  this_week_cakes = 3 * C + 3 * nm + 3 * nr → 
  usual_cakes = C + nm + nr → 
  this_week_cakes = usual_cakes + increase → 
  C = 6 :=
by
  intros hnm hnr hinc htw husual hcakes
  sorry

end carter_cheesecakes_l218_218094


namespace adults_on_field_trip_l218_218261

-- Define the conditions
def van_capacity : ℕ := 7
def num_students : ℕ := 33
def num_vans : ℕ := 6

-- Define the total number of people that can be transported given the number of vans and capacity per van
def total_people : ℕ := num_vans * van_capacity

-- The number of people that can be transported minus the number of students gives the number of adults
def num_adults : ℕ := total_people - num_students

-- Theorem to prove the number of adults is 9
theorem adults_on_field_trip : num_adults = 9 :=
by
  -- Skipping the proof
  sorry

end adults_on_field_trip_l218_218261


namespace op_4_neg3_eq_neg28_l218_218782

def op (x y : Int) : Int := x * (y + 2) + 2 * x * y

theorem op_4_neg3_eq_neg28 : op 4 (-3) = -28 := by
  sorry

end op_4_neg3_eq_neg28_l218_218782


namespace cost_of_running_tv_for_week_l218_218564

def powerUsage : ℕ := 125
def hoursPerDay : ℕ := 4
def costPerkWh : ℕ := 14

theorem cost_of_running_tv_for_week :
  let dailyConsumption := powerUsage * hoursPerDay
  let dailyConsumptionkWh := dailyConsumption / 1000
  let weeklyConsumption := dailyConsumptionkWh * 7
  let weeklyCost := weeklyConsumption * costPerkWh
  weeklyCost = 49 := by
  let dailyConsumption := powerUsage * hoursPerDay
  let dailyConsumptionkWh := dailyConsumption / 1000
  let weeklyConsumption := dailyConsumptionkWh * 7
  let weeklyCost := weeklyConsumption * costPerkWh
  sorry

end cost_of_running_tv_for_week_l218_218564


namespace wholesale_cost_per_bag_l218_218483

theorem wholesale_cost_per_bag (W : ℝ) (h1 : 1.12 * W = 28) : W = 25 :=
sorry

end wholesale_cost_per_bag_l218_218483


namespace eval_expression_l218_218790

theorem eval_expression : 9^9 * 3^3 / 3^30 = 1 / 19683 := by
  sorry

end eval_expression_l218_218790


namespace doug_lost_marbles_l218_218082

-- Definitions based on the conditions
variables (D D' : ℕ) -- D is the number of marbles Doug originally had, D' is the number Doug has now

-- Condition 1: Ed had 10 more marbles than Doug originally.
def ed_marble_initial (D : ℕ) : ℕ := D + 10

-- Condition 2: Ed had 45 marbles originally.
axiom ed_initial_marble_count : ed_marble_initial D = 45

-- Solve for D from condition 2
noncomputable def doug_initial_marble_count : ℕ := 45 - 10

-- Condition 3: Ed now has 21 more marbles than Doug.
axiom ed_current_marble_difference : 45 = D' + 21

-- Translate what we need to prove
theorem doug_lost_marbles : (doug_initial_marble_count - D') = 11 :=
by
    -- Insert math proof steps here
    sorry

end doug_lost_marbles_l218_218082


namespace inverse_of_B_cubed_l218_218707

variable (B : Matrix (Fin 2) (Fin 2) ℝ)
def B_inv := Matrix.of ![![3, -2], ![0, -1]]
noncomputable def B_cubed_inv := ((B_inv) 3)^3

theorem inverse_of_B_cubed :
  B_inv = Matrix.of ![![27, -24], ![0, -1]] :=
by
  sorry

end inverse_of_B_cubed_l218_218707


namespace number_of_authors_l218_218875

/-- Define the number of books each author has and the total number of books. -/
def books_per_author : ℕ := 33
def total_books : ℕ := 198

/-- Main theorem stating that the number of authors Jack has is derived by dividing total books by the number of books per author. -/
theorem number_of_authors (n : ℕ) (h : total_books = n * books_per_author) : n = 6 := by
  sorry

end number_of_authors_l218_218875


namespace find_x_l218_218122

theorem find_x (x : ℝ) (a : ℝ × ℝ := (1, 2)) (b : ℝ × ℝ := (x, 1)) :
  ((2 * a.fst - x, 2 * a.snd + 1) • b = 0) → x = -1 ∨ x = 3 :=
by
  sorry

end find_x_l218_218122


namespace original_faculty_number_l218_218150

theorem original_faculty_number (x : ℝ) (h : 0.85 * x = 195) : x = 229 := by
  sorry

end original_faculty_number_l218_218150


namespace algebraic_expression_evaluation_l218_218774

theorem algebraic_expression_evaluation (a b : ℝ) (h₁ : a ≠ b) 
  (h₂ : a^2 - 8 * a + 5 = 0) (h₃ : b^2 - 8 * b + 5 = 0) :
  (b - 1) / (a - 1) + (a - 1) / (b - 1) = -20 :=
by
  sorry

end algebraic_expression_evaluation_l218_218774


namespace range_of_m_three_zeros_l218_218203

noncomputable def f (x m : ℝ) : ℝ :=
if h : x < 0 then -x + m else x^2 - 1

theorem range_of_m_three_zeros (h : 0 < m) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f (f x1 m) m - 1 = 0 ∧ f (f x2 m) m - 1 = 0 ∧ f (f x3 m) m - 1 = 0) ↔ (0 < m ∧ m < 1) :=
by
  sorry

end range_of_m_three_zeros_l218_218203


namespace line_intersects_circle_l218_218279

theorem line_intersects_circle 
  (radius : ℝ) 
  (distance_center_line : ℝ) 
  (h_radius : radius = 4) 
  (h_distance : distance_center_line = 3) : 
  radius > distance_center_line := 
by 
  sorry

end line_intersects_circle_l218_218279


namespace number_of_perfect_cubes_l218_218890

theorem number_of_perfect_cubes (n : ℤ) : 
  (∃ (count : ℤ), (∀ (x : ℤ), (100 < x^3 ∧ x^3 < 400) ↔ x = 5 ∨ x = 6 ∨ x = 7) ∧ (count = 3)) := 
sorry

end number_of_perfect_cubes_l218_218890


namespace identify_urea_decomposing_bacteria_l218_218568

-- Definitions of different methods
def methodA (culture_medium : Type) := 
  ∃ (urea_only : culture_medium) (phenol_red : culture_medium), phenol_red = urea_only

def methodB (culture_medium : Type) := 
  ∃ (urea_only : culture_medium) (EMB_reagent : culture_medium), EMB_reagent = urea_only

def methodC (culture_medium : Type) := 
  ∃ (urea_only : culture_medium) (Sudan_III : culture_medium), Sudan_III = urea_only

def methodD (culture_medium : Type) := 
  ∃ (urea_only : culture_medium) (Biuret_reagent : culture_medium), Biuret_reagent = urea_only

-- The proof problem statement
theorem identify_urea_decomposing_bacteria (culture_medium : Type) :
  methodA culture_medium :=
sorry

end identify_urea_decomposing_bacteria_l218_218568


namespace sum_of_fractions_l218_218990

theorem sum_of_fractions : (3 / 20 : ℝ) + (5 / 50 : ℝ) + (7 / 2000 : ℝ) = 0.2535 :=
by sorry

end sum_of_fractions_l218_218990


namespace PQ_relationship_l218_218028

-- Define the sets P and Q
def P := {x : ℝ | x >= 5}
def Q := {x : ℝ | 5 <= x ∧ x <= 7}

-- Statement to be proved
theorem PQ_relationship : Q ⊆ P ∧ Q ≠ P :=
by
  sorry

end PQ_relationship_l218_218028


namespace distribution_problem_distribution_problem_variable_distribution_problem_equal_l218_218785

def books_distribution_fixed (n : ℕ) (a b c : ℕ) : ℕ :=
  if h : a + b + c = n then n.factorial / (a.factorial * b.factorial * c.factorial) else 0

theorem distribution_problem (n a b c : ℕ) (h : a + b + c = n) : 
  books_distribution_fixed n a b c = 1260 :=
sorry

def books_distribution_variable (n : ℕ) (a b c : ℕ) : ℕ :=
  if h : a + b + c = n then (n.factorial / (a.factorial * b.factorial * c.factorial)) * 6 else 0

theorem distribution_problem_variable (n a b c : ℕ) (h : a + b + c = n) : 
  books_distribution_variable n a b c = 7560 :=
sorry

def books_distribution_equal (n : ℕ) (k : ℕ) : ℕ :=
  if h : 3 * k = n then n.factorial / (k.factorial * k.factorial * k.factorial) else 0

theorem distribution_problem_equal (n k : ℕ) (h : 3 * k = n) : 
  books_distribution_equal n k = 1680 :=
sorry

end distribution_problem_distribution_problem_variable_distribution_problem_equal_l218_218785


namespace complete_square_l218_218433

theorem complete_square {x : ℝ} :
  x^2 - 6 * x - 8 = 0 ↔ (x - 3)^2 = 17 :=
sorry

end complete_square_l218_218433


namespace solution_set_of_inequality_l218_218798

variable (f : ℝ → ℝ)

theorem solution_set_of_inequality :
  (∀ x, f (x) = f (-x)) →               -- f(x) is even
  (∀ x y, 0 < x → x < y → f y ≤ f x) →   -- f(x) is monotonically decreasing on (0, +∞)
  f 2 = 0 →                              -- f(2) = 0
  {x : ℝ | (f x + f (-x)) / (3 * x) < 0} = 
    {x : ℝ | -2 < x ∧ x < 0} ∪ {x : ℝ | 2 < x} :=
by sorry

end solution_set_of_inequality_l218_218798


namespace capsule_cost_difference_l218_218824

theorem capsule_cost_difference :
  let cost_per_capsule_r := 6.25 / 250
  let cost_per_capsule_t := 3.00 / 100
  cost_per_capsule_t - cost_per_capsule_r = 0.005 := by
  sorry

end capsule_cost_difference_l218_218824


namespace eval_five_over_two_l218_218916

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^x - 2 else Real.log (x - 1) / Real.log 2

theorem eval_five_over_two : f (5 / 2) = -1 := by
  sorry

end eval_five_over_two_l218_218916


namespace number_of_girls_in_school_l218_218336

theorem number_of_girls_in_school
  (total_students : ℕ)
  (avg_age_boys avg_age_girls avg_age_school : ℝ)
  (B G : ℕ)
  (h1 : total_students = 640)
  (h2 : avg_age_boys = 12)
  (h3 : avg_age_girls = 11)
  (h4 : avg_age_school = 11.75)
  (h5 : B + G = total_students)
  (h6 : (avg_age_boys * B + avg_age_girls * G = avg_age_school * total_students)) :
  G = 160 :=
by
  sorry

end number_of_girls_in_school_l218_218336


namespace proj_v_w_l218_218178

noncomputable def proj (v w : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2
  let w_dot_w := dot_product w w
  let v_dot_w := dot_product v w
  let scalar := v_dot_w / w_dot_w
  (scalar * w.1, scalar * w.2)

theorem proj_v_w :
  let v := (4, -3)
  let w := (12, 5)
  proj v w = (396 / 169, 165 / 169) :=
by
  sorry

end proj_v_w_l218_218178


namespace sum_divisibility_l218_218416

theorem sum_divisibility (a b : ℤ) (h : 6 * a + 11 * b ≡ 0 [ZMOD 31]) : a + 7 * b ≡ 0 [ZMOD 31] :=
sorry

end sum_divisibility_l218_218416


namespace customers_left_tip_l218_218301

-- Definition of the given conditions
def initial_customers : ℕ := 29
def added_customers : ℕ := 20
def customers_didnt_tip : ℕ := 34

-- Lean 4 statement proving that the number of customers who did leave a tip (answer) equals 15
theorem customers_left_tip : (initial_customers + added_customers - customers_didnt_tip) = 15 :=
by
  sorry

end customers_left_tip_l218_218301


namespace insects_legs_l218_218255

theorem insects_legs (L N : ℕ) (hL : L = 54) (hN : N = 9) : (L / N = 6) :=
by sorry

end insects_legs_l218_218255


namespace abhay_speed_l218_218443

variables (A S : ℝ)

theorem abhay_speed (h1 : 24 / A = 24 / S + 2) (h2 : 24 / (2 * A) = 24 / S - 1) : A = 12 :=
by {
  sorry
}

end abhay_speed_l218_218443


namespace fraction_of_shaded_area_l218_218933

theorem fraction_of_shaded_area
  (total_smaller_rectangles : ℕ)
  (shaded_smaller_rectangles : ℕ)
  (h1 : total_smaller_rectangles = 18)
  (h2 : shaded_smaller_rectangles = 4) :
  (shaded_smaller_rectangles : ℚ) / total_smaller_rectangles = 1 / 4 := 
sorry

end fraction_of_shaded_area_l218_218933


namespace solve_system_l218_218058

variables (a b c d : ℝ)

theorem solve_system :
  (a + c = -4) ∧
  (a * c + b + d = 6) ∧
  (a * d + b * c = -5) ∧
  (b * d = 2) →
  ((a = -3 ∧ b = 2 ∧ c = -1 ∧ d = 1) ∨
   (a = -1 ∧ b = 1 ∧ c = -3 ∧ d = 2)) :=
by
  intro h
  -- Insert proof here
  sorry

end solve_system_l218_218058


namespace shorter_piece_length_l218_218945

def wireLength := 150
def ratioLongerToShorter := 5 / 8

theorem shorter_piece_length : ∃ x : ℤ, x + (5 / 8) * x = wireLength ∧ x = 92 := by
  sorry

end shorter_piece_length_l218_218945


namespace cone_volume_l218_218163

theorem cone_volume (r l: ℝ) (h: ℝ) (hr : r = 1) (hl : l = 2) (hh : h = Real.sqrt (l^2 - r^2)) : 
  (1 / 3) * Real.pi * r^2 * h = (Real.sqrt 3 * Real.pi) / 3 :=
by 
  sorry

end cone_volume_l218_218163


namespace map_area_ratio_l218_218777

theorem map_area_ratio (l w : ℝ) (hl : l > 0) (hw : w > 0) :
  ¬ ((l * w) / ((500 * l) * (500 * w)) = 1 / 500) :=
by
  -- The proof will involve calculations showing the true ratio is 1/250000
  sorry

end map_area_ratio_l218_218777


namespace approximate_number_of_fish_in_pond_l218_218184

theorem approximate_number_of_fish_in_pond :
  ∃ N : ℕ, N = 800 ∧
  (40 : ℕ) / N = (2 : ℕ) / (40 : ℕ) := 
sorry

end approximate_number_of_fish_in_pond_l218_218184


namespace conditions_for_k_b_l218_218281

theorem conditions_for_k_b (k b : ℝ) :
  (∀ x : ℝ, (x - (kx + b) + 2) * (2) > 0) →
  (k = 1) ∧ (b < 2) :=
by
  intros h
  sorry

end conditions_for_k_b_l218_218281


namespace cos_pi_over_2_plus_2theta_l218_218554

theorem cos_pi_over_2_plus_2theta (θ : ℝ) (hcos : Real.cos θ = 1 / 3) (hθ : 0 < θ ∧ θ < Real.pi) :
    Real.cos (Real.pi / 2 + 2 * θ) = - (4 * Real.sqrt 2) / 9 := 
sorry

end cos_pi_over_2_plus_2theta_l218_218554


namespace simplify_exponent_multiplication_l218_218639

theorem simplify_exponent_multiplication (x : ℝ) : x^5 * x^3 = x^8 :=
by sorry

end simplify_exponent_multiplication_l218_218639


namespace determine_a_l218_218495

theorem determine_a 
  (a : ℝ)
  (h : ∀ x : ℝ, |a * x - 2| < 3 ↔ - 5 / 3 < x ∧ x < 1 / 3) : 
  a = -3 := by 
  sorry

end determine_a_l218_218495


namespace solution_set_f_ge_0_l218_218324

noncomputable def f (x a : ℝ) : ℝ := 1 / Real.exp x - a / x

theorem solution_set_f_ge_0 (a m n : ℝ) (h : ∀ x, m ≤ x ∧ x ≤ n ↔ 1 / Real.exp x - a / x ≥ 0) : 
  0 < a ∧ a < 1 / Real.exp 1 :=
  sorry

end solution_set_f_ge_0_l218_218324


namespace length_of_platform_is_correct_l218_218421

noncomputable def length_of_platform : ℝ :=
  let train_length := 200 -- in meters
  let train_speed := 80 * 1000 / 3600 -- kmph to m/s
  let crossing_time := 22 -- in seconds
  (train_speed * crossing_time) - train_length

theorem length_of_platform_is_correct :
  length_of_platform = 2600 / 9 :=
by 
  -- proof would go here
  sorry

end length_of_platform_is_correct_l218_218421


namespace b_n_expression_l218_218792

-- Define sequence a_n as an arithmetic sequence with given conditions
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ a1 d, ∀ n, a n = a1 + d * (n - 1)

-- Define the conditions for the sequence a_n
def a_conditions (a : ℕ → ℤ) : Prop :=
  a 2 = 8 ∧ a 8 = 26

-- Define the new sequence b_n based on the terms of a_n
def b (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  a (3^n)

theorem b_n_expression (a : ℕ → ℤ) (n : ℕ)
  (h_arith : is_arithmetic_sequence a)
  (h_conditions : a_conditions a) :
  b a n = 3^(n + 1) + 2 := 
sorry

end b_n_expression_l218_218792


namespace problem_1_problem_2_problem_3_l218_218557

variable (α : ℝ)
variable (tan_alpha_two : Real.tan α = 2)

theorem problem_1 : (2 * Real.cos α + 3 * Real.sin α) / (3 * Real.cos α + Real.sin α) = 8 / 5 :=
by
  sorry

theorem problem_2 : (Real.cos α ^ 2 + Real.sin α * Real.cos α) / (2 * Real.sin α * Real.cos α + Real.sin α ^ 2) = 3 / 8 :=
by
  sorry

theorem problem_3 : (Real.sin α ^ 2 - Real.sin α * Real.cos α + 2) = 12 / 5 :=
by
  sorry

end problem_1_problem_2_problem_3_l218_218557


namespace polynomial_remainder_l218_218494

theorem polynomial_remainder (x : ℤ) : 
  (2 * x + 3) ^ 504 % (x^2 - x + 1) = (16 * x + 5) :=
by
  sorry

end polynomial_remainder_l218_218494


namespace min_third_side_triangle_l218_218640

theorem min_third_side_triangle (a b c : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
    (h_distinct_1 : 42 * a ≠ 72 * b) (h_distinct_2 : 42 * a ≠ c) (h_distinct_3 : 72 * b ≠ c) :
    (42 * a + 72 * b > c) ∧ (42 * a + c > 72 * b) ∧ (72 * b + c > 42 * a) → c ≥ 7 :=
sorry

end min_third_side_triangle_l218_218640


namespace quadratic_single_root_a_l218_218311

theorem quadratic_single_root_a (a : ℝ) : 
  (∃! x : ℝ, a * x^2 + 2 * x + 1 = 0) → (a = 0 ∨ a = 1) :=
by
  sorry

end quadratic_single_root_a_l218_218311


namespace fido_leash_yard_reach_area_product_l218_218807

noncomputable def fido_leash_yard_fraction : ℝ :=
  let a := 2 + Real.sqrt 2
  let b := 8
  a * b

theorem fido_leash_yard_reach_area_product :
  ∃ (a b : ℝ), 
  (fido_leash_yard_fraction = (a * b)) ∧ 
  (1 > a) ∧ -- Regular Octagon computation constraints
  (b = 8) ∧ 
  a = 2 + Real.sqrt 2 :=
sorry

end fido_leash_yard_reach_area_product_l218_218807


namespace consecutive_even_product_l218_218434

-- Define that there exist three consecutive even numbers such that the product equals 87526608.
theorem consecutive_even_product (a : ℤ) : 
  (a - 2) * a * (a + 2) = 87526608 → ∃ b : ℤ, b = a - 2 ∧ b % 2 = 0 ∧ ∃ c : ℤ, c = a ∧ c % 2 = 0 ∧ ∃ d : ℤ, d = a + 2 ∧ d % 2 = 0 :=
sorry

end consecutive_even_product_l218_218434


namespace apples_ratio_l218_218329

theorem apples_ratio (initial_apples rickis_apples end_apples samsons_apples : ℕ)
(h_initial : initial_apples = 74)
(h_ricki : rickis_apples = 14)
(h_end : end_apples = 32)
(h_samson : initial_apples - rickis_apples - end_apples = samsons_apples) :
  samsons_apples / Nat.gcd samsons_apples rickis_apples = 2 ∧ rickis_apples / Nat.gcd samsons_apples rickis_apples = 1 :=
by
  sorry

end apples_ratio_l218_218329


namespace commission_percentage_l218_218635

-- Define the given conditions
def cost_of_item : ℝ := 17
def observed_price : ℝ := 25.50
def desired_profit_percentage : ℝ := 0.20

-- Calculate the desired profit in dollars
def desired_profit : ℝ := desired_profit_percentage * cost_of_item

-- Calculate the total desired price for the distributor
def total_desired_price : ℝ := cost_of_item + desired_profit

-- Calculate the commission in dollars
def commission_in_dollars : ℝ := observed_price - total_desired_price

-- Prove that commission percentage taken by the online store is 20%
theorem commission_percentage :
  (commission_in_dollars / observed_price) * 100 = 20 := 
by
  -- This is the placeholder for the proof
  sorry

end commission_percentage_l218_218635


namespace fred_paid_amount_l218_218600

def ticket_price : ℝ := 5.92
def number_of_tickets : ℕ := 2
def borrowed_movie_price : ℝ := 6.79
def change_received : ℝ := 1.37

def total_cost : ℝ := (number_of_tickets : ℝ) * ticket_price + borrowed_movie_price
def amount_paid : ℝ := total_cost + change_received

theorem fred_paid_amount : amount_paid = 20.00 := sorry

end fred_paid_amount_l218_218600


namespace cost_of_four_enchiladas_and_five_tacos_l218_218482

-- Define the cost of an enchilada and a taco
variables (e t : ℝ)

-- Define the conditions given in the problem
def condition1 : Prop := e + 4 * t = 2.30
def condition2 : Prop := 4 * e + t = 3.10

-- Define the final cost of four enchiladas and five tacos
def cost : ℝ := 4 * e + 5 * t

-- State the theorem we need to prove
theorem cost_of_four_enchiladas_and_five_tacos 
  (h1 : condition1 e t) 
  (h2 : condition2 e t) : 
  cost e t = 4.73 := 
sorry

end cost_of_four_enchiladas_and_five_tacos_l218_218482


namespace cos_transformation_l218_218468

variable {θ a : ℝ}

theorem cos_transformation (h : Real.sin (θ + π / 12) = a) :
  Real.cos (θ + 7 * π / 12) = -a := 
sorry

end cos_transformation_l218_218468


namespace percent_increase_is_fifteen_l218_218392

noncomputable def percent_increase_from_sale_price_to_regular_price (P : ℝ) : ℝ :=
  ((P - (0.87 * P)) / (0.87 * P)) * 100

theorem percent_increase_is_fifteen (P : ℝ) (h : P > 0) :
  percent_increase_from_sale_price_to_regular_price P = 15 :=
by
  -- The proof is not required, so we use sorry.
  sorry

end percent_increase_is_fifteen_l218_218392


namespace bill_experience_now_l218_218575

theorem bill_experience_now (B J : ℕ) 
  (h1 : J = 3 * B) 
  (h2 : J + 5 = 2 * (B + 5)) : B + 5 = 10 :=
by
  sorry

end bill_experience_now_l218_218575


namespace cows_in_group_l218_218579

variable (c h : ℕ)

/--
In a group of cows and chickens, the number of legs was 20 more than twice the number of heads.
Cows have 4 legs each and chickens have 2 legs each.
Each animal has one head.
-/
theorem cows_in_group (h : ℕ) (hc : 4 * c + 2 * h = 2 * (c + h) + 20) : c = 10 :=
by
  sorry

end cows_in_group_l218_218579


namespace n_product_expression_l218_218207

theorem n_product_expression (n : ℕ) : n * (n + 1) * (n + 2) * (n + 3) + 1 = (n * (n + 3) + 1)^2 :=
sorry

end n_product_expression_l218_218207


namespace neg_and_implication_l218_218065

variable (p q : Prop)

theorem neg_and_implication : ¬ (p ∧ q) → ¬ p ∨ ¬ q := by
  sorry

end neg_and_implication_l218_218065


namespace deanna_wins_l218_218210

theorem deanna_wins (A B C D : ℕ) (total_games : ℕ) (total_wins : ℕ) (A_wins : A = 5) (B_wins : B = 2)
  (C_wins : C = 1) (total_games_def : total_games = 6) (total_wins_def : total_wins = 12)
  (total_wins_eq : A + B + C + D = total_wins) : D = 4 :=
by
  sorry

end deanna_wins_l218_218210


namespace CarlyWorkedOnElevenDogs_l218_218189

-- Given conditions
def CarlyTrimmedNails : ℕ := 164
def DogsWithThreeLegs : ℕ := 3
def NailsPerPaw : ℕ := 4
def PawsPerThreeLeggedDog : ℕ := 3
def PawsPerFourLeggedDog : ℕ := 4

-- Deduction steps
def TotalPawsWorkedOn := CarlyTrimmedNails / NailsPerPaw
def PawsOnThreeLeggedDogs := DogsWithThreeLegs * PawsPerThreeLeggedDog
def PawsOnFourLeggedDogs := TotalPawsWorkedOn - PawsOnThreeLeggedDogs
def CountFourLeggedDogs := PawsOnFourLeggedDogs / PawsPerFourLeggedDog

-- Total dogs Carly worked on
def TotalDogsCarlyWorkedOn := CountFourLeggedDogs + DogsWithThreeLegs

-- The statement we need to prove
theorem CarlyWorkedOnElevenDogs : TotalDogsCarlyWorkedOn = 11 := by
  sorry

end CarlyWorkedOnElevenDogs_l218_218189


namespace find_number_l218_218763

theorem find_number (x : ℕ) (h : x / 5 = 80 + x / 6) : x = 2400 := 
sorry

end find_number_l218_218763


namespace work_completion_time_l218_218179

theorem work_completion_time (a b c : ℕ) (ha : a = 36) (hb : b = 18) (hc : c = 6) : (1 / (1 / a + 1 / b + 1 / c) = 4) := by
  sorry

end work_completion_time_l218_218179


namespace triangle_side_lengths_l218_218606

theorem triangle_side_lengths (a b c : ℝ) (h1 : a / b = 3 / 4) (h2 : b / c = 4 / 5) (h3 : a + b + c = 60) :
  (a = 15 ∧ b = 20 ∧ c = 25) :=
sorry

end triangle_side_lengths_l218_218606


namespace non_zero_real_positive_integer_l218_218204

theorem non_zero_real_positive_integer (x : ℝ) (h : x ≠ 0) : 
  (∃ k : ℤ, k > 0 ∧ (x - |x-1|) / x = k) ↔ x = 1 := 
sorry

end non_zero_real_positive_integer_l218_218204


namespace total_order_cost_l218_218308

theorem total_order_cost (n : ℕ) (cost_geo cost_eng : ℝ)
  (h1 : n = 35)
  (h2 : cost_geo = 10.50)
  (h3 : cost_eng = 7.50) :
  n * cost_geo + n * cost_eng = 630 := by
  -- proof steps should go here
  sorry

end total_order_cost_l218_218308


namespace conference_center_distance_l218_218438

theorem conference_center_distance
  (d : ℝ)  -- total distance to the conference center
  (t : ℝ)  -- total on-time duration
  (h1 : d = 40 * (t + 1.5))  -- condition from initial speed and late time
  (h2 : d - 40 = 60 * (t - 1.75))  -- condition from increased speed and early arrival
  : d = 310 := 
sorry

end conference_center_distance_l218_218438


namespace A_B_symmetric_x_axis_l218_218107

-- Definitions of points A and B
def A : ℝ × ℝ := (-2, 3)
def B : ℝ × ℝ := (-2, -3)

-- Theorem stating the symmetry relationship between points A and B with respect to the x-axis
theorem A_B_symmetric_x_axis (xA yA xB yB : ℝ) (hA : A = (xA, yA)) (hB : B = (xB, yB)) :
  xA = xB ∧ yA = -yB := by
  sorry

end A_B_symmetric_x_axis_l218_218107


namespace sin_x_solution_l218_218175

theorem sin_x_solution (A B C x : ℝ) (h : A * Real.cos x + B * Real.sin x = C) :
  ∃ (u v : ℝ),  -- We assert the existence of u and v such that 
    Real.sin x = (A * C + B * u) / (A^2 + B^2) ∨ 
    Real.sin x = (A * C - B * v) / (A^2 + B^2) :=
sorry

end sin_x_solution_l218_218175


namespace max_value_of_f_l218_218653

noncomputable def f (x : ℝ) := 2 * Real.cos x + Real.sin x

theorem max_value_of_f : ∃ x, f x = Real.sqrt 5 := sorry

end max_value_of_f_l218_218653


namespace set_equality_proof_l218_218303

theorem set_equality_proof :
  {x : ℕ | x > 1 ∧ x ≤ 3} = {x : ℕ | x = 2 ∨ x = 3} :=
by
  sorry

end set_equality_proof_l218_218303


namespace find_f_log2_3_l218_218953

noncomputable def f : ℝ → ℝ := sorry

axiom f_mono : ∀ x y : ℝ, x ≤ y → f x ≤ f y
axiom f_condition : ∀ x : ℝ, f (f x + 2 / (2^x + 1)) = (1 / 3)

theorem find_f_log2_3 : f (Real.log 3 / Real.log 2) = (1 / 2) :=
by
  sorry

end find_f_log2_3_l218_218953


namespace find_M_l218_218011

theorem find_M :
  (∃ (M : ℕ), (5 + 7 + 9) / 3 = (4020 + 4021 + 4022) / M) → M = 1723 :=
  by
  sorry

end find_M_l218_218011


namespace expand_expression_l218_218551

variable (x y : ℝ)

theorem expand_expression (x y : ℝ) : 12 * (3 * x + 4 - 2 * y) = 36 * x + 48 - 24 * y :=
by
  sorry

end expand_expression_l218_218551


namespace base_conversion_l218_218310

noncomputable def b_value : ℝ := Real.sqrt 21

theorem base_conversion (b : ℝ) (h : b = Real.sqrt 21) : 
  (1 * b^2 + 0 * b + 2) = 23 := 
by
  rw [h]
  sorry

end base_conversion_l218_218310


namespace min_handshakes_l218_218148

theorem min_handshakes (n : ℕ) (h1 : n = 25) 
  (h2 : ∀ (p : ℕ), p < n → ∃ q r : ℕ, q ≠ r ∧ q < n ∧ r < n ∧ q ≠ p ∧ r ≠ p) 
  (h3 : ∃ a b c : ℕ, a < n ∧ b < n ∧ c < n ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ ¬(∃ d : ℕ, (d = a ∨ d = b ∨ d = c) ∧ (¬(a = d ∨ b = d ∨ c = d)) ∧ d < n)) :
  ∃ m : ℕ, m = 28 :=
by
  sorry

end min_handshakes_l218_218148


namespace strawberry_blueberry_price_difference_l218_218956

theorem strawberry_blueberry_price_difference
  (s p t : ℕ → ℕ)
  (strawberries_sold blueberries_sold strawberries_sale_revenue blueberries_sale_revenue strawberries_loss blueberries_loss : ℕ)
  (h1 : strawberries_sold = 54)
  (h2 : strawberries_sale_revenue = 216)
  (h3 : strawberries_loss = 108)
  (h4 : blueberries_sold = 36)
  (h5 : blueberries_sale_revenue = 144)
  (h6 : blueberries_loss = 72)
  (h7 : p strawberries_sold = strawberries_sale_revenue + strawberries_loss)
  (h8 : p blueberries_sold = blueberries_sale_revenue + blueberries_loss)
  : p strawberries_sold / strawberries_sold - p blueberries_sold / blueberries_sold = 0 :=
by
  sorry

end strawberry_blueberry_price_difference_l218_218956


namespace crossed_out_digit_l218_218007

theorem crossed_out_digit (N S S' x : ℕ) (hN : N % 9 = 3) (hS : S % 9 = 3) (hS' : S' % 9 = 7)
  (hS'_eq : S' = S - x) : x = 5 :=
by
  sorry

end crossed_out_digit_l218_218007


namespace FG_square_l218_218089

def trapezoid_EFGH (EF FG GH EH : ℝ) : Prop :=
  ∃ x y : ℝ, 
  EF = 4 ∧
  EH = 31 ∧
  FG = x ∧
  GH = y ∧
  x^2 + (y - 4)^2 = 961 ∧
  x^2 = 4 * y

theorem FG_square (EF EH FG GH x y : ℝ) (h : trapezoid_EFGH EF FG GH EH) :
  FG^2 = 132 :=
by
  obtain ⟨x, y, h1, h2, h3, h4, h5, h6⟩ := h
  exact sorry

end FG_square_l218_218089


namespace negation_of_proposition_l218_218810

theorem negation_of_proposition (a b : ℝ) : 
  ¬ (∀ a b : ℝ, (a = 1 → a + b = 1)) ↔ (∃ a b : ℝ, a = 1 ∧ a + b ≠ 1) :=
by
  sorry

end negation_of_proposition_l218_218810


namespace base_salary_at_least_l218_218660

-- Definitions for the conditions.
def previous_salary : ℕ := 75000
def commission_rate : ℚ := 0.15
def sale_value : ℕ := 750
def min_sales_required : ℚ := 266.67

-- Calculate the commission per sale
def commission_per_sale : ℚ := commission_rate * sale_value

-- Calculate the total commission for the minimum sales required
def total_commission : ℚ := min_sales_required * commission_per_sale

-- The base salary S required to not lose money
theorem base_salary_at_least (S : ℚ) : S + total_commission ≥ previous_salary ↔ S ≥ 45000 := 
by
  -- Use sorry to skip the proof
  sorry

end base_salary_at_least_l218_218660


namespace probability_spade_then_ace_l218_218709

theorem probability_spade_then_ace :
  let total_cards := 52
  let total_aces := 4
  let total_spades := 13
  let ace_of_spades := 1
  let non_ace_spades := total_spades - ace_of_spades
  (non_ace_spades / total_cards) * (total_aces / (total_cards - 1)) +
  (ace_of_spades / total_cards) * ((total_aces - ace_of_spades) / (total_cards - 1)) = (1 / 52) :=
by
  sorry

end probability_spade_then_ace_l218_218709


namespace power_mod_444_444_l218_218436

open Nat

theorem power_mod_444_444 : (444:ℕ)^444 % 13 = 1 := by
  sorry

end power_mod_444_444_l218_218436


namespace ratio_of_black_to_white_after_border_l218_218668

def original_tiles (black white : ℕ) : Prop := black = 14 ∧ white = 21
def original_dimensions (length width : ℕ) : Prop := length = 5 ∧ width = 7

def border_added (length width l w : ℕ) : Prop := l = length + 2 ∧ w = width + 2

def total_white_tiles (initial_white new_white total_white : ℕ) : Prop :=
  total_white = initial_white + new_white

def black_white_ratio (black_tiles white_tiles : ℕ) (ratio : ℚ) : Prop :=
  ratio = black_tiles / white_tiles

theorem ratio_of_black_to_white_after_border 
  (black_white_tiles : ℕ → ℕ → Prop)
  (dimensions : ℕ → ℕ → Prop)
  (border : ℕ → ℕ → ℕ → ℕ → Prop)
  (total_white : ℕ → ℕ → ℕ → Prop)
  (ratio : ℕ → ℕ → ℚ → Prop)
  (black_tiles white_tiles initial_white total_white_new length width l w : ℕ)
  (rat : ℚ) :
  black_white_tiles black_tiles initial_white →
  dimensions length width →
  border length width l w →
  total_white initial_white (l * w - length * width) white_tiles →
  ratio black_tiles white_tiles rat →
  rat = 2 / 7 :=
by
  intros
  sorry

end ratio_of_black_to_white_after_border_l218_218668


namespace base_conversion_subtraction_l218_218487

def base6_to_base10 (n : ℕ) : ℕ :=
3 * (6^2) + 2 * (6^1) + 5 * (6^0)

def base5_to_base10 (m : ℕ) : ℕ :=
2 * (5^2) + 3 * (5^1) + 1 * (5^0)

theorem base_conversion_subtraction : 
  base6_to_base10 325 - base5_to_base10 231 = 59 :=
by
  sorry

end base_conversion_subtraction_l218_218487


namespace sum_product_distinct_zero_l218_218928

open BigOperators

theorem sum_product_distinct_zero {n : ℕ} (h : n ≥ 3) (a : Fin n → ℝ) (ha : Function.Injective a) :
  (∑ i, (a i) * ∏ j in Finset.univ \ {i}, (1 / (a i - a j))) = 0 := 
by
  sorry

end sum_product_distinct_zero_l218_218928


namespace minimum_throws_for_repetition_of_sum_l218_218177

/-- To ensure that the same sum is rolled twice when throwing four fair six-sided dice,
you must throw the dice at least 22 times. -/
theorem minimum_throws_for_repetition_of_sum :
  ∀ (throws : ℕ), (∀ (sum : ℕ), 4 ≤ sum ∧ sum ≤ 24 → ∃ (count : ℕ), count ≤ 21 ∧ sum = count + 4) → throws ≥ 22 :=
by
  sorry

end minimum_throws_for_repetition_of_sum_l218_218177


namespace total_marbles_l218_218747

theorem total_marbles (boxes : ℕ) (marbles_per_box : ℕ) (h1 : boxes = 10) (h2 : marbles_per_box = 100) : (boxes * marbles_per_box = 1000) :=
by
  sorry

end total_marbles_l218_218747


namespace decreasing_interval_of_logarithm_derived_function_l218_218754

theorem decreasing_interval_of_logarithm_derived_function :
  ∀ (x : ℝ), 1 < x → ∃ (f : ℝ → ℝ), (f x = x / (x - 1)) ∧ (∀ (h : x ≠ 1), deriv f x < 0) :=
by
  sorry

end decreasing_interval_of_logarithm_derived_function_l218_218754


namespace distinct_solutions_equation_number_of_solutions_a2019_l218_218059

theorem distinct_solutions_equation (a : ℕ) (ha : a > 1) : 
  ∃ (x y : ℕ), (1 / (x : ℚ) + 1 / (y : ℚ) = 1 / (a : ℚ)) ∧ x > 0 ∧ y > 0 ∧ (x ≠ y) ∧ 
  ∃ (x₁ y₁ x₂ y₂ : ℕ), (1 / (x₁ : ℚ) + 1 / (y₁ : ℚ) = 1 / (a : ℚ)) ∧
  (1 / (x₂ : ℚ) + 1 / (y₂ : ℚ) = 1 / (a : ℚ)) ∧
  x₁ ≠ y₁ ∧ x₂ ≠ y₂ ∧ (x₁, y₁) ≠ (x₂, y₂) := 
sorry

theorem number_of_solutions_a2019 :
  ∃ n, n = (3 * 3) := 
by {
  -- use 2019 = 3 * 673 and divisor count
  sorry 
}

end distinct_solutions_equation_number_of_solutions_a2019_l218_218059


namespace total_students_is_45_l218_218499

def num_students_in_class 
  (excellent_chinese : ℕ) 
  (excellent_math : ℕ) 
  (excellent_both : ℕ) 
  (no_excellent : ℕ) : ℕ :=
  excellent_chinese + excellent_math - excellent_both + no_excellent

theorem total_students_is_45 
  (h1 : excellent_chinese = 15)
  (h2 : excellent_math = 18)
  (h3 : excellent_both = 8)
  (h4 : no_excellent = 20) : 
  num_students_in_class excellent_chinese excellent_math excellent_both no_excellent = 45 := 
  by 
    sorry

end total_students_is_45_l218_218499


namespace no_beverages_l218_218338

noncomputable def businessmen := 30
def coffee := 15
def tea := 13
def water := 6
def coffee_tea := 7
def tea_water := 3
def coffee_water := 2
def all_three := 1

theorem no_beverages (businessmen coffee tea water coffee_tea tea_water coffee_water all_three):
  businessmen - (coffee + tea + water - coffee_tea - tea_water - coffee_water + all_three) = 7 :=
by sorry

end no_beverages_l218_218338


namespace gcd_of_polynomial_l218_218152

theorem gcd_of_polynomial (x : ℕ) (hx : 32515 ∣ x) :
    Nat.gcd ((3 * x + 5) * (5 * x + 3) * (11 * x + 7) * (x + 17)) x = 35 :=
sorry

end gcd_of_polynomial_l218_218152


namespace general_term_formula_sum_first_n_terms_l218_218188

theorem general_term_formula :
  ∀ (a : ℕ → ℝ), 
  (∀ n, a n > 0) →
  a 1 = 1 / 2 →
  (∀ n, (a (n + 1))^2 = a n^2 + 2 * ↑n) →
  (∀ n, a n = n - 1 / 2) := 
  sorry

theorem sum_first_n_terms :
  ∀ (a : ℕ → ℝ) (b : ℕ → ℝ) (S : ℕ → ℝ),
  (∀ n, a n > 0) →
  a 1 = 1 / 2 →
  (∀ n, (a (n + 1))^2 = a n^2 + 2 * ↑n) →
  (∀ n, a n = n - 1 / 2) →
  (∀ n, b n = 1 / (a n * a (n + 1))) →
  (∀ n, S n = 2 * (1 - 1 / (2 * n + 1))) →
  (S n = 4 * n / (2 * n + 1)) :=
  sorry

end general_term_formula_sum_first_n_terms_l218_218188


namespace smallest_value_expression_l218_218465

theorem smallest_value_expression (x y : ℝ) (hx : -5 ≤ x ∧ x ≤ -3) (hy : 3 ≤ y ∧ y ≤ 5) :
  ∃ m, m = y ∧ m = 3 :=
by
  sorry

end smallest_value_expression_l218_218465


namespace probability_of_yellow_face_l218_218345

def total_faces : ℕ := 12
def red_faces : ℕ := 5
def yellow_faces : ℕ := 4
def blue_faces : ℕ := 2
def green_faces : ℕ := 1

theorem probability_of_yellow_face : (yellow_faces : ℚ) / (total_faces : ℚ) = 1 / 3 := by
  sorry

end probability_of_yellow_face_l218_218345


namespace maximum_value_x_2y_2z_l218_218543

noncomputable def max_sum (x y z : ℝ) (h : x^2 + y^2 + z^2 = 25) : ℝ :=
  x + 2*y + 2*z

theorem maximum_value_x_2y_2z (x y z : ℝ) (h : x^2 + y^2 + z^2 = 25) : 
  max_sum x y z h ≤ 15 :=
sorry

end maximum_value_x_2y_2z_l218_218543


namespace correct_overestimation_l218_218534

theorem correct_overestimation (y : ℕ) : 
  25 * y + 4 * y = 29 * y := 
by 
  sorry

end correct_overestimation_l218_218534


namespace income_increase_correct_l218_218114

noncomputable def income_increase_percentage (I1 : ℝ) (S1 : ℝ) (E1 : ℝ) (I2 : ℝ) (S2 : ℝ) (E2 : ℝ) (P : ℝ) :=
  S1 = 0.5 * I1 ∧
  S2 = 2 * S1 ∧
  E1 = 0.5 * I1 ∧
  E2 = I2 - S2 ∧
  I2 = I1 * (1 + P / 100) ∧
  E1 + E2 = 2 * E1

theorem income_increase_correct (I1 : ℝ) (S1 : ℝ) (E1 : ℝ) (I2 : ℝ) (S2 : ℝ) (E2 : ℝ) (P : ℝ)
  (h1 : income_increase_percentage I1 S1 E1 I2 S2 E2 P) : P = 50 :=
sorry

end income_increase_correct_l218_218114


namespace minimization_problem_l218_218319

theorem minimization_problem (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 1) (h5 : x ≤ y) (h6 : y ≤ z) (h7 : z ≤ 3 * x) :
  x * y * z ≥ 1 / 18 := 
sorry

end minimization_problem_l218_218319


namespace find_x0_l218_218021

-- Define the given conditions
variable (p x_0 : ℝ) (P : ℝ × ℝ) (O : ℝ × ℝ)
variable (h_parabola : x_0^2 = 2 * p * 1)
variable (h_p_gt_zero : p > 0)
variable (h_point_P : P = (x_0, 1))
variable (h_origin : O = (0, 0))
variable (h_distance_condition : dist (x_0, 1) (0, 0) = dist (x_0, 1) (0, -p / 2))

-- The theorem we aim to prove
theorem find_x0 : x_0 = 2 * Real.sqrt 2 :=
  sorry

end find_x0_l218_218021


namespace circulation_ratio_l218_218910

theorem circulation_ratio (A C_1971 C_total : ℕ) 
(hC1971 : C_1971 = 4 * A) 
(hCtotal : C_total = C_1971 + 9 * A) : 
(C_1971 : ℚ) / (C_total : ℚ) = 4 / 13 := 
sorry

end circulation_ratio_l218_218910


namespace value_of_k_l218_218452

theorem value_of_k (k : ℕ) : (∃ b : ℕ, x^2 - 20 * x + k = (x + b)^2) → k = 100 := by
  sorry

end value_of_k_l218_218452


namespace largest_possible_b_l218_218429

theorem largest_possible_b (a b c : ℤ) (h1 : a > b) (h2 : b > c) (h3 : c > 2) (h4 : a * b * c = 360) : b = 10 :=
sorry

end largest_possible_b_l218_218429


namespace find_possible_values_for_P_l218_218149

theorem find_possible_values_for_P (x y P : ℕ) (h1 : x < y) :
  P = (x^3 - y) / (1 + x * y) → (P = 0 ∨ P ≥ 2) :=
by
  sorry

end find_possible_values_for_P_l218_218149


namespace derivative_of_y_l218_218075

noncomputable def y (x : ℝ) : ℝ := (Real.log x) / x + x * Real.exp x

theorem derivative_of_y (x : ℝ) (hx : x > 0) : 
  deriv y x = (1 - Real.log x) / (x^2) + (x + 1) * Real.exp x := by
  sorry

end derivative_of_y_l218_218075


namespace lloyd_earnings_l218_218731

theorem lloyd_earnings:
  let regular_hours := 7.5
  let regular_rate := 4.50
  let overtime_multiplier := 2.0
  let hours_worked := 10.5
  let overtime_hours := hours_worked - regular_hours
  let overtime_rate := overtime_multiplier * regular_rate
  let regular_pay := regular_hours * regular_rate
  let overtime_pay := overtime_hours * overtime_rate
  let total_earnings := regular_pay + overtime_pay
  total_earnings = 60.75 :=
by
  sorry

end lloyd_earnings_l218_218731


namespace find_price_per_backpack_l218_218672

noncomputable def original_price_of_each_backpack
  (total_backpacks : ℕ)
  (monogram_cost : ℕ)
  (total_cost : ℕ)
  (backpacks_cost_before_discount : ℕ) : ℕ :=
total_cost - (total_backpacks * monogram_cost)

theorem find_price_per_backpack
  (total_backpacks : ℕ := 5)
  (monogram_cost : ℕ := 12)
  (total_cost : ℕ := 140)
  (expected_price_per_backpack : ℕ := 16) :
  original_price_of_each_backpack total_backpacks monogram_cost total_cost / total_backpacks = expected_price_per_backpack :=
by
  sorry

end find_price_per_backpack_l218_218672


namespace equivalent_statements_l218_218008

variable (P Q : Prop)

theorem equivalent_statements (h : P → Q) :
  (¬Q → ¬P) ∧ (¬P ∨ Q) :=
by 
  sorry

end equivalent_statements_l218_218008


namespace max_reflections_l218_218829

theorem max_reflections (angle_increase : ℕ := 10) (max_angle : ℕ := 90) :
  ∃ n : ℕ, 10 * n ≤ max_angle ∧ ∀ m : ℕ, (10 * (m + 1) > max_angle → m < n) := 
sorry

end max_reflections_l218_218829


namespace infinite_n_exists_r_s_t_l218_218115

noncomputable def a (n : ℕ) : ℝ := n^(1/3 : ℝ)
noncomputable def b (n : ℕ) : ℝ := 1 / (a n - ⌊a n⌋)
noncomputable def c (n : ℕ) : ℝ := 1 / (b n - ⌊b n⌋)

theorem infinite_n_exists_r_s_t :
  ∃ (n : ℕ) (r s t : ℤ), (0 < n ∧ ¬∃ k : ℕ, n = k^3) ∧ (¬(r = 0 ∧ s = 0 ∧ t = 0)) ∧ (r * a n + s * b n + t * c n = 0) :=
sorry

end infinite_n_exists_r_s_t_l218_218115


namespace increase_in_circumference_by_2_cm_l218_218174

noncomputable def radius_increase_by_two (r : ℝ) : ℝ := r + 2
noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

theorem increase_in_circumference_by_2_cm (r : ℝ) : 
    circumference (radius_increase_by_two r) - circumference r = 12.56 :=
by sorry

end increase_in_circumference_by_2_cm_l218_218174


namespace milk_production_days_l218_218976

theorem milk_production_days (x : ℕ) (h : x > 0) :
  let daily_production_per_cow := (x + 1) / (x * (x + 2))
  let total_daily_production := (x + 4) * daily_production_per_cow
  ((x + 7) / total_daily_production) = (x * (x + 2) * (x + 7)) / ((x + 1) * (x + 4)) := 
by
  sorry

end milk_production_days_l218_218976


namespace necessary_but_not_sufficient_l218_218881

theorem necessary_but_not_sufficient (x: ℝ) :
  (1 < x ∧ x < 4) → (1 < x ∧ x < 3) := by
sorry

end necessary_but_not_sufficient_l218_218881


namespace bus_driver_earnings_l218_218502

variables (rate : ℝ) (regular_hours overtime_hours : ℕ) (regular_rate overtime_rate : ℝ)

def calculate_regular_earnings (regular_rate : ℝ) (regular_hours : ℕ) : ℝ :=
  regular_rate * regular_hours

def calculate_overtime_earnings (overtime_rate : ℝ) (overtime_hours : ℕ) : ℝ :=
  overtime_rate * overtime_hours

def total_compensation (regular_rate overtime_rate : ℝ) (regular_hours overtime_hours : ℕ) : ℝ :=
  calculate_regular_earnings regular_rate regular_hours + calculate_overtime_earnings overtime_rate overtime_hours

theorem bus_driver_earnings :
  let regular_rate := 16
  let overtime_rate := regular_rate * 1.75
  let regular_hours := 40
  let total_hours := 44
  let overtime_hours := total_hours - regular_hours
  total_compensation regular_rate overtime_rate regular_hours overtime_hours = 752 :=
by
  sorry

end bus_driver_earnings_l218_218502


namespace sqrt5_minus_2_power_2023_mul_sqrt5_plus_2_power_2023_eq_one_l218_218246

-- Defining the terms and the theorem
theorem sqrt5_minus_2_power_2023_mul_sqrt5_plus_2_power_2023_eq_one :
  (Real.sqrt 5 - 2) ^ 2023 * (Real.sqrt 5 + 2) ^ 2023 = 1 := 
by
  sorry

end sqrt5_minus_2_power_2023_mul_sqrt5_plus_2_power_2023_eq_one_l218_218246


namespace gcd_78_36_l218_218159

theorem gcd_78_36 : Nat.gcd 78 36 = 6 := 
by
  sorry

end gcd_78_36_l218_218159


namespace discount_percentage_l218_218832

variable (P : ℝ) (r : ℝ) (S : ℝ)

theorem discount_percentage (hP : P = 20) (hr : r = 30 / 100) (hS : S = 13) :
  (P * (1 + r) - S) / (P * (1 + r)) * 100 = 50 := 
sorry

end discount_percentage_l218_218832


namespace beast_of_war_running_time_correct_l218_218983

def running_time_millennium : ℕ := 120

def running_time_alpha_epsilon (rt_millennium : ℕ) : ℕ := rt_millennium - 30

def running_time_beast_of_war (rt_alpha_epsilon : ℕ) : ℕ := rt_alpha_epsilon + 10

theorem beast_of_war_running_time_correct :
  running_time_beast_of_war (running_time_alpha_epsilon running_time_millennium) = 100 := by sorry

end beast_of_war_running_time_correct_l218_218983


namespace original_faculty_is_287_l218_218513

noncomputable def original_faculty (F : ℝ) : Prop :=
  (F * 0.85 * 0.80 = 195)

theorem original_faculty_is_287 : ∃ F : ℝ, original_faculty F ∧ F = 287 := 
by 
  use 287
  sorry

end original_faculty_is_287_l218_218513


namespace ella_incorrect_answers_l218_218874

theorem ella_incorrect_answers
  (marion_score : ℕ)
  (ella_score : ℕ)
  (total_items : ℕ)
  (h1 : marion_score = 24)
  (h2 : marion_score = (ella_score / 2) + 6)
  (h3 : total_items = 40) : 
  total_items - ella_score = 4 :=
by
  sorry

end ella_incorrect_answers_l218_218874


namespace antipov_inequality_l218_218332

theorem antipov_inequality (a b c : ℕ) 
  (h1 : ¬ (a ∣ b ∨ b ∣ a ∨ a ∣ c ∨ c ∣ a ∨ b ∣ c ∨ c ∣ b)) 
  (h2 : (ab + 1) ∣ (abc + 1)) : c ≥ b :=
sorry

end antipov_inequality_l218_218332


namespace tan_prod_eq_sqrt_seven_l218_218441

theorem tan_prod_eq_sqrt_seven : 
  let x := (Real.pi / 7) 
  let y := (2 * Real.pi / 7)
  let z := (3 * Real.pi / 7)
  Real.tan x * Real.tan y * Real.tan z = Real.sqrt 7 :=
by
  sorry

end tan_prod_eq_sqrt_seven_l218_218441


namespace unique_solution_system_eqns_l218_218524

theorem unique_solution_system_eqns (a b c : ℕ) :
  (a^3 - b^3 - c^3 = 3 * a * b * c) ∧ (a^2 = 2 * (b + c)) ↔ (a = 2 ∧ b = 1 ∧ c = 1) := by 
  sorry

end unique_solution_system_eqns_l218_218524


namespace polygon_parallel_edges_l218_218518

theorem polygon_parallel_edges (n : ℕ) (h : n > 2) :
  (∃ i j, i ≠ j ∧ (i + 1) % n = (j + 1) % n) ↔ (∃ k, n = 2 * k) :=
  sorry

end polygon_parallel_edges_l218_218518


namespace geometric_series_common_ratio_l218_218239

theorem geometric_series_common_ratio :
  ∀ (a r : ℝ), (r ≠ 1) → 
  (∑' n, a * r^n = 64 * ∑' n, a * r^(n+4)) →
  r = 1 / 2 :=
by
  intros a r hnr heq
  have hsum1 : ∑' n, a * r^n = a / (1 - r) := sorry
  have hsum2 : ∑' n, a * r^(n+4) = a * r^4 / (1 - r) := sorry
  rw [hsum1, hsum2] at heq
  -- Further steps to derive r = 1/2 are omitted
  sorry

end geometric_series_common_ratio_l218_218239


namespace determine_d_l218_218994

theorem determine_d (f g : ℝ → ℝ) (c d : ℝ) (h1 : ∀ x, f x = 5 * x + c) (h2 : ∀ x, g x = c * x + 3) (h3 : ∀ x, f (g x) = 15 * x + d) : d = 18 := 
  sorry

end determine_d_l218_218994


namespace average_computation_l218_218644

variable {a b c X Y Z : ℝ}

theorem average_computation 
  (h1 : a + b + c = 15)
  (h2 : X + Y + Z = 21) :
  ((2 * a + 3 * X) + (2 * b + 3 * Y) + (2 * c + 3 * Z)) / 3 = 31 :=
by
  sorry

end average_computation_l218_218644


namespace evaluate_expression_l218_218442

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := 
by 
  sorry

end evaluate_expression_l218_218442


namespace question1_question2_l218_218237

noncomputable def f (x b c : ℝ) := x^2 + b * x + c

theorem question1 (b c : ℝ) (h : ∀ x : ℝ, 2 * x + b ≤ f x b c) (x : ℝ) (hx : 0 ≤ x) :
  f x b c ≤ (x + c)^2 :=
sorry

theorem question2 (b c m : ℝ) (h : ∀ b c : ℝ, b ≠ c → f c b b - f b b b ≤ m * (c^2 - b^2)) :
  m ≥ 3/2 :=
sorry

end question1_question2_l218_218237


namespace find_x_y_z_l218_218397

theorem find_x_y_z (x y z : ℝ) (h1 : 1 ≤ x ∧ 1 ≤ y ∧ 1 ≤ z) (h2 : x * y * z = 10)
  (h3 : x ^ Real.log x * y ^ Real.log y * z ^ Real.log z = 10) :
  (x = 1 ∧ y = 1 ∧ z = 10) ∨ (x = 10 ∧ y = 1 ∧ z = 1) ∨ (x = 1 ∧ y = 10 ∧ z = 1) :=
sorry

end find_x_y_z_l218_218397


namespace average_monthly_growth_rate_l218_218344

-- Define the initial and final production quantities
def initial_production : ℝ := 100
def final_production : ℝ := 144

-- Define the average monthly growth rate
def avg_monthly_growth_rate (x : ℝ) : Prop :=
  initial_production * (1 + x)^2 = final_production

-- Statement of the problem to be verified
theorem average_monthly_growth_rate :
  ∃ x : ℝ, avg_monthly_growth_rate x ∧ x = 0.2 :=
by
  sorry

end average_monthly_growth_rate_l218_218344


namespace fraction_simplification_l218_218426

variable (a b x : ℝ)
variable (h1 : x = a / b)
variable (h2 : a ≠ b)
variable (h3 : b ≠ 0)
variable (h4 : a = b * x ^ 2)

theorem fraction_simplification : (a + b) / (a - b) = (x ^ 2 + 1) / (x ^ 2 - 1) := by
  sorry

end fraction_simplification_l218_218426


namespace polygon_sides_l218_218835

theorem polygon_sides (h : 1440 = (n - 2) * 180) : n = 10 := 
by {
  -- Here, the proof would show the steps to solve the equation h and confirm n = 10
  sorry
}

end polygon_sides_l218_218835


namespace max_sin_B_l218_218383

theorem max_sin_B (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
    (AB BC : ℝ)
    (hAB : AB = 25) (hBC : BC = 20) :
    ∃ sinB : ℝ, sinB = 3 / 5 := sorry

end max_sin_B_l218_218383


namespace original_price_of_cycle_l218_218394

variable (P : ℝ)

theorem original_price_of_cycle (h : 0.92 * P = 1610) : P = 1750 :=
sorry

end original_price_of_cycle_l218_218394


namespace max_value_is_one_l218_218979

noncomputable def max_value_fraction (x : ℝ) : ℝ :=
  (1 + Real.cos x) / (Real.sin x + Real.cos x + 2)

theorem max_value_is_one : ∃ x : ℝ, max_value_fraction x = 1 := by
  sorry

end max_value_is_one_l218_218979


namespace farmer_apples_l218_218742

theorem farmer_apples (initial_apples : ℕ) (given_apples : ℕ) (final_apples : ℕ) 
  (h1 : initial_apples = 127) (h2 : given_apples = 88) 
  (h3 : final_apples = initial_apples - given_apples) : final_apples = 39 :=
by {
  -- proof steps would go here, but since only the statement is needed, we use 'sorry' to skip the proof
  sorry
}

end farmer_apples_l218_218742


namespace west_1000_move_l218_218153

def eastMovement (d : Int) := d  -- east movement positive
def westMovement (d : Int) := -d -- west movement negative

theorem west_1000_move : westMovement 1000 = -1000 :=
  by
    sorry

end west_1000_move_l218_218153


namespace shelves_per_case_l218_218413

noncomputable section

-- Define the total number of ridges
def total_ridges : ℕ := 8640

-- Define the number of ridges per record
def ridges_per_record : ℕ := 60

-- Define the number of records per shelf when the shelf is 60% full
def records_per_shelf : ℕ := (60 * 20) / 100

-- Define the number of ridges per shelf
def ridges_per_shelf : ℕ := records_per_shelf * ridges_per_record

-- Given 4 cases, we need to determine the number of shelves per case
theorem shelves_per_case (cases shelves : ℕ) (h₁ : cases = 4) (h₂ : shelves * ridges_per_shelf = total_ridges) :
  shelves / cases = 3 := by
  sorry

end shelves_per_case_l218_218413


namespace twelfth_term_geometric_sequence_l218_218334

theorem twelfth_term_geometric_sequence :
  let a1 := 5
  let r := (2 / 5 : ℝ)
  (a1 * r ^ 11) = (10240 / 48828125 : ℝ) :=
by
  sorry

end twelfth_term_geometric_sequence_l218_218334


namespace solve_equation_l218_218862

theorem solve_equation (x : ℝ) (h1 : x ≠ 1) (h2 : x^2 - 1 ≠ 0) : (x / (x - 1) = 2 / (x^2 - 1)) → (x = -2) :=
by
  intro h
  sorry

end solve_equation_l218_218862


namespace number_of_pipes_used_l218_218796

-- Definitions
def T1 : ℝ := 15
def T2 : ℝ := T1 - 5
def T3 : ℝ := T2 - 4
def condition : Prop := 1 / T1 + 1 / T2 = 1 / T3

-- Proof Statement
theorem number_of_pipes_used : condition → 3 = 3 :=
by intros h; sorry

end number_of_pipes_used_l218_218796


namespace mason_grandmother_age_l218_218533

theorem mason_grandmother_age (mason_age: ℕ) (sydney_age: ℕ) (father_age: ℕ) (grandmother_age: ℕ)
  (h1: mason_age = 20)
  (h2: mason_age * 3 = sydney_age)
  (h3: sydney_age + 6 = father_age)
  (h4: father_age * 2 = grandmother_age) : 
  grandmother_age = 132 :=
by
  sorry

end mason_grandmother_age_l218_218533


namespace train_speed_comparison_l218_218659

variables (V_A V_B : ℝ)

open Classical

theorem train_speed_comparison
  (distance_AB : ℝ)
  (h_distance : distance_AB = 360)
  (h_time_limit : V_A ≤ 72)
  (h_meeting_time : 3 * V_A + 2 * V_B > 360) :
  V_B > V_A :=
by {
  sorry
}

end train_speed_comparison_l218_218659


namespace find_other_number_l218_218078

theorem find_other_number
  (B : ℕ)
  (hcf_condition : Nat.gcd 24 B = 12)
  (lcm_condition : Nat.lcm 24 B = 396) :
  B = 198 :=
by
  sorry

end find_other_number_l218_218078


namespace square_area_parabola_inscribed_l218_218980

theorem square_area_parabola_inscribed (s : ℝ) (x y : ℝ) :
  (y = x^2 - 6 * x + 8) ∧
  (s = -2 + 2 * Real.sqrt 5) ∧
  (x = 3 - s / 2 ∨ x = 3 + s / 2) →
  s ^ 2 = 24 - 8 * Real.sqrt 5 :=
by
  sorry

end square_area_parabola_inscribed_l218_218980


namespace arithmetic_sequence_properties_l218_218278

theorem arithmetic_sequence_properties 
  (a : ℕ → ℤ) 
  (h1 : a 1 + a 2 + a 3 = 21) 
  (h2 : a 1 * a 2 * a 3 = 231) :
  (a 2 = 7) ∧ (∀ n, a n = -4 * n + 15 ∨ a n = 4 * n - 1) := 
by
  sorry

end arithmetic_sequence_properties_l218_218278


namespace area_of_region_l218_218322

noncomputable def T := 516

def region (x y : ℝ) : Prop :=
  |x| - |y| ≤ T - 500 ∧ |y| ≤ T - 500

theorem area_of_region :
  (4 * (T - 500)^2 = 1024) :=
  sorry

end area_of_region_l218_218322


namespace geom_seq_308th_term_l218_218526

def geom_seq (a : ℤ) (r : ℤ) (n : ℕ) : ℤ :=
  a * r ^ n

-- Given conditions
def a := 10
def r := -1

theorem geom_seq_308th_term : geom_seq a r 307 = -10 := by
  sorry

end geom_seq_308th_term_l218_218526


namespace count_possible_third_side_lengths_l218_218871

theorem count_possible_third_side_lengths : ∀ (n : ℤ), 2 < n ∧ n < 14 → ∃ s : Finset ℤ, s.card = 11 ∧ ∀ x ∈ s, 2 < x ∧ x < 14 := by
  sorry

end count_possible_third_side_lengths_l218_218871


namespace difference_of_two_numbers_l218_218289

def nat_sum := 22305
def a := ∃ a: ℕ, 5 ∣ a
def is_b (a b: ℕ) := b = a / 10 + 3

theorem difference_of_two_numbers (a b : ℕ) (h : a + b = nat_sum) (h1 : 5 ∣ a) (h2 : is_b a b) : a - b = 14872 :=
by
  sorry

end difference_of_two_numbers_l218_218289


namespace compute_fg_l218_218576

def g (x : ℕ) : ℕ := 2 * x + 6
def f (x : ℕ) : ℕ := 4 * x - 8
def x : ℕ := 10

theorem compute_fg : f (g x) = 96 := by
  sorry

end compute_fg_l218_218576


namespace class_students_l218_218498

theorem class_students (A B : ℕ) 
  (h1 : A + B = 85) 
  (h2 : (3 * A) / 8 + (3 * B) / 5 = 42) : 
  A = 40 ∧ B = 45 :=
by
  sorry

end class_students_l218_218498


namespace goats_count_l218_218935

variable (h d c t g : Nat)
variable (l : Nat)

theorem goats_count 
  (h_eq : h = 2)
  (d_eq : d = 5)
  (c_eq : c = 7)
  (t_eq : t = 3)
  (l_eq : l = 72)
  (legs_eq : 4 * h + 4 * d + 4 * c + 4 * t + 4 * g = l) : 
  g = 1 := by
  sorry

end goats_count_l218_218935


namespace negative_square_inequality_l218_218100

theorem negative_square_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > b^2 :=
sorry

end negative_square_inequality_l218_218100


namespace P_has_real_root_l218_218788

def P : ℝ → ℝ := sorry
variables (a1 a2 a3 b1 b2 b3 : ℝ)

axiom a1_nonzero : a1 ≠ 0
axiom a2_nonzero : a2 ≠ 0
axiom a3_nonzero : a3 ≠ 0

axiom functional_eq (x : ℝ) :
  P (a1 * x + b1) + P (a2 * x + b2) = P (a3 * x + b3)

theorem P_has_real_root :
  ∃ x : ℝ, P x = 0 :=
sorry

end P_has_real_root_l218_218788


namespace boat_travel_distance_l218_218926

theorem boat_travel_distance
  (D : ℝ) -- Distance traveled in both directions
  (t : ℝ) -- Time in hours it takes to travel upstream
  (speed_boat : ℝ) -- Speed of the boat in still water
  (speed_stream : ℝ) -- Speed of the stream
  (time_diff : ℝ) -- Difference in time between downstream and upstream travel
  (h1 : speed_boat = 10)
  (h2 : speed_stream = 2)
  (h3 : time_diff = 1.5)
  (h4 : D = 8 * t)
  (h5 : D = 12 * (t - time_diff)) :
  D = 36 := by
  sorry

end boat_travel_distance_l218_218926


namespace crackers_count_l218_218085

theorem crackers_count (crackers_Marcus crackers_Mona crackers_Nicholas : ℕ) 
  (h1 : crackers_Marcus = 3 * crackers_Mona)
  (h2 : crackers_Nicholas = crackers_Mona + 6)
  (h3 : crackers_Marcus = 27) : crackers_Nicholas = 15 := 
by 
  sorry

end crackers_count_l218_218085


namespace key_lime_yield_l218_218218

def audrey_key_lime_juice_yield (cup_to_key_lime_juice_ratio: ℚ) (lime_juice_doubling_factor: ℚ) (tablespoons_per_cup: ℕ) (num_key_limes: ℕ) : ℚ :=
  let total_lime_juice_cups := cup_to_key_lime_juice_ratio * lime_juice_doubling_factor
  let total_lime_juice_tablespoons := total_lime_juice_cups * tablespoons_per_cup
  total_lime_juice_tablespoons / num_key_limes

-- Statement of the problem
theorem key_lime_yield :
  audrey_key_lime_juice_yield (1/4) 2 16 8 = 1 := 
by 
  sorry

end key_lime_yield_l218_218218


namespace midpoint_coords_product_l218_218061

def midpoint_prod (x1 y1 x2 y2 : ℤ) : ℤ :=
  let mx := (x1 + x2) / 2
  let my := (y1 + y2) / 2
  mx * my

theorem midpoint_coords_product :
  midpoint_prod 4 (-7) (-8) 9 = -2 := by
  sorry

end midpoint_coords_product_l218_218061


namespace swan_percentage_not_ducks_l218_218993

theorem swan_percentage_not_ducks (total_birds geese swans herons ducks : ℝ)
  (h_total : total_birds = 100)
  (h_geese : geese = 0.30 * total_birds)
  (h_swans : swans = 0.20 * total_birds)
  (h_herons : herons = 0.20 * total_birds)
  (h_ducks : ducks = 0.30 * total_birds) :
  (swans / (total_birds - ducks) * 100) = 28.57 :=
by
  sorry

end swan_percentage_not_ducks_l218_218993


namespace digits_in_number_l218_218192

def four_digit_number (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def contains_digits (n : ℕ) (d1 d2 d3 : ℕ) : Prop :=
  (n / 1000 = d1 ∨ n / 100 % 10 = d1 ∨ n / 10 % 10 = d1 ∨ n % 10 = d1) ∧
  (n / 1000 = d2 ∨ n / 100 % 10 = d2 ∨ n / 10 % 10 = d2 ∨ n % 10 = d2) ∧
  (n / 1000 = d3 ∨ n / 100 % 10 = d3 ∨ n / 10 % 10 = d3 ∨ n % 10 = d3)

def exactly_two_statements_true (s1 s2 s3 : Prop) : Prop :=
  (s1 ∧ s2 ∧ ¬s3) ∨ (s1 ∧ ¬s2 ∧ s3) ∨ (¬s1 ∧ s2 ∧ s3)

theorem digits_in_number (n : ℕ) 
  (h1 : four_digit_number n)
  (h2 : contains_digits n 1 4 5 ∨ contains_digits n 1 5 9 ∨ contains_digits n 7 8 9)
  (h3 : exactly_two_statements_true (contains_digits n 1 4 5) (contains_digits n 1 5 9) (contains_digits n 7 8 9)) :
  contains_digits n 1 4 5 ∧ contains_digits n 1 5 9 :=
sorry

end digits_in_number_l218_218192


namespace min_value_inequality_l218_218049

noncomputable def min_value_expression (x y z : ℝ) : ℝ :=
  (Real.sqrt ((x^2 + y^2 + z^2) * (4 * x^2 + y^2 + z^2))) / (x * y * z)

theorem min_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  min_value_expression x y z ≥ 3 / 2 := by
  sorry

end min_value_inequality_l218_218049


namespace speed_ratio_l218_218970

def distance_to_work := 28
def speed_back := 14
def total_time := 6

theorem speed_ratio 
  (d : ℕ := distance_to_work) 
  (v_2 : ℕ := speed_back) 
  (t : ℕ := total_time) : 
  ∃ v_1 : ℕ, (d / v_1 + d / v_2 = t) ∧ (v_2 / v_1 = 2) :=
by 
  sorry

end speed_ratio_l218_218970


namespace geom_sequence_ratio_l218_218221

-- Definitions and assumptions for the problem
noncomputable def geom_seq (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, 0 < r ∧ r < 1 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geom_sequence_ratio (a : ℕ → ℝ) (r : ℝ) 
  (h_geom: geom_seq a)
  (h_r: 0 < r ∧ r < 1)
  (h_seq: ∀ n : ℕ, a (n + 1) = a n * r)
  (ha1: a 7 * a 14 = 6)
  (ha2: a 4 + a 17 = 5) :
  (a 5 / a 18) = (3 / 2) :=
sorry

end geom_sequence_ratio_l218_218221


namespace initial_sum_l218_218803

theorem initial_sum (P : ℝ) (compound_interest : ℝ) (r1 r2 r3 r4 r5 : ℝ) 
  (h1 : r1 = 0.06) (h2 : r2 = 0.08) (h3 : r3 = 0.07) (h4 : r4 = 0.09) (h5 : r5 = 0.10)
  (interest_sum : compound_interest = 4016.25) :
  P = 4016.25 / ((1 + r1) * (1 + r2) * (1 + r3) * (1 + r4) * (1 + r5) - 1) :=
by
  sorry

end initial_sum_l218_218803


namespace total_fundamental_particles_l218_218013

def protons := 9
def neutrons := 19 - protons
def electrons := protons
def total_particles := protons + neutrons + electrons

theorem total_fundamental_particles : total_particles = 28 := by
  sorry

end total_fundamental_particles_l218_218013


namespace monotonicity_of_f_range_of_a_if_no_zeros_l218_218276

noncomputable def f (a x : ℝ) := a^2 * x^2 + a * x - 3 * Real.log x + 1

theorem monotonicity_of_f (a : ℝ) (h : a > 0) :
  (∀ x, x > 0 → x < 1/a → deriv (f a) x < 0) ∧
  (∀ x, x > 1/a → deriv (f a) x > 0) := sorry

theorem range_of_a_if_no_zeros 
  (h1 : ∀ x > 0, f a x ≠ 0) : a > 1 / Real.exp 1 := sorry

end monotonicity_of_f_range_of_a_if_no_zeros_l218_218276


namespace other_number_is_31_l218_218492

namespace LucasProblem

-- Definitions of the integers a and b and the condition on their sum
variables (a b : ℤ)
axiom h_sum : 3 * a + 4 * b = 161
axiom h_one_is_17 : a = 17 ∨ b = 17

-- The theorem we need to prove
theorem other_number_is_31 (h_one_is_17 : a = 17 ∨ b = 17) : 
  (b = 17 → a = 31) ∧ (a = 17 → false) :=
by
  sorry

end LucasProblem

end other_number_is_31_l218_218492


namespace trig_expr_eval_sin_minus_cos_l218_218214

-- Problem 1: Evaluation of trigonometric expression
theorem trig_expr_eval : 
    (Real.sin (-π / 2) + 3 * Real.cos 0 - 2 * Real.tan (3 * π / 4) - 4 * Real.cos (5 * π / 3)) = 2 :=
by 
    sorry

-- Problem 2: Given tangent value and angle constraints, find sine minus cosine
theorem sin_minus_cos {θ : ℝ} 
    (h1 : Real.tan θ = 4 / 3)
    (h2 : 0 < θ)
    (h3 : θ < π / 2) : 
    (Real.sin θ - Real.cos θ) = 1 / 5 :=
by 
    sorry

end trig_expr_eval_sin_minus_cos_l218_218214


namespace no_real_solution_for_pairs_l218_218110

theorem no_real_solution_for_pairs (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ¬ (1 / a + 1 / b = 1 / (a + b)) :=
by
  sorry

end no_real_solution_for_pairs_l218_218110


namespace tan_ratio_triangle_area_l218_218227

theorem tan_ratio (a b c A B C : ℝ) (h1 : c = -3 * b * Real.cos A) :
  Real.tan A / Real.tan B = -4 := by
  sorry

theorem triangle_area (a b c A B C : ℝ) (h1 : c = -3 * b * Real.cos A)
  (h2 : c = 2) (h3 : Real.tan C = 3 / 4) :
  ∃ S : ℝ, S = 1 / 2 * b * c * Real.sin A ∧ S = 4 / 3 := by
  sorry

end tan_ratio_triangle_area_l218_218227


namespace square_side_length_l218_218952

theorem square_side_length (s : ℝ) (h : s^2 = 12 * s) : s = 12 :=
by
  sorry

end square_side_length_l218_218952


namespace particle_paths_l218_218589

open Nat

-- Define the conditions of the problem
def move_right (a b : ℕ) : ℕ × ℕ := (a + 1, b)
def move_up (a b : ℕ) : ℕ × ℕ := (a, b + 1)
def move_diagonal (a b : ℕ) : ℕ × ℕ := (a + 1, b + 1)

-- Define a function to count paths without right-angle turns
noncomputable def count_paths (n : ℕ) : ℕ :=
  if n = 6 then 247 else 0

-- The theorem to be proven
theorem particle_paths :
  count_paths 6 = 247 :=
  sorry

end particle_paths_l218_218589


namespace find_expression_value_l218_218083

def g (p q r s : ℝ) (x : ℝ) : ℝ := p * x^3 + q * x^2 + r * x + s

theorem find_expression_value (p q r s : ℝ) (h1 : g p q r s (-1) = 2) (h2 : g p q r s (-2) = -1) (h3 : g p q r s (1) = -2) :
  9 * p - 3 * q + 3 * r - s = -2 :=
by
  sorry

end find_expression_value_l218_218083


namespace coffee_processing_completed_l218_218765

-- Define the initial conditions
def CoffeeBeansProcessed (m n : ℕ) : Prop :=
  let mass: ℝ := 1
  let days_single_machine: ℕ := 5
  let days_both_machines: ℕ := 4
  let half_mass: ℝ := mass / 2
  let total_ground_by_June_10 := (days_single_machine * m + days_both_machines * (m + n)) = half_mass
  total_ground_by_June_10

-- Define the final proof problem
theorem coffee_processing_completed (m n : ℕ) (h: CoffeeBeansProcessed m n) : ∃ d : ℕ, d = 15 := by
  -- Processed in 15 working days
  sorry

end coffee_processing_completed_l218_218765


namespace smallest_class_size_l218_218361

/--
In a science class, students are separated into five rows for an experiment. 
The class size must be greater than 50. 
Three rows have the same number of students, one row has two more students than the others, 
and another row has three more students than the others.
Prove that the smallest possible class size for this science class is 55.
-/
theorem smallest_class_size (class_size : ℕ) (n : ℕ) 
  (h1 : class_size = 3 * n + (n + 2) + (n + 3))
  (h2 : class_size > 50) :
  class_size = 55 :=
sorry

end smallest_class_size_l218_218361


namespace gcd_three_digit_numbers_l218_218786

theorem gcd_three_digit_numbers (a b c : ℕ) (h1 : b = a + 1) (h2 : c = a + 2) :
  ∃ k, (∀ n, n = 100 * a + 10 * b + c + 100 * c + 10 * b + a → n = 212 * k) :=
by
  sorry

end gcd_three_digit_numbers_l218_218786


namespace final_amount_l218_218648

-- Definitions for the initial amount, price per pound, and quantity purchased.
def initial_amount : ℕ := 20
def price_per_pound : ℕ := 2
def quantity_purchased : ℕ := 3

-- Formalizing the statement
theorem final_amount (A P Q : ℕ) (hA : A = initial_amount) (hP : P = price_per_pound) (hQ : Q = quantity_purchased) :
  A - P * Q = 14 :=
by
  sorry

end final_amount_l218_218648


namespace ratio_of_mustang_models_length_l218_218074

theorem ratio_of_mustang_models_length :
  ∀ (full_size_length mid_size_length smallest_model_length : ℕ),
    full_size_length = 240 →
    mid_size_length = full_size_length / 10 →
    smallest_model_length = 12 →
    smallest_model_length / mid_size_length = 1/2 :=
by
  intros full_size_length mid_size_length smallest_model_length h1 h2 h3
  sorry

end ratio_of_mustang_models_length_l218_218074


namespace find_k_l218_218447

theorem find_k : ∃ k : ℕ, ∀ n : ℕ, n > 0 → (2^n + 11) % (2^k - 1) = 0 ↔ k = 4 :=
by
  sorry

end find_k_l218_218447


namespace find_width_of_rectangle_l218_218277

-- Given conditions
variable (P l w : ℕ)
variable (h1 : P = 240)
variable (h2 : P = 3 * l)

-- Prove the width of the rectangular field is 40 meters
theorem find_width_of_rectangle : w = 40 :=
  by 
  -- Add the necessary logical steps here
  sorry

end find_width_of_rectangle_l218_218277


namespace problem_solution_l218_218034

open Set

theorem problem_solution
    (a b : ℝ)
    (ineq : ∀ x : ℝ, 1 < x ∧ x < b → a * x^2 - 3 * x + 2 < 0)
    (f : ℝ → ℝ := λ x => (2 * a + b) * x - 1 / ((a - b) * (x - 1))) :
    a = 1 ∧ b = 2 ∧ (∀ x, 1 < x ∧ x < b → f x ≥ 8 ∧ (f x = 8 ↔ x = 3 / 2)) :=
by
  sorry

end problem_solution_l218_218034


namespace minimum_area_of_triangle_is_sqrt_58_div_2_l218_218240

noncomputable def smallest_area_of_triangle (t s : ℝ) : ℝ :=
  (1/2) * Real.sqrt (5 * s^2 - 4 * s * t - 4 * s + 2 * t^2 + 10 * t + 13)

theorem minimum_area_of_triangle_is_sqrt_58_div_2 : ∃ t s : ℝ, smallest_area_of_triangle t s = Real.sqrt 58 / 2 := 
  by
  sorry

end minimum_area_of_triangle_is_sqrt_58_div_2_l218_218240


namespace triangle_condition_A_triangle_condition_B_triangle_condition_C_triangle_condition_D_problem_solution_l218_218187

def triangle (a b c : ℝ) : Prop := 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ a + c > b

def right_triangle (a b c : ℝ) : Prop := 
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem triangle_condition_A (a b c : ℝ) (h : triangle a b c) : 
  b^2 = (a + c) * (c - a) → right_triangle a c b := 
sorry

theorem triangle_condition_B (A B C : ℝ) (h : A + B + C = 180) : 
  A = B + C → 90 = A :=
sorry

theorem triangle_condition_C (A B C : ℝ) (h : A + B + C = 180) : 
  3 * (A / 12) = A ∧ 4 * (A / 12) = B ∧ 5 * (A / 12) = C → 
  ¬ (right_triangle A B C) :=
sorry

theorem triangle_condition_D : 
  right_triangle 6 8 10 := 
sorry

theorem problem_solution (a b c : ℝ) (A B C : ℝ) (hABC : triangle a b c) : 
  (b^2 = (a + c) * (c - a) → right_triangle a c b) ∧
  ((A + B + C = 180) ∧ (A = B + C) → 90 = A) ∧
  (3 * (A / 12) = A ∧ 4 * (A / 12) = B ∧ 5 * (A / 12) = C → ¬ right_triangle a b c) ∧
  (right_triangle 6 8 10) → 
  ∃ (cond : Prop), cond = (3 * (A / 12) = A ∧ 4 * (A / 12) = B ∧ 5 * (A / 12) = C) := 
sorry

end triangle_condition_A_triangle_condition_B_triangle_condition_C_triangle_condition_D_problem_solution_l218_218187


namespace set_equivalence_l218_218027

variable (M : Set ℕ)

theorem set_equivalence (h : M ∪ {1} = {1, 2, 3}) : M = {1, 2, 3} :=
sorry

end set_equivalence_l218_218027


namespace mass_scientific_notation_l218_218661

def mass := 37e-6

theorem mass_scientific_notation : mass = 3.7 * 10^(-5) :=
by
  sorry

end mass_scientific_notation_l218_218661


namespace ellen_lost_legos_l218_218830

theorem ellen_lost_legos (L_initial L_final : ℕ) (h1 : L_initial = 2080) (h2 : L_final = 2063) : L_initial - L_final = 17 := by
  sorry

end ellen_lost_legos_l218_218830


namespace find_smallest_b_l218_218427

theorem find_smallest_b :
  ∃ b : ℕ, 
    (∀ r s : ℤ, r * s = 3960 → r + s ≠ b ∨ r + s > 0) ∧ 
    (∀ r s : ℤ, r * s = 3960 → (r + s < b → r + s ≤ 0)) ∧ 
    b = 126 :=
by
  sorry

end find_smallest_b_l218_218427


namespace num_two_digit_palindromes_l218_218562

theorem num_two_digit_palindromes : 
  let is_palindrome (n : ℕ) : Prop := (n / 10) = (n % 10)
  ∃ n : ℕ, 10 ≤ n ∧ n < 90 ∧ is_palindrome n →
  ∃ count : ℕ, count = 9 := 
sorry

end num_two_digit_palindromes_l218_218562


namespace variance_of_given_data_is_2_l218_218368

-- Define the data set
def data_set : List ℕ := [198, 199, 200, 201, 202]

-- Define the mean function for a given data set
noncomputable def mean (data : List ℕ) : ℝ :=
  (data.sum : ℝ) / data.length

-- Define the variance function for a given data set
noncomputable def variance (data : List ℕ) : ℝ :=
  let μ := mean data
  (data.map (λ x => (x : ℝ) - μ) |>.map (λ x => x^2)).sum / data.length

-- Proposition that the variance of the given data set is 2
theorem variance_of_given_data_is_2 : variance data_set = 2 := by
  sorry

end variance_of_given_data_is_2_l218_218368


namespace compare_fractions_l218_218365

theorem compare_fractions : (-8 / 21: ℝ) > (-3 / 7: ℝ) :=
sorry

end compare_fractions_l218_218365


namespace number_of_diagonals_in_octagon_l218_218896

theorem number_of_diagonals_in_octagon :
  let n : ℕ := 8
  let num_diagonals := n * (n - 3) / 2
  num_diagonals = 20 := by
  sorry

end number_of_diagonals_in_octagon_l218_218896


namespace quadratic_to_general_form_l218_218079

theorem quadratic_to_general_form (x : ℝ) :
  ∃ b : ℝ, (∀ a c : ℝ, (a = 3) ∧ (c = 1) → (a * x^2 + c = 6 * x) → b = -6) :=
by
  sorry

end quadratic_to_general_form_l218_218079


namespace nancy_marks_home_economics_l218_218812

-- Definitions from conditions
def marks_american_lit := 66
def marks_history := 75
def marks_physical_ed := 68
def marks_art := 89
def average_marks := 70
def num_subjects := 5
def total_marks := average_marks * num_subjects
def marks_other_subjects := marks_american_lit + marks_history + marks_physical_ed + marks_art

-- Statement to prove
theorem nancy_marks_home_economics : 
  (total_marks - marks_other_subjects = 52) := by 
  sorry

end nancy_marks_home_economics_l218_218812


namespace ratio_problem_l218_218160

theorem ratio_problem
  (a b c d : ℝ)
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 2)
  (h3 : c / d = 49) :
  d / a = 1 / 122.5 :=
by {
  -- Proof steps would go here
  sorry
}

end ratio_problem_l218_218160


namespace total_weekly_allowance_l218_218004

theorem total_weekly_allowance
  (total_students : ℕ)
  (students_6dollar : ℕ)
  (students_4dollar : ℕ)
  (students_7dollar : ℕ)
  (allowance_6dollar : ℕ)
  (allowance_4dollar : ℕ)
  (allowance_7dollar : ℕ)
  (days_in_week : ℕ) :
  total_students = 100 →
  students_6dollar = 60 →
  students_4dollar = 25 →
  students_7dollar = 15 →
  allowance_6dollar = 6 →
  allowance_4dollar = 4 →
  allowance_7dollar = 7 →
  days_in_week = 7 →
  (students_6dollar * allowance_6dollar + students_4dollar * allowance_4dollar + students_7dollar * allowance_7dollar) * days_in_week = 3955 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end total_weekly_allowance_l218_218004


namespace absolute_value_equality_l218_218665

variables {a b c d : ℝ}

theorem absolute_value_equality (h1 : |a - b| + |c - d| = 99) (h2 : |a - c| + |b - d| = 1) : |a - d| + |b - c| = 99 :=
sorry

end absolute_value_equality_l218_218665


namespace plane_arrival_time_l218_218238

-- Define the conditions
def departure_time := 11 -- common departure time in hours (11:00)
def bus_speed := 100 -- bus speed in km/h
def train_speed := 300 -- train speed in km/h
def plane_speed := 900 -- plane speed in km/h
def bus_arrival := 20 -- bus arrival time in hours (20:00)
def train_arrival := 14 -- train arrival time in hours (14:00)

-- Given these conditions, we need to prove the plane arrival time
theorem plane_arrival_time : (departure_time + (900 / plane_speed)) = 12 := by
  sorry

end plane_arrival_time_l218_218238


namespace category_B_count_solution_hiring_probability_l218_218716

-- Definitions and conditions
def category_A_count : Nat := 12

def total_selected_housekeepers : Nat := 20
def category_B_selected_housekeepers : Nat := 16
def category_A_selected_housekeepers := total_selected_housekeepers - category_B_selected_housekeepers

-- The value of x
def category_B_count (x : Nat) : Prop :=
  (category_A_selected_housekeepers * x) / category_A_count = category_B_selected_housekeepers

-- Assertion for the value of x
theorem category_B_count_solution : category_B_count 48 :=
by sorry

-- Conditions for the second part of the problem
def remaining_category_A : Nat := 3
def remaining_category_B : Nat := 2
def total_remaining := remaining_category_A + remaining_category_B

def possible_choices := remaining_category_A * (remaining_category_A - 1) / 2 + remaining_category_A * remaining_category_B + remaining_category_B * (remaining_category_B - 1) / 2
def successful_choices := remaining_category_A * remaining_category_B

def probability (a b : Nat) := (successful_choices % total_remaining) / (possible_choices % total_remaining)

-- Assertion for the probability
theorem hiring_probability : probability remaining_category_A remaining_category_B = 3 / 5 :=
by sorry

end category_B_count_solution_hiring_probability_l218_218716


namespace children_getting_on_bus_l218_218848

theorem children_getting_on_bus (a b c: ℕ) (ha : a = 64) (hb : b = 78) (hc : c = b - a) : c = 14 :=
by
  sorry

end children_getting_on_bus_l218_218848


namespace speed_of_first_train_l218_218315

/-
Problem:
Two trains, with lengths 150 meters and 165 meters respectively, are running in opposite directions. One train is moving at 65 kmph, and they take 7.82006405004841 seconds to completely clear each other from the moment they meet. Prove that the speed of the first train is 79.99 kmph.
-/

theorem speed_of_first_train :
  ∀ (length1 length2 : ℝ) (speed2 : ℝ) (time : ℝ) (speed1 : ℝ),
  length1 = 150 → length2 = 165 → speed2 = 65 → time = 7.82006405004841 →
  ( 3.6 * (length1 + length2) / time = speed1 + speed2 ) →
  speed1 = 79.99 :=
by
  intros length1 length2 speed2 time speed1 h_length1 h_length2 h_speed2 h_time h_formula
  rw [h_length1, h_length2, h_speed2, h_time] at h_formula
  sorry

end speed_of_first_train_l218_218315


namespace math_problem_l218_218256

theorem math_problem (a b c : ℝ) (h1 : a + 2 * b + 3 * c = 12) (h2 : a^2 + b^2 + c^2 = a * b + b * c + c * a) : a + b^2 + c^3 = 14 :=
by
  sorry

end math_problem_l218_218256


namespace complex_multiplication_l218_218036

theorem complex_multiplication {i : ℂ} (h : i^2 = -1) : i * (1 - i) = 1 + i := 
by 
  sorry

end complex_multiplication_l218_218036


namespace find_p_l218_218942

theorem find_p (p q : ℚ) (h1 : 5 * p + 6 * q = 10) (h2 : 6 * p + 5 * q = 17) : p = 52 / 11 :=
by {
  -- Proof steps would go here
  sorry
}

end find_p_l218_218942


namespace apollonius_circle_equation_l218_218162

theorem apollonius_circle_equation (x y : ℝ) (A B : ℝ × ℝ) (hA : A = (2, 0)) (hB : B = (8, 0))
  (h : dist (x, y) A / dist (x, y) B = 1 / 2) : x^2 + y^2 = 16 := 
sorry

end apollonius_circle_equation_l218_218162


namespace ratio_and_tangent_l218_218833

-- Definitions for the problem
def acute_triangle (A B C : Point) : Prop := 
  -- acute angles condition
  sorry

def is_diameter (A B C D : Point) : Prop := 
  -- D is midpoint of BC condition
  sorry

def divide_in_half (A B C : Point) (D : Point) : Prop := 
  -- D divides BC in half condition
  sorry

def divide_in_ratio (A B C : Point) (D : Point) (ratio : ℚ) : Prop := 
  -- D divides AC in the given ratio condition
  sorry

def tan (angle : ℝ) : ℝ := 
  -- Tangent function
  sorry

def angle (A B C : Point) : ℝ := 
  -- Angle at B of triangle ABC
  sorry

-- The statement of the problem in Lean
theorem ratio_and_tangent (A B C D : Point) :
  acute_triangle A B C →
  is_diameter A B C D →
  divide_in_half A B C D →
  (divide_in_ratio A B C D (1 / 3) ↔ tan (angle A B C) = 2 * tan (angle A C B)) :=
by sorry

end ratio_and_tangent_l218_218833


namespace units_digit_difference_l218_218371

-- Conditions based on the problem statement
def units_digit_of_power_of_5 (n : ℕ) : ℕ := 5

def units_digit_of_power_of_3 (n : ℕ) : ℕ :=
  match n % 4 with
  | 0     => 1
  | 1     => 3
  | 2     => 9
  | 3     => 7
  | _     => 0  -- impossible due to mod 4

-- Problem statement in Lean as a theorem
theorem units_digit_difference : (5^2019 - 3^2019) % 10 = 8 :=
by
  have h1 : (5^2019 % 10) = units_digit_of_power_of_5 2019 := sorry
  have h2 : (3^2019 % 10) = units_digit_of_power_of_3 2019 := sorry
  -- The core proof step will go here
  sorry

end units_digit_difference_l218_218371


namespace sam_distinct_meals_count_l218_218072

-- Definitions based on conditions
def main_dishes := ["Burger", "Pasta", "Salad"]
def beverages := ["Soda", "Juice"]
def snacks := ["Chips", "Cookie", "Apple"]

-- Definition to exclude invalid combinations
def is_valid_combination (main : String) (beverage : String) : Bool :=
  if main = "Burger" && beverage = "Soda" then false else true

-- Number of valid combinations
def count_valid_meals : Nat :=
  main_dishes.length * beverages.length * snacks.length - snacks.length

theorem sam_distinct_meals_count : count_valid_meals = 15 := 
  sorry

end sam_distinct_meals_count_l218_218072


namespace find_a_l218_218532

theorem find_a (a : ℝ) (h : (1 / Real.log 2 / Real.log a) + (1 / Real.log 3 / Real.log a) + (1 / Real.log 5 / Real.log a) = 2) : a = Real.sqrt 30 := 
by 
  sorry

end find_a_l218_218532


namespace total_cars_for_sale_l218_218877

-- Define the conditions given in the problem
def salespeople : Nat := 10
def cars_per_salesperson_per_month : Nat := 10
def months : Nat := 5

-- Statement to prove the total number of cars for sale
theorem total_cars_for_sale : (salespeople * cars_per_salesperson_per_month) * months = 500 := by
  -- Proof goes here
  sorry

end total_cars_for_sale_l218_218877


namespace Vasya_has_larger_amount_l218_218367

-- Defining the conditions and given data
variables (V P : ℝ)

-- Vasya's profit calculation
def Vasya_profit (V : ℝ) : ℝ := 0.20 * V

-- Petya's profit calculation considering exchange rate increase
def Petya_profit (P : ℝ) : ℝ := 0.2045 * P

-- Proof statement
theorem Vasya_has_larger_amount (h : Vasya_profit V = Petya_profit P) : V > P :=
sorry

end Vasya_has_larger_amount_l218_218367


namespace last_ball_probability_l218_218962

variables (p q : ℕ)

def probability_white_last_ball (p : ℕ) : ℝ :=
  if p % 2 = 0 then 0 else 1

theorem last_ball_probability :
  ∀ {p q : ℕ},
    probability_white_last_ball p = if p % 2 = 0 then 0 else 1 :=
by
  intros
  sorry

end last_ball_probability_l218_218962


namespace ratio_of_sides_l218_218678

theorem ratio_of_sides (
  perimeter_triangle perimeter_square : ℕ)
  (h_triangle : perimeter_triangle = 48)
  (h_square : perimeter_square = 64) :
  (perimeter_triangle / 3) / (perimeter_square / 4) = 1 :=
by
  sorry

end ratio_of_sides_l218_218678


namespace simplify_expression_l218_218251

-- Define the problem and its conditions
theorem simplify_expression :
  (81 * 10^12) / (9 * 10^4) = 900000000 :=
by
  sorry  -- Proof placeholder

end simplify_expression_l218_218251


namespace simplify_powers_of_ten_l218_218337

theorem simplify_powers_of_ten :
  (10^0.4) * (10^0.5) * (10^0.2) * (10^(-0.6)) * (10^0.5) = 10 := 
by
  sorry

end simplify_powers_of_ten_l218_218337


namespace cab_speed_ratio_l218_218316

variable (S_u S_c : ℝ)

theorem cab_speed_ratio (h1 : ∃ S_u S_c : ℝ, S_u * 25 = S_c * 30) : S_c / S_u = 5 / 6 :=
by
  sorry

end cab_speed_ratio_l218_218316


namespace children_on_bus_after_events_l218_218275

-- Definition of the given problem parameters
def initial_children : Nat := 21
def got_off : Nat := 10
def got_on : Nat := 5

-- The theorem we want to prove
theorem children_on_bus_after_events : initial_children - got_off + got_on = 16 :=
by
  -- This is where the proof would go, but we leave it as sorry for now
  sorry

end children_on_bus_after_events_l218_218275


namespace total_logs_in_both_stacks_l218_218802

-- Define the number of logs in the first stack
def first_stack_logs : Nat :=
  let bottom_row := 15
  let top_row := 4
  let number_of_terms := bottom_row - top_row + 1
  let average_logs := (bottom_row + top_row) / 2
  average_logs * number_of_terms

-- Define the number of logs in the second stack
def second_stack_logs : Nat :=
  let bottom_row := 5
  let top_row := 10
  let number_of_terms := top_row - bottom_row + 1
  let average_logs := (bottom_row + top_row) / 2
  average_logs * number_of_terms

-- Prove the total number of logs in both stacks
theorem total_logs_in_both_stacks : first_stack_logs + second_stack_logs = 159 := by
  sorry

end total_logs_in_both_stacks_l218_218802


namespace men_in_first_group_l218_218545

variable (M : ℕ) (daily_wage : ℝ)
variable (h1 : M * 10 * daily_wage = 1200)
variable (h2 : 9 * 6 * daily_wage = 1620)
variable (dw_eq : daily_wage = 30)

theorem men_in_first_group : M = 4 :=
by sorry

end men_in_first_group_l218_218545


namespace unique_positive_integer_divisibility_l218_218456

theorem unique_positive_integer_divisibility (n : ℕ) (h : n > 0) : 
  (5^(n-1) + 3^(n-1)) ∣ (5^n + 3^n) ↔ n = 1 :=
by
  sorry

end unique_positive_integer_divisibility_l218_218456


namespace total_students_l218_218523

theorem total_students (p q r s : ℕ) 
  (h1 : 1 < p)
  (h2 : p < q)
  (h3 : q < r)
  (h4 : r < s)
  (h5 : p * q * r * s = 1365) :
  p + q + r + s = 28 :=
sorry

end total_students_l218_218523


namespace range_of_a_over_b_l218_218766

variable (a b : ℝ)

theorem range_of_a_over_b (ha : 1 < a ∧ a < 2) (hb : -2 < b ∧ b < -1) :
  -2 < a / b ∧ a / b < -1 / 2 :=
by
  sorry

end range_of_a_over_b_l218_218766


namespace square_side_length_theorem_l218_218984

-- Define the properties of the geometric configurations
def is_tangent_to_extension_segments (circle_radius : ℝ) (segment_length : ℝ) : Prop :=
  segment_length = circle_radius

def angle_between_tangents_from_point (angle : ℝ) : Prop :=
  angle = 60 

def square_side_length (side : ℝ) : Prop :=
  side = 4 * (Real.sqrt 2 - 1)

-- Main theorem
theorem square_side_length_theorem (circle_radius : ℝ) (segment_length : ℝ) (angle : ℝ) (side : ℝ)
  (h1 : is_tangent_to_extension_segments circle_radius segment_length)
  (h2 : angle_between_tangents_from_point angle) :
  square_side_length side :=
by
  sorry

end square_side_length_theorem_l218_218984


namespace yuan_older_than_david_l218_218234

theorem yuan_older_than_david (David_age : ℕ) (Yuan_age : ℕ) 
  (h1 : Yuan_age = 2 * David_age) 
  (h2 : David_age = 7) : 
  Yuan_age - David_age = 7 := by
  sorry

end yuan_older_than_david_l218_218234


namespace total_spent_on_birthday_presents_l218_218972

noncomputable def leonards_total_before_discount :=
  (3 * 35.50) + (2 * 120.75) + 44.25

noncomputable def leonards_total_after_discount :=
  leonards_total_before_discount - (0.10 * leonards_total_before_discount)

noncomputable def michaels_total_before_discount :=
  89.50 + (3 * 54.50) + 24.75

noncomputable def michaels_total_after_discount :=
  michaels_total_before_discount - (0.15 * michaels_total_before_discount)

noncomputable def emilys_total_before_tax :=
  (2 * 69.25) + (4 * 14.80)

noncomputable def emilys_total_after_tax :=
  emilys_total_before_tax + (0.08 * emilys_total_before_tax)

noncomputable def total_amount_spent :=
  leonards_total_after_discount + michaels_total_after_discount + emilys_total_after_tax

theorem total_spent_on_birthday_presents :
  total_amount_spent = 802.64 :=
by
  sorry

end total_spent_on_birthday_presents_l218_218972


namespace greatest_product_of_two_even_integers_whose_sum_is_300_l218_218671

theorem greatest_product_of_two_even_integers_whose_sum_is_300 :
  ∃ (x y : ℕ), (2 ∣ x) ∧ (2 ∣ y) ∧ (x + y = 300) ∧ (x * y = 22500) :=
by
  sorry

end greatest_product_of_two_even_integers_whose_sum_is_300_l218_218671


namespace find_number_l218_218949

theorem find_number (x : ℝ) (h : x + (2/3) * x + 1 = 10) : x = 27/5 := 
by
  sorry

end find_number_l218_218949


namespace polygon_sides_eq_four_l218_218430

theorem polygon_sides_eq_four (n : ℕ)
  (h_interior : (n - 2) * 180 = 360)
  (h_exterior : ∀ (m : ℕ), m = n -> 360 = 360) :
  n = 4 :=
sorry

end polygon_sides_eq_four_l218_218430


namespace scientific_notation_of_viewers_l218_218435

def million : ℝ := 10^6
def viewers : ℝ := 70.62 * million

theorem scientific_notation_of_viewers : viewers = 7.062 * 10^7 := by
  sorry

end scientific_notation_of_viewers_l218_218435


namespace calculate_x_l218_218664

theorem calculate_x : 121 + 2 * 11 * 8 + 64 = 361 :=
by
  sorry

end calculate_x_l218_218664


namespace four_four_four_digits_eight_eight_eight_digits_l218_218864

theorem four_four_four_digits_eight_eight_eight_digits (n : ℕ) :
  (4 * (10 ^ (n + 1) - 1) * (10 ^ n) + 8 * (10^n - 1) + 9) = 
  (6 * 10^n + 7) * (6 * 10^n + 7) :=
sorry

end four_four_four_digits_eight_eight_eight_digits_l218_218864


namespace solution_set_g_lt_6_range_of_values_a_l218_218358

-- Definitions
def f (a x : ℝ) : ℝ := 3 * |x - a| + |3 * x + 1|
def g (x : ℝ) : ℝ := |4 * x - 1| - |x + 2|

-- First part: solution set for g(x) < 6
theorem solution_set_g_lt_6 :
  {x : ℝ | g x < 6} = {x : ℝ | -7/5 < x ∧ x < 3} :=
sorry

-- Second part: range of values for a such that f(x1) and g(x2) are opposite numbers
theorem range_of_values_a (a : ℝ) :
  (∃ x1 x2 : ℝ, f a x1 = -g x2) → -13/12 ≤ a ∧ a ≤ 5/12 :=
sorry

end solution_set_g_lt_6_range_of_values_a_l218_218358


namespace simplify_fraction_l218_218104

theorem simplify_fraction (d : ℤ) : (5 + 4 * d) / 9 - 3 = (4 * d - 22) / 9 :=
by
  sorry

end simplify_fraction_l218_218104


namespace find_k_l218_218244

theorem find_k (x y k : ℝ) (h1 : 2 * x - y = 4) (h2 : k * x - 3 * y = 12) : k = 6 := by
  sorry

end find_k_l218_218244


namespace gumball_problem_l218_218693
-- Step d: Lean 4 statement conversion

/-- 
  Suppose Joanna initially had 40 gumballs, Jacques had 60 gumballs, 
  and Julia had 80 gumballs.
  Joanna purchased 5 times the number of gumballs she initially had,
  Jacques purchased 3 times the number of gumballs he initially had,
  and Julia purchased 2 times the number of gumballs she initially had.
  Prove that after adding their purchases:
  1. Each person will have 240 gumballs.
  2. If they combine all their gumballs and share them equally, 
     each person will still get 240 gumballs.
-/
theorem gumball_problem :
  let joanna_initial := 40 
  let jacques_initial := 60 
  let julia_initial := 80 
  let joanna_final := joanna_initial + 5 * joanna_initial 
  let jacques_final := jacques_initial + 3 * jacques_initial 
  let julia_final := julia_initial + 2 * julia_initial 
  let total_gumballs := joanna_final + jacques_final + julia_final 
  (joanna_final = 240) ∧ (jacques_final = 240) ∧ (julia_final = 240) ∧ 
  (total_gumballs / 3 = 240) :=
by
  let joanna_initial := 40 
  let jacques_initial := 60 
  let julia_initial := 80 
  let joanna_final := joanna_initial + 5 * joanna_initial 
  let jacques_final := jacques_initial + 3 * jacques_initial 
  let julia_final := julia_initial + 2 * julia_initial 
  let total_gumballs := joanna_final + jacques_final + julia_final 
  
  have h_joanna : joanna_final = 240 := sorry
  have h_jacques : jacques_final = 240 := sorry
  have h_julia : julia_final = 240 := sorry
  have h_total : total_gumballs / 3 = 240 := sorry
  
  exact ⟨h_joanna, h_jacques, h_julia, h_total⟩

end gumball_problem_l218_218693


namespace converse_l218_218354

variables {x : ℝ}

def P (x : ℝ) : Prop := x < 0
def Q (x : ℝ) : Prop := x^2 > 0

theorem converse (h : Q x) : P x :=
sorry

end converse_l218_218354


namespace salary_of_thomas_l218_218598

variable (R Ro T : ℕ)

theorem salary_of_thomas 
  (h1 : R + Ro = 8000) 
  (h2 : R + Ro + T = 15000) : 
  T = 7000 := by
  sorry

end salary_of_thomas_l218_218598


namespace three_digit_number_l218_218509

theorem three_digit_number (x y z : ℕ) 
  (h1: z^2 = x * y)
  (h2: y = (x + z) / 6)
  (h3: x - z = 4) :
  100 * x + 10 * y + z = 824 := 
by sorry

end three_digit_number_l218_218509


namespace sum_of_two_digit_divisors_l218_218330

theorem sum_of_two_digit_divisors (d : ℕ) (h_pos : d > 0) (h_mod : 145 % d = 4) : d = 47 := 
by sorry

end sum_of_two_digit_divisors_l218_218330


namespace bottles_left_after_purchase_l218_218587

def initial_bottles : ℕ := 35
def jason_bottles : ℕ := 5
def harry_bottles : ℕ := 6
def jason_effective_bottles (n : ℕ) : ℕ := n  -- Jason buys 5 bottles
def harry_effective_bottles (n : ℕ) : ℕ := n + 1 -- Harry gets one additional free bottle

theorem bottles_left_after_purchase (j_b h_b i_b : ℕ) (j_effective h_effective : ℕ → ℕ) :
  j_b = 5 → h_b = 6 → i_b = 35 → j_effective j_b = 5 → h_effective h_b = 7 →
  i_b - (j_effective j_b + h_effective h_b) = 23 :=
by
  intros
  sorry

end bottles_left_after_purchase_l218_218587


namespace pencils_before_buying_l218_218030

theorem pencils_before_buying (x total bought : Nat) 
  (h1 : bought = 7) 
  (h2 : total = 10) 
  (h3 : total = x + bought) : x = 3 :=
by
  sorry

end pencils_before_buying_l218_218030


namespace add_second_largest_to_sum_l218_218396

def is_valid_digit (d : ℕ) : Prop := d = 2 ∨ d = 5 ∨ d = 8

def form_number (d1 d2 d3 : ℕ) : ℕ := 100 * d1 + 10 * d2 + d3

def largest_number : ℕ := form_number 8 5 2
def smallest_number : ℕ := form_number 2 5 8
def second_largest_number : ℕ := form_number 8 2 5

theorem add_second_largest_to_sum : 
  second_largest_number + (largest_number + smallest_number) = 1935 := 
  sorry

end add_second_largest_to_sum_l218_218396


namespace min_vertical_segment_length_l218_218892

noncomputable def minVerticalSegLength : ℤ → ℝ 
| x => abs (2 * abs x + x^2 + 4 * x + 1)

theorem min_vertical_segment_length :
  ∀ x : ℤ, minVerticalSegLength x = 1 ↔  x = 0 := 
by
  intros x
  sorry

end min_vertical_segment_length_l218_218892


namespace angle_sum_proof_l218_218957

theorem angle_sum_proof (x α β : ℝ) (h1 : 3 * x + 4 * x + α = 180)
 (h2 : α + 5 * x + β = 180)
 (h3 : 2 * x + 2 * x + 6 * x = 180) :
  x = 18 := by
  sorry

end angle_sum_proof_l218_218957


namespace positive_integer_solutions_count_l218_218684

theorem positive_integer_solutions_count :
  ∃ (s : Finset ℕ), (∀ x ∈ s, 24 < (x + 2) ^ 2 ∧ (x + 2) ^ 2 < 64) ∧ s.card = 4 := 
by
  sorry

end positive_integer_solutions_count_l218_218684


namespace delta_discount_percentage_l218_218129

theorem delta_discount_percentage (original_delta : ℝ) (original_united : ℝ)
  (united_discount_percent : ℝ) (savings : ℝ) (delta_discounted : ℝ) : 
  original_delta - delta_discounted = 0.2 * original_delta := by
  -- Given conditions
  let discounted_united := original_united * (1 - united_discount_percent / 100)
  have : delta_discounted = discounted_united - savings := sorry
  let delta_discount_amount := original_delta - delta_discounted
  have : delta_discount_amount = 0.2 * original_delta := sorry
  exact this

end delta_discount_percentage_l218_218129


namespace ratio_of_boys_to_girls_l218_218033

variable {α β γ : ℝ}
variable (x y : ℕ)

theorem ratio_of_boys_to_girls (hα : α ≠ 1/2) (hprob : (x * β + y * γ) / (x + y) = 1/2) :
  (x : ℝ) / (y : ℝ) = (1/2 - γ) / (β - 1/2) :=
by
  sorry

end ratio_of_boys_to_girls_l218_218033


namespace pieces_of_gum_l218_218181

variable (initial_gum total_gum given_gum : ℕ)

theorem pieces_of_gum (h1 : given_gum = 16) (h2 : total_gum = 54) : initial_gum = 38 :=
by
  sorry

end pieces_of_gum_l218_218181


namespace initial_ripe_peaches_l218_218014

theorem initial_ripe_peaches (P U R: ℕ) (H1: P = 18) (H2: 2 * 5 = 10) (H3: (U + 7) + U = 15 - 3) (H4: R + 10 = U + 7) : 
  R = 1 :=
by
  sorry

end initial_ripe_peaches_l218_218014


namespace solve_system_of_inequalities_l218_218351

theorem solve_system_of_inequalities (x : ℝ) :
  (x + 1 < 5) ∧ (2 * x - 1) / 3 ≥ 1 ↔ 2 ≤ x ∧ x < 4 :=
by
  sorry

end solve_system_of_inequalities_l218_218351


namespace total_loss_l218_218145

theorem total_loss (P : ℝ) (A : ℝ) (L : ℝ) (h1 : A = (1/9) * P) (h2 : 603 = (P / (A + P)) * L) : 
  L = 670 :=
by
  sorry

end total_loss_l218_218145


namespace arithmetic_sequence_sum_l218_218815

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (d : ℤ)
    (h₁ : ∀ n, a (n + 1) = a n + d)
    (h₂ : a 3 + a 5 + a 7 + a 9 + a 11 = 20) : a 1 + a 13 = 8 := 
by 
  sorry

end arithmetic_sequence_sum_l218_218815


namespace distance_between_cities_l218_218720

-- Definitions
def map_distance : ℝ := 120 -- Distance on the map in cm
def scale_factor : ℝ := 10  -- Scale factor in km per cm

-- Theorem statement
theorem distance_between_cities :
  map_distance * scale_factor = 1200 :=
by
  sorry

end distance_between_cities_l218_218720


namespace calculation_result_l218_218286

theorem calculation_result :
  (10 * 19 * 20 * 53 * 100 + 601) / 13 = 1549277 :=
by 
  sorry

end calculation_result_l218_218286


namespace brandon_businesses_l218_218951

theorem brandon_businesses (total_businesses: ℕ) (fire_fraction: ℚ) (quit_fraction: ℚ) 
  (h_total: total_businesses = 72) 
  (h_fire_fraction: fire_fraction = 1/2) 
  (h_quit_fraction: quit_fraction = 1/3) : 
  total_businesses - (total_businesses * fire_fraction + total_businesses * quit_fraction) = 12 :=
by 
  sorry

end brandon_businesses_l218_218951


namespace sum_of_primes_is_prime_l218_218816

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k : ℕ, k ∣ n → k = 1 ∨ k = n

theorem sum_of_primes_is_prime (P Q : ℕ) :
  is_prime P → is_prime Q → is_prime (P - Q) → is_prime (P + Q) →
  ∃ n : ℕ, n = P + Q + (P - Q) + (P + Q) ∧ is_prime n := by
  sorry

end sum_of_primes_is_prime_l218_218816


namespace calculate_principal_amount_l218_218919

theorem calculate_principal_amount (P : ℝ) (h1 : P * 0.1025 - P * 0.1 = 25) : 
  P = 10000 :=
by
  sorry

end calculate_principal_amount_l218_218919


namespace problem1_problem2_l218_218679

def f (x : ℝ) : ℝ := |x - 3|

theorem problem1 :
  {x : ℝ | f x < 2 + |x + 1|} = {x : ℝ | 0 < x} := sorry

theorem problem2 (m n : ℝ) (h_mn : m > 0) (h_nn : n > 0) (h : (1 / m) + (1 / n) = 2 * m * n) :
  m * f n + n * f (-m) ≥ 6 := sorry

end problem1_problem2_l218_218679


namespace tens_digit_of_square_ending_in_six_odd_l218_218006

theorem tens_digit_of_square_ending_in_six_odd 
   (N : ℤ) 
   (a : ℤ) 
   (b : ℕ) 
   (hle : 0 ≤ b) 
   (hge : b < 10) 
   (hexp : N = 10 * a + b) 
   (hsqr : (N^2) % 10 = 6) : 
   ∃ k : ℕ, (N^2 / 10) % 10 = 2 * k + 1 :=
sorry -- Proof goes here

end tens_digit_of_square_ending_in_six_odd_l218_218006


namespace ellipse_equation_l218_218376

theorem ellipse_equation
  (P : ℝ × ℝ)
  (a b c : ℝ)
  (h1 : a > b ∧ b > 0)
  (h2 : 2 * a = 5 + 3)
  (h3 : (2 * c) ^ 2 = 5 ^ 2 - 3 ^ 2)
  (h4 : P.1 ^ 2 / a ^ 2 + P.2 ^ 2 / b ^ 2 = 1 ∨ P.2 ^ 2 / a ^ 2 + P.1 ^ 2 / b ^ 2 = 1)
  : ((a = 4) ∧ (c = 2) ∧ (b ^ 2 = 12) ∧
    (P.1 ^ 2 / 16 + P.2 ^ 2 / 12 = 1) ∨
    (P.2 ^ 2 / 16 + P.1 ^ 2 / 12 = 1)) :=
sorry

end ellipse_equation_l218_218376


namespace determine_y_increase_volume_l218_218102

noncomputable def volume_increase_y (r h y : ℝ) : Prop :=
  (1/3) * Real.pi * (r + y)^2 * h = (1/3) * Real.pi * r^2 * (h + y)

theorem determine_y_increase_volume (y : ℝ) :
  volume_increase_y 5 12 y ↔ y = 31 / 12 :=
by
  sorry

end determine_y_increase_volume_l218_218102


namespace S_13_eq_3510_l218_218317

def S (n : ℕ) : ℕ := n * (n + 2) * (n + 4) + n * (n + 2)

theorem S_13_eq_3510 : S 13 = 3510 :=
by
  sorry

end S_13_eq_3510_l218_218317


namespace minimum_roots_in_interval_l218_218271

noncomputable def g : ℝ → ℝ := sorry

lemma symmetry_condition_1 (x : ℝ) : g (3 + x) = g (3 - x) := sorry
lemma symmetry_condition_2 (x : ℝ) : g (8 + x) = g (8 - x) := sorry
lemma initial_condition : g 1 = 0 := sorry

theorem minimum_roots_in_interval : 
  ∃ k, ∀ x, -1000 ≤ x ∧ x ≤ 1000 → g x = 0 ∧ 
  (2 * k) = 286 := sorry

end minimum_roots_in_interval_l218_218271


namespace sum_of_digits_3n_l218_218453

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem sum_of_digits_3n (n : ℕ) (hn1 : digit_sum n = 100) (hn2 : digit_sum (44 * n) = 800) : digit_sum (3 * n) = 300 := by
  sorry

end sum_of_digits_3n_l218_218453


namespace find_n_that_satisfies_l218_218961

theorem find_n_that_satisfies :
  ∃ (n : ℕ), (1 / (n + 2 : ℕ) + 2 / (n + 2) + (n + 1) / (n + 2) = 2) ∧ (n = 0) :=
by 
  existsi (0 : ℕ)
  sorry

end find_n_that_satisfies_l218_218961


namespace dave_bought_packs_l218_218009

def packs_of_white_shirts (bought_total : ℕ) (white_per_pack : ℕ) (blue_packs : ℕ) (blue_per_pack : ℕ) : ℕ :=
  (bought_total - blue_packs * blue_per_pack) / white_per_pack

theorem dave_bought_packs : packs_of_white_shirts 26 6 2 4 = 3 :=
by
  sorry

end dave_bought_packs_l218_218009


namespace find_f2_l218_218076

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x + a^(-x)

theorem find_f2 (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 1 = 3) : f a 2 = 7 := 
by 
  sorry

end find_f2_l218_218076


namespace liz_three_pointers_l218_218985

-- Define the points scored by Liz's team in the final quarter.
def points_scored_by_liz (free_throws jump_shots three_pointers : ℕ) : ℕ :=
  free_throws * 1 + jump_shots * 2 + three_pointers * 3

-- Define the points needed to tie the game.
def points_needed_to_tie (initial_deficit points_lost other_team_points : ℕ) : ℕ :=
  points_lost + (initial_deficit - points_lost) + other_team_points

-- The total points scored by Liz from free throws and jump shots.
def liz_regular_points (free_throws jump_shots : ℕ) : ℕ :=
  free_throws * 1 + jump_shots * 2

theorem liz_three_pointers :
  ∀ (free_throws jump_shots liz_team_deficit_final quarter_deficit other_team_points liz_team_deficit_end final_deficit : ℕ),
    liz_team_deficit_final = 20 →
    free_throws = 5 →
    jump_shots = 4 →
    other_team_points = 10 →
    liz_team_deficit_end = 8 →
    final_deficit = liz_team_deficit_final - liz_team_deficit_end →
    (free_throws * 1 + jump_shots * 2 + 3 * final_deficit) = 
      points_needed_to_tie 20 other_team_points 8 →
    (3 * final_deficit) = 9 →
    final_deficit = 3 →
    final_deficit = 3 :=
by
  intros 
  try sorry

end liz_three_pointers_l218_218985


namespace price_of_jumbo_pumpkin_l218_218457

theorem price_of_jumbo_pumpkin (total_pumpkins : ℕ) (total_revenue : ℝ)
  (regular_pumpkins : ℕ) (price_regular : ℝ)
  (sold_jumbo_pumpkins : ℕ) (revenue_jumbo : ℝ): 
  total_pumpkins = 80 →
  total_revenue = 395.00 →
  regular_pumpkins = 65 →
  price_regular = 4.00 →
  sold_jumbo_pumpkins = total_pumpkins - regular_pumpkins →
  revenue_jumbo = total_revenue - (price_regular * regular_pumpkins) →
  revenue_jumbo / sold_jumbo_pumpkins = 9.00 :=
by
  intro h_total_pumpkins
  intro h_total_revenue
  intro h_regular_pumpkins
  intro h_price_regular
  intro h_sold_jumbo_pumpkins
  intro h_revenue_jumbo
  sorry

end price_of_jumbo_pumpkin_l218_218457


namespace total_balloons_correct_l218_218940

-- Define the number of balloons each person has
def dan_balloons : ℕ := 29
def tim_balloons : ℕ := 7 * dan_balloons
def molly_balloons : ℕ := 5 * dan_balloons

-- Define the total number of balloons
def total_balloons : ℕ := dan_balloons + tim_balloons + molly_balloons

-- The theorem to prove
theorem total_balloons_correct : total_balloons = 377 :=
by
  -- This part is where the proof will go
  sorry

end total_balloons_correct_l218_218940


namespace area_of_park_l218_218093

theorem area_of_park (L B : ℝ) (h1 : L / B = 1 / 3) (h2 : 12 * 1000 / 60 * 4 = 2 * (L + B)) : 
  L * B = 30000 :=
by
  sorry

end area_of_park_l218_218093


namespace smallest_d0_l218_218180

theorem smallest_d0 (r : ℕ) (hr : r ≥ 3) : ∃ d₀, d₀ = 2^(r - 2) ∧ (7^d₀ ≡ 1 [MOD 2^r]) :=
by
  sorry

end smallest_d0_l218_218180


namespace andreas_living_room_floor_area_l218_218719

-- Definitions
def carpet_length : ℝ := 4
def carpet_width : ℝ := 9
def carpet_coverage_percentage : ℝ := 0.30
def carpet_area : ℝ := carpet_length * carpet_width

-- Theorem statement
theorem andreas_living_room_floor_area (A : ℝ) 
  (h1 : carpet_coverage_percentage * A = carpet_area) :
  A = 120 :=
by
  sorry

end andreas_living_room_floor_area_l218_218719


namespace points_deducted_for_incorrect_answer_is_5_l218_218304

-- Define the constants and variables used in the problem
def total_questions : ℕ := 30
def points_per_correct_answer : ℕ := 20
def correct_answers : ℕ := 19
def incorrect_answers : ℕ := total_questions - correct_answers
def final_score : ℕ := 325

-- Define a function that models the total score calculation
def calculate_final_score (points_deducted_per_incorrect : ℕ) : ℕ :=
  (correct_answers * points_per_correct_answer) - (incorrect_answers * points_deducted_per_incorrect)

-- The theorem that states the problem and expected solution
theorem points_deducted_for_incorrect_answer_is_5 :
  ∃ (x : ℕ), calculate_final_score x = final_score ∧ x = 5 :=
by
  sorry

end points_deducted_for_incorrect_answer_is_5_l218_218304


namespace probability_two_boys_l218_218300

-- Definitions for the conditions
def total_students : ℕ := 4
def boys : ℕ := 3
def girls : ℕ := 1
def select_students : ℕ := 2

-- Combination function definition
def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_two_boys :
  (combination boys select_students) / (combination total_students select_students) = 1 / 2 := by
  sorry

end probability_two_boys_l218_218300


namespace rho_square_max_value_l218_218113

variable {a b x y c : ℝ}
variable (ha_pos : a > 0) (hb_pos : b > 0)
variable (ha_ge_b : a ≥ b)
variable (hx_range : 0 ≤ x ∧ x < a)
variable (hy_range : 0 ≤ y ∧ y < b)
variable (h_eq1 : a^2 + y^2 = b^2 + x^2)
variable (h_eq2 : b^2 + x^2 = (a - x)^2 + (b - y)^2 + c^2)

theorem rho_square_max_value : (a / b) ^ 2 ≤ 4 / 3 :=
sorry

end rho_square_max_value_l218_218113


namespace Jeremy_songs_l218_218349

theorem Jeremy_songs (songs_yesterday : ℕ) (songs_difference : ℕ) (songs_today : ℕ) (total_songs : ℕ) :
  songs_yesterday = 9 ∧ songs_difference = 5 ∧ songs_today = songs_yesterday + songs_difference ∧ 
  total_songs = songs_yesterday + songs_today → total_songs = 23 :=
by
  intros h
  sorry

end Jeremy_songs_l218_218349


namespace sum_of_values_of_N_l218_218886

-- Given conditions
variables (N R : ℝ)
-- Condition that needs to be checked
def condition (N R : ℝ) : Prop := N + 3 / N = R ∧ N ≠ 0

-- The statement to prove
theorem sum_of_values_of_N (N R : ℝ) (h: condition N R) : N + (3 / N) = R :=
sorry

end sum_of_values_of_N_l218_218886


namespace rectangle_area_l218_218706

variable (l w : ℕ)

def length_is_three_times_width := l = 3 * w

def perimeter_is_160 := 2 * l + 2 * w = 160

theorem rectangle_area : 
  length_is_three_times_width l w → 
  perimeter_is_160 l w → 
  l * w = 1200 :=
by
  intros h₁ h₂
  sorry

end rectangle_area_l218_218706


namespace acute_triangle_B_area_l218_218682

-- Basic setup for the problem statement
variable {A B C : ℝ} -- Angles of the triangle
variable {a b c : ℝ} -- Sides opposite to respective angles

-- The theorem to be proven
theorem acute_triangle_B_area (h_acute : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2) 
                              (h_sides : a = 2 * b * Real.sin A)
                              (h_a : a = 3 * Real.sqrt 3) 
                              (h_c : c = 5) : 
  B = π / 6 ∧ (1/2) * a * c * Real.sin B = (15 * Real.sqrt 3) / 4 :=
by
  sorry

end acute_triangle_B_area_l218_218682


namespace find_original_number_l218_218302

theorem find_original_number (N : ℕ) (h : ∃ k : ℕ, N - 5 = 13 * k) : N = 18 :=
sorry

end find_original_number_l218_218302


namespace remainder_mod_of_a_squared_subtract_3b_l218_218095

theorem remainder_mod_of_a_squared_subtract_3b (a b : ℕ) (h₁ : a % 7 = 2) (h₂ : b % 7 = 5) (h₃ : a^2 > 3 * b) : 
  (a^2 - 3 * b) % 7 = 3 := 
sorry

end remainder_mod_of_a_squared_subtract_3b_l218_218095


namespace find_x_l218_218746
-- Lean 4 equivalent problem setup

-- Assuming a and b are the tens and units digits respectively.
def number (a b : ℕ) := 10 * a + b
def interchangedNumber (a b : ℕ) := 10 * b + a
def digitsDifference (a b : ℕ) := a - b

-- Given conditions
variable (a b k : ℕ)

def condition1 := number a b = k * digitsDifference a b
def condition2 (x : ℕ) := interchangedNumber a b = x * digitsDifference a b

-- Theorem to prove
theorem find_x (h1 : condition1 a b k) : ∃ x, condition2 a b x ∧ x = k - 9 := 
by sorry

end find_x_l218_218746


namespace total_games_single_elimination_l218_218127

theorem total_games_single_elimination (teams : ℕ) (h_teams : teams = 24)
  (preliminary_matches : ℕ) (h_preliminary_matches : preliminary_matches = 8)
  (preliminary_teams : ℕ) (h_preliminary_teams : preliminary_teams = 16)
  (idle_teams : ℕ) (h_idle_teams : idle_teams = 8)
  (main_draw_teams : ℕ) (h_main_draw_teams : main_draw_teams = 16) :
  (games : ℕ) -> games = 23 :=
by
  sorry

end total_games_single_elimination_l218_218127


namespace find_k_l218_218841

def equation (k : ℝ) (x : ℝ) : Prop := 2 * x^2 + 3 * x - k = 0

theorem find_k (k : ℝ) (h : equation k 7) : k = 119 :=
by
  sorry

end find_k_l218_218841


namespace cole_round_trip_time_l218_218155

/-- Prove that the total round trip time is 2 hours given the conditions -/
theorem cole_round_trip_time :
  ∀ (speed_to_work : ℝ) (speed_back_home : ℝ) (time_to_work_min : ℝ),
  speed_to_work = 50 → speed_back_home = 110 → time_to_work_min = 82.5 →
  ((time_to_work_min / 60) * speed_to_work + (time_to_work_min * speed_to_work / speed_back_home) / 60) = 2 :=
by
  intros
  sorry

end cole_round_trip_time_l218_218155


namespace smallest_possible_Y_l218_218117

def digits (n : ℕ) : List ℕ := -- hypothetical function to get the digits of a number
  sorry

def is_divisible (n d : ℕ) : Prop := d ∣ n

theorem smallest_possible_Y :
  ∃ (U : ℕ), (∀ d ∈ digits U, d = 0 ∨ d = 1) ∧ is_divisible U 18 ∧ U / 18 = 61728395 :=
by
  sorry

end smallest_possible_Y_l218_218117


namespace Jason_earned_60_dollars_l218_218674

-- Define initial and final amounts of money
variable (Jason_initial Jason_final : ℕ)

-- State the assumption about Jason's initial and final amounts of money
variable (h_initial : Jason_initial = 3) (h_final : Jason_final = 63)

-- Define the amount of money Jason earned
def Jason_earn := Jason_final - Jason_initial

-- Prove that Jason earned 60 dollars by delivering newspapers
theorem Jason_earned_60_dollars : Jason_earn Jason_initial Jason_final = 60 := by
  sorry

end Jason_earned_60_dollars_l218_218674


namespace probability_function_has_zero_point_l218_218631

noncomputable def probability_of_zero_point : ℚ :=
by
  let S := ({-1, 1, 2} : Finset ℤ).product ({-1, 1, 2} : Finset ℤ)
  let zero_point_pairs := S.filter (λ p => (p.1 * p.2 ≤ 1))
  let favorable_outcomes := zero_point_pairs.card
  let total_outcomes := S.card
  exact favorable_outcomes / total_outcomes

theorem probability_function_has_zero_point :
  probability_of_zero_point = (2 / 3 : ℚ) :=
  sorry

end probability_function_has_zero_point_l218_218631


namespace problem_statement_l218_218264

theorem problem_statement (n : ℕ) (h : ∀ (a b : ℕ), ¬ (n ∣ (2^a * 3^b + 1))) :
  ∀ (c d : ℕ), ¬ (n ∣ (2^c + 3^d)) := by
  sorry

end problem_statement_l218_218264


namespace andy_demerits_l218_218313

theorem andy_demerits (x : ℕ) :
  (∀ x, 6 * x + 15 = 27 → x = 2) :=
by
  intro
  sorry

end andy_demerits_l218_218313


namespace ratio_comparison_l218_218544

-- Define the ratios in the standard and sport formulations
def ratio_flavor_corn_standard : ℚ := 1 / 12
def ratio_flavor_water_standard : ℚ := 1 / 30
def ratio_flavor_water_sport : ℚ := 1 / 60

-- Define the amounts of corn syrup and water in the sport formulation
def corn_syrup_sport : ℚ := 2
def water_sport : ℚ := 30

-- Calculate the amount of flavoring in the sport formulation
def flavoring_sport : ℚ := water_sport / 60

-- Calculate the ratio of flavoring to corn syrup in the sport formulation
def ratio_flavor_corn_sport : ℚ := flavoring_sport / corn_syrup_sport

-- Define the theorem to prove the ratio comparison
theorem ratio_comparison :
  (ratio_flavor_corn_sport / ratio_flavor_corn_standard) = 3 :=
by
  -- Using the given conditions and definitions, prove the theorem
  sorry

end ratio_comparison_l218_218544


namespace absolute_value_equation_solution_l218_218613

theorem absolute_value_equation_solution (x : ℝ) : |x - 30| + |x - 24| = |3 * x - 72| ↔ x = 26 :=
by sorry

end absolute_value_equation_solution_l218_218613


namespace least_subtracted_correct_second_num_correct_l218_218846

-- Define the given numbers
def given_num : ℕ := 1398
def remainder : ℕ := 5
def num1 : ℕ := 7
def num2 : ℕ := 9
def num3 : ℕ := 11

-- Least number to subtract to satisfy the condition
def least_subtracted : ℕ := 22

-- Second number in the sequence
def second_num : ℕ := 2069

-- Define the hypotheses and statements to be proved
theorem least_subtracted_correct : given_num - least_subtracted ≡ remainder [MOD num1]
∧ given_num - least_subtracted ≡ remainder [MOD num2]
∧ given_num - least_subtracted ≡ remainder [MOD num3] := sorry

theorem second_num_correct : second_num ≡ remainder [MOD num1 * num2 * num3] := sorry

end least_subtracted_correct_second_num_correct_l218_218846


namespace evaluate_expression_l218_218809

def operation (x y : ℚ) : ℚ := x^2 / y

theorem evaluate_expression : 
  (operation (operation 3 4) 2) - (operation 3 (operation 4 2)) = 45 / 32 :=
by
  sorry

end evaluate_expression_l218_218809


namespace ratio_fraction_l218_218497

theorem ratio_fraction (x : ℚ) : x = 2 / 9 ↔ (2 / 6) / x = (3 / 4) / (1 / 2) := by
  sorry

end ratio_fraction_l218_218497


namespace Jane_indisposed_days_l218_218736

-- Definitions based on conditions
def John_completion_days := 18
def Jane_completion_days := 12
def total_task_days := 10.8
def work_per_day_by_john := 1 / John_completion_days
def work_per_day_by_jane := 1 / Jane_completion_days
def work_per_day_together := work_per_day_by_john + work_per_day_by_jane

-- Equivalent proof problem
theorem Jane_indisposed_days : 
  ∃ (x : ℝ), 
    (10.8 - x) * work_per_day_together + x * work_per_day_by_john = 1 ∧
    x = 6 := 
by 
  sorry

end Jane_indisposed_days_l218_218736


namespace part1_part2_l218_218907

variable {A B C a b c : ℝ}

-- Part (1): Prove that 2a^2 = b^2 + c^2 given the condition
theorem part1 (h : Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A)) : 2 * a^2 = b^2 + c^2 := 
sorry

-- Part (2): Prove the perimeter of triangle ABC
theorem part2 (a : ℝ) (h_a : a = 5) (cosA : ℝ) (h_cosA : cosA = 25 / 31) : 5 + b + c = 14 := 
sorry

end part1_part2_l218_218907


namespace solve_equation_l218_218260

theorem solve_equation (x : ℝ) (h1 : 2 * x + 1 ≠ 0) (h2 : 4 * x ≠ 0) : 
  (3 / (2 * x + 1) = 5 / (4 * x)) ↔ (x = 2.5) :=
by 
  sorry

end solve_equation_l218_218260


namespace f_value_third_quadrant_l218_218854

noncomputable def f (α : ℝ) : ℝ :=
  (Real.cos (Real.pi / 2 + α) * Real.cos (2 * Real.pi - α) * Real.sin (-α + 3 * Real.pi / 2)) /
  (Real.sin (-Real.pi - α) * Real.sin (3 * Real.pi / 2 + α))

theorem f_value_third_quadrant (α : ℝ) (h1 : (3 * Real.pi / 2 < α ∧ α < 2 * Real.pi)) (h2 : Real.cos (α - 3 * Real.pi / 2) = 1 / 5) :
  f α = 2 * Real.sqrt 6 / 5 :=
sorry

end f_value_third_quadrant_l218_218854


namespace max_value_of_a_l218_218305

noncomputable def maximum_a : ℝ := 1/3

theorem max_value_of_a :
  ∀ x : ℝ, 1 + maximum_a * Real.cos x ≥ (2/3) * Real.sin ((Real.pi / 2) + 2 * x) :=
by 
  sorry

end max_value_of_a_l218_218305


namespace train_speed_is_72_km_per_hr_l218_218738

-- Define the conditions
def length_of_train : ℕ := 180   -- Length in meters
def time_to_cross_pole : ℕ := 9  -- Time in seconds

-- Conversion factor
def conversion_factor : ℝ := 3.6

-- Prove that the speed of the train is 72 km/hr
theorem train_speed_is_72_km_per_hr :
  (length_of_train / time_to_cross_pole) * conversion_factor = 72 := by
  sorry

end train_speed_is_72_km_per_hr_l218_218738


namespace largest_expression_l218_218658

def U := 2 * 2004^2005
def V := 2004^2005
def W := 2003 * 2004^2004
def X := 2 * 2004^2004
def Y := 2004^2004
def Z := 2004^2003

theorem largest_expression :
  U - V > V - W ∧
  U - V > W - X ∧
  U - V > X - Y ∧
  U - V > Y - Z :=
by
  sorry

end largest_expression_l218_218658


namespace solution_set_inequality_l218_218350

variable (a b c : ℝ)
variable (condition1 : ∀ x : ℝ, ax^2 + bx + c < 0 ↔ x < -1 ∨ 2 < x)

theorem solution_set_inequality (h : a < 0 ∧ b = -a ∧ c = -2 * a) :
  ∀ x : ℝ, (bx^2 + ax - c ≤ 0) ↔ (-1 ≤ x ∧ x ≤ 2) :=
by
  intro x
  sorry

end solution_set_inequality_l218_218350


namespace wheel_radius_increase_proof_l218_218197

noncomputable def radius_increase (orig_distance odometer_distance : ℝ) (orig_radius : ℝ) : ℝ :=
  let orig_circumference := 2 * Real.pi * orig_radius
  let distance_per_rotation := orig_circumference / 63360
  let num_rotations_orig := orig_distance / distance_per_rotation
  let num_rotations_new := odometer_distance / distance_per_rotation
  let new_distance := orig_distance
  let new_radius := (new_distance / num_rotations_new) * 63360 / (2 * Real.pi)
  new_radius - orig_radius

theorem wheel_radius_increase_proof :
  radius_increase 600 580 16 = 0.42 :=
by 
  -- The proof is skipped.
  sorry

end wheel_radius_increase_proof_l218_218197


namespace min_expression_value_2023_l218_218609

noncomputable def min_expr_val := ∀ x : ℝ, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024 ≥ 2023

noncomputable def least_value : ℝ := 2023

theorem min_expression_value_2023 : min_expr_val ∧ (∃ x : ℝ, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024 = least_value) := 
by sorry

end min_expression_value_2023_l218_218609


namespace value_of_f_at_2_l218_218840

def f (x : ℝ) : ℝ := x^2 - 3 * x

theorem value_of_f_at_2 : f 2 = -2 := 
by 
  sorry

end value_of_f_at_2_l218_218840


namespace Ramya_reads_total_124_pages_l218_218121

theorem Ramya_reads_total_124_pages :
  let total_pages : ℕ := 300
  let pages_read_monday := (1/5 : ℚ) * total_pages
  let pages_remaining := total_pages - pages_read_monday
  let pages_read_tuesday := (4/15 : ℚ) * pages_remaining
  pages_read_monday + pages_read_tuesday = 124 := 
by
  sorry

end Ramya_reads_total_124_pages_l218_218121


namespace pear_sales_l218_218920

theorem pear_sales (sale_afternoon : ℕ) (h1 : sale_afternoon = 260)
  (h2 : ∃ sale_morning : ℕ, sale_afternoon = 2 * sale_morning) :
  sale_afternoon / 2 + sale_afternoon = 390 :=
by
  sorry

end pear_sales_l218_218920


namespace prime_cubed_plus_prime_plus_one_not_square_l218_218689

theorem prime_cubed_plus_prime_plus_one_not_square (p : ℕ) (hp : Nat.Prime p) :
  ¬ ∃ k : ℕ, k * k = p^3 + p + 1 :=
by
  sorry

end prime_cubed_plus_prime_plus_one_not_square_l218_218689


namespace complement_A_in_B_l218_218623

-- Define the sets A and B
def A : Set ℕ := {2, 3}
def B : Set ℕ := {0, 1, 2, 3, 4}

-- Define the complement of A in B
def complement (U : Set ℕ) (A : Set ℕ) : Set ℕ := {x ∈ U | x ∉ A}

-- Statement to prove
theorem complement_A_in_B :
  complement B A = {0, 1, 4} := by
  sorry

end complement_A_in_B_l218_218623


namespace math_problem_l218_218448

theorem math_problem (a b : ℝ) (h : a / (1 + a) + b / (1 + b) = 1) : 
  a / (1 + b^2) - b / (1 + a^2) = a - b := 
sorry

end math_problem_l218_218448


namespace find_smaller_number_l218_218154

theorem find_smaller_number
  (x y : ℝ) (m : ℝ)
  (h1 : x - y = 9) 
  (h2 : x + y = 46)
  (h3 : x = m * y) : 
  min x y = 18.5 :=
by 
  sorry

end find_smaller_number_l218_218154


namespace problem_l218_218541

theorem problem 
  (x : ℝ) 
  (h1 : x ∈ Set.Icc (-3 : ℝ) 3) 
  (h2 : x ≠ -5/3) : 
  (4 * x ^ 2 + 2) / (5 + 3 * x) ≥ 1 ↔ x ∈ (Set.Icc (-3) (-3/4) ∪ Set.Icc 1 3) :=
sorry

end problem_l218_218541


namespace find_judes_age_l218_218380

def jude_age (H : ℕ) (J : ℕ) : Prop :=
  H + 5 = 3 * (J + 5)

theorem find_judes_age : ∃ J : ℕ, jude_age 16 J ∧ J = 2 :=
by
  sorry

end find_judes_age_l218_218380


namespace combined_yells_l218_218680

def yells_at_obedient : ℕ := 12
def yells_at_stubborn (y_obedient : ℕ) : ℕ := 4 * y_obedient
def total_yells (y_obedient : ℕ) (y_stubborn : ℕ) : ℕ := y_obedient + y_stubborn

theorem combined_yells : total_yells yells_at_obedient (yells_at_stubborn yells_at_obedient) = 60 := 
by
  sorry

end combined_yells_l218_218680


namespace quadratic_complete_square_l218_218627

/-- Given quadratic expression, complete the square to find the equivalent form
    and calculate the sum of the coefficients a, h, k. -/
theorem quadratic_complete_square (a h k : ℝ) :
  (∀ x : ℝ, 2 * x^2 - 8 * x + 2 = a * (x - h)^2 + k) → a + h + k = -2 :=
by
  intro h₁
  sorry

end quadratic_complete_square_l218_218627


namespace fraction_value_l218_218620

theorem fraction_value (a b : ℚ) (h₁ : b / (a - 2) = 3 / 4) (h₂ : b / (a + 9) = 5 / 7) : b / a = 165 / 222 := 
by sorry

end fraction_value_l218_218620


namespace minimum_average_cost_l218_218225

noncomputable def average_cost (x : ℝ) : ℝ :=
  let y := (x^2) / 10 - 30 * x + 4000
  y / x

theorem minimum_average_cost : 
  ∃ (x : ℝ), 150 ≤ x ∧ x ≤ 250 ∧ (∀ (x' : ℝ), 150 ≤ x' ∧ x' ≤ 250 → average_cost x ≤ average_cost x') ∧ average_cost x = 10 := 
by
  sorry

end minimum_average_cost_l218_218225


namespace remainder_of_sum_of_integers_mod_15_l218_218718

theorem remainder_of_sum_of_integers_mod_15 (a b c : ℕ) (h1 : a % 15 = 11) (h2 : b % 15 = 13) (h3 : c % 15 = 14) : (a + b + c) % 15 = 8 :=
by
  sorry

end remainder_of_sum_of_integers_mod_15_l218_218718


namespace only_natural_number_dividing_power_diff_l218_218422

theorem only_natural_number_dividing_power_diff (n : ℕ) (h : n ∣ (2^n - 1)) : n = 1 :=
by
  sorry

end only_natural_number_dividing_power_diff_l218_218422


namespace heartsuit_ratio_l218_218922

def heartsuit (n m : ℕ) : ℕ := n^4 * m^3

theorem heartsuit_ratio :
  (heartsuit 2 4) / (heartsuit 4 2) = 1 / 2 := by
  sorry

end heartsuit_ratio_l218_218922


namespace combined_weight_l218_218086

theorem combined_weight (x y z : ℕ) (h1 : x + y = 110) (h2 : y + z = 130) (h3 : z + x = 150) : x + y + z = 195 :=
by
  sorry

end combined_weight_l218_218086


namespace distance_from_town_l218_218424

theorem distance_from_town (d : ℝ) :
  (7 < d ∧ d < 8) ↔ (d < 8 ∧ d > 7 ∧ d > 6 ∧ d ≠ 9) :=
by sorry

end distance_from_town_l218_218424


namespace new_number_of_groups_l218_218975

-- Define the number of students
def total_students : ℕ := 2808

-- Define the initial and new number of groups
def initial_groups (n : ℕ) : ℕ := n + 4
def new_groups (n : ℕ) : ℕ := n

-- Condition: Fewer than 30 students per new group
def fewer_than_30_students_per_group (n : ℕ) : Prop :=
  total_students / n < 30

-- Condition: n and n + 4 must be divisors of total_students
def is_divisor (d : ℕ) (a : ℕ) : Prop :=
  a % d = 0

def valid_group_numbers (n : ℕ) : Prop :=
  is_divisor n total_students ∧ is_divisor (n + 4) total_students ∧ n > 93

-- The main theorem
theorem new_number_of_groups : ∃ n : ℕ, valid_group_numbers n ∧ fewer_than_30_students_per_group n ∧ n = 104 :=
by
  sorry

end new_number_of_groups_l218_218975


namespace find_x_l218_218721

theorem find_x (a b x : ℝ) (h1 : ∀ a b, a * b = 2 * a - b) (h2 : 2 * (6 * x) = 2) : x = 10 := 
sorry

end find_x_l218_218721


namespace simplify_polynomial_l218_218909

theorem simplify_polynomial : 
  ∀ (x : ℝ), 
    (2 * x + 1) ^ 5 - 5 * (2 * x + 1) ^ 4 + 10 * (2 * x + 1) ^ 3 - 10 * (2 * x + 1) ^ 2 + 5 * (2 * x + 1) - 1 
    = 32 * x ^ 5 := 
by sorry

end simplify_polynomial_l218_218909


namespace pentagon_area_l218_218228

theorem pentagon_area 
  (PQ QR RS ST TP : ℝ) 
  (angle_TPQ angle_PQR : ℝ) 
  (hPQ : PQ = 8) 
  (hQR : QR = 2) 
  (hRS : RS = 13) 
  (hST : ST = 13) 
  (hTP : TP = 8) 
  (hangle_TPQ : angle_TPQ = 90) 
  (hangle_PQR : angle_PQR = 90) : 
  PQ * QR + (1 / 2) * (TP - QR) * PQ + (1 / 2) * 10 * 12 = 100 := 
by
  sorry

end pentagon_area_l218_218228


namespace polygon_not_hexagon_if_quadrilateral_after_cut_off_l218_218986

-- Definition of polygonal shape and quadrilateral condition
def is_quadrilateral (sides : Nat) : Prop := sides = 4

-- Definition of polygonal shape with general condition of cutting off one angle
def after_cut_off (original_sides : Nat) (remaining_sides : Nat) : Prop :=
  original_sides > remaining_sides ∧ remaining_sides + 1 = original_sides

-- Problem statement: If a polygon's one angle cut-off results in a quadrilateral, then it is not a hexagon
theorem polygon_not_hexagon_if_quadrilateral_after_cut_off
  (original_sides : Nat) (remaining_sides : Nat) :
  after_cut_off original_sides remaining_sides → is_quadrilateral remaining_sides → original_sides ≠ 6 :=
by
  sorry

end polygon_not_hexagon_if_quadrilateral_after_cut_off_l218_218986


namespace solve_for_x_l218_218872

theorem solve_for_x (x y : ℤ) (h1 : x + 2 * y = 10) (h2 : y = 3) : x = 4 := by
  sorry

end solve_for_x_l218_218872


namespace negation_proof_l218_218390

theorem negation_proof :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) :=
by
  sorry

end negation_proof_l218_218390


namespace identity_of_brothers_l218_218379

theorem identity_of_brothers
  (first_brother_speaks : Prop)
  (second_brother_speaks : Prop)
  (one_tells_truth : first_brother_speaks → ¬ second_brother_speaks)
  (other_tells_truth : ¬first_brother_speaks → second_brother_speaks) :
  first_brother_speaks = false ∧ second_brother_speaks = true :=
by
  sorry

end identity_of_brothers_l218_218379


namespace message_hours_needed_l218_218856

-- Define the sequence and the condition
def S (n : ℕ) : ℕ := 2^(n + 1) - 2

theorem message_hours_needed : ∃ n : ℕ, S n > 55 ∧ n = 5 := by
  sorry

end message_hours_needed_l218_218856


namespace max_sum_value_l218_218417

noncomputable def max_sum (x y : ℝ) (h : 3 * (x^2 + y^2) = x - y) : ℝ :=
  x + y

theorem max_sum_value :
  ∃ x y : ℝ, ∃ h : 3 * (x^2 + y^2) = x - y, max_sum x y h = 1/3 :=
sorry

end max_sum_value_l218_218417


namespace distance_from_axis_gt_l218_218891

theorem distance_from_axis_gt 
  (a b x1 x2 y1 y2 : ℝ) (h₁ : a > 0) 
  (h₂ : y1 = a * x1^2 - 2 * a * x1 + b) 
  (h₃ : y2 = a * x2^2 - 2 * a * x2 + b) 
  (h₄ : y1 > y2) : 
  |x1 - 1| > |x2 - 1| := 
sorry

end distance_from_axis_gt_l218_218891


namespace computation_l218_218599

theorem computation :
  (13 + 12)^2 - (13 - 12)^2 = 624 :=
by
  sorry

end computation_l218_218599


namespace rearrangements_of_abcde_l218_218850

def is_adjacent (c1 c2 : Char) : Bool :=
  (c1 == 'a' ∧ c2 == 'b') ∨ 
  (c1 == 'b' ∧ c1 == 'a') ∨ 
  (c1 == 'b' ∧ c2 == 'c') ∨ 
  (c1 == 'c' ∧ c2 == 'b') ∨ 
  (c1 == 'c' ∧ c2 == 'd') ∨ 
  (c1 == 'd' ∧ c2 == 'c') ∨ 
  (c1 == 'd' ∧ c2 == 'e') ∨ 
  (c1 == 'e' ∧ c2 == 'd')

def is_valid_rearrangement (lst : List Char) : Bool :=
  match lst with
  | [] => true
  | [_] => true
  | c1 :: c2 :: rest => 
    ¬is_adjacent c1 c2 ∧ is_valid_rearrangement (c2 :: rest)

def count_valid_rearrangements (chars : List Char) : Nat :=
  chars.permutations.filter is_valid_rearrangement |>.length

theorem rearrangements_of_abcde : count_valid_rearrangements ['a', 'b', 'c', 'd', 'e'] = 8 := 
by
  sorry

end rearrangements_of_abcde_l218_218850


namespace sum_product_of_pairs_l218_218808

theorem sum_product_of_pairs (x y z : ℝ) 
  (h1 : x + y + z = 20) 
  (h2 : x^2 + y^2 + z^2 = 200) :
  x * y + x * z + y * z = 100 := 
by
  sorry

end sum_product_of_pairs_l218_218808


namespace geometric_sequence_ratio_l218_218818

variable {α : Type*} [Field α]

def geometric_sequence (a_1 q : α) (n : ℕ) : α :=
  a_1 * q ^ (n - 1)

theorem geometric_sequence_ratio (a1 q a4 a14 a5 a13 : α)
  (h_seq : ∀ n, geometric_sequence a1 q (n + 1) = a_5) 
  (h0 : geometric_sequence a1 q 5 * geometric_sequence a1 q 13 = 6) 
  (h1 : geometric_sequence a1 q 4 + geometric_sequence a1 q 14 = 5) :
  (∃ (k : α), k = 2 / 3 ∨ k = 3 / 2) → 
  geometric_sequence a1 q 80 / geometric_sequence a1 q 90 = k :=
by
  sorry

end geometric_sequence_ratio_l218_218818


namespace black_marbles_count_l218_218883

theorem black_marbles_count :
  ∀ (white_marbles total_marbles : ℕ), 
  white_marbles = 19 → total_marbles = 37 → total_marbles - white_marbles = 18 :=
by
  intros white_marbles total_marbles h_white h_total
  sorry

end black_marbles_count_l218_218883


namespace average_next_3_numbers_l218_218364

theorem average_next_3_numbers 
  (a1 a2 b1 b2 b3 c1 c2 c3 : ℝ)
  (h_avg_total : (a1 + a2 + b1 + b2 + b3 + c1 + c2 + c3) / 8 = 25)
  (h_avg_first2: (a1 + a2) / 2 = 20)
  (h_c1_c2 : c1 + 4 = c2)
  (h_c1_c3 : c1 + 6 = c3)
  (h_c3_value : c3 = 30) :
  (b1 + b2 + b3) / 3 = 26 := 
sorry

end average_next_3_numbers_l218_218364


namespace even_perfect_squares_between_50_and_200_l218_218616

theorem even_perfect_squares_between_50_and_200 : ∃ s : Finset ℕ, 
  (∀ n ∈ s, (n^2 ≥ 50) ∧ (n^2 ≤ 200) ∧ n^2 % 2 = 0) ∧ s.card = 4 := by
  sorry

end even_perfect_squares_between_50_and_200_l218_218616


namespace positive_integers_are_N_star_l218_218205

def Q := { x : ℚ | true } -- The set of rational numbers
def N := { x : ℕ | true } -- The set of natural numbers
def N_star := { x : ℕ | x > 0 } -- The set of positive integers
def Z := { x : ℤ | true } -- The set of integers

theorem positive_integers_are_N_star : 
  ∀ x : ℕ, (x ∈ N_star) ↔ (x > 0) := 
sorry

end positive_integers_are_N_star_l218_218205


namespace number_of_valid_permutations_l218_218630

noncomputable def count_valid_permutations : Nat :=
  let multiples_of_77 := [154, 231, 308, 385, 462, 539, 616, 693, 770, 847, 924]
  let total_count := multiples_of_77.foldl (fun acc x =>
    if x == 770 then
      acc + 3
    else if x == 308 then
      acc + 6 - 2
    else
      acc + 6) 0
  total_count

theorem number_of_valid_permutations : count_valid_permutations = 61 :=
  sorry

end number_of_valid_permutations_l218_218630


namespace find_m_value_l218_218755

theorem find_m_value :
  62519 * 9999 = 625127481 :=
  by sorry

end find_m_value_l218_218755


namespace lucky_sum_mod_1000_l218_218385

def is_lucky (n : ℕ) : Prop := ∀ d ∈ n.digits 10, d = 7

def first_twenty_lucky_numbers : List ℕ :=
  [7, 77] ++ List.replicate 18 777

theorem lucky_sum_mod_1000 :
  (first_twenty_lucky_numbers.sum % 1000) = 70 := 
sorry

end lucky_sum_mod_1000_l218_218385


namespace volume_at_20_deg_l218_218040

theorem volume_at_20_deg
  (ΔV_per_ΔT : ∀ ΔT : ℕ, ΔT = 5 → ∀ V : ℕ, V = 5)
  (initial_condition : ∀ V : ℕ, V = 40 ∧ ∀ T : ℕ, T = 40) :
  ∃ V : ℕ, V = 20 :=
by
  sorry

end volume_at_20_deg_l218_218040


namespace miyoung_largest_square_side_l218_218610

theorem miyoung_largest_square_side :
  ∃ (G : ℕ), G > 0 ∧ ∀ (a b : ℕ), (a = 32) → (b = 74) → (gcd a b = G) → (G = 2) :=
by {
  sorry
}

end miyoung_largest_square_side_l218_218610


namespace melissa_points_per_game_l218_218522

theorem melissa_points_per_game (total_points : ℕ) (games_played : ℕ) (h1 : total_points = 1200) (h2 : games_played = 10) : (total_points / games_played) = 120 := 
by
  -- Here we would insert the proof steps, but we use sorry to represent the omission
  sorry

end melissa_points_per_game_l218_218522


namespace six_digit_numbers_with_zero_l218_218959

theorem six_digit_numbers_with_zero :
  let total_digits := (9 * 10 * 10 * 10 * 10 * 10 : ℕ)
  let non_zero_digits := (9 * 9 * 9 * 9 * 9 * 9 : ℕ)
  total_digits - non_zero_digits = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l218_218959


namespace find_a10_l218_218081

noncomputable def geometric_sequence (a : ℕ → ℝ) := 
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

def a2_eq_4 (a : ℕ → ℝ) := a 2 = 4

def a6_eq_6 (a : ℕ → ℝ) := a 6 = 6

theorem find_a10 (a : ℕ → ℝ) (h_geom : geometric_sequence a) (h2 : a2_eq_4 a) (h6 : a6_eq_6 a) : 
  a 10 = 9 :=
sorry

end find_a10_l218_218081


namespace g_min_value_l218_218629

noncomputable def g (x : ℝ) : ℝ :=
  x + x / (x^2 + 2) + (x * (x + 5)) / (x^2 + 3) + (3 * (x + 3)) / (x * (x^2 + 3))

theorem g_min_value (x : ℝ) (h : x > 0) : g x >= 6 :=
sorry

end g_min_value_l218_218629


namespace green_chips_correct_l218_218472

-- Definitions
def total_chips : ℕ := 120
def blue_chips : ℕ := total_chips / 4
def red_chips : ℕ := total_chips * 20 / 100
def yellow_chips : ℕ := total_chips / 10
def non_green_chips : ℕ := blue_chips + red_chips + yellow_chips
def green_chips : ℕ := total_chips - non_green_chips

-- Statement to prove
theorem green_chips_correct : green_chips = 54 := by
  -- Proof would go here
  sorry

end green_chips_correct_l218_218472


namespace inner_circle_radius_is_sqrt_2_l218_218165

noncomputable def radius_of_inner_circle (side_length : ℝ) : ℝ :=
  let semicircle_radius := side_length / 4
  let distance_from_center_to_semicircle_center :=
    Real.sqrt ((side_length / 2) ^ 2 + (side_length / 2) ^ 2)
  let inner_circle_radius := (distance_from_center_to_semicircle_center - semicircle_radius)
  inner_circle_radius

theorem inner_circle_radius_is_sqrt_2 (side_length : ℝ) (h: side_length = 4) : 
  radius_of_inner_circle side_length = Real.sqrt 2 :=
by
  sorry

end inner_circle_radius_is_sqrt_2_l218_218165


namespace xiaohong_money_l218_218440

def cost_kg_pears (x : ℝ) := x

def cost_kg_apples (x : ℝ) := x + 1.1

theorem xiaohong_money (x : ℝ) (hx : 6 * x - 3 = 5 * (x + 1.1) - 4) : 6 * x - 3 = 24 :=
by sorry

end xiaohong_money_l218_218440


namespace find_h_l218_218536

-- Define the polynomial f(x)
def f (x : ℤ) := x^4 - 2 * x^3 + x - 1

-- Define the condition that f(x) + h(x) = 3x^2 + 5x - 4
def condition (f h : ℤ → ℤ) := ∀ x, f x + h x = 3 * x^2 + 5 * x - 4

-- Define the solution for h(x) to be proved
def h_solution (x : ℤ) := -x^4 + 2 * x^3 + 3 * x^2 + 4 * x - 3

-- State the theorem to be proved
theorem find_h (h : ℤ → ℤ) (H : condition f h) : h = h_solution :=
by
  sorry

end find_h_l218_218536


namespace range_of_y0_l218_218136

theorem range_of_y0
  (y0 : ℝ)
  (h_tangent : ∃ N : ℝ × ℝ, (N.1^2 + N.2^2 = 1) ∧ ((↑(Real.sqrt 3 - N.1)^2 + (y0 - N.2)^2) = 1))
  (h_angle : ∀ N : ℝ × ℝ, (N.1^2 + N.2^2 = 1) ∧ ((↑(Real.sqrt 3 - N.1)^2 + (y0 - N.2)^2 = 1)) → (Real.arccos ((Real.sqrt 3 - N.1)/Real.sqrt ((3 - 2 * N.1 * Real.sqrt 3 + N.1^2) + (y0 - N.2)^2)) ≥ π / 6)) :
  -1 ≤ y0 ∧ y0 ≤ 1 :=
by
  sorry

end range_of_y0_l218_218136


namespace inequality_solution_correct_l218_218655

variable (f : ℝ → ℝ)

def f_one : Prop := f 1 = 1

def f_prime_half : Prop := ∀ x : ℝ, (deriv f x) > (1 / 2)

def inequality_solution_set : Prop := ∀ x : ℝ, f (x^2) < (x^2 / 2 + 1 / 2) ↔ -1 < x ∧ x < 1

theorem inequality_solution_correct (h1 : f_one f) (h2 : f_prime_half f) : inequality_solution_set f := sorry

end inequality_solution_correct_l218_218655


namespace classify_discuss_l218_218259

theorem classify_discuss (a b c : ℚ) (h : a * b * c > 0) : 
  (|a| / a + |b| / b + |c| / c = 3) ∨ (|a| / a + |b| / b + |c| / c = -1) :=
sorry

end classify_discuss_l218_218259


namespace ben_less_than_jack_l218_218486

def jack_amount := 26
def total_amount := 50
def eric_ben_difference := 10

theorem ben_less_than_jack (E B J : ℕ) (h1 : E = B - eric_ben_difference) (h2 : J = jack_amount) (h3 : E + B + J = total_amount) :
  J - B = 9 :=
by sorry

end ben_less_than_jack_l218_218486


namespace find_other_divisor_l218_218370

theorem find_other_divisor (x : ℕ) (h : x ≠ 35) (h1 : 386 % 35 = 1) (h2 : 386 % x = 1) : x = 11 :=
sorry

end find_other_divisor_l218_218370


namespace running_hours_per_week_l218_218283

theorem running_hours_per_week 
  (initial_days : ℕ) (additional_days : ℕ) (morning_run_time : ℕ) (evening_run_time : ℕ)
  (total_days : ℕ) (total_run_time_per_day : ℕ) (total_run_time_per_week : ℕ)
  (H1 : initial_days = 3)
  (H2 : additional_days = 2)
  (H3 : morning_run_time = 1)
  (H4 : evening_run_time = 1)
  (H5 : total_days = initial_days + additional_days)
  (H6 : total_run_time_per_day = morning_run_time + evening_run_time)
  (H7 : total_run_time_per_week = total_days * total_run_time_per_day) :
  total_run_time_per_week = 10 := 
sorry

end running_hours_per_week_l218_218283


namespace fraction_of_3_4_is_4_27_l218_218242

theorem fraction_of_3_4_is_4_27 (a b : ℚ) (h1 : a = 3/4) (h2 : b = 1/9) :
  b / a = 4 / 27 :=
by
  sorry

end fraction_of_3_4_is_4_27_l218_218242


namespace mixed_nuts_price_l218_218312

theorem mixed_nuts_price (total_weight : ℝ) (peanut_price : ℝ) (cashew_price : ℝ) (cashew_weight : ℝ) 
  (H1 : total_weight = 100) 
  (H2 : peanut_price = 3.50) 
  (H3 : cashew_price = 4.00) 
  (H4 : cashew_weight = 60) : 
  (cashew_weight * cashew_price + (total_weight - cashew_weight) * peanut_price) / total_weight = 3.80 :=
by 
  sorry

end mixed_nuts_price_l218_218312


namespace beth_lost_red_marbles_l218_218118

-- Definitions from conditions
def total_marbles : ℕ := 72
def marbles_per_color : ℕ := total_marbles / 3
variable (R : ℕ)  -- Number of red marbles Beth lost
def blue_marbles_lost : ℕ := 2 * R
def yellow_marbles_lost : ℕ := 3 * R
def marbles_left : ℕ := 42

-- Theorem we want to prove
theorem beth_lost_red_marbles (h : total_marbles - (R + blue_marbles_lost R + yellow_marbles_lost R) = marbles_left) :
  R = 5 :=
by
  sorry

end beth_lost_red_marbles_l218_218118


namespace baker_extra_cakes_l218_218712

-- Defining the conditions
def original_cakes : ℕ := 78
def total_cakes : ℕ := 87
def extra_cakes := total_cakes - original_cakes

-- The statement to prove
theorem baker_extra_cakes : extra_cakes = 9 := by
  sorry

end baker_extra_cakes_l218_218712


namespace carol_meets_alice_in_30_minutes_l218_218643

def time_to_meet (alice_speed carol_speed initial_distance : ℕ) : ℕ :=
((initial_distance * 60) / (alice_speed + carol_speed))

theorem carol_meets_alice_in_30_minutes :
  time_to_meet 4 6 5 = 30 := 
by 
  sorry

end carol_meets_alice_in_30_minutes_l218_218643


namespace min_cost_example_l218_218043

-- Define the numbers given in the problem
def num_students : Nat := 25
def num_vampire : Nat := 11
def num_pumpkin : Nat := 14
def pack_cost : Nat := 3
def individual_cost : Nat := 1
def pack_size : Nat := 5

-- Define the cost calculation function
def min_cost (num_v: Nat) (num_p: Nat) : Nat :=
  let num_v_packs := num_v / pack_size  -- number of packs needed for vampire bags
  let num_v_individual := num_v % pack_size  -- remaining vampire bags needed
  let num_v_cost := (num_v_packs * pack_cost) + (num_v_individual * individual_cost)
  let num_p_packs := num_p / pack_size  -- number of packs needed for pumpkin bags
  let num_p_individual := num_p % pack_size  -- remaining pumpkin bags needed
  let num_p_cost := (num_p_packs * pack_cost) + (num_p_individual * individual_cost)
  num_v_cost + num_p_cost

-- The statement to prove
theorem min_cost_example : min_cost num_vampire num_pumpkin = 17 :=
  by
  sorry

end min_cost_example_l218_218043


namespace exponential_decreasing_iff_frac_inequality_l218_218274

theorem exponential_decreasing_iff_frac_inequality (a : ℝ) :
  (0 < a ∧ a < 1) ↔ (a ≠ 1 ∧ a * (a - 1) ≤ 0) :=
by
  sorry

end exponential_decreasing_iff_frac_inequality_l218_218274


namespace line_through_circles_l218_218291

theorem line_through_circles (D1 E1 D2 E2 : ℝ)
  (h1 : 2 * D1 - E1 + 2 = 0)
  (h2 : 2 * D2 - E2 + 2 = 0) :
  (2 * D1 - E1 + 2 = 0) ∧ (2 * D2 - E2 + 2 = 0) :=
by
  exact ⟨h1, h2⟩

end line_through_circles_l218_218291


namespace each_album_contains_correct_pictures_l218_218143

def pictures_in_each_album (pictures_phone pictures_camera albums pictures_per_album_phone pictures_per_album_camera : Nat) :=
  (pictures_per_album_phone + pictures_per_album_camera)

theorem each_album_contains_correct_pictures (pictures_phone pictures_camera albums pictures_per_album_phone pictures_per_album_camera : Nat)
  (h1 : pictures_phone = 80)
  (h2 : pictures_camera = 40)
  (h3 : albums = 10)
  (h4 : pictures_per_album_phone = 8)
  (h5 : pictures_per_album_camera = 4)
  : pictures_in_each_album pictures_phone pictures_camera albums pictures_per_album_phone pictures_per_album_camera = 12 := by
  sorry

end each_album_contains_correct_pictures_l218_218143


namespace fraction_budget_paid_l218_218077

variable (B : ℝ) (b k : ℝ)

-- Conditions
def condition1 : b = 0.30 * (B - k) := by sorry
def condition2 : k = 0.10 * (B - b) := by sorry

-- Proof that Jenny paid 35% of her budget for her book and snack
theorem fraction_budget_paid :
  b + k = 0.35 * B :=
by
  -- use condition1 and condition2 to prove the theorem
  sorry

end fraction_budget_paid_l218_218077


namespace point_on_parabola_touching_x_axis_l218_218469

theorem point_on_parabola_touching_x_axis (a b c : ℤ) (h : ∃ r : ℤ, a * (r * r) + b * r + c = 0 ∧ (r * r) = 0) :
  ∃ (a' b' : ℤ), ∃ k : ℤ, (k * k) + a' * k + b' = 0 ∧ (k * k) = 0 :=
sorry

end point_on_parabola_touching_x_axis_l218_218469


namespace determine_g_l218_218446

noncomputable def g : ℝ → ℝ := sorry 

lemma g_functional_equation (x y : ℝ) : g (x * y) = g ((x^2 + y^2 + 1) / 3) + (x - y)^2 :=
sorry

lemma g_at_zero : g 0 = 1 :=
sorry

theorem determine_g (x : ℝ) : g x = 2 - 2 * x :=
sorry

end determine_g_l218_218446


namespace complex_purely_imaginary_l218_218099

theorem complex_purely_imaginary (m : ℝ) :
  (m^2 - 3*m + 2 = 0) ∧ (m^2 - 2*m ≠ 0) → m = 1 :=
by {
  sorry
}

end complex_purely_imaginary_l218_218099


namespace olivia_total_pieces_l218_218209

def initial_pieces_folder1 : ℕ := 152
def initial_pieces_folder2 : ℕ := 98
def used_pieces_folder1 : ℕ := 78
def used_pieces_folder2 : ℕ := 42

def remaining_pieces_folder1 : ℕ :=
  initial_pieces_folder1 - used_pieces_folder1

def remaining_pieces_folder2 : ℕ :=
  initial_pieces_folder2 - used_pieces_folder2

def total_remaining_pieces : ℕ :=
  remaining_pieces_folder1 + remaining_pieces_folder2

theorem olivia_total_pieces : total_remaining_pieces = 130 :=
  by sorry

end olivia_total_pieces_l218_218209


namespace remainder_b22_div_35_l218_218235

def b_n (n : ℕ) : Nat :=
  ((List.range (n + 1)).drop 1).foldl (λ acc k => acc * 10^(Nat.digits 10 k).length + k) 0

theorem remainder_b22_div_35 : (b_n 22) % 35 = 17 :=
  sorry

end remainder_b22_div_35_l218_218235


namespace maximize_area_playground_l218_218182

noncomputable def maxAreaPlayground : ℝ :=
  let l := 100
  let w := 100
  l * w

theorem maximize_area_playground : ∀ (l w : ℝ),
  (2 * l + 2 * w = 400) ∧ (l ≥ 100) ∧ (w ≥ 60) → l * w ≤ maxAreaPlayground :=
by
  intros l w h
  sorry

end maximize_area_playground_l218_218182


namespace problem1_problem2_l218_218507

-- Problem 1: Proving the range of m values for the given inequality
theorem problem1 (m : ℝ) : (∀ x : ℝ, |x + 1| + |x - m| ≥ 3) ↔ (m ≤ -4 ∨ m ≥ 2) :=
sorry

-- Problem 2: Proving the range of m values given a non-empty solution set for the inequality
theorem problem2 (m : ℝ) : (∃ x : ℝ, |m + 1| - 2 * m ≥ x^2 - x) ↔ (m ≤ 5/4) :=
sorry

end problem1_problem2_l218_218507


namespace toy_store_restock_l218_218144

theorem toy_store_restock 
  (initial_games : ℕ) (games_sold : ℕ) (after_restock_games : ℕ) 
  (initial_games_condition : initial_games = 95)
  (games_sold_condition : games_sold = 68)
  (after_restock_games_condition : after_restock_games = 74) :
  after_restock_games - (initial_games - games_sold) = 47 :=
by {
  sorry
}

end toy_store_restock_l218_218144


namespace cats_weight_difference_l218_218728

-- Define the weights of Anne's and Meg's cats
variables (A M : ℕ)

-- Given conditions:
-- 1. Ratio of weights Meg's cat to Anne's cat is 13:21
-- 2. Meg's cat's weight is 20 kg plus half the weight of Anne's cat

theorem cats_weight_difference (h1 : M = 20 + (A / 2)) (h2 : 13 * A = 21 * M) : A - M = 64 := 
by {
    sorry
}

end cats_weight_difference_l218_218728


namespace vertical_increase_is_100m_l218_218044

theorem vertical_increase_is_100m 
  (a b x : ℝ)
  (hypotenuse : a = 100 * Real.sqrt 5)
  (slope_ratio : b = 2 * x)
  (pythagorean_thm : x^2 + b^2 = a^2) : 
  x = 100 :=
by
  sorry

end vertical_increase_is_100m_l218_218044


namespace total_value_of_coins_l218_218496

theorem total_value_of_coins (num_quarters num_nickels : ℕ) (val_quarter val_nickel : ℝ)
  (h_quarters : num_quarters = 8) (h_nickels : num_nickels = 13)
  (h_total_coins : num_quarters + num_nickels = 21) (h_val_quarter : val_quarter = 0.25)
  (h_val_nickel : val_nickel = 0.05) :
  num_quarters * val_quarter + num_nickels * val_nickel = 2.65 := 
sorry

end total_value_of_coins_l218_218496


namespace cameron_list_count_l218_218208

-- Definitions
def is_multiple_of (a b : ℕ) : Prop := ∃ k, a = k * b
def is_perfect_square (n : ℕ) : Prop := ∃ m, n = m * m
def is_perfect_cube (n : ℕ) : Prop := ∃ m, n = m * m * m

-- The main statement
theorem cameron_list_count :
  let smallest_square := 25
  let smallest_cube := 125
  (∀ n : ℕ, is_multiple_of n 25 → smallest_square ≤ n → n ≤ smallest_cube) →
  ∃ count : ℕ, count = 5 :=
by 
  sorry

end cameron_list_count_l218_218208


namespace possible_values_of_K_l218_218593

theorem possible_values_of_K (K M : ℕ) (h : K * (K + 1) = M^2) (hM : M < 100) : K = 8 ∨ K = 35 :=
by sorry

end possible_values_of_K_l218_218593


namespace tulips_for_each_eye_l218_218845

theorem tulips_for_each_eye (R : ℕ) : 2 * R + 18 + 9 * 18 = 196 → R = 8 :=
by
  intro h
  sorry

end tulips_for_each_eye_l218_218845


namespace scientific_notation_of_3300000000_l218_218751

theorem scientific_notation_of_3300000000 :
  3300000000 = 3.3 * 10^9 :=
sorry

end scientific_notation_of_3300000000_l218_218751


namespace smallest_positive_debt_resolvable_l218_218540

/-- The value of a sheep in dollars -/
def sheep_value : ℕ := 250

/-- The value of a lamb in dollars -/
def lamb_value : ℕ := 150

/-- Given a debt D that can be expressed in the form of 250s + 150l for integers s and l,
prove that the smallest positive amount of D is 50 dollars -/
theorem smallest_positive_debt_resolvable : 
  ∃ (s l : ℤ), sheep_value * s + lamb_value * l = 50 :=
sorry

end smallest_positive_debt_resolvable_l218_218540


namespace system1_solution_system2_solution_l218_218908

theorem system1_solution (x y : ℤ) 
  (h1 : x = 2 * y - 1) 
  (h2 : 3 * x + 4 * y = 17) : 
  x = 3 ∧ y = 2 :=
by 
  sorry

theorem system2_solution (x y : ℤ) 
  (h1 : 2 * x - y = 0) 
  (h2 : 3 * x - 2 * y = 5) : 
  x = -5 ∧ y = -10 := 
by 
  sorry

end system1_solution_system2_solution_l218_218908


namespace product_xyz_equals_one_l218_218750

theorem product_xyz_equals_one (x y z : ℝ) (h1 : x + (1/y) = 2) (h2 : y + (1/z) = 2) : x * y * z = 1 := 
by
  sorry

end product_xyz_equals_one_l218_218750


namespace solve_inequality_l218_218359

variables (a b c x α β : ℝ)

theorem solve_inequality 
  (h1 : ∀ x, a * x^2 + b * x + c > 0 ↔ α < x ∧ x < β)
  (h2 : β > α)
  (ha : a < 0)
  (h3 : α + β = -b / a)
  (h4 : α * β = c / a) :
  ∀ x, (c * x^2 + b * x + a < 0 ↔ x < 1 / β ∨ x > 1 / α) := 
  by
    -- A detailed proof would follow here.
    sorry

end solve_inequality_l218_218359


namespace power_decomposition_l218_218395

theorem power_decomposition (n m : ℕ) (h1 : n ≥ 2) 
  (h2 : n * n = 1 + 3 + 5 + 7 + 9 + 11 + 13 + 15 + 17 + 19) 
  (h3 : Nat.succ 19 = 21) 
  : m + n = 15 := sorry

end power_decomposition_l218_218395


namespace find_minimal_product_l218_218997

theorem find_minimal_product : ∃ x y : ℤ, (20 * x + 19 * y = 2019) ∧ (x * y = 2623) ∧ (∀ z w : ℤ, (20 * z + 19 * w = 2019) → |x - y| ≤ |z - w|) :=
by
  -- definitions and theorems to prove the problem would be placed here
  sorry

end find_minimal_product_l218_218997


namespace fraction_zero_l218_218241

theorem fraction_zero (x : ℝ) (h : (x - 1) * (x + 2) = 0) (hne : x^2 - 1 ≠ 0) : x = -2 :=
by
  sorry

end fraction_zero_l218_218241


namespace y_at_x_eq_120_l218_218789

@[simp] def custom_op (a b : ℕ) : ℕ := List.prod (List.map (λ i => a + i) (List.range b))

theorem y_at_x_eq_120 {x y : ℕ}
  (h1 : custom_op x (custom_op y 2) = 420)
  (h2 : x = 4)
  (h3 : y = 2) :
  custom_op y x = 120 := by
  sorry

end y_at_x_eq_120_l218_218789


namespace smallest_value_of_3a_plus_2_l218_218865

variable (a : ℝ)

theorem smallest_value_of_3a_plus_2 (h : 5 * a^2 + 7 * a + 2 = 1) : 3 * a + 2 = -1 :=
sorry

end smallest_value_of_3a_plus_2_l218_218865


namespace binom_30_3_is_4060_l218_218667

theorem binom_30_3_is_4060 : Nat.choose 30 3 = 4060 := by
  sorry

end binom_30_3_is_4060_l218_218667


namespace eighth_hexagonal_number_l218_218381

theorem eighth_hexagonal_number : (8 * (2 * 8 - 1)) = 120 :=
  by
  sorry

end eighth_hexagonal_number_l218_218381


namespace seth_oranges_l218_218484

def initial_boxes := 9
def boxes_given_to_mother := 1

def remaining_boxes_after_giving_to_mother := initial_boxes - boxes_given_to_mother
def boxes_given_away := remaining_boxes_after_giving_to_mother / 2
def boxes_left := remaining_boxes_after_giving_to_mother - boxes_given_away

theorem seth_oranges : boxes_left = 4 := by
  sorry

end seth_oranges_l218_218484


namespace total_apple_trees_is_800_l218_218003

variable (T P A : ℕ) -- Total number of trees, peach trees, and apple trees respectively
variable (samples_peach samples_apple : ℕ) -- Sampled peach trees and apple trees respectively
variable (sampled_percentage : ℕ) -- Percentage of total trees sampled

-- Given conditions
axiom H1 : sampled_percentage = 10
axiom H2 : samples_peach = 50
axiom H3 : samples_apple = 80

-- Theorem to prove the number of apple trees
theorem total_apple_trees_is_800 : A = 800 :=
by sorry

end total_apple_trees_is_800_l218_218003


namespace Mary_bought_stickers_initially_l218_218913

variable (S A M : ℕ) -- Define S, A, and M as natural numbers

-- Given conditions in the problem
def condition1 : Prop := S = A
def condition2 : Prop := M = 3 * A
def condition3 : Prop := A + (2 / 3) * M = 900

-- The theorem we need to prove
theorem Mary_bought_stickers_initially
  (h1 : condition1 S A)
  (h2 : condition2 A M)
  (h3 : condition3 A M)
  : S + A + M = 1500 :=
sorry -- Proof

end Mary_bought_stickers_initially_l218_218913


namespace repeating_sequence_length_1_over_221_l218_218842

theorem repeating_sequence_length_1_over_221 : ∃ n : ℕ, (10 ^ n ≡ 1 [MOD 221]) ∧ (∀ m : ℕ, (10 ^ m ≡ 1 [MOD 221]) → (n ≤ m)) ∧ n = 48 :=
by
  sorry

end repeating_sequence_length_1_over_221_l218_218842


namespace sequence_has_both_max_and_min_l218_218211

noncomputable def a_n (n : ℕ) : ℝ :=
  (n + 1) * ((-10 / 11) ^ n)

theorem sequence_has_both_max_and_min :
  ∃ (max min : ℝ) (N M : ℕ), 
    (∀ n : ℕ, a_n n ≤ max) ∧ (∀ n : ℕ, min ≤ a_n n) ∧ 
    (a_n N = max) ∧ (a_n M = min) := 
sorry

end sequence_has_both_max_and_min_l218_218211


namespace asphalt_road_proof_l218_218084

-- We define the initial conditions given in the problem
def man_hours (men days hours_per_day : Nat) : Nat :=
  men * days * hours_per_day

-- Given the conditions for asphalting 1 km road
def conditions_1 (men1 days1 hours_per_day1 : Nat) : Prop :=
  man_hours men1 days1 hours_per_day1 = 2880

-- Given that the second road is 2 km long
def conditions_2 (man_hours1 : Nat) : Prop :=
  2 * man_hours1 = 5760

-- Given the working conditions for the second road
def conditions_3 (men2 days2 hours_per_day2 : Nat) : Prop :=
  men2 * days2 * hours_per_day2 = 5760

-- The theorem to prove
theorem asphalt_road_proof 
  (men1 days1 hours_per_day1 days2 hours_per_day2 men2 : Nat)
  (H1 : conditions_1 men1 days1 hours_per_day1)
  (H2 : conditions_2 (man_hours men1 days1 hours_per_day1))
  (H3 : men2 * days2 * hours_per_day2 = 5760)
  : men2 = 20 :=
by
  sorry

end asphalt_road_proof_l218_218084


namespace range_of_m_l218_218132

def A := {x : ℝ | x^2 - 3*x + 2 = 0}
def B (m : ℝ) := {x : ℝ | x^2 - 2*x + m = 0}

theorem range_of_m (m : ℝ) : (A ∪ B m = A) ↔ m ∈ Set.Ici 1 :=
by
  sorry

end range_of_m_l218_218132


namespace solve_for_a_minus_b_l218_218853

theorem solve_for_a_minus_b (a b : ℝ) (h1 : |a| = 5) (h2 : |b| = 7) (h3 : |a + b| = a + b) : a - b = -2 := 
sorry

end solve_for_a_minus_b_l218_218853


namespace players_quit_l218_218020

theorem players_quit (initial_players remaining_lives lives_per_player : ℕ) 
  (h1 : initial_players = 8) (h2 : remaining_lives = 15) (h3 : lives_per_player = 5) :
  initial_players - (remaining_lives / lives_per_player) = 5 :=
by
  -- A proof is required here
  sorry

end players_quit_l218_218020


namespace n_value_l218_218662

theorem n_value (n : ℕ) (h1 : ∃ a b : ℕ, a = (5 + 10 + 15 + 20 + 25 + 30 + 35) / 7 ∧ b = 2 * n ∧ a ^ 2 - b ^ 2 = 0) : n = 10 := 
  by sorry

end n_value_l218_218662


namespace james_spent_6_dollars_l218_218320

-- Define the constants based on the conditions
def cost_milk : ℝ := 3
def cost_bananas : ℝ := 2
def tax_rate : ℝ := 0.20

-- Define the total cost before tax
def total_cost_before_tax : ℝ := cost_milk + cost_bananas

-- Define the sales tax
def sales_tax : ℝ := total_cost_before_tax * tax_rate

-- Define the total amount spent
def total_amount_spent : ℝ := total_cost_before_tax + sales_tax

-- The theorem to prove that James spent $6
theorem james_spent_6_dollars : total_amount_spent = 6 := by
  sorry

end james_spent_6_dollars_l218_218320


namespace pastries_solution_l218_218700

def pastries_problem : Prop :=
  ∃ (F Calvin Phoebe Grace : ℕ),
  (Calvin = F + 8) ∧
  (Phoebe = F + 8) ∧
  (Grace = 30) ∧
  (F + Calvin + Phoebe + Grace = 97) ∧
  (Grace - Calvin = 5) ∧
  (Grace - Phoebe = 5)

theorem pastries_solution : pastries_problem :=
by
  sorry

end pastries_solution_l218_218700


namespace robotics_club_neither_l218_218128

theorem robotics_club_neither (n c e b neither : ℕ) (h1 : n = 80) (h2 : c = 50) (h3 : e = 40) (h4 : b = 25) :
  neither = n - (c - b + e - b + b) :=
by 
  rw [h1, h2, h3, h4]
  sorry

end robotics_club_neither_l218_218128


namespace no_such_number_exists_l218_218899

theorem no_such_number_exists : ¬ ∃ n : ℕ, 10^(n+1) + 35 ≡ 0 [MOD 63] :=
by {
  sorry 
}

end no_such_number_exists_l218_218899


namespace sum_of_possible_two_digit_values_l218_218108

theorem sum_of_possible_two_digit_values (d : ℕ) (h1 : 0 < d) (h2 : d < 100) (h3 : 137 % d = 6) : d = 131 :=
by
  sorry

end sum_of_possible_two_digit_values_l218_218108


namespace solve_quadratic_completing_square_l218_218045

theorem solve_quadratic_completing_square (x : ℝ) :
  x^2 - 4 * x + 3 = 0 → (x - 2)^2 = 1 :=
by sorry

end solve_quadratic_completing_square_l218_218045


namespace additional_flowers_grew_l218_218941

-- Define the initial conditions
def initial_flowers : ℕ := 10  -- Dane’s two daughters planted 5 flowers each (5 + 5).
def flowers_died : ℕ := 10     -- 10 flowers died.
def baskets : ℕ := 5
def flowers_per_basket : ℕ := 4

-- Total flowers harvested (from the baskets)
def total_harvested : ℕ := baskets * flowers_per_basket  -- 5 * 4 = 20

-- The proof to show additional flowers grown
theorem additional_flowers_grew : (total_harvested - initial_flowers + flowers_died) = 10 :=
by
  -- The final number of flowers and the initial number of flowers are known
  have final_flowers : ℕ := total_harvested
  have initial_plus_grown : ℕ := initial_flowers + (total_harvested - initial_flowers)
  -- Show the equality that defines the additional flowers grown
  show (total_harvested - initial_flowers + flowers_died) = 10
  sorry

end additional_flowers_grew_l218_218941


namespace Traci_trip_fraction_l218_218051

theorem Traci_trip_fraction :
  let total_distance := 600
  let first_stop_distance := total_distance / 3
  let remaining_distance_after_first_stop := total_distance - first_stop_distance
  let final_leg_distance := 300
  let distance_between_stops := remaining_distance_after_first_stop - final_leg_distance
  (distance_between_stops / remaining_distance_after_first_stop) = 1 / 4 :=
by
  let total_distance := 600
  let first_stop_distance := 600 / 3
  let remaining_distance_after_first_stop := 600 - first_stop_distance
  let final_leg_distance := 300
  let distance_between_stops := remaining_distance_after_first_stop - final_leg_distance
  have h1 : total_distance = 600 := by exact rfl
  have h2 : first_stop_distance = 200 := by norm_num [first_stop_distance]
  have h3 : remaining_distance_after_first_stop = 400 := by norm_num [remaining_distance_after_first_stop]
  have h4 : distance_between_stops = 100 := by norm_num [distance_between_stops]
  show (distance_between_stops / remaining_distance_after_first_stop) = 1/4
  -- Proof omitted
  sorry

end Traci_trip_fraction_l218_218051


namespace expression_divisible_by_11_l218_218917

theorem expression_divisible_by_11 (n : ℕ) : (6^(2*n) + 3^(n+2) + 3^n) % 11 = 0 := 
sorry

end expression_divisible_by_11_l218_218917


namespace Evelyn_bottle_caps_problem_l218_218676

theorem Evelyn_bottle_caps_problem (E : ℝ) (H1 : E - 18.0 = 45) : E = 63.0 := 
by
  sorry


end Evelyn_bottle_caps_problem_l218_218676


namespace equation_of_perpendicular_line_l218_218284

theorem equation_of_perpendicular_line (a b c : ℝ) (p q : ℝ) (hx : a ≠ 0) (hy : b ≠ 0)
  (h_perpendicular : a * 2 + b * 1 = 0) (h_point : (-1) * a + 2 * b + c = 0)
  : a = 1 ∧ b = -2 ∧ c = -5 → (x:ℝ) * 1 + (y:ℝ) * (-2) + (-5) = 0 :=
by sorry

end equation_of_perpendicular_line_l218_218284


namespace find_constants_monotonicity_range_of_k_l218_218352

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (b - 2 ^ x) / (2 ^ (x + 1) + a)

theorem find_constants (h_odd : ∀ x : ℝ, f x a b = - f (-x) a b) :
  a = 2 ∧ b = 1 :=
sorry

theorem monotonicity (a : ℝ) (b : ℝ) (h_constants : a = 2 ∧ b = 1) :
  ∀ x y : ℝ, x < y → f y a b ≤ f x a b :=
sorry

theorem range_of_k (a : ℝ) (b : ℝ) (h_constants : a = 2 ∧ b = 1)
  (h_pos : ∀ x : ℝ, x ≥ 1 → f (k * 3^x) a b + f (3^x - 9^x + 2) a b > 0) :
  k < 4 / 3 :=
sorry

end find_constants_monotonicity_range_of_k_l218_218352


namespace T_5_value_l218_218293

noncomputable def T (y : ℝ) (m : ℕ) : ℝ := y^m + (1 / y)^m

theorem T_5_value (y : ℝ) (h : y + 1 / y = 5) : T y 5 = 2525 := 
by {
  sorry
}

end T_5_value_l218_218293


namespace selling_price_of_article_l218_218944

theorem selling_price_of_article (cost_price gain_percent : ℝ) (h1 : cost_price = 100) (h2 : gain_percent = 30) : 
  cost_price + (gain_percent / 100) * cost_price = 130 := 
by 
  sorry

end selling_price_of_article_l218_218944


namespace find_m_l218_218514

theorem find_m (m : ℝ) (h : 2 / m = (m + 1) / 3) : m = -3 := by
  sorry

end find_m_l218_218514


namespace N_is_even_l218_218262

def sum_of_digits : ℕ → ℕ := sorry

theorem N_is_even 
  (N : ℕ)
  (h1 : sum_of_digits N = 100)
  (h2 : sum_of_digits (5 * N) = 50) : 
  Even N :=
sorry

end N_is_even_l218_218262


namespace cooper_saved_days_l218_218621

variable (daily_saving : ℕ) (total_saving : ℕ) (n : ℕ)

-- Conditions
def cooper_saved (daily_saving total_saving n : ℕ) : Prop :=
  total_saving = daily_saving * n

-- Theorem stating the question equals the correct answer
theorem cooper_saved_days :
  cooper_saved 34 12410 365 :=
by
  sorry

end cooper_saved_days_l218_218621


namespace solve_for_m_l218_218066

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (2 / (2^x + 1)) + m

theorem solve_for_m (m : ℝ) : (∀ x : ℝ, f m (-x) = - (f m x)) ↔ m = -1 := by
sorry

end solve_for_m_l218_218066


namespace common_ratio_geometric_series_l218_218870

theorem common_ratio_geometric_series 
  (a : ℚ) (b : ℚ) (r : ℚ)
  (h_a : a = 4 / 5)
  (h_b : b = -5 / 12)
  (h_r : r = b / a) :
  r = -25 / 48 :=
by sorry

end common_ratio_geometric_series_l218_218870


namespace minimum_value_ineq_l218_218285

theorem minimum_value_ineq (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_prod : a * b * c = 1) :
  (a^2 / b) + (b^2 / c) + (c^2 / a) ≥ 3 :=
by
  sorry

end minimum_value_ineq_l218_218285


namespace infinite_fixpoints_l218_218290

variable {f : ℕ+ → ℕ+}
variable (H : ∀ (m n : ℕ+), (∃ k : ℕ+ , k ≤ f n ∧ n ∣ f (m + k)) ∧ (∀ j : ℕ+ , j ≤ f n → j ≠ k → ¬ n ∣ f (m + j)))

theorem infinite_fixpoints : ∃ᶠ n in at_top, f n = n :=
sorry

end infinite_fixpoints_l218_218290


namespace max_product_is_negative_one_l218_218937

def f (x : ℝ) : ℝ := sorry    -- Assume some function f
def g (x : ℝ) : ℝ := sorry    -- Assume some function g

theorem max_product_is_negative_one (h_f_range : ∀ y, 1 ≤ y ∧ y ≤ 6 → ∃ x, f x = y) 
    (h_g_range : ∀ y, -4 ≤ y ∧ y ≤ -1 → ∃ x, g x = y) : 
    ∃ b, b = -1 ∧ ∀ x, f x * g x ≤ b :=
sorry

end max_product_is_negative_one_l218_218937


namespace not_all_divisible_by_6_have_prime_neighbors_l218_218339

theorem not_all_divisible_by_6_have_prime_neighbors :
  ¬ ∀ n : ℕ, (6 ∣ n) → (Prime (n - 1) ∨ Prime (n + 1)) := by
  sorry

end not_all_divisible_by_6_have_prime_neighbors_l218_218339


namespace voting_problem_l218_218897

theorem voting_problem (x y x' y' : ℕ) (m : ℕ) (h1 : x + y = 500) (h2 : y > x)
    (h3 : y - x = m) (h4 : x' = (10 * y) / 9) (h5 : x' + y' = 500)
    (h6 : x' - y' = 3 * m) :
    x' - x = 59 := 
sorry

end voting_problem_l218_218897


namespace difference_between_numbers_l218_218826

theorem difference_between_numbers :
  ∃ X Y : ℕ, 
    100 ≤ X ∧ X < 1000 ∧
    100 ≤ Y ∧ Y < 1000 ∧
    X + Y = 999 ∧
    1000 * X + Y = 6 * (1000 * Y + X) ∧
    (X - Y = 715 ∨ Y - X = 715) :=
by
  sorry

end difference_between_numbers_l218_218826


namespace simplify_fraction_l218_218963

theorem simplify_fraction :
  ( (5^2010)^2 - (5^2008)^2 ) / ( (5^2009)^2 - (5^2007)^2 ) = 25 := by
  sorry

end simplify_fraction_l218_218963


namespace julia_played_more_kids_l218_218461

variable (kidsPlayedMonday : Nat) (kidsPlayedTuesday : Nat)

theorem julia_played_more_kids :
  kidsPlayedMonday = 11 →
  kidsPlayedTuesday = 12 →
  kidsPlayedTuesday - kidsPlayedMonday = 1 :=
by
  intros hMonday hTuesday
  sorry

end julia_played_more_kids_l218_218461


namespace find_savings_l218_218574

-- Definitions of given conditions
def income : ℕ := 10000
def ratio_income_expenditure : ℕ × ℕ := (10, 8)

-- Proving the savings based on given conditions
theorem find_savings (income : ℕ) (ratio_income_expenditure : ℕ × ℕ) :
  let expenditure := (ratio_income_expenditure.2 * income) / ratio_income_expenditure.1
  let savings := income - expenditure
  savings = 2000 :=
by
  sorry

end find_savings_l218_218574


namespace correct_operation_is_a_l218_218601

theorem correct_operation_is_a (a b : ℝ) : 
  (a^4 * a^3 = a^7) ∧ 
  ((a^2)^3 ≠ a^5) ∧ 
  (3 * a^2 - a^2 ≠ 2) ∧ 
  ((a - b)^2 ≠ a^2 - b^2) := 
by {
  -- Here, you would fill in the proof
  sorry
}

end correct_operation_is_a_l218_218601


namespace smallest_b_l218_218741

open Real

theorem smallest_b (a b : ℝ) (h1 : 2 < a) (h2 : a < b)
  (h3 : ¬ ∃ A B C : ℝ, A > 0 ∧ B > 0 ∧ C > 0 ∧ (A = 2 ∧ B = a ∧ C = b ∨ A = 2 ∧ B = b ∧ C = a ∨ A = a ∧ B = b ∧ C = 2) ∧ A + B > C ∧ A + C > B ∧ B + C > A)
  (h4 : ¬ ∃ A B C : ℝ, A > 0 ∧ B > 0 ∧ C > 0 ∧ (A = 1 / b ∧ B = 1 / a ∧ C = 2 ∨ A = 1 / a ∧ B = 1 / b ∧ C = 2 ∨ A = 1 / b ∧ B = 2 ∧ C = 1 / a ∨ A = 1 / a ∧ B = 2 ∧ C = 1 / b ∨ A = 2 ∧ B = 1 / a ∧ C = 1 / b ∨ A = 2 ∧ B = 1 / b ∧ C = 1 / a) ∧ A + B > C ∧ A + C > B ∧ B + C > A) :
  b = 2 := 
sorry

end smallest_b_l218_218741


namespace circle_chords_intersect_radius_square_l218_218650

theorem circle_chords_intersect_radius_square
  (r : ℝ) -- The radius of the circle
  (AB CD BP : ℝ) -- The lengths of chords AB, CD, and segment BP
  (angle_APD : ℝ) -- The angle ∠APD in degrees
  (AB_len : AB = 8)
  (CD_len : CD = 12)
  (BP_len : BP = 10)
  (angle_APD_val : angle_APD = 60) :
  r^2 = 91 := 
sorry

end circle_chords_intersect_radius_square_l218_218650


namespace speed_ratio_correct_l218_218806

noncomputable def boat_speed_still_water := 12 -- Boat's speed in still water (in mph)
noncomputable def current_speed := 4 -- Current speed of the river (in mph)

-- Calculate the downstream speed
noncomputable def downstream_speed := boat_speed_still_water + current_speed

-- Calculate the upstream speed
noncomputable def upstream_speed := boat_speed_still_water - current_speed

-- Assume a distance for the trip (1 mile each up and down)
noncomputable def distance := 1

-- Calculate time for downstream
noncomputable def time_downstream := distance / downstream_speed

-- Calculate time for upstream
noncomputable def time_upstream := distance / upstream_speed

-- Calculate total time for the round trip
noncomputable def total_time := time_downstream + time_upstream

-- Calculate total distance for the round trip
noncomputable def total_distance := 2 * distance

-- Calculate the average speed for the round trip
noncomputable def avg_speed_trip := total_distance / total_time

-- Calculate the ratio of average speed to speed in still water
noncomputable def speed_ratio := avg_speed_trip / boat_speed_still_water

theorem speed_ratio_correct : speed_ratio = 8/9 := by
  sorry

end speed_ratio_correct_l218_218806


namespace no_term_in_sequence_is_3_alpha_5_beta_l218_218451

theorem no_term_in_sequence_is_3_alpha_5_beta :
  ∀ (v : ℕ → ℕ),
    v 0 = 0 →
    v 1 = 1 →
    (∀ n, 1 ≤ n → v (n + 1) = 8 * v n * v (n - 1)) →
    ∀ n, ∀ (α β : ℕ), α > 0 → β > 0 → v n ≠ 3^α * 5^β := by
  intros v h0 h1 recurrence n α β hα hβ
  sorry

end no_term_in_sequence_is_3_alpha_5_beta_l218_218451


namespace prime_gt_10_exists_m_n_l218_218201

theorem prime_gt_10_exists_m_n (p : ℕ) (hp_prime : Nat.Prime p) (hp_gt_10 : p > 10) :
  ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ m + n < p ∧ p ∣ (5^m * 7^n - 1) :=
by
  sorry

end prime_gt_10_exists_m_n_l218_218201


namespace exists_bounding_constant_M_l218_218619

variable (α : ℝ) (a : ℕ → ℝ)
variable (hα : α > 1)
variable (h_seq : ∀ n : ℕ, n > 0 →
  a n.succ = a n + (a n / n) ^ α)

theorem exists_bounding_constant_M (h_a1 : 0 < a 1 ∧ a 1 < 1) : 
  ∃ M, ∀ n > 0, a n ≤ M := 
sorry

end exists_bounding_constant_M_l218_218619


namespace total_words_in_poem_l218_218572

theorem total_words_in_poem (s l w : ℕ) (h1 : s = 35) (h2 : l = 15) (h3 : w = 12) : 
  s * l * w = 6300 := 
by 
  -- the proof will be inserted here
  sorry

end total_words_in_poem_l218_218572


namespace number_of_ways_to_assign_roles_l218_218103

theorem number_of_ways_to_assign_roles : 
  let male_roles := 3
  let female_roles := 2
  let either_gender_roles := 1
  let men := 4
  let women := 5
  let total_roles := male_roles + female_roles + either_gender_roles
  let ways_to_assign_males := men * (men-1) * (men-2)
  let ways_to_assign_females := women * (women-1)
  let remaining_actors := men + women - male_roles - female_roles
  let ways_to_assign_either_gender := remaining_actors
  let total_ways := ways_to_assign_males * ways_to_assign_females * ways_to_assign_either_gender

  total_ways = 1920 :=
by
  sorry

end number_of_ways_to_assign_roles_l218_218103


namespace fraction_inequality_solution_l218_218889

theorem fraction_inequality_solution (x : ℝ) :
  (x < -5 ∨ x ≥ 2) ↔ (x-2) / (x+5) ≥ 0 :=
sorry

end fraction_inequality_solution_l218_218889


namespace polynomial_bound_l218_218626

theorem polynomial_bound (a b c d : ℝ) 
  (h1 : ∀ x : ℝ, |x| ≤ 1 → |a * x^3 + b * x^2 + c * x + d| ≤ 1) : 
  |a| + |b| + |c| + |d| ≤ 7 := 
sorry

end polynomial_bound_l218_218626


namespace angle_no_complement_greater_than_90_l218_218546

-- Definition of angle
def angle (A : ℝ) : Prop := 
  A = 100 + (15 / 60)

-- Definition of complement
def has_complement (A : ℝ) : Prop :=
  A < 90

-- Theorem: Angles greater than 90 degrees do not have complements
theorem angle_no_complement_greater_than_90 {A : ℝ} (h: angle A) : ¬ has_complement A :=
by sorry

end angle_no_complement_greater_than_90_l218_218546


namespace exists_rational_non_integer_xy_no_rational_non_integer_xy_l218_218924

-- Part (a)
theorem exists_rational_non_integer_xy 
  (x y : ℚ) (h1 : ¬ ∃ z : ℤ, x = z ∧ y = z) : 
  (∃ x y : ℚ, ¬(∃ z : ℤ, x = z ∨ y = z) ∧ 
   ∃ z1 z2 : ℤ, 19 * x + 8 * y = ↑z1 ∧ 8 * x + 3 * y = ↑z2) :=
sorry

-- Part (b)
theorem no_rational_non_integer_xy 
  (x y : ℚ) (h1 : ¬ ∃ z : ℤ, x = z ∧ y = z) : 
  ¬ ∃ x y : ℚ, ¬(∃ z : ℤ, x = z ∨ y = z) ∧ 
  ∃ z1 z2 : ℤ, 19 * x^2 + 8 * y^2 = ↑z1 ∧ 8 * x^2 + 3 * y^2 = ↑z2 :=
sorry

end exists_rational_non_integer_xy_no_rational_non_integer_xy_l218_218924


namespace fraction_zero_solution_l218_218825

theorem fraction_zero_solution (x : ℝ) (h1 : x - 5 = 0) (h2 : 4 * x^2 - 1 ≠ 0) : x = 5 :=
by {
  sorry -- The proof
}

end fraction_zero_solution_l218_218825


namespace largest_integer_x_divisible_l218_218852

theorem largest_integer_x_divisible (x : ℤ) : 
  (∃ x : ℤ, (x^2 + 3 * x + 8) % (x - 2) = 0 ∧ x ≤ 1) → x = 1 :=
sorry

end largest_integer_x_divisible_l218_218852


namespace find_certain_number_l218_218198

theorem find_certain_number :
  ∃ C, ∃ A B, (A + B = 15) ∧ (A = 7) ∧ (C * B = 5 * A - 11) ∧ (C = 3) :=
by
  sorry

end find_certain_number_l218_218198


namespace collinear_points_l218_218382

variables (a b : ℝ × ℝ) (A B C D : ℝ × ℝ)

-- Define the vectors
noncomputable def vec_AB : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)
noncomputable def vec_BC : ℝ × ℝ := (2 * a.1 + 8 * b.1, 2 * a.2 + 8 * b.2)
noncomputable def vec_CD : ℝ × ℝ := (3 * (a.1 - b.1), 3 * (a.2 - b.2))

-- Define the collinearity condition
def collinear (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k * v.1, k * v.2)

-- Translate the problem statement into Lean
theorem collinear_points (h₀ : a ≠ (0, 0)) (h₁ : b ≠ (0, 0)) (h₂ : ¬ (a.1 * b.2 - a.2 * b.1 = 0)):
  collinear (6 * (a.1 + b.1), 6 * (a.2 + b.2)) (5 * (a.1 + b.1, a.2 + b.2)) :=
sorry

end collinear_points_l218_218382


namespace anoop_joined_after_6_months_l218_218232

theorem anoop_joined_after_6_months (arjun_investment : ℕ) (anoop_investment : ℕ) (months_in_year : ℕ)
  (arjun_time : ℕ) (anoop_time : ℕ) :
  arjun_investment * arjun_time = anoop_investment * anoop_time →
  anoop_investment = 2 * arjun_investment →
  arjun_time = months_in_year →
  anoop_time + arjun_time = months_in_year →
  anoop_time = 6 :=
by sorry

end anoop_joined_after_6_months_l218_218232


namespace area_of_rectangular_field_l218_218323

def length (L : ℝ) : Prop := L > 0
def breadth (L : ℝ) (B : ℝ) : Prop := B = 0.6 * L
def perimeter (L : ℝ) (B : ℝ) : Prop := 2 * L + 2 * B = 800
def area (L : ℝ) (B : ℝ) (A : ℝ) : Prop := A = L * B

theorem area_of_rectangular_field (L B A : ℝ) 
  (h1 : breadth L B) 
  (h2 : perimeter L B) : 
  area L B 37500 :=
sorry

end area_of_rectangular_field_l218_218323


namespace find_k_l218_218236

open BigOperators

def a (n : ℕ) : ℕ := 2 ^ n

theorem find_k (k : ℕ) (h : a (k+1) + a (k+2) + a (k+3) + a (k+4) + a (k+5) + a (k+6) + a (k+7) + a (k+8) + a (k+9) + a (k+10) = 2 ^ 15 - 2 ^ 5) : k = 4 :=
sorry

end find_k_l218_218236


namespace number_of_BMWs_sold_l218_218901

-- Defining the percentages of Mercedes, Toyota, and Acura cars sold
def percentageMercedes : ℕ := 18
def percentageToyota  : ℕ := 25
def percentageAcura   : ℕ := 15

-- Defining the total number of cars sold
def totalCars : ℕ := 250

-- The theorem to be proved
theorem number_of_BMWs_sold : (totalCars * (100 - (percentageMercedes + percentageToyota + percentageAcura)) / 100) = 105 := by
  sorry -- Proof to be filled in later

end number_of_BMWs_sold_l218_218901


namespace line_slope_intercept_l218_218761

theorem line_slope_intercept :
  ∃ k b, (∀ x y : ℝ, 2 * x - 3 * y + 6 = 0 → y = k * x + b) ∧ k = 2/3 ∧ b = 2 :=
by
  sorry

end line_slope_intercept_l218_218761


namespace solution_set_of_inequality_l218_218369

theorem solution_set_of_inequality:
  {x : ℝ | x^2 - |x-1| - 1 ≤ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} :=
  sorry

end solution_set_of_inequality_l218_218369


namespace prob_no_rain_correct_l218_218029

-- Define the probability of rain on each of the next five days
def prob_rain_each_day : ℚ := 1 / 2

-- Define the probability of no rain on a single day
def prob_no_rain_one_day : ℚ := 1 - prob_rain_each_day

-- Define the probability of no rain in any of the next five days
def prob_no_rain_five_days : ℚ := prob_no_rain_one_day ^ 5

-- Theorem statement
theorem prob_no_rain_correct : prob_no_rain_five_days = 1 / 32 := by
  sorry

end prob_no_rain_correct_l218_218029


namespace solution_interval_l218_218109

noncomputable def f (x : ℝ) : ℝ := Real.log x + x - 4

theorem solution_interval :
  ∃ x_0, f x_0 = 0 ∧ 2 < x_0 ∧ x_0 < 3 :=
by
  sorry

end solution_interval_l218_218109


namespace resulting_solution_percentage_l218_218887

theorem resulting_solution_percentage (w_original: ℝ) (w_replaced: ℝ) (c_original: ℝ) (c_new: ℝ) :
  c_original = 0.9 → w_replaced = 0.7142857142857143 → c_new = 0.2 →
  (0.2571428571428571 + 0.14285714285714285) / (0.2857142857142857 + 0.7142857142857143) * 100 = 40 := 
by
  intros h1 h2 h3
  sorry

end resulting_solution_percentage_l218_218887


namespace jake_weight_l218_218331

variable (J S : ℕ)

theorem jake_weight (h1 : J - 15 = 2 * S) (h2 : J + S = 132) : J = 93 := by
  sorry

end jake_weight_l218_218331


namespace intersection_point_sum_l218_218828

noncomputable def h : ℝ → ℝ := sorry
noncomputable def j : ℝ → ℝ := sorry

axiom h2 : h 2 = 2
axiom j2 : j 2 = 2
axiom h4 : h 4 = 6
axiom j4 : j 4 = 6
axiom h6 : h 6 = 12
axiom j6 : j 6 = 12
axiom h8 : h 8 = 12
axiom j8 : j 8 = 12

theorem intersection_point_sum :
  (∃ x, h (x + 2) = j (2 * x)) →
  (h (2 + 2) = j (2 * 2) ∨ h (4 + 2) = j (2 * 4)) →
  (h (4) = 6 ∧ j (4) = 6 ∧ h 6 = 12 ∧ j 8 = 12) →
  (∃ x, (x = 2 ∧ (x + h (x + 2) = 8) ∨ x = 4 ∧ (x + h (x + 2) = 16))) :=
by
  sorry

end intersection_point_sum_l218_218828


namespace mike_total_cards_l218_218425

variable (original_cards : ℕ) (birthday_cards : ℕ)

def initial_cards : ℕ := 64
def received_cards : ℕ := 18

theorem mike_total_cards :
  original_cards = 64 →
  birthday_cards = 18 →
  original_cards + birthday_cards = 82 :=
by
  intros
  sorry

end mike_total_cards_l218_218425


namespace garage_sale_items_count_l218_218481

theorem garage_sale_items_count :
  (16 + 22) + 1 = 38 :=
by
  -- proof goes here
  sorry

end garage_sale_items_count_l218_218481


namespace smallest_integer_with_divisors_l218_218624

theorem smallest_integer_with_divisors :
  ∃ (n : ℕ), 
    (∀ d : ℕ, d ∣ n → d % 2 = 1 → (∃! k : ℕ, d = (3 ^ k) * 5 ^ (7 - k))) ∧ 
    (∀ d : ℕ, d ∣ n → d % 2 = 0 → (∃! k : ℕ, d = 2 ^ k * m)) ∧ 
    (n = 1080) :=
sorry

end smallest_integer_with_divisors_l218_218624


namespace determine_x_l218_218767

theorem determine_x (p q : ℝ) (hpq : p ≠ q) : 
  ∃ (c d : ℝ), (x = c*p + d*q) ∧ c = 2 ∧ d = -2 :=
by 
  sorry

end determine_x_l218_218767


namespace total_cases_after_third_day_l218_218050

-- Definitions for the conditions
def day1_cases : Nat := 2000
def day2_new_cases : Nat := 500
def day2_recoveries : Nat := 50
def day3_new_cases : Nat := 1500
def day3_recoveries : Nat := 200

-- Theorem stating the total number of cases after the third day
theorem total_cases_after_third_day : day1_cases + (day2_new_cases - day2_recoveries) + (day3_new_cases - day3_recoveries) = 3750 :=
by
  sorry

end total_cases_after_third_day_l218_218050


namespace f_g_of_2_eq_4_l218_218947

def f (x : ℝ) : ℝ := x^2 - 2*x + 1
def g (x : ℝ) : ℝ := 2*x - 5

theorem f_g_of_2_eq_4 : f (g 2) = 4 := by
  sorry

end f_g_of_2_eq_4_l218_218947


namespace sqrt_expression_meaningful_domain_l218_218025

theorem sqrt_expression_meaningful_domain {x : ℝ} (h : 3 - x ≥ 0) : x ≤ 3 := by
  sorry

end sqrt_expression_meaningful_domain_l218_218025


namespace statement_A_statement_E_statement_B_false_statement_C_false_statement_D_false_l218_218432

theorem statement_A : ∃ n : ℤ, 20 = 4 * n := by 
  sorry

theorem statement_E : ∃ n : ℤ, 180 = 9 * n := by 
  sorry

theorem statement_B_false : ¬ (19 ∣ 57) := by 
  sorry

theorem statement_C_false : 30 ∣ 90 := by 
  sorry

theorem statement_D_false : 17 ∣ 51 := by 
  sorry

end statement_A_statement_E_statement_B_false_statement_C_false_statement_D_false_l218_218432


namespace range_of_set_is_8_l218_218362

theorem range_of_set_is_8 (a b c : ℕ) 
  (h1 : (a + b + c) / 3 = 6) 
  (h2 : b = 6) 
  (h3 : a = 2) 
  : max a (max b c) - min a (min b c) = 8 := 
by sorry

end range_of_set_is_8_l218_218362


namespace cubes_with_two_or_three_blue_faces_l218_218989

theorem cubes_with_two_or_three_blue_faces 
  (four_inch_cube : ℝ)
  (painted_blue_faces : ℝ)
  (one_inch_cubes : ℝ) :
  (four_inch_cube = 4) →
  (painted_blue_faces = 6) →
  (one_inch_cubes = 64) →
  (num_cubes_with_two_or_three_blue_faces = 32) :=
sorry

end cubes_with_two_or_three_blue_faces_l218_218989


namespace floor_of_neg_seven_fourths_l218_218398

theorem floor_of_neg_seven_fourths : 
  Int.floor (-7 / 4 : ℚ) = -2 := 
by sorry

end floor_of_neg_seven_fourths_l218_218398


namespace part1_part2_l218_218528

noncomputable def f (x : ℝ) : ℝ := |3 * x + 2|

theorem part1 (x : ℝ): f x < 6 - |x - 2| ↔ (-3/2 < x ∧ x < 1) :=
by sorry

theorem part2 (a : ℝ) (m n : ℝ) (h₁ : 0 < m) (h₂ : 0 < n) (h₃ : m + n = 4) (h₄ : 0 < a) (h₅ : ∀ x, |x - a| - f x ≤ 1/m + 1/n) :
    0 < a ∧ a ≤ 1/3 :=
by sorry

end part1_part2_l218_218528


namespace smallest_N_l218_218821

theorem smallest_N (N : ℕ) (hN : N > 70) (hDiv : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_N_l218_218821


namespace exists_prime_and_positive_integer_l218_218649

theorem exists_prime_and_positive_integer (a : ℕ) (h : a = 9) : 
  ∃ (p : ℕ) (hp : Nat.Prime p) (b : ℕ) (hb : b ≥ 2), (a^p - a) / p = b^2 := 
  by
  sorry

end exists_prime_and_positive_integer_l218_218649


namespace equal_phrases_impossible_l218_218139

-- Define the inhabitants and the statements they make.
def inhabitants : ℕ := 1234

-- Define what it means to be a knight or a liar.
inductive Person
| knight : Person
| liar : Person

-- Define the statements "He is a knight!" and "He is a liar!"
inductive Statement
| is_knight : Statement
| is_liar : Statement

-- Define the pairings and types of statements 
def pairings (inhabitant1 inhabitant2 : Person) : Statement :=
match inhabitant1, inhabitant2 with
| Person.knight, Person.knight => Statement.is_knight
| Person.liar, Person.liar => Statement.is_knight
| Person.knight, Person.liar => Statement.is_liar
| Person.liar, Person.knight => Statement.is_knight

-- Define the total number of statements
def total_statements (pairs : ℕ) : ℕ := 2 * pairs

-- Theorem stating the mathematical equivalent proof problem
theorem equal_phrases_impossible :
  ¬ ∃ n : ℕ, n = inhabitants / 2 ∧ total_statements n = inhabitants ∧
    (pairings Person.knight Person.liar = Statement.is_knight ∧
     pairings Person.liar Person.knight = Statement.is_knight ∧
     (pairings Person.knight Person.knight = Statement.is_knight ∧
      pairings Person.liar Person.liar = Statement.is_knight) ∨
      (pairings Person.knight Person.liar = Statement.is_liar ∧
       pairings Person.liar Person.knight = Statement.is_liar)) :=
sorry

end equal_phrases_impossible_l218_218139


namespace vector_sum_l218_218836

-- Define the vectors a and b according to the conditions.
def a : (ℝ × ℝ) := (2, 1)
def b : (ℝ × ℝ) := (-3, 4)

-- Prove that the vector sum a + b is (-1, 5).
theorem vector_sum : (a.1 + b.1, a.2 + b.2) = (-1, 5) :=
by
  -- include the proof later
  sorry

end vector_sum_l218_218836


namespace radius_of_shorter_cylinder_l218_218732

theorem radius_of_shorter_cylinder (h r : ℝ) (V_s V_t : ℝ) (π : ℝ) : 
  V_s = 500 → 
  V_t = 500 → 
  V_t = π * 5^2 * 4 * h → 
  V_s = π * r^2 * h → 
  r = 10 :=
by 
  sorry

end radius_of_shorter_cylinder_l218_218732


namespace molecular_weight_proof_l218_218727

noncomputable def molecular_weight_C7H6O2 := 
  (7 * 12.01) + (6 * 1.008) + (2 * 16.00) -- molecular weight of one mole of C7H6O2

noncomputable def total_molecular_weight_9_moles := 
  9 * molecular_weight_C7H6O2 -- total molecular weight of 9 moles of C7H6O2

theorem molecular_weight_proof : 
  total_molecular_weight_9_moles = 1099.062 := 
by
  sorry

end molecular_weight_proof_l218_218727


namespace final_price_of_pencil_l218_218090

-- Define the initial constants
def initialCost : ℝ := 4.00
def christmasDiscount : ℝ := 0.63
def seasonalDiscountRate : ℝ := 0.07
def finalDiscountRate : ℝ := 0.05
def taxRate : ℝ := 0.065

-- Define the steps of the problem concisely
def priceAfterChristmasDiscount := initialCost - christmasDiscount
def priceAfterSeasonalDiscount := priceAfterChristmasDiscount * (1 - seasonalDiscountRate)
def priceAfterFinalDiscount := priceAfterSeasonalDiscount * (1 - finalDiscountRate)
def finalPrice := priceAfterFinalDiscount * (1 + taxRate)

-- The theorem to be proven
theorem final_price_of_pencil :
  abs (finalPrice - 3.17) < 0.01 := by
  sorry

end final_price_of_pencil_l218_218090


namespace smallest_two_digit_integer_l218_218191

theorem smallest_two_digit_integer (n a b : ℕ) (h1 : n = 10 * a + b) (h2 : 2 * n = 10 * b + a + 5) (h3 : 1 ≤ a) (h4 : a ≤ 9) (h5 : 0 ≤ b) (h6 : b ≤ 9) : n = 69 := 
by 
  sorry

end smallest_two_digit_integer_l218_218191


namespace midpoint_set_of_segments_eq_circle_l218_218464

-- Define the existence of skew perpendicular lines with given properties
variable (a d : ℝ)

-- Conditions: Distance between lines is a, segment length is d
-- The coordinates system configuration
-- Point on the first line: (x, 0, 0)
-- Point on the second line: (0, y, a)
def are_midpoints_of_segments_of_given_length
  (p : ℝ × ℝ × ℝ) : Prop :=
  ∃ (x y : ℝ), 
    p = (x / 2, y / 2, a / 2) ∧ 
    x^2 + y^2 = d^2 - a^2

-- Proof statement
theorem midpoint_set_of_segments_eq_circle :
  { p : ℝ × ℝ × ℝ | are_midpoints_of_segments_of_given_length a d p } =
  { p : ℝ × ℝ × ℝ | ∃ (r : ℝ), p = (r * (d^2 - a^2) / (2*d), r * (d^2 - a^2) / (2*d), a / 2)
    ∧ r^2 * (d^2 - a^2) = (d^2 - a^2) } :=
sorry

end midpoint_set_of_segments_eq_circle_l218_218464


namespace max_stickers_single_player_l218_218479

noncomputable def max_stickers (num_players : ℕ) (average_stickers : ℕ) : ℕ :=
  let total_stickers := num_players * average_stickers
  let min_stickers_one_player := 1
  let min_stickers_others := (num_players - 1) * min_stickers_one_player
  total_stickers - min_stickers_others

theorem max_stickers_single_player : 
  ∀ (num_players average_stickers : ℕ), 
    num_players = 25 → 
    average_stickers = 4 →
    ∀ player_stickers : ℕ, player_stickers ≤ max_stickers num_players average_stickers → player_stickers = 76 :=
    by
      intro num_players average_stickers players_eq avg_eq player_stickers player_le_max
      sorry

end max_stickers_single_player_l218_218479


namespace jenna_remaining_money_l218_218133

theorem jenna_remaining_money (m c : ℝ) (h : (1 / 4) * m = (1 / 2) * c) : (m - c) / m = 1 / 2 :=
by
  sorry

end jenna_remaining_money_l218_218133


namespace lipstick_cost_is_correct_l218_218791

noncomputable def cost_of_lipstick (palette_cost : ℝ) (num_palettes : ℝ) (hair_color_cost : ℝ) (num_hair_colors : ℝ) (total_paid : ℝ) (num_lipsticks : ℝ) : ℝ :=
  let total_palette_cost := num_palettes * palette_cost
  let total_hair_color_cost := num_hair_colors * hair_color_cost
  let remaining_amount := total_paid - (total_palette_cost + total_hair_color_cost)
  remaining_amount / num_lipsticks

theorem lipstick_cost_is_correct :
  cost_of_lipstick 15 3 4 3 67 4 = 2.5 :=
by
  sorry

end lipstick_cost_is_correct_l218_218791


namespace xy_eq_one_l218_218799

theorem xy_eq_one (x y : ℝ) (h : x + y = (1 / x) + (1 / y) ∧ x + y ≠ 0) : x * y = 1 := by
  sorry

end xy_eq_one_l218_218799


namespace negation_equiv_l218_218119

variable (p : Prop) [Nonempty ℝ]

def proposition := ∃ x : ℝ, Real.exp x - x - 1 ≤ 0

def negation_of_proposition : Prop := ∀ x : ℝ, Real.exp x - x - 1 > 0

theorem negation_equiv
  (h : proposition = p) : (¬ proposition) = negation_of_proposition := by
  sorry

end negation_equiv_l218_218119


namespace probability_of_two_co_presidents_l218_218172

noncomputable section

def binomial (n k : ℕ) : ℕ :=
  if h : n ≥ k then Nat.choose n k else 0

def club_prob (n : ℕ) : ℚ :=
  (binomial (n-2) 2 : ℚ) / (binomial n 4 : ℚ)

def total_probability : ℚ :=
  (1/4 : ℚ) * (club_prob 6 + club_prob 8 + club_prob 9 + club_prob 10)

theorem probability_of_two_co_presidents : total_probability = 0.2286 := by
  -- We expect this to be true based on the given solution
  sorry

end probability_of_two_co_presidents_l218_218172


namespace initial_average_mark_l218_218405

-- Define the initial conditions
def num_students : ℕ := 9
def excluded_students_avg : ℕ := 44
def remaining_students_avg : ℕ := 80

-- Define the variables for total marks we calculated in the solution
def total_marks_initial := num_students * (num_students * excluded_students_avg / 5 + remaining_students_avg / (num_students - 5) * (num_students - 5))

-- The theorem we need to prove:
theorem initial_average_mark :
  (num_students * (excluded_students_avg * 5 + remaining_students_avg * (num_students - 5))) / num_students = 60 := 
  by
  -- step-by-step solution proof could go here, but we use sorry as placeholder
  sorry

end initial_average_mark_l218_218405


namespace no_such_abc_exists_l218_218470

theorem no_such_abc_exists : ¬ ∃ (a b c : ℝ), ∀ (x y : ℝ),
  |x + a| + |x + y + b| + |y + c| > |x| + |x + y| + |y| :=
by
  sorry

end no_such_abc_exists_l218_218470


namespace valid_two_digit_numbers_l218_218614

def is_valid_two_digit_number_pair (a b : ℕ) : Prop :=
  0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ a > b ∧ (Nat.gcd (10 * a + b) (10 * b + a) = a^2 - b^2)

theorem valid_two_digit_numbers :
  (is_valid_two_digit_number_pair 2 1 ∨ is_valid_two_digit_number_pair 5 4) ∧
  ∀ a b, is_valid_two_digit_number_pair a b → (a = 2 ∧ b = 1 ∨ a = 5 ∧ b = 4) :=
by
  sorry

end valid_two_digit_numbers_l218_218614


namespace parabola_directrix_l218_218946

theorem parabola_directrix (y : ℝ) : 
  x = -((1:ℝ)/4)*y^2 → x = 1 :=
by 
  sorry

end parabola_directrix_l218_218946


namespace find_first_term_of_arithmetic_progression_l218_218406

-- Definitions for the proof
def arithmetic_progression_first_term (L n d : ℕ) : ℕ :=
  L - (n - 1) * d

-- Theorem stating the proof problem
theorem find_first_term_of_arithmetic_progression (L n d : ℕ) (hL : L = 62) (hn : n = 31) (hd : d = 2) :
  arithmetic_progression_first_term L n d = 2 :=
by
  -- proof omitted
  sorry

end find_first_term_of_arithmetic_progression_l218_218406


namespace eval_expression_l218_218618

theorem eval_expression : 3 - (-1) + 4 - 5 + (-6) - (-7) + 8 - 9 = 3 := 
  sorry

end eval_expression_l218_218618


namespace range_of_reciprocal_sum_l218_218216

theorem range_of_reciprocal_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) :
    4 ≤ (1/x + 1/y) :=
by
  sorry

end range_of_reciprocal_sum_l218_218216


namespace probability_green_jelly_bean_l218_218088

theorem probability_green_jelly_bean :
  let red := 10
  let green := 9
  let yellow := 5
  let blue := 7
  let total := red + green + yellow + blue
  (green : ℚ) / (total : ℚ) = 9 / 31 := by
  sorry

end probability_green_jelly_bean_l218_218088


namespace find_k_l218_218041

theorem find_k (a b k : ℝ) (h1 : a ≠ b ∨ a = b)
    (h2 : a^2 - 12 * a + k + 2 = 0)
    (h3 : b^2 - 12 * b + k + 2 = 0)
    (h4 : 4^2 - 12 * 4 + k + 2 = 0) :
    k = 34 ∨ k = 30 :=
by
  sorry

end find_k_l218_218041


namespace percent_of_ducks_among_non_swans_l218_218768

theorem percent_of_ducks_among_non_swans
  (total_birds : ℕ) 
  (percent_ducks percent_swans percent_eagles percent_sparrows : ℕ)
  (h1 : percent_ducks = 40) 
  (h2 : percent_swans = 20) 
  (h3 : percent_eagles = 15) 
  (h4 : percent_sparrows = 25)
  (h_sum : percent_ducks + percent_swans + percent_eagles + percent_sparrows = 100) :
  (percent_ducks * 100) / (100 - percent_swans) = 50 :=
by
  sorry

end percent_of_ducks_among_non_swans_l218_218768


namespace volleyball_team_starters_l218_218858

-- Define the team and the triplets
def total_players : ℕ := 14
def triplet_count : ℕ := 3
def remaining_players : ℕ := total_players - triplet_count

-- Define the binomial coefficient function
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Define the problem
theorem volleyball_team_starters : 
  C total_players 6 - C remaining_players 3 = 2838 :=
by sorry

end volleyball_team_starters_l218_218858


namespace white_paint_amount_l218_218230

theorem white_paint_amount (total_blue_paint additional_blue_paint total_mix blue_parts red_parts white_parts green_parts : ℕ) 
    (h_ratio: blue_parts = 7 ∧ red_parts = 2 ∧ white_parts = 1 ∧ green_parts = 1)
    (total_blue_paint_eq: total_blue_paint = 140)
    (max_total_mix: additional_blue_paint ≤ 220 - total_blue_paint) 
    : (white_parts * (total_blue_paint / blue_parts)) = 20 := 
by 
  sorry

end white_paint_amount_l218_218230


namespace number_is_160_l218_218711

theorem number_is_160 (x : ℝ) (h : x / 5 + 4 = x / 4 - 4) : x = 160 :=
by
  sorry

end number_is_160_l218_218711


namespace solve_for_x_l218_218325

theorem solve_for_x (x : ℕ) (h : (1 / 8) * 2 ^ 36 = 8 ^ x) : x = 11 :=
by
sorry

end solve_for_x_l218_218325


namespace pages_filled_with_images_ratio_l218_218268

theorem pages_filled_with_images_ratio (total_pages intro_pages text_pages : ℕ) 
  (h_total : total_pages = 98)
  (h_intro : intro_pages = 11)
  (h_text : text_pages = 19)
  (h_blank : 2 * text_pages = total_pages - intro_pages - 2 * text_pages) :
  (total_pages - intro_pages - text_pages - text_pages) / total_pages = 1 / 2 :=
by
  sorry

end pages_filled_with_images_ratio_l218_218268


namespace volume_of_solid_rotation_l218_218757

noncomputable def volume_of_solid := 
  (∫ y in (0:ℝ)..(1:ℝ), (y^(2/3) - y^2)) * Real.pi 

theorem volume_of_solid_rotation :
  volume_of_solid = (4 * Real.pi / 15) :=
by
  sorry

end volume_of_solid_rotation_l218_218757


namespace suzanna_textbooks_page_total_l218_218973

theorem suzanna_textbooks_page_total :
  let H := 160
  let G := H + 70
  let M := (H + G) / 2
  let S := 2 * H
  let L := (H + G) - 30
  let E := M + L + 25
  H + G + M + S + L + E = 1845 := by
  sorry

end suzanna_textbooks_page_total_l218_218973


namespace find_m_l218_218493

-- Definitions based on conditions
def Point (α : Type) := α × α

def A : Point ℝ := (2, -3)
def B : Point ℝ := (4, 3)
def C (m : ℝ) : Point ℝ := (5, m)

-- The collinearity condition
def collinear (p1 p2 p3 : Point ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x2) = (y3 - y2) * (x2 - x1)

-- The proof problem
theorem find_m (m : ℝ) : collinear A B (C m) → m = 6 :=
by
  sorry

end find_m_l218_218493


namespace find_y_values_l218_218535

theorem find_y_values (x : ℝ) (h1 : x^2 + 4 * ( (x + 1) / (x - 3) )^2 = 50)
  (y := ( (x - 3)^2 * (x + 4) ) / (2 * x - 4)) :
  y = -32 / 7 ∨ y = 2 :=
sorry

end find_y_values_l218_218535


namespace legs_heads_difference_l218_218219

variables (D C L H : ℕ)

theorem legs_heads_difference
    (hC : C = 18)
    (hL : L = 2 * D + 4 * C)
    (hH : H = D + C) :
    L - 2 * H = 36 :=
by
  have h1 : C = 18 := hC
  have h2 : L = 2 * D + 4 * C := hL
  have h3 : H = D + C := hH
  sorry

end legs_heads_difference_l218_218219


namespace caleb_ice_cream_l218_218991

theorem caleb_ice_cream (x : ℕ) (hx1 : ∃ x, x ≥ 0) (hx2 : 4 * x - 36 = 4) : x = 10 :=
by {
  sorry
}

end caleb_ice_cream_l218_218991


namespace minimum_embrasure_length_l218_218656

theorem minimum_embrasure_length : ∀ (s : ℝ), 
  (∀ t : ℝ, (∃ k : ℤ, t = k / 2 ∧ k % 2 = 0) ∨ (∃ k : ℤ, t = (k + 1) / 2 ∧ k % 2 = 1)) → 
  (∃ z : ℝ, z = 2 / 3) := 
sorry

end minimum_embrasure_length_l218_218656


namespace golf_balls_count_l218_218444

theorem golf_balls_count (dozen_count : ℕ) (balls_per_dozen : ℕ) (total_balls : ℕ) 
  (h1 : dozen_count = 13) 
  (h2 : balls_per_dozen = 12) 
  (h3 : total_balls = dozen_count * balls_per_dozen) : 
  total_balls = 156 := 
sorry

end golf_balls_count_l218_218444


namespace number_of_free_ranging_chickens_l218_218652

-- Define the conditions as constants
def coop_chickens : ℕ := 14
def run_chickens : ℕ := 2 * coop_chickens
def barn_chickens : ℕ := coop_chickens / 2
def total_chickens_in_coop_and_run : ℕ := coop_chickens + run_chickens    
def free_ranging_chickens_condition : ℕ := 2 * run_chickens - 4
def ratio_condition : Prop := total_chickens_in_coop_and_run * 5 = 2 * (total_chickens_in_coop_and_run + free_ranging_chickens_condition)
def target_free_ranging_chickens : ℕ := 105

-- The proof statement
theorem number_of_free_ranging_chickens : 
  total_chickens_in_coop_and_run * 5 = 2 * (total_chickens_in_coop_and_run + target_free_ranging_chickens) →
  free_ranging_chickens_condition = target_free_ranging_chickens :=
by {
  sorry
}

end number_of_free_ranging_chickens_l218_218652


namespace train_relative_speed_l218_218016

-- Definitions of given conditions
def initialDistance : ℝ := 13
def speedTrainA : ℝ := 37
def speedTrainB : ℝ := 43

-- Definition of the relative speed
def relativeSpeed : ℝ := speedTrainB - speedTrainA

-- Theorem to prove the relative speed
theorem train_relative_speed
  (h1 : initialDistance = 13)
  (h2 : speedTrainA = 37)
  (h3 : speedTrainB = 43) :
  relativeSpeed = 6 := by
  -- Placeholder for the actual proof
  sorry

end train_relative_speed_l218_218016


namespace find_the_number_l218_218156

theorem find_the_number :
  ∃ X : ℝ, (66.2 = (6.620000000000001 / 100) * X) ∧ X = 1000 :=
by
  sorry

end find_the_number_l218_218156


namespace smallest_n_for_divisibility_l218_218978

theorem smallest_n_for_divisibility (n : ℕ) (h : 2 ∣ 3^(2*n) - 1) (k : ℕ) : n = 2^(2007) := by
  sorry

end smallest_n_for_divisibility_l218_218978


namespace probability_no_adjacent_birch_l218_218813

theorem probability_no_adjacent_birch (m n : ℕ):
  let maple_trees := 5
  let oak_trees := 4
  let birch_trees := 6
  let total_trees := maple_trees + oak_trees + birch_trees
  (∀ (prob : ℚ), prob = (2 : ℚ) / 45) → (m + n = 47) := by
  sorry

end probability_no_adjacent_birch_l218_218813


namespace number_of_students_in_third_grade_l218_218964

theorem number_of_students_in_third_grade
    (total_students : ℕ)
    (sample_size : ℕ)
    (students_first_grade : ℕ)
    (students_second_grade : ℕ)
    (sample_first_and_second : ℕ)
    (students_in_third_grade : ℕ)
    (h1 : total_students = 2000)
    (h2 : sample_size = 100)
    (h3 : sample_first_and_second = students_first_grade + students_second_grade)
    (h4 : students_first_grade = 30)
    (h5 : students_second_grade = 30)
    (h6 : sample_first_and_second = 60)
    (h7 : sample_size - sample_first_and_second = students_in_third_grade)
    (h8 : students_in_third_grade * total_students = 40 * total_students / 100) :
  students_in_third_grade = 800 :=
sorry

end number_of_students_in_third_grade_l218_218964


namespace megan_broke_3_eggs_l218_218943

variables (total_eggs B C P : ℕ)

theorem megan_broke_3_eggs (h1 : total_eggs = 24) (h2 : C = 2 * B) (h3 : P = 24 - (B + C)) (h4 : P - C = 9) : B = 3 := by
  sorry

end megan_broke_3_eggs_l218_218943


namespace problem1_problem2_l218_218407

-- Define a and b as real numbers
variables (a b : ℝ)

-- Problem 1: Prove (a-2b)^2 - (b-a)(a+b) = 2a^2 - 4ab + 3b^2
theorem problem1 : (a - 2 * b) ^ 2 - (b - a) * (a + b) = 2 * a ^ 2 - 4 * a * b + 3 * b ^ 2 :=
sorry

-- Problem 2: Prove (2a-b)^2 \cdot (2a+b)^2 = 16a^4 - 8a^2b^2 + b^4
theorem problem2 : (2 * a - b) ^ 2 * (2 * a + b) ^ 2 = 16 * a ^ 4 - 8 * a ^ 2 * b ^ 2 + b ^ 4 :=
sorry

end problem1_problem2_l218_218407


namespace emily_small_gardens_l218_218505

theorem emily_small_gardens (total_seeds : ℕ) (big_garden_seeds : ℕ) (seeds_per_small_garden : ℕ) (num_small_gardens : ℕ) :
  total_seeds = 41 →
  big_garden_seeds = 29 →
  seeds_per_small_garden = 4 →
  num_small_gardens = (total_seeds - big_garden_seeds) / seeds_per_small_garden →
  num_small_gardens = 3 :=
by
  intros h_total h_big h_seeds_per_small h_num_small
  rw [h_total, h_big, h_seeds_per_small] at h_num_small
  exact h_num_small

end emily_small_gardens_l218_218505


namespace range_of_2a_plus_3b_inequality_between_expressions_l218_218797

-- First proof problem
theorem range_of_2a_plus_3b (a b : ℝ) (h1 : -1 ≤ a + b) (h2 : a + b ≤ 1) (h3 : -1 ≤ a - b) (h4 : a - b ≤ 1) :
  -3 ≤ 2 * a + 3 * b ∧ 2 * a + 3 * b ≤ 3 :=
sorry

-- Second proof problem
theorem inequality_between_expressions (a b c : ℝ) (h : a^2 + b^2 + c^2 = 6) :
  (1 / (a^2 + 1) + 1 / (b^2 + 2)) > (1 / 2 - 1 / (c^2 + 3)) :=
sorry

end range_of_2a_plus_3b_inequality_between_expressions_l218_218797


namespace manager_salary_l218_218126

theorem manager_salary 
    (avg_salary_18 : ℕ)
    (new_avg_salary : ℕ)
    (num_employees : ℕ)
    (num_employees_with_manager : ℕ)
    (old_total_salary : ℕ := num_employees * avg_salary_18)
    (new_total_salary : ℕ := num_employees_with_manager * new_avg_salary) :
    (new_avg_salary = avg_salary_18 + 200) →
    (old_total_salary = 18 * 2000) →
    (new_total_salary = 19 * (2000 + 200)) →
    new_total_salary - old_total_salary = 5800 :=
by
  intros h1 h2 h3
  sorry

end manager_salary_l218_218126


namespace sam_digits_memorized_l218_218053

-- Definitions
def carlos_memorized (c : ℕ) := (c * 6 = 24)
def sam_memorized (s c : ℕ) := (s = c + 6)
def mina_memorized := 24

-- Theorem
theorem sam_digits_memorized (s c : ℕ) (h_c : carlos_memorized c) (h_s : sam_memorized s c) : s = 10 :=
by {
  sorry
}

end sam_digits_memorized_l218_218053


namespace length_of_MN_eq_5_sqrt_10_div_3_l218_218717

theorem length_of_MN_eq_5_sqrt_10_div_3 
  (A : ℝ × ℝ) 
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)
  (D : ℝ × ℝ)
  (M : ℝ × ℝ)
  (N : ℝ × ℝ)
  (hyp_A : A = (1, 3))
  (hyp_B : B = (25 / 3, 5 / 3))
  (hyp_C : C = (22 / 3, 14 / 3))
  (hyp_eq_edges : (dist (0, 0) M = dist M N) ∧ (dist M N = dist N B))
  (hyp_D : D = (5 / 2, 15 / 2))
  (hyp_M : M = (5 / 3, 5)) :
  dist M N = 5 * Real.sqrt 10 / 3 :=
sorry

end length_of_MN_eq_5_sqrt_10_div_3_l218_218717


namespace sin_theta_plus_2pi_div_3_cos_theta_minus_5pi_div_6_l218_218141

variable (θ : ℝ)

theorem sin_theta_plus_2pi_div_3 (h : Real.sin (θ - Real.pi / 3) = 1 / 3) :
  Real.sin (θ + 2 * Real.pi / 3) = -1 / 3 :=
  sorry

theorem cos_theta_minus_5pi_div_6 (h : Real.sin (θ - Real.pi / 3) = 1 / 3) :
  Real.cos (θ - 5 * Real.pi / 6) = 1 / 3 :=
  sorry

end sin_theta_plus_2pi_div_3_cos_theta_minus_5pi_div_6_l218_218141


namespace time_to_fill_pool_l218_218930

-- Define constants based on the conditions
def pool_capacity : ℕ := 30000
def hose_count : ℕ := 5
def flow_rate_per_hose : ℕ := 25 / 10  -- 2.5 gallons per minute
def conversion_minutes_to_hours : ℕ := 60

-- Define the total flow rate per minute
def total_flow_rate_per_minute : ℕ := hose_count * flow_rate_per_hose

-- Define the total flow rate per hour
def total_flow_rate_per_hour : ℕ := total_flow_rate_per_minute * conversion_minutes_to_hours

-- Theorem stating the number of hours required to fill the pool
theorem time_to_fill_pool : pool_capacity / total_flow_rate_per_hour = 40 := by
  sorry -- Proof will be provided here

end time_to_fill_pool_l218_218930


namespace theta_solutions_count_l218_218965

theorem theta_solutions_count :
  (∃ (count : ℕ), count = 4 ∧ ∀ θ, 0 < θ ∧ θ ≤ 2 * Real.pi ∧ 1 - 4 * Real.sin θ + 5 * Real.cos (2 * θ) = 0 ↔ count = 4) :=
sorry

end theta_solutions_count_l218_218965


namespace find_q_zero_l218_218418

theorem find_q_zero
  (p q r : ℝ → ℝ)  -- Define p, q, r as functions from ℝ to ℝ (since they are polynomials)
  (h1 : ∀ x, r x = p x * q x + 2)  -- Condition 1: r(x) = p(x) * q(x) + 2
  (h2 : p 0 = 6)                   -- Condition 2: constant term of p(x) is 6
  (h3 : r 0 = 5)                   -- Condition 3: constant term of r(x) is 5
  : q 0 = 1 / 2 :=                 -- Conclusion: q(0) = 1/2
sorry

end find_q_zero_l218_218418


namespace simplify_sqrt_l218_218506

noncomputable def simplify_expression : ℝ :=
  Real.sqrt (8 + 6 * Real.sqrt 2) + Real.sqrt (8 - 6 * Real.sqrt 2)

theorem simplify_sqrt (h : simplify_expression = 2 * Real.sqrt 6) : 
    Real.sqrt (8 + 6 * Real.sqrt 2) + Real.sqrt (8 - 6 * Real.sqrt 2) = 2 * Real.sqrt 6 :=
  by sorry

end simplify_sqrt_l218_218506


namespace right_triangle_m_c_l218_218573

theorem right_triangle_m_c (a b c : ℝ) (m_c : ℝ) 
  (h : (1 / a) + (1 / b) = 3 / c) : 
  m_c = (c * (1 + Real.sqrt 10)) / 9 :=
sorry

end right_triangle_m_c_l218_218573


namespace find_a4_l218_218299

-- Define the arithmetic sequence and the sum of the first N terms
def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

-- Sum of the first N terms in an arithmetic sequence
def sum_arithmetic_seq (a d N : ℕ) : ℕ := N * (2 * a + (N - 1) * d) / 2

-- Define the conditions
def condition1 (a d : ℕ) : Prop := a + (a + 2 * d) + (a + 4 * d) = 15
def condition2 (a d : ℕ) : Prop := sum_arithmetic_seq a d 4 = 16

-- Lean 4 statement to prove the value of a_4
theorem find_a4 (a d : ℕ) (h1 : condition1 a d) (h2 : condition2 a d) : arithmetic_seq a d 4 = 7 :=
sorry

end find_a4_l218_218299


namespace no_outliers_in_dataset_l218_218530

theorem no_outliers_in_dataset :
  let D := [7, 20, 34, 34, 40, 42, 42, 44, 52, 58]
  let Q1 := 34
  let Q3 := 44
  let IQR := Q3 - Q1
  let lower_threshold := Q1 - 1.5 * IQR
  let upper_threshold := Q3 + 1.5 * IQR
  (∀ x ∈ D, x ≥ lower_threshold) ∧ (∀ x ∈ D, x ≤ upper_threshold) →
  ∀ x ∈ D, ¬(x < lower_threshold ∨ x > upper_threshold) :=
by 
  sorry

end no_outliers_in_dataset_l218_218530


namespace fritz_has_40_dollars_l218_218743

variable (F S R : ℝ)
variable (h1 : S = (1 / 2) * F + 4)
variable (h2 : R = 3 * S)
variable (h3 : R + S = 96)

theorem fritz_has_40_dollars : F = 40 :=
by
  sorry

end fritz_has_40_dollars_l218_218743


namespace total_money_l218_218542

-- Define the variables A, B, and C as real numbers.
variables (A B C : ℝ)

-- Define the conditions as hypotheses.
def conditions : Prop :=
  A + C = 300 ∧ B + C = 150 ∧ C = 50

-- State the theorem to prove the total amount of money A, B, and C have.
theorem total_money (h : conditions A B C) : A + B + C = 400 :=
by {
  -- This proof is currently omitted.
  sorry
}

end total_money_l218_218542


namespace g_one_third_value_l218_218073

noncomputable def g : ℚ → ℚ := sorry

theorem g_one_third_value : (∀ (x : ℚ), x ≠ 0 → (4 * g (1 / x) + 3 * g x / x^2 = x^3)) → g (1 / 3) = 21 / 44 := by
  intro h
  sorry

end g_one_third_value_l218_218073


namespace minutes_to_seconds_l218_218215

theorem minutes_to_seconds (m : ℝ) (hm : m = 6.5) : m * 60 = 390 := by
  sorry

end minutes_to_seconds_l218_218215


namespace value_of_m_l218_218787

theorem value_of_m : (∀ x : ℝ, (1 + 2 * x) ^ 3 = 1 + 6 * x + m * x ^ 2 + 8 * x ^ 3 → m = 12) := 
by {
  -- This is where the proof would go
  sorry
}

end value_of_m_l218_218787


namespace connie_marbles_l218_218297

theorem connie_marbles (j c : ℕ) (h1 : j = 498) (h2 : j = c + 175) : c = 323 :=
by
  -- Placeholder for the proof
  sorry

end connie_marbles_l218_218297


namespace find_n_tan_l218_218387

theorem find_n_tan (n : ℤ) (hn : -90 < n ∧ n < 90) (htan : Real.tan (n * Real.pi / 180) = Real.tan (312 * Real.pi / 180)) : 
  n = -48 := 
sorry

end find_n_tan_l218_218387


namespace geometric_sequence_min_value_l218_218000

theorem geometric_sequence_min_value
  (a : ℕ → ℝ)
  (h1 : ∀ n, 0 < a n) 
  (h2 : a 9 = 9 * a 7)
  (exists_m_n : ∃ m n, a m * a n = 9 * (a 1)^2):
  ∀ m n, (m + n = 4) → (1 / m + 9 / n) ≥ 4 :=
by
  intros m n h
  sorry

end geometric_sequence_min_value_l218_218000


namespace eric_running_time_l218_218231

-- Define the conditions
variables (jog_time to_park_time return_time : ℕ)
axiom jog_time_def : jog_time = 10
axiom return_time_def : return_time = 90
axiom trip_relation : return_time = 3 * to_park_time

-- Define the question
def run_time : ℕ := to_park_time - jog_time

-- State the problem: Prove that given the conditions, the running time is 20 minutes.
theorem eric_running_time : run_time = 20 :=
by
  -- Proof goes here
  sorry

end eric_running_time_l218_218231


namespace largest_of_seven_consecutive_integers_l218_218403

theorem largest_of_seven_consecutive_integers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 2821) : 
  n + 6 = 406 := 
by
  -- Proof steps can be added here
  sorry

end largest_of_seven_consecutive_integers_l218_218403


namespace simplify_expression_l218_218346

-- Define the given expressions
def numerator : ℕ := 5^5 + 5^3 + 5
def denominator : ℕ := 5^4 - 2 * 5^2 + 5

-- Define the simplified fraction
def simplified_fraction : ℚ := numerator / denominator

-- Prove that the simplified fraction is equivalent to 651 / 116
theorem simplify_expression : simplified_fraction = 651 / 116 := by
  sorry

end simplify_expression_l218_218346


namespace total_infections_second_wave_l218_218900

theorem total_infections_second_wave (cases_per_day_first_wave : ℕ)
                                     (factor_increase : ℕ)
                                     (duration_weeks : ℕ)
                                     (days_per_week : ℕ) :
                                     cases_per_day_first_wave = 300 →
                                     factor_increase = 4 →
                                     duration_weeks = 2 →
                                     days_per_week = 7 →
                                     (duration_weeks * days_per_week) * (cases_per_day_first_wave + factor_increase * cases_per_day_first_wave) = 21000 :=
by 
  intros h1 h2 h3 h4
  sorry

end total_infections_second_wave_l218_218900


namespace calc_expression_l218_218838

theorem calc_expression : (113^2 - 104^2) / 9 = 217 := by
  sorry

end calc_expression_l218_218838


namespace min_bounces_l218_218400

theorem min_bounces
  (h₀ : ℝ := 160)  -- initial height
  (r : ℝ := 3/4)  -- bounce ratio
  (final_h : ℝ := 20)  -- desired height
  (b : ℕ)  -- number of bounces
  : ∃ b, (h₀ * (r ^ b) < final_h ∧ ∀ b', b' < b → ¬(h₀ * (r ^ b') < final_h)) :=
sorry

end min_bounces_l218_218400


namespace initial_amount_is_825_l218_218097

theorem initial_amount_is_825 (P R : ℝ) 
    (h1 : 956 = P * (1 + 3 * R / 100))
    (h2 : 1055 = P * (1 + 3 * (R + 4) / 100)) : 
    P = 825 := 
by 
  sorry

end initial_amount_is_825_l218_218097


namespace gold_copper_alloy_ratio_l218_218764

theorem gold_copper_alloy_ratio {G C A : ℝ} (hC : C = 9) (hA : A = 18) (hG : 9 < G ∧ G < 18) :
  ∃ x : ℝ, 18 = x * G + (1 - x) * 9 :=
by
  sorry

end gold_copper_alloy_ratio_l218_218764


namespace hyperbola_equation_l218_218459

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (asymptote_eq : ∀ x y, 3*x + 4*y = 0 → y = (-3/4) * x)
  (focus_eq : (0, 5) = (0, 5)) :
  ∃ a b : ℝ, a = 3 ∧ b = 4 ∧ (∀ y x, (y^2 / 9 - x^2 / 16 = 1)) :=
sorry

end hyperbola_equation_l218_218459


namespace puppies_brought_in_l218_218353

open Nat

theorem puppies_brought_in (orig_puppies adopt_rate days total_adopted brought_in_puppies : ℕ) 
  (h_orig : orig_puppies = 3)
  (h_adopt_rate : adopt_rate = 3)
  (h_days : days = 2)
  (h_total_adopted : total_adopted = adopt_rate * days)
  (h_equation : total_adopted = orig_puppies + brought_in_puppies) :
  brought_in_puppies = 3 :=
by
  sorry

end puppies_brought_in_l218_218353


namespace distance_between_stations_l218_218341

theorem distance_between_stations
  (time_start_train1 time_meet time_start_train2 : ℕ) -- time in hours (7 a.m., 11 a.m., 8 a.m.)
  (speed_train1 speed_train2 : ℕ) -- speed in kmph (20 kmph, 25 kmph)
  (distance_covered_train1 distance_covered_train2 : ℕ)
  (total_distance : ℕ) :
  time_start_train1 = 7 ∧ time_meet = 11 ∧ time_start_train2 = 8 ∧ speed_train1 = 20 ∧ speed_train2 = 25 ∧
  distance_covered_train1 = (time_meet - time_start_train1) * speed_train1 ∧
  distance_covered_train2 = (time_meet - time_start_train2) * speed_train2 ∧
  total_distance = distance_covered_train1 + distance_covered_train2 →
  total_distance = 155 := by
{
  sorry
}

end distance_between_stations_l218_218341


namespace smaller_angle_at_3_20_correct_l218_218298

noncomputable def smaller_angle_at_3_20 (angle_3_00 : ℝ)
  (minute_hand_rate : ℝ)
  (hour_hand_rate : ℝ) : ℝ :=
  let angle_change := (3.20 - 3.00) * (minute_hand_rate - hour_hand_rate)
  let total_angle := angle_3_00 + angle_change
  let smaller_angle := if total_angle <= 180 then total_angle else 360 - total_angle
  smaller_angle

theorem smaller_angle_at_3_20_correct :
  smaller_angle_at_3_20 90 6 0.5 = 160.0 :=
by
  sorry

end smaller_angle_at_3_20_correct_l218_218298


namespace opposite_sides_of_line_l218_218673

theorem opposite_sides_of_line 
  (x₀ y₀ : ℝ) 
  (h : (3 * x₀ + 2 * y₀ - 8) * (3 * 1 + 2 * 2 - 8) < 0) :
  3 * x₀ + 2 * y₀ > 8 :=
by
  sorry

end opposite_sides_of_line_l218_218673


namespace quadratic_transformed_correct_l218_218087

noncomputable def quadratic_transformed (a b c : ℝ) (r s : ℝ) (h1 : a ≠ 0) 
  (h_roots : r + s = -b / a ∧ r * s = c / a) : Polynomial ℝ :=
Polynomial.C (a * b * c) + Polynomial.C ((-(a + b) * b)) * Polynomial.X + Polynomial.X^2

-- The theorem statement
theorem quadratic_transformed_correct (a b c r s : ℝ) (h1 : a ≠ 0)
  (h_roots : r + s = -b / a ∧ r * s = c / a) :
  (quadratic_transformed a b c r s h1 h_roots).roots = {a * (r + b), a * (s + b)} :=
sorry

end quadratic_transformed_correct_l218_218087


namespace ratio_pen_pencil_l218_218296

theorem ratio_pen_pencil (P : ℝ) (pencil_cost total_cost : ℝ) 
  (hc1 : pencil_cost = 8) 
  (hc2 : total_cost = 12)
  (hc3 : P + pencil_cost = total_cost) : 
  P / pencil_cost = 1 / 2 :=
by 
  sorry

end ratio_pen_pencil_l218_218296


namespace rem_sum_a_b_c_l218_218855

theorem rem_sum_a_b_c (a b c : ℤ) (h1 : a * b * c ≡ 1 [ZMOD 5]) (h2 : 3 * c ≡ 1 [ZMOD 5]) (h3 : 4 * b ≡ 1 + b [ZMOD 5]) : 
  (a + b + c) % 5 = 3 := by 
  sorry

end rem_sum_a_b_c_l218_218855


namespace parabola_unique_solution_l218_218844

theorem parabola_unique_solution (b c : ℝ) :
  (∀ x y : ℝ, (x, y) = (-2, -8) ∨ (x, y) = (4, 28) ∨ (x, y) = (1, 4) →
    (y = x^2 + b * x + c)) →
  b = 4 ∧ c = -1 :=
by
  intro h
  have h₁ := h (-2) (-8) (Or.inl rfl)
  have h₂ := h 4 28 (Or.inr (Or.inl rfl))
  have h₃ := h 1 4 (Or.inr (Or.inr rfl))
  sorry

end parabola_unique_solution_l218_218844


namespace whole_number_N_l218_218880

theorem whole_number_N (N : ℤ) : (9 < N / 4 ∧ N / 4 < 10) ↔ (N = 37 ∨ N = 38 ∨ N = 39) := 
by sorry

end whole_number_N_l218_218880


namespace coeff_x3_in_expansion_l218_218958

noncomputable def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

theorem coeff_x3_in_expansion :
  (2 : ℚ)^(4 - 2) * binomial_coeff 4 2 = 24 := by 
  sorry

end coeff_x3_in_expansion_l218_218958


namespace regular_hours_l218_218321

variable (R : ℕ)

theorem regular_hours (h1 : 5 * R + 6 * (44 - R) + 5 * R + 6 * (48 - R) = 472) : R = 40 :=
by
  sorry

end regular_hours_l218_218321


namespace percentage_charge_l218_218725

def car_cost : ℝ := 14600
def initial_savings : ℝ := 14500
def trip_charge : ℝ := 1.5
def number_of_trips : ℕ := 40
def grocery_value : ℝ := 800
def final_savings_needed : ℝ := car_cost - initial_savings

-- The amount earned from trips
def amount_from_trips : ℝ := number_of_trips * trip_charge

-- The amount needed from percentage charge on groceries
def amount_from_percentage (P: ℝ) : ℝ := grocery_value * P

-- The required amount from percentage charge on groceries
def required_amount_from_percentage : ℝ := final_savings_needed - amount_from_trips

theorem percentage_charge (P: ℝ) (h: amount_from_percentage P = required_amount_from_percentage) : P = 0.05 :=
by 
  -- Proof follows from the given condition that amount_from_percentage P = required_amount_from_percentage
  sorry

end percentage_charge_l218_218725


namespace solve_quadratic_and_linear_equations_l218_218091

theorem solve_quadratic_and_linear_equations :
  (∀ x : ℝ, x^2 - 4*x - 1 = 0 → x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5) ∧
  (∀ x : ℝ, (x + 3) * (x - 3) = 3 * (x + 3) → x = -3 ∨ x = 6) :=
by
  sorry

end solve_quadratic_and_linear_equations_l218_218091


namespace remainder_of_86_l218_218971

theorem remainder_of_86 {m : ℕ} (h1 : m ≠ 1) 
  (h2 : 69 % m = 90 % m) (h3 : 90 % m = 125 % m) : 86 % m = 2 := 
by
  sorry

end remainder_of_86_l218_218971


namespace product_of_3_point_6_and_0_point_25_l218_218519

theorem product_of_3_point_6_and_0_point_25 : 3.6 * 0.25 = 0.9 := 
by 
  sorry

end product_of_3_point_6_and_0_point_25_l218_218519


namespace ellipse_condition_l218_218895

theorem ellipse_condition (m : ℝ) : 
  (∃ x y : ℝ, m * (x^2 + y^2 + 2*y + 1) = (x - 2*y + 3)^2) → m > 5 :=
by
  intro h
  sorry

end ellipse_condition_l218_218895


namespace flight_duration_l218_218538

noncomputable def departure_time_pst := 9 * 60 + 15 -- in minutes
noncomputable def arrival_time_est := 17 * 60 + 40 -- in minutes
noncomputable def time_difference := 3 * 60 -- in minutes

theorem flight_duration (h m : ℕ) 
  (h_cond : 0 < m ∧ m < 60) 
  (total_flight_time : (arrival_time_est - (departure_time_pst + time_difference)) = h * 60 + m) : 
  h + m = 30 :=
sorry

end flight_duration_l218_218538


namespace even_digit_perfect_squares_odd_digit_perfect_squares_l218_218280

-- Define the property of being a four-digit number
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

-- Define the property of having even digits
def is_even_digit_number (n : ℕ) : Prop :=
  ∀ digit ∈ (n.digits 10), digit % 2 = 0

-- Define the property of having odd digits
def is_odd_digit_number (n : ℕ) : Prop :=
  ∀ digit ∈ (n.digits 10), digit % 2 = 1

-- Part (a) statement
theorem even_digit_perfect_squares :
  ∀ n : ℕ, is_four_digit n ∧ is_even_digit_number n ∧ ∃ m : ℕ, n = m * m ↔ 
    n = 4624 ∨ n = 6084 ∨ n = 6400 ∨ n = 8464 :=
sorry

-- Part (b) statement
theorem odd_digit_perfect_squares :
  ∀ n : ℕ, is_four_digit n ∧ is_odd_digit_number n ∧ ∃ m : ℕ, n = m * m → false :=
sorry

end even_digit_perfect_squares_odd_digit_perfect_squares_l218_218280


namespace sum_of_products_l218_218140

theorem sum_of_products (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 62)
  (h2 : a + b + c = 18) : 
  a * b + b * c + c * a = 131 :=
sorry

end sum_of_products_l218_218140


namespace work_problem_l218_218360

theorem work_problem (days_B : ℝ) (h : (1 / 20) + (1 / days_B) = 1 / 8.571428571428571) : days_B = 15 :=
sorry

end work_problem_l218_218360


namespace largest_square_not_divisible_by_100_l218_218582

theorem largest_square_not_divisible_by_100
  (n : ℕ) (h1 : ∃ a : ℕ, a^2 = n) 
  (h2 : n % 100 ≠ 0)
  (h3 : ∃ m : ℕ, m * 100 + n % 100 = n ∧ ∃ b : ℕ, b^2 = m) :
  n = 1681 := sorry

end largest_square_not_divisible_by_100_l218_218582


namespace not_true_diamond_self_zero_l218_218902

-- Define the operator ⋄
def diamond (x y : ℝ) := |x - 2*y|

-- The problem statement in Lean4
theorem not_true_diamond_self_zero : ¬ (∀ x : ℝ, diamond x x = 0) := by
  sorry

end not_true_diamond_self_zero_l218_218902


namespace unit_prices_max_books_l218_218556

-- Definitions based on conditions 1 and 2
def unit_price_A (x : ℝ) : Prop :=
  x > 5 ∧ (1200 / x = 900 / (x - 5))

-- Definitions based on conditions 3, 4, and 5
def max_books_A (y : ℝ) : Prop :=
  0 ≤ y ∧ y ≤ 300 ∧ 0.9 * 20 * y + 15 * (300 - y) ≤ 5100

theorem unit_prices
  (x : ℝ)
  (h : unit_price_A x) :
  x = 20 ∧ x - 5 = 15 :=
sorry

theorem max_books
  (y : ℝ)
  (hy : max_books_A y) :
  y ≤ 200 :=
sorry

end unit_prices_max_books_l218_218556


namespace cubic_expression_equals_two_l218_218023

theorem cubic_expression_equals_two (x : ℝ) (h : 2 * x ^ 2 - 3 * x - 2022 = 0) :
  2 * x ^ 3 - x ^ 2 - 2025 * x - 2020 = 2 :=
sorry

end cubic_expression_equals_two_l218_218023


namespace combined_hits_and_misses_total_l218_218584

/-
  Prove that given the conditions for each day regarding the number of misses and
  the ratio of misses to hits, the combined total of hits and misses for the 
  three days is 322.
-/

theorem combined_hits_and_misses_total :
  (∀ (H1 : ℕ) (H2 : ℕ) (H3 : ℕ), 
    (2 * H1 = 60) ∧ (3 * H2 = 84) ∧ (5 * H3 = 100) →
    60 + 84 + 100 + H1 + H2 + H3 = 322) :=
by
  sorry

end combined_hits_and_misses_total_l218_218584


namespace circle_common_chord_l218_218288

theorem circle_common_chord (x y : ℝ) :
  (x^2 + y^2 - 4 * x + 6 * y = 0) ∧
  (x^2 + y^2 - 6 * x = 0) →
  (x + 3 * y = 0) :=
by
  sorry

end circle_common_chord_l218_218288


namespace cost_of_filling_all_pots_l218_218801

def cost_palm_fern : ℝ := 15.00
def cost_creeping_jenny_per_plant : ℝ := 4.00
def num_creeping_jennies : ℝ := 4
def cost_geranium_per_plant : ℝ := 3.50
def num_geraniums : ℝ := 4
def cost_elephant_ear_per_plant : ℝ := 7.00
def num_elephant_ears : ℝ := 2
def cost_purple_fountain_grass_per_plant : ℝ := 6.00
def num_purple_fountain_grasses : ℝ := 3
def num_pots : ℝ := 4

def total_cost_per_pot : ℝ := 
  cost_palm_fern +
  (num_creeping_jennies * cost_creeping_jenny_per_plant) +
  (num_geraniums * cost_geranium_per_plant) +
  (num_elephant_ears * cost_elephant_ear_per_plant) +
  (num_purple_fountain_grasses * cost_purple_fountain_grass_per_plant)

def total_cost : ℝ := total_cost_per_pot * num_pots

theorem cost_of_filling_all_pots : total_cost = 308.00 := by
  sorry

end cost_of_filling_all_pots_l218_218801


namespace num_factors_2012_l218_218608

theorem num_factors_2012 : (Nat.factors 2012).length = 6 := by
  sorry

end num_factors_2012_l218_218608


namespace twentieth_term_arithmetic_sequence_eq_neg49_l218_218170

-- Definitions based on the conditions
def a1 : ℤ := 8
def d : ℤ := 5 - 8
def a (n : ℕ) : ℤ := a1 + (n - 1) * d

-- The proof statement
theorem twentieth_term_arithmetic_sequence_eq_neg49 : a 20 = -49 :=
by 
  -- Proof will be inserted here
  sorry

end twentieth_term_arithmetic_sequence_eq_neg49_l218_218170


namespace average_cups_of_tea_sold_l218_218602

theorem average_cups_of_tea_sold (x_avg : ℝ) (y_regression : ℝ → ℝ) 
  (h1 : x_avg = 12) (h2 : ∀ x, y_regression x = -2*x + 58) : 
  y_regression x_avg = 34 := by
  sorry

end average_cups_of_tea_sold_l218_218602


namespace samantha_total_cost_l218_218473

noncomputable def daily_rental_rate : ℝ := 30
noncomputable def daily_rental_days : ℝ := 3
noncomputable def cost_per_mile : ℝ := 0.15
noncomputable def miles_driven : ℝ := 500

theorem samantha_total_cost :
  (daily_rental_rate * daily_rental_days) + (cost_per_mile * miles_driven) = 165 :=
by
  sorry

end samantha_total_cost_l218_218473


namespace rachel_baked_brownies_l218_218795

theorem rachel_baked_brownies (b : ℕ) (h : 3 * b / 5 = 18) : b = 30 :=
by
  sorry

end rachel_baked_brownies_l218_218795


namespace common_denominator_first_set_common_denominator_second_set_l218_218070

theorem common_denominator_first_set (x y : ℕ) (h₁ : y ≠ 0) : Nat.lcm (3 * y) (2 * y^2) = 6 * y^2 :=
by sorry

theorem common_denominator_second_set (a b c : ℕ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) : Nat.lcm (a^2 * b) (3 * a * b^2) = 3 * a^2 * b^2 :=
by sorry

end common_denominator_first_set_common_denominator_second_set_l218_218070


namespace value_of_a_l218_218431

theorem value_of_a (a x : ℝ) (h : (3 * x^2 + 2 * a * x = 0) → (x^3 + a * x^2 - (4 / 3) * a = 0)) :
  a = 0 ∨ a = 3 ∨ a = -3 :=
by
  sorry

end value_of_a_l218_218431


namespace highest_probability_ksi_expected_value_ksi_equals_l218_218399

noncomputable def probability_ksi_equals (k : ℕ) : ℚ :=
  match k with
  | 2 => 9 / 64
  | 3 => 18 / 64
  | 4 => 21 / 64
  | 5 => 12 / 64
  | 6 => 4 / 64
  | _ => 0

noncomputable def expected_value_ksi : ℚ :=
  2 * (9 / 64) + 3 * (18 / 64) + 4 * (21 / 64) + 5 * (12 / 64) + 6 * (4 / 64)

theorem highest_probability_ksi :
  ∃ k : ℕ, (∀ m : ℕ, probability_ksi_equals k ≥ probability_ksi_equals m) ∧ k = 4 :=
by
  sorry

theorem expected_value_ksi_equals :
  expected_value_ksi = 15 / 4 :=
by
  sorry

end highest_probability_ksi_expected_value_ksi_equals_l218_218399


namespace uniformity_of_scores_l218_218410

/- Problem statement:
  Randomly select 10 students from class A and class B to participate in an English oral test. 
  The variances of their test scores are S1^2 = 13.2 and S2^2 = 26.26, respectively. 
  Then, we show that the scores of the 10 students from class A are more uniform than 
  those of the 10 students from class B.
-/

theorem uniformity_of_scores (S1 S2 : ℝ) (h1 : S1^2 = 13.2) (h2 : S2^2 = 26.26) : 
    13.2 < 26.26 := 
by 
  sorry

end uniformity_of_scores_l218_218410


namespace fraction_comparison_l218_218476

theorem fraction_comparison : (5555553 / 5555557 : ℚ) > (6666664 / 6666669 : ℚ) :=
  sorry

end fraction_comparison_l218_218476


namespace angle_B_magnitude_value_of_b_l218_218869
open Real

theorem angle_B_magnitude (B : ℝ) (h : 2 * sin B - 2 * sin B ^ 2 - cos (2 * B) = sqrt 3 - 1) :
  B = π / 3 ∨ B = 2 * π / 3 := sorry

theorem value_of_b (a B S : ℝ) (hB : B = π / 3) (ha : a = 6) (hS : S = 6 * sqrt 3) :
  let c := 4
  let b := 2 * sqrt 7
  let half_angle_B := 1 / 2 * a * c * sin B
  half_angle_B = S :=
by
  sorry

end angle_B_magnitude_value_of_b_l218_218869


namespace value_of_a_l218_218735

theorem value_of_a {a : ℝ} (A : Set ℝ) (B : Set ℝ) (hA : A = {-1, 0, 2}) (hB : B = {2^a}) (hSub : B ⊆ A) : a = 1 := 
sorry

end value_of_a_l218_218735


namespace range_of_a_l218_218851

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x = 3 ∧ 3 * x - (a * x + 1) / 2 < 4 * x / 3) → a > 3 :=
by
  intro h
  obtain ⟨x, hx1, hx2⟩ := h
  sorry

end range_of_a_l218_218851


namespace largest_n_l218_218147

theorem largest_n (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) : 
  ∃ n : ℕ, n > 0 ∧ n = 10 ∧ n^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 5 * x + 5 * y + 5 * z - 12 := 
sorry

end largest_n_l218_218147


namespace num_five_digit_integers_l218_218685

theorem num_five_digit_integers
  (total_digits : ℕ := 8)
  (repeat_3 : ℕ := 2)
  (repeat_6 : ℕ := 3)
  (repeat_8 : ℕ := 2)
  (arrangements : ℕ := Nat.factorial total_digits / (Nat.factorial repeat_3 * Nat.factorial repeat_6 * Nat.factorial repeat_8)) :
  arrangements = 1680 := by
  sorry

end num_five_digit_integers_l218_218685


namespace even_number_less_than_its_square_l218_218988

theorem even_number_less_than_its_square (m : ℕ) (h1 : 2 ∣ m) (h2 : m > 1) : m < m^2 :=
by
sorry

end even_number_less_than_its_square_l218_218988


namespace proof_complement_union_l218_218157

-- Definition of the universal set U
def U : Finset ℕ := {0, 1, 2, 3, 4}

-- Definition of the subset A
def A : Finset ℕ := {0, 3, 4}

-- Definition of the subset B
def B : Finset ℕ := {1, 3}

-- Definition of the complement of A in U
def complement_A : Finset ℕ := U \ A

-- Definition of the union of the complement of A and B
def union_complement_A_B : Finset ℕ := complement_A ∪ B

-- Statement of the theorem
theorem proof_complement_union :
  union_complement_A_B = {1, 2, 3} :=
sorry

end proof_complement_union_l218_218157


namespace inequality_transitivity_l218_218539

theorem inequality_transitivity (a b c : ℝ) (h : a > b) : 
  a + c > b + c :=
sorry

end inequality_transitivity_l218_218539


namespace max_value_of_expression_l218_218552

open Real

theorem max_value_of_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 3) :
  a + sqrt (a * b) + (a * b * c) ^ (1 / 4) ≤ 10 / 3 := sorry

end max_value_of_expression_l218_218552


namespace square_of_negative_is_positive_l218_218713

-- Define P as a negative integer
variable (P : ℤ) (hP : P < 0)

-- Theorem statement that P² is always positive.
theorem square_of_negative_is_positive : P^2 > 0 :=
sorry

end square_of_negative_is_positive_l218_218713


namespace min_abs_sum_l218_218866

theorem min_abs_sum (a b c : ℝ) (h₁ : a + b + c = -2) (h₂ : a * b * c = -4) :
  ∃ (m : ℝ), m = min (abs a + abs b + abs c) 6 :=
sorry

end min_abs_sum_l218_218866


namespace ratio_of_books_l218_218921

theorem ratio_of_books (longest_pages : ℕ) (middle_pages : ℕ) (shortest_pages : ℕ) :
  longest_pages = 396 ∧ middle_pages = 297 ∧ shortest_pages = longest_pages / 4 →
  (middle_pages / shortest_pages = 3) :=
by
  intros h
  obtain ⟨h_longest, h_middle, h_shortest⟩ := h
  sorry

end ratio_of_books_l218_218921


namespace largest_two_digit_number_divisible_by_6_and_ends_in_4_l218_218292

theorem largest_two_digit_number_divisible_by_6_and_ends_in_4 :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ n % 6 = 0 ∧ n % 10 = 4 ∧ n = 84 :=
by
  sorry

end largest_two_digit_number_divisible_by_6_and_ends_in_4_l218_218292


namespace cab_to_bus_ratio_l218_218570

noncomputable def train_distance : ℤ := 300
noncomputable def bus_distance : ℤ := train_distance / 2
noncomputable def total_distance : ℤ := 500
noncomputable def cab_distance : ℤ := total_distance - (train_distance + bus_distance)
noncomputable def ratio : ℚ := cab_distance / bus_distance

theorem cab_to_bus_ratio :
  ratio = 1 / 3 := by
  sorry

end cab_to_bus_ratio_l218_218570


namespace unique_shell_arrangements_l218_218966

theorem unique_shell_arrangements : 
  let shells := 12
  let symmetry_ops := 12
  let total_arrangements := Nat.factorial shells
  let distinct_arrangements := total_arrangements / symmetry_ops
  distinct_arrangements = 39916800 := by
  sorry

end unique_shell_arrangements_l218_218966


namespace isosceles_triangle_angles_sum_l218_218012

theorem isosceles_triangle_angles_sum (x : ℝ) 
  (h_triangle_sum : ∀ a b c : ℝ, a + b + c = 180)
  (h_isosceles : ∃ a b : ℝ, (a = 50 ∧ b = x) ∨ (a = x ∧ b = 50)) :
  50 + x + (180 - 50 * 2) + 65 + 80 = 195 :=
by
  sorry

end isosceles_triangle_angles_sum_l218_218012


namespace pyramid_base_side_length_l218_218253

theorem pyramid_base_side_length
  (area_lateral_face : ℝ)
  (slant_height : ℝ)
  (s : ℝ)
  (h_area_lateral_face : area_lateral_face = 144)
  (h_slant_height : slant_height = 24) :
  (1 / 2) * s * slant_height = area_lateral_face → s = 12 :=
by
  sorry

end pyramid_base_side_length_l218_218253


namespace tablecloth_diameter_l218_218428

theorem tablecloth_diameter (r : ℝ) (h : r = 5) : 2 * r = 10 :=
by
  simp [h]
  sorry

end tablecloth_diameter_l218_218428


namespace sin_transform_l218_218055

theorem sin_transform (θ : ℝ) (h : Real.sin (θ - π / 12) = 3 / 4) :
  Real.sin (2 * θ + π / 3) = -1 / 8 :=
by
  -- Proof would go here
  sorry

end sin_transform_l218_218055


namespace bisection_next_interval_l218_218067

-- Define the function f(x) = x^3 - 2x - 1
def f (x : ℝ) : ℝ := x^3 - 2*x - 1

-- Define the intervals (1, 2) and (1.5, 2)
def interval_initial : Set ℝ := {x | 1 < x ∧ x < 2}
def interval_next : Set ℝ := {x | 1.5 < x ∧ x < 2}

-- State the theorem, with conditions
theorem bisection_next_interval 
  (root_in_interval_initial : ∃ x, f x = 0 ∧ x ∈ interval_initial)
  (f_1_negative : f 1 < 0)
  (f_2_positive : f 2 > 0)
  : ∃ x, f x = 0 ∧ x ∈ interval_next :=
sorry

end bisection_next_interval_l218_218067


namespace enchanted_creatures_gala_handshakes_l218_218893

theorem enchanted_creatures_gala_handshakes :
  let goblins := 30
  let trolls := 20
  let goblin_handshakes := goblins * (goblins - 1) / 2
  let troll_to_goblin_handshakes := trolls * goblins
  goblin_handshakes + troll_to_goblin_handshakes = 1035 := 
by
  sorry

end enchanted_creatures_gala_handshakes_l218_218893


namespace perimeter_of_square_fence_l218_218423

theorem perimeter_of_square_fence :
  ∀ (n : ℕ) (post_gap post_width : ℝ), 
  4 * n - 4 = 24 →
  post_gap = 6 →
  post_width = 5 / 12 →
  4 * ((n - 1) * post_gap + n * post_width) = 156 :=
by
  intros n post_gap post_width h1 h2 h3
  sorry

end perimeter_of_square_fence_l218_218423


namespace not_polynomial_option_B_l218_218510

-- Definitions
def is_polynomial (expr : String) : Prop :=
  -- Assuming we have a function that determines if a given string expression is a polynomial.
  sorry

def option_A : String := "m+n"
def option_B : String := "x=1"
def option_C : String := "xy"
def option_D : String := "0"

-- Problem Statement
theorem not_polynomial_option_B : ¬ is_polynomial option_B := 
sorry

end not_polynomial_option_B_l218_218510


namespace matrix_vec_addition_l218_218185

def matrix := (Fin 2 → Fin 2 → ℤ)
def vector := Fin 2 → ℤ

def m : matrix := ![![4, -2], ![6, 5]]
def v1 : vector := ![-2, 3]
def v2 : vector := ![1, -1]

def matrix_vec_mul (m : matrix) (v : vector) : vector :=
  ![m 0 0 * v 0 + m 0 1 * v 1,
    m 1 0 * v 0 + m 1 1 * v 1]

def vec_add (v1 v2 : vector) : vector :=
  ![v1 0 + v2 0, v1 1 + v2 1]

theorem matrix_vec_addition :
  vec_add (matrix_vec_mul m v1) v2 = ![-13, 2] :=
by
  sorry

end matrix_vec_addition_l218_218185


namespace monotonic_function_range_l218_218480

theorem monotonic_function_range (a : ℝ) :
  (∀ x : ℝ, -3 * x^2 + 2 * a * x - 1 ≤ 0) → -Real.sqrt 3 ≤ a ∧ a ≤ Real.sqrt 3 :=
by
  sorry

end monotonic_function_range_l218_218480


namespace cyclic_quadrilateral_sides_equal_l218_218863

theorem cyclic_quadrilateral_sides_equal
  (A B C D P : ℝ) -- Points represented as reals for simplicity
  (AB CD BC AD : ℝ) -- Lengths of sides AB, CD, BC, AD
  (a b c d e θ : ℝ) -- Various lengths and angle as given in the solution
  (h1 : a + e = b + c + d)
  (h2 : (1 / 2) * a * e * Real.sin θ = (1 / 2) * b * e * Real.sin θ + (1 / 2) * c * d * Real.sin θ) :
  c = e ∨ d = e := sorry

end cyclic_quadrilateral_sides_equal_l218_218863


namespace online_game_months_l218_218663

theorem online_game_months (m : ℕ) (initial_cost monthly_cost total_cost : ℕ) 
  (h1 : initial_cost = 5) (h2 : monthly_cost = 8) (h3 : total_cost = 21) 
  (h_equation : initial_cost + monthly_cost * m = total_cost) : m = 2 :=
by {
  -- Placeholder for the proof, as we don't need to include it
  sorry
}

end online_game_months_l218_218663


namespace value_of_unknown_number_l218_218120

theorem value_of_unknown_number (x n : ℤ) 
  (h1 : x = 88320) 
  (h2 : x + n + 9211 - 1569 = 11901) : 
  n = -84061 :=
by
  sorry

end value_of_unknown_number_l218_218120


namespace ordered_pair_l218_218692

-- Definitions
def P (x : ℝ) := x^4 - 8 * x^3 + 20 * x^2 - 34 * x + 15
def D (k : ℝ) (x : ℝ) := x^2 - 3 * x + k
def R (a : ℝ) (x : ℝ) := x + a

-- Hypothesis
def condition (k a : ℝ) : Prop := ∀ x : ℝ, P x % D k x = R a x

-- Theorem
theorem ordered_pair (k a : ℝ) (h : condition k a) : (k, a) = (5, 15) := 
  sorry

end ordered_pair_l218_218692


namespace larger_number_l218_218531

theorem larger_number (x y : ℕ) (h1 : x + y = 40) (h2 : x - y = 4) : x = 22 := by
  sorry

end larger_number_l218_218531


namespace smallest_number_of_white_marbles_l218_218583

theorem smallest_number_of_white_marbles
  (n : ℕ)
  (hn1 : n > 0)
  (orange_marbles : ℕ := n / 5)
  (hn_orange : n % 5 = 0)
  (purple_marbles : ℕ := n / 6)
  (hn_purple : n % 6 = 0)
  (green_marbles : ℕ := 9)
  : (n - (orange_marbles + purple_marbles + green_marbles)) = 10 → n = 30 :=
by
  sorry

end smallest_number_of_white_marbles_l218_218583


namespace roots_eq_202_l218_218125

theorem roots_eq_202 (p q : ℝ) 
  (h1 : ∀ x : ℝ, ((x + p) * (x + q) * (x + 10) = 0 ↔ (x = -p ∨ x = -q ∨ x = -10)) ∧ 
       ∀ x : ℝ, ((x + 5) ^ 2 = 0 ↔ x = -5)) 
  (h2 : ∀ x : ℝ, ((x + 2 * p) * (x + 4) * (x + 8) = 0 ↔ (x = -2 * p ∨ x = -4 ∨ x = -8)) ∧ 
       ∀ x : ℝ, ((x + q) * (x + 10) = 0 ↔ (x = -q ∨ x = -10))) 
  (hpq : p = q) (neq_5 : q ≠ 5) (p_2 : p = 2):
  100 * p + q = 202 := sorry

end roots_eq_202_l218_218125


namespace minimize_sum_of_squares_of_roots_l218_218517

theorem minimize_sum_of_squares_of_roots (m : ℝ) (h : 100 - 20 * m ≥ 0) :
  (∀ a b : ℝ, (∀ x : ℝ, 5 * x^2 - 10 * x + m = 0 → x = a ∨ x = b) → (4 - 2 * m / 5) ≥ (4 - 2 * 5 / 5)) :=
by
  sorry

end minimize_sum_of_squares_of_roots_l218_218517


namespace initial_girls_count_l218_218046

-- Define the variables
variables (b g : ℕ)

-- Conditions
def condition1 := b = 3 * (g - 20)
def condition2 := 4 * (b - 60) = g - 20

-- Statement of the problem
theorem initial_girls_count
  (h1 : condition1 b g)
  (h2 : condition2 b g) : g = 460 / 11 := 
sorry

end initial_girls_count_l218_218046


namespace power_function_alpha_l218_218419

theorem power_function_alpha (α : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = x^α) (point_condition : f 8 = 2) : 
  α = 1 / 3 :=
by
  sorry

end power_function_alpha_l218_218419


namespace expression_range_l218_218873

theorem expression_range (a b c d : ℝ) (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2)
  (hc : 0 ≤ c ∧ c ≤ 2) (hd : 0 ≤ d ∧ d ≤ 2) :
  4 * Real.sqrt 2 ≤ (Real.sqrt (a^2 + (2 - b)^2) + Real.sqrt (b^2 + (2 - c)^2)
  + Real.sqrt (c^2 + (2 - d)^2) + Real.sqrt (d^2 + (2 - a)^2)) ∧ 
  (Real.sqrt (a^2 + (2 - b)^2) + Real.sqrt (b^2 + (2 - c)^2) + Real.sqrt (c^2 + (2 - d)^2) + Real.sqrt (d^2 + (2 - a)^2)) ≤ 8 :=
sorry

end expression_range_l218_218873


namespace correct_calculation_l218_218628

def original_number (x : ℕ) : Prop := x + 12 = 48

theorem correct_calculation (x : ℕ) (h : original_number x) : x + 22 = 58 := by
  sorry

end correct_calculation_l218_218628


namespace find_m_l218_218525

theorem find_m 
  (m : ℝ)
  (h_pos : 0 < m)
  (asymptote_twice_angle : ∃ l : ℝ, l = 3 ∧ (x - l * y = 0 ∧ m * x^2 - y^2 = m)) :
  m = 3 :=
by
  sorry

end find_m_l218_218525


namespace center_of_circle_tangent_to_parallel_lines_l218_218169

-- Define the line equations
def line1 (x y : ℝ) : Prop := 3 * x - 4 * y = 40
def line2 (x y : ℝ) : Prop := 3 * x - 4 * y = -20
def line3 (x y : ℝ) : Prop := x - 2 * y = 0

-- The proof problem
theorem center_of_circle_tangent_to_parallel_lines
  (x y : ℝ)
  (h1 : line1 x y → false)
  (h2 : line2 x y → false)
  (h3 : line3 x y) :
  x = 10 ∧ y = 5 := by
  sorry

end center_of_circle_tangent_to_parallel_lines_l218_218169


namespace harry_carries_buckets_rounds_l218_218388

noncomputable def george_rate := 2
noncomputable def total_buckets := 110
noncomputable def total_rounds := 22
noncomputable def harry_buckets_each_round := 3

theorem harry_carries_buckets_rounds :
  (george_rate * total_rounds + harry_buckets_each_round * total_rounds = total_buckets) :=
by sorry

end harry_carries_buckets_rounds_l218_218388


namespace remainder_of_m_l218_218245

theorem remainder_of_m (m : ℕ) (h₁ : m ^ 3 % 7 = 6) (h₂ : m ^ 4 % 7 = 4) : m % 7 = 3 := 
sorry

end remainder_of_m_l218_218245


namespace ball_and_ring_problem_l218_218263

theorem ball_and_ring_problem (x y : ℕ) (m_x m_y : ℕ) : 
  m_x + 2 = y ∧ 
  m_y = x + 2 ∧
  x * m_x + y * m_y - 800 = 2 * (y - x) ∧
  x^2 + y^2 = 881 →
  (x = 25 ∧ y = 16) ∨ (x = 16 ∧ y = 25) := 
by 
  sorry

end ball_and_ring_problem_l218_218263


namespace arithmetic_sequence_a1_a6_l218_218691

theorem arithmetic_sequence_a1_a6
  (a : ℕ → ℤ)
  (h_arith_seq : ∀ n : ℕ, a n = a 1 + (n - 1) * (a 2 - a 1))
  (h_a2 : a 2 = 3)
  (h_sum : a 3 + a 4 = 9) : a 1 * a 6 = 14 :=
sorry

end arithmetic_sequence_a1_a6_l218_218691


namespace find_a_and_b_minimum_value_of_polynomial_l218_218594

noncomputable def polynomial_has_maximum (x y a b : ℝ) : Prop :=
  y = a * x ^ 3 + b * x ^ 2 ∧ x = 1 ∧ y = 3

noncomputable def polynomial_minimum_value (y : ℝ) : Prop :=
  y = 0

theorem find_a_and_b (a b x y : ℝ) (h : polynomial_has_maximum x y a b) :
  a = -6 ∧ b = 9 :=
by sorry

theorem minimum_value_of_polynomial (a b y : ℝ) (h : a = -6 ∧ b = 9) :
  polynomial_minimum_value y :=
by sorry

end find_a_and_b_minimum_value_of_polynomial_l218_218594


namespace number_condition_l218_218974

theorem number_condition (x : ℤ) (h : x - 7 = 9) : 5 * x = 80 := by
  sorry

end number_condition_l218_218974


namespace points_four_units_away_l218_218683

theorem points_four_units_away (x : ℤ) : (x - (-1) = 4 ∨ x - (-1) = -4) ↔ (x = 3 ∨ x = -5) :=
by
  sorry

end points_four_units_away_l218_218683


namespace truck_and_trailer_total_weight_l218_218987

def truck_weight : ℝ := 4800
def trailer_weight (truck_weight : ℝ) : ℝ := 0.5 * truck_weight - 200
def total_weight (truck_weight trailer_weight : ℝ) : ℝ := truck_weight + trailer_weight 

theorem truck_and_trailer_total_weight : 
  total_weight truck_weight (trailer_weight truck_weight) = 7000 :=
by 
  sorry

end truck_and_trailer_total_weight_l218_218987


namespace sum_of_digits_l218_218391

def S (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits :
  (Finset.range 2013).sum S = 28077 :=
by 
  sorry

end sum_of_digits_l218_218391


namespace largest_x_value_l218_218005

noncomputable def quadratic_eq (x : ℝ) : Prop :=
  3 * (9 * x^2 + 15 * x + 20) = x * (9 * x - 60)

theorem largest_x_value (x : ℝ) :
  quadratic_eq x → x = - ((35 - Real.sqrt 745) / 12) ∨
  x = - ((35 + Real.sqrt 745) / 12) :=
by
  intro h
  sorry

end largest_x_value_l218_218005


namespace sequence_formula_l218_218898

theorem sequence_formula (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n ≥ 2, a n = 2 * a (n - 1) + 1) :
  ∀ n : ℕ, a n = 2 ^ n - 1 :=
sorry

end sequence_formula_l218_218898


namespace smallest_four_digit_divisible_by_6_l218_218878

-- Define the smallest four-digit number
def smallest_four_digit_number := 1000

-- Define divisibility conditions
def divisible_by_2 (n : Nat) := n % 2 = 0
def divisible_by_3 (n : Nat) := n % 3 = 0
def divisible_by_6 (n : Nat) := divisible_by_2 n ∧ divisible_by_3 n

-- Prove that the smallest four-digit number divisible by 6 is 1002
theorem smallest_four_digit_divisible_by_6 : ∃ n : Nat, n ≥ smallest_four_digit_number ∧ divisible_by_6 n ∧ ∀ m : Nat, m ≥ smallest_four_digit_number ∧ divisible_by_6 m → n ≤ m :=
  sorry

end smallest_four_digit_divisible_by_6_l218_218878


namespace min_value_xy_inv_xy_l218_218567

theorem min_value_xy_inv_xy (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy_sum : x + y = 2) :
  ∃ m : ℝ, m = xy + 4 / xy ∧ m ≥ 5 :=
by
  sorry

end min_value_xy_inv_xy_l218_218567


namespace original_three_numbers_are_arith_geo_seq_l218_218069

theorem original_three_numbers_are_arith_geo_seq
  (x y z : ℕ) (h1 : ∃ k : ℕ, x = 3*k ∧ y = 4*k ∧ z = 5*k)
  (h2 : ∃ r : ℝ, (x + 1) / y = r ∧ y / z = r ∧ r^2 = z / y):
  x = 15 ∧ y = 20 ∧ z = 25 :=
by 
  sorry

end original_three_numbers_are_arith_geo_seq_l218_218069


namespace trains_meet_time_l218_218295

theorem trains_meet_time :
  (∀ (D : ℝ) (s1 s2 t1 t2 : ℝ),
    D = 155 ∧ 
    s1 = 20 ∧ 
    s2 = 25 ∧ 
    t1 = 7 ∧ 
    t2 = 8 →
    (∃ t : ℝ, 20 * t + 25 * t = D - 20)) →
  8 + 3 = 11 :=
by {
  sorry
}

end trains_meet_time_l218_218295


namespace tree_current_height_l218_218404

theorem tree_current_height 
  (growth_rate_per_week : ℕ)
  (weeks_per_month : ℕ)
  (total_height_after_4_months : ℕ) 
  (growth_rate_per_week_eq : growth_rate_per_week = 2)
  (weeks_per_month_eq : weeks_per_month = 4)
  (total_height_after_4_months_eq : total_height_after_4_months = 42) : 
  (∃ (current_height : ℕ), current_height = 10) :=
by
  sorry

end tree_current_height_l218_218404


namespace chef_bought_kilograms_of_almonds_l218_218101

def total_weight_of_nuts : ℝ := 0.52
def weight_of_pecans : ℝ := 0.38
def weight_of_almonds : ℝ := total_weight_of_nuts - weight_of_pecans

theorem chef_bought_kilograms_of_almonds : weight_of_almonds = 0.14 := by
  sorry

end chef_bought_kilograms_of_almonds_l218_218101


namespace distinct_diagonals_nonagon_l218_218950

def n : ℕ := 9

def diagonals_nonagon (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem distinct_diagonals_nonagon : diagonals_nonagon n = 27 :=
by
  unfold diagonals_nonagon
  norm_num
  sorry

end distinct_diagonals_nonagon_l218_218950


namespace find_divisor_l218_218843

variable (dividend quotient remainder divisor : ℕ)

theorem find_divisor (h1 : dividend = 52) (h2 : quotient = 16) (h3 : remainder = 4) (h4 : dividend = divisor * quotient + remainder) : 
  divisor = 3 := by
  sorry

end find_divisor_l218_218843


namespace determinant_matrix_A_l218_218831

open Matrix

def matrix_A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![5, 0, -2], ![1, 3, 4], ![0, -1, 1]]

theorem determinant_matrix_A :
  det matrix_A = 33 :=
by
  sorry

end determinant_matrix_A_l218_218831


namespace automobile_travel_distance_l218_218571

theorem automobile_travel_distance (a r : ℝ) :
  (2 * a / 5) / (2 * r) * 5 * 60 / 3 = 20 * a / r :=
by 
  -- skipping proof details
  sorry

end automobile_travel_distance_l218_218571


namespace largest_divisor_three_consecutive_l218_218233

theorem largest_divisor_three_consecutive (u v w : ℤ) (h1 : u + 1 = v) (h2 : v + 1 = w) (h3 : ∃ n : ℤ, (u = 5 * n) ∨ (v = 5 * n) ∨ (w = 5 * n)) : 
  ∀ d ∈ {d | ∀ a b c : ℤ, a * b * c = u * v * w → d ∣ a * b * c}, 
  15 ∈ {d | ∀ a b c : ℤ, a * b * c = u * v * w → d ∣ a * b * c} :=
sorry

end largest_divisor_three_consecutive_l218_218233


namespace cost_per_square_meter_l218_218474

-- Definitions from conditions
def lawn_length : ℝ := 80
def lawn_breadth : ℝ := 50
def road_width : ℝ := 10
def total_cost : ℝ := 3600

-- Theorem to prove the cost per square meter of traveling the roads
theorem cost_per_square_meter :
  total_cost / 
  ((lawn_length * road_width) + (lawn_breadth * road_width) - (road_width * road_width)) = 3 := by
  sorry

end cost_per_square_meter_l218_218474


namespace simplify_expression_l218_218938

variable (b : ℝ)

theorem simplify_expression (b : ℝ) : 
  (3 * b + 7 - 5 * b) / 3 = (-2 / 3) * b + (7 / 3) :=
by
  sorry

end simplify_expression_l218_218938


namespace remaining_budget_correct_l218_218762

def cost_item1 := 13
def cost_item2 := 24
def last_year_remaining_budget := 6
def this_year_budget := 50

theorem remaining_budget_correct :
    (last_year_remaining_budget + this_year_budget - (cost_item1 + cost_item2) = 19) :=
by
  -- This is the statement only, with the proof omitted
  sorry

end remaining_budget_correct_l218_218762


namespace new_area_rhombus_l218_218603

theorem new_area_rhombus (d1 d2 : ℝ) (h : (d1 * d2) / 2 = 3) : 
  ((5 * d1) * (5 * d2)) / 2 = 75 := 
by
  sorry

end new_area_rhombus_l218_218603


namespace find_triples_solution_l218_218378

theorem find_triples_solution (x y z : ℕ) (h : x^5 + x^4 + 1 = 3^y * 7^z) :
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 1 ∧ z = 0) ∨ (x = 2 ∧ y = 0 ∧ z = 2) :=
by
  sorry

end find_triples_solution_l218_218378


namespace DeMorgansLaws_l218_218326

variable (U : Type) (A B : Set U)

theorem DeMorgansLaws :
  (A ∪ B)ᶜ = Aᶜ ∩ Bᶜ ∧ (A ∩ B)ᶜ = Aᶜ ∪ Bᶜ :=
by
  -- Statement of the theorems, proof is omitted
  sorry

end DeMorgansLaws_l218_218326


namespace geometric_series_first_term_l218_218402

theorem geometric_series_first_term (r : ℚ) (S : ℚ) (a : ℚ) (h_r : r = 1 / 4) (h_S : S = 80) (h_sum : S = a / (1 - r)) : a = 60 :=
by {
  sorry
}

end geometric_series_first_term_l218_218402


namespace pentagon_area_eq_half_l218_218355

variables {A B C D E : Type*} -- Assume A, B, C, D, E are some points in a plane

-- Assume the given conditions in the problem
variables (angle_A angle_C : ℝ)
variables (AB AE BC CD AC : ℝ)
variables (pentagon_area : ℝ)

-- Assume the constraints from the problem statement
axiom angle_A_eq_90 : angle_A = 90
axiom angle_C_eq_90 : angle_C = 90
axiom AB_eq_AE : AB = AE
axiom BC_eq_CD : BC = CD
axiom AC_eq_1 : AC = 1

theorem pentagon_area_eq_half : pentagon_area = 1 / 2 :=
sorry

end pentagon_area_eq_half_l218_218355


namespace sides_of_right_triangle_l218_218455

theorem sides_of_right_triangle (r : ℝ) (a b c : ℝ) 
  (h_area : (2 / (2 / r)) * 2 = 2 * r) 
  (h_right : a^2 + b^2 = c^2) :
  (a = r ∧ b = (4 / 3) * r ∧ c = (5 / 3) * r) ∨
  (b = r ∧ a = (4 / 3) * r ∧ c = (5 / 3) * r) :=
sorry

end sides_of_right_triangle_l218_218455


namespace xy_computation_l218_218769

theorem xy_computation (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : 
  x * y = 21 := by
  sorry

end xy_computation_l218_218769


namespace total_amount_paid_l218_218475

def price_grapes (kg: ℕ) (rate: ℕ) : ℕ := kg * rate
def price_mangoes (kg: ℕ) (rate: ℕ) : ℕ := kg * rate
def price_pineapple (kg: ℕ) (rate: ℕ) : ℕ := kg * rate
def price_kiwi (kg: ℕ) (rate: ℕ) : ℕ := kg * rate

theorem total_amount_paid :
  price_grapes 14 54 + price_mangoes 10 62 + price_pineapple 8 40 + price_kiwi 5 30 = 1846 :=
by
  sorry

end total_amount_paid_l218_218475


namespace find_pairs_l218_218783

theorem find_pairs (x y p : ℕ)
  (h1 : 1 ≤ x) (h2 : 1 ≤ y) (h3 : x ≤ y) (h4 : Prime p) :
  (x = 3 ∧ y = 5 ∧ p = 7) ∨ (x = 1 ∧ ∃ q, Prime q ∧ y = q + 1 ∧ p = q ∧ q ≠ 7) ↔
  (x + y) * (x * y - 1) / (x * y + 1) = p := 
sorry

end find_pairs_l218_218783


namespace red_users_count_l218_218906

noncomputable def total_students : ℕ := 70
noncomputable def green_users : ℕ := 52
noncomputable def both_colors_users : ℕ := 38

theorem red_users_count : 
  ∀ (R : ℕ), total_students = green_users + R - both_colors_users → R = 56 :=
by
  sorry

end red_users_count_l218_218906


namespace min_value_of_quadratic_l218_218560

theorem min_value_of_quadratic : ∀ x : ℝ, z = x^2 + 16*x + 20 → ∃ m : ℝ, m ≤ z :=
by
  sorry

end min_value_of_quadratic_l218_218560


namespace divide_90_into_two_parts_l218_218415

theorem divide_90_into_two_parts (x y : ℝ) (h : x + y = 90) 
  (cond : 0.4 * x = 0.3 * y + 15) : x = 60 ∨ y = 60 := 
by
  sorry

end divide_90_into_two_parts_l218_218415


namespace muffin_banana_ratio_l218_218503

variables (m b : ℝ)

theorem muffin_banana_ratio (h1 : 4 * m + 3 * b = x) 
                            (h2 : 2 * (4 * m + 3 * b) = 2 * m + 16 * b) : 
                            m / b = 5 / 3 :=
by sorry

end muffin_banana_ratio_l218_218503


namespace game_rounds_l218_218903

noncomputable def play_game (A B C D : ℕ) : ℕ := sorry

theorem game_rounds : play_game 16 15 14 13 = 49 :=
by
  sorry

end game_rounds_l218_218903


namespace remainder_when_P_divided_by_ab_l218_218229

-- Given conditions
variables {P a b c Q Q' R R' : ℕ}

-- Provided equations as conditions
def equation1 : P = a * Q + R :=
sorry

def equation2 : Q = (b + c) * Q' + R' :=
sorry

-- Proof problem statement
theorem remainder_when_P_divided_by_ab :
  P % (a * b) = (a * c * Q' + a * R' + R) % (a * b) :=
by
  sorry

end remainder_when_P_divided_by_ab_l218_218229


namespace smallest_n_for_factorization_l218_218636

theorem smallest_n_for_factorization :
  ∃ n : ℤ, (∀ A B : ℤ, A * B = 60 ↔ n = 5 * B + A) ∧ n = 56 :=
by
  sorry

end smallest_n_for_factorization_l218_218636


namespace k_sq_geq_25_over_4_l218_218775

theorem k_sq_geq_25_over_4
  (a1 a2 a3 a4 a5 k : ℝ)
  (h1 : |a1 - a2| ≥ 1 ∧ |a1 - a3| ≥ 1 ∧ |a1 - a4| ≥ 1 ∧ |a1 - a5| ≥ 1 ∧
       |a2 - a3| ≥ 1 ∧ |a2 - a4| ≥ 1 ∧ |a2 - a5| ≥ 1 ∧
       |a3 - a4| ≥ 1 ∧ |a3 - a5| ≥ 1 ∧
       |a4 - a5| ≥ 1)
  (h2 : a1 + a2 + a3 + a4 + a5 = 2 * k)
  (h3 : a1^2 + a2^2 + a3^2 + a4^2 + a5^2 = 2 * k^2) :
  k^2 ≥ 25 / 4 :=
sorry

end k_sq_geq_25_over_4_l218_218775


namespace remainders_sum_l218_218804

theorem remainders_sum (a b c : ℕ) 
  (h1 : a % 30 = 15) 
  (h2 : b % 30 = 20) 
  (h3 : c % 30 = 10) : 
  (a + b + c) % 30 = 15 := 
by
  sorry

end remainders_sum_l218_218804


namespace calc_fraction_l218_218724

theorem calc_fraction:
  (125: ℕ) = 5 ^ 3 →
  (25: ℕ) = 5 ^ 2 →
  (25 ^ 40) / (125 ^ 20) = 5 ^ 20 :=
by
  intros h1 h2
  sorry

end calc_fraction_l218_218724


namespace twenty_four_times_ninety_nine_l218_218183

theorem twenty_four_times_ninety_nine : 24 * 99 = 2376 :=
by sorry

end twenty_four_times_ninety_nine_l218_218183


namespace find_positive_integer_k_l218_218488

theorem find_positive_integer_k (p : ℕ) (hp : Prime p) (hp2 : Odd p) : 
  ∃ k : ℕ, k > 0 ∧ ∃ n : ℕ, n * n = k - p * k ∧ k = ((p + 1) * (p + 1)) / 4 :=
by
  sorry

end find_positive_integer_k_l218_218488


namespace work_done_together_in_one_day_l218_218373

-- Defining the conditions
def time_to_finish_a : ℕ := 12
def time_to_finish_b : ℕ := time_to_finish_a / 2

-- Defining the work done in one day
def work_done_by_a_in_one_day : ℚ := 1 / time_to_finish_a
def work_done_by_b_in_one_day : ℚ := 1 / time_to_finish_b

-- The proof statement
theorem work_done_together_in_one_day : 
  work_done_by_a_in_one_day + work_done_by_b_in_one_day = 1 / 4 := by
  sorry

end work_done_together_in_one_day_l218_218373


namespace simplify_expression_l218_218641

theorem simplify_expression : (8^(1/3) / 8^(1/6)) = 8^(1/6) :=
by
  sorry

end simplify_expression_l218_218641


namespace eliminate_y_l218_218220

theorem eliminate_y (x y : ℝ) (h1 : 2 * x + 3 * y = 1) (h2 : 3 * x - 6 * y = 7) :
  (4 * x + 6 * y) + (3 * x - 6 * y) = 9 :=
by
  sorry

end eliminate_y_l218_218220


namespace crayons_per_box_l218_218690

theorem crayons_per_box (total_crayons : ℕ) (total_boxes : ℕ)
  (h1 : total_crayons = 321)
  (h2 : total_boxes = 45) :
  (total_crayons / total_boxes) = 7 :=
by
  sorry

end crayons_per_box_l218_218690


namespace remainder_549547_div_7_l218_218450

theorem remainder_549547_div_7 : 549547 % 7 = 5 :=
by
  sorry

end remainder_549547_div_7_l218_218450


namespace other_factor_of_lcm_l218_218715

theorem other_factor_of_lcm (A B : ℕ) 
  (hcf : Nat.gcd A B = 23) 
  (hA : A = 345) 
  (hcf_factor : 15 ∣ Nat.lcm A B) 
  : 23 ∣ Nat.lcm A B / 15 :=
sorry

end other_factor_of_lcm_l218_218715


namespace range_of_a_if_odd_symmetric_points_l218_218776

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a

theorem range_of_a_if_odd_symmetric_points (a : ℝ): 
  (∃ x₀ : ℝ, x₀ ≠ 0 ∧ f x₀ a = -f (-x₀) a) → (1 < a) :=
by 
  sorry

end range_of_a_if_odd_symmetric_points_l218_218776


namespace OJ_perpendicular_PQ_l218_218412

noncomputable def quadrilateral (A B C D : Point) : Prop := sorry

noncomputable def inscribed (A B C D : Point) : Prop := sorry

noncomputable def circumscribed (A B C D : Point) : Prop := sorry

noncomputable def no_diameter (A B C D : Point) : Prop := sorry

noncomputable def intersection_of_external_bisectors (A B C D : Point) (P : Point) : Prop := sorry

noncomputable def incenter (A B C D J : Point) : Prop := sorry

noncomputable def circumcenter (A B C D O : Point) : Prop := sorry

noncomputable def PQ_perpendicular (O J P Q : Point) : Prop := sorry

theorem OJ_perpendicular_PQ (A B C D P Q J O : Point) :
  quadrilateral A B C D →
  inscribed A B C D →
  circumscribed A B C D →
  no_diameter A B C D →
  intersection_of_external_bisectors A B C D P →
  intersection_of_external_bisectors C D A B Q →
  incenter A B C D J →
  circumcenter A B C D O →
  PQ_perpendicular O J P Q :=
sorry

end OJ_perpendicular_PQ_l218_218412


namespace fraction_upgraded_l218_218981

theorem fraction_upgraded :
  ∀ (N U : ℕ), 24 * N = 6 * U → (U : ℚ) / (24 * N + U) = 1 / 7 :=
by
  intros N U h_eq
  sorry

end fraction_upgraded_l218_218981


namespace odd_periodic_function_l218_218190

variable {f : ℝ → ℝ}

-- Given conditions
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def periodic_function (f : ℝ → ℝ) : Prop := ∀ x, f (x + 1) = -f x

-- Problem statement
theorem odd_periodic_function (h_odd : odd_function f)
  (h_period : periodic_function f) (h_half : f 0.5 = 1) : f 7.5 = -1 :=
sorry

end odd_periodic_function_l218_218190


namespace max_value_of_m_l218_218929

variable (m : ℝ)

noncomputable def satisfies_inequality (m : ℝ) : Prop :=
∀ x > 0, m * x * Real.log x - (x + m) * Real.exp ((x - m) / m) ≤ 0

theorem max_value_of_m (h1 : 0 < m) (h2 : satisfies_inequality m) : m ≤ Real.exp 2 := sorry

end max_value_of_m_l218_218929


namespace probability_of_triangle_or_circle_l218_218001

/-- The total number of figures -/
def total_figures : ℕ := 10

/-- The number of triangles -/
def triangles : ℕ := 3

/-- The number of circles -/
def circles : ℕ := 3

/-- The number of figures that are either triangles or circles -/
def favorable_figures : ℕ := triangles + circles

/-- The probability that the chosen figure is either a triangle or a circle -/
theorem probability_of_triangle_or_circle : (favorable_figures : ℚ) / (total_figures : ℚ) = 3 / 5 := 
by
  sorry

end probability_of_triangle_or_circle_l218_218001


namespace number_of_students_l218_218273

theorem number_of_students (x : ℕ) (h : x * (x - 1) = 210) : x = 15 := 
by sorry

end number_of_students_l218_218273


namespace range_of_p_l218_218867

def p (x : ℝ) : ℝ := x^6 + 6 * x^3 + 9

theorem range_of_p : Set.Ici 9 = { y | ∃ x ≥ 0, p x = y } :=
by
  -- We skip the proof to only provide the statement as requested.
  sorry

end range_of_p_l218_218867


namespace shareCoins_l218_218927

theorem shareCoins (a b c d e d : ℝ)
  (h1 : b = a - d)
  (h2 : ((a-2*d) + b = a + (a+d) + (a+2*d)))
  (h3 : (a-2*d) + b + a + (a+d) + (a+2*d) = 5) :
  b = 7 / 6 :=
by
  sorry

end shareCoins_l218_218927


namespace Kelly_egg_price_l218_218960

/-- Kelly has 8 chickens, and each chicken lays 3 eggs per day.
Kelly makes $280 in 4 weeks by selling all the eggs.
We want to prove that Kelly sells a dozen eggs for $5. -/
theorem Kelly_egg_price (chickens : ℕ) (eggs_per_day_per_chicken : ℕ) (earnings_in_4_weeks : ℕ)
  (days_in_4_weeks : ℕ) (eggs_per_dozen : ℕ) (price_per_dozen : ℕ) :
  chickens = 8 →
  eggs_per_day_per_chicken = 3 →
  earnings_in_4_weeks = 280 →
  days_in_4_weeks = 28 →
  eggs_per_dozen = 12 →
  price_per_dozen = earnings_in_4_weeks / ((chickens * eggs_per_day_per_chicken * days_in_4_weeks) / eggs_per_dozen) →
  price_per_dozen = 5 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end Kelly_egg_price_l218_218960


namespace seashells_total_l218_218748

def seashells_sam : ℕ := 18
def seashells_mary : ℕ := 47
def seashells_john : ℕ := 32
def seashells_emily : ℕ := 26

theorem seashells_total : seashells_sam + seashells_mary + seashells_john + seashells_emily = 123 := by
    sorry

end seashells_total_l218_218748


namespace smallest_number_of_rectangles_needed_l218_218249

theorem smallest_number_of_rectangles_needed :
  ∃ n, (n * 12 = 144) ∧ (∀ k, (k * 12 = 144) → k ≥ n) := by
  sorry

end smallest_number_of_rectangles_needed_l218_218249


namespace correct_triangle_l218_218595

-- Define the conditions for the sides of each option
def sides_A := (1, 2, 3)
def sides_B := (3, 4, 5)
def sides_C := (3, 1, 1)
def sides_D := (3, 4, 7)

-- Conditions for forming a triangle
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Prove the problem statement
theorem correct_triangle : is_triangle 3 4 5 :=
by
  sorry

end correct_triangle_l218_218595


namespace radius_ratio_of_circumscribed_truncated_cone_l218_218814

theorem radius_ratio_of_circumscribed_truncated_cone 
  (R r ρ : ℝ) 
  (h : ℝ) 
  (Vcs Vg : ℝ) 
  (h_eq : h = 2 * ρ)
  (Vcs_eq : Vcs = (π / 3) * h * (R^2 + r^2 + R * r))
  (Vg_eq : Vg = (4 * π * (ρ^3)) / 3)
  (Vcs_Vg_eq : Vcs = 2 * Vg) :
  (R / r) = (3 + Real.sqrt 5) / 2 := 
sorry

end radius_ratio_of_circumscribed_truncated_cone_l218_218814


namespace simplify_expression_l218_218805

def a : ℕ := 1050
def p : ℕ := 2101
def q : ℕ := 1050 * 1051

theorem simplify_expression : 
  (1051 / 1050) - (1050 / 1051) = (p : ℚ) / (q : ℚ) ∧ Nat.gcd p a = 1 ∧ Nat.gcd p (a + 1) = 1 :=
by 
  sorry

end simplify_expression_l218_218805


namespace comprehensive_score_l218_218611

variable (regularAssessmentScore : ℕ)
variable (finalExamScore : ℕ)
variable (regularAssessmentWeighting : ℝ)
variable (finalExamWeighting : ℝ)

theorem comprehensive_score 
  (h1 : regularAssessmentScore = 95)
  (h2 : finalExamScore = 90)
  (h3 : regularAssessmentWeighting = 0.20)
  (h4 : finalExamWeighting = 0.80) :
  (regularAssessmentScore * regularAssessmentWeighting + finalExamScore * finalExamWeighting) = 91 :=
sorry

end comprehensive_score_l218_218611


namespace triplets_of_positive_integers_l218_218439

/-- We want to determine all positive integer triplets (a, b, c) such that
    ab - c, bc - a, and ca - b are all powers of 2.
    A power of 2 is an integer of the form 2^n, where n is a non-negative integer.-/
theorem triplets_of_positive_integers (a b c : ℕ) (h1: 0 < a) (h2: 0 < b) (h3: 0 < c) :
  ((∃ k1 : ℕ, ab - c = 2^k1) ∧ (∃ k2 : ℕ, bc - a = 2^k2) ∧ (∃ k3 : ℕ, ca - b = 2^k3))
  ↔ (a = 2 ∧ b = 2 ∧ c = 2) ∨ (a = 3 ∧ b = 2 ∧ c = 2) ∨ (a = 2 ∧ b = 6 ∧ c = 11) ∨ (a = 3 ∧ b = 5 ∧ c = 7) :=
sorry

end triplets_of_positive_integers_l218_218439


namespace problem_a_problem_b_l218_218206

-- Define the conditions for problem (a):
variable (x y z : ℝ)
variable (h_xyz : x * y * z = 1)

theorem problem_a (hx : x ≠ 1) (hy : y ≠ 1) (hz : z ≠ 1) :
  (x^2 / (x - 1)^2) + (y^2 / (y - 1)^2) + (z^2 / (z - 1)^2) ≥ 1 :=
sorry

-- Define the conditions for problem (b):
variable (a b c : ℚ)

theorem problem_b (h_abc : a * b * c = 1) :
  ∃ (x y z : ℚ), x ≠ 1 ∧ y ≠ 1 ∧ z ≠ 1 ∧ (x * y * z = 1) ∧ 
  (x^2 / (x - 1)^2 + y^2 / (y - 1)^2 + z^2 / (z - 1)^2 = 1) :=
sorry

end problem_a_problem_b_l218_218206


namespace tony_rope_length_l218_218585

-- Definitions based on the conditions in the problem
def rope_lengths : List ℝ := [8, 20, 2, 2, 2, 7]
def knot_loss_per_knot : ℝ := 1.2
def number_of_knots : ℕ := 5

-- The final length of the rope after tying all pieces together and losing length per knot
def final_rope_length (lengths : List ℝ) (loss_per_knot : ℝ) (number_of_knots : ℕ) : ℝ :=
  List.sum lengths - (loss_per_knot * number_of_knots)

theorem tony_rope_length :
  final_rope_length rope_lengths knot_loss_per_knot number_of_knots = 35 := by
  sorry

end tony_rope_length_l218_218585


namespace select_student_D_l218_218224

-- Define the scores and variances based on the conditions
def avg_A : ℝ := 96
def avg_B : ℝ := 94
def avg_C : ℝ := 93
def avg_D : ℝ := 96

def var_A : ℝ := 1.2
def var_B : ℝ := 1.2
def var_C : ℝ := 0.6
def var_D : ℝ := 0.4

-- Proof statement in Lean 4
theorem select_student_D (avg_A avg_B avg_C avg_D var_A var_B var_C var_D : ℝ) 
                         (h_avg_A : avg_A = 96)
                         (h_avg_B : avg_B = 94)
                         (h_avg_C : avg_C = 93)
                         (h_avg_D : avg_D = 96)
                         (h_var_A : var_A = 1.2)
                         (h_var_B : var_B = 1.2)
                         (h_var_C : var_C = 0.6)
                         (h_var_D : var_D = 0.4) 
                         (h_D_highest_avg : avg_D = max avg_A (max avg_B (max avg_C avg_D)))
                         (h_D_lowest_var : var_D = min (min (min var_A var_B) var_C) var_D) :
  avg_D = 96 ∧ var_D = 0.4 := 
by 
  -- As we're not asked to prove, we put sorry here to indicate the proof step is omitted.
  sorry

end select_student_D_l218_218224


namespace half_is_greater_than_third_by_one_sixth_l218_218460

theorem half_is_greater_than_third_by_one_sixth : (0.5 : ℝ) - (1 / 3 : ℝ) = 1 / 6 := by
  sorry

end half_is_greater_than_third_by_one_sixth_l218_218460


namespace triangle_area_l218_218520

theorem triangle_area {a c : ℝ} (h_a : a = 3 * Real.sqrt 3) (h_c : c = 2) (angle_B : ℝ) (h_B : angle_B = Real.pi / 3) : 
  (1 / 2) * a * c * Real.sin angle_B = 9 / 2 :=
by
  rw [h_a, h_c, h_B]
  sorry

end triangle_area_l218_218520


namespace adoption_event_l218_218847

theorem adoption_event (c : ℕ) 
  (h1 : ∀ d : ℕ, d = 8) 
  (h2 : ∀ fees_dog : ℕ, fees_dog = 15) 
  (h3 : ∀ fees_cat : ℕ, fees_cat = 13)
  (h4 : ∀ donation : ℕ, donation = 53)
  (h5 : fees_dog * 8 + fees_cat * c = 159) :
  c = 3 :=
by 
  sorry

end adoption_event_l218_218847


namespace num_intersecting_chords_on_circle_l218_218669

theorem num_intersecting_chords_on_circle (points : Fin 20 → Prop) : 
  ∃ num_chords : ℕ, num_chords = 156180 :=
by
  sorry

end num_intersecting_chords_on_circle_l218_218669


namespace inequality_solution_l218_218861

theorem inequality_solution
  : {x : ℝ | (x^2 / (x + 2)^2) ≥ 0} = {x : ℝ | x ≠ -2} :=
by
  sorry

end inequality_solution_l218_218861


namespace find_amplitude_l218_218800

theorem find_amplitude (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : ∀ x, a * Real.cos (b * x - c) ≤ 3) 
  (h5 : ∀ x, abs (a * Real.cos (b * x - c) - a * Real.cos (b * (x + 2 * π / b) - c)) = 0) :
  a = 3 := 
sorry

end find_amplitude_l218_218800


namespace hyperbola_asymptotes_l218_218549

theorem hyperbola_asymptotes (x y : ℝ) :
  (x^2 / 4 - y^2 = 1) → (y = x / 2 ∨ y = -x / 2) :=
sorry

end hyperbola_asymptotes_l218_218549


namespace largest_mersenne_prime_less_than_500_l218_218401

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_mersenne_prime (p : ℕ) : Prop := ∃ n : ℕ, is_prime n ∧ p = 2^n - 1

theorem largest_mersenne_prime_less_than_500 : ∃ p : ℕ, is_mersenne_prime p ∧ p < 500 ∧ ∀ q : ℕ, is_mersenne_prime q ∧ q < 500 → q ≤ p :=
sorry

end largest_mersenne_prime_less_than_500_l218_218401


namespace wood_length_equation_l218_218705

theorem wood_length_equation (x : ℝ) :
  (1 / 2) * (x + 4.5) = x - 1 :=
sorry

end wood_length_equation_l218_218705


namespace triangle_non_existence_no_solution_max_value_expression_l218_218888

-- Define sides and angles
variables {A B C : ℝ} -- Angles of the triangle
variables {a b c : ℝ} -- Corresponding opposite sides

-- Define the triangle conditions
def triangle_sides_angles (a b c A B C : ℝ) : Prop := 
  (a^2 = (1 - Real.cos A) / (1 - Real.cos B)) ∧ 
  (b = 1) ∧ 
  -- Additional properties ensuring we have a valid triangle can be added here
  (A ≠ B) -- Non-isosceles condition (equivalent to angles being different).

-- Prove non-existence under given conditions
theorem triangle_non_existence_no_solution (h : triangle_sides_angles a b c A B C) : false := 
sorry 

-- Define the maximization problem
theorem max_value_expression (h : a^2 = (1 - Real.cos A) / (1 - Real.cos B)) : 
(∃ b c, (b = 1) → ∀ a, a > 0 → (c > 0) ∧ ((1/c) * (1/b - 1/a)) ≤ (3 - 2 * Real.sqrt 2)) := 
sorry

end triangle_non_existence_no_solution_max_value_expression_l218_218888


namespace find_a_div_b_l218_218462

theorem find_a_div_b (a b : ℝ) (h : a ≠ b) (h_eq : a / b + (a + 6 * b) / (b + 6 * a) = 3) : 
  a / b = (8 + Real.sqrt 46) / 6 ∨ a / b = (8 - Real.sqrt 46) / 6 :=
by 
  sorry

end find_a_div_b_l218_218462


namespace rent_increase_percentage_l218_218703

theorem rent_increase_percentage :
  ∀ (initial_avg new_avg rent : ℝ) (num_friends : ℝ),
    num_friends = 4 →
    initial_avg = 800 →
    new_avg = 850 →
    rent = 800 →
    ((num_friends * new_avg) - (num_friends * initial_avg)) / rent * 100 = 25 :=
by
  intros initial_avg new_avg rent num_friends h_num h_initial h_new h_rent
  sorry

end rent_increase_percentage_l218_218703


namespace total_pizzas_eaten_l218_218294

-- Definitions for the conditions
def pizzasA : ℕ := 8
def pizzasB : ℕ := 7

-- Theorem stating the total number of pizzas eaten by both classes
theorem total_pizzas_eaten : pizzasA + pizzasB = 15 := 
by
  -- Proof is not required for the task, so we use sorry
  sorry

end total_pizzas_eaten_l218_218294


namespace num_valid_values_n_l218_218932

theorem num_valid_values_n :
  ∃ n : ℕ, (∃ a b c : ℕ,
    8 * a + 88 * b + 888 * c = 8880 ∧
    n = a + 2 * b + 3  * c) ∧
  (∃! k : ℕ, k = 119) :=
by sorry

end num_valid_values_n_l218_218932


namespace largest_square_multiple_of_18_under_500_l218_218071

theorem largest_square_multiple_of_18_under_500 : 
  ∃ n : ℕ, n * n < 500 ∧ n * n % 18 = 0 ∧ (∀ m : ℕ, m * m < 500 ∧ m * m % 18 = 0 → m * m ≤ n * n) → 
  n * n = 324 :=
by
  sorry

end largest_square_multiple_of_18_under_500_l218_218071


namespace bus_trip_cost_l218_218135

-- Problem Statement Definitions
def distance_AB : ℕ := 4500
def cost_per_kilometer_bus : ℚ := 0.20

-- Theorem Statement
theorem bus_trip_cost : distance_AB * cost_per_kilometer_bus = 900 := by
  sorry

end bus_trip_cost_l218_218135


namespace fraction_taken_out_is_one_sixth_l218_218217

-- Define the conditions
def original_cards : ℕ := 43
def cards_added_by_Sasha : ℕ := 48
def cards_left_after_Karen_took_out : ℕ := 83

-- Calculate the total number of cards initially after Sasha added hers
def total_cards_after_Sasha : ℕ := original_cards + cards_added_by_Sasha

-- Calculate the number of cards Karen took out
def cards_taken_out_by_Karen : ℕ := total_cards_after_Sasha - cards_left_after_Karen_took_out

-- Define the fraction of the cards Sasha added that Karen took out
def fraction_taken_out : ℚ := cards_taken_out_by_Karen / cards_added_by_Sasha

-- Proof statement: Fraction of the cards Sasha added that Karen took out is 1/6
theorem fraction_taken_out_is_one_sixth : fraction_taken_out = 1 / 6 :=
by
    -- Sorry is a placeholder for the proof, which is not required.
    sorry

end fraction_taken_out_is_one_sixth_l218_218217


namespace hyperbola_through_focus_and_asymptotes_l218_218834

noncomputable def parabola_focus : ℝ × ℝ := (1, 0)

def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 = 1

def asymptotes_holds (x y : ℝ) : Prop :=
  (x + y = 0) ∨ (x - y = 0)

theorem hyperbola_through_focus_and_asymptotes :
  hyperbola parabola_focus.1 parabola_focus.2 ∧ asymptotes_holds parabola_focus.1 parabola_focus.2 :=
sorry

end hyperbola_through_focus_and_asymptotes_l218_218834


namespace range_of_2alpha_minus_beta_l218_218911

def condition_range_alpha_beta (α β : ℝ) : Prop := 
  - (Real.pi / 2) < α ∧ α < β ∧ β < (Real.pi / 2)

theorem range_of_2alpha_minus_beta (α β : ℝ) (h : condition_range_alpha_beta α β) : 
  - Real.pi < 2 * α - β ∧ 2 * α - β < Real.pi / 2 :=
sorry

end range_of_2alpha_minus_beta_l218_218911


namespace fraction_ordering_l218_218266

theorem fraction_ordering :
  let a := (6 : ℚ) / 22
  let b := (8 : ℚ) / 32
  let c := (10 : ℚ) / 29
  a < b ∧ b < c :=
by
  sorry

end fraction_ordering_l218_218266


namespace meaningful_sqrt_l218_218466

theorem meaningful_sqrt (a : ℝ) (h : a ≥ 4) : a = 6 ↔ ∃ x ∈ ({-1, 0, 2, 6} : Set ℝ), x = 6 := 
by
  sorry

end meaningful_sqrt_l218_218466


namespace simplify_and_evaluate_at_3_l218_218420

noncomputable def expression (x : ℝ) : ℝ := 
  (3 / (x - 1) - x - 1) / ((x^2 - 4 * x + 4) / (x - 1))

theorem simplify_and_evaluate_at_3 : expression 3 = -5 := 
  sorry

end simplify_and_evaluate_at_3_l218_218420


namespace boat_speed_in_still_water_l218_218458

theorem boat_speed_in_still_water :
  ∀ (V_b V_s : ℝ) (distance time : ℝ),
  V_s = 5 →
  time = 4 →
  distance = 84 →
  (distance / time) = V_b + V_s →
  V_b = 16 :=
by
  -- Given definitions and values
  intros V_b V_s distance time
  intro hV_s
  intro htime
  intro hdistance
  intro heq
  sorry -- Placeholder for the actual proof

end boat_speed_in_still_water_l218_218458


namespace white_square_area_l218_218876

theorem white_square_area
    (edge_length : ℝ)
    (total_paint : ℝ)
    (total_surface_area : ℝ)
    (green_paint_per_face : ℝ)
    (white_square_area_per_face: ℝ) :
    edge_length = 12 →
    total_paint = 432 →
    total_surface_area = 6 * (edge_length ^ 2) →
    green_paint_per_face = total_paint / 6 →
    white_square_area_per_face = (edge_length ^ 2) - green_paint_per_face →
    white_square_area_per_face = 72
:= sorry

end white_square_area_l218_218876


namespace hyperbola_asymptote_product_l218_218454

theorem hyperbola_asymptote_product (k1 k2 : ℝ) (h1 : k1 = 1) (h2 : k2 = -1) :
  k1 * k2 = -1 :=
by
  rw [h1, h2]
  norm_num

end hyperbola_asymptote_product_l218_218454


namespace quadratic_real_roots_iff_range_of_a_l218_218882

theorem quadratic_real_roots_iff_range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 + 2*x + a = 0) ↔ a ≤ 1 :=
by
  sorry

end quadratic_real_roots_iff_range_of_a_l218_218882


namespace geometric_sequence_k_value_l218_218605

theorem geometric_sequence_k_value :
  ∀ {S : ℕ → ℤ} (a : ℕ → ℤ) (k : ℤ),
    (∀ n, S n = 3 * 2^n + k) → 
    (∀ n ≥ 2, a n = S n - S (n - 1)) → 
    (∀ n ≥ 2, a n ^ 2 = a 1 * a 3) → 
    k = -3 :=
by
  sorry

end geometric_sequence_k_value_l218_218605


namespace identify_linear_equation_l218_218467

def is_linear_equation (eq : String) : Prop := sorry

theorem identify_linear_equation :
  is_linear_equation "2x = 0" ∧ ¬is_linear_equation "x^2 - 4x = 3" ∧ ¬is_linear_equation "x + 2y = 1" ∧ ¬is_linear_equation "x - 1 = 1 / x" :=
by 
  sorry

end identify_linear_equation_l218_218467


namespace range_a_l218_218527

noncomputable def A (a : ℝ) : Set ℝ := {x | 2 * a + 1 ≤ x ∧ x ≤ 3 * a - 5}

noncomputable def B : Set ℝ := {x | x < -1 ∨ x > 16}

theorem range_a (a : ℝ) : (A a ∩ B = A a) → (a < 6 ∨ a > 7.5) :=
by
  intro h
  sorry

end range_a_l218_218527


namespace union_is_correct_l218_218753

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {0, 1, 2}
def union_set : Set ℤ := {-1, 0, 1, 2}

theorem union_is_correct : M ∪ N = union_set :=
  by sorry

end union_is_correct_l218_218753


namespace constructible_iff_multiple_of_8_l218_218553

def is_constructible_with_L_tetromino (m n : ℕ) : Prop :=
  ∃ (k : ℕ), 4 * k = m * n

theorem constructible_iff_multiple_of_8 (m n : ℕ) (hm : 1 < m) (hn : 1 < n) : 
  is_constructible_with_L_tetromino m n ↔ 8 ∣ m * n :=
sorry

end constructible_iff_multiple_of_8_l218_218553


namespace value_of_f_at_2_l218_218047

def f (x : ℝ) : ℝ :=
  x^3 - x - 1

theorem value_of_f_at_2 : f 2 = 5 := by
  -- Proof goes here
  sorry

end value_of_f_at_2_l218_218047


namespace tom_tickets_left_l218_218912

-- Define the conditions
def tickets_whack_a_mole : ℕ := 32
def tickets_skee_ball : ℕ := 25
def tickets_spent_on_hat : ℕ := 7

-- Define what we need to prove
theorem tom_tickets_left : tickets_whack_a_mole + tickets_skee_ball - tickets_spent_on_hat = 50 :=
by sorry

end tom_tickets_left_l218_218912


namespace circle_and_tangent_lines_l218_218894

open Real

noncomputable def equation_of_circle_center_on_line (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), (x - a)^2 + (y - (a + 1))^2 = 2 ∧ (a = 4) ∧ (b = 5)

noncomputable def tangent_line_through_point (x y : ℝ) : Prop :=
  y = x - 1 ∨ y = (23 / 7) * x - (23 / 7)

theorem circle_and_tangent_lines :
  (∃ (a b : ℝ), (a = 4) ∧ (b = 5) ∧ (∀ x y : ℝ, equation_of_circle_center_on_line x y)) ∧
  (∀ x y : ℝ, tangent_line_through_point x y) := 
  by
  sorry

end circle_and_tangent_lines_l218_218894


namespace problem_part1_problem_part2_l218_218586

def ellipse_condition (m : ℝ) : Prop :=
  m + 1 > 4 - m ∧ 4 - m > 0

def circle_condition (m : ℝ) : Prop :=
  m^2 - 4 > 0

theorem problem_part1 (m : ℝ) :
  ellipse_condition m → (3 / 2 < m ∧ m < 4) :=
sorry

theorem problem_part2 (m : ℝ) :
  ellipse_condition m ∧ circle_condition m → (2 < m ∧ m < 4) :=
sorry

end problem_part1_problem_part2_l218_218586


namespace larger_number_l218_218905

theorem larger_number (L S : ℕ) (h1 : L - S = 1345) (h2 : L = 6 * S + 15) : L = 1611 :=
by
  sorry

end larger_number_l218_218905


namespace find_number_l218_218992

theorem find_number 
  (x : ℝ)
  (h : (258 / 100 * x) / 6 = 543.95) :
  x = 1265 :=
sorry

end find_number_l218_218992


namespace monthly_increase_per_ticket_l218_218771

variable (x : ℝ)

theorem monthly_increase_per_ticket
    (initial_premium : ℝ := 50)
    (percent_increase_per_accident : ℝ := 0.10)
    (tickets : ℕ := 3)
    (final_premium : ℝ := 70) :
    initial_premium * (1 + percent_increase_per_accident) + tickets * x = final_premium → x = 5 :=
by
  intro h
  sorry

end monthly_increase_per_ticket_l218_218771


namespace value_of_seventh_observation_l218_218686

-- Given conditions
def sum_of_first_six_observations : ℕ := 90
def new_total_sum : ℕ := 98

-- Problem: prove the value of the seventh observation
theorem value_of_seventh_observation : new_total_sum - sum_of_first_six_observations = 8 :=
by
  sorry

end value_of_seventh_observation_l218_218686


namespace find_a_l218_218617

theorem find_a (a : ℝ) (h : (1 - 2016 * a) = 2017) : a = -1 := by
  -- proof omitted
  sorry

end find_a_l218_218617


namespace rowing_speed_in_still_water_l218_218745

theorem rowing_speed_in_still_water (speed_of_current : ℝ) (time_seconds : ℝ) (distance_meters : ℝ) (S : ℝ)
  (h_current : speed_of_current = 3) 
  (h_time : time_seconds = 9.390553103577801) 
  (h_distance : distance_meters = 60) 
  (h_S : S = 20) : 
  (distance_meters / 1000) / (time_seconds / 3600) - speed_of_current = S :=
by 
  sorry

end rowing_speed_in_still_water_l218_218745


namespace julia_baking_days_l218_218138

variable (bakes_per_day : ℕ)
variable (clifford_eats_per_two_days : ℕ)
variable (final_cakes : ℕ)

def number_of_baking_days : ℕ :=
  2 * (final_cakes / (bakes_per_day * 2 - clifford_eats_per_two_days))

theorem julia_baking_days (h1 : bakes_per_day = 4)
                        (h2 : clifford_eats_per_two_days = 1)
                        (h3 : final_cakes = 21) :
  number_of_baking_days bakes_per_day clifford_eats_per_two_days final_cakes = 6 :=
by {
  sorry
}

end julia_baking_days_l218_218138


namespace points_on_ellipse_satisfying_dot_product_l218_218739

theorem points_on_ellipse_satisfying_dot_product :
  ∃ P1 P2 : ℝ × ℝ,
    P1 = (0, 3) ∧ P2 = (0, -3) ∧
    ∀ P : ℝ × ℝ, 
    (P ∈ ({p : ℝ × ℝ | (p.1 / 5)^2 + (p.2 / 3)^2 = 1}) → 
     ((P.1 - (-4)) * (P.1 - 4) + P.2^2 = -7) →
     (P = P1 ∨ P = P2))
:=
sorry

end points_on_ellipse_satisfying_dot_product_l218_218739


namespace find_p_plus_q_l218_218250

noncomputable def calculate_p_plus_q (DE EF FD WX : ℕ) (Area : ℕ → ℝ) : ℕ :=
  let s := (DE + EF + FD) / 2
  let triangle_area := (Real.sqrt (s * (s - DE) * (s - EF) * (s - FD))) / 2
  let delta := triangle_area / (225 * WX)
  let gcd := Nat.gcd 41 225
  let p := 41 / gcd
  let q := 225 / gcd
  p + q

theorem find_p_plus_q : calculate_p_plus_q 13 30 19 15 (fun θ => 30 * θ - (41 / 225) * θ^2) = 266 := by
  sorry

end find_p_plus_q_l218_218250


namespace max_value_quadratic_expression_l218_218478

theorem max_value_quadratic_expression : ∃ x : ℝ, -3 * x^2 + 18 * x - 5 ≤ 22 ∧ ∀ y : ℝ, -3 * y^2 + 18 * y - 5 ≤ 22 := 
by 
  sorry

end max_value_quadratic_expression_l218_218478


namespace rachel_total_time_l218_218710

-- Define the conditions
def num_chairs : ℕ := 20
def num_tables : ℕ := 8
def time_per_piece : ℕ := 6

-- Proof statement
theorem rachel_total_time : (num_chairs + num_tables) * time_per_piece = 168 := by
  sorry

end rachel_total_time_l218_218710


namespace smallest_n_satisfying_ratio_l218_218698

-- Definitions and conditions from problem
def sum_first_n_odd_numbers_starting_from_3 (n : ℕ) : ℕ := n^2 + 2 * n
def sum_first_n_even_numbers (n : ℕ) : ℕ := n * (n + 1)

theorem smallest_n_satisfying_ratio :
  ∃ n : ℕ, n > 0 ∧ (sum_first_n_odd_numbers_starting_from_3 n : ℚ) / (sum_first_n_even_numbers n : ℚ) = 49 / 50 ∧ n = 51 :=
by
  use 51
  exact sorry

end smallest_n_satisfying_ratio_l218_218698


namespace cuboid_properties_l218_218758

-- Given definitions from conditions
variables (l w h : ℝ)
variables (h_edge_length : 4 * (l + w + h) = 72)
variables (h_ratio : l / w = 3 / 2 ∧ w / h = 2 / 1)

-- Define the surface area and volume based on the given conditions
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)
def volume (l w h : ℝ) : ℝ := l * w * h

-- Theorem statement
theorem cuboid_properties :
  surface_area l w h = 198 ∧ volume l w h = 162 :=
by
  -- Code to provide the proof goes here
  sorry

end cuboid_properties_l218_218758


namespace beret_count_l218_218018

/-- James can make a beret from 3 spools of yarn. 
    He has 12 spools of red yarn, 15 spools of black yarn, and 6 spools of blue yarn.
    Prove that he can make 11 berets in total. -/
theorem beret_count (red_yarn : ℕ) (black_yarn : ℕ) (blue_yarn : ℕ) (spools_per_beret : ℕ) 
  (total_yarn : ℕ) (num_berets : ℕ) (h1 : red_yarn = 12) (h2 : black_yarn = 15) (h3 : blue_yarn = 6)
  (h4 : spools_per_beret = 3) (h5 : total_yarn = red_yarn + black_yarn + blue_yarn) 
  (h6 : num_berets = total_yarn / spools_per_beret) : 
  num_berets = 11 :=
by sorry

end beret_count_l218_218018


namespace gcd_of_gx_and_x_l218_218998

theorem gcd_of_gx_and_x (x : ℕ) (h : 7200 ∣ x) : Nat.gcd ((5 * x + 3) * (11 * x + 2) * (17 * x + 5) * (4 * x + 7)) x = 30 := 
by 
  sorry

end gcd_of_gx_and_x_l218_218998


namespace division_remainder_l218_218646

noncomputable def remainder (p q : Polynomial ℝ) : Polynomial ℝ :=
  p % q

theorem division_remainder :
  remainder (Polynomial.X ^ 3) (Polynomial.X ^ 2 + 7 * Polynomial.X + 2) = 47 * Polynomial.X + 14 :=
by
  sorry

end division_remainder_l218_218646


namespace gcd_lcm_sum_l218_218393

theorem gcd_lcm_sum (a b : ℕ) (ha : a = 45) (hb : b = 4050) :
  Nat.gcd a b + Nat.lcm a b = 4095 := by
  sorry

end gcd_lcm_sum_l218_218393


namespace two_bedroom_units_l218_218577

theorem two_bedroom_units (x y : ℕ) (h1 : x + y = 12) (h2 : 360 * x + 450 * y = 4950) : y = 7 :=
by
  sorry

end two_bedroom_units_l218_218577


namespace convert_to_rectangular_form_l218_218106

theorem convert_to_rectangular_form :
  2 * Real.sqrt 3 * Complex.exp (13 * Real.pi * Complex.I / 6) = 3 + Complex.I * Real.sqrt 3 :=
by
  sorry

end convert_to_rectangular_form_l218_218106


namespace meena_sold_to_stone_l218_218342

def total_cookies_baked : ℕ := 5 * 12
def cookies_bought_brock : ℕ := 7
def cookies_bought_katy : ℕ := 2 * cookies_bought_brock
def cookies_left : ℕ := 15
def cookies_sold_total : ℕ := total_cookies_baked - cookies_left
def cookies_bought_friends : ℕ := cookies_bought_brock + cookies_bought_katy
def cookies_sold_stone : ℕ := cookies_sold_total - cookies_bought_friends
def dozens_sold_stone : ℕ := cookies_sold_stone / 12

theorem meena_sold_to_stone : dozens_sold_stone = 2 := by
  sorry

end meena_sold_to_stone_l218_218342


namespace quadratic_distinct_real_roots_range_quadratic_root_product_value_l218_218694

theorem quadratic_distinct_real_roots_range (k : ℝ) :
  (∀ x : ℝ, (x^2 + (2*k + 1)*x + k^2 + 1 = 0) → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x - x₁) * (x - x₂) = 0) 
  → k > 3 / 4 :=
sorry

theorem quadratic_root_product_value (k : ℝ) :
  (∀ x : ℝ, (x^2 + (2*k + 1)*x + k^2 + 1 = 0) → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x - x₁) * (x - x₂) = 0) 
  → (∀ x₁ x₂ : ℝ, (x^2 + (2*k + 1)*x + k^2 + 1 = 0) → x₁ * x₂ = 5) 
  → k = 2 :=
sorry

end quadratic_distinct_real_roots_range_quadratic_root_product_value_l218_218694


namespace inequality_inequality_always_holds_l218_218098

theorem inequality_inequality_always_holds (x y : ℝ) (h : x > y) : |x| > y :=
sorry

end inequality_inequality_always_holds_l218_218098


namespace walking_speed_10_mph_l218_218491

theorem walking_speed_10_mph 
  (total_minutes : ℕ)
  (distance : ℕ)
  (rest_per_segment : ℕ)
  (rest_time : ℕ)
  (segments : ℕ)
  (walk_time : ℕ)
  (walk_time_hours : ℕ) :
  total_minutes = 328 → 
  distance = 50 → 
  rest_per_segment = 7 → 
  segments = 4 →
  rest_time = segments * rest_per_segment →
  walk_time = total_minutes - rest_time →
  walk_time_hours = walk_time / 60 →
  distance / walk_time_hours = 10 :=
by
  sorry

end walking_speed_10_mph_l218_218491


namespace system_of_equations_solution_l218_218272

theorem system_of_equations_solution :
  ∀ (x y z : ℝ),
  4 * x + 2 * y + z = 20 →
  x + 4 * y + 2 * z = 26 →
  2 * x + y + 4 * z = 28 →
  20 * x^2 + 24 * x * y + 20 * y^2 + 12 * z^2 = 500 :=
by
  intros x y z h1 h2 h3
  sorry

end system_of_equations_solution_l218_218272


namespace total_buttons_l218_218934

-- Define the conditions
def shirts_per_kid : Nat := 3
def number_of_kids : Nat := 3
def buttons_per_shirt : Nat := 7

-- Define the statement to prove
theorem total_buttons : shirts_per_kid * number_of_kids * buttons_per_shirt = 63 := by
  sorry

end total_buttons_l218_218934


namespace number_exceeds_25_percent_by_150_l218_218722

theorem number_exceeds_25_percent_by_150 (x : ℝ) : (0.25 * x + 150 = x) → x = 200 :=
by
  sorry

end number_exceeds_25_percent_by_150_l218_218722


namespace forgot_to_take_capsules_l218_218778

theorem forgot_to_take_capsules (total_days : ℕ) (days_taken : ℕ) 
  (h1 : total_days = 31) 
  (h2 : days_taken = 29) : 
  total_days - days_taken = 2 := 
by 
  sorry

end forgot_to_take_capsules_l218_218778


namespace math_problem_l218_218737

theorem math_problem
  (p q r s : ℕ)
  (hpq : p^3 = q^2)
  (hrs : r^4 = s^3)
  (hrp : r - p = 25) :
  s - q = 73 := by
  sorry

end math_problem_l218_218737


namespace sum_of_coefficients_proof_l218_218817

-- Problem statement: Define the expressions and prove the sum of the coefficients
def expr1 (c : ℝ) : ℝ := -(3 - c) * (c + 2 * (3 - c))
def expanded_form (c : ℝ) : ℝ := -c^2 + 9 * c - 18
def sum_of_coefficients (p : ℝ) := -1 + 9 - 18

theorem sum_of_coefficients_proof (c : ℝ) : sum_of_coefficients (expr1 c) = -10 := by
  sorry

end sum_of_coefficients_proof_l218_218817


namespace gcf_90_108_l218_218904

-- Given two integers 90 and 108
def a : ℕ := 90
def b : ℕ := 108

-- Question: What is the greatest common factor (GCF) of 90 and 108?
theorem gcf_90_108 : Nat.gcd a b = 18 :=
by {
  sorry
}

end gcf_90_108_l218_218904


namespace calculate_interest_rate_l218_218581

theorem calculate_interest_rate
  (total_investment : ℝ)
  (invested_at_eleven_percent : ℝ)
  (total_interest : ℝ)
  (interest_rate_first_type : ℝ) :
  total_investment = 100000 ∧ 
  invested_at_eleven_percent = 30000 ∧ 
  total_interest = 9.6 → 
  interest_rate_first_type = 9 :=
by
  intros
  sorry

end calculate_interest_rate_l218_218581


namespace hall_ratio_l218_218923

variable (w l : ℝ)

theorem hall_ratio
  (h1 : w * l = 200)
  (h2 : l - w = 10) :
  w / l = 1 / 2 := 
by
  sorry

end hall_ratio_l218_218923


namespace mass_percentage_Al_in_Al2CO33_l218_218213
-- Importing the required libraries

-- Define the necessary constants for molar masses
def molar_mass_Al : ℝ := 26.98
def molar_mass_C : ℝ := 12.01
def molar_mass_O : ℝ := 16.00
def molar_mass_Al2CO33 : ℝ := 2 * molar_mass_Al + 3 * molar_mass_C + 9 * molar_mass_O
def mass_Al_in_Al2CO33 : ℝ := 2 * molar_mass_Al

-- Define the main theorem to prove the mass percentage of Al in Al2(CO3)3
theorem mass_percentage_Al_in_Al2CO33 :
  (mass_Al_in_Al2CO33 / molar_mass_Al2CO33) * 100 = 23.05 :=
by
  simp [molar_mass_Al, molar_mass_C, molar_mass_O, molar_mass_Al2CO33, mass_Al_in_Al2CO33]
  -- Calculation result based on given molar masses
  sorry

end mass_percentage_Al_in_Al2CO33_l218_218213


namespace general_form_equation_l218_218726

theorem general_form_equation (x : ℝ) : 
  x * (2 * x - 1) = 5 * (x + 3) ↔ 2 * x^2 - 6 * x - 15 = 0 := 
by 
  sorry

end general_form_equation_l218_218726


namespace abs_sum_lt_abs_l218_218756

theorem abs_sum_lt_abs (a b : ℝ) (h : a * b < 0) : |a + b| < |a| + |b| :=
sorry

end abs_sum_lt_abs_l218_218756


namespace find_g_3_8_l218_218925

variable (g : ℝ → ℝ)
variable (x : ℝ)

-- Conditions
axiom g0 : g 0 = 0
axiom monotonicity (x y : ℝ) : 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y
axiom symmetry (x : ℝ) : 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x
axiom scaling (x : ℝ) : 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3

-- Statement to prove
theorem find_g_3_8 : g (3 / 8) = 2 / 9 := 
sorry

end find_g_3_8_l218_218925


namespace angle_same_terminal_side_210_l218_218042

theorem angle_same_terminal_side_210 (n : ℤ) : 
  ∃ k : ℤ, 210 = -510 + k * 360 ∧ 0 ≤ 210 ∧ 210 < 360 :=
by
  use 2
  -- proof steps will go here
  sorry

end angle_same_terminal_side_210_l218_218042


namespace find_c_share_l218_218637

theorem find_c_share (a b c : ℕ) 
  (h1 : a + b + c = 1760)
  (h2 : ∃ x : ℕ, a = 3 * x ∧ b = 4 * x ∧ c = 5 * x)
  (h3 : 6 * a = 8 * b ∧ 8 * b = 20 * c) : 
  c = 250 :=
by
  sorry

end find_c_share_l218_218637


namespace admission_charge_l218_218374

variable (A : ℝ) -- Admission charge in dollars
variable (tour_charge : ℝ)
variable (group1_size : ℕ)
variable (group2_size : ℕ)
variable (total_earnings : ℝ)

-- Given conditions
axiom h1 : tour_charge = 6
axiom h2 : group1_size = 10
axiom h3 : group2_size = 5
axiom h4 : total_earnings = 240
axiom h5 : (group1_size * A + group1_size * tour_charge) + (group2_size * A) = total_earnings

theorem admission_charge : A = 12 :=
by
  sorry

end admission_charge_l218_218374


namespace cows_count_l218_218151

theorem cows_count (D C : ℕ) (h1 : 2 * (D + C) + 32 = 2 * D + 4 * C) : C = 16 :=
by
  sorry

end cows_count_l218_218151


namespace sum_of_digits_of_m_l218_218176

-- Define the logarithms and intermediate expressions
noncomputable def log_b (b x : ℝ) := Real.log x / Real.log b

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem sum_of_digits_of_m :
  ∃ m : ℕ, log_b 3 (log_b 81 m) = log_b 9 (log_b 9 m) ∧ sum_of_digits m = 10 := 
by
  sorry

end sum_of_digits_of_m_l218_218176


namespace number_of_divisors_of_3003_l218_218366

theorem number_of_divisors_of_3003 :
  ∃ d, d = 16 ∧ 
  (3003 = 3^1 * 7^1 * 11^1 * 13^1) →
  d = (1 + 1) * (1 + 1) * (1 + 1) * (1 + 1) := 
by 
  sorry

end number_of_divisors_of_3003_l218_218366


namespace angle_B_l218_218699

/-- 
  Given that the area of triangle ABC is (sqrt 3 / 2) 
  and the dot product of vectors AB and BC is 3, 
  prove that the measure of angle B is 5π/6. 
--/
theorem angle_B (A B C : ℝ) (a c : ℝ) (h1 : 0 ≤ B ∧ B ≤ π)
  (h_area : (1 / 2) * a * c * (Real.sin B) = (Real.sqrt 3 / 2))
  (h_dot : a * c * (Real.cos B) = -3) :
  B = 5 * Real.pi / 6 :=
sorry

end angle_B_l218_218699


namespace n_cubed_plus_5n_divisible_by_6_l218_218130

theorem n_cubed_plus_5n_divisible_by_6 (n : ℕ) : ∃ k : ℤ, n^3 + 5 * n = 6 * k :=
by
  sorry

end n_cubed_plus_5n_divisible_by_6_l218_218130


namespace percentage_increase_mario_salary_is_zero_l218_218002

variable (M : ℝ) -- Mario's salary last year
variable (P : ℝ) -- Percentage increase in Mario's salary

-- Condition 1: Mario's salary increased to $4000 this year
def mario_salary_increase (M P : ℝ) : Prop :=
  M * (1 + P / 100) = 4000 

-- Condition 2: Bob's salary last year was 3 times Mario's salary this year
def bob_salary_last_year (M : ℝ) : Prop :=
  3 * 4000 = 12000 

-- Condition 3: Bob's current salary is 20% more than his salary last year
def bob_current_salary : Prop :=
  12000 * 1.2 = 14400

-- Theorem : The percentage increase in Mario's salary is 0%
theorem percentage_increase_mario_salary_is_zero
  (h1 : mario_salary_increase M P)
  (h2 : bob_salary_last_year M)
  (h3 : bob_current_salary) : 
  P = 0 := 
sorry

end percentage_increase_mario_salary_is_zero_l218_218002


namespace player_winning_strategy_l218_218651

-- Define the game conditions
def Sn (n : ℕ) : Type := Equiv.Perm (Fin n)

def game_condition (n : ℕ) : Prop :=
  n > 1 ∧ (∀ G : Set (Sn n), ∃ x : Sn n, x ∈ G → G ≠ (Set.univ : Set (Sn n)))

-- Statement of the proof problem
theorem player_winning_strategy (n : ℕ) (hn : n > 1) : 
  ((n = 2 ∨ n = 3) → (∃ strategyA : Sn n → (Sn n → Prop), ∀ x : Sn n, strategyA x x)) ∧ 
  ((n ≥ 4 ∧ n % 2 = 1) → (∃ strategyB : Sn n → (Sn n → Prop), ∀ x : Sn n, strategyB x x)) :=
by
  sorry

end player_winning_strategy_l218_218651


namespace crayons_per_box_l218_218839

theorem crayons_per_box (total_crayons : ℝ) (total_boxes : ℝ) (h1 : total_crayons = 7.0) (h2 : total_boxes = 1.4) : total_crayons / total_boxes = 5 :=
by
  sorry

end crayons_per_box_l218_218839


namespace find_b_l218_218760

noncomputable def P (x a b c : ℝ) : ℝ := 2 * x^3 + a * x^2 + b * x + c

theorem find_b (a b c : ℝ) (h1: P 0 a b c = 12)
  (h2: (-c / 2) * 1 = -6)
  (h3: (2 + a + b + c) = -6)
  (h4: a + b + 14 = -6) : b = -56 :=
sorry

end find_b_l218_218760


namespace range_of_m_l218_218504

theorem range_of_m (m : ℝ) :
  (∀ x1 x2 : ℝ, 0 ≤ x1 ∧ x1 ≤ 2 ∧ 0 ≤ x2 ∧ x2 ≤ 2 → |((x2^2 - m * x2) - (x1^2 - m * x1))| ≤ 9) →
  -5 / 2 ≤ m ∧ m ≤ 13 / 2 :=
sorry

end range_of_m_l218_218504


namespace sequence_first_last_four_equal_l218_218019

theorem sequence_first_last_four_equal (S : List ℕ) (n : ℕ)
  (hS : S.length = n)
  (h_max : ∀ T : List ℕ, (∀ i j : ℕ, i < j → i ≤ n-5 → j ≤ n-5 → 
                        (S.drop i).take 5 ≠ (S.drop j).take 5) → T.length ≤ n)
  (h_distinct : ∀ i j : ℕ, i < j → i ≤ n-5 → j ≤ n-5 → 
                (S.drop i).take 5 ≠ (S.drop j).take 5) :
  (S.take 4 = S.drop (n-4)) :=
by
  sorry

end sequence_first_last_four_equal_l218_218019


namespace dan_violet_marbles_l218_218784

def InitMarbles : ℕ := 128
def MarblesGivenMary : ℕ := 24
def MarblesGivenPeter : ℕ := 16
def MarblesReceived : ℕ := 10

def FinalMarbles : ℕ := InitMarbles - MarblesGivenMary - MarblesGivenPeter + MarblesReceived

theorem dan_violet_marbles : FinalMarbles = 98 := 
by 
  sorry

end dan_violet_marbles_l218_218784


namespace geometric_a1_value_l218_218193

noncomputable def geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ :=
  a1 * q ^ (n - 1)

theorem geometric_a1_value (a3 a5 : ℝ) (q : ℝ) : 
  a3 = geometric_sequence a1 q 3 →
  a5 = geometric_sequence a1 q 5 →
  a1 = 2 :=
by
  sorry

end geometric_a1_value_l218_218193


namespace clothing_price_l218_218490

theorem clothing_price
  (total_spent : ℕ)
  (num_pieces : ℕ)
  (price_piece_1 : ℕ)
  (price_piece_2 : ℕ)
  (num_remaining_pieces : ℕ)
  (total_remaining_pieces_price : ℕ)
  (price_remaining_piece : ℕ) :
  total_spent = 610 →
  num_pieces = 7 →
  price_piece_1 = 49 →
  price_piece_2 = 81 →
  num_remaining_pieces = 5 →
  total_spent = price_piece_1 + price_piece_2 + total_remaining_pieces_price →
  total_remaining_pieces_price = price_remaining_piece * num_remaining_pieces →
  price_remaining_piece = 96 :=
by
  intros h_total_spent h_num_pieces h_price_piece_1 h_price_piece_2 h_num_remaining_pieces h_total_remaining_price h_price_remaining_piece
  sorry

end clothing_price_l218_218490


namespace jessica_watermelons_l218_218914

theorem jessica_watermelons (original : ℕ) (eaten : ℕ) (remaining : ℕ) 
    (h1 : original = 35) 
    (h2 : eaten = 27) 
    (h3 : remaining = original - eaten) : 
  remaining = 8 := 
by {
    -- This is where the proof would go
    sorry
}

end jessica_watermelons_l218_218914


namespace benny_kids_l218_218859

theorem benny_kids (total_money : ℕ) (cost_per_apple : ℕ) (apples_per_kid : ℕ) (total_apples : ℕ) (kids : ℕ) :
  total_money = 360 →
  cost_per_apple = 4 →
  apples_per_kid = 5 →
  total_apples = total_money / cost_per_apple →
  kids = total_apples / apples_per_kid →
  kids = 18 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end benny_kids_l218_218859


namespace kamal_average_marks_l218_218449

theorem kamal_average_marks :
  (76 / 120) * 0.2 + 
  (60 / 110) * 0.25 + 
  (82 / 100) * 0.15 + 
  (67 / 90) * 0.2 + 
  (85 / 100) * 0.15 + 
  (78 / 95) * 0.05 = 0.70345 :=
by 
  sorry

end kamal_average_marks_l218_218449


namespace tom_splitting_slices_l218_218199

theorem tom_splitting_slices :
  ∃ S : ℕ, (∃ t, t = 3/8 * S) → 
          (∃ u, u = 1/2 * (S - t)) → 
          (∃ v, v = u + t) → 
          (v = 5) → 
          (S / 2 = 8) :=
sorry

end tom_splitting_slices_l218_218199


namespace lcm_of_ratio_hcf_l218_218633

theorem lcm_of_ratio_hcf {a b : ℕ} (ratioCond : a = 14 * 28) (ratioCond2 : b = 21 * 28) (hcfCond : Nat.gcd a b = 28) : Nat.lcm a b = 1176 := by
  sorry

end lcm_of_ratio_hcf_l218_218633


namespace two_digit_multiples_of_6_and_9_l218_218037

theorem two_digit_multiples_of_6_and_9 : ∃ n : ℕ, n = 5 ∧ (∀ k : ℤ, 10 ≤ k ∧ k < 100 ∧ (k % 6 = 0) ∧ (k % 9 = 0) → 
    k = 18 ∨ k = 36 ∨ k = 54 ∨ k = 72 ∨ k = 90) := 
sorry

end two_digit_multiples_of_6_and_9_l218_218037


namespace inequality_solution_l218_218936

theorem inequality_solution (x y : ℝ) : y - x < abs x ↔ y < 0 ∨ y < 2 * x :=
by sorry

end inequality_solution_l218_218936


namespace P_and_S_could_not_be_fourth_l218_218868

-- Define the relationships between the runners using given conditions
variables (P Q R S T U : ℕ)

axiom P_beats_Q : P < Q
axiom Q_beats_R : Q < R
axiom R_beats_S : R < S
axiom T_after_P_before_R : P < T ∧ T < R
axiom U_before_R_after_S : S < U ∧ U < R

-- Prove that P and S could not be fourth
theorem P_and_S_could_not_be_fourth : ¬((Q < U ∧ U < P) ∨ (Q > S ∧ S < P)) :=
by sorry

end P_and_S_could_not_be_fourth_l218_218868


namespace smallest_next_divisor_l218_218565

def is_even (n : ℕ) : Prop := n % 2 = 0

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def is_divisor (a b : ℕ) : Prop := b % a = 0

theorem smallest_next_divisor (m : ℕ) (h_even : is_even m)
  (h_four_digit : is_four_digit m)
  (h_div_437 : is_divisor 437 m) :
  ∃ next_div : ℕ, next_div > 437 ∧ is_divisor next_div m ∧ 
  ∀ d, d > 437 ∧ is_divisor d m → next_div ≤ d :=
sorry

end smallest_next_divisor_l218_218565


namespace sum_of_n_terms_l218_218697

noncomputable def S : ℕ → ℕ :=
sorry -- We define S, but its exact form is not used in the statement directly

noncomputable def a : ℕ → ℕ := 
sorry -- We define a, but its exact form is not used in the statement directly

-- Conditions
axiom S3_eq : S 3 = 1
axiom a_rec : ∀ n : ℕ, 0 < n → a (n + 3) = 2 * (a n)

-- Proof problem
theorem sum_of_n_terms : S 2019 = 2^673 - 1 :=
sorry

end sum_of_n_terms_l218_218697


namespace brooke_kent_ratio_l218_218559

theorem brooke_kent_ratio :
  ∀ (alison brooke brittany kent : ℕ),
  (kent = 1000) →
  (alison = 4000) →
  (alison = brittany / 2) →
  (brittany = 4 * brooke) →
  brooke / kent = 2 :=
by
  intros alison brooke brittany kent kent_val alison_val alison_brittany brittany_brooke
  sorry

end brooke_kent_ratio_l218_218559


namespace terminating_decimal_multiples_l218_218625

theorem terminating_decimal_multiples :
  (∃ n : ℕ, 20 = n ∧ ∀ m, 1 ≤ m ∧ m ≤ 180 → 
  (∃ k : ℕ, m = 9 * k)) :=
by
  sorry

end terminating_decimal_multiples_l218_218625


namespace balls_in_boxes_l218_218773

theorem balls_in_boxes : (3^4 = 81) :=
by
  sorry

end balls_in_boxes_l218_218773


namespace total_investment_sum_l218_218638

-- Definitions of the problem
variable (Raghu Trishul Vishal : ℕ)
variable (h1 : Raghu = 2000)
variable (h2 : Trishul = Nat.div (Raghu * 9) 10)
variable (h3 : Vishal = Nat.div (Trishul * 11) 10)

-- The theorem to prove
theorem total_investment_sum :
  Vishal + Trishul + Raghu = 5780 :=
by
  sorry

end total_investment_sum_l218_218638


namespace arithmetic_progression_a_eq_1_l218_218318

theorem arithmetic_progression_a_eq_1 
  (a : ℝ) 
  (h1 : 6 + 2 * a - 1 = 10 + 5 * a - (6 + 2 * a)) : 
  a = 1 :=
by
  sorry

end arithmetic_progression_a_eq_1_l218_218318


namespace exponential_equality_l218_218116

theorem exponential_equality (n : ℕ) (h : 4 ^ n = 64 ^ 2) : n = 6 :=
  sorry

end exponential_equality_l218_218116


namespace vikki_take_home_pay_l218_218632

-- Define the conditions
def hours_worked : ℕ := 42
def pay_rate : ℝ := 10
def tax_rate : ℝ := 0.20
def insurance_rate : ℝ := 0.05
def union_dues : ℝ := 5

-- Define the gross earnings function
def gross_earnings (hours_worked : ℕ) (pay_rate : ℝ) : ℝ := hours_worked * pay_rate

-- Define the deductions functions
def tax_deduction (gross : ℝ) (rate : ℝ) : ℝ := gross * rate
def insurance_deduction (gross : ℝ) (rate : ℝ) : ℝ := gross * rate
def total_deductions (tax : ℝ) (insurance : ℝ) (dues : ℝ) : ℝ := tax + insurance + dues

-- Define the take-home pay function
def take_home_pay (gross : ℝ) (deductions : ℝ) : ℝ := gross - deductions

theorem vikki_take_home_pay :
  take_home_pay (gross_earnings hours_worked pay_rate)
    (total_deductions (tax_deduction (gross_earnings hours_worked pay_rate) tax_rate)
                      (insurance_deduction (gross_earnings hours_worked pay_rate) insurance_rate)
                      union_dues) = 310 :=
by
  sorry

end vikki_take_home_pay_l218_218632


namespace area_of_EFGH_l218_218356

variables (EF FG EH HG EG : ℝ)
variables (distEFGH : EF ≠ HG ∧ EG = 5 ∧ EF^2 + FG^2 = 25 ∧ EH^2 + HG^2 = 25)

theorem area_of_EFGH : 
  ∃ EF FG EH HG : ℕ, EF ≠ HG ∧ EG = 5 
  ∧ EF^2 + FG^2 = 25 
  ∧ EH^2 + HG^2 = 25 
  ∧ EF * FG / 2 + EH * HG / 2 = 12 :=
by { sorry }

end area_of_EFGH_l218_218356


namespace student_opinion_change_l218_218967

theorem student_opinion_change (init_enjoy : ℕ) (init_not_enjoy : ℕ)
                               (final_enjoy : ℕ) (final_not_enjoy : ℕ) :
  init_enjoy = 40 ∧ init_not_enjoy = 60 ∧ final_enjoy = 75 ∧ final_not_enjoy = 25 →
  ∃ y_min y_max : ℕ, 
    y_min = 35 ∧ y_max = 75 ∧ (y_max - y_min = 40) :=
by
  sorry

end student_opinion_change_l218_218967


namespace groceries_spent_l218_218752

/-- Defining parameters from the conditions provided -/
def rent : ℝ := 5000
def milk : ℝ := 1500
def education : ℝ := 2500
def petrol : ℝ := 2000
def miscellaneous : ℝ := 700
def savings_rate : ℝ := 0.10
def savings : ℝ := 1800

/-- Adding an assertion for the total spent on groceries -/
def groceries : ℝ := 4500

theorem groceries_spent (total_salary total_expenses : ℝ) :
  total_salary = savings / savings_rate →
  total_expenses = rent + milk + education + petrol + miscellaneous →
  groceries = total_salary - (total_expenses + savings) :=
by
  intros h_salary h_expenses
  sorry

end groceries_spent_l218_218752


namespace cost_per_night_l218_218548

variable (x : ℕ)

theorem cost_per_night (h : 3 * x - 100 = 650) : x = 250 :=
sorry

end cost_per_night_l218_218548


namespace acute_triangle_probability_l218_218173

open Finset

noncomputable def isAcuteTriangleProb (n : ℕ) : Prop :=
  ∃ k : ℕ, (n = 2 * k ∧ (3 * (k - 2)) / (2 * (2 * k - 1)) = 93 / 125) ∨ (n = 2 * k + 1 ∧ (3 * (k - 1)) / (2 * (2 * k - 1)) = 93 / 125)

theorem acute_triangle_probability (n : ℕ) : isAcuteTriangleProb n → n = 376 ∨ n = 127 :=
by
  sorry

end acute_triangle_probability_l218_218173


namespace distance_from_origin_12_5_l218_218080

def distance_from_origin (x y : ℕ) : ℕ := 
  Int.natAbs (Nat.sqrt (x * x + y * y))

theorem distance_from_origin_12_5 : distance_from_origin 12 5 = 13 := by
  sorry

end distance_from_origin_12_5_l218_218080


namespace geometric_sequence_properties_l218_218654

/-- Given {a_n} is a geometric sequence, a_1 = 1 and a_4 = 1/8, 
the common ratio q of {a_n} is 1/2 and the sum of the first 5 terms of {1/a_n} is 31. -/
theorem geometric_sequence_properties (a : ℕ → ℝ) (h1 : a 1 = 1) (h4 : a 4 = 1 / 8) : 
  (∃ q : ℝ, (∀ n : ℕ, a n = a 1 * q ^ (n - 1)) ∧ q = 1 / 2) ∧ 
  (∃ S : ℝ, S = 31 ∧ S = (1 - 2^5) / (1 - 2)) :=
by
  -- Skipping the proof
  sorry

end geometric_sequence_properties_l218_218654


namespace sample_size_correct_l218_218823

def total_students (freshmen sophomores juniors : ℕ) : ℕ :=
  freshmen + sophomores + juniors

def sample_size (total : ℕ) (prob : ℝ) : ℝ :=
  total * prob

theorem sample_size_correct (f : ℕ) (s : ℕ) (j : ℕ) (p : ℝ) (h_f : f = 400) (h_s : s = 320) (h_j : j = 280) (h_p : p = 0.2) :
  sample_size (total_students f s j) p = 200 :=
by
  sorry

end sample_size_correct_l218_218823


namespace solution_set_f_pos_l218_218781

noncomputable def f : ℝ → ℝ := sorry -- Definition of the function f(x)

-- Conditions
axiom h1 : ∀ x, f (-x) = -f x     -- f(x) is odd
axiom h2 : f 2 = 0                -- f(2) = 0
axiom h3 : ∀ x > 0, 2 * f x + x * (deriv f x) > 0 -- 2f(x) + xf'(x) > 0 for x > 0

-- Theorem to prove
theorem solution_set_f_pos : { x : ℝ | f x > 0 } = { x : ℝ | x > 2 ∨ (-2 < x ∧ x < 0) } :=
sorry

end solution_set_f_pos_l218_218781


namespace coffee_ratio_l218_218270

/-- Define the conditions -/
def initial_coffees_per_day := 4
def initial_price_per_coffee := 2
def price_increase_percentage := 50 / 100
def savings_per_day := 2

/-- Define the price calculations -/
def new_price_per_coffee := initial_price_per_coffee + (initial_price_per_coffee * price_increase_percentage)
def initial_daily_cost := initial_coffees_per_day * initial_price_per_coffee
def new_daily_cost := initial_daily_cost - savings_per_day
def new_coffees_per_day := new_daily_cost / new_price_per_coffee

/-- Prove the ratio -/
theorem coffee_ratio : (new_coffees_per_day / initial_coffees_per_day) = (1 : ℝ) / (2 : ℝ) :=
  by sorry

end coffee_ratio_l218_218270


namespace scientific_notation_256000_l218_218982

theorem scientific_notation_256000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 256000 = a * 10^n ∧ a = 2.56 ∧ n = 5 :=
by
  sorry

end scientific_notation_256000_l218_218982


namespace sum_of_ages_l218_218657

-- Define the ages of Maggie, Juliet, and Ralph
def maggie_age : ℕ := by
  let juliet_age := 10
  let maggie_age := juliet_age - 3
  exact maggie_age

def ralph_age : ℕ := by
  let juliet_age := 10
  let ralph_age := juliet_age + 2
  exact ralph_age

-- The main theorem: The sum of Maggie's and Ralph's ages
theorem sum_of_ages : maggie_age + ralph_age = 19 := by
  sorry

end sum_of_ages_l218_218657


namespace cyclic_quadrilateral_angle_D_l218_218999

theorem cyclic_quadrilateral_angle_D (A B C D : ℝ) (h₁ : A + B + C + D = 360) (h₂ : ∃ x, A = 3 * x ∧ B = 4 * x ∧ C = 6 * x) :
  D = 100 :=
by
  sorry

end cyclic_quadrilateral_angle_D_l218_218999


namespace lateral_surface_area_of_cone_l218_218267

theorem lateral_surface_area_of_cone (r l : ℝ) (h₁ : r = 3) (h₂ : l = 5) :
  π * r * l = 15 * π :=
by sorry

end lateral_surface_area_of_cone_l218_218267


namespace find_other_number_l218_218879

theorem find_other_number (a b lcm hcf : ℕ) (h_lcm : lcm = 2310) (h_hcf : hcf = 61) (h_first_number : a = 210) :
  a * b = lcm * hcf → b = 671 :=
by 
  -- setup
  sorry

end find_other_number_l218_218879


namespace combined_weight_of_three_parcels_l218_218819

theorem combined_weight_of_three_parcels (x y z : ℕ)
  (h1 : x + y = 112) (h2 : y + z = 146) (h3 : z + x = 132) :
  x + y + z = 195 :=
by
  sorry

end combined_weight_of_three_parcels_l218_218819


namespace cos_squared_alpha_plus_pi_over_4_correct_l218_218112

variable (α : ℝ)
axiom sin_two_alpha : Real.sin (2 * α) = 2 / 3

theorem cos_squared_alpha_plus_pi_over_4_correct :
  Real.cos (α + Real.pi / 4) ^ 2 = 1 / 6 :=
by
  sorry

end cos_squared_alpha_plus_pi_over_4_correct_l218_218112


namespace none_of_these_l218_218915

variables (a b c d e f : Prop)

-- Given conditions
axiom condition1 : a > b → c > d
axiom condition2 : c < d → e > f

-- Invalid conclusions
theorem none_of_these :
  ¬(a < b → e > f) ∧
  ¬(e > f → a < b) ∧
  ¬(e < f → a > b) ∧
  ¬(a > b → e < f) := sorry

end none_of_these_l218_218915


namespace solve_system_of_equations_l218_218969

theorem solve_system_of_equations (x1 x2 x3 x4 x5 y : ℝ) :
  x5 + x2 = y * x1 ∧
  x1 + x3 = y * x2 ∧
  x2 + x4 = y * x3 ∧
  x3 + x5 = y * x4 ∧
  x4 + x1 = y * x5 →
  (y = 2 ∧ x1 = x2 ∧ x2 = x3 ∧ x3 = x4 ∧ x4 = x5) ∨
  (y ≠ 2 ∧ (y^2 + y - 1 ≠ 0 ∧ x1 = 0 ∧ x2 = 0 ∧ x3 = 0 ∧ x4 = 0 ∧ x5 = 0) ∨
  (y^2 + y - 1 = 0 ∧ y = (1 / 2) * (-1 + Real.sqrt 5) ∨ y = (1 / 2) * (-1 - Real.sqrt 5) ∧
    ∃ a b : ℝ, x1 = a ∧ x2 = b ∧ x3 = y * b - a ∧ x4 = - y * (a + b) ∧ x5 = y * a - b))
:=
sorry

end solve_system_of_equations_l218_218969


namespace cube_minus_self_divisible_by_6_l218_218511

theorem cube_minus_self_divisible_by_6 (n : ℕ) : 6 ∣ (n^3 - n) :=
sorry

end cube_minus_self_divisible_by_6_l218_218511


namespace inverse_function_of_f_l218_218340

noncomputable def f (x : ℝ) : ℝ := (3 * x + 1) / x
noncomputable def f_inv (x : ℝ) : ℝ := 1 / (x - 3)

theorem inverse_function_of_f:
  ∀ x : ℝ, x ≠ 3 → f (f_inv x) = x ∧ f_inv (f x) = x := by
sorry

end inverse_function_of_f_l218_218340


namespace problem1_problem2_l218_218645

-- Define sets A and B based on the conditions
def setA (a : ℝ) : Set ℝ := {x : ℝ | a ≤ x ∧ x ≤ a + 3}
def setB : Set ℝ := {x : ℝ | x < -2 ∨ x > 6}

-- Define the two proof problems as Lean statements
theorem problem1 (a : ℝ) : setA a ∩ setB = ∅ ↔ -2 ≤ a ∧ a ≤ 3 := by
  sorry

theorem problem2 (a : ℝ) : setA a ⊆ setB ↔ (a < -5 ∨ a > 6) := by
  sorry

end problem1_problem2_l218_218645


namespace girl_name_correct_l218_218335

-- The Russian alphabet positions as a Lean list
def russianAlphabet : List (ℕ × Char) := [(1, 'А'), (2, 'Б'), (3, 'В'), (4, 'Г'), (5, 'Д'), (6, 'Е'), (7, 'Ё'), 
                                           (8, 'Ж'), (9, 'З'), (10, 'И'), (11, 'Й'), (12, 'К'), (13, 'Л'), 
                                           (14, 'М'), (15, 'Н'), (16, 'О'), (17, 'П'), (18, 'Р'), (19, 'С'), 
                                           (20, 'Т'), (21, 'У'), (22, 'Ф'), (23, 'Х'), (24, 'Ц'), (25, 'Ч'), 
                                           (26, 'Ш'), (27, 'Щ'), (28, 'Ъ'), (29, 'Ы'), (30, 'Ь'), (31, 'Э'), 
                                           (32, 'Ю'), (33, 'Я')]

-- The sequence of numbers representing the girl's name
def nameSequence : ℕ := 2011533

-- The corresponding name derived from the sequence
def derivedName : String := "ТАНЯ"

-- The equivalence proof statement
theorem girl_name_correct : 
  (nameSequence = 2011533 → derivedName = "ТАНЯ") :=
by
  intro h
  sorry

end girl_name_correct_l218_218335


namespace solve_tan_equation_l218_218508

theorem solve_tan_equation (x : ℝ) (k : ℤ) :
  8.456 * (Real.tan x)^2 * (Real.tan (3 * x))^2 * Real.tan (4 * x) = 
  (Real.tan x)^2 - (Real.tan (3 * x))^2 + Real.tan (4 * x) ->
  x = π * k ∨ x = π / 4 * (2 * k + 1) := sorry

end solve_tan_equation_l218_218508


namespace unlock_probability_l218_218740

/--
Xiao Ming set a six-digit passcode for his phone using the numbers 0-9, but he forgot the last digit.
The probability that Xiao Ming can unlock his phone with just one try is 1/10.
-/
theorem unlock_probability (n : ℕ) (h : n ≥ 0 ∧ n ≤ 9) : 
  1 / 10 = 1 / (10 : ℝ) :=
by
  -- Skipping proof
  sorry

end unlock_probability_l218_218740


namespace line_always_passes_fixed_point_l218_218578

theorem line_always_passes_fixed_point (m : ℝ) :
  m * 1 + (1 - m) * 2 + m - 2 = 0 :=
by
  sorry

end line_always_passes_fixed_point_l218_218578


namespace trace_bag_weight_is_two_l218_218166

-- Given the conditions in the problem
def weight_gordon_bag₁ : ℕ := 3
def weight_gordon_bag₂ : ℕ := 7
def num_traces_bag : ℕ := 5

-- Total weight of Gordon's bags is 10
def total_weight_gordon := weight_gordon_bag₁ + weight_gordon_bag₂

-- Trace's bags weight
def total_weight_trace := total_weight_gordon

-- All conditions must imply this equation is true
theorem trace_bag_weight_is_two :
  (num_traces_bag * 2 = total_weight_trace) → (2 = 2) :=
  by
    sorry

end trace_bag_weight_is_two_l218_218166


namespace distance_car_to_stream_l218_218471

theorem distance_car_to_stream (total_distance : ℝ) (stream_to_meadow : ℝ) (meadow_to_campsite : ℝ) (h1 : total_distance = 0.7) (h2 : stream_to_meadow = 0.4) (h3 : meadow_to_campsite = 0.1) :
  (total_distance - (stream_to_meadow + meadow_to_campsite) = 0.2) :=
by
  sorry

end distance_car_to_stream_l218_218471


namespace prob_seven_heads_in_ten_tosses_l218_218269

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  (Nat.choose n k)

noncomputable def probability_of_heads (n k : ℕ) : ℚ :=
  (binomial_coefficient n k) * (0.5^k : ℚ) * (0.5^(n - k) : ℚ)

theorem prob_seven_heads_in_ten_tosses :
  probability_of_heads 10 7 = 15 / 128 :=
by
  sorry

end prob_seven_heads_in_ten_tosses_l218_218269


namespace symmetric_axis_of_quadratic_l218_218186

-- Define the quadratic function
def quadratic_function (x : ℝ) : ℝ := (x - 3) * (x + 5)

-- Prove that the symmetric axis of the quadratic function is the line x = -1
theorem symmetric_axis_of_quadratic : ∀ (x : ℝ), quadratic_function x = (x - 3) * (x + 5) → x = -1 :=
by
  intro x h
  sorry

end symmetric_axis_of_quadratic_l218_218186


namespace evaluate_at_3_l218_218580

def g (x : ℝ) : ℝ := 3 * x^4 - 5 * x^3 + 2 * x^2 + x + 6

theorem evaluate_at_3 : g 3 = 135 := 
  by
  sorry

end evaluate_at_3_l218_218580


namespace workshopA_more_stable_than_B_l218_218996

-- Given data sets for workshops A and B
def workshopA_data := [102, 101, 99, 98, 103, 98, 99]
def workshopB_data := [110, 115, 90, 85, 75, 115, 110]

-- Define stability of a product in terms of the standard deviation or similar metric
def is_more_stable (dataA dataB : List ℕ) : Prop :=
  sorry -- Replace with a definition comparing stability based on a chosen metric, e.g., standard deviation

-- Prove that Workshop A's product is more stable than Workshop B's product
theorem workshopA_more_stable_than_B : is_more_stable workshopA_data workshopB_data :=
  sorry

end workshopA_more_stable_than_B_l218_218996


namespace stones_required_to_pave_hall_l218_218592

theorem stones_required_to_pave_hall :
    let length_hall_m := 36
    let breadth_hall_m := 15
    let length_stone_dm := 3
    let breadth_stone_dm := 5
    let length_hall_dm := length_hall_m * 10
    let breadth_hall_dm := breadth_hall_m * 10
    let area_hall_dm2 := length_hall_dm * breadth_hall_dm
    let area_stone_dm2 := length_stone_dm * breadth_stone_dm
    (area_hall_dm2 / area_stone_dm2) = 3600 :=
by
    -- Definitions
    let length_hall_m := 36
    let breadth_hall_m := 15
    let length_stone_dm := 3
    let breadth_stone_dm := 5

    -- Convert to decimeters
    let length_hall_dm := length_hall_m * 10
    let breadth_hall_dm := breadth_hall_m * 10
    
    -- Calculate areas
    let area_hall_dm2 := length_hall_dm * breadth_hall_dm
    let area_stone_dm2 := length_stone_dm * breadth_stone_dm
    
    -- Calculate number of stones 
    let number_of_stones := area_hall_dm2 / area_stone_dm2

    -- Prove the required number of stones
    have h : number_of_stones = 3600 := sorry
    exact h

end stones_required_to_pave_hall_l218_218592


namespace sum_of_fractions_is_514_l218_218563

theorem sum_of_fractions_is_514 : 
  (1 / (2 * 3) + 1 / (3 * 4) + 1 / (4 * 5) + 1 / (5 * 6) + 1 / (6 * 7)) = 5 / 14 := 
by
  sorry

end sum_of_fractions_is_514_l218_218563


namespace gcd_7654321_6789012_l218_218622

theorem gcd_7654321_6789012 : Nat.gcd 7654321 6789012 = 3 := by
  sorry

end gcd_7654321_6789012_l218_218622


namespace current_speed_l218_218561

-- The main statement of our problem
theorem current_speed (v_with_current v_against_current c man_speed : ℝ) 
  (h1 : v_with_current = man_speed + c) 
  (h2 : v_against_current = man_speed - c) 
  (h_with : v_with_current = 15) 
  (h_against : v_against_current = 9.4) : 
  c = 2.8 :=
by
  sorry

end current_speed_l218_218561


namespace range_of_x_for_fx1_positive_l218_218955

-- Define the conditions
def is_even (f : ℝ → ℝ) := ∀ x, f x = f (-x)
def is_monotonic_decreasing_on_nonneg (f : ℝ → ℝ) := ∀ ⦃x y⦄, 0 ≤ x → x ≤ y → f y ≤ f x
def f_at_2_eq_zero (f : ℝ → ℝ) := f 2 = 0

-- Define the problem statement that needs to be proven
theorem range_of_x_for_fx1_positive (f : ℝ → ℝ) :
  is_even f →
  is_monotonic_decreasing_on_nonneg f →
  f_at_2_eq_zero f →
  ∀ x, f (x - 1) > 0 ↔ -1 < x ∧ x < 3 :=
by sorry

end range_of_x_for_fx1_positive_l218_218955


namespace find_m_l218_218555

def vec_a (m : ℝ) : ℝ × ℝ := (1, 2 * m)
def vec_b (m : ℝ) : ℝ × ℝ := (m + 1, 1)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem find_m (m : ℝ) : dot_product (vec_a m) (vec_b m) = 0 ↔ m = -1/3 := by 
  sorry

end find_m_l218_218555


namespace andy_time_difference_l218_218164

def time_dawn : ℕ := 20
def time_andy : ℕ := 46
def double_time_dawn : ℕ := 2 * time_dawn

theorem andy_time_difference :
  time_andy - double_time_dawn = 6 := by
  sorry

end andy_time_difference_l218_218164


namespace isosceles_triangle_congruent_side_length_l218_218024

theorem isosceles_triangle_congruent_side_length
  (B : ℕ) (A : ℕ) (P : ℕ) (L : ℕ)
  (h₁ : B = 36) (h₂ : A = 108) (h₃ : P = 84) :
  L = 24 :=
by
  -- Proof omitted
  sorry

end isosceles_triangle_congruent_side_length_l218_218024


namespace remainder_div_2468135790_101_l218_218827

theorem remainder_div_2468135790_101 : 2468135790 % 101 = 50 :=
by
  sorry

end remainder_div_2468135790_101_l218_218827


namespace number_of_questions_in_exam_l218_218734

theorem number_of_questions_in_exam :
  ∀ (typeA : ℕ) (typeB : ℕ) (timeA : ℝ) (timeB : ℝ) (totalTime : ℝ),
    typeA = 100 →
    timeA = 1.2 →
    timeB = 0.6 →
    totalTime = 180 →
    120 = typeA * timeA →
    totalTime - 120 = typeB * timeB →
    typeA + typeB = 200 :=
by
  intros typeA typeB timeA timeB totalTime h_typeA h_timeA h_timeB h_totalTime h_timeA_calc h_remaining_time
  sorry

end number_of_questions_in_exam_l218_218734


namespace longer_side_length_l218_218254

theorem longer_side_length (total_rope_length shorter_side_length longer_side_length : ℝ) 
  (h1 : total_rope_length = 100) 
  (h2 : shorter_side_length = 22) 
  : 2 * shorter_side_length + 2 * longer_side_length = total_rope_length -> longer_side_length = 28 :=
by sorry

end longer_side_length_l218_218254


namespace multiple_of_3_l218_218849

theorem multiple_of_3 (n : ℕ) (h1 : n ≥ 2) (h2 : n ∣ 2^n + 1) : 3 ∣ n :=
sorry

end multiple_of_3_l218_218849


namespace jack_marbles_l218_218347

theorem jack_marbles (initial_marbles share_marbles : ℕ) (h_initial : initial_marbles = 62) (h_share : share_marbles = 33) : 
  initial_marbles - share_marbles = 29 :=
by 
  sorry

end jack_marbles_l218_218347


namespace ab_perpendicular_cd_l218_218328

variables {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

-- Assuming points are members of a metric space and distances are calculated using the distance function
variables (a b c d : A)

-- Given condition
def given_condition : Prop := 
  dist a c ^ 2 + dist b d ^ 2 = dist a d ^ 2 + dist b c ^ 2

-- Statement that needs to be proven
theorem ab_perpendicular_cd (h : given_condition a b c d) : dist a b * dist c d = 0 :=
sorry

end ab_perpendicular_cd_l218_218328


namespace unique_solution_values_a_l218_218550

theorem unique_solution_values_a (a : ℝ) : 
  (∃ x y : ℝ, |x| + |y - 1| = 1 ∧ y = a * x + 2012) ∧ 
  (∀ x1 y1 x2 y2 : ℝ, (|x1| + |y1 - 1| = 1 ∧ y1 = a * x1 + 2012) ∧ 
                      (|x2| + |y2 - 1| = 1 ∧ y2 = a * x2 + 2012) → 
                      (x1 = x2 ∧ y1 = y2)) ↔ 
  a = 2011 ∨ a = -2011 := 
sorry

end unique_solution_values_a_l218_218550


namespace fewer_bees_than_flowers_l218_218068

theorem fewer_bees_than_flowers : 5 - 3 = 2 := by
  sorry

end fewer_bees_than_flowers_l218_218068


namespace problem_equiv_proof_l218_218647

theorem problem_equiv_proof :
  2015 * (1 + 1999 / 2015) * (1 / 4) - (2011 / 2015) = 503 := 
by
  sorry

end problem_equiv_proof_l218_218647


namespace g_at_5_l218_218730

def g (x : ℝ) : ℝ := 3 * x^5 - 15 * x^4 + 30 * x^3 - 45 * x^2 + 24 * x + 50

theorem g_at_5 : g 5 = 2795 :=
by
  sorry

end g_at_5_l218_218730


namespace different_prime_factors_mn_is_five_l218_218348

theorem different_prime_factors_mn_is_five {m n : ℕ} 
  (m_prime_factors : ∃ (p_1 p_2 p_3 p_4 : ℕ), True)  -- m has 4 different prime factors
  (n_prime_factors : ∃ (q_1 q_2 q_3 : ℕ), True)  -- n has 3 different prime factors
  (gcd_m_n : Nat.gcd m n = 15) : 
  (∃ k : ℕ, k = 5 ∧ (∃ (x_1 x_2 x_3 x_4 x_5 : ℕ), True)) := sorry

end different_prime_factors_mn_is_five_l218_218348


namespace find_divisor_l218_218822

/-- Given a dividend of 15698, a quotient of 89, and a remainder of 14, find the divisor. -/
theorem find_divisor :
  ∃ D : ℕ, 15698 = 89 * D + 14 ∧ D = 176 :=
by
  sorry

end find_divisor_l218_218822


namespace spend_on_candy_l218_218131

variable (initial_money spent_on_oranges spent_on_apples remaining_money spent_on_candy : ℕ)

-- Conditions
axiom initial_amount : initial_money = 95
axiom spent_on_oranges_value : spent_on_oranges = 14
axiom spent_on_apples_value : spent_on_apples = 25
axiom remaining_amount : remaining_money = 50

-- Question as a theorem
theorem spend_on_candy :
  spent_on_candy = initial_money - (spent_on_oranges + spent_on_apples) - remaining_money :=
by sorry

end spend_on_candy_l218_218131


namespace part1_inequality_part2_min_value_l218_218171

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  4^x + m * 2^x

theorem part1_inequality (x : ℝ) : f x (-3) > 4 → x > 2 :=
  sorry

theorem part2_min_value (h : (∀ x : ℝ, f x m + f (-x) m ≥ -4)) : m = -3 :=
  sorry

end part1_inequality_part2_min_value_l218_218171


namespace point_on_x_axis_l218_218026

theorem point_on_x_axis (x : ℝ) (A : ℝ × ℝ) (h : A = (2 - x, x + 3)) (hy : A.snd = 0) : A = (5, 0) :=
by
  sorry

end point_on_x_axis_l218_218026


namespace nina_money_l218_218701

theorem nina_money (C : ℝ) (h1 : C > 0) (h2 : 6 * C = 8 * (C - 2)) : 6 * C = 48 :=
by
  sorry

end nina_money_l218_218701


namespace cube_inscribed_sphere_volume_l218_218596

noncomputable def cubeSurfaceArea (a : ℝ) : ℝ := 6 * a^2
noncomputable def sphereVolume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3
noncomputable def inscribedSphereRadius (a : ℝ) : ℝ := a / 2

theorem cube_inscribed_sphere_volume :
  ∀ (a : ℝ), cubeSurfaceArea a = 24 → sphereVolume (inscribedSphereRadius a) = (4 / 3) * Real.pi := 
by 
  intros a h₁
  sorry

end cube_inscribed_sphere_volume_l218_218596


namespace sum_of_series_l218_218939

theorem sum_of_series (h1 : 2 + 4 + 6 + 8 + 10 = 30) (h2 : 1 + 3 + 5 + 7 + 9 = 25) : 
  ((2 + 4 + 6 + 8 + 10) / (1 + 3 + 5 + 7 + 9)) + ((1 + 3 + 5 + 7 + 9) / (2 + 4 + 6 + 8 + 10)) = 61 / 30 := by
  sorry

end sum_of_series_l218_218939


namespace abs_frac_sqrt_l218_218779

theorem abs_frac_sqrt (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) (h_eq : a^2 + b^2 = 9 * a * b) : 
  abs ((a + b) / (a - b)) = Real.sqrt (11 / 7) :=
by
  sorry

end abs_frac_sqrt_l218_218779


namespace calculate_expression_l218_218477

theorem calculate_expression (x y : ℚ) (hx : x = 5 / 6) (hy : y = 6 / 5) : 
  (1 / 3) * (x ^ 8) * (y ^ 9) = 2 / 5 :=
by
  sorry

end calculate_expression_l218_218477


namespace A_3_2_eq_29_l218_218137

def A : ℕ → ℕ → ℕ
| 0, n     => n + 1
| (m + 1), 0 => A m 1
| (m + 1), (n + 1) => A m (A (m + 1) n)

theorem A_3_2_eq_29 : A 3 2 = 29 := by
  sorry

end A_3_2_eq_29_l218_218137


namespace num_students_play_cricket_l218_218048

theorem num_students_play_cricket 
  (total_students : ℕ)
  (play_football : ℕ)
  (play_both : ℕ)
  (play_neither : ℕ)
  (C : ℕ) :
  total_students = 450 →
  play_football = 325 →
  play_both = 100 →
  play_neither = 50 →
  (total_students - play_neither = play_football + C - play_both) →
  C = 175 := by
  intros h0 h1 h2 h3 h4
  sorry

end num_students_play_cricket_l218_218048


namespace probability_of_karnataka_student_l218_218954

-- Defining the conditions

-- Number of students from each region
def total_students : ℕ := 10
def maharashtra_students : ℕ := 4
def karnataka_students : ℕ := 3
def goa_students : ℕ := 3

-- Number of students to be selected
def students_to_select : ℕ := 4

-- Total ways to choose 4 students out of 10
def C_total : ℕ := Nat.choose total_students students_to_select

-- Ways to select 4 students from the 7 students not from Karnataka
def non_karnataka_students : ℕ := maharashtra_students + goa_students
def C_non_karnataka : ℕ := Nat.choose non_karnataka_students students_to_select

-- Probability calculations
def P_no_karnataka : ℚ := C_non_karnataka / C_total
def P_at_least_one_karnataka : ℚ := 1 - P_no_karnataka

-- The statement to be proved
theorem probability_of_karnataka_student :
  P_at_least_one_karnataka = 5 / 6 :=
sorry

end probability_of_karnataka_student_l218_218954


namespace slope_of_line_l218_218408

theorem slope_of_line (x y : ℝ) (h : x + 2 * y + 1 = 0) : y = - (1 / 2) * x - (1 / 2) :=
by
  sorry -- The solution would be filled in here

#check slope_of_line -- additional check to ensure theorem implementation is correct

end slope_of_line_l218_218408


namespace fraction_of_salary_on_rent_l218_218811

theorem fraction_of_salary_on_rent
  (S : ℝ) (food_fraction : ℝ) (clothes_fraction : ℝ) (remaining_amount : ℝ) (approx_salary : ℝ)
  (food_fraction_eq : food_fraction = 1 / 5)
  (clothes_fraction_eq : clothes_fraction = 3 / 5)
  (remaining_amount_eq : remaining_amount = 19000)
  (approx_salary_eq : approx_salary = 190000) :
  ∃ (H : ℝ), H = 1 / 10 :=
by
  sorry

end fraction_of_salary_on_rent_l218_218811


namespace painted_surface_area_of_pyramid_l218_218195

/--
Given 19 unit cubes arranged in a 4-layer pyramid-like structure, where:
- The top layer has 1 cube,
- The second layer has 3 cubes,
- The third layer has 5 cubes,
- The bottom layer has 10 cubes,

Prove that the total painted surface area is 43 square meters.
-/
theorem painted_surface_area_of_pyramid :
  let layer1 := 1 -- top layer
  let layer2 := 3 -- second layer
  let layer3 := 5 -- third layer
  let layer4 := 10 -- bottom layer
  let total_cubes := layer1 + layer2 + layer3 + layer4
  let top_faces := layer1 * 1 + layer2 * 1 + layer3 * 1 + layer4 * 1
  let side_faces_layer1 := layer1 * 5
  let side_faces_layer2 := layer2 * 3
  let side_faces_layer3 := layer3 * 2
  let side_faces := side_faces_layer1 + side_faces_layer2 + side_faces_layer3
  let total_surface_area := top_faces + side_faces
  total_cubes = 19 → total_surface_area = 43 :=
by
  intros
  sorry

end painted_surface_area_of_pyramid_l218_218195


namespace given_fraction_l218_218057

variable (initial_cards : ℕ)
variable (cards_given_to_friend : ℕ)
variable (fraction_given_to_brother : ℚ)

noncomputable def fraction_given (initial_cards cards_given_to_friend : ℕ) (fraction_given_to_brother : ℚ) : Prop :=
  let cards_left := initial_cards / 2
  initial_cards - cards_left - cards_given_to_friend = fraction_given_to_brother * initial_cards

theorem given_fraction
  (h_initial : initial_cards = 16)
  (h_given_to_friend : cards_given_to_friend = 2)
  (h_fraction : fraction_given_to_brother = 3 / 8) :
  fraction_given initial_cards cards_given_to_friend fraction_given_to_brother :=
by
  sorry

end given_fraction_l218_218057


namespace find_height_of_door_l218_218749

noncomputable def height_of_door (x : ℝ) (w : ℝ) (h : ℝ) : ℝ := h

theorem find_height_of_door :
  ∃ x w h, (w = x - 4) ∧ (h = x - 2) ∧ (x^2 = w^2 + h^2) ∧ height_of_door x w h = 8 :=
by {
  sorry
}

end find_height_of_door_l218_218749


namespace proof_calculate_expr_l218_218857

def calculate_expr : Prop :=
  (4 + 4 + 6) / 3 - 2 / 3 = 4

theorem proof_calculate_expr : calculate_expr := 
by 
  sorry

end proof_calculate_expr_l218_218857


namespace meeting_time_l218_218265

theorem meeting_time (x : ℝ) :
  (1/6) * x + (1/4) * (x - 1) = 1 :=
sorry

end meeting_time_l218_218265


namespace problem_statement_l218_218677

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + (Real.pi / 4))

theorem problem_statement :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ∈ Set.Icc (-Real.sqrt 2 / 2) 1) ∧
  (f (Real.pi / 2) = -Real.sqrt 2 / 2) ∧
  (∀ k : ℤ, ∀ x ∈ Set.Icc (k * Real.pi - 3 * Real.pi / 8) (k * Real.pi + Real.pi / 8), 
    ∃ δ > 0, ∀ y ∈ Set.Ioc x (x + δ), f x < f y) :=
by {
  sorry
}

end problem_statement_l218_218677


namespace total_books_together_l218_218015

-- Given conditions
def SamBooks : Nat := 110
def JoanBooks : Nat := 102

-- Theorem to prove the total number of books they have together
theorem total_books_together : SamBooks + JoanBooks = 212 := 
by
  sorry

end total_books_together_l218_218015


namespace quotient_base4_correct_l218_218307

noncomputable def base4_to_base10 (n : ℕ) : ℕ :=
  match n with
  | 1302 => 1 * 4^3 + 3 * 4^2 + 0 * 4^1 + 2 * 4^0
  | 12 => 1 * 4^1 + 2 * 4^0
  | _ => 0

def base10_to_base4 (n : ℕ) : ℕ :=
  match n with
  | 19 => 1 * 4^2 + 0 * 4^1 + 3 * 4^0
  | _ => 0

theorem quotient_base4_correct : base10_to_base4 (114 / 6) = 103 := 
  by sorry

end quotient_base4_correct_l218_218307


namespace polynomial_evaluation_l218_218948

theorem polynomial_evaluation :
  ∀ x : ℤ, x = -2 → (x^3 + x^2 + x + 1 = -5) :=
by
  intros x hx
  rw [hx]
  norm_num

end polynomial_evaluation_l218_218948


namespace roots_reciprocal_sum_l218_218123

theorem roots_reciprocal_sum
  {a b c : ℂ}
  (h_roots : ∀ x : ℂ, (x - a) * (x - b) * (x - c) = x^3 - x + 1) :
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) = -2 :=
by
  sorry

end roots_reciprocal_sum_l218_218123


namespace range_of_k_for_distinct_real_roots_l218_218772

theorem range_of_k_for_distinct_real_roots (k : ℝ) : 
  (∀ x : ℝ, (k - 1) * x^2 - 2 * x + 1 = 0) → (k < 2 ∧ k ≠ 1) :=
by
  sorry

end range_of_k_for_distinct_real_roots_l218_218772


namespace max_min_difference_abc_l218_218372

theorem max_min_difference_abc (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) :
    let M := 1
    let m := -1/2
    M - m = 3/2 :=
by
  sorry

end max_min_difference_abc_l218_218372


namespace max_distinct_numbers_example_l218_218794

def max_distinct_numbers (a b c d e : ℕ) : ℕ := sorry

theorem max_distinct_numbers_example
  (A B : ℕ) :
  max_distinct_numbers 100 200 400 A B = 64 := sorry

end max_distinct_numbers_example_l218_218794


namespace yen_per_pound_l218_218820

theorem yen_per_pound 
  (pounds_initial : ℕ) 
  (euros : ℕ) 
  (yen_initial : ℕ) 
  (pounds_per_euro : ℕ) 
  (yen_total : ℕ) 
  (hp : pounds_initial = 42) 
  (he : euros = 11) 
  (hy : yen_initial = 3000) 
  (hpe : pounds_per_euro = 2) 
  (hy_total : yen_total = 9400) 
  : (yen_total - yen_initial) / (pounds_initial + euros * pounds_per_euro) = 100 := 
by
  sorry

end yen_per_pound_l218_218820


namespace impossible_to_form_palindrome_l218_218064

-- Define the possible cards
inductive Card
| abc | bca | cab

-- Define the rule for palindrome formation
def canFormPalindrome (w : List Card) : Prop :=
  sorry  -- Placeholder for the actual formation rule

-- Define the theorem statement
theorem impossible_to_form_palindrome (w : List Card) :
  ¬canFormPalindrome w :=
sorry

end impossible_to_form_palindrome_l218_218064


namespace hillary_sunday_spend_l218_218096

noncomputable def spend_per_sunday (total_spent : ℕ) (weeks : ℕ) (weekday_price : ℕ) (weekday_papers : ℕ) : ℕ :=
  (total_spent - weeks * weekday_papers * weekday_price) / weeks

theorem hillary_sunday_spend :
  spend_per_sunday 2800 8 50 3 = 200 :=
sorry

end hillary_sunday_spend_l218_218096


namespace number_of_people_quit_l218_218223

-- Define the conditions as constants.
def initial_team_size : ℕ := 25
def new_members : ℕ := 13
def final_team_size : ℕ := 30

-- Define the question as a function.
def people_quit (Q : ℕ) : Prop :=
  initial_team_size - Q + new_members = final_team_size

-- Prove the main statement assuming the conditions.
theorem number_of_people_quit (Q : ℕ) (h : people_quit Q) : Q = 8 :=
by
  sorry -- Proof is not required, so we use sorry to skip it.

end number_of_people_quit_l218_218223


namespace max_chord_length_l218_218884

theorem max_chord_length (x1 y1 x2 y2 : ℝ) (h_parabola1 : x1^2 = 8 * y1) (h_parabola2 : x2^2 = 8 * y2)
  (h_midpoint_ordinate : (y1 + y2) / 2 = 4) :
  abs ((y1 + y2) + 4) = 12 :=
by
  sorry

end max_chord_length_l218_218884


namespace pencil_pen_cost_l218_218675

theorem pencil_pen_cost 
  (p q : ℝ) 
  (h1 : 6 * p + 3 * q = 3.90) 
  (h2 : 2 * p + 5 * q = 4.45) :
  3 * p + 4 * q = 3.92 :=
by
  sorry

end pencil_pen_cost_l218_218675


namespace numbers_not_equal_l218_218681

theorem numbers_not_equal
  (a b c S : ℝ)
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (h1 : a + b^2 + c^2 = S)
  (h2 : b + a^2 + c^2 = S)
  (h3 : c + a^2 + b^2 = S) :
  ¬ (a = b ∧ b = c) :=
by sorry

end numbers_not_equal_l218_218681


namespace average_billboards_per_hour_l218_218248

def first_hour_billboards : ℕ := 17
def second_hour_billboards : ℕ := 20
def third_hour_billboards : ℕ := 23

theorem average_billboards_per_hour : 
  (first_hour_billboards + second_hour_billboards + third_hour_billboards) / 3 = 20 := 
by
  sorry

end average_billboards_per_hour_l218_218248


namespace complete_the_square_k_l218_218161

theorem complete_the_square_k (x : ℝ) : ∃ k : ℝ, (∃ a h: ℝ, (a = 1) ∧ (h = 7 / 2) ∧ (x^2 - 7*x = a * (x - h)^2 + k)) → k = -49 / 4 :=
by
  sorry

end complete_the_square_k_l218_218161


namespace total_games_l218_218837

-- Define the conditions
def games_this_year : ℕ := 4
def games_last_year : ℕ := 9

-- Define the proposition that we want to prove
theorem total_games : games_this_year + games_last_year = 13 := by
  sorry

end total_games_l218_218837


namespace max_value_M_l218_218060

open Real

theorem max_value_M :
  ∃ M : ℝ, ∀ x y z u : ℝ, 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < u ∧ z ≥ y ∧ (x - 2 * y = z - 2 * u) ∧ (2 * y * z = u * x) →
  M ≤ z / y ∧ M = 6 + 4 * sqrt 2 := 
  sorry

end max_value_M_l218_218060


namespace perimeter_of_square_C_l218_218515

theorem perimeter_of_square_C (a b : ℝ) 
  (hA : 4 * a = 16) 
  (hB : 4 * b = 32) : 
  4 * (a + b) = 48 := by
  sorry

end perimeter_of_square_C_l218_218515


namespace train_speed_l218_218039

noncomputable def train_length : ℝ := 1500
noncomputable def bridge_length : ℝ := 1200
noncomputable def crossing_time : ℝ := 30

theorem train_speed :
  (train_length + bridge_length) / crossing_time = 90 := by
  sorry

end train_speed_l218_218039


namespace tasks_completed_correctly_l218_218704

theorem tasks_completed_correctly (x y : ℕ) (h1 : 9 * x - 5 * y = 57) (h2 : x + y ≤ 15) : x = 8 := 
by
  sorry

end tasks_completed_correctly_l218_218704


namespace find_p_l218_218142

theorem find_p (p q : ℚ) (h1 : 3 * p + 4 * q = 15) (h2 : 4 * p + 3 * q = 18) : p = 27 / 7 :=
by
  sorry

end find_p_l218_218142


namespace fruit_salad_cherries_l218_218793

theorem fruit_salad_cherries (b r g c : ℕ) 
(h1 : b + r + g + c = 360)
(h2 : r = 3 * b) 
(h3 : g = 4 * c)
(h4 : c = 5 * r) :
c = 68 := 
sorry

end fruit_salad_cherries_l218_218793


namespace sum_of_cubes_divisible_by_middle_integer_l218_218521

theorem sum_of_cubes_divisible_by_middle_integer (a : ℤ) : 
  (a - 1)^3 + a^3 + (a + 1)^3 ∣ 3 * a :=
sorry

end sum_of_cubes_divisible_by_middle_integer_l218_218521


namespace chandler_total_rolls_l218_218566

-- Definitions based on given conditions
def rolls_sold_grandmother : ℕ := 3
def rolls_sold_uncle : ℕ := 4
def rolls_sold_neighbor : ℕ := 3
def rolls_needed_more : ℕ := 2

-- Total rolls sold so far and needed
def total_rolls_to_sell : ℕ :=
  rolls_sold_grandmother + rolls_sold_uncle + rolls_sold_neighbor + rolls_needed_more

theorem chandler_total_rolls : total_rolls_to_sell = 12 :=
by
  sorry

end chandler_total_rolls_l218_218566


namespace arithmetic_sequence_tenth_term_l218_218744

theorem arithmetic_sequence_tenth_term :
  ∀ (a : ℚ) (a_20 : ℚ) (a_10 : ℚ),
    a = 5 / 11 →
    a_20 = 9 / 11 →
    a_10 = a + (9 * ((a_20 - a) / 19)) →
    a_10 = 1233 / 2309 :=
by
  intros a a_20 a_10 h_a h_a_20 h_a_10
  sorry

end arithmetic_sequence_tenth_term_l218_218744


namespace emptying_tank_time_l218_218038

theorem emptying_tank_time :
  let V := 30 * 12^3 -- volume of the tank in cubic inches
  let r_in := 3 -- rate of inlet pipe in cubic inches per minute
  let r_out1 := 12 -- rate of first outlet pipe in cubic inches per minute
  let r_out2 := 6 -- rate of second outlet pipe in cubic inches per minute
  let net_rate := r_out1 + r_out2 - r_in
  V / net_rate = 3456 := by
sorry

end emptying_tank_time_l218_218038


namespace biology_marks_l218_218375

theorem biology_marks (english : ℕ) (math : ℕ) (physics : ℕ) (chemistry : ℕ) (average : ℕ) (biology : ℕ) 
  (h1 : english = 36) 
  (h2 : math = 35) 
  (h3 : physics = 42) 
  (h4 : chemistry = 57) 
  (h5 : average = 45) 
  (h6 : (english + math + physics + chemistry + biology) / 5 = average) : 
  biology = 55 := 
by
  sorry

end biology_marks_l218_218375


namespace units_digit_of_product_l218_218054

def factorial : Nat → Nat
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def units_digit (n : Nat) : Nat :=
  n % 10

def target_product : Nat :=
  factorial 1 * factorial 2 * factorial 3 * factorial 4

theorem units_digit_of_product : units_digit target_product = 8 :=
  by
    sorry

end units_digit_of_product_l218_218054


namespace fraction_product_is_simplified_form_l218_218282

noncomputable def fraction_product : ℚ := (2 / 3) * (5 / 11) * (3 / 8)

theorem fraction_product_is_simplified_form :
  fraction_product = 5 / 44 :=
by
  sorry

end fraction_product_is_simplified_form_l218_218282


namespace inclination_angle_of_line_l218_218642

noncomputable def angle_of_inclination (m : ℝ) : ℝ :=
  Real.arctan m

theorem inclination_angle_of_line (α : ℝ) :
  angle_of_inclination (-1) = 3 * Real.pi / 4 :=
by
  sorry

end inclination_angle_of_line_l218_218642


namespace find_m_minus_n_l218_218167

theorem find_m_minus_n (m n : ℝ) (h1 : -5 + 1 = m) (h2 : -5 * 1 = n) : m - n = 1 :=
sorry

end find_m_minus_n_l218_218167


namespace time_to_watch_all_episodes_l218_218052

theorem time_to_watch_all_episodes 
    (n_seasons : ℕ) (episodes_per_season : ℕ) (last_season_extra_episodes : ℕ) (hours_per_episode : ℚ)
    (h1 : n_seasons = 9)
    (h2 : episodes_per_season = 22)
    (h3 : last_season_extra_episodes = 4)
    (h4 : hours_per_episode = 0.5) :
    n_seasons * episodes_per_season + (episodes_per_season + last_season_extra_episodes) * hours_per_episode = 112 :=
by
  sorry

end time_to_watch_all_episodes_l218_218052


namespace participation_schemes_count_l218_218063

-- Define the conditions
def num_people : ℕ := 6
def num_selected : ℕ := 4
def subjects : List String := ["math", "physics", "chemistry", "english"]
def not_in_english : List String := ["A", "B"]

-- Define the problem 
theorem participation_schemes_count : 
  ∃ total_schemes : ℕ , (total_schemes = 240) :=
by {
  sorry
}

end participation_schemes_count_l218_218063


namespace mike_can_buy_nine_games_l218_218194

noncomputable def mike_dollars (initial_dollars : ℕ) (spent_dollars : ℕ) (game_cost : ℕ) : ℕ :=
  (initial_dollars - spent_dollars) / game_cost

theorem mike_can_buy_nine_games : mike_dollars 69 24 5 = 9 := by
  sorry

end mike_can_buy_nine_games_l218_218194


namespace m_plus_n_l218_218512

theorem m_plus_n (m n : ℕ) (hm : 0 < m) (hn : 1 < n) (h : m ^ n = 2^25 * 3^40) : m + n = 209957 :=
  sorry

end m_plus_n_l218_218512


namespace part1_solution_set_part2_range_of_a_l218_218445

def f (x a : ℝ) := |x - a^2| + |x - (2*a + 1)|

-- Part (1)
theorem part1_solution_set (x : ℝ) :
  ∀ x, f x 2 ≥ 4 ↔ x ≤ 3 / 2 ∨ x ≥ 11 / 2 := sorry

-- Part (2)
theorem part2_range_of_a (a : ℝ) :
  (∀ x, f x a ≥ 4) ↔ (a ≤ -1 ∨ a ≥ 3) := sorry

end part1_solution_set_part2_range_of_a_l218_218445


namespace rectangle_area_l218_218357

theorem rectangle_area 
  (length_to_width_ratio : Real) 
  (width : Real) 
  (area : Real) 
  (h1 : length_to_width_ratio = 0.875) 
  (h2 : width = 24) 
  (h_area : area = 504) : 
  True := 
sorry

end rectangle_area_l218_218357


namespace problem_D_l218_218200

-- Define the lines m and n, and planes α and β
variables (m n : Type) (α β : Type)

-- Define the parallel and perpendicular relations
variables (parallel : Type → Type → Prop) (perpendicular : Type → Type → Prop)

-- Assume the conditions of problem D
variables (h1 : perpendicular m α) (h2 : parallel n β) (h3 : parallel α β)

-- The proof problem statement: Prove that under these assumptions, m is perpendicular to n
theorem problem_D : perpendicular m n :=
sorry

end problem_D_l218_218200


namespace find_m_l218_218569

theorem find_m (m : ℝ) :
  (m - 2013 = 0) → (m = 2013) ∧ (m - 1 ≠ 0) :=
by {
  sorry
}

end find_m_l218_218569


namespace initial_people_count_l218_218243

theorem initial_people_count (C : ℝ) (n : ℕ) (h : n > 1) :
  ((C / (n - 1)) - (C / n) = 0.125) →
  n = 8 := by
  sorry

end initial_people_count_l218_218243


namespace ranch_cows_variance_l218_218333

variable (n : ℕ)
variable (p : ℝ)

-- Definition of the variance of a binomial distribution
def binomial_variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

theorem ranch_cows_variance : 
  binomial_variance 10 0.02 = 0.196 :=
by
  sorry

end ranch_cows_variance_l218_218333


namespace benjamin_trip_odd_number_conditions_l218_218314

theorem benjamin_trip_odd_number_conditions (a b c : ℕ) 
  (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a)
  (h4 : a + b + c ≤ 9) 
  (h5 : ∃ x : ℕ, 60 * x = 99 * (c - a)) :
  a^2 + b^2 + c^2 = 35 := 
sorry

end benjamin_trip_odd_number_conditions_l218_218314


namespace emily_art_supplies_l218_218463

theorem emily_art_supplies (total_spent skirts_cost skirt_quantity : ℕ) 
  (total_spent_eq : total_spent = 50) 
  (skirt_cost_eq : skirts_cost = 15) 
  (skirt_quantity_eq : skirt_quantity = 2) :
  total_spent - skirt_quantity * skirts_cost = 20 :=
by
  sorry

end emily_art_supplies_l218_218463


namespace sequence_term_l218_218202

theorem sequence_term (a : ℕ → ℝ) (h₀ : a 1 = 1) (h₁ : ∀ n : ℕ, (n + 1) * a n = 2 * n * a (n + 1)) : 
  ∀ n : ℕ, a n = n / 2^(n - 1) :=
by
  sorry

end sequence_term_l218_218202


namespace people_in_each_bus_l218_218343

-- Definitions and conditions
def num_vans : ℕ := 2
def num_buses : ℕ := 3
def people_per_van : ℕ := 8
def total_people : ℕ := 76

-- Theorem statement to prove the number of people in each bus
theorem people_in_each_bus : (total_people - num_vans * people_per_van) / num_buses = 20 :=
by
    -- The actual proof would go here
    sorry

end people_in_each_bus_l218_218343


namespace intersection_line_circle_diameter_l218_218977

noncomputable def length_of_AB : ℝ := 2

theorem intersection_line_circle_diameter 
  (x y : ℝ)
  (h_line : x - 2*y - 1 = 0)
  (h_circle : (x - 1)^2 + y^2 = 1) :
  |(length_of_AB)| = 2 := 
sorry

end intersection_line_circle_diameter_l218_218977


namespace find_x_l218_218687

theorem find_x : 
  ∃ x : ℝ, 4.7 * 13.26 + 4.7 * 9.43 + 4.7 * x = 470 ∧ x = 77.31 :=
by
  sorry

end find_x_l218_218687


namespace gretchen_charge_per_drawing_l218_218500

-- Given conditions
def sold_on_Saturday : ℕ := 24
def sold_on_Sunday : ℕ := 16
def total_amount : ℝ := 800
def total_drawings := sold_on_Saturday + sold_on_Sunday

-- Assertion to prove
theorem gretchen_charge_per_drawing (x : ℝ) (h : total_drawings * x = total_amount) : x = 20 :=
by
  sorry

end gretchen_charge_per_drawing_l218_218500


namespace circle_through_three_points_l218_218759

open Real

structure Point where
  x : ℝ
  y : ℝ

def circle_equation (D E F : ℝ) (P : Point) : Prop :=
  P.x^2 + P.y^2 + D * P.x + E * P.y + F = 0

theorem circle_through_three_points :
  ∃ (D E F : ℝ), 
    (circle_equation D E F ⟨1, 12⟩) ∧ 
    (circle_equation D E F ⟨7, 10⟩) ∧ 
    (circle_equation D E F ⟨-9, 2⟩) ∧
    (D = -2) ∧ (E = -4) ∧ (F = -95) :=
by
  sorry

end circle_through_three_points_l218_218759


namespace range_of_a_l218_218547

variable {a x : ℝ}

theorem range_of_a (h_eq : 2 * (x + a) = x + 3) (h_ineq : 2 * x - 10 > 8 * a) : a < -1 / 3 := 
sorry

end range_of_a_l218_218547


namespace find_d_l218_218666

theorem find_d (d : ℝ) (h1 : ∃ (x y : ℝ), y = x + d ∧ x = -y + d ∧ x = d-1 ∧ y = d) : d = 1 :=
sorry

end find_d_l218_218666


namespace fiona_reaches_goal_l218_218485

-- Define the set of lily pads
def pads : Finset ℕ := Finset.range 15

-- Define the start, predator, and goal pads
def start_pad : ℕ := 0
def predator_pads : Finset ℕ := {4, 8}
def goal_pad : ℕ := 13

-- Define the hop probabilities
def hop_next : ℚ := 1/3
def hop_two : ℚ := 1/3
def hop_back : ℚ := 1/3

-- Define the transition probabilities (excluding jumps to negative pads)
def transition (current next : ℕ) : ℚ :=
  if next = current + 1 ∨ next = current + 2 ∨ (next = current - 1 ∧ current > 0)
  then 1/3 else 0

-- Define the function to check if a pad is safe
def is_safe (pad : ℕ) : Prop := ¬ (pad ∈ predator_pads)

-- Define the probability that Fiona reaches pad 13 without landing on 4 or 8
noncomputable def probability_reach_13 : ℚ :=
  -- Function to recursively calculate the probability
  sorry

-- Statement to prove
theorem fiona_reaches_goal : probability_reach_13 = 16 / 177147 := 
sorry

end fiona_reaches_goal_l218_218485


namespace each_person_pays_l218_218770

def numPeople : ℕ := 6
def rentalDays : ℕ := 4
def weekdayRate : ℕ := 420
def weekendRate : ℕ := 540
def numWeekdays : ℕ := 2
def numWeekends : ℕ := 2

theorem each_person_pays : 
  (numWeekdays * weekdayRate + numWeekends * weekendRate) / numPeople = 320 :=
by
  sorry

end each_person_pays_l218_218770


namespace relatively_prime_pair_count_l218_218860

theorem relatively_prime_pair_count :
  (∃ m n : ℕ, m > 0 ∧ n > 0 ∧ m + n = 190 ∧ Nat.gcd m n = 1) →
  (∃! k : ℕ, k = 26) :=
by
  sorry

end relatively_prime_pair_count_l218_218860


namespace sqrt_of_16_l218_218537

theorem sqrt_of_16 (x : ℝ) (hx : x^2 = 16) : x = 4 ∨ x = -4 := 
by
  sorry

end sqrt_of_16_l218_218537


namespace expenditure_should_increase_by_21_percent_l218_218597

noncomputable def old_income := 100.0
noncomputable def ratio_exp_sav := (3 : ℝ) / (2 : ℝ)
noncomputable def income_increase_percent := 15.0 / 100.0
noncomputable def savings_increase_percent := 6.0 / 100.0
noncomputable def old_expenditure := old_income * (3 / (3 + 2))
noncomputable def old_savings := old_income * (2 / (3 + 2))
noncomputable def new_income := old_income * (1 + income_increase_percent)
noncomputable def new_savings := old_savings * (1 + savings_increase_percent)
noncomputable def new_expenditure := new_income - new_savings
noncomputable def expenditure_increase_percent := ((new_expenditure - old_expenditure) / old_expenditure) * 100

theorem expenditure_should_increase_by_21_percent :
  expenditure_increase_percent = 21 :=
sorry

end expenditure_should_increase_by_21_percent_l218_218597


namespace range_of_a_l218_218226

open Real

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 ≤ x → x^2 + 2 * x - a > 0) → a < 3 :=
by
  sorry

end range_of_a_l218_218226


namespace sum_of_rationals_l218_218105

theorem sum_of_rationals (r1 r2 : ℚ) : ∃ r : ℚ, r = r1 + r2 :=
sorry

end sum_of_rationals_l218_218105


namespace star_wars_cost_l218_218032

theorem star_wars_cost 
    (LK_cost LK_earn SW_earn: ℕ) 
    (half_profit: ℕ → ℕ)
    (h1: LK_cost = 10)
    (h2: LK_earn = 200)
    (h3: SW_earn = 405)
    (h4: LK_earn - LK_cost = half_profit SW_earn)
    (h5: half_profit SW_earn * 2 = SW_earn - (LK_earn - LK_cost)) :
    ∃ SW_cost : ℕ, SW_cost = 25 := 
by
  sorry

end star_wars_cost_l218_218032


namespace initial_distance_from_lens_l218_218377

def focal_length := 150 -- focal length F in cm
def screen_shift := 40  -- screen moved by 40 cm

theorem initial_distance_from_lens (d : ℝ) (f : ℝ) (s : ℝ) 
  (h_focal_length : f = focal_length) 
  (h_screen_shift : s = screen_shift) 
  (h_parallel_beam : d = f / 2 ∨ d = 3 * f / 2) : 
  d = 130 ∨ d = 170 := 
by 
  sorry

end initial_distance_from_lens_l218_218377


namespace maximum_discount_rate_l218_218729

theorem maximum_discount_rate (cost_price selling_price : ℝ) (min_profit_margin : ℝ) :
  cost_price = 4 ∧ 
  selling_price = 5 ∧ 
  min_profit_margin = 0.4 → 
  ∃ x : ℝ, 5 * (1 - x / 100) - 4 ≥ 0.4 ∧ x = 12 :=
by
  sorry

end maximum_discount_rate_l218_218729


namespace polygon_interior_angles_sum_l218_218146

theorem polygon_interior_angles_sum {n : ℕ} 
  (h1 : ∀ (k : ℕ), k > 2 → (360 = k * 40)) :
  180 * (9 - 2) = 1260 :=
by
  sorry

end polygon_interior_angles_sum_l218_218146


namespace cubic_inequality_l218_218306

theorem cubic_inequality (x y z : ℝ) :
  x^3 + y^3 + z^3 + 3 * x * y * z ≥ x^2 * (y + z) + y^2 * (z + x) + z^2 * (x + y) :=
sorry

end cubic_inequality_l218_218306


namespace find_value_of_expression_l218_218124

-- Define non-negative variables
variables (x y z : ℝ) 

-- Conditions
def cond1 := x ^ 2 + x * y + y ^ 2 / 3 = 25
def cond2 := y ^ 2 / 3 + z ^ 2 = 9
def cond3 := z ^ 2 + z * x + x ^ 2 = 16

-- Target statement to be proven
theorem find_value_of_expression (h1 : cond1 x y) (h2 : cond2 y z) (h3 : cond3 z x) : 
  x * y + 2 * y * z + 3 * z * x = 24 * Real.sqrt 3 :=
sorry

end find_value_of_expression_l218_218124


namespace arithmetic_sequence_sum_l218_218612

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (d : ℤ)
  (h_arith : ∀ n, a n = a 0 + n * d)
  (h1 : a 0 + a 3 + a 6 = 45)
  (h2 : a 1 + a 4 + a 7 = 39) :
  a 2 + a 5 + a 8 = 33 := 
by
  sorry

end arithmetic_sequence_sum_l218_218612


namespace find_y_intercept_l218_218257

theorem find_y_intercept (x1 y1 x2 y2 : ℝ) (h₁ : (x1, y1) = (2, -2)) (h₂ : (x2, y2) = (6, 6)) : 
  ∃ b : ℝ, (∀ x : ℝ, y = 2 * x + b) ∧ b = -6 :=
by
  sorry

end find_y_intercept_l218_218257


namespace faster_speed_l218_218384

theorem faster_speed (Speed1 : ℝ) (ExtraDistance : ℝ) (ActualDistance : ℝ) (v : ℝ) : 
  Speed1 = 10 ∧ ExtraDistance = 31 ∧ ActualDistance = 20.67 ∧ 
  (ActualDistance / Speed1 = (ActualDistance + ExtraDistance) / v) → 
  v = 25 :=
by
  sorry

end faster_speed_l218_218384


namespace tangent_line_eq_bounded_area_l218_218092

-- Given two parabolas and a tangent line, and a positive constant a
variables (a : ℝ)
variables (y1 y2 l : ℝ → ℝ)

-- Conditions:
def parabola1 := ∀ (x : ℝ), y1 x = x^2 + a * x
def parabola2 := ∀ (x : ℝ), y2 x = x^2 - 2 * a * x
def tangent_line := ∀ (x : ℝ), l x = - (a / 2) * x - (9 * a^2 / 16)
def a_positive := a > 0

-- Proof goals:
theorem tangent_line_eq : 
  parabola1 a y1 ∧ parabola2 a y2 ∧ tangent_line a l ∧ a_positive a 
  → ∀ x, (y1 x = l x ∨ y2 x = l x) :=
sorry

theorem bounded_area : 
  parabola1 a y1 ∧ parabola2 a y2 ∧ tangent_line a l ∧ a_positive a 
  → ∫ (x : ℝ) in (-3 * a / 4)..(3 * a / 4), (y1 x - l x) + (y2 x - l x) = 9 * a^3 / 8 :=
sorry

end tangent_line_eq_bounded_area_l218_218092


namespace integral_abs_x_minus_two_l218_218590

theorem integral_abs_x_minus_two : ∫ x in (0:ℝ)..4, |x - 2| = 4 := 
by
  sorry

end integral_abs_x_minus_two_l218_218590


namespace kurt_savings_l218_218607

def daily_cost_old : ℝ := 0.85
def daily_cost_new : ℝ := 0.45
def days : ℕ := 30

theorem kurt_savings : (daily_cost_old * days) - (daily_cost_new * days) = 12.00 := by
  sorry

end kurt_savings_l218_218607


namespace findWorkRateB_l218_218056

-- Define the work rates of A and C given in the problem
def workRateA : ℚ := 1 / 8
def workRateC : ℚ := 1 / 16

-- Combined work rate when A, B, and C work together to complete the work in 4 days
def combinedWorkRate : ℚ := 1 / 4

-- Define the work rate of B that we need to prove
def workRateB : ℚ := 1 / 16

-- Theorem to prove that workRateB is equal to B's work rate given the conditions
theorem findWorkRateB : workRateA + workRateB + workRateC = combinedWorkRate :=
  by
  sorry

end findWorkRateB_l218_218056


namespace pow_mod_eq_residue_l218_218516

theorem pow_mod_eq_residue :
  (3 : ℤ)^(2048) % 11 = 5 :=
sorry

end pow_mod_eq_residue_l218_218516


namespace factor_expression_l218_218411

theorem factor_expression (a b c : ℝ) : 
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 = (a - b) * (b - c) * (c - a) * (a * b^2 + a * c^2) :=
by 
  sorry

end factor_expression_l218_218411
