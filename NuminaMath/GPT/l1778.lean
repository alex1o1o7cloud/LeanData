import Mathlib

namespace width_of_door_is_correct_l1778_177880

theorem width_of_door_is_correct
  (L : ℝ) (W : ℝ) (H : ℝ := 12)
  (door_height : ℝ := 6) (window_height : ℝ := 4) (window_width : ℝ := 3)
  (cost_per_square_foot : ℝ := 10) (total_cost : ℝ := 9060) :
  (L = 25 ∧ W = 15) →
  2 * (L + W) * H - (door_height * width_door + 3 * (window_height * window_width)) * cost_per_square_foot = total_cost →
  width_door = 3 :=
by
  intros h1 h2
  sorry

end width_of_door_is_correct_l1778_177880


namespace jogger_ahead_of_train_l1778_177865

noncomputable def distance_ahead_of_train (v_j v_t : ℕ) (L_t t : ℕ) : ℕ :=
  let relative_speed_kmh := v_t - v_j
  let relative_speed_ms := (relative_speed_kmh * 1000) / 3600
  let total_distance := relative_speed_ms * t
  total_distance - L_t

theorem jogger_ahead_of_train :
  distance_ahead_of_train 10 46 120 46 = 340 :=
by
  sorry

end jogger_ahead_of_train_l1778_177865


namespace find_subtracted_value_l1778_177864

theorem find_subtracted_value (N : ℕ) (V : ℕ) (hN : N = 2976) (h : (N / 12) - V = 8) : V = 240 := by
  sorry

end find_subtracted_value_l1778_177864


namespace inverse_fourier_transform_l1778_177800

noncomputable def F (p : ℝ) : ℂ :=
if 0 < p ∧ p < 1 then 1 else 0

noncomputable def f (x : ℝ) : ℂ :=
(1 / Real.sqrt (2 * Real.pi)) * ((1 - Complex.exp (-Complex.I * x)) / (Complex.I * x))

theorem inverse_fourier_transform :
  ∀ x, (f x) = (1 / (Real.sqrt (2 * Real.pi))) * ((1 - Complex.exp (-Complex.I * x)) / (Complex.I * x)) := by
  intros
  sorry

end inverse_fourier_transform_l1778_177800


namespace cost_per_liter_l1778_177855

/-
Given:
- Service cost per vehicle: $2.10
- Number of mini-vans: 3
- Number of trucks: 2
- Total cost: $299.1
- Mini-van's tank size: 65 liters
- Truck's tank is 120% bigger than a mini-van's tank
- All tanks are empty

Prove that the cost per liter of fuel is $0.60
-/

theorem cost_per_liter (service_cost_per_vehicle : ℝ) 
(number_of_minivans number_of_trucks : ℕ)
(total_cost : ℝ)
(minivan_tank_size : ℝ)
(truck_tank_multiplier : ℝ)
(fuel_cost : ℝ)
(total_fuel : ℝ) :
  service_cost_per_vehicle = 2.10 ∧
  number_of_minivans = 3 ∧
  number_of_trucks = 2 ∧
  total_cost = 299.1 ∧
  minivan_tank_size = 65 ∧
  truck_tank_multiplier = 1.2 ∧
  fuel_cost = (total_cost - (number_of_minivans + number_of_trucks) * service_cost_per_vehicle) ∧
  total_fuel = (number_of_minivans * minivan_tank_size + number_of_trucks * (minivan_tank_size * (1 + truck_tank_multiplier))) →
  (fuel_cost / total_fuel) = 0.60 :=
sorry

end cost_per_liter_l1778_177855


namespace sum_of_first_six_terms_l1778_177823

variable {a_n : ℕ → ℕ}
variable {d : ℕ}

def is_arithmetic_sequence (a_n : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, a_n (n + 1) = a_n n + d 

theorem sum_of_first_six_terms (a_3 a_4 : ℕ) (h : a_3 + a_4 = 30) :
  ∃ a_n d, is_arithmetic_sequence a_n d ∧ 
  a_n 3 = a_3 ∧ a_n 4 = a_4 ∧ 
  (3 * (a_n 1 + (a_n 1 + 5 * d))) = 90 := 
sorry

end sum_of_first_six_terms_l1778_177823


namespace no_solution_for_b_a_divides_a_b_minus_1_l1778_177893

theorem no_solution_for_b_a_divides_a_b_minus_1 :
  ¬ (∃ a b : ℕ, 1 ≤ a ∧ 1 ≤ b ∧ b^a ∣ a^b - 1) :=
by
  sorry

end no_solution_for_b_a_divides_a_b_minus_1_l1778_177893


namespace percentage_of_Hindu_boys_l1778_177827

theorem percentage_of_Hindu_boys (total_boys : ℕ) (muslim_percentage : ℕ) (sikh_percentage : ℕ)
  (other_community_boys : ℕ) (H : total_boys = 850) (H1 : muslim_percentage = 44) 
  (H2 : sikh_percentage = 10) (H3 : other_community_boys = 153) :
  let muslim_boys := muslim_percentage * total_boys / 100
  let sikh_boys := sikh_percentage * total_boys / 100
  let non_hindu_boys := muslim_boys + sikh_boys + other_community_boys
  let hindu_boys := total_boys - non_hindu_boys
  (hindu_boys * 100 / total_boys : ℚ) = 28 := 
by
  sorry

end percentage_of_Hindu_boys_l1778_177827


namespace rose_bushes_unwatered_l1778_177894

theorem rose_bushes_unwatered (n V A : ℕ) (V_set A_set : Finset ℕ) (hV : V = 1003) (hA : A = 1003) (hTotal : n = 2006) (hIntersection : V_set.card = 3) :
  n - (V + A - V_set.card) = 3 :=
by
  sorry

end rose_bushes_unwatered_l1778_177894


namespace sin_cos_identity_l1778_177838

theorem sin_cos_identity (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : Real.sin x * Real.cos x = 4 / 17 := 
by
  sorry

end sin_cos_identity_l1778_177838


namespace necessary_condition_for_line_passes_quadrants_l1778_177816

theorem necessary_condition_for_line_passes_quadrants (m n : ℝ) (h_line : ∀ x : ℝ, x * (m / n) - (1 / n) < 0 ∨ x * (m / n) - (1 / n) > 0) : m * n < 0 :=
by
  sorry

end necessary_condition_for_line_passes_quadrants_l1778_177816


namespace max_product_of_real_roots_quadratic_eq_l1778_177886

theorem max_product_of_real_roots_quadratic_eq : ∀ (k : ℝ), (∃ x y : ℝ, 4 * x ^ 2 - 8 * x + k = 0 ∧ 4 * y ^ 2 - 8 * y + k = 0) 
    → k = 4 :=
sorry

end max_product_of_real_roots_quadratic_eq_l1778_177886


namespace max_value_of_f_l1778_177849

noncomputable def f (x : ℝ) : ℝ := x - Real.sin x

theorem max_value_of_f : ∃ x ∈ Set.Icc (Real.pi / 2) Real.pi, ∀ y ∈ Set.Icc (Real.pi / 2) Real.pi, f y ≤ f x ∧ f x = Real.pi := 
by
  sorry

end max_value_of_f_l1778_177849


namespace minimum_value_of_sum_of_squares_l1778_177854

theorem minimum_value_of_sum_of_squares (x y z : ℝ) (h : 2 * x - y - 2 * z = 6) : 
  x^2 + y^2 + z^2 ≥ 4 :=
sorry

end minimum_value_of_sum_of_squares_l1778_177854


namespace principal_amount_l1778_177841

theorem principal_amount (P : ℝ) (CI SI : ℝ) 
  (H1 : CI = P * 0.44) 
  (H2 : SI = P * 0.4) 
  (H3 : CI - SI = 216) : 
  P = 5400 :=
by {
  sorry
}

end principal_amount_l1778_177841


namespace geometric_sequence_common_ratio_l1778_177872

theorem geometric_sequence_common_ratio (r : ℝ) (a : ℝ) (a3 : ℝ) :
  a = 3 → a3 = 27 → r = 3 ∨ r = -3 :=
by
  intros ha ha3
  sorry

end geometric_sequence_common_ratio_l1778_177872


namespace factorize_x_cubed_minus_9x_l1778_177856

theorem factorize_x_cubed_minus_9x (x : ℝ) : x^3 - 9 * x = x * (x + 3) * (x - 3) :=
sorry

end factorize_x_cubed_minus_9x_l1778_177856


namespace certain_number_is_50_l1778_177833

theorem certain_number_is_50 (x : ℝ) (h : 0.6 * x = 0.42 * 30 + 17.4) : x = 50 :=
by
  sorry

end certain_number_is_50_l1778_177833


namespace ferris_wheel_seats_l1778_177889

-- Define the total number of seats S as a variable
variables (S : ℕ)

-- Define the conditions
def seat_capacity : ℕ := 15

def broken_seats : ℕ := 10

def max_riders : ℕ := 120

-- The theorem statement
theorem ferris_wheel_seats :
  ((S - broken_seats) * seat_capacity = max_riders) → S = 18 :=
by
  sorry

end ferris_wheel_seats_l1778_177889


namespace quadrilateral_diagonals_perpendicular_l1778_177805

def convex_quadrilateral (A B C D : Type) : Prop := sorry -- Assume it’s defined elsewhere 
def tangent_to_all_sides (circle : Type) (A B C D : Type) : Prop := sorry -- Assume it’s properly specified with its conditions elsewhere
def tangent_to_all_extensions (circle : Type) (A B C D : Type) : Prop := sorry -- Same as above

theorem quadrilateral_diagonals_perpendicular
  (A B C D : Type)
  (h_convex : convex_quadrilateral A B C D)
  (incircle excircle : Type)
  (h_incircle : tangent_to_all_sides incircle A B C D)
  (h_excircle : tangent_to_all_extensions excircle A B C D) : 
  (⊥ : Prop) :=  -- statement indicating perpendicularity 
sorry

end quadrilateral_diagonals_perpendicular_l1778_177805


namespace number_of_rows_l1778_177848

-- Definitions of conditions
def tomatoes : ℕ := 3 * 5
def cucumbers : ℕ := 5 * 4
def potatoes : ℕ := 30
def additional_vegetables : ℕ := 85
def spaces_per_row : ℕ := 15

-- Total number of vegetables already planted
def planted_vegetables : ℕ := tomatoes + cucumbers + potatoes

-- Total capacity of the garden
def garden_capacity : ℕ := planted_vegetables + additional_vegetables

-- Number of rows in the garden
def rows_in_garden : ℕ := garden_capacity / spaces_per_row

theorem number_of_rows : rows_in_garden = 10 := by
  sorry

end number_of_rows_l1778_177848


namespace determine_remainder_l1778_177875

theorem determine_remainder (a b c : ℕ) (h1 : a < 7) (h2 : b < 7) (h3 : c < 7)
  (H1 : (a + 2 * b + 3 * c) % 7 = 1) 
  (H2 : (2 * a + 3 * b + c) % 7 = 2) 
  (H3 : (3 * a + b + 2 * c) % 7 = 1) : 
  (a * b * c) % 7 = 0 := 
sorry

end determine_remainder_l1778_177875


namespace stickers_started_with_l1778_177837

-- Definitions for the conditions
def stickers_given (Emily : ℕ) : Prop := Emily = 7
def stickers_ended_with (Willie_end : ℕ) : Prop := Willie_end = 43

-- The main proof statement
theorem stickers_started_with (Willie_start : ℕ) :
  stickers_given 7 →
  stickers_ended_with 43 →
  Willie_start = 43 - 7 :=
by
  intros h₁ h₂
  sorry

end stickers_started_with_l1778_177837


namespace lcm_factor_is_one_l1778_177844

theorem lcm_factor_is_one
  (A B : ℕ)
  (hcf : A.gcd B = 42)
  (larger_A : A = 588)
  (other_factor : ∃ X, A.lcm B = 42 * X * 14) :
  ∃ X, X = 1 :=
  sorry

end lcm_factor_is_one_l1778_177844


namespace g_inv_g_inv_14_l1778_177807

def g (x : ℝ) : ℝ := 5 * x - 3

noncomputable def g_inv (y : ℝ) : ℝ := (y + 3) / 5

theorem g_inv_g_inv_14 : g_inv (g_inv 14) = 32 / 25 :=
by
  sorry

end g_inv_g_inv_14_l1778_177807


namespace rectangle_perimeter_l1778_177847

theorem rectangle_perimeter (a b c d e f g : ℕ)
  (h1 : a + b + c = d)
  (h2 : d + e = g)
  (h3 : b + c = f)
  (h4 : c + f = g)
  (h5 : Nat.gcd (a + b + g) (d + e) = 1)
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (c_pos : 0 < c)
  (d_pos : 0 < d)
  (e_pos : 0 < e)
  (f_pos : 0 < f)
  (g_pos : 0 < g) :
  2 * (a + b + g + d + e) = 40 :=
sorry

end rectangle_perimeter_l1778_177847


namespace find_y_l1778_177882

theorem find_y (h1 : ∀ (a b : ℤ), a * b = (a - 1) * (b - 1)) (h2 : y * 12 = 110) : y = 11 :=
by
  sorry

end find_y_l1778_177882


namespace algebraic_expression_value_l1778_177867

theorem algebraic_expression_value 
  (x y : ℝ) 
  (h : 2 * x + y = 1) : 
  (y + 1) ^ 2 - (y ^ 2 - 4 * x + 4) = -1 := 
by 
  sorry

end algebraic_expression_value_l1778_177867


namespace smallest_three_digit_multiple_of_6_5_8_9_eq_360_l1778_177863

theorem smallest_three_digit_multiple_of_6_5_8_9_eq_360 :
  ∃ n : ℕ, n ≥ 100 ∧ n ≤ 999 ∧ (n % 6 = 0 ∧ n % 5 = 0 ∧ n % 8 = 0 ∧ n % 9 = 0) ∧ n = 360 := 
by
  sorry

end smallest_three_digit_multiple_of_6_5_8_9_eq_360_l1778_177863


namespace eval_expression_l1778_177866

theorem eval_expression : (2^2 - 2) - (3^2 - 3) + (4^2 - 4) - (5^2 - 5) + (6^2 - 6) = 18 :=
by
  sorry

end eval_expression_l1778_177866


namespace range_of_y_is_correct_l1778_177868

noncomputable def range_of_y (n : ℝ) : ℝ :=
  if n > 2 then 1 / n else 2 * n^2 + 1

theorem range_of_y_is_correct :
  (∀ n, 0 < range_of_y n ∧ range_of_y n < 1 / 2 ∧ n > 2) ∨ (∀ n, 1 ≤ range_of_y n ∧ n ≤ 2) :=
sorry

end range_of_y_is_correct_l1778_177868


namespace right_triangle_hypotenuse_l1778_177885

theorem right_triangle_hypotenuse (a b : ℕ) (h₁ : a = 75) (h₂ : b = 100) : ∃ c, c = 125 ∧ c^2 = a^2 + b^2 :=
by
  sorry

end right_triangle_hypotenuse_l1778_177885


namespace geometric_sequence_fifth_term_l1778_177810

theorem geometric_sequence_fifth_term (x y : ℝ) (r : ℝ) 
  (h1 : x + y ≠ 0) (h2 : x - y ≠ 0) (h3 : x ≠ 0) (h4 : y ≠ 0)
  (h_ratio_1 : (x - y) / (x + y) = r)
  (h_ratio_2 : (x^2 * y) / (x - y) = r)
  (h_ratio_3 : (x * y^2) / (x^2 * y) = r) :
  (x * y^2 * ((y / x) * r)) = y^3 := 
by 
  sorry

end geometric_sequence_fifth_term_l1778_177810


namespace proof_problem_l1778_177859

open Set

def Point : Type := ℝ × ℝ

structure Triangle :=
(A : Point)
(B : Point)
(C : Point)

def area_of_triangle (T : Triangle) : ℝ :=
   0.5 * abs ((T.B.1 - T.A.1) * (T.C.2 - T.A.2) - (T.C.1 - T.A.1) * (T.B.2 - T.A.2))

def area_of_grid (length width : ℝ) : ℝ :=
   length * width

def problem_statement : Prop :=
   let T : Triangle := {A := (1,3), B := (5,1), C := (4,4)} 
   let S1 := area_of_triangle T
   let S := area_of_grid 6 5
   (S1 / S) = 1 / 6

theorem proof_problem : problem_statement := 
by
  sorry


end proof_problem_l1778_177859


namespace find_original_production_planned_l1778_177834

-- Definition of the problem
variables (x : ℕ)
noncomputable def original_production_planned (x : ℕ) :=
  (6000 / (x + 500)) = (4500 / x)

-- The theorem to prove the original number planned is 1500
theorem find_original_production_planned (x : ℕ) (h : original_production_planned x) : x = 1500 :=
sorry

end find_original_production_planned_l1778_177834


namespace total_wheels_in_storage_l1778_177842

def wheels (n_bicycles n_tricycles n_unicycles n_quadbikes : ℕ) : ℕ :=
  (n_bicycles * 2) + (n_tricycles * 3) + (n_unicycles * 1) + (n_quadbikes * 4)

theorem total_wheels_in_storage :
  let n_bicycles := 24
  let n_tricycles := 14
  let n_unicycles := 10
  let n_quadbikes := 8
  wheels n_bicycles n_tricycles n_unicycles n_quadbikes = 132 :=
by
  let n_bicycles := 24
  let n_tricycles := 14
  let n_unicycles := 10
  let n_quadbikes := 8
  show wheels n_bicycles n_tricycles n_unicycles n_quadbikes = 132
  sorry

end total_wheels_in_storage_l1778_177842


namespace steves_earning_l1778_177830

variable (pounds_picked : ℕ → ℕ) -- pounds picked on day i: 0=Monday, 1=Tuesday, 2=Wednesday, 3=Thursday

def payment_per_pound : ℕ := 2

def total_money_made : ℕ :=
  (pounds_picked 0 * payment_per_pound) + 
  (pounds_picked 1 * payment_per_pound) + 
  (pounds_picked 2 * payment_per_pound) + 
  (pounds_picked 3 * payment_per_pound)

theorem steves_earning 
  (h0 : pounds_picked 0 = 8)
  (h1 : pounds_picked 1 = 3 * pounds_picked 0)
  (h2 : pounds_picked 2 = 0)
  (h3 : pounds_picked 3 = 18) : 
  total_money_made pounds_picked = 100 := by
  sorry

end steves_earning_l1778_177830


namespace percentage_increase_l1778_177873

theorem percentage_increase (A B x y : ℝ) (h1 : A / B = (5 * y^2) / (6 * x)) (h2 : 2 * x + 3 * y = 42) :  
  (B - A) / A * 100 = ((126 - 9 * y - 5 * y^2) / (5 * y^2)) * 100 :=
by
  sorry

end percentage_increase_l1778_177873


namespace alice_average_speed_l1778_177815

def average_speed (distance1 speed1 distance2 speed2 totalDistance totalTime : ℚ) :=
  totalDistance / totalTime

theorem alice_average_speed : 
  let d1 := 45
  let s1 := 15
  let d2 := 15
  let s2 := 45
  let totalDistance := d1 + d2
  let totalTime := (d1 / s1) + (d2 / s2)
  average_speed d1 s1 d2 s2 totalDistance totalTime = 18 :=
by
  sorry

end alice_average_speed_l1778_177815


namespace sellable_fruit_l1778_177820

theorem sellable_fruit :
  let total_oranges := 30 * 300
  let total_damaged_oranges := total_oranges * 10 / 100
  let sellable_oranges := total_oranges - total_damaged_oranges

  let total_nectarines := 45 * 80
  let nectarines_taken := 5 * 20
  let sellable_nectarines := total_nectarines - nectarines_taken

  let total_apples := 20 * 120
  let bad_apples := 50
  let sellable_apples := total_apples - bad_apples

  sellable_oranges + sellable_nectarines + sellable_apples = 13950 :=
by
  sorry

end sellable_fruit_l1778_177820


namespace parallelogram_height_base_difference_l1778_177824

theorem parallelogram_height_base_difference (A B H : ℝ) (hA : A = 24) (hB : B = 4) (hArea : A = B * H) :
  H - B = 2 := by
  sorry

end parallelogram_height_base_difference_l1778_177824


namespace f_2019_equals_neg2_l1778_177850

variable (f : ℝ → ℝ)
variable (h_odd : ∀ x : ℝ, f (-x) = -f x)
variable (h_period : ∀ x : ℝ, f (x + 4) = f x)
variable (h_defined : ∀ x : ℝ, 0 < x ∧ x < 2 → f x = 2 * x^2)

theorem f_2019_equals_neg2 : f 2019 = -2 :=
by 
  sorry

end f_2019_equals_neg2_l1778_177850


namespace green_disks_more_than_blue_l1778_177839

theorem green_disks_more_than_blue 
  (total_disks : ℕ) (blue_ratio yellow_ratio green_ratio red_ratio : ℕ)
  (h1 : total_disks = 132)
  (h2 : blue_ratio = 3)
  (h3 : yellow_ratio = 7)
  (h4 : green_ratio = 8)
  (h5 : red_ratio = 4)
  : 6 * green_ratio - 6 * blue_ratio = 30 :=
by
  sorry

end green_disks_more_than_blue_l1778_177839


namespace men_in_room_l1778_177888
noncomputable def numMenInRoom (x : ℕ) : ℕ := 4 * x + 2

theorem men_in_room (x : ℕ) (h_initial_ratio : true) (h_after_events : true) (h_double_women : 2 * (5 * x - 3) = 24) :
  numMenInRoom x = 14 :=
sorry

end men_in_room_l1778_177888


namespace relationship_x_a_b_l1778_177801

theorem relationship_x_a_b (x a b : ℝ) (h1 : x < b) (h2 : b < a) (h3 : a < 0) : 
  x^2 > a * b ∧ a * b > a^2 :=
by
  sorry

end relationship_x_a_b_l1778_177801


namespace polynomial_perfect_square_l1778_177877

theorem polynomial_perfect_square (k : ℝ) 
  (h : ∃ a : ℝ, x^2 + 8*x + k = (x + a)^2) : 
  k = 16 :=
by
  sorry

end polynomial_perfect_square_l1778_177877


namespace work_done_together_l1778_177832

theorem work_done_together
    (fraction_work_left : ℚ)
    (A_days : ℕ)
    (B_days : ℚ) :
    A_days = 20 →
    fraction_work_left = 2 / 3 →
    4 * (1 / 20 + 1 / B_days) = 1 / 3 →
    B_days = 30 := 
by
  intros hA hfrac heq
  sorry

end work_done_together_l1778_177832


namespace bus_system_carry_per_day_l1778_177846

theorem bus_system_carry_per_day (total_people : ℕ) (weeks : ℕ) (days_in_week : ℕ) (people_per_day : ℕ) :
  total_people = 109200000 →
  weeks = 13 →
  days_in_week = 7 →
  people_per_day = total_people / (weeks * days_in_week) →
  people_per_day = 1200000 :=
by
  intros htotal hweeks hdays hcalc
  sorry

end bus_system_carry_per_day_l1778_177846


namespace problem_proof_l1778_177808

theorem problem_proof (A B : ℝ) (h1 : A ≠ B) (h2 : ∀ x : ℝ, (A * (B * x^2 + A)^2 + B) - (B * (A * x^2 + B)^2 + A) = A^2 - B^2) :
  A^2 + B^2 = - (A * B) := 
sorry

end problem_proof_l1778_177808


namespace part1_part2_l1778_177822

-- Conditions and the equation of the circle
def circleCenterLine (a : ℝ) : Prop := ∃ y, y = a + 2
def circleRadius : ℝ := 2
def pointOnCircle (A : ℝ × ℝ) (a : ℝ) : Prop := (A.1 - a)^2 + (A.2 - (a + 2))^2 = circleRadius^2
def tangentToYAxis (a : ℝ) : Prop := abs a = circleRadius

-- Problem 1: Proving the equation of the circle C
def circleEq (x y a : ℝ) : Prop := (x - a)^2 + (y - (a + 2))^2 = circleRadius^2

theorem part1 (a : ℝ) (h : abs a = circleRadius) (h1 : pointOnCircle (2, 2) a) 
    (h2 : circleCenterLine a) : circleEq 2 0 2 := 
sorry

-- Conditions and the properties for Problem 2
def distSquared (P Q : ℝ × ℝ) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2
def QCondition (Q : ℝ × ℝ) : Prop := 
  distSquared Q (1, 3) - distSquared Q (1, 1) = 32
def onCircle (Q : ℝ × ℝ) (a : ℝ) : Prop := (Q.1 - a)^2 + (Q.2 - (a + 2))^2 = circleRadius^2

-- Problem 2: Proving the range of the abscissa a
theorem part2 (Q : ℝ × ℝ) (a : ℝ) 
    (hQ : QCondition Q) (hCircle : onCircle Q a) : 
    -3 ≤ a ∧ a ≤ 1 := 
sorry

end part1_part2_l1778_177822


namespace part1_intersection_part2_sufficient_not_necessary_l1778_177803

open Set

-- Definition of sets A and B
def set_A (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ a + 1}
def set_B : Set ℝ := {x | x ≤ -2 ∨ x ≥ 3}

-- Part (1)
theorem part1_intersection (a : ℝ) (h : a = -2) : set_A a ∩ set_B = {x | -3 ≤ x ∧ x ≤ -2} := by
  sorry

-- Part (2)
theorem part2_sufficient_not_necessary (p q : Prop) (hp : ∀ x, set_A a x → set_B x) (h_suff : p → q) (h_not_necess : ¬(q → p)) : set_A a ⊆ set_B → a ∈ Iic (-3) ∪ Ici 4 := by
  sorry

end part1_intersection_part2_sufficient_not_necessary_l1778_177803


namespace difference_in_circumferences_l1778_177876

theorem difference_in_circumferences (r_inner r_outer : ℝ) (h1 : r_inner = 15) (h2 : r_outer = r_inner + 8) : 
  2 * Real.pi * r_outer - 2 * Real.pi * r_inner = 16 * Real.pi :=
by
  rw [h1, h2]
  sorry

end difference_in_circumferences_l1778_177876


namespace find_ordered_triple_l1778_177862

theorem find_ordered_triple :
  ∃ (a b c : ℝ), a > 2 ∧ b > 2 ∧ c > 2 ∧
    (a + b + c = 30) ∧
    ( (a = 13) ∧ (b = 11) ∧ (c = 6) ) ∧
    ( ( ( (a + 3)^2 / (b + c - 3) ) + ( (b + 5)^2 / (c + a - 5) ) + ( (c + 7)^2 / (a + b - 7) ) = 45 ) ) :=
sorry

end find_ordered_triple_l1778_177862


namespace quadratic_inequality_l1778_177843

theorem quadratic_inequality (a : ℝ) (h : ∀ x : ℝ, x^2 - a * x + a > 0) : 0 < a ∧ a < 4 :=
sorry

end quadratic_inequality_l1778_177843


namespace company_profit_growth_l1778_177811

theorem company_profit_growth (x : ℝ) (h : 1.6 * (1 + x / 100)^2 = 2.5) : x = 25 :=
sorry

end company_profit_growth_l1778_177811


namespace solve_system_eq_l1778_177871

theorem solve_system_eq (x y z : ℝ) :
    (x^2 - y^2 + z = 64 / (x * y)) ∧
    (y^2 - z^2 + x = 64 / (y * z)) ∧
    (z^2 - x^2 + y = 64 / (x * z)) ↔ 
    (x = 4 ∧ y = 4 ∧ z = 4) ∨ 
    (x = -4 ∧ y = -4 ∧ z = 4) ∨ 
    (x = -4 ∧ y = 4 ∧ z = -4) ∨ 
    (x = 4 ∧ y = -4 ∧ z = -4) := by
  sorry

end solve_system_eq_l1778_177871


namespace right_angle_triangle_l1778_177861

theorem right_angle_triangle (a b c : ℝ) (h : (a + b) ^ 2 - c ^ 2 = 2 * a * b) : a ^ 2 + b ^ 2 = c ^ 2 := 
by
  sorry

end right_angle_triangle_l1778_177861


namespace problem_inequality_l1778_177857

variable (a b : ℝ)

theorem problem_inequality (h_pos : 0 < a) (h_pos' : 0 < b) (h_sum : a + b = 1) :
  (1 / a^2 - a^3) * (1 / b^2 - b^3) ≥ (31 / 8)^2 := 
  sorry

end problem_inequality_l1778_177857


namespace log_equation_solution_l1778_177899

theorem log_equation_solution (x : ℝ) (hx_pos : 0 < x) : 
  (Real.log x / Real.log 4) * (Real.log 9 / Real.log x) = Real.log 9 / Real.log 4 ↔ x ≠ 1 :=
by
  sorry

end log_equation_solution_l1778_177899


namespace find_initial_men_l1778_177813

noncomputable def initial_men_planned (M : ℕ) : Prop :=
  let initial_days := 10
  let additional_days := 20
  let total_days := initial_days + additional_days
  let men_sent := 25
  let initial_work := M * initial_days
  let remaining_men := M - men_sent
  let remaining_work := remaining_men * total_days
  initial_work = remaining_work 

theorem find_initial_men :
  ∃ M : ℕ, initial_men_planned M ∧ M = 38 :=
by
  have h : initial_men_planned 38 :=
    by
      sorry
  exact ⟨38, h, rfl⟩

end find_initial_men_l1778_177813


namespace max_z_value_l1778_177821

theorem max_z_value (x y z : ℝ) (h : x + y + z = 3) (h' : x * y + y * z + z * x = 2) : z ≤ 5 / 3 :=
  sorry


end max_z_value_l1778_177821


namespace eval_expr1_l1778_177809

theorem eval_expr1 : 
  ( (27 / 8) ^ (-2 / 3) - (49 / 9) ^ 0.5 + (0.008) ^ (-2 / 3) * (2 / 25) ) = 1 / 9 :=
by 
  sorry

end eval_expr1_l1778_177809


namespace calculate_120ab_l1778_177814

variable (a b : ℚ)

theorem calculate_120ab (h1 : 10 * a = 20) (h2 : 6 * b = 20) : 120 * (a * b) = 800 := by
  sorry

end calculate_120ab_l1778_177814


namespace central_angle_measure_l1778_177835

theorem central_angle_measure (α r : ℝ) (h1 : α * r = 2) (h2 : 1/2 * α * r^2 = 2) : α = 1 := 
sorry

end central_angle_measure_l1778_177835


namespace find_monthly_income_l1778_177845

-- Given condition
def deposit : ℝ := 3400
def percentage : ℝ := 0.15

-- Goal: Prove Sheela's monthly income
theorem find_monthly_income : (deposit / percentage) = 22666.67 := by
  -- Skip the proof for now
  sorry

end find_monthly_income_l1778_177845


namespace remainder_when_divided_by_x_plus_2_l1778_177896

-- Define the polynomial q(x) = D*x^4 + E*x^2 + F*x + 8
variable (D E F : ℝ)
def q (x : ℝ) : ℝ := D * x^4 + E * x^2 + F * x + 8

-- Given condition: q(2) = 12
axiom h1 : q D E F 2 = 12

-- Prove that q(-2) = 4
theorem remainder_when_divided_by_x_plus_2 : q D E F (-2) = 4 := by
  sorry

end remainder_when_divided_by_x_plus_2_l1778_177896


namespace has_exactly_two_solutions_iff_l1778_177874

theorem has_exactly_two_solutions_iff (a : ℝ) :
  (∃! x : ℝ, x^2 + 2 * x + 2 * (|x + 1|) = a) ↔ a > -1 :=
sorry

end has_exactly_two_solutions_iff_l1778_177874


namespace line_equation_exists_l1778_177878

noncomputable def P : ℝ × ℝ := (-2, 5)
noncomputable def m : ℝ := -3 / 4

theorem line_equation_exists (x y : ℝ) : 
  (y - 5 = -3 / 4 * (x + 2)) ↔ (3 * x + 4 * y - 14 = 0) := 
by 
  sorry

end line_equation_exists_l1778_177878


namespace minimum_value_of_f_l1778_177804

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x ^ 3 - x + (1 / 3)

theorem minimum_value_of_f :
  (∃ m : ℝ, ∀ x : ℝ, f x ≤ 1) → (∀ x : ℝ, f 1 = -(1 / 3)) :=
by
  sorry

end minimum_value_of_f_l1778_177804


namespace quadratic_root_difference_l1778_177818

theorem quadratic_root_difference (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ + x₂ = 2 ∧ x₁ * x₂ = a ∧ (x₁ - x₂)^2 = 20) → a = -4 := 
by
  sorry

end quadratic_root_difference_l1778_177818


namespace no_solution_for_t_and_s_l1778_177879

theorem no_solution_for_t_and_s (m : ℝ) :
  (¬∃ t s : ℝ, (1 + 7 * t = -3 + 2 * s) ∧ (3 - 5 * t = 4 + m * s)) ↔ m = -10 / 7 :=
by
  sorry

end no_solution_for_t_and_s_l1778_177879


namespace num_dinosaur_dolls_l1778_177892

-- Define the number of dinosaur dolls
def dinosaur_dolls : Nat := 3

-- Define the theorem to prove the number of dinosaur dolls
theorem num_dinosaur_dolls : dinosaur_dolls = 3 := by
  -- Add sorry to skip the proof
  sorry

end num_dinosaur_dolls_l1778_177892


namespace operation_eval_l1778_177891

def my_operation (a b : ℤ) := a * (b + 2) + a * (b + 1)

theorem operation_eval : my_operation 3 (-1) = 3 := by
  sorry

end operation_eval_l1778_177891


namespace find_missing_number_l1778_177869

theorem find_missing_number (x : ℝ) (h : 0.25 / x = 2 / 6) : x = 0.75 :=
by
  sorry

end find_missing_number_l1778_177869


namespace solve_mod_equation_l1778_177890

theorem solve_mod_equation (y b n : ℤ) (h1 : 15 * y + 4 ≡ 7 [ZMOD 18]) (h2 : y ≡ b [ZMOD n]) (h3 : 2 ≤ n) (h4 : b < n) : b + n = 11 :=
sorry

end solve_mod_equation_l1778_177890


namespace combined_weight_is_18442_l1778_177802

noncomputable def combined_weight_proof : ℝ :=
  let elephant_weight_tons := 3
  let donkey_weight_percentage := 0.1
  let giraffe_weight_tons := 1.5
  let hippopotamus_weight_kg := 4000
  let elephant_food_oz := 16
  let donkey_food_lbs := 5
  let giraffe_food_kg := 3
  let hippopotamus_food_g := 5000

  let ton_to_pounds := 2000
  let kg_to_pounds := 2.20462
  let oz_to_pounds := 1 / 16
  let g_to_pounds := 0.00220462

  let elephant_weight_pounds := elephant_weight_tons * ton_to_pounds
  let donkey_weight_pounds := (1 - donkey_weight_percentage) * elephant_weight_pounds
  let giraffe_weight_pounds := giraffe_weight_tons * ton_to_pounds
  let hippopotamus_weight_pounds := hippopotamus_weight_kg * kg_to_pounds

  let elephant_food_pounds := elephant_food_oz * oz_to_pounds
  let giraffe_food_pounds := giraffe_food_kg * kg_to_pounds
  let hippopotamus_food_pounds := hippopotamus_food_g * g_to_pounds

  elephant_weight_pounds + donkey_weight_pounds + giraffe_weight_pounds + hippopotamus_weight_pounds +
  elephant_food_pounds + donkey_food_lbs + giraffe_food_pounds + hippopotamus_food_pounds

theorem combined_weight_is_18442 : combined_weight_proof = 18442 := by
  sorry

end combined_weight_is_18442_l1778_177802


namespace largest_integer_inequality_l1778_177829

theorem largest_integer_inequality (x : ℤ) (h : 10 - 3 * x > 25) : x = -6 :=
sorry

end largest_integer_inequality_l1778_177829


namespace isosceles_triangle_perimeter_l1778_177840

theorem isosceles_triangle_perimeter
  (x y : ℝ)
  (h : |x - 3| + (y - 1)^2 = 0)
  (isosceles_triangle : ∃ a b c, (a = x ∧ b = x ∧ c = y) ∨ (a = x ∧ b = y ∧ c = y) ∨ (a = y ∧ b = y ∧ c = x)):
  ∃ perimeter : ℝ, perimeter = 7 :=
by
  sorry

end isosceles_triangle_perimeter_l1778_177840


namespace sheila_attends_picnic_l1778_177826

theorem sheila_attends_picnic :
  let probRain := 0.30
  let probSunny := 0.50
  let probCloudy := 0.20
  let probAttendIfRain := 0.15
  let probAttendIfSunny := 0.85
  let probAttendIfCloudy := 0.40
  (probRain * probAttendIfRain + probSunny * probAttendIfSunny + probCloudy * probAttendIfCloudy) = 0.55 :=
by sorry

end sheila_attends_picnic_l1778_177826


namespace calc_probability_10_or_9_ring_calc_probability_less_than_9_ring_l1778_177858

def probability_10_ring : ℝ := 0.13
def probability_9_ring : ℝ := 0.28
def probability_8_ring : ℝ := 0.31

def probability_10_or_9_ring : ℝ := probability_10_ring + probability_9_ring

def probability_less_than_9_ring : ℝ := 1 - probability_10_or_9_ring

theorem calc_probability_10_or_9_ring :
  probability_10_or_9_ring = 0.41 :=
by
  sorry

theorem calc_probability_less_than_9_ring :
  probability_less_than_9_ring = 0.59 :=
by
  sorry

end calc_probability_10_or_9_ring_calc_probability_less_than_9_ring_l1778_177858


namespace sprinter_time_no_wind_l1778_177851

theorem sprinter_time_no_wind :
  ∀ (x y : ℝ), (90 / (x + y) = 10) → (70 / (x - y) = 10) → x = 8 * y → 100 / x = 12.5 :=
by
  intros x y h1 h2 h3
  sorry

end sprinter_time_no_wind_l1778_177851


namespace guo_can_pay_exact_amount_l1778_177806

-- Define the denominations and total amount Guo has
def note_denominations := [1, 10, 20, 50]
def total_amount := 20000
def cost_computer := 10000

-- The main theorem stating that Guo can pay exactly 10000 yuan
theorem guo_can_pay_exact_amount : ∃ bills : List ℕ, ∀ (b : ℕ), b ∈ bills → b ∈ note_denominations ∧
  bills.sum = cost_computer :=
sorry

end guo_can_pay_exact_amount_l1778_177806


namespace cos_neg_17pi_over_4_l1778_177828

noncomputable def cos_value : ℝ := (Real.pi / 4).cos

theorem cos_neg_17pi_over_4 :
  (Real.cos (-17 * Real.pi / 4)) = cos_value :=
by
  -- Define even property of cosine and angle simplification
  sorry

end cos_neg_17pi_over_4_l1778_177828


namespace sum_abc_geq_half_l1778_177819

theorem sum_abc_geq_half (a b c : ℝ) (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) (h_nonneg_c : 0 ≤ c) 
(h_abs_sum : |a - b| + |b - c| + |c - a| = 1) : 
a + b + c ≥ 0.5 := 
sorry

end sum_abc_geq_half_l1778_177819


namespace statue_of_liberty_ratio_l1778_177898

theorem statue_of_liberty_ratio :
  let H_statue := 305 -- height in feet
  let H_model := 10 -- height in inches
  H_statue / H_model = 30.5 := 
by
  let H_statue := 305
  let H_model := 10
  sorry

end statue_of_liberty_ratio_l1778_177898


namespace chord_length_of_concentric_circles_l1778_177897

theorem chord_length_of_concentric_circles 
  (R r : ℝ) (h1 : R^2 - r^2 = 15) (h2 : ∀ s, s = 2 * R) :
  ∃ c : ℝ, c = 2 * Real.sqrt 15 ∧ ∀ x, x = c := 
by 
  sorry

end chord_length_of_concentric_circles_l1778_177897


namespace maximum_piles_l1778_177831

theorem maximum_piles (n : ℕ) (h : n = 660) : 
  ∃ m, m = 30 ∧ 
       ∀ (piles : Finset ℕ), (piles.sum id = n) →
       (∀ x ∈ piles, ∀ y ∈ piles, x ≤ y → y < 2 * x) → 
       (piles.card ≤ m) :=
by
  sorry

end maximum_piles_l1778_177831


namespace compare_three_and_negfour_l1778_177817

theorem compare_three_and_negfour : 3 > -4 := by
  sorry

end compare_three_and_negfour_l1778_177817


namespace eric_return_home_time_l1778_177887

-- Definitions based on conditions
def time_running_to_park : ℕ := 20
def time_jogging_to_park : ℕ := 10
def trip_to_park_time : ℕ := time_running_to_park + time_jogging_to_park
def return_time_multiplier : ℕ := 3

-- Statement of the problem
theorem eric_return_home_time : 
  return_time_multiplier * trip_to_park_time = 90 :=
by 
  -- Skipping proof steps
  sorry

end eric_return_home_time_l1778_177887


namespace inequality_proof_l1778_177884

theorem inequality_proof 
  (a b c : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) : 
  a ^ a * b ^ b * c ^ c ≥ 1 / (a * b * c) := 
sorry

end inequality_proof_l1778_177884


namespace train_cross_time_l1778_177881

theorem train_cross_time (length_of_train : ℕ) (speed_in_kmh : ℕ) (conversion_factor : ℕ) (speed_in_mps : ℕ) (time : ℕ) :
  length_of_train = 120 →
  speed_in_kmh = 72 →
  conversion_factor = 1000 / 3600 →
  speed_in_mps = speed_in_kmh * conversion_factor →
  time = length_of_train / speed_in_mps →
  time = 6 :=
by
  intros hlength hspeed hconversion hspeed_mps htime
  have : conversion_factor = 5 / 18 := sorry
  have : speed_in_mps = 20 := sorry
  exact sorry

end train_cross_time_l1778_177881


namespace sequence_general_term_l1778_177895

namespace SequenceSum

def Sn (n : ℕ) : ℕ :=
  2 * n^2 + n

def a₁ (n : ℕ) : ℕ :=
  if n = 1 then Sn n else (Sn n - Sn (n - 1))

theorem sequence_general_term (n : ℕ) (hn : n > 0) : 
  a₁ n = 4 * n - 1 :=
sorry

end SequenceSum

end sequence_general_term_l1778_177895


namespace theater_ticket_sales_l1778_177860

theorem theater_ticket_sales (x y : ℕ) (h1 : x + y = 175) (h2 : 6 * x + 2 * y = 750) : y = 75 :=
sorry

end theater_ticket_sales_l1778_177860


namespace power_eq_l1778_177883

open Real

theorem power_eq {x : ℝ} (h : x^3 + 4 * x = 8) : x^7 + 64 * x^2 = 128 :=
by
  sorry

end power_eq_l1778_177883


namespace question_d_not_true_l1778_177870

variable {a b c d : ℚ}

theorem question_d_not_true (h : a * b = c * d) : (a + 1) / (c + 1) ≠ (d + 1) / (b + 1) := 
sorry

end question_d_not_true_l1778_177870


namespace bob_pennies_l1778_177825

variable (a b : ℕ)

theorem bob_pennies : 
  (b + 2 = 4 * (a - 2)) →
  (b - 3 = 3 * (a + 3)) →
  b = 78 :=
by
  intros h1 h2
  sorry

end bob_pennies_l1778_177825


namespace diego_can_carry_home_l1778_177853

theorem diego_can_carry_home (T W G O A : ℕ) (hT : T = 20) (hW : W = 1) (hG : G = 1) (hO : O = 1) : A = T - (W + G + O) → A = 17 := by
  sorry

end diego_can_carry_home_l1778_177853


namespace total_number_of_meetings_proof_l1778_177812

-- Define the conditions in Lean
variable (A B : Type)
variable (starting_time : ℕ)
variable (location_A location_B : A × B)

-- Define speeds
variable (speed_A speed_B : ℕ)

-- Define meeting counts
variable (total_meetings : ℕ)

-- Define A reaches point B 2015 times
variable (A_reaches_B_2015 : Prop)

-- Define that B travels twice as fast as A
axiom speed_ratio : speed_B = 2 * speed_A

-- Define that A reaches point B for the 5th time when B reaches it for the 9th time
axiom meeting_times : A_reaches_B_2015 → (total_meetings = 6044)

-- The Lean statement to prove
theorem total_number_of_meetings_proof : A_reaches_B_2015 → total_meetings = 6044 := by
  sorry

end total_number_of_meetings_proof_l1778_177812


namespace fraction_multiplication_exponent_l1778_177852

theorem fraction_multiplication_exponent :
  ( (8 : ℚ) / 9 )^2 * ( (1 : ℚ) / 3 )^2 = (64 / 729 : ℚ) := 
by
  sorry

end fraction_multiplication_exponent_l1778_177852


namespace remainder_is_3_l1778_177836

theorem remainder_is_3 (x y r : ℕ) (h1 : x = 7 * y + r) (h2 : 2 * x = 18 * y + 2) (h3 : 11 * y - x = 1)
  (hrange : 0 ≤ r ∧ r < 7) : r = 3 := 
sorry

end remainder_is_3_l1778_177836
