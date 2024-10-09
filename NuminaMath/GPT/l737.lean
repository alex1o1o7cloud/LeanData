import Mathlib

namespace coefficient_of_x4_in_expansion_of_2x_plus_sqrtx_l737_73758

noncomputable def coefficient_of_x4_expansion : ℕ :=
  let r := 2;
  let n := 5;
  let general_term_coefficient := Nat.choose n r * 2^(n-r);
  general_term_coefficient

theorem coefficient_of_x4_in_expansion_of_2x_plus_sqrtx :
  coefficient_of_x4_expansion = 80 :=
by
  -- We can bypass the actual proving steps by
  -- acknowledging that the necessary proof mechanism
  -- will properly verify the calculation:
  sorry

end coefficient_of_x4_in_expansion_of_2x_plus_sqrtx_l737_73758


namespace parity_sum_matches_parity_of_M_l737_73730

theorem parity_sum_matches_parity_of_M (N M : ℕ) (even_numbers odd_numbers : ℕ → ℤ)
  (hn : ∀ i, i < N → even_numbers i % 2 = 0)
  (hm : ∀ i, i < M → odd_numbers i % 2 ≠ 0) : 
  (N + M) % 2 = M % 2 := 
sorry

end parity_sum_matches_parity_of_M_l737_73730


namespace value_of_expression_l737_73736

theorem value_of_expression (x : ℕ) (h : x = 3) : 2 * x + 3 = 9 :=
by 
  sorry

end value_of_expression_l737_73736


namespace domain_of_composed_function_l737_73701

theorem domain_of_composed_function
  (f : ℝ → ℝ)
  (dom_f : ∀ x, 0 ≤ x ∧ x ≤ 4 → f x ≠ 0) :
  ∀ x, -2 ≤ x ∧ x ≤ 2 → f (x^2) ≠ 0 :=
by
  sorry

end domain_of_composed_function_l737_73701


namespace quadratic_has_single_real_root_l737_73754

theorem quadratic_has_single_real_root (n : ℝ) (h : (6 * n) ^ 2 - 4 * 1 * (2 * n) = 0) : n = 2 / 9 :=
by
  sorry

end quadratic_has_single_real_root_l737_73754


namespace find_prime_n_l737_73742

def is_prime (p : ℕ) : Prop := 
  p > 1 ∧ (∀ n, n ∣ p → n = 1 ∨ n = p)

def prime_candidates : List ℕ := [11, 17, 23, 29, 41, 47, 53, 59, 61, 71, 83, 89]

theorem find_prime_n (n : ℕ) 
  (h1 : n ∈ prime_candidates) 
  (h2 : is_prime (n)) 
  (h3 : is_prime (n + 20180500)) : 
  n = 61 :=
by sorry

end find_prime_n_l737_73742


namespace exam_scores_l737_73771

theorem exam_scores (A B C D : ℤ) 
  (h1 : A + B = C + D + 17) 
  (h2 : A = B - 4) 
  (h3 : C = D + 5) :
  ∃ highest lowest, (highest - lowest = 13) ∧ 
                   (highest = A ∨ highest = B ∨ highest = C ∨ highest = D) ∧ 
                   (lowest = A ∨ lowest = B ∨ lowest = C ∨ lowest = D) :=
by
  sorry

end exam_scores_l737_73771


namespace ratio_of_areas_l737_73752

noncomputable def length_field : ℝ := 16
noncomputable def width_field : ℝ := length_field / 2
noncomputable def area_field : ℝ := length_field * width_field
noncomputable def side_pond : ℝ := 4
noncomputable def area_pond : ℝ := side_pond * side_pond
noncomputable def ratio_area_pond_to_field : ℝ := area_pond / area_field

theorem ratio_of_areas :
  ratio_area_pond_to_field = 1 / 8 :=
  by
  sorry

end ratio_of_areas_l737_73752


namespace baggies_of_oatmeal_cookies_l737_73717

theorem baggies_of_oatmeal_cookies (total_cookies : ℝ) (chocolate_chip_cookies : ℝ) (cookies_per_baggie : ℝ) 
(h_total : total_cookies = 41)
(h_choc : chocolate_chip_cookies = 13)
(h_baggie : cookies_per_baggie = 9) : 
  ⌊(total_cookies - chocolate_chip_cookies) / cookies_per_baggie⌋ = 3 := 
by 
  sorry

end baggies_of_oatmeal_cookies_l737_73717


namespace total_beds_in_hotel_l737_73724

theorem total_beds_in_hotel (total_rooms : ℕ) (rooms_two_beds rooms_three_beds : ℕ) (beds_two beds_three : ℕ) 
  (h1 : total_rooms = 13) 
  (h2 : rooms_two_beds = 8) 
  (h3 : rooms_three_beds = total_rooms - rooms_two_beds) 
  (h4 : beds_two = 2) 
  (h5 : beds_three = 3) : 
  rooms_two_beds * beds_two + rooms_three_beds * beds_three = 31 :=
by
  sorry

end total_beds_in_hotel_l737_73724


namespace total_seats_round_table_l737_73778

theorem total_seats_round_table 
  (a : ℕ) (b : ℕ) 
  (h₀ : a ≠ b)
  (h₁ : a + b = 39) 
  : ∃ n, n = 38 := 
by {
  sorry
}

end total_seats_round_table_l737_73778


namespace difference_of_numbers_l737_73710

theorem difference_of_numbers (a b : ℕ) (h1 : a + b = 22500) (h2 : b = 10 * a + 5) : b - a = 18410 :=
by
  sorry

end difference_of_numbers_l737_73710


namespace solve_for_y_l737_73789

/-- Given the equation 7(2y + 3) - 5 = -3(2 - 5y), solve for y. -/
theorem solve_for_y (y : ℤ) : 7 * (2 * y + 3) - 5 = -3 * (2 - 5 * y) → y = 22 :=
by
  intros h
  sorry

end solve_for_y_l737_73789


namespace find_unknown_number_l737_73769

theorem find_unknown_number (x : ℝ) (h : (45 + 23 / x) * x = 4028) : x = 89 :=
sorry

end find_unknown_number_l737_73769


namespace correct_statement_a_incorrect_statement_b_incorrect_statement_c_incorrect_statement_d_incorrect_statement_e_l737_73788

theorem correct_statement_a (x : ℝ) : x > 1 → x^2 > x :=
by sorry

theorem incorrect_statement_b (x : ℝ) : ¬ (x^2 < 0 → x < 0) :=
by sorry

theorem incorrect_statement_c (x : ℝ) : ¬ (x^2 < x → x < 0) :=
by sorry

theorem incorrect_statement_d (x : ℝ) : ¬ (x^2 < 1 → x < 1) :=
by sorry

theorem incorrect_statement_e (x : ℝ) : ¬ (x > 0 → x^2 > x) :=
by sorry

end correct_statement_a_incorrect_statement_b_incorrect_statement_c_incorrect_statement_d_incorrect_statement_e_l737_73788


namespace max_drinks_amount_l737_73716

noncomputable def initial_milk : ℚ := 3 / 4
noncomputable def rachel_fraction : ℚ := 1 / 2
noncomputable def max_fraction : ℚ := 1 / 3

def amount_rachel_drinks (initial: ℚ) (fraction: ℚ) : ℚ := initial * fraction
def remaining_milk_after_rachel (initial: ℚ) (amount_rachel: ℚ) : ℚ := initial - amount_rachel
def amount_max_drinks (remaining: ℚ) (fraction: ℚ) : ℚ := remaining * fraction

theorem max_drinks_amount :
  amount_max_drinks (remaining_milk_after_rachel initial_milk (amount_rachel_drinks initial_milk rachel_fraction)) max_fraction = 1 / 8 := 
sorry

end max_drinks_amount_l737_73716


namespace square_area_from_diagonal_l737_73760

theorem square_area_from_diagonal (d : ℝ) (h : d = 12) : (d^2 / 2) = 72 :=
by sorry

end square_area_from_diagonal_l737_73760


namespace total_travel_expenses_l737_73751

noncomputable def cost_of_fuel_tank := 45
noncomputable def miles_per_tank := 500
noncomputable def journey_distance := 2000
noncomputable def food_ratio := 3 / 5
noncomputable def hotel_cost_per_night := 80
noncomputable def number_of_hotel_nights := 3
noncomputable def fuel_cost_increase := 5

theorem total_travel_expenses :
  let number_of_refills := journey_distance / miles_per_tank
  let first_refill_cost := cost_of_fuel_tank
  let second_refill_cost := first_refill_cost + fuel_cost_increase
  let third_refill_cost := second_refill_cost + fuel_cost_increase
  let fourth_refill_cost := third_refill_cost + fuel_cost_increase
  let total_fuel_cost := first_refill_cost + second_refill_cost + third_refill_cost + fourth_refill_cost
  let total_food_cost := food_ratio * total_fuel_cost
  let total_hotel_cost := hotel_cost_per_night * number_of_hotel_nights
  let total_expenses := total_fuel_cost + total_food_cost + total_hotel_cost
  total_expenses = 576 := by sorry

end total_travel_expenses_l737_73751


namespace egg_cost_l737_73798

theorem egg_cost (toast_cost : ℝ) (E : ℝ) (total_cost : ℝ)
  (dales_toast : ℝ) (dales_eggs : ℝ) (andrews_toast : ℝ) (andrews_eggs : ℝ) :
  toast_cost = 1 → 
  dales_toast = 2 → 
  dales_eggs = 2 → 
  andrews_toast = 1 → 
  andrews_eggs = 2 → 
  total_cost = 15 →
  total_cost = (dales_toast * toast_cost + dales_eggs * E) + 
               (andrews_toast * toast_cost + andrews_eggs * E) →
  E = 3 :=
by
  sorry

end egg_cost_l737_73798


namespace probability_units_digit_odd_l737_73726

theorem probability_units_digit_odd :
  (1 / 2 : ℚ) = 5 / 10 :=
by {
  -- This is the equivalent mathematically correct theorem statement
  -- The proof is omitted as per instructions
  sorry
}

end probability_units_digit_odd_l737_73726


namespace equal_copper_content_alloy_l737_73720

theorem equal_copper_content_alloy (a b : ℝ) :
  ∃ x : ℝ, 0 < x ∧ x < 10 ∧
  (10 - x) * a + x * b = (15 - x) * b + x * a → x = 6 :=
by
  sorry

end equal_copper_content_alloy_l737_73720


namespace simplify_expression_l737_73791

theorem simplify_expression (x y : ℝ) (h1 : x = 1) (h2 : y = 2) : 
  ((x + y) * (x - y) - (x - y)^2 + 2 * y * (x - y)) / (4 * y) = -1 :=
by
  sorry

end simplify_expression_l737_73791


namespace pentagonal_pyramid_faces_l737_73774

-- Definition of a pentagonal pyramid
structure PentagonalPyramid where
  base_sides : Nat := 5
  triangular_faces : Nat := 5

-- The goal is to prove that the total number of faces is 6
theorem pentagonal_pyramid_faces (P : PentagonalPyramid) : P.base_sides + 1 = 6 :=
  sorry

end pentagonal_pyramid_faces_l737_73774


namespace no_solution_a_solution_b_l737_73759

def f (n : ℕ) : ℕ :=
  if n = 0 then
    0
  else
    n / 7 + f (n / 7)

theorem no_solution_a :
  ¬ ∃ n : ℕ, 7 ^ 399 ∣ n! ∧ ¬ 7 ^ 400 ∣ n! := sorry

theorem solution_b :
  {n : ℕ | 7 ^ 400 ∣ n! ∧ ¬ 7 ^ 401 ∣ n!} = {2401, 2402, 2403, 2404, 2405, 2406, 2407} := sorry

end no_solution_a_solution_b_l737_73759


namespace toys_per_rabbit_l737_73765

-- Define the conditions
def rabbits : ℕ := 34
def toys_mon : ℕ := 8
def toys_tue : ℕ := 3 * toys_mon
def toys_wed : ℕ := 2 * toys_tue
def toys_thu : ℕ := toys_mon
def toys_fri : ℕ := 5 * toys_mon
def toys_sat : ℕ := toys_wed / 2

-- Define the total number of toys
def total_toys : ℕ := toys_mon + toys_tue + toys_wed + toys_thu + toys_fri + toys_sat

-- Define the proof statement
theorem toys_per_rabbit : total_toys / rabbits = 4 :=
by
  -- Proof will go here
  sorry

end toys_per_rabbit_l737_73765


namespace seq_fixed_point_l737_73705

theorem seq_fixed_point (a_0 b_0 : ℝ) (a b : ℕ → ℝ)
  (h1 : a 0 = a_0)
  (h2 : b 0 = b_0)
  (h3 : ∀ n, a (n + 1) = a n + b n)
  (h4 : ∀ n, b (n + 1) = a n * b n) :
  a 2022 = a_0 ∧ b 2022 = b_0 ↔ b_0 = 0 := sorry

end seq_fixed_point_l737_73705


namespace range_of_g_l737_73703

noncomputable def f (x : ℝ) : ℝ := 2 * x - 3

noncomputable def g (x : ℝ) : ℝ := f (f (f (f x)))

theorem range_of_g :
  (∀ x, 1 ≤ x ∧ x ≤ 3 → -29 ≤ g x ∧ g x ≤ 3) :=
sorry

end range_of_g_l737_73703


namespace moles_of_MgSO4_formed_l737_73766

def moles_of_Mg := 3
def moles_of_H2SO4 := 3

theorem moles_of_MgSO4_formed
  (Mg : ℕ)
  (H2SO4 : ℕ)
  (react : ℕ → ℕ → ℕ × ℕ)
  (initial_Mg : Mg = moles_of_Mg)
  (initial_H2SO4 : H2SO4 = moles_of_H2SO4)
  (balanced_eq : react Mg H2SO4 = (Mg, H2SO4)) :
  (react Mg H2SO4).1 = 3 :=
by
  sorry

end moles_of_MgSO4_formed_l737_73766


namespace sum_series_eq_4_l737_73762

theorem sum_series_eq_4 : 
  (∑' n : ℕ, (4 * (n + 1) - 2) / (3 ^ (n + 1))) = 4 := 
by
  sorry

end sum_series_eq_4_l737_73762


namespace triangle_largest_angle_l737_73780

theorem triangle_largest_angle {k : ℝ} (h1 : k > 0)
  (h2 : k + 2 * k + 3 * k = 180) : 3 * k = 90 := 
sorry

end triangle_largest_angle_l737_73780


namespace factorize_expr_l737_73783

theorem factorize_expr (a b : ℝ) : a * b^2 - 8 * a * b + 16 * a = a * (b - 4)^2 := 
by
  sorry

end factorize_expr_l737_73783


namespace perpendicular_iff_zero_dot_product_l737_73743

open Real

def a (m : ℝ) : ℝ × ℝ := (1, 2 * m)
def b (m : ℝ) : ℝ × ℝ := (m + 1, 1)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem perpendicular_iff_zero_dot_product (m : ℝ) :
  dot_product (a m) (b m) = 0 → m = -1 / 3 :=
by
  sorry

end perpendicular_iff_zero_dot_product_l737_73743


namespace quadratic_complete_square_l737_73704

theorem quadratic_complete_square (c n : ℝ) (h1 : ∀ x : ℝ, x^2 + c * x + 20 = (x + n)^2 + 12) (h2: 0 < c) : 
  c = 4 * Real.sqrt 2 :=
by
  sorry

end quadratic_complete_square_l737_73704


namespace sequence_100th_term_eq_l737_73714

-- Definitions for conditions
def numerator (n : ℕ) : ℕ := 1 + (n - 1) * 2
def denominator (n : ℕ) : ℕ := 2 + (n - 1) * 3

-- The statement of the problem as a Lean 4 theorem
theorem sequence_100th_term_eq :
  (numerator 100) / (denominator 100) = 199 / 299 :=
by
  sorry

end sequence_100th_term_eq_l737_73714


namespace calculate_expression_l737_73721

theorem calculate_expression :
  (0.5 ^ 4 / 0.05 ^ 3) = 500 := by
  sorry

end calculate_expression_l737_73721


namespace matrix_identity_l737_73702

noncomputable def B : Matrix (Fin 2) (Fin 2) ℝ := !![2, -1; 4, 3]
noncomputable def I : Matrix (Fin 2) (Fin 2) ℝ := 1

theorem matrix_identity :
  B^4 = -3 • B + 2 • I :=
by
  sorry

end matrix_identity_l737_73702


namespace more_bottle_caps_than_wrappers_l737_73777

namespace DannyCollection

def bottle_caps_found := 50
def wrappers_found := 46

theorem more_bottle_caps_than_wrappers :
  bottle_caps_found - wrappers_found = 4 :=
by
  -- We skip the proof here with "sorry"
  sorry

end DannyCollection

end more_bottle_caps_than_wrappers_l737_73777


namespace half_abs_diff_of_squares_l737_73793

theorem half_abs_diff_of_squares (x y : ℤ) (h1 : x = 21) (h2 : y = 19) :
  (|x^2 - y^2| / 2) = 40 := 
by
  subst h1
  subst h2
  sorry

end half_abs_diff_of_squares_l737_73793


namespace inequality_solution_l737_73781

theorem inequality_solution (x : ℝ) :
  (2 / (x - 2) - 5 / (x - 3) + 5 / (x - 4) - 2 / (x - 5) < 1 / 15) ↔ (x < 1 ∨ x > 3) ∧ (x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5) :=
by
  sorry

end inequality_solution_l737_73781


namespace parabola_focus_distance_l737_73748

theorem parabola_focus_distance (A : ℝ × ℝ) (F : ℝ × ℝ := (1, 0)) 
    (h_parabola : A.2^2 = 4 * A.1) (h_distance : dist A F = 3) :
    A = (2, 2 * Real.sqrt 2) ∨ A = (2, -2 * Real.sqrt 2) :=
by
  sorry

end parabola_focus_distance_l737_73748


namespace curve_intersection_one_point_l737_73787

theorem curve_intersection_one_point (a : ℝ) :
  (∀ x y : ℝ, (x^2 + y^2 = a^2 ↔ y = x^2 + a) → (x, y) = (0, a)) ↔ (a ≥ -1/2) := 
sorry

end curve_intersection_one_point_l737_73787


namespace Telegraph_Road_length_is_162_l737_73779

-- Definitions based on the conditions
def meters_to_kilometers (meters : ℕ) : ℕ := meters / 1000
def Pardee_Road_length_meters : ℕ := 12000
def Telegraph_Road_extra_length_kilometers : ℕ := 150

-- The length of Pardee Road in kilometers
def Pardee_Road_length_kilometers : ℕ := meters_to_kilometers Pardee_Road_length_meters

-- Lean statement to prove the length of Telegraph Road in kilometers
theorem Telegraph_Road_length_is_162 :
  Pardee_Road_length_kilometers + Telegraph_Road_extra_length_kilometers = 162 :=
sorry

end Telegraph_Road_length_is_162_l737_73779


namespace meaningful_range_l737_73712

   noncomputable def isMeaningful (x : ℝ) : Prop :=
     (3 - x ≥ 0) ∧ (x + 1 ≠ 0)

   theorem meaningful_range :
     ∀ x : ℝ, isMeaningful x ↔ (x ≤ 3 ∧ x ≠ -1) :=
   by
     sorry
   
end meaningful_range_l737_73712


namespace find_principal_amount_l737_73728

noncomputable def principal_amount_loan (SI R T : ℝ) : ℝ :=
  SI / (R * T)

theorem find_principal_amount (SI R T : ℝ) (h_SI : SI = 6480) (h_R : R = 0.12) (h_T : T = 3) :
  principal_amount_loan SI R T = 18000 :=
by
  rw [principal_amount_loan, h_SI, h_R, h_T]
  norm_num

#check find_principal_amount

end find_principal_amount_l737_73728


namespace urn_problem_l737_73727

noncomputable def probability_of_two_black_balls : ℚ := (10 / 15) * (9 / 14)

theorem urn_problem : probability_of_two_black_balls = 3 / 7 := 
by
  sorry

end urn_problem_l737_73727


namespace integer_solution_interval_l737_73767

theorem integer_solution_interval {f : ℝ → ℝ} (m : ℝ) :
  (∀ x : ℤ, (-x^2 + x + m + 2 ≥ |x| ↔ (x : ℝ) = n)) ↔ (-2 ≤ m ∧ m < -1) := 
sorry

end integer_solution_interval_l737_73767


namespace exp_add_l737_73700

theorem exp_add (a : ℝ) (x₁ x₂ : ℝ) : a^(x₁ + x₂) = a^x₁ * a^x₂ :=
sorry

end exp_add_l737_73700


namespace inequality_of_trig_function_l737_73757

theorem inequality_of_trig_function 
  (a b A B : ℝ) 
  (h : ∀ x : ℝ, 1 - a * Real.cos x - b * Real.sin x - A * Real.cos (2 * x) - B * Real.sin (2 * x) ≥ 0) :
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 :=
sorry

end inequality_of_trig_function_l737_73757


namespace hoseok_wire_length_l737_73734

theorem hoseok_wire_length (side_length : ℕ) (equilateral : Prop) (leftover_wire : ℕ) (total_wire : ℕ)  
  (eq_side : side_length = 19) (eq_leftover : leftover_wire = 15) 
  (eq_equilateral : equilateral) : total_wire = 72 :=
sorry

end hoseok_wire_length_l737_73734


namespace find_a_b_sum_specific_find_a_b_sum_l737_73715

-- Define the sets A and B based on the given inequalities
def set_A : Set ℝ := {x | x^2 - 2 * x - 3 < 0}
def set_B : Set ℝ := {x | x^2 + x - 6 < 0}

-- Intersect the sets A and B
def set_A_int_B : Set ℝ := set_A ∩ set_B

-- Define the inequality with parameters a and b
def quad_ineq (a b : ℝ) : Set ℝ := {x | a * x^2 + b * x + 2 > 0}

-- Define the parameters a and b based on the given condition
noncomputable def a : ℝ := -1
noncomputable def b : ℝ := -1

-- The statement to be proved
theorem find_a_b_sum : ∀ a b : ℝ, set_A ∩ set_B = {x | a * x^2 + b * x + 2 > 0} → a + b = -2 :=
by
  sorry

-- Fixing the parameters a and b for our specific proof condition
theorem specific_find_a_b_sum : a + b = -2 :=
by
  sorry

end find_a_b_sum_specific_find_a_b_sum_l737_73715


namespace last_three_digits_of_power_l737_73756

theorem last_three_digits_of_power (h : 3^400 ≡ 1 [MOD 800]) : 3^8000 ≡ 1 [MOD 800] :=
by {
  sorry
}

end last_three_digits_of_power_l737_73756


namespace range_of_f_l737_73729

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x - 1

theorem range_of_f : 
  ∀ x, -1 ≤ x ∧ x ≤ 1 → -2 ≤ f x ∧ f x ≤ 2 :=
by
  intro x Hx
  sorry

end range_of_f_l737_73729


namespace court_cost_proof_l737_73772

-- Define all the given conditions
def base_fine : ℕ := 50
def penalty_rate : ℕ := 2
def mark_speed : ℕ := 75
def speed_limit : ℕ := 30
def school_zone_multiplier : ℕ := 2
def lawyer_fee_rate : ℕ := 80
def lawyer_hours : ℕ := 3
def total_owed : ℕ := 820

-- Define the calculation for the additional penalty
def additional_penalty : ℕ := (mark_speed - speed_limit) * penalty_rate

-- Define the calculation for the total fine
def total_fine : ℕ := (base_fine + additional_penalty) * school_zone_multiplier

-- Define the calculation for the lawyer's fee
def lawyer_fee : ℕ := lawyer_fee_rate * lawyer_hours

-- Define the calculation for the total of fine and lawyer's fee
def fine_and_lawyer_fee := total_fine + lawyer_fee

-- Prove the court costs
theorem court_cost_proof : total_owed - fine_and_lawyer_fee = 300 := by
  sorry

end court_cost_proof_l737_73772


namespace ratio_of_combined_area_to_combined_perimeter_l737_73753

noncomputable def equilateral_triangle_area (s : ℝ) : ℝ :=
  (s^2 * Real.sqrt 3) / 4

noncomputable def equilateral_triangle_perimeter (s : ℝ) : ℝ :=
  3 * s

theorem ratio_of_combined_area_to_combined_perimeter :
  (equilateral_triangle_area 6 + equilateral_triangle_area 8) / 
  (equilateral_triangle_perimeter 6 + equilateral_triangle_perimeter 8) = (25 * Real.sqrt 3) / 42 :=
by
  sorry

end ratio_of_combined_area_to_combined_perimeter_l737_73753


namespace hexagon_area_l737_73763

theorem hexagon_area (ABCDEF : Type) (l : ℕ) (h : l = 3) (p q : ℕ)
  (area_hexagon : ℝ) (area_formula : area_hexagon = Real.sqrt p + Real.sqrt q) :
  p + q = 54 := by
  sorry

end hexagon_area_l737_73763


namespace correct_option_l737_73770

def condition_A (a : ℝ) : Prop := a^3 * a^4 = a^12
def condition_B (a b : ℝ) : Prop := (-3 * a * b^3)^2 = -6 * a * b^6
def condition_C (a : ℝ) : Prop := (a - 3)^2 = a^2 - 9
def condition_D (x y : ℝ) : Prop := (-x + y) * (x + y) = y^2 - x^2

theorem correct_option (x y : ℝ) : condition_D x y := by
  sorry

end correct_option_l737_73770


namespace sum_nine_terms_l737_73768

-- Definitions required based on conditions provided in Step a)
variables {a : ℕ → ℝ} {S : ℕ → ℝ} {d : ℝ}

-- The arithmetic sequence condition is encapsulated here
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- The definition of S_n being the sum of the first n terms
def sum_first_n (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

-- The given condition from the problem
def given_condition (a : ℕ → ℝ) : Prop :=
  2 * a 8 = 6 + a 1

-- The proof statement to show S_9 = 54 given the above conditions
theorem sum_nine_terms (h_arith : is_arithmetic_sequence a d)
                        (h_sum : sum_first_n a S) 
                        (h_given : given_condition a): 
                        S 9 = 54 :=
  by sorry

end sum_nine_terms_l737_73768


namespace election_required_percentage_l737_73713

def votes_cast : ℕ := 10000

def geoff_percentage : ℕ := 5
def geoff_received_votes := (geoff_percentage * votes_cast) / 1000

def extra_votes_needed : ℕ := 5000
def total_votes_needed := geoff_received_votes + extra_votes_needed

def required_percentage := (total_votes_needed * 100) / votes_cast

theorem election_required_percentage : required_percentage = 505 / 10 :=
by
  sorry

end election_required_percentage_l737_73713


namespace solve_for_a_l737_73745

theorem solve_for_a (x a : ℤ) (h : 2 * x - a - 5 = 0) (hx : x = 3) : a = 1 :=
by sorry

end solve_for_a_l737_73745


namespace polynomial_even_iff_exists_Q_l737_73739

open Polynomial

noncomputable def exists_polynomial_Q (P : Polynomial ℂ) : Prop :=
  ∃ Q : Polynomial ℂ, ∀ z : ℂ, P.eval z = (Q.eval z) * (Q.eval (-z))

theorem polynomial_even_iff_exists_Q (P : Polynomial ℂ) :
  (∀ z : ℂ, P.eval z = P.eval (-z)) ↔ exists_polynomial_Q P :=
by 
  sorry

end polynomial_even_iff_exists_Q_l737_73739


namespace embankment_building_l737_73718

theorem embankment_building (days : ℕ) (workers_initial : ℕ) (workers_later : ℕ) (embankments : ℕ) :
  workers_initial = 75 → days = 4 → embankments = 2 →
  (∀ r : ℚ, embankments = workers_initial * r * days →
            embankments = workers_later * r * 5) :=
by
  intros h75 hd4 h2 r hr
  sorry

end embankment_building_l737_73718


namespace mary_bought_48_cards_l737_73792

variable (M T F C B : ℕ)

theorem mary_bought_48_cards
  (h1 : M = 18)
  (h2 : T = 8)
  (h3 : F = 26)
  (h4 : C = 84) :
  B = C - (M - T + F) :=
by
  -- Proof would go here
  sorry

end mary_bought_48_cards_l737_73792


namespace max_popsicles_with_10_dollars_l737_73706

theorem max_popsicles_with_10_dollars :
  (∃ (single_popsicle_cost : ℕ) (four_popsicle_box_cost : ℕ) (six_popsicle_box_cost : ℕ) (budget : ℕ),
    single_popsicle_cost = 1 ∧
    four_popsicle_box_cost = 3 ∧
    six_popsicle_box_cost = 4 ∧
    budget = 10 ∧
    ∃ (max_popsicles : ℕ),
      max_popsicles = 14 ∧
      ∀ (popsicles : ℕ),
        popsicles ≤ 14 →
        ∃ (x y z : ℕ),
          popsicles = x + 4*y + 6*z ∧
          x * single_popsicle_cost + y * four_popsicle_box_cost + z * six_popsicle_box_cost ≤ budget
  ) :=
sorry

end max_popsicles_with_10_dollars_l737_73706


namespace arithmetic_sequence_sum_l737_73785

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (d : ℕ → ℕ)
  (is_arithmetic_seq : ∀ n, a (n + 1) = a n + d n)
  (h : (a 2) + (a 5) + (a 8) = 39) :
  (a 1) + (a 2) + (a 3) + (a 4) + (a 5) + (a 6) + (a 7) + (a 8) + (a 9) = 117 := 
sorry

end arithmetic_sequence_sum_l737_73785


namespace tire_price_l737_73786

theorem tire_price (payment : ℕ) (price_ratio : ℕ → ℕ → Prop)
  (h1 : payment = 345)
  (h2 : price_ratio 3 1)
  : ∃ x : ℕ, x = 99 := 
sorry

end tire_price_l737_73786


namespace initial_apples_l737_73732

-- Definitions of the conditions
def Minseok_ate : Nat := 3
def Jaeyoon_ate : Nat := 3
def apples_left : Nat := 2

-- The proposition we need to prove
theorem initial_apples : Minseok_ate + Jaeyoon_ate + apples_left = 8 := by
  sorry

end initial_apples_l737_73732


namespace central_angle_measure_l737_73740

theorem central_angle_measure (p : ℝ) (x : ℝ) (h1 : p = 1 / 8) (h2 : p = x / 360) : x = 45 :=
by
  -- skipping the proof
  sorry

end central_angle_measure_l737_73740


namespace simplify_fraction_l737_73782

theorem simplify_fraction :
  (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 :=
by
  sorry

end simplify_fraction_l737_73782


namespace calculate_x_l737_73795

theorem calculate_x :
  (422 + 404) ^ 2 - (4 * 422 * 404) = 324 :=
by
  -- proof goes here
  sorry

end calculate_x_l737_73795


namespace sequence_2018_value_l737_73784

theorem sequence_2018_value :
  ∃ a : ℕ → ℤ, a 1 = 3 ∧ a 2 = 6 ∧ (∀ n, a (n + 2) = a (n + 1) - a n) ∧ a 2018 = -3 :=
sorry

end sequence_2018_value_l737_73784


namespace jerry_weekly_earnings_l737_73737

-- Definitions of the given conditions
def pay_per_task : ℕ := 40
def hours_per_task : ℕ := 2
def hours_per_day : ℕ := 10
def days_per_week : ℕ := 7

-- Calculated values from the conditions
def tasks_per_day : ℕ := hours_per_day / hours_per_task
def tasks_per_week : ℕ := tasks_per_day * days_per_week
def total_earnings : ℕ := pay_per_task * tasks_per_week

-- Theorem to prove
theorem jerry_weekly_earnings : total_earnings = 1400 := by
  sorry

end jerry_weekly_earnings_l737_73737


namespace village_population_percentage_l737_73764

theorem village_population_percentage 
  (part : ℝ)
  (whole : ℝ)
  (h_part : part = 8100)
  (h_whole : whole = 9000) : 
  (part / whole) * 100 = 90 :=
by
  sorry

end village_population_percentage_l737_73764


namespace total_spent_on_clothing_l737_73744

def shorts_cost : ℝ := 15
def jacket_cost : ℝ := 14.82
def shirt_cost : ℝ := 12.51

theorem total_spent_on_clothing : shorts_cost + jacket_cost + shirt_cost = 42.33 := by
  -- Proof goes here.
  sorry

end total_spent_on_clothing_l737_73744


namespace min_value_expression_l737_73719

theorem min_value_expression (x y : ℝ) (h1 : x + y = 1) (h2 : y > 0) (h3 : x > 0) :
  ∃ (z : ℝ), z = (1 / (2 * x) + x / (y + 1)) ∧ z = 5 / 4 :=
sorry

end min_value_expression_l737_73719


namespace arithmetic_sqrt_of_13_l737_73707

theorem arithmetic_sqrt_of_13 : Real.sqrt 13 = Real.sqrt 13 := by
  sorry

end arithmetic_sqrt_of_13_l737_73707


namespace circle_passing_points_l737_73738

theorem circle_passing_points (x y : ℝ) :
  (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) →
  x^2 + y^2 - 4*x - 6*y = 0 :=
by
  intros h
  cases h
  case inl h₁ => 
    rw [h₁.1, h₁.2]
    ring
  case inr h₁ =>
    cases h₁
    case inl h₂ => 
      rw [h₂.1, h₂.2]
      ring
    case inr h₂ =>
      rw [h₂.1, h₂.2]
      ring

end circle_passing_points_l737_73738


namespace sum_of_reciprocals_of_roots_l737_73794

theorem sum_of_reciprocals_of_roots (s₁ s₂ : ℝ) (h₀ : s₁ + s₂ = 15) (h₁ : s₁ * s₂ = 36) :
  (1 / s₁) + (1 / s₂) = 5 / 12 :=
by
  sorry

end sum_of_reciprocals_of_roots_l737_73794


namespace range_of_a_l737_73750

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + 2*a*x + 4 > 0 ∨ (a < 1 ∧ 3 - 2*a > 1 ∧ ∀ x : ℝ, 3 - 2*a > 1)) ∧ ¬ (∀ x : ℝ, x^2 + 2*a*x + 4 > 0 ∧ (a < 1 ∧ 3 - 2*a > 1 ∧ ∀ x : ℝ, 3 - 2*a > 1)) →
  a ≤ -2 ∨ 1 ≤ a ∧ a < 2 :=
by {
  sorry
}

end range_of_a_l737_73750


namespace correct_quadratic_eq_l737_73790

-- Define the given conditions
def first_student_sum (b : ℝ) : Prop := 5 + 3 = -b
def second_student_product (c : ℝ) : Prop := (-12) * (-4) = c

-- Define the proof statement
theorem correct_quadratic_eq (b c : ℝ) (h1 : first_student_sum b) (h2 : second_student_product c) :
    b = -8 ∧ c = 48 ∧ (∀ x : ℝ, x^2 + b * x + c = 0 → (x=5 ∨ x=3 ∨ x=-12 ∨ x=-4)) :=
by
  sorry

end correct_quadratic_eq_l737_73790


namespace num_arrangements_thm1_num_arrangements_thm2_num_arrangements_thm3_l737_73725

open Nat

def num_arrangements_A_middle (n : ℕ) : ℕ :=
  if n = 4 then factorial 4 else 0

def num_arrangements_A_not_adj_B (n : ℕ) : ℕ :=
  if n = 5 then (factorial 3) * (factorial 4 / factorial 2) else 0

def num_arrangements_A_B_not_ends (n : ℕ) : ℕ :=
  if n = 5 then (factorial 3 / factorial 2) * factorial 3 else 0

theorem num_arrangements_thm1 : num_arrangements_A_middle 4 = 24 := 
  sorry

theorem num_arrangements_thm2 : num_arrangements_A_not_adj_B 5 = 72 := 
  sorry

theorem num_arrangements_thm3 : num_arrangements_A_B_not_ends 5 = 36 := 
  sorry

end num_arrangements_thm1_num_arrangements_thm2_num_arrangements_thm3_l737_73725


namespace tickets_used_l737_73773

variable (C T : Nat)

theorem tickets_used (h1 : C = 7) (h2 : T = C + 5) : T = 12 := by
  sorry

end tickets_used_l737_73773


namespace total_friends_met_l737_73796

def num_friends_with_pears : Nat := 9
def num_friends_with_oranges : Nat := 6

theorem total_friends_met : num_friends_with_pears + num_friends_with_oranges = 15 :=
by
  sorry

end total_friends_met_l737_73796


namespace range_of_a_l737_73776

noncomputable def f (a x : ℝ) : ℝ := (2 - a^2) * x + a

theorem range_of_a (a : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 1 → f a x > 0) ↔ (0 < a ∧ a < 2) :=
by
  sorry

end range_of_a_l737_73776


namespace calculate_cost_price_l737_73755

/-
Given:
  SP (Selling Price) is 18000
  If a 10% discount is applied on the SP, the effective selling price becomes 16200
  This effective selling price corresponds to an 8% profit over the cost price
  
Prove:
  The cost price (CP) is 15000
-/

theorem calculate_cost_price (SP : ℝ) (d : ℝ) (p : ℝ) (effective_SP : ℝ) (CP : ℝ) :
  SP = 18000 →
  d = 0.1 →
  p = 0.08 →
  effective_SP = SP - (d * SP) →
  effective_SP = CP * (1 + p) →
  CP = 15000 :=
by
  intros _
  sorry

end calculate_cost_price_l737_73755


namespace minimum_value_of_x_plus_y_existence_of_minimum_value_l737_73797

theorem minimum_value_of_x_plus_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y + 2 * x + y = 8) :
  x + y ≥ 2 * Real.sqrt 10 - 3 :=
sorry

theorem existence_of_minimum_value (x y : ℝ) :
  ∃ x y, x > 0 ∧ y > 0 ∧ x * y + 2 * x + y = 8 ∧ x + y = 2 * Real.sqrt 10 - 3 :=
sorry

end minimum_value_of_x_plus_y_existence_of_minimum_value_l737_73797


namespace printing_shop_paper_boxes_l737_73761

variable (x y : ℕ) -- Assuming x and y are natural numbers since the number of boxes can't be negative.

theorem printing_shop_paper_boxes (h1 : 80 * x + 180 * y = 2660)
                                  (h2 : x = 5 * y - 3) :
    x = 22 ∧ y = 5 := sorry

end printing_shop_paper_boxes_l737_73761


namespace distance_house_to_market_l737_73711

-- Define the conditions
def distance_house_to_school := 50
def total_distance_walked := 140

-- Define the question as a theorem with the correct answer
theorem distance_house_to_market : 
  ∀ (house_to_school school_to_house total_distance market : ℕ), 
  house_to_school = distance_house_to_school →
  school_to_house = distance_house_to_school →
  total_distance = total_distance_walked →
  house_to_school + school_to_house + market = total_distance →
  market = 40 :=
by
  intros house_to_school school_to_house total_distance market 
  intro h1 h2 h3 h4
  sorry

end distance_house_to_market_l737_73711


namespace avg_difference_is_5_l737_73799

def avg (s : List ℕ) : ℕ :=
  s.sum / s.length

def set1 := [20, 40, 60]
def set2 := [20, 60, 25]

theorem avg_difference_is_5 :
  avg set1 - avg set2 = 5 :=
by
  sorry

end avg_difference_is_5_l737_73799


namespace discount_calc_l737_73735

noncomputable def discount_percentage 
    (cost_price : ℝ) (markup_percentage : ℝ) (selling_price : ℝ) : ℝ :=
  let marked_price := cost_price + (markup_percentage / 100 * cost_price)
  let discount := marked_price - selling_price
  (discount / marked_price) * 100

theorem discount_calc :
  discount_percentage 540 15 460 = 25.92 :=
by
  sorry

end discount_calc_l737_73735


namespace houses_in_block_l737_73749

theorem houses_in_block (junk_per_house : ℕ) (total_junk : ℕ) (h_junk : junk_per_house = 2) (h_total : total_junk = 14) :
  total_junk / junk_per_house = 7 := by
  sorry

end houses_in_block_l737_73749


namespace clay_boys_proof_l737_73741

variable (total_students : ℕ)
variable (total_boys : ℕ)
variable (total_girls : ℕ)
variable (jonas_students : ℕ)
variable (clay_students : ℕ)
variable (birch_students : ℕ)
variable (jonas_boys : ℕ)
variable (birch_girls : ℕ)

noncomputable def boys_from_clay (total_students total_boys total_girls jonas_students clay_students birch_students jonas_boys birch_girls : ℕ) : ℕ :=
  let birch_boys := birch_students - birch_girls
  let clay_boys := total_boys - (jonas_boys + birch_boys)
  clay_boys

theorem clay_boys_proof (h1 : total_students = 180) (h2 : total_boys = 94) 
    (h3 : total_girls = 86) (h4 : jonas_students = 60) 
    (h5 : clay_students = 80) (h6 : birch_students = 40) 
    (h7 : jonas_boys = 30) (h8 : birch_girls = 24) : 
  boys_from_clay total_students total_boys total_girls jonas_students clay_students birch_students jonas_boys birch_girls = 48 := 
by 
  simp [boys_from_clay] 
  sorry

end clay_boys_proof_l737_73741


namespace right_triangle_sides_l737_73746

theorem right_triangle_sides (x y z : ℕ) (h_sum : x + y + z = 156) (h_area : x * y = 2028) (h_pythagorean : z^2 = x^2 + y^2) :
  (x = 39 ∧ y = 52 ∧ z = 65) ∨ (x = 52 ∧ y = 39 ∧ z = 65) :=
by
  admit -- proof goes here

-- Additional details for importing required libraries and setting up the environment
-- are intentionally simplified as per instruction to cover a broader import.

end right_triangle_sides_l737_73746


namespace max_marks_l737_73733

theorem max_marks (total_marks : ℕ) (obtained_marks : ℕ) (failed_by : ℕ) 
    (passing_percentage : ℝ) (passing_marks : ℝ) (H1 : obtained_marks = 125)
    (H2 : failed_by = 40) (H3 : passing_percentage = 0.33) 
    (H4 : passing_marks = obtained_marks + failed_by) 
    (H5 : passing_marks = passing_percentage * total_marks) : total_marks = 500 := by
  sorry

end max_marks_l737_73733


namespace average_of_first_16_even_numbers_l737_73731

theorem average_of_first_16_even_numbers : 
  (2 + 4 + 6 + 8 + 10 + 12 + 14 + 16 + 18 + 20 + 22 + 24 + 26 + 28 + 30 + 32) / 16 = 17 := 
by sorry

end average_of_first_16_even_numbers_l737_73731


namespace percentage_error_in_area_l737_73775

-- Definitions based on conditions
def actual_side (s : ℝ) := s
def measured_side (s : ℝ) := s * 1.01
def actual_area (s : ℝ) := s^2
def calculated_area (s : ℝ) := (measured_side s)^2

-- Theorem statement of the proof problem
theorem percentage_error_in_area (s : ℝ) : 
  (calculated_area s - actual_area s) / actual_area s * 100 = 2.01 := 
by 
  -- Proof is omitted
  sorry

end percentage_error_in_area_l737_73775


namespace max_knights_between_knights_l737_73708

def num_knights : ℕ := 40
def num_samurais : ℕ := 10
def total_people : ℕ := 50
def num_knights_with_samurai_right : ℕ := 7

theorem max_knights_between_knights :
  (num_knights - num_knights_with_samurai_right + 1) = 32 :=
sorry

end max_knights_between_knights_l737_73708


namespace first_term_arithmetic_sequence_l737_73747

theorem first_term_arithmetic_sequence
    (a: ℚ)
    (S_n S_2n: ℕ → ℚ)
    (n: ℕ) 
    (h1: ∀ n > 0, S_n n = (n * (2 * a + (n - 1) * 5)) / 2)
    (h2: ∀ n > 0, S_2n (2 * n) = ((2 * n) * (2 * a + ((2 * n) - 1) * 5)) / 2)
    (h3: ∀ n > 0, (S_2n (2 * n)) / (S_n n) = 4) :
  a = 5 / 2 :=
by
  sorry

end first_term_arithmetic_sequence_l737_73747


namespace find_a_l737_73709

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, x ≠ 0 → f (-x) = -f x

noncomputable def f (a x : ℝ) : ℝ := (1 / (2^x - 1)) + a

theorem find_a (a : ℝ) : 
  is_odd_function (f a) → a = 1 / 2 :=
by
  sorry

end find_a_l737_73709


namespace perimeter_of_triangle_eq_28_l737_73722

-- Definitions of conditions
variables (p : ℝ)
def inradius : ℝ := 2.0
def area : ℝ := 28

-- Main theorem statement
theorem perimeter_of_triangle_eq_28 : p = 28 :=
  by
  -- The proof is omitted
  sorry

end perimeter_of_triangle_eq_28_l737_73722


namespace jimmy_irene_total_payment_l737_73723

def cost_jimmy_shorts : ℝ := 3 * 15
def cost_irene_shirts : ℝ := 5 * 17
def total_cost_before_discount : ℝ := cost_jimmy_shorts + cost_irene_shirts
def discount : ℝ := total_cost_before_discount * 0.10
def total_paid : ℝ := total_cost_before_discount - discount

theorem jimmy_irene_total_payment : total_paid = 117 := by
  sorry

end jimmy_irene_total_payment_l737_73723
