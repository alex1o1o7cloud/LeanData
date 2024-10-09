import Mathlib

namespace sanity_proof_l378_37805

-- Define the characters and their sanity status as propositions
variables (Griffin QuasiTurtle Lobster : Prop)

-- Conditions
axiom Lobster_thinks : (Griffin ∧ ¬QuasiTurtle ∧ ¬Lobster) ∨ (¬Griffin ∧ QuasiTurtle ∧ ¬Lobster) ∨ (¬Griffin ∧ ¬QuasiTurtle ∧ Lobster)
axiom QuasiTurtle_thinks : Griffin

-- Statement to prove
theorem sanity_proof : ¬Griffin ∧ ¬QuasiTurtle ∧ ¬Lobster :=
by {
  sorry
}

end sanity_proof_l378_37805


namespace extra_interest_l378_37813

def principal : ℝ := 7000
def rate1 : ℝ := 0.18
def rate2 : ℝ := 0.12
def time : ℝ := 2

def interest (P R T : ℝ) : ℝ := P * R * T

theorem extra_interest :
  interest principal rate1 time - interest principal rate2 time = 840 := by
  sorry

end extra_interest_l378_37813


namespace solve_expression_l378_37828

theorem solve_expression :
  ( (12.05 * 5.4 + 0.6) / (2.3 - 1.8) * (7/3) - (4.07 * 3.5 + 0.45) ^ 2) = 90.493 := 
by 
  sorry

end solve_expression_l378_37828


namespace olympiad_not_possible_l378_37803

theorem olympiad_not_possible (x : ℕ) (y : ℕ) (h1 : x + y = 1000) (h2 : y = x + 43) : false := by
  sorry

end olympiad_not_possible_l378_37803


namespace find_z_l378_37835

open Complex

theorem find_z (z : ℂ) (h1 : (z + 2 * I).im = 0) (h2 : ((z / (2 - I)).im = 0)) : z = 4 - 2 * I :=
by
  sorry

end find_z_l378_37835


namespace sum_abc_l378_37899

theorem sum_abc (a b c : ℝ) 
  (h : (a - 6)^2 + (b - 3)^2 + (c - 2)^2 = 0) : 
  a + b + c = 11 := 
by 
  sorry

end sum_abc_l378_37899


namespace jeans_price_increase_l378_37871

theorem jeans_price_increase (manufacturing_cost customer_price : ℝ) 
  (h1 : customer_price = 1.40 * (1.40 * manufacturing_cost))
  : (customer_price - manufacturing_cost) / manufacturing_cost * 100 = 96 :=
by sorry

end jeans_price_increase_l378_37871


namespace find_a1_and_d_l378_37850

-- Defining the arithmetic sequence and its properties
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Given conditions
def conditions (a : ℕ → ℤ) (a_1 : ℤ) (d : ℤ) : Prop :=
  (a 4 + a 5 + a 6 + a 7 = 56) ∧ (a 4 * a 7 = 187) ∧ (a 1 = a_1) ∧ is_arithmetic_sequence a d

-- Proving the solution
theorem find_a1_and_d :
  ∃ (a : ℕ → ℤ) (a_1 d : ℤ),
    conditions a a_1 d ∧ ((a_1 = 5 ∧ d = 2) ∨ (a_1 = 23 ∧ d = -2)) :=
by
  sorry

end find_a1_and_d_l378_37850


namespace range_of_a_l378_37884

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (2 * a * x + 3 * x > 2 * a + 3) ↔ (x < 1)) → (a < -3 / 2) :=
by
  intro h
  sorry

end range_of_a_l378_37884


namespace shell_count_l378_37877

theorem shell_count (initial_shells : ℕ) (ed_limpet : ℕ) (ed_oyster : ℕ) (ed_conch : ℕ) (jacob_extra : ℕ)
  (h1 : initial_shells = 2)
  (h2 : ed_limpet = 7) 
  (h3 : ed_oyster = 2) 
  (h4 : ed_conch = 4) 
  (h5 : jacob_extra = 2) : 
  (initial_shells + ed_limpet + ed_oyster + ed_conch + (ed_limpet + ed_oyster + ed_conch + jacob_extra)) = 30 := 
by 
  sorry

end shell_count_l378_37877


namespace total_journey_distance_l378_37862

variable (D : ℝ) (T : ℝ) (v₁ : ℝ) (v₂ : ℝ)

theorem total_journey_distance :
  T = 10 → 
  v₁ = 21 → 
  v₂ = 24 → 
  (T = (D / (2 * v₁)) + (D / (2 * v₂))) → 
  D = 224 :=
by
  intros hT hv₁ hv₂ hDistance
  -- Proof goes here
  sorry

end total_journey_distance_l378_37862


namespace triangle_area_l378_37804

theorem triangle_area (a b c : ℕ) (h1 : a = 9) (h2 : b = 12) (h3 : c = 15) (right_triangle : a^2 + b^2 = c^2) : 
  (1 / 2) * a * b = 54 := by
    sorry

end triangle_area_l378_37804


namespace circle_value_in_grid_l378_37872

theorem circle_value_in_grid :
  ∃ (min_circle_val : ℕ), min_circle_val = 21 ∧ (∀ (max_circle_val : ℕ), ∃ (L : ℕ), L > max_circle_val) :=
by
  sorry

end circle_value_in_grid_l378_37872


namespace solve_diamond_l378_37865

theorem solve_diamond (d : ℕ) (hd : d < 10) (h : d * 9 + 6 = d * 10 + 3) : d = 3 :=
sorry

end solve_diamond_l378_37865


namespace max_capacity_tank_l378_37812

-- Definitions of the conditions
def water_loss_1 := 32000 * 5
def water_loss_2 := 10000 * 10
def total_loss := water_loss_1 + water_loss_2
def water_added := 40000 * 3
def missing_water := 140000

-- Definition of the maximum capacity
def max_capacity := total_loss + water_added + missing_water

-- The theorem to prove
theorem max_capacity_tank : max_capacity = 520000 := by
  sorry

end max_capacity_tank_l378_37812


namespace problem_part1_problem_part2_l378_37853

-- Statement for Part (1)
theorem problem_part1 (a b : ℝ) (h_sol : {x : ℝ | ax^2 - 3 * x + 2 > 0} = {x : ℝ | x < 1 ∨ x > 2}) :
  a = 1 ∧ b = 2 := sorry

-- Statement for Part (2)
theorem problem_part2 (x y k : ℝ) (hx_pos : x > 0) (hy_pos : y > 0) 
  (h_eq : (1 / x) + (2 / y) = 1) (h_ineq : 2 * x + y ≥ k^2 + k + 2) :
  -3 ≤ k ∧ k ≤ 2 := sorry

end problem_part1_problem_part2_l378_37853


namespace option_D_correct_l378_37814

-- Formal statement in Lean 4
theorem option_D_correct (m : ℝ) : 6 * m + (-2 - 10 * m) = -4 * m - 2 :=
by
  -- Proof is skipped per instruction
  sorry

end option_D_correct_l378_37814


namespace part_I_part_II_l378_37868

noncomputable def f (x : ℝ) : ℝ := 2 * |x + 1| + |2 * x - 1|

theorem part_I (x : ℝ) : 
  (f x > f 1) ↔ (x < -3/2 ∨ x > 1) :=
by
  -- We leave the proof as sorry to indicate it needs to be filled in
  sorry

theorem part_II (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∀ x : ℝ, f x ≥ 1/m + 1/n) → m + n ≥ 4/3 :=
by
  -- We leave the proof as sorry to indicate it needs to be filled in
  sorry

end part_I_part_II_l378_37868


namespace ones_digit_9_pow_53_l378_37896

theorem ones_digit_9_pow_53 :
  (9 ^ 53) % 10 = 9 :=
by
  sorry

end ones_digit_9_pow_53_l378_37896


namespace number_reduced_by_10_eq_0_09_three_point_two_four_increased_to_three_two_four_zero_l378_37867

theorem number_reduced_by_10_eq_0_09 : ∃ (x : ℝ), x / 10 = 0.09 ∧ x = 0.9 :=
sorry

theorem three_point_two_four_increased_to_three_two_four_zero : ∃ (y : ℝ), 3.24 * y = 3240 ∧ y = 1000 :=
sorry

end number_reduced_by_10_eq_0_09_three_point_two_four_increased_to_three_two_four_zero_l378_37867


namespace circle_radius_is_7_5_l378_37847

noncomputable def radius_of_circle (side_length : ℝ) : ℝ := sorry

theorem circle_radius_is_7_5 :
  radius_of_circle 12 = 7.5 := sorry

end circle_radius_is_7_5_l378_37847


namespace bicycle_speed_l378_37818

theorem bicycle_speed (x : ℝ) :
  (10 / x = 10 / (2 * x) + 1 / 3) → x = 15 :=
by
  intro h
  sorry

end bicycle_speed_l378_37818


namespace max_participants_win_at_least_three_matches_l378_37879

theorem max_participants_win_at_least_three_matches (n : ℕ) (h : n = 200) : 
  ∃ k : ℕ, k = 66 ∧ ∀ m : ℕ, (k * 3 ≤ m) ∧ (m ≤ 199) → k ≤ m / 3 := 
by
  sorry

end max_participants_win_at_least_three_matches_l378_37879


namespace degenerate_ellipse_single_point_c_l378_37898

theorem degenerate_ellipse_single_point_c (c : ℝ) :
  (∀ x y : ℝ, 2 * x^2 + y^2 + 8 * x - 10 * y + c = 0 → x = -2 ∧ y = 5) →
  c = 33 :=
by
  intros h
  sorry

end degenerate_ellipse_single_point_c_l378_37898


namespace acute_triangle_conditions_l378_37863

-- Definitions exclusively from the conditions provided.
def condition_A (AB AC : ℝ) : Prop :=
  AB * AC > 0

def condition_B (sinA sinB sinC : ℝ) : Prop :=
  sinA / sinB = 4 / 5 ∧ sinA / sinC = 4 / 6 ∧ sinB / sinC = 5 / 6

def condition_C (cosA cosB cosC : ℝ) : Prop :=
  cosA * cosB * cosC > 0

def condition_D (tanA tanB : ℝ) : Prop :=
  tanA * tanB = 2

-- Prove which conditions guarantee that triangle ABC is acute.
theorem acute_triangle_conditions (AB AC sinA sinB sinC cosA cosB cosC tanA tanB : ℝ) :
  (condition_B sinA sinB sinC ∨ condition_C cosA cosB cosC ∨ condition_D tanA tanB) →
  (∀ (A B C : ℝ), A < π / 2 ∧ B < π / 2 ∧ C < π / 2) :=
sorry

end acute_triangle_conditions_l378_37863


namespace unique_combinations_bathing_suits_l378_37843

theorem unique_combinations_bathing_suits
  (men_styles : ℕ) (men_sizes : ℕ) (men_colors : ℕ)
  (women_styles : ℕ) (women_sizes : ℕ) (women_colors : ℕ)
  (h_men_styles : men_styles = 5) (h_men_sizes : men_sizes = 3) (h_men_colors : men_colors = 4)
  (h_women_styles : women_styles = 4) (h_women_sizes : women_sizes = 4) (h_women_colors : women_colors = 5) :
  men_styles * men_sizes * men_colors + women_styles * women_sizes * women_colors = 140 :=
by
  sorry

end unique_combinations_bathing_suits_l378_37843


namespace has_exactly_one_zero_point_l378_37839

noncomputable def f (x a b : ℝ) : ℝ := (x - 1) * Real.exp x - a * x^2 + b

theorem has_exactly_one_zero_point
  (a b : ℝ) 
  (h1 : (1/2 < a ∧ a ≤ Real.exp 2 / 2 ∧ b > 2 * a) ∨ (0 < a ∧ a < 1/2 ∧ b ≤ 2 * a)) :
  ∃! x : ℝ, f x a b = 0 := 
sorry

end has_exactly_one_zero_point_l378_37839


namespace distance_covered_l378_37841

/-- 
Given the following conditions:
1. The speed of Abhay (A) is 5 km/h.
2. The time taken by Abhay to cover a distance is 2 hours more than the time taken by Sameer.
3. If Abhay doubles his speed, then he would take 1 hour less than Sameer.
Prove that the distance (D) they are covering is 30 kilometers.
-/
theorem distance_covered (D S : ℝ) (A : ℝ) (hA : A = 5) 
  (h1 : D / A = D / S + 2) 
  (h2 : D / (2 * A) = D / S - 1) : 
  D = 30 := by
    sorry

end distance_covered_l378_37841


namespace total_payment_correct_l378_37876

noncomputable def calculate_total_payment : ℝ :=
  let original_price_vase := 200
  let discount_vase := 0.35 * original_price_vase
  let sale_price_vase := original_price_vase - discount_vase
  let tax_vase := 0.10 * sale_price_vase

  let original_price_teacups := 300
  let discount_teacups := 0.20 * original_price_teacups
  let sale_price_teacups := original_price_teacups - discount_teacups
  let tax_teacups := 0.08 * sale_price_teacups

  let original_price_plate := 500
  let sale_price_plate := original_price_plate
  let tax_plate := 0.10 * sale_price_plate

  (sale_price_vase + tax_vase) + (sale_price_teacups + tax_teacups) + (sale_price_plate + tax_plate)

theorem total_payment_correct : calculate_total_payment = 952.20 :=
by sorry

end total_payment_correct_l378_37876


namespace golden_section_AP_length_l378_37823

noncomputable def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2

noncomputable def golden_ratio_recip : ℝ := (Real.sqrt 5 - 1) / 2

theorem golden_section_AP_length (AB : ℝ) (P : ℝ) 
  (h1 : AB = 2) (h2 : P = golden_ratio_recip * AB) : 
  P = Real.sqrt 5 - 1 :=
by
  sorry

end golden_section_AP_length_l378_37823


namespace inverse_function_shift_l378_37851

-- Conditions
variable {f : ℝ → ℝ} {f_inv : ℝ → ℝ}
variable (hf : ∀ x : ℝ, f_inv (f x) = x ∧ f (f_inv x) = x)
variable (point_B : f 3 = -1)

-- Proof statement
theorem inverse_function_shift :
  f_inv (-3 + 2) = 3 :=
by
  -- Proof goes here
  sorry

end inverse_function_shift_l378_37851


namespace length_percentage_increase_l378_37836

/--
Given that the area of a rectangle is 460 square meters and the breadth is 20 meters,
prove that the percentage increase in length compared to the breadth is 15%.
-/
theorem length_percentage_increase (A : ℝ) (b : ℝ) (l : ℝ) (hA : A = 460) (hb : b = 20) (hl : l = A / b) :
  ((l - b) / b) * 100 = 15 :=
by
  sorry

end length_percentage_increase_l378_37836


namespace license_plate_increase_l378_37888

-- definitions from conditions
def old_plates_count : ℕ := 26 ^ 2 * 10 ^ 3
def new_plates_count : ℕ := 26 ^ 4 * 10 ^ 2

-- theorem stating the increase in the number of license plates
theorem license_plate_increase : 
  (new_plates_count : ℚ) / (old_plates_count : ℚ) = 26 ^ 2 / 10 :=
by
  sorry

end license_plate_increase_l378_37888


namespace achieve_100_with_fewer_threes_example1_achieve_100_with_fewer_threes_example2_l378_37842

theorem achieve_100_with_fewer_threes_example1 :
  ((333 / 3) - (33 / 3) = 100) :=
by
  sorry

theorem achieve_100_with_fewer_threes_example2 :
  ((33 * 3) + (3 / 3) = 100) :=
by
  sorry

end achieve_100_with_fewer_threes_example1_achieve_100_with_fewer_threes_example2_l378_37842


namespace not_necessarily_prime_sum_l378_37824

theorem not_necessarily_prime_sum (nat_ordered_sequence : ℕ → ℕ) :
  (∀ n1 n2 n3 : ℕ, n1 < n2 → n2 < n3 → nat_ordered_sequence n1 + nat_ordered_sequence n2 + nat_ordered_sequence n3 ≠ prime) :=
sorry

end not_necessarily_prime_sum_l378_37824


namespace root_conditions_l378_37815

theorem root_conditions (m : ℝ) : (∃ a b : ℝ, a < 2 ∧ b > 2 ∧ a * b = -1 ∧ a + b = m) ↔ m > 3 / 2 := sorry

end root_conditions_l378_37815


namespace newer_model_distance_l378_37880

theorem newer_model_distance (d_old : ℝ) (p_increase : ℝ) (d_new : ℝ) (h1 : d_old = 300) (h2 : p_increase = 0.30) (h3 : d_new = d_old * (1 + p_increase)) : d_new = 390 :=
by
  sorry

end newer_model_distance_l378_37880


namespace approximate_roots_l378_37882

noncomputable def f (x : ℝ) : ℝ := 0.3 * x^3 - 2 * x^2 - 0.2 * x + 0.5

theorem approximate_roots : 
  ∃ x₁ x₂ x₃ : ℝ, 
    (f x₁ = 0 ∧ |x₁ + 0.4| < 0.1) ∧ 
    (f x₂ = 0 ∧ |x₂ - 0.5| < 0.1) ∧ 
    (f x₃ = 0 ∧ |x₃ - 2.6| < 0.1) :=
by
  sorry

end approximate_roots_l378_37882


namespace total_wings_count_l378_37881

theorem total_wings_count (num_planes : ℕ) (wings_per_plane : ℕ) (h_planes : num_planes = 54) (h_wings : wings_per_plane = 2) : num_planes * wings_per_plane = 108 :=
by 
  sorry

end total_wings_count_l378_37881


namespace find_y_l378_37855

def is_divisible_by (x y : ℕ) : Prop := x % y = 0

def ends_with_digit (x : ℕ) (d : ℕ) : Prop :=
  x % 10 = d

theorem find_y (y : ℕ) :
  (y > 0) ∧
  is_divisible_by y 4 ∧
  is_divisible_by y 5 ∧
  is_divisible_by y 7 ∧
  is_divisible_by y 13 ∧
  ¬ is_divisible_by y 8 ∧
  ¬ is_divisible_by y 15 ∧
  ¬ is_divisible_by y 50 ∧
  ends_with_digit y 0
  → y = 1820 :=
sorry

end find_y_l378_37855


namespace jeanne_additional_tickets_l378_37846

-- Define the costs
def ferris_wheel_cost : ℕ := 5
def roller_coaster_cost : ℕ := 4
def bumper_cars_cost : ℕ := 4
def jeanne_tickets : ℕ := 5

-- Calculate the total cost
def total_cost : ℕ := ferris_wheel_cost + roller_coaster_cost + bumper_cars_cost

-- Define the proof problem
theorem jeanne_additional_tickets : total_cost - jeanne_tickets = 8 :=
by sorry

end jeanne_additional_tickets_l378_37846


namespace inequality_0_lt_a_lt_1_l378_37866

theorem inequality_0_lt_a_lt_1 (a : ℝ) (h : 0 < a ∧ a < 1) : 
  (1 / a) + (4 / (1 - a)) ≥ 9 :=
by
  sorry

end inequality_0_lt_a_lt_1_l378_37866


namespace sarah_problem_sum_l378_37829

theorem sarah_problem_sum (x y : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 100 ≤ y ∧ y < 1000) (h : 1000 * x + y = 9 * x * y) :
  x + y = 126 :=
sorry

end sarah_problem_sum_l378_37829


namespace unattainable_value_l378_37869

theorem unattainable_value : ∀ x : ℝ, x ≠ -4/3 → (y = (2 - x) / (3 * x + 4) → y ≠ -1/3) :=
by
  intro x hx h
  rw [eq_comm] at h
  sorry

end unattainable_value_l378_37869


namespace max_sum_at_1008_l378_37874

noncomputable def sum_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n * (a 1 + a n)) / 2

theorem max_sum_at_1008 (a : ℕ → ℝ) : 
  sum_sequence a 2015 > 0 → 
  sum_sequence a 2016 < 0 → 
  ∃ n, n = 1008 ∧ ∀ m, sum_sequence a m ≤ sum_sequence a 1008 :=
by
  intros h1 h2
  sorry

end max_sum_at_1008_l378_37874


namespace same_solutions_implies_k_value_l378_37801

theorem same_solutions_implies_k_value (k : ℤ) : (∀ x : ℤ, 2 * x = 4 ↔ 3 * x + k = -2) → k = -8 :=
by
  sorry

end same_solutions_implies_k_value_l378_37801


namespace perfect_square_expression_l378_37837

theorem perfect_square_expression (n : ℕ) : ∃ t : ℕ, n^2 - 4 * n + 11 = t^2 ↔ n = 5 :=
by
  sorry

end perfect_square_expression_l378_37837


namespace population_increase_rate_is_20_percent_l378_37831

noncomputable def population_increase_rate 
  (initial_population final_population : ℕ) : ℕ :=
  ((final_population - initial_population) * 100) / initial_population

theorem population_increase_rate_is_20_percent :
  population_increase_rate 2000 2400 = 20 :=
by
  unfold population_increase_rate
  sorry

end population_increase_rate_is_20_percent_l378_37831


namespace isabella_hair_growth_l378_37822

theorem isabella_hair_growth :
  ∀ (initial final : ℤ), initial = 18 → final = 24 → final - initial = 6 :=
by
  intros initial final h_initial h_final
  rw [h_initial, h_final]
  exact rfl
-- sorry

end isabella_hair_growth_l378_37822


namespace volume_of_resulting_solid_is_9_l378_37887

-- Defining the initial cube with edge length 3
def initial_cube_edge_length : ℝ := 3

-- Defining the volume of the initial cube
def initial_cube_volume : ℝ := initial_cube_edge_length^3

-- Defining the volume of the resulting solid after some parts are cut off
def resulting_solid_volume : ℝ := 9

-- Theorem stating that given the initial conditions, the volume of the resulting solid is 9
theorem volume_of_resulting_solid_is_9 : resulting_solid_volume = 9 :=
by
  sorry

end volume_of_resulting_solid_is_9_l378_37887


namespace max_sections_with_five_lines_l378_37854

def sections (n : ℕ) : ℕ :=
  if n = 0 then 1 else
  n * (n + 1) / 2 + 1

theorem max_sections_with_five_lines : sections 5 = 16 := by
  sorry

end max_sections_with_five_lines_l378_37854


namespace no_b_for_221_square_l378_37808

theorem no_b_for_221_square (b : ℕ) (h : b ≥ 3) :
  ¬ ∃ n : ℕ, 2 * b^2 + 2 * b + 1 = n^2 :=
by
  sorry

end no_b_for_221_square_l378_37808


namespace janet_savings_l378_37800

def wall1_area := 5 * 8 -- wall 1 area
def wall2_area := 7 * 8 -- wall 2 area
def wall3_area := 6 * 9 -- wall 3 area
def total_area := wall1_area + wall2_area + wall3_area
def tiles_per_square_foot := 4
def total_tiles := total_area * tiles_per_square_foot

def turquoise_tile_cost := 13
def turquoise_labor_cost := 6
def total_cost_turquoise := (total_tiles * turquoise_tile_cost) + (total_area * turquoise_labor_cost)

def purple_tile_cost := 11
def purple_labor_cost := 8
def total_cost_purple := (total_tiles * purple_tile_cost) + (total_area * purple_labor_cost)

def orange_tile_cost := 15
def orange_labor_cost := 5
def total_cost_orange := (total_tiles * orange_tile_cost) + (total_area * orange_labor_cost)

def least_expensive_option := total_cost_purple
def most_expensive_option := total_cost_orange

def savings := most_expensive_option - least_expensive_option

theorem janet_savings : savings = 1950 := by
  sorry

end janet_savings_l378_37800


namespace set_A_roster_l378_37821

def is_nat_not_greater_than_4 (x : ℕ) : Prop := x ≤ 4

def A : Set ℕ := {x | is_nat_not_greater_than_4 x}

theorem set_A_roster : A = {0, 1, 2, 3, 4} := by
  sorry

end set_A_roster_l378_37821


namespace problem1_problem2_l378_37858

-- Proof Problem for (1)
theorem problem1 : -15 - (-5) + 6 = -4 := sorry

-- Proof Problem for (2)
theorem problem2 : 81 / (-9 / 5) * (5 / 9) = -25 := sorry

end problem1_problem2_l378_37858


namespace max_red_socks_l378_37826

-- Define r (red socks), b (blue socks), t (total socks), with the given constraints
def socks_problem (r b t : ℕ) : Prop :=
  t = r + b ∧
  t ≤ 2023 ∧
  (2 * r * (r - 1) + 2 * b * (b - 1)) = 2 * 5 * t * (t - 1)

-- State the theorem that the maximum number of red socks is 990
theorem max_red_socks : ∃ r b t, socks_problem r b t ∧ r = 990 :=
sorry

end max_red_socks_l378_37826


namespace compute_fraction_power_l378_37859

theorem compute_fraction_power :
  8 * (1 / 4) ^ 4 = 1 / 32 := 
by
  sorry

end compute_fraction_power_l378_37859


namespace S_2011_value_l378_37883

-- Definitions based on conditions provided in the problem
def arithmetic_seq (a_n : ℕ → ℤ) : Prop :=
  ∃ d, ∀ n, a_n (n + 1) = a_n n + d

def sum_seq (S_n : ℕ → ℤ) (a_n : ℕ → ℤ) : Prop :=
  ∀ n, S_n n = (n * (a_n 1 + a_n n)) / 2

-- Problem statement
theorem S_2011_value
  (a_n : ℕ → ℤ)
  (S_n : ℕ → ℤ)
  (h_arith : arithmetic_seq a_n)
  (h_sum : sum_seq S_n a_n)
  (h_init : a_n 1 = -2011)
  (h_cond : (S_n 2010) / 2010 - (S_n 2008) / 2008 = 2) :
  S_n 2011 = -2011 := 
sorry

end S_2011_value_l378_37883


namespace find_multiple_sales_l378_37833

theorem find_multiple_sales 
  (A : ℝ) 
  (M : ℝ)
  (h : M * A = 0.35294117647058826 * (11 * A + M * A)) 
  : M = 6 :=
sorry

end find_multiple_sales_l378_37833


namespace expand_polynomial_identity_l378_37834

variable {x : ℝ}

theorem expand_polynomial_identity : (7 * x + 5) * (5 * x ^ 2 - 2 * x + 4) = 35 * x ^ 3 + 11 * x ^ 2 + 18 * x + 20 := by
    sorry

end expand_polynomial_identity_l378_37834


namespace mixed_operations_with_rationals_l378_37875

theorem mixed_operations_with_rationals :
  let a := 1 / 4
  let b := 1 / 2
  let c := 2 / 3
  (a - b + c) * (-12) = -8 :=
by
  sorry

end mixed_operations_with_rationals_l378_37875


namespace boat_speed_in_still_water_l378_37845

theorem boat_speed_in_still_water : 
  ∀ (V_b V_s : ℝ), 
  V_b + V_s = 15 → 
  V_b - V_s = 5 → 
  V_b = 10 :=
by
  intros V_b V_s h1 h2
  have h3 : 2 * V_b = 20 := by linarith
  linarith

end boat_speed_in_still_water_l378_37845


namespace slope_magnitude_l378_37838

-- Definitions based on given conditions
def parabola : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ y^2 = 4 * x }
def line (k m : ℝ) : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ y = k * x + m }
def focus : ℝ × ℝ := (1, 0)
def intersects (l p : Set (ℝ × ℝ)) : Prop := ∃ x1 y1 x2 y2, (x1, y1) ∈ l ∧ (x1, y1) ∈ p ∧ (x2, y2) ∈ l ∧ (x2, y2) ∈ p ∧ (x1, y1) ≠ (x2, y2)

theorem slope_magnitude (k m : ℝ) (h_k_nonzero : k ≠ 0) 
  (h_intersects : intersects (line k m) parabola) 
  (h_AF_2FB : ∀ x1 y1 x2 y2, (x1, y1) ∈ line k m → (x1, y1) ∈ parabola → 
                          (x2, y2) ∈ line k m → (x2, y2) ∈ parabola → 
                          (1 - x1 = 2 * (x2 - 1)) ∧ (-y1 = 2 * y2)) :
  |k| = 2 * Real.sqrt 2 :=
sorry

end slope_magnitude_l378_37838


namespace f_3_neg3div2_l378_37802

noncomputable def f : ℝ → ℝ :=
sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom symm_f : ∀ t : ℝ, f t = f (1 - t)
axiom restriction_f : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1/2 → f x = -x^2

theorem f_3_neg3div2 :
  f 3 + f (-3/2) = -1/4 :=
sorry

end f_3_neg3div2_l378_37802


namespace time_to_cross_bridge_l378_37856

noncomputable def train_length := 300  -- in meters
noncomputable def train_speed_kmph := 72  -- in km/h
noncomputable def bridge_length := 1500  -- in meters

-- Define the conversion from km/h to m/s
noncomputable def train_speed_mps := (train_speed_kmph * 1000) / 3600  -- in m/s

-- Define the total distance to be traveled
noncomputable def total_distance := train_length + bridge_length  -- in meters

-- Define the time to cross the bridge
noncomputable def time_to_cross := total_distance / train_speed_mps  -- in seconds

theorem time_to_cross_bridge : time_to_cross = 90 := by
  -- skipping the proof
  sorry

end time_to_cross_bridge_l378_37856


namespace right_triangle_has_one_right_angle_l378_37819

def is_right_angle (θ : ℝ) : Prop := θ = 90

def sum_of_triangle_angles (α β γ : ℝ) : Prop := α + β + γ = 180

def right_triangle (α β γ : ℝ) : Prop := is_right_angle α ∨ is_right_angle β ∨ is_right_angle γ

theorem right_triangle_has_one_right_angle (α β γ : ℝ) :
  right_triangle α β γ → sum_of_triangle_angles α β γ →
  (is_right_angle α ∧ ¬is_right_angle β ∧ ¬is_right_angle γ) ∨
  (¬is_right_angle α ∧ is_right_angle β ∧ ¬is_right_angle γ) ∨
  (¬is_right_angle α ∧ ¬is_right_angle β ∧ is_right_angle γ) :=
by
  sorry

end right_triangle_has_one_right_angle_l378_37819


namespace find_constant_a_l378_37891

theorem find_constant_a :
  (∃ (a : ℝ), a > 0 ∧ (a + 2 * a + 3 * a + 4 * a = 1)) →
  ∃ (a : ℝ), a = 1 / 10 :=
sorry

end find_constant_a_l378_37891


namespace train_crossing_time_l378_37810

-- Definitions from conditions
def length_of_train : ℕ := 120
def length_of_bridge : ℕ := 150
def speed_kmph : ℕ := 36
def speed_mps : ℕ := speed_kmph * 1000 / 3600 -- Convert km/h to m/s
def total_distance : ℕ := length_of_train + length_of_bridge

-- Theorem statement
theorem train_crossing_time : total_distance / speed_mps = 27 := by
  sorry

end train_crossing_time_l378_37810


namespace problem_1_solution_set_problem_2_minimum_value_a_l378_37885

-- Define the function f with given a value
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := |x + 1| - a * |x - 1|

-- Problem 1: Prove the solution set for f(x) > 5 when a = -2 is {x | x < -4/3 ∨ x > 2}
theorem problem_1_solution_set (x : ℝ) : f x (-2) > 5 ↔ x < -4 / 3 ∨ x > 2 :=
by
  sorry

-- Problem 2: Prove the minimum value of a ensures f(x) ≤ a * |x + 3| is 1/2
theorem problem_2_minimum_value_a : (∀ x : ℝ, f x a ≤ a * |x + 3| ∨ a ≥ 1/2) :=
by
  sorry

end problem_1_solution_set_problem_2_minimum_value_a_l378_37885


namespace bob_spends_more_time_l378_37886

def pages := 760
def time_per_page_bob := 45
def time_per_page_chandra := 30
def total_time_bob := pages * time_per_page_bob
def total_time_chandra := pages * time_per_page_chandra
def time_difference := total_time_bob - total_time_chandra

theorem bob_spends_more_time : time_difference = 11400 :=
by
  sorry

end bob_spends_more_time_l378_37886


namespace form_regular_octagon_l378_37893

def concentric_squares_form_regular_octagon (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a^2 / b^2 = 2) : Prop :=
  ∀ (p : ℂ), ∃ (h₃ : ∀ (pvertices : ℤ → ℂ), -- vertices of the smaller square
                ∀ (lperpendiculars : ℤ → ℂ), -- perpendicular line segments
                true), -- additional conditions representing the perpendicular lines construction
    -- proving that the formed shape is a regular octagon:
    true -- Placeholder for actual condition/check for regular octagon

theorem form_regular_octagon (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a^2 / b^2 = 2) :
  concentric_squares_form_regular_octagon a b h₀ h₁ h₂ :=
by sorry

end form_regular_octagon_l378_37893


namespace mul_65_35_eq_2275_l378_37890

theorem mul_65_35_eq_2275 : 65 * 35 = 2275 := by
  sorry

end mul_65_35_eq_2275_l378_37890


namespace pascal_triangle_contains_53_once_l378_37857

theorem pascal_triangle_contains_53_once
  (h_prime : Nat.Prime 53) :
  ∃! n, ∃ k, n ≥ k ∧ n > 0 ∧ k > 0 ∧ Nat.choose n k = 53 := by
  sorry

end pascal_triangle_contains_53_once_l378_37857


namespace shaded_area_of_octagon_l378_37864

def side_length := 12
def octagon_area := 288

theorem shaded_area_of_octagon (s : ℕ) (h0 : s = side_length):
  (2 * s * s - 2 * s * s / 2) * 2 / 2 = octagon_area :=
by
  skip
  sorry

end shaded_area_of_octagon_l378_37864


namespace total_people_in_cars_by_end_of_race_l378_37830

-- Define the initial conditions and question
def initial_num_cars : ℕ := 20
def initial_num_passengers_per_car : ℕ := 2
def initial_num_drivers_per_car : ℕ := 1
def extra_passengers_per_car : ℕ := 1

-- Define the number of people per car initially
def initial_people_per_car : ℕ := initial_num_passengers_per_car + initial_num_drivers_per_car

-- Define the number of people per car after gaining extra passenger
def final_people_per_car : ℕ := initial_people_per_car + extra_passengers_per_car

-- The statement to be proven
theorem total_people_in_cars_by_end_of_race : initial_num_cars * final_people_per_car = 80 := by
  -- Prove the theorem
  sorry

end total_people_in_cars_by_end_of_race_l378_37830


namespace roots_quadratic_eq_value_l378_37892

theorem roots_quadratic_eq_value (d e : ℝ) (h : 3 * d^2 + 4 * d - 7 = 0) (h' : 3 * e^2 + 4 * e - 7 = 0) : 
  (d - 2) * (e - 2) = 13 / 3 := 
by
  sorry

end roots_quadratic_eq_value_l378_37892


namespace correct_statement_l378_37861

variable {a b : Type} -- Let a and b be types representing lines
variable {α β : Type} -- Let α and β be types representing planes

-- Define parallel relations for lines and planes
def parallel (L P : Type) : Prop := sorry

-- Define the subset relation for lines in planes
def subset (L P : Type) : Prop := sorry

-- Now state the theorem corresponding to the correct answer
theorem correct_statement (h1 : parallel α β) (h2 : subset a α) : parallel a β :=
sorry

end correct_statement_l378_37861


namespace janet_total_earnings_l378_37860

-- Definitions based on conditions from step a)
def hourly_wage := 70
def hours_worked := 20
def rate_per_pound := 20
def weight_sculpture1 := 5
def weight_sculpture2 := 7

-- Statement for the proof problem
theorem janet_total_earnings : 
  let earnings_from_extermination := hourly_wage * hours_worked
  let earnings_from_sculpture1 := rate_per_pound * weight_sculpture1
  let earnings_from_sculpture2 := rate_per_pound * weight_sculpture2
  earnings_from_extermination + earnings_from_sculpture1 + earnings_from_sculpture2 = 1640 := 
by
  sorry

end janet_total_earnings_l378_37860


namespace Barkley_bones_l378_37827

def bones_per_month : ℕ := 10
def months : ℕ := 5
def bones_received : ℕ := bones_per_month * months
def bones_buried : ℕ := 42
def bones_available : ℕ := 8

theorem Barkley_bones :
  bones_received - bones_buried = bones_available := by sorry

end Barkley_bones_l378_37827


namespace lily_remaining_money_l378_37811

def initial_amount := 55
def spent_on_shirt := 7
def spent_at_second_shop := 3 * spent_on_shirt
def total_spent := spent_on_shirt + spent_at_second_shop
def remaining_amount := initial_amount - total_spent

theorem lily_remaining_money : remaining_amount = 27 :=
by
  sorry

end lily_remaining_money_l378_37811


namespace range_of_a_l378_37844

noncomputable def geometric_seq (r : ℝ) (n : ℕ) (a₁ : ℝ) : ℝ := a₁ * r ^ (n - 1)

theorem range_of_a (a : ℝ) :
  (∃ a_seq b_seq : ℕ → ℝ, a_seq 1 = a ∧ (∀ n, b_seq n = (a_seq n - 2) / (a_seq n - 1)) ∧ (∀ n, a_seq n > a_seq (n+1)) ∧ (∀ n, b_seq (n + 1) = geometric_seq (2/3) (n + 1) (b_seq 1))) → 2 < a :=
by
  sorry

end range_of_a_l378_37844


namespace gerald_total_pieces_eq_672_l378_37873

def pieces_per_table : Nat := 12
def pieces_per_chair : Nat := 8
def num_tables : Nat := 24
def num_chairs : Nat := 48

def total_pieces : Nat := pieces_per_table * num_tables + pieces_per_chair * num_chairs

theorem gerald_total_pieces_eq_672 : total_pieces = 672 :=
by
  sorry

end gerald_total_pieces_eq_672_l378_37873


namespace solve_quadratic_polynomial_l378_37897

noncomputable def q (x : ℝ) : ℝ := -4.5 * x^2 + 4.5 * x + 135

theorem solve_quadratic_polynomial : 
  (q (-5) = 0) ∧ (q 6 = 0) ∧ (q 7 = -54) :=
by
  sorry

end solve_quadratic_polynomial_l378_37897


namespace newspapers_ratio_l378_37870

theorem newspapers_ratio :
  (∀ (j m : ℕ), j = 234 → m = 4 * j + 936 → (m / 4) / j = 2) :=
by
  sorry

end newspapers_ratio_l378_37870


namespace neg_fraction_comparison_l378_37878

theorem neg_fraction_comparison : - (4 / 5 : ℝ) > - (5 / 6 : ℝ) :=
by {
  -- sorry to skip the proof
  sorry
}

end neg_fraction_comparison_l378_37878


namespace tony_combined_lift_weight_l378_37806

noncomputable def tony_exercises :=
  let curl_weight := 90 -- pounds.
  let military_press_weight := 2 * curl_weight -- pounds.
  let squat_weight := 5 * military_press_weight -- pounds.
  let bench_press_weight := 1.5 * military_press_weight -- pounds.
  squat_weight + bench_press_weight

theorem tony_combined_lift_weight :
  tony_exercises = 1170 := by
  -- Here we will include the necessary proof steps
  sorry

end tony_combined_lift_weight_l378_37806


namespace tangent_line_to_curve_perpendicular_l378_37852

noncomputable def perpendicular_tangent_line (x y : ℝ) : Prop :=
  y = x^4 ∧ (4*x - y - 3 = 0)

theorem tangent_line_to_curve_perpendicular {x y : ℝ} (h : y = x^4 ∧ (4*x - y - 3 = 0)) :
  ∃ (x y : ℝ), (x+4*y-8=0) ∧ (4*x - y - 3 = 0) :=
by
  sorry

end tangent_line_to_curve_perpendicular_l378_37852


namespace number_of_subsets_of_three_element_set_l378_37817

theorem number_of_subsets_of_three_element_set :
  ∃ (S : Finset ℕ), S.card = 3 ∧ S.powerset.card = 8 :=
sorry

end number_of_subsets_of_three_element_set_l378_37817


namespace max_a_value_l378_37809

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b

theorem max_a_value :
  (∀ x : ℝ, ∃ y : ℝ, f y a b = f x a b + y) → a ≤ 1/2 :=
by
  sorry

end max_a_value_l378_37809


namespace percent_motorists_no_ticket_l378_37816

theorem percent_motorists_no_ticket (M : ℝ) :
  (0.14285714285714285 * M - 0.10 * M) / (0.14285714285714285 * M) * 100 = 30 :=
by
  sorry

end percent_motorists_no_ticket_l378_37816


namespace koala_fiber_intake_l378_37849

theorem koala_fiber_intake (x : ℝ) (h1 : 0.3 * x = 12) : x = 40 := 
by 
  sorry

end koala_fiber_intake_l378_37849


namespace find_a_plus_b_l378_37832

noncomputable def f (a b : ℝ) (x : ℝ) := a * x + b

noncomputable def h (x : ℝ) := 3 * x + 2

theorem find_a_plus_b (a b : ℝ) (x : ℝ) (h_condition : ∀ x, h (f a b x) = 4 * x - 1) :
  a + b = 1 / 3 := 
by
  sorry

end find_a_plus_b_l378_37832


namespace area_of_region_l378_37889

theorem area_of_region : 
  (∃ x y : ℝ, (x + 5)^2 + (y - 3)^2 = 32) → (π * 32 = 32 * π) :=
by 
  sorry

end area_of_region_l378_37889


namespace intersection_with_y_axis_l378_37820

-- Define the given function
def f (x : ℝ) := x^2 + x - 2

-- Prove that the intersection point with the y-axis is (0, -2)
theorem intersection_with_y_axis : f 0 = -2 :=
by {
  sorry
}

end intersection_with_y_axis_l378_37820


namespace outer_perimeter_fence_l378_37894

-- Definitions based on given conditions
def total_posts : Nat := 16
def post_width_feet : Real := 0.5 -- 6 inches converted to feet
def gap_length_feet : Real := 6 -- gap between posts in feet
def num_sides : Nat := 4 -- square field has 4 sides

-- Hypotheses that capture conditions and intermediate calculations
def num_corners : Nat := 4
def non_corner_posts : Nat := total_posts - num_corners
def non_corner_posts_per_side : Nat := non_corner_posts / num_sides
def posts_per_side : Nat := non_corner_posts_per_side + 2
def gaps_per_side : Nat := posts_per_side - 1
def length_gaps_per_side : Real := gaps_per_side * gap_length_feet
def total_post_width_per_side : Real := posts_per_side * post_width_feet
def length_one_side : Real := length_gaps_per_side + total_post_width_per_side
def perimeter : Real := num_sides * length_one_side

-- The theorem to prove
theorem outer_perimeter_fence : perimeter = 106 := by
  sorry

end outer_perimeter_fence_l378_37894


namespace sum_of_odd_integers_l378_37848

theorem sum_of_odd_integers (n : ℕ) (h1 : 4970 = n * (1 + n)) : (n ^ 2 = 4900) :=
by
  sorry

end sum_of_odd_integers_l378_37848


namespace Maryann_total_minutes_worked_l378_37807

theorem Maryann_total_minutes_worked (c a t : ℕ) (h1 : c = 70) (h2 : a = 7 * c) (h3 : t = c + a) : t = 560 := by
  sorry

end Maryann_total_minutes_worked_l378_37807


namespace calc_theoretical_yield_l378_37825
-- Importing all necessary libraries

-- Define the molar masses
def molar_mass_NaNO3 : ℝ := 85

-- Define the initial moles
def initial_moles_NH4NO3 : ℝ := 2
def initial_moles_NaOH : ℝ := 2

-- Define the final yield percentage
def yield_percentage : ℝ := 0.85

-- State the proof problem
theorem calc_theoretical_yield :
  let moles_NaNO3 := (2 : ℝ) * 2 * yield_percentage
  let grams_NaNO3 := moles_NaNO3 * molar_mass_NaNO3
  grams_NaNO3 = 289 :=
by 
  sorry

end calc_theoretical_yield_l378_37825


namespace exp_add_exp_nat_mul_l378_37840

noncomputable def Exp (z : ℝ) : ℝ := Real.exp z

theorem exp_add (a b x : ℝ) :
  Exp ((a + b) * x) = Exp (a * x) * Exp (b * x) := sorry

theorem exp_nat_mul (x : ℝ) (k : ℕ) :
  Exp (k * x) = (Exp x) ^ k := sorry

end exp_add_exp_nat_mul_l378_37840


namespace books_at_end_of_year_l378_37895

def init_books : ℕ := 72
def monthly_books : ℕ := 12 -- 1 book each month for 12 months
def books_bought1 : ℕ := 5
def books_bought2 : ℕ := 2
def books_gift1 : ℕ := 1
def books_gift2 : ℕ := 4
def books_donated : ℕ := 12
def books_sold : ℕ := 3

theorem books_at_end_of_year :
  init_books + monthly_books + books_bought1 + books_bought2 + books_gift1 + books_gift2 - books_donated - books_sold = 81 :=
by
  sorry

end books_at_end_of_year_l378_37895
