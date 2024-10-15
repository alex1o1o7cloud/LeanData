import Mathlib

namespace NUMINAMATH_GPT_larger_integer_value_l818_81889

-- Define the conditions as Lean definitions
def quotient_condition (a b : ℕ) : Prop := a / b = 5 / 2
def product_condition (a b : ℕ) : Prop := a * b = 160
def larger_integer (a b : ℕ) : ℕ := if a > b then a else b

-- State the theorem with conditions and expected outcome
theorem larger_integer_value (a b : ℕ) (h1 : quotient_condition a b) (h2 : product_condition a b) :
  larger_integer a b = 20 :=
sorry -- Proof to be provided

end NUMINAMATH_GPT_larger_integer_value_l818_81889


namespace NUMINAMATH_GPT_cylinder_volume_l818_81843

theorem cylinder_volume (r : ℝ) (h : ℝ) (A : ℝ) (V : ℝ) 
  (sphere_surface_area : A = 256 * Real.pi)
  (cylinder_height : h = 2 * r) 
  (sphere_surface_formula : A = 4 * Real.pi * r^2) 
  (cylinder_volume_formula : V = Real.pi * r^2 * h) : V = 1024 * Real.pi := 
by
  -- Definitions provided as conditions
  sorry

end NUMINAMATH_GPT_cylinder_volume_l818_81843


namespace NUMINAMATH_GPT_plane_equation_l818_81858

variable (x y z : ℝ)

def line1 := 3 * x - 2 * y + 5 * z + 3 = 0
def line2 := x + 2 * y - 3 * z - 11 = 0
def origin_plane := 18 * x - 8 * y + 23 * z = 0

theorem plane_equation : 
  (∀ x y z, line1 x y z → line2 x y z → origin_plane x y z) :=
by
  sorry

end NUMINAMATH_GPT_plane_equation_l818_81858


namespace NUMINAMATH_GPT_white_triangle_pairs_condition_l818_81812

def number_of_white_pairs (total_triangles : Nat) 
                          (red_pairs : Nat) 
                          (blue_pairs : Nat)
                          (mixed_pairs : Nat) : Nat :=
  let red_involved := red_pairs * 2
  let blue_involved := blue_pairs * 2
  let remaining_red := total_triangles / 2 * 5 - red_involved - mixed_pairs
  let remaining_blue := total_triangles / 2 * 4 - blue_involved - mixed_pairs
  (total_triangles / 2 * 7) - (remaining_red + remaining_blue)/2

theorem white_triangle_pairs_condition : number_of_white_pairs 32 3 2 1 = 6 := by
  sorry

end NUMINAMATH_GPT_white_triangle_pairs_condition_l818_81812


namespace NUMINAMATH_GPT_M_subset_N_l818_81817

def M : Set ℚ := { x | ∃ k : ℤ, x = k / 2 + 1 / 4 }
def N : Set ℚ := { x | ∃ k : ℤ, x = k / 4 + 1 / 2 }

theorem M_subset_N : M ⊆ N :=
sorry

end NUMINAMATH_GPT_M_subset_N_l818_81817


namespace NUMINAMATH_GPT_river_width_l818_81859

variable (depth : ℝ) (flow_rate_kmph : ℝ) (volume_per_minute : ℝ)

-- Define the given conditions:
def depth_of_river : ℝ := 4
def flow_rate : ℝ := 4
def volume_per_minute_water : ℝ := 10666.666666666666

-- The proposition to prove:
theorem river_width :
  let flow_rate_m_per_min := (flow_rate * 1000) / 60
  let width := volume_per_minute / (flow_rate_m_per_min * depth)
  width = 40 :=
by
  sorry

end NUMINAMATH_GPT_river_width_l818_81859


namespace NUMINAMATH_GPT_relationship_and_range_dimensions_when_area_18_impossibility_of_area_21_l818_81821

variable (x y : ℝ)
variable (h1 : 2 * (x + y) = 18)
variable (h2 : x * y = 18)
variable (h3 : x > 0) (h4 : y > 0) (h5 : x > y)
variable (h6 : x * y = 21)

theorem relationship_and_range : (y = 9 - x ∧ 0 < x ∧ x < 9) :=
by sorry

theorem dimensions_when_area_18 :
  (x = 6 ∧ y = 3) ∨ (x = 3 ∧ y = 6) :=
by sorry

theorem impossibility_of_area_21 :
  ¬(∃ x y, x * y = 21 ∧ 2 * (x + y) = 18 ∧ x > y) :=
by sorry

end NUMINAMATH_GPT_relationship_and_range_dimensions_when_area_18_impossibility_of_area_21_l818_81821


namespace NUMINAMATH_GPT_find_f_2_solve_inequality_l818_81867

noncomputable def f : ℝ → ℝ :=
  sorry -- definition of f cannot be constructed without further info

axiom f_decreasing : ∀ x y : ℝ, 0 < x → 0 < y → (x ≤ y → f x ≥ f y)

axiom f_additive : ∀ x y : ℝ, 0 < x → 0 < y → f (x + y) = f x + f y - 1

axiom f_4 : f 4 = 5

theorem find_f_2 : f 2 = 3 :=
  sorry

theorem solve_inequality (m : ℝ) (h : f (m - 2) ≤ 3) : m ≥ 4 :=
  sorry

end NUMINAMATH_GPT_find_f_2_solve_inequality_l818_81867


namespace NUMINAMATH_GPT_min_distance_to_line_value_of_AB_l818_81837

noncomputable def point_B : ℝ × ℝ := (1, 1)
noncomputable def point_A : ℝ × ℝ := (4 * Real.sqrt 2, Real.pi / 4)

noncomputable def curve_C (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos θ, Real.sqrt 3 * Real.sin θ)

noncomputable def polar_line_l (a : ℝ) (θ : ℝ) : ℝ :=
  a * Real.cos (θ - Real.pi / 4)

noncomputable def line_l1 (m : ℝ) (x y : ℝ) : Prop :=
  x + y + m = 0

theorem min_distance_to_line {θ : ℝ} (a : ℝ) :
  polar_line_l a θ = 4 * Real.sqrt 2 → 
  ∃ d, d = (8 * Real.sqrt 2 - Real.sqrt 14) / 2 :=
by
  sorry

theorem value_of_AB :
  ∃ AB, AB = 12 * Real.sqrt 2 / 7 :=
by
  sorry

end NUMINAMATH_GPT_min_distance_to_line_value_of_AB_l818_81837


namespace NUMINAMATH_GPT_curve_symmetric_about_y_eq_x_l818_81881

theorem curve_symmetric_about_y_eq_x (x y : ℝ) (h : x * y * (x + y) = 1) :
  (y * x * (y + x) = 1) :=
by
  sorry

end NUMINAMATH_GPT_curve_symmetric_about_y_eq_x_l818_81881


namespace NUMINAMATH_GPT_find_vector_at_t_0_l818_81893

def vec2 := ℝ × ℝ

def line_at_t (a d : vec2) (t : ℝ) : vec2 :=
  (a.1 + t * d.1, a.2 + t * d.2)

-- Given conditions
def vector_at_t_1 (v : vec2) : Prop :=
  v = (2, 3)

def vector_at_t_4 (v : vec2) : Prop :=
  v = (8, -5)

-- Prove that the vector at t = 0 is (0, 17/3)
theorem find_vector_at_t_0 (a d: vec2) (h1: line_at_t a d 1 = (2, 3)) (h4: line_at_t a d 4 = (8, -5)) :
  line_at_t a d 0 = (0, 17 / 3) :=
sorry

end NUMINAMATH_GPT_find_vector_at_t_0_l818_81893


namespace NUMINAMATH_GPT_least_positive_three_digit_multiple_of_8_l818_81802

theorem least_positive_three_digit_multiple_of_8 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ n % 8 = 0 ∧ ∀ m, 100 ≤ m ∧ m < n ∧ m % 8 = 0 → false :=
sorry

end NUMINAMATH_GPT_least_positive_three_digit_multiple_of_8_l818_81802


namespace NUMINAMATH_GPT_undecided_voters_percentage_l818_81841

theorem undecided_voters_percentage
  (biff_percent : ℝ)
  (total_people : ℤ)
  (marty_votes : ℤ)
  (undecided_percent : ℝ) :
  biff_percent = 0.45 →
  total_people = 200 →
  marty_votes = 94 →
  undecided_percent = ((total_people - (marty_votes + (biff_percent * total_people))) / total_people) * 100 →
  undecided_percent = 8 :=
by 
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_undecided_voters_percentage_l818_81841


namespace NUMINAMATH_GPT_power_function_through_point_l818_81895

noncomputable def f (x k α : ℝ) : ℝ := k * x ^ α

theorem power_function_through_point (k α : ℝ) (h : f (1/2) k α = Real.sqrt 2) : 
  k + α = 1/2 := 
by 
  sorry

end NUMINAMATH_GPT_power_function_through_point_l818_81895


namespace NUMINAMATH_GPT_find_a_l818_81892

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (3 - a) * x - a else Real.log x / Real.log a

theorem find_a (a : ℝ) (h : f a (f a 1) = 2) : a = -2 := by
  sorry

end NUMINAMATH_GPT_find_a_l818_81892


namespace NUMINAMATH_GPT_not_p_is_sufficient_but_not_necessary_for_not_q_l818_81801

variable (x : ℝ)

def proposition_p : Prop := |x| < 2
def proposition_q : Prop := x^2 - x - 2 < 0

theorem not_p_is_sufficient_but_not_necessary_for_not_q :
  (¬ proposition_p x) → (¬ proposition_q x) ∧ (¬ proposition_q x) → (¬ proposition_p x) → False := by
  sorry

end NUMINAMATH_GPT_not_p_is_sufficient_but_not_necessary_for_not_q_l818_81801


namespace NUMINAMATH_GPT_bus_departure_interval_l818_81898

theorem bus_departure_interval
  (v : ℝ) -- speed of B (per minute)
  (t_A : ℝ := 10) -- A is overtaken every 10 minutes
  (t_B : ℝ := 6) -- B is overtaken every 6 minutes
  (v_A : ℝ := 3 * v) -- speed of A
  (d_A : ℝ := v_A * t_A) -- distance covered by A in 10 minutes
  (d_B : ℝ := v * t_B) -- distance covered by B in 6 minutes
  (v_bus_minus_vA : ℝ := d_A / t_A) -- bus speed relative to A
  (v_bus_minus_vB : ℝ := d_B / t_B) -- bus speed relative to B) :
  (t : ℝ) -- time interval between bus departures
  : t = 5 := sorry

end NUMINAMATH_GPT_bus_departure_interval_l818_81898


namespace NUMINAMATH_GPT_triangle_inequality_proof_l818_81832

theorem triangle_inequality_proof (a b c : ℝ) (h : a + b > c) : a^3 + b^3 + 3 * a * b * c > c^3 :=
by sorry

end NUMINAMATH_GPT_triangle_inequality_proof_l818_81832


namespace NUMINAMATH_GPT_abs_eq_iff_x_eq_2_l818_81829

theorem abs_eq_iff_x_eq_2 (x : ℝ) : |x - 1| = |x - 3| → x = 2 := by
  sorry

end NUMINAMATH_GPT_abs_eq_iff_x_eq_2_l818_81829


namespace NUMINAMATH_GPT_constants_inequality_value_l818_81809

theorem constants_inequality_value
  (a b c d : ℝ)
  (h1 : a < b)
  (h2 : b < c)
  (h3 : ∀ x, (1 ≤ x ∧ x ≤ 5) ∨ (24 ≤ x ∧ x ≤ 26) ∨ x < -4 ↔ (x - a) * (x - b) * (x - c) / (x - d) ≤ 0) :
  a + 3 * b + 3 * c + 4 * d = 72 :=
sorry

end NUMINAMATH_GPT_constants_inequality_value_l818_81809


namespace NUMINAMATH_GPT_capacity_of_new_bathtub_is_400_liters_l818_81839

-- Definitions based on conditions
def possible_capacities : Set ℕ := {4, 40, 400, 4000}  -- The possible capacities

-- Proof statement
theorem capacity_of_new_bathtub_is_400_liters (c : ℕ) 
  (h : c ∈ possible_capacities) : 
  c = 400 := 
sorry

end NUMINAMATH_GPT_capacity_of_new_bathtub_is_400_liters_l818_81839


namespace NUMINAMATH_GPT_handshake_problem_l818_81851

theorem handshake_problem (n : ℕ) (H : (n * (n - 1)) / 2 = 28) : n = 8 := 
sorry

end NUMINAMATH_GPT_handshake_problem_l818_81851


namespace NUMINAMATH_GPT_proof_f_f_pi_div_12_l818_81899

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then 4 * x^2 - 1 else (Real.sin x)^2 - (Real.cos x)^2

theorem proof_f_f_pi_div_12 : f (f (Real.pi / 12)) = 2 := by
  sorry

end NUMINAMATH_GPT_proof_f_f_pi_div_12_l818_81899


namespace NUMINAMATH_GPT_new_total_energy_l818_81873

-- Define the problem conditions
def identical_point_charges_positioned_at_vertices_of_equilateral_triangle (charges : ℕ) (initial_energy : ℝ) : Prop :=
  charges = 3 ∧ initial_energy = 18

def charge_moved_one_third_along_side (move_fraction : ℝ) : Prop :=
  move_fraction = 1/3

-- Define the theorem and proof goal
theorem new_total_energy (charges : ℕ) (initial_energy : ℝ) (move_fraction : ℝ) :
  identical_point_charges_positioned_at_vertices_of_equilateral_triangle charges initial_energy →
  charge_moved_one_third_along_side move_fraction →
  ∃ (new_energy : ℝ), new_energy = 21 :=
by
  intros h_triangle h_move
  sorry

end NUMINAMATH_GPT_new_total_energy_l818_81873


namespace NUMINAMATH_GPT_solve_abs_quadratic_eq_and_properties_l818_81869

theorem solve_abs_quadratic_eq_and_properties :
  ∃ x1 x2 : ℝ, (|x1|^2 + 2 * |x1| - 8 = 0) ∧ (|x2|^2 + 2 * |x2| - 8 = 0) ∧
               (x1 = 2 ∨ x1 = -2) ∧ (x2 = 2 ∨ x2 = -2) ∧
               (x1 + x2 = 0) ∧ (x1 * x2 = -4) :=
by
  sorry

end NUMINAMATH_GPT_solve_abs_quadratic_eq_and_properties_l818_81869


namespace NUMINAMATH_GPT_smallest_positive_angle_l818_81882

open Real

theorem smallest_positive_angle :
  ∃ x : ℝ, x > 0 ∧ x < 90 ∧ tan (4 * x * degree) = (cos (x * degree) - sin (x * degree)) / (cos (x * degree) + sin (x * degree)) ∧ x = 9 :=
sorry

end NUMINAMATH_GPT_smallest_positive_angle_l818_81882


namespace NUMINAMATH_GPT_find_number_l818_81888

theorem find_number (x : ℕ) (h : x / 5 = 75 + x / 6) : x = 2250 := 
by
  sorry

end NUMINAMATH_GPT_find_number_l818_81888


namespace NUMINAMATH_GPT_find_a_b_l818_81813

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin x + b

theorem find_a_b 
  (h_max : ∀ x, f a b x ≤ 3)
  (h_min : ∀ x, f a b x ≥ 2)
  : (a = 0.5 ∨ a = -0.5) ∧ b = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_find_a_b_l818_81813


namespace NUMINAMATH_GPT_negation_of_exists_log3_nonnegative_l818_81868

variable (x : ℝ)

theorem negation_of_exists_log3_nonnegative :
  (¬ (∃ x : ℝ, Real.logb 3 x ≥ 0)) ↔ (∀ x : ℝ, Real.logb 3 x < 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_exists_log3_nonnegative_l818_81868


namespace NUMINAMATH_GPT_union_A_B_l818_81818

def A (x : ℝ) : Set ℝ := {x ^ 2, 2 * x - 1, -4}
def B (x : ℝ) : Set ℝ := {x - 5, 1 - x, 9}

theorem union_A_B (x : ℝ) (h : {9} = A x ∩ B x) :
  (A x ∪ B x) = {(-8 : ℝ), -7, -4, 4, 9} := by
  sorry

end NUMINAMATH_GPT_union_A_B_l818_81818


namespace NUMINAMATH_GPT_angle_covered_in_three_layers_l818_81876

theorem angle_covered_in_three_layers 
  (total_coverage : ℝ) (sum_of_angles : ℝ) 
  (h1 : total_coverage = 90) (h2 : sum_of_angles = 290) : 
  ∃ x : ℝ, 3 * x + 2 * (90 - x) = 290 ∧ x = 20 :=
by
  sorry

end NUMINAMATH_GPT_angle_covered_in_three_layers_l818_81876


namespace NUMINAMATH_GPT_max_D_n_l818_81897

-- Define the properties for each block
structure Block where
  shape : ℕ -- 1 for Square, 2 for Circular
  color : ℕ -- 1 for Red, 2 for Yellow
  city  : ℕ -- 1 for Nanchang, 2 for Beijing

-- The 8 blocks
def blocks : List Block := [
  { shape := 1, color := 1, city := 1 },
  { shape := 2, color := 1, city := 1 },
  { shape := 2, color := 2, city := 1 },
  { shape := 1, color := 2, city := 1 },
  { shape := 1, color := 1, city := 2 },
  { shape := 2, color := 1, city := 2 },
  { shape := 2, color := 2, city := 2 },
  { shape := 1, color := 2, city := 2 }
]

-- Define D_n counting function (to be implemented)
noncomputable def D_n (n : ℕ) : ℕ := sorry

-- Define the required proof
theorem max_D_n : 2 ≤ n → n ≤ 8 → ∃ k : ℕ, 2 ≤ k ∧ k ≤ 8 ∧ D_n k = 240 := sorry

end NUMINAMATH_GPT_max_D_n_l818_81897


namespace NUMINAMATH_GPT_calculate_full_recipes_needed_l818_81825

def initial_attendance : ℕ := 125
def attendance_drop_percentage : ℝ := 0.40
def cookies_per_student : ℕ := 2
def cookies_per_recipe : ℕ := 18

theorem calculate_full_recipes_needed :
  let final_attendance := initial_attendance * (1 - attendance_drop_percentage : ℝ)
  let total_cookies_needed := (final_attendance * (cookies_per_student : ℕ))
  let recipes_needed := total_cookies_needed / (cookies_per_recipe : ℕ)
  ⌈recipes_needed⌉ = 9 :=
  by
  sorry

end NUMINAMATH_GPT_calculate_full_recipes_needed_l818_81825


namespace NUMINAMATH_GPT_valentine_cards_l818_81814

theorem valentine_cards (x y : ℕ) (h : x * y = x + y + 18) : x * y = 40 :=
by
  sorry

end NUMINAMATH_GPT_valentine_cards_l818_81814


namespace NUMINAMATH_GPT_boat_cost_per_foot_l818_81879

theorem boat_cost_per_foot (total_savings : ℝ) (license_cost : ℝ) (docking_fee_multiplier : ℝ) (max_boat_length : ℝ) 
  (h1 : total_savings = 20000) 
  (h2 : license_cost = 500) 
  (h3 : docking_fee_multiplier = 3) 
  (h4 : max_boat_length = 12) 
  : (total_savings - (license_cost + docking_fee_multiplier * license_cost)) / max_boat_length = 1500 :=
by
  sorry

end NUMINAMATH_GPT_boat_cost_per_foot_l818_81879


namespace NUMINAMATH_GPT_abs_inequality_l818_81804

variables (a b c : ℝ)

theorem abs_inequality (h : |a - c| < |b|) : |a| < |b| + |c| :=
sorry

end NUMINAMATH_GPT_abs_inequality_l818_81804


namespace NUMINAMATH_GPT_altered_solution_detergent_volume_l818_81860

theorem altered_solution_detergent_volume 
  (bleach : ℕ)
  (detergent : ℕ)
  (water : ℕ)
  (h1 : bleach / detergent = 4 / 40)
  (h2 : detergent / water = 40 / 100)
  (ratio_tripled : 3 * (bleach / detergent) = bleach / detergent)
  (ratio_halved : (detergent / water) / 2 = (detergent / water))
  (altered_water : water = 300) : 
  detergent = 60 := 
  sorry

end NUMINAMATH_GPT_altered_solution_detergent_volume_l818_81860


namespace NUMINAMATH_GPT_complete_the_square_l818_81803

theorem complete_the_square (x : ℝ) : 
  (∃ a b : ℝ, (x + a) ^ 2 = b ∧ b = 11 ∧ a = -4) ↔ (x ^ 2 - 8 * x + 5 = 0) :=
by
  sorry

end NUMINAMATH_GPT_complete_the_square_l818_81803


namespace NUMINAMATH_GPT_fewer_white_chairs_than_green_blue_l818_81845

-- Definitions of the conditions
def blue_chairs : ℕ := 10
def green_chairs : ℕ := 3 * blue_chairs
def total_chairs : ℕ := 67
def green_blue_chairs : ℕ := green_chairs + blue_chairs
def white_chairs : ℕ := total_chairs - green_blue_chairs

-- Statement of the theorem
theorem fewer_white_chairs_than_green_blue : green_blue_chairs - white_chairs = 13 :=
by
  -- This is where the proof would go, but we're omitting it as per instruction
  sorry

end NUMINAMATH_GPT_fewer_white_chairs_than_green_blue_l818_81845


namespace NUMINAMATH_GPT_min_value_proof_l818_81811

noncomputable def minimum_value (a b c : ℝ) : ℝ :=
  9 / a + 16 / b + 25 / (c ^ 2)

theorem min_value_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 5) :
  minimum_value a b c ≥ 50 :=
sorry

end NUMINAMATH_GPT_min_value_proof_l818_81811


namespace NUMINAMATH_GPT_total_value_of_item_l818_81840

theorem total_value_of_item
  (import_tax : ℝ)
  (V : ℝ)
  (h₀ : import_tax = 110.60)
  (h₁ : import_tax = 0.07 * (V - 1000)) :
  V = 2579.43 := 
sorry

end NUMINAMATH_GPT_total_value_of_item_l818_81840


namespace NUMINAMATH_GPT_least_clock_equivalent_hour_l818_81831

theorem least_clock_equivalent_hour (h : ℕ) (h_gt_9 : h > 9) (clock_equiv : (h^2 - h) % 12 = 0) : h = 13 :=
sorry

end NUMINAMATH_GPT_least_clock_equivalent_hour_l818_81831


namespace NUMINAMATH_GPT_blowfish_stayed_own_tank_l818_81864

def number_clownfish : ℕ := 50
def number_blowfish : ℕ := 50
def number_clownfish_display_initial : ℕ := 24
def number_clownfish_display_final : ℕ := 16

theorem blowfish_stayed_own_tank : 
    (number_clownfish + number_blowfish = 100) ∧ 
    (number_clownfish = number_blowfish) ∧ 
    (number_clownfish_display_final = 2 / 3 * number_clownfish_display_initial) →
    ∀ (blowfish : ℕ), 
    blowfish = number_blowfish - number_clownfish_display_initial → 
    blowfish = 26 :=
sorry

end NUMINAMATH_GPT_blowfish_stayed_own_tank_l818_81864


namespace NUMINAMATH_GPT_wage_constraint_l818_81805

/-- Wage constraints for hiring carpenters and tilers given a budget -/
theorem wage_constraint (x y : ℕ) (h_carpenter_wage : 50 * x + 40 * y = 2000) : 5 * x + 4 * y = 200 := by
  sorry

end NUMINAMATH_GPT_wage_constraint_l818_81805


namespace NUMINAMATH_GPT_exists_l_l818_81830

theorem exists_l (n : ℕ) (h : n ≥ 4011^2) : ∃ l : ℤ, n < l^2 ∧ l^2 < (1 + 1/2005) * n := 
sorry

end NUMINAMATH_GPT_exists_l_l818_81830


namespace NUMINAMATH_GPT_ratio_of_squares_l818_81885

theorem ratio_of_squares : (1523^2 - 1517^2) / (1530^2 - 1510^2) = 3 / 10 := 
sorry

end NUMINAMATH_GPT_ratio_of_squares_l818_81885


namespace NUMINAMATH_GPT_value_of_f_sin_20_l818_81870

theorem value_of_f_sin_20 (f : ℝ → ℝ) (h : ∀ x, f (Real.cos x) = Real.sin (3 * x)) :
  f (Real.sin (20 * Real.pi / 180)) = -1 / 2 :=
by sorry

end NUMINAMATH_GPT_value_of_f_sin_20_l818_81870


namespace NUMINAMATH_GPT_find_minimum_a_l818_81842

theorem find_minimum_a (a x : ℤ) : 
  (x - a < 0) → 
  (x > -3 / 2) → 
  (∃ n : ℤ, ∀ y : ℤ, y ∈ {k | -1 ≤ k ∧ k ≤ n} ∧ y < a) → 
  a = 3 := sorry

end NUMINAMATH_GPT_find_minimum_a_l818_81842


namespace NUMINAMATH_GPT_peyton_total_yards_l818_81856

def distance_on_Saturday (throws: Nat) (yards_per_throw: Nat) : Nat :=
  throws * yards_per_throw

def distance_on_Sunday (throws: Nat) (yards_per_throw: Nat) : Nat :=
  throws * yards_per_throw

def total_distance (distance_Saturday: Nat) (distance_Sunday: Nat) : Nat :=
  distance_Saturday + distance_Sunday

theorem peyton_total_yards :
  let throws_Saturday := 20
  let yards_per_throw_Saturday := 20
  let throws_Sunday := 30
  let yards_per_throw_Sunday := 40
  distance_on_Saturday throws_Saturday yards_per_throw_Saturday +
  distance_on_Sunday throws_Sunday yards_per_throw_Sunday = 1600 :=
by
  sorry

end NUMINAMATH_GPT_peyton_total_yards_l818_81856


namespace NUMINAMATH_GPT_cassandra_overall_score_l818_81849

theorem cassandra_overall_score 
  (score1_percent : ℤ) (score1_total : ℕ)
  (score2_percent : ℤ) (score2_total : ℕ)
  (score3_percent : ℤ) (score3_total : ℕ) :
  score1_percent = 60 → score1_total = 15 →
  score2_percent = 75 → score2_total = 20 →
  score3_percent = 85 → score3_total = 25 →
  let correct1 := (score1_percent * score1_total) / 100
  let correct2 := (score2_percent * score2_total) / 100
  let correct3 := (score3_percent * score3_total) / 100
  let total_correct := correct1 + correct2 + correct3
  let total_problems := score1_total + score2_total + score3_total
  75 = (100 * total_correct) / total_problems := by
  intros h1 h2 h3 h4 h5 h6
  let correct1 := (60 * 15) / 100
  let correct2 := (75 * 20) / 100
  let correct3 := (85 * 25) / 100
  let total_correct := correct1 + correct2 + correct3
  let total_problems := 15 + 20 + 25
  suffices 75 = (100 * total_correct) / total_problems by sorry
  sorry

end NUMINAMATH_GPT_cassandra_overall_score_l818_81849


namespace NUMINAMATH_GPT_sequence_decreasing_l818_81861

noncomputable def x_n (a b : ℝ) (n : ℕ) : ℝ := 2 ^ n * (b ^ (1 / 2 ^ n) - a ^ (1 / 2 ^ n))

theorem sequence_decreasing (a b : ℝ) (h1 : 1 < a) (h2 : a < b) : ∀ n : ℕ, x_n a b n > x_n a b (n + 1) :=
by
  sorry

end NUMINAMATH_GPT_sequence_decreasing_l818_81861


namespace NUMINAMATH_GPT_curtain_additional_material_l818_81853

theorem curtain_additional_material
  (room_height_feet : ℕ)
  (curtain_length_inches : ℕ)
  (height_conversion_factor : ℕ)
  (desired_length : ℕ)
  (h_room_height_conversion : room_height_feet * height_conversion_factor = 96)
  (h_desired_length : desired_length = 101) :
  curtain_length_inches = desired_length - (room_height_feet * height_conversion_factor) :=
by
  sorry

end NUMINAMATH_GPT_curtain_additional_material_l818_81853


namespace NUMINAMATH_GPT_solve_B_l818_81838

theorem solve_B (B : ℕ) (h1 : 0 ≤ B) (h2 : B ≤ 9) (h3 : 7 ∣ (4000 + 110 * B + 2)) : B = 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_B_l818_81838


namespace NUMINAMATH_GPT_solve_equations_l818_81884

theorem solve_equations :
  (∃ x1 x2 : ℝ, x1 = 2 + Real.sqrt 5 ∧ x2 = 2 - Real.sqrt 5 ∧ (x1^2 - 4 * x1 - 1 = 0) ∧ (x2^2 - 4 * x2 - 1 = 0)) ∧
  (∃ y1 y2 : ℝ, y1 = -4 ∧ y2 = 1 ∧ ((y1 + 4)^2 = 5 * (y1 + 4)) ∧ ((y2 + 4)^2 = 5 * (y2 + 4))) :=
by
  sorry

end NUMINAMATH_GPT_solve_equations_l818_81884


namespace NUMINAMATH_GPT_revenue_fell_by_percentage_l818_81807

theorem revenue_fell_by_percentage :
  let old_revenue : ℝ := 69.0
  let new_revenue : ℝ := 52.0
  let percentage_decrease : ℝ := ((old_revenue - new_revenue) / old_revenue) * 100
  abs (percentage_decrease - 24.64) < 1e-2 :=
by
  sorry

end NUMINAMATH_GPT_revenue_fell_by_percentage_l818_81807


namespace NUMINAMATH_GPT_worker_idle_days_l818_81863

theorem worker_idle_days (W I : ℕ) 
  (h1 : 20 * W - 3 * I = 280)
  (h2 : W + I = 60) : 
  I = 40 :=
sorry

end NUMINAMATH_GPT_worker_idle_days_l818_81863


namespace NUMINAMATH_GPT_relationship_y1_y2_y3_l818_81816

-- Conditions
def y1 := -((-4)^2) + 5
def y2 := -((-1)^2) + 5
def y3 := -(2^2) + 5

-- Statement to be proved
theorem relationship_y1_y2_y3 : y2 > y3 ∧ y3 > y1 := by
  sorry

end NUMINAMATH_GPT_relationship_y1_y2_y3_l818_81816


namespace NUMINAMATH_GPT_cookies_eaten_is_correct_l818_81894

-- Define initial and remaining cookies
def initial_cookies : ℕ := 7
def remaining_cookies : ℕ := 5
def cookies_eaten : ℕ := initial_cookies - remaining_cookies

-- The theorem we need to prove
theorem cookies_eaten_is_correct : cookies_eaten = 2 :=
by
  -- Here we would provide the proof
  sorry

end NUMINAMATH_GPT_cookies_eaten_is_correct_l818_81894


namespace NUMINAMATH_GPT_cards_per_pack_l818_81808

-- Definitions from the problem conditions
def packs := 60
def cards_per_page := 10
def pages_needed := 42

-- Theorem statement for the mathematically equivalent proof problem
theorem cards_per_pack : (pages_needed * cards_per_page) / packs = 7 :=
by sorry

end NUMINAMATH_GPT_cards_per_pack_l818_81808


namespace NUMINAMATH_GPT_cat_count_after_10_days_l818_81862

def initial_cats := 60 -- Shelter had 60 cats before the intake
def intake_cats := 30 -- Shelter took in 30 cats
def total_cats_at_start := initial_cats + intake_cats -- 90 cats after intake

def even_days_adoptions := 5 -- Cats adopted on even days
def odd_days_adoptions := 15 -- Cats adopted on odd days
def total_adoptions := even_days_adoptions + odd_days_adoptions -- Total adoptions over 10 days

def day4_births := 10 -- Kittens born on day 4
def day7_births := 5 -- Kittens born on day 7
def total_births := day4_births + day7_births -- Total births over 10 days

def claimed_pets := 2 -- Number of mothers claimed as missing pets

def final_cat_count := total_cats_at_start - total_adoptions + total_births - claimed_pets -- Final cat count

theorem cat_count_after_10_days : final_cat_count = 83 := by
  sorry

end NUMINAMATH_GPT_cat_count_after_10_days_l818_81862


namespace NUMINAMATH_GPT_max_sum_x1_x2_x3_l818_81826

theorem max_sum_x1_x2_x3 : 
  ∀ (x1 x2 x3 x4 x5 x6 x7 : ℕ), 
    x1 < x2 → x2 < x3 → x3 < x4 → x4 < x5 → x5 < x6 → x6 < x7 →
    x1 + x2 + x3 + x4 + x5 + x6 + x7 = 159 →
    x1 + x2 + x3 = 61 :=
by
  intros x1 x2 x3 x4 x5 x6 x7 h1 h2 h3 h4 h5 h6 h_sum
  sorry

end NUMINAMATH_GPT_max_sum_x1_x2_x3_l818_81826


namespace NUMINAMATH_GPT_tan_alpha_minus_2beta_l818_81872

theorem tan_alpha_minus_2beta (α β : Real) 
  (h1 : Real.tan (α - β) = 2 / 5)
  (h2 : Real.tan β = 1 / 2) :
  Real.tan (α - 2 * β) = -1 / 12 := 
by 
  sorry

end NUMINAMATH_GPT_tan_alpha_minus_2beta_l818_81872


namespace NUMINAMATH_GPT_combined_capacity_eq_l818_81847

variable {x y z : ℚ}

-- Container A condition
def containerA_full (x : ℚ) := 0.75 * x
def containerA_initial (x : ℚ) := 0.30 * x
def containerA_diff (x : ℚ) := containerA_full x - containerA_initial x = 36

-- Container B condition
def containerB_full (y : ℚ) := 0.70 * y
def containerB_initial (y : ℚ) := 0.40 * y
def containerB_diff (y : ℚ) := containerB_full y - containerB_initial y = 20

-- Container C condition
def containerC_full (z : ℚ) := (2 / 3) * z
def containerC_initial (z : ℚ) := 0.50 * z
def containerC_diff (z : ℚ) := containerC_full z - containerC_initial z = 12

-- Theorem to prove the total capacity
theorem combined_capacity_eq : containerA_diff x → containerB_diff y → containerC_diff z → 
(218 + 2 / 3 = x + y + z) :=
by
  intros hA hB hC
  sorry

end NUMINAMATH_GPT_combined_capacity_eq_l818_81847


namespace NUMINAMATH_GPT_eight_pow_2012_mod_10_l818_81857

theorem eight_pow_2012_mod_10 : (8 ^ 2012) % 10 = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_eight_pow_2012_mod_10_l818_81857


namespace NUMINAMATH_GPT_distribute_balls_into_boxes_l818_81878

/--
Given 6 distinguishable balls and 3 distinguishable boxes, 
there are 3^6 = 729 ways to distribute the balls into the boxes.
-/
theorem distribute_balls_into_boxes : (3 : ℕ)^6 = 729 := 
by
  sorry

end NUMINAMATH_GPT_distribute_balls_into_boxes_l818_81878


namespace NUMINAMATH_GPT_price_after_reductions_l818_81874

theorem price_after_reductions (P : ℝ) : ((P * 0.85) * 0.90) = P * 0.765 :=
by sorry

end NUMINAMATH_GPT_price_after_reductions_l818_81874


namespace NUMINAMATH_GPT_joshua_crates_l818_81875

def joshua_packs (b : ℕ) (not_packed : ℕ) (b_per_crate : ℕ) : ℕ :=
  (b - not_packed) / b_per_crate

theorem joshua_crates : joshua_packs 130 10 12 = 10 := by
  sorry

end NUMINAMATH_GPT_joshua_crates_l818_81875


namespace NUMINAMATH_GPT_variance_eta_l818_81824

noncomputable def xi : ℝ := sorry -- Define ξ as a real number (will be specified later)
noncomputable def eta : ℝ := sorry -- Define η as a real number (will be specified later)

-- Conditions
axiom xi_distribution : xi = 3 + 2*Real.sqrt 4 -- ξ follows a normal distribution with mean 3 and variance 4
axiom relationship : xi = 2*eta + 3 -- Given relationship between ξ and η

-- Theorem to prove the question
theorem variance_eta : sorry := sorry

end NUMINAMATH_GPT_variance_eta_l818_81824


namespace NUMINAMATH_GPT_smallest_consecutive_integers_product_l818_81827

theorem smallest_consecutive_integers_product (n : ℕ) 
  (h : n * (n + 1) * (n + 2) * (n + 3) = 5040) : 
  n = 7 :=
sorry

end NUMINAMATH_GPT_smallest_consecutive_integers_product_l818_81827


namespace NUMINAMATH_GPT_function_symmetry_property_l818_81834

noncomputable def f (x : ℝ) : ℝ :=
  x ^ 2

def symmetry_property := 
  ∀ (x : ℝ), (-1 < x ∧ x ≤ 1) →
    (¬ (f (-x) = f x) ∧ ¬ (f (-x) = -f x))

theorem function_symmetry_property :
  symmetry_property :=
by
  sorry

end NUMINAMATH_GPT_function_symmetry_property_l818_81834


namespace NUMINAMATH_GPT_cookies_in_jar_l818_81835

noncomputable def C : ℕ := sorry

theorem cookies_in_jar (h : C - 1 = (C + 5) / 2) : C = 7 := by
  sorry

end NUMINAMATH_GPT_cookies_in_jar_l818_81835


namespace NUMINAMATH_GPT_escalator_time_l818_81886

theorem escalator_time (speed_escalator: ℝ) (length_escalator: ℝ) (speed_person: ℝ) (combined_speed: ℝ)
  (h1: speed_escalator = 20) (h2: length_escalator = 250) (h3: speed_person = 5) (h4: combined_speed = speed_escalator + speed_person) :
  length_escalator / combined_speed = 10 := by
  sorry

end NUMINAMATH_GPT_escalator_time_l818_81886


namespace NUMINAMATH_GPT_inversely_proportional_solve_y_l818_81836

theorem inversely_proportional_solve_y (k : ℝ) (x y : ℝ)
  (h1 : x * y = k)
  (h2 : x + y = 60)
  (h3 : x = 3 * y)
  (hx : x = -10) :
  y = -67.5 :=
by
  sorry

end NUMINAMATH_GPT_inversely_proportional_solve_y_l818_81836


namespace NUMINAMATH_GPT_at_least_one_gt_one_l818_81833

theorem at_least_one_gt_one (x y : ℝ) (h : x + y > 2) : ¬(x > 1 ∨ y > 1) → (x ≤ 1 ∧ y ≤ 1) := 
by
  sorry

end NUMINAMATH_GPT_at_least_one_gt_one_l818_81833


namespace NUMINAMATH_GPT_value_of_c_l818_81815

theorem value_of_c (b c : ℝ) (h : (∀ x : ℝ, (x + 3) * (x + b) = x^2 + c * x + 12)) : c = 7 :=
sorry

end NUMINAMATH_GPT_value_of_c_l818_81815


namespace NUMINAMATH_GPT_quiz_points_minus_homework_points_l818_81880

theorem quiz_points_minus_homework_points
  (total_points : ℕ)
  (quiz_points : ℕ)
  (test_points : ℕ)
  (homework_points : ℕ)
  (h1 : total_points = 265)
  (h2 : test_points = 4 * quiz_points)
  (h3 : homework_points = 40)
  (h4 : homework_points + quiz_points + test_points = total_points) :
  quiz_points - homework_points = 5 :=
by sorry

end NUMINAMATH_GPT_quiz_points_minus_homework_points_l818_81880


namespace NUMINAMATH_GPT_magnitude_of_error_l818_81828

theorem magnitude_of_error (x : ℝ) (hx : 0 < x) :
  abs ((4 * x) - (x / 4)) / (4 * x) * 100 = 94 := 
sorry

end NUMINAMATH_GPT_magnitude_of_error_l818_81828


namespace NUMINAMATH_GPT_find_f_of_2_l818_81855

noncomputable def f (x : ℕ) : ℕ := x^x + 2*x + 2

theorem find_f_of_2 : f 1 + 1 = 5 := 
by 
  sorry

end NUMINAMATH_GPT_find_f_of_2_l818_81855


namespace NUMINAMATH_GPT_tangent_line_eq_l818_81822

noncomputable def f (x : ℝ) : ℝ := (x - 1) * Real.exp x

noncomputable def f' (x : ℝ) : ℝ := (x : ℝ) * Real.exp x

theorem tangent_line_eq (x : ℝ) (h : x = 0) : 
  ∃ (c : ℝ), (1 : ℝ) = 1 ∧ f x = c ∧ f' x = 0 ∧ (∀ y, y = c) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_eq_l818_81822


namespace NUMINAMATH_GPT_probability_ge_sqrt2_l818_81820

noncomputable def probability_length_chord_ge_sqrt2
  (a : ℝ)
  (h : a ≠ 0)
  (intersect_cond : ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ (x - a)^2 + (y - a)^2 = 1)
  : ℝ :=
  if -1 ≤ a ∧ a ≤ 1 then (1 / Real.sqrt (1^2 + 1^2)) else 0

theorem probability_ge_sqrt2 
  (a : ℝ) 
  (h : a ≠ 0) 
  (intersect_cond : ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ (x - a)^2 + (y - a)^2 = 1)
  (length_cond : (Real.sqrt (4 - 2*a^2) ≥ Real.sqrt 2)) : 
  probability_length_chord_ge_sqrt2 a h intersect_cond = (Real.sqrt 2 / 2) :=
by
  sorry

end NUMINAMATH_GPT_probability_ge_sqrt2_l818_81820


namespace NUMINAMATH_GPT_polynomial_horner_value_l818_81896

def f (x : ℤ) : ℤ :=
  7 * x^7 + 6 * x^6 + 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x

def horner (x : ℤ) : ℤ :=
  ((((((7 * x + 6) * x + 5) * x + 4) * x + 3) * x + 2) * x + 1)

theorem polynomial_horner_value :
  horner 3 = 262 := by
  sorry

end NUMINAMATH_GPT_polynomial_horner_value_l818_81896


namespace NUMINAMATH_GPT_probability_of_even_sum_is_two_thirds_l818_81887

def first_twelve_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

noncomputable def choose_4_without_2 : ℕ := (Nat.factorial 11) / ((Nat.factorial 4) * (Nat.factorial 7))

noncomputable def choose_4_from_12 : ℕ := (Nat.factorial 12) / ((Nat.factorial 4) * (Nat.factorial 8))

noncomputable def probability_even_sum : ℚ := (choose_4_without_2 : ℚ) / (choose_4_from_12 : ℚ)

theorem probability_of_even_sum_is_two_thirds :
  probability_even_sum = (2 / 3 : ℚ) :=
sorry

end NUMINAMATH_GPT_probability_of_even_sum_is_two_thirds_l818_81887


namespace NUMINAMATH_GPT_voldemort_lunch_calories_l818_81866

def dinner_cake_calories : Nat := 110
def chips_calories : Nat := 310
def coke_calories : Nat := 215
def breakfast_calories : Nat := 560
def daily_intake_limit : Nat := 2500
def remaining_calories : Nat := 525

def total_dinner_snacks_breakfast : Nat :=
  dinner_cake_calories + chips_calories + coke_calories + breakfast_calories

def total_remaining_allowance : Nat :=
  total_dinner_snacks_breakfast + remaining_calories

def lunch_calories : Nat :=
  daily_intake_limit - total_remaining_allowance

theorem voldemort_lunch_calories:
  lunch_calories = 780 := by
  sorry

end NUMINAMATH_GPT_voldemort_lunch_calories_l818_81866


namespace NUMINAMATH_GPT_length_of_floor_is_10_l818_81854

variable (L : ℝ) -- Declare the variable representing the length of the floor

-- Conditions as definitions
def width_of_floor := 8
def strip_width := 2
def area_of_rug := 24
def rug_length := L - 2 * strip_width
def rug_width := width_of_floor - 2 * strip_width

-- Math proof problem statement
theorem length_of_floor_is_10
  (h1 : rug_length * rug_width = area_of_rug)
  (h2 : width_of_floor = 8)
  (h3 : strip_width = 2) :
  L = 10 :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_length_of_floor_is_10_l818_81854


namespace NUMINAMATH_GPT_intersection_A_B_range_of_a_l818_81823

-- Problem 1: Prove the intersection of A and B when a = 4
theorem intersection_A_B (a : ℝ) (h : a = 4) :
  { x : ℝ | 5 ≤ x ∧ x ≤ 7 } ∩ { x : ℝ | x ≤ 3 ∨ 5 < x} = {6, 7} :=
by sorry

-- Problem 2: Prove the range of values for a such that A ⊆ B
theorem range_of_a :
  { a : ℝ | (a < 2) ∨ (a > 4) } :=
by sorry

end NUMINAMATH_GPT_intersection_A_B_range_of_a_l818_81823


namespace NUMINAMATH_GPT_area_increase_by_16_percent_l818_81877

theorem area_increase_by_16_percent (L B : ℝ) :
  ((1.45 * L) * (0.80 * B)) / (L * B) = 1.16 :=
by
  sorry

end NUMINAMATH_GPT_area_increase_by_16_percent_l818_81877


namespace NUMINAMATH_GPT_no_tiling_possible_with_given_dimensions_l818_81883

theorem no_tiling_possible_with_given_dimensions :
  ¬(∃ (n : ℕ), n * (2 * 2 * 1) = (3 * 4 * 5) ∧ 
   (∀ i j k : ℕ, i * 2 = 3 ∨ i * 2 = 4 ∨ i * 2 = 5) ∧
   (∀ i j k : ℕ, j * 2 = 3 ∨ j * 2 = 4 ∨ j * 2 = 5) ∧
   (∀ i j k : ℕ, k * 1 = 3 ∨ k * 1 = 4 ∨ k * 1 = 5)) :=
sorry

end NUMINAMATH_GPT_no_tiling_possible_with_given_dimensions_l818_81883


namespace NUMINAMATH_GPT_ten_percent_eq_l818_81810

variable (s t : ℝ)

def ten_percent_of (x : ℝ) : ℝ := 0.1 * x

theorem ten_percent_eq (h : ten_percent_of s = t) : s = 10 * t :=
by sorry

end NUMINAMATH_GPT_ten_percent_eq_l818_81810


namespace NUMINAMATH_GPT_smallest_circle_area_l818_81800

/-- The smallest possible area of a circle passing through two given points in the coordinate plane. -/
theorem smallest_circle_area (P Q : ℝ × ℝ) (hP : P = (-3, -2)) (hQ : Q = (2, 4)) : 
  ∃ (A : ℝ), A = (61 * Real.pi) / 4 :=
by
  sorry

end NUMINAMATH_GPT_smallest_circle_area_l818_81800


namespace NUMINAMATH_GPT_lychee_production_increase_l818_81848

variable (x : ℕ) -- percentage increase as a natural number

def lychee_increase_2006 (x : ℕ) : ℕ :=
  (1 + x)*(1 + x)

theorem lychee_production_increase (x : ℕ) :
  lychee_increase_2006 x = (1 + x) * (1 + x) :=
by
  sorry

end NUMINAMATH_GPT_lychee_production_increase_l818_81848


namespace NUMINAMATH_GPT_solve_for_n_l818_81806

theorem solve_for_n (n : ℝ) : 
  (0.05 * n + 0.06 * (30 + n)^2 = 45) ↔ 
  (n = -2.5833333333333335 ∨ n = -58.25) :=
sorry

end NUMINAMATH_GPT_solve_for_n_l818_81806


namespace NUMINAMATH_GPT_part1_f_ge_0_part2_number_of_zeros_part2_number_of_zeros_case2_l818_81871

-- Part 1: Prove f(x) ≥ 0 when a = 1
noncomputable def f (x : ℝ) : ℝ := Real.exp x - x - 1

theorem part1_f_ge_0 : ∀ x : ℝ, f x ≥ 0 := sorry

-- Part 2: Discuss the number of zeros of the function f(x)
noncomputable def g (a x : ℝ) : ℝ := Real.exp x - a * x - 1

theorem part2_number_of_zeros (a : ℝ) : 
  (a ≤ 0 ∨ a = 1) → ∃! x : ℝ, g a x = 0 := sorry

theorem part2_number_of_zeros_case2 (a : ℝ) : 
  (0 < a ∧ a < 1) ∨ (a > 1) → ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ g a x1 = 0 ∧ g a x2 = 0 := sorry

end NUMINAMATH_GPT_part1_f_ge_0_part2_number_of_zeros_part2_number_of_zeros_case2_l818_81871


namespace NUMINAMATH_GPT_Yuna_drank_most_l818_81819

noncomputable def Jimin_juice : ℝ := 0.7
noncomputable def Eunji_juice : ℝ := Jimin_juice - 1/10
noncomputable def Yoongi_juice : ℝ := 4/5
noncomputable def Yuna_juice : ℝ := Jimin_juice + 0.2

theorem Yuna_drank_most :
  Yuna_juice = max (max Jimin_juice Eunji_juice) (max Yoongi_juice Yuna_juice) :=
by
  sorry

end NUMINAMATH_GPT_Yuna_drank_most_l818_81819


namespace NUMINAMATH_GPT_fixed_point_exists_l818_81844

-- Defining the function f
def f (a x : ℝ) : ℝ := a * x - 3 + 3

-- Stating that there exists a fixed point (3, 3a)
theorem fixed_point_exists (a : ℝ) : ∃ y : ℝ, f a 3 = y :=
by
  use (3 * a)
  simp [f]
  sorry

end NUMINAMATH_GPT_fixed_point_exists_l818_81844


namespace NUMINAMATH_GPT_least_multiple_of_29_gt_500_l818_81890

theorem least_multiple_of_29_gt_500 : ∃ n : ℕ, n > 0 ∧ 29 * n > 500 ∧ 29 * n = 522 :=
by
  use 18
  sorry

end NUMINAMATH_GPT_least_multiple_of_29_gt_500_l818_81890


namespace NUMINAMATH_GPT_sum_of_cubes_consecutive_integers_divisible_by_9_l818_81865

theorem sum_of_cubes_consecutive_integers_divisible_by_9 (n : ℤ) : 
  (n^3 + (n+1)^3 + (n+2)^3) % 9 = 0 :=
sorry

end NUMINAMATH_GPT_sum_of_cubes_consecutive_integers_divisible_by_9_l818_81865


namespace NUMINAMATH_GPT_no_three_perfect_squares_l818_81852

theorem no_three_perfect_squares (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ¬(∃ k₁ k₂ k₃ : ℕ, k₁^2 = a^2 + b + c ∧ k₂^2 = b^2 + c + a ∧ k₃^2 = c^2 + a + b) :=
sorry

end NUMINAMATH_GPT_no_three_perfect_squares_l818_81852


namespace NUMINAMATH_GPT_price_per_exercise_book_is_correct_l818_81850

-- Define variables and conditions from the problem statement
variables (xM xH booksM booksH pricePerBook : ℝ)
variables (xH_gives_xM : ℝ)

-- Conditions set up from the problem statement
axiom pooled_money : xM = xH
axiom books_ming : booksM = 8
axiom books_hong : booksH = 12
axiom amount_given : xH_gives_xM = 1.1

-- Problem statement to prove
theorem price_per_exercise_book_is_correct :
  (8 + 12) * pricePerBook / 2 = 1.1 → pricePerBook = 0.55 := by
  sorry

end NUMINAMATH_GPT_price_per_exercise_book_is_correct_l818_81850


namespace NUMINAMATH_GPT_arithmetic_sequence_value_l818_81846

theorem arithmetic_sequence_value :
  ∀ (a : ℕ → ℤ), 
  a 1 = 1 → 
  a 3 = -5 → 
  (a 1 - a 2 - a 3 - a 4 = 16) :=
by
  intros a h1 h3
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_value_l818_81846


namespace NUMINAMATH_GPT_total_bike_cost_l818_81891

def marions_bike_cost : ℕ := 356
def stephanies_bike_cost : ℕ := 2 * marions_bike_cost

theorem total_bike_cost : marions_bike_cost + stephanies_bike_cost = 1068 := by
  sorry

end NUMINAMATH_GPT_total_bike_cost_l818_81891
