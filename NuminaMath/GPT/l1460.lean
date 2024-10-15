import Mathlib

namespace NUMINAMATH_GPT_union_sets_l1460_146038

def A : Set ℕ := {1, 2, 4}
def B : Set ℕ := {2, 4, 6}

theorem union_sets : A ∪ B = {1, 2, 4, 6} := by
  sorry

end NUMINAMATH_GPT_union_sets_l1460_146038


namespace NUMINAMATH_GPT_albums_total_l1460_146026

noncomputable def miriam_albums (katrina_albums : ℕ) := 5 * katrina_albums
noncomputable def katrina_albums (bridget_albums : ℕ) := 6 * bridget_albums
noncomputable def bridget_albums (adele_albums : ℕ) := adele_albums - 15
noncomputable def total_albums (adele_albums : ℕ) (bridget_albums : ℕ) (katrina_albums : ℕ) (miriam_albums : ℕ) := 
  adele_albums + bridget_albums + katrina_albums + miriam_albums

theorem albums_total (adele_has_30 : adele_albums = 30) : 
  total_albums 30 (bridget_albums 30) (katrina_albums (bridget_albums 30)) (miriam_albums (katrina_albums (bridget_albums 30))) = 585 := 
by 
  -- In Lean, replace the term 'sorry' below with the necessary steps to finish the proof
  sorry

end NUMINAMATH_GPT_albums_total_l1460_146026


namespace NUMINAMATH_GPT_card_area_after_reduction_width_l1460_146032

def initial_length : ℕ := 5
def initial_width : ℕ := 8
def new_width := initial_width - 2
def expected_new_area : ℕ := 24

theorem card_area_after_reduction_width :
  initial_length * new_width = expected_new_area := 
by
  -- initial_length = 5, new_width = 8 - 2 = 6
  -- 5 * 6 = 30, which was corrected to 24 given the misinterpretation mentioned.
  sorry

end NUMINAMATH_GPT_card_area_after_reduction_width_l1460_146032


namespace NUMINAMATH_GPT_quadratic_has_one_solution_implies_m_eq_3_l1460_146048

theorem quadratic_has_one_solution_implies_m_eq_3 {m : ℝ} (h : ∃ x : ℝ, 3 * x^2 - 6 * x + m = 0 ∧ ∃! u, 3 * u^2 - 6 * u + m = 0) : m = 3 :=
by sorry

end NUMINAMATH_GPT_quadratic_has_one_solution_implies_m_eq_3_l1460_146048


namespace NUMINAMATH_GPT_waxberry_problem_l1460_146004

noncomputable def batch_cannot_be_sold : ℚ := 1 - (8 / 9 * 9 / 10)

def probability_distribution (X : ℚ) : ℚ := 
  if X = -3200 then (1 / 5)^4 else
  if X = -2000 then 4 * (1 / 5)^3 * (4 / 5) else
  if X = -800 then 6 * (1 / 5)^2 * (4 / 5)^2 else
  if X = 400 then 4 * (1 / 5) * (4 / 5)^3 else
  if X = 1600 then (4 / 5)^4 else 0

noncomputable def expected_profit : ℚ :=
  -3200 * probability_distribution (-3200) +
  -2000 * probability_distribution (-2000) +
  -800 * probability_distribution (-800) +
  400 * probability_distribution (400) +
  1600 * probability_distribution (1600)

theorem waxberry_problem : 
  batch_cannot_be_sold = 1 / 5 ∧ 
  (probability_distribution (-3200) = 1 / 625 ∧ 
   probability_distribution (-2000) = 16 / 625 ∧ 
   probability_distribution (-800) = 96 / 625 ∧ 
   probability_distribution (400) = 256 / 625 ∧ 
   probability_distribution (1600) = 256 / 625) ∧ 
  expected_profit = 640 :=
by 
  sorry

end NUMINAMATH_GPT_waxberry_problem_l1460_146004


namespace NUMINAMATH_GPT_find_x_l1460_146039

-- Define the conditions
def is_purely_imaginary (z : Complex) : Prop :=
  z.re = 0

-- Define the problem
theorem find_x (x : ℝ) (z : Complex) (h1 : z = Complex.ofReal (x^2 - 1) + Complex.I * (x + 1)) (h2 : is_purely_imaginary z) : x = 1 :=
sorry

end NUMINAMATH_GPT_find_x_l1460_146039


namespace NUMINAMATH_GPT_max_value_of_a_l1460_146018

noncomputable def f : ℝ → ℝ := sorry

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def decreasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f y ≤ f x

theorem max_value_of_a
  (odd_f : odd_function f)
  (decr_f : decreasing_function f)
  (h : ∀ x : ℝ, f (Real.cos (2 * x) + Real.sin x) + f (Real.sin x - a) ≤ 0) :
  a ≤ -3 :=
sorry

end NUMINAMATH_GPT_max_value_of_a_l1460_146018


namespace NUMINAMATH_GPT_quadratic_inequality_solution_set_l1460_146031

theorem quadratic_inequality_solution_set (a : ℝ) (h : a < 0) :
  {x : ℝ | ax^2 - (2 + a) * x + 2 > 0} = {x | 2 / a < x ∧ x < 1} :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_set_l1460_146031


namespace NUMINAMATH_GPT_range_of_m_l1460_146027

theorem range_of_m (m : ℝ) :
  (∃ (m : ℝ), (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + m * x1 + 1 = 0 ∧ x2^2 + m * x2 + 1 = 0) ∧ 
  (∃ x : ℝ, 4 * x^2 + 4 * (m - 2) * x + 1 ≤ 0)) ↔ (m ≤ 1 ∨ m ≥ 3 ∨ m < -2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1460_146027


namespace NUMINAMATH_GPT_integral_one_over_x_l1460_146082

theorem integral_one_over_x:
  ∫ x in (1 : ℝ)..(Real.exp 1), 1 / x = 1 := 
by 
  sorry

end NUMINAMATH_GPT_integral_one_over_x_l1460_146082


namespace NUMINAMATH_GPT_height_of_wooden_box_l1460_146083

theorem height_of_wooden_box 
  (height : ℝ)
  (h₁ : ∀ (length width : ℝ), length = 8 ∧ width = 10)
  (h₂ : ∀ (small_length small_width small_height : ℕ), small_length = 4 ∧ small_width = 5 ∧ small_height = 6)
  (h₃ : ∀ (num_boxes : ℕ), num_boxes = 4000000) :
  height = 6 := 
sorry

end NUMINAMATH_GPT_height_of_wooden_box_l1460_146083


namespace NUMINAMATH_GPT_average_speed_of_tiger_exists_l1460_146068

-- Conditions
def head_start_distance (v_t : ℝ) : ℝ := 5 * v_t
def zebra_distance : ℝ := 6 * 55
def tiger_distance (v_t : ℝ) : ℝ := 6 * v_t

-- Problem statement
theorem average_speed_of_tiger_exists (v_t : ℝ) (h : zebra_distance = head_start_distance v_t + tiger_distance v_t) : v_t = 30 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_of_tiger_exists_l1460_146068


namespace NUMINAMATH_GPT_min_of_x_squared_y_squared_z_squared_l1460_146088

theorem min_of_x_squared_y_squared_z_squared (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  x^2 + y^2 + z^2 ≥ 4 :=
by sorry

end NUMINAMATH_GPT_min_of_x_squared_y_squared_z_squared_l1460_146088


namespace NUMINAMATH_GPT_ratio_amyl_alcohol_to_ethanol_l1460_146069

noncomputable def mol_amyl_alcohol : ℕ := 3
noncomputable def mol_hcl : ℕ := 3
noncomputable def mol_ethanol : ℕ := 1
noncomputable def mol_h2so4 : ℕ := 1
noncomputable def mol_ch3_cl2_c5_h9 : ℕ := 3
noncomputable def mol_h2o : ℕ := 3
noncomputable def mol_ethyl_dimethylpropyl_sulfate : ℕ := 1

theorem ratio_amyl_alcohol_to_ethanol : 
  (mol_amyl_alcohol / mol_ethanol = 3) :=
by 
  have h1 : mol_amyl_alcohol = 3 := by rfl
  have h2 : mol_ethanol = 1 := by rfl
  sorry

end NUMINAMATH_GPT_ratio_amyl_alcohol_to_ethanol_l1460_146069


namespace NUMINAMATH_GPT_at_least_one_less_than_two_l1460_146096

theorem at_least_one_less_than_two (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : 2 < a + b) :
  (1 + b) / a < 2 ∨ (1 + a) / b < 2 := 
by
  sorry

end NUMINAMATH_GPT_at_least_one_less_than_two_l1460_146096


namespace NUMINAMATH_GPT_age_transition_l1460_146011

theorem age_transition (initial_ages : List ℕ) : 
  initial_ages = [19, 34, 37, 42, 48] →
  (∃ x, 0 < x ∧ x < 10 ∧ 
  new_ages = List.map (fun age => age + x) initial_ages ∧ 
  new_ages = [25, 40, 43, 48, 54]) →
  x = 6 :=
by
  intros h_initial_ages h_exist_x
  sorry

end NUMINAMATH_GPT_age_transition_l1460_146011


namespace NUMINAMATH_GPT_surface_area_of_second_cube_l1460_146033

theorem surface_area_of_second_cube (V1 V2: ℝ) (a2: ℝ):
  (V1 = 16 ∧ V2 = 4 * V1 ∧ a2 = (V2)^(1/3)) → 6 * a2^2 = 96 :=
by intros h; sorry

end NUMINAMATH_GPT_surface_area_of_second_cube_l1460_146033


namespace NUMINAMATH_GPT_series_sum_correct_l1460_146062

noncomputable def series_sum : ℝ := ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_sum_correct : series_sum = 3 / 4 :=
sorry

end NUMINAMATH_GPT_series_sum_correct_l1460_146062


namespace NUMINAMATH_GPT_initial_pigs_count_l1460_146094

theorem initial_pigs_count (P : ℕ) (h1 : 2 + P + 6 + 3 + 5 + 2 = 21) : P = 3 :=
by
  sorry

end NUMINAMATH_GPT_initial_pigs_count_l1460_146094


namespace NUMINAMATH_GPT_remaining_walking_time_is_30_l1460_146099

-- Define all the given conditions
def total_distance_to_store : ℝ := 2.5
def distance_already_walked : ℝ := 1.0
def time_per_mile : ℝ := 20.0

-- Define the target remaining walking time
def remaining_distance : ℝ := total_distance_to_store - distance_already_walked
def remaining_time : ℝ := remaining_distance * time_per_mile

-- Prove the remaining walking time is 30 minutes
theorem remaining_walking_time_is_30 : remaining_time = 30 :=
by
  -- Formal proof would go here using corresponding Lean tactics
  sorry

end NUMINAMATH_GPT_remaining_walking_time_is_30_l1460_146099


namespace NUMINAMATH_GPT_james_bought_100_cattle_l1460_146030

noncomputable def number_of_cattle (purchase_price : ℝ) (feeding_ratio : ℝ) (weight_per_cattle : ℝ) (price_per_pound : ℝ) (profit : ℝ) : ℝ :=
  let feeding_cost := purchase_price * feeding_ratio
  let total_feeding_cost := purchase_price + feeding_cost
  let total_cost := purchase_price + total_feeding_cost
  let selling_price_per_cattle := weight_per_cattle * price_per_pound
  let total_revenue := total_cost + profit
  total_revenue / selling_price_per_cattle

theorem james_bought_100_cattle :
  number_of_cattle 40000 0.20 1000 2 112000 = 100 :=
by {
  sorry
}

end NUMINAMATH_GPT_james_bought_100_cattle_l1460_146030


namespace NUMINAMATH_GPT_deck_of_1000_transformable_l1460_146093

def shuffle (n : ℕ) (deck : List ℕ) : List ℕ :=
  -- Definition of the shuffle operation as described in the problem
  sorry

noncomputable def transformable_in_56_shuffles (n : ℕ) : Prop :=
  ∀ (initial final : List ℕ) (h₁ : initial.length = n) (h₂ : final.length = n),
  -- Prove that any initial arrangement can be transformed to any final arrangement in at most 56 shuffles
  sorry

theorem deck_of_1000_transformable : transformable_in_56_shuffles 1000 :=
  -- Implement the proof here
  sorry

end NUMINAMATH_GPT_deck_of_1000_transformable_l1460_146093


namespace NUMINAMATH_GPT_min_value_of_a_l1460_146042

noncomputable def P (x : ℕ) : ℤ := sorry

def smallest_value_of_a (a : ℕ) : Prop :=
  a > 0 ∧
  (P 1 = a ∧ P 3 = a ∧ P 5 = a ∧ P 7 = a ∧ P 9 = a ∧
   P 2 = -a ∧ P 4 = -a ∧ P 6 = -a ∧ P 8 = -a ∧ P 10 = -a)

theorem min_value_of_a : ∃ a : ℕ, smallest_value_of_a a ∧ a = 6930 :=
sorry

end NUMINAMATH_GPT_min_value_of_a_l1460_146042


namespace NUMINAMATH_GPT_car_speed_constant_l1460_146001

theorem car_speed_constant (v : ℝ) (hv : v ≠ 0)
  (condition_1 : (1 / 36) * 3600 = 100) 
  (condition_2 : (1 / v) * 3600 = 120) :
  v = 30 := by
  sorry

end NUMINAMATH_GPT_car_speed_constant_l1460_146001


namespace NUMINAMATH_GPT_terminal_side_quadrant_l1460_146051

theorem terminal_side_quadrant (α : ℝ) (k : ℤ) (hk : α = 45 + k * 180) :
  (∃ n : ℕ, k = 2 * n ∧ α = 45) ∨ (∃ n : ℕ, k = 2 * n + 1 ∧ α = 225) :=
sorry

end NUMINAMATH_GPT_terminal_side_quadrant_l1460_146051


namespace NUMINAMATH_GPT_inclination_angle_of_focal_chord_l1460_146013

theorem inclination_angle_of_focal_chord
  (p : ℝ)
  (h_parabola : ∀ x y : ℝ, y^2 = 2 * p * x → True)
  (h_focal_chord_length : ∀ A B : ℝ, |A - B| = 8 * p → True) :
  ∃ θ : ℝ, (θ = π / 6 ∨ θ = 5 * π / 6) :=
by
  sorry

end NUMINAMATH_GPT_inclination_angle_of_focal_chord_l1460_146013


namespace NUMINAMATH_GPT_height_of_wall_l1460_146064

-- Definitions
def brick_length : ℝ := 25
def brick_width : ℝ := 11.25
def brick_height : ℝ := 6
def wall_length : ℝ := 850
def wall_width : ℝ := 22.5
def num_bricks : ℝ := 6800

-- Total volume of bricks
def total_brick_volume : ℝ := num_bricks * brick_length * brick_width * brick_height

-- Volume of the wall
def wall_volume (height : ℝ) : ℝ := wall_length * wall_width * height

-- Proof statement
theorem height_of_wall : ∃ h : ℝ, wall_volume h = total_brick_volume ∧ h = 600 := 
sorry

end NUMINAMATH_GPT_height_of_wall_l1460_146064


namespace NUMINAMATH_GPT_possible_values_for_xyz_l1460_146059

theorem possible_values_for_xyz:
  (∀ (x y z : ℕ), x > 0 → y > 0 → z > 0 →
   x + 2 * y = z →
   x^2 - 4 * y^2 + z^2 = 310 →
   (∃ (k : ℕ), k = x * y * z ∧ (k = 11935 ∨ k = 2015))) :=
by
  intros x y z hx hy hz h1 h2
  sorry

end NUMINAMATH_GPT_possible_values_for_xyz_l1460_146059


namespace NUMINAMATH_GPT_max_min_rounded_value_l1460_146025

theorem max_min_rounded_value (n : ℝ) (h : 3.75 ≤ n ∧ n < 3.85) : 
  (∀ n, 3.75 ≤ n ∧ n < 3.85 → n ≤ 3.84 ∧ n ≥ 3.75) :=
sorry

end NUMINAMATH_GPT_max_min_rounded_value_l1460_146025


namespace NUMINAMATH_GPT_two_f_one_lt_f_four_l1460_146017

theorem two_f_one_lt_f_four
  (f : ℝ → ℝ)
  (h1 : ∀ x, f (x + 2) = f (-x - 2))
  (h2 : ∀ x, x > 2 → x * (deriv f x) > 2 * (deriv f x) + f x) :
  2 * f 1 < f 4 :=
sorry

end NUMINAMATH_GPT_two_f_one_lt_f_four_l1460_146017


namespace NUMINAMATH_GPT_product_of_two_numbers_l1460_146045

theorem product_of_two_numbers (x y : ℝ) 
  (h₁ : x + y = 50) 
  (h₂ : x - y = 6) : 
  x * y = 616 := 
by
  sorry

end NUMINAMATH_GPT_product_of_two_numbers_l1460_146045


namespace NUMINAMATH_GPT_solve_equation_l1460_146006

theorem solve_equation (x : ℝ) (h : (x - 7) / 2 - (1 + x) / 3 = 1) : x = 29 :=
sorry

end NUMINAMATH_GPT_solve_equation_l1460_146006


namespace NUMINAMATH_GPT_operation_results_in_m4_l1460_146029

variable (m : ℤ)

theorem operation_results_in_m4 :
  (-m^2)^2 = m^4 :=
sorry

end NUMINAMATH_GPT_operation_results_in_m4_l1460_146029


namespace NUMINAMATH_GPT_map_a_distance_map_b_distance_miles_map_b_distance_km_l1460_146019

theorem map_a_distance (distance_cm : ℝ) (scale_cm : ℝ) (scale_km : ℝ) (actual_distance : ℝ) : 
  distance_cm = 80.5 → scale_cm = 0.6 → scale_km = 6.6 → actual_distance = (distance_cm * scale_km) / scale_cm → actual_distance = 885.5 :=
by
  intros h1 h2 h3 h4
  sorry

theorem map_b_distance_miles (distance_cm : ℝ) (scale_cm : ℝ) (scale_miles : ℝ) (actual_distance_miles : ℝ) : 
  distance_cm = 56.3 → scale_cm = 1.1 → scale_miles = 7.7 → actual_distance_miles = (distance_cm * scale_miles) / scale_cm → actual_distance_miles = 394.1 :=
by
  intros h1 h2 h3 h4
  sorry

theorem map_b_distance_km (distance_miles : ℝ) (conversion_factor : ℝ) (actual_distance_km : ℝ) :
  conversion_factor = 1.60934 → distance_miles = 394.1 → actual_distance_km = distance_miles * conversion_factor → actual_distance_km = 634.3 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_map_a_distance_map_b_distance_miles_map_b_distance_km_l1460_146019


namespace NUMINAMATH_GPT_determine_m_l1460_146080

theorem determine_m {m : ℕ} : 
  (∃ (p : ℕ), p = 5 ∧ p = max (max (max 1 (1 + (m+1))) (3+1)) 4) → m = 3 := by
  sorry

end NUMINAMATH_GPT_determine_m_l1460_146080


namespace NUMINAMATH_GPT_min_value_trig_expression_l1460_146020

theorem min_value_trig_expression : (∃ x : ℝ, 3 * Real.cos x - 4 * Real.sin x = -5) :=
by
  sorry

end NUMINAMATH_GPT_min_value_trig_expression_l1460_146020


namespace NUMINAMATH_GPT_calculate_probability_l1460_146071

theorem calculate_probability :
  let letters_in_bag : List Char := ['C', 'A', 'L', 'C', 'U', 'L', 'A', 'T', 'E']
  let target_letters : List Char := ['C', 'U', 'T']
  let total_outcomes := letters_in_bag.length
  let favorable_outcomes := (letters_in_bag.filter (λ c => c ∈ target_letters)).length
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ) = 4 / 9 := sorry

end NUMINAMATH_GPT_calculate_probability_l1460_146071


namespace NUMINAMATH_GPT_part1_part2_l1460_146095

/- Define the function f(x) = |x-1| + |x-a| -/
def f (x a : ℝ) := abs (x - 1) + abs (x - a)

/- Part 1: Prove that if f(x) ≥ 2 implies the solution set {x | x ≤ 1/2 or x ≥ 5/2}, then a = 2 -/
theorem part1 (a : ℝ) (h : ∀ x : ℝ, f x a ≥ 2 → (x ≤ 1/2 ∨ x ≥ 5/2)) : a = 2 :=
  sorry

/- Part 2: Prove that for all x ∈ ℝ, f(x) + |x-1| ≥ 1 implies a ∈ [2, +∞) -/
theorem part2 (a : ℝ) (h : ∀ x : ℝ, f x a + abs (x - 1) ≥ 1) : 2 ≤ a :=
  sorry

end NUMINAMATH_GPT_part1_part2_l1460_146095


namespace NUMINAMATH_GPT_ratio_Bill_Cary_l1460_146034

noncomputable def Cary_height : ℝ := 72
noncomputable def Jan_height : ℝ := 42
noncomputable def Bill_height : ℝ := Jan_height - 6

theorem ratio_Bill_Cary : Bill_height / Cary_height = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_Bill_Cary_l1460_146034


namespace NUMINAMATH_GPT_available_floor_space_equals_110_sqft_l1460_146007

-- Definitions for the conditions
def tile_side_in_feet : ℝ := 0.5
def width_main_section_tiles : ℕ := 15
def length_main_section_tiles : ℕ := 25
def width_alcove_tiles : ℕ := 10
def depth_alcove_tiles : ℕ := 8
def width_pillar_tiles : ℕ := 3
def length_pillar_tiles : ℕ := 5

-- Conversion of tiles to feet
def width_main_section_feet : ℝ := width_main_section_tiles * tile_side_in_feet
def length_main_section_feet : ℝ := length_main_section_tiles * tile_side_in_feet
def width_alcove_feet : ℝ := width_alcove_tiles * tile_side_in_feet
def depth_alcove_feet : ℝ := depth_alcove_tiles * tile_side_in_feet
def width_pillar_feet : ℝ := width_pillar_tiles * tile_side_in_feet
def length_pillar_feet : ℝ := length_pillar_tiles * tile_side_in_feet

-- Area calculations
def area_main_section : ℝ := width_main_section_feet * length_main_section_feet
def area_alcove : ℝ := width_alcove_feet * depth_alcove_feet
def total_area : ℝ := area_main_section + area_alcove
def area_pillar : ℝ := width_pillar_feet * length_pillar_feet
def available_floor_space : ℝ := total_area - area_pillar

-- Proof statement
theorem available_floor_space_equals_110_sqft 
  (h1 : width_main_section_feet = width_main_section_tiles * tile_side_in_feet)
  (h2 : length_main_section_feet = length_main_section_tiles * tile_side_in_feet)
  (h3 : width_alcove_feet = width_alcove_tiles * tile_side_in_feet)
  (h4 : depth_alcove_feet = depth_alcove_tiles * tile_side_in_feet)
  (h5 : width_pillar_feet = width_pillar_tiles * tile_side_in_feet)
  (h6 : length_pillar_feet = length_pillar_tiles * tile_side_in_feet) 
  (h7 : area_main_section = width_main_section_feet * length_main_section_feet)
  (h8 : area_alcove = width_alcove_feet * depth_alcove_feet)
  (h9 : total_area = area_main_section + area_alcove)
  (h10 : area_pillar = width_pillar_feet * length_pillar_feet)
  (h11 : available_floor_space = total_area - area_pillar) : 
  available_floor_space = 110 := 
by 
  sorry

end NUMINAMATH_GPT_available_floor_space_equals_110_sqft_l1460_146007


namespace NUMINAMATH_GPT_delta_epsilon_time_l1460_146086

variable (D E Z h t : ℕ)

theorem delta_epsilon_time :
  (t = D - 8) →
  (t = E - 3) →
  (t = Z / 3) →
  (h = 3 * t) → 
  h = 15 / 8 :=
by
  intros h₁ h₂ h₃ h₄
  sorry

end NUMINAMATH_GPT_delta_epsilon_time_l1460_146086


namespace NUMINAMATH_GPT_flower_bed_area_l1460_146000

noncomputable def area_of_triangle (a b c : ℝ) (h : a^2 + b^2 = c^2) : ℝ :=
  (1/2) * a * b

theorem flower_bed_area : 
  area_of_triangle 6 8 10 (by norm_num) = 24 := 
sorry

end NUMINAMATH_GPT_flower_bed_area_l1460_146000


namespace NUMINAMATH_GPT_number_of_days_in_first_part_l1460_146043

variable {x : ℕ}

-- Conditions
def avg_exp_first_part (x : ℕ) : ℕ := 350 * x
def avg_exp_next_four_days : ℕ := 420 * 4
def total_days (x : ℕ) : ℕ := x + 4
def avg_exp_whole_week (x : ℕ) : ℕ := 390 * total_days x

-- Equation based on the conditions
theorem number_of_days_in_first_part :
  avg_exp_first_part x + avg_exp_next_four_days = avg_exp_whole_week x →
  x = 3 :=
by
  sorry

end NUMINAMATH_GPT_number_of_days_in_first_part_l1460_146043


namespace NUMINAMATH_GPT_total_students_l1460_146037

theorem total_students (students_in_front : ℕ) (position_from_back : ℕ) : 
  students_in_front = 6 ∧ position_from_back = 5 → 
  students_in_front + 1 + (position_from_back - 1) = 11 :=
by
  sorry

end NUMINAMATH_GPT_total_students_l1460_146037


namespace NUMINAMATH_GPT_bananas_on_first_day_l1460_146056

theorem bananas_on_first_day (total_bananas : ℕ) (days : ℕ) (increment : ℕ) (bananas_first_day : ℕ) :
  (total_bananas = 100) ∧ (days = 5) ∧ (increment = 6) ∧ ((bananas_first_day + (bananas_first_day + increment) + 
  (bananas_first_day + 2*increment) + (bananas_first_day + 3*increment) + (bananas_first_day + 4*increment)) = total_bananas) → 
  bananas_first_day = 8 :=
by
  sorry

end NUMINAMATH_GPT_bananas_on_first_day_l1460_146056


namespace NUMINAMATH_GPT_vasya_triangle_rotation_l1460_146052

theorem vasya_triangle_rotation :
  (∀ (θ1 θ2 θ3 : ℝ), (12 * θ1 = 360) ∧ (6 * θ2 = 360) ∧ (θ1 + θ2 + θ3 = 180) → ∃ n : ℕ, (n * θ3 = 360) ∧ n ≥ 4) :=
by
  -- The formal proof is omitted, inserting "sorry" to indicate incomplete proof
  sorry

end NUMINAMATH_GPT_vasya_triangle_rotation_l1460_146052


namespace NUMINAMATH_GPT_find_HCF_of_two_numbers_l1460_146016

theorem find_HCF_of_two_numbers (a b H : ℕ) 
  (H_HCF : Nat.gcd a b = H) 
  (H_LCM_Factors : Nat.lcm a b = H * 13 * 14) 
  (H_largest_number : 322 = max a b) : 
  H = 14 :=
sorry

end NUMINAMATH_GPT_find_HCF_of_two_numbers_l1460_146016


namespace NUMINAMATH_GPT_geometric_seq_a6_l1460_146081

noncomputable def geometric_sequence (a : ℕ → ℝ) := ∃ q, ∀ n, a (n + 1) = a n * q

theorem geometric_seq_a6 {a : ℕ → ℝ} (h : geometric_sequence a) (h1 : a 1 * a 3 = 4) (h2 : a 4 = 4) : a 6 = 8 :=
sorry

end NUMINAMATH_GPT_geometric_seq_a6_l1460_146081


namespace NUMINAMATH_GPT_option_B_is_equal_to_a_8_l1460_146036

-- Statement: (a^2)^4 equals a^8
theorem option_B_is_equal_to_a_8 (a : ℝ) : (a^2)^4 = a^8 :=
by { sorry }

end NUMINAMATH_GPT_option_B_is_equal_to_a_8_l1460_146036


namespace NUMINAMATH_GPT_extremum_problem_l1460_146054

def f (x a b : ℝ) := x^3 + a*x^2 + b*x + a^2

def f_prime (x a b : ℝ) := 3*x^2 + 2*a*x + b

theorem extremum_problem (a b : ℝ) 
  (cond1 : f_prime 1 a b = 0)
  (cond2 : f 1 a b = 10) :
  (a, b) = (4, -11) := 
sorry

end NUMINAMATH_GPT_extremum_problem_l1460_146054


namespace NUMINAMATH_GPT_unique_parallelogram_l1460_146005

theorem unique_parallelogram :
  ∃! (A B D C : ℤ × ℤ), 
  A = (0, 0) ∧ 
  (B.2 = B.1) ∧ 
  (D.2 = 2 * D.1) ∧ 
  (C.2 = 3 * C.1) ∧ 
  (A.1 = 0 ∧ A.2 = 0) ∧ 
  (B.1 > 0 ∧ B.2 > 0) ∧ 
  (D.1 > 0 ∧ D.2 > 0) ∧ 
  (C.1 > 0 ∧ C.2 > 0) ∧ 
  (B.1 - A.1, B.2 - A.2) + (D.1 - A.1, D.2 - A.2) = (C.1 - A.1, C.2 - A.2) ∧
  (abs ((B.1 * C.2 + C.1 * D.2 + D.1 * A.2 + A.1 * B.2) - (A.1 * C.2 + B.1 * D.2 + C.1 * B.2 + D.1 * A.2)) / 2) = 2000000 
  := by sorry

end NUMINAMATH_GPT_unique_parallelogram_l1460_146005


namespace NUMINAMATH_GPT_jen_triple_flips_l1460_146003

-- Definitions based on conditions
def tyler_double_flips : ℕ := 12
def flips_per_double_flip : ℕ := 2
def flips_by_tyler : ℕ := tyler_double_flips * flips_per_double_flip
def flips_ratio : ℕ := 2
def flips_per_triple_flip : ℕ := 3
def flips_by_jen : ℕ := flips_by_tyler * flips_ratio

-- Lean 4 statement
theorem jen_triple_flips : flips_by_jen / flips_per_triple_flip = 16 :=
by 
    -- Proof contents should go here. We only need the statement as per the instruction.
    sorry

end NUMINAMATH_GPT_jen_triple_flips_l1460_146003


namespace NUMINAMATH_GPT_second_discount_percentage_l1460_146053

/-- 
  Given:
  - The listed price of Rs. 560.
  - The final sale price after successive discounts of 20% and another discount is Rs. 313.6.
  Prove:
  - The second discount percentage is 30%.
-/
theorem second_discount_percentage (list_price final_price : ℝ) (first_discount_percentage : ℝ) : 
  list_price = 560 → 
  final_price = 313.6 → 
  first_discount_percentage = 20 → 
  ∃ (second_discount_percentage : ℝ), second_discount_percentage = 30 :=
by
  sorry

end NUMINAMATH_GPT_second_discount_percentage_l1460_146053


namespace NUMINAMATH_GPT_jacques_initial_gumballs_l1460_146061

def joanna_initial_gumballs : ℕ := 40
def each_shared_gumballs_after_purchase : ℕ := 250

theorem jacques_initial_gumballs (J : ℕ) (h : 2 * (joanna_initial_gumballs + J + 4 * (joanna_initial_gumballs + J)) = 2 * each_shared_gumballs_after_purchase) : J = 60 :=
by
  sorry

end NUMINAMATH_GPT_jacques_initial_gumballs_l1460_146061


namespace NUMINAMATH_GPT_remainder_of_13_pow_a_mod_37_l1460_146076

theorem remainder_of_13_pow_a_mod_37 (a : ℕ) (h_pos : a > 0) (h_mult : ∃ k : ℕ, a = 3 * k) : (13^a) % 37 = 1 := 
sorry

end NUMINAMATH_GPT_remainder_of_13_pow_a_mod_37_l1460_146076


namespace NUMINAMATH_GPT_train_speed_is_72_kmph_l1460_146021

noncomputable def train_length : ℝ := 110
noncomputable def bridge_length : ℝ := 112
noncomputable def crossing_time : ℝ := 11.099112071034318

theorem train_speed_is_72_kmph :
  let total_distance := train_length + bridge_length
  let speed_m_per_s := total_distance / crossing_time
  let speed_kmph := speed_m_per_s * 3.6
  speed_kmph = 72 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_is_72_kmph_l1460_146021


namespace NUMINAMATH_GPT_no_solution_for_equation_l1460_146063

theorem no_solution_for_equation :
  ¬(∃ x : ℝ, x ≠ 2 ∧ x ≠ -2 ∧ (x+2)/(x-2) - x/(x+2) = 16/(x^2-4)) :=
by
    sorry

end NUMINAMATH_GPT_no_solution_for_equation_l1460_146063


namespace NUMINAMATH_GPT_parallelogram_angle_A_l1460_146070

theorem parallelogram_angle_A 
  (A B : ℝ) (h1 : A + B = 180) (h2 : A - B = 40) :
  A = 110 :=
by sorry

end NUMINAMATH_GPT_parallelogram_angle_A_l1460_146070


namespace NUMINAMATH_GPT_isosceles_triangle_l1460_146046

theorem isosceles_triangle
  (α β γ x y z w : ℝ)
  (h_triangle : α + β + γ = 180)
  (h_quad : x + y + z + w = 360)
  (h_conditions : (x = α + β) ∧ (y = β + γ) ∧ (z = γ + α) ∨ (w = α + β) ∧ (x = β + γ) ∧ (y = γ + α) ∨ (z = α + β) ∧ (w = β + γ) ∧ (x = γ + α) ∨ (y = α + β) ∧ (z = β + γ) ∧ (w = γ + α))
  : α = β ∨ β = γ ∨ γ = α := 
sorry

end NUMINAMATH_GPT_isosceles_triangle_l1460_146046


namespace NUMINAMATH_GPT_team_selection_l1460_146049

-- Define the number of boys and girls in the club
def boys : Nat := 10
def girls : Nat := 12

-- Define the number of boys and girls to be selected for the team
def boys_team : Nat := 4
def girls_team : Nat := 4

-- Calculate the number of combinations using Nat.choose
noncomputable def choosing_boys : Nat := Nat.choose boys boys_team
noncomputable def choosing_girls : Nat := Nat.choose girls girls_team

-- Calculate the total number of ways to form the team
noncomputable def total_combinations : Nat := choosing_boys * choosing_girls

-- Theorem stating the total number of combinations equals the correct answer
theorem team_selection :
  total_combinations = 103950 := by
  sorry

end NUMINAMATH_GPT_team_selection_l1460_146049


namespace NUMINAMATH_GPT_division_correct_multiplication_correct_l1460_146089

theorem division_correct : 400 / 5 = 80 := by
  sorry

theorem multiplication_correct : 230 * 3 = 690 := by
  sorry

end NUMINAMATH_GPT_division_correct_multiplication_correct_l1460_146089


namespace NUMINAMATH_GPT_polygon_a_largest_area_l1460_146078

open Real

/-- Lean 4 statement to prove that Polygon A has the largest area among the given polygons -/
theorem polygon_a_largest_area :
  let area_polygon_a := 4 + 2 * (1 / 2 * 2 * 2)
  let area_polygon_b := 3 + 3 * (1 / 2 * 1 * 1)
  let area_polygon_c := 6
  let area_polygon_d := 5 + (1 / 2) * π * 1^2
  let area_polygon_e := 7
  area_polygon_a > area_polygon_b ∧
  area_polygon_a > area_polygon_c ∧
  area_polygon_a > area_polygon_d ∧
  area_polygon_a > area_polygon_e :=
by
  let area_polygon_a := 4 + 2 * (1 / 2 * 2 * 2)
  let area_polygon_b := 3 + 3 * (1 / 2 * 1 * 1)
  let area_polygon_c := 6
  let area_polygon_d := 5 + (1 / 2) * π * 1^2
  let area_polygon_e := 7
  sorry

end NUMINAMATH_GPT_polygon_a_largest_area_l1460_146078


namespace NUMINAMATH_GPT_part1_part2_part3_l1460_146040

-- Define the complex number z
def z (m : ℝ) : ℂ :=
  ⟨m^2 - 3*m + 2, m^2 - 1⟩  -- Note: This forms a complex number with real and imaginary parts

-- (1) Proof for z = 0 if and only if m = 1
theorem part1 (m : ℝ) : z m = 0 ↔ m = 1 :=
by sorry

-- (2) Proof for z being a pure imaginary number if and only if m = 2
theorem part2 (m : ℝ) : (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = 2 :=
by sorry

-- (3) Proof for the point corresponding to z being in the second quadrant if and only if 1 < m < 2
theorem part3 (m : ℝ) : (z m).re < 0 ∧ (z m).im > 0 ↔ 1 < m ∧ m < 2 :=
by sorry

end NUMINAMATH_GPT_part1_part2_part3_l1460_146040


namespace NUMINAMATH_GPT_fish_tank_problem_l1460_146002

def number_of_fish_in_first_tank
  (F : ℕ)          -- Let F represent the number of fish in the first tank
  (twoF : ℕ)       -- Let twoF represent twice the number of fish in the first tank
  (total : ℕ) :    -- Let total represent the total number of fish
  Prop :=
  (2 * F = twoF)  -- The other two tanks each have twice as many fish as the first
  ∧ (F + twoF + twoF = total)  -- The sum of the fish in all three tanks equals the total number of fish

theorem fish_tank_problem
  (F : ℕ)
  (H : number_of_fish_in_first_tank F (2 * F) 100) : F = 20 :=
by
  sorry

end NUMINAMATH_GPT_fish_tank_problem_l1460_146002


namespace NUMINAMATH_GPT_calc_price_per_litre_l1460_146041

noncomputable def pricePerLitre (initial final totalCost : ℝ) : ℝ :=
  totalCost / (final - initial)

theorem calc_price_per_litre :
  pricePerLitre 10 50 36.60 = 91.5 :=
by
  sorry

end NUMINAMATH_GPT_calc_price_per_litre_l1460_146041


namespace NUMINAMATH_GPT_symmetric_point_x_axis_l1460_146090

def symmetric_point (M : ℝ × ℝ) : ℝ × ℝ := (M.1, -M.2)

theorem symmetric_point_x_axis :
  ∀ (M : ℝ × ℝ), M = (3, -4) → symmetric_point M = (3, 4) :=
by
  intros M h
  rw [h]
  dsimp [symmetric_point]
  congr
  sorry

end NUMINAMATH_GPT_symmetric_point_x_axis_l1460_146090


namespace NUMINAMATH_GPT_proof_intersection_l1460_146008

def setA : Set ℤ := {x | abs x ≤ 2}

def setB : Set ℝ := {x | x^2 - 2 * x - 8 ≥ 0}

def complementB : Set ℝ := {x | x^2 - 2 * x - 8 < 0}

def intersectionAComplementB : Set ℤ := {x | x ∈ setA ∧ (x : ℝ) ∈ complementB}

theorem proof_intersection : intersectionAComplementB = {-1, 0, 1, 2} := by
  sorry

end NUMINAMATH_GPT_proof_intersection_l1460_146008


namespace NUMINAMATH_GPT_find_starting_number_l1460_146074

theorem find_starting_number : 
  ∃ x : ℕ, (∀ k : ℕ, (k < 12 → (x + 3 * k) ≤ 46) ∧ 12 = (46 - x) / 3 + 1) 
  ∧ x = 12 := 
by 
  sorry

end NUMINAMATH_GPT_find_starting_number_l1460_146074


namespace NUMINAMATH_GPT_proof_case_a_proof_case_b_l1460_146014

noncomputable def proof_problem_a (x y z p q : ℝ) (n : ℕ) 
  (h1 : y = x^n + p*x + q) 
  (h2 : z = y^n + p*y + q) 
  (h3 : x = z^n + p*z + q) : Prop :=
  x^2 * y + y^2 * z + z^2 * x >= x^2 * z + y^2 * x + z^2 * y

theorem proof_case_a (x y z p q : ℝ) 
  (h1 : y = x^2 + p*x + q) 
  (h2 : z = y^2 + p*y + q) 
  (h3 : x = z^2 + p*z + q) : 
  proof_problem_a x y z p q 2 h1 h2 h3 := 
sorry

theorem proof_case_b (x y z p q : ℝ) 
  (h1 : y = x^2010 + p*x + q) 
  (h2 : z = y^2010 + p*y + q) 
  (h3 : x = z^2010 + p*z + q) : 
  proof_problem_a x y z p q 2010 h1 h2 h3 := 
sorry

end NUMINAMATH_GPT_proof_case_a_proof_case_b_l1460_146014


namespace NUMINAMATH_GPT_unique_three_digit_numbers_l1460_146060

theorem unique_three_digit_numbers (d1 d2 d3 : ℕ) :
  (d1 = 3 ∧ d2 = 0 ∧ d3 = 8) →
  ∃ nums : Finset ℕ, 
  (∀ n ∈ nums, (∃ h t u : ℕ, n = 100 * h + 10 * t + u ∧ 
                h ≠ 0 ∧ (h = d1 ∨ h = d2 ∨ h = d3) ∧ 
                (t = d1 ∨ t = d2 ∨ t = d3) ∧ (u = d1 ∨ u = d2 ∨ u = d3) ∧ 
                h ≠ t ∧ t ≠ u ∧ u ≠ h)) ∧ nums.card = 4 :=
by
  sorry

end NUMINAMATH_GPT_unique_three_digit_numbers_l1460_146060


namespace NUMINAMATH_GPT_remainder_of_product_mod_7_l1460_146023

   theorem remainder_of_product_mod_7 :
     (7 * 17 * 27 * 37 * 47 * 57 * 67) % 7 = 0 := 
   by
     sorry
   
end NUMINAMATH_GPT_remainder_of_product_mod_7_l1460_146023


namespace NUMINAMATH_GPT_no_k_such_that_a_divides_2k_plus_1_and_b_divides_2k_minus_1_l1460_146092

theorem no_k_such_that_a_divides_2k_plus_1_and_b_divides_2k_minus_1 :
  ∀ (a b n : ℕ), (a > 1) → (b > 1) → (a ∣ 2^n - 1) → (b ∣ 2^n + 1) → ∀ (k : ℕ), ¬ (a ∣ 2^k + 1 ∧ b ∣ 2^k - 1) :=
by
  intros a b n a_gt_1 b_gt_1 a_div_2n_minus_1 b_div_2n_plus_1 k
  sorry

end NUMINAMATH_GPT_no_k_such_that_a_divides_2k_plus_1_and_b_divides_2k_minus_1_l1460_146092


namespace NUMINAMATH_GPT_complex_problem_l1460_146085

open Complex

noncomputable def z : ℂ := (1 + I) / Real.sqrt 2

theorem complex_problem :
  1 + z^50 + z^100 = I := 
by
  -- Subproofs or transformations will be here.
  sorry

end NUMINAMATH_GPT_complex_problem_l1460_146085


namespace NUMINAMATH_GPT_least_number_subtracted_l1460_146079

theorem least_number_subtracted (n : ℕ) (x : ℕ) (h_n : n = 4273981567) (h_x : x = 17) : 
  (n - x) % 25 = 0 := by
  sorry

end NUMINAMATH_GPT_least_number_subtracted_l1460_146079


namespace NUMINAMATH_GPT_symmetric_points_x_axis_l1460_146035

theorem symmetric_points_x_axis (a b : ℝ) (P Q : ℝ × ℝ)
  (hP : P = (a + 2, -2))
  (hQ : Q = (4, b))
  (hx : (a + 2) = 4)
  (hy : b = 2) :
  (a^b) = 4 := by
sorry

end NUMINAMATH_GPT_symmetric_points_x_axis_l1460_146035


namespace NUMINAMATH_GPT_num_pairs_of_nat_numbers_satisfying_eq_l1460_146012

theorem num_pairs_of_nat_numbers_satisfying_eq (n : ℕ) :
  n = 5 ↔ ∃ (a b : ℕ), a ≥ b ∧ (1/a : ℚ) + (1/b : ℚ) = (1/6 : ℚ) := sorry

end NUMINAMATH_GPT_num_pairs_of_nat_numbers_satisfying_eq_l1460_146012


namespace NUMINAMATH_GPT_gardner_bakes_brownies_l1460_146084

theorem gardner_bakes_brownies : 
  ∀ (cookies cupcakes brownies students sweet_treats_per_student total_sweet_treats total_cookies_and_cupcakes : ℕ),
  cookies = 20 →
  cupcakes = 25 →
  students = 20 →
  sweet_treats_per_student = 4 →
  total_sweet_treats = students * sweet_treats_per_student →
  total_cookies_and_cupcakes = cookies + cupcakes →
  brownies = total_sweet_treats - total_cookies_and_cupcakes →
  brownies = 35 :=
by
  intros cookies cupcakes brownies students sweet_treats_per_student total_sweet_treats total_cookies_and_cupcakes
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_gardner_bakes_brownies_l1460_146084


namespace NUMINAMATH_GPT_company_employees_after_reduction_l1460_146075

theorem company_employees_after_reduction :
  let original_number := 224.13793103448276
  let reduction := 0.13 * original_number
  let current_number := original_number - reduction
  current_number = 195 :=
by
  let original_number := 224.13793103448276
  let reduction := 0.13 * original_number
  let current_number := original_number - reduction
  sorry

end NUMINAMATH_GPT_company_employees_after_reduction_l1460_146075


namespace NUMINAMATH_GPT_specified_time_eq_l1460_146066

def distance : ℕ := 900
def ts (x : ℕ) : ℕ := x + 1
def tf (x : ℕ) : ℕ := x - 3

theorem specified_time_eq (x : ℕ) (h1 : x > 3) : 
  (distance / tf x) = 2 * (distance / ts x) :=
sorry

end NUMINAMATH_GPT_specified_time_eq_l1460_146066


namespace NUMINAMATH_GPT_jack_correct_percentage_l1460_146065

theorem jack_correct_percentage (y : ℝ) (h : y ≠ 0) :
  ((8 * y - (2 * y - 3)) / (8 * y)) * 100 = 75 + (75 / (2 * y)) :=
by
  sorry

end NUMINAMATH_GPT_jack_correct_percentage_l1460_146065


namespace NUMINAMATH_GPT_row_3_seat_6_representation_l1460_146028

-- Given Conditions
def seat_representation (r : ℕ) (s : ℕ) : (ℕ × ℕ) :=
  (r, s)

-- Proof Statement
theorem row_3_seat_6_representation :
  seat_representation 3 6 = (3, 6) :=
by
  sorry

end NUMINAMATH_GPT_row_3_seat_6_representation_l1460_146028


namespace NUMINAMATH_GPT_find_value_of_x_l1460_146072

theorem find_value_of_x (b : ℕ) (x : ℝ) (h_b_pos : b > 0) (h_x_pos : x > 0) 
  (h_r1 : r = 4 ^ (2 * b)) (h_r2 : r = 2 ^ b * x ^ b) : x = 8 :=
by
  -- Proof omitted for brevity
  sorry

end NUMINAMATH_GPT_find_value_of_x_l1460_146072


namespace NUMINAMATH_GPT_sqrt_six_estimation_l1460_146024

theorem sqrt_six_estimation : 2 < Real.sqrt 6 ∧ Real.sqrt 6 < 3 :=
by 
  sorry

end NUMINAMATH_GPT_sqrt_six_estimation_l1460_146024


namespace NUMINAMATH_GPT_min_operator_result_l1460_146047

theorem min_operator_result : 
  min ((-3) + (-6)) (min ((-3) - (-6)) (min ((-3) * (-6)) ((-3) / (-6)))) = -9 := 
by 
  sorry

end NUMINAMATH_GPT_min_operator_result_l1460_146047


namespace NUMINAMATH_GPT_letter_addition_problem_l1460_146073

theorem letter_addition_problem (S I X : ℕ) (E L V N : ℕ) 
  (hS : S = 8) 
  (hX_odd : X % 2 = 1)
  (h_diff_digits : ∀ (a b c d e f : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ e ∧ e ≠ f ∧ f ≠ a)
  (h_sum : 2 * S * 100 + 2 * I * 10 + 2 * X = E * 10000 + L * 1000 + E * 100 + V * 10 + E + N) :
  I = 3 :=
by
  sorry

end NUMINAMATH_GPT_letter_addition_problem_l1460_146073


namespace NUMINAMATH_GPT_percentage_reduction_in_price_l1460_146010

variable (R P : ℝ) (R_eq : R = 30) (H : 600 / R - 600 / P = 4)

theorem percentage_reduction_in_price (R_eq : R = 30) (H : 600 / R - 600 / P = 4) :
  ((P - R) / P) * 100 = 20 := sorry

end NUMINAMATH_GPT_percentage_reduction_in_price_l1460_146010


namespace NUMINAMATH_GPT_total_tiles_l1460_146087

/-- A square-shaped floor is covered with congruent square tiles. 
If the total number of tiles on the two diagonals is 88 and the floor 
forms a perfect square with an even side length, then the number of tiles 
covering the floor is 1936. -/
theorem total_tiles (n : ℕ) (hn_even : n % 2 = 0) (h_diag : 2 * n = 88) : n^2 = 1936 := 
by 
  sorry

end NUMINAMATH_GPT_total_tiles_l1460_146087


namespace NUMINAMATH_GPT_JessicaPathsAvoidRiskySite_l1460_146077

-- Definitions for the conditions.
def West (x y : ℕ) : Prop := (x > 0)
def East (x y : ℕ) : Prop := (x < 4)
def North (x y : ℕ) : Prop := (y < 3)
def AtOrigin (x y : ℕ) : Prop := (x = 0 ∧ y = 0)
def AtAnna (x y : ℕ) : Prop := (x = 4 ∧ y = 3)
def RiskySite (x y : ℕ) : Prop := (x = 2 ∧ y = 1)

-- Function to calculate binomial coefficient, binom(n, k)
def binom : ℕ → ℕ → ℕ
  | n, 0 => 1
  | 0, k + 1 => 0
  | n + 1, k + 1 => binom n k + binom n (k + 1)

-- Number of total valid paths avoiding the risky site.
theorem JessicaPathsAvoidRiskySite :
  let totalPaths := binom 7 4
  let pathsThroughRisky := binom 3 2 * binom 4 2
  (totalPaths - pathsThroughRisky) = 17 :=
by
  sorry

end NUMINAMATH_GPT_JessicaPathsAvoidRiskySite_l1460_146077


namespace NUMINAMATH_GPT_sequence_sum_relation_l1460_146015

theorem sequence_sum_relation (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (∀ n, 4 * S n = (a n + 1) ^ 2) →
  (S 1 = a 1) →
  a 1 = 1 →
  (∀ n, a (n + 1) = a n + 2) →
  a 2023 = 4045 :=
by
  sorry

end NUMINAMATH_GPT_sequence_sum_relation_l1460_146015


namespace NUMINAMATH_GPT_range_of_independent_variable_l1460_146022

theorem range_of_independent_variable (x : ℝ) : x ≠ -3 ↔ ∃ y : ℝ, y = 1 / (x + 3) :=
by 
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_range_of_independent_variable_l1460_146022


namespace NUMINAMATH_GPT_problem_statement_l1460_146058

-- Assume F is a function defined such that given the point (4,4) is on the graph y = F(x)
def F : ℝ → ℝ := sorry

-- Hypothesis: (4, 4) is on the graph of y = F(x)
axiom H : F 4 = 4

-- We need to prove that F(4) = 4
theorem problem_statement : F 4 = 4 :=
by exact H

end NUMINAMATH_GPT_problem_statement_l1460_146058


namespace NUMINAMATH_GPT_proof_problem_l1460_146057

-- Necessary types and noncomputable definitions
noncomputable def a_seq : ℕ → ℕ := sorry
noncomputable def b_seq : ℕ → ℕ := sorry

-- The conditions in the problem are used as assumptions
axiom partition : ∀ (n : ℕ), n > 0 → a_seq n < a_seq (n + 1)
axiom b_def : ∀ (n : ℕ), n > 0 → b_seq n = a_seq n + n

-- The mathematical equivalent proof problem stated
theorem proof_problem (n : ℕ) (hn : n > 0) : a_seq n + b_seq n = a_seq (b_seq n) :=
sorry

end NUMINAMATH_GPT_proof_problem_l1460_146057


namespace NUMINAMATH_GPT_projection_inequality_l1460_146097

theorem projection_inequality
  (a b c : ℝ)
  (h : c^2 = a^2 + b^2) :
  c ≥ (a + b) / Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_projection_inequality_l1460_146097


namespace NUMINAMATH_GPT_percentage_of_women_in_study_group_l1460_146067

theorem percentage_of_women_in_study_group
  (W : ℝ) -- percentage of women in decimal form
  (h1 : 0 < W ∧ W ≤ 1) -- percentage of women should be between 0 and 1
  (h2 : 0.4 * W = 0.32) -- 40 percent of women are lawyers, and probability is 0.32
  : W = 0.8 :=
  sorry

end NUMINAMATH_GPT_percentage_of_women_in_study_group_l1460_146067


namespace NUMINAMATH_GPT_choir_minimum_members_l1460_146098

theorem choir_minimum_members (n : ℕ) :
  (∃ k1, n = 8 * k1) ∧ (∃ k2, n = 9 * k2) ∧ (∃ k3, n = 10 * k3) → n = 360 :=
by
  sorry

end NUMINAMATH_GPT_choir_minimum_members_l1460_146098


namespace NUMINAMATH_GPT_determine_number_of_quarters_l1460_146091

def number_of_coins (Q D : ℕ) : Prop := Q + D = 23

def total_value (Q D : ℕ) : Prop := 25 * Q + 10 * D = 335

theorem determine_number_of_quarters (Q D : ℕ) 
  (h1 : number_of_coins Q D) 
  (h2 : total_value Q D) : 
  Q = 7 :=
by
  -- Equating and simplifying using h2, we find 15Q = 105, hence Q = 7
  sorry

end NUMINAMATH_GPT_determine_number_of_quarters_l1460_146091


namespace NUMINAMATH_GPT_prob_t_prob_vowel_l1460_146009

def word := "mathematics"
def total_letters : ℕ := 11
def t_count : ℕ := 2
def vowel_count : ℕ := 4

-- Definition of being a letter "t"
def is_t (c : Char) : Prop := c = 't'

-- Definition of being a vowel
def is_vowel (c : Char) : Prop := c = 'a' ∨ c = 'e' ∨ c = 'i'

theorem prob_t : (t_count : ℚ) / total_letters = 2 / 11 :=
by
  sorry

theorem prob_vowel : (vowel_count : ℚ) / total_letters = 4 / 11 :=
by
  sorry

end NUMINAMATH_GPT_prob_t_prob_vowel_l1460_146009


namespace NUMINAMATH_GPT_avg_age_all_l1460_146044

-- Define the conditions
def avg_age_seventh_graders (n₁ : Nat) (a₁ : Nat) : Prop :=
  n₁ = 40 ∧ a₁ = 13

def avg_age_parents (n₂ : Nat) (a₂ : Nat) : Prop :=
  n₂ = 50 ∧ a₂ = 40

-- Define the problem to prove
def avg_age_combined (n₁ n₂ a₁ a₂ : Nat) : Prop :=
  (n₁ * a₁ + n₂ * a₂) / (n₁ + n₂) = 28

-- The main theorem
theorem avg_age_all (n₁ n₂ a₁ a₂ : Nat):
  avg_age_seventh_graders n₁ a₁ → avg_age_parents n₂ a₂ → avg_age_combined n₁ n₂ a₁ a₂ :=
by 
  intros h1 h2
  sorry

end NUMINAMATH_GPT_avg_age_all_l1460_146044


namespace NUMINAMATH_GPT_current_ratio_of_employees_l1460_146055

-- Definitions for the number of current male employees and the ratio if 3 more men are hired
variables (M : ℕ) (F : ℕ)
variables (hM : M = 189)
variables (ratio_hired : (M + 3) / F = 8 / 9)

-- Conclusion we want to prove
theorem current_ratio_of_employees (M F : ℕ) (hM : M = 189) (ratio_hired : (M + 3) / F = 8 / 9) : 
  M / F = 7 / 8 :=
sorry

end NUMINAMATH_GPT_current_ratio_of_employees_l1460_146055


namespace NUMINAMATH_GPT_value_of_expression_l1460_146050

theorem value_of_expression (x : ℝ) (h : 2 * x^2 + 3 * x + 7 = 8) : 4 * x^2 + 6 * x - 9 = -7 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1460_146050
