import Mathlib

namespace NUMINAMATH_GPT_evaluate_neg2012_l1881_188140

def func (a b c x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 1

theorem evaluate_neg2012 (a b c : ℝ) (h : func a b c 2012 = 3) : func a b c (-2012) = -1 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_neg2012_l1881_188140


namespace NUMINAMATH_GPT_integer_triples_condition_l1881_188144

theorem integer_triples_condition (p q r : ℤ) (h1 : 1 < p) (h2 : p < q) (h3 : q < r) 
  (h4 : ((p - 1) * (q - 1) * (r - 1)) ∣ (p * q * r - 1)) : (p = 2 ∧ q = 4 ∧ r = 8) ∨ (p = 3 ∧ q = 5 ∧ r = 15) :=
sorry

end NUMINAMATH_GPT_integer_triples_condition_l1881_188144


namespace NUMINAMATH_GPT_total_team_formation_plans_l1881_188143

def numberOfWaysToChooseDoctors (m f : ℕ) (k : ℕ) : ℕ :=
  (Nat.choose m (k - 1) * Nat.choose f 1) +
  (Nat.choose m 1 * Nat.choose f (k - 1))

theorem total_team_formation_plans :
  let m := 5
  let f := 4
  let total := 3
  numberOfWaysToChooseDoctors m f total = 70 :=
by
  let m := 5
  let f := 4
  let total := 3
  unfold numberOfWaysToChooseDoctors
  sorry

end NUMINAMATH_GPT_total_team_formation_plans_l1881_188143


namespace NUMINAMATH_GPT_hyperbola_condition_l1881_188109

theorem hyperbola_condition (k : ℝ) : (3 - k) * (k - 2) < 0 ↔ k < 2 ∨ k > 3 := by
  sorry

end NUMINAMATH_GPT_hyperbola_condition_l1881_188109


namespace NUMINAMATH_GPT_petya_cannot_win_l1881_188157

theorem petya_cannot_win (n : ℕ) (h : n ≥ 3) : ¬ ∃ strategy : ℕ → ℕ → Prop, 
  (∀ k, strategy k (k+1) ∧ strategy k (k-1))
  ∧ ∀ m, ¬ strategy n m :=
sorry

end NUMINAMATH_GPT_petya_cannot_win_l1881_188157


namespace NUMINAMATH_GPT_woman_wait_time_to_be_caught_l1881_188122

theorem woman_wait_time_to_be_caught 
  (man_speed_mph : ℝ) (woman_speed_mph : ℝ) (wait_time_minutes : ℝ) 
  (conversion_factor : ℝ) (distance_apart_miles : ℝ) :
  man_speed_mph = 6 →
  woman_speed_mph = 12 →
  wait_time_minutes = 10 →
  conversion_factor = 1 / 60 →
  distance_apart_miles = (woman_speed_mph * conversion_factor) * wait_time_minutes →
  ∃ minutes_to_catch_up : ℝ, minutes_to_catch_up = distance_apart_miles / (man_speed_mph * conversion_factor) ∧ minutes_to_catch_up = 20 := sorry

end NUMINAMATH_GPT_woman_wait_time_to_be_caught_l1881_188122


namespace NUMINAMATH_GPT_line_through_points_l1881_188181

theorem line_through_points (x1 y1 x2 y2 : ℝ) (m b : ℝ) 
  (h1 : x1 = -3) (h2 : y1 = 1) (h3 : x2 = 1) (h4 : y2 = 3)
  (h5 : y1 = m * x1 + b) (h6 : y2 = m * x2 + b) :
  m + b = 3 := 
sorry

end NUMINAMATH_GPT_line_through_points_l1881_188181


namespace NUMINAMATH_GPT_value_of_a_b_l1881_188104

theorem value_of_a_b:
  ∃ (a b : ℕ), a = 3 ∧ b = 2 ∧ (a + 6 * 10^3 + 7 * 10^2 + 9 * 10 + b) % 72 = 0 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_b_l1881_188104


namespace NUMINAMATH_GPT_least_cost_planting_l1881_188168

theorem least_cost_planting :
  let region1_area := 3 * 1
  let region2_area := 4 * 4
  let region3_area := 7 * 2
  let region4_area := 5 * 4
  let region5_area := 5 * 6
  let easter_lilies_cost_per_sqft := 3.25
  let dahlias_cost_per_sqft := 2.75
  let cannas_cost_per_sqft := 2.25
  let begonias_cost_per_sqft := 1.75
  let asters_cost_per_sqft := 1.25
  region1_area * easter_lilies_cost_per_sqft +
  region2_area * dahlias_cost_per_sqft +
  region3_area * cannas_cost_per_sqft +
  region4_area * begonias_cost_per_sqft +
  region5_area * asters_cost_per_sqft =
  156.75 := 
sorry

end NUMINAMATH_GPT_least_cost_planting_l1881_188168


namespace NUMINAMATH_GPT_eggs_today_l1881_188183

-- Condition definitions
def eggs_yesterday : ℕ := 10
def difference : ℕ := 59

-- Statement of the problem
theorem eggs_today : eggs_yesterday + difference = 69 := by
  sorry

end NUMINAMATH_GPT_eggs_today_l1881_188183


namespace NUMINAMATH_GPT_find_f_2014_l1881_188106

noncomputable def f (x : ℝ) : ℝ := sorry

axiom functional_eqn : ∀ x : ℝ, f x = f (x + 1) - f (x + 2)
axiom interval_def : ∀ x, 0 < x ∧ x < 3 → f x = x^2

theorem find_f_2014 : f 2014 = -1 := sorry

end NUMINAMATH_GPT_find_f_2014_l1881_188106


namespace NUMINAMATH_GPT_altitude_inequality_l1881_188197

theorem altitude_inequality
  (a b m_a m_b : ℝ)
  (h1 : a > b)
  (h2 : a * m_a = b * m_b) :
  a^2010 + m_a^2010 ≥ b^2010 + m_b^2010 :=
sorry

end NUMINAMATH_GPT_altitude_inequality_l1881_188197


namespace NUMINAMATH_GPT_directrix_of_parabola_l1881_188117

-- Define the given condition:
def parabola_eq (x : ℝ) : ℝ := 8 * x^2 + 4 * x + 2

-- State the theorem:
theorem directrix_of_parabola :
  (∀ x : ℝ, parabola_eq x = 8 * (x + 1/4)^2 + 1) → (y = 31 / 32) :=
by
  -- We'll prove this later
  sorry

end NUMINAMATH_GPT_directrix_of_parabola_l1881_188117


namespace NUMINAMATH_GPT_find_n_l1881_188126

theorem find_n (n : ℤ) 
  (h : (3 + 16 + 33 + (n + 1)) / 4 = 20) : n = 27 := 
by
  sorry

end NUMINAMATH_GPT_find_n_l1881_188126


namespace NUMINAMATH_GPT_y_x_cubed_monotonic_increasing_l1881_188121

theorem y_x_cubed_monotonic_increasing : 
  ∀ x1 x2 : ℝ, (x1 ≤ x2) → (x1^3 ≤ x2^3) :=
by
  intros x1 x2 h
  sorry

end NUMINAMATH_GPT_y_x_cubed_monotonic_increasing_l1881_188121


namespace NUMINAMATH_GPT_arithmetic_identity_l1881_188159

theorem arithmetic_identity : 45 * 27 + 73 * 45 = 4500 := by sorry

end NUMINAMATH_GPT_arithmetic_identity_l1881_188159


namespace NUMINAMATH_GPT_problem_l1881_188198

theorem problem (a : ℕ) (b : ℚ) (c : ℤ) 
  (h1 : a = 1) 
  (h2 : b = 0) 
  (h3 : abs (c) = 6) :
  (a - b + c = (7 : ℤ)) ∨ (a - b + c = (-5 : ℤ)) := by
  sorry

end NUMINAMATH_GPT_problem_l1881_188198


namespace NUMINAMATH_GPT_probability_approx_l1881_188110

noncomputable def circumscribed_sphere_volume (R : ℝ) : ℝ :=
  (4 / 3) * Real.pi * R^3

noncomputable def single_sphere_volume (R : ℝ) : ℝ :=
  (4 / 3) * Real.pi * (R / 3)^3

noncomputable def total_spheres_volume (R : ℝ) : ℝ :=
  6 * single_sphere_volume R

noncomputable def probability_inside_spheres (R : ℝ) : ℝ :=
  total_spheres_volume R / circumscribed_sphere_volume R

theorem probability_approx (R : ℝ) (hR : R > 0) : 
  abs (probability_inside_spheres R - 0.053) < 0.001 := sorry

end NUMINAMATH_GPT_probability_approx_l1881_188110


namespace NUMINAMATH_GPT_area_ratio_trapezoid_abm_abcd_l1881_188194

-- Definitions based on conditions
variables {A B C D M : Type} [Zero A] [Zero B] [Zero C] [Zero D] [Zero M]
variables (BC AD : ℝ)

-- Condition: ABCD is a trapezoid with BC parallel to AD and diagonals AC and BD intersect M
-- Given BC = b and AD = a

-- Theorem statement
theorem area_ratio_trapezoid_abm_abcd (a b : ℝ) (h1 : BC = b) (h2 : AD = a) : 
  ∃ S_ABM S_ABCD : ℝ,
  (S_ABM / S_ABCD = a * b / (a + b)^2) :=
sorry

end NUMINAMATH_GPT_area_ratio_trapezoid_abm_abcd_l1881_188194


namespace NUMINAMATH_GPT_Tim_running_hours_per_week_l1881_188190

noncomputable def running_time_per_week : ℝ :=
  let MWF_morning : ℝ := (1 * 60 + 20 - 10) / 60 -- minutes to hours
  let MWF_evening : ℝ := (45 - 10) / 60 -- minutes to hours
  let TS_morning : ℝ := (1 * 60 + 5 - 10) / 60 -- minutes to hours
  let TS_evening : ℝ := (50 - 10) / 60 -- minutes to hours
  let MWF_total : ℝ := (MWF_morning + MWF_evening) * 3
  let TS_total : ℝ := (TS_morning + TS_evening) * 2
  MWF_total + TS_total

theorem Tim_running_hours_per_week : running_time_per_week = 8.42 := by
  -- Add the detailed proof here
  sorry

end NUMINAMATH_GPT_Tim_running_hours_per_week_l1881_188190


namespace NUMINAMATH_GPT_determine_A_l1881_188156

theorem determine_A (x y A : ℝ) 
  (h : (x + y) ^ 3 - x * y * (x + y) = (x + y) * A) : 
  A = x^2 + x * y + y^2 := 
by
  sorry

end NUMINAMATH_GPT_determine_A_l1881_188156


namespace NUMINAMATH_GPT_mean_of_remaining_two_l1881_188135

def seven_numbers := [1865, 1990, 2015, 2023, 2105, 2120, 2135]

def mean (l : List ℕ) : ℕ :=
  l.sum / l.length

theorem mean_of_remaining_two
  (h : mean (seven_numbers.take 5) = 2043) :
  mean (seven_numbers.drop 5) = 969 :=
by
  sorry

end NUMINAMATH_GPT_mean_of_remaining_two_l1881_188135


namespace NUMINAMATH_GPT_total_number_of_elements_l1881_188147

theorem total_number_of_elements (a b c : ℕ) : 
  (a = 2 ∧ b = 2 ∧ c = 2) ∧ 
  (3.95 = ((4.4 * 2 + 3.85 * 2 + 3.6000000000000014 * 2) / 6)) ->
  a + b + c = 6 := 
by
  sorry

end NUMINAMATH_GPT_total_number_of_elements_l1881_188147


namespace NUMINAMATH_GPT_cannot_form_right_triangle_setA_l1881_188162

def is_right_triangle (a b c : ℝ) : Prop :=
  (a^2 + b^2 = c^2) ∨ (a^2 + c^2 = b^2) ∨ (b^2 + c^2 = a^2)

theorem cannot_form_right_triangle_setA (a b c : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : c = 4) :
  ¬ is_right_triangle a b c :=
by {
  sorry
}

end NUMINAMATH_GPT_cannot_form_right_triangle_setA_l1881_188162


namespace NUMINAMATH_GPT_length_of_second_square_l1881_188179

-- Define conditions as variables
def Area_flag := 135
def Area_square1 := 40
def Area_square3 := 25

-- Define the length variable for the second square
variable (L : ℕ)

-- Define the area of the second square in terms of L
def Area_square2 : ℕ := 7 * L

-- Lean statement to be proved
theorem length_of_second_square :
  Area_square1 + Area_square2 L + Area_square3 = Area_flag → L = 10 :=
by sorry

end NUMINAMATH_GPT_length_of_second_square_l1881_188179


namespace NUMINAMATH_GPT_number_of_perfect_square_factors_l1881_188152

theorem number_of_perfect_square_factors (a b c d : ℕ) :
  (∀ a b c d, 
    (0 ≤ a ∧ a ≤ 4) ∧ 
    (0 ≤ b ∧ b ≤ 2) ∧ 
    (0 ≤ c ∧ c ≤ 1) ∧ 
    (0 ≤ d ∧ d ≤ 1) ∧ 
    (a % 2 = 0) ∧ 
    (b % 2 = 0) ∧ 
    (c = 0) ∧ 
    (d = 0)
  → 3 * 2 * 1 * 1 = 6) := by
  sorry

end NUMINAMATH_GPT_number_of_perfect_square_factors_l1881_188152


namespace NUMINAMATH_GPT_baker_sold_cakes_l1881_188108

def initialCakes : Nat := 110
def additionalCakes : Nat := 76
def remainingCakes : Nat := 111
def cakesSold : Nat := 75

theorem baker_sold_cakes :
  initialCakes + additionalCakes - remainingCakes = cakesSold := by
  sorry

end NUMINAMATH_GPT_baker_sold_cakes_l1881_188108


namespace NUMINAMATH_GPT_xiaoqiang_average_score_l1881_188167

theorem xiaoqiang_average_score
    (x : ℕ)
    (prev_avg : ℝ)
    (next_score : ℝ)
    (target_avg : ℝ)
    (h_prev_avg : prev_avg = 84)
    (h_next_score : next_score = 100)
    (h_target_avg : target_avg = 86) :
    (86 * x - (84 * (x - 1)) = 100) → x = 8 := 
by
  intros h_eq
  sorry

end NUMINAMATH_GPT_xiaoqiang_average_score_l1881_188167


namespace NUMINAMATH_GPT_coefficient_x_squared_in_expansion_l1881_188142

theorem coefficient_x_squared_in_expansion :
  (∃ c : ℤ, (1 + x)^6 * (1 - x) = c * x^2 + b * x + a) → c = 9 :=
by
  sorry

end NUMINAMATH_GPT_coefficient_x_squared_in_expansion_l1881_188142


namespace NUMINAMATH_GPT_forces_angle_result_l1881_188171

noncomputable def forces_angle_condition (p1 p2 p : ℝ) (α : ℝ) : Prop :=
  p^2 = p1 * p2

noncomputable def angle_condition_range (p1 p2 : ℝ) : Prop :=
  (3 - Real.sqrt 5) / 2 ≤ p1 / p2 ∧ p1 / p2 ≤ (3 + Real.sqrt 5) / 2

theorem forces_angle_result (p1 p2 p α : ℝ) (h : forces_angle_condition p1 p2 p α) :
  120 * π / 180 ≤ α ∧ α ≤ 120 * π / 180 ∧ (angle_condition_range p1 p2) := 
sorry

end NUMINAMATH_GPT_forces_angle_result_l1881_188171


namespace NUMINAMATH_GPT_shaded_fraction_of_rectangle_l1881_188185

theorem shaded_fraction_of_rectangle (a b : ℕ) (h_dim : a = 15 ∧ b = 24) (h_shaded : ∃ s, s = (1/3 : ℚ)) :
  ∃ f, f = (1/9 : ℚ) := 
by
  sorry

end NUMINAMATH_GPT_shaded_fraction_of_rectangle_l1881_188185


namespace NUMINAMATH_GPT_total_people_100_l1881_188191

noncomputable def total_people (P : ℕ) : Prop :=
  (2 / 5 : ℚ) * P = 40 ∧ (1 / 4 : ℚ) * P ≤ P ∧ P ≥ 40 

theorem total_people_100 {P : ℕ} (h : total_people P) : P = 100 := 
by 
  sorry -- proof would go here

end NUMINAMATH_GPT_total_people_100_l1881_188191


namespace NUMINAMATH_GPT_x_must_be_even_l1881_188113

theorem x_must_be_even (x : ℤ) (h : ∃ (n : ℤ), (2 * x / 3 - x / 6) = n) : ∃ (k : ℤ), x = 2 * k :=
by
  sorry

end NUMINAMATH_GPT_x_must_be_even_l1881_188113


namespace NUMINAMATH_GPT_rectangle_length_is_4_l1881_188199

theorem rectangle_length_is_4 (a : ℕ) (s : a = 4) (area_square : ℕ) 
(area_square_eq : area_square = a * a) 
(area_rectangle_eq : area_square = a * 4) : 
4 = a := by
  sorry

end NUMINAMATH_GPT_rectangle_length_is_4_l1881_188199


namespace NUMINAMATH_GPT_gcd_of_X_and_Y_l1881_188134

theorem gcd_of_X_and_Y (X Y : ℕ) (h1 : Nat.lcm X Y = 180) (h2 : 5 * X = 4 * Y) :
  Nat.gcd X Y = 9 := 
sorry

end NUMINAMATH_GPT_gcd_of_X_and_Y_l1881_188134


namespace NUMINAMATH_GPT_Tom_allowance_leftover_l1881_188139

theorem Tom_allowance_leftover :
  let initial_allowance := 12
  let first_week_spending := (1/3) * initial_allowance
  let remaining_after_first_week := initial_allowance - first_week_spending
  let second_week_spending := (1/4) * remaining_after_first_week
  let final_amount := remaining_after_first_week - second_week_spending
  final_amount = 6 :=
by
  sorry

end NUMINAMATH_GPT_Tom_allowance_leftover_l1881_188139


namespace NUMINAMATH_GPT_area_ratio_l1881_188101

noncomputable def initial_areas (a b c : ℝ) :=
  a > 0 ∧ b > 0 ∧ c > 0

noncomputable def misallocated_areas (a b : ℝ) :=
  let b' := b + 0.1 * a - 0.5 * b
  b' = 0.4 * (a + b)

noncomputable def final_ratios (a b c : ℝ) :=
  let a' := 0.9 * a + 0.5 * b
  let b' := b + 0.1 * a - 0.5 * b
  let c' := 0.5 * c
  a' + b' + c' = a + b + c ∧ a' / b' = 2 ∧ b' / c' = 1 

theorem area_ratio (a b c m : ℝ) (h1 : initial_areas a b c) 
  (h2 : misallocated_areas a b)
  (h3 : final_ratios a b c) : 
  (m = 0.4 * a) → (m / (a + b + c) = 1 / 20) :=
sorry

end NUMINAMATH_GPT_area_ratio_l1881_188101


namespace NUMINAMATH_GPT_image_of_center_after_transform_l1881_188187

structure Point where
  x : ℤ
  y : ℤ

def reflect_across_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

def translate_right (p : Point) (units : ℤ) : Point :=
  { x := p.x + units, y := p.y }

def transform_point (p : Point) : Point :=
  translate_right (reflect_across_x p) 5

theorem image_of_center_after_transform :
  transform_point {x := -3, y := 4} = {x := 2, y := -4} := by
  sorry

end NUMINAMATH_GPT_image_of_center_after_transform_l1881_188187


namespace NUMINAMATH_GPT_arithmetic_sum_calculation_l1881_188131

theorem arithmetic_sum_calculation :
  3 * (71 + 75 + 79 + 83 + 87 + 91) = 1458 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sum_calculation_l1881_188131


namespace NUMINAMATH_GPT_composite_numbers_characterization_l1881_188174

noncomputable def is_sum_and_product_seq (n : ℕ) (seq : List ℕ) : Prop :=
  seq.sum = n ∧ seq.prod = n ∧ 2 ≤ seq.length ∧ ∀ x ∈ seq, 1 ≤ x

theorem composite_numbers_characterization (n : ℕ) :
  (∃ seq : List ℕ, is_sum_and_product_seq n seq) ↔ ¬Nat.Prime n ∧ 1 < n :=
sorry

end NUMINAMATH_GPT_composite_numbers_characterization_l1881_188174


namespace NUMINAMATH_GPT_sphere_radius_l1881_188180

theorem sphere_radius (x y z r : ℝ) (h1 : 2 * x * y + 2 * y * z + 2 * z * x = 384)
  (h2 : x + y + z = 28) (h3 : (2 * r)^2 = x^2 + y^2 + z^2) : r = 10 := sorry

end NUMINAMATH_GPT_sphere_radius_l1881_188180


namespace NUMINAMATH_GPT_xena_head_start_l1881_188120

theorem xena_head_start
  (xena_speed : ℝ) (dragon_speed : ℝ) (time : ℝ) (burn_distance : ℝ) 
  (xena_speed_eq : xena_speed = 15) 
  (dragon_speed_eq : dragon_speed = 30) 
  (time_eq : time = 32) 
  (burn_distance_eq : burn_distance = 120) :
  (dragon_speed * time - burn_distance) - (xena_speed * time) = 360 := 
  by 
  sorry

end NUMINAMATH_GPT_xena_head_start_l1881_188120


namespace NUMINAMATH_GPT_sequence_product_l1881_188193

theorem sequence_product (a : ℕ → ℝ) (q : ℝ) (h : ∀ n, a (n + 1) = q * a n) (h₄ : a 4 = 2) :
  a 2 * a 3 * a 5 * a 6 = 16 :=
sorry

end NUMINAMATH_GPT_sequence_product_l1881_188193


namespace NUMINAMATH_GPT_diagonals_of_seven_sided_polygon_l1881_188150

-- Define the number of sides of the polygon
def n : ℕ := 7

-- Calculate the number of diagonals in a polygon with n sides
def numberOfDiagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

-- The statement to prove
theorem diagonals_of_seven_sided_polygon : numberOfDiagonals n = 14 := by
  -- Here we will write the proof steps, but they're not needed now.
  sorry

end NUMINAMATH_GPT_diagonals_of_seven_sided_polygon_l1881_188150


namespace NUMINAMATH_GPT_corresponding_angle_C1_of_similar_triangles_l1881_188153

theorem corresponding_angle_C1_of_similar_triangles
  (α β γ : ℝ)
  (ABC_sim_A1B1C1 : true)
  (angle_A : α = 50)
  (angle_B : β = 95) :
  γ = 35 :=
by
  sorry

end NUMINAMATH_GPT_corresponding_angle_C1_of_similar_triangles_l1881_188153


namespace NUMINAMATH_GPT_probability_z_l1881_188146

variable (p q x y z : ℝ)

-- Conditions
def condition1 : Prop := z = p * y + q * x
def condition2 : Prop := x = p + q * x^2
def condition3 : Prop := y = q + p * y^2
def condition4 : Prop := x ≠ y

-- Theorem Statement
theorem probability_z : condition1 p q x y z ∧ condition2 p q x ∧ condition3 p q y ∧ condition4 x y → z = 2 * q := by
  sorry

end NUMINAMATH_GPT_probability_z_l1881_188146


namespace NUMINAMATH_GPT_not_jog_probability_eq_l1881_188148

def P_jog : ℚ := 5 / 8

theorem not_jog_probability_eq :
  1 - P_jog = 3 / 8 :=
by
  sorry

end NUMINAMATH_GPT_not_jog_probability_eq_l1881_188148


namespace NUMINAMATH_GPT_rain_in_first_hour_l1881_188165

theorem rain_in_first_hour (x : ℝ) (h1 : ∀ y : ℝ, y = 2 * x + 7) (h2 : x + (2 * x + 7) = 22) : x = 5 :=
sorry

end NUMINAMATH_GPT_rain_in_first_hour_l1881_188165


namespace NUMINAMATH_GPT_num_carnations_l1881_188128

-- Define the conditions
def num_roses : ℕ := 5
def total_flowers : ℕ := 10

-- Define the statement we want to prove
theorem num_carnations : total_flowers - num_roses = 5 :=
by {
  -- The proof itself is not required, so we use 'sorry' to indicate incomplete proof
  sorry
}

end NUMINAMATH_GPT_num_carnations_l1881_188128


namespace NUMINAMATH_GPT_sequence_general_term_l1881_188118

variable {a : ℕ → ℚ}
variable {S : ℕ → ℚ}

theorem sequence_general_term (h : ∀ n : ℕ, S n = 2 * n - a n) :
  ∀ n : ℕ, a n = (2^n - 1) / (2^(n-1)) :=
by
  sorry

end NUMINAMATH_GPT_sequence_general_term_l1881_188118


namespace NUMINAMATH_GPT_price_of_water_margin_comics_l1881_188116

-- Define the conditions
variables (x : ℕ) (y : ℕ)

-- Condition 1: Price relationship
def price_relationship : Prop := y = x + 60

-- Condition 2: Total expenditure on Romance of the Three Kingdoms comic books
def total_expenditure_romance_three_kingdoms : Prop := 60 * (y / 60) = 3600

-- Condition 3: Total expenditure on Water Margin comic books
def total_expenditure_water_margin : Prop := 120 * (x / 120) = 4800

-- Condition 4: Number of sets relationship
def number_of_sets_relationship : Prop := y = (4800 / x) / 2

-- The main statement to prove
theorem price_of_water_margin_comics (x : ℕ) (h1: price_relationship x (x + 60))
  (h2: total_expenditure_romance_three_kingdoms x)
  (h3: total_expenditure_water_margin x)
  (h4: number_of_sets_relationship x (x + 60)) : x = 120 :=
sorry

end NUMINAMATH_GPT_price_of_water_margin_comics_l1881_188116


namespace NUMINAMATH_GPT_inequality_solution_set_l1881_188172

theorem inequality_solution_set (x : ℝ) : 3 ≤ abs (5 - 2 * x) ∧ abs (5 - 2 * x) < 9 ↔ (x > -2 ∧ x ≤ 1) ∨ (x ≥ 4 ∧ x < 7) := sorry

end NUMINAMATH_GPT_inequality_solution_set_l1881_188172


namespace NUMINAMATH_GPT_perp_line_eq_l1881_188169

theorem perp_line_eq (x y : ℝ) (c : ℝ) (hx : x = 1) (hy : y = 2) (hline : 2 * x + y - 5 = 0) :
  x - 2 * y + c = 0 ↔ c = 3 := 
by
  sorry

end NUMINAMATH_GPT_perp_line_eq_l1881_188169


namespace NUMINAMATH_GPT_favorable_probability_l1881_188102

noncomputable def probability_favorable_events (L : ℝ) : ℝ :=
  1 - (0.5 * (5 / 12 * L)^2 / (0.5 * L^2))

theorem favorable_probability (L : ℝ) (x y : ℝ)
  (h1 : 0 ≤ x) (h2 : x ≤ L)
  (h3 : 0 ≤ y) (h4 : y ≤ L)
  (h5 : 0 ≤ x + y) (h6 : x + y ≤ L)
  (h7 : x ≤ 5 / 12 * L) (h8 : y ≤ 5 / 12 * L)
  (h9 : x + y ≥ 7 / 12 * L) :
  probability_favorable_events L = 15 / 16 :=
by sorry

end NUMINAMATH_GPT_favorable_probability_l1881_188102


namespace NUMINAMATH_GPT_remainder_50_pow_50_mod_7_l1881_188160

theorem remainder_50_pow_50_mod_7 : (50^50) % 7 = 1 := by
  sorry

end NUMINAMATH_GPT_remainder_50_pow_50_mod_7_l1881_188160


namespace NUMINAMATH_GPT_box_volume_l1881_188111

theorem box_volume
  (L W H : ℝ)
  (h1 : L * W = 120)
  (h2 : W * H = 72)
  (h3 : L * H = 60) :
  L * W * H = 720 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_box_volume_l1881_188111


namespace NUMINAMATH_GPT_right_triangle_sqrt_l1881_188195

noncomputable def sqrt_2 := Real.sqrt 2
noncomputable def sqrt_3 := Real.sqrt 3
noncomputable def sqrt_5 := Real.sqrt 5

theorem right_triangle_sqrt: 
  (sqrt_2 ^ 2 + sqrt_3 ^ 2 = sqrt_5 ^ 2) :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_sqrt_l1881_188195


namespace NUMINAMATH_GPT_students_with_equal_scores_l1881_188189

theorem students_with_equal_scores 
  (n : ℕ)
  (scores : Fin n → Fin (n - 1)): 
  ∃ i j : Fin n, i ≠ j ∧ scores i = scores j := 
by 
  sorry

end NUMINAMATH_GPT_students_with_equal_scores_l1881_188189


namespace NUMINAMATH_GPT_perimeter_difference_zero_l1881_188158

theorem perimeter_difference_zero :
  let shape1_length := 4
  let shape1_width := 3
  let shape2_length := 6
  let shape2_width := 1
  let perimeter (l w : ℕ) := 2 * (l + w)
  perimeter shape1_length shape1_width = perimeter shape2_length shape2_width :=
by
  sorry

end NUMINAMATH_GPT_perimeter_difference_zero_l1881_188158


namespace NUMINAMATH_GPT_determine_function_l1881_188163

def satisfies_condition (f : ℤ → ℤ) : Prop :=
  ∀ a b : ℤ, f (2 * a) + 2 * f b = f (f (a + b))

theorem determine_function (f : ℤ → ℤ) (h : satisfies_condition f) :
  ∀ n : ℤ, f n = 0 ∨ ∃ K : ℤ, f n = 2 * n + K :=
sorry

end NUMINAMATH_GPT_determine_function_l1881_188163


namespace NUMINAMATH_GPT_total_number_of_fish_l1881_188186

-- Define the number of each type of fish
def goldfish : ℕ := 23
def blue_fish : ℕ := 15
def angelfish : ℕ := 8
def neon_tetra : ℕ := 12

-- Theorem stating the total number of fish
theorem total_number_of_fish : goldfish + blue_fish + angelfish + neon_tetra = 58 := by
  sorry

end NUMINAMATH_GPT_total_number_of_fish_l1881_188186


namespace NUMINAMATH_GPT_xyz_inequality_l1881_188161

theorem xyz_inequality (x y z : ℝ) (h : x^2 + y^2 + z^2 = 1) : 
  (x - y) * (y - z) * (x - z) ≤ 1 / Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_xyz_inequality_l1881_188161


namespace NUMINAMATH_GPT_triangle_area_upper_bound_l1881_188100

variable {α : Type u}
variable [LinearOrderedField α]
variable {A B C : α} -- Points A, B, C as elements of some field.

-- Definitions for the lengths of the sides, interpreted as scalar distances.
variable (AB AC : α)

-- Assume that AB and AC are lengths of sides of the triangle
-- Assume the area of the triangle is non-negative and does not exceed the specified bound.
theorem triangle_area_upper_bound (S : α) (habc : S = (1 / 2) * AB * AC) :
  S ≤ (1 / 2) * AB * AC := 
sorry

end NUMINAMATH_GPT_triangle_area_upper_bound_l1881_188100


namespace NUMINAMATH_GPT_radius_increase_l1881_188138

theorem radius_increase (ΔC : ℝ) (ΔC_eq : ΔC = 0.628) : Δr = 0.1 :=
by
  sorry

end NUMINAMATH_GPT_radius_increase_l1881_188138


namespace NUMINAMATH_GPT_sum_11_terms_l1881_188127

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop := 
  ∃ d, ∀ n, a (n + 1) = a n + d

def sum_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n / 2) * (a 1 + a n)

def condition (a : ℕ → ℝ) : Prop :=
  a 5 + a 7 = 14

-- Proof Problem
theorem sum_11_terms (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith_seq : arithmetic_sequence a)
  (h_sum_formula : sum_first_n_terms S a)
  (h_condition : condition a) :
  S 11 = 77 := 
sorry

end NUMINAMATH_GPT_sum_11_terms_l1881_188127


namespace NUMINAMATH_GPT_jade_handled_80_transactions_l1881_188141

variable (mabel anthony cal jade : ℕ)

-- Conditions
def mabel_transactions : mabel = 90 :=
by sorry

def anthony_transactions : anthony = mabel + (10 * mabel / 100) :=
by sorry

def cal_transactions : cal = 2 * anthony / 3 :=
by sorry

def jade_transactions : jade = cal + 14 :=
by sorry

-- Proof problem
theorem jade_handled_80_transactions :
  mabel = 90 →
  anthony = mabel + (10 * mabel / 100) →
  cal = 2 * anthony / 3 →
  jade = cal + 14 →
  jade = 80 :=
by
  intros
  subst_vars
  -- The proof steps would normally go here, but we leave it with sorry.
  sorry

end NUMINAMATH_GPT_jade_handled_80_transactions_l1881_188141


namespace NUMINAMATH_GPT_find_m_l1881_188178

namespace MathProof

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 5
def g (x : ℝ) (m : ℝ) : ℝ := x^2 - m * x - 8

-- State the problem
theorem find_m (m : ℝ) (h : f 5 - g 5 m = 15) : m = -15 := by
  sorry

end MathProof

end NUMINAMATH_GPT_find_m_l1881_188178


namespace NUMINAMATH_GPT_cube_volume_doubled_l1881_188103

theorem cube_volume_doubled (a : ℝ) (h : a > 0) : 
  ((2 * a)^3 - a^3) / a^3 = 7 :=
by
  sorry

end NUMINAMATH_GPT_cube_volume_doubled_l1881_188103


namespace NUMINAMATH_GPT_range_of_m_l1881_188182

theorem range_of_m (f : ℝ → ℝ) {m : ℝ} (h_dec : ∀ x y, -1 ≤ x ∧ x ≤ 1 ∧ -1 ≤ y ∧ y ≤ 1 ∧ x < y → f x ≥ f y)
  (h_ineq : f (m - 1) > f (2 * m - 1)) : 0 < m ∧ m ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1881_188182


namespace NUMINAMATH_GPT_probability_sum_8_twice_l1881_188192

-- Define a structure for the scenario: a 7-sided die.
structure Die7 :=
(sides : Fin 7)

-- Define a function to check if the sum of two dice equals 8.
def is_sum_8 (d1 d2 : Die7) : Prop :=
  (d1.sides.val + 1) + (d2.sides.val + 1) = 8

-- Define the probability of the event given the conditions.
def probability_event_twice (successes total_outcomes : ℕ) : ℚ :=
  (successes / total_outcomes) * (successes / total_outcomes)

-- The total number of outcomes when rolling two 7-sided dice.
def total_outcomes : ℕ := 7 * 7

-- The number of successful outcomes that yield a sum of 8 with two rolls.
def successful_outcomes : ℕ := 7

-- Main theorem statement to be proved.
theorem probability_sum_8_twice :
  probability_event_twice successful_outcomes total_outcomes = 1 / 49 :=
by
  -- Sorry to indicate that the proof is omitted.
  sorry

end NUMINAMATH_GPT_probability_sum_8_twice_l1881_188192


namespace NUMINAMATH_GPT_union_sets_l1881_188175

def A : Set ℕ := {2, 1, 3}
def B : Set ℕ := {2, 3, 5}

theorem union_sets : A ∪ B = {1, 2, 3, 5} := 
by {
  sorry
}

end NUMINAMATH_GPT_union_sets_l1881_188175


namespace NUMINAMATH_GPT_parallelogram_perimeter_l1881_188137

def perimeter_of_parallelogram (a b : ℝ) : ℝ :=
  2 * (a + b)

theorem parallelogram_perimeter
  (side1 side2 : ℝ)
  (h_side1 : side1 = 18)
  (h_side2 : side2 = 12) :
  perimeter_of_parallelogram side1 side2 = 60 := 
by
  sorry

end NUMINAMATH_GPT_parallelogram_perimeter_l1881_188137


namespace NUMINAMATH_GPT_stans_average_speed_l1881_188119

noncomputable def average_speed (distance1 distance2 distance3 : ℝ) (time1_hrs time1_mins time2 time3_hrs time3_mins : ℝ) : ℝ :=
  let total_distance := distance1 + distance2 + distance3
  let total_time := time1_hrs + time1_mins / 60 + time2 + time3_hrs + time3_mins / 60
  total_distance / total_time

theorem stans_average_speed  :
  average_speed 350 420 330 5 40 7 5 30 = 60.54 :=
by
  -- sorry block indicates missing proof
  sorry

end NUMINAMATH_GPT_stans_average_speed_l1881_188119


namespace NUMINAMATH_GPT_correct_equation_l1881_188176

-- Conditions:
def number_of_branches (x : ℕ) := x
def number_of_small_branches (x : ℕ) := x * x
def total_number (x : ℕ) := 1 + number_of_branches x + number_of_small_branches x

-- Proof Problem:
theorem correct_equation (x : ℕ) : total_number x = 43 → x^2 + x + 1 = 43 :=
by 
  sorry

end NUMINAMATH_GPT_correct_equation_l1881_188176


namespace NUMINAMATH_GPT_rain_ratio_l1881_188112

def monday_rain := 2 + 1 -- inches of rain on Monday
def wednesday_rain := 0 -- inches of rain on Wednesday
def thursday_rain := 1 -- inches of rain on Thursday
def average_rain_per_day := 4 -- daily average rain total
def days_in_week := 5 -- days in a week
def weekly_total_rain := average_rain_per_day * days_in_week

-- Theorem statement
theorem rain_ratio (tuesday_rain : ℝ) (friday_rain : ℝ) 
  (h1 : friday_rain = monday_rain + tuesday_rain + wednesday_rain + thursday_rain)
  (h2 : monday_rain + tuesday_rain + wednesday_rain + thursday_rain + friday_rain = weekly_total_rain) :
  tuesday_rain / monday_rain = 2 := 
sorry

end NUMINAMATH_GPT_rain_ratio_l1881_188112


namespace NUMINAMATH_GPT_geometric_sequence_value_l1881_188155

noncomputable def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

theorem geometric_sequence_value
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geo : geometric_sequence a r)
  (h_pos : ∀ n, a n > 0)
  (h_roots : ∀ (a1 a19 : ℝ), a1 = a 1 → a19 = a 19 → a1 * a19 = 16 ∧ a1 + a19 = 10) :
  a 8 * a 10 * a 12 = 64 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_value_l1881_188155


namespace NUMINAMATH_GPT_sum_of_numbers_l1881_188177

theorem sum_of_numbers (a b c : ℝ) :
  a^2 + b^2 + c^2 = 138 → ab + bc + ca = 131 → a + b + c = 20 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_numbers_l1881_188177


namespace NUMINAMATH_GPT_coeff_x3_in_expansion_l1881_188105

theorem coeff_x3_in_expansion : 
  ∃ c : ℕ, (c = 80) ∧ (∃ r : ℕ, r = 1 ∧ (2 * x + 1 / x) ^ 5 = (2 * x) ^ (5 - r) * (1 / x) ^ r)
:= sorry

end NUMINAMATH_GPT_coeff_x3_in_expansion_l1881_188105


namespace NUMINAMATH_GPT_find_k_l1881_188123

-- Define the lines as given in the problem
def line1 (k : ℝ) (x y : ℝ) : Prop := k * x + (1 - k) * y - 3 = 0
def line2 (k : ℝ) (x y : ℝ) : Prop := (k - 1) * x + (2 * k + 3) * y - 2 = 0

-- Define the condition for perpendicular lines
def perpendicular (k : ℝ) : Prop :=
  let slope1 := -k / (1 - k)
  let slope2 := -(k - 1) / (2 * k + 3)
  slope1 * slope2 = -1

-- Problem statement: Prove that the lines are perpendicular implies k == 1 or k == -3
theorem find_k (k : ℝ) : perpendicular k → (k = 1 ∨ k = -3) :=
sorry

end NUMINAMATH_GPT_find_k_l1881_188123


namespace NUMINAMATH_GPT_scaled_triangle_height_l1881_188125

theorem scaled_triangle_height (h b₁ h₁ b₂ h₂ : ℝ)
  (h₁_eq : h₁ = 6) (b₁_eq : b₁ = 12) (b₂_eq : b₂ = 8) :
  (b₁ / h₁ = b₂ / h₂) → h₂ = 4 :=
by
  -- Given conditions
  have h₁_eq : h₁ = 6 := h₁_eq
  have b₁_eq : b₁ = 12 := b₁_eq
  have b₂_eq : b₂ = 8 := b₂_eq
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_scaled_triangle_height_l1881_188125


namespace NUMINAMATH_GPT_algebraic_expression_value_l1881_188196

def algebraic_expression (a b : ℤ) :=
  a + 2 * b + 2 * (a + 2 * b) + 1

theorem algebraic_expression_value :
  algebraic_expression 1 (-1) = -2 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1881_188196


namespace NUMINAMATH_GPT_calculate_value_l1881_188145

theorem calculate_value (h : 2994 * 14.5 = 179) : 29.94 * 1.45 = 0.179 :=
by
  sorry

end NUMINAMATH_GPT_calculate_value_l1881_188145


namespace NUMINAMATH_GPT_exponentiated_value_l1881_188188

theorem exponentiated_value (x a b : ℝ) (h1 : x^a = 2) (h2 : x^b = 3) : x^(3 * a + b) = 24 := by
  sorry

end NUMINAMATH_GPT_exponentiated_value_l1881_188188


namespace NUMINAMATH_GPT_minimum_value_l1881_188170

noncomputable def min_value_condition (a b : ℝ) : Prop :=
  a + b = 1 / 2

theorem minimum_value (a b : ℝ) (h : min_value_condition a b) :
  (4 / a) + (1 / b) ≥ 18 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_l1881_188170


namespace NUMINAMATH_GPT_trapezoid_area_l1881_188151

theorem trapezoid_area (u l h : ℕ) (hu : u = 12) (hl : l = u + 4) (hh : h = 10) : 
  (1 / 2 : ℚ) * (u + l) * h = 140 := by
  sorry

end NUMINAMATH_GPT_trapezoid_area_l1881_188151


namespace NUMINAMATH_GPT_sum_max_min_interval_l1881_188124

def f (x : ℝ) : ℝ := 2 * x^2 - 6 * x + 1

theorem sum_max_min_interval (a b : ℝ) (h₁ : a = -1) (h₂ : b = 1) :
  let M := max (f a) (f b)
  let m := min (f a) (f b)
  M + m = 6 :=
by
  rw [h₁, h₂]
  let M := max (f (-1)) (f 1)
  let m := min (f (-1)) (f 1)
  sorry

end NUMINAMATH_GPT_sum_max_min_interval_l1881_188124


namespace NUMINAMATH_GPT_employee_payment_l1881_188164

theorem employee_payment (X Y : ℝ) 
  (h1 : X + Y = 880) 
  (h2 : X = 1.2 * Y) : Y = 400 := by
  sorry

end NUMINAMATH_GPT_employee_payment_l1881_188164


namespace NUMINAMATH_GPT_calculate_expression_evaluate_expression_l1881_188114

theorem calculate_expression (a : ℕ) (h : a = 2020) :
  (a^4 - 3*a^3*(a+1) + 4*a*(a+1)^3 - (a+1)^4 + 1) / (a*(a+1)) = a^2 + 4*a + 6 :=
by sorry

theorem evaluate_expression :
  (2020^2 + 4 * 2020 + 6) = 4096046 :=
by sorry

end NUMINAMATH_GPT_calculate_expression_evaluate_expression_l1881_188114


namespace NUMINAMATH_GPT_pond_water_after_evaporation_l1881_188130

theorem pond_water_after_evaporation 
  (I R D : ℕ) 
  (h_initial : I = 250)
  (h_evaporation_rate : R = 1)
  (h_days : D = 50) : 
  I - (R * D) = 200 := 
by 
  sorry

end NUMINAMATH_GPT_pond_water_after_evaporation_l1881_188130


namespace NUMINAMATH_GPT_total_shoes_tried_on_l1881_188115

variable (T : Type)
variable (store1 store2 store3 store4 : T)
variable (pair_of_shoes : T → ℕ)
variable (c1 : pair_of_shoes store1 = 7)
variable (c2 : pair_of_shoes store2 = pair_of_shoes store1 + 2)
variable (c3 : pair_of_shoes store3 = 0)
variable (c4 : pair_of_shoes store4 = 2 * (pair_of_shoes store1 + pair_of_shoes store2 + pair_of_shoes store3))

theorem total_shoes_tried_on (store1 store2 store3 store4 : T) (pair_of_shoes : T → ℕ) : 
  pair_of_shoes store1 = 7 →
  pair_of_shoes store2 = pair_of_shoes store1 + 2 →
  pair_of_shoes store3 = 0 →
  pair_of_shoes store4 = 2 * (pair_of_shoes store1 + pair_of_shoes store2 + pair_of_shoes store3) →
  pair_of_shoes store1 + pair_of_shoes store2 + pair_of_shoes store3 + pair_of_shoes store4 = 48 := by
  intro c1 c2 c3 c4
  sorry

end NUMINAMATH_GPT_total_shoes_tried_on_l1881_188115


namespace NUMINAMATH_GPT_find_m_l1881_188173

def setA (m : ℝ) : Set ℝ := {1, m - 2}
def setB : Set ℝ := {2}

theorem find_m (m : ℝ) (H : setA m ∩ setB = {2}) : m = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1881_188173


namespace NUMINAMATH_GPT_xyz_sum_is_22_l1881_188132

theorem xyz_sum_is_22 (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h1 : x * y = 24) (h2 : x * z = 48) (h3 : y * z = 72) : 
  x + y + z = 22 :=
sorry

end NUMINAMATH_GPT_xyz_sum_is_22_l1881_188132


namespace NUMINAMATH_GPT_cube_root_of_nine_irrational_l1881_188129

theorem cube_root_of_nine_irrational : ¬ ∃ (r : ℚ), r^3 = 9 :=
by sorry

end NUMINAMATH_GPT_cube_root_of_nine_irrational_l1881_188129


namespace NUMINAMATH_GPT_moon_radius_scientific_notation_l1881_188184

noncomputable def moon_radius : ℝ := 1738000

theorem moon_radius_scientific_notation :
  moon_radius = 1.738 * 10^6 :=
by
  sorry

end NUMINAMATH_GPT_moon_radius_scientific_notation_l1881_188184


namespace NUMINAMATH_GPT_smallest_n_l1881_188154

theorem smallest_n (n : ℕ) (h1 : n ≡ 1 [MOD 3]) (h2 : n ≡ 4 [MOD 5]) (h3 : n > 20) : n = 34 := 
sorry

end NUMINAMATH_GPT_smallest_n_l1881_188154


namespace NUMINAMATH_GPT_probability_of_non_perimeter_square_l1881_188166

-- Defining the total number of squares on a 10x10 board
def total_squares : ℕ := 10 * 10

-- Defining the number of perimeter squares
def perimeter_squares : ℕ := 10 + 10 + (10 - 2) * 2

-- Defining the number of non-perimeter squares
def non_perimeter_squares : ℕ := total_squares - perimeter_squares

-- Defining the probability of selecting a non-perimeter square
def probability_non_perimeter : ℚ := non_perimeter_squares / total_squares

-- The main theorem statement to be proved
theorem probability_of_non_perimeter_square:
  probability_non_perimeter = 16 / 25 := 
sorry

end NUMINAMATH_GPT_probability_of_non_perimeter_square_l1881_188166


namespace NUMINAMATH_GPT_minimum_value_l1881_188107

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem minimum_value :
  ∃ x₀ : ℝ, (∀ x : ℝ, f x₀ ≤ f x) ∧ f x₀ = -2 := by
  sorry

end NUMINAMATH_GPT_minimum_value_l1881_188107


namespace NUMINAMATH_GPT_probability_prime_sum_is_correct_l1881_188133

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def cube_rolls_prob_prime_sum : ℚ :=
  let possible_outcomes := 36
  let prime_sums_count := 15
  prime_sums_count / possible_outcomes

theorem probability_prime_sum_is_correct :
  cube_rolls_prob_prime_sum = 5 / 12 :=
by
  -- The problem statement verifies that we have to show the calculation is correct
  sorry

end NUMINAMATH_GPT_probability_prime_sum_is_correct_l1881_188133


namespace NUMINAMATH_GPT_compute_expression_l1881_188136

theorem compute_expression : 10 * (3 / 27) * 36 = 40 := 
by 
  sorry

end NUMINAMATH_GPT_compute_expression_l1881_188136


namespace NUMINAMATH_GPT_differential_equation_approx_solution_l1881_188149

open Real

noncomputable def approximate_solution (x : ℝ) : ℝ := 0.1 * exp (x ^ 2 / 2)

theorem differential_equation_approx_solution :
  ∀ (x : ℝ), -1/2 ≤ x ∧ x ≤ 1/2 →
  ∀ (y : ℝ), -1/2 ≤ y ∧ y ≤ 1/2 →
  abs (approximate_solution x - y) < 1 / 650 :=
sorry

end NUMINAMATH_GPT_differential_equation_approx_solution_l1881_188149
