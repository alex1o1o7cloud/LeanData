import Mathlib

namespace NUMINAMATH_GPT_find_AX_bisect_ACB_l183_18354

theorem find_AX_bisect_ACB (AC BX BC : ℝ) (h₁ : AC = 21) (h₂ : BX = 28) (h₃ : BC = 30) :
  ∃ (AX : ℝ), AX = 98 / 5 :=
by
  existsi 98 / 5
  sorry

end NUMINAMATH_GPT_find_AX_bisect_ACB_l183_18354


namespace NUMINAMATH_GPT_ken_climbing_pace_l183_18366

noncomputable def sari_pace : ℝ := 350 -- Sari's pace in meters per hour, derived from 700 meters in 2 hours.

def ken_pace : ℝ := 500 -- We will need to prove this.

theorem ken_climbing_pace :
  let start_time_sari := 5
  let start_time_ken := 7
  let end_time_ken := 12
  let time_ken_climbs := end_time_ken - start_time_ken
  let sari_initial_headstart := 700 -- meters
  let sari_behind_ken := 50 -- meters
  let sari_total_climb := sari_pace * time_ken_climbs
  let total_distance_ken := sari_total_climb + sari_initial_headstart + sari_behind_ken
  ken_pace = total_distance_ken / time_ken_climbs :=
by
  sorry

end NUMINAMATH_GPT_ken_climbing_pace_l183_18366


namespace NUMINAMATH_GPT_smallest_pos_multiple_6_15_is_30_l183_18359

theorem smallest_pos_multiple_6_15_is_30 :
  ∃ b > 0, 6 ∣ b ∧ 15 ∣ b ∧ (∀ b', b' > 0 ∧ b' < b → ¬ (6 ∣ b' ∧ 15 ∣ b')) :=
by
  -- Implementation to be done
  sorry

end NUMINAMATH_GPT_smallest_pos_multiple_6_15_is_30_l183_18359


namespace NUMINAMATH_GPT_max_individual_score_l183_18350

open Nat

theorem max_individual_score (n : ℕ) (total_points : ℕ) (minimum_points : ℕ) (H1 : n = 12) (H2 : total_points = 100) (H3 : ∀ i : Fin n, 7 ≤ minimum_points) :
  ∃ max_points : ℕ, max_points = 23 :=
by 
  sorry

end NUMINAMATH_GPT_max_individual_score_l183_18350


namespace NUMINAMATH_GPT_volume_frustum_correct_l183_18316

noncomputable def volume_of_frustum 
  (base_edge_orig : ℝ) 
  (altitude_orig : ℝ) 
  (base_edge_small : ℝ) 
  (altitude_small : ℝ) : ℝ :=
  let volume_ratio := (base_edge_small / base_edge_orig) ^ 3
  let base_area_orig := (Real.sqrt 3 / 4) * base_edge_orig ^ 2
  let volume_orig := (1 / 3) * base_area_orig * altitude_orig
  let volume_small := volume_ratio * volume_orig
  let volume_frustum := volume_orig - volume_small
  volume_frustum

theorem volume_frustum_correct :
  volume_of_frustum 18 9 9 3 = 212.625 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_volume_frustum_correct_l183_18316


namespace NUMINAMATH_GPT_part1_part2_part3_l183_18318

-- Given conditions and definitions
def A : ℝ := 1
def B : ℝ := 3
def y1 : ℝ := sorry  -- simply a placeholder value as y1 == y2
def y2 : ℝ := y1
def y (x m n : ℝ) : ℝ := x^2 + m * x + n

-- (1) Proof of m = -4
theorem part1 (n : ℝ) (h1 : y A m n = y1) (h2 : y B m n = y2) : m = -4 := sorry

-- (2) Proof of n = 4 when the parabola intersects the x-axis at one point
theorem part2 (h : ∃ n, ∀ x : ℝ, y x (-4) n = 0 → x = (x - 2)^2) : n = 4 := sorry

-- (3) Proof of the range of real number values for a
theorem part3 (a : ℝ) (b1 b2 : ℝ) (n : ℝ) (h1 : y a (-4) n = b1) 
  (h2 : y B (-4) n = b2) (h3 : b1 > b2) : a < 1 ∨ a > 3 := sorry

end NUMINAMATH_GPT_part1_part2_part3_l183_18318


namespace NUMINAMATH_GPT_sin_210_eq_neg_half_l183_18313

theorem sin_210_eq_neg_half : Real.sin (210 * Real.pi / 180) = -1 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_sin_210_eq_neg_half_l183_18313


namespace NUMINAMATH_GPT_radius_of_inscribed_circle_l183_18368

noncomputable def inscribed_circle_radius (AB AC BC : ℝ) : ℝ :=
  let s := (AB + AC + BC) / 2
  let K := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC))
  K / s

theorem radius_of_inscribed_circle :
  inscribed_circle_radius 6 8 10 = 2 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_inscribed_circle_l183_18368


namespace NUMINAMATH_GPT_group1_calculation_group2_calculation_l183_18310

theorem group1_calculation : 9 / 3 * (9 - 1) = 24 := by
  sorry

theorem group2_calculation : 7 * (3 + 3 / 7) = 24 := by
  sorry

end NUMINAMATH_GPT_group1_calculation_group2_calculation_l183_18310


namespace NUMINAMATH_GPT_quadratic_trinomials_unique_root_value_l183_18390

theorem quadratic_trinomials_unique_root_value (p q : ℝ) :
  ∀ x, (x^2 + p * x + q) + (x^2 + q * x + p) = (2 * x^2 + (p + q) * x + (p + q)) →
  (((p + q = 0 ∨ p + q = 8) → (2 * (2:ℝ)^2 + (p + q) * 2 + (p + q) = 8 ∨ 2 * (2:ℝ)^2 + (p + q) * 2 + (p + q) = 32))) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_trinomials_unique_root_value_l183_18390


namespace NUMINAMATH_GPT_profit_per_meter_is_25_l183_18363

def sell_price : ℕ := 8925
def cost_price_per_meter : ℕ := 80
def meters_sold : ℕ := 85
def total_cost_price : ℕ := cost_price_per_meter * meters_sold
def total_profit : ℕ := sell_price - total_cost_price
def profit_per_meter : ℕ := total_profit / meters_sold

theorem profit_per_meter_is_25 : profit_per_meter = 25 := by
  sorry

end NUMINAMATH_GPT_profit_per_meter_is_25_l183_18363


namespace NUMINAMATH_GPT_repeated_root_value_l183_18326

theorem repeated_root_value (m : ℝ) :
  (∃ x : ℝ, x ≠ 1 ∧ (2 / (x - 1) + 3 = m / (x - 1)) ∧ 
            ∀ y : ℝ, y ≠ 1 ∧ (2 / (y - 1) + 3 = m / (y - 1)) → y = x) →
  m = 2 :=
by
  sorry

end NUMINAMATH_GPT_repeated_root_value_l183_18326


namespace NUMINAMATH_GPT_largest_n_sum_pos_l183_18367

section
variables {a : ℕ → ℤ}
variables {d : ℤ}
variables {n : ℕ}

axiom a_seq (n : ℕ) : a n = a 1 + (n - 1) * d
axiom a1_pos : a 1 > 0
axiom a2013_2014_pos : a 2013 + a 2014 > 0
axiom a2013_2014_neg : a 2013 * a 2014 < 0

theorem largest_n_sum_pos :
  ∃ n : ℕ, (∀ k ≤ n, (k * (2 * a 1 + (k - 1) * d) / 2) > 0) → n = 4026 := sorry

end

end NUMINAMATH_GPT_largest_n_sum_pos_l183_18367


namespace NUMINAMATH_GPT_base_length_of_isosceles_triangle_l183_18352

theorem base_length_of_isosceles_triangle 
  (a b : ℕ) 
  (h1 : a = 6) 
  (h2 : b = 6) 
  (perimeter : ℕ) 
  (h3 : 2*a + b = perimeter)
  (h4 : perimeter = 20) 
  : b = 8 := 
by
  sorry

end NUMINAMATH_GPT_base_length_of_isosceles_triangle_l183_18352


namespace NUMINAMATH_GPT_statement_bug_travel_direction_l183_18311

/-
  Theorem statement: On a plane with a grid formed by regular hexagons of side length 1,
  if a bug traveled from node A to node B along the shortest path of 100 units,
  then the bug traveled exactly 50 units in one direction.
-/
theorem bug_travel_direction (side_length : ℝ) (total_distance : ℝ) 
  (hexagonal_grid : Π (x y : ℝ), Prop) (A B : ℝ × ℝ) 
  (shortest_path : ℝ) :
  side_length = 1 ∧ shortest_path = 100 →
  ∃ (directional_travel : ℝ), directional_travel = 50 :=
by
  sorry

end NUMINAMATH_GPT_statement_bug_travel_direction_l183_18311


namespace NUMINAMATH_GPT_paint_price_and_max_boxes_l183_18336

theorem paint_price_and_max_boxes (x y m : ℕ) 
  (h1 : x + 2 * y = 56) 
  (h2 : 2 * x + y = 64) 
  (h3 : 24 * m + 16 * (200 - m) ≤ 3920) : 
  x = 24 ∧ y = 16 ∧ m ≤ 90 := 
  by 
    sorry

end NUMINAMATH_GPT_paint_price_and_max_boxes_l183_18336


namespace NUMINAMATH_GPT_optimal_purchasing_plan_l183_18383

def price_carnation := 5
def price_lily := 10
def total_flowers := 300
def max_carnations (x : ℕ) : Prop := x ≤ 2 * (total_flowers - x)

theorem optimal_purchasing_plan :
  ∃ (x y : ℕ), (x + y = total_flowers) ∧ (x = 200) ∧ (y = 100) ∧ (max_carnations x) ∧ 
  ∀ (x' y' : ℕ), (x' + y' = total_flowers) → max_carnations x' →
    (price_carnation * x + price_lily * y ≤ price_carnation * x' + price_lily * y') :=
by
  sorry

end NUMINAMATH_GPT_optimal_purchasing_plan_l183_18383


namespace NUMINAMATH_GPT_max_red_balls_l183_18317

theorem max_red_balls (R B G : ℕ) (h1 : G = 12) (h2 : R + B + G = 28) (h3 : R + G < 24) : R ≤ 11 := 
by
  sorry

end NUMINAMATH_GPT_max_red_balls_l183_18317


namespace NUMINAMATH_GPT_assign_students_to_villages_l183_18323

theorem assign_students_to_villages (n m : ℕ) (hn : n = 5) (hm : m = 3) :
  ∃ N : ℕ, N = 70 ∧ 
  (∃ (f : Fin n → Fin m), (∀ i j, f i = f j ↔ i = j) ∧ 
  (∀ x : Fin m, ∃ y : Fin n, f y = x)) :=
by
  sorry

end NUMINAMATH_GPT_assign_students_to_villages_l183_18323


namespace NUMINAMATH_GPT_algebraic_expression_value_l183_18332

theorem algebraic_expression_value (x : ℝ) (h : x = 4 * Real.sin (Real.pi / 4) - 2) :
  (1 / (x - 1) / (x + 2) / (x ^ 2 - 2 * x + 1) - x / (x + 2)) = - (Real.sqrt 2 / 4) :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l183_18332


namespace NUMINAMATH_GPT_intersecting_lines_solution_l183_18395

theorem intersecting_lines_solution (a b : ℝ) :
  (∃ (a b : ℝ), 
    ((a^2 + 1) * 2 - 2 * b * (-3) = 4) ∧ 
    ((1 - a) * 2 + b * (-3) = 9)) →
  (a, b) = (4, -5) ∨ (a, b) = (-2, -1) :=
by
  sorry

end NUMINAMATH_GPT_intersecting_lines_solution_l183_18395


namespace NUMINAMATH_GPT_rhombus_perimeter_l183_18357

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 16) : 
  ∃ p : ℝ, p = 16 * Real.sqrt 13 := by
sorry

end NUMINAMATH_GPT_rhombus_perimeter_l183_18357


namespace NUMINAMATH_GPT_samantha_sleep_hours_l183_18381

def time_in_hours (hours minutes : ℕ) : ℕ :=
  hours + (minutes / 60)

def hours_slept (bed_time wake_up_time : ℕ) : ℕ :=
  if bed_time < wake_up_time then wake_up_time - bed_time + 12 else 24 - bed_time + wake_up_time

theorem samantha_sleep_hours : hours_slept 7 11 = 16 := by
  sorry

end NUMINAMATH_GPT_samantha_sleep_hours_l183_18381


namespace NUMINAMATH_GPT_students_more_than_guinea_pigs_l183_18392

-- Definitions based on the problem's conditions
def students_per_classroom : Nat := 22
def guinea_pigs_per_classroom : Nat := 3
def classrooms : Nat := 5

-- The proof statement
theorem students_more_than_guinea_pigs :
  (students_per_classroom * classrooms) - (guinea_pigs_per_classroom * classrooms) = 95 :=
by
  sorry

end NUMINAMATH_GPT_students_more_than_guinea_pigs_l183_18392


namespace NUMINAMATH_GPT_meaningful_fraction_l183_18307

theorem meaningful_fraction (x : ℝ) : (∃ y : ℝ, y = 5 / (x - 3)) ↔ x ≠ 3 :=
by
  sorry

end NUMINAMATH_GPT_meaningful_fraction_l183_18307


namespace NUMINAMATH_GPT_sqrt_21_between_4_and_5_l183_18362

theorem sqrt_21_between_4_and_5 : 4 < Real.sqrt 21 ∧ Real.sqrt 21 < 5 := 
by 
  sorry

end NUMINAMATH_GPT_sqrt_21_between_4_and_5_l183_18362


namespace NUMINAMATH_GPT_revenue_from_full_price_tickets_l183_18335

-- Let's define our variables and assumptions
variables (f h p: ℕ)

-- Total number of tickets sold
def total_tickets (f h: ℕ) : Prop := f + h = 200

-- Total revenue from tickets
def total_revenue (f h p: ℕ) : Prop := f * p + h * (p / 3) = 2500

-- Statement to prove the revenue from full-price tickets
theorem revenue_from_full_price_tickets (f h p: ℕ) (hf: total_tickets f h) 
  (hr: total_revenue f h p): f * p = 1250 :=
sorry

end NUMINAMATH_GPT_revenue_from_full_price_tickets_l183_18335


namespace NUMINAMATH_GPT_pow_mod_remainder_l183_18328

theorem pow_mod_remainder :
  (3 ^ 2023) % 5 = 2 :=
by sorry

end NUMINAMATH_GPT_pow_mod_remainder_l183_18328


namespace NUMINAMATH_GPT_determine_m_l183_18349

-- Define a complex number structure in Lean
structure ComplexNumber where
  re : ℝ  -- real part
  im : ℝ  -- imaginary part

-- Define the condition where the complex number is purely imaginary
def is_purely_imaginary (z : ComplexNumber) : Prop :=
  z.re = 0

-- State the Lean theorem
theorem determine_m (m : ℝ) (h : is_purely_imaginary (ComplexNumber.mk (m^2 - m) m)) : m = 1 :=
by
  sorry

end NUMINAMATH_GPT_determine_m_l183_18349


namespace NUMINAMATH_GPT_Hillary_activities_LCM_l183_18308

theorem Hillary_activities_LCM :
  let swim := 6
  let run := 4
  let cycle := 16
  Nat.lcm (Nat.lcm swim run) cycle = 48 :=
by
  sorry

end NUMINAMATH_GPT_Hillary_activities_LCM_l183_18308


namespace NUMINAMATH_GPT_math_problem_l183_18331

noncomputable def f (x a : ℝ) : ℝ := -4 * (Real.cos x) ^ 2 + 4 * Real.sqrt 3 * a * (Real.sin x) * (Real.cos x) + 2

theorem math_problem (a : ℝ) :
  (∃ a, ∀ x, f x a = f (π/6 - x) a) →    -- Symmetry condition
  (a = 1 ∧
  ∀ k : ℤ, ∀ x, (x ∈ Set.Icc (π/3 + k * π) (5 * π / 6 + k * π) → 
    x ∈ Set.Icc (π/3 + k * π) (5 * π / 6 + k * π)) ∧  -- Decreasing intervals
  (∀ x, 2 * x - π / 6 ∈ Set.Icc (-2 * π / 3) (π / 6) → 
    f x a ∈ Set.Icc (-4 : ℝ) 2)) := -- Range on given interval
sorry

end NUMINAMATH_GPT_math_problem_l183_18331


namespace NUMINAMATH_GPT_base_conversion_problem_l183_18384

theorem base_conversion_problem :
  ∃ A B : ℕ, 0 ≤ A ∧ A < 8 ∧ 0 ≤ B ∧ B < 6 ∧
           8 * A + B = 6 * B + A ∧
           8 * A + B = 45 :=
by
  sorry

end NUMINAMATH_GPT_base_conversion_problem_l183_18384


namespace NUMINAMATH_GPT_virginia_more_than_adrienne_l183_18372

def teaching_years (V A D : ℕ) : Prop :=
  V + A + D = 102 ∧ D = 43 ∧ V = D - 9

theorem virginia_more_than_adrienne (V A : ℕ) (h : teaching_years V A 43) : V - A = 9 :=
by
  sorry

end NUMINAMATH_GPT_virginia_more_than_adrienne_l183_18372


namespace NUMINAMATH_GPT_train_cross_platform_time_l183_18315

def train_length : ℝ := 300
def platform_length : ℝ := 550
def signal_pole_time : ℝ := 18

theorem train_cross_platform_time :
  let speed : ℝ := train_length / signal_pole_time
  let total_distance : ℝ := train_length + platform_length
  let crossing_time : ℝ := total_distance / speed
  crossing_time = 51 :=
by
  sorry

end NUMINAMATH_GPT_train_cross_platform_time_l183_18315


namespace NUMINAMATH_GPT_parabola_behavior_l183_18305

theorem parabola_behavior (x : ℝ) (h : x < 0) : ∃ y, y = 2*x^2 - 1 ∧ ∀ x1 x2 : ℝ, x1 < x2 ∧ x1 < 0 ∧ x2 < 0 → (2*x1^2 - 1) > (2*x2^2 - 1) :=
by
  sorry

end NUMINAMATH_GPT_parabola_behavior_l183_18305


namespace NUMINAMATH_GPT_circle_line_intersection_l183_18347

theorem circle_line_intersection (a : ℝ) : 
  (∃ x y : ℝ, (x - y + 1 = 0) ∧ ((x - a)^2 + y^2 = 2)) ↔ -3 ≤ a ∧ a ≤ 1 := 
by
  sorry

end NUMINAMATH_GPT_circle_line_intersection_l183_18347


namespace NUMINAMATH_GPT_find_y_l183_18322

theorem find_y (AB BC : ℕ) (y x : ℕ) 
  (h1 : AB = 3 * y)
  (h2 : BC = 2 * x)
  (h3 : AB * BC = 2400) 
  (h4 : AB * BC = 6 * x * y) :
  y = 20 := by
  sorry

end NUMINAMATH_GPT_find_y_l183_18322


namespace NUMINAMATH_GPT_Deepak_age_l183_18356

theorem Deepak_age (A D : ℕ) (h1 : A / D = 4 / 3) (h2 : A + 6 = 26) : D = 15 :=
by
  sorry

end NUMINAMATH_GPT_Deepak_age_l183_18356


namespace NUMINAMATH_GPT_trig_identity_product_l183_18325

theorem trig_identity_product :
  (1 + Real.cos (Real.pi / 12)) * (1 + Real.cos (5 * Real.pi / 12)) * 
  (1 + Real.cos (7 * Real.pi / 12)) * (1 + Real.cos (11 * Real.pi / 12)) = 1 / 16 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_product_l183_18325


namespace NUMINAMATH_GPT_julian_comic_pages_l183_18348

-- Definitions from conditions
def frames_per_page : ℝ := 143.0
def total_frames : ℝ := 1573.0

-- The theorem stating the proof problem
theorem julian_comic_pages : total_frames / frames_per_page = 11 :=
by
  sorry

end NUMINAMATH_GPT_julian_comic_pages_l183_18348


namespace NUMINAMATH_GPT_remainder_1125_1127_1129_div_12_l183_18370

theorem remainder_1125_1127_1129_div_12 :
  (1125 * 1127 * 1129) % 12 = 3 :=
by
  -- Proof can be written here
  sorry

end NUMINAMATH_GPT_remainder_1125_1127_1129_div_12_l183_18370


namespace NUMINAMATH_GPT_min_value_of_function_l183_18364

theorem min_value_of_function (x : ℝ) (h : x > -1) : 
  (∀ x₀ : ℝ, x₀ > -1 → (x₀ + 1 + 1 / (x₀ + 1) - 1) ≥ 1) ∧ (x = 0) :=
sorry

end NUMINAMATH_GPT_min_value_of_function_l183_18364


namespace NUMINAMATH_GPT_enter_exit_ways_eq_sixteen_l183_18376

theorem enter_exit_ways_eq_sixteen (n : ℕ) (h : n = 4) : n * n = 16 :=
by sorry

end NUMINAMATH_GPT_enter_exit_ways_eq_sixteen_l183_18376


namespace NUMINAMATH_GPT_geometric_sum_n_eq_3_l183_18302

theorem geometric_sum_n_eq_3 :
  (∃ n : ℕ, (1 / 2) * (1 - (1 / 3) ^ n) = 728 / 2187) ↔ n = 3 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sum_n_eq_3_l183_18302


namespace NUMINAMATH_GPT_sin_tan_correct_value_l183_18371

noncomputable def sin_tan_value (x y : ℝ) (h : x^2 + y^2 = 1) : ℝ :=
  let sin_alpha := y
  let tan_alpha := y / x
  sin_alpha * tan_alpha

theorem sin_tan_correct_value :
  sin_tan_value (3/5) (-4/5) (by norm_num) = 16/15 := 
by
  sorry

end NUMINAMATH_GPT_sin_tan_correct_value_l183_18371


namespace NUMINAMATH_GPT_simplify_expression_l183_18373

theorem simplify_expression (z : ℝ) : (5 - 2*z^2) - (4*z^2 - 7) = 12 - 6*z^2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l183_18373


namespace NUMINAMATH_GPT_cos_diff_l183_18391

theorem cos_diff (x y : ℝ) (hx : x = Real.cos (20 * Real.pi / 180)) (hy : y = Real.cos (40 * Real.pi / 180)) :
  x - y = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_cos_diff_l183_18391


namespace NUMINAMATH_GPT_cherries_initially_l183_18393

theorem cherries_initially (x : ℕ) (h₁ : x - 6 = 10) : x = 16 :=
by
  sorry

end NUMINAMATH_GPT_cherries_initially_l183_18393


namespace NUMINAMATH_GPT_focus_coordinates_of_hyperbola_l183_18396

theorem focus_coordinates_of_hyperbola (x y : ℝ) :
  (∃ c : ℝ, (c = 5 ∧ y = 10) ∧ (c = 5 + Real.sqrt 97)) ↔ 
  (x, y) = (5 + Real.sqrt 97, 10) :=
by
  sorry

end NUMINAMATH_GPT_focus_coordinates_of_hyperbola_l183_18396


namespace NUMINAMATH_GPT_true_compound_proposition_l183_18327

-- Define conditions and propositions in Lean
def proposition_p : Prop := ∃ (x : ℝ), x^2 + x + 1 < 0
def proposition_q : Prop := ∀ (x : ℝ), 1 ≤ x → x ≤ 2 → x^2 - 1 ≥ 0

-- Define the compound proposition
def correct_proposition : Prop := ¬ proposition_p ∧ proposition_q

-- Prove the correct compound proposition
theorem true_compound_proposition : correct_proposition :=
by
  sorry

end NUMINAMATH_GPT_true_compound_proposition_l183_18327


namespace NUMINAMATH_GPT_h_inverse_left_h_inverse_right_l183_18387

noncomputable def f (x : ℝ) : ℝ := 4 * x + 5
noncomputable def g (x : ℝ) : ℝ := x^2 - 2 * x - 1
noncomputable def h (x : ℝ) : ℝ := f (g x)
noncomputable def h_inv (y : ℝ) : ℝ := 1 + (Real.sqrt (3 * y + 12)) / 4 -- Correct answer

-- Theorem statements to prove the inverse relationship
theorem h_inverse_left (x : ℝ) : h (h_inv x) = x :=
by
  sorry -- Proof of the left inverse

theorem h_inverse_right (y : ℝ) : h_inv (h y) = y :=
by
  sorry -- Proof of the right inverse

end NUMINAMATH_GPT_h_inverse_left_h_inverse_right_l183_18387


namespace NUMINAMATH_GPT_closest_ratio_adults_children_l183_18334

theorem closest_ratio_adults_children (a c : ℕ) 
  (h1 : 30 * a + 15 * c = 2250) 
  (h2 : a ≥ 50) 
  (h3 : c ≥ 20) : a = 50 ∧ c = 50 :=
by {
  sorry
}

end NUMINAMATH_GPT_closest_ratio_adults_children_l183_18334


namespace NUMINAMATH_GPT_students_per_class_l183_18340

variable (c : ℕ) (s : ℕ)

def books_per_month := 6
def months_per_year := 12
def books_per_year := books_per_month * months_per_year
def total_books_read := 72

theorem students_per_class : (s * c = 1 ∧ s * books_per_year = total_books_read) → s = 1 := by
  intros h
  have h1: books_per_year = total_books_read := by
    calc
      books_per_year = books_per_month * months_per_year := rfl
      _ = 6 * 12 := rfl
      _ = 72 := rfl
  sorry

end NUMINAMATH_GPT_students_per_class_l183_18340


namespace NUMINAMATH_GPT_planes_formed_through_three_lines_l183_18378

theorem planes_formed_through_three_lines (L1 L2 L3 : ℝ × ℝ × ℝ → Prop) (P : ℝ × ℝ × ℝ) :
  (∀ (x : ℝ × ℝ × ℝ), L1 x → L2 x → L3 x → x = P) →
  (∃ n : ℕ, n = 1 ∨ n = 3) :=
sorry

end NUMINAMATH_GPT_planes_formed_through_three_lines_l183_18378


namespace NUMINAMATH_GPT_find_third_side_l183_18380

def vol_of_cube (side : ℝ) : ℝ := side ^ 3

def vol_of_box (length width height : ℝ) : ℝ := length * width * height

theorem find_third_side (n : ℝ) (vol_cube : ℝ) (num_cubes : ℝ) (l w : ℝ) (vol_box : ℝ) :
  num_cubes = 24 →
  vol_cube = 27 →
  l = 8 →
  w = 12 →
  vol_box = num_cubes * vol_cube →
  vol_box = vol_of_box l w n →
  n = 6.75 :=
by
  intros hcubes hc_vol hl hw hvbox1 hvbox2
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_find_third_side_l183_18380


namespace NUMINAMATH_GPT_algebraic_expression_value_l183_18338

theorem algebraic_expression_value (x : ℝ) (h : x^2 + x - 3 = 0) :
  (x - 1)^2 - x * (x - 3) + (x + 1) * (x - 1) = 3 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l183_18338


namespace NUMINAMATH_GPT_highest_numbered_street_l183_18351

theorem highest_numbered_street (L : ℕ) (d : ℕ) (H : L = 15000 ∧ d = 500) : 
    (L / d) - 2 = 28 :=
by
  sorry

end NUMINAMATH_GPT_highest_numbered_street_l183_18351


namespace NUMINAMATH_GPT_find_a_l183_18300

noncomputable def f (x : ℝ) : ℝ := 3 * ((x - 1) / 2) - 2

theorem find_a (x a : ℝ) (hx : f a = 4) (ha : a = 2 * x + 1) : a = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l183_18300


namespace NUMINAMATH_GPT_sum_interior_numbers_eight_l183_18394

noncomputable def sum_interior_numbers (n : ℕ) : ℕ :=
  2^(n-1) - 2 -- This is a general formula derived from the pattern

theorem sum_interior_numbers_eight :
  sum_interior_numbers 8 = 126 :=
by
  -- No proof required, so we use sorry.
  sorry

end NUMINAMATH_GPT_sum_interior_numbers_eight_l183_18394


namespace NUMINAMATH_GPT_not_divisible_by_2006_l183_18353

theorem not_divisible_by_2006 (k : ℤ) : ¬ ∃ m : ℤ, k^2 + k + 1 = 2006 * m :=
sorry

end NUMINAMATH_GPT_not_divisible_by_2006_l183_18353


namespace NUMINAMATH_GPT_union_sets_l183_18321

open Set

def setM : Set ℝ := {x : ℝ | x^2 < x}
def setN : Set ℝ := {x : ℝ | x^2 + 2*x - 3 < 0}

theorem union_sets : setM ∪ setN = {x : ℝ | -3 < x ∧ x < 1} :=
by
  sorry

end NUMINAMATH_GPT_union_sets_l183_18321


namespace NUMINAMATH_GPT_paving_stones_needed_l183_18312

variables (length_courtyard width_courtyard num_paving_stones length_paving_stone area_courtyard area_paving_stone : ℝ)
noncomputable def width_paving_stone := 2

theorem paving_stones_needed : 
  length_courtyard = 60 → 
  width_courtyard = 14 → 
  num_paving_stones = 140 →
  length_paving_stone = 3 →
  area_courtyard = length_courtyard * width_courtyard →
  area_paving_stone = length_paving_stone * width_paving_stone →
  num_paving_stones = area_courtyard / area_paving_stone :=
by
  intros h_length_courtyard h_width_courtyard h_num_paving_stones h_length_paving_stone h_area_courtyard h_area_paving_stone
  rw [h_length_courtyard, h_width_courtyard, h_length_paving_stone] at *
  simp at *
  sorry

end NUMINAMATH_GPT_paving_stones_needed_l183_18312


namespace NUMINAMATH_GPT_total_bales_stored_l183_18309

theorem total_bales_stored 
  (initial_bales : ℕ := 540) 
  (new_bales : ℕ := 2) : 
  initial_bales + new_bales = 542 :=
by
  sorry

end NUMINAMATH_GPT_total_bales_stored_l183_18309


namespace NUMINAMATH_GPT_chord_length_ne_l183_18379

-- Define the ellipse
def ellipse (x y : ℝ) := (x^2 / 8) + (y^2 / 4) = 1

-- Define the first line
def line_l (k x : ℝ) := (k * x + 1)

-- Define the second line
def line_l_option_D (k x y : ℝ) := (k * x + y - 2)

-- Prove the chord length inequality for line_l_option_D
theorem chord_length_ne (k : ℝ) :
  ∀ x y : ℝ, ellipse x y →
  ∃ x1 x2 y1 y2 : ℝ, ellipse x1 y1 ∧ line_l k x1 = y1 ∧ ellipse x2 y2 ∧ line_l k x2 = y2 ∧
  ∀ x3 x4 y3 y4 : ℝ, ellipse x3 y3 ∧ line_l_option_D k x3 y3 = 0 ∧ ellipse x4 y4 ∧ line_l_option_D k x4 y4 = 0 →
  dist (x1, y1) (x2, y2) ≠ dist (x3, y3) (x4, y4) :=
sorry

end NUMINAMATH_GPT_chord_length_ne_l183_18379


namespace NUMINAMATH_GPT_coordinates_of_P_l183_18386

theorem coordinates_of_P (A B : ℝ × ℝ × ℝ) (m : ℝ) :
  A = (1, 0, 2) ∧ B = (1, -3, 1) ∧ (0, 0, m) = (0, 0, -3) :=
by 
  sorry

end NUMINAMATH_GPT_coordinates_of_P_l183_18386


namespace NUMINAMATH_GPT_contributions_before_john_l183_18358

theorem contributions_before_john (n : ℕ) (A : ℚ) 
  (h1 : 1.5 * A = 75)
  (h2 : (n * A + 225) / (n + 1) = 75) : n = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_contributions_before_john_l183_18358


namespace NUMINAMATH_GPT_find_fourth_score_l183_18346

theorem find_fourth_score
  (a b c : ℕ) (d : ℕ)
  (ha : a = 70) (hb : b = 80) (hc : c = 90)
  (average_eq : (a + b + c + d) / 4 = 70) :
  d = 40 := 
sorry

end NUMINAMATH_GPT_find_fourth_score_l183_18346


namespace NUMINAMATH_GPT_fifth_derivative_l183_18345

noncomputable def f (x : ℝ) : ℝ := (2 * x ^ 2 - 7) * Real.log (x - 1)

theorem fifth_derivative :
  ∀ x, (deriv^[5] f) x = 8 * (x ^ 2 - 5 * x - 11) / ((x - 1) ^ 5) :=
by
  sorry

end NUMINAMATH_GPT_fifth_derivative_l183_18345


namespace NUMINAMATH_GPT_find_function_ex_l183_18389

theorem find_function_ex (f : ℝ → ℝ) (a : ℝ) :
  (∀ x y : ℝ, f (f x + y) = 2 * x + f (f y - x)) →
  (∀ x : ℝ, f x = x - a) :=
by
  intros h x
  sorry

end NUMINAMATH_GPT_find_function_ex_l183_18389


namespace NUMINAMATH_GPT_total_renovation_cost_eq_l183_18333

-- Define the conditions
def hourly_rate_1 := 15
def hourly_rate_2 := 20
def hourly_rate_3 := 18
def hourly_rate_4 := 22
def hours_per_day := 8
def days := 10
def meal_cost_per_professional_per_day := 10
def material_cost := 2500
def plumbing_issue_cost := 750
def electrical_issue_cost := 500
def faulty_appliance_cost := 400

-- Define the calculated values based on the conditions
def daily_labor_cost_condition := 
  hourly_rate_1 * hours_per_day + 
  hourly_rate_2 * hours_per_day + 
  hourly_rate_3 * hours_per_day + 
  hourly_rate_4 * hours_per_day
def total_labor_cost := daily_labor_cost_condition * days

def daily_meal_cost := meal_cost_per_professional_per_day * 4
def total_meal_cost := daily_meal_cost * days

def unexpected_repair_costs := plumbing_issue_cost + electrical_issue_cost + faulty_appliance_cost

def total_cost := total_labor_cost + total_meal_cost + material_cost + unexpected_repair_costs

-- The theorem to prove that the total cost of the renovation is $10,550
theorem total_renovation_cost_eq : total_cost = 10550 := by
  sorry

end NUMINAMATH_GPT_total_renovation_cost_eq_l183_18333


namespace NUMINAMATH_GPT_total_jumps_is_400_l183_18397

-- Define the variables according to the conditions 
def Ronald_jumps := 157
def Rupert_jumps := Ronald_jumps + 86

-- Prove the total jumps
theorem total_jumps_is_400 : Ronald_jumps + Rupert_jumps = 400 := by
  sorry

end NUMINAMATH_GPT_total_jumps_is_400_l183_18397


namespace NUMINAMATH_GPT_find_a₁_l183_18343

noncomputable def S_3 (a₁ q : ℝ) : ℝ := a₁ + a₁ * q + a₁ * q^2

theorem find_a₁ (S₃_eq : S_3 a₁ q = a₁ + 3 * (a₁ * q)) (a₄_eq : a₁ * q^3 = 8) : a₁ = 1 :=
by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_find_a₁_l183_18343


namespace NUMINAMATH_GPT_sqrt_value_l183_18314

theorem sqrt_value (h : Real.sqrt 100.4004 = 10.02) : Real.sqrt 1.004004 = 1.002 := 
by
  sorry

end NUMINAMATH_GPT_sqrt_value_l183_18314


namespace NUMINAMATH_GPT_sqrt_product_simplification_l183_18365

variable (q : ℝ)

theorem sqrt_product_simplification (hq : q ≥ 0) : 
  Real.sqrt (15 * q) * Real.sqrt (8 * q^3) * Real.sqrt (12 * q^5) = 12 * q^4 * Real.sqrt (10 * q) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_product_simplification_l183_18365


namespace NUMINAMATH_GPT_exists_infinitely_many_n_l183_18361

def sum_of_digits (m : ℕ) : ℕ := 
  m.digits 10 |>.sum

theorem exists_infinitely_many_n (S : ℕ → ℕ) (h_sum_of_digits : ∀ m, S m = sum_of_digits m) :
  ∀ N : ℕ, ∃ n ≥ N, S (3 ^ n) ≥ S (3 ^ (n + 1)) :=
by { sorry }

end NUMINAMATH_GPT_exists_infinitely_many_n_l183_18361


namespace NUMINAMATH_GPT_student_distribution_l183_18398

theorem student_distribution (a b : ℕ) (h1 : a + b = 81) (h2 : a = b - 9) : a = 36 ∧ b = 45 := 
by
  sorry

end NUMINAMATH_GPT_student_distribution_l183_18398


namespace NUMINAMATH_GPT_total_cost_is_eight_times_l183_18355

theorem total_cost_is_eight_times (cost_shorts cost_shirt cost_boots cost_shin_guards : ℕ) :
  (cost_shorts + cost_shirt = 2 * cost_shorts) →
  (cost_shorts + cost_boots = 5 * cost_shorts) →
  (cost_shorts + cost_shin_guards = 3 * cost_shorts) →
  cost_shorts + cost_shirt + cost_boots + cost_shin_guards = 8 * cost_shorts := 
by 
  intros h_shirt h_boots h_shin_guards 
  sorry

end NUMINAMATH_GPT_total_cost_is_eight_times_l183_18355


namespace NUMINAMATH_GPT_find_pqr_eq_1680_l183_18337

theorem find_pqr_eq_1680
  {p q r : ℤ} (hpqz : p ≠ 0) (hqqz : q ≠ 0) (hrqz : r ≠ 0)
  (h_sum : p + q + r = 30)
  (h_cond : (1:ℚ) / p + (1:ℚ) / q + (1:ℚ) / r + 390 / (p * q * r) = 1) :
  p * q * r = 1680 :=
sorry

end NUMINAMATH_GPT_find_pqr_eq_1680_l183_18337


namespace NUMINAMATH_GPT_tax_difference_is_250000_l183_18360

noncomputable def old_tax_rate : ℝ := 0.20
noncomputable def new_tax_rate : ℝ := 0.30
noncomputable def old_income : ℝ := 1000000
noncomputable def new_income : ℝ := 1500000
noncomputable def old_taxes_paid := old_tax_rate * old_income
noncomputable def new_taxes_paid := new_tax_rate * new_income
noncomputable def tax_difference := new_taxes_paid - old_taxes_paid

theorem tax_difference_is_250000 : tax_difference = 250000 := by
  sorry

end NUMINAMATH_GPT_tax_difference_is_250000_l183_18360


namespace NUMINAMATH_GPT_int_valued_fractions_l183_18303

theorem int_valued_fractions (a : ℤ) :
  ∃ k : ℤ, (a^2 - 21 * a + 17) = k * a ↔ a = 1 ∨ a = -1 ∨ a = 17 ∨ a = -17 :=
by {
  sorry
}

end NUMINAMATH_GPT_int_valued_fractions_l183_18303


namespace NUMINAMATH_GPT_parabola_intersections_l183_18382

open Real

-- Definition of the two parabolas
def parabola1 (x : ℝ) : ℝ := 3*x^2 - 6*x + 2
def parabola2 (x : ℝ) : ℝ := 9*x^2 - 4*x - 5

-- Theorem stating the intersections are (-7/3, 9) and (0.5, -0.25)
theorem parabola_intersections : 
  {p : ℝ × ℝ | parabola1 p.1 = p.2 ∧ parabola2 p.1 = p.2} =
  {(-7/3, 9), (0.5, -0.25)} :=
by 
  sorry

end NUMINAMATH_GPT_parabola_intersections_l183_18382


namespace NUMINAMATH_GPT_triangle_area_ratio_l183_18330

theorem triangle_area_ratio
  (a b c : ℕ) (S_triangle : ℕ) -- assuming S_triangle represents the area of the original triangle
  (S_bisected_triangle : ℕ) -- assuming S_bisected_triangle represents the area of the bisected triangle
  (is_angle_bisector : ∀ x y z : ℕ, ∃ k, k = (2 * a * b * c * x) / ((a + b) * (a + c) * (b + c))) :
  S_bisected_triangle = (2 * a * b * c) / ((a + b) * (a + c) * (b + c)) * S_triangle :=
sorry

end NUMINAMATH_GPT_triangle_area_ratio_l183_18330


namespace NUMINAMATH_GPT_remainder_14_plus_x_mod_31_l183_18324

theorem remainder_14_plus_x_mod_31 (x : ℕ) (hx : 7 * x ≡ 1 [MOD 31]) : (14 + x) % 31 = 23 := 
sorry

end NUMINAMATH_GPT_remainder_14_plus_x_mod_31_l183_18324


namespace NUMINAMATH_GPT_height_at_2_years_l183_18388

variable (height : ℕ → ℕ) -- height function representing the height of the tree at the end of n years
variable (triples_height : ∀ n, height (n + 1) = 3 * height n) -- tree triples its height every year
variable (height_4 : height 4 = 81) -- height at the end of 4 years is 81 feet

-- We need the height at the end of 2 years
theorem height_at_2_years : height 2 = 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_height_at_2_years_l183_18388


namespace NUMINAMATH_GPT_sum_digits_B_of_4444_4444_l183_18304

noncomputable def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum

theorem sum_digits_B_of_4444_4444 :
  let A : ℕ := sum_digits (4444 ^ 4444)
  let B : ℕ := sum_digits A
  sum_digits B = 7 :=
by
  sorry

end NUMINAMATH_GPT_sum_digits_B_of_4444_4444_l183_18304


namespace NUMINAMATH_GPT_total_cards_given_away_l183_18320

-- Define the conditions in Lean
def Jim_initial_cards : ℕ := 365
def sets_given_to_brother : ℕ := 8
def sets_given_to_sister : ℕ := 5
def sets_given_to_friend : ℕ := 2
def cards_per_set : ℕ := 13

-- Define a theorem to prove the total number of cards given away
theorem total_cards_given_away : 
  sets_given_to_brother + sets_given_to_sister + sets_given_to_friend = 15 ∧
  15 * cards_per_set = 195 := 
by
  sorry

end NUMINAMATH_GPT_total_cards_given_away_l183_18320


namespace NUMINAMATH_GPT_area_of_triangle_formed_by_tangent_line_l183_18341

noncomputable def curve (x : ℝ) : ℝ := Real.log x - 2 * x

noncomputable def slope_of_tangent_at (x : ℝ) : ℝ := (1 / x) - 2

def point_of_tangency : ℝ × ℝ := (1, -2)

-- Define the tangent line equation at the point (1, -2)
noncomputable def tangent_line (x : ℝ) : ℝ := -x - 1

-- Define x and y intercepts of the tangent line
def x_intercept_of_tangent : ℝ := -1
def y_intercept_of_tangent : ℝ := -1

-- Define the area of the triangle formed by the tangent line and the coordinate axes
def triangle_area : ℝ := 0.5 * (-1) * (-1)

-- State the theorem to prove the area of the triangle
theorem area_of_triangle_formed_by_tangent_line : 
  triangle_area = 0.5 := by 
sorry

end NUMINAMATH_GPT_area_of_triangle_formed_by_tangent_line_l183_18341


namespace NUMINAMATH_GPT_order_of_y1_y2_y3_l183_18375

/-
Given three points A(-3, y1), B(3, y2), and C(4, y3) all lie on the parabola y = 2*(x - 2)^2 + 1,
prove that y2 < y3 < y1.
-/
theorem order_of_y1_y2_y3 :
  let y1 := 2 * (-3 - 2)^2 + 1
  let y2 := 2 * (3 - 2)^2 + 1
  let y3 := 2 * (4 - 2)^2 + 1
  y2 < y3 ∧ y3 < y1 :=
by
  let y1 := 2 * (-3 - 2)^2 + 1
  let y2 := 2 * (3 - 2)^2 + 1
  let y3 := 2 * (4 - 2)^2 + 1
  sorry

end NUMINAMATH_GPT_order_of_y1_y2_y3_l183_18375


namespace NUMINAMATH_GPT_first_interest_rate_l183_18301

theorem first_interest_rate (r : ℝ) : 
  (70000:ℝ) = (60000:ℝ) + (10000:ℝ) →
  (8000:ℝ) = (60000 * r / 100) + (10000 * 20 / 100) →
  r = 10 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_first_interest_rate_l183_18301


namespace NUMINAMATH_GPT_jo_climb_stairs_ways_l183_18329

def f : ℕ → ℕ
| 0 => 1
| 1 => 1
| 2 => 2
| (n + 3) => f (n + 2) + f (n + 1) + f n

theorem jo_climb_stairs_ways : f 8 = 81 :=
by
    sorry

end NUMINAMATH_GPT_jo_climb_stairs_ways_l183_18329


namespace NUMINAMATH_GPT_find_percentage_of_alcohol_l183_18339

theorem find_percentage_of_alcohol 
  (Vx : ℝ) (Px : ℝ) (Vy : ℝ) (Py : ℝ) (Vp : ℝ) (Pp : ℝ)
  (hx : Px = 10) (hvx : Vx = 300) (hvy : Vy = 100) (hvxy : Vx + Vy = 400) (hpxy : Pp = 15) :
  (Vy * Py / 100) = 30 :=
by
  sorry

end NUMINAMATH_GPT_find_percentage_of_alcohol_l183_18339


namespace NUMINAMATH_GPT_solve_for_x_l183_18369

theorem solve_for_x (x : ℤ) (h : 3 * x = 2 * x + 6) : x = 6 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l183_18369


namespace NUMINAMATH_GPT_shopkeeper_gain_first_pulse_shopkeeper_gain_second_pulse_shopkeeper_gain_third_pulse_l183_18374

def false_weight_kgs (false_weight_g : ℕ) : ℚ := false_weight_g / 1000

def shopkeeper_gain_percentage (false_weight_g price_per_kg : ℕ) : ℚ :=
  let actual_price := false_weight_kgs false_weight_g * price_per_kg
  let gain := price_per_kg - actual_price
  (gain / actual_price) * 100

theorem shopkeeper_gain_first_pulse :
  shopkeeper_gain_percentage 950 10 = 5.26 := 
sorry

theorem shopkeeper_gain_second_pulse :
  shopkeeper_gain_percentage 960 15 = 4.17 := 
sorry

theorem shopkeeper_gain_third_pulse :
  shopkeeper_gain_percentage 970 20 = 3.09 := 
sorry

end NUMINAMATH_GPT_shopkeeper_gain_first_pulse_shopkeeper_gain_second_pulse_shopkeeper_gain_third_pulse_l183_18374


namespace NUMINAMATH_GPT_students_voted_both_l183_18342

def total_students : Nat := 300
def students_voted_first : Nat := 230
def students_voted_second : Nat := 190
def students_voted_none : Nat := 40

theorem students_voted_both :
  students_voted_first + students_voted_second - (total_students - students_voted_none) = 160 :=
by
  sorry

end NUMINAMATH_GPT_students_voted_both_l183_18342


namespace NUMINAMATH_GPT_angle_OA_plane_ABC_l183_18385

noncomputable def sphere_radius (A B C : Type*) (O : Type*) : ℝ :=
  let surface_area : ℝ := 48 * Real.pi
  let AB : ℝ := 2
  let BC : ℝ := 4
  let angle_ABC : ℝ := Real.pi / 3
  let radius := Real.sqrt (surface_area / (4 * Real.pi))
  radius

noncomputable def length_AC (A B C : Type*) : ℝ :=
  let AB : ℝ := 2
  let BC : ℝ := 4
  let angle_ABC : ℝ := Real.pi / 3
  let AC := Real.sqrt (AB ^ 2 + BC ^ 2 - 2 * AB * BC * Real.cos angle_ABC)
  AC

theorem angle_OA_plane_ABC 
(A B C O : Type*)
(radius : ℝ)
(AC : ℝ) :
radius = 2 * Real.sqrt 3 ∧
AC = 2 * Real.sqrt 3 ∧ 
(AB : ℝ) = 2 ∧ 
(BC : ℝ) = 4 ∧ 
(angle_ABC : ℝ) = Real.pi / 3
→ ∃ (angle_OA_plane_ABC : ℝ), angle_OA_plane_ABC = Real.arccos (Real.sqrt 3 / 3) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_angle_OA_plane_ABC_l183_18385


namespace NUMINAMATH_GPT_original_amount_spent_l183_18377

noncomputable def price_per_mango : ℝ := 383.33 / 115
noncomputable def new_price_per_mango : ℝ := 0.9 * price_per_mango

theorem original_amount_spent (N : ℝ) (H1 : (N + 12) * new_price_per_mango = N * price_per_mango) : 
  N * price_per_mango = 359.64 :=
by 
  sorry

end NUMINAMATH_GPT_original_amount_spent_l183_18377


namespace NUMINAMATH_GPT_evaluate_series_l183_18306

noncomputable def infinite_series :=
  ∑' n, (n^3 + 2*n^2 - 3) / (n+3).factorial

theorem evaluate_series : infinite_series = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_series_l183_18306


namespace NUMINAMATH_GPT_find_constant_k_l183_18399

theorem find_constant_k (k : ℝ) :
  (-x^2 - (k + 9) * x - 8 = - (x - 2) * (x - 4)) → k = -15 :=
by 
  sorry

end NUMINAMATH_GPT_find_constant_k_l183_18399


namespace NUMINAMATH_GPT_first_part_results_count_l183_18344

theorem first_part_results_count : 
    ∃ n, n * 10 + 90 + (25 - n) * 20 = 25 * 18 ∧ n = 14 :=
by
  sorry

end NUMINAMATH_GPT_first_part_results_count_l183_18344


namespace NUMINAMATH_GPT_proof_g_2_l183_18319

def g (x : ℝ) : ℝ := 3 * x ^ 8 - 4 * x ^ 4 + 2 * x ^ 2 - 6

theorem proof_g_2 :
  g (-2) = 10 → g (2) = 1402 := by
  sorry

end NUMINAMATH_GPT_proof_g_2_l183_18319
