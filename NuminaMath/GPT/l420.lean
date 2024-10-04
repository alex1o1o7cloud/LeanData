import Mathlib

namespace b_plus_c_neg_seven_l420_420033

theorem b_plus_c_neg_seven {A B : Set ℝ} (hA : A = {x : ℝ | x > 3 ∨ x < -1}) (hB : B = {x : ℝ | -1 ≤ x ∧ x ≤ 4})
  (h_union : A ∪ B = Set.univ) (h_inter : A ∩ B = {x : ℝ | 3 < x ∧ x ≤ 4}) :
  ∃ b c : ℝ, (∀ x, x^2 + b * x + c ≤ 0 ↔ x ∈ B) ∧ b + c = -7 :=
by
  sorry

end b_plus_c_neg_seven_l420_420033


namespace pyramid_triangular_face_area_l420_420208

theorem pyramid_triangular_face_area 
  (base : ℝ) (lateral : ℝ)
  (h_base : base = 8) (h_lateral : lateral = 7) :
  let height := sqrt (lateral^2 - (base / 2)^2) in
  4 * (1 / 2 * base * height) = 16 * sqrt 33 := 
by
  sorry

end pyramid_triangular_face_area_l420_420208


namespace tangent_line_at_3_monotonic_intervals_l420_420883

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x^3 - x - 6

theorem tangent_line_at_3 : ∃ (m : ℝ) (c : ℝ), m = 8 ∧ c = -24 ∧ (∀ x y : ℝ, y = f 3 + m * (x - 3) ↔ y = 8 * x - 24) := by
  sorry

theorem monotonic_intervals :
  (∀ x, x < -1 → (f' x) > 0) ∧ (∀ x, x > 1 → (f' x) > 0) ∧ (∀ x, -1 < x ∧ x < 1 → (f' x) < 0) := by
  sorry

end tangent_line_at_3_monotonic_intervals_l420_420883


namespace sum_mn_l420_420460

theorem sum_mn (m n : ℕ) (h1 : A n m = 272) (h2 : 2 * C n m = 272) (h3 : m ≠ 0) (h4 : n ≠ 0) : m + n = 19 :=
sorry

end sum_mn_l420_420460


namespace manager_salary_calculation_l420_420923

theorem manager_salary_calculation :
  let percent_marketers := 0.60
  let salary_marketers := 50000
  let percent_engineers := 0.20
  let salary_engineers := 80000
  let percent_sales_reps := 0.10
  let salary_sales_reps := 70000
  let percent_managers := 0.10
  let total_average_salary := 75000
  let total_contribution := percent_marketers * salary_marketers + percent_engineers * salary_engineers + percent_sales_reps * salary_sales_reps
  let managers_total_contribution := total_average_salary - total_contribution
  let manager_salary := managers_total_contribution / percent_managers
  manager_salary = 220000 :=
by
  sorry

end manager_salary_calculation_l420_420923


namespace mean_of_combined_set_l420_420634

theorem mean_of_combined_set (A B : Finset ℝ)
  (hA_card : A.card = 5) (hB_card : B.card = 6)
  (hA_mean : A.sum / 5 = 13) (hB_mean : B.sum / 6 = 24) :
  (A.sum + B.sum) / 11 = 19 :=
by
  sorry

end mean_of_combined_set_l420_420634


namespace time_after_6666_seconds_l420_420950

noncomputable def initial_time : Nat := 3 * 3600
noncomputable def additional_seconds : Nat := 6666

-- Function to convert total seconds to "HH:MM:SS" format
def time_in_seconds (h m s : Nat) : Nat :=
  h*3600 + m*60 + s

noncomputable def new_time : Nat :=
  initial_time + additional_seconds

-- Convert the new total time back to "HH:MM:SS" format (expected: 4:51:06)
def hours (secs : Nat) : Nat := secs / 3600
def minutes (secs : Nat) : Nat := (secs % 3600) / 60
def seconds (secs : Nat) : Nat := (secs % 3600) % 60

theorem time_after_6666_seconds :
  hours new_time = 4 ∧ minutes new_time = 51 ∧ seconds new_time = 6 :=
by
  sorry

end time_after_6666_seconds_l420_420950


namespace dk_eq_dl_l420_420958

open Locale Classical

noncomputable theory

-- Lean 4 math problem statement

variables {A B C D E F G K L : Type}
variables (iso_tri : is_isosceles_triangle A B C)
variables (incircle_touches_AB : incircle_touches A B C D)
variables (incircle_touches_BC : incircle_touches B C E)
variables (line_through_A_intersects : ∃ F G, line_through A ∩ incircle A B C = {F, G})
variables (AB_intersects_EF_EG : ∃ K L, line_intersections AB F G E K L)

-- Prove that DK = DL
theorem dk_eq_dl : DK = DL :=
sorry

end dk_eq_dl_l420_420958


namespace cannot_color_all_gray_l420_420082

def is_gray (grid : ℕ × ℕ → bool) (r c : ℕ) : bool := grid (r, c)

def gray_neighbors (grid : ℕ × ℕ → bool) (r c : ℕ) : ℕ :=
  (if is_gray grid (r - 1) c then 1 else 0) +
  (if is_gray grid (r + 1) c then 1 else 0) +
  (if is_gray grid r (c - 1) then 1 else 0) +
  (if is_gray grid r (c + 1) then 1 else 0)

def can_color (grid : ℕ × ℕ → bool) (r c : ℕ) : bool :=
  grid (r, c) = false ∧ (gray_neighbors grid r c = 1 ∨ gray_neighbors grid r c = 3)

theorem cannot_color_all_gray : ∀ (grid : ℕ × ℕ → bool),
  (∃ (r c : ℕ), 1 ≤ r ∧ r ≤ 8 ∧ 1 ≤ c ∧ c ≤ 8 ∧ grid (r, c) = true) →
  (∀ r c, 1 ≤ r ∧ r ≤ 8 ∧ 1 ≤ c ∧ c ≤ 8 → (grid (r, c) = true ∨ can_color grid r c = false)) →
  ¬ (∀ r c, 1 ≤ r ∧ r ≤ 8 ∧ 1 ≤ c ∧ c ≤ 8 → grid (r, c) = true) :=
begin
  sorry
end

end cannot_color_all_gray_l420_420082


namespace valid_plantings_count_l420_420357

-- Define the grid structure
structure Grid3x3 :=
  (sections : Fin 9 → String)

noncomputable def crops := ["corn", "wheat", "soybeans", "potatoes", "oats"]

-- Define the adjacency relationships and restrictions as predicates
def adjacent (i j : Fin 9) : Prop :=
  (i = j + 1 ∧ j % 3 ≠ 2) ∨ (i = j - 1 ∧ i % 3 ≠ 2) ∨ (i = j + 3) ∨ (i = j - 3)

def valid_crop_planting (g : Grid3x3) : Prop :=
  ∀ i j, adjacent i j →
    (¬(g.sections i = "corn" ∧ g.sections j = "wheat") ∧ 
    ¬(g.sections i = "wheat" ∧ g.sections j = "corn") ∧
    ¬(g.sections i = "soybeans" ∧ g.sections j = "potatoes") ∧
    ¬(g.sections i = "potatoes" ∧ g.sections j = "soybeans") ∧
    ¬(g.sections i = "oats" ∧ g.sections j = "potatoes") ∧ 
    ¬(g.sections i = "potatoes" ∧ g.sections j = "oats"))

noncomputable def count_valid_plantings : Nat :=
  -- Placeholder for the actual count computing function
  sorry

theorem valid_plantings_count : count_valid_plantings = 5 :=
  sorry

end valid_plantings_count_l420_420357


namespace line_through_B_l420_420610

theorem line_through_B
  (S1 S2 : Circle)
  (A B : Point)
  (h_intersect : intersects S1 S2 A B)
  (P : rotationalHomothety S1 S2 A)
  (M1 : Point)
  (h_M1_on_S1 : onCircle M1 S1)
  (M2 : Point)
  (h_M2_on_S2 : P M1 = M2) :
  collinear M1 M2 B :=
sorry

end line_through_B_l420_420610


namespace spider_total_distance_l420_420374

theorem spider_total_distance
    (radius : ℝ)
    (diameter : ℝ)
    (half_diameter : ℝ)
    (final_leg : ℝ)
    (total_distance : ℝ) :
    radius = 75 →
    diameter = 2 * radius →
    half_diameter = diameter / 2 →
    final_leg = 90 →
    (half_diameter ^ 2 + final_leg ^ 2 = diameter ^ 2) →
    total_distance = diameter + half_diameter + final_leg →
    total_distance = 315 :=
by
  intros
  sorry

end spider_total_distance_l420_420374


namespace midpoint_bc_am_eq_two_l420_420560

variables {A B C M : Type*} [add_group A] [normed_group A] [normed_group B] [normed_group C] [normed_group M]
variables (AB AC AM BC : A) (BM MC : B) (real4 : ℝ)
variables [a_b_c_m : add_group A] [norm4_group : add_group B]

/-- Let point M be the midpoint of the line segment BC, and point A be outside the line BC. Given
BC^2 = 16 and ||AB + AC|| = ||AB - AC||, find the value of ||AM||.
-/
theorem midpoint_bc_am_eq_two
  (hm : 2 • M = B + C)
  (hm_mid : BC^2 = 16)
  (h_eq : ||AB + AC|| = ||AB - AC||) :
  ||AM|| = 2 := by
sorry

end midpoint_bc_am_eq_two_l420_420560


namespace smallest_positive_period_range_of_f_l420_420022

noncomputable def f (x : ℝ) : ℝ := 4 * sin x * cos (x + π / 6) + 1

theorem smallest_positive_period : (∃ T > 0, ∀ x, f(x + T) = f(x)) ∧ T = π := by
  sorry

theorem range_of_f : set.range (λ x : ℝ, f x) = set.Icc (-2 : ℝ) 1 := by
  sorry

end smallest_positive_period_range_of_f_l420_420022


namespace train_length_l420_420748

theorem train_length (speed_km_hr : ℝ) (time_sec : ℝ) (length_m : ℝ) 
  (h1 : speed_km_hr = 60) (h2 : time_sec = 9) : length_m = 150 := by
  sorry

end train_length_l420_420748


namespace distance_between_trains_l420_420186

theorem distance_between_trains
  (v1 v2 : ℕ) (d_diff : ℕ)
  (h_v1 : v1 = 50) (h_v2 : v2 = 60) (h_d_diff : d_diff = 100) :
  ∃ d, d = 1100 :=
by
  sorry

-- Explanation:
-- v1 is the speed of the first train.
-- v2 is the speed of the second train.
-- d_diff is the difference in the distances traveled by the two trains at the time of meeting.
-- h_v1 states that the speed of the first train is 50 kmph.
-- h_v2 states that the speed of the second train is 60 kmph.
-- h_d_diff states that the second train travels 100 km more than the first train.
-- The existential statement asserts that there exists a distance d such that d equals 1100 km.

end distance_between_trains_l420_420186


namespace num_ways_A_middle_l420_420439

def is_middle_position (n: ℕ) := n = 1 ∨ n = 2

theorem num_ways_A_middle : 
  let positions := {0, 1, 2, 3}
  let perms := positions - {0, 3}
  list.card perms = 18 :=
by sorry

end num_ways_A_middle_l420_420439


namespace third_red_yellow_flash_is_60_l420_420741

-- Define the flashing intervals for red, yellow, and green lights
def red_interval : Nat := 3
def yellow_interval : Nat := 4
def green_interval : Nat := 8

-- Define the function for finding the time of the third occurrence of only red and yellow lights flashing together
def third_red_yellow_flash : Nat :=
  let lcm_red_yellow := Nat.lcm red_interval yellow_interval
  let times := (List.range (100)).filter (fun t => t % lcm_red_yellow = 0 ∧ t % green_interval ≠ 0)
  times[2] -- Getting the third occurrence

-- Prove that the third occurrence time is 60 seconds
theorem third_red_yellow_flash_is_60 :
  third_red_yellow_flash = 60 :=
  by
    -- Proof goes here
    sorry

end third_red_yellow_flash_is_60_l420_420741


namespace pyramid_total_area_l420_420225

/-- The total area of the four triangular faces of a right, square-based pyramid
whose base edges measure 8 units and lateral edges measure 7 units is 16√33. -/
theorem pyramid_total_area :
  let base_edge := 8
  let lateral_edge := 7
  4 * (1 / 2 * base_edge * Real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 16 * Real.sqrt 33 := 
by
  let base_edge := 8
  let lateral_edge := 7
  have h1 : 4 * (1 / 2 * base_edge * Real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 
              4 * (1 / 2 * 8 * Real.sqrt (49 - 16)) := 
    by sorry
  have h2 : 4 * (1 / 2 * 8 * Real.sqrt (49 - 16)) = 4 * (4 * Real.sqrt 33) := 
    by sorry
  have h3 : 4 * (4 * Real.sqrt 33) = 16 * Real.sqrt 33 := 
    by sorry
  exact eq.trans (eq.trans h1 h2) h3

end pyramid_total_area_l420_420225


namespace proof_problem_l420_420865

noncomputable def circle_equation : Prop :=
  ∃ (D E F : ℝ), 
  E = D + 4 ∧
  13 + 3 * D - 2 * E + F = 0 ∧
  25 - 5 * E + F = 0 ∧
  D = 0 ∧ E = 4 ∧ F = -5

noncomputable def tangent_relationship : Prop :=
  ∀ x y, 
  (x ≠ 0 → (x, y) ∈ setOf (λ p => p ∈ (λ x y, x^2 + (y + 2)^2 = 9))→
  let R_x := 6 * x / (y + 5) in
  let S_x := 3 * x / (y + 5) in
  let r := 3 * |x| / |y + 5| in
  let distance := 3 * |x| / |y + 5| in
  let (mx, my) := (x, y - 2) in
  circle_equation → 
  (mx * mx + my * my = 9 →
  distance = r)

noncomputable def x_coordinate_range : Prop :=
  ∃ (a : ℝ),
  (a, 3) ∈ setOf (λ c => c ∈ (λ x y, x^2 + (y + 2)^2 = 9)) ∧
  (∃ (N : ℝ × ℝ), (N.1, N.2) ∈ setOf (λ x y, x^2 + (y + 1)^2 = 4 ∧ |Real.sqrt ((x - 0)^2 + (y - 3)^2) = 2 * Real.sqrt((x - (a, a - 2).1)^2 + (y - (a, a - 2).2)^2)) → 
  let EC := Real.sqrt(2 * a^2 - 2 * a + 1) in
  1 ≤ EC ∧ EC ≤ 6 →
  (-3 ≤ a ∧ a ≤ 0) ∨ (1 ≤ a ∧ a ≤ 4))

theorem proof_problem : circle_equation ∧ tangent_relationship ∧ x_coordinate_range := ⟨sorry, sorry, sorry⟩

end proof_problem_l420_420865


namespace smallest_z_value_l420_420940

theorem smallest_z_value :
  ∀ w x y z : ℤ, (∃ k : ℤ, w = 2 * k - 1 ∧ x = 2 * k + 1 ∧ y = 2 * k + 3 ∧ z = 2 * k + 5) ∧
    w^3 + x^3 + y^3 = z^3 →
    z = 9 :=
sorry

end smallest_z_value_l420_420940


namespace unique_position_Q_l420_420896

theorem unique_position_Q (E F P A C B D Q : Point)
  (e a A_E C_F A_F C_E B_D : Line)
  (h1 : E ≠ F ∧ F ≠ P ∧ P ≠ E)
  (h2 : (E ∈ e) ∧ (F ∈ e) ∧ (P ∈ e))
  (h3 : (A ∈ a) ∧ (C ∈ a) ∧ (P ∈ a))
  (h4 : intersects e a = P)
  (h5 : A ∉ {P})
  (h6 : C ∉ {P})
  (h7 : B = line_intersection A_E C_F)
  (h8 : D = line_intersection A_F C_E)
  (h9 : Q = line_intersection e B_D)
  : Q = Q := 
sorry

end unique_position_Q_l420_420896


namespace vertex_parabola_part_l420_420435

noncomputable def vertex_xt (t : ℝ) : ℝ := -t / (t^2 + 1)
noncomputable def vertex_yt (t : ℝ) (c : ℝ) : ℝ := c - t^2 / (t^2 + 1)

theorem vertex_parabola_part (c : ℝ) (h : c > 0) :
  ∃ t : ℝ, (vertex_xt t, vertex_yt t c) ∈ {p : ℝ × ℝ | True} ∧
  ¬ (vertex_xt t, vertex_yt t c) forms a complete standard parabola :=
sorry

end vertex_parabola_part_l420_420435


namespace score_difference_l420_420928

noncomputable def mean_score (scores pcts : List ℕ) : ℚ := 
  (List.zipWith (· * ·) scores pcts).sum / 100

def median_score (scores pcts : List ℕ) : ℚ := 75

theorem score_difference :
  let scores := [60, 75, 85, 95]
  let pcts := [20, 50, 15, 15]
  abs (median_score scores pcts - mean_score scores pcts) = 1.5 := by
  sorry

end score_difference_l420_420928


namespace maximum_ratio_squared_l420_420970

theorem maximum_ratio_squared (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_ge : a ≥ b)
  (x y : ℝ) (h_x : 0 ≤ x) (h_xa : x < a) (h_y : 0 ≤ y) (h_yb : y < b)
  (h_eq1 : a^2 + y^2 = b^2 + x^2)
  (h_eq2 : b^2 + x^2 = (a - x)^2 + (b - y)^2) :
  (a / b)^2 ≤ 4 / 3 :=
sorry

end maximum_ratio_squared_l420_420970


namespace determine_coordinates_l420_420409

noncomputable def sun_overhead_coordinates := 
  let budapest_coord : (ℝ × ℝ) := (19.1, 47.5)
  let dec22 := [(-23.5, 80.8), (-23.5, -42.6)]
  let mar21_sep23 := [(0.0, 109.1), (0.0, -70.9)]
  let jun22 := [(23.5, 137.4), (23.5, 99.2)]
  let angle60 := [(17.5, 129.2), (17.5, -91.0)]
  let angle30 := [(-12.5, 95.1), (-12.5, -56.9)]
  (dec22, mar21_sep23, jun22, angle60, angle30)

-- Prove that these are the correct latitudes and longitudes for the specified conditions.
theorem determine_coordinates : 
  ∃ (dec22 mar21_sep23 jun22 angle60 angle30 : List (ℝ × ℝ)),
  sun_overhead_coordinates = (dec22, mar21_sep23, jun22, angle60, angle30) ∧ 
  dec22 = [(-23.5, 80.8), (-23.5, -42.6)] ∧ 
  mar21_sep23 = [(0.0, 109.1), (0.0, -70.9)] ∧ 
  jun22 = [(23.5, 137.4), (23.5, 99.2)] ∧ 
  angle60 = [(17.5, 129.2), (17.5, -91.0)] ∧ 
  angle30 = [(-12.5, 95.1), (-12.5, -56.9)] 
  := 
  sorry

end determine_coordinates_l420_420409


namespace april_revenue_l420_420390

def revenue_after_tax (initial_roses : ℕ) (initial_tulips : ℕ) (initial_daisies : ℕ)
                      (final_roses : ℕ) (final_tulips : ℕ) (final_daisies : ℕ)
                      (price_rose : ℝ) (price_tulip : ℝ) (price_daisy : ℝ) (tax_rate : ℝ) : ℝ :=
(price_rose * (initial_roses - final_roses) + price_tulip * (initial_tulips - final_tulips) + price_daisy * (initial_daisies - final_daisies)) * (1 + tax_rate)

theorem april_revenue :
  revenue_after_tax 13 10 8 4 3 1 4 3 2 0.10 = 78.10 := by
  sorry

end april_revenue_l420_420390


namespace equation_of_AD_equation_of_AE_l420_420467

-- Definitions based on given conditions.
def point := ℝ × ℝ

def A : point := (1, -1)
def B : point := (-1, 3)
def C : point := (3, 0)

def line (p1 p2 : point) : ℝ × ℝ × ℝ :=
  let k := (p2.2 - p1.2) / (p2.1 - p1.1) in
  let intercept := p1.2 - k * p1.1 in
  (k, -1, intercept)

-- A line in standard form ax + by + c = 0
def line_eq (a b c : ℝ) (x y : ℝ) : Prop := a * x + b * y + c = 0

-- Proof problem for part (1): Equation of line AD.
theorem equation_of_AD :
  ∃ (a b c : ℝ), line_eq 4 (-3) (-7) (1 : ℝ) (-1) ∧
                 ∀ p : point, line (B, C) p → p.1 * 4 + p.2 * -3 + -7 = 0 :=
begin
  sorry
end

-- Proof problem for part (2): Equation of line AE.
theorem equation_of_AE :
  ∃ (a b c : ℝ), line_eq 1 (-1) (-4) (1 : ℝ) (-1) ∧
                 ∀ p : point, line (A, C) p → p.1 * 1 + p.2 * -1 + -4 = 0 :=
begin
  sorry
end

end equation_of_AD_equation_of_AE_l420_420467


namespace total_area_of_pyramid_faces_l420_420267

-- Define the basic parameters of the pyramid
def base_edges := 8
def lateral_edges := 7

-- Define the Pythagorean theorem components
def altitude_squared := lateral_edges^2 - (base_edges / 2)^2
def altitude := real.sqrt altitude_squared

-- Define the area of one triangular face using half-base and altitude
def one_face_area := (1 / 2) * base_edges * altitude

-- Define the total area of the four triangular faces
def total_area_of_faces := 4 * one_face_area

-- Statement to prove
theorem total_area_of_pyramid_faces : total_area_of_faces = 16 * real.sqrt 33 :=
by
  -- Necessary calculations are assured correct by previous definitions and attributes
  sorry

end total_area_of_pyramid_faces_l420_420267


namespace cone_height_l420_420927

theorem cone_height (r l h : ℝ) (h_r : r = 1) (h_l : l = 4) : h = Real.sqrt 15 :=
by
  -- proof steps would go here
  sorry

end cone_height_l420_420927


namespace roots_quadratic_inequality_l420_420128

theorem roots_quadratic_inequality (t x1 x2 : ℝ) (h_eqn : x1 ^ 2 - t * x1 + t = 0) 
  (h_eqn2 : x2 ^ 2 - t * x2 + t = 0) (h_real : x1 + x2 = t) (h_prod : x1 * x2 = t) :
  x1 ^ 2 + x2 ^ 2 ≥ 2 * (x1 + x2) := 
sorry

end roots_quadratic_inequality_l420_420128


namespace train_length_160_l420_420332

noncomputable def length_of_train (speed_kmph : ℕ) (time_s : ℕ) : ℕ :=
  (speed_kmph * 1000 / 3600) * time_s

theorem train_length_160 (h : ∀ s t, length_of_train s t = (s * 1000 / 3600) * t) : length_of_train 72 8 = 160 := 
by {
  have h_speed_conv : (72 * 1000 / 3600) = 20 := by norm_num,
  rw h_speed_conv,
  have h_length : 20 * 8 = 160 := by norm_num,
  exact h_length,
}

end train_length_160_l420_420332


namespace determine_rectangle_and_side_lengths_l420_420584

-- Define two intersecting lines at angle 2φ
variables {L₁ L₂ : set (ℝ × ℝ)}
variables {φ : ℝ}
variables {d : ℝ}

-- Define that they intersect at angle 2φ
def intersecting_at_angle (L₁ L₂ : set (ℝ × ℝ)) (φ : ℝ) : Prop :=
∃ A ∈ L₁, ∃ B ∈ L₂, (L₁ ∩ L₂).nonempty ∧ -- existence of intersection point
  ∃ R, ∃ S, R ≠ S ∧ -- distinct points required to define angle
  angle R (intersect_point R S) S = 2 * φ -- angle between lines is 2φ

-- Define the rectangle locus satisfying the distance condition
def rectangle_locus_is_correct (L₁ L₂ : set (ℝ × ℝ)) (φ d : ℝ) : Prop :=
∃ R : set (ℝ × ℝ), 
(is_rectangle R ∧ -- it forms a rectangle
∀ P ∈ R, -- for every point P on the rectangle
  distance_from_line P L₁ + distance_from_line P L₂ = d) -- sum of distances is d

-- Define the side lengths using d and φ
def side_lengths_correct (φ d : ℝ) : Prop :=
∃ a b,
a = d / real.sin φ ∧
b = d / real.cos φ

-- Main statement to show locus forms a rectangle and calculate side lengths.
theorem determine_rectangle_and_side_lengths
  (L₁ L₂ : set (ℝ × ℝ)) (φ d : ℝ) 
  (h_intersecting : intersecting_at_angle L₁ L₂ φ) :
  rectangle_locus_is_correct L₁ L₂ φ d ∧ side_lengths_correct φ d :=
sorry

end determine_rectangle_and_side_lengths_l420_420584


namespace sum_of_arithmetic_sequence_l420_420693

theorem sum_of_arithmetic_sequence (a d : ℤ) (n : ℕ) (h1 : a = -3) (h2 : d = 7) (h3 : n = 10) :
  let aₙ := a + (n - 1) * d in
  (a + aₙ) * n / 2 = 285 :=
by
  sorry

end sum_of_arithmetic_sequence_l420_420693


namespace probability_sum_of_terms_l420_420852

theorem probability_sum_of_terms (f g : ℝ → ℝ) (a : ℝ) (h1 : ∀ x, g x ≠ 0)
  (h2 : ∀ x, f x * deriv g x > deriv f x * g x)
  (h3 : ∀ x, f x = a^x * g x) (h4 : 0 < a ∧ a ≠ 1)
  (h5 : f 1 / g 1 + f (-1) / g (-1) = 5 / 2) :
  let s := λ n : ℕ, (finset.range n).sum (λ k, (a:ℝ) ^ (k + 1))
  in (finset.range 6).card.to_rat / 10 = 3 / 5 :=
sorry

end probability_sum_of_terms_l420_420852


namespace sphere_and_tube_dimensions_l420_420361

-- Definitions of given conditions
def original_radius : ℝ := 6 -- Radius of original sphere in dm
def thickness_cm : ℝ := 2 / 10 -- Thickness of the cylindrical tube in dm
def thickness_addition : ℝ := 0.02 -- Thickness in dm

-- Required results to prove
def expected_diameter_sphere : ℝ := 7.6
def expected_height_tube : ℝ := 12.94

-- Mathematical proof problem
theorem sphere_and_tube_dimensions :
  let r := original_radius in
  let third_radius := r in
  let r_small := (2 * r * real.sqrt 2) / 3 in
  let volume_sphere_segment := (4 / 3) * real.pi * (r_small^3) in
  let volume_smaller_sphere := volume_sphere_segment / 2 in
  let diameter_smaller_sphere := 2 * real.cbrt volume_smaller_sphere  in
  let volume_cylindrical_tube := (160 * real.pi) in
  let volume_cylinder := volume_cylindrical_tube / (real.pi * (diameter_smaller_sphere^2)) in
  diameter_smaller_sphere ≈ expected_diameter_sphere ∧
  volume_cylinder ≈ expected_height_tube := by 
  sorry

end sphere_and_tube_dimensions_l420_420361


namespace smallest_positive_period_range_of_f_range_of_m_l420_420021

-- The function f(x)
def f (x : ℝ) : ℝ := 2 * sqrt 3 * sin (π - x) * cos x + 2 * cos x ^ 2

-- Problem 1: Prove the smallest positive period
theorem smallest_positive_period : ∀ x : ℝ, f (x + π) = f x :=
by sorry

-- Problem 2: Prove the range of f(x) for x in [-π/6, π/3]
theorem range_of_f : set.range (f ∘ (λ x : ℝ, x - π / 6)) ⊆ set.Icc 0 3 :=
by sorry

-- Problem 3: Prove the range of m such that g(x) = f(x) - 1 has exactly two zeros in [-π/6, m]
theorem range_of_m (f_eq : ∀ x, f x = 2 * sin (2 * x + π / 6) + 1) :
  ∀ m : ℝ, (∀ x ∈ set.Icc (-π / 6) m, (2 * sin (2 * x + π / 6) = 0) ↔ x = -π / 6 ∨ x = m) ↔ m ∈ set.Icc (5 * π / 12) (11 * π / 12) :=
by sorry

end smallest_positive_period_range_of_f_range_of_m_l420_420021


namespace max_value_abs_ab_l420_420893

-- Define the quadratic function f(x)
def f (a b c x : ℝ) :=
  a * (3 * a + 2 * c) * x^2 - 2 * b * (2 * a + c) * x + b^2 + (c + a)^2

-- Problem statement: Prove that the maximum value of |ab| is 3√3/8 given the conditions
theorem max_value_abs_ab (a b c : ℝ) (h : ∀ x : ℝ, f a b c x ≤ 1) : |a * b| ≤ 3 * real.sqrt 3 / 8 :=
sorry

end max_value_abs_ab_l420_420893


namespace vector_orthogonality_solution_l420_420490

theorem vector_orthogonality_solution :
  let a := (3, -2)
  let b := (x, 1)
  (a.1 * b.1 + a.2 * b.2 = 0) →
  x = 2 / 3 :=
by
  intro h
  sorry

end vector_orthogonality_solution_l420_420490


namespace breadth_remains_the_same_l420_420628

variable (L B : ℝ)

theorem breadth_remains_the_same 
  (A : ℝ) (hA : A = L * B) 
  (L_half : ℝ) (hL_half : L_half = L / 2) 
  (B' : ℝ)
  (A' : ℝ) (hA' : A' = L_half * B') 
  (hA_change : A' = 0.5 * A) : 
  B' = B :=
  sorry

end breadth_remains_the_same_l420_420628


namespace correct_statements_l420_420479

noncomputable def function_axiom (ω : ℝ) (hω : ω > 0) (x : ℝ) : ℝ :=
  Real.sin (ω * x + (Real.pi / 4))

def is_monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a < x ∧ y < b ∧ x < y → f x < f y

theorem correct_statements (ω : ℝ) (hω : ω > 0) :
  (B := "The minimum positive period of f(x) could be 2π/3.") ∧
  (D := "f(x) is monotonically increasing on the interval (0, π/15).") :=
by
  let f := function_axiom ω hω
  have period_correct : ∃ T > 0, ∀ x, f (x + T) = f x := sorry
  have min_positive_period_correct := sorry
  have monotonic_behavior : is_monotonically_increasing f 0 (Real.pi / 15) := sorry
  show B ∧ D, from ⟨min_positive_period_correct, monotonic_behavior⟩

end correct_statements_l420_420479


namespace cos_neg_13pi_over_4_l420_420644

theorem cos_neg_13pi_over_4 : Real.cos (-13 * Real.pi / 4) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_neg_13pi_over_4_l420_420644


namespace overall_yield_percentage_is_4point22_l420_420151

-- Definitions for the given conditions
def stockA_value := 500
def stockB_value := 750
def stockC_value := 1000

def stockA_yield := 0.14
def stockB_yield := 0.08
def stockC_yield := 0.12

def tax_rate := 0.02
def commission_fee := 50

def total_market_value := stockA_value + stockB_value + stockC_value

def yield_after_tax (yield before_tax : ℝ) := yield_before_tax - yield_before_tax * tax_rate

def stockA_yield_after_tax := yield_after_tax (stockA_yield * stockA_value)
def stockB_yield_after_tax := yield_after_tax (stockB_yield * stockB_value)
def stockC_yield_after_tax := yield_after_tax (stockC_yield * stockC_value)

def total_yield_after_tax := stockA_yield_after_tax + stockB_yield_after_tax + stockC_yield_after_tax
def total_commission_fees := 3 * commission_fee

def total_yield_after_commission := total_yield_after_tax - total_commission_fees

def overall_yield_percentage := (total_yield_after_commission / total_market_value) * 100

-- The final proof statement that needs to be proved
theorem overall_yield_percentage_is_4point22 : overall_yield_percentage = 4.22 := 
sorry

end overall_yield_percentage_is_4point22_l420_420151


namespace find_second_quadrant_point_l420_420766

def is_second_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

theorem find_second_quadrant_point :
  (is_second_quadrant (2, 3) = false) ∧
  (is_second_quadrant (2, -3) = false) ∧
  (is_second_quadrant (-2, -3) = false) ∧
  (is_second_quadrant (-2, 3) = true) := 
sorry

end find_second_quadrant_point_l420_420766


namespace find_B_l420_420553

theorem find_B (A B : ℕ) (h1 : Prime A) (h2 : Prime B) (h3 : A > 0) (h4 : B > 0) 
  (h5 : 1 / A - 1 / B = 192 / (2005^2 - 2004^2)) : B = 211 :=
sorry

end find_B_l420_420553


namespace joint_distribution_determined_l420_420139

variables {Ω : Type*} [probability_space Ω]

-- Definitions for conditions
def prob_X_eq_1 (X : Ω → bool) : ℝ := probability (event_of (λ ω, X ω = tt))
def prob_Y_eq_1 (Y : Ω → bool) : ℝ := probability (event_of (λ ω, Y ω = tt))
def cov_X_Y (X Y : Ω → bool) : ℝ := 
  expectation (λ ω, (X ω : ℝ) * (Y ω : ℝ)) - expectation (λ ω, (X ω : ℝ)) * expectation (λ ω, (Y ω : ℝ))

-- Problem Statement
theorem joint_distribution_determined (X Y : Ω → bool) :
  (∃ (pX pY cov : ℝ), 
    prob_X_eq_1 X = pX ∧ prob_Y_eq_1 Y = pY ∧ cov_X_Y X Y = cov) →
  (∀ pX pY cov, 
    prob_X_eq_1 X = pX → 
    prob_Y_eq_1 Y = pY → 
    cov_X_Y X Y = cov →
    ∃ (PX_Y : ℙ), joint_distribution X Y PX_Y = (pX, pY, cov)) ∧
  (cov_X_Y X Y = 0 → 
    indep_events (event_of (λ ω, X ω = tt)) (event_of (λ ω, Y ω = tt))) :=
sorry

end joint_distribution_determined_l420_420139


namespace color_theorem_l420_420982

open Function

theorem color_theorem (n : ℕ) (h1 : 0 < n) (colored : Fin n → Fin _) 
  (h2 : ∀ (a b : Fin n), a < b → a + b < n → (colored a = colored b ∨ colored a = colored (a + b) ∨ colored b = colored (a + b))) :
  ∃ c, (Fin n).count (λ x => colored x = c) ≥ 2 * n / 5 :=
sorry

end color_theorem_l420_420982


namespace pyramid_face_area_total_l420_420238

theorem pyramid_face_area_total 
  (base_edge : ℝ) (lateral_edge : ℝ) 
  (h_base_edge : base_edge = 8) 
  (h_lateral_edge : lateral_edge = 7) : 
  4 * (1 / 2 * base_edge * real.sqrt (lateral_edge^2 - (base_edge / 2)^2)) = 16 * real.sqrt 33 :=
by
  sorry

end pyramid_face_area_total_l420_420238


namespace melissa_games_l420_420999

noncomputable def total_points_scored := 91
noncomputable def points_per_game := 7
noncomputable def number_of_games_played := total_points_scored / points_per_game

theorem melissa_games : number_of_games_played = 13 :=
by 
  sorry

end melissa_games_l420_420999


namespace triangle_side_length_l420_420125

theorem triangle_side_length (P Q : ℝ × ℝ) 
  (hP : P.2 = -1/2 * P.1^2) 
  (hQ : Q.2 = -1/2 * Q.1^2) 
  (hEquilateral : ∀ (O : ℝ × ℝ), O = (0, 0) → 
                dist P O = dist Q O ∧ dist P O = dist P Q) : 
  dist P (0, 0) = 4 * real.sqrt 3 :=
begin
  sorry
end

end triangle_side_length_l420_420125


namespace min_possible_value_of_coefficient_x_l420_420904

theorem min_possible_value_of_coefficient_x 
  (c d : ℤ) 
  (h1 : c * d = 15) 
  (h2 : ∃ (C : ℤ), C = c + d) 
  (h3 : c ≠ d ∧ c ≠ 34 ∧ d ≠ 34) :
  (∃ (C : ℤ), C = c + d ∧ C = 34) :=
sorry

end min_possible_value_of_coefficient_x_l420_420904


namespace solution_set_of_abs_inequality_is_real_l420_420055

theorem solution_set_of_abs_inequality_is_real (m : ℝ) :
  (∀ x : ℝ, |x + 1| + |x - 2| + m - 7 > 0) ↔ m > 4 :=
by
  sorry

end solution_set_of_abs_inequality_is_real_l420_420055


namespace problem_statement_l420_420149

theorem problem_statement (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h1 : x * y * z = 1) (h2 : x + 1 / z = 6) (h3 : y + 1 / x = 30) : 
  z + 1 / y = 38 / 179 :=
  sorry

end problem_statement_l420_420149


namespace solution1_solution2_l420_420876

noncomputable def problem1 : Prop :=
  ∃ (a b : ℤ), 
  (∃ (n : ℤ), 3*a - 14 = n ∧ a - 2 = n) ∧ 
  (b - 15 = -27) ∧ 
  a = 4 ∧ 
  b = -12 ∧ 
  (4*a + b = 4)

noncomputable def problem2 : Prop :=
  ∀ (a b : ℤ), 
  (a = 4) ∧ 
  (b = -12) → 
  (4*a + b = 4) → 
  (∃ n, n^2 = 4 ∧ (n = 2 ∨ n = -2))

theorem solution1 : problem1 := by { sorry }
theorem solution2 : problem2 := by { sorry }

end solution1_solution2_l420_420876


namespace solve_for_x_l420_420496

theorem solve_for_x (x : ℝ) : 10 ^ (Real.log10 7) = 5 * x + 8 → x = -1 / 5 :=
by 
  intro h
  sorry

end solve_for_x_l420_420496


namespace distinct_flavors_l420_420433

/-- There are five red candies and four green candies. -/
structure Candies where
  red : Nat
  green : Nat

/-- Definition of flavors based on combinations of red and green candies. -/
def flavors (c : Candies) : Nat :=
  -- all the valid ratios
  let potential_ratios : Finset (Nat × Nat) :=
    (Finset.range (c.red + 1)).product (Finset.range (c.green + 1)) \ {(0, 0)}
  -- counting unique canonical ratios
  (potential_ratios.image (λ (r : Nat × Nat) => if r.2 = 0 then (1, 0) else r.gcd.natAbs)).card

/-- There are 17 distinct flavors. -/
theorem distinct_flavors : flavors ⟨5, 4⟩ = 17 := by
  sorry

end distinct_flavors_l420_420433


namespace num_counterexamples_l420_420820

def digits_sum_to_five (n : ℕ) : Prop :=
  (n.digits 10).sum = 5

def contains_no_zero (n : ℕ) : Prop :=
  ¬(0 ∈ n.digits 10)

def is_counterexample (n : ℕ) : Prop :=
  digits_sum_to_five n ∧ contains_no_zero n ∧ ¬Prime n

theorem num_counterexamples : 
  (Finset.filter is_counterexample (Finset.range 100000)).card = 10 := 
by
  sorry

end num_counterexamples_l420_420820


namespace parking_decision_l420_420903

theorem parking_decision:
  (parking_fee : ℕ) (fine : ℕ) (patrol_interval : ℕ) (error_rate : ℚ) (parking_time : ℕ)
  (expected_fine : ℚ) (p_no_pay : ℚ) :
  parking_fee = 2 ∧ fine = 48 ∧ patrol_interval = 120 ∧ error_rate = 1 / 4 ∧
  parking_time = 10 ∧ p_no_pay = (error_rate * (parking_time / (patrol_interval : ℚ))) ∧ 
  expected_fine = p_no_pay * fine →
  expected_fine < parking_fee := 
by
  sorry

end parking_decision_l420_420903


namespace number_of_tiles_in_each_row_l420_420601

-- Define the given conditions
def area_of_room : ℝ := 256
def tile_size_in_inches : ℝ := 8
def inches_per_foot : ℝ := 12

-- Length of the room in feet derived from the given area
def side_length_in_feet := Real.sqrt area_of_room

-- Convert side length from feet to inches
def side_length_in_inches := side_length_in_feet * inches_per_foot

-- The question: Prove that the number of tiles in each row is 24
theorem number_of_tiles_in_each_row :
  side_length_in_inches / tile_size_in_inches = 24 :=
sorry

end number_of_tiles_in_each_row_l420_420601


namespace find_ff_1_l420_420019

def f (x : ℝ) : ℝ :=
if x ≤ 0 then x^2 + 1 else -2 * x

theorem find_ff_1 : f (f 1) = 5 :=
by sorry

end find_ff_1_l420_420019


namespace total_area_of_pyramid_faces_l420_420263

-- Define the basic parameters of the pyramid
def base_edges := 8
def lateral_edges := 7

-- Define the Pythagorean theorem components
def altitude_squared := lateral_edges^2 - (base_edges / 2)^2
def altitude := real.sqrt altitude_squared

-- Define the area of one triangular face using half-base and altitude
def one_face_area := (1 / 2) * base_edges * altitude

-- Define the total area of the four triangular faces
def total_area_of_faces := 4 * one_face_area

-- Statement to prove
theorem total_area_of_pyramid_faces : total_area_of_faces = 16 * real.sqrt 33 :=
by
  -- Necessary calculations are assured correct by previous definitions and attributes
  sorry

end total_area_of_pyramid_faces_l420_420263


namespace classify_conic_section_third_quadrant_l420_420861

theorem classify_conic_section_third_quadrant (q : ℝ) (hq : π < q ∧ q < 3 * π / 2) : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 
    (∀ x y : ℝ, x^2 + y^2 * sin q = cos q ↔ 
    x^2 / (a * a) + y^2 / (b * b) = 1) ∧ b/a > 1 :=
sorry

end classify_conic_section_third_quadrant_l420_420861


namespace Cameron_list_count_l420_420779

theorem Cameron_list_count :
  ∃ n : ℕ, (900 ≤ 30 * n ∧ 30 * n ≤ 810000)
  ∧ (n ≤ 27000 - 30 + 1 = 26971) :=
sorry

end Cameron_list_count_l420_420779


namespace lines_concurrent_l420_420771

structure Circle (α : Type) :=
(center : α) 
(radius : ℝ)

structure Line (α : Type) :=
(point1 : α) 
(point2 : α)

structure Point (α : Type) :=
(coord : α)

variable {α : Type} [LinearOrderedField α]

noncomputable def midpoint (P Q : Point α) : Point α :=
Point.mk ((P.coord + Q.coord) / 2)

-- Definitions of given structures
variables (Γ₁ Γ₂ : Circle α)
variables (M N A B C D X P Q K L : Point α)
variables (l AX DX BM CM XK ML PQ : Line α)

-- Conditions
axiom circles_intersect : Γ₁.center = M.coord ∧ Γ₂.center = N.coord
axiom line_intersections : (l.point1 = A.coord ∧ l.point2 = C.coord) ∧ (l.point1 = B.coord ∧ l.point2 = D.coord)
axiom X_on_MN : X.coord = M.coord ∧ X.coord = N.coord
axiom M_between_XN : M.coord < X.coord ∧ X.coord < N.coord
axiom lines_intersections : AX.point2 = P.coord ∧ BM.point1 = P.coord ∧ DX.point2 = Q.coord ∧ CM.point1 = Q.coord 
axiom midpoints : K.coord = midpoint A.coord D.coord ∧ L.coord = midpoint B.coord C.coord 

-- Theorem
theorem lines_concurrent : ∃ R : Point α, (XK.point2 = R.coord ∧ ML.point2 = R.coord ∧ PQ.point2 = R.coord) :=
sorry

end lines_concurrent_l420_420771


namespace forgetful_scientist_has_two_packages_at_end_l420_420512

noncomputable def forgetful_scientist_probability : ℝ :=
  let initial_pills := 10
  let total_days := 365
  let average_events_per_week := 7 -- Assuming consumption happens several times a week
  let transitions_per_day := average_events_per_week / 7
  let states := [2, 3, ..., 11] -- All possible pill states
  let probability_transition (state : ℕ) : ℝ := 1 / state
  let geometric_series_sum (n : ℕ) :=
    (1 - (1 / 2^n)) / (1 - (1 / 2))
  let final_sum := geometric_series_sum states.length
  probability_transition initial_pills * final_sum / initial_pills

theorem forgetful_scientist_has_two_packages_at_end :
  forgetful_scientist_probability ≈ 0.1998 := sorry

end forgetful_scientist_has_two_packages_at_end_l420_420512


namespace true_propositions_l420_420463

-- Definitions encoded directly from the conditions in the math problem.
def proposition_p1 (a b : ℝ) : Prop := 
  let z1 := a + b * complex.I
  let z2 := -a + b * complex.I
  z1.im = z2.im ∧ z1.re = -z2.re

def proposition_p2 : Prop :=
  ∃ z : ℂ, (1 - complex.I) * z = 1 + complex.I ∧ z.im ≠ 0 ∧ z.re = 0

def proposition_p3 (z1 z2 : ℂ) : Prop :=
  (z1 * z2).im = 0 → z2 = complex.conj z1

def proposition_p4 : Prop :=
  ∃ z : ℂ, z^2 + 1 = 0 ∧ (z = complex.I ∨ z = -complex.I)

-- Final proof statement verifying the true propositions are exactly p2 and p4.
theorem true_propositions : proposition_p2 ∧ proposition_p4 := by
  sorry

end true_propositions_l420_420463


namespace expression_equality_l420_420345

theorem expression_equality : 
  (∀ (x : ℝ) (a k n : ℝ), (3 * x + 2) * (2 * x - 3) = a * x^2 + k * x + n → a = 6 ∧ k = -5 ∧ n = -6) → 
  ∀ (a k n : ℝ), a = 6 → k = -5 → n = -6 → a - n + k = 7 :=
by
  intro h
  intros a k n ha hk hn
  rw [ha, hk, hn]
  norm_num

end expression_equality_l420_420345


namespace teacher_pencil_distribution_l420_420648

theorem teacher_pencil_distribution (num_children : ℕ) (pencils_per_dozen : ℕ) (dozens_per_child : ℕ) : 
  num_children = 46 → pencils_per_dozen = 12 → dozens_per_child = 4 →
  (num_children * dozens_per_child * pencils_per_dozen) = 2208 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end teacher_pencil_distribution_l420_420648


namespace parallelepiped_diagonals_l420_420087

theorem parallelepiped_diagonals (x y z : ℝ) 
  (h1 : sqrt (x^2 + y^2 + z^2) = 15)
  (h2 : sqrt (x^2 + y^2 + z^2) = 17)
  (h3 : sqrt (x^2 + y^2 + z^2) = 21)
  (h4 : sqrt (x^2 + y^2 + z^2) = 23) :
  x^2 + y^2 + z^2 = 371 := 
sorry

end parallelepiped_diagonals_l420_420087


namespace part1_part2_part3_l420_420847

namespace MathProof

variables {a b m n : ℕ}
variables (an bn : ℕ → ℕ)
variables (c : ℕ)
variables (d : ℕ → ℕ)
variables (h1 : ∀ n, an (n+1) = a + n * b)
variables (h2 : ∀ n, bn n = b * a ^ (n-1))
variables (h3: a * b < 3 * b)
variables (a b m n ≥ 1)

-- Part I
theorem part1 : a = 2 := sorry

-- Part II
theorem part2 : a = 2 → c = 3 * 2^(n-1) := sorry

-- Part III
theorem part3 :
  (∀ m, d m = (a + m * (b - 1)) / (2 * m)) →
  (∑ k in range n, k / ((1 + d 1) * (1 + d 2) * ... * (1 + d k))) < 2.
:= sorry

end MathProof

end part1_part2_part3_l420_420847


namespace max_difference_of_primes_l420_420718

theorem max_difference_of_primes (a b c : ℕ) (h1 : Nat.Prime a) (h2 : Nat.Prime b) (h3 : Nat.Prime c)
    (h4 : Nat.Prime (a + b - c)) (h5 : Nat.Prime (a + c - b)) (h6 : Nat.Prime (b + c - a)) (h7 : Nat.Prime (a + b + c))
    (h8 : pair := ∃ (x y : ℕ), x ∈ {a, b, c} ∧ y ∈ {a, b, c} ∧ x ≠ y ∧ x + y = 800)
    (h_distinct : {a, b, c, a+b-c, a+c-b, b+c-a, a+b+c}.to_finset.card = 7) :
    ∃ d, d = (max (max (max (max (max (max a b) c) (a + b - c)) (a + c - b)) (b + c - a)) (a + b + c)) - 
                (min (min (min (min (min (min a b) c) (a + b - c)) (a + c - b)) (b + c - a)) (a + b + c)) := by
  sorry

end max_difference_of_primes_l420_420718


namespace a_n_expression_T_n_expression_l420_420002

-- Definitions based on conditions
def Sn (n : ℕ) (a : ℕ → ℝ) : ℝ := (∑ k in finset.range (n + 1), a k)

-- Given conditions as assumptions
axiom S3_eq : ∀ (a : ℕ → ℝ), Sn 3 a = 7 / 2
axiom S6_eq : ∀ (a : ℕ → ℝ), Sn 6 a = 63 / 16

-- Defined sequence a_n
def a_n (n : ℕ) : ℝ := (1 / 2)^(n - 2)

-- Defined sequence b_n
def b_n (n : ℕ) : ℝ := 2^(n - 2) + n

-- Sum of the b_n sequence
def T_n (n : ℕ) : ℝ := (∑ k in finset.range (n + 1), b_n k)

-- Proof problem for a_n expression
theorem a_n_expression :
  ∀ (n : ℕ), a_n n = (1 / 2)^(n - 2)
:= sorry

-- Proof problem for T_n expression
theorem T_n_expression :
  ∀ (n : ℕ), T_n n = (2^n + n^2 + n - 1) / 2
:= sorry

end a_n_expression_T_n_expression_l420_420002


namespace pyramid_area_l420_420251

theorem pyramid_area :
  ∀ (a b : ℝ), a = 8 → b = 7 → 4 * (1/2 * a * sqrt (b^2 - (a/2)^2)) = 16 * sqrt 33 :=
by
  intros a b ha hb
  rw [ha, hb]
  have h1 : a / 2 = 4 := by norm_num [ha]
  have h2 : b^2 - (a / 2)^2 = 33 :=
    by
      calc
        b^2 - (a / 2)^2 = 49 - 16 := by norm_num [hb]
        ... = 33 := by norm_num
  rw [h1, h2, sqrt 33, mul_one, mul_one, half_mul, mul_comm (1/2) a, ←mul_assoc, mul_comm 4 4]
  norm_num
  sorry

end pyramid_area_l420_420251


namespace series_sum_lt_one_l420_420789

/--
  The series sum from 1 to 2012 of (n / (n+1)!) is less than 1.
-/
theorem series_sum_lt_one :
  (∑ n in Finset.range (2012+1), (n / (n+1)! : ℝ)) < 1 := 
  sorry

end series_sum_lt_one_l420_420789


namespace largest_gold_coins_l420_420312

noncomputable def max_gold_coins (n : ℕ) : ℕ :=
  if h : ∃ k : ℕ, n = 13 * k + 3 ∧ n < 150 then
    n
  else 0

theorem largest_gold_coins : max_gold_coins 146 = 146 :=
by
  sorry

end largest_gold_coins_l420_420312


namespace problem_l420_420887

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := m * Real.sin x + Real.cos x

theorem problem
  (m : ℝ)
  (h1 : f (π / 2) m = 1)
  (h2 : ∀ A : ℝ, 
    A < π / 2 → 
    A > 0 →
    ∃ (AC BC : ℝ), 
      (f (π / 12) 1 = √2 * Real.sin A) ∧ 
      (1 / 2) * 2 * AC * Real.sin A = (3 * √3) / 2 ∧ 
      BC^2 = AC^2 + 4 - 2 * 2 * AC * Real.cos A) :
  (f (x) 1) = √2 * Real.sin (x + π / 4) ∧
  (∀ x : ℝ, f (x) 1 ≤ √2) ∧ (∀ x : ℝ, f (x) 1 ≥ -√2) ∧
  (∃ (AC BC : ℝ), AC = 3 ∧ BC = √7) :=
by sorry

end problem_l420_420887


namespace log_negative_l420_420492

open Real

theorem log_negative (a : ℝ) (h : a > 0) : log (-a) = log a := sorry

end log_negative_l420_420492


namespace determinant_expression_l420_420105

theorem determinant_expression (a b c d p q r : ℝ)
  (h1: (∃ x: ℝ, x^4 + p*x^2 + q*x + r = 0) → (x = a ∨ x = b ∨ x = c ∨ x = d))
  (h2: a*b + a*c + a*d + b*c + b*d + c*d = p)
  (h3: a*b*c + a*b*d + a*c*d + b*c*d = q)
  (h4: a*b*c*d = -r):
  (Matrix.det ![![1 + a, 1, 1, 1], ![1, 1 + b, 1, 1], ![1, 1, 1 + c, 1], ![1, 1, 1, 1 + d]]) 
  = r + q + p := 
sorry

end determinant_expression_l420_420105


namespace committee_member_count_l420_420655

theorem committee_member_count (n : ℕ) (M : ℕ) (Q : ℚ) 
  (h₁ : M = 6) 
  (h₂ : 2 * n = M) 
  (h₃ : Q = 0.4) 
  (h₄ : Q = (n - 1) / (M - 1)) : 
  n = 3 :=
by
  sorry

end committee_member_count_l420_420655


namespace monotonic_intervals_extreme_values_l420_420477

noncomputable def f (x : ℝ) : ℝ := 3 * x^3 - 9 * x + 5

theorem monotonic_intervals :
  (∀ x, (x < -1 ∨ x > 1) → f'(x) > 0) ∧
  (∀ x, (-1 < x ∧ x < 1) → f'(x) < 0) := sorry

theorem extreme_values :
  f (-1) = 11 ∧ f (1) = -1 := sorry

end monotonic_intervals_extreme_values_l420_420477


namespace sum_of_angles_eq_990_l420_420411

def is_polar_form (z : ℂ) (r θ : ℝ) : Prop :=
  z = r * complex.exp (θ * complex.I)

def is_root (z : ℂ) : Prop :=
  z^6 = 64 * complex.I

def valid_angle (θ : ℝ) : Prop :=
  0 ≤ θ ∧ θ < 360

def valid_root (z : ℂ) : Prop :=
  ∃ (r θ : ℝ), r > 0 ∧ is_polar_form z r θ ∧ valid_angle θ

theorem sum_of_angles_eq_990 :
  ∀ {z : ℂ}, is_root z → valid_root z →
  ∑ (k: Fin 6), 
  let θ_k := (90 + 360 * k) / 6 in 
  θ_k = 990 :=
sorry

end sum_of_angles_eq_990_l420_420411


namespace pyramid_four_triangular_faces_area_l420_420258

theorem pyramid_four_triangular_faces_area 
  (base_edge : ℝ) (lateral_edge : ℝ) 
  (h_base : base_edge = 8)
  (h_lateral : lateral_edge = 7) :
  let h := Real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)
  let triangle_area := (1 / 2) * base_edge * h
  let total_area := 4 * triangle_area
  total_area = 16 * Real.sqrt 33 :=
by
  -- Definitions to introduce local values
  let half_base := base_edge / 2
  let h := Real.sqrt (lateral_edge ^ 2 - half_base ^ 2)
  let triangle_area := (1 / 2) * base_edge * h
  let total_area := 4 * triangle_area
  -- Assertion to compare calculated total area with given correct answer
  have h_eq : h = Real.sqrt 33 := by sorry
  have triangle_area_eq : triangle_area = 4 * Real.sqrt 33 := by sorry
  have total_area_eq : total_area = 16 * Real.sqrt 33 := by sorry
  exact total_area_eq

end pyramid_four_triangular_faces_area_l420_420258


namespace pyramid_triangular_face_area_l420_420216

theorem pyramid_triangular_face_area 
  (base : ℝ) (lateral : ℝ)
  (h_base : base = 8) (h_lateral : lateral = 7) :
  let height := sqrt (lateral^2 - (base / 2)^2) in
  4 * (1 / 2 * base * height) = 16 * sqrt 33 := 
by
  sorry

end pyramid_triangular_face_area_l420_420216


namespace packs_sold_to_uncle_is_correct_l420_420436

-- Define the conditions and constants
def total_packs_needed := 50
def packs_sold_to_grandmother := 12
def packs_sold_to_neighbor := 5
def packs_left_to_sell := 26

-- Calculate total packs sold so far
def total_packs_sold := total_packs_needed - packs_left_to_sell

-- Calculate total packs sold to grandmother and neighbor
def packs_sold_to_grandmother_and_neighbor := packs_sold_to_grandmother + packs_sold_to_neighbor

-- The pack sold to uncle
def packs_sold_to_uncle := total_packs_sold - packs_sold_to_grandmother_and_neighbor

-- Prove the packs sold to uncle
theorem packs_sold_to_uncle_is_correct : packs_sold_to_uncle = 7 := by
  -- The proof steps are omitted
  sorry

end packs_sold_to_uncle_is_correct_l420_420436


namespace pyramid_area_l420_420250

theorem pyramid_area :
  ∀ (a b : ℝ), a = 8 → b = 7 → 4 * (1/2 * a * sqrt (b^2 - (a/2)^2)) = 16 * sqrt 33 :=
by
  intros a b ha hb
  rw [ha, hb]
  have h1 : a / 2 = 4 := by norm_num [ha]
  have h2 : b^2 - (a / 2)^2 = 33 :=
    by
      calc
        b^2 - (a / 2)^2 = 49 - 16 := by norm_num [hb]
        ... = 33 := by norm_num
  rw [h1, h2, sqrt 33, mul_one, mul_one, half_mul, mul_comm (1/2) a, ←mul_assoc, mul_comm 4 4]
  norm_num
  sorry

end pyramid_area_l420_420250


namespace seating_arrangement_l420_420515

def num_ways_seated (total_passengers : ℕ) (window_seats : ℕ) : ℕ :=
  window_seats * (total_passengers - 1) * (total_passengers - 2) * (total_passengers - 3)

theorem seating_arrangement (passengers_seats taxi_window_seats : ℕ)
  (h1 : passengers_seats = 4) (h2 : taxi_window_seats = 2) :
  num_ways_seated passengers_seats taxi_window_seats = 12 :=
by
  -- proof will go here
  sorry

end seating_arrangement_l420_420515


namespace axis_of_symmetry_parabola_l420_420606

theorem axis_of_symmetry_parabola (a b : ℝ) (h₁ : a = -3) (h₂ : b = 6) :
  -b / (2 * a) = 1 :=
by
  sorry

end axis_of_symmetry_parabola_l420_420606


namespace number_of_members_l420_420731

theorem number_of_members (n : ℕ) (h : n^2 = 5929) : n = 77 :=
sorry

end number_of_members_l420_420731


namespace pyramid_area_l420_420275

theorem pyramid_area (base_edge lateral_edge : ℝ) (H_base_edge : base_edge = 8) (H_lateral_edge : lateral_edge = 7) :
  4 * (1 / 2 * base_edge * real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 16 * real.sqrt 33 :=
by
  sorry

end pyramid_area_l420_420275


namespace largest_number_of_gold_coins_l420_420316

theorem largest_number_of_gold_coins (n : ℕ) :
  (∃ k : ℕ, n = 13 * k + 3 ∧ n < 150) → n ≤ 146 :=
by
  sorry

end largest_number_of_gold_coins_l420_420316


namespace find_p_of_f_find_m_range_of_f_l420_420988

noncomputable def f (a x : ℝ) : ℝ := log (a + x) / log 3 + log (2 - x) / log 3

-- given conditions
axiom even_function (a : ℝ) : ∀ x : ℝ, f a (-x) = f a x

-- the proof problem translated into Lean
theorem find_p_of_f (a p : ℝ) (h : f 2 p = 1) : p = 1 ∨ p = -1 :=
sorry

theorem find_m_range_of_f (a m : ℝ) : (∃ m : ℝ, f 2 (2 * m - 1) < f 2 m) → m ∈ Ioo (-(1 / 2)) (1 / 3) ∨ m ∈ Ioo 1 (3 / 2) :=
sorry

end find_p_of_f_find_m_range_of_f_l420_420988


namespace emily_cleaning_time_l420_420110

noncomputable def total_time : ℝ := 8 -- total time in hours
noncomputable def lilly_fiona_time : ℝ := 1/4 * total_time -- Lilly and Fiona's combined time in hours
noncomputable def jack_time : ℝ := 1/3 * total_time -- Jack's time in hours
noncomputable def emily_time : ℝ := total_time - lilly_fiona_time - jack_time -- Emily's time in hours
noncomputable def emily_time_minutes : ℝ := emily_time * 60 -- Emily's time in minutes

theorem emily_cleaning_time :
  emily_time_minutes = 200 := by
  sorry

end emily_cleaning_time_l420_420110


namespace pond_painted_area_l420_420706

-- Definitions of the dimensions of the pond
def length : ℝ := 18
def width : ℝ := 10
def height : ℝ := 2

-- Definition of total painted area
def total_painted_area : ℝ :=
  let A_bottom := length * width
  let A_long_side := length * height
  let Total_A_long_sides := 2 * A_long_side
  let A_short_side := width * height
  let Total_A_short_sides := 2 * A_short_side
  A_bottom + Total_A_long_sides + Total_A_short_sides

-- The statement to prove 
theorem pond_painted_area : total_painted_area = 292 := by
  sorry

end pond_painted_area_l420_420706


namespace pyramid_four_triangular_faces_area_l420_420260

theorem pyramid_four_triangular_faces_area 
  (base_edge : ℝ) (lateral_edge : ℝ) 
  (h_base : base_edge = 8)
  (h_lateral : lateral_edge = 7) :
  let h := Real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)
  let triangle_area := (1 / 2) * base_edge * h
  let total_area := 4 * triangle_area
  total_area = 16 * Real.sqrt 33 :=
by
  -- Definitions to introduce local values
  let half_base := base_edge / 2
  let h := Real.sqrt (lateral_edge ^ 2 - half_base ^ 2)
  let triangle_area := (1 / 2) * base_edge * h
  let total_area := 4 * triangle_area
  -- Assertion to compare calculated total area with given correct answer
  have h_eq : h = Real.sqrt 33 := by sorry
  have triangle_area_eq : triangle_area = 4 * Real.sqrt 33 := by sorry
  have total_area_eq : total_area = 16 * Real.sqrt 33 := by sorry
  exact total_area_eq

end pyramid_four_triangular_faces_area_l420_420260


namespace samia_walk_distance_l420_420580

noncomputable def total_distance := (54 * 20 / 60) + ((54 * 3) / (60 * 3))

noncomputable def walked_distance (d : ℝ) := d * 3 / 4 / 20 + d / 4 / 3

theorem samia_walk_distance (d : ℝ) : 
  (9 / 10) = (29 * d) / 240 → 
  floor ((d / 4) * 10) / 10 = 1.9 :=
by
  intro h
  sorry

end samia_walk_distance_l420_420580


namespace pyramid_four_triangular_faces_area_l420_420252

theorem pyramid_four_triangular_faces_area 
  (base_edge : ℝ) (lateral_edge : ℝ) 
  (h_base : base_edge = 8)
  (h_lateral : lateral_edge = 7) :
  let h := Real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)
  let triangle_area := (1 / 2) * base_edge * h
  let total_area := 4 * triangle_area
  total_area = 16 * Real.sqrt 33 :=
by
  -- Definitions to introduce local values
  let half_base := base_edge / 2
  let h := Real.sqrt (lateral_edge ^ 2 - half_base ^ 2)
  let triangle_area := (1 / 2) * base_edge * h
  let total_area := 4 * triangle_area
  -- Assertion to compare calculated total area with given correct answer
  have h_eq : h = Real.sqrt 33 := by sorry
  have triangle_area_eq : triangle_area = 4 * Real.sqrt 33 := by sorry
  have total_area_eq : total_area = 16 * Real.sqrt 33 := by sorry
  exact total_area_eq

end pyramid_four_triangular_faces_area_l420_420252


namespace lunch_packing_days_l420_420930

def school_days (A B C : ℕ) := true

def packs_lunch_Aliyah (x : ℕ) : ℕ := (3 * x) / 5
def packs_lunch_Becky (y : ℕ) (P_A : ℕ) : ℕ := (3 * y) / 20
def packs_lunch_Charlie (z : ℕ) (P_B : ℕ) : ℕ := (3 * z) / 10
def packs_lunch_Dana (x : ℕ) : ℕ := x / 3

theorem lunch_packing_days (x y z : ℕ) :
  ∀ (P_A : ℕ) (P_B : ℕ) (P_C : ℕ) (P_D : ℕ),
  P_A = packs_lunch_Aliyah x →
  P_B = packs_lunch_Becky y P_A →
  P_C = packs_lunch_Charlie z P_B →
  P_D = packs_lunch_Dana x →
  ((P_A = (3 * x) / 5) ∧ (P_B = (3 * y) / 20) ∧ (P_C = (3 * z) / 10) ∧ (P_D = x / 3)) :=
by
  intros P_A P_B P_C P_D hA hB hC hD
  rw [hA, hB, hC, hD]
  simp
  done

end lunch_packing_days_l420_420930


namespace set_intersection_l420_420562

-- Define sets M and N
def M : Set ℕ := {0, 1, 2}
def N : Set ℝ := {x : ℝ | x^2 - 3*x + 2 ≤ 0}

-- Translate the solution's conclusion into the Lean statement
theorem set_intersection : (M ∩ N) = {1, 2} := 
by 
  sorry

end set_intersection_l420_420562


namespace cookie_batches_needed_l420_420112

theorem cookie_batches_needed :
  let 
    cookies_per_student := 12
    total_students := 36
    total_cookies_needed := total_students * cookies_per_student
    chocolate_chip_batches := 3
    oatmeal_raisin_batches := 2
    cookies_per_chocolate_chip_batch := 3 * 12
    cookies_per_oatmeal_raisin_batch := 4 * 12
    cookies_made := (chocolate_chip_batches * cookies_per_chocolate_chip_batch) + (oatmeal_raisin_batches * cookies_per_oatmeal_raisin_batch)
    remaining_cookies_needed := total_cookies_needed - cookies_made
    cookies_per_peanut_butter_batch := 5 * 12
    batches_needed := (remaining_cookies_needed + cookies_per_peanut_butter_batch - 1) / cookies_per_peanut_butter_batch
  in batches_needed = 4 :=
by 
  sorry

end cookie_batches_needed_l420_420112


namespace trig_identity_proof_l420_420708

theorem trig_identity_proof (α : ℝ) :
  1 - cos ( (3/2) * real.pi - 3 * α ) - sin( (3/2) * α )^2 + cos( (3/2) * α )^2
  = 2 * real.sqrt 2 * cos ( (3/2) * α ) * sin ( (3/2) * α + real.pi / 4) := by
  sorry

end trig_identity_proof_l420_420708


namespace triangle_properties_l420_420850

noncomputable def centroid (A B C : Point) : Point := sorry
noncomputable def equilateral_outward_center (A B C : Point) : Triangle := sorry
noncomputable def equilateral_inward_center (A B C : Point) : Triangle := sorry
noncomputable def area (Δ : Triangle) : ℝ := sorry

theorem triangle_properties (A B C : Point) :
  let Δ := equilateral_outward_center A B C
  let δ := equilateral_inward_center A B C
  let G := centroid A B C
  (Δ.equilateral ∧ δ.equilateral) ∧
  (Δ.center = G ∧ δ.center = G) ∧
  (area Δ - area δ = area (triangle A B C)) :=
begin
  sorry,
end

end triangle_properties_l420_420850


namespace z_in_second_quadrant_l420_420859

-- Define the complex number z
def z : ℂ := (↑(0 + 1 * complex.I)) / (1 - 2 * complex.I)

-- State the theorem that z is in the second quadrant
theorem z_in_second_quadrant : z.re < 0 ∧ z.im > 0 :=
  sorry

end z_in_second_quadrant_l420_420859


namespace series_sum_lt_one_l420_420790

/--
  The series sum from 1 to 2012 of (n / (n+1)!) is less than 1.
-/
theorem series_sum_lt_one :
  (∑ n in Finset.range (2012+1), (n / (n+1)! : ℝ)) < 1 := 
  sorry

end series_sum_lt_one_l420_420790


namespace trader_gain_percentage_l420_420393

-- Definitions based on conditions
def cost_of_one_pen (C : ℝ) := C
def cost_of_100_pens (C : ℝ) : ℝ := 100 * C
def gain (C : ℝ) : ℝ := 30 * C
def gain_percentage (C : ℝ) : ℝ := (gain C / cost_of_100_pens C) * 100

-- Theorem stating the problem and desired proof
theorem trader_gain_percentage (C : ℝ) (h : C > 0) :
  gain_percentage C = 30 :=
by 
  sorry

end trader_gain_percentage_l420_420393


namespace union_A_B_l420_420895

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 4}
def B : Set ℝ := {x | x > 2}

theorem union_A_B :
  A ∪ B = {x : ℝ | 1 ≤ x} := sorry

end union_A_B_l420_420895


namespace max_area_difference_l420_420672

theorem max_area_difference (l w l' w' : ℕ) 
  (h1 : 2 * l + 2 * w = 160) 
  (h2 : 2 * l' + 2 * w' = 160) : 
  abs ((l * w) - (l' * w')) ≤ 1600 := 
by 
  sorry

end max_area_difference_l420_420672


namespace parallelogram_coordinate_sum_l420_420180

theorem parallelogram_coordinate_sum:
  (A B D : ℝ × ℝ) -- Define the vertices of the parallelogram
  (hA : A = (-1, 1))
  (hB : B = (3, 5))
  (hD : D = (11, -3))
  (area : ℝ)
  (harea : area = 48) :
  ∃ (C : ℝ × ℝ), 
    let x := C.1 in let y := C.2 in x + y = 0 :=
by
  sorry

end parallelogram_coordinate_sum_l420_420180


namespace no_x_squared_term_l420_420915

theorem no_x_squared_term :
  ∀ (a : ℝ), (a = -3) → 
  (∀ x : ℝ, (x^2 + a * x + 5) * (-2 * x) - 6 * x^2 = -2 * x^3 - 10 * x) :=
by
  intro a ha
  intro x
  rw ha
  ring
  sorry

end no_x_squared_term_l420_420915


namespace find_m_n_sum_l420_420506

theorem find_m_n_sum (m n : ℕ) (hm : m > 1) (hn : n > 1) 
  (h : 2005^2 + m^2 = 2004^2 + n^2) : 
  m + n = 211 :=
sorry

end find_m_n_sum_l420_420506


namespace stratified_sampling_elderly_l420_420802

theorem stratified_sampling_elderly (total_infected : ℕ) (num_young : ℕ) (num_elderly : ℕ) 
  (num_children : ℕ) (num_selected : ℕ) (proportion_elderly : α) : 
  (total_infected = 100) →
  (num_young = 10) →
  (num_elderly = 60) →
  (num_children = 30) →
  (num_selected = 20) →
  (proportion_elderly = (num_elderly : α) / (total_infected : α)) →
  (num_elderly_selected = (num_selected : α) * proportion_elderly) →
  (num_elderly_selected = 12) := 
by
  intros
  sorry

end stratified_sampling_elderly_l420_420802


namespace z_sum_of_squares_eq_101_l420_420978

open Complex

noncomputable def z_distances_sum_of_squares (z : ℂ) (h : abs (z - (3 + -3 * I)) = 3) : ℝ :=
  abs (z - (1 + 1 * I)) ^ 2 + abs (z - (5 - 5 * I)) ^ 2

theorem z_sum_of_squares_eq_101 (z : ℂ) (h : abs (z - (3 + -3 * I)) = 3) : 
  z_distances_sum_of_squares z h = 101 :=
by
  sorry

end z_sum_of_squares_eq_101_l420_420978


namespace small_supermarkets_sample_count_l420_420059

def large := 300
def medium := 600
def small := 2100
def sample_size := 100
def total := large + medium + small

theorem small_supermarkets_sample_count :
  small * (sample_size / total) = 70 := by
  sorry

end small_supermarkets_sample_count_l420_420059


namespace constant_term_in_expansion_l420_420611

def polynomial := (x^2 + (1 / x^2) - 2) ^ 3

theorem constant_term_in_expansion : ∃ c : ℤ, 
  (∀ (x : ℂ), polynomial.eval x ((x ^ 2 + 1 / x ^ 2 - 2) ^ 3) = c) ∧ c = -20 := 
by
  sorry

end constant_term_in_expansion_l420_420611


namespace carson_speed_is_8_mph_l420_420956

-- Assume Jerry's one-way trip takes 15 minutes
def jerry_one_way_time : ℝ := 15 / 60 -- time in hours
def jerry_distance : ℝ := 4 -- distance to school in miles

-- Jerry's round trip distance is twice the distance to school
def jerry_round_trip_distance : ℝ := 2 * jerry_distance

-- Jerry's round trip time is twice the one-way time
def jerry_round_trip_time : ℝ := 2 * jerry_one_way_time

-- Jerry's speed in miles per hour
def jerry_speed : ℝ := jerry_distance / jerry_one_way_time

-- Assume Carson's time to run to the school is equivalent to Jerry's round trip time
def carson_one_way_time : ℝ := jerry_round_trip_time

-- Carson's running speed in miles per hour
def carson_distance : ℝ := jerry_distance
def carson_speed : ℝ := carson_distance / carson_one_way_time

-- The theorem to prove Carson's speed
theorem carson_speed_is_8_mph : carson_speed = 8 := 
by
  -- compute and verify the statement here
  sorry

end carson_speed_is_8_mph_l420_420956


namespace circle_to_semicircle_ratio_l420_420728

theorem circle_to_semicircle_ratio
    (R r : ℝ) -- radii of the semicircle and inscribed circle
    (O P A : EuclideanSpace ℝ (Fin 2)) -- centers of the semicircle and inscribed circle, and point on semicircle's circumference
    (h1 : dist P A = dist P O)
    (h2 : dist O A = R)
    (h3 : dist P O = R - r)
    (h4 : dist A P = sqrt ((R/2)^2 + r^2)) :
    r / R = 3 / 8 :=
sorry

end circle_to_semicircle_ratio_l420_420728


namespace gallery_solution_l420_420767

def gallery_problem (A : ℕ) : Prop :=
  let displayed := A / 3 in
  let sculptures_on_display := displayed / 6 in
  let not_on_display := A - displayed in
  let paintings_not_on_display := not_on_display / 3 in
  let sculptures_not_on_display := not_on_display - paintings_not_on_display in
  sculptures_not_on_display = 400 ->
  A = 900

theorem gallery_solution : ∃ (A : ℕ), gallery_problem A :=
  by
    use 900
    sorry  -- Proof omitted

end gallery_solution_l420_420767


namespace tiles_per_row_l420_420599

theorem tiles_per_row (area : ℝ) (tile_length : ℝ) (h1 : area = 256) (h2 : tile_length = 2/3) : 
  (16 * 12) / (8) = 24 :=
by {
  sorry
}

end tiles_per_row_l420_420599


namespace length_of_AD_in_parallelogram_l420_420935

theorem length_of_AD_in_parallelogram
  (x : ℝ)
  (AB BC CD : ℝ)
  (AB_eq : AB = x + 3)
  (BC_eq : BC = x - 4)
  (CD_eq : CD = 16)
  (parallelogram_ABCD : AB = CD ∧ AD = BC) :
  AD = 9 := by
sorry

end length_of_AD_in_parallelogram_l420_420935


namespace greatest_area_difference_l420_420662

theorem greatest_area_difference 
  (l w l' w' : ℕ) 
  (h1 : 2 * l + 2 * w = 160) 
  (h2 : 2 * l' + 2 * w' = 160) :
  abs ((l * w) - (l' * w')) ≤ 1521 := 
sorry

end greatest_area_difference_l420_420662


namespace martha_total_points_l420_420997

-- Define the costs and points
def cost_beef := 11 * 3
def cost_fruits_vegetables := 4 * 8
def cost_spices := 6 * 3
def cost_other := 37

def total_spending := cost_beef + cost_fruits_vegetables + cost_spices + cost_other

def points_per_dollar := 50 / 10
def base_points := total_spending * points_per_dollar
def bonus_points := if total_spending > 100 then 250 else 0

def total_points := base_points + bonus_points

-- The theorem to prove the question == answer given the conditions
theorem martha_total_points :
  total_points = 850 :=
by
  sorry

end martha_total_points_l420_420997


namespace sum_not_even_l420_420810

theorem sum_not_even (x y : ℤ) (h : 7 * x + 5 * y = 11111) : ¬ even (x + y) :=
by
  sorry

end sum_not_even_l420_420810


namespace pyramid_face_area_total_l420_420235

theorem pyramid_face_area_total 
  (base_edge : ℝ) (lateral_edge : ℝ) 
  (h_base_edge : base_edge = 8) 
  (h_lateral_edge : lateral_edge = 7) : 
  4 * (1 / 2 * base_edge * real.sqrt (lateral_edge^2 - (base_edge / 2)^2)) = 16 * real.sqrt 33 :=
by
  sorry

end pyramid_face_area_total_l420_420235


namespace pyramid_area_l420_420274

theorem pyramid_area (base_edge lateral_edge : ℝ) (H_base_edge : base_edge = 8) (H_lateral_edge : lateral_edge = 7) :
  4 * (1 / 2 * base_edge * real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 16 * real.sqrt 33 :=
by
  sorry

end pyramid_area_l420_420274


namespace opposite_z_is_E_l420_420616

noncomputable def cube_faces := ["A", "B", "C", "D", "E", "z"]

def opposite_face (net : List String) (face : String) : String :=
  if face = "z" then "E" else sorry  -- generalize this function as needed

theorem opposite_z_is_E :
  opposite_face cube_faces "z" = "E" :=
by
  sorry

end opposite_z_is_E_l420_420616


namespace pyramid_area_l420_420282

theorem pyramid_area (base_edge lateral_edge : ℝ) (H_base_edge : base_edge = 8) (H_lateral_edge : lateral_edge = 7) :
  4 * (1 / 2 * base_edge * real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 16 * real.sqrt 33 :=
by
  sorry

end pyramid_area_l420_420282


namespace Haley_initial_trees_l420_420038

theorem Haley_initial_trees (T : ℕ) (h1 : T - 4 ≥ 0) (h2 : (T - 4) + 5 = 10): T = 9 :=
by
  -- proof goes here
  sorry

end Haley_initial_trees_l420_420038


namespace petya_equals_vasya_l420_420123

def petya_word_count (m : ℕ) : ℕ :=
  sorry -- The actual count of m-letter words with equal T's and O's using letters T, O, W, and N.

def vasya_word_count (m : ℕ) : ℕ :=
  sorry -- The actual count of 2m-letter words with equal T's and O's using only letters T and O.

theorem petya_equals_vasya (m : ℕ) : petya_word_count m = vasya_word_count m :=
  sorry

end petya_equals_vasya_l420_420123


namespace bubble_pass_probability_l420_420402

noncomputable def probability_s30_at_s40 (s : Fin 50 → ℝ) : ℚ :=
  if h_distinct : Function.Injective s ∧
    s.card = 50 then
    1 / 1640
  else
    0

theorem bubble_pass_probability (s : Fin 50 → ℝ) (h_distinct : Function.Injective s)
  (h_ordered : ∀ i j : Fin 50, i < j → s i ≠ s j) :
  probability_s30_at_s40 s = 1 / 1640 :=
by
  sorry

end bubble_pass_probability_l420_420402


namespace alice_bob_task_l420_420380

theorem alice_bob_task (t : ℝ) (h₁ : 1/4 + 1/6 = 5/12) (h₂ : t - 1/2 ≠ 0) :
    (5/12) * (t - 1/2) = 1 :=
sorry

end alice_bob_task_l420_420380


namespace car_travel_distance_l420_420040

-- Define the conditions given in the problem
def car_speed (train_speed : ℝ) : ℝ := (5 / 8) * train_speed
def travel_time_minutes : ℝ := 45
def minutes_to_hours (minutes : ℝ) : ℝ := minutes / 60

-- Given conditions
def train_speed : ℝ := 100 -- miles per hour

-- The required distance to prove
def car_distance (speed : ℝ) (time : ℝ) : ℝ := speed * minutes_to_hours(time)

-- Prove the expected distance
theorem car_travel_distance :
  car_distance (car_speed train_speed) travel_time_minutes = 46.875 :=
by
  sorry

end car_travel_distance_l420_420040


namespace distance_focus_to_asymptote_of_hyperbola_l420_420814

open Real

noncomputable def distance_from_focus_to_asymptote_of_hyperbola : ℝ :=
  let a := 2
  let b := 1
  let c := sqrt (a^2 + b^2)
  let foci1 := (sqrt (a^2 + b^2), 0)
  let foci2 := (-sqrt (a^2 + b^2), 0)
  let asymptote_slope := a / b
  let distance_formula := (|abs (sqrt 5)|) / (sqrt (1 + asymptote_slope^2))
  distance_formula

theorem distance_focus_to_asymptote_of_hyperbola :
  distance_from_focus_to_asymptote_of_hyperbola = 1 :=
sorry

end distance_focus_to_asymptote_of_hyperbola_l420_420814


namespace max_k_value_l420_420843

noncomputable def max_k : ℝ := Real.sqrt (9 + 6 * Real.sqrt 3)

theorem max_k_value
  (x y z : ℝ)
  (hx : 0 ≤ x)
  (hy : 0 ≤ y)
  (hz : 0 ≤ z) :
  x^3 + y^3 + z^3 - 3 * x * y * z ≥ max_k * Real.abs ((x - y) * (y - z) * (z - x)) :=
sorry

end max_k_value_l420_420843


namespace time_to_traverse_nth_mile_l420_420365

theorem time_to_traverse_nth_mile (n : ℕ) (h : n ≥ 2) :
  (∃ k : ℝ, (∀ d : ℝ, d = n - 1 → (s_n = k / d)) ∧ (s_2 = 1 / 2)) → 
  t_n = 2 * (n - 1) :=
by 
  sorry

end time_to_traverse_nth_mile_l420_420365


namespace constant_term_expansion_l420_420681

noncomputable def constant_term_in_expansion : ℕ := 26730

theorem constant_term_expansion :
  let expr := (λ (x y : ℚ), (x ^ (1 / 2) + 3 / (x ^ 2) + y) ^ 12)
  ∃ (x y : ℚ), x ≠ 0 ∧ expr x y = constant_term_in_expansion :=
begin
  sorry
end

end constant_term_expansion_l420_420681


namespace sin_ratio_of_triangle_on_ellipse_l420_420523

open Real

/-- Given a triangle ABC in a rectangular coordinate system with vertices 
    A at (-4,0), C at (4,0), and B lying on the ellipse x²/25 + y²/9 = 1,
    prove that (sin A + sin C) / sin B = 5/4. -/
theorem sin_ratio_of_triangle_on_ellipse :
  ∀ (A B C : ℝ × ℝ)
  (hA : A = (-4, 0)) (hC : C = (4, 0))
  (hB_ellipse : (B.1)^2 / 25 + (B.2)^2 / 9 = 1), 
  (sin (angle B A C) + sin (angle A B C)) / sin (angle A C B) = 5 / 4 := 
  sorry

end sin_ratio_of_triangle_on_ellipse_l420_420523


namespace necessary_sufficient_condition_l420_420437

def indicator (U : set ℝ) (x : ℝ) : ℝ := if x ∈ U then 1 else 0

theorem necessary_sufficient_condition {A B : set ℝ} :
  (∀ x : ℝ, indicator A x + indicator B x = 1) ↔ (A ∪ B = set.univ ∧ A ∩ B = ∅) :=
begin
  sorry
end

end necessary_sufficient_condition_l420_420437


namespace stock_investment_decrease_l420_420375

theorem stock_investment_decrease (x : ℝ) (d1 d2 : ℝ) (hx : x > 0)
  (increase : x * 1.30 = 1.30 * x) :
  d1 = 20 ∧ d2 = 3.85 → 1.30 * (1 - d1 / 100) * (1 - d2 / 100) = 1 := by
  sorry

end stock_investment_decrease_l420_420375


namespace jars_water_fraction_l420_420420

theorem jars_water_fraction (S L W : ℝ) (h1 : W = 1/6 * S) (h2 : W = 1/5 * L) : 
  (2 * W / L) = 2 / 5 :=
by
  -- We are only stating the theorem here, not proving it.
  sorry

end jars_water_fraction_l420_420420


namespace actual_distance_traveled_l420_420047

-- Definitions based on conditions
def original_speed : ℕ := 12
def increased_speed : ℕ := 20
def distance_difference : ℕ := 24

-- We need to prove the actual distance traveled by the person.
theorem actual_distance_traveled : 
  ∃ t : ℕ, increased_speed * t = original_speed * t + distance_difference → original_speed * t = 36 :=
by
  sorry

end actual_distance_traveled_l420_420047


namespace pyramid_triangular_face_area_l420_420215

theorem pyramid_triangular_face_area 
  (base : ℝ) (lateral : ℝ)
  (h_base : base = 8) (h_lateral : lateral = 7) :
  let height := sqrt (lateral^2 - (base / 2)^2) in
  4 * (1 / 2 * base * height) = 16 * sqrt 33 := 
by
  sorry

end pyramid_triangular_face_area_l420_420215


namespace rectangular_prism_surface_area_l420_420196

/-- The surface area of a rectangular prism with edge lengths 2, 3, and 4 is 52. -/
theorem rectangular_prism_surface_area :
  let a := 2
  let b := 3
  let c := 4
  2 * (a * b + a * c + b * c) = 52 :=
by
  let a := 2
  let b := 3
  let c := 4
  show 2 * (a * b + a * c + b * c) = 52
  sorry

end rectangular_prism_surface_area_l420_420196


namespace find_y_l420_420720

theorem find_y (θ : ℝ) (y : ℝ) :
  (let P := (4, y),
       sin_θ := -2 * real.sqrt 5 / 5,
       sin_def := y / real.sqrt (4^2 + y^2)
   in sin_def = sin_θ) → y = -8 :=
by
  sorry

end find_y_l420_420720


namespace relationship_among_a_ae_ea_minus_one_l420_420539

theorem relationship_among_a_ae_ea_minus_one (a : ℝ) (h : 0 < a ∧ a < 1) :
  (Real.exp a - 1 > a ∧ a > Real.exp a - 1 ∧ a > a^(Real.exp 1)) :=
by
  sorry

end relationship_among_a_ae_ea_minus_one_l420_420539


namespace find_positive_integer_solutions_l420_420431

-- Define the problem conditions
variable {x y z : ℕ}

-- Main theorem statement
theorem find_positive_integer_solutions 
  (h1 : Prime y)
  (h2 : ¬ 3 ∣ z)
  (h3 : ¬ y ∣ z)
  (h4 : x^3 - y^3 = z^2) : 
  x = 8 ∧ y = 7 ∧ z = 13 := 
sorry

end find_positive_integer_solutions_l420_420431


namespace find_monthly_income_l420_420583

-- Given condition
def deposit : ℝ := 3400
def percentage : ℝ := 0.15

-- Goal: Prove Sheela's monthly income
theorem find_monthly_income : (deposit / percentage) = 22666.67 := by
  -- Skip the proof for now
  sorry

end find_monthly_income_l420_420583


namespace prob_no_infection_correct_prob_one_infection_correct_l420_420760

-- Probability that no chicken is infected
def prob_no_infection (p_not_infected : ℚ) (n : ℕ) : ℚ := p_not_infected^n

-- Given
def p_not_infected : ℚ := 4 / 5
def n : ℕ := 5

-- Expected answer for no chicken infected
def expected_prob_no_infection : ℚ := 1024 / 3125

-- Lean statement
theorem prob_no_infection_correct : 
  prob_no_infection p_not_infected n = expected_prob_no_infection := by
  sorry

-- Probability that exactly one chicken is infected
def prob_one_infection (p_infected : ℚ) (p_not_infected : ℚ) (n : ℕ) : ℚ := 
  (n * p_not_infected^(n-1) * p_infected)

-- Given
def p_infected : ℚ := 1 / 5

-- Expected answer for exactly one chicken infected
def expected_prob_one_infection : ℚ := 256 / 625

-- Lean statement
theorem prob_one_infection_correct : 
  prob_one_infection p_infected p_not_infected n = expected_prob_one_infection := by
  sorry

end prob_no_infection_correct_prob_one_infection_correct_l420_420760


namespace odd_function_decreasing_function_max_min_values_on_interval_l420_420106

variable (f : ℝ → ℝ)

axiom func_additive : ∀ x y : ℝ, f (x + y) = f x + f y
axiom func_negative_for_positive : ∀ x : ℝ, (0 < x) → f x < 0
axiom func_value_at_one : f 1 = -2

theorem odd_function : ∀ x : ℝ, f (-x) = -f x := by
  have f_zero : f 0 = 0 := by sorry
  sorry

theorem decreasing_function : ∀ x₁ x₂ : ℝ, (x₁ < x₂) → f x₁ > f x₂ := by sorry

theorem max_min_values_on_interval :
  (f (-3) = 6) ∧ (f 3 = -6) := by sorry

end odd_function_decreasing_function_max_min_values_on_interval_l420_420106


namespace pyramid_total_area_l420_420286

noncomputable def pyramid_base_edge : ℝ := 8
noncomputable def pyramid_lateral_edge : ℝ := 7

theorem pyramid_total_area 
  (base_edge : ℝ)
  (lateral_edge : ℝ)
  (h_base_edge : base_edge = pyramid_base_edge)
  (h_lateral_edge : lateral_edge = pyramid_lateral_edge) 
: 4 * (1 / 2 * base_edge * real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 16 * real.sqrt 33 := 
sorry

end pyramid_total_area_l420_420286


namespace total_area_of_pyramid_faces_l420_420273

-- Define the basic parameters of the pyramid
def base_edges := 8
def lateral_edges := 7

-- Define the Pythagorean theorem components
def altitude_squared := lateral_edges^2 - (base_edges / 2)^2
def altitude := real.sqrt altitude_squared

-- Define the area of one triangular face using half-base and altitude
def one_face_area := (1 / 2) * base_edges * altitude

-- Define the total area of the four triangular faces
def total_area_of_faces := 4 * one_face_area

-- Statement to prove
theorem total_area_of_pyramid_faces : total_area_of_faces = 16 * real.sqrt 33 :=
by
  -- Necessary calculations are assured correct by previous definitions and attributes
  sorry

end total_area_of_pyramid_faces_l420_420273


namespace find_n_l420_420845

theorem find_n (n X : ℕ) (h1 : ∀ k, k ∈ {1, 2, 3, ..., n} → P(X = k) = 1 / n) (h2 : P(X < 4) = 0.3) : 
  n = 10 := 
sorry

end find_n_l420_420845


namespace pyramid_total_area_l420_420289

noncomputable def pyramid_base_edge : ℝ := 8
noncomputable def pyramid_lateral_edge : ℝ := 7

theorem pyramid_total_area 
  (base_edge : ℝ)
  (lateral_edge : ℝ)
  (h_base_edge : base_edge = pyramid_base_edge)
  (h_lateral_edge : lateral_edge = pyramid_lateral_edge) 
: 4 * (1 / 2 * base_edge * real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 16 * real.sqrt 33 := 
sorry

end pyramid_total_area_l420_420289


namespace pyramid_total_area_l420_420226

/-- The total area of the four triangular faces of a right, square-based pyramid
whose base edges measure 8 units and lateral edges measure 7 units is 16√33. -/
theorem pyramid_total_area :
  let base_edge := 8
  let lateral_edge := 7
  4 * (1 / 2 * base_edge * Real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 16 * Real.sqrt 33 := 
by
  let base_edge := 8
  let lateral_edge := 7
  have h1 : 4 * (1 / 2 * base_edge * Real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 
              4 * (1 / 2 * 8 * Real.sqrt (49 - 16)) := 
    by sorry
  have h2 : 4 * (1 / 2 * 8 * Real.sqrt (49 - 16)) = 4 * (4 * Real.sqrt 33) := 
    by sorry
  have h3 : 4 * (4 * Real.sqrt 33) = 16 * Real.sqrt 33 := 
    by sorry
  exact eq.trans (eq.trans h1 h2) h3

end pyramid_total_area_l420_420226


namespace residue_7_1234_mod_13_l420_420688

theorem residue_7_1234_mod_13 : (7^1234 : ℕ) % 13 = 4 :=
by
  have h1: 7 % 13 = 7 := rfl
  have h2: (7^2) % 13 = 10 := by norm_num
  have h3: (7^3) % 13 = 5 := by norm_num
  have h4: (7^4) % 13 = 9 := by norm_num
  have h5: (7^5) % 13 = 11 := by norm_num
  have h6: (7^6) % 13 = 12 := by norm_num
  have h7: (7^7) % 13 = 6 := by norm_num
  have h8: (7^8) % 13 = 3 := by norm_num
  have h9: (7^9) % 13 = 8 := by norm_num
  have h10: (7^10) % 13 = 4 := by norm_num
  have h11: (7^11) % 13 = 2 := by norm_num
  have h12: (7^12) % 13 = 1 := by norm_num
  sorry

end residue_7_1234_mod_13_l420_420688


namespace residue_7_1234_mod_13_l420_420689

theorem residue_7_1234_mod_13 : (7^1234 : ℕ) % 13 = 4 :=
by
  have h1: 7 % 13 = 7 := rfl
  have h2: (7^2) % 13 = 10 := by norm_num
  have h3: (7^3) % 13 = 5 := by norm_num
  have h4: (7^4) % 13 = 9 := by norm_num
  have h5: (7^5) % 13 = 11 := by norm_num
  have h6: (7^6) % 13 = 12 := by norm_num
  have h7: (7^7) % 13 = 6 := by norm_num
  have h8: (7^8) % 13 = 3 := by norm_num
  have h9: (7^9) % 13 = 8 := by norm_num
  have h10: (7^10) % 13 = 4 := by norm_num
  have h11: (7^11) % 13 = 2 := by norm_num
  have h12: (7^12) % 13 = 1 := by norm_num
  sorry

end residue_7_1234_mod_13_l420_420689


namespace maia_remaining_requests_after_two_weeks_l420_420992

noncomputable def total_requests_received_per_week (weekdays_requests : ℕ) (weekend_requests : ℕ) : ℕ :=
  weekdays_requests * 5 + weekend_requests * 2

noncomputable def total_requests_worked_on_per_week (weekdays_worked : ℕ) (saturday_worked : ℕ) : ℕ :=
  weekdays_worked * 5 + saturday_worked

theorem maia_remaining_requests_after_two_weeks (requests_per_weekday : ℕ) (requests_per_weekend : ℕ) 
(requests_worked_on_per_day : ℕ) (requests_worked_on_saturday : ℕ) :
  requests_per_weekday = 8 →
  requests_per_weekend = 5 →
  requests_worked_on_per_day = 4 →
  requests_worked_on_saturday = 4 →
  let total_requests := 2 * total_requests_received_per_week 8 5 in
  let total_worked := 2 * total_requests_worked_on_per_week 4 4 in
  total_requests - total_worked = 52 :=
by
  intros
  simp only[]
  sorry

end maia_remaining_requests_after_two_weeks_l420_420992


namespace probability_all_boys_probability_one_girl_probability_at_least_one_girl_l420_420138

-- Assumptions and Definitions
def total_outcomes := Nat.choose 5 3
def all_boys_outcomes := Nat.choose 3 3
def one_girl_outcomes := Nat.choose 3 2 * Nat.choose 2 1
def at_least_one_girl_outcomes := one_girl_outcomes + Nat.choose 3 1 * Nat.choose 2 2

-- The probability calculation proofs
theorem probability_all_boys : all_boys_outcomes / total_outcomes = 1 / 10 := by 
  sorry

theorem probability_one_girl : one_girl_outcomes / total_outcomes = 6 / 10 := by 
  sorry

theorem probability_at_least_one_girl : at_least_one_girl_outcomes / total_outcomes = 9 / 10 := by 
  sorry

end probability_all_boys_probability_one_girl_probability_at_least_one_girl_l420_420138


namespace a6_mod_n_l420_420977

variables (a n : ℕ)
noncomputable def a_inverse_mod_n := nat.gcd_a (a^3) n

theorem a6_mod_n (h1 : n > 0) (h2 : a^3 % n = a_inverse_mod_n % n) : (a^6) % n = 1 :=
by {
  sorry
}

end a6_mod_n_l420_420977


namespace stamp_collection_value_l420_420381

-- Define the conditions
variable (num_stamps : ℕ) -- Total number of stamps
variable (value_4_stamps : ℕ) -- Value of 4 stamps
variable (same_value : ℕ) -- Value of each stamp

-- Main statement
theorem stamp_collection_value (h_num_stamps : num_stamps = 20) 
                               (h_value_4_stamps : value_4_stamps = 16) 
                               (h_same_value : ∀ i, i < 4 → same_value = value_4_stamps / 4) :
                               num_stamps * same_value = 80 :=
begin
  sorry -- Proof
end

end stamp_collection_value_l420_420381


namespace transformed_stats_l420_420450

variable {n : ℕ}
variable (x : Fin n → ℝ) (x_bar s : ℝ)

def mean (x : Fin n → ℝ) : ℝ :=
  (∑ i, x i) / n

def stddev (x : Fin n → ℝ) (x_bar : ℝ) : ℝ :=
  sqrt ((∑ i, (x i - x_bar)^2) / n)

def transformed_mean (x_bar : ℝ) : ℝ :=
  3 * x_bar - 1

def transformed_variance (s : ℝ) : ℝ :=
  9 * s^2

theorem transformed_stats (h_mean : mean x = x_bar) (h_stddev : stddev x x_bar = s) :
  mean (fun i => 3 * x i - 1) = transformed_mean x_bar ∧
  stddev (fun i => 3 * x i - 1) (transformed_mean x_bar) = transformed_variance s := 
by
  sorry

end transformed_stats_l420_420450


namespace minimal_rectangle_exists_minimal_rectangle_size_l420_420831

universe u

-- Definitions
def binary_string (len : ℕ) := vector (fin 2) len

structure cell where
  content : option (fin 2)

structure rectangle (m n : ℕ) :=
(rows : matrix (fin m) (fin n) cell)
(is_complete : ∀ (S : binary_string n), ∃ r : fin m, ∀ j : fin n, rows r j).content = some (S.nth j))
(is_minimal : ∀ r : fin m, rectangle (m-1) n → ¬ (rectangle (m-1) n).is_complete)

-- Theorem for part (a)
theorem minimal_rectangle_exists (k : ℕ) (h : 0 < k ∧ k ≤ 2018) :
  ∃ (m : ℕ) (R : rectangle m 2018), m = 2^k ∧ count_columns_with_both_zero_and_one R = k ∧ R.is_minimal := 
sorry

-- Theorem for part (b)
theorem minimal_rectangle_size (m k : ℕ) (R : rectangle m 2018)
  (h : count_nonempty_columns R = k ∧ R.is_minimal) : m ≤ 2^k := 
sorry

end minimal_rectangle_exists_minimal_rectangle_size_l420_420831


namespace order_of_magnitude_l420_420003

theorem order_of_magnitude (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 1 < b) :
  log a b < a ^ b ∧ a ^ b < b ^ a :=
by sorry

end order_of_magnitude_l420_420003


namespace jill_arrives_before_jack_l420_420953

def pool_distance : ℝ := 2
def jill_speed : ℝ := 12
def jack_speed : ℝ := 4
def hours_to_minutes (hours : ℝ) : ℝ := hours * 60

theorem jill_arrives_before_jack
    (d : ℝ) (v_jill : ℝ) (v_jack : ℝ) (convert : ℝ → ℝ)
    (h_d : d = pool_distance)
    (h_vj : v_jill = jill_speed)
    (h_vk : v_jack = jack_speed)
    (h_convert : convert = hours_to_minutes) :
  convert (d / v_jack) - convert (d / v_jill) = 20 := by
  sorry

end jill_arrives_before_jack_l420_420953


namespace min_value_S_l420_420458

noncomputable def S (a b : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range (n - 1), ∑ j in Finset.rangeFrom (i + 1) (n - (i + 1)), a (i + 1) * b (j + 1)

theorem min_value_S
  (n : ℕ)
  (a b : ℕ → ℝ)
  (h_a_nonneg : ∀ i, 1 ≤ i → i ≤ n → 0 ≤ a i)
  (h_b_nonneg : ∀ i, 1 ≤ i → i ≤ n → 0 ≤ b i)
  (h_a_decreasing : ∀ i j, 1 ≤ i → i < j → j ≤ n → a i ≥ a j)
  (h_b_increasing : ∀ i j, 1 ≤ i → i < j → j ≤ n → b i ≤ b j)
  (h_a_sum : ∑ i in Finset.range n, a (i + 1) * a (n - i) = 1)
  (h_b_sum : ∑ i in Finset.range n, b (i + 1) * b (n - i) = 1) :
  S a b n ≥ (n - 1 : ℝ) / 2 :=
sorry

end min_value_S_l420_420458


namespace replace_digits_and_check_divisibility_l420_420135

theorem replace_digits_and_check_divisibility (a b : ℕ) (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) :
    (30 * 10^5 + a * 10^4 + b * 10^2 + 3 ≠ 0 ∧ 
     (30 * 10^5 + a * 10^4 + b * 10^2 + 3) % 13 = 0) ↔ 
    (30 * 10^5 + a * 10^4 + b * 10^2 + 3 = 3000803 ∨ 
     30 * 10^5 + a * 10^4 + b * 10^2 + 3 = 3020303 ∨ 
     30 * 10^5 + a * 10^4 + b * 10^2 + 3 = 3030703 ∨ 
     30 * 10^5 + a * 10^4 + b * 10^2 + 3 = 3050203 ∨ 
     30 * 10^5 + a * 10^4 + b * 10^2 + 3 = 3060603 ∨ 
     30 * 10^5 + a * 10^4 + b * 10^2 + 3 = 3080103 ∨ 
     30 * 10^5 + a * 10^4 + b * 10^2 + 3 = 3090503) := sorry

end replace_digits_and_check_divisibility_l420_420135


namespace find_x_l420_420487

noncomputable def A (x : ℝ) : set ℝ := {0, 1, real.log (x^2 + 2) / real.log 3, x^2 - 3 * x}

theorem find_x (x : ℝ) (h : -2 ∈ A x) : x = 2 :=
sorry

end find_x_l420_420487


namespace symmetry_of_circles_l420_420008

noncomputable def is_symmetric_about (C₁ C₂ : ℝ × ℝ → Prop) (l : ℝ × ℝ → Prop) : Prop :=
  ∀⦃x y⦄, C₁ (x, y) ↔ C₂ (2 - x, 2 - y)

def circle1 (p : ℝ × ℝ) : Prop :=
  let (x, y) := p in x^2 + y^2 = 4

def circle2 (p : ℝ × ℝ) : Prop :=
  let (x, y) := p in x^2 + y^2 - 4 * x + 4 * y + 4 = 0

def line_l (p : ℝ × ℝ) : Prop :=
  let (x, y) := p in y = x - 2

theorem symmetry_of_circles :
  is_symmetric_about circle1 circle2 line_l :=
sorry

end symmetry_of_circles_l420_420008


namespace angle_AHC_l420_420383

theorem angle_AHC (A B C D E H : Type)
  (h1 : angle A B C = 83)
  (h2 : angle B A C = 34)
  (h3 : ∃ D E : Type, is_altitude A B C D E)
  (h4 : ∃ H : Type, is_orthocenter A B C H) : angle A H C = 117 :=
by
  sorry

end angle_AHC_l420_420383


namespace find_r_l420_420813

theorem find_r (r : ℝ) (h : ⌊r⌋ + r = 16.5) : r = 8.5 :=
sorry

end find_r_l420_420813


namespace area_of_shaded_design_l420_420680

-- Define vertices of the triangles
def triangle1_vertices : finset (ℕ × ℕ) := {(0,0), (4,0), (0,4)}
def triangle2_vertices : finset (ℕ × ℕ) := {(4,0), (4,4), (0,4)}

-- Function to calculate the area using Pick's theorem
def area_of_triangle (vertices : finset (ℕ × ℕ)) : ℕ := 
let i := 6 in  -- interior points
let b := 6 in  -- boundary points
i + b / 2 - 1

-- Total area of the two triangles combined
def total_area_of_shaded_design : ℕ :=
  area_of_triangle triangle1_vertices + area_of_triangle triangle2_vertices

-- Theorem to assert the total area
theorem area_of_shaded_design : total_area_of_shaded_design = 16 := 
by
  -- Using Pick's theorem and symmetry
  have area_t1 : area_of_triangle triangle1_vertices = 8 := by sorry,
  have area_t2 : area_of_triangle triangle2_vertices = 8 := by sorry,
  rw [total_area_of_shaded_design, area_t1, area_t2],
  norm_num

end area_of_shaded_design_l420_420680


namespace total_area_of_pyramid_faces_l420_420268

-- Define the basic parameters of the pyramid
def base_edges := 8
def lateral_edges := 7

-- Define the Pythagorean theorem components
def altitude_squared := lateral_edges^2 - (base_edges / 2)^2
def altitude := real.sqrt altitude_squared

-- Define the area of one triangular face using half-base and altitude
def one_face_area := (1 / 2) * base_edges * altitude

-- Define the total area of the four triangular faces
def total_area_of_faces := 4 * one_face_area

-- Statement to prove
theorem total_area_of_pyramid_faces : total_area_of_faces = 16 * real.sqrt 33 :=
by
  -- Necessary calculations are assured correct by previous definitions and attributes
  sorry

end total_area_of_pyramid_faces_l420_420268


namespace number_of_valid_lineups_l420_420826

def is_valid_lineup (lineup : List ℕ) : Prop :=
  ∀ i, i < lineup.length - 1 → abs (lineup[i] - lineup[i + 1]) ≠ 1

theorem number_of_valid_lineups :
  ∃ n, n = 14 ∧ ∀ (lineup : List ℕ), lineup.perm [65, 66, 67, 68, 69] → is_valid_lineup lineup →
  n = 14 := by
  sorry

end number_of_valid_lineups_l420_420826


namespace product_segment_doubles_l420_420952

-- Define the problem conditions and proof statement in Lean.
theorem product_segment_doubles
  (a b e : ℝ)
  (d : ℝ := (a * b) / e)
  (e' : ℝ := e / 2)
  (d' : ℝ := (a * b) / e') :
  d' = 2 * d := 
  sorry

end product_segment_doubles_l420_420952


namespace find_m_l420_420050

/-- 
If the function y=x + m/(x-1) defined for x > 1 attains its minimum value at x = 3,
then the positive number m is 4.
-/
theorem find_m (m : ℝ) (h : ∀ x : ℝ, 1 < x -> x + m / (x - 1) ≥ 3 + m / 2):
  m = 4 :=
sorry

end find_m_l420_420050


namespace sweets_distribution_l420_420571

theorem sweets_distribution (S X : ℕ) (h1 : S = 112 * X) (h2 : S = 80 * (X + 6)) :
  X = 15 := 
by
  sorry

end sweets_distribution_l420_420571


namespace trajectory_of_M_l420_420488

-- Points A and B, and moving point M
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)

-- Define M as a moving point with coordinates (x, y)
variables {x y : ℝ}

-- The condition on the product of the slopes of lines AM and BM
def slopes_condition (A B : ℝ × ℝ) (M : ℝ × ℝ) : Prop :=
  let (x, y) := M in
  x ≠ 1 ∧ x ≠ -1 ∧ (y / (x + 1)) * (y / (x - 1)) = -2

-- The equation of the trajectory of point M
def trajectory_equation (x y : ℝ) : Prop :=
  x^2 + (y^2 / 2) = 1 ∧ x ≠ 1 ∧ x ≠ -1

-- The proof problem: prove that under given conditions, the trajectory of point M satisfies the equation.
theorem trajectory_of_M (h : slopes_condition A B (x, y)) : trajectory_equation x y :=
  sorry

end trajectory_of_M_l420_420488


namespace number_of_rotational_phenomena_is_4_l420_420765

def is_rotational (phenomenon : ℕ) : Prop :=
  phenomenon = 3 ∨ phenomenon = 4 ∨ phenomenon = 5 ∨ phenomenon = 6

theorem number_of_rotational_phenomena_is_4 :
  (finset.card (finset.filter is_rotational (finset.range 7))) = 4 :=
by
  sorry

end number_of_rotational_phenomena_is_4_l420_420765


namespace find_middle_part_length_l420_420739

theorem find_middle_part_length (a b c : ℝ) 
  (h1 : a + b + c = 28) 
  (h2 : (a - 0.5 * a) + b + 0.5 * c = 16) :
  b = 4 :=
by
  sorry

end find_middle_part_length_l420_420739


namespace pyramid_four_triangular_faces_area_l420_420261

theorem pyramid_four_triangular_faces_area 
  (base_edge : ℝ) (lateral_edge : ℝ) 
  (h_base : base_edge = 8)
  (h_lateral : lateral_edge = 7) :
  let h := Real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)
  let triangle_area := (1 / 2) * base_edge * h
  let total_area := 4 * triangle_area
  total_area = 16 * Real.sqrt 33 :=
by
  -- Definitions to introduce local values
  let half_base := base_edge / 2
  let h := Real.sqrt (lateral_edge ^ 2 - half_base ^ 2)
  let triangle_area := (1 / 2) * base_edge * h
  let total_area := 4 * triangle_area
  -- Assertion to compare calculated total area with given correct answer
  have h_eq : h = Real.sqrt 33 := by sorry
  have triangle_area_eq : triangle_area = 4 * Real.sqrt 33 := by sorry
  have total_area_eq : total_area = 16 * Real.sqrt 33 := by sorry
  exact total_area_eq

end pyramid_four_triangular_faces_area_l420_420261


namespace speed_of_second_train_l420_420677

-- Defining the conditions
def length_train1 : ℝ := 130 -- length of the first train in meters
def length_train2 : ℝ := 160 -- length of the second train in meters
def crossing_time : ℝ := 10.439164866810657 -- time to cross each other in seconds
def speed_train1_kmh : ℝ := 60 -- speed of the first train in km/hr

-- Conversion constants
def kmh_to_ms : ℝ := 1000.0 / 3600.0 -- conversion factor from km/hr to m/s

-- Conversion of speed of first train to m/s
def speed_train1_ms : ℝ := speed_train1_kmh * kmh_to_ms

-- Calculation of total distance covered during crossing
def total_distance : ℝ := length_train1 + length_train2

-- Calculation of relative speed in m/s
def relative_speed : ℝ := total_distance / crossing_time

-- Calculation of speed of the second train in m/s
def speed_train2_ms : ℝ := relative_speed - speed_train1_ms

-- Conversion of speed of the second train to km/hr
def speed_train2_kmh : ℝ := speed_train2_ms / kmh_to_ms

-- The main theorem statement
theorem speed_of_second_train : speed_train2_kmh = 40 := by sorry

end speed_of_second_train_l420_420677


namespace pyramid_total_area_l420_420227

/-- The total area of the four triangular faces of a right, square-based pyramid
whose base edges measure 8 units and lateral edges measure 7 units is 16√33. -/
theorem pyramid_total_area :
  let base_edge := 8
  let lateral_edge := 7
  4 * (1 / 2 * base_edge * Real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 16 * Real.sqrt 33 := 
by
  let base_edge := 8
  let lateral_edge := 7
  have h1 : 4 * (1 / 2 * base_edge * Real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 
              4 * (1 / 2 * 8 * Real.sqrt (49 - 16)) := 
    by sorry
  have h2 : 4 * (1 / 2 * 8 * Real.sqrt (49 - 16)) = 4 * (4 * Real.sqrt 33) := 
    by sorry
  have h3 : 4 * (4 * Real.sqrt 33) = 16 * Real.sqrt 33 := 
    by sorry
  exact eq.trans (eq.trans h1 h2) h3

end pyramid_total_area_l420_420227


namespace pyramid_face_area_total_l420_420239

theorem pyramid_face_area_total 
  (base_edge : ℝ) (lateral_edge : ℝ) 
  (h_base_edge : base_edge = 8) 
  (h_lateral_edge : lateral_edge = 7) : 
  4 * (1 / 2 * base_edge * real.sqrt (lateral_edge^2 - (base_edge / 2)^2)) = 16 * real.sqrt 33 :=
by
  sorry

end pyramid_face_area_total_l420_420239


namespace summation_series_lt_one_l420_420788

theorem summation_series_lt_one :
  (∑ k in finset.range 2012, (k + 1 : ℝ) / ((k + 2)! : ℝ)) < 1 :=
by sorry

end summation_series_lt_one_l420_420788


namespace pyramid_area_l420_420298

noncomputable def base_edge : ℝ := 8
noncomputable def lateral_edge : ℝ := 7
noncomputable def height := real.sqrt (lateral_edge^2 - (base_edge / 2)^2)
noncomputable def one_triangle_area := (1 / 2) * base_edge * height
noncomputable def total_area := 4 * one_triangle_area

theorem pyramid_area : total_area = 16 * real.sqrt 33 := by
  sorry

end pyramid_area_l420_420298


namespace sum_series_eq_half_l420_420395

theorem sum_series_eq_half :
  ∑' n : ℕ, (3^(n+1) / (9^(n+1) - 1)) = 1/2 := 
sorry

end sum_series_eq_half_l420_420395


namespace expected_sequence_examples_geometric_sequence_expected_arithmetic_sequence_expected_l420_420558

-- Definition of an nth order expected sequence
def is_expected_sequence (n : ℕ) (a : ℕ → ℚ) : Prop :=
  (finset.range n).sum a = 0 ∧ (finset.range n).sum (λ i, |a i|) = 1

-- Part I: Examples of 3rd and 4th order expected sequences
theorem expected_sequence_examples :
  is_expected_sequence 3 (λ i, if i = 0 then -1/2 else if i = 1 then 0 else 1/2) ∧
  is_expected_sequence 4 (λ i, if i = 0 then -3/8 else if i = 1 then -1/8 else if i = 2 then 1/8 else 3/8) :=
by
  sorry

-- Part II: Geometric sequence with common ratio q that forms a 2014th order expected sequence
theorem geometric_sequence_expected (a : ℕ → ℚ) (q : ℚ) (h : is_expected_sequence 2014 a) :
  (∀ n, a (n + 1) = q * a n) → q = -1 :=
by
  sorry

-- Part III: Arithmetic sequence general formula for 2k-th order expected sequence
theorem arithmetic_sequence_expected (a : ℕ → ℚ) (k : ℕ) (h₀ : is_expected_sequence (2 * k) a)
  (h₁ : ∀ n, a (n + 1) - a n > 0) :
  ∃ d : ℚ, d = 1 / k^2 ∧ ∀ n, a n = (n / k^2) - (2 * k + 1) / (2 * k^2) :=
by
  sorry

end expected_sequence_examples_geometric_sequence_expected_arithmetic_sequence_expected_l420_420558


namespace shoe_price_on_monday_l420_420114

theorem shoe_price_on_monday
  (price_on_thursday : ℝ)
  (price_increase : ℝ)
  (discount : ℝ)
  (price_on_friday : ℝ := price_on_thursday * (1 + price_increase))
  (price_on_monday : ℝ := price_on_friday * (1 - discount))
  (price_on_thursday_eq : price_on_thursday = 50)
  (price_increase_eq : price_increase = 0.2)
  (discount_eq : discount = 0.15) :
  price_on_monday = 51 :=
by
  sorry

end shoe_price_on_monday_l420_420114


namespace total_area_of_triangular_faces_l420_420204

noncomputable def pyramid_base_edge : ℝ := 8
noncomputable def pyramid_lateral_edge : ℝ := 7

structure Pyramid where
  base_edge : ℝ
  lateral_edge : ℝ

def myPyramid : Pyramid :=
{ base_edge := pyramid_base_edge,
  lateral_edge := pyramid_lateral_edge }

theorem total_area_of_triangular_faces :
  4 * (1 / 2) * myPyramid.base_edge * (Real.sqrt (myPyramid.lateral_edge^2 - (myPyramid.base_edge / 2)^2)) = 16 * Real.sqrt 33 :=
by
  sorry

end total_area_of_triangular_faces_l420_420204


namespace sufficient_but_not_necessary_l420_420498

theorem sufficient_but_not_necessary (a b x : ℝ) (h : x > a^2 + b^2) : 
  (x > 2ab) ∧ ¬ (∀ x, x > 2ab → x > a^2 + b^2) :=
by {
  sorry,
}

end sufficient_but_not_necessary_l420_420498


namespace num_pairs_satisfy_condition_l420_420564

def S := {0, 1, 2}

def op (i j : ℕ) : ℕ := (i + j) % 3

def satisfies_condition (i j : ℕ) : Prop :=
  (op (op i j) i = 0)

theorem num_pairs_satisfy_condition :
  (set_of (λ (i : ℕ × ℕ), satisfies_condition i.fst i.snd) 
  ∩ ({1, 2, 3} × {1, 2, 3})).card = 3 :=
by sorry

end num_pairs_satisfy_condition_l420_420564


namespace parabola_distance_pf_l420_420905

noncomputable def parabola_focus_x (x1 x9 : ℕ) (h_sum : x1 + x9 = 45) (arithmetic_seq : x1 + x9 = 2 * x1 + 7 * x9) : ℕ :=
  let x5 := 5 in
  x5 + 1

theorem parabola_distance_pf (x1 x9 : ℕ) (h_sum : x1 + x9 = 45) (arithmetic_seq : x1 + x9 = 2 * x1 + 7 * x9) : 
  parabola_focus_x  x1 x9 h_sum arithmetic_seq = 6 :=
by
  sorry

end parabola_distance_pf_l420_420905


namespace total_area_of_triangular_faces_l420_420206

noncomputable def pyramid_base_edge : ℝ := 8
noncomputable def pyramid_lateral_edge : ℝ := 7

structure Pyramid where
  base_edge : ℝ
  lateral_edge : ℝ

def myPyramid : Pyramid :=
{ base_edge := pyramid_base_edge,
  lateral_edge := pyramid_lateral_edge }

theorem total_area_of_triangular_faces :
  4 * (1 / 2) * myPyramid.base_edge * (Real.sqrt (myPyramid.lateral_edge^2 - (myPyramid.base_edge / 2)^2)) = 16 * Real.sqrt 33 :=
by
  sorry

end total_area_of_triangular_faces_l420_420206


namespace solve_for_x_l420_420507

theorem solve_for_x (x y z w : ℤ) (h1 : x + y = 4) (h2 : x - y = 36) 
(h3 : x * z + y * w = 50) (h4 : z - w = 5) : x = 20 := 
by 
  sorry

end solve_for_x_l420_420507


namespace area_of_rectangle_l420_420738

theorem area_of_rectangle (P : ℝ) (w : ℝ) (h : ℝ) (A : ℝ) 
  (hP : P = 28) 
  (hw : w = 6) 
  (hP_formula : P = 2 * (h + w)) 
  (hA_formula : A = h * w) : 
  A = 48 :=
by
  sorry

end area_of_rectangle_l420_420738


namespace sequence_first_element_eventually_one_l420_420951

-- Define the sequence transformation function
def transform_sequence (seq : List ℕ) : List ℕ :=
  let k := seq.head!
  (seq.take k).reverse ++ seq.drop k

-- Define the predicate to check if the first element eventually becomes 1
def eventually_first_element_is_one (seq : List ℕ) : Prop :=
  ∃ n, (iterate transform_sequence n seq).head! = 1

-- Define the initial sequence condition for the proof
def is_permutation_of_first_n_numbers (seq : List ℕ) : Prop :=
  seq ~ List.range' 1 (seq.length + 1)

-- The main theorem statement
theorem sequence_first_element_eventually_one (seq : List ℕ) (h : seq.length = 2017) 
(h_perm : is_permutation_of_first_n_numbers seq) : eventually_first_element_is_one seq :=
  sorry

end sequence_first_element_eventually_one_l420_420951


namespace kolya_cannot_guess_within_3steps_l420_420120

noncomputable def perimeter_guess_precision_3steps (polygon : Type) (circle : Type) : Prop :=
  ∀ (Kolya : polygon → list Prop), 
  (∀ p : polygon, ∃ l₁ l₂ l₃ : polygon → Prop, 
    Kolya p = [l₁ p, l₂ p, l₃ p] ∧
    (∃ p₁ p₂ : polygon, perimeter p₁ < 5.68 ∧ perimeter p₂ > 6.28 ∧ 
    all_intersects_circle [l₁, l₂, l₃] ∧ ¬distinguishable p₁ p₂ 0.3)) → 
  ¬can_guess_perimeter Kolya 3 0.3

theorem kolya_cannot_guess_within_3steps (polygon : Type) (circle : Type) : 
  convex polygon ∧ in_circle_with_center circle 1 polygon → 
  perimeter_guess_precision_3steps polygon circle :=
sorry

end kolya_cannot_guess_within_3steps_l420_420120


namespace largest_number_of_gold_coins_l420_420317

theorem largest_number_of_gold_coins (n : ℕ) :
  (∃ k : ℕ, n = 13 * k + 3 ∧ n < 150) → n ≤ 146 :=
by
  sorry

end largest_number_of_gold_coins_l420_420317


namespace circle_equation_l420_420432

theorem circle_equation
  (a b r : ℝ) 
  (h1 : a^2 + b^2 = r^2) 
  (h2 : (a - 2)^2 + b^2 = r^2) 
  (h3 : b / (a - 2) = 1) : 
  (x - 1)^2 + (y + 1)^2 = 2 := 
by
  sorry

end circle_equation_l420_420432


namespace find_phi_l420_420095

noncomputable def phi : ℝ :=
  let z := Complex.exp (2 * Real.pi * Complex.I / 7)
  let Q := z * z ^ 2 * z ^ 3
  let (s, θ) := Complex.polarCoords Q
  θ

theorem find_phi : φ = 148.57 := by
  sorry

end find_phi_l420_420095


namespace rearrangement_count_correct_l420_420829

def original_number := "1234567890"

def is_valid_rearrangement (n : String) : Prop :=
  n.length = 10 ∧ n.front ≠ '0'
  
def count_rearrangements (n : String) : ℕ :=
  if n = original_number 
  then 232
  else 0

theorem rearrangement_count_correct :
  count_rearrangements original_number = 232 :=
sorry


end rearrangement_count_correct_l420_420829


namespace total_area_of_pyramid_faces_l420_420270

-- Define the basic parameters of the pyramid
def base_edges := 8
def lateral_edges := 7

-- Define the Pythagorean theorem components
def altitude_squared := lateral_edges^2 - (base_edges / 2)^2
def altitude := real.sqrt altitude_squared

-- Define the area of one triangular face using half-base and altitude
def one_face_area := (1 / 2) * base_edges * altitude

-- Define the total area of the four triangular faces
def total_area_of_faces := 4 * one_face_area

-- Statement to prove
theorem total_area_of_pyramid_faces : total_area_of_faces = 16 * real.sqrt 33 :=
by
  -- Necessary calculations are assured correct by previous definitions and attributes
  sorry

end total_area_of_pyramid_faces_l420_420270


namespace max_area_difference_160_perimeter_rectangles_l420_420659

theorem max_area_difference_160_perimeter_rectangles : 
  ∃ (l1 w1 l2 w2 : ℕ), (2 * l1 + 2 * w1 = 160) ∧ (2 * l2 + 2 * w2 = 160) ∧ 
  (l1 * w1 - l2 * w2 = 1521) := sorry

end max_area_difference_160_perimeter_rectangles_l420_420659


namespace total_area_of_pyramid_faces_l420_420264

-- Define the basic parameters of the pyramid
def base_edges := 8
def lateral_edges := 7

-- Define the Pythagorean theorem components
def altitude_squared := lateral_edges^2 - (base_edges / 2)^2
def altitude := real.sqrt altitude_squared

-- Define the area of one triangular face using half-base and altitude
def one_face_area := (1 / 2) * base_edges * altitude

-- Define the total area of the four triangular faces
def total_area_of_faces := 4 * one_face_area

-- Statement to prove
theorem total_area_of_pyramid_faces : total_area_of_faces = 16 * real.sqrt 33 :=
by
  -- Necessary calculations are assured correct by previous definitions and attributes
  sorry

end total_area_of_pyramid_faces_l420_420264


namespace maximum_area_of_triangle_in_rectangle_l420_420369

theorem maximum_area_of_triangle_in_rectangle :
  ∃ (K : ℝ), 
    let a := 12 in
    let b := 13 in
    (∀ (triangle : Type), 
       is_equilateral_triangle triangle → 
       (∀ (point : Point), point ∈ triangle → point ∈ Rectangle a b) → 
       area triangle ≤ K) 
    ∧ K = 275 * Real.sqrt 3 - 390 := sorry

end maximum_area_of_triangle_in_rectangle_l420_420369


namespace sum_x_y_z_w_l420_420148

-- Define the conditions in Lean
variables {x y z w : ℤ}
axiom h1 : x - y + z = 7
axiom h2 : y - z + w = 8
axiom h3 : z - w + x = 4
axiom h4 : w - x + y = 3

-- Prove the result
theorem sum_x_y_z_w : x + y + z + w = 22 := by
  sorry

end sum_x_y_z_w_l420_420148


namespace pyramid_triangular_face_area_l420_420209

theorem pyramid_triangular_face_area 
  (base : ℝ) (lateral : ℝ)
  (h_base : base = 8) (h_lateral : lateral = 7) :
  let height := sqrt (lateral^2 - (base / 2)^2) in
  4 * (1 / 2 * base * height) = 16 * sqrt 33 := 
by
  sorry

end pyramid_triangular_face_area_l420_420209


namespace z_completion_time_l420_420339

variable (x y z : Type) [NormedField x] [NormedField y] [NormedField z]

theorem z_completion_time : 
  (x := 40) → (y := 30) → 
  (x_worked : ∀ (days_worked: ℕ) (total_days: ℕ), total_days = 40 → 
    days_worked = 8 → x := 1 / total_days * days_worked) →
  (y_and_z_wr: ∀ (days_worked: ℕ), (y := 1 / 30 + 1/z * days_worked) → 
    days_worked = 20 → (x := 8 / 40 * 8) → 
    4 / 5 = 20 * (1 / 30 + 1 / d)) → (d = 150) := 
  sorry

end z_completion_time_l420_420339


namespace Sue_waited_in_NY_l420_420145

-- Define the conditions as constants and assumptions
def T_NY_SF : ℕ := 24
def T_total : ℕ := 58
def T_NO_NY : ℕ := (3 * T_NY_SF) / 4

-- Define the waiting time
def T_wait : ℕ := T_total - T_NO_NY - T_NY_SF

-- Theorem stating the problem
theorem Sue_waited_in_NY :
  T_wait = 16 :=
by
  -- Implicitly using the given conditions
  sorry

end Sue_waited_in_NY_l420_420145


namespace student_number_in_eighth_group_l420_420355

-- Definitions corresponding to each condition
def students : ℕ := 50
def group_size : ℕ := 5
def third_group_student_number : ℕ := 12
def kth_group_number (k : ℕ) (n : ℕ) : ℕ := n + (k - 3) * group_size

-- Main statement to prove
theorem student_number_in_eighth_group :
  kth_group_number 8 third_group_student_number = 37 :=
  by
  sorry

end student_number_in_eighth_group_l420_420355


namespace q_0_plus_q_5_l420_420552

noncomputable def q (x : ℝ) : ℝ := sorry
axiom q_monic : polynomial.leadingCoeff q = 1
axiom q_deg5 : polynomial.degree q = 5
axiom q_1 : q 1 = 10
axiom q_2 : q 2 = 20
axiom q_3 : q 3 = 30

theorem q_0_plus_q_5 : q 0 + q 5 = -40 := sorry

end q_0_plus_q_5_l420_420552


namespace yao_ming_mcgrady_probability_l420_420773

theorem yao_ming_mcgrady_probability
        (p : ℝ) (q : ℝ)
        (h1 : p = 0.8)
        (h2 : q = 0.7) :
        (2 * p * (1 - p)) * (2 * q * (1 - q)) = 0.1344 := 
by
  sorry

end yao_ming_mcgrady_probability_l420_420773


namespace inequality_proof_l420_420104

theorem inequality_proof (x y z t : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (ht : 0 ≤ t) (h : x + y + z + t = 7) :
  sqrt (x^2 + y^2) + sqrt (x^2 + 1) + sqrt (z^2 + y^2) + sqrt (t^2 + 64) + sqrt (z^2 + t^2) ≥ 17 := 
sorry

end inequality_proof_l420_420104


namespace harmonic_inequalities_harmonic_sum_for_n2_l420_420129

theorem harmonic_inequalities (n : ℕ) (h : n > 1) :
  (n + 2) / 2 < ∑ i in Finset.range (2 * n + 1), (1 : ℚ) / i ∧ ∑ i in Finset.range (2 * n + 1), (1 : ℚ) / i < n + 1 := sorry

theorem harmonic_sum_for_n2 :
  ∑ i in Finset.range 5, (1 : ℚ) / i = 1 + 1 / 2 + 1 / 3 + 1 / 4 := sorry

end harmonic_inequalities_harmonic_sum_for_n2_l420_420129


namespace max_tournament_rounds_l420_420932

-- Define the problem using Lean 4 statements
theorem max_tournament_rounds (n k : ℕ) (h_k_pos : k > 0) :
  ∀ (G : SimpleGraph (Fin n)), G.IsKRegular k → (tournament_rounds G = k) := 
sorry

end max_tournament_rounds_l420_420932


namespace outermost_diameter_l420_420929

def radius_of_fountain := 6 -- derived from the information that 12/2 = 6
def width_of_garden := 9
def width_of_inner_walking_path := 3
def width_of_outer_walking_path := 7

theorem outermost_diameter :
  2 * (radius_of_fountain + width_of_garden + width_of_inner_walking_path + width_of_outer_walking_path) = 50 :=
by
  sorry

end outermost_diameter_l420_420929


namespace water_added_to_solution_l420_420349

theorem water_added_to_solution :
  let initial_volume := 340
  let initial_sugar := 0.20 * initial_volume
  let added_sugar := 3.2
  let added_kola := 6.8
  let final_sugar := initial_sugar + added_sugar
  let final_percentage_sugar := 19.66850828729282 / 100
  let final_volume := final_sugar / final_percentage_sugar
  let added_water := final_volume - initial_volume - added_sugar - added_kola
  added_water = 12 :=
by
  sorry

end water_added_to_solution_l420_420349


namespace point_on_x_axis_l420_420501

theorem point_on_x_axis (x : ℝ) (A : ℝ × ℝ) (h : A = (2 - x, x + 3)) (hy : A.snd = 0) : A = (5, 0) :=
by
  sorry

end point_on_x_axis_l420_420501


namespace pyramid_face_area_total_l420_420237

theorem pyramid_face_area_total 
  (base_edge : ℝ) (lateral_edge : ℝ) 
  (h_base_edge : base_edge = 8) 
  (h_lateral_edge : lateral_edge = 7) : 
  4 * (1 / 2 * base_edge * real.sqrt (lateral_edge^2 - (base_edge / 2)^2)) = 16 * real.sqrt 33 :=
by
  sorry

end pyramid_face_area_total_l420_420237


namespace greatest_area_difference_l420_420668

theorem greatest_area_difference (l₁ w₁ l₂ w₂ : ℕ) (h₁ : 2 * l₁ + 2 * w₁ = 160) (h₂ : 2 * l₂ + 2 * w₂ = 160) : 
  abs (l₁ * w₁ - l₂ * w₂) = 1521 :=
sorry

end greatest_area_difference_l420_420668


namespace minimum_a_l420_420503

theorem minimum_a (a : ℝ) : (∀ x : ℝ, 2 * x^2 - (x - a) * |x - a| - 2 ≥ 0) → a ≥ Real.sqrt 3 := 
by 
  sorry

end minimum_a_l420_420503


namespace total_unused_crayons_l420_420117

/-
  Madeline has 12 boxes of crayons. Various conditions about the number of crayons and their usage are given.
  Prove that the total number of unused crayons is 289.
-/

def crayon_data : List (ℕ × ℚ) := [
  (3, 30 * 1/2),   -- First 3 boxes
  (2, 36 * 3/4),   -- Next 2 boxes
  (2, 40 * 2/5),   -- 6th and 7th boxes
  (1, 45 * 5/9),   -- 8th box
  (2, 48 * 7/8),   -- 9th and 10th boxes
  (1, 27 * 5/6),   -- 11th box
  (1, 54 * 1/2)    -- Last box
]

theorem total_unused_crayons : (crayon_data.map (λ x, x.1 * x.2)).sum = 289 := by
  -- Need to prove the above statement; the current proof is omitted
  sorry

end total_unused_crayons_l420_420117


namespace improvement_sum_exists_l420_420398

-- Define the problem's parameters and conditions
def improve (x : ℕ) : ℕ := 
  let bp := 2 ^ (Nat.log2 (3 * x)) in 
  if bp = 0 then 1 else bp

theorem improvement_sum_exists (a : Fin 2 ^ 100 → ℕ) 
  (h_sum : (∑ i, a i) = 2 ^ 100) :
  ∃ (b : Fin 2 ^ 100 → ℕ), 
    (∀ i, b i = improve (a i))
    ∧ (∑ i, b i = 2 ^ 100) 
    ∧ (∀ i, 1 ≤ b i ∧ b i ≤ 3 * a i) := 
sorry

end improvement_sum_exists_l420_420398


namespace range_of_x_l420_420483

noncomputable def f (x : ℝ) := |x - 2|

theorem range_of_x (a b x : ℝ) (h₀ : a ≠ 0) (h₁ : f(x) ≤ 2) :
  |a + b| + |a - b| ≥ |a| * f(x) :=
by {
  have h₂ : ∀ x, 0 ≤ x → x ≤ 4 → |x - 2| ≤ 2, by {
    intros x hx₀ hx₁,
    calc 
      |x - 2| = abs (x - 2) : by sorry -- Properties of absolute value
           ... ≤ 2 : by sorry -- Given in problem conditions
  },
  sorry -- The rest of the proof, where we would use h₀, h₁, and h₂
}

end range_of_x_l420_420483


namespace thursday_occurs_five_times_l420_420590

-- Definitions and assumptions based on conditions
def july_has_five_tuesdays (N : ℕ) : Prop :=
∃ (day : ℕ), (1 ≤ day ∧ day ≤ 3) ∧
  (day + 7 ≤ 31) ∧ (day + 14 ≤ 31) ∧ (day + 21 ≤ 31) ∧ (day + 28 ≤ 31)

def august_has_31_days : Prop := true

-- The theorem stating the question
theorem thursday_occurs_five_times (N : ℕ) (h1 : july_has_five_tuesdays N) (h2 : august_has_31_days) :
  ∃ (day : ℕ), (1 ≤ day ∧ day ≤ 7) ∧
    ((day + 4) mod 7 = 4) ∧
    list_count (list.map (λ d, (d + (1 - day.days))) (list.range 31)) 4 = 5 :=
sorry

end thursday_occurs_five_times_l420_420590


namespace sum_of_two_cosines_not_equal_to_third_l420_420174

theorem sum_of_two_cosines_not_equal_to_third
  (α β γ : ℝ)
  (h_pos_α : 0 < α)
  (h_pos_β : 0 < β)
  (h_pos_γ : 0 < γ)
  (h_sum_angles : α + β + γ = π / 2) :
  ¬ (cos α + cos β = cos γ) :=
sorry

end sum_of_two_cosines_not_equal_to_third_l420_420174


namespace three_in_A_even_not_in_A_l420_420031

def A : Set ℤ := {x | ∃ m n : ℤ, x = m^2 - n^2}

-- (1) Prove that 3 ∈ A
theorem three_in_A : 3 ∈ A :=
sorry

-- (2) Prove that ∀ k ∈ ℤ, 4k - 2 ∉ A
theorem even_not_in_A (k : ℤ) : (4 * k - 2) ∉ A :=
sorry

end three_in_A_even_not_in_A_l420_420031


namespace area_of_right_triangle_l420_420556

def is_on_ellipse (a b : ℝ) (e : ℝ) (x y : ℝ) : Prop :=
  (x^2)/(a^2) + (y^2)/(b^2) = 1

theorem area_of_right_triangle (a b : ℝ) : 
  let F1 := (-√(a^2 - b^2), 0)
  let F2 := (√(a^2 - b^2), 0)
  let P := (0, b)
  in is_on_ellipse a b 0 (P.1) (P.2) ∧ 
     ∠ (F1, F2, P) = 90 → 
     2 * F1.1 * (b^2 / a) / 2 = 48 / 5 :=
sorry

end area_of_right_triangle_l420_420556


namespace total_juice_sold_3_days_l420_420531

def juice_sales_problem (V_L V_M V_S : ℕ) (d1 d2 d3 : ℕ) :=
  (d1 = V_L + 4 * V_M) ∧ 
  (d2 = 2 * V_L + 6 * V_S) ∧ 
  (d3 = V_L + 3 * V_M + 3 * V_S) ∧
  (d1 = d2) ∧
  (d2 = d3)

theorem total_juice_sold_3_days (V_L V_M V_S d1 d2 d3 : ℕ) 
  (h : juice_sales_problem V_L V_M V_S d1 d2 d3) 
  (h_VM : V_M = 3) 
  (h_VL : V_L = 6) : 
  3 * d1 = 54 := 
by 
  -- Proof will be filled in
  sorry

end total_juice_sold_3_days_l420_420531


namespace distance_covered_by_end_of_day_seven_distance_covered_by_end_of_day_ten_day_less_than_0_point_001_meters_l420_420716

def half_distance_travel (day : ℕ) : ℝ :=
  let initial_distance := 10.0 
  initial_distance * (0.5 ^ day)

theorem distance_covered_by_end_of_day_seven :
  let total_distance := ∑ i in Finset.range 7, half_distance_travel (i+1)
  total_distance = 9.921875 :=
by
  sorry

theorem distance_covered_by_end_of_day_ten :
  let total_distance := ∑ i in Finset.range 10, half_distance_travel (i+1)
  total_distance = 9.990234375 :=
by
  sorry

theorem day_less_than_0_point_001_meters (day : ℕ) :
  let total_distance := ∑ i in Finset.range day, half_distance_travel (i+1)
  10.0 - total_distance < 0.001 ↔ day ≥ 14 :=
by
  sorry

end distance_covered_by_end_of_day_seven_distance_covered_by_end_of_day_ten_day_less_than_0_point_001_meters_l420_420716


namespace celine_smartphones_l420_420372

-- Definitions based on the conditions
def laptop_cost : ℕ := 600
def smartphone_cost : ℕ := 400
def num_laptops_bought : ℕ := 2
def initial_amount : ℕ := 3000
def change_received : ℕ := 200

-- The proof goal is to show that the number of smartphones bought is 4
theorem celine_smartphones (laptop_cost smartphone_cost num_laptops_bought initial_amount change_received : ℕ)
  (h1 : laptop_cost = 600)
  (h2 : smartphone_cost = 400)
  (h3 : num_laptops_bought = 2)
  (h4 : initial_amount = 3000)
  (h5 : change_received = 200) :
  (initial_amount - change_received - num_laptops_bought * laptop_cost) / smartphone_cost = 4 := 
by
  sorry

end celine_smartphones_l420_420372


namespace f_expression_on_interval_l420_420107

noncomputable def f : ℝ → ℝ := sorry

theorem f_expression_on_interval (x : ℝ) (hx : -3 ≤ x ∧ x ≤ -1) :
  f x = x^2 + 2x + 4 :=
by
  have periodicity : ∀ x : ℝ, f (x - 2) = f x := sorry
  have f_defined_on_interval : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x = x^2 - 2x + 4 := sorry
  sorry

end f_expression_on_interval_l420_420107


namespace triangle_cosine_sum_l420_420510

variable (A B C : Type) [Triangle A B C] (a b c : ℝ)
variable [Cosine A B C]
variable [SideLength A B C]

theorem triangle_cosine_sum 
  (ha : SideLength A B C A = 2)
  : SideLength A B C B * Cosine A B C C + SideLength A B C C * Cosine A B C B = 2 := 
by
  sorry

end triangle_cosine_sum_l420_420510


namespace total_area_of_triangular_faces_l420_420200

noncomputable def pyramid_base_edge : ℝ := 8
noncomputable def pyramid_lateral_edge : ℝ := 7

structure Pyramid where
  base_edge : ℝ
  lateral_edge : ℝ

def myPyramid : Pyramid :=
{ base_edge := pyramid_base_edge,
  lateral_edge := pyramid_lateral_edge }

theorem total_area_of_triangular_faces :
  4 * (1 / 2) * myPyramid.base_edge * (Real.sqrt (myPyramid.lateral_edge^2 - (myPyramid.base_edge / 2)^2)) = 16 * Real.sqrt 33 :=
by
  sorry

end total_area_of_triangular_faces_l420_420200


namespace total_erasers_is_35_l420_420780

def Celine : ℕ := 10

def Gabriel : ℕ := Celine / 2

def Julian : ℕ := Celine * 2

def total_erasers : ℕ := Celine + Gabriel + Julian

theorem total_erasers_is_35 : total_erasers = 35 :=
  by
  sorry

end total_erasers_is_35_l420_420780


namespace probability_point_in_square_l420_420743

theorem probability_point_in_square (r : ℝ) (hr : 0 < r) :
  (∃ p : ℝ, p = 2 / Real.pi) :=
by
  sorry

end probability_point_in_square_l420_420743


namespace quadrilateral_properties_l420_420980

noncomputable def distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

noncomputable def perimeter (P Q R S : (ℝ × ℝ)) : ℝ :=
  distance P Q + distance Q R + distance R S + distance S P

noncomputable def area_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  (1 / 2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

noncomputable def area_quadrilateral (P Q R S : (ℝ × ℝ)) : ℝ :=
  area_triangle P Q S + area_triangle Q R S

def P := (1, 0) : (ℝ × ℝ)
def Q := (4, 6) : (ℝ × ℝ)
def R := (8, 4) : (ℝ × ℝ)
def S := (11, 0) : (ℝ × ℝ)

theorem quadrilateral_properties :
  perimeter P Q R S = 5 * Real.sqrt 5 + 15 ∧ area_quadrilateral P Q R S = 35 :=
by 
  sorry

end quadrilateral_properties_l420_420980


namespace relationship_of_a_b_c_l420_420445

noncomputable def a : ℝ := Real.log 3 / Real.log 2  -- a = log2(1/3)
noncomputable def b : ℝ := Real.exp (1 / 3)  -- b = e^(1/3)
noncomputable def c : ℝ := 1 / 3  -- c = e^ln(1/3) = 1/3

theorem relationship_of_a_b_c : b > c ∧ c > a :=
by
  -- Proof would go here
  sorry

end relationship_of_a_b_c_l420_420445


namespace cyclist_saving_percentage_l420_420356

theorem cyclist_saving_percentage
  (amount_saved : ℤ) (amount_spent : ℤ)
  (hS : amount_saved = 8)
  (hP : amount_spent = 32) :
  let original_price := amount_spent + amount_saved in
  let percentage_saved := ((amount_saved : ℚ) / original_price) * 100 in
  percentage_saved = 20 := by
  sorry

end cyclist_saving_percentage_l420_420356


namespace profit_percent_300_l420_420335

theorem profit_percent_300 (SP : ℝ) (CP : ℝ) (h : CP = 0.25 * SP) : ((SP - CP) / CP) * 100 = 300 :=
by
  sorry

end profit_percent_300_l420_420335


namespace tan_angle_sum_l420_420461

theorem tan_angle_sum (a : ℝ) (ha : sin a = -5/13 ∧ (cos a > 0)) :
  tan (a + π/4) = 7/17 :=
sorry

end tan_angle_sum_l420_420461


namespace part_1_part_2_l420_420037

variables (x m : ℝ)

def p (x : ℝ) : Prop := |x + 1| ≤ 3
def q (x m : ℝ) : Prop := x^2 - 2x + 1 - m^2 ≤ 0

theorem part_1 (h : m = 2) (hpq_or : p x ∨ q x m) (hpq_and : ¬(p x ∧ q x m)) : -4 ≤ x ∧ x < -1 ∨ 2 < x ∧ x ≤ 3 :=
sorry

theorem part_2 (hpq_necessary : ∀ x, p x → q x m) (hpq_not_sufficient : ∃ x, q x m ∧ ¬p x) : 0 < m ∧ m ≤ 1 :=
sorry

end part_1_part_2_l420_420037


namespace sequence_sum_l420_420342

theorem sequence_sum :
  let seq := λ n, if even n then 1990 - n * 10 else 1990 - n * 10
  let terms := 199
  (((list.range terms).map seq)).sum = 1000 := by
  sorry

end sequence_sum_l420_420342


namespace binomial_expansion_constant_term_l420_420469

theorem binomial_expansion_constant_term (m n : ℝ) :
  (binomial_expansion_constant_term : (m * 2 ^ 2 n ^ 2) 6 = 60) -> m 2  n  = 4 :=
begin
  sorry
end

end binomial_expansion_constant_term_l420_420469


namespace tiling_board_8x1_colored_tiles_l420_420387

theorem tiling_board_8x1_colored_tiles :
  let N := number_of_tilings_with_all_colors 8 in
  N % 1000 = 504 :=
sorry

end tiling_board_8x1_colored_tiles_l420_420387


namespace unique_point_with_property_l420_420124

universe u

structure ConvexPolygon (n : ℕ) :=
  (vertices : Fin n → ℝ × ℝ)
  (convex : ∀ (i j k : Fin n), i ≠ j → j ≠ k → k ≠ i → 
    (∃ (a b c : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 1 ∧ 
      a • vertices i + b • vertices j + c • vertices k = vertices (Fin.succ (Fin.succ k))))

variable {n : ℕ}

def line_through_point (p Q : ℝ × ℝ) : ℝ × ℝ → Prop :=
  λ R, ∃ a b c : ℝ, a • p + b • Q = R

def has_property (P : ℝ × ℝ) (poly : ConvexPolygon n) : Prop :=
  ∀ i : Fin n, ∃ j : Fin n, i ≠ j ∧ ∃ R, line_through_point P (poly.vertices i) R ∧ R = poly.vertices j

theorem unique_point_with_property (poly : ConvexPolygon n) (O : ℝ × ℝ)
  (hO : has_property O poly) :
  ∀ P : ℝ × ℝ, has_property P poly → P = O := sorry

end unique_point_with_property_l420_420124


namespace part_1_part_2_l420_420920

-- Conditions and definitions
noncomputable def triangle_ABC (a b c S : ℝ) (A B C : ℝ) :=
  a * Real.sin B = -b * Real.sin (A + Real.pi / 3) ∧
  S = Real.sqrt 3 / 4 * c^2

-- 1. Prove A = 5 * Real.pi / 6
theorem part_1 (a b c S A B C : ℝ) (h : triangle_ABC a b c S A B C) :
  A = 5 * Real.pi / 6 :=
  sorry

-- 2. Prove sin C = sqrt 7 / 14 given S = sqrt 3 / 4 * c^2
theorem part_2 (a b c S A B C : ℝ) (h : triangle_ABC a b c S A B C) :
  Real.sin C = Real.sqrt 7 / 14 :=
  sorry

end part_1_part_2_l420_420920


namespace pyramid_area_l420_420279

theorem pyramid_area (base_edge lateral_edge : ℝ) (H_base_edge : base_edge = 8) (H_lateral_edge : lateral_edge = 7) :
  4 * (1 / 2 * base_edge * real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 16 * real.sqrt 33 :=
by
  sorry

end pyramid_area_l420_420279


namespace min_distance_parallel_lines_l420_420857

theorem min_distance_parallel_lines :
  let l1 (x y : ℝ) := x + 3*y - 9 = 0
  let l2 (x y : ℝ) := x + 3*y + 1 = 0
  ∃ P1 P2 : ℝ × ℝ, (l1 P1.1 P1.2 ∧ l2 P2.1 P2.2) → 
    dist P1 P2 = sqrt 10 := by
  sorry

end min_distance_parallel_lines_l420_420857


namespace tangent_lines_ln_e_proof_l420_420918

noncomputable def tangent_tangent_ln_e : Prop :=
  ∀ (x₁ x₂ : ℝ) 
  (h₁ : x₁ > 0) 
  (h₁_eq : x₂ = -Real.log x₁)
  (h₂_eq : Real.log x₁ - 1 = (Real.exp x₂) * (1 - x₂)),
  (2 / (x₁ - 1) + x₂ = -1)

theorem tangent_lines_ln_e_proof : tangent_tangent_ln_e :=
  sorry

end tangent_lines_ln_e_proof_l420_420918


namespace count_four_digit_integers_divisible_by_5_and_8_l420_420902

theorem count_four_digit_integers_divisible_by_5_and_8 : 
  let lcm := Nat.lcm 5 8,
      a := 1000,
      l := 9960,
      d := lcm,
      n := (l - a) / d + 1 in
  lcm = 40 ∧ n = 225 :=
by
  let lcm := 40
  let a := 1000
  let l := 9960
  let d := 40
  let n := (l - a) / d + 1
  have h_lcm : lcm = Nat.lcm 5 8 := by sorry
  have h_n : n = 225 := by sorry
  exact ⟨h_lcm, h_n⟩

end count_four_digit_integers_divisible_by_5_and_8_l420_420902


namespace residue_7_1234_mod_13_l420_420690

theorem residue_7_1234_mod_13 : (7^1234 : ℕ) % 13 = 4 :=
by
  have h1: 7 % 13 = 7 := rfl
  have h2: (7^2) % 13 = 10 := by norm_num
  have h3: (7^3) % 13 = 5 := by norm_num
  have h4: (7^4) % 13 = 9 := by norm_num
  have h5: (7^5) % 13 = 11 := by norm_num
  have h6: (7^6) % 13 = 12 := by norm_num
  have h7: (7^7) % 13 = 6 := by norm_num
  have h8: (7^8) % 13 = 3 := by norm_num
  have h9: (7^9) % 13 = 8 := by norm_num
  have h10: (7^10) % 13 = 4 := by norm_num
  have h11: (7^11) % 13 = 2 := by norm_num
  have h12: (7^12) % 13 = 1 := by norm_num
  sorry

end residue_7_1234_mod_13_l420_420690


namespace sum_difference_l420_420642

theorem sum_difference (h_odd_sum : ∑ n in finset.range 100, (2 * (n + 1) - 1) = 10000)
                       (h_even_sum : 2 * (∑ n in finset.range 100, (2 * (n + 1))) = 20200) :
  20200 - 10000 = 10200 :=
by
  sorry

end sum_difference_l420_420642


namespace sum_of_sequence_l420_420394

theorem sum_of_sequence :
  3 + 15 + 27 + 53 + 65 + 17 + 29 + 41 + 71 + 83 = 404 :=
by
  sorry

end sum_of_sequence_l420_420394


namespace train_length_calculation_l420_420754

def speed_km_per_hr : ℝ := 60
def time_sec : ℝ := 9
def length_of_train : ℝ := 150

theorem train_length_calculation :
  (speed_km_per_hr * 1000 / 3600) * time_sec = length_of_train := by
  sorry

end train_length_calculation_l420_420754


namespace complex_number_second_quadrant_l420_420410

theorem complex_number_second_quadrant 
  : (2 + 3 * Complex.I) / (1 - Complex.I) ∈ { z : Complex | z.re < 0 ∧ z.im > 0 } := 
by
  sorry

end complex_number_second_quadrant_l420_420410


namespace bounded_sequence_iff_l420_420559

def euler_totient (n : ℕ) : ℕ :=
  nat.totient n

def f (k n : ℕ) : ℕ :=
  k * euler_totient n

def sequence_bounded (k a : ℕ) : Prop :=
  ∃ M, ∀ m, a > 1 → k > 1 → ∃ (x_m : ℕ), x_m <= M ∧ (x_m = a ∨ ∃ i, x_m = f k x_m ∧ x_m < x_m)

theorem bounded_sequence_iff (k : ℕ) :
  (∀ (a : ℕ), a > 1 → sequence_bounded k a) ↔ (k = 2 ∨ k = 3) :=
sorry

end bounded_sequence_iff_l420_420559


namespace helmet_regression_prediction_l420_420188

theorem helmet_regression_prediction :
  let x := [1, 2, 3, 4]
  let y := [1150, 1000, 900, 750]
  let n := 4
  let mean (l : List ℕ) := (l.sum) / (l.length : ℕ)
  let x̄ := mean x
  let ȳ := mean y
  let Σxy := (List.zip x y).map (λ (xi, yi), (xi - x̄) * (yi - ȳ)).sum
  let Σx² := x.map (λ xi, (xi - x̄) ^ 2).sum
  let b := Σxy / Σx²
  let a := ȳ - b * x̄
  let regression_line (x : ℕ) := b * x + a in
  a = 1300 ∧ b = -140 ∧ regression_line 5 = 700 :=
by 
  -- Insert detailed proof here
  sorry

end helmet_regression_prediction_l420_420188


namespace min_value_proof_l420_420912

noncomputable def min_value {a b : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : ℝ :=
  \frac{1}{a} + \frac{2}{b}

theorem min_value_proof {a b : ℝ} 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a + b = 1) : 
  min_value h1 h2 h3 = 3 + 2 * Real.sqrt 2 :=
by
  sorry

end min_value_proof_l420_420912


namespace pyramid_area_l420_420297

noncomputable def base_edge : ℝ := 8
noncomputable def lateral_edge : ℝ := 7
noncomputable def height := real.sqrt (lateral_edge^2 - (base_edge / 2)^2)
noncomputable def one_triangle_area := (1 / 2) * base_edge * height
noncomputable def total_area := 4 * one_triangle_area

theorem pyramid_area : total_area = 16 * real.sqrt 33 := by
  sorry

end pyramid_area_l420_420297


namespace divisible_by_24_l420_420347

theorem divisible_by_24 (n : ℕ) : ∃ k : ℤ, n^4 + 2 * n^3 + 11 * n^2 + 10 * n = 24 * k := sorry

end divisible_by_24_l420_420347


namespace colorings_unique_l420_420624

def color := ℕ
def red : color := 0
def blue : color := 1
def green : color := 2

structure HexGrid :=
(columns : ℕ)
(hexagon : ℕ -> ℕ -> color)
(column_valid : ∀ c r, 0 ≤ c ∧ c < columns → 0 ≤ r → 
  (c = 0 → hexagon c r = red) ∧
  (0 < c → hexagon (c - 1) r ≠ hexagon c r ∧ 
           (r > 0 → hexagon c (r - 1) ≠ hexagon c r)))

theorem colorings_unique (G : HexGrid) : 
  (∀ c, 0 ≤ c ∧ c < G.columns → ∃! h : (ℕ → color), ∀ r, 0 ≤ r → G.hexagon c r = h r) 
  → G.columns = 1 ∨ G.columns = 2
  →  ∃! f : HexGrid, ∀ c, 0 ≤ c ∧ c < G.columns → 
     (∀ r, 0 ≤ r → G.hexagon c r = f.hexagon c r) :=
  by
    sorry

end colorings_unique_l420_420624


namespace nine_tuples_remainder_l420_420960

def S := Finset.range 2014 -- S = {1, 2, ..., 2013}

-- Define the conditions
def subset_conditions (S1 S2 S3 S4 S5 S6 S7 S8 S9 : Finset ℕ) : Prop :=
  ∀ (n : ℕ), n ∈ {0, 1, 2, 3, 4} →
    (S1 ⊆ S2 ∧ S3 ⊆ S2 ∧
    S3 ⊆ S4 ∧ S5 ⊆ S4 ∧
    S5 ⊆ S6 ∧ S7 ⊆ S6 ∧
    S7 ⊆ S8 ∧ S9 ⊆ S8 ∧
    S1 ⊆ S ∧ S2 ⊆ S ∧ S3 ⊆ S ∧ S4 ⊆ S ∧ S5 ⊆ S ∧ S6 ⊆ S ∧ S7 ⊆ S ∧ S8 ⊆ S ∧ S9 ⊆ S)

-- Define the main statement to be proved
theorem nine_tuples_remainder :
  ∃ (N : ℕ), (N = 512 ^ 2013) ∧ (N % 1000 = 72) :=
by
  sorry

end nine_tuples_remainder_l420_420960


namespace min_translation_overlap_l420_420052

theorem min_translation_overlap :
  ∀ (φ : ℝ),
    (∀ x : ℝ, sin (2 * (x + φ)) = cos (2 * (x - φ) - π / 6)) ↔ φ = π / 12 :=
sorry

end min_translation_overlap_l420_420052


namespace range_of_BP_dot_BA_l420_420947

open Real

-- Define points A and B, and curve P
def point_A := (1 : ℝ, 0 : ℝ)
def point_B := (0 : ℝ, -1 : ℝ)
def curve_P (x : ℝ) : ℝ := sqrt (1 - x^2)

-- Define BA vector
def vector_BA := (point_A.1 - point_B.1, point_A.2 - point_B.2)

-- Define BP vector when P is on the curve
def vector_BP (α : ℝ) : ℝ × ℝ := (cos α, sin α + 1)

-- Range of the dot product of vectors BP and BA
theorem range_of_BP_dot_BA :
  (∀ α ∈ Icc 0 π, (vector_BP α).1 * vector_BA.1 + (vector_BP α).2 * vector_BA.2 ∈ Icc 0 (1 + sqrt 2)) :=
sorry

end range_of_BP_dot_BA_l420_420947


namespace solution1_solution2_l420_420875

noncomputable def problem1 : Prop :=
  ∃ (a b : ℤ), 
  (∃ (n : ℤ), 3*a - 14 = n ∧ a - 2 = n) ∧ 
  (b - 15 = -27) ∧ 
  a = 4 ∧ 
  b = -12 ∧ 
  (4*a + b = 4)

noncomputable def problem2 : Prop :=
  ∀ (a b : ℤ), 
  (a = 4) ∧ 
  (b = -12) → 
  (4*a + b = 4) → 
  (∃ n, n^2 = 4 ∧ (n = 2 ∨ n = -2))

theorem solution1 : problem1 := by { sorry }
theorem solution2 : problem2 := by { sorry }

end solution1_solution2_l420_420875


namespace range_of_f_l420_420812

noncomputable def f (x : ℝ) : ℝ :=
  Real.arctan x + Real.arctan ((1 - x) / (1 + x)) + Real.arctan (2 * x)

theorem range_of_f : Set.Ioo (-(Real.pi / 2)) (Real.pi / 2) = Set.range f :=
  sorry

end range_of_f_l420_420812


namespace bad_arrangements_count_l420_420769

def numbers := [1, 2, 3, 4, 6]

def is_bad_arrangement (arrangement : List ℕ) : Prop :=
  ∃ (n : ℕ), n ∈ finset.range 1 17 ∧ ¬ ∃ (sublist : List ℕ), sublist.sum = n ∧ sublist <+ arrangement ∧
    sublist.length > 0

def unique_bad_arrangements := {x : List ℕ // Multiset.erased_dup x}.toFinset.card

theorem bad_arrangements_count : unique_bad_arrangements = 3 :=
by
  sorry

end bad_arrangements_count_l420_420769


namespace floor_div_eq_floor_floor_div_l420_420127

theorem floor_div_eq_floor_floor_div (α : ℝ) (d : ℕ) (hα : 0 < α) :
  ⌊α / d⌋ = ⌊⌊α⌋ / d⌋ :=
by sorry

end floor_div_eq_floor_floor_div_l420_420127


namespace largest_number_of_gold_coins_l420_420326

theorem largest_number_of_gold_coins 
  (num_friends : ℕ)
  (extra_coins : ℕ)
  (total_coins : ℕ) :
  num_friends = 13 →
  extra_coins = 3 →
  total_coins < 150 →
  ∀ k : ℕ, total_coins = num_friends * k + extra_coins →
  total_coins ≤ 146 :=
by
  sorry

end largest_number_of_gold_coins_l420_420326


namespace maximum_rectangle_area_l420_420440

variable (x y : ℝ)

def area (x y : ℝ) : ℝ :=
  x * y

def similarity_condition (x y : ℝ) : Prop :=
  (11 - x) / (y - 6) = 2

theorem maximum_rectangle_area :
  ∃ (x y : ℝ), similarity_condition x y ∧ area x y = 66 :=  by
  sorry

end maximum_rectangle_area_l420_420440


namespace triangles_congruent_l420_420652

noncomputable def circles_passing_through_P (P A B C A1 B1 C1 A2 B2 C2 : Point) : Prop :=
∃ (circle1 circle2 circle3 : Circle), 
  circle1 ∋ P ∧ circle1 ∋ A ∧
  circle2 ∋ P ∧ circle2 ∋ B ∧
  circle3 ∋ P ∧ circle3 ∋ C ∧
  A1 = second_intersection (line P A).circle circle1 ∧
  B1 = second_intersection (line P B).circle circle2 ∧
  C1 = second_intersection (line P C).circle circle3 ∧
  C2 = intersection (line A B1) (line B A1) ∧
  A2 = intersection (line B C1) (line C B1) ∧
  B2 = intersection (line A C1) (line C A1) ∧
  collinear A B C

theorem triangles_congruent 
  (P A B C A1 B1 C1 A2 B2 C2 : Point) 
  (h : circles_passing_through_P P A B C A1 B1 C1 A2 B2 C2) :
  congruent (triangle A1 B1 C1) (triangle A2 B2 C2) :=
sorry

end triangles_congruent_l420_420652


namespace total_area_of_pyramid_faces_l420_420266

-- Define the basic parameters of the pyramid
def base_edges := 8
def lateral_edges := 7

-- Define the Pythagorean theorem components
def altitude_squared := lateral_edges^2 - (base_edges / 2)^2
def altitude := real.sqrt altitude_squared

-- Define the area of one triangular face using half-base and altitude
def one_face_area := (1 / 2) * base_edges * altitude

-- Define the total area of the four triangular faces
def total_area_of_faces := 4 * one_face_area

-- Statement to prove
theorem total_area_of_pyramid_faces : total_area_of_faces = 16 * real.sqrt 33 :=
by
  -- Necessary calculations are assured correct by previous definitions and attributes
  sorry

end total_area_of_pyramid_faces_l420_420266


namespace residue_7_1234_mod_13_l420_420685

theorem residue_7_1234_mod_13 :
  (7 : ℤ) ^ 1234 % 13 = 12 :=
by
  -- given conditions as definitions
  have h1 : (7 : ℤ) % 13 = 7 := by norm_num
  
  -- auxiliary calculations
  have h2 : (49 : ℤ) % 13 = 10 := by norm_num
  have h3 : (100 : ℤ) % 13 = 9 := by norm_num
  have h4 : (81 : ℤ) % 13 = 3 := by norm_num
  have h5 : (729 : ℤ) % 13 = 1 := by norm_num

  -- the actual problem we want to prove
  sorry

end residue_7_1234_mod_13_l420_420685


namespace value_range_of_a_l420_420004

variables (a : ℝ)

def A (a : ℝ) : set ℝ := {x | -2 ≤ x ∧ x ≤ a}
def B (a : ℝ) : set ℝ := {y | ∃ (x : ℝ), y = 2*x + 3 ∧ x ∈ A a}
def C (a : ℝ) : set ℝ := {t | ∃ (x : ℝ), t = x^2 ∧ x ∈ A a}

theorem value_range_of_a (h1 : a ≥ 2) 
  (h2 : C a ⊆ B a) : 2 ≤ a ∧ a ≤ 3 :=
by
  sorry

end value_range_of_a_l420_420004


namespace symmetric_cosine_l420_420476

def f (x : Real) (ω φ : Real) : Real :=
  3 * Real.cos (ω * x + φ)

theorem symmetric_cosine (ω φ : Real) (f_eq : ∀ x : Real, f (x + π / 6) ω φ = f (x - π / 6) ω φ) :
  f (π / 6) ω φ = 3 ∨ f (π / 6) ω φ = -3 :=
by
  sorry

end symmetric_cosine_l420_420476


namespace probability_at_least_3_of_5_long_service_high_award_high_award_dependency_on_service_years_l420_420761

-- Define the condition for "long-service workers" receiving a high saving award
def long_service_high_award_probability := 20 / 60

-- Define the condition for the chi-square test data
def long_service_high_award := 20
def long_service_low_award := 40
def short_service_high_award := 10
def short_service_low_award := 70
def total_high_award := 30
def total_low_award := 110
def total_long_service := 60
def total_short_service := 80
def total_workers := 140
def chi_square_critical_value := 7.879

-- Calculate chi-square statistic based on the table
def chi_square_statistic := 
  total_workers * ((long_service_high_award * short_service_low_award - long_service_low_award * short_service_high_award) ^ 2) / 
  (total_high_award * total_low_award * total_short_service * total_long_service)

-- Lean 4 theorem statement for Part (1)
theorem probability_at_least_3_of_5_long_service_high_award : 
  let p := long_service_high_award_probability
  let probability_3 := 5.choose 3 * p^3 * (1 - p)^2 
  let probability_4 := 5.choose 4 * p^4 * (1 - p) 
  let probability_5 := p^5 
  probability_3 + probability_4 + probability_5 = 17 / 81 := sorry

-- Lean 4 theorem statement for Part (2)
theorem high_award_dependency_on_service_years : 
  chi_square_statistic > chi_square_critical_value := sorry

end probability_at_least_3_of_5_long_service_high_award_high_award_dependency_on_service_years_l420_420761


namespace pencils_purchased_l420_420058

theorem pencils_purchased (n : ℕ) (h1: n ≤ 10) 
  (h2: 2 ≤ 10) 
  (h3: (10 - 2) / 10 * (10 - 2 - 1) / (10 - 1) * (10 - 2 - 2) / (10 - 2) = 0.4666666666666667) :
  n = 3 :=
sorry

end pencils_purchased_l420_420058


namespace ninth_graders_only_math_l420_420171

theorem ninth_graders_only_math 
  (total_students : ℕ)
  (math_students : ℕ)
  (foreign_language_students : ℕ)
  (science_only_students : ℕ)
  (math_and_foreign_language_no_science : ℕ)
  (h1 : total_students = 120)
  (h2 : math_students = 85)
  (h3 : foreign_language_students = 75)
  (h4 : science_only_students = 20)
  (h5 : math_and_foreign_language_no_science = 40) :
  math_students - math_and_foreign_language_no_science = 45 :=
by 
  sorry

end ninth_graders_only_math_l420_420171


namespace tan_ratio_tan_A_minus_B_max_val_l420_420525

variable (A B C a b c : ℝ)

-- Conditions in problem a)
variable (h1 : ∀ (A B C : ℝ), ∀ (a b c : ℝ), ∃ (k : ℝ), a * Real.cos B - b * Real.cos A = k * c ∧ k = 3 / 5)

-- Translation to a proof problem:
theorem tan_ratio (h : ∀ (A B C : ℝ), ∃ (a b c : ℝ), a * Real.cos B - b * Real.cos A = 3 / 5 * c) :
  ∃ (t : ℝ), Real.tan A / Real.tan B = t := 
by
  sorry

theorem tan_A_minus_B_max_val (h : ∀ (A B C : ℝ), ∃ (a b c : ℝ), a * Real.cos B - b * Real.cos A = 3 / 5 * c) : 
  ∀ (u : ℝ), (u = Real.tan A - Real.tan B) → Real.max u = 3 / 4 :=
by
  sorry

end tan_ratio_tan_A_minus_B_max_val_l420_420525


namespace part1_general_formula_part2_no_integer_T_n_eq_4_minus_n_l420_420870

noncomputable def S (n : ℕ) : ℚ := (n * (n + 1) : ℚ) / 2

def b (n : ℕ) : ℕ := n

noncomputable def c (n : ℕ) : ℚ := b n / (2 : ℚ)^(n - 1)

noncomputable def T (n : ℕ) : ℚ := (Finset.range n).sum (λ i, c (i + 1))

theorem part1_general_formula (n : ℕ) (h : n ≥ 1) :
  b n = n := sorry

theorem part2_no_integer_T_n_eq_4_minus_n :
  ¬ ∃ n : ℕ, n > 0 ∧ T n = 4 - n := sorry

end part1_general_formula_part2_no_integer_T_n_eq_4_minus_n_l420_420870


namespace Jonah_profit_l420_420530

theorem Jonah_profit
  (num_pineapples : ℕ)
  (cost_per_pineapple : ℕ)
  (rings_per_pineapple : ℕ)
  (rings_sold_per_set : ℕ)
  (profit_per_set : ℕ)
  (total_cost : ℕ)
  (total_rings : ℕ)
  (total_sets : ℕ)
  (total_revenue : ℕ)
  (profit : ℕ) :
  num_pineapples = 6 →
  cost_per_pineapple = 3 →
  rings_per_pineapple = 12 →
  rings_sold_per_set = 4 →
  profit_per_set = 5 * rings_sold_per_set →
  total_cost = num_pineapples * cost_per_pineapple →
  total_rings = num_pineapples * rings_per_pineapple →
  total_sets = total_rings / rings_sold_per_set →
  total_revenue = total_sets * profit_per_set →
  profit = total_revenue - total_cost →
  profit = 342 :=
begin
  intros,
  sorry
end

end Jonah_profit_l420_420530


namespace frisbee_price_l420_420742

theorem frisbee_price (F : Nat) (P : Nat) (F ≥ 8) (64 = F + (64 - F)) (4 * F + P * (64 - F) = 200) : P = 3 := 
by
  sorry

end frisbee_price_l420_420742


namespace no_prime_roots_of_quadratic_l420_420776

open Int Nat

theorem no_prime_roots_of_quadratic (k : ℤ) :
  ¬ (∃ p q : ℤ, Prime p ∧ Prime q ∧ p + q = 107 ∧ p * q = k) :=
by
  sorry

end no_prime_roots_of_quadratic_l420_420776


namespace remainder_b_div_6_l420_420639

theorem remainder_b_div_6 (a b : ℕ) (r_a r_b : ℕ) 
  (h1 : a ≡ r_a [MOD 6]) 
  (h2 : b ≡ r_b [MOD 6]) 
  (h3 : a > b) 
  (h4 : (a - b) % 6 = 5) 
  : b % 6 = 0 := 
sorry

end remainder_b_div_6_l420_420639


namespace garage_travel_time_correct_l420_420954

theorem garage_travel_time_correct :
  let floors := 12
  let gate_interval := 3
  let gate_time := 2 * 60 -- 2 minutes in seconds
  let distance_per_floor := 800
  let speed := 10 in
  let num_gates := floors / gate_interval
  let total_gate_time := num_gates * gate_time
  let time_per_floor := distance_per_floor / speed
  let total_drive_time := floors * time_per_floor in
  total_gate_time + total_drive_time = 1440 := by
  sorry

end garage_travel_time_correct_l420_420954


namespace probability_of_intersection_l420_420855

-- Define the sets A and B.
def set_A : Set ℝ := {x : ℝ | -2 < x ∧ x < 4}
def set_B : Set ℝ := {x : ℝ | x < 0 ∨ x > 3}

-- Define the probability calculation
def length (s : Set ℝ) : ℝ := sup s - inf s

noncomputable def probability (A B : Set ℝ) : ℝ :=
  length (A ∩ B) / length A

-- State the theorem
theorem probability_of_intersection :
  probability set_A set_B = 1 / 2 := 
sorry

end probability_of_intersection_l420_420855


namespace max_value_and_period_l420_420548

def vector_mult (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 * b.1, a.2 * b.2)

variables (m : ℝ × ℝ) (n : ℝ × ℝ) (P Q : ℝ × ℝ)
variables (f : ℝ → ℝ)

-- Given conditions
axiom P_on_curve : ∀ x, P = (x, sin (x / 2))
axiom vector_eq : ∀ x, Q = vector_mult m P + n

-- Question transformed into a proof problem
theorem max_value_and_period :
  f (x : ℝ) = (1 / 2) * sin (x / 2) + 1 →
  ∃ A T, A = 3 / 2 ∧ T = 4 * real.pi :=
sorry

end max_value_and_period_l420_420548


namespace brave_2022_first_reappearance_l420_420632

theorem brave_2022_first_reappearance :
  ∃ n : ℕ, 
  (n ≠ 0 ∧ 
   (∀ m : ℕ, m < n → ((List.cycle "BRAVE".toList m ≠ "BRAVE".toList) ∨ (List.cycle "2022".toList m ≠ "2022".toList)))) ∧ 
  (n % 5 = 0) ∧ (n % 4 = 0) ∧ n = 20 :=
by
  use 20
  split
  {
    intro h
    use 19
    exact λ hlt => ⟨9, Nat.lt_of_succ_le hlt, (by linarith)⟩
  }
  split
  { exact Nat.mod_eq_zero_of_dvd (dvd_of_mod_eq_zero h) }
  split
  { exact Nat.mod_eq_zero_of_dvd (dvd_of_mod_eq_zero h) }
  exact by rfl

end brave_2022_first_reappearance_l420_420632


namespace perimeter_triangle_VWX_l420_420373

def point (P : Type) := × P

variables (h_prism : ∀ (PQRSTU : point), PQRSTU.height = 20
  ∧ (∀ (base : tri), base.side_length = 10)
  ∧ (V W X : point),
    (V.point = midpoints P Q)
    ∧ (W.point = midpoints Q R)
    ∧ (X.point = midpoints R S)
)

theorem perimeter_triangle_VWX :
  ∀ (PQRSTU : point) (V W X : point),
    h_prism PQRSTU →
    (perimeter (triangle V W X) = 5 + 10 * sqrt 5) :=
by
  sorry

end perimeter_triangle_VWX_l420_420373


namespace remaining_hair_length_is_1_l420_420578

-- Variables to represent the inches of hair
variable (initial_length cut_length : ℕ)

-- Given initial length and cut length
def initial_length_is_14 (initial_length : ℕ) := initial_length = 14
def cut_length_is_13 (cut_length : ℕ) := cut_length = 13

-- Definition of the remaining hair length
def remaining_length (initial_length cut_length : ℕ) := initial_length - cut_length

-- Main theorem: Proving the remaining hair length is 1 inch
theorem remaining_hair_length_is_1 : initial_length_is_14 initial_length → cut_length_is_13 cut_length → remaining_length initial_length cut_length = 1 := by
  intros h1 h2
  rw [initial_length_is_14, cut_length_is_13] at *
  simp [remaining_length]
  sorry

end remaining_hair_length_is_1_l420_420578


namespace total_amount_proof_l420_420056

def total_shared_amount : ℝ :=
  let z := 250
  let y := 1.20 * z
  let x := 1.25 * y
  x + y + z

theorem total_amount_proof : total_shared_amount = 925 :=
by
  sorry

end total_amount_proof_l420_420056


namespace minimum_value_proof_l420_420933

variables {A B C : ℝ}
variable (triangle_ABC : 
  ∀ {A B C : ℝ}, 
  (A > 0 ∧ A < π / 2) ∧ 
  (B > 0 ∧ B < π / 2) ∧ 
  (C > 0 ∧ C < π / 2))

noncomputable def minimum_value (A B C : ℝ) :=
  3 * (Real.tan B) * (Real.tan C) + 
  2 * (Real.tan A) * (Real.tan C) + 
  1 * (Real.tan A) * (Real.tan B)

theorem minimum_value_proof (h : 
  ∀ (A B C : ℝ), 
  (1 / (Real.tan A * Real.tan B)) + 
  (1 / (Real.tan B * Real.tan C)) + 
  (1 / (Real.tan C * Real.tan A)) = 1) 
  : minimum_value A B C = 6 + 2 * Real.sqrt 3 + 2 * Real.sqrt 2 + 2 * Real.sqrt 6 :=
sorry

end minimum_value_proof_l420_420933


namespace interval_of_monotonic_increase_area_triangle_ABC_l420_420898

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (Real.sin x, -1)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, -1/2)
noncomputable def f (x : ℝ) : ℝ := 
  let a := vector_a x
  let b := vector_b x
  (a.1 + b.1, a.2 + b.2).1 * a.1 + (a.1 + b.1, a.2 + b.2).2 * a.2 - 2

theorem interval_of_monotonic_increase (k : ℤ) : 
  f (x) = Real.sin (2 * x - π / 6) ∧ 
  (f' (x) > 0 ↔ x ∈ [k * π - π / 6, k * π + π / 3]) := 
sorry

theorem area_triangle_ABC (a b c : ℝ) (A : ℝ) (h1 : a = Real.sqrt 3) (h2 : c = 1) (h3 : f A = 1) : 
  A = π / 3 ∧ b = 2 ∧ 
  S = (1/2) * b * c * Real.sin A → 
  S = Real.sqrt 3 / 2 := 
sorry

end interval_of_monotonic_increase_area_triangle_ABC_l420_420898


namespace hexagon_coloring_ways_l420_420622

-- Define the conditions
def hexagon_coloring_conditions (colors : List (List (Option ℕ))) : Prop :=
  ∃ R_pos color, colors.head.head = some R_pos ∧
  colors.head.length = 2 ∧
  colors.tail.head.length = 2 ∧
  colors.size = 3 ∧
  ∀ i j, 
    (colors[i][j] = some 0 → (colors[i][j+1] ≠ some 0 ∧ colors[i+1][j] ≠ some 0)) ∧
    (colors[i][j] = some 1 → (colors[i][j+1] ≠ some 1 ∧ colors[i+1][j] ≠ some 1)) ∧
    (colors[i][j] = some 2 → (colors[i][j+1] ≠ some 2 ∧ colors[i+1][j] ≠ some 2)) 

-- Main theorem
theorem hexagon_coloring_ways : ∃ n : ℕ, n = 2 ∧
  ∀ colors : List (List (Option ℕ)), hexagon_coloring_conditions colors → true :=
by
  sorry

end hexagon_coloring_ways_l420_420622


namespace exists_n0_series_sum_l420_420983

theorem exists_n0_series_sum (p : ℕ) (hp : 2 ≤ p) : ∃ n0 : ℕ, ∑ k in Finset.range (n0 + 1) \ k, (1 / (k * (k + 1)^((1 : ℝ) / p))) > (p : ℕ) :=
by
  sorry

end exists_n0_series_sum_l420_420983


namespace white_portion_area_l420_420740

theorem white_portion_area (height width : ℕ) (M A T H_white : ℕ) :
  height = 8 → width = 24 → M = 24 → A = 14 → T = 13 → H_white = 19 →
  (height * width) - (M + A + T + H_white) = 122 :=
by
  intros h w M_eq A_eq T_eq H_eq
  rw [h, w, M_eq, A_eq, T_eq, H_eq]
  norm_num

end white_portion_area_l420_420740


namespace largest_gold_coins_l420_420315

noncomputable def max_gold_coins (n : ℕ) : ℕ :=
  if h : ∃ k : ℕ, n = 13 * k + 3 ∧ n < 150 then
    n
  else 0

theorem largest_gold_coins : max_gold_coins 146 = 146 :=
by
  sorry

end largest_gold_coins_l420_420315


namespace combined_investment_yield_l420_420152

def stock_info := { yield: ℝ, market_value: ℝ }

def StockA : stock_info := { yield := 0.14, market_value := 500 }
def StockB : stock_info := { yield := 0.08, market_value := 750 }
def StockC : stock_info := { yield := 0.12, market_value := 1000 }

def tax_rate : ℝ := 0.02 
def commission_fee : ℝ := 50 

def calculate_net_yield (info : stock_info) : ℝ :=
  let initial_yield := info.yield * info.market_value
  let tax := tax_rate * initial_yield
  initial_yield - tax

def total_net_yield : ℝ :=
  calculate_net_yield StockA + calculate_net_yield StockB + calculate_net_yield StockC - 3 * commission_fee

def total_market_value : ℝ :=
  StockA.market_value + StockB.market_value + StockC.market_value

def overall_yield_percentage : ℝ :=
  (total_net_yield / total_market_value) * 100

theorem combined_investment_yield : overall_yield_percentage ≈ 4.22 := 
  by 
    have h : overall_yield_percentage = 95 / 2250 * 100 := 
      by 
        -- elaborate the steps, skipping with sorry
        sorry
    exact_mod_cast h

end combined_investment_yield_l420_420152


namespace find_length_of_BC_l420_420065

def circle_geometry_problem (O A B C D : Point) (BO : ℝ)
  (angle_ABO : ℝ) (arc_CD : ℝ) : Prop :=
  diameter O A D ∧ chord O A B C ∧ BO = 7 ∧ angle_ABO = 90 ∧ arc_CD = 90

theorem find_length_of_BC (O A B C D : Point) (BO : ℝ)
  (angle_ABO : ℝ) (arc_CD : ℝ) (h : circle_geometry_problem O A B C D BO angle_ABO arc_CD) :
  length B C = 7 := by
  sorry

end find_length_of_BC_l420_420065


namespace more_permutations_withP_l420_420981

open Finset

def hasPropertyP (n : ℕ) (σ : Fin (2 * n) → ℕ) : Prop :=
  ∃ i : Fin (2 * n - 1), abs (σ i - σ (i + 1)) = n

theorem more_permutations_withP (n : ℕ) (hn : 0 < n) :
  (univ.filter (hasPropertyP n)).card > (2 * n)! / 2 :=
by
  sorry

end more_permutations_withP_l420_420981


namespace vector_relation_l420_420036

variables (OA OB OC : Type)
variables (a b c : OA)

variables AC : ∀ (OC OA : OA), OC - OA 
variables BC : ∀ (OC OB : OB), OC - OB

variables (h : AC OC OA = 3 * (BC OC OB))

theorem vector_relation : c = - (1/2) * a + (3/2) * b :=
by
  sorry

end vector_relation_l420_420036


namespace find_a_l420_420504

-- Define the lines l1 and l2
def line1 (a : ℝ) : ℝ → ℝ → Prop := λ x y, a * x + 3 * y + 1 = 0
def line2 (a : ℝ) : ℝ → ℝ → Prop := λ x y, 2 * x + (a + 1) * y + 1 = 0

-- Condition for parallel lines (coefficients ratio must be equal)
def parallel_lines (a : ℝ) : Prop := a / 3 = 2 / (a + 1)

noncomputable def is_solution (a : ℝ) : Prop :=
  parallel_lines a ∧ ¬(line1 a = line2 a)

-- Proof goal
theorem find_a (a : ℝ) : is_solution a ↔ a = -3 :=
by sorry

end find_a_l420_420504


namespace grain_spilled_correct_l420_420371

variable (original_grain : ℕ) (remaining_grain : ℕ) (grain_spilled : ℕ)

theorem grain_spilled_correct : 
  original_grain = 50870 → remaining_grain = 918 → grain_spilled = original_grain - remaining_grain → grain_spilled = 49952 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end grain_spilled_correct_l420_420371


namespace correct_option_b_l420_420311

/-- Given the following conditions:
    (1) log4(3) ≠ 2 * log2(3)
    (2) (-a^2)^3 = -a^6
    (3) (sqrt(2) - 1)^0 ≠ 0
    (4) lg(2) + lg(3) ≠ lg(5)
    Prove that ( -a^2 )^3 = -a^6
-/
theorem correct_option_b (a : ℝ) (h1 : log 4 3 ≠ 2 * log 2 3)
  (h2 : (-a^2)^3 = -a^6) (h3 : (sqrt 2 - 1)^0 ≠ 0)
  (h4 : lg 2 + lg 3 ≠ lg 5) :
  (-a^2)^3 = -a^6 :=
by {
  exact h2
}

end correct_option_b_l420_420311


namespace problem_part_one_problem_part_two_l420_420485

def sequence (a : ℕ → ℤ) : Prop :=
  (a 1 = 1) ∧ (a 2 = 2) ∧ ∀ n, a (n + 2) = if (a n * a (n + 1)) % 2 = 0 then 5 * a (n + 1) - 3 * a n else a (n + 1) - a n

theorem problem_part_one (a : ℕ → ℤ) (h : sequence a) : ∃ inf_pos inf_neg : ℕ, inf_pos > 0 ∧ inf_neg > 0 ∧ 
  (∃ n : ℕ, a n > 0) ∧ (∃ n : ℕ, a n < 0) := sorry

theorem problem_part_two (a : ℕ → ℤ) (h : sequence a) : ∀ k : ℕ, k ≥ 2 → 7 ∣ a (2 ^ k - 1) := sorry

end problem_part_one_problem_part_two_l420_420485


namespace tangents_intersect_locus_l420_420470

variable {R : Type*} [LinearOrderedField R]

-- Definitions and conditions
def circle (x y : R) : Prop := x^2 + y^2 = 25
def point (x y : R) : Prop := (x, y)
def line_through_M (x0 y0 : R) : Prop := -2 * x0 + 3 * y0 = 25
def tangent_to_circle (x0 y0 x1 y1 : R) : Prop := x0 * x1 + y0 * y1 = 25
def point_Q_locus (x0 y0 : R) : Prop := 2 * x0 - 3 * y0 + 25 = 0

-- Problem statement
theorem tangents_intersect_locus : ∀ (x0 y0 x1 y1 : R),
  circle x1 y1 ∧ line_through_M x0 y0 ∧ tangent_to_circle x0 y0 x1 y1 →
  point_Q_locus x0 y0 :=
by sorry

end tangents_intersect_locus_l420_420470


namespace symmetric_points_distance_l420_420012

noncomputable def parabola (x : ℝ) : ℝ := -x^2 + 3

def is_symmetric_about (A B : ℝ × ℝ) : Prop :=
  let (xA, yA) := A
  let (xB, yB) := B
  xA + yA = 0 ∧ xB + yB = 0 ∧ xA ≠ xB ∧ yA ≠ yB

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  let (xA, yA) := A
  let (xB, yB) := B
  (sqrt ((xB - xA)^2 + (yB - yA)^2))

theorem symmetric_points_distance :
  ∀ (A B : ℝ × ℝ), 
  parabola A.1 = A.2 → 
  parabola B.1 = B.2 → 
  is_symmetric_about A B →
  distance A B = 3 * sqrt 2 :=
by
  intro A B h_parabola_A h_parabola_B h_symmetric
  sorry

end symmetric_points_distance_l420_420012


namespace roots_not_real_l420_420801

noncomputable def discriminant (a b c : ℂ) : ℂ := b^2 - 4 * a * c

theorem roots_not_real (m : ℂ) :
  let Δ := discriminant 5 (-7*complex.I) (-m)
  let root₁ := (7 * complex.I + complex.sqrt Δ) / 10
  let root₂ := (7 * complex.I - complex.sqrt Δ) / 10
  (root₁.im ≠ 0 ∨ root₂.im ≠ 0) :=
by {
  sorry
}

end roots_not_real_l420_420801


namespace open_box_volume_l420_420331

theorem open_box_volume :
  let original_length := 50
  let original_width := 36
  let square_side := 8
  let new_length := original_length - 2 * square_side
  let new_width := original_width - 2 * square_side
  let height := square_side
  let volume := new_length * new_width * height
  volume = 5440 :=
by 
  have original_length : ℕ := 50
  have original_width : ℕ := 36
  have square_side : ℕ := 8
  have new_length : ℕ := original_length - 2 * square_side
  have new_width : ℕ := original_width - 2 * square_side
  have height : ℕ := square_side
  have volume : ℕ := new_length * new_width * height
  show volume = 5440
  sorry

end open_box_volume_l420_420331


namespace problem_solution_l420_420181

-- Definitions based on conditions
def O : ℝ × ℝ := (0, 0)
def Q : ℝ × ℝ := (6, 0)
def P : ℝ × ℝ := (6, 6)

-- Define the rotation function
def rotate90ccw (point : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := point in (-y, x)

-- State the theorem
theorem problem_solution : rotate90ccw P = (-6, 6) :=
  by
    -- (Solution proof required here, skipped for now)
    sorry

end problem_solution_l420_420181


namespace problem_1_problem_2_problem_3_l420_420885

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 
  Real.log x + a * x^2 + b * x

theorem problem_1 :
  let a := 0
  let b := 1
  let f := f x 0 1
  ∀ x, f 1 = 1 → (∀ x, ∂f/∂x 1 = 2) → 2x - y - 1 = 0 := by
  sorry

theorem problem_2 :
  let a := (1/2)
  let b := -3
  let f := f x (1/2) -3
  ∀ x, (Real.log x + (1/2) * x^2 - 3 * x) ≥ -1 → 
  (∀ α, Real.tan α ≥ -1 → α ∈ [0, π/2) ∪ [3π/4, π)) := by
  sorry 

theorem problem_3 :
  let b := -1
  ∀ a ∈ [1/2, ∞), (∀ x1 x2 ∈ (0, ∞), x1 ≠ x2 → 
  (Real.log x1 + a * x2^2 - x1) - (Real.log x2 + a * x1^2 - x2) / (x1 - x2) > 1) := by
  sorry

end problem_1_problem_2_problem_3_l420_420885


namespace breadth_halved_of_percentage_change_area_l420_420631

theorem breadth_halved_of_percentage_change_area {L B B' : ℝ} (h : 0 < L ∧ 0 < B) 
  (h1 : L / 2 * B' = 0.5 * (L * B)) : B' = 0.5 * B :=
sorry

end breadth_halved_of_percentage_change_area_l420_420631


namespace complex_vector_PQ_l420_420048

theorem complex_vector_PQ (P Q : ℂ) (hP : P = 3 + 1 * I) (hQ : Q = 2 + 3 * I) : 
  (Q - P) = -1 + 2 * I :=
by sorry

end complex_vector_PQ_l420_420048


namespace range_of_m_l420_420878

noncomputable def f (x : ℝ) : ℝ := 2^(x + 1)
def g (x : ℝ) : ℝ := 2^x + 2^(-x)
def h (x : ℝ) : ℝ := 2^x - 2^(-x)
def p (t : ℝ) (m : ℝ) : ℝ := t^2 + 2 * m * t + m^2 - m + 1
def t (x : ℝ) : ℝ := 2^x - 2^(-x)
def phi (t : ℝ) : ℝ := -(t^2 + 2) / (2 * t)

theorem range_of_m (m : ℝ) : ∀ x ∈ set.Icc 1 2, p (t x) m ≥ m^2 - m - 1 ↔ m ≥ -17/12 :=
by sorry

end range_of_m_l420_420878


namespace pyramid_face_area_total_l420_420236

theorem pyramid_face_area_total 
  (base_edge : ℝ) (lateral_edge : ℝ) 
  (h_base_edge : base_edge = 8) 
  (h_lateral_edge : lateral_edge = 7) : 
  4 * (1 / 2 * base_edge * real.sqrt (lateral_edge^2 - (base_edge / 2)^2)) = 16 * real.sqrt 33 :=
by
  sorry

end pyramid_face_area_total_l420_420236


namespace point_in_third_quadrant_cos_sin_l420_420637

theorem point_in_third_quadrant_cos_sin (P : ℝ × ℝ) (hP : P = (Real.cos (2009 * Real.pi / 180), Real.sin (2009 * Real.pi / 180))) :
  P.1 < 0 ∧ P.2 < 0 :=
by
  sorry

end point_in_third_quadrant_cos_sin_l420_420637


namespace small_rectangle_perimeter_eq_15_l420_420391

noncomputable def small_rectangle_perimeter
(w : ℝ) (L : ℝ) : ℝ :=
  2 * L + 2 * w

noncomputable def large_square_perimeter
(w : ℝ) : ℝ :=
  4 * 5 * w

theorem small_rectangle_perimeter_eq_15
  (w L : ℝ)
  (h1 : large_square_perimeter(w) = small_rectangle_perimeter(w, L) + 10)
  (h2 : 5 * w = large_square_perimeter(w) / 4) :
  small_rectangle_perimeter(w, L) = 15 :=
by
  sorry

end small_rectangle_perimeter_eq_15_l420_420391


namespace pyramid_total_area_l420_420224

/-- The total area of the four triangular faces of a right, square-based pyramid
whose base edges measure 8 units and lateral edges measure 7 units is 16√33. -/
theorem pyramid_total_area :
  let base_edge := 8
  let lateral_edge := 7
  4 * (1 / 2 * base_edge * Real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 16 * Real.sqrt 33 := 
by
  let base_edge := 8
  let lateral_edge := 7
  have h1 : 4 * (1 / 2 * base_edge * Real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 
              4 * (1 / 2 * 8 * Real.sqrt (49 - 16)) := 
    by sorry
  have h2 : 4 * (1 / 2 * 8 * Real.sqrt (49 - 16)) = 4 * (4 * Real.sqrt 33) := 
    by sorry
  have h3 : 4 * (4 * Real.sqrt 33) = 16 * Real.sqrt 33 := 
    by sorry
  exact eq.trans (eq.trans h1 h2) h3

end pyramid_total_area_l420_420224


namespace number_of_subsets_of_P_l420_420032

def P : Set ℝ := {x | ∫ t in 0..x, (3 * t^2 - 8 * t + 3) = 0 ∧ x > 0}

theorem number_of_subsets_of_P : ∃ P : Set ℝ, (∀ x, x ∈ P ↔ (∫ t in 0..x, (3 * t^2 - 8 * t + 3) = 0 ∧ x > 0)) ∧ (P = {1, 3}) ∧ (2 ^ 2 = 4) :=
by
  use { 1, 3 }
  split
  { intros x
    split
    { intro hx
      sorry }, -- proof needed to show x ∈ P ↔ (∫ t in 0..x, (3 * t^2 - 8 * t + 3) = 0 ∧ x > 0)
    { intro hx
      sorry } -- proof needed to show x ∈ {1, 3}↔ (∫ t in 0..x, (3 * t^2 - 8 * t + 3) = 0 ∧ x > 0) 
  },
  split
  { reflexivity }, -- P = {1, 3}
  { rfl } -- 2^2 = 4 

end number_of_subsets_of_P_l420_420032


namespace largest_number_of_gold_coins_l420_420324

theorem largest_number_of_gold_coins 
  (num_friends : ℕ)
  (extra_coins : ℕ)
  (total_coins : ℕ) :
  num_friends = 13 →
  extra_coins = 3 →
  total_coins < 150 →
  ∀ k : ℕ, total_coins = num_friends * k + extra_coins →
  total_coins ≤ 146 :=
by
  sorry

end largest_number_of_gold_coins_l420_420324


namespace sum_odd_terms_zero_l420_420538

-- Let’s define the problem statement in Lean
def g (x : ℂ) (n : ℕ) : ℂ := (1 + x^3)^n
def b (i : ℕ) (n : ℕ) : ℂ := (g x n).coeff i

theorem sum_odd_terms_zero (n : ℕ) :
  let t := ∑ k in range (n + 1), if (3 * k + 1) < 3 * n then b (3 * k + 1)  (3 * k + 3)  (3 * n + 2)  else 0  + b (6) (13) (4) + b ( 12) + b  ∈  set (3 * k + 2) { - b  + (3 * n + 1 ).coeff.coeff : ℕ},
  t = 0 := sorry

end sum_odd_terms_zero_l420_420538


namespace proof_cos_value_l420_420863

noncomputable def f (α : Real) : Real := 
  (tan (π - α) * cos (2 * π - α) * sin (π / 2 + α)) / cos (-α - π)

theorem proof_cos_value (α : Real) (h1 : f α = 4 / 5) (h2 : π / 2 < α ∧ α < π) :
  cos (2 * α + π / 3) = (24 * Real.sqrt 3 - 7) / 50 :=
  sorry

end proof_cos_value_l420_420863


namespace area_CEF_l420_420080

theorem area_CEF (ABC : Triangle) (E F : Point) 
  (h1 : trisects E ABC.A ABC.C)
  (h2 : trisects F ABC.A ABC.B) 
  (h3 : area ABC = 27) : 
  area (Triangle.mk ABC.C E F) = 3 :=
sorry

end area_CEF_l420_420080


namespace number_of_girls_l420_420514

theorem number_of_girls 
  (X Y : ℕ) 
  (h1 : X + Y = 400) 
  (h2 : 0.60 * X + 0.80 * Y = 260) : 
  Y = 100 :=
sorry

end number_of_girls_l420_420514


namespace luke_skee_ball_tickets_l420_420990

theorem luke_skee_ball_tickets :
  ∀ (tickets_whack_a_mole tickets_per_candy candies : ℕ),
    tickets_whack_a_mole = 2 →
    tickets_per_candy = 3 →
    candies = 5 →
    ∃ (tickets_skee_ball : ℕ), tickets_skee_ball = (candies * tickets_per_candy) - tickets_whack_a_mole :=
begin
  intros tickets_whack_a_mole tickets_per_candy candies H1 H2 H3,
  use 13,
  rw [H1, H2, H3],
  norm_num,
  exact eq.refl 13,
end

end luke_skee_ball_tickets_l420_420990


namespace perp_and_equal_l420_420509

-- Definitions of given conditions
variables {A B C O I D E : Point}
variables {α β γ : ℝ}

-- Assume that A, B, C form a triangle with angle C = 30 degrees
variable (hABC : ∠ A B C = 30)

-- Assume that O is the circumcenter of triangle ABC
variable (hO : is_circumcenter O A B C)

-- Assume that I is the incenter of triangle ABC
variable (hI : is_incenter I A B C)

-- Assume that D is on AC and E is on BC such that AD = BE = AB
variable (hD : on_line D A C)
variable (hE : on_line E B C)
variable (hDE : dist A D = dist B E ∧ dist A D = dist A B)

-- The theorem to be proved
theorem perp_and_equal (hABC: ∠ A B C = 30) (hO : is_circumcenter O A B C) 
(hI : is_incenter I A B C) (hD : on_line D A C) (hE : on_line E B C) 
(hDE : dist A D = dist B E ∧ dist A D = dist A B) :
  (perp O I D E) ∧ (dist O I = dist D E) :=
  sorry

end perp_and_equal_l420_420509


namespace num_ways_correct_l420_420386

def girls := ℕ
def songs := ℕ

structure likes :=
(Amy_likes : songs → Prop)
(Beth_likes : songs → Prop)
(Jo_likes : songs → Prop)
(noyose_liked_by_all_three : ∀ (s : songs), ¬ (likes.Amy_likes s ∧ likes.Beth_likes s ∧ likes.Jo_likes s))
(exists_song_Amy_Beth_not_Jo : ∃ (s : songs), likes.Amy_likes s ∧ likes.Beth_likes s ∧ ¬ likes.Jo_likes s)
(exists_song_Beth_Jo_not_Amy : ∃ (s : songs), likes.Beth_likes s ∧ likes.Jo_likes s ∧ ¬ likes.Amy_likes s)
(exists_song_Jo_Amy_not_Beth : ∃ (s : songs), likes.Jo_likes s ∧ likes.Amy_likes s ∧ ¬ likes.Beth_likes s)

def num_ways_Amy_Beth_Jo_like_songs : ℕ :=
sorry

theorem num_ways_correct :
  num_ways_Amy_Beth_Jo_like_songs = 132 :=
sorry

end num_ways_correct_l420_420386


namespace sum_of_bn_l420_420871

noncomputable def S (n : ℕ) : ℚ :=
  (n^2 + n) / 2
  
noncomputable def a (n : ℕ) : ℚ :=
  S n - S (n - 1)

noncomputable def b (n : ℕ) : ℚ :=
  a n * 3^(a n)

noncomputable def T (n : ℕ) : ℚ :=
  (3 / 4) + ((n / 2) - (1 / 4)) * 3^(n + 1)

theorem sum_of_bn (n : ℕ) (hn : n > 0) : 
  ∑ i in finset.range n, b (i + 1) = T n := by 
  sorry

end sum_of_bn_l420_420871


namespace relationship_among_a_b_c_l420_420840

noncomputable def a := (0.3)^3
noncomputable def b := Real.logBase 3 0.3
noncomputable def c := Real.logBase 0.3 3

theorem relationship_among_a_b_c : a > c ∧ c > b :=
by
  sorry

end relationship_among_a_b_c_l420_420840


namespace area_G1G2G3_l420_420543

-- Define the point and area structures first
structure Point where
  x : ℝ
  y : ℝ

-- Define basic properties of triangles
structure Triangle where
  A B C : Point

-- Define a function to calculate areas if necessary
def area (T : Triangle) : ℝ := sorry

-- Define midpoints and centroids
def midpoint (A B : Point) : Point :=
  { x := (A.x + B.x) / 2, y := (A.y + B.y) / 2 }

def centroid (T : Triangle) : Point :=
  { x := (T.A.x + T.B.x + T.C.x) / 3, y := (T.A.y + T.B.y + T.C.y) / 3 }

-- Given conditions and the required proof
theorem area_G1G2G3 {ABC : Triangle} (h₁ : area ABC = 36)
  (P : Point) (hP : P = midpoint ABC.B ABC.C)
  (G1 G2 G3 : Point)
  (hG1 : G1 = centroid ⟨P, ABC.B, ABC.C⟩)
  (hG2 : G2 = centroid ⟨P, ABC.C, ABC.A⟩)
  (hG3 : G3 = centroid ⟨P, ABC.A, ABC.B⟩) :
  area ⟨G1, G2, G3⟩ = 4 :=
by
  sorry

end area_G1G2G3_l420_420543


namespace geometric_locus_of_projections_l420_420382

-- We define the conditions and the goal in Lean
def point := ℝ × ℝ
def segment (A B : point) : set point := { P | ∃ t: ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) * A + t * B }

def circle (O : point) (r : ℝ) : set point :=
  { Q | (Q.1 - O.1)^2 + (Q.2 - O.2)^2 = r^2 }

def O : point := (0, 0)
def A : point := (1, 0)
def B : point := (0, 1)

def diameter_circle (A O : point) : set point :=
  circle O (1/2 * ((A.1 - O.1)^2 + (A.2 - O.2)^2)^(1/2))

theorem geometric_locus_of_projections (A B O : point) :
  ∀ P ∈ segment A B,
  ∃ rA rB,
  (rA = 1/2 * ((A.1 - O.1)^2 + (A.2 - O.2)^2)^(1/2)) ∧
  (rB = 1/2 * ((B.1 - O.1)^2 + (B.2 - O.2)^2)^(1/2)) ∧
  ∃ (locus : set point),
  locus = { Q | (Q.1 - O.1)^2 + (Q.2 - O.2)^2 = rA^2 ∧
             ((Q.1 - O.1)^2 + (Q.2 - O.2)^2 ≠ rB^2) } :=
by sorry

end geometric_locus_of_projections_l420_420382


namespace pyramid_face_area_total_l420_420233

theorem pyramid_face_area_total 
  (base_edge : ℝ) (lateral_edge : ℝ) 
  (h_base_edge : base_edge = 8) 
  (h_lateral_edge : lateral_edge = 7) : 
  4 * (1 / 2 * base_edge * real.sqrt (lateral_edge^2 - (base_edge / 2)^2)) = 16 * real.sqrt 33 :=
by
  sorry

end pyramid_face_area_total_l420_420233


namespace geometric_sequence_common_ratio_and_general_formula_l420_420944

variable (a : ℕ → ℝ)

theorem geometric_sequence_common_ratio_and_general_formula (h₁ : a 1 = 1) (h₃ : a 3 = 4) : 
  (∃ q : ℝ, q = 2 ∨ q = -2 ∧ (∀ n : ℕ, a n = 2^(n-1) ∨ a n = (-2)^(n-1))) := 
by
  sorry

end geometric_sequence_common_ratio_and_general_formula_l420_420944


namespace total_area_of_triangular_faces_l420_420203

noncomputable def pyramid_base_edge : ℝ := 8
noncomputable def pyramid_lateral_edge : ℝ := 7

structure Pyramid where
  base_edge : ℝ
  lateral_edge : ℝ

def myPyramid : Pyramid :=
{ base_edge := pyramid_base_edge,
  lateral_edge := pyramid_lateral_edge }

theorem total_area_of_triangular_faces :
  4 * (1 / 2) * myPyramid.base_edge * (Real.sqrt (myPyramid.lateral_edge^2 - (myPyramid.base_edge / 2)^2)) = 16 * Real.sqrt 33 :=
by
  sorry

end total_area_of_triangular_faces_l420_420203


namespace ratio_Sid_Max_courses_l420_420998

variables (Courses_Max Courses_Total : ℕ)
variables (m : ℕ)

-- Given conditions
def Max_courses_attended := Courses_Max = 40
def Total_courses_attended := Courses_Total = 200
def Sid_courses_attended := Total_courses_attended ∧  ∃ m, (Courses_Total - Courses_Max = m * Courses_Max) -- Sid's courses

-- Prove the ratio
theorem ratio_Sid_Max_courses
  (h1 : Max_courses_attended)
  (h2 : Total_courses_attended)
  (h3 : Sid_courses_attended) :
  (Courses_Total - Courses_Max) / Courses_Max = 4 := 
sorry

end ratio_Sid_Max_courses_l420_420998


namespace breadth_remains_the_same_l420_420627

variable (L B : ℝ)

theorem breadth_remains_the_same 
  (A : ℝ) (hA : A = L * B) 
  (L_half : ℝ) (hL_half : L_half = L / 2) 
  (B' : ℝ)
  (A' : ℝ) (hA' : A' = L_half * B') 
  (hA_change : A' = 0.5 * A) : 
  B' = B :=
  sorry

end breadth_remains_the_same_l420_420627


namespace train_length_calculation_l420_420753

def speed_km_per_hr : ℝ := 60
def time_sec : ℝ := 9
def length_of_train : ℝ := 150

theorem train_length_calculation :
  (speed_km_per_hr * 1000 / 3600) * time_sec = length_of_train := by
  sorry

end train_length_calculation_l420_420753


namespace find_common_difference_sum_first_n_terms_l420_420938

open BigOperators

-- Definitions based on problem conditions
def a (n : ℕ) (d : ℕ) : ℕ := 1 + d * (n - 1)

-- Statement to prove the common difference of the arithmetic sequence is 2
theorem find_common_difference (d : ℕ) : 
  a 1 d = 1 ∧ a 1 d, a 2 d, a 5 d form a geometric sequence with a common ratio ≠ 1 →
  d = 2 := by
  sorry

-- Sum calculation statement
theorem sum_first_n_terms (n : ℕ) : 
  let d := 2 in
  ∑ i in Finset.range n, 1 / (a i d * a (i+1) d) = n / (2 * n + 1) := by
  sorry

end find_common_difference_sum_first_n_terms_l420_420938


namespace compare_ln2_values_l420_420839

theorem compare_ln2_values :
  let a := Real.log 2,
      b := Real.exp (-Real.log 2),
      c := Real.log (Real.log 2) / Real.log 10 in
  a > b ∧ b > c :=
by
  let a := Real.log 2
  let b := Real.exp (- Real.log 2)
  let c := Real.log (Real.log 2) / Real.log 10
  sorry

end compare_ln2_values_l420_420839


namespace angle_BEA_l420_420079

noncomputable def triangle_ABC (A B C E : Point) : Prop :=
  is_on_segment B C E ∧ 
  angle B A E = 20 ∧ 
  angle E A C = 40

theorem angle_BEA (A B C E : Point) (h : triangle_ABC A B C E) : 
  angle B E A = 80 := 
sorry

end angle_BEA_l420_420079


namespace pattern_D_cannot_form_tetrahedron_l420_420800

theorem pattern_D_cannot_form_tetrahedron :
  (¬ ∃ (f : ℝ × ℝ → ℝ × ℝ),
      f (0, 0) = (1, 1) ∧ f (1, 0) = (1, -1) ∧ f (2, 0) = (-1, 1) ∧ f (3, 0) = (-1, -1)) :=
by
  -- proof will go here
  sorry

end pattern_D_cannot_form_tetrahedron_l420_420800


namespace fold_square_find_FD_length_l420_420519

/-- Given $ABCD$ is a square with side 8 cm and $C$ folded to $E$ which is one-third along $\overline{AD}$ from $A$, prove that the length $\overline{FD}$ of the crease is $\frac{32}{9}$ cm. -/
theorem fold_square_find_FD_length :
  let AD := 8
  let E := AD / 3
  let CD := 8
  let ED := 8 / 3
  let FD := 32 / 9
  in ED = 8 / 3 ∧
     ∃ (x : ℝ), x = 32 / 9 ∧ (8 - x)^2 = x^2 + (8 / 3)^2 :=
by
  sorry

end fold_square_find_FD_length_l420_420519


namespace bus_speed_with_stoppages_l420_420422

theorem bus_speed_with_stoppages :
  ∀ (speed_excluding_stoppages : ℕ) (stop_minutes : ℕ) (total_minutes : ℕ)
  (speed_including_stoppages : ℕ),
  speed_excluding_stoppages = 80 →
  stop_minutes = 15 →
  total_minutes = 60 →
  speed_including_stoppages = (speed_excluding_stoppages * (total_minutes - stop_minutes) / total_minutes) →
  speed_including_stoppages = 60 := by
  sorry

end bus_speed_with_stoppages_l420_420422


namespace pyramid_four_triangular_faces_area_l420_420254

theorem pyramid_four_triangular_faces_area 
  (base_edge : ℝ) (lateral_edge : ℝ) 
  (h_base : base_edge = 8)
  (h_lateral : lateral_edge = 7) :
  let h := Real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)
  let triangle_area := (1 / 2) * base_edge * h
  let total_area := 4 * triangle_area
  total_area = 16 * Real.sqrt 33 :=
by
  -- Definitions to introduce local values
  let half_base := base_edge / 2
  let h := Real.sqrt (lateral_edge ^ 2 - half_base ^ 2)
  let triangle_area := (1 / 2) * base_edge * h
  let total_area := 4 * triangle_area
  -- Assertion to compare calculated total area with given correct answer
  have h_eq : h = Real.sqrt 33 := by sorry
  have triangle_area_eq : triangle_area = 4 * Real.sqrt 33 := by sorry
  have total_area_eq : total_area = 16 * Real.sqrt 33 := by sorry
  exact total_area_eq

end pyramid_four_triangular_faces_area_l420_420254


namespace max_abs_xk_l420_420103

theorem max_abs_xk (n k : ℕ) (x : ℕ → ℝ) (hn : n ≥ 2) (hk : 1 ≤ k) (hnk : k ≤ n)
    (h : ∑ i in finset.range n, x i ^ 2 + ∑ i in finset.range (n - 1), x i * x (i + 1) = 1) :
    |x k| ≤ Real.sqrt (2 * k * (n + 1 - k) / (n + 1)) := by
  sorry

end max_abs_xk_l420_420103


namespace probability_odd_number_die_l420_420654

theorem probability_odd_number_die :
  let total_outcomes := 6
  let favorable_outcomes := 3
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ) = 1 / 2 :=
by
  sorry

end probability_odd_number_die_l420_420654


namespace edge_length_of_smaller_cubes_l420_420901

noncomputable def edge_length_of_cubic_box : ℝ := 1
noncomputable def number_of_smaller_cubes : ℝ := 125

theorem edge_length_of_smaller_cubes :
  ∃ (n : ℝ), n ≈ (edge_length_of_cubic_box / (number_of_smaller_cubes)^(1/3)) ∧ n ≈ 0.2 := sorry

end edge_length_of_smaller_cubes_l420_420901


namespace trapezoid_mid_segment_length_l420_420607

theorem trapezoid_mid_segment_length (a c : ℝ) (hEF : ℝ) 
  (h_trapezoid : ∀ h, ∃ h',  1 / 2 * (a + c) * h = h  * ((1 / 2) * a + (1 / 2) * c) / 2 :=
begin
  sorry
end    


end trapezoid_mid_segment_length_l420_420607


namespace cos_of_arcsin_l420_420400

-- Define the hypothesis and the corresponding proof problem
theorem cos_of_arcsin :
  ∀ (θ : ℝ), (sin θ = 8 / 17) → (θ = Real.arcsin (8 / 17)) → Real.cos (Real.arcsin (8 / 17)) = 15 / 17 :=
by
  intros θ hsin harcsin
  -- proof
  sorry

end cos_of_arcsin_l420_420400


namespace sum_of_valid_y_l420_420195

theorem sum_of_valid_y : 
  let condition (y : ℝ) := (median {3, 7, 11, 21, y} = (3 + 7 + 11 + 21 + y) / 5) ∧ y < 15 in
  ∑ (y : ℝ) in {3, 7, 11, 21, -7, 13} ∩ {y | condition y}, y = 6 :=
by
  sorry

end sum_of_valid_y_l420_420195


namespace residue_7_1234_mod_13_l420_420686

theorem residue_7_1234_mod_13 :
  (7 : ℤ) ^ 1234 % 13 = 12 :=
by
  -- given conditions as definitions
  have h1 : (7 : ℤ) % 13 = 7 := by norm_num
  
  -- auxiliary calculations
  have h2 : (49 : ℤ) % 13 = 10 := by norm_num
  have h3 : (100 : ℤ) % 13 = 9 := by norm_num
  have h4 : (81 : ℤ) % 13 = 3 := by norm_num
  have h5 : (729 : ℤ) % 13 = 1 := by norm_num

  -- the actual problem we want to prove
  sorry

end residue_7_1234_mod_13_l420_420686


namespace fraction_sum_equals_decimal_l420_420403

theorem fraction_sum_equals_decimal : 
  (3 / 30 + 9 / 300 + 27 / 3000 = 0.139) :=
by sorry

end fraction_sum_equals_decimal_l420_420403


namespace johns_improvement_l420_420085

-- Declare the variables for the initial and later lap times.
def initial_minutes : ℕ := 50
def initial_laps : ℕ := 25
def later_minutes : ℕ := 54
def later_laps : ℕ := 30

-- Calculate the initial and later lap times in seconds, and the improvement.
def initial_lap_time_seconds := (initial_minutes * 60) / initial_laps 
def later_lap_time_seconds := (later_minutes * 60) / later_laps
def improvement := initial_lap_time_seconds - later_lap_time_seconds

-- State the theorem to prove the improvement is 12 seconds per lap.
theorem johns_improvement : improvement = 12 := by
  sorry

end johns_improvement_l420_420085


namespace rocky_first_round_knockouts_in_middleweight_l420_420415

-- Conditions
def total_fights_middleweight : Nat := 100
def win_percentage_middleweight : ℕ → ℚ := λ total_fights, 0.70 * total_fights
def knockout_percentage_wins_middleweight : ℚ → ℚ := λ wins, 0.60 * wins
def first_round_knockout_percentage : ℚ → ℚ := λ knockouts, 0.20 * knockouts

-- Proof problem
theorem rocky_first_round_knockouts_in_middleweight : 
  first_round_knockout_percentage (knockout_percentage_wins_middleweight (win_percentage_middleweight total_fights_middleweight)).nat_floor = 8 :=
by sorry

end rocky_first_round_knockouts_in_middleweight_l420_420415


namespace monotone_increasing_interval_solve_inequality_l420_420482

noncomputable def f (x : ℝ) : ℝ :=
  a * x ^ 3 + c * x

-- Given conditions
def graph_symmetrical_about_origin (a c : ℝ) : Prop :=
  ∀ x : ℝ, f(-x) = -f(x)

def min_value_at_one (a c : ℝ) : Prop :=
  f(1) = -2

-- Monotonically increasing interval statement
theorem monotone_increasing_interval (a c : ℝ) 
  (h1 : graph_symmetrical_about_origin a c)
  (h2 : min_value_at_one a c) :
  { x // x < -1 ∨ x > 1 } :=
sorry

-- Inequality solving
theorem solve_inequality (a c m : ℝ) 
  (h1 : graph_symmetrical_about_origin a c)
  (h2 : min_value_at_one a c)
  (h3 : ∀ x : ℝ, f(x) > 5 * m * x^2 - (4 * m^2 + 3) * x) : 
  { x // (m = 0 ∧ x > 0) ∨ 
        (m > 0 ∧ (x > 4 * m ∨ 0 < x ∧ x < m)) ∨ 
        (m < 0 ∧ (x > 0 ∨ 4 * m < x ∧ x < m)) } :=
sorry

end monotone_increasing_interval_solve_inequality_l420_420482


namespace max_number_reusable_cards_l420_420650

/-- Define digits that form valid numbers when reversed -/
def is_reversible_digit (d : ℕ) : Prop :=
  d = 0 ∨ d = 1 ∨ d = 8 ∨ d = 6 ∨ d = 9

/-- Define a valid three-digit number when reversed -/
def is_reversible_number (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n % 100) / 10
  let d3 := n % 10
  is_reversible_digit d1 ∧ is_reversible_digit d2 ∧ is_reversible_digit d3 ∧
  d1 ≠ 0

/-- Count the total number of valid three-digit numbers with reversible digits -/
noncomputable def count_valid_reversible_numbers : ℕ :=
  (Finset.filter is_reversible_number (Finset.range 1000)).card

/-- List of self-reversible three-digit numbers -/
def self_reversible_numbers : List ℕ := 
  [101, 111, 181, 808, 818, 888, 609, 619, 689, 906, 916, 986]

/-- Define the count of self-reversible numbers -/
def count_self_reversible_numbers : ℕ :=
  self_reversible_numbers.length

/-- Define the total number of pairs excluding self-reversible numbers -/
noncomputable def count_reversible_pairs : ℕ :=
  (count_valid_reversible_numbers - count_self_reversible_numbers) / 2

/-- Define the maximum number of reusable cards -/
noncomputable def max_reusable_cards : ℕ :=
  count_reversible_pairs + count_self_reversible_numbers

/-- The final statement to prove: -/
theorem max_number_reusable_cards : max_reusable_cards = 56 :=
by
  sorry

end max_number_reusable_cards_l420_420650


namespace series_less_than_one_l420_420792

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def sum_fractions (n : Nat) : Real :=
  match n with
  | 0 => 0
  | n + 1 => sum_fractions n + n / factorial (n + 1)

theorem series_less_than_one :
  sum_fractions 2012 < 1 :=
  sorry

end series_less_than_one_l420_420792


namespace period_of_f_cos_2alpha_l420_420446

-- Definitions for the problem
def f (x : ℝ) : ℝ := 4 * Real.cos x * Real.sin (x + Real.pi / 6) - 1

-- Part (1): Prove that the period of f(x) is π
theorem period_of_f : (∀ x, f(x + Real.pi) = f(x)) := 
by sorry

-- Part (2): Prove that if f(α) = 6/5, where α ∈ (0, π/4), then cos 2α = (3 + 4 * sqrt 3) / 10
theorem cos_2alpha (α : ℝ) (h1 : 0 < α) (h2 : α < Real.pi / 4) (h3 : f(α) = 6 / 5) : 
  Real.cos (2 * α) = (3 + 4 * Real.sqrt 3) / 10 := 
by sorry

end period_of_f_cos_2alpha_l420_420446


namespace possible_integer_values_of_x_l420_420497

theorem possible_integer_values_of_x :
  {x : ℕ | 196 < x ∧ x ≤ 225}.card = 29 := by
  sorry

end possible_integer_values_of_x_l420_420497


namespace lemuel_total_points_l420_420534

theorem lemuel_total_points (two_point_shots : ℕ) (three_point_shots : ℕ) (points_from_two : ℕ) (points_from_three : ℕ) :
  two_point_shots = 7 →
  three_point_shots = 3 →
  points_from_two = 2 →
  points_from_three = 3 →
  two_point_shots * points_from_two + three_point_shots * points_from_three = 23 :=
by
  sorry

end lemuel_total_points_l420_420534


namespace tangent_lines_from_point_to_circle_l420_420379

noncomputable def point := (5, -2 : ℝ)
noncomputable def circle_eq (x y : ℝ) := x^2 + y^2 - 4 * x - 4 * y - 1 = 0

theorem tangent_lines_from_point_to_circle : 
  ∃ k1 k2: ℝ, (7 * k1 + 24 * k2 + 13 = 0 ∨ k1 = 5) ∧ 
             (∀ x y : ℝ, circle_eq x y → k1 * x + y + k2 = 0) := sorry

end tangent_lines_from_point_to_circle_l420_420379


namespace combined_set_median_l420_420489

theorem combined_set_median (a b : ℝ) (h1 : (3 + 2 * a + 5 + b) / 4 = 6)
  (h2 : (a + 4 + 2 * b) / 3 = 6) : 
  let s := multiset.map rational.to_real (3 :: 2 * a :: 5 :: b :: a :: 4 :: 2 * b :: ([])) in
  s.median = 5 :=
by
  sorry

end combined_set_median_l420_420489


namespace ellipse_problem_l420_420449

open Real

theorem ellipse_problem 
    (a b : ℝ) 
    (h₁ : a > b) 
    (h₂ : b > 0) 
    (ecc : a^2 = 3 * b^2) 
    (M : ℝ × ℝ) 
    (hM : M = (0, 1)) 
    (k₁ k₂ : ℝ)
    (hAB_symmetric : A = (-B.1, -B.2)) 
    (hM_on_ellipse : (0^2 / a^2 + 1^2 / b^2 = 1)) 
    (h_k_sum : k₁ + k₂ = 3) :
    (k₁ * k₂ = -1/3) ∧ 
    (line_AB_passes_fixed_point : passes_through_fixed_point) ∧ 
    (k_range : k ∈ set.Ioo (-∞) (-12/23) ∪ set.Ioo 0 3 ∪ set.Ioo 3 ∞) :=
  sorry

end ellipse_problem_l420_420449


namespace ratio_of_second_drink_to_first_remaining_l420_420532

-- Definitions based on the conditions
def initial_water_volume : ℕ := 4                 -- Condition 1
def first_drink_fraction : ℚ := 1 / 4            -- Condition 2
def remaining_after_first : ℕ := initial_water_volume - nat.floor (first_drink_fraction * initial_water_volume) -- Condition 3
def remaining_after_second : ℕ := 1              -- Condition 4
def second_drink_volume : ℕ := remaining_after_first - remaining_after_second

-- Theorem statement
theorem ratio_of_second_drink_to_first_remaining : second_drink_volume.to_rat / remaining_after_first.to_rat = 2/3 :=
by
  sorry

end ratio_of_second_drink_to_first_remaining_l420_420532


namespace height_difference_l420_420909

variables {Ruby Pablo Janet Charlene Xavier Paul : ℝ}

-- Conditions
def janet_height := Janet = 62
def charlene_height := Charlene = 2 * Janet
def pablo_height := Pablo = Charlene + 70
def ruby_height := Ruby = Pablo - 2
def xavier_height1 := Xavier = Charlene + 84
def xavier_height2 := Xavier = Paul - 38
def paul_height := Paul = Ruby + 45

-- Goal
theorem height_difference : janet_height ∧ charlene_height ∧ pablo_height ∧ ruby_height ∧ xavier_height1 ∧ xavier_height2 ∧ paul_height → (Xavier - Ruby = 7) :=
by
  intros
  sorry

end height_difference_l420_420909


namespace sum_of_squares_nonzero_l420_420679

theorem sum_of_squares_nonzero {a b : ℝ} (h : a^2 + b^2 ≠ 0) : a ≠ 0 ∨ b ≠ 0 :=
sorry

end sum_of_squares_nonzero_l420_420679


namespace income_of_first_member_l420_420176

-- Define the number of family members
def num_members : ℕ := 4

-- Define the average income per member
def avg_income : ℕ := 10000

-- Define the known incomes of the other three members
def income2 : ℕ := 15000
def income3 : ℕ := 6000
def income4 : ℕ := 11000

-- Define the total income of the family
def total_income : ℕ := avg_income * num_members

-- Define the total income of the other three members
def total_other_incomes : ℕ := income2 + income3 + income4

-- Define the income of the first member
def income1 : ℕ := total_income - total_other_incomes

-- The theorem to prove
theorem income_of_first_member : income1 = 8000 := by
  sorry

end income_of_first_member_l420_420176


namespace disproving_form_of_polynomials_l420_420100

def f (x y z : ℝ) : ℝ := x^2 + y^2 + z^2 + x * y * z

theorem disproving_form_of_polynomials (a b c : ℝ → ℝ → ℝ → ℝ) :
  (∀ x y z, f (a x y z) (b x y z) (c x y z) = f x y z) →
  ¬(∃ px py pz : ℝ → ℝ → ℝ → ℝ, 
    (px = λ x y z, x ∨ px = λ x y z, -x) ∧
    (py = λ x y z, y ∨ py = λ x y z, -y) ∧
    (pz = λ x y z, z ∨ pz = λ x y z, -z) ∧ 
    even (card {px, py, pz} (λ f, f = (λ x y z, -x)))) :=
by
  sorry

end disproving_form_of_polynomials_l420_420100


namespace greatest_number_in_set_l420_420337

theorem greatest_number_in_set (S : Set ℕ) (h₁ : ∀ n ∈ S, n = 108 + (n - 108) / 8 * 8)
  (h₂ : S.card = 100) : 900 ∈ S ∧ ∀ m ∈ S, m ≤ 900 :=
by
  sorry

end greatest_number_in_set_l420_420337


namespace smallest_k_l420_420537

def S_n (n : ℕ) (hn : 1 < n) : Set ℕ := { x | ∃ k : ℕ, x = n^k }

theorem smallest_k (n : ℕ) (hn : 1 < n) : 
  ∃! k : ℕ, (∃ m : ℕ, ∃ a b : Fin k → ℕ, (∀ i, a i ∈ S_n n hn ∧ b i ∈ S_n n hn ∧ (Sum (Finset.univ.image a) = m) = (Sum (Finset.univ.image b) = m) ∧ a ≠ b) ∧ k = n + 1) :=
begin
  sorry
end

end smallest_k_l420_420537


namespace exists_subset_with_intersections_l420_420088

theorem exists_subset_with_intersections
  (n k : ℕ)
  (h1 : n ≥ k)
  (h2 : k ≥ 2)
  (S : fin n → set ℤ)
  (h3 : ∀ i, (S i).nonempty)
  (h4 : ∀ (T : finset (fin n)), T.card = k → ∃ i j, i ≠ j ∧ (S i ∩ S j).nonempty) : 
  ∃ X : finset ℤ, X.card = k - 1 ∧ ∀ i, ∃ x ∈ X, x ∈ S i :=
sorry

end exists_subset_with_intersections_l420_420088


namespace train_length_correct_l420_420752

noncomputable def train_length (speed_kmh : ℝ) (time_s : ℝ) : ℝ :=
  let speed_ms := (speed_kmh * 1000) / 3600
  speed_ms * time_s

theorem train_length_correct 
  (speed_kmh : ℝ := 60) 
  (time_s : ℝ := 9) :
  train_length speed_kmh time_s = 150.03 := by 
  sorry

end train_length_correct_l420_420752


namespace min_PA_PE_l420_420006

-- Define points and geometry concepts
structure Point : Type :=
(x : ℝ)
(y : ℝ)

-- Define the parabola y = -(1/8)*(x-4)^2
def is_on_parabola (p : Point) : Prop :=
  p.y = -(1/8) * (p.x - 4)^2

-- Define focus of the parabola
def is_focus (p : Point) : Prop :=
  p = ⟨4, 2⟩

-- Define vertex of the parabola
def vertex : Point :=
  ⟨4, 0⟩

-- Define distance function between two points
def dist (p1 p2 : Point) : ℝ :=
  real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define the directrix of the parabola
def is_on_directrix (p : Point) : Prop :=
  p.y = 2

-- Given conditions
axiom AF_eq_4 (A F : Point) (hF : is_focus F) (hA : is_on_parabola A) : dist A F = 4

-- Proving minimum value of |PA| + |PE| is 2√13
theorem min_PA_PE (P E A F : Point) (hA : is_on_parabola A) (hF : is_focus F) (hP : is_on_directrix P) (hAF : dist A F = 4) :
  ∃ (A : Point), (dist A ⟨0, 4⟩ = 2 * real.sqrt 13) := by
  sorry

end min_PA_PE_l420_420006


namespace interest_difference_l420_420993

noncomputable def annual_amount (P r t : ℝ) : ℝ :=
P * (1 + r)^t

noncomputable def monthly_amount (P r n t : ℝ) : ℝ :=
P * (1 + r / n)^(n * t)

theorem interest_difference
  (P : ℝ)
  (r : ℝ)
  (n : ℕ)
  (t : ℝ)
  (annual_compounded : annual_amount P r t = 8000 * (1 + 0.10)^3)
  (monthly_compounded : monthly_amount P r 12 3 = 8000 * (1 + 0.10 / 12) ^ (12 * 3)) :
  (monthly_amount P r 12 t - annual_amount P r t) = 142.80 := 
sorry

end interest_difference_l420_420993


namespace flux_vector_field_through_Sigma_l420_420817

open Real Set MeasureTheory

noncomputable def vector_field (x y z : ℝ) : ℝ × ℝ × ℝ :=
  (y^2 + z^2, x*y + y^2, x*z + z)

noncomputable def surface_Σ : Set (ℝ × ℝ × ℝ) :=
  {p | (∃ z ∈ Icc (0:ℝ) (1:ℝ), (p.1)^2 + (p.2)^2 = 1 ∧ p.3 = z) ∨
       (p.3 = 0 ∧ (p.1)^2 + (p.2)^2 ≤ 1) ∨
       (p.3 = 1 ∧ (p.1)^2 + (p.2)^2 ≤ 1) }

theorem flux_vector_field_through_Sigma :
  ∫ (p : ℝ × ℝ × ℝ) in surface_Σ, 
  (vector_field p.1 p.2 p.3).1 * p.1 + (vector_field p.1 p.2 p.3).2 * p.2 + 
  (vector_field p.1 p.2 p.3).3 * p.3 = 2 * π :=
sorry

end flux_vector_field_through_Sigma_l420_420817


namespace actual_revenue_percent_projected_revenue_l420_420119

theorem actual_revenue_percent_projected_revenue
  {R : ℝ}  -- last year's revenue
  (h1 : ∀ R, projected_revenue R = 1.2 * R)  -- Condition 1: projected revenue is a 20% increase over last year's revenue
  (h2 : ∀ R, actual_revenue R = 0.9 * R)  -- Condition 2: actual revenue is a 10% decrease from last year's revenue
  : ∀ R, (actual_revenue R / projected_revenue R) * 100 = 75 :=
by
  -- Proof steps would go here, but we'll use 'sorry' to skip the proof.
  sorry

end actual_revenue_percent_projected_revenue_l420_420119


namespace number_of_tiles_per_row_l420_420597

-- Define the conditions
def area (a : ℝ) : ℝ := a * a
def side_length (area : ℝ) : ℝ := real.sqrt area
def feet_to_inches (feet : ℝ) : ℝ := feet * 12
def tiles_per_row (room_length_inches : ℝ) (tile_size_inches : ℝ) : ℕ := 
  int.to_nat ⟨room_length_inches / tile_size_inches, by sorry⟩

-- Given constants in the problem
def area_of_room : ℝ := 256
def tile_size : ℝ := 8

-- Derived lengths
def length_of_side := side_length area_of_room
def length_of_side_in_inches := feet_to_inches length_of_side

-- The theorem to prove
theorem number_of_tiles_per_row : tiles_per_row length_of_side_in_inches tile_size = 24 :=
sorry

end number_of_tiles_per_row_l420_420597


namespace f_is_odd_f_is_decreasing_in_0_2_l420_420884

def f (x : ℝ) : ℝ := (x^2 + 4) / x

-- Prove that f(x) is an odd function
theorem f_is_odd : ∀ x, f (-x) = - f x := by
  sorry

-- Prove that f(x) is a decreasing function in the interval (0,2)
theorem f_is_decreasing_in_0_2 : ∀ ⦃x₁ x₂ : ℝ⦄, 0 < x₁ → x₁ < x₂ → x₂ < 2 → f x₁ > f x₂ := by
  sorry

end f_is_odd_f_is_decreasing_in_0_2_l420_420884


namespace find_y_l420_420045

theorem find_y (y : ℕ) (h : 2^10 = 32^y) : y = 2 :=
by {
  sorry
}

end find_y_l420_420045


namespace find_a_and_b_find_sqrt_4a_plus_b_l420_420873

section

variable {a b : ℤ}

-- Define the conditions
def condition1 (a b : ℤ) := ((3 * a - 14)^2 = (a - 2)^2) 
def condition2 (b : ℤ) := (b - 15) ^ (3 : ℝ) = -27

-- Prove the given values of a and b
theorem find_a_and_b : ∃ a b : ℤ, condition1 a b ∧ condition2 b :=
begin
  use [4, -12],
  split,
  -- prove condition1
  { unfold condition1, norm_num },
  -- prove condition2
  { unfold condition2, norm_num }
end

-- Prove the square root of 4a + b
theorem find_sqrt_4a_plus_b (a b : ℤ) (h₁ : condition1 a b) (h₂ : condition2 b) : 
  sqrt (4 * a + b) = 2 ∨ sqrt (4 * a + b) = -2 :=
begin
  -- use given conditions to prove the statement
  have ha : a = 4, from sorry,
  have hb : b = -12, from sorry,
  rw [ha, hb],
  norm_num,
  left,
  norm_num,
end

end

end find_a_and_b_find_sqrt_4a_plus_b_l420_420873


namespace count_rational_or_integer_numbers_from_list_l420_420441

theorem count_rational_or_integer_numbers_from_list : 
  let S := {n : ℕ | ∃ x : ℝ, x = (216 : ℝ)^(1 / (n : ℝ)) ∧ (x ∈ ℚ ∨ ∃ k : ℕ, x = k)}
  in S.card = 2 :=
by
  sorry

end count_rational_or_integer_numbers_from_list_l420_420441


namespace tokens_game_ends_after_56_rounds_l420_420360

noncomputable def TokensGame := 
  let initial_tokens := (17, 16, 15, 14)
  let rounds := 56
  ∀ r, if ∃ t, t ∈ [initial_tokens.1, initial_tokens.2, initial_tokens.3, initial_tokens.4], t - 4 * r ≤ 0 then
        r <= rounds

theorem tokens_game_ends_after_56_rounds : ∀ (tokens : ℕ × ℕ × ℕ × ℕ),
  tokens = (17, 16, 15, 14) →
  ∃ r : ℕ, 
  (∀ i ∈ [tokens.1, tokens.2, tokens.3, tokens.4], i - 4 * r ≤ 0) ∧ r = 56 := 
sorry

end tokens_game_ends_after_56_rounds_l420_420360


namespace factorial_expression_computation_l420_420796

theorem factorial_expression_computation :
  (11.factorial / (7.factorial * 4.factorial) * 2) = 660 := 
by
  -- Placeholder for the proof
  sorry

end factorial_expression_computation_l420_420796


namespace train_length_correct_l420_420751

noncomputable def train_length (speed_kmh : ℝ) (time_s : ℝ) : ℝ :=
  let speed_ms := (speed_kmh * 1000) / 3600
  speed_ms * time_s

theorem train_length_correct 
  (speed_kmh : ℝ := 60) 
  (time_s : ℝ := 9) :
  train_length speed_kmh time_s = 150.03 := by 
  sorry

end train_length_correct_l420_420751


namespace ordered_triple_l420_420730

-- Definitions from conditions
def x (t : ℝ) : ℝ := 3 * Real.cos t - 2 * Real.sin t
def y (t : ℝ) : ℝ := 5 * Real.sin t

-- Prove that these a, b, c are the solution
theorem ordered_triple (a b c : ℝ) :
  (∀ t : ℝ, a * (x t)^2 + b * (x t) * (y t) + c * (y t)^2 = 9) → 
  a = 1 / 9 ∧ b = 4 / 45 ∧ c = 4 / 225 := by
sorry

end ordered_triple_l420_420730


namespace largest_gold_coins_l420_420314

noncomputable def max_gold_coins (n : ℕ) : ℕ :=
  if h : ∃ k : ℕ, n = 13 * k + 3 ∧ n < 150 then
    n
  else 0

theorem largest_gold_coins : max_gold_coins 146 = 146 :=
by
  sorry

end largest_gold_coins_l420_420314


namespace sum_of_integer_solutions_l420_420194

theorem sum_of_integer_solutions :
  (∑ x in {x : ℤ | 1 < (x - 3)^2 ∧ (x - 3)^2 < 36}, x) = 24 :=
by
  -- This is a placeholder to indicate where the proof would go.
  sorry

end sum_of_integer_solutions_l420_420194


namespace cross_product_magnitude_l420_420007

open Real EuclideanSpace

theorem cross_product_magnitude (a b : ℝ × ℝ)
  (h1 : a = (-√3, -1))
  (h2 : b = (1, √3)) :
  (∥a.1 * b.2 - a.2 * b.1∥) = 2 := by
  sorry

end cross_product_magnitude_l420_420007


namespace quad_area_proof_l420_420072

variable (Point : Type) [affine_space Point]
variables (A B C D O M N : Point)
variables (AC BD : line Point)

-- Define midpoints and their roles
def is_midpoint (M : Point) (A C : Point) : Prop := (vector_of_points M A + vector_of_points M C) = 0
def is_intersect (O : Point) (BA CD : line Point) : Prop := O ∈ BA ∧ O ∈ CD

-- Define area using vector cross product
def area (P Q R : Point) : ℝ :=
  let u := vector_of_points P Q in
  let v := vector_of_points P R in
  0.5 * (u και v).norm

-- Assertions derived from conditions
axiom midpoint_M : is_midpoint M A C
axiom midpoint_N : is_midpoint N B D
axiom intersect_O : is_intersect O BA CD

-- The problem statement
theorem quad_area_proof (A B C D O M N : Point)
                        (midpoint_M : is_midpoint M A C)
                        (midpoint_N : is_midpoint N B D)
                        (intersect_O : is_intersect O BA CD) :
  4 * area O M N = area A B C D :=
sorry

end quad_area_proof_l420_420072


namespace total_shaded_area_l420_420077

-- Definitions for the problem conditions
def larger_circle_area : ℝ := 100 * Real.pi
def shaded_ratio_larger : ℝ := 2 / 3
def distance_center_smaller_circle_to_O : ℝ := 2

-- The correct answer for the total shaded area
def expected_total_shaded_area : ℝ := (296 * Real.pi) / 3

-- Main theorem statement to be proved
theorem total_shaded_area
  (larger_circle_area : ℝ :=
   100 * Real.pi)
  (shaded_ratio_larger : ℝ :=
   2 / 3)
  (distance_center_smaller_circle_to_O : ℝ :=
   2)
  (expected_total_shaded_area : ℝ :=
   (296 * Real.pi) / 3) :
  ∃ (radius_larger radius_smaller : ℝ),
    (Real.pi * radius_larger^2 = larger_circle_area) ∧
    (Real.pi * (radius_larger - distance_center_smaller_circle_to_O)^2 = Real.pi * 8^2) ∧
    (shaded_ratio_larger * larger_circle_area + (1 / 2) * Real.pi * (radius_larger - distance_center_smaller_circle_to_O)^2 = expected_total_shaded_area) :=
sorry

end total_shaded_area_l420_420077


namespace negation_equiv_l420_420170

-- Define the proposition to be negated.
def exists_le_zero (x : ℝ) : Prop := ∃ x0 : ℝ, 2^x0 ≤ 0

-- Define the negation of the proposition.
def forall_gt_zero : Prop := ∀ x : ℝ, 2^x > 0

theorem negation_equiv : ¬ exists_le_zero x ↔ forall_gt_zero := by
  sorry

end negation_equiv_l420_420170


namespace two_digit_solutions_eq_3_l420_420827

def P (n : ℕ) : ℕ := n % 3 + n % 4 + n % 5 + n % 7 + n % 8 + n % 9

def delta (n k : ℕ) : ℤ :=
if n % k = k - 1 then -(k - 1) else 1

def isSolution (n : ℕ) : Prop :=
P n = P (n + 1)

def numSolutions : ℕ :=
(Finset.filter (λ n, isSolution n) (Finset.range (100) \ Finset.range (10))).card

theorem two_digit_solutions_eq_3 :
  numSolutions = 3 :=
begin
  sorry
end

end two_digit_solutions_eq_3_l420_420827


namespace range_of_a_l420_420886

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * Real.exp (-x)

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (D^2) (λ x, f a x) x ≥ 2 * Real.sqrt 6) → a ≥ 6 :=
by
  sorry

end range_of_a_l420_420886


namespace greatest_area_difference_l420_420669

theorem greatest_area_difference (l₁ w₁ l₂ w₂ : ℕ) (h₁ : 2 * l₁ + 2 * w₁ = 160) (h₂ : 2 * l₂ + 2 * w₂ = 160) : 
  abs (l₁ * w₁ - l₂ * w₂) = 1521 :=
sorry

end greatest_area_difference_l420_420669


namespace max_area_difference_l420_420673

theorem max_area_difference (l w l' w' : ℕ) 
  (h1 : 2 * l + 2 * w = 160) 
  (h2 : 2 * l' + 2 * w' = 160) : 
  abs ((l * w) - (l' * w')) ≤ 1600 := 
by 
  sorry

end max_area_difference_l420_420673


namespace pyramid_triangular_face_area_l420_420217

theorem pyramid_triangular_face_area 
  (base : ℝ) (lateral : ℝ)
  (h_base : base = 8) (h_lateral : lateral = 7) :
  let height := sqrt (lateral^2 - (base / 2)^2) in
  4 * (1 / 2 * base * height) = 16 * sqrt 33 := 
by
  sorry

end pyramid_triangular_face_area_l420_420217


namespace length_of_BC_l420_420063

-- Definitions corresponding to the conditions in the problem.
def Circle (O : Point) (r : ℝ) : Set Point := {P | dist O P = r}

-- Setup
variables {O A B C D : Point}           -- Points on the circle
variable {r : ℝ}                        -- Radius of circle
variable [metric_space Point]
variable [euclidean_space ℝ Point]      -- Assuming the points lie in a Euclidean space

-- Conditions
variables (h_circle : ∀ (P ∈ Circle O r), dist O P = r)  -- O is the center of the circle
variables (h_diameter : dist A D = 2 * r)       -- AD is a diameter
variables (h_chord : A ∈ Circle O r)            -- A, B, and C on the circle
variables (h_chord_2 : B ∈ Circle O r)
variables (h_chord_3 : C ∈ Circle O r)
variables (h_BO : dist O B = 7)                 -- BO = 7
variables (h_angle_ABO : angle A B O = 90)      -- ∠ABO = 90°
variables (h_arc_CD : Arc CD = 90)             -- arc CD = 90°

-- Conclusion
theorem length_of_BC : dist B C = 7 :=
by sorry

end length_of_BC_l420_420063


namespace nth_equation_l420_420570

theorem nth_equation (n : ℕ) (h : n > 0) : 9 * (n - 1) + n = (n - 1) * 10 + 1 :=
sorry

end nth_equation_l420_420570


namespace distance_sum_eq_l420_420834

-- Geometric definitions, including points and distances
variable (A B C D P Q R : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variable (radius_A radius_B radius_C radius_D : ℝ)
variable (AB CD PQ AR BR CR DR : ℝ)

-- Conditions from the problem
axiom radius_relations_A : radius_A = (3 / 4) * radius_B
axiom radius_relations_C : radius_C = 2 * radius_D
axiom distance_AB : AB = 42
axiom distance_CD : CD = 50
axiom length_PQ : PQ = 52

-- Point R is the midpoint of segment PQ
axiom midpoint_R : R = (P + Q) / 2  -- Assuming types allow such structure

-- Proof goal
theorem distance_sum_eq : AR + BR + CR + DR = 184 := by
  sorry

end distance_sum_eq_l420_420834


namespace compare_f_values_l420_420879

noncomputable def f (x : ℝ) : ℝ := x^2 - Real.cos x

theorem compare_f_values :
  f 0.6 > f (-0.5) ∧ f (-0.5) > f 0 := by
  sorry

end compare_f_values_l420_420879


namespace domain_ln_x_plus_one_l420_420815

theorem domain_ln_x_plus_one :
  ∀ (x : ℝ), ∃ (y : ℝ), y = Real.log (x + 1) ↔ x > -1 :=
by sorry

end domain_ln_x_plus_one_l420_420815


namespace arithmetic_result_l420_420822

/-- Define the constants involved in the arithmetic operation. -/
def a : ℕ := 999999999999
def b : ℕ := 888888888888
def c : ℕ := 111111111111

/-- The theorem stating that the given arithmetic operation results in the expected answer. -/
theorem arithmetic_result :
  a - b + c = 222222222222 :=
by
  sorry

end arithmetic_result_l420_420822


namespace number_of_factors_M_l420_420494

noncomputable def M : ℕ := 2^5 * 3^4 * 5^3 * 7^3 * 11^2

theorem number_of_factors_M : (finset.range (5 + 1)).card * 
                              (finset.range (4 + 1)).card * 
                              (finset.range (3 + 1)).card * 
                              (finset.range (3 + 1)).card * 
                              (finset.range (2 + 1)).card = 
                              1440 := by
  -- proof would go here
  sorry

end number_of_factors_M_l420_420494


namespace part1_part2_part3_l420_420089

noncomputable def A : Set ℝ := { x | x ≥ 1 ∨ x ≤ -3 }
noncomputable def B : Set ℝ := { x | -4 < x ∧ x < 0 }
noncomputable def C : Set ℝ := { x | x ≤ -4 ∨ x ≥ 0 }

theorem part1 : A ∩ B = { x | -4 < x ∧ x ≤ -3 } := 
by { sorry }

theorem part2 : A ∪ B = { x | x < 0 ∨ x ≥ 1 } := 
by { sorry }

theorem part3 : A ∪ C = { x | x ≤ -3 ∨ x ≥ 0 } := 
by { sorry }

end part1_part2_part3_l420_420089


namespace Martha_points_l420_420995

def beef_cost := 3 * 11
def fv_cost := 8 * 4
def spice_cost := 3 * 6
def other_cost := 37

def total_spent := beef_cost + fv_cost + spice_cost + other_cost
def points_per_10 := 50
def bonus := 250

def increments := total_spent / 10
def points := increments * points_per_10
def total_points := points + bonus

theorem Martha_points : total_points = 850 :=
by
  sorry

end Martha_points_l420_420995


namespace complex_square_l420_420015

def z : ℂ := 1 - 2 * Complex.i

theorem complex_square : z ^ 2 = -3 - 4 * Complex.i :=
by
  sorry

end complex_square_l420_420015


namespace find_length_of_BC_l420_420066

def circle_geometry_problem (O A B C D : Point) (BO : ℝ)
  (angle_ABO : ℝ) (arc_CD : ℝ) : Prop :=
  diameter O A D ∧ chord O A B C ∧ BO = 7 ∧ angle_ABO = 90 ∧ arc_CD = 90

theorem find_length_of_BC (O A B C D : Point) (BO : ℝ)
  (angle_ABO : ℝ) (arc_CD : ℝ) (h : circle_geometry_problem O A B C D BO angle_ABO arc_CD) :
  length B C = 7 := by
  sorry

end find_length_of_BC_l420_420066


namespace f_is_odd_function_f_is_increasing_f_max_min_in_interval_l420_420447

variable {f : ℝ → ℝ}

-- The conditions:
axiom additivity : ∀ x y : ℝ, f (x + y) = f x + f y
axiom positive_for_positive : ∀ x : ℝ, x > 0 → f x > 0
axiom f_one_is_two : f 1 = 2

-- The proof tasks:
theorem f_is_odd_function : ∀ x : ℝ, f (-x) = -f x := 
sorry

theorem f_is_increasing : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2 := 
sorry

theorem f_max_min_in_interval : 
  (∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 → f x ≤ 6) ∧ (∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 → f x ≥ -6) :=
sorry

end f_is_odd_function_f_is_increasing_f_max_min_in_interval_l420_420447


namespace pyramid_area_l420_420301

noncomputable def base_edge : ℝ := 8
noncomputable def lateral_edge : ℝ := 7
noncomputable def height := real.sqrt (lateral_edge^2 - (base_edge / 2)^2)
noncomputable def one_triangle_area := (1 / 2) * base_edge * height
noncomputable def total_area := 4 * one_triangle_area

theorem pyramid_area : total_area = 16 * real.sqrt 33 := by
  sorry

end pyramid_area_l420_420301


namespace squares_below_line_l420_420620

theorem squares_below_line (a b c : ℤ) (ha : a = 6) (hb : b = 143) (hc : c = 858) :
  let intercept_x := c / a
  let intercept_y := c / b
  let total_squares := intercept_x * intercept_y
  let equation_of_line := λ x, c / b / (c / a) * x
  let horizontal_intersection := intercept_y - 1
  let vertical_intersection := intercept_x - 1
  let total_diagonal_squares := horizontal_intersection + vertical_intersection + 1
  let non_diagonal_squares := total_squares - total_diagonal_squares
  let below_diagonal_squares := non_diagonal_squares / 2
  below_diagonal_squares = 355 :=
sorry

end squares_below_line_l420_420620


namespace workout_goal_l420_420582

def monday_situps : ℕ := 12
def tuesday_situps : ℕ := 19
def wednesday_situps_needed : ℕ := 59

theorem workout_goal : monday_situps + tuesday_situps + wednesday_situps_needed = 90 := by
  sorry

end workout_goal_l420_420582


namespace symmetry_about_origin_l420_420166

def f (x : ℝ) : ℝ := x^3 - x

theorem symmetry_about_origin : 
  ∀ x : ℝ, f (-x) = -f x := by
  sorry

end symmetry_about_origin_l420_420166


namespace acute_angle_between_AC_and_BD_l420_420946

noncomputable def angle_between_lines (A B C D M : Point) (AC BD : Line) : ℝ :=
  if h : is_midpoint M A D ∧ angle B M C = 90 ∧ distance A B = distance B C ∧ distance B C = distance C D then
    30
  else
    0  -- Using 0 as a placeholder for failed conditions

theorem acute_angle_between_AC_and_BD {A B C D M : Point} {AC BD : Line} :
  is_quadrilateral A B C D →
  distance A B = distance B C →
  distance B C = distance C D →
  is_midpoint M A D →
  angle B M C = 90 →
  angle_between_lines A B C D M AC BD = 30 :=
begin
  -- Proof is omitted
  sorry
end

end acute_angle_between_AC_and_BD_l420_420946


namespace pyramid_four_triangular_faces_area_l420_420255

theorem pyramid_four_triangular_faces_area 
  (base_edge : ℝ) (lateral_edge : ℝ) 
  (h_base : base_edge = 8)
  (h_lateral : lateral_edge = 7) :
  let h := Real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)
  let triangle_area := (1 / 2) * base_edge * h
  let total_area := 4 * triangle_area
  total_area = 16 * Real.sqrt 33 :=
by
  -- Definitions to introduce local values
  let half_base := base_edge / 2
  let h := Real.sqrt (lateral_edge ^ 2 - half_base ^ 2)
  let triangle_area := (1 / 2) * base_edge * h
  let total_area := 4 * triangle_area
  -- Assertion to compare calculated total area with given correct answer
  have h_eq : h = Real.sqrt 33 := by sorry
  have triangle_area_eq : triangle_area = 4 * Real.sqrt 33 := by sorry
  have total_area_eq : total_area = 16 * Real.sqrt 33 := by sorry
  exact total_area_eq

end pyramid_four_triangular_faces_area_l420_420255


namespace b_2009_l420_420972

noncomputable def b : ℕ → ℝ 
| 1       := 2 + Real.sqrt 8
| n@(2+_) := 14 + Real.sqrt 8
| n       := b (n - 1) * b (n + 1)

theorem b_2009 : b 2009 = -7 + 3.5 * Real.sqrt 8 := by
  -- Definitions based on conditions
  sorry

end b_2009_l420_420972


namespace marble_ratio_l420_420899

theorem marble_ratio :
  ∃ (marbles_given marbles_lost : ℕ), 
  let initial_marbles := 26 in
  let found_marbles := 6 in
  let lost_marbles := 10 in
  let final_marbles := 42 in
  let marbles_after_loss := initial_marbles + found_marbles - lost_marbles in
  let marbles_given := final_marbles - marbles_after_loss in
  marbles_lost = lost_marbles ∧ marbles_given * 1 = 2 * marbles_lost :=
begin
  use [20, 10],
  sorry
end

end marble_ratio_l420_420899


namespace candy_store_spending_l420_420039

noncomputable def weekly_allowance : ℝ := 4.80
noncomputable def arcade_fraction : ℝ := 3 / 5
noncomputable def toy_store_fraction : ℝ := 1 / 3

theorem candy_store_spending :
  let arcade_spending := arcade_fraction * weekly_allowance in
  let remaining_after_arcade := weekly_allowance - arcade_spending in
  let toy_store_spending := toy_store_fraction * remaining_after_arcade in
  let remaining_after_toy_store := remaining_after_arcade - toy_store_spending in
  remaining_after_toy_store = 1.28 :=
by
  sorry

end candy_store_spending_l420_420039


namespace siamese_cats_initially_l420_420737

theorem siamese_cats_initially (house_cats: ℕ) (cats_sold: ℕ) (cats_left: ℕ) (initial_siamese: ℕ) :
  house_cats = 5 → 
  cats_sold = 10 → 
  cats_left = 8 → 
  (initial_siamese + house_cats - cats_sold = cats_left) → 
  initial_siamese = 13 :=
by
  intros h1 h2 h3 h4
  sorry

end siamese_cats_initially_l420_420737


namespace mango_selling_price_l420_420359

theorem mango_selling_price
  (CP SP_loss SP_profit : ℝ)
  (h1 : SP_loss = 0.8 * CP)
  (h2 : SP_profit = 1.05 * CP)
  (h3 : SP_profit = 6.5625) :
  SP_loss = 5.00 :=
by
  sorry

end mango_selling_price_l420_420359


namespace problem_1_problem_2_problem_3_problem_4_l420_420397

-- Problem 1
theorem problem_1 : (-18) + 9 - (-5) - 7 = -11 := by
  sorry

-- Problem 2
theorem problem_2 : 
  let a := 2 + 1/4
  let b := 3/2 : ℚ
  let c := -2/3 : ℚ
  let d := abs (-1/6 : ℚ)
  (a / b * c + d : ℚ) = -5/6 := by
  sorry

-- Problem 3
theorem problem_3 :
  let x := 1/2
  let y := 2/3
  let z := 2/5 : ℚ
  let w := -1/30 : ℚ
  5 - ((x - y + z) / w) = 12 := by
  sorry

-- Problem 4
theorem problem_4 : 
  let a := 4
  let b := -32 : ℚ
  let c := (-2/3 : ℚ)^3
  let d := (-3 : ℚ)^2
  let e := -11/3 : ℚ
  (a^3 / b - (c * d + e)) = 4 + 1/3 := by
  sorry

end problem_1_problem_2_problem_3_problem_4_l420_420397


namespace DVDs_already_in_book_l420_420350

variables (total_capacity empty_spaces already_in_book : ℕ)

-- Conditions given in the problem
def total_capacity : ℕ := 126
def empty_spaces : ℕ := 45

-- The problem to prove
theorem DVDs_already_in_book : already_in_book = total_capacity - empty_spaces :=
by
  let already_in_book := total_capacity - empty_spaces
  trivial

end DVDs_already_in_book_l420_420350


namespace expand_product_l420_420808

theorem expand_product (x: ℝ) : (x + 4) * (x^2 - 9) = x^3 + 4x^2 - 9x - 36 :=
by sorry

end expand_product_l420_420808


namespace pyramid_total_area_l420_420223

/-- The total area of the four triangular faces of a right, square-based pyramid
whose base edges measure 8 units and lateral edges measure 7 units is 16√33. -/
theorem pyramid_total_area :
  let base_edge := 8
  let lateral_edge := 7
  4 * (1 / 2 * base_edge * Real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 16 * Real.sqrt 33 := 
by
  let base_edge := 8
  let lateral_edge := 7
  have h1 : 4 * (1 / 2 * base_edge * Real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 
              4 * (1 / 2 * 8 * Real.sqrt (49 - 16)) := 
    by sorry
  have h2 : 4 * (1 / 2 * 8 * Real.sqrt (49 - 16)) = 4 * (4 * Real.sqrt 33) := 
    by sorry
  have h3 : 4 * (4 * Real.sqrt 33) = 16 * Real.sqrt 33 := 
    by sorry
  exact eq.trans (eq.trans h1 h2) h3

end pyramid_total_area_l420_420223


namespace min_value_exponential_sub_l420_420841

theorem min_value_exponential_sub (x y : ℝ) (hx : x > 0) (hy : y > 0)
    (h : x + 2 * y = x * y) : ∃ y₀ > 0, ∀ y > 1, e^y - 8 / x ≥ e :=
by
  sorry

end min_value_exponential_sub_l420_420841


namespace garage_travel_time_correct_l420_420955

theorem garage_travel_time_correct :
  let floors := 12
  let gate_interval := 3
  let gate_time := 2 * 60 -- 2 minutes in seconds
  let distance_per_floor := 800
  let speed := 10 in
  let num_gates := floors / gate_interval
  let total_gate_time := num_gates * gate_time
  let time_per_floor := distance_per_floor / speed
  let total_drive_time := floors * time_per_floor in
  total_gate_time + total_drive_time = 1440 := by
  sorry

end garage_travel_time_correct_l420_420955


namespace pyramid_total_area_l420_420292

noncomputable def pyramid_base_edge : ℝ := 8
noncomputable def pyramid_lateral_edge : ℝ := 7

theorem pyramid_total_area 
  (base_edge : ℝ)
  (lateral_edge : ℝ)
  (h_base_edge : base_edge = pyramid_base_edge)
  (h_lateral_edge : lateral_edge = pyramid_lateral_edge) 
: 4 * (1 / 2 * base_edge * real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 16 * real.sqrt 33 := 
sorry

end pyramid_total_area_l420_420292


namespace fermats_little_theorem_analogue_l420_420576

theorem fermats_little_theorem_analogue 
  (a : ℤ) (h1 : Int.gcd a 561 = 1) : a ^ 560 ≡ 1 [ZMOD 561] := 
sorry

end fermats_little_theorem_analogue_l420_420576


namespace range_of_angle_formed_by_skew_lines_l420_420156

/-- Let θ be the angle formed by skew lines a and b. Prove that the range of θ is (0, π/2]. -/
theorem range_of_angle_formed_by_skew_lines (a b : Line) (θ : Real) 
  (h_θ : is_angle_formed_by_skew_lines a b θ) : 
  (0 : Real) < θ ∧ θ ≤ (Real.pi / 2) := 
sorry

end range_of_angle_formed_by_skew_lines_l420_420156


namespace cube_edge_length_l420_420122

variables (a h x : ℝ)

theorem cube_edge_length 
  (pyramid_base : a > 0) 
  (pyramid_height : h > 0)
  (cube_constraints : 
    (exists one_face_on_base : { 
      exists vertices_on_one_lateral : 
        (exists two_vertices_on_same_face : 
          (exists two_vertices_on_different_faces))) 
  ) :
  x = (3 * a * h) / (3 * a + h * (3 + 2 * Real.sqrt 3)) :=
sorry

end cube_edge_length_l420_420122


namespace sum_of_arithmetic_sequence_l420_420692

theorem sum_of_arithmetic_sequence (a d : ℤ) (n : ℕ) (h1 : a = -3) (h2 : d = 7) (h3 : n = 10) :
  let aₙ := a + (n - 1) * d in
  (a + aₙ) * n / 2 = 285 :=
by
  sorry

end sum_of_arithmetic_sequence_l420_420692


namespace select_numbers_with_second_largest_seven_l420_420385

open Finset

theorem select_numbers_with_second_largest_seven : 
  (univ.filter (λ s : Finset ℕ, s.card = 4 ∧ 7 ∈ s ∧ secondLargest s = some 7)).card = 45 :=
sorry

end select_numbers_with_second_largest_seven_l420_420385


namespace breadth_remains_the_same_l420_420626

variable (L B : ℝ)

theorem breadth_remains_the_same 
  (A : ℝ) (hA : A = L * B) 
  (L_half : ℝ) (hL_half : L_half = L / 2) 
  (B' : ℝ)
  (A' : ℝ) (hA' : A' = L_half * B') 
  (hA_change : A' = 0.5 * A) : 
  B' = B :=
  sorry

end breadth_remains_the_same_l420_420626


namespace max_value_of_y_l420_420344

theorem max_value_of_y (x : ℝ) (h : x < 5/4) : 
  ∃ y : ℝ, y = 4 * x - 2 + 1 / (4 * x - 5) ∧ y ≤ 1 :=
sorry

end max_value_of_y_l420_420344


namespace pyramid_area_l420_420243

theorem pyramid_area :
  ∀ (a b : ℝ), a = 8 → b = 7 → 4 * (1/2 * a * sqrt (b^2 - (a/2)^2)) = 16 * sqrt 33 :=
by
  intros a b ha hb
  rw [ha, hb]
  have h1 : a / 2 = 4 := by norm_num [ha]
  have h2 : b^2 - (a / 2)^2 = 33 :=
    by
      calc
        b^2 - (a / 2)^2 = 49 - 16 := by norm_num [hb]
        ... = 33 := by norm_num
  rw [h1, h2, sqrt 33, mul_one, mul_one, half_mul, mul_comm (1/2) a, ←mul_assoc, mul_comm 4 4]
  norm_num
  sorry

end pyramid_area_l420_420243


namespace problem_solution_l420_420910

theorem problem_solution (x : ℝ) (h : x + 1 / x = 8) : x^2 + 1 / x^2 = 62 := 
by
  sorry

end problem_solution_l420_420910


namespace prove_midpoint_and_perpendicular_l420_420961

variable {Point : Type}
variables (A B C D E F P M : Point)
variables [Isosceles ABC BC] [Isosceles DBC BC]
variable (h1 : ∠ABD = π / 2)
variable (h2 : midpoint M B C)
variable (h3 : E ∈ line(A, B))
variable (h4 : P ∈ line(M, C))
variable (h5 : C ∈ line(A, F))
variable (h6 : ∠BDE = ∠ADP = ∠CDF)

theorem prove_midpoint_and_perpendicular :
  (midpoint P E F) ∧ (perpendicular D P E F) := sorry

end prove_midpoint_and_perpendicular_l420_420961


namespace distance_focus_directrix_l420_420613

theorem distance_focus_directrix (p : ℝ) :
  (∀ (x y : ℝ), y^2 = 2 * p * x ∧ x = 6 ∧ dist (x, y) (p/2, 0) = 10) →
  abs (p) = 8 :=
by
  sorry

end distance_focus_directrix_l420_420613


namespace consecutive_integer_sum_product_l420_420193

theorem consecutive_integer_sum_product :
  let S := (250 / 2) * (-125 + 124)
  let P := 2 * 4 * 6 * 8 * 10 * 12 * 14 * 16 * 18 * 20
  S * P = -464485600000 := 
by
  let S := 125 * (-1)
  let P := 2 * 4 * 6 * 8 * 10 * 12 * 14 * 16 * 18 * 20
  have hS : S = -125 := by
    sorry
  have hP : P = 3715884800 := by
    sorry
  calc
    S * P
        = (-125) * 3715884800 : by rw [hS, hP]
    ... = -464485600000 : by norm_num

end consecutive_integer_sum_product_l420_420193


namespace evaluate_f_of_composed_g_l420_420973

def f (x : ℤ) : ℤ := 3 * x - 4
def g (x : ℤ) : ℤ := x + 2

theorem evaluate_f_of_composed_g :
  f (2 + g 3) = 17 :=
by
  sorry

end evaluate_f_of_composed_g_l420_420973


namespace password_count_l420_420165

def unique_numbers (numbers : List ℕ) : List ℕ :=
  numbers.erase_dup

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def combinations (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

def arrangements_of_unique_numbers (numbers : List ℕ) : ℕ :=
  factorial (unique_numbers numbers).length

def non_adjacent_placements (spaces : ℕ) (choices : ℕ) : ℕ :=
  combinations spaces choices

theorem password_count (numbers : List ℕ) (spaces choices : ℕ) : (arrangements_of_unique_numbers numbers) * (non_adjacent_placements spaces choices) = 240 := by
  sorry

#eval password_count [6, 1, 8, 3, 3, 9] 5 2

end password_count_l420_420165


namespace final_number_not_zero_l420_420121

theorem final_number_not_zero :
  ∀ (numbers : List ℕ), (numbers.length = 1974) →
  (∀ a b : ℕ, a ∈ numbers → b ∈ numbers →
  (let new_num := [a + b, a - b] in
  ∃ c, c ∈ new_num ∧ List.length (numbers.erase a).erase b + 1 = numbers.length)) →
  ∀ a ∈ numbers, a ≠ 0 := sorry

end final_number_not_zero_l420_420121


namespace curve_standard_eq_and_chord_length_l420_420027

theorem curve_standard_eq_and_chord_length :
  (∀ (ρ θ : ℝ), ρ^2 * (Real.cos (2*θ)) = 1 → x^2 - y^2 = 1) ∧
  (∀ (t : ℝ), (x = 2 + (1/2) * t) ∧ (y = (√3 / 2) * t) →
     let t1 := 4 - √40 / 2;
         t2 := 4 + √40 / 2;
     abs(t1 - t2) = 2 * √10) :=
by {
  sorry
}

end curve_standard_eq_and_chord_length_l420_420027


namespace distance_point_line_cartesian_to_polar_circle_l420_420892

-- Define the given conditions
def polar_line (rho theta : ℝ) : Prop := 2 * rho * sin (theta - π / 4) = sqrt 2
def polar_point_A : ℝ × ℝ := (2 * sqrt 2, 7 * π / 4)
def cartesian_circle_eq (x y r : ℝ) : Prop := x^2 + y^2 = r^2 ∧ r > 0

-- Define the distance calculation problem statement
theorem distance_point_line :
  polar_line 2 (7 * π / 4) → 
  (let (x, y) := (2, -2) in abs (x + y + 1) / sqrt 2 = 5 * sqrt 2 / 2) := 
by sorry

-- Define the polar coordinate conversion problem statement
theorem cartesian_to_polar_circle (r : ℝ) :
  cartesian_circle_eq x y r → (ρ = r ∧ (0 ≤ θ ∧ θ < 2 * π))
:= 
by sorry

end distance_point_line_cartesian_to_polar_circle_l420_420892


namespace equilateral_triangle_circumcircle_point_distance_sum_l420_420414

-- Define the equilateral triangle, points, and distances
variables {A B C P : Type}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space P]
variables {a : ℝ} -- side length of the equilateral triangle
variables (h_equilateral : equilateral A B C a)
variables (h_circumcircle : circumcircle A B C)
variables (h_point_on_circle : point_on_circumcircle P A B C)

-- Objective: Prove PB + PC = PA
theorem equilateral_triangle_circumcircle_point_distance_sum
  (h_P_distances : PB + PC = PA) :
  PB + PC = PA :=
sorry

end equilateral_triangle_circumcircle_point_distance_sum_l420_420414


namespace pyramid_area_l420_420277

theorem pyramid_area (base_edge lateral_edge : ℝ) (H_base_edge : base_edge = 8) (H_lateral_edge : lateral_edge = 7) :
  4 * (1 / 2 * base_edge * real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 16 * real.sqrt 33 :=
by
  sorry

end pyramid_area_l420_420277


namespace factorial_expression_3121_min_a1_plus_b1_l420_420423

-- Definitions and the problem states
/-- Expresses that 3121 is equal to the fraction of factorials. -/
def expr_fact (a b : ℕ) : Prop :=
  3121 = a! / b!

theorem factorial_expression_3121_min_a1_plus_b1 (a1 b1 : ℕ) (a : ℕ → ℕ) (b : ℕ → ℕ)
  (h1 : expr_fact a1 b1)
  (h2 : ∀ i, a i ≥ a (i + 1))
  (h3 : ∀ i, b i ≥ b (i + 1))
  (h4 : a1 > 0 ∧ b1 > 0)
  (h5 : ∀ x y, x + y < a1 + b1 → 3121 ≠ 3121) -- This condition assures minimality, to be revised appropriately
  : |a1 - b1| = 2 :=
by
  sorry

end factorial_expression_3121_min_a1_plus_b1_l420_420423


namespace pyramid_triangular_face_area_l420_420218

theorem pyramid_triangular_face_area 
  (base : ℝ) (lateral : ℝ)
  (h_base : base = 8) (h_lateral : lateral = 7) :
  let height := sqrt (lateral^2 - (base / 2)^2) in
  4 * (1 / 2 * base * height) = 16 * sqrt 33 := 
by
  sorry

end pyramid_triangular_face_area_l420_420218


namespace find_fx_plus_gx_l420_420462

variables {R : Type*} [CommRing R]
variable (f g : R → R)

def is_odd (f : R → R) : Prop := ∀ x, f (-x) = -f(x)
def is_even (g : R → R) : Prop := ∀ x, g (-x) = g(x)

theorem find_fx_plus_gx (h_odd_f : is_odd f) (h_even_g : is_even g) (h_condition : ∀ x, f(x) - g(x) = 2*x - 3) :
  ∀ x, f(x) + g(x) = 2*x + 3 :=
by
  sorry

end find_fx_plus_gx_l420_420462


namespace tangent_line_parallel_to_given_line_l420_420478

noncomputable def f (x : ℝ) : ℝ := 1 / Real.exp x

theorem tangent_line_parallel_to_given_line :
  (∃ (m : ℝ) (b : ℝ), (∀ x, f' x = -1) ∧ ∀ x y, y = f x → x + y - 1 = 0) :=
begin
  sorry
end

end tangent_line_parallel_to_given_line_l420_420478


namespace shifted_polynomial_roots_are_shifted_l420_420971

noncomputable def original_polynomial : Polynomial ℝ := Polynomial.X ^ 3 - 5 * Polynomial.X + 7
noncomputable def shifted_polynomial : Polynomial ℝ := Polynomial.X ^ 3 + 6 * Polynomial.X ^ 2 + 7 * Polynomial.X + 5

theorem shifted_polynomial_roots_are_shifted :
  (∀ (a b c : ℝ), (original_polynomial.eval a = 0) ∧ (original_polynomial.eval b = 0) ∧ (original_polynomial.eval c = 0) 
    → (shifted_polynomial.eval (a - 2) = 0) ∧ (shifted_polynomial.eval (b - 2) = 0) ∧ (shifted_polynomial.eval (c - 2) = 0)) :=
by
  sorry

end shifted_polynomial_roots_are_shifted_l420_420971


namespace class_5_matches_l420_420931

theorem class_5_matches (matches_c1 matches_c2 matches_c3 matches_c4 matches_c5 : ℕ)
  (C1 : matches_c1 = 2)
  (C2 : matches_c2 = 4)
  (C3 : matches_c3 = 4)
  (C4 : matches_c4 = 3) :
  matches_c5 = 3 :=
sorry

end class_5_matches_l420_420931


namespace hyperbola_min_eccentricity_l420_420009

def minimum_eccentricity (c : ℝ) (a : ℝ) : ℝ := c / a

theorem hyperbola_min_eccentricity :
  let c := 3
  let a := Real.sqrt 5
  minimum_eccentricity c a = (3 * Real.sqrt 5) / 5 := 
by
  sorry

end hyperbola_min_eccentricity_l420_420009


namespace cost_per_box_types_l420_420726

-- Definitions based on conditions
def cost_type_B := 1500
def cost_type_A := cost_type_B + 500

-- Given conditions
def condition1 : cost_type_A = cost_type_B + 500 := by sorry
def condition2 : 6000 / (cost_type_B + 500) = 4500 / cost_type_B := by sorry

-- Theorem to be proved
theorem cost_per_box_types :
  cost_type_A = 2000 ∧ cost_type_B = 1500 ∧
  (∃ (m : ℕ), 20 ≤ m ∧ m ≤ 25 ∧ 2000 * (50 - m) + 1500 * m ≤ 90000) ∧
  (∃ (a b : ℕ), 2500 * a + 3500 * b = 87500 ∧ a + b ≤ 33) :=
sorry

end cost_per_box_types_l420_420726


namespace irrational_count_l420_420508

noncomputable def is_irrational (x : ℝ) : Prop := ¬ ∃ (n m : ℤ), m ≠ 0 ∧ x = n / m

theorem irrational_count : 
  (list.count is_irrational [2 * Real.pi, 0.4583, -2.7, 3.14, 4, -Real.of_digit_list 10 (-23) [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, ...]]) = 1 := 
sorry

end irrational_count_l420_420508


namespace abs_c_eq_142_l420_420589

-- Define the problem conditions
variables {a b c : ℤ}

-- Define the polynomial with given roots
def g (x : ℂ) : ℂ := a * x^4 + b * x^3 + c * x^2 + b * x + a

-- Define the root
def root := (3 + complex.I)

-- Define the main theorem to prove |c| = 142 given the conditions
theorem abs_c_eq_142 (h : g root = 0) (coprime_ab_c : (a.gcd b).gcd c = 1) : |c| = 142 :=
sorry

end abs_c_eq_142_l420_420589


namespace kaleb_spent_on_supplies_l420_420086

variable (springEarnings summerEarnings remainingAmount totalEarnings expenses : ℕ)

def earnings (springEarnings summerEarnings : ℕ) : ℕ := springEarnings + summerEarnings

theorem kaleb_spent_on_supplies :
  springEarnings = 4 →
  summerEarnings = 50 →
  remainingAmount = 50 →
  expenses = earnings springEarnings summerEarnings - remainingAmount →
  expenses = 4 :=
by 
  intros h1 h2 h3 h4
  rw [h1, h2, h3, earnings, h4]
  sorry

end kaleb_spent_on_supplies_l420_420086


namespace rent_expression_l420_420727

theorem rent_expression (x : ℕ) (h : x ≥ 2) : 
  let daily_rent_1 := 0.6
  let daily_rent_2 := 0.3
  let total_rent := daily_rent_1 * 2 + daily_rent_2 * (x - 2)
  total_rent = 0.3 * x + 0.6 :=
by
  sorry

end rent_expression_l420_420727


namespace find_length_of_BC_l420_420067

def circle_geometry_problem (O A B C D : Point) (BO : ℝ)
  (angle_ABO : ℝ) (arc_CD : ℝ) : Prop :=
  diameter O A D ∧ chord O A B C ∧ BO = 7 ∧ angle_ABO = 90 ∧ arc_CD = 90

theorem find_length_of_BC (O A B C D : Point) (BO : ℝ)
  (angle_ABO : ℝ) (arc_CD : ℝ) (h : circle_geometry_problem O A B C D BO angle_ABO arc_CD) :
  length B C = 7 := by
  sorry

end find_length_of_BC_l420_420067


namespace meaningful_fraction_l420_420911

theorem meaningful_fraction (x : ℝ) : 
  (sqrt (x + 1) / (x - 3)^2).isDefined ↔ (x ≥ -1 ∧ x ≠ 3) := by
  sorry

end meaningful_fraction_l420_420911


namespace tiles_per_row_l420_420600

theorem tiles_per_row (area : ℝ) (tile_length : ℝ) (h1 : area = 256) (h2 : tile_length = 2/3) : 
  (16 * 12) / (8) = 24 :=
by {
  sorry
}

end tiles_per_row_l420_420600


namespace arithmetic_mean_no_digit_eight_l420_420157

-- Definitions of the set and the condition that N does not contain the digit 8.
def set_of_numbers : List ℕ := [8, 88, 888, 8888, 88888, 888888, 8888888, 88888888, 888888888]

def arithmetic_mean (lst : List ℕ) : ℕ :=
  lst.sum / lst.length

def distinct_digits (n : ℕ) : Prop :=
  (n.digits.to_finset.card = n.digits.length)

def arithmetic_mean_of_set := arithmetic_mean set_of_numbers

-- The proof problem: Prove that the arithmetic mean does not contain the digit 8.
theorem arithmetic_mean_no_digit_eight :
  distinct_digits arithmetic_mean_of_set ∧ ¬ (8 ∈ arithmetic_mean_of_set.digits) :=
sorry

end arithmetic_mean_no_digit_eight_l420_420157


namespace prime_number_p_squared_divides_sum_or_cube_divides_sum_of_cubes_l420_420535

variable {p a b : ℤ}

theorem prime_number_p_squared_divides_sum_or_cube_divides_sum_of_cubes
  (hp : Prime p) (hp_ne_3 : p ≠ 3)
  (h1 : p ∣ (a + b)) (h2 : p^2 ∣ (a^3 + b^3)) :
  p^2 ∣ (a + b) ∨ p^3 ∣ (a^3 + b^3) :=
sorry

end prime_number_p_squared_divides_sum_or_cube_divides_sum_of_cubes_l420_420535


namespace probability_Alex_wins_4_Mel_wins_1_Chelsea_wins_1_l420_420068
-- Required import for Lean Math library

-- Defining the conditions for the problem
def prob_Alex_wins : ℝ := 1 / 3
def prob_Mel_wins : ℝ := 1 / 6
def prob_Chelsea_wins : ℝ := 1 / 2
def num_rounds : ℕ := 6

theorem probability_Alex_wins_4_Mel_wins_1_Chelsea_wins_1 :
    let outcome_probability := (prob_Alex_wins ^ 4) * (prob_Mel_wins) * (prob_Chelsea_wins) in
    let num_arrangements := Nat.choose 6 4 * Nat.choose 2 1 in
    outcome_probability * num_arrangements = 5 / 162 := by
  sorry

end probability_Alex_wins_4_Mel_wins_1_Chelsea_wins_1_l420_420068


namespace marge_final_plant_count_l420_420565

/-- Define the initial conditions of the garden -/
def initial_seeds : ℕ := 23
def marigold_seeds : ℕ := 10
def sunflower_seeds : ℕ := 8
def lavender_seeds : ℕ := 5
def seeds_without_growth : ℕ := 5

/-- Growth rates for each type of plant -/
def marigold_growth_rate : ℕ := 4
def sunflower_growth_rate : ℕ := 4
def lavender_growth_rate : ℕ := 3

/-- Impact of animals -/
def marigold_eaten_by_squirrels : ℕ := 2
def sunflower_eaten_by_rabbits : ℕ := 1

/-- Impact of pest control -/
def marigold_pest_control_reduction : ℕ := 0
def sunflower_pest_control_reduction : ℕ := 0
def lavender_pest_control_protected : ℕ := 2

/-- Impact of weeds -/
def weeds_strangled_plants : ℕ := 2

/-- Weeds left as plants -/
def weeds_kept_as_plants : ℕ := 1

/-- Marge's final number of plants -/
def survived_plants :=
  (marigold_growth_rate - marigold_eaten_by_squirrels - marigold_pest_control_reduction) +
  (sunflower_growth_rate - sunflower_eaten_by_rabbits - sunflower_pest_control_reduction) +
  (lavender_growth_rate - (lavender_growth_rate - lavender_pest_control_protected)) - weeds_strangled_plants

theorem marge_final_plant_count :
  survived_plants + weeds_kept_as_plants = 6 :=
by
  sorry

end marge_final_plant_count_l420_420565


namespace overall_yield_percentage_is_4point22_l420_420150

-- Definitions for the given conditions
def stockA_value := 500
def stockB_value := 750
def stockC_value := 1000

def stockA_yield := 0.14
def stockB_yield := 0.08
def stockC_yield := 0.12

def tax_rate := 0.02
def commission_fee := 50

def total_market_value := stockA_value + stockB_value + stockC_value

def yield_after_tax (yield before_tax : ℝ) := yield_before_tax - yield_before_tax * tax_rate

def stockA_yield_after_tax := yield_after_tax (stockA_yield * stockA_value)
def stockB_yield_after_tax := yield_after_tax (stockB_yield * stockB_value)
def stockC_yield_after_tax := yield_after_tax (stockC_yield * stockC_value)

def total_yield_after_tax := stockA_yield_after_tax + stockB_yield_after_tax + stockC_yield_after_tax
def total_commission_fees := 3 * commission_fee

def total_yield_after_commission := total_yield_after_tax - total_commission_fees

def overall_yield_percentage := (total_yield_after_commission / total_market_value) * 100

-- The final proof statement that needs to be proved
theorem overall_yield_percentage_is_4point22 : overall_yield_percentage = 4.22 := 
sorry

end overall_yield_percentage_is_4point22_l420_420150


namespace area_of_transformed_region_l420_420546

noncomputable def matrix_transformation_area (A : Matrix (Fin 2) (Fin 2) ℝ) (area_T : ℝ) : ℝ :=
  matrix.det A * area_T

theorem area_of_transformed_region : 
  let T_area := 15
  let A := Matrix.of_list 2 2 [[3, 4], [6, -2]]
  let T'_area := matrix_transformation_area A T_area
  T'_area = 450 :=
by
  let T_area := 15
  let A := Matrix.of_list 2 2 [[3, 4], [6, -2]]
  let T'_area := matrix_transformation_area A T_area
  show T'_area = 450
  sorry

end area_of_transformed_region_l420_420546


namespace simplify_polynomial_l420_420142

variable {R : Type*} [CommRing R]

theorem simplify_polynomial (x : R) :
  (12 * x ^ 10 + 9 * x ^ 9 + 5 * x ^ 8) + (2 * x ^ 12 + x ^ 10 + 2 * x ^ 9 + 3 * x ^ 8 + 4 * x ^ 4 + 6 * x ^ 2 + 9) =
  2 * x ^ 12 + 13 * x ^ 10 + 11 * x ^ 9 + 8 * x ^ 8 + 4 * x ^ 4 + 6 * x ^ 2 + 9 :=
  sorry

end simplify_polynomial_l420_420142


namespace lines_intersect_at_l420_420427

theorem lines_intersect_at :
  ∃ (x y : ℚ), 3 * y = -2 * x + 6 ∧ 7 * y = -3 * x - 4 ∧ x = 54 / 5 ∧ y = -26 / 5 := 
by
  sorry

end lines_intersect_at_l420_420427


namespace simplify_expression_l420_420140

noncomputable def i : ℂ := Complex.I

theorem simplify_expression : 7*(4 - 2*i) + 4*i*(3 - 2*i) = 36 - 2*i :=
by
  sorry

end simplify_expression_l420_420140


namespace registration_count_l420_420185

noncomputable def num_registration_results : ℕ := 36

theorem registration_count :
  let S := {ZheJiang, Fudan, ShanghaiJiaoTong} : Finset String,
      choices := Finset.powersetLen 1 S ∪ Finset.powersetLen 2 S,
      student1_choices := choices.product choices,
      student2_choices := choices.product choices in
  ∃ num : ℕ, num = student1_choices.card + student2_choices.card ∧ num = 36 :=
by
  intros S choices student1_choices student2_choices
  use 36
  rw [Finset.card_product, Finset.card_univ, Finset.card_powersetLen, Finset.union_card_disjoint]
  simp only [Finset.card_powersetLen]
  sorry

end registration_count_l420_420185


namespace greatest_area_difference_l420_420663

theorem greatest_area_difference 
  (l w l' w' : ℕ) 
  (h1 : 2 * l + 2 * w = 160) 
  (h2 : 2 * l' + 2 * w' = 160) :
  abs ((l * w) - (l' * w')) ≤ 1521 := 
sorry

end greatest_area_difference_l420_420663


namespace number_of_tiles_per_row_l420_420596

-- Define the conditions
def area (a : ℝ) : ℝ := a * a
def side_length (area : ℝ) : ℝ := real.sqrt area
def feet_to_inches (feet : ℝ) : ℝ := feet * 12
def tiles_per_row (room_length_inches : ℝ) (tile_size_inches : ℝ) : ℕ := 
  int.to_nat ⟨room_length_inches / tile_size_inches, by sorry⟩

-- Given constants in the problem
def area_of_room : ℝ := 256
def tile_size : ℝ := 8

-- Derived lengths
def length_of_side := side_length area_of_room
def length_of_side_in_inches := feet_to_inches length_of_side

-- The theorem to prove
theorem number_of_tiles_per_row : tiles_per_row length_of_side_in_inches tile_size = 24 :=
sorry

end number_of_tiles_per_row_l420_420596


namespace triangle_count_l420_420635

theorem triangle_count :
  let x y : ℕ in
  (∀ (x y : ℕ), x ≤ 8 → y ≤ 8 → x + y > 8 → x ≤ y → True) →
  ∃ (n : ℕ), n = 20 :=
by
  sorry

end triangle_count_l420_420635


namespace range_of_a_l420_420472

theorem range_of_a (a : ℝ) : 
  (∀ y : ℝ, ∃ x : ℝ, y = log (1/2) (x^2 - 2 * a * x + 3)) ↔ a ≥ real.sqrt 3 ∨ a ≤ -real.sqrt 3 := 
sorry

end range_of_a_l420_420472


namespace cost_of_article_l420_420334

variable (C G : ℝ)

theorem cost_of_article (h1 : 340 = C + G) (h2 : 350 = C + G + 0.05 * G) : C = 140 :=
by
  have h3 : 0.05 * G = 10 :=
    calc 0.05 * G = (350 - 340) : by linarith [h2, h1]
                 ... = 10 : by norm_num
  have h4 : G = 200 := by linarith
  linarith

end cost_of_article_l420_420334


namespace fish_left_in_the_sea_l420_420346

-- Define the given conditions
def fish_westward : ℕ := 1800
def fish_eastward : ℕ := 3200
def fish_north : ℕ := 500

-- Define the fractions of fish caught
def fraction_caught_eastward : ℚ := 2 / 5
def fraction_caught_westward : ℚ := 3 / 4

-- Calculate the number of caught fish for eastward and westward
def fish_caught_eastward : ℕ := (fraction_caught_eastward * fish_eastward).natAbs
def fish_caught_westward : ℕ := (fraction_caught_westward * fish_westward).natAbs

-- Total number of fish caught
def total_fish_caught : ℕ := fish_caught_eastward + fish_caught_westward

-- Total number of fish initially in the sea
def total_fish : ℕ := fish_westward + fish_eastward + fish_north

-- Proof statement: calculate the number of fish left in the sea
theorem fish_left_in_the_sea : total_fish - total_fish_caught = 2870 := by
  -- Lean should calculate it as expected
  sorry

end fish_left_in_the_sea_l420_420346


namespace area_triangle_theorem_l420_420615

noncomputable def area_of_triangle (a b c : ℝ) : ℝ :=
  real.sqrt (1 / 4 * (a ^ 2 * c ^ 2 - ( (a ^ 2 + c ^ 2 - b ^ 2) / 2) ^ 2))

theorem area_triangle_theorem (a b c : ℝ) (S : ℝ)
    (h1 : a ^ 2 * real.sin C = 4 * real.sin A)
    (h2 : (a + c) ^ 2 = 12 + b ^ 2) :
  S = area_of_triangle a b c → S = real.sqrt 3 :=
begin
  sorry
end

end area_triangle_theorem_l420_420615


namespace ninety_third_term_l420_420569

theorem ninety_third_term : 
  let seq (n m : Nat) := (m, n - m + 1)
  let total_terms_up_to (n : Nat) := (n * (n + 1)) / 2
  let group (n k : Nat) := (k, n - k + 1)
  total_terms_up_to 13 = 91 ∧ 
  group 14 2 = (2, 13) →
  nth (total_terms_up_to 13 + 2) = (2, 13) :=
begin
  sorry
end

end ninety_third_term_l420_420569


namespace max_number_of_people_l420_420116

open Classical

structure ItemPrices where
  sandwich : ℝ
  roll : ℝ
  pastry : ℝ
  juice_pack : ℝ
  small_soda : ℝ
  large_soda : ℝ

constant itemPrices : ItemPrices := { sandwich := 0.80, roll := 0.60, pastry := 1.00, juice_pack := 0.50, small_soda := 0.75, large_soda := 1.25 }

constant budget : ℝ := 12.50
constant min_spending_on_food : ℝ := 10.00

noncomputable def max_people_with_unique_combinations (itemPrices : ItemPrices) (budget min_spending_on_food : ℝ) : ℕ :=
  -- Function definition to calculate the maximum number of people
  sorry

theorem max_number_of_people : max_people_with_unique_combinations itemPrices budget min_spending_on_food = 5 :=
  by sorry

end max_number_of_people_l420_420116


namespace train_length_l420_420749

theorem train_length (speed_km_hr : ℝ) (time_sec : ℝ) (length_m : ℝ) 
  (h1 : speed_km_hr = 60) (h2 : time_sec = 9) : length_m = 150 := by
  sorry

end train_length_l420_420749


namespace correct_proposition_l420_420017

noncomputable def f1 (x : ℝ) : ℝ := x^3 - 3*x^2
noncomputable def f1_derivative (x : ℝ) : ℝ := 3*x^2 - 6*x

def proposition_1 := ∀ x : ℝ, f1_derivative x ≠ 0 -> f1_derivative x > 0
def proposition_2 := ∀ x : ℝ, x < 2 -> ∃ c : ℝ, f1 c > f1 x
def integral :=∫ (a : ℝ) in 0..1, a - a^2
def proposition_3 := integral = (1/6 : ℝ)

noncomputable def f4 (x a : ℝ) : ℝ := Real.log x + a*x
noncomputable def f4_derivative (x a : ℝ) : ℝ := (1 / x) + a

def tangent_line_parallel (f : ℝ → ℝ) (x : ℝ) : Prop := ∃ y : ℝ, deriv f x = 2
def proposition_4 := ∀ a : ℝ, a < 2 ∧ ∃ x : ℝ, x > 0 ∧ f4_derivative x a = 2

theorem correct_proposition : proposition_3 :=
by
  sorry

end correct_proposition_l420_420017


namespace platform_length_is_correct_l420_420348

-- Given Definitions
def length_of_train : ℝ := 300
def time_to_cross_platform : ℝ := 42
def time_to_cross_pole : ℝ := 18

-- Definition to prove
theorem platform_length_is_correct :
  ∃ L : ℝ, L = 400 ∧ (length_of_train + L) / time_to_cross_platform = length_of_train / time_to_cross_pole :=
by
  sorry

end platform_length_is_correct_l420_420348


namespace largest_number_of_gold_coins_l420_420318

theorem largest_number_of_gold_coins (n : ℕ) :
  (∃ k : ℕ, n = 13 * k + 3 ∧ n < 150) → n ≤ 146 :=
by
  sorry

end largest_number_of_gold_coins_l420_420318


namespace compare_f_neg_x1_neg_x2_l420_420844

noncomputable def f : ℝ → ℝ := sorry

theorem compare_f_neg_x1_neg_x2 
  (h1 : ∀ x : ℝ, f (1 + x) = f (1 - x)) 
  (h2 : ∀ x y : ℝ, 1 ≤ x → 1 ≤ y → x < y → f x < f y)
  (x1 x2 : ℝ)
  (hx1 : x1 < 0)
  (hx2 : x2 > 0)
  (hx1x2 : x1 + x2 < -2) :
  f (-x1) > f (-x2) :=
by sorry

end compare_f_neg_x1_neg_x2_l420_420844


namespace polynomial_g_eq_find_g_of_x2_minus_2_l420_420101

open Polynomial

noncomputable def g (x : ℝ) : ℝ := x^2 + 2 * x - 4

theorem polynomial_g_eq (x : ℝ) : g (x^2 + 2) = x^4 + 6 * x^2 + 4 :=
by sorry

theorem find_g_of_x2_minus_2 (x : ℝ) : g(x^2 - 2) = x^4 - 2 * x^2 - 4 :=
by sorry

end polynomial_g_eq_find_g_of_x2_minus_2_l420_420101


namespace percentage_gain_is_27_point_27_l420_420736

-- Define the conditions
def cost_per_bowl : ℝ := 10
def selling_price_per_bowl : ℝ := 14
def initial_bowls : ℝ := 110
def sold_bowls : ℝ := 100
def broken_bowls : ℝ := initial_bowls - sold_bowls

-- Define the calculations for percentage gain
def total_cost : ℝ := initial_bowls * cost_per_bowl
def total_selling_price : ℝ := sold_bowls * selling_price_per_bowl
def gain : ℝ := total_selling_price - total_cost
def percentage_gain : ℝ := (gain / total_cost) * 100

-- The statement of the problem we want to prove
theorem percentage_gain_is_27_point_27 :
  percentage_gain = 27.27 := by
  sorry

end percentage_gain_is_27_point_27_l420_420736


namespace largest_coins_l420_420322

theorem largest_coins (n k : ℕ) (h1 : n = 13 * k + 3) (h2 : n < 150) : n = 146 :=
by
  sorry

end largest_coins_l420_420322


namespace tangent_line_problem_l420_420916

theorem tangent_line_problem 
  (x1 x2 : ℝ)
  (h1 : (1 / x1) = Real.exp x2)
  (h2 : Real.log x1 - 1 = Real.exp x2 * (1 - x2)) :
  (2 / (x1 - 1) + x2 = -1) :=
by 
  sorry

end tangent_line_problem_l420_420916


namespace pentagon_area_l420_420366

def isosceles_right_triangle_area (a : ℝ) : ℝ :=
1 / 2 * a * a

def isosceles_triangle_120_area (a b : ℝ) (deg120 : ℝ) : ℝ :=
1 / 2 * a * b * Real.sin (deg120 * Real.pi / 180)

theorem pentagon_area :
  let FG := 3
  let GH := 3 * Real.sqrt 2
  let HI := 3
  let IJ := 3 * Real.sqrt 2
  let FJ := 6
  let angle_G_FGJ := 120
  let area_FGH := isosceles_right_triangle_area FG -- Note: GH = FG = 3
  let area_HIJ := isosceles_right_triangle_area HI -- Note: HI = IJ = 3
  let area_FGJ := isosceles_triangle_120_area FG FJ angle_G_FGJ
  in area_FGH + area_HIJ + area_FGJ = 9 + 4.5 * Real.sqrt 3 :=
by sorry

end pentagon_area_l420_420366


namespace math_problem_min_k_triangulation_l420_420442

theorem math_problem_min_k_triangulation
    (S : Finset ℕ)
    (hS : ∀ x ∈ S, 1 ≤ x ∧ x ≤ 2004)
    (h_triangle : ∀ T ⊆ S, T.card = 3 → ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
                      a < b + c ∧ b < a + c ∧ c < a + b) :
  17 ≤ S.card :=
sorry

end math_problem_min_k_triangulation_l420_420442


namespace eval_expr_l420_420396

theorem eval_expr : 3 + (-3) ^ (-3) = 80 / 27 := 
by
  sorry

end eval_expr_l420_420396


namespace loraine_wax_sculptures_l420_420113

theorem loraine_wax_sculptures :
  (∃ (large_animals small_animals : ℕ), 
    small_animals = 5 * large_animals ∧ 
    30 = 3 * small_animals ∧ 
    42 = 6 * large_animals + 30) :=
begin
  sorry
end

end loraine_wax_sculptures_l420_420113


namespace vector_addition_l420_420838

def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (-3, 4)

theorem vector_addition :
  2 • a + b = (1, 2) :=
by
  sorry

end vector_addition_l420_420838


namespace breadth_halved_of_percentage_change_area_l420_420630

theorem breadth_halved_of_percentage_change_area {L B B' : ℝ} (h : 0 < L ∧ 0 < B) 
  (h1 : L / 2 * B' = 0.5 * (L * B)) : B' = 0.5 * B :=
sorry

end breadth_halved_of_percentage_change_area_l420_420630


namespace proof_statement_l420_420340

variables {K_c A_c K_d B_d A_d B_c : ℕ}

def conditions (K_c A_c K_d B_d A_d B_c : ℕ) :=
  K_c > A_c ∧ K_d > B_d ∧ A_d > K_d ∧ B_c > A_c

noncomputable def statement (K_c A_c K_d B_d A_d B_c : ℕ) (h : conditions K_c A_c K_d B_d A_d B_c) : Prop :=
  A_d > max K_d B_d

theorem proof_statement (K_c A_c K_d B_d A_d B_c : ℕ) (h : conditions K_c A_c K_d B_d A_d B_c) : statement K_c A_c K_d B_d A_d B_c h :=
sorry

end proof_statement_l420_420340


namespace trajectory_equation_line_equation_l420_420075

-- Condition definitions
variables {x y : ℝ}
def a : ℝ × ℝ := (2 * x + 3, y)
def b : ℝ × ℝ := (2 * x - 3, 3 * y)

-- Conditions
def dot_product_condition : Prop :=
  (a.1 * b.1 + a.2 * b.2 = 3)

-- The trajectory equation of point P
theorem trajectory_equation (h : dot_product_condition) : 
  let ellipse_eq : Prop := (x^2) / 3 + (y^2) / 4 = 1
  ellipse_eq :=
sorry

-- Line condition and intersection
noncomputable def intersect_line_ellipse (k : ℝ) (x1 x2 : ℝ) : Prop :=
  let lhs := ((4 + 3 * k^2) * x^2 + 6 * k * x - 9)
  let d := 16 / 5
  lhs = d

theorem line_equation (h : dot_product_condition) : 
  let line_eq_1 : Prop := y = (Real.sqrt 3) / 3 * x + 1
  let line_eq_2 : Prop := y = - (Real.sqrt 3) / 3 * x + 1
  (∃ k : ℝ, intersect_line_ellipse k x1 x2) → (line_eq_1 ∨ line_eq_2) :=
sorry

end trajectory_equation_line_equation_l420_420075


namespace find_a_l420_420880

noncomputable def f : ℝ → ℝ :=
  λ x, if x > 0 then Real.log x / Real.log 2 else 2^x

theorem find_a :
  {a : ℝ | f a = 0.5} = {-1, Real.sqrt 2} :=
by
  sorry

end find_a_l420_420880


namespace sum_of_weights_is_three_pow_seventyfive_l420_420799

def latin_alphabet := fin 26

def weight (used_letters : finset latin_alphabet) : ℚ :=
  1 / (latin_alphabet.card - used_letters.card + 1)

def all_possible_words : finset (latin_alphabet → latin_alphabet) :=
  univ.product (finset.univ : finset (latin_alphabet → latin_alphabet))

def sum_of_weights : ℚ :=
  (all_possible_words.sum (λ word, weight (latin_alphabet.filter (λ l, word l ∈ univ))))

theorem sum_of_weights_is_three_pow_seventyfive :
  sum_of_weights = 3^75 :=
sorry

end sum_of_weights_is_three_pow_seventyfive_l420_420799


namespace number_of_ways_to_pair_l420_420646

-- Define the people as a finite set.
inductive Person : Type
| p1 | p2 | p3 | p4 | p5 | p6 | p7 | p8

open Person

-- Define the knows relation as per the given problem.
def knows : Person → Person → Prop
| p1 p2 := true
| p1 p3 := true
| p1 p5 := true
| p2 p1 := true
| p2 p3 := true
| p2 p4 := true
| p3 p2 := true
| p3 p4 := true
| p3 p6 := true
| p4 p3 := true
| p4 p5 := true
| p4 p2 := true
| p5 p4 := true
| p5 p6 := true
| p5 p1 := true
| p6 p5 := true
| p6 p7 := true
| p6 p3 := true
| p7 p6 := true
| p7 p8 := true
| p7 p4 := true
| p8 p7 := true
| p8 p1 := true
| p8 p5 := true
| _ _ := false

-- Define the main theorem with the correct answer.
theorem number_of_ways_to_pair :
  {pairs // ∀p1 p2, p1 ∈ pairs -> p2 ∈ pairs -> knows p1 p2} = 8 :=
by sorry

end number_of_ways_to_pair_l420_420646


namespace find_a_for_exponential_function_l420_420049

theorem find_a_for_exponential_function (a : ℝ) :
  a - 2 = 1 ∧ a > 0 ∧ a ≠ 1 → a = 3 :=
by
  intro h
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end find_a_for_exponential_function_l420_420049


namespace miles_drivable_l420_420566

theorem miles_drivable (mileage_per_gallon cost_per_gallon money : ℕ) (h1 : mileage_per_gallon = 40) (h2 : cost_per_gallon = 5) 
    (h3 : money = 25) : 
    mileage_per_gallon * (money / cost_per_gallon) = 200 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end miles_drivable_l420_420566


namespace lemuel_total_points_l420_420533

theorem lemuel_total_points (two_point_shots : ℕ) (three_point_shots : ℕ) (points_from_two : ℕ) (points_from_three : ℕ) :
  two_point_shots = 7 →
  three_point_shots = 3 →
  points_from_two = 2 →
  points_from_three = 3 →
  two_point_shots * points_from_two + three_point_shots * points_from_three = 23 :=
by
  sorry

end lemuel_total_points_l420_420533


namespace find_a_and_b_find_sqrt_4a_plus_b_l420_420874

section

variable {a b : ℤ}

-- Define the conditions
def condition1 (a b : ℤ) := ((3 * a - 14)^2 = (a - 2)^2) 
def condition2 (b : ℤ) := (b - 15) ^ (3 : ℝ) = -27

-- Prove the given values of a and b
theorem find_a_and_b : ∃ a b : ℤ, condition1 a b ∧ condition2 b :=
begin
  use [4, -12],
  split,
  -- prove condition1
  { unfold condition1, norm_num },
  -- prove condition2
  { unfold condition2, norm_num }
end

-- Prove the square root of 4a + b
theorem find_sqrt_4a_plus_b (a b : ℤ) (h₁ : condition1 a b) (h₂ : condition2 b) : 
  sqrt (4 * a + b) = 2 ∨ sqrt (4 * a + b) = -2 :=
begin
  -- use given conditions to prove the statement
  have ha : a = 4, from sorry,
  have hb : b = -12, from sorry,
  rw [ha, hb],
  norm_num,
  left,
  norm_num,
end

end

end find_a_and_b_find_sqrt_4a_plus_b_l420_420874


namespace necessary_and_sufficient_condition_l420_420046

theorem necessary_and_sufficient_condition (a : ℝ) : (a > 0) ↔ (a + 1 / a ≥ 2) :=
sorry

end necessary_and_sufficient_condition_l420_420046


namespace total_area_of_triangular_faces_l420_420201

noncomputable def pyramid_base_edge : ℝ := 8
noncomputable def pyramid_lateral_edge : ℝ := 7

structure Pyramid where
  base_edge : ℝ
  lateral_edge : ℝ

def myPyramid : Pyramid :=
{ base_edge := pyramid_base_edge,
  lateral_edge := pyramid_lateral_edge }

theorem total_area_of_triangular_faces :
  4 * (1 / 2) * myPyramid.base_edge * (Real.sqrt (myPyramid.lateral_edge^2 - (myPyramid.base_edge / 2)^2)) = 16 * Real.sqrt 33 :=
by
  sorry

end total_area_of_triangular_faces_l420_420201


namespace circle_area_difference_l420_420042

noncomputable def area (r : ℝ) : ℝ := Real.pi * r^2

theorem circle_area_difference :
  let radius1 := 20
  let diameter2 := 20
  let radius2 := diameter2 / 2
  area radius1 - area radius2 = 300 * Real.pi :=
by
  sorry

end circle_area_difference_l420_420042


namespace harmonic_series_inequality_l420_420097

theorem harmonic_series_inequality (n : ℕ) (h₁ : 2 ≤ n) :
  (4 / 7 : ℝ) < (∑ i in range(n), if i.even then -(1 / (i + 1)) else 1 / (i + 1)) < (real.sqrt 2 / 2) :=
sorry

end harmonic_series_inequality_l420_420097


namespace geometric_sequence_common_ratio_l420_420164

theorem geometric_sequence_common_ratio (q : ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, S n = 1 / (1 - q)) 
  (h2 : q > 0)
  (h3 : tendsto S at_top (𝓝 2)) :
  q = 1 / 2 :=
sorry

end geometric_sequence_common_ratio_l420_420164


namespace slope_angle_of_line_l420_420823

def slope (x y : ℝ) : ℝ := -1 / √3

theorem slope_angle_of_line :
  let m := slope 1 (√3)
  θ := Real.arctan m
  θ = 150 :=
  sorry

end slope_angle_of_line_l420_420823


namespace pyramid_area_l420_420278

theorem pyramid_area (base_edge lateral_edge : ℝ) (H_base_edge : base_edge = 8) (H_lateral_edge : lateral_edge = 7) :
  4 * (1 / 2 * base_edge * real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 16 * real.sqrt 33 :=
by
  sorry

end pyramid_area_l420_420278


namespace angle_ratio_l420_420517

variables (EFGH : Type) [parallelogram EFGH]
variables (E F G H O : point EFGH)
variables (phi s : ℝ)

-- Given angles in a parallelogram
axiom angle_EFG_eq_3phi : ∠(E F G) = 3 * φ
axiom angle_HGF_eq_3phi : ∠(H G F) = 3 * φ
axiom angle_HGE_eq_phi : ∠(H G E) = φ
axiom angle_EGF_eq_s_angle_EOH : ∠(E G F) = s * ∠(E O H)
axiom O_diagonal_intersection : ∃ O, is_intersection_of_diagonals O E G F H

-- Prove the ratio s
theorem angle_ratio:
  s = sin(7 * φ) / sin(2 * φ) :=
sorry

end angle_ratio_l420_420517


namespace perpendicular_lines_slope_l420_420054

theorem perpendicular_lines_slope (a : ℝ) :
  (∀ x y : ℝ, ax + y - 5 = 0) ∧ (∀ x y : ℝ, y = 7x - 2) → a = 1 / 7 :=
by
  sorry

end perpendicular_lines_slope_l420_420054


namespace jackson_class_probability_l420_420513

-- Define the conditions
def distinct_initials (class_size : ℕ) : Prop :=
  ∀ (i j : ℕ), i < class_size → j < class_size → i ≠ j → (initials i ≠ initials j)

def initials (n : ℕ) : Char := sorry   -- Imagine this gives the initials of the n-th student

def vowel (c : Char) : Prop :=
  c = 'A' ∨ c = 'E' ∨ c = 'I' ∨ c = 'O' ∨ c = 'U'

-- The problem statement
theorem jackson_class_probability (class_size : ℕ) 
  (h_class_size : class_size = 26) 
  (h_distinct : distinct_initials class_size) :
  (prob_vowel : ℚ) := 
  prob_vowel = 5 / 26 :=
sorry

end jackson_class_probability_l420_420513


namespace intersection_point_parallel_line_through_intersection_l420_420035

-- Definitions for the problem
def l1 (x y : ℝ) : Prop := x + 8 * y + 7 = 0
def l2 (x y : ℝ) : Prop := 2 * x + y - 1 = 0
def l3 (x y : ℝ) : Prop := x + y + 1 = 0
def parallel (x y c : ℝ) : Prop := x + y + c = 0
def point (x y : ℝ) : Prop := x = 1 ∧ y = -1

-- (1) Proof that the intersection point of l1 and l2 is (1, -1)
theorem intersection_point : ∃ (x y : ℝ), l1 x y ∧ l2 x y ∧ point x y :=
by 
  sorry

-- (2) Proof that the line passing through the intersection point of l1 and l2
-- which is parallel to l3 is x + y = 0
theorem parallel_line_through_intersection : ∃ (c : ℝ), parallel 1 (-1) c ∧ c = 0 :=
by 
  sorry

end intersection_point_parallel_line_through_intersection_l420_420035


namespace divide_circle_into_11_eq_8_over_11_l420_420413

theorem divide_circle_into_11_eq_8_over_11 :
  let unit_circle := 1 in
  let part := unit_circle / 11 in
  8 * part = 8 / 11 :=
by
  -- Circle is divided into 11 parts, each part is 1/11
  let unit_circle := 1
  let part := unit_circle / 11
  -- We want to prove 8 parts is 8/11
  have : 8 * part = 8 / 11,
  {
    sorry,
  }
  exact this

end divide_circle_into_11_eq_8_over_11_l420_420413


namespace smallest_integer_neither_prime_nor_square_no_prime_factor_lt_60_l420_420691

theorem smallest_integer_neither_prime_nor_square_no_prime_factor_lt_60 :
  ∃ (n : ℕ), n = 4091 ∧ ¬(prime n) ∧ ¬(∃ (k : ℕ), n = k * k) ∧ (∀ p : ℕ, prime p → p ∣ n → p ≥ 60) :=
by {
  -- skip the proof with sorry
  sorry
}

end smallest_integer_neither_prime_nor_square_no_prime_factor_lt_60_l420_420691


namespace gcd_9_factorial_7_factorial_square_l420_420426

theorem gcd_9_factorial_7_factorial_square : Nat.gcd (Nat.factorial 9) ((Nat.factorial 7) ^ 2) = 362880 :=
by
  sorry

end gcd_9_factorial_7_factorial_square_l420_420426


namespace largest_number_of_gold_coins_l420_420327

theorem largest_number_of_gold_coins 
  (num_friends : ℕ)
  (extra_coins : ℕ)
  (total_coins : ℕ) :
  num_friends = 13 →
  extra_coins = 3 →
  total_coins < 150 →
  ∀ k : ℕ, total_coins = num_friends * k + extra_coins →
  total_coins ≤ 146 :=
by
  sorry

end largest_number_of_gold_coins_l420_420327


namespace cubic_inches_in_one_cubic_foot_l420_420406

-- Definition for the given conversion between foot and inches
def foot_to_inches : ℕ := 12

-- The theorem to prove the cubic conversion
theorem cubic_inches_in_one_cubic_foot : (foot_to_inches ^ 3) = 1728 := by
  -- Skipping the actual proof
  sorry

end cubic_inches_in_one_cubic_foot_l420_420406


namespace train_length_l420_420747

theorem train_length (speed_km_hr : ℝ) (time_sec : ℝ) (length_m : ℝ) 
  (h1 : speed_km_hr = 60) (h2 : time_sec = 9) : length_m = 150 := by
  sorry

end train_length_l420_420747


namespace area_of_region_bounded_by_curves_l420_420189

noncomputable def area_under_curves : ℝ :=
  ∫ x in 0..1, (x ^ (1 / 2003) - x ^ 2003)

theorem area_of_region_bounded_by_curves : 
  area_under_curves = 1001 / 1002 :=
by {
  sorry
}

end area_of_region_bounded_by_curves_l420_420189


namespace pyramid_area_l420_420283

theorem pyramid_area (base_edge lateral_edge : ℝ) (H_base_edge : base_edge = 8) (H_lateral_edge : lateral_edge = 7) :
  4 * (1 / 2 * base_edge * real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 16 * real.sqrt 33 :=
by
  sorry

end pyramid_area_l420_420283


namespace count_of_tangent_lines_l420_420836

-- Definitions for the radii and the distance between the centers
def radius1 := 4
def radius2 := 6
def d : ℝ -- distance between the centers of the two circles

-- Theorem stating the number of different values for k
theorem count_of_tangent_lines : ∃ (k_values : Finset ℕ), k_values.card = 5 := sorry

end count_of_tangent_lines_l420_420836


namespace min_circled_medians_max_circled_medians_l420_420405

-- Defining the 3x3 table and the median concept
structure MedianTable where
  table : Fin 3 × Fin 3 → ℕ
  cond : ∀ i j, table i j ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Defining medians for rows, columns, and diagonals
def row_median (mt : MedianTable) (i : Fin 3) : ℕ :=
  let r := List.sort (List.map (mt.table i) [⟨0, Nat.zero_lt_succ 2⟩, ⟨1, by decide⟩, ⟨2, by decide⟩])
  r.nthLe 1 (by decide)

def col_median (mt : MedianTable) (j : Fin 3) : ℕ :=
  let c := List.sort (List.map (λ i => mt.table i j) [⟨0, Nat.zero_lt_succ 2⟩, ⟨1, by decide⟩, ⟨2, by decide⟩])
  c.nthLe 1 (by decide)

def main_diag_median (mt : MedianTable) : ℕ :=
  let d := List.sort [mt.table ⟨0, Nat.zero_lt_succ 2⟩ ⟨0, Nat.zero_lt_succ 2⟩,
                      mt.table ⟨1, by decide⟩ ⟨1, by decide⟩,
                      mt.table ⟨2, by decide⟩ ⟨2, by decide⟩]
  d.nthLe 1 (by decide)

def anti_diag_median (mt : MedianTable) : ℕ :=
  let d := List.sort [mt.table ⟨0, Nat.zero_lt_succ 2⟩ ⟨2, by decide⟩,
                      mt.table ⟨1, by decide⟩ ⟨1, by decide⟩,
                      mt.table ⟨2, by decide⟩ ⟨0, Nat.zero_lt_succ 2⟩]
  d.nthLe 1 (by decide)

-- Proving the minimum and maximum number of circled medians
theorem min_circled_medians (mt : MedianTable) : ∃ mt',
  row_median mt' 0 ≠ row_median mt' 1 ∧ row_median mt' 0 ≠ row_median mt' 2 ∧
  col_median mt' 0 ≠ col_median mt' 1 ∧ col_median mt' 0 ≠ col_median mt' 2 ∧
  mt'.main_diag_median ≠ mt'.anti_diag_median ∧
  ∑ (i : Fin 2), [mt'.row_median i = mt'.main_diag_median ∨ 
                  mt'.col_median i = mt'.anti_diag_median] = 1 := sorry

theorem max_circled_medians (mt : MedianTable) : ∃ mt',
  row_median mt' 0 ≠ row_median mt' 1 ∧ row_median mt' 0 ≠ row_median mt' 2 ∧
  col_median mt' 0 ≠ col_median mt' 1 ∧ col_median mt' 0 ≠ col_median mt' 2 ∧
  mt'.main_diag_median ≠ mt'.anti_diag_median ∧
  ∑ (i : Fin 6), [mt'.row_median i = mt'.main_diag_median ∨ 
                  mt'.col_median i = mt'.anti_diag_median] = 5 := sorry

end min_circled_medians_max_circled_medians_l420_420405


namespace find_x_l420_420500

variable (x : ℝ)

def condition1 : ℝ := 5 * x

def condition2 : ℝ := condition1 x - (1 / 3) * condition1 x

def condition3 : ℝ := condition2 x / 10

def condition4 : ℝ := condition3 x + (1 / 3) * x + (1 / 2) * x + (1 / 4) * x

theorem find_x : condition4 x = 68 → x = 48 :=
by sorry

end find_x_l420_420500


namespace slower_speed_percentage_l420_420362

theorem slower_speed_percentage (S S' T T' D : ℝ) (h1 : T = 8) (h2 : T' = T + 24) (h3 : D = S * T) (h4 : D = S' * T') : 
  (S' / S) * 100 = 25 := by
  sorry

end slower_speed_percentage_l420_420362


namespace average_age_after_leaves_is_27_l420_420158

def average_age_of_remaining_people (initial_avg_age : ℕ) (initial_people_count : ℕ) 
    (age_leave1 : ℕ) (age_leave2 : ℕ) (remaining_people_count : ℕ) : ℕ :=
  let initial_total_age := initial_avg_age * initial_people_count
  let new_total_age := initial_total_age - (age_leave1 + age_leave2)
  new_total_age / remaining_people_count

theorem average_age_after_leaves_is_27 :
  average_age_of_remaining_people 25 6 20 22 4 = 27 :=
by
  -- Proof is skipped
  sorry

end average_age_after_leaves_is_27_l420_420158


namespace pyramid_triangular_face_area_l420_420213

theorem pyramid_triangular_face_area 
  (base : ℝ) (lateral : ℝ)
  (h_base : base = 8) (h_lateral : lateral = 7) :
  let height := sqrt (lateral^2 - (base / 2)^2) in
  4 * (1 / 2 * base * height) = 16 * sqrt 33 := 
by
  sorry

end pyramid_triangular_face_area_l420_420213


namespace unique_number_not_in_range_l420_420617

-- Let's define the function f
noncomputable def f (a b c d x : ℝ) : ℝ := (a * x + b) / (c * x + d)

-- Define the conditions from the problem
axiom a_ne_zero (a b c d : ℝ) : a ≠ 0
axiom b_ne_zero (a b c d : ℝ) : b ≠ 0
axiom c_ne_zero (a b c d : ℝ) : c ≠ 0
axiom d_ne_zero (a b c d : ℝ) : d ≠ 0

-- Define the functional properties
axiom f_11 (a b c d : ℝ) : f a b c d 11 = 11
axiom f_41 (a b c d : ℝ) : f a b c d 41 = 41
axiom f_involution (a b c d : ℝ) (x : ℝ) (h : x ≠ -d/c) : f a b c d (f a b c d x) = x

-- Define the main conjecture
theorem unique_number_not_in_range (a b c d : ℝ) (h1 : a_ne_zero a b c d) (h2 : b_ne_zero a b c d)
  (h3 : c_ne_zero a b c d) (h4 : d_ne_zero a b c d) (h5 : f_11 a b c d) (h6 : f_41 a b c d)
  (h7 : ∀ x : ℝ, x ≠ -d/c → f_involution a b c d x (by assuming x ≠ -d / c)) :
  ∃ x : ℝ, x = a / 12 ∧ ∀ y : ℝ, y = f a b c d x → y ≠ a / 12 :=
sorry

end unique_number_not_in_range_l420_420617


namespace cos_frac_less_sin_frac_l420_420969

theorem cos_frac_less_sin_frac : 
  let a := Real.cos (3 / 2)
  let b := Real.sin (1 / 10)
  a < b :=
by
  let a := Real.cos (3 / 2)
  let b := Real.sin (1 / 10)
  sorry -- proof skipped

end cos_frac_less_sin_frac_l420_420969


namespace complex_midpoint_proof_l420_420076

def is_midpoint (zA zB zC : ℂ) : Prop := zC = (zA + zB) / 2

theorem complex_midpoint_proof :
  ∀ (zA zB zC : ℂ),
  zA = 6 + 5 * complex.i →
  zB = -2 + 3 * complex.i →
  is_midpoint zA zB zC →
  zC = 2 + 4 * complex.i := by
  intros zA zB zC hA hB hMidpoint
  sorry

end complex_midpoint_proof_l420_420076


namespace find_n_l420_420833

noncomputable def binom (n k : ℕ) : ℕ := nat.choose n k

theorem find_n (x a n : ℕ) (h1 : binom n 2 * x^(n - 2) * a^2 = 84) 
    (h2 : binom n 3 * x^(n - 3) * a^3 = 280) 
    (h3 : binom n 4 * x^(n - 4) * a^4 = 560) : 
    n = 7 :=
by
  sorry

end find_n_l420_420833


namespace chord_length_in_circle_l_l420_420521

noncomputable def circle_eqn_cartesian : ℝ × ℝ → Prop := 
  λ p, (p.1 - 2)^2 + p.2^2 = 4

noncomputable def line_eqn : ℝ × ℝ → Prop := 
  λ p, 4 * p.1 - 3 * p.2 - 3 = 0

def chord_length_proof (t : ℝ) : ℝ :=
  2 * Real.sqrt(4 - 1^2)

theorem chord_length_in_circle_l :
  ∀ (t : ℝ), ∃ (x y : ℝ), circle_eqn_cartesian (x, y) ∧ line_eqn (x, y) ∧ chord_length_proof t = 2 * Real.sqrt 3 := 
  by
    intros t
    use 0
    use 0
    sorry

end chord_length_in_circle_l_l420_420521


namespace tangent_lines_ln_e_proof_l420_420919

noncomputable def tangent_tangent_ln_e : Prop :=
  ∀ (x₁ x₂ : ℝ) 
  (h₁ : x₁ > 0) 
  (h₁_eq : x₂ = -Real.log x₁)
  (h₂_eq : Real.log x₁ - 1 = (Real.exp x₂) * (1 - x₂)),
  (2 / (x₁ - 1) + x₂ = -1)

theorem tangent_lines_ln_e_proof : tangent_tangent_ln_e :=
  sorry

end tangent_lines_ln_e_proof_l420_420919


namespace pyramid_area_l420_420242

theorem pyramid_area :
  ∀ (a b : ℝ), a = 8 → b = 7 → 4 * (1/2 * a * sqrt (b^2 - (a/2)^2)) = 16 * sqrt 33 :=
by
  intros a b ha hb
  rw [ha, hb]
  have h1 : a / 2 = 4 := by norm_num [ha]
  have h2 : b^2 - (a / 2)^2 = 33 :=
    by
      calc
        b^2 - (a / 2)^2 = 49 - 16 := by norm_num [hb]
        ... = 33 := by norm_num
  rw [h1, h2, sqrt 33, mul_one, mul_one, half_mul, mul_comm (1/2) a, ←mul_assoc, mul_comm 4 4]
  norm_num
  sorry

end pyramid_area_l420_420242


namespace lindy_total_distance_l420_420084

theorem lindy_total_distance (d_jc : ℕ) (v_j : ℕ) (v_c : ℕ) (v_l : ℕ) (t : ℕ) 
    (distance_jc : d_jc = 360) (speed_jack : v_j = 6) (speed_christina : v_c = 4) (speed_lindy : v_l = 12) 
    (time_to_meet : t = d_jc / (v_j + v_c)) :
    v_l * t = 432 := 
by 
simp [distance_jc, speed_jack, speed_christina, speed_lindy, time_to_meet]
sorry

end lindy_total_distance_l420_420084


namespace harmonious_union_not_real_l420_420093

def harmonious_set (S : set ℝ) : Prop :=
S.nonempty ∧ ∀ a b : ℝ, a ∈ S → b ∈ S → (a + b) ∈ S ∧ (a - b) ∈ S

theorem harmonious_union_not_real :
∃ S1 S2 : set ℝ, harmonious_set S1 ∧ harmonious_set S2 ∧ S1 ∪ S2 ≠ set.univ :=
by
  sorry

end harmonious_union_not_real_l420_420093


namespace necessary_but_not_sufficient_condition_l420_420000

-- Given conditions and translated inequalities
variable {x : ℝ}
variable (h_pos : 0 < x) (h_bound : x < π / 2)
variable (h_sin_pos : 0 < Real.sin x) (h_sin_bound : Real.sin x < 1)

-- Define the inequalities we are dealing with
def ineq_1 (x : ℝ) := Real.sqrt x - 1 / Real.sin x < 0
def ineq_2 (x : ℝ) := 1 / Real.sin x - x > 0

-- The main proof statement
theorem necessary_but_not_sufficient_condition 
  (h1 : ineq_1 x) 
  (hx : 0 < x) (hπ : x < π/2) : 
  ineq_2 x → False := by
  sorry

end necessary_but_not_sufficient_condition_l420_420000


namespace f_simplify_f_value_l420_420468

def in_third_quadrant (α : ℝ) : Prop :=
  π < α ∧ α < 3 * π / 2

def f (α : ℝ) : ℝ :=
  (sin (α - π / 2) * cos (3 * π / 2 + α) * tan (π - α)) /
  (tan (-α - π) * sin (-π - α))

theorem f_simplify (α : ℝ) (h : in_third_quadrant α) : f α = -cos α :=
sorry

theorem f_value (α : ℝ) (h : in_third_quadrant α) (h1 : cos (α - 3 * π / 2) = 1 / 5) : f α = 2 * sqrt 6 / 5 :=
sorry

end f_simplify_f_value_l420_420468


namespace total_area_of_triangular_faces_l420_420199

noncomputable def pyramid_base_edge : ℝ := 8
noncomputable def pyramid_lateral_edge : ℝ := 7

structure Pyramid where
  base_edge : ℝ
  lateral_edge : ℝ

def myPyramid : Pyramid :=
{ base_edge := pyramid_base_edge,
  lateral_edge := pyramid_lateral_edge }

theorem total_area_of_triangular_faces :
  4 * (1 / 2) * myPyramid.base_edge * (Real.sqrt (myPyramid.lateral_edge^2 - (myPyramid.base_edge / 2)^2)) = 16 * Real.sqrt 33 :=
by
  sorry

end total_area_of_triangular_faces_l420_420199


namespace missing_digit_first_digit_l420_420725

-- Definitions derived from conditions
def is_three_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999
def is_divisible_by_six (n : ℕ) : Prop := n % 6 = 0
def multiply_by_two (d : ℕ) : ℕ := 2 * d

-- Main statement to prove
theorem missing_digit_first_digit (d : ℕ) (n : ℕ) 
  (h1 : multiply_by_two d = n) 
  (h2 : is_three_digit_number n) 
  (h3 : is_divisible_by_six n)
  (h4 : d = 2)
  : n / 100 = 2 :=
sorry

end missing_digit_first_digit_l420_420725


namespace sextuple_angle_terminal_side_on_xaxis_l420_420424

-- Define angle and conditions
variable (α : ℝ)
variable (isPositiveAngle : 0 < α ∧ α < 360)
variable (sextupleAngleOnXAxis : ∃ k : ℕ, 6 * α = k * 360)

-- Prove the possible values of the angle
theorem sextuple_angle_terminal_side_on_xaxis :
  α = 60 ∨ α = 120 ∨ α = 180 ∨ α = 240 ∨ α = 300 :=
  sorry

end sextuple_angle_terminal_side_on_xaxis_l420_420424


namespace cylinder_volume_l420_420778

theorem cylinder_volume (short_side long_side : ℝ) (h_short_side : short_side = 12) (h_long_side : long_side = 18) : 
  ∀ (r h : ℝ) (h_radius : r = short_side / 2) (h_height : h = long_side), 
    volume = π * r^2 * h := 
by
  sorry

end cylinder_volume_l420_420778


namespace ME_eq_DN_l420_420937

variables {A B C D E M N : Type*} [acute_triangle A B C]
variables (AD : altitude A D B C) (CE : altitude C E A B)
variables (DE : line D E)
variables (M : perp_foot A DE) (N : perp_foot C DE)

theorem ME_eq_DN : ME = DN :=
sorry

end ME_eq_DN_l420_420937


namespace find_union_and_intersection_find_range_of_a_l420_420034

open Set Real

variable (R : Set ℝ) (A B C : Set ℝ)

noncomputable def setA : Set ℝ := { x | x^2 - 9 * x + 18 ≥ 0 }
noncomputable def setB : Set ℝ := { x | -2 < x ∧ x < 9 }
noncomputable def setC (a : ℝ) : Set ℝ := { x | a < x ∧ x < a + 1 }

-- 1. Find A ∪ B and (∁ₙA) ∩ B
theorem find_union_and_intersection : 
  A = setA → B = setB → (A ∪ B = R) ∧ ((R \ A) ∩ B = { x | 3 < x ∧ x < 6 }) :=
by
  intros hA hB
  split
  -- Part 1: Prove A ∪ B = R
  {
    sorry
  }
  -- Part 2: Prove (∁ₙA) ∩ B = { x | 3 < x ∧ x < 6 }
  {
    sorry
  }

-- 2. Find the range of values for the real number a
theorem find_range_of_a : 
  (C = setC a) → (C ⊆ B) → (-2 ≤ a ∧ a ≤ 8) :=
by
  intros hC hCB
  sorry

end find_union_and_intersection_find_range_of_a_l420_420034


namespace find_x2_plus_y2_l420_420495

theorem find_x2_plus_y2 (x y : ℝ) (h1 : x * y = 10) (h2 : x^2 * y + x * y^2 + 2 * x + 2 * y = 88) :
    x^2 + y^2 = 304 / 9 := sorry

end find_x2_plus_y2_l420_420495


namespace pyramid_area_l420_420245

theorem pyramid_area :
  ∀ (a b : ℝ), a = 8 → b = 7 → 4 * (1/2 * a * sqrt (b^2 - (a/2)^2)) = 16 * sqrt 33 :=
by
  intros a b ha hb
  rw [ha, hb]
  have h1 : a / 2 = 4 := by norm_num [ha]
  have h2 : b^2 - (a / 2)^2 = 33 :=
    by
      calc
        b^2 - (a / 2)^2 = 49 - 16 := by norm_num [hb]
        ... = 33 := by norm_num
  rw [h1, h2, sqrt 33, mul_one, mul_one, half_mul, mul_comm (1/2) a, ←mul_assoc, mul_comm 4 4]
  norm_num
  sorry

end pyramid_area_l420_420245


namespace cosine_monotone_interval_l420_420169

theorem cosine_monotone_interval :
  ∀ (k : ℤ), 
  monotone_decreasing (λ x : ℝ, cos (2 * x + (π / 4))) [k * π - (π / 8), k * π + (3 * π / 8)] :=
by
  sorry

end cosine_monotone_interval_l420_420169


namespace largest_gold_coins_l420_420313

noncomputable def max_gold_coins (n : ℕ) : ℕ :=
  if h : ∃ k : ℕ, n = 13 * k + 3 ∧ n < 150 then
    n
  else 0

theorem largest_gold_coins : max_gold_coins 146 = 146 :=
by
  sorry

end largest_gold_coins_l420_420313


namespace max_area_difference_160_perimeter_rectangles_l420_420661

theorem max_area_difference_160_perimeter_rectangles : 
  ∃ (l1 w1 l2 w2 : ℕ), (2 * l1 + 2 * w1 = 160) ∧ (2 * l2 + 2 * w2 = 160) ∧ 
  (l1 * w1 - l2 * w2 = 1521) := sorry

end max_area_difference_160_perimeter_rectangles_l420_420661


namespace volume_of_geometric_body_l420_420643

def equilateral_triangle_base_area (side_length : ℝ) : ℝ :=
  (math.sqrt 3) / 4 * side_length^2

def tetrahedron_volume (base_area : ℝ) (height : ℝ) : ℝ :=
  (1 / 3) * base_area * height

def geometric_body_volume (side_length : ℝ) (height : ℝ) : ℝ :=
  2 * tetrahedron_volume (equilateral_triangle_base_area side_length) height

theorem volume_of_geometric_body :
  geometric_body_volume 1 ((math.sqrt 3) / 2) = 1 / 4 :=
sorry

end volume_of_geometric_body_l420_420643


namespace pyramid_area_l420_420280

theorem pyramid_area (base_edge lateral_edge : ℝ) (H_base_edge : base_edge = 8) (H_lateral_edge : lateral_edge = 7) :
  4 * (1 / 2 * base_edge * real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 16 * real.sqrt 33 :=
by
  sorry

end pyramid_area_l420_420280


namespace train_length_correct_l420_420750

noncomputable def train_length (speed_kmh : ℝ) (time_s : ℝ) : ℝ :=
  let speed_ms := (speed_kmh * 1000) / 3600
  speed_ms * time_s

theorem train_length_correct 
  (speed_kmh : ℝ := 60) 
  (time_s : ℝ := 9) :
  train_length speed_kmh time_s = 150.03 := by 
  sorry

end train_length_correct_l420_420750


namespace total_number_of_fish_l420_420177

def number_of_tuna : Nat := 5
def number_of_spearfish : Nat := 2

theorem total_number_of_fish : number_of_tuna + number_of_spearfish = 7 := by
  sorry

end total_number_of_fish_l420_420177


namespace pyramid_face_area_total_l420_420240

theorem pyramid_face_area_total 
  (base_edge : ℝ) (lateral_edge : ℝ) 
  (h_base_edge : base_edge = 8) 
  (h_lateral_edge : lateral_edge = 7) : 
  4 * (1 / 2 * base_edge * real.sqrt (lateral_edge^2 - (base_edge / 2)^2)) = 16 * real.sqrt 33 :=
by
  sorry

end pyramid_face_area_total_l420_420240


namespace pyramid_area_l420_420241

theorem pyramid_area :
  ∀ (a b : ℝ), a = 8 → b = 7 → 4 * (1/2 * a * sqrt (b^2 - (a/2)^2)) = 16 * sqrt 33 :=
by
  intros a b ha hb
  rw [ha, hb]
  have h1 : a / 2 = 4 := by norm_num [ha]
  have h2 : b^2 - (a / 2)^2 = 33 :=
    by
      calc
        b^2 - (a / 2)^2 = 49 - 16 := by norm_num [hb]
        ... = 33 := by norm_num
  rw [h1, h2, sqrt 33, mul_one, mul_one, half_mul, mul_comm (1/2) a, ←mul_assoc, mul_comm 4 4]
  norm_num
  sorry

end pyramid_area_l420_420241


namespace area_of_quadrilateral_NLMK_l420_420848

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem area_of_quadrilateral_NLMK 
  (AB BC AC AK CN CL : ℝ)
  (h_AB : AB = 13)
  (h_BC : BC = 20)
  (h_AC : AC = 21)
  (h_AK : AK = 4)
  (h_CN : CN = 1)
  (h_CL : CL = 20 / 21) : 
  triangle_area AB BC AC - 
  (1 * CL / (BC * AC) * triangle_area AB BC AC) - 
  (9 * (BC - CN) / (AB * BC) * triangle_area AB BC AC) -
  (16 * 41 / (169 * 21) * triangle_area AB BC AC) = 
  493737 / 11830 := 
sorry

end area_of_quadrilateral_NLMK_l420_420848


namespace biology_marks_l420_420805

theorem biology_marks 
  (e : ℕ) (m : ℕ) (p : ℕ) (c : ℕ) (a : ℕ) (n : ℕ) (b : ℕ) 
  (h_e : e = 96) (h_m : m = 95) (h_p : p = 82) (h_c : c = 97) (h_a : a = 93) (h_n : n = 5)
  (h_total : e + m + p + c + b = a * n) :
  b = 95 :=
by 
  sorry

end biology_marks_l420_420805


namespace boxes_contain_same_number_of_apples_l420_420137

theorem boxes_contain_same_number_of_apples (total_apples boxes : ℕ) (h1 : total_apples = 49) (h2 : boxes = 7) : 
  total_apples / boxes = 7 :=
by
  sorry

end boxes_contain_same_number_of_apples_l420_420137


namespace pyramid_total_area_l420_420219

/-- The total area of the four triangular faces of a right, square-based pyramid
whose base edges measure 8 units and lateral edges measure 7 units is 16√33. -/
theorem pyramid_total_area :
  let base_edge := 8
  let lateral_edge := 7
  4 * (1 / 2 * base_edge * Real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 16 * Real.sqrt 33 := 
by
  let base_edge := 8
  let lateral_edge := 7
  have h1 : 4 * (1 / 2 * base_edge * Real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 
              4 * (1 / 2 * 8 * Real.sqrt (49 - 16)) := 
    by sorry
  have h2 : 4 * (1 / 2 * 8 * Real.sqrt (49 - 16)) = 4 * (4 * Real.sqrt 33) := 
    by sorry
  have h3 : 4 * (4 * Real.sqrt 33) = 16 * Real.sqrt 33 := 
    by sorry
  exact eq.trans (eq.trans h1 h2) h3

end pyramid_total_area_l420_420219


namespace inequality_transformation_l420_420044

theorem inequality_transformation (m n : ℝ) (h : -m / 2 < -n / 6) : 3 * m > n := by
  sorry

end inequality_transformation_l420_420044


namespace martians_cannot_join_hands_l420_420416

theorem martians_cannot_join_hands (num_martians : ℕ) (num_hands_per_martian : ℕ) (h : num_martians = 2021) (h' : num_hands_per_martian = 3) :
  ¬ ∃ (pairs : ℕ), 2 * pairs = num_martians * num_hands_per_martian :=
by
  have h1 : num_martians = 2021 := h
  have h2 : num_hands_per_martian = 3 := h'
  have h3 : num_martians * num_hands_per_martian = 6063 := by
    rw [h1, h2]
    norm_num
  have h4 : 6063 % 2 = 1 := by norm_num
  rw [←h3, h4]
  sorry

end martians_cannot_join_hands_l420_420416


namespace problem_statement_l420_420029

noncomputable def polynomial (a b c d : ℝ) : (ℝ → ℝ) :=
  λ x, x^4 + a*x^3 + b*x^2 + c*x + d

theorem problem_statement (a b c d : ℝ)
  (h1 : polynomial a b c d 1 = 2)
  (h2 : polynomial a b c d 2 = 4)
  (h3 : polynomial a b c d 3 = 6) :
  polynomial a b c d 0 + polynomial a b c d 4 = 32 :=
sorry

end problem_statement_l420_420029


namespace tiles_per_row_l420_420598

theorem tiles_per_row (area : ℝ) (tile_length : ℝ) (h1 : area = 256) (h2 : tile_length = 2/3) : 
  (16 * 12) / (8) = 24 :=
by {
  sorry
}

end tiles_per_row_l420_420598


namespace no_real_roots_l420_420640

-- Define the coefficients of the quadratic equation
def a : ℝ := 1
def b : ℝ := 2
def c : ℝ := 4

-- Define the quadratic equation
def quadratic_eqn (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the discriminant
def discriminant : ℝ := b^2 - 4 * a * c

-- State the theorem: The quadratic equation has no real roots because the discriminant is negative
theorem no_real_roots : discriminant < 0 := by
  unfold discriminant
  unfold a b c
  sorry

end no_real_roots_l420_420640


namespace probability_in_interval_l420_420368

def F (x : ℝ) : ℝ :=
  if x ≤ -1 then 0
  else if x ≤ 1/3 then (3/4) * x + 3/4
  else 1

theorem probability_in_interval (X : ℝ → ℝ) (hF : X = F) :
  (F (1/3) - F 0) = 1/4 :=
by
  dsimp [F]
  rw [if_pos  (le_refl (-1)),
      if_neg (lt_of_lt_of_le (by norm_num) (le_refl (1/3))),
      if_neg (lt_irrefl (-1)),
      if_pos  (le_refl (1/3)),
      if_neg (ne_of_gt (by norm_num : 1/3 > 0))]
  exact by norm_num

end probability_in_interval_l420_420368


namespace probability_y_greater_than_x_l420_420781

theorem probability_y_greater_than_x (x y : ℝ) (hx : x ∈ set.Icc 0 1500) (hy : y ∈ set.Icc 0 3000) :
  set.Icc 0 3000 ≠ ∅ → (probability (λ (y > x), y > x) = 3 / 4) :=
by
  sorry

end probability_y_greater_than_x_l420_420781


namespace trajectory_equation_length_of_AB_l420_420074

-- Given conditions
def point_on_left_of_y_axis (M : ℝ × ℝ) : Prop := M.1 < 0
def distance_difference_condition (M : ℝ × ℝ) : Prop :=
  real.sqrt ((M.1 + 1)^2 + M.2^2) - (-M.1) = 1

def fixed_point : ℝ × ℝ := (-1, 0)
def midpoint (P A B : ℝ × ℝ) : Prop := 
  P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def curve_equation (M : ℝ × ℝ) : Prop := M.2^2 = -4 * M.1

-- Problem 1: Find the equation of trajectory C
theorem trajectory_equation (M : ℝ × ℝ) (h₁ : point_on_left_of_y_axis M) (h₂ : distance_difference_condition M) : 
  curve_equation M := sorry

-- Problem 2: Find the length of segment AB
def on_curve (A : ℝ × ℝ) : Prop := curve_equation A
def line_through_p (P A B : ℝ × ℝ) : Prop :=
  (A.2 - P.2) / (A.1 - P.1) = (B.2 - P.2) / (B.1 - P.1)

theorem length_of_AB (A B P : ℝ × ℝ) (h₁ : on_curve A) (h₂ : on_curve B) 
  (h₃ : midpoint P A B) (h₄ : P = (-3, -2)) : 
  ∥A - B∥ = 8 := sorry

end trajectory_equation_length_of_AB_l420_420074


namespace modular_remainder_of_valid_sets_l420_420968

open Function

def S := {i : ℕ | 1 ≤ i ∧ i ≤ 12}

def count_sets (A B : Finset ℕ) (S : Finset ℕ) : ℕ :=
  (∑ k in (Finset.range 5), ((Finset.card S).choose (k+3) * (Finset.card (S \ (A ∪ B))).choose k))

theorem modular_remainder_of_valid_sets :
  let n := count_sets (Finset.range 13).to_finset (Finset.range 13).to_finset S;
  n % 1000 = 252 :=
by sorry

end modular_remainder_of_valid_sets_l420_420968


namespace prove_Q_value_l420_420499

def f (P x : ℝ) : ℝ := (25^x) / (25^x + P)

def Q (P : ℝ) : ℝ := ∑ i in (Finset.range 24).map (λ n, n+1) | (1/25 : ℝ) + (i / 25 : ℝ), f P (i / 25)

theorem prove_Q_value (P : ℝ) : Q P = 12 :=
sorry

end prove_Q_value_l420_420499


namespace greatest_area_difference_l420_420664

theorem greatest_area_difference 
  (l w l' w' : ℕ) 
  (h1 : 2 * l + 2 * w = 160) 
  (h2 : 2 * l' + 2 * w' = 160) :
  abs ((l * w) - (l' * w')) ≤ 1521 := 
sorry

end greatest_area_difference_l420_420664


namespace tan_double_angle_sum_l420_420872

theorem tan_double_angle_sum (α : ℝ) (h : Real.tan α = 3 / 2) :
  Real.tan (2 * α + Real.pi / 4) = -7 / 17 := 
sorry

end tan_double_angle_sum_l420_420872


namespace det_B_squared_minus_3B_l420_420888

open Matrix
open_locale matrix big_operators

variable (R : Type*) [CommRing R]

def B : Matrix (Fin 2) (Fin 2) R :=
  ![![2, 4], ![3, 2]]

theorem det_B_squared_minus_3B :
  det (B R ^ 2 - 3 • B R) = (88 : R) := by
  sorry

end det_B_squared_minus_3B_l420_420888


namespace angle_PQR_is_60_deg_l420_420542

open Real EuclideanSpace

-- Definitions for points P, Q, R in a 3-dimensional Euclidean space
def P : EuclideanSpace ℝ := ![-3, 1, 7]
def Q : EuclideanSpace ℝ := ![-4, 0, 3]
def R : EuclideanSpace ℝ := ![-5, 0, 4]

-- Compute angle ∠PQR
def angle_PQR : ℝ := 
  let PQ := dist P Q
  let PR := dist P R
  let QR := dist Q R
  real.arccos ((PQ ^ 2 + QR ^ 2 - PR ^ 2) / (2 * PQ * QR)) * 180 / pi

theorem angle_PQR_is_60_deg : angle_PQR = 60 := by
  sorry

end angle_PQR_is_60_deg_l420_420542


namespace coefficient_p_neg1_in_expansion_l420_420078

theorem coefficient_p_neg1_in_expansion :
  coeff_in_binom_expansion (λ (p : ℝ), (p - 1 / p.sqrt) ^ 8) (-1) = 28 := by
sorry

end coefficient_p_neg1_in_expansion_l420_420078


namespace truncated_tetrahedron_edges_l420_420798

theorem truncated_tetrahedron_edges :
  ∀ (T : SimpleGraph) (V : Finset (T.V)), 
  (T.is_tetrahedron ∧ V.card = 4) → 
  let new_edges := 4 * 3 in
  let original_edges := 6 in
  T.edges.card + new_edges = 18 :=
by
  intro T V h
  let new_edges := 4 * 3
  let original_edges := 6
  have original_edges_count : T.edges.card = original_edges,
    from sorry -- This follows from the tetrahedron structure
  have edges_with_truncation : T.edges.card + new_edges = 18,
    from sorry -- This is proven by the calculation in the conditions
  exact edges_with_truncation

end truncated_tetrahedron_edges_l420_420798


namespace upgrade_cost_l420_420115

theorem upgrade_cost (sandwich_price : ℝ) (salad_price : ℝ) (total_bill : ℝ) (coupon_fraction : ℝ) :
  sandwich_price = 8 ∧ salad_price = 3 ∧ total_bill = 12 ∧ coupon_fraction = 1 / 4 →
  ∃ upgrade_cost : ℝ, upgrade_cost = 3 :=
by
  intros h
  cases h with h1 h_rest
  cases h_rest with h2 h_rest'
  cases h_rest' with h3 h4
  use 3
  sorry

end upgrade_cost_l420_420115


namespace pyramid_four_triangular_faces_area_l420_420262

theorem pyramid_four_triangular_faces_area 
  (base_edge : ℝ) (lateral_edge : ℝ) 
  (h_base : base_edge = 8)
  (h_lateral : lateral_edge = 7) :
  let h := Real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)
  let triangle_area := (1 / 2) * base_edge * h
  let total_area := 4 * triangle_area
  total_area = 16 * Real.sqrt 33 :=
by
  -- Definitions to introduce local values
  let half_base := base_edge / 2
  let h := Real.sqrt (lateral_edge ^ 2 - half_base ^ 2)
  let triangle_area := (1 / 2) * base_edge * h
  let total_area := 4 * triangle_area
  -- Assertion to compare calculated total area with given correct answer
  have h_eq : h = Real.sqrt 33 := by sorry
  have triangle_area_eq : triangle_area = 4 * Real.sqrt 33 := by sorry
  have total_area_eq : total_area = 16 * Real.sqrt 33 := by sorry
  exact total_area_eq

end pyramid_four_triangular_faces_area_l420_420262


namespace largest_divisor_of_visible_product_l420_420389

theorem largest_divisor_of_visible_product :
  let Q := ∏ i in {1, 2, 3, 4, 5, 6, 7, 8}.erase k, i
  in ∃ k ∈ {1, 2, 3, 4, 5, 6, 7, 8}, ∀ n, n ∣ Q → n ≤ 192 :=
by
  sorry

end largest_divisor_of_visible_product_l420_420389


namespace length_of_BC_l420_420060

-- Definitions corresponding to the conditions in the problem.
def Circle (O : Point) (r : ℝ) : Set Point := {P | dist O P = r}

-- Setup
variables {O A B C D : Point}           -- Points on the circle
variable {r : ℝ}                        -- Radius of circle
variable [metric_space Point]
variable [euclidean_space ℝ Point]      -- Assuming the points lie in a Euclidean space

-- Conditions
variables (h_circle : ∀ (P ∈ Circle O r), dist O P = r)  -- O is the center of the circle
variables (h_diameter : dist A D = 2 * r)       -- AD is a diameter
variables (h_chord : A ∈ Circle O r)            -- A, B, and C on the circle
variables (h_chord_2 : B ∈ Circle O r)
variables (h_chord_3 : C ∈ Circle O r)
variables (h_BO : dist O B = 7)                 -- BO = 7
variables (h_angle_ABO : angle A B O = 90)      -- ∠ABO = 90°
variables (h_arc_CD : Arc CD = 90)             -- arc CD = 90°

-- Conclusion
theorem length_of_BC : dist B C = 7 :=
by sorry

end length_of_BC_l420_420060


namespace pyramid_four_triangular_faces_area_l420_420259

theorem pyramid_four_triangular_faces_area 
  (base_edge : ℝ) (lateral_edge : ℝ) 
  (h_base : base_edge = 8)
  (h_lateral : lateral_edge = 7) :
  let h := Real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)
  let triangle_area := (1 / 2) * base_edge * h
  let total_area := 4 * triangle_area
  total_area = 16 * Real.sqrt 33 :=
by
  -- Definitions to introduce local values
  let half_base := base_edge / 2
  let h := Real.sqrt (lateral_edge ^ 2 - half_base ^ 2)
  let triangle_area := (1 / 2) * base_edge * h
  let total_area := 4 * triangle_area
  -- Assertion to compare calculated total area with given correct answer
  have h_eq : h = Real.sqrt 33 := by sorry
  have triangle_area_eq : triangle_area = 4 * Real.sqrt 33 := by sorry
  have total_area_eq : total_area = 16 * Real.sqrt 33 := by sorry
  exact total_area_eq

end pyramid_four_triangular_faces_area_l420_420259


namespace length_of_BC_l420_420061

-- Definitions corresponding to the conditions in the problem.
def Circle (O : Point) (r : ℝ) : Set Point := {P | dist O P = r}

-- Setup
variables {O A B C D : Point}           -- Points on the circle
variable {r : ℝ}                        -- Radius of circle
variable [metric_space Point]
variable [euclidean_space ℝ Point]      -- Assuming the points lie in a Euclidean space

-- Conditions
variables (h_circle : ∀ (P ∈ Circle O r), dist O P = r)  -- O is the center of the circle
variables (h_diameter : dist A D = 2 * r)       -- AD is a diameter
variables (h_chord : A ∈ Circle O r)            -- A, B, and C on the circle
variables (h_chord_2 : B ∈ Circle O r)
variables (h_chord_3 : C ∈ Circle O r)
variables (h_BO : dist O B = 7)                 -- BO = 7
variables (h_angle_ABO : angle A B O = 90)      -- ∠ABO = 90°
variables (h_arc_CD : Arc CD = 90)             -- arc CD = 90°

-- Conclusion
theorem length_of_BC : dist B C = 7 :=
by sorry

end length_of_BC_l420_420061


namespace right_triangle_hypotenuse_length_l420_420168

theorem right_triangle_hypotenuse_length:
  ∀ (x : ℝ), (2 * x + 3 > 0) →
  (x * (2 * x + 3) = 168) →
  (√( (x^2) + ((2 * x + 3)^2)) = 5 * √109 / 2) :=
by {
  intros x h1 h2,
  sorry
}

end right_triangle_hypotenuse_length_l420_420168


namespace probability_neither_red_nor_purple_l420_420330

theorem probability_neither_red_nor_purple 
    (total_balls : ℕ)
    (white_balls : ℕ) 
    (green_balls : ℕ) 
    (yellow_balls : ℕ) 
    (red_balls : ℕ) 
    (purple_balls : ℕ) 
    (h_total : total_balls = white_balls + green_balls + yellow_balls + red_balls + purple_balls)
    (h_counts : white_balls = 50 ∧ green_balls = 30 ∧ yellow_balls = 8 ∧ red_balls = 9 ∧ purple_balls = 3):
    (88 : ℚ) / 100 = 0.88 :=
by
  sorry

end probability_neither_red_nor_purple_l420_420330


namespace count_households_in_apartment_l420_420567

noncomputable def total_households 
  (houses_left : ℕ)
  (houses_right : ℕ)
  (floors_above : ℕ)
  (floors_below : ℕ) 
  (households_per_house : ℕ) : ℕ :=
(houses_left + houses_right) * (floors_above + floors_below) * households_per_house

theorem count_households_in_apartment : 
  ∀ (houses_left houses_right floors_above floors_below households_per_house : ℕ),
  houses_left = 1 →
  houses_right = 6 →
  floors_above = 1 →
  floors_below = 3 →
  households_per_house = 3 →
  total_households houses_left houses_right floors_above floors_below households_per_house = 105 :=
by
  intros houses_left houses_right floors_above floors_below households_per_house hl hr fa fb hh
  rw [hl, hr, fa, fb, hh]
  unfold total_households
  norm_num
  sorry

end count_households_in_apartment_l420_420567


namespace arithmetic_sequence_sum_l420_420694

theorem arithmetic_sequence_sum :
  let a := -3
  let d := 7
  let n := 10
  let s := n * (2 * a + (n - 1) * d) / 2
  s = 285 :=
by
  -- Details of the proof are omitted as per instructions
  sorry

end arithmetic_sequence_sum_l420_420694


namespace sin_identity_l420_420448

theorem sin_identity (x y z : ℝ)
    (hx : sin x ≠ 0) 
    (hy : sin y ≠ 0) 
    (hz : sin z ≠ 0)
    (hxy : sin (x - y) ≠ 0)
    (hyz : sin (y - z) ≠ 0)
    (hzx : sin (z - x) ≠ 0) :
    let m := sin x / sin (y - z),
        n := sin y / sin (z - x),
        p := sin z / sin (x - y) in 
    m * n + n * p + p * m = -1 := by
sorry

end sin_identity_l420_420448


namespace sqrt_problem_l420_420444

theorem sqrt_problem (x : ℝ) (h1 : sqrt 99225 = 315) (h2 : sqrt x = 3.15) : x = 9.9225 :=
sorry

end sqrt_problem_l420_420444


namespace max_selling_price_max_total_profit_l420_420945

-- Definitions for problem (1)
def cost_price : ℝ := 6
def selling_price : ℝ := 8
def monthly_sales_volume : ℝ := 50000
def price_increase_effect (x : ℝ) : ℝ := monthly_sales_volume - ((x - selling_price) / 0.5) * 2000

-- Monthly profit inequality for problem (1)
def monthly_total_profit (x : ℝ) : ℝ := price_increase_effect(x) * (x - cost_price)
def original_monthly_profit : ℝ := monthly_sales_volume * (selling_price - cost_price)

theorem max_selling_price (x : ℝ) :
  monthly_total_profit x ≥ original_monthly_profit ↔ x ≤ 18.5 :=
sorry

-- Definitions for problem (2)
def new_price (x : ℝ) (hx : x ≥ 9) : ℝ := (26 / 5) * (x - 9)
def volume_change (x : ℝ) : ℝ := (5 - (x - 8) / 0.5 * (0.2 / (x - 8)^2)) * 10000
def total_profit (x : ℝ) : ℝ := (volume_change x) * (x - 6) - new_price x (by linarith)

theorem max_total_profit (x : ℝ) (hx : x ≥ 9) :
  max (total_profit x) 14 :=
sorry

end max_selling_price_max_total_profit_l420_420945


namespace smallest_n_for_imaginary_part_l420_420563

noncomputable def z : Complex := Complex.ofReal (Real.cos (1 / 1000)) + Complex.I * Complex.ofReal (Real.sin (1 / 1000))

theorem smallest_n_for_imaginary_part :
  ∃ (n : ℕ), (Complex.im (z^(n : ℕ)) > 1 / 2) ∧ ∀ (m : ℕ), m < n → Complex.im (z^(m : ℕ)) ≤ 1 / 2 :=
begin
  sorry  -- Proof omitted as per instructions
end

end smallest_n_for_imaginary_part_l420_420563


namespace arithmetic_sequence_sum_l420_420695

theorem arithmetic_sequence_sum :
  let a := -3
  let d := 7
  let n := 10
  let s := n * (2 * a + (n - 1) * d) / 2
  s = 285 :=
by
  -- Details of the proof are omitted as per instructions
  sorry

end arithmetic_sequence_sum_l420_420695


namespace magnitude_calc_l420_420986

noncomputable def magnitude_of_product (z : ℂ) (z_conj : ℂ) : ℝ :=
  |(1 - z) * z_conj|

theorem magnitude_calc : 
  let z := -1 - complex.i
  let z_conj := complex.conj z
  magnitude_of_product z z_conj = real.sqrt 10 :=
by
  sorry

end magnitude_calc_l420_420986


namespace total_area_of_pyramid_faces_l420_420272

-- Define the basic parameters of the pyramid
def base_edges := 8
def lateral_edges := 7

-- Define the Pythagorean theorem components
def altitude_squared := lateral_edges^2 - (base_edges / 2)^2
def altitude := real.sqrt altitude_squared

-- Define the area of one triangular face using half-base and altitude
def one_face_area := (1 / 2) * base_edges * altitude

-- Define the total area of the four triangular faces
def total_area_of_faces := 4 * one_face_area

-- Statement to prove
theorem total_area_of_pyramid_faces : total_area_of_faces = 16 * real.sqrt 33 :=
by
  -- Necessary calculations are assured correct by previous definitions and attributes
  sorry

end total_area_of_pyramid_faces_l420_420272


namespace pyramid_area_l420_420246

theorem pyramid_area :
  ∀ (a b : ℝ), a = 8 → b = 7 → 4 * (1/2 * a * sqrt (b^2 - (a/2)^2)) = 16 * sqrt 33 :=
by
  intros a b ha hb
  rw [ha, hb]
  have h1 : a / 2 = 4 := by norm_num [ha]
  have h2 : b^2 - (a / 2)^2 = 33 :=
    by
      calc
        b^2 - (a / 2)^2 = 49 - 16 := by norm_num [hb]
        ... = 33 := by norm_num
  rw [h1, h2, sqrt 33, mul_one, mul_one, half_mul, mul_comm (1/2) a, ←mul_assoc, mul_comm 4 4]
  norm_num
  sorry

end pyramid_area_l420_420246


namespace residue_7_pow_1234_l420_420684

theorem residue_7_pow_1234 : (7^1234) % 13 = 4 := by
  sorry

end residue_7_pow_1234_l420_420684


namespace pyramid_volume_l420_420131

noncomputable def volume_of_pyramid (EF FG QE : ℝ) : ℝ :=
  (1 / 3) * (EF * FG) * QE

theorem pyramid_volume
  (EF FG QE : ℝ)
  (h1 : EF = 10)
  (h2 : FG = 5)
  (h3 : QE = 9) :
  volume_of_pyramid EF FG QE = 150 :=
by
  simp [volume_of_pyramid, h1, h2, h3]
  sorry

end pyramid_volume_l420_420131


namespace breadth_halved_of_percentage_change_area_l420_420629

theorem breadth_halved_of_percentage_change_area {L B B' : ℝ} (h : 0 < L ∧ 0 < B) 
  (h1 : L / 2 * B' = 0.5 * (L * B)) : B' = 0.5 * B :=
sorry

end breadth_halved_of_percentage_change_area_l420_420629


namespace problem_statement_l420_420010

-- Define the function and its period condition
def f (x : ℝ) (ω : ℝ) := √3 * sin (ω * x) * cos (ω * x) + sin (ω * x) ^ 2 - 1 / 2

-- Assume that the period of the function f, given a certain omega, is pi
theorem problem_statement (ω : ℝ) (h : f = λ x, sin (2 * x - π / 6)) (period_f : ∀ x, f (x + π / ω) = f x) :
  -- Prove that the expression for the function remains the same
  ∃ (ω : ℝ), (ω = 1) ∧ (f = λ x, sin (2 * x - π / 6)) ∧
  -- Prove the max and min values of the function in the interval [0, π/2]
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ π / 2 → -1 / 2 ≤ f x ∧ f x ≤ 1) :=
begin
  use ω,
  split,
  { sorry },
  split,
  { sorry },
  { intro x,
    intro hx,
    sorry }
end

end problem_statement_l420_420010


namespace shortest_side_of_TriangleXYZ_is_1_5_l420_420922

-- Definitions and context
noncomputable def TriangleXYZ (X Y Z P Q : Type) [MetricSpace X] [MetricSpace Y] [MetricSpace Z] [MetricSpace P] [MetricSpace Q] := 
  ∃ (XY a : ℝ) (YZ b : ℝ) (XZ c : ℝ), 
  XY = a ∧ YZ = b ∧ XZ = c ∧ 
  let XP := 3 in let PQ := 4 in let QZ := 8 in 
  (XP + PQ + QZ = a + b + c) ∧ 
  c = 1.5

-- Prove that the shortest side is 1.5
theorem shortest_side_of_TriangleXYZ_is_1_5 {X Y Z P Q : Type}  [MetricSpace X] [MetricSpace Y] [MetricSpace Z] [MetricSpace P] [MetricSpace Q]: 
  TriangleXYZ X Y Z P Q → 
  ∃ c : ℝ, c = 1.5 ∧ c ≤ ∀ (a b : ℝ), a ≠ b ∧ ¬(a = c) ∧ ¬(b = c) :=
by sorry

end shortest_side_of_TriangleXYZ_is_1_5_l420_420922


namespace largest_number_of_gold_coins_l420_420325

theorem largest_number_of_gold_coins 
  (num_friends : ℕ)
  (extra_coins : ℕ)
  (total_coins : ℕ) :
  num_friends = 13 →
  extra_coins = 3 →
  total_coins < 150 →
  ∀ k : ℕ, total_coins = num_friends * k + extra_coins →
  total_coins ≤ 146 :=
by
  sorry

end largest_number_of_gold_coins_l420_420325


namespace pyramid_area_l420_420244

theorem pyramid_area :
  ∀ (a b : ℝ), a = 8 → b = 7 → 4 * (1/2 * a * sqrt (b^2 - (a/2)^2)) = 16 * sqrt 33 :=
by
  intros a b ha hb
  rw [ha, hb]
  have h1 : a / 2 = 4 := by norm_num [ha]
  have h2 : b^2 - (a / 2)^2 = 33 :=
    by
      calc
        b^2 - (a / 2)^2 = 49 - 16 := by norm_num [hb]
        ... = 33 := by norm_num
  rw [h1, h2, sqrt 33, mul_one, mul_one, half_mul, mul_comm (1/2) a, ←mul_assoc, mul_comm 4 4]
  norm_num
  sorry

end pyramid_area_l420_420244


namespace max_area_difference_l420_420675

theorem max_area_difference (l w l' w' : ℕ) 
  (h1 : 2 * l + 2 * w = 160) 
  (h2 : 2 * l' + 2 * w' = 160) : 
  abs ((l * w) - (l' * w')) ≤ 1600 := 
by 
  sorry

end max_area_difference_l420_420675


namespace envelope_addressing_equation_l420_420734

theorem envelope_addressing_equation (x : ℝ) :
  (800 / 10 + 800 / x + 800 / 5) * (3 / 800) = 1 / 3 :=
  sorry

end envelope_addressing_equation_l420_420734


namespace sum_of_alternating_sums_8_l420_420434

def alternating_sum (s : Finset ℕ) : ℤ :=
  s.sort (≥).alternating_sum ℤ

def all_alternating_sums (n : ℕ) : ℤ :=
  (Finset.powerset (Finset.range n)).sum alternating_sum

theorem sum_of_alternating_sums_8 : all_alternating_sums 8 = 1024 := by
  sorry

end sum_of_alternating_sums_8_l420_420434


namespace pyramid_total_area_l420_420293

noncomputable def pyramid_base_edge : ℝ := 8
noncomputable def pyramid_lateral_edge : ℝ := 7

theorem pyramid_total_area 
  (base_edge : ℝ)
  (lateral_edge : ℝ)
  (h_base_edge : base_edge = pyramid_base_edge)
  (h_lateral_edge : lateral_edge = pyramid_lateral_edge) 
: 4 * (1 / 2 * base_edge * real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 16 * real.sqrt 33 := 
sorry

end pyramid_total_area_l420_420293


namespace area_of_transformed_region_l420_420547

noncomputable def matrix_transformation_area (A : Matrix (Fin 2) (Fin 2) ℝ) (area_T : ℝ) : ℝ :=
  matrix.det A * area_T

theorem area_of_transformed_region : 
  let T_area := 15
  let A := Matrix.of_list 2 2 [[3, 4], [6, -2]]
  let T'_area := matrix_transformation_area A T_area
  T'_area = 450 :=
by
  let T_area := 15
  let A := Matrix.of_list 2 2 [[3, 4], [6, -2]]
  let T'_area := matrix_transformation_area A T_area
  show T'_area = 450
  sorry

end area_of_transformed_region_l420_420547


namespace pyramid_area_l420_420249

theorem pyramid_area :
  ∀ (a b : ℝ), a = 8 → b = 7 → 4 * (1/2 * a * sqrt (b^2 - (a/2)^2)) = 16 * sqrt 33 :=
by
  intros a b ha hb
  rw [ha, hb]
  have h1 : a / 2 = 4 := by norm_num [ha]
  have h2 : b^2 - (a / 2)^2 = 33 :=
    by
      calc
        b^2 - (a / 2)^2 = 49 - 16 := by norm_num [hb]
        ... = 33 := by norm_num
  rw [h1, h2, sqrt 33, mul_one, mul_one, half_mul, mul_comm (1/2) a, ←mul_assoc, mul_comm 4 4]
  norm_num
  sorry

end pyramid_area_l420_420249


namespace largest_smallest_sector_area_l420_420656

theorem largest_smallest_sector_area :
  ∃ (a b : ℕ), gcd a b = 1 ∧ (10 ^ 2 * a) + b = 106 ∧
  (∀ (radius : ℝ), radius = 1 →
  ∀ (sectors : ℕ), sectors = 5 →
  ∀ (divided : ℕ), divided = 3 →
  ∀ (πn : ℝ), πn = π →
  let sector_area := (πn / 6) in
  sector_area = (π / 6) ∧ (10^2 * 1 + 6 = 106)) := sorry

end largest_smallest_sector_area_l420_420656


namespace problem_solution_l420_420092

def is_multiple_of_12 (n : ℕ) : Prop :=
  n % 12 = 0

def has_distinct_digits (n : ℕ) : Prop :=
  ∀ i j, i ≠ j → n.digits.get! i ≠ n.digits.get! j

def problem_statement : Prop :=
  let M := (Finset.range 10).perm.toFinset.filter (λ n, is_multiple_of_12 n ∧ has_distinct_digits n).max
  (M % 2000 = 960)

theorem problem_solution : problem_statement :=
by sorry

end problem_solution_l420_420092


namespace kilometers_per_gallon_correct_l420_420724

-- Define the total distance traveled
def total_distance : ℝ := 150

-- Define the total gallons of gasoline used
def total_gasoline : ℝ := 3.75

-- Define kilometers per gallon as the ratio of distance to gasoline
def kilometers_per_gallon : ℝ := total_distance / total_gasoline

-- Theorem stating that the kilometers per gallon is 40
theorem kilometers_per_gallon_correct : kilometers_per_gallon = 40 := 
by
  -- Proof is not required
  sorry

end kilometers_per_gallon_correct_l420_420724


namespace fixed_point_l420_420621

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x - 1) + 3

theorem fixed_point (a : ℝ) : f a 1 = 4 := by
  sorry

end fixed_point_l420_420621


namespace inequality_log_inequality_exp_l420_420329

open Real

-- Definitions for the first problem
def log_base_halve (x : ℝ) : ℝ := log x / log (1/2)

def log_ratio_base (x : ℝ) (a b : ℝ) : ℝ := log x / log (a / b)

-- Proof problem 1
theorem inequality_log (x : ℝ) (h1 : 3 < x) (h2 : x < 9) :
  9.271 * log_base_halve (x - 3) - log_base_halve (x + 3) - log_ratio_base 2 (x + 3) (x - 3) > 0 :=
sorry

-- Definitions for the second problem
def abs_expr (x : ℝ) : ℝ := 2^(4*x^2 - 1) - 5

-- Proof problem 2
theorem inequality_exp (x : ℝ)
  (h1 : x ∈ set.Icc (-1) (-1 / sqrt 2) ∨ x ∈ set.Icc (1 / sqrt 2) 1) :
  9.272 * abs (abs_expr x) ≤ 3 :=
sorry

end inequality_log_inequality_exp_l420_420329


namespace max_min_values_l420_420633

theorem max_min_values : 
  ∃ t_max t_min : ℝ, (∀ t ∈ Icc (-1 : ℝ) 1, t^2 + 4*t - 5 ≤ 0) ∧ 
  (∀ t ∈ Icc (-1 : ℝ) 1, -8 ≤ t^2 + 4*t - 5) ∧ 
  (t_max ∈ Icc (-1 : ℝ) 1 ∧ (t_max^2 + 4*t_max - 5 = 0)) ∧ 
  (t_min ∈ Icc (-1 : ℝ) 1 ∧ (t_min^2 + 4*t_min - 5 = -8)) :=
by {
  sorry
}

end max_min_values_l420_420633


namespace g_symmetric_about_pi_div_2_l420_420167

def f (x : ℝ) : ℝ := Real.sin (2 * x)

def g (x : ℝ) : ℝ := Real.sin (2 * (x + Real.pi / 4))

theorem g_symmetric_about_pi_div_2 : ∀ x, g (Real.pi / 2 - x) = g (Real.pi / 2 + x) :=
  by
  sorry

end g_symmetric_about_pi_div_2_l420_420167


namespace sin_cos_sum_identity_l420_420777

noncomputable def trigonometric_identity (x y z w : ℝ) := 
  (Real.sin x * Real.cos y + Real.sin z * Real.cos w) = Real.sqrt 2 / 2

theorem sin_cos_sum_identity :
  trigonometric_identity 347 148 77 58 :=
by sorry

end sin_cos_sum_identity_l420_420777


namespace age_multiple_l420_420136

variables {R J K : ℕ}

theorem age_multiple (h1 : R = J + 6) (h2 : R = K + 3) (h3 : (R + 4) * (K + 4) = 108) :
  ∃ M : ℕ, R + 4 = M * (J + 4) ∧ M = 2 :=
sorry

end age_multiple_l420_420136


namespace find_b_l420_420914

variables {a b c y1 y2 : ℝ}

-- Define the conditions
def point_on_parabola_1 (a b c : ℝ) : Prop := y1 = a * 2^2 + b * 2 + c
def point_on_parabola_2 (a b c : ℝ) : Prop := y2 = a * (-2)^2 + b * (-2) + c
def y1_y2_diff : Prop := y1 - y2 = -16

-- State the theorem
theorem find_b
  (h1 : point_on_parabola_1 a b c)
  (h2 : point_on_parabola_2 a b c)
  (h3 : y1_y2_diff) :
  b = -4 :=
by sorry

end find_b_l420_420914


namespace floor_of_smallest_zero_l420_420975
noncomputable def g (x : ℝ) := 3 * Real.sin x - Real.cos x + 2 * Real.tan x
def smallest_zero (s : ℝ) : Prop := s > 0 ∧ g s = 0 ∧ ∀ x, 0 < x ∧ x < s → g x ≠ 0

theorem floor_of_smallest_zero (s : ℝ) (h : smallest_zero s) : ⌊s⌋ = 4 :=
sorry

end floor_of_smallest_zero_l420_420975


namespace liars_knights_l420_420175

theorem liars_knights (n : ℕ) (knight liar : Fin n → Prop) 
  (P : ∀ i, knight i ↔ liars_knights i)
  (total : n = 10)
  (liar_must_be_balanced : ∀ k, liar k ↔ k ≥ 5):
  ∃ k, k = 5 := sorry

end liars_knights_l420_420175


namespace transformed_stats_l420_420451

variable {n : ℕ}
variable (x : Fin n → ℝ) (x_bar s : ℝ)

def mean (x : Fin n → ℝ) : ℝ :=
  (∑ i, x i) / n

def stddev (x : Fin n → ℝ) (x_bar : ℝ) : ℝ :=
  sqrt ((∑ i, (x i - x_bar)^2) / n)

def transformed_mean (x_bar : ℝ) : ℝ :=
  3 * x_bar - 1

def transformed_variance (s : ℝ) : ℝ :=
  9 * s^2

theorem transformed_stats (h_mean : mean x = x_bar) (h_stddev : stddev x x_bar = s) :
  mean (fun i => 3 * x i - 1) = transformed_mean x_bar ∧
  stddev (fun i => 3 * x i - 1) (transformed_mean x_bar) = transformed_variance s := 
by
  sorry

end transformed_stats_l420_420451


namespace intersection_of_A_and_B_l420_420090

open Set

def A := {x : ℝ | 2 + x ≥ 4}
def B := {x : ℝ | -1 ≤ x ∧ x ≤ 5}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 2 ≤ x ∧ x ≤ 5} := sorry

end intersection_of_A_and_B_l420_420090


namespace final_temperature_pressure_ratio_l420_420746

variables (v1 v2 : ℝ) (t1_C t2_C : ℝ)

-- Given conditions
def post_conversion_t1_C := t1_C + 273  -- Converting t1 from Celsius to Kelvin
def post_conversion_t2_C := t2_C + 273  -- Converting t2 from Celsius to Kelvin

-- Prove that the final temperature in Kelvin (t_C) is correctly calculated
theorem final_temperature (h1 : v1 = 0.2) (h2 : v2 = 0.8) (h3 : t1_C = 127) (h4 : t2_C = 7) :
  let t_K := (v1 * post_conversion_t1_C + v2 * post_conversion_t2_C) / (v1 + v2)
  in t_K - 273 = 31 := 
begin
  -- Sorry to skip the proof
  sorry
end

-- Prove that the ratio of the final to initial pressures in the first part of the vessel is 0.76
theorem pressure_ratio (h1 : v1 = 0.2) (h2 : t1_C = 127) :
  let t1_K := t1_C + 273
  let t_K := 304  -- Directly use the proven final temperature in Kelvin
  in t_K / t1_K = 0.76 := 
begin
  -- Sorry to skip the proof
  sorry
end

end final_temperature_pressure_ratio_l420_420746


namespace positive_option_l420_420408

theorem positive_option :
  (cos 2) < 0 → (tan 3) < 0 → (tan 3 * cos 2) > 0 :=
by
  intro hcos
  intro htan
  sorry

end positive_option_l420_420408


namespace triangle_is_isosceles_l420_420921

theorem triangle_is_isosceles (A B C : ℝ) (a b c : ℝ) 
  (h1 : c = 2 * a * Real.cos B) 
  (h2 : a = b) :
  ∃ (isIsosceles : Bool), isIsosceles := 
sorry

end triangle_is_isosceles_l420_420921


namespace combustion_result_l420_420425

noncomputable def combustion_reaction : Prop :=
  ∀ (C2H6 O2 CO2 H2O N2 : ℕ),
    C2H6 = 5 →
    O2 = 7 →
    N2 = 2 →
    (O2 * (2 / 7) ≥ C2H6) →
    (CO2 = 4 ∧ H2O = 6)

theorem combustion_result : combustion_reaction :=
begin
  intros C2H6 O2 CO2 H2O N2 hC2H6 hO2 hN2 hLim,
  have total_C2H6 := min C2H6 (O2 * (2 / 7)) := by sorry,
  have CO2_amount := total_C2H6 * 2 := by sorry,
  have H2O_amount := total_C2H6 * 3 := by sorry,
  split;
  assumption,
end

end combustion_result_l420_420425


namespace lim_f_n_x_to_infty_lim_f_n_n_to_infty_l420_420830

open Real

def f_n (n : ℕ) (x : ℝ) : ℝ :=
  (∑ k in {-n : fin (2*n+1).succ}, sqrt (x + ↑k)) - (2 * ↑n + 1) * sqrt x

-- Proof statement for part (a)
theorem lim_f_n_x_to_infty (n : ℕ) : tendsto (f_n n) at_top (𝓝 0) :=
sorry

-- Proof statement for part (b)
theorem lim_f_n_n_to_infty : tendsto (λ n : ℕ, f_n n n) at_top (𝓝 0) :=
sorry

end lim_f_n_x_to_infty_lim_f_n_n_to_infty_l420_420830


namespace area_of_region_l420_420190

open Classical

noncomputable def point := (ℚ × ℚ)

def line1 (x : ℚ) : ℚ := 2 * x + 3
def line2 (x : ℚ) : ℚ := -3 * x + 24

def intersection_1 : point := (0, line1 0)
def intersection_2 : point := (0, line2 0)
def intersection_3 : point :=
  let x := 21 / 5 in
  (x, line1 x)

def area_of_triangle (A B C : point) : ℚ :=
  abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)

theorem area_of_region :
  area_of_triangle intersection_1 intersection_2 intersection_3 = 44.1 :=
by
  sorry

end area_of_region_l420_420190


namespace total_points_l420_420528

/-- Joan, Jessica, and Jeremy found seashells of different points. This theorem states the total 
    points earned for all seashells is 48. -/
theorem total_points :
  let joan_points := 6 * 2 in
  let jessica_points := 8 * 3 in
  let jeremy_points := 12 * 1 in
  joan_points + jessica_points + jeremy_points = 48 :=
by
  -- We acknowledge the steps to calculate the points individually.
  let joan_points := 6 * 2
  let jessica_points := 8 * 3
  let jeremy_points := 12 * 1
  -- We add them up to verify the total points.
  show joan_points + jessica_points + jeremy_points = 48
  sorry

end total_points_l420_420528


namespace relationship_among_abc_l420_420908

noncomputable def a : ℝ := Real.pi^(-2)
noncomputable def b : ℝ := a^a
noncomputable def c : ℝ := a^(a^a)

theorem relationship_among_abc : c < b ∧ b > a := 
by
  sorry

end relationship_among_abc_l420_420908


namespace points_concyclic_l420_420772

theorem points_concyclic 
  (A B C D E P K L : Type)
  (h1 : ordered_on_line A B C D E)
  (h2 : BC = CD = sqrt AB * DE)
  (h3 : perpendicular_bisector P B D) 
  (h4 : PB = PD)
  (h5 : on_segments PB K)
  (h6 : on_segments PD L) 
  (h7 : angle_bisector K C B K E)
  (h8 : angle_bisector C A L D)
  : concyclic A K L E :=
sorry

end points_concyclic_l420_420772


namespace circle_through_points_and_tangent_l420_420816

theorem circle_through_points_and_tangent (m n : ℝ) :
  (∃ (m n : ℝ), ((m ≠ 0) ∧ n > 0 ∧ ((m = 1 ∧ n = 1) ∨ (m = -3 ∧ n = 5)) ∧
  ∀ x y : ℝ, (x - m)^2 + (y - n)^2 = n^2 ↔ ((x = 0 ∧ y = 1) ∨ (x = 1 ∧ y = 2) ∨ (y = 0))) :=
begin
  sorry
end

end circle_through_points_and_tangent_l420_420816


namespace octahedron_volume_ratio_l420_420797

theorem octahedron_volume_ratio (a : ℝ) (h₁ : a > 0) :
  let V_cube := a^3,
      V_octahedron := (sqrt 2 / 3) * (a * sqrt 2)^3 in
  V_octahedron / V_cube = 1 / 6 :=
by
  sorry

end octahedron_volume_ratio_l420_420797


namespace q_value_l420_420184

-- Define the problem conditions
def prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_multiple_of (a b : ℕ) : Prop := ∃ k, a = k * b

-- Statement of the problem
theorem q_value (p q : ℕ) (hp : prime p) (hq : prime q) (h1 : q = 13 * p + 2) (h2 : is_multiple_of (q - 1) 3) : q = 67 :=
sorry

end q_value_l420_420184


namespace area_of_square_l420_420846

/-- Given a right triangle with vertices B(0, 0), C(4, 0), and E(4, 3) where the hypotenuse BE is 5,
prove that the area of the square ABCD, with B and C as adjacent vertices, is 16. -/
theorem area_of_square (B C E : ℤ × ℤ) (BC_CE : B = (0, 0) ∧ C = (4, 0) ∧ E = (4, 3))
  (hypotenuse : dist B E = 5) : (let side_length := dist B C in side_length^2) = 16 :=
sorry

end area_of_square_l420_420846


namespace hectares_per_day_initial_l420_420358

variable (x : ℝ) -- x is the number of hectares one tractor ploughs initially per day

-- Condition 1: A field can be ploughed by 6 tractors in 4 days.
def total_area_initial := 6 * x * 4

-- Condition 2: 6 tractors plough together a certain number of hectares per day, denoted as x hectares/day.
-- This is incorporated in the variable declaration of x.

-- Condition 3: If 2 tractors are moved to another field, the remaining 4 tractors can plough the same field in 5 days.
-- Condition 4: One of the 4 tractors ploughs 144 hectares a day when 4 tractors plough the field in 5 days.
def total_area_with_4_tractors := 4 * 144 * 5

-- The statement that equates the two total area expressions.
theorem hectares_per_day_initial : total_area_initial x = total_area_with_4_tractors := by
  sorry

end hectares_per_day_initial_l420_420358


namespace probability_difference_one_is_one_third_l420_420768

open Finset

-- Let an urn contain 6 balls numbered from 1 to 6
def urn := {1, 2, 3, 4, 5, 6}

-- Define the event that two balls are drawn
def two_balls : Finset (ℕ × ℕ) := {(x, y) | x ∈ urn ∧ y ∈ urn ∧ x < y}

-- Define the event that the difference between two drawn balls is 1
def difference_one (x y : ℕ) : Prop := abs (x - y) = 1

-- Calculate the total number of ways to draw two balls from the urn
def total_draws : ℕ := two_balls.card

-- Calculate the number of favorable outcomes
def favorable_draws : ℕ := (two_balls.filter (λ (p : ℕ × ℕ), difference_one p.1 p.2)).card

-- Calculate the probability of the difference being 1
noncomputable def probability_difference_one : ℚ := favorable_draws / total_draws

-- State the theorem
theorem probability_difference_one_is_one_third :
  probability_difference_one = 1 / 3 :=
sorry

end probability_difference_one_is_one_third_l420_420768


namespace travel_methods_l420_420649

theorem travel_methods (bus_services : ℕ) (train_services : ℕ) (ship_services : ℕ) :
  bus_services = 8 → train_services = 3 → ship_services = 2 → 
  bus_services + train_services + ship_services = 13 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num

end travel_methods_l420_420649


namespace persons_in_first_group_l420_420588

theorem persons_in_first_group (P : ℕ) :
  (P * 5 * 12 = 30 * 6 * 11) → P = 33 :=
by
  intro h
  have h1 : 30 * 6 * 11 = 1980 := rfl
  rw h1 at h
  have h2 : P * 60 = 1980 := h
  have h3 : P = 1980 / 60 := sorry
  exact h3

end persons_in_first_group_l420_420588


namespace pyramid_total_area_l420_420229

/-- The total area of the four triangular faces of a right, square-based pyramid
whose base edges measure 8 units and lateral edges measure 7 units is 16√33. -/
theorem pyramid_total_area :
  let base_edge := 8
  let lateral_edge := 7
  4 * (1 / 2 * base_edge * Real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 16 * Real.sqrt 33 := 
by
  let base_edge := 8
  let lateral_edge := 7
  have h1 : 4 * (1 / 2 * base_edge * Real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 
              4 * (1 / 2 * 8 * Real.sqrt (49 - 16)) := 
    by sorry
  have h2 : 4 * (1 / 2 * 8 * Real.sqrt (49 - 16)) = 4 * (4 * Real.sqrt 33) := 
    by sorry
  have h3 : 4 * (4 * Real.sqrt 33) = 16 * Real.sqrt 33 := 
    by sorry
  exact eq.trans (eq.trans h1 h2) h3

end pyramid_total_area_l420_420229


namespace lauri_ate_days_l420_420585

theorem lauri_ate_days
    (simone_rate : ℚ)
    (simone_days : ℕ)
    (lauri_rate : ℚ)
    (total_apples : ℚ)
    (simone_apples : ℚ)
    (lauri_apples : ℚ)
    (lauri_days : ℚ) :
  simone_rate = 1/2 → 
  simone_days = 16 →
  lauri_rate = 1/3 →
  total_apples = 13 →
  simone_apples = simone_rate * simone_days →
  lauri_apples = total_apples - simone_apples →
  lauri_days = lauri_apples / lauri_rate →
  lauri_days = 15 :=
by
  intros
  sorry

end lauri_ate_days_l420_420585


namespace length_of_tangent_segment_l420_420541

noncomputable def circle (center : ℝ × ℝ) (radius : ℝ) := 
  λ (p : ℝ × ℝ), (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2

theorem length_of_tangent_segment :
  let C1 := circle (12, 0) 5 in
  let C2 := circle (-18, 0) 8 in
  ∃ (R S : ℝ × ℝ),
    (C1 R ∧ C2 S ∧ 
     ∀ (P Q : ℝ × ℝ), (C1 P ∧ C2 Q → dist R S ≤ dist P Q)) ∧ 
        dist R S = 339 / 13 := 
by
  sorry

end length_of_tangent_segment_l420_420541


namespace sum_of_corners_9x9_checkerboard_l420_420388

-- Define the size of the checkerboard
def size := 9

-- Define the numbers in the four corners
def top_left := 1
def top_right := size
def bottom_right := size * size
def bottom_left := (size * (size - 1)) + 1

-- Define the sum of the corners
def sum_of_corners := top_left + top_right + bottom_right + bottom_left

-- The statement to prove
theorem sum_of_corners_9x9_checkerboard : sum_of_corners = 164 :=
by {
  -- Calculate each corner value explicitly
  have h_tl : top_left = 1 := rfl,
  have h_tr : top_right = size := rfl,
  have h_bl : bottom_left = (size * (size - 1)) + 1 := rfl,
  have h_br : bottom_right = size * size := rfl,

  -- Simplify by substitution
  rw [h_tl, h_tr, h_bl, h_br],

  -- By definitions we have:
  have h_sum : 1 + size + ((size * (size - 1)) + 1) + (size * size) = 164,
  -- This can be broken down further if needed:
  -- 1 + 9 + (9 * 8 + 1) + (9 * 9) = 164
  sorry -- completing the arithmetics manually or automatically
}

end sum_of_corners_9x9_checkerboard_l420_420388


namespace isosceles_triangle_circles_ratio_l420_420155

theorem isosceles_triangle_circles_ratio (α : ℝ) (hα : 0 < α ∧ α < π / 2) : 
  let r := (x : ℝ) / 2 * tan (α / 2),
      R := x / (4 * cos α * sin α)
  in r / R = tan (α / 2) * sin (2 * α) :=
by
  sorry

end isosceles_triangle_circles_ratio_l420_420155


namespace inequality_holds_l420_420549

theorem inequality_holds (a b : ℝ) (h_a : a > 0) (h_b : b > 0) :
  (∀ x : ℝ, |x + 1| < b) → (a > 2 → ∀ x : ℝ, |4x^2 + 6x - 2| < a) :=
by
  sorry

end inequality_holds_l420_420549


namespace ptolemys_theorem_l420_420645

-- Definition of the variables describing the lengths of the sides and diagonals
variables {a b c d m n : ℝ}

-- We declare that they belong to a cyclic quadrilateral
def cyclic_quadrilateral (a b c d m n : ℝ) : Prop :=
∃ (A B C D : ℝ), 
  A + C = 180 ∧ 
  B + D = 180 ∧ 
  m = (A * C) ∧ 
  n = (B * D) ∧ 
  a = (A * B) ∧ 
  b = (B * C) ∧ 
  c = (C * D) ∧ 
  d = (D * A)

-- The theorem statement in Lean form
theorem ptolemys_theorem (h : cyclic_quadrilateral a b c d m n) : m * n = a * c + b * d :=
sorry

end ptolemys_theorem_l420_420645


namespace series_less_than_one_l420_420794

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def sum_fractions (n : Nat) : Real :=
  match n with
  | 0 => 0
  | n + 1 => sum_fractions n + n / factorial (n + 1)

theorem series_less_than_one :
  sum_fractions 2012 < 1 :=
  sorry

end series_less_than_one_l420_420794


namespace series_sum_lt_one_l420_420791

/--
  The series sum from 1 to 2012 of (n / (n+1)!) is less than 1.
-/
theorem series_sum_lt_one :
  (∑ n in Finset.range (2012+1), (n / (n+1)! : ℝ)) < 1 := 
  sorry

end series_sum_lt_one_l420_420791


namespace pyramid_area_l420_420281

theorem pyramid_area (base_edge lateral_edge : ℝ) (H_base_edge : base_edge = 8) (H_lateral_edge : lateral_edge = 7) :
  4 * (1 / 2 * base_edge * real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 16 * real.sqrt 33 :=
by
  sorry

end pyramid_area_l420_420281


namespace perpendicular_line_tangent_slope_l420_420466

theorem perpendicular_line_tangent_slope (a b : ℝ) :
  aeval (1 : ℝ) (X^3 - 1) = 0 -- point P(1,1) lies on y = x^3
  → let mₜ := deriv (λ x : ℝ, x^3) 1 in -- deriv of y = x^3 at x = 1
  mₜ = 3 -- slope of tangent line at P(1,1)
  → let mₗ := aeval (1 : ℝ) (a - b * X - 2) = 0 in -- P(1,1) on ax - by - 2 = 0
  (mₗ : ℝ) / b = a / b -- slope of line ax - by - 2 = a/b
  → let condition := mₜ * (a / b) = -1 in -- slopes are negative reciprocals
  condition
  → b / a = -3 := sorry

end perpendicular_line_tangent_slope_l420_420466


namespace find_length_of_BC_l420_420064

def circle_geometry_problem (O A B C D : Point) (BO : ℝ)
  (angle_ABO : ℝ) (arc_CD : ℝ) : Prop :=
  diameter O A D ∧ chord O A B C ∧ BO = 7 ∧ angle_ABO = 90 ∧ arc_CD = 90

theorem find_length_of_BC (O A B C D : Point) (BO : ℝ)
  (angle_ABO : ℝ) (arc_CD : ℝ) (h : circle_geometry_problem O A B C D BO angle_ABO arc_CD) :
  length B C = 7 := by
  sorry

end find_length_of_BC_l420_420064


namespace multiply_increase_by_196_l420_420364

theorem multiply_increase_by_196 (x : ℕ) (h : 14 * x = 14 + 196) : x = 15 :=
sorry

end multiply_increase_by_196_l420_420364


namespace pyramid_volume_l420_420132

noncomputable def volume_of_pyramid (EF FG QE : ℝ) : ℝ :=
  (1 / 3) * (EF * FG) * QE

theorem pyramid_volume
  (EF FG QE : ℝ)
  (h1 : EF = 10)
  (h2 : FG = 5)
  (h3 : QE = 9) :
  volume_of_pyramid EF FG QE = 150 :=
by
  simp [volume_of_pyramid, h1, h2, h3]
  sorry

end pyramid_volume_l420_420132


namespace regular_octagon_interior_angle_l420_420418

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (∑ i in (finset.range n), (360 : ℝ) / ↑n) = 360 → (∀ i, i < n → 180 - (360 / n) = 135) :=
by
  intros n hn hsum i hi
  have h1 : (360 : ℝ) / 8 = 45 := by norm_num
  have h2 : 180 - 45 = 135 := by norm_num
  rw [hn] at *
  exact h2
  sorry

end regular_octagon_interior_angle_l420_420418


namespace intersection_domain_l420_420473

open Set

noncomputable def f (x : ℝ) : ℝ := log (x + 3)

theorem intersection_domain (g : ℝ → ℝ) (M N : Set ℝ) :
  (M = {x | -3 < x}) →
  (N = univ) →
  (M ∩ N = {x | -3 < x}) :=
by
  intros hM hN
  rw [hM, hN, inter_univ]
  admit -- proof is omitted

end intersection_domain_l420_420473


namespace avg_displacement_per_man_l420_420719

-- Problem definition as per the given conditions
def num_men : ℕ := 50
def tank_length : ℝ := 40  -- 40 meters
def tank_width : ℝ := 20   -- 20 meters
def rise_in_water_level : ℝ := 0.25  -- 25 cm -> 0.25 meters

-- Given the conditions, we need to prove the average displacement per man
theorem avg_displacement_per_man :
  (tank_length * tank_width * rise_in_water_level) / num_men = 4 := by
  sorry

end avg_displacement_per_man_l420_420719


namespace base8_product_correct_l420_420821

theorem base8_product_correct :
  let n1 := 3 * 8^2 + 2 * 8^1 + 7 * 8^0
  let n2 := 7
  let product := n1 * n2
  let base8_rep := (2 * 8^3) + (7 * 8^2) + (4 * 8^1) + (1 * 8^0)
  product = 1505 ∧ base8_rep = 2741 := 
by {
  -- Definitions from the conditions
  let n1 := 3 * 8^2 + 2 * 8^1 + 7 * 8^0
  let n2 := 7
  let product := n1 * n2
  -- Base 10 product
  have h1 : product = 1505, from rfl,
  -- Base 8 representation of 1505
  let base8_rep := (2 * 8^3) + (7 * 8^2) + (4 * 8^1) + (1 * 8^0)
  have h2 : base8_rep = 2741, from rfl,
  exact ⟨h1, h2⟩
}

end base8_product_correct_l420_420821


namespace problem1_problem2_l420_420081

theorem problem1 (A : ℝ) : sin (A + π / 6) = 2 * cos A → A = π / 3 := 
sorry

theorem problem2 (b c : ℝ) :
  a = sqrt 3 ∧ (height : ℝ → ℝ) (λ x, x = 2 / 3) →
  a = sqrt 3 → height (sqrt 3 / 2) = 2 / 3 →
  let A := π / 3 in
    b^2 + c^2 - 2 * b * c * cos A = (b + c)^2 - 4 → 
    (b + c)^2 = 7 →
    b + c = sqrt 7 :=
sorry

end problem1_problem2_l420_420081


namespace purely_imaginary_condition_l420_420976

-- Define the necessary conditions
def real_part_eq_zero (m : ℝ) : Prop := 2 * m^2 - 3 * m - 2 = 0
def imaginary_part_neq_zero (m : ℝ) : Prop := m^2 - 3 * m + 2 ≠ 0

-- State the theorem to be proved
theorem purely_imaginary_condition (m : ℝ) :
  real_part_eq_zero m ∧ imaginary_part_neq_zero m ↔ m = -1/2 :=
sorry

end purely_imaginary_condition_l420_420976


namespace problem_base_conversion_l420_420614

theorem problem_base_conversion :
  ∀ (b : ℕ), (base_eq b)  :=
by 
  -- Definitions derived from the given problem
  def base_eq (b : ℕ) : Prop :=
    (b + 2) * (4 * b + 3) = b^3 ∧ b = 6 →
    12 * b + 43 * b = 10 * b^3
  
  sorry

end problem_base_conversion_l420_420614


namespace probability_of_A_winning_l420_420013

theorem probability_of_A_winning:
  let P_Draw := (1 : ℚ) / 2,
  let P_B_wins := (1 : ℚ) / 3,
  let P_all := 1,
  P_A = P_all - P_Draw - P_B_wins → 
  P_A = (1 : ℚ) / 6 :=
by
  sorry

end probability_of_A_winning_l420_420013


namespace intersection_convex_l420_420341

variables {K1 K2 K12 : Set ℝ}

-- Definitions of convex sets
def is_convex (S : Set ℝ) : Prop :=
  ∀ ⦃x y : ℝ⦄, x ∈ S → y ∈ S → ∀ ⦃θ : ℝ⦄, 0 ≤ θ → θ ≤ 1 → (θ * x + (1 - θ) * y) ∈ S

-- Condition: K1, K2 are convex, K12 is their intersection
axiom K1_convex : is_convex K1
axiom K2_convex : is_convex K2
def K12 := K1 ∩ K2

-- Prove that K12 is convex
theorem intersection_convex : is_convex K12 :=
sorry

end intersection_convex_l420_420341


namespace h_at_3_l420_420957

noncomputable def f (x : ℝ) := 3 * x + 4
noncomputable def g (x : ℝ) := Real.sqrt (f x) - 3
noncomputable def h (x : ℝ) := g (f x)

theorem h_at_3 : h 3 = Real.sqrt 43 - 3 := by
  sorry

end h_at_3_l420_420957


namespace sum_of_coordinates_of_point_D_l420_420574

theorem sum_of_coordinates_of_point_D
  (N : ℝ × ℝ := (6,2))
  (C : ℝ × ℝ := (10, -2))
  (h : ∃ D : ℝ × ℝ, (N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2))) :
  ∃ (D : ℝ × ℝ), D.1 + D.2 = 8 := 
by
  obtain ⟨D, hD⟩ := h
  sorry

end sum_of_coordinates_of_point_D_l420_420574


namespace pyramid_triangular_face_area_l420_420210

theorem pyramid_triangular_face_area 
  (base : ℝ) (lateral : ℝ)
  (h_base : base = 8) (h_lateral : lateral = 7) :
  let height := sqrt (lateral^2 - (base / 2)^2) in
  4 * (1 / 2 * base * height) = 16 * sqrt 33 := 
by
  sorry

end pyramid_triangular_face_area_l420_420210


namespace pyramid_total_area_l420_420222

/-- The total area of the four triangular faces of a right, square-based pyramid
whose base edges measure 8 units and lateral edges measure 7 units is 16√33. -/
theorem pyramid_total_area :
  let base_edge := 8
  let lateral_edge := 7
  4 * (1 / 2 * base_edge * Real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 16 * Real.sqrt 33 := 
by
  let base_edge := 8
  let lateral_edge := 7
  have h1 : 4 * (1 / 2 * base_edge * Real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 
              4 * (1 / 2 * 8 * Real.sqrt (49 - 16)) := 
    by sorry
  have h2 : 4 * (1 / 2 * 8 * Real.sqrt (49 - 16)) = 4 * (4 * Real.sqrt 33) := 
    by sorry
  have h3 : 4 * (4 * Real.sqrt 33) = 16 * Real.sqrt 33 := 
    by sorry
  exact eq.trans (eq.trans h1 h2) h3

end pyramid_total_area_l420_420222


namespace Martha_points_l420_420994

def beef_cost := 3 * 11
def fv_cost := 8 * 4
def spice_cost := 3 * 6
def other_cost := 37

def total_spent := beef_cost + fv_cost + spice_cost + other_cost
def points_per_10 := 50
def bonus := 250

def increments := total_spent / 10
def points := increments * points_per_10
def total_points := points + bonus

theorem Martha_points : total_points = 850 :=
by
  sorry

end Martha_points_l420_420994


namespace combined_investment_yield_l420_420153

def stock_info := { yield: ℝ, market_value: ℝ }

def StockA : stock_info := { yield := 0.14, market_value := 500 }
def StockB : stock_info := { yield := 0.08, market_value := 750 }
def StockC : stock_info := { yield := 0.12, market_value := 1000 }

def tax_rate : ℝ := 0.02 
def commission_fee : ℝ := 50 

def calculate_net_yield (info : stock_info) : ℝ :=
  let initial_yield := info.yield * info.market_value
  let tax := tax_rate * initial_yield
  initial_yield - tax

def total_net_yield : ℝ :=
  calculate_net_yield StockA + calculate_net_yield StockB + calculate_net_yield StockC - 3 * commission_fee

def total_market_value : ℝ :=
  StockA.market_value + StockB.market_value + StockC.market_value

def overall_yield_percentage : ℝ :=
  (total_net_yield / total_market_value) * 100

theorem combined_investment_yield : overall_yield_percentage ≈ 4.22 := 
  by 
    have h : overall_yield_percentage = 95 / 2250 * 100 := 
      by 
        -- elaborate the steps, skipping with sorry
        sorry
    exact_mod_cast h

end combined_investment_yield_l420_420153


namespace correct_average_l420_420711

theorem correct_average (incorrect_avg : ℝ) (n : ℕ) (wrong_num correct_num : ℝ)
  (h_avg : incorrect_avg = 23)
  (h_n : n = 10)
  (h_wrong : wrong_num = 26)
  (h_correct : correct_num = 36) :
  (incorrect_avg * n + (correct_num - wrong_num)) / n = 24 :=
by
  -- Proof goes here
  sorry

end correct_average_l420_420711


namespace minimum_value_l420_420001

/-- Given a triangle ABC and a point M inside it, we are provided with:
1. The dot product of vectors AB and AC is 2\sqrt{3}
2. The angle BAC is 30 degrees (or π/6 radians)
3. The areas of triangles MBC, MCA, and MAB are 1/2, x, and y respectively.
We need to prove that the minimum value of (1/x) + (4/y) is 18.
--/

variables {A B C M : Type*} [inner_product_space ℝ A]

def dot_product_condition (AB AC : A) : Prop := 
  (inner_product AB AC = 2 * real.sqrt 3)

def angle_condition (BAC : ℝ) : Prop :=
  (BAC = real.pi / 6)

noncomputable def area_condition (area_MBC x y : ℝ) : Prop := 
  (area_MBC = 1 / 2) ∧ (area_tria_MCA = x) ∧ (area_tria_MAB = y)

theorem minimum_value (AB AC : A) (BAC : ℝ) (area_MBC x y : ℝ)
  (dot_cond : dot_product_condition AB AC)
  (angle_cond : angle_condition BAC)
  (area_cond : area_condition area_MBC x y) :
  ∃ x y, (1 / x) + (4 / y) = 18 :=
begin
  sorry,
end

end minimum_value_l420_420001


namespace modulus_of_z_l420_420557

def z : ℂ := (1 - complex.I) / (1 + complex.I) + 2 * complex.I

theorem modulus_of_z : complex.abs z = 1 := 
by 
  sorry

end modulus_of_z_l420_420557


namespace dice_probability_l420_420700

open ProbabilityTheory

-- Definitions based on conditions
def dice_faces := {n : ℕ | 1 ≤ n ∧ n ≤ 6}
def event (a b c : ℕ) : Prop := (a - 1) * (b - 1) * (c - 1) ≠ 0

-- The proof task itself
theorem dice_probability :
  ∀ (a b c : ℕ),
  (a ∈ dice_faces) ∧ (b ∈ dice_faces) ∧ (c ∈ dice_faces) →
  (probability (event a b c) = 125 / 216) :=
by
  sorry

end dice_probability_l420_420700


namespace funcB_is_quadratic_funcA_not_quadratic_funcC_not_quadratic_funcD_not_quadratic_only_funcB_is_quadratic_l420_420703

def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x : ℝ, f x = a * x^2 + b * x + c

def funcA (x : ℝ) : ℝ := -3 * x + 5
def funcB (x : ℝ) : ℝ := 2 * x^2
def funcC (x : ℝ) : ℝ := (x + 1)^2 - x^2
def funcD (x : ℝ) : ℝ := 3 / x^2

theorem funcB_is_quadratic : is_quadratic funcB :=
sorry

theorem funcA_not_quadratic : ¬ is_quadratic funcA :=
sorry

theorem funcC_not_quadratic : ¬ is_quadratic (λ x, funcC x) :=
sorry

theorem funcD_not_quadratic : ¬ is_quadratic funcD :=
sorry

theorem only_funcB_is_quadratic :
  funcA = funcA → funcB = funcB → funcC = funcC → funcD = funcD → is_quadratic funcB ∧
  (¬ is_quadratic funcA ∧ ¬ is_quadratic funcC ∧ ¬ is_quadratic funcD) :=
by
  exact ⟨funcB_is_quadratic, ⟨funcA_not_quadratic, funcC_not_quadratic, funcD_not_quadratic⟩⟩

end funcB_is_quadratic_funcA_not_quadratic_funcC_not_quadratic_funcD_not_quadratic_only_funcB_is_quadratic_l420_420703


namespace find_a_l420_420561

open Set Real

-- Defining sets A and B, and the condition A ∩ B = {3}
def A : Set ℝ := {-1, 1, 3}
def B (a : ℝ) : Set ℝ := {a + 2, a^2 + 4}

-- Mathematically equivalent proof statement
theorem find_a (a : ℝ) (h : A ∩ B a = {3}) : a = 1 :=
  sorry

end find_a_l420_420561


namespace monotonic_intervals_range_of_x_for_g_neg_range_of_a_for_inequality_l420_420025

noncomputable def f (a x : ℝ) : ℝ := x^3 + 3 * a * x - 1

noncomputable def f' (a x : ℝ) : ℝ := 3 * x^2 + 3 * a

noncomputable def g (a x : ℝ) : ℝ := f' a x - a * x - 3

noncomputable def g' (a x : ℝ) : ℝ := 6 * x - a

theorem monotonic_intervals (a x : ℝ) (h : a = -2) :
  (∃ a b : ℝ, f' a x > 0 ∧ (x < b ∨ x > a)) ∧ (∃ c d : ℝ, f' a x < 0 ∧ (x ∈ set.Ioo c d)) := sorry

theorem range_of_x_for_g_neg (a x : ℝ) (h : -1 ≤ a ∧ a ≤ 1) :
  g a x < 0 → 0 < x ∧ x < 1/3 := sorry

theorem range_of_a_for_inequality (a x : ℝ) (h : 2 ≤ x) :
  x * g' a x + log x > 0 → a < 12 + log 2 / 2 := sorry

end monotonic_intervals_range_of_x_for_g_neg_range_of_a_for_inequality_l420_420025


namespace number_of_tiles_in_each_row_l420_420602

-- Define the given conditions
def area_of_room : ℝ := 256
def tile_size_in_inches : ℝ := 8
def inches_per_foot : ℝ := 12

-- Length of the room in feet derived from the given area
def side_length_in_feet := Real.sqrt area_of_room

-- Convert side length from feet to inches
def side_length_in_inches := side_length_in_feet * inches_per_foot

-- The question: Prove that the number of tiles in each row is 24
theorem number_of_tiles_in_each_row :
  side_length_in_inches / tile_size_in_inches = 24 :=
sorry

end number_of_tiles_in_each_row_l420_420602


namespace transformed_data_average_variance_l420_420452

-- Definitions and conditions from the problem
variables {n : ℕ} (x : Fin n → ℝ)
def mean (x : Fin n → ℝ) := ∑ i, x i / n
def std_dev (x : Fin n → ℝ) := real.sqrt (∑ i, (x i - mean x)^2 / n)

-- Transformed data
def transformed_data (x : Fin n → ℝ) (i : Fin n) := 3 * x i - 1

-- Lean 4 Statement of the problem
theorem transformed_data_average_variance (x : Fin n → ℝ) :
  mean (transformed_data x) = 3 * mean x - 1 ∧
  std_dev (transformed_data x) ^ 2 = 9 * (std_dev x) ^ 2 :=
sorry

end transformed_data_average_variance_l420_420452


namespace systematic_sampling_definition_l420_420309

variables (N n : ℕ) (P : ℕ → Prop)

noncomputable def systematic_sampling (population_size sample_size : ℕ) := 
  ∃ (parts : ℕ) (rule : ℕ → Prop) (draw : ℕ → ℕ),
    population_size > parts * sample_size ∧
    ∀ i, 0 ≤ i ∧ i < parts → rule i ∧ draw i < sample_size

theorem systematic_sampling_definition 
  (N n : ℕ) (hnN : N > n) :
  systematic_sampling N n :=
begin
  sorry
end

end systematic_sampling_definition_l420_420309


namespace remainder_when_divided_by_6_l420_420913

theorem remainder_when_divided_by_6 (n : ℕ) (h1 : Nat.Prime (n + 3)) (h2 : Nat.Prime (n + 7)) : n % 6 = 4 :=
  sorry

end remainder_when_divided_by_6_l420_420913


namespace num_real_solutions_eq_five_l420_420401

theorem num_real_solutions_eq_five :
  ∃ (x y z w : ℝ), 
    (x = y + z + y * x * w) ∧ 
    (y = z + w + z * y * w) ∧ 
    (z = w + x + w * z * x) ∧ 
    (w = x + y + x * y * z) ∧
    -- There are exactly 5 distinct tuples (x, y, z, w)
    (5 = fintype.card {p : ℝ × ℝ × ℝ × ℝ | 
         let (x, y, z, w) := p in 
         x = y + z + y * x * w ∧ 
         y = z + w + z * y * w ∧ 
         z = w + x + w * z * x ∧ 
         w = x + y + x * y * z}) :=
sorry

end num_real_solutions_eq_five_l420_420401


namespace average_weight_of_class_l420_420354

theorem average_weight_of_class (n_boys n_girls : ℕ) (avg_weight_boys avg_weight_girls : ℝ)
    (h_boys : n_boys = 5) (h_girls : n_girls = 3)
    (h_avg_weight_boys : avg_weight_boys = 60) (h_avg_weight_girls : avg_weight_girls = 50) :
    (n_boys * avg_weight_boys + n_girls * avg_weight_girls) / (n_boys + n_girls) = 56.25 := 
by
  sorry

end average_weight_of_class_l420_420354


namespace number_of_men_in_first_group_l420_420144

-- Defining constants for the number of hours per day and the work done as identical for both groups
constant hours_per_day : ℕ := 8
constant work_done (men : ℕ) (days : ℕ) : ℕ := men * hours_per_day * days

theorem number_of_men_in_first_group :
  ∀ (M : ℕ), work_done M 18 = work_done 12 12 → M = 8 :=
 by
  intros M h
  have h1 : work_done M 18 = M * hours_per_day * 18 := rfl
  have h2 : work_done 12 12 = 12 * hours_per_day * 12 := rfl
  rw [h1, h2] at h
  sorry

end number_of_men_in_first_group_l420_420144


namespace pyramid_face_area_total_l420_420230

theorem pyramid_face_area_total 
  (base_edge : ℝ) (lateral_edge : ℝ) 
  (h_base_edge : base_edge = 8) 
  (h_lateral_edge : lateral_edge = 7) : 
  4 * (1 / 2 * base_edge * real.sqrt (lateral_edge^2 - (base_edge / 2)^2)) = 16 * real.sqrt 33 :=
by
  sorry

end pyramid_face_area_total_l420_420230


namespace find_minimum_value_l420_420429

open Real

noncomputable def g (x : ℝ) : ℝ := 
  x + x / (x^2 + 2) + x * (x + 3) / (x^2 + 3) + 3 * (x + 1) / (x * (x^2 + 3))

theorem find_minimum_value (x : ℝ) (hx : x > 0) : g x ≥ 4 := 
sorry

end find_minimum_value_l420_420429


namespace cost_price_of_computer_table_l420_420173

variable (C : ℝ) (SP : ℝ)
variable (h1 : SP = 5400)
variable (h2 : SP = C * 1.32)

theorem cost_price_of_computer_table : C = 5400 / 1.32 :=
by
  -- We are required to prove C = 5400 / 1.32
  sorry

end cost_price_of_computer_table_l420_420173


namespace combined_area_of_quadrilaterals_l420_420758

variable (Δ : Type) [HasArea Δ]

-- Define the conditions of the problem
def divided_triangle (triangles : List ℕ) : Prop :=
  triangles = [2, 5, 5, 10]

def quadrilateral_area (triangles : List ℕ) (combined_area : ℕ) : Prop :=
  divided_triangle triangles → combined_area = 22

-- The theorem to prove the combined_area of the two quadrilaterals
theorem combined_area_of_quadrilaterals : ∃ combined_area, quadrilateral_area [2, 5, 5, 10] combined_area :=
by
  use 22
  sorry

end combined_area_of_quadrilaterals_l420_420758


namespace min_value_f_when_a_eq_one_range_of_a_for_inequality_l420_420989

noncomputable def f (x a : ℝ) : ℝ := |x + 1| + |x - 4| - a

-- Question 1: When a = 1, find the minimum value of the function f(x)
theorem min_value_f_when_a_eq_one : ∃ x : ℝ, ∀ y : ℝ, f y 1 ≥ f x 1 ∧ f x 1 = 4 :=
by
  sorry

-- Question 2: For which values of a does f(x) ≥ 4/a + 1 hold for all real numbers x
theorem range_of_a_for_inequality : (∀ x : ℝ, f x a ≥ 4 / a + 1) ↔ (a < 0 ∨ a = 2) :=
by
  sorry

end min_value_f_when_a_eq_one_range_of_a_for_inequality_l420_420989


namespace daniel_candy_removal_l420_420803

theorem daniel_candy_removal (n k : ℕ) (h1 : n = 24) (h2 : k = 4) : ∃ m : ℕ, n % k = 0 → m = 0 :=
by
  sorry

end daniel_candy_removal_l420_420803


namespace doubling_marks_doubles_average_l420_420160

theorem doubling_marks_doubles_average (n : ℕ) (a : ℕ) (h : n = 12 ∧ a = 36) : 
  (2 * (n * a) / n) = 72 :=
by
  have hn := h.1
  have ha := h.2
  rw [hn, ha]
  sorry

end doubling_marks_doubles_average_l420_420160


namespace length_of_major_axis_is_eight_l420_420851

-- Definition of focal points
def f1 : ℝ × ℝ := (-2, 0)
def f2 : ℝ × ℝ := (2, 0)

-- Ellipse condition: the sum of distances from any point on the ellipse to the foci is constant
def isEllipse (P : ℝ × ℝ) (a : ℝ) : Prop :=
  let d1 := (P.1 - f1.1)^2 + (P.2 - f1.2)^2
  let d2 := (P.1 - f2.1)^2 + (P.2 - f2.2)^2
  (real.sqrt d1) + (real.sqrt d2) = 2 * a

-- Intersection condition: the line intersects the ellipse at exactly one point
def intersectAtOnePoint (a : ℝ) : Prop :=
  ∃ (x y : ℝ), isEllipse (x, y) a ∧ x + y + 4 = 0

-- The goal is to prove that the length of the major axis is 8
theorem length_of_major_axis_is_eight : ∀ {a : ℝ}, intersectAtOnePoint a → 2 * a = 8 := by
  sorry

end length_of_major_axis_is_eight_l420_420851


namespace sin_A_condition_A_Gt_PiDiv6_Necessary_but_not_Sufficient_l420_420948

-- Defining the context for angle A in a triangle
variables {A : ℝ} (h₀ : 0 < A) (h₁ : A < π)

-- The problem states that we need to prove the following:
theorem sin_A_condition (hA : A > π / 6) : ¬ (sin A > 1 / 2) := 
by sorry

-- Prove that $A > \frac{π}{6}$ is necessary but not sufficient for $sin A > \frac{1}{2}$.
theorem A_Gt_PiDiv6_Necessary_but_not_Sufficient 
(h : sin A > 1 / 2 ↔ A > π / 6) : false :=
by
  have : ¬ (A > π / 6 ↔ sin A > 1 / 2), from
    λ H, ⟨λ h, sin_A_condition h, λ h, by linarith⟩ h
  contradiction

end sin_A_condition_A_Gt_PiDiv6_Necessary_but_not_Sufficient_l420_420948


namespace pyramid_area_l420_420306

noncomputable def base_edge : ℝ := 8
noncomputable def lateral_edge : ℝ := 7
noncomputable def height := real.sqrt (lateral_edge^2 - (base_edge / 2)^2)
noncomputable def one_triangle_area := (1 / 2) * base_edge * height
noncomputable def total_area := 4 * one_triangle_area

theorem pyramid_area : total_area = 16 * real.sqrt 33 := by
  sorry

end pyramid_area_l420_420306


namespace pyramid_triangular_face_area_l420_420211

theorem pyramid_triangular_face_area 
  (base : ℝ) (lateral : ℝ)
  (h_base : base = 8) (h_lateral : lateral = 7) :
  let height := sqrt (lateral^2 - (base / 2)^2) in
  4 * (1 / 2 * base * height) = 16 * sqrt 33 := 
by
  sorry

end pyramid_triangular_face_area_l420_420211


namespace geometric_sequence_sum_l420_420943

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) (n : ℕ)
  (h1 : ∀ i : ℕ, a (i + 1) = q * a i)
  (h2 : a 4 = 3 * a 3) :
  (∑ i in finset.range n, (a (2 * (i + 1)) / a (i + 1))) = (3 ^ (n + 1) - 3) / 2 :=
by {
  sorry
}

end geometric_sequence_sum_l420_420943


namespace average_15_19_x_eq_20_l420_420159

theorem average_15_19_x_eq_20 (x : ℝ) : (15 + 19 + x) / 3 = 20 → x = 26 :=
by
  sorry

end average_15_19_x_eq_20_l420_420159


namespace pyramid_four_triangular_faces_area_l420_420257

theorem pyramid_four_triangular_faces_area 
  (base_edge : ℝ) (lateral_edge : ℝ) 
  (h_base : base_edge = 8)
  (h_lateral : lateral_edge = 7) :
  let h := Real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)
  let triangle_area := (1 / 2) * base_edge * h
  let total_area := 4 * triangle_area
  total_area = 16 * Real.sqrt 33 :=
by
  -- Definitions to introduce local values
  let half_base := base_edge / 2
  let h := Real.sqrt (lateral_edge ^ 2 - half_base ^ 2)
  let triangle_area := (1 / 2) * base_edge * h
  let total_area := 4 * triangle_area
  -- Assertion to compare calculated total area with given correct answer
  have h_eq : h = Real.sqrt 33 := by sorry
  have triangle_area_eq : triangle_area = 4 * Real.sqrt 33 := by sorry
  have total_area_eq : total_area = 16 * Real.sqrt 33 := by sorry
  exact total_area_eq

end pyramid_four_triangular_faces_area_l420_420257


namespace same_root_implies_a_vals_l420_420502

-- Define the first function f(x) = x - a
def f (x a : ℝ) : ℝ := x - a

-- Define the second function g(x) = x^2 + ax - 2
def g (x a : ℝ) : ℝ := x^2 + a * x - 2

-- Theorem statement
theorem same_root_implies_a_vals (a : ℝ) (x : ℝ) (hf : f x a = 0) (hg : g x a = 0) : a = 1 ∨ a = -1 := 
sorry

end same_root_implies_a_vals_l420_420502


namespace total_area_of_triangular_faces_l420_420207

noncomputable def pyramid_base_edge : ℝ := 8
noncomputable def pyramid_lateral_edge : ℝ := 7

structure Pyramid where
  base_edge : ℝ
  lateral_edge : ℝ

def myPyramid : Pyramid :=
{ base_edge := pyramid_base_edge,
  lateral_edge := pyramid_lateral_edge }

theorem total_area_of_triangular_faces :
  4 * (1 / 2) * myPyramid.base_edge * (Real.sqrt (myPyramid.lateral_edge^2 - (myPyramid.base_edge / 2)^2)) = 16 * Real.sqrt 33 :=
by
  sorry

end total_area_of_triangular_faces_l420_420207


namespace sum_of_integers_binom_eq_l420_420824

theorem sum_of_integers_binom_eq :
  let k_values := { k : ℕ | binomial 31 k = binomial 30 5 + binomial 30 6 }
  (finset.sum (finset.filter (λ k, k ∈ k_values) (finset.range 32))) = 31 :=
by
  sorry

end sum_of_integers_binom_eq_l420_420824


namespace composite_n_function_form_l420_420963

-- Define the required properties and conditions
def arithmetic_progression (a : List ℤ) : Prop :=
  ∃ d : ℤ, ∀ i : ℕ, i < a.length - 1 → a[i + 1] - a[i] = d

def function_property (f : ℤ → ℤ) (n : ℕ) : Prop :=
  ∀ (a : List ℤ), a.length = n → arithmetic_progression a →
    ∃ b : List ℤ, (b.perm (List.map f a)) ∧ arithmetic_progression b

-- State the theorem
theorem composite_n_function_form {n : ℕ} (h : n ≥ 3) :
  (∀ (f : ℤ → ℤ), (function_property f n → ∃ (c d : ℤ), ∀ x : ℤ, f(x) = c * x + d)) ↔ ¬ Nat.Prime n :=
sorry

end composite_n_function_form_l420_420963


namespace comparison_l420_420962

-- Given conditions
variables {a b m : ℝ}
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_m : 0 < m
axiom a_gt_b : a > b

-- Definitions based on the problem
def A := Real.sqrt (a + m) - Real.sqrt a
def B := Real.sqrt (b + m) - Real.sqrt b

-- Statement to be proved
theorem comparison : B > A :=
by sorry

end comparison_l420_420962


namespace domain_of_function_l420_420191

theorem domain_of_function : ∀ x : ℝ, (x ∈ set.Ioo(-∞) -3 ∪ set.Ioo -3 2 ∪ set.Ioo 2 ∞) ↔ f x ≠ 0 :=
begin
  sorry
end

end domain_of_function_l420_420191


namespace pyramid_total_area_l420_420295

noncomputable def pyramid_base_edge : ℝ := 8
noncomputable def pyramid_lateral_edge : ℝ := 7

theorem pyramid_total_area 
  (base_edge : ℝ)
  (lateral_edge : ℝ)
  (h_base_edge : base_edge = pyramid_base_edge)
  (h_lateral_edge : lateral_edge = pyramid_lateral_edge) 
: 4 * (1 / 2 * base_edge * real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 16 * real.sqrt 33 := 
sorry

end pyramid_total_area_l420_420295


namespace boat_against_stream_distance_l420_420934

def speed_in_still_water : ℝ := 7 -- speed of the boat in still water (km/hr)
def distance_along_stream : ℝ := 11 -- distance traveled along stream in one hour (km)

theorem boat_against_stream_distance (v_s : ℝ) (h1 : speed_in_still_water + v_s = distance_along_stream) :
  let effective_speed_against_stream := speed_in_still_water - v_s in
  effective_speed_against_stream * 1 = 3 :=
by
  let effective_speed_against_stream := speed_in_still_water - v_s
  have distance : effective_speed_against_stream = 3 := sorry
  exact distance  

end boat_against_stream_distance_l420_420934


namespace triangle_circumcircle_problem_l420_420182

theorem triangle_circumcircle_problem :
  let PQ : ℝ := 14
  let QR : ℝ := 48
  let RP : ℝ := 50
  let N := midpoint P R  -- Midpoint of PR, conceptually
  let omega := circumscribed_circle P Q R
  let S := intersection_point omega (perpendicular_bisector PR) (not_on_same_side Q)
  let PS := distance P S
  let a := 38
  let b := 1 in
  (PS = a * real.sqrt b ∧ int.floor (a + real.sqrt b) = 39) :=
sorry

end triangle_circumcircle_problem_l420_420182


namespace max_area_difference_l420_420676

theorem max_area_difference (l w l' w' : ℕ) 
  (h1 : 2 * l + 2 * w = 160) 
  (h2 : 2 * l' + 2 * w' = 160) : 
  abs ((l * w) - (l' * w')) ≤ 1600 := 
by 
  sorry

end max_area_difference_l420_420676


namespace general_formula_a_n_sum_S_n_formula_l420_420966

variable {q : ℕ} {n : ℕ}

-- a_n is a geometric sequence with a positive common ratio, a1 = 2, and a3 = a2 + 4
def a_n (n : ℕ) : ℕ := 2 ^ n
axiom a_n_def (n : ℕ) : a_n n = 2 ^ n

-- b_n is an arithmetic sequence with first term 1 and common difference 2
def b_n (n : ℕ) : ℕ := 2 * n - 1
axiom b_n_def (n : ℕ) : b_n n = 2 * n - 1

-- The general formula for the geometric sequence {a_n}
theorem general_formula_a_n : ∀ n: ℕ, a_n n = 2 ^ n := by
  intro n
  exact a_n_def n

-- The sum of the first n terms of {a_n + b_n}
def S_n (n : ℕ) : ℕ :=
  ∑ k in Finset.range n, a_n k + b_n k

theorem sum_S_n_formula : ∀ n: ℕ, S_n n = 2 ^ (n+1) + n^2 - 2 := by
  intro n
  sorry

end general_formula_a_n_sum_S_n_formula_l420_420966


namespace pyramid_area_l420_420302

noncomputable def base_edge : ℝ := 8
noncomputable def lateral_edge : ℝ := 7
noncomputable def height := real.sqrt (lateral_edge^2 - (base_edge / 2)^2)
noncomputable def one_triangle_area := (1 / 2) * base_edge * height
noncomputable def total_area := 4 * one_triangle_area

theorem pyramid_area : total_area = 16 * real.sqrt 33 := by
  sorry

end pyramid_area_l420_420302


namespace colorings_unique_l420_420625

def color := ℕ
def red : color := 0
def blue : color := 1
def green : color := 2

structure HexGrid :=
(columns : ℕ)
(hexagon : ℕ -> ℕ -> color)
(column_valid : ∀ c r, 0 ≤ c ∧ c < columns → 0 ≤ r → 
  (c = 0 → hexagon c r = red) ∧
  (0 < c → hexagon (c - 1) r ≠ hexagon c r ∧ 
           (r > 0 → hexagon c (r - 1) ≠ hexagon c r)))

theorem colorings_unique (G : HexGrid) : 
  (∀ c, 0 ≤ c ∧ c < G.columns → ∃! h : (ℕ → color), ∀ r, 0 ≤ r → G.hexagon c r = h r) 
  → G.columns = 1 ∨ G.columns = 2
  →  ∃! f : HexGrid, ∀ c, 0 ≤ c ∧ c < G.columns → 
     (∀ r, 0 ≤ r → G.hexagon c r = f.hexagon c r) :=
  by
    sorry

end colorings_unique_l420_420625


namespace neg_p_necessary_but_not_sufficient_for_neg_q_l420_420551

variable (p q : Prop)

theorem neg_p_necessary_but_not_sufficient_for_neg_q
  (h1 : p → q)
  (h2 : ¬ (q → p)) : 
  (¬p → ¬q) ∧ (¬q → ¬p) := 
sorry

end neg_p_necessary_but_not_sufficient_for_neg_q_l420_420551


namespace sum_of_ratios_of_ellipse_l420_420404

theorem sum_of_ratios_of_ellipse (x y : ℝ) :
  (3 * x^2 + 2 * x * y + 4 * y^2 - 15 * x - 24 * y + 56 = 0) →
  let ratio := λ x y, x / y in
  ∃ a b : ℝ, by (forall p : ((x, y) ∈ set_of (λ (x y : ℝ), 3 * x^2 + 2 * x * y + 4 * y^2 - 15 * x - 24 * y + 56 = 0)),
  a + b = 272 / 447 ∧
  (ratio p.1 p.2) = a ∨
  (ratio p.1 p.2) = b := by sorry

end sum_of_ratios_of_ellipse_l420_420404


namespace greatest_area_difference_l420_420670

theorem greatest_area_difference (l₁ w₁ l₂ w₂ : ℕ) (h₁ : 2 * l₁ + 2 * w₁ = 160) (h₂ : 2 * l₂ + 2 * w₂ = 160) : 
  abs (l₁ * w₁ - l₂ * w₂) = 1521 :=
sorry

end greatest_area_difference_l420_420670


namespace product_of_roots_l420_420964

def P (x : ℚ) : ℚ[x] := x^3 - 21 * x - 56

theorem product_of_roots :
  (\exists x : ℚ, ((x^3 - 21 * x - 56) = 0) ∧ (x = (7^(1/3) + (49)^(1/3)))) → 
  ∏ (r : ℚ) in (P x).roots, r = 56 :=
by
  -- Proof steps will go here
  sorry

end product_of_roots_l420_420964


namespace sum_of_elements_in_B_intersection_C_l420_420979

def A : Set ℕ := {1, 2, 3}

def B : Set ℕ := {n | ∃ x y ∈ A, x < y ∧ n = 2 * x + y}

def C : Set ℕ := {n | ∃ x y ∈ A, x > y ∧ n = 2 * x + y}

def intersection_sum (s1 s2 : Set ℕ) : ℕ :=
  (s1 ∩ s2).toFinset.sum id

theorem sum_of_elements_in_B_intersection_C :
  intersection_sum B C = 12 := by
  sorry

end sum_of_elements_in_B_intersection_C_l420_420979


namespace particle_sphere_intersection_distance_l420_420073

theorem particle_sphere_intersection_distance :
  let start := (1 : ℝ, 2 : ℝ, 3 : ℝ),
      end := (0 : ℝ, -2 : ℝ, -2 : ℝ),
      center := (0 : ℝ, 0 : ℝ, 0 : ℝ),
      radius := 2 
  in 
  -- Parametrize the line
  let line := λ t : ℝ, (1 - t, 2 - 4 * t, 3 - 5 * t),
      dist := λ (p1 p2 : ℝ × ℝ × ℝ), 
        Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2 + (p1.3 - p2.3) ^ 2),
      t_1 := (15 - 6 * Real.sqrt 11) / 21,
      t_2 := (15 + 6 * Real.sqrt 11) / 21,
      intersect1 := line t_1,
      intersect2 := line t_2
  in
  dist intersect1 intersect2 = 12 * Real.sqrt 66 / 7 ∧ (12 + 66) = 78 :=
by
  sorry

end particle_sphere_intersection_distance_l420_420073


namespace statement_1_statement_2_statement_3_statement_4_l420_420109

-- Define the set M
def M : Set ℤ := {x | ∃ a b : ℤ, x = a^2 - b^2}

-- Statement ①: All odd numbers belong to M
theorem statement_1 (k : ℤ) : (2 * k + 1) ∈ M := 
by exists (k + 1), k; exact rfl
-- Add sorry to skip the proof details
sorry

-- Statement ②: If (2 * k) ∈ M, does k ∈ M?
theorem statement_2 (k : ℤ) : (2 * k) ∈ M → k ∈ M := 
by exists 4, 2; exact rfl
-- This is actually false as per counterexample, consider providing a counterexample.
-- 12 ∈ M but 6 ∉ M
example : (12 ∈ M) ∧ (6 ∉ M) := 
by 
  rfl; not_m

-- Statement ③: If a ∈ M and b ∈ M, then ab ∈ M
theorem statement_3 (a b : ℤ) (ha : a ∈ M) (hb : b ∈ M) : (a * b) ∈ M := 
by exists 4, 2; exact rfl
-- Add sorry to skip the proof details
sorry

-- Statement ④: If we arrange non-M numbers, their sum belongs to M
theorem statement_4 (n : ℕ) (h : 1 ∈ List.filter (λ x, x ∉ M) (List.range n)) : 
  Σi<n, i ∉ M ∈ M :=
-- Since n = 1 leads to contradiction.
otherwise_right

-- Adding sorry to complete and bypass proofs

end statement_1_statement_2_statement_3_statement_4_l420_420109


namespace pyramid_area_l420_420248

theorem pyramid_area :
  ∀ (a b : ℝ), a = 8 → b = 7 → 4 * (1/2 * a * sqrt (b^2 - (a/2)^2)) = 16 * sqrt 33 :=
by
  intros a b ha hb
  rw [ha, hb]
  have h1 : a / 2 = 4 := by norm_num [ha]
  have h2 : b^2 - (a / 2)^2 = 33 :=
    by
      calc
        b^2 - (a / 2)^2 = 49 - 16 := by norm_num [hb]
        ... = 33 := by norm_num
  rw [h1, h2, sqrt 33, mul_one, mul_one, half_mul, mul_comm (1/2) a, ←mul_assoc, mul_comm 4 4]
  norm_num
  sorry

end pyramid_area_l420_420248


namespace milk_cartons_total_l420_420511

theorem milk_cartons_total (regular_milk soy_milk : ℝ) (h1 : regular_milk = 0.5) (h2 : soy_milk = 0.1) :
  regular_milk + soy_milk = 0.6 :=
by
  rw [h1, h2]
  norm_num

end milk_cartons_total_l420_420511


namespace trigonometric_relation_l420_420842

theorem trigonometric_relation (x y z : ℝ) (hx : 0 < x ∧ x < π / 2)
  (hy : 0 < y ∧ y < π / 2) (hz : 0 < z ∧ z < π / 2)
  (h : (sin x + cos x) * (sin y + 2 * cos y) * (sin z + 3 * cos z) = 10) :
  x = π / 4 ∧ x > y ∧ y > z :=
sorry

end trigonometric_relation_l420_420842


namespace ben_david_bagel_cost_l420_420438

theorem ben_david_bagel_cost (B D : ℝ)
  (h1 : D = 0.5 * B)
  (h2 : B = D + 16) :
  B + D = 48 := 
sorry

end ben_david_bagel_cost_l420_420438


namespace digits_right_of_decimal_point_l420_420041

theorem digits_right_of_decimal_point
  (a b c : ℕ) (h_a : a = 5^8) (h_b : b = 10^5) (h_c : c = 16) :
  let expr := (a: ℚ) / (b * c) in
  let decimal := (5^3 : ℚ) / (2^9) in
  (decimal.digits 10).length - 1 = 9 := 
by
  sorry

end digits_right_of_decimal_point_l420_420041


namespace polynomial_g_eq_find_g_of_x2_minus_2_l420_420102

open Polynomial

noncomputable def g (x : ℝ) : ℝ := x^2 + 2 * x - 4

theorem polynomial_g_eq (x : ℝ) : g (x^2 + 2) = x^4 + 6 * x^2 + 4 :=
by sorry

theorem find_g_of_x2_minus_2 (x : ℝ) : g(x^2 - 2) = x^4 - 2 * x^2 - 4 :=
by sorry

end polynomial_g_eq_find_g_of_x2_minus_2_l420_420102


namespace pyramid_total_area_l420_420228

/-- The total area of the four triangular faces of a right, square-based pyramid
whose base edges measure 8 units and lateral edges measure 7 units is 16√33. -/
theorem pyramid_total_area :
  let base_edge := 8
  let lateral_edge := 7
  4 * (1 / 2 * base_edge * Real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 16 * Real.sqrt 33 := 
by
  let base_edge := 8
  let lateral_edge := 7
  have h1 : 4 * (1 / 2 * base_edge * Real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 
              4 * (1 / 2 * 8 * Real.sqrt (49 - 16)) := 
    by sorry
  have h2 : 4 * (1 / 2 * 8 * Real.sqrt (49 - 16)) = 4 * (4 * Real.sqrt 33) := 
    by sorry
  have h3 : 4 * (4 * Real.sqrt 33) = 16 * Real.sqrt 33 := 
    by sorry
  exact eq.trans (eq.trans h1 h2) h3

end pyramid_total_area_l420_420228


namespace find_value_of_p_l420_420867

variable (x y : ℝ)

/-- Given that the hyperbola has the equation x^2 / 4 - y^2 / 12 = 1
    and the eccentricity e = 2, and that the parabola x = 2 * p * y^2 has its focus at (e, 0), 
    prove that the value of the real number p is 1/8. -/
theorem find_value_of_p :
  (∃ (p : ℝ), 
    (∀ (x y : ℝ), x^2 / 4 - y^2 / 12 = 1) ∧ 
    (∀ (x y : ℝ), x = 2 * p * y^2) ∧
    (2 = 2)) →
    ∃ (p : ℝ), p = 1/8 :=
by 
  sorry

end find_value_of_p_l420_420867


namespace general_term_an_sum_bn_formula_l420_420094

def seq_an (n : ℕ) : ℝ :=
  if n = 1 then 1 else -2 / ((2 * n - 1) * (2 * n - 3))

def seq_sn (n : ℕ) : ℝ :=
  (List.range (n + 1)).map (λ i => seq_an (i + 1)).sum

def seq_bn (n : ℕ) : ℝ :=
  seq_sn n / (2 * n + 1)

def sum_bn (n : ℕ) : ℝ :=
  (List.range (n + 1)).map (λ i => seq_bn (i + 1)).sum

theorem general_term_an (n : ℕ) : seq_an n = 
  if n = 1 then 1 else -2 / ((2 * n - 1) * (2 * n - 3)) := sorry

theorem sum_bn_formula (n : ℕ) : sum_bn n = n / (2 * n + 1) := sorry

end general_term_an_sum_bn_formula_l420_420094


namespace find_rotation_and_center_of_rotation_l420_420183

theorem find_rotation_and_center_of_rotation :
  let A := (0, 0)
  let B := (0, 10)
  let C := (20, 0)
  let P := (30, 0)
  let Q := (30, 20)
  let R := (50, 0)
  let n := 90
  let x := 30
  let y := 0
  0 < n ∧ n < 180 →
  by 
    let rotation := (n, (x, y))
    sorry
    { (P, Q, R) = rotate_triangle (A, B, C) rotation } →
  n + x + y = 120 :=
by sorry

end find_rotation_and_center_of_rotation_l420_420183


namespace length_CF_l420_420942

-- Definitions of the geometric shape and its properties
def isosceles_trapezoid (ABCD : Type) [has_length ABCD] := 
  ∃ (AD BC : ℝ), AD = 7 ∧ BC = 7 ∧ side_length ABCD "AB" = 6 ∧ side_length ABCD "DC" = 12

-- Definitions of point relations in the trapezoid
def point_relations (DF : ℝ) (EB : ℝ) : Prop :=
  ∃ (DE EB : ℝ), DE = 2 * EB

-- Theorem stating the proof problem
theorem length_CF (ABCD : Type) [has_length ABCD] (DF : ℝ) (EB : ℝ) 
  (h_trap : isosceles_trapezoid ABCD) (h_point : point_relations DF EB) : 
  side_length "CF" = 6 :=
by
  sorry

end length_CF_l420_420942


namespace walkways_area_calc_l420_420592

structure Garden :=
  (bed_length : ℝ)
  (bed_width  : ℝ)
  (num_rows   : ℕ)
  (num_cols   : ℕ)
  (walkway_width : ℝ)
  (pond_length : ℝ)
  (pond_width  : ℝ)

def total_garden_area (G : Garden) : ℝ :=
  let bed_total_width := G.num_cols * G.bed_length
  let walkway_total_width := (G.num_cols + 1) * G.walkway_width
  let total_width := bed_total_width + walkway_total_width

  let bed_total_height := G.num_rows * G.bed_width
  let walkway_total_height := (G.num_rows + 1) * G.walkway_width
  let total_height := bed_total_height + walkway_total_height
  
  total_width * total_height

def pond_area (G : Garden) : ℝ := G.pond_length * G.pond_width

def adjusted_garden_area (G : Garden) : ℝ :=
  total_garden_area G - pond_area G

def total_beds_area (G : Garden) : ℝ :=
  (G.num_rows * G.num_cols) * (G.bed_length * G.bed_width)

def walkway_area (G : Garden) : ℝ :=
  adjusted_garden_area G - total_beds_area G

theorem walkways_area_calc (G : Garden):
  G.bed_length = 4 → 
  G.bed_width = 3 → 
  G.num_rows = 4 → 
  G.num_cols = 3 → 
  G.walkway_width = 2 → 
  G.pond_length = 3 → 
  G.pond_width = 2 → 
  walkway_area G = 290 :=
by {
  intros,
  -- Proof goes here
  sorry
}

end walkways_area_calc_l420_420592


namespace xy_parallel_ab_l420_420536

open EuclideanGeometry

variable {A B C M K X Y : Point}
variable {triangle_ABC : Triangle A B C}
variable {circumcircle_ABC : Circle circumcenter radius}
variable {midpoint_M : Midpoint M A C}
variable {K_on_minor_arc_AC : InMinorArc circumcircle_ABC K A C}
variable {AKM_ninety_deg : ∠(A, K, M) = 90}
variable {intersection_X : X = LineIntersection (line_through B K) (line_through A M)}
variable {A_altitude_meet_BM_at_Y : Y = LineIntersection (line_through_A_altitude A B C) (line_through B M)}

theorem xy_parallel_ab :
  Parallel (line_through X Y) (line_through A B) := 
by
  sorry

end xy_parallel_ab_l420_420536


namespace compare_neg5_neg7_l420_420785

theorem compare_neg5_neg7 : -5 > -7 := 
by
  sorry

end compare_neg5_neg7_l420_420785


namespace pyramid_area_l420_420303

noncomputable def base_edge : ℝ := 8
noncomputable def lateral_edge : ℝ := 7
noncomputable def height := real.sqrt (lateral_edge^2 - (base_edge / 2)^2)
noncomputable def one_triangle_area := (1 / 2) * base_edge * height
noncomputable def total_area := 4 * one_triangle_area

theorem pyramid_area : total_area = 16 * real.sqrt 33 := by
  sorry

end pyramid_area_l420_420303


namespace sum_of_reciprocals_leq_l420_420608

theorem sum_of_reciprocals_leq (N : ℕ) (S : Finset ℕ) (H : ∀ n ∈ S, ¬ has_sequence_2048 n) :
  (∑ n in S, (1 : ℝ) / n) ≤ 400000 := 
sorry

end sum_of_reciprocals_leq_l420_420608


namespace rectangle_area_l420_420146

def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

def area_of_rectangle 
  (A B C : ℝ × ℝ) 
  (dist_AB : distance A B = real.sqrt 41) 
  (dist_BC : distance B C = 7) 
  (A_eq : A = (-3, 6)) 
  (B_eq : B = (1, 1)) 
  (C_eq : C = (1, -6)) : ℝ :=
  dist_AB * dist_BC

theorem rectangle_area 
  (A B C : ℝ × ℝ) 
  (h1 : A = (-3, 6)) 
  (h2 : B = (1, 1)) 
  (h3 : C = (1, -6)) 
  (h4 : distance A B = real.sqrt 41) 
  (h5 : distance B C = 7) : 
  area_of_rectangle A B C h4 h5 h1 h2 h3 = 7 * real.sqrt 41 := 
sorry

end rectangle_area_l420_420146


namespace inverse_of_A_l420_420819

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![4, 5], ![-2, 9]]

def A_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![9 / 46, -5 / 46], ![1 / 23, 2 / 23]]

theorem inverse_of_A :
  A⁻¹ = A_inv :=
by
  sorry -- proof omitted

end inverse_of_A_l420_420819


namespace find_linear_function_b_l420_420051

theorem find_linear_function_b (b : ℝ) :
  (∃ b, (∀ x y, y = 2 * x + b - 2 → (x = -1 ∧ y = 0)) → b = 4) :=
sorry

end find_linear_function_b_l420_420051


namespace max_area_difference_160_perimeter_rectangles_l420_420660

theorem max_area_difference_160_perimeter_rectangles : 
  ∃ (l1 w1 l2 w2 : ℕ), (2 * l1 + 2 * w1 = 160) ∧ (2 * l2 + 2 * w2 = 160) ∧ 
  (l1 * w1 - l2 * w2 = 1521) := sorry

end max_area_difference_160_perimeter_rectangles_l420_420660


namespace bus_seats_capacity_l420_420926

-- Define the conditions
variable (x : ℕ) -- number of people each seat can hold
def left_side_seats := 15
def right_side_seats := left_side_seats - 3
def back_seat_capacity := 7
def total_capacity := left_side_seats * x + right_side_seats * x + back_seat_capacity

-- State the theorem
theorem bus_seats_capacity :
  total_capacity x = 88 → x = 3 := by
  sorry

end bus_seats_capacity_l420_420926


namespace pyramid_area_l420_420296

noncomputable def base_edge : ℝ := 8
noncomputable def lateral_edge : ℝ := 7
noncomputable def height := real.sqrt (lateral_edge^2 - (base_edge / 2)^2)
noncomputable def one_triangle_area := (1 / 2) * base_edge * height
noncomputable def total_area := 4 * one_triangle_area

theorem pyramid_area : total_area = 16 * real.sqrt 33 := by
  sorry

end pyramid_area_l420_420296


namespace area_of_gray_region_l420_420783

-- Define the centers and radii of the circles
def center_A : ℝ × ℝ := (4, 4)
def center_B : ℝ × ℝ := (12, 4)
def radius_A : ℝ := 4
def radius_B : ℝ := 4

-- Define the calculation for the area of the gray region
def area_gray_region : ℝ :=
  let length_rectangle := 12 - 4
  let width_rectangle := 4 - 0
  let area_rectangle := length_rectangle * width_rectangle
  let area_sector := (1 / 4) * real.pi * radius_A^2
  let total_sector_area := 2 * area_sector
  area_rectangle - total_sector_area

-- Statement to be proved: The area of the gray region is 32 - 8π
theorem area_of_gray_region :
  area_gray_region = 32 - 8 * real.pi := sorry

end area_of_gray_region_l420_420783


namespace largest_coins_l420_420323

theorem largest_coins (n k : ℕ) (h1 : n = 13 * k + 3) (h2 : n < 150) : n = 146 :=
by
  sorry

end largest_coins_l420_420323


namespace pyramid_area_l420_420284

theorem pyramid_area (base_edge lateral_edge : ℝ) (H_base_edge : base_edge = 8) (H_lateral_edge : lateral_edge = 7) :
  4 * (1 / 2 * base_edge * real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 16 * real.sqrt 33 :=
by
  sorry

end pyramid_area_l420_420284


namespace ratio_father_to_children_after_5_years_l420_420809

def father's_age := 15
def sum_children_ages := father's_age / 3

def father's_age_after_5_years := father's_age + 5
def sum_children_ages_after_5_years := sum_children_ages + 10

theorem ratio_father_to_children_after_5_years :
  father's_age_after_5_years / sum_children_ages_after_5_years = 4 / 3 := by
  sorry

end ratio_father_to_children_after_5_years_l420_420809


namespace leak_empties_tank_in_15_hours_l420_420710

theorem leak_empties_tank_in_15_hours 
  (A_fills_tank_6_hours : ∀ t:ℝ, t = 6 → A_rate : ℝ → A_rate = 1/t)
  (A_leak_fills_tank_10_hours : ∀ t:ℝ, t = 10 → A_leak_rate : ℝ → A_leak_rate = 1/t) : 
  ∃ L : ℝ, L = 1/15 := by
sorry

end leak_empties_tank_in_15_hours_l420_420710


namespace integral_result_l420_420713

theorem integral_result : ∫ x in (0 : ℝ)..π / 4, (x^2 + 17.5) * sin (2 * x) = (68 + π) / 8 := by
  sorry

end integral_result_l420_420713


namespace residue_7_1234_mod_13_l420_420687

theorem residue_7_1234_mod_13 :
  (7 : ℤ) ^ 1234 % 13 = 12 :=
by
  -- given conditions as definitions
  have h1 : (7 : ℤ) % 13 = 7 := by norm_num
  
  -- auxiliary calculations
  have h2 : (49 : ℤ) % 13 = 10 := by norm_num
  have h3 : (100 : ℤ) % 13 = 9 := by norm_num
  have h4 : (81 : ℤ) % 13 = 3 := by norm_num
  have h5 : (729 : ℤ) % 13 = 1 := by norm_num

  -- the actual problem we want to prove
  sorry

end residue_7_1234_mod_13_l420_420687


namespace summation_series_lt_one_l420_420786

theorem summation_series_lt_one :
  (∑ k in finset.range 2012, (k + 1 : ℝ) / ((k + 2)! : ℝ)) < 1 :=
by sorry

end summation_series_lt_one_l420_420786


namespace total_area_of_triangular_faces_l420_420197

noncomputable def pyramid_base_edge : ℝ := 8
noncomputable def pyramid_lateral_edge : ℝ := 7

structure Pyramid where
  base_edge : ℝ
  lateral_edge : ℝ

def myPyramid : Pyramid :=
{ base_edge := pyramid_base_edge,
  lateral_edge := pyramid_lateral_edge }

theorem total_area_of_triangular_faces :
  4 * (1 / 2) * myPyramid.base_edge * (Real.sqrt (myPyramid.lateral_edge^2 - (myPyramid.base_edge / 2)^2)) = 16 * Real.sqrt 33 :=
by
  sorry

end total_area_of_triangular_faces_l420_420197


namespace pyramid_total_area_l420_420220

/-- The total area of the four triangular faces of a right, square-based pyramid
whose base edges measure 8 units and lateral edges measure 7 units is 16√33. -/
theorem pyramid_total_area :
  let base_edge := 8
  let lateral_edge := 7
  4 * (1 / 2 * base_edge * Real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 16 * Real.sqrt 33 := 
by
  let base_edge := 8
  let lateral_edge := 7
  have h1 : 4 * (1 / 2 * base_edge * Real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 
              4 * (1 / 2 * 8 * Real.sqrt (49 - 16)) := 
    by sorry
  have h2 : 4 * (1 / 2 * 8 * Real.sqrt (49 - 16)) = 4 * (4 * Real.sqrt 33) := 
    by sorry
  have h3 : 4 * (4 * Real.sqrt 33) = 16 * Real.sqrt 33 := 
    by sorry
  exact eq.trans (eq.trans h1 h2) h3

end pyramid_total_area_l420_420220


namespace find_13th_result_l420_420604

theorem find_13th_result
  (avg_25 : ℕ → ℕ)
  (avg_1_to_12 : ℕ → ℕ)
  (avg_14_to_25 : ℕ → ℕ)
  (h1 : avg_25 25 = 50)
  (h2 : avg_1_to_12 12 = 14)
  (h3 : avg_14_to_25 12 = 17) :
  ∃ (X : ℕ), X = 878 := sorry

end find_13th_result_l420_420604


namespace gcd_sum_4n_plus_6_l420_420770

theorem gcd_sum_4n_plus_6 (n : ℕ) (h : n > 0) : 
  let possible_gcds := {d : ℕ | d = Nat.gcd (4 * n + 6) n ∧ d > 0}
  in possible_gcds.sum = 12 := 
by
  sorry

end gcd_sum_4n_plus_6_l420_420770


namespace pyramid_total_area_l420_420287

noncomputable def pyramid_base_edge : ℝ := 8
noncomputable def pyramid_lateral_edge : ℝ := 7

theorem pyramid_total_area 
  (base_edge : ℝ)
  (lateral_edge : ℝ)
  (h_base_edge : base_edge = pyramid_base_edge)
  (h_lateral_edge : lateral_edge = pyramid_lateral_edge) 
: 4 * (1 / 2 * base_edge * real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 16 * real.sqrt 33 := 
sorry

end pyramid_total_area_l420_420287


namespace max_area_difference_160_perimeter_rectangles_l420_420658

theorem max_area_difference_160_perimeter_rectangles : 
  ∃ (l1 w1 l2 w2 : ℕ), (2 * l1 + 2 * w1 = 160) ∧ (2 * l2 + 2 * w2 = 160) ∧ 
  (l1 * w1 - l2 * w2 = 1521) := sorry

end max_area_difference_160_perimeter_rectangles_l420_420658


namespace probability_divisible_by_8_l420_420187

def roll_dice : Type := fin 6

def product_of_eight_rolls (rolls : vector roll_dice 8) : ℕ :=
  rolls.to_list.map (λ r, r + 1).prod

def is_divisible_by_8 (n : ℕ) : Prop := n % 8 = 0

theorem probability_divisible_by_8 :
  let prob := (733 : ℚ) / 768 in
  ∀ (rolls : vector roll_dice 8), is_divisible_by_8 (product_of_eight_rolls rolls) -> Pr (prob) = 733 / 768 := 
sorry

end probability_divisible_by_8_l420_420187


namespace modulus_of_Z_Z_is_root_of_quadratic_l420_420014

noncomputable def Z : ℂ := (1 / 2) / (1 + complex.i) + (-5 / 4 + (9 / 4) * complex.i)

theorem modulus_of_Z : complex.abs Z = real.sqrt 5 := 
sorry

theorem Z_is_root_of_quadratic {p q : ℝ} (h : 2 * Z^2 + p * Z + q = 0) : p = 4 ∧ q = 10 := 
sorry

end modulus_of_Z_Z_is_root_of_quadratic_l420_420014


namespace transformed_data_average_variance_l420_420453

-- Definitions and conditions from the problem
variables {n : ℕ} (x : Fin n → ℝ)
def mean (x : Fin n → ℝ) := ∑ i, x i / n
def std_dev (x : Fin n → ℝ) := real.sqrt (∑ i, (x i - mean x)^2 / n)

-- Transformed data
def transformed_data (x : Fin n → ℝ) (i : Fin n) := 3 * x i - 1

-- Lean 4 Statement of the problem
theorem transformed_data_average_variance (x : Fin n → ℝ) :
  mean (transformed_data x) = 3 * mean x - 1 ∧
  std_dev (transformed_data x) ^ 2 = 9 * (std_dev x) ^ 2 :=
sorry

end transformed_data_average_variance_l420_420453


namespace no_closed_non_self_intersecting_diagonal_path_l420_420949

open Finset

-- Define the vertices of the cube
def vertices : Finset (ℕ × ℕ × ℕ) := 
  {(0, 0, 0), (0, 0, 3), (0, 3, 0), (0, 3, 3), 
   (3, 0, 0), (3, 0, 3), (3, 3, 0), (3, 3, 3)}

-- Define the parity of a vertex sum (even or odd)
def parity (v : ℕ × ℕ × ℕ) : ℤ := (v.1 + v.2 + v.3) % 2

theorem no_closed_non_self_intersecting_diagonal_path : ¬ ∃ (path : List (ℕ × ℕ × ℕ)),
  (∀ (v ∈ vertices), v ∈ path) ∧
  (∀ (i : ℕ), i < path.length - 1 →
     ((path.nth i).match (i1: ℕ × ℕ × ℕ × ℕ × ℕ × ℕ) → 
        (∃ (a b : ℕ), (abs (a - b) = 1 ∧ 
        abs (c - d) = 1) ∧ e = f)) ∧
     path.nth i ∈ vertices) ∧
  path.head = path.last ∧
  path.nodup ∧
  (∀ (i j : ℕ), i ≠ j → path.nth i ≠ path.nth j) → 
  ((parity (path.head)) = (parity (path.last))) :=
by
  sorry

end no_closed_non_self_intersecting_diagonal_path_l420_420949


namespace total_area_of_triangular_faces_l420_420198

noncomputable def pyramid_base_edge : ℝ := 8
noncomputable def pyramid_lateral_edge : ℝ := 7

structure Pyramid where
  base_edge : ℝ
  lateral_edge : ℝ

def myPyramid : Pyramid :=
{ base_edge := pyramid_base_edge,
  lateral_edge := pyramid_lateral_edge }

theorem total_area_of_triangular_faces :
  4 * (1 / 2) * myPyramid.base_edge * (Real.sqrt (myPyramid.lateral_edge^2 - (myPyramid.base_edge / 2)^2)) = 16 * Real.sqrt 33 :=
by
  sorry

end total_area_of_triangular_faces_l420_420198


namespace quadrilateral_inequality_l420_420518

variable {A B C D : Type} [IsConvexQuadrilateral A B C D]

theorem quadrilateral_inequality (h: AB + BD ≤ AC + CD) : AB < AC :=
by
  sorry

end quadrilateral_inequality_l420_420518


namespace greatest_area_difference_l420_420665

theorem greatest_area_difference 
  (l w l' w' : ℕ) 
  (h1 : 2 * l + 2 * w = 160) 
  (h2 : 2 * l' + 2 * w' = 160) :
  abs ((l * w) - (l' * w')) ≤ 1521 := 
sorry

end greatest_area_difference_l420_420665


namespace circle_center_sum_l420_420609

theorem circle_center_sum (x y : ℝ) (hx : (x, y) = (3, -4)) :
  (x + y) = -1 :=
by {
  -- We are given that the center of the circle is (3, -4)
  sorry -- Proof is omitted
}

end circle_center_sum_l420_420609


namespace radius_of_tangent_circle_l420_420370

theorem radius_of_tangent_circle (R r1 : ℝ) (hR : R = 2) (hr1 : r1 = 1) : ∃ r : ℝ, r = 8 / 9 :=
by
  use 8 / 9
  sorry

end radius_of_tangent_circle_l420_420370


namespace greatest_area_difference_l420_420671

theorem greatest_area_difference (l₁ w₁ l₂ w₂ : ℕ) (h₁ : 2 * l₁ + 2 * w₁ = 160) (h₂ : 2 * l₂ + 2 * w₂ = 160) : 
  abs (l₁ * w₁ - l₂ * w₂) = 1521 :=
sorry

end greatest_area_difference_l420_420671


namespace pyramid_volume_QEFGH_l420_420133

noncomputable def volume_of_pyramid (EF FG QE : ℝ) : ℝ :=
  (1 / 3) * EF * FG * QE

theorem pyramid_volume_QEFGH :
  let EF := 10
  let FG := 5
  let QE := 9
  volume_of_pyramid EF FG QE = 150 := by
  sorry

end pyramid_volume_QEFGH_l420_420133


namespace find_fffive_l420_420020

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 4*x + 3 else 3 - x

theorem find_fffive : f (f 5) = -1 :=
  sorry

end find_fffive_l420_420020


namespace part_I_part_II_part_III_l420_420881

open Real

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 1 / (log (x^2 + 1) - 2 * log x) - x^2 - a * x

theorem part_I (x : ℝ) (h : x > 0) : f x 0 > 0 :=
sorry

theorem part_II (x : ℝ) (h : x > 0) (hx : f x a > 0) : a ≤ 0 :=
sorry

theorem part_III (n : ℕ) (hn : n ≥ 2) : 
  log ((finset.range (n + 1)).prod (λ k, 1 + ((k + 2)^2))) < 1 + 2 * log ((finset.range (n - 1 + 1)).prod (λ k, k + 2)) :=
sorry

end part_I_part_II_part_III_l420_420881


namespace area_of_transformed_region_l420_420544

theorem area_of_transformed_region : 
  let T : ℝ := 15
  let A : Matrix (Fin 2) (Fin 2) ℝ := ![![3, 4], ![6, -2]]
  (abs (Matrix.det A) * T = 450) := 
  sorry

end area_of_transformed_region_l420_420544


namespace isosceles_right_triangle_probability_l420_420516

theorem isosceles_right_triangle_probability :
  ∀ (A B C : Type*) [metric_space A] [metric_space B] [metric_space C]
  (triangle_ABC : triangle A B C)
  (is_isosceles_right : isosceles_right_triangle triangle_ABC)
  (M : A),
  (M ∈ segment A B) →
  (probability (λ M, dist A M < dist A C) = (√2 / 2) : ℝ) := sorry

end isosceles_right_triangle_probability_l420_420516


namespace perpendicular_PA_BD_length_PC_l420_420862

noncomputable def point := ℝ × ℝ × ℝ
noncomputable def vector (a b : point) := (b.1 - a.1, b.2 - a.2, b.3 - a.3)
def dot_product (u v : point) := u.1 * v.1 + u.2 * v.2 + u.3 * v.3
def norm (u : point) := real.sqrt (dot_product u u)

variables (P A B C D : point)
variable (PA PB PD : ℝ)
variables (AB AD : ℝ)

-- Given conditions
axiom P_outside_plane : ¬ collinear P A B ∧ ¬ collinear P A D
axiom angle_PAB_90 : dot_product (vector P A) (vector A B) = 0
axiom angle_PAD_90 : dot_product (vector P A) (vector A D) = 0
axiom length_PA : norm (vector P A) = PA
axiom length_AB : norm (vector A B) = AB
axiom length_AD : norm (vector A D) = AD
axiom value_PA : PA = 12
axiom value_AB : AB = 3
axiom value_AD : AD = 4

-- Questions to prove
theorem perpendicular_PA_BD : ∀ P A B D PA PB PD, PA ⟂ BD := by {
  sorry
}
theorem length_PC : ∀ P A B C D PA PB PD, PC = 13 := by {
  sorry
}

end perpendicular_PA_BD_length_PC_l420_420862


namespace rectangle_area_48_l420_420936

-- Define the rectangle structure
structure Rectangle :=
  (A B C D : Point)
  (AB BC CD DA AC: ℝ)
  (is_rectangle : (ABCD: Rectangle) ∧ (angle A B C = 90) ∧ (AB = 8) ∧ (AC = 10))

-- Define a point in 2D space
structure Point :=
  (x y : ℝ)

-- Define the theorem for area calculation of given rectangle
theorem rectangle_area_48 {R : Rectangle} (h_cond : R.AB = 8 ∧ R.AC = 10 ∧ R.is_rectangle)
: R.AB * R.BC = 48 :=
sorry

end rectangle_area_48_l420_420936


namespace sum_of_possible_integer_values_l420_420860

theorem sum_of_possible_integer_values (m : ℤ) (h : 1 < 3 * m ∧ 3 * m ≤ 45) :
  ∑ i in finset.filter (λ x, 1 < 3 * x ∧ 3 * x ≤ 45) (finset.range 16), i = 120 := by
  sorry

end sum_of_possible_integer_values_l420_420860


namespace sufficient_condition_for_proposition_l420_420854

theorem sufficient_condition_for_proposition :
  ∀ (a : ℝ), (0 < a ∧ a < 4) → (∀ x : ℝ, a * x ^ 2 + a * x + 1 > 0) := 
sorry

end sufficient_condition_for_proposition_l420_420854


namespace percentage_round_trip_completed_l420_420745

theorem percentage_round_trip_completed :
  ∀ (d_center : ℝ) (d_return : ℝ),
    d_center = 200 →
    d_return = d_center * 1.1 →
    ∀ (percentage_return_trip_completed : ℝ) (distance_traveled_return : ℝ) (total_distance_traveled : ℝ) (entire_round_trip_distance : ℝ),
      percentage_return_trip_completed = 0.4 →
      distance_traveled_return = percentage_return_trip_completed * d_return →
      total_distance_traveled = d_center + distance_traveled_return →
      entire_round_trip_distance = d_center + d_return →
      total_distance_traveled / entire_round_trip_distance * 100 ≈ 68.57 :=
begin
  sorry
end

end percentage_round_trip_completed_l420_420745


namespace number_of_tiles_in_each_row_l420_420603

-- Define the given conditions
def area_of_room : ℝ := 256
def tile_size_in_inches : ℝ := 8
def inches_per_foot : ℝ := 12

-- Length of the room in feet derived from the given area
def side_length_in_feet := Real.sqrt area_of_room

-- Convert side length from feet to inches
def side_length_in_inches := side_length_in_feet * inches_per_foot

-- The question: Prove that the number of tiles in each row is 24
theorem number_of_tiles_in_each_row :
  side_length_in_inches / tile_size_in_inches = 24 :=
sorry

end number_of_tiles_in_each_row_l420_420603


namespace coordinates_of_point_P_l420_420464

theorem coordinates_of_point_P 
  (x y : ℝ)
  (h1 : y = x^3 - x)
  (h2 : (3 * x^2 - 1) = 2)
  (h3 : ∀ x y, x + 2 * y = 0 → ∃ m, -1/(m) = 2) :
  (x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = 0) :=
by
  sorry

end coordinates_of_point_P_l420_420464


namespace principal_sum_investment_l420_420744

theorem principal_sum_investment 
    (P R : ℝ) 
    (h1 : (P * 5 * (R + 2)) / 100 - (P * 5 * R) / 100 = 180)
    (h2 : (P * 5 * (R + 3)) / 100 - (P * 5 * R) / 100 = 270) :
    P = 1800 :=
by
  -- These are the hypotheses generated for Lean, the proof steps are omitted
  sorry

end principal_sum_investment_l420_420744


namespace martha_total_points_l420_420996

-- Define the costs and points
def cost_beef := 11 * 3
def cost_fruits_vegetables := 4 * 8
def cost_spices := 6 * 3
def cost_other := 37

def total_spending := cost_beef + cost_fruits_vegetables + cost_spices + cost_other

def points_per_dollar := 50 / 10
def base_points := total_spending * points_per_dollar
def bonus_points := if total_spending > 100 then 250 else 0

def total_points := base_points + bonus_points

-- The theorem to prove the question == answer given the conditions
theorem martha_total_points :
  total_points = 850 :=
by
  sorry

end martha_total_points_l420_420996


namespace approximate_value_in_interval_l420_420057

-- Define the interval [x_i, x_{i+1}]
variables {x_i x_i_plus_1 ξ_i : ℝ}

-- Define the function f
variable {f : ℝ → ℝ}

-- The condition for the interval
axiom interval_condition : x_i ≤ ξ_i ∧ ξ_i ≤ x_i_plus_1

theorem approximate_value_in_interval :
  f(ξ_i) = f(x_i) ∨ f(ξ_i) = f(x_i_plus_1) ∨ (x_i ≤ ξ_i ∧ ξ_i ≤ x_i_plus_1 → f(ξ_i) = f(ξ_i)) :=
by
  -- The proof is omitted
  sorry

end approximate_value_in_interval_l420_420057


namespace pyramid_triangular_face_area_l420_420212

theorem pyramid_triangular_face_area 
  (base : ℝ) (lateral : ℝ)
  (h_base : base = 8) (h_lateral : lateral = 7) :
  let height := sqrt (lateral^2 - (base / 2)^2) in
  4 * (1 / 2 * base * height) = 16 * sqrt 33 := 
by
  sorry

end pyramid_triangular_face_area_l420_420212


namespace integral_of_x_squared_eq_9_l420_420906

theorem integral_of_x_squared_eq_9 (T : ℝ) (h : ∫ x in 0..T, x^2 = 9) : T = 3 :=
sorry

end integral_of_x_squared_eq_9_l420_420906


namespace find_p_inversely_proportional_l420_420147

theorem find_p_inversely_proportional :
  ∀ (p q r : ℚ), (p * (r * q) = k) → (p = 16) → (q = 8) → (r = 2) →
  (k = 256) → (q' = 10) → (r' = 3) →
  (∃ p' : ℚ, p' = 128 / 15) :=
by
  sorry

end find_p_inversely_proportional_l420_420147


namespace geom_seq_term_eq_6_l420_420641

open Nat

theorem geom_seq_term_eq_6 {a : ℕ → ℝ} (S : ℕ → ℝ) (a1 a2 a3: ℝ)
  (h1: S (2 * n) = 4 * (∑ i in range n, a (2 * i + 1)))
  (h2: a1 * a2 * a3 = 27) : 
  a 6 = 243 := by
  sorry

end geom_seq_term_eq_6_l420_420641


namespace math_problem_l420_420869

-- Define the arithmetic sequence with common difference d and first term a₁
noncomputable def arithmetic_seq (d : ℕ) (a₁ : ℕ := 2) : ℕ → ℕ
| 0       := 0
| (n + 1) := a₁ + n * d

-- Define the b_n sequence based on a_n
noncomputable def b_seq (a_seq : ℕ → ℕ) : ℕ → ℚ
| n := a_seq n - 2^n - (1 / 2)

-- Define the sum of the first n terms of sequence b_n
noncomputable def sum_b_seq (b_seq : ℕ → ℚ) : ℕ → ℚ
| 0       := 0
| (n + 1) := sum_b_seq n + b_seq (n + 1)

-- Define the problem statement in Lean
theorem math_problem :
  let a₁ := 2
  let a := arithmetic_seq 3
  let b := b_seq a
  ∀ (n : ℕ), 
    (a n = 3 * n - 1) ∧ 
    (sum_b_seq b n = (3 / 2) * n^2 - 2^(n + 1) + 2) :=
by 
  sorry

end math_problem_l420_420869


namespace basketball_starting_lineup_l420_420352

theorem basketball_starting_lineup (total_players triplets twins remaining_players lineup players_chosen : ℕ)
  (h_total : total_players = 18)
  (h_triplets : triplets = 3)
  (h_twins : twins = 2)
  (h_lineup : lineup = 8)
  (h_players_chosen : players_chosen = triplets + twins)  -- Calculate chosen players (5).
  (h_remaining_players : remaining_players = total_players - players_chosen)  -- Remaining players (13).
  (additional_players : ℕ) (h_additional : additional_players = lineup - players_chosen) -- Additional players to select (3).
  : choose remaining_players additional_players = 286 :=
sorry

end basketball_starting_lineup_l420_420352


namespace length_of_BC_l420_420062

-- Definitions corresponding to the conditions in the problem.
def Circle (O : Point) (r : ℝ) : Set Point := {P | dist O P = r}

-- Setup
variables {O A B C D : Point}           -- Points on the circle
variable {r : ℝ}                        -- Radius of circle
variable [metric_space Point]
variable [euclidean_space ℝ Point]      -- Assuming the points lie in a Euclidean space

-- Conditions
variables (h_circle : ∀ (P ∈ Circle O r), dist O P = r)  -- O is the center of the circle
variables (h_diameter : dist A D = 2 * r)       -- AD is a diameter
variables (h_chord : A ∈ Circle O r)            -- A, B, and C on the circle
variables (h_chord_2 : B ∈ Circle O r)
variables (h_chord_3 : C ∈ Circle O r)
variables (h_BO : dist O B = 7)                 -- BO = 7
variables (h_angle_ABO : angle A B O = 90)      -- ∠ABO = 90°
variables (h_arc_CD : Arc CD = 90)             -- arc CD = 90°

-- Conclusion
theorem length_of_BC : dist B C = 7 :=
by sorry

end length_of_BC_l420_420062


namespace ratio_of_juniors_to_freshmen_l420_420924

variables (f j : ℕ) 

theorem ratio_of_juniors_to_freshmen (h1 : (1/4 : ℚ) * f = (1/2 : ℚ) * j) :
  j = f / 2 :=
by
  sorry

end ratio_of_juniors_to_freshmen_l420_420924


namespace average_velocity_of_particle_l420_420735

theorem average_velocity_of_particle (t : ℝ) (s : ℝ → ℝ) (h_s : ∀ t, s t = t^2 + 1) :
  (s 2 - s 1) / (2 - 1) = 3 :=
by {
  sorry
}

end average_velocity_of_particle_l420_420735


namespace original_side_length_l420_420378

theorem original_side_length (x : ℝ) 
  (h1 : (x - 4) * (x - 3) = 120) : x = 12 :=
sorry

end original_side_length_l420_420378


namespace loaves_of_bread_l420_420421

variable (B : ℕ) -- Number of loaves of bread Erik bought
variable (total_money : ℕ := 86) -- Money given to Erik
variable (money_left : ℕ := 59) -- Money left after purchase
variable (cost_bread : ℕ := 3) -- Cost of each loaf of bread
variable (cost_oj : ℕ := 6) -- Cost of each carton of orange juice
variable (num_oj : ℕ := 3) -- Number of cartons of orange juice bought

theorem loaves_of_bread (h1 : total_money - money_left = num_oj * cost_oj + B * cost_bread) : B = 3 := 
by sorry

end loaves_of_bread_l420_420421


namespace pyramid_total_area_l420_420291

noncomputable def pyramid_base_edge : ℝ := 8
noncomputable def pyramid_lateral_edge : ℝ := 7

theorem pyramid_total_area 
  (base_edge : ℝ)
  (lateral_edge : ℝ)
  (h_base_edge : base_edge = pyramid_base_edge)
  (h_lateral_edge : lateral_edge = pyramid_lateral_edge) 
: 4 * (1 / 2 * base_edge * real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 16 * real.sqrt 33 := 
sorry

end pyramid_total_area_l420_420291


namespace sum_of_squares_minus_product_equals_1949_l420_420455

def sequence_a : ℕ → ℕ
| 0     := 0
| 1     := 1
| 2     := 2
| 3     := 3
| 4     := 4
| 5     := 5
| (n+6) := (sequence_a 1) * (sequence_a 2) * (sequence_a 3) * (sequence_a 4) * (sequence_a 5) * 
           (sequence_a (n+5)) - 1

def sum_of_squares_up_to_2019 : ℕ := (Finset.range 2020).sum (λ i, (sequence_a i) * (sequence_a i))

def product_up_to_2019 : ℕ := (Finset.range 2020).prod sequence_a

theorem sum_of_squares_minus_product_equals_1949 : sum_of_squares_up_to_2019 - product_up_to_2019 = 1949 :=
by {
  sorry
}

end sum_of_squares_minus_product_equals_1949_l420_420455


namespace problem_statement_l420_420866

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 1 then 2 - x else 2 - (x % 2)

theorem problem_statement : 
  (∀ x : ℝ, f (-x) = f x) →
  (∀ x : ℝ, f (x + 1) + f x = 3) →
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = 2 - x) →
  f (-2007.5) = 1.5 :=
by sorry

end problem_statement_l420_420866


namespace total_area_of_triangular_faces_l420_420202

noncomputable def pyramid_base_edge : ℝ := 8
noncomputable def pyramid_lateral_edge : ℝ := 7

structure Pyramid where
  base_edge : ℝ
  lateral_edge : ℝ

def myPyramid : Pyramid :=
{ base_edge := pyramid_base_edge,
  lateral_edge := pyramid_lateral_edge }

theorem total_area_of_triangular_faces :
  4 * (1 / 2) * myPyramid.base_edge * (Real.sqrt (myPyramid.lateral_edge^2 - (myPyramid.base_edge / 2)^2)) = 16 * Real.sqrt 33 :=
by
  sorry

end total_area_of_triangular_faces_l420_420202


namespace count_solutions_ffx_eq_6_l420_420984

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 1 then -x + 4 else 3 * x - 7

theorem count_solutions_ffx_eq_6 : 
  set_of (λ x, f (f x) = 6).finite.to_finset.card = 3 := 
sorry

end count_solutions_ffx_eq_6_l420_420984


namespace Sohan_work_time_l420_420443

theorem Sohan_work_time (G R S : ℚ) (h1 : G + R + S = 1/16) (h2 : G + R = 1/24) : S = 1/48 :=
by
  sorry

end Sohan_work_time_l420_420443


namespace shaded_region_area_l420_420784

noncomputable def area_of_shaded_region 
  (P Q R S T G H : Type*) 
  [MetricSpace P] [MetricSpace Q] [MetricSpace R] 
  [MetricSpace S] [MetricSpace T] [MetricSpace G] [MetricSpace H]
  (radius : ℝ) 
  (PR_length : ℝ) 
  (GH_length : ℝ) 
  (midpoint_R : ∀ P Q, P / 2 + Q / 2 = R)
  (tangent_RS_RT : ∀ P Q, ∃ S T, (Segment RS ∈ TangentCircle P) ∧ (Segment RT ∈ TangentCircle Q))
  (common_tangent : ∀ P Q, ∃ G H, Segment GH ∈ TangentCircle P ∧ Segment GH ∈ TangentCircle Q) :
  Real :=
  36 * Real.sqrt 2 - 9 - 9 * Real.pi / 4

theorem shaded_region_area (P Q R S T G H : Type*) 
  [MetricSpace P] [MetricSpace Q] [MetricSpace R] 
  [MetricSpace S] [MetricSpace T] [MetricSpace G] [MetricSpace H]
  (radius : ℝ) 
  (PR_length : ℝ) 
  (GH_length : ℝ) 
  (midpoint_R : ∀ P Q, P / 2 + Q / 2 = R)
  (tangent_RS_RT : ∀ P Q, ∃ S T, (Segment RS ∈ TangentCircle P) ∧ (Segment RT ∈ TangentCircle Q))
  (common_tangent : ∀ P Q, ∃ G H, Segment GH ∈ TangentCircle P ∧ Segment GH ∈ TangentCircle Q) :
  (area_of_shaded_region P Q R S T G H radius PR_length GH_length midpoint_R tangent_RS_RT common_tangent) 
    = 36 * Real.sqrt 2 - 9 - 9 * Real.pi / 4 :=
sorry

end shaded_region_area_l420_420784


namespace sin_square_cos_identity_l420_420707

theorem sin_square_cos_identity (A B C : ℝ) : 
  (sin A)^2 + (sin B)^2 + (sin C)^2 = 2 + 2 * (cos A) * (cos B) * (cos C) := 
by 
  sorry  -- Proof is omitted, as only the statement is required.

end sin_square_cos_identity_l420_420707


namespace ratio_of_sides_JAMES_l420_420573

noncomputable def AM := 1
noncomputable def SJ := 1
noncomputable def JA := 2

axiom JAMES_pentagon (AM SJ JA : ℝ)
  (h1 : AM = SJ)
  (h2 : ∀ (a b c d e : ℝ), ∠a = 90 ∧ ∠b = 90 ∧ ∠c = 90)
  (h3 : ∠ M = ∠ S) :
  (AM = 1 / (4 * JA)) 

theorem ratio_of_sides_JAMES :
  AM / JA = 1 / 4 :=
  sorry

end ratio_of_sides_JAMES_l420_420573


namespace part_I_part_II_part_III_l420_420474

noncomputable def f (a : ℝ) (x : ℝ) := a * x^2 + Real.log (x + 1)
noncomputable def g (a : ℝ) (b : ℝ) (x : ℝ) := (deriv (f a) (x - 1)) + b * (f a (x - 1)) - a * b * (x - 1)^2 + 2 * a
noncomputable def h (a : ℝ) (c : ℝ) (x : ℝ) := x^4 + ((f a x) - Real.log (x + 1)) * (x + 1/x) + c * x^2 + deriv (f a) 0

theorem part_I (a : ℝ) (c : ℝ) : (∃ x ∈ (0, ∞), h a c x = 0) → a^2 + c^2 ≥ 4/5 :=
by
  sorry

theorem part_II : (∀ x ∈ set.Icc (1 : ℝ) Real.exp, 2/Real.exp ≤ g (1/2) b x ∧ g (1/2) b x ≤ 2 * Real.exp) →
  (b ∈ set.Icc (1/Real.exp - Real.exp) (Real.exp - 1/Real.exp)) :=
by
  sorry

theorem part_III : (∀ x ∈ set.Ici (0 : ℝ), f a x ≤ x) → (a ∈ set.Iic 0) :=
by
  sorry

end part_I_part_II_part_III_l420_420474


namespace minimize_f_l420_420430

noncomputable def f (x : ℝ) : ℝ := 3 * real.sqrt x + 2 / x^2

theorem minimize_f : ∀ x > 0, f x ≥ 5 :=
  sorry

example : f 1 = 5 :=
by
  rw [f]
  norm_num

end minimize_f_l420_420430


namespace parabola_x_intercepts_count_l420_420493

theorem parabola_x_intercepts_count :
  let a := -3
  let b := 4
  let c := -1
  let discriminant := b ^ 2 - 4 * a * c
  discriminant ≥ 0 →
  let num_roots := if discriminant > 0 then 2 else if discriminant = 0 then 1 else 0
  num_roots = 2 := 
by {
  sorry
}

end parabola_x_intercepts_count_l420_420493


namespace prove_constants_and_inequality_l420_420480

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * real.log x - x + (1 / x)

noncomputable def g (b : ℝ) (x : ℝ) : ℝ :=
  x^2 + x - b

noncomputable def h (a b : ℝ) (x : ℝ) : ℝ :=
  f a x / g b x

theorem prove_constants_and_inequality :
  (∃ P : ℝ × ℝ, P = (1, 0) ∧ 
    (∀ x : ℝ, f 2 x = 2 * real.log x - x + (1 / x)) ∧ 
    (∀ x : ℝ, g 2 x = x^2 + x - 2)) ∧ 
  (∀ x : ℝ, x > 0 → x ≠ 1 → h 2 2 x < 0) :=
begin
  sorry
end

end prove_constants_and_inequality_l420_420480


namespace total_area_of_triangular_faces_l420_420205

noncomputable def pyramid_base_edge : ℝ := 8
noncomputable def pyramid_lateral_edge : ℝ := 7

structure Pyramid where
  base_edge : ℝ
  lateral_edge : ℝ

def myPyramid : Pyramid :=
{ base_edge := pyramid_base_edge,
  lateral_edge := pyramid_lateral_edge }

theorem total_area_of_triangular_faces :
  4 * (1 / 2) * myPyramid.base_edge * (Real.sqrt (myPyramid.lateral_edge^2 - (myPyramid.base_edge / 2)^2)) = 16 * Real.sqrt 33 :=
by
  sorry

end total_area_of_triangular_faces_l420_420205


namespace average_increase_l420_420353

theorem average_increase (runs_17th_inning : ℕ) (final_average : ℕ) (initial_innings : ℕ) (initial_runs : ℕ) :
  (runs_17th_inning = 87) → (final_average = 23) → (initial_innings = 16) → (initial_runs = initial_innings * (initial_runs / initial_innings)) →
  let average_increase := final_average - (initial_runs / initial_innings) in
  average_increase = 4 :=
by {
  intros h1 h2 h3 h4,
  have hnine_16A : initial_innings * (initial_runs / initial_innings) = initial_runs,
  { exact h4, },
  have h5 : initial_runs / initial_innings = 19, -- derived change from given data
  { sorry, }, -- detailed algebraic steps can be verified here, ensuring 16 * 19 + 87 = 391.
  have h6 : final_average - (initial_runs / initial_innings) = 4,
  { sorry, }, -- ensure the average increase calculation matches
  exact eq.trans (by rw [h5, h6]) rfl,
}

end average_increase_l420_420353


namespace trigonometric_identity_proof_l420_420412

theorem trigonometric_identity_proof :
  sin (130 * Real.pi / 180) * cos (10 * Real.pi / 180) + sin (40 * Real.pi / 180) * cos (10 * Real.pi / 180) = (Real.sqrt 3) / 2 := by
  sorry

end trigonometric_identity_proof_l420_420412


namespace minimum_value_a_l420_420023

noncomputable def f (x a : ℝ) : ℝ := Real.exp x * (x^3 - 3 * x + 3) - a * Real.exp x - x

def g (x : ℝ) : ℝ := x^3 - 3 * x + 3 - x / Real.exp x

def g' (x : ℝ) : ℝ := 3 * x^2 - 3 + (x - 1) / Real.exp x

theorem minimum_value_a :
  (∃ (a : ℝ), (∀ (x : ℝ), x ≥ -2 → f x a ≤ 0)) ↔ (∃ a, a = 1 - (1 / Real.exp 1)) :=
sorry

end minimum_value_a_l420_420023


namespace trigonometric_identity_l420_420011

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 1) : 
  1 - 2 * Real.sin α * Real.cos α - 3 * (Real.cos α)^2 = -3 / 2 :=
sorry

end trigonometric_identity_l420_420011


namespace tangent_line_problem_l420_420917

theorem tangent_line_problem 
  (x1 x2 : ℝ)
  (h1 : (1 / x1) = Real.exp x2)
  (h2 : Real.log x1 - 1 = Real.exp x2 * (1 - x2)) :
  (2 / (x1 - 1) + x2 = -1) :=
by 
  sorry

end tangent_line_problem_l420_420917


namespace alternating_sums_sum_eq_5120_l420_420894

-- Define the set {1, 2, 3, ..., 10}
def S : Finset ℕ := Finset.range 10

-- Define the function to compute an alternating sum of a non-empty subset
def alternatingSum (s : Finset ℕ) : ℕ :=
  if h : s.card > 0 then
    s.sort (≥) |>.zipWith (λ (i, a) => if i % 2 = 0 then a else -a) (Finset.range s.card)
    |> List.sum
  else 0

-- Problem statement
theorem alternating_sums_sum_eq_5120 : 
  let nonEmptySubsets := S.powerset.filter (λ s => s.card > 0)
  ∑ s in nonEmptySubsets, alternatingSum s = 5120 := 
by
  sorry

end alternating_sums_sum_eq_5120_l420_420894


namespace cube_surface_area_l420_420179

noncomputable def distance (P Q : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2 + (Q.3 - P.3) ^ 2)

def A := (5, 9, 6)
def B := (5, 14, 6)
def C := (5, 14, 11)

theorem cube_surface_area :
  distance A B = 5 ∧ distance B C = 5 ∧ distance A C = 5 * real.sqrt 2 →
  6 * 5^2 = 150 :=
by
  intros h
  sorry

end cube_surface_area_l420_420179


namespace hiker_final_distance_l420_420732

theorem hiker_final_distance :
  let east := 24
  let north := 7
  let west := 15
  let south := 5
  let net_east := east - west
  let net_north := north - south
  net_east = 9 ∧ net_north = 2 →
  Real.sqrt ((net_east)^2 + (net_north)^2) = Real.sqrt 85 :=
by
  intros
  sorry

end hiker_final_distance_l420_420732


namespace triangle_area_calculation_l420_420099

noncomputable def isosceles_triangle_conditions :=
  let BO := 4 * Real.sqrt 2
  let BH := 2 * Real.sqrt 2
  let HO := BH
  let AB := 4 * Real.sqrt 2
  let BC := AB
  let ∠BAC := Real.pi / 6
  (BO, BH, HO, AB, BC, ∠BAC)

theorem triangle_area_calculation :
  ∀ (O A B C H D : Type) 
  [center_of_circumscribed_circle O (is_isosceles_triangle A B C)]
  (h1 : (distance B O) = 4 * Real.sqrt 2)
  (h2 : (height_from_vertex B to_base A C) = 2 * Real.sqrt 2)
  (h3 : (height_from_vertex B to_base A C) = (height_from_foot H to_base O)) 
  (h4 : is_equilateral_triangle A O B) 
  (h5 : (distance A B) = (distance B C))
  (h6 : (angle BAC) = Real.pi / 6)
  (h7 : (angle BDC) = Real.pi / 6)
  (DB : Type) 
  (x := let DC := creation_of_side hypocenter_appropriate DB in DC)
  (NEED_ROOTS : x = (4 * Real.sqrt 3) - 4 ∨ x = (4 * Real.sqrt 3) + 4) :
  let areas :=
    [ 8 * (Real.sqrt 3 - 1),
      8 * (Real.sqrt 3 + 1) ]
  in ∃ s ∈ areas, s = area of Triangle BDV
  := sorry

end triangle_area_calculation_l420_420099


namespace no_solution_system_eqs_l420_420577

theorem no_solution_system_eqs (x y : ℝ) :
  ¬ (x^3 + x + y + 1 = 0 ∧ y * x^2 + x + y = 0 ∧ y^2 + y - x^2 + 1 = 0) := 
by
  intro h
  cases h with h1 h_rest
  cases h_rest with h2 h3
  sorry

end no_solution_system_eqs_l420_420577


namespace max_area_difference_l420_420674

theorem max_area_difference (l w l' w' : ℕ) 
  (h1 : 2 * l + 2 * w = 160) 
  (h2 : 2 * l' + 2 * w' = 160) : 
  abs ((l * w) - (l' * w')) ≤ 1600 := 
by 
  sorry

end max_area_difference_l420_420674


namespace more_karabases_than_barabases_l420_420941

theorem more_karabases_than_barabases 
    (K B : ℕ) 
    (h1 : K * 9 = B * 10) : 
    K > B :=
by
  have h2 : K = (10 * B) / 9 := by sorry
  have h3 : 10 * B % 9 = 0 := by sorry
  have h4 : (10 * B) / 9 > B := by sorry
  exact h4


end more_karabases_than_barabases_l420_420941


namespace temperature_on_friday_l420_420605

-- Define the temperatures on different days
variables (T W Th F : ℝ)

-- Define the conditions
def condition1 : Prop := (T + W + Th) / 3 = 32
def condition2 : Prop := (W + Th + F) / 3 = 34
def condition3 : Prop := T = 38

-- State the theorem to prove the temperature on Friday
theorem temperature_on_friday (h1 : condition1 T W Th) (h2 : condition2 W Th F) (h3 : condition3 T) : F = 44 :=
  sorry

end temperature_on_friday_l420_420605


namespace eval_at_3_l420_420807

theorem eval_at_3 : (3^3)^(3^3) = 27^27 :=
by sorry

end eval_at_3_l420_420807


namespace card_sequence_probability_l420_420178

-- Let's define the conditions
def deck_size : ℕ := 52
def hearts_count : ℕ := 13
def spades_count : ℕ := 13
def clubs_count : ℕ := 13

-- Define the probability of the desired sequence of cards being drawn.
def desired_probability : ℚ := (13 : ℚ) / deck_size * (13 : ℚ) / (deck_size - 1) * (13 : ℚ) / (deck_size - 2)

-- The statement that needs to be proved.
theorem card_sequence_probability :
  desired_probability = 2197 / 132600 :=
by
  sorry

end card_sequence_probability_l420_420178


namespace sum_of_parallelogram_sides_l420_420053

-- Definitions of the given conditions.
def length_one_side : ℕ := 10
def length_other_side : ℕ := 7

-- Theorem stating the sum of the lengths of the four sides of the parallelogram.
theorem sum_of_parallelogram_sides : 
    (length_one_side + length_one_side + length_other_side + length_other_side) = 34 :=
by
    sorry

end sum_of_parallelogram_sides_l420_420053


namespace count_valid_pins_l420_420568

namespace PINProblem

-- Define a PIN as a tuple of four integers
structure PIN where
  d1 d2 d3 d4 : ℕ
  no_repeats : d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4
  within_set : d1 ∈ { 0, 1, 2, ..., 9 } ∧ d2 ∈ { 0, 1, 2, ..., 9 } ∧ d3 ∈ { 0, 1, 2, ..., 9 } ∧ d4 ∈ { 0, 1, 2, ..., 9 }

-- The fixed PIN 4023
def myPIN : PIN := {
  d1 := 4, d2 := 0, d3 := 2, d4 := 3,
  no_repeats := by {
    split; repeat { decide }
  },
  within_set := by {
    split; repeat { decide }
  }
}

-- defining conditions in Lean
def isValidPIN (pin : PIN) : Prop :=
  pin ≠ myPIN ∧
  (∀ i j k l, (i, j, k, l) ∈ { 
    (pin.d1, pin.d2, pin.d3, pin.d4),
    (pin.d1, pin.d2, pin.d4, pin.d3),
    (pin.d1, pin.d3, pin.d2, pin.d4),
    (pin.d1, pin.d3, pin.d4, pin.d2),
    (pin.d1, pin.d4, pin.d2, pin.d3),
    (pin.d1, pin.d4, pin.d3, pin.d2),
    (pin.d2, pin.d1, pin.d3, pin.d4),
    (pin.d2, pin.d1, pin.d4, pin.d3),
    (pin.d2, pin.d3, pin.d1, pin.d4),
    (pin.d2, pin.d3, pin.d4, pin.d1),
    (pin.d2, pin.d4, pin.d1, pin.d3),
    (pin.d2, pin.d4, pin.d3, pin.d1),
    (pin.d3, pin.d1, pin.d2, pin.d4),
    (pin.d3, pin.d1, pin.d4, pin.d2),
    (pin.d3, pin.d2, pin.d1, pin.d4),
    (pin.d3, pin.d2, pin.d4, pin.d1),
    (pin.d3, pin.d4, pin.d1, pin.d2),
    (pin.d3, pin.d4, pin.d2, pin.d1),
    (pin.d4, pin.d1, pin.d2, pin.d3),
    (pin.d4, pin.d1, pin.d3, pin.d2),
    (pin.d4, pin.d2, pin.d1, pin.d3),
    (pin.d4, pin.d2, pin.d3, pin.d1),
    (pin.d4, pin.d3, pin.d1, pin.d2),
    (pin.d4, pin.d3, pin.d2, pin.d1)
    } ∨ 
  (∀ x ∈ { pin.d1, pin.d2, pin.d3, pin.d4 }, x ≠ 4 ∧ x ≠ 0 ∧ x ≠ 2 ∧ x ≠ 3)) ∧
  (∀ x y k l, x ≠ 4 ∨ y ≠ 0 ∨ k ≠ 2 ∨ l ≠ 3)

-- Proving the final count of valid PINs
theorem count_valid_pins : ∃ n, n = 4997 :=
  sorry

end PINProblem

end count_valid_pins_l420_420568


namespace fresh_grapes_water_percentage_l420_420835

-- Variables
variable (P : ℝ) -- percentage of water in fresh grapes

-- Constants and Conditions
def fresh_grapes_weight := 30 -- weight of fresh grapes in kg
def dried_grapes_weight := 15 -- weight of dried grapes in kg
def water_percentage_in_dried_grapes := 0.20 -- 20% water in dried grapes by weight
def non_water_fraction_in_dried_grapes := 0.80 -- 80% non-water in dried grapes by weight

-- Fresh Grapes non-water content calculation
def non_water_in_fresh_grapes := (100 - P) / 100 * fresh_grapes_weight

-- Dried Grapes non-water content
def non_water_in_dried_grapes := dried_grapes_weight * non_water_fraction_in_dried_grapes

-- Proof Statement
theorem fresh_grapes_water_percentage (h : non_water_in_fresh_grapes P = non_water_in_dried_grapes) : P = 60 := by
  sorry

end fresh_grapes_water_percentage_l420_420835


namespace total_area_of_pyramid_faces_l420_420269

-- Define the basic parameters of the pyramid
def base_edges := 8
def lateral_edges := 7

-- Define the Pythagorean theorem components
def altitude_squared := lateral_edges^2 - (base_edges / 2)^2
def altitude := real.sqrt altitude_squared

-- Define the area of one triangular face using half-base and altitude
def one_face_area := (1 / 2) * base_edges * altitude

-- Define the total area of the four triangular faces
def total_area_of_faces := 4 * one_face_area

-- Statement to prove
theorem total_area_of_pyramid_faces : total_area_of_faces = 16 * real.sqrt 33 :=
by
  -- Necessary calculations are assured correct by previous definitions and attributes
  sorry

end total_area_of_pyramid_faces_l420_420269


namespace parabola_focus_directrix_distance_l420_420163

theorem parabola_focus_directrix_distance :
  ∀ {x y : ℝ}, y^2 = (1/4) * x → dist (1/16, 0) (-1/16, 0) = 1/8 := by
sorry

end parabola_focus_directrix_distance_l420_420163


namespace inequality_proof_l420_420715

theorem inequality_proof (a b x y z : ℝ) (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x / (a * y + b * z)) + (y / (a * z + b * x)) + (z / (a * x + b * y)) ≥ (3 / (a + b)) :=
by
  sorry

end inequality_proof_l420_420715


namespace pyramid_area_l420_420300

noncomputable def base_edge : ℝ := 8
noncomputable def lateral_edge : ℝ := 7
noncomputable def height := real.sqrt (lateral_edge^2 - (base_edge / 2)^2)
noncomputable def one_triangle_area := (1 / 2) * base_edge * height
noncomputable def total_area := 4 * one_triangle_area

theorem pyramid_area : total_area = 16 * real.sqrt 33 := by
  sorry

end pyramid_area_l420_420300


namespace stool_height_l420_420763

noncomputable def proof_problem : Prop :=
  ∃ (h : ℕ), 
    let ceiling_height := 300 in
    let bulb_below_ceiling := 15 in
    let alice_height := 160 in
    let alice_reach := 50 in
    let book_height := 5 in
    ceiling_height - bulb_below_ceiling = 285 ∧
    (alice_height + alice_reach) + book_height + h = 285 ∧
    h = 70

theorem stool_height : proof_problem :=
by exact ⟨70, by simp [proof_problem]⟩

end stool_height_l420_420763


namespace exists_perpendicular_tanned_vector_l420_420721

noncomputable theory

def isTanned (v : ℤ × ℤ × ℤ) : Prop := v ≠ (0, 0, 0)

def length (v : ℤ × ℤ × ℤ) : ℝ := real.sqrt (v.1^2 + v.2^2 + v.3^2)

def dotProduct (v w : ℤ × ℤ × ℤ) : ℤ := v.1 * w.1 + v.2 * w.2 + v.3 * w.3

theorem exists_perpendicular_tanned_vector (v : ℤ × ℤ × ℤ) 
  (hv1 : isTanned v) 
  (hv2 : length v ≤ 2021) : 
  ∃ w : ℤ × ℤ × ℤ, isTanned w ∧ length w ≤ 100 ∧ dotProduct v w = 0 :=
by
  sorry

end exists_perpendicular_tanned_vector_l420_420721


namespace infinite_pairs_exists_l420_420828

noncomputable def distance_to_closest_integer (x : ℝ) : ℝ :=
  abs (x - round x)

theorem infinite_pairs_exists
  (x : ℕ → ℝ)
  (hx : ∀ n, 0 ≤ x n ∧ x n < 1)
  (ε : ℝ)
  (hε : ε > 0) :
  ∃ (N : ℕ → ℕ × ℕ), ∀ k : ℕ, let ⟨n, m⟩ := N k in n ≠ m ∧ distance_to_closest_integer (x n - x m) < min ε (1 / (2 * abs (n - m))) :=
sorry

end infinite_pairs_exists_l420_420828


namespace pyramid_four_triangular_faces_area_l420_420256

theorem pyramid_four_triangular_faces_area 
  (base_edge : ℝ) (lateral_edge : ℝ) 
  (h_base : base_edge = 8)
  (h_lateral : lateral_edge = 7) :
  let h := Real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)
  let triangle_area := (1 / 2) * base_edge * h
  let total_area := 4 * triangle_area
  total_area = 16 * Real.sqrt 33 :=
by
  -- Definitions to introduce local values
  let half_base := base_edge / 2
  let h := Real.sqrt (lateral_edge ^ 2 - half_base ^ 2)
  let triangle_area := (1 / 2) * base_edge * h
  let total_area := 4 * triangle_area
  -- Assertion to compare calculated total area with given correct answer
  have h_eq : h = Real.sqrt 33 := by sorry
  have triangle_area_eq : triangle_area = 4 * Real.sqrt 33 := by sorry
  have total_area_eq : total_area = 16 * Real.sqrt 33 := by sorry
  exact total_area_eq

end pyramid_four_triangular_faces_area_l420_420256


namespace total_area_of_pyramid_faces_l420_420265

-- Define the basic parameters of the pyramid
def base_edges := 8
def lateral_edges := 7

-- Define the Pythagorean theorem components
def altitude_squared := lateral_edges^2 - (base_edges / 2)^2
def altitude := real.sqrt altitude_squared

-- Define the area of one triangular face using half-base and altitude
def one_face_area := (1 / 2) * base_edges * altitude

-- Define the total area of the four triangular faces
def total_area_of_faces := 4 * one_face_area

-- Statement to prove
theorem total_area_of_pyramid_faces : total_area_of_faces = 16 * real.sqrt 33 :=
by
  -- Necessary calculations are assured correct by previous definitions and attributes
  sorry

end total_area_of_pyramid_faces_l420_420265


namespace angle_C_is_30_degrees_ab_range_when_c_is_1_l420_420457

theorem angle_C_is_30_degrees (A B C : ℝ) (a b c : ℝ) 
  (h_triangle_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2)
  (h_opposite_sides : A + B + C = π)
  (h_tan_C : tan C = ab / (a^2 + b^2 - c^2)) :
  C = π / 6 :=
sorry

theorem ab_range_when_c_is_1 (A B C : ℝ) (a b : ℝ) 
  (h_triangle_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2)
  (h_opposite_sides : A + B + C = π)
  (h_sine_rule : a / sin A = b / sin B = c / sin C)
  (h_c1 : c = 1) :
  2 * sqrt 3 < ab ∧ ab ≤ 2 + sqrt 3 :=
sorry

end angle_C_is_30_degrees_ab_range_when_c_is_1_l420_420457


namespace units_digit_37_pow_37_l420_420696

theorem units_digit_37_pow_37 : (37 ^ 37) % 10 = 7 := by
  -- The proof is omitted as per instructions.
  sorry

end units_digit_37_pow_37_l420_420696


namespace residue_7_pow_1234_l420_420682

theorem residue_7_pow_1234 : (7^1234) % 13 = 4 := by
  sorry

end residue_7_pow_1234_l420_420682


namespace oblique_projection_parallelogram_l420_420701

theorem oblique_projection_parallelogram (P : Type) [plane_figure P] (parallelogram : P) :
  is_oblique_projection parallelogram → is_parallelogram (intuitive_image parallelogram) :=
sorry

end oblique_projection_parallelogram_l420_420701


namespace inequality_solution_sets_min_value_exists_l420_420028

-- Define the function f
def f (x : ℝ) (m : ℝ) : ℝ := m * x^2 - 2 * x - 3

-- Existence of roots at -1 and n
def roots_of_quadratic (m : ℝ) (n : ℝ) : Prop :=
  m * (-1)^2 - 2 * (-1) - 3 = 0 ∧ m * n^2 - 2 * n - 3 = 0 ∧ m > 0

-- Main problem statements
theorem inequality_solution_sets (a : ℝ) (m : ℝ) (n : ℝ)
  (h1 : roots_of_quadratic m n) (h2 : m = 1) (h3 : n = 3) (h4 : a > 0) :
  if 0 < a ∧ a ≤ 1 then 
    ∀ x : ℝ, x > 2 / a ∨ x < 2
  else if 1 < a ∧ a < 2 then
    ∀ x : ℝ, x > 2 ∨ x < 2 / a
  else 
    False :=
sorry

theorem min_value_exists (a : ℝ) (m : ℝ)
  (h1 : 0 < a ∧ a < 1) (h2 : m = 1) (h3 : f (a^2) m - 3*a^3 = -5) :
  a = (Real.sqrt 5 - 1) / 2 :=
sorry

end inequality_solution_sets_min_value_exists_l420_420028


namespace det_B_squared_minus_3B_l420_420890

open Matrix

-- Define the matrix B
def B : Matrix (fin 2) (fin 2) ℝ :=
  ![![2, 4], ![3, 2]]

-- State the theorem
theorem det_B_squared_minus_3B : det (B * B - 3 • B) = 88 :=
  sorry

end det_B_squared_minus_3B_l420_420890


namespace pyramid_area_l420_420304

noncomputable def base_edge : ℝ := 8
noncomputable def lateral_edge : ℝ := 7
noncomputable def height := real.sqrt (lateral_edge^2 - (base_edge / 2)^2)
noncomputable def one_triangle_area := (1 / 2) * base_edge * height
noncomputable def total_area := 4 * one_triangle_area

theorem pyramid_area : total_area = 16 * real.sqrt 33 := by
  sorry

end pyramid_area_l420_420304


namespace initial_friends_count_l420_420651

noncomputable def initial_friends (joined_friends new_lives total_lives : ℕ) : ℕ :=
  let F := (total_lives - new_lives * joined_friends) / new_lives in F

theorem initial_friends_count : 
  initial_friends 2 7 63 = 7 :=
by
  sorry

end initial_friends_count_l420_420651


namespace hexagon_coloring_ways_l420_420623

-- Define the conditions
def hexagon_coloring_conditions (colors : List (List (Option ℕ))) : Prop :=
  ∃ R_pos color, colors.head.head = some R_pos ∧
  colors.head.length = 2 ∧
  colors.tail.head.length = 2 ∧
  colors.size = 3 ∧
  ∀ i j, 
    (colors[i][j] = some 0 → (colors[i][j+1] ≠ some 0 ∧ colors[i+1][j] ≠ some 0)) ∧
    (colors[i][j] = some 1 → (colors[i][j+1] ≠ some 1 ∧ colors[i+1][j] ≠ some 1)) ∧
    (colors[i][j] = some 2 → (colors[i][j+1] ≠ some 2 ∧ colors[i+1][j] ≠ some 2)) 

-- Main theorem
theorem hexagon_coloring_ways : ∃ n : ℕ, n = 2 ∧
  ∀ colors : List (List (Option ℕ)), hexagon_coloring_conditions colors → true :=
by
  sorry

end hexagon_coloring_ways_l420_420623


namespace pirate_flag_minimal_pieces_l420_420043

theorem pirate_flag_minimal_pieces (original_stripes : ℕ) (desired_stripes : ℕ) (cuts_needed : ℕ) : 
  original_stripes = 12 →
  desired_stripes = 10 →
  cuts_needed = 1 →
  ∃ pieces : ℕ, pieces = 2 ∧ 
  (∀ (top_stripes bottom_stripes: ℕ), top_stripes + bottom_stripes = original_stripes → top_stripes = desired_stripes → 
   pieces = 1 + (if bottom_stripes = original_stripes - desired_stripes then 1 else 0)) :=
by intros;
   sorry

end pirate_flag_minimal_pieces_l420_420043


namespace area_of_triangle_given_altitudes_l420_420524

theorem area_of_triangle_given_altitudes 
  (h_a h_b h_c : ℝ) (hapos : 0 < h_a) (hbpos : 0 < h_b) (hcpos : 0 < h_c) : 
  ∃ S : ℝ, S = 1 / sqrt ((1 / h_a + 1 / h_b + 1 / h_c) *
                          (1 / h_a + 1 / h_b - 1 / h_c) *
                          (1 / h_a + 1 / h_c - 1 / h_b) *
                          (1 / h_b + 1 / h_c - 1 / h_a)) :=
sorry

end area_of_triangle_given_altitudes_l420_420524


namespace max_area_difference_160_perimeter_rectangles_l420_420657

theorem max_area_difference_160_perimeter_rectangles : 
  ∃ (l1 w1 l2 w2 : ℕ), (2 * l1 + 2 * w1 = 160) ∧ (2 * l2 + 2 * w2 = 160) ∧ 
  (l1 * w1 - l2 * w2 = 1521) := sorry

end max_area_difference_160_perimeter_rectangles_l420_420657


namespace farthest_point_l420_420704

-- Define the distance function
def distance_from_origin (p : ℚ × ℚ) : ℚ :=
  let (x, y) := p
  in (x^2 + y^2).toReal.sqrt

-- Define the points we need to compare
def p1 : ℚ × ℚ := (2, 5)
def p2 : ℚ × ℚ := (3, 1)
def p3 : ℚ × ℚ := (4, -3)
def p4 : ℚ × ℚ := (7, 0)
def p5 : ℚ × ℚ := (0, -6)

-- State the problem
theorem farthest_point :
  let d1 := distance_from_origin p1
  let d2 := distance_from_origin p2
  let d3 := distance_from_origin p3
  let d4 := distance_from_origin p4
  let d5 := distance_from_origin p5
  in d4 = max (max (max d1 d2) (max d3 d5)) then (7, 0) :=
begin
  let d1 := distance_from_origin p1,
  let d2 := distance_from_origin p2,
  let d3 := distance_from_origin p3,
  let d4 := distance_from_origin p4,
  let d5 := distance_from_origin p5,
  have h1 : d1 = real.sqrt (4 + 25) := by simp [distance_from_origin, p1],
  have h2 : d2 = real.sqrt (9 + 1) := by simp [distance_from_origin, p2],
  have h3 : d3 = real.sqrt (16 + 9) := by simp [distance_from_origin, p3],
  have h4 : d4 = real.sqrt 49 := by simp [distance_from_origin, p4],
  have h5 : d5 = real.sqrt 36 := by simp [distance_from_origin, p5],
  have h_max : d4 = max (max (max d1 d2) (max d3 d5)) := sorry,
  exact h4.symm.trans h_max
end

end farthest_point_l420_420704


namespace volume_of_region_l420_420825

theorem volume_of_region :
  (∃ (V : ℝ), V = 62.5 ∧
    ∀ (x y z : ℝ), |x + y + z| + |x + y - z| ≤ 10 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 →
    V = 62.5) :=
begin
  sorry
end

end volume_of_region_l420_420825


namespace no_six_digit_number_with_digits_0_1_2_3_4_5_is_divisible_by_11_l420_420126

-- Define that a number is composed of the digits 0, 1, 2, 3, 4, 5
def is_six_digit_with_digits (n : ℕ) : Prop :=
  ∃ (a b c d e f : ℕ),
    {a, b, c, d, e, f} = {0, 1, 2, 3, 4, 5} ∧
    n = a * 10^5 + b * 10^4 + c * 10^3 + d * 10^2 + e * 10 + f

-- Define divisibility by 11
def divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

-- Prove that any six-digit number composed of 0, 1, 2, 3, 4, 5 without repetition cannot be divisible by 11
theorem no_six_digit_number_with_digits_0_1_2_3_4_5_is_divisible_by_11 :
  ¬ ∃ (n : ℕ), is_six_digit_with_digits n ∧ divisible_by_11 n :=
by
  sorry

end no_six_digit_number_with_digits_0_1_2_3_4_5_is_divisible_by_11_l420_420126


namespace part_I_part_II_l420_420343

section
-- Definition of f(x) and g(x)
def f (x : ℝ) := |x - 3| - 2
def g (x : ℝ) := 4 - |x + 1|

-- Prove part I
theorem part_I (x : ℝ) : (f x ≥ g x) → (x ≤ -2 ∨ x ≥ 4) :=
begin
  intro h,
  sorry
end

-- Prove part II
theorem part_II (a : ℝ) : (∀ (x : ℝ), f x - g x ≥ a^2 - 3 * a) ↔ (1 ≤ a ∧ a ≤ 2) :=
begin
  split,
  { intro h,
    sorry },
  { intro h,
    sorry }
end

end

end part_I_part_II_l420_420343


namespace det_B_squared_minus_3B_l420_420891

open Matrix

-- Define the matrix B
def B : Matrix (fin 2) (fin 2) ℝ :=
  ![![2, 4], ![3, 2]]

-- State the theorem
theorem det_B_squared_minus_3B : det (B * B - 3 • B) = 88 :=
  sorry

end det_B_squared_minus_3B_l420_420891


namespace part_a_part_b_part_c_l420_420619

-- Define the gcd function from Mathlib
def gcd (a b : ℕ) : ℕ := Nat.gcd a b

-- Define the psi function 
def psi (n : ℕ) : ℕ := ∑ k in Finset.range n.succ, gcd k n

-- (a) Prove that ψ(mn) = ψ(m)ψ(n) for every two relatively prime m, n ∈ ℕ
theorem part_a (m n : ℕ) (h_rel_prime : gcd m n = 1) : psi (m * n) = psi m * psi n := sorry

-- (b) Prove that for each a ∈ ℕ the equation ψ(x) = ax has a solution
theorem part_b (a : ℕ) : ∃ x : ℕ, psi x = a * x := sorry

-- (c) Find all a ∈ ℕ such that the equation ψ(x) = ax has a unique solution
theorem part_c (a : ℕ) : (∀ x y : ℕ, ψ x = a * x → ψ y = a * y → x = y) ↔ ∃ k : ℕ, a = 2^k := sorry

end part_a_part_b_part_c_l420_420619


namespace pyramid_face_area_total_l420_420232

theorem pyramid_face_area_total 
  (base_edge : ℝ) (lateral_edge : ℝ) 
  (h_base_edge : base_edge = 8) 
  (h_lateral_edge : lateral_edge = 7) : 
  4 * (1 / 2 * base_edge * real.sqrt (lateral_edge^2 - (base_edge / 2)^2)) = 16 * real.sqrt 33 :=
by
  sorry

end pyramid_face_area_total_l420_420232


namespace construct_right_angle_quadrilateral_l420_420486

noncomputable def intersection_points (E F G H : Point) : Prop :=
  -- Assumes definitions such as Point, Line, and intersection conditions are already defined
  intersect Line1 (side E F) ∧ intersect Line2 (side F G) ∧
  intersect Line3 (side G H) ∧ intersect Line4 (side H E)

def side_length (P Q : Point) (a : ℝ) : Prop :=
  dist P Q = a

theorem construct_right_angle_quadrilateral
  {E F G H : Point} {a : ℝ} :
  intersection_points E F G H ∧ side_length E F a → 
  ∃ quadrilateral : Quadrilateral, 
    is_right_angled_quadrilateral quadrilateral ∧
    (∃! K, K ∈ quadrilateral.points ∧
           (∀ P ∈ quadrilateral.points, dist E P < dist F P)) :=
sorry

end construct_right_angle_quadrilateral_l420_420486


namespace analects_deductive_reasoning_l420_420154

theorem analects_deductive_reasoning :
  (∀ (P Q R S T U V : Prop), 
    (P → Q) → 
    (Q → R) → 
    (R → S) → 
    (S → T) → 
    (T → U) → 
    ((P → U) ↔ deductive_reasoning)) :=
sorry

end analects_deductive_reasoning_l420_420154


namespace prove_ABdotAEplusADdotAFeqACsq_l420_420554

-- Let us define our setup

variables {A B C D E F : Type} [add_comm_group F]
variables [module ℝ A] [module ℝ B] [module ℝ C]

def parallelogram (A B C D : A) : Prop := 
  ∃ m n : ℝ, A = m • B + n • C ∧ D = n • B + m • C

def perpendicular (P Q R : A) : Prop := 
  ∃ m : ℝ, P = m • Q + R ∧ ∀ x, Q ⟂ R

def problem (A B C D E F : A) [parallelogram A B C D] [perpendicular C E (line AB)] [perpendicular C F (line AD)] :=
  AB • AE + AD • AF = AC • AC

theorem prove_ABdotAEplusADdotAFeqACsq (A B C D E F : Type) [parallelogram A B C D] [perpendicular C E (line AB)] [perpendicular C F (line AD)] : 
  problem A B C D E F :=
by sorry -- Proof is omitted

end prove_ABdotAEplusADdotAFeqACsq_l420_420554


namespace sum_of_coefficients_l420_420555

theorem sum_of_coefficients (s : ℕ → ℝ) (a b c : ℝ) : 
  s 0 = 3 ∧ s 1 = 7 ∧ s 2 = 17 ∧ 
  (∀ k ≥ 2, s (k + 1) = a * s k + b * s (k - 1) + c * s (k - 2)) → 
  a + b + c = 12 := 
by
  sorry

end sum_of_coefficients_l420_420555


namespace area_of_transformed_region_l420_420545

theorem area_of_transformed_region : 
  let T : ℝ := 15
  let A : Matrix (Fin 2) (Fin 2) ℝ := ![![3, 4], ![6, -2]]
  (abs (Matrix.det A) * T = 450) := 
  sorry

end area_of_transformed_region_l420_420545


namespace ratio_of_volumes_l420_420192

noncomputable def volume (r h : ℝ) : ℝ :=
  (1 / 3) * Math.pi * r^2 * h

theorem ratio_of_volumes :
  volume 20 40 / volume 40 20 = 1 / 2 := by
  sorry

end ratio_of_volumes_l420_420192


namespace bound_on_A_l420_420540

variable (n : ℕ)

def is_ntuple (x : Fin n → Fin 3) := True

def valid_set (A : Set (Fin n → Fin 3)) :=
  ∀ x y z ∈ A, x ≠ y → y ≠ z → x ≠ z →
    ∃ i : Fin n, (x i = 0 ∧ y i = 1 ∧ z i = 2) ∨
                 (x i = 0 ∧ y i = 2 ∧ z i = 1) ∨
                 (x i = 1 ∧ y i = 0 ∧ z i = 2) ∨
                 (x i = 1 ∧ y i = 2 ∧ z i = 0) ∨
                 (x i = 2 ∧ y i = 0 ∧ z i = 1) ∨
                 (x i = 2 ∧ y i = 1 ∧ z i = 0)

theorem bound_on_A (A : Set (Fin n → Fin 3)) 
  (hA : valid_set A) :
  ↑(size A) ≤ 2 * (3 / 2) ^ n :=
  sorry

end bound_on_A_l420_420540


namespace problem_statement_l420_420098

theorem problem_statement (r p q : ℝ) (h1 : r > 0) (h2 : p * q ≠ 0) (h3 : p^2 * r > q^2 * r) : p^2 > q^2 := 
sorry

end problem_statement_l420_420098


namespace sharks_count_l420_420991

theorem sharks_count (sharks_percentage : ℝ) (fish_day1 : ℕ) (fish_mult : ℝ) (fish_day2_decay : ℝ) :
  sharks_percentage = 0.2 →
  fish_day1 = 20 →
  fish_mult = 3 →
  fish_day2_decay = 0.25 →
  let fish_day2 := fish_mult * fish_day1 in
  let fish_day3 := fish_day2 - (fish_day2_decay * fish_day2) in
  let sharks_day1 := sharks_percentage * fish_day1 in
  let sharks_day2 := sharks_percentage * fish_day2 in
  let sharks_day3 := sharks_percentage * fish_day3 in
  sharks_day1 + sharks_day2 + sharks_day3 = 25 :=
by
  intros
  sorry

end sharks_count_l420_420991


namespace odds_against_C_winning_l420_420491

theorem odds_against_C_winning (prob_A: ℚ) (prob_B: ℚ) (prob_C: ℚ)
    (odds_A: prob_A = 1 / 5) (odds_B: prob_B = 2 / 5) 
    (total_prob: prob_A + prob_B + prob_C = 1):
    ((1 - prob_C) / prob_C) = 3 / 2 :=
by
  sorry

end odds_against_C_winning_l420_420491


namespace arithmetic_sequence_problem_l420_420939

variable {a : ℕ → ℝ}  -- Variable for the arithmetic sequence indexed by natural numbers

-- Define the common difference and the first term of the arithmetic sequence
def d := a (1) - a (0)
noncomputable def a1 := a (1) - d

-- Condition given in the problem
axiom h_sum : a 4 + a 6 + a 8 + a 10 + a 12 = 120

-- Prove that a₉ - ½a₁₀ = 12
theorem arithmetic_sequence_problem :
  a 9 - (1 / 2) * a 10 = 12 := by
  sorry

end arithmetic_sequence_problem_l420_420939


namespace bridget_in_middle_l420_420419

variable (H E C B : ℕ) -- H, E, C, B represent the scores of Hannah, Ella, Cassie, and Bridget respectively

-- Conditions translated into Lean statements
def condition1 : Prop :=
  C > H ∧ C > E

def condition2 : Prop :=
  B ≥ H ∧ B ≥ E

-- The main theorem that should be proved based on the conditions
theorem bridget_in_middle (Hscore : H) (Escore : E) (Cscore : C) (Bscore : B) :
    (H < B ∧ B < C) ∨ (E < B ∧ B < C) :=
by
  sorry

end bridget_in_middle_l420_420419


namespace sphere_surface_area_l420_420071

-- Define a structure for the sphere and inscribed cylinder
structure SphereCylinder :=
  (R : ℝ)          -- Radius of the sphere
  (r : ℝ)          -- Radius of the base of the cylinder
  (h : ℝ)          -- Height of the cylinder
  (inscribed : h = 2 * real.sqrt (R^2 - r^2))
  (lateral_surface_area : 2 * real.pi * r * h = 16 * real.sqrt 2 * real.pi)

-- Define the problem: proving the surface area of the sphere
theorem sphere_surface_area (sc : SphereCylinder) : 4 * real.pi * (sc.R ^ 2) = 48 * real.pi :=
sorry

end sphere_surface_area_l420_420071


namespace pyramid_total_area_l420_420288

noncomputable def pyramid_base_edge : ℝ := 8
noncomputable def pyramid_lateral_edge : ℝ := 7

theorem pyramid_total_area 
  (base_edge : ℝ)
  (lateral_edge : ℝ)
  (h_base_edge : base_edge = pyramid_base_edge)
  (h_lateral_edge : lateral_edge = pyramid_lateral_edge) 
: 4 * (1 / 2 * base_edge * real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 16 * real.sqrt 33 := 
sorry

end pyramid_total_area_l420_420288


namespace simplify_trig_expr_l420_420141

theorem simplify_trig_expr :
  (tan (10 * π / 180) + tan (30 * π / 180) + tan (50 * π / 180) + tan (70 * π / 180)) / cos (40 * π / 180) = -sin (20 * π / 180) :=
by
  sorry

end simplify_trig_expr_l420_420141


namespace function_value_at_919_l420_420858

noncomputable def f : ℝ → ℝ := sorry -- Define the function f

axiom even_function : ∀ x : ℝ, f(x) = f(-x)
axiom periodic_function : ∀ x : ℝ, f(x + 4) = f(x - 2)
axiom function_on_interval : ∀ x : ℝ, -3 ≤ x ∧ x ≤ 0 → f(x) = 6^(-x)

theorem function_value_at_919 : f 919 = 6 := by
  sorry

end function_value_at_919_l420_420858


namespace f_is_odd_l420_420618

open Real

noncomputable def f (x : ℝ) (n : ℕ) : ℝ :=
  (1 + sin x)^(2 * n) - (1 - sin x)^(2 * n)

theorem f_is_odd (n : ℕ) (h : n > 0) : ∀ x : ℝ, f (-x) n = -f x n :=
by
  intros x
  -- Proof goes here
  sorry

end f_is_odd_l420_420618


namespace find_counterfeit_bag_l420_420647

-- Define the weights and the number of bags
variables {w x : ℝ}
constant eqn_genuine_weight : ∀ a : ℝ, a = w
constant eqn_counterfeit_weight : ∀ b : ℝ, b = w + x
constant scale : ℝ → ℝ → ℝ

-- There are 11 bags, with
def num_bags : ℕ := 11

-- There is one counterfeit bag
def counterfeit_bag_exists : ∃ b : ℕ, b < num_bags ∧ b ≠ 0

-- Two weighings with detailed balance observations
def first_weighing_balance : ℝ :=
  let w1 := 10 * w in
  let w2 := 10 * (w + x) in
  scale w1 w2

def second_weighing_balance : ℝ :=
  let w1 := (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10) * w in
  let w2 := 55 * (w + x) in
  scale w1 w2

-- The ratio of the differences from both weighings identifies the counterfeit bag
def counterfeit_bag_identifiable : Prop :=
  ∀ diff1 : ℝ, ∀ diff2 : ℝ, diff1 = scale (10 * w) (10 * (w + x)) →
  diff2 = scale (55 * w) (55 * (w + x)) →
  (diff2 / diff1 = 5.5 ∧ ∃ i : ℕ, i < 10 ∧ i = diff2 / diff1)

theorem find_counterfeit_bag (w x : ℝ) :
  ∃ b : ℕ, counterfeit_bag_identifiable :=
sorry

end find_counterfeit_bag_l420_420647


namespace equidistant_point_on_Ox_axis_l420_420459

theorem equidistant_point_on_Ox_axis 
  (x1 y1 z1 x2 y2 z2 : ℝ)
  (hne : x2 ≠ x1) :
  ∃ x : ℝ, x = (x2^2 - x1^2 + y2^2 - y1^2 + z2^2 - z1^2) / (2 * (x2 - x1)) :=
begin
  use (x2^2 - x1^2 + y2^2 - y1^2 + z2^2 - z1^2) / (2 * (x2 - x1)),
  sorry
end

end equidistant_point_on_Ox_axis_l420_420459


namespace part1_part2_part3_part4_l420_420877

variables (m : ℝ)

def z (m : ℝ) : ℂ := complex.mk (2 * m ^ 2 + 3 * m - 2) (m ^ 2 + m - 2)

def is_real (z : ℂ) : Prop := im z = 0
def is_imaginary (z : ℂ) : Prop := re z = 0
def is_pure_imaginary (z : ℂ) : Prop := re z = 0 ∧ im z ≠ 0
def is_zero (z : ℂ) : Prop := re z = 0 ∧ im z = 0

theorem part1 : is_real (z m) ↔ m = -2 ∨ m = 1 :=
sorry

theorem part2 : is_imaginary (z m) ↔ m ≠ -2 ∧ m ≠ 1 :=
sorry

theorem part3 : is_pure_imaginary (z m) ↔ m = 1 / 2 :=
sorry

theorem part4 : is_zero (z m) ↔ m = -2 :=
sorry

end part1_part2_part3_part4_l420_420877


namespace largest_number_of_gold_coins_l420_420319

theorem largest_number_of_gold_coins (n : ℕ) :
  (∃ k : ℕ, n = 13 * k + 3 ∧ n < 150) → n ≤ 146 :=
by
  sorry

end largest_number_of_gold_coins_l420_420319


namespace image_center_coordinates_l420_420399

-- Define the point reflecting across the x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Define the point translation by adding some units to the y-coordinate
def translate_y (p : ℝ × ℝ) (dy : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + dy)

-- Define the initial point and translation
def initial_point : ℝ × ℝ := (3, -4)
def translation_units : ℝ := 5

-- Prove the final coordinates of the image of the center of circle Q
theorem image_center_coordinates : translate_y (reflect_x initial_point) translation_units = (3, 9) :=
  sorry

end image_center_coordinates_l420_420399


namespace pyramid_area_l420_420276

theorem pyramid_area (base_edge lateral_edge : ℝ) (H_base_edge : base_edge = 8) (H_lateral_edge : lateral_edge = 7) :
  4 * (1 / 2 * base_edge * real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 16 * real.sqrt 33 :=
by
  sorry

end pyramid_area_l420_420276


namespace zero_point_inequality_l420_420907

variable {f : ℝ → ℝ}
variable {a x : ℝ}

def is_zero_point (f : ℝ → ℝ) (a : ℝ) : Prop :=
  f(a) = 0

noncomputable def g (x : ℝ) : ℝ := sin x / x

theorem zero_point_inequality (h_zero : is_zero_point (λ x, sin x - x * cos x) a) 
  (ha : a ∈ Set.Ioo 0 (2 * Real.pi))
  (hx : x ∈ Set.Ioo 0 (2 * Real.pi)) :
  g x ≥ g a :=
sorry

end zero_point_inequality_l420_420907


namespace determine_coefficients_l420_420806

theorem determine_coefficients (a b c : ℝ) (x y : ℝ) :
  (x = 3/4 ∧ y = 5/8) →
  (a * (x - 1) + 2 * y = 1) →
  (b * |x - 1| + c * y = 3) →
  (a = 1 ∧ b = 2 ∧ c = 4) := 
by 
  intros 
  sorry

end determine_coefficients_l420_420806


namespace fx_le_2x_l420_420987

variable {f : ℝ → ℝ}

-- Conditions
axiom domain_nonnegative : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f(x)
axiom boundary_condition : f 1 = 1
axiom combining_condition : ∀ x₁ x₂ : ℝ, x₁ + x₂ ≤ 1 → f (x₁ + x₂) ≥ f(x₁) + f(x₂)

-- Proof the main theorem
theorem fx_le_2x (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : f(x) ≤ 2 * x := by
  sorry

-- Example showing some function does not satisfy f(x) ≤ 1.9x
example (c : ℝ) (hc : c < 2) : ∃ x, 0 ≤ x ∧ x ≤ 1 ∧ f(x) > c * x := by
  sorry

end fx_le_2x_l420_420987


namespace avg_xy_36_l420_420636

-- Given condition: The average of the numbers 2, 6, 10, x, and y is 18
def avg_condition (x y : ℝ) : Prop :=
  (2 + 6 + 10 + x + y) / 5 = 18

-- Goal: To prove that the average of x and y is 36
theorem avg_xy_36 (x y : ℝ) (h : avg_condition x y) : (x + y) / 2 = 36 :=
by
  sorry

end avg_xy_36_l420_420636


namespace pyramid_four_triangular_faces_area_l420_420253

theorem pyramid_four_triangular_faces_area 
  (base_edge : ℝ) (lateral_edge : ℝ) 
  (h_base : base_edge = 8)
  (h_lateral : lateral_edge = 7) :
  let h := Real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)
  let triangle_area := (1 / 2) * base_edge * h
  let total_area := 4 * triangle_area
  total_area = 16 * Real.sqrt 33 :=
by
  -- Definitions to introduce local values
  let half_base := base_edge / 2
  let h := Real.sqrt (lateral_edge ^ 2 - half_base ^ 2)
  let triangle_area := (1 / 2) * base_edge * h
  let total_area := 4 * triangle_area
  -- Assertion to compare calculated total area with given correct answer
  have h_eq : h = Real.sqrt 33 := by sorry
  have triangle_area_eq : triangle_area = 4 * Real.sqrt 33 := by sorry
  have total_area_eq : total_area = 16 * Real.sqrt 33 := by sorry
  exact total_area_eq

end pyramid_four_triangular_faces_area_l420_420253


namespace pyramid_area_l420_420247

theorem pyramid_area :
  ∀ (a b : ℝ), a = 8 → b = 7 → 4 * (1/2 * a * sqrt (b^2 - (a/2)^2)) = 16 * sqrt 33 :=
by
  intros a b ha hb
  rw [ha, hb]
  have h1 : a / 2 = 4 := by norm_num [ha]
  have h2 : b^2 - (a / 2)^2 = 33 :=
    by
      calc
        b^2 - (a / 2)^2 = 49 - 16 := by norm_num [hb]
        ... = 33 := by norm_num
  rw [h1, h2, sqrt 33, mul_one, mul_one, half_mul, mul_comm (1/2) a, ←mul_assoc, mul_comm 4 4]
  norm_num
  sorry

end pyramid_area_l420_420247


namespace james_winning_strategy_l420_420529

noncomputable def has_winning_strategy (player : String) : Prop :=
  ∃ (strategy : List ℕ → ℕ), ∀ (moves : List ℕ), (moves.length < 25) → 
  let remaining_moves := (Set.toFinset (Finset.range 25) \ moves.toFinset).toList in
  let next_move := strategy moves in
  next_move ∈ remaining_moves

theorem james_winning_strategy :
  let coins : List ℕ := (List.range 1 26) in
  ∀ (moves : List ℕ), moves.length < 25 →
    (∃ (strategy_john : List ℕ → ℕ),
      ∃ (strategy_james : List ℕ → ℕ),
        ∀ moves, has_winning_strategy "James") :=
begin
  sorry
end

end james_winning_strategy_l420_420529


namespace mahdi_plays_tennis_on_friday_l420_420118

-- Definitions for the days of the week
inductive WeekDay
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday
deriving DecidableEq, Repr

-- Definitions for the sports
inductive Sport
| Running | Basketball | Golf | Swimming | Tennis
deriving DecidableEq, Repr

-- Mahdi's schedule
structure Schedule :=
  (activity : WeekDay → Sport)
  (one_sport_per_day : ∀ day1 day2, day1 ≠ day2 → activity day1 ≠ activity day2)
  (runs_three_days : ∃ days, days.card = 3 ∧ ∀ d ∈ days, activity d = Sport.Running)
  (non_consecutive_running : ∀ d1 d2, activity d1 = Sport.Running → activity d2 = Sport.Running → (d1 ≠ d2 ∧ |d1.to_nat - d2.to_nat| > 1))
  (plays_basketball_on_tuesday : activity WeekDay.Tuesday = Sport.Basketball)
  (plays_golf_on_thursday : activity WeekDay.Thursday = Sport.Golf)
  (plays_tennis_but_not_before_swimming : ∀ d1 d2, activity d1 = Sport.Tennis → activity d2 = Sport.Swimming → d1 ≠ d2 - 1)

-- The proposition we need to prove
theorem mahdi_plays_tennis_on_friday :
  ∃ schedule : Schedule, schedule.activity WeekDay.Friday = Sport.Tennis :=
sorry

end mahdi_plays_tennis_on_friday_l420_420118


namespace proof_statements_correctness_l420_420456

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f(x) = -f(-x)

def functional_eq (f : ℝ → ℝ) : Prop :=
∀ x, f(x + 1) = f(x - 1)

noncomputable def correct_statements (f : ℝ → ℝ) :=
{ (1, 3) }

theorem proof_statements_correctness (f : ℝ → ℝ) (h1 : is_odd_function f) (h2 : functional_eq f) :
  correct_statements f = { (1, 3) } :=
by
  sorry

end proof_statements_correctness_l420_420456


namespace total_withdrawal_eq_l420_420586

theorem total_withdrawal_eq 
  (initial_deposit : ℕ) 
  (increase_per_year : ℕ) 
  (years : ℕ) 
  (p : ℝ) 
  (final_years : ℕ) 
  (h1 : initial_deposit = 1000)
  (h2 : increase_per_year = 1000)
  (h3 : years = 6) 
  (h4 : final_years = 7) 
  : 
    ∑ i in finset.range years, (initial_deposit + i * increase_per_year) * (1 + p) ^ (final_years - i) 
    = (1 / p^2) * ((1 + p)^8 - (1 + p) * (1 + 7 * p)) := 
    sorry

end total_withdrawal_eq_l420_420586


namespace pyramid_total_area_l420_420290

noncomputable def pyramid_base_edge : ℝ := 8
noncomputable def pyramid_lateral_edge : ℝ := 7

theorem pyramid_total_area 
  (base_edge : ℝ)
  (lateral_edge : ℝ)
  (h_base_edge : base_edge = pyramid_base_edge)
  (h_lateral_edge : lateral_edge = pyramid_lateral_edge) 
: 4 * (1 / 2 * base_edge * real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 16 * real.sqrt 33 := 
sorry

end pyramid_total_area_l420_420290


namespace find_coefficients_l420_420959

noncomputable def sequence (p q r : ℤ) : ℕ → ℤ
| 0     => 1 -- initial value a_1
| 1     => 1 -- initial value a_2
| (n+2) => p * sequence n.succ + q * sequence n + r

theorem find_coefficients (p q r : ℤ) (h : ∀ n : ℕ, 
  1 ≤ sequence p q r n ∧ sequence p q r n ≤ sequence p q r (n+1) ∧
  (sequence p q r n ^ 2 + sequence p q r (n+1) ^ 2 + sequence p q r n + sequence p q r (n+1) + 1) / (sequence p q r n * sequence p q r (n+1)) ∈ ℕ) :
  (p, q, r) = (5, -1, -1) :=
sorry

end find_coefficients_l420_420959


namespace part1_part2_l420_420985

open Set

noncomputable def A (p q : ℝ) : Set ℝ := {x | 9 ^ x + p * 3 ^ x + q = 0}
noncomputable def B (p q : ℝ) : Set ℝ := {x | q * 9 ^ x + p * 3 ^ x + 1 = 0}

theorem part1 (p q x0 : ℝ) (h : x0 ∈ A p q) : (-x0) ∈ B p q := by
  sorry

theorem part2 : ∃ (p q : ℝ), (A p q ∩ B p q ≠ ∅) ∧ (A p q ∩ (Compl (B p q)) = {1}) ∧ p = -4 ∧ q = 3 := by
  sorry

end part1_part2_l420_420985


namespace pyramid_face_area_total_l420_420234

theorem pyramid_face_area_total 
  (base_edge : ℝ) (lateral_edge : ℝ) 
  (h_base_edge : base_edge = 8) 
  (h_lateral_edge : lateral_edge = 7) : 
  4 * (1 / 2 * base_edge * real.sqrt (lateral_edge^2 - (base_edge / 2)^2)) = 16 * real.sqrt 33 :=
by
  sorry

end pyramid_face_area_total_l420_420234


namespace rationalize_eq_sum_l420_420130

def rationalize_denominator (x y z : ℚ) : ℚ :=
    x * y / z

theorem rationalize_eq_sum :
  ∃ (A B C D E F : ℤ), F > 0 ∧ (∀ p, prime p → ¬ (p^2 ∣ B)) ∧ (∀ p, prime p → ¬ (p^2 ∣ D)) ∧
  gcd (gcd (gcd A C) E) F = 1 ∧
  A * B + C * D + E + F * (B + D) = 51 := 
by
  -- Expression for rationalizing the denominator 
  let expr := rationalize_denominator (7:ℚ) (3 + 2 * Real.sqrt 2 - Real.sqrt 3) (20 + 12 * Real.sqrt 2 - 6 * Real.sqrt 3 - 4 * Real.sqrt 6)
  -- Assigning values
  let A : ℤ := 21
  let B : ℤ := 1
  let C : ℤ := 14
  let D : ℤ := 2
  let E : ℤ := -7
  let F : ℤ := 20
  
  -- Constraints
  have F_pos : F > 0 := by norm_num
  have B_not_prime_square: ∀ p, prime p → ¬ (p^2 ∣ B) := by
    intros p hp
    have p_squared_div_B := is_prime.pow_dvd_iff_le hp
    sorry
  have D_not_prime_square: ∀ p, prime p → ¬ (p^2 ∣ D) := by
    intros p hp
    have p_squared_div_D := is_prime.pow_dvd_iff_le hp
    sorry
  have gcd_condition: gcd (gcd (gcd A C) E) F = 1 := by
    sorry
  
  existsi [A, B, C, D, E, F]
  split
  exact F_pos
  split
  exact B_not_prime_square
  split
  exact D_not_prime_square
  split
  exact gcd_condition
  have expected_sum: 21 + 14 - 7 + 20 + 1 + 2 = 51 := by norm_num
  exact expected_sum

end rationalize_eq_sum_l420_420130


namespace solve_for_x_l420_420587

theorem solve_for_x (x : ℝ) (h : (7 * x + 3) / (x + 5) - 5 / (x + 5) = 2 / (x + 5)) : x = 4 / 7 :=
by
  sorry

end solve_for_x_l420_420587


namespace polynomial_division_l420_420717

variable (a p x : ℝ)

theorem polynomial_division :
  (p^8 * x^4 - 81 * a^12) / (p^6 * x^3 - 3 * a^3 * p^4 * x^2 + 9 * a^6 * p^2 * x - 27 * a^9) = p^2 * x + 3 * a^3 :=
by sorry

end polynomial_division_l420_420717


namespace roots_poly_cond_l420_420965

theorem roots_poly_cond (α β p q γ δ : ℝ) 
  (h1 : α ^ 2 + p * α - 1 = 0) 
  (h2 : β ^ 2 + p * β - 1 = 0) 
  (h3 : γ ^ 2 + q * γ - 1 = 0) 
  (h4 : δ ^ 2 + q * δ - 1 = 0)
  (h5 : γ * δ = -1) :
  (α - γ) * (β - γ) * (α - δ) * (β - δ) = -(p - q) ^ 2 := 
by 
  sorry

end roots_poly_cond_l420_420965


namespace greatest_area_difference_l420_420667

theorem greatest_area_difference (l₁ w₁ l₂ w₂ : ℕ) (h₁ : 2 * l₁ + 2 * w₁ = 160) (h₂ : 2 * l₂ + 2 * w₂ = 160) : 
  abs (l₁ * w₁ - l₂ * w₂) = 1521 :=
sorry

end greatest_area_difference_l420_420667


namespace mathematical_problem_equiv_proof_problem_l420_420853

/-
Given:
  - Point P(0, -2)
  - Points A and B are the left and right vertices of the ellipse E: x^2 / a^2 + y^2 / b^2 = 1 (a > b > 0)
  - Line BP intersects E at point Q
  - Triangle ABP is an isosceles right triangle with ∠APB = 90°
  - ∥PQ∥ = 3/2 ∥QB∥
  - Variable line l passing through point P intersects the ellipse E at points M and N
  - Origin O is outside the circle with diameter MN

Prove:
  1. The equation of the ellipse is ∃ a b: ℝ, a = 2 ∧ b^2 = 1 ∧ E = (x^2 / a^2 + y^2 / b^2 = 1)

  2. The range of the slope k of l is in (-2, -√3/2) ∪ (√3/2, 2)
-/

noncomputable def ellipse_eq : Prop :=
∃ (a b : ℝ), a = 2 ∧ b^2 = 1 ∧ (∀ (x y : ℝ), (x^2 / a^2 + y^2 = 1) ↔ (x / 2)^2 + y^2 = 1)

noncomputable def slope_range : Prop :=
∀ (k : ℝ), (∃ (P Q: ℝ × ℝ), P = (0, -2) ∧ Q = (2, 0) ∧ (k > √(3) / 2 ∧ k < 2) ∨ (k > -2 ∧ k < -√(3) / 2))

theorem mathematical_problem_equiv_proof_problem (k : ℝ) :
  ellipse_eq ∧ slope_range :=
begin
  split,
  { 
    unfold ellipse_eq,
    existsi [2:ℝ, 1:ℝ],
    split,
    {
      norm_num
    },
    {
      norm_num,
      intros x y,
      split; intro h,
      {
        norm_num at *,
        exact h
      },
      {
        norm_num at *,
        exact h
      }
    }
  },
  {
    unfold slope_range,
    intros k _,
    exact sorry,
  },
end

end mathematical_problem_equiv_proof_problem_l420_420853


namespace number_of_tiles_per_row_l420_420595

-- Define the conditions
def area (a : ℝ) : ℝ := a * a
def side_length (area : ℝ) : ℝ := real.sqrt area
def feet_to_inches (feet : ℝ) : ℝ := feet * 12
def tiles_per_row (room_length_inches : ℝ) (tile_size_inches : ℝ) : ℕ := 
  int.to_nat ⟨room_length_inches / tile_size_inches, by sorry⟩

-- Given constants in the problem
def area_of_room : ℝ := 256
def tile_size : ℝ := 8

-- Derived lengths
def length_of_side := side_length area_of_room
def length_of_side_in_inches := feet_to_inches length_of_side

-- The theorem to prove
theorem number_of_tiles_per_row : tiles_per_row length_of_side_in_inches tile_size = 24 :=
sorry

end number_of_tiles_per_row_l420_420595


namespace bee_paths_to_hive_6_correct_l420_420774

noncomputable def num_paths_to_hive_6 : ℕ := 21

theorem bee_paths_to_hive_6_correct
  (start_pos : ℕ)
  (end_pos : ℕ)
  (bee_can_only_crawl : Prop)
  (bee_can_move_right : Prop)
  (bee_can_move_upper_right : Prop)
  (bee_can_move_lower_right : Prop)
  (total_hives : ℕ)
  (start_pos_is_initial : start_pos = 0)
  (end_pos_is_six : end_pos = 6) :
  num_paths_to_hive_6 = 21 :=
by
  sorry

end bee_paths_to_hive_6_correct_l420_420774


namespace N_subset_M_l420_420837
-- Import necessary Lean library

-- Define the sets M and N in Lean
def M : set (ℝ × ℝ) := {p | ∃ x, p = (x, 2 * x + 1)}
def N : set (ℝ × ℝ) := {p | ∃ x, p = (x, -x^2)}

-- The statement to be proved
theorem N_subset_M : N ⊆ M :=
sorry

end N_subset_M_l420_420837


namespace part1_part2_l420_420526

variables {a b c : ℝ}
variables {A B C : ℝ}

noncomputable def cos_B : ℝ := 1 / 2  -- from B = π / 3

-- Given Conditions
axiom condition1 : (c - 2 * a) * cos B + b * cos C = 0
axiom condition2 : a + b + c = 6
axiom condition3 : b = 2

-- Desired Results
theorem part1 : B = π / 3 :=
by
  sorry

noncomputable def area_ABC : ℝ := 1 / 2 * a * c * sin B

theorem part2 : area_ABC = sqrt 3 :=
by
  sorry

end part1_part2_l420_420526


namespace pyramid_total_area_l420_420221

/-- The total area of the four triangular faces of a right, square-based pyramid
whose base edges measure 8 units and lateral edges measure 7 units is 16√33. -/
theorem pyramid_total_area :
  let base_edge := 8
  let lateral_edge := 7
  4 * (1 / 2 * base_edge * Real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 16 * Real.sqrt 33 := 
by
  let base_edge := 8
  let lateral_edge := 7
  have h1 : 4 * (1 / 2 * base_edge * Real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 
              4 * (1 / 2 * 8 * Real.sqrt (49 - 16)) := 
    by sorry
  have h2 : 4 * (1 / 2 * 8 * Real.sqrt (49 - 16)) = 4 * (4 * Real.sqrt 33) := 
    by sorry
  have h3 : 4 * (4 * Real.sqrt 33) = 16 * Real.sqrt 33 := 
    by sorry
  exact eq.trans (eq.trans h1 h2) h3

end pyramid_total_area_l420_420221


namespace traveler_payment_problem_l420_420757

-- Define the rings and conditions
def rings : Type := Fin 7
def chain (r : rings) : Bool := true  -- Simplified condition to indicate a ring is in the chain

-- Define the condition of the problem
structure TravelPayment :=
  (chain : rings → Bool)
  (cutRing : rings)

constant R1 R2 R3 R4 R5 R6 R7 : rings

axiom ring_chain : ∀ r : rings, chain r
axiom cut_ring : cutRing = R3

-- Define the sequence of ring payments
def payment_sequence : list rings :=
  [R3,
   R1, R2, R3,
   R3,
   R4, R5, R6, R7, R1, R2, R3,
   R3,
   R1, R2, R3,
   R3]

-- Define the Lean theorem
theorem traveler_payment_problem (t : TravelPayment) :
  ∀ (day : Fin 7), (∃ seq : list rings, payment_sequence) :=
by
  intro day
  use payment_sequence
  sorry

end traveler_payment_problem_l420_420757


namespace profit_percentage_l420_420723

theorem profit_percentage (cost_price selling_price : ℝ) (h_cost : cost_price = 60) (h_selling : selling_price = 75) : 
  ((selling_price - cost_price) / cost_price) * 100 = 25 := by
  sorry

end profit_percentage_l420_420723


namespace number_of_subsets_l420_420172

theorem number_of_subsets {A : set ℕ} (H1: {1} ⊆ A) (H2: A ⊆ {1, 2, 3, 4}) :
  (A = {1} ∨ A = {1, 2} ∨ A = {1, 3} ∨ A = {1, 4} ∨
   A = {1, 2, 3} ∨ A = {1, 2, 4} ∨ A = {1, 3, 4} ∨ A = {1, 2, 3, 4}) :=
sorry

end number_of_subsets_l420_420172


namespace idiom1_correct_idiom2_correct_idiom3_correct_idiom4_correct_idiom5_correct_l420_420795

-- Definitions for each of the idioms based on the conditions provided

def idiom1_question := "Eight-row dance in the courtyard, \_\_\_\_\_\_, \_\_\_\_\_\_?"
def idiom1_answer := "it is as if the heavens and the earth are in harmony, and all things are flourishing."

def idiom2_question := "It is difficult to find a true friend in life. Zilu took this as his life's aspiration: \_\_\_\_\_\_\_\_\_\_, \_\_\_\_\_\_\_\_\_\_."
def idiom2_answer := "if one is fortunate enough to be born with a true friend, then tens of thousands of families' wealth cannot compare to having a heart-to-heart connection with you."

def idiom3_question := "Benevolent men have a mind to achieve benevolence; \_\_\_\_\_\_\_\_, \_\_\_\_\_\_\_\_."
def idiom3_answer := "they must be courageous and never tire of learning."

def idiom4_question := "Teachers should \_\_\_\_\_\_, \_\_\_\_\_\_, suggesting not to guide students prematurely."
def idiom4_answer := "teach in accordance with students' aptitudes and answer their questions when they are eager to learn."

def idiom5_question := "Only extremely strong winds can carry the large Peng bird flying towards the Southern Darkness. \_\_\_\_\_\_, \_\_\_\_\_\_."
def idiom5_answer := "only when wind accumulates can it carry the large Peng bird flying towards the Southern Darkness."

-- Theorem statements to prove the answers are correct given the conditions
theorem idiom1_correct : idiom1_question = idiom1_answer :=
by
  sorry

theorem idiom2_correct : idiom2_question = idiom2_answer :=
by
  sorry

theorem idiom3_correct : idiom3_question = idiom3_answer :=
by
  sorry

theorem idiom4_correct : idiom4_question = idiom4_answer :=
by
  sorry

theorem idiom5_correct : idiom5_question = idiom5_answer :=
by
  sorry

end idiom1_correct_idiom2_correct_idiom3_correct_idiom4_correct_idiom5_correct_l420_420795


namespace find_largest_root_l420_420428

noncomputable def largest_root_of_equation : ℝ := 3

theorem find_largest_root (x : ℝ) :
  3 * real.sqrt (x - 2) + 2 * real.sqrt (2 * x + 3) + real.sqrt (x + 1) = 11 → x ≥ 2 → x = largest_root_of_equation :=
begin
  sorry
end

end find_largest_root_l420_420428


namespace correct_propositions_l420_420018

noncomputable def propositions : List Prop :=
  [ (∀ x, f x = ln (x - 1) + 2 → (f 1 = 2 → False)),
    (∀ (f : ℝ → ℝ), Function.domain f = Icc (-1 : ℝ) (1 : ℝ) → Function.domain (f ∘ (λ x, 2 * x - 1)) = Icc (-3 : ℝ) (1 : ℝ)),
    (∀ (f : {a, b} → {-1, 0, 1}), f b = 0 → number_of_mappings_with_property = 3),
    (∀ a, Function.domain (λ (x : ℝ), log 2 (x^2 - 2*a*x + 1)) = ℝ → -1 < a ∧ a < 1),
    (∀ x, f x = exp x → (∃ y, (exp⁻¹ y) = ln x)) ]

theorem correct_propositions (h₁ : ∀ x, f x = ln (x - 1) + 2 → (f 1 = 2 → False))
                            (h₂ : ∀ (f : ℝ → ℝ), Function.domain f = Icc (-1 : ℝ) (1 : ℝ) → Function.domain (f ∘ (λ x, 2 * x - 1)) = Icc (0 : ℝ) (1 : ℝ))
                            (h₃ : ∀ (f : {a, b} → {-1, 0, 1}), f b = 0 → number_of_mappings_with_property = 3)
                            (h₄ : ∀ a, Function.domain (λ (x : ℝ), log 2 (x^2 - 2*a*x + 1)) = ℝ → -1 < a ∧ a < 1)
                            (h₅ : ∀ x, f x = exp x → (∃ y, (exp⁻¹ y) = ln x)) : 
                            propositions = [False, False, True, True, False] :=
by sorry

end correct_propositions_l420_420018


namespace harry_less_than_half_selena_l420_420581

-- Definitions of the conditions
def selena_book_pages := 400
def harry_book_pages := 180
def half (n : ℕ) := n / 2

-- The theorem to prove that Harry's book is 20 pages less than half of Selena's book.
theorem harry_less_than_half_selena :
  harry_book_pages = half selena_book_pages - 20 := 
by
  sorry

end harry_less_than_half_selena_l420_420581


namespace sufficient_and_necessary_cond_l420_420376

theorem sufficient_and_necessary_cond (x : ℝ) : |x| > 2 ↔ (x > 2) :=
sorry

end sufficient_and_necessary_cond_l420_420376


namespace solve_problem_l420_420550

theorem solve_problem : 
  ∃ p q : ℝ, 
    (p ≠ q) ∧ 
    ((∀ x : ℝ, (x = p ∨ x = q) ↔ (x-4)*(x+4) = 24*x - 96)) ∧ 
    (p > q) ∧ 
    (p - q = 16) :=
by
  sorry

end solve_problem_l420_420550


namespace batsman_avg_after_17th_inning_l420_420722

def batsman_average : Prop :=
  ∃ (A : ℕ), 
    (A + 3 = (16 * A + 92) / 17) → 
    (A + 3 = 44)

theorem batsman_avg_after_17th_inning : batsman_average :=
by
  sorry

end batsman_avg_after_17th_inning_l420_420722


namespace max_f_min_f_inequality_f_l420_420024

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem max_f : ∃ x ∈ Icc (1/3 : ℝ) 3, f x = 3 * Real.log 3 :=
sorry

theorem min_f : ∃ x ∈ Icc (1/3 : ℝ) 3, f x = -1/Real.exp(1) :=
sorry

noncomputable def h (x : ℝ) : ℝ := Real.log x - x + 1

theorem inequality_f (x : ℝ) (hx : 0 < x) : h x ≤ 0 :=
sorry

end max_f_min_f_inequality_f_l420_420024


namespace triangle_PAQ_is_isosceles_right_triangle_l420_420678

noncomputable def isosceles_right_triangle (A P Q : ℝ × ℝ) : Prop :=
  let AP := (P.1 - A.1, P.2 - A.2)
  let AQ := (Q.1 - A.1, Q.2 - A.2)
  (AP.1 * AQ.1 + AP.2 * AQ.2 = 0) ∧ -- AP and AQ are perpendicular
  (AP.1^2 + AP.2^2 = AQ.1^2 + AQ.2^2) -- |AP| = |AQ|

theorem triangle_PAQ_is_isosceles_right_triangle
  (A B C M H N G E D P Q : ℝ × ℝ)
  (h1: vector AB = b) (h2: vector AC = c)
  (h3: vector BM = -k * b) (h4: vector CN = k * c)
  (h5: vector BE = k * (b - c)) (h6: vector CD = k * (b - c))
  (h7: vector BP = -k * c) (h8: vector CQ = k * b)
  (h9: vector AP = b - k * c) (h10: vector AQ = c + k * b) :
  isosceles_right_triangle A P Q := sorry

end triangle_PAQ_is_isosceles_right_triangle_l420_420678


namespace count_integer_solutions_in_circle_l420_420832

theorem count_integer_solutions_in_circle :
  let circle_equation := λ (x y : ℝ), (x - 3)^2 + (y - 3)^2 ≤ 144
  ∃ count : ℕ, count = 15 ∧ ∀ x : ℤ, circle_equation (x : ℝ) (-x : ℝ) → -7 ≤ x ∧ x ≤ 7 :=
by
  let circle_equation := λ (x y : ℝ), (x - 3)^2 + (y - 3)^2 ≤ 144
  sorry

end count_integer_solutions_in_circle_l420_420832


namespace new_parallelogram_vertices_exists_l420_420733

variable {Point : Type} [AddCommGroup Point] [Module ℝ Point]

structure Parallelogram (P : Type) [AddCommGroup P] [Module ℝ P] :=
(A B C D O : P)
(center_eq_bisector : O = (A + C) / 2 ∧ O = (B + D) / 2)
(sides_intersect : ∃ P Q : P, ∃ line_through_center : P → P, 
                   (line_through_center O = line_through_center P) ∧ 
                   (line_through_center O = line_through_center Q) ∧
                   ((A + P) = line_through_center A) ∧
                   ((C + Q) = line_through_center C) ∧
                   ((B + P) = line_through_center B) ∧
                   ((D + Q) = line_through_center D))

noncomputable def new_parallelogram_vertices {P : Type} [AddCommGroup P] [Module ℝ P]
  (paral : Parallelogram P)
  (segments : ∀ {A B C D O : P} (P Q : P), Parallelogram P) : 
  Prop :=
∃ new_parallelogram : Parallelogram P,
(intersection_AP_AC := ∃ X, ∃ Y, X = intersection (segment A P) (diagonal A C) ∧
                                        Y = new_parallelogram A B C D O),
(intersection_BP_BD := ∃ X, ∃ Y, X = intersection (segment B P) (diagonal B D) ∧
                                        Y = new_parallelogram A B C D O), 
(intersection_CQ_AC := ∃ X, ∃ Y, X = intersection (segment C Q) (diagonal A C) ∧
                                        Y = new_parallelogram A B C D O), 
(intersection_DQ_BD := ∃ X, ∃ Y, X = intersection (segment D Q) (diagonal B D) ∧
                                        Y = new_parallelogram A B C D O)

theorem new_parallelogram_vertices_exists {P : Type} [AddCommGroup P] [Module ℝ P]
  {A B C D O P Q : Point}
  (paral : Parallelogram P) 
  (line_through_center_intersects : ∃ line_through_center : P → P, 
                   (line_through_center O = line_through_center P) ∧ 
                   (line_through_center O = line_through_center Q))
  :
  new_parallelogram_vertices paral line_through_center_intersects := 
  sorry

end new_parallelogram_vertices_exists_l420_420733


namespace vector_magnitude_cos_sin_l420_420849

theorem vector_magnitude_cos_sin {α : ℝ} (a : ℝ × ℝ) 
  (h : a = (Real.cos α, Real.sin α)) : ∥a∥ = 1 := 
by 
  sorry

end vector_magnitude_cos_sin_l420_420849


namespace combined_dog_years_difference_l420_420307

theorem combined_dog_years_difference 
  (Max_age : ℕ) 
  (small_breed_rate medium_breed_rate large_breed_rate : ℕ) 
  (Max_turns_age : ℕ) 
  (small_breed_diff medium_breed_diff large_breed_diff combined_diff : ℕ) :
  Max_age = 3 →
  small_breed_rate = 5 →
  medium_breed_rate = 7 →
  large_breed_rate = 9 →
  Max_turns_age = 6 →
  small_breed_diff = small_breed_rate * Max_turns_age - Max_turns_age →
  medium_breed_diff = medium_breed_rate * Max_turns_age - Max_turns_age →
  large_breed_diff = large_breed_rate * Max_turns_age - Max_turns_age →
  combined_diff = small_breed_diff + medium_breed_diff + large_breed_diff →
  combined_diff = 108 :=
by
  intros
  sorry

end combined_dog_years_difference_l420_420307


namespace find_k_l420_420030

def setA (k : ℝ) : set ℝ := {x | 1 < x ∧ x < k}
def setB (k : ℝ) : set ℝ := {y | ∃ x, y = 2 * x - 5 ∧ 1 < x ∧ x < k}

theorem find_k (k : ℝ) :
  (setA k ∩ setB k = {x | 1 < x ∧ x < 2}) ↔ k = 3.5 :=
by
  sorry

end find_k_l420_420030


namespace largest_both_writers_editors_l420_420775

-- Define the conditions
def writers : ℕ := 45
def editors_gt : ℕ := 38
def total_attendees : ℕ := 90
def both_writers_editors (x : ℕ) : ℕ := x
def neither_writers_editors (x : ℕ) : ℕ := x / 2

-- Define the main proof statement
theorem largest_both_writers_editors :
  ∃ x : ℕ, x ≤ 4 ∧
  (writers + (editors_gt + (0 : ℕ)) + neither_writers_editors x + both_writers_editors x = total_attendees) :=
sorry

end largest_both_writers_editors_l420_420775


namespace system1_solution_system2_solution_l420_420143

-- System (1)
theorem system1_solution (x y : ℚ) (h1 : 3 * y - 4 * x = 0) (h2 : 4 * x + y = 8) : 
  x = 3 / 2 ∧ y = 2 :=
by
  sorry

-- System (2)
theorem system2_solution (x y : ℚ) (h1 : x + y = 3) (h2 : (x - 1) / 4 + y / 2 = 3 / 4) : 
  x = 2 ∧ y = 1 :=
by
  sorry

end system1_solution_system2_solution_l420_420143


namespace tina_jumps_more_than_cindy_l420_420782

def cindy_jumps : ℕ := 12
def betsy_jumps : ℕ := cindy_jumps / 2
def tina_jumps : ℕ := betsy_jumps * 3

theorem tina_jumps_more_than_cindy : tina_jumps - cindy_jumps = 6 := by
  sorry

end tina_jumps_more_than_cindy_l420_420782


namespace continuity_of_g_at_0_f_discontinuity_at_0_g_continuous_at_0_l420_420527

noncomputable def f (x : ℝ) : ℝ := if x = 0 then 3 else sin (3 * x) / x

theorem continuity_of_g_at_0:
  ∀ ε > 0, ∃ δ > 0, ∀ x, abs x < δ → abs (f x - 3) < ε :=
sorry

theorem f_discontinuity_at_0 :
  (∀ ε > 0, ∃ δ > 0, ∀ x, x ≠ 0 ∧ abs x < δ → abs (sin (3 * x) / x - 3) < ε)
:=
sorry

def g (x : ℝ) : ℝ :=
  if x = 0 then 3 else sin (3 * x) / x

theorem g_continuous_at_0 :
  continuous_at g 0 :=
sorry

end continuity_of_g_at_0_f_discontinuity_at_0_g_continuous_at_0_l420_420527


namespace weight_of_replaced_person_l420_420161

theorem weight_of_replaced_person (avg_increase : ℝ) (new_person_weight : ℝ) (num_people : ℕ)
  (h_avg_increase : avg_increase = 4.5) (h_new_person_weight : new_person_weight = 101)
  (h_num_people : num_people = 8) : 
  let total_increase := num_people * avg_increase in
  let replaced_weight := new_person_weight - total_increase in
  replaced_weight = 65 :=
by
  have h1 : total_increase = 8 * 4.5, from sorry,
  have h2 : replaced_weight = 101 - total_increase, from sorry,
  exact sorry

end weight_of_replaced_person_l420_420161


namespace pyramid_area_l420_420305

noncomputable def base_edge : ℝ := 8
noncomputable def lateral_edge : ℝ := 7
noncomputable def height := real.sqrt (lateral_edge^2 - (base_edge / 2)^2)
noncomputable def one_triangle_area := (1 / 2) * base_edge * height
noncomputable def total_area := 4 * one_triangle_area

theorem pyramid_area : total_area = 16 * real.sqrt 33 := by
  sorry

end pyramid_area_l420_420305


namespace correct_option_l420_420310

theorem correct_option :
  (3 * a^2 + 5 * a^2 ≠ 8 * a^4) ∧
  (5 * a^2 * b - 6 * a * b^2 ≠ -a * b^2) ∧
  (2 * x + 3 * y ≠ 5 * x * y) ∧
  (9 * x * y - 6 * x * y = 3 * x * y) :=
by
  sorry

end correct_option_l420_420310


namespace largest_coins_l420_420320

theorem largest_coins (n k : ℕ) (h1 : n = 13 * k + 3) (h2 : n < 150) : n = 146 :=
by
  sorry

end largest_coins_l420_420320


namespace min_value_l420_420005

-- Define the variables and conditions
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → (m = 1 ∨ m = n)

constants (a b c d : ℕ)
constants (x : ℕ)

-- Given conditions
axiom prime_a : is_prime a
axiom prime_b : is_prime b
axiom prime_c : is_prime c
axiom prime_d : is_prime d
axiom sum_35 : a * b * c * d = 35 * (x + 17)

-- The theorem stating the minimum value to prove
theorem min_value : a + b + c + d = 22 :=
sorry

end min_value_l420_420005


namespace find_k_l420_420697
-- Import the necessary library

-- Given conditions as definitions
def circle_eq (x y : ℝ) (k : ℝ) : Prop :=
  x^2 + 8 * x + y^2 + 2 * y + k = 0

def radius_sq : ℝ := 25  -- since radius = 5, radius squared is 25

-- The statement to prove
theorem find_k (x y k : ℝ) : circle_eq x y k → radius_sq = 25 → k = -8 :=
by
  sorry

end find_k_l420_420697


namespace zane_purchased_shirts_l420_420328

-- Let's define the conditions
def regular_price : ℝ := 50
def discount : ℝ := 0.40
def total_paid : ℝ := 60

-- Define the price after discount
def discounted_price (original_price : ℝ) (discount : ℝ) : ℝ := original_price * (1 - discount)

-- Define the goal: number of shirts bought
def number_of_shirts_bought (total_paid : ℝ) (price_per_shirt : ℝ) : ℝ := total_paid / price_per_shirt 

-- Our theorem stating that the number of shirts bought is 2
theorem zane_purchased_shirts : number_of_shirts_bought total_paid (discounted_price regular_price discount) = 2 :=
by
  sorry

end zane_purchased_shirts_l420_420328


namespace AF_less_than_BD_l420_420572

theorem AF_less_than_BD
  (A B C D E F : Point)
  (h_tri : regular_triangle A B E)
  (h_rhomb : rhombus B C D E)
  (h_ext : built_outside BE B C D E)
  (h_intersect : intersect AC BD F) :
  length AF < length BD := 
sorry

end AF_less_than_BD_l420_420572


namespace solve_geometric_series_problem_l420_420091

def geo_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

def C_n (n : ℕ) : ℝ :=
  geo_sum 880 (1/3) n

def D_n (n : ℕ) : ℝ :=
  geo_sum 1344 (-1/3) n

theorem solve_geometric_series_problem :
  ∃ (n : ℕ), n ≥ 1 ∧ C_n n = D_n n ∧ n = 2 := 
sorry

end solve_geometric_series_problem_l420_420091


namespace total_area_of_pyramid_faces_l420_420271

-- Define the basic parameters of the pyramid
def base_edges := 8
def lateral_edges := 7

-- Define the Pythagorean theorem components
def altitude_squared := lateral_edges^2 - (base_edges / 2)^2
def altitude := real.sqrt altitude_squared

-- Define the area of one triangular face using half-base and altitude
def one_face_area := (1 / 2) * base_edges * altitude

-- Define the total area of the four triangular faces
def total_area_of_faces := 4 * one_face_area

-- Statement to prove
theorem total_area_of_pyramid_faces : total_area_of_faces = 16 * real.sqrt 33 :=
by
  -- Necessary calculations are assured correct by previous definitions and attributes
  sorry

end total_area_of_pyramid_faces_l420_420271


namespace heather_distance_is_approx_ten_point_two_seven_l420_420338

open Real

def stacy_heather_walk :=
  let distance := 25 -- initial distance between them in miles
  let H := 5 -- Heather's speed in miles/hour
  let S := 6 -- Stacy's speed in miles/hour (H + 1)
  let delay := 0.4 -- Heather starts 0.4 hours after Stacy
  let distanceStacyCovers := S * delay -- Distance stacy covers in delay hours
  let remainingDistance := distance - distanceStacyCovers -- Remaining distance after delay
  let timeToMeet := remainingDistance / (H + S) -- Time it takes for Heather and Stacy to meet
  let heatherDistance := H * timeToMeet -- Distance Heather walks when they meet
  heatherDistance

theorem heather_distance_is_approx_ten_point_two_seven :
  (stacy_heather_walk ≈ 10.27) :=
by sorry -- Placeholder for the actual proof

end heather_distance_is_approx_ten_point_two_seven_l420_420338


namespace measure_dihedral_angle_A_BD1_A1_60deg_l420_420392

noncomputable def measure_dihedral_angle (A B D1 A1 : Point) (cube : Cube A B D1 A1) : Real :=
  dihedral_angle (Plane A B D1) (Plane A E F)

theorem measure_dihedral_angle_A_BD1_A1_60deg (A B D1 A1 : Point) (cube : Cube A B D1 A1) : 
  measure_dihedral_angle A B D1 A1 cube = 60 :=
sorry

end measure_dihedral_angle_A_BD1_A1_60deg_l420_420392


namespace fraction_sum_eq_five_l420_420699

theorem fraction_sum_eq_five : 
  let num := 54
  let denom := 81
  let gcd_num_denom := Int.gcd num denom
  let simplest_num := num / gcd_num_denom
  let simplest_denom := denom / gcd_num_denom
  simplest_num + simplest_denom = 5 :=
by
  have gcd_num_denom_eq : gcd_num_denom = Int.gcd 54 81
  have gcd_num_denom_val : gcd_num_denom = 27
  have simplest_num_eq : simplest_num = 54 / 27
  have simplest_denom_eq : simplest_denom = 81 / 27
  have simplest_num_val : simplest_num = 2
  have simplest_denom_val : simplest_denom = 3
  rw [simplest_num_val, simplest_denom_val]
  exact rfl

end fraction_sum_eq_five_l420_420699


namespace problem_statement_l420_420575

theorem problem_statement (n : ℕ) (h1 : n > 1) :
  let middle_expr := (1 + ∑ i in finset.range (n+1), 1 / (2^i : ℝ)) in
  (n = 2) → middle_expr = 1 + 1 / 2 + 1 / 3 + 1 / 4 :=
by
  intros h2
  simp [h2]
  sorry

end problem_statement_l420_420575


namespace find_height_of_cylinder_l420_420818

theorem find_height_of_cylinder (r SA : ℝ) (h : ℝ) (h_r : r = 3) (h_SA : SA = 30 * Real.pi) :
  SA = 2 * Real.pi * r^2 + 2 * Real.pi * r * h → h = 2 :=
by
  sorry

end find_height_of_cylinder_l420_420818


namespace truck_speed_l420_420759

theorem truck_speed {d t : ℝ} (h_d : d = 600) (h_t : t = 30) :
  (d / t) * (1 / 1000) * 3600 = 72 :=
by
  rw [h_d, h_t]
  norm_num
  sorry

end truck_speed_l420_420759


namespace coefficient_of_x_in_binomial_expansion_l420_420407

theorem coefficient_of_x_in_binomial_expansion :
  let f := λ (x : ℝ), (2 * x^2 - 1 / x)^5 in
  (coeff f 1) = -40 :=
by
  -- define the function f
  let f : ℝ → ℝ := λ x, (2 * x^2 - 1 / x)^5
  -- define the coefficient function
  let coeff : (ℝ → ℝ) → ℕ → ℝ := sorry
  -- add the statement to be proven
  have : coeff f 1 = -40 := sorry
  exact this
  sorry

end coefficient_of_x_in_binomial_expansion_l420_420407


namespace construct_right_triangle_with_median_l420_420026

noncomputable def problem (PQ RS : ℝ) : Prop :=
  ∃ (P Q R S T U V : Type)
    (PQ_segment RS_segment : P × Q)
    (QT_segment PT_segment T_line P_triangle P_square R_parallel_line U_perpendicular V_midpoint : Set Type),
    (PQ_segment ≈ PQ)
    ∧ (RS_segment ≈ RS)
    ∧ (QT_segment ≈ RS)
    ∧ (PT_segment ≈ QT)
    ∧ (P_triangle ≈ RightTriangle)
    ∧ (M_median (PQ_segment QT_segment PT_segment U_perpendicular) ≈ PQ ∪ RS)

theorem construct_right_triangle_with_median 
  (PQ RS : ℝ) : problem PQ RS :=
by
  sorry

end construct_right_triangle_with_median_l420_420026


namespace intersection_is_correct_l420_420505

def M : Set ℝ := { x | -2 < x ∧ x < 3 }
def N : Set ℝ := { x | 2^(x+1) ≥ 1 }
def intersect_M_N : Set ℝ := { x | -1 ≤ x ∧ x < 3 }

theorem intersection_is_correct : M ∩ N = intersect_M_N :=
by
  sorry

end intersection_is_correct_l420_420505


namespace number_of_ways_to_form_divisible_by_18_l420_420520

theorem number_of_ways_to_form_divisible_by_18:
  let base_number := [2, 0, 1, 6, 0, 2] in
  let possibles := {1,2,3,4,5,6,7,8,9}.toFinset in
  ∃ replacements : list ℕ, (replacements.length = 6 ∧ 
    (replacements ⊆ possibles.toList) ∧ 
    ((base_number ++ replacements).nth 11 ∈ {2, 4, 6, 8}) ∧ 
    (((base_number.sum + replacements.sum) % 9 = 0)) ∧ 
    (replacements.prod = 26244)) := sorry

end number_of_ways_to_form_divisible_by_18_l420_420520


namespace fraction_furniture_spent_l420_420111

theorem fraction_furniture_spent (S T : ℕ) (hS : S = 600) (hT : T = 300) : (S - T) / S = 1 / 2 :=
by
  sorry

end fraction_furniture_spent_l420_420111


namespace can_obtain_numbers_l420_420653

theorem can_obtain_numbers (x y : ℝ) (h1 : x > 0) (h2 : y > 0) :
  ∃ z1 z2 : ℝ, (∀ (a b : ℝ), a ∈ {x, y, z1, z2, 1, (1 / x), (x + 1), (1 / (x + 1)), ((1 / x) - (1 / (x + 1))), (x / ((x * (x + 1)))), x^2, xy} ∧ b ∈ {x, y, z1, z2, 1, (1 / x), (x + 1), (1 / (x + 1)), ((1 / x) - (1 / (x + 1))), (x / ((x * (x + 1)))), x^2, xy}
    → (a + b) ∈ {x, y, z1, z2, 1, (1 / x), (x + 1), (1 / (x + 1)), ((1 / x) - (1 / (x + 1))), (x / ((x * (x + 1)))), x^2, xy} ∧
      (a - b) ∈ {x, y, z1, z2, 1, (1 / x), (x + 1), (1 / (x + 1)), ((1 / x) - (1 / (x + 1))), (x / ((x * (x + 1)))), x^2, xy} ∧
      (1 / a) ∈ {x, y, z1, z2, 1, (1 / x), (x + 1), (1 / (x + 1)), ((1 / x) - (1 / (x + 1))), (x / ((x * (x + 1)))), x^2, xy}) ∧
  ((∃ (a b : ℝ), a ∈ {x, y, z1, z2, 1, (1 / x), (x + 1), (1 / (x + 1)), ((1 / x) - (1 / (x + 1))), (x / ((x * (x + 1)))), x^2, xy} ∧ 
    b ∈ {x, y, z1, z2, 1, (1 / x), (x + 1), (1 / (x + 1)), ((1 / x) - (1 / (x + 1))), (x / ((x * (x + 1)))), x^2, xy} ∧
    a * b = x^2) ∧
   (∃ (a b : ℝ), a ∈ {x, y, z1, z2, 1, (1 / x), (x + 1), (1 / (x + 1)), ((1 / x) - (1 / (x + 1))), (x / ((x * (x + 1)))), x^2, xy} ∧ 
    b ∈ {x, y, z1, z2, 1, (1 / x), (x + 1), (1 / (x + 1)), ((1 / x) - (1 / (x + 1))), (x / ((x * (x + 1)))), x^2, xy} ∧
    a * b = x * y)) :=
by sorry

end can_obtain_numbers_l420_420653


namespace pyramid_total_area_l420_420285

noncomputable def pyramid_base_edge : ℝ := 8
noncomputable def pyramid_lateral_edge : ℝ := 7

theorem pyramid_total_area 
  (base_edge : ℝ)
  (lateral_edge : ℝ)
  (h_base_edge : base_edge = pyramid_base_edge)
  (h_lateral_edge : lateral_edge = pyramid_lateral_edge) 
: 4 * (1 / 2 * base_edge * real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 16 * real.sqrt 33 := 
sorry

end pyramid_total_area_l420_420285


namespace exists_set_of_k_integers_no_infinite_set_of_not_coprime_two_coprime_three_l420_420709

-- Definition for proof problem 1
theorem exists_set_of_k_integers (k : ℕ) (hk : k ≥ 3) : 
  ∃ (A : Finset ℕ), A.card = k ∧ (∀ a b ∈ A, a ≠ b → ¬ coprime a b) ∧ (∀ a b c ∈ A, a ≠ b → b ≠ c → a ≠ c → coprime a (b * c)) :=
sorry

-- Definition for proof problem 2
theorem no_infinite_set_of_not_coprime_two_coprime_three :
  ¬ ∃ (A : Set ℕ), (∀ a b ∈ A, a ≠ b → ¬ coprime a b) ∧ (∀ a b c ∈ A, a ≠ b → b ≠ c → a ≠ c → coprime a (b * c)) ∧ infinite A :=
sorry

end exists_set_of_k_integers_no_infinite_set_of_not_coprime_two_coprime_three_l420_420709


namespace identify_mathematicians_l420_420764

def famous_people := List (Nat × String)

def is_mathematician : Nat → Bool
| 1 => false  -- Bill Gates
| 2 => true   -- Gauss
| 3 => false  -- Yuan Longping
| 4 => false  -- Nobel
| 5 => true   -- Chen Jingrun
| 6 => true   -- Hua Luogeng
| 7 => false  -- Gorky
| 8 => false  -- Einstein
| _ => false  -- default case

theorem identify_mathematicians (people : famous_people) : 
  (people.filter (fun (n, _) => is_mathematician n)) = [(2, "Gauss"), (5, "Chen Jingrun"), (6, "Hua Luogeng")] :=
by sorry

end identify_mathematicians_l420_420764


namespace complex_number_location_l420_420471

noncomputable def z : ℂ := (i + i^2 + i^3 + ... + i^2017) / (1 + i)

theorem complex_number_location :
  let num := (i + i^2 + i^3 + ... + i^2017) in
  let denom := (1 + i) in
  let z := num / denom in
  ∃ (x y : ℝ), z = x + y * I ∧ (0 < x ∧ 0 < y) := 
sorry

end complex_number_location_l420_420471


namespace area_triangle_OAB_l420_420484

theorem area_triangle_OAB :
  let O := (0 : ℝ, 0 : ℝ),
      A := (π / 6, 2 * Real.cos (π / 6)),
      B := (5 * π / 6, 2 * Real.cos (5 * π / 6)) in
  -- Calculate the intersection points
  A = (π / 6, sqrt 3) ∧ B = (5 * π / 6, -sqrt 3) →
  -- Calculate the area of the triangle
  let area := |(π / 2) * 2 * sqrt 3 / 2| in
  area = (sqrt 3 * π) / 2 :=
by
  sorry

end area_triangle_OAB_l420_420484


namespace find_scalars_l420_420967

open Matrix

def B : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![2, -1],
    ![4, 3]]

def I : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 0],
    ![0, 1]]

theorem find_scalars (r s : ℤ) (h : B^6 = r • B + s • I) :
  (r = 1125) ∧ (s = -1875) :=
sorry

end find_scalars_l420_420967


namespace correct_option_is_C_l420_420702

theorem correct_option_is_C : 
  (sqrt 4 = 2 ∨ sqrt 4 ≠ ± 2) ∧ 
  (3 - 27 / 64 = -3 / 4) ∧ 
  (abs (sqrt 2 - 1) = sqrt 2 - 1) → 
  (3 - 8 = -5) :=
by sorry

end correct_option_is_C_l420_420702


namespace square_side_length_l420_420465

theorem square_side_length (x : ℝ) 
  (h : x^2 = 6^2 + 8^2) : x = 10 := 
by sorry

end square_side_length_l420_420465


namespace C₁_C₂_intersect_at_two_points_l420_420638

open Real

variable (m : ℝ)

/-- Condition: C₁ is a circle and C₂ is a straight line, with given equations -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 2 * x = 0
def C₂ (x y : ℝ) : Prop := 2 * x - y - 2 * m - 1 = 0

/-- Proof problem: if circle C₁ and line C₂ intersect at two different points,
    then the parameter m falls within a specific range -/
theorem C₁_C₂_intersect_at_two_points (h : ∃ x y, C₁ x y ∧ C₂ x y) :
  (1 - sqrt 5) / 2 < m ∧ m < (1 + sqrt 5) / 2 :=
by
  sorry

end C₁_C₂_intersect_at_two_points_l420_420638


namespace cost_of_fencing_l420_420333

noncomputable def fencingCost :=
  let π := 3.14159
  let diameter := 32
  let costPerMeter := 1.50
  let circumference := π * diameter
  let totalCost := costPerMeter * circumference
  totalCost

theorem cost_of_fencing :
  let roundedCost := (fencingCost).round
  roundedCost = 150.80 :=
by
  sorry

end cost_of_fencing_l420_420333


namespace pyramid_area_l420_420299

noncomputable def base_edge : ℝ := 8
noncomputable def lateral_edge : ℝ := 7
noncomputable def height := real.sqrt (lateral_edge^2 - (base_edge / 2)^2)
noncomputable def one_triangle_area := (1 / 2) * base_edge * height
noncomputable def total_area := 4 * one_triangle_area

theorem pyramid_area : total_area = 16 * real.sqrt 33 := by
  sorry

end pyramid_area_l420_420299


namespace ravi_profit_l420_420336

theorem ravi_profit
  (cost_price_refrigerator : ℕ) (cost_price_refrigerator = 15000)
  (loss_percent_refrigerator : ℕ) (loss_percent_refrigerator = 4)
  (cost_price_mobile : ℕ) (cost_price_mobile = 8000)
  (profit_percent_mobile : ℕ) (profit_percent_mobile = 10) :
  OverallProfit 200 := 
    sorry -- The proof will go here.

end ravi_profit_l420_420336


namespace equation_of_midpoint_trajectory_l420_420856

theorem equation_of_midpoint_trajectory
  (M : ℝ × ℝ)
  (hM : M.1 ^ 2 + M.2 ^ 2 = 1)
  (N : ℝ × ℝ := (2, 0))
  (P : ℝ × ℝ := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)) :
  (P.1 - 1) ^ 2 + P.2 ^ 2 = 1 / 4 := 
sorry

end equation_of_midpoint_trajectory_l420_420856


namespace min_abs_x1_x2_l420_420481

noncomputable def f (a x : ℝ) : ℝ := a * Real.sin x - 2 * Real.sqrt 3 * Real.cos x

theorem min_abs_x1_x2 
  (a x1 x2 : ℝ)
  (h_symmetry : ∃ c : ℝ, c = -Real.pi / 6 ∧ (∀ x, f a (x - c) = f a x))
  (h_product : f a x1 * f a x2 = -16) :
  ∃ m : ℝ, m = abs (x1 + x2) ∧ m = 2 * Real.pi / 3 :=
by sorry

end min_abs_x1_x2_l420_420481


namespace largest_coins_l420_420321

theorem largest_coins (n k : ℕ) (h1 : n = 13 * k + 3) (h2 : n < 150) : n = 146 :=
by
  sorry

end largest_coins_l420_420321


namespace pyramid_total_area_l420_420294

noncomputable def pyramid_base_edge : ℝ := 8
noncomputable def pyramid_lateral_edge : ℝ := 7

theorem pyramid_total_area 
  (base_edge : ℝ)
  (lateral_edge : ℝ)
  (h_base_edge : base_edge = pyramid_base_edge)
  (h_lateral_edge : lateral_edge = pyramid_lateral_edge) 
: 4 * (1 / 2 * base_edge * real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 16 * real.sqrt 33 := 
sorry

end pyramid_total_area_l420_420294


namespace sum_y_eq_exp_m_sub_2_l420_420096

variable (m : ℕ) (h_pos : 0 < m)

def y : ℕ → ℝ
| 0       := 1
| 1       := m
| (k + 2) := ((m - 2) * y (k + 1) + (k - m) * y k) / (k + 2)

theorem sum_y_eq_exp_m_sub_2 : (∑ i, y m i) = Real.exp (m - 2) := by
  sorry

end sum_y_eq_exp_m_sub_2_l420_420096


namespace series_less_than_one_l420_420793

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def sum_fractions (n : Nat) : Real :=
  match n with
  | 0 => 0
  | n + 1 => sum_fractions n + n / factorial (n + 1)

theorem series_less_than_one :
  sum_fractions 2012 < 1 :=
  sorry

end series_less_than_one_l420_420793


namespace continuous_stripe_probability_l420_420417

-- Definitions of conditions
def stripe_orientation (face: ℕ) : ℕ := sorry  -- Each face has two possible stripe orientations
def stripe_color (face: ℕ) : ℕ := sorry  -- Each stripe can be either red or blue
def faces : fin 6 := sorry  -- The cube has six faces

-- Total possible stripe combinations
def total_combinations : ℕ := 4 ^ 6

-- Counting favorable outcomes
def favorable_outcomes : ℕ :=
  let color_orientations_per_face : ℕ := 2
  let alignment_pairs : ℕ := 3
  let favorable_per_color : ℕ := color_orientations_per_face * alignment_pairs in
  2 * favorable_per_color -- for both red and blue

-- Probability calculation
def probability_favorable_outcome : ℚ :=
  favorable_outcomes / total_combinations

-- Theorem statement
theorem continuous_stripe_probability :
  probability_favorable_outcome = 3 / 512 :=
sorry

end continuous_stripe_probability_l420_420417


namespace Danny_bottle_caps_l420_420804

theorem Danny_bottle_caps (r w c : ℕ) (h1 : r = 11) (h2 : c = r + 1) : c = 12 := by
  sorry

end Danny_bottle_caps_l420_420804


namespace wicket_keeper_proof_l420_420729

noncomputable def wicket_keeper_age (team_avg_age : ℕ) (wicket_keeper_age_diff : ℕ) 
  (team_size : ℕ) (remaining_players_avg_age_diff : ℕ) (team_avg_age' : ℕ) : Prop :=
  let total_team_age := team_avg_age * team_size in
  let excluded_two_members := 2 in
  let remaining_players := team_size - excluded_two_members in
  let remaining_players_avg_age := team_avg_age' - remaining_players_avg_age_diff in
  let total_remaining_players_age := remaining_players_avg_age * remaining_players in
  let age_with_one_average := total_remaining_players_age + team_avg_age' in
  let wicket_keeper := total_team_age - age_with_one_average in
  wicket_keeper - team_avg_age' = wicket_keeper_age_diff

theorem wicket_keeper_proof :
  wicket_keeper_age 21 9 11 1 21 :=
by
  sorry

end wicket_keeper_proof_l420_420729


namespace area_curvilinear_trapezoid_l420_420594

-- Define the function y = sqrt(x)
def f (x : ℝ) : ℝ := real.sqrt x

-- Define the theorem statement about the area
theorem area_curvilinear_trapezoid : 
  (∫ x in 0..4, f x) = 16 / 3 := by
  sorry

end area_curvilinear_trapezoid_l420_420594


namespace residue_7_pow_1234_l420_420683

theorem residue_7_pow_1234 : (7^1234) % 13 = 4 := by
  sorry

end residue_7_pow_1234_l420_420683


namespace total_amount_l420_420377

theorem total_amount (x y z total : ℝ) (h1 : y = 0.45 * x) (h2 : z = 0.50 * x) (h3 : y = 27) : total = 117 :=
by
  -- Proof here
  sorry

end total_amount_l420_420377


namespace frequency_of_middle_group_l420_420070

theorem frequency_of_middle_group (sample_size : ℕ) (x : ℝ) (h : sample_size = 160) (h_rel_freq : x = 0.2) 
  (h_relation : x = (1 / 4) * (10 * x)) : 
  sample_size * x = 32 :=
by
  sorry

end frequency_of_middle_group_l420_420070


namespace Mary_younger_than_Albert_l420_420762

-- Define the basic entities and conditions
def Betty_age : ℕ := 11
def Albert_age : ℕ := 4 * Betty_age
def Mary_age : ℕ := Albert_age / 2

-- Define the property to prove
theorem Mary_younger_than_Albert : Albert_age - Mary_age = 22 :=
by 
  sorry

end Mary_younger_than_Albert_l420_420762


namespace circumradius_greatest_distance_inradius_least_distance_circumradius_greater_than_inradius_l420_420714

-- Definitions of Triangle, Circumcenter, Circumradius, Incenter, and Inradius
variables (ABC : Triangle) (O : Point) (O1 : Point) 
variables (R r : ℝ)

-- Conditions
axiom circumcenter_of_ABC : is_circumcenter ABC O
axiom circumradius_of_ABC : is_circumradius ABC R
axiom incenter_of_ABC : is_incenter ABC O1
axiom inradius_of_ABC : is_inradius ABC r

-- Theorem for parts (a) and (b)
theorem circumradius_greatest_distance :
  ∀ P : Point, distance O P ≤ R :=
sorry

theorem inradius_least_distance :
  ∀ P : Point, distance O1 P ≥ r :=
sorry

-- Main theorem that we need to prove
theorem circumradius_greater_than_inradius : R > r :=
sorry

end circumradius_greatest_distance_inradius_least_distance_circumradius_greater_than_inradius_l420_420714


namespace problem_I_problem_II_l420_420475

-- (I) Prove that given \( f(x) = |x - 3| + |x - 2| \), the solution set for the inequality \( f(x) \geq 3 \) is \( \{ x \mid x \leq 1 \text{ or } x \geq 4 \} \) when \( a = -3 \).
theorem problem_I (x : ℝ) (f : ℝ → ℝ) (h_f : f = (λ x, |x - 3| + |x - 2|)) (a : ℝ) (h_a : a = -3) :
  f(x) ≥ 3 ↔ x ≤ 1 ∨ x ≥ 4 := by
  sorry

-- (II) Prove that for the function \( f(x) = |x + a| + |x - 2| \) and the inequality \( f(x) \leq |x - 4| \) to hold over the interval \([1, 2]\), \( a \) must lie in the range \([-3, 0]\).
theorem problem_II (a : ℝ) (f : ℝ → ℝ) (h_f : f = (λ x, |x + a| + |x - 2|)) :
  (∀ x ∈ set.Icc 1 2, f(x) ≤ |x - 4|) ↔ -3 ≤ a ∧ a ≤ 0 := by
  sorry

end problem_I_problem_II_l420_420475


namespace DVDs_already_in_book_l420_420351

variables (total_capacity empty_spaces already_in_book : ℕ)

-- Conditions given in the problem
def total_capacity : ℕ := 126
def empty_spaces : ℕ := 45

-- The problem to prove
theorem DVDs_already_in_book : already_in_book = total_capacity - empty_spaces :=
by
  let already_in_book := total_capacity - empty_spaces
  trivial

end DVDs_already_in_book_l420_420351


namespace bob_cleaning_time_l420_420083

theorem bob_cleaning_time (alice_time : ℕ) (h1 : alice_time = 25) (bob_ratio : ℚ) (h2 : bob_ratio = 2 / 5) : 
  bob_time = 10 :=
by
  -- Definitions for conditions
  let bob_time := bob_ratio * alice_time
  -- Sorry to represent the skipped proof
  sorry

end bob_cleaning_time_l420_420083


namespace derivative_of_f_l420_420162

variable (a x : ℝ)

def f (x : ℝ) (a : ℝ) : ℝ := (x + 2 * a) * (x - a)^2

theorem derivative_of_f :
  (deriv (f x a)) = 3 * (x^2 - a^2) := by
    sorry

end derivative_of_f_l420_420162


namespace math_problem_l420_420974

def f (x : ℝ) : ℝ := x^2 + 3
def g (x : ℝ) : ℝ := 2 * x + 5

theorem math_problem : f (g 4) - g (f 4) = 129 := by
  sorry

end math_problem_l420_420974


namespace probability_of_choosing_perfect_square_is_0_08_l420_420363

-- Definitions for the conditions
def n : ℕ := 100
def p : ℚ := 1 / 200
def probability (m : ℕ) : ℚ := if m ≤ 50 then p else 3 * p
def perfect_squares_before_50 : Finset ℕ := {1, 4, 9, 16, 25, 36, 49}
def perfect_squares_between_51_and_100 : Finset ℕ := {64, 81, 100}
def total_perfect_squares : Finset ℕ := perfect_squares_before_50 ∪ perfect_squares_between_51_and_100

-- Statement to prove that the probability of selecting a perfect square is 0.08
theorem probability_of_choosing_perfect_square_is_0_08 :
  (perfect_squares_before_50.card * p + perfect_squares_between_51_and_100.card * 3 * p) = 0.08 := 
by
  -- Adding sorry to skip the proof
  sorry

end probability_of_choosing_perfect_square_is_0_08_l420_420363


namespace train_length_calculation_l420_420755

def speed_km_per_hr : ℝ := 60
def time_sec : ℝ := 9
def length_of_train : ℝ := 150

theorem train_length_calculation :
  (speed_km_per_hr * 1000 / 3600) * time_sec = length_of_train := by
  sorry

end train_length_calculation_l420_420755


namespace constant_term_expansion_l420_420612

theorem constant_term_expansion :
  (x : ℝ) → (x^2 + 2) * (1/x - 1)^5 = constant_term * 1 :=
  by
  let f1 := x^2 + 2
  let f2 := (1/x - 1)^5
  let constant_term := -12
  sorry

end constant_term_expansion_l420_420612


namespace roots_of_quadratic_discriminant_positive_l420_420308

theorem roots_of_quadratic_discriminant_positive {a b c : ℝ} (h : b^2 - 4 * a * c > 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0) :=
by {
  sorry
}

end roots_of_quadratic_discriminant_positive_l420_420308


namespace exists_equal_radii_line_l420_420454

-- Defining the problem conditions
variables {A B C : Point}

-- Defining the necessary constructions and properties mentioned
-- Assumes that the line segment BD passes through vertex B to divide the triangle ABC
def equal_inscribed_circle_radii (A B C : Point) : Prop :=
  ∃ D : Point, 
    (let Δ1 := triangle A B D,
         Δ2 := triangle B C D,
         r1 := radius_inscribed Δ1,
         r2 := radius_inscribed Δ2 
     in r1 = r2)

-- The line passing through vertex B that divides the triangle
theorem exists_equal_radii_line (A B C : Point) : equal_inscribed_circle_radii A B C :=
  sorry

end exists_equal_radii_line_l420_420454


namespace sector_area_l420_420868

-- Define the properties and conditions
def perimeter_of_sector (r l : ℝ) : Prop :=
  l + 2 * r = 8

def central_angle_arc_length (r : ℝ) : ℝ :=
  2 * r

-- Theorem to prove the area of the sector
theorem sector_area (r : ℝ) (l : ℝ) 
  (h_perimeter : perimeter_of_sector r l) 
  (h_arc_length : l = central_angle_arc_length r) : 
  1 / 2 * l * r = 4 := 
by
  -- This is the place where the proof would go; we use sorry to indicate it's incomplete
  sorry

end sector_area_l420_420868


namespace correct_statement_about_biological_experiments_and_research_l420_420384

theorem correct_statement_about_biological_experiments_and_research :
  (B : ∀ s : string, s = "In the fruit fly crossbreeding experiment, Morgan and others used the hypothesis-deduction method to prove that genes are on chromosomes") :=
by
  sorry

end correct_statement_about_biological_experiments_and_research_l420_420384


namespace pyramid_volume_QEFGH_l420_420134

noncomputable def volume_of_pyramid (EF FG QE : ℝ) : ℝ :=
  (1 / 3) * EF * FG * QE

theorem pyramid_volume_QEFGH :
  let EF := 10
  let FG := 5
  let QE := 9
  volume_of_pyramid EF FG QE = 150 := by
  sorry

end pyramid_volume_QEFGH_l420_420134


namespace samantha_coins_worth_l420_420579

-- Define the conditions and the final question with an expected answer.
theorem samantha_coins_worth (n d : ℕ) (h1 : n + d = 30)
  (h2 : 10 * n + 5 * d = 5 * n + 10 * d + 120) :
  (5 * n + 10 * d) = 165 := 
sorry

end samantha_coins_worth_l420_420579


namespace triangle_angle_sum_l420_420069

theorem triangle_angle_sum {A B C : Type} 
  (angle_ABC : ℝ) (angle_BAC : ℝ) (angle_BCA : ℝ) (x : ℝ) 
  (h1: angle_ABC = 90) 
  (h2: angle_BAC = 3 * x) 
  (h3: angle_BCA = x + 10)
  : x = 20 :=
by
  sorry

end triangle_angle_sum_l420_420069


namespace complete_contingency_table_chi_sq_test_result_expected_value_X_l420_420593

noncomputable def probability_set := {x : ℚ // x ≥ 0 ∧ x ≤ 1}

variable (P : probability_set → probability_set)

-- Conditions from the problem
def P_A_given_not_B : probability_set := ⟨2 / 5, by norm_num⟩
def P_B_given_not_A : probability_set := ⟨5 / 8, by norm_num⟩
def P_B : probability_set := ⟨3 / 4, by norm_num⟩

-- Definitions related to counts and probabilities
def total_students : ℕ := 200
def male_students := P_A_given_not_B.val * total_students
def female_students := total_students - male_students
def score_exceeds_85 := P_B.val * total_students
def score_not_exceeds_85 := total_students - score_exceeds_85

-- Expected counts based on given probabilities
def male_score_not_exceeds_85 := P_A_given_not_B.val * score_not_exceeds_85
def female_score_not_exceeds_85 := score_not_exceeds_85 - male_score_not_exceeds_85
def male_score_exceeds_85 := male_students - male_score_not_exceeds_85
def female_score_exceeds_85 := female_students - female_score_not_exceeds_85

-- Chi-squared test independence 
def chi_squared := (total_students * (male_score_not_exceeds_85 * female_score_exceeds_85 - female_score_not_exceeds_85 * male_score_exceeds_85) ^ 2) / 
                    (male_students * female_students * score_not_exceeds_85 * score_exceeds_85)
def is_related : Prop := chi_squared > 10.828

-- Expected distributions and expectation of X
def P_X_0 := (1 / 4) ^ 2 * (1 / 3) ^ 2
def P_X_1 := 2 * (3 / 4) * (1 / 4) * (1 / 3) ^ 2 + 2 * (2 / 3) * (1 / 3) * (1 / 4) ^ 2
def P_X_2 := (3 / 4) ^ 2 * (1 / 3) ^ 2 + (1 / 4) ^ 2 * (2 / 3) ^ 2 + 2 * (2 / 3) * (1 / 3) * (3 / 4) * (1 / 4)
def P_X_3 := (3 / 4) ^ 2 * 2 * (2 / 3) * (1 / 3) + 2 * (3 / 4) * (1 / 4) * (2 / 3) ^ 2
def P_X_4 := (3 / 4) ^ 2 * (2 / 3) ^ 2
def expectation_X := 0 * P_X_0 + 1 * P_X_1 + 2 * P_X_2 + 3 * P_X_3 + 4 * P_X_4

-- Lean theorem statements for answers using the above definitions
theorem complete_contingency_table :
  male_score_not_exceeds_85 + female_score_not_exceeds_85 = score_not_exceeds_85 ∧
  male_score_exceeds_85 + female_score_exceeds_85 = score_exceeds_85 ∧
  male_students + female_students = total_students := sorry

theorem chi_sq_test_result :
  is_related = true := sorry

theorem expected_value_X :
  expectation_X = 17 / 6 := sorry

end complete_contingency_table_chi_sq_test_result_expected_value_X_l420_420593


namespace greatest_area_difference_l420_420666

theorem greatest_area_difference 
  (l w l' w' : ℕ) 
  (h1 : 2 * l + 2 * w = 160) 
  (h2 : 2 * l' + 2 * w' = 160) :
  abs ((l * w) - (l' * w')) ≤ 1521 := 
sorry

end greatest_area_difference_l420_420666


namespace blocks_divisible_l420_420591

theorem blocks_divisible (n : ℕ) 
  (weights : Fin n → ℕ)
  (h1 : ∀ i, 1 ≤ weights i ∧ weights i < n)
  (h2 : (∑ i, weights i) < 2 * n) :
  ∃ (S : Finset (Fin n)), (∑ i in S, weights i) = n :=
by
  sorry

end blocks_divisible_l420_420591


namespace total_price_after_conversion_and_tax_l420_420925

theorem total_price_after_conversion_and_tax:
    (original_price_book : ℝ) (original_price_bookmark : ℝ) (original_price_poster : ℝ)
    (conversion_rate : ℝ) (sales_tax : ℝ)
    (discount_book : ℝ) (increase_book : ℝ)
    (discount_bookmark : ℝ) (increase_bookmark : ℝ)
    (discount_poster : ℝ)
    (original_price_book = 100)
    (original_price_bookmark = 100)
    (original_price_poster = 100)
    (conversion_rate = 1.5)
    (sales_tax = 0.05)
    (discount_book = 0.2)
    (increase_book = 0.1)
    (discount_bookmark = 0.3)
    (increase_bookmark = 0.15)
    (discount_poster = 0.25) : 
    let book_price_after_sale := (original_price_book * (1 - discount_book)) * (1 + increase_book) in  
    let bookmark_price_after_sale := (original_price_bookmark * (1 - discount_bookmark)) * (1 + increase_bookmark) in
    let poster_price_after_sale := original_price_poster * (1 - discount_poster) in
    let total_price_currency_A := book_price_after_sale + bookmark_price_after_sale + poster_price_after_sale in
    let total_price_currency_B := total_price_currency_A / conversion_rate in
    let final_price_with_tax := total_price_currency_B * (1 + sales_tax) in
    final_price_with_tax = 170.1 := 
by 
  sorry

end total_price_after_conversion_and_tax_l420_420925


namespace train_arrival_time_west_coast_l420_420756

noncomputable def trainJourney : Prop :=
  let departureEastCoast : Nat := timestamp (2023, 5, 30, 5, 0) -- Using (year, month, day, hour, minute)
  let travelTime1 : Nat := 12 * 60 -- in minutes
  let layover : Nat := 3 * 60 -- in minutes
  let travelTime2 : Nat := (21 * 60) - layover
  let totalTravelTime : Nat := travelTime1 + travelTime2
  let arrivalEastCoast : Nat := departureEastCoast + totalTravelTime
  let timezoneDifference : Nat := 2 * 60 -- 2 hours in minutes
  let arrivalWestCoast : Nat := arrivalEastCoast - timezoneDifference
  arrivalWestCoast = timestamp (2023, 5, 31, 9, 0)

theorem train_arrival_time_west_coast :
  trainJourney :=
by
  sorry

end train_arrival_time_west_coast_l420_420756


namespace min_dist_PQ_l420_420864

noncomputable theory

open Real

def point := (ℝ × ℝ)

def A : point := (0, 2) -- Point A at (0,2)
def line (x y : ℝ) : Prop := x + y + 2 = 0 -- Line equation x + y + 2 = 0
def circle (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y = 0 -- Circle equation x^2 + y^2 - 4*x - 2*y = 0

def distance (p1 p2 : point) : ℝ := sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def PQ_minimizer : Prop :=
∀ P Q : point, 
  line P.1 P.2 →
  circle Q.1 Q.2 →
  distance P A + distance P Q ≥ 2 * sqrt 5

theorem min_dist_PQ : PQ_minimizer :=
by
  sorry

end min_dist_PQ_l420_420864


namespace triangle_hypotenuse_length_l420_420698

-- Defining the conditions and the goal to prove the hypotenuse length of the given right triangle.
theorem triangle_hypotenuse_length
(x y : ℝ)
(vol1 : (1 / 3) * Real.pi * y^2 * x = 1000 * Real.pi) 
(vol2 : (1 / 3) * Real.pi * x^2 * y = 2250 * Real.pi) : 
sqrt (x^2 + y^2) ≈ 39.08 := 
sorry

end triangle_hypotenuse_length_l420_420698


namespace find_x_l420_420016

theorem find_x (x : ℕ) :
  (let digits := {1, 4, 5, x} 
   in digits.card = 4 ∧ ∀ d, d ∈ digits → d ≠ 0 → d ∈ {1, 4, 5, x} ∧ 
     (∑ i in permutations digits, (∑ j in i, j)) = 288) → 
  x = 2 :=
by
  sorry

end find_x_l420_420016


namespace pyramid_face_area_total_l420_420231

theorem pyramid_face_area_total 
  (base_edge : ℝ) (lateral_edge : ℝ) 
  (h_base_edge : base_edge = 8) 
  (h_lateral_edge : lateral_edge = 7) : 
  4 * (1 / 2 * base_edge * real.sqrt (lateral_edge^2 - (base_edge / 2)^2)) = 16 * real.sqrt 33 :=
by
  sorry

end pyramid_face_area_total_l420_420231


namespace rectangular_equation_C1_intersection_point_polar_coordinates_l420_420522

-- Definition of the parametric equation of curve C1
def parametric_equation_C1 (θ : ℝ) : ℝ × ℝ :=
  let x := 2 + cos θ
  let y := sin θ
  (x, y)

-- Prove the rectangular coordinate equation of curve C1
theorem rectangular_equation_C1 (x y : ℝ) (θ : ℝ) :
  parametric_equation_C1 θ = (x, y) → (x - 2)^2 + y^2 = 1 :=
  sorry

-- Definition of the polar coordinate equation of curve C2
def polar_equation_C2 : ℝ → Prop :=
  λ θ, θ = π / 6

-- Define a function to convert rectangular coordinates to polar coordinates
def rectangular_to_polar (x y : ℝ) : ℝ × ℝ :=
  (sqrt (x^2 + y^2), atan2 y x)

-- Given the intersection point, prove its polar coordinates are (√3, π/6)
theorem intersection_point_polar_coordinates (x y : ℝ) :
  (x - 2)^2 + y^2 = 1 →
  y = (sqrt 3) / 3 * x →
  rectangular_to_polar x y = (sqrt 3, π / 6) :=
  sorry

end rectangular_equation_C1_intersection_point_polar_coordinates_l420_420522


namespace summation_series_lt_one_l420_420787

theorem summation_series_lt_one :
  (∑ k in finset.range 2012, (k + 1 : ℝ) / ((k + 2)! : ℝ)) < 1 :=
by sorry

end summation_series_lt_one_l420_420787


namespace cats_left_l420_420367

theorem cats_left (siamese_cats : ℕ) (house_cats : ℕ) (cats_sold : ℕ) 
  (h1 : siamese_cats = 13) (h2 : house_cats = 5) (h3 : cats_sold = 10) : 
  siamese_cats + house_cats - cats_sold = 8 :=
by
  sorry

end cats_left_l420_420367


namespace pyramid_triangular_face_area_l420_420214

theorem pyramid_triangular_face_area 
  (base : ℝ) (lateral : ℝ)
  (h_base : base = 8) (h_lateral : lateral = 7) :
  let height := sqrt (lateral^2 - (base / 2)^2) in
  4 * (1 / 2 * base * height) = 16 * sqrt 33 := 
by
  sorry

end pyramid_triangular_face_area_l420_420214


namespace max_plus_min_eq_four_l420_420108

theorem max_plus_min_eq_four {g : ℝ → ℝ} (h_odd_function : ∀ x, g (-x) = -g x)
  (M m : ℝ) (h_f : ∀ x, 2 + g x ≤ M) (h_f' : ∀ x, m ≤ 2 + g x) :
  M + m = 4 :=
by
  sorry

end max_plus_min_eq_four_l420_420108


namespace projection_of_a_onto_b_l420_420897

noncomputable def projection_coords (a b : ℝ × ℝ × ℝ) :=
  let dot_product := a.1 * b.1 + a.2 * b.2 + a.3 * b.3
  let b_magnitude_sq := b.1 ^ 2 + b.2 ^ 2 + b.3 ^ 2
  let scale := dot_product / b_magnitude_sq
  (scale * b.1, scale * b.2, scale * b.3)

theorem projection_of_a_onto_b :
  projection_coords (2, 3, 0) (1, 0, 3) = (1 / 5, 0, 3 / 5) :=
by
  sorry

end projection_of_a_onto_b_l420_420897


namespace book_price_range_l420_420705

variable (x : ℝ) -- Assuming x is a real number

theorem book_price_range 
    (hA : ¬(x ≥ 20)) 
    (hB : ¬(x ≤ 15)) : 
    15 < x ∧ x < 20 := 
by
  sorry

end book_price_range_l420_420705


namespace slower_train_crosses_faster_in_36_seconds_l420_420712

-- Define the conditions of the problem
def speed_fast_train_kmph : ℚ := 110
def speed_slow_train_kmph : ℚ := 90
def length_fast_train_km : ℚ := 1.10
def length_slow_train_km : ℚ := 0.90

-- Convert speeds to m/s
def speed_fast_train_mps : ℚ := speed_fast_train_kmph * (1000 / 3600)
def speed_slow_train_mps : ℚ := speed_slow_train_kmph * (1000 / 3600)

-- Relative speed when moving in opposite directions
def relative_speed_mps : ℚ := speed_fast_train_mps + speed_slow_train_mps

-- Convert lengths to meters
def length_fast_train_m : ℚ := length_fast_train_km * 1000
def length_slow_train_m : ℚ := length_slow_train_km * 1000

-- Combined length of both trains in meters
def combined_length_m : ℚ := length_fast_train_m + length_slow_train_m

-- Time taken for the slower train to cross the faster train
def crossing_time : ℚ := combined_length_m / relative_speed_mps

theorem slower_train_crosses_faster_in_36_seconds :
  crossing_time = 36 := by
  sorry

end slower_train_crosses_faster_in_36_seconds_l420_420712


namespace find_positive_integer_cube_root_divisible_by_21_l420_420811

theorem find_positive_integer_cube_root_divisible_by_21 (m : ℕ) (h1: m = 735) :
  m % 21 = 0 ∧ 9 < (m : ℝ)^(1/3) ∧ (m : ℝ)^(1/3) < 9.1 :=
by {
  sorry
}

end find_positive_integer_cube_root_divisible_by_21_l420_420811


namespace heptagon_symmetry_count_l420_420900

def regular_heptagon (H : Type) [heptagon H] : Prop :=
  ∀ (a b : vertex H), edge_length H a b = edge_length H b a ∧ angle H a = angle H b

def irregular_heptagon (H : Type) [heptagon H] : Prop :=
  ¬ regular_heptagon H

def partially_symmetric_heptagon (H : Type) [heptagon H] : Prop :=
  ∃ (sym_line : line H), 
    ∀ (a b : vertex H), 
      (reflection_through_line sym_line a = b) → 
      (edge_length H a b = edge_length H b a ∧ angle H a = angle H b)

theorem heptagon_symmetry_count (H : Type) [heptagon H] : 
  (regular_heptagon H → symmetry_count H = 7) ∧
  (irregular_heptagon H → symmetry_count H = 0) ∧
  (partially_symmetric_heptagon H → symmetry_count H = 1) :=
sorry

end heptagon_symmetry_count_l420_420900


namespace det_B_squared_minus_3B_l420_420889

open Matrix
open_locale matrix big_operators

variable (R : Type*) [CommRing R]

def B : Matrix (Fin 2) (Fin 2) R :=
  ![![2, 4], ![3, 2]]

theorem det_B_squared_minus_3B :
  det (B R ^ 2 - 3 • B R) = (88 : R) := by
  sorry

end det_B_squared_minus_3B_l420_420889


namespace prove_value_range_for_a_l420_420882

noncomputable def f (x a : ℝ) : ℝ :=
  (x^2 + a*x + 7 + a) / (x + 1)

noncomputable def g (x : ℝ) : ℝ := 
  - ((x + 1) + (8 / (x + 1))) + 6

theorem prove_value_range_for_a (a : ℝ) :
  (∀ x : ℕ, x > 0 → f x a ≥ 4) ↔ (a ≥ 1 / 3) :=
sorry

end prove_value_range_for_a_l420_420882
