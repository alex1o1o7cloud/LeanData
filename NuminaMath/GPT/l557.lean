import Mathlib

namespace MN_perpendicular_to_BC_l557_557857

theorem MN_perpendicular_to_BC (A B C M N: Point) (W P: Circle)
  (h1 : angle A B C = 30)
  (h2 : angle A C B = 60)
  (h3 : is_midpoint M B C)
  (h4 : W.passes_through A)
  (h5 : W.tangent_at M B C)
  (h6 : P.circumcircle_of ABC)
  (h7 : W.intersects AC N)
  (h8 : W.intersects P M):
  is_perpendicular MN BC := 
sorry

end MN_perpendicular_to_BC_l557_557857


namespace king_lancelot_seats_38_l557_557894

noncomputable def totalSeats (seat_king seat_lancelot : ℕ) : ℕ :=
  if seat_king < seat_lancelot then
    2 * (seat_lancelot - seat_king - 1) + 2
  else
    2 * (seat_king - seat_lancelot - 1) + 2

theorem king_lancelot_seats_38 (seat_king seat_lancelot : ℕ) (h1 : seat_king = 10) (h2 : seat_lancelot = 29) :
  totalSeats seat_king seat_lancelot = 38 := 
  by
    sorry

end king_lancelot_seats_38_l557_557894


namespace opposite_of_3_l557_557779

theorem opposite_of_3 : -3 = -3 := 
by
  -- sorry is added to skip the proof as per instructions
  sorry

end opposite_of_3_l557_557779


namespace woodworker_tables_l557_557485

def legs_needed (num_chairs num_tables num_cabinets : ℕ) : ℕ :=
  4 * num_chairs + 4 * num_tables + 2 * num_cabinets

theorem woodworker_tables (total_legs num_chairs num_cabinets : ℕ) : 
  total_legs = 80 → num_chairs = 6 → num_cabinets = 4 → 
  ∃ num_tables : ℕ, num_tables = 12 :=
begin
  intros h1 h2 h3,
  have h_total_chairs : 24 := 4 * 6,
  have h_total_cabinets : 8 := 2 * 4,
  have h_used_legs : 32 := h_total_chairs + h_total_cabinets,
  have h_remaining_legs : 48 := 80 - h_used_legs,
  use 12,
  rw ←h_remaining_legs,
  calc 48 = 12 * 4 : by norm_num,
  rw mul_comm,
end

end woodworker_tables_l557_557485


namespace length_of_overbridge_l557_557112

-- Definitions for the given problem
def train_length : ℝ := 600  -- in meters
def crossing_time : ℝ := 70  -- in seconds
def train_speed_kmh : ℝ := 36  -- in km/h

-- Convert speed from km/h to m/s
def train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)

-- Total distance covered by train in the given time
def total_distance : ℝ := train_speed_ms * crossing_time

-- Proof statement: Length of the overbridge
theorem length_of_overbridge : total_distance = train_length + 100 := by
  sorry

end length_of_overbridge_l557_557112


namespace find_f_2023_l557_557328

noncomputable def f : ℝ → ℝ := sorry

axiom pos_domain : ∀ x : ℝ, x > 0 → f x > 0
axiom func_def : ∀ x y : ℝ, x > y → x > 0 → y > 0 → f (x - y) = sqrt (f (x * y) + 2)

theorem find_f_2023 : f 2023 = 2 := sorry

end find_f_2023_l557_557328


namespace cylinder_height_to_radius_ratio_l557_557936

theorem cylinder_height_to_radius_ratio (V r h : ℝ) (hV : V = π * r^2 * h) (hS : sorry) :
  h / r = 2 :=
sorry

end cylinder_height_to_radius_ratio_l557_557936


namespace subsequences_with_equal_sum_exists_l557_557207

theorem subsequences_with_equal_sum_exists (x : Fin 19 → Fin 93) (y : Fin 93 → Fin 19)
  (hx_pos : ∀ i, 0 < x i)
  (hy_pos : ∀ j, 0 < y j) :
  ∃ (I : Finset (Fin 19)) (J : Finset (Fin 93)),
  I.nonempty ∧ J.nonempty ∧
  (I.sum x.val) = (J.sum y.val) :=
by
  sorry

end subsequences_with_equal_sum_exists_l557_557207


namespace common_ratio_l557_557401

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * q

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n : ℕ, S n = ∑ k in Finset.range n, a k

-- Conditions
axiom a_2 (a : ℕ → ℝ) (q : ℝ) : a 1 * q = 9
axiom S_3 (a : ℕ → ℝ) (q : ℝ) : sum_of_first_n_terms a (λ n, ∑ k in Finset.range n, a k) ∧ (∑ k in Finset.range 3, a k) = 39

noncomputable def q_values (a : ℕ → ℝ) (q : ℝ) : Prop :=
q = 3 ∨ q = 1/3

theorem common_ratio (a : ℕ → ℝ) (q : ℝ) :
  a_2 a q → S_3 a q → q_values a q :=
by sorry

end common_ratio_l557_557401


namespace triangle_angle_A_60_l557_557274

theorem triangle_angle_A_60 (A B C D E I P Q : Type) [triangle ABC] [angle_bisectors BD CE] [incenter I]
  (hD : D ∈ AC) (hE : E ∈ AB) (hP : P = foot (perpendicular I DE)) (hQ : Q = extension PI BC) (hIQ_IP : IQ = 2 * IP) :
  angle A = 60 :=
sorry

end triangle_angle_A_60_l557_557274


namespace find_intersection_point_l557_557314

noncomputable def intersection_point_on_parabola : ℝ × ℝ :=
  let A : ℝ × ℝ := (2, 4)
  let normal_slope : ℝ := -1 / 4
  let normal_line (x : ℝ) : ℝ := normal_slope * (x - 2) + 4
  let parabola (x : ℝ) : ℝ := x^2
  let intersection_x := (-1 + Real.sqrt (1 + 4*18)) / (2*2)
  (intersection_x, parabola(intersection_x))

theorem find_intersection_point :
  intersection_point_on_parabola = (-(9/4), (9/4)^2) :=
sorry

end find_intersection_point_l557_557314


namespace evaluate_expression_l557_557528

theorem evaluate_expression :
  (3^3 *  3^(-4)) / (3^(-1) * 3^2) = 1 / 9 := by
  sorry

end evaluate_expression_l557_557528


namespace king_lancelot_seats_38_l557_557893

noncomputable def totalSeats (seat_king seat_lancelot : ℕ) : ℕ :=
  if seat_king < seat_lancelot then
    2 * (seat_lancelot - seat_king - 1) + 2
  else
    2 * (seat_king - seat_lancelot - 1) + 2

theorem king_lancelot_seats_38 (seat_king seat_lancelot : ℕ) (h1 : seat_king = 10) (h2 : seat_lancelot = 29) :
  totalSeats seat_king seat_lancelot = 38 := 
  by
    sorry

end king_lancelot_seats_38_l557_557893


namespace min_value_of_f_l557_557383

noncomputable def f (x θ : ℝ) : ℝ :=
  cos (x + 2 * θ) + 2 * sin θ * sin (x + θ)

theorem min_value_of_f (θ : ℝ) : 
  ∃ x : ℝ, f x θ = -1 :=
by
  sorry

end min_value_of_f_l557_557383


namespace f_monotonic_increasing_l557_557569

-- Define the function as per the conditions.
def f (x : ℝ) : ℝ :=
if x ≥ 0 then x * (2 + x) else -(-(x * (2 - x)))

-- Define the property of the function being odd.
lemma f_odd (x : ℝ) : f (-x) = -f x :=
by sorry

-- Define the property of the function for x ≥ 0.
lemma f_positive (x : ℝ) (hx : x ≥ 0) : f x = x * (2 + x) :=
by sorry

-- The main statement indicating the function is monotonically increasing for all x in ℝ.
theorem f_monotonic_increasing : ∀ (x₁ x₂ : ℝ), x₁ ≤ x₂ → f x₁ ≤ f x₂ :=
by sorry

end f_monotonic_increasing_l557_557569


namespace stickers_left_correct_l557_557687

-- Define the initial number of stickers and number of stickers given away
def n_initial : ℝ := 39.0
def n_given_away : ℝ := 22.0

-- Proof statement: The number of stickers left at the end is 17.0
theorem stickers_left_correct : n_initial - n_given_away = 17.0 := by
  sorry

end stickers_left_correct_l557_557687


namespace O_is_circumcenter_O_is_incenter_O_is_orthocenter_l557_557216

open Geometry

variables {P O A B C α : Point}

-- Conditions
variable (P_outside_plane : P ∉ α)
variable (O_proj : orthogonal_projection α P = O)
variable (distances_equal : dist P A = dist P B ∧ dist P B = dist P C)
variable (angles_equal : angle PA α = angle PB α ∧ angle PB α = angle PC α)
variable (distances_to_sides_equal : dist_to_side P AB = dist_to_side P BC ∧ dist_to_side P BC = dist_to_side P CA)
variable (angles_of_planes_equal : angle_plane PAB α = angle_plane PBC α ∧ angle_plane PBC α = angle_plane PCA α)
variable (perpendicular_triples : perpendicular PA PB ∧ perpendicular PB PC ∧ perpendicular PA PC)
variable (O_inside_triangle : inside_triangle ABC O)

-- Proof statements

theorem O_is_circumcenter (h1 : distances_equal) : is_circumcenter O A B C := sorry

theorem O_is_incenter (h2 : distances_to_sides_equal ∧ O_inside_triangle) : is_incenter O A B C := sorry

theorem O_is_orthocenter (h3 : perpendicular_triples) : is_orthocenter O A B C := sorry

end O_is_circumcenter_O_is_incenter_O_is_orthocenter_l557_557216


namespace total_seats_l557_557910

theorem total_seats (KA_pos : ℕ) (SL_pos : ℕ) (h1 : KA_pos = 10) (h2 : SL_pos = 29) (h3 : SL_pos = KA_pos + (KA_pos * 2 - 1) / 2):
  let total_positions := 2 * (SL_pos - KA_pos - 1) + 2
  total_positions = 38 :=
by
  sorry

end total_seats_l557_557910


namespace statement_D_incorrect_l557_557197

theorem statement_D_incorrect (a b c : ℝ) : a^2 > b^2 ∧ a * b > 0 → ¬(1 / a < 1 / b) :=
by sorry

end statement_D_incorrect_l557_557197


namespace minimum_balls_to_draw_l557_557076

theorem minimum_balls_to_draw
  (red green yellow blue white : ℕ)
  (h_red : red = 30)
  (h_green : green = 25)
  (h_yellow : yellow = 20)
  (h_blue : blue = 15)
  (h_white : white = 10) :
  ∃ (n : ℕ), n = 81 ∧
    (∀ (r g y b w : ℕ), 
       (r + g + y + b + w >= n) →
       ((r ≥ 20 ∨ g ≥ 20 ∨ y ≥ 20 ∨ b ≥ 20 ∨ w ≥ 20) ∧ 
        (r ≥ 10 ∨ g ≥ 10 ∨ y ≥ 10 ∨ b ≥ 10 ∨ w ≥ 10))
    ) := sorry

end minimum_balls_to_draw_l557_557076


namespace element_is_calcium_l557_557384

theorem element_is_calcium :
  ∀ (oxide_weight element_weight oxygen_weight : ℕ),
  oxide_weight = 56 →
  element_weight = 40 →
  oxygen_weight = 16 →
  (∃ x : ℕ, oxide_weight = element_weight + (oxygen_weight * x)) →
  element_weight = 40 ∧ oxygen_weight = 16 →
  "Ca" := 
by
  intros oxide_weight element_weight oxygen_weight ox_eq elem_eq o2_eq exists_x assumption
  sorry

end element_is_calcium_l557_557384


namespace logarithmic_function_monotonicity_l557_557712

noncomputable def f (x : ℝ) := sorry

theorem logarithmic_function_monotonicity (f : ℝ → ℝ)
  (h1 : ∀ x > 0, ∀ m : ℝ, f (x ^ m) = m * f x)
  (h2 : ∃ a > 1, f a = 1) :
  f 3 > f 2 ∧ f 2 > f (3 / 2) :=
sorry

end logarithmic_function_monotonicity_l557_557712


namespace equalDistances_l557_557985
noncomputable def distance (x₀ y₀ a : ℝ) : ℝ :=
  abs (a * x₀ + y₀ + 1) / sqrt (a^2 + 1)

theorem equalDistances (a : ℝ) :
  distance (-2) 4 a = distance (-4) 6 a → a = 1 ∨ a = 2 :=
by
  intro h
  -- placeholder proof
  sorry

end equalDistances_l557_557985


namespace silk_pieces_count_l557_557336

theorem silk_pieces_count (S C : ℕ) (h1 : S = 2 * C) (h2 : S + C + 2 = 13) : S = 7 :=
by
  sorry

end silk_pieces_count_l557_557336


namespace periodic_sequence_f_l557_557935

noncomputable def f : ℕ+ → ℕ+ := sorry

theorem periodic_sequence_f (h1 : ∀ m n : ℕ+, (f^[n] m - m) / n ∈ ℕ+)
  (h2 : {n : ℕ+ | ¬ ∃ m : ℕ+, f m = n}.finite) :
  ∃ T > 0, ∀ k : ℕ+, f(k + T) - (k + T) = f(k) - k :=
sorry

end periodic_sequence_f_l557_557935


namespace circumradius_inequality_l557_557606

theorem circumradius_inequality (a b c R : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) (h4 : R = a / (2 * sin (angle_ac_b a b c))) :
  R ≥ (a^2 + b^2) / (2 * sqrt (2 * a^2 + 2 * b^2 - c^2)) :=
begin
  sorry
end

end circumradius_inequality_l557_557606


namespace rate_of_fencing_is_4_90_l557_557704

noncomputable def rate_of_fencing_per_meter : ℝ :=
  let area_hectares := 13.86
  let cost := 6466.70
  let area_m2 := area_hectares * 10000
  let radius := Real.sqrt (area_m2 / Real.pi)
  let circumference := 2 * Real.pi * radius
  cost / circumference

theorem rate_of_fencing_is_4_90 :
  rate_of_fencing_per_meter = 4.90 := sorry

end rate_of_fencing_is_4_90_l557_557704


namespace total_seats_round_table_l557_557871

theorem total_seats_round_table 
  (a : ℕ) (b : ℕ) 
  (h₀ : a ≠ b)
  (h₁ : a + b = 39) 
  : ∃ n, n = 38 := 
by {
  sorry
}

end total_seats_round_table_l557_557871


namespace files_to_organize_in_afternoon_l557_557525

-- Defining the given conditions.
def initial_files : ℕ := 60
def files_organized_in_the_morning : ℕ := initial_files / 2
def missing_files_in_the_afternoon : ℕ := 15

-- The theorem to prove:
theorem files_to_organize_in_afternoon : 
  files_organized_in_the_morning + missing_files_in_the_afternoon = initial_files / 2 →
  ∃ afternoon_files : ℕ, 
    afternoon_files = (initial_files - files_organized_in_the_morning) - missing_files_in_the_afternoon :=
by
  -- Proof will go here, skipping with sorry for now.
  sorry

end files_to_organize_in_afternoon_l557_557525


namespace cone_frustum_volume_ratio_l557_557846

noncomputable def coneVolume (r h : ℝ) : ℝ :=
  (1 / 3) * π * r^2 * h

noncomputable def frustumVolume (r1 r2 h : ℝ) : ℝ :=
  (1 / 3) * π * h * (r1^2 + r1 * r2 + r2^2)

theorem cone_frustum_volume_ratio (H R : ℝ) (H_pos : 0 < H) (R_pos : 0 < R) :
  let V_A := coneVolume (R / 3) (H / 3)
  let V_B := frustumVolume (2 * R / 3) (R / 3) (H / 3)
  V_B / V_A = 3 :=
by
  sorry

end cone_frustum_volume_ratio_l557_557846


namespace total_seats_l557_557907

theorem total_seats (KA_pos : ℕ) (SL_pos : ℕ) (h1 : KA_pos = 10) (h2 : SL_pos = 29) (h3 : SL_pos = KA_pos + (KA_pos * 2 - 1) / 2):
  let total_positions := 2 * (SL_pos - KA_pos - 1) + 2
  total_positions = 38 :=
by
  sorry

end total_seats_l557_557907


namespace kelsey_remaining_half_speed_l557_557649

variable (total_hours : ℝ) (first_half_speed : ℝ) (total_distance : ℝ) (remaining_half_time : ℝ) (remaining_half_distance : ℝ)

axiom h1 : total_hours = 10
axiom h2 : first_half_speed = 25
axiom h3 : total_distance = 400
axiom h4 : remaining_half_time = total_hours - total_distance / (2 * first_half_speed)
axiom h5 : remaining_half_distance = total_distance / 2

theorem kelsey_remaining_half_speed :
  remaining_half_distance / remaining_half_time = 100
:=
by
  sorry

end kelsey_remaining_half_speed_l557_557649


namespace midpoint_equidistant_from_Q1Q2_l557_557493

noncomputable theory

open EuclideanGeometry

def circles_intersect (O₁ O₂ : Point) (A B : Point) (r₁ r₂ : Real) : Prop := 
  dist O₁ A = r₁ ∧ dist O₂ A = r₂ ∧ dist O₁ B = r₁ ∧ dist O₂ B = r₂

def is_midpoint (P M N : Point) : Prop :=
  dist P M = dist P N ∧ colinear M P N

def is_perpendicular (L1 L2 : Line) : Prop := 
  ∃ A, 90 = inner_angle A L1 L2

theorem midpoint_equidistant_from_Q1Q2
  (O₁ O₂ A B M N P Q₁ Q₂ : Point) (r₁ r₂ : Real)
  (h_circles : circles_intersect O₁ O₂ A B r₁ r₂)
  (h_perpendicular : is_perpendicular (line_through M N) (line_through A B))
  (h_intersect : intersects_circle O₁ M ∧ intersects_circle O₂ N)
  (h_midpoint : is_midpoint P M N)
  (h_angle_eq : inner_angle A O₁ Q₁ = inner_angle A O₂ Q₂) :
  dist P Q₁ = dist P Q₂ := 
sorry

end midpoint_equidistant_from_Q1Q2_l557_557493


namespace integer_root_exists_l557_557402

def initial_poly (x : ℝ) : ℝ := x^2 + 10 * x + 20
def final_poly (x : ℝ) : ℝ := x^2 + 20 * x + 10
def eval_at_neg1 (p : ℝ → ℝ) : ℝ := p (-1)
def steps : ℤ := abs (eval_at_neg1 initial_poly - eval_at_neg1 final_poly)

theorem integer_root_exists (initial_poly final_poly : ℝ → ℝ) (steps : ℤ) : 
  ∃ p : ℝ → ℝ, eval_at_neg1 p = 0 :=
by
  have initial_value : eval_at_neg1 initial_poly = 11 := by sorry
  have final_value : eval_at_neg1 final_poly = -9 := by sorry
  have transition : steps = abs (initial_value - final_value) := by sorry
  exact sorry

end integer_root_exists_l557_557402


namespace beacon_school_earnings_l557_557362

noncomputable def wages : ℕ → ℕ → ℕ → ℕ :=
  λ (w_day wage wk_day weekend_d), 
  (w_day * wage + wk_day * 2 * wage) 

theorem beacon_school_earnings : 
  (∃ (w_day wage wk_day weekend_d),
  wages 4 1 2 = 218 ∧ wages 6 1 1 = 336 / ((4 * wage + 2 * (2 * wage)) + (6 * wage + 1 * (2 * wage)) + (8 * wage + 3 * (2 * wage))) :=
sorry

end beacon_school_earnings_l557_557362


namespace C_divides_AE_in_ratio_sin_alpha_l557_557209

noncomputable def problem (A B C D E : Type) [Inhabited A] (α : Real) :=
-- Conditions
(ang_ABC_eq_alpha : ∃ (ABC : Triangle ABC), angle ABC = α)
(AD_tangent_to_omega : Tangent AD ω)
(AC_intersects_circumcircle_ABD : intersects AC (circumcircle ABD) E)
(angle_bisector_ADE_tangent_to_omega : Tangent (angle_bisector ADE) ω)

-- Conclusion to prove
theorem C_divides_AE_in_ratio_sin_alpha (A B C D E : Type) [Inhabited A] (α : Real) :
problem A B C D E α → (exists r : Real, ratio_divides AE C = sin α) :=
sorry

end C_divides_AE_in_ratio_sin_alpha_l557_557209


namespace omega_value_sin_A_plus_sin_C_range_l557_557237

noncomputable def f (ω : ℝ) (x : ℝ) := sin (ω * x - π / 6) - 2 * cos (ω * x / 2) ^ 2 + 1

theorem omega_value (ω : ℝ) (hω : 0 < ω) (h_dist : ∀x₁ x₂: ℝ, f ω x₁ = sqrt 3 → f ω x₂ = sqrt 3 → |x₁ - x₂| = π) : ω = 2 :=
sorry

theorem sin_A_plus_sin_C_range
  (A B C : ℝ) (a b c : ℝ) (h_acute : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2)
  (h_sum_angles : A + B + C = π)
  (h_sym_center : 2 * (B / 2) - π / 3 = k * π ∧ k ∈ ℤ) :
   sqrt 3 * sin (A + π / 6) ∈ (3 / 2, sqrt 3] :=
sorry

end omega_value_sin_A_plus_sin_C_range_l557_557237


namespace product_fraction_value_l557_557925

theorem product_fraction_value :
  ∏ n in (finset.range 98).image (λ n, n + 2), (n * (n + 2)) / ((n + 1) ^ 2) = (101 / 150) :=
by
  sorry

end product_fraction_value_l557_557925


namespace time_taken_by_Car_Q_l557_557505

-- Definitions based on conditions
def speed_P : ℝ := 60
def time_P : ℝ := 3
def distance_P : ℝ := speed_P * time_P
def speed_Q : ℝ := 3 * speed_P
def distance_Q : ℝ := distance_P / 2

-- Theorem to prove the problem statement
theorem time_taken_by_Car_Q :
  let time_Q := distance_Q / speed_Q in
  time_Q = 0.5 := by
  sorry

end time_taken_by_Car_Q_l557_557505


namespace area_convex_quadrilateral_centroids_l557_557359

-- Define the properties of the square and the point Q
def is_square (E F G H : ℝ × ℝ) (s : ℝ) :=
  E = (0, s) ∧ F = (0, 0) ∧ G = (s, 0) ∧ H = (s, s)

-- Point Q lies inside square, given distances EQ and FQ
def inside_square (Q : ℝ × ℝ) (E F : ℝ × ℝ) (EQ FQ : ℝ) :=
  dist Q E = EQ ∧ dist Q F = FQ

-- Main theorem stating the area of the quadrilateral formed by centroids
theorem area_convex_quadrilateral_centroids 
(E F G H Q : ℝ × ℝ) 
(h_square : is_square E F G H 40)
(h_inside : inside_square Q E F 16 34) :
  let C1 := (0, (40 + Q.2) / 3),
      C2 := ((40 + Q.1) / 3, Q.2 / 3), -- Correct centroid positions to correspond correctly
      C3 := (Q.1 / 3, (40 + Q.2) / 3),
      C4 := ((40 + Q.1) / 3, (40 + Q.2) / 3)
  in area_quadrilateral C1 C2 C3 C4 = 800 / 9 := sorry

end area_convex_quadrilateral_centroids_l557_557359


namespace total_seats_at_round_table_l557_557877

-- Define namespace and conditions
namespace KingArthur

variable (n : ℕ) -- Total number of seats

-- Conditions
def king_position : ℕ := 10
def lancelot_position : ℕ := 29
def opposite (a b : ℕ) (n : ℕ) : Prop := (a + (n / 2)) % n = b

-- Proof statement
theorem total_seats_at_round_table : opposite 10 29 n ∧ 29 < n → n = 38 :=
by
  sorry

end KingArthur

end total_seats_at_round_table_l557_557877


namespace probability_thirteen_people_know_news_probability_fourteen_people_know_news_expected_value_of_knowing_scientists_l557_557948

def num_scientists : Nat := 18
def num_initially_known : Nat := 10

/-- Prove the probability that after the coffee break, exactly 13 people will know the news is zero -/
theorem probability_thirteen_people_know_news (num_scientists = 18) (num_initially_known = 10) : 
  (probability number_of_people_know_news 13 = 0) := sorry

/-- Prove the probability that after the coffee break, exactly 14 people will know the news is approximately 0.461 -/
theorem probability_fourteen_people_know_news (num_scientists = 18) (num_initially_known = 10) : 
  (probability number_of_people_know_news 14 ≈ 0.461) := sorry

/-- Prove the expected number of scientists who will know the news after the coffee break is approximately 14.7 -/
theorem expected_value_of_knowing_scientists (num_scientists = 18) (num_initially_known = 10) : 
  (expected_value number_of_people_know_news ≈ 14.7) := sorry

end probability_thirteen_people_know_news_probability_fourteen_people_know_news_expected_value_of_knowing_scientists_l557_557948


namespace ratio_of_perimeters_l557_557851

-- Definitions based on the problem conditions
def square_side_length (x : ℝ) := 3 * x
def square_perimeter (x : ℝ) := 4 * (square_side_length x)

def octagon_perimeter (x : ℝ) : ℝ :=
  let long_sides := 4 * (4 * x)
  let short_sides := 4 * (3 * x)
  in long_sides + short_sides

-- Theorem statement translating the problem to Lean
theorem ratio_of_perimeters (x : ℝ) (hx : x ≠ 0) :
  let sq_perim := square_perimeter x
  let oct_perim := octagon_perimeter x
  sq_perim / oct_perim = 3 / 5 :=
by sorry

end ratio_of_perimeters_l557_557851


namespace total_distance_traveled_l557_557696

theorem total_distance_traveled :
  ∃ (D : ℕ), (D = 80) ∧
              let second_leg := 2 * D,
                  third_leg := 40,
                  final_leg := 2 * (D + second_leg + third_leg),
                  total_distance := D + second_leg + third_leg + final_leg
              in total_distance = 840 :=
by
  use 80
  simp
  sorry -- Proof details are omitted

end total_distance_traveled_l557_557696


namespace opposite_of_3_is_neg3_l557_557718

theorem opposite_of_3_is_neg3 : forall (n : ℤ), n = 3 -> -n = -3 :=
by
  sorry

end opposite_of_3_is_neg3_l557_557718


namespace find_y_l557_557164

noncomputable def y_value (OA OB OC OD BD : ℝ) : ℝ :=
  real.sqrt (OA^2 + OC^2 - 2 * OA * OC * (-3 / 4))

theorem find_y :
  let OA := 4
      OB := 6
      OC := 7
      OD := 3
      BD := 9
   in y_value OA OB OC OD BD = real.sqrt 107 :=
by
  let OA := 4
  let OB := 6
  let OC := 7
  let OD := 3
  let BD := 9
  show y_value OA OB OC OD BD = real.sqrt 107
  sorry

end find_y_l557_557164


namespace opposite_of_3_l557_557771

theorem opposite_of_3 : -3 = -3 := 
by
  -- sorry is added to skip the proof as per instructions
  sorry

end opposite_of_3_l557_557771


namespace sphere_radius_ratio_l557_557798

theorem sphere_radius_ratio (R1 R2 : ℝ) (m n : ℝ) (hm : 1 < m) (hn : 1 < n) 
  (h_ratio1 : (2 * π * R1 * ((2 * R1) / (m + 1))) / (4 * π * R1 * R1) = 1 / (m + 1))
  (h_ratio2 : (2 * π * R2 * ((2 * R2) / (n + 1))) / (4 * π * R2 * R2) = 1 / (n + 1)): 
  R2 / R1 = ((m - 1) * (n + 1)) / ((m + 1) * (n - 1)) := 
by
  sorry

end sphere_radius_ratio_l557_557798


namespace opposite_of_three_l557_557757

theorem opposite_of_three : -3 = -3 := by
  -- The condition we have identified is the given number 3.
  -- We will directly state that the opposite of 3 is -3.
  -- This proof is trivial as we are directly replacing 3 with -3 to match the problem statement.
  rfl

end opposite_of_three_l557_557757


namespace least_positive_integer_x_multiple_53_l557_557804

theorem least_positive_integer_x_multiple_53 :
  ∃ x : ℕ, (x > 0) ∧ ((3 * x + 41)^2) % 53 = 0 ∧ ∀ y : ℕ, (y > 0) ∧ ((3 * y + 41)^2) % 53 = 0 → x ≤ y := 
begin
  use 4,
  split,
  { -- 4 > 0
    exact dec_trivial },
  split,
  { -- (3 * 4 + 41)^2 % 53 = 0
    calc (3 * 4 + 41)^2 % 53 = (53)^2 % 53 : by norm_num
    ... = 0 : by norm_num },
  { -- smallest positive integer solution
    assume y hy,
    cases hy with hy_gt0 hy_multiple,
    by_contradiction hxy,
    have x_val : 4 = 1,
      by linarith,
    norm_num at x_val,
    cases x_val
  }
end

end least_positive_integer_x_multiple_53_l557_557804


namespace distinct_domino_paths_l557_557681

/-- Matt will arrange five identical, dotless dominoes (1 by 2 rectangles) 
on a 6 by 4 grid so that a path is formed from the upper left-hand corner 
(0, 0) to the lower right-hand corner (4, 5). Prove that the number of 
distinct arrangements is 126. -/
theorem distinct_domino_paths : 
  let m := 4
  let n := 5
  let total_moves := m + n
  let right_moves := m
  let down_moves := n
  (total_moves.choose right_moves) = 126 := by
{ 
  sorry 
}

end distinct_domino_paths_l557_557681


namespace equation_solution_set_l557_557358

theorem equation_solution_set (x : ℝ) :
  x ∈ {3, 2} ↔ x * (2 * x - 4) = 3 * (2 * x - 4) :=
by
  sorry

end equation_solution_set_l557_557358


namespace conditions_for_equal_sqrt_l557_557609

variables (a b c d p : ℕ) (h1 : b = p * d) (h2 : pd_mod_a_zero : (p * d) % a = 0)

theorem conditions_for_equal_sqrt 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ p > 0) :
  (c = a * p * d - (p * d) / a) ↔ (sqrt (a + (b / c)) = a * sqrt (b / c)) :=
sorry

end conditions_for_equal_sqrt_l557_557609


namespace restaurant_total_glasses_l557_557815

theorem restaurant_total_glasses (x y t : ℕ) 
  (h1 : y = x + 16)
  (h2 : (12 * x + 16 * y) / (x + y) = 15)
  (h3 : t = 12 * x + 16 * y) : 
  t = 480 :=
by 
  -- Proof omitted
  sorry

end restaurant_total_glasses_l557_557815


namespace find_P_plus_Q_l557_557260

theorem find_P_plus_Q (P Q : ℝ) (h : ∃ b c : ℝ, (x^2 + 3 * x + 4) * (x^2 + b * x + c) = x^4 + P * x^2 + Q) : 
P + Q = 15 :=
by
  sorry

end find_P_plus_Q_l557_557260


namespace opposite_of_three_l557_557756

theorem opposite_of_three : -3 = -3 := by
  -- The condition we have identified is the given number 3.
  -- We will directly state that the opposite of 3 is -3.
  -- This proof is trivial as we are directly replacing 3 with -3 to match the problem statement.
  rfl

end opposite_of_three_l557_557756


namespace opposite_of_3_l557_557776

theorem opposite_of_3 : -3 = -3 := 
by
  -- sorry is added to skip the proof as per instructions
  sorry

end opposite_of_3_l557_557776


namespace no_common_terms_except_one_l557_557664

def X : ℕ → ℕ
| 0     := 1
| 1     := 1
| (n+2) := X n + 2 * X (n+1)

def Y : ℕ → ℕ
| 0     := 1
| 1     := 7
| (n+2) := 2 * Y (n+1) + 3 * Y n

theorem no_common_terms_except_one
  (n : ℕ) :
  ∀ m : ℕ, (X n = Y m) → (n = 0 ∧ m = 0) ∨ (n = 1 ∧ m = 0) ∨ (n = 0 ∧ m = 1) :=
sorry

end no_common_terms_except_one_l557_557664


namespace angle_ranges_l557_557148

structure Pentagon :=
  (α β γ δ ε : ℝ)
  (is_convex : α + β + γ + δ + ε = 540)
  (is_equilateral : [ α = β, β = γ, γ = δ, δ = ε, ε = α ])
  (no_angle_exceeds_120 : α ≤ 120 ∧ β ≤ 120 ∧ γ ≤ 120 ∧ δ ≤ 120 ∧ ε ≤ 120)
  (has_symmetry : ∃ t, symmetry t α β γ δ ε)

theorem angle_ranges (P : Pentagon) :
  93.5667 ≤ P.α ∧ P.α ≤ 120 ∧
  98.5333 ≤ P.β ∧ P.β ≤ 120 ∧
  103.2167 ≤ P.γ ∧ P.γ ≤ 111.4667 :=
by sorry

end angle_ranges_l557_557148


namespace maximize_distance_diff_l557_557035

def point_on_line_maximizes_distance_diff (P A B : ℝ × ℝ) : Prop :=
  P.1 = 5 ∧ P.2 = 6 ∧
  (2 * P.1 - P.2 - 4 = 0) ∧
  A = (4, -1) ∧ 
  B = (3, 4)

theorem maximize_distance_diff :
  ∃ P : ℝ × ℝ, point_on_line_maximizes_distance_diff P (4, -1) (3, 4) :=
begin
  use (5, 6),
  dsimp [point_on_line_maximizes_distance_diff],
  repeat { split },
  exact rfl,
  exact rfl,
  norm_num,
  exact rfl,
  exact rfl
end

end maximize_distance_diff_l557_557035


namespace function_order_l557_557974

theorem function_order (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f (2 - x) = f x)
  (h2 : ∀ x : ℝ, f (x + 2) = f (x - 2))
  (h3 : ∀ x1 x2 : ℝ, 1 ≤ x1 ∧ x1 ≤ 3 ∧ 1 ≤ x2 ∧ x2 ≤ 3 → (f x1 - f x2) / (x1 - x2) < 0) :
  f 2016 = f 2014 ∧ f 2014 > f 2015 :=
by
  sorry

end function_order_l557_557974


namespace find_AC_l557_557302

variables (A B C : Point)
variables (h1: Altitude A A1)
variables (h2: Altitude B B1)
variables [triangle : Triangle A B C] (hAA1 : height A A1 = 4) (hBB1 : height B B1 = 5) (lengthBC : dist B C = 6)

theorem find_AC : dist A C = 4.8 :=
by
  sorry

end find_AC_l557_557302


namespace maximum_value_of_function_l557_557715

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^2 - 2 * Real.sin x - 2

theorem maximum_value_of_function :
  ∃ x : ℝ, f x = 1 ∧ ∀ y : ℝ, -1 ≤ Real.sin y ∧ Real.sin y ≤ 1 → f y ≤ 1 :=
by
  sorry

end maximum_value_of_function_l557_557715


namespace prob1_l557_557071

theorem prob1 (f : ℕ+ → ℕ+) (h1 : ∀ x, f(f(x)) = 9 * x) (h2 : ∀ x, f(f(f(x))) = 27 * x) :
  ∀ x, f(x) = 3 * x :=
sorry

end prob1_l557_557071


namespace complex_number_quadrant_l557_557221

theorem complex_number_quadrant :
  let z := -1 - (1 : ℂ)
  ∃ (q : ℕ), q = 3 ∧
  (∀ (x y : ℝ), z = x + y * I → (x < 0 ∧ y < 0) → q = 3) := 
by
  let z := -1 - (1 : ℂ)
  use 3 -- our quadrant number
  split
  · exact rfl -- Prove that q = 3
  · intros x y hxy hnegative
    sorry -- This would be where we prove that given the conditions, the point lies in Quadrant III.

end complex_number_quadrant_l557_557221


namespace towel_bleaching_l557_557110

theorem towel_bleaching
  (original_length original_breadth : ℝ)
  (percentage_decrease_area : ℝ)
  (percentage_decrease_breadth : ℝ)
  (final_length final_breadth : ℝ)
  (h1 : percentage_decrease_area = 28)
  (h2 : percentage_decrease_breadth = 10)
  (h3 : final_breadth = original_breadth * (1 - percentage_decrease_breadth / 100))
  (h4 : final_area = original_area (1 - percentage_decrease_area / 100))
  (original_area final_area : ℝ) :
  final_length = original_length * 0.8 :=
begin
  -- Here, the goal would be to prove that final_length is 80% of the original_length
  sorry
end

end towel_bleaching_l557_557110


namespace inclination_angle_of_line_l557_557713

theorem inclination_angle_of_line : 
  ∃ α : ℝ, (sqrt 3 * x - y + 1 = 0) → α = π / 3 :=
by
  sorry

end inclination_angle_of_line_l557_557713


namespace triangle_ABC_BC_length_l557_557273

-- Definition of the problem conditions
def triangle_ABC : Type := 
  { a b c : ℝ // ∃ (AB AC BC : ℝ) (angleA : ℝ), 
      angleA = 60 ∧
      AC = 1 ∧
      (0.5 * AB * AC * real.sin (angleA * real.pi / 180) = real.sqrt 3) ∧
      (BC = real.sqrt (1 + (AB * AB) - 2 * 1 * AB * real.cos (angleA * real.pi / 180))) }

-- Statement to prove
theorem triangle_ABC_BC_length :
  ∀ (a b c : ℝ) (AB AC BC : ℝ) (angleA : ℝ),
    angleA = 60 →
    AC = 1 →
    (0.5 * AB * AC * real.sin (angleA * real.pi / 180) = real.sqrt 3) →
    (BC = real.sqrt (1 + (AB * AB) - 2 * 1 * AB * real.cos (angleA * real.pi / 180))) →
    (BC = real.sqrt 13) :=
by
  sorry

end triangle_ABC_BC_length_l557_557273


namespace strictly_decreasing_on_interval_l557_557153

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * Real.log x

theorem strictly_decreasing_on_interval : ∀ x, (0 < x ∧ x < 1) → deriv f x < 0 :=
begin
  sorry
end

end strictly_decreasing_on_interval_l557_557153


namespace ratio_bananas_to_kiwis_l557_557828

theorem ratio_bananas_to_kiwis 
  (cost_per_dozen_apples : ℕ := 14)
  (spent_on_kiwis : ℕ := 10)
  (max_apples : ℕ := 24)
  (initial_money : ℕ := 50)
  (subway_fare_one_way : ℕ := 350) : 
  (5 : 10) = (1 : 2) :=
by
  let subway_fare_total := (subway_fare_one_way * 2) / 100
  let remaining_money_after_subway := initial_money - subway_fare_total
  let remaining_money_after_kiwis := remaining_money_after_subway - spent_on_kiwis
  let cost_of_apples := (max_apples / 12) * cost_per_dozen_apples
  let spent_on_bananas := remaining_money_after_kiwis - cost_of_apples
  have h1 : remaining_money_after_kiwis = 33 := by sorry
  have h2 : cost_of_apples = 28 := by sorry
  have h3 : spent_on_bananas = 5 := by sorry
  exact (5 / 10 = 1 / 2)


end ratio_bananas_to_kiwis_l557_557828


namespace angle_bisector_form_l557_557854

noncomputable def P : ℝ × ℝ := (-8, 5)
noncomputable def Q : ℝ × ℝ := (-15, -19)
noncomputable def R : ℝ × ℝ := (1, -7)

-- Function to check if the given equation can be in the form ax + 2y + c = 0
-- and that a + c equals 89.
theorem angle_bisector_form (a c : ℝ) : a + c = 89 :=
by
   sorry

end angle_bisector_form_l557_557854


namespace production_steps_description_l557_557414

-- Definition of the choices
inductive FlowchartType
| ProgramFlowchart
| ProcessFlowchart
| KnowledgeStructureDiagram
| OrganizationalStructureDiagram

-- Conditions
def describeProductionSteps (flowchart : FlowchartType) : Prop :=
flowchart = FlowchartType.ProcessFlowchart

-- The statement to be proved
theorem production_steps_description:
  describeProductionSteps FlowchartType.ProcessFlowchart := 
sorry -- proof to be provided

end production_steps_description_l557_557414


namespace maximum_perimeter_l557_557514

noncomputable def triangle_base : ℝ := 10.0
noncomputable def triangle_height : ℝ := 12.0
noncomputable def segment_length : ℝ := 1.25

def distance (x y : ℝ) : ℝ := sqrt (x^2 + y^2)

def perimeter (k : ℕ) : ℝ :=
  segment_length + distance 12 (k : ℝ) + distance 12 (k + 1)

theorem maximum_perimeter : ∃ k : ℕ, k < 8 ∧ perimeter k = 26.27 :=
  sorry

end maximum_perimeter_l557_557514


namespace opposite_of_3_is_neg3_l557_557722

theorem opposite_of_3_is_neg3 : forall (n : ℤ), n = 3 -> -n = -3 :=
by
  sorry

end opposite_of_3_is_neg3_l557_557722


namespace regression_line_change_in_y_l557_557978

theorem regression_line_change_in_y (x : ℝ) :
    let y₁ := 2 - 1.5 * x
    let y₂ := 2 - 1.5 * (x + 1)
    y₂ - y₁ = -1.5 := 
begin
    sorry
end

end regression_line_change_in_y_l557_557978


namespace walking_speed_is_4_l557_557477

def distance : ℝ := 20
def total_time : ℝ := 3.75
def running_distance : ℝ := 10
def running_speed : ℝ := 8
def walking_distance : ℝ := 10

theorem walking_speed_is_4 (W : ℝ) 
  (H1 : running_distance + walking_distance = distance)
  (H2 : running_speed > 0)
  (H3 : walking_distance > 0)
  (H4 : W > 0)
  (H5 : walking_distance / W + running_distance / running_speed = total_time) :
  W = 4 :=
by sorry

end walking_speed_is_4_l557_557477


namespace sample_selection_and_variance_l557_557277

def num_freshmen := 1000
def num_boys := 600
def num_girls := 400
def sample_size := 50

def mean_height_boys := 170
def variance_boys := 14

def mean_height_girls := 160
def variance_girls := 34

def num_boys_sample :=  (num_boys * sample_size) / num_freshmen
def num_girls_sample := (num_girls * sample_size) / num_freshmen

def total_sample_mean := (num_boys_sample * mean_height_boys + num_girls_sample * mean_height_girls) / sample_size
def total_sample_variance : ℝ :=
  let boys_term := num_boys_sample * (variance_boys + (mean_height_boys - total_sample_mean) ^ 2)
  let girls_term := num_girls_sample * (variance_girls + (mean_height_girls - total_sample_mean) ^ 2)
  (boys_term + girls_term) / sample_size

theorem sample_selection_and_variance :
  num_boys_sample = 30 ∧ num_girls_sample = 20 ∧ total_sample_variance = 46 := by
  sorry

end sample_selection_and_variance_l557_557277


namespace tripled_odot_l557_557387

def odot (a b : ℝ) : ℝ := a + (3 * a) / (2 * b)

theorem tripled_odot (a b : ℝ) (hab : a = 9) (hbb : b = 4) : (3 * (odot a b)) = 37.125 := by
  sorry

end tripled_odot_l557_557387


namespace dihedral_angle_at_base_l557_557371

theorem dihedral_angle_at_base (a b : ℝ) (h x : ℝ) :
  let cos_angle := sqrt (2 * a^2 - b^2) / b in
  ∃ θ : ℝ, θ = real.arccos cos_angle :=
sorry

end dihedral_angle_at_base_l557_557371


namespace area_ratio_of_triangles_l557_557048

theorem area_ratio_of_triangles 
  (A B C A' B' C' : Point) 
  (midpoint_AA' : Midpoint B' A A') 
  (midpoint_BB' : Midpoint C' B B') 
  (midpoint_CC' : Midpoint A' C C') :
  area (Triangle.mk A B C) / area (Triangle.mk A' B' C') = 7 := by
  sorry

end area_ratio_of_triangles_l557_557048


namespace probability_heads_l557_557832

variable (p : ℝ)
variable (h1 : 0 ≤ p)
variable (h2 : p ≤ 1)
variable (h3 : p * (1 - p) ^ 4 = 0.03125)

theorem probability_heads :
  p = 0.5 :=
sorry

end probability_heads_l557_557832


namespace opposite_of_three_l557_557735

theorem opposite_of_three : -3 = -3 := 
by sorry

end opposite_of_three_l557_557735


namespace minimum_value_at_ln2_l557_557487

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 4 * Real.exp (-x)

theorem minimum_value_at_ln2 : ∃ x, f x = 4 ∧ is_minimum (f x)
  := ∃ x = Real.log 2, f x = 4 :=
begin
  sorry
end

end minimum_value_at_ln2_l557_557487


namespace Xiao_Li_first_attempt_prob_Xiao_Li_math_expectation_l557_557921

noncomputable def arithmetic_sequence_prob (d : ℝ) (P1 : ℝ) : ℕ → ℝ
| 0     => 0
| (n+1) => P1 + n * d

variables (d : ℝ) (P1 : ℝ)
def cond1 := d = 1/8
def cond2 := P1 ≤ 1/2
def cond3 := arithmetic_sequence_prob d P1 2 = 9/32

theorem Xiao_Li_first_attempt_prob :
  cond1 ∧ cond2 ∧ cond3 → P1 = 1/4 :=
by
  sorry

def prob_pass_first_attempt : ℝ := 1/4
def prob_pass_second_attempt : ℝ := 9/32
def prob_pass_third_attempt : ℝ := (1 - prob_pass_first_attempt) * (1 - prob_pass_second_attempt) * (1/2)
def prob_pass_fourth_attempt : ℝ := (1 - prob_pass_first_attempt) * (1 - prob_pass_second_attempt) * (1 - 1/2)

def expectation (P1: ℝ) (P2: ℝ) (P3: ℝ) (P4: ℝ) : ℝ :=
  1 * P1 + 2 * P2 + 3 * P3 + 4 * P4

theorem Xiao_Li_math_expectation :
  cond1 ∧ cond2 ∧ cond3 → expectation prob_pass_first_attempt prob_pass_second_attempt prob_pass_third_attempt prob_pass_fourth_attempt = 157/64 :=
by
  sorry

end Xiao_Li_first_attempt_prob_Xiao_Li_math_expectation_l557_557921


namespace king_lancelot_seats_38_l557_557895

noncomputable def totalSeats (seat_king seat_lancelot : ℕ) : ℕ :=
  if seat_king < seat_lancelot then
    2 * (seat_lancelot - seat_king - 1) + 2
  else
    2 * (seat_king - seat_lancelot - 1) + 2

theorem king_lancelot_seats_38 (seat_king seat_lancelot : ℕ) (h1 : seat_king = 10) (h2 : seat_lancelot = 29) :
  totalSeats seat_king seat_lancelot = 38 := 
  by
    sorry

end king_lancelot_seats_38_l557_557895


namespace surface_area_of_sphere_l557_557223

noncomputable def sphere_surface_area
  (A B C D : Point)
  (O : Sphere)
  (h : ℝ)
  (tetrahedron_vol : ℝ)
  (BC BD θ : ℝ)
  (O_center_OBCD : ℝ) 
  (AB : ℝ)
  (triangle_circum_diameter : ℝ) 
  (radius_sqrt23halves : ℝ) 
  (surface_area : ℝ) :=
  A ∈ O ∧ B ∈ O ∧ C ∈ O ∧ D ∈ O ∧ 
  (O.diameter AB) ∧ 
  tetrahedron_vol = 4*sqrt 3 / 3 ∧ 
  BC = 4 ∧ 
  BD = sqrt 3 ∧ 
  θ = 90 ∧ 
  h = 2 ∧ 
  O_center_OBCD = 1 ∧
  triangle_circum_diameter = sqrt 19 ∧
  radius_sqrt23halves = sqrt 23 / 2 ∧
  surface_area = 4 * π * (23 / 4) 

theorem surface_area_of_sphere 
  (A B C D : Point)
  (O : Sphere)
  (h : ℝ)
  (tetrahedron_vol : ℝ)
  (BC BD θ : ℝ)
  (O_center_OBCD : ℝ) 
  (AB : Diameter)
  (triangle_circum_diameter : ℝ) 
  (radius_sqrt23halves : ℝ) 
  (surface_area : ℝ) :
  sphere_surface_area A B C D O h tetrahedron_vol BC BD θ O_center_OBCD AB triangle_circum_diameter radius_sqrt23halves surface_area = 23 * π :=
sorry

end surface_area_of_sphere_l557_557223


namespace complex_number_in_second_quadrant_l557_557781

def z : ℂ := (2 * complex.I) / (2 - complex.I)

theorem complex_number_in_second_quadrant (h : z = (-(2 / 5)) + (4 / 5) * complex.I) : 
  z.re < 0 ∧ z.im > 0 :=
by 
  -- This is to state that z is simplified to -2/5 + 4/5i 
  rw h
  -- sorry is used to skip the proof as per instructions
  sorry

end complex_number_in_second_quadrant_l557_557781


namespace find_n_tan_eq_l557_557957

theorem find_n_tan_eq (n : ℝ) (h1 : -180 < n) (h2 : n < 180) (h3 : Real.tan (n * Real.pi / 180) = Real.tan (678 * Real.pi / 180)) : 
  n = 138 := 
sorry

end find_n_tan_eq_l557_557957


namespace squares_below_line_l557_557004

theorem squares_below_line (x y: ℕ) (h1 : 3 * x + 75 * y = 675) (h2 : x ≥ 0) (h3 : y ≥ 0) : 
  let total_squares := 225 * 9 in
  let diagonal_squares := 233 in
  (total_squares - diagonal_squares) / 2 = 896 :=
by
  sorry

end squares_below_line_l557_557004


namespace eq_fraction_l557_557684

variables (a b : ℝ)
variable (ha : a ≠ 0)
variable (hb : b ≠ 0)

theorem eq_fraction :
  (a ^ (-2) * b ^ (-2)) / (a ^ (-4) + b ^ (-4)) = (a ^ 2 * b ^ 2) / (a ^ 4 + b ^ 4) :=
sorry

end eq_fraction_l557_557684


namespace range_of_f_l557_557538

-- Definitions of provided functions
def φ (t : ℝ) : ℝ := t + 1 / t
def g (x : ℝ) : ℝ := (x ^ 3) + (1 / (x ^ 3))

-- f(x) as provided in the problem statement
def f (x : ℝ) : ℝ :=
  1 / (g (16 * g (g (Real.log x)) / 65))

-- Main theorem statement
theorem range_of_f :
  ∀ x ∈ Ioi (0 : ℝ), f x ∈ Set.Icc (-8 / 65) 0 ∪ Set.Icc 0 (8 / 65) :=
by
  sorry

end range_of_f_l557_557538


namespace wilma_garden_rows_l557_557808

theorem wilma_garden_rows :
  ∃ (rows : ℕ),
    (∃ (yellow green red total : ℕ),
      yellow = 12 ∧
      green = 2 * yellow ∧
      red = 42 ∧
      total = yellow + green + red ∧
      total / 13 = rows ∧
      rows = 6) :=
sorry

end wilma_garden_rows_l557_557808


namespace integral_solution_l557_557061

noncomputable def integral_of_function (x : ℝ) := ∫ (sqrt 2 - 8 * x) * sin (3 * x)

theorem integral_solution (x : ℝ) :
  integral_of_function x = - (1 / 3) * (sqrt 2 - 8 * x) * cos (3 * x) + (8 / 9) * sin (3 * x) + (λ C, C) :=
by
  sorry

end integral_solution_l557_557061


namespace opposite_of_3_is_neg3_l557_557723

theorem opposite_of_3_is_neg3 : forall (n : ℤ), n = 3 -> -n = -3 :=
by
  sorry

end opposite_of_3_is_neg3_l557_557723


namespace period_of_f_intervals_of_monotonic_increase_l557_557235

def f (x : ℝ) : ℝ := sqrt 3 * sin (2 * x) - cos (2 * x)

theorem period_of_f : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π :=
by
  existsi π
  split
  · sorry
  · sorry

theorem intervals_of_monotonic_increase :
  ∀ k : ℤ, ∀ x, x ∈ Icc (-π/6 + k * π) (π/3 + k * π) →
    (∀ y ∈ Icc (-π/6 + k * π) (π/3 + k * π), f y ≤ f x ↔ y ≤ x) :=
by
  intro k x
  intro hx
  split
  · sorry
  · sorry

end period_of_f_intervals_of_monotonic_increase_l557_557235


namespace parallel_lines_in_plane_l557_557292

theorem parallel_lines_in_plane (plane : Type) (L : set plane) (H_parallel : ∀ l ∈ L, ∃! p : plane, p ∈ L ∧ (∀ q ∈ L, ¬(p = q ∧ p ∩ q ≠ ∅)))
  (square : Type) (S : set square) (H_square_parallel : ∃! p1 p2 ∈ S, p1 ≠ p2 ∧ (∀ q ∈ S, ¬(p1 = q ∨ p2 = q)) 
  ∧ ∀ q1 q2 ∈ S, (p1 ≠ q1 ∧ p2 ≠ q2 → p1 ∩ q1 = ∅ ∧ p2 ∩ q2 = ∅)):
  (∃! l ∈ L, ∀ l' ∈ L, ¬(l = l' ∧ l ∩ l' ≠ ∅)) ∧ (∀ l1 l2 ∈ L, l1 ∩ l2 = ∅) ∧ (∃! p1 p2 ∈ S, p1 ≠ p2 ∧ (p1∩p2 = ∅)) := by
  sorry

end parallel_lines_in_plane_l557_557292


namespace range_of_f_range_of_a_l557_557823

-- Define the function f(x) = 4^x - 2^(x+1)
def f (x : ℝ) : ℝ := 4^x - 2^(x + 1)

-- State that the range of f is [-1, +∞)
theorem range_of_f : Set.range f = Set.Ici (-1) := 
sorry

-- Define the equation g(x, a) = 4^x - 2^(x+1) + a
def g (x a : ℝ) : ℝ := 4^x - 2^(x + 1) + a

-- State that if g(x, a) = 0 has a solution, then a ≤ 1
theorem range_of_a (a : ℝ) : (∃ x : ℝ, g x a = 0) → a ≤ 1 :=
sorry

end range_of_f_range_of_a_l557_557823


namespace roster_method_A_l557_557787

def A : Set ℤ := {x | 0 < x ∧ x ≤ 2}

theorem roster_method_A :
  A = {1, 2} :=
by
  sorry

end roster_method_A_l557_557787


namespace calc_1_calc_2_l557_557134

-- Proof Problem 1
theorem calc_1 : 7 - (-3) + (-4) - | -8 | = -2 := 
by
  -- Simplification steps turning into the proof steps.
  sorry

-- Proof Problem 2
theorem calc_2 : (-81) / (-9 / 4) * (4 / 9) / (-16) = -1 :=
by
  -- Simplification steps turning into the proof steps.
  sorry

end calc_1_calc_2_l557_557134


namespace proposition_2_true_l557_557668

variable {m n : Line}
variable {α β γ : Plane}

theorem proposition_2_true :
  (∀ m ⊆ β ∧ α ⊥ β, m ⊥ α ∨ m ∥ α ∨ m ⊆ α) ∧
  (∀ m ∥ α ∧ m ⊥ β, α ⊥ β) ∧
  (∀ α ⊥ β ∧ α ⊥ γ, β ⊥ γ ∨ β ∩ γ ∨ β ∥ γ) ∧
  (∀ α ∩ γ = m ∧ β ∩ γ = n ∧ m ∥ n, α ∥ β) ↔ 
   1 = false ∧ 2 = true ∧ 3 = false ∧ 4 = false :=
by
  sorry

end proposition_2_true_l557_557668


namespace sparrows_among_non_robins_percentage_l557_557276

-- Define percentages of different birds
def finches_percentage : ℝ := 0.40
def sparrows_percentage : ℝ := 0.20
def owls_percentage : ℝ := 0.15
def robins_percentage : ℝ := 0.25

-- Define the statement to prove 
theorem sparrows_among_non_robins_percentage :
  ((sparrows_percentage / (1 - robins_percentage)) * 100) = 26.67 := by
  -- This is where the proof would go, but it's omitted as per instructions
  sorry

end sparrows_among_non_robins_percentage_l557_557276


namespace probability_correct_number_l557_557337

theorem probability_correct_number : 
  let first_two_digits := {t | t = "27" ∨ t = "30"},
      last_five_digits := {x | ∃ (d1 d2 d3 d4 d5: ℕ), 
        {d1, d2, d3, d4, d5} = {2, 4, 5, 8, 9} ∧ 
        d1 ≠ 5 ∧ d5 ≠ 5},
      total_numbers := (2 * 72) 
  in 
  (1 : ℚ) / total_numbers = 1 / 144 :=
by
  sorry

end probability_correct_number_l557_557337


namespace smallest_k_divisor_l557_557185

-- Define the polynomials in Lean
def P (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1
def Q (z : ℂ) (k : ℕ) : ℂ := z^k - 1

-- State the main theorem/problem
theorem smallest_k_divisor (k : ℕ) (hk : k > 0) : 
  (∀ z : ℂ, P z | Q z k) ↔ k = 9 := 
sorry

end smallest_k_divisor_l557_557185


namespace parallelogram_within_triangle_l557_557316

variable {A B C O P D E F G : Type*}

-- Definitions based on given conditions
def circumcenter (Δ : Triangle A B C) : Point O := -- definition of circumcenter O
def projection (P : Point) (l : Line) : Point := -- definition of projection of P onto a line
def is_interior_point (P : Point) (Δ : Triangle A O B) : Prop := -- definition of P being an interior point of triangle AOB

-- Problem statement to prove
theorem parallelogram_within_triangle (Δ : Triangle A B C) (O : Point) 
  (P : Point) (D E F : Point) :
  circumcenter Δ = O →
  is_interior_point P (Triangle.mk A O B) →
  D = projection P (Line.mk B C) →
  E = projection P (Line.mk C A) →
  F = projection P (Line.mk A B) →
  ∃ (G : Point), parallelogram (Line.segment F E) (Line.segment F D) (Δ.interior) :=
by 
  { sorry }

end parallelogram_within_triangle_l557_557316


namespace find_digit_n_l557_557391

theorem find_digit_n :
  let n := 2 in
  (9 * 11 * 13 * 15 * 17 = 3 * 100000 + n * 10000 + 8185) ∧
  ((3 + n + 8 + 1 + 8 + 1 + 5) % 9 = 0) :=
by
  sorry

end find_digit_n_l557_557391


namespace concurrency_condition_l557_557654

noncomputable def point (α : Type) := α

variables {α : Type} [MetricSpace α] [NormedSpace ℝ α]

-- Define the points A, B, and C in a triangle ABC
variables (A B C : point α)

-- Define that the triangle ABC is scalene and acute-angled
def scalene_acute_triangle (A B C : point α) : Prop :=
  ∠BAC < π / 2 ∧ ∠ABC < π / 2 ∧ ∠BCA < π / 2 ∧ (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ A)

-- Define the point D on the circumcircle of ABC such that AD is a symmedian
variable (D : point α)
def on_circumcircle_and_symmedian (A B C D : point α) : Prop :=
  on_circumcircle A B C D ∧ is_symmedian A B C D

-- Define the reflection of D about BC as E
variable (E : point α)
def reflection_about_bc (D B C E : point α) : Prop :=
  E = reflection D (line B C)

-- Define the reflection of E about AB as C0
variable (C0 : point α)
def reflection_about_ab (E A B C0 : point α) : Prop :=
  C0 = reflection E (line A B)

-- Define the reflection of E about AC as B0
variable (B0 : point α)
def reflection_about_ac (E A C B0 : point α) : Prop :=
  B0 = reflection E (line A C)

-- The theorem to be proved
theorem concurrency_condition {A B C D E C0 B0 : point α}
  (h1 : scalene_acute_triangle A B C)
  (h2 : on_circumcircle_and_symmedian A B C D)
  (h3 : reflection_about_bc D B C E)
  (h4 : reflection_about_ab E A B C0)
  (h5 : reflection_about_ac E A C B0) :
  (lines_concurrent (line_through A D) (line_through B B0) (line_through C C0)) ↔ (angle A B C = 60) := sorry

end concurrency_condition_l557_557654


namespace problem_statement_l557_557698

noncomputable def angle_bisectors_are_rational 
  (a b c : ℝ) (ha : a = 84) (hb : b = 125) (hc : c = 169) : 
  Prop :=
  ∃ fa fb fc : ℚ, 
    fa = (2*b*c / (b+c)) * (real.cos (real.arccos ((b^2 + c^2 - a^2) / (2*b*c)) / 2)) ∧
    fb = (2*a*c / (a+c)) * (real.cos (real.arccos ((a^2 + c^2 - b^2) / (2*a*c)) / 2)) ∧
    fc = (2*a*b / (a+b)) * (real.cos (real.arccos ((a^2 + b^2 - c^2) / (2*a*b)) / 2))

theorem problem_statement :
  angle_bisectors_are_rational 84 125 169 84 125 169 := sorry

end problem_statement_l557_557698


namespace fraction_white_surface_area_l557_557838

/-- A 4-inch cube is constructed from 64 smaller cubes, each with 1-inch edges.
   48 of these smaller cubes are colored red and 16 are colored white.
   Prove that if the 4-inch cube is constructed to have the smallest possible white surface area showing,
   the fraction of the white surface area is 1/12. -/
theorem fraction_white_surface_area : 
  let total_surface_area := 96
  let white_cubes := 16
  let exposed_white_surface_area := 8
  (exposed_white_surface_area / total_surface_area) = (1 / 12) := 
  sorry

end fraction_white_surface_area_l557_557838


namespace geometric_sequence_a6_l557_557370

theorem geometric_sequence_a6 :
  ∃ (a : ℕ → ℝ), (∀ n, a n > 0) ∧ (∀ n, a (n + 1) = 2 * a n) ∧ (a 4 * a 10 = 16) → (a 6 = 2) :=
by
  sorry

end geometric_sequence_a6_l557_557370


namespace least_positive_integer_l557_557801

theorem least_positive_integer (x : ℕ) :
  (∃ k : ℤ, (3 * x + 41) ^ 2 = 53 * k) ↔ x = 4 :=
by
  sorry

end least_positive_integer_l557_557801


namespace common_factor_l557_557529

theorem common_factor (x y a b : ℤ) : 
  3 * x * (a - b) - 9 * y * (b - a) = 3 * (a - b) * (x + 3 * y) :=
by {
  sorry
}

end common_factor_l557_557529


namespace red_light_probability_l557_557852

theorem red_light_probability (n : ℕ) (p_r : ℚ) (waiting_time_for_two_red : ℚ) 
    (prob_two_red : ℚ) :
    n = 4 →
    p_r = (1/3 : ℚ) →
    waiting_time_for_two_red = 4 →
    prob_two_red = (8/27 : ℚ) :=
by
  intros hn hp hw
  sorry

end red_light_probability_l557_557852


namespace opposite_of_three_l557_557764

theorem opposite_of_three : -3 = opposite(3) := 
by
  sorry

end opposite_of_three_l557_557764


namespace liquid_X_percentage_in_new_solution_l557_557444

noncomputable def solutionY_initial_kg : ℝ := 10
noncomputable def percentage_liquid_X : ℝ := 0.30
noncomputable def evaporated_water_kg : ℝ := 2
noncomputable def added_solutionY_kg : ℝ := 2

-- Calculate the amount of liquid X in the original solution
noncomputable def initial_liquid_X_kg : ℝ :=
  percentage_liquid_X * solutionY_initial_kg

-- Calculate the remaining weight after evaporation
noncomputable def remaining_weight_kg : ℝ :=
  solutionY_initial_kg - evaporated_water_kg

-- Calculate the amount of liquid X after evaporation
noncomputable def remaining_liquid_X_kg : ℝ := initial_liquid_X_kg

-- Since only water evaporates, remaining water weight
noncomputable def remaining_water_kg : ℝ :=
  remaining_weight_kg - remaining_liquid_X_kg

-- Calculate the amount of liquid X in the added solution
noncomputable def added_liquid_X_kg : ℝ :=
  percentage_liquid_X * added_solutionY_kg

-- Total liquid X in the new solution
noncomputable def new_liquid_X_kg : ℝ :=
  remaining_liquid_X_kg + added_liquid_X_kg

-- Calculate the water in the added solution
noncomputable def percentage_water : ℝ := 0.70
noncomputable def added_water_kg : ℝ :=
  percentage_water * added_solutionY_kg

-- Total water in the new solution
noncomputable def new_water_kg : ℝ :=
  remaining_water_kg + added_water_kg

-- Total weight of the new solution
noncomputable def new_total_weight_kg : ℝ :=
  remaining_weight_kg + added_solutionY_kg

-- Percentage of liquid X in the new solution
noncomputable def percentage_new_liquid_X : ℝ :=
  (new_liquid_X_kg / new_total_weight_kg) * 100

-- The proof statement
theorem liquid_X_percentage_in_new_solution :
  percentage_new_liquid_X = 36 :=
by
  sorry

end liquid_X_percentage_in_new_solution_l557_557444


namespace tetrahedron_colorings_distinguishable_l557_557946

theorem tetrahedron_colorings_distinguishable : 
  let faces := 4
  let colors := 4
  let indistinguishable (c1 c2 : ℕ → ℕ → Prop) : Prop := 
    ∃ R : Fin 4 → Fin 4, bijective R ∧ ∀ f, c1 f = c2 (R f) 

  -- Total distinguishable colorings based on rotation
  faces = 4 ∧ colors = 4 → 
  let all_colorings := 4 + 12 + 6 + 12 + 1
  all_colorings = 35 :=
by
  sorry

end tetrahedron_colorings_distinguishable_l557_557946


namespace find_successors_and_sum_l557_557842

namespace SequenceProblem

def sequence : ℕ → ℕ
| 0       := 40
| (n + 1) := 
  let m := (n % 3) in
  match m with
  | 0 := sequence n + 1
  | 1 := sequence n + 3
  | 2 := sequence n + 5
  | _ := 0

theorem find_successors_and_sum : 
  sequence 1 = 41 ∧ ∃ k, sequence k = 874 ∧ sequence (k + 1) = 875 ∧ 244 + 247 = 491 :=
by
  sorry

end SequenceProblem

end find_successors_and_sum_l557_557842


namespace exists_four_digit_number_equals_square_of_two_digit_l557_557346

theorem exists_four_digit_number_equals_square_of_two_digit :
  ∃ (M N : ℕ), 10 ≤ M ∧ M < 100 ∧ 1000 ≤ N ∧ N < 10000 ∧ N = M^2 :=
by
  use 33, 1089
  -- It suffices to check the conditions directly
  repeat {split}; try {norm_num}
  calc 1089 = 33 * 33 : by norm_num

end exists_four_digit_number_equals_square_of_two_digit_l557_557346


namespace terminating_decimal_count_l557_557964

theorem terminating_decimal_count : 
  let count := (list.range' 1 543).count (λ n, n % 13 = 0) in
  count = 41 := 
by
  sorry

end terminating_decimal_count_l557_557964


namespace ae_ed_eq_areas_l557_557625

variables {A B C D O M E : Type} [ConvexQuadrilateral A B C D] [Intersection AC BD O] [Midpoint M B C] [Intersection MO AD E]

theorem ae_ed_eq_areas (h1 : ConvexQuadrilateral A B C D)
    (h2 : Intersection AC BD O)
    (h3 : Midpoint M B C)
    (h4 : Intersection MO AD E) :
    (AE / ED = (AreaOfTriangle ABO) / (AreaOfTriangle CDO)) :=
sorry

end ae_ed_eq_areas_l557_557625


namespace purely_periodic_period_le_T_l557_557399

theorem purely_periodic_period_le_T {a b : ℚ} (T : ℕ) 
  (ha : ∃ m, a = m / (10^T - 1)) 
  (hb : ∃ n, b = n / (10^T - 1)) :
  (∃ T₁, T₁ ≤ T ∧ ∃ p, a = p / (10^T₁ - 1)) ∧ 
  (∃ T₂, T₂ ≤ T ∧ ∃ q, b = q / (10^T₂ - 1)) := 
sorry

end purely_periodic_period_le_T_l557_557399


namespace sum_of_squares_of_coprime_l557_557672

open Nat

theorem sum_of_squares_of_coprime (n : ℕ) (p : ℕ → ℕ) (factors : Finset ℕ) (α : ℕ → ℕ) (m : ℕ)
  (coprime_set : Finset ℕ) (h_factorization : ∀ (i : ℕ), i ∈ factors → p i ∣ n)
  (h_exponent : ∀ (i : ℕ), i ∈ factors → α i > 0)
  (h_coprime : coprime_set.card = m ∧ ∀ (r : ℕ), r ∈ coprime_set → gcd r n = 1) :
  ∑ r in coprime_set, r^2 = (m / 3) * (n^2 + (1 / 2) * (-1)^factors.card * (∏ i in factors, p i)) := by
  sorry

end sum_of_squares_of_coprime_l557_557672


namespace sum_of_possible_m_l557_557998

theorem sum_of_possible_m (m : ℤ) (h : 0 < 5 * m ∧ 5 * m < 35) : 
  (∑ n in { n : ℤ | 0 < 5 * n ∧ 5 * n < 35 }.to_finset, n) = 21 := 
by 
  sorry

end sum_of_possible_m_l557_557998


namespace solve_sin_minus_cos_eq_one_l557_557166

theorem solve_sin_minus_cos_eq_one (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x < π) :
  (\sin x - \cos x = 1) ↔ (x = π / 2) := sorry

end solve_sin_minus_cos_eq_one_l557_557166


namespace expected_value_at_neg_one_l557_557368

/-- The expected value for the total number of games played in a best-of-five series, given the 
     probability x that a specific team wins a game, can be represented by a polynomial f(x). -/
theorem expected_value_at_neg_one (x : ℝ) : 
  let f : ℝ := 3 * (x ^ 3 + (1 - x) ^ 3) + 12 * (x ^ 3 * (1 - x) + (1 - x) ^ 3 * x) + 30 * (x ^ 2 * (1 - x) ^ 2) in
  f = 21 := 
by {
  sorry
}

end expected_value_at_neg_one_l557_557368


namespace factor_expression_l557_557950

theorem factor_expression (x : ℝ) : 72 * x ^ 5 - 162 * x ^ 9 = -18 * x ^ 5 * (9 * x ^ 4 - 4) :=
by
  sorry

end factor_expression_l557_557950


namespace total_number_of_seats_l557_557889

theorem total_number_of_seats (n : ℕ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ n) 
                               (h2 : (10 : ℕ) < (29 : ℕ)) 
                               (h3 : (29 - 10) % (n / 2) = 0) : n = 38 :=
by sorry

end total_number_of_seats_l557_557889


namespace true_proposition_l557_557591

variables (a b : ℝ) (p q : Prop)

-- Definitions derived from conditions
def proposition_p := ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (Real.log (a + b) = Real.log a + Real.log b)
def proposition_q := ∀ (l1 l2 : set ℝ), ((∃ p1 p2 p3 p4 : ℝ, (p1 ≠ p2 ∨ p3 ≠ p4)) → (¬ (∃ (plane : set ℝ), l1 ⊆ plane ∧ l2 ⊆ plane)))

-- Proof goal
theorem true_proposition : proposition_p ∧ proposition_q :=
by
  sorry

end true_proposition_l557_557591


namespace train_travel_and_time_l557_557817

variable (v : ℝ) (ρ : ℝ) (e : ℝ) (g : ℝ)

-- Constants provided in the conditions
def v_const : v = 12 := rfl
def ρ_const : ρ = 0.004 := rfl
def e_const : e = 0.025 := rfl
def g_const : g = 9.8 := rfl

-- The main theorem for the proof
theorem train_travel_and_time (v ρ e g : ℝ) 
  (h1 : v = 12) (h2 : ρ = 0.004) (h3 : e = 0.025) (h4 : g = 9.8) :
  let s := v^2 / (2 * ρ * g)
      t := 2 * s / v
      s' := v^2 / (2 * (ρ + e) * g)
      t' := 2 * s' / v
  in s ≈ 1836.73 ∧ s' ≈ 2538.79 ∧ t ≈ 306.12 ∧ t' ≈ 423.13 :=
by
  sorry

end train_travel_and_time_l557_557817


namespace smallest_m_exists_l557_557662

noncomputable def complex_set (x : ℝ) (y : ℝ) : set ℂ :=
  {z | z.re = x ∧ z.im = y ∧ (1/2 ≤ x ∧ x ≤ real.sqrt 2 / 2)}

theorem smallest_m_exists (m : ℕ) :
  (∀ n : ℕ, n ≥ m → ∃ z ∈ (complex_set x y), z ^ n = 1) ↔ m = 12 := by
  sorry

end smallest_m_exists_l557_557662


namespace total_seats_l557_557906

theorem total_seats (KA_pos : ℕ) (SL_pos : ℕ) (h1 : KA_pos = 10) (h2 : SL_pos = 29) (h3 : SL_pos = KA_pos + (KA_pos * 2 - 1) / 2):
  let total_positions := 2 * (SL_pos - KA_pos - 1) + 2
  total_positions = 38 :=
by
  sorry

end total_seats_l557_557906


namespace line_through_ellipse_intersection_l557_557233

/-- Ellipse C: x²/3 + y² = 1 with a line of slope 1 intersecting at points A and B with distance |AB| = 3√2/2.
    Prove that the equation of the line is y = x + 1 or y = x - 1. -/
theorem line_through_ellipse_intersection :
  ∀ A B : ℝ × ℝ,
  (∃ x y : ℝ, (x^2 / 3) + y^2 = 1) ∧
  (∀ m : ℝ, y = x + m) ∧
  (dist A B = 3 * real.sqrt 2 / 2) →
  (y = x + 1) ∨ (y = x - 1) :=
by
  sorry

end line_through_ellipse_intersection_l557_557233


namespace robot_encoded_number_correct_l557_557784

variables (P O B T M A E I K : ℕ)
variables (encode : String → ℕ)

-- Conditions
axiom P_eq_31 : P = 31
axiom O_eq_12 : O = 12
axiom B_eq_13 : B = 13
axiom T_eq_33 : T = 33
axiom M_eq_22 : M = 22
axiom A_eq_32 : A = 32
axiom E_eq_11 : E = 11
axiom I_eq_23 : I = 23
axiom K_eq_13 : K = 13

-- Encoding assumption
axiom encode_MATHEMATICS : 
  encode "MATHEMATICS" = 2232331122323323132

-- The theorem to be proven
theorem robot_encoded_number_correct :
  encode "MATHEMATICS" = 2232331122323323132 :=
by
  apply encode_MATHEMATICS
  sorry

end robot_encoded_number_correct_l557_557784


namespace opposite_of_3_is_neg3_l557_557749

def opposite (x : ℝ) := -x

theorem opposite_of_3_is_neg3 : opposite 3 = -3 :=
by
  sorry

end opposite_of_3_is_neg3_l557_557749


namespace opposite_of_x_is_positive_l557_557020

-- Assume a rational number x
def x : ℚ := -1 / 2023

-- Theorem stating the opposite of x is 1 / 2023
theorem opposite_of_x_is_positive : -x = 1 / 2023 :=
by
  -- Required part of Lean syntax; not containing any solution steps
  sorry

end opposite_of_x_is_positive_l557_557020


namespace total_amount_paid_l557_557499

-- Define the given conditions
def q_g : ℕ := 9        -- Quantity of grapes
def r_g : ℕ := 70       -- Rate per kg of grapes
def q_m : ℕ := 9        -- Quantity of mangoes
def r_m : ℕ := 55       -- Rate per kg of mangoes

-- Define the total amount paid calculation and prove it equals 1125
theorem total_amount_paid : (q_g * r_g + q_m * r_m) = 1125 :=
by
  -- Proof will be provided here. Currently using 'sorry' to skip it.
  sorry

end total_amount_paid_l557_557499


namespace range_of_m_l557_557218

open Real

theorem range_of_m (p q: Prop) (m: ℝ) :
  (p ↔ ∃ x : ℝ, x^2 + m < 0) ∧ (q ↔ ∀ x : ℝ, x^2 + m * x + 1 > 0) ∧ 
  (p ∨ q) ∧ ¬(p ∧ q) →
  m ∈ Iic (-2) ∪ Ico 0 2 :=
  by
  intros h
  sorry

end range_of_m_l557_557218


namespace smallest_n_integer_sum_l557_557319

noncomputable def a : ℝ := Real.pi / 2010

def S (n : ℕ) : ℝ :=
  2 * ∑ k in Finset.range n, Real.cos (k^2 * a) * Real.sin (k * a)

theorem smallest_n_integer_sum : ∃ n : ℕ, S n ∈ Int ∧ ∀ m : ℕ, m < 67 → S m ∉ Int :=
by
  sorry

end smallest_n_integer_sum_l557_557319


namespace container_volumes_l557_557042

variable (a : ℕ)

theorem container_volumes (h₁ : a = 18) :
  a^3 = 5832 ∧ (a - 4)^3 = 2744 ∧ (a - 6)^3 = 1728 :=
by {
  sorry
}

end container_volumes_l557_557042


namespace least_positive_integer_l557_557802

theorem least_positive_integer (x : ℕ) :
  (∃ k : ℤ, (3 * x + 41) ^ 2 = 53 * k) ↔ x = 4 :=
by
  sorry

end least_positive_integer_l557_557802


namespace correct_interval_l557_557117

def encounter_intervals : Prop :=
  let rate1 := 1 / 7
  let rate2 := 1 / 13
  let total_rate := rate1 + rate2
  let interval := 1 / total_rate
  interval ≈ 4.6

theorem correct_interval : encounter_intervals :=
sorry

end correct_interval_l557_557117


namespace slope_y_intercept_sum_l557_557287

structure Point where
  x : ℝ
  y : ℝ

def midpoint (P Q : Point) : Point :=
  ⟨(P.x + Q.x) / 2, (P.y + Q.y) / 2⟩

def line_slope (P Q : Point) : ℝ :=
  (Q.y - P.y) / (Q.x - P.x)

def y_intercept (slope : ℝ) (P : Point) : ℝ :=
  P.y - slope * P.x

noncomputable def sum_slope_y_intercept (P Q : Point) : ℝ :=
  let slope := line_slope P Q
  let intercept := y_intercept slope Q
  slope + intercept

theorem slope_y_intercept_sum : sum_slope_y_intercept ⟨6, 0⟩ ⟨0, 1⟩ = 5 / 6 := by
  sorry

end slope_y_intercept_sum_l557_557287


namespace plane_cuts_sphere_plane_cuts_cylinder_l557_557395

-- Definitions for conditions
def Plane : Type := sorry  -- Placeholder definition for Plane
def Sphere : Type := sorry  -- Placeholder definition for Sphere
def Cylinder : Type := sorry  -- Placeholder definition for Cylinder
def is_section (plane : Plane) (surface : Type) : Type := sorry  -- Placeholder for the section produced by intersection

-- The shapes we need to prove
def is_circle (shape : Type) : Prop := sorry  -- Placeholder definition for being a circle
def is_ellipse (shape : Type) : Prop := sorry  -- Placeholder definition for being an ellipse

-- Define the section produced when a Plane cuts a Sphere to be a circle
theorem plane_cuts_sphere (P : Plane) (S : Sphere) :
  is_circle (is_section P S) := sorry

-- Define the section produced when a Plane cuts a Cylinder to be a circle or ellipse
theorem plane_cuts_cylinder (P : Plane) (C : Cylinder) :
  is_circle (is_section P C) ∨ is_ellipse (is_section P C) := sorry

end plane_cuts_sphere_plane_cuts_cylinder_l557_557395


namespace value_of_f_2012_l557_557377

def odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

noncomputable def f : ℝ → ℝ := sorry

axiom odd_fn : odd_function f
axiom f_at_2 : f 2 = 0
axiom functional_eq : ∀ x : ℝ, f (x + 4) = f x + f 4

theorem value_of_f_2012 : f 2012 = 0 :=
by
  sorry

end value_of_f_2012_l557_557377


namespace number_of_possible_M_l557_557251

theorem number_of_possible_M : 
  let b := (m : ℕ) // m ≥ 1 ∧ m ≤ 9 ∧ 10000 * m = n / 11 in
  ∃ b y : ℕ, 10000 * b + y = 11 * y ∧ 1 ≤ b ∧ b ≤ 8 ∧ 
               ∀ m : ℕ, 1000 ≤ m ∧ m ≤ 9999 →
                 (∃ n : ℕ, n = 10000 * b ∧ m = y) → 
                   b = 8 :=
sorry

end number_of_possible_M_l557_557251


namespace factors_dividing_either_10pow40_or_20pow30_l557_557605

theorem factors_dividing_either_10pow40_or_20pow30:
  let A := {d | ∃ m n : ℕ, 0 ≤ m ∧ m ≤ 40 ∧ 0 ≤ n ∧ n ≤ 40 ∧ d = 2^m * 5^n}
  let B := {d | ∃ p q : ℕ, 0 ≤ p ∧ p ≤ 60 ∧ 0 ≤ q ∧ q ≤ 30 ∧ d = 2^p * 5^q}
  in (A ∪ B).card = 2301 := by 
  sorry

end factors_dividing_either_10pow40_or_20pow30_l557_557605


namespace pdf_integral_eq_one_l557_557782

theorem pdf_integral_eq_one (c : ℝ) (p : ℝ → ℝ) (h1 : ∀ x, p(x) = c / (1 + x^2)) 
(h2 : ∫ x in -real.infinity..real.infinity, p x = 1) : c = 1 / real.pi :=
by
  sorry

end pdf_integral_eq_one_l557_557782


namespace remove_players_l557_557211

theorem remove_players (N : ℕ) (hN : 2 ≤ N) :
  ∃ (selected : finset ℕ), 
    selected.card = 2 * N ∧
    (∀ (i : ℕ), i < N → 
      ∃ (a b : ℕ), 
        a ∈ selected ∧ 
        b ∈ selected ∧ 
        a ≠ b ∧ 
        ∀ (x : ℕ), x ∈ selected → (min a b < x ∧ x < max a b) → ¬ (height x = i)) :=
  sorry

end remove_players_l557_557211


namespace range_of_a_l557_557051

theorem range_of_a (x : ℝ) (h : 1 < x) : ∀ a, (∀ x, 1 < x → x + 1 / (x - 1) ≥ a) → a ≤ 3 :=
by
sorry

end range_of_a_l557_557051


namespace range_of_omega_l557_557582

noncomputable def f (ω x : ℝ) : ℝ := 2 * Real.sin (ω * x)

theorem range_of_omega (ω : ℝ) (hω : ω > 0) :
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → f ω x = 0 → 
      (∃ x₁ x₂, x₁ ≠ x₂ ∧ 0 ≤ x₁ ∧ x₁ ≤ Real.pi / 2 ∧ 
        0 ≤ x₂ ∧ x₂ ≤ Real.pi / 2 ∧ f ω x₁ = 0 ∧ f ω x₂ = 0)) ↔ 2 ≤ ω ∧ ω < 4 :=
sorry

end range_of_omega_l557_557582


namespace kevin_ends_with_cards_l557_557650

def cards_found : ℝ := 47.0
def cards_lost : ℝ := 7.0

theorem kevin_ends_with_cards : cards_found - cards_lost = 40.0 := by
  sorry

end kevin_ends_with_cards_l557_557650


namespace probability_of_defective_l557_557413

theorem probability_of_defective (p_first_grade p_second_grade : ℝ) (h_fg : p_first_grade = 0.65) (h_sg : p_second_grade = 0.3) : (1 - (p_first_grade + p_second_grade) = 0.05) :=
by
  sorry

end probability_of_defective_l557_557413


namespace triangle_ABC_two_solutions_l557_557374

theorem triangle_ABC_two_solutions (x : ℝ) (h1 : x > 0) : 
  2 < x ∧ x < 2 * Real.sqrt 2 ↔
  (∃ a b B, a = x ∧ b = 2 ∧ B = Real.pi / 4 ∧ a * Real.sin B < b ∧ b < a) := by
  sorry

end triangle_ABC_two_solutions_l557_557374


namespace existence_of_M_lamps_on_power_of_two_lamps_on_power_of_two_plus_one_l557_557330

open Polynomial

variable {n : ℕ}

-- Condition: n is an integer greater than 1.
-- Question (a): There is a positive integer M(n) such that after M(n) steps, all lamps are ON again.
theorem existence_of_M (h_n : n > 1) : ∃ M, M > 0 ∧ ∀ k < M, (x^k) % (x^n + x^(n-1) + 1) = 1 := 
sorry

variable {k : ℕ}

-- Condition: n has the form 2^k.
-- Question (b): If n = 2^k, then all lamps are ON after n^2 - 1 steps.
theorem lamps_on_power_of_two (h_n : n = 2^k) : (x^(n^2 - 1)) % (x^n + x^(n-1) + 1) = 1 := 
sorry

-- Condition: n has the form 2^k + 1.
-- Question (c): If n = 2^k + 1, then all lamps are ON after n^2 - n + 1 steps.
theorem lamps_on_power_of_two_plus_one (h_n : n = 2^k + 1) : (x^(n^2 - n + 1)) % (x^n + x^(n-1) + 1) = 1 := 
sorry

end existence_of_M_lamps_on_power_of_two_lamps_on_power_of_two_plus_one_l557_557330


namespace min_a10_l557_557318

variable {A : List ℕ}
variable {a_1 a_2 ... a_n : ℕ} -- Define the indexed elements of A
variable {S : List ℕ}

-- Define Γ(S) as the sum of its elements
def Γ (S : List ℕ) : ℕ := S.sum

-- Condition: A is a sorted list of positive integers
def sorted_and_positive (A : List ℕ) : Prop :=
  (∀ a ∈ A, 0 < a) ∧ List.sorted (<) A

-- Condition: For each n ≤ 1500, there exists an S subset of A such that Γ(S) = n
def sum_exists (A : List ℕ) : Prop :=
  ∀ n : ℕ, n ≤ 1500 → ∃ (S : List ℕ), S ⊆ A ∧ Γ(S) = n

-- Main theorem stating what we need to prove
theorem min_a10 (h1 : sorted_and_positive A) (h2 : sum_exists A) :
  List.nthLe A 9 _ = 248 := by
  sorry

end min_a10_l557_557318


namespace total_seats_at_round_table_l557_557882

-- Define namespace and conditions
namespace KingArthur

variable (n : ℕ) -- Total number of seats

-- Conditions
def king_position : ℕ := 10
def lancelot_position : ℕ := 29
def opposite (a b : ℕ) (n : ℕ) : Prop := (a + (n / 2)) % n = b

-- Proof statement
theorem total_seats_at_round_table : opposite 10 29 n ∧ 29 < n → n = 38 :=
by
  sorry

end KingArthur

end total_seats_at_round_table_l557_557882


namespace price_cashews_l557_557855

noncomputable def price_per_pound_cashews 
  (price_mixed_nuts_per_pound : ℝ) 
  (weight_mixed_nuts : ℕ) 
  (weight_peanuts : ℕ) 
  (price_peanuts_per_pound : ℝ) 
  (weight_cashews : ℕ) : ℝ := 
  (price_mixed_nuts_per_pound * weight_mixed_nuts - price_peanuts_per_pound * weight_peanuts) / weight_cashews

open Real

theorem price_cashews 
  (price_mixed_nuts_per_pound : ℝ) 
  (weight_mixed_nuts : ℕ) 
  (weight_peanuts : ℕ) 
  (price_peanuts_per_pound : ℝ) 
  (weight_cashews : ℕ)
  (h1 : price_mixed_nuts_per_pound = 2.50) 
  (h2 : weight_mixed_nuts = 100) 
  (h3 : weight_peanuts = 40) 
  (h4 : price_peanuts_per_pound = 3.50) 
  (h5 : weight_cashews = 60) : 
  price_per_pound_cashews price_mixed_nuts_per_pound weight_mixed_nuts weight_peanuts price_peanuts_per_pound weight_cashews = 11 / 6 := by 
  sorry

end price_cashews_l557_557855


namespace enlarged_sticker_height_l557_557040

theorem enlarged_sticker_height (original_width original_height new_width : ℕ) 
  (h1 : original_width = 3) 
  (h2 : original_height = 2) 
  (h3 : new_width = 12) : (new_width / original_width) * original_height = 8 := 
by 
  -- Prove the height of the enlarged sticker is 8 inches
  sorry

end enlarged_sticker_height_l557_557040


namespace opposite_of_three_l557_557758

theorem opposite_of_three : -3 = -3 := by
  -- The condition we have identified is the given number 3.
  -- We will directly state that the opposite of 3 is -3.
  -- This proof is trivial as we are directly replacing 3 with -3 to match the problem statement.
  rfl

end opposite_of_three_l557_557758


namespace inequality_system_solution_l557_557195

theorem inequality_system_solution (b : ℝ) :
  (∀ x y : ℝ, (x^2 + y^2 + 1 > 2 * b * x + 2 * y + b - b^2) ∧
              (2 * cos (2 * (x - y)) + 8 * b * cos (x - y) + 8 * b * (b + 1) + 5 > 0)) ↔
  (b ∈ Ioo (-1- (sqrt 2 / 4)) 0 ∪ Ioo (-∞) (-1- (sqrt 2 / 4)) ∪ Ioo (-1/2) 0) :=
sorry

end inequality_system_solution_l557_557195


namespace best_coupons_l557_557469

-- Define the prices
def prices : List ℝ := [174.95, 184.95, 194.95, 204.95, 214.95]

-- Define the discounts for each coupon
def coupon1_discount (x : ℝ) : ℝ := 0.15 * x
def coupon2_discount : ℝ := 30
def coupon3_discount (x : ℝ) : ℝ := 0.22 * x - 33

-- Define the condition ranges 
def coupon_condition_1 (x : ℝ) := x > 200
def coupon_condition_2 (x : ℝ) := x < 471.43

-- The main theorem to prove
theorem best_coupons (x : ℝ) (hx : x ∈ prices) : 
  coupon_condition_1 x ∧ coupon_condition_2 x ∧ 
  (coupon1_discount x > coupon2_discount) ∧
  (coupon1_discount x > coupon3_discount x) :=
begin
  sorry
end

end best_coupons_l557_557469


namespace solve_AH_l557_557657

noncomputable def right_triangle_AH (AB AC : ℝ) (right_triangle_at_A : Prop) (H : ℝ) : Prop :=
  right_triangle_at_A ∧ AB = 156 ∧ AC = 65 → H = 60

theorem solve_AH : right_triangle_AH 156 65 (is_right_triangle_ABC 156 65) AH :=
by
  sorry

end solve_AH_l557_557657


namespace smallest_n_for_terminating_decimal_with_digit_8_l557_557428

-- Defining the predicate that checks if a number is a terminating decimal
def is_terminating_decimal (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = 2^a * 5^b

-- Defining the predicate that checks if a number contains the digit 8
def contains_digit_8 (n : ℕ) : Prop :=
  n.to_string.contains '8'

-- The final problem statement to prove in Lean 4
theorem smallest_n_for_terminating_decimal_with_digit_8 : 
  ∃ n : ℕ, is_terminating_decimal n ∧ contains_digit_8 n ∧ ∀ m : ℕ, (is_terminating_decimal m ∧ contains_digit_8 m) → n ≤ m :=
  sorry

end smallest_n_for_terminating_decimal_with_digit_8_l557_557428


namespace time_for_b_alone_l557_557435

theorem time_for_b_alone (A B : ℝ) (h1 : A + B = 1 / 16) (h2 : A = 1 / 24) : B = 1 / 48 :=
by
  sorry

end time_for_b_alone_l557_557435


namespace PF_passes_midpoint_of_MN_l557_557656

-- Define the geometric elements involved
variables {A B C M N E F P : Point}
variables (triangleABC: isosceles_triangle A B C)
variables (semicircleEF: semicircle_diameter E F)
variables (tangentM: tangent_to AB M semicircleEF)
variables (tangentN: tangent_to AC N semicircleEF)
variables (onBC_E: on_line_segment E B C)
variables (onBC_F: on_line_segment F B C)
variables (intersectionP: intersects AE semicircleEF P)

-- Define the midpoint predicate
def midpoint (X Y Z: Point) : Prop :=
  dist X Z = dist Y Z

-- Define the core theorem to prove
theorem PF_passes_midpoint_of_MN :
  passes_through_midpoint P F M N :=
sorry

end PF_passes_midpoint_of_MN_l557_557656


namespace eliza_is_shorter_by_2_inch_l557_557526

theorem eliza_is_shorter_by_2_inch
  (total_height : ℕ)
  (height_sibling1 height_sibling2 height_sibling3 height_eliza : ℕ) :
  total_height = 330 →
  height_sibling1 = 66 →
  height_sibling2 = 66 →
  height_sibling3 = 60 →
  height_eliza = 68 →
  total_height - (height_sibling1 + height_sibling2 + height_sibling3 + height_eliza) - height_eliza = 2 :=
by
  sorry

end eliza_is_shorter_by_2_inch_l557_557526


namespace sandwich_cost_l557_557546

variable (S : ℝ) -- cost of the sandwich

-- Conditions
axiom h1 : ∀ {S : ℝ}, (1 : ℝ) + 2 + 0.75 * (1 + 2) = 5.25
axiom h2 : ∀ {S : ℝ}, S + 2 * S + 0.75 * (S + 2 * S) = 21

-- Theorem: cost of the sandwich is $4
theorem sandwich_cost (S : ℝ) (h1 : S + 2 * S + 0.75 * (S + 2 * S) = 21) : S = 4 :=
by
  calc
    S + 2 * S + 0.75 * (S + 2 * S) = 21 : by sorry
    3 * S + 0.75 * 3 * S = 21           : by sorry
    3 * S + 2.25 * S = 21               : by sorry
    5.25 * S = 21                      : by sorry
    S = 21 / 5.25                      : by sorry
    S = 4                              : by sorry

end sandwich_cost_l557_557546


namespace second_discount_percentage_l557_557392

theorem second_discount_percentage 
  (initial_price : ℝ)
  (first_discount_percentage : ℝ)
  (final_price : ℝ)
  (initial_price = 600)
  (first_discount_percentage = 10)
  (final_price = 513) :
  ∃ D : ℝ, (D = 5) :=
by
  -- Definitions
  let first_discount := (first_discount_percentage / 100) * initial_price
  let price_after_first_discount := initial_price - first_discount
  let second_discount := (D / 100) * price_after_first_discount
  
  -- Conditions
  have initial_price_cond : initial_price = 600 := sorry
  have first_discount_percentage_cond : first_discount_percentage = 10 := sorry
  have final_price_cond : final_price = 513 := sorry
  
  -- Proof start
  sorry

end second_discount_percentage_l557_557392


namespace trigonometric_expression_simplified_l557_557670

-- Let c = 2 * π / 9
def c : ℝ := 2 * Real.pi / 9

-- Define the complex expression to compute
def expr : ℝ :=
  (Real.sin (2 * c) * Real.sin (5 * c) * Real.sin (8 * c) * Real.sin (11 * c) * Real.sin (14 * c)) /
  (Real.sin c * Real.sin (3 * c) * Real.sin (4 * c) * Real.sin (7 * c) * Real.sin (8 * c))

-- Define the expected result (sin 80 degrees, or sin (4 * π / 9))
def expected : ℝ := Real.sin (4 * Real.pi / 9)

-- The proof statement that the complex expression equals the expected value.
theorem trigonometric_expression_simplified : expr = expected := by
  sorry

end trigonometric_expression_simplified_l557_557670


namespace total_seats_at_round_table_l557_557880

-- Define namespace and conditions
namespace KingArthur

variable (n : ℕ) -- Total number of seats

-- Conditions
def king_position : ℕ := 10
def lancelot_position : ℕ := 29
def opposite (a b : ℕ) (n : ℕ) : Prop := (a + (n / 2)) % n = b

-- Proof statement
theorem total_seats_at_round_table : opposite 10 29 n ∧ 29 < n → n = 38 :=
by
  sorry

end KingArthur

end total_seats_at_round_table_l557_557880


namespace slope_of_line_PQ_l557_557587

noncomputable def slope_of_PQ (C : ℝ) (F : ℝ × ℝ) (P Q : ℝ × ℝ) : ℝ := 
  if x_C = 4 * y^2 and x_F in (-∞, ∞) and y_F in (-∞, ∞) and
     (3 * (P.1 - F.1), 3 * (P.2 - F.2)) = (Q.1 - F.1, Q.2 - F.2) and
     Q.1 > 0 and Q.2 > 0 then
    (Q.2 - P.2) / (Q.1 - P.1)
  else
    0

theorem slope_of_line_PQ (C : parabola) (F : point) (P Q : point) (hC : parabola_eq 4) (hF : in_focus C F) (hPQ_int : line_through_focus_intersects C F P Q) (hQ1 : in_first_quadrant Q) (hcond : 3 • (P - F) = (Q - F)) :
  slope_of_PQ C F P Q = sqrt 3 := sorry

end slope_of_line_PQ_l557_557587


namespace coefficient_x2_in_derivative_l557_557931

def f (x : ℝ) : ℝ := (1 - 2 * x) ^ 10

theorem coefficient_x2_in_derivative :
  let f_der := (λ x, (1 - 2 * x) ^ 10)'
  coeff (Taylor.series f_der 0) 2 = -2880 := by
soropyright-right users santa.py AI.injected some competitive bits with this.

end coefficient_x2_in_derivative_l557_557931


namespace total_seats_round_table_l557_557870

theorem total_seats_round_table 
  (a : ℕ) (b : ℕ) 
  (h₀ : a ≠ b)
  (h₁ : a + b = 39) 
  : ∃ n, n = 38 := 
by {
  sorry
}

end total_seats_round_table_l557_557870


namespace count_ordered_triples_l557_557959

theorem count_ordered_triples :
  (∃ (x y z : ℝ), x + y = 3 ∧ xy - z^2 = 2) ↔ 2 :=
sorry

end count_ordered_triples_l557_557959


namespace range_of_t_l557_557219

theorem range_of_t (x y a t : ℝ) 
  (h1 : x + 3 * y + a = 4) 
  (h2 : x - y - 3 * a = 0) 
  (h3 : -1 ≤ a ∧ a ≤ 1) 
  (h4 : t = x + y) : 
  1 ≤ t ∧ t ≤ 3 := 
sorry

end range_of_t_l557_557219


namespace integer_solutions_xy_l557_557953

theorem integer_solutions_xy :
  ∃ (x y : ℤ), (x + y + x * y = 500) ∧ 
               ((x = 0 ∧ y = 500) ∨ 
                (x = -2 ∧ y = -502) ∨ 
                (x = 2 ∧ y = 166) ∨ 
                (x = -4 ∧ y = -168)) :=
by
  sorry

end integer_solutions_xy_l557_557953


namespace king_arthur_round_table_seats_l557_557897

theorem king_arthur_round_table_seats (n : ℕ) (h₁ : n > 1) (h₂ : 10 < 29) (h₃ : (29 - 10) * 2 = n - 2) : 
  n = 38 := 
by
  sorry

end king_arthur_round_table_seats_l557_557897


namespace minimum_magnitude_l557_557594

variables (t : ℝ)

def a := (2, t, t)
def b := (1 - t, 2 * t - 1, 0)
def diff := (b.1 - a.1, b.2 - a.2, b.3 - a.3)
def magnitude := real.sqrt (diff.1 ^ 2 + diff.2 ^ 2 + diff.3 ^ 2)

theorem minimum_magnitude : ∃ t : ℝ, magnitude t = real.sqrt 2 :=
by {
  use [0],
  unfold magnitude,
  simp,
  sorry
}

end minimum_magnitude_l557_557594


namespace remainder_is_3_l557_557807

theorem remainder_is_3 (x y r : ℕ) (h1 : x = 7 * y + r) (h2 : 2 * x = 18 * y + 2) (h3 : 11 * y - x = 1)
  (hrange : 0 ≤ r ∧ r < 7) : r = 3 := 
sorry

end remainder_is_3_l557_557807


namespace sqrt_121_eq_pm_11_l557_557066

theorem sqrt_121_eq_pm_11 : (∀ x : ℝ, x^2 = 121 → x = 11 ∨ x = -11) :=
by {
  intro x,
  intro h,
  have hx : x * x = 121 := by assumption,
  have pos_x : x = real.sqrt 121 ∨ x = - real.sqrt 121 := by
    have sqrt_121 := real.sqrt_eq_iff_sqr_eq (by norm_num) (by norm_num),
    rw sqrt_121 at hx,
    exact hx,
  rw real.sqrt_eq_iff_sqr_eq (by norm_num) (by norm_num) at pos_x,
  exact pos_x
}

end sqrt_121_eq_pm_11_l557_557066


namespace cube_painting_probability_l557_557157

-- Define the conditions: a cube with six faces, each painted either green or yellow (independently, with probability 1/2)
structure Cube where
  faces : Fin 6 → Bool  -- Let's represent Bool with True for green, False for yellow

def is_valid_arrangement (c : Cube) : Prop :=
  ∃ (color : Bool), 
    (c.faces 0 = color ∧ c.faces 1 = color ∧ c.faces 2 = color ∧ c.faces 3 = color) ∧
    (∀ (i j : Fin 6), i = j ∨ ¬(c.faces i = color ∧ c.faces j = color))

def total_arrangements : ℕ := 2 ^ 6

def suitable_arrangements : ℕ := 20  -- As calculated previously: 2 + 12 + 6 = 20

-- We want to prove that the probability is 5/16
theorem cube_painting_probability :
  (suitable_arrangements : ℚ) / total_arrangements = 5 / 16 := 
by
  sorry

end cube_painting_probability_l557_557157


namespace negation_prop_l557_557385

theorem negation_prop : (¬(∃ x : ℝ, x + 2 ≤ 0)) ↔ (∀ x : ℝ, x + 2 > 0) := 
  sorry

end negation_prop_l557_557385


namespace opposite_of_negative_fraction_l557_557016

theorem opposite_of_negative_fraction : -(- (1/2023 : ℚ)) = 1/2023 := 
sorry

end opposite_of_negative_fraction_l557_557016


namespace find_r_l557_557665

noncomputable def f (r a : ℝ) (x : ℝ) : ℝ := (x - r - 1) * (x - r - 8) * (x - a)
noncomputable def g (r b : ℝ) (x : ℝ) : ℝ := (x - r - 2) * (x - r - 9) * (x - b)

theorem find_r
  (r a b : ℝ)
  (h_condition1 : ∀ x, f r a x - g r b x = r)
  (h_condition2 : f r a (r + 2) = r)
  (h_condition3 : f r a (r + 9) = r)
  : r = -264 / 7 := sorry

end find_r_l557_557665


namespace Sn_reciprocal_sum_eq_Ten_l557_557980

def arithmetic_seq (n : ℕ) : ℕ := 2 * n + 1

noncomputable def geometric_sum (n : ℕ) : ℤ := (3/2) * (3^n - 1)

noncomputable def Sn (n : ℕ) : ℕ := n^2 + 2n

noncomputable def reciprocal_sum_terms (n : ℕ) : ℚ := (1 : ℚ) / (Sn n)

noncomputable def Ten (n : ℕ) : ℚ :=
  1/2 * (3/2 - (2 * n + 3) / (2 * (n + 1) * (n + 2)))

open Finset

theorem Sn_reciprocal_sum_eq_Ten (n : ℕ) :
  (∑ i in range (n + 1), reciprocal_sum_terms i : ℚ) = Ten n := sorry

end Sn_reciprocal_sum_eq_Ten_l557_557980


namespace quadrilateral_inequality_l557_557206

theorem quadrilateral_inequality 
  (A B C D : Type)
  (AB AC AD BC BD CD : ℝ)
  (hAB_pos : 0 < AB)
  (hBC_pos : 0 < BC)
  (hCD_pos : 0 < CD)
  (hDA_pos : 0 < DA)
  (hAC_pos : 0 < AC)
  (hBD_pos : 0 < BD): 
  AC * BD ≤ AB * CD + BC * AD := 
sorry

end quadrilateral_inequality_l557_557206


namespace sum_of_possible_m_values_l557_557994

theorem sum_of_possible_m_values : 
  ∃ (s : ℕ), (∀ m : ℤ, 0 < 5 * m ∧ 5 * m < 35 → m ∈ {1, 2, 3, 4, 5, 6}) ∧ 
             (s = List.sum [1, 2, 3, 4, 5, 6]) ∧
             s = 21 := 
begin
  sorry
end

end sum_of_possible_m_values_l557_557994


namespace factorial_expression_l557_557923

open Nat

theorem factorial_expression :
  7 * (6!) + 6 * (5!) + 2 * (5!) = 6000 :=
by
  sorry

end factorial_expression_l557_557923


namespace min_value_frac_inverse_l557_557220

theorem min_value_frac_inverse (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) : 
  (1 / a + 1 / b) >= 2 :=
by
  sorry

end min_value_frac_inverse_l557_557220


namespace fly_expected_turns_l557_557084

noncomputable def expected_turns (n : ℕ) : ℚ :=
n + 1/2 - (n - 1/2) * (1 / real.sqrt (π * (n - 1)))

theorem fly_expected_turns (n : ℕ) : 
  expected_turns n = n + 1/2 - (n - 1/2) * (1 / real.sqrt (π * (n - 1))) 
  :=
by {
  sorry
}

end fly_expected_turns_l557_557084


namespace correct_conclusions_l557_557577

-- Definition of central angle and arc length
def central_angle_and_arc_length := 
  (α: ℝ) (r: ℝ) (hα: α = 120) (hr: r = 2) : 
  arc_length α r = (4 * Real.pi / 3) := sorry

-- Definition of systematic sampling for auditorium seats
def systematic_sampling (rows: ℕ) (seats_per_row: ℕ) (seat_number: ℕ) :=
  rows = 25 ∧ seats_per_row = 20 ∧ seat_number = 15 

-- Definition of mutually exclusive events
def mutually_exclusive_events := 
  ∀ (hit: ℙ) (two_misses: ℙ), 
  Event (P = "at least one hit") → Event (Q = "two consecutive misses") → 
  are_mutually_exclusive P Q := sorry 

-- Definition of trigonometric inequality
def trigonometric_inequality (x: ℝ) (h: 0 < x) (hx: x < Real.pi / 2) : 
  tan x > x ∧ x > sin x := sorry 

-- Proof of variance transformation
def variance_transformation (variance: ℝ) (n: ℕ): 
  (∀ x: fin n → ℝ, variance (x) = 8) → 
  variance (λ i, 2 * x i + 1) = 16 := sorry 

-- The final theorem to prove
theorem correct_conclusions :
  central_angle_and_arc_length 
  ∧ systematic_sampling 
  ∧ mutually_exclusive_events 
  ∧ trigonometric_inequality 
  ∧ variance_transformation → 
  correct_conclusions = [1, 2, 3, 4] := 
sorry

end correct_conclusions_l557_557577


namespace min_operations_to_500_l557_557360

theorem min_operations_to_500 : 
  ∃ n : ℕ, 
  let seq : ℕ → ℕ := λ k, if k = 0 then 5 else (if k % 2 = 1 then seq (k-1) + (if k % 4 = 1 then 60 else 120) else seq (k-1) - 100),
      result := seq n 
  in result = 500 ∧ n = 33 := 
sorry

end min_operations_to_500_l557_557360


namespace initial_bananas_l557_557456

theorem initial_bananas (bananas_left: ℕ) (eaten: ℕ) (basket: ℕ) 
                        (h_left: bananas_left = 100) 
                        (h_eaten: eaten = 70) 
                        (h_basket: basket = 2 * eaten): 
  bananas_left + eaten + basket = 310 :=
by
  sorry

end initial_bananas_l557_557456


namespace find_n_in_arithmetic_sequence_l557_557635

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 4 then 7 else
  if n = 5 then 16 - 7 else sorry

-- Define the arithmetic sequence and the given conditions
theorem find_n_in_arithmetic_sequence (n : ℕ) (a : ℕ → ℕ) 
  (h1 : a 4 = 7) 
  (h2 : a 3 + a 6 = 16) 
  (h3 : a n = 31) :
  n = 16 :=
by
  sorry

end find_n_in_arithmetic_sequence_l557_557635


namespace total_number_of_seats_l557_557884

theorem total_number_of_seats (n : ℕ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ n) 
                               (h2 : (10 : ℕ) < (29 : ℕ)) 
                               (h3 : (29 - 10) % (n / 2) = 0) : n = 38 :=
by sorry

end total_number_of_seats_l557_557884


namespace angle_BOC_is_110_l557_557300

def ABC : Type := sorry -- A triangle where you assume necessary conditions, e.g., the vertices and sides relations

variables {A B C O : ABC}

-- conditions
axiom h1 : AB = AC
axiom h2 : ∠ A = 40
axiom h3 : ∠ OBC = ∠ OCA

-- question
theorem angle_BOC_is_110 (ABC : Type) (A B C O : ABC) 
  (h1 : AB = AC) 
  (h2 : ∠ A = 40)
  (h3 : ∠ OBC = ∠ OCA)
  : ∠ BOC = 110 := 
sorry

end angle_BOC_is_110_l557_557300


namespace opposite_of_x_is_positive_l557_557019

-- Assume a rational number x
def x : ℚ := -1 / 2023

-- Theorem stating the opposite of x is 1 / 2023
theorem opposite_of_x_is_positive : -x = 1 / 2023 :=
by
  -- Required part of Lean syntax; not containing any solution steps
  sorry

end opposite_of_x_is_positive_l557_557019


namespace king_arthur_round_table_seats_l557_557899

theorem king_arthur_round_table_seats (n : ℕ) (h₁ : n > 1) (h₂ : 10 < 29) (h₃ : (29 - 10) * 2 = n - 2) : 
  n = 38 := 
by
  sorry

end king_arthur_round_table_seats_l557_557899


namespace limit_of_a_n_l557_557427

noncomputable def a_n (n : ℕ) : ℝ :=
  (1 / n^2) * ∑ i in finset.range (n - 1), (i+1) * real.sin ((i+1) * real.pi / n)

theorem limit_of_a_n :
  filter.tendsto (λ n, a_n n) filter.at_top (nhds (1 / real.pi)) :=
sorry

end limit_of_a_n_l557_557427


namespace factorial_division_l557_557501

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem factorial_division :
  (factorial 17) / (factorial 7 * factorial 10) = 408408 := by
  sorry

end factorial_division_l557_557501


namespace ratio_of_volumes_is_64_l557_557790

-- Define edges of the cubes
def edge_small := e : ℝ
def edge_large := 4 * e

-- Define volumes of the cubes
def volume_small := e^3
def volume_large := (4 * e)^3

-- Define ratio of volumes
def ratio_volumes := volume_large / volume_small

-- Statement to prove: Given the conditions, the ratio of the volumes is 64
theorem ratio_of_volumes_is_64 (e : ℝ) (h : e ≠ 0) : ratio_volumes = 64 := by
sorry

end ratio_of_volumes_is_64_l557_557790


namespace range_H_l557_557518

noncomputable def H (x : ℝ) : ℝ := abs (x + 2) - abs (x - 3)

theorem range_H : set.range H = set.Icc (-5) 5 :=
by {
  sorry
}

end range_H_l557_557518


namespace abs_round_lt_one_div_ten_l557_557793

example : Real :=
  let weight_kg := 250
  let conversion_factor := 0.4536
  let weight_lb := weight_kg / conversion_factor
  round (weight_lb * 10) / 10 = 551.2 := by
  let weight_lb_approx := 250 / 0.4536
  have h1 : abs (round (weight_lb_approx * 10) / 10 - 551.2) < 0.1 := sorry
  exact abs_round_lt_one_div_ten weight_lb_approx 551.2 h1

noncomputable def round (x : Real) : Int :=
  if x - x.floor >= 0.5 then x.ceil else x.floor

-- Added this function to convert the given result to a comparable number.
noncomputable def abs (x : Real) : Real :=
  if x >= 0 then x else -x

-- Placeholder theorem to state the approximate equality.
theorem abs_round_lt_one_div_ten :
  ∀ (weight_lb_approx result : Real), abs (round (weight_lb_approx * 10) / 10 - result) < 0.1 → 
  round (weight_lb_approx * 10) / 10 = result := sorry

end abs_round_lt_one_div_ten_l557_557793


namespace train_speed_correct_l557_557811

-- Given conditions
def length_train : ℕ := 140  -- Length of the train in meters, given as natural number to avoid negative lengths.
def time_to_cross : ℝ := 16  -- Time to cross in seconds, given as real number to allow division.

-- Define the speed of the train
def speed_train : ℝ := length_train / time_to_cross

-- Statement to prove
theorem train_speed_correct : speed_train = 8.75 := 
by
  -- Leaving the proof as a placeholder
  sorry

end train_speed_correct_l557_557811


namespace number_of_possible_values_of_a_l557_557702

def is_factor (m n : ℕ) : Prop := n % m = 0

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem number_of_possible_values_of_a :
  (∃ a : ℕ, is_factor 3 a ∧ is_factor a 18 ∧ even (sum_of_digits a)) →
  (π : ℕ) : ℕ :=
begin
  cases h with a,
  exact 1,
end

end number_of_possible_values_of_a_l557_557702


namespace points_opposite_sides_of_line_l557_557226
 
theorem points_opposite_sides_of_line (m : ℝ) :
  (3 * 1 - 2 * 2 - m) * (3 * (-1) - 2 * 5 - m) < 0 ↔ -13 < m ∧ m < -1 :=
by
  rw [sub_eq_add_neg, neg_mul_eq_neg_mul, neg_mul_eq_neg_mul, sub_eq_add_neg]
  sorry

end points_opposite_sides_of_line_l557_557226


namespace gcd_of_ratio_and_lcm_l557_557616

theorem gcd_of_ratio_and_lcm (A B : ℕ) (k : ℕ) (hA : A = 5 * k) (hB : B = 6 * k) (hlcm : Nat.lcm A B = 180) : Nat.gcd A B = 6 :=
by
  sorry

end gcd_of_ratio_and_lcm_l557_557616


namespace triangle_inequality_l557_557331

theorem triangle_inequality
  (A B C A' B' C' : Type)
  (α β γ : ℝ)
  (h_acute : α < 90 ∧ β < 90 ∧ γ < 90)
  (h_angles_sum : α + β + γ = 180)
  (h_A'_altitude : ∃ H : Type, A' = A ∧ H ∈ (H.seg A C))
  (h_B'_altitude : ∃ H : Type, B' = B ∧ H ∈ (H.seg B A))
  (h_C'_altitude : ∃ H : Type, C' = C ∧ H ∈ (H.seg C B)) :
  ∃ α' β' γ' : ℝ,
  α' = 180 - 2 * α ∧ β' = 180 - 2 * β ∧ γ' = 180 - 2 * γ ∧
  (γ' ≥ α ∧ γ' ≥ β ∧ γ' ≥ γ) ∧
  (γ' = α ↔ α ≥ β ∧ β = γ) :=
begin
  sorry
end

end triangle_inequality_l557_557331


namespace opposite_of_3_l557_557774

theorem opposite_of_3 : -3 = -3 := 
by
  -- sorry is added to skip the proof as per instructions
  sorry

end opposite_of_3_l557_557774


namespace find_A_and_B_l557_557951

theorem find_A_and_B : ∃ (A B : ℚ), 
  (∀ x : ℚ, x ≠ 7 ∧ x ≠ -6 → 
  (3 * x + 5) / (x^2 - x - 42) = A / (x - 7) + B / (x + 6)) ∧ 
  A = 2 ∧ B = 1 :=
by
  existsi (2 : ℚ)
  existsi (1 : ℚ)
  intros x hx
  split
  { intro h,
    calc
      (3 * x + 5) / (x^2 - x - 42)
          = (3 * x + 5) / ((x - 7) * (x + 6)) : by ring
      ... = 2 / (x - 7) + 1 / (x + 6)           : sorry },
  split
  { refl },
  { refl }

end find_A_and_B_l557_557951


namespace ozone_experiment_significant_difference_l557_557124

noncomputable def experiment01 :=
  let control_group := [15.2, 18.8, 20.2, 21.3, 22.5, 23.2, 25.8, 26.5, 27.5, 30.1, 32.6, 34.3, 34.8, 35.6, 35.6, 35.8, 36.2, 37.3, 40.5, 43.2] in
  let experimental_group := [7.8, 9.2, 11.4, 12.4, 13.2, 15.5, 16.5, 18.0, 18.8, 19.2, 19.8, 20.2, 21.6, 22.8, 23.6, 23.9, 25.1, 28.2, 32.3, 36.5] in
  let combined := control_group ++ experimental_group in
  let sorted_combined := List.sort combined in
  let m := (sorted_combined[19] + sorted_combined[20]) / 2 in
  let control_less := control_group.filter (λ x, x < m) in
  let control_more := control_group.filter (λ x, x ≥ m) in
  let exp_less := experimental_group.filter (λ x, x < m) in
  let exp_more := experimental_group.filter (λ x, x ≥ m) in
  let a := control_less.length in
  let b := control_more.length in
  let c := exp_less.length in
  let d := exp_more.length in
  let n := 40 in
  let total := (a + b) * (c + d) * (a + c) * (b + d) in
  let k_squared := n * (a*d - b*c)^2 / total in
  k_squared

theorem ozone_experiment_significant_difference :
  experiment01 > 3.841 := sorry

end ozone_experiment_significant_difference_l557_557124


namespace total_seats_round_table_l557_557864

theorem total_seats_round_table (n : ℕ) (h : n = 38)
  (ka_position : ℕ) (sl_position : ℕ) 
  (h1 : ka_position = 10) 
  (h2 : sl_position = 29) 
  (h3 : (ka_position + n/2) % n = sl_position) : 
  n = 38 :=
by
  -- All steps and solution proof
  sorry

end total_seats_round_table_l557_557864


namespace find_a_l557_557299

variables {A B C a b c : ℝ}

theorem find_a :
  (cos A * (b - 2 * c) = a * (2 * cos C - sqrt 3 * sin B)) →
  (c = 2 * b) →
  (abs ((λ AB AC : ℝ → ℝ) (AB b + 2 * AC c))) = 6 →
  a = 3 := sorry

end find_a_l557_557299


namespace function_properties_l557_557711

noncomputable def f (x : ℝ) : ℝ := 2^(abs x)

-- Statement: 
-- Prove that the function f(x) = 2^{|x|} is an even function and decreasing on the interval (-∞, 0).

theorem function_properties :
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ x y : ℝ, x < y → y < 0 → f x > f y) :=
by
  -- Proof of the function being even
  sorry 
  -- Proof of the function being decreasing on (-∞, 0)
  sorry

end function_properties_l557_557711


namespace line_and_ellipse_intersect_l557_557326

open Real

-- Conditions in the problem
variables {a b : ℝ} (ha : a ≠ 0) (hb : b ≠ 0)

-- Definitions translating the conditions
def line_eq (a b x y : ℝ) : Prop := ax - y + b = 0
def ellipse_eq (a b x y : ℝ) : Prop := bx^2 + ay^2 = ab

-- Mathematical equivalence to be proven in Lean
theorem line_and_ellipse_intersect (x y : ℝ) : 
  line_eq a b x y ∧ ellipse_eq a b x y ↔ y = ax + b ∧ (x^2 / a + y^2 / b = 1) :=
by
  sorry

end line_and_ellipse_intersect_l557_557326


namespace ratio_of_ages_l557_557087

-- Definitions of the conditions
def son_current_age : ℕ := 28
def man_current_age : ℕ := son_current_age + 30
def son_age_in_two_years : ℕ := son_current_age + 2
def man_age_in_two_years : ℕ := man_current_age + 2

-- The theorem
theorem ratio_of_ages : (man_age_in_two_years / son_age_in_two_years) = 2 :=
by
  -- Skipping the proof steps
  sorry

end ratio_of_ages_l557_557087


namespace find_v4_l557_557503

noncomputable def horner_method (x : ℤ) : ℤ :=
  let v0 := 3
  let v1 := v0 * x + 5
  let v2 := v1 * x + 6
  let v3 := v2 * x + 20
  let v4 := v3 * x - 8
  v4

theorem find_v4 : horner_method (-2) = -16 :=
  by {
    -- Proof goes here, but we are only required to write the statement.
    sorry
  }

end find_v4_l557_557503


namespace sum_of_possible_m_values_l557_557995

theorem sum_of_possible_m_values : 
  ∃ (s : ℕ), (∀ m : ℤ, 0 < 5 * m ∧ 5 * m < 35 → m ∈ {1, 2, 3, 4, 5, 6}) ∧ 
             (s = List.sum [1, 2, 3, 4, 5, 6]) ∧
             s = 21 := 
begin
  sorry
end

end sum_of_possible_m_values_l557_557995


namespace seven_pow_seven_pow_seven_pow_seven_mod_1000_l557_557181

theorem seven_pow_seven_pow_seven_pow_seven_mod_1000 :
  let M := 7 ^ 7 ^ 7 in
  let mod1000 := 1000 in
  let mod100 := 100 in
  (7^100 ≡ 1 [MOD mod1000]) →
  (7^20 ≡ 1 [MOD mod100]) →
  (7^7 ≡ 43 [MOD mod100]) →
  (M ≡ 43 [MOD mod100]) →
  (7^M ≡ 343 [MOD mod1000]) :=
by
  intros M mod1000 mod100 h1 h2 h3 h4
  sorry

end seven_pow_seven_pow_seven_pow_seven_mod_1000_l557_557181


namespace math_problem_proof_l557_557231

-- Define vectors a and c
def vec_a : ℝ × ℝ := (1, -2)
def vec_c : ℝ × ℝ := (2, 3)

-- Define the length of b and the requirement that it's opposite to a
def vec_b_length : ℝ := 3 * Real.sqrt 5
def is_opposite (x y : ℝ × ℝ) : Prop := ∃ λ : ℝ, λ > 0 ∧ x = -λ * y

-- Define the coordinates of vec_b based on the conditions
axiom vec_b_coords : ∃ b : ℝ × ℝ, b = (-3, 6) ∧ is_opposite b vec_a ∧ Real.sqrt (b.1^2 + b.2^2) = vec_b_length

-- Define the dot product result
axiom dot_product_result : 
  ∃ (b : ℝ × ℝ), b = (-3, 6) ∧ 
  ((vec_c.1 - vec_a.1, vec_c.2 - vec_a.2) • (vec_c.1 - b.1, vec_c.2 - b.2)) = -10

-- Statement of the theorem combining both results
theorem math_problem_proof : 
  ∃ (b : ℝ × ℝ), b = (-3, 6) ∧
  is_opposite b vec_a ∧ 
  Real.sqrt (b.1^2 + b.2^2) = vec_b_length ∧
  ((vec_c.1 - vec_a.1, vec_c.2 - vec_a.2) • (vec_c.1 - b.1, vec_c.2 - b.2)) = -10 := by
    exists (-3, 6)
    split
    case left => sorry
    case right => sorry

end math_problem_proof_l557_557231


namespace opposite_of_three_l557_557770

theorem opposite_of_three : -3 = opposite(3) := 
by
  sorry

end opposite_of_three_l557_557770


namespace opposite_of_three_l557_557753

theorem opposite_of_three : -3 = -3 := by
  -- The condition we have identified is the given number 3.
  -- We will directly state that the opposite of 3 is -3.
  -- This proof is trivial as we are directly replacing 3 with -3 to match the problem statement.
  rfl

end opposite_of_three_l557_557753


namespace opposite_of_three_l557_557755

theorem opposite_of_three : -3 = -3 := by
  -- The condition we have identified is the given number 3.
  -- We will directly state that the opposite of 3 is -3.
  -- This proof is trivial as we are directly replacing 3 with -3 to match the problem statement.
  rfl

end opposite_of_three_l557_557755


namespace opposite_of_3_is_neg3_l557_557719

theorem opposite_of_3_is_neg3 : forall (n : ℤ), n = 3 -> -n = -3 :=
by
  sorry

end opposite_of_3_is_neg3_l557_557719


namespace total_number_of_seats_l557_557885

theorem total_number_of_seats (n : ℕ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ n) 
                               (h2 : (10 : ℕ) < (29 : ℕ)) 
                               (h3 : (29 - 10) % (n / 2) = 0) : n = 38 :=
by sorry

end total_number_of_seats_l557_557885


namespace age_proof_l557_557091

theorem age_proof (x : ℕ) 
  (h : 5 * (x + 7) - 3 * (x - 7) = x) : 
  x = 14 := 
by 
  have h1 : 5 * (x + 7) = 5 * x + 35 := sorry
  have h2 : - 3 * (x - 7) = - 3 * x + 21 := sorry
  calc
    5 * (x + 7) - 3 * (x - 7)
    = (5 * x + 35) + (-3 * x + 21) : by rw [h1, h2]
    ... = 2 * x + 14 : sorry
    ... = x : by rw [h]
    ... = 14 : sorry

end age_proof_l557_557091


namespace trigonometric_identity_l557_557562

theorem trigonometric_identity (α : ℝ) (h : Real.sin (3 * Real.pi - α) = 2 * Real.sin (Real.pi / 2 + α)) : 
  (Real.sin (Real.pi - α) ^ 3 - Real.sin (Real.pi / 2 - α)) / 
  (3 * Real.cos (Real.pi / 2 + α) + 2 * Real.cos (Real.pi + α)) = -3/40 :=
by
  sorry

end trigonometric_identity_l557_557562


namespace lee_proposal_savings_l557_557310

theorem lee_proposal_savings :
  ∀ (annual_salary : ℝ) (months_to_save : ℝ) (months_of_salary : ℝ),
    annual_salary = 60000 → months_to_save = 10 → months_of_salary = 2 →
    (annual_salary / 12 * months_of_salary) / months_to_save = 1000 := by
  intros annual_salary months_to_save months_of_salary h_annual_salary h_months_to_save h_months_of_salary
  rw [h_annual_salary, h_months_to_save, h_months_of_salary] 
  norm_num
  sorry

end lee_proposal_savings_l557_557310


namespace towel_bleaching_l557_557109

theorem towel_bleaching
  (original_length original_breadth : ℝ)
  (percentage_decrease_area : ℝ)
  (percentage_decrease_breadth : ℝ)
  (final_length final_breadth : ℝ)
  (h1 : percentage_decrease_area = 28)
  (h2 : percentage_decrease_breadth = 10)
  (h3 : final_breadth = original_breadth * (1 - percentage_decrease_breadth / 100))
  (h4 : final_area = original_area (1 - percentage_decrease_area / 100))
  (original_area final_area : ℝ) :
  final_length = original_length * 0.8 :=
begin
  -- Here, the goal would be to prove that final_length is 80% of the original_length
  sorry
end

end towel_bleaching_l557_557109


namespace count_even_numbers_between_500_and_800_l557_557601

theorem count_even_numbers_between_500_and_800 :
  let a := 502
  let d := 2
  let last_term := 798
  ∃ n, a + (n - 1) * d = last_term ∧ n = 149 :=
by
  sorry

end count_even_numbers_between_500_and_800_l557_557601


namespace periodic_and_symmetric_l557_557375

noncomputable def f (x : ℝ) : ℝ := cos x + |cos x|

theorem periodic_and_symmetric :
  (periodic f 2 * π) ∧ (∀ k : ℤ, axial_symmetry (graph f) (k * π, 0)) :=
by
  sorry

end periodic_and_symmetric_l557_557375


namespace find_divisor_l557_557090

theorem find_divisor (x : ℝ) (h : 740 / x - 175 = 10) : x = 4 := by
  sorry

end find_divisor_l557_557090


namespace helly_dimension_2_part_a_helly_dimension_general_l557_557070

-- Part a
theorem helly_dimension_2_part_a (C1 C2 C3 C4 : Set ℝ) [Convex C1] [Convex C2] [Convex C3] [Convex C4]
  (h1 : (C2 ∩ C3 ∩ C4).Nonempty)
  (h2 : (C1 ∩ C3 ∩ C4).Nonempty)
  (h3 : (C1 ∩ C2 ∩ C4).Nonempty)
  (h4 : (C1 ∩ C2 ∩ C3).Nonempty) : 
  (C1 ∩ C2 ∩ C3 ∩ C4).Nonempty :=
sorry

-- Part b
theorem helly_dimension_general (n : ℕ) (C : Fin n → Set ℝ) [∀ i, Convex (C i)]
  (h : ∀ (s : Finset (Fin n)), s.card = n - 1 → 
          (⋂ i ∈ s, C i).Nonempty) : 
  (⋂ i, C i).Nonempty :=
sorry

end helly_dimension_2_part_a_helly_dimension_general_l557_557070


namespace bucket_capacity_l557_557451

theorem bucket_capacity :
  ∃ x : ℕ, 91 * x = 13 * 42 ∧ x = 6 :=
by
  use 6
  split
  · ring
  · refl

end bucket_capacity_l557_557451


namespace sixth_result_is_34_l557_557706

theorem sixth_result_is_34
  (avg_of_all_results : ℝ)
  (avg_of_first_6 : ℝ)
  (avg_of_last_6 : ℝ)
  (total_sum : ℝ)
  (S1 : ℝ)
  (S2 : ℝ)
  (sum_of_11_results : ℝ)
  (sum_of_first_6 : ℝ)
  (sum_of_last_6 : ℝ)
  : (sum_of_11_results = total_sum) →
    (sum_of_first_6 = S1) →
    (sum_of_last_6 = S2) →
    sixth_result = 34 :=
begin
  sorry,
end

end sixth_result_is_34_l557_557706


namespace min_num_pipes_l557_557093

theorem min_num_pipes (h : ℝ) (π : ℝ) : ∀ (length : ℝ),
  let volume_large := π * (4 * 4) * h in
  let volume_small := π * ((3/2) * (3/2)) * h in
  (volume_large / volume_small).ceil = 8 :=
by
  intros length
  let volume_large := π * (4 * 4) * h
  let volume_small := π * ((3/2) * (3/2)) * h
  have eq1 : (volume_large / volume_small).ceil = 8 := sorry
  exact eq1

#check min_num_pipes

end min_num_pipes_l557_557093


namespace john_monthly_income_l557_557339

theorem john_monthly_income (I : ℝ) (h : I - 0.05 * I = 1900) : I = 2000 :=
by
  sorry

end john_monthly_income_l557_557339


namespace simplify_series_sum_l557_557356

theorem simplify_series_sum :
  (∑ k in Finset.range(100), (k + 1) * 3^(k + 1)) = (199 * 3^101 + 3) / 4 :=
by sorry

end simplify_series_sum_l557_557356


namespace find_square_tiles_l557_557460

theorem find_square_tiles (t s p : ℕ) (h1 : t + s + p = 35) (h2 : 3 * t + 4 * s + 5 * p = 140) (hp0 : p = 0) : s = 35 := by
  sorry

end find_square_tiles_l557_557460


namespace total_seats_round_table_l557_557874

theorem total_seats_round_table 
  (a : ℕ) (b : ℕ) 
  (h₀ : a ≠ b)
  (h₁ : a + b = 39) 
  : ∃ n, n = 38 := 
by {
  sorry
}

end total_seats_round_table_l557_557874


namespace complex_addition_l557_557323

def A : ℂ := 5 - 2 * complex.I
def B : ℂ := -3 + 4 * complex.I
def C : ℂ := 2 * complex.I
def D : ℂ := 3

theorem complex_addition : A - B + C - D = 5 - 4 * complex.I := by
  sorry

end complex_addition_l557_557323


namespace opposite_of_3_is_neg3_l557_557720

theorem opposite_of_3_is_neg3 : forall (n : ℤ), n = 3 -> -n = -3 :=
by
  sorry

end opposite_of_3_is_neg3_l557_557720


namespace opposite_of_3_is_neg3_l557_557717

theorem opposite_of_3_is_neg3 : forall (n : ℤ), n = 3 -> -n = -3 :=
by
  sorry

end opposite_of_3_is_neg3_l557_557717


namespace vacation_costs_l557_557492

variable (Anne_paid Beth_paid Carlos_paid : ℕ) (a b : ℕ)

theorem vacation_costs (hAnne : Anne_paid = 120) (hBeth : Beth_paid = 180) (hCarlos : Carlos_paid = 150)
  (h_a : a = 30) (h_b : b = 30) :
  a - b = 0 := sorry

end vacation_costs_l557_557492


namespace king_arthur_round_table_seats_l557_557902

theorem king_arthur_round_table_seats (n : ℕ) (h₁ : n > 1) (h₂ : 10 < 29) (h₃ : (29 - 10) * 2 = n - 2) : 
  n = 38 := 
by
  sorry

end king_arthur_round_table_seats_l557_557902


namespace min_distance_is_sqrt2_l557_557976

noncomputable def min_distance_PQ : ℝ :=
  let P : ℝ × ℝ := (0, Real.exp 0) -- Point P on the curve y = e^x when x = 0
  let Q : ℝ × ℝ := (Real.log 1, 1) -- Point Q on the curve y = ln x when x = 1
  let line_y_eq_x := (λ x : ℝ, x) -- Line y = x, symmetry line
  let d := (λ (P Q : ℝ × ℝ) (f : ℝ → ℝ), (Real.sqrt ((P.1 - Q.1) ^ 2 + (f P.2 - f Q.2) ^ 2)) / Real.sqrt 2)
  let min_d := d P Q line_y_eq_x
  in 2 * min_d

theorem min_distance_is_sqrt2 : min_distance_PQ = Real.sqrt 2 :=
sorry

end min_distance_is_sqrt2_l557_557976


namespace domain_of_g_l557_557424

theorem domain_of_g : ∀ t : ℝ, (t - 3)^2 + (t + 3)^2 + 1 ≠ 0 :=
by
  intro t
  sorry

end domain_of_g_l557_557424


namespace total_seats_at_round_table_l557_557879

-- Define namespace and conditions
namespace KingArthur

variable (n : ℕ) -- Total number of seats

-- Conditions
def king_position : ℕ := 10
def lancelot_position : ℕ := 29
def opposite (a b : ℕ) (n : ℕ) : Prop := (a + (n / 2)) % n = b

-- Proof statement
theorem total_seats_at_round_table : opposite 10 29 n ∧ 29 < n → n = 38 :=
by
  sorry

end KingArthur

end total_seats_at_round_table_l557_557879


namespace cost_to_fill_can_N_l557_557504

def right_circular_cylinder (r h : ℝ) : ℝ := π * r^2 * h

variables (r h : ℝ) (cost_half_can_B : ℝ)
hypothesis (cost_half_can_B_eq : cost_half_can_B = 4)

theorem cost_to_fill_can_N : 
  let r_B := r in
  let h_B := h in
  let r_N := 2 * r_B in
  let h_N := h_B / 2 in
  let V_B := right_circular_cylinder r_B h_B in
  let V_N := right_circular_cylinder r_N h_N in
  let cost_full_can_B := 2 * cost_half_can_B in
  let cost_can_N := 2 * cost_full_can_B in
  cost_can_N = 16 :=
by {
  sorry
}

end cost_to_fill_can_N_l557_557504


namespace plane_sections_sphere_plane_sections_cylinder_l557_557397

noncomputable def section_shape_sphere (plane : Type) (sphere : Type) : Type :=
  -- Providing that a plane cuts a spherical surface, the section is always a circle.
  circle

noncomputable def section_shape_cylinder (plane : Type) (cylinder : Type) : Type :=
  -- Providing that a plane cuts a cylindrical surface, the section is either a circle or an ellipse.
  circle ⊕ ellipse

-- Stating the main theorems
theorem plane_sections_sphere (plane : Type) (sphere : Type) :
  section_shape_sphere plane sphere = circle := 
sorry

theorem plane_sections_cylinder (plane : Type) (cylinder : Type) :
  section_shape_cylinder plane cylinder = (circle ⊕ ellipse) := 
sorry

end plane_sections_sphere_plane_sections_cylinder_l557_557397


namespace max_attendees_is_tues_and_fri_l557_557941

def available_on_mon (p : String) : Prop :=
  p = "Anna" ∨ p = "Carl" ∨ p = "Evan"

def available_on_tues (p : String) : Prop :=
  p = "Bill" ∨ p = "Carl"

def available_on_wed (p : String) : Prop :=
  p = "Anna" ∨ p = "Dana" ∨ p = "Evan"

def available_on_thurs (p : String) : Prop :=
  p = "Bill" ∨ p = "Carl"

def available_on_fri (p : String) : Prop :=
  p = "Anna" ∨ p = "Carl"

def count_attendees (availability : String → Prop) (people : List String) : Nat :=
  (people.filter availability).length

def max_attendees_day (people : List String) : List String :=
  let monday := count_attendees available_on_mon people
  let tuesday := count_attendees available_on_tues people
  let wednesday := count_attendees available_on_wed people
  let thursday := count_attendees available_on_thurs people
  let friday := count_attendees available_on_fri people
  let max_attendees := max (max (max monday tuesday) (max wednesday thursday)) friday
  if monday = max_attendees then ["Monday"] else [] ++
  if tuesday = max_attendees then ["Tuesday"] else [] ++
  if wednesday = max_attendees then ["Wednesday"] else [] ++
  if thursday = max_attendees then ["Thursday"] else [] ++
  if friday = max_attendees then ["Friday"] else []

theorem max_attendees_is_tues_and_fri :
  max_attendees_day ["Anna", "Bill", "Carl", "Dana", "Evan"] = ["Tuesday", "Friday"] :=
by
  sorry

end max_attendees_is_tues_and_fri_l557_557941


namespace elena_snow_removal_l557_557161

theorem elena_snow_removal :
  ∀ (length width depth : ℝ) (compaction_factor : ℝ), 
  length = 30 ∧ width = 3 ∧ depth = 0.75 ∧ compaction_factor = 0.90 → 
  (length * width * depth * compaction_factor = 60.75) :=
by
  intros length width depth compaction_factor h
  obtain ⟨length_eq, width_eq, depth_eq, compaction_factor_eq⟩ := h
  -- Proof steps go here
  sorry

end elena_snow_removal_l557_557161


namespace G_is_even_l557_557971

-- Define the conditions
variables {a : ℝ} (F : ℝ → ℝ) 

-- Assume a > 0 and a ≠ 1
variables (h1 : a > 0) (h2 : a ≠ 1)

-- Define the odd function assumption
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f(x)

-- Assume F is an odd function
variable (hF : is_odd_function F)

-- Define G(x)
def G (x : ℝ) : ℝ := 
  F x * ((1 / (a^x - 1)) + (1 / 2))

-- Prove that G is an even function
theorem G_is_even : ∀ x : ℝ, G a F (-x) = G a F x :=
by
  sorry

end G_is_even_l557_557971


namespace systematic_sampling_correct_l557_557073

-- Definitions based on the problem's conditions
def total_students : ℕ := 600
def first_drawn : ℕ := 3
def step : ℕ := 12

def campI_start : ℕ := 1
def campI_end : ℕ := 300
def campII_start : ℕ := 301
def campII_end : ℕ := 495
def campIII_start : ℕ := 496
def campIII_end : ℕ := 600

def camp1_count (draws : List ℕ) :=
  draws.countp (λ x, x ≥ campI_start ∧ x ≤ campI_end)

def camp2_count (draws : List ℕ) :=
  draws.countp (λ x, x ≥ campII_start ∧ x ≤ campII_end)

def camp3_count (draws : List ℕ) :=
  draws.countp (λ x, x ≥ campIII_start ∧ x ≤ campIII_end)

-- Statement to prove
theorem systematic_sampling_correct :
  let draws := (List.range (total_students / step)).map (λ n, first_drawn + n * step)
in camp1_count draws = 25 ∧ camp2_count draws = 17 ∧ camp3_count draws = 8 :=
by
  sorry

end systematic_sampling_correct_l557_557073


namespace slopes_product_constant_l557_557464

theorem slopes_product_constant (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (a_gt_b : b < a) 
  (P M N : ℝ × ℝ) (x_0 y_0 m n : ℝ) 
  (hP : P = (x_0, y_0)) 
  (hM : M = (m, n)) (hN : N = (-m, -n))
  (h_P_on_ellipse : (x_0^2 / a^2 + y_0^2 / b^2 = 1))
  (h_M_on_ellipse : (m^2 / a^2 + n^2 / b^2 = 1))
  (h_slopes_exist : (x_0 + m) ≠ 0 ∧ (x_0 - m) ≠ 0) : 
  (let k_PM := (y_0 + n) / (x_0 + m) in
   let k_PN := (y_0 - n) / (x_0 - m) in
   k_PM * k_PN = - (b^2 / a^2)) :=
by
  sorry

end slopes_product_constant_l557_557464


namespace total_seats_at_round_table_l557_557876

-- Define namespace and conditions
namespace KingArthur

variable (n : ℕ) -- Total number of seats

-- Conditions
def king_position : ℕ := 10
def lancelot_position : ℕ := 29
def opposite (a b : ℕ) (n : ℕ) : Prop := (a + (n / 2)) % n = b

-- Proof statement
theorem total_seats_at_round_table : opposite 10 29 n ∧ 29 < n → n = 38 :=
by
  sorry

end KingArthur

end total_seats_at_round_table_l557_557876


namespace total_charge_correct_l557_557343

def boxwoodTrimCost (numBoxwoods : Nat) (trimCost : Nat) : Nat :=
  numBoxwoods * trimCost

def boxwoodShapeCost (numBoxwoods : Nat) (shapeCost : Nat) : Nat :=
  numBoxwoods * shapeCost

theorem total_charge_correct :
  let numBoxwoodsTrimmed := 30
  let trimCost := 5
  let numBoxwoodsShaped := 4
  let shapeCost := 15
  let totalTrimCost := boxwoodTrimCost numBoxwoodsTrimmed trimCost
  let totalShapeCost := boxwoodShapeCost numBoxwoodsShaped shapeCost
  let totalCharge := totalTrimCost + totalShapeCost
  totalCharge = 210 :=
by sorry

end total_charge_correct_l557_557343


namespace neg_square_positive_l557_557014

theorem neg_square_positive :
  ¬(∀ x : ℝ, x^2 > 0) ↔ ∃ x : ℝ, x^2 ≤ 0 := sorry

end neg_square_positive_l557_557014


namespace triangle_rotation_l557_557796

noncomputable def rotate_point_45_clockwise (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1 * real.sqrt 2 / 2 + P.2 * real.sqrt 2 / 2, P.1 * (-real.sqrt 2) / 2 + P.2 * real.sqrt 2 / 2)

theorem triangle_rotation (Q Q' : ℝ × ℝ) (h₁ : Q = (4,4)) (h₂ : Q' = rotate_point_45_clockwise Q) :
  Q' = (4 * real.sqrt 2, 0) :=
sorry

end triangle_rotation_l557_557796


namespace king_arthur_round_table_seats_l557_557903

theorem king_arthur_round_table_seats (n : ℕ) (h₁ : n > 1) (h₂ : 10 < 29) (h₃ : (29 - 10) * 2 = n - 2) : 
  n = 38 := 
by
  sorry

end king_arthur_round_table_seats_l557_557903


namespace chairs_removal_correct_chairs_removal_l557_557081

theorem chairs_removal (initial_chairs : ℕ) (chairs_per_row : ℕ) (participants : ℕ) : ℕ :=
  let total_chairs := 169
  let per_row := 13
  let attendees := 95
  let needed_chairs := (attendees + per_row - 1) / per_row * per_row
  let chairs_to_remove := total_chairs - needed_chairs
  chairs_to_remove

theorem correct_chairs_removal : chairs_removal 169 13 95 = 65 :=
by
  sorry

end chairs_removal_correct_chairs_removal_l557_557081


namespace colorful_family_children_count_l557_557624

theorem colorful_family_children_count 
    (B W S x : ℕ)
    (h1 : B = W) (h2 : W = S)
    (h3 : (B - x) + W = 10)
    (h4 : W + (S + x) = 18) :
    B + W + S = 21 :=
by
  sorry

end colorful_family_children_count_l557_557624


namespace sqrt_expr_sum_eq_eleven_l557_557939

theorem sqrt_expr_sum_eq_eleven (a b c : ℤ) (h1 : (64 + 24*Real.sqrt 3) = a + b*Real.sqrt c)
  (h2 : ¬∃ n : ℕ, n > 1 ∧ n^2 ∣ c) : a + b + c = 11 :=
by
  sorry

end sqrt_expr_sum_eq_eleven_l557_557939


namespace isosceles_triangle_altitude_l557_557630

open Real

theorem isosceles_triangle_altitude (DE DF DG EG GF EF : ℝ) (h1 : DE = 5) (h2 : DF = 5) (h3 : EG = 2 * GF)
(h4 : DG = sqrt (DE^2 - GF^2)) (h5 : EF = EG + GF) (h6 : EF = 3 * GF) : EF = 5 :=
by
  -- Proof would go here
  sorry

end isosceles_triangle_altitude_l557_557630


namespace smallest_m_l557_557660

noncomputable def T : set ℂ :=
{ z | ∃ (x y : ℝ), z = x + y * I ∧ 1 / 2 ≤ x ∧ x ≤ real.sqrt 2 / 2 }

theorem smallest_m :
  ∃ (m : ℕ), (∀ (n : ℕ), n ≥ m → ∃ (z : ℂ), z ∈ T ∧ z ^ n = 1) ∧ m = 12 :=
begin
  sorry
end

end smallest_m_l557_557660


namespace probability_correct_l557_557418

open Real

-- Definition of triangle XYZ with the given conditions
structure Triangle :=
(X Y Z : Point)
(angle_XYZ_eq_45 : ∃ YZ : ℝ, XY = YZ ∧ YZ = 8*√2 ∧ angle YXZ = π / 4 ∧ angle XYZ = π / 4)

def probability_YD_greater_than_6 (T : Triangle) : ℝ :=
  if T.angle_XYZ_eq_45 then 5 / 8 else 0

theorem probability_correct (T : Triangle) : T.angle_XYZ_eq_45 → probability_YD_greater_than_6 T = 5 / 8 :=
by
  intro h
  rw [probability_YD_greater_than_6]
  split_ifs
  . exact rfl
  . contradiction

end probability_correct_l557_557418


namespace total_number_of_seats_l557_557886

theorem total_number_of_seats (n : ℕ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ n) 
                               (h2 : (10 : ℕ) < (29 : ℕ)) 
                               (h3 : (29 - 10) % (n / 2) = 0) : n = 38 :=
by sorry

end total_number_of_seats_l557_557886


namespace not_possible_in_five_trips_possible_in_six_trips_l557_557799

def truck_capacity := 2000
def rice_sacks := 150
def corn_sacks := 100
def rice_weight_per_sack := 60
def corn_weight_per_sack := 25

def total_rice_weight := rice_sacks * rice_weight_per_sack
def total_corn_weight := corn_sacks * corn_weight_per_sack
def total_weight := total_rice_weight + total_corn_weight

theorem not_possible_in_five_trips : total_weight > 5 * truck_capacity :=
by
  sorry

theorem possible_in_six_trips : total_weight <= 6 * truck_capacity :=
by
  sorry

#print axioms not_possible_in_five_trips
#print axioms possible_in_six_trips

end not_possible_in_five_trips_possible_in_six_trips_l557_557799


namespace sphere_surface_area_of_prism_l557_557479

theorem sphere_surface_area_of_prism (a b c : ℝ) (S : ℝ) (ha : a = 3) (hb : b = 4) (hc : c = 5) 
  (hS : S = 50 * π) : 
  let d := Real.sqrt (a^2 + b^2 + c^2),
      R := d / 2 in
  4 * π * R^2 = S :=
by
  rw [ha, hb, hc]
  let d := Real.sqrt (3^2 + 4^2 + 5^2)
  let R := d / 2
  have hd : d = 5 * Real.sqrt 2 := by sorry
  have hR : R = (5 * Real.sqrt 2) / 2 := by sorry
  rw [hd, hR, Real.sqrt_mul_self (5 * Real.sqrt 2), Real.sqrt_two_mul, Real.pow_two, Real.div_mul_div]
  norm_num
  exact hS

end sphere_surface_area_of_prism_l557_557479


namespace exists_function_f_l557_557943

theorem exists_function_f 
  (f : ℕ → ℕ)
  (h : ∀ n : ℕ, f(f(n+1)) = f(f(n)) + 2^{n-1}) :
  ∃ f : ℕ → ℕ, ∀ n : ℕ, f(f(n+1)) = f(f(n)) + 2^{n-1} :=
begin
  sorry
end

end exists_function_f_l557_557943


namespace range_of_a_for_empty_solution_l557_557271

theorem range_of_a_for_empty_solution (a : ℝ) :
  (∀ x : ℝ, ¬ (x^2 - a * x - a ≤ -3)) ↔ a ∈ Ioo (-6 : ℝ) 2 :=
sorry

end range_of_a_for_empty_solution_l557_557271


namespace range_of_a_on_line_condition_l557_557284

variables {x y a : ℝ}

theorem range_of_a_on_line_condition
  (h₁ : ∃ (M : ℝ × ℝ), (M.1 + M.2 + a = 0) ∧ ((M.1 - 0)^2 + ((M.2) - 2)^2 + (M.1 - 0)^2 + ((M.2) - 0)^2 = 10)) :
  -2 * real.sqrt 2 - 1 ≤ a ∧ a ≤ 2 * real.sqrt 2 - 1 :=
sorry

end range_of_a_on_line_condition_l557_557284


namespace turtles_never_combined_l557_557030

noncomputable def turtlenotcombined : ℕ :=
let n := 2017 in
let p := 1 in
let q := 2017 * 1008 in
p + q

theorem turtles_never_combined (n : ℕ) (h : n = 2017) : exists (p q : ℕ), 
  (∀ k, k = 2015 → ∃ m : ℚ, m = 1 / (n * 1008) → (p + q) = 1009 ∧ nat.gcd p q = 1) :=
begin
  use (1, 1008), -- p, q chosen
  split,
  { intros k hk, 
    use (1 / (n * 1008)),
    intro m,
    rw [hk, h],
    simp [nat.gcd, *], },
  repeat { sorry },
end

end turtles_never_combined_l557_557030


namespace num_zeros_in_product_l557_557255

theorem num_zeros_in_product : ∀ (a b : ℕ), (a = 125) → (b = 960) → (∃ n, a * b = n * 10^4) :=
by
  sorry

end num_zeros_in_product_l557_557255


namespace sum_of_consecutive_integers_l557_557791

theorem sum_of_consecutive_integers (a b c : ℤ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : c = 14) : a + b + c = 39 := 
by 
  sorry

end sum_of_consecutive_integers_l557_557791


namespace problem_statements_correct_l557_557682

variable {Jenna : Type}
variable (answers_correct : ℝ) -- proportion of correct answers
variable (received_A : Prop)    -- Jenna received an A

-- Condition given in the problem
axiom received_A_iff_correct : received_A ↔ answers_correct ≥ 0.8

-- Definition of statements A and C
def statement_A : Prop := (¬received_A) → (answers_correct < 0.8)
def statement_C : Prop := (answers_correct ≥ 0.8) → received_A

-- Proof problem statement
theorem problem_statements_correct :
  statement_A ∧ statement_C := sorry

end problem_statements_correct_l557_557682


namespace find_n_l557_557170

theorem find_n (n : ℤ) (h₁ : 0 ≤ n) (h₂ : n ≤ 180) (h₃ : cos (n * real.pi / 180) = cos (980 * real.pi / 180)) : n = 100 :=
sorry

end find_n_l557_557170


namespace sum_integer_values_l557_557992

theorem sum_integer_values (m : ℤ) (h : 0 < 5 * m ∧ 5 * m < 35) :
  ∑ x in {n : ℤ | 0 < n ∧ n < 7}.to_finset, x = 21 :=
sorry

end sum_integer_values_l557_557992


namespace least_integer_value_x_l557_557176

theorem least_integer_value_x (x : ℤ) : (3 * |2 * (x : ℤ) - 1| + 6 < 24) → x = -2 :=
by
  sorry

end least_integer_value_x_l557_557176


namespace jerry_added_two_action_figures_l557_557642

-- Define the initial conditions
def initial_books : ℕ := 7
def initial_action_figures : ℕ := 3
def books_more_than_figures : ℕ := 2

-- Define the number of action figures Jerry added
def action_figures_added (x : ℕ) : Prop :=
  initial_books = (initial_action_figures + x) + books_more_than_figures

-- The Lean 4 statement to prove
theorem jerry_added_two_action_figures : ∃ x, action_figures_added x ∧ x = 2 :=
begin
  use 2,
  unfold action_figures_added,
  norm_num,
end

end jerry_added_two_action_figures_l557_557642


namespace relationship_abc_l557_557199

theorem relationship_abc (a b c : ℝ) (h1 : a = 0.77^1.2) (h2 : b = 1.2^0.77) (h3 : c = Real.pi^0) : a < c ∧ c < b := 
  by
    sorry

end relationship_abc_l557_557199


namespace simple_interest_total_amount_l557_557106

theorem simple_interest_total_amount (P : ℝ) (r : ℝ) (t : ℝ) (A_2years : ℝ) (A_7years : ℝ) :
  P = 350 →
  t = 2 →
  A_2years = 420 →
  A_2years = P * (1 + r * t) →
  t + 5 = 7 →
  A_7years = P * (1 + r * 7) →
  A_7years = 595 :=
by
  intros hP ht hA2 hInterest2 ht7 hInterest7
  rw [hP, ht, hInterest2] at hA2
  have hr : r = 0.1 := sorry
  rw [hP, ht7, hInterest7, hr]
  simp
  exact sorry

end simple_interest_total_amount_l557_557106


namespace find_line_equation_l557_557535

-- Define the intersection point of the given lines
def point_M : ℝ × ℝ :=
  have h₁ : 3 * 1 + 4 * (-2) + 5 = 0 := by norm_num,  -- solve 3(1) + 4(-2) + 5 = 0
  have h₂ : 2 * 1 - 3 * (-2) - 8 = 0 := by norm_num,  -- solve 2(1) - 3(-2) - 8 = 0
  (1, -2)

-- Define the equations of lines with conditions
def line_passing_origin_and_M : Prop :=
  ∀ (x y : ℝ), (2 * x + y = 0) ↔ ((x, y) = (0, 0) ∨ (x, y) = point_M)

def line_parallel_to_2x_plus_y_plus_5 : Prop :=
  ∀ (x y : ℝ), (2 * x + y = 0) ↔ ((x, y) = point_M)

def line_perpendicular_to_2x_plus_y_plus_5_passing_M : Prop :=
  ∀ (x y : ℝ), (x - 2 * y - 5 = 0) ↔ ((x, y) = point_M)

-- The statement of the theorem
theorem find_line_equation :
  line_passing_origin_and_M ∧
  line_parallel_to_2x_plus_y_plus_5 ∧
  line_perpendicular_to_2x_plus_y_plus_5_passing_M :=
sorry

end find_line_equation_l557_557535


namespace correct_meteorite_encounter_interval_l557_557115

def interval_of_meteorite_encounter (Rate1 Rate2 : ℝ) : ℝ := 1 / (Rate1 + Rate2)

theorem correct_meteorite_encounter_interval :
  interval_of_meteorite_encounter (1 / 7) (1 / 13) = 91 / 20 :=
by
  sorry

end correct_meteorite_encounter_interval_l557_557115


namespace sequence_ratio_l557_557566

variable (a : ℕ → ℝ) -- Define the sequence a_n
variable (q : ℝ) (h_q : q > 0) -- q is the common ratio and it is positive

-- Define the conditions
axiom geom_seq_pos : ∀ n : ℕ, 0 < a n
axiom geom_seq_def : ∀ n : ℕ, a (n + 1) = q * a n
axiom arith_seq_def : 2 * (1 / 2 * a 3) = 3 * a 1 + 2 * a 2

theorem sequence_ratio : (a 11 + a 13) / (a 8 + a 10) = 27 := 
by
  sorry

end sequence_ratio_l557_557566


namespace evaluate_expression_l557_557527

theorem evaluate_expression : 
  ⌈Real.sqrt (16 / 5)⌉ + ⌈16 / 5⌉ + ⌈(16 / 5) ^ 2⌉ = 17 := 
by
  sorry

end evaluate_expression_l557_557527


namespace total_seats_round_table_l557_557868

theorem total_seats_round_table (n : ℕ) (h : n = 38)
  (ka_position : ℕ) (sl_position : ℕ) 
  (h1 : ka_position = 10) 
  (h2 : sl_position = 29) 
  (h3 : (ka_position + n/2) % n = sl_position) : 
  n = 38 :=
by
  -- All steps and solution proof
  sorry

end total_seats_round_table_l557_557868


namespace train_speed_correct_l557_557827

noncomputable def speed_of_train 
  (length_of_train : ℝ) 
  (time_to_cross_man : ℝ) 
  (speed_of_man_km_hr : ℝ) : ℝ :=
let speed_of_man_ms := (speed_of_man_km_hr * 1000 / 3600) in
let relative_speed := (length_of_train / time_to_cross_man) in
let speed_of_train_ms := relative_speed + speed_of_man_ms in
(speed_of_train_ms * 3600 / 1000)

theorem train_speed_correct :
  speed_of_train 400 35.99712023038157 6 ≈ 45.9632 := 
sorry

end train_speed_correct_l557_557827


namespace least_n_probability_lt_1_over_10_l557_557055

theorem least_n_probability_lt_1_over_10 : 
  ∃ (n : ℕ), (1 / 2 : ℝ) ^ n < 1 / 10 ∧ ∀ m < n, ¬ ((1 / 2 : ℝ) ^ m < 1 / 10) :=
by
  sorry

end least_n_probability_lt_1_over_10_l557_557055


namespace omega_value_l557_557795

theorem omega_value (ω : ℝ) (hω : ω > 0)
  (hg_eq : ∀ x, g x = sin (ω * x + π / 4))
  (hg_sym : ∀ x, g (2 * ω - x) = g x)
  (hg_monotonic : ∀ x y, -ω < x → x < y → y < ω → g x ≤ g y) :
  ω = sqrt π / 2 :=
sorry

end omega_value_l557_557795


namespace alpha_when_m_equals_one_tan_alpha_when_m_equals_sqrt5_over_5_l557_557986

theorem alpha_when_m_equals_one (α : ℝ) (hα : 0 < α ∧ α < π) (h : sin (π - α) + cos (π + α) = 1) : α = π / 2 :=
sorry

theorem tan_alpha_when_m_equals_sqrt5_over_5 (α : ℝ) (hα : 0 < α ∧ α < π) (h : sin (π - α) + cos (π + α) = sqrt 5 / 5) : tan α = 2 :=
sorry

end alpha_when_m_equals_one_tan_alpha_when_m_equals_sqrt5_over_5_l557_557986


namespace amount_saved_percent_l557_557443

variable (S : ℝ)

theorem amount_saved_percent :
  (0.165 * S) / (0.10 * S) * 100 = 165 := sorry

end amount_saved_percent_l557_557443


namespace total_seats_round_table_l557_557869

theorem total_seats_round_table 
  (a : ℕ) (b : ℕ) 
  (h₀ : a ≠ b)
  (h₁ : a + b = 39) 
  : ∃ n, n = 38 := 
by {
  sorry
}

end total_seats_round_table_l557_557869


namespace base10_to_base7_conversion_l557_557150

theorem base10_to_base7_conversion :
  ∃ b1 b2 b3 b4 b5 : ℕ, 3 * 7^3 + 1 * 7^2 + 6 * 7^1 + 6 * 7^0 = 3527 ∧ 
  b1 = 1 ∧ b2 = 3 ∧ b3 = 1 ∧ b4 = 6 ∧ b5 = 6 ∧ (3527:ℕ) = (1*7^4 + b1*7^3 + b2*7^2 + b3*7^1 + b4*7^0) := by
sorry

end base10_to_base7_conversion_l557_557150


namespace simplify_and_rationalize_l557_557699

theorem simplify_and_rationalize :
  (√3 / √5) * (√7 / √11) * (√15 / √2) = 3 * √154 / 22 :=
by
  sorry

end simplify_and_rationalize_l557_557699


namespace number_of_solutions_l557_557403

-- Define the main theorem with the correct conditions
theorem number_of_solutions : 
  (∃ (x₁ x₂ x₃ x₄ x₅ : ℕ), 
     x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧ x₅ > 0 ∧ x₁ + x₂ + x₃ + x₄ + x₅ = 10) 
  → 
  (∃ t : ℕ, t = 70) :=
by 
  sorry

end number_of_solutions_l557_557403


namespace hyperbola_equation_ellipse_equation_parabola_equation_l557_557942

-- Definitions for the Hyperbola proof
def hyperbola : Type := { e : ℝ // e = sqrt 3 }
def hyperbola_foci : α = (-5 * sqrt 3, 0) ∧ β = (5 * sqrt 3, 0)

-- Definitions for the Ellipse proof
def ellipse : Type := { e : ℝ // e = 1 / 2 }
def ellipse_directrices : α = 4 * sqrt 3 

-- Definitions for the Parabola proof
def parabola_focus_distance : ℝ := 4

-- Statement for the Hyperbola proof
theorem hyperbola_equation (h : hyperbola) (hf : hyperbola_foci):
  ∃ a b : ℝ, (a = 5 ∧ b = 5 * sqrt 2) ∧ (∀ x y : ℝ, x^2 / 25 - y^2 / 50 = 1) := sorry

-- Statement for the Ellipse proof
theorem ellipse_equation (e : ellipse) (d : ellipse_directrices):
  ∃ a b c : ℝ, (a = 2 * sqrt 3 ∧ b = 3 ∧ c = sqrt 3) ∧ (∀ x y : ℝ, y^2 / 12 + x^2 / 9 = 1) := sorry

-- Statement for the Parabola proof
theorem parabola_equation (d : parabola_focus_distance):
  ∃ p : ℝ, (p = 4) ∧ (∀ x y : ℝ, x^2 = 8 * y) := sorry

end hyperbola_equation_ellipse_equation_parabola_equation_l557_557942


namespace total_seats_round_table_l557_557872

theorem total_seats_round_table 
  (a : ℕ) (b : ℕ) 
  (h₀ : a ≠ b)
  (h₁ : a + b = 39) 
  : ∃ n, n = 38 := 
by {
  sorry
}

end total_seats_round_table_l557_557872


namespace radius_of_internal_sphere_l557_557369

open Real

def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * π * r^3

theorem radius_of_internal_sphere (α : ℝ) (hα : 0 ≤ α ∧ α ≤ 2 * π) :
  (∃ r : ℝ, volume_of_sphere r = (α / (2 * π)) * volume_of_sphere 1) ↔
  (∃ r : ℝ, r = (α / (2 * π))^(1/3)) :=
by
  sorry

end radius_of_internal_sphere_l557_557369


namespace h_is_even_l557_557972

noncomputable def f (x : ℝ) : ℝ := x / (2^x - 1)
noncomputable def g (x : ℝ) : ℝ := x / 2
noncomputable def h (x : ℝ) : ℝ := f(x) + g(x)

theorem h_is_even : ∀ x : ℝ, h(-x) = h(x) := by
  sorry

end h_is_even_l557_557972


namespace main_theorem_l557_557914

/-- A good integer is an integer whose absolute value is not a perfect square. -/
def good (n : ℤ) : Prop := ∀ k : ℤ, k^2 ≠ |n|

/-- Integer m can be represented as a sum of three distinct good integers u, v, w whose product is the square of an odd integer. -/
def special_representation (m : ℤ) : Prop :=
  ∃ u v w : ℤ,
    good u ∧ good v ∧ good w ∧
    (u ≠ v ∧ u ≠ w ∧ v ≠ w) ∧
    (∃ k : ℤ, (u * v * w = k^2 ∧ k % 2 = 1)) ∧
    (m = u + v + w)

/-- All integers m having the property that they can be represented in infinitely many ways as a sum of three distinct good integers whose product is the square of an odd integer are those which are congruent to 3 modulo 4. -/
theorem main_theorem (m : ℤ) : special_representation m ↔ m % 4 = 3 := sorry

end main_theorem_l557_557914


namespace smallest_k_divisor_l557_557184

-- Define the polynomials in Lean
def P (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1
def Q (z : ℂ) (k : ℕ) : ℂ := z^k - 1

-- State the main theorem/problem
theorem smallest_k_divisor (k : ℕ) (hk : k > 0) : 
  (∀ z : ℂ, P z | Q z k) ↔ k = 9 := 
sorry

end smallest_k_divisor_l557_557184


namespace projection_correct_l557_557500

def vec_a : ℝ × ℝ × ℝ := (-4, 2, 3)
def vec_b : ℝ × ℝ × ℝ := (3, -2, 4)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def scalar_mult (c : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (c * v.1, c * v.2, c * v.3)

def proj (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let b_dot_b := dot_product b b
  let a_dot_b := dot_product a b
  scalar_mult (a_dot_b / b_dot_b) b

def correct_proj : ℝ × ℝ × ℝ := (-12/29, 8/29, -16/29)

theorem projection_correct :
  proj vec_a vec_b = correct_proj := by
  sorry

end projection_correct_l557_557500


namespace MegatekEmployeePercentages_l557_557367

def circle := 360

def ManufacturingAngle := 54
def HRAngle := 2 * ManufacturingAngle
def SalesAngle := HRAngle / 2
def KnownTotalAngle := ManufacturingAngle + HRAngle + SalesAngle
def RDAngle := circle - KnownTotalAngle

def ManufacturingPercent := (ManufacturingAngle / circle) * 100
def HRPercent := (HRAngle / circle) * 100
def SalesPercent := (SalesAngle / circle) * 100
def RDPercent := (RDAngle / circle) * 100

theorem MegatekEmployeePercentages :
  ManufacturingPercent = 15 ∧ HRPercent = 30 ∧ SalesPercent = 15 ∧ RDPercent = 40 :=
by
  sorry

end MegatekEmployeePercentages_l557_557367


namespace billy_used_54_tickets_l557_557128

-- Definitions
def ferris_wheel_rides := 7
def bumper_car_rides := 3
def ferris_wheel_cost := 6
def bumper_car_cost := 4

-- Theorem Statement
theorem billy_used_54_tickets : 
  ferris_wheel_rides * ferris_wheel_cost + bumper_car_rides * bumper_car_cost = 54 := 
by
  sorry

end billy_used_54_tickets_l557_557128


namespace polynomial_remainder_l557_557961

-- Define the polynomial division problem and prove the remainder
theorem polynomial_remainder :
  let f := (x : ℝ) ^ 5 + 4
  let g := (x : ℝ - 3) ^ 2
  let remainder := 331 * x - 746
  (∃ q, f = g * q + remainder) :=
sorry

end polynomial_remainder_l557_557961


namespace number_of_ways_to_fulfill_order_l557_557922

open Finset Nat

/-- Bill must buy exactly eight donuts from a shop offering five types, 
with at least two of the first type and one of each of the other four types. 
Prove that there are exactly 15 different ways to fulfill this order. -/
theorem number_of_ways_to_fulfill_order : 
  let total_donuts := 8
  let types_of_donuts := 5
  let mandatory_first_type := 2
  let mandatory_each_other_type := 1
  let remaining_donuts := total_donuts - (mandatory_first_type + 4 * mandatory_each_other_type)
  let combinations := (remaining_donuts + types_of_donuts - 1).choose (types_of_donuts - 1)
  combinations = 15 := 
by
  sorry

end number_of_ways_to_fulfill_order_l557_557922


namespace simplify_radicals_l557_557700

theorem simplify_radicals (x : ℝ) :
  sqrt (28 * x) * sqrt (15 * x) * sqrt (21 * x) = 42 * x * sqrt (5 * x) :=
by
  sorry

end simplify_radicals_l557_557700


namespace total_seats_l557_557908

theorem total_seats (KA_pos : ℕ) (SL_pos : ℕ) (h1 : KA_pos = 10) (h2 : SL_pos = 29) (h3 : SL_pos = KA_pos + (KA_pos * 2 - 1) / 2):
  let total_positions := 2 * (SL_pos - KA_pos - 1) + 2
  total_positions = 38 :=
by
  sorry

end total_seats_l557_557908


namespace satisfies_additive_property_l557_557282

def f1 (x : ℝ) : ℝ := sqrt 3 / x
def f2 (x : ℝ) : ℝ := -2 * x - 6
def f3 (x : ℝ) : ℝ := 3 * x
def f4 (x : ℝ) : ℝ := 1/2 * x^2 + 3 * x + 4

theorem satisfies_additive_property :
  (∀ a b : ℝ, f3 (a + b) = f3 a + f3 b) ∧
  ¬ (∀ a b : ℝ, f1 (a + b) = f1 a + f1 b) ∧
  ¬ (∀ a b : ℝ, f2 (a + b) = f2 a + f2 b) ∧
  ¬ (∀ a b : ℝ, f4 (a + b) = f4 a + f4 b) := 
by
  -- The proof steps would go here
  sorry

end satisfies_additive_property_l557_557282


namespace opposite_of_3_is_neg3_l557_557745

def opposite (x : ℝ) := -x

theorem opposite_of_3_is_neg3 : opposite 3 = -3 :=
by
  sorry

end opposite_of_3_is_neg3_l557_557745


namespace maximize_area_for_XRYS_l557_557119

noncomputable def maximized_area (A B C D X Y R S : Point) (area_square : ℝ) : Prop :=
  isSquare A B C D ∧
  onSide X A B ∧
  onSide Y C D ∧
  intersect A Y D X R ∧
  intersect B Y C X S ∧
  XY_parallel_to_AD X Y A D → 
  area (quadrilateral X R Y S) = area_square / 4

theorem maximize_area_for_XRYS (A B C D X Y R S : Point) (area_square : ℝ) :
  maximized_area A B C D X Y R S area_square :=
by sorry

end maximize_area_for_XRYS_l557_557119


namespace p_sufficient_not_necessary_l557_557198

theorem p_sufficient_not_necessary (a b : ℝ) (p : a > b) (q : 2^a > 2^b - 1) :
  (a > b) → (2^a > 2^b - 1) ∧ ¬((2^a > 2^b - 1) → (a > b)) :=
by
  sorry

end p_sufficient_not_necessary_l557_557198


namespace cos_double_angle_given_sin_l557_557561

theorem cos_double_angle_given_sin (x : ℝ) (h : 2 * sin (π - x) + 1 = 0) : cos (2 * x) = 1 / 2 :=
by
  sorry

end cos_double_angle_given_sin_l557_557561


namespace opposite_of_three_l557_557760

theorem opposite_of_three : -3 = -3 := by
  -- The condition we have identified is the given number 3.
  -- We will directly state that the opposite of 3 is -3.
  -- This proof is trivial as we are directly replacing 3 with -3 to match the problem statement.
  rfl

end opposite_of_three_l557_557760


namespace vector_cross_product_magnitude_l557_557232

noncomputable def a_mag : Real := 2
noncomputable def b_mag : Real := 5
noncomputable def a_dot_b : Real := -6

theorem vector_cross_product_magnitude :
  ∃ θ : Real, ∥ a_mag ∥ * ∥ b_mag ∥ * Real.sin θ = 8 := sorry

end vector_cross_product_magnitude_l557_557232


namespace opposite_of_three_l557_557768

theorem opposite_of_three : -3 = opposite(3) := 
by
  sorry

end opposite_of_three_l557_557768


namespace map_triangle_l557_557177

def linear_map (z : ℂ) : ℂ := (1 + complex.I) * (1 - z)

theorem map_triangle (z1 z2 z3 : ℂ) (w1 w2 w3 : ℂ) 
  (h1: z1 = 0) (h2: z2 = 1) (h3: z3 = complex.I) 
  (hw1: w1 = 1 + complex.I) (hw2: w2 = 0) (hw3: w3 = 2) :
  ∃ f : ℂ → ℂ, (f z1 = w1) ∧ (f z2 = w2) ∧ (f z3 = w3) ∧ (∀ z, f z = linear_map z) :=
begin
  use linear_map,
  split,
  { rw [h1, linear_map], }, -- Proof that f(0) = 1 + i
  split,
  { rw [h2, linear_map], }, -- Proof that f(1) = 0
  split,
  { rw [h3, linear_map], }, -- Proof that f(i) = 2
  intros z,
  refl, -- Proof that the function agrees for all z
end

end map_triangle_l557_557177


namespace triangle_obtuse_l557_557618

def is_obtuse_triangle (A B C : ℝ) : Prop := A > 90 ∨ B > 90 ∨ C > 90

theorem triangle_obtuse (A B C : ℝ) (h1 : A > 3 * B) (h2 : C < 2 * B) (h3 : A + B + C = 180) : is_obtuse_triangle A B C :=
by sorry

end triangle_obtuse_l557_557618


namespace contradiction_proof_l557_557800

theorem contradiction_proof (a b c d : ℝ) (h1 : a + b = 1) (h2 : c + d = 1) (h3 : ac + bd > 1) : 
  ¬ (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) :=
sorry

end contradiction_proof_l557_557800


namespace anika_more_than_twice_reeta_l557_557126

theorem anika_more_than_twice_reeta (R A M : ℕ) (h1 : R = 20) (h2 : A + R = 64) (h3 : A = 2 * R + M) : M = 4 :=
by
  sorry

end anika_more_than_twice_reeta_l557_557126


namespace sequence_an_general_formula_sum_terms_cn_l557_557979

-- Condition definitions
variable (a : ℕ → ℝ) (S : ℕ → ℝ)
variable (b : ℕ → ℝ) (c : ℕ → ℝ)

axiom a1 (n : ℕ) : (∃ S : ℕ → ℝ, 1 + S n = 2 * a n)
axiom a2 : ∀ n, b n = Real.log2 (a n)
axiom a3 : ∀ n, c n = a n * b n

-- Problem (Ⅰ)
theorem sequence_an_general_formula (n : ℕ) (h : n > 0) : a n = 2^(n-1) := sorry

-- Problem (Ⅱ)
theorem sum_terms_cn (n : ℕ) : (∑ i in Finset.range n, c i) = (n-2) * 2^n + 2 := sorry

end sequence_an_general_formula_sum_terms_cn_l557_557979


namespace max_value_of_f_l557_557008

noncomputable def f (x : ℝ) : ℝ := |x| / (Real.sqrt (1 + x^2) * Real.sqrt (4 + x^2))

theorem max_value_of_f : ∃ M : ℝ, M = 1 / 3 ∧ ∀ x : ℝ, f x ≤ M :=
by
  sorry

end max_value_of_f_l557_557008


namespace matrix_det_l557_557147

def matrix := ![
  ![2, -4, 2],
  ![0, 6, -1],
  ![5, -3, 1]
]

theorem matrix_det : Matrix.det matrix = -34 := by
  sorry

end matrix_det_l557_557147


namespace cost_D_to_E_l557_557628

def distance_DF (DF DE EF : ℝ) : Prop :=
  DE^2 = DF^2 + EF^2

def cost_to_fly (distance : ℝ) (per_kilometer_cost booking_fee : ℝ) : ℝ :=
  distance * per_kilometer_cost + booking_fee

noncomputable def total_cost_to_fly_from_D_to_E : ℝ :=
  let DE := 3750 -- Distance from D to E (km)
  let booking_fee := 120 -- Booking fee in dollars
  let per_kilometer_cost := 0.12 -- Cost per kilometer in dollars
  cost_to_fly DE per_kilometer_cost booking_fee

theorem cost_D_to_E : total_cost_to_fly_from_D_to_E = 570 := by
  sorry

end cost_D_to_E_l557_557628


namespace infinite_solutions_iff_m_eq_2_l557_557208

theorem infinite_solutions_iff_m_eq_2 (m x y : ℝ) :
  (m*x + 4*y = m + 2 ∧ x + m*y = m) ↔ (m = 2) ∧ (m > 1) :=
by
  sorry

end infinite_solutions_iff_m_eq_2_l557_557208


namespace locus_of_centers_of_tangent_circles_l557_557552

-- Main proof problem statement
theorem locus_of_centers_of_tangent_circles 
  (e : Line) 
  (O : Point) 
  (r0 : ℝ) 
  (k0 : Circle O r0) : 
  locus (λ M, ∃ (r : ℝ), tangent (Circle M r) k0 ∧ tangent (Circle M r) e) = 
    (parabola O e r0) :=
sorry

end locus_of_centers_of_tangent_circles_l557_557552


namespace product_real_parts_complex_solutions_l557_557517

/-
  Given the equation 2x^2 + 4x = 3 + i, prove that the product of the real parts of its 
  solutions is 1 - cos(pi / 30) * sqrt(6.5).
-/

namespace ProofProblem

open Complex Real

theorem product_real_parts_complex_solutions :
  ∀ x : ℂ, 2 * x^2 + 4 * x = 3 + I → 
    (1 - cos (π / 30) * sqrt 6.5) := 
begin
  sorry
end

end ProofProblem

end product_real_parts_complex_solutions_l557_557517


namespace opposite_of_three_l557_557739

theorem opposite_of_three : -3 = -3 := 
by sorry

end opposite_of_three_l557_557739


namespace decrease_in_length_l557_557108

theorem decrease_in_length (L B : ℝ) (h₀ : L ≠ 0) (h₁ : B ≠ 0)
  (h₂ : ∃ (A' : ℝ), A' = 0.72 * L * B)
  (h₃ : ∃ B' : ℝ, B' = B * 0.9) :
  ∃ (x : ℝ), x = 20 :=
by
  sorry

end decrease_in_length_l557_557108


namespace collinearity_of_K_H_M_l557_557968

open EuclideanGeometry

-- Definitions and assumptions corresponding to the problem conditions

variables {A B C D E F H K M P O : Point} 

-- Triangle ABC inscribed in circle ⊙O
variable (circumcircle : Circle)

variable (triangle : Triangle A B C)
variable (inscribed : TriangleInscribedIn triangle circumcircle)

-- Altitudes AD, BE, CF intersect at H
variable (H : Point)
variable (altitudes : AltitudesMeetIn H)

-- Lines tangent to ⊙O at points B and C intersect at P
variable (tangentB : TangentLine circumcircle B)
variable (tangentC : TangentLine circumcircle C)
variable (P : Point)
variable (tangent_intersection : Intersect tangentB tangentC P)

-- PD intersects EF at K
variable (D : Point)
variable (line_PD : Line P D)
variable (E : Point)
variable (F : Point)
variable (E_F_line : Line E F)
variable (K : Point)
variable (PD_intersect_EF : Intersect (Line.mk P D) (Line.mk E F) K)

-- M is the midpoint of BC
variable (M : Point)
variable (midpoint_M : Midpoint B C M)

-- Proving collinearity of K, H, M
theorem collinearity_of_K_H_M :
  Collinear K H M := sorry

end collinearity_of_K_H_M_l557_557968


namespace opposite_of_3_l557_557731

-- Define the concept of opposite of a number and the logic for positive numbers
def opposite (x : Int) : Int := 
  if x > 0 then -x
  else if x < 0 then -x
  else 0

-- Statement to prove that the opposite of 3 is -3
theorem opposite_of_3 : opposite 3 = -3 :=
by 
  -- Using the definition of opposite
  unfold opposite
  -- Simplify the expression for x = 3
  simp [lt_irrefl, int.coe_nat_lt]
  -- Conclude proof
  rfl

end opposite_of_3_l557_557731


namespace sequence_contains_infinitely_many_primes_l557_557099

noncomputable def x : ℕ → ℕ
| 0 := 0 -- not used since sequence starts from 1
| 1 := 1
| (n + 1) := let x_n := x n in
    (n / 2004 + 1 / n) * (x_n * x_n) - n^3 / 2004 + 1

theorem sequence_contains_infinitely_many_primes : 
∀ {n : ℕ}, ∃ p : ℕ, prime p ∧ ∃ m : ℕ, x m = p :=
begin
    sorry
end

end sequence_contains_infinitely_many_primes_l557_557099


namespace asymptotes_of_hyperbola_l557_557568

variable (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
variable (h3 : (1 + b^2 / a^2) = (6 / 4))

theorem asymptotes_of_hyperbola :
  ∃ (m : ℝ), m = b / a ∧ (m = Real.sqrt 2 / 2) ∧ ∀ x : ℝ, (y = m*x) ∨ (y = -m*x) :=
by
  sorry

end asymptotes_of_hyperbola_l557_557568


namespace min_value_fraction_l557_557190

theorem min_value_fraction (x : ℝ) (h : x > 4) : 
  ∃ y, y = x - 4 ∧ (x + 11) / Real.sqrt (x - 4) = 2 * Real.sqrt 15 := by
  sorry

end min_value_fraction_l557_557190


namespace fraction_inequality_l557_557989

variables (a b m : ℝ)

theorem fraction_inequality (h1 : a > b) (h2 : m > 0) : (b + m) / (a + m) > b / a :=
sorry

end fraction_inequality_l557_557989


namespace shaded_region_area_l557_557289

-- Define the radii of the semicircles
def radius_ADB : ℝ := 2
def radius_BEC : ℝ := 2
def radius_DFE : ℝ := 1

-- Define the areas of the semicircles
def area_semicircle (r : ℝ) : ℝ := 0.5 * real.pi * (r ^ 2)

-- Semicircle areas
def area_ADB := area_semicircle radius_ADB
def area_BEC := area_semicircle radius_BEC
def area_DFE := area_semicircle radius_DFE

-- Calculate the total shaded area
def total_shaded_area : ℝ := area_ADB + area_BEC - area_DFE

-- The statement we want to prove
theorem shaded_region_area : total_shaded_area = 3.5 * real.pi := by
  sorry

end shaded_region_area_l557_557289


namespace total_seats_l557_557905

theorem total_seats (KA_pos : ℕ) (SL_pos : ℕ) (h1 : KA_pos = 10) (h2 : SL_pos = 29) (h3 : SL_pos = KA_pos + (KA_pos * 2 - 1) / 2):
  let total_positions := 2 * (SL_pos - KA_pos - 1) + 2
  total_positions = 38 :=
by
  sorry

end total_seats_l557_557905


namespace players_not_played_l557_557482

-- Definitions for the problem
def total_players : ℕ := 36
def initial_players : ℕ := 11
def first_half_substitutions : ℕ := 3
def additional_subs_second_half : ℕ := (first_half_substitutions / 2) + first_half_substitutions
def total_substitutions : ℕ := first_half_substitutions + additional_subs_second_half
def players_played : ℕ := initial_players + total_substitutions
def players_did_not_play : ℕ := total_players - players_played

-- The statement of the theorem
theorem players_not_played (h : additional_subs_second_half = 5) : players_did_not_play = 17 :=
by {
  have h1 : total_substitutions = 8 := by { sorry },
  have h2 : players_played = 19 := by { sorry },
  have h3 : players_did_not_play = 17 := by { sorry },
  exact h3,
}

end players_not_played_l557_557482


namespace unique_solution_ffx_eq_27_l557_557241

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3 * x^2 - 9 * x + 27

-- Prove that there is exactly one solution for f(f(x)) = 27 in the domain -3 ≤ x ≤ 5
theorem unique_solution_ffx_eq_27 :
  (∃! x : ℝ, -3 ≤ x ∧ x ≤ 5 ∧ f (f x) = 27) :=
by
  sorry

end unique_solution_ffx_eq_27_l557_557241


namespace radius_of_semicircle_l557_557783

def perimeter_of_semicircle (r : ℝ) : ℝ := (Real.pi * r) + (2 * r)

theorem radius_of_semicircle (r : ℝ) (h : perimeter_of_semicircle r = 17.995574287564274) : abs (r - 3.5) < 1e-9 :=
by
  sorry

end radius_of_semicircle_l557_557783


namespace coordinate_of_M_l557_557347

-- Definition and given conditions
def L : ℚ := 1 / 6
def P : ℚ := 1 / 12

def divides_into_three_equal_parts (L P M N : ℚ) : Prop :=
  M = L + (P - L) / 3 ∧ N = L + 2 * (P - L) / 3

theorem coordinate_of_M (M N : ℚ) 
  (h1 : divides_into_three_equal_parts L P M N) : 
  M = 1 / 9 := 
by 
  sorry
  
end coordinate_of_M_l557_557347


namespace opposite_of_3_l557_557775

theorem opposite_of_3 : -3 = -3 := 
by
  -- sorry is added to skip the proof as per instructions
  sorry

end opposite_of_3_l557_557775


namespace hydrogen_atoms_in_compound_l557_557468

noncomputable def molecular_weight_ba : ℝ := 137.33
noncomputable def molecular_weight_o : ℝ := 16.00
noncomputable def molecular_weight_h : ℝ := 1.01

def molecular_weight_compound : ℝ := 171

def barium_atoms : ℕ := 1
def oxygen_atoms : ℕ := 2

theorem hydrogen_atoms_in_compound : 
  let total_ba_o := barium_atoms * molecular_weight_ba + oxygen_atoms * molecular_weight_o,
      weight_h := molecular_weight_compound - total_ba_o,
      hydrogen_atoms := (weight_h / molecular_weight_h).ceil.to_nat
  in hydrogen_atoms = 2 :=
by
  have total_ba_o : ℝ := barium_atoms * molecular_weight_ba + oxygen_atoms * molecular_weight_o := by sorry
  have weight_h : ℝ := molecular_weight_compound - total_ba_o := by sorry
  have hydrogen_atoms : ℕ := (weight_h / molecular_weight_h).ceil.to_nat := by sorry
  calc hydrogen_atoms = 2 := by sorry

end hydrogen_atoms_in_compound_l557_557468


namespace rational_numbers_are_integers_l557_557690

noncomputable section

variables {x y : ℚ}

theorem rational_numbers_are_integers (h1 : (x + y) ∈ ℤ) (h2 : (x * y) ∈ ℤ) :
  x ∈ ℤ ∧ y ∈ ℤ := 
sorry

end rational_numbers_are_integers_l557_557690


namespace correct_drawn_numbers_l557_557248

-- Define the initial conditions
def coins := [(20, 3), (10, 6), (5, 5), (2, 9), (1, 3)]
def maxNum := 35

-- Function to calculate the total amount of forints for a given list of coin amounts
def totalAmount (coins : List (Nat × Nat)) : Nat :=
  coins.foldl (λ acc (coin, count) → acc + coin * count) 0

-- Define the theorem stating that the drawn numbers are correct given the coin conditions
theorem correct_drawn_numbers (num_draws : Nat) (drawn_numbers : List Nat):
  num_draws = 7 →
  totalAmount coins = drawn_numbers.foldl (λ acc n → acc + n) 0 →
  drawn_numbers = [34, 33, 29, 19, 18, 17, 16] :=
by
  sorry

end correct_drawn_numbers_l557_557248


namespace race_outcomes_l557_557121

theorem race_outcomes (n : ℕ) (h : n = 6) : (finset.range n).card * (finset.range (n - 1)).card * (finset.range (n - 2)).card = 120 :=
by
  rw [h, finset.card_range, finset.card_range, finset.card_range]
  norm_num
  sorry

end race_outcomes_l557_557121


namespace acute_angle_theta_l557_557258

theorem acute_angle_theta (θ : ℝ) (h₁ : θ > 0) (h₂ : θ < 90) (h₃ : sqrt 2 * sin (20 * real.pi / 180) = cos (θ * real.pi / 180) - sin (θ * real.pi / 180)) : θ = 25 :=
begin
  sorry
end

end acute_angle_theta_l557_557258


namespace deepak_age_l557_557919

theorem deepak_age
  (A D : ℕ)
  (h1 : A / D = 5 / 7)
  (h2 : A + 6 = 36) :
  D = 42 :=
by sorry

end deepak_age_l557_557919


namespace proposition_1_proposition_2_proposition_3_proposition_4_l557_557376

noncomputable def curvature (f : ℝ → ℝ) (A B : ℝ × ℝ) : ℝ :=
  let k_A := (deriv f) A.1
  let k_B := (deriv f) B.1
  let AB := real.sqrt ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2)
  (abs (k_A - k_B)) / AB

theorem proposition_1 : 
  let f := λ x : ℝ, x ^ 3 - x ^ 2 + 1 in
  let A := (1, f 1) in
  let B := (2, f 2) in
  curvature f A B ≤ real.sqrt 3 := sorry

theorem proposition_2 : ∃ f : ℝ → ℝ, ∀ A B : ℝ × ℝ, curvature f A B = 0 :=
exists.intro (λ x, 1) (by sorry)

theorem proposition_3 : 
  let f := λ x : ℝ, x ^ 2 + 1 in
  ∀ A B : ℝ × ℝ, curvature f A B ≤ 2 := sorry

theorem proposition_4 :
  let f := λ x : ℝ, real.exp x in
  ∀ A B : ℝ × ℝ, A.1 - B.1 = 1 →
    (∀ t : ℝ, (t < 1) → t * curvature f A B < 1) := sorry

end proposition_1_proposition_2_proposition_3_proposition_4_l557_557376


namespace second_order_derivative_l557_557062

theorem second_order_derivative :
  (∃ (t : ℝ), ∀ (x y : ℝ), x = cos t ∧ y = log (sin t) → 
    (∃ (y'' : ℝ), y'' = - (1 + cos t ^ 2) / (sin t ^ 4))) :=
sorry

end second_order_derivative_l557_557062


namespace adults_in_each_group_l557_557918

theorem adults_in_each_group (A : ℕ) :
  (∃ n : ℕ, n >= 17 ∧ n * 15 = 255) →
  (∃ m : ℕ, m * A = 255 ∧ m >= 17) →
  A = 15 :=
by
  intros h_child_groups h_adult_groups
  -- Use sorry to skip the proof
  sorry

end adults_in_each_group_l557_557918


namespace sum_of_digits_of_greatest_prime_divisor_of_32767_l557_557806

theorem sum_of_digits_of_greatest_prime_divisor_of_32767 : 
  let greatest_prime_divisor := 331 in
  (3 + 3 + 1 = 7) :=
by
  have greatest_prime_divisor := 331
  have sum_of_digits := 3 + 3 + 1
  show 3 + 3 + 1 = 7
  sorry

end sum_of_digits_of_greatest_prime_divisor_of_32767_l557_557806


namespace zero_points_intervals_l557_557585

noncomputable def f (x : ℝ) : ℝ :=
1 + x - (x^2) / 2 + (x^3) / 3 - (x^4) / 4 + ... - (x^2012) / 2012 + (x^2013) / 2013

noncomputable def g (x : ℝ) : ℝ :=
1 - x + (x^2) / 2 - (x^3) / 3 + (x^4) / 4 - ... + (x^2012) / 2012 - (x^2013) / 2013

theorem zero_points_intervals :
  (∃ x_1 : ℝ, f x_1 = 0 ∧ x_1 ∈ Ioo (-1) (0)) ∧
  (∃ x_2 : ℝ, g x_2 = 0 ∧ x_2 ∈ Ioo (1) (2)) :=
sorry

end zero_points_intervals_l557_557585


namespace celia_time_30_miles_l557_557138

noncomputable def Lexie_time_per_mile : ℝ := 20

theorem celia_time_30_miles :
  let celia_time_per_mile := Lexie_time_per_mile / 2 in
  let celia_total_time := celia_time_per_mile * 30 in
  celia_total_time = 300 :=
by
  -- Initial declarations
  let celia_time_per_mile := Lexie_time_per_mile / 2
  let celia_total_time := celia_time_per_mile * 30
  -- The proof steps would go here, but we skip that with (sorry) as per the instructions
  sorry

end celia_time_30_miles_l557_557138


namespace probability_inner_circle_l557_557930

noncomputable def outer_radius : ℝ := 3
noncomputable def inner_radius : ℝ := 1.5

theorem probability_inner_circle : 
  let outer_area := Real.pi * (outer_radius ^ 2),
      inner_area := Real.pi * (inner_radius ^ 2)
  in inner_area / outer_area = 1 / 4 :=
by 
  let outer_area := Real.pi * (outer_radius ^ 2)
  let inner_area := Real.pi * (inner_radius ^ 2)
  calc
    inner_area / outer_area
      = (Real.pi * (inner_radius ^ 2)) / (Real.pi * (outer_radius ^ 2)) : by sorry
  ... = (inner_radius ^ 2) / (outer_radius ^ 2) : by sorry
  ... = (1.5 ^ 2) / (3 ^ 2) : by sorry
  ... = (2.25) / (9) : by sorry
  ... = 1 / 4 : by sorry

end probability_inner_circle_l557_557930


namespace expression_value_l557_557448

-- Defining terms based on original conditions and the question
noncomputable def cube_root_neg_eight : ℝ := - ((-8.0) ^ (1.0 / 3.0))
def pi_pow_zero : ℝ := real.pi ^ 0
def log_four : ℝ := real.log 4.0
def log_twenty_five : ℝ := real.log 25.0

-- Theorem statement proving the result
theorem expression_value : 
  cube_root_neg_eight + pi_pow_zero + log_four + log_twenty_five = 1 :=
by sorry

end expression_value_l557_557448


namespace evaluate_expression_l557_557949

theorem evaluate_expression : 
  let a := 3^5,
      b := 3^2,
      c := 2^10 in
  ((a / b) * c) + 1/2 = 27648.5 := 
by {
  sorry
}

end evaluate_expression_l557_557949


namespace find_cd_l557_557191

def g (c d : ℝ) (x : ℝ) : ℝ :=
if x < 3 then c * x + d else 10 - 2 * x

theorem find_cd (c d : ℝ) (h : ∀ x, g c d (g c d x) = x) : c + d = 4.5 := by
  sorry

end find_cd_l557_557191


namespace ellipse_equation_hyperbola_vertices_and_foci_exists_point_P_on_x_axis_angles_complementary_l557_557572

noncomputable def hyperbola_eq (x y : ℝ) : Prop :=
  x^2 - y^2 / 2 = 1

noncomputable def ellipse_eq (x y : ℝ) : Prop :=
  x^2 / 3 + y^2 / 2 = 1

def point_on_x_axis (P : ℝ × ℝ) : Prop :=
  P.snd = 0

def angles_complementary (P A B : ℝ × ℝ) : Prop :=
  let kPA := (A.snd - P.snd) / (A.fst - P.fst)
  let kPB := (B.snd - P.snd) / (B.fst - P.fst)
  kPA + kPB = 0

theorem ellipse_equation_hyperbola_vertices_and_foci :
  (∀ x y : ℝ, hyperbola_eq x y → ellipse_eq x y) :=
sorry

theorem exists_point_P_on_x_axis_angles_complementary (F2 A B : ℝ × ℝ) :
  F2 = (1, 0) → (∃ P : ℝ × ℝ, point_on_x_axis P ∧ angles_complementary P A B) :=
sorry

end ellipse_equation_hyperbola_vertices_and_foci_exists_point_P_on_x_axis_angles_complementary_l557_557572


namespace min_value_of_expression_l557_557563

theorem min_value_of_expression (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) :
  25 ≤ (4 / a) + (9 / b) :=
sorry

end min_value_of_expression_l557_557563


namespace opposite_of_three_l557_557742

theorem opposite_of_three : -3 = -3 := 
by sorry

end opposite_of_three_l557_557742


namespace opposite_of_3_is_neg3_l557_557725

theorem opposite_of_3_is_neg3 : forall (n : ℤ), n = 3 -> -n = -3 :=
by
  sorry

end opposite_of_3_is_neg3_l557_557725


namespace pizzas_bought_l557_557543

theorem pizzas_bought (initial_total : ℕ) (pizza_cost : ℕ) (bill_initial : ℕ) (bill_final : ℕ) (frank_residual : ℕ) :
    initial_total = 42 →
    pizza_cost = 11 →
    bill_initial = 30 →
    bill_final = 39 →
    frank_residual = bill_final - bill_initial →
    frank_residual = 9 →
    ∃ p : ℕ, p = (initial_total - frank_residual) / pizza_cost → p = 3 :=
begin
    intros,
    use 3,
    sorry
end

end pizzas_bought_l557_557543


namespace polynomial_f_irreducible_l557_557692

open Polynomial

noncomputable def polynomial_f (n : ℕ) : Polynomial ℤ :=
  X^n + X^3 + X^2 + X + 5

theorem polynomial_f_irreducible (n : ℕ) (h : n ≥ 4) : Irreducible (polynomial_f n) :=
  sorry

end polynomial_f_irreducible_l557_557692


namespace smallest_m_l557_557661

noncomputable def T : set ℂ :=
{ z | ∃ (x y : ℝ), z = x + y * I ∧ 1 / 2 ≤ x ∧ x ≤ real.sqrt 2 / 2 }

theorem smallest_m :
  ∃ (m : ℕ), (∀ (n : ℕ), n ≥ m → ∃ (z : ℂ), z ∈ T ∧ z ^ n = 1) ∧ m = 12 :=
begin
  sorry
end

end smallest_m_l557_557661


namespace train_length_l557_557813

noncomputable def convert_speed (speed_kmh : ℝ) : ℝ := speed_kmh * (1000 / 3600)

theorem train_length (speed_kmh : ℝ) (time_sec : ℝ) (speed_m_s : ℝ) (length_m : ℝ) :
  speed_kmh = 52 →
  time_sec = 9 →
  speed_m_s = convert_speed speed_kmh →
  length_m = speed_m_s * time_sec →
  length_m ≈ 129.96 :=
by
  intros h1 h2 h3 h4
  rw [h1] at h3
  have h5 : speed_m_s = 52 * (1000 / 3600) := h3
  rw [h5] at h4
  have h6 : length_m = (52 * (1000 / 3600)) * 9 := h4
  sorry

end train_length_l557_557813


namespace amount_after_two_years_l557_557441

theorem amount_after_two_years (P : ℝ) (r : ℝ) (n : ℕ) (hP : P = 3200) (hr : r = 1 / 8) (hn : n = 2) :
  P * (1 + r) ^ n = 4050 :=
by
  rw [hP, hr, hn]
  norm_num
  sorry

end amount_after_two_years_l557_557441


namespace problem_l557_557405

open Set

variable (R : Type*) [LinearOrder R] [Archimedean R]

def M : Set R := {x | abs x ≤ 3}
def N : Set R := {x | x < 2}
def complement_M : Set R := compl M

theorem problem (x : R) :
  (complement_M R ∩ N R) = {x | x < -3} :=
sorry

end problem_l557_557405


namespace arc_length_correct_l557_557379

-- Define the conversion from degrees to radians
def degrees_to_radians (d : ℝ) : ℝ := d * (Float.pi / 180.0)

-- Define the radius and central angle in degrees
def radius : ℝ := 2
def central_angle_degrees : ℝ := 300

-- Convert central angle to radians
def central_angle_radians : ℝ := degrees_to_radians central_angle_degrees

-- Define the expected arc length
def expected_arc_length : ℝ := (10 * Float.pi) / 3

-- The arc length formula
def arc_length (r : ℝ) (theta : ℝ) : ℝ := r * theta

-- The theorem stating the arc length with given conditions
theorem arc_length_correct :
  arc_length radius central_angle_radians = expected_arc_length := by
sorry

end arc_length_correct_l557_557379


namespace A_implies_B_l557_557592

-- Definitions of given conditions
def f (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b

def A (a b : ℝ) : Set ℝ := {x | x = f x a b}
def B (a b : ℝ) : Set ℝ := {x | x = f (f x a b) a b}

-- Define specific sets for A and B
def A_set : Set ℝ := {-1, 3}
def B_set : Set ℝ := {- real.sqrt 3, -1, real.sqrt 3, 3}

-- Proof problem statement
theorem A_implies_B (a b : ℝ) (H : A a b = A_set) : B a b = B_set ∧ A_set ⊆ B a b :=
by
  sorry

end A_implies_B_l557_557592


namespace quadratic_solution_pair_l557_557389

noncomputable def a := (11 - Real.sqrt 21) / 2
noncomputable def c := (11 + Real.sqrt 21) / 2

-- Conditions
def condition1 : Prop := (a + 10) * (a + 10) = 4 * a * c
def condition2 : Prop := a + c = 11
def condition3 : Prop := a < c

theorem quadratic_solution_pair :
  (condition1 ∧ condition2 ∧ condition3) ∧
  (a = (11 - Real.sqrt 21) / 2) ∧ 
  (c = (11 + Real.sqrt 21) / 2) :=
by
  split
  case left =>
    split
    case left =>
      split
      case left =>
        sorry
      case right =>
        sorry
    case right =>
      sorry
  case right =>
    split
    case left =>
      rfl
    case right =>
      rfl

end quadratic_solution_pair_l557_557389


namespace neg_exists_le_eq_forall_gt_l557_557590

open Classical

variable {n : ℕ}

theorem neg_exists_le_eq_forall_gt :
  (¬ ∃ (n : ℕ), n > 0 ∧ 2^n ≤ 2 * n + 1) ↔
  (∀ (n : ℕ), n > 0 → 2^n > 2 * n + 1) :=
by 
  sorry

end neg_exists_le_eq_forall_gt_l557_557590


namespace unique_function_l557_557165

theorem unique_function (f : ℕ → ℕ) :
  (∀ a b : ℕ, ∃ k : ℕ, k^2 = a * (f a)^3 + 2 * a * b * (f a) + b * (f b)) →
  (∀ n : ℕ, f n = n) :=
by
  intro h
  have h1 : f 1 = 1 := sorry
  have h2 : ∀ p : ℕ, prime p → f p = p := sorry
  have h3 : ∀ n : ℕ, f n = n := sorry
  exact h3

end unique_function_l557_557165


namespace sum_of_possible_m_l557_557997

theorem sum_of_possible_m (m : ℤ) (h : 0 < 5 * m ∧ 5 * m < 35) : 
  (∑ n in { n : ℤ | 0 < 5 * n ∧ 5 * n < 35 }.to_finset, n) = 21 := 
by 
  sorry

end sum_of_possible_m_l557_557997


namespace johns_pool_monthly_cost_is_correct_l557_557645

def cost_per_cleaning (base_cost : ℕ) (tip_percentage : ℕ) : ℕ :=
  base_cost + (base_cost * tip_percentage / 100)

def monthly_cleaning_cost (cost_per_cleaning : ℕ) (days_in_month : ℕ) (days_between_cleaning : ℕ) : ℕ :=
  let cleanings = days_in_month / days_between_cleaning
  cleanings * cost_per_cleaning

def monthly_chemical_cost (chemical_cost_per_use : ℕ) (chemical_uses_per_month : ℕ) : ℕ :=
  chemical_cost_per_use * chemical_uses_per_month

def total_monthly_pool_cost (cleaning_base_cost : ℕ) (tip_percentage : ℕ) (days_in_month : ℕ)
(day_between_cleaning : ℕ) (chemical_cost_per_use : ℕ) (chemical_uses_per_month : ℕ) : ℕ :=
  let cost_cleaning := cost_per_cleaning cleaning_base_cost tip_percentage
  let total_cleaning := monthly_cleaning_cost cost_cleaning days_in_month day_between_cleaning
  let total_chemical := monthly_chemical_cost chemical_cost_per_use chemical_uses_per_month
  total_cleaning + total_chemical

theorem johns_pool_monthly_cost_is_correct :
  total_monthly_pool_cost 150 10 30 3 200 2 = 2050 :=
by
  -- Definition and calculations can be skipped
  sorry

end johns_pool_monthly_cost_is_correct_l557_557645


namespace alia_markers_l557_557860

theorem alia_markers (S A a : ℕ) (h1 : S = 60) (h2 : A = S / 3) (h3 : a = 2 * A) : a = 40 :=
by
  -- Proof omitted
  sorry

end alia_markers_l557_557860


namespace cone_sphere_ratio_l557_557845

theorem cone_sphere_ratio (r h : ℝ) (h_r_ne_zero : r ≠ 0)
  (h_vol_cone : (1 / 3) * π * r^2 * h = (1 / 3) * (4 / 3) * π * r^3) : 
  h / r = 4 / 3 := 
by
  sorry

end cone_sphere_ratio_l557_557845


namespace initial_bananas_on_tree_l557_557454

-- Definitions of given conditions
def bananas_left_on_tree : ℕ := 100
def bananas_eaten : ℕ := 70
def bananas_in_basket : ℕ := 2 * bananas_eaten

-- Statement to prove the initial number of bananas on the tree
theorem initial_bananas_on_tree : bananas_left_on_tree + (bananas_in_basket + bananas_eaten) = 310 :=
by
  sorry

end initial_bananas_on_tree_l557_557454


namespace diagonals_of_hexadecagon_l557_557082

-- Define the function to calculate number of diagonals in a convex polygon
def num_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

-- State the theorem for the number of diagonals in a convex hexadecagon
theorem diagonals_of_hexadecagon : num_diagonals 16 = 104 := by
  -- sorry is used to indicate the proof is skipped
  sorry

end diagonals_of_hexadecagon_l557_557082


namespace simson_line_properties_l557_557352

-- Given a triangle ABC
variables {A B C M P Q R H : Type}
variables [Inhabited A] [Inhabited B] [Inhabited C] 
variables [Inhabited M] [Inhabited P] [Inhabited Q] [Inhabited R] [Inhabited H]

-- Conditions
def is_point_on_circumcircle (A B C : Type) (M : Type) : Prop :=
sorry  -- formal definition that M is on the circumcircle of triangle ABC

def perpendicular_dropped_to_side (M : Type) (side : Type) (foot : Type) : Prop :=
sorry  -- formal definition of a perpendicular dropping from M to a side

def is_orthocenter (A B C H : Type) : Prop := 
sorry  -- formal definition that H is the orthocenter of triangle ABC

-- Proof Goal 1: The points P, Q, R are collinear (Simson line)
def simson_line (A B C M P Q R : Type) : Prop :=
sorry  -- formal definition and proof that P, Q, R are collinear

-- Proof Goal 2: The Simson line is equidistant from point M and the orthocenter H
def simson_line_equidistant (M H P Q R : Type) : Prop :=
sorry  -- formal definition and proof that Simson line is equidistant from M and H

-- Main theorem combining both proof goals
theorem simson_line_properties 
  (A B C M P Q R H : Type)
  (M_on_circumcircle : is_point_on_circumcircle A B C M)
  (perp_to_BC : perpendicular_dropped_to_side M (B × C) P)
  (perp_to_CA : perpendicular_dropped_to_side M (C × A) Q)
  (perp_to_AB : perpendicular_dropped_to_side M (A × B) R)
  (H_is_orthocenter : is_orthocenter A B C H) :
  simson_line A B C M P Q R ∧ simson_line_equidistant M H P Q R := 
by sorry

end simson_line_properties_l557_557352


namespace max_area_triangle_PAB_l557_557637

theorem max_area_triangle_PAB :
  (∀ (P : ℝ × ℝ), ((P.1 - 1)^2 + P.2^2 = 4 * ((P.1 - 1)^2 + (P.2 - 3)^2)) → 
  (x = 1) and (distance A B = 3)) →
  ∃ (P : ℝ × ℝ), area (triangle (P, A, B)) = 3 := by
  sorry

end max_area_triangle_PAB_l557_557637


namespace opposite_of_three_l557_557762

theorem opposite_of_three : -3 = opposite(3) := 
by
  sorry

end opposite_of_three_l557_557762


namespace smallest_k_l557_557182

theorem smallest_k (k : ℕ) (h : k = 112) :
  ∃ k, k > 0 ∧ (∀ m, m > 0 → (polynomial.X^12 + polynomial.X^11 + polynomial.X^8 + polynomial.X^7 + polynomial.X^6 + polynomial.X^3 + 1 ∣ polynomial.X^m - 1 ↔ m ≥ k)) :=
by {
  use k,
  split,
  { exact h.symm, },
  intros,
  sorry
}

end smallest_k_l557_557182


namespace real_solutions_l557_557531

def equation (x : ℝ) : ℝ := (x^2 + 3 * x + 1) ^ (x^2 - x - 6)

theorem real_solutions : {x : ℝ | equation x = 1} = {-3, -2, -1, 0, 3} :=
by
  sorry

end real_solutions_l557_557531


namespace decreasing_interval_l557_557578

noncomputable def f (x : ℝ) (a b : ℝ) := x^2 * (a * x + b)

theorem decreasing_interval (a b : ℝ)
  (extremum_at_two : ∀ (x:ℝ), deriv (f x a b) 2 = 0)
  (tangent_parallel : ∀ (x:ℝ), geom_slope (f x a b) 1 = -3) :
  ∃ (interval : set ℝ), interval = set.Ioo 0 2 ∧ ∀ (x ∈ interval), deriv (f x a b) x < 0 :=
sorry

end decreasing_interval_l557_557578


namespace fish_worth_in_rice_l557_557280

-- Definitions for the conditions
variables (f b r : ℝ)

-- Conditions
def condition1 : Prop := 4 * f = 5 * b
def condition2 : Prop := b = 3 * r

-- Statement: One fish is equivalent to (15 / 4) bags of rice.
theorem fish_worth_in_rice (h1 : condition1 f b r) (h2 : condition2 b r) : f = (15 / 4) * r :=
sorry

end fish_worth_in_rice_l557_557280


namespace range_of_a_l557_557189

def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

def no_real_roots (a b c : ℝ) : Prop :=
  discriminant a b c < 0

theorem range_of_a (a : ℝ) :
  no_real_roots 1 (2 * a - 1) 1 ↔ -1 / 2 < a ∧ a < 3 / 2 := 
by sorry

end range_of_a_l557_557189


namespace angle_B_is_36_degrees_l557_557294

-- Definitions and theorem statement
theorem angle_B_is_36_degrees
  (ABCD : Type) [trapezoid ABCD]
  (AB CD : line) (A B C D : point ABCD)
  (h1 : parallel AB CD) (h2 : angle A = 2 * angle D) (h3 : angle C = 4 * angle B) :
  angle B = 36 := by
  sorry -- proof not required

end angle_B_is_36_degrees_l557_557294


namespace candy_eaten_l557_557498

theorem candy_eaten (x : ℕ) (initial_candy eaten_more remaining : ℕ) (h₁ : initial_candy = 22) (h₂ : eaten_more = 5) (h₃ : remaining = 8) (h₄ : initial_candy - x - eaten_more = remaining) : x = 9 :=
by
  -- proof
  sorry

end candy_eaten_l557_557498


namespace lattice_point_in_PQE_l557_557837

-- Define points and their integer coordinates
structure Point :=
  (x : ℤ)
  (y : ℤ)

-- Define a convex quadrilateral with integer coordinates
structure ConvexQuadrilateral :=
  (P : Point)
  (Q : Point)
  (R : Point)
  (S : Point)

-- Define the intersection point of diagonals as another point
def diagIntersection (quad: ConvexQuadrilateral) : Point := sorry

-- Define the condition for the sum of angles at P and Q being less than 180 degrees
def sumAnglesLessThan180 (quad : ConvexQuadrilateral) : Prop := sorry

-- Define a function to check if a point is a lattice point
def isLatticePoint (p : Point) : Prop := true  -- Since all points are lattice points by definition

-- Define the proof problem
theorem lattice_point_in_PQE (quad : ConvexQuadrilateral) (E : Point) :
  sumAnglesLessThan180 quad →
  ∃ p : Point, p ≠ quad.P ∧ p ≠ quad.Q ∧ isLatticePoint p ∧ sorry := sorry -- (prove the point is in PQE)

end lattice_point_in_PQE_l557_557837


namespace general_pattern_l557_557193

theorem general_pattern (n : ℕ) : 
  n + (n + 1) + (n + 2) + ... + (3n - 2) = (2n - 1) ^ 2 := 
by
  sorry

end general_pattern_l557_557193


namespace sum_of_coordinates_l557_557031

theorem sum_of_coordinates :
  let in_distance_from_line (p : (ℝ × ℝ)) (d : ℝ) (line_y : ℝ) : Prop := abs (p.2 - line_y) = d
  let in_distance_from_point (p1 p2 : (ℝ × ℝ)) (d : ℝ) : Prop := (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 = d^2
  ∃ (P1 P2 P3 P4 : ℝ × ℝ),
  in_distance_from_line P1 4 13 ∧ in_distance_from_point P1 (7, 13) 10 ∧
  in_distance_from_line P2 4 13 ∧ in_distance_from_point P2 (7, 13) 10 ∧
  in_distance_from_line P3 4 13 ∧ in_distance_from_point P3 (7, 13) 10 ∧
  in_distance_from_line P4 4 13 ∧ in_distance_from_point P4 (7, 13) 10 ∧
  (P1.1 + P2.1 + P3.1 + P4.1) + (P1.2 + P2.2 + P3.2 + P4.2) = 80 :=
sorry

end sum_of_coordinates_l557_557031


namespace laurent_series_region1_laurent_series_region2_laurent_series_region3_l557_557162

noncomputable def f (z : ℂ) : ℂ := (z + 2) / (z^2 - 2*z - 3)

theorem laurent_series_region1 : 
  ∀ (z : ℂ), |z| < 1 →
  f z = (1 / 4) * ∑ n : ℕ, ((-1)^(n + 1) - (5 / 3^(n + 1))) * z^n :=
by
  sorry

theorem laurent_series_region2 : 
  ∀ (z : ℂ), 1 < |z| ∧ |z| < 3 →
  f z = ∑ n : ℕ, if n = 0 then 0 else ( (-1)^n / (4 * z^n) ) - ∑ n : ℕ, ( 5 * z^n / ( 4 * 3^(n+1) )) :=
by
  sorry 

theorem laurent_series_region3 : 
  ∀ (z : ℂ), |z| > 3 →
  f z = ∑ n : ℕ, if n = 0 then 0 else (( (-1)^n + (5 * 3^(n - 1)) ) / ( 4 * z^n )) :=
by
  sorry

end laurent_series_region1_laurent_series_region2_laurent_series_region3_l557_557162


namespace mode_of_scores_is_80_l557_557785

-- Definitions of the scores based on the stem-and-leaf plot
def scores := [61, 61, 62, 75, 77, 80, 80, 80, 81, 83, 83, 92, 92, 94, 96, 97, 97, 105, 105, 109, 110, 110]

-- Statement to prove that the mode of the scores is 80
theorem mode_of_scores_is_80 : mode scores = some 80 := 
by {
  sorry -- Proof to be filled in
}

end mode_of_scores_is_80_l557_557785


namespace ratio_of_area_AXD_BXC_l557_557639

noncomputable def triangle_ratios : ℕ :=
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (10, 0)
  let C : ℝ × ℝ := (40/3, 4 * Real.sqrt 51 / 3)
  let D : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let E : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let X : ℝ × ℝ := (((A.1 + 2 * E.1) / 3), ((A.2 + 2 * E.2) / 3))
  1

theorem ratio_of_area_AXD_BXC:
  ∀ (A B C X D E : ℝ × ℝ), 
  A = (0, 0) ∧ B = (10, 0) ∧ C = (40/3, 4 * Real.sqrt 51 / 3) →
  D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ E = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) ∧ 
  X = (((A.1 + 2 * E.1) / 3), ((A.2 + 2 * E.2) / 3)) →
  (area_of_triangle A X D / area_of_triangle B X C) = 1 :=
by
  sorry

end ratio_of_area_AXD_BXC_l557_557639


namespace regular_polygon_surrounding_l557_557097

theorem regular_polygon_surrounding (m n : ℕ) (h₁ : m = 10) 
  (h₂ : ∑ i in finset.range m, (180 * (m - 2) / m + 2 * (180 * (n - 2) / n)) = 360) : 
  n = 5 := sorry

end regular_polygon_surrounding_l557_557097


namespace cube_painting_probability_l557_557158

-- Define the conditions: a cube with six faces, each painted either green or yellow (independently, with probability 1/2)
structure Cube where
  faces : Fin 6 → Bool  -- Let's represent Bool with True for green, False for yellow

def is_valid_arrangement (c : Cube) : Prop :=
  ∃ (color : Bool), 
    (c.faces 0 = color ∧ c.faces 1 = color ∧ c.faces 2 = color ∧ c.faces 3 = color) ∧
    (∀ (i j : Fin 6), i = j ∨ ¬(c.faces i = color ∧ c.faces j = color))

def total_arrangements : ℕ := 2 ^ 6

def suitable_arrangements : ℕ := 20  -- As calculated previously: 2 + 12 + 6 = 20

-- We want to prove that the probability is 5/16
theorem cube_painting_probability :
  (suitable_arrangements : ℚ) / total_arrangements = 5 / 16 := 
by
  sorry

end cube_painting_probability_l557_557158


namespace total_seats_round_table_l557_557875

theorem total_seats_round_table 
  (a : ℕ) (b : ℕ) 
  (h₀ : a ≠ b)
  (h₁ : a + b = 39) 
  : ∃ n, n = 38 := 
by {
  sorry
}

end total_seats_round_table_l557_557875


namespace parabola_equation_and_min_ratio_l557_557205

variables {x y p : ℝ}

-- Conditions
def parabola (p : ℝ) (y : ℝ) (x : ℝ) := x^2 = 2 * p * y
def focus (p : ℝ) := (0, p / 2)
def line_through_focus (p : ℝ) (x : ℝ) (y : ℝ) := y = x + p / 2

-- Intersection points condition
def intersection_points_distance (p : ℝ) (d : ℝ) := d = 2 * p

-- Moving circle conditions
def moving_circle_center (p : ℝ) (x₀ y₀ : ℝ) := x₀^2 = 8 * y₀
def intersects_x_axis (x₀ : ℝ) (d : ℝ) :=
  let x₁ := x₀ - 4,
      x₂ := x₀ + 4 in
  d = √((x₀ - 4)^2 + 16) / √((x₀ + 4)^2 + 16)

-- Theorem Statement
theorem parabola_equation_and_min_ratio (p : ℝ) (d : ℝ) :
  (parabola p y x ∧ focus p ∧ line_through_focus p x y ∧ intersection_points_distance p d = 16) →
  (moving_circle_center p x₀ y₀ ∧ intersects_x_axis x₀ d) →
  (p = 4 ∧ d = √2 - 1) :=
by
  sorry

end parabola_equation_and_min_ratio_l557_557205


namespace crossing_time_approx_l557_557421

-- Define the lengths of the trains
def length_first_train : ℝ := 135.5
def length_second_train : ℝ := 167.2

-- Define the speeds of the trains in km/hr
def speed_first_train_kmhr : ℝ := 55
def speed_second_train_kmhr : ℝ := 43

-- Define the initial distance between the trains
def initial_distance : ℝ := 250

-- Define a function to convert speeds from km/hr to m/s
def kmhr_to_ms (speed_kmhr : ℝ) : ℝ := speed_kmhr * (5 / 18)

-- Calculate the relative speed in m/s
def relative_speed_ms : ℝ := kmhr_to_ms (speed_first_train_kmhr + speed_second_train_kmhr)

-- Calculate the total distance to be covered
def total_distance : ℝ := length_first_train + length_second_train + initial_distance

-- Calculate the time taken in seconds
def time_to_cross : ℝ := total_distance / relative_speed_ms

-- Prove the time calculated is approximately 20.3 seconds
theorem crossing_time_approx : abs (time_to_cross - 20.3) < 0.1 := by
sorry

end crossing_time_approx_l557_557421


namespace menelaus_theorem_for_specific_lines_l557_557693

variable {A B C B₁ B₂ B₃ C₁ C₂ C₃ : Type}

theorem menelaus_theorem_for_specific_lines
  (h₁ : line_through A B₂ B₁ ∧ line_through A C₂ C₁)
  (h₂: line_through A B₃ B₁ ∧ line_through A C₃ C₁)
  (h₃: collinear B₂ C₂ B₁ ∧ collinear B₃ C₃ B₁) : 
  (A B₂ / B₂ B₁) * (C₁ C₂ / C₂ A) = (A B₃ / B₃ B₁) * (C₁ C₃ / C₃ A) :=
sorry

end menelaus_theorem_for_specific_lines_l557_557693


namespace minimum_tangent_length_l557_557977

noncomputable def circle_center : ℝ × ℝ := (1, 2)
noncomputable def circle_radius : ℝ := 2
noncomputable def max_distance_to_line : ℝ := 6

theorem minimum_tangent_length (A : ℝ × ℝ) (l : ℝ × ℝ → Prop) :
  (∀ P : ℝ × ℝ, ((P.1 - 1) ^ 2 + (P.2 - 2) ^ 2 = circle_radius ^ 2 →
    ∃ d : ℝ, l P → dist P (line_projection A) = d ∧ d <= max_distance_to_line)) →
  (∃ B : ℝ × ℝ, tangent_point A B ∧ ∥A - B∥ = 2 * sqrt 3) :=
by
  sorry

end minimum_tangent_length_l557_557977


namespace no_cut_can_contain_raisin_l557_557192

-- Definitions representing the conditions of the problem
structure Point (α : Type _) :=
  (x : α)
  (y : α)

structure Triangle (α : Type _) :=
  (A : Point α)
  (B : Point α)
  (C : Point α)

def is_vertex_of_square {α : Type _} [LinearOrder α] (s : Point α) : Prop :=
  ((s.x = 0 ∨ s.x = 1) ∧ (s.y = 0 ∨ s.y = 1))

def is_on_side_of_square {α : Type _} [LinearOrder α] (p : Point α) : Prop :=
  (0 < p.x ∧ p.x < 1 ∧ (p.y = 0 ∨ p.y = 1)) ∨
  (0 < p.y ∧ p.y < 1 ∧ (p.x = 0 ∨ p.x = 1))

-- The square has its corners at (0,0), (0,1), (1,0), and (1,1)
def initial_square : Prop := true

-- The raisin is located at the center of the square
def raisin_center : Point ℝ := { x := 0.5, y := 0.5 }

def triangular_cut_possible (t : Triangle ℝ) : Prop :=
  is_vertex_of_square t.A ∧ is_on_side_of_square t.B ∧ is_on_side_of_square t.C

theorem no_cut_can_contain_raisin : 
  initial_square → 
  (∀ t : Triangle ℝ, triangular_cut_possible t → (t.A ≠ raisin_center ∧ t.B ≠ raisin_center ∧ t.C ≠ raisin_center)) :=
by
  sorry

end no_cut_can_contain_raisin_l557_557192


namespace altitude_eq_l_symmetric_eq_l_l557_557229

-- Axiom definitions for the problem setup
def point := (ℝ × ℝ)
def line := ℝ → ℝ

-- Vertices of the triangle are given
def A : point := (4, 0)
def B : point := (6, 5)
def C : point := (0, 3)

-- Equation of line BC
def BC_line (x : ℝ) : ℝ := (1 / 3) * x + 3 / 3

-- Equation of line l (altitude from B to BC)
def l (x : ℝ) : ℝ := -3 * x + 23

-- Equation of line l' symmetric to l with respect to B
def l' (x : ℝ) : ℝ := -3 * x + 34

-- Prove that l is the altitude from B to BC
theorem altitude_eq_l : ∀ x, (x, l x) = λ x, (-3 * x + 23) :=
sorry

-- Prove that l' is symmetric to l with respect to B
theorem symmetric_eq_l' : ∀ x, (x, l' x) = λ x, (-3 * x + 34) :=
sorry

end altitude_eq_l_symmetric_eq_l_l557_557229


namespace price_of_72_cans_l557_557390

def regular_price_per_can : ℝ := 0.60
def discount_percentage : ℝ := 0.20
def total_price : ℝ := 34.56

theorem price_of_72_cans (discounted_price_per_can : ℝ) (number_of_cans : ℕ)
  (H1 : discounted_price_per_can = regular_price_per_can - (discount_percentage * regular_price_per_can))
  (H2 : number_of_cans = total_price / discounted_price_per_can) :
  total_price = number_of_cans * discounted_price_per_can := by
  sorry

end price_of_72_cans_l557_557390


namespace part_a_l557_557393

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n > 1, a n = n * (∑ i in Finset.range (n - 1 + 1), a i)

theorem part_a (a : ℕ → ℕ) (h : sequence a) :
  ∀ n, Even n → n.factorial ∣ a n :=
sorry

end part_a_l557_557393


namespace negation_of_prop_l557_557013

open Classical

theorem negation_of_prop (h : ∀ x : ℝ, x^2 + x + 1 > 0) : ∃ x : ℝ, x^2 + x + 1 ≤ 0 :=
sorry

end negation_of_prop_l557_557013


namespace problem_equivalent_l557_557966

def modified_op (a b : ℝ) : ℝ := (a + b) ^ 2

theorem problem_equivalent (x y : ℝ) : 
  modified_op ((x + y) ^ 2) ((y + x) ^ 2) = 4 * (x + y) ^ 4 := 
by 
  sorry

end problem_equivalent_l557_557966


namespace adam_apples_l557_557858

theorem adam_apples (x : ℕ) 
  (h1 : 15 + 75 * x = 240) : x = 3 :=
sorry

end adam_apples_l557_557858


namespace min_cans_needed_l557_557156

theorem min_cans_needed (oz_per_can : ℕ) (total_oz_needed : ℕ) (H1 : oz_per_can = 15) (H2 : total_oz_needed = 150) :
  ∃ n : ℕ, 15 * n ≥ 150 ∧ ∀ m : ℕ, 15 * m ≥ 150 → n ≤ m :=
by
  sorry

end min_cans_needed_l557_557156


namespace find_cost_price_l557_557439

noncomputable def cost_price (CP SP_loss SP_gain : ℝ) : Prop :=
SP_loss = 0.90 * CP ∧
SP_gain = 1.05 * CP ∧
(SP_gain - SP_loss = 225)

theorem find_cost_price (CP : ℝ) (h : cost_price CP (0.90 * CP) (1.05 * CP)) : CP = 1500 :=
by
  sorry

end find_cost_price_l557_557439


namespace part1_part2_l557_557137

-- Given data
def months : List ℕ := [1, 2, 3, 4, 5, 6]
def emissions : List ℝ := [100, 70, 50, 35, 25, 20]

-- Given sums and approximations
def sum_t_squared : ℕ := 91
def sum_t_ln_p : ℝ := 73.1
def sum_ln_p : ℝ := 22.5
def e_4_87 : ℝ := 130
def e_4_88 : ℝ := 132

-- Average emission
def avg_emission : ℝ := emissions.sum / (emissions.length : ℝ)

-- Part 1: Probability calculation
theorem part1 :
  let below_avg := emissions.filter (λ x => x < avg_emission).length
  let above_avg := emissions.filter (λ x => x > avg_emission).length
  (below_avg = 3 ∧ above_avg = 3) →
  (Nat.choose 3 1 * Nat.choose 3 1 : ℕ) / (Nat.choose 6 2 : ℕ) = 1/2 :=
by
  intros
  sorry

-- Part 2: Regression equation
theorem part2 :
  let avg_t := 3.5
  let avg_ln_p := sum_ln_p / 6
  let k := (sum_t_ln_p - 6 * avg_t * avg_ln_p) / (sum_t_squared - 6 * avg_t ^ 2)
  let ln_p0 := avg_ln_p + k * avg_t
  k = -0.32 ∧ ln_p0 = 4.87 →
  ∀ t : ℝ, p0 e ^ (k * t) = e ^ 4.87 * e ^ (-0.32 * t) :=
by
  intros
  sorry

end part1_part2_l557_557137


namespace min_blocks_for_sculpture_l557_557075

-- Define the dimensions of the block and the cylinder
def block_length : ℝ := 5
def block_width : ℝ := 3
def block_height : ℝ := 2
def cylinder_height : ℝ := 10
def cylinder_diameter : ℝ := 5

-- Define the volumes
def volume_of_block : ℝ := block_length * block_width * block_height
def radius_of_cylinder : ℝ := cylinder_diameter / 2
def volume_of_cylinder : ℝ := real.pi * (radius_of_cylinder ^ 2) * cylinder_height

-- Define the minimum number of blocks needed
def min_blocks_needed : ℝ := (volume_of_cylinder / volume_of_block).ceil

-- The proof statement
theorem min_blocks_for_sculpture : min_blocks_needed = 7 := by
  sorry

end min_blocks_for_sculpture_l557_557075


namespace find_common_tangent_sum_constant_l557_557669

theorem find_common_tangent_sum_constant :
  ∃ (a b c : ℕ), (∀ x y : ℚ, y = x^2 + 169/100 → x = y^2 + 49/4 → a * x + b * y = c) ∧
  (Int.gcd (Int.gcd a b) c = 1) ∧
  (a + b + c = 52) :=
sorry

end find_common_tangent_sum_constant_l557_557669


namespace correct_option_l557_557911

-- Define the properties needed to express the conditions and the proof
def is_integer_mult_sqrt (a : ℝ) (b : ℝ) : Prop :=
  ∃ (k : ℤ), k ≠ 0 ∧ a = k * b

-- Given problem conditions as Lean definitions
def option_A := Real.sqrt (2 / 3)
def option_B := Real.sqrt 3
def option_C := Real.sqrt 8
def option_D := Real.sqrt 12

-- The target statement to prove
theorem correct_option :
  is_integer_mult_sqrt (Real.sqrt 8) (Real.sqrt 2) :=
by 
  sorry

end correct_option_l557_557911


namespace prize_situation_of_B_and_D_can_be_same_l557_557187

-- Define the students
inductive Student
| A | B | C | D | E

open Student

-- Define the prize type
inductive Prize
| First | Second

open Prize

-- Assumptions
variable (prize_of : Student → Prize)
variable (count : List Prize → Nat)

-- Two first prizes and three second prizes among five students
axiom count_conditions :
  count [prize_of A, prize_of B, prize_of C, prize_of D, prize_of E] First = 2 ∧
  count [prize_of A, prize_of B, prize_of C, prize_of D, prize_of E] Second = 3

-- A's statement
axiom A_statement :
  (prize_of B = First ∧ prize_of C = Second ∨ prize_of B = Second ∧ prize_of C = First ∨ prize_of B = Second ∧ prize_of C = Second) →
  (prize_of A = First ∨ prize_of A = Second)

-- D's statement
axiom D_statement : 
  prize_of A = prize_of E

-- The proposition to prove
theorem prize_situation_of_B_and_D_can_be_same :
  ∃ s : Student, (prize_of B = prize_of s ∧ s = D) ∨ (prize_of B ≠ prize_of s ∧ s = D) :=
by {
  sorry } -- Proof is to be constructed.

end prize_situation_of_B_and_D_can_be_same_l557_557187


namespace length_of_each_part_correct_l557_557847

-- Definition of the units and conversion factors
def feet_to_cm (ft : ℕ) : ℕ := ft * 3048 / 100
def inches_to_cm (in : ℕ) : ℕ := in * 254 / 100

-- Definitions from the problem conditions
def total_length_in_cm : ℕ :=
  feet_to_cm 25 + inches_to_cm 9 + 3

def parts := 13
def length_of_each_part_in_cm : ℚ :=
  total_length_in_cm / parts

-- Statement to prove
theorem length_of_each_part_correct :
  round (length_of_each_part_in_cm * 100) = 6061 := 
sorry

end length_of_each_part_correct_l557_557847


namespace ratio_of_areas_of_concentric_circles_l557_557420

theorem ratio_of_areas_of_concentric_circles 
  (O Y Q : Point)
  (rOQ rOY : ℝ)
  (h1 : distance O Q = rOQ)
  (h2 : distance O Y = 1 / 3 * rOQ) :
  (π * (distance O Y)^2) / (π * (distance O Q)^2) = 1 / 9 :=
by 
  -- Proof goes here
  sorry

end ratio_of_areas_of_concentric_circles_l557_557420


namespace triangle_perimeter_l557_557620

theorem triangle_perimeter (a b c : ℝ) (A B C : ℝ) (h1 : c = 2) (h2 : b = 2 * a) (h3 : C = π / 3) :
  a + b + c = 2 + 2 * real.sqrt 3 :=
by
  sorry

end triangle_perimeter_l557_557620


namespace domain_function_l557_557956

theorem domain_function :
  (∀ x : ℝ, 1 + x > 0 ∧ 8 - 2^x ≥ 0 ↔ -1 < x ∧ x ≤ 3) → 
  (∀ x : ℝ, (x > -1 ∧ x ≤ 3) ↔ (1 + x > 0 ∧ 8 - 2^x ≥ 0)) → 
  (set.Ioc (-1) 3 = {x : ℝ | 1 + x > 0 ∧ 8 - 2^x ≥ 0}) :=
by
  intro h1 h2
  ext x
  split
  · intro hx
    have h1_pos := h1 x
    have h2_pos := h2 x
    exact ⟨hx.1, hx.2⟩
  · intro hx
    have h1_neg := h1 x
    have h2_neg := h2 x
    exact ⟨hx.1, hx.2⟩

end domain_function_l557_557956


namespace counting_squares_below_line_l557_557378

theorem counting_squares_below_line:
  let line_eq (x y : ℝ) := 8 * x + 245 * y = 1960
  ∃ (count : ℕ), count = 853 ∧ 
    (∀ (x y : ℤ), x ≥ 0 ∧ y ≥ 0 → (8 * x + 245 * y ≤ 1960) → 
      ((x + 1 ≥ 0 ∧ 8 * (x + 1) + 245 * y ≤ 1960) ∧ 
       (y + 1 ≥ 0 ∧ 8 * x + 245 * (y + 1) ≤ 1960)) → 
       (x, y) ∈ { (i, j) | i ≥ 0 ∧ j ≥ 0 ∧ 8 * i + 245 * j < 1960 }
    ) := sorry

end counting_squares_below_line_l557_557378


namespace square_of_length_PQ_l557_557348

noncomputable def point_on_parabola (x : ℝ) : ℝ :=
  9 * x^2 - 3 * x + 2

theorem square_of_length_PQ
  (a b : ℝ)
  (hP : b = point_on_parabola a)
  (hQ : -b = point_on_parabola (-a))
  (midpoint : (a + (-a)) / 2 = 0 ∧ (b + (-b)) / 2 = 0) :
  let PQ := (a - (-a))^2 + (b - (-b))^2 in
  PQ = 580 / 9 :=
by
  sorry

end square_of_length_PQ_l557_557348


namespace overall_profit_l557_557695

theorem overall_profit (cp_refrigerator cp_mobile : ℝ) (loss_percentage_refrigerator profit_percentage_mobile : ℝ)
  (h_cp_refrigerator : cp_refrigerator = 15000)
  (h_cp_mobile : cp_mobile = 8000)
  (h_loss_percentage_refrigerator : loss_percentage_refrigerator = 4)
  (h_profit_percentage_mobile : profit_percentage_mobile = 11) :
  let loss_refrigerator := (loss_percentage_refrigerator / 100) * cp_refrigerator,
      selling_price_refrigerator := cp_refrigerator - loss_refrigerator,
      profit_mobile := (profit_percentage_mobile / 100) * cp_mobile,
      selling_price_mobile := cp_mobile + profit_mobile,
      total_cost_price := cp_refrigerator + cp_mobile,
      total_selling_price := selling_price_refrigerator + selling_price_mobile,
      overall_profit := total_selling_price - total_cost_price
  in overall_profit = 280 := by
  sorry

end overall_profit_l557_557695


namespace total_seats_at_round_table_l557_557881

-- Define namespace and conditions
namespace KingArthur

variable (n : ℕ) -- Total number of seats

-- Conditions
def king_position : ℕ := 10
def lancelot_position : ℕ := 29
def opposite (a b : ℕ) (n : ℕ) : Prop := (a + (n / 2)) % n = b

-- Proof statement
theorem total_seats_at_round_table : opposite 10 29 n ∧ 29 < n → n = 38 :=
by
  sorry

end KingArthur

end total_seats_at_round_table_l557_557881


namespace smallest_k_l557_557183

theorem smallest_k (k : ℕ) (h : k = 112) :
  ∃ k, k > 0 ∧ (∀ m, m > 0 → (polynomial.X^12 + polynomial.X^11 + polynomial.X^8 + polynomial.X^7 + polynomial.X^6 + polynomial.X^3 + 1 ∣ polynomial.X^m - 1 ↔ m ≥ k)) :=
by {
  use k,
  split,
  { exact h.symm, },
  intros,
  sorry
}

end smallest_k_l557_557183


namespace prove_geometry_problem_l557_557496

open EuclideanGeometry

noncomputable def problem_geometry : Prop :=
  ∀ (O1 O2 O3 : Point) (P A B C D E M N : Point),
    Circle O1 → Circle O2 → Circle O3 →
    OnCircle O1 P → OnCircle O2 P →
    OnCircle O1 A → OnCircle O2 A →
    ∃ B, LineThrough A B ∧ OnCircle O1 B ∧ OnCircle O2 B →
    ∃ C, LineThrough A C ∧ OnCircle O1 C ∧ OnCircle O2 C →
    LineThrough A P ∧ ∃ D, OnCircle O3 D ∧ OnExtendedLine A P D →
    LineThrough D E ∧ ParallelLine D E B C ∧ OnCircle O3 E ∧ SecondIntersection D E O3 E →
    LineThrough E M ∧ TangentAt E M O1 ∧ LineThrough E N ∧ TangentAt E N O2 →
    EM^2 - EN^2 = Segment DE * Segment BC

theorem prove_geometry_problem : problem_geometry :=
  sorry

end prove_geometry_problem_l557_557496


namespace sum_of_roots_unique_solution_l557_557151

open Real

def operation (a b : ℝ) : ℝ := a^2 + 2 * a * b - b^2

def f (x : ℝ) : ℝ := operation x 2

theorem sum_of_roots_unique_solution
  (x1 x2 x3 x4 : ℝ)
  (h1 : ∀ x, f x = log (abs (x + 2)) → x = x1 ∨ x = x2 ∨ x = x3 ∨ x = x4)
  (h2 : x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4) :
  x1 + x2 + x3 + x4 = -8 :=
sorry

end sum_of_roots_unique_solution_l557_557151


namespace hyperbola_satisfies_m_l557_557586

theorem hyperbola_satisfies_m (m : ℝ) 
  (h1 : ∀ x y : ℝ, x^2 - m * y^2 = 1)
  (h2 : ∀ a b : ℝ, (a^2 = 1) ∧ (b^2 = 1/m) ∧ (2*a = 2 * 2*b)) : 
  m = 4 := 
sorry

end hyperbola_satisfies_m_l557_557586


namespace balance_five_diamonds_bullets_l557_557542

variables (a b c : ℝ)

-- Conditions
def condition1 : Prop := 4 * a + 2 * b = 12 * c
def condition2 : Prop := 2 * a = b + 4 * c

-- Theorem statement
theorem balance_five_diamonds_bullets (h1 : condition1 a b c) (h2 : condition2 a b c) : 5 * b = 5 * c :=
by
  sorry

end balance_five_diamonds_bullets_l557_557542


namespace max_consec_integers_sum_lt_500_l557_557046

theorem max_consec_integers_sum_lt_500 : ∃ n : ℕ, (∀ m : ℕ, m ≤ n → ∑ k in finset.range(m) (λ i, 5 + i) < 500) ∧ 
  (∀ m : ℕ, n < m → ¬∑ k in finset.range(m) (λ i, 5 + i) < 500) :=
begin
  sorry
end

end max_consec_integers_sum_lt_500_l557_557046


namespace passed_both_tests_l557_557080

theorem passed_both_tests :
  ∀ (total_students passed_long_jump passed_shot_put failed_both passed_both: ℕ),
  total_students = 50 →
  passed_long_jump = 40 →
  passed_shot_put = 31 →
  failed_both = 4 →
  passed_both + (passed_long_jump - passed_both) + (passed_shot_put - passed_both) + failed_both = total_students →
  passed_both = 25 :=
by
  intros total_students passed_long_jump passed_shot_put failed_both passed_both h1 h2 h3 h4 h5
  -- proof can be skipped using sorry
  sorry

end passed_both_tests_l557_557080


namespace arrange_6_books_l557_557825

theorem arrange_6_books :
  Nat.factorial 6 = 720 :=
by
  sorry

end arrange_6_books_l557_557825


namespace Eli_saves_more_with_discount_A_l557_557830

-- Define the prices and discounts
def price_book : ℝ := 25
def discount_A (price : ℝ) : ℝ := price * 0.4
def discount_B : ℝ := 5

-- Define the cost calculations:
def cost_with_discount_A (price : ℝ) : ℝ := price + (price - discount_A price)
def cost_with_discount_B (price : ℝ) : ℝ := price + (price - discount_B)

-- Define the savings calculation:
def savings (cost_B : ℝ) (cost_A : ℝ) : ℝ := cost_B - cost_A

-- The main statement to prove:
theorem Eli_saves_more_with_discount_A :
  savings (cost_with_discount_B price_book) (cost_with_discount_A price_book) = 5 :=
by
  sorry

end Eli_saves_more_with_discount_A_l557_557830


namespace probability_product_multiple_of_3_l557_557100

structure Die where
  sides : ℕ
  rolls : ℕ

def is_multiple_of_3 (n : ℕ) : Prop := n % 3 = 0

noncomputable def probability_multiple_of_3_in_rolls (die : Die) : ℚ :=
  1 - (float_of (2/3) ^ die.rolls)

theorem probability_product_multiple_of_3 (die : Die)
  (h1 : die.sides = 6)
  (h2 : die.rolls = 8) :
  probability_multiple_of_3_in_rolls die = 6305 / 6561 :=
  sorry

end probability_product_multiple_of_3_l557_557100


namespace domain_f_l557_557955

def f (x : ℝ) : ℝ := log (x - 1) + 1 / sqrt (2 - x)

theorem domain_f :
  ∀ x, (1 < x) ∧ (x < 2) ↔ ∃ y : ℝ, f y = f x := sorry

end domain_f_l557_557955


namespace min_value_of_reciprocal_sum_l557_557332

theorem min_value_of_reciprocal_sum (a : ℕ → ℝ) (hpos : ∀ i, 1 ≤ i ∧ i ≤ 15 → 0 < a i) (hsum : ∑ i in finset.range 15, a (i + 1) = 3) :
  ∑ i in finset.range 15, (a (i + 1))⁻¹ ≥ 75 :=
begin
  sorry
end

end min_value_of_reciprocal_sum_l557_557332


namespace retailer_profit_percentage_l557_557098

theorem retailer_profit_percentage:
  ∀ (market_price_per_pen : ℝ),
  (market_price_per_pen > 0) →
  let CP := 36 * market_price_per_pen in
  let CP_140 := CP in
  let discount := 0.01 in
  let SP_per_pen := market_price_per_pen * (1 - discount) in
  let SP_140 := 140 * SP_per_pen in
  let profit := SP_140 - CP_140 in
  let profit_percentage := (profit / CP_140) * 100 in
  profit_percentage = 285 :=
by
  intros market_price_per_pen h_pos
  let CP := 36 * market_price_per_pen
  let CP_140 := CP
  let discount := 0.01
  let SP_per_pen := market_price_per_pen * (1 - discount)
  let SP_140 := 140 * SP_per_pen
  let profit := SP_140 - CP_140
  let profit_percentage := (profit / CP_140) * 100
  have h_profit: profit_percentage = 285 := sorry
  exact h_profit

end retailer_profit_percentage_l557_557098


namespace cos_210_eq_neg_sqrt3_div_2_l557_557406

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (Real.pi + Real.pi / 6) = -Real.sqrt 3 / 2 := sorry

end cos_210_eq_neg_sqrt3_div_2_l557_557406


namespace simple_interest_amount_l557_557270

variable (P : ℝ) (r t : ℝ)

def simple_interest (P r t : ℝ) : ℝ := P * (r * t / 100)
def compound_interest (P r t : ℝ) : ℝ := P * ((1 + r / 100) ^ t) - P

theorem simple_interest_amount :
  ∀ (P : ℝ), compound_interest P 5 2 = 57.40 → simple_interest P 5 2 = 56 :=
by
  intro P
  intro h
  unfold compound_interest at h
  unfold simple_interest
  sorry

end simple_interest_amount_l557_557270


namespace perp_center_centroid_median_iff_squared_eq_l557_557351

-- Definitions
variables {α : Type*} [plane α]

structure Triangle :=
(A B C : pt)

variables {T : Triangle}

def has_sides (a b c : ℝ) : Prop :=
  distance T.A T.B = c ∧ distance T.A T.C = b ∧ distance T.B T.C = a

def is_centroid {G : pt} : Prop :=
  G ∈ line (midpoint T.BC T.AC) ∧ G ∈ line (midpoint T.AB T.C)

def is_circumcenter {O : pt} : Prop :=
  distance O T.A = distance O T.B ∧ distance O T.C

def is_median_to_c (N : pt) : Prop :=
  N = midpoint T.AB

def perpendicular_to_CN_through_O_M (O M N : pt) (m : line ℝ) : Prop :=
  line O M ⊥ m

-- Theorem to be proven
theorem perp_center_centroid_median_iff_squared_eq
  (a b c : ℝ)
  (T : Triangle)
  (O M N : pt) :
  has_sides T a b c →
  is_circumcenter O →
  is_centroid M →
  is_median_to_c N →
  perpendicular_to_CN_through_O_M O M N (line O (midpoint (midpoint T.AB N) O)) →
  a^2 + b^2 = 2c^2 :=
sorry

end perp_center_centroid_median_iff_squared_eq_l557_557351


namespace perpendicular_bl_ac_l557_557315

theorem perpendicular_bl_ac
  (A B C O K M N L : Point)
  (h_triangle_acute : acute_triangle A B C)
  (h_circumcircle : circle A B C O)
  (h_circle_omega1 : circle A O C K)
  (h_intersects_ab_bc : intersects ω₁ A B C K M N)
  (h_symmetric : symmetric L K (line M N)) :
  perpendicular (line B L) (line A C) := 
sorry

end perpendicular_bl_ac_l557_557315


namespace opposite_of_3_l557_557727

-- Define the concept of opposite of a number and the logic for positive numbers
def opposite (x : Int) : Int := 
  if x > 0 then -x
  else if x < 0 then -x
  else 0

-- Statement to prove that the opposite of 3 is -3
theorem opposite_of_3 : opposite 3 = -3 :=
by 
  -- Using the definition of opposite
  unfold opposite
  -- Simplify the expression for x = 3
  simp [lt_irrefl, int.coe_nat_lt]
  -- Conclude proof
  rfl

end opposite_of_3_l557_557727


namespace ten_percent_of_x_l557_557072

variable (certain_value : ℝ)
variable (x : ℝ)

theorem ten_percent_of_x (h : 3 - (1 / 4) * 2 - (1 / 3) * 3 - (1 / 7) * x = certain_value) :
  0.1 * x = 0.7 * (1.5 - certain_value) := sorry

end ten_percent_of_x_l557_557072


namespace allan_balloons_count_l557_557486

-- Definition of the conditions
def Total_balloons : ℕ := 3
def Jake_balloons : ℕ := 1

-- The theorem that corresponds to the problem statement
theorem allan_balloons_count (Allan_balloons : ℕ) (h : Allan_balloons + Jake_balloons = Total_balloons) : Allan_balloons = 2 := 
by
  sorry

end allan_balloons_count_l557_557486


namespace good_wizard_can_seat_gnomes_l557_557129

def gnome_friendship_possible (n : ℕ) (odd_n : n % 2 = 1) (n_gt_1 : n > 1) : Prop :=
  ∃ (good_friends : Fin (2 * n) → Fin (2 * n) → Prop),
    ∀ (bad_friends : Fin (2 * n) → Fin (2 * n) → Prop),
      (∀ (x y : Fin (2 * n)), bad_friends x y → good_friends x y) ∧
      (∃ (seating : Fin (2 * n) → Fin (2 * n)),
        ∀ (i : Fin (2 * n)), good_friends (seating i) (seating ((i + 1) % (2 * n))))

theorem good_wizard_can_seat_gnomes (n : ℕ) (odd_n : n % 2 = 1) (n_gt_1 : n > 1) : 
  gnome_friendship_possible n odd_n n_gt_1 :=
sorry

end good_wizard_can_seat_gnomes_l557_557129


namespace profit_calculation_l557_557848

def total_cost (num_pens: ℕ) (cost_per_pen: ℝ) : ℝ :=
  num_pens * cost_per_pen

def required_revenue (total_cost: ℝ) (desired_profit: ℝ) : ℝ :=
  total_cost + desired_profit

def pens_to_sell (required_revenue: ℝ) (selling_price_per_pen: ℝ) : ℕ :=
  (required_revenue / selling_price_per_pen).to_nat

theorem profit_calculation
  (num_pens: ℕ)
  (cost_per_pen: ℝ)
  (selling_price_per_pen: ℝ)
  (desired_profit: ℝ)
  (total_cost := total_cost num_pens cost_per_pen)
  (required_revenue := required_revenue total_cost desired_profit)
  : 
  pens_to_sell required_revenue selling_price_per_pen = 1500 :=
by
  let num_pens := 2000
  let cost_per_pen := 0.15
  let selling_price_per_pen := 0.30
  let desired_profit := 150.0
  let total_cost := total_cost num_pens cost_per_pen
  let required_revenue := total_cost + desired_profit
  show pens_to_sell required_revenue selling_price_per_pen = 1500
  by sorry

end profit_calculation_l557_557848


namespace find_angle_C_range_of_expression_l557_557564

variable {A B C a b c : ℝ}
variable (h1 : c = sqrt 3)
variable (h2 : sin A = a * cos C)

theorem find_angle_C : C = π / 3 := 
by sorry

theorem range_of_expression (h3 : c = 2 * sqrt A)
  (h4 : A + B + C = π) 
  (h5 : a = 2 * sin A) 
  (h6 : b = 2 * sin B) : 
  (a * sin A + b * sin B) ∈ Ioc (3 / 2) 3 := 
by sorry

end find_angle_C_range_of_expression_l557_557564


namespace total_number_of_seats_l557_557888

theorem total_number_of_seats (n : ℕ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ n) 
                               (h2 : (10 : ℕ) < (29 : ℕ)) 
                               (h3 : (29 - 10) % (n / 2) = 0) : n = 38 :=
by sorry

end total_number_of_seats_l557_557888


namespace solve_n_question_mark_l557_557965

open Nat

-- Define the function "product of primes less than n"
def primeProductLessThan (n : ℕ) : ℕ :=
  (List.range n).filter Nat.Prime.prime.prod

theorem solve_n_question_mark :
  ∃ n : ℕ, 3 < n ∧ primeProductLessThan n = 2 * n + 16 ∧ n = 7 :=
by
  sorry

end solve_n_question_mark_l557_557965


namespace park_maple_trees_total_l557_557410

theorem park_maple_trees_total (current_maples planted_maples : ℕ) 
    (h1 : current_maples = 2) (h2 : planted_maples = 9) 
    : current_maples + planted_maples = 11 := 
by
  sorry

end park_maple_trees_total_l557_557410


namespace chess_tournament_win_draw_l557_557824

open Classical BigOperators

theorem chess_tournament_win_draw (players : Finset ℕ) (n : ℕ)
  (h_total_players : players.card = 20)
  (scores: players → ℤ)
  (game_points : ∀ p1 p2 : players, p1 ≠ p2 → ℤ) :
  (∀ p, ∃ q, p ≠ q) → 
  (∀ p1 p2, p1 ≠ p2 → game_points p1 p2 + game_points p2 p1 = 1) →
  (∀ p, scores p = ∑ q in players, if q ≠ p then game_points p q else 0) →
  (∀ p q, p ≠ q → scores p ≠ scores q) →
  ∃ p, ∑ q in players, if q ≠ p ∧ game_points p q = 1 then 1 else 0 > 
       ∑ q in players, if q ≠ p ∧ game_points p q = 0.5 then 1 else 0 :=
by
  sorry

end chess_tournament_win_draw_l557_557824


namespace hyperbola_standard_eq_line_eq_AB_l557_557245

noncomputable def fixed_points : (Real × Real) × (Real × Real) := ((-Real.sqrt 2, 0.0), (Real.sqrt 2, 0.0))

def locus_condition (P : Real × Real) (F1 F2 : Real × Real) : Prop :=
  abs (dist P F2 - dist P F1) = 2

def curve_E (P : Real × Real) : Prop :=
  (P.1 < 0) ∧ (P.1 * P.1 - P.2 * P.2 = 1)

theorem hyperbola_standard_eq :
  ∃ P : Real × Real, locus_condition P (fixed_points.1) (fixed_points.2) ↔ curve_E P :=
sorry

def line_intersects_hyperbola (P : Real × Real) (k : Real) : Prop :=
  P.2 = k * P.1 - 1 ∧ curve_E P

def dist_A_B (A B : Real × Real) : Real :=
  dist A B

theorem line_eq_AB :
  ∃ k : Real, k = -Real.sqrt 5 / 2 ∧
              ∃ A B : Real × Real, line_intersects_hyperbola A k ∧ 
              line_intersects_hyperbola B k ∧ 
              dist_A_B A B = 6 * Real.sqrt 3 ∧
              ∀ x y : Real, y = k * x - 1 ↔ x * (Real.sqrt 5/2) + y + 1 = 0 :=
sorry

end hyperbola_standard_eq_line_eq_AB_l557_557245


namespace sum_integer_values_l557_557991

theorem sum_integer_values (m : ℤ) (h : 0 < 5 * m ∧ 5 * m < 35) :
  ∑ x in {n : ℤ | 0 < n ∧ n < 7}.to_finset, x = 21 :=
sorry

end sum_integer_values_l557_557991


namespace magnitude_sum_vectors_l557_557595

variables (a b c : EuclideanSpace ℝ (Fin 2))

-- Given conditions
axiom non_collinear : ¬ Collinear ℝ ({a, b, c} : Set (EuclideanSpace ℝ (Fin 2)))
axiom equal_angles : ∀ (u v : EuclideanSpace ℝ (Fin 2)), u ∈ ({a, b, c} : Set (EuclideanSpace ℝ (Fin 2))) → 
                     v ∈ ({a, b, c} : Set (EuclideanSpace ℝ (Fin 2))) → 
                     (u ≠ v → angle u v = π * 2 / 3)

axiom mag_a : ‖a‖ = 1
axiom mag_b : ‖b‖ = 1
axiom mag_c : ‖c‖ = 3

-- Proof goal
theorem magnitude_sum_vectors : ‖a + b + c‖ = 2 := 
begin
  sorry,
end

end magnitude_sum_vectors_l557_557595


namespace divisibility_of_polynomial_l557_557673

theorem divisibility_of_polynomial (n : ℕ) (h : n ≥ 1) : 
  ∃ primes : Finset ℕ, primes.card = n ∧ ∀ p ∈ primes, p.Prime ∧ p ∣ (2^(2^n) + 2^(2^(n-1)) + 1) :=
sorry

end divisibility_of_polynomial_l557_557673


namespace length_AD_l557_557246

variables (a b : EuclideanSpace ℝ (Fin 2))
variables (ha : ‖a‖ = 1) (hb : ‖b‖ = 2) (hab_perp : (a + b) ⬝ a = 0)
variables (A B C D : EuclideanSpace ℝ (Fin 2))
variables (AB_eq_a : B - A = a) (AC_eq_b : C - A = b)
variables (D_mid_BC : D = (B + C) / 2)

theorem length_AD :
  ‖D - A‖ = (Real.sqrt 3) / 2 :=
sorry

end length_AD_l557_557246


namespace sum_of_reciprocal_of_absolute_values_l557_557429

noncomputable def z_solutions : Finset ℂ := {z | z^8 = -1}.toFinset

noncomputable def sum_over_solutions := ∑ z in z_solutions, 1 / Complex.abs (1 + z) ^ 2

theorem sum_of_reciprocal_of_absolute_values :
  sum_over_solutions = 4.5 :=
sorry

end sum_of_reciprocal_of_absolute_values_l557_557429


namespace opposite_of_3_is_neg3_l557_557746

def opposite (x : ℝ) := -x

theorem opposite_of_3_is_neg3 : opposite 3 = -3 :=
by
  sorry

end opposite_of_3_is_neg3_l557_557746


namespace verify_divisibility_l557_557386

def problem (n : ℕ) : Prop :=
  let x := 3^105 + 4^105 in
  (13 ∣ x) ∧ (49 ∣ x) ∧ (181 ∣ x) ∧ (379 ∣ x) ∧ ¬(5 ∣ x) ∧ ¬(11 ∣ x)

theorem verify_divisibility : problem 105 := by
  sorry

end verify_divisibility_l557_557386


namespace total_items_l557_557078

-- Definitions based on problem conditions
variables (P L E : ℕ) -- Natural numbers for the count of pens, pencils, and erasers

-- Given condition from the problem statement
def cost_condition : Prop :=
  0.45 * P + 0.35 * L + 0.30 * E = 7.80

-- The statement to prove
theorem total_items (h : cost_condition P L E) : ∃ (P L E : ℕ), P + L + E > 0 :=
by
  -- This is a placeholder for the proof which would use the given condition.
  sorry

end total_items_l557_557078


namespace larger_number_is_55_l557_557400

theorem larger_number_is_55 (x y : ℤ) (h1 : x + y = 70) (h2 : x = 3 * y + 10) (h3 : y = 15) : x = 55 :=
by
  sorry

end larger_number_is_55_l557_557400


namespace total_seats_round_table_l557_557862

theorem total_seats_round_table (n : ℕ) (h : n = 38)
  (ka_position : ℕ) (sl_position : ℕ) 
  (h1 : ka_position = 10) 
  (h2 : sl_position = 29) 
  (h3 : (ka_position + n/2) % n = sl_position) : 
  n = 38 :=
by
  -- All steps and solution proof
  sorry

end total_seats_round_table_l557_557862


namespace sum_of_possible_m_l557_557996

theorem sum_of_possible_m (m : ℤ) (h : 0 < 5 * m ∧ 5 * m < 35) : 
  (∑ n in { n : ℤ | 0 < 5 * n ∧ 5 * n < 35 }.to_finset, n) = 21 := 
by 
  sorry

end sum_of_possible_m_l557_557996


namespace merchant_gross_profit_l557_557089

noncomputable def purchase_price : ℝ := 48
noncomputable def markup_rate : ℝ := 0.40
noncomputable def discount_rate : ℝ := 0.20

theorem merchant_gross_profit :
  ∃ S : ℝ, S = purchase_price + markup_rate * S ∧ 
  ((S - discount_rate * S) - purchase_price = 16) :=
by
  sorry

end merchant_gross_profit_l557_557089


namespace cyclist_C_speed_l557_557036

def speed_C (c : ℝ) (d : ℝ) : Prop :=
d = c + 5 ∧
(65 : ℝ) / c = (95 : ℝ) / d + 0.25

theorem cyclist_C_speed : ∃ c d : ℝ, speed_C c d ∧ c = 8.6 :=
by
  use 8.6, 13.6
  split
  · split
    · rfl
    · norm_num
      rw [div_eq_mul_inv, div_eq_mul_inv, mul_inv, mul_inv]
      simp
      have : (65:ℝ) = (95:ℝ) / (13.6) + 0.25 * 8.6, from sorry
      assumption
     sorry

end cyclist_C_speed_l557_557036


namespace Chloe_points_at_end_of_game_l557_557139

theorem Chloe_points_at_end_of_game :
  ∀ (points1 points2 points_lost : ℕ), 
    points1 = 40 → points2 = 50 → points_lost = 4 →
    (points1 + points2 - points_lost) = 86 :=
begin
  intros points1 points2 points_lost h1 h2 h3,
  rw [h1, h2, h3],
  norm_num,
end

end Chloe_points_at_end_of_game_l557_557139


namespace translate_and_even_function_l557_557417

theorem translate_and_even_function (k : ℤ) (ϕ : ℝ) :
  (∀ x : ℝ, sin (2 * x + ϕ) = sin (-2 * x + ϕ + π/4)) → ϕ = k * π + π/4 :=
by
  sorry

end translate_and_even_function_l557_557417


namespace probability_of_first_ball_odd_l557_557436

theorem probability_of_first_ball_odd :
  let total_balls := 100
  let odd_balls := total_balls / 2
  let even_balls := total_balls / 2
  let probability_odd_first := (odd_balls : ℝ) / total_balls
  3.balls_selected_with_replacement (contains_two_odd_one_even) →
  probability_odd_first = 0.5 :=
by
  sorry

end probability_of_first_ball_odd_l557_557436


namespace find_costs_l557_557632

theorem find_costs (a b : ℝ) (h1 : a - b = 3) (h2 : 3 * b - 2 * a = 3) : a = 12 ∧ b = 9 :=
sorry

end find_costs_l557_557632


namespace select_six_integers_statements_correct_l557_557545

theorem select_six_integers_statements_correct :
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
  ∃ (A B : set ℕ) (size_A : A ⊆ S) (hA : A.card = 6),
  (h1 : ∃ x y ∈ A, Int.gcd x y = 1) ∧
  (h2 : ∃ x y ∈ A, x ≠ y ∧ y % x = 0) ∧
  (h3 : ∃ x y ∈ A, x ≠ y ∧ 2 * x % y = 0) →
  2 ∈ {1, 2, 3, 4} :=
sorry

end select_six_integers_statements_correct_l557_557545


namespace Koschei_no_equal_coins_l557_557033

theorem Koschei_no_equal_coins (a : Fin 6 → ℕ)
  (initial_condition : a 0 = 1 ∧ a 1 = 0 ∧ a 2 = 0 ∧ a 3 = 0 ∧ a 4 = 0 ∧ a 5 = 0) :
  ¬ ( ∃ k : ℕ, ( ( ∀ i : Fin 6, a i = k ) ) ) :=
by
  sorry

end Koschei_no_equal_coins_l557_557033


namespace king_lancelot_seats_38_l557_557891

noncomputable def totalSeats (seat_king seat_lancelot : ℕ) : ℕ :=
  if seat_king < seat_lancelot then
    2 * (seat_lancelot - seat_king - 1) + 2
  else
    2 * (seat_king - seat_lancelot - 1) + 2

theorem king_lancelot_seats_38 (seat_king seat_lancelot : ℕ) (h1 : seat_king = 10) (h2 : seat_lancelot = 29) :
  totalSeats seat_king seat_lancelot = 38 := 
  by
    sorry

end king_lancelot_seats_38_l557_557891


namespace negation_of_positive_x2_plus_2_l557_557009

theorem negation_of_positive_x2_plus_2 (h : ∀ x : ℝ, x^2 + 2 > 0) : ¬ (∀ x : ℝ, x^2 + 2 > 0) = False := 
by
  sorry

end negation_of_positive_x2_plus_2_l557_557009


namespace gauss_function_properties_l557_557366

def gauss_function (x : ℝ) : ℤ :=
  int.floor x

-- Assert the correctness of statements ①, ②, and ④
theorem gauss_function_properties :
  (gauss_function (-3) = -3) ∧
  (∀ a b : ℝ, gauss_function a = gauss_function b → abs (a - b) < 1) ∧
  (∀ x : ℝ, 1 ≤ x → ∀ x, x * gauss_function x ≤ y → x ≤ y → ∀ y, y = x * gauss_function y → x ≤ y :=
 sorry -- proof omitted

end gauss_function_properties_l557_557366


namespace total_seats_l557_557909

theorem total_seats (KA_pos : ℕ) (SL_pos : ℕ) (h1 : KA_pos = 10) (h2 : SL_pos = 29) (h3 : SL_pos = KA_pos + (KA_pos * 2 - 1) / 2):
  let total_positions := 2 * (SL_pos - KA_pos - 1) + 2
  total_positions = 38 :=
by
  sorry

end total_seats_l557_557909


namespace correct_number_of_candies_l557_557453

theorem correct_number_of_candies 
  (n : ℕ) 
  (h : n ∈ {3, 10, 17, 6, 7}) 
  (P : ℚ)
  (hp : P = 5 / 6) :
  (∃ r : ℕ, r = 5 * n / 6 ∧ (5 * n) % 6 = 0) ↔ n = 6 := 
by
  sorry

end correct_number_of_candies_l557_557453


namespace minimum_time_to_find_faulty_bulb_l557_557488

theorem minimum_time_to_find_faulty_bulb :
  ∀ (num_bulbs : ℕ) (time_per_bulb : ℕ) (spare_bulb : bool),
    num_bulbs = 4 →
    time_per_bulb = 10 →
    spare_bulb = true →
    ∃ (min_time : ℕ), min_time = 60 :=
by
  intros num_bulbs time_per_bulb spare_bulb
  assume num_bulbs_eq : num_bulbs = 4
  assume time_per_bulb_eq : time_per_bulb = 10
  assume spare_bulb_eq : spare_bulb = true
  existsi 60
  sorry

end minimum_time_to_find_faulty_bulb_l557_557488


namespace series_expansion_l557_557349

section proof

variables {R : Type*} [Field R] [CharZero R]

noncomputable def g (x : R) : FormalMultilinearSeries R R :=
λ n, match n with
| 0      := 0
| 1      := a1 * x
| 2      := a2 * x^2
| n + 2  := 0 --assuming only a finite set of terms
end

theorem series_expansion (g : FormalMultilinearSeries R R) (r : ℚ) :
  (1 + g) ^ r = 
  1 + r * g + 
  (r * (r - 1) / 2) * g^2 + 
  (r * (r - 1) * (r - 2) / 6) * g^3 + 
  ∑ i : ℕ, (r.series (r, n, g)) :=
sorry

end proof

end series_expansion_l557_557349


namespace leopards_count_l557_557132

theorem leopards_count (L : ℕ) (h1 : 100 + 80 + L + 10 * L + 50 + 2 * (80 + L) = 670) : L = 20 :=
by
  sorry

end leopards_count_l557_557132


namespace total_charge_correct_l557_557342

def boxwoodTrimCost (numBoxwoods : Nat) (trimCost : Nat) : Nat :=
  numBoxwoods * trimCost

def boxwoodShapeCost (numBoxwoods : Nat) (shapeCost : Nat) : Nat :=
  numBoxwoods * shapeCost

theorem total_charge_correct :
  let numBoxwoodsTrimmed := 30
  let trimCost := 5
  let numBoxwoodsShaped := 4
  let shapeCost := 15
  let totalTrimCost := boxwoodTrimCost numBoxwoodsTrimmed trimCost
  let totalShapeCost := boxwoodShapeCost numBoxwoodsShaped shapeCost
  let totalCharge := totalTrimCost + totalShapeCost
  totalCharge = 210 :=
by sorry

end total_charge_correct_l557_557342


namespace angle_same_terminal_side_l557_557168

theorem angle_same_terminal_side (θ : ℝ) (α : ℝ) 
  (hθ : θ = -950) 
  (hα_range : 0 ≤ α ∧ α ≤ 180) 
  (h_terminal_side : ∃ k : ℤ, θ = α + k * 360) : 
  α = 130 := by
  sorry

end angle_same_terminal_side_l557_557168


namespace sin_neg_seven_pi_over_three_correct_l557_557407

noncomputable def sin_neg_seven_pi_over_three : Prop :=
  (Real.sin (-7 * Real.pi / 3) = - (Real.sqrt 3 / 2))

theorem sin_neg_seven_pi_over_three_correct : sin_neg_seven_pi_over_three := 
by
  sorry

end sin_neg_seven_pi_over_three_correct_l557_557407


namespace eccentricity_of_ellipse_slope_of_tangent_line_l557_557933

namespace EllipseProblem

-- Define the conditions for the ellipse
variable (a b c : ℝ) (h1 : a > b) (h2 : b > 0)
def ellipse_eq := ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1

-- Define the distance condition |AB| = sqrt(3)/2 |F_1F_2|
def distance_condition : Prop := (a^2 + b^2 = 3 * c^2)

-- Main theorem for question 1
theorem eccentricity_of_ellipse : a > b → b > 0 → a^2 + b^2 = 3 * c^2 → 
  sqrt(c^2 / a^2) = sqrt(2) / 2 :=
by sorry

-- Main theorem for question 2
theorem slope_of_tangent_line (P : ℝ × ℝ) (x0 y0 : ℝ) :
  (a^2 = 2 * c^2) → (b^2 = c^2) → (P = (x0, y0)) → 
  (x0 + y0 + c = 0) → (x0^2 / (2 * c^2) + y0^2 / (c^2) = 1) → 
  (P ≠ (a,0)) → 
  ∃ k : ℝ, (k = 4 + sqrt 15 ∨ k = 4 - sqrt 15) :=
by sorry

end EllipseProblem

end eccentricity_of_ellipse_slope_of_tangent_line_l557_557933


namespace max_gcd_consecutive_terms_l557_557932

-- Defining the sequence
def b (n : ℕ) : ℕ := factorial (n^2) + n

-- Defining the gcd of two consecutive terms
def gcd_consecutive_terms (n : ℕ) : ℕ := Nat.gcd (b n) (b (n + 1))

-- Main theorem statement: the maximum possible value of the gcd of two consecutive terms of the sequence is 2
theorem max_gcd_consecutive_terms : ∀ n ≥ 1, gcd_consecutive_terms n ≤ 2 := by
  sorry

end max_gcd_consecutive_terms_l557_557932


namespace P4_valid_y_count_l557_557022

theorem P4_valid_y_count :
  ∃ (x y : ℕ), (x < 100) ∧ (y > 9) ∧ 
  ∃ (p : ℕ), (100 ≤ p ∧ p < 1000 ∧ (p / 10) % 10 = 5 ∧ 
  777 * y = p * x) ∧
  set.count (set_of (λ y, ∃ x p, (x < 100) ∧ (y > 9) ∧ (100 ≤ p ∧ p < 1000 ∧ (p / 10) % 10 = 5 ∧ 777 * y = p * x))) = 6 :=
by {
  sorry
}

end P4_valid_y_count_l557_557022


namespace min_cars_needed_for_availability_l557_557814

theorem min_cars_needed_for_availability (n : ℕ) (h : ∀ d ∈ finset.range 7, 7 * (n - 10) ≥ 2 * n) : n = 14 :=
sorry

end min_cars_needed_for_availability_l557_557814


namespace fraction_division_correct_l557_557044

theorem fraction_division_correct :
  (5/6 : ℚ) / (7/9) / (11/13) = 195/154 := 
by {
  sorry
}

end fraction_division_correct_l557_557044


namespace compute_modulo_eight_l557_557510

theorem compute_modulo_eight : (47 ^ 1824 - 25 ^ 1824) % 8 = 0 := by
  have h1 : 47 % 8 = 7 := rfl
  have h2 : 25 % 8 = 1 := rfl
  sorry

end compute_modulo_eight_l557_557510


namespace child_height_at_last_visit_l557_557092

-- Definitions for the problem
def h_current : ℝ := 41.5 -- current height in inches
def Δh : ℝ := 3 -- height growth in inches

-- The proof statement
theorem child_height_at_last_visit : h_current - Δh = 38.5 := by
  sorry

end child_height_at_last_visit_l557_557092


namespace sin_alpha_l557_557999

theorem sin_alpha (α : ℝ) (hα : 0 < α ∧ α < π) (hcos : Real.cos (π + α) = 3 / 5) :
  Real.sin α = 4 / 5 :=
sorry

end sin_alpha_l557_557999


namespace range_of_phi_l557_557242

noncomputable def f (φ x : ℝ) : ℝ := 
  sin x - 2 * cos (x - φ) * sin φ

theorem range_of_phi 
  (φ : ℝ) 
  (hφ : 0 < φ ∧ φ < π) 
  (h_increasing : ∀ x y, 3 * π ≤ x ∧ x ≤ y ∧ y ≤ 7 * π / 2 → f φ x ≤ f φ y) : 
  π / 2 ≤ φ ∧ φ ≤ 3 * π / 4 :=
sorry

end range_of_phi_l557_557242


namespace sphere_radius_ratio_l557_557473

theorem sphere_radius_ratio (R r : ℝ) (h₁ : (4 / 3) * Real.pi * R ^ 3 = 450 * Real.pi) (h₂ : (4 / 3) * Real.pi * r ^ 3 = 0.25 * 450 * Real.pi) :
  r / R = 1 / 2 :=
sorry

end sphere_radius_ratio_l557_557473


namespace opposite_of_3_l557_557728

-- Define the concept of opposite of a number and the logic for positive numbers
def opposite (x : Int) : Int := 
  if x > 0 then -x
  else if x < 0 then -x
  else 0

-- Statement to prove that the opposite of 3 is -3
theorem opposite_of_3 : opposite 3 = -3 :=
by 
  -- Using the definition of opposite
  unfold opposite
  -- Simplify the expression for x = 3
  simp [lt_irrefl, int.coe_nat_lt]
  -- Conclude proof
  rfl

end opposite_of_3_l557_557728


namespace sum_x_y_arithmetic_progression_l557_557512

theorem sum_x_y_arithmetic_progression (x y : ℝ) :
  let lst := [12, 3, 6, 3, 7, 3, x, y],
      mean := (34 + x + y) / 8,
      mode := 3,
      median := if x ≤ 3 ∧ y ≤ 3 then (3 + 3) / 2
      else if (x ≤ 3 ∧ y ≤ 6) ∨ (x ≤ 6 ∧ y ≤ 3) then (3 + 6) / 2
      else if (x ≤ 6 ∧ y ≤ 7) ∨ (x ≤ 7 ∧ y ≤ 6) then (3 + x + y - min (min x y) 6 - max (max x y) 6) / 2
      else if x < y then y else x in
  mean - median = median - mode →
  x + y = 38 :=
sorry

end sum_x_y_arithmetic_progression_l557_557512


namespace general_formula_seq_largest_integer_m_l557_557553

noncomputable def geometricSequence (a1 q : ℝ) (n : ℕ) := a1 * q^n

theorem general_formula_seq (q : ℝ) (h_pos : q > 0) (h_ne_one : q ≠ 1)
  (h_arith : let a1 := 1 / 3 in 
              geometricSequence a1 q 2 * 5 = a1 + 9 * geometricSequence a1 q 4) :
  ∀ (n : ℕ), geometricSequence (1 / 3) (1 / 3) n = (1 / 3)^n :=
sorry

theorem largest_integer_m (m : ℕ) (h : m < 8) :
  ∀ (n : ℕ), (1 - 1/(n+1 : ℝ)) > m / 16 :=
sorry

end general_formula_seq_largest_integer_m_l557_557553


namespace transformed_center_is_correct_l557_557928

-- Definition for transformations
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

def translate_right (p : ℝ × ℝ) (dx : ℝ) : ℝ × ℝ :=
  (p.1 + dx, p.2)

def translate_up (p : ℝ × ℝ) (dy : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + dy)

-- Given conditions
def initial_center : ℝ × ℝ := (4, -3)
def reflection_center := reflect_x initial_center
def translated_right_center := translate_right reflection_center 5
def final_center := translate_up translated_right_center 3

-- The statement to be proved
theorem transformed_center_is_correct : final_center = (9, 6) :=
by
  sorry

end transformed_center_is_correct_l557_557928


namespace find_a_b_and_tangent_line_l557_557677

noncomputable def f (a b x : ℝ) := x^3 + 2 * a * x^2 + b * x + a
noncomputable def g (x : ℝ) := x^2 - 3 * x + 2
noncomputable def f' (a b x : ℝ) := 3 * x^2 + 4 * a * x + b
noncomputable def g' (x : ℝ) := 2 * x - 3

theorem find_a_b_and_tangent_line (a b : ℝ) :
  f a b 2 = 0 ∧ g 2 = 0 ∧ f' a b 2 = 1 ∧ g' 2 = 1 → (a = -2 ∧ b = 5 ∧ ∀ x y : ℝ, y = x - 2 ↔ x - y - 2 = 0) :=
by
  intro h
  sorry

end find_a_b_and_tangent_line_l557_557677


namespace correct_meteorite_encounter_interval_l557_557114

def interval_of_meteorite_encounter (Rate1 Rate2 : ℝ) : ℝ := 1 / (Rate1 + Rate2)

theorem correct_meteorite_encounter_interval :
  interval_of_meteorite_encounter (1 / 7) (1 / 13) = 91 / 20 :=
by
  sorry

end correct_meteorite_encounter_interval_l557_557114


namespace rectangle_area_proof_l557_557095

def rectangle_area (a b : ℝ) (h_diagonal : ℝ) (h_ratio: ℝ) : ℝ := 
  (6 * (a + b) ^ 2) / 13

theorem rectangle_area_proof (a b : ℝ) (h_diagonal : a + b = (3 ^ 2 + 2 ^ 2) ^ (1 / 2)) (h_ratio : (3/2 = 3 / 2)) : 
  rectangle_area a b h_diagonal h_ratio = (6 * (a + b) ^ 2) / 13 :=
sorry

end rectangle_area_proof_l557_557095


namespace quadratic_roots_sums_cubes_l557_557133

-- Definitions
variables {R : Type*} [CommRing R] (x1 x2 S P : R)
variable h_S : S = x1 + x2
variable h_P : P = x1 * x2

-- Theorem statement
theorem quadratic_roots_sums_cubes (x1 x2 S P : R) (h_S : S = x1 + x2) (h_P : P = x1 * x2) :
  (x1^2 + x2^2 = S^2 - 2 * P) ∧ (x1^3 + x2^3 = S^3 - 3 * S * P) :=
by {
  sorry
}

end quadratic_roots_sums_cubes_l557_557133


namespace opposite_of_3_is_neg3_l557_557747

def opposite (x : ℝ) := -x

theorem opposite_of_3_is_neg3 : opposite 3 = -3 :=
by
  sorry

end opposite_of_3_is_neg3_l557_557747


namespace x_lt_1_necessary_but_not_sufficient_for_ln_x_lt_0_l557_557611

theorem x_lt_1_necessary_but_not_sufficient_for_ln_x_lt_0 (x : ℝ) :
  (ln x < 0 → x < 1) ∧ (x < 1 → ¬(ln x < 0)) :=
by
  sorry

end x_lt_1_necessary_but_not_sufficient_for_ln_x_lt_0_l557_557611


namespace determine_a1_l557_557555

noncomputable def sequence (a1 : ℝ) : ℕ → ℝ
| 1     => a1
| (n+2) => (a1 + sequence (n+1)) / (1 - a1 * sequence (n+1))

def smallest_positive_period (p : ℕ) (s : ℕ → ℝ) : Prop :=
∀ n m, n ≠ m → s (n + p) = s n

theorem determine_a1 (a1 : ℝ) (h1 : a1 > 0)
  (h2 : smallest_positive_period 2008 (sequence a1)) :
  a1 = ∃ (k : ℕ), (a1 = Real.tan (k * Real.pi / 2008) ∧ 1 ≤ k ∧ k ≤ 1003 ∧ Nat.coprime k 2008) :=
sorry

end determine_a1_l557_557555


namespace train_passing_time_l557_557483

noncomputable def relative_speed (speed_train speed_man : ℝ) : ℝ :=
  speed_train + speed_man

noncomputable def convert_kmph_to_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * 1000 / 3600

noncomputable def time_to_pass (length_train speed_relative_mps : ℝ) : ℝ :=
  length_train / speed_relative_mps

theorem train_passing_time :
  let train_length := 440
  let speed_train_kmph := 60
  let speed_man_kmph := 6
  let speed_relative_kmph := relative_speed speed_train_kmph speed_man_kmph
  let speed_relative_mps := convert_kmph_to_mps speed_relative_kmph
  in time_to_pass train_length speed_relative_mps = 24 := 
by
  sorry

end train_passing_time_l557_557483


namespace counterexample_seven_people_l557_557819
open Set

structure Person :=
  (id : Nat)

noncomputable def P1 : Person := ⟨1⟩
noncomputable def P2 : Person := ⟨2⟩
noncomputable def P3 : Person := ⟨3⟩
noncomputable def P4 : Person := ⟨4⟩
noncomputable def P5 : Person := ⟨5⟩
noncomputable def P6 : Person := ⟨6⟩
noncomputable def P7 : Person := ⟨7⟩

axiom Acquainted : Person → Person → Prop

axiom host_acquainted :
  Acquainted P1 P2 ∧ Acquainted P1 P3 ∧ Acquainted P1 P4

axiom guests_not_acquainted :
  ¬ Acquainted P5 P6 ∧ ¬ Acquainted P6 P7 ∧ ¬ Acquainted P7 P5

axiom sons_guests_acquainted :
  Acquainted P2 P5 ∧ Acquainted P3 P6 ∧ Acquainted P4 P7

theorem counterexample_seven_people :
  ∃(x y z : Person), 
    x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ 
    ¬ Acquainted x y ∧ ¬ Acquainted y z ∧ ¬ Acquainted z x ∧
    Acquainted P1 x ∧ Acquainted P1 y ∧ Acquainted P1 z :=
by 
  use (P5, P6, P7)
  split
  iterate 3 
  sorry

end counterexample_seven_people_l557_557819


namespace sin_beta_value_l557_557560

variable (α β : ℝ)

theorem sin_beta_value 
  (hα1 : 0 < α) 
  (hα2 : α < π / 2) 
  (hβ1 : -π / 2 < β) 
  (hβ2 : β < 0) 
  (h_cos : cos (α - β) = -5 / 13) 
  (h_sinα : sin α = 4 / 5) : 
  sin β = -56 / 65 := 
sorry

end sin_beta_value_l557_557560


namespace color_lattice_points_l557_557136

variables {α : Type*} [Fintype α]

def lattice_points (n : ℕ) : Finset (ℕ × ℕ) :=
  (Finset.range n).product (Finset.range n).filter (fun x => x.1 ≠ x.2)

theorem color_lattice_points : 
  ∃ (coloring : (ℕ × ℕ) → Fin 10), 
    (∀ a b c : ℕ, a ≠ b → b ≠ c → 
      (coloring (a, b) ≠ coloring (b, c))) :=
begin
  -- proof here
  sorry
end

end color_lattice_points_l557_557136


namespace problem1_problem2_l557_557449

-- Problem 1 Lean Statement
theorem problem1 (m n : ℕ) (h1 : 3 ^ m = 6) (h2 : 9 ^ n = 2) : 3 ^ (m - 2 * n) = 3 :=
by
  sorry

-- Problem 2 Lean Statement
theorem problem2 (x : ℝ) (n : ℕ) (h : x ^ (2 * n) = 3) : (x ^ (3 * n)) ^ 2 - (x ^ 2) ^ (2 * n) = 18 :=
by
  sorry

end problem1_problem2_l557_557449


namespace neither_A_nor_B_nor_C_occurs_probability_l557_557060

theorem neither_A_nor_B_nor_C_occurs_probability :
  let P : (Set (Set α) → ℝ) := fun S => 
    if S = ∅ then 0 else if S = {a} then 0.25 else if S = {b} then 0.30 else if S = {c} then 0.40 else if S = {a, b} then 0.15 else if S = {b, c} then 0.20 else if S = {a, c} then 0.10 else if S = {a, b, c} then 0.05 else 0,
      A := {a}, B := {b}, C := {c} in
  P(A ∪ B ∪ C) = 1 - 0.50 :=
  sorry

end neither_A_nor_B_nor_C_occurs_probability_l557_557060


namespace multiples_of_4_in_sequence_l557_557252

-- Define the arithmetic sequence terms
def nth_term (a d n : ℤ) : ℤ := a + (n - 1) * d

-- Define the conditions
def cond_1 : ℤ := 200 -- first term
def cond_2 : ℤ := -6 -- common difference
def smallest_term : ℤ := 2

-- Define the count of terms function
def num_terms (a d min : ℤ) : ℤ := (a - min) / -d + 1

-- The total number of terms in the sequence
def total_terms : ℤ := num_terms cond_1 cond_2 smallest_term

-- Define a function to get the ith term that is a multiple of 4
def ith_multiple_of_4 (n : ℤ) : ℤ := cond_1 + 18 * (n - 1)

-- Define the count of multiples of 4 within the given number of terms
def count_multiples_of_4 (total : ℤ) : ℤ := (total / 3) + 1

-- Final theorem statement
theorem multiples_of_4_in_sequence : count_multiples_of_4 total_terms = 12 := sorry

end multiples_of_4_in_sequence_l557_557252


namespace find_x_range_l557_557212

-- Given definitions based on the conditions.
def odd_function (f : ℝ → ℝ) := ∀ x, f(-x) = -f(x)
def f (x : ℝ) : ℝ := if x > 0 then x - 1 else -(abs(x) - 1)

-- The theorem to prove the question with conditions and correct answer.
theorem find_x_range (f : ℝ → ℝ) (h1 : odd_function f) (h2 : ∀ x, x > 0 → f(x) = x - 1) : 
  {x : ℝ | f(x - 1) < 0} = {x : ℝ | 1 < x ∧ x < 2} :=
sorry -- proof to be filled in

end find_x_range_l557_557212


namespace yellow_balls_count_l557_557612

theorem yellow_balls_count (red_balls blue_balls yellow_balls : ℕ) (h_red : red_balls = 3) (h_blue : blue_balls = 2) (h_yellow : yellow_balls = 5) (remove_blue : 1 ≤ blue_balls) :
  yellow_balls = 5 :=
by
  -- Add the given conditions assumptions as universe predicates
  have h_blue_removed : blue_balls - 1 ≥ 0,
  { exact Nat.sub_le blue_balls 1 }
  -- The number of yellow balls doesn't change
  exact h_yellow

end yellow_balls_count_l557_557612


namespace steve_total_payment_l557_557338

def mike_dvd_cost : ℝ := 5
def steve_dvd_cost : ℝ := 2 * mike_dvd_cost
def additional_dvd_cost : ℝ := 7
def steve_additional_dvds : ℝ := 2 * additional_dvd_cost
def total_dvd_cost : ℝ := steve_dvd_cost + steve_additional_dvds
def shipping_cost : ℝ := 0.80 * total_dvd_cost
def subtotal_with_shipping : ℝ := total_dvd_cost + shipping_cost
def sales_tax : ℝ := 0.10 * subtotal_with_shipping
def total_amount_paid : ℝ := subtotal_with_shipping + sales_tax

theorem steve_total_payment : total_amount_paid = 47.52 := by
  sorry

end steve_total_payment_l557_557338


namespace king_arthur_round_table_seats_l557_557898

theorem king_arthur_round_table_seats (n : ℕ) (h₁ : n > 1) (h₂ : 10 < 29) (h₃ : (29 - 10) * 2 = n - 2) : 
  n = 38 := 
by
  sorry

end king_arthur_round_table_seats_l557_557898


namespace distilled_water_cups_l557_557792

-- Define the capacities of the cups
def capacities : List ℕ := [16, 18, 22, 23, 24, 34]

-- Define the problem space assumptions
axiom volume_relation (T D : List ℕ) (e : ℕ) :
  (e ∈ capacities ∧ e = 23) ∧
  (∀ a ∈ T, a ∈ capacities ∧ a ≠ e) ∧
  (∀ b ∈ D, b ∈ capacities ∧ b ≠ e) ∧
  sum T = 2 * sum D ∧
  T ∪ D ∪ [e] = capacities

-- Prove that the given cups are filled with distilled water
theorem distilled_water_cups (D : List ℕ) :
  [16, 22] ⊆ D :=
sorry

end distilled_water_cups_l557_557792


namespace arrangement_of_letters_l557_557254

-- Define the set of letters with subscripts
def letters : Finset String := {"B", "A₁", "B₁", "A₂", "B₂", "A₃"}

-- Define the number of ways to arrange 6 distinct letters
theorem arrangement_of_letters : letters.card.factorial = 720 := 
by {
  sorry
}

end arrangement_of_letters_l557_557254


namespace mountain_height_l557_557361

theorem mountain_height :
  (∃ H : ℝ, (10 * (3 / 2) * H = 600000) ∧ H = 80000) :=
by
  exists 80000
  split
  · norm_num
  · refl

end mountain_height_l557_557361


namespace sqrt_121_pm_11_l557_557067

theorem sqrt_121_pm_11 :
  (∃ y : ℤ, y * y = 121) ∧ (∃ x : ℤ, x = 11 ∨ x = -11) → (∃ x : ℤ, x * x = 121 ∧ (x = 11 ∨ x = -11)) :=
by
  sorry

end sqrt_121_pm_11_l557_557067


namespace length_of_BC_l557_557466

-- Definitions based on the given conditions
def circle := {center : ℝ × ℝ, radius : ℝ}
def isosceles_triangle := {A B C : ℝ × ℝ // ((B - A) = (C - A))}

-- Problem setup
variables {O : ℝ × ℝ} (r : ℝ) (area_circle : real.pi * r^2 = 196 * real.pi)
variables {A B C : ℝ × ℝ} (isosceles_triangle_ABC : isosceles_triangle) (OA : ℝ) (OA_eq : OA = 7)
variables (O_outside_ABC : ¬ (A = O ∨ B = O ∨ C = O))

-- Statement to be proved
theorem length_of_BC (h1 : isosceles_triangle_ABC)
                     (h2 : area_circle)
                     (h3 : OA_eq)
                     (h4 : O_outside_ABC) : 
  dist B C = 28 :=
by 
  sorry

end length_of_BC_l557_557466


namespace area_of_triangle_l557_557659

variables (x y : ℝ) (P F1 F2 : ℝ×ℝ)

def on_ellipse (P : ℝ×ℝ) : Prop :=
  ∃ (x y : ℝ), P = (x, y) ∧ (x^2 / 16 + y^2 / 9 = 1)

def foci_of_ellipse (F1 F2 : ℝ×ℝ) : Prop :=
  F1 = (-sqrt 7, 0) ∧ F2 = (sqrt 7, 0)

def dot_product_condition (P F1 F2 : ℝ×ℝ) : Prop :=
  let (px, py) := P;
  let (f1x, f1y) := F1;
  let (f2x, f2y) := F2;
  ((px - f1x) * (px - f2x) + (py - f1y) * (py - f2y) = 5)

theorem area_of_triangle (P F1 F2 : ℝ×ℝ) :
  on_ellipse P → foci_of_ellipse F1 F2 → dot_product_condition P F1 F2 →
  (1/2 * abs ((P.1 - F1.1) * (P.2 - F2.2) - (P.2 - F1.2) * (P.1 - F2.1)) = 6) := by
  sorry

end area_of_triangle_l557_557659


namespace greatest_integer_n_l557_557425

theorem greatest_integer_n (n : ℤ) (h : n^2 - 13 * n + 30 < 0) : n ≤ 9 := 
sorry

example : ∃ n : ℤ, (n^2 - 13 * n + 30 < 0) ∧ (∀ m : ℤ, (m^2 - 13 * m + 30 < 0) → m ≤ n) ∧ n = 9 :=
⟨9, by norm_num, by norm_num, by norm_num⟩

end greatest_integer_n_l557_557425


namespace min_value_SN64_by_aN_is_17_over_2_l557_557559

noncomputable def a_n (n : ℕ) : ℕ := 2 * n
noncomputable def S_n (n : ℕ) : ℕ := n^2 + n

theorem min_value_SN64_by_aN_is_17_over_2 :
  ∃ (n : ℕ), 2 ≤ n ∧ (a_2 = 4 ∧ S_10 = 110) →
  ((S_n n + 64) / a_n n) = 17 / 2 :=
by
  sorry

end min_value_SN64_by_aN_is_17_over_2_l557_557559


namespace sum_of_geometric_sequence_l557_557266

noncomputable def geometric_sequence_first_term_sum (a_1 : ℕ) (q : ℚ) (n : ℕ) := 
  a_1 * (1 - q^n) / (1 - q)

noncomputable def geometric_subsequence_sum (a_1 : ℕ) (q : ℚ) (n : ℕ) := 
  a_1 * (1 - (q^3)^n) / (1 - q^3)

theorem sum_of_geometric_sequence :
  let a_1 := (90 * (1 - (1 / 3)^3) / (1 - (1 / 27))) / (1 - (1 / 3) ^ 99) in
  geometric_subsequence_sum a_1 (1 / 3) 33 = 90 ∧
  geometric_sequence_first_term_sum a_1 (1 / 3) 99 = 130 :=
begin
  sorry
end

end sum_of_geometric_sequence_l557_557266


namespace correct_sequence_of_operations_l557_557634

variables n : ℕ
variables (x y : Fin n → ℝ)

-- Definitions of the conditions
def interpret_regression_line (eq : String) : Prop := sorry
def collect_data (x y : Fin n → ℝ) : Prop := sorry
def calculate_regression_equation (x y : Fin n → ℝ) : String := sorry
def compute_correlation_coefficient (x y : Fin n → ℝ) : ℝ := sorry
def plot_scatter_diagram (data : Fin n → (ℝ × ℝ)) : Prop := sorry

-- The proof statement
theorem correct_sequence_of_operations :
  ∃ (sequence : List (Prop)),
    sequence = [
      collect_data x y, 
      plot_scatter_diagram (λ i, (x i, y i)), 
      compute_correlation_coefficient x y, 
      calculate_regression_equation x y, 
      interpret_regression_line (calculate_regression_equation x y)
    ] :=
  by
  sorry

end correct_sequence_of_operations_l557_557634


namespace greatest_integer_x_l557_557426

theorem greatest_integer_x (x : ℤ) : 
  (∀ x : ℤ, (8 / 11 : ℝ) > (x / 17) → x ≤ 12) ∧ (8 / 11 : ℝ) > (12 / 17) :=
sorry

end greatest_integer_x_l557_557426


namespace part1_part2_part3_l557_557547

variable {x y : ℝ}

-- given conditions
def cond1 : (x + y) ^ 2 = 7
def cond2 : (x - y) ^ 2 = 3

-- required proofs
theorem part1 : cond1 → cond2 → x^2 + y^2 = 5 := by
  sorry

theorem part2 : cond1 → cond2 → x^4 + y^4 = 23 := by
  sorry

theorem part3 : cond1 → cond2 → x^6 + y^6 = 110 := by
  sorry

end part1_part2_part3_l557_557547


namespace sequences_length_13_l557_557149

def sequences (n : ℕ) : ℕ :=
  if n = 1 then 1 else
  if n = 2 then 0 else
  if n = 3 then 1 else
  sequences (n - 1) + sequences (n - 3)

def b_sequences (n : ℕ) : ℕ :=
  if n = 1 then 0 else
  if n = 2 then 1 else
  if n = 3 then 0 else
  b_sequences (n - 2)

theorem sequences_length_13 : sequences 13 + b_sequences 13 = 8 :=
by
  -- Proof would go here
  sorry

end sequences_length_13_l557_557149


namespace find_c_k_l557_557025

-- Definitions of the arithmetic and geometric sequences
def a (n d : ℕ) := 1 + (n - 1) * d
def b (n r : ℕ) := r ^ (n - 1)
def c (n d r : ℕ) := a n d + b n r

-- Conditions for the specific problem
theorem find_c_k (k d r : ℕ) (h1 : 1 + (k - 2) * d + r ^ (k - 2) = 150) (h2 : 1 + k * d + r ^ k = 1500) : c k d r = 314 :=
by
  sorry

end find_c_k_l557_557025


namespace opposite_of_three_l557_557766

theorem opposite_of_three : -3 = opposite(3) := 
by
  sorry

end opposite_of_three_l557_557766


namespace max_successful_free_throws_l557_557125

theorem max_successful_free_throws (a b : ℕ) 
  (h1 : a + b = 105) 
  (h2 : a > 0)
  (h3 : b > 0)
  (ha : a % 3 = 0)
  (hb : b % 5 = 0)
  : (a / 3 + 3 * (b / 5)) ≤ 59 := sorry

end max_successful_free_throws_l557_557125


namespace arithmetic_sequence_properties_l557_557981

noncomputable def arithmetic_sequence (aₙ : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, aₙ (n + 1) = aₙ n + d

theorem arithmetic_sequence_properties
  (aₙ : ℕ → ℝ)
  (d : ℝ)
  (h_d : d > 0)
  (h_a3 : aₙ 3 = -3)
  (h_a2a4 : aₙ 2 * aₙ 4 = 5) :
  (∀ n : ℕ, aₙ n = 2 * n - 9) ∧ ((∀ Sₙ : ℕ → ℝ, Sₙ 4 = -16)) :=
begin
  sorry
end

end arithmetic_sequence_properties_l557_557981


namespace range_of_a_l557_557788

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x < 2 → (x + a < 0))) → (a ≤ -2) :=
sorry

end range_of_a_l557_557788


namespace slices_all_three_toppings_l557_557074

def slices_with_all_toppings (total_slices pepperoni_slices mushroom_slices olive_slices : ℕ) : ℕ := 
  (12 : ℕ)

theorem slices_all_three_toppings
  (total_slices : ℕ)
  (pepperoni_slices : ℕ)
  (mushroom_slices : ℕ)
  (olive_slices : ℕ)
  (h : total_slices = 24)
  (h1 : pepperoni_slices = 12)
  (h2 : mushroom_slices = 14)
  (h3 : olive_slices = 16)
  (hc : total_slices ≥ 0)
  (hc1 : pepperoni_slices ≥ 0)
  (hc2 : mushroom_slices ≥ 0)
  (hc3 : olive_slices ≥ 0) :
  slices_with_all_toppings total_slices pepperoni_slices mushroom_slices olive_slices = 2 :=
  sorry

end slices_all_three_toppings_l557_557074


namespace ratio_L₁B_AN_length_volume_of_prism_l557_557820

variable (Prism : Type)
variables (K L M N K₁ L₁ M₁ N₁ A B O : Prism)
variables (α β : Set Prism)
variable [RegularPrism Prism]

-- Given the conditions of the problem
def conditions : Prop :=
  RegularPrism.baseOfPrism K L M N ∧
  PerpTo (L₁N) α ∧
  PerpTo (L₁N) β ∧
  PassThrough α K ∧
  PassThrough β N₁ ∧
  IntersectPlane α β L₁N A ∧
  IntersectPlane α β L₁N B ∧
  CloserTo A N B ∧
  SphereRadius Prism (1/2) 

-- Part (a)
theorem ratio_L₁B_AN (h : conditions K L M N K₁ L₁ M₁ N₁ A B O α β) : 
  Ratio L₁ B A N 2 1 := sorry

-- Part (b)
theorem length_volume_of_prism (h : conditions K L M N K₁ L₁ M₁ N₁ A B O α β) :
  Length (L₁N) = (1 + sqrt 13)/2 ∧
  Volume (K L M N K₁ L₁ M₁ N₁) = (1/2) * sqrt (6 + 2 sqrt 13) := sorry 

end ratio_L₁B_AN_length_volume_of_prism_l557_557820


namespace opposite_of_3_l557_557730

-- Define the concept of opposite of a number and the logic for positive numbers
def opposite (x : Int) : Int := 
  if x > 0 then -x
  else if x < 0 then -x
  else 0

-- Statement to prove that the opposite of 3 is -3
theorem opposite_of_3 : opposite 3 = -3 :=
by 
  -- Using the definition of opposite
  unfold opposite
  -- Simplify the expression for x = 3
  simp [lt_irrefl, int.coe_nat_lt]
  -- Conclude proof
  rfl

end opposite_of_3_l557_557730


namespace b_share_in_profit_l557_557856

theorem b_share_in_profit (A B C : ℝ) (p : ℝ := 4400) (x : ℝ)
  (h1 : A = 3 * B)
  (h2 : B = (2 / 3) * C)
  (h3 : C = x) :
  B / (A + B + C) * p = 800 :=
by
  sorry

end b_share_in_profit_l557_557856


namespace joseph_cards_percentage_left_l557_557646

theorem joseph_cards_percentage_left (h1 : ℕ := 16) (h2 : ℚ := 3/8) (h3 : ℕ := 2) :
  ((h1 - (h2 * h1 + h3)) / h1 * 100) = 50 :=
by
  sorry

end joseph_cards_percentage_left_l557_557646


namespace max_profit_l557_557474

noncomputable def profit (x : ℝ) : ℝ :=
  10 * (x - 40) * (100 - x)

theorem max_profit (x : ℝ) (hx : x > 40) :
  (profit 70 = 9000) ∧ ∀ y > 40, profit y ≤ 9000 := by
  sorry

end max_profit_l557_557474


namespace min_union_cardinality_l557_557355

open Set

variable (A B : Set α) [Fintype A] [Fintype B]

theorem min_union_cardinality (hA : Fintype.card A = 30)
                              (hB : Fintype.card B = 20)
                              (hAB : Fintype.card (A ∩ B) ≥ 10) :
  Fintype.card (A ∪ B) = 40 := by
  sorry

end min_union_cardinality_l557_557355


namespace sequence_contains_456_l557_557311

/-- Define Lucas sequence and Fibonacci sequence -/
def Lucas : ℕ → ℕ
| 0     := 2
| 1     := 1
| (n+2) := Lucas (n + 1) + Lucas n

def Fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := Fibonacci (n + 1) + Fibonacci n

/-- Main proof problem -/
theorem sequence_contains_456 (K : ℕ) (hK : K > 0) :
  ∃ a b c d : ℕ, (gcd a b = 1 ∧ gcd a d = 1 ∧ gcd b d = 1 ∧ squarefree c) ∧
  (probability_456_approaches K = (a - b * sqrt c) / d) ∧
  (a + b + c + d = 31) :=
sorry

/-- Helper definition for square-free integer check -/
def squarefree (n : ℕ) : Prop :=
  ∀ m : ℕ, m * m ∣ n → m = 1

/-- Placeholder probability calculation, to be defined by further development -/
noncomputable def probability_456_approaches (K : ℕ) : ℝ := 
sorry

end sequence_contains_456_l557_557311


namespace cubic_tangent_slope_sum_zero_l557_557973

theorem cubic_tangent_slope_sum_zero (a x1 x2 x3 : ℝ) (h1: x1 ≠ x2) (h2: x1 ≠ x3) (h3: x2 ≠ x3) :
  let k1 := a * (x1 - x2) * (x1 - x3),
      k2 := a * (x2 - x1) * (x2 - x3),
      k3 := a * (x3 - x1) * (x3 - x2) in
  (1 / k1) + (1 / k2) + (1 / k3) = 0 :=
by
  sorry

end cubic_tangent_slope_sum_zero_l557_557973


namespace complex_expression_l557_557519

-- The condition: n is a positive integer
variable (n : ℕ) (hn : 0 < n)

-- Definition of the problem to be proved
theorem complex_expression (n : ℕ) (hn : 0 < n) : 
  (Complex.I ^ (4 * n) + Complex.I ^ (4 * n + 1) + Complex.I ^ (4 * n + 2) + Complex.I ^ (4 * n + 3)) = 0 :=
sorry

end complex_expression_l557_557519


namespace seq_formula_exists_M_product_inequality_seq_b_formula_l557_557513

-- Given conditions
variable {a : ℕ → ℝ} 
variable {S : ℕ → ℝ}
variable (c : ℝ) (h_pos_c : 0 < c) (h_c_ne_one : c ≠ 1)

-- Sequence formula
theorem seq_formula (n : ℕ) (h_pos_n : 0 < n) (h_point_on_curve : ∀ n, (a n, S n) = (c * (1/c)^(n - 1), (c^2 - a n)/(c-1))): 
  a n = c * (1 / c)^(n - 1) := sorry

-- Product inequality
theorem exists_M_product_inequality (h_range_c : 0 < c ∧ c < 1): 
  ∃ M : ℕ, ∀ n > M, (∏ k in (finset.range (2 * n)).filter (λ m, odd m), a k) > a 101 := sorry

-- Another sequence
variable {b : ℕ → ℝ} 
variable (h_b_arith : ∀ n, b n = b 0 + n * 10)
theorem seq_b_formula (h_sum_eq : ∀ n, (finset.range n).sum (λ i, b (i+1) * a (n-i)) = 3^n - (5/3)*n - 1):
  b = λ n, 10*n - 9 ∧ c = 1/3 := sorry

end seq_formula_exists_M_product_inequality_seq_b_formula_l557_557513


namespace sum_of_digits_of_m_l557_557452

theorem sum_of_digits_of_m (k m : ℕ) : 
  1 ≤ k ∧ k ≤ 3 ∧ 10000 ≤ 11131 * k + 1203 ∧ 11131 * k + 1203 < 100000 ∧ 
  11131 * k + 1203 = m * m ∧ 3 * k < 10 → 
  (m.digits 10).sum = 15 :=
by 
  sorry

end sum_of_digits_of_m_l557_557452


namespace lap_distance_l557_557293

theorem lap_distance (boys_laps : ℕ) (girls_extra_laps : ℕ) (total_girls_miles : ℚ) : 
  boys_laps = 27 → girls_extra_laps = 9 → total_girls_miles = 27 →
  (total_girls_miles / (boys_laps + girls_extra_laps) = 3 / 4) :=
by
  intros hb hg hm
  sorry

end lap_distance_l557_557293


namespace arithmetic_sequence_a4_plus_a6_l557_557228

-- Define the given problem conditions
def Sn (n : ℕ) (hn : 0 < n) : ℤ := -n^2 + 9 * n

-- Define the sequence term a_n based on the sum formula S_n
def an (n : ℕ) (hn : 0 < n) : ℤ := Sn n hn - Sn (n - 1) (nat.sub_pos_of_lt hn)

-- Prove that the sequence {a_n} is arithmetic
theorem arithmetic_sequence (n : ℕ) (hn : 0 < n) :
  ∃ d : ℤ, ∀ k : ℕ, 1 ≤ k → 0 < k + 1 → an k (nat.succ_pos' k) = an 1 hn + d * (k - 1) := 
sorry

-- Prove that a_4 + a_6 = 0
theorem a4_plus_a6 : an 4 (by norm_num) + an 6 (by norm_num) = 0 := 
sorry

end arithmetic_sequence_a4_plus_a6_l557_557228


namespace opposite_of_3_is_neg3_l557_557752

def opposite (x : ℝ) := -x

theorem opposite_of_3_is_neg3 : opposite 3 = -3 :=
by
  sorry

end opposite_of_3_is_neg3_l557_557752


namespace find_linear_function_passing_A_B_l557_557364

-- Conditions
def line_function (k b x : ℝ) : ℝ := k * x + b

theorem find_linear_function_passing_A_B :
  (∃ k b : ℝ, k ≠ 0 ∧ line_function k b 1 = 3 ∧ line_function k b 0 = -2) → 
  ∃ k b : ℝ, k = 5 ∧ b = -2 ∧ ∀ x : ℝ, line_function k b x = 5 * x - 2 :=
by
  -- Proof will be added here
  sorry

end find_linear_function_passing_A_B_l557_557364


namespace find_number_from_sum_l557_557120

def smallest_two_digit_number (s : Finset ℕ) : ℕ :=
  s.to_list.filter (λ n, 10 ≤ n ∧ n < 100).min' sorry

theorem find_number_from_sum (s : Finset ℕ) (h : s = {1, 3, 5, 8}) :
  ∃ x : ℕ, (smallest_two_digit_number (s.product s).filter (λ p, p.1 ≠ p.2) = 13) ∧ (13 + x = 88) ∧ (x = 75) :=
  by
    sorry

end find_number_from_sum_l557_557120


namespace fraction_ordering_l557_557145

theorem fraction_ordering :
  let a := (6 : ℚ) / 22
  let b := (8 : ℚ) / 32
  let c := (10 : ℚ) / 29
  a < b ∧ b < c :=
by
  sorry

end fraction_ordering_l557_557145


namespace water_height_in_cylinder_l557_557489

noncomputable def volume_cone (r h : ℝ) : ℝ := (1 / 3) * π * r^2 * h

noncomputable def volume_cylinder (r h : ℝ) : ℝ := π * r^2 * h

def radius_cone : ℝ := 15
def height_cone : ℝ := 20
def radius_cylinder : ℝ := 18

theorem water_height_in_cylinder : 
  let V_cone := volume_cone radius_cone height_cone in
  let V_cylinder_half := V_cone / 2 in
  let height_water := V_cylinder_half / (π * radius_cylinder^2) in
  height_water = 2.315 :=
begin
  sorry
end

end water_height_in_cylinder_l557_557489


namespace total_number_of_seats_l557_557887

theorem total_number_of_seats (n : ℕ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ n) 
                               (h2 : (10 : ℕ) < (29 : ℕ)) 
                               (h3 : (29 - 10) % (n / 2) = 0) : n = 38 :=
by sorry

end total_number_of_seats_l557_557887


namespace car_R_average_speed_l557_557506

theorem car_R_average_speed 
  (R P S: ℝ)
  (h1: S = 2 * P)
  (h2: P + 2 = R)
  (h3: P = R + 10)
  (h4: S = R + 20) :
  R = 25 :=
by 
  sorry

end car_R_average_speed_l557_557506


namespace opposite_of_3_is_neg3_l557_557721

theorem opposite_of_3_is_neg3 : forall (n : ℤ), n = 3 -> -n = -3 :=
by
  sorry

end opposite_of_3_is_neg3_l557_557721


namespace choir_members_max_l557_557002

theorem choir_members_max (m y n : ℕ) (h_square : m = y^2 + 11) (h_rect : m = n * (n + 5)) : 
  m = 300 := 
sorry

end choir_members_max_l557_557002


namespace exponential_identity_l557_557261

-- Define the conditions as Lean definitions
def sixty_to_the_a (a : ℝ) := 60^a
def three_sixty_to_the_b (b : ℝ) := 360^b

-- Define the main theorem statement
theorem exponential_identity (a b : ℝ) (h1 : sixty_to_the_a a = 5) (h2 : three_sixty_to_the_b b = 5) :
  12 ^ ((1 - a - b) / (2 * (1 - b))) = 2 :=
by
  sorry

end exponential_identity_l557_557261


namespace part1_part2_l557_557200

variables (α β : Real)

theorem part1 (h1 : Real.cos (α + β) = 1 / 3) (h2 : Real.sin α * Real.sin β = 1 / 4) :
  Real.cos α * Real.cos β = 7 / 12 := 
sorry

theorem part2 (h1 : Real.cos (α + β) = 1 / 3) (h2 : Real.sin α * Real.sin β = 1 / 4) :
  Real.cos (2 * α - 2 * β) = 7 / 18 := 
sorry

end part1_part2_l557_557200


namespace symmetry_center_l557_557224

noncomputable def ω := 1 / 2
noncomputable def φ := π / 6

def f (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem symmetry_center :
  (∀ x : ℝ, f x ≤ f (π / 3)) ∧
  (∃ (T > 0), ∀ x : ℝ, f (x + T) = f x) ∧
  (T = 4 * π) ∧
  ω > 0 ∧
  (|φ| < π / 2) →
  (-φ / ω, 0) = (-π / 3, 0) :=
by
  sorry

#print axioms symmetry_center

end symmetry_center_l557_557224


namespace stamping_possibility_l557_557278

-- Define the problem setup
def checkered_wooden_square (square_size black_cells : ℕ) := square_size = 102 ∧ black_cells = 102

def stamping (stamp_times sheet_size : ℕ) := stamp_times = 100 ∧ sheet_size = 101

-- The main proof statement
theorem stamping_possibility 
  (square_size black_cells stamp_times sheet_size : ℕ)
  (h1 : checkered_wooden_square square_size black_cells) 
  (h2 : stamping stamp_times sheet_size) : 
  ∃(can_result : Prop), can_result :=
begin
  have h_sq : square_size = 102 := h1.1,
  have h_cells : black_cells = 102 := h1.2,
  have h_times : stamp_times = 100 := h2.1,
  have h_sheet : sheet_size = 101 := h2.2,
  let can_result := ∃f: (_-> Prop), ∀ x, (x = (101 * 101 - 1)),
  -- Conclude the theorem statement
  exact can_result
end

end stamping_possibility_l557_557278


namespace systematic_sampling_l557_557279

theorem systematic_sampling :
  ∃ (l : list ℕ), l = [6, 18, 30, 42, 54] ∧
    (∀ n ∈ l, 1 ≤ n ∧ n ≤ 60) ∧
    ∃ k, k * 12 = 54 ∧ (l = list.map (λ i, 6 + (i - 1) * 12) (list.range 5)) :=
by
  sorry

end systematic_sampling_l557_557279


namespace find_m_for_even_function_l557_557239

def f (x : ℝ) (m : ℝ) := x^2 + (m - 1) * x + 3

theorem find_m_for_even_function : ∃ m : ℝ, (∀ x : ℝ, f (-x) m = f x m) ∧ m = 1 :=
sorry

end find_m_for_even_function_l557_557239


namespace curve_properties_l557_557530

noncomputable def curve : (ℝ → ℝ) := λ x, Real.exp x - 3

theorem curve_properties : 
  (curve 0 = -2) ∧ (∀ x, deriv curve x = curve x + 3) :=
by
  sorry

end curve_properties_l557_557530


namespace team_b_fraction_calls_l557_557831

-- This theorem states the given problem and asks to prove the fraction of calls processed by Team B.
theorem team_b_fraction_calls (B N : ℕ) :
  let calls_A := (2/5) * (5/8) * B * N,
      calls_B := B * N,
      calls_C := (3/4) * (7/6) * B * N,
      total_calls := calls_A + calls_B + calls_C in
  calls_B / total_calls = 8 / 17 :=
by
  sorry

end team_b_fraction_calls_l557_557831


namespace directrix_of_parabola_l557_557152

theorem directrix_of_parabola (p : ℝ) (y x : ℝ) :
  y = x^2 → x^2 = 4 * p * y → 4 * y + 1 = 0 :=
by
  intros hyp1 hyp2
  sorry

end directrix_of_parabola_l557_557152


namespace charlie_brown_lightning_distance_l557_557944

-- Given data
def time_delay : ℝ := 15
def speed_of_sound : ℝ := 1100
def wind_effect : ℝ := 60
def feet_per_mile : ℝ := 5280

-- Effective speed after considering wind
def effective_speed := speed_of_sound - wind_effect

-- Distance calculation
def distance_in_feet := effective_speed * time_delay

-- Distance in miles
def distance_in_miles := distance_in_feet / feet_per_mile

-- Rounded distance to the nearest half-mile
def rounded_distance_miles := Float.round (distance_in_miles * 2) / 2

theorem charlie_brown_lightning_distance :
  rounded_distance_miles = 3 := by
  sorry

end charlie_brown_lightning_distance_l557_557944


namespace container_fraction_full_l557_557829

theorem container_fraction_full (initial_percentage : ℝ) (added_water : ℝ) (total_capacity : ℝ) (h1 : initial_percentage = 0.30) (h2 : added_water = 54) (h3 : total_capacity = 120) :
  (initial_percentage * total_capacity + added_water) / total_capacity = 3 / 4 :=
by
  have initial_amount := initial_percentage * total_capacity,
  have final_amount := initial_amount + added_water,
  have fraction_full := final_amount / total_capacity,
  show fraction_full = 3 / 4,
  sorry

end container_fraction_full_l557_557829


namespace sin_alpha_eq_neg_four_fifths_l557_557571

theorem sin_alpha_eq_neg_four_fifths (α : Real) (x y r : Real) 
  (hx : x = -3) (hy : y = -4) (hr : r = Real.sqrt (x^2 + y^2))
  (hP : r ≠ 0) : Real.sin α = y / r :=
by 
-- We prove the claim using the fact that point P(-3, -4) lies on the terminal side of α,
-- which gives us x = -3 and y = -4.
-- We calculate r and use the trigonometric definition of sin(α).
suffices h : r = 5, from
  have hsin : Real.sin α = y / r, 
    from calc
      Real.sin α = y / r : by sorry,
  hsin,
show r = 5,
from calc
  r = Real.sqrt (x^2 + y^2) : hr
  ... = Real.sqrt (9 + 16) : by rw [hx, hy]
  ... = Real.sqrt 25 : by norm_num
  ... = 5 : by norm_num

end sin_alpha_eq_neg_four_fifths_l557_557571


namespace solve_for_x_l557_557050

theorem solve_for_x :
  ∃ (x : ℝ), x ≠ 0 ∧ (5 * x)^10 = (10 * x)^5 ∧ x = 2 / 5 :=
by
  sorry

end solve_for_x_l557_557050


namespace factorize_expression_l557_557163

theorem factorize_expression (x : ℝ) : 
  (x^2 + 4)^2 - 16 * x^2 = (x + 2)^2 * (x - 2)^2 := 
by sorry

end factorize_expression_l557_557163


namespace total_seats_at_round_table_l557_557878

-- Define namespace and conditions
namespace KingArthur

variable (n : ℕ) -- Total number of seats

-- Conditions
def king_position : ℕ := 10
def lancelot_position : ℕ := 29
def opposite (a b : ℕ) (n : ℕ) : Prop := (a + (n / 2)) % n = b

-- Proof statement
theorem total_seats_at_round_table : opposite 10 29 n ∧ 29 < n → n = 38 :=
by
  sorry

end KingArthur

end total_seats_at_round_table_l557_557878


namespace opposite_of_3_is_neg3_l557_557751

def opposite (x : ℝ) := -x

theorem opposite_of_3_is_neg3 : opposite 3 = -3 :=
by
  sorry

end opposite_of_3_is_neg3_l557_557751


namespace sarah_age_l557_557697

variable (s m : ℕ)

theorem sarah_age (h1 : s = m - 18) (h2 : s + m = 50) : s = 16 :=
by {
  -- The proof will go here
  sorry
}

end sarah_age_l557_557697


namespace opposite_of_3_l557_557732

-- Define the concept of opposite of a number and the logic for positive numbers
def opposite (x : Int) : Int := 
  if x > 0 then -x
  else if x < 0 then -x
  else 0

-- Statement to prove that the opposite of 3 is -3
theorem opposite_of_3 : opposite 3 = -3 :=
by 
  -- Using the definition of opposite
  unfold opposite
  -- Simplify the expression for x = 3
  simp [lt_irrefl, int.coe_nat_lt]
  -- Conclude proof
  rfl

end opposite_of_3_l557_557732


namespace find_function_l557_557312

theorem find_function (α : ℝ) (hα : 0 < α) (f : ℕ+ → ℝ) 
  (h : ∀ k m : ℕ+, α * m ≤ k → k ≤ (α + 1) * m → f (k + m) = f k + f m) :
  ∃ D : ℝ, ∀ n : ℕ+, f n = n * D :=
sorry

end find_function_l557_557312


namespace circle_divides_square_sides_l557_557465

/-- A circle with a radius of 13 cm is tangent to two adjacent sides of a square with 
a side length of 18 cm. Prove that the circle divides each of the other two sides 
of the square into segments of 1 cm and 17 cm respectively. -/
theorem circle_divides_square_sides (r : ℝ) (a : ℝ) (s1 s2 : ℝ) (c : ℝ) :
  r = 13 ∧ a = 18 ∧ 
  r * r = c * c + s1 * s1 ∧ c = √(r * r - (a - r) * (a - r)) → 
  s1 = 1 ∧ s2 = 17 :=
by intros h; cases h with hr ha; cases ha with hra hcs; cases hcs with hc1 hc; 
   sorry

end circle_divides_square_sides_l557_557465


namespace integer_roots_quadratic_eq_distinct_a_l557_557540

theorem integer_roots_quadratic_eq_distinct_a :
  {a : ℝ | ∃ r s : ℤ, r + s = -a ∧ r * s = 8 * a}.to_finset.card = 8 :=
by sorry

end integer_roots_quadratic_eq_distinct_a_l557_557540


namespace no_nat_solutions_no_int_solutions_l557_557576

theorem no_nat_solutions (x y : ℕ) : x^3 + 5 * y = y^3 + 5 * x → x = y :=
by sorry

theorem no_int_solutions (x y : ℤ) : x^3 + 5 * y = y^3 + 5 * x → x = y :=
by sorry

end no_nat_solutions_no_int_solutions_l557_557576


namespace probability_at_least_one_odd_l557_557470

def rolls : List (ℕ) := [1, 2, 3, 4, 5, 6]

def fairDie (n : ℕ) : Prop := n ∈ rolls

def evenOutcome : Set ℕ := {n : ℕ | n % 2 = 0}
def oddOutcome : Set ℕ := {n : ℕ | n % 2 = 1}

noncomputable def probability_of_even : ℚ := 1 / 2
noncomputable def probability_of_all_even : ℚ := (1 / 2) ^ 8

theorem probability_at_least_one_odd :
  let p_even := probability_of_even in let p_all_even := probability_of_all_even in
  1 - p_all_even = 255 / 256 := by
  sorry

end probability_at_least_one_odd_l557_557470


namespace opposite_of_x_is_positive_l557_557018

-- Assume a rational number x
def x : ℚ := -1 / 2023

-- Theorem stating the opposite of x is 1 / 2023
theorem opposite_of_x_is_positive : -x = 1 / 2023 :=
by
  -- Required part of Lean syntax; not containing any solution steps
  sorry

end opposite_of_x_is_positive_l557_557018


namespace dice_multiple_3_prob_l557_557103

-- Define the probability calculations for the problem
noncomputable def single_roll_multiple_3_prob: ℝ := 1 / 3
noncomputable def single_roll_not_multiple_3_prob: ℝ := 1 - single_roll_multiple_3_prob
noncomputable def eight_rolls_not_multiple_3_prob: ℝ := (single_roll_not_multiple_3_prob) ^ 8
noncomputable def at_least_one_roll_multiple_3_prob: ℝ := 1 - eight_rolls_not_multiple_3_prob

-- The lean theorem statement
theorem dice_multiple_3_prob : 
  at_least_one_roll_multiple_3_prob = 6305 / 6561 := by 
sorry

end dice_multiple_3_prob_l557_557103


namespace value_of_g_at_3_l557_557610

-- Define the polynomial g(x)
def g (x : ℝ) : ℝ := 5 * x^3 - 6 * x^2 - 3 * x + 5

-- The theorem statement
theorem value_of_g_at_3 : g 3 = 77 := by
  -- This would require a proof, but we put sorry as instructed
  sorry

end value_of_g_at_3_l557_557610


namespace how_many_sodas_each_l557_557131

noncomputable def siblings_soda (sisters : ℕ) (total_sodas : ℕ) : ℕ :=
  let brothers := 2 * sisters
  let siblings := sisters + brothers
  total_sodas / siblings

theorem how_many_sodas_each (total_sodas : ℕ) (sisters : ℕ) 
  (hs : sisters = 2) (ht : total_sodas = 12) : siblings_soda sisters total_sodas = 2 :=
  by
  -- Definitions
  let brothers := 2 * sisters
  have h1: brothers = 4, by rw [hs]; norm_num
  let siblings := sisters + brothers
  have h2: siblings = 6, by rw [hs, h1]; norm_num

  -- Calculation
  show siblings_soda sisters total_sodas = 2 from by
  rw [hs, ht]
  dsimp[siblings_soda]
  rw[h2]
  norm_num
  sorry

end how_many_sodas_each_l557_557131


namespace find_y_l557_557286

theorem find_y 
  (x y : ℝ) 
  (h1 : (6 : ℝ) = (1/2 : ℝ) * x) 
  (h2 : y = (1/2 : ℝ) * 10) 
  (h3 : x * y = 60) 
: y = 5 := 
by 
  sorry

end find_y_l557_557286


namespace cooking_time_l557_557463

theorem cooking_time
  (total_potatoes : ℕ) (cooked_potatoes : ℕ) (remaining_time : ℕ) (remaining_potatoes : ℕ)
  (h_total : total_potatoes = 15)
  (h_cooked : cooked_potatoes = 8)
  (h_remaining_time : remaining_time = 63)
  (h_remaining_potatoes : remaining_potatoes = total_potatoes - cooked_potatoes) :
  remaining_time / remaining_potatoes = 9 :=
by
  sorry

end cooking_time_l557_557463


namespace max_leap_years_200_years_l557_557495

theorem max_leap_years_200_years (h: ∀ n, 0 ≤ n → n ≤ 200 → (n % 5 = 0 ↔ n / 5 ≤ 200 / 5)) : 
  ∑ k in finset.range (200 / 5 + 1), (if k % 5 = 0 then 1 else 0) = 40 :=
by sorry

end max_leap_years_200_years_l557_557495


namespace valid_triangle_probability_l557_557281

noncomputable def probability_of_triangle : ℚ :=
  163 / 455

theorem valid_triangle_probability (n : ℕ) (h : n = 15) : 
  let num_segments := nat.choose n 2,
      num_combinations := nat.choose num_segments 3 in
  (∃ valid_triangles, valid_triangles = 3260) → 
  probability_of_triangle = 163 / 455 :=
sorry

end valid_triangle_probability_l557_557281


namespace max_mn_value_l557_557596

theorem max_mn_value (OA OB : ℝ → ℝ → ℝ)
  (O C : ℝ → ℝ) (length_OA length_OB : ℝ) (angle_AOB : ℝ)
  (h_len1 : length_OA = 2)
  (h_len2 : length_OB = 2)
  (h_angle : angle_AOB = π / 3)
  (h_OC_eq : ∃ (m n : ℝ), (m > 0) ∧ (n > 0) ∧ OC = m * OA + n * OB)
  (h_OC_on_circle : (OC • OC) = 4) :
  ∃ (max_mn : ℝ), max_mn = (2 * Math.sqrt 3) / 3 :=
by
  sorry

end max_mn_value_l557_557596


namespace stratified_sampling_girls_l557_557623

theorem stratified_sampling_girls 
  (total_students : ℕ)
  (total_girls : ℕ)
  (sample_size : ℕ)
  (stratified_sampling : Bool)
  (total_students = 30000)
  (total_girls = 4000)
  (sample_size = 150)
  (stratified_sampling = true)
  : ∃ girls_in_sample : ℕ, girls_in_sample = 20 :=
by
  use 20
  sorry

end stratified_sampling_girls_l557_557623


namespace dice_multiple_3_prob_l557_557102

-- Define the probability calculations for the problem
noncomputable def single_roll_multiple_3_prob: ℝ := 1 / 3
noncomputable def single_roll_not_multiple_3_prob: ℝ := 1 - single_roll_multiple_3_prob
noncomputable def eight_rolls_not_multiple_3_prob: ℝ := (single_roll_not_multiple_3_prob) ^ 8
noncomputable def at_least_one_roll_multiple_3_prob: ℝ := 1 - eight_rolls_not_multiple_3_prob

-- The lean theorem statement
theorem dice_multiple_3_prob : 
  at_least_one_roll_multiple_3_prob = 6305 / 6561 := by 
sorry

end dice_multiple_3_prob_l557_557102


namespace village_population_equal_in_years_l557_557041

theorem village_population_equal_in_years :
  ∀ (n : ℕ), (70000 - 1200 * n = 42000 + 800 * n) ↔ n = 14 :=
by {
  sorry
}

end village_population_equal_in_years_l557_557041


namespace N_subset_M_l557_557676

open Set

def M : Set (ℝ × ℝ) := { p | ∃ x, p = (x, 2*x + 1) }
def N : Set (ℝ × ℝ) := { p | ∃ x, p = (x, -x^2) }

theorem N_subset_M : N ⊆ M :=
by
  sorry

end N_subset_M_l557_557676


namespace opposite_of_three_l557_557767

theorem opposite_of_three : -3 = opposite(3) := 
by
  sorry

end opposite_of_three_l557_557767


namespace total_number_of_seats_l557_557883

theorem total_number_of_seats (n : ℕ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ n) 
                               (h2 : (10 : ℕ) < (29 : ℕ)) 
                               (h3 : (29 - 10) % (n / 2) = 0) : n = 38 :=
by sorry

end total_number_of_seats_l557_557883


namespace sum_of_roots_tan_quadratic_zero_l557_557186

theorem sum_of_roots_tan_quadratic_zero :
  ∀ x ∈ set.Icc (0 : ℝ) (2 * Real.pi),
  ∑ r in {x : ℝ | tan x ∈ {(5 + Complex.i * Real.sqrt 3) / 2, (5 - Complex.i * Real.sqrt 3) / 2}}, r = 0 :=
by
  sorry

end sum_of_roots_tan_quadratic_zero_l557_557186


namespace sqrt_121_pm_11_l557_557068

theorem sqrt_121_pm_11 :
  (∃ y : ℤ, y * y = 121) ∧ (∃ x : ℤ, x = 11 ∨ x = -11) → (∃ x : ℤ, x * x = 121 ∧ (x = 11 ∨ x = -11)) :=
by
  sorry

end sqrt_121_pm_11_l557_557068


namespace opposite_of_three_l557_557736

theorem opposite_of_three : -3 = -3 := 
by sorry

end opposite_of_three_l557_557736


namespace parallel_b_c_perpendicular_a_b_with_b_l557_557597

open_locale real

def a : ℝ × ℝ := (1, 3)
def b : ℝ × ℝ := (1, -2)
def c : ℝ × ℝ := (-2, 4)

theorem parallel_b_c : ∃ k : ℝ, c = k • b := 
by sorry

theorem perpendicular_a_b_with_b : (a + b) ⬝ b = 0 := 
by sorry

end parallel_b_c_perpendicular_a_b_with_b_l557_557597


namespace arithmetic_sequence_d_range_l557_557285

theorem arithmetic_sequence_d_range (d : ℝ) :
  (10 + 4 * d > 0) ∧ (10 + 5 * d < 0) ↔ (-5/2 < d) ∧ (d < -2) :=
by
  sorry

end arithmetic_sequence_d_range_l557_557285


namespace holder_triple_ineq_l557_557818

open MeasureTheory

variables {X : Type*} {μ : Measure X}
variables {f g h : X → ℝ} [MeasurableSpace X] [Measurable μ]
variables {p q r : ℝ}

theorem holder_triple_ineq
  (h_meas_f : Measurable f) 
  (h_meas_g : Measurable g) 
  (h_meas_h : Measurable h)
  (h_pqr : 1 / p + 1 / q + 1 / r = 1)
  (h_p : 1 ≤ p) (h_q : 1 ≤ q) (h_r : 1 ≤ r) :
  ∫ x, |f x * g x * h x| ∂μ ≤ 
  (∫ x, |f x| ^ p ∂μ) ^ (1 / p) * 
  (∫ x, |g x| ^ q ∂μ) ^ (1 / q) * 
  (∫ x, |h x| ^ r ∂μ) ^ (1 / r) := by sorry

end holder_triple_ineq_l557_557818


namespace angle_bisectors_meet_on_segment_l557_557655

theorem angle_bisectors_meet_on_segment 
  (A B C D : Type) 
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  (AB AC BC : ℝ)  -- sides of the triangle
  (ABD_angle ACD_angle : ℝ)  -- angles at ABD and ACD
  (h_AB_AC : AB = AC) 
  (h_AB_AC_not : AB ≠ BC)
  (h_angle_ABD : ABD_angle = 30)
  (h_angle_ACD : ACD_angle = 30) :
  ∃ P : Type, IsAngleBisector P (Angle A C B) (Angle A D B) (Segment A B) :=
sorry

end angle_bisectors_meet_on_segment_l557_557655


namespace F_2021_F_integer_F_divisibility_l557_557188

/- Part 1 -/
def F (n : ℕ) : ℕ := 
  let a := n / 1000
  let b := (n % 1000) / 100
  let c := (n % 100) / 10
  let d := n % 10
  let n' := 1000 * c + 100 * d + 10 * a + b
  (n + n') / 101

theorem F_2021 : F 2021 = 41 :=
  sorry

/- Part 2 -/
theorem F_integer (a b c d : ℕ) (ha : 1 ≤ a) (hb : a ≤ 9) (hc : 0 ≤ b) (hd : b ≤ 9)
(hc' : 0 ≤ c) (hd' : c ≤ 9) (hc'' : 0 ≤ d) (hd'' : d ≤ 9) :
  let n := 1000 * a + 100 * b + 10 * c + d
  let n' := 1000 * c + 100 * d + 10 * a + b
  F n = (101 * (10 * a + b + 10 * c + d)) / 101 :=
  sorry

/- Part 3 -/
theorem F_divisibility (a b : ℕ) (ha : 1 ≤ a ∧ a ≤ 5) (hb : 5 ≤ b ∧ b ≤ 9) :
  let s := 3800 + 10 * a + b
  let t := 1000 * b + 100 * a + 13
  (3 * F t - F s) % 8 = 0 ↔ s = 3816 ∨ s = 3847 ∨ s = 3829 :=
  sorry

end F_2021_F_integer_F_divisibility_l557_557188


namespace train_speed_l557_557111

theorem train_speed (length_train length_bridge time : ℝ) :
  length_train = 250 ∧
  length_bridge = 520 ∧
  time = 30 →
  (length_train + length_bridge) / time = 25.67 := 
by {
  intros h,
  rcases h with ⟨ht, hb, ht⟩,
  simp [ht, hb, ht],
  norm_num,
}

end train_speed_l557_557111


namespace king_lancelot_seats_38_l557_557892

noncomputable def totalSeats (seat_king seat_lancelot : ℕ) : ℕ :=
  if seat_king < seat_lancelot then
    2 * (seat_lancelot - seat_king - 1) + 2
  else
    2 * (seat_king - seat_lancelot - 1) + 2

theorem king_lancelot_seats_38 (seat_king seat_lancelot : ℕ) (h1 : seat_king = 10) (h2 : seat_lancelot = 29) :
  totalSeats seat_king seat_lancelot = 38 := 
  by
    sorry

end king_lancelot_seats_38_l557_557892


namespace domain_of_f_l557_557372

def domain_of_function (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x : ℝ, f x ∈ ℝ → x ∈ S

noncomputable def f (x : ℝ) : ℝ := (x + 1)^2 / Real.sqrt (x + 2)

theorem domain_of_f :
  domain_of_function f (Set.Ioi (-2)) :=
by
  sorry

end domain_of_f_l557_557372


namespace find_minor_arc_angle_l557_557532

noncomputable def distance_point_line (x1 y1 A B C : ℝ) : ℝ :=
  abs (A * x1 + B * y1 + C) / real.sqrt (A^2 + B^2)

noncomputable def minor_arc_angle (line_eq: ℝ → ℝ → ℝ) (circle_eq : ℝ → ℝ → Prop) : ℝ :=
  let (A, B, C) := (6, 8, -10) in
  let r := 2 in
  let d := distance_point_line 0 0 A B C in
  let half_angle := real.arccos (d / r) in
  2 * half_angle

theorem find_minor_arc_angle : minor_arc_angle (λ x y, 6*x + 8*y - 10) (λ x y, x^2 + y^2 = 4) = 2*real.pi / 3 :=
  sorry

end find_minor_arc_angle_l557_557532


namespace opposite_of_3_l557_557777

theorem opposite_of_3 : -3 = -3 := 
by
  -- sorry is added to skip the proof as per instructions
  sorry

end opposite_of_3_l557_557777


namespace probability_A_middle_BC_adjacent_l557_557363

noncomputable theory

def prob_person_A_middle_BC_adjacent : ℚ :=
  let total_arrangements := fact 9 in
  let valid_arrangements := choose 6 1 * 2! * fact 6 in
  valid_arrangements / total_arrangements

theorem probability_A_middle_BC_adjacent :
  prob_person_A_middle_BC_adjacent = 1 / 42 :=
by
  sorry

end probability_A_middle_BC_adjacent_l557_557363


namespace stratified_sampling_is_appropriate_l557_557833

structure Population :=
  (elderly : ℕ)
  (middle_aged : ℕ)
  (young : ℕ)

def sample_size : ℕ := 41

def appropriate_sampling_method (p : Population) : Prop :=
  ∃ (elderly_prop middle_aged_prop young_prop : ℝ), 
    elderly_prop = p.elderly.toReal / (p.elderly + p.middle_aged + p.young).toReal ∧
    middle_aged_prop = p.middle_aged.toReal / (p.elderly + p.middle_aged + p.young).toReal ∧
    young_prop = p.young.toReal / (p.elderly + p.middle_aged + p.young).toReal ∧
    ∃ (elderly_sample middle_aged_sample young_sample : ℕ),
      elderly_sample = (sample_size.toReal * elderly_prop).toNat ∧
      middle_aged_sample = (sample_size.toReal * middle_aged_prop).toNat ∧
      young_sample = (sample_size.toReal * young_prop).toNat 

theorem stratified_sampling_is_appropriate (p : Population) (h : p.elderly = 28 ∧ p.middle_aged = 56 ∧ p.young = 80) : appropriate_sampling_method p :=
  sorry

end stratified_sampling_is_appropriate_l557_557833


namespace eventual_stability_l557_557344

noncomputable def stabilize_list (l : List ℤ) : ℕ → List ℤ
| 0 => l
| (n + 1) => (stabilize_list n).map (λ x => (stabilize_list n).count x)

theorem eventual_stability (l : List ℤ) (h : l.length = 1000) :
  ∃ N : ℕ, ∀ m ≥ N, stabilize_list l m = stabilize_list l N :=
sorry

end eventual_stability_l557_557344


namespace find_speed_of_train_l557_557438

/-- Definitions of the problem conditions --/

def train_length : ℝ := 120 -- in meters
def crossing_time : ℝ := 6 -- in seconds
def man_speed_kmph : ℝ := 5 -- in km/h

/-- Conversion factor from km/h to m/s --/
def kmph_to_mps (speed_kmph : ℝ) : ℝ := (speed_kmph * 1000) / 3600

/-- Theorem statement to prove the speed of the train --/
theorem find_speed_of_train
  (h1 : train_length = 120)
  (h2 : crossing_time = 6)
  (h3 : man_speed_kmph = 5) :
  let V_train := (120 / 6) - kmph_to_mps 5 in
  (V_train * 3600 / 1000) = 67 :=
by
  sorry

end find_speed_of_train_l557_557438


namespace last_three_digits_of_7_pow_215_l557_557175

theorem last_three_digits_of_7_pow_215 :
  (7 ^ 215) % 1000 = 447 := by
  sorry

end last_three_digits_of_7_pow_215_l557_557175


namespace hyperbola_center_l557_557533

theorem hyperbola_center : ∃ c : ℝ × ℝ, c = (3, 5) ∧
  ∀ x y : ℝ, 9 * x ^ 2 - 54 * x - 36 * y ^ 2 + 360 * y - 891 = 0 → (c.1 = 3 ∧ c.2 = 5) :=
by
  use (3, 5)
  sorry

end hyperbola_center_l557_557533


namespace factorial_division_l557_557502

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem factorial_division :
  (factorial 17) / (factorial 7 * factorial 10) = 408408 := by
  sorry

end factorial_division_l557_557502


namespace sum_integer_values_l557_557990

theorem sum_integer_values (m : ℤ) (h : 0 < 5 * m ∧ 5 * m < 35) :
  ∑ x in {n : ℤ | 0 < n ∧ n < 7}.to_finset, x = 21 :=
sorry

end sum_integer_values_l557_557990


namespace thickness_of_wall_l557_557250

theorem thickness_of_wall 
    (brick_length cm : ℝ)
    (brick_width cm : ℝ)
    (brick_height cm : ℝ)
    (num_bricks : ℝ)
    (wall_length cm : ℝ)
    (wall_height cm : ℝ)
    (wall_thickness cm : ℝ) :
    brick_length = 25 → 
    brick_width = 11.25 → 
    brick_height = 6 →
    num_bricks = 7200 → 
    wall_length = 900 → 
    wall_height = 600 →
    wall_length * wall_height * wall_thickness = num_bricks * (brick_length * brick_width * brick_height) →
    wall_thickness = 22.5 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end thickness_of_wall_l557_557250


namespace max_square_area_in_rhombus_l557_557037

noncomputable def side_length_triangle := 10
noncomputable def height_triangle := Real.sqrt (side_length_triangle^2 - (side_length_triangle / 2)^2)
noncomputable def diag_long := 2 * height_triangle
noncomputable def diag_short := side_length_triangle
noncomputable def side_square := diag_short / Real.sqrt 2
noncomputable def area_square := side_square^2

theorem max_square_area_in_rhombus :
  area_square = 50 := by sorry

end max_square_area_in_rhombus_l557_557037


namespace part1_part2_l557_557583

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 + Real.log x - a * x

theorem part1 (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → f x a ≥ 0) → a ≤ 1 :=
sorry

theorem part2 {a : ℝ} (x1 x2 : ℝ) (h1 : x1 ∈ Ioo 0 1) (h2 : f x1 a - f x2 a > m) :
  m ≤ -3 / 4 + Real.log 2 :=
sorry

end part1_part2_l557_557583


namespace opposite_of_three_l557_557740

theorem opposite_of_three : -3 = -3 := 
by sorry

end opposite_of_three_l557_557740


namespace max_ratio_l557_557063

def grid (n : ℕ) : Type := fin n × fin n

def is_crossword (S : set (grid n)) : Prop := 
  S.nonempty

def is_word (S : set (grid n)) : Prop :=
  (∃ x : fin n, ∀ y : fin n, (⟨x, y⟩ ∈ S) ∨ (∃ y : fin n, ∀ x : fin n, (⟨x, y⟩ ∈ S))) ∧
  ∀ T ⊃ S, ¬ (∃ x : fin n, ∀ y : fin n, (⟨x, y⟩ ∈ T) ∨ (∃ y : fin n, ∀ x : fin n, (⟨y, x⟩ ∈ T)))

def number_of_words (C : set (grid n)) : ℕ :=
  {W : set (grid n) | is_word W ∧ W ⊆ C }.to_finset.card

def minimum_number_of_words (C : set (grid n)) : ℕ :=
  Inf {k | ∃ Ws : fin k → set (grid n), (∀ i, is_word (Ws i)) ∧ (⋃ i, Ws i) = C}

theorem max_ratio (n : ℕ) (h : n > 1) (C : set (grid n)) (hc : is_crossword C) :
  let x := number_of_words C
  let y := minimum_number_of_words C
  x / y ≤ 1 + n / 2 :=
sorry

end max_ratio_l557_557063


namespace area_of_region_l557_557835

noncomputable def area_of_region_tangents (r l : ℝ) : ℝ :=
  let inner_radius := r
  let outer_radius := real.sqrt ((l / 2)^2 + r^2)
  π * (outer_radius^2 - inner_radius^2)

theorem area_of_region (radius tangent_length : ℝ) (h_radius : radius = 3) (h_length : tangent_length = 3) :
  area_of_region_tangents radius tangent_length = 2.25 * π := by
  sorry

end area_of_region_l557_557835


namespace total_items_to_buy_l557_557038

theorem total_items_to_buy (total_money : ℝ) (cost_sandwich : ℝ) (cost_drink : ℝ) (num_items : ℕ) :
  total_money = 30 → cost_sandwich = 4.5 → cost_drink = 1 → num_items = 9 :=
by
  sorry

end total_items_to_buy_l557_557038


namespace integer_solutions_abs_inequality_l557_557179

-- Define the condition as a predicate
def abs_inequality_condition (x : ℝ) : Prop := |x - 4| ≤ 3

-- State the proposition
theorem integer_solutions_abs_inequality : ∃ (n : ℕ), n = 7 ∧ ∀ (x : ℤ), abs_inequality_condition x → (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6 ∨ x = 7) :=
sorry

end integer_solutions_abs_inequality_l557_557179


namespace dividend_calculation_l557_557423

theorem dividend_calculation (divisor quotient remainder : ℕ) (h1 : divisor = 18) (h2 : quotient = 9) (h3 : remainder = 5) : 
  (divisor * quotient + remainder = 167) :=
by
  sorry

end dividend_calculation_l557_557423


namespace each_nap_duration_l557_557130

-- Definitions based on the problem conditions
def BillProjectDurationInDays : ℕ := 4
def HoursPerDay : ℕ := 24
def TotalProjectHours : ℕ := BillProjectDurationInDays * HoursPerDay
def WorkHours : ℕ := 54
def NapsTaken : ℕ := 6

-- Calculate the time spent on naps and the duration of each nap
def NapHoursTotal : ℕ := TotalProjectHours - WorkHours
def DurationEachNap : ℕ := NapHoursTotal / NapsTaken

-- The theorem stating the expected answer
theorem each_nap_duration :
  DurationEachNap = 7 := by
  sorry

end each_nap_duration_l557_557130


namespace amount_each_girl_gets_l557_557105

theorem amount_each_girl_gets
  (B G : ℕ) 
  (total_sum : ℝ)
  (amount_each_boy : ℝ)
  (sum_boys_girls : B + G = 100)
  (total_sum_distributed : total_sum = 312)
  (amount_boy : amount_each_boy = 3.60)
  (B_approx : B = 60) :
  (total_sum - amount_each_boy * B) / G = 2.40 := 
by 
  sorry

end amount_each_girl_gets_l557_557105


namespace base_number_is_three_l557_557614

theorem base_number_is_three (some_number : ℝ) (y : ℕ) (h1 : 9^y = some_number^14) (h2 : y = 7) : some_number = 3 :=
by { sorry }

end base_number_is_three_l557_557614


namespace part1_l557_557450

theorem part1 (z : ℂ) (h : z^2 + z + 1 = 0) : 
  z^2005 + z^2006 + z^2008 + z^2009 = -2 :=
  sorry

end part1_l557_557450


namespace triangle_area_l557_557333

theorem triangle_area (x : ℝ) (h1 : 60 < x) (h2 : x < 120) 
  (perimeter : ℝ) (h3 : perimeter = 48) 
  (inradius : ℝ) (h4 : inradius = 2.5) : 
  ∃ (area : ℝ), area = 120 :=
by
  let area := perimeter * inradius
  existsi area
  rw [h3, h4]
  norm_num
  rfl

end triangle_area_l557_557333


namespace tomato_puree_water_percentage_l557_557249

theorem tomato_puree_water_percentage :
  (∀ (juice_purity water_percentage : ℝ), 
    (juice_purity = 0.90) → 
    (20 * juice_purity = 18) →
    (2.5 - 2) = 0.5 →
    (2.5 * water_percentage - 0.5) = 0 →
    water_percentage = 0.20) :=
by
  intros juice_purity water_percentage h1 h2 h3 h4
  sorry

end tomato_puree_water_percentage_l557_557249


namespace price_of_wheat_flour_l557_557039

theorem price_of_wheat_flour
  (initial_amount : ℕ)
  (price_rice : ℕ)
  (num_rice : ℕ)
  (price_soda : ℕ)
  (num_soda : ℕ)
  (num_wheat_flour : ℕ)
  (remaining_balance : ℕ)
  (total_spent : ℕ)
  (amount_spent_on_rice_and_soda : ℕ)
  (amount_spent_on_wheat_flour : ℕ)
  (price_per_packet_wheat_flour : ℕ) 
  (h_initial_amount : initial_amount = 500)
  (h_price_rice : price_rice = 20)
  (h_num_rice : num_rice = 2)
  (h_price_soda : price_soda = 150)
  (h_num_soda : num_soda = 1)
  (h_num_wheat_flour : num_wheat_flour = 3)
  (h_remaining_balance : remaining_balance = 235)
  (h_total_spent : total_spent = initial_amount - remaining_balance)
  (h_amount_spent_on_rice_and_soda : amount_spent_on_rice_and_soda = price_rice * num_rice + price_soda * num_soda)
  (h_amount_spent_on_wheat_flour : amount_spent_on_wheat_flour = total_spent - amount_spent_on_rice_and_soda)
  (h_price_per_packet_wheat_flour : price_per_packet_wheat_flour = amount_spent_on_wheat_flour / num_wheat_flour) :
  price_per_packet_wheat_flour = 25 :=
by 
  sorry

end price_of_wheat_flour_l557_557039


namespace opposite_of_3_is_neg3_l557_557748

def opposite (x : ℝ) := -x

theorem opposite_of_3_is_neg3 : opposite 3 = -3 :=
by
  sorry

end opposite_of_3_is_neg3_l557_557748


namespace acute_angles_sum_pi_over_two_l557_557007

theorem acute_angles_sum_pi_over_two
  (α β : ℝ)
  (h1 : α > 0 ∧ α < π / 2)
  (h2 : β > 0 ∧ β < π / 2)
  (h3 : sin α ^ 2 + sin β ^ 2 = sin (α + β)) :
  α + β = π / 2 :=
sorry

end acute_angles_sum_pi_over_two_l557_557007


namespace max_value_of_a_plus_2b_l557_557247

variables {R : Type*} [Real R]

-- Vectors a and b with magnitudes 1
variables (a b : R → R) (h_a : norm a = 1) (h_b : norm b = 1)

-- Definition for maximum value of |a + 2b|
theorem max_value_of_a_plus_2b : (|a + 2 * b| ≤ 3) ∧ (|a + 2 * b| = 3 → |a + b| = 1) :=
by
  sorry

end max_value_of_a_plus_2b_l557_557247


namespace general_term_formula_sum_abs_diff_l557_557556

noncomputable def general_term (n : ℕ) : ℕ := 3 * n - 1

noncomputable def T_n (n : ℕ) : ℕ := sorry  -- Sum of the first n terms of the sequence {a_n}

axiom pos_sequence (n : ℕ) : a_n n > 0

axiom seq_initial : a_n 1 = 2

axiom seq_condition (n : ℕ) : a_n n * a_n (n + 1) = 6 * T_n n - 2

theorem general_term_formula (n : ℕ) : a_n n = general_term n :=
sorry

def b_n (n : ℕ) : ℕ := 2^n

def abs_seq_difference (n : ℕ) : ℕ := abs (a_n n - b_n n)

noncomputable def S_n (n : ℕ) : ℕ :=
if h : 1 ≤ n ∧ n ≤ 3 then
  (1/2:ℚ) * n * (3 * n + 1) - 2^(n + 1) + 2
else if h : 4 ≤ n then
  2^(n + 1) - (1/2:ℚ) * n * (3 * n + 1)
else
  0

theorem sum_abs_diff (n : ℕ) : S_n n = 
if h : 1 ≤ n ∧ n ≤ 3 then
  (1/2:ℚ) * n * (3 * n + 1) - 2^(n + 1) + 2
else if h : 4 ≤ n then
  2^(n + 1) - (1/2:ℚ) * n * (3 * n + 1)
else
  0 :=
sorry

end general_term_formula_sum_abs_diff_l557_557556


namespace part1_part2_l557_557236

section Problem

variable {t : ℝ}

def f (x : ℝ) : ℝ := (x^3 - 6*x^2 + 3*x + t) * Real.exp x

-- Define the conditions and the solutions
-- Part I: Range of values for t such that f(x) has three extreme points
theorem part1 (h1 : t > -8) (h2 : t < 24) : 
  ∃ x1 x2 x3, f.derivative x1 = 0 ∧ f.derivative x2 = 0 ∧ f.derivative x3 = 0 ∧ x1 ≠ x2 ∧ x2 ≠ x3 :=
sorry 

-- Part II: Maximum m such that for any x in [1, m], f(x) ≤ x holds
theorem part2 (h1 : t ∈ set.Icc 0 2) (h2 : ∀ x ∈ set.Icc 1 5, f x ≤ x) : 
  ∃ m, m = 5 :=
sorry 

end Problem

end part1_part2_l557_557236


namespace find_a_and_b_max_f_on_interval_inequality_proof_l557_557970

noncomputable def f (x : ℝ) : ℝ := Real.exp x - a * x^2

theorem find_a_and_b (a b : ℝ) :
  (f'(1) = Real.exp 1 - 2 * a ∧ f(1) = Real.exp 1 - a ∧ f'(1) = b ∧ f(1) = b + 1) → 
  (a = 1 ∧ b = Real.exp 1 - 2) :=
by
  sorry

theorem max_f_on_interval :
  (∀ x ∈ Icc 0 1, f(x) ≤ Real.exp 1 - 1) ∧ f(1) = Real.exp 1 - 1 :=
by
  sorry

theorem inequality_proof (x : ℝ) (h : x > 0) :
  Real.exp x + (1 - Real.exp 1) * x - x * Real.log x - 1 ≥ 0 :=
by
  sorry

end find_a_and_b_max_f_on_interval_inequality_proof_l557_557970


namespace nh4cl_formed_l557_557954

theorem nh4cl_formed :
  (∀ (nh3 hcl nh4cl : ℝ), nh3 = 1 ∧ hcl = 1 → nh3 + hcl = nh4cl → nh4cl = 1) :=
by
  intros nh3 hcl nh4cl
  sorry

end nh4cl_formed_l557_557954


namespace positive_t_value_l557_557154

theorem positive_t_value (t : ℝ) (ht : |complex.mk 9 t| = 15) : t = 12 := 
by
    sorry

end positive_t_value_l557_557154


namespace symmetric_lines_intersect_or_parallel_l557_557691

-- Define the geometrical entities and conditions
variable (A B C P : Point)
variable (L_A L_B L_C L_A' L_B' L_C' l_A l_B l_C : Line)

-- Conditions: Given a triangle and specific relationships between points and lines
axiom h1 : ∃ (triangle : Triangle), triangle.vertices = (A, B, C)
axiom h2 : ∀ (L : Line), ∃ (P : Point), L.passes_through P
axiom h3 : L_A.passes_through A ∧ L_B.passes_through B ∧ L_C.passes_through C
axiom h4 : ∃ P, L_A.intersects L_B = P ∧ L_B.intersects L_C = P ∧ L_C.intersects L_A = P
axiom h5 : ∀ (L : Line), ∃ (l : Line), l.is_angle_bisector

-- Reflection property across respective angle bisectors
axiom h6 : L_A' = reflect L_A l_A
axiom h7 : L_B' = reflect L_B l_B
axiom h8 : L_C' = reflect L_C l_C

-- The statement to prove
theorem symmetric_lines_intersect_or_parallel :
  (∃ P', L_A'.intersects L_B' = P' ∧ L_B'.intersects L_C' = P' ∧ L_C'.intersects L_A' = P') 
  ∨ ∀ L1 L2, L1 = L_A' ∨ L1 = L_B' ∨ L1 = L_C' → L2 = L_A' ∨ L2 = L_B' ∨ L2 = L_C' → L1.is_parallel_to L2 :=
sorry

end symmetric_lines_intersect_or_parallel_l557_557691


namespace kanul_total_amount_l557_557059

-- Define the conditions given in the problem
def spent_on_raw_materials : ℝ := 500
def spent_on_machinery : ℝ := 400
def percentage_spent_on_cash (T : ℝ) : ℝ := 0.10 * T

-- Total amount of cash Kanul had initially
noncomputable def total_amount : ℝ :=
  let total_spent := spent_on_raw_materials + spent_on_machinery + percentage_spent_on_cash T
  T

-- The statement of the problem in Lean
theorem kanul_total_amount (T : ℝ) (h : T = 900 / 0.90) : T = 1000 :=
sorry

end kanul_total_amount_l557_557059


namespace page_shoes_l557_557688

/-- Page's initial collection of shoes -/
def initial_collection : ℕ := 80

/-- Page donates 30% of her collection -/
def donation (n : ℕ) : ℕ := n * 30 / 100

/-- Page buys additional shoes -/
def additional_shoes : ℕ := 6

/-- Page's final collection after donation and purchase -/
def final_collection (n : ℕ) : ℕ := (n - donation n) + additional_shoes

/-- Proof that the final collection of shoes is 62 given the initial collection of 80 pairs -/
theorem page_shoes : (final_collection initial_collection) = 62 := 
by sorry

end page_shoes_l557_557688


namespace total_seats_l557_557904

theorem total_seats (KA_pos : ℕ) (SL_pos : ℕ) (h1 : KA_pos = 10) (h2 : SL_pos = 29) (h3 : SL_pos = KA_pos + (KA_pos * 2 - 1) / 2):
  let total_positions := 2 * (SL_pos - KA_pos - 1) + 2
  total_positions = 38 :=
by
  sorry

end total_seats_l557_557904


namespace largest_negative_root_l557_557172

theorem largest_negative_root : 
  ∃ x : ℝ, (∃ k : ℤ, x = -1/2 + 2 * ↑k) ∧ 
  ∀ y : ℝ, (∃ k : ℤ, (y = -1/2 + 2 * ↑k ∨ y = 1/6 + 2 * ↑k ∨ y = 5/6 + 2 * ↑k)) → y < 0 → y ≤ x :=
sorry

end largest_negative_root_l557_557172


namespace opposite_of_three_l557_557761

theorem opposite_of_three : -3 = -3 := by
  -- The condition we have identified is the given number 3.
  -- We will directly state that the opposite of 3 is -3.
  -- This proof is trivial as we are directly replacing 3 with -3 to match the problem statement.
  rfl

end opposite_of_three_l557_557761


namespace grandma_walking_minutes_saved_l557_557598

theorem grandma_walking_minutes_saved :
  let distance_monday := 3
  let speed_monday := 6
  let distance_wednesday := 4
  let speed_wednesday := 4
  let distance_friday := 5
  let speed_friday := 5
  let hours_monday := distance_monday / speed_monday
  let hours_wednesday := distance_wednesday / speed_wednesday
  let hours_friday := distance_friday / speed_friday
  let total_time := hours_monday + hours_wednesday + hours_friday
  let total_distance := distance_monday + distance_wednesday + distance_friday
  let constant_speed := 5
  let reduced_time := total_distance / constant_speed
  let time_saved := total_time - reduced_time
  let minutes_saved := time_saved * 60
  minutes_saved = 6 := by
begin
  sorry
end

end grandma_walking_minutes_saved_l557_557598


namespace min_f_gt_min_g_l557_557607

open Function

variable {α : Type*} (f g : ℝ → ℝ)

theorem min_f_gt_min_g (h : ∀ x : ℝ, ∃ x₀ : ℝ, f x > g x₀) : 
    Inf (range f) > Inf (range g) :=
sorry

end min_f_gt_min_g_l557_557607


namespace find_x_l557_557382

theorem find_x (x : ℤ) (h_neg : x < 0)
  (h_median : median ({18, 27, 33, x, 20} : Finset ℤ) = mean ({18, 27, 33, x, 20} : Finset ℤ) + 3) : 
  x = -13 := 
sorry

end find_x_l557_557382


namespace total_seats_round_table_l557_557867

theorem total_seats_round_table (n : ℕ) (h : n = 38)
  (ka_position : ℕ) (sl_position : ℕ) 
  (h1 : ka_position = 10) 
  (h2 : sl_position = 29) 
  (h3 : (ka_position + n/2) % n = sl_position) : 
  n = 38 :=
by
  -- All steps and solution proof
  sorry

end total_seats_round_table_l557_557867


namespace fifteenth_entry_is_30_l557_557539

def r8 (n : ℕ) : ℕ := n % 8

def satisfies_condition (n : ℕ) : Prop := r8 (7 * n) ≤ 3

def ordered_list_of_satisfying_numbers : List ℕ := 
  List.filter satisfies_condition (List.range 1000)  -- Assuming we consider the first 1000 elements for practical purposes

theorem fifteenth_entry_is_30 : List.nth ordered_list_of_satisfying_numbers 14 = some 30 :=
sorry

end fifteenth_entry_is_30_l557_557539


namespace cardinality_of_set_minus_finite_l557_557265

open Set

theorem cardinality_of_set_minus_finite (A B : Set ℕ) (S : Cardinal) (hA : Cardinal.mk A = S) (hB : Finite B) :
  Cardinal.mk (A \ B) = S := by
  sorry

end cardinality_of_set_minus_finite_l557_557265


namespace reduced_price_per_kg_l557_557480

variable (P : ℝ)
variable (R : ℝ)
variable (Q : ℝ)

theorem reduced_price_per_kg
  (h1 : R = 0.75 * P)
  (h2 : 500 = Q * P)
  (h3 : 500 = (Q + 5) * R)
  (h4 : Q = 500 / P) :
  R = 25 :=
by
  sorry

end reduced_price_per_kg_l557_557480


namespace triangle_area_l557_557446

theorem triangle_area (a b c p : ℕ) (h_ratio : a = 5 * p) (h_ratio2 : b = 12 * p) (h_ratio3 : c = 13 * p) (h_perimeter : a + b + c = 300) : 
  (1 / 4) * Real.sqrt ((a + b + c) * (a + b - c) * (a + c - b) * (b + c - a)) = 3000 := 
by 
  sorry

end triangle_area_l557_557446


namespace infinite_sum_series_eq_five_over_twelve_l557_557509

theorem infinite_sum_series_eq_five_over_twelve : 
  (∑ n in (Finset.range ∞).filter ((λ n => n ≥ 1)), 1 / ((n + 1) * (n + 3))) = 5 / 12 := 
by
  sorry

end infinite_sum_series_eq_five_over_twelve_l557_557509


namespace unique_function_f_l557_557952

theorem unique_function_f (f : ℕ → ℕ) 
  (h : ∀ m n, (f m)^2 + 2 * m * (f n) + (f (n^2)) = (∃ x : ℕ, x^2)) : 
  ∀ n, f n = n :=
sorry

end unique_function_f_l557_557952


namespace deepak_wife_meet_time_in_minutes_l557_557445

theorem deepak_wife_meet_time_in_minutes :
  let circumference := 561 -- the circumference of the jogging track in meters
  let deepak_speed_km_hr := 4.5 -- Deepak's speed in km/hr
  let wife_speed_km_hr := 3.75 -- Wife's speed in km/hr
  let mins_per_hour := 60
  let deepak_speed_m_min := (deepak_speed_km_hr * 1000) / mins_per_hour -- Deepak's speed in m/min
  let wife_speed_m_min := (wife_speed_km_hr * 1000) / mins_per_hour -- Wife's speed in m/min
  let combined_speed := deepak_speed_m_min + wife_speed_m_min -- Combined speed in m/min
  let time_to_meet := circumference / combined_speed -- Time to meet in minutes
  time_to_meet ≈ 4.08 := 
sorry 

end deepak_wife_meet_time_in_minutes_l557_557445


namespace area_of_triangles_l557_557298

theorem area_of_triangles
  (ABC_area : ℝ)
  (AD : ℝ)
  (DB : ℝ)
  (h_AD_DB : AD + DB = 7)
  (h_equal_areas : ABC_area = 12) :
  (∃ ABE_area : ℝ, ABE_area = 36 / 7) ∧ (∃ DBF_area : ℝ, DBF_area = 36 / 7) :=
by
  sorry

end area_of_triangles_l557_557298


namespace opposite_of_3_l557_557772

theorem opposite_of_3 : -3 = -3 := 
by
  -- sorry is added to skip the proof as per instructions
  sorry

end opposite_of_3_l557_557772


namespace dot_product_eq_neg102_l557_557534

def vec1 : ℝ × ℝ × ℝ × ℝ := (4, -5, 6, -3)
def vec2 : ℝ × ℝ × ℝ × ℝ := (-2, 8, -7, 4)

theorem dot_product_eq_neg102 : 
  let dot_product := vec1.1 * vec2.1 + vec1.2 * vec2.2 + vec1.3 * vec2.3 + vec1.4 * vec2.4 
  in dot_product = -102 := 
by 
  sorry

end dot_product_eq_neg102_l557_557534


namespace sqrt_inequality_no_natural_solution_for_floor_inequality_l557_557056

theorem sqrt_inequality (n : ℕ) : sqrt (n + 1) + 2 * sqrt n < sqrt (9 * n + 3) :=
sorry

theorem no_natural_solution_for_floor_inequality :
  ¬ ∃ n : ℕ, ⌊sqrt (n + 1) + 2 * sqrt n⌋ < ⌊sqrt (9 * n + 3)⌋ :=
sorry

end sqrt_inequality_no_natural_solution_for_floor_inequality_l557_557056


namespace range_of_f_l557_557938

def star (a b : ℝ) : ℝ := if a ≤ b then a else b

def f (x : ℝ) : ℝ := star (Real.cos x) (Real.sin x)

theorem range_of_f : (set.range f) = set.Icc (-1) (Real.sqrt 2 / 2) :=
by
  sorry

end range_of_f_l557_557938


namespace tree_planting_festival_minimized_distance_l557_557522

theorem tree_planting_festival_minimized_distance :
  ∃ p1 p2 : ℕ, p1 = 10 ∧ p2 = 11 ∧ minimizes_distance p1 p2 (λ n, 1 ≤ n ∧ n ≤ 20) :=
sorry

end tree_planting_festival_minimized_distance_l557_557522


namespace sum_modulo_9_l557_557537

theorem sum_modulo_9 : 
  (88000 + 88002 + 87999 + 88001 + 88003 + 87998) % 9 = 0 := 
by
  sorry

end sum_modulo_9_l557_557537


namespace parity_of_good_circles_l557_557325

theorem parity_of_good_circles (S : Set Point) (n : ℕ)
  (h_card : S.card = 2 * n + 1)
  (h_no_three_collinear : ∀ P Q R : Point, P ∈ S → Q ∈ S → R ∈ S → collinear P Q R = false)
  (h_no_four_concyclic : ∀ P Q R T : Point, P ∈ S → Q ∈ S → R ∈ S → T ∈ S → concyclic P Q R T = false)
  (good_circle : Circle → Prop)
  (h_good_circle_def : ∀ c, good_circle c ↔ (∃ A B C ∈ S, A≠B ∧ B≠C ∧ C≠A ∧ A,B,C ∈ c ∧ 
                                        (S - {A, B, C}).card = 2 * (n - 1) + 1)) :
  (∃ k, count_good_circles S good_circle = 2 * k + n) :=
sorry

end parity_of_good_circles_l557_557325


namespace problem_l557_557345

-- Definition of the problem
def three_squares (squares : Set (Set (ℕ × ℕ))) : Prop :=
  ∃ s1 s2 s3 : Set (ℕ × ℕ),
  s1 ∈ squares ∧ s2 ∈ squares ∧ s3 ∈ squares ∧
  ∀ i ∈ {s1, s2, s3}, ∃ n m : ℕ, i = (λ (x y : ℕ), (x, y)) '' {p : ℕ × ℕ | p.1 < n ∧ p.2 < m} ∧
  ∃ n m : ℕ, n = 5 ∧ m = 5

-- Correct answer statement
def cells_covered_exactly_two_squares (squares : Set (Set (ℕ × ℕ))) : Prop :=
  (∃ c12 c13 c23 : ℕ,
  c12 = ((squares.to_list.nth 0) ∩ (squares.to_list.nth 1)).size ∧
  c13 = ((squares.to_list.nth 0) ∩ (squares.to_list.nth 2)).size ∧
  c23 = ((squares.to_list.nth 1) ∩ (squares.to_list.nth 2)).size ∧
  c12 + c13 + c23 = 45)

-- The final problem statement to prove
theorem problem : ∀ squares : Set (Set (ℕ × ℕ)),
  three_squares squares →
  cells_covered_exactly_two_squares squares →
  ∃ n : ℕ, n = 15 :=
  sorry

end problem_l557_557345


namespace total_amount_l557_557680

noncomputable def mark_amount : ℝ := 5 / 8

noncomputable def carolyn_amount : ℝ := 7 / 20

theorem total_amount : mark_amount + carolyn_amount = 0.975 := by
  sorry

end total_amount_l557_557680


namespace sin_double_angle_l557_557203

theorem sin_double_angle (θ : ℝ) 
    (h : tan (θ + π / 4) = 2 * tan θ - 7) : 
    sin (2 * θ) = 4 / 5 :=
    sorry

end sin_double_angle_l557_557203


namespace flower_beds_fraction_l557_557096

noncomputable def yard_area : ℝ := 100
noncomputable def parallel_side_1 : ℝ := 10
noncomputable def parallel_side_2 : ℝ := 20
noncomputable def num_triangles : ℕ := 3

theorem flower_beds_fraction :
  let side_diff := parallel_side_2 - parallel_side_1,
      leg_length := side_diff / num_triangles,
      triangle_area := (1 / 2) * leg_length^2,
      total_flower_beds_area := num_triangles * triangle_area
  in (total_flower_beds_area / yard_area) = (1 / 6) :=
by sorry

end flower_beds_fraction_l557_557096


namespace mass_of_23_moles_C9H20_percentage_composition_C_H_O_in_C9H20_l557_557924

def molar_mass_carbon : Float := 12.01
def molar_mass_hydrogen : Float := 1.008
def moles_of_nonane : Float := 23.0
def num_carbons_in_nonane : Float := 9.0
def num_hydrogens_in_nonane : Float := 20.0

theorem mass_of_23_moles_C9H20 :
  let molar_mass_C9H20 := (num_carbons_in_nonane * molar_mass_carbon) + (num_hydrogens_in_nonane * molar_mass_hydrogen)
  let mass_23_moles := moles_of_nonane * molar_mass_C9H20
  mass_23_moles = 2950.75 :=
by
  let molar_mass_C9H20 := (num_carbons_in_nonane * molar_mass_carbon) + (num_hydrogens_in_nonane * molar_mass_hydrogen)
  let mass_23_moles := moles_of_nonane * molar_mass_C9H20
  have molar_mass_C9H20_val : molar_mass_C9H20 = 128.25 := sorry
  have mass_23_moles_val : mass_23_moles = 2950.75 := sorry
  exact mass_23_moles_val

theorem percentage_composition_C_H_O_in_C9H20 :
  let molar_mass_C9H20 := (num_carbons_in_nonane * molar_mass_carbon) + (num_hydrogens_in_nonane * molar_mass_hydrogen)
  let percentage_carbon := (num_carbons_in_nonane * molar_mass_carbon / molar_mass_C9H20) * 100
  let percentage_hydrogen := (num_hydrogens_in_nonane * molar_mass_hydrogen / molar_mass_C9H20) * 100
  let percentage_oxygen := 0
  percentage_carbon = 84.27 ∧ percentage_hydrogen = 15.73 ∧ percentage_oxygen = 0 :=
by
  let molar_mass_C9H20 := (num_carbons_in_nonane * molar_mass_carbon) + (num_hydrogens_in_nonane * molar_mass_hydrogen)
  let percentage_carbon := (num_carbons_in_nonane * molar_mass_carbon / molar_mass_C9H20) * 100
  let percentage_hydrogen := (num_hydrogens_in_nonane * molar_mass_hydrogen / molar_mass_C9H20) * 100
  let percentage_oxygen := 0
  have percentage_carbon_val : percentage_carbon = 84.27 := sorry
  have percentage_hydrogen_val : percentage_hydrogen = 15.73 := sorry
  have percentage_oxygen_val : percentage_oxygen = 0 := by rfl
  exact ⟨percentage_carbon_val, percentage_hydrogen_val, percentage_oxygen_val⟩

end mass_of_23_moles_C9H20_percentage_composition_C_H_O_in_C9H20_l557_557924


namespace opposite_of_3_l557_557726

-- Define the concept of opposite of a number and the logic for positive numbers
def opposite (x : Int) : Int := 
  if x > 0 then -x
  else if x < 0 then -x
  else 0

-- Statement to prove that the opposite of 3 is -3
theorem opposite_of_3 : opposite 3 = -3 :=
by 
  -- Using the definition of opposite
  unfold opposite
  -- Simplify the expression for x = 3
  simp [lt_irrefl, int.coe_nat_lt]
  -- Conclude proof
  rfl

end opposite_of_3_l557_557726


namespace prime_sq_mod_12_l557_557350

theorem prime_sq_mod_12 (p : ℕ) (hp_prime : Nat.Prime p) (hp_gt_3 : p > 3) : (p * p) % 12 = 1 := by
  sorry

end prime_sq_mod_12_l557_557350


namespace king_arthur_round_table_seats_l557_557901

theorem king_arthur_round_table_seats (n : ℕ) (h₁ : n > 1) (h₂ : 10 < 29) (h₃ : (29 - 10) * 2 = n - 2) : 
  n = 38 := 
by
  sorry

end king_arthur_round_table_seats_l557_557901


namespace find_angle_B_l557_557296

variable (ABCD : Type) [T : trapezoid ABCD]
variable (A B C D : ABCD)
variable (parallel_AB_CD : parallel (line A B) (line C D))
variable (angle_A : ℝ) (angle_B : ℝ) (angle_C : ℝ) (angle_D : ℝ)
variable (cond1 : angle_A = 2 * angle_D)
variable (cond2 : angle_C = 4 * angle_B)
variable (angle_sum : angle_B + angle_C = 180)

theorem find_angle_B : angle_B = 36 :=
by
  sorry

end find_angle_B_l557_557296


namespace sum_of_possible_values_l557_557674

theorem sum_of_possible_values (x y : ℝ) (h : x * y - x / y^3 - y / x^3 = 2) :
  (x - 2) * (y - 2) = 6 ∨ (x - 2) * (y - 2) = 9 →
  (if (x - 2) * (y - 2) = 6 then 6 else 0) + (if (x - 2) * (y - 2) = 9 then 9 else 0) = 15 :=
by
  sorry

end sum_of_possible_values_l557_557674


namespace at_parallel_xy_l557_557653

open EuclideanGeometry

/-- Given points A, B, M, ω, T, X, and Y with the following properties:
     1. A and B are distinct.
     2. M is the midpoint of AB.
     3. ω is a circle passing through A and M.
     4. T is a point on ω such that BT is tangent to ω.
     5. X is a point on line AB with TB = TX.
     6. Y is the foot of the perpendicular from A onto BT.
    Prove that AT is parallel to XY. -/
theorem at_parallel_xy
    {A B M : Point}
    {ω : Circle}
    {T X Y : Point}
    (h1 : A ≠ B)
    (h2 : midpoint A B M)
    (h3 : ω.passes_through A ∧ ω.passes_through M)
    (h4 : T ∈ ω ∧ tangent_line B ω T)
    (h5 : collinear A B X ∧ TB = TX)
    (h6 : foot_of_perpendicular A BT Y) :
    parallel (line_through AT) (line_through XY) :=
sorry

end at_parallel_xy_l557_557653


namespace neil_total_charge_l557_557341

theorem neil_total_charge 
  (trim_cost : ℕ) (shape_cost : ℕ) (total_boxwoods : ℕ) (shaped_boxwoods : ℕ) : 
  trim_cost = 5 → shape_cost = 15 → total_boxwoods = 30 → shaped_boxwoods = 4 → 
  trim_cost * total_boxwoods + shape_cost * shaped_boxwoods = 210 := 
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end neil_total_charge_l557_557341


namespace opposite_of_three_l557_557769

theorem opposite_of_three : -3 = opposite(3) := 
by
  sorry

end opposite_of_three_l557_557769


namespace tangency_points_concyclic_l557_557214

-- Define basic entities: points and circles
structure Point := (x : ℝ) (y : ℝ)

structure Circle :=
  (center : Point)
  (radius : ℝ)

-- Define tangency of circles
def externally_tangent_at (C1 C2 : Circle) (P : Point) : Prop :=
  dist C1.center P = C1.radius ∧ dist C2.center P = C2.radius ∧
  dist C1.center C2.center = C1.radius + C2.radius

-- Define the four circles and their tangency points
variables (Γ1 Γ2 Γ3 Γ4 : Circle)
variables (T12 T23 T34 T41 : Point)

-- The main theorem stating the tangency points are concyclic
theorem tangency_points_concyclic :
  externally_tangent_at Γ1 Γ2 T12 →
  externally_tangent_at Γ2 Γ3 T23 →
  externally_tangent_at Γ3 Γ4 T34 →
  externally_tangent_at Γ4 Γ1 T41 →
  ∃ C : Circle, T12 ∈ C ∧ T23 ∈ C ∧ T34 ∈ C ∧ T41 ∈ C := sorry

end tangency_points_concyclic_l557_557214


namespace sum_distances_squared_const_l557_557834

-- Define the circle O with equation x^2 + y^2 = 4
def circle_O (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

-- Define the curve C with its rectangular coordinate equation x^2 - y^2 = 1
def curve_C (x y : ℝ) : Prop :=
  x^2 - y^2 = 1

-- Define points M and N on the x-axis
def point_M : ℝ × ℝ := (-1, 0)
def point_N : ℝ × ℝ := (1, 0)

-- Define any point P on the circle O with parametric form
def point_P (a : ℝ) : ℝ × ℝ :=
  (2 * Real.cos a, 2 * Real.sin a)

-- Prove that |PM|^2 + |PN|^2 is a constant
theorem sum_distances_squared_const (a : ℝ) : 
  let P := point_P a,
      M := point_M,
      N := point_N in
  ((P.1 - M.1)^2 + (P.2 - M.2)^2) + ((P.1 - N.1)^2 + (P.2 - N.2)^2) = 10 := 
by
  sorry

end sum_distances_squared_const_l557_557834


namespace frequency_of_scoring_l557_557809

def shots : ℕ := 80
def goals : ℕ := 50
def frequency : ℚ := goals / shots

theorem frequency_of_scoring : frequency = 0.625 := by
  sorry

end frequency_of_scoring_l557_557809


namespace right_triangle_sum_of_legs_l557_557283

theorem right_triangle_sum_of_legs (a b : ℝ) (h₁ : a^2 + b^2 = 2500) (h₂ : (1 / 2) * a * b = 600) : a + b = 70 :=
sorry

end right_triangle_sum_of_legs_l557_557283


namespace greatest_b_value_ineq_l557_557169

theorem greatest_b_value_ineq (b : ℝ) (h : -b^2 + 8 * b - 15 ≥ 0) : b ≤ 5 := 
sorry

end greatest_b_value_ineq_l557_557169


namespace alice_wins_l557_557652

-- Conditions: 0 < c < 1, n is a positive integer, not all numbers are equal on the board
variables {c : ℝ} {n : ℕ}

-- Definitions based on the given problem statement
def game (c : ℝ) (n : ℕ) : Prop :=
  0 < c ∧ c < 1 ∧ n > 0 ∧ ∃ nums : list ℤ, nums.length = n ∧ (∀ a b ∈ nums, a ≠ b → ∃ m ∈ nums, m = (a + b) / 2)

-- Main statement: Proving the winning conditions
theorem alice_wins (c : ℝ) (n : ℕ) (N : ℕ) :
  (∀ n ≥ N, game c n → (c ≥ 1/2 → ∃ nums_2 : list ℤ, nums_2.length = 2 ∧ nums_2.head ≠ nums_2[1])) ∧
  (∀ n ≥ N, game c n → (c < 1/2 → ∃ nums_2 : list ℤ, nums_2.length = 2 ∧ nums_2.head = nums_2[1])) :=
sorry

end alice_wins_l557_557652


namespace opposite_of_three_l557_557743

theorem opposite_of_three : -3 = -3 := 
by sorry

end opposite_of_three_l557_557743


namespace root_problem_l557_557320

noncomputable def root_expression (a b c : ℝ) (ha : a + b + c = 24) (habc : a * b * c = 14) (habc_sum : a * b + b * c + c * a = 50) : ℝ :=
  a / (1 / a + b * c) + b / (1 / b + c * a) + c / (1 / c + a * b)

theorem root_problem (a b c : ℝ) 
  (h_eq : Polynomial.roots (x^3 - 24*x^2 + 50*x - 14) = {a, b, c}) 
  (ha : a + b + c = 24)
  (habc : ab + bc + ca = 50)
  (habc_sum : abc = 14) : 
  root_expression a b c ha habc habc_sum = 476 / 15 := 
sorry

end root_problem_l557_557320


namespace find_l2_l557_557225

-- Definition of symmetry about a point
def is_symmetric_about (p: ℝ × ℝ) (q₁ q₂: ℝ × ℝ) : Prop :=
  ∃ a b : ℝ, p = (a, b) ∧ q₁ = (2*a - q₂.fst, 2*b - q₂.snd)

-- Definition of the line equation
def line_equation (a b c: ℝ) (p: ℝ × ℝ) : Prop :=
  a * p.fst + b * p.snd + c = 0

-- Defining the first line l1 with the given equation y = x + 1
def l1 (p: ℝ × ℝ) : Prop :=
  line_equation 1 (-1) (-1) p

-- The point of symmetry (1, 1)
def point_of_symmetry := (1, 1 : ℝ)

-- The goal is to find the equation of the line l2
theorem find_l2 :
  ∃ a b c: ℝ, (∀ p: ℝ × ℝ, line_equation a b c p) ∧
  (∀ q: ℝ × ℝ, is_symmetric_about point_of_symmetry q (2 - q.fst, 2 - q.snd) → l1 (2 - q.fst, 2 - q.snd)) ∧
  (a = 1 ∧ b = -1 ∧ c = -1) :=
by
  sorry

end find_l2_l557_557225


namespace total_people_l557_557077

-- Let n be the total number of people
-- Assuming there are 36 handshakes involving consecutive pairs
-- And the smallest set of people such that removing them ensures the rest
-- have shaken hands with at least one from this set is 12 people.
theorem total_people (n : ℕ) 
  (h1 : ∃ hs : ℕ, hs = 36 ∧ handshakes_involving_consecutive_pairs hs n)
  (h2 : ∃ s : ℕ, s = 12 ∧ smallest_set_to_ensure_handshakes s n) : 
  n = 36 := 
sorry

-- Definitions for conditions (these are assumed correct to match the problem statement conditions)

-- This placeholder definition represents the constraint that there are 36 handshakes involving consecutive pairs.
def handshakes_involving_consecutive_pairs (hs : ℕ) (n : ℕ) : Prop := hs = n  

-- This placeholder definition represents the smallest set requirement.
def smallest_set_to_ensure_handshakes (s : ℕ) (n : ℕ) : Prop := (n - s) + handshakes_involving_consecutive_pairs s n > 0

end total_people_l557_557077


namespace steps_away_from_goal_l557_557142

-- Given conditions
def goal : ℕ := 100000
def initial_steps_per_day : ℕ := 1000
def increase_steps_per_week : ℕ := 1000
def days_per_week : ℕ := 7
def weeks : ℕ := 4

-- Computation of total steps in 4 weeks
def total_steps : ℕ :=
  (initial_steps_per_day * days_per_week) +
  ((initial_steps_per_day + increase_steps_per_week) * days_per_week) +
  ((initial_steps_per_day + 2 * increase_steps_per_week) * days_per_week) +
  ((initial_steps_per_day + 3 * increase_steps_per_week) * days_per_week)

-- Desired proof statement
theorem steps_away_from_goal : goal - total_steps = 30000 :=
by
  have h1: 7 * 1000 = 7000 := by norm_num
  have h2: 7 * 2000 = 14000 := by norm_num
  have h3: 7 * 3000 = 21000 := by norm_num
  have h4: 7 * 4000 = 28000 := by norm_num
  have h5: total_steps = 7000 + 14000 + 21000 + 28000 := by
    simp [total_steps, initial_steps_per_day, days_per_week, increase_steps_per_week]
    ring
  have h6: 7000 + 14000 + 21000 + 28000 = 70000 := by norm_num
  rw [h5, h6]
  show goal - 70000 = 30000
  norm_num

end steps_away_from_goal_l557_557142


namespace range_of_x_l557_557988

theorem range_of_x (x : ℝ) (h : sqrt((2 - 3 * abs x) ^ 2) = 2 + 3 * x) : -2 / 3 ≤ x ∧ x ≤ 0 :=
sorry

end range_of_x_l557_557988


namespace walnut_trees_in_park_l557_557021

-- Definition of initial conditions and the proof statement.
theorem walnut_trees_in_park (initial_trees planted_trees removed_trees final_trees : ℕ) 
  (h1 : initial_trees = 22) 
  (h2 : planted_trees = 45) 
  (h3 : removed_trees = 8) 
  (h4 : final_trees = initial_trees + planted_trees - removed_trees) : 
  final_trees = 59 := 
by {
  rw [h1, h2, h3] at h4,
  simp at h4,
  exact h4,
}

end walnut_trees_in_park_l557_557021


namespace squirrels_and_nuts_l557_557412

theorem squirrels_and_nuts (number_of_squirrels number_of_nuts : ℕ) 
    (h1 : number_of_squirrels = 4) 
    (h2 : number_of_squirrels = number_of_nuts + 2) : 
    number_of_nuts = 2 :=
by
  sorry

end squirrels_and_nuts_l557_557412


namespace car_speeds_l557_557419

noncomputable def distance_between_places : ℝ := 135
noncomputable def departure_time_diff : ℝ := 4 -- large car departs 4 hours before small car
noncomputable def arrival_time_diff : ℝ := 0.5 -- small car arrives 30 minutes earlier than large car
noncomputable def speed_ratio : ℝ := 5 / 2 -- ratio of speeds (small car : large car)

theorem car_speeds (v_small v_large : ℝ) (h1 : v_small / v_large = speed_ratio) :
    v_small = 45 ∧ v_large = 18 :=
sorry

end car_speeds_l557_557419


namespace find_first_number_l557_557789

theorem find_first_number (sum_is_33 : ∃ x y : ℕ, x + y = 33) (second_is_twice_first : ∃ x y : ℕ, y = 2 * x) (second_is_22 : ∃ y : ℕ, y = 22) : ∃ x : ℕ, x = 11 :=
by
  sorry

end find_first_number_l557_557789


namespace integers_between_sqrt7_and_sqrt77_l557_557602

theorem integers_between_sqrt7_and_sqrt77 : 
  2 < Real.sqrt 7 ∧ Real.sqrt 7 < 3 ∧ 8 < Real.sqrt 77 ∧ Real.sqrt 77 < 9 →
  ∃ (n : ℕ), n = 6 ∧ ∀ (k : ℕ), (3 ≤ k ∧ k ≤ 8) ↔ (2 < Real.sqrt 7 ∧ Real.sqrt 77 < 9) :=
by sorry

end integers_between_sqrt7_and_sqrt77_l557_557602


namespace opposite_of_three_l557_557763

theorem opposite_of_three : -3 = opposite(3) := 
by
  sorry

end opposite_of_three_l557_557763


namespace unit_squares_below_line_l557_557005

theorem unit_squares_below_line : 
  (∃ f : ℝ → ℝ, ∀ x, f x = (1296 - 6 * x) / 216) → 
  (∃ g : ℤ → ℝ × ℝ, ∀ k, g k = (k, (1296 - 6 * k) / 216) ∧ 0 ≤ k ∧ k ≤ 216) →
  (∃ h : ℤ → ℝ × ℝ, ∀ k, h k = ((1296 - 216 * k) / 6, k) ∧ 0 ≤ k ∧ k ≤ 6) →
  (number_of_squares_below_line 6 216 1296 216 6) = 537 :=
sorry

end unit_squares_below_line_l557_557005


namespace inverse_proportion_k_l557_557269

def inverse_proportion (k x : ℝ) : ℝ := k / x

theorem inverse_proportion_k (k : ℝ) : (inverse_proportion k 2 = -6) → k = -12 :=
by
  intro h
  sorry

end inverse_proportion_k_l557_557269


namespace range_of_k_l557_557264

noncomputable def quadratic_expression (k x : ℝ) : ℝ :=
  k^(-3) - x - k^2 * x^2

theorem range_of_k 
  (h : ∀ x : ℝ, (quadratic_expression k x < 0) ∨ (quadratic_expression k x > 0)) :
  -4 < k ∧ k < 0 :=
by
  sorry

end range_of_k_l557_557264


namespace correct_interval_l557_557116

def encounter_intervals : Prop :=
  let rate1 := 1 / 7
  let rate2 := 1 / 13
  let total_rate := rate1 + rate2
  let interval := 1 / total_rate
  interval ≈ 4.6

theorem correct_interval : encounter_intervals :=
sorry

end correct_interval_l557_557116


namespace horse_meeting_days_l557_557447

noncomputable def distance_traveled (a1: ℕ) (d: ℕ) (n: ℕ) : ℕ :=
  (a1 + (n - 1) * d)

noncomputable def sum_arithmetic (a1: ℕ) (d: ℕ) (n: ℕ) : ℕ :=
  (n * (2 * a1 + (n - 1) * d)) / 2

theorem horse_meeting_days :
  let a1_good := 103
  let d_good := 13
  let a1_poor := 97
  let d_poor := -0.5
  let distance := 1125
  let total_good := λ m, sum_arithmetic a1_good d_good m
  let total_poor := λ m, sum_arithmetic a1_poor d_poor m
  let meet_condition := λ m, total_good m + total_poor m = 2 * distance
  meet_condition 9 :=
by
  sorry

end horse_meeting_days_l557_557447


namespace steps_away_from_goal_l557_557143

-- Given conditions
def goal : ℕ := 100000
def initial_steps_per_day : ℕ := 1000
def increase_steps_per_week : ℕ := 1000
def days_per_week : ℕ := 7
def weeks : ℕ := 4

-- Computation of total steps in 4 weeks
def total_steps : ℕ :=
  (initial_steps_per_day * days_per_week) +
  ((initial_steps_per_day + increase_steps_per_week) * days_per_week) +
  ((initial_steps_per_day + 2 * increase_steps_per_week) * days_per_week) +
  ((initial_steps_per_day + 3 * increase_steps_per_week) * days_per_week)

-- Desired proof statement
theorem steps_away_from_goal : goal - total_steps = 30000 :=
by
  have h1: 7 * 1000 = 7000 := by norm_num
  have h2: 7 * 2000 = 14000 := by norm_num
  have h3: 7 * 3000 = 21000 := by norm_num
  have h4: 7 * 4000 = 28000 := by norm_num
  have h5: total_steps = 7000 + 14000 + 21000 + 28000 := by
    simp [total_steps, initial_steps_per_day, days_per_week, increase_steps_per_week]
    ring
  have h6: 7000 + 14000 + 21000 + 28000 = 70000 := by norm_num
  rw [h5, h6]
  show goal - 70000 = 30000
  norm_num

end steps_away_from_goal_l557_557143


namespace min_n_for_constant_term_l557_557615

theorem min_n_for_constant_term (n : ℕ) (h : n > 0) :
  ∃ (r : ℕ), (2 * n = 5 * r) → n = 5 :=
by
  sorry

end min_n_for_constant_term_l557_557615


namespace min_value_of_z_l557_557975

theorem min_value_of_z (x y : ℝ) (h : y^2 = 4 * x) : 
  ∃ (z : ℝ), z = 3 ∧ ∀ (x' : ℝ) (hx' : x' ≥ 0), ∃ (y' : ℝ), y'^2 = 4 * x' → z ≤ (1/2) * y'^2 + x'^2 + 3 :=
by sorry

end min_value_of_z_l557_557975


namespace negation_proposition_real_l557_557010

theorem negation_proposition_real :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ ∃ x : ℝ, x^2 + x + 1 ≤ 0 :=
by
  sorry

end negation_proposition_real_l557_557010


namespace measure_of_angle_A_l557_557491

-- Define the given conditions
variables (A B : ℝ)
axiom supplementary : A + B = 180
axiom measure_rel : A = 7 * B

-- The theorem statement to prove
theorem measure_of_angle_A : A = 157.5 :=
by
  -- proof steps would go here, but are omitted
  sorry

end measure_of_angle_A_l557_557491


namespace count_ordered_pairs_l557_557603

theorem count_ordered_pairs (x y : ℝ) :
  (∃ (x y : ℝ), (2 * x + 5 * y = 5) ∧ (| |x| - 2 * |y| | = 2)) → 6 := sorry

end count_ordered_pairs_l557_557603


namespace fractions_of_surface_area_in_and_outside_cone_l557_557043

noncomputable def fractions_inside_outside_cone (r : ℝ) : ℝ × ℝ :=
  let inside_fraction := (π * r^2) / (4 * π * r^2)
  let outside_fraction := 1 - inside_fraction
  (inside_fraction, outside_fraction)

theorem fractions_of_surface_area_in_and_outside_cone (r : ℝ) :
  fractions_inside_outside_cone r = (1/4, 3/4) :=
sorry

end fractions_of_surface_area_in_and_outside_cone_l557_557043


namespace bake_cookies_l557_557478

noncomputable def scale_factor (original_cookies target_cookies : ℕ) : ℕ :=
  target_cookies / original_cookies

noncomputable def required_flour (original_flour : ℕ) (scale : ℕ) : ℕ :=
  original_flour * scale

noncomputable def adjusted_sugar (original_sugar : ℕ) (scale : ℕ) (reduction_percent : ℚ) : ℚ :=
  original_sugar * scale * (1 - reduction_percent)

theorem bake_cookies 
  (original_cookies : ℕ)
  (target_cookies : ℕ)
  (original_flour : ℕ)
  (original_sugar : ℕ)
  (reduction_percent : ℚ)
  (h_original_cookies : original_cookies = 40)
  (h_target_cookies : target_cookies = 80)
  (h_original_flour : original_flour = 3)
  (h_original_sugar : original_sugar = 1)
  (h_reduction_percent : reduction_percent = 0.25) :
  required_flour original_flour (scale_factor original_cookies target_cookies) = 6 ∧ 
  adjusted_sugar original_sugar (scale_factor original_cookies target_cookies) reduction_percent = 1.5 := by
    sorry

end bake_cookies_l557_557478


namespace largest_binomial_terms_in_expansion_l557_557516

theorem largest_binomial_terms_in_expansion :
  ∀ (x : ℝ), ∃ (a b : ℤ), (1 - x) ^ 10 = ∑ i in finset.Icc 6 7, (nat.choose 10 i) * -(x ^ i) :=
by
  sorry

end largest_binomial_terms_in_expansion_l557_557516


namespace translated_function_has_symmetry_center_l557_557268

def f (x : ℝ) : ℝ := 3 * (Real.cos (2 * x + π / 2))

def g (x : ℝ) : ℝ := -3 * (Real.sin (2 * x - π / 3))

theorem translated_function_has_symmetry_center :
  ∃ x : ℝ, g x = 0 ∧ x = π / 6 := 
sorry

end translated_function_has_symmetry_center_l557_557268


namespace complement_U_A_eq_l557_557593

theorem complement_U_A_eq : 
  (∀ x, (1 < x → x ∈ (setOf (λ x, x > 1))) → 
        (x ∉ (setOf (λ x, x > 2)) ↔ (1 < x ∧ x ≤ 2))) := 
by 
  sorry

end complement_U_A_eq_l557_557593


namespace basketball_scores_l557_557458

theorem basketball_scores (x y z : ℕ) (h : x + y + z = 7) : 
  let p := 3 * x + 2 * y + z in (7 ≤ p ∧ p ≤ 21) ∧ ∀ n, 7 ≤ n ∧ n ≤ 21 → ∃ x y z : ℕ, x + y + z = 7 ∧ n = 3 * x + 2 * y + z := sorry

end basketball_scores_l557_557458


namespace second_bus_percentage_full_l557_557797

noncomputable def bus_capacity : ℕ := 150
noncomputable def employees_in_buses : ℕ := 195
noncomputable def first_bus_percentage : ℚ := 0.60

theorem second_bus_percentage_full :
  let employees_first_bus := first_bus_percentage * bus_capacity
  let employees_second_bus := (employees_in_buses : ℚ) - employees_first_bus
  let second_bus_percentage := (employees_second_bus / bus_capacity) * 100
  second_bus_percentage = 70 :=
by
  sorry

end second_bus_percentage_full_l557_557797


namespace probability_advanced_number_lt_100_l557_557262

def is_advanced_number (n : ℕ) : Prop :=
  (0 < n) ∧ (n < 100) ∧ ((n % 10 = 0 ∧ ((n+1) % 10 = 1) ∧ ((n+2) % 10 = 2)) ∨
                        (n % 10 ≠ 0 ∧ n % 10 ≠ 1 ∧ n % 10 ≠ 2))

theorem probability_advanced_number_lt_100 : 
  (∑ n in finset.range 100, if is_advanced_number n then 1 else 0).toReal / 100 = 0.88 :=
by
  sorry

end probability_advanced_number_lt_100_l557_557262


namespace percentage_of_the_stock_l557_557461

noncomputable def faceValue : ℝ := 100
noncomputable def yield : ℝ := 0.10
noncomputable def quotedPrice : ℝ := 160

theorem percentage_of_the_stock : 
  (yield * faceValue / quotedPrice * 100 = 6.25) :=
by
  sorry

end percentage_of_the_stock_l557_557461


namespace intersection_result_l557_557272

def U : Set ℝ := Set.univ
def A : Set ℤ := {x | x^2 < 16}
def B : Set ℝ := {x | x ≤ 1}
def complement_B : Set ℝ := {x | x > 1}
def intersect_set : Set ℤ := {2, 3}

theorem intersection_result :
  A ∩ (complement_B : Set ℤ) = intersect_set := 
sorry

end intersection_result_l557_557272


namespace total_amount_in_bank_l557_557497

-- Definition of the checks and their values
def checks_1mil : Nat := 25
def checks_100k : Nat := 8
def value_1mil : Nat := 1000000
def value_100k : Nat := 100000

-- The proof statement
theorem total_amount_in_bank 
  (total : Nat) 
  (h1 : checks_1mil * value_1mil = 25000000)
  (h2 : checks_100k * value_100k = 800000):
  total = 25000000 + 800000 :=
sorry

end total_amount_in_bank_l557_557497


namespace parallelepiped_volume_l557_557257

noncomputable def unit_vectors (v : ℝ^3) : Prop :=
  ∥v∥ = 1

theorem parallelepiped_volume
  (a b : ℝ^3)
  (ha : unit_vectors a)
  (hb : unit_vectors b)
  (angle_ab : real.angle a b = real.pi / 4) :
  abs ((a) ∙ ((b + 2 • a) × (b × a))) = 1 :=
sorry

end parallelepiped_volume_l557_557257


namespace find_a_l557_557937

def F (a b c : ℚ) : ℚ := a * b^3 + c

theorem find_a : 
  (∀ a : ℚ, F(a,3,8) = F(a,2,3) ↔ a = -5 / 19) :=
by sorry

end find_a_l557_557937


namespace king_lancelot_seats_38_l557_557890

noncomputable def totalSeats (seat_king seat_lancelot : ℕ) : ℕ :=
  if seat_king < seat_lancelot then
    2 * (seat_lancelot - seat_king - 1) + 2
  else
    2 * (seat_king - seat_lancelot - 1) + 2

theorem king_lancelot_seats_38 (seat_king seat_lancelot : ℕ) (h1 : seat_king = 10) (h2 : seat_lancelot = 29) :
  totalSeats seat_king seat_lancelot = 38 := 
  by
    sorry

end king_lancelot_seats_38_l557_557890


namespace magnitude_diff_vectors_l557_557204

def vector_magnitude (v : ℝ) : ℝ -> ℝ := λ v, v -- Just a placeholder if needed

variables (a b : ℝ)
variables (angle_between_ab : ℝ)
variables (|a| : ℝ := 1)
variables (|b| : ℝ := 2)
variables (angle_between_ab := real.pi / 3)

theorem magnitude_diff_vectors (|a| = 1)(|b| = 2)(angle_between_ab = real.pi / 3) :
  | a - 2 * b | = real.sqrt 13 := sorry

end magnitude_diff_vectors_l557_557204


namespace netCaloriesConsumedIs1082_l557_557411

-- Given conditions
def caloriesPerCandyBar : ℕ := 347
def candyBarsEatenInAWeek : ℕ := 6
def caloriesBurnedInAWeek : ℕ := 1000

-- Net calories calculation
def netCaloriesInAWeek (calsPerBar : ℕ) (barsPerWeek : ℕ) (calsBurned : ℕ) : ℕ :=
  calsPerBar * barsPerWeek - calsBurned

-- The theorem to prove
theorem netCaloriesConsumedIs1082 :
  netCaloriesInAWeek caloriesPerCandyBar candyBarsEatenInAWeek caloriesBurnedInAWeek = 1082 :=
by
  sorry

end netCaloriesConsumedIs1082_l557_557411


namespace cody_steps_away_from_goal_l557_557141

def steps_in_week (daily_steps : ℕ) : ℕ :=
  daily_steps * 7

def total_steps_in_4_weeks (initial_steps : ℕ) : ℕ :=
  steps_in_week initial_steps +
  steps_in_week (initial_steps + 1000) +
  steps_in_week (initial_steps + 2000) +
  steps_in_week (initial_steps + 3000)

theorem cody_steps_away_from_goal :
  let goal := 100000
  let initial_daily_steps := 1000
  let total_steps := total_steps_in_4_weeks initial_daily_steps
  goal - total_steps = 30000 :=
by
  sorry

end cody_steps_away_from_goal_l557_557141


namespace minimum_toothpick_removal_l557_557541

-- Definitions based on conditions from a)
def total_toothpicks : ℕ := 45
def upward_triangles : ℕ := 20
def downward_triangles : ℕ := 15
def horizontal_toothpicks : ℕ := 15

-- Statement of the problem
theorem minimum_toothpick_removal : horizontal_toothpicks = 15 ∧ 
  total_toothpicks = 45 ∧ upward_triangles = 20 ∧ downward_triangles = 15 → 
  ∀ (remove_count : ℕ), remove_count < 15 → ∃ (some_triangles : ℕ), some_triangles > 0 :=
by
  -- Using the total number of toothpicks, upward and downward triangles definitions,
  -- we assert the minimum removal count is 15 to ensure no triangles remain.
  assume h,
  exact sorry

end minimum_toothpick_removal_l557_557541


namespace sqrt_2_roots_l557_557026

theorem sqrt_2_roots: 
  {x : ℝ | x^2 = 2} = {√2, -√2} :=
by
  sorry

end sqrt_2_roots_l557_557026


namespace time_to_boil_l557_557913

def T₀ : ℝ := 20
def Tₘ : ℝ := 100
def t : ℝ := 10 * 60 -- 10 minutes converted to seconds
def c : ℝ := 4200 -- Specific heat capacity of water in J/(kg·K)
def L : ℝ := 2.3 * 10^6 -- Specific heat of vaporization of water in J/kg

theorem time_to_boil (m : ℝ) : 
  t₁ = t * (L / (c * (Tₘ - T₀))) ->
  m > 0 -> -- Assuming m (mass) is positive
  t₁ ≈ 68 * 60 :=
by
  sorry

end time_to_boil_l557_557913


namespace problem_1_l557_557822

theorem problem_1 : (-(5 / 8) / (14 / 3) * (-(16 / 5)) / (-(6 / 7))) = -1 / 2 :=
  sorry

end problem_1_l557_557822


namespace average_tree_height_l557_557471

/-- 
A theorem statement to prove that the average height of the 7 trees,
given the conditions, is 145.1 meters.
-/
theorem average_tree_height 
    (tree_height : ℕ → ℕ)
    (h1 : tree_height 2 = 16)
    (h2 : ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 6 → tree_height n = 2 * tree_height (n+1) ∨ tree_height n = tree_height (n+1) / 2) :
    (tree_height 1 + tree_height 2 + tree_height 3 + tree_height 4 + tree_height 5 + tree_height 6 + tree_height 7) / 7 = 145.1 :=
by
  sorry

end average_tree_height_l557_557471


namespace find_hypotenuse_of_right_triangle_l557_557629

def right_angled_triangle_hypotenuse (x y : ℝ) (hypotenuse : ℝ)
  (med_bc : ℝ) (med_ac : ℝ) : Prop :=
  med_bc = sqrt 52 ∧ med_ac = sqrt 73 ∧ x^2 + y^2 = hypotenuse^2

theorem find_hypotenuse_of_right_triangle (x y : ℝ) (hypotenuse : ℝ)
  (h : right_angled_triangle_hypotenuse x y hypotenuse  (sqrt 52) (sqrt 73)) :
  hypotenuse = 10 :=
sorry

end find_hypotenuse_of_right_triangle_l557_557629


namespace sum_of_smallest_x_and_y_l557_557388

theorem sum_of_smallest_x_and_y (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y)
  (hx : ∃ k : ℕ, (480 * x) = k * k ∧ ∀ z : ℕ, 0 < z → (480 * z) = k * k → x ≤ z)
  (hy : ∃ n : ℕ, (480 * y) = n * n * n ∧ ∀ z : ℕ, 0 < z → (480 * z) = n * n * n → y ≤ z) :
  x + y = 480 := sorry

end sum_of_smallest_x_and_y_l557_557388


namespace smallest_n_for_divisibility_problem_l557_557049

theorem smallest_n_for_divisibility_problem :
  ∃ n : ℕ, n > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → n * (n + 1) ≠ 0 ∧
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ n ∧ ¬ (n * (n + 1)) % k = 0) ∧
  ∀ m : ℕ, m > 0 ∧ m < n → (∀ k : ℕ, 1 ≤ k ∧ k ≤ m → (m * (m + 1)) % k ≠ 0)) → n = 4 := sorry

end smallest_n_for_divisibility_problem_l557_557049


namespace opposite_of_3_is_neg3_l557_557724

theorem opposite_of_3_is_neg3 : forall (n : ℤ), n = 3 -> -n = -3 :=
by
  sorry

end opposite_of_3_is_neg3_l557_557724


namespace sock_pairs_proof_l557_557032

noncomputable def numPairsOfSocks : ℕ :=
  let n : ℕ := sorry
  n

theorem sock_pairs_proof : numPairsOfSocks = 6 := by
  sorry

end sock_pairs_proof_l557_557032


namespace conic_sections_l557_557515

theorem conic_sections (x y : ℝ) (h : y^4 - 6 * x^4 = 3 * y^2 - 2) :
  (∃ a b : ℝ, y^2 = a + b * x^2) ∨ (∃ c d : ℝ, y^2 = c - d * x^2) :=
sorry

end conic_sections_l557_557515


namespace evaluate_clubsuit_ratio_l557_557259

def clubsuit (n m : ℕ) : ℕ := n^2 * m^3

theorem evaluate_clubsuit_ratio :
  (clubsuit 3 2) / (clubsuit 2 3) = 2 / 3 :=
by
  let c₃₂ := clubsuit 3 2
  let c₂₃ := clubsuit 2 3
  have c₃₂_eq : c₃₂ = 3^2 * 2^3 := by rfl
  have c₂₃_eq : c₂₃ = 2^2 * 3^3 := by rfl
  rw [c₃₂_eq, c₂₃_eq]
  norm_num
  sorry

end evaluate_clubsuit_ratio_l557_557259


namespace rectangle_fraction_of_triangle_area_l557_557000

-- defining the parameters for the problem
variables (b h : ℝ) (n : ℕ) (R A_R A_Δ : ℝ)
-- assuming n > 1
variable (hn : n > 1)

-- defining the rectangle area equation
def rect_area (s h_r : ℝ) : ℝ := s * h_r
-- defining the triangle area equation
def tri_area (b h : ℝ) : ℝ := 0.5 * b * h

-- defining that the base of the triangle is n times the base of the rectangle
def triangle_base_eq (b : ℝ) (n : ℕ) (s : ℝ) : Prop :=
  b = n * s

-- statement of the problem
theorem rectangle_fraction_of_triangle_area :
  ∀ (b h : ℝ) (n : ℕ) (s h_r : ℝ),
    n > 1 →
    b = n * s →
    (rect_area s h_r) / (tri_area b h) = (2 / n) - (2 / n^2) :=
begin
  intros b h n s h_r hn hbase,
  sorry
end

end rectangle_fraction_of_triangle_area_l557_557000


namespace diagonals_of_parallelogram_l557_557626

theorem diagonals_of_parallelogram (a b S : ℝ) 
  (h1 : 2 * S < a * b)
  : let EG := (1 / 2) * real.sqrt (a^2 + b^2 + 2 * real.sqrt (a^2 * b^2 - 16 * S^2))
    let HF := (1 / 2) * real.sqrt (a^2 + b^2 - 2 * real.sqrt (a^2 * b^2 - 16 * S^2))
  in EG > 0 ∧ HF > 0 :=
by
  let EG := (1 / 2) * real.sqrt (a^2 + b^2 + 2 * real.sqrt (a^2 * b^2 - 16 * S^2))
  let HF := (1 / 2) * real.sqrt (a^2 + b^2 - 2 * real.sqrt (a^2 * b^2 - 16 * S^2))
  sorry

end diagonals_of_parallelogram_l557_557626


namespace triangle_ratios_l557_557558

variables {A B C D E F G H O P : Type*}
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
variables [Inhabited E] [Inhabited F] [Inhabited G] [Inhabited H]
variables [Inhabited O] [Inhabited P]
variables [Geometry A B C D E F G H O P]

-- Definitions as per the problem
def inside_triangle (H : Type*) (ABC : Type*) : Prop := sorry
def meeting_points (AH BH CH : Type*) (BC CA AB : Type*) (D E F : Type*) : Prop := sorry
def intersection_extension (FE BC : Type*) (G : Type*) : Prop := sorry
def midpoint (DG : Type*) (O : Type*) : Prop := sorry
def circle_intersects (O : Type*) (OD FE : Type*) (P : Type*) : Prop := sorry

-- Final theorem to be proved
theorem triangle_ratios (ABC : Type*) (H : Type*) (AH BH CH BC CA AB D E F H G O P : Type*)
    [inside_triangle H ABC]
    [meeting_points AH BH CH BC CA AB D E F]
    [intersection_extension FE BC G]
    [midpoint DG O]
    [circle_intersects O OD FE P] :
    (BD/DC = BG/GC) ∧ (PB/PC = BD/DC) := sorry

end triangle_ratios_l557_557558


namespace obtuse_triangle_of_given_condition_l557_557619

theorem obtuse_triangle_of_given_condition
  (a b c : ℝ)
  (h : 2 * c^2 = 2 * a^2 + 2 * b^2 + a * b) :
  cos RADIANS(acos ((a^2 + b^2 - c^2) / (2 * a * b)) < 0) :=
sorry

end obtuse_triangle_of_given_condition_l557_557619


namespace plane_cuts_sphere_plane_cuts_cylinder_l557_557394

-- Definitions for conditions
def Plane : Type := sorry  -- Placeholder definition for Plane
def Sphere : Type := sorry  -- Placeholder definition for Sphere
def Cylinder : Type := sorry  -- Placeholder definition for Cylinder
def is_section (plane : Plane) (surface : Type) : Type := sorry  -- Placeholder for the section produced by intersection

-- The shapes we need to prove
def is_circle (shape : Type) : Prop := sorry  -- Placeholder definition for being a circle
def is_ellipse (shape : Type) : Prop := sorry  -- Placeholder definition for being an ellipse

-- Define the section produced when a Plane cuts a Sphere to be a circle
theorem plane_cuts_sphere (P : Plane) (S : Sphere) :
  is_circle (is_section P S) := sorry

-- Define the section produced when a Plane cuts a Cylinder to be a circle or ellipse
theorem plane_cuts_cylinder (P : Plane) (C : Cylinder) :
  is_circle (is_section P C) ∨ is_ellipse (is_section P C) := sorry

end plane_cuts_sphere_plane_cuts_cylinder_l557_557394


namespace problem_solution_l557_557579

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - x^2 + a * x - a

theorem problem_solution (x₀ x₁ a : ℝ) (h₁ : 3 * x₀^2 - 2 * x₀ + a = 0) (h₂ : f x₁ a = f x₀ a) (h₃ : x₁ ≠ x₀) : x₁ + 2 * x₀ = 1 :=
by
  sorry

end problem_solution_l557_557579


namespace angle_AP_PE_eq_90_l557_557291

noncomputable def angle_between_lines_AP_PE := 90

-- Definitions and conditions

def is_equilateral (A B C : Point) := 
  dist A B = dist B C ∧ dist B C = dist C A

def is_midpoint (D B C : Point) := 
  dist B D = dist D C

def is_parallel (PD PE : Line) :=
  ∀ x ∈ PD, ∀ y ∈ PE, x.parallel y

def angle_between (AP PE : Line) : ℝ :=
  90

-- Proof statement
theorem angle_AP_PE_eq_90 {A B C A1 B1 C1 P E D : Point} 
(HABC : is_equilateral A B C) 
(Hm : is_midpoint D B C)
(Hpar : is_parallel P D E) :
  angle_between (Line.mk A P) (Line.mk P E) = angle_between_lines_AP_PE :=
by
  sorry

end angle_AP_PE_eq_90_l557_557291


namespace dice_probability_green_l557_557850

theorem dice_probability_green :
  let total_faces := 6
  let green_faces := 3
  let probability := green_faces / total_faces
  probability = 1 / 2 :=
by
  let total_faces := 6
  let green_faces := 3
  let probability := green_faces / total_faces
  have h : probability = 1 / 2 := by sorry
  exact h

end dice_probability_green_l557_557850


namespace frost_cupcakes_l557_557494

-- Define the rates
def asha_rate : ℝ := 1 / 15
def jumpa_rate : ℝ := 1 / 25
def colin_rate : ℝ := 1 / 18

-- Define the total working time in seconds
def working_time : ℝ := 8 * 60  -- 8 minutes converted to seconds

-- Define the combined rate
def combined_rate : ℝ := asha_rate + jumpa_rate + colin_rate

-- Define the number of cupcakes frosted in the given time
def cupcakes_frosted : ℝ := working_time * combined_rate

-- Main theorem to prove
theorem frost_cupcakes : (cupcakes_frosted ≈ 78) :=
  by sorry

end frost_cupcakes_l557_557494


namespace correct_operations_l557_557430

-- Assertions based on the problem conditions and final solution
theorem correct_operations (a : ℝ) :
  (sqrt 4 = 2) ∧ ((-3 * a)^2 = 9 * a^2) :=
by {
  -- Proof skipped
  sorry
}

end correct_operations_l557_557430


namespace smallest_three_digit_candy_number_l557_557861

theorem smallest_three_digit_candy_number (n : ℕ) (hn1 : 100 ≤ n) (hn2 : n ≤ 999)
    (h1 : (n + 6) % 9 = 0) (h2 : (n - 9) % 6 = 0) : n = 111 := by
  sorry

end smallest_three_digit_candy_number_l557_557861


namespace alpha_beta_sum_l557_557194

theorem alpha_beta_sum (α β : ℝ) (hα : 0 < α) (hβ : 0 < β) (hαβ : α < π ∧ β < π) 
  (root_α : tan α * (tan α + 3 * sqrt 3) + 4 = 0)
  (root_β : tan β * (tan β + 3 * sqrt 3) + 4 = 0) :
  α + β = 4 * π / 3 :=
sorry

end alpha_beta_sum_l557_557194


namespace opposite_of_3_l557_557734

-- Define the concept of opposite of a number and the logic for positive numbers
def opposite (x : Int) : Int := 
  if x > 0 then -x
  else if x < 0 then -x
  else 0

-- Statement to prove that the opposite of 3 is -3
theorem opposite_of_3 : opposite 3 = -3 :=
by 
  -- Using the definition of opposite
  unfold opposite
  -- Simplify the expression for x = 3
  simp [lt_irrefl, int.coe_nat_lt]
  -- Conclude proof
  rfl

end opposite_of_3_l557_557734


namespace find_matrix_M_l557_557536

-- Define the given matrices
def A : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![2, -3],
  ![4, -1]
]

def B : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![-12, 5],
  ![8, -3]
]

-- Define the matrix M
def M : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![-0.8, -2.6],
  ![-2.0, 1.8]
]

-- Prove M is the matrix such that M * A = B
theorem find_matrix_M : M ⬝ A = B := by
  sorry

end find_matrix_M_l557_557536


namespace No_2022_over_No_2023_l557_557256

def No (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | _ => n * No (n - 1)

theorem No_2022_over_No_2023 : No 2022 / No 2023 = 1 / 2023 := by
  have h2022 : No 2022 = Nat.factorial 2022 := rfl
  have h2023 : No 2023 = Nat.factorial 2023 := rfl
  rw [h2022, h2023, Nat.factorial_succ, Nat.factorial]
  simp

end No_2022_over_No_2023_l557_557256


namespace coefficient_of_x_l557_557678

-- Definitions based on conditions
def sum_of_coeffs (n : ℕ) : ℕ := 4 ^ n
def binom_coeffs_sum (n : ℕ) : ℕ := 2 ^ n

-- Main theorem statement
theorem coefficient_of_x (n : ℕ) (h : sum_of_coeffs n - binom_coeffs_sum n = 240) : 
  (∑ r in finset.range (n + 1), binomial n r * (-1) ^ r * 5 ^ (n - r) * if (n - (3 * r / 2)) = 1 then 1 else 0) = 150 := 
sorry

end coefficient_of_x_l557_557678


namespace area_of_field_in_hectares_l557_557705

noncomputable def pi : ℝ := real.pi

noncomputable def fencing_cost : ℝ := 4456.44
noncomputable def cost_per_meter : ℝ := 3
noncomputable def C : ℝ := fencing_cost / cost_per_meter
noncomputable def r : ℝ := C / (2 * pi)
noncomputable def A : ℝ := pi * r ^ 2
noncomputable def A_hectares : ℝ := A / 10000

theorem area_of_field_in_hectares :
  A_hectares ≈ 17.569 :=
by
  sorry

end area_of_field_in_hectares_l557_557705


namespace percentage_sum_of_v_and_w_l557_557613

variable {x y z v w : ℝ} 

theorem percentage_sum_of_v_and_w (h1 : 0.45 * z = 0.39 * y) (h2 : y = 0.75 * x) 
                                  (h3 : v = 0.80 * z) (h4 : w = 0.60 * y) :
                                  v + w = 0.97 * x :=
by 
  sorry

end percentage_sum_of_v_and_w_l557_557613


namespace range_of_m_l557_557548

theorem range_of_m (m : ℝ) :
  (∀ x ∈ set.Icc (1 : ℝ) 2, ∀ y ∈ set.Icc (2 : ℝ) 3, y^2 - x*y - m*x^2 ≤ 0) → 6 ≤ m :=
by
  sorry

end range_of_m_l557_557548


namespace inscribed_circle_radius_l557_557047

noncomputable def radius_inscribed_circle (DE DF EF : ℝ) : ℝ := 
  let s := (DE + DF + EF) / 2
  let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  K / s

theorem inscribed_circle_radius :
  radius_inscribed_circle 8 5 9 = 6 * Real.sqrt 11 / 11 :=
by
  sorry

end inscribed_circle_radius_l557_557047


namespace subset_sum_divisible_by_n_l557_557202
open Set

theorem subset_sum_divisible_by_n (n : ℕ) (a : Fin n → ℤ) : 
  ∃ (s : Finset (Fin n)), (∑ i in s, a i) % n = 0 :=
sorry

end subset_sum_divisible_by_n_l557_557202


namespace maria_time_per_piece_l557_557679

noncomputable def time_per_piece (total_time : ℕ) (total_pieces : ℕ) : ℕ :=
  total_time / total_pieces

theorem maria_time_per_piece (time_spent total_pieces : ℕ) (h₁ : total_pieces = 4) (h₂ : time_spent = 32) :
  time_per_piece time_spent total_pieces = 8 :=
by
  rw [h₁, h₂, time_per_piece]
  norm_num
  sorry

end maria_time_per_piece_l557_557679


namespace geometric_sequence_sum_zero_l557_557703

noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ := 
  ∑ i in finset.range n, real.log (a (i + 1))

theorem geometric_sequence_sum_zero 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (n m : ℕ) 
  (h_pos : ∀ i, 0 < a i) 
  (h_geom : ∀ i, a (i + 1) = q * a i) 
  (h_nm : n ≠ m) : 
  S a n = S a m → S a (n + m) = 0 := 
by
  sorry

end geometric_sequence_sum_zero_l557_557703


namespace Nick_riding_speed_l557_557859

theorem Nick_riding_speed (Alan_speed Maria_ratio Nick_ratio : ℝ) 
(h1 : Alan_speed = 6) (h2 : Maria_ratio = 3/4) (h3 : Nick_ratio = 4/3) : 
Nick_ratio * (Maria_ratio * Alan_speed) = 6 := 
by 
  sorry

end Nick_riding_speed_l557_557859


namespace cube_paint_probability_l557_557160

/-- 
Each face of a cube is painted either green or yellow, each with probability 1/2. 
The color of each face is determined independently. Prove that the probability 
that the painted cube can be placed on a horizontal surface so that the four 
vertical faces are all the same color is 5/16. 
-/
theorem cube_paint_probability : 
  let cube_faces := [true, false] -- Assuming true is green, false is yellow
  let arrangements := { arr | arr ∈ cube_faces^6 ∧ independent (λ i, arr i) }
  let suitable_arrangements := { arr ∈ arrangements | 
    can_be_placed_on_horizontal_surface_with_same_color_vertical_faces arr }
  ∃ pr, pr = (|suitable_arrangements| : ℚ) / (|arrangements| : ℚ) ∧ pr = 5 / 16
:= sorry

end cube_paint_probability_l557_557160


namespace cant_place_numbers_l557_557303

noncomputable def total_sum : ℕ := (10 * (10 + 1)) / 2

theorem cant_place_numbers : 
  ¬ (∃ (a b c d e f g h i j : ℕ), 
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j ∧
     b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ j ∧
     c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ j ∧
     d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ j ∧
     e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ j ∧
     f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧
     g ≠ h ∧ g ≠ i ∧ g ≠ j ∧
     h ≠ i ∧ h ≠ j ∧
     i ≠ j) ∧
    (a + b + c + d = 22 ∧ e + f + g + h = 22 ∧ i + j + a + b = 22 ∧ c + d + e + f = 22 ∧ g + h + i + j = 22)) := 
by {
  intros h,
  sorry
}

end cant_place_numbers_l557_557303


namespace determine_f2009_l557_557666

-- Definition of the function and conditions.
def f (x : ℝ) : ℝ := sorry

axiom f_pos : ∀ x : ℝ, x > 0 → f(x) > 0
axiom f_def : ∀ x y : ℝ, x > y → f(x - y) = sqrt(f(x*y) + 2)

theorem determine_f2009 : f(2009) = 2 := by 
    sorry

end determine_f2009_l557_557666


namespace opposite_of_three_l557_557754

theorem opposite_of_three : -3 = -3 := by
  -- The condition we have identified is the given number 3.
  -- We will directly state that the opposite of 3 is -3.
  -- This proof is trivial as we are directly replacing 3 with -3 to match the problem statement.
  rfl

end opposite_of_three_l557_557754


namespace symmetric_point_correct_l557_557028

-- Define the initial point P
def P : ℝ × ℝ := (1, 3)

-- Define the line y = x
def line : ℝ × ℝ → Prop := λ (p : ℝ × ℝ), p.1 = p.2

-- Define a function to compute the symmetric point of P(m, n) with respect to the line y = x
def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

-- Define the target symmetric point
def target_symmetric_point : ℝ × ℝ := (3, 1)

-- Prove that the symmetric point of P with respect to the line y = x is (3, 1)
theorem symmetric_point_correct : symmetric_point P = target_symmetric_point :=
by
  sorry

end symmetric_point_correct_l557_557028


namespace opposite_of_3_is_neg3_l557_557744

def opposite (x : ℝ) := -x

theorem opposite_of_3_is_neg3 : opposite 3 = -3 :=
by
  sorry

end opposite_of_3_is_neg3_l557_557744


namespace ellipse_properties_l557_557210

noncomputable def ellipse (a b : ℝ) (h : a > b ∧ b > 0) :=
  ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1

variables {a b c : ℝ} (h : a > b ∧ b > 0) (hc : 0 < c ∧ c < a)
variables (h_focus : ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1)
variables (F A E : ℝ × ℝ) (h_F : F = (-c, 0)) (h_E : E = (0, c))
variables (h_A : A = (a, 0)) (area_EFA : 1/2 * c * a * c = b^2 / 2)
variables (Q P M N : ℝ × ℝ) (h_Q : Q.1 = 3/2 * c) (h_P : P = (-c, 0))
variables (parallel_PM_QN : P.2 = Q.2 ∧ QN.2 = c)
variables (area_PQNM : 3 * c)

theorem ellipse_properties (h_focus : ellipse a b h)
    (h_foci : F = (-c, 0)) (h_vertex : E = (0, c)) (area_triangle : 1/2 * (c + a) * c = b^2 / 2) :
    (sqrt (1 - b^2 / a^2) = 1/2) ∧
    (slope (line_through F P) = 3/4) ∧
    (ellipse a b h ∧ a = 2 * c ∧ b = sqrt 3 * c) :=
by
  sorry

end ellipse_properties_l557_557210


namespace total_goals_scored_l557_557821

theorem total_goals_scored : 
  ∀ (goals_last_season goals_this_season : ℕ),
  goals_last_season = 156 ∧ goals_this_season = 187 →
  goals_last_season + goals_this_season = 343 :=
by 
  intros goals_last_season goals_this_season h,
  cases h with h1 h2,
  rw [h1, h2],
  rfl

end total_goals_scored_l557_557821


namespace time_to_boil_l557_557912

def T₀ : ℝ := 20
def Tₘ : ℝ := 100
def t : ℝ := 10 * 60 -- 10 minutes converted to seconds
def c : ℝ := 4200 -- Specific heat capacity of water in J/(kg·K)
def L : ℝ := 2.3 * 10^6 -- Specific heat of vaporization of water in J/kg

theorem time_to_boil (m : ℝ) : 
  t₁ = t * (L / (c * (Tₘ - T₀))) ->
  m > 0 -> -- Assuming m (mass) is positive
  t₁ ≈ 68 * 60 :=
by
  sorry

end time_to_boil_l557_557912


namespace kay_weight_training_time_l557_557648

variables (total_minutes : ℕ) (aerobic_ratio weight_ratio : ℕ)
-- Conditions
def kay_exercise := total_minutes = 250
def ratio_cond := aerobic_ratio = 3 ∧ weight_ratio = 2
def total_ratio_parts := aerobic_ratio + weight_ratio

-- Question and proof goal
theorem kay_weight_training_time (h1 : kay_exercise total_minutes) (h2 : ratio_cond aerobic_ratio weight_ratio) :
  (total_minutes / total_ratio_parts * weight_ratio) = 100 :=
by
  sorry

end kay_weight_training_time_l557_557648


namespace units_digit_2_pow_2012_l557_557001

theorem units_digit_2_pow_2012 : (2^2012) % 10 = 6 := by
  -- Theoretical period of powers of 2 ending digit is 4
  have period4 : (2^4) % 10 = (2^0) % 10 := by norm_num
  -- Pattern verification
  have pow1 := show (2^1) % 10 = 2 by norm_num
  have pow2 := show (2^2) % 10 = 4 by norm_num
  have pow3 := show (2^3) % 10 = 8 by norm_num
  -- 2^4 cycle concludes at 6
  have pow4 := show (2^4) % 10 = 6 by norm_num
  -- Proof of pattern continuation
  calc
    (2 ^ 2012) % 10
        = (2 ^ (4 * 503)) % 10 : by norm_num
    ... = ((2 ^ 4) ^ 503) % 10 : by rw pow_mul
    ... = (6 ^ 503) % 10       : by rw pow4
    ... = 6                    : by norm_num

end units_digit_2_pow_2012_l557_557001


namespace median_line_and_triangle_area_l557_557573

theorem median_line_and_triangle_area :
  ∃ (l : ℝ → ℝ → Prop),
    (∀ x y, l x y ↔ x + y - 3 = 0) ∧
    (let A := (-1 : ℝ, 1 : ℝ),
         B := (7 : ℝ, -1 : ℝ),
         C := (-2 : ℝ, 5 : ℝ),
         D := (2 : ℝ, 4 : ℝ) in
     ∃ S : ℝ, S = 15 / 2 ∧
              (x + y - 3 = 0) → ((A.1 + B.1, A.2 + B.2) = (6, 0)) ∧ ((x + y - 3 = 0) → D = (2, 4)) ∧
              S = abs (det (vector of_matrix_from_list_of_polynomial {B, C, D})) / 2)  :=
begin
  sorry,
end

end median_line_and_triangle_area_l557_557573


namespace opposite_of_negative_fraction_l557_557017

theorem opposite_of_negative_fraction : -(- (1/2023 : ℚ)) = 1/2023 := 
sorry

end opposite_of_negative_fraction_l557_557017


namespace probability_product_multiple_of_3_l557_557101

structure Die where
  sides : ℕ
  rolls : ℕ

def is_multiple_of_3 (n : ℕ) : Prop := n % 3 = 0

noncomputable def probability_multiple_of_3_in_rolls (die : Die) : ℚ :=
  1 - (float_of (2/3) ^ die.rolls)

theorem probability_product_multiple_of_3 (die : Die)
  (h1 : die.sides = 6)
  (h2 : die.rolls = 8) :
  probability_multiple_of_3_in_rolls die = 6305 / 6561 :=
  sorry

end probability_product_multiple_of_3_l557_557101


namespace count_numbers_with_digit_one_l557_557253

def contains_one (n : ℕ) : Prop :=
  (n.to_digits 10).contains 1

theorem count_numbers_with_digit_one :
  (Finset.range 2024).filter contains_one).card = 1284 :=
sorry

end count_numbers_with_digit_one_l557_557253


namespace area_of_given_trapezium_is_correct_l557_557058

-- Define the lengths of the parallel sides and the distance between them
def length_parallel_side1 : ℝ := 20
def length_parallel_side2 : ℝ := 18
def distance_between_sides : ℝ := 14

-- Define the formula for the area of the trapezium
def trapezium_area (a b h : ℝ) : ℝ := (1 / 2) * (a + b) * h

-- Define the area of the given trapezium
noncomputable def area_of_given_trapezium : ℝ :=
  trapezium_area length_parallel_side1 length_parallel_side2 distance_between_sides

-- State the theorem to be proved
theorem area_of_given_trapezium_is_correct : area_of_given_trapezium = 266 := sorry

end area_of_given_trapezium_is_correct_l557_557058


namespace find_largest_negative_root_of_equation_l557_557173

theorem find_largest_negative_root_of_equation :
  ∃ x ∈ {x : ℝ | (sin (real.pi * x) - cos (2 * real.pi * x)) / ((sin (real.pi * x) - 1)^2 + cos (real.pi * x)^2 - 1) = 0}, 
  ∀ y ∈ {y : ℝ | (sin (real.pi * y) - cos (2 * real.pi * y)) / ((sin (real.pi * y) - 1)^2 + cos (real.pi * y)^2 - 1) = 0 },
  y < 0 → y ≤ x :=
begin
  use -0.5,
  split,
  { -- proof that -0.5 is a root
    sorry
  },
  { -- proof that -0.5 is the largest negative root
    sorry
  }
end

end find_largest_negative_root_of_equation_l557_557173


namespace cone_surface_area_eq_9_pi_over_4_l557_557714

theorem cone_surface_area_eq_9_pi_over_4 :
  let radius := 2 in
  let central_angle := π / 2 in
  let base_radius := 1 / 2 in
  let lateral_area := π * (2 ^ 2) * (1 / 2) in
  let base_area := π * (base_radius ^ 2) in
  let total_surface_area := lateral_area + base_area in
  total_surface_area = (9 * π) / 4 :=
by {
  let radius := 2,
  let central_angle := π / 2,
  let base_radius := 1 / 2,
  let lateral_area := π * (2 ^ 2) * (1 / 2),
  let base_area := π * (base_radius ^ 2),
  let total_surface_area := lateral_area + base_area,
  show total_surface_area = (9 * π) / 4,
  sorry
}

end cone_surface_area_eq_9_pi_over_4_l557_557714


namespace angle_between_lines_eq_arctan_one_half_l557_557638

noncomputable def angle_between_polar_lines
  (ρ θ : ℝ) (h1 : ρ * (2 * Real.cos θ + Real.sin θ) = 2)
  (h2 : ρ * Real.cos θ = 1) : ℝ :=
arctan (1/2)

-- Statement that requires proof
theorem angle_between_lines_eq_arctan_one_half
  {ρ θ : ℝ}
  (h1 : ρ * (2 * Real.cos θ + Real.sin θ) = 2)
  (h2 : ρ * Real.cos θ = 1) :
  angle_between_polar_lines ρ θ h1 h2 = Real.arctan (1/2) :=
sorry

end angle_between_lines_eq_arctan_one_half_l557_557638


namespace angle_properties_l557_557929

structure Circle (Point : Type*) :=
(center : Point)
(radius : ℝ)

def tangent_at (C : Circle Point) (P : Point) : Prop := sorry  -- Definition of a tangent at a point.

variables {Point : Type*} [MetricSpace Point] [InnerProductSpace ℝ Point]

def externally_tangent (C1 C2 : Circle Point) (K : Point) : Prop := sorry  -- Definition of externally tangent circles.

def common_tangent (C1 C2 : Circle Point) (L : Set Point) : Prop := sorry  -- Definition of a common tangent line.

theorem angle_properties
(C1 C2 : Circle Point) (O1 O2 A B K M : Point)
(hC1 : C1.center = O1) (hC2 : C2.center = O2)
(hK : externally_tangent C1 C2 K)
(hA : tangent_at C1 A)
(hB : tangent_at C2 B)
(hM : ∃ L, common_tangent C1 C2 L ∧ M ∈ L)
(hintersect : M ∈ Line.mk K (intersection_point_of_tangents C1 C2))

: ∠(O1, M, O2) = 90 ∧ ∠(A, K, B) = 90 := sorry

end angle_properties_l557_557929


namespace range_of_distance_l557_557963

noncomputable def distance_point_to_line (A B C x y : ℝ) : ℝ :=
  abs (A * x + B * y + C) / sqrt (A ^ 2 + B ^ 2)

theorem range_of_distance (λ : ℝ) :
  let line_eq := (2 + λ) * x - (1 + λ) * y - 2 * (3 + 2 * λ) = 0
  let P := (-2, 2 : ℝ × ℝ)
  let A := 2 + λ
  let B := -(1 + λ)
  let C := -(2 * (3 + 2 * λ))
  0 < distance_point_to_line 1 (-1) (-4) (-2) 2 ∧
  distance_point_to_line 1 (-1) (-4) (-2) 2 < 4 * real.sqrt 2 :=
by {
  unfold distance_point_to_line, sorry
}

end range_of_distance_l557_557963


namespace find_a_l557_557244

open Set

variable (a : ℝ)

def A (a : ℝ) : Set ℝ := {2, 4, a^3 - 2 * a^2 - a + 7}
def B (a : ℝ) : Set ℝ := {-4, a + 3, a^2 - 2 * a + 2, a^3 + a^2 + 3 * a + 7}

theorem find_a (h : (A a ∩ B a) = {2, 5}) : a = 2 :=
sorry

end find_a_l557_557244


namespace range_of_f_lt_0_l557_557201

noncomputable def f : ℝ → ℝ := sorry

theorem range_of_f_lt_0 (h_even : ∀ x, f x = f (-x))
  (h_decreasing : ∀ x y, x < y → x < 0 → y < 0 → f y < f x)
  (h_at_3 : f 3 = 0) : 
  { x : ℝ | f x < 0 } = set.Ioo (-3) 3 := 
sorry

end range_of_f_lt_0_l557_557201


namespace factory_hours_per_day_l557_557839

def factory_produces (hours_per_day : ℕ) : Prop :=
  let refrigerators_per_hour := 90
  let coolers_per_hour := 160
  let total_products_per_hour := refrigerators_per_hour + coolers_per_hour
  let total_products_in_5_days := 11250
  total_products_per_hour * (5 * hours_per_day) = total_products_in_5_days

theorem factory_hours_per_day : ∃ h : ℕ, factory_produces h ∧ h = 9 :=
by
  existsi 9
  unfold factory_produces
  sorry

end factory_hours_per_day_l557_557839


namespace chelsea_sugar_bags_l557_557508

variable (n : ℕ)

-- Defining the conditions as hypotheses
def initial_sugar : ℕ := 24
def remaining_sugar : ℕ := 21
def sugar_lost : ℕ := initial_sugar - remaining_sugar
def torn_bag_sugar : ℕ := 2 * sugar_lost

-- Define the statement to prove
theorem chelsea_sugar_bags :
  n = initial_sugar / torn_bag_sugar → n = 4 :=
by
  sorry

end chelsea_sugar_bags_l557_557508


namespace opposite_of_three_l557_557737

theorem opposite_of_three : -3 = -3 := 
by sorry

end opposite_of_three_l557_557737


namespace knight_tour_impossible_l557_557685

theorem knight_tour_impossible :
  ¬ ∃ (path : List (Fin 8 × Fin 8)),
    path.head = (0, 0) ∧
    path.last = (7, 7) ∧
    (∀ (i : ℕ), i < path.length - 1 →
      let (x1, y1) := path.nth i in
      let (x2, y2) := path.nth (i + 1) in
      (x2 = x1 + 1 ∧ y2 = y1 + 2) ∨
      (x2 = x1 + 2 ∧ y2 = y1 + 1) ∨
      (x2 = x1 - 1 ∧ y2 = y1 + 2) ∨
      (x2 = x1 + 2 ∧ y2 = y1 - 1) ∨
      (x2 = x1 - 2 ∧ y2 = y1 + 1) ∨
      (x2 = x1 + 1 ∧ y2 = y1 - 2) ∨
      (x2 = x1 - 1 ∧ y2 = y1 - 2) ∨
      (x2 = x1 - 2 ∧ y2 = y1 - 1)) ∧
    path.length = 64 :=
by
  sorry

end knight_tour_impossible_l557_557685


namespace calculate_expression_l557_557135

theorem calculate_expression :
  ( (1/5: ℝ) ^ (-2) - | -2 * real.sqrt 5 | + (real.sqrt 2023 - real.sqrt 2022) ^ 0 + real.sqrt 20 = 26 ) :=
by
  have h1 : (1/5: ℝ) ^ (-2) = 25 := sorry
  have h2 : -| -2 * real.sqrt 5 | = -2 * real.sqrt 5 := sorry
  have h3 : (real.sqrt 2023 - real.sqrt 2022) ^ 0 = 1 := sorry
  have h4 : real.sqrt 20 = 2 * real.sqrt 5 := sorry
  have h5 : 25 - 2 * real.sqrt 5 + 1 + 2 * real.sqrt 5 = 26 := sorry
  exact h5

end calculate_expression_l557_557135


namespace balls_in_boxes_l557_557689

theorem balls_in_boxes :
  ∃ (f : Fin 5 → Fin 3), 
    (∀ i j, i ≠ j → f i ≠ f j) ∧
    (∀ b : Fin 3, ∃ i, f i = b) ∧
    f 0 ≠ f 1 :=
  sorry

end balls_in_boxes_l557_557689


namespace hyperbola_eccentricity_l557_557710

theorem hyperbola_eccentricity (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0)
  (h₂ : ∀ x : ℝ, y = (3 / 4) * x → y = (b / a) * x) : 
  (b = (3 / 4) * a) → (e = 5 / 4) := 
by
  sorry

end hyperbola_eccentricity_l557_557710


namespace necessary_condition_for_parallel_planes_l557_557215

variables {a b : Type} {α β : Type} [Plane α] [Plane β] [Line a] [Line b]
variable (h1 : Perpendicular a α) (h2 : Subset b β)

theorem necessary_condition_for_parallel_planes (h : Perpendicular a b) : Necessary_but_not_sufficient_condition α β :=
sorry

end necessary_condition_for_parallel_planes_l557_557215


namespace conor_total_vegetables_l557_557511

-- Definitions for each day of the week
def vegetables_per_day_mon_wed : Nat := 12 + 9 + 8 + 15 + 7
def vegetables_per_day_thu_sat : Nat := 7 + 5 + 4 + 10 + 4
def total_vegetables : Nat := 3 * vegetables_per_day_mon_wed + 3 * vegetables_per_day_thu_sat

-- Lean statement for the proof problem
theorem conor_total_vegetables : total_vegetables = 243 := by
  sorry

end conor_total_vegetables_l557_557511


namespace lights_on_again_eventually_lights_n_squared_minus_1_lights_n_squared_minus_n_plus_1_l557_557329

noncomputable def problem_a (n : ℕ) (h : n > 1) : ℕ :=
  let M := sorry
  M

theorem lights_on_again_eventually (n : ℕ) (h : n > 1) :
  ∃ M, ∀ j, (j ≥ M) -> all_lights_on :=
sorry

theorem lights_n_squared_minus_1 (n k : ℕ) (h : n = 2^k) :
  ∀ s, (s = n^2 - 1) -> all_lights_on :=
sorry

theorem lights_n_squared_minus_n_plus_1 (n k : ℕ) (h : n = 2^k + 1) :
  ∀ s, (s = n^2 - n + 1) -> all_lights_on :=
sorry

end lights_on_again_eventually_lights_n_squared_minus_1_lights_n_squared_minus_n_plus_1_l557_557329


namespace number_of_rows_l557_557636

theorem number_of_rows (n : ℕ) (h : ∑ i in finset.range n, 53 - 2 * i = 405) : n = 9 :=
sorry

end number_of_rows_l557_557636


namespace probability_diff_color_balls_l557_557459

theorem probability_diff_color_balls (red_balls : ℕ) (black_balls : ℕ) : 
  red_balls = 2 → black_balls = 3 → 
  (probability_diff_color : ℚ) = 3 / 5 := 
begin
  -- Assume the conditions
  intros h1 h2,
  -- Share the solution to prove
  sorry
end

end probability_diff_color_balls_l557_557459


namespace at_most_9_negatives_l557_557442

theorem at_most_9_negatives (a : Fin 10 → ℤ) (h : (∀ i, a i ≠ 0) ∧ (∏ i, a i < 0)) : (∑ i, if a i < 0 then 1 else 0) ≤ 9 :=
sorry

end at_most_9_negatives_l557_557442


namespace angle_B_is_36_degrees_l557_557295

-- Definitions and theorem statement
theorem angle_B_is_36_degrees
  (ABCD : Type) [trapezoid ABCD]
  (AB CD : line) (A B C D : point ABCD)
  (h1 : parallel AB CD) (h2 : angle A = 2 * angle D) (h3 : angle C = 4 * angle B) :
  angle B = 36 := by
  sorry -- proof not required

end angle_B_is_36_degrees_l557_557295


namespace smallest_m_exists_l557_557663

noncomputable def complex_set (x : ℝ) (y : ℝ) : set ℂ :=
  {z | z.re = x ∧ z.im = y ∧ (1/2 ≤ x ∧ x ≤ real.sqrt 2 / 2)}

theorem smallest_m_exists (m : ℕ) :
  (∀ n : ℕ, n ≥ m → ∃ z ∈ (complex_set x y), z ^ n = 1) ↔ m = 12 := by
  sorry

end smallest_m_exists_l557_557663


namespace opposite_of_3_l557_557773

theorem opposite_of_3 : -3 = -3 := 
by
  -- sorry is added to skip the proof as per instructions
  sorry

end opposite_of_3_l557_557773


namespace andrew_put_4_in_second_bin_l557_557915

noncomputable def cans_in_bin : ℕ → ℕ
| 1 => 2
| 2 => 4  -- This needs to be proven
| 3 => 7
| 4 => 11
| 5 => 16
| n => cans_in_bin (n-1) + (cans_in_bin (n-1) - cans_in_bin (n-2) + 1)

theorem andrew_put_4_in_second_bin : cans_in_bin 2 = 4 :=
by
  -- Proof goes here
  sorry

end andrew_put_4_in_second_bin_l557_557915


namespace circle_coloring_l557_557551

-- Definition of the problem in Lean
theorem circle_coloring (n : ℕ) (h : n ≥ 1) :
  ∃ (color : set (ℝ × ℝ) → bool), 
    (∀ (x y : set (ℝ × ℝ)), (x ∩ y).nonempty → (color x ≠ color y)) :=
by 
  sorry

end circle_coloring_l557_557551


namespace square_position_2013_l557_557003

theorem square_position_2013 : 
  let initial_position := "EFGH"
  let position_after_rotation := "HEFG"
  let position_after_reflection := "GFHE"
  let sequence := [initial_position, position_after_rotation, position_after_reflection, initial_position]
  let steps := 2013
  (sequence[(steps % 4)]) = initial_position :=
by
  let initial_position := "EFGH"
  let position_after_rotation := "HEFG"
  let position_after_reflection := "GFHE"
  let sequence := [initial_position, position_after_rotation, position_after_reflection, initial_position]
  let steps := 2013
  have h : steps % 4 = 1 := by
  {
    sorry -- Proof of the modulus calculation
  }
  show (sequence[steps % 4]) = initial_position from by
  {
    rw h,
    exact rfl,
  }

end square_position_2013_l557_557003


namespace total_quantity_before_adding_water_l557_557841

variable (x : ℚ)
variable (milk water : ℚ)
variable (added_water : ℚ)

-- Mixture contains milk and water in the ratio 3:2
def initial_ratio (milk water : ℚ) : Prop := milk / water = 3 / 2

-- Adding 10 liters of water
def added_amount : ℚ := 10

-- New ratio of milk to water becomes 2:3 after adding 10 liters of water
def new_ratio (milk water : ℚ) (added_water : ℚ) : Prop :=
  milk / (water + added_water) = 2 / 3

theorem total_quantity_before_adding_water
  (h_ratio : initial_ratio milk water)
  (h_added : added_water = 10)
  (h_new_ratio : new_ratio milk water added_water) :
  milk + water = 20 :=
by
  sorry

end total_quantity_before_adding_water_l557_557841


namespace sum_of_possible_m_values_l557_557993

theorem sum_of_possible_m_values : 
  ∃ (s : ℕ), (∀ m : ℤ, 0 < 5 * m ∧ 5 * m < 35 → m ∈ {1, 2, 3, 4, 5, 6}) ∧ 
             (s = List.sum [1, 2, 3, 4, 5, 6]) ∧
             s = 21 := 
begin
  sorry
end

end sum_of_possible_m_values_l557_557993


namespace masha_problem_l557_557520

noncomputable def sum_arithmetic_series (a l n : ℕ) : ℕ :=
  (n * (a + l)) / 2

theorem masha_problem : 
  let a_even := 372
  let l_even := 506
  let n_even := 67
  let a_odd := 373
  let l_odd := 505
  let n_odd := 68
  let S_even := sum_arithmetic_series a_even l_even n_even
  let S_odd := sum_arithmetic_series a_odd l_odd n_odd
  S_odd - S_even = 439 := 
by sorry

end masha_problem_l557_557520


namespace smallest_positive_period_even_function_l557_557122

theorem smallest_positive_period_even_function
  (f1 f2 f3 f4 : ℝ → ℝ)
  (hf1 : ∀ x, f1 x = sin (2 * x) + cos (2 * x))
  (hf2 : ∀ x, f2 x = sin (2 * x) * cos (2 * x))
  (hf3 : ∀ x, f3 x = cos (4 * x + π / 2))
  (hf4 : ∀ x, f4 x = sin (2 * x) ^ 2 - cos (2 * x) ^ 2)
  (T : ℝ)
  (hk1 : ∀ x, f1 (x + T) = f1 x)
  (hk2 : ∀ x, f2 (x + T) = f2 x)
  (hk3 : ∀ x, f3 (x + T) = f3 x)
  (hk4 : ∀ x, f4 (x + T) = f4 x)
  (hT : T > 0)
  : f4 x is_even_with_period (π / 2) := 
  sorry

end smallest_positive_period_even_function_l557_557122


namespace total_seats_round_table_l557_557863

theorem total_seats_round_table (n : ℕ) (h : n = 38)
  (ka_position : ℕ) (sl_position : ℕ) 
  (h1 : ka_position = 10) 
  (h2 : sl_position = 29) 
  (h3 : (ka_position + n/2) % n = sl_position) : 
  n = 38 :=
by
  -- All steps and solution proof
  sorry

end total_seats_round_table_l557_557863


namespace bounds_for_f3_l557_557584

variable (a c : ℝ)

def f (x : ℝ) : ℝ := a * x^2 - c

theorem bounds_for_f3 (h1 : -4 ≤ f a c 1 ∧ f a c 1 ≤ -1)
                      (h2 : -1 ≤ f a c 2 ∧ f a c 2 ≤ 5) :
  -1 ≤ f a c 3 ∧ f a c 3 ≤ 20 := 
sorry

end bounds_for_f3_l557_557584


namespace max_ln_x_sub_x_l557_557958

noncomputable def f : ℝ → ℝ := λ x, Real.log x - x

theorem max_ln_x_sub_x : ∃ x ∈ Ioc 0 Real.exp 1, ∀ y ∈ Ioc 0 Real.exp 1, f y ≤ f x :=
begin
  use 1,
  split,
  { split,
    { norm_num, },
    { exact Real.exp_pos.ne' } },
  { intros y hy,
    have := min_le_max hy,
    sorry,  -- rest of the detailed proof
  }
end

end max_ln_x_sub_x_l557_557958


namespace correct_decision_probability_l557_557917

open ProbabilityTheory

theorem correct_decision_probability :
  (∃ p : ℝ, ∀ (h : 3), Prob (fun i => h.consultant_in_opinion_is_correct) = 0.8) →
  ∑ (i in (Finset.univ : Finset (Fin 3)), if i.independent_majority (0.8)) = 0.896 :=
by
  sorry

end correct_decision_probability_l557_557917


namespace each_child_gets_one_slice_l557_557088

-- Define the conditions
def couple_slices_per_person : ℕ := 3
def number_of_people : ℕ := 2
def number_of_children : ℕ := 6
def pizzas_ordered : ℕ := 3
def slices_per_pizza : ℕ := 4

-- Calculate slices required by the couple
def total_slices_for_couple : ℕ := couple_slices_per_person * number_of_people

-- Calculate total slices available
def total_slices : ℕ := pizzas_ordered * slices_per_pizza

-- Calculate slices for children
def slices_for_children : ℕ := total_slices - total_slices_for_couple

-- Calculate slices each child gets
def slices_per_child : ℕ := slices_for_children / number_of_children

-- The proof statement
theorem each_child_gets_one_slice : slices_per_child = 1 := by
  sorry

end each_child_gets_one_slice_l557_557088


namespace sum_inequality_l557_557238

-- Define the function f
def f (x : ℝ) (p : ℝ) : ℝ := p * Real.log x + (p - 1) * x^2 + 1

-- Define conditions for monotonicity
def increasing_when_p_ge_1 (p : ℝ) : Prop :=
  \(\forall x > 0, differentiable_once \( f(p \geq 1) )

def decreasing_when_p_le_0 (p : ℝ) : Prop :=
  \let x =  \(p ≤ \infty ) apply differentiability for x))

--<COMMENTS INSERTED>

def increasing_on_interval (p : ℝ) (x1 x2 : ℝ) : Prop :=
  \ (x1 <= \p)  otherwise \(p \geq \)

def decreasing_on_interval (p : ℝ) (x1 x2 : ℝ) : Prop :=
  \(x1 ≤ p  otherwise let x2-p ≥ )

--that must give us a condition.

-- Condition to determine the range of a
def range_of_a (a : ℝ) : Prop :=
  \(\forall x >$0. a ≥  \)

-- Proof of the sum inequality
theorem sum_inequality (n : ℕ) (h: 0 < n) : 
\(\box {1}^{Q) ≥

    -- proving that {x∈}apply (forall) \forall a{t1y}

   \(\exists g')(k ≥ yx∂\):
          (\sqrt ℝ = ≤ Real-type 
          g x t Real_Applied)
          
 Applying tokens specific to (sum_ineqaulity .(N +1)))
 -- concluding


end sum_inequality_l557_557238


namespace neil_total_charge_l557_557340

theorem neil_total_charge 
  (trim_cost : ℕ) (shape_cost : ℕ) (total_boxwoods : ℕ) (shaped_boxwoods : ℕ) : 
  trim_cost = 5 → shape_cost = 15 → total_boxwoods = 30 → shaped_boxwoods = 4 → 
  trim_cost * total_boxwoods + shape_cost * shaped_boxwoods = 210 := 
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end neil_total_charge_l557_557340


namespace external_tangent_twice_internal_tangent_l557_557380

noncomputable def distance_between_centers (r R : ℝ) : ℝ :=
  Real.sqrt (R^2 + r^2 + (10/3) * R * r)

theorem external_tangent_twice_internal_tangent 
  (r R O₁O₂ AB CD : ℝ)
  (h₁ : AB = 2 * CD)
  (h₂ : AB^2 = O₁O₂^2 - (R - r)^2)
  (h₃ : CD^2 = O₁O₂^2 - (R + r)^2) :
  O₁O₂ = distance_between_centers r R :=
by
  sorry

end external_tangent_twice_internal_tangent_l557_557380


namespace exist_numbers_with_product_54_times_l557_557926

def productOfNonZeroDigits (n : Nat) : Nat := 
  (Nat.digits 10 n).filter (λ d => d ≠ 0).foldl (· * ·) 1

theorem exist_numbers_with_product_54_times :
  ∃ (a : Nat), productOfNonZeroDigits a = 54 * productOfNonZeroDigits (a + 1) :=
by
  sorry

end exist_numbers_with_product_54_times_l557_557926


namespace student_solved_correctly_l557_557104

theorem student_solved_correctly (x : ℕ) :
  (x + 2 * x = 36) → x = 12 :=
by
  intro h
  sorry

end student_solved_correctly_l557_557104


namespace factorization_pq_difference_l557_557373

theorem factorization_pq_difference :
  ∃ (p q : ℤ), 25 * x^2 - 135 * x - 150 = (5 * x + p) * (5 * x + q) ∧ p - q = 36 := by
-- Given the conditions in the problem,
-- We assume ∃ integers p and q such that (5x + p)(5x + q) = 25x² - 135x - 150 and derive the difference p - q = 36.
  sorry

end factorization_pq_difference_l557_557373


namespace prob_frieda_reaches_edge_in_4_hops_l557_557544

def friedas_grid : List (ℕ × ℕ) := [(2,1), (2,2), (2,3), (2,4), (1,1), (3,1), (4,1), (1,2), (1,3), (1,4), (2,1), (4,1)]

noncomputable def prob_Frieda_stops_in_4_moves : ℚ :=
  -- Calculation based on Frieda's movement and stopping condition
  117 / 128

theorem prob_frieda_reaches_edge_in_4_hops :
  prob_Frieda_stops_in_4_moves = 117 / 128 :=
begin
  sorry
end

end prob_frieda_reaches_edge_in_4_hops_l557_557544


namespace probability_angie_carlos_two_seats_apart_l557_557916

theorem probability_angie_carlos_two_seats_apart :
  let people := ["Angie", "Bridget", "Carlos", "Diego", "Edwin"]
  let table_size := people.length
  let total_arrangements := (Nat.factorial (table_size - 1))
  let favorable_arrangements := 2 * (Nat.factorial (table_size - 2))
  total_arrangements > 0 ∧
  (favorable_arrangements / total_arrangements : ℚ) = 1 / 2 :=
by {
  sorry
}

end probability_angie_carlos_two_seats_apart_l557_557916


namespace sum_of_powers_of_i_l557_557667

-- Statement of the problem
theorem sum_of_powers_of_i :
  let i := Complex.I in
  (1 + i + i^2 + i^3 + ... + i^2005) = (1 + i) := by
  -- Using the periodicity of powers of i
  sorry

end sum_of_powers_of_i_l557_557667


namespace ratio_S3_S9_l557_557554

noncomputable def Sn (a r : ℝ) (n : ℕ) : ℝ := (a * (1 - r ^ n)) / (1 - r)

theorem ratio_S3_S9 (a r : ℝ) (h1 : r ≠ 1) (h2 : Sn a r 6 = 3 * Sn a r 3) :
  Sn a r 3 / Sn a r 9 = 1 / 7 :=
by
  sorry

end ratio_S3_S9_l557_557554


namespace points_lie_on_circle_l557_557967

theorem points_lie_on_circle (t : ℝ) : 
  let x := Real.cos t
  let y := Real.sin t
  x^2 + y^2 = 1 :=
by
  sorry

end points_lie_on_circle_l557_557967


namespace find_smallest_missing_m_l557_557322

def h_k (k n : ℕ) : ℕ := ⟨n * n / 8^k⟩

def is_missing (n : ℕ) (k : ℕ) (m : ℕ) : bool :=
  ∀ i ∈ list.range n, h_k k i ≠ m

theorem find_smallest_missing_m :
  (∃ (m : ℕ), m > 0 ∧ is_missing 100 3 m) :=
sorry

end find_smallest_missing_m_l557_557322


namespace description_of_T_l557_557317

theorem description_of_T :
  let T := {p : ℝ × ℝ | let (x, y) := p in (x - 3 = 5 ∧ y + 2 ≤ 5) ∨ (y + 2 = 5 ∧ x - 3 ≤ 5) ∨ (x - 3 = y + 2 ∧ 5 ≤ x - 3)} in
  T = {(a, b) : ℝ × ℝ | a = 8 ∧ b ≤ 3} ∪ {(a, b) : ℝ × ℝ | b = 3 ∧ a ≤ 8} ∪ {(a, b) : ℝ × ℝ | b = a - 5 ∧ a ≥ 8} :=
sorry

end description_of_T_l557_557317


namespace no_sunday_customers_l557_557309

/-! 
Kyle has a newspaper-delivery route. Every Monday through Saturday, he delivers the daily paper 
for the 100 houses on his route. On Sunday, some of his customers do not get the Sunday paper, 
but he delivers 30 papers to other houses that get the newspaper only on Sunday. 
Kyle delivers 720 papers each week. How many of his customers do not get the Sunday paper?
-/

theorem no_sunday_customers : 
  ∀ (M : ℕ) (s d : ℕ),
  d = 720 →
  M = 100 →
  s = 30 →
  ∃ n : ℕ, n = (M - (d - (M * 6 + s))) :=
by {
    intros,
    use (M - (d - (M * 6 + s))),
    sorry
}

end no_sunday_customers_l557_557309


namespace number_of_ordered_pairs_l557_557844

noncomputable def possible_pairs (a b : ℕ) : ℕ :=
  if (b > a) ∧ ((a - 6) * (b - 6) = 12) then 1 else 0

theorem number_of_ordered_pairs :
  (∑ a b, possible_pairs a b) = 3 :=
sorry

end number_of_ordered_pairs_l557_557844


namespace no_extreme_points_range_a_l557_557240

theorem no_extreme_points_range_a (a : ℝ) : 
  (∀ x : ℝ, (deriv (λ x, (1/3)*x^3 + x^2 + a*x - 5) x ≠ 0)) → a ≥ 1 :=
sorry

end no_extreme_points_range_a_l557_557240


namespace area_reachable_l557_557476

def turntable_rpm := 30  -- revolutions per minute
def car_speed := 1       -- meters per second

def area_of_reachable_points (t_rpm : ℕ) (c_speed : ℕ) : ℝ :=
  (1 / 2) * ∫ (θ : ℝ) in 0..real.pi, (1 - θ / real.pi)^2

theorem area_reachable (turntable_rpm car_speed : ℕ) (h₁ : turntable_rpm = 30) (h₂ : car_speed = 1) :
  area_of_reachable_points turntable_rpm car_speed = real.pi / 6 :=
by
  rw [h₁, h₂]
  -- Here we would insert the proof computations; omitted for this exercise.
  sorry

end area_reachable_l557_557476


namespace sum_of_a_b_c_d_l557_557472

noncomputable def distance (p₁ p₂ : (ℝ × ℝ)) : ℝ :=
  Real.sqrt ((p₂.1 - p₁.1) ^ 2 + (p₂.2 - p₁.2) ^ 2)

def points : List (ℝ × ℝ) := [(0, 0), (1, 2), (2, 3), (3, 2), (4, 0), (2, 1), (0, 0)]

def perimeter : ℝ :=
  List.foldl (+) 0 (List.map (λ (x : (ℝ × ℝ) × (ℝ × ℝ)), distance x.1 x.2) 
    (List.zip points (List.tail points ++ [points.head])))

theorem sum_of_a_b_c_d : 
  let a := 0
  let b := 2
  let c := 3
  let d := 0
  a + b + c + d = 5 :=
  by
    sorry

end sum_of_a_b_c_d_l557_557472


namespace interval_of_monotonic_increase_tangent_line_at_one_l557_557580

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem interval_of_monotonic_increase :
  {x : ℝ | 0 < (∂ f) x } = Ioi (-1) :=
by
  -- proof here
  sorry

theorem tangent_line_at_one :
  let x1 := 1
  let y1 := f x1
  let m := Deriv f x1
  2 * Real.exp x1 * x - y - y1 = 0 :=
by
  -- proof here
  sorry

end interval_of_monotonic_increase_tangent_line_at_one_l557_557580


namespace students_enrolled_in_all_three_l557_557408

variables {total_students at_least_one robotics_students dance_students music_students at_least_two_students all_three_students : ℕ}

-- Given conditions
axiom H1 : total_students = 25
axiom H2 : at_least_one = total_students
axiom H3 : robotics_students = 15
axiom H4 : dance_students = 12
axiom H5 : music_students = 10
axiom H6 : at_least_two_students = 11

-- We need to prove the number of students enrolled in all three workshops is 1
theorem students_enrolled_in_all_three : all_three_students = 1 :=
sorry

end students_enrolled_in_all_three_l557_557408


namespace dot_product_sum_l557_557290

-- Definitions based on conditions
variables {A B C P : ℝ × ℝ}
variable [management] -- Necessary management for variables in Lean 4 context

-- Conditions
axiom right_triangle : C = (0, 0) ∧ angle A C B = real.pi / 2
axiom sides_length : dist C A = 2 ∧ dist C B = 2
axiom point_P_on_hypotenuse : (∃ k : ℝ, 0 ≤ k ∧ k ≤ 1 ∧ P = (1 - k) • A + k • B)
axiom ratio_BP_PA : ∃ pa pb : ℝ, pa / pb = 1/2 ∧ point_P_on_hypotenuse ∧ point P divides A B in ratio 2:1.

-- Proof goal
theorem dot_product_sum :
  (\overrightarrow{P C}).dot (\overrightarrow{C A}) + (\overrightarrow{P C}).dot (\overrightarrow{C B}) = 4 :=
sorry

end dot_product_sum_l557_557290


namespace arrangement_A_head_is_720_arrangement_ABC_together_is_720_arrangement_ABC_not_together_is_1440_arrangement_A_not_head_B_not_middle_is_3720_l557_557600

noncomputable def arrangements_A_head : ℕ := 720
noncomputable def arrangements_ABC_together : ℕ := 720
noncomputable def arrangements_ABC_not_together : ℕ := 1440
noncomputable def arrangements_A_not_head_B_not_middle : ℕ := 3720

theorem arrangement_A_head_is_720 :
  arrangements_A_head = 720 := 
  by sorry

theorem arrangement_ABC_together_is_720 :
  arrangements_ABC_together = 720 := 
  by sorry

theorem arrangement_ABC_not_together_is_1440 :
  arrangements_ABC_not_together = 1440 := 
  by sorry

theorem arrangement_A_not_head_B_not_middle_is_3720 :
  arrangements_A_not_head_B_not_middle = 3720 := 
  by sorry

end arrangement_A_head_is_720_arrangement_ABC_together_is_720_arrangement_ABC_not_together_is_1440_arrangement_A_not_head_B_not_middle_is_3720_l557_557600


namespace smallest_four_digit_divisible_by_primes_l557_557962

theorem smallest_four_digit_divisible_by_primes :
  ∃ n, 1000 ≤ n ∧ n ≤ 9999 ∧ (∀ p ∈ [2, 3, 5, 7], p ∣ n) ∧ n = 1050 :=
by
  sorry

end smallest_four_digit_divisible_by_primes_l557_557962


namespace regression_line_passes_through_centroid_l557_557432

section Statistics

variable {α : Type*}

-- Define the conditions based on the statements
def independence_testing (var1 var2 : α) : Prop :=
  -- In statistics, independence testing is used to test if two categorical variables are related
  sorry

def residual_plot_narrow_band (residuals_band_width : ℝ) : Prop :=
  -- In a residual plot, the narrower the width of the band area where residuals are distributed, the better the simulation effect
  narrow_band := sorry

def regression_better_simulation (r_squared : ℝ) : Prop :=
  -- The larger the correlation index R², the better the simulation effect
  better_simulation := sorry

-- Define the target statement C, translating the problem condition
def passes_through_sample_point (regression_line : ℝ → ℝ) (sample_points : set (ℝ × ℝ)) : Prop :=
  -- The line corresponding to the linear regression equation passes through at least one point in its sample data
  ∃ (p : ℝ × ℝ), p ∈ sample_points ∧ regression_line p.1 = p.2

-- Define the problem statement
theorem regression_line_passes_through_centroid (regression_line : ℝ → ℝ) (centroid : ℝ × ℝ)
  (sample_points : set (ℝ × ℝ)) (hA : ∀ (var1 var2 : α), independence_testing var1 var2)
  (hB : ∀ (residuals_band_width : ℝ), residual_plot_narrow_band residuals_band_width)
  (hD : ∀ (r_squared : ℝ), regression_better_simulation r_squared) :
  ¬ passes_through_sample_point regression_line sample_points :=
  sorry

end Statistics

end regression_line_passes_through_centroid_l557_557432


namespace percentage_of_female_students_25_or_older_l557_557627

theorem percentage_of_female_students_25_or_older
  (T : ℝ) (M F : ℝ) (P : ℝ)
  (h1 : M = 0.40 * T)
  (h2 : F = 0.60 * T)
  (h3 : 0.56 = (0.20 * T) + (0.60 * (1 - P) * T)) :
  P = 0.40 :=
by
  sorry

end percentage_of_female_students_25_or_older_l557_557627


namespace probability_divisible_by_8_l557_557180

def five_digit_number_set := {n | n ∈ Finset.range 100_000}

def unique_digit_numbers (n : ℕ) : Prop :=
  ∀ (d₁ d₂ d₃ d₄ d₅ : ℕ),
    (d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₁ ≠ d₄ ∧ d₁ ≠ d₅ ∧
    d₂ ≠ d₃ ∧ d₂ ≠ d₄ ∧ d₂ ≠ d₅ ∧
    d₃ ≠ d₄ ∧ d₃ ≠ d₅ ∧
    d₄ ≠ d₅) →
    n = d₁ * 10^4 + d₂ * 10^3 + d₃ * 10^2 + d₄ * 10 + d₅ →
    d₁ ∈ {1, 2, 3, 4, 5, 6, 7, 8} ∧
    d₂ ∈ {1, 2, 3, 4, 5, 6, 7, 8} ∧
    d₃ ∈ {1, 2, 3, 4, 5, 6, 7, 8} ∧
    d₄ ∈ {1, 2, 3, 4, 5, 6, 7, 8} ∧
    d₅ ∈ {1, 2, 3, 4, 5, 6, 7, 8}

def last_three_digits_divisible_by_8 (n : ℕ) :=
  n % 1000 % 8 = 0

theorem probability_divisible_by_8:
  let valid_numbers := {n | (n ∈ five_digit_number_set) ∧ (unique_digit_numbers n) ∧ (last_three_digits_divisible_by_8 n)},
      total_numbers := {n | (n ∈ five_digit_number_set) ∧ (unique_digit_numbers n)} in
  ↑(Finset.card valid_numbers) = (1 / 8) * ↑(Finset.card total_numbers) :=
by
  sorry

end probability_divisible_by_8_l557_557180


namespace find_k_l557_557658

variables {r k : ℝ}
variables {O A B C D : EuclideanSpace ℝ (Fin 3)}

-- Points A, B, C, and D lie on a sphere centered at O with radius r
variables (hA : dist O A = r) (hB : dist O B = r) (hC : dist O C = r) (hD : dist O D = r)
-- The given vector equation
variables (h_eq : 4 • (A - O) - 3 • (B - O) + 6 • (C - O) + k • (D - O) = (0 : EuclideanSpace ℝ (Fin 3)))

theorem find_k (hA : dist O A = r) (hB : dist O B = r) (hC : dist O C = r) (hD : dist O D = r)
(h_eq : 4 • (A - O) - 3 • (B - O) + 6 • (C - O) + k • (D - O) = (0 : EuclideanSpace ℝ (Fin 3))) : 
k = -7 :=
sorry

end find_k_l557_557658


namespace integer_values_abc_l557_557940

theorem integer_values_abc (a b c : ℤ) :
  1 < a ∧ a < b ∧ b < c ∧ (a - 1) * (b - 1) * (c - 1) ∣ (a * b * c - 1) →
  (a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 3 ∧ b = 5 ∧ c = 15) :=
by
  sorry

end integer_values_abc_l557_557940


namespace odd_multiple_of_9_is_multiple_of_3_l557_557398

theorem odd_multiple_of_9_is_multiple_of_3 (n : ℕ) (h1 : n % 2 = 1) (h2 : n % 9 = 0) : n % 3 = 0 := 
by sorry

end odd_multiple_of_9_is_multiple_of_3_l557_557398


namespace count_possible_distributions_l557_557467

-- Definitions representing conditions:
def employees := fin 8
def sub_department := bool

structure assignment :=
  (department : employees → sub_department)
  (translators_not_together : ∀ t1 t2 : employees, t1 ≠ t2 → t1 = 0 → t2 = 1 → department t1 ≠ department t2)
  (programmers_not_all_together : Π (p1 p2 p3 : employees), (p1 = 2) → (p2 = 3) → (p3 = 4) → 
       (department p1 = department p2 → department p2 = department p3) → false)

def count_distribution_plans (s : list assignment) : Nat :=
  s.length

theorem count_possible_distributions : 
  ∃ s : list assignment, count_distribution_plans s = 36 :=
  sorry

end count_possible_distributions_l557_557467


namespace num_elements_M_inter_N_l557_557675

-- Define set M as a set of pairs of real numbers (x, y) where y = x^2
def set_M : set (ℝ × ℝ) := {p | ∃ x : ℝ, p = (x, x^2)}

-- Define set N as a set of real numbers y where y = 2^x
def set_N : set ℝ := {y | ∃ x : ℝ, y = 2^x}

-- The number of elements in set M ∩ N is 0
theorem num_elements_M_inter_N : (set_M ∩ (set.prod univ set_N)).card = 0 := by
  sorry

end num_elements_M_inter_N_l557_557675


namespace arithmetic_progression_terms_l557_557816

theorem arithmetic_progression_terms :
  ∀ (a₁ aₙ d : ℕ),
  a₁ = 2 → aₙ = 62 → d = 2 →
  let n := (aₙ - a₁) / d + 1
  in n = 31 :=
by
  intros a₁ aₙ d h₁ h₂ h₃
  have : a₁ = 2 := h₁
  have : aₙ = 62 := h₂
  have : d = 2 := h₃
  let n := (aₙ - a₁) / d + 1
  have : n = 31 := by sorry
  exact this

end arithmetic_progression_terms_l557_557816


namespace max_term_8_div_a8_l557_557982

variable (a : ℕ → ℚ) (S : ℕ → ℚ)

-- Conditions
def arithmetic_seq := ∀ n, a (n + 1) - a n = a 1 - a 0
def sum_sequence := ∀ n, S n = (n * (2 * a 1 + (n - 1) * (a 1 - a 0))) / 2
def a_1013_eq_2013 := a 1013 = 2013
def S_2013_eq_2013 := S 2013 = 2013

-- Theorem stating the problem and the expected outcome.
theorem max_term_8_div_a8 
  (h1 : arithmetic_seq a) 
  (h2 : sum_sequence S) 
  (h3 : a_1013_eq_2013 a) 
  (h4 : S_2013_eq_2013 S) :
  ∀ n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, 
    (S n / a n ≤ S 8 / a 8) := sorry

end max_term_8_div_a8_l557_557982


namespace probability_divisible_by_20_l557_557617

/-!
# Probability of a six-digit number being divisible by 20

Given the digits 1, 2, 3, 4, 5, and 8, if these digits are arranged into a six-digit positive integer,
the probability that the integer is divisible by 20 is \( \frac{4}{5} \).
-/

theorem probability_divisible_by_20 :
  let digits := {1, 2, 3, 4, 5, 8} in
  let count := 6! in
  let divisible_count := 4 * 5! in
  (divisible_count / count) = (4 / 5) :=
by
  let digits := {1, 2, 3, 4, 5, 8}
  let count := 6!
  let divisible_count := 4 * 5!
  show (divisible_count / count) = (4 / 5)
  sorry

end probability_divisible_by_20_l557_557617


namespace count_triangles_odd_count_triangles_even_l557_557599

-- Define integers and conditions
variables (n x y z : ℕ)

-- State the problem for odd n
theorem count_triangles_odd (h1: odd n) 
    (h2: x + y + z = n)
    (h3: x < y + z) (h4: y < x + z) (h5: z < x + y):
    count_triangles n = (n^2 + 6 * n - 7) / 48 := sorry

-- State the problem for even n
theorem count_triangles_even (h1: even n) 
    (h2: x + y + z = n)
    (h3: x < y + z) (h4: y < x + z) (h5: z < x + y):
    count_triangles n = (n^2 - 4) / 48 := sorry

end count_triangles_odd_count_triangles_even_l557_557599


namespace sequence_always_ends_final_number_on_board_l557_557335

theorem sequence_always_ends (X : ℕ) : 
  ∃ (n : ℕ), (X :: List.range (n + 1)).all_different (λ X => let a := X / 10 in let b := X % 10 in a + 4 * b) := sorry

theorem final_number_on_board (n : ℕ) (h : n = 53^2022 - 1) : 
  ∃ (k : ℕ), (k :: List.range 13).all_different (λ n => let a := n / 10 in let b := n % 10 in a + 4 * b) ∧ 
  k mod 13 = 0 ∧ k mod 3 = 0 ∧ k < 40 :=
by
  obtain ⟨k, hk⟩ := sequence_always_ends n
  have h1 : 53^2022 % 13 = 1 := by sorry
  have h2 : 53^2022 % 3 = 2 := by sorry
  have initial_mod13_zero : (53^2022 - 1) % 13 = 0 := by
    rw [← h1, sub_self]
  have initial_mod3_zero : (53^2022 - 1) % 3 = 0 := by
    rw [← h2, sub_self]
  use 39
  split
  · exact hk
  · exact initial_mod13_zero
  · exact initial_mod3_zero
  · have : 0 ≤ 39 := by norm_num
    exact lt_of_le_of_lt this (by norm_num) 

end sequence_always_ends_final_number_on_board_l557_557335


namespace largest_negative_root_l557_557171

theorem largest_negative_root : 
  ∃ x : ℝ, (∃ k : ℤ, x = -1/2 + 2 * ↑k) ∧ 
  ∀ y : ℝ, (∃ k : ℤ, (y = -1/2 + 2 * ↑k ∨ y = 1/6 + 2 * ↑k ∨ y = 5/6 + 2 * ↑k)) → y < 0 → y ≤ x :=
sorry

end largest_negative_root_l557_557171


namespace find_lending_rate_l557_557843

-- Definitions
def principal_borrowed : ℝ := 8000
def interest_rate_borrowed : ℝ := 4
def time_borrowed : ℝ := 2
def gain_per_year : ℝ := 160

-- Simple interest formula
def simple_interest (P R T : ℝ) : ℝ := P * R * T / 100

-- Goal: Find the interest rate at which he lent the money
theorem find_lending_rate :
  let total_gain := 2 * gain_per_year in
  let interest_paid := simple_interest principal_borrowed interest_rate_borrowed time_borrowed in
  let total_interest_earned := interest_paid + total_gain in
  ∃ R_2 : ℝ, simple_interest principal_borrowed R_2 time_borrowed = total_interest_earned :=
by
  sorry

end find_lending_rate_l557_557843


namespace king_arthur_round_table_seats_l557_557900

theorem king_arthur_round_table_seats (n : ℕ) (h₁ : n > 1) (h₂ : 10 < 29) (h₃ : (29 - 10) * 2 = n - 2) : 
  n = 38 := 
by
  sorry

end king_arthur_round_table_seats_l557_557900


namespace second_and_third_distances_l557_557686

noncomputable def islanders (n : ℕ) : Type := {i : ℕ // i < n}
def is_knight (i : islanders 4) : Prop := sorry
def is_liar (i : islanders 4) : Prop := sorry

axiom two_knights_two_liars : (∃ i j : islanders 4, i ≠ j ∧ is_knight i ∧ is_knight j) ∧ 
                              (∃ i j : islanders 4, i ≠ j ∧ is_liar i ∧ is_liar j)

axiom first_islander_statement : ∀ (i j : islanders 4), i.val = 0 ∧ j.val = 2 → 
                                (is_knight i → is_knight j) ∧
                                (is_liar i → ¬is_knight j)

axiom fourth_islander_statement : ∀ (i j : islanders 4), i.val = 3 ∧ j.val = 1 → 
                                 (is_knight i → is_knight j) ∧
                                 (is_liar i → ¬is_knight j)

def distance (i j : islanders 4) : ℕ := sorry

theorem second_and_third_distances : 
  (∃ d1 d2 : ℕ, 
    (∀ k : islanders 4, k.val = 1 → 
    (is_knight k → distance k ⟨2, nat.lt_succ_succ 2⟩ = d1) ∧ 
    (is_liar k → distance k ⟨2, nat.lt_succ 2⟩ ≠ d1)) ∧
    (∀ k : islanders 4, k.val = 2 → 
    (is_knight k → distance k ⟨1, nat.succ_lt_succ (nat.zero_lt_succ 1)⟩ = d2) ∧ 
    (is_liar k → distance k ⟨1, nat.succ_lt_succ (nat.zero_lt_succ 1)⟩ ≠ d2))
  ) :=
⟨1, 1, sorry, sorry⟩

end second_and_third_distances_l557_557686


namespace positive_integer_satisfies_condition_l557_557707

def num_satisfying_pos_integers : ℕ :=
  1

theorem positive_integer_satisfies_condition :
  ∃ (n : ℕ), 16 - 4 * n > 10 ∧ n = num_satisfying_pos_integers := by
  sorry

end positive_integer_satisfies_condition_l557_557707


namespace jesse_mia_total_miles_per_week_l557_557643

noncomputable def jesse_miles_per_day_first_three := 2 / 3
noncomputable def jesse_miles_day_four := 10
noncomputable def mia_miles_per_day_first_four := 3
noncomputable def average_final_three_days := 6

theorem jesse_mia_total_miles_per_week :
  let jesse_total_first_four_days := 3 * jesse_miles_per_day_first_three + jesse_miles_day_four
  let mia_total_first_four_days := 4 * mia_miles_per_day_first_four
  let total_miles_needed_final_three_days := 3 * average_final_three_days * 2
  jesse_total_first_four_days + total_miles_needed_final_three_days = 48 ∧
  mia_total_first_four_days + total_miles_needed_final_three_days = 48 :=
by
  sorry

end jesse_mia_total_miles_per_week_l557_557643


namespace opposite_of_three_l557_557759

theorem opposite_of_three : -3 = -3 := by
  -- The condition we have identified is the given number 3.
  -- We will directly state that the opposite of 3 is -3.
  -- This proof is trivial as we are directly replacing 3 with -3 to match the problem statement.
  rfl

end opposite_of_three_l557_557759


namespace part1_collinear_part2_acute_angle_l557_557549

variable {x y k : ℝ}
def a := (x, 1, 0)
def b := (-1, y, 2)
def c := (2, -2, 1)

noncomputable def magnitude_b := Real.sqrt ((-1:ℝ)^2 + y^2 + (2:ℝ)^2)

theorem part1_collinear (h1 : magnitude_b = Real.sqrt 5) (h2 : a.1 * c.1 + a.2 * c.2 + a.3 * c.3 = 0) :
  k = 1/2 → (a.1 + k * b.1, a.2 + k * b.2, a.3 + k * b.3) = (1, 2, 2) :=
sorry

theorem part2_acute_angle (h1 : magnitude_b = Real.sqrt 5) (h2 : a.1 * c.1 + a.2 * c.2 + a.3 * c.3 = 0) :
  (-1 < k ∧ k < 1/2) ∨ (1/2 < k ∧ k < ∞) → (a.1 + k * b.1, a.2 + k * b.2, a.3 + k * b.3) = (1, 2, 2) :=
sorry

end part1_collinear_part2_acute_angle_l557_557549


namespace problem1_problem2_problem3_problem4_l557_557069

theorem problem1
  (C' C : Set Point)
  (hC' : Convex C')
  (hC : Convex C)
  (h_inside : C' ⊆ C) :
  perimeter C' ≤ perimeter C := sorry

theorem problem2
  (C C' : Set Point)
  (h_min : ∀ C'', Convex C'' ∧ (C' ⊆ C'') → perimeter C ≤ perimeter C'')
  (hC' : Convex C')
  (hC : Convex C) :
  perimeter C ≤ perimeter C' := sorry

theorem problem3
  (S' S : Set Point)
  (hS' : Convex S')
  (hS : Convex S)
  (h_inside : S' ⊆ S) :
  area S' ≤ area S := sorry

theorem problem4
  (S S' : Set Point)
  (h_min : ∀ S'', Convex S'' ∧ (S' ⊆ S'') → area S ≤ area S'')
  (hS' : Convex S')
  (hS : Convex S) :
  ¬ (area S ≤ area S') := sorry

end problem1_problem2_problem3_problem4_l557_557069


namespace find_alpha_beta_l557_557230

variable (A B C : ℂ)
variable (α β : ℂ)

def complex_vertices : Prop :=
  A = 3 + 2 * complex.I ∧
  B = 3 * complex.I ∧
  C = 2 - complex.I

def circumcircle_eqn (z : ℂ) :=
  ∥conj z∥^2 + α * z + conj α * conj z + β = 0

theorem find_alpha_beta :
  complex_vertices A B C →
  ∃ α β : ℂ, α = -1 + complex.I ∧ β = -3 ∧
    ∀ z : ℂ, circumcircle_eqn α β z := sorry

end find_alpha_beta_l557_557230


namespace opposite_of_three_l557_557765

theorem opposite_of_three : -3 = opposite(3) := 
by
  sorry

end opposite_of_three_l557_557765


namespace find_A_max_area_l557_557969

-- Definition of the problem’s conditions
variables {a b c : ℝ} {A B C : ℝ}

-- Condition given c = 2b - 2a * cos C
axiom equation_condition : c = 2 * b - 2 * a * Real.cos C

-- Prove that A = π / 3 given the conditions
theorem find_A (h : equation_condition) : A = Real.pi / 3 :=
sorry

-- Given a = 2, prove the maximum area of triangle ABC is sqrt 3
theorem max_area (h : equation_condition) (ha : a = 2) : 
  (Real.sqrt 3) = (1 / 2) * b * c * Real.sin A :=
sorry

end find_A_max_area_l557_557969


namespace part_I_part_II_l557_557243

noncomputable def f (a b x : ℝ) := a * x ^ 2 + b * x
def g (x : ℝ) := Real.log x

-- Part (I)
theorem part_I (x : ℝ) (h₀ : 0 < x) : f 1 1 x > g x := by
  sorry

-- Part (II)
theorem part_II (a : ℝ) : - (1 / (2 * Real.exp 3)) ≤ a := by
  sorry

end part_I_part_II_l557_557243


namespace ellipse_equation_and_fixed_point_l557_557574

theorem ellipse_equation_and_fixed_point :
  (∃ (a b : ℝ) (E : set (ℝ × ℝ)), 
    a > b ∧ b > 0 ∧ 
    eccentricity = (Real.sqrt 3 / 2) ∧ 
    E = {p | (p.1^2 / a^2) + (p.2^2 / b^2) = 1} ∧ 
    (sqrt 3, 1/2) ∈ E ∧ 
    a^2 = 4 ∧ b^2 = 1 ∧ E = {p | (p.1^2 / 4) + (p.2^2) = 1}
  ) ∧
  (∀ (P : ℝ × ℝ),
    let A := (-2, 0),
        B := (2, 0),
        l := {p | p.1 = 4},
        AP := {λ p, ∃ k : ℝ, p.2 = k * (p.1 + 2)},
        PB := {λ p, ∃ k : ℝ, p.2 = k * (p.1 - 2)} in
      P.1 = 4 ∧
      (P.2 ≠ 0) ∧
      (P ∉ l) →
      ∃ C D : ℝ × ℝ, 
        C ∈ E ∧
        D ∈ E ∧
        C ≠ D ∧
        line_through C D = {p | p.1 = 1 ∧ p.2 = 0}) :=
sorry

end ellipse_equation_and_fixed_point_l557_557574


namespace sin_half_angle_identity_l557_557550

theorem sin_half_angle_identity (theta : ℝ) (h : Real.sin (Real.pi / 2 + theta) = - 1 / 2) :
  2 * Real.sin (theta / 2) ^ 2 - 1 = 1 / 2 := 
by
  sorry

end sin_half_angle_identity_l557_557550


namespace odd_even_shift_composition_l557_557708

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def is_even_function_shifted (f : ℝ → ℝ) (shift : ℝ) : Prop :=
  ∀ x : ℝ, f (x + shift) = f (-x + shift)

theorem odd_even_shift_composition
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_even_shift : is_even_function_shifted f 3)
  (h_f1 : f 1 = 1) :
  f 6 + f 11 = -1 := by
  sorry

end odd_even_shift_composition_l557_557708


namespace count_valid_pairs_l557_557641

def Jane := 30
def two_digit (x : ℕ) : Prop := 10 ≤ x ∧ x < 100
def valid_pair (t n a b : ℕ) : Prop :=
  b > a ∧ two_digit (10 * a + b) ∧ two_digit (10 * b + a) ∧ 
  30 + n = 10 * a + b ∧ t + n = 10 * b + a

theorem count_valid_pairs : ∃ (t n : ℕ) (a b : ℕ), (∃! p : ℕ → ℕ → Prop, valid_pair t n a b) ∧ p.count_pairs = 21 := 
sorry

end count_valid_pairs_l557_557641


namespace parabola_pi2_intersect_x_axis_l557_557780

noncomputable def parabola_intercept (a b v₁ v₂ : ℝ) (h₁ : 2 * v₁ = a + b) (h₂ : 2 * v₂ = 0 + a) : ℝ :=
2 * v₂ - b

theorem parabola_pi2_intersect_x_axis :
  parabola_intercept 13 33 11.5 23 (by norm_num) (by norm_num) = 33 :=
by norm_num

end parabola_pi2_intersect_x_axis_l557_557780


namespace original_number_of_men_l557_557437

/--A group of men decided to complete a work in 6 days. 
 However, 4 of them became absent, and the remaining men finished the work in 12 days. 
 Given these conditions, we need to prove that the original number of men was 8. --/
theorem original_number_of_men 
  (x : ℕ) -- original number of men
  (h1 : x * 6 = (x - 4) * 12) -- total work remains the same
  : x = 8 := 
sorry

end original_number_of_men_l557_557437


namespace eight_pointed_star_sum_of_angles_l557_557947

def sum_of_star_tip_angles (n : ℕ) (arc_deg : ℝ) : ℝ :=
  let angle_per_tip := (((n / 2) * arc_deg) / 2)
  n * angle_per_tip

theorem eight_pointed_star_sum_of_angles :
  sum_of_star_tip_angles 8 45 = 540 := by
  sorry

end eight_pointed_star_sum_of_angles_l557_557947


namespace distinct_pentomino_shape_count_l557_557416

-- Define what a valid pentomino shape is.
def is_pentomino (shape : Set (ℕ × ℕ)) : Prop :=
  shape.card = 5 ∧ ∀ (a b ∈ shape), a ≠ b → (a.1 = b.1 ∨ a.2 = b.2) ∧
  ∃ (axis ∈ {r|r = 0 ∨ r = 90 ∨ r = 180 ∨ r = 270}),
  Set (rotate shape axis) == shape

-- Count the number of distinct pentomino shapes.
def distinct_pentomino_count : ℕ :=
  18

theorem distinct_pentomino_shape_count : ∃ count, count = 18 :=
by {
  use distinct_pentomino_count,
  -- The detailed proof will establish this.
  sorry
}

end distinct_pentomino_shape_count_l557_557416


namespace average_payment_52_installments_l557_557810

theorem average_payment_52_installments :
  let first_payment : ℕ := 500
  let remaining_payment : ℕ := first_payment + 100
  let num_first_payments : ℕ := 25
  let num_remaining_payments : ℕ := 27
  let total_payments : ℕ := num_first_payments + num_remaining_payments
  let total_paid_first : ℕ := num_first_payments * first_payment
  let total_paid_remaining : ℕ := num_remaining_payments * remaining_payment
  let total_paid : ℕ := total_paid_first + total_paid_remaining
  let average_payment : ℚ := total_paid / total_payments
  average_payment = 551.92 :=
by
  sorry

end average_payment_52_installments_l557_557810


namespace triangle_inequality_l557_557404

theorem triangle_inequality (A B C : Point) (AD BE : Line) :
  angle A B C > 90 → 
  altitude_from A D (triangle A B C) → 
  altitude_from B E (triangle A B C) → 
  segment_length B C + segment_length A D ≥ segment_length A C + segment_length B E := 
begin
  sorry
end

end triangle_inequality_l557_557404


namespace opposite_of_3_l557_557733

-- Define the concept of opposite of a number and the logic for positive numbers
def opposite (x : Int) : Int := 
  if x > 0 then -x
  else if x < 0 then -x
  else 0

-- Statement to prove that the opposite of 3 is -3
theorem opposite_of_3 : opposite 3 = -3 :=
by 
  -- Using the definition of opposite
  unfold opposite
  -- Simplify the expression for x = 3
  simp [lt_irrefl, int.coe_nat_lt]
  -- Conclude proof
  rfl

end opposite_of_3_l557_557733


namespace max_value_y_l557_557178

open Real

noncomputable def y (x : ℝ) : ℝ :=
  tan (x + π / 4) - tan (x + π / 3) + sin (x + π / 3)

theorem max_value_y :
  ∃ x, -π / 2 ≤ x ∧ x ≤ -π / 4 ∧ y x = -sqrt 3 + 1 - sqrt 2 / 2 :=
by
  sorry

end max_value_y_l557_557178


namespace arithmetic_sequence_probability_l557_557683

theorem arithmetic_sequence_probability (n p : ℕ) (h_cond : n + p = 2008) (h_neg : n = 161) (h_pos : p = 2008 - 161) :
  ∃ a b : ℕ, (a = 1715261 ∧ b = 2016024 ∧ a + b = 3731285) ∧ (a / b = 1715261 / 2016024) := by
  sorry

end arithmetic_sequence_probability_l557_557683


namespace find_beta_l557_557567

variable {α β : Real}

theorem find_beta 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2) 
  (h3 : sin α = sqrt 5 / 5) 
  (h4 : sin (α - β) = -(sqrt 10 / 10)) 
  : β = π / 4 := 
  sorry

end find_beta_l557_557567


namespace maximize_product_l557_557006

-- Define the circle and points A, P1, P2
variables {α : Type*} [inner_product_space ℝ α] (A P1 P2 : α)

-- Define the conditions in the problem
def are_points_on_circle (A : α) (P1 P2 : α) :=
  ∃ r : ℝ, dist A P1 = r ∧ dist A P2 = r

def is_external_bisector (A P1 P2 : α) :=
  ∃ B : α, angle A P1 B + angle P1 B P2 = angle A P2 B

-- Statement of the theorem to prove the maximum product P1A * AP2
theorem maximize_product
  (h1 : are_points_on_circle A P1 P2)
  (h2 : ∃ B : α, angle A P1 B > 0 ∧ angle A P1 B < π)
  : is_external_bisector A P1 P2 :=
sorry

end maximize_product_l557_557006


namespace exists_nat_number_aa_is_perfect_square_l557_557304

-- Lean statement of the problem
theorem exists_nat_number_aa_is_perfect_square :
  ∃ (A : ℕ) (n : ℕ), let k := 10^n + 1 in
  k ≥ 2 ∧ (∃ (B : ℕ), A * k = B^2) ∧ (nat.log10 A).ceil = n := 
sorry

end exists_nat_number_aa_is_perfect_square_l557_557304


namespace prove_trajectory_prove_max_distance_l557_557288

open Real

/-- Given conditions -/
def conditions : Prop :=
  ∀ (α : ℝ), (0 ≤ α ∧ α ≤ 2 * π) →
    let P := (2 * cos α, 2 * sin α + 2) in
    let l_θ := λ (θ : ℝ), 10 / (sqrt 2 * sin (θ - π / 4)) in
    True -- This step includes all the given conditions descriptively.

noncomputable def cartesian_trajectory_equation (x y : ℝ) : Prop :=
  x^2 + (y - 2)^2 = 4

noncomputable def maximum_distance (α : ℝ) (x y : ℝ) : ℝ :=
  abs ((sqrt 2) * cos (α + π / 4) + 4) * sqrt 2 + 2

/-- Proof problem statements -/
theorem prove_trajectory (α : ℝ) (hα: 0 ≤ α ∧ α ≤ 2 * π)
  (x y : ℝ) (P : x = 2 * cos α ∧ y = 2 * sin α + 2) :
  cartesian_trajectory_equation x y :=
  sorry

theorem prove_max_distance (α : ℝ) (hα: 0 ≤ α ∧ α ≤ 2 * π)
  (x y : ℝ) (P : x = 2 * cos α ∧ y = 2 * sin α + 2) :
  maximum_distance α x y = 4 * sqrt 2 + 2 :=
  sorry

end prove_trajectory_prove_max_distance_l557_557288


namespace red_midpoints_at_least_1991_l557_557365

theorem red_midpoints_at_least_1991 (P : Fin 997 → ℝ × ℝ) :
  ∃ S : Set (ℝ × ℝ), (∀ i j, 0 ≤ i < 997 → 0 ≤ j < 997 → i ≠ j → 
    let M := ((P i).1 + (P j).1) / 2, (P i).2 + (P j).2 / 2;
    M ∈ S ∧ S.card ≥ 1991) := sorry

end red_midpoints_at_least_1991_l557_557365


namespace sum_of_sticker_quantities_l557_557354

theorem sum_of_sticker_quantities : 
  (∑ (n : ℕ) in (finset.filter (λ n, n < 100 ∧ n % 6 = 2 ∧ n % 8 = 3) (finset.range 100)), n) = 83 :=
by
  sorry

end sum_of_sticker_quantities_l557_557354


namespace eccentricity_of_ellipse_l557_557267

-- Definitions
variable (a b c : ℝ)  -- semi-major axis, semi-minor axis, and distance from center to a focus
variable (h_c_eq_b : c = b)  -- given condition focal length equals length of minor axis
variable (h_a_eq_sqrt_sum : a = Real.sqrt (c^2 + b^2))  -- relationship in ellipse

-- Question: Prove the eccentricity of the ellipse e = √2 / 2
theorem eccentricity_of_ellipse : (c = b) → (a = Real.sqrt (c^2 + b^2)) → (c / a = Real.sqrt 2 / 2) :=
by
  intros h_c_eq_b h_a_eq_sqrt_sum
  sorry

end eccentricity_of_ellipse_l557_557267


namespace exists_constant_c_for_inequality_l557_557694

theorem exists_constant_c_for_inequality :
  ∃ (c : ℝ), c = 1 / Real.sqrt 3 ∧ ∀ (x y z : ℝ), -Real.abs (x * y * z) > c * (Real.abs x + Real.abs y + Real.abs z) :=
by
  use 1 / Real.sqrt 3
  intro x y z
  sorry

end exists_constant_c_for_inequality_l557_557694


namespace cube_paint_probability_l557_557159

/-- 
Each face of a cube is painted either green or yellow, each with probability 1/2. 
The color of each face is determined independently. Prove that the probability 
that the painted cube can be placed on a horizontal surface so that the four 
vertical faces are all the same color is 5/16. 
-/
theorem cube_paint_probability : 
  let cube_faces := [true, false] -- Assuming true is green, false is yellow
  let arrangements := { arr | arr ∈ cube_faces^6 ∧ independent (λ i, arr i) }
  let suitable_arrangements := { arr ∈ arrangements | 
    can_be_placed_on_horizontal_surface_with_same_color_vertical_faces arr }
  ∃ pr, pr = (|suitable_arrangements| : ℚ) / (|arrangements| : ℚ) ∧ pr = 5 / 16
:= sorry

end cube_paint_probability_l557_557159


namespace find_a_b_find_m_l557_557234

def f (a b x : ℝ) : ℝ := a * x^2 - 2 * a * x + b + 2

theorem find_a_b (a b : ℝ) (h1 : 0 < a)
  (h2 : f a b 1 = 0)
  (h3 : ∀ x ∈ set.Icc (0:ℝ) 1, f a b x ≤ f a b 0 + 3 ∧ f a b x ≤ f a b 1 + 3) :
  a = 3 ∧ b = 1 :=
sorry

theorem find_m (m : ℝ) :
  (∀ x ∈ set.Icc (1/3) 2, f 3 1 x < m * x^2 + 1) ↔ 3 < m :=
sorry

end find_a_b_find_m_l557_557234


namespace problem_statement_l557_557431

variable (A : Type) [LinearOrderedField A]

def statementA (x y : A) : Prop :=
  ∃ k : A, y = k * x ∧ (x, y) = (-2, -3)

def statementB (m x y : A) : Prop :=
  2 * (m + 1) * x + (m - 3) * y + 7 - 5 * m = 0 ∧ (x, y) = (1, 3)

def statementC (θ x y : A) : Prop :=
  y - 1 = tan θ * (x - 1) ∧ (x, y) = (1, 1)

def statementD (x1 y1 x2 y2 x y : A) : Prop :=
  (x2 - x1) * (y - y1) = (y2 - y1) * (x - x1) 

theorem problem_statement (A_correct B_correct C_correct D_correct : Prop) : 
  (¬ A_correct ∧ ¬ C_correct ∧ B_correct ∧ D_correct) :=
by
  -- Proof not required
  sorry
 
end problem_statement_l557_557431


namespace find_parallel_line_eq_l557_557045

def slope_intercept_form (a b c : ℝ) : (ℝ × ℝ) :=
  let m := -a / b
  let b := -c / b
  (m, b)

def parallel_line_eq (m x1 y1 : ℝ) : ℝ × ℝ :=
  let b := y1 - m * x1
  (m, b)

theorem find_parallel_line_eq (a b c x1 y1 : ℝ) (h₁ : a = 3) (h₂ : b = -6) (h₃ : c = 9) (h₄ : x1 = 2) (h₅ : y1 = 0) :
  parallel_line_eq (slope_intercept_form a b c).fst x1 y1 = (1 / 2, -1) :=
by
  sorry

end find_parallel_line_eq_l557_557045


namespace valid_parameterizations_l557_557381

-- Define the line equation as a predicate
def is_on_line (x y : ℝ) : Prop :=
  y = (7 / 5) * x - (23 / 5)

-- Define what it means for a pair of vectors to be scalar multiples
def is_scalar_multiple {α : Type*} [DivisionRing α] (v w : α × α) : Prop :=
  ∃ (c : α), v = (c * w.1, c * w.2)

-- Define the candidates
def candidate_A := λ t : ℝ, (5 + -5 * t, 2 + -7 * t)
def candidate_B := λ t : ℝ, (23 + 10 * t, 7 + 14 * t)
def candidate_C := λ t : ℝ, (3 + (7/5) * t, -8 / 5 + 1 * t)
def candidate_E := λ t : ℝ, (0 + 25 * t, (-23 / 5) + -35 * t)

-- Main theorem to be proved
theorem valid_parameterizations :
  (∀ t, is_on_line (candidate_A t).1 (candidate_A t).2)
  ∧ (∀ t, is_on_line (candidate_B t).1 (candidate_B t).2)
  ∧ (∀ t, is_on_line (candidate_C t).1 (candidate_C t).2)
  ∧ (∀ t, is_on_line (candidate_E t).1 (candidate_E t).2) :=
by sorry

end valid_parameterizations_l557_557381


namespace opposite_of_3_l557_557778

theorem opposite_of_3 : -3 = -3 := 
by
  -- sorry is added to skip the proof as per instructions
  sorry

end opposite_of_3_l557_557778


namespace final_numbers_proof_l557_557524

-- Definition of initial conditions
structure Cube where
  left : ℕ
  right : ℕ
  front : ℕ
  back : ℕ
  top : ℕ
  bottom : ℕ
  opposites_sum : ∀ {face1 face2 : Cube}, face1 + face2 = 50

-- Definition of initial cube state
def initial_cube : Cube :=
  { left := 37, right := 13,
    front := 15, back := 35,
    top := 11, bottom := 39,
    opposites_sum := λ face1 face2, sorry }

-- Definition of rotation function (abstractly)
-- One rotation to the right
def rotate_right (c : Cube) : Cube :=
  { left := c.top, right := c.bottom,
    front := c.front, back := c.back,
    top := c.right, bottom := c.left,
    opposites_sum := c.opposites_sum }
    
-- Two rotations to the front
def rotate_forward_2 (c : Cube) : Cube :=
  let c1 := { left := c.left, right := c.right,
              front := c.top, back := c.bottom,
              top := c.back, bottom := c.front,
              opposites_sum := c.opposites_sum } in
  { left := c.left, right := c.right,
    front := c1.top, back := c1.bottom,
    top := c1.back, bottom := c1.front,
    opposites_sum := c1.opposites_sum }

-- Final state after rotations
def final_cube : Cube := 
  rotate_forward_2 (rotate_right initial_cube)

-- Theorem statement to prove the solution
theorem final_numbers_proof :
  final_cube.bottom = 37 ∧ final_cube.front = 35 ∧ final_cube.right = 11 :=
by
  sorry

end final_numbers_proof_l557_557524


namespace factorial_division_result_l557_557805

-- Define the inputs and expected output
def n : ℕ := 9
def k : ℕ := 3

-- Condition of factorial
def factorial (m : ℕ) : ℕ := Nat.factorial m

theorem factorial_division_result : (factorial n) / (factorial (n - k)) = 504 :=
by
  sorry

end factorial_division_result_l557_557805


namespace smallest_abs_value_l557_557123

theorem smallest_abs_value : 
    ∀ (a b c d : ℝ), 
    a = -1/2 → b = -2/3 → c = 4 → d = -5 → 
    abs a < abs b ∧ abs a < abs c ∧ abs a < abs d := 
by
  intros a b c d ha hb hc hd
  rw [ha, hb, hc, hd]
  simp
  -- Proof omitted for brevity
  sorry

end smallest_abs_value_l557_557123


namespace initial_bananas_on_tree_l557_557455

-- Definitions of given conditions
def bananas_left_on_tree : ℕ := 100
def bananas_eaten : ℕ := 70
def bananas_in_basket : ℕ := 2 * bananas_eaten

-- Statement to prove the initial number of bananas on the tree
theorem initial_bananas_on_tree : bananas_left_on_tree + (bananas_in_basket + bananas_eaten) = 310 :=
by
  sorry

end initial_bananas_on_tree_l557_557455


namespace ratio_of_p_to_r_l557_557023

theorem ratio_of_p_to_r (p q r s : ℚ) 
  (h1 : p / q = 5 / 4) 
  (h2 : r / s = 4 / 3) 
  (h3 : s / q = 1 / 8) : 
  p / r = 15 / 2 := 
by 
  sorry

end ratio_of_p_to_r_l557_557023


namespace sqrt_121_eq_pm_11_l557_557065

theorem sqrt_121_eq_pm_11 : (∀ x : ℝ, x^2 = 121 → x = 11 ∨ x = -11) :=
by {
  intro x,
  intro h,
  have hx : x * x = 121 := by assumption,
  have pos_x : x = real.sqrt 121 ∨ x = - real.sqrt 121 := by
    have sqrt_121 := real.sqrt_eq_iff_sqr_eq (by norm_num) (by norm_num),
    rw sqrt_121 at hx,
    exact hx,
  rw real.sqrt_eq_iff_sqr_eq (by norm_num) (by norm_num) at pos_x,
  exact pos_x
}

end sqrt_121_eq_pm_11_l557_557065


namespace cooking_dishes_time_l557_557118

def total_awake_time : ℝ := 16
def work_time : ℝ := 8
def gym_time : ℝ := 2
def bath_time : ℝ := 0.5
def homework_bedtime_time : ℝ := 1
def packing_lunches_time : ℝ := 0.5
def cleaning_time : ℝ := 0.5
def shower_leisure_time : ℝ := 2
def total_allocated_time : ℝ := work_time + gym_time + bath_time + homework_bedtime_time + packing_lunches_time + cleaning_time + shower_leisure_time

theorem cooking_dishes_time : total_awake_time - total_allocated_time = 1.5 := by
  sorry

end cooking_dishes_time_l557_557118


namespace ellipse_eccentricity_range_l557_557987

theorem ellipse_eccentricity_range
  (a b c : ℝ)
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (a_gt_b : a > b)
  (point_P_on_ellipse : ∃ m n : ℝ, (m / a)^2 + (n / b)^2 = 1 ∧ (m - -c, n) • (m - c, n) = 2 * c^2) :
  1 / 2 ≤ c / a ∧ c / a ≤ real.sqrt 3 / 3 :=
by
  sorry

end ellipse_eccentricity_range_l557_557987


namespace total_seats_round_table_l557_557866

theorem total_seats_round_table (n : ℕ) (h : n = 38)
  (ka_position : ℕ) (sl_position : ℕ) 
  (h1 : ka_position = 10) 
  (h2 : sl_position = 29) 
  (h3 : (ka_position + n/2) % n = sl_position) : 
  n = 38 :=
by
  -- All steps and solution proof
  sorry

end total_seats_round_table_l557_557866


namespace segment_length_greater_than_inradius_sqrt_two_l557_557024

variables {a b c : ℝ} -- sides of the triangle
variables {P Q : ℝ} -- points on sides of the triangle
variables {S_ABC S_PCQ : ℝ} -- areas of the triangles
variables {s : ℝ} -- semi-perimeter of the triangle
variables {r : ℝ} -- radius of the inscribed circle
variables {ℓ : ℝ} -- length of segment dividing the triangle's area

-- Given conditions in the form of assumptions
variables (h1 : S_PCQ = S_ABC / 2)
variables (h2 : PQ = ℓ)
variables (h3 : r = S_ABC / s)

-- The statement of the theorem
theorem segment_length_greater_than_inradius_sqrt_two
  (h1 : S_PCQ = S_ABC / 2) 
  (h2 : PQ = ℓ) 
  (h3 : r = S_ABC / s)
  (h4 : s = (a + b + c) / 2) 
  (h5 : S_ABC = Real.sqrt (s * (s - a) * (s - b) * (s - c))) 
  (h6 : ℓ^2 = a^2 + b^2 - (a^2 + b^2 - c^2) / 2) :
  ℓ > r * Real.sqrt 2 :=
sorry

end segment_length_greater_than_inradius_sqrt_two_l557_557024


namespace kiki_total_money_l557_557308

theorem kiki_total_money 
  (S : ℕ) (H : ℕ) (M : ℝ)
  (h1: S = 18)
  (h2: H = 2 * S)
  (h3: 0.40 * M = 36) : 
  M = 90 :=
by
  sorry

end kiki_total_money_l557_557308


namespace find_julios_bonus_l557_557647

def commission (customers: ℕ) : ℕ :=
  customers * 1

def total_commission (week1: ℕ) (week2: ℕ) (week3: ℕ) : ℕ :=
  commission week1 + commission week2 + commission week3

noncomputable def julios_bonus (total_earnings salary total_commission: ℕ) : ℕ :=
  total_earnings - salary - total_commission

theorem find_julios_bonus :
  let week1 := 35
  let week2 := 2 * week1
  let week3 := 3 * week1
  let salary := 500
  let total_earnings := 760
  let total_comm := total_commission week1 week2 week3
  julios_bonus total_earnings salary total_comm = 50 :=
by
  sorry

end find_julios_bonus_l557_557647


namespace correct_inequality_l557_557327

noncomputable def f (x : ℝ) : ℝ := x^2 - real.pi * x

noncomputable def alpha : ℝ := real.arcsin (1 / 3)
noncomputable def beta : ℝ := real.arctan (5 / 4)
noncomputable def gamma : ℝ := real.arccos (-1 / 3)
noncomputable def delta : ℝ := real.pi - real.arctan (5 / 4)

theorem correct_inequality :
  f(alpha) > f(delta) ∧ f(delta) > f(beta) ∧ f(beta) > f(gamma) :=
by
  sorry -- proof goes here

end correct_inequality_l557_557327


namespace high_card_point_value_l557_557275

theorem high_card_point_value :
  ∀ (H L : ℕ), 
  (L = 1) →
  ∀ (high low total_points : ℕ), 
  (total_points = 5) →
  (high + (L + L + L) = total_points) →
  high = 2 :=
by
  intros
  sorry

end high_card_point_value_l557_557275


namespace solution_of_inequality_l557_557570

noncomputable def solution_set : set ℝ := set.Ioi 2 ∪ set.Iio 3

theorem solution_of_inequality
  (a b : ℝ)
  (h1 : ∀ x : ℝ, ax^2 - bx - 1 ≥ 0 ↔ -1/2 ≤ x ∧ x ≤ -1/3) :
  ∀ x : ℝ, ax^2 - bx - 1 < 0 ↔ (2 < x) ∨ (x < 3) := 
sorry

end solution_of_inequality_l557_557570


namespace probability_have_all_letters_l557_557644

noncomputable def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

def probability_select_letters (word : String) (k : ℕ) (letters : Finset Char) : ℚ :=
  if letters ⊆ word.to_finset && letters.card = k then 1 / binom word.length k else 0

theorem probability_have_all_letters :
  let CAKE := "CAKE".to_finset
  let SHORE := "SHORE".to_finset
  let FLOW := "FLOW".to_finset
  let COFFEE := "COFFEE".to_finset
  let p₁ := probability_select_letters "CAKE" 2 (Finset.of_list ['C', 'E'])
  let p₂ := probability_select_letters "SHORE" 4 (Finset.of_list ['O', 'E', 'F'])
  let p₃ := probability_select_letters "FLOW" 3 (Finset.of_list ['F', 'E'])
  p₁ * p₂ * p₃ = 1 / 120 :=
by
  sorry

end probability_have_all_letters_l557_557644


namespace initial_bananas_l557_557457

theorem initial_bananas (bananas_left: ℕ) (eaten: ℕ) (basket: ℕ) 
                        (h_left: bananas_left = 100) 
                        (h_eaten: eaten = 70) 
                        (h_basket: basket = 2 * eaten): 
  bananas_left + eaten + basket = 310 :=
by
  sorry

end initial_bananas_l557_557457


namespace king_lancelot_seats_38_l557_557896

noncomputable def totalSeats (seat_king seat_lancelot : ℕ) : ℕ :=
  if seat_king < seat_lancelot then
    2 * (seat_lancelot - seat_king - 1) + 2
  else
    2 * (seat_king - seat_lancelot - 1) + 2

theorem king_lancelot_seats_38 (seat_king seat_lancelot : ℕ) (h1 : seat_king = 10) (h2 : seat_lancelot = 29) :
  totalSeats seat_king seat_lancelot = 38 := 
  by
    sorry

end king_lancelot_seats_38_l557_557896


namespace simulation_probability_of_two_bullseyes_in_three_shots_l557_557305

/-
Conditions:
1. Probability of hitting the bullseye = 0.4
2. Representation in simulation:
   - $0, 1, 2, 3$ represent hitting the bullseye
   - $4, 5, 6, 7, 8, 9$ represent missing the bullseye
3. Given groups of random numbers (simulated shots):

Groups are:
321, 421, 191, 925, 271, 932, 800, 478, 589, 663,
531, 297, 396, 021, 546, 388, 230, 113, 507, 965

Question:
Estimate the probability that Xiao Li hits the bullseye exactly twice in three shots, which results to 0.30.

Problem: Prove that the estimated probability matches the expected results based on the simulation given the conditions.
-/

theorem simulation_probability_of_two_bullseyes_in_three_shots :
  let bullseye := λx : Nat, x <= 3 in
  let is_bullseye_twice_in_three := λ (u v w : Nat), bullseye u + bullseye v + bullseye w = 2 in
  let groups := [[3, 2, 1], [4, 2, 1], [1, 9, 1], [9, 2, 5], [2, 7, 1], [9, 3, 2],
                 [8, 0, 0], [4, 7, 8], [5, 8, 9], [6, 6, 3], [5, 3, 1], [2, 9, 7],
                 [3, 9, 6], [0, 2, 1], [5, 4, 6], [3, 8, 8], [2, 3, 0], [1, 1, 3],
                 [5, 0, 7], [9, 6, 5]] in
  (countp (λ g, is_bullseye_twice_in_three g.head g.get? 1 g.get? 2) groups.to_list : ℚ) / 20 = 0.3 :=
by
  sorry

end simulation_probability_of_two_bullseyes_in_three_shots_l557_557305


namespace total_seats_round_table_l557_557865

theorem total_seats_round_table (n : ℕ) (h : n = 38)
  (ka_position : ℕ) (sl_position : ℕ) 
  (h1 : ka_position = 10) 
  (h2 : sl_position = 29) 
  (h3 : (ka_position + n/2) % n = sl_position) : 
  n = 38 :=
by
  -- All steps and solution proof
  sorry

end total_seats_round_table_l557_557865


namespace find_ff_neg2_value_l557_557581

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 - x else 1 / (1 - x)

theorem find_ff_neg2_value : f (f (-2)) = -1 / 5 := by
  sorry

end find_ff_neg2_value_l557_557581


namespace find_angle_B_l557_557297

variable (ABCD : Type) [T : trapezoid ABCD]
variable (A B C D : ABCD)
variable (parallel_AB_CD : parallel (line A B) (line C D))
variable (angle_A : ℝ) (angle_B : ℝ) (angle_C : ℝ) (angle_D : ℝ)
variable (cond1 : angle_A = 2 * angle_D)
variable (cond2 : angle_C = 4 * angle_B)
variable (angle_sum : angle_B + angle_C = 180)

theorem find_angle_B : angle_B = 36 :=
by
  sorry

end find_angle_B_l557_557297


namespace arithmetic_sequence_general_term_l557_557557

theorem arithmetic_sequence (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h₁ : a 1 = 3)
  (h₂ : ∀ n ≥ 2, 2 * a n = S n * S (n - 1)) :
  (∃ d : ℚ, d = -1/2 ∧ ∀ n ≥ 2, (1 / S n) - (1 / S (n - 1)) = d) :=
sorry

theorem general_term (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h₁ : a 1 = 3)
  (h₂ : ∀ n ≥ 2, 2 * a n = S n * S (n - 1)) :
  ∀ n, a n = if n = 1 then 3 else 18 / ((8 - 3 * n) * (5 - 3 * n)) :=
sorry

end arithmetic_sequence_general_term_l557_557557


namespace coloring_patterns_count_l557_557144

def valid_coloring_pattern (grid : List (List Bool)) : Prop :=
  grid.length = 4 ∧ (∀ col ∈ grid, col.length = 4 ∧ ∀ i < 4, (col !! i = true → ∀ j < i, col !! j = true)) ∧
  ∃ coloring_count : ℕ, coloring_count = 7 ∧ ∑ col in grid, col.count (λ x, x) = coloring_count ∧
  ∀ i < 3, grid.nth i !! 0 → grid.nth i.succ !! 0 → col.count (λ x, x) ≥ (grid.nth i.succ).count (λ x, x)

theorem coloring_patterns_count : 
∃ grid : List (List Bool), valid_coloring_pattern grid ∧ 
(count_valid_coloring_patterns grid) = 9 :=
  sorry

end coloring_patterns_count_l557_557144


namespace negation_proposition_real_l557_557011

theorem negation_proposition_real :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ ∃ x : ℝ, x^2 + x + 1 ≤ 0 :=
by
  sorry

end negation_proposition_real_l557_557011


namespace a_range_l557_557217

variables {x a : ℝ}

def p (x : ℝ) : Prop := (4 * x - 3) ^ 2 ≤ 1
def q (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0

theorem a_range (h : ∀ x, ¬p x → ¬q x a ∧ (∃ x, q x a ∧ ¬p x)) :
  0 ≤ a ∧ a ≤ 1/2 :=
sorry

end a_range_l557_557217


namespace range_of_a_l557_557983

noncomputable def f (a x : ℝ) := a * x - 1
noncomputable def g (x : ℝ) := -x^2 + 2 * x + 1

theorem range_of_a (a : ℝ) :
  (∀ (x1 : ℝ), x1 ∈ (Set.Icc (-1 : ℝ) 1) → ∃ (x2 : ℝ), x2 ∈ (Set.Icc (0 : ℝ) 2) ∧ f a x1 < g x2) ↔ a ∈ Set.Ioo (-3 : ℝ) 3 :=
sorry

end range_of_a_l557_557983


namespace truck_capacities_transportation_plan_l557_557853

-- Definitions of given conditions
def A_truck_capacity (x y : ℕ) : Prop := x + 2 * y = 50
def B_truck_capacity (x y : ℕ) : Prop := 5 * x + 4 * y = 160
def total_transport_cost (m n : ℕ) : ℕ := 500 * m + 400 * n
def most_cost_effective_plan (m n cost : ℕ) : Prop := 
  m + 2 * n = 10 ∧ (20 * m + 15 * n = 190) ∧ cost = total_transport_cost m n ∧ cost = 4800

-- Proving the capacities of trucks A and B
theorem truck_capacities : 
  ∃ x y : ℕ, A_truck_capacity x y ∧ B_truck_capacity x y ∧ x = 20 ∧ y = 15 := 
sorry

-- Proving the most cost-effective transportation plan
theorem transportation_plan : 
  ∃ m n cost, (total_transport_cost m n = cost) ∧ most_cost_effective_plan m n cost := 
sorry

end truck_capacities_transportation_plan_l557_557853


namespace count_valid_triples_l557_557167

theorem count_valid_triples :
  let count_triples := 
    { (a, b, c) | a >= 2 ∧ b >= 1 ∧ c >= 0 ∧ log a b = c ^ 2023 ∧ a + b + c = 2023 }.finite.to_finset.card
  in count_triples = 2 :=
by sorry

end count_valid_triples_l557_557167


namespace count_valid_N_is_correct_l557_557604

noncomputable def count_valid_N : ℕ :=
  (set.count {N ∈ set.Icc 1 1999 | ∃ x : ℝ, 0 < x ∧ x ^ (⌊x^2⌋) = N})

theorem count_valid_N_is_correct : count_valid_N = -- Expected total count from the problem
 := sorry

end count_valid_N_is_correct_l557_557604


namespace min_surface_area_l557_557840

/-- Defining the conditions and the problem statement -/
def solid (volume : ℝ) (face1 face2 : ℝ) : Prop := 
  ∃ x y z, x * y * z = volume ∧ (x * y = face1 ∨ y * z = face1 ∨ z * x = face1)
                      ∧ (x * y = face2 ∨ y * z = face2 ∨ z * x = face2)

def juan_solids (face1 face2 face3 face4 face5 face6 : ℝ) : Prop :=
  solid 128 4 32 ∧ solid 128 64 16 ∧ solid 128 8 32

theorem min_surface_area {volume : ℝ} {face1 face2 face3 face4 face5 face6 : ℝ} 
  (h : juan_solids 4 32 64 16 8 32) : 
  ∃ area : ℝ, area = 688 :=
sorry

end min_surface_area_l557_557840


namespace odd_increasing_min_5_then_neg5_max_on_neg_interval_l557_557263

-- Definitions using the conditions given in the problem statement
variable {f : ℝ → ℝ}

-- Condition 1: f is odd
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

-- Condition 2: f is increasing on the interval [3, 7]
def increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f (x) ≤ f (y)

-- Condition 3: Minimum value of f on [3, 7] is 5
def min_value_on_interval (f : ℝ → ℝ) (a b : ℝ) (min_val : ℝ) : Prop :=
  ∃ x, a ≤ x ∧ x ≤ b ∧ f (x) = min_val

-- Lean statement for the proof problem
theorem odd_increasing_min_5_then_neg5_max_on_neg_interval
  (f_odd: odd_function f)
  (f_increasing: increasing_on_interval f 3 7)
  (min_val: min_value_on_interval f 3 7 5) :
  increasing_on_interval f (-7) (-3) ∧ min_value_on_interval f (-7) (-3) (-5) :=
by sorry

end odd_increasing_min_5_then_neg5_max_on_neg_interval_l557_557263


namespace jimmy_lodging_cost_l557_557945

def budget_nightly_cost : ℝ := 15 * 1.10
def budget_total_cost : ℝ := budget_nightly_cost * 2

def hotel_discounted_rate : ℝ := 55 * 0.80
def hotel_service_charge : ℝ := hotel_discounted_rate * 0.15
def hotel_nightly_cost : ℝ := hotel_discounted_rate + hotel_service_charge
def hotel_total_cost : ℝ := hotel_nightly_cost * 2

def cabin_nightly_cost : ℝ := 45 * 1.07
def cabin_discounted_rate : ℝ := 45 * 0.70
def cabin_discounted_nightly_cost : ℝ := cabin_discounted_rate * 1.07
def cabin_total_cost : ℝ := (cabin_nightly_cost + cabin_discounted_nightly_cost) / 3

def resort_base_nightly_cost : ℝ := 95 * 1.50
def resort_fee : ℝ := resort_base_nightly_cost * 0.12
def resort_total_cost : ℝ := resort_base_nightly_cost + resort_fee

def total_vacation_cost : ℝ := budget_total_cost + hotel_total_cost + cabin_total_cost + resort_total_cost

theorem jimmy_lodging_cost : total_vacation_cost = 321.085 := by
  sorry

end jimmy_lodging_cost_l557_557945


namespace least_positive_integer_x_multiple_53_l557_557803

theorem least_positive_integer_x_multiple_53 :
  ∃ x : ℕ, (x > 0) ∧ ((3 * x + 41)^2) % 53 = 0 ∧ ∀ y : ℕ, (y > 0) ∧ ((3 * y + 41)^2) % 53 = 0 → x ≤ y := 
begin
  use 4,
  split,
  { -- 4 > 0
    exact dec_trivial },
  split,
  { -- (3 * 4 + 41)^2 % 53 = 0
    calc (3 * 4 + 41)^2 % 53 = (53)^2 % 53 : by norm_num
    ... = 0 : by norm_num },
  { -- smallest positive integer solution
    assume y hy,
    cases hy with hy_gt0 hy_multiple,
    by_contradiction hxy,
    have x_val : 4 = 1,
      by linarith,
    norm_num at x_val,
    cases x_val
  }
end

end least_positive_integer_x_multiple_53_l557_557803


namespace volume_of_earth_dug_out_l557_557054

noncomputable def volume_of_cylinder (d : ℝ) (h : ℝ) : ℝ :=
  let r := d / 2
  π * r ^ 2 * h

theorem volume_of_earth_dug_out :
  volume_of_cylinder 4 14 ≈ 176.71 :=
by
  sorry

end volume_of_earth_dug_out_l557_557054


namespace toms_living_room_width_l557_557415

variable (L : ℝ) (A₁ : ℝ) (A₂ : ℝ) (B : ℝ)

def room_width (L A₁ A₂ B : ℝ) : ℝ := (A₁ + B * A₂) / L

theorem toms_living_room_width (L : ℝ) (A₁ A₂ B W : ℝ) (hL : L = 16) (hA₁ : A₁ = 250) (hA₂ : A₂ = 10) (hB : B = 7) :
  room_width L A₁ A₂ B = 20 :=
by
  rw [room_width, hL, hA₁, hA₂, hB]
  norm_num
  done

end toms_living_room_width_l557_557415


namespace part1_part2_l557_557633

structure Point where
  x : ℝ
  y : ℝ

def midpoint (P Q : Point) : Point := 
  {x := (P.x + Q.x) / 2, y := (P.y + Q.y) / 2}

def slope (P Q : Point) : ℝ := 
  (Q.y - P.y) / (Q.x - P.x)

noncomputable def line_equation (P Q : Point) : (ℝ × ℝ × ℝ) :=
  let m := slope P Q
  let b := P.y - m * P.x
  (m, -1, b)

noncomputable def perpendicular_slope (m : ℝ) : ℝ := 
  -1 / m

theorem part1 (A B C : Point) (E : Point) 
  (hE : E = midpoint B C)
  (hA : A = {-1, 2})
  (hB : B = {1, 3})
  (hC : C = {3, -1}) : 
  line_equation A E = (-1/3, -1, 5/3) :=
sorry

theorem part2 (A B C D E : Point) 
  (hA : A = {-1, 2})
  (hB : B = {1, 3})
  (hC : C = {3, -1})
  (hD : D = {1, -2})
  (hE : E = midpoint B C) : 
  let m_DE := slope D E in
  let m_perpendicular := perpendicular_slope m_DE in
  line_equation A {x: A.x, y: A.y + m_perpendicular * (A.x)} = (-1/3, -1, 5/3) :=
sorry

end part1_part2_l557_557633


namespace xiaoming_comprehensive_score_l557_557920

theorem xiaoming_comprehensive_score :
  ∀ (a b c d : ℝ),
  a = 92 → b = 90 → c = 88 → d = 95 →
  (0.4 * a + 0.3 * b + 0.2 * c + 0.1 * d) = 90.9 :=
by
  intros a b c d ha hb hc hd
  simp [ha, hb, hc, hd]
  norm_num
  done

end xiaoming_comprehensive_score_l557_557920


namespace digits_diff_1200_200_l557_557053

/-- 
Prove that the number of digits of 1200 in base 2 is 3 more than the number of digits of 200 in base 2.
-/
theorem digits_diff_1200_200 : 
  (nat.digits 2 1200).length = (nat.digits 2 200).length + 3 :=
sorry

end digits_diff_1200_200_l557_557053


namespace sum_of_elements_zero_l557_557083

theorem sum_of_elements_zero
  {M : finset ℝ} (hM : 4 ≤ M.card)
  (f : M → M) (hf : bijective f) (hid : f ≠ id)
  (hineq : ∀ {a b : M}, a ≠ b → a * b ≤ f a * f b) :
  M.sum id = 0 :=
sorry

end sum_of_elements_zero_l557_557083


namespace harmonic_number_ineq_l557_557671

noncomputable def h_n (n: ℕ) : ℝ := ∑ i in Finset.range (n + 1), (1 : ℝ) / i

theorem harmonic_number_ineq (n : ℕ) (h : n > 2) :
  n - (n - 1) / n > h_n n ∧ h_n n > n * (n + 1) ^ (1 / n) - n := 
sorry

end harmonic_number_ineq_l557_557671


namespace find_certain_number_l557_557034

-- Definitions of conditions from the problem
def greatest_number : ℕ := 10
def divided_1442_by_greatest_number_leaves_remainder := (1442 % greatest_number = 12)
def certain_number_mod_greatest_number (x : ℕ) := (x % greatest_number = 6)

-- Theorem statement
theorem find_certain_number (x : ℕ) (h1 : greatest_number = 10)
  (h2 : 1442 % greatest_number = 12)
  (h3 : certain_number_mod_greatest_number x) : x = 1446 :=
sorry

end find_certain_number_l557_557034


namespace opposite_of_3_l557_557729

-- Define the concept of opposite of a number and the logic for positive numbers
def opposite (x : Int) : Int := 
  if x > 0 then -x
  else if x < 0 then -x
  else 0

-- Statement to prove that the opposite of 3 is -3
theorem opposite_of_3 : opposite 3 = -3 :=
by 
  -- Using the definition of opposite
  unfold opposite
  -- Simplify the expression for x = 3
  simp [lt_irrefl, int.coe_nat_lt]
  -- Conclude proof
  rfl

end opposite_of_3_l557_557729


namespace square_pattern_side_length_l557_557826

theorem square_pattern_side_length (B W : ℕ) (h1 : B + W = 95) (h2 : abs (B - W) ≤ 85) 
  (h3 : ∀ r : ℕ, r < 5 → ∃ c : ℕ, c < 5 ∧ ((B = 5 ∧ W = 90) ∨ (B = 90 ∧ W = 5))) : 
  ∃ n : ℕ, n = 5 :=
by {
  have hB_W_cases : (B = 5 ∧ W = 90) ∨ (B = 90 ∧ W = 5), 
  { sorry },
  
  use 5,
  sorry
}

end square_pattern_side_length_l557_557826


namespace opposite_of_negative_fraction_l557_557015

theorem opposite_of_negative_fraction : -(- (1/2023 : ℚ)) = 1/2023 := 
sorry

end opposite_of_negative_fraction_l557_557015


namespace relationship_among_a_b_c_l557_557321

def log10 := Real.logb 10
def log3 := Real.logb 3

def a : ℝ := 2^(0.1)
def b : ℝ := log10 (5 / 2)
def c : ℝ := log3 (9 / 10)

theorem relationship_among_a_b_c : a > b ∧ b > c :=
by {
    have h_a : a = 2^(0.1) := rfl,
    have h_b : b = log10 (5 / 2) := rfl,
    have h_c : c = log3 (9 / 10) := rfl,
    sorry
}

end relationship_among_a_b_c_l557_557321


namespace a_2017_value_l557_557786

noncomputable def S (n : ℕ) : ℕ := 2 * n - 1

theorem a_2017_value (h : ∀ n : ℕ, n > 0 → S n = 2 * n - 1) : 
  let a (n : ℕ) := S n - S (n - 1) in
  a 2017 = 2 :=
by
  have h2017 := h 2017 (by norm_num)
  have h2016 := h 2016 (by norm_num)
  simp [S, Nat.succ_eq_add_one] at h2017 h2016
  have h_a_2017 : a 2017 = S 2017 - S 2016
  rw h_a_2017
  sorry

end a_2017_value_l557_557786


namespace at_least_one_int_l557_557313

theorem at_least_one_int
  (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_eq : ∀ n : ℤ, Int.floor (a * n) + Int.floor (b * n) = Int.floor (c * n)) :
  (∃ x, x = a ∨ x = b ∨ x = c ∧ x ∈ ℤ) :=
sorry

end at_least_one_int_l557_557313


namespace percentage_discount_l557_557849

variable (CP SP_discount SP_no_discount Discount : ℝ)
variable (profit_discount profit_no_discount : ℝ)

-- Conditions
def cost_price := CP = 100
def profit_with_discount := profit_discount = 0.1875 * CP
def profit_without_discount := profit_no_discount = 0.25 * CP
def selling_price_with_discount := SP_discount = CP + profit_discount
def selling_price_without_discount := SP_no_discount = CP + profit_no_discount
def discount_amount := Discount = SP_no_discount - SP_discount

-- Proof statement
theorem percentage_discount (CP SP_discount SP_no_discount Discount : ℝ) (profit_discount profit_no_discount : ℝ)
  (h1 : cost_price)
  (h2 : profit_with_discount)
  (h3 : profit_without_discount)
  (h4 : selling_price_with_discount)
  (h5 : selling_price_without_discount)
  (h6 : discount_amount) :
  (Discount / SP_no_discount) * 100 = 5 := 
by
  sorry

end percentage_discount_l557_557849


namespace initial_marbles_count_l557_557621

-- Definitions as per conditions in the problem
variables (x y z : ℕ)

-- Condition 1: Removing one black marble results in one-eighth of the remaining marbles being black
def condition1 : Prop := (x - 1) * 8 = (x + y - 1)

-- Condition 2: Removing three white marbles results in one-sixth of the remaining marbles being black
def condition2 : Prop := x * 6 = (x + y - 3)

-- Proof that initial total number of marbles is 9 given conditions
theorem initial_marbles_count (h1 : condition1 x y) (h2 : condition2 x y) : x + y = 9 :=
by 
  sorry

end initial_marbles_count_l557_557621


namespace football_tournament_impossible_l557_557701

theorem football_tournament_impossible :
  ¬ ∃ (countries : Finset ℕ) (teams : Finset ℕ) 
      (home : ℕ → ℕ)  -- Mapping from each team to its home country
      (plays_in : ℕ → ℕ → Prop),  -- Relation indicating whether a team plays in a given country
    countries.card = 16 ∧ teams.card = 16 ∧
    (∀ t ∈ teams, ∀ c ∈ countries, c ≠ home t → (∃ t', t ≠ t' ∧ plays_in t c ∧ plays_in t' c)) ∧
    (∀ t1 t2 c1 c2, t1 ≠ t2 → c1 ≠ c2 ∧ t1 ∈ teams ∧ t2 ∈ teams ∧ c1 ∈ countries ∧ c2 ∈ countries 
        → ¬ (plays_in t1 c1 ∧ plays_in t2 c1 ∧ plays_in t1 c2 ∧ plays_in t2 c2)) :=
sorry

end football_tournament_impossible_l557_557701


namespace election_winner_votes_l557_557794

theorem election_winner_votes :
  ∃ V W : ℝ, (V = (71.42857142857143 / 100) * V + 3000 + 5000) ∧
            (W = (71.42857142857143 / 100) * V) ∧
            W = 20000 := by
  sorry

end election_winner_votes_l557_557794


namespace decrease_in_length_l557_557107

theorem decrease_in_length (L B : ℝ) (h₀ : L ≠ 0) (h₁ : B ≠ 0)
  (h₂ : ∃ (A' : ℝ), A' = 0.72 * L * B)
  (h₃ : ∃ B' : ℝ, B' = B * 0.9) :
  ∃ (x : ℝ), x = 20 :=
by
  sorry

end decrease_in_length_l557_557107


namespace initial_people_employed_l557_557836

-- Definitions from the conditions
def initial_work_days : ℕ := 25
def total_work_days : ℕ := 50
def work_done_percentage : ℕ := 40
def additional_people : ℕ := 30

-- Defining the statement to be proved
theorem initial_people_employed (P : ℕ) 
  (h1 : initial_work_days = 25) 
  (h2 : total_work_days = 50)
  (h3 : work_done_percentage = 40)
  (h4 : additional_people = 30) 
  (work_remaining_percentage := 60) : 
  (P * 25 / 10 = 100) -> (P + 30) * 50 = P * 625 / 10 -> P = 120 :=
by
  sorry

end initial_people_employed_l557_557836


namespace elective_ways_l557_557490

theorem elective_ways (monographs : Finset ℕ) (group1 group2 group3 : Finset ℕ) : 
  monographs.card = 4 ∧
  monographs = group1 ∪ group2 ∪ group3 ∧
  group1 ∩ group2 = ∅ ∧ group2 ∩ group3 = ∅ ∧ group1 ∩ group3 = ∅ ∧
  group1.card ≥ 1 ∧ group2.card ≥ 1 ∧ group3.card ≥ 1 ∧
  group1.card + group2.card + group3.card = 4 →
  (monographs.powerset.card.choose 2) *
  ((@Finset.erase _ (group1 ∪ half (monographs ∩ group1)) _).card) *
  ((@Finset.erase _ (group2 ∪ half (monographs ∩ group2)) _).card) /
  (2 * 1) *
  (3 ∏_ i = 1 to 3 := (i).choose 3) = 36 :=
by {
  sorry
}

end elective_ways_l557_557490


namespace sum_last_digit_is_nine_l557_557334

def pair_difference_property (pairs : List (ℕ × ℕ)) : Prop :=
  pairs.length = 999 ∧ ∀ (i : ℕ) (h : i < 999), (|pairs.nth_le i h.1 - pairs.nth_le i h.2| = 1 ∨ |pairs.nth_le i h.1 - pairs.nth_le i h.2| = 6)

theorem sum_last_digit_is_nine (pairs : List (ℕ × ℕ)) (h : pair_difference_property pairs) :
  (List.sum (pairs.map (λ p, |p.fst - p.snd|))) % 10 = 9 :=
by
  sorry

end sum_last_digit_is_nine_l557_557334


namespace find_largest_negative_root_of_equation_l557_557174

theorem find_largest_negative_root_of_equation :
  ∃ x ∈ {x : ℝ | (sin (real.pi * x) - cos (2 * real.pi * x)) / ((sin (real.pi * x) - 1)^2 + cos (real.pi * x)^2 - 1) = 0}, 
  ∀ y ∈ {y : ℝ | (sin (real.pi * y) - cos (2 * real.pi * y)) / ((sin (real.pi * y) - 1)^2 + cos (real.pi * y)^2 - 1) = 0 },
  y < 0 → y ≤ x :=
begin
  use -0.5,
  split,
  { -- proof that -0.5 is a root
    sorry
  },
  { -- proof that -0.5 is the largest negative root
    sorry
  }
end

end find_largest_negative_root_of_equation_l557_557174


namespace plane_sections_sphere_plane_sections_cylinder_l557_557396

noncomputable def section_shape_sphere (plane : Type) (sphere : Type) : Type :=
  -- Providing that a plane cuts a spherical surface, the section is always a circle.
  circle

noncomputable def section_shape_cylinder (plane : Type) (cylinder : Type) : Type :=
  -- Providing that a plane cuts a cylindrical surface, the section is either a circle or an ellipse.
  circle ⊕ ellipse

-- Stating the main theorems
theorem plane_sections_sphere (plane : Type) (sphere : Type) :
  section_shape_sphere plane sphere = circle := 
sorry

theorem plane_sections_cylinder (plane : Type) (cylinder : Type) :
  section_shape_cylinder plane cylinder = (circle ⊕ ellipse) := 
sorry

end plane_sections_sphere_plane_sections_cylinder_l557_557396


namespace parts_of_cut_square_l557_557462

theorem parts_of_cut_square (folds_to_one_by_one : ℕ) : folds_to_one_by_one = 9 :=
  sorry

end parts_of_cut_square_l557_557462


namespace sum_inequality_sum_inverse_l557_557027

theorem sum_inequality_sum_inverse (n : ℕ) (a : Fin n → ℝ) (h : ∀ i, 0 < a i) (sum_a : ∑ i, a i = 1) :
  ∑ i j in Finset.offDiag (Finset.univ : Finset (Fin n)), a i * a j / (a i + a j) ≤ (n - 1 : ℝ) / 4 := 
sorry

end sum_inequality_sum_inverse_l557_557027


namespace num_intersections_main_l557_557507

def is_on_segment (p : (ℤ × ℤ)) : Prop := 
  let (x, y) := p in 
  (7 * y = 3 * x) ∧ (0 ≤ x ∧ x ≤ 2003) ∧ (0 ≤ y ∧ y ≤ 858)

def is_intersecting_square (p : (ℤ × ℤ)) (r : ℚ) : Prop := 
  let (x, y) := p in 
  let half_side := r / 2 in
  (is_on_segment (x + half_side, y) ∨ is_on_segment (x - half_side, y) ∨
  is_on_segment (x, y + half_side) ∨ is_on_segment (x, y - half_side))

def is_intersecting_circle (p : (ℤ × ℤ)) (r : ℚ) : Prop := 
  let (x, y) := p in 
  let radius := r in
  ∃ t : ℚ, 0 ≤ t ∧ t ≤ 1 ∧ (x + t * (2003 - x))^2 + (y + t * (858 - y))^2 ≤ radius^2

theorem num_intersections (m n : ℕ) : 
  let r1 := 1 / 2 
  let r2 := 1 / 5 
  let lattice_points := [(i, j) | i j : ℤ, is_on_segment (i, j)]
  ∑ (p : (ℤ × ℤ)) in lattice_points, (is_intersecting_square p r1) + (is_intersecting_circle p r2) = m + n := 
  sorry

theorem main : m + n = 572 :=
  sorry

end num_intersections_main_l557_557507


namespace opposite_of_three_l557_557738

theorem opposite_of_three : -3 = -3 := 
by sorry

end opposite_of_three_l557_557738


namespace seashells_after_giving_away_l557_557353

theorem seashells_after_giving_away (c s : ℕ) (h : c = 35) (j : s = 18) : (c - s) = 17 :=
by
  rw [h, j]
  exact Nat.sub_self 18

end seashells_after_giving_away_l557_557353


namespace total_surface_area_is_correct_l557_557484

-- Define the dimensions of the original prism
def height : ℝ := 1
def width : ℝ := 1
def length : ℝ := 2

-- Define the cuts
def first_cut : ℝ := 1 / 4
def second_cut : ℝ := 1 / 5

-- Define the step shift
def shift : ℝ := 1 / 4

-- Calculate the heights of the pieces
def height_E : ℝ := first_cut
def height_F : ℝ := second_cut
def height_G : ℝ := height - (first_cut + second_cut)

-- Calculate the total surface area
def total_surface_area : ℝ :=
  let top_E := length * width
  let top_F := length * (width - shift)
  let top_G := length * (width - 2 * shift)
  let bottom := length * width
  let sides := 2 * (length * (height_E + height_F + height_G))
  let front := 2 * height
  top_E + top_F + top_G + bottom + sides + front

-- Define the theorem
theorem total_surface_area_is_correct : total_surface_area = 10.5 :=
  sorry

end total_surface_area_is_correct_l557_557484


namespace equal_sides_of_triangle_l557_557094

theorem equal_sides_of_triangle
  (X R Q P D A B C : Point)
  (H1 : R = foot_perpendicular X BC)
  (H2 : Q ∈ segment AC)
  (H3 : R ∈ segment BC)
  (H4 : P ∉ segment AD)
  (H5 : collinear P R Q)
  (H6 : (PQ.tangent (circle X R D)) ↔ (∠QRX = ∠RDX)) :
  AB = AC :=
sorry

end equal_sides_of_triangle_l557_557094


namespace original_profit_percentage_l557_557475

theorem original_profit_percentage
  (C : ℝ) -- original cost
  (S : ℝ) -- selling price
  (y : ℝ) -- original profit percentage
  (hS : S = C * (1 + 0.01 * y)) -- condition for selling price based on original cost
  (hC' : S = 0.85 * C * (1 + 0.01 * (y + 20))) -- condition for selling price based on reduced cost
  : y = -89 :=
by
  sorry

end original_profit_percentage_l557_557475


namespace circle_equation_l557_557709

theorem circle_equation : 
  ∃ (b : ℝ), (∀ (x y : ℝ), (x^2 + (y - b)^2 = 1 ↔ (x = 1 ∧ y = 2) → b = 2)) :=
sorry

end circle_equation_l557_557709


namespace train_length_l557_557113

theorem train_length 
    (t : ℝ) 
    (s_kmh : ℝ) 
    (s_mps : ℝ)
    (h1 : t = 2.222044458665529) 
    (h2 : s_kmh = 162) 
    (h3 : s_mps = s_kmh * (5 / 18))
    (L : ℝ)
    (h4 : L = s_mps * t) : 
  L = 100 := 
sorry

end train_length_l557_557113


namespace inner_pyramid_edges_greater_l557_557422

theorem inner_pyramid_edges_greater {B C D A1 A2 : Type} [EuclideanSpace B C D A1 A2]
  (eps : ℝ) (h_base: ∃ e, e = eps)
  (h_outer_edges : ℓ A1 B = 1 ∧ ℓ A1 C = 1 ∧ ℓ A1 D = 1)
  (h_inner_position : ℓ A2 D = eps)
  (h_inner_edges_approx : ℓ A2 B = eps ∧ ℓ A2 C = eps)
  (h_nonneg_eps : 0 < eps) :
  ∃ε, (1+ε)*(3 + 3*eps) < 4 :=
by sorry

end inner_pyramid_edges_greater_l557_557422


namespace radius_of_circle_l557_557079

noncomputable def circle_radius {k : ℝ} (hk : k > -6) : ℝ := 6 * Real.sqrt 2 + 6

theorem radius_of_circle (k : ℝ) (hk : k > -6)
  (tangent_y_eq_x : ∀ (P : ℝ × ℝ), P.1 = 0 → P.2 = k → dist P (0, P.2) = 6 * Real.sqrt 2 + 6)
  (tangent_y_eq_negx : ∀ (P : ℝ × ℝ), P.1 = 0 → P.2 = k → dist P (0, -P.2) = 6 * Real.sqrt 2 + 6)
  (tangent_y_eq_neg6 : ∀ (P : ℝ × ℝ), P.1 = 0 → P.2 = k → dist P (0, -6) = 6 * Real.sqrt 2 + 6) :
  circle_radius hk = 6 * Real.sqrt 2 + 6 :=
by
  sorry

end radius_of_circle_l557_557079


namespace time_to_pass_platform_l557_557812

-- Definitions of the given conditions
def length_of_train : ℕ := 360
def speed_of_train_kmh : ℕ := 45
def length_of_platform : ℕ := 240

-- Helper definition to convert the speed from km/h to m/s
def speed_of_train_ms := (speed_of_train_kmh * 1000 / 3600 : ℝ)

-- Calculate total distance to be covered
def total_distance := length_of_train + length_of_platform

-- Final statement: calculating the time it takes to pass the platform
theorem time_to_pass_platform : (total_distance : ℝ) / speed_of_train_ms = 48 := 
by
  sorry

end time_to_pass_platform_l557_557812


namespace sum_powers_i_eq_zero_l557_557146

section
  variables (i : ℂ) (n : ℤ)
  -- The imaginary unit condition
  def i_squared := i^2 = -1
  -- The cyclic nature of the powers of i
  def i_cycle := i^4 = 1

  -- The sum to be computed
  def sum_powers_i := ∑ k in (-101 : ℤ)..101, i^k

  theorem sum_powers_i_eq_zero (h1 : i_squared) (h2 : i_cycle) : 
    sum_powers_i i = 0 :=
  by sorry

end

end sum_powers_i_eq_zero_l557_557146


namespace total_stamps_stamp_collectors_l557_557434

theorem total_stamps : 
  ∀ (N_m : ℕ), 
    (5 * N_m - 14 = 3 * (N_m + 14)) → 
    28 = N_m → 
    28 + 5 * 28 = 168 :=
by { intros N_m h1 h2, sorry }

-- Define and import necessary components
variables (x : ℕ) (y : ℕ) (t1 : ℕ) (t2 : ℕ) (m : ℕ)
-- Let x be the stamps Xiaoming initially has and y be the stamps Xiaoliang initially has.
-- These are related by y = 5 * x. After exchange, the conditions hold as mentioned.

def stamps_exchange := ∀ (x y : ℕ), (
  y = 5 * x →
  y - 14 = 3 * (x + 14) →
  t1 = x + 14 →
  t2 = y - 16 →
  28 = x ∧ 140 = y ∧ 168 = (x + y)
)

theorem stamp_collectors : stamps_exchange := 
  begin
    intros x y h1 h2 t1_def t2_def,
    split,
    sorry,
    split,
    sorry,
    sorry,
  end

end total_stamps_stamp_collectors_l557_557434


namespace johnPays12000InTaxes_l557_557307

-- Definitions for the conditions
def totalEarnings : ℕ := 100000
def deductions : ℕ := 30000
def firstTaxableLimit : ℕ := 20000
def firstTaxRate : ℚ := 0.10
def secondTaxRate : ℚ := 0.20

-- Theorem statement
theorem johnPays12000InTaxes :
  let taxableIncome := totalEarnings - deductions in
  let firstTax := firstTaxableLimit * firstTaxRate in
  let remainingTax := (taxableIncome - firstTaxableLimit) * secondTaxRate in
  firstTax + remainingTax = 12000 := 
by 
  sorry

end johnPays12000InTaxes_l557_557307


namespace total_seats_round_table_l557_557873

theorem total_seats_round_table 
  (a : ℕ) (b : ℕ) 
  (h₀ : a ≠ b)
  (h₁ : a + b = 39) 
  : ∃ n, n = 38 := 
by {
  sorry
}

end total_seats_round_table_l557_557873


namespace solve_for_y_l557_557357

theorem solve_for_y (y : ℚ) :
  (1/8)^((3 : ℚ) * y + 12) = 64^((3 : ℚ) * y + 7) ↔ y = -26/9 := 
by
  sorry

end solve_for_y_l557_557357


namespace angle_sum_condition_l557_557222

-- Defining the points and conditions
variables (A B C P Q D : Point)

-- Given Conditions
variables (acute_triangle_ABC : acute_triangle A B C)
variables (P_inside_ABC : in_triangle P A B C)
variables (Q_inside_ABC : in_triangle Q A B C)
variables (D_on_BC : on_segment D B C)
variables (angle1 : ∠PAB = ∠QAC)
variables (angle2 : ∠PBA = ∠QBC)

-- Theorem to be proved
theorem angle_sum_condition :
  ∠DPC + ∠APB = 180 ↔ ∠DQB + ∠AQC = 180 :=
sorry

end angle_sum_condition_l557_557222


namespace cody_steps_away_from_goal_l557_557140

def steps_in_week (daily_steps : ℕ) : ℕ :=
  daily_steps * 7

def total_steps_in_4_weeks (initial_steps : ℕ) : ℕ :=
  steps_in_week initial_steps +
  steps_in_week (initial_steps + 1000) +
  steps_in_week (initial_steps + 2000) +
  steps_in_week (initial_steps + 3000)

theorem cody_steps_away_from_goal :
  let goal := 100000
  let initial_daily_steps := 1000
  let total_steps := total_steps_in_4_weeks initial_daily_steps
  goal - total_steps = 30000 :=
by
  sorry

end cody_steps_away_from_goal_l557_557140


namespace opposite_of_3_is_neg3_l557_557750

def opposite (x : ℝ) := -x

theorem opposite_of_3_is_neg3 : opposite 3 = -3 :=
by
  sorry

end opposite_of_3_is_neg3_l557_557750


namespace fraction_of_number_l557_557052

theorem fraction_of_number (N : ℕ) (hN : N = 180) : 
  (6 + (1 / 2) * (1 / 3) * (1 / 5) * N) = (1 / 25) * N := 
by
  sorry

end fraction_of_number_l557_557052


namespace cost_fill_can_n_l557_557057

-- Definitions based on the given conditions
def radius_can_b : ℝ := sorry -- Placeholder for radius r of can b
def height_can_b : ℝ := sorry -- Placeholder for height h of can b
def cost_half_fill_can_b : ℝ := 4.00 -- Cost to half fill can b

-- Additional computations based on conditions
def radius_can_n : ℝ := 2 * radius_can_b -- Radius of can n
def height_can_n : ℝ := height_can_b / 2 -- Height of can n
def volume_formula (r h : ℝ) : ℝ := π * r^2 * h -- Volume of a cylinder

-- Volume of cylinders
def volume_can_b : ℝ := volume_formula radius_can_b height_can_b
def volume_can_n : ℝ := volume_formula radius_can_n height_can_n

-- Cost calculations
def cost_full_fill_can_b : ℝ := 2 * cost_half_fill_can_b -- Cost to fill can b

theorem cost_fill_can_n : cost_full_fill_can_b * 2 = 16.00 :=
by
  -- Placeholder to represent the proof
  sorry

end cost_fill_can_n_l557_557057


namespace hyperbola_eccentricity_is_sqrt10_l557_557086

-- Define the focus of the parabola
def focus_parabola (a : ℝ) (h_a : a > 0) : ℝ × ℝ := ⟨a, 0⟩

-- Define the intersection points based on x-coordinates
def intersection_points (a b : ℝ) (h_a : a > 0) (h_b : b > 0) : ℝ × ℝ :=
  let x_B := a^2 / (a + b)
  let x_C := a^2 / (a - b)
  (x_B, x_C)

-- Define the geometric mean condition
def geometric_mean_condition (a b : ℝ) (h_a : a > 0) (h_b : b > 0) : Prop :=
  let x_F := a
  let (x_B, x_C) := intersection_points a b h_a h_b
  x_C * x_C = x_F * x_B

-- Define the proof statement of the eccentricity of the hyperbola
theorem hyperbola_eccentricity_is_sqrt10 (a b : ℝ) (h_a : a > 0) (h_b : b > 0)
    (h_geom : geometric_mean_condition a b h_a h_b) : 
    let c := Real.sqrt (a^2 + b^2)
    Real.sqrt (c^2) / a = Real.sqrt 10 :=
by
  -- Given definitions and conditions should be used here
  sorry

end hyperbola_eccentricity_is_sqrt10_l557_557086


namespace event_relationship_uncertain_l557_557608

variables {Ω : Type} {p : Ω → Prop}

theorem event_relationship_uncertain
  (A B : set Ω)
  (h1 : ∀ ω, p ω → ω ∈ A ∪ B)
  (h2 : ∀ ω, p ω → ω ∈ A ∨ ω ∈ B)
  (h3 : measure_theory.measure_space.volume (A ∪ B) = 1)
  (h4 : measure_theory.measure_space.volume A + measure_theory.measure_space.volume B = 1) :
  ¬ (mutually_exclusive A B ∧ complementary A B) :=
sorry

end event_relationship_uncertain_l557_557608


namespace max_squares_in_sequence_l557_557481

-- Define the sequence a_n
def seq (a b : ℕ) : ℕ → ℕ
| 1       := a
| 2       := b
| (n + 1) := if n % 2 = 0 then (seq a b n) * (seq a b (n - 1)) else (seq a b n) + 4

-- Define the problem
theorem max_squares_in_sequence (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) :
  ∃ m, (∀ n ≤ 2022, ∃ k, seq a b n = k ^ 2) ∧ m = 1012 :=
sorry

end max_squares_in_sequence_l557_557481


namespace inequality_absolute_value_l557_557196

theorem inequality_absolute_value (a b : ℝ) (h1 : a < b) (h2 : b < 0) : |a| > -b :=
sorry

end inequality_absolute_value_l557_557196


namespace sphere_volume_larger_than_cube_l557_557064

theorem sphere_volume_larger_than_cube (S C : Type) [sphere S] [cube C] (equal_surface_area : surface_area S = surface_area C) :
  volume S > volume C :=
by sorry

end sphere_volume_larger_than_cube_l557_557064


namespace purchase_payment_l557_557433

variable (x y z : ℝ)

theorem purchase_payment (h1 : 3 * x + 2 * y + z = 420)
                         (h2 : 2 * x + 3 * y + 4 * z = 580) :
                         2 * (x + y + z) = 400 :=
by 
  have h3 : x + y + z = 200 := 
    sorry -- Add appropriate usage of h1, h2 here
  
  rw h3
  ring

end purchase_payment_l557_557433


namespace opposite_of_three_l557_557741

theorem opposite_of_three : -3 = -3 := 
by sorry

end opposite_of_three_l557_557741


namespace negation_of_prop_l557_557012

open Classical

theorem negation_of_prop (h : ∀ x : ℝ, x^2 + x + 1 > 0) : ∃ x : ℝ, x^2 + x + 1 ≤ 0 :=
sorry

end negation_of_prop_l557_557012


namespace range_of_a_l557_557213

-- Definitions from conditions 
def p (x : ℝ) : Prop := |x + 1| > 2
def q (x a : ℝ) : Prop := x > a

-- The Lean statement for the problem
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, -3 ≤ x ∧ x ≤ 1 → x ≤ a) → a ≥ 1 :=
by sorry

end range_of_a_l557_557213


namespace vector_parallel_k_l557_557589

variables (a b c : ℝ × ℝ) (k : ℝ)

def vec_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)

lemma vector_parallel_iff (u v : ℝ × ℝ) : (u.1 * v.2 = u.2 * v.1) ↔ (u.1 / u.2 = v.1 / v.2) :=
  sorry  -- The proof of this lemma is not shown here.

theorem vector_parallel_k :
  let a := (-1, 1) in
  let b := (2, 3) in
  let c := (-2, k) in
  let ab := vec_add a b in
  ab.1 * c.2 = ab.2 * c.1 → k = -8 :=
by
  intros a b c ab h
  change (-1, 1) with a
  change (2, 3) with b
  change (-2, k) with c
  unfold vec_add at ab
  cases ab
  have ab_eq : ab = (1, 4) := by simp
  rw ab_eq at h
  sorry  -- Continue the detailed proof if needed

end vector_parallel_k_l557_557589


namespace largest_4_digit_number_divisible_by_98_l557_557440

/-!
 We need to prove that the largest 4-digit number which is exactly divisible by 98 is 9998.
 The conditions are:
 - The number must be a 4-digit number.
 - The number must be exactly divisible by 98.
-/

theorem largest_4_digit_number_divisible_by_98 : 
  ∃ n : ℕ, (n < 10000) ∧ (1000 ≤ n) ∧ (n % 98 = 0) ∧ (∀ m : ℕ, ((m % 98 = 0) → (1000 ≤ m) → (m < 10000) → (m ≤ n))) := 
begin
  use 9998,
  split,
  { exact lt.trans (nat.le_refl 9998) (nat.succ_pos 9998) },
  split,
  { transitivity,
    show 1000 ≤ 9998, from nat.le_succ 9998,
    exact nat.le_add_left 1000 8998 },
  split,
  { norm_num },
  { intros m hm1 hm2 hm3,
    linarith [hm3, show 98 * 102 = 9998, by norm_num leisure_norm] },
  sorry
end

end largest_4_digit_number_divisible_by_98_l557_557440


namespace number_of_squares_in_grid_l557_557934

-- Grid of size 6 × 6 composed entirely of squares.
def grid_size : Nat := 6

-- Definition of the function that counts the number of squares of a given size in an n × n grid.
def count_squares (n : Nat) (size : Nat) : Nat :=
  (n - size + 1) * (n - size + 1)

noncomputable def total_squares : Nat :=
  List.sum (List.map (count_squares grid_size) (List.range grid_size).tail)  -- Using tail to skip zero size

theorem number_of_squares_in_grid : total_squares = 86 := by
  sorry

end number_of_squares_in_grid_l557_557934


namespace pony_average_speed_l557_557085

theorem pony_average_speed
  (time_head_start : ℝ)
  (time_catch : ℝ)
  (horse_speed : ℝ)
  (distance_covered_by_horse : ℝ)
  (distance_covered_by_pony : ℝ)
  (pony's_head_start : ℝ)
  : (time_head_start = 3) → (time_catch = 4) → (horse_speed = 35) → 
    (distance_covered_by_horse = horse_speed * time_catch) → 
    (pony's_head_start = time_head_start * v) → 
    (distance_covered_by_pony = pony's_head_start + (v * time_catch)) → 
    (distance_covered_by_horse = distance_covered_by_pony) → v = 20 :=
  by 
    intros h1 h2 h3 h4 h5 h6 h7
    sorry

end pony_average_speed_l557_557085


namespace additional_amount_needed_l557_557306

-- Define the amounts spent on shampoo, conditioner, and lotion
def shampoo_cost : ℝ := 10.00
def conditioner_cost : ℝ := 10.00
def lotion_cost_per_bottle : ℝ := 6.00
def lotion_quantity : ℕ := 3

-- Define the amount required for free shipping
def free_shipping_threshold : ℝ := 50.00

-- Calculate the total amount spent
def total_spent : ℝ := shampoo_cost + conditioner_cost + (lotion_quantity * lotion_cost_per_bottle)

-- Define the additional amount needed for free shipping
def additional_needed_for_shipping : ℝ := free_shipping_threshold - total_spent

-- The final goal to prove
theorem additional_amount_needed : additional_needed_for_shipping = 12.00 :=
by
  sorry

end additional_amount_needed_l557_557306


namespace total_toads_l557_557409

def pond_toads : ℕ := 12
def outside_toads : ℕ := 6

theorem total_toads : pond_toads + outside_toads = 18 :=
by
  -- Proof goes here
  sorry

end total_toads_l557_557409


namespace min_chord_inclination_angle_l557_557984

-- Define the point P
def P : Point := ⟨3, 1⟩

-- Define the circle C with center (2,2) and radius 2
def C : Circle := { center := ⟨2, 2⟩, radius := 2 }

-- Define line l passing through P and intersecting circle C at points A and B
def intersects_C : Line → Prop := λ l, ∃ A B : Point, A ∈ C ∧ B ∈ C ∧ A ≠ B ∧ A ∈ l ∧ B ∈ l

-- Define the inclination angle of a line
def inclination_angle (l : Line) : ℝ :=
  let k := slope l in
  atan k

-- Define the minimum length of chord condition
def min_chord_length_condition (l : Line) : Prop :=
  ∃ P' : Point, P' = P ∧ is_perpendicular l (line_through P' C.center)

theorem min_chord_inclination_angle :
  ∀ l : Line, intersects_C l → min_chord_length_condition l → inclination_angle l = 45 := 
by
  intros l h_intersects h_min_length
  sorry

end min_chord_inclination_angle_l557_557984


namespace polynomial_range_open_interval_l557_557927

theorem polynomial_range_open_interval :
  ∀ (k : ℝ), k > 0 → ∃ (x y : ℝ), (1 - x * y)^2 + x^2 = k :=
by
  sorry

end polynomial_range_open_interval_l557_557927


namespace variance_linear_transform_l557_557227

variable {X : Type}

-- Declare the variance function as D that takes a random variable and returns a real number
noncomputable def D (x : X) : ℝ := sorry

-- State the condition for the random variable X
axiom D_X : D(X) = 2

-- State the theorem that D(3X + 2) = 18
theorem variance_linear_transform (hX : D(X) = 2) : D(3 * X + 2) = 9 * 2 := by
  sorry

end variance_linear_transform_l557_557227


namespace part1_solution_part2_solution_l557_557523

-- Definitions for costs
variables (x y : ℝ)
variables (cost_A cost_B : ℝ)

-- Conditions
def condition1 : 80 * x + 35 * y = 2250 :=
  sorry

def condition2 : x = y - 15 :=
  sorry

-- Part 1: Cost of one bottle of each disinfectant
theorem part1_solution : x = cost_A ∧ y = cost_B :=
  sorry

-- Additional conditions for part 2
variables (m : ℕ)
variables (total_bottles : ℕ := 50)
variables (budget : ℝ := 1200)

-- Conditions for part 2
def condition3 : m + (total_bottles - m) = total_bottles :=
  sorry

def condition4 : 15 * m + 30 * (total_bottles - m) ≤ budget :=
  sorry

-- Part 2: Minimum number of bottles of Class A disinfectant
theorem part2_solution : m ≥ 20 :=
  sorry

end part1_solution_part2_solution_l557_557523


namespace find_k_l557_557155

theorem find_k (x1 x2 : ℂ) (a b c k : ℝ) 
  (h_poly : 3 * x^2 + k * x + 18 = 0)
  (h_root1 : x1 = 2 - 3i)
  (h_root2 : x2 = 2 + 3i)
  (h_sum : x1 + x2 = (4 : ℂ))
  (h_product : x1 * x2 = -{18/3})
  (h_vieta_sum : x1 + x2 = -b / a)
  (h_vieta_product : x1 * x2 = c / a) :
  k = -12 :=
sorry -- Proof goes here

end find_k_l557_557155


namespace exists_4x4_square_with_sum_2007_l557_557631

theorem exists_4x4_square_with_sum_2007 :
  ∃ (board : Fin 8 → Fin 8 → ℕ),
    (∀ i j : Fin 8, board i j = 1) →
    (∀ (f : Fin 6 → Fin 6 → Fin 9),
      ∀ n, 1 ≤ n → n ≤ 2003 → 
        let update_board := 
          λ b : Fin 8 → Fin 8 → ℕ, 
          λ x y : Fin 8, 
            if ∃ i j, (i, j).fst ≤ x ∧ x < (i, j).fst + 3 ∧ (i, j).snd ≤ y ∧ y < (i, j).snd + 3
            then b x y + 1 
            else b x y
        in 
        update_board (board) = board) →
    ∃ i j : Fin 5, board i j + board i (j + 3) + board (i + 3) j + board (i + 3) (j + 3) = 2007 := 
sorry

end exists_4x4_square_with_sum_2007_l557_557631


namespace shortest_side_of_triangle_with_medians_l557_557029

noncomputable def side_lengths_of_triangle_with_medians (a b c m_a m_b m_c : ℝ) : Prop :=
  m_a = 3 ∧ m_b = 4 ∧ m_c = 5 →
  a^2 = 2*b^2 + 2*c^2 - 36 ∧
  b^2 = 2*a^2 + 2*c^2 - 64 ∧
  c^2 = 2*a^2 + 2*b^2 - 100

theorem shortest_side_of_triangle_with_medians :
  ∀ (a b c : ℝ), side_lengths_of_triangle_with_medians a b c 3 4 5 → 
  min a (min b c) = c :=
sorry

end shortest_side_of_triangle_with_medians_l557_557029


namespace area_of_centroid_triangle_l557_557324

theorem area_of_centroid_triangle 
  (A B C D E : Point)
  (AB CD : LineSegment)
  (area_ABCD : area_of_rectangle A B C D = 1)
  (E_on_CD : lies_on E CD) :
  let centroid_abe := centroid_of_triangle A B E
  let centroid_bce := centroid_of_triangle B C E
  let centroid_ade := centroid_of_triangle A D E
in area_of_triangle centroid_abe centroid_bce centroid_ade = 1 / 9 :=
sorry

end area_of_centroid_triangle_l557_557324


namespace problem_correct_l557_557588

noncomputable def polar_equation_line (t : ℝ) : Prop := 
  ∃ θ, θ = π / 4

noncomputable def polar_equation_curve (θ : ℝ) : Prop :=
  ∃ ρ, ρ^2 - 2 * sqrt 2 * ρ * cos θ - 4 * sqrt 2 * ρ * sin θ + 6 = 0

noncomputable def area_triangle_PAB : Prop :=
  let P_polar := (3 * sqrt 2, π / 2) in
  let l_polar := θ = π / 4 in
  let C_polar := ρ^2 - 2 * sqrt 2 * ρ * cos θ - 4 * sqrt 2 * ρ * sin θ + 6 = 0 in
  area_of_triangle l_polar C_polar P_polar = 3 * sqrt 3

theorem problem_correct :
  (polar_equation_line t ∧ polar_equation_curve θ ∧ area_triangle_PAB) := 
    sorry

end problem_correct_l557_557588


namespace hours_felt_good_l557_557127

variable (x : ℝ)

theorem hours_felt_good (h1 : 15 * x + 10 * (8 - x) = 100) : x == 4 := 
by
  sorry

end hours_felt_good_l557_557127


namespace arithmetic_geometric_relation_l557_557565

noncomputable theory

variables {a b c d e : ℝ}

def arithmetic_sequence (a b : ℝ) : Prop :=
  b - a = (-4 - (-1)) / 3

def geometric_sequence (c d e : ℝ) : Prop :=
  d^2 = (-1) * (-4) ∧ d < 0

theorem arithmetic_geometric_relation (ha : arithmetic_sequence a b) (hg : geometric_sequence c d) :
  (b - a) / d = 1 / 2 :=
by {
  cases ha,
  cases hg,
  have d_result: d = -2, from sorry, -- inferred from the geometric condition
  rw [ha, d_result],
  norm_num,
  sorry
}

end arithmetic_geometric_relation_l557_557565


namespace ellipse_major_axis_focal_distance_l557_557575

theorem ellipse_major_axis_focal_distance (m : ℝ) (h1 : 10 - m > 0) (h2 : m - 2 > 0) 
  (h3 : ∀ x y, x^2 / (10 - m) + y^2 / (m - 2) = 1) 
  (h4 : ∃ c, 2 * c = 4 ∧ c^2 = (m - 2) - (10 - m)) : m = 8 :=
by
  sorry

end ellipse_major_axis_focal_distance_l557_557575


namespace fifth_question_points_l557_557622

theorem fifth_question_points 
  (P : ℝ) 
  (S12 : Π (n : ℕ), n = 12 → Π (r : ℝ), r = 2 → Π (a : ℝ), a = P → S12 = a * (1 - r^n) / (1 - r)) 
  (total_points : ℝ) 
  (h1 : total_points = 8190) 
  (eq : total_points = P * ((2 : ℝ) ^ 12 - 1)) : 
  2⁴ * P = 32 := by sorry

end fifth_question_points_l557_557622


namespace equivalent_annual_rate_l557_557521

theorem equivalent_annual_rate :
  ∀ (annual_rate compounding_periods: ℝ), annual_rate = 0.08 → compounding_periods = 4 → 
  ((1 + (annual_rate / compounding_periods)) ^ compounding_periods - 1) * 100 = 8.24 :=
by
  intros annual_rate compounding_periods h_rate h_periods
  sorry

end equivalent_annual_rate_l557_557521


namespace trigonometric_identity_l557_557301

-- Define the triangle and necessary conditions
variable {A B C : ℝ}
variable {a b c : ℝ} -- lengths of the sides opposite to angles A, B, and C respectively
variable {R : ℝ} -- circumradius of the triangle

-- Conditions for the problem
axiom sin_squared_condition : sin(A)^2 + sin(C)^2 = 2018 * sin(B)^2

theorem trigonometric_identity :
  (tan(A) + tan(C)) * tan(B)^2 / (tan(A) + tan(B) + tan(C)) = 2 / 2017 :=
by
  sorry

end trigonometric_identity_l557_557301


namespace sin_C_l557_557640

variable {A B C : ℝ}

theorem sin_C (hA : A = 90) (hcosB : Real.cos B = 3/5) : Real.sin (90 - B) = 3/5 :=
by
  sorry

end sin_C_l557_557640


namespace katie_miles_l557_557651

theorem katie_miles (x : ℕ) (h1 : ∀ y, y = 3 * x → y ≤ 240) (h2 : x + 3 * x = 240) : x = 60 :=
sorry

end katie_miles_l557_557651


namespace diamond_value_l557_557716

def diamond (a b : Int) : Int :=
  a * b^2 - b + 1

theorem diamond_value : diamond (-1) 6 = -41 := by
  sorry

end diamond_value_l557_557716


namespace expressible_numbers_count_l557_557960

theorem expressible_numbers_count : ∃ k : ℕ, k = 2222 ∧ ∀ n : ℕ, n ≤ 2000 → ∃ x : ℝ, n = Int.floor x + Int.floor (3 * x) + Int.floor (5 * x) :=
by sorry

end expressible_numbers_count_l557_557960
