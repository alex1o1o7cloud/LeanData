import Mathlib

namespace NUMINAMATH_GPT_target_hit_probability_l546_54654

-- Define the probabilities of Person A and Person B hitting the target
def prob_A_hits := 0.8
def prob_B_hits := 0.7

-- Define the probability that the target is hit when both shoot independently at the same time
def prob_target_hit := 1 - (1 - prob_A_hits) * (1 - prob_B_hits)

theorem target_hit_probability : prob_target_hit = 0.94 := 
by
  sorry

end NUMINAMATH_GPT_target_hit_probability_l546_54654


namespace NUMINAMATH_GPT_total_bananas_in_collection_l546_54669

theorem total_bananas_in_collection (g b T : ℕ) (h₀ : g = 196) (h₁ : b = 2) (h₂ : T = 392) : g * b = T :=
by
  sorry

end NUMINAMATH_GPT_total_bananas_in_collection_l546_54669


namespace NUMINAMATH_GPT_perpendicular_vector_l546_54675

theorem perpendicular_vector {a : ℝ × ℝ} (h : a = (1, -2)) : ∃ (b : ℝ × ℝ), b = (2, 1) ∧ (a.1 * b.1 + a.2 * b.2 = 0) :=
by 
  sorry

end NUMINAMATH_GPT_perpendicular_vector_l546_54675


namespace NUMINAMATH_GPT_basketball_court_length_difference_l546_54670

theorem basketball_court_length_difference :
  ∃ (l w : ℕ), l = 31 ∧ w = 17 ∧ l - w = 14 := by
  sorry

end NUMINAMATH_GPT_basketball_court_length_difference_l546_54670


namespace NUMINAMATH_GPT_evaluate_expression_l546_54674

noncomputable def absoluteValue (x : ℝ) : ℝ := |x|

noncomputable def ceilingFunction (x : ℝ) : ℤ := ⌈x⌉

theorem evaluate_expression : ceilingFunction (absoluteValue (-52.7)) = 53 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l546_54674


namespace NUMINAMATH_GPT_number_of_solutions_l546_54614

noncomputable def g_n (n : ℕ) (x : ℝ) := (Real.sin x)^(2 * n) + (Real.cos x)^(2 * n)

theorem number_of_solutions : ∀ (x : ℝ), x ∈ Set.Icc 0 (2 * Real.pi) -> 
  8 * g_n 3 x - 6 * g_n 2 x = 3 * g_n 1 x -> false :=
by sorry

end NUMINAMATH_GPT_number_of_solutions_l546_54614


namespace NUMINAMATH_GPT_full_size_mustang_length_l546_54692

theorem full_size_mustang_length 
  (smallest_model_length : ℕ)
  (mid_size_factor : ℕ)
  (full_size_factor : ℕ)
  (h1 : smallest_model_length = 12)
  (h2 : mid_size_factor = 2)
  (h3 : full_size_factor = 10) :
  (smallest_model_length * mid_size_factor) * full_size_factor = 240 := 
sorry

end NUMINAMATH_GPT_full_size_mustang_length_l546_54692


namespace NUMINAMATH_GPT_num_sequences_eq_15_l546_54684

noncomputable def num_possible_sequences : ℕ :=
  let angles_increasing_arith_seq := ∃ (x d : ℕ), x > 0 ∧ x + 4 * d < 140 ∧ 5 * x + 10 * d = 540 ∧ d ≠ 0
  by sorry

theorem num_sequences_eq_15 : num_possible_sequences = 15 := 
  by sorry

end NUMINAMATH_GPT_num_sequences_eq_15_l546_54684


namespace NUMINAMATH_GPT_correct_transformation_l546_54656

theorem correct_transformation (x y : ℤ) (h : x = y) : x - 2 = y - 2 :=
by
  sorry

end NUMINAMATH_GPT_correct_transformation_l546_54656


namespace NUMINAMATH_GPT_solution_set_f_l546_54650

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then 2^(x - 1) - 2 else 2^(1 - x) - 2

theorem solution_set_f (x : ℝ) : 
  (1 ≤ x ∧ x ≤ 3) ↔ (f (x - 1) ≤ 0) :=
sorry

end NUMINAMATH_GPT_solution_set_f_l546_54650


namespace NUMINAMATH_GPT_tan_sin_cos_proof_l546_54655

theorem tan_sin_cos_proof (h1 : Real.sin (Real.pi / 6) = 1 / 2)
    (h2 : Real.cos (Real.pi / 6) = Real.sqrt 3 / 2)
    (h3 : Real.tan (Real.pi / 6) = Real.sin (Real.pi / 6) / Real.cos (Real.pi / 6)) :
    ((Real.tan (Real.pi / 6))^2 - (Real.sin (Real.pi / 6))^2) / ((Real.tan (Real.pi / 6))^2 * (Real.cos (Real.pi / 6))^2) = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_tan_sin_cos_proof_l546_54655


namespace NUMINAMATH_GPT_two_colonies_reach_limit_l546_54638

noncomputable def bacteria_growth (n : ℕ) : ℕ := 2^n

theorem two_colonies_reach_limit (days : ℕ) (h : bacteria_growth days = (2^20)) : 
  bacteria_growth days = bacteria_growth 20 := 
by sorry

end NUMINAMATH_GPT_two_colonies_reach_limit_l546_54638


namespace NUMINAMATH_GPT_number_multiplies_xz_l546_54658

theorem number_multiplies_xz (x y z w A B : ℝ) (h1 : 4 * x * z + y * w = 3) (h2 : x * w + y * z = 6) :
  A * B = 4 :=
sorry

end NUMINAMATH_GPT_number_multiplies_xz_l546_54658


namespace NUMINAMATH_GPT_length_of_field_l546_54610

variable (w : ℕ)   -- Width of the rectangular field
variable (l : ℕ)   -- Length of the rectangular field
variable (pond_side : ℕ)  -- Side length of the square pond
variable (pond_area field_area : ℕ)  -- Areas of the pond and field
variable (cond1 : l = 2 * w)  -- Condition 1: Length is double the width
variable (cond2 : pond_side = 4)  -- Condition 2: Side of the pond is 4 meters
variable (cond3 : pond_area = pond_side * pond_side)  -- Condition 3: Area of square pond
variable (cond4 : pond_area = (1 / 8) * field_area)  -- Condition 4: Area of pond is 1/8 of the area of the field

theorem length_of_field :
  pond_area = pond_side * pond_side →
  pond_area = (1 / 8) * (l * w) →
  l = 2 * w →
  w = 8 →
  l = 16 :=
by
  intro h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_length_of_field_l546_54610


namespace NUMINAMATH_GPT_inequality_solution_l546_54617

theorem inequality_solution (x : ℝ) : 
  (3 / 20 + abs (2 * x - 5 / 40) < 9 / 40) → (1 / 40 < x ∧ x < 1 / 10) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l546_54617


namespace NUMINAMATH_GPT_quadratic_equation_solution_l546_54643

theorem quadratic_equation_solution : ∀ x : ℝ, x^2 - 9 = 0 ↔ (x = 3 ∨ x = -3) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_equation_solution_l546_54643


namespace NUMINAMATH_GPT_tom_watching_days_l546_54639

noncomputable def total_watch_time : ℕ :=
  30 * 22 + 28 * 25 + 27 * 29 + 20 * 31 + 25 * 27 + 20 * 35

noncomputable def daily_watch_time : ℕ := 2 * 60

theorem tom_watching_days : ⌈(total_watch_time / daily_watch_time : ℚ)⌉ = 35 := by
  sorry

end NUMINAMATH_GPT_tom_watching_days_l546_54639


namespace NUMINAMATH_GPT_convex_polygon_max_interior_angles_l546_54646

theorem convex_polygon_max_interior_angles (n : ℕ) (h1 : n ≥ 3) (h2 : n < 360) :
  ∃ x, x ≤ 4 ∧ ∀ k, k > 4 → False :=
by
  sorry

end NUMINAMATH_GPT_convex_polygon_max_interior_angles_l546_54646


namespace NUMINAMATH_GPT_trapezoid_area_l546_54647

variables (R₁ R₂ : ℝ)

theorem trapezoid_area (h_eq : h = 4 * R₁ * R₂ / (R₁ + R₂)) (mn_eq : mn = 2 * Real.sqrt (R₁ * R₂)) :
  S_ABCD = 8 * R₁ * R₂ * Real.sqrt (R₁ * R₂) / (R₁ + R₂) :=
sorry

end NUMINAMATH_GPT_trapezoid_area_l546_54647


namespace NUMINAMATH_GPT_assignment_methods_l546_54693

theorem assignment_methods : 
  let doctors := 2
  let nurses := 4
  let schools := 2
  let doctors_per_school := 1
  let nurses_per_school := 2
  (doctors * (nurses.choose nurses_per_school)) = 12 := by
  sorry

end NUMINAMATH_GPT_assignment_methods_l546_54693


namespace NUMINAMATH_GPT_intersection_point_with_y_axis_l546_54679

theorem intersection_point_with_y_axis : 
  ∃ y, (0, y) = (0, 3) ∧ (y = 0 + 3) :=
by
  sorry

end NUMINAMATH_GPT_intersection_point_with_y_axis_l546_54679


namespace NUMINAMATH_GPT_eval_expression_l546_54600

theorem eval_expression : (Int.ceil (7 / 3 : ℚ) + Int.floor (-7 / 3 : ℚ)) = 0 := by
  sorry

end NUMINAMATH_GPT_eval_expression_l546_54600


namespace NUMINAMATH_GPT_push_ups_total_l546_54602

theorem push_ups_total (d z : ℕ) (h1 : d = 51) (h2 : d = z + 49) : d + z = 53 := by
  sorry

end NUMINAMATH_GPT_push_ups_total_l546_54602


namespace NUMINAMATH_GPT_compute_expression_l546_54635

theorem compute_expression : 1010^2 - 990^2 - 1005^2 + 995^2 + 1012^2 - 988^2 = 68000 := 
by
  sorry

end NUMINAMATH_GPT_compute_expression_l546_54635


namespace NUMINAMATH_GPT_ant_travel_distance_l546_54686

theorem ant_travel_distance (r1 r2 r3 : ℝ) (h1 : r1 = 5) (h2 : r2 = 10) (h3 : r3 = 15) :
  let A_large := (1/3) * 2 * Real.pi * r3
  let D_radial := (r3 - r2) + (r2 - r1)
  let A_middle := (1/3) * 2 * Real.pi * r2
  let D_small := 2 * r1
  let A_small := (1/2) * 2 * Real.pi * r1
  A_large + D_radial + A_middle + D_small + A_small = (65 * Real.pi / 3) + 20 :=
by
  sorry

end NUMINAMATH_GPT_ant_travel_distance_l546_54686


namespace NUMINAMATH_GPT_tables_in_conference_hall_l546_54667

theorem tables_in_conference_hall (c t : ℕ) 
  (h1 : c = 8 * t) 
  (h2 : 4 * c + 4 * t = 648) : 
  t = 18 :=
by sorry

end NUMINAMATH_GPT_tables_in_conference_hall_l546_54667


namespace NUMINAMATH_GPT_triangle_area_is_six_l546_54632

-- Conditions
def line_equation (Q : ℝ) : Prop :=
  ∀ (x y : ℝ), 12 * x - 4 * y + (Q - 305) = 0

def area_of_triangle (Q R : ℝ) : Prop :=
  R = (305 - Q) ^ 2 / 96

-- Question: Given a line equation forming a specific triangle, prove the area R equals 6.
theorem triangle_area_is_six (Q : ℝ) (h1 : Q = 281 ∨ Q = 329) :
  ∃ R : ℝ, line_equation Q → area_of_triangle Q R → R = 6 :=
by {
  sorry -- Proof to be provided
}

end NUMINAMATH_GPT_triangle_area_is_six_l546_54632


namespace NUMINAMATH_GPT_base_9_units_digit_of_sum_l546_54620

def base_n_units_digit (n : ℕ) (a : ℕ) : ℕ :=
a % n

theorem base_9_units_digit_of_sum : base_n_units_digit 9 (45 + 76) = 2 :=
by
  sorry

end NUMINAMATH_GPT_base_9_units_digit_of_sum_l546_54620


namespace NUMINAMATH_GPT_point_Q_in_third_quadrant_l546_54682

-- Define point P in the fourth quadrant with coordinates a and b.
variable (a b : ℝ)
variable (h1 : a > 0)  -- Condition for the x-coordinate of P in fourth quadrant
variable (h2 : b < 0)  -- Condition for the y-coordinate of P in fourth quadrant

-- Point Q is defined by the coordinates (-a, b-1). We need to show it lies in the third quadrant.
theorem point_Q_in_third_quadrant : (-a < 0) ∧ (b - 1 < 0) :=
  by
    sorry

end NUMINAMATH_GPT_point_Q_in_third_quadrant_l546_54682


namespace NUMINAMATH_GPT_expand_expression_l546_54697

theorem expand_expression (x y : ℝ) : 
  (16 * x + 18 - 7 * y) * (3 * x) = 48 * x^2 + 54 * x - 21 * x * y :=
by
  sorry

end NUMINAMATH_GPT_expand_expression_l546_54697


namespace NUMINAMATH_GPT_solve_for_n_l546_54640

theorem solve_for_n :
  ∃ n : ℤ, -90 ≤ n ∧ n ≤ 90 ∧ Real.sin (n * Real.pi / 180) = Real.sin (750 * Real.pi / 180) ∧ n = 30 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_n_l546_54640


namespace NUMINAMATH_GPT_sin_cos_sum_l546_54695

theorem sin_cos_sum (x y r : ℝ) (h : r = Real.sqrt (x^2 + y^2)) (ha : (x = 5) ∧ (y = -12)) :
  (y / r) + (x / r) = -7 / 13 :=
by
  sorry

end NUMINAMATH_GPT_sin_cos_sum_l546_54695


namespace NUMINAMATH_GPT_JaneTotalEarningsIs138_l546_54608

structure FarmData where
  chickens : ℕ
  ducks : ℕ
  quails : ℕ
  chickenEggsPerWeek : ℕ
  duckEggsPerWeek : ℕ
  quailEggsPerWeek : ℕ
  chickenPricePerDozen : ℕ
  duckPricePerDozen : ℕ
  quailPricePerDozen : ℕ

def JaneFarmData : FarmData := {
  chickens := 10,
  ducks := 8,
  quails := 12,
  chickenEggsPerWeek := 6,
  duckEggsPerWeek := 4,
  quailEggsPerWeek := 10,
  chickenPricePerDozen := 2,
  duckPricePerDozen := 3,
  quailPricePerDozen := 4
}

def eggsLaid (f : FarmData) : ℕ × ℕ × ℕ :=
((f.chickens * f.chickenEggsPerWeek), 
 (f.ducks * f.duckEggsPerWeek), 
 (f.quails * f.quailEggsPerWeek))

def earningsForWeek1 (f : FarmData) : ℕ :=
let (chickenEggs, duckEggs, quailEggs) := eggsLaid f
let chickenDozens := chickenEggs / 12
let duckDozens := duckEggs / 12
let quailDozens := (quailEggs / 12) / 2
(chickenDozens * f.chickenPricePerDozen) + (duckDozens * f.duckPricePerDozen) + (quailDozens * f.quailPricePerDozen)

def earningsForWeek2 (f : FarmData) : ℕ :=
let (chickenEggs, duckEggs, quailEggs) := eggsLaid f
let chickenDozens := chickenEggs / 12
let duckDozens := (3 * duckEggs / 4) / 12
let quailDozens := quailEggs / 12
(chickenDozens * f.chickenPricePerDozen) + (duckDozens * f.duckPricePerDozen) + (quailDozens * f.quailPricePerDozen)

def earningsForWeek3 (f : FarmData) : ℕ :=
let (_, duckEggs, quailEggs) := eggsLaid f
let duckDozens := duckEggs / 12
let quailDozens := quailEggs / 12
(duckDozens * f.duckPricePerDozen) + (quailDozens * f.quailPricePerDozen)

def totalEarnings (f : FarmData) : ℕ :=
earningsForWeek1 f + earningsForWeek2 f + earningsForWeek3 f

theorem JaneTotalEarningsIs138 : totalEarnings JaneFarmData = 138 := by
  sorry

end NUMINAMATH_GPT_JaneTotalEarningsIs138_l546_54608


namespace NUMINAMATH_GPT_decimal_fraction_error_l546_54653

theorem decimal_fraction_error (A B C D E : ℕ) (hA : A < 100) 
    (h10B : 10 * B = A + C) (h10C : 10 * C = 6 * A + D) (h10D : 10 * D = 7 * A + E) 
    (hBCDE_lt_A : B < A ∧ C < A ∧ D < A ∧ E < A) : 
    false :=
sorry

end NUMINAMATH_GPT_decimal_fraction_error_l546_54653


namespace NUMINAMATH_GPT_arithmetic_sequence_ratio_a10_b10_l546_54688

variable {a : ℕ → ℕ} {b : ℕ → ℕ}
variable {S T : ℕ → ℕ}

-- We assume S_n and T_n are the sums of the first n terms of sequences a and b respectively.
-- We also assume the provided ratio condition between S_n and T_n.
axiom sum_of_first_n_terms_a (n : ℕ) : S n = (n * (2 * a 1 + (n - 1) * (a 2 - a 1))) / 2
axiom sum_of_first_n_terms_b (n : ℕ) : T n = (n * (2 * b 1 + (n - 1) * (b 2 - b 1))) / 2
axiom ratio_condition (n : ℕ) : (S n) / (T n) = (3 * n - 1) / (2 * n + 3)

theorem arithmetic_sequence_ratio_a10_b10 : (a 10) / (b 10) = 56 / 41 :=
by sorry

end NUMINAMATH_GPT_arithmetic_sequence_ratio_a10_b10_l546_54688


namespace NUMINAMATH_GPT_enclosed_area_abs_eq_54_l546_54613

theorem enclosed_area_abs_eq_54 :
  (∃ (x y : ℝ), abs x + abs (3 * y) = 9) → True := 
by
  sorry

end NUMINAMATH_GPT_enclosed_area_abs_eq_54_l546_54613


namespace NUMINAMATH_GPT_intersection_A_B_l546_54623

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x | ∃ m : ℝ, x = m^2 + 1}

theorem intersection_A_B : A ∩ B = {1} := by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l546_54623


namespace NUMINAMATH_GPT_simplify_expression_to_polynomial_l546_54698

theorem simplify_expression_to_polynomial :
    (3 * x^2 + 4 * x + 8) * (2 * x + 1) - 
    (2 * x + 1) * (x^2 + 5 * x - 72) + 
    (4 * x - 15) * (2 * x + 1) * (x + 6) = 
    12 * x^3 + 22 * x^2 - 12 * x - 10 :=
by
    sorry

end NUMINAMATH_GPT_simplify_expression_to_polynomial_l546_54698


namespace NUMINAMATH_GPT_mean_eq_median_of_set_l546_54662

theorem mean_eq_median_of_set (x : ℕ) (hx : 0 < x) :
  let s := [1, 2, 4, 5, x]
  let mean := (1 + 2 + 4 + 5 + x) / 5
  let median := if x ≤ 2 then 2 else if x ≤ 4 then x else 4
  mean = median → (x = 3 ∨ x = 8) :=
by {
  sorry
}

end NUMINAMATH_GPT_mean_eq_median_of_set_l546_54662


namespace NUMINAMATH_GPT_geometric_sequence_problem_l546_54615

theorem geometric_sequence_problem (a : ℕ → ℝ) (r : ℝ) 
  (h_geo : ∀ n, a (n + 1) = r * a n) 
  (h_cond: a 4 + a 6 = 8) : 
  a 1 * a 7 + 2 * a 3 * a 7 + a 3 * a 9 = 64 :=
  sorry

end NUMINAMATH_GPT_geometric_sequence_problem_l546_54615


namespace NUMINAMATH_GPT_distance_travelled_l546_54671

theorem distance_travelled (t : ℝ) (h : 15 * t = 10 * t + 20) : 10 * t = 40 :=
by
  have ht : t = 4 := by linarith
  rw [ht]
  norm_num

end NUMINAMATH_GPT_distance_travelled_l546_54671


namespace NUMINAMATH_GPT_bus_passengers_l546_54607

variable (P : ℕ) -- P represents the initial number of passengers

theorem bus_passengers (h1 : P + 16 - 17 = 49) : P = 50 :=
by
  sorry

end NUMINAMATH_GPT_bus_passengers_l546_54607


namespace NUMINAMATH_GPT_M_eq_N_l546_54649

def M : Set ℤ := {u | ∃ m n l : ℤ, u = 12 * m + 8 * n + 4 * l}
def N : Set ℤ := {u | ∃ p q r : ℤ, u = 20 * p + 16 * q + 12 * r}

theorem M_eq_N : M = N :=
by
  sorry

end NUMINAMATH_GPT_M_eq_N_l546_54649


namespace NUMINAMATH_GPT_litter_patrol_total_pieces_l546_54633

theorem litter_patrol_total_pieces :
  let glass_bottles := 25
  let aluminum_cans := 18
  let plastic_bags := 12
  let paper_cups := 7
  let cigarette_packs := 5
  let discarded_face_masks := 3
  glass_bottles + aluminum_cans + plastic_bags + paper_cups + cigarette_packs + discarded_face_masks = 70 :=
by
  sorry

end NUMINAMATH_GPT_litter_patrol_total_pieces_l546_54633


namespace NUMINAMATH_GPT_y_coordinate_midpoint_l546_54618

theorem y_coordinate_midpoint : 
  let L : (ℝ → ℝ) := λ x => x - 1
  let P : (ℝ → ℝ) := λ y => 8 * (y^2)
  ∃ (x₁ x₂ y₁ y₂ : ℝ),
    P (L x₁) = y₁ ∧ P (L x₂) = y₂ ∧ 
    L x₁ = y₁ ∧ L x₂ = y₂ ∧ 
    x₁ + x₂ = 10 ∧ y₁ + y₂ = 8 ∧
    (y₁ + y₂) / 2 = 4 := sorry

end NUMINAMATH_GPT_y_coordinate_midpoint_l546_54618


namespace NUMINAMATH_GPT_movie_box_office_revenue_l546_54624

variable (x : ℝ)

theorem movie_box_office_revenue (h : 300 + 300 * (1 + x) + 300 * (1 + x)^2 = 1000) :
  3 + 3 * (1 + x) + 3 * (1 + x)^2 = 10 :=
by
  sorry

end NUMINAMATH_GPT_movie_box_office_revenue_l546_54624


namespace NUMINAMATH_GPT_solution_set_ineq_l546_54690

theorem solution_set_ineq (x : ℝ) : (x - 2) / (x - 5) ≥ 3 ↔ 5 < x ∧ x ≤ 13 / 2 :=
sorry

end NUMINAMATH_GPT_solution_set_ineq_l546_54690


namespace NUMINAMATH_GPT_fraction_simplification_l546_54663

def numerator : Int := 5^4 + 5^2 + 5
def denominator : Int := 5^3 - 2 * 5

theorem fraction_simplification :
  (numerator : ℚ) / (denominator : ℚ) = 27 + (14 / 23) := by
  sorry

end NUMINAMATH_GPT_fraction_simplification_l546_54663


namespace NUMINAMATH_GPT_product_of_two_numbers_l546_54664

theorem product_of_two_numbers (a b : ℝ) 
  (h1 : a - b = 2 * k)
  (h2 : a + b = 8 * k)
  (h3 : 2 * a * b = 30 * k) : a * b = 15 :=
by
  sorry

end NUMINAMATH_GPT_product_of_two_numbers_l546_54664


namespace NUMINAMATH_GPT_remainder_mod_68_l546_54659

theorem remainder_mod_68 (n : ℕ) (h : 67^67 + 67 ≡ 66 [MOD n]) : n = 68 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_mod_68_l546_54659


namespace NUMINAMATH_GPT_monotonically_decreasing_iff_a_lt_1_l546_54603

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + (1/2) * a * x^2 - 2 * x

theorem monotonically_decreasing_iff_a_lt_1 {a : ℝ} (h : ∀ x > 0, (deriv (f a) x) < 0) : a < 1 :=
sorry

end NUMINAMATH_GPT_monotonically_decreasing_iff_a_lt_1_l546_54603


namespace NUMINAMATH_GPT_ineq_a3b3c3_l546_54651

theorem ineq_a3b3c3 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a^3 + b^3 + c^3 ≥ a^2 * b + b^2 * c + c^2 * a ∧ (a^3 + b^3 + c^3 = a^2 * b + b^2 * c + c^2 * a ↔ a = b ∧ b = c) :=
by
  sorry

end NUMINAMATH_GPT_ineq_a3b3c3_l546_54651


namespace NUMINAMATH_GPT_jerome_contacts_total_l546_54687

def jerome_classmates : Nat := 20
def jerome_out_of_school_friends : Nat := jerome_classmates / 2
def jerome_family_members : Nat := 2 + 1
def jerome_total_contacts : Nat := jerome_classmates + jerome_out_of_school_friends + jerome_family_members

theorem jerome_contacts_total : jerome_total_contacts = 33 := by
  sorry

end NUMINAMATH_GPT_jerome_contacts_total_l546_54687


namespace NUMINAMATH_GPT_students_not_playing_games_l546_54622

theorem students_not_playing_games 
  (total_students : ℕ)
  (basketball_players : ℕ)
  (volleyball_players : ℕ)
  (both_players : ℕ)
  (h1 : total_students = 20)
  (h2 : basketball_players = (1 / 2) * total_students)
  (h3 : volleyball_players = (2 / 5) * total_students)
  (h4 : both_players = (1 / 10) * total_students) :
  total_students - ((basketball_players + volleyball_players) - both_players) = 4 :=
by
  sorry

end NUMINAMATH_GPT_students_not_playing_games_l546_54622


namespace NUMINAMATH_GPT_correlated_relationships_l546_54699

-- Definitions for the conditions are arbitrary
-- In actual use cases, these would be replaced with real mathematical conditions
def great_teachers_produce_outstanding_students : Prop := sorry
def volume_of_sphere_with_radius : Prop := sorry
def apple_production_climate : Prop := sorry
def height_and_weight : Prop := sorry
def taxi_fare_distance_traveled : Prop := sorry
def crows_cawing_bad_omen : Prop := sorry

-- The final theorem statement
theorem correlated_relationships : 
  great_teachers_produce_outstanding_students ∧
  apple_production_climate ∧
  height_and_weight ∧
  ¬ volume_of_sphere_with_radius ∧ 
  ¬ taxi_fare_distance_traveled ∧ 
  ¬ crows_cawing_bad_omen :=
sorry

end NUMINAMATH_GPT_correlated_relationships_l546_54699


namespace NUMINAMATH_GPT_max_photo_area_correct_l546_54672

def frame_area : ℝ := 59.6
def num_photos : ℕ := 4
def max_photo_area : ℝ := 14.9

theorem max_photo_area_correct : frame_area / num_photos = max_photo_area :=
by sorry

end NUMINAMATH_GPT_max_photo_area_correct_l546_54672


namespace NUMINAMATH_GPT_rectangle_area_l546_54609

theorem rectangle_area (b : ℕ) (l : ℕ) (h1 : l = 3 * b) (h2 : 2 * (l + b) = 48) : l * b = 108 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l546_54609


namespace NUMINAMATH_GPT_molecular_weight_of_7_moles_boric_acid_l546_54680

-- Define the given constants.
def atomic_weight_H : ℝ := 1.008
def atomic_weight_B : ℝ := 10.81
def atomic_weight_O : ℝ := 16.00

-- Define the molecular formula for boric acid.
def molecular_weight_H3BO3 : ℝ :=
  3 * atomic_weight_H + 1 * atomic_weight_B + 3 * atomic_weight_O

-- Define the number of moles.
def moles_boric_acid : ℝ := 7

-- Calculate the total weight for 7 moles of boric acid.
def total_weight_boric_acid : ℝ :=
  moles_boric_acid * molecular_weight_H3BO3

-- The target statement to prove.
theorem molecular_weight_of_7_moles_boric_acid :
  total_weight_boric_acid = 432.838 := by
  sorry

end NUMINAMATH_GPT_molecular_weight_of_7_moles_boric_acid_l546_54680


namespace NUMINAMATH_GPT_longer_diagonal_is_116_l546_54637

-- Given conditions
def side_length : ℕ := 65
def short_diagonal : ℕ := 60

-- Prove that the length of the longer diagonal in the rhombus is 116 units.
theorem longer_diagonal_is_116 : 
  let s := side_length
  let d1 := short_diagonal / 2
  let d2 := (s^2 - d1^2).sqrt
  (2 * d2) = 116 :=
by
  sorry

end NUMINAMATH_GPT_longer_diagonal_is_116_l546_54637


namespace NUMINAMATH_GPT_recruits_total_l546_54628

theorem recruits_total (P N D : ℕ) (total_recruits : ℕ) 
  (h1 : P = 50) 
  (h2 : N = 100) 
  (h3 : D = 170)
  (h4 : (∃ x y, (x = 50) ∧ (y = 100) ∧ (x = 4 * y))
        ∨ (∃ x z, (x = 50) ∧ (z = 170) ∧ (x = 4 * z))
        ∨ (∃ y z, (y = 100) ∧ (z = 170) ∧ (y = 4 * z))) : 
  total_recruits = 211 :=
by
  sorry

end NUMINAMATH_GPT_recruits_total_l546_54628


namespace NUMINAMATH_GPT_sum_of_interior_angles_of_hexagon_l546_54696

theorem sum_of_interior_angles_of_hexagon
  (n : ℕ)
  (h : n = 6) :
  (n - 2) * 180 = 720 := by
  sorry

end NUMINAMATH_GPT_sum_of_interior_angles_of_hexagon_l546_54696


namespace NUMINAMATH_GPT_find_b_plus_c_l546_54612

variable {a b c d : ℝ}

theorem find_b_plus_c
  (h1 : a + b = 4)
  (h2 : c + d = 3)
  (h3 : a + d = 2) :
  b + c = 5 := 
  by
  sorry

end NUMINAMATH_GPT_find_b_plus_c_l546_54612


namespace NUMINAMATH_GPT_period_of_sin3x_plus_cos3x_l546_54621

noncomputable def period_of_trig_sum (x : ℝ) : ℝ := 
  let y := (fun x => Real.sin (3 * x) + Real.cos (3 * x))
  (2 * Real.pi) / 3

theorem period_of_sin3x_plus_cos3x : (fun x => Real.sin (3 * x) + Real.cos (3 * x)) =
  (fun x => Real.sin (3 * (x + period_of_trig_sum x)) + Real.cos (3 * (x + period_of_trig_sum x))) :=
by
  sorry

end NUMINAMATH_GPT_period_of_sin3x_plus_cos3x_l546_54621


namespace NUMINAMATH_GPT_proof_aim_l546_54673

variables (a : ℝ)

def p : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q : Prop := ∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + (2 - a) = 0

theorem proof_aim (hp : p a) (hq : q a) : a ≤ -2 ∨ a = 1 :=
sorry

end NUMINAMATH_GPT_proof_aim_l546_54673


namespace NUMINAMATH_GPT_double_luckiness_l546_54668

variable (oats marshmallows : ℕ)
variable (initial_luckiness doubled_luckiness : ℚ)

def luckiness (marshmallows total_pieces : ℕ) : ℚ :=
  marshmallows / total_pieces

theorem double_luckiness (h_oats : oats = 90) (h_marshmallows : marshmallows = 9)
  (h_initial : initial_luckiness = luckiness marshmallows (oats + marshmallows))
  (h_doubled : doubled_luckiness = 2 * initial_luckiness) :
  ∃ x : ℕ, doubled_luckiness = luckiness (marshmallows + x) (oats + marshmallows + x) :=
  sorry

#check double_luckiness

end NUMINAMATH_GPT_double_luckiness_l546_54668


namespace NUMINAMATH_GPT_sufficient_condition_for_A_l546_54657

variables {A B C : Prop}

theorem sufficient_condition_for_A (h1 : A ↔ B) (h2 : C → B) : C → A :=
sorry

end NUMINAMATH_GPT_sufficient_condition_for_A_l546_54657


namespace NUMINAMATH_GPT_first_applicant_earnings_l546_54642

def first_applicant_salary : ℕ := 42000
def first_applicant_training_cost_per_month : ℕ := 1200
def first_applicant_training_months : ℕ := 3
def second_applicant_salary : ℕ := 45000
def second_applicant_bonus_percentage : ℕ := 1
def company_earnings_from_second_applicant : ℕ := 92000
def earnings_difference : ℕ := 850

theorem first_applicant_earnings 
  (salary1 : first_applicant_salary = 42000)
  (train_cost_per_month : first_applicant_training_cost_per_month = 1200)
  (train_months : first_applicant_training_months = 3)
  (salary2 : second_applicant_salary = 45000)
  (bonus_percentage : second_applicant_bonus_percentage = 1)
  (earnings2 : company_earnings_from_second_applicant = 92000)
  (earning_diff : earnings_difference = 850) :
  (company_earnings_from_second_applicant - (second_applicant_salary + (second_applicant_salary * second_applicant_bonus_percentage / 100)) - earnings_difference) = 45700 := 
by 
  sorry

end NUMINAMATH_GPT_first_applicant_earnings_l546_54642


namespace NUMINAMATH_GPT_find_a_l546_54683

theorem find_a (a : ℝ) : 
  (∃ (a : ℝ), a * 15 + 6 = -9) → a = -1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_a_l546_54683


namespace NUMINAMATH_GPT_least_number_l546_54604

theorem least_number (n : ℕ) (h1 : n % 31 = 3) (h2 : n % 9 = 3) : n = 282 :=
sorry

end NUMINAMATH_GPT_least_number_l546_54604


namespace NUMINAMATH_GPT_solution_interval_l546_54625

-- Define the differentiable function f over the interval (-∞, 0)
variable {f : ℝ → ℝ}
variable (hf : ∀ x < 0, HasDerivAt f (f' x) x)
variable (hx_cond : ∀ x < 0, 2 * f x + x * (deriv f x) > x^2)

-- Proof statement to show the solution interval
theorem solution_interval :
  {x : ℝ | (x + 2018)^2 * f (x + 2018) - 4 * f (-2) > 0} = {x | x < -2020} :=
sorry

end NUMINAMATH_GPT_solution_interval_l546_54625


namespace NUMINAMATH_GPT_cardinality_union_l546_54641

open Finset

theorem cardinality_union (A B : Finset ℕ) (h : 2 ^ A.card + 2 ^ B.card - 2 ^ (A ∩ B).card = 144) : (A ∪ B).card = 8 := 
by 
  sorry

end NUMINAMATH_GPT_cardinality_union_l546_54641


namespace NUMINAMATH_GPT_min_y_in_quadratic_l546_54616

theorem min_y_in_quadratic (x : ℝ) : ∃ y : ℝ, (y = x^2 + 16 * x + 20) ∧ ∀ y', (y' = x^2 + 16 * x + 20) → y ≤ y' := 
sorry

end NUMINAMATH_GPT_min_y_in_quadratic_l546_54616


namespace NUMINAMATH_GPT_intersection_points_count_l546_54661

theorem intersection_points_count (A : ℝ) (hA : A > 0) :
  ((A > 1 / 4) → ∃! (x y : ℝ), (y = A * x^2) ∧ (x^2 + y^2 = 4 * y) ∧
                              (x ≠ 0 ∨ y ≠ 0)) ∧
  ((A ≤ 1 / 4) → ∃! (x y : ℝ), (y = A * x^2) ∧ (x^2 + y^2 = 4 * y)) :=
by
  sorry

end NUMINAMATH_GPT_intersection_points_count_l546_54661


namespace NUMINAMATH_GPT_increasing_f_iff_m_ge_two_inequality_when_m_equals_three_l546_54694

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x^2 - 4 * x + m * Real.log x

-- Part (1): Prove m >= 2 is the range for which f(x) is increasing
theorem increasing_f_iff_m_ge_two (m : ℝ) : (∀ x > 0, (2 * x - 4 + m / x) ≥ 0) ↔ m ≥ 2 := sorry

-- Part (2): Prove the given inequality for m = 3
theorem inequality_when_m_equals_three (x : ℝ) (h : x > 0) : (1 / 9) * x ^ 3 - (f x 3) > 2 := sorry

end NUMINAMATH_GPT_increasing_f_iff_m_ge_two_inequality_when_m_equals_three_l546_54694


namespace NUMINAMATH_GPT_range_of_a_l546_54619

namespace ProofProblem

theorem range_of_a (a : ℝ) (H1 : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → ∃ y : ℝ, y = a * x + 2 * a + 1 ∧ y > 0 ∧ y < 0) : 
  -1 < a ∧ a < -1/3 := 
sorry

end ProofProblem

end NUMINAMATH_GPT_range_of_a_l546_54619


namespace NUMINAMATH_GPT_num_boys_in_circle_l546_54634

theorem num_boys_in_circle (n : ℕ) 
  (h : ∃ k, n = 2 * k ∧ k = 40 - 10) : n = 60 :=
by
  sorry

end NUMINAMATH_GPT_num_boys_in_circle_l546_54634


namespace NUMINAMATH_GPT_complement_problem_l546_54689

open Set

variable (U A : Set ℕ)

def complement (U A : Set ℕ) : Set ℕ := U \ A

theorem complement_problem
  (hU : U = {1, 2, 3, 4, 5})
  (hA : A = {1, 3}) :
  complement U A = {2, 4, 5} :=
by
  rw [complement, hU, hA]
  sorry

end NUMINAMATH_GPT_complement_problem_l546_54689


namespace NUMINAMATH_GPT_comp_figure_perimeter_l546_54630

-- Given conditions
def side_length_square : ℕ := 2
def side_length_triangle : ℕ := 1
def number_of_squares : ℕ := 4
def number_of_triangles : ℕ := 3

-- Define the perimeter calculation
def perimeter_of_figure : ℕ :=
  let perimeter_squares := (2 * (number_of_squares - 2) + 2 * 2 + 2 * 1) * side_length_square
  let perimeter_triangles := number_of_triangles * side_length_triangle
  perimeter_squares + perimeter_triangles

-- Target theorem
theorem comp_figure_perimeter : perimeter_of_figure = 17 := by
  sorry

end NUMINAMATH_GPT_comp_figure_perimeter_l546_54630


namespace NUMINAMATH_GPT_not_possible_to_get_105_single_stone_piles_l546_54627

noncomputable def piles : List Nat := [51, 49, 5]
def combine (a b : Nat) : Nat := a + b
def split (a : Nat) : List Nat := if a % 2 = 0 then [a / 2, a / 2] else [a]

theorem not_possible_to_get_105_single_stone_piles 
  (initial_piles : List Nat := piles) 
  (combine : Nat → Nat → Nat := combine) 
  (split : Nat → List Nat := split) :
  ¬ ∃ (final_piles : List Nat), final_piles.length = 105 ∧ (∀ n ∈ final_piles, n = 1) :=
by
  sorry

end NUMINAMATH_GPT_not_possible_to_get_105_single_stone_piles_l546_54627


namespace NUMINAMATH_GPT_fraction_difference_l546_54685

theorem fraction_difference : 7 / 12 - 3 / 8 = 5 / 24 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_difference_l546_54685


namespace NUMINAMATH_GPT_inequality_solution_l546_54645

theorem inequality_solution (x y : ℝ) (h1 : y ≥ x^2 + 1) :
    2^y - 2 * Real.cos x + Real.sqrt (y - x^2 - 1) ≤ 0 ↔ x = 0 ∧ y = 1 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l546_54645


namespace NUMINAMATH_GPT_cuberoot_inequality_l546_54606

theorem cuberoot_inequality (a b : ℝ) : a < b → (∃ x y : ℝ, x^3 = a ∧ y^3 = b ∧ (x = y ∨ x > y)) := 
sorry

end NUMINAMATH_GPT_cuberoot_inequality_l546_54606


namespace NUMINAMATH_GPT_volume_of_pyramid_l546_54691

noncomputable def greatest_pyramid_volume (AB AC sin_α : ℝ) (max_angle : ℝ) : ℝ :=
  if AB = 3 ∧ AC = 5 ∧ sin_α = 4 / 5 ∧ max_angle ≤ 60 then
    5 * Real.sqrt 39 / 2
  else
    0

theorem volume_of_pyramid :
  greatest_pyramid_volume 3 5 (4 / 5) 60 = 5 * Real.sqrt 39 / 2 := by
  sorry -- Proof omitted as per instruction

end NUMINAMATH_GPT_volume_of_pyramid_l546_54691


namespace NUMINAMATH_GPT_bob_cleaning_time_l546_54676

-- Define the conditions
def timeAlice : ℕ := 30
def fractionBob : ℚ := 1 / 3

-- Define the proof problem
theorem bob_cleaning_time : (fractionBob * timeAlice : ℚ) = 10 := by
  sorry

end NUMINAMATH_GPT_bob_cleaning_time_l546_54676


namespace NUMINAMATH_GPT_correct_average_is_40_point_3_l546_54678

noncomputable def incorrect_average : ℝ := 40.2
noncomputable def incorrect_total_sum : ℝ := incorrect_average * 10
noncomputable def incorrect_first_number_adjustment : ℝ := 17
noncomputable def incorrect_second_number_actual : ℝ := 31
noncomputable def incorrect_second_number_provided : ℝ := 13
noncomputable def correct_total_sum : ℝ := incorrect_total_sum - incorrect_first_number_adjustment + (incorrect_second_number_actual - incorrect_second_number_provided)
noncomputable def number_of_values : ℝ := 10

theorem correct_average_is_40_point_3 :
  correct_total_sum / number_of_values = 40.3 :=
by
  sorry

end NUMINAMATH_GPT_correct_average_is_40_point_3_l546_54678


namespace NUMINAMATH_GPT_parallelogram_height_same_area_l546_54611

noncomputable def rectangle_area (length width : ℕ) : ℕ := length * width

theorem parallelogram_height_same_area (length width base height : ℕ) 
  (h₁ : rectangle_area length width = base * height) 
  (h₂ : length = 12) 
  (h₃ : width = 6) 
  (h₄ : base = 12) : 
  height = 6 := 
sorry

end NUMINAMATH_GPT_parallelogram_height_same_area_l546_54611


namespace NUMINAMATH_GPT_range_of_m_l546_54648

variable (m : ℝ)

def p : Prop := ∀ x : ℝ, 2 * x > m * (x^2 + 1)
def q : Prop := ∃ x0 : ℝ, x0^2 + 2 * x0 - m - 1 = 0

theorem range_of_m (hp : p m) (hq : q m) : -2 ≤ m ∧ m < -1 :=
sorry

end NUMINAMATH_GPT_range_of_m_l546_54648


namespace NUMINAMATH_GPT_number_of_whole_numbers_with_cube_roots_less_than_8_l546_54681

theorem number_of_whole_numbers_with_cube_roots_less_than_8 :
  ∃ (n : ℕ), (∀ (x : ℕ), (1 ≤ x ∧ x < 512) → x ≤ n) ∧ n = 511 := 
sorry

end NUMINAMATH_GPT_number_of_whole_numbers_with_cube_roots_less_than_8_l546_54681


namespace NUMINAMATH_GPT_distance_from_reflected_point_l546_54636

theorem distance_from_reflected_point
  (P : ℝ × ℝ) (P' : ℝ × ℝ)
  (hP : P = (3, 2))
  (hP' : P' = (3, -2))
  : dist P P' = 4 := sorry

end NUMINAMATH_GPT_distance_from_reflected_point_l546_54636


namespace NUMINAMATH_GPT_complement_of_A_in_U_l546_54605

-- Define the universal set U and set A
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {3, 4, 5}

-- Define the complement of A in U
theorem complement_of_A_in_U : (U \ A) = {1, 2} :=
by
  sorry

end NUMINAMATH_GPT_complement_of_A_in_U_l546_54605


namespace NUMINAMATH_GPT_zongzi_packing_l546_54660

theorem zongzi_packing (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  (8 * x + 10 * y = 200) ↔ (x, y) = (5, 16) ∨ (x, y) = (10, 12) ∨ (x, y) = (15, 8) ∨ (x, y) = (20, 4) := 
sorry

end NUMINAMATH_GPT_zongzi_packing_l546_54660


namespace NUMINAMATH_GPT_garden_roller_length_l546_54652

theorem garden_roller_length
  (diameter : ℝ)
  (total_area : ℝ)
  (revolutions : ℕ)
  (pi : ℝ)
  (circumference : ℝ)
  (area_per_revolution : ℝ)
  (length : ℝ)
  (h1 : diameter = 1.4)
  (h2 : total_area = 44)
  (h3 : revolutions = 5)
  (h4 : pi = (22 / 7))
  (h5 : circumference = pi * diameter)
  (h6 : area_per_revolution = total_area / (revolutions : ℝ))
  (h7 : area_per_revolution = circumference * length) :
  length = 7 := by
  sorry

end NUMINAMATH_GPT_garden_roller_length_l546_54652


namespace NUMINAMATH_GPT_total_amount_paid_l546_54666

-- Definitions based on the conditions in part (a)
def cost_of_manicure : ℝ := 30
def tip_percentage : ℝ := 0.30

-- Proof statement based on conditions and answer in part (c)
theorem total_amount_paid : cost_of_manicure + (cost_of_manicure * tip_percentage) = 39 := by
  sorry

end NUMINAMATH_GPT_total_amount_paid_l546_54666


namespace NUMINAMATH_GPT_ladybugs_with_spots_l546_54677

theorem ladybugs_with_spots (total_ladybugs without_spots with_spots : ℕ) 
  (h1 : total_ladybugs = 67082) 
  (h2 : without_spots = 54912) 
  (h3 : with_spots = total_ladybugs - without_spots) : 
  with_spots = 12170 := 
by 
  -- hole for the proof 
  sorry

end NUMINAMATH_GPT_ladybugs_with_spots_l546_54677


namespace NUMINAMATH_GPT_bushes_needed_for_octagon_perimeter_l546_54629

theorem bushes_needed_for_octagon_perimeter
  (side_length : ℝ) (spacing : ℝ)
  (octagonal : ∀ (s : ℝ), s = 8 → 8 * s = 64)
  (spacing_condition : ∀ (p : ℝ), p = 64 → p / spacing = 32) :
  spacing = 2 → side_length = 8 → (64 / 2 = 32) := 
by
  sorry

end NUMINAMATH_GPT_bushes_needed_for_octagon_perimeter_l546_54629


namespace NUMINAMATH_GPT_ferry_time_difference_l546_54601

theorem ferry_time_difference :
  ∃ (t : ℕ), (∀ (dP : ℕ) (sP : ℕ) (sQ : ℕ), dP = sP * 3 →
   dP = 24 →
   sP = 8 →
   sQ = sP + 1 →
   t = (dP * 3) / sQ - 3) ∧ t = 5 := 
  sorry

end NUMINAMATH_GPT_ferry_time_difference_l546_54601


namespace NUMINAMATH_GPT_chocolate_ratio_l546_54631

theorem chocolate_ratio (N A : ℕ) (h1 : N = 10) (h2 : A - 5 = N + 15) : A / N = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_chocolate_ratio_l546_54631


namespace NUMINAMATH_GPT_mike_games_l546_54626

theorem mike_games (init_money spent_money game_cost : ℕ) (h1 : init_money = 42) (h2 : spent_money = 10) (h3 : game_cost = 8) :
  (init_money - spent_money) / game_cost = 4 :=
by
  sorry

end NUMINAMATH_GPT_mike_games_l546_54626


namespace NUMINAMATH_GPT_distance_between_P_and_Q_l546_54644

theorem distance_between_P_and_Q : 
  let initial_speed := 40  -- Speed in kmph
  let increment := 20      -- Speed increment in kmph after every 12 minutes
  let segment_duration := 12 / 60 -- Duration of each segment in hours (12 minutes in hours)
  let total_duration := 48 / 60    -- Total duration in hours (48 minutes in hours)
  let total_segments := total_duration / segment_duration -- Number of segments
  (total_segments = 4) ∧ 
  (∀ n : ℕ, n ≥ 0 → n < total_segments → 
    let speed := initial_speed + n * increment
    let distance := speed * segment_duration
    distance = speed * (12 / 60)) 
  → (40 * (12 / 60) + 60 * (12 / 60) + 80 * (12 / 60) + 100 * (12 / 60)) = 56 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_P_and_Q_l546_54644


namespace NUMINAMATH_GPT_share_per_person_is_135k_l546_54665

noncomputable def calculate_share : ℝ :=
  (0.90 * (500000 * 1.20)) / 4

theorem share_per_person_is_135k : calculate_share = 135000 :=
by
  sorry

end NUMINAMATH_GPT_share_per_person_is_135k_l546_54665
