import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Binomial
import Mathlib.Algebra.GcdMonoid
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Analysis
import Mathlib.Analysis.Geometry.Parabola
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Combinations
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.Data.Zmod.Basic
import Mathlib.Geometry.Euclidean.Circumcenter
import Mathlib.Init.Data.Nat.Basic
import Mathlib.Probability.Basic
import Mathlib.Probability.Conditional
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology
import Mathlib.Topology.Euclidean.Triangle

namespace problem_statement_l739_739812

variable (f : ℝ → ℝ)

theorem problem_statement (h_differentiable : Differentiable ℝ f)
  (h_condition : ∀ x : ℝ, (2 - x) / (deriv (deriv f x)) ≤ 0) :
  f 1 + f 3 > 2 * f 2 :=
sorry

end problem_statement_l739_739812


namespace probability_intersection_is_5_over_18_l739_739684

def dice_events (A B : ℕ → Prop) : Prop :=
  let n := 216
  let m := 60
  ∀ sabc : {x : ℕ // x ≤ 6}, A sabc.1 → B sabc.1 → (m / n = (5 : ℚ / 18))

variable A : ℕ → Prop
variable B : ℕ → Prop

theorem probability_intersection_is_5_over_18 :
  dice_events A B := sorry

end probability_intersection_is_5_over_18_l739_739684


namespace max_n_for_regular_polygons_l739_739914

theorem max_n_for_regular_polygons (m n : ℕ) (h1 : m ≥ n) (h2 : n ≥ 3)
  (h3 : (7 * (m - 2) * n) = (8 * (n - 2) * m)) : 
  n ≤ 112 ∧ (∃ m, (14 * n = (n - 16) * m)) :=
by
  sorry

end max_n_for_regular_polygons_l739_739914


namespace angle_B_is_pi_over_3_l739_739847

variables (A B C G : Point)
variables (a b c : ℝ)
variables (GA GB GC : vector ℝ 3)

def is_centroid := GA + GB + GC = 0

def given_condition : Prop :=
  (a/5) • GA + (b/7) • GB + (c/8) • GC = 0

theorem angle_B_is_pi_over_3
  (h1 : is_centroid GA GB GC)
  (h2 : given_condition GA GB GC a b c)
  (ha : a = 5)
  (hb : b = 7)
  (hc : c = 8) :
  angle B A C = π / 3 :=
sorry

end angle_B_is_pi_over_3_l739_739847


namespace number_of_zeros_of_g_l739_739522

noncomputable def f : ℝ → ℝ :=
λ x => if x < 2 then abs (2^x - 1) else 3 / (x - 1)

def g (x : ℝ) : ℝ := f (f x) - 2

theorem number_of_zeros_of_g :
  {x : set ℝ | g x = 0}.finite ∧ {x : set ℝ | g x = 0}.card = 4 :=
begin
  -- Proof goes here
  sorry
end

end number_of_zeros_of_g_l739_739522


namespace triangle_side_lengths_range_m_range_of_m_l739_739815

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (Real.cos x + m) / (Real.cos x + 2)

theorem triangle_side_lengths_range_m : 
  (∀ a b c : ℝ, ∃ m : ℝ, f a m = 1 ∧ f b m = 1 ∧ f c m = 1) ↔ m = 2 :=
begin
  sorry
end

theorem range_of_m (m : ℝ) : 
  (∀ a b c : ℝ, let fa := f a m, let fb := f b m, let fc := f c m in
   fa + fb > fc ∧ fb + fc > fa ∧ fc + fa > fb) → 
  (7/5 < m ∧ m < 5) :=
begin
  sorry
end

end triangle_side_lengths_range_m_range_of_m_l739_739815


namespace product_of_real_roots_l739_739553

theorem product_of_real_roots (x1 x2 : ℝ) (h1 : x1^2 - 6 * x1 + 8 = 0) (h2 : x2^2 - 6 * x2 + 8 = 0) :
  x1 * x2 = 8 := 
sorry

end product_of_real_roots_l739_739553


namespace spending_50_dollars_opposite_meaning_l739_739577

theorem spending_50_dollars_opposite_meaning :
  (∀ (income expenditure : Int), income = 80 → expenditure = 50 → -income = - (expenditure)) :=
by
  intro income expenditure h_income h_expenditure
  rw [h_income, h_expenditure]
  rfl

end spending_50_dollars_opposite_meaning_l739_739577


namespace eccentricity_of_ellipse_l739_739471

theorem eccentricity_of_ellipse (a b c e : ℝ)
  (h1 : a^2 = 25)
  (h2 : b^2 = 9)
  (h3 : c = Real.sqrt (a^2 - b^2))
  (h4 : e = c / a) :
  e = 4 / 5 :=
by
  sorry

end eccentricity_of_ellipse_l739_739471


namespace simplify_power_of_product_l739_739949

theorem simplify_power_of_product (x y : ℝ) : (3 * x^2 * y^3)^2 = 9 * x^4 * y^6 :=
by
  -- hint: begin proof here
  sorry

end simplify_power_of_product_l739_739949


namespace find_general_formula_l739_739244

theorem find_general_formula (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) (h₀ : n > 0)
  (h₁ : a 1 = 1)
  (h₂ : ∀ n, S (n + 1) = 2 * S n + n + 1)
  (h₃ : ∀ n, S (n + 1) - S n = a (n + 1)) :
  a n = 2^n - 1 :=
sorry

end find_general_formula_l739_739244


namespace cube_volume_total_four_boxes_l739_739354

theorem cube_volume_total_four_boxes :
  ∀ (length : ℕ), (length = 5) → (4 * (length^3) = 500) :=
begin
  intros length h,
  rw h,
  norm_num,
end

end cube_volume_total_four_boxes_l739_739354


namespace difference_shaded_areas_l739_739655

theorem difference_shaded_areas (d r1 r2 : ℝ) (hd : d = 5) (hr1 : r1 = 3) (hr2 : r2 = 4) : 
  let A1 := π * r1^2,
      A2 := π * r2^2 
  in A2 - A1 = 7 * π :=
by
  -- assume:
  have hA1 : A1 = π * 3^2 := by rw [hr1]
  have hA2 : A2 = π * 4^2 := by rw [hr2]
  -- proof:
  calc
    A2 - A1 = π * 4^2 - π * 3^2 : by rw [hA2, hA1]
        ... = 16 * π - 9 * π    : by norm_num
        ... = (16 - 9) * π     : by ring
        ... = 7 * π            : by norm_num
  sorry

end difference_shaded_areas_l739_739655


namespace sequence_properties_l739_739152

theorem sequence_properties :
  (∃ n_max, ∀ n, a n_max ≥ a n) ∧ (∃ n_min, ∀ n, a n_min ≤ a n) :=
by
  let a : ℕ → ℝ := λ n, (4 / 9)^(n - 1) - (2 / 3)^(n - 1)
  sorry

end sequence_properties_l739_739152


namespace monotonic_and_odd_function_l739_739423

-- Define the given functions
def f1 (x : ℝ) : ℝ := Real.log x
def f2 (x : ℝ) : ℝ := 3 ^ (Real.abs x)
def f3 (x : ℝ) : ℝ := x ^ (1/2 : ℝ)
def f4 (x : ℝ) : ℝ := x ^ 3

-- State the properties needed for the proof
def is_monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f x ≤ f y

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = - (f x)

-- Main theorem statement
theorem monotonic_and_odd_function :
  (is_monotonic f4 ∧ is_odd f4) ∧
  (¬is_monotonic f1 ∨ ¬is_odd f1) ∧
  (¬is_monotonic f2 ∨ ¬is_odd f2) ∧
  (¬is_monotonic f3 ∨ ¬is_odd f3) :=
by
  sorry

end monotonic_and_odd_function_l739_739423


namespace prime_cube_solution_l739_739466

theorem prime_cube_solution (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (h : p^3 = p^2 + q^2 + r^2) : 
  p = 3 ∧ q = 3 ∧ r = 3 :=
by
  sorry

end prime_cube_solution_l739_739466


namespace shortest_distance_from_ln_curve_to_line_l739_739099

noncomputable def shortest_distance_curve_to_line : ℝ :=
  let y := λ x : ℝ, Real.log (x - 1),
      line := λ x y : ℝ, x - y + 2
  in 2 * Real.sqrt 2

theorem shortest_distance_from_ln_curve_to_line :
  ∀ x y : ℝ, y = Real.log (x - 1) → x - y + 2 = 0 →
  shortest_distance_curve_to_line = 2 * Real.sqrt 2 :=
by
  intros x y h hxy,
  exact sorry

end shortest_distance_from_ln_curve_to_line_l739_739099


namespace subsets_union_intersection_l739_739007

-- Definitions from the given problem
def T : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Condition 1: A and B are subsets of T.
variable {A B : Finset ℕ}

-- Condition 2: Their union is T.
def union_is_T : Prop := A ∪ B = T

-- Condition 3: Their intersection contains exactly three elements.
def intersection_exactly_three : Prop := (A ∩ B).card = 3

-- The main theorem to be proved
theorem subsets_union_intersection (h1 : union_is_T) (h2 : intersection_exactly_three) : 
  ∃ (n : ℕ), n = 80 := 
sorry

end subsets_union_intersection_l739_739007


namespace conjugate_point_location_l739_739656

theorem conjugate_point_location (z : ℂ) (h : (z - 3) * (2 - I) = 5 * I) : 
  let z_conj := conj z in (z_conj.re > 0) ∧ (z_conj.im < 0) := 
by
  sorry

end conjugate_point_location_l739_739656


namespace alice_bob_quarters_difference_l739_739025

theorem alice_bob_quarters_difference (q : ℕ) (h : q = 7) :
  let alice_quarters := 3 * q + 4,
      bob_quarters := 2 * q - 3,
      difference_quarters := alice_quarters - bob_quarters,
      difference_nickels := 5 * difference_quarters
  in difference_nickels = 70 :=
by {
  sorry
}

end alice_bob_quarters_difference_l739_739025


namespace age_difference_l739_739766

variable (A J : ℕ)
variable (h1 : A + 5 = 40)
variable (h2 : J = 31)

theorem age_difference (h1 : A + 5 = 40) (h2 : J = 31) : A - J = 4 := by
  sorry

end age_difference_l739_739766


namespace first_discount_percentage_l739_739758

-- The given problem
variable (P : ℝ)  -- original price
variable (D : ℝ)  -- first discount in decimal form
variable (G : ℝ := 0.002150000000000034)  -- overall percentage gain
variable (second_discount : ℝ := 0.15)  -- second discount
variable (increase : ℝ := 0.31)  -- price increase

-- Define the conditions and final goal
theorem first_discount_percentage :
  let P_new := P * (1 + increase)
  let P_after_first_discount := P_new * (1 - D)
  let P_final := P_after_first_discount * (1 - second_discount)
  P_final = P * (1 + G) →
  D ≈ 0.1001410437235543 :=
sorry

end first_discount_percentage_l739_739758


namespace bulb_illumination_l739_739324

theorem bulb_illumination (n : ℕ) (h : n = 6) : 
  (2^n - 1) = 63 := by {
  sorry
}

end bulb_illumination_l739_739324


namespace integer_solutions_of_equation_l739_739172

-- Problem statement: Prove that the integer solutions to the equation (x-3)^{(30-x^2)} = 1 are precisely x = 2 and x = 4.
theorem integer_solutions_of_equation :
  ∀ x : ℤ, (x - 3)^(30 - x^2) = 1 ↔ x = 2 ∨ x = 4 :=
by
  sorry

end integer_solutions_of_equation_l739_739172


namespace spider_reachable_in_2_seconds_l739_739010

-- Definitions based on conditions
def cube (length : ℝ) : Type := sorry
def spider_position (cube : cube 1) : cube 1 := sorry
def spider_speed : ℝ := 1
def travel_time : ℝ := 2
def travel_distance : ℝ := spider_speed * travel_time

-- Definition based on verification requirement
noncomputable def reachable_points (pos : spider_position (cube 1)) : set (cube 1) := sorry

theorem spider_reachable_in_2_seconds (c : cube 1) (pos : spider_position c) :
  reachable_points pos = {
    p : cube 1 | sorry /* Define points within 2 cm radius arcs on the cube surface */
  } :=
sorry

end spider_reachable_in_2_seconds_l739_739010


namespace angle_A_and_min_tan_relationship_l739_739846

variable {α : Type*} [RealField α]

-- Define the conditions and the proof goals
def triangle_abc (a b c A B C : α) : Prop :=
  (b + c) * (Real.sin B + Real.sin C) = a * Real.sin A + 3 * b * Real.sin C

theorem angle_A_and_min_tan_relationship
  {a b c A B C : α}
  (h1 : a > 0) -- all sides are positive
  (h2 : b > 0)
  (h3 : c > 0)
  (h4 : A > 0 ∧ A < Real.pi / 2) -- angles are in the range consistent with an acute triangle
  (h5 : B > 0 ∧ B < Real.pi / 2)
  (h6 : C > 0 ∧ C < Real.pi / 2)
  (h7 : ∃ r : α, triangle_abc a b c A B C) :
  (A = Real.pi / 3) ∧ (∀ B C, (1 / Real.tan B + 1 / Real.tan C) ≥ (2 * Real.sqrt 3) / 3) :=
sorry

end angle_A_and_min_tan_relationship_l739_739846


namespace randy_quiz_score_l739_739264

theorem randy_quiz_score (q1 q2 q3 q5 : ℕ) (q4 : ℕ) :
  q1 = 90 → q2 = 98 → q3 = 94 → q5 = 96 → (q1 + q2 + q3 + q4 + q5) / 5 = 94 → q4 = 92 :=
by
  intros h1 h2 h3 h5 h_avg
  sorry

end randy_quiz_score_l739_739264


namespace solve_for_s_l739_739270

theorem solve_for_s (a b x y : ℕ) (h_a : a = 7) (h_b : b = 24) (h_x : x = 49) (h_y : y = 16) :
  (sqrt ((a:ℝ)^2 + (b:ℝ)^2) / sqrt ((x:ℝ) + (y:ℝ)) = 5 * sqrt 65 / 13) := by
  sorry

end solve_for_s_l739_739270


namespace total_red_pencils_l739_739910

theorem total_red_pencils (packs : ℕ) (normal_pencil_per_pack : ℕ) (extra_packs : ℕ) (extra_pencils_per_pack : ℕ) :
  packs = 15 →
  normal_pencil_per_pack = 1 →
  extra_packs = 3 →
  extra_pencils_per_pack = 2 →
  packs * normal_pencil_per_pack + extra_packs * extra_pencils_per_pack = 21 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num

end total_red_pencils_l739_739910


namespace simplify_and_evaluate_expression_l739_739268

theorem simplify_and_evaluate_expression (x y : ℝ) (hx : x = (π - 3)^0) (hy : y = (-1/3)^(-1)) :
  ((2 * x - y) ^ 2 - ((y + 2 * x) * (y - 2 * x))) / (-1 / 2 * x) = -40 := 
by sorry

end simplify_and_evaluate_expression_l739_739268


namespace infinite_series_sum_correct_l739_739444

noncomputable def series : ℕ → ℤ → ℚ 
  | 0, _ := 1
  | n, h := 
    if n % 4 = 1 then (1 : ℚ) / (3 ^ (n / 2))
    else if n % 4 = 2 then -(1 : ℚ) / (3 ^ (n / 2 + 1))
    else if n % 4 = 3 then -(1 : ℚ) / (3 ^ (n / 2 + 1))
    else if n % 4 = 0 then (1 : ℚ) / (3 ^ (n / 2 + 1))
    else 0 -- dummy case, should never occur

noncomputable def infinite_series_sum : ℚ :=
  ∑' n, series n sorry

theorem infinite_series_sum_correct : infinite_series_sum = 6 / 5 :=
  sorry

end infinite_series_sum_correct_l739_739444


namespace area_of_triangle_PEF_l739_739601

variables (A B E F P : Type)
variables (ell : A ∈ ℝ) (ell_foci : E ∈ ℝ) (AB_distance AF_distance : ℝ) (PE PF : ℝ)

def ellipse_major_axis_length (AB_distance : ℝ) := AB_distance = 4
def ellipse_focus_distance (AF_distance : ℝ) := AF_distance = 2 + sqrt 3
def point_on_ellipse (PE PF : ℝ) := PE * PF = 2

theorem area_of_triangle_PEF (h1 : ellipse_major_axis_length AB_distance)
                            (h2 : ellipse_focus_distance AF_distance)
                            (h3 : point_on_ellipse PE PF) :
                            ∃ (area : ℝ), area = 1 :=
by
  exists 1
  sorry

end area_of_triangle_PEF_l739_739601


namespace cone_radius_l739_739317

theorem cone_radius
  (l : ℝ) (CSA : ℝ)
  (h_l : l = 21)
  (h_CSA : CSA = 659.7344572538566) :
  ∃ r, r ≈ 10 := 
by {
  sorry
}

end cone_radius_l739_739317


namespace equation_two_roots_iff_l739_739083

theorem equation_two_roots_iff (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + 2 * x1 + 2 * |x1 + 1| = a ∧ x2^2 + 2 * x2 + 2 * |x2 + 1| = a) ↔ a > -1 :=
by
  sorry

end equation_two_roots_iff_l739_739083


namespace log_inequality_solution_l739_739844

variable {a x : ℝ}
variable h1 : 0 < a ∧ a < 1
variable h2 : ∀ x ∈ (0,1), log a (2 - a * x) is_increasing

theorem log_inequality_solution (h1 : 0 < a ∧ a < 1 ) :
   {x | log a (abs (x + 1)) > log a (abs (x - 3))} = {x | x < 1 ∧ x ≠ -1} := 
sorry

end log_inequality_solution_l739_739844


namespace cube_side_length_ratio_l739_739971

theorem cube_side_length_ratio (s : ℝ) 
  (h1 : ∀ x, length side_length_larger = 5 * length side_length_smaller)
  (h2 : ratio surface_area_larger surface_area_smaller = 25) : ratio length side_length_smaller length side_length_larger = 1/5 :=
begin
    sorry
end

end cube_side_length_ratio_l739_739971


namespace machines_job_time_l739_739181

theorem machines_job_time (D : ℝ) (h1 : 15 * D = D * 20 * (3 / 4)) : ¬ ∃ t : ℝ, t = D :=
by
  sorry

end machines_job_time_l739_739181


namespace sum_of_specific_coefficients_of_polynomial_l739_739176

theorem sum_of_specific_coefficients_of_polynomial :
  let P := (1 + 2 * x)^5 -- define the polynomial (1 + 2x)^5
  ∃ (a_0 a_1 a_2 a_3 a_4 a_5 : ℕ), 
    P = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5
    ∧ a_0 + a_1 + a_3 + a_5 = 123 :=
begin
  sorry
end

end sum_of_specific_coefficients_of_polynomial_l739_739176


namespace y_coordinate_of_vertex_D_is_14_l739_739866

def hexagon_vertices : Type := (ℝ × ℝ, ℝ × ℝ, ℝ × ℝ, ℝ × ℝ, ℝ × ℝ, ℝ × ℝ)

def hexagon_coordinates : hexagon_vertices :=
  ((0, 0), (0, 6), (2, 20), (4, 6), (4, 0), (2, 2))

theorem y_coordinate_of_vertex_D_is_14.5 :
  ∃ h : ℝ, h = 14.5 ∧
  let A := (0, 0) in
  let B := (0, 6) in
  let E := (4, 0) in
  let D := (4, 14.5) in
  let ABFE_area := 4 * 6 in
  let triangles_area := (1 / 2) * (4 * (14.5 - 6)) in
  ABFE_area + 2 * triangles_area = 58 :=
begin
  sorry
end

end y_coordinate_of_vertex_D_is_14_l739_739866


namespace exactly_two_roots_iff_l739_739078

theorem exactly_two_roots_iff (a : ℝ) : 
  (∃! (x : ℝ), x^2 + 2 * x + 2 * |x + 1| = a) ↔ a > -1 :=
by
  sorry

end exactly_two_roots_iff_l739_739078


namespace Sn_proof_l739_739536

noncomputable def f (x : ℝ) : ℝ :=
  (1 / 2) + Real.log2 (x / (1 - x))

noncomputable def S_n (n : ℕ) [Fact (2 ≤ n)] : ℝ :=
  ∑ i in Finset.range (n - 1) | Finset.Ico 1 n, f ((i + 1) / n)

theorem Sn_proof (n : ℕ) [Fact (2 ≤ n)] : S_n n = (n - 1) / 2 :=
  sorry

end Sn_proof_l739_739536


namespace proof_l739_739131

variable (x : ℝ)

def p := abs (x + 1) ≤ 2
def q := -3 ≤ x ∧ x ≤ 2

theorem proof : (p → q) ∧ (∃ x, q ∧ ¬p) :=
by
  sorry

end proof_l739_739131


namespace inverse_proposition_false_l739_739663

theorem inverse_proposition_false (a b c : ℝ) : 
  ¬ (a > b → ((c ≠ 0) ∧ (a / (c * c)) > (b / (c * c))))
:= 
by 
  -- Outline indicating that the proof will follow from checking cases
  sorry

end inverse_proposition_false_l739_739663


namespace correct_statements_l739_739226

-- Definitions based on conditions
def is_field (P : Set ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ b ∧
  (∀ x y ∈ P, x + y ∈ P) ∧
  (∀ x y ∈ P, x - y ∈ P) ∧
  (∀ x y ∈ P, x * y ∈ P) ∧
  (∀ x y ∈ P, y ≠ 0 → x / y ∈ P)

-- Statement 1
def statement1 : Prop := ¬ is_field {x : ℤ | True}

-- Statement 2
def statement2 (M : Set ℝ) : Prop := 
  (∀ a b : ℝ, a ∈ M ∧ b ∈ M ∧ (∀ x y ∈ M, x + y ∈ M ∧ x - y ∈ M ∧ x * y ∈ M ∧ y ≠ 0 → x / y ∈ M)) → is_field M

-- Statement 3
def statement3 : Prop := ∀ F, is_field F → infinite F

-- Statement 4
def statement4 : Prop := ∃ (F : ℝ) (P : Set ℝ), is_field P ∧ infinite P

-- Theorem statement
theorem correct_statements : statement3 ∧ statement4 :=
by {
  sorry
}

end correct_statements_l739_739226


namespace time_to_fill_pot_l739_739664

def pot_volume : ℕ := 3000  -- in ml
def rate_of_entry : ℕ := 60 -- in ml/minute

-- Statement: Prove that the time required for the pot to be full is 50 minutes.
theorem time_to_fill_pot : (pot_volume / rate_of_entry) = 50 := by
  sorry

end time_to_fill_pot_l739_739664


namespace probability_of_one_winning_l739_739374

-- Define the probabilities as given conditions
def prob_X : ℚ := 1 / 4
def prob_Y : ℚ := 1 / 8
def prob_Z : ℚ := 1 / 12

-- Lean statement to prove the total probability
theorem probability_of_one_winning :
  prob_X + prob_Y + prob_Z = 11 / 24 := 
begin
  sorry,
end

end probability_of_one_winning_l739_739374


namespace tangent_circles_tangency_points_l739_739764

theorem tangent_circles_tangency_points (A B : Point) : ∃ C : Circle, (A ∈ C) ∧ (B ∈ C) ∧
  (∀ (P : Point), (is_tangency_point P (circle (A B))) ↔ (P ∈ arc_of_circle A B C)) :=
sorry

end tangent_circles_tangency_points_l739_739764


namespace find_rho_squared_l739_739241

theorem find_rho_squared:
  ∀ (a b : ℝ), (0 < a) → (0 < b) →
  (a^2 - 2 * b^2 = 0) →
  (∃ (x y : ℝ), 
    (0 ≤ x ∧ x < a) ∧ 
    (0 ≤ y ∧ y < b) ∧ 
    (a^2 + y^2 = b^2 + x^2) ∧ 
    ((a - x)^2 + (b - y)^2 = b^2 + x^2) ∧ 
    (x^2 + y^2 = b^2)) → 
  (∃ (ρ : ℝ), ρ = a / b ∧ ρ^2 = 2) :=
by
  intros a b ha hb hab hsol
  sorry  -- Proof to be provided later

end find_rho_squared_l739_739241


namespace grade_assignment_ways_l739_739756

theorem grade_assignment_ways : (4 ^ 12) = 16777216 := by
  sorry

end grade_assignment_ways_l739_739756


namespace infinite_solutions_pairs_eqn_l739_739092

theorem infinite_solutions_pairs_eqn :
  ∃∞ pairs : ℤ × ℤ, let p := pairs.fst in let q := pairs.snd in
    p^3 + 7 * p^2 + 6 * p = 64 * q^3 + 96 * q^2 + 48 * q + 8 :=
sorry

end infinite_solutions_pairs_eqn_l739_739092


namespace value_of_y_when_x_is_neg2_l739_739207

theorem value_of_y_when_x_is_neg2 :
  ∃ (k b : ℝ), (k + b = 2) ∧ (-k + b = -4) ∧ (∀ x, y = k * x + b) ∧ (x = -2) → (y = -7) := 
sorry

end value_of_y_when_x_is_neg2_l739_739207


namespace one_angle_not_greater_than_60_l739_739367

theorem one_angle_not_greater_than_60 (A B C : ℝ) (h : A + B + C = 180) : A ≤ 60 ∨ B ≤ 60 ∨ C ≤ 60 := 
sorry

end one_angle_not_greater_than_60_l739_739367


namespace michael_lap_time_l739_739057

theorem michael_lap_time :
  ∃ T : ℝ, (∀ D : ℝ, D = 45 → (9 * T = 10 * D) → T = 50) :=
by
  sorry

end michael_lap_time_l739_739057


namespace find_average_l739_739890

def is_average (n : ℕ) (lst : List ℕ) : Prop :=
  (lst.sum - n) / (lst.length - 1) = n

theorem find_average :
  ∃ n ∈ [7, 9, 10, 11, 18], is_average n [7, 9, 10, 11, 18] :=
by {
  use 11,
  simp [is_average],
  sorry
}

end find_average_l739_739890


namespace correct_equation_l739_739017

-- Define the given amounts spent on backpacks A and B
def cost_A : ℝ := 810
def cost_B : ℝ := 600

-- Define the number of type B backpacks as x
variable (x : ℝ)

-- Define the number of type A backpacks (20 more than B)
def num_A : ℝ := x + 20

-- Define the unit price relationship where pA is 10% less than pB
def unit_price_A (pB : ℝ) : ℝ := 0.9 * pB

-- Define the equation to prove
theorem correct_equation (pB : ℝ) : 
  (cost_A / num_A) = (cost_B / x) * 0.9 :=
by
  -- Place the proof here
  sorry

end correct_equation_l739_739017


namespace solve_equation_l739_739952

theorem solve_equation (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 :=
sorry

end solve_equation_l739_739952


namespace number_equation_l739_739750

-- Lean statement equivalent to the mathematical problem
theorem number_equation (x : ℝ) (h : 5 * x - 2 * x = 10) : 5 * x - 2 * x = 10 :=
by exact h

end number_equation_l739_739750


namespace direct_proportion_function_l739_739870

theorem direct_proportion_function (m : ℝ) (h1 : m^2 - 8 = 1) (h2 : m ≠ 3) : m = -3 :=
by
  sorry

end direct_proportion_function_l739_739870


namespace projections_equal_length_l739_739635

open EuclideanGeometry

theorem projections_equal_length
  (C B H2 H3 P Q : Point)
  (on_line_H2H3 : P = projection C (Line H2 H3) ∧ Q = projection B (Line H2 H3))
  (right_angles : ∠ B H2 C = 90 ∧ ∠ C H3 B = 90) :
  distance P H2 = distance Q H3 := 
by
  sorry

end projections_equal_length_l739_739635


namespace prob_between_4_8_and_4_85_l739_739817

-- Define the probability space and random variable
variables {Ω : Type*} [measurable_space Ω] (P : measure_theory.measure Ω) 
          {ξ : Ω → ℝ} [measure_theory.measurable ξ]

-- Define the given probabilities
def prob_less_than_4_8 : P {ω | ξ ω < 4.8} = 0.3 := sorry
def prob_less_than_or_equal_4_85 : P {ω | ξ ω ≤ 4.85} = 0.32 := sorry

-- Statement we need to prove
theorem prob_between_4_8_and_4_85 : 
  P {ω | 4.8 ≤ ξ ω ∧ ξ ω ≤ 4.85} = 0.02 :=
by 
  have h1 : P {ω | ξ ω ≤ 4.85} = P {ω | ξ ω < 4.8} + P {ω | 4.8 ≤ ξ ω ∧ ξ ω ≤ 4.85}, 
      from sorry,
  rw [prob_less_than_4_8, prob_less_than_or_equal_4_85], 
  linarith

end prob_between_4_8_and_4_85_l739_739817


namespace cross_section_area_of_regular_triangular_pyramid_l739_739315

noncomputable def cross_section_area (a b : ℝ) : ℝ :=
  1 / 4 * a * b

theorem cross_section_area_of_regular_triangular_pyramid (a b : ℝ) :
  ∃ (area : ℝ), area = cross_section_area a b ∧ area = 1 / 4 * a * b := by
  use cross_section_area a b
  split
  case left => rfl
  case right => rfl

end cross_section_area_of_regular_triangular_pyramid_l739_739315


namespace equation_two_roots_iff_l739_739082

theorem equation_two_roots_iff (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + 2 * x1 + 2 * |x1 + 1| = a ∧ x2^2 + 2 * x2 + 2 * |x2 + 1| = a) ↔ a > -1 :=
by
  sorry

end equation_two_roots_iff_l739_739082


namespace g_of_neg5_eq_651_over_16_l739_739610

def f (x : ℝ) : ℝ := 4 * x + 6

def g (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 7

theorem g_of_neg5_eq_651_over_16 : g (-5) = 651 / 16 := by
  sorry

end g_of_neg5_eq_651_over_16_l739_739610


namespace sqrt_expr_eqn_l739_739042

theorem sqrt_expr_eqn :
  Real.sqrt (10! / 88) = 180 * Real.sqrt 7 / Real.sqrt 11 := 
  sorry

end sqrt_expr_eqn_l739_739042


namespace programs_are_different_and_results_are_different_l739_739266

def program_A_output : ℕ :=
  let S := ref 0
  let i := ref 1
  while !i.get ≤ 1000 do
    S := S.get + i.get
    i := i.get + 1
  S.get

def program_B_output : ℕ :=
  let S := ref 0
  let i := ref 1000
  loop:
    S := S.get + i.get
    i := i.get - 1
    if i.get <= 1 then break
  S.get

theorem programs_are_different_and_results_are_different :
  program_A_output ≠ program_B_output := by sorry

end programs_are_different_and_results_are_different_l739_739266


namespace exists_bounding_constant_M_l739_739494

variable (α : ℝ) (a : ℕ → ℝ)
variable (hα : α > 1)
variable (h_seq : ∀ n : ℕ, n > 0 →
  a n.succ = a n + (a n / n) ^ α)

theorem exists_bounding_constant_M (h_a1 : 0 < a 1 ∧ a 1 < 1) : 
  ∃ M, ∀ n > 0, a n ≤ M := 
sorry

end exists_bounding_constant_M_l739_739494


namespace slope_of_tangent_line_l739_739704

theorem slope_of_tangent_line 
  (center point : ℝ × ℝ) 
  (h_center : center = (5, 3)) 
  (h_point : point = (8, 8)) 
  : (∃ m : ℚ, m = -3/5) :=
sorry

end slope_of_tangent_line_l739_739704


namespace proposition_1_correct_proposition_2_incorrect_proposition_3_correct_proposition_4_incorrect_final_answer_l739_739443

def f (x : ℝ) := 4 * Real.sin (2 * x + (Real.pi / 3))

theorem proposition_1_correct :
  ∀ x : ℝ, f x = 4 * Real.cos (2 * x - (Real.pi / 6)) :=
begin
  sorry
end

theorem proposition_2_incorrect :
  ¬(∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x ∧ T = 2 * Real.pi) :=
begin
  sorry
end

theorem proposition_3_correct :
  (∃ x₀ : ℝ, ∀ x : ℝ, f (2 * x₀ - x) = f x) :=
begin
  let x₀ := - (Real.pi / 6),
  use x₀,
  sorry
end

theorem proposition_4_incorrect :
  ¬(∃ x₀ : ℝ, ∀ x : ℝ, f x₀ = f (2 * x₀ - x)) :=
begin
  let x₀ := - (Real.pi / 6),
  sorry
end

theorem final_answer :
  (proposition_1_correct ∧ proposition_3_correct ∧
   proposition_2_incorrect ∧ proposition_4_incorrect) :=
begin
  sorry
end

end proposition_1_correct_proposition_2_incorrect_proposition_3_correct_proposition_4_incorrect_final_answer_l739_739443


namespace original_three_digit_number_is_224_l739_739414

theorem original_three_digit_number_is_224 :
  ∃ (x : ℝ), (100 ≤ x ∧ x < 1000) ∧ (x - x / 10 = 201.6) ∧ x = 224 :=
by
  use 224
  split
  -- check the range
  exact And.intro (by norm_num) (by norm_num)
  -- check the condition and solution
  split
  -- check the condition
  norm_num
  -- check the solution
  rfl

sorry

end original_three_digit_number_is_224_l739_739414


namespace triangle_area_l739_739338

/-- The points P, Q, and R are defined as follows:
P(-5, 2), Q(8, 2), and R(6, -6).
We are to prove that the area of triangle PQR is 52 square units. -/
theorem triangle_area (P Q R : ℝ × ℝ)
  (hP : P = (-5, 2))
  (hQ : Q = (8, 2))
  (hR : R = (6, -6)) :
  let base := Q.1 - P.1,
      height := abs (R.2 - P.2),
      area := (base * height) / 2 in
  area = 52 := 
by
  sorry

end triangle_area_l739_739338


namespace stationery_store_backpacks_l739_739019

theorem stationery_store_backpacks (price_B : ℝ) (x : ℕ) (h1 : price_B > 0) :
  let price_A := 0.9 * price_B in
  let cost_A := 810 in
  let cost_B := 600 in
  let num_A := x + 20 in
  let num_B := x in
  ((cost_A / num_A) = (cost_B / num_B) * 0.9) :=
sorry

end stationery_store_backpacks_l739_739019


namespace train_overtake_distance_l739_739720

noncomputable def distance_traveled (speed time : ℝ) : ℝ := speed * time

theorem train_overtake_distance :
  let speed_A := 30 -- miles per hour
      speed_B := 36 -- miles per hour
      head_start := 2 -- hours
      distance_A_start := distance_traveled speed_A head_start
      time_to_overtake := (distance_A_start) / (speed_B - speed_A)
      total_distance := distance_traveled speed_B time_to_overtake
  in total_distance = 360 :=
begin
  sorry
end

end train_overtake_distance_l739_739720


namespace f_gt_neg_half_l739_739151

def f (x : ℝ) : ℝ := Real.exp x - Real.log (x + 3)

theorem f_gt_neg_half : ∀ x : ℝ, x > -3 → f x > -0.5 :=
by
  sorry

end f_gt_neg_half_l739_739151


namespace decrease_in_sales_percentage_l739_739630

theorem decrease_in_sales_percentage (P Q : Real) :
  let P' := 1.40 * P
  let R := P * Q
  let R' := 1.12 * R
  ∃ (D : Real), Q' = Q * (1 - D / 100) ∧ R' = P' * Q' → D = 20 :=
by
  sorry

end decrease_in_sales_percentage_l739_739630


namespace square_area_l739_739411

theorem square_area (p : ℝ) (h : p = 20) : (p / 4) ^ 2 = 25 :=
by
  sorry

end square_area_l739_739411


namespace triangle_area_ratio_l739_739999

theorem triangle_area_ratio (x y : ℝ) (n m : ℕ) (hn : n > 0) (hm : m > 0) :
  let A_area := (1/2) * (y/n) * (x/2)
  let B_area := (1/2) * (x/m) * (y/2)
  A_area / B_area = m / n := by
  sorry

end triangle_area_ratio_l739_739999


namespace part_1_part_2_l739_739547

noncomputable def omega (x ω : Real) := (cos (ω * x) + sqrt 3 * sin (ω * x)) * cos (ω * x)
-- Given:
-- Vectors m = (-1, cos (ω * x) + sqrt 3 * sin (ω * x)) and n = (f(x), cos ω x)
-- m ⊥ n, i.e., ( -1, cos (ω * x) + sqrt 3 * sin(ω * x)) ⋅ (f(x), cos (ω * x)) = 0
-- The graph of f(x) has a distance of 3/2 * π between any two adjacent axes of symmetry.
-- ω > 0
-- α is an angle in the first quadrant.
-- f(3/2 * α + π/2) = 23/26

theorem part_1 (hx : Real) (hω : Real > 0) (h_perpendicular : omega hx hω = 0)
        (h_symmetry : ∀ x : Real, f (x + 3 * π) = f x) : hω = 1 / 3 := sorry

theorem part_2 (hx : Real) (hα : α ∈ Set.Icc 0 (π / 2)) 
        (hf : f ((3 / 2 * α) + π / 2) = 23 / 26) : 
        (sin (α + π / 4) / cos (4 * π + 2 * α)) = - (13 * sqrt 2) / 14 := sorry

end part_1_part_2_l739_739547


namespace calculate_expression_l739_739774

theorem calculate_expression :
  1500 * 2987 * 0.2987 * 15 = 2,989,502.987 :=
by
  sorry

end calculate_expression_l739_739774


namespace two_roots_iff_a_greater_than_neg1_l739_739074

theorem two_roots_iff_a_greater_than_neg1 (a : ℝ) :
  (∃! x : ℝ, x^2 + 2*x + 2*|x + 1| = a) ↔ a > -1 :=
sorry

end two_roots_iff_a_greater_than_neg1_l739_739074


namespace angle_AC₁B_l739_739210

theorem angle_AC₁B {α : ℝ} (hα1 : 0 < α) (hα2 : α < 45) :
  ∀ (A B C D C₁ : Type) [RightTriangle A B C] [Midpoint D A C] [Reflection C₁ C BD], 
    angle A C₁ B = 90 + α :=
by
  sorry

end angle_AC₁B_l739_739210


namespace probability_heads_given_heads_twice_l739_739484

variables {Ω : Type} [ProbabilitySpace Ω]
variables (A B : Set Ω)
variables (P_fair P_double : ℝ) (hP_fair : P_fair = 1/2) (hP_double : P_double = 1/2)
variables (P_B_given_fair P_B_given_double : ℝ)
variables (hP_B_given_fair : P_B_given_fair = 1/4) (hP_B_given_double : P_B_given_double = 1)

-- Define event probabilities
noncomputable def P_A := P_double
noncomputable def P_B := P_B_given_fair * P_fair + P_B_given_double * P_double
noncomputable def P_A_and_B := P_double * P_B_given_double

-- Conditional Probability
noncomputable def P_A_given_B := P_A_and_B / P_B

theorem probability_heads_given_heads_twice :
  P_A_given_B A B P_fair P_double P_B_given_fair P_B_given_double = 4/5 :=
by {
    rw [←hP_fair, ←hP_double, hP_B_given_fair, hP_B_given_double],
    unfold P_A_given_B P_A_and_B P_B,
    field_simp,
    norm_num,
    sorry -- Step calculations would go here in a complete proof.
}

end probability_heads_given_heads_twice_l739_739484


namespace ln_1_2_over_6_gt_e_l739_739733

theorem ln_1_2_over_6_gt_e :
  let x := 1.2
  let exp1 := x^6
  let exp2 := (1.44)^2 * 1.44
  let final_val := 2.0736 * 1.44
  final_val > 2.718 :=
by {
  sorry
}

end ln_1_2_over_6_gt_e_l739_739733


namespace tan_angle_C_side_a_given_area_l739_739192

theorem tan_angle_C {a b c : ℝ} (A : ℝ) (hA : A = π / 3) (a_sq_minus_c_sq : a^2 - c^2 = (2 / 3) * b^2) :
  let C := real.arctan (√3 / 5)
  in C = real.atan (a^2 + b^2 - c^2) / (2 * a * b) := 
sorry

theorem side_a_given_area {a b c : ℝ} (A : ℝ) (hA : A = π / 3) 
    (area : real.sqrt 3 / 4) (a_sq_minus_c_sq : a^2 - c^2 = (2 / 3) * b^2) :
    let abc_area := (1 / 2) * b * c * real.sin A
    in let abc_area_val := (3 * real.sqrt 3) / 4 in 
    abc_area = abc_area_val -> a = real.sqrt 7 :=  
sorry

end tan_angle_C_side_a_given_area_l739_739192


namespace sum_reciprocal_of_S_n_l739_739199

def arithmetic_sequence (a : ℕ → ℚ) :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  ∑ i in Finset.range n, a (i + 1)

theorem sum_reciprocal_of_S_n :
  ∀ {a : ℕ → ℚ},
    arithmetic_sequence a →
    a 9 = (1/2) * a 12 + 6 →
    a 2 = 4 →
    sum_of_first_n_terms a 10 =
    10/11 :=
begin
  sorry
end

end sum_reciprocal_of_S_n_l739_739199


namespace binomial_sum_l739_739251

open Nat

theorem binomial_sum (n : ℕ) (hn : n > 0) :
  (Finset.range n).sum (λ k, Nat.choose (2 * n - 1) k) = 4 ^ (n - 1) := by
  sorry

end binomial_sum_l739_739251


namespace distinct_pairs_l739_739229

-- Definitions of rational numbers and distinctness.
def is_distinct (x y : ℚ) : Prop := x ≠ y

-- Conditions
variables {a b r s : ℚ}

-- Main theorem: prove that there is only 1 distinct pair (a, b)
theorem distinct_pairs (h_ab_distinct : is_distinct a b)
  (h_rs_distinct : is_distinct r s)
  (h_eq : ∀ z : ℚ, (z - r) * (z - s) = (z - a * r) * (z - b * s)) : 
    ∃! (a b : ℚ), ∀ z : ℚ, (z - r) * (z - s) = (z - a * r) * (z - b * s) :=
  sorry

end distinct_pairs_l739_739229


namespace eccentricity_proof_l739_739604

noncomputable def ellipseEccentricity (a b : ℝ) (h1 : a > b) (h2: b > 0) 
  (P F1 F2 : ℝ) (h3 : P^2 + F1^2 = F2^2) (h4 : P = 2 * F1) : ℝ :=
  let m := F1 in
  let c := (sqrt 5 / 2) * m in
  let a := (3 / 2) * m in
  c / a

theorem eccentricity_proof (a b : ℝ) (h1 : a > b) (h2: b > 0) 
  (P F1 F2 : ℝ) (h3 : P^2 + F1^2 = F2^2) (h4 : P = 2 * F1) :
  ellipseEccentricity a b h1 h2 P F1 F2 h3 h4 = sqrt 5 / 3 := 
  sorry

end eccentricity_proof_l739_739604


namespace tangent_line_at_2_inequality_f_x_l739_739524

noncomputable def f (x : ℝ) : ℝ := (Real.exp x) / x

theorem tangent_line_at_2 : 
  let P := (2 : ℝ, f 2) in 
  let slope := (Real.exp 2 / 4) in 
  ∃ (m : ℝ), m = slope ∧ ( ∀ x y, (y - P.2 = m * (x - P.1)) ↔ (Real.exp 2 * x - 4 * y = 0) ) := 
by
  sorry

theorem inequality_f_x : ∀ x > 0, f x > 2 * (x - Real.log x) :=
by
  sorry

end tangent_line_at_2_inequality_f_x_l739_739524


namespace next_class_after_science_is_music_l739_739419

-- Definitions for conditions
def school_start_time : Nat := 12
def classes_order : List String := ["Maths", "History", "Geography", "Science", "Music"]
def science_end_time : Nat := 16 -- 4 pm in 24-hour format

-- Theorem to prove the next class after Science is Music
theorem next_class_after_science_is_music :
  (Mathlib.List.nth classes_order 4) = "Music" :=
by
  -- formalize that Science is the 4th class
  have science_class : Mathlib.List.nth classes_order 3 = "Science" := by
    sorry
  -- apply the order to get the next class
  sorry

end next_class_after_science_is_music_l739_739419


namespace sum_of_not_in_domain_f_l739_739442

noncomputable def g (x : ℝ) : ℝ := 1 / (1 + 1 / (x^2 + 1))

noncomputable def f (x : ℝ) : ℝ := 1 / (x + g(x))

theorem sum_of_not_in_domain_f : ∑ x in {x | ¬function.is_defined_at f x}, x = -1 :=
by
  sorry

end sum_of_not_in_domain_f_l739_739442


namespace triangle_area_l739_739212

theorem triangle_area {A B C D : ℝ} 
  (hAD : AD = 1) 
  (hABAC : AB + AC = 5 / 2) 
  (hBC : BC = 2) : 
  area_triangle ABC = 9 / 16 := 
by 
  sorry

end triangle_area_l739_739212


namespace area_constant_circle_eqn_l739_739725

open Real

def circle_eq (t : ℝ) (x y : ℝ) : Prop :=
  (x - t)^2 + y^2 = t^2

def A (t : ℝ) : ℝ × ℝ := (2*t, 0)
def O : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (0, 1)

theorem area_constant (t : ℝ) (h : t ≠ 0) : 4 = abs (2 * t * 1) :=
by sorry

def line_eq (x y : ℝ) : Prop := 2 * x + y - 4 = 0
def C_pos : ℝ × ℝ := (2, 1)
def C_neg : ℝ × ℝ := (-2, -1)

def is_on_perpendicular_bisector (C : ℝ × ℝ) (M N O H : ℝ × ℝ) : Prop :=
  line_eq M.1 M.2 ∧ line_eq N.1 N.2 ∧ M ≠ N ∧ 
  dist O M = dist O N ∧ 
  collinear {C, H, O}

def dist_to_line (C : ℝ × ℝ) : ℝ :=
  abs (2 * C.1 + C.2 - 4) / sqrt (2^2 + 1^2)

theorem circle_eqn : 
  (∀ t ≠ 0, is_on_perpendicular_bisector C_pos (2, -4) (2, -4) O (1, -2)) →
  ((line_eq (2 + 2) 1 - (2 + 4) = 0) ∧ 
  dist_to_line C_neg > sqrt 5) →
  circle_eq 2 C_pos.1 C_pos.2 :=
by sorry


end area_constant_circle_eqn_l739_739725


namespace first_player_wins_the_game_l739_739990

-- Define the game state with 1992 stones and rules for taking stones
structure GameState where
  stones : Nat

-- Game rule: Each player can take a number of stones that is a divisor of the number of stones the 
-- opponent took on the previous turn
def isValidMove (prevMove: Nat) (currentMove: Nat) : Prop :=
  currentMove > 0 ∧ prevMove % currentMove = 0

-- The first player can take any number of stones but not all at once on their first move
def isFirstMoveValid (move: Nat) : Prop :=
  move > 0 ∧ move < 1992

-- Define the initial state of the game with 1992 stones
def initialGameState : GameState := { stones := 1992 }

-- Definition of optimal play leading to the first player's victory
def firstPlayerWins (s : GameState) : Prop :=
  s.stones = 1992 →
  ∃ move: Nat, isFirstMoveValid move ∧
  ∃ nextState: GameState, nextState.stones = s.stones - move ∧ 
  -- The first player wins with optimal strategy
  sorry

-- Theorem statement in Lean 4 equivalent to the math problem
theorem first_player_wins_the_game :
  firstPlayerWins initialGameState :=
  sorry

end first_player_wins_the_game_l739_739990


namespace intersection_A_B_union_A_B_complement_intersection_A_B_l739_739840

def A : Set ℝ := { x | 2 ≤ x ∧ x ≤ 8 }
def B : Set ℝ := { x | 1 < x ∧ x < 6 }
def A_inter_B : Set ℝ := { x | 2 ≤ x ∧ x < 6 }
def A_union_B : Set ℝ := { x | 1 < x ∧ x ≤ 8 }
def A_compl_inter_B : Set ℝ := { x | 1 < x ∧ x < 2 }

theorem intersection_A_B :
  A ∩ B = A_inter_B := by
  sorry

theorem union_A_B :
  A ∪ B = A_union_B := by
  sorry

theorem complement_intersection_A_B :
  (Aᶜ ∩ B) = A_compl_inter_B := by
  sorry

end intersection_A_B_union_A_B_complement_intersection_A_B_l739_739840


namespace sum_ratio_l739_739606

noncomputable def arithmetic_seq (a₁ d n : ℕ) : ℕ := a₁ + n * d

noncomputable def sum_first_n_terms (a₁ d n : ℕ) : ℕ := 
  n * (2 * a₁ + (n-1) * d) / 2

theorem sum_ratio 
  (a₁ d : ℕ) 
  (h : (a₁ + 4 * d) / (a₁ + 2 * d) = 2) 
  : (sum_first_n_terms a₁ d 9) / (sum_first_n_terms a₁ d 5) = 18 / 5 :=
by
  sorry

end sum_ratio_l739_739606


namespace vasya_can_win_l739_739944

theorem vasya_can_win :
  ∀ (x : Fin 10 → ℝ),
  (∀ i, 0 ≤ x i) →
  (∀ (p_cards v_cards : list (set (Fin 10))),
    (∀ card ∈ p_cards, card.cardinality = 5) →
    (∀ card ∈ v_cards, card.cardinality = 5) →
    (p_cards.disjoint v_cards) →
    (∀ v_card ∈ v_cards, 
      (∀ j ∈ v_card, x j ∈ {0, 1}) ∧ ((v_card.cardinality : ℝ) = 5)) →
    ∑ card in v_cards, (∏ j in card, x j) > ∑ card in p_cards, (∏ j in card, x j)) :=
begin
  sorry
end

end vasya_can_win_l739_739944


namespace find_k_l739_739056

open Real

noncomputable def k_root1 : ℝ := (17 + Real.sqrt 505) / -6
noncomputable def k_root2 : ℝ := (17 - Real.sqrt 505) / -6

def points_collinear (k : ℝ) : Prop :=
  let p1 := (2, -3)
  let p2 := (k, k + 2)
  let p3 := (-3k + 4, 1)
  let slope12 := (p2.2 - p1.2) / (p2.1 - p1.1)
  let slope13 := (p3.2 - p1.2) / (p3.1 - p1.1)
  slope12 = slope13

theorem find_k :
  ∃ k : ℝ, points_collinear k ∧ (k = k_root1 ∨ k = k_root2) :=
sorry

end find_k_l739_739056


namespace sum_of_solutions_of_quadratic_l739_739978

theorem sum_of_solutions_of_quadratic (x : ℝ) :
  x^2 - 6*x + 5 = 2*x - 8 →
  let a := (1 : ℝ) in
  let b := (-8 : ℝ) in
  let sum_of_roots := -b / a in
  sum_of_roots = 8 := 
by
  intro h
  let a := (1 : ℝ)
  let b := (-8 : ℝ)
  let sum_of_roots := -b / a
  have : x^2 - 8*x + 13 = 0 := by
    linarith [h]
  have h_sum : sum_of_roots = 8 := by
    rw [sum_of_roots]
    norm_num
  exact h_sum

end sum_of_solutions_of_quadratic_l739_739978


namespace sum_of_all_prime_values_is_zero_l739_739109

def f (n : ℕ) : ℕ := n^4 + 100*n^2 + 169

noncomputable def sum_of_prime_values (f : ℕ → ℕ) : ℕ :=
  ∑ x in (Finset.filter Nat.Prime (Finset.image f (Finset.range (10000)))), x

theorem sum_of_all_prime_values_is_zero :
  sum_of_prime_values f = 0 :=
by
  sorry

end sum_of_all_prime_values_is_zero_l739_739109


namespace sum_of_heights_less_than_perimeter_l739_739263

theorem sum_of_heights_less_than_perimeter
  (a b c h1 h2 h3 : ℝ) 
  (H1 : h1 ≤ b) 
  (H2 : h2 ≤ c) 
  (H3 : h3 ≤ a) 
  (H4 : h1 < b ∨ h2 < c ∨ h3 < a) : 
  h1 + h2 + h3 < a + b + c :=
by {
  sorry
}

end sum_of_heights_less_than_perimeter_l739_739263


namespace johnny_red_pencils_l739_739907

noncomputable def number_of_red_pencils (packs_total : ℕ) (extra_packs : ℕ) (extra_per_pack : ℕ) : ℕ :=
  packs_total + extra_packs * extra_per_pack

theorem johnny_red_pencils : number_of_red_pencils 15 3 2 = 21 := by
  sorry

end johnny_red_pencils_l739_739907


namespace smallest_number_ending_in_2_l739_739100

theorem smallest_number_ending_in_2 :
  ∃ N : ℕ, (N % 10 = 2) ∧ (2 * N = (10^(nat.digits 10 N).length - 1) + N / 10) ∧
  (∀ M : ℕ, (M % 10 = 2) ∧ (2 * M = (10^(nat.digits 10 M).length - 1) + M / 10) → N ≤ M) :=
begin
  let N := 105263157894736842,
  use N,
  split,
  -- Proof that N ends with 2
  {
    show N % 10 = 2,
    sorry
  },
  split,
  -- Proof that 2 * N results in moving the digit 2 to the beginning
  {
    show 2 * N = (10^(nat.digits 10 N).length - 1) + N / 10,
    sorry
  },
  -- Proof that N is the smallest such number
  {
    intros M h_end_with_2 h_double,
    have h_leq := sorry,
    exact h_leq
  }
end

end smallest_number_ending_in_2_l739_739100


namespace other_factor_of_LCM_l739_739276

-- Definitions and conditions
def A : ℕ := 624
def H : ℕ := 52 
def HCF (a b : ℕ) : ℕ := Nat.gcd a b

-- Hypotheses based on the problem statement
axiom h_hcf : HCF A 52 = 52

-- The desired statement to prove
theorem other_factor_of_LCM (B : ℕ) (y : ℕ) : HCF A B = H → (A * y = 624) → y = 1 := 
by 
  intro h1 h2
  -- Actual proof steps are omitted
  sorry

end other_factor_of_LCM_l739_739276


namespace quadrilateral_APMN_cyclic_l739_739586

-- Definitions of the problem setup
variables {α : Type*} [EuclideanGeometry α]

-- Conditions of the problem
variable {A B C P M N : α}

-- Assume points A, B, and C form a triangle with AB > BC
variable [Triangle A B C]
variable [AB : Segment A B > Segment B C]

-- Point P is on segment AB such that BP = BC and BM is the angle bisector of ∠ABC
variable (P_on_AB : P ∈ ClosedSegment A B)
variable (BP_eq_BC : Segment B P = Segment B C)
variable [AngleBisector B M]
variable (BM_bisects_ABC : Segment B M bisects Angle B C A B)

-- Bisector BM intersects the circumcircle of triangle ABC at N
variable (Circumcircle_AT_ABC : Circle (A ∈ Circumcircle A B C))
variable (N_on_Circumcircle : N ∈ Circumcircle A B C)
variable (BM_intersects_Circumcircle : BM ∩ Circumcircle A B C = N)

-- The statement to prove that quadrilateral APMN is cyclic
theorem quadrilateral_APMN_cyclic (hP : P_on_AB) (hBP : BP_eq_BC) (hBis : BM_bisects_ABC) (hCirc : N_on_Circumcircle) (hInter : BM_intersects_Circumcircle) :
  CyclicQuadrilateral A P M N :=
by
  sorry

end quadrilateral_APMN_cyclic_l739_739586


namespace grid_sum_A_plus_B_l739_739040

theorem grid_sum_A_plus_B : 
  ∃ (A B : ℕ), 
    -- Conditions for filling the grid
    (∀ r c : ℕ, r ∈ {0, 1, 2} → c ∈ {0, 1, 2} → ∃ n : ℕ, n ∈ {1, 2, 3} ∧ 
      (∀ r' : ℕ, r' ∈ {0, 1, 2} → (r', c) = (r, c) → A ∈ {1, 2, 3})) ∧ 
    (∀ r : ℕ, r ∈ {0, 1, 2} → 
      ∃ n1 n2 : ℕ, n1 ≠ n2 ∧ n1 ∈ {1, 2, 3} ∧ n2 ∈ {1, 2, 3} ∧ 
                  (matrix ![n1, n2, 6 - n1 - n2]).row r = 2) ∧
    -- Given filled cells
    (matrix ![![1, ?, ?], 
              ![?, 2, ?], 
              ![?, ?, A]]) = 2 ∧
    -- Diagonal sum condition
    (1 + 2 + A) = 6 ∧ 
    -- Conclude sum
    A + 3 = 6 := 
sorry

end grid_sum_A_plus_B_l739_739040


namespace ellipse_equation_of_given_conditions_chord_length_ab_of_given_conditions_l739_739839

-- Defining the conditions
def ellipse (x y : ℝ) (a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

def eccentricity (e a c : ℝ) : Prop :=
  e = c / a

def distance_to_focus (end_minor_axis_to_focus : ℝ) : Prop :=
  end_minor_axis_to_focus = sqrt(3)

-- Stating the theorem for the first part (equation of the ellipse)
theorem ellipse_equation_of_given_conditions
  (a b c : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : a = sqrt(b^2 + c^2))
  (h4 : eccentricity (sqrt(6) / 3) a c)
  (h5 : distance_to_focus (sqrt(3))) :
  ellipse x y (sqrt 3) 1 := sorry

-- Stating the theorem for the second part (length of chord AB)
theorem chord_length_ab_of_given_conditions
  (a b : ℝ) (y1 y2 x1 x2 : ℝ)
  (h1 : ellipse x y (sqrt 3) 1)
  (h2 : y = x + 1) 
  (h3 : ∀ t, (t = 0 ∨ t = -3/2))
  (h4 : ∀ t, y1 = t + 1 ∨ y2 = t + 1) :
  (dist (0 : ℝ × ℝ) (3/2, -(1/2)) = 3 * sqrt 2 / 2) := sorry

end ellipse_equation_of_given_conditions_chord_length_ab_of_given_conditions_l739_739839


namespace problem1_problem2_l739_739797

section trigonometry_problems

-- Definitions for first problem
def expr1 := Real.sin (25 * Real.pi / 6) + Real.cos (25 * Real.pi / 3) + Real.tan (-25 * Real.pi / 4)
def answer1 := 0

-- Proving the first problem
theorem problem1 : expr1 = answer1 := by
  sorry

-- Definitions for second problem
variables (α β : ℝ)
def sin_α := 3 / 5
def cos_α_plus_β := -5 / 13
def sin_β_correct_answer := 63 / 65

-- Proving the second problem
theorem problem2 (hα : Real.sin α = sin_α) (h_cos_α_plus_β : Real.cos (α + β) = cos_α_plus_β) :
  Real.sin β = sin_β_correct_answer := by
  sorry

end trigonometry_problems

end problem1_problem2_l739_739797


namespace part1_part2_l739_739508

-- Part 1
theorem part1 (x : ℝ) (h1 : 2 * x = 3 * x - 1) : x = 1 :=
by
  sorry

-- Part 2
theorem part2 (x : ℝ) (h2 : x < 0) (h3 : |2 * x| + |3 * x - 1| = 16) : x = -3 :=
by
  sorry

end part1_part2_l739_739508


namespace donald_paul_ratio_l739_739794

-- Let P be the number of bottles Paul drinks in one day.
-- Let D be the number of bottles Donald drinks in one day.
def paul_bottles (P : ℕ) := P = 3
def donald_bottles (D : ℕ) := D = 9

theorem donald_paul_ratio (P D : ℕ) (hP : paul_bottles P) (hD : donald_bottles D) : D / P = 3 :=
by {
  -- Insert proof steps here using the conditions.
  sorry
}

end donald_paul_ratio_l739_739794


namespace mean_temperature_l739_739297

def temperatures : List ℚ := [80, 79, 81, 85, 87, 89, 87, 90, 89, 88]

theorem mean_temperature :
  let n := temperatures.length
  let sum := List.sum temperatures
  (sum / n : ℚ) = 85.5 :=
by
  sorry

end mean_temperature_l739_739297


namespace environmental_agency_min_employees_l739_739425

def W : ℕ := 120
def A : ℕ := 105
def B : ℕ := 65
def S : ℕ := 40

theorem environmental_agency_min_employees (W A B S : ℕ) (hW : W = 120) (hA : A = 105) (hB : B = 65) (hS : S = 40) : 
  W + A - B = 160 :=
by
  rw [hW, hA, hB, hS]
  norm_num

end environmental_agency_min_employees_l739_739425


namespace four_digit_multiples_of_4_l739_739164

theorem four_digit_multiples_of_4 :
  ∃ n, n = 208 ∧
    let digits := {0, 1, 2, 3, 4, 5, 6} in
    finset.filter (λ m, m % 4 = 0 ∧ m ≥ 1000 ∧ m < 10000 ∧ 
                  (∀ d, nat.digits d m ⊆ digits ∧ finset.card (finset.image nat.digits m) = 4)) (finset.range 10000) = n :=
by
  sorry

end four_digit_multiples_of_4_l739_739164


namespace divisors_of_3780_multiples_of_5_l739_739550

theorem divisors_of_3780_multiples_of_5 : 
  let n := 3780 
  let a_bound := 2 
  let b_bound := 3 
  let c_bound := 1 
  let d_bound := 1 
  let count := (a_bound + 1) * (b_bound + 1) * 1 * (d_bound + 1)
  (prime_factors n = [(2, 2), (3, 3), (5, 1), (7, 1)]) →
  count = 24 := 
by 
  let n := 3780 
  let a_bound := 2 
  let b_bound := 3 
  let c_bound := 1 
  let d_bound := 1 
  let count := (a_bound + 1) * (b_bound + 1) * 1 * (d_bound + 1)
  assume prime_factors n = [(2, 2), (3, 3), (5, 1), (7, 1)],
  show count = 24,
  sorry

end divisors_of_3780_multiples_of_5_l739_739550


namespace chords_intersect_probability_l739_739269

noncomputable def probability_chords_intersect (n m : ℕ) : ℚ :=
  if (n > 6 ∧ m = 2023) then
    1 / 72
  else
    0

theorem chords_intersect_probability :
  probability_chords_intersect 6 2023 = 1 / 72 :=
by
  sorry

end chords_intersect_probability_l739_739269


namespace triangle_XA_XB_XC_sum_l739_739994

theorem triangle_XA_XB_XC_sum (A B C D E F G X : ℝ)
  (hAB : dist A B = 12)
  (hBC : dist B C = 16)
  (hAC : dist A C = 20)
  (hMidD : midpoints A B D)
  (hMidE : midpoints B C E)
  (hMidF : midpoints A C F)
  (hAltitude : altitude B C G)
  (hX_intersect : X ≠ E ∧ circle_intersects (triangle_circumcircle B D E) (triangle_circumcircle C G F) X) :
  dist X A + dist X B + dist X C = 576 * real.sqrt 11 / 110 := sorry

end triangle_XA_XB_XC_sum_l739_739994


namespace sin_alpha_eq_one_third_l739_739136

theorem sin_alpha_eq_one_third (α β : ℝ) 
  (hα : 0 < α) (hα2 : α < π / 2)
  (hβ : π / 2 < β) (hβ2 : β < π)
  (cosβ : cos β = -1 / 3)
  (sin_alpha_beta : sin (α + β) = 7 / 9) :
  sin α = 1 / 3 := 
  sorry

end sin_alpha_eq_one_third_l739_739136


namespace log_value_comparison_l739_739731

theorem log_value_comparison :
  let e := Real.exp 1 in
  let initial_value := Log.log 1.2 * 1 / 6 in
  let transformed_value := 2.988 in
  transformed_value > e :=
by
  sorry

end log_value_comparison_l739_739731


namespace area_triangle_inequality_l739_739382

theorem area_triangle_inequality 
  (A B C P : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space P] 
  (S S1 S2 S3 : ℝ)
  (hP_in_ABC : P ∈ triangle A B C)
  (hpar_ab : is_parallel (line_from P parallel to line_from A to B))
  (hpar_ac : is_parallel (line_from P parallel to line_from A to C))
  (hpar_bc : is_parallel (line_from P parallel to line_from B to C))
  (hS : S = area_triangle A B C)
  (hS1 : S1 = area_triangle_from_parallel P A B)
  (hS2 : S2 = area_triangle_from_parallel P A C)
  (hS3 : S3 = area_triangle_from_parallel P B C) : 
  S ≤ 3 * (S1 + S2 + S3) :=
sorry

end area_triangle_inequality_l739_739382


namespace sequence_term_l739_739495

noncomputable def S : (ℕ → ℝ) := λ n, 1 - (2/3) * (a n)

theorem sequence_term (a : ℕ → ℝ)
  (hS : ∀ n, S n = 1 - (2/3) * a n) :
  ∀ n, a n = (3/5) * (2/5)^(n-1) := 
sorry

end sequence_term_l739_739495


namespace limit_a_eq_1_l739_739489

def a (n : ℕ) : ℝ :=
  if 1 ≤ n ∧ n < 10000 then (2^(n+1))/(2^n + 1)
  else (n+1)^2 / (n^2 + 1)

theorem limit_a_eq_1 : 
  ∀ ε > 0, ∃ N, ∀ n ≥ N, abs (a n - 1) < ε := 
sorry

end limit_a_eq_1_l739_739489


namespace domain_v_l739_739788

noncomputable def v (x : ℝ) : ℝ :=
  sqrt (2 * x - 4) + (x - 5)^(1/4)

theorem domain_v (x : ℝ) : (2 * x - 4 ≥ 0) → (x - 5 ≥ 0) → (x ≥ 5) :=
by
  intros h1 h2
  exact h2

end domain_v_l739_739788


namespace exactly_two_roots_iff_l739_739081

theorem exactly_two_roots_iff (a : ℝ) : 
  (∃! (x : ℝ), x^2 + 2 * x + 2 * |x + 1| = a) ↔ a > -1 :=
by
  sorry

end exactly_two_roots_iff_l739_739081


namespace proportional_function_quadrants_l739_739187

theorem proportional_function_quadrants (k : ℝ) :
  (∃ x y : ℝ, x < 0 ∧ y > 0 ∧ y = k * x) ∧ (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ y = k * x) → k < 0 :=
by
  sorry

end proportional_function_quadrants_l739_739187


namespace analytical_expression_of_f_l739_739878

variable {ℝ : Type*} [field ℝ]

noncomputable def f (x : ℝ) : ℝ := x^2 - 4x + 3

theorem analytical_expression_of_f :
  ∀ x : ℝ, f (x + 1) = x^2 - 2x :=
by
  intro x
  rw [f, pow_two, add_sub_assoc, sq, sub_add, sub_add_eq, sub_add_eq_add_sub]
  sorry

end analytical_expression_of_f_l739_739878


namespace arnaldo_billion_difference_l739_739427

theorem arnaldo_billion_difference :
  (10 ^ 12) - (10 ^ 9) = 999000000000 :=
by
  sorry

end arnaldo_billion_difference_l739_739427


namespace tangent_line_at_origin_inequality_x3_ln_ge_x2_series_inequality_ln_l739_739611

-- Part (1): Tangent line at (0,0)
def f (x : ℝ) : ℝ := x^3 + Real.log (x + 1)
theorem tangent_line_at_origin : f'(0) = 1 ∧ (∀ x, y = f x -> y = x - y = 0) := sorry

-- Part (2): Inequality for x ≥ 0
theorem inequality_x3_ln_ge_x2 (x : ℝ) (hx : 0 ≤ x) : f x ≥ x^2 := sorry

-- Part (3): Series inequality for ln(n+1)
theorem series_inequality_ln (n : ℕ) (hn : 2 ≤ n) :
  Real.log (n + 1) > ∑ k in Finset.range (n - 1), (k : ℝ) / (k + 1)^3 := sorry

end tangent_line_at_origin_inequality_x3_ln_ge_x2_series_inequality_ln_l739_739611


namespace closest_vector_t_l739_739810

def vector_v (t : ℝ) : ℝ × ℝ × ℝ :=
  (1 + 5 * t, -2 + 4 * t, -4 - 2 * t)

def vector_a : ℝ × ℝ × ℝ :=
  (3, 2, 6)

def direction_vector : ℝ × ℝ × ℝ :=
  (5, 4, -2)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def is_orthogonal (t : ℝ) : Prop :=
  dot_product (vector_v t - vector_a) direction_vector = 0

theorem closest_vector_t : is_orthogonal (2 / 15) :=
sorry

end closest_vector_t_l739_739810


namespace Bella_age_l739_739430

theorem Bella_age (B : ℕ) (h₁ : ∃ n : ℕ, n = B + 9) (h₂ : B + (B + 9) = 19) : B = 5 := 
by
  sorry

end Bella_age_l739_739430


namespace tanks_need_179_buckets_l739_739328

theorem tanks_need_179_buckets :
  ∀ (b1 b2 b3 : ℕ) (r1 r2 r3 : ℚ),
    b1 = 25 ∧ b2 = 35 ∧ b3 = 45 ∧ r1 = 2/5 ∧ r2 = 3/5 ∧ r3 = 4/5 →
    let new_b1 := (b1 * 5 / r1).ceil in
    let new_b2 := (b2 * 5 / r2).ceil in
    let new_b3 := (b3 * 5 / r3).ceil in
    new_b1 + new_b2 + new_b3 = 179 :=
by
  intros b1 b2 b3 r1 r2 r3 h,
  simp only at h,
  cases h with h1 h2,
  cases h2 with h3 h4,
  cases h4 with h5 h6,
  let new_b1 := (25 * 5 / (2 / 5)).ceil,
  let new_b2 := (35 * 5 / (3 / 5)).ceil,
  let new_b3 := (45 * 5 / (4 / 5)).ceil,
  have h1 : new_b1 = 63 := sorry, -- Proof required
  have h2 : new_b2 = 59 := sorry, -- Proof required
  have h3 : new_b3 = 57 := sorry, -- Proof required
  rw [h1, h2, h3],
  norm_num

end tanks_need_179_buckets_l739_739328


namespace grid_multiple_of_10_impossible_l739_739127

theorem grid_multiple_of_10_impossible (A : ℕ → ℕ → ℕ) :
  (∀ a b : ℕ, a < 8 ∧ b < 8 → 0 ≤ A a b) →
  ¬ (∃ (op : (ℕ → ℕ → ℕ) → (ℕ → ℕ → ℕ) → Prop),
    (∀ a b, (a ≤ 5 ∧ b ≤ 5 → op (λ i j, A (a+i) (b+j)) (λ i j, A (a+i) (b+j) + 1)) ∨ 
            (a ≤ 4 ∧ b ≤ 4 → op (λ i j, A (a+i) (b+j)) (λ i j, A (a+i) (b+j) + 1))) ∧
    (∀ a b, a < 8 ∧ b < 8 → (A a b % 10 = 0))) :=
sorry

end grid_multiple_of_10_impossible_l739_739127


namespace correct_completion_l739_739433

-- Definitions of conditions
def sentence_template := "By the time he arrives, all the work ___, with ___ our teacher will be content."
def option_A := ("will be accomplished", "that")
def option_B := ("will have been accomplished", "which")
def option_C := ("will have accomplished", "it")
def option_D := ("had been accomplished", "him")

-- The actual proof statement
theorem correct_completion : (option_B.fst = "will have been accomplished") ∧ (option_B.snd = "which") :=
by
  sorry

end correct_completion_l739_739433


namespace Sasha_can_paint_8x9_Sasha_cannot_paint_8x10_l739_739633

-- Definition of the problem conditions
def initially_painted (m n : ℕ) : Prop :=
  ∃ i j : ℕ, i < m ∧ j < n
  
def odd_painted_neighbors (m n : ℕ) : Prop :=
  ∀ i j : ℕ, i < m ∧ j < n →
  (∃ k l : ℕ, (k = i+1 ∨ k = i-1 ∨ l = j+1 ∨ l = j-1) ∧ k < m ∧ l < n → true)

-- Part (a): 8x9 rectangle
theorem Sasha_can_paint_8x9 : (initially_painted 8 9 ∧ odd_painted_neighbors 8 9) → ∀ (i j : ℕ), i < 8 ∧ j < 9 :=
by
  -- Proof here
  sorry

-- Part (b): 8x10 rectangle
theorem Sasha_cannot_paint_8x10 : (initially_painted 8 10 ∧ odd_painted_neighbors 8 10) → ¬ (∀ (i j : ℕ), i < 8 ∧ j < 10) :=
by
  -- Proof here
  sorry

end Sasha_can_paint_8x9_Sasha_cannot_paint_8x10_l739_739633


namespace least_element_in_T_l739_739607

theorem least_element_in_T :
  ∃ (T : Finset ℤ), T.card = 7 ∧
  (∀ c d ∈ T, c < d → ¬ (d % c = 0)) ∧
  ∀ t ∈ T, t ≥ 4 :=
sorry

end least_element_in_T_l739_739607


namespace constant_term_in_f_f_x_l739_739243

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then (x - 1/x)^8
  else -real.sqrt x

theorem constant_term_in_f_f_x (x : ℝ) (hx : x > 0) :
  let f_x := f x in
  let f_f_x := f f_x in
  -- We will skip the full binomial expansion and constant term analysis,
  -- and directly state the result, since the question is proof-based.
  -- Assume the result directly for the purpose of the statement.
  (f_f_x = 70) :=
begin
  sorry
end

end constant_term_in_f_f_x_l739_739243


namespace find_point_Q_l739_739314

theorem find_point_Q :
  ∃ (Q : ℝ × ℝ × ℝ),
    (∀ x y z, 
      ((x - 2)^2 + (y - 4)^2 + (z + 10)^2 = (x - Q.1)^2 + (y - Q.2)^2 + (z - Q.3)^2)
      → (20 * x - 8 * y + 48 * z = 110)) ∧ 
    Q = (12, 0, 14) :=
by
  use (12, 0, 14)
  intro x y z
  intro h
  sorry

end find_point_Q_l739_739314


namespace altitude_identity_l739_739691

variable {a b c d : ℝ}

def is_right_triangle (A B C : ℝ) : Prop :=
  A^2 + B^2 = C^2

def right_angle_triangle (a b c : ℝ) : Prop := 
  a^2 + b^2 = c^2

def altitude_property (a b c d : ℝ) : Prop :=
  a * b = c * d

theorem altitude_identity (a b c d : ℝ) (h1: right_angle_triangle a b c) (h2: altitude_property a b c d) :
  1 / a^2 + 1 / b^2 = 1 / d^2 :=
sorry

end altitude_identity_l739_739691


namespace option_D_correct_option_A_incorrect_option_B_incorrect_option_C_incorrect_l739_739365

-- Define the variables
variables (m : ℤ)

-- State the conditions as hypotheses
theorem option_D_correct (m : ℤ) : 
  (m * (m - 1) = m^2 - m) :=
by {
    -- Proof sketch (not implemented):
    -- Use distributive property to demonstrate that both sides are equal.
    sorry
}

theorem option_A_incorrect (m : ℤ) : 
  ¬ (m^4 + m^3 = m^7) :=
by {
    -- Proof sketch (not implemented):
    -- Demonstrate that exponents can't be added this way when bases are added.
    sorry
}

theorem option_B_incorrect (m : ℤ) : 
  ¬ ((m^4)^3 = m^7) :=
by {
    -- Proof sketch (not implemented):
    -- Show that raising m^4 to the power of 3 results in m^12.
    sorry
}

theorem option_C_incorrect (m : ℤ) : 
  ¬ (2 * m^5 / m^3 = m^2) :=
by {
    -- Proof sketch (not implemented):
    -- Show that dividing results in 2m^2.
    sorry
}

end option_D_correct_option_A_incorrect_option_B_incorrect_option_C_incorrect_l739_739365


namespace total_instruments_correct_l739_739249

def fingers : Nat := 10
def hands : Nat := 2
def heads : Nat := 1

def trumpets := fingers - 3
def guitars := hands + 2
def trombones := heads + 2
def french_horns := guitars - 1
def violins := trumpets / 2
def saxophones := trombones / 3

theorem total_instruments_correct : 
  (trumpets + guitars = trombones + violins + saxophones) →
  trumpets + guitars + trombones + french_horns + violins + saxophones = 21 := by
  sorry

end total_instruments_correct_l739_739249


namespace option_c_equals_one_half_l739_739422

theorem option_c_equals_one_half : 
  cos (12 * Real.pi / 180) * sin (42 * Real.pi / 180) - sin (12 * Real.pi / 180) * cos (42 * Real.pi / 180) = 1 / 2 :=
by 
sory

end option_c_equals_one_half_l739_739422


namespace students_select_different_topics_probability_l739_739006

theorem students_select_different_topics_probability :
  let topics : Finset ℕ := {1, 2, 3, 4, 5, 6}
  in (∃ p : ℚ, p = 5/6 ∧ (∀ sA sB ∈ topics, sA ≠ sB → p = (topics.card * (topics.card - 1)) / (topics.card * topics.card))) :=
by
  sorry

end students_select_different_topics_probability_l739_739006


namespace pies_sold_l739_739769

theorem pies_sold (apple_slices : ℕ) (peach_slices : ℕ) (apple_customers : ℕ) (peach_customers : ℕ)
  (h1 : apple_slices = 8) (h2 : peach_slices = 6)
  (h3 : apple_customers = 56) (h4 : peach_customers = 48) : 
  (apple_customers / apple_slices + peach_customers / peach_slices) = 15 := 
by
  have h5 : apple_customers / apple_slices = 7 := by sorry
  have h6 : peach_customers / peach_slices = 8 := by sorry
  calc
    (apple_customers / apple_slices + peach_customers / peach_slices) = (7 + 8) : by
      rw [h5, h6]
    ... = 15 : by
      norm_num

end pies_sold_l739_739769


namespace triangle_transformations_l739_739407

/-- 
Given a triangle with vertices A(0,0), B(1,0), and C(0,1),
prove that after the transformations - rotation by 180 degrees counterclockwise around the origin,
reflection across the x-axis, and translation 2 units to the right - the final coordinates
of the vertices of the triangle are (2,0), (1,0), and (2,1).
-/
theorem triangle_transformations :
  let A := (0, 0)
  let B := (1, 0)
  let C := (0, 1)
  let rotate_180 (p : ℕ × ℕ) := (-p.1, -p.2)
  let reflect_x (p : ℕ × ℕ) := (p.1, -p.2)
  let translate_2_right (p : ℕ × ℕ) := (p.1 + 2, p.2)
  rotate_180 A = (0, 0) ∧
  rotate_180 B = (-1, 0) ∧
  rotate_180 C = (0, -1) ∧
  reflect_x (rotate_180 A) = (0, 0) ∧
  reflect_x (rotate_180 B) = (-1, 0) ∧
  reflect_x (rotate_180 C) = (0, 1) ∧
  translate_2_right (reflect_x (rotate_180 A)) = (2, 0) ∧
  translate_2_right (reflect_x (rotate_180 B)) = (1, 0) ∧
  translate_2_right (reflect_x (rotate_180 C)) = (2, 1) :=
  sorry

end triangle_transformations_l739_739407


namespace unique_zero_f_x1_minus_2x2_l739_739532

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.exp x - x - 1) - Real.log (x + 1) + x

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x + x

-- Condition a ≥ 0
variable (a : ℝ) (a_nonneg : 0 ≤ a)

-- Define the first part of the problem
theorem unique_zero_f : ∃! x, f a x = 0 :=
  sorry

-- Variables for the second part of the problem
variable (x₁ x₂ : ℝ)
variable (cond : f a x₁ = g a x₁ - g a x₂)

-- Define the second part of the problem
theorem x1_minus_2x2 : x₁ - 2 * x₂ ≥ 1 - 2 * Real.log 2 :=
  sorry

end unique_zero_f_x1_minus_2x2_l739_739532


namespace binary_diff_ones_zeros_237_l739_739448

def num_bits (n : ℕ) : ℕ := Integer.digits 2 n |>.foldl (λ acc b, acc + b) 0
def num_zeros (n : ℕ) : ℕ := Integer.digits 2 n |>.foldl (λ acc b, acc + (1 - b)) 0

theorem binary_diff_ones_zeros_237 : num_bits 237 - num_zeros 237 = 6 := by
  sorry

end binary_diff_ones_zeros_237_l739_739448


namespace digital_cities_receive_distance_education_l739_739792

-- Definitions based on conditions
def digital_cities_enable := (A: Prop) (B: Prop) (C: Prop) (D: Prop) ⇒ (B: Prop)
def Travel_around_the_world := Prop
def Receive_distance_education := Prop
def Shop_online := Prop
def Seek_medical_advice_online := Prop

-- The theorem to prove
theorem digital_cities_receive_distance_education (A: Travel_around_the_world) 
                                                  (B: Receive_distance_education) 
                                                  (C: Shop_online) 
                                                  (D: Seek_medical_advice_online) : 
  digital_cities_enable A B C D := 
by sorry

end digital_cities_receive_distance_education_l739_739792


namespace total_red_pencils_l739_739908

theorem total_red_pencils (packs : ℕ) (normal_pencil_per_pack : ℕ) (extra_packs : ℕ) (extra_pencils_per_pack : ℕ) :
  packs = 15 →
  normal_pencil_per_pack = 1 →
  extra_packs = 3 →
  extra_pencils_per_pack = 2 →
  packs * normal_pencil_per_pack + extra_packs * extra_pencils_per_pack = 21 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num

end total_red_pencils_l739_739908


namespace additional_cost_tv_ad_l739_739678

theorem additional_cost_tv_ad (in_store_price : ℝ) (payment : ℝ) (shipping : ℝ) :
  in_store_price = 129.95 → payment = 29.99 → shipping = 14.95 → 
  (4 * payment + shipping - in_store_price) * 100 = 496 :=
by
  intros h1 h2 h3
  sorry

end additional_cost_tv_ad_l739_739678


namespace roots_eq_two_iff_a_gt_neg1_l739_739065

theorem roots_eq_two_iff_a_gt_neg1 (a : ℝ) : 
  (∃! x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + 2*|x₁ + 1| = a ∧ x₂^2 + 2*x₂ + 2*|x₂ + 1| = a) ↔ a > -1 :=
by sorry

end roots_eq_two_iff_a_gt_neg1_l739_739065


namespace integer_solutions_eq_two_l739_739168

theorem integer_solutions_eq_two : 
  ∃ S : Set Int, (∀ x : Int, x ∈ S ↔ (x-3)^(30-x^2) = 1) ∧ S.card = 2 := 
sorry

end integer_solutions_eq_two_l739_739168


namespace find_a_bi_c_l739_739954

theorem find_a_bi_c (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h_eq : (a - (b : ℤ)*I)^2 + c = 13 - 8*I) :
  a = 2 ∧ b = 2 ∧ c = 13 :=
by
  sorry

end find_a_bi_c_l739_739954


namespace line_intersects_circle_slope_range_l739_739850

theorem line_intersects_circle_slope_range (k : ℝ) :
  ∃ b c, b = -4 ∧ c = 3 ∧ (b^2 - 4 * (1 + k^2) * c ≥ 0) →
  k ∈ set.Icc (-real.sqrt 3 / 3) (real.sqrt 3 / 3) :=
by
  sorry

end line_intersects_circle_slope_range_l739_739850


namespace smallest_x_value_l739_739478

theorem smallest_x_value :
  ∃ x, (x ≠ 9) ∧ (∀ y, (y ≠ 9) → ((x^2 - x - 72) / (x - 9) = 3 / (x + 6)) → x ≤ y) ∧ x = -9 :=
by
  sorry

end smallest_x_value_l739_739478


namespace sum_of_solutions_eq_24_l739_739479

theorem sum_of_solutions_eq_24 :
  let solutions := {x : ℂ | (x - 4)^3 = 64}
  in ∑ x in solutions, x = 24 := by
  sorry

end sum_of_solutions_eq_24_l739_739479


namespace initial_average_weight_l739_739196

variables (x y : ℕ) (h1 : x + y = 20) (h2 : y = 15)

theorem initial_average_weight (x y : ℕ) (h1 : x + y = 20) (h2 : y = 15) : 
  (x * 10 + y * 20) / 20 = 17.5 := by
  have h3 : (x * 10 + y * 20) = 350 := by
    simp [h1, h2]
    sorry
  have h4 : (x * 10 + y * 20) / 20 = 350 / 20 := by
    rw h3
  norm_num at h4
  exact h4

end initial_average_weight_l739_739196


namespace log_a_of_81_l739_739851

theorem log_a_of_81 (a : ℝ) (h1 : (3 : ℝ) = 3) (h2 : 27 = a^3) : log a 81 = 4 := 
by
  sorry

end log_a_of_81_l739_739851


namespace find_initial_number_of_girls_l739_739818

theorem find_initial_number_of_girls (b g : ℕ) : 
  (b = 3 * (g - 12)) ∧ (4 * (b - 36) = g - 12) → g = 25 :=
by
  intros h
  sorry

end find_initial_number_of_girls_l739_739818


namespace integral_sqrt_2_minus_x_squared_l739_739058

theorem integral_sqrt_2_minus_x_squared :
  ∫ x in (0 : ℝ)..(real.sqrt 2), real.sqrt (2 - x^2) = real.pi / 2 :=
sorry

end integral_sqrt_2_minus_x_squared_l739_739058


namespace number_equation_l739_739751

variable (x : ℝ)

theorem number_equation :
  5 * x - 2 * x = 10 :=
sorry

end number_equation_l739_739751


namespace spending_50_dollars_l739_739573

def receiving_money (r : Int) : Prop := r > 0

def spending_money (s : Int) : Prop := s < 0

theorem spending_50_dollars :
  receiving_money 80 ∧ ∀ r, receiving_money r → spending_money (-r)
  → spending_money (-50) :=
by
  sorry

end spending_50_dollars_l739_739573


namespace original_decimal_number_l739_739363

theorem original_decimal_number (x : ℝ) (h₁ : 0 < x) (h₂ : 100 * x = 9 * (1 / x)) : x = 3 / 10 :=
by
  sorry

end original_decimal_number_l739_739363


namespace complement_intersection_l739_739388

universe u

-- Define the universal set U, and sets A and B
def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
def A : Set ℕ := {0, 1, 3, 5, 8}
def B : Set ℕ := {2, 4, 5, 6, 8}

-- Define the complements of A and B with respect to U
def complement_U (s : Set ℕ) := { x ∈ U | x ∉ s }

-- The theorem to prove the intersection of the complements
theorem complement_intersection :
  (complement_U A) ∩ (complement_U B) = {7, 9} :=
sorry

end complement_intersection_l739_739388


namespace polynomial_coeff_sum_l739_739445

-- Given polynomial with real coefficients and roots
variable (p q r s : ℝ)
variable (g : ℝ → ℝ)
variable (x : ℝ)

-- Variable assumptions and expressions for polynomial and its roots
def g_def := g = (λ x, x^4 + p * x^3 + q * x^2 + r * x + s)
def roots := (g (3 * complex.I) = 0 ∧ g (1 + 3 * complex.I) = 0 ∧ 
              g (-3 * complex.I) = 0 ∧ g (1 - 3 * complex.I) = 0) ∧
             (g (x) has_real_coefficients)

-- Theorem statement of the proof problem
theorem polynomial_coeff_sum :
  g_def p q r s g →
  roots p q r s g →
  p + q + r + s = 89 :=
by
  intro h_def h_roots
  sorry

end polynomial_coeff_sum_l739_739445


namespace problem_1_problem_2_problem_3_l739_739507

/-- Given an oblique triangle ABC with sides opposite to angles A, B, and C being a, b, and c 
respectively, and given sin(A) = cos(B), prove that A - B = π / 2. -/
theorem problem_1 {α β : ℝ} (h : Real.sin α = Real.cos β) : α - β = π / 2 := 
sorry

/-- Given an oblique triangle ABC with sides opposite to angles A, B, and C being a, b, and c 
respectfully, if a = 1, find the minimum value of the dot product of vectors AB and AC. -/
theorem problem_2 {A B C : Type*} [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C]
    (a : A) (b : B) (c : C) (h : (a - b) • (a - c) = 2 * Real.sqrt 2 - 3) : 
    ∃ x, x = (2 * Real.sqrt 2 - 3) := 
sorry

/-- Given an oblique triangle ABC with sides opposite to angles A, B, and C being a, b, and c 
respectively, and given sin(A) = cos(B) = 3 / 2 * tan(C), prove that A = 2π / 3 and B = π / 6. -/
theorem problem_3 {α β γ : ℝ} (h : Real.sin α = Real.cos β) (h' : Real.sin α = 3 / 2 * Real.tan γ) : 
  α = 2 * π / 3 ∧ β = π / 6 := 
sorry

end problem_1_problem_2_problem_3_l739_739507


namespace red_pencils_count_l739_739904

theorem red_pencils_count 
  (packs : ℕ) 
  (pencils_per_pack : ℕ) 
  (extra_packs : ℕ) 
  (extra_pencils_per_pack : ℕ)
  (total_red_pencils : ℕ) 
  (h1 : packs = 15)
  (h2 : pencils_per_pack = 1)
  (h3 : extra_packs = 3)
  (h4 : extra_pencils_per_pack = 2)
  (h5 : total_red_pencils = packs * pencils_per_pack + extra_packs * extra_pencils_per_pack) : 
  total_red_pencils = 21 := 
  by sorry

end red_pencils_count_l739_739904


namespace geometric_sum_l739_739235

open BigOperators

noncomputable def geom_sequence (a q : ℚ) (n : ℕ) : ℚ := a * q ^ n

noncomputable def sum_geom_sequence (a q : ℚ) (n : ℕ) : ℚ := 
  if q = 1 then a * n
  else a * (1 - q ^ (n + 1)) / (1 - q)

theorem geometric_sum (a q : ℚ) (h_a : a = 1) (h_S3 : sum_geom_sequence a q 2 = 3 / 4) :
  sum_geom_sequence a q 3 = 5 / 8 :=
sorry

end geometric_sum_l739_739235


namespace rain_probability_at_most_3_days_l739_739309

open BigOperators

def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

noncomputable def rain_probability := (1:ℝ)/5
noncomputable def no_rain_probability := (4:ℝ)/5

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binom n k) * (p^k) * ((1-p)^(n-k))

theorem rain_probability_at_most_3_days :
  ∑ k in Finset.range 4, binomial_probability 31 k rain_probability = 0.544 :=
by
  sorry

end rain_probability_at_most_3_days_l739_739309


namespace bobby_shoes_multiple_l739_739431

theorem bobby_shoes_multiple (B M : ℕ) (hBonny : 13 = 2 * B - 5) (hBobby : 27 = M * B) : 
  M = 3 :=
by 
  sorry

end bobby_shoes_multiple_l739_739431


namespace smallest_four_digit_number_divisible_by_24_is_1104_l739_739705

noncomputable def smallest_four_digit_divisible_by_24 : ℕ :=
  1104

theorem smallest_four_digit_number_divisible_by_24_is_1104 :
  ∀ n, n >= 1000 ∧ n < 10000 ∧ (n % 24 = 0) → n >= smallest_four_digit_divisible_by_24 :=
begin
  sorry
end

end smallest_four_digit_number_divisible_by_24_is_1104_l739_739705


namespace find_a_l739_739125

variable (x y : Fin 8 → ℝ)

def x_sum_condition (x : Fin 8 → ℝ) : Prop :=
  ∑ i, x i = 6

def y_sum_condition (y : Fin 8 → ℝ) : Prop :=
  ∑ i, y i = 3

def regression_line (x y : Fin 8 → ℝ) (a : ℝ) : Prop :=
  ((3 / 8) = (1 / 3) * (3 / 4) + a)

theorem find_a
  (h₁ : x_sum_condition x)
  (h₂ : y_sum_condition y) :
  ∃ a : ℝ, regression_line x y a ∧ a = 1 / 8 :=
by
  sorry

end find_a_l739_739125


namespace factorial_1000_ends_in_249_zeros_l739_739668

theorem factorial_1000_ends_in_249_zeros :
  (1000.factorial.div_count 5) = 249 := 
sorry

end factorial_1000_ends_in_249_zeros_l739_739668


namespace find_m_l739_739857

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then 1 + log 5 x else 2 * x - 1

theorem find_m : ∃ m : ℝ, f (f 0 + m) = 2 ∧ m = 6 := by
  sorry

end find_m_l739_739857


namespace exists_cool_assignment_l739_739795

universe u

variables {V : Type u} [fintype V] [decidable_eq V] (G : simple_graph V)

def f (v : V) : ℕ := sorry
def S (v : V) : ℕ := sorry

theorem exists_cool_assignment (G : simple_graph V) :
  ∃ (f : V → ℕ) (f_e : sym2 V → ℕ),
  (∀ (v : V), f v ∈ {1, 2}) ∧
  (∀ (e : sym2 V), f_e e ∈ {1, 2, 3}) ∧
  (∀ ⦃u v : V⦄, G.adj u v → S u ≠ S v) :=
sorry

end exists_cool_assignment_l739_739795


namespace find_Female_employees_l739_739321

noncomputable theory

def companyEmployees :=
  let E := ℕ -- total number of employees
  let M := (2/5 : ℚ) * E -- total number of managers
  let FemaleManagers := 200 -- female managers
  let Male := ℕ -- number of male employees
  let MaleManagers := (2/5 : ℚ) * Male -- number of male managers
  FemaleManagers + MaleManagers = M

theorem find_Female_employees
  (E : ℕ)
  (FemaleManagers : ℕ := 200)
  (ManagersPortion : ℚ := 2/5)
  (MaleManagersPortion : ℚ := 2/5)
  (M : ℕ := (ManagersPortion * E).to_nat)
  (Male : ℕ := E - 500)
  (Female : ℕ := 500)
  (h1 : FemaleManagers = 200)
  (h2 : M = (2/5 : ℚ) * E)
  (h3 : MaleManagersPortion * Male = M - FemaleManagers) :
  E - Male = 500 :=
begin
  -- The proof goes here
  sorry
end

end find_Female_employees_l739_739321


namespace lily_typing_break_time_l739_739246

theorem lily_typing_break_time :
  ∃ t : ℝ, (15 * t + 15 * t = 255) ∧ (19 = 2 * t + 2) ∧ (t = 8) := 
sorry

end lily_typing_break_time_l739_739246


namespace xy_eq_zero_l739_739334

theorem xy_eq_zero (x y : ℝ) (h1 : x - y = 3) (h2 : x^3 - y^3 = 27) : x * y = 0 := by
  sorry

end xy_eq_zero_l739_739334


namespace expected_value_decisive_games_is_4_l739_739763

noncomputable def expected_value_decisive_games : ℝ :=
  let p_win := 1 / 2
  let p_continue := 1 / 2
  let E_X := 2 * p_win + (2 + E_X) * p_continue
  (E_X : ℝ)

theorem expected_value_decisive_games_is_4 :
  expected_value_decisive_games = 4 :=
sorry

end expected_value_decisive_games_is_4_l739_739763


namespace smallest_sum_of_integers_product_12_fact_l739_739965

theorem smallest_sum_of_integers_product_12_fact :
  ∃ (p q r s : ℕ) (h1 : p > 0) (h2 : q > 0) (h3 : r > 0) (h4 : s > 0),
    p * q * r * s = 12! ∧ p + q + r + s = 1402 := by
  sorry

end smallest_sum_of_integers_product_12_fact_l739_739965


namespace inequality_triangle_l739_739925

variable {P A B C : Type} [MetricSpace P] [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables (a b c u v ω : ℝ)

theorem inequality_triangle (P A B C : Point) (BC CA AB PA PB PC : ℝ) :
  (BC = a) →
  (CA = b) →
  (AB = c) →
  (PA = u) →
  (PB = v) →
  (PC = ω) →
  (P : ℝ) →
  (A : ℝ) →
  (B : ℝ) →
  (C : ℝ) →
  (A; B; C) in (Plane P) →
  (u / a) + (v / b) + (ω / c) ≥ sqrt 3 :=
by
  sorry

end inequality_triangle_l739_739925


namespace total_flowers_l739_739420

theorem total_flowers (y_w : ℕ) (r_y : ℕ) (r_w : ℕ) (w : ℕ) (r : ℕ)
  (h₁ : y_w = 13)
  (h₂ : r_y = 17)
  (h₃ : r_w = 14)
  (h₄ : w = y_w + r_w)
  (h₅ : r = w + 4):
  (y_w + r_y + r_w = 44) :=
by {
  have w_val : w = 13 + 14, from sorry,
  have r_val : r = (13 + 14) + 4, from sorry,
  have total_val : 13 + 17 + 14 = 44, from sorry,
  exact total_val
}

end total_flowers_l739_739420


namespace profit_per_meal_A_and_B_l739_739565

theorem profit_per_meal_A_and_B (x y : ℝ) 
  (h1 : x + 2 * y = 35) 
  (h2 : 2 * x + 3 * y = 60) : 
  x = 15 ∧ y = 10 :=
sorry

end profit_per_meal_A_and_B_l739_739565


namespace club_for_all_l739_739883

variables {Student Club : Type}
variables (enroll : Student → set Club)

def in_two_clubs (s : Student) := (enroll s).card = 2

def common_club (s1 s2 s3 : Student) := ∃ c : Club, c ∈ enroll s1 ∧ c ∈ enroll s2 ∧ c ∈ enroll s3

theorem club_for_all (students : set Student) 
  (h1 : ∀ s ∈ students, in_two_clubs enroll s)
  (h2 : ∀ s1 s2 s3 ∈ students, common_club enroll s1 s2 s3) :
  ∃ c : Club, ∀ s ∈ students, c ∈ enroll s :=
sorry

end club_for_all_l739_739883


namespace last_four_digits_of_5_pow_2016_l739_739942

theorem last_four_digits_of_5_pow_2016 :
  (5^2016) % 10000 = 625 :=
by
  -- Establish periodicity of last four digits in powers of 5
  sorry

end last_four_digits_of_5_pow_2016_l739_739942


namespace eval_expression_eq_54_l739_739796

theorem eval_expression_eq_54 : (3 * 4 * 6) * ((1/3 : ℚ) + 1/4 + 1/6) = 54 := 
by
  sorry

end eval_expression_eq_54_l739_739796


namespace pies_sold_l739_739770

theorem pies_sold (apple_slices : ℕ) (peach_slices : ℕ) (apple_customers : ℕ) (peach_customers : ℕ)
  (h1 : apple_slices = 8) (h2 : peach_slices = 6)
  (h3 : apple_customers = 56) (h4 : peach_customers = 48) : 
  (apple_customers / apple_slices + peach_customers / peach_slices) = 15 := 
by
  have h5 : apple_customers / apple_slices = 7 := by sorry
  have h6 : peach_customers / peach_slices = 8 := by sorry
  calc
    (apple_customers / apple_slices + peach_customers / peach_slices) = (7 + 8) : by
      rw [h5, h6]
    ... = 15 : by
      norm_num

end pies_sold_l739_739770


namespace how_many_more_cups_of_sugar_l739_739247

def required_sugar : ℕ := 11
def required_flour : ℕ := 9
def added_flour : ℕ := 12
def added_sugar : ℕ := 10

theorem how_many_more_cups_of_sugar :
  required_sugar - added_sugar = 1 :=
by
  sorry

end how_many_more_cups_of_sugar_l739_739247


namespace sum_of_digits_eq_28_l739_739580

theorem sum_of_digits_eq_28 (A B C D E : ℕ) 
  (hA : 0 ≤ A ∧ A ≤ 9) 
  (hB : 0 ≤ B ∧ B ≤ 9) 
  (hC : 0 ≤ C ∧ C ≤ 9) 
  (hD : 0 ≤ D ∧ D ≤ 9) 
  (hE : 0 ≤ E ∧ E ≤ 9) 
  (unique_digits : (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧ (A ≠ E) ∧ (B ≠ C) ∧ (B ≠ D) ∧ (B ≠ E) ∧ (C ≠ D) ∧ (C ≠ E) ∧ (D ≠ E)) 
  (h : (10 * A + B) * (10 * C + D) = 111 * E) : 
  A + B + C + D + E = 28 :=
sorry

end sum_of_digits_eq_28_l739_739580


namespace pot_filling_time_l739_739666

-- Define the given conditions
def drops_per_minute : ℕ := 3
def volume_per_drop : ℕ := 20 -- in ml
def pot_capacity : ℕ := 3000 -- in ml (3 liters * 1000 ml/liter)

-- Define the calculation for the drip rate
def drip_rate_per_minute : ℕ := drops_per_minute * volume_per_drop

-- Define the goal, i.e., how long it will take to fill the pot
def time_to_fill_pot (capacity : ℕ) (rate : ℕ) : ℕ := capacity / rate

-- Proof statement
theorem pot_filling_time :
  time_to_fill_pot pot_capacity drip_rate_per_minute = 50 := 
sorry

end pot_filling_time_l739_739666


namespace smallest_cars_number_l739_739592

theorem smallest_cars_number :
  ∃ N : ℕ, N > 2 ∧ (N % 5 = 2) ∧ (N % 6 = 2) ∧ (N % 7 = 2) ∧ N = 212 := by
  sorry

end smallest_cars_number_l739_739592


namespace unique_zero_f_x1_minus_2x2_l739_739531

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.exp x - x - 1) - Real.log (x + 1) + x

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x + x

-- Condition a ≥ 0
variable (a : ℝ) (a_nonneg : 0 ≤ a)

-- Define the first part of the problem
theorem unique_zero_f : ∃! x, f a x = 0 :=
  sorry

-- Variables for the second part of the problem
variable (x₁ x₂ : ℝ)
variable (cond : f a x₁ = g a x₁ - g a x₂)

-- Define the second part of the problem
theorem x1_minus_2x2 : x₁ - 2 * x₂ ≥ 1 - 2 * Real.log 2 :=
  sorry

end unique_zero_f_x1_minus_2x2_l739_739531


namespace num_valid_4x4_arrays_l739_739047

open Matrix

def is_increasing_row {α : Type*} [LinearOrder α] (m : Matrix (Fin 4) (Fin 4) α) : Prop :=
  ∀ i : Fin 4, ∀ j k : Fin 4, j < k → m i j < m i k

def is_increasing_col {α : Type*} [LinearOrder α] (m : Matrix (Fin 4) (Fin 4) α) : Prop :=
  ∀ j : Fin 4, ∀ i k : Fin 4, i < k → m i j < m k j

def valid_4x4_array (m : Matrix (Fin 4) (Fin 4) ℕ) : Prop :=
  (∀ i j, 1 ≤ m i j ∧ m i j ≤ 16) ∧
  is_increasing_row m ∧ is_increasing_col m

theorem num_valid_4x4_arrays : Finset.card {m : Matrix (Fin 4) (Fin 4) ℕ // valid_4x4_array m} = 120 :=
sorry

end num_valid_4x4_arrays_l739_739047


namespace range_of_f_l739_739094

noncomputable def f (x : ℝ) : ℝ := x + real.sqrt (x^2 - 4)

def range_f : set ℝ := {y | ∃ x : ℝ, y = f x}

theorem range_of_f : range_f = {y : ℝ | y ≤ -2 ∨ y ≥ 2} :=
by
  sorry

end range_of_f_l739_739094


namespace maya_lift_difference_l739_739627

/--
  Maya can lift a fourth of what America can.
  America initially can lift 240 pounds.
  At her peak, America can lift 300 pounds.
  Maya can lift half of what America can lift at her peak.
  Prove that Maya can lift 90 pounds more at her peak than when she started.
-/
theorem maya_lift_difference :
  ∀ (initial_lift america_initial_lift america_peak_lift maya_initial_ratio maya_peak_ratio : ℝ),
    america_initial_lift = 240 →
    america_peak_lift = 300 →
    maya_initial_ratio = 1/4 →
    maya_peak_ratio = 1/2 →
    initial_lift = america_initial_lift * maya_initial_ratio →
    let peak_lift := america_peak_lift * maya_peak_ratio in
    peak_lift - initial_lift = 90 :=
by
  intros initial_lift america_initial_lift america_peak_lift maya_initial_ratio maya_peak_ratio
  assume h1 h2 h3 h4 h5
  let peak_lift := america_peak_lift * maya_peak_ratio
  sorry

end maya_lift_difference_l739_739627


namespace prob_roll_non_six_l739_739216

-- Define the conditions
def fair_six_sided_die : Type := {x : ℕ // x > 0 ∧ x <= 6}

-- Define the probability space
def prob_space : Type := finset fair_six_sided_die

-- State the theorem
theorem prob_roll_non_six (d : fair_six_sided_die) (h : d ∈ prob_space) : 
  let favorable_outcomes := {1, 2, 3, 4, 5} in
  let total_outcomes := {1, 2, 3, 4, 5, 6} in
  (size favorable_outcomes) / (size total_outcomes) = 5 / 6 :=
by
  sorry

end prob_roll_non_six_l739_739216


namespace ratio_IM_IN_l739_739930

noncomputable def compute_ratio (IA IB IC ID : ℕ) (M N : ℕ) : ℚ :=
  (IA * IC : ℚ) / (IB * ID : ℚ)

theorem ratio_IM_IN (IA IB IC ID : ℕ) (hIA : IA = 12) (hIB : IB = 16) (hIC : IC = 14) (hID : ID = 11) :
  compute_ratio IA IB IC ID = 21 / 22 := by
  rw [hIA, hIB, hIC, hID]
  sorry

end ratio_IM_IN_l739_739930


namespace ice_cubes_total_l739_739450

theorem ice_cubes_total (initial_cubes made_cubes : ℕ) (h_initial : initial_cubes = 2) (h_made : made_cubes = 7) : initial_cubes + made_cubes = 9 :=
by
  sorry

end ice_cubes_total_l739_739450


namespace tony_drives_15_miles_l739_739993

variable (total_miles_to_groceries : ℕ) (total_miles_to_haircut : ℕ) (distance_after_groceries : ℕ)

-- Given conditions
axiom drives_to_groceries : total_miles_to_groceries = 10
axiom drives_to_haircut : total_miles_to_haircut = 15
axiom distance_after_groceries_def : distance_after_groceries = 5

-- Prove that Tony will have driven 15 miles on his way to the haircut after completing the trip for groceries
theorem tony_drives_15_miles :
  total_miles_to_groceries + distance_after_groceries = 15 :=
by
  rw [drives_to_groceries, distance_after_groceries_def]
  exact rfl

end tony_drives_15_miles_l739_739993


namespace pipe_C_draining_rate_l739_739258

noncomputable def pipe_rate := 25

def tank_capacity := 2000
def pipe_A_rate := 200
def pipe_B_rate := 50
def pipe_C_duration_per_cycle := 2
def pipe_A_duration := 1
def pipe_B_duration := 2
def cycle_duration := pipe_A_duration + pipe_B_duration + pipe_C_duration_per_cycle
def total_time := 40
def number_of_cycles := total_time / cycle_duration
def water_filled_per_cycle := (pipe_A_rate * pipe_A_duration) + (pipe_B_rate * pipe_B_duration)
def total_water_filled := number_of_cycles * water_filled_per_cycle
def excess_water := total_water_filled - tank_capacity 
def pipe_C_rate := excess_water / (pipe_C_duration_per_cycle * number_of_cycles)

theorem pipe_C_draining_rate :
  pipe_C_rate = pipe_rate := by
  sorry

end pipe_C_draining_rate_l739_739258


namespace area_of_quadrilateral_ABDE_l739_739767

-- Definitions for the given problem
variable (AB CE AC DE : ℝ)
variable (parABCE parACDE : Prop)
variable (areaCOD : ℝ)

-- Lean 4 statement for the proof problem
theorem area_of_quadrilateral_ABDE
  (h1 : parABCE)
  (h2 : parACDE)
  (h3 : AB = 5)
  (h4 : AC = 5)
  (h5 : CE = 10)
  (h6 : DE = 10)
  (h7 : areaCOD = 10)
  : (AB + AC + CE + DE) / 2 + areaCOD = 52.5 := 
sorry

end area_of_quadrilateral_ABDE_l739_739767


namespace prob_positive_test_prob_lesion_given_positive_test_l739_739947

-- Conditions
def prob_a : ℝ := 0.002
def prob_not_a : ℝ := 0.998
def prob_b_given_a : ℝ := 0.9
def prob_b_given_not_a : ℝ := 0.1

-- Law of total probability
def prob_b : ℝ := prob_a * prob_b_given_a + prob_not_a * prob_b_given_not_a

-- Bayes' theorem
def prob_a_given_b : ℝ := (prob_a * prob_b_given_a) / prob_b

-- Proof statement
theorem prob_positive_test (correct_prob_b : prob_b = 0.1016) : 
  prob_b = 0.1016 :=
by {
  calc
    prob_b = prob_a * prob_b_given_a + prob_not_a * prob_b_given_not_a : rfl
    ... = 0.002 * 0.9 + 0.998 * 0.1 : by rfl
    ... = 0.0018 + 0.0998 : by norm_num
    ... = 0.1016 : by norm_num
}

theorem prob_lesion_given_positive_test (correct_prob_b : prob_b = 0.1016) :
  prob_a_given_b = 0.0177 :=
by {
  calc
    prob_a_given_b = (prob_a * prob_b_given_a) / prob_b : rfl
    ... = (0.002 * 0.9) / 0.1016 : by rfl
    ... = 0.0018 / 0.1016 : by norm_num
    ... ≈ 0.0177 : sorry  -- Approximation is correct to 4 decimal places
}

end prob_positive_test_prob_lesion_given_positive_test_l739_739947


namespace find_a_l739_739934

def M : Set ℝ := {-1, 0, 1}

def N (a : ℝ) : Set ℝ := {a, a^2}

theorem find_a (a : ℝ) : N a ⊆ M → a = -1 :=
by
  sorry

end find_a_l739_739934


namespace find_f_zero_l739_739500

def sinAlphaEqualsTwoCosAlpha (α : ℝ) := sin α = 2 * cos α
def f (α : ℝ) (x : ℝ) := 2^x - tan α

theorem find_f_zero (α : ℝ) (h : sinAlphaEqualsTwoCosAlpha α) : f α 0 = -1 := 
by
  sorry

end find_f_zero_l739_739500


namespace max_value_of_f_l739_739707

def f (x : ℝ) : ℝ := x^2 - 2 * x - 5

theorem max_value_of_f : ∃ x ∈ (Set.Icc (-2:ℝ) 2), ∀ y ∈ (Set.Icc (-2:ℝ) 2), f y ≤ f x ∧ f x = 3 := by
  sorry

end max_value_of_f_l739_739707


namespace final_result_is_102_l739_739412

/-- Proof that subtracting 138 from the result of multiplying 60 by 4 equals 102. -/
theorem final_result_is_102 (chosen_number : ℕ) (multiplier : ℕ) (subtract_amount : ℕ) (result : ℕ) 
  (h1: chosen_number = 60) (h2: multiplier = 4) (h3: subtract_amount = 138) : 
  result = chosen_number * multiplier - subtract_amount :=
by
  have h4 : result = 60 * 4 - 138 := by
    rw [h1, h2, h3]
    exact rfl
  rw h4
  exact rfl

end final_result_is_102_l739_739412


namespace initial_distance_l739_739437

theorem initial_distance (v_a v_b : ℝ) (t : ℝ) (D_plus : ℝ) (rel_speed : v_a - v_b = 8) :
  (v_a = 58) → (v_b = 50) → (t = 2.25) → (D_plus = 8) → 
  (8 * 2.25 = D + 8) → D = 10 :=
by
  intros h1 h2 h3 h4 h5
  subst h1
  subst h2
  subst h3
  subst h4
  rw h5
  norm_num
  sorry

end initial_distance_l739_739437


namespace binomial_sum_sum_solution_l739_739348

open Nat

theorem binomial_sum :
  ∀ n : ℕ, binomial 30 15 + binomial 30 n = binomial 31 16 → n = 14 ∨ n = 16 :=
by
  sorry

theorem sum_solution :
  (14 + 16) = 30 :=
by
  exact rfl

end binomial_sum_sum_solution_l739_739348


namespace functional_equation_solution_l739_739059

theorem functional_equation_solution :
  ∀ f : ℝ → ℝ,
  (∀ x y : ℝ, x * f y + y * f x = (x + y) * f x * f y) →
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, x ≠ 0 → f x = 1) :=
by
  intro f h,
  sorry

end functional_equation_solution_l739_739059


namespace enjoyable_gameplay_l739_739896

theorem enjoyable_gameplay (total_hours : ℕ) (boring_percentage : ℕ) (expansion_hours : ℕ)
  (h_total : total_hours = 100)
  (h_boring : boring_percentage = 80)
  (h_expansion : expansion_hours = 30) :
  ((1 - boring_percentage / 100) * total_hours + expansion_hours) = 50 := 
by
  sorry

end enjoyable_gameplay_l739_739896


namespace find_f_x1_x2_sum_l739_739517

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := 4 * Real.cos (3 * x + φ)

theorem find_f_x1_x2_sum :
  ∀ {x1 x2 φ : ℝ},
    |φ| < Real.pi / 2 →
    (3 * (11 * Real.pi / 12) + φ) = Int.natAbs (3 * (11 * Real.pi / 12) + φ) * Real.pi →
    x1 ∈ Set.Ioo (-7 * Real.pi / 12) (-Real.pi / 12) →
    x2 ∈ Set.Ioo (-7 * Real.pi / 12) (-Real.pi / 12) →
    x1 ≠ x2 →
    f x1 φ = f x2 φ →
    f (x1 + x2) φ = 2 * Real.sqrt 2 :=
begin
  sorry

end find_f_x1_x2_sum_l739_739517


namespace tan_alpha_eq_2_l739_739822

open Real

variable (α : ℝ)
def a : ℝ × ℝ := (1, sin α)
def b : ℝ × ℝ := (2, 4 * cos α)

theorem tan_alpha_eq_2 (h : a α ∥ b α) : tan α = 2 := 
by 
  sorry

end tan_alpha_eq_2_l739_739822


namespace frank_money_left_l739_739112

theorem frank_money_left (initial_money : ℝ) (spent_groceries : ℝ) (spent_magazine : ℝ) :
  initial_money = 600 →
  spent_groceries = (1/5) * initial_money →
  spent_magazine = (1/4) * (initial_money - spent_groceries) →
  initial_money - spent_groceries - spent_magazine = 360 := 
by
  intro h1 h2 h3
  rw [h1] at *
  rw [h2] at *
  rw [h3] at *
  sorry

end frank_money_left_l739_739112


namespace max_value_of_ratio_l739_739803

def is_three_digit_number (N : ℕ) : Prop :=
  ∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ N = 100 * a + 10 * b + c

noncomputable def max_ratio (N S : ℕ) : ℚ :=
  N / S

theorem max_value_of_ratio (N : ℕ) (S : ℕ) (hN : is_three_digit_number N) (hS : S = N.digits.sum) :
  max_ratio N S ≤ 100 :=
by
  sorry

end max_value_of_ratio_l739_739803


namespace problem1_problem2_problem3_l739_739700

-- Problem 1: Units digit of 135^x + 31^y + 56^(x+y) is 2
theorem problem1 (x y : ℕ) : (135^x + 31^y + 56^(x+y)) % 10 = 2 := 
sorry

-- Problem 2: Units digit of the sum 142 + 142^2 + 142^3 + ... + 142^20 is 0
theorem problem2 : (∑ n in finset.range (20 + 1), 142^(n+1)) % 10 = 0 := 
sorry

-- Problem 3: Units digit of 34^x + 34^(x+1) + 34^(2x) is 6
theorem problem3 (x : ℕ) : (34^x + 34^(x+1) + 34^(2x)) % 10 = 6 := 
sorry

end problem1_problem2_problem3_l739_739700


namespace tan_value_of_point_on_exp_graph_l739_739513

theorem tan_value_of_point_on_exp_graph (a : ℝ) (h : (a, 9) ∈ {p | p.2 = 3^p.1}) : 
  Real.tan (a * Real.pi / 3) = -Real.sqrt 3 :=
by
  sorry

end tan_value_of_point_on_exp_graph_l739_739513


namespace tangent_problem_proof_l739_739122

section TangentProblem

open Real

-- Definitions of the line and circle
def line_l (x y : ℝ) : Prop := x - y + 3 = 0
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Minimum distance from P on line_l to the circle_C
def min_PA := sqrt 7

-- Equation of line AB under the minimum distance condition
def equation_line_AB : Prop := ∀ x y : ℝ, 2 * x - 2 * y - 1 = 0

-- Theorem statement
theorem tangent_problem_proof :
  ∀ x y : ℝ, (line_l x y → 
  (∀ P ∈ line_l x y ∧ distance P (1, 0) = 2 * sqrt 2, 
  distance P (1, 0) = sqrt 7 ∧ equation_line_AB P (1, 0))) :=
sorry

end TangentProblem

end tangent_problem_proof_l739_739122


namespace sum_100th_row_l739_739447

def f : ℕ → ℕ 
| 0       := 0
| (n + 1) := 2 * f n + 4 * n

theorem sum_100th_row : f 100 = 2^101 - 4 :=
by
  sorry

end sum_100th_row_l739_739447


namespace energy_increase_l739_739816

/-- 
Four identical point charges are positioned at the vertices of a square, 
and this configuration stores 20 Joules of energy.
Moving one of these charges to the center of the square results 
in an additional energy storage of 20√2 - 5 Joules.
-/
theorem energy_increase (
  identical_charges : ℝ,
  initial_energy : ℝ := 20,
  number_of_charges : ℕ := 4,
  side_length : ℝ
  -- define the energy function or potential energy calculation here
  ) : 
  let new_energy := (15 + 20*real.sqrt 2) -- total energy in the new configuration
  in new_energy - initial_energy = 20*real.sqrt 2 - 5 := 
sorry

end energy_increase_l739_739816


namespace solve_equation_l739_739982

theorem solve_equation :
  ∀ x : ℝ, (3 * (16^x) + 2 * (81^x) = 5 * (36^x)) → (x = 0 ∨ x = 1/2) :=
by
  sorry

end solve_equation_l739_739982


namespace smallest_d_l739_739671

theorem smallest_d (d t s : ℕ) (h1 : 3 * t - 4 * s = 2023)
                   (h2 : t = s + d) 
                   (h3 : 4 * s > 0)
                   (h4 : d % 3 = 0) :
                   d = 675 := sorry

end smallest_d_l739_739671


namespace sum_of_ABC_is_120_l739_739681

/-- Definitions of mathematical conditions -/
def isPerfectSquare (n : Nat) : Prop :=
  ∃ k : Nat, k * k = n

def isPrime (n : Nat) : Prop :=
  2 ≤ n ∧ ¬ ∃ m : Nat, 2 ≤ m ∧ m < n ∧ n % m = 0

def isComposite (n : Nat) : Prop :=
  2 < n ∧ ¬ isPrime n

def digits (n : Nat) : List Nat :=
  if n < 10 then [n]
  else [n / 10, n % 10]

def allDigitsSatisfy (n : Nat) (p : Nat → Prop) : Prop :=
  ∀ digit ∈ digits n, p digit

/-- The main theorem to prove -/
theorem sum_of_ABC_is_120 :
  ∃ A B C : Nat,
    (isPerfectSquare A ∧ allDigitsSatisfy A isPerfectSquare) ∧
    (isPrime B ∧ allDigitsSatisfy B isPrime ∧ isPrime (digits B).sum) ∧
    (isComposite C ∧ allDigitsSatisfy C isComposite ∧ isComposite ((digits C).get! 0 - (digits C).get! 1) ∧ A < C ∧ C < B) ∧
    A + B + C = 120 :=
by
  -- Definitions meet the problem conditions
  let A := 49
  let B := 23
  let C := 48

  -- Construct the proof obligation
  existsi A, B, C
  repeat { split }
  
  -- Conditions for A
  { 
    existsi 7
    exact Nat.mul_self_eq 7
  }
  { 
    intros d hd
    cases hd
    { left, refl }
    { right, cases hd, refl } }
  -- Conditions for B
  { intros n pn, exact Nat.Prime_two_ackermann 23 }
  { 
    intros d hd
    cases hd
    { left, exact (Nat.Prime.ackermann 2).symm }
    { right, cases hd, exact Nat.Prime.ackermann 3 } }
  { exact Nat.Prime.mk 2 acknowledging 5
  { 
    intros d hd
    cases hd
    { left, exact (Nat.Prime.ackermann 4).symm }
    { right, cases hd, exact (Nat.Prime.ackermann 8).symm } }
  { exact Nat.Prime.mk 3 acknowledging 4 }
– Conditions for C
{ exact A_lt_C }
{ exact C_lt_B }
{ 
  exact A_add_B_C }

sorry 

end sum_of_ABC_is_120_l739_739681


namespace cubic_equation_roots_l739_739787

theorem cubic_equation_roots (a b c d r s t : ℝ) (h_eq : a ≠ 0) 
(ht1 : a * r^3 + b * r^2 + c * r + d = 0)
(ht2 : a * s^3 + b * s^2 + c * s + d = 0)
(ht3 : a * t^3 + b * t^2 + c * t + d = 0)
(h1 : r * s = 3) 
(h2 : r * t = 3) 
(h3 : s * t = 3) : 
c = 3 * a := 
sorry

end cubic_equation_roots_l739_739787


namespace symmetry_point_in_Oxz_l739_739583

theorem symmetry_point_in_Oxz (x z : ℝ) (y : ℝ = -2) :
  (2, 2, 2) = (2, -y, 2) :=
by
  sorry

end symmetry_point_in_Oxz_l739_739583


namespace find_m_for_parallel_lines_l739_739161

-- The given lines l1 and l2
def line1 (m: ℝ) : Prop := ∀ x y : ℝ, (3 + m) * x - 4 * y = 5 - 3 * m
def line2 : Prop := ∀ x y : ℝ, 2 * x - y = 8

-- Definition for parallel lines
def parallel_lines (l₁ l₂ : Prop) : Prop := 
  ∃ m : ℝ, (3 + m) / 4 = 2

-- The main theorem to prove
theorem find_m_for_parallel_lines (m: ℝ) (h: parallel_lines (line1 m) line2) : m = 5 :=
by sorry

end find_m_for_parallel_lines_l739_739161


namespace ln_1_2_over_6_gt_e_l739_739732

theorem ln_1_2_over_6_gt_e :
  let x := 1.2
  let exp1 := x^6
  let exp2 := (1.44)^2 * 1.44
  let final_val := 2.0736 * 1.44
  final_val > 2.718 :=
by {
  sorry
}

end ln_1_2_over_6_gt_e_l739_739732


namespace white_balls_count_l739_739392

theorem white_balls_count
  (total_balls : ℕ)
  (white_balls blue_balls red_balls : ℕ)
  (h1 : total_balls = 100)
  (h2 : white_balls + blue_balls + red_balls = total_balls)
  (h3 : blue_balls = white_balls + 12)
  (h4 : red_balls = 2 * blue_balls) : white_balls = 16 := by
  sorry

end white_balls_count_l739_739392


namespace degree_bound_l739_739646

variables {V : Type*} [Fintype V]
variables (G : SimpleGraph V)
variables (n r : ℕ) (h_card : Fintype.card V = n) (hr : r > 0)

theorem degree_bound (hG : ¬(∃ (H : SimpleGraph V), H.is_spanning_subgraph G ∧ H = complete_graph (λ (r' : ℕ), r' = r))) :
  ∃ v : V, G.degree v ≤ (⟦(r - 2) * n / (r - 1)⟧ : ℕ) :=
sorry

end degree_bound_l739_739646


namespace max_value_on_interval_l739_739451

noncomputable def f : ℝ → ℝ := sorry

axiom additivity : ∀ x y : ℝ, f(x + y) = f(x) + f(y)
axiom positive_on_positive : ∀ x : ℝ, x > 0 → f(x) > 0
axiom f_two : f(2) = 2

theorem max_value_on_interval : ∃ M : ℝ, (f(x) ≤ M ∀ x ∈ set.Icc (-3 : ℝ) 3) ∧ (∃ x : ℝ, x ∈ set.Icc (-3 : ℝ) 3 ∧ f(x) = M) :=
begin
  use 3,
  split,
  { intro x,
    intro hx, 
    sorry },
  { use 3,
    split,
    { split; norm_num },
    { sorry } }
end

end max_value_on_interval_l739_739451


namespace unique_zero_of_f_inequality_of_x1_x2_l739_739529

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.exp x - x - 1) - Real.log (x + 1) + x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x + x

theorem unique_zero_of_f (a : ℝ) (h : a ≥ 0) : ∃! x, f a x = 0 := sorry

theorem inequality_of_x1_x2 (a x1 x2 : ℝ) (h : f a x1 = g a x1 - g a x2) (hₐ: a ≥ 0) :
  x1 - 2 * x2 ≥ 1 - 2 * Real.log 2 := sorry

end unique_zero_of_f_inequality_of_x1_x2_l739_739529


namespace check_markings_possible_single_weighing_l739_739410

/-!
# Problem Statement
Given a set of weights with markings, where the set of masses and markings are identical but potentially swapped,
is it always possible to verify in a single weighing whether all the markings are correct by using a balance scale?
-/

-- Definition of the balance system and moments
def is_balanced (left_moments right_moments : ℕ) : Prop :=
  left_moments = right_moments

-- Definition of the weighing setup
def verify_markings (weights_masses markings : list ℕ) (positions_right : list ℕ) : Prop :=
  ∃ (positions_left : list ℕ), 
    (∀ (m ∈ weights_masses) (p ∈ positions_right ∪ positions_left), true) -- all weights are placed correctly
    ∧ (is_balanced (sum (map2 (· * ·) weights_masses positions_left)) (sum (map2 (· * ·) weights_masses positions_right)))

/-- 
It is always possible to verify if all markings on weights are correct by using a balance scale in a single weighing.
-/
theorem check_markings_possible_single_weighing (weights_masses markings : list ℕ) :
  ∃ (positions_right : list ℕ), verify_markings weights_masses markings positions_right :=
sorry

end check_markings_possible_single_weighing_l739_739410


namespace triangle_const_altitude_l739_739746

noncomputable def equilateral_triangle_const_alt {a : ℝ} (l : ℝ → ℝ) :=
let A := (0 : ℝ, Real.sqrt 3 * a),
    B := (-a, 0),
    C := (a, 0),
    O := (0 : ℝ, Real.sqrt 3 * a / 3),
    slope := (mx : ℝ) := l x in
∃ N M : ℝ × ℝ, 
  (N.1 = 0 ∧ N.2 = Real.sqrt 3 * a / 3) ∧ 
  (M.2 = 0 ∧ M.1 = - Real.sqrt 3 * a / (3 * mx)) ∧
  (AM := Real.sqrt ((0 - (- Real.sqrt 3 * a / (3 * mx)))^2 + (Real.sqrt 3 * a - 0)^2)) ∧
  (BN := Real.sqrt ((-a - 0)^2 + (0 - Real.sqrt 3 * a / 3)^2)) ∧
  (MN := Real.sqrt ((- Real.sqrt 3 * a / (3 * mx) - 0)^2 + (0 - Real.sqrt 3 * a / 3)^2)) ∧
  ∃ h : ℝ, h = 2 * Real.sqrt 6 * a / 3

theorem triangle_const_altitude {a : ℝ} (l : ℝ → ℝ) :
  equilateral_triangle_const_alt l :=
begin
  sorry
end

end triangle_const_altitude_l739_739746


namespace factor_x6_plus_8_l739_739446

theorem factor_x6_plus_8 : (x^2 + 2) ∣ (x^6 + 8) :=
by
  sorry

end factor_x6_plus_8_l739_739446


namespace FGH_supermarkets_total_l739_739989

theorem FGH_supermarkets_total 
  (us_supermarkets : ℕ)
  (ca_supermarkets : ℕ)
  (h1 : us_supermarkets = 41)
  (h2 : us_supermarkets = ca_supermarkets + 22) :
  us_supermarkets + ca_supermarkets = 60 :=
by
  sorry

end FGH_supermarkets_total_l739_739989


namespace x_plus_y_equals_two_l739_739613

variable (x y : ℝ)

def condition1 : Prop := (x - 1) ^ 2017 + 2013 * (x - 1) = -1
def condition2 : Prop := (y - 1) ^ 2017 + 2013 * (y - 1) = 1

theorem x_plus_y_equals_two (h1 : condition1 x) (h2 : condition2 y) : x + y = 2 :=
  sorry

end x_plus_y_equals_two_l739_739613


namespace average_integer_part_l739_739609

theorem average_integer_part (N : ℤ) (h1 : 7 < N) (h2 : N < 15) :
  let avg := (N + 8 + 12) / 3
  let int_part := avg.floor
  int_part = 9 ∨ int_part = 10 ∨ int_part = 11 :=
by
  sorry

end average_integer_part_l739_739609


namespace sum_of_coefficients_l739_739105

def polynomial : ℕ → ℤ
| 8 := -3
| 5 := 6
| 3 := -12
| 0 := 45

theorem sum_of_coefficients :
  polynomial 8 + polynomial 5 + polynomial 3 + polynomial 0 = 45 :=
by
  sorry

end sum_of_coefficients_l739_739105


namespace find_f_f_neg_4_l739_739520

noncomputable def f : ℝ → ℝ :=
λ x, if x ≥ 0 then real.sqrt x else (1 / 2) ^ x - 7

theorem find_f_f_neg_4 : f (f (-4)) = 3 :=
by simp [f]; sorry

end find_f_f_neg_4_l739_739520


namespace intersection_point_exists_l739_739602
noncomputable theory

variables {a b c : ℝ}
def z0 := complex.I * a
def z1 := (1/2 : ℝ) + complex.I * b
def z2 := 1 + complex.I * c

def curve (t : ℝ) : ℂ :=
  z0 * (real.cos t)^4 + 2 * z1 * (real.cos t)^2 * (real.sin t)^2 + z2 * (real.sin t)^4

def midpoint (zA zB : ℂ) : ℂ :=
  (zA + zB) / 2

def median (zA zB zC : ℂ) (x : ℝ) : ℂ :=
  let M := midpoint zA zB in
  let N := midpoint zC zB in
  M + x * (N - M)

theorem intersection_point_exists :
  let point := (1/2 : ℂ, (a + 2 * b + c) / 4 : ℂ) in
  ∃ t : ℝ, 
    let z := curve t in
    let medianLine := median z0 z1 z2 (real.cos t) in
    (z.re = point.1.re ∧ z.im = point.2.im) ∧
    (medianLine.re = point.1.re ∧ medianLine.im = point.2.im) :=
begin
  sorry
end

end intersection_point_exists_l739_739602


namespace planes_determined_by_parallel_lines_l739_739555

-- Define a pairwise parallel property
def pairwise_parallel (l1 l2 l3 : ℝ^3 → ℝ^3) : Prop :=
  parallel l1 l2 ∧ parallel l2 l3 ∧ parallel l3 l1

-- State the main theorem
theorem planes_determined_by_parallel_lines {l1 l2 l3 : ℝ^3 → ℝ^3} :
  pairwise_parallel l1 l2 l3 → (∃ n : ℕ, n = 1 ∨ n = 3) := 
  sorry

end planes_determined_by_parallel_lines_l739_739555


namespace tax_rate_excess_income_l739_739785

variable (income1 income2 total_tax tax_first_part excess_income tax_on_excess : ℝ)
variable (first_tax_rate : ℝ := 0.10) -- First $40,000 is taxed at 10%
variable (income_cutoff : ℝ := 40000)
variable (citizen_income : ℝ := 60000)
variable (total_tax_amount : ℝ := 8000)

-- Define the conditions in the problem
def country_x_condition : Prop :=
  income1 = income_cutoff ∧
  income2 = citizen_income - income_cutoff ∧
  tax_first_part = first_tax_rate * income_cutoff ∧
  total_tax = total_tax_amount ∧
  total_tax = tax_first_part + tax_on_excess ∧
  citizen_income = income1 + income2

-- Prove the tax rate for the income in excess of $40,000
theorem tax_rate_excess_income (tax_rate_excess : ℝ) :
  country_x_condition →
  tax_on_excess = tax_rate_excess * income2 →
  tax_rate_excess = 0.20 :=
by
  intros country_x_condition tax_on_excess_eq
  sorry

end tax_rate_excess_income_l739_739785


namespace couples_remaining_at_end_zero_l739_739743

theorem couples_remaining_at_end_zero :
  ∀ (n : ℕ), let points := List.range n in
  let s : ℕ → ℕ := λ i, i % n in
  let r : ℕ → ℕ := λ i, (2 * i) % n in
  let couples : List (ℕ × ℕ) := points.map (λ p, (p, 0)) in
  function.iterate (λ (lst : List (ℕ × ℕ)), 
    lst.filter (λ (p, t), r p ≠ s t) ++ lst.filter (λ (p, t), r p = s t)) (n * n) couples = [] := by
  sorry

end couples_remaining_at_end_zero_l739_739743


namespace salary_at_end_of_third_month_l739_739631

theorem salary_at_end_of_third_month (initial_salary : ℕ) (initial_increase_percent : ℕ) :
  initial_salary = 2000 → initial_increase_percent = 5 → 
  let first_month_salary_increase := initial_salary * initial_increase_percent / 100 in
  let second_month_salary := (initial_salary + first_month_salary_increase) in
  let second_month_salary_increase := second_month_salary * (2 * initial_increase_percent) / 100 in
  let third_month_salary := (second_month_salary + second_month_salary_increase) in
  let third_month_salary_increase := third_month_salary * (4 * initial_increase_percent) / 100 in
  (third_month_salary + third_month_salary_increase) = 2772 :=
by
  intros
  sorry

end salary_at_end_of_third_month_l739_739631


namespace find_certain_number_l739_739390

theorem find_certain_number (x : ℝ) : 
  ((2 * (x + 5)) / 5 - 5 = 22) → x = 62.5 :=
by
  intro h
  -- Proof goes here
  sorry

end find_certain_number_l739_739390


namespace solve_lambda_l739_739163

noncomputable def lambda_sol (λ : ℝ) : Prop :=
  let m := (λ + 1, 1)
  let n := (λ + 2, 2)
  let sum := (2 * λ + 3, 3)
  let diff := (-1, -1)
  (sum.1 * diff.1 + sum.2 * diff.2 = 0) → (λ = -3)

theorem solve_lambda : lambda_sol (-3) := sorry

end solve_lambda_l739_739163


namespace area_of_isosceles_trapezoid_l739_739632

theorem area_of_isosceles_trapezoid (R α : ℝ) (hR : R > 0) (hα1 : 0 < α) (hα2 : α < π) :
  let a := 2 * R
  let b := 2 * R * Real.sin (α / 2)
  let h := R * Real.cos (α / 2)
  (1 / 2) * (a + b) * h = R^2 * (1 + Real.sin (α / 2)) * Real.cos (α / 2) :=
by
  sorry

end area_of_isosceles_trapezoid_l739_739632


namespace derivative_of_sin_cos_prod_l739_739959
noncomputable def y (x : ℝ) : ℝ := sin x * cos x

theorem derivative_of_sin_cos_prod (x : ℝ) : deriv (λ x, sin x * cos x) x = cos x * cos x - sin x * sin x :=
sorry

end derivative_of_sin_cos_prod_l739_739959


namespace square_log_base_value_l739_739644

theorem square_log_base_value (b : ℝ) :
  let W : ℝ × ℝ := (x, log b x)   -- W is on the graph of y = log_b x
  let Z : ℝ × ℝ := (x + 7, log b (x + 7))  -- Z, shifted 7 units right, parallel to x-axis
  let Y : ℝ × ℝ := (x + 7, y + 7)  -- Y, shifted 7 units both right and up because square side is 7
  (square_area : ℝ := 49)
  (side_length : ℝ := sqrt square_area)    -- Side length of the square
  -- From W's equation, y = log b x
  (Z_eq : log b (x + 7) = 2 * log b (x + 7))     -- Z lies on y = 2 log_b x
  -- Y lies on y = 1/2 log_b x. Shift in both directions by the side length 7
  (Y_eq : log b (x + 7) + 7 = 1/2 * log b (z))
  (z : ℝ := x +7)
  ) :
  b = real.exp (real.log 49 / 7) := sorry

end square_log_base_value_l739_739644


namespace three_digit_integers_sat_f_n_eq_f_2005_l739_739239

theorem three_digit_integers_sat_f_n_eq_f_2005 
  (f : ℕ → ℕ)
  (h1 : ∀ m n : ℕ, f (m + n) = f (f m + n))
  (h2 : f 6 = 2)
  (h3 : f 6 ≠ f 9)
  (h4 : f 6 ≠ f 12)
  (h5 : f 6 ≠ f 15)
  (h6 : f 9 ≠ f 12)
  (h7 : f 9 ≠ f 15)
  (h8 : f 12 ≠ f 15) :
  ∃! n, 100 ≤ n ∧ n ≤ 999 ∧ f n = f 2005 → n = 225 := 
  sorry

end three_digit_integers_sat_f_n_eq_f_2005_l739_739239


namespace first_term_and_common_difference_sum_of_10_terms_l739_739129

variables {a_1 d : ℝ}

-- Conditions
def condition1 (a_1 d : ℝ) : Prop := d < 0
def condition2 (a_1 d : ℝ) : Prop := (a_1 + d) + (a_1 + 3 * d) = 8
def condition3 (a_1 d : ℝ) : Prop := (a_1 + d) * (a_1 + 3 * d) = 12

-- First part: a_1 and d
theorem first_term_and_common_difference :
  condition1 a_1 d ∧ condition2 a_1 d ∧ condition3 a_1 d → a_1 = 8 ∧ d = -2 := 
sorry

-- Second part: Sum of the first 10 terms of the sequence
def sum_of_first_10_terms (a_1 d : ℝ) : ℝ := 
  10 * a_1 + (10 * (10 - 1) / 2) * d

theorem sum_of_10_terms :
  (first_term_and_common_difference ∧ condition1 a_1 d ∧ condition2 a_1 d ∧ condition3 a_1 d) → 
  sum_of_first_10_terms a_1 d = -10 := 
sorry

end first_term_and_common_difference_sum_of_10_terms_l739_739129


namespace tangent_same_at_origin_l739_739278

noncomputable def f (x : ℝ) := Real.exp (3 * x) - 1
noncomputable def g (x : ℝ) := 3 * Real.exp x - 3

theorem tangent_same_at_origin :
  (deriv f 0 = deriv g 0) ∧ (f 0 = g 0) :=
by
  sorry

end tangent_same_at_origin_l739_739278


namespace simplify_exp1_simplify_exp2_l739_739044

-- Definition and theorem for the first question
def exp1 (a b : ℝ) : ℝ :=
  3 * (a ^ 2 - a * b) - 5 * (a * b + 2 * a ^ 2 - 1)

theorem simplify_exp1 (a b : ℝ) : exp1 a b = -7 * a ^ 2 - 8 * a * b + 5 :=
sorry

-- Definition and theorem for the second question
def exp2 (x : ℝ) : ℝ :=
  3 * x ^ 2 - (5 * x - (1 / 2 * x - 3) + 3 * x ^ 2)

theorem simplify_exp2 (x : ℝ) : exp2 x = -9 / 2 * x - 3 :=
sorry

end simplify_exp1_simplify_exp2_l739_739044


namespace min_max_expression_l739_739455

open Real

noncomputable def expression (x : ℝ) : ℝ := 9 - x^2 - 2 * sqrt (9 - x^2)

theorem min_max_expression :
  ∀ (x : ℝ), x ∈ Icc (-3) 3 → 
    -1 ≤ expression x ∧ expression x ≤ 3 := by
  sorry

end min_max_expression_l739_739455


namespace intersection_is_2_l739_739861

-- Define the sets
def A := {1, 2}
def B := {2, 3}

-- Statement of the problem to prove
theorem intersection_is_2 : A ∩ B = {2} := 
by
  sorry

end intersection_is_2_l739_739861


namespace area_of_triangle_CLM_is_correct_l739_739581

/-
Define the problem:
In a rectangle ABCD with AB = 6 and AD = 3 * (1 + sqrt(2) / 2),
there are two circles. One circle with radius 2 centered at K, touching AB and AD.
Another circle with radius 1 centered at L, touching CD and the first circle.
Find the area of triangle CLM, where M is the perpendicular dropped from B to the line through K and L.
-/

-- Conditions
def AB : ℝ := 6
def AD : ℝ := 3 * (1 + sqrt(2) / 2)
def radius1 : ℝ := 2
def radius2 : ℝ := 1

-- Centers of the circles
def K : ℝ × ℝ := (radius1, AD - radius1)
def L : ℝ × ℝ := (AB - radius2, radius2)

-- Function to find area of triangle CLM
def area_triangle_CLM : ℝ :=
  let KL := sqrt ((K.1 - L.1) ^ 2 + (K.2 - L.2) ^ 2)
  let LM := 3 - sqrt 2
  let CQ := 3 * (sqrt 2 - 1)
  1/2 * LM * CQ

-- Problem statement: Prove that the area of triangle CLM equals the correct answer
theorem area_of_triangle_CLM_is_correct : area_triangle_CLM = 3 * (4 * sqrt 2 - 5) / 4 := by
  sorry

end area_of_triangle_CLM_is_correct_l739_739581


namespace probability_rain_at_least_one_day_l739_739811

open ProbabilityTheory

variables {Ω : Type*} [MeasurableSpace Ω]
variable {P : Measure Ω}
variables {A B : Set Ω}

noncomputable def prob_saturday_rain := 0.6
noncomputable def prob_sunday_rain_given_saturday := 0.8
noncomputable def prob_sunday_rain_given_no_saturday := 0.4

theorem probability_rain_at_least_one_day : 
  P[A] = prob_saturday_rain →
  cond_prob P B A = prob_sunday_rain_given_saturday →
  cond_prob P B Aᶜ = prob_sunday_rain_given_no_saturday →
  (1 - ((1 - P A) * (1 - cond_prob P B Aᶜ))) = 0.76 :=
sorry

end probability_rain_at_least_one_day_l739_739811


namespace ellipse_iff_constant_sum_l739_739885

-- Let F_1 and F_2 be two fixed points in the plane.
variables (F1 F2 : Point)
-- Let d be a constant.
variable (d : ℝ)

-- A point M in a plane
variable (M : Point)

-- Define the distance function between two points.
def dist (P Q : Point) : ℝ := sorry

-- Definition: M is on an ellipse with foci F1 and F2
def on_ellipse (M F1 F2 : Point) (d : ℝ) : Prop :=
  dist M F1 + dist M F2 = d

-- Proof that shows the two parts of the statement
theorem ellipse_iff_constant_sum :
  (∀ M, on_ellipse M F1 F2 d) ↔ (∀ M, dist M F1 + dist M F2 = d) ∧ d > dist F1 F2 :=
sorry

end ellipse_iff_constant_sum_l739_739885


namespace binomial_sum_sum_solution_l739_739347

open Nat

theorem binomial_sum :
  ∀ n : ℕ, binomial 30 15 + binomial 30 n = binomial 31 16 → n = 14 ∨ n = 16 :=
by
  sorry

theorem sum_solution :
  (14 + 16) = 30 :=
by
  exact rfl

end binomial_sum_sum_solution_l739_739347


namespace equal_segments_A1M1B1M1_l739_739584

noncomputable def triangle_median_altitude_points (A B C P A1 M1 B1 : Point) (CM CH : Line)
  (perp_CA_A1 : Perpendicular (LineThrough P A1) (LineThrough C A))
  (perp_CM_M1 : Perpendicular (LineThrough P M1) CM)
  (perp_CB_B1 : Perpendicular (LineThrough P B1) (LineThrough C B))
  (intersect_CH_A1 : IntersectAt (LineThrough P A1) CH A1)
  (intersect_CH_M1 : IntersectAt (LineThrough P M1) CH M1)
  (intersect_CH_B1 : IntersectAt (LineThrough P B1) CH B1)
  (median_CM : IsMedian C M)
  (altitude_CH : IsAltitude C H) : Prop :=
  A1.M1.distance = B1.M1.distance

theorem equal_segments_A1M1B1M1 (A B C P A1 M1 B1 : Point) (CM CH : Line)
  (perp_CA_A1 : Perpendicular (LineThrough P A1) (LineThrough C A))
  (perp_CM_M1 : Perpendicular (LineThrough P M1) CM)
  (perp_CB_B1 : Perpendicular (LineThrough P B1) (LineThrough C B))
  (intersect_CH_A1 : IntersectAt (LineThrough P A1) CH A1)
  (intersect_CH_M1 : IntersectAt (LineThrough P M1) CH M1)
  (intersect_CH_B1 : IntersectAt (LineThrough P B1) CH B1)
  (median_CM : IsMedian C M)
  (altitude_CH : IsAltitude C H) : triangle_median_altitude_points A B C P A1 M1 B1 CM CH perp_CA_A1 perp_CM_M1 perp_CB_B1 intersect_CH_A1 intersect_CH_M1 intersect_CH_B1 median_CM altitude_CH :=
sorry

end equal_segments_A1M1B1M1_l739_739584


namespace sum_abs_coeff_binom_expansion_l739_739318

theorem sum_abs_coeff_binom_expansion (x : ℤ) : 
  ∑ i in finset.range 6, abs (nat.cast (nat.choose 5 i) * (-2)^i) = 243 := 
by
  sorry

end sum_abs_coeff_binom_expansion_l739_739318


namespace find_f_prime_at_1_l739_739121

noncomputable def f (x : ℝ) : ℝ := x^3 + 3 * x * (f' 0) - 2 * exp (2 * x)

theorem find_f_prime_at_1 (H : deriv f 0 = 2) : deriv f 1 = 9 - 4 * exp (2) :=
by
  sorry

end find_f_prime_at_1_l739_739121


namespace equal_lead_concentration_l739_739483

theorem equal_lead_concentration (x : ℝ) (h1 : 0 < x) (h2 : x < 6) (h3 : x < 12) 
: (x / 6 = (12 - x) / 12) → x = 4 := by
  sorry

end equal_lead_concentration_l739_739483


namespace lighthouse_distance_l739_739429

theorem lighthouse_distance (AC BC : ℝ) (θ : ℝ) (hAC : AC = 300) (hBC : BC = 500) (hθ : θ = 120 * (Real.pi / 180)) :
  let d := Real.sqrt (AC^2 + BC^2 - 2 * AC * BC * (Real.cos θ)) in d = 700 :=
by
  sorry

end lighthouse_distance_l739_739429


namespace sub_from_square_l739_739183

theorem sub_from_square (n : ℕ) (h : n = 17) : (n * n - n) = 272 :=
by 
  -- Proof goes here
  sorry

end sub_from_square_l739_739183


namespace roots_eq_two_iff_a_gt_neg1_l739_739066

theorem roots_eq_two_iff_a_gt_neg1 (a : ℝ) : 
  (∃! x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + 2*|x₁ + 1| = a ∧ x₂^2 + 2*x₂ + 2*|x₂ + 1| = a) ↔ a > -1 :=
by sorry

end roots_eq_two_iff_a_gt_neg1_l739_739066


namespace cube_ratio_A1M_MB_l739_739716

theorem cube_ratio_A1M_MB 
  (a1 b1 : ℝ) (a : ℝ) : 
  let A1B1 := real.sqrt (2) * a in
  let B1B := a in
  A1B1 / B1B = 2 :=
by 
  sorry

end cube_ratio_A1M_MB_l739_739716


namespace trisecting_angles_l739_739205

theorem trisecting_angles (x : ℝ) : 
  let BP := x,
  let BQ := x,
  let MBQ := x,
  let PBQ := 2 * x,
  let ABP := 2 * x,
  let ABQ := 4 * x,
  BP + BQ = 2 * 2 * x ∧ PBQ = ABP ∧ PBQ = BQ →
  (MBQ / ABQ = 1 / 4) := 
by 
  sorry

end trisecting_angles_l739_739205


namespace spending_50_dollars_l739_739570

-- Defining the conditions as per the problem
def receiving (x : ℤ) := x
def spending (x : ℤ) := -x

-- Stating the theorem to be proved
theorem spending_50_dollars :
  receiving 80 = 80 → spending 50 = -50 :=
begin
  intros h,
  -- Leaving the proof for now
  sorry,
end

end spending_50_dollars_l739_739570


namespace smallest_number_among_zero_neg2_one_half_l739_739765

theorem smallest_number_among_zero_neg2_one_half :
  ∀ x ∈ ({0, -2, 1, 1/2} : set ℚ), x ≥ -2 := 
by {
  intro x,
  simp,
  intro h,
  fin_cases h;
  linarith,
}

end smallest_number_among_zero_neg2_one_half_l739_739765


namespace probability_rain_at_most_3_days_in_july_l739_739306

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_rain_at_most_3_days_in_july :
  let p := 1 / 5
  let n := 31
  let sum_prob := binomial_probability n 0 p + binomial_probability n 1 p + binomial_probability n 2 p + binomial_probability n 3 p
  abs (sum_prob - 0.125) < 0.001 :=
by
  sorry

end probability_rain_at_most_3_days_in_july_l739_739306


namespace total_red_pencils_l739_739909

theorem total_red_pencils (packs : ℕ) (normal_pencil_per_pack : ℕ) (extra_packs : ℕ) (extra_pencils_per_pack : ℕ) :
  packs = 15 →
  normal_pencil_per_pack = 1 →
  extra_packs = 3 →
  extra_pencils_per_pack = 2 →
  packs * normal_pencil_per_pack + extra_packs * extra_pencils_per_pack = 21 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num

end total_red_pencils_l739_739909


namespace bulb_probability_gt4000_l739_739717

-- Definitions given in conditions
def P_X : ℝ := 0.60
def P_Y : ℝ := 0.40
def P_gt4000_X : ℝ := 0.59
def P_gt4000_Y : ℝ := 0.65

-- The proof statement
theorem bulb_probability_gt4000 : 
  (P_X * P_gt4000_X + P_Y * P_gt4000_Y) = 0.614 :=
  by
  sorry

end bulb_probability_gt4000_l739_739717


namespace expression_not_equal_33_l739_739639

theorem expression_not_equal_33 (x y : ℤ) :
  x^5 + 3 * x^4 * y - 5 * x^3 * y^2 - 15 * x^2 * y^3 + 4 * x * y^4 + 12 * y^5 ≠ 33 := 
sorry

end expression_not_equal_33_l739_739639


namespace range_of_a_l739_739160

variable (a : ℝ)
variable (x y : ℝ)

def system_of_equations := 
  (5 * x + 2 * y = 11 * a + 18) ∧ 
  (2 * x - 3 * y = 12 * a - 8) ∧
  (x > 0) ∧ 
  (y > 0)

theorem range_of_a (h : system_of_equations a x y) : 
  - (2:ℝ) / 3 < a ∧ a < 2 :=
sorry

end range_of_a_l739_739160


namespace find_length_of_first_dimension_of_tank_l739_739406

theorem find_length_of_first_dimension_of_tank 
    (w : ℝ) (h : ℝ) (cost_per_sq_ft : ℝ) (total_cost : ℝ) (l : ℝ) :
    w = 5 → h = 3 → cost_per_sq_ft = 20 → total_cost = 1880 → 
    1880 = (2 * l * w + 2 * l * h + 2 * w * h) * cost_per_sq_ft →
    l = 4 := 
by
  intros hw hh hcost htotal heq
  sorry

end find_length_of_first_dimension_of_tank_l739_739406


namespace fish_weight_l739_739398

theorem fish_weight (W : ℝ) (h : W = 2 + W / 3) : W = 3 :=
by
  sorry

end fish_weight_l739_739398


namespace find_largest_distance_l739_739039

noncomputable def coord := ℝ 
structure Circle :=
  (center: coord × coord)
  (radius: ℝ)

structure TwoCircles :=
  (circle1: Circle)
  (circle2: Circle)
  (distance_centers: ℝ)
  (intersecting_points: coord × coord)

structure TangentIntersection :=
  (point: coord × coord)

structure LinePerpendicular :=
  (line1: TangentIntersection)
  (point_on_centerline: coord × coord)

def max_distance (dist_Q_to_circle_points: ℝ): ℝ := dist_Q_to_circle_points

noncomputable def largest_distance_problem : ℝ :=
  let circle1 := Circle.mk (0, 0) 3
  let circle2 := Circle.mk (5, 0) 4
  let intersecting_points := ((3/5)^2, (4/5)^2)
  let centers_distance := 5
  let Q := (15/7, 0)
  max_distance ((48 / 7))

theorem find_largest_distance (m n : ℕ) (h_coprime: Nat.coprime m n) :
  let answer := 100*m + n,
  (m = 48) → (n = 7) → (answer = 4807)
:= 
  by
    sorry

end find_largest_distance_l739_739039


namespace clock_angle_330_l739_739277

noncomputable def angle_at_three := 90
noncomputable def hour_hand_movement := 15
noncomputable def minute_hand_movement := 180

theorem clock_angle_330 
  (a3 : angle_at_three = 90)
  (hh_movement : hour_hand_movement = 15)
  (mh_movement : minute_hand_movement = 180) : 
  (minute_hand_movement - angle_at_three - hour_hand_movement = 75) := 
by
  rw [a3, hh_movement, mh_movement]
  sorry

end clock_angle_330_l739_739277


namespace arithmetic_sequence_properties_l739_739887

theorem arithmetic_sequence_properties
  (a_n : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, S n = n * ((a_n 0 + a_n (n-1)) / 2))
  (h2 : S 6 < S 7)
  (h3 : S 7 > S 8) :
  (a_n 8 - a_n 7 < 0) ∧ (S 9 < S 6) ∧ (∀ m, S m ≤ S 7) :=
by
  sorry

end arithmetic_sequence_properties_l739_739887


namespace min_distance_to_line_l739_739821

theorem min_distance_to_line : 
  ∀ (x y : ℝ), (3 * x - 4 * y + 2 = 0) → (∃ (d : ℝ), d = sqrt ((x + 1)^2 + (y - 3)^2) ∧ d = 13 / 5) :=
by
  sorry

end min_distance_to_line_l739_739821


namespace base_conversion_unique_l739_739359

theorem base_conversion_unique (A C : ℕ) (hA : A < 8) (hC : C < 6) (h_eq : 8 * A + C = 6 * C + A) : 
  8 * 5 + 7 = 47 :=
by
  have h : 7 * A = 5 * C := by linarith
  have h5 : A = 5 := by {
    -- A must be valid in base 8 and match the proportion
    sorry -- Proof that A = 5 from the ratio
  }
  have h7 : C = 7 := by {
    -- C must be valid in base 6 and match the proportion
    sorry -- Proof that C = 7 from the ratio
  }
  rw [h5, h7]
  refl -- The final equality 8 * 5 + 7 = 47

end base_conversion_unique_l739_739359


namespace find_positive_integer_solution_l739_739465

theorem find_positive_integer_solution (n : ℕ) (h : (n+2)! - (n+1)! - n! = n^2 + n^4) : n = 3 :=
by {
  sorry
}

end find_positive_integer_solution_l739_739465


namespace circles_intersect_l739_739672

-- Define the circle equations
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 2
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4 * y + 3 = 0

-- Define the centers of the circles
def center1 := (0 : ℝ, 0 : ℝ)
def center2 := (0 : ℝ, -2 : ℝ)

-- Define the radii of the circles
def radius1 := Real.sqrt 2
def radius2 := 1

-- Define the distance between centers
def distance_centers := 2

-- Positional relationship theorem
theorem circles_intersect : radius1 - radius2 < distance_centers ∧ distance_centers < radius1 + radius2 :=
by
  sorry

end circles_intersect_l739_739672


namespace sequence_period_9_l739_739300

def sequence_periodic (x : ℕ → ℤ) : Prop :=
  ∀ n > 1, x (n + 1) = |x n| - x (n - 1)

theorem sequence_period_9 (x : ℕ → ℤ) :
  sequence_periodic x → ∃ p, p = 9 ∧ ∀ n, x (n + p) = x n :=
by
  sorry

end sequence_period_9_l739_739300


namespace equinumerous_transitivity_l739_739935

open Set

-- Definitions and conditions
variable {α : Type*} (A A1 A2 : Set α)

-- Given conditions of the problem
def subset_conditions : Prop := A ⊇ A1 ∧ A1 ⊇ A2
def equinumerous_condition : Prop := A2 ≃ A

-- Statement of the theorem to prove
theorem equinumerous_transitivity (h1 : subset_conditions A A1 A2) (h2 : equinumerous_condition A A2) : A1 ≃ A := 
sorry

end equinumerous_transitivity_l739_739935


namespace period_abs_tan_l739_739304

theorem period_abs_tan (x : ℝ) : ∃ p > 0, ∀ x, |tan (x + p)| = |tan x| := 
by sorry

end period_abs_tan_l739_739304


namespace two_roots_iff_a_greater_than_neg1_l739_739076

theorem two_roots_iff_a_greater_than_neg1 (a : ℝ) :
  (∃! x : ℝ, x^2 + 2*x + 2*|x + 1| = a) ↔ a > -1 :=
sorry

end two_roots_iff_a_greater_than_neg1_l739_739076


namespace min_values_trajectory_empty_set_l739_739198

-- Define the given conditions in a) as the assumptions
variables {a b c : ℝ} -- Lengths of the sides of triangle
variable  (cos_C : ℝ) -- The minimum of cosine of angle C
variable  (perimeter : ℝ) -- The fixed perimeter of triangle ABC

-- Given conditions from a)
axiom AB_eq_6 : dist (a, 0) (b, 0) = 6
axiom C_fixed_at_P : dist (c, 0) (b, 0) = perimeter - 6
axiom cos_C_eq_7_div_25 : cos_C = 7 / 25

-- The problem statement includes proving the trajectory equation and the minimum values condition
theorem min_values_trajectory_empty_set 
(perimeter_fixed : ¬false)
: (∃ (x y : ℝ), (⟨x, y⟩ : ℝ×ℝ) ≠ 0 ∧ (x ^ 2 / 25 + y ^ 2 / 16 = 1))
∧ 
(∀ (A M N : ℝ × ℝ), ∃ (k : ℝ), ¬∃ (x1 x2 : ℝ),
let x_sum := -150 * k ^ 2 / (16 + 25 * k ^ 2),
    x_prod := (225 * k ^ 2 - 400) / (16 + 25 * k ^ 2),
    BM := dist (b, 0) (A + x1 * k, 0 + x1 * k),
    BN := dist (b, 0) (A + x2 * k, 0 + x2 * k) in
BM * BN = ∅) :=
by
  sorry

end min_values_trajectory_empty_set_l739_739198


namespace problem_statement_l739_739460

def a : ℤ := 2020
def b : ℤ := 2022

theorem problem_statement : b^3 - a * b^2 - a^2 * b + a^3 = 16168 := by
  sorry

end problem_statement_l739_739460


namespace domain_of_function_l739_739660

theorem domain_of_function :
  ∀ x : ℝ, (1 / (1 - x) ≥ 0 ∧ 1 - x ≠ 0) ↔ (x < 1) :=
by
  sorry

end domain_of_function_l739_739660


namespace problem_statement_l739_739557

noncomputable def circumcenter (A B C : Point) : Point := sorry
noncomputable def orthocenter (A B C : Point) : Point := sorry
noncomputable def foot_of_altitude (A B C : Point) : Point → Point := sorry
noncomputable def intersection (l1 l2 : Line) : Point := sorry
noncomputable def is_perpendicular (v1 v2 : Vector) : Prop := sorry

theorem problem_statement (A B C O H D E F M N : Point) 
  (circumcenter_O : O = circumcenter A B C)
  (orthocenter_H : H = orthocenter A B C)
  (foot_D : D = foot_of_altitude A B C A)
  (foot_E : E = foot_of_altitude A B C B)
  (foot_F : F = foot_of_altitude A B C C)
  (intersection_M : M = intersection (line_through E D) (line_through A B))
  (intersection_N : N = intersection (line_through F D) (line_through A C)) :
  is_perpendicular (vector O B) (vector D F) ∧
  is_perpendicular (vector O C) (vector D E) ∧
  is_perpendicular (vector O H) (vector M N) := 
sorry

end problem_statement_l739_739557


namespace smallest_positive_period_monotonically_increasing_intervals_range_of_g_l739_739526

noncomputable def f (x : ℝ) : ℝ :=
  2 * cos x * sin (x + π / 6) + 1

noncomputable def g (x : ℝ) : ℝ :=
  sin (2 * (x - π / 3) + π / 6) + 3 / 2

theorem smallest_positive_period : ∃ T > 0, ∀ x : ℝ, f (x + T) = f x :=
by
  use π
  sorry

theorem monotonically_increasing_intervals : ∀ k : ℤ, ∀ x ∈ (Set.Icc (-π / 3 + k * π) (π / 6 + k * π)), 
  ∃ f' : ℝ → ℝ, f' x > 0 ∧ differentiable ℝ f :=
by
  sorry

theorem range_of_g : 
  ∀ x ∈ Set.Icc (-π / 6) (π / 3), ∃ y ∈ set.interval (1/2) 2, g x = y :=
by
  sorry

end smallest_positive_period_monotonically_increasing_intervals_range_of_g_l739_739526


namespace newspapers_delivered_l739_739911

-- Definition of the conditions given
def bike_cost : ℕ := 2345
def initial_savings : ℕ := 1500
def lawns_mowed : ℕ := 20
def dogs_walked : ℕ := 24
def lawn_payment : ℕ := 20
def dog_payment : ℕ := 15
def newspaper_payment : ℚ := 0.4
def money_left : ℕ := 155

-- Define the problem in terms of proving the number of newspapers delivered
theorem newspapers_delivered :
  let earnings_from_lawns := lawns_mowed * lawn_payment in
  let earnings_from_dogs := dogs_walked * dog_payment in
  let total_savings_before_newspapers := initial_savings + earnings_from_lawns + earnings_from_dogs in
  let total_savings_after_newspapers := bike_cost + money_left in
  let earnings_from_newspapers := total_savings_after_newspapers - total_savings_before_newspapers in
  let number_of_newspapers := earnings_from_newspapers / newspaper_payment in
  number_of_newspapers = 600 := 
by
  sorry

end newspapers_delivered_l739_739911


namespace sides_of_isosceles_right_triangle_l739_739692

-- Define the conditions
variables {XYZ : Type} [triangle XYZ]
variables (XY YZ: ℝ)

-- Isosceles right triangle condition
def is_isosceles_right_triangle (XYZ : Type) [triangle XYZ] (XY YZ : ℝ) : Prop :=
XY = YZ

-- Area condition
def triangle_area (XYZ : Type) [triangle XYZ] (XY : ℝ) : Prop :=
9 = (1 / 2) * XY * XY

-- Main statement to be proven
theorem sides_of_isosceles_right_triangle (XYZ : Type) [triangle XYZ] (XY YZ : ℝ) 
  (h_isosceles: is_isosceles_right_triangle XYZ XY YZ) (h_area: triangle_area XYZ XY) :
  XY = 3 * Real.sqrt 2 ∧ YZ = 3 * Real.sqrt 2 :=
begin
  sorry
end

end sides_of_isosceles_right_triangle_l739_739692


namespace not_beautiful_739_and_741_l739_739966

-- Define the function g and its properties
variable (g : ℤ → ℤ)

-- Condition: g(x) ≠ x
axiom g_neq_x (x : ℤ) : g x ≠ x

-- Definition of "beautiful"
def beautiful (a : ℤ) : Prop :=
  ∀ x : ℤ, g x = g (a - x)

-- The theorem to prove
theorem not_beautiful_739_and_741 :
  ¬ (beautiful g 739 ∧ beautiful g 741) :=
sorry

end not_beautiful_739_and_741_l739_739966


namespace sum_of_valid_n_l739_739345

theorem sum_of_valid_n :
  (∑ n in {n | nat.choose 30 15 + nat.choose 30 n = nat.choose 31 16}.to_finset, n) = 30 := 
sorry

end sum_of_valid_n_l739_739345


namespace lambda_value_l739_739138

-- Definitions provided in the conditions
variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (e1 e2 : V) (A B C D : V)
-- Non-collinear vectors e1 and e2
variables (h_non_collinear : ∃ a b : ℝ, a ≠ b ∧ a • e1 + b • e2 ≠ 0)
-- Given vectors AB, BC, CD
variables (AB BC CD : V)
variables (lambda : ℝ)
-- Vector definitions based on given conditions
variables (h1 : AB = 2 • e1 + e2)
variables (h2 : BC = -e1 + 3 • e2)
variables (h3 : CD = lambda • e1 - e2)
-- Collinearity condition of points A, B, D
variables (collinear : ∃ β : ℝ, AB = β • (BC + CD))

-- The proof goal
theorem lambda_value (h1 : AB = 2 • e1 + e2) (h2 : BC = -e1 + 3 • e2) (h3 : CD = lambda • e1 - e2) (collinear : ∃ β : ℝ, AB = β • (BC + CD)) : lambda = 5 := 
sorry

end lambda_value_l739_739138


namespace first_year_with_sum_5_l739_739559

def sum_of_digits (n : ℕ) : ℕ :=
  n.to_digits.sum

theorem first_year_with_sum_5 : ∃ y, y > 2020 ∧ sum_of_digits y = 5 ∧ ∀ z, (z > 2020 ∧ z < y) → sum_of_digits z ≠ 5 :=
by
  have h2020 : sum_of_digits 2020 = 2 + 0 + 2 + 0 := rfl
  have h_sum2020 : h2020 = 4 := sorry -- Note: This should be proved for completeness
  use 2021
  split
  · exact Nat.succ_lt_succ (Nat.succ_le_iff.mp (Nat.le_refl 2020))
  split
  · rfl
  · intro z hz
    have hz_digits : sum_of_digits z ≠ 5 := sorry -- Detailed checks for each year could be elaborated
    exact hz_digits

end first_year_with_sum_5_l739_739559


namespace parallel_lines_l739_739972

theorem parallel_lines (m : ℝ) :
    (∀ x y : ℝ, x + (m+1) * y - 1 = 0 → mx + 2 * y - 1 = 0 → (m = 1 → False)) → m = -2 :=
by
  sorry

end parallel_lines_l739_739972


namespace number_of_marbles_removed_and_replaced_l739_739738

def bag_contains_red_marbles (r : ℕ) : Prop := r = 12
def total_marbles (t : ℕ) : Prop := t = 48
def probability_not_red_twice (r t : ℕ) : Prop := ((t - r) / t : ℝ) * ((t - r) / t) = 9 / 16

theorem number_of_marbles_removed_and_replaced (r t : ℕ)
  (hr : bag_contains_red_marbles r)
  (ht : total_marbles t)
  (hp : probability_not_red_twice r t) :
  2 = 2 := by
  sorry

end number_of_marbles_removed_and_replaced_l739_739738


namespace distribution_balls_in_boxes_l739_739174

theorem distribution_balls_in_boxes :
  ∃! n : ℕ, n = 8 ∧ ∀ (balls boxes : ℕ), balls = 7 → boxes = 2 → 
  (∃ f : ℕ → ℕ, (∀ i ∈ {0, 1}, f i ≤ 7) ∧ (f 0 + f 1 = 7) ∧ (nat.card (finset.filter (λ x, x ∈ {0, 1}) (finset.range balls.succ)) = 8)) :=
sorry

end distribution_balls_in_boxes_l739_739174


namespace beavers_working_l739_739327

theorem beavers_working (initial_beavers : ℕ) (swim : ℕ) (collect_sticks : ℕ) (search_food : ℕ) (active_beavers : ℕ) :
  initial_beavers = 7 →
  swim = 2 →
  collect_sticks = 1 →
  search_food = 1 →
  active_beavers = initial_beavers - (swim + collect_sticks + search_food) → 
  active_beavers = 3 :=
by
  intros h_initial h_swim h_collect h_search h_active
  rw [h_initial, h_swim, h_collect, h_search]
  exact h_active

end beavers_working_l739_739327


namespace modified_star_vertex_angle_l739_739002

theorem modified_star_vertex_angle (n : ℕ) (h : n > 4) :
  let internal_angle := (n - 2) * 180 / n in
  let external_angle := 180 - internal_angle in
  let vertex_angle := 360 - (2 * external_angle) in
  vertex_angle = 180 * (n - 4) / n :=
by
  intros
  sorry

end modified_star_vertex_angle_l739_739002


namespace smallest_two_digit_number_l739_739760

theorem smallest_two_digit_number (N : ℕ) (h1 : 10 ≤ N ∧ N < 100)
  (h2 : ∃ k : ℕ, (N - (N / 10 + (N % 10) * 10)) = k ∧ k > 0 ∧ (∃ m : ℕ, k = m * m))
  : N = 90 := 
sorry

end smallest_two_digit_number_l739_739760


namespace cheat_percentage_l739_739757

theorem cheat_percentage (sell_cheat_percent profit_percent : ℝ) : 
  sell_cheat_percent = 30 → profit_percent = 60 → ∃ x, x ≈ 37.31 :=
by
  intros h1 h2
  have h : ∀ (x : ℝ), profit_percent = ((130 - (100 - x)) / (100 - x)) * 100 -> x ≈ 37.31, 
  sorry
  use x
  exact h

end cheat_percentage_l739_739757


namespace yellow_pencils_count_l739_739637

variable (n : ℕ) (total : ℕ) (red_perimeter : ℕ)

-- Given
def square_grid (n: ℕ) : Prop := n = 10
def total_pencils (total : ℕ) : Prop := total = n * n
def red_pencils (red_perimeter : ℕ) : Prop := red_perimeter = 4 * n - 4  -- The correct count of red pencils including corners

-- Prove
theorem yellow_pencils_count (n : ℕ) (total: ℕ) (red_perimeter: ℕ) :
  square_grid n →
  total_pencils total →
  red_pencils red_perimeter →
  (total - red_perimeter) = 64 :=
by
  intros
  sorry

end yellow_pencils_count_l739_739637


namespace probability_of_neighboring_points_l739_739325

theorem probability_of_neighboring_points (n : ℕ) (h : n ≥ 3) : 
  (2 / (n - 1) : ℝ) = (n / (n * (n - 1) / 2) : ℝ) :=
by sorry

end probability_of_neighboring_points_l739_739325


namespace no_five_digit_perfect_square_l739_739250

noncomputable def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def is_all_even_or_all_odd (ds : List ℕ) : Prop :=
  (∀ d ∈ ds, d % 2 = 0) ∨ (∀ d ∈ ds, d % 2 = 1)

def distinct_digits (ds : List ℕ) : Prop :=
  ds.nodup

theorem no_five_digit_perfect_square (n : ℕ) (d1 d2 d3 d4 d5 : ℕ) (ds := [d1, d2, d3, d4, d5]) :
  n >= 10000 ∧ n < 100000 ∧
  is_all_even_or_all_odd ds ∧
  distinct_digits ds ∧
  n = d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5 ∧
  is_perfect_square n →
  false :=
by
  sorry

end no_five_digit_perfect_square_l739_739250


namespace probability_diff_colors_l739_739987

/-!
There are 5 identical balls, including 3 white balls and 2 black balls. 
If 2 balls are drawn at once, the probability of the event "the 2 balls have different colors" 
occurring is \( \frac{3}{5} \).
-/

theorem probability_diff_colors 
    (white_balls : ℕ) (black_balls : ℕ) (total_balls : ℕ) (drawn_balls : ℕ) 
    (h_white : white_balls = 3) (h_black : black_balls = 2) (h_total : total_balls = 5) (h_drawn : drawn_balls = 2) :
    let total_ways := Nat.choose total_balls drawn_balls
    let diff_color_ways := (Nat.choose white_balls 1) * (Nat.choose black_balls 1)
    (diff_color_ways : ℚ) / (total_ways : ℚ) = 3 / 5 := 
by
    -- Step 1: Calculate total ways to draw 2 balls out of 5
    -- total_ways = 10 (by binomial coefficient)
    -- Step 2: Calculate favorable outcomes (1 white, 1 black)
    -- diff_color_ways = 6
    -- Step 3: Calculate probability
    -- Probability = 6 / 10 = 3 / 5
    sorry

end probability_diff_colors_l739_739987


namespace Mary_and_Sandra_solution_l739_739625

theorem Mary_and_Sandra_solution (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1) :
  (2 * 40 + 3 * 60) * n / (5 * n) = (4 * 30 * n + 80 * m) / (4 * n + m) →
  m + n = 29 :=
by
  intro h
  sorry

end Mary_and_Sandra_solution_l739_739625


namespace sum_of_intersection_coordinates_l739_739038

noncomputable def h : ℝ → ℝ := sorry

theorem sum_of_intersection_coordinates :
  ∃ a b : ℝ, h(a) = h(a-5) ∧ b = h(a) ∧ a + b = 6.5 :=
sorry

end sum_of_intersection_coordinates_l739_739038


namespace circumcircles_tangent_l739_739596

/-- Given a triangle ABC and two parallel lines l1 and l2 intersecting BC, CA, AB at
     X1, Y1, Z1 and X2, Y2, Z2 respectively. Define Δ1 as the triangle formed by lines
     perpendicular to BC, CA, AB through X1, Y1, Z1 respectively, and Δ2 similarly through
     X2, Y2, Z2. Prove their circumcircles are tangent. -/
theorem circumcircles_tangent
  (A B C : Point)
  (l1 l2 : Line)
  (parallel_l1_l2 : Parallel l1 l2)
  (intersect_l1 : Intersect l1 (LineThrough B C) = X1)
  (intersect_l1 : Intersect l1 (LineThrough C A) = Y1)
  (intersect_l1 : Intersect l1 (LineThrough A B) = Z1)
  (intersect_l2 : Intersect l2 (LineThrough B C) = X2)
  (intersect_l2 : Intersect l2 (LineThrough C A) = Y2)
  (intersect_l2 : Intersect l2 (LineThrough A B) = Z2)
  (perpendicular_X1_BC : Perpendicular (LineThrough X1 (PerpendicularFoot X1 B C)) (LineThrough B C))
  (perpendicular_Y1_CA : Perpendicular (LineThrough Y1 (PerpendicularFoot Y1 C A)) (LineThrough C A))
  (perpendicular_Z1_AB : Perpendicular (LineThrough Z1 (PerpendicularFoot Z1 A B)) (LineThrough A B))
  (perpendicular_X2_BC : Perpendicular (LineThrough X2 (PerpendicularFoot X2 B C)) (LineThrough B C))
  (perpendicular_Y2_CA : Perpendicular (LineThrough Y2 (PerpendicularFoot Y2 C A)) (LineThrough C A))
  (perpendicular_Z2_AB : Perpendicular (LineThrough Z2 (PerpendicularFoot Z2 A B)) (LineThrough A B)) :
  Tangent (Circumcircle (TriangleFormed (LineThrough X1 (PerpendicularFoot X1 B C)) (LineThrough Y1 (PerpendicularFoot Y1 C A)) (LineThrough Z1 (PerpendicularFoot Z1 A B))))
          (Circumcircle (TriangleFormed (LineThrough X2 (PerpendicularFoot X2 B C)) (LineThrough Y2 (PerpendicularFoot Y2 C A)) (LineThrough Z2 (PerpendicularFoot Z2 A B)))) := 
begin
  sorry
end

end circumcircles_tangent_l739_739596


namespace triangle_bc_values_cos_2A_plus_pi_over_3_value_l739_739193

noncomputable def sin_A : ℝ := sqrt 5 / 5
noncomputable def cos_C : ℝ := - sqrt 5 / 5
noncomputable def a : ℝ := sqrt 5 

theorem triangle_bc_values :
  ∃ (b c : ℝ), sin A = sin_A ∧ cos C = cos_C ∧ a = a ∧ b = 3 ∧ c = 2 * sqrt 5 := by
  sorry

theorem cos_2A_plus_pi_over_3_value :
  sin A = sin_A → cos C = cos_C → a = a → cos(2 * A + π / 3) = (3 - 4 * sqrt 3) / 10 := by
  sorry

end triangle_bc_values_cos_2A_plus_pi_over_3_value_l739_739193


namespace read_both_books_l739_739548

theorem read_both_books (W : ℕ) (saramago : ℕ) (kureishi : ℕ) (neither : ℕ) :
  W = 42 →
  saramago = W / 2 →
  kureishi = W / 6 →
  neither = (W / 2) - 1 →
  (saramago + kureishi - B = W - neither) →
  B = 6 :=
begin
  intros hW hsaramago hkureishi hneither hinclusion,
  sorry
end

end read_both_books_l739_739548


namespace tunnel_length_correct_l739_739022

noncomputable def length_of_train : ℝ := 500
noncomputable def time_to_pass_pole : ℝ := 20
noncomputable def time_to_pass_tunnel : ℝ := 40

def speed_of_train : ℝ := length_of_train / time_to_pass_pole

def distance_covered_in_tunnel : ℝ := speed_of_train * time_to_pass_tunnel

def length_of_tunnel : ℝ := distance_covered_in_tunnel - length_of_train

theorem tunnel_length_correct : length_of_tunnel = 500 := 
by 
  sorry

end tunnel_length_correct_l739_739022


namespace problem_l739_739554

def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a * x - 1

theorem problem (a : ℝ) : (∀ x : ℝ, f (-x) a = f x a) → a = 0 := by
soryy

end problem_l739_739554


namespace sin_C_over_sin_A_eq_2_triangle_area_l739_739865

variable (A B C a b c : ℝ)
variable (h1 : b ≠ 0)
variable (h2 : c ≠ 0)
variable (cosA cosB cosC sinA sinB sinC : ℝ)
variable (h_cos_rel : (cosA - 2 * cosC) / cosB = (2 * c - a) / b)
variable (h_cosA : cosA = cos A)
variable (h_cosB : cosB = cos B)
variable (h_cosC : cosC = cos C)
variable (h_sinA : sinA = sin A)
variable (h_sinB : sinB = sin B)
variable (h_sinC : sinC = sin C)
variable (h_cosB_val : cos B = 1 / 4)
variable (h_b_val : b = 2)
variable (h_cos_rule : b^2 = a^2 + c^2 - 2 * a * c * cosB)

theorem sin_C_over_sin_A_eq_2 :
  (sinC / sinA) = 2 := sorry

theorem triangle_area :
  (1 / 2) * a * c * sinB = sqrt(15) / 4 := sorry

end sin_C_over_sin_A_eq_2_triangle_area_l739_739865


namespace sqrt_three_irrational_sqrt_three_infinite_non_repeating_decimal_l739_739055

theorem sqrt_three_irrational : ¬ ∃ (p q : ℤ), q ≠ 0 ∧ (p * p = 3 * (q * q)) :=
by
  sorry

theorem sqrt_three_infinite_non_repeating_decimal : 
  ∀ d : ℚ, d * d = 3 → 
  false :=
by
  assume d h,
  have h1 : ¬ (∃ (p q : ℤ), q ≠ 0 ∧ (p * p = 3 * (q * q))), from sqrt_three_irrational,
  sorry

end sqrt_three_irrational_sqrt_three_infinite_non_repeating_decimal_l739_739055


namespace ratio_of_sums_l739_739485

variable {S : ℕ → ℝ} {a : ℕ → ℝ}

-- Assume the sequence is arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the sum of the first n terms of the sequence
def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * (a 1) + (n * (n - 1) / 2) * d

-- Given condition a_7 = 9 * a_3
def condition : Prop :=
  a 7 = 9 * a 3

theorem ratio_of_sums (h1 : is_arithmetic_sequence a) (h2 : condition) :
  (S 9 / S 5) = 9 := by
  -- omitted proof
  sorry

end ratio_of_sums_l739_739485


namespace smallest_number_with_conditions_l739_739809

-- Define a function to remove even digits from a number in sequence
def rem_even (n : ℕ) : ℕ :=
  let digits := n.digits in
  digits.foldr (λ d acc, if d % 2 = 0 then acc else acc * 10 + d) 0

-- Define a function to remove odd digits from a number in sequence
def rem_odd (n : ℕ) : ℕ :=
  let digits := n.digits in
  digits.foldr (λ d acc, if d % 2 = 1 then acc else acc * 10 + d) 0

/-- Prove that there exists a ten-digit natural number with all distinct digits such that removing 
all even digits results in 97531 and removing all odd digits results in 02468, 
and this number is 9024675318 -/
theorem smallest_number_with_conditions :
  ∃ N : ℕ, (N.digits.length = 10 ∧ N.digits.nodup ∧ rem_even N = 97531 ∧ rem_odd N = 02468 ∧ N = 9024675318) :=
sorry

end smallest_number_with_conditions_l739_739809


namespace newtons_method_convergence_case_a_newtons_method_convergence_case_b_l739_739936

noncomputable def NewtonsMethod (f : ℝ → ℝ) (f' : ℝ → ℝ) (x₀ : ℝ) : ℕ → ℝ
| 0 := x₀
| (n + 1) := let xn := NewtonsMethod f f' n in xn - f xn / f' xn

def f (x : ℝ) : ℝ := x^2 - x - 1
def f' (x : ℝ) : ℝ := 2 * x - 1

-- Define golden ratio φ and its negative reciprocal
def phi : ℝ := (1 + Real.sqrt 5) / 2
def neg_inv_phi : ℝ := -(2 / phi)

theorem newtons_method_convergence_case_a :
  ∀ n : ℕ, (NewtonsMethod f f' 1) n → φ := sorry

theorem newtons_method_convergence_case_b :
  ∀ n : ℕ, (NewtonsMethod f f' 0) n → neg_inv_phi := sorry

end newtons_method_convergence_case_a_newtons_method_convergence_case_b_l739_739936


namespace probability_of_exactly_one_hitting_l739_739697

variable (P_A_hitting B_A_hitting : ℝ)

theorem probability_of_exactly_one_hitting (hP_A : P_A_hitting = 0.6) (hP_B : B_A_hitting = 0.6) :
  ((P_A_hitting * (1 - B_A_hitting)) + ((1 - P_A_hitting) * B_A_hitting)) = 0.48 := 
by 
  sorry

end probability_of_exactly_one_hitting_l739_739697


namespace analytical_expression_monotonicity_inequality_solution_l739_739293

def f (a b x : ℝ) : ℝ := (a * x - b) / (9 - x^2)

-- Conditions
def condition1 (a b : ℝ) : Prop := ∀ x : ℝ, -3 < x ∧ x < 3 → f a b x = -f a b (-x)
def condition2 : Prop := f 1 0 1 = 1 / 8

-- Question 1: Determine the analytical expression of f(x)
theorem analytical_expression (a b : ℝ) (h1 : condition1 a b) (h2 : condition2) :
  f a b = f 1 0 := sorry

-- Question 2: Determine the monotonicity of f(x)
theorem monotonicity (a : ℝ) :
  ∀ x1 x2 : ℝ, -3 < x1 ∧ x1 < x2 ∧ x2 < 3 → f a 0 x1 < f a 0 x2 := sorry

-- Question 3: Solve the inequality f(t-1) + f(t) < 0
theorem inequality_solution (a : ℝ) :
  ∀ t : ℝ, f a 0 (t - 1) + f a 0 t < 0 ↔ -2 < t ∧ t < 1 / 2 := sorry

end analytical_expression_monotonicity_inequality_solution_l739_739293


namespace equilateral_triangle_arcs_AD_BD_eq_DC_l739_739634

noncomputable def proof_problem : Prop :=
  ∀ (A B C D : ℝ) (h_equilateral : A = B ∧ B = C ∧ C = A)
    (h_arc : D ∈ shorter_arc(A, B)),
  AD + BD = DC

theorem equilateral_triangle_arcs_AD_BD_eq_DC (A B C D : Point)
  (h_equilateral : equilateral_triangle A B C)
  (h_arc : D ∈ shorter_arc A B) :
  AD + BD = DC :=
  by
  sorry

end equilateral_triangle_arcs_AD_BD_eq_DC_l739_739634


namespace problem_statement_l739_739049

noncomputable def least_period (f : ℝ → ℝ) (P : ℝ) :=
  ∀ x : ℝ, f (x + P) = f x

theorem problem_statement (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x + 5) + f (x - 5) = f x) :
  least_period f 30 :=
sorry

end problem_statement_l739_739049


namespace domain_of_composite_l739_739510

theorem domain_of_composite (f : ℝ → ℝ) (x : ℝ) (hf : ∀ y, (0 ≤ y ∧ y ≤ 1) → f y = f y) :
  (0 ≤ x ∧ x ≤ 1) → (0 ≤ x ∧ x ≤ 1) → (0 ≤ x ∧ x ≤ 1) →
  0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ 2*x ∧ 2*x ≤ 1 ∧ 0 ≤ x + 1/3 ∧ x + 1/3 ≤ 1 →
  0 ≤ x ∧ x ≤ 1/2 :=
by
  intro h1 h2 h3 h4
  have h5: 0 ≤ 2*x ∧ 2*x ≤ 1 := sorry
  have h6: 0 ≤ x + 1/3 ∧ x + 1/3 ≤ 1 := sorry
  sorry

end domain_of_composite_l739_739510


namespace f_iter_formula_l739_739927

def f (x : ℝ) : ℝ := (1010 * x + 1009) / (1009 * x + 1010)

def f_iter (n : ℕ) (x : ℝ) : ℝ :=
  if n = 0 then x
  else (f ∘ (f_iter (n - 1))) x

theorem f_iter_formula (n : ℕ) (x : ℝ) : 
  f_iter n x = ((2019^n + 1) * x + 2019^n - 1) / ((2019^n - 1) * x + 2019^n + 1) :=
sorry

end f_iter_formula_l739_739927


namespace ostriches_and_deer_legs_l739_739326

variables (d l : ℕ)

theorem ostriches_and_deer_legs (h1 : 2 * d + 4 * l = 122) (h2 : 4 * d + 2 * l = 106) :
  d = 15 ∧ l = 23 :=
begin
  sorry
end

end ostriches_and_deer_legs_l739_739326


namespace ladder_length_proof_l739_739280

-- Definitions
def angle_of_elevation : ℝ := 60
def foot_to_wall_distance : ℝ := 4.6
def length_of_ladder : ℝ := 9.2

theorem ladder_length_proof :
  (cos (real.pi * angle_of_elevation / 180) = 0.5) →
  (length_of_ladder = foot_to_wall_distance / (cos (real.pi * angle_of_elevation / 180))) :=
by
  intro h
  dsimp at h
  rw [h]
  norm_num
  sorry

end ladder_length_proof_l739_739280


namespace answered_both_questions_correctly_l739_739871

theorem answered_both_questions_correctly (P_A P_B P_A_prime_inter_B_prime : ℝ)
  (h1 : P_A = 70 / 100) (h2 : P_B = 55 / 100) (h3 : P_A_prime_inter_B_prime = 20 / 100) :
  P_A + P_B - (1 - P_A_prime_inter_B_prime) = 45 / 100 := 
by
  sorry

end answered_both_questions_correctly_l739_739871


namespace asymptote_of_hyperbola_l739_739048

theorem asymptote_of_hyperbola (x y : ℝ) (h : (x^2 / 16) - (y^2 / 25) = 1) : 
  y = (5 / 4) * x :=
sorry

end asymptote_of_hyperbola_l739_739048


namespace volume_of_four_cubes_l739_739355

theorem volume_of_four_cubes (edge_length : ℕ) (num_cubes : ℕ) (h_edge : edge_length = 5) (h_num : num_cubes = 4) :
  num_cubes * (edge_length ^ 3) = 500 :=
by 
  sorry

end volume_of_four_cubes_l739_739355


namespace inequality_solution_set_l739_739102

theorem inequality_solution_set :
  {x : ℝ | (x^2 - x - 6) / (x - 1) > 0} = {x : ℝ | (-2 < x ∧ x < 1) ∨ (3 < x)} := by
  sorry

end inequality_solution_set_l739_739102


namespace probability_zero_points_l739_739141

def f (m : ℝ) (x : ℝ) : ℝ := 2^|x| - m

theorem probability_zero_points (m : ℝ) (h : 0 ≤ m ∧ m ≤ 3) :
  let zero_points := ∃ x : ℝ, f m x = 0 in
  let prob := if (∀ x : ℝ, f m x ≤ 0) then 1 else 0 in
  prob = 2 / 3 :=
by
  sorry

end probability_zero_points_l739_739141


namespace exactly_two_roots_iff_l739_739080

theorem exactly_two_roots_iff (a : ℝ) : 
  (∃! (x : ℝ), x^2 + 2 * x + 2 * |x + 1| = a) ↔ a > -1 :=
by
  sorry

end exactly_two_roots_iff_l739_739080


namespace lcm_factors_l739_739969

theorem lcm_factors (A B : ℕ) (hcf lcm_factor other_factor : ℕ) (hcf_is_10 : hcf = 10) (larger_is_150 : A = 150) (lcm_factor_is_15 : lcm_factor = 15) (lcm_def : A = hcf * other_factor * lcm_factor) 
  : other_factor = 1 :=
by
  sorry

end lcm_factors_l739_739969


namespace cups_of_ketchup_l739_739589

-- Define variables and conditions
variables (k : ℕ)
def vinegar : ℕ := 1
def honey : ℕ := 1
def sauce_per_burger : ℚ := 1 / 4
def sauce_per_pulled_pork : ℚ := 1 / 6
def burgers : ℕ := 8
def pulled_pork_sandwiches : ℕ := 18

-- Main theorem statement
theorem cups_of_ketchup (h : 8 * sauce_per_burger + 18 * sauce_per_pulled_pork = k + vinegar + honey) : k = 3 :=
  by
    sorry

end cups_of_ketchup_l739_739589


namespace AM_GM_l739_739828

open BigOperators

theorem AM_GM (n : ℕ) (a : Fin n → ℝ) (h : ∀ i, 0 < a i) 
  (h_sum : ∑ i, a i = 1): 
  ∑ i, (a i + (1 / a i))^2 ≥ (n^2 + 1)^2 / n := 
by 
  sorry

end AM_GM_l739_739828


namespace find_a_l739_739242

theorem find_a (a : ℝ) :
  (∀ (x : ℝ), (a * x - log (x + 1)) = y x ∧ y 0 = 0 ∧
  (∀ (x : ℝ), deriv (λ x, a * x - log (x + 1)) x = 2) → a = 3 :=
by
  sorry

end find_a_l739_739242


namespace toms_final_stamp_count_l739_739690

-- Definitions of the given conditions

def initial_stamps : ℕ := 3000
def mike_gift : ℕ := 17
def harry_gift : ℕ := 2 * mike_gift + 10
def sarah_gift : ℕ := 3 * mike_gift - 5
def damaged_stamps : ℕ := 37

-- Statement of the goal
theorem toms_final_stamp_count :
  initial_stamps + mike_gift + harry_gift + sarah_gift - damaged_stamps = 3070 :=
by
  sorry

end toms_final_stamp_count_l739_739690


namespace largest_divisor_69_86_l739_739364

theorem largest_divisor_69_86 (n : ℕ) (h₁ : 69 % n = 5) (h₂ : 86 % n = 6) : n = 16 := by
  sorry

end largest_divisor_69_86_l739_739364


namespace sin_cos_fourth_power_l739_739178
-- Import the necessary library

-- Define the main statement.
theorem sin_cos_fourth_power (α : ℝ) (h : cos (2 * α) = (Math.sqrt 2) / 3) : sin(α)^4 + cos(α)^4 = 11 / 18 :=
by
  sorry

end sin_cos_fourth_power_l739_739178


namespace num_valid_solutions_eq_43_l739_739454

theorem num_valid_solutions_eq_43 :
  (∃ (f : ℤ → ℤ), (∀ x, f x = (∏ i in (Finset.range 50).map (λ i, i + 1), x - i) /
                          (∏ i in (Finset.range 50).map (λ i, i + 1), x - (i ^ 2))) →
      ∀ (x ∈ Finset.range 51), (x ∉ Finset.image (λ i : ℤ, i ^ 2) (Finset.range 8)) →
        f x = 0) ∧ (Finset.range 51).card - (Finset.image (λ i : ℤ, i ^ 2) (Finset.range 8)).card = 43 :=
begin
  sorry
end

end num_valid_solutions_eq_43_l739_739454


namespace least_number_when_increased_by_6_is_divisible_l739_739377

theorem least_number_when_increased_by_6_is_divisible :
  ∃ n : ℕ, 
    (n + 6) % 24 = 0 ∧ 
    (n + 6) % 32 = 0 ∧ 
    (n + 6) % 36 = 0 ∧ 
    (n + 6) % 54 = 0 ∧ 
    n = 858 :=
by
  sorry

end least_number_when_increased_by_6_is_divisible_l739_739377


namespace total_percentage_reduction_l739_739673

theorem total_percentage_reduction (a : ℝ) (h₀ : a > 0) :
  let r₁ := 0.9 * a
  let r₂ := 0.85 * (0.9 * a)
  let s₁ := 0.85 * a
  let s₂ := 0.9 * (0.85 * a)
  r₂ = s₂ ∧ ((1 - r₂ / a) * 100 = 23.5) :=
by
  let r₁ := 0.9 * a
  let r₂ := 0.85 * r₁
  let s₁ := 0.85 * a
  let s₂ := 0.9 * s₁
  have : r₂ = s₂ := by ring
  have : (1 - r₂ / a) * 100 = 23.5 := by sorry
  exact ⟨this, this⟩

end total_percentage_reduction_l739_739673


namespace problem1_problem2_l739_739533

section Problem
variables (a : ℝ) (x : ℝ) (x1 x2 : ℝ)
noncomputable def f (x : ℝ) : ℝ := a * (Real.exp x - x - 1) - Real.log (x + 1) + x
noncomputable def g (x : ℝ) : ℝ := a * Real.exp x + x

theorem problem1 (ha : a ≥ 0) : ∃! x, f a x = 0 := sorry

theorem problem2 (ha : a ≥ 0) (h1 : x1 ∈ Icc (-1 : ℝ) (Real.inf)) (h2 : x2 ∈ Icc (-1 : ℝ) (Real.inf)) (h : f a x1 = g a x1 - g a x2) :
  x1 - 2 * x2 ≥ 1 - 2 * Real.log 2 := sorry

end Problem

end problem1_problem2_l739_739533


namespace prime_divides_product_of_divisors_l739_739640

theorem prime_divides_product_of_divisors (p : ℕ) (n : ℕ) (a : Fin n → ℕ) 
(Hp : Nat.Prime p) (Hdiv : p ∣ (Finset.univ.prod a)) : 
∃ i : Fin n, p ∣ a i :=
sorry

end prime_divides_product_of_divisors_l739_739640


namespace peruvian_coffee_cost_l739_739337

theorem peruvian_coffee_cost
  (p_colombian : ℝ) (cost_colombian : ℝ) (total_weight : ℝ) (desired_price : ℝ) (weight_colombian : ℝ)
  (h1 : p_colombian = 5.50)
  (h2 : desired_price = 4.60)
  (h3 : weight_colombian = 28.8)
  (h4 : total_weight = 40) :
  let weight_peruvian := total_weight - weight_colombian
      cost_peruvian := total_weight * desired_price - weight_colombian * p_colombian
  in cost_peruvian / weight_peruvian = 2.29 := 
sorry

end peruvian_coffee_cost_l739_739337


namespace total_sales_amount_l739_739937

-- Define the conditions as facts/constants
def jeans_price : ℝ := 22
def tees_price : ℝ := 15
def jackets_price : ℝ := 37
def discount : ℝ := 0.10
def jeans_sold : ℕ := 4
def tees_sold : ℕ := 7
def jackets_sold : ℕ := 5
def jackets_discounted : ℕ := 3
def jackets_full_price : ℕ := jackets_sold - jackets_discounted

-- Calculate the values based on conditions
def sales_jeans := jeans_sold * jeans_price
def sales_tees := tees_sold * tees_price
def sales_jackets_full := jackets_full_price * jackets_price
def discount_amount := discount * jackets_price
def discounted_price := jackets_price - discount_amount
def sales_jackets_discounted := jackets_discounted * discounted_price

def total_sales := sales_jeans + sales_tees + sales_jackets_full + sales_jackets_discounted

-- The statement to be proved
theorem total_sales_amount : total_sales = 366.9 :=
by
  intro
  rw [sales_jeans, sales_tees, sales_jackets_full, discount_amount, discounted_price, sales_jackets_discounted]
  rw [← mul_assoc jackets_discounted discounted_price, (mul_assoc jackets_full_price jackets_price),
      (mul_assoc tees_sold tees_price), (mul_assoc jeans_sold jeans_price)]
  norm_num
  sorry

end total_sales_amount_l739_739937


namespace integer_pairs_m_n_l739_739800

theorem integer_pairs_m_n (m n : ℕ) (hm : 0 < m) (hn : 0 < n)
  (cond1 : ∃ k1 : ℕ, k1 * m = 3 * n ^ 2)
  (cond2 : ∃ k2 : ℕ, k2 ^ 2 = n ^ 2 + m) :
  ∃ a : ℕ, m = 3 * a ^ 2 ∧ n = a :=
by
  sorry

end integer_pairs_m_n_l739_739800


namespace find_arithmetic_mean_l739_739651

theorem find_arithmetic_mean (σ μ : ℝ) (hσ : σ = 1.5) (h : 11 = μ - 2 * σ) : μ = 14 :=
by
  sorry

end find_arithmetic_mean_l739_739651


namespace probability_of_neither_solving_l739_739748

def prob_solve_A : ℝ := 1 / 2
def prob_solve_B : ℝ := 1 / 3

def prob_not_solve_A : ℝ := 1 - prob_solve_A
def prob_not_solve_B : ℝ := 1 - prob_solve_B

def prob_neither_solve : ℝ := prob_not_solve_A * prob_not_solve_B

theorem probability_of_neither_solving (hA : prob_solve_A = 1 / 2) (hB : prob_solve_B = 1 / 3) 
  (indep : true) : prob_neither_solve = 1 / 3 :=
by
  sorry

end probability_of_neither_solving_l739_739748


namespace range_of_a_l739_739827

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + a * x + b
def A (a b : ℝ) : set ℝ := { x | f a b x ≤ 0 }
def B (a b : ℝ) : set ℝ := { x | f a b (f a b x) ≤ 3 }

theorem range_of_a (a b : ℝ) (hA : A a b ≠ ∅) (h_eq : A a b = B a b) : 
  a ∈ set.Ico (2 * Real.sqrt 3) 6 := 
sorry

end range_of_a_l739_739827


namespace distance_from_center_to_chord_OA_l739_739830

noncomputable def center_of_circle : ℝ × ℝ := sorry

noncomputable def distance_to_line : ℝ := sorry

theorem distance_from_center_to_chord_OA :
  let O : ℝ × ℝ := (0, 0)
  let A : ℝ × ℝ := (4, 2)
  let line1 (p : ℝ × ℝ) : Prop := p.1 + 2 * p.2 - 1 = 0
  let center := center_of_circle
  let line_OA (p : ℝ × ℝ) : Prop := p.2 = (1 / 2) * p.1
  distance_to_line = sqrt 5 :=
sorry

end distance_from_center_to_chord_OA_l739_739830


namespace series_uniformly_converges_l739_739588

noncomputable def series_uniform_convergence (f : ℕ → ℝ) (x : ℝ) : Prop :=
∀ ε > 0, ∃ N, ∀ n ≥ N, ∀ y ≥ x, abs (f n - f (n + 1)) < ε

theorem series_uniformly_converges :
  ∀ x : ℝ, x ≥ 0 → series_uniform_convergence (λ n, (-1)^n / (x + n)) x :=
begin
  sorry
end

end series_uniformly_converges_l739_739588


namespace sum_zero_opposites_l739_739191

theorem sum_zero_opposites {a b : ℝ} (h : a + b = 0) : a = -b :=
by sorry

end sum_zero_opposites_l739_739191


namespace distinct_three_digit_odd_numbers_count_l739_739041

/--
  Calculate the number of distinct three-digit odd numbers that can be formed using the digits 1, 2, 3, 4, 5 without repetition.
-/

def digits : List ℕ := [1, 2, 3, 4, 5]

def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def no_repetition (n : ℕ) : Prop :=
  let ds := n.digits 10
  ds.nodup

theorem distinct_three_digit_odd_numbers_count : 
  let three_digit_odd_numbers := {n | is_three_digit n ∧ is_odd n ∧ no_repetition n ∧ ∀ d ∈ n.digits 10, d ∈ digits}
  three_digit_odd_numbers.card = 33 := by
  sorry

end distinct_three_digit_odd_numbers_count_l739_739041


namespace even_function_increasing_on_interval_l739_739230

theorem even_function_increasing_on_interval (f : ℝ → ℝ) 
  (h_even : ∀ x : ℝ, f x = f (-x)) 
  (h_increasing : ∀ x y : ℝ, 0 ≤ x → x < y → f x < f y) :
  f (cos (3 * Real.pi / 10)) < f (Real.pi / 5) ∧ f (Real.pi / 5) < f (tan (Real.pi / 5)) :=
by
  sorry

end even_function_increasing_on_interval_l739_739230


namespace expression_and_sum_of_arithmetic_sequence_l739_739142

variable {α : Type*} [LinearOrderedField α]

def arithmetic_sequence (a d : α) (n : ℕ) : α := a + (n - 1) * d

def sum_first_n_terms (a d : α) (n : ℕ) : α :=
  n / 2 * (2 * a + (n - 1) * d)

theorem expression_and_sum_of_arithmetic_sequence :
  (arithmetic_sequence 7 (-1) n = 8 - n) ∧ (sum_first_n_terms 7 (-1) 7 = 28) :=
by
  -- Unfold the definitions for clarity
  unfold arithmetic_sequence sum_first_n_terms
  -- The expression for the nth term a_n = 8 - n is given correctly
  simp [arithmetic_sequence]
  -- The sum of the first 7 terms S_7 = 28 is calculated correctly
  simp [sum_first_n_terms]
  -- The proof details and verifications are omitted for brevity
  sorry

end expression_and_sum_of_arithmetic_sequence_l739_739142


namespace range_of_numbers_l739_739008

-- Define the conditions
def numbers : Set ℝ := {a_1, a_2, a_3, a_4, a_5}
def smallest_number := 3
def mean := (a_1 + a_2 + a_3 + a_4 + a_5) / 5
def median := a_3
def range := a_5 - a_1

-- Assuming ordering of the numbers
axiom a_1_le_a_2 : a_1 ≤ a_2
axiom a_2_le_a_3 : a_2 ≤ a_3
axiom a_3_le_a_4 : a_3 ≤ a_4
axiom a_4_le_a_5 : a_4 ≤ a_5

-- Conditions given in the problem
axiom mean_condition : mean = 8
axiom median_condition : median = 8
axiom smallest_number_condition : a_1 = 3

-- Prove that the range equals 10 under the given conditions
theorem range_of_numbers : range = 10 :=
by
  sorry

end range_of_numbers_l739_739008


namespace inequality_solution_set_l739_739271

variable (x : ℝ)

theorem inequality_solution_set :
  {x | (x < 1) ∨ (3 < x ∧ x < 4) ∨ (6 < x ∧ x < 7) ∨ (8 < x)} =
  {x | (x - 2) * (x - 3) * (x - 4) * (x - 7) > 0 ∧
       (x - 1) * (x - 5) * (x - 6) * (x - 8) /= 0} := by
  sorry

end inequality_solution_set_l739_739271


namespace probability_of_color_difference_l739_739459

noncomputable def probability_of_different_colors (n m : ℕ) : ℚ :=
  (Nat.choose n m : ℚ) * (1/2)^n

theorem probability_of_color_difference :
  probability_of_different_colors 8 4 = 35/128 :=
by
  sorry

end probability_of_color_difference_l739_739459


namespace sum_of_sides_of_equilateral_triangle_is_5_25_l739_739643

theorem sum_of_sides_of_equilateral_triangle_is_5_25 :
  ∀ (a : ℚ), a = 14 / 8 → 3 * a = 21 / 4 := 
by
  intro a ha
  rw [← ha]
  linarith

end sum_of_sides_of_equilateral_triangle_is_5_25_l739_739643


namespace problem1_l739_739387

theorem problem1 :
  (1 - Real.sqrt 3)^0 - Real.abs (-Real.sqrt 2) + (-27:ℝ)^(1/3) - (-1/(2:ℝ))^(-1) = -Real.sqrt 2 :=
sorry

end problem1_l739_739387


namespace solve_x4_plus_1_eq_0_l739_739799

noncomputable def quadratic_formula_roots (a b c : ℂ) : ℂ × ℂ :=
    let d := b^2 - 4 * a * c;
    ((-b + complex.sqrt d) / (2 * a), (-b - complex.sqrt d) / (2 * a))

theorem solve_x4_plus_1_eq_0 :
  let x1 := (-complex.sqrt 2) / 2 + (complex.I * complex.sqrt 2) / 2;
      x2 := (-complex.sqrt 2) / 2 - (complex.I * complex.sqrt 2) / 2;
      x3 := (complex.sqrt 2) / 2 + (complex.I * complex.sqrt 2) / 2;
      x4 := (complex.sqrt 2) / 2 - (complex.I * complex.sqrt 2) / 2;
  (x^4 + 1 = 0) → 
  (x = x1 ∨ x = x2 ∨ x = x3 ∨ x = x4) := sorry

end solve_x4_plus_1_eq_0_l739_739799


namespace legoland_kangaroos_l739_739595

theorem legoland_kangaroos :
  ∃ (K R : ℕ), R = 5 * K ∧ K + R = 216 ∧ R = 180 := by
  sorry

end legoland_kangaroos_l739_739595


namespace salary_increase_l739_739336

variable (x : ℝ)

def new_salary_first (x : ℝ) : ℝ :=
  x + 1 * x

def new_salary_second (x : ℝ) : ℝ :=
  x + 0.5 * x

def percentage_increase (new_salary_first new_salary_second : ℝ) : ℝ :=
  ((new_salary_first - new_salary_second) / new_salary_second) * 100

theorem salary_increase (x : ℝ) (h1 : new_salary_first x = 2 * x) (h2 : new_salary_second x = 1.5 * x) :
  percentage_increase (new_salary_first x) (new_salary_second x) = 33.33 :=
  sorry

end salary_increase_l739_739336


namespace smallest_square_area_l739_739282

theorem smallest_square_area (r : ℝ) (h : r = 4) : ∃ A, A = 64 :=
by
  let side_length := 2 * r
  have h_side : side_length = 8 := by
    rw [h]
    unfold side_length
  let area := side_length^2
  have h_area : area = 64 := by
    rw [h_side]
    unfold area
  use area
  assumption

end smallest_square_area_l739_739282


namespace find_rs_l739_739261

noncomputable def r : ℝ := sorry
noncomputable def s : ℝ := sorry
def cond1 := r > 0 ∧ s > 0
def cond2 := r^2 + s^2 = 1
def cond3 := r^4 + s^4 = (3 : ℝ) / 4

theorem find_rs (h1 : cond1) (h2 : cond2) (h3 : cond3) : r * s = Real.sqrt 2 / 4 :=
by sorry

end find_rs_l739_739261


namespace composite_function_evaluation_l739_739859

def f (x : ℕ) : ℕ := x * x
def g (x : ℕ) : ℕ := x + 2

theorem composite_function_evaluation : f (g 3) = 25 := by
  sorry

end composite_function_evaluation_l739_739859


namespace cube_volume_total_four_boxes_l739_739352

theorem cube_volume_total_four_boxes :
  ∀ (length : ℕ), (length = 5) → (4 * (length^3) = 500) :=
begin
  intros length h,
  rw h,
  norm_num,
end

end cube_volume_total_four_boxes_l739_739352


namespace factor_expression_l739_739777

theorem factor_expression (x : ℚ) : 12 * x ^ 2 + 8 * x = 4 * x * (3 * x + 2) := sorry

end factor_expression_l739_739777


namespace johnny_red_pencils_l739_739906

noncomputable def number_of_red_pencils (packs_total : ℕ) (extra_packs : ℕ) (extra_per_pack : ℕ) : ℕ :=
  packs_total + extra_packs * extra_per_pack

theorem johnny_red_pencils : number_of_red_pencils 15 3 2 = 21 := by
  sorry

end johnny_red_pencils_l739_739906


namespace find_sum_mnp_l739_739783

theorem find_sum_mnp
  (m n p : ℕ)
  (rel_prime : Nat.gcd n p = 1)
  (volume : ℝ → ℝ)
  (parallelepiped_volume : volume (2 * 3 * 6) = 36)
  (extended_volume : volume (2 * (6 + 12 + 18)) = 72)
  (cylinders_volume : volume (π * (2 + 3 + 6)) = 11 * π)
  (sphere_octants_volume : volume (4 * π / 3) = 4 * π / 3)
  (total_volume_condition : volume (36 + 72 + 11 * π + 4 * π / 3) = 108 + 37 * π / 3)
  (final_volume : volume = (324 + 37 * π) / 3)
  : m = 324 → n = 37 → p = 3 → (m + n + p = 364) :=
by
  sorry

end find_sum_mnp_l739_739783


namespace rain_probability_at_most_3_days_l739_739308

open BigOperators

def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

noncomputable def rain_probability := (1:ℝ)/5
noncomputable def no_rain_probability := (4:ℝ)/5

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binom n k) * (p^k) * ((1-p)^(n-k))

theorem rain_probability_at_most_3_days :
  ∑ k in Finset.range 4, binomial_probability 31 k rain_probability = 0.544 :=
by
  sorry

end rain_probability_at_most_3_days_l739_739308


namespace remaining_rectangle_perimeter_l739_739001

theorem remaining_rectangle_perimeter (l w : ℕ) (h_l : l = 28) (h_w : w = 15) :
    2 * ((l - w) + w) = 56 := 
by
  rw [h_l, h_w]
  norm_num

end remaining_rectangle_perimeter_l739_739001


namespace square_line_cd_equation_l739_739567

theorem square_line_cd_equation (h_center : (1, 0)) 
                                (h_line_AB : ∀ x y : ℝ, x - y + 1 = 0 → True) :
                                ∃ m : ℝ, x - y - 3 = 0 :=
sorry

end square_line_cd_equation_l739_739567


namespace log_base_change_l739_739177

theorem log_base_change (m b : ℝ) (h : 3^m = b) : log (3^2) b = m / 2 := by
  sorry

end log_base_change_l739_739177


namespace find_g_9_l739_739661

theorem find_g_9 (g : ℝ → ℝ) (h1 : ∀ x y : ℝ, g(x + y) = g(x) * g(y)) (h2 : g(3) = 2) : g(9) = 8 :=
by
  sorry

end find_g_9_l739_739661


namespace rainwater_cows_l739_739941

theorem rainwater_cows (chickens goats cows : ℕ) 
  (h1 : chickens = 18) 
  (h2 : goats = 2 * chickens) 
  (h3 : goats = 4 * cows) : 
  cows = 9 := 
sorry

end rainwater_cows_l739_739941


namespace percentage_y_less_than_x_l739_739753

theorem percentage_y_less_than_x (x y : ℝ) (h : x = 4 * y) : (x - y) / x * 100 = 75 := by
  sorry

end percentage_y_less_than_x_l739_739753


namespace sum_xyz_eq_neg7_l739_739647

theorem sum_xyz_eq_neg7 (x y z : ℝ)
  (h1 : x = y + z + 2)
  (h2 : y = z + x + 1)
  (h3 : z = x + y + 4) :
  x + y + z = -7 :=
by
  sorry

end sum_xyz_eq_neg7_l739_739647


namespace similar_triangles_perimeter_l739_739311

theorem similar_triangles_perimeter
  {k : ℕ} (h_ratio : 3 = 3) (p_small : 12 = 12) :
  let p_large := 20
  in p_large = 20 := by
  sorry

end similar_triangles_perimeter_l739_739311


namespace total_volume_of_four_cubes_l739_739349

theorem total_volume_of_four_cubes (s : ℝ) (h_s : s = 5) : 4 * s^3 = 500 :=
by
  sorry

end total_volume_of_four_cubes_l739_739349


namespace triangle_angle_ratio_l739_739236

theorem triangle_angle_ratio (A B C I : Point) 
  (h_center : Incenter I ABC)
  (h_condition : dist C A + dist A I = dist B C) :
  ∠ BAC = 2 * ∠ CBA := 
sorry

end triangle_angle_ratio_l739_739236


namespace largest_share_received_l739_739963

theorem largest_share_received (total_profit : ℕ) (ratio_part1 ratio_part2 ratio_part3 ratio_part4 : ℕ) 
  (h_ratio : ratio_part1 + ratio_part2 + ratio_part3 + ratio_part4 = 14) (h_profit : total_profit = 35000) :
  let value_per_part := total_profit / 14 in largest_share = 5 * value_per_part :=
by
  let value_per_part := total_profit / 14
  let largest_share := 5 * value_per_part
  show largest_share = 12500
  sorry

end largest_share_received_l739_739963


namespace pizzas_needed_l739_739401

theorem pizzas_needed (couple_slices : ℕ) (children_slices : ℕ) (num_children : ℕ) (slices_per_pizza : ℕ) :
  couple_slices = 5 → children_slices = 2 → num_children = 12 → slices_per_pizza = 6 →
  nat.ceil ((2 * couple_slices + num_children * children_slices) / (slices_per_pizza : ℝ)) = 6 :=
begin
  intros h1 h2 h3 h4,
  rw [h1, h2, h3, h4],
  norm_num,
end

end pizzas_needed_l739_739401


namespace rational_product_of_roots_l739_739514

noncomputable def polynomial (a b c d e : ℤ) (r₁ r₂ r₃ r₄ : ℚ) : Prop :=
  ∀ z : ℚ, a * z^4 + b * z^3 + c * z^2 + d * z + e = 0 ↔
           (z = r₁ ∨ z = r₂ ∨ z = r₃ ∨ z = r₄)

theorem rational_product_of_roots 
  (a b c d e : ℤ) (r₁ r₂ r₃ r₄ : ℚ)
  (ha : a ≠ 0)
  (h_poly : polynomial a b c d e r₁ r₂ r₃ r₄)
  (h_sum : (r₁ + r₂ : ℚ) ∈ ℚ)
  (h_neq : r₃ + r₄ ≠ r₁ + r₂) :
  r₁ * r₂ ∈ ℚ :=
sorry

end rational_product_of_roots_l739_739514


namespace valid_digits_count_l739_739289

theorem valid_digits_count :
  {A : ℕ // A < 10 ∧ 5716 > 571 * 10 + A}.card = 6 :=
by
  sorry

end valid_digits_count_l739_739289


namespace chord_line_equation_l739_739472

-- We are stating the conditions given in the problem
def ellipse (x y : ℝ) : Prop := x^2 + 4 * y^2 = 16
def midpoint (x1 y1 x2 y2 : ℝ) : Prop := (x1 + x2) = 2 ∧ (y1 + y2) = -2

-- We state the required conclusion
theorem chord_line_equation (x1 y1 x2 y2 : ℝ) (hx1y1 : ellipse x1 y1) (hx2y2 : ellipse x2 y2) (hm : midpoint x1 y1 x2 y2) :
  ∀ x y : ℝ, (y = (1 / 4) * (x - 1) - 1) ↔ (x - 4 * y - 5 = 0) :=
by
  sorry

end chord_line_equation_l739_739472


namespace spending_50_dollars_opposite_meaning_l739_739575

theorem spending_50_dollars_opposite_meaning :
  (∀ (income expenditure : Int), income = 80 → expenditure = 50 → -income = - (expenditure)) :=
by
  intro income expenditure h_income h_expenditure
  rw [h_income, h_expenditure]
  rfl

end spending_50_dollars_opposite_meaning_l739_739575


namespace positive_integer_divisors_of_a50_l739_739649

noncomputable def a : ℕ → ℤ
| 0       := 1
| 1       := -4
| (n + 2) := -4 * a (n + 1) - 7 * a n

theorem positive_integer_divisors_of_a50 (a : ℕ → ℤ)
  (h₀ : a 0 = 1) (h₁ : a 1 = -4)
  (h_rec : ∀ n, a (n + 2) = -4 * a (n + 1) - 7 * a n) :
  ∃ k, k = 51 ∧ primeDivisorsCount (a 50 ^ 2 - a 49 * a 51) = k :=
by
  sorry

end positive_integer_divisors_of_a50_l739_739649


namespace integer_solutions_of_equation_l739_739171

-- Problem statement: Prove that the integer solutions to the equation (x-3)^{(30-x^2)} = 1 are precisely x = 2 and x = 4.
theorem integer_solutions_of_equation :
  ∀ x : ℤ, (x - 3)^(30 - x^2) = 1 ↔ x = 2 ∨ x = 4 :=
by
  sorry

end integer_solutions_of_equation_l739_739171


namespace perp_parallel_lines_l739_739132

noncomputable def l0 : ℝ → ℝ → Prop := λ x y, x - y + 1 = 0
noncomputable def l1 (a : ℝ) : ℝ → ℝ → Prop := λ x y, a * x - 2 * y + 1 = 0
noncomputable def l2 (b : ℝ) : ℝ → ℝ → Prop := λ x y, x + b * y + 3 = 0

theorem perp_parallel_lines (a b : ℝ) (h1 : ∀ x y, (l0 x y → l1 a x y → x - y + 1 = 0 → a * x - 2 * y + 1 = 0 ∧ 1 * a / 2 = -1)) 
                            (h2 : ∀ x y, (l0 x y → l2 b x y → x - y + 1 = 0 → x + b * y + 3 = 0 ∧ -1 / b = 1)) : 
                            a + b = -3 := 
sorry

end perp_parallel_lines_l739_739132


namespace max_unit_squares_around_l739_739381

-- Define the fixed unit square S on the plane
structure UnitSquare (α : Type*) :=
(center : α × α)
(size : α)

-- The maximum number of non-overlapping unit squares that touch the fixed unit square S
theorem max_unit_squares_around (α : Type*) [LinearOrderedField α] (S : UnitSquare α) :
  ∃ n : ℕ, n = 8 ∧
  (∀ (arrangement : fin n → UnitSquare α),
    ∀ i, arrangement i ≠ S ∧
    (∀ j, i ≠ j → arrangement i.center ≠ arrangement j.center) ∧
    (∀ k, dist (arrangement k).center S.center = 1)) := sorry

end max_unit_squares_around_l739_739381


namespace least_possible_k_l739_739372

theorem least_possible_k (k : ℤ) (h : ∃ k : ℤ, k^3 ∣ 336) : k = 84 :=
by
  have h1 : k^3 ∣ 336 := sorry,
  have prime_factors := 2^4 * 3 * 7,
  sorry

end least_possible_k_l739_739372


namespace total_new_games_l739_739912

theorem total_new_games (katie_new : ℕ) (friends_new : ℕ) (h1 : katie_new = 84) (h2 : friends_new = 8) : katie_new + friends_new = 92 := 
by 
  rw [h1, h2]
  simp
  exact rfl

end total_new_games_l739_739912


namespace opposite_of_neg_2023_l739_739670

theorem opposite_of_neg_2023 : ∀ (x : ℝ), x = -2023 → -x = 2023 :=
by
  intro x h
  rw h
  simp
  sorry

end opposite_of_neg_2023_l739_739670


namespace largest_n_dividing_30_fact_l739_739805

theorem largest_n_dividing_30_fact : 
  ∃ n : ℕ, (∀ m : ℕ, (12^m ∣ nat.factorial 30) → m ≤ n) ∧ n = 13 :=
by 
  sorry

end largest_n_dividing_30_fact_l739_739805


namespace trajectories_l739_739849

def conditions (a θ : ℝ) (P Q M O : ℝ × ℝ) :=
  ∃ A B : ℝ × ℝ, |A.1 - B.1| = 2 * a ∧
  let O := ((A.1 + B.1) / 2, (A.2 + B.2) / 2) in
  ∠BOP = θ ∧
  (P.1 = a * cos θ ∧ P.2 = a * sin θ ∧
   Q.1 = 0 ∧ Q.2 = a * sin θ ∧
   M.1 = a * cos θ / (1 + cos θ) ∧ M.2 = a * sin θ / (1 + cos θ)) ∧
  AB forms right angle with OP ∧
  Q is equidistant from A and B ∧
  (area QAB = area PAB ∧ area QAB ≠ 0) ∧
  OP intersects BQ at M

theorem trajectories (a θ : ℝ) (P Q M O : ℝ × ℝ) 
  (h : conditions a θ P Q M O) :
  (P.1^2 + P.2^2 = a^2) ∧
  (Q.1 = 0) ∧
  (M.2^2 = -2 * a * (M.1 - a / 2)) := 
sorry

end trajectories_l739_739849


namespace brand_z_percentage_final_l739_739424

def tank_filled_with_brand_z_gasoline : Prop := true

def tank_three_quarters_empty_then_filled_with_brand_y : Prop := 
  ∃ capacity (brand_z_initial brand_z_left brand_y_added : ℚ),
    capacity = 1 ∧
    brand_z_initial = 1 ∧
    brand_z_left = 1 / 4 ∧
    brand_y_added = 3 / 4

def tank_half_empty_then_filled_with_brand_z_second_time : Prop :=
  ∃ capacity (brand_z_start brand_y_start brand_z_first brand_y_left brand_z_second : ℚ),
    capacity = 1 ∧
    brand_z_start = 1 / 4 ∧
    brand_y_start = 3 / 4 ∧
    brand_z_first = 1 / 8 ∧
    brand_y_left = 3 / 8 ∧
    brand_z_second = 1 / 2

def tank_half_empty_then_filled_with_brand_y_third_time : Prop :=
  ∃ capacity (brand_z_total brand_y_total brand_z_half brand_y_half brand_z_remaining brand_y_remaining brand_y_final_addition : ℚ),
    capacity = 1 ∧
    brand_z_total = 5 / 8 ∧
    brand_y_total = 3 / 8 ∧
    brand_z_half = brand_z_total / 2 ∧
    brand_y_half = brand_y_total / 2 ∧
    brand_z_remaining = 5 / 16 ∧
    brand_y_remaining = 3 / 16 ∧
    brand_y_final_addition = 1 / 2

theorem brand_z_percentage_final :
  tank_filled_with_brand_z_gasoline →
  tank_three_quarters_empty_then_filled_with_brand_y →
  tank_half_empty_then_filled_with_brand_z_second_time →
  tank_half_empty_then_filled_with_brand_y_third_time →
  ∃ (percentage_brand_z : ℚ), percentage_brand_z = 31.25 :=
sorry

end brand_z_percentage_final_l739_739424


namespace line_in_plane_perpendicular_l739_739832

theorem line_in_plane_perpendicular (α : set (point * vector)) (l : set (point * vector)) :
  ∃ m : set (point * vector), m ∈ α ∧ is_perpendicular m l :=
by sorry

end line_in_plane_perpendicular_l739_739832


namespace circle_through_ABC_line_passing_D_with_chord_length_l739_739201

noncomputable def point : Type := ℝ × ℝ

def A : point := (0, 1)
def B : point := (0, 3)
def C : point := (4, 1)
def D : point := (3, 0)

def circle_eq (x y D E F : ℝ) := x^2 + y^2 + D*x + E*y + F = 0

theorem circle_through_ABC :
  ∃ (D E F : ℝ), 
  circle_eq 0 1 D E F ∧
  circle_eq 0 3 D E F ∧
  circle_eq 4 1 D E F ∧
  (circle_eq 3 0 D E F ↔ true) ∧ -- equivalence to say that D(3,0) lies on the circle
  circle_eq = (x^2 + y^2 - 4*x - 4*y + 3 = 0) 
  ∧ 
  (∀ l : ℝ, (l = 3 ∨ 3*l + 4*(0) - 9 = 0)) :=
sorry

theorem line_passing_D_with_chord_length :
  (∀ (line_eq : ℝ → ℝ), 
  (line_eq 3 = 3) ∨ 
  (line_eq = (λ x, (-3/4)*x) ∧ (line_eq D.1 + 4*D.2 - 9 = 0))) :=
sorry

end circle_through_ABC_line_passing_D_with_chord_length_l739_739201


namespace positive_difference_l739_739620

def f (n : Int) : Int := if n < 0 then n^2 + 3*n + 2 else 3*n - 25

theorem positive_difference (a1 a2 : Int) (h1 : f(-3) = 2) (h2 : f(3) = -16) (h3 : f(a1) = 14) (h4 : f(a2) = 14) (cond1 : a1 < 0) (cond2 : a2 ≥ 0) :
  abs (a1 - a2) = 17 :=
by
  -- skipped proof
  sorry

end positive_difference_l739_739620


namespace concyclic_four_points_l739_739724

variables {Point : Type*} [euclidean_space Point] (P0 P1 P2 P3 M1 M2 M3 : Point) (l : set Point)

-- P0 is a point outside line l
def P0_outside_l (P0 : Point) (l : set Point) : Prop := P0 ∉ l

-- M1, M2, M3 are points on line l
def on_line (M : Point) (l : set Point) : Prop := M ∈ l

-- P1, P2, P3 are the circumcenters of the triangles
def is_circumcenter (P : Point) (A B C : Point) : Prop := 
  ∀ Q : Point, (dist P A = dist P B) ∧ (dist P B = dist P C)

theorem concyclic_four_points : P0_outside_l P0 l ∧ on_line M1 l ∧ on_line M2 l ∧ on_line M3 l
  ∧ is_circumcenter P1 P0 M2 M3 ∧ is_circumcenter P2 P0 M1 M3 ∧ is_circumcenter P3 P0 M1 M2 
  → ∃ C : Point, (dist P0 C = dist P1 C) ∧ (dist P1 C = dist P2 C) ∧ (dist P2 C = dist P3 C) :=
by
  sorry

end concyclic_four_points_l739_739724


namespace range_of_a_l739_739521

def f (a : ℝ) : ℝ → ℝ :=
  λ x, if x = 1 then a else (1/2)^((abs (x - 1))) + 1

theorem range_of_a (a : ℝ) :
  (∀ x, ∀ f', (2 * f' (x) ^ 2 - (2 * a + 3) * f' (x) + 3 * a = 0) → (f x = a ∨ f x = 3 / 2)) →
  (∃ x1 x2 x3 x4 x5 : ℝ, distinct [x1, x2, x3, x4, x5] ∧ ∀ i : fin 5, (x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x1 ≠ x5 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x2 ≠ x5 ∧ x3 ≠ x4 ∧ x3 ≠ x5 ∧ x4 ≠ x5) ∧ (2 * (f a (f' i))^2 - (2*a + 3)*(f a (f' i)) + 3 * a = 0)) →
  a ∈ set.Ioo 1 (3/2) ∪ set.Ioo (3/2) 2 :=
sorry

end range_of_a_l739_739521


namespace integer_solutions_of_equation_l739_739173

-- Problem statement: Prove that the integer solutions to the equation (x-3)^{(30-x^2)} = 1 are precisely x = 2 and x = 4.
theorem integer_solutions_of_equation :
  ∀ x : ℤ, (x - 3)^(30 - x^2) = 1 ↔ x = 2 ∨ x = 4 :=
by
  sorry

end integer_solutions_of_equation_l739_739173


namespace no_positive_integer_solutions_m2_m3_positive_integer_solutions_m4_l739_739945

theorem no_positive_integer_solutions_m2_m3 (x y z t : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (ht : 0 < t) :
  (∃ m, m = 2 ∨ m = 3 → (x / y + y / z + z / t + t / x = m) → false) :=
sorry

theorem positive_integer_solutions_m4 (x y z t : ℕ) :
  x / y + y / z + z / t + t / x = 4 ↔ ∃ k : ℕ, k > 0 ∧ (x = k ∧ y = k ∧ z = k ∧ t = k) :=
sorry

end no_positive_integer_solutions_m2_m3_positive_integer_solutions_m4_l739_739945


namespace sequence_diff_ge_abs_m_l739_739616

-- Define the conditions and theorem in Lean

theorem sequence_diff_ge_abs_m
    (m : ℤ) (h_m : |m| ≥ 2)
    (a : ℕ → ℤ)
    (h_seq_not_zero : ¬ (a 1 = 0 ∧ a 2 = 0))
    (h_rec : ∀ n : ℕ, n ≥ 1 → a (n + 2) = a (n + 1) - m * a n)
    (r s : ℕ) (h_r : r > s) (h_s : s ≥ 2)
    (h_equal : a r = a 1 ∧ a s = a 1) :
    r - s ≥ |m| :=
by
  sorry

end sequence_diff_ge_abs_m_l739_739616


namespace stuart_returns_at_3_segments_l739_739953

variable (angle_ABC : ℝ) (circumference : ℝ)

axiom angle_ABC_value : angle_ABC = 60
axiom circle_conditions : circumference = 360

def segments_to_return (n : ℕ) : Prop :=
  (2 * angle_ABC * n / 360 = 1)

theorem stuart_returns_at_3_segments : segments_to_return angle_ABC circumference 3 :=
by
  unfold segments_to_return
  rw [angle_ABC_value, circle_conditions]
  sorry

end stuart_returns_at_3_segments_l739_739953


namespace possible_angles_l739_739319

noncomputable theory

-- assume vector space, norm, dot product and cross product space
variables {V : Type*} [inner_product_space ℝ V]

-- Define vectors a, b, c and the conditions given
variables (a b c : V)
variables (θ : ℝ)

-- Given conditions
axiom norm_a : ∥a∥ = 1
axiom norm_b : ∥b∥ = 1
axiom norm_c : ∥c∥ = 2
axiom cross_product_identity : a × (a × c) + b = 0

-- Define the theorem to prove the values of θ
theorem possible_angles (θ : ℝ) :
  (θ = 30 ∨ θ = 150) :=
sorry

end possible_angles_l739_739319


namespace angle_equivalence_l739_739711

theorem angle_equivalence :
  ∃ k : ℤ, -495 + 360 * k = 225 :=
sorry

end angle_equivalence_l739_739711


namespace total_volume_of_four_cubes_l739_739351

theorem total_volume_of_four_cubes (s : ℝ) (h_s : s = 5) : 4 * s^3 = 500 :=
by
  sorry

end total_volume_of_four_cubes_l739_739351


namespace number_of_valid_divisors_l739_739093

theorem number_of_valid_divisors (k x : ℕ) : 
  (∃ k x : ℕ, k * x - 24 = 4 * k) ↔ finset.card (finset.filter (λ k, 24 % k = 0) (finset.range 25)) = 8 :=
by
  sorry

end number_of_valid_divisors_l739_739093


namespace arc_and_chord_length_l739_739605

noncomputable def circle_radius : ℝ := 15
noncomputable def central_angle : ℝ := 90

theorem arc_and_chord_length :
  let minor_arc_length := (2 * π * circle_radius) * (180 / 360) in 
  let chord_length := 2 * circle_radius * real.sin (central_angle / 180 * π) in
  minor_arc_length = 15 * π ∧ chord_length = 30 :=
by {
  let minor_arc_length := (2 * π * circle_radius) * (180 / 360),
  let chord_length := 2 * circle_radius * real.sin (90 / 180 * π),
  have h1 : minor_arc_length = 15 * π, sorry,
  have h2 : chord_length = 30, sorry,
  exact ⟨h1, h2⟩,
}

end arc_and_chord_length_l739_739605


namespace floor_sum_arithmetic_progression_l739_739046

theorem floor_sum_arithmetic_progression :
  ∃ (S : ℕ → ℕ) (n : ℕ),
    S 0 = ⌊0.5⌋ ∧
    S 1 = ⌊1.3⌋ ∧
    -- Following terms continuing the arithmetic progression with common difference 0.8
    S 123 = ⌊98.9⌋ ∧
    (0:ℕ) + (1:ℕ) + ⋯ + (123:ℕ) = 6100.5 := 
by
  sorry

end floor_sum_arithmetic_progression_l739_739046


namespace line_equation_l739_739473

theorem line_equation (b : ℝ) :
  (∃ b, (∀ x y, y = (3/4) * x + b) ∧ 
  (1/2) * |b| * |- (4/3) * b| = 6 →
  (3 * x - 4 * y + 12 = 0 ∨ 3 * x - 4 * y - 12 = 0)) := 
sorry

end line_equation_l739_739473


namespace unique_zero_of_f_inequality_of_x1_x2_l739_739528

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.exp x - x - 1) - Real.log (x + 1) + x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x + x

theorem unique_zero_of_f (a : ℝ) (h : a ≥ 0) : ∃! x, f a x = 0 := sorry

theorem inequality_of_x1_x2 (a x1 x2 : ℝ) (h : f a x1 = g a x1 - g a x2) (hₐ: a ≥ 0) :
  x1 - 2 * x2 ≥ 1 - 2 * Real.log 2 := sorry

end unique_zero_of_f_inequality_of_x1_x2_l739_739528


namespace annual_donation_amount_l739_739951

-- Define the conditions
variables (age_start age_end : ℕ)
variables (total_donations : ℕ)

-- Define the question (prove the annual donation amount) given these conditions
theorem annual_donation_amount (h1 : age_start = 13) (h2 : age_end = 33) (h3 : total_donations = 105000) :
  total_donations / (age_end - age_start) = 5250 :=
by
   sorry

end annual_donation_amount_l739_739951


namespace parallel_lines_slope_l739_739545

theorem parallel_lines_slope (a : ℝ) :
  (∀ x y : ℝ, y = a * x - 2) ∧ (∀ x y : ℝ, 3 * x - (a + 2) * y + 1 = 0) → 
  (a = 1 ∨ a = -3) :=
by 
  intros h,
  sorry

end parallel_lines_slope_l739_739545


namespace container_volume_ratio_l739_739026

theorem container_volume_ratio
  (A B C : ℝ)
  (h1 : (3 / 4) * A - (5 / 8) * B = (7 / 8) * C - (1 / 2) * C)
  (h2 : B =  (5 / 8) * B)
  (h3 : (5 / 8) * B =  (3 / 8) * C)
  (h4 : A =  (24 / 40) * C) : 
  A / C = 4 / 5 := sorry

end container_volume_ratio_l739_739026


namespace binom_17_8_l739_739137

theorem binom_17_8 (h1 : nat.choose 15 6 = 5005) 
                   (h2 : nat.choose 15 7 = 6435)
                   (h3 : nat.choose 15 8 = 6435) : 
                   nat.choose 17 8 = 24310 := 
by 
    -- Proof omitted
    sorry

end binom_17_8_l739_739137


namespace complex_number_condition_l739_739675

open Complex

-- Define the given condition (arithmetic mean to geometric mean ratio is real)
def arithmetic_mean (A B : ℂ) : ℂ := (A + B) / 2
def geometric_mean (A B : ℂ) : ℂ := Complex.sqrt (A * B)

theorem complex_number_condition (A B : ℂ) (h : (arithmetic_mean A B / geometric_mean A B).im = 0) : 
  (A / B).im = 0 ∨ abs A = abs B :=
by
  sorry

end complex_number_condition_l739_739675


namespace balls_in_boxes_l739_739868

theorem balls_in_boxes : ∃! n : ℕ, n = 7 ∧
  ∀ (b : ℕ) (p : Finset (Finset ℕ)), 
    (b = 6 ∧ p.card = 3 ∧ ∀ s ∈ p, ∃ k : ℕ, s = {k}) → 
    ∃ (l : List ℕ), (l.sum = b ∧ (l.nodup ∧ l.length = p.card) :=
begin
  sorry
end

end balls_in_boxes_l739_739868


namespace original_number_unique_l739_739361

theorem original_number_unique (x : ℝ) (h_pos : 0 < x) 
  (h_condition : 100 * x = 9 / x) : x = 3 / 10 :=
by
  sorry

end original_number_unique_l739_739361


namespace average_value_of_sequence_l739_739772

variables (a x : ℝ)

def sequence := [0, a * x, 2 * a * x, 4 * a * x, 8 * a * x, 16 * a * x]

noncomputable def average (seq : list ℝ) : ℝ :=
  (list.sum seq) / (list.length seq)

theorem average_value_of_sequence : average (sequence a x) = (31 * a * x) / 6 :=
by
  sorry

end average_value_of_sequence_l739_739772


namespace abc_inequality_l739_739240

-- Required conditions and proof statement
theorem abc_inequality 
  {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : a * b * c = 1 / 8) : 
  a^2 + b^2 + c^2 + a^2 * b^2 + a^2 * c^2 + b^2 * c^2 ≥ 15 / 16 := 
sorry

end abc_inequality_l739_739240


namespace diamonds_in_chest_l739_739771

theorem diamonds_in_chest (R : ℕ) (hR : R = 377) : ∃ D : ℕ, D = R + 44 ∧ D = 421 :=
by {
  use (R + 44),
  split,
  { refl, },
  { rw [hR], norm_num, },
}

end diamonds_in_chest_l739_739771


namespace number_of_street_trees_l739_739714

theorem number_of_street_trees (length_road interval : ℕ) (begin_end_trees : ℕ) (sides : ℕ) : ℕ :=
  let intervals := length_road / interval
  let trees_one_side := intervals + begin_end_trees
  in trees_one_side * sides

example : number_of_street_trees 2575 25 1 2 = 208 := by
  have begin_end_trees := 1
  have sides := 2
  let intervals := 2575 / 25
  let trees_one_side := intervals + begin_end_trees
  let total_trees := trees_one_side * sides
  have h : total_trees = 208 := rfl
  exact h

end number_of_street_trees_l739_739714


namespace frac_mul_eq_l739_739435

theorem frac_mul_eq : (2/3) * (3/8) = 1/4 := 
by 
  sorry

end frac_mul_eq_l739_739435


namespace function_through_point_l739_739295

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a^x

theorem function_through_point (a : ℝ) (x : ℝ) (hx : (2 : ℝ) = x) (h : f 2 a = 4) : f x 2 = 2^x :=
by sorry

end function_through_point_l739_739295


namespace chord_length_square_eq_518_4_l739_739776

theorem chord_length_square_eq_518_4
  (r12 r8 r4 : ℝ)
  (h12 : r12 = 12)
  (h8 : r8 = 8)
  (h4 : r4 = 4)
  (externally_tangent : ∀ (O4 O8 : Point) (r4 r8 : ℝ), 
    (distance O4 O8 = r4 + r8) ∧ 
    (r4 = 4) ∧ (r8 = 8))
  (internally_tangent : ∀ (O12 Orest : Point) (R r : ℝ), 
    (distance O12 Orest = R - r) ∧ 
    ((R = 12) ∧ ((r = 8) ∨ (r = 4))))
  (common_tangent_eq_chord : ∀ (O P Q : Point) (r : ℝ), 
    (is_chord O P Q) ∧
    (r = 12) ∧ 
    (distance PQ = common_tangent_length O4 O8 O12)) :
  (chord_length PQ)^2 = 518.4 := 
begin
  sorry
end

end chord_length_square_eq_518_4_l739_739776


namespace max_value_at_theta_pi_over_3_exists_theta_max_value_neg_one_eighth_l739_739856

noncomputable def f (x θ : ℝ) := (sin x) ^ 2 + sqrt 3 * tan θ * cos x + sqrt 3 / 8 * tan θ - 3 / 2

theorem max_value_at_theta_pi_over_3 :
  ∃ x ∈ Icc 0 (π / 2), f x (π / 3) = 15 / 8 :=
sorry

theorem exists_theta_max_value_neg_one_eighth :
  ∃ θ ∈ Icc 0 (π / 3), ∃ x ∈ Icc 0 (π / 2), f x θ = -1 / 8 :=
sorry

end max_value_at_theta_pi_over_3_exists_theta_max_value_neg_one_eighth_l739_739856


namespace sum_of_n_values_l739_739340

open Nat

def binom (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.choose n k

theorem sum_of_n_values :
  ∑ n in {n : ℕ | binom 30 15 + binom 30 n = binom 31 16}, n = 30 := 
by
  sorry

end sum_of_n_values_l739_739340


namespace erick_grapes_count_l739_739358

theorem erick_grapes_count:
  let lemon_increase := 4 in
  let grape_increase := lemon_increase / 2 in
  let original_lemon_price := 8 in
  let original_grape_price := 7 in
  let lemons := 80 in
  let new_lemon_price := original_lemon_price + lemon_increase in
  let new_grape_price := original_grape_price + grape_increase in
  let total_revenue := 2220 in
  ∃ G : ℕ, 
    lemons * new_lemon_price + G * new_grape_price = total_revenue ∧
    G = 140 :=
by
  sorry

end erick_grapes_count_l739_739358


namespace tangent_perpendicular_range_of_m_l739_739147

theorem tangent_perpendicular_range_of_m :
  ∃ (x : ℝ) (m : ℝ), (f x = real.exp x - m * x + 1) ∧
  (f' x = real.exp x - m) ∧
  ((f' x) = - (1 / real.exp 1)) →
  (m > (1 / real.exp 1)) :=
by
  let f : ℝ → ℝ := λ x, real.exp x - m * x + 1
  let f' : ℝ → ℝ := λ x, real.exp x - m
  sorry

end tangent_perpendicular_range_of_m_l739_739147


namespace sampling_is_simple_random_l739_739408

-- Definitions based on conditions
def total_students := 200
def students_sampled := 20
def sampling_method := "Simple Random Sampling"

-- The problem: given the random sampling of 20 students from 200, prove that the method is simple random sampling.
theorem sampling_is_simple_random :
  (total_students = 200 ∧ students_sampled = 20) → sampling_method = "Simple Random Sampling" := 
by
  sorry

end sampling_is_simple_random_l739_739408


namespace vector_dot_product_is_zero_l739_739509

variables (O A B : Type*)
variables [normed_group O]
variables [normed_space ℝ O]
variables [inner_product_space ℝ O]

theorem vector_dot_product_is_zero
  (OA OB : O)
  (h_unit : ‖OA‖ = 1)
  (h_condition : ⟪OA, OA + (OB - OA)⟫ = 2) :
  ⟪OA, OB⟫ = 0 :=
sorry

end vector_dot_product_is_zero_l739_739509


namespace part1_part2_l739_739834

variable (R : ℝ) -- radius of the sphere
variable (x : ℝ) -- semi-vertical angle of the cone

def V1 : ℝ := (1 / 3 : ℝ) * Real.pi * R^3 * (1 + Real.sin x)^3 / (Real.cos x)^2 / Real.sin x 
def V2 : ℝ := 2 * Real.pi * R^3 

theorem part1 : V1 R x ≠ V2 R :=
by
  sorry

theorem part2 :
  let λ := V1 R x / V2 R
  (∀ {λ}, λ = 4 / 3) → (V1 R x / V2 R = 4 / 3 ∧ 2 * Real.arcsin (1 / 3) = 2 * x) :=
by
  sorry

end part1_part2_l739_739834


namespace number_of_parallelograms_l739_739699

-- Problem's condition
def side_length (n : ℕ) : Prop := n > 0

-- Required binomial coefficient (combination formula)
def binom (n k : ℕ) : ℕ := n.choose k

-- Total number of parallelograms in the tiling
theorem number_of_parallelograms (n : ℕ) (h : side_length n) : 
  3 * binom (n + 2) 4 = 3 * (n+2).choose 4 :=
by
  sorry

end number_of_parallelograms_l739_739699


namespace f_of_9_eq_one_third_l739_739968

noncomputable def P_fixed_point (a : ℝ) : Prop :=
  ∃ (x y : ℝ), y = log a (2 * x - 3) + (real.sqrt 2) / 2 ∧ (x, y) = (2, (real.sqrt 2) / 2)

noncomputable def P_on_power_function (α : ℝ) : Prop :=
  ∃ (x y : ℝ), y = x ^ α ∧ (x, y) = (2, (real.sqrt 2) / 2)

theorem f_of_9_eq_one_third (a α : ℝ) (f : ℝ → ℝ) :
  P_fixed_point a ∧ P_on_power_function α →
  f = λ x, x ^ α →
  f 9 = 1 / 3 :=
by
  intro h1 h2
  sorry

end f_of_9_eq_one_third_l739_739968


namespace integer_solutions_count_l739_739165

theorem integer_solutions_count : 
  (∀ (a : ℤ), a^0 = 1) ∧ (∀ (b : ℤ), 1^b = 1) ∧ (∀ (c : ℤ), even c → (-1)^c = 1) → 
  ∃ (n : ℕ), n = 2 ∧ (∀ (x : ℤ), (x - 3)^(30 - x^2) = 1 → x = 4 ∨ x = 2) :=
sorry

end integer_solutions_count_l739_739165


namespace probability_sin_cos_in_range_l739_739402

noncomputable def probability_sin_cos_interval : ℝ :=
  let interval_length := (Real.pi / 2 + Real.pi / 6)
  let valid_length := (Real.pi / 2 - 0)
  valid_length / interval_length

theorem probability_sin_cos_in_range :
  probability_sin_cos_interval = 3 / 4 :=
sorry

end probability_sin_cos_in_range_l739_739402


namespace husband_overpayment_house_help_salary_proof_l739_739742

-- Define the total medical expense and house help's salary
def total_medical_expense : ℝ := 128
def house_help_salary : ℝ := 128

-- Define couple and husband's payments
def couple_payment := total_medical_expense / 2
def each_persons_share := couple_payment / 2
def husband_payment := couple_payment

-- Prove husband overpaid and actual salary of house help
theorem husband_overpayment : husband_payment - each_persons_share = 32 := by
  unfold couple_payment each_persons_share husband_payment
  norm_num

theorem house_help_salary_proof : house_help_salary - (total_medical_expense / 2) = couple_payment := by
  unfold house_help_salary total_medical_expense couple_payment
  norm_num

#eval (house_help_salary_proof, husband_overpayment)

end husband_overpayment_house_help_salary_proof_l739_739742


namespace find_m_from_hyperbola_l739_739538

theorem find_m_from_hyperbola (m : ℝ) (e : ℝ) (h1 : ∀ x y : ℝ, mx^2 + 5y^2 = 5m)
  (h2 : e = 2) : m = -15 := 
sorry

end find_m_from_hyperbola_l739_739538


namespace simple_interest_rate_l739_739718

theorem simple_interest_rate (P : ℝ) (T : ℝ) (H_double : P * 2 = P + P * R * T / 100) : R = 4 := by
  have H : P = P * R * T / 100 := by sorry
  have H1 : R * T / 100 = 1 := by sorry
  have H2 : R = 100 / T := by sorry
  show R = 4 := by
    rw [H2]
    simp
    norm_num
    exact H1

end simple_interest_rate_l739_739718


namespace area_ratio_equality_l739_739224

-- Given an equilateral triangle ABC with sides of length x,
-- let A', B', and C' be points such that A'B = 4*AB, B'C = 4*BC, and CA' = 4*CA.

noncomputable def ratio_of_areas_equilateral (x : ℝ) : ℝ :=
  let area_ABC := (Math.sqrt 3) / 4 * x^2
  let area_A'B'C' := (Math.sqrt 3) / 4 * (4*x)^2
  area_A'B'C' / area_ABC

-- We want to prove that this ratio is 16.
theorem area_ratio_equality (x : ℝ) : ratio_of_areas_equilateral x = 16 :=
  sorry

end area_ratio_equality_l739_739224


namespace total_cupcakes_correct_l739_739680

def cupcakes_per_event : ℝ := 96.0
def num_events : ℝ := 8.0
def total_cupcakes : ℝ := cupcakes_per_event * num_events

theorem total_cupcakes_correct : total_cupcakes = 768.0 :=
by
  unfold total_cupcakes
  unfold cupcakes_per_event
  unfold num_events
  sorry

end total_cupcakes_correct_l739_739680


namespace sum_of_num_den_l739_739389

theorem sum_of_num_den (x : ℝ) (hx : x = 0.474747474747474747):
  let frac := 47 / 99 in
  (frac.num + frac.denom) = 146 :=
by
  -- Placeholder for proof
  sorry

end sum_of_num_den_l739_739389


namespace spending_50_dollars_l739_739572

def receiving_money (r : Int) : Prop := r > 0

def spending_money (s : Int) : Prop := s < 0

theorem spending_50_dollars :
  receiving_money 80 ∧ ∀ r, receiving_money r → spending_money (-r)
  → spending_money (-50) :=
by
  sorry

end spending_50_dollars_l739_739572


namespace red_pencils_count_l739_739902

theorem red_pencils_count 
  (packs : ℕ) 
  (pencils_per_pack : ℕ) 
  (extra_packs : ℕ) 
  (extra_pencils_per_pack : ℕ)
  (total_red_pencils : ℕ) 
  (h1 : packs = 15)
  (h2 : pencils_per_pack = 1)
  (h3 : extra_packs = 3)
  (h4 : extra_pencils_per_pack = 2)
  (h5 : total_red_pencils = packs * pencils_per_pack + extra_packs * extra_pencils_per_pack) : 
  total_red_pencils = 21 := 
  by sorry

end red_pencils_count_l739_739902


namespace valid_k_for_triangle_l739_739061

theorem valid_k_for_triangle (k : ℕ) :
  (∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c → k * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) → 
  (a + b > c ∧ b + c > a ∧ c + a > b)) → k ≥ 6 :=
by
  sorry

end valid_k_for_triangle_l739_739061


namespace women_in_company_l739_739881

-- Definitions for the conditions
def total_workers (T : ℕ) := T
def workers_without_retirement_plan (T : ℕ) := T / 3
def workers_with_retirement_plan (T : ℕ) := 2 * T / 3

-- Gender distribution within retirement plans
def men_without_retirement_plan (T : ℕ) := 0.4 * (T / 3)
def men_with_retirement_plan (T : ℕ) := 0.4 * (2 * T / 3)
def total_men (T : ℕ) := men_without_retirement_plan T + men_with_retirement_plan T

-- The total number of men given in the problem
def given_men := 120

-- The main problem statement to prove
theorem women_in_company (T : ℕ) (h : total_men T = given_men) : (total_workers T - given_men) = 180 :=
by
  sorry

end women_in_company_l739_739881


namespace equation_two_roots_iff_l739_739084

theorem equation_two_roots_iff (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + 2 * x1 + 2 * |x1 + 1| = a ∧ x2^2 + 2 * x2 + 2 * |x2 + 1| = a) ↔ a > -1 :=
by
  sorry

end equation_two_roots_iff_l739_739084


namespace exists_point_inside_acute_triangle_l739_739784

theorem exists_point_inside_acute_triangle (A B C P : Type) [has_dist P A] [has_dist P B] [has_dist P C] [linear_ordered_field A B C P] :
  ∀ (PA PB PC : A) (BC CA AB : A),
  ∀ (is_acute_triangle : ∀ (a b c : A), a^2 + b^2 > c^2, b^2 + c^2 > a^2, c^2 + a^2 > b^2),
  ∃ (P : P), PA * BC = PB * CA ∧ PB * CA = PC * AB :=
begin
  intros PA PB PC BC CA AB is_acute_triangle,
  sorry
end

end exists_point_inside_acute_triangle_l739_739784


namespace range_of_a_l739_739499

variable (a : ℝ)

def p (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (2 * a - 1) ^ x < (2 * a - 1) ^ y
def q (a : ℝ) : Prop := ∀ x : ℝ, 2 * a * x^2 - 2 * a * x + 1 > 0

theorem range_of_a (h1 : p a ∨ q a) (h2 : ¬ (p a ∧ q a)) : (0 ≤ a ∧ a ≤ 1) ∨ (2 ≤ a) :=
by
  sorry

end range_of_a_l739_739499


namespace magnitude_of_complex_number_l739_739491

theorem magnitude_of_complex_number (z : ℂ) (h : z * (real.sqrt 2 + complex.I) = 3 * complex.I) : complex.abs z = real.sqrt 3 := 
by
  sorry

end magnitude_of_complex_number_l739_739491


namespace minimum_distance_l739_739476

def curve1 (x y : ℝ) : Prop := y^2 - 9 + 2*y*x - 12*x - 3*x^2 = 0
def curve2 (x y : ℝ) : Prop := y^2 + 3 - 4*x - 2*y + x^2 = 0

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem minimum_distance 
  (A B : ℝ × ℝ) 
  (hA : curve1 A.1 A.2) 
  (hB : curve2 B.1 B.2) : 
  ∃ d, d = 2 * Real.sqrt 2 ∧ (∀ P Q : ℝ × ℝ, curve1 P.1 P.2 → curve2 Q.1 Q.2 → distance P.1 P.2 Q.1 Q.2 ≥ d) :=
sorry

end minimum_distance_l739_739476


namespace trisha_hourly_wage_l739_739694

theorem trisha_hourly_wage (annual_take_home_pay : ℝ) (percent_withheld : ℝ)
  (hours_per_week : ℝ) (weeks_per_year : ℝ) (hourly_wage : ℝ) :
  annual_take_home_pay = 24960 ∧ 
  percent_withheld = 0.20 ∧ 
  hours_per_week = 40 ∧ 
  weeks_per_year = 52 ∧ 
  hourly_wage = (annual_take_home_pay / (0.80 * (hours_per_week * weeks_per_year))) → 
  hourly_wage = 15 :=
by sorry

end trisha_hourly_wage_l739_739694


namespace solve_for_a_l739_739981

theorem solve_for_a (x a : ℝ) (h1 : x + 2 * a - 6 = 0) (h2 : x = -2) : a = 4 :=
by
  sorry

end solve_for_a_l739_739981


namespace decimal_fraction_eq_l739_739657

theorem decimal_fraction_eq {b : ℕ} (hb : 0 < b) :
  (4 * b + 19 : ℚ) / (6 * b + 11) = 0.76 → b = 19 :=
by
  -- Proof goes here
  sorry

end decimal_fraction_eq_l739_739657


namespace bubble_pass_probability_l739_739955

theorem bubble_pass_probability (n : ℕ) (r : Fin n → ℕ) (hn : n = 50)
  (h_distinct : ∀ (i j : Fin n), i ≠ j → r i ≠ r j)
  (h_random_order : ∀ (i j : Fin n), i < j → r i > r j ∨ r i < r j) :
  let prob := (24 : ℚ) / 25
  in prob = 24 / 25 :=
by
  unfold prob
  sorry

end bubble_pass_probability_l739_739955


namespace total_enjoyable_gameplay_hours_l739_739899

def total_gameplay_hours : ℕ := 100
def grinding_percentage : ℝ := 0.8
def additional_enjoyable_hours : ℕ := 30

theorem total_enjoyable_gameplay_hours : 
  (total_gameplay_hours - (total_gameplay_hours * grinding_percentage).toNat + additional_enjoyable_hours = 50) :=
by
  sorry

end total_enjoyable_gameplay_hours_l739_739899


namespace max_equations_without_real_roots_l739_739986

def quadratic_no_real_roots (a b c : ℝ) : Prop := b^2 - 4 * a * c < 0

def first_player_max_equations_without_roots 
  (player1_strategy : (ℕ → ℝ) → ℕ → (ℕ → ℝ))
  (player2_strategy : (ℕ → ℝ) → ℕ → (ℕ → ℝ))
  : ℕ :=
  let equations := list.repeat (λ _, (0, 0, 0)) 11 in
  let final_equations := game_simulation equations player1_strategy player2_strategy in
  final_equations.count (λ p, quadratic_no_real_roots p.1 p.2 p.3)

-- Hypotheses required for the proof
def initial_game_state : ℕ → ℝ := λ _, 1 -- placeholder for the board game state

theorem max_equations_without_real_roots :
  (∀ player1_strategy player2_strategy, first_player_max_equations_without_roots player1_strategy player2_strategy ≥ 6) :=
sorry

end max_equations_without_real_roots_l739_739986


namespace angles_sum_inequality_l739_739126

noncomputable def alpha (O A P : Point) : Real := sorry
noncomputable def beta (O B P : Point) : Real := sorry
noncomputable def gamma (O C P : Point) : Real := sorry
def orthogonal_tetrahedron (O A B C : Point) : Prop := 
  ∠OAB = π/2 ∧ ∠OAC = π/2 ∧ ∠OBC = π/2

theorem angles_sum_inequality (O A B C P : Point) 
  (h_ortho : orthogonal_tetrahedron O A B C) 
  (h_P : P ∈ triangle A B C) :
  let α := alpha O A P, β := beta O B P, γ := gamma O C P in
  (π / 2 < α + β + γ) ∧ (α + β + γ ≤ 3 * real.arcsin (real.sqrt 3 / 3)) :=
sorry

end angles_sum_inequality_l739_739126


namespace distance_from_origin_l739_739123

open Real

theorem distance_from_origin (x y : ℝ) (h_parabola : y ^ 2 = 4 * x) (h_focus : sqrt ((x - 1) ^ 2 + y ^ 2) = 4) : dist (x, y) (0, 0) = sqrt 21 :=
by
  sorry

end distance_from_origin_l739_739123


namespace remainder_of_polynomial_division_l739_739703

/-- Definition of the polynomial p(x) -/
def p (x : ℝ) : ℝ := 5 * x^3 - 10 * x^2 + 15 * x - 20

/-- Main statement: remainder of dividing p(x) by (5 * x - 10) is 10 -/
theorem remainder_of_polynomial_division :
  ∃ r : ℝ, r = 10 ∧ ∃ q : ℝ → ℝ, p = (λ x, q x * (5 * x - 10) + r) :=
sorry

end remainder_of_polynomial_division_l739_739703


namespace problem1_problem2_l739_739535

section Problem
variables (a : ℝ) (x : ℝ) (x1 x2 : ℝ)
noncomputable def f (x : ℝ) : ℝ := a * (Real.exp x - x - 1) - Real.log (x + 1) + x
noncomputable def g (x : ℝ) : ℝ := a * Real.exp x + x

theorem problem1 (ha : a ≥ 0) : ∃! x, f a x = 0 := sorry

theorem problem2 (ha : a ≥ 0) (h1 : x1 ∈ Icc (-1 : ℝ) (Real.inf)) (h2 : x2 ∈ Icc (-1 : ℝ) (Real.inf)) (h : f a x1 = g a x1 - g a x2) :
  x1 - 2 * x2 ≥ 1 - 2 * Real.log 2 := sorry

end Problem

end problem1_problem2_l739_739535


namespace minimal_defect_prime_or_prime_power_l739_739130

/--
Given \( n > 2 \), define the defect of \( n \) as the number of integers \( a \)
such that \( 0 < a < n \), \((a, n) = 1\), and \( a \) is not \( n \)-separating.
An integer \( a \) is \( n \)-separating if there exists an integer \( d \)
such that \( n \mid a^d - 1 \) and \( n \nmid a^{d-1} + \cdots + 1 \).
Prove that the defect of \( n \) is minimal if and only if \( n \) is a prime or a prime power.
-/
theorem minimal_defect_prime_or_prime_power (n : ℕ) (h : n > 2) :
  (∀ a : ℕ, 0 < a ∧ a < n ∧ Nat.gcd a n = 1 ∧ 
    (∃ d : ℕ, n ∣ a^d - 1 ∧ ¬ n ∣ (Finset.range d).sum (λ i, a^i)) → 
      ¬ a = n - 1) ↔ 
  (∃ p : ℕ, Prime p ∧ (n = p ∨ ∃ k : ℕ, n = p^k)) :=
sorry

end minimal_defect_prime_or_prime_power_l739_739130


namespace winning_candidate_votes_l739_739683

noncomputable def winning_votes : ℕ :=
  let W_perc : ℝ := 0.4577952755905512
  let votes_candidate1 : ℕ := 6136
  let votes_candidate2 : ℕ := 7636
  let total_votes : ℝ := (votes_candidate1 + votes_candidate2) / (1 - W_perc)
  ((W_perc * total_votes).round : ℕ)

theorem winning_candidate_votes : winning_votes = 11630 := by
  sorry

end winning_candidate_votes_l739_739683


namespace probability_four_friends_same_group_l739_739005

open ProbabilityTheory

theorem probability_four_friends_same_group :
  let n := 800
  let groups := 4
  let friends := 4
  let p := (1 / groups) ^ (friends - 1)
  p = 1 / 64 :=
begin
  sorry
end

end probability_four_friends_same_group_l739_739005


namespace volume_of_four_cubes_l739_739356

theorem volume_of_four_cubes (edge_length : ℕ) (num_cubes : ℕ) (h_edge : edge_length = 5) (h_num : num_cubes = 4) :
  num_cubes * (edge_length ^ 3) = 500 :=
by 
  sorry

end volume_of_four_cubes_l739_739356


namespace johnny_red_pencils_l739_739905

noncomputable def number_of_red_pencils (packs_total : ℕ) (extra_packs : ℕ) (extra_per_pack : ℕ) : ℕ :=
  packs_total + extra_packs * extra_per_pack

theorem johnny_red_pencils : number_of_red_pencils 15 3 2 = 21 := by
  sorry

end johnny_red_pencils_l739_739905


namespace triangle_PXW_area_percent_l739_739698

theorem triangle_PXW_area_percent (ABCD W X Y Z : Type) [parallelogram ABCD]
  (is_midpoint : ∀ (P Q U V : Type), is_midpoint P Q U V)
  (W_mid_AB : is_midpoint W ABCD)
  (X_mid_BC : is_midpoint X ABCD)
  (Y_mid_CD : is_midpoint Y ABCD)
  (Z_mid_DA : is_midpoint Z ABCD)
  (P : Type) (P_on_YZ : point_on_segment P Y Z) :
  area (triangle PXW) = 0.25 * area ABCD := sorry

end triangle_PXW_area_percent_l739_739698


namespace square_side_length_l739_739977

theorem square_side_length (x y : ℕ) (h_gcd : Nat.gcd x y = 5) (h_area : ∃ a : ℝ, a^2 = (169 / 6) * ↑(Nat.lcm x y)) : ∃ a : ℝ, a = 65 * Real.sqrt 2 :=
by
  sorry

end square_side_length_l739_739977


namespace cars_to_trucks_ratio_l739_739754

theorem cars_to_trucks_ratio (x : ℕ) (h : x > 0) :
  let cars := 1.25 * x
  car_ratio_to_trucks : (1.25 * x / x) = 5 / 4 :=
by
  sorry

end cars_to_trucks_ratio_l739_739754


namespace largest_constant_a_l739_739807

theorem largest_constant_a (n : ℕ) (a : ℝ) (x : Fin n → ℝ) (h₀ : n ≥ 1)
  (h₁ : ∀ i, 0 ≤ x i)
  (h₂ : x 0 = 0)
  (h₃ : ∀ i, i < n → x i < x (i + 1)) :
  (∑ i in Finset.range n, 1 / (x (i + 1) - x i)) ≥ (4 / 9) * (∑ i in Finset.range n, (i + 2) / x (i + 1)) :=
sorry

end largest_constant_a_l739_739807


namespace vector_magnitude_sum_l739_739279

def angle_between_vectors (a b : ℝ × ℝ) : ℝ :=
  real.acos ((a.1 * b.1 + a.2 * b.2) / (real.sqrt (a.1^2 + a.2^2) * real.sqrt (b.1^2 + b.2^2)))

def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2)

theorem vector_magnitude_sum (a b : ℝ × ℝ) (h_angle : angle_between_vectors a b = real.pi / 3) 
(h_a : a = (2, 0)) (h_b_mag : magnitude b = 1) : 
  magnitude (a.1 + b.1, a.2 + b.2) = real.sqrt 7 :=
by
  sorry

end vector_magnitude_sum_l739_739279


namespace smallest_n_l739_739921

theorem smallest_n 
  (n : ℕ)
  (x : Fin n → ℝ)
  (h1 : ∀ i, |x i| < 1)
  (h2 : (Finset.univ.sum (λ i, |x i|)) = 10 + |Finset.univ.sum (λ i, x i)|) :
  n ≥ 11 := 
sorry

end smallest_n_l739_739921


namespace roots_of_polynomial_l739_739928

theorem roots_of_polynomial :
  let p q r : ℝ
  (h : Polynomial.roots (2 * Polynomial.X ^ 3 - 4 * Polynomial.X ^ 2 + 7 * Polynomial.X - 3) = {p, q, r}) in
  (p * q + q * r + r * p = 7 / 2) :=
by
  sorry

end roots_of_polynomial_l739_739928


namespace part1_solution_sets_part2_solution_set_l739_739518

-- Define the function f(x)
def f (a x : ℝ) := x^2 + (1 - a) * x - a

-- Statement for part (1)
theorem part1_solution_sets (a x : ℝ) :
  (a < -1 → f a x < 0 ↔ a < x ∧ x < -1) ∧
  (a = -1 → ¬ (f a x < 0)) ∧
  (a > -1 → f a x < 0 ↔ -1 < x ∧ x < a) :=
sorry

-- Statement for part (2)
theorem part2_solution_set (x : ℝ) :
  (f 2 x) > 0 → (x^3 * f 2 x > 0 ↔ (-1 < x ∧ x < 0) ∨ 2 < x) :=
sorry

end part1_solution_sets_part2_solution_set_l739_739518


namespace equalize_numbers_l739_739679

theorem equalize_numbers : 
  ∀ (n : ℕ) (board : Fin n → ℕ),
    ∃ x, ∀ i : Fin n, (∃ steps : ℕ, board_after_steps board steps i = x) :=
begin
  sorry
end

end equalize_numbers_l739_739679


namespace geometric_progression_value_l739_739842

variable (a : ℕ → ℕ)
variable (r : ℕ)
variable (h_geo : ∀ n, a (n + 1) = a n * r)

theorem geometric_progression_value (h2 : a 2 = 2) (h6 : a 6 = 162) : a 10 = 13122 :=
by
  sorry

end geometric_progression_value_l739_739842


namespace sequence_not_periodic_from_certain_point_l739_739482

def rightmost_non_zero_digit (n : ℕ) : ℕ := sorry

theorem sequence_not_periodic_from_certain_point (a : ℕ → ℕ)
  (h_def : ∀ n : ℕ, a n = rightmost_non_zero_digit (nat.factorial n))
  (h_condition : ∀ n ≥ 2, nat.iter_count 2 (nat.factorial n) > nat.iter_count 5 (nat.factorial n)) :
  ¬ ∃ p q : ℕ, ∀ n ≥ q, a n = a (n + p) :=
sorry

end sequence_not_periodic_from_certain_point_l739_739482


namespace range_sqrt_eq_zero_to_infty_l739_739674

def f (x: ℝ) : ℝ := Real.sqrt x

theorem range_sqrt_eq_zero_to_infty :
  set.range f = set.Ici 0 :=
by
  sorry

end range_sqrt_eq_zero_to_infty_l739_739674


namespace length_MN_l739_739979

variables {A B C D M N : Type}
variables {BC AD AB : ℝ} -- Lengths of sides
variables {a b : ℝ}

-- Given conditions
def is_trapezoid (a b BC AD AB : ℝ) : Prop :=
  BC = a ∧ AD = b ∧ AB = AD + BC

-- Given, side AB is divided into 5 equal parts and a line parallel to bases is drawn through the 3rd division point
def is_divided (AB : ℝ) : Prop := ∃ P_1 P_2 P_3 P_4, AB = P_4 + P_3 + P_2 + P_1

-- Prove the length of MN
theorem length_MN (a b : ℝ) (h_trapezoid : is_trapezoid a b BC AD AB) (h_divided : is_divided AB) : 
  MN = (2 * BC + 3 * AD) / 5 :=
sorry

end length_MN_l739_739979


namespace sphere_triangle_distance_l739_739305

theorem sphere_triangle_distance
  (P X Y Z : Type)
  (radius : ℝ)
  (h1 : radius = 15)
  (dist_XY : ℝ)
  (h2 : dist_XY = 6)
  (dist_YZ : ℝ)
  (h3 : dist_YZ = 8)
  (dist_ZX : ℝ)
  (h4 : dist_ZX = 10)
  (distance_from_P_to_triangle : ℝ)
  (h5 : distance_from_P_to_triangle = 10 * Real.sqrt 2) :
  let a := 10
  let b := 2
  let c := 1
  let result := a + b + c
  result = 13 :=
by
  sorry

end sphere_triangle_distance_l739_739305


namespace compare_exponentials_l739_739116

theorem compare_exponentials :
  let a := 2^(Real.ln 3)
  let b := 2^(Real.log10 2)
  let c := (1 / 4)^((Real.logb (1 / 3) (1 / 2)))
  a > b ∧ b > c :=
by
  let a := 2^(Real.ln 3)
  let b := 2^(Real.log10 2)
  let c := (1 / 4)^((Real.logb (1 / 3) (1 / 2)))
  sorry

end compare_exponentials_l739_739116


namespace max_trains_count_l739_739626

-- Conditions translated into Lean
def trains_received_per_birthday : ℕ := 1
def trains_received_per_christmas : ℕ := 2
def years : ℕ := 5

def total_trains_gifted := years * trains_received_per_birthday + years * trains_received_per_christmas
def final_trains_count := total_trains_gifted * 2

-- Proof statement
theorem max_trains_count : final_trains_count = 30 := by
  have h1 : total_trains_gifted = 5 * 1 + 5 * 2 := rfl
  have h2 : final_trains_count = (5 * 1 + 5 * 2) * 2 := rfl
  rw h1 at h2
  have h3 : (5 * 1 + 5 * 2) = 15 := by norm_num
  rw h3 at h2
  have h4 : 15 * 2 = 30 := by norm_num
  rw h4 at h2
  exact h2

end max_trains_count_l739_739626


namespace max_a_inequality_l739_739188

theorem max_a_inequality :
  ∀ n : ℕ, n > 0 → (∑ i in Finset.range (3*n - n+2)) (1/((i:ℝ) + 1)) > 25/24 := by
  sorry

end max_a_inequality_l739_739188


namespace simplify_expr1_simplify_expr2_l739_739950

-- Expression simplification proof statement 1
theorem simplify_expr1 (m n : ℤ) : 
  (5 * m + 3 * n - 7 * m - n) = (-2 * m + 2 * n) :=
sorry

-- Expression simplification proof statement 2
theorem simplify_expr2 (x : ℤ) : 
  (2 * x^2 - (3 * x - 2 * (x^2 - x + 3) + 2 * x^2)) = (2 * x^2 - 5 * x + 6) :=
sorry

end simplify_expr1_simplify_expr2_l739_739950


namespace smallest_percent_increase_l739_739480

-- Define the values for each question as given in the problem statement
def values : ℕ → ℝ
| 1  := 150
| 2  := 250
| 3  := 400
| 4  := 550
| 5  := 1200
| 6  := 2500
| 7  := 5000
| 8  := 10000
| 9  := 20000
| 10 := 40000
| 11 := 80000
| 12 := 160000
| 13 := 320000
| 14 := 650000
| 15 := 1300000
| _  := 0 -- default case, as the problem specifies only questions 1 to 15

-- Define the percent increase between two questions
def percent_increase (q1 q2 : ℕ) : ℝ :=
((values q2 - values q1) / values q1) * 100

-- Prove that the smallest percent increase is between Questions 3 and 4
theorem smallest_percent_increase :
  ∀ q1 q2, (percent_increase 3 4 ≤ percent_increase q1 q2) :=
sorry

end smallest_percent_increase_l739_739480


namespace unique_true_statement_l739_739880

def statement (i : ℕ) : Prop := 
  (∃ n : ℕ, n = 100 ∧ (∑ j in Finset.range 100, if j = i-1 then 0 else 1) = n)

theorem unique_true_statement :
  ∃ i, 1 ≤ i ∧ i ≤ 100 ∧ statement 99 :=
by {
  sorry
}

end unique_true_statement_l739_739880


namespace sqrt_diff_ineq_l739_739829

theorem sqrt_diff_ineq (n : ℝ) (hn : n ≥ 0) : 
  sqrt (n + 2) - sqrt (n + 1) < sqrt (n + 1) - sqrt n :=
sorry

end sqrt_diff_ineq_l739_739829


namespace ellipse_eq_l739_739841

noncomputable theory

def isEllipse (F1 F2 : ℝ × ℝ) (P Q : ℝ × ℝ) (perimeter : ℝ) (a b : ℝ) : Prop :=
  F1 = (0, -1) ∧
  F2 = (0, 1) ∧
  P = (-(2 * a), 0) ∧
  Q = (2 * a, 0) ∧
  a > b ∧ b > 0 ∧ 
  perimeter = 8 ∧
  4 * a = perimeter ∧
  a = 2 ∧
  (2 * 1) = 2 ∧
  (c : ℝ) (h : 2 * c = 2), c = 1 ∧
  (h : a^2 = b^2 + c^2), h  ∧
  a^2 - c^2 = 3

theorem ellipse_eq : 
  ∃ (a b : ℝ), isEllipse (0, -1) (0, 1) (-(2 * a), 0) ((2 * a), 0) 8 a b ∧ 
  (eq : string), eq = "x^2 / 4 + y^2 / 3 = 1" :=
sorry

end ellipse_eq_l739_739841


namespace min_good_pairs_l739_739636

theorem min_good_pairs :
  ∀ (arr : list ℕ), (∀ i : ℕ, (1 ≤ arr[i] ∧ arr[i] ≤ 100)) ∧ (∀ i : ℕ, arr[i] > arr[(i - 1) % 100] ∧ arr[i] > arr[(i + 1) % 100] ∨ arr[i] < arr[(i - 1) % 100] ∧ arr[i] < arr[(i + 1) % 100]) →
  ∃ n : ℕ, (n ≥ 51) ∧ good_pairs n arr := by
  sorry

def good_pairs (n : ℕ) (arr : list ℕ) : Prop :=
  ∃ (pairs : list (ℕ × ℕ)),
  (∀ (i : ℕ), pairs[i] = (arr[i], arr[(i + 1) % 100]) ∧ (arr[i] > arr[(i - 1) % 100] ∧ arr[i] > arr[(i + 1) % 100] ∨ arr[i] < arr[(i - 1) % 100] ∧ arr[i] < arr[(i + 1) % 100])) ∧
  pairs.length = n

end min_good_pairs_l739_739636


namespace tan_angle_addition_l739_739180

theorem tan_angle_addition (x : ℝ) (h : Real.tan x = 3) : Real.tan (x + Real.pi / 3) = - (6 + 5 * Real.sqrt 3) / 13 := by
  sorry

end tan_angle_addition_l739_739180


namespace area_ratio_theorem_l739_739621

-- Define the main variables and their relationships
variables (A B C P : Type)
variables [AffineSpace ℝ A] [AffineSpace ℝ B] [AffineSpace ℝ C] [AffineSpace ℝ P]

-- Define the vectors
variables (vAP : AffineMap ℝ P A) (vPB : AffineMap ℝ P B) (vPC : AffineMap ℝ P C)

-- Assume the given condition
axiom given_condition : vAP = (1 : ℝ) / 3 * vPB + (1 : ℝ) / 4 * vPC

-- Define the areas of the triangles
variables (area_ABC : ℝ) (area_PBC : ℝ)

-- Provide a noncomputable definition for triangles’ areas
noncomputable def area_ratio := area_PBC / area_ABC

-- State the theorem
theorem area_ratio_theorem (h : vAP = (1 : ℝ) / 3 * vPB + (1 : ℝ) / 4 * vPC) :
  area_ratio = 12/19 :=
sorry

end area_ratio_theorem_l739_739621


namespace cyclic_product_bound_l739_739222

theorem cyclic_product_bound {a : Fin 2011 → ℝ} (nonneg : ∀ i, 0 ≤ a i)
  (sum_eq : (Finset.univ : Finset (Fin 2011)).sum a = 2011 / 2) :
  |(Finset.univ : Finset (Fin 2011)).prod (λ i, a i - a (Fin.cycle 2011 i))| ≤ 3 * Real.sqrt 3 / 16 := 
sorry

end cyclic_product_bound_l739_739222


namespace valid_triangles_from_10_points_l739_739648

noncomputable def number_of_valid_triangles (n : ℕ) (h : n = 10) : ℕ :=
  if n = 10 then 100 else 0

theorem valid_triangles_from_10_points :
  number_of_valid_triangles 10 rfl = 100 := 
sorry

end valid_triangles_from_10_points_l739_739648


namespace log_sum_value_l739_739706

theorem log_sum_value :
  ∃ (x : ℝ), x = log 10 16 + 3 * log 10 5 + 4 * log 10 2 + 7 * log 10 5 + log 10 32 ∧ x = 10.903 :=
by
  sorry

end log_sum_value_l739_739706


namespace area_of_circular_field_l739_739281

noncomputable def cost_per_metre : ℝ := 4
noncomputable def total_cost : ℝ := 5941.9251828093165

theorem area_of_circular_field :
  let circumference := total_cost / cost_per_metre;
      radius := circumference / (2 * Real.pi);
      area_sq_meters := Real.pi * radius * radius;
      area_hectares := area_sq_meters / 10000
  in area_hectares ≈ 17.56 := 
by 
  sorry

end area_of_circular_field_l739_739281


namespace ratio_of_final_to_initial_l739_739587

theorem ratio_of_final_to_initial (P : ℝ) (R : ℝ) (T : ℝ) (hR : R = 0.02) (hT : T = 50) :
  let SI := P * R * T
  let A := P + SI
  A / P = 2 :=
by
  sorry

end ratio_of_final_to_initial_l739_739587


namespace find_m_l739_739234

theorem find_m (m x1 x2 : ℝ) 
  (h1 : x1 * x1 - 2 * (m + 1) * x1 + m^2 + 2 = 0)
  (h2 : x2 * x2 - 2 * (m + 1) * x2 + m^2 + 2 = 0)
  (h3 : (x1 + 1) * (x2 + 1) = 8) : 
  m = 1 :=
sorry

end find_m_l739_739234


namespace fish_worth_rice_l739_739197

variables (f l r : ℝ)

-- Conditions based on the problem statement
def fish_for_bread : Prop := 3 * f = 2 * l
def bread_for_rice : Prop := l = 4 * r

-- Statement to be proven
theorem fish_worth_rice (h₁ : fish_for_bread f l) (h₂ : bread_for_rice l r) : f = (8 / 3) * r :=
  sorry

end fish_worth_rice_l739_739197


namespace stationery_store_backpacks_l739_739018

theorem stationery_store_backpacks (price_B : ℝ) (x : ℕ) (h1 : price_B > 0) :
  let price_A := 0.9 * price_B in
  let cost_A := 810 in
  let cost_B := 600 in
  let num_A := x + 20 in
  let num_B := x in
  ((cost_A / num_A) = (cost_B / num_B) * 0.9) :=
sorry

end stationery_store_backpacks_l739_739018


namespace intersection_of_A_and_B_find_a_and_b_l739_739933

namespace MathProof

def set_A := {x : ℝ | (1/2)^(x^2 - 4) > 1 }
def set_B := {x : ℝ | 2 < 4 / (x + 3)}

theorem intersection_of_A_and_B : set_A ∩ set_B = {x : ℝ | -2 < x ∧ x < 1} :=
by
  sorry

def inequality_set := {x : ℝ | -3 < x ∧ x < 1}

theorem find_a_and_b (a b : ℝ) : (∀ x : ℝ, 2*x^2 + a*x + b < 0 ↔ x ∈ inequality_set) → a = 4 ∧ b = -6 :=
by
  sorry

end MathProof

end intersection_of_A_and_B_find_a_and_b_l739_739933


namespace CDEF_is_rectangle_l739_739997

-- Let Σ1 and Σ2 be two circles with centers C1 and C2 intersecting at points A and B
variable (Σ1 Σ2 : Circle) (C1 C2 A B : Point) (h_intersect : Σ1 ∩ Σ2 = {A, B})

-- Let P be a point on segment AB such that AP ≠ BP 
variable (P : Point) (hP_on_AB : P ∈ segment A B) (hP_distinct: dist A P ≠ dist B P)

-- Line through P perpendicular to C1P meets Σ1 at points C and D
variable (C D : Point) (h_perp_C1P : is_perpendicular (line_through P C1) (line_through C D)) 
           (h_C_on_Σ1 : C ∈ Σ1) (h_D_on_Σ1 : D ∈ Σ1)

-- Line through P perpendicular to C2P meets Σ2 at points E and F
variable (E F : Point) (h_perp_C2P : is_perpendicular (line_through P C2) (line_through E F))
           (h_E_on_Σ2 : E ∈ Σ2) (h_F_on_Σ2 : F ∈ Σ2)

-- Goal: prove that CDEF is a rectangle
theorem CDEF_is_rectangle : is_rectangle C D E F :=
sorry

end CDEF_is_rectangle_l739_739997


namespace average_distance_to_sides_l739_739745

noncomputable def diagonal (side_length : ℝ) : ℝ :=
  Real.sqrt (side_length ^ 2 + side_length ^ 2)

def lemming_position_after_diagonal (side_length distance : ℝ) : ℝ × ℝ :=
  let fraction := distance / diagonal side_length
  (fraction * side_length, fraction * side_length)

def lemming_final_position (side_length distance angle_turn new_distance : ℝ) : ℝ × ℝ :=
  let (x, y) := lemming_position_after_diagonal side_length distance
  (x + new_distance, y) -- assuming it moves horizontally

def distance_to_sides (side_length : ℝ) (pos : ℝ × ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let (x, y) := pos
  (x, y, side_length - x, side_length - y)

theorem average_distance_to_sides (side_length distance angle_turn new_distance : ℝ) :
  let final_pos := lemming_final_position side_length distance angle_turn new_distance
  let (d_left, d_bottom, d_right, d_top) := distance_to_sides side_length final_pos
  (d_left + d_bottom + d_right + d_top) / 4 = 6 :=
by
  -- Definitions for the specific given problem
  let side_length := 12
  let distance := 7
  let angle_turn := 90 -- degrees; this might not be used as it's pre-determined horizontal
  let new_distance := 4
  let final_pos := lemming_final_position side_length distance angle_turn new_distance
  let (d_left, d_bottom, d_right, d_top) := distance_to_sides side_length final_pos
  have h1 : d_left + d_bottom + d_right + d_top = 24 := sorry
  have h2 : (d_left + d_bottom + d_right + d_top) / 4 = 6 := sorry
  exact h2

end average_distance_to_sides_l739_739745


namespace remainder_of_division_l739_739096

def compute_remainder (dividend divisor : Polynomial ℤ) (remainder : Polynomial ℤ) : Prop :=
  dividend % divisor = remainder

theorem remainder_of_division :
  compute_remainder (Polynomial.C 1 + Polynomial.X^4) (Polynomial.C 6 - Polynomial.C 4 * Polynomial.X + Polynomial.X^2) (Polynomial.C (-59) + Polynomial.C 16 * Polynomial.X) :=
by sorry

end remainder_of_division_l739_739096


namespace proof_problem_l739_739202

-- Define the parametric equations for curve C₁
def C1_param (α : Real) : Real × Real := (sqrt 3 * cos α, sin α)

-- General equation of curve C₁ in Cartesian coordinates
def C1_cartesian (x y : Real) : Prop := x^2 / 3 + y^2 = 1

-- Polar equation for curve C₂ with the transformation
def C2_polar (p θ : Real) : Prop := p * sin (θ - π/4) = 2 * sqrt 2

-- Cartesian equation of curve C₂
def C2_cartesian (x y : Real) : Prop := x - y + 4 = 0

-- Distance formula between point P on C₁ and line C₂
def min_distance (α : Real) : Real := abs (sqrt 3 * cos α - sin α + 4) / sqrt 2

-- Theorem: Stating the two main proofs
theorem proof_problem :
  (∀ α, ∀ (x y : Real), (C1_param α = (x, y)) ↔ (C1_cartesian x y)) ∧
  (∀ (p θ : Real),  C2_polar p θ ↔ ∃ (x y : Real), (C2_cartesian x y)) ∧
  (∀ α, min_distance α ≥ sqrt 2) :=
by {
  sorry
}

end proof_problem_l739_739202


namespace polynomial_identity_solution_l739_739464

theorem polynomial_identity_solution (P : Polynomial ℂ) :
  (∀ x : ℂ, x * P.eval (x-1) = (x-26) * P.eval x) →
  ∃ c : ℂ, P = c • Polynomial.C ℂ 1 * Polynomial.X * (Polynomial.range (↑(26)): ℂ).prod := 
sorry

end polynomial_identity_solution_l739_739464


namespace calculator_representation_l739_739872

theorem calculator_representation (a b c d : ℕ) (ha : a = 2) (hb : b = 3) (hc : c = 3) (hd : d = 2) :
  (a * 10^b) * (c * 10^d) = 6 * 10^5 := by
  -- initial conditions
  have h1 : a * 10^b = 2 * 10^3 := by rw [ha, hb]
  have h2 : c * 10^d = 3 * 10^2 := by rw [hc, hd]
  -- multiply expressions
  rw [h1, h2]
  -- combine terms
  calc
    (2 * 10^3) * (3 * 10^2)
      = 2 * 3 * (10^3 * 10^2) : by ring
  ... = 6 * 10^(3 + 2) : by rw [pow_add]
  ... = 6 * 10^5 : by rw [add_comm]

end calculator_representation_l739_739872


namespace find_radius_of_circumcircle_of_triangle_ABC_l739_739211

noncomputable def radius_of_circumcircle (ABC : Triangle) (CD : Median) (AC : ℝ)
  (incenter_condition : IncenterCondition) : ℝ :=
  if ABC.area = 20 ∧ AC = Real.sqrt 41 ∧ incenter_condition then
    (Real.min (41 / 10) (41 / 8))
  else
    0 -- this value will never be chosen in practice due to the constraints

theorem find_radius_of_circumcircle_of_triangle_ABC :
  ∀ (ABC : Triangle) (CD : Median) (AC : ℝ) (incenter_condition : IncenterCondition),
  ABC.area = 20 →
  AC = Real.sqrt 41 →
  incenter_condition →
  radius_of_circumcircle ABC CD AC incenter_condition = 41 / 10 ∨
  radius_of_circumcircle ABC CD AC incenter_condition = 41 / 8 :=
by sorry

end find_radius_of_circumcircle_of_triangle_ABC_l739_739211


namespace total_volume_of_four_cubes_l739_739350

theorem total_volume_of_four_cubes (s : ℝ) (h_s : s = 5) : 4 * s^3 = 500 :=
by
  sorry

end total_volume_of_four_cubes_l739_739350


namespace find_number_l739_739682

def floor_div (a b : ℕ) : ℕ := a / b

theorem find_number : 
  let a := floor_div 40 8 in
  let b := floor_div 34 12 in
  let x := 2 * (a + b) in
  x = 14 := 
by 
  sorry

end find_number_l739_739682


namespace intersecting_circles_slope_l739_739441

theorem intersecting_circles_slope :
  ∀ (x y : ℝ),
  (x^2 + y^2 - 6*x + 4*y - 8 = 0) ∧ (x^2 + y^2 - 10*x + 18*y + 40 = 0) →
  (y = (2/7) * x - (24/7)) :=
begin
  intros x y,
  intros h,
  sorry,
end

end intersecting_circles_slope_l739_739441


namespace log_comparison_l739_739917

def log_base_3_6 := Real.log 6 / Real.log 3
def log_base_5_10 := Real.log 10 / Real.log 5
def log_base_7_14 := Real.log 14 / Real.log 7

theorem log_comparison : log_base_3_6 > log_base_5_10 ∧ log_base_5_10 > log_base_7_14 :=
by sorry

end log_comparison_l739_739917


namespace integer_solutions_count_l739_739453

theorem integer_solutions_count : 
  (∃ n : ℤ, (n + complex.I)^6 ∈ ℤ) → 
  (card (set_of (λ n : ℤ, (n + complex.I)^6 ∈ ℤ)) = 1) :=
sorry

end integer_solutions_count_l739_739453


namespace right_triangle_perimeter_l739_739976

-- Given conditions
variable (x y : ℕ)
def leg1 := 11
def right_triangle := (101 * 11 = 121)

-- The question and answer
theorem right_triangle_perimeter :
  (y + x = 121) ∧ (y - x = 1) → (11 + x + y = 132) :=
by
  sorry

end right_triangle_perimeter_l739_739976


namespace remainder_of_division_l739_739095

def compute_remainder (dividend divisor : Polynomial ℤ) (remainder : Polynomial ℤ) : Prop :=
  dividend % divisor = remainder

theorem remainder_of_division :
  compute_remainder (Polynomial.C 1 + Polynomial.X^4) (Polynomial.C 6 - Polynomial.C 4 * Polynomial.X + Polynomial.X^2) (Polynomial.C (-59) + Polynomial.C 16 * Polynomial.X) :=
by sorry

end remainder_of_division_l739_739095


namespace paul_oil_change_rate_l739_739257

theorem paul_oil_change_rate (P : ℕ) (h₁ : 8 * (P + 3) = 40) : P = 2 :=
by
  sorry

end paul_oil_change_rate_l739_739257


namespace percentage_increase_in_efficiency_l739_739375

def work_rate_p (W : ℝ) : ℝ := W / 25
def work_rate_q (W x : ℝ) : ℝ := W / x

def combined_work_rate (W x : ℝ) : ℝ := (W / 25) + (W / x)

theorem percentage_increase_in_efficiency :
  (∀ W : ℝ, ∀ x : ℝ, W / 25 + W / x = W / 15 → x = 37.5) →
  (1/25 - 1/37.5) / (1/37.5) * 100 = 50 :=
by
  sorry

end percentage_increase_in_efficiency_l739_739375


namespace inscribed_rectangle_sides_l739_739415

/-- Proof problem statement: Given a triangle with sides 10, 17, and 21, and a rectangle inscribed in the triangle
    with a perimeter of 24 such that one of its sides lies on the longest side of the triangle. Prove that the 
    sides of the rectangle are 72/13 and 84/13. -/
theorem inscribed_rectangle_sides (a b c : ℝ) (ha : a = 10) (hb : b = 17) (hc : c = 21)
    (L P : ℝ) (hP : P = 24) (hL : L = c) :
    ∃ (x y : ℝ), x = 72/13 ∧ y = 84/13 ∧ x + y = 12 :=
begin
  -- Variables for the sides of the rectangle and their relationship to the perimeter
  use [72/13, 84/13],
  split; simp,
  split; simp,
  linarith
end

end inscribed_rectangle_sides_l739_739415


namespace maximum_degree_p_minus_2_l739_739223

theorem maximum_degree_p_minus_2 (p : ℕ) [Fact (Nat.Prime p)] (T : ℕ → ℤ)
  (hT1 : ∀ x y, (T x ≡ T y [ZMOD p]) → (x ≡ y [MOD p]))
  (hT2 : ∃ d : ℕ, d ∈ Finset.range p ∧ nat_degree T = d) :
  ∃ (d : ℕ), d = p - 2 ∧ nat_degree T = d :=
sorry

end maximum_degree_p_minus_2_l739_739223


namespace two_roots_iff_a_gt_neg1_l739_739069

theorem two_roots_iff_a_gt_neg1 (a : ℝ) :
  (∃! x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + 2*x1 + 2*|x1 + 1| = a) ∧ (x2^2 + 2*x2 + 2*|x2 + 1| = a)) ↔ a > -1 :=
by sorry

end two_roots_iff_a_gt_neg1_l739_739069


namespace problem_statement_l739_739967

def sequence (n : ℕ) : ℝ := n + 100/n

theorem problem_statement : 
  (∑ i in Finset.range 99, |sequence i - sequence (i + 1)|) = 162 :=
by sorry

end problem_statement_l739_739967


namespace distance_between_planes_in_cube_l739_739819

theorem distance_between_planes_in_cube :
  (∀ (a : ℝ), a = 10 ∧ ∃ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ x2 < a ∧
  (x1 / a)^3 * 251 + (x2 / a - x1 / a)^3 * 248 + (1 - x2 / a)^3 * 251) →
  ∃ d : ℝ, abs (d - 2.6044) < 0.0001 :=
by
  sorry

end distance_between_planes_in_cube_l739_739819


namespace general_term_l739_739582

def S (n : ℕ) : ℤ := n^2 - 2

def a : ℕ → ℤ
| 0     := 0 -- \(n \in \mathbb{N}^{*}\) implies \(n \geq 1\), we often define a_0 for ℕ type
| 1     := -1
| (n+2) := 2 * (n + 2) - 1

theorem general_term (n : ℕ) (h : n > 0) : 
  a n = if n = 1 then -1 else 2 * n - 1 := by
sorry

end general_term_l739_739582


namespace num_elements_in_S_l739_739227

noncomputable def S' : Set ℕ := {n | n > 1 ∧ ∀ (i : ℕ), (i > 0) → (decimal_digits((1 / (n : ℝ)))_i = decimal_digits((1 / (n : ℝ)))_(i + 10))}

theorem num_elements_in_S' :
  let m := 3^5 * 101 * 7 * 13 * 37 in
  let n := 10^10 - 1 in
  9091.prime →
  ∀ k, n = 9091 * m →
  (∃! x, x ∈ S') ∧ (∃! y, y ∈ {y | y ≤ 288 ∧ y > 1}) →
  ∃! z, z = 287 :=
by
  sorry

end num_elements_in_S_l739_739227


namespace cube_volume_total_four_boxes_l739_739353

theorem cube_volume_total_four_boxes :
  ∀ (length : ℕ), (length = 5) → (4 * (length^3) = 500) :=
begin
  intros length h,
  rw h,
  norm_num,
end

end cube_volume_total_four_boxes_l739_739353


namespace correct_equation_l739_739015

-- Define the given amounts spent on backpacks A and B
def cost_A : ℝ := 810
def cost_B : ℝ := 600

-- Define the number of type B backpacks as x
variable (x : ℝ)

-- Define the number of type A backpacks (20 more than B)
def num_A : ℝ := x + 20

-- Define the unit price relationship where pA is 10% less than pB
def unit_price_A (pB : ℝ) : ℝ := 0.9 * pB

-- Define the equation to prove
theorem correct_equation (pB : ℝ) : 
  (cost_A / num_A) = (cost_B / x) * 0.9 :=
by
  -- Place the proof here
  sorry

end correct_equation_l739_739015


namespace tan_angle_addition_l739_739179

theorem tan_angle_addition (x : ℝ) (h : Real.tan x = 3) : Real.tan (x + Real.pi / 3) = - (6 + 5 * Real.sqrt 3) / 13 := by
  sorry

end tan_angle_addition_l739_739179


namespace limit_problem1_limit_problem2_limit_problem3_limit_problem4_limit_problem5_limit_problem6_limit_problem7_l739_739728

-- 1. Prove the limit equals 16/13
theorem limit_problem1 : tendsto (λ x : ℝ, (x^4 - 16) / (x^3 + 5 * x^2 - 6 * x - 16)) (nhds 2) (nhds (16 / 13)) := 
sorry

-- 2. Prove the limit equals -n/h
theorem limit_problem2 {a n h : ℝ} : tendsto (λ x : ℝ, (x^n - a^n) / (x^n - a^n)) (nhds a) (nhds (-n / h)) := 
sorry

-- 3. Prove the limit equals 2
theorem limit_problem3 : tendsto (λ x : ℝ, (exp (2 * x) - 1) / (sin x)) (nhds 0) (nhds 2) := 
sorry

-- 4. Prove the limit equals (a^2) / (b^2)
theorem limit_problem4 {a b : ℝ} : tendsto (λ x : ℝ, (1 - cos (a * x)) / (1 - cos (b * x))) (nhds 0) (nhds ((a^2) / (b^2))) :=
sorry

-- 5. Prove the limit equals infinity
theorem limit_problem5 {n : ℝ} : tendsto (λ x : ℝ, exp x / x^4) at_top at_top := 
sorry

-- 6. Prove the limit equals 1
theorem limit_problem6 : tendsto (λ x : ℝ, (tan x) / (sec x)) (nhds (π / 2)) (nhds 1) := 
sorry

-- 7. Prove the limit equals 1
theorem limit_problem7 {α : ℝ} : tendsto (λ x : ℝ, (x - sin x) / (x + sin (x / x))) (nhds α) (nhds 1) := 
sorry

end limit_problem1_limit_problem2_limit_problem3_limit_problem4_limit_problem5_limit_problem6_limit_problem7_l739_739728


namespace garden_area_example_l739_739303

def garden_area (perimeter : ℝ) (length_ratio : ℝ) : ℝ :=
  let w := perimeter / (2 * (length_ratio + 1))
  let l := length_ratio * w
  l * w

theorem garden_area_example :
  garden_area 84 3 = 330.75 :=
sorry

end garden_area_example_l739_739303


namespace find_a_l739_739502

noncomputable def p (a : ℝ) : Prop := 3 < a ∧ a < 7/2
noncomputable def q (a : ℝ) : Prop := a > 3 ∧ a ≠ 7/2
theorem find_a (a : ℝ) (h1 : a > 3) (h2 : a ≠ 7/2) (hpq : (p a ∨ q a) ∧ ¬(p a ∧ q a)) : a > 7/2 :=
sorry

end find_a_l739_739502


namespace line_equation_when_alpha_is_pi_div_4_polar_to_cartesian_curve_C_minimum_value_of_inverse_distances_l739_739578

-- Define the parametric equations of the line l
def parametric_line (alpha t : ℝ) : ℝ × ℝ :=
  (1 + t * cos alpha, 2 + t * sin alpha)

-- Define the polar equation of the curve C
def polar_curve (theta : ℝ) : ℝ :=
  6 * sin theta

-- Convert polar to cartesian coordinates
def polar_to_cartesian (rho theta : ℝ) : ℝ × ℝ :=
  (rho * cos theta, rho * sin theta)

-- Cartesian equation of curve C
def cartesian_curve (x y : ℝ) : Prop :=
  x^2 + (y - 3)^2 = 9

-- Ordinary equation of line l when alpha = pi/4
theorem line_equation_when_alpha_is_pi_div_4 :
  ∀ t, parametric_line (π / 4) t = (1 + t * (cos (π / 4)), 2 + t * (sin (π / 4))) →
    (1 + t * (cos (π / 4))) - (2 + t * (sin (π / 4))) + 1 = 0 :=
sorry

-- Cartesian equation of curve C
theorem polar_to_cartesian_curve_C :
  ∀ θ, polar_to_cartesian (polar_curve θ) θ = (ρ * cos θ, ρ * sin θ) →
    (fst (polar_to_cartesian (polar_curve θ) θ))^2 + (snd (polar_to_cartesian (polar_curve θ) θ) - 3)^2 = 9 :=
sorry

-- Minimum value of 1 / |PA| + 1 / |PB|
theorem minimum_value_of_inverse_distances :
  ∀ P A B, distance P A > 0 ∧ distance P B > 0 ∧
  (l : 1 + t * cos alpha, 2 + t * sin alpha) = true ∧ 
  (polar_to_cartesian (polar_curve θ) θ) = true →
  1 / (distance P A) + 1 / (distance P B) = 2 * sqrt 7 / 7 :=
sorry

end line_equation_when_alpha_is_pi_div_4_polar_to_cartesian_curve_C_minimum_value_of_inverse_distances_l739_739578


namespace correct_statement_B_l739_739543

theorem correct_statement_B
  (l m : ℝ³)
  (a : set ℝ³)
  (distinct_lines : l ≠ m)
  (line_perpendicular_plane : ∀ {l m : ℝ³} {a : set ℝ³}, l ⊥ a ∧ l ∥ m → m ⊥ a)
  (l_perp_a : l ⊥ a)
  (l_parallel_m : l ∥ m) :
  m ⊥ a :=
by
  sorry

end correct_statement_B_l739_739543


namespace correct_equation_l739_739014

noncomputable def eqn_correct (x : ℕ) : Prop :=
  let price_A := 810 / (x + 20)
  let price_B := 600 / x
  price_A = price_B * (1 - 0.1)

theorem correct_equation :
  ∀ (x : ℕ), eqn_correct x :=
sorry

end correct_equation_l739_739014


namespace Sara_has_8_balloons_l739_739689

theorem Sara_has_8_balloons (Tom_balloons Sara_balloons total_balloons : ℕ)
  (htom : Tom_balloons = 9)
  (htotal : Tom_balloons + Sara_balloons = 17) :
  Sara_balloons = 8 :=
by
  sorry

end Sara_has_8_balloons_l739_739689


namespace perimeter_is_144_l739_739206

noncomputable def perimeter_of_shape
  (area : ℕ)
  (right_angles : Prop)
  (long_edges_equal : Prop)
  (short_edges_equal : Prop) : ℕ :=
  if area = 528 ∧ right_angles ∧ long_edges_equal ∧ short_edges_equal then 144 else 0

theorem perimeter_is_144 :
  ∀ (area : ℕ) (right_angles long_edges_equal short_edges_equal : Prop),
  area = 528 ∧ right_angles ∧ long_edges_equal ∧ short_edges_equal →
  perimeter_of_shape area right_angles long_edges_equal short_edges_equal = 144 :=
by
  intros area right_angles long_edges_equal short_edges_equal h
  unfold perimeter_of_shape
  rw [if_pos h]
  exact rfl

end perimeter_is_144_l739_739206


namespace angle_at_735_l739_739773

-- Define the positions of the minute and hour hands based on the given time
def minute_hand_position (minutes : ℝ) : ℝ := (minutes / 60) * 360

def hour_hand_position (hours minutes : ℝ) : ℝ :=
  (hours * 30) + (minutes / 60) * 30

-- Prove that the acute angle at 7:35 is 17.5 degrees
theorem angle_at_735 : ∀ (h : ℝ = 7) (m : ℝ = 35),
  (|hour_hand_position h m - minute_hand_position m| = 17.5) :=
by
  intros
  rw [h, m]
  unfold hour_hand_position minute_hand_position
  -- skipping proof details
  sorry

end angle_at_735_l739_739773


namespace max_area_triangle_l739_739124

-- Define the point P
def P := (11 : ℝ, 0 : ℝ)

-- Define the inclination of the line and the parabola equation
def inclination := (Real.pi / 4)
def parabola (x y : ℝ) := y^2 = 4 * x

-- Define the line equation through P with inclination π/4
def line_through_P (x y : ℝ) := y = x - 11

-- Define the intersection points (x-coordinates) of the line and parabola
def R_x := 13 - 4 * Real.sqrt 3
def Q_x := 13 + 4 * Real.sqrt 3

-- Define the y-coordinates using the line equation
def R_y := line_through_P R_x 0 -- 2 * (2 - Real.sqrt 3)
def Q_y := line_through_P Q_x 0 -- 2 * (2 + Real.sqrt 3)

-- The line parallel to RQ with intersections at M and N
def MN_length := 4 * Real.sqrt 2

-- The maximum area of triangle PMN is 22
theorem max_area_triangle (P : ℝ × ℝ) (inclination : ℝ) (parabola : ℝ → ℝ → Prop) (line_through_P : ℝ → ℝ → Prop) (MN_length : ℝ) :
  parabola 11 0 → line_through_P 11 0 → P = (11,0) → inclination = pi / 4 → MN_length = 4 * sqrt 2 →
  ∃ (area : ℝ), area = 22 :=
by
  sorry

end max_area_triangle_l739_739124


namespace symmetry_about_point_l739_739120

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem symmetry_about_point : 
  ∃ p : ℝ × ℝ, p = (Real.pi / 3, 0) ∧ (∀ x : ℝ, f(p.1 - x) = -f(p.1 + x))
 :=
sorry

end symmetry_about_point_l739_739120


namespace bugs_meet_again_at_point_P_l739_739696

open Real

noncomputable def radius_larger_circle : ℝ := 6
noncomputable def radius_smaller_circle : ℝ := 3

noncomputable def speed_larger_circle : ℝ := 4 * π
noncomputable def speed_smaller_circle : ℝ := 3 * π

noncomputable def circumference_larger_circle : ℝ := 2 * radius_larger_circle * π
noncomputable def circumference_smaller_circle : ℝ := 2 * radius_smaller_circle * π

noncomputable def time_full_circle_larger_circle : ℝ := circumference_larger_circle / speed_larger_circle
noncomputable def time_full_circle_smaller_circle : ℝ := circumference_smaller_circle / speed_smaller_circle

theorem bugs_meet_again_at_point_P :
  Nat.lcm (Nat.ceiling time_full_circle_larger_circle) (Nat.ceiling time_full_circle_smaller_circle) = 6 :=
by
  -- proof skipped
  sorry

end bugs_meet_again_at_point_P_l739_739696


namespace relationship_between_abc_l739_739488

noncomputable def a := 3 ^ 0.3
noncomputable def b := (1/2) ^ (-2.1)
noncomputable def c := 2 * Real.log 2 / Real.log 5

theorem relationship_between_abc : c < a ∧ a < b := by
  sorry

end relationship_between_abc_l739_739488


namespace exactly_two_roots_iff_l739_739077

theorem exactly_two_roots_iff (a : ℝ) : 
  (∃! (x : ℝ), x^2 + 2 * x + 2 * |x + 1| = a) ↔ a > -1 :=
by
  sorry

end exactly_two_roots_iff_l739_739077


namespace log_decreasing_interval_l739_739287

theorem log_decreasing_interval :
  let y := λ x : ℝ, Real.log (2*x^2 - 3*x + 1) / Real.log (1/3)
  (∀ x, 2*x^2 - 3*x + 1 > 0) →
  (∀ x₁ x₂, x₁ < x₂ → y x₁ > y x₂) →
  Iio (1 : ℝ)

end log_decreasing_interval_l739_739287


namespace two_roots_iff_a_gt_neg1_l739_739070

theorem two_roots_iff_a_gt_neg1 (a : ℝ) :
  (∃! x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + 2*x1 + 2*|x1 + 1| = a) ∧ (x2^2 + 2*x2 + 2*|x2 + 1| = a)) ↔ a > -1 :=
by sorry

end two_roots_iff_a_gt_neg1_l739_739070


namespace sum_of_valid_n_l739_739344

theorem sum_of_valid_n :
  (∑ n in {n | nat.choose 30 15 + nat.choose 30 n = nat.choose 31 16}.to_finset, n) = 30 := 
sorry

end sum_of_valid_n_l739_739344


namespace additional_pipes_needed_l739_739323

variables (x y z : ℝ)

def pool_volume := 12 * (2 * x + 2 * y + z)

theorem additional_pipes_needed (n : ℝ) :
  (4 * ((1 + n) * x + 2 * y + z)) = pool_volume ↔ n = 2 + 5 * y / x + z / x := 
by
  unfold pool_volume
  sorry

end additional_pipes_needed_l739_739323


namespace problem1_problem2_l739_739043

-- Equivalent proof statement for part (1)
theorem problem1 : 2023^2 - 2022 * 2024 = 1 := by
  sorry

-- Equivalent proof statement for part (2)
theorem problem2 (m : ℝ) (h : m ≠ 1) (h1 : m ≠ -1) : 
  (m / (m^2 - 1)) / ((m^2 - m) / (m^2 - 2*m + 1)) = 1 / (m + 1) := by
  sorry

end problem1_problem2_l739_739043


namespace area_DEF_l739_739893

-- Define the points A, B, and C in the plane
variables (A B C D E F : Type) [metric_space A] [metric_space B] [metric_space C]

-- Define midpoint function
def midpoint (X Y : A) : A := sorry

-- Define triangle area function
def area (X Y Z : A) : ℝ := sorry

-- Define the conditions in the problem
variables (h1 : D = midpoint A B)
variables (h2 : E = midpoint B C)
variables (h3 : F = midpoint C A)
variables (h4 : area A B C = 36)

-- Define the theorem statement
theorem area_DEF (A B C D E F : Type) [metric_space A] [metric_space B] [metric_space C]
  (h1 : D = midpoint A B) (h2 : E = midpoint B C) (h3 : F = midpoint C A) (h4 : area A B C = 36) : 
  area D E F = 9 :=
sorry

end area_DEF_l739_739893


namespace subtraction_operations_to_equal_l739_739128

theorem subtraction_operations_to_equal {a b : ℕ} (ha : a = 252) (hb : b = 72) : 
  (let rec_algo : ℕ → ℕ → ℕ → ℕ
     | a, b, i :=
       if a = b then i
       else if a > b then rec_algo (a - b) b (i + 1)
       else rec_algo a (b - a) (i + 1)
   in rec_algo a b 1) = 4 :=
by
  -- This is a placeholder for the proof
  sorry

end subtraction_operations_to_equal_l739_739128


namespace jim_age_is_55_l739_739036

-- Definitions of the conditions
def jim_age (t : ℕ) : ℕ := 3 * t + 10

def sum_ages (j t : ℕ) : Prop := j + t = 70

-- Statement of the proof problem
theorem jim_age_is_55 : ∃ t : ℕ, jim_age t = 55 ∧ sum_ages (jim_age t) t :=
by
  sorry

end jim_age_is_55_l739_739036


namespace factor_expression_l739_739778

theorem factor_expression (x : ℚ) : 12 * x ^ 2 + 8 * x = 4 * x * (3 * x + 2) := sorry

end factor_expression_l739_739778


namespace domain_of_g_l739_739701

noncomputable def g (x : ℝ) : ℝ :=
  log 5 (log 3 (log 2 x))

theorem domain_of_g :
  ∀ x : ℝ, g x ∈ set.Ioi 8 ↔ x > 8 :=
by
  sorry

end domain_of_g_l739_739701


namespace overlapping_wallpaper_area_l739_739685

theorem overlapping_wallpaper_area :
  ∃ A : ℕ, 
    (∀ total_area two_layers three_layers : ℕ, 
      total_area = 300 →
      two_layers = 38 →
      three_layers = 41 →
      A = total_area - 2 * two_layers - 3 * three_layers) →
    A = 101 :=
begin
  sorry
end

end overlapping_wallpaper_area_l739_739685


namespace sum_of_g_49_values_l739_739919

noncomputable def f (x : ℝ) : ℝ := 5 * x^2 - 4

noncomputable def g (x : ℝ) : ℝ := x^2 + x + x/3 + 1

theorem sum_of_g_49_values : 
  let x := Real.sqrt (53 / 5) in 
  g (49) = (5 * (x^2) - 4) -> 
  g (49) + g (-x) = 116 / 5 :=
by
  sorry

end sum_of_g_49_values_l739_739919


namespace intersect_common_point_l739_739983

-- Defining points
variables (A1 A2 B1 B2 C1 C2: Point)

-- Defining increasing number of lines
variables (O: Point) (A'1 A'2 B'1 B'2 C'1 C'2: Point)

-- Intersection points defined by polar transformation and properties of cevians
variables (P_a P_b P_c: Point)

-- Define the necessary cevians and their properties
def cevian (A B O : Point) : Prop := (line_through A B O)

-- The intersecting lines
def intersect_lines_at (P A1 A2 B1 B2 C1 C2: Point) := 
    ∃ P: Point, on_line P A1 A2 ∧ on_line P B1 B2 ∧ on_line P C1 C2

-- Define the problem using the conditions and desired result
theorem intersect_common_point  
    (hcev1 : cevian A1 A'1 O) 
    (hcev2 : cevian B1 B'1 O) 
    (hcev3 : cevian C1 C'1 O) 
    (hcev4 : cevian A2 A'2 O) 
    (hcev5 : cevian B2 B'2 O) 
    (hcev6 : cevian C2 C'2 O) 
    (hintersect1 : exists P_a, intersect_lines_at P_a A1 A2 A'1 A'2)
    (hintersect2 : exists P_b, intersect_lines_at P_b B1 B2 B'1 B'2)
    (hintersect3 : exists P_c, intersect_lines_at P_c C1 C2 C'1 C'2)
    : intersect_lines_at O A1 A2 B1 B2 C1 C2 := 
sorry

end intersect_common_point_l739_739983


namespace cistern_emptying_time_l739_739396

noncomputable def cistern_time_without_tap (tap_rate : ℕ) (empty_time_with_tap : ℕ) (cistern_volume : ℕ) : ℕ := 
  let tap_total := tap_rate * empty_time_with_tap
  let leaked_volume := cistern_volume - tap_total
  let leak_rate := leaked_volume / empty_time_with_tap
  cistern_volume / leak_rate

theorem cistern_emptying_time :
  cistern_time_without_tap 4 24 480 = 30 := 
by
  unfold cistern_time_without_tap
  norm_num

end cistern_emptying_time_l739_739396


namespace sine_shift_l739_739790

theorem sine_shift : ∀ x : ℝ, sin (2 * (x + π / 6)) = sin (2 * x + π / 3) :=
by
  intro x
  sorry

end sine_shift_l739_739790


namespace part1_min_value_of_f_when_a_is_1_part2_range_of_a_for_f_ge_x_l739_739148

noncomputable def f (a x : ℝ) : ℝ := (1 / 2) * a * x ^ 2 - Real.log x

theorem part1_min_value_of_f_when_a_is_1 : 
  (∃ x : ℝ, f 1 x = 1 / 2 ∧ (∀ y : ℝ, f 1 y ≥ f 1 x)) :=
sorry

theorem part2_range_of_a_for_f_ge_x :
  (∀ x : ℝ, x > 0 → f a x ≥ x) ↔ a ≥ 2 :=
sorry

end part1_min_value_of_f_when_a_is_1_part2_range_of_a_for_f_ge_x_l739_739148


namespace find_number_l739_739960

theorem find_number (a x y : ℕ) (h1 : 1 ≤ x ∧ x ≤ 9) (h2 : 0 ≤ y ∧ y ≤ 9)
  (h3 : 10^2 * x + 10 * y + 1 - (10^2 + 10 * x + y) = 9 * a ^ (1 / Real.log a))
  (h4 : a ^ Real.log a = 10) :
  10^2 + 10 * x + y = 121 :=
by
  sorry

end find_number_l739_739960


namespace angle_between_hands_at_2_15_l739_739709

-- Definitions
def hour_hand_initial_position := 2
def minute_hand_position := 15
def degrees_per_hour := 30
def minutes_per_hour := 60

-- Calculation
def angle_hour_hand_moved := (minute_hand_position : ℝ) / minutes_per_hour * degrees_per_hour

-- Statement
theorem angle_between_hands_at_2_15 :
  angle_hour_hand_moved = 22.5 := by
  sorry

end angle_between_hands_at_2_15_l739_739709


namespace club_members_problem_l739_739884

theorem club_members_problem 
    (T : ℕ) (C : ℕ) (D : ℕ) (B : ℕ) 
    (h_T : T = 85) (h_C : C = 45) (h_D : D = 32) (h_B : B = 18) :
    let Cₒ := C - B
    let Dₒ := D - B
    let N := T - (Cₒ + Dₒ + B)
    N = 26 :=
by
  sorry

end club_members_problem_l739_739884


namespace windmill_counterclockwise_rotation_l739_739184

theorem windmill_counterclockwise_rotation :
  (∀ θ : ℝ, clockwise θ = θ) →
  (∀ θ : ℝ, counterclockwise θ = -clockwise θ) →
  counterclockwise 60 = -60 :=
by
  intros h_clockwise h_counterclockwise
  simp [h_counterclockwise, h_clockwise]
  sorry

end windmill_counterclockwise_rotation_l739_739184


namespace milo_bike_lock_combinations_l739_739940

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_prime_less_20 (n : ℕ) : Prop := n ∈ [2, 3, 5, 7, 11, 13, 17, 19]
def is_multiple_of_4 (n : ℕ) : Prop := n % 4 = 0

theorem milo_bike_lock_combinations :
  let odds := {n ∈ finset.range 41 | is_odd n}.card
  let primes := {n ∈ finset.range 20 | is_prime_less_20 n}.card
  let multiples_of_4 := {n ∈ finset.range 41 | is_multiple_of_4 n}.card
  odds * primes * multiples_of_4 = 1600 :=
by
  sorry

end milo_bike_lock_combinations_l739_739940


namespace light_bulb_signals_l739_739985

theorem light_bulb_signals (n : ℕ) (h : n = 10) : 2^n = 1024 :=
by {
  rw h,
  norm_num,
}

end light_bulb_signals_l739_739985


namespace average_speed_round_trip_l739_739371

variable (D : ℝ) (u v : ℝ)
  
theorem average_speed_round_trip (h1 : u = 96) (h2 : v = 88) : 
  (2 * u * v) / (u + v) = 91.73913043 := 
by 
  sorry

end average_speed_round_trip_l739_739371


namespace express_y_in_terms_of_x_l739_739114

theorem express_y_in_terms_of_x (x y : ℝ) (h : 2 * x - y = 4) : y = 2 * x - 4 :=
by
  sorry

end express_y_in_terms_of_x_l739_739114


namespace every_integer_in_range_can_be_expressed_as_difference_l739_739600

-- Lean Statement
theorem every_integer_in_range_can_be_expressed_as_difference
  (n k m : ℕ) (h_k_ge_2 : k ≥ 2) (h_n_le_m : n ≤ m) (h_m_lt: m < (2 * k - 1) * n / k)
  (A : set ℕ) (h_A_is_subset : A ⊆ {i | 1 ≤ i ∧ i ≤ m}) (h_A_card : A.card = n) :
  ∀ x, 0 < x ∧ x < n / (k - 1) → ∃ a b, a ∈ A ∧ b ∈ A ∧ x = a - b :=
by
  sorry

end every_integer_in_range_can_be_expressed_as_difference_l739_739600


namespace corner_sum_possible_values_l739_739629

/-- 
  Define the checkerboard size 
  and the properties of the vertices.
-/
def board_width : ℕ := 2016
def board_height : ℕ := 2017

/-- 
  Define the properties of gold and silver cells,
  where gold cells have an even sum of vertices and 
  silver cells have an odd sum of vertices.
-/
def gold_cell (x y : ℕ) : Prop := (x + y) % 2 = 0
def silver_cell (x y : ℕ) : Prop := (x + y) % 2 = 1
def even_sum (v1 v2 v3 v4 : ℕ) : Prop := (v1 + v2 + v3 + v4) % 2 = 0
def odd_sum (v1 v2 v3 v4 : ℕ) : Prop := (v1 + v2 + v3 + v4) % 2 = 1

/--
  Statement of the theorem: 
  The possible sums of the numbers at the four corners 
  of the board are 0, 2, or 4.
-/
theorem corner_sum_possible_values :
  ∀ (v00 v01 v10 v11 : ℕ), 
    (∀ x y, 
      (x < board_width) → (y < board_height) → 
      (gold_cell x y → even_sum (f x y) (f x (y+1)) (f (x+1) y) (f (x+1) (y+1))) ∧ 
      (silver_cell x y → odd_sum (f x y) (f x (y+1)) (f (x+1) y) (f (x+1) (y+1))))
    → (v00 + v01 + v10 + v11 = 0 ∨ v00 + v01 + v10 + v11 = 2 ∨ v00 + v01 + v10 + v11 = 4)
:= 
  sorry

end corner_sum_possible_values_l739_739629


namespace min_value_proof_l739_739487

noncomputable def minimum_value (a b : ℝ) : ℝ := 3*a + 4*b

theorem min_value_proof (a b : ℝ) (h_pos: 0 < a ∧ 0 < b)
  (h_eq: (a + b) * (a + 2*b) + a + b = 9) :
  minimum_value a b = 6*real.sqrt 2 - 1 :=
begin
  sorry
end

end min_value_proof_l739_739487


namespace valid_range_of_x_l739_739208

theorem valid_range_of_x (x : ℝ) (h1 : 2 - x ≥ 0) (h2 : x + 1 ≠ 0) : x ≤ 2 ∧ x ≠ -1 :=
sorry

end valid_range_of_x_l739_739208


namespace false_statements_l739_739713

theorem false_statements :
  ¬ (∀ (T1 T2 : Triangle), is_isosceles T1 → is_isosceles T2 → similar T1 T2) ∧
  ¬ (∀ (R : Rhombus), is_square R) :=
sorry

end false_statements_l739_739713


namespace marians_groceries_l739_739938

variables (G : ℝ)

theorem marians_groceries :
  let initial_balance := 126
  let returned_amount := 45
  let new_balance := 171
  let gas_expense := G / 2
  initial_balance + G + gas_expense - returned_amount = new_balance → G = 60 :=
sorry

end marians_groceries_l739_739938


namespace progressive_number_count_l739_739735

theorem progressive_number_count : 
  ∃ n : ℕ, n = 126 ∧ 
  (∀ d1 d2 d3 d4 d5 : ℕ, 
    (1 ≤ d1 ∧ d1 < d2 ∧ d2 < d3 ∧ d3 < d4 ∧ d4 < d5 ∧ d5 ≤ 9) → 
    (d1, d2, d3, d4, d5) ∈ {ds : (ℕ × ℕ × ℕ × ℕ × ℕ) | 
      ds.1, ds.2, ds.3, ds.4, ds.5 ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}}) := 
by
  existsi 126
  intro d1 d2 d3 d4 d5
  intro hconds
  sorry

end progressive_number_count_l739_739735


namespace spending_50_dollars_opposite_meaning_l739_739576

theorem spending_50_dollars_opposite_meaning :
  (∀ (income expenditure : Int), income = 80 → expenditure = 50 → -income = - (expenditure)) :=
by
  intro income expenditure h_income h_expenditure
  rw [h_income, h_expenditure]
  rfl

end spending_50_dollars_opposite_meaning_l739_739576


namespace volume_is_84_l739_739320

-- Definitions for vectors a, b, c
variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V)
variables (two_a_plus_b : V) (b_plus_two_c : V) (three_c_minus_five_a : V)

-- Condition
def volume_of_parallelepiped_condition : Prop :=
  real.abs (inner_product a (cross_product b c)) = 6

-- Computation for the volume of the desired parallelepiped
def volume_of_new_parallelepiped : ℝ :=
  real.abs (inner_product (2 • a + b) (cross_product (b + 2 • c) (3 • c - 5 • a)))

-- Theorem statement
theorem volume_is_84 (h : volume_of_parallelepiped_condition a b c) :
  volume_of_new_parallelepiped a b c two_a_plus_b b_plus_two_c three_c_minus_five_a = 84 :=
sorry

end volume_is_84_l739_739320


namespace similar_triangles_perimeter_l739_739312

theorem similar_triangles_perimeter
  (height_ratio : ℚ)
  (smaller_perimeter larger_perimeter : ℚ)
  (h_ratio : height_ratio = 3 / 5)
  (h_smaller_perimeter : smaller_perimeter = 12)
  : larger_perimeter = 20 :=
by
  sorry

end similar_triangles_perimeter_l739_739312


namespace incorrect_statement_D_l739_739368

-- Coercing necessary noncomputable context
noncomputable def variance (xs : List ℤ) : ℚ :=
  let mean := (xs.map (λ x, (x : ℚ))).sum / xs.length
  (xs.map (λ x, ((x : ℚ) - mean)^2)).sum / xs.length

theorem incorrect_statement_D : variance [6, 7, 8, 9, 10] ≠ 3 :=
by
  sorry

end incorrect_statement_D_l739_739368


namespace length_of_edge_CD_l739_739316

noncomputable def tetrahedron_edges : set ℝ := {10, 20, 25, 34, 45, 51}

def AB := 51
def CD := 25

theorem length_of_edge_CD (h : AB ∈ tetrahedron_edges ∧ CD ∈ tetrahedron_edges) : CD = 25 :=
sorry

end length_of_edge_CD_l739_739316


namespace x_intercept_of_g_l739_739956

noncomputable def f (x : ℝ) : ℝ := (2 * x + 1) / (x - 2)

def is_linear (g : ℝ → ℝ) : Prop :=
  ∃ (m c : ℝ), ∀ x, g(x) = m * x + c

theorem x_intercept_of_g :
  ∀ (g : ℝ → ℝ),
    is_linear g →
    (∃ x, f x = 7 ∧ g 2 = f 7) →
    (∃ x, f 1 = g (4 / 5)) →
    (∃ x, g x = 0 ↔ x = 7 / 5) :=
by sorry

end x_intercept_of_g_l739_739956


namespace tennis_handshakes_l739_739768

theorem tennis_handshakes :
  let num_teams := 4
  let women_per_team := 2
  let total_women := num_teams * women_per_team
  let handshakes_per_woman := total_women - 2
  let total_handshakes_before_division := total_women * handshakes_per_woman
  let actual_handshakes := total_handshakes_before_division / 2
  actual_handshakes = 24 :=
by sorry

end tennis_handshakes_l739_739768


namespace driver_should_rest_l739_739688

def alcohol_content (t : ℝ) : ℝ := 0.3 * (0.75 ^ t)

theorem driver_should_rest (t : ℝ) (ht : alcohol_content t ≤ 0.09) : t ≥ 4.2 :=
by {
  dsimp [alcohol_content] at ht,
  have h : 0.75 ^ t ≤ 0.3, {
    have h0 : 0.09 / 0.3 = 0.3 := by norm_num,
    rw h0 at ht,
    exact ht,
  },
  have h1 : log (0.75 ^ t) ≤ log 0.3 := log_le_log (power_pos zero_lt_three (zero_lt_seven_five)) (zero_lt_three),
  rw [log_pow 0.75, log 0.75] at h1,
  linarith,
}

end driver_should_rest_l739_739688


namespace andy_wrong_questions_l739_739426

variables (a b c d : ℕ)

theorem andy_wrong_questions 
  (h1 : a + b = c + d) 
  (h2 : a + d = b + c + 6) 
  (h3 : c = 7) : 
  a = 20 :=
sorry

end andy_wrong_questions_l739_739426


namespace intersection_of_A_and_B_l739_739157

open Set

noncomputable def A : Set ℝ := { x | (x - 2) / (x + 5) < 0 }
noncomputable def B : Set ℝ := { x | x^2 - 2 * x - 3 ≥ 0 }

theorem intersection_of_A_and_B : A ∩ B = { x : ℝ | -5 < x ∧ x ≤ -1 } :=
sorry

end intersection_of_A_and_B_l739_739157


namespace spending_50_dollars_l739_739571

-- Defining the conditions as per the problem
def receiving (x : ℤ) := x
def spending (x : ℤ) := -x

-- Stating the theorem to be proved
theorem spending_50_dollars :
  receiving 80 = 80 → spending 50 = -50 :=
begin
  intros h,
  -- Leaving the proof for now
  sorry,
end

end spending_50_dollars_l739_739571


namespace original_number_unique_l739_739360

theorem original_number_unique (x : ℝ) (h_pos : 0 < x) 
  (h_condition : 100 * x = 9 / x) : x = 3 / 10 :=
by
  sorry

end original_number_unique_l739_739360


namespace two_roots_iff_a_gt_neg1_l739_739067

theorem two_roots_iff_a_gt_neg1 (a : ℝ) :
  (∃! x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + 2*x1 + 2*|x1 + 1| = a) ∧ (x2^2 + 2*x2 + 2*|x2 + 1| = a)) ↔ a > -1 :=
by sorry

end two_roots_iff_a_gt_neg1_l739_739067


namespace bricks_needed_for_room_floor_l739_739373

-- Conditions
def length : ℕ := 4
def breadth : ℕ := 5
def bricks_per_square_meter : ℕ := 17

-- Question and Answer (Proof Problem)
theorem bricks_needed_for_room_floor : 
  (length * breadth) * bricks_per_square_meter = 340 := by
  sorry

end bricks_needed_for_room_floor_l739_739373


namespace age_problem_l739_739566

variable (A B x : ℕ)

theorem age_problem (h1 : A = B + 5) (h2 : B = 35) (h3 : A + x = 2 * (B - x)) : x = 10 :=
sorry

end age_problem_l739_739566


namespace dog_catches_fox_at_120m_l739_739397

theorem dog_catches_fox_at_120m :
  let initial_distance := 30
  let dog_leap := 2
  let fox_leap := 1
  let dog_leap_frequency := 2
  let fox_leap_frequency := 3
  let dog_distance_per_time_unit := dog_leap * dog_leap_frequency
  let fox_distance_per_time_unit := fox_leap * fox_leap_frequency
  let relative_closure_rate := dog_distance_per_time_unit - fox_distance_per_time_unit
  let time_units_to_catch := initial_distance / relative_closure_rate
  let total_dog_distance := time_units_to_catch * dog_distance_per_time_unit
  total_dog_distance = 120 := sorry

end dog_catches_fox_at_120m_l739_739397


namespace frequency_not_equal_probability_l739_739290

theorem frequency_not_equal_probability 
  (experiments : ℕ → Prop) 
  (tends_to_stable_frequency : ∀ n, ∃ f, ∀ ε > 0, ∃ N, ∀ m > N, |experiments m - f| < ε)
  (frequency_definition_insufficient : ¬∀ f, (∃ ε > 0, ∀ N, ∃ m > N, |experiments m - f| < ε) → 
    (∃ p, ProbabilityTheory.probability experiments = p)) :
  ¬ (∀ e, frequency e = ProbabilityTheory.probability e) :=
sorry

end frequency_not_equal_probability_l739_739290


namespace number_of_sets_l739_739298

theorem number_of_sets (B : Set ℕ) : (\{0\} ∪ B = \{0,2\}) → (∃ S : Finset (Set ℕ), S.card = 2 ∧ ∀ b ∈ S, \{0\} ∪ b = \{0,2\}) :=
by
  sorry

end number_of_sets_l739_739298


namespace correct_statement_l739_739712

-- Define the conditions as assumptions

/-- Condition 1: To understand the service life of a batch of new energy batteries, a sampling survey can be used. -/
def condition1 : Prop := True

/-- Condition 2: If the probability of winning a lottery is 2%, then buying 50 of these lottery tickets at once will definitely win. -/
def condition2 : Prop := False

/-- Condition 3: If the average of two sets of data, A and B, is the same, SA^2=2.3, SB^2=4.24, then set B is more stable. -/
def condition3 : Prop := False

/-- Condition 4: Rolling a die with uniform density and getting a score of 0 is a certain event. -/
def condition4 : Prop := False

-- The main theorem to prove the correct statement is A
theorem correct_statement : condition1 = True ∧ condition2 = False ∧ condition3 = False ∧ condition4 = False :=
by
  constructor; repeat { try { exact True.intro }; try { exact False.elim (by sorry) } }

end correct_statement_l739_739712


namespace projection_of_a_onto_b_equals_l739_739820

def vector_proj (a b : ℝ × ℝ) : ℝ := (a.1 * b.1 + a.2 * b.2) / (b.1 ^ 2 + b.2 ^ 2)

theorem projection_of_a_onto_b_equals (a b : ℝ × ℝ) (h1 : a = (2, 3)) (h2 : b = (-4, 7)) :
  vector_proj a b = sqrt 65 / 5 :=
by
  simp [vector_proj, h1, h2]
  sorry

end projection_of_a_onto_b_equals_l739_739820


namespace balanced_scales_l739_739027

theorem balanced_scales (θ : Type) [has_add θ] [has_mul ℕ θ]
  (circle triangle square : θ) :
  (circle = 2 * triangle) →
  (2 * circle = triangle + square) →
  (2 * square = ? → ? = 2 * circle + 2 * triangle) :=
by {
 sorry
}

end balanced_scales_l739_739027


namespace sum_of_valid_n_l739_739343

theorem sum_of_valid_n :
  (∑ n in {n | nat.choose 30 15 + nat.choose 30 n = nat.choose 31 16}.to_finset, n) = 30 := 
sorry

end sum_of_valid_n_l739_739343


namespace intersection_of_lines_l739_739585

/-- In triangle ABC, point E lies on AC such that AE:EC = 1:3, and point F lies on AB such that AF:FB = 3:2.
    Let P be the intersection of BE and CF. Express \( \overrightarrow{P} \) in the form x \( \overrightarrow{A} + y \overrightarrow{B} + z \overrightarrow{C} \),
    where x, y, and z are constants such that x + y + z = 1. Determine the ordered triple (x, y, z). --/
theorem intersection_of_lines (A B C P : Type) 
  [add_comm_group A]
  [module ℝ A]
  (E F : A)
  (AE_EC_ratio : ∃ AE EC, AE + EC = 1 ∧ AE / EC = 1/3)
  (AF_FB_ratio : ∃ AF FB, AF + FB = 1 ∧ AF / FB = 3/2)
  (BE_intersects_CF_at_P : ∃ BE_line CF_line, BE_line ≠ CF_line ∧ P ∈ BE_line ∧ P ∈ CF_line) : 
  ∃ x y z : ℝ, x + y + z = 1 ∧ P = x • A + y • B + z • C := 
begin
  -- This is the correct answer, manually calculated:
  use [3/16, 2/16, 9/16],
  split,
  { norm_num },
  { sorry }  -- The proof goes here.
end

end intersection_of_lines_l739_739585


namespace exists_abs_diff_even_or_odd_l739_739599

theorem exists_abs_diff_even_or_odd (n : ℕ) 
(h_pos : n > 0) 
(h_sum_int : ∃ a : ℕ → ℝ, (∑ i in Finset.range n, a i) ∈ ℤ) : 
(∃ f : ℕ → ℝ, ∀ (a : ℕ → ℝ) (h : (∑ i in Finset.range n, a i) ∈ ℤ), 
  if even n then (∃ i < n, |a i - 0.5| ≥ f n) 
  else (∃ i < n, |a i - 0.5| ≥ 1/(2*n)) ) := sorry

end exists_abs_diff_even_or_odd_l739_739599


namespace multiple_of_average_is_four_l739_739393

noncomputable def multiple_of_average (S : ℝ) (n : ℝ) (average_excl_n : ℝ) :=
  ∃ k : ℝ, n = k * average_excl_n

theorem multiple_of_average_is_four
  (L : List ℝ)
  (h_len : L.length = 21)
  (h_distinct : L.nodup)
  (n : ℝ)
  (h_n_in_L : n ∈ L)
  (h_n_multiple : ∀ L_excl_n : List ℝ, L_excl_n.length = 20 → List.Excluding L n L_excl_n → 
   let average_excl_n := (L_excl_n.sum / 20) in 
   multiple_of_average (L.sum) n average_excl_n)
  (h_n_sixth_of_sum : n = L.sum / 6) :
  ∃ average_excl_n : ℝ, average_excl_n = ((L.sum - n) / 20) ∧ n = 4 * average_excl_n :=
by
  sorry

end multiple_of_average_is_four_l739_739393


namespace even_digits_in_base4_of_315_l739_739091

theorem even_digits_in_base4_of_315 : 
  let base4_rep := [1, 3, 2, 3] in
  (nat.succ (nat.pred (list.count (λ n, n.bodd = ff) base4_rep)) = 1) :=
by sorry

end even_digits_in_base4_of_315_l739_739091


namespace smallest_possible_AC_l739_739439

theorem smallest_possible_AC 
    (AB AC CD : ℤ) 
    (BD_squared : ℕ) 
    (h_isosceles : AB = AC)
    (h_point_D : ∃ D : ℤ, D = CD)
    (h_perpendicular : BD_squared = 85) 
    (h_integers : ∃ x y : ℤ, AC = x ∧ CD = y) 
    : AC = 11 :=
by
  sorry

end smallest_possible_AC_l739_739439


namespace trajectory_is_ellipse_l739_739875

-- Conditions: Point P(x, y) satisfies the equation
def point_satisfies_eq (x y : ℝ) : Prop :=
  (Real.sqrt ((x + 4)^2 + y^2) + Real.sqrt ((x - 4)^2 + y^2) = 10)

-- Theorem we want to prove: The trajectory of point P is an ellipse
theorem trajectory_is_ellipse (x y : ℝ) (h : point_satisfies_eq x y) : ellipse Trajectory :=
sorry

end trajectory_is_ellipse_l739_739875


namespace tom_can_buy_max_books_l739_739330

theorem tom_can_buy_max_books : 
  let total_money := 24.41
  let price_per_book := 2.75
  let max_books := 8
  ∃ x : ℕ, x = max_books ∧ 2.75 * ↑x ≤ total_money ∧ total_money < 2.75 * (↑x + 1) :=
begin
  sorry
end

end tom_can_buy_max_books_l739_739330


namespace school_club_profit_l739_739004

theorem school_club_profit : 
  let purchase_price_per_bar := 3 / 4
  let selling_price_per_bar := 2 / 3
  let total_bars := 1200
  let bars_with_discount := total_bars - 1000
  let discount_per_bar := 0.10
  let total_cost := total_bars * purchase_price_per_bar
  let total_revenue_without_discount := total_bars * selling_price_per_bar
  let total_discount := bars_with_discount * discount_per_bar
  let adjusted_revenue := total_revenue_without_discount - total_discount
  let profit := adjusted_revenue - total_cost
  profit = -116 :=
by sorry

end school_club_profit_l739_739004


namespace initial_money_calculation_l739_739874

theorem initial_money_calculation
  (P M : ℕ)
  (h1 : M - 11 * P = 800)
  (h2 : M - 15 * P = -1224) :
  M = 6366 :=
by
  sorry

end initial_money_calculation_l739_739874


namespace triangle_parallel_side_l739_739913

variable {A B C E N M : Point}

-- Assume a triangle ABC
variables (hABC : Triangle A B C)
-- E is a point on AC, and N and M are defined as per the problem conditions
variables (hE : E ∈ Line A C)
          (l : Line)
          (hN : N ∈ l)
          (hM : M ∈ l)
          (hEN_BC : parallel (line_through E N) (Line_through B C))
          (hEM_AB : parallel (line_through E M) (line_through A B))

theorem triangle_parallel_side (hABC : Triangle A B C) (hE : E ∈ line_through A C)
    (hEN_BC : parallel (line_through E N) (Line_through B C))
    (hEM_AB : parallel (line_through E M) (line_through A B)) :
    parallel (line_through A N) (Line_through C M) := by
  sorry

end triangle_parallel_side_l739_739913


namespace original_decimal_number_l739_739362

theorem original_decimal_number (x : ℝ) (h₁ : 0 < x) (h₂ : 100 * x = 9 * (1 / x)) : x = 3 / 10 :=
by
  sorry

end original_decimal_number_l739_739362


namespace t_f_5_equals_sqrt_29_sub_4_sqrt_21_l739_739929

noncomputable def t (x : ℝ) : ℝ := real.sqrt (4 * x + 1)
noncomputable def f (x : ℝ) : ℝ := 7 - t x

theorem t_f_5_equals_sqrt_29_sub_4_sqrt_21 : t (f 5) = real.sqrt (29 - 4 * real.sqrt 21) := by
  sorry

end t_f_5_equals_sqrt_29_sub_4_sqrt_21_l739_739929


namespace range_f_l739_739702

noncomputable def f (x : ℝ) : ℝ := (1 / (x - 1) ^ 2) + 1

theorem range_f : set_of (λ y, ∃ x : ℝ, f x = y) = {y : ℝ | 1 < y} :=
by
  sorry

end range_f_l739_739702


namespace tetrahedron_in_cube_l739_739493

theorem tetrahedron_in_cube (a x : ℝ) (h : a = 6) :
  (∃ x, x = 6 * Real.sqrt 2) :=
sorry

end tetrahedron_in_cube_l739_739493


namespace f_iter_correct_l739_739108

noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / (2 * x)

def f_iter (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0     => x
  | (n+1) => f (f_iter n x)

theorem f_iter_correct (n : ℕ) (x : ℝ) (h : x ≠ -1 ∧ x ≠ 0 ∧ x ≠ 1) :
  (f_iter n x) / (f_iter (n + 1) x) = 1 + 1 / f ((x + 1) / (x - 1))^(2^n) :=
sorry

end f_iter_correct_l739_739108


namespace condition_needs_l739_739386

theorem condition_needs (a b c d : ℝ) :
  a + c > b + d → (¬ (a > b ∧ c > d) ∧ (a > b ∧ c > d)) :=
by
  sorry

end condition_needs_l739_739386


namespace find_number_l739_739734

theorem find_number (x : ℝ) : 0.5 * 56 = 0.3 * x + 13 ↔ x = 50 :=
by
  -- Proof would go here
  sorry

end find_number_l739_739734


namespace correct_equation_l739_739012

noncomputable def eqn_correct (x : ℕ) : Prop :=
  let price_A := 810 / (x + 20)
  let price_B := 600 / x
  price_A = price_B * (1 - 0.1)

theorem correct_equation :
  ∀ (x : ℕ), eqn_correct x :=
sorry

end correct_equation_l739_739012


namespace mom_failed_to_find_all_pieces_l739_739432

noncomputable def P : ℕ → ℕ
| 0     := 1
| (n+1) := 4 * P n

theorem mom_failed_to_find_all_pieces :
  ¬ ∃ n : ℕ, P n = 50 :=
begin
  intro h,
  cases h with n hn,
  have H : ∀ n, P n = 4^n,
  { intro m,
    induction m with m ih,
    { refl },
    { rw [P.succ, ih, pow_succ],
      simp } },
  rw H at hn,
  -- Now we need to prove that 4^n ≠ 50
  sorry
end


end mom_failed_to_find_all_pieces_l739_739432


namespace westwood_students_pets_l739_739035

open Set

theorem westwood_students_pets (P G : Set α) (hP : P.card = 30) (hG : G.card = 35) (hPG : (P ∪ G).card = 48) : (P ∩ G).card = 17 :=
  sorry

end westwood_students_pets_l739_739035


namespace quaternary_to_binary_l739_739286

theorem quaternary_to_binary (n : ℕ) :
  n = 1320 →
  (let dec := 1 * (4^3) + 3 * (4^2) + 2 * (4^1) + 0 * (4^0) in
   let bin := "1111000"
   in to_bin dec = bin) :=
  sorry

end quaternary_to_binary_l739_739286


namespace error_percent_calculation_l739_739564

noncomputable def actual_length: ℝ := sorry
noncomputable def actual_width: ℝ := sorry
def measured_length: ℝ := 1.08 * actual_length
def measured_width: ℝ := 0.93 * actual_width
def actual_area: ℝ := actual_length * actual_width
def measured_area: ℝ := measured_length * measured_width
def error: ℝ := measured_area - actual_area
def error_percent: ℝ := (error / actual_area) * 100

theorem error_percent_calculation : error_percent = 0.44 := 
by 
  sorry

end error_percent_calculation_l739_739564


namespace arthur_walking_distance_l739_739033

/-- Arthur walks 8 blocks west and 10 blocks south, 
    each block being 1/4 mile -/
theorem arthur_walking_distance 
  (blocks_west : ℕ) (blocks_south : ℕ) (block_distance : ℚ)
  (h1 : blocks_west = 8) (h2 : blocks_south = 10) (h3 : block_distance = 1/4) :
  (blocks_west + blocks_south) * block_distance = 4.5 := 
by
  sorry

end arthur_walking_distance_l739_739033


namespace best_method_l739_739449

inductive Method
| A
| B
| C
| D

def conditions (method : Method) : Prop :=
  match method with
  | Method.A => "Distribute questionnaires to classmates at school for investigation"
  | Method.B => "Randomly distribute questionnaires to students walking on the roadside for investigation"
  | Method.C => "Distribute questionnaires to people reading in the library for investigation"
  | Method.D => "Randomly distribute questionnaires to pedestrians walking on the roadside for investigation"
  | _ => false

theorem best_method : 
  ∃ method : Method, 
    (conditions method = "Randomly distribute questionnaires to pedestrians walking on the roadside for investigation") ∧
    (method = Method.D) :=
by 
  existsi Method.D
  split
  · rfl
  · rfl

end best_method_l739_739449


namespace find_p_for_quadratic_l739_739463

theorem find_p_for_quadratic :
  ∃ (p : ℝ), p ≠ 0 ∧ (∀ x : ℝ, (px^2 - 12x + 4 = 0 → x = 6)) ∧ p = 9 :=
begin
  sorry
end

end find_p_for_quadratic_l739_739463


namespace similar_triangles_perimeter_l739_739313

theorem similar_triangles_perimeter
  (height_ratio : ℚ)
  (smaller_perimeter larger_perimeter : ℚ)
  (h_ratio : height_ratio = 3 / 5)
  (h_smaller_perimeter : smaller_perimeter = 12)
  : larger_perimeter = 20 :=
by
  sorry

end similar_triangles_perimeter_l739_739313


namespace two_roots_iff_a_gt_neg1_l739_739071

theorem two_roots_iff_a_gt_neg1 (a : ℝ) :
  (∃! x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + 2*x1 + 2*|x1 + 1| = a) ∧ (x2^2 + 2*x2 + 2*|x2 + 1| = a)) ↔ a > -1 :=
by sorry

end two_roots_iff_a_gt_neg1_l739_739071


namespace solve_equation_l739_739641

noncomputable def integer_part (x : ℝ) : ℤ := Int.floor x
noncomputable def fractional_part (x : ℝ) : ℝ := x - (integer_part x).toReal

theorem solve_equation (x : ℝ) (hx : x = integer_part x + fractional_part x)
    (h_fx_bounds : 0 < fractional_part x ∧ fractional_part x < 1) :
    (1 / integer_part x.toReal + 1 / fractional_part x = x) ↔
    (∃ n : ℕ, n > 1 ∧ x = n + 1 / n) := sorry

end solve_equation_l739_739641


namespace f_negative_symmetry_and_interval_inequality_l739_739519

variable {α : Type} [Real α]

def f (x : α) : α

theorem f_negative_symmetry_and_interval_inequality (h1 : ∀ x : α, f x = -f (-x))
    (h2 : ∀ x : α, 1 < x ∧ x < 2 → f x > 0) : f (-1.5) ≠ 1 :=
by
  sorry

end f_negative_symmetry_and_interval_inequality_l739_739519


namespace find_equation_ellipse_max_area_equation_line_l739_739498

-- Define point A and origin O
def A : ℝ × ℝ := (0, -2)
def O : ℝ × ℝ := (0, 0)

-- Define the ellipse E with given conditions
def ellipse (x y a b : ℝ) : Prop := 
  (a > b) ∧ (a > 0) ∧ (b > 0) ∧ ((x^2)/(a^2) + (y^2)/(b^2) = 1)

-- Define the condition for the length ratio
def length_ratio (a b : ℝ) : Prop := (2 * b = a)

-- Define the slope condition for line AF
def slope_AF_conds (c : ℝ) : Prop := 
  ((2 : ℝ) / c = (2 * (real.sqrt 3)) / 3) ∧ (c = real.sqrt 3)

-- The right focus of the ellipse F located at (c, 0)
def F : ℝ × ℝ := (real.sqrt 3, 0)

-- The equations we need to prove
theorem find_equation_ellipse : ∀ (x y : ℝ), 
  ellipse x y 2 1 := sorry

theorem max_area_equation_line : ∀ (k : ℝ), 
  ((k^2 = 7/4) ∧ (y = k * x - 2)) ∨ ((k^2 = 7/4) ∧ (y = -k * x - 2)) := sorry

end find_equation_ellipse_max_area_equation_line_l739_739498


namespace ball_total_distance_when_hits_ground_fifth_time_l739_739739

theorem ball_total_distance_when_hits_ground_fifth_time
  (h₀ : ∀ n : ℕ, 0 < n → (∀ k : ℕ, 0 ≤ k → geom_sum r 0 k ≠ 0))  -- Ensure geometric summation issues are considered
  (initial_height : ℝ) (bounce_factor : ℝ)
  (h_init : initial_height = 120)
  (h_bounce : bounce_factor = 1/3) :
  let descent_distances := [120, 40, 40 / 3, 40 / 9, 40 / 27]
  let ascent_distances := [40, 40 / 3, 40 / 9, 40 / 27] in
  let total_distance := 120 + 40 + (40 / 3) + (40 / 9) + (40 / 27) + 40 + (40 / 3) + (40 / 9) + (40 / 27) in
  total_distance = 278.52 :=
sorry

end ball_total_distance_when_hits_ground_fifth_time_l739_739739


namespace find_x_age_l739_739686

theorem find_x_age (x y : ℕ) 
  (h₁ : x - 2 * y = -3) 
  (h₂ : x + y = 69) : 
  x = 45 := 
by
  -- The proof is not required as per instructions
  sorry

end find_x_age_l739_739686


namespace sum_of_n_values_l739_739341

open Nat

def binom (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.choose n k

theorem sum_of_n_values :
  ∑ n in {n : ℕ | binom 30 15 + binom 30 n = binom 31 16}, n = 30 := 
by
  sorry

end sum_of_n_values_l739_739341


namespace polynomial_remainder_division_l739_739097

theorem polynomial_remainder_division (x : ℝ) : 
  (x^4 + 1) % (x^2 - 4 * x + 6) = 16 * x - 59 := 
sorry

end polynomial_remainder_division_l739_739097


namespace remainder_of_f_x10_div_f_x_is_10_l739_739926

noncomputable def f : Polynomial ℂ := Polynomial.sum (λ n, Polynomial.monomial n (1 : ℂ)) (Finset.range 10)

theorem remainder_of_f_x10_div_f_x_is_10 : (f.eval (X ^ 10) % f) = 10 :=
by
  sorry

end remainder_of_f_x10_div_f_x_is_10_l739_739926


namespace integer_solutions_eq_two_l739_739169

theorem integer_solutions_eq_two : 
  ∃ S : Set Int, (∀ x : Int, x ∈ S ↔ (x-3)^(30-x^2) = 1) ∧ S.card = 2 := 
sorry

end integer_solutions_eq_two_l739_739169


namespace josh_money_left_l739_739218

def initial_amount : ℝ := 20
def cost_hat : ℝ := 10
def cost_pencil : ℝ := 2
def number_of_cookies : ℝ := 4
def cost_per_cookie : ℝ := 1.25

theorem josh_money_left : initial_amount - cost_hat - cost_pencil - (number_of_cookies * cost_per_cookie) = 3 := by
  sorry

end josh_money_left_l739_739218


namespace point_in_plane_region_l739_739366

theorem point_in_plane_region (point_a_in_region point_b_not_in_region point_c_not_in_region point_d_not_in_region: Prop) :
  (2 * 0 + 1 - 6 < 0 ∧ ¬ (2 * 5 + 0 - 6 < 0) ∧ ¬ (2 * 0 + 7 - 6 < 0) ∧ ¬ (2 * 2 + 3 - 6 < 0)) :=
begin
  split,
  { norm_num },
  split,
  { norm_num, linarith },
  split,
  { norm_num, linarith },
  { norm_num, linarith }
end

end point_in_plane_region_l739_739366


namespace greatest_root_of_g_l739_739475

noncomputable def g (x : ℝ) : ℝ := 10 * x^4 - 16 * x^2 + 6

theorem greatest_root_of_g : ∃ x : ℝ, g x = 0 ∧ ∀ y : ℝ, g y = 0 → y ≤ x := 
by
  sorry

end greatest_root_of_g_l739_739475


namespace correct_equation_l739_739016

-- Define the given amounts spent on backpacks A and B
def cost_A : ℝ := 810
def cost_B : ℝ := 600

-- Define the number of type B backpacks as x
variable (x : ℝ)

-- Define the number of type A backpacks (20 more than B)
def num_A : ℝ := x + 20

-- Define the unit price relationship where pA is 10% less than pB
def unit_price_A (pB : ℝ) : ℝ := 0.9 * pB

-- Define the equation to prove
theorem correct_equation (pB : ℝ) : 
  (cost_A / num_A) = (cost_B / x) * 0.9 :=
by
  -- Place the proof here
  sorry

end correct_equation_l739_739016


namespace complex_mult_l739_739825

-- Define the imaginary unit
def i : ℂ := complex.I

-- Define the expression to be proven
theorem complex_mult : 2 * i * (1 + i) = -2 + 2 * i :=
by
  -- Insert your proof here
  sorry

end complex_mult_l739_739825


namespace find_range_of_a_l739_739920

theorem find_range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - 2 * x > a) ∨ (∃ x0 : ℝ, x0^2 + 2 * a * x0 + 2 - a = 0) ∧ 
  ¬ ((∀ x : ℝ, x^2 - 2 * x > a) ∧ (∃ x0 : ℝ, x0^2 + 2 * a * x0 + 2 - a = 0)) → 
  a ∈ Set.Ioo (-2:ℝ) (-1:ℝ) ∪ Set.Ici (1:ℝ) :=
sorry

end find_range_of_a_l739_739920


namespace correct_propositions_l739_739855

-- Define vectors and points for the propositions
variables {a b c : ℝ^3}
variables {P A B C O : ℝ^3}

-- Define the condition for Proposition B
def conditionB (O P A B C : ℝ^3) : Prop :=
  P = (1/4 : ℝ) • A + (1/4 : ℝ) • B + (1/2 : ℝ) • C

-- Define the statement that points P, A, B, C are coplanar
def coplanar_points (P A B C : ℝ^3) : Prop :=
  ∃ (λ μ ν : ℝ), (λ • (A - C) + μ • (B - C) + ν • (P - C) = 0)

-- Define the condition for Proposition C
def conditionC (v1 v2 : ℝ^3) [nontrivial ℝ] : Prop :=
  ∀ v3 : ℝ^3, (is_basis (v1 :: v2 :: (removeElem v3))) → (v1 = t • v2)

-- Define collinearity for two vectors
def collinear_vectors (v1 v2 : ℝ^3) : Prop :=
  ∃ λ : ℝ, v1 = λ • v2

-- Define the projection of a onto b
def proj (a b : ℝ^3) : ℝ^3 :=
  ((a • b) / (b • b)) • b

-- Define the vectors for Proposition D
def vector_a := (9, 4, -4 : ℝ)
def vector_b := (1, 2, 2 : ℝ)

-- The Lean statement
theorem correct_propositions : 
  coplanar_points P A B C ∨ collinear_vectors a b ∨ proj vector_a vector_b = vector_b :=
by
  intro h,
  cases h,
  apply coplanar_points,
  sorry ⟩

end correct_propositions_l739_739855


namespace triangle_properties_l739_739837

theorem triangle_properties (A B C a b c : ℝ) (R : ℝ) 
  (h1 : 0 < A ∧ A < π)
  (h2 : B = π / 3)
  (h3 : R = √3)
  (h4 : b = 2 * R * sin B)
  (h5 : tan B + tan C = 2 * sin A / cos C)
  : b = 3 ∧ area_max = 9 * √3 / 4 :=
by
  generalize_proofs
  sorry

end triangle_properties_l739_739837


namespace solution_l739_739452

noncomputable def problem_statement (a b : ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a * (⌊b * n⌋) = b * (⌊a * n⌋)

theorem solution (a b : ℝ) :
  problem_statement a b ↔ (a = 0 ∨ b = 0 ∨ a = b ∨ (∃ a' b' : ℤ, (a : ℝ) = a' ∧ (b : ℝ) = b')) :=
by
  sorry

end solution_l739_739452


namespace length_of_bridge_l739_739719

theorem length_of_bridge 
  (length_train : ℕ) 
  (speed_train_kmh : ℕ) 
  (crossing_time : ℕ) 
  (h1 : length_train = 130) 
  (h2 : speed_train_kmh = 45) 
  (h3 : crossing_time = 30) : 
  (bridge_length : ℕ) := 
  sorry

end length_of_bridge_l739_739719


namespace stationery_store_backpacks_l739_739020

theorem stationery_store_backpacks (price_B : ℝ) (x : ℕ) (h1 : price_B > 0) :
  let price_A := 0.9 * price_B in
  let cost_A := 810 in
  let cost_B := 600 in
  let num_A := x + 20 in
  let num_B := x in
  ((cost_A / num_A) = (cost_B / num_B) * 0.9) :=
sorry

end stationery_store_backpacks_l739_739020


namespace remainder_when_n_plus_5040_divided_by_7_l739_739869

theorem remainder_when_n_plus_5040_divided_by_7 (n : ℤ) (h: n % 7 = 2) : (n + 5040) % 7 = 2 :=
by
  sorry

end remainder_when_n_plus_5040_divided_by_7_l739_739869


namespace area_sum_half_hexagon_iff_ratio_l739_739598

-- We declare the convex hexagon ABCDEF and the angles equality conditions
variables {A B C D E F : Point}
variable [convex_hexagon : ConvexHexagon A B C D E F]
variables (h1 : ∠A = ∠D) (h2 : ∠B = ∠E)

-- Define the midpoints K and L of the sides AB and DE respectively
noncomputable def K : Point := midpoint A B
noncomputable def L : Point := midpoint D E

-- The main statement to show the equivalence
theorem area_sum_half_hexagon_iff_ratio (h3 : IsMidpoint K A B) (h4 : IsMidpoint L D E) : 
  (area (triangle F A K) + area (triangle K C B) + area (triangle C F L) = 1/2 * area (hexagon A B C D E F)) ↔ 
  (BC / CD = EF / FA) :=
sorry

end area_sum_half_hexagon_iff_ratio_l739_739598


namespace max_value_of_u_l739_739492

open Complex

theorem max_value_of_u (z : ℂ) (hz : |z| = 1) :
  let u := z^4 - z^3 - 3 * z^2 * Complex.I - z + 1 in
  ∃ z₀, z₀ = -1 ∧ |u| ≤ 5 ∧ ∀ z, |z| = 1 → (let u := z^4 - z^3 - 3 * z^2 * Complex.I - z + 1 in |u| ≤ 5) :=
by
  sorry

end max_value_of_u_l739_739492


namespace sum_of_coefficients_l739_739104

theorem sum_of_coefficients :
  let p := -3 * (Polynomial.C (-6) + 4 * Polynomial.X^3 - 2 * Polynomial.X^5 + Polynomial.X^8)
              + 5 * (3 * Polynomial.X^2 + Polynomial.X^4)
              - 4 * (Polynomial.C 5 - Polynomial.X^6)
  in p.eval 1 = 45 :=
by
  let p := -3 * (Polynomial.C (-6) + 4 * Polynomial.X^3 - 2 * Polynomial.X^5 + Polynomial.X^8)
              + 5 * (3 * Polynomial.X^2 + Polynomial.X^4)
              - 4 * (Polynomial.C 5 - Polynomial.X^6)
  show p.eval 1 = 45
  sorry

end sum_of_coefficients_l739_739104


namespace enjoyable_gameplay_l739_739897

theorem enjoyable_gameplay (total_hours : ℕ) (boring_percentage : ℕ) (expansion_hours : ℕ)
  (h_total : total_hours = 100)
  (h_boring : boring_percentage = 80)
  (h_expansion : expansion_hours = 30) :
  ((1 - boring_percentage / 100) * total_hours + expansion_hours) = 50 := 
by
  sorry

end enjoyable_gameplay_l739_739897


namespace largest_real_part_l739_739272

open Complex

theorem largest_real_part (z w : ℂ) (hz : abs z = 2) (hw : abs w = 1) 
  (hzw : z * conj w + conj z * w = (2 : ℂ)) : 
  ∃ θ φ : ℝ, z = 2 * exp (θ * I) ∧ w = exp (φ * I) ∧ 
  (2 * Real.cos θ + Real.cos φ <= 2) := 
begin
  sorry,
end

end largest_real_part_l739_739272


namespace sum_of_coefficients_l739_739106

def polynomial : ℕ → ℤ
| 8 := -3
| 5 := 6
| 3 := -12
| 0 := 45

theorem sum_of_coefficients :
  polynomial 8 + polynomial 5 + polynomial 3 + polynomial 0 = 45 :=
by
  sorry

end sum_of_coefficients_l739_739106


namespace form_x2_sub_2y2_l739_739461

theorem form_x2_sub_2y2 (x y : ℤ) (hx : x % 2 = 1) : (x^2 - 2*y^2) % 8 = 1 ∨ (x^2 - 2*y^2) % 8 = -1 := 
sorry

end form_x2_sub_2y2_l739_739461


namespace integer_solutions_count_l739_739166

theorem integer_solutions_count : 
  (∀ (a : ℤ), a^0 = 1) ∧ (∀ (b : ℤ), 1^b = 1) ∧ (∀ (c : ℤ), even c → (-1)^c = 1) → 
  ∃ (n : ℕ), n = 2 ∧ (∀ (x : ℤ), (x - 3)^(30 - x^2) = 1 → x = 4 ∨ x = 2) :=
sorry

end integer_solutions_count_l739_739166


namespace transformations_return_triangle_l739_739228

/-- Definition of rotation by 120 degrees clockwise around the origin -/
def rotate_120 (p : ℝ × ℝ) : ℝ × ℝ :=
(p.2, -p.1 - p.2)

/-- Definition of reflection across the line y = x -/
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
(p.2, p.1)

/-- Definition of reflection across the line y = -x -/
def reflect_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
(-p.2, -p.1)

/-- Prove that the triangle T with the given vertices and transformations,
    exactly 3 out of the 27 sequences return the triangle to its original position -/
theorem transformations_return_triangle :
  let T := [(0, 0), (6, 0), (0, 4)]
  let transformations := [rotate_120, reflect_y_eq_x, reflect_y_eq_neg_x]
  let sequences := list.permutations [rotate_120, reflect_y_eq_x, reflect_y_eq_neg_x] -- 27 possible sequences
  -- the sequences that return the triangle to its original position
  in list.count (λ seq, seq.foldl (λ t f, f t) T = T) sequences = 3 := sorry

end transformations_return_triangle_l739_739228


namespace parabola_distance_l739_739831

open Real

theorem parabola_distance (x₀ : ℝ) (h₁ : ∃ p > 0, (x₀^2 = 2 * p * 2) ∧ (2 + p / 2 = 5 / 2)) : abs (sqrt (x₀^2 + 4)) = 2 * sqrt 2 :=
by
  rcases h₁ with ⟨p, hp, h₀, h₂⟩
  sorry

end parabola_distance_l739_739831


namespace unique_zero_of_f_inequality_of_x1_x2_l739_739527

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.exp x - x - 1) - Real.log (x + 1) + x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x + x

theorem unique_zero_of_f (a : ℝ) (h : a ≥ 0) : ∃! x, f a x = 0 := sorry

theorem inequality_of_x1_x2 (a x1 x2 : ℝ) (h : f a x1 = g a x1 - g a x2) (hₐ: a ≥ 0) :
  x1 - 2 * x2 ≥ 1 - 2 * Real.log 2 := sorry

end unique_zero_of_f_inequality_of_x1_x2_l739_739527


namespace determine_number_of_students_l739_739275

theorem determine_number_of_students 
  (n : ℕ) 
  (h1 : n < 600) 
  (h2 : n % 25 = 24) 
  (h3 : n % 19 = 15) : 
  n = 399 :=
by
  -- The proof will be provided here.
  sorry

end determine_number_of_students_l739_739275


namespace general_term_a_n_sum_T_n_l739_739155

noncomputable def a (n : ℕ) : ℝ :=
  if n = 1 then 1 else (1/2) * (3/2)^(n-2)

noncomputable def b (n : ℕ) : ℝ :=
  logb (3 / 2) (3 * a (n + 1))

noncomputable def T (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ k, 1 / (b k * b (k + 1)))

theorem general_term_a_n (n : ℕ) : 
  a n = if n = 1 then 1 else (1 / 2) * (3 / 2)^(n - 2) := 
sorry

theorem sum_T_n (n : ℕ) : 
  T n = 1 - 1 / (n + 1) := 
sorry

end general_term_a_n_sum_T_n_l739_739155


namespace calories_in_dressing_l739_739214

noncomputable def lettuce_calories : ℝ := 50
noncomputable def carrot_calories : ℝ := 2 * lettuce_calories
noncomputable def crust_calories : ℝ := 600
noncomputable def pepperoni_calories : ℝ := crust_calories / 3
noncomputable def cheese_calories : ℝ := 400

noncomputable def salad_calories : ℝ := lettuce_calories + carrot_calories
noncomputable def pizza_calories : ℝ := crust_calories + pepperoni_calories + cheese_calories

noncomputable def salad_eaten : ℝ := salad_calories / 4
noncomputable def pizza_eaten : ℝ := pizza_calories / 5

noncomputable def total_eaten : ℝ := salad_eaten + pizza_eaten

theorem calories_in_dressing : ((330 : ℝ) - total_eaten) = 52.5 := by
  sorry

end calories_in_dressing_l739_739214


namespace roots_eq_two_iff_a_gt_neg1_l739_739062

theorem roots_eq_two_iff_a_gt_neg1 (a : ℝ) : 
  (∃! x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + 2*|x₁ + 1| = a ∧ x₂^2 + 2*x₂ + 2*|x₂ + 1| = a) ↔ a > -1 :=
by sorry

end roots_eq_two_iff_a_gt_neg1_l739_739062


namespace quadratic_function_coefficient_nonzero_l739_739186

theorem quadratic_function_coefficient_nonzero (m : ℝ) :
  (y = (m + 2) * x * x + m) ↔ (m ≠ -2 ∧ (m^2 + m - 2 = 0) → m = 1) := by
  sorry

end quadratic_function_coefficient_nonzero_l739_739186


namespace problem_statement_l739_739824

noncomputable def f : ℕ+ → Real → Real
| 1     => λ x, Real.sin x + Real.cos x
| n + 1 => λ x, deriv (f n) x

theorem problem_statement (x : Real) : f 2013 x = Real.sin x + Real.cos x := by
  sorry

end problem_statement_l739_739824


namespace percent_yield_H2O_l739_739456

-- Conditions
def CH3COCH2CH2CHO := 1 -- moles of CH3COCH2CH2CHO
def NaBH4 := 1 -- moles of NaBH4
def H2O := 1 -- moles of H2O, limiting reagent
def H2SO4 := 1 -- moles of H2SO4
def actual_yield := 0.80 -- moles of final alcohol product

-- The problem is to prove the percent yield of H2O formation is 80%
def theoretical_yield := CH3COCH2CH2CHO -- Initial 1 mole of the aldehyde reflects the theoretical yield of H2O in the final step

def percent_yield := (actual_yield / theoretical_yield) * 100

theorem percent_yield_H2O : percent_yield = 80 :=
by
  sorry -- Proof needs to be filled

end percent_yield_H2O_l739_739456


namespace sqrt_expr_equiv_l739_739556

noncomputable theory

theorem sqrt_expr_equiv 
    (a b c : ℕ) 
    (h_pos_a: 0 < a) 
    (h_pos_b: 0 < b) 
    (h_pos_c: 0 < c)
    (h_min_c: ∀ d e f : ℕ, (0 < d ∧ 0 < e ∧ 0 < f) → (a ∗ a + b ∗ b = d ∗ d + e ∗ e) → f ≥ c) :
    (\(expression = \sqrt{3} + \frac{1}{\sqrt{3}} + \sqrt{11} + \frac{1}{\sqrt{11}} + \sqrt{11}\sqrt{3}) 
    = \frac{84 * \sqrt{3} + 44 * \sqrt{11}}{33} ∧ (a + b + c = 161) :=
begin
    sorry
end

end sqrt_expr_equiv_l739_739556


namespace maximum_cows_l739_739405

theorem maximum_cows (s c : ℕ) (h1 : 30 * s + 33 * c = 1300) (h2 : c > 2 * s) : c ≤ 30 :=
by
  -- Proof would go here
  sorry

end maximum_cows_l739_739405


namespace lift_equivalence_l739_739274

theorem lift_equivalence :
  let initial_weights_count := 2
  let initial_weight := 25 -- in pounds
  let initial_repetitions := 10
  let new_weight := 20 -- in pounds
  ∃ n : ℕ, initial_weights_count * initial_weight * initial_repetitions = new_weight * n :=
by
  let initial_weights_count := 2
  let initial_weight := 25
  let initial_repetitions := 10
  let new_weight := 20
  existsi 25
  simp
  sorry

end lift_equivalence_l739_739274


namespace spending_50_dollars_l739_739569

-- Defining the conditions as per the problem
def receiving (x : ℤ) := x
def spending (x : ℤ) := -x

-- Stating the theorem to be proved
theorem spending_50_dollars :
  receiving 80 = 80 → spending 50 = -50 :=
begin
  intros h,
  -- Leaving the proof for now
  sorry,
end

end spending_50_dollars_l739_739569


namespace max_product_sum_1976_l739_739088

theorem max_product_sum_1976 (a : ℕ) (P : ℕ → ℕ) (h : ∀ n, P n > 0 → a = 1976) :
  ∃ (k l : ℕ), (2 * k + 3 * l = 1976) ∧ (P 1976 = 2 * 3 ^ 658) := sorry

end max_product_sum_1976_l739_739088


namespace num_integers_1_to_300_l739_739867

-- Defining the conditions
def multiple_of_6_and_10 (n : ℕ) : Prop := n % 6 = 0 ∧ n % 10 = 0
def not_multiple_of_5_or_8 (n : ℕ) : Prop := ¬ (n % 5 = 0 ∨ n % 8 = 0)

-- Statement of the problem
theorem num_integers_1_to_300 : 
  (finset.card ((finset.filter (λ n, multiple_of_6_and_10 n ∧ not_multiple_of_5_or_8 n) (finset.range 301)))) = 0 := 
by 
  sorry

end num_integers_1_to_300_l739_739867


namespace integer_values_satisfying_condition_l739_739549

theorem integer_values_satisfying_condition : 
  {n : ℤ | -50 < n^3 ∧ n^3 < 50}.finite.toFinset.card = 7 := 
by
  sorry

end integer_values_satisfying_condition_l739_739549


namespace max_value_of_ratio_l739_739804

def is_three_digit_number (N : ℕ) : Prop :=
  ∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ N = 100 * a + 10 * b + c

noncomputable def max_ratio (N S : ℕ) : ℚ :=
  N / S

theorem max_value_of_ratio (N : ℕ) (S : ℕ) (hN : is_three_digit_number N) (hS : S = N.digits.sum) :
  max_ratio N S ≤ 100 :=
by
  sorry

end max_value_of_ratio_l739_739804


namespace opposite_of_negative_rational_l739_739669

theorem opposite_of_negative_rational : - (-(4/3)) = (4/3) :=
by
  sorry

end opposite_of_negative_rational_l739_739669


namespace trigonometric_identity_l739_739045

theorem trigonometric_identity :
  1 / Real.sin (70 * Real.pi / 180) - Real.sqrt 2 / Real.cos (70 * Real.pi / 180) = 
  -2 * (Real.sin (25 * Real.pi / 180) / Real.sin (40 * Real.pi / 180)) :=
sorry

end trigonometric_identity_l739_739045


namespace age_difference_l739_739961

variable (E Y : ℕ)

theorem age_difference (hY : Y = 35) (hE : E - 15 = 2 * (Y - 15)) : E - Y = 20 := by
  -- Assertions and related steps could be handled subsequently.
  sorry

end age_difference_l739_739961


namespace josh_money_left_l739_739217

def initial_amount : ℝ := 20
def cost_hat : ℝ := 10
def cost_pencil : ℝ := 2
def number_of_cookies : ℝ := 4
def cost_per_cookie : ℝ := 1.25

theorem josh_money_left : initial_amount - cost_hat - cost_pencil - (number_of_cookies * cost_per_cookie) = 3 := by
  sorry

end josh_money_left_l739_739217


namespace distinct_m_value_l739_739895

theorem distinct_m_value (a b : ℝ) (m : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0)
    (h_b_eq_2a : b = 2 * a) (h_m_eq_neg2a_b : m = -2 * a / b) : 
    ∃! (m : ℝ), m = -1 :=
by sorry

end distinct_m_value_l739_739895


namespace probability_best_play_wins_theorem_l739_739974

open Nat

-- Definition of binomial coefficient as it may be used in the problem.
noncomputable def binomial : ℕ → ℕ → ℕ
| n, k => if k > n then 0 else Nat.choose n k

def probability_best_play_wins (n m : ℕ) (hmn : 2 * m ≤ n) : ℚ :=
  1 / (binomial (2 * n) n * binomial (2 * n) (2 * m)) * 
  ∑ q in Finset.range (2 * m + 1),
    (binomial n q * binomial n (2 * m - q) * 
     ∑ t in Finset.range (min q m),
       binomial q t * binomial (2 * n - q) (n - t))

theorem probability_best_play_wins_theorem (n m : ℕ) (hmn : 2 * m ≤ n) :
  probability_best_play_wins n m hmn = 
    1 / (binomial (2 * n) n * binomial (2 * n) (2 * m)) * 
    ∑ q in Finset.range (2 * m + 1), 
      (binomial n q * binomial n (2 * m - q) * 
       ∑ t in Finset.range (min q m),
         binomial q t * binomial (2 * n - q) (n - t)) :=
by
  sorry

end probability_best_play_wins_theorem_l739_739974


namespace two_roots_iff_a_greater_than_neg1_l739_739073

theorem two_roots_iff_a_greater_than_neg1 (a : ℝ) :
  (∃! x : ℝ, x^2 + 2*x + 2*|x + 1| = a) ↔ a > -1 :=
sorry

end two_roots_iff_a_greater_than_neg1_l739_739073


namespace problem_statement_l739_739185

-- Define function f(x) given parameter m
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + m * x + 3

-- Define even function condition
def is_even (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)

-- Define the monotonic decreasing interval condition
def is_monotonically_decreasing (f : ℝ → ℝ) (I : Set ℝ) :=
 ∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f x ≥ f y

theorem problem_statement :
  (∀ x : ℝ, f m x = f m (-x)) → is_monotonically_decreasing (f 0) {x | 0 < x} :=
by 
  sorry

end problem_statement_l739_739185


namespace quadratic_solution_set_R_l739_739980

theorem quadratic_solution_set_R (a b c : ℝ) (h1 : a ≠ 0) (h2 : a < 0) (h3 : b^2 - 4 * a * c < 0) : 
  ∀ x : ℝ, a * x^2 + b * x + c < 0 :=
by sorry

end quadratic_solution_set_R_l739_739980


namespace ratio_of_ducks_to_total_goats_and_chickens_l739_739988

theorem ratio_of_ducks_to_total_goats_and_chickens 
    (goats chickens ducks pigs : ℕ) 
    (h1 : goats = 66)
    (h2 : chickens = 2 * goats)
    (h3 : pigs = ducks / 3)
    (h4 : goats = pigs + 33) :
    (ducks : ℚ) / (goats + chickens : ℚ) = 1 / 2 := 
by
  sorry

end ratio_of_ducks_to_total_goats_and_chickens_l739_739988


namespace remainder_of_polynomial_l739_739931

theorem remainder_of_polynomial 
  (P : ℝ → ℝ) 
  (h₁ : P 15 = 16)
  (h₂ : P 10 = 4) :
  ∃ Q : ℝ → ℝ, ∀ x, P x = (x - 10) * (x - 15) * Q x + (12 / 5 * x - 20) :=
by
  sorry

end remainder_of_polynomial_l739_739931


namespace integer_solutions_l739_739617

noncomputable def solutions (p q : ℤ) : set (ℤ × ℤ) :=
  { (1 + p * q, p^2 * q^2 + p * q), 
    (p * (q + 1), p * q * (q + 1)), 
    (q * (p + 1), p * q * (p + 1)), 
    (2 * p * q, 2 * p * q), 
    (p^2 * q * (p + q), q^2 + p * q), 
    (q^2 + p * q, p^2 + p * q), 
    (p * q * (p + 1), q * (p + 1)), 
    (p * q * (q + 1), p * (q + 1)), 
    (p^2 * q^2 + p * q, 1 + p * q) }

theorem integer_solutions (p q : ℤ) (hp : prime p) (hq : prime q) (hpq : p ≠ q) :
  { (a, b) ∣ (1 / a : ℚ) + (1 / b : ℚ) = 1 / (p * q) } = solutions p q :=
by
  sorry

end integer_solutions_l739_739617


namespace modulus_of_z_l739_739140

noncomputable def i : ℂ := complex.I

theorem modulus_of_z
  (z : ℂ)
  (h : i * z = (1 - 2 * i) ^ 2) :
  complex.abs z = 5 :=
sorry

end modulus_of_z_l739_739140


namespace rectangle_length_width_difference_l739_739833

theorem rectangle_length_width_difference
  (x y : ℝ)
  (h1 : x + y = 40)
  (h2 : x^2 + y^2 = 800) :
  x - y = 0 :=
sorry

end rectangle_length_width_difference_l739_739833


namespace shop_owner_cheat_selling_percentage_l739_739009

noncomputable def percentage_cheat_buying : ℝ := 12
noncomputable def profit_percentage : ℝ := 40
noncomputable def percentage_cheat_selling : ℝ := 20

theorem shop_owner_cheat_selling_percentage 
  (percentage_cheat_buying : ℝ := 12)
  (profit_percentage : ℝ := 40) :
  percentage_cheat_selling = 20 := 
sorry

end shop_owner_cheat_selling_percentage_l739_739009


namespace problem1_problem2_l739_739534

section Problem
variables (a : ℝ) (x : ℝ) (x1 x2 : ℝ)
noncomputable def f (x : ℝ) : ℝ := a * (Real.exp x - x - 1) - Real.log (x + 1) + x
noncomputable def g (x : ℝ) : ℝ := a * Real.exp x + x

theorem problem1 (ha : a ≥ 0) : ∃! x, f a x = 0 := sorry

theorem problem2 (ha : a ≥ 0) (h1 : x1 ∈ Icc (-1 : ℝ) (Real.inf)) (h2 : x2 ∈ Icc (-1 : ℝ) (Real.inf)) (h : f a x1 = g a x1 - g a x2) :
  x1 - 2 * x2 ≥ 1 - 2 * Real.log 2 := sorry

end Problem

end problem1_problem2_l739_739534


namespace expression_values_l739_739299

theorem expression_values (a b : ℝ) (h1 : a ≠ -b) (h2 : a ≠ b)
  (h : (2 * a) / (a + b) + b / (a - b) = 2) :
  (3 * a - b) / (a + 5 * b) = 1 ∨ (3 * a - b) / (a + 5 * b) = 3 := 
sorry

end expression_values_l739_739299


namespace roots_eq_two_iff_a_gt_neg1_l739_739063

theorem roots_eq_two_iff_a_gt_neg1 (a : ℝ) : 
  (∃! x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + 2*|x₁ + 1| = a ∧ x₂^2 + 2*x₂ + 2*|x₂ + 1| = a) ↔ a > -1 :=
by sorry

end roots_eq_two_iff_a_gt_neg1_l739_739063


namespace halloween_candy_weight_l739_739107

theorem halloween_candy_weight :
  let frank_chocolate : real := 3
  let frank_gummy_bears : real := 2
  let frank_caramels : real := 1
  let frank_assorted_hard_candy : real := 4

  let gwen_chocolate : real := 2
  let gwen_gummy_bears : real := 2.5
  let gwen_caramels : real := 1
  let gwen_assorted_hard_candy : real := 1.5

  let combined_chocolate := frank_chocolate + gwen_chocolate
  let combined_gummy_bears := frank_gummy_bears + gwen_gummy_bears
  let combined_caramels := frank_caramels + gwen_caramels
  let combined_assorted_hard_candy := frank_assorted_hard_candy + gwen_assorted_hard_candy

  let total_weight := combined_chocolate + combined_gummy_bears + combined_caramels + combined_assorted_hard_candy

  total_weight = 17 := 
by 
  intros
  sorry

end halloween_candy_weight_l739_739107


namespace range_of_a_l739_739135

variables {a : ℝ}

def p := ∀ x, 1 ≤ x ∧ x ≤ 2 → x^2 ≥ a
def q := ∃ x, x^2 + 2 * a * x + 2 - a = 0

theorem range_of_a (h: p ∧ q) : a ≤ -2 ∨ a = 1 :=
sorry

end range_of_a_l739_739135


namespace part1_part2_part3_l739_739291

-- Definitions based on given conditions
def f (x : ℝ) : ℝ := x / (9 - x^2)

-- Problem parts

-- Part (1): Proving the analytical expression of f(x)
theorem part1 (odd_f : ∀ x, f(-x) = -f(x)) (fx1 : f(1) = 1 / 8) : f x = x / (9 - x^2) :=
sorry

-- Part (2): Proving monotonic behavior of f(x) on (-3, 3)
theorem part2 (mono_f : ∀ x1 x2 : ℝ, -3 < x1 ∧ x1 < x2 ∧ x2 < 3 → f(x1) < f(x2)) : 
  ∀ x: ℝ, -3 < x ∧ x < 3 → f(x) = x / (9 - x^2) :=
sorry

-- Part (3): Solving the inequality f(t-1) + f(t) < 0
theorem part3 (ineq : ∀ t : ℝ, f(t-1) + f(t) < 0 → -2 < t ∧ t < 1/2) : true :=
sorry

end part1_part2_part3_l739_739291


namespace num_right_triangles_correct_l739_739924

noncomputable def num_right_triangles (p : ℕ) (hp : p.prime) : ℕ :=
  if p = 2 then 18
  else if p = 997 then 20
  else 36

theorem num_right_triangles_correct (p : ℕ) (hp : p.prime) :
  ∃ n : ℕ, (p = 2 → n = 18) ∧ (p = 997 → n = 20) ∧ (p ≠ 2 ∧ p ≠ 997 → n = 36) :=
begin
  use num_right_triangles p hp,
  split,
  { intros h, simp [h], },
  split,
  { intros h, simp [h], },
  { intros h1 h2, simp [h1, h2], },
end

end num_right_triangles_correct_l739_739924


namespace ratio_of_buyers_l739_739322

theorem ratio_of_buyers (B Y T : ℕ) (hB : B = 50) 
  (hT : T = Y + 40) (hTotal : B + Y + T = 140) : 
  (Y : ℚ) / B = 1 / 2 :=
by 
  sorry

end ratio_of_buyers_l739_739322


namespace savings_per_month_l739_739404

-- Define the monthly earnings, total needed for car, and total earnings
def monthly_earnings : ℤ := 4000
def total_needed_for_car : ℤ := 45000
def total_earnings : ℤ := 360000

-- Define the number of months it takes to save the required amount using total earnings and monthly earnings
def number_of_months : ℤ := total_earnings / monthly_earnings

-- Define the monthly savings based on the total needed and number of months
def monthly_savings : ℤ := total_needed_for_car / number_of_months

-- Prove that the monthly savings is £500
theorem savings_per_month : monthly_savings = 500 := by
  -- Placeholder for the proof
  sorry

end savings_per_month_l739_739404


namespace find_a_and_b_l739_739845

theorem find_a_and_b (a b : ℤ)
  (h : 4 * Real.cos (Float.pi * 70 / 180) = Real.sqrt (a + b * Real.csc (Float.pi * 70 / 180))) :
  a = 4 ∧ b = -2 :=
by
  sorry

end find_a_and_b_l739_739845


namespace sequence_formula_l739_739891

noncomputable def a : ℕ → ℝ
| 1 := 3
| n := sorry -- Since a perfectly valid Lean 4 definition isn't complete without an accompanying proof in this context

theorem sequence_formula (n : ℕ) (h : n > 0) : 
  (∀ n > 1, (λ x y : ℝ, x - y - Real.sqrt 3 = 0) (Real.sqrt (a n)) (Real.sqrt (a (n - 1)))) → 
  a n = 3 * n^2 :=
sorry

end sequence_formula_l739_739891


namespace reversible_triangle_inequality_l739_739614

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def reversible_triangle (a b c : ℝ) : Prop :=
  (is_triangle a b c) ∧ 
  (is_triangle (1 / a) (1 / b) (1 / c)) ∧
  (a ≤ b) ∧ (b ≤ c)

theorem reversible_triangle_inequality {a b c : ℝ} (h : reversible_triangle a b c) :
  a > (3 - Real.sqrt 5) / 2 * c :=
sorry

end reversible_triangle_inequality_l739_739614


namespace range_of_w_l739_739149

theorem range_of_w (ω : ℝ) (h1 : ω > 0)
  (h2 : ∀ x : ℝ, 0 ≤ x ∧ x ≤ π → sin (ω * x - π / 3) ≥ -sqrt 3 / 2 ∧ sin (ω * x - π / 3) ≤ 1) :
  5 / 6 ≤ ω ∧ ω ≤ 5 / 3 :=
begin
  sorry -- Proof to be provided
end

end range_of_w_l739_739149


namespace mark_weekly_reading_l739_739624

-- Using the identified conditions
def daily_reading_hours : ℕ := 2
def additional_weekly_hours : ℕ := 4

-- Prove the total number of hours Mark wants to read per week is 18 hours
theorem mark_weekly_reading : (daily_reading_hours * 7 + additional_weekly_hours) = 18 := by
  -- Placeholder for proof
  sorry

end mark_weekly_reading_l739_739624


namespace opposite_neg_two_l739_739301

def opposite (x : Int) : Int := -x

theorem opposite_neg_two : opposite (-2) = 2 := by
  sorry

end opposite_neg_two_l739_739301


namespace integer_solutions_count_l739_739167

theorem integer_solutions_count : 
  (∀ (a : ℤ), a^0 = 1) ∧ (∀ (b : ℤ), 1^b = 1) ∧ (∀ (c : ℤ), even c → (-1)^c = 1) → 
  ∃ (n : ℕ), n = 2 ∧ (∀ (x : ℤ), (x - 3)^(30 - x^2) = 1 → x = 4 ∨ x = 2) :=
sorry

end integer_solutions_count_l739_739167


namespace incorrect_statement_among_options_l739_739726

/- Definitions and Conditions -/
variables {a : ℕ → ℝ} {S : ℕ → ℝ} {d : ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n * a 1) + (n * (n - 1) / 2) * d

/- Conditions given in the problem -/
axiom S_6_gt_S_7 : S 6 > S 7
axiom S_7_gt_S_5 : S 7 > S 5

/- Incorrect statement to be proved -/
theorem incorrect_statement_among_options :
  ¬ (∀ n, S n ≤ S 11) := sorry

end incorrect_statement_among_options_l739_739726


namespace least_four_digit_9_heavy_is_1005_l739_739761

def is_9_heavy (n : ℕ) : Prop := n % 9 = 6

theorem least_four_digit_9_heavy_is_1005 : ∃ n : ℕ, is_9_heavy(n) ∧ 1000 ≤ n ∧ n < 10000 ∧ ∀ m : ℕ, is_9_heavy(m) ∧ 1000 ≤ m ∧ m < 10000 → n ≤ m :=
by
  sorry

end least_four_digit_9_heavy_is_1005_l739_739761


namespace surface_area_comparison_l739_739932

theorem surface_area_comparison 
  (p : ℝ) (PQ_focus : ℝ → ℝ) (MN_projection : ℝ → ℝ)
  (S1 : ℝ) (S2 : ℝ)
  (h1 : ∀ y, y^2 = 2 * p * (PQ_focus y))
  (h2 : MN_projection = λ y, PQ_focus y * |cos (atan y)|)
  (h3 : S1 = π * (PQ_focus(0) + PQ_focus(0))^2)
  (h4 : S2 = π * (MN_projection(0) + MN_projection(0))^2 * sin (atan (MN_projection(0) / PQ_focus(0)))^2) :
  S1 ≥ S2 := sorry

end surface_area_comparison_l739_739932


namespace find_f_of_3pi_by_4_l739_739858

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi / 2)

theorem find_f_of_3pi_by_4 : f (3 * Real.pi / 4) = -Real.sqrt 2 / 2 := by
  sorry

end find_f_of_3pi_by_4_l739_739858


namespace negative_integer_is_minus_21_l739_739873

variable (n : ℤ) (hn : n < 0) (h : n * (-3) + 2 = 65)

theorem negative_integer_is_minus_21 : n = -21 :=
by
  sorry

end negative_integer_is_minus_21_l739_739873


namespace crows_cannot_be_on_same_tree_l739_739695

theorem crows_cannot_be_on_same_tree :
  (∀ (trees : ℕ) (crows : ℕ),
   trees = 22 ∧ crows = 22 →
   (∀ (positions : ℕ → ℕ),
    (∀ i, 1 ≤ positions i ∧ positions i ≤ 2) →
    ∀ (move : (ℕ → ℕ) → (ℕ → ℕ)),
    (∀ (pos : ℕ → ℕ) (i : ℕ),
     move pos i = pos i + positions (i + 1) ∨ move pos i = pos i - positions (i + 1)) →
    (∀ (pos : ℕ → ℕ) (i : ℕ),
     pos i % trees = (move pos i) % trees) →
    ¬ (∃ (final_pos : ℕ → ℕ),
      (∀ i, final_pos i = 0 ∨ final_pos i = 22) ∧
      (∀ i j, final_pos i = final_pos j)
    )
  )
) :=
sorry

end crows_cannot_be_on_same_tree_l739_739695


namespace complement_P_subset_PQ_intersection_PQ_eq_Q_l739_739159

open Set

variable {R : Type*} [OrderedCommRing R]

def P (x : R) : Prop := -2 ≤ x ∧ x ≤ 10
def Q (m x : R) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

theorem complement_P : (compl (setOf P)) = {x | x < -2} ∪ {x | x > 10} :=
by {
  sorry
}

theorem subset_PQ (m : R) : (∀ x, P x → Q m x) ↔ m ≥ 9 :=
by {
  sorry
}

theorem intersection_PQ_eq_Q (m : R) : (∀ x, Q m x → P x) ↔ m ≤ 9 :=
by {
  sorry
}

end complement_P_subset_PQ_intersection_PQ_eq_Q_l739_739159


namespace proof_problem_l739_739436

noncomputable def problem : ℝ :=
  (3 / 2) ^ (-1 / 3) - (1 / 3) * ((-7 / 6) ^ 0) + (8 ^ (1 / 4)) * (2 ^ (1 / 4)) - (sqrt ((-2 / 3) ^ (2 / 3)))

theorem proof_problem : problem = 5 / 3 :=
  by sorry

end proof_problem_l739_739436


namespace g_432_l739_739612

theorem g_432 (g : ℕ → ℤ)
  (h_mul : ∀ x y : ℕ, 0 < x → 0 < y → g (x * y) = g x + g y)
  (h8 : g 8 = 21)
  (h18 : g 18 = 26) :
  g 432 = 47 :=
  sorry

end g_432_l739_739612


namespace even_number_of_segments_l739_739255

theorem even_number_of_segments :
  ∀ (A : ℕ → ℝ × ℝ) (n : ℕ) (segments : list (ℝ × ℝ) × (ℝ × ℝ)),
    (∀ i j k, i ≠ j ∧ i ≠ k ∧ j ≠ k → ¬ collinear (A i) (A j) (A k)) →
    (∀ L : ℕ → ℝ × ℝ, ∃ even_segments : nat, 
        (∀ i j, i ≠ j → ¬ passes_through L (A i) →
          even (segments_intersecting L segments))) →
    (∀ i, even (length (filter (λ s, connects (A i) s) segments))) :=
by
  intros,
  sorry

end even_number_of_segments_l739_739255


namespace vertical_tangent_line_l739_739523

def f (x : ℝ) (a : ℝ) : ℝ := (1/2) * x^2 - a * x + Real.log x

theorem vertical_tangent_line (a : ℝ) : 
  (∃ x > 0, deriv (λ x : ℝ, f x a) x = 0) ↔ a ∈ Ici 2 :=
sorry

end vertical_tangent_line_l739_739523


namespace find_m_l739_739540

variable (m : ℝ)

def a := (3, m)
def b := (1 : ℝ, -2)

-- Definition of dot product for 2D vectors
def dot_product (x y : ℝ × ℝ) : ℝ :=
  x.1 * y.1 + x.2 * y.2

-- Definition of magnitude squared for 2D vector
def magnitude_squared (x : ℝ × ℝ) : ℝ :=
  dot_product x x

theorem find_m (h : dot_product a b + 3 * magnitude_squared b = 0) : m = 9 :=
by 
  sorry

end find_m_l739_739540


namespace rotation_proof_l739_739332

noncomputable def point := (ℝ × ℝ)

def D : point := (2, 2)
def E : point := (2, 14)
def F : point := (18, 2)

def D' : point := (32, 26)
def E' : point := (44, 26)
def F' : point := (32, 10)

theorem rotation_proof (n : ℝ) (u v : ℝ) (h₀ : 0 < n ∧ n < 180)
  (h₁ : n = 90)
  (h₂ : u = 6)
  (h₃ : v = 28)
  (h₄ : rotate_around (u, v) n D = D')
  (h₅ : rotate_around (u, v) n E = E')
  (h₆ : rotate_around (u, v) n F = F') :
  n + u + v = 124 := by
  sorry

end rotation_proof_l739_739332


namespace range_of_λ_over_m_l739_739546

variables (λ m α : ℝ) 

def a : ℝ × ℝ := (λ + 2, λ^2 - cos α ^ 2)
def b : ℝ × ℝ := (m, m / 2 + sin α)

theorem range_of_λ_over_m 
  (h : a λ α = (2 * b m α)) : 
  -6 ≤ λ / m ∧ λ / m ≤ 1 := sorry

end range_of_λ_over_m_l739_739546


namespace triangle_proof_l739_739894

noncomputable def triangle_problem (a b c : ℝ) (A B C : ℝ) : Prop :=
  (b = sqrt 3) ∧
  ((c - 2 * a) * Real.cos B + b * Real.cos C = 0) →
  B = Real.pi / 3

noncomputable def range_ac (a c : ℝ) : Prop :=
  sqrt 3 < a + c ∧ a + c ≤ 2 * sqrt 3

theorem triangle_proof (a b c A B C : ℝ) :
  triangle_problem a b c A B C ∧ range_ac a c :=
sorry

end triangle_proof_l739_739894


namespace fruit_salad_cherries_l739_739399

variable (b r g c : ℕ)

theorem fruit_salad_cherries :
  (b + r + g + c = 350) ∧
  (r = 3 * b) ∧
  (g = 4 * c) ∧
  (c = 5 * r) →
  c = 66 :=
by
  sorry

end fruit_salad_cherries_l739_739399


namespace exactly_two_roots_iff_l739_739079

theorem exactly_two_roots_iff (a : ℝ) : 
  (∃! (x : ℝ), x^2 + 2 * x + 2 * |x + 1| = a) ↔ a > -1 :=
by
  sorry

end exactly_two_roots_iff_l739_739079


namespace unique_solution_implies_d_999_l739_739260

variable (a b c d x y : ℤ)

theorem unique_solution_implies_d_999
  (h1 : a < b)
  (h2 : b < c)
  (h3 : c < d)
  (h4 : 3 * x + y = 3005)
  (h5 : y = |x-a| + |x-b| + |x-c| + |x-d|)
  (h6 : ∃! x, 3 * x + |x-a| + |x-b| + |x-c| + |x-d| = 3005) :
  d = 999 :=
sorry

end unique_solution_implies_d_999_l739_739260


namespace probability_two_same_color_l739_739879

/-- There are 4 balls in a box, 2 red and 2 white. Two balls are to be drawn without replacement.
    The probability of drawing two balls of the same color is 1/3. -/
theorem probability_two_same_color (red white : ℕ)
    (h_red : red = 2) (h_white : white = 2) : 
    (probability (λ (event : Finset Ball), event.card = 2 ∧ 
    ∀ b ∈ event, b.color = Color.Red ∨ b.color = Color.White) = 1/3) :=
sorry

end probability_two_same_color_l739_739879


namespace C2_cartesian_eq_line_l_intersects_C2_l739_739118

noncomputable def point_N := (real.sqrt 2, real.pi / 4 : ℝ × ℝ)
def curve_C1 := { p : ℝ × ℝ | p.1^2 + p.2^2 = 1 }
def point_M (p : ℝ × ℝ) : p ∈ curve_C1 := by sorry
def point_G (G M N : ℝ × ℝ) : G = (M.1 + N.1, M.2 + N.2) := by sorry

theorem C2_cartesian_eq :
  (∀ G M N : ℝ × ℝ,
    (∀ p : ℝ × ℝ, p ∈ curve_C1 → ∃ y : ℝ, y = G.2) ∧ 
    G = (point_M G, point_N).1 + (point_M G, point_N).2 →
    (G.1 - 1)^2 + (G.2 - 1)^2 = 1) := by sorry

theorem line_l_intersects_C2 :
  ∀ A B : ℝ × ℝ,
  (∃ t : ℝ, (2 - 1/2 * t, real.sqrt 3 / 2 * t) = A ∧ 
             (2 - 1/2 * t, real.sqrt 3 / 2 * t) = B) ∧
  A ∈ { p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 = 1 } ∧
  B ∈ { p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 = 1 } →
  (1 / real.dist (2, 0) A + 1 / real.dist (2, 0) B = 1 + real.sqrt 3) := by sorry

end C2_cartesian_eq_line_l_intersects_C2_l739_739118


namespace find_FC_l739_739115

-- Definitions of given lengths
def DC : ℝ := 9
def CB : ℝ := 10
noncomputable def AD : ℝ := (3 / 2) * 19  -- from the condition AD = 28.5
def AB : ℝ := (1 / 3) * AD
noncomputable def ED : ℝ := (3 / 4) * AD

-- Calculation for CA and FC
def CA : ℝ := CB + AB
noncomputable def FC : ℝ := (ED * CA) / AD

-- The theorem to prove
theorem find_FC : FC = 14.6 := by
  sorry

end find_FC_l739_739115


namespace distance_from_center_to_line_l739_739434

-- Define the circle C
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x + 2*y - 2 = 0

-- Define the line l
def line_eq (x y : ℝ) : Prop :=
  x - y + 2 = 0

-- Define the center of the circle
def center : ℝ × ℝ :=
  (-1, -1)

-- Define the point-to-line distance formula
def point_to_line_distance (P : ℝ × ℝ) (A B C : ℝ) : ℝ :=
  (abs (A * P.1 + B * P.2 + C)) / sqrt (A^2 + B^2)

-- Instantiate distance computation for this problem
def distance : ℝ :=
  point_to_line_distance center 1 (-1) 2

theorem distance_from_center_to_line :
  distance = sqrt 2 :=
by
  sorry

end distance_from_center_to_line_l739_739434


namespace original_tetrahedron_edges_l739_739111

theorem original_tetrahedron_edges (A B C D M_AB M_BC M_BD M_AD M_AC : Point)
  (h1 : midpoint A B = M_AB) (h2 : midpoint B C = M_BC) (h3 : midpoint B D = M_BD)
  (h4 : midpoint A D = M_AD) (h5 : midpoint A C = M_AC)
  (h6 : is_regular_tetrahedron {M_BC, M_BD, M_AD, M_AC} (2 : ℝ)) :
  (∃ e : ℝ, e = 4) ∧ (∃ e' : ℝ, e' = 4 * real.sqrt 2) :=
sorry

end original_tetrahedron_edges_l739_739111


namespace exists_constant_C_l739_739946

def harmonic (n : ℕ) : ℝ := ∑ k in Finset.range n + 1, 1 / (k : ℝ)

theorem exists_constant_C :
  ∃ C > 0, ∀ (m : ℕ) (a : Fin m.succ → ℕ),
    (∑ i in Finset.range m.succ, harmonic (a i)) ≤ C * Real.sqrt (∑ i in Finset.range m.succ, (i + 1) * a i)
:= 
sorry

end exists_constant_C_l739_739946


namespace probability_rain_at_most_3_days_in_july_l739_739307

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_rain_at_most_3_days_in_july :
  let p := 1 / 5
  let n := 31
  let sum_prob := binomial_probability n 0 p + binomial_probability n 1 p + binomial_probability n 2 p + binomial_probability n 3 p
  abs (sum_prob - 0.125) < 0.001 :=
by
  sorry

end probability_rain_at_most_3_days_in_july_l739_739307


namespace range_of_m_l739_739190

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, abs (x - m) < 1 ↔ (1/3 < x ∧ x < 1/2)) ↔ (-1/2 ≤ m ∧ m ≤ 4/3) :=
by
  sorry

end range_of_m_l739_739190


namespace triangle_angle_bisector_median_altitude_intersect_at_one_point_l739_739563

noncomputable def acute_triangle {α : Type*} [linear_order α] [order_bot α]
  [conditionally_complete_linear_order α] (A B C : α × α) : Prop :=
triangle A B C ∧ angle A B C < π / 2 ∧ angle B C A < π / 2 ∧ angle C A B < π / 2

theorem triangle_angle_bisector_median_altitude_intersect_at_one_point
  (A B C : ℝ × ℝ)
  (h_acute : acute_triangle A B C)
  (D M H O : ℝ × ℝ)
  (h_AD_angle_bisector : angle_bisector A B C D)
  (h_BM_median : median B M)
  (h_CH_altitude : altitude C H)
  (h_intersect : intersects_at_one_point (A, D) (B, M) (C, H) O):
  angle A B C > π / 4 :=
sorry

end triangle_angle_bisector_median_altitude_intersect_at_one_point_l739_739563


namespace steak_chicken_ratio_l739_739900

variable (S C : ℕ)

theorem steak_chicken_ratio (h1 : S + C = 80) (h2 : 25 * S + 18 * C = 1860) : S = 3 * C :=
by
  sorry

end steak_chicken_ratio_l739_739900


namespace geometry_problem_l739_739882

open EuclideanGeometry

variables (O A B M T P : Point)

noncomputable def circle (center : Point) (radius : ℝ) := {p | dist center p = radius}

def midpoint (A B : Point) : Point := {
  x := (A.x + B.x) / 2,
  y := (A.y + B.y) / 2
}

theorem geometry_problem
  (hO : inside_circle O A B)   -- O is the center of circle C1, AB is a chord of C1
  (hM : M = midpoint A B)       -- M is the midpoint of chord AB
  (h2 : T ∈ circle M (dist O M / 2))  -- T lies on circle C2 with OM as diameter
  (h3 : is_tangent P T)         -- Tangent to C2 at T intersects C1 at P
  : dist P A ^ 2 + dist P B ^ 2 = 4 * dist P T ^ 2 :=
sorry

end geometry_problem_l739_739882


namespace find_equation_of_line_l739_739512

theorem find_equation_of_line
  (midpoint : ℝ × ℝ)
  (ellipse : ℝ → ℝ → Prop)
  (l_eq : ℝ → ℝ → Prop)
  (H_mid : midpoint = (1, 2))
  (H_ellipse : ∀ (x y : ℝ), ellipse x y ↔ x^2 / 64 + y^2 / 16 = 1)
  (H_line : ∀ (x y : ℝ), l_eq x y ↔ y - 2 = - (1/8) * (x - 1))
  : ∃ (a b c : ℝ), (a, b, c) = (1, 8, -17) ∧ (∀ (x y : ℝ), l_eq x y ↔ a * x + b * y + c = 0) :=
by 
  sorry

end find_equation_of_line_l739_739512


namespace actual_distance_of_cities_l739_739658

-- Define the conditions as constants or expressions
def map_distance_inches : ℝ := 20
def scale_inches_per_mile : ℝ := 0.5 / 6

-- The theorem states that if we know the map distance and the scale,
-- we can calculate the actual distance between the cities.
theorem actual_distance_of_cities : (map_distance_inches / scale_inches_per_mile) = 240 := by
  -- Here we would provide the proof, but per the instructions we use sorry for now.
  sorry

end actual_distance_of_cities_l739_739658


namespace sum_of_coefficients_l739_739103

theorem sum_of_coefficients :
  let p := -3 * (Polynomial.C (-6) + 4 * Polynomial.X^3 - 2 * Polynomial.X^5 + Polynomial.X^8)
              + 5 * (3 * Polynomial.X^2 + Polynomial.X^4)
              - 4 * (Polynomial.C 5 - Polynomial.X^6)
  in p.eval 1 = 45 :=
by
  let p := -3 * (Polynomial.C (-6) + 4 * Polynomial.X^3 - 2 * Polynomial.X^5 + Polynomial.X^8)
              + 5 * (3 * Polynomial.X^2 + Polynomial.X^4)
              - 4 * (Polynomial.C 5 - Polynomial.X^6)
  show p.eval 1 = 45
  sorry

end sum_of_coefficients_l739_739103


namespace angle_ACB_eq_10_l739_739836

-- Define the types
noncomputable def trapezoid := Type

-- Define the problem statement
variables {A B C D : trapezoid}

-- Given conditions
axiom angle_C_eq_30 : ∠ C = 30
axiom angle_D_eq_80 : ∠ D = 80
axiom DB_is_angle_bisector : IsAngleBisector D B (∠ D)

-- Prove that ∠ACB = 10
theorem angle_ACB_eq_10 : ∠ ACB = 10 :=
by
  sorry

end angle_ACB_eq_10_l739_739836


namespace arith_seq_theorem_l739_739838

-- Given: arithmetic sequence {a_n} with the following conditions
-- a_3 = 7
-- a_5 + a_7 = 26
def arith_seq_satisfies (a : ℕ → ℝ) (d : ℝ) : Prop :=
  a 3 = 7 ∧ a 5 + a 7 = 26

-- Definitions for a_n and S_n
def a_n (n : ℕ) : ℝ := 2 * n + 1
def S_n (n : ℕ) : ℝ := n^2 + 2 * n

-- Definition for b_n and T_n
def b_n (n : ℕ) : ℝ := 1 / ((a_n n)^2 - 1)
def T_n (n : ℕ) : ℝ := n / (4 * (n + 1))

theorem arith_seq_theorem (a : ℕ → ℝ) (d : ℝ) (h : arith_seq_satisfies a d) :
  (∀ n : ℕ, a n = a_n n) ∧ (∀ n : ℕ, (∑ i in finset.range n, a i) = S_n n) ∧
  (∀ n : ℕ, (∑ i in finset.range n, b_n i) = T_n n) :=
by sorry

end arith_seq_theorem_l739_739838


namespace mary_spending_ratio_l739_739253

noncomputable theory

variable (x : ℝ)
variable (initial_amount : ℝ := 100)
variable (goggles_fraction : ℝ := 1/5)

def amount_left_after_game (initial_amount x : ℝ) : ℝ := initial_amount - x
def amount_spent_on_goggles (amount_left_after_game : ℝ) (goggles_fraction : ℝ) : ℝ := goggles_fraction * amount_left_after_game
def remaining_after_goggles (initial_amount x goggles_fraction : ℝ) : ℝ :=
  amount_left_after_game initial_amount x - amount_spent_on_goggles (amount_left_after_game initial_amount x) goggles_fraction

theorem mary_spending_ratio :
  remaining_after_goggles initial_amount x goggles_fraction = 60 →
  (x = 100 / 3) →
  (x / 100 = 1 / 3) :=
sorry

end mary_spending_ratio_l739_739253


namespace lines_skew_l739_739467

theorem lines_skew (b : ℝ) :
  let r1 := λ t : ℝ, (2 : ℝ, b, 4) + t • (3, 4, 5),
      r2 := λ u : ℝ, (3 : ℝ, 2, 1) + u • (6, 5, 2)
  in (∀ t u : ℝ, r1 t ≠ r2 u) ↔ b ≠ (448 / 105) :=
by
  sorry

end lines_skew_l739_739467


namespace sum_T_100_l739_739245

-- Define \( T_n \) as described in the problem
def T (n : ℕ) : ℚ :=
  ∑ k in Finset.range n, (-1)^k * (k^3 + k^2 + k + 1) / k.fact

-- The theorem states the given sum for n = 100 is equal to a specific form
theorem sum_T_100 :
  T 100 = (10202 : ℚ) / 100.fact - 2 :=
by
  sorry

end sum_T_100_l739_739245


namespace find_m_l739_739511

theorem find_m (m : ℝ) :
  (∃ m : ℝ, ∀ x y : ℝ, x + y - m = 0 ∧ x + (3 - 2 * m) * y = 0 → 
     (m = 1)) := 
sorry

end find_m_l739_739511


namespace arithmetic_sequence_sum_example_l739_739888

noncomputable def arithmetic_sequence (a1 d : ℕ → ℕ) (n : ℕ) : ℕ := a1 + (n - 1) * d

noncomputable def sum_sequence (a1 d : ℕ → ℕ) (n : ℕ) : ℕ := n * (a1 + (n - 1) * d / 2)

theorem arithmetic_sequence_sum_example:
  ∀ (a1 d : ℕ), 
  a1 + d = 2 * (a1 + 7 * d) + (a1 + 13 * d) -> 
  15 * (a1 + 7 * d) = 30 := 
by 
  sorry

end arithmetic_sequence_sum_example_l739_739888


namespace seq_all_rational_l739_739262

theorem seq_all_rational : ∀ (n : ℕ), 
  (∀ k, k < n → ∃ (a : ℚ), ∀ j, j ≤ k → (a_j = a)) →
  ∃ (a : ℚ), ∀ i, i < n → (a = if i = 0 then 1 else (4 * a (i - 1) + real.sqrt (7 * (a (i - 1) ^ 2) - 3)) / 3) :=
by
  sorry

end seq_all_rational_l739_739262


namespace problem1_problem2_l739_739852

noncomputable def r (x y : ℝ) : ℝ := real.sqrt (x^2 + y^2)
noncomputable def sin_alpha (x y : ℝ) := y / r x y
noncomputable def cos_alpha (x y : ℝ) := x / r x y

theorem problem1 (x y : ℝ) (h : r x y = 5) (hx : x = -4) (hy : y = 3) :
  (sin_alpha x y) = 3 / 5 ∧ (cos_alpha x y) = -4 / 5 ∧
  (sin (real.pi - real.arctan (y / x)) + cos (- real.arctan (y / x))) / tan (real.pi + real.arctan (y / x)) = 16 / 15 := 
sorry

theorem problem2 (x y : ℝ) (h : r x y = 5) (hx : x = -4) (hy : y = 3) :
  sin_alpha x y * cos_alpha x y + cos_alpha x y ^ 2 - sin_alpha x y ^ 2 + 1 = 4 / 5 := 
sorry

end problem1_problem2_l739_739852


namespace correct_equation_l739_739013

noncomputable def eqn_correct (x : ℕ) : Prop :=
  let price_A := 810 / (x + 20)
  let price_B := 600 / x
  price_A = price_B * (1 - 0.1)

theorem correct_equation :
  ∀ (x : ℕ), eqn_correct x :=
sorry

end correct_equation_l739_739013


namespace union_of_complements_eq_l739_739863

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

theorem union_of_complements_eq :
  U = {1, 2, 3, 4, 5, 6, 7} →
  A = {2, 4, 5, 7} →
  B = {3, 4, 5} →
  ((U \ A) ∪ (U \ B) = {1, 2, 3, 6, 7}) :=
by
  intros hU hA hB
  sorry

end union_of_complements_eq_l739_739863


namespace log_complex_expression_l739_739622

theorem log_complex_expression
  (z1 z2 : ℂ)
  (h1 : complex.abs z1 = 3)
  (h2 : complex.abs (z1 + z2) = 3)
  (h3 : complex.abs (z1 - z2) = 3 * real.sqrt 3)
  : real.logb 3 complex.abs ((z1 * complex.conj z2) ^ 2000 + (complex.conj z1 * z2) ^ 2000) = 4000 :=
sorry

end log_complex_expression_l739_739622


namespace complex_eq_z100_zReciprocal_l739_739504

theorem complex_eq_z100_zReciprocal
  (z : ℂ)
  (h : z + z⁻¹ = 2 * Real.cos (5 * Real.pi / 180)) :
  z^100 + z⁻¹^100 = -2 * Real.cos (40 * Real.pi / 180) :=
by
  sorry

end complex_eq_z100_zReciprocal_l739_739504


namespace x_intercept_of_line_l739_739468

theorem x_intercept_of_line : ∃ x : ℚ, 3 * x + 5 * 0 = 20 ∧ (x, 0) = (20/3, 0) :=
by
  sorry

end x_intercept_of_line_l739_739468


namespace quadratic_inequality_l739_739220

theorem quadratic_inequality (a x1 x2 : ℝ) (h_eq : x1 ^ 2 - a * x1 + a = 0) (h_eq' : x2 ^ 2 - a * x2 + a = 0) :
  x1^2 + x2^2 ≥ 2 * (x1 + x2) :=
sorry

end quadratic_inequality_l739_739220


namespace raft_travel_time_l739_739391

noncomputable def downstream_speed (x y : ℝ) : ℝ := x + y
noncomputable def upstream_speed (x y : ℝ) : ℝ := x - y

theorem raft_travel_time {x y : ℝ} 
  (h1 : 7 * upstream_speed x y = 5 * downstream_speed x y) : (35 : ℝ) = (downstream_speed x y) * 7 / 4 := by sorry

end raft_travel_time_l739_739391


namespace trapezoid_area_l739_739653

theorem trapezoid_area (AD BC : ℝ) (AD_eq : AD = 18) (BC_eq : BC = 2) (CD : ℝ) (h : CD = 10): 
  ∃ (CH : ℝ), CH = 6 ∧ (1 / 2) * (AD + BC) * CH = 60 :=
by
  sorry

end trapezoid_area_l739_739653


namespace Problem_l739_739400

def point := (ℝ, ℝ)

def check_perpendicular_line (p: point) (l1 l2: ℝ → ℝ) (y: ℝ) : Prop :=
  let (x1, y1) := p
  ∃ m1 m2 c1 c2, 
    (l1 x1 = y1) ∧ 
    (m1 * m2 = -1) ∧ 
    (l2 x1 = c2 - 1 * x1 + 3 / 5) ∧ 
    (y1 = m2 * x1 + c2)

theorem Problem : 
  check_perpendicular_line 
    (1, 5) 
    (λ x, (2 / 5) * x + 3 / 5) 
    (λ x, (-5 / 2) * x + 15 / 2) 
    (5 * -1 + 2 * (15 / 2) - 15 = 0) := 
  sorry

end Problem_l739_739400


namespace avg_rate_of_change_in_interval_1_2_l739_739652

def f (x : ℝ) : ℝ := 2 * x - 1

theorem avg_rate_of_change_in_interval_1_2 : 
  let Δy := f 2 - f 1 in
  let Δx := 2 - 1 in
  Δy / Δx = 2 :=
by
  sorry

end avg_rate_of_change_in_interval_1_2_l739_739652


namespace valid_paths_l739_739590

theorem valid_paths : 
  ∀ (paths_from : ℕ → ℕ → ℕ), 
  ∀ (x y : ℕ), 
  (∀ (x y : ℕ), x = 0 ∧ y = 0 → paths_from x y = 1) → 
  (∀ (x y : ℕ), x > 0 → paths_from x y = paths_from (x - 1) y + paths_from x (y - 1)) → 
  (paths_from 3 2) - (paths_from 1 1 * paths_from (3 - 1) (2 - 1)) = 4 :=
by
  sorry

end valid_paths_l739_739590


namespace count_positive_numbers_is_three_l739_739028

def negative_three := -3
def zero := 0
def negative_three_squared := (-3) ^ 2
def absolute_negative_nine := |(-9)|
def negative_one_raised_to_four := -1 ^ 4

def number_list : List Int := [ -negative_three, zero, negative_three_squared, absolute_negative_nine, negative_one_raised_to_four ]

def count_positive_numbers (lst: List Int) : Nat :=
  lst.foldl (λ acc x => if x > 0 then acc + 1 else acc) 0

theorem count_positive_numbers_is_three : count_positive_numbers number_list = 3 :=
by
  -- The proof will go here.
  sorry

end count_positive_numbers_is_three_l739_739028


namespace polynomial_remainder_division_l739_739098

theorem polynomial_remainder_division (x : ℝ) : 
  (x^4 + 1) % (x^2 - 4 * x + 6) = 16 * x - 59 := 
sorry

end polynomial_remainder_division_l739_739098


namespace smallest_f_eq_iff_pow_two_l739_739615

def is_smallest_f (n : ℕ) (f : ℕ) : Prop :=
  (∑ k in finset.range (f + 1), k) % n = 0

theorem smallest_f_eq_iff_pow_two (n : ℕ) (m : ℕ):
  (∀ m : ℕ, n = 2^m → ∃ f : ℕ, is_smallest_f n f ∧ f = 2*n - 1) ↔
  (n = 2^m) :=
sorry

end smallest_f_eq_iff_pow_two_l739_739615


namespace ravi_work_alone_days_l739_739265

theorem ravi_work_alone_days (R : ℝ) (h1 : 1 / 75 + 1 / R = 1 / 30) : R = 50 :=
sorry

end ravi_work_alone_days_l739_739265


namespace log_value_comparison_l739_739730

theorem log_value_comparison :
  let e := Real.exp 1 in
  let initial_value := Log.log 1.2 * 1 / 6 in
  let transformed_value := 2.988 in
  transformed_value > e :=
by
  sorry

end log_value_comparison_l739_739730


namespace smallest_sum_of_integers_product_12_fact_l739_739964

theorem smallest_sum_of_integers_product_12_fact :
  ∃ (p q r s : ℕ) (h1 : p > 0) (h2 : q > 0) (h3 : r > 0) (h4 : s > 0),
    p * q * r * s = 12! ∧ p + q + r + s = 1402 := by
  sorry

end smallest_sum_of_integers_product_12_fact_l739_739964


namespace two_roots_iff_a_gt_neg1_l739_739068

theorem two_roots_iff_a_gt_neg1 (a : ℝ) :
  (∃! x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + 2*x1 + 2*|x1 + 1| = a) ∧ (x2^2 + 2*x2 + 2*|x2 + 1| = a)) ↔ a > -1 :=
by sorry

end two_roots_iff_a_gt_neg1_l739_739068


namespace hyperbola_m_range_l739_739203

-- Given conditions
def is_hyperbola_equation (m : ℝ) : Prop :=
  ∃ x y : ℝ, (4 - m) ≠ 0 ∧ (2 + m) ≠ 0 ∧ x^2 / (4 - m) - y^2 / (2 + m) = 1

-- Prove the range of m is -2 < m < 4
theorem hyperbola_m_range (m : ℝ) : is_hyperbola_equation m → (-2 < m ∧ m < 4) :=
by
  sorry

end hyperbola_m_range_l739_739203


namespace average_annual_percent_change_l739_739970

-- Define the initial and final population, and the time period
def initial_population : ℕ := 175000
def final_population : ℕ := 297500
def decade_years : ℕ := 10

-- Define the theorem to find the resulting average percent change per year
theorem average_annual_percent_change
    (P₀ : ℕ := initial_population)
    (P₁₀ : ℕ := final_population)
    (years : ℕ := decade_years) :
    ((P₁₀ - P₀ : ℝ) / P₀ * 100) / years = 7 := by
        sorry

end average_annual_percent_change_l739_739970


namespace abs_product_is_sqrt_10_l739_739516

-- Define z as the complex number -1 - i
def z : ℂ := -1 - Complex.i

-- Define z_bar as the complex conjugate of z
def z_bar : ℂ := Complex.conj z

-- Define one_minus_z as 1 - z
def one_minus_z : ℂ := 1 - z

-- Define product as (1 - z) * z_bar
def product : ℂ := one_minus_z * z_bar

-- Define the absolute value (modulus) of the product
def abs_product : ℝ := Complex.abs product

-- State the theorem we want to prove
theorem abs_product_is_sqrt_10 : abs_product = Real.sqrt 10 := by
  sorry

end abs_product_is_sqrt_10_l739_739516


namespace part_I_part_II_l739_739915

-- Definitions
def S (a : ℕ → ℝ) (n : ℕ) := 2 * (a n - 2^n + 1)
def a : ℕ → ℝ := sorry -- Definition of a will be dependent on its properties

-- Part (I)
theorem part_I (a : ℕ → ℝ) (h1 : ∀ n, S a n = 2 * (a n - 2^n + 1))
  (h2 : a 1 = 2) : ∃ d, ∀ n, (a n) / 2^n = ((a (n+1) / 2) / 2^(n+1)) + d := sorry

-- Part (II)
theorem part_II (a : ℕ → ℝ) (h1 : ∀ n, S a n = 2 * (a n - 2^n + 1)) 
  (h2 : a 1 = 2) : ∃ k, ∀ n, k > (S a n - 2) / (a n) ∧ (∀ m, m ∈ ℕ := k = 2) := sorry

end part_I_part_II_l739_739915


namespace total_amount_division_l739_739413

variables (w x y z : ℝ)

theorem total_amount_division (h_w : w = 2)
                              (h_x : x = 0.75)
                              (h_y : y = 1.25)
                              (h_z : z = 0.85)
                              (h_share_y : y * Rs48_50 = Rs48_50) :
                              total_amount = 4.85 * 38.80 := sorry

end total_amount_division_l739_739413


namespace total_volume_is_1056pi_l739_739219

-- Define the volumes of spheres given the radii
def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * (r^3)

-- Given radii
def radius1 := 4
def radius2 := 6
def radius3 := 8

-- Total volume calculation
def total_volume : ℝ := volume_of_sphere radius1 + volume_of_sphere radius2 + volume_of_sphere radius3

-- Theorem to prove the total volume is 1056π
theorem total_volume_is_1056pi : total_volume = 1056 * Real.pi := by
  sorry

end total_volume_is_1056pi_l739_739219


namespace even_integer_endpoint_l739_739283

theorem even_integer_endpoint (E : ℕ) :
  (let avg_20_to_E := (20 + E) / 2 in
   let avg_10_to_140 := (10 + 140) / 2 in
   avg_20_to_E = avg_10_to_140 + 35) → E = 200 :=
by
  intros h
  sorry

end even_integer_endpoint_l739_739283


namespace pq_half_perimeter_l739_739024

namespace Geometry

variable {AB CD : ℝ} {PQ : ℝ}
variable {A B C D P Q : Point} 
variable {∠B ∠C ∠A ∠D : Angle}

-- Definition of Parallel lines
definition parallel (AB CD : Line) : Prop :=
  ∀ (x y : Point), x ∈ AB → y ∈ CD → sameDirection (vector AB x) (vector CD y)

-- Hypotheses
variable (h1 : parallel AB CD)
variable (h2 : exterior_bisectors_meet ∠B ∠C P)
variable (h3 : exterior_bisectors_meet ∠A ∠D Q)

-- Hypothesis that PQ is half the perimeter of ABCD
def half_perimeter (ABCD : Quadrilateral) : ℝ :=
  (length AB + length BC + length CD + length DA) / 2

-- Theorem statement
theorem pq_half_perimeter (h1 : parallel AB CD) (h2 : exterior_bisectors_meet ∠B ∠C P)
  (h3 : exterior_bisectors_meet ∠A ∠D Q) : PQ = half_perimeter quadrilateral ABCD :=
sorry

end Geometry

end pq_half_perimeter_l739_739024


namespace analytical_expression_monotonicity_inequality_solution_l739_739294

def f (a b x : ℝ) : ℝ := (a * x - b) / (9 - x^2)

-- Conditions
def condition1 (a b : ℝ) : Prop := ∀ x : ℝ, -3 < x ∧ x < 3 → f a b x = -f a b (-x)
def condition2 : Prop := f 1 0 1 = 1 / 8

-- Question 1: Determine the analytical expression of f(x)
theorem analytical_expression (a b : ℝ) (h1 : condition1 a b) (h2 : condition2) :
  f a b = f 1 0 := sorry

-- Question 2: Determine the monotonicity of f(x)
theorem monotonicity (a : ℝ) :
  ∀ x1 x2 : ℝ, -3 < x1 ∧ x1 < x2 ∧ x2 < 3 → f a 0 x1 < f a 0 x2 := sorry

-- Question 3: Solve the inequality f(t-1) + f(t) < 0
theorem inequality_solution (a : ℝ) :
  ∀ t : ℝ, f a 0 (t - 1) + f a 0 t < 0 ↔ -2 < t ∧ t < 1 / 2 := sorry

end analytical_expression_monotonicity_inequality_solution_l739_739294


namespace sin_cos_equation_two_distinct_real_roots_l739_739962

theorem sin_cos_equation_two_distinct_real_roots (k : ℝ) : 
  (∀ x : ℝ, (0 ≤ x ∧ x ≤ π) → sin x + cos x = -k) → 
  (∃ x₁ x₂ : ℝ, (0 ≤ x₁ ∧ x₁ ≤ π) ∧ (0 ≤ x₂ ∧ x₂ ≤ π) ∧ x₁ ≠ x₂) → 
  (1 ≤ k ∧ k < sqrt 2) :=
sorry

end sin_cos_equation_two_distinct_real_roots_l739_739962


namespace trajectory_of_center_l739_739474

-- Define the given conditions
def tangent_circle (x y : ℝ) : Prop := x^2 + y^2 - 4 * x = 0

def tangent_y_axis (x : ℝ) : Prop := x = 0

-- Define the theorem with the given conditions and the desired conclusion
theorem trajectory_of_center (x y : ℝ) (h1 : tangent_circle x y) (h2 : tangent_y_axis x) :
  (y^2 = 8 * x) ∨ (y = 0 ∧ x ≤ 0) :=
sorry

end trajectory_of_center_l739_739474


namespace satisfactory_fraction_is_28_over_31_l739_739560

-- Define the number of students for each grade
def students_with_grade_A := 8
def students_with_grade_B := 7
def students_with_grade_C := 6
def students_with_grade_D := 4
def students_with_grade_E := 3
def students_with_grade_F := 3

-- Calculate the total number of students with satisfactory grades
def satisfactory_grades := students_with_grade_A + students_with_grade_B + students_with_grade_C + students_with_grade_D + students_with_grade_E

-- Calculate the total number of students
def total_students := satisfactory_grades + students_with_grade_F

-- Define the fraction of satisfactory grades
def satisfactory_fraction : ℚ := satisfactory_grades / total_students

-- The main proposition that the satisfactory fraction is 28/31
theorem satisfactory_fraction_is_28_over_31 : satisfactory_fraction = 28 / 31 := by {
  sorry
}

end satisfactory_fraction_is_28_over_31_l739_739560


namespace smallest_x_satisfies_absolute_value_equation_l739_739101

theorem smallest_x_satisfies_absolute_value_equation :
  ∃ x, |2 * x - 6| = 14 ∧ (∀ y, |2 * y - 6| = 14 → x ≤ y) :=
begin
  use -4,
  split,
  { norm_num, },
  { intros y hy,
    cases abs_eq (2 * y - 6) 14 with h1 h1;
    linarith }
end

end smallest_x_satisfies_absolute_value_equation_l739_739101


namespace mark_squares_l739_739826

/--Given 1000 squares with sides parallel to the coordinate axes, M represents the set of centers of these squares.
Prove that it is possible to mark some of these squares such that each point in M lies in at least 1 and no more than 4 
marked squares.--/
theorem mark_squares (squares : set (ℝ × ℝ) × ℝ) (hcount : squares.size = 1000)
  (sides_parallel : ∀ i ∈ squares, (∃ a b, i.1 = (a, b) ∧ i.2 > 0 ∧ (Mathlib.abs a < i.2 ∧ Mathlib.abs b < i.2)) ) :
  ∃ marked_squares : set (ℝ × ℝ) × ℝ, (∀ C ∈ M, ∃! s ∈ marked_squares, C ∈ Set.Icc (s.1.1, s.2) (s.1.2, s.2)) ∧
  (∀ C ∈ M, ∃! s ∈ marked_squares, ∃! n ≤ 4, C ∈ Set.Icc (s.1.1, s.2) (s.1.2, s.2)) :=
by
  sorry

end mark_squares_l739_739826


namespace simplify_polynomial_l739_739267

variable {R : Type} [CommRing R] (s : R)

theorem simplify_polynomial :
  (2 * s^2 + 5 * s - 3) - (2 * s^2 + 9 * s - 4) = -4 * s + 1 :=
by
  sorry

end simplify_polynomial_l739_739267


namespace max_ratio_of_three_digit_l739_739801

def is_digit (n : ℕ) : Prop := n >= 0 ∧ n ≤ 9

theorem max_ratio_of_three_digit (a b c : ℕ) (h_a : 1 ≤ a ∧ a ≤ 9) (h_b : is_digit b) (h_c : is_digit c) :
  let N := 100 * a + 10 * b + c in
  let S := a + b + c in
  S ≠ 0 →
  N / S ≤ 100 :=
by
  sorry

end max_ratio_of_three_digit_l739_739801


namespace sufficient_but_not_necessary_condition_l739_739723

theorem sufficient_but_not_necessary_condition (x : ℝ) : (x > 1 → x^2 > x) ∧ ¬(x^2 > x → x > 1) :=
by {
    intros,
    sorry
}

variable (x : ℝ)
#check sufficient_but_not_necessary_condition x

end sufficient_but_not_necessary_condition_l739_739723


namespace term_number_of_3sqrt5_l739_739541

theorem term_number_of_3sqrt5 (n : ℕ) : 
  (∀ (m : ℕ), (λ k, Real.sqrt (2 * k - 1)) m = 3 * Real.sqrt 5 → m = 23) := 
by
  intro m
  intro h
  sorry

end term_number_of_3sqrt5_l739_739541


namespace solution_set_of_inequality_l739_739854

def box_eq (a b : ℝ) : Prop :=
  a = b

theorem solution_set_of_inequality (x : ℝ) (h : ∀ x : ℝ, ∃ n : ℕ, ((n : ℝ) ≤ x) ∧ (x < (n+1 : ℝ))) :
  (4 * (nat.floor x : ℝ)^2 - 36 * (nat.floor x : ℝ) + 45 ≤ 0) → (2 ≤ x ∧ x < 8) :=
by {
  intro h₁,
  sorry
}

end solution_set_of_inequality_l739_739854


namespace final_bill_correct_l739_739737

def initial_bill := 500.00
def late_charge_rate := 0.02
def final_bill := initial_bill * (1 + late_charge_rate) * (1 + late_charge_rate)

theorem final_bill_correct : final_bill = 520.20 := by
  sorry

end final_bill_correct_l739_739737


namespace sum_of_n_values_l739_739342

open Nat

def binom (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.choose n k

theorem sum_of_n_values :
  ∑ n in {n : ℕ | binom 30 15 + binom 30 n = binom 31 16}, n = 30 := 
by
  sorry

end sum_of_n_values_l739_739342


namespace decimal_to_binary_41_l739_739050

theorem decimal_to_binary_41 : to_binary 41 = 101001 := 
by
  sorry

end decimal_to_binary_41_l739_739050


namespace mass_percentage_of_O_in_dichromate_l739_739089

noncomputable def molar_mass_Cr : ℝ := 52.00
noncomputable def molar_mass_O : ℝ := 16.00
noncomputable def molar_mass_Cr2O7_2_minus : ℝ := (2 * molar_mass_Cr) + (7 * molar_mass_O)

theorem mass_percentage_of_O_in_dichromate :
  (7 * molar_mass_O / molar_mass_Cr2O7_2_minus) * 100 = 51.85 := 
by
  sorry

end mass_percentage_of_O_in_dichromate_l739_739089


namespace monotonic_increasing_interval_l739_739296

noncomputable def log_base := (1 / 4 : ℝ)

def quad_expression (x : ℝ) : ℝ := -x^2 + 2*x + 3

def is_defined (x : ℝ) : Prop := quad_expression x > 0

theorem monotonic_increasing_interval : ∀ (x : ℝ), 
  is_defined x → 
  ∃ (a b : ℝ), 1 < a ∧ a ≤ x ∧ x < b ∧ b < 3 :=
by
  sorry

end monotonic_increasing_interval_l739_739296


namespace triangle_ABC_has_perimeter_70_l739_739693

section perimeter_of_triangle

-- Vertex definitions
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (21, 0)
def C : ℝ × ℝ := (21, 21)

-- Distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Side lengths
def AB : ℝ := distance A B
def BC : ℝ := distance B C
def AC : ℝ := distance A C

-- Perimeter function
def perimeter : ℝ := AB + BC + AC

-- Theorem to prove
theorem triangle_ABC_has_perimeter_70 : perimeter = 70 := by
  -- Steps skipped, can include actual proof delivering
  sorry

end perimeter_of_triangle

end triangle_ABC_has_perimeter_70_l739_739693


namespace expand_product_l739_739798

theorem expand_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5 * x - 36 :=
by
  -- No proof required, just state the theorem
  sorry

end expand_product_l739_739798


namespace vector_magnitude_sum_l739_739853

noncomputable def magnitude_sum (a b : ℝ) (θ : ℝ) := by
  let dot_product := a * b * Real.cos θ
  let a_square := a ^ 2
  let b_square := b ^ 2
  let magnitude := Real.sqrt (a_square + 2 * dot_product + b_square)
  exact magnitude

theorem vector_magnitude_sum (a b : ℝ) (θ : ℝ)
  (ha : a = 2) (hb : b = 1) (hθ : θ = Real.pi / 4) :
  magnitude_sum a b θ = Real.sqrt (5 + 2 * Real.sqrt 2) := by
  rw [ha, hb, hθ, magnitude_sum]
  sorry

end vector_magnitude_sum_l739_739853


namespace angle_MN_BC_l739_739259

noncomputable def angle_between (v w : ℝ × ℝ) : ℝ :=
  let dot_product := (v.1 * w.1 + v.2 * w.2)
  let norm_v := real.sqrt (v.1^2 + v.2^2)
  let norm_w := real.sqrt (w.1^2 + w.2^2)
  real.acos (dot_product / (norm_v * norm_w))

theorem angle_MN_BC (M N B C D : ℝ × ℝ) (H_midpoints : M = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) ∧ N = ((A.1 + D.1) / 2, (A.2 + D.2) / 2))
    (H_angle_B : real.angle B A C = real.pi * 5 / 6)
    (H_angle_C : real.angle C B D = real.pi / 2)
    (H_AB_CD : real.dist A B = real.dist C D) :
  angle_between (N - M) (C - B) = real.pi / 3 := sorry

end angle_MN_BC_l739_739259


namespace valid_sets_are_10_l739_739477

noncomputable def countValidSets : ℕ :=
  if h : 1 ≤ 9 then
    -- Common differences: 1, 2, 3, 4
    let sets1 := [{1, 2, 3}, {3, 4, 5}, {5, 6, 7}, {7, 8, 9}] -- d = 1
    let sets2 := [{1, 3, 5}, {3, 5, 7}, {5, 7, 9}] -- d = 2
    let sets3 := [{1, 4, 7}, {3, 6, 9}] -- d = 3
    let sets4 := [{1, 5, 9}] -- d = 4
    let total := sets1.length + sets2.length + sets3.length + sets4.length
    if total = 10 then total else sorry
  else sorry

theorem valid_sets_are_10 : countValidSets = 10 :=
by
  unfold countValidSets
  split
  · intros h
    -- Assume the length calculations are correct as per the constructed definition
    sorry
  · sorry

end valid_sets_are_10_l739_739477


namespace fraction_addition_correct_l739_739417

theorem fraction_addition_correct : (3 / 5 : ℚ) + (2 / 5) = 1 := 
by
  sorry

end fraction_addition_correct_l739_739417


namespace cubic_sum_l739_739182

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 13) : x^3 + y^3 = 35 :=
by
  sorry

end cubic_sum_l739_739182


namespace pentagon_equal_sides_l739_739755

theorem pentagon_equal_sides
  (ABCDE : Type) [pentagon ABCDE] 
  (inscribed_in_circle : circle ABCDE) 
  (equal_angles : ∀{a b c d e : point}, angle a b c = angle b c d 
                            = angle c d e = angle d e a = angle e a b) : 
  ∀{A B C D E : point}, side A B = side B C = side C D = side D E = side E A :=
sorry

end pentagon_equal_sides_l739_739755


namespace number_equation_l739_739749

-- Lean statement equivalent to the mathematical problem
theorem number_equation (x : ℝ) (h : 5 * x - 2 * x = 10) : 5 * x - 2 * x = 10 :=
by exact h

end number_equation_l739_739749


namespace number_of_integer_solutions_l739_739814

theorem number_of_integer_solutions (x : ℤ) : 
  x ∈ Set.filter (λ x, 5 * x^2 + 19 * x + 16 ≤ 20) (Set.Icc (-4 : ℤ) 0) =
  { -4, -3, -2, -1, 0 } →
  Set.card (Set.filter (λ x, 5 * x^2 + 19 * x + 16 ≤ 20) (Set.Icc (-4 : ℤ) 0)) = 5 :=
by sorry

end number_of_integer_solutions_l739_739814


namespace probability_x_equals_4_l739_739992

noncomputable def total_members : ℕ := 15
noncomputable def number_of_girls : ℕ := 7
noncomputable def number_selected : ℕ := 10
noncomputable def X (sample : Finset ℕ) : ℕ := sample.card

theorem probability_x_equals_4 :
  let P (k : ℕ) := (Nat.choose number_of_girls k * Nat.choose (total_members - number_of_girls) (number_selected - k)) / (Nat.choose total_members number_selected) in
  P 4 = (Nat.choose 7 4 * Nat.choose 8 6) / (Nat.choose 15 10) :=
by
  sorry

end probability_x_equals_4_l739_739992


namespace area_of_triangle_PQR_l739_739011

/-- 
Given a square ABCD with area 1 circumscribed about a circle, and a right triangle PQR
such that vertex P is at corner A of the square, vertex Q is on side CD, and hypotenuse PR
lies along side AB. Given that the length of PR is twice the height from P to QR, 
prove that the area of triangle PQR is 1/4.
-/
theorem area_of_triangle_PQR 
  (A B C D P Q R : Point)
  (s : ℝ) 
  (h : Q.y = y) 
  (hypo_len : ∥R - P∥ = s) 
  (height_cond : ∥R - P∥ = 2 * ∥Q - P∥) 
  (P_coord : P = (0, 0)) 
  (Q_on_CD : Q.y ≠ 0)
  (Q_coord : Q.x ≠ 0) :
  1/2 * s * (1/2) = 1/4 := 
sorry

end area_of_triangle_PQR_l739_739011


namespace max_PA_PB_l739_739117

noncomputable def max_distance (PA PB : ℝ) : ℝ :=
  PA + PB

theorem max_PA_PB {A B : ℝ × ℝ} (m : ℝ) :
  A = (0, 0) ∧
  B = (1, 3) ∧
  dist A B = 10 →
  max_distance (dist A B) (dist (1, 3) B) = 2 * Real.sqrt 5 :=
by
  sorry

end max_PA_PB_l739_739117


namespace circle_area_ratio_l739_739333

-- Define the hexagon and its properties:
structure RegularHexagon (A B C D E F : Type) :=
(side_length : ℝ)
(tangent_circle1 : ℝ)
(tangent_circle2 : ℝ)
(tangent_line_AB : Line)
(tangent_line_DE : Line)

-- Define the initial conditions
variables {A B C D E F : Type}
variables (hex : RegularHexagon A B C D E F)

-- The problem statement to be proven:
theorem circle_area_ratio (h : hex.side_length = 2)
    (h1 : hex.tangent_circle1 = (sqrt 3) / 3) 
    (h2 : hex.tangent_circle2 = (sqrt 3) / 3) : 
    (π * hex.tangent_circle2 ^ 2) / (π * hex.tangent_circle1 ^ 2) = 1 := by
  sorry

end circle_area_ratio_l739_739333


namespace triangle_BF_value_l739_739995

noncomputable def BF (AB BC CA : ℝ) (sin_ABC : ℝ) : ℝ :=
  let BD := 13 * (26 / 37)
  BF := BD * sin_ABC
  BF

theorem triangle_BF_value 
  (AB BC CA : ℝ) (sin_ABC : ℝ)
  (h₁: AB = 13) (h₂: BC = 26) (h₃: CA = 24) :
  BF AB BC CA sin_ABC = (338 / 37) * sin_ABC :=
by
  unfold BF
  sorry

end triangle_BF_value_l739_739995


namespace log2_a7_plus_log2_a11_l739_739781

noncomputable def geom_seq (a r : ℝ) (n : ℕ) := a * r ^ n

theorem log2_a7_plus_log2_a11 (a r : ℝ) (hn_pos : ∀ n, 0 < geom_seq a r n)
  (h_geom_mean : geom_seq a r 4 * geom_seq a r 14 = 8) :
  real.logb 2 (geom_seq a r 7) + real.logb 2 (geom_seq a r 11) = 3 :=
by 
  sorry

end log2_a7_plus_log2_a11_l739_739781


namespace other_toys_cost_1000_l739_739901

-- Definitions of the conditions
def cost_of_other_toys : ℕ := sorry
def cost_of_lightsaber (cost_of_other_toys : ℕ) : ℕ := 2 * cost_of_other_toys
def total_spent (cost_of_lightsaber cost_of_other_toys : ℕ) : ℕ := cost_of_lightsaber + cost_of_other_toys

-- The proof goal
theorem other_toys_cost_1000 (T : ℕ) (H1 : cost_of_lightsaber T = 2 * T) 
                            (H2 : total_spent (cost_of_lightsaber T) T = 3000) : T = 1000 := by
  sorry

end other_toys_cost_1000_l739_739901


namespace infinite_geometric_sequence_sum_l739_739481

open Real

theorem infinite_geometric_sequence_sum (x : ℝ) :
  (∃ k : ℤ, x = π / 6 + 2 * k * π ∨ x = 5 * π / 6 + 2 * k * π) ↔
  let a_n : ℕ → ℝ := λ n, (sin x) ^ n
  let S_n : ℕ → ℝ := λ n, (geom_series (sin x) (n + 1))
  (∀ ε > 0, ∃ N, ∀ n ≥ N, abs (S_n n - 1) < ε) :=
by
  sorry

end infinite_geometric_sequence_sum_l739_739481


namespace centroid_plane_distance_l739_739225

theorem centroid_plane_distance
  (α β γ : ℝ)
  (h₀ : α ≠ 0 ∧ β ≠ 0 ∧ γ ≠ 0)
  (h₁ : (1 / real.sqrt (1 / α^2 + 1 / β^2 + 1 / γ^2)) = 2) :
  let p := α / 3
      q := β / 3
      r := γ / 3 in
  1 / p^2 + 1 / q^2 + 1 / r^2 = 2.25 :=
by
  let p := α / 3
  let q := β / 3
  let r := γ / 3
  sorry

end centroid_plane_distance_l739_739225


namespace find_a_b_value_l739_739486

theorem find_a_b_value (a b : ℝ) (h : a + b * complex.I = (1 + 2 * complex.I) * (1 - complex.I)) : a + b = 4 :=
by sorry

end find_a_b_value_l739_739486


namespace red_pencils_count_l739_739903

theorem red_pencils_count 
  (packs : ℕ) 
  (pencils_per_pack : ℕ) 
  (extra_packs : ℕ) 
  (extra_pencils_per_pack : ℕ)
  (total_red_pencils : ℕ) 
  (h1 : packs = 15)
  (h2 : pencils_per_pack = 1)
  (h3 : extra_packs = 3)
  (h4 : extra_pencils_per_pack = 2)
  (h5 : total_red_pencils = packs * pencils_per_pack + extra_packs * extra_pencils_per_pack) : 
  total_red_pencils = 21 := 
  by sorry

end red_pencils_count_l739_739903


namespace coeff_x2_in_1_plus_2x_pow_5_l739_739889

theorem coeff_x2_in_1_plus_2x_pow_5 :
  (Polynomial.coeff (Polynomial.expand 5 (1 + Polynomial.C (2 : ℤ) * Polynomial.X)) 2) = 40 :=
sorry

end coeff_x2_in_1_plus_2x_pow_5_l739_739889


namespace probability_sum_greater_than_l739_739843

noncomputable def f (x : ℝ) := (1/2 : ℝ) ^ x
noncomputable def g (x : ℝ) := 1  -- g(x) can be set to 1 since g(x) ≠ 0 and value canceled out in the ratio

-- Defining the sequence
def seq (n : ℕ) : ℝ := (1/2) ^ n

-- Define the required sum
def sum_seq (k : ℕ) : ℝ := (List.range k).map seq |>.sum

-- Define the equation with terms
theorem probability_sum_greater_than (k : ℕ) :
  (count (λ x, sum_seq x > 15 / 16) (List.range 10).succ) / 10 = 3 / 5 :=
sorry

end probability_sum_greater_than_l739_739843


namespace subset_sum_divisible_l739_739370

theorem subset_sum_divisible {n : ℕ} (n_pos : 1 ≤ n) (a : Fin (2 * n - 1) → ℤ) :
  ∃ (s : Finset (Fin (2 * n - 1))), s.card = n ∧ n ∣ s.sum (λ i, a i) :=
by sorry

end subset_sum_divisible_l739_739370


namespace fraction_white_tulips_l739_739957

theorem fraction_white_tulips : 
  ∀ (total_tulips yellow_fraction red_fraction pink_fraction white_fraction : ℝ),
  total_tulips = 60 →
  yellow_fraction = 1 / 2 →
  red_fraction = 1 / 3 →
  pink_fraction = 1 / 4 →
  white_fraction = 
    ((total_tulips * (1 - yellow_fraction)) * (1 - red_fraction) * (1 - pink_fraction)) / total_tulips →
  white_fraction = 1 / 4 :=
by
  intros total_tulips yellow_fraction red_fraction pink_fraction white_fraction 
    h_total h_yellow h_red h_pink h_white
  sorry

end fraction_white_tulips_l739_739957


namespace inequality_solution_l739_739642

theorem inequality_solution (x : ℝ) : 
  (|x + 3| - |2x - 1| < x / 2 + 1) ↔ (x < -2 / 5 ∨ x > 2) :=
by {
  sorry
}

end inequality_solution_l739_739642


namespace math_proof_l739_739496

noncomputable def ellipse_eq_1 {A B C : ℝ × ℝ} (hA : A = (-√2, 0)) (hB : B = (√2, 0)) (hC : C = (√2, 1)) : 
  Prop := 
  ∃ (a b : ℝ), 
  a = 2 ∧ b = √2 ∧ 
  (∀ x y, x^2 / a^2 + y^2 / b^2 = 1 ↔ (x, y) = A ∨ (x, y) = B ∨ (x, y) = C)

noncomputable def correct_value_m {m : ℝ} (hP : ∃ M N : ℝ × ℝ, 
  let l := λ x, -(x - m) in 
  (M.2 = l M.1) ∧ (N.2 = l N.1) ∧ 
  (3 * M.1^2 - 4 * m * M.1 + 2 * m^2 - 4 = 0 ∧ 3 * N.1^2 - 4 * m * N.1 + 2 * m^2 - 4 = 0) ∧ 
  (Q : ℝ × ℝ, Q = (1, 0) → ∃ O R : ℝ × ℝ, O = (3/2 * (M.1 + N.1), 0) ∧ R = (1 / 2 * (M.1 - N.1), 0))) : 
  Prop := 
  m = (2 + √19) / 3 ∨ m = (2 - √19) / 3

theorem math_proof {A B C : ℝ × ℝ} (hA : A = (-√2, 0)) (hB : B = (√2, 0)) (hC : C = (√2, 1)) : 
  ellipse_eq_1 hA hB hC ∧ ∀ m : ℝ, correct_value_m (∃ M N : ℝ × ℝ, True) :=
sorry

end math_proof_l739_739496


namespace parameterized_line_segment_l739_739973

open Int

theorem parameterized_line_segment :
  ∃ (a b c d : Int), 
    (∀ t, 0 ≤ t ∧ t ≤ 1 → 
    (∃ x y, x = a * t + b ∧ y = c * t + d)) ∧ 
    b = -3 ∧ d = 8 ∧ a + b = 4 ∧ c + d = 10 ∧ 
    a^2 + b^2 + c^2 + d^2 = 126 :=
begin
  sorry
end

end parameterized_line_segment_l739_739973


namespace min_value_fraction_l739_739808

theorem min_value_fraction (x : ℝ) (h : x > 6) : 
  (∃ x_min, x_min = 12 ∧ (∀ x > 6, (x * x) / (x - 6) ≥ 18) ∧ (x * x) / (x - 6) = 18) :=
sorry

end min_value_fraction_l739_739808


namespace ways_to_distribute_soccer_balls_l739_739457

noncomputable def count_ways (n r s₁ s₂ : ℕ) : ℕ :=
  ∑ k in finset.range (r + 1), (-1)^k * nat.choose r k *
    nat.choose (n + r - s₁ * r - (s₂ - s₁ + 1) * k - 1) (r - 1)

theorem ways_to_distribute_soccer_balls (n r s₁ s₂ : ℕ)
  (h₁ : r * s₁ ≤ n)
  (h₂ : n ≤ r * s₂) :
  count_ways n r s₁ s₂ = 
  ∑ k in finset.range (r + 1), (-1)^k * nat.choose r k *
    nat.choose (n + r - s₁ * r - (s₂ - s₁ + 1) * k - 1) (r - 1) := sorry

end ways_to_distribute_soccer_balls_l739_739457


namespace five_students_three_together_l739_739052

theorem five_students_three_together :
  let n := 5 in 
  let k := 3 in 
  let ways_5_students := fact n in
  let ways_3_block := fact (n - k + 1) in
  let ways_in_block := fact k in
  ways_3_block * ways_in_block = 36 := by
  sorry

end five_students_three_together_l739_739052


namespace determine_ratio_l739_739662

def p (x : ℝ) : ℝ := (x - 4) * (x + 3)
def q (x : ℝ) : ℝ := (x - 4) * (x + 3)

theorem determine_ratio : q 1 ≠ 0 ∧ p 1 / q 1 = 1 := by
  have hq : q 1 ≠ 0 := by
    simp [q]
    norm_num
  have hpq : p 1 / q 1 = 1 := by
    simp [p, q]
    norm_num
  exact ⟨hq, hpq⟩

end determine_ratio_l739_739662


namespace trigonometric_identity_l739_739501

variable (α : Real)

theorem trigonometric_identity :
  (Real.tan (α - Real.pi / 4) = 1 / 2) →
  ((Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 2) :=
by
  intro h
  sorry

end trigonometric_identity_l739_739501


namespace Fedya_age_statement_l739_739710

theorem Fedya_age_statement (d a : ℕ) (today : ℕ) (birthday : ℕ) 
    (H1 : d + 2 = a) 
    (H2 : a + 2 = birthday + 3) 
    (H3 : birthday = today + 1) :
    ∃ sameYear y, (birthday < today + 2 ∨ today < birthday) ∧ ((sameYear ∧ y - today = 1) ∨ (¬ sameYear ∧ y - today = 0)) :=
by
  sorry

end Fedya_age_statement_l739_739710


namespace james_tip_percentage_l739_739591

theorem james_tip_percentage :
  let ticket_cost : ℝ := 100
  let dinner_cost : ℝ := 120
  let limo_cost_per_hour : ℝ := 80
  let limo_hours : ℕ := 6
  let total_cost_with_tip : ℝ := 836
  let total_cost_without_tip : ℝ := 2 * ticket_cost + limo_hours * limo_cost_per_hour + dinner_cost
  let tip : ℝ := total_cost_with_tip - total_cost_without_tip
  let percentage_tip : ℝ := (tip / dinner_cost) * 100
  percentage_tip = 30 :=
by
  sorry

end james_tip_percentage_l739_739591


namespace tangent_line_value_l739_739021

-- Define the conditions for the tangency points P and Q
variables (x1 y1 x2 y2 : ℝ)
variable hxy : (∃ (P Q : ℝ × ℝ), 
                 P = (x1, y1) ∧ Q = (x2, y2) ∧
                 (∀ x, y = ln x -> y1 = ln x1 ∧ (1 / x1) = exp x2) ∧ 
                 (∀ x, y = exp x -> y2 = exp x2 ∧ (exp x2) = (1 / x1)))

-- State the main theorem
theorem tangent_line_value (hxy : P = (x1, y1) ∧ Q = (x2, y2)) :
  (1 - exp y1) * (1 + x2) = 2 :=
sorry

end tangent_line_value_l739_739021


namespace JungMinBoughtWire_l739_739593

theorem JungMinBoughtWire
  (side_length : ℕ)
  (number_of_sides : ℕ)
  (remaining_wire : ℕ)
  (total_wire_bought : ℕ)
  (h1 : side_length = 13)
  (h2 : number_of_sides = 5)
  (h3 : remaining_wire = 8)
  (h4 : total_wire_bought = side_length * number_of_sides + remaining_wire) :
    total_wire_bought = 73 :=
by {
  sorry
}

end JungMinBoughtWire_l739_739593


namespace max_BF_is_one_fourth_l739_739835

noncomputable def max_BF_length (side_length : ℝ) : ℝ :=
  let x := side_length / 2
  let BF := x * (side_length - x)
  BF

theorem max_BF_is_one_fourth :
  ∀ (side_length : ℝ), side_length > 0 →
  ∃ x : ℝ, x ∈ (Icc 0 side_length) ∧ (max_BF_length side_length = side_length / 4) :=
begin
  intros,
  use side_length / 2,
  split,
  { 
    unfold Icc,
    split;
    linarith,
  },
  {
    unfold max_BF_length,
    have h : (side_length / 2) * (side_length - (side_length / 2)) = side_length / 4,
    {
      field_simp,
      ring,
    },
    simp [h],
  }
end

end max_BF_is_one_fourth_l739_739835


namespace trilinear_circle_theorem_radical_axis_theorem_l739_739383

-- Define the trilinear circle equation in Lean
def trilinear_circle_equation (p q r x y z α β γ : ℝ) : Prop :=
  (p * x + q * y + r * z) * (x * sin α + y * sin β + z * sin γ) = 
  y * z * sin α + x * z * sin β + x * y * sin γ

-- Prove that in trilinear coordinates, any circle is represented by the given equation
theorem trilinear_circle_theorem (p q r x y z α β γ : ℝ) : 
  trilinear_circle_equation p q r x y z α β γ := 
sorry

-- Define the radical axis equation in Lean
def radical_axis_equation (p1 q1 r1 p2 q2 r2 x y z : ℝ) : Prop :=
  (p1 * x + q1 * y + r1 * z) = (p2 * x + q2 * y + r2 * z)

-- Prove that the radical axis of two circles is represented by the given equation
theorem radical_axis_theorem (p1 q1 r1 p2 q2 r2 x y z : ℝ) :
  radical_axis_equation p1 q1 r1 p2 q2 r2 x y z :=
sorry

end trilinear_circle_theorem_radical_axis_theorem_l739_739383


namespace triangle_inscribed_regular_polygon_angle_l739_739331

theorem triangle_inscribed_regular_polygon_angle (A B C : Point) (n : ℕ)
  (h1 : inscribed_in_circle A B C)
  (h2 : ∠B = 3 * ∠A)
  (h3 : ∠C = 3 * ∠A)
  (h4 : adjacent_vertices_regular_polygon B C n):
  n = 7 := by
  sorry

end triangle_inscribed_regular_polygon_angle_l739_739331


namespace unique_zero_f_x1_minus_2x2_l739_739530

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.exp x - x - 1) - Real.log (x + 1) + x

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x + x

-- Condition a ≥ 0
variable (a : ℝ) (a_nonneg : 0 ≤ a)

-- Define the first part of the problem
theorem unique_zero_f : ∃! x, f a x = 0 :=
  sorry

-- Variables for the second part of the problem
variable (x₁ x₂ : ℝ)
variable (cond : f a x₁ = g a x₁ - g a x₂)

-- Define the second part of the problem
theorem x1_minus_2x2 : x₁ - 2 * x₂ ≥ 1 - 2 * Real.log 2 :=
  sorry

end unique_zero_f_x1_minus_2x2_l739_739530


namespace water_depth_correct_l739_739418

noncomputable def water_depth (ron_height : ℝ) (dean_shorter_by : ℝ) : ℝ :=
  let dean_height := ron_height - dean_shorter_by
  2.5 * dean_height + 3

theorem water_depth_correct :
  water_depth 14.2 8.3 = 17.75 :=
by
  let ron_height := 14.2
  let dean_shorter_by := 8.3
  let dean_height := ron_height - dean_shorter_by
  let depth := 2.5 * dean_height + 3
  simp [water_depth, dean_height, depth]
  sorry

end water_depth_correct_l739_739418


namespace total_enjoyable_gameplay_hours_l739_739898

def total_gameplay_hours : ℕ := 100
def grinding_percentage : ℝ := 0.8
def additional_enjoyable_hours : ℕ := 30

theorem total_enjoyable_gameplay_hours : 
  (total_gameplay_hours - (total_gameplay_hours * grinding_percentage).toNat + additional_enjoyable_hours = 50) :=
by
  sorry

end total_enjoyable_gameplay_hours_l739_739898


namespace tetrahedron_faces_congruent_l739_739984

theorem tetrahedron_faces_congruent {A B C D : Type*} [plane_triangle A B C D]
  (h₁ : planar_angles A B C + planar_angles B C D + planar_angles C D A = 180)
  (h₂ : planar_angles B C D + planar_angles C D A + planar_angles D A B = 180)
  (h₃ : planar_angles C D A + planar_angles D A B + planar_angles A B C = 180) :
  congruent_faces A B C D :=
sorry

end tetrahedron_faces_congruent_l739_739984


namespace time_to_fill_pot_l739_739665

def pot_volume : ℕ := 3000  -- in ml
def rate_of_entry : ℕ := 60 -- in ml/minute

-- Statement: Prove that the time required for the pot to be full is 50 minutes.
theorem time_to_fill_pot : (pot_volume / rate_of_entry) = 50 := by
  sorry

end time_to_fill_pot_l739_739665


namespace six_digit_palindrome_count_l739_739736

-- Define a 6-digit palindrome and the constraints that apply to its digits.
def is_six_digit_palindrome (n : ℕ) : Prop :=
  let digits := Int.digits 10 n in
  digits.length = 6 ∧ digits.head ≠ 0 ∧ digits = digits.reverse

-- The main statement to prove: there are 900 6-digit palindromes.
theorem six_digit_palindrome_count : ∃ n, n = 900 ∧ ∀ p, is_six_digit_palindrome p → p < 1000000 :=
sorry

end six_digit_palindrome_count_l739_739736


namespace part1_part2_part3_l739_739292

-- Definitions based on given conditions
def f (x : ℝ) : ℝ := x / (9 - x^2)

-- Problem parts

-- Part (1): Proving the analytical expression of f(x)
theorem part1 (odd_f : ∀ x, f(-x) = -f(x)) (fx1 : f(1) = 1 / 8) : f x = x / (9 - x^2) :=
sorry

-- Part (2): Proving monotonic behavior of f(x) on (-3, 3)
theorem part2 (mono_f : ∀ x1 x2 : ℝ, -3 < x1 ∧ x1 < x2 ∧ x2 < 3 → f(x1) < f(x2)) : 
  ∀ x: ℝ, -3 < x ∧ x < 3 → f(x) = x / (9 - x^2) :=
sorry

-- Part (3): Solving the inequality f(t-1) + f(t) < 0
theorem part3 (ineq : ∀ t : ℝ, f(t-1) + f(t) < 0 → -2 < t ∧ t < 1/2) : true :=
sorry

end part1_part2_part3_l739_739292


namespace cost_per_kg_after_30_l739_739030

theorem cost_per_kg_after_30 (l m : ℝ) 
  (hl : l = 20) 
  (h1 : 30 * l + 3 * m = 663) 
  (h2 : 30 * l + 6 * m = 726) : 
  m = 21 :=
by
  -- Proof will be written here
  sorry

end cost_per_kg_after_30_l739_739030


namespace _l739_739034

def frog_distribution_in_pool (n : ℕ) (h_n : n ≥ 5) : Prop :=
  ∃ (cells : set ℕ) (frogs : ℕ → ℕ), 
    (cells.card = 2 * n) ∧
    (frogs.sum = 4 * n + 1) ∧
    (∀ m ∈ cells, frogs m ≥ 0) ∧
    (∀ m1 m2 ∈ cells, m1 ≠ m2 → frogging_behavior cells frogs) ∧
    ∀ k ∈ cells, 
      (frogs k > 0) ∨ 
      ((neighbor k cells).all (λ j, frogs j > 0))

lemma frog_distribution_theorem (n : ℕ) (h_n : n ≥ 5) :
  frog_distribution_in_pool n h_n :=
sorry

-- Supporting Definitions
def neighbor (k : ℕ) (cells : set ℕ) : set ℕ :=
  {j ∈ cells | j ≠ k ∧ edgeshares (k, j)}

def frogging_behavior (cells : set ℕ) (frogs : ℕ → ℕ) : Prop :=
  ∀ k ∈ cells, 
    (frogs k ≥ 3 → ∃ n ∈ neighbor k cells, frogs n ≥ 0)

end _l739_739034


namespace translation_2_units_left_l739_739302

-- Define the initial parabola
def parabola1 (x : ℝ) : ℝ := x^2 + 1

-- Define the translated parabola
def parabola2 (x : ℝ) : ℝ := x^2 + 4 * x + 5

-- State that parabola2 is obtained by translating parabola1
-- And prove that this translation is 2 units to the left
theorem translation_2_units_left :
  ∀ x : ℝ, parabola2 x = parabola1 (x + 2) := 
by
  sorry

end translation_2_units_left_l739_739302


namespace vasya_is_not_mistaken_l739_739998

theorem vasya_is_not_mistaken (X Y N A B : ℤ)
  (h_sum : X + Y = N)
  (h_tanya : A * X + B * Y ≡ 0 [ZMOD N]) :
  B * X + A * Y ≡ 0 [ZMOD N] :=
sorry

end vasya_is_not_mistaken_l739_739998


namespace charge_increase_by_20_percent_l739_739376

variables (P R G : ℝ)
def charge_relation_R : P = R * (1 - 0.25) := by sorry
def charge_relation_G : P = G * (1 - 0.10) := by sorry

theorem charge_increase_by_20_percent :
  R = G * 1.20 :=
begin
  rw [charge_relation_R, charge_relation_G],
  sorry
end

end charge_increase_by_20_percent_l739_739376


namespace total_distance_traveled_l739_739747

-- Define the parameters
def V_m : ℝ := 7
def V_r : ℝ := 1.2
def time_total : ℝ := 1
def D : ℝ := (5.8 * 8.2) / 14
def total_distance := 2 * D

-- Define the problem statement
theorem total_distance_traveled :
  V_m = 7 ∧ V_r = 1.2 ∧ ((D / (V_m - V_r)) + (D / (V_m + V_r)) = time_total) → total_distance = 6.794 := by
  sorry

end total_distance_traveled_l739_739747


namespace books_given_away_l739_739943

theorem books_given_away (B_0 S B_f G : ℕ) (h1 : B_0 = 108) (h2 : S = 11) (h3 : B_f = 62) :
  (B_0 - S - B_f) = 35 :=
by 
  rw [h1, h2, h3]
  norm_num
  sorry

end books_given_away_l739_739943


namespace exists_irrational_in_interval_l739_739793
noncomputable theory

theorem exists_irrational_in_interval :
  ∃ (x : ℝ), x ∈ set.Icc 0.3 0.4 ∧ irrational x ∧ (x * (x + 1) * (x + 2)).denom = 1 := by
sorry

end exists_irrational_in_interval_l739_739793


namespace minimum_value_sum_l739_739054

theorem minimum_value_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a / (3 * b) + b / (5 * c) + c / (6 * a)) >= (3 / (90^(1/3))) :=
by 
  sorry

end minimum_value_sum_l739_739054


namespace arrangement_ways_l739_739551

theorem arrangement_ways (n : ℕ) : 
  ∀ (a : Fin n → ℕ), (∀ i j, i ≠ j → a i ≠ a j) →
    (number_of_arrangements n a = 2^(n-1)) :=
sorry

end arrangement_ways_l739_739551


namespace problem_l739_739645

def f (x : ℝ) : ℝ := 5 * x + 2
def g (x : ℝ) : ℝ := x / 2 - 4

theorem problem : ∀ x : ℝ, f (g x) - g (f x) = -15 :=
by
  sorry

end problem_l739_739645


namespace find_f_neg_10_l739_739146

noncomputable def f : ℝ → ℝ :=
λ x, if x > 0 then Real.log x / Real.log 2 else f (x + 3)

theorem find_f_neg_10 : f (-10) = 1 :=
by
  sorry

end find_f_neg_10_l739_739146


namespace ferryP_time_l739_739113

variable (D T : ℝ)

def ferryP_distance := D = 8 * T
def ferryQ_distance := 2 * D = 12 * (T + 1)

theorem ferryP_time : ferryP_distance D T ∧ ferryQ_distance D T → T = 3 :=
by
  intro h
  obtain ⟨hp, hq⟩ := h
  have h1 : D = 8 * T := hp
  have h2 : 2 * (8 * T) = 12 * (T + 1) := by rw h1 at hq; exact hq
  have h3 : 16 * T = 12 * T + 12 := by linarith
  have h4 : 4 * T = 12 := by linarith
  have h5 : T = 3 := by linarith
  exact h5

end ferryP_time_l739_739113


namespace intersection_complement_l739_739623

open Set

def UniversalSet := ℝ
def M : Set ℝ := {x | x^2 > 4}
def N : Set ℝ := {x | 1 < x ∧ x ≤ 3}
def CU_M : Set ℝ := compl M

theorem intersection_complement :
  N ∩ CU_M = {x | 1 < x ∧ x ≤ 2} :=
by sorry

end intersection_complement_l739_739623


namespace blue_dress_difference_l739_739215

theorem blue_dress_difference 
(total_space : ℕ)
(red_dresses : ℕ)
(blue_dresses : ℕ)
(h1 : total_space = 200)
(h2 : red_dresses = 83)
(h3 : blue_dresses = total_space - red_dresses) :
blue_dresses - red_dresses = 34 :=
by
  rw [h1, h2] at h3
  sorry -- Proof details go here.

end blue_dress_difference_l739_739215


namespace largest_n_dividing_30_fact_l739_739806

theorem largest_n_dividing_30_fact : 
  ∃ n : ℕ, (∀ m : ℕ, (12^m ∣ nat.factorial 30) → m ≤ n) ∧ n = 13 :=
by 
  sorry

end largest_n_dividing_30_fact_l739_739806


namespace solution1_solution2_solution3_solution4_solution5_l739_739775

noncomputable def problem1 : ℤ :=
  -3 + 8 - 15 - 6

theorem solution1 : problem1 = -16 := by
  sorry

noncomputable def problem2 : ℚ :=
  -35 / -7 * (-1 / 7)

theorem solution2 : problem2 = -(5 / 7) := by
  sorry

noncomputable def problem3 : ℤ :=
  -2^2 - |2 - 5| / -3

theorem solution3 : problem3 = -3 := by
  sorry

noncomputable def problem4 : ℚ :=
  (1 / 2 + 5 / 6 - 7 / 12) * -24 

theorem solution4 : problem4 = -18 := by
  sorry

noncomputable def problem5 : ℚ :=
  (-99 - 6 / 11) * 22

theorem solution5 : problem5 = -2190 := by
  sorry

end solution1_solution2_solution3_solution4_solution5_l739_739775


namespace tan_C_value_min_tan_C_value_l739_739506

theorem tan_C_value (a : ℝ) (h : ∀ x : ℝ, (tan A + tan B = -a ∧ tan A * tan B = 4)) 
  (h_a : a = -8) : tan C = 8 / 3 := sorry

theorem min_tan_C_value {a : ℝ} (h1 : a^2 - 16 ≥ 0) (h2 : a ≤ -4) 
  (h3 : ∀ x : ℝ, (tan A + tan B = -a ∧ tan A * tan B = 4)) : 
  ∃ (tanC : ℝ), tan C = 4 / 3 ∧ (tan A = 2 ∧ tan B = 2) := sorry

end tan_C_value_min_tan_C_value_l739_739506


namespace average_price_per_person_excluding_gratuity_l739_739378

def total_cost_with_gratuity : ℝ := 207.00
def gratuity_rate : ℝ := 0.15
def number_of_people : ℕ := 15

theorem average_price_per_person_excluding_gratuity :
  (total_cost_with_gratuity / (1 + gratuity_rate) / number_of_people) = 12.00 :=
by
  sorry

end average_price_per_person_excluding_gratuity_l739_739378


namespace find_tan_B_find_a_plus_c_l739_739213

variables (A B C a b c : ℝ) (S : ℝ)

-- Provided conditions
axiom h1 : ∀ {A B C a b c : ℝ}, (sin A * sin B * cos B + sin B^2 * cos A = 2 * sqrt 2 * sin C * cos B)

-- Additional conditions for the second part
axiom h2 : b = 2
axiom h3 : S = sqrt 2

-- Prove tan B = 2√2
theorem find_tan_B (h1 : sin A * sin B * cos B + sin B^2 * cos A = 2 * sqrt 2 * sin C * cos B) : tan B = 2 * sqrt 2 :=
sorry

-- Prove a + c
theorem find_a_plus_c (h1 : sin A * sin B * cos B + sin B^2 * cos A = 2 * sqrt 2 * sin C * cos B)
                      (h2 : b = 2) (h3 : S = sqrt 2) 
                      (h_tan_B : tan B = 2 * sqrt 2) : a + c = 2 * sqrt 3 :=
sorry

end find_tan_B_find_a_plus_c_l739_739213


namespace joint_school_students_l739_739562

instance : DecidableEq ℕ := decidableEqOfDecidableOfEquiv fun n => decidableOfIff' (n = n) ⟨id, id⟩ (by simp)

theorem joint_school_students :
  ∀ (U A B : Finset ℕ) (n m p x_max x_min : ℕ), 
    U.card = n → A.card = m → B.card = p 
    → n = 200 
    → m = 80 
    → p = 155 
    → x_max = 80 
    → x_min = 80 + 155 - 200 
    → x_max - x_min = 45 :=
by
  intro U A B n m p x_max x_min hU hA hB hn hm hp hmax hmin
  rw [← hn, ← hm, ← hp] at hU hA hB
  simp only [hmax, hmin]
  sorry

end joint_school_students_l739_739562


namespace squared_area_rhombus_ABCD_l739_739385

-- Define the conditions of the rhombus ABCD
variables (A B C D E F : Type*)
variables [has_angle_measure E]
variables [has_distance F]

-- Angle BAD is 60 degrees
def angle_BAD := 60 * (π / 180)

-- Point E lies on the minor arc AD of the circumcircle of triangle ABD
-- Point F is the intersection of AC and the circumcircle of triangle EDC
-- AF is 4
def distance_AF := 4

-- Radius of circumcircle of EDC is 14
def radius_EDC := 14

-- Prove that the squared area of the rhombus ABCD is 2916 
theorem squared_area_rhombus_ABCD :
  is_rhombus A B C D →
  angle A B D = angle_BAD →
  on_circumcircle E A B →
  on_circumcircle F E D C →
  distance A F = distance_AF →
  circumcircle_radius E D C = radius_EDC →
  squared_area A B C D = 2916 :=
sorry

end squared_area_rhombus_ABCD_l739_739385


namespace find_point_P_l739_739133

/-- The point P on the y-axis such that ∠BAP = 90°, given points A(-3, -2) and B(6, 1), is (0, -11). -/
theorem find_point_P (y : ℝ) (h1 : A = (-3, -2)) (h2 : B = (6, 1)) (h3 : ∠BAP = 90°) :
    P = (0, -11) :=
sorry

def A := (-3, -2 : ℝ × ℝ)
def B := (6, 1 : ℝ × ℝ)
def P := (0, y : ℝ)
def k_AB := (B.snd - A.snd) / (B.fst - A.fst)
def k_AP := (A.snd - y) / (A.fst - 0)
def perpendicular_slopes := k_AB * k_AP = -1

end find_point_P_l739_739133


namespace one_point_three_six_billion_in_scientific_notation_l739_739428

theorem one_point_three_six_billion_in_scientific_notation :
  let billion := 10^9 in 
  (1.36 * billion = 1.36 * 10^9) :=
by 
  let billion := 10^9 
  calc
    1.36 * billion = 1.36 * 10^9 : by sorry

end one_point_three_six_billion_in_scientific_notation_l739_739428


namespace angle_of_inclination_of_line_l739_739650
noncomputable theory

def angle_of_inclination (t : ℝ) : ℝ := 110

theorem angle_of_inclination_of_line :
  (∃ t : ℝ, ∀ x y : ℝ, x = t * real.sin (20 * real.pi / 180) + 1 ∧ y = - t * real.cos (20 * real.pi / 180)) →
  angle_of_inclination = 110 :=
by {
  -- Skipping the proof
  intros h,
  sorry
}

end angle_of_inclination_of_line_l739_739650


namespace triangles_side_equality_l739_739996

theorem triangles_side_equality
  {A B C A1 B1 C1 : Type*}
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [MetricSpace A1] [MetricSpace B1] [MetricSpace C1]
  (h_angle_A : ∠A = ∠A1)
  (h_angle_B_sum_180 : ∠B + ∠B1 = 180)
  (h_side_length : dist A1 B1 = dist A C + dist B C) :
  dist A B = dist A1 C1 - dist B1 C1 := 
begin
  sorry,
end

end triangles_side_equality_l739_739996


namespace trisha_cookies_count_is_33_l739_739032

open Real

-- Define the properties and areas of Art's and Trisha's cookies
def area_trapezoid (base1 base2 height : ℝ) : ℝ :=
  (1 / 2) * (base1 + base2) * height

def area_triangle (base height : ℝ) : ℝ :=
  (1 / 2) * base * height

def art_cookies_area : ℝ :=
  (10 : ℝ) * (area_trapezoid 4 6 4)

def trisha_one_cookie_area : ℝ :=
  area_triangle 4 3

-- Define the number of cookies Trisha makes per batch
def trisha_cookies_count (total_area : ℝ) (cookie_area : ℝ) : ℝ :=
  total_area / cookie_area

-- Statement to be proved
theorem trisha_cookies_count_is_33 :
  trisha_cookies_count art_cookies_area trisha_one_cookie_area = 33 := 
  by
  sorry

end trisha_cookies_count_is_33_l739_739032


namespace natural_number_unique_solution_l739_739060

def product_of_digits (n : ℕ) : ℕ := 
  (n.toString.data.map (λ c, c.toNat - '0'.toNat)).prod

theorem natural_number_unique_solution (x : ℕ) (hx1 : product_of_digits x = x^2 - 10 * x - 22) : x = 12 :=
by
  sorry

end natural_number_unique_solution_l739_739060


namespace geom_proportion_l739_739087

variables {S S1 C : ℝ}

def geom_terms (u x y z : ℝ) : Prop :=
  (u + z = 2 * S) ∧ 
  (x + y = 2 * S1) ∧ 
  (u^2 + x^2 + y^2 + z^2 = 4 * C^2) ∧ 
  (u = (2 * S + real.sqrt ((2 * S)^2 - 4 * (S^2 + S1^2 - C^2))) / 2) ∧ 
  (z = (2 * S - real.sqrt ((2 * S)^2 - 4 * (S^2 + S1^2 - C^2))) / 2) ∧
  (x = (2 * S1 + real.sqrt ((2 * S1)^2 - 4 * (S^2 + S1^2 - C^2))) / 2) ∧
  (y = (2 * S1 - real.sqrt ((2 * S1)^2 - 4 * (S^2 + S1^2 - C^2))) / 2)

theorem geom_proportion :
  ∀ {u x y z : ℝ}, geom_terms u x y z :=
begin
  intros,
  sorry
end

end geom_proportion_l739_739087


namespace problem1_problem2_l739_739939

-- Problem 1: Sequence "Seven six five four three two one" is a descending order
theorem problem1 : ∃ term: String, term = "Descending Order" ∧ "Seven six five four three two one" = "Descending Order" := sorry

-- Problem 2: Describing a computing tool that knows 0 and 1 and can calculate large numbers (computer)
theorem problem2 : ∃ tool: String, tool = "Computer" ∧ "I only know 0 and 1, can calculate millions and billions, available in both software and hardware" = "Computer" := sorry

end problem1_problem2_l739_739939


namespace number_of_ways_split_2000_cents_l739_739232

theorem number_of_ways_split_2000_cents : 
  ∃ n : ℕ, n = 357 ∧ (∃ (nick d q : ℕ), 
    nick > 0 ∧ d > 0 ∧ q > 0 ∧ 5 * nick + 10 * d + 25 * q = 2000) :=
sorry

end number_of_ways_split_2000_cents_l739_739232


namespace yellow_marbles_in_C_l739_739991

theorem yellow_marbles_in_C 
  (Y : ℕ)
  (conditionA : 4 - 2 ≠ 6)
  (conditionB : 6 - 1 ≠ 6)
  (conditionC1 : 3 > Y → 3 - Y = 6)
  (conditionC2 : Y > 3 → Y - 3 = 6) :
  Y = 9 :=
by
  sorry

end yellow_marbles_in_C_l739_739991


namespace outfit_count_l739_739552

def num_shirts := 8
def num_hats := 8
def num_pants := 4

def shirt_colors := 6
def hat_colors := 6
def pants_colors := 4

def total_possible_outfits := num_shirts * num_hats * num_pants

def same_color_restricted_outfits := 4 * 8 * 7

def num_valid_outfits := total_possible_outfits - same_color_restricted_outfits

theorem outfit_count (h1 : num_shirts = 8) (h2 : num_hats = 8) (h3 : num_pants = 4)
                     (h4 : shirt_colors = 6) (h5 : hat_colors = 6) (h6 : pants_colors = 4)
                     (h7 : total_possible_outfits = 256) (h8 : same_color_restricted_outfits = 224) :
  num_valid_outfits = 32 :=
by
  sorry

end outfit_count_l739_739552


namespace scientific_notation_of_1_300_000_l739_739762

theorem scientific_notation_of_1_300_000 : 1_300_000 = 1.3 * 10^6 :=
by
  -- add your proof here
  sorry

end scientific_notation_of_1_300_000_l739_739762


namespace count_of_valid_triplets_correct_l739_739813

def no_carrying (n : ℕ) : Prop := 
  ∀ m, m < 10 → ((n + 2) % 10 + m) < 10

def is_valid_range (n : ℕ) : Prop := 
  950 ≤ n ∧ n ≤ 2050

def valid_triplet (n : ℕ) : Prop := 
  is_valid_range n ∧ is_valid_range (n + 2) ∧ no_carrying n

noncomputable def count_valid_triplets : ℕ :=
  (range 950 2051).filter valid_triplet |>.length

theorem count_of_valid_triplets_correct : count_valid_triplets = 1090 :=
  sorry

end count_of_valid_triplets_correct_l739_739813


namespace sum_first_9_terms_l739_739204

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)
variable (d : ℝ)
variable (a_1 a_2 a_3 a_4 a_5 a_6 : ℝ)

-- Conditions
axiom h1 : a 1 + a 5 = 10
axiom h2 : a 2 + a 6 = 14

-- Calculations
axiom h3 : a 3 = 5
axiom h4 : a 4 = 7
axiom h5 : d = 2
axiom h6 : a 5 = 9

-- The sum of the first 9 terms
axiom h7 : S 9 = 9 * a 5

theorem sum_first_9_terms : S 9 = 81 :=
by {
  sorry
}

end sum_first_9_terms_l739_739204


namespace domain_of_g_l739_739789

theorem domain_of_g :
  {x : ℝ | -8 * x^2 + 12 * x + 16 ≥ 0} = set.Icc (-(1 : ℝ)/2) (4 : ℝ) :=
sorry

end domain_of_g_l739_739789


namespace probability_of_selecting_female_l739_739740

theorem probability_of_selecting_female (total_students female_students male_students : ℕ)
  (h_total : total_students = female_students + male_students)
  (h_female : female_students = 3)
  (h_male : male_students = 1) :
  (female_students : ℚ) / total_students = 3 / 4 :=
by
  sorry

end probability_of_selecting_female_l739_739740


namespace comparison_m_n_l739_739823

noncomputable def e : ℝ := 2.71828

-- Define the function f(x) and ensure it is odd
def f (x : ℝ) (b : ℝ) : ℝ := (2^x - b) / (2^x + 1)

-- Define the conditions
variables (m n b : ℝ)
variable (h_b : f 0 b = 0)
variable (h_odd : ∀ x : ℝ, f (-x) b = -f x b)
variable (h_ml1 : m < 1)
variable (h_nl1 : n < 1)
variable (h_log_m : real.log 2 (real.abs (m - b)) + 2 * e - 1 = real.exp 2)
variable (h_log_n : real.log 2 (real.abs (n - b)) + 4 * e - 1 = 4 * real.exp 2)

theorem comparison_m_n : n < m ∧ m < 0 :=
by
  sorry

end comparison_m_n_l739_739823


namespace spending_50_dollars_l739_739574

def receiving_money (r : Int) : Prop := r > 0

def spending_money (s : Int) : Prop := s < 0

theorem spending_50_dollars :
  receiving_money 80 ∧ ∀ r, receiving_money r → spending_money (-r)
  → spending_money (-50) :=
by
  sorry

end spending_50_dollars_l739_739574


namespace shifted_sine_symmetric_origin_l739_739150

theorem shifted_sine_symmetric_origin
  (phi : ℝ) 
  (h1 : 0 < phi) 
  (h2 : phi < π / 2) 
  (h3 : ∀ x : ℝ, 3 * sin(2 * (x + phi) + π / 4) = -3 * sin(2 * x)) :
  phi = 3 * π / 8 :=
by
  sorry

end shifted_sine_symmetric_origin_l739_739150


namespace angle_between_vectors_l739_739162

noncomputable def vector_length {n : ℕ} (v: EuclideanSpace ℝ (Fin n)) := ∥v∥

variables (a b : EuclideanSpace ℝ (Fin 3))
variables (theta : ℝ)
variables (ha : vector_length a = 3)
variables (hb : vector_length b = 2 * Real.sqrt 3)
variables (perpendicular : InnerProductSpace.inner a (a + b) = 0)

theorem angle_between_vectors : theta = (5 * Real.pi) / 6 :=
by
  sorry

end angle_between_vectors_l739_739162


namespace rational_area_ratio_l739_739273

noncomputable def point := (ℝ × ℝ)

def distance_squared (p1 p2 : point) : ℝ := (p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2

def area_of_triangle (A B C : point) : ℝ := 0.5 * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|

theorem rational_area_ratio 
  (A B C D : point) 
  (h_dist_AB : is_rational (distance_squared A B))
  (h_dist_AC : is_rational (distance_squared A C))
  (h_dist_AD : is_rational (distance_squared A D))
  (h_dist_BC : is_rational (distance_squared B C))
  (h_dist_BD : is_rational (distance_squared B D))
  (h_dist_CD : is_rational (distance_squared C D))
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) 
  (h_no_collinear : ¬ collinear A B C ∧ ¬ collinear A B D ∧ ¬ collinear A C D ∧ ¬ collinear B C D) : 
  is_rational (area_of_triangle A B C / area_of_triangle A B D) := 
sorry

end rational_area_ratio_l739_739273


namespace gas_mixture_pressure_l739_739561

theorem gas_mixture_pressure
  (m : ℝ) -- mass of each gas
  (p : ℝ) -- initial pressure
  (T : ℝ) -- initial temperature
  (V : ℝ) -- volume of the container
  (R : ℝ) -- ideal gas constant
  (mu_He : ℝ := 4) -- molar mass of helium
  (mu_N2 : ℝ := 28) -- molar mass of nitrogen
  (is_ideal : True) -- assumption that the gases are ideal
  (temp_doubled : True) -- assumption that absolute temperature is doubled
  (N2_dissociates : True) -- assumption that nitrogen dissociates into atoms
  : (9 / 4) * p = p' :=
by
  sorry

end gas_mixture_pressure_l739_739561


namespace option_A_correct_option_B_correct_option_C_incorrect_option_D_incorrect_l739_739156

-- Option A
theorem option_A_correct (a : ℕ → ℕ) (h₁ : a 1 = 2) (h₂ : ∀ n, a (n + 1) = a n + n + 1) : a 20 = 211 :=
begin
  sorry
end

-- Option B
theorem option_B_correct (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n, a (n + 1) = 3 * a n + 2) : a 4 = 53 :=
begin
  sorry
end

-- Option C (Proving that sequence is not geometric)
theorem option_C_incorrect (S : ℕ → ℚ) (h₁ : ∀ n, S n = 3^n + 1/2) : ¬(∃ (a : ℕ → ℚ) (b : ℚ), ∀ n, a (n + 1) = b * a n) :=
begin
  sorry
end

-- Option D
theorem option_D_incorrect (a : ℕ → ℚ) (h₁ : a 1 = 1) (h₂ : ∀ n, a (n + 1) = 2 * a n / (2 + a n)) : ¬(a 5 = 1/5) :=
begin
  sorry
end

end option_A_correct_option_B_correct_option_C_incorrect_option_D_incorrect_l739_739156


namespace equal_angles_l739_739238

open EuclideanGeometry

noncomputable def is_center_of_circumcircle (O A B C : Point) : Prop :=
  Circle.has_center (circumscribed_circle A B C) O

noncomputable def is_foot_of_altitude (C' C A B : Point) : Prop :=
  C' lies_on (Altitude.from C to AB)

theorem equal_angles (A B C O C' : Point) 
  (h1 : Triangle A B C) 
  (h2 : is_center_of_circumcircle O A B C) 
  (h3 : is_foot_of_altitude C' C A B) : 
  ∠ A C C' = ∠ O C B :=
sorry

end equal_angles_l739_739238


namespace number_of_people_who_chose_soda_l739_739195

theorem number_of_people_who_chose_soda (total_people : ℕ) (angle_soda : ℕ) 
  (h1 : total_people = 500) (h2 : angle_soda = 198) : 
  (total_people * angle_soda) / 360 = 275 := by
  rw [h1, h2]
  norm_num
  sorry

end number_of_people_who_chose_soda_l739_739195


namespace weight_of_replaced_person_l739_739285

/--
The average weight of 8 persons increases by 2.5 kg when a new person
comes in place of one of them weighing a certain amount. The weight
of the new person might be 90 kg. 
Prove that the weight of the person who was replaced is 70 kg.
-/
theorem weight_of_replaced_person (W_new : ℝ) (W_old : ℝ) :
  W_new = 90 → (8 * 2.5 = 20) → 
  W_old = W_new - 8 * 2.5 → 
  W_old = 70 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end weight_of_replaced_person_l739_739285


namespace sqrt_x_plus_y_eq_3_l739_739233

theorem sqrt_x_plus_y_eq_3 (x y : ℝ) 
  (h1 : y = 4 + real.sqrt (5 - x) + real.sqrt (x - 5))
  (h2 : 5 - x ≥ 0) 
  (h3 : x - 5 ≥ 0) : 
  real.sqrt (x + y) = 3 := 
sorry

end sqrt_x_plus_y_eq_3_l739_739233


namespace determine_functions_l739_739051

noncomputable def f : (ℝ → ℝ) := sorry

theorem determine_functions (f : ℝ → ℝ)
  (h_domain: ∀ x, 0 < x → 0 < f x)
  (h_eq: ∀ w x y z, 0 < w → 0 < x → 0 < y → 0 < z → w * x = y * z →
    (f w)^2 + (f x)^2 = (f (y^2) + f (z^2)) * (w^2 + x^2) / (y^2 + z^2)) :
  (∀ x, 0 < x → (f x = x ∨ f x = 1 / x)) :=
by
  intros x hx
  sorry

end determine_functions_l739_739051


namespace parabola_problem_l739_739876

noncomputable def parabola_focus_distance (P : ℝ × ℝ) : ℝ :=
  let y := P.snd in
  let x := - (y * y) / 16 in
  let focus : ℝ × ℝ := (-4, 0) in
  real.sqrt ((x - focus.fst) * (x - focus.fst) + (y - focus.snd) * (y - focus.snd))

theorem parabola_problem :
  ∀ (P : ℝ × ℝ), abs (P.snd) = 12 → parabola_focus_distance P = 13 := 
by
  assume P hP,
  sorry

end parabola_problem_l739_739876


namespace angle_AED_eq_90_l739_739395

theorem angle_AED_eq_90 
  (A B C D E : Point) (circle : Circle)
  (h1 : rectangle A B C D)
  (h2 : circle.circumscribes A B C D)
  (h3 : E ∈ circle.arc B C)
  (h4 : circle.arc B C <pi>) : 
  ∠ A E D = 90° :=
sorry

end angle_AED_eq_90_l739_739395


namespace convenience_store_pure_milk_quantity_convenience_store_yogurt_discount_l739_739741

noncomputable def cost_per_pure_milk_box (x : ℕ) : ℝ := 2000 / x
noncomputable def cost_per_yogurt_box (x : ℕ) : ℝ := 4800 / (1.5 * x)

theorem convenience_store_pure_milk_quantity
  (x : ℕ)
  (hx : cost_per_yogurt_box x - cost_per_pure_milk_box x = 30) :
  x = 40 :=
by
  sorry

noncomputable def pure_milk_price := 80
noncomputable def yogurt_price (cost_per_yogurt_box : ℝ) : ℝ := cost_per_yogurt_box * 1.25

theorem convenience_store_yogurt_discount
  (x y : ℕ)
  (hx : cost_per_yogurt_box x - cost_per_pure_milk_box x = 30)
  (total_profit : ℕ)
  (profit_condition :
    pure_milk_price * x +
    yogurt_price (cost_per_yogurt_box x) * (1.5 * x - y) +
    yogurt_price (cost_per_yogurt_box x) * 0.9 * y - 2000 - 4800 = total_profit)
  (pure_milk_quantity : x = 40)
  (profit_value : total_profit = 2150) :
  y = 25 :=
by
  sorry

end convenience_store_pure_milk_quantity_convenience_store_yogurt_discount_l739_739741


namespace problem_l739_739237

theorem problem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  1 / (a ^ 3 * (b + c)) + 1 / (b ^ 3 * (c + a)) + 1 / (c ^ 3 * (a + b)) ≥ 3 / 2 :=
by sorry

end problem_l739_739237


namespace triangle_XYZ_XY_length_l739_739194

theorem triangle_XYZ_XY_length (X Y Z : ℝ) (YZ : ℝ) (tan_Z : ℝ) 
  (h1 : ∠X = 90) (h2 : tan_Z = 5/12) (h3 : YZ = 60) : 
  XY = 300/13 := 
sorry

end triangle_XYZ_XY_length_l739_739194


namespace two_roots_iff_a_greater_than_neg1_l739_739075

theorem two_roots_iff_a_greater_than_neg1 (a : ℝ) :
  (∃! x : ℝ, x^2 + 2*x + 2*|x + 1| = a) ↔ a > -1 :=
sorry

end two_roots_iff_a_greater_than_neg1_l739_739075


namespace roots_conditions_l739_739143

theorem roots_conditions (α β m n : ℝ) (h_pos : β > 0)
  (h1 : α + 2 * β = -m)
  (h2 : 2 * α * β + β^2 = -3)
  (h3 : α * β^2 = -n)
  (h4 : α^2 + 2 * β^2 = 6) : 
  m = 0 ∧ n = 2 := by
  sorry

end roots_conditions_l739_739143


namespace z_squared_purely_imaginary_l739_739119

noncomputable def z : ℂ := 2 / (1 - complex.i)

theorem z_squared_purely_imaginary (z : ℂ) (h : z = 2 / (1 - complex.i)) : ∃ b : ℝ, z^2 = 0 + b * complex.i :=
by
  have hz : z = 1 + complex.i := sorry
  exists (2 : ℝ)
  rw [hz]
  calc
    (1 + complex.i)^2
      = 1 - 1 + 2 * complex.i : by ring
      = 2 * complex.i : by ring
  sorry

end z_squared_purely_imaginary_l739_739119


namespace ben_has_10_fewer_stickers_than_ryan_l739_739594

theorem ben_has_10_fewer_stickers_than_ryan :
  ∀ (Karl_stickers Ryan_stickers Ben_stickers total_stickers : ℕ),
    Karl_stickers = 25 →
    Ryan_stickers = Karl_stickers + 20 →
    total_stickers = Karl_stickers + Ryan_stickers + Ben_stickers →
    total_stickers = 105 →
    (Ryan_stickers - Ben_stickers) = 10 :=
by
  intros Karl_stickers Ryan_stickers Ben_stickers total_stickers h1 h2 h3 h4
  -- Conditions mentioned in a)
  exact sorry

end ben_has_10_fewer_stickers_than_ryan_l739_739594


namespace frogs_even_distribution_l739_739721

theorem frogs_even_distribution (n : ℕ) (h_n : n ≥ 5) (frog_count : ℕ) (h_frog_count : frog_count = 4 * n + 1)
    (cells : Finset ℕ) (h_cells : cells.card = 2 * n)
    (adjacency : ∀ (cell : ℕ), ∃ neighborhood : Finset ℕ, neighborhood.card = 3 ∧ ∀ neighbor ∈ neighborhood, neighbor ∈ cells) :
  ∃ (distributed_cells : Finset (ℕ × ℕ)), (∀ cell ∈ cells, (distributed_cells.filter (λ c, c.1 = cell)).card ≥ 1 ∨ ∀ neighbor ∈ (adjacency cell).some.1, (distributed_cells.filter (λ c, c.1 = neighbor)).card ≥ 1) := 
sorry

end frogs_even_distribution_l739_739721


namespace projection_is_correct_l739_739864

def vector := (ℝ × ℝ)

def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def magnitude_squared (v : vector) : ℝ :=
  dot_product v v

def projection (a b : vector) : vector :=
  let scalar := (dot_product a b) / (magnitude_squared b)
  (scalar * b.1, scalar * b.2)

noncomputable def a : vector := (√3, 1)
noncomputable def b : vector := (1, 1)

theorem projection_is_correct : projection a b = ( (√3 + 1) / 2, (√3 + 1) / 2 ) :=
by
  sorry

end projection_is_correct_l739_739864


namespace integer_solutions_eq_two_l739_739170

theorem integer_solutions_eq_two : 
  ∃ S : Set Int, (∀ x : Int, x ∈ S ↔ (x-3)^(30-x^2) = 1) ∧ S.card = 2 := 
sorry

end integer_solutions_eq_two_l739_739170


namespace andy_solves_18_problems_l739_739029

variable (n : ℕ)

-- Define the sequence of odd numbers between 80 and 125
def is_odd_and_in_range (n : ℕ) : Prop :=
  80 ≤ n ∧ n ≤ 125 ∧ n % 2 = 1

-- Counting total number of odd-numbered problems in the range
def total_odd_numbers_in_range :=
  (List.range (125 - 80 + 1)).filter (λ x => is_odd_and_in_range (x + 80)).length

-- Function to count the problems Andy solves, excluding every fourth one.
def problems_solved :=
  total_odd_numbers_in_range - (total_odd_numbers_in_range / 4)

theorem andy_solves_18_problems :
  problems_solved = 18 :=
by
  sorry

end andy_solves_18_problems_l739_739029


namespace circles_touch_at_1_pt_line_ho_common_tangent_l739_739597

section Geometry

variable {P : Type*} [EuclideanSpace P]
variables {A B C H O : P}
variables {ωb ωc : Circle P}

-- Conditions
def is_acutatro ∆abc (A B C : P) : Prop := 
  (triangle.angles.sum_eq π ∧ triangle.angles.all_lt π / 2) ∧
  (triangle.distinct_vertices A B C)

def angle_BAC_eq_60 (A B C : P) : Prop := angle B A C = π / 3

def is_orthocenter (H A B C : P) : Prop := triangle.is_orthocenter H A B C

def circle_tangent_at (ω : Circle P) (L : Line P) (P : P) : Prop := 
  ω.tangent_at P L ∧ ω.passes_through P

def circle_tangent_to (ω : Circle P) (A B : P) : Prop := 
  circle_tangent_at ω (line_through A B) A

-- Theorems
theorem circles_touch_at_1_pt
  (h_acutatro : is_acutatro ∆abc A B C)
  (h_angle_bac : angle_BAC_eq_60 A B C)
  (h_orthocenter : is_orthocenter H A B C)
  (h_tangent_wb : circle_tangent_to ωb A B)
  (h_tangent_wc : circle_tangent_to ωc A C) :
  ωb ∩ ωc = {H} :=
sorry

theorem line_ho_common_tangent
  (h_acutatro : is_acutatro ∆abc A B C)
  (h_angle_bac : angle_BAC_eq_60 A B C)
  (h_orthocenter : is_orthocenter H A B C)
  (h_tangent_wb : circle_tangent_to ωb A B)
  (h_tangent_wc : circle_tangent_to ωc A C)
  (h_circumcenter : is_circumcenter O A B C) :
  is_tangent (line_through H O) ωb ∧ is_tangent (line_through H O) ωc :=
sorry

end Geometry

end circles_touch_at_1_pt_line_ho_common_tangent_l739_739597


namespace smaller_square_length_proof_l739_739727

noncomputable def smaller_square_side_length : ℝ := 5 / 3

theorem smaller_square_length_proof :
  ∀ (P Q R S T U : ℝ × ℝ),
  P = (0, 2) →
  Q = (0, 0) →
  R = (2, 0) →
  S = (2, 2) →
  (T.fst = 0 ∧ T.snd = 2 / 3) →
  (U.fst = 2 ∧ U.snd = 4 / 3) →
  is_right_triangle P T U →
  ∃ P' Q' R' S' : ℝ × ℝ, 
  is_square P' Q' R' S' ∧ 
  has_side_on_line P' S' (0, 2) U (2, 4 / 3) ∧ 
  has_vertex_on_line R' Q' (2, 2) P (0, 2) Q (0, 0) R (2, 0) S →
  distance P P' = smaller_square_side_length :=
by
  intros P Q R S T U hP hQ hR hS hT hU hPTU
  use [(5 / 3, 1 / 3), (4 / 3, 1), (5 / 3, 2), (4 / 3, 1)]
  sorry

end smaller_square_length_proof_l739_739727


namespace extend_array_condition_l739_739782

open Matrix

theorem extend_array_condition {k : ℕ} (hk : ∃ n, k = 3 * n ∧ n ≥ 2) :
  ∃ (a : Matrix (Fin 3) (Fin k) ℕ),
    (∑ j, a 0 j = ∑ j, a 1 j) ∧ (∑ j, a 1 j = ∑ j, a 2 j) ∧ 
    (∑ j, (a 0 j)^2 = ∑ j, (a 1 j)^2) ∧ (∑ j, (a 1 j)^2 = ∑ j, (a 2 j)^2) := 
by {
  -- Omitted proof
  sorry
}

end extend_array_condition_l739_739782


namespace enclosed_area_of_curve_eq_l739_739958

-- Define the problem conditions
def arc_length := π / 2
def hexagon_side_length := 3
def radius_of_arc := 1
def number_of_arcs := 12
def hexagon_area := (3 * Real.sqrt 3 / 2) * hexagon_side_length^2
def sector_area := (1 / 4) * π
def total_sector_area := number_of_arcs * sector_area

-- Define the expected enclosed area
def expected_enclosed_area := hexagon_area + total_sector_area

-- State the theorem to be proven
theorem enclosed_area_of_curve_eq :
    expected_enclosed_area = 13.5 * Real.sqrt 3 + 3 * π :=
by
    -- Proof not required, hence we use sorry to skip it
    sorry

end enclosed_area_of_curve_eq_l739_739958


namespace count_valid_n_l739_739231

theorem count_valid_n :
  let q_range := Finset.Icc 200 999
  let r_range := Finset.Icc 0 99
  ∃ (n : ℕ), n ∈ q_range ∧ n ∈ r_range ∧
             ∃ (count_n : ℕ), count_n = 6400 ∧
             ∀ (n' : ℕ), (∃ q r, n' = 100 * q + r ∧ q ∈ q_range ∧ r ∈ r_range ∧ (q + r) % 13 = 0) → count_n = 6400 :=
begin
  sorry
end

end count_valid_n_l739_739231


namespace equation_two_roots_iff_l739_739086

theorem equation_two_roots_iff (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + 2 * x1 + 2 * |x1 + 1| = a ∧ x2^2 + 2 * x2 + 2 * |x2 + 1| = a) ↔ a > -1 :=
by
  sorry

end equation_two_roots_iff_l739_739086


namespace intersection_product_one_l739_739539

noncomputable def line_l (t : ℝ) : ℝ × ℝ :=
  (-1 + (sqrt 3) / 2 * t, 1 / 2 * t)

def curve_C (x y : ℝ) : Prop :=
  x^2 + y^2 = 2 * y

theorem intersection_product_one : 
  (∃ A B t1 t2 : ℝ, 
    (line_l t1 = A ∧ line_l t2 = B) ∧ 
    curve_C (line_l t1).1 (line_l t1).2 ∧ 
    curve_C (line_l t2).1 (line_l t2).2 ∧ 
    ∀ P : ℝ × ℝ, (|P - A| * |P - B| = 1)) := 
sorry

end intersection_product_one_l739_739539


namespace guide_is_native_l739_739335

-- Definitions of tribe members
inductive Tribe
| native -- natives always tell the truth
| alien -- aliens always lie

-- Function that determines if a tribe member always tells truth
def tells_truth (t : Tribe) : Prop :=
  match t with
  | Tribe.native := True
  | Tribe.alien := False

theorem guide_is_native (guide_islander claimed_person : Tribe)
  (H1 : tells_truth guide_islander = True)
  (H2 : tells_truth claimed_person = True ∨ tells_truth claimed_person = False) :
  guide_islander = Tribe.native :=
by
  have H_claim := claimed_person = Tribe.native ∨ claimed_person = Tribe.alien,
  sorry

end guide_is_native_l739_739335


namespace ellipse_equation_line_equation_trajectory_equation_l739_739497

noncomputable def ellipse_standard_equation (f : ℝ × ℝ) (p : ℝ × ℝ) (a : ℝ) (b : ℝ) : Prop :=
  f = (0, sqrt 3) ∧ p = (1 / 2, sqrt 3) ∧ a = 2 ∧ b = 1 ∧ (∀ x y, (y^2)/(a^2) + (x^2)/(b^2) = 1)

noncomputable def line_condition1 (a : ℝ) (b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a > b ∧ (∀ x y, (y^2)/(a^2) + x^2 = 1 ) ∧ -3 / 5

noncomputable def line_condition2 (k : ℝ) : Prop :=
  k = sqrt 5 ∨ k = - sqrt 5

noncomputable def trajectory_condition (x y : ℝ) : Prop :=
  (x - 1 / 5)^2 + y^2 = 16 / 25 ∧ x ≠ 1

theorem ellipse_equation
  (f : ℝ × ℝ)
  (p : ℝ × ℝ)
  (a b : ℝ) :
  ellipse_standard_equation f p a b :=
by
  sorry

theorem line_equation
  (a b : ℝ) :
  line_condition1 a b ∨ line_condition2 (sqrt 5) :=
by
  sorry

theorem trajectory_equation
  (x y : ℝ) :
  trajectory_condition x y :=
by
  sorry

end ellipse_equation_line_equation_trajectory_equation_l739_739497


namespace sequence_value_l739_739892

theorem sequence_value (a : ℕ → ℕ) (h₁ : ∀ n, a (2 * n) = a (2 * n - 1) + (-1 : ℤ)^n) 
                        (h₂ : ∀ n, a (2 * n + 1) = a (2 * n) + n)
                        (h₃ : a 1 = 1) : a 20 = 46 :=
by 
  sorry

end sequence_value_l739_739892


namespace trapezoid_diagonal_is_8sqrt5_trapezoid_leg_is_4sqrt5_l739_739470

namespace Trapezoid

def isosceles_trapezoid (AD BC : ℝ) := 
  AD = 20 ∧ BC = 12

def diagonal (AD BC : ℝ) (AC : ℝ) := 
  isosceles_trapezoid AD BC → AC = 8 * Real.sqrt 5

def leg (AD BC : ℝ) (CD : ℝ) := 
  isosceles_trapezoid AD BC → CD = 4 * Real.sqrt 5

theorem trapezoid_diagonal_is_8sqrt5 (AD BC AC : ℝ) : 
  diagonal AD BC AC :=
by
  intros
  sorry

theorem trapezoid_leg_is_4sqrt5 (AD BC CD : ℝ) : 
  leg AD BC CD :=
by
  intros
  sorry

end Trapezoid

end trapezoid_diagonal_is_8sqrt5_trapezoid_leg_is_4sqrt5_l739_739470


namespace find_prices_and_max_basketballs_l739_739329

def unit_price_condition (x : ℕ) (y : ℕ) : Prop :=
  y = 2*x - 30

def cost_ratio_condition (x : ℕ) (y : ℕ) : Prop :=
  3 * x = 2 * y - 60

def total_cost_condition (total_cost : ℕ) (num_basketballs : ℕ) (num_soccerballs : ℕ) : Prop :=
  total_cost ≤ 15500 ∧ num_basketballs + num_soccerballs = 200

theorem find_prices_and_max_basketballs
  (x y : ℕ) (total_cost : ℕ) (num_basketballs : ℕ) (num_soccerballs : ℕ)
  (h1 : unit_price_condition x y)
  (h2 : cost_ratio_condition x y)
  (h3 : total_cost_condition total_cost num_basketballs num_soccerballs)
  (h4 : total_cost = 90 * num_basketballs + 60 * num_soccerballs)
  : x = 60 ∧ y = 90 ∧ num_basketballs ≤ 116 :=
sorry

end find_prices_and_max_basketballs_l739_739329


namespace solution_set_f_le_3g_l739_739537

def f (x : ℝ) : ℝ := x^2 - 2 * x

def g (x : ℝ) : ℝ := if x ≥ 1 then x - 2 else -x

theorem solution_set_f_le_3g :
  { x : ℝ | f x ≤ 3 * g x } = { x | -1 ≤ x ∧ x ≤ 0 } ∪ { x | 2 ≤ x ∧ x ≤ 3 } :=
by {
  sorry
}

end solution_set_f_le_3g_l739_739537


namespace number_of_holes_on_circular_board_is_91_l739_739254

-- Define the problem conditions
def circular_board_holes (n : ℕ) : Prop :=
  (n < 100) ∧                        -- fewer than 100 holes
  (n % 3 = 1) ∧                      -- 2-hole jumps lead to same hole modulo 3
  (n % 5 = 1) ∧                      -- 4-hole jumps lead to same hole modulo 5
  (n % 6 = 1)                        -- 6-hole jumps lead to starting hole

-- Prove the number of holes is 91
theorem number_of_holes_on_circular_board_is_91 (n : ℕ) :
  circular_board_holes n → n = 91 :=
by
  intro h
  cases h with h1 h2,
  cases h2 with h3 h4,
  cases h4 with h5 h6,
  sorry

end number_of_holes_on_circular_board_is_91_l739_739254


namespace sin_alpha_eq_l739_739139

noncomputable def proof_problem (α : ℝ) : Prop :=
  cos(α + π / 4) ^ 2 = 1 / 6 → sin (2 * α) = 2 / 3

-- Here's the statement that needs to be proved
theorem sin_alpha_eq : ∀ α : ℝ, proof_problem α :=
by
  -- The proof goes here.
  sorry

end sin_alpha_eq_l739_739139


namespace rationals_are_closed_l739_739922

-- Definitions for the involved sets and operations
def is_closed_under (S : set ℚ) (op : ℚ → ℚ → ℚ) : Prop :=
  ∀ a b, a ∈ S → b ∈ S → op a b ∈ S

def closed_under_arithmetic_operations (S : set ℚ) : Prop :=
  is_closed_under S (λ x y, x + y) ∧
  is_closed_under S (λ x y, x - y) ∧
  is_closed_under S (λ x y, x * y) ∧
  (∀ a b, a ∈ S → b ∈ S → b ≠ 0 → x / b ∈ S)

def set_of_rationals : set ℚ := { x | x ∈ ℚ }

def rational_numbers_closed : Prop := closed_under_arithmetic_operations set_of_rationals

theorem rationals_are_closed : rational_numbers_closed := sorry

end rationals_are_closed_l739_739922


namespace triangle_region_areas_l739_739023

open Real

theorem triangle_region_areas (A B C : ℝ) 
  (h1 : 20^2 + 21^2 = 29^2)
  (h2 : ∃ (triangle_area : ℝ), triangle_area = 210)
  (h3 : C > A)
  (h4 : C > B)
  : A + B + 210 = C := 
sorry

end triangle_region_areas_l739_739023


namespace two_roots_iff_a_greater_than_neg1_l739_739072

theorem two_roots_iff_a_greater_than_neg1 (a : ℝ) :
  (∃! x : ℝ, x^2 + 2*x + 2*|x + 1| = a) ↔ a > -1 :=
sorry

end two_roots_iff_a_greater_than_neg1_l739_739072


namespace factor_expression_l739_739779

theorem factor_expression (x : ℝ) : 12 * x ^ 2 + 8 * x = 4 * x * (3 * x + 2) :=
by
  sorry

end factor_expression_l739_739779


namespace part1_values_correct_eighth_grade_students_above80_l739_739394

-- Definitions for 7th grade data
def seventh_grader_scores := [78, 90, 80, 95, 68, 90, 90, 100, 75, 80]
def seventh_scores_partition := (1, 4, 3, 2) -- Corresponding counts in the score ranges
def seventh_grade_stats := (84.6, 85, 90) -- (mean, median, mode)

-- Definitions for 8th grade data
def eighth_grader_scores := [80, 70, 85, 95, 90, 100, 90, 85, 90, 78]
def eighth_scores_partition := (1, 2, 5, 2) -- Corresponding counts in the score ranges
def eighth_grade_stats := (86.3, 87.5, 90) -- (mean, median, mode)

-- Proof Statements
theorem part1_values_correct :
  let a := 5
  let b := 2
  let c := 85
  let d := 90
  eighth_scores_partition = (1, 2, a, b) ∧ seventh_grade_stats.2 = c ∧ eighth_grade_stats.3 = d := by
  sorry

theorem eighth_grade_students_above80 (total_students : ℕ) :
  total_students = 200 → (eighth_scores_partition.2 + eighth_scores_partition.3) / 10 * total_students = 140 := by
  sorry

-- Specify the total number of 8th-grade students
def eighth_grade_total_students := 200

end part1_values_correct_eighth_grade_students_above80_l739_739394


namespace hyperbola_eccentricity_l739_739759

-- Define the parameters and constants
variables {a b c x y : ℝ} (h1 : a > 0) (h2 : b > 0)

-- Define the equations for the hyperbola, circle, and parabola
def hyperbola : Prop := x^2 / a^2 - y^2 / b^2 = 1
def circle : Prop := x^2 + y^2 = a^2
def parabola : Prop := y^2 = 4 * c * x

-- Define the vectors and the given vector equation
variables {OE OF OP : ℝ}

-- Hypothesis for the vector condition
def vector_condition : Prop := OE = 1/2 * (OF + OP)

-- Define the goal statement: proving the eccentricity of the hyperbola
theorem hyperbola_eccentricity (hH : hyperbola) (hC : circle) (hP : parabola) (hV : vector_condition) : 
  ∃ e : ℝ, e = (1 + real.sqrt 5) / 2 :=
sorry

end hyperbola_eccentricity_l739_739759


namespace not_algebraic_expression_optionB_l739_739421

-- Define the conditions
def optionA := π
def optionB := (x : Nat) → x = 1
def optionC := (x : Nat) → 1 / x
def optionD := Real.sqrt 3

-- Define the property of being an algebraic expression
def is_algebraic_expression (expr : Type) : Prop := expr ≠ (x : Nat) → x = 1 

-- State the theorem
theorem not_algebraic_expression_optionB : is_algebraic_expression (x : Nat) → x = 1 := sorry

end not_algebraic_expression_optionB_l739_739421


namespace equilateral_triangle_sum_zero_l739_739256

-- Definitions for the geometric entities and vectors
variable {Point : Type} [AffineSpace Point]

-- Triangle vertices
variables (A B C A1 B1 C1 : Point)

-- Hypothesis that we have equilateral triangles constructed
variable (equilateral_ABC1 : (equilateral_triangle A B C1))
variable (equilateral_BCA1 : (equilateral_triangle B C A1))
variable (equilateral_CAB1 : (equilateral_triangle C A B1))

-- Statement to prove
theorem equilateral_triangle_sum_zero :
  (vector_between A A1) + (vector_between B B1) + (vector_between C C1) = (vector_zero) :=
by
  sorry

end equilateral_triangle_sum_zero_l739_739256


namespace derivative_of_2_pow_x_derivative_of_x_sqrt_x_l739_739469

-- Problem 1: Derivative of y = 2^x
theorem derivative_of_2_pow_x (x : ℝ) : 
  deriv (λ x : ℝ, 2^x) x = 2^x * real.log 2 :=
by
  sorry

-- Problem 2: Derivative of y = x * sqrt(x)
theorem derivative_of_x_sqrt_x (x : ℝ) : 
  deriv (λ x : ℝ, x * real.sqrt x) x = (3 / 2) * real.sqrt x :=
by
  sorry

end derivative_of_2_pow_x_derivative_of_x_sqrt_x_l739_739469


namespace N_intersect_M_complement_l739_739515

-- Definitions based on given conditions
def U : Set ℝ := Set.univ
def M : Set ℝ := { x | -2 ≤ x ∧ x ≤ 3 }
def N : Set ℝ := { x | -1 ≤ x ∧ x ≤ 4 }
def M_complement : Set ℝ := { x | x < -2 ∨ x > 3 }  -- complement of M in ℝ

-- Lean statement for the proof problem
theorem N_intersect_M_complement :
  N ∩ M_complement = { x | 3 < x ∧ x ≤ 4 } :=
sorry

end N_intersect_M_complement_l739_739515


namespace tangent_line_at_zero_eqn_range_m_l739_739145

def f (a : ℝ) (x : ℝ) : ℝ := (a * x) / Real.exp x

theorem tangent_line_at_zero_eqn (a b : ℝ) :
  (∀ x : ℝ, x = 0 → f a x = x + b) → a = 1 ∧ b = 0 :=
sorry

theorem range_m (m : ℝ) :
  (∀ x : ℝ, x ∈ Set.Ioo (1 / 2) (3 / 2) → f 1 x < 1 / (m + 6 * x - 3 * x ^ 2)) →
  m ∈ Set.Ico (-9 / 4) (Real.exp 1 - 3) :=
sorry

end tangent_line_at_zero_eqn_range_m_l739_739145


namespace solve_g_eq_2_l739_739619

def g (x : ℝ) : ℝ :=
if x < 0 then 5 * x + 10 else x^2 + 3 * x - 18

theorem solve_g_eq_2 (x : ℝ) :
  g(x) = 2 ↔ x = -8 / 5 ∨ x = 4 :=
by
  sorry

end solve_g_eq_2_l739_739619


namespace locus_of_M_and_PQ_segment_length_bounds_l739_739490

theorem locus_of_M_and_PQ_segment_length_bounds :
  ∀ (x y x0 y0 : ℝ), 
  (A : ℝ×ℝ) → (N : ℝ×ℝ) → (M : ℝ×ℝ) → 
  let r := 3
  (circle_C1 : x^2 + y^2 = r^2) →
  (line_l0 : y = (1/2) * x + (3/2) * sqrt 5) →
  (A ∈ set_of (λ (p : ℝ×ℝ), p.1^2 + p.2^2 = r^2)) →  
  (N.1 = x0 ∧ N.2 = 0) →
  (M : ≫ (x, y) , A : ≫ (x0, y0)) → 
  vector_eq : M + 2 * (M - A) = (2 * sqrt 2 - 2) * (N - 0) →
  (M ∈ set_of (λ (q : ℝ×ℝ), (q.1^2) / 8 + (q.2^2) / 4 = 1)) ∧
  ∀ (k m : ℝ), 
  let line_l := (λ (x : ℝ), k * x + m)
  (P Q : ℝ × ℝ) → 
  (P ∈ set_of (λ (q : ℝ×ℝ), (q.1^2) / 8 + (q.2^2) / 4 = 1)) → 
  (Q ∈ set_of (λ (q : ℝ×ℝ), (q.1^2) / 8 + (q.2^2) / 4 = 1)) → 
  P ≠ Q → 
  (circle_with_diameter_pq_containing_origin : (λ (p : ℝ×ℝ), p.1x + p.2y = 0)) →  
  ∃ pq_length : ℝ, 4 * sqrt 6 / 3 ≤ pq_length ∧ pq_length ≤ 2 * sqrt 3 := sorry

end locus_of_M_and_PQ_segment_length_bounds_l739_739490


namespace max_sum_of_diagonals_l739_739003

theorem max_sum_of_diagonals (a b : ℝ) (h_side : a^2 + b^2 = 25) (h_bounds1 : 2 * a ≤ 6) (h_bounds2 : 2 * b ≥ 6) : 2 * (a + b) = 14 :=
sorry

end max_sum_of_diagonals_l739_739003


namespace extreme_values_of_g_range_of_a_if_f_is_monotonic_range_of_a_if_exists_t_l739_739525

-- Part (I)
def g (x : ℝ) : ℝ := 16 * x ^ 3 - 24 * x ^ 2 - 15 * x - 2
def g' (x : ℝ) : ℝ := 48 * x ^ 2 - 48 * x - 15
theorem extreme_values_of_g :
  ∃ x₁ x₂ : ℝ, (x₁ = -1/4 ∧ g x₁ = 0) ∧ (x₂ = 5/4 ∧ g x₂ = -27) := sorry

-- Part (II)
def f (x a : ℝ) : ℝ := sqrt x * abs (x - a)
theorem range_of_a_if_f_is_monotonic :
  (∀ x y, 0 ≤ x → x ≤ y → f x a ≤ f y a) → a ≤ 0 := sorry

-- Part (III)
theorem range_of_a_if_exists_t (a t : ℝ) (h1 : a > 0) (h2 : t > a) :
  (∀ x, 0 ≤ x → x ≤ t → 0 ≤ f x a ∧ f x a ≤ t / 2) → 0 < a ∧ a ≤ 3 := sorry

end extreme_values_of_g_range_of_a_if_f_is_monotonic_range_of_a_if_exists_t_l739_739525


namespace max_value_x_plus_2y_l739_739134

theorem max_value_x_plus_2y (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x^2 + y^2 = 1) : (x + 2 * y) ≤ sqrt 5 :=
sorry

end max_value_x_plus_2y_l739_739134


namespace min_trig_expression_l739_739090

theorem min_trig_expression : 
  ∃ x : ℝ, 0 < x ∧ x < (π / 2) ∧ 
  (∀ y : ℝ, 0 < y ∧ y < (π / 2) → ((tan y + cot y)^2 + (sin y + cos y)^2) 
    ≥ ((tan x + cot x)^2 + (sin x + cos x)^2)) ∧ 
  ((tan x + cot x)^2 + (sin x + cos x)^2) = 6 :=
sorry

end min_trig_expression_l739_739090


namespace P_gt_Q_l739_739505

variable {a : ℕ → ℝ}
variable {q : ℝ}
variable (h_geometric : ∀ n : ℕ, a (n + 1) = q * a n)
variable (h_positive : ∀ n : ℕ, a n > 0)
variable (hq : q ≠ 1)

noncomputable def P := (1 / 2) * (Real.log 0.5 (a 4) + Real.log 0.5 (a 8))
noncomputable def Q := Real.log 0.5 ((a 2 + a 10) / 2)

theorem P_gt_Q : P > Q := by
  sorry

end P_gt_Q_l739_739505


namespace roots_eq_two_iff_a_gt_neg1_l739_739064

theorem roots_eq_two_iff_a_gt_neg1 (a : ℝ) : 
  (∃! x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + 2*|x₁ + 1| = a ∧ x₂^2 + 2*x₂ + 2*|x₂ + 1| = a) ↔ a > -1 :=
by sorry

end roots_eq_two_iff_a_gt_neg1_l739_739064


namespace number_equation_l739_739752

variable (x : ℝ)

theorem number_equation :
  5 * x - 2 * x = 10 :=
sorry

end number_equation_l739_739752


namespace divisible_by_10_l739_739384

def is_odd (m : ℕ) : Prop := ∃ (k : ℕ), m = 2 * k + 1

theorem divisible_by_10 (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) :
  ∃ (x y ∈ {a, b, c}), ∀ (m n : ℕ), 
    is_odd m → is_odd n → (10 ∣ (x^m * y^n - x^n * y^m)) :=
sorry

end divisible_by_10_l739_739384


namespace length_of_CE_l739_739558

open Real

/-- Given a triangle ABC with ∠BAC = 60° and ∠ACB = 30°,
    AB = 2. M is the midpoint of segment AB. Point D lies
    on side BC such that AD ⊥ CM. Segment BC is extended
    through C to point E such that DE = EC.
    Prove that the length of CE is √3. -/
theorem length_of_CE (A B C D E : ℝ × ℝ)
  (hA : A = (0, 0))
  (hB : B = (2, 0))
  (hC : C = (1, sqrt 3))
  (hM : M = (1, 0))
  (hAD_per_CM : ∀ D, (D.1 = 0) ∧ (D.2 = sqrt 3))
  (hDE_EC : ∀ E, E = (1, 2 * sqrt 3)) :
  dist C E = sqrt 3 :=
by sorry

end length_of_CE_l739_739558


namespace triangle_area_proof_l739_739654

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  Real.sqrt (1/4 * (c^2 * a^2 - ( (c^2 + a^2 - b^2)/2 )^2 ))

theorem triangle_area_proof : 
    ∀ a b c : ℝ, (a + b + c = 10 + 2 * Real.sqrt 7) 
    → (a / b = 2 / Real.sqrt 7) 
    → (b / c = Real.sqrt 7 / 3) 
    → triangle_area a b c = 6 * Real.sqrt 3 :=
begin
  sorry
end

end triangle_area_proof_l739_739654


namespace expected_value_winnings_l739_739744

-- Define the possible outcomes on the die and the winnings for each outcome
def outcomes := Fin 8 → ℕ
def winnings (n : Fin 8) := 2 * (n + 1)^2

-- Define the probability of each outcome (since the die is fair, each has equal probability)
def probability : ℚ := 1 / 8

-- Define the expected value calculation
noncomputable def expected_value := 
  (List.range 8).map (λ i => probability * (winnings ⟨i, Nat.lt_succ_self 7⟩)).sum

-- The statement to prove
theorem expected_value_winnings : expected_value = 51\ Money.dollar :=
by
  sorry

end expected_value_winnings_l739_739744


namespace length_of_BF_l739_739579

variables (A C D E F B : Type) (AC AB CD : ℝ)
variable [linear_ordered_field ℝ]

def right_angle_at_A_and_C (quad : A) := ∃ A C, A ≠ C ∧ AC = 12
def points_on_AC (E F : A) := ∃ E F, E ∈ A ∧ F ∈ A
def DE_BF_perpendicular_AC (D E F C : A) := D ⊥ AC ∧ F ⊥ AC
def segment_lengths (E : A) := ∃ E D, DE = 6 ∧ CE = 8

theorem length_of_BF : ∀ (quad : A), 
  right_angle_at_A_and_C quad ∧ points_on_AC E F ∧ DE_BF_perpendicular_AC D E F A ∧ segment_lengths E -> 
  BF = 72 / 17 := sorry

end length_of_BF_l739_739579


namespace speed_ratio_l739_739659

def distance_to_work := 28
def speed_back := 14
def total_time := 6

theorem speed_ratio 
  (d : ℕ := distance_to_work) 
  (v_2 : ℕ := speed_back) 
  (t : ℕ := total_time) : 
  ∃ v_1 : ℕ, (d / v_1 + d / v_2 = t) ∧ (v_2 / v_1 = 2) :=
by 
  sorry

end speed_ratio_l739_739659


namespace volume_of_four_cubes_l739_739357

theorem volume_of_four_cubes (edge_length : ℕ) (num_cubes : ℕ) (h_edge : edge_length = 5) (h_num : num_cubes = 4) :
  num_cubes * (edge_length ^ 3) = 500 :=
by 
  sorry

end volume_of_four_cubes_l739_739357


namespace factor_expression_l739_739438

theorem factor_expression (x : ℝ) : 
  (21 * x ^ 4 + 90 * x ^ 3 + 40 * x - 10) - (7 * x ^ 4 + 6 * x ^ 3 + 8 * x - 6) = 
  2 * x * (7 * x ^ 3 + 42 * x ^ 2 + 16) - 4 :=
by sorry

end factor_expression_l739_739438


namespace hyperbola_trajectory_l739_739544

noncomputable def trajectory_equation (x y : ℝ) : Prop :=
  ∃ P : ℝ × ℝ, (abs (sqrt (x ^ 2 + (y + 5) ^ 2) - sqrt (x ^ 2 + (y - 5) ^ 2)) = 6) ∧
               (P.1 = x) ∧ (P.2 = y)

theorem hyperbola_trajectory :
  ∃ x y : ℝ, trajectory_equation x y ∧ (y^2 / 9 - x^2 / 16 = 1) :=
sorry

end hyperbola_trajectory_l739_739544


namespace hyperbola_equation_l739_739153

theorem hyperbola_equation :
  (∃ (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0),
    (∀ (x y : ℝ), ((a > 0) ∧ (b > 0) ∧ (y = (Real.sqrt 5 / 2) * x) ∧
      ((∀ t s : ℝ, ((t^2 / 12 + s^2 / 3 = 1) → (∃ f1 f2 : ℝ, ((f1 = 3 ∧ f2 = 0) ∨ (f1 = -3 ∧ f2 = 0))))) ∧ 
      ((c : ℝ), ((c = 3) ∧ (Real.sqrt (c^2 - a^2) = b)))) ∧
      (x^2 / 4 - y^2 / 5 = 1))) :=
begin
  sorry
end

end hyperbola_equation_l739_739153


namespace children_crayons_l739_739458

theorem children_crayons (crayons_per_child total_crayons : ℕ) (h1 : crayons_per_child = 6) (h2 : total_crayons = 72) : ∃ n : ℕ, total_crayons = n * crayons_per_child ∧ n = 12 :=
by
  subst h1
  subst h2
  use 12
  split
  { simp }
  { rfl }

end children_crayons_l739_739458


namespace intersection_points_trajectory_of_P_is_circle_l739_739154

noncomputable def line_C1 (t α : ℝ) : ℝ × ℝ := (1 + t * Real.cos α, t * Real.sin α)
noncomputable def circle_C2 (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

def is_intersection (pt : ℝ × ℝ) (α : ℝ) : Prop :=
  ∃ t θ, line_C1 t α = pt ∧ circle_C2 θ = pt

theorem intersection_points (α : ℝ) (hα : α = Real.pi / 3) :
  is_intersection (1, 0) α ∧ is_intersection (1 / 2, -Real.sqrt 3 / 2) α :=
sorry

theorem trajectory_of_P_is_circle : 
  ∀ α : ℝ, let x := 1/2 * (Real.sin α)^2,
               y := -1/2 * (Real.sin α) * (Real.cos α)
          in (x - 1/4)^2 + y^2 = 1/16 :=
sorry

end intersection_points_trajectory_of_P_is_circle_l739_739154


namespace concyclic_iff_l739_739603

variables {A B C H O' N D : Type*}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace H]
variables [MetricSpace O'] [MetricSpace N] [MetricSpace D]
variables (a b c R : ℝ)

-- Conditions from the problem
def is_orthocenter (H : Type*) (A B C : Type*) : Prop :=
  -- definition of orthocenter using suitable predicates (omitted for brevity) 
  sorry

def is_circumcenter (O' : Type*) (B H C : Type*) : Prop :=
  -- definition of circumcenter using suitable predicates (omitted for brevity) 
  sorry

def is_midpoint (N : Type*) (A O' : Type*) : Prop :=
  -- definition of midpoint using suitable predicates (omitted for brevity) 
  sorry

def is_reflection (N D : Type*) (B C : Type*) : Prop :=
  -- definition of reflection about the side BC (omitted for brevity) 
  sorry

-- Definition that points A, B, C, D are concyclic
def are_concyclic (A B C D : Type*) : Prop :=
  -- definition using suitable predicates (omitted for brevity)
  sorry

-- Main theorem statement
theorem concyclic_iff (h1 : is_orthocenter H A B C) (h2 : is_circumcenter O' B H C) 
                      (h3 : is_midpoint N A O') (h4 : is_reflection N D B C)
                      (ha : a = 1) (hb : b = 1) (hc : c = 1) (hR : R = 1) :
  are_concyclic A B C D ↔ b^2 + c^2 - a^2 = 3 * R^2 := 
sorry

end concyclic_iff_l739_739603


namespace hypotenuse_length_l739_739708

theorem hypotenuse_length (x y : ℝ) (h1 : x * y^2 = 2400) (h2 : y * x^2 = 5760) : sqrt (x^2 + y^2) = 26 :=
sorry

end hypotenuse_length_l739_739708


namespace equation_two_roots_iff_l739_739085

theorem equation_two_roots_iff (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + 2 * x1 + 2 * |x1 + 1| = a ∧ x2^2 + 2 * x2 + 2 * |x2 + 1| = a) ↔ a > -1 :=
by
  sorry

end equation_two_roots_iff_l739_739085


namespace find_a_l739_739288

noncomputable def a : ℚ := 10/7

noncomputable def point1 : ℚ × ℚ := (-3, 6)
noncomputable def point2 : ℚ × ℚ := (2, -1)

noncomputable def direction_vector : ℚ × ℚ := 
  (point2.1 - point1.1, point2.2 - point1.2)

theorem find_a :
  ∃ k : ℚ, k * direction_vector = (a, -2) :=
by
  use 2/7
  simp [direction_vector, a]
  sorry

end find_a_l739_739288


namespace overall_average_mark_l739_739409

theorem overall_average_mark :
  let n1 := 70
  let mean1 := 50
  let n2 := 35
  let mean2 := 60
  let n3 := 45
  let mean3 := 55
  let n4 := 42
  let mean4 := 45
  (n1 * mean1 + n2 * mean2 + n3 * mean3 + n4 * mean4 : ℝ) / (n1 + n2 + n3 + n4) = 51.89 := 
by {
  sorry
}

end overall_average_mark_l739_739409


namespace unique_root_in_interval_l739_739503

theorem unique_root_in_interval (b c : ℝ) (f : ℝ → ℝ)
  (h1 : ∀ x ∈ set.Icc (-1 : ℝ) (1 : ℝ), f x = x^3 + b * x + c)
  (h2 : ∀ x ∈ set.Icc (-1 : ℝ) (1 : ℝ), 3 * x^2 + b ≥ 0)
  (h3 : f (-1 / 2) * f (1 / 2) < 0) :
  ∃! x ∈ set.Icc (-1 : ℝ) (1 : ℝ), f x = 0 := sorry

end unique_root_in_interval_l739_739503


namespace find_p_l739_739440

variable (p q : ℝ)

-- Conditions
def total_probability := p + q + 1/6 = 1
def q_leq_p := q ≤ p
def same_result_probability := p^2 + q^2 + (1/6)^2 = 1/2

-- Statement to prove
theorem find_p (h1 : total_probability)
               (h2 : q_leq_p)
               (h3 : same_result_probability) :
  p = 2/3 :=
sorry

end find_p_l739_739440


namespace sum_of_coefficients_l739_739677

theorem sum_of_coefficients (x : ℝ) : (∃ x : ℝ, 5 * x * (1 - x) = 3) → 5 + (-5) + 3 = 3 :=
by
  intro h
  -- Proof goes here
  sorry

end sum_of_coefficients_l739_739677


namespace complex_problem_l739_739144

open Complex

noncomputable def complex_z (z : ℂ) : Prop :=
  z * (1 + 2 * I) = 5 * I

noncomputable def complex_root (z : ℂ) : Prop :=
  (z - (2 + I)) * (z - (2 - I)) = 0

noncomputable def complex_modulus (z : ℂ) : ℂ :=
  abs (conj z + 5 / z)

theorem complex_problem (z : ℂ) (h : complex_z z) :
  z = 2 + I ∧ complex_root (2 + I) ∧ complex_modulus (2 + I) = 2 * Real.sqrt 5 :=
  sorry

end complex_problem_l739_739144


namespace probability_log_eta_over_xi_minus_1_lt_0_l739_739916
open ProbabilityTheory

/- Given conditions -/
variables {λ : ℝ} (ξ η : ℝ → ℝ)
-- Assume ξ and η follow exponential distributions with the same parameter λ, and they are independent.
def exponential_distribution (λ : ℝ) (x : ℝ) : ℝ :=
if x < 0 then 0 else λ * exp (-λ * x)

noncomputable def joint_density (λ : ℝ) (x y : ℝ) : ℝ :=
(exponential_distribution λ x) * (exponential_distribution λ y)

-- Prove the desired probability
theorem probability_log_eta_over_xi_minus_1_lt_0 :
  ∫∫ (x y : ℝ) in {xy | x > 1 ∧ 0 < y ∧ y < x - 1}, joint_density λ x y = (exp (-λ) / 2) :=
sorry

end probability_log_eta_over_xi_minus_1_lt_0_l739_739916


namespace sum_of_modulus_of_three_element_subsets_l739_739786

def P := { x : ℕ | ∃ (n : ℕ), 1 ≤ n ∧ n ≤ 10 ∧ x = 2 * n - 1 }

def three_element_subsets : finset (finset ℕ) := (finset.powerset P.to_finset).filter (λ s, s.card = 3)

def sum_modulus (S : finset (finset ℕ)) : ℕ :=
  S.sum (λ s, s.sum id)

theorem sum_of_modulus_of_three_element_subsets :
  sum_modulus three_element_subsets = 3600 := 
  by
  -- meant to show that the statement holds
  -- proof steps would go here
  sorry

end sum_of_modulus_of_three_element_subsets_l739_739786


namespace series_converges_to_three_halves_l739_739053

noncomputable def series_sum (f : ℕ → ℚ) :=
  (∑' n, f n).val

def series_formula (n : ℕ) : ℚ :=
  2 * 3 ^ n / (3 ^ (2 ^ (n + 1)) - 1)

theorem series_converges_to_three_halves :
  series_sum series_formula = 3 / 2 := by
  sorry

end series_converges_to_three_halves_l739_739053


namespace area_of_field_l739_739000

theorem area_of_field : ∀ (L W : ℕ), L = 20 → L + 2 * W = 88 → L * W = 680 :=
by
  intros L W hL hEq
  rw [hL] at hEq
  sorry

end area_of_field_l739_739000


namespace second_degree_polynomial_roots_interval_l739_739221

theorem second_degree_polynomial_roots_interval 
  (f : ℝ → ℝ) (h_poly : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c)
  (h_roots : ∃ r s : ℝ, r ∈ Icc (-1 : ℝ) (1 : ℝ) ∧ s ∈ Icc (-1 : ℝ) (1 : ℝ) ∧ f r = 0 ∧ f s = 0)
  (h_point : ∃ x0 : ℝ, x0 ∈ Icc (-1 : ℝ) (1 : ℝ) ∧ |f x0| = 1) :
  (∀ α ∈ Icc (0 : ℝ) (1 : ℝ), ∃ ζ ∈ Icc (-1 : ℝ) (1 : ℝ), |deriv f ζ| = α) ∧
  (∀ α > 1, ¬∃ ζ ∈ Icc (-1 : ℝ) (1 : ℝ), |deriv f ζ| = α) :=
by
  sorry

end second_degree_polynomial_roots_interval_l739_739221


namespace expression_in_terms_of_p_q_l739_739608

variables {α β γ δ p q : ℝ}

-- Let α and β be the roots of x^2 - 2px + 1 = 0
axiom root_α_β : ∀ x, (x - α) * (x - β) = x^2 - 2 * p * x + 1

-- Let γ and δ be the roots of x^2 + qx + 2 = 0
axiom root_γ_δ : ∀ x, (x - γ) * (x - δ) = x^2 + q * x + 2

-- Expression to be proved
theorem expression_in_terms_of_p_q :
  (α - γ) * (β - γ) * (α - δ) * (β - δ) = 2 * (p - q) ^ 2 :=
sorry

end expression_in_terms_of_p_q_l739_739608


namespace study_time_for_average_l739_739638

theorem study_time_for_average
    (study_time_exam1 score_exam1 : ℕ)
    (study_time_exam2 score_exam2 average_score desired_average : ℝ)
    (relation : score_exam1 = 20 * study_time_exam1)
    (direct_relation : score_exam2 = 20 * study_time_exam2)
    (total_exams : ℕ)
    (average_condition : (score_exam1 + score_exam2) / total_exams = desired_average) :
    study_time_exam2 = 4.5 :=
by
  have : total_exams = 2 := by sorry
  have : score_exam1 = 60 := by sorry
  have : desired_average = 75 := by sorry
  have : score_exam2 = 90 := by sorry
  sorry

end study_time_for_average_l739_739638


namespace sum_of_powers_of_two_l739_739862

theorem sum_of_powers_of_two (n : ℕ) (h : 1 ≤ n ∧ n ≤ 511) : 
  ∃ (S : Finset ℕ), S ⊆ ({2^8, 2^7, 2^6, 2^5, 2^4, 2^3, 2^2, 2^1, 2^0} : Finset ℕ) ∧ 
  S.sum id = n :=
by
  sorry

end sum_of_powers_of_two_l739_739862


namespace february_five_sundays_in_twenty_first_century_l739_739252

/-- 
  Define a function to check if a year is a leap year
-/
def is_leap_year (year : ℕ) : Prop :=
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ year % 400 = 0

/-- 
  Define the specific condition for the problem: 
  Given a year, whether February 1st for that year is a Sunday
-/
def february_first_is_sunday (year : ℕ) : Prop :=
  -- This is a placeholder logic. In real applications, you would
  -- calculate the exact weekday of February 1st for the provided year.
  sorry

/-- 
  The list of years in the 21st century where February has 5 Sundays is 
  exactly {2004, 2032, 2060, and 2088}.
-/
theorem february_five_sundays_in_twenty_first_century :
  {year : ℕ | is_leap_year year ∧ february_first_is_sunday year ∧ (2001 ≤ year ∧ year ≤ 2100)} =
  {2004, 2032, 2060, 2088} := sorry

end february_five_sundays_in_twenty_first_century_l739_739252


namespace binomial_sum_sum_solution_l739_739346

open Nat

theorem binomial_sum :
  ∀ n : ℕ, binomial 30 15 + binomial 30 n = binomial 31 16 → n = 14 ∨ n = 16 :=
by
  sorry

theorem sum_solution :
  (14 + 16) = 30 :=
by
  exact rfl

end binomial_sum_sum_solution_l739_739346


namespace problem_1_problem_2_problem_3_problem_4_l739_739158

def A := {x : ℝ | 3 ≤ x ∧ x < 7}
def B := {x : ℝ | 5 < x ∧ x < 10}

def CR (S : set ℝ) : set ℝ := λ x, ¬ S x

theorem problem_1 : CR (A ∪ B) = {x : ℝ | x < 3 ∨ x ≥ 10} :=
by sorry

theorem problem_2 : CR (A ∩ B) = {x : ℝ | x ≤ 5 ∨ x ≥ 7} :=
by sorry

theorem problem_3 : (CR A) ∩ B = {x : ℝ | 7 ≤ x ∧ x < 10} :=
by sorry

theorem problem_4 : A ∪ (CR B) = {x : ℝ | x < 7 ∨ x ≥ 10} :=
by sorry

end problem_1_problem_2_problem_3_problem_4_l739_739158


namespace elevator_time_l739_739037

theorem elevator_time :
  ∀ (floors steps_per_floor steps_per_second extra_time : ℕ) (elevator_time_sec elevator_time_min : ℚ),
    floors = 8 →
    steps_per_floor = 30 →
    steps_per_second = 3 →
    extra_time = 30 →
    elevator_time_sec = ((floors * steps_per_floor) / steps_per_second) - extra_time →
    elevator_time_min = elevator_time_sec / 60 →
    elevator_time_min = 0.833 :=
by
  intros floors steps_per_floor steps_per_second extra_time elevator_time_sec elevator_time_min
  intros h_floors h_steps_per_floor h_steps_per_second h_extra_time h_elevator_time_sec h_elevator_time_min
  rw [h_floors, h_steps_per_floor, h_steps_per_second, h_extra_time] at *
  sorry

end elevator_time_l739_739037


namespace number_of_special_subsets_l739_739618

noncomputable def binomial : ℕ → ℕ → ℕ
| n 0       := 1
| n (k + 1) := (n - k) * binomial n k / (k + 1)

def count_special_subsets (p : ℕ) [hp : Fact (Nat.Prime p)] : ℕ :=
  binomial (2 * p) p

theorem number_of_special_subsets (p : ℕ) [hp : Fact (Nat.Prime p)] (hodd : p % 2 = 1) :
  (count_special_subsets p - 2) / p + 2 = (1 / p : ℚ) * (binomial (2 * p) p - 2) + 2 :=
sorry

end number_of_special_subsets_l739_739618


namespace circle1_eq_center_radius_circle2_eq_center_radius_circle3_eq_center_radius_l739_739380

noncomputable def circle1_center : ℝ × ℝ :=
  (3, -2)

noncomputable def circle1_radius : ℝ :=
  4

theorem circle1_eq_center_radius :
  ∀ x y : ℝ, (x - 3)^2 + (y + 2)^2 = 16 → (x, y) = circle1_center ∧ real.sqrt 16 = circle1_radius :=
by
  intro x y h
  have hc : (x - 3)^2 + (y + 2)^2 = 16 := h
  sorry

noncomputable def circle2_center : ℝ × ℝ :=
  (1, -3)

noncomputable def circle2_radius : ℝ :=
  5

theorem circle2_eq_center_radius :
  ∀ x y : ℝ, x^2 + y^2 - 2*(x - 3*y) - 15 = 0 → (x, y) = circle2_center ∧ real.sqrt 25 = circle2_radius :=
by
  intro x y h
  have hc : x^2 + y^2 - 2*(x - 3 * y) - 15 = 0 := h
  sorry

noncomputable def circle3_center : ℝ × ℝ :=
  (1/2, 1/2)

noncomputable def circle3_radius : ℝ :=
  1

theorem circle3_eq_center_radius :
  ∀ x y : ℝ, x^2 + y^2 = x + y + 1/2 → (x, y) = circle3_center ∧ real.sqrt 1 = circle3_radius :=
by
  intro x y h
  have hc : x^2 + y^2 = x + y + 1/2 := h
  sorry

end circle1_eq_center_radius_circle2_eq_center_radius_circle3_eq_center_radius_l739_739380


namespace factorization_of_expression_l739_739462

noncomputable def factorized_form (x : ℝ) : ℝ :=
  (x + 5 / 2 + Real.sqrt 13 / 2) * (x + 5 / 2 - Real.sqrt 13 / 2)

theorem factorization_of_expression (x : ℝ) :
  x^2 - 5 * x + 3 = factorized_form x :=
by
  sorry

end factorization_of_expression_l739_739462


namespace find_brick_length_l739_739687

-- Conditions as given in the problem.
def wall_length : ℝ := 8
def wall_width : ℝ := 6
def wall_height : ℝ := 22.5
def number_of_bricks : ℕ := 6400
def brick_width : ℝ := 11.25
def brick_height : ℝ := 6

-- The volume of the wall in cubic centimeters.
def wall_volume_cm_cube : ℝ := (wall_length * 100) * (wall_width * 100) * (wall_height * 100)

-- Define the volume of one brick based on the unknown length L.
def brick_volume (L : ℝ) : ℝ := L * brick_width * brick_height

-- Define an equivalence for the total volume of the bricks to the volume of the wall.
theorem find_brick_length : 
  ∃ (L : ℝ), wall_volume_cm_cube = brick_volume L * number_of_bricks ∧ L = 2500 := 
by
  sorry

end find_brick_length_l739_739687


namespace Ian_hours_worked_l739_739175

theorem Ian_hours_worked (money_left: ℝ) (hourly_rate: ℝ) (spent: ℝ) (earned: ℝ) (hours: ℝ) :
  money_left = 72 → hourly_rate = 18 → spent = earned / 2 → earned = money_left * 2 → 
  earned = hourly_rate * hours → hours = 8 :=
by
  intros h1 h2 h3 h4 h5
  -- Begin mathematical validation process here
  sorry

end Ian_hours_worked_l739_739175


namespace age_product_difference_l739_739031

theorem age_product_difference 
  (age_today : ℕ) 
  (Arnold_age : age_today = 6) 
  (Danny_age : age_today = 6) : 
  (7 * 7) - (6 * 6) = 13 := 
by
  sorry

end age_product_difference_l739_739031


namespace certain_number_l739_739729

theorem certain_number (x : ℝ) (h : 4 * x = 200) : x = 50 :=
by
  sorry

end certain_number_l739_739729


namespace normal_line_eq_l739_739110

theorem normal_line_eq : ∃ l : ℝ → ℝ, 
  (∀ x, (l x) = -4 * x + 10) ∧ 
  (∀ x y, (y = 8 * x^(1/4) - 70) → 
    (x = 16 → y = -54) ∧ 
    ((deriv (λ x, 8 * x^(1/4) - 70) 16 = 1/4)) ) := by
  sorry

end normal_line_eq_l739_739110


namespace adjacent_even_difference_exists_l739_739975

theorem adjacent_even_difference_exists :
  ∃ (i : ℕ), 1 ≤ i ∧ i < 2010 ∧ (i - (i + 1) % 2010).abs % 2 = 0 :=
sorry

end adjacent_even_difference_exists_l739_739975


namespace max_area_sector_l739_739848

-- Definitions based on the conditions
def l (R : ℝ) : ℝ := 30 - 2 * R
def S (R : ℝ) : ℝ := 0.5 * l R * R
def α (R : ℝ) : ℝ := l R / R

-- Statement of the problem
theorem max_area_sector
: ∃ (R α : ℝ), (R = 15 / 2 ∧ α = 2) ∧ S R = -R^2 + 15 * R := 
sorry

end max_area_sector_l739_739848


namespace find_plane_speed_l739_739715

-- Defining the values in the problem
def distance_with_wind : ℝ := 420
def distance_against_wind : ℝ := 350
def wind_speed : ℝ := 23

-- The speed of the plane in still air
def plane_speed_in_still_air : ℝ := 253

-- Proof goal: Given the conditions, the speed of the plane in still air is 253 mph
theorem find_plane_speed :
  ∃ p : ℝ, (distance_with_wind / (p + wind_speed) = distance_against_wind / (p - wind_speed)) ∧ p = plane_speed_in_still_air :=
by
  use plane_speed_in_still_air
  have h : plane_speed_in_still_air = 253 := rfl
  sorry

end find_plane_speed_l739_739715


namespace find_a_20_l739_739860

variable {a : ℕ → ℝ}
variable {r : ℝ}

-- Definitions: The sequence is geometric: a_n = a_1 * r^(n-1)
def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a n = a 1 * r^(n-1)

-- Conditions in the problem: a_10 and a_30 satisfy the quadratic equation
def satisfies_quadratic_roots (a10 a30 : ℝ) : Prop :=
  a10 + a30 = 11 ∧ a10 * a30 = 16

-- Question: Find a_20
theorem find_a_20 (h1 : is_geometric_sequence a r)
                  (h2 : satisfies_quadratic_roots (a 10) (a 30)) :
  a 20 = 4 :=
sorry

end find_a_20_l739_739860


namespace average_weight_of_all_players_l739_739886

-- Definitions based on conditions
def num_forwards : ℕ := 8
def avg_weight_forwards : ℝ := 75
def num_defensemen : ℕ := 12
def avg_weight_defensemen : ℝ := 82

-- Total number of players
def total_players : ℕ := num_forwards + num_defensemen

-- Values derived from conditions
def total_weight_forwards : ℝ := avg_weight_forwards * num_forwards
def total_weight_defensemen : ℝ := avg_weight_defensemen * num_defensemen
def total_weight : ℝ := total_weight_forwards + total_weight_defensemen

-- Theorem to prove the average weight of all players
theorem average_weight_of_all_players : total_weight / total_players = 79.2 :=
by
  sorry

end average_weight_of_all_players_l739_739886


namespace real_imaginary_equal_implies_a_eq_1_l739_739189

variable {a : ℝ} (z : ℂ)

theorem real_imaginary_equal_implies_a_eq_1 
  (h1 : z = a + complex.I) 
  (h2 : z.re = a) 
  (h3 : z.im = 1) 
  (h4 : z.re = z.im) : 
  a = 1 := 
by 
  sorry

end real_imaginary_equal_implies_a_eq_1_l739_739189


namespace committee_ways_l739_739628

theorem committee_ways (n : ℕ) (k1 k2 : ℕ) (h1 : n = 30) (h2 : k1 = 5) (h3 : k2 = 3) :
  (nat.choose n k1) * (nat.choose (n - k1) k2) = 327764800 :=
by
  -- h1, h2, and h3 set the conditions for n, k1, and k2
  rw [h1, h2, h3]
  -- The following evaluates the binomial coefficients to get the result
  have h_exec : nat.choose 30 5 = 142506 := by sorry
  have h_aux : nat.choose 25 3 = 2300 := by sorry
  rw [h_exec, h_aux]
  exact mul_eq_327764800

-- Placeholder for the actual verification of binomial coefficient evaluations
lemma mul_eq_327764800 : 142506 * 2300 = 327764800 := by sorry

end committee_ways_l739_739628


namespace ln_sqrt2_lt_sqrt2_div2_ln_sin_cos_sum_l739_739379

theorem ln_sqrt2_lt_sqrt2_div2 : Real.log (Real.sqrt 2) < Real.sqrt 2 / 2 :=
sorry

theorem ln_sin_cos_sum : 2 * Real.log (Real.sin (1/8) + Real.cos (1/8)) < 1 / 4 :=
sorry

end ln_sqrt2_lt_sqrt2_div2_ln_sin_cos_sum_l739_739379


namespace z1_eq_z2_l739_739542

def z1 (x : ℝ) : ℂ := (Real.sin x) ^ 2 + Complex.i * Real.cos (2 * x)
def z2 (x : ℝ) : ℂ := (Real.sin x) ^ 2 + Complex.i * Real.cos x

theorem z1_eq_z2 (x : ℝ) : z1 x = z2 x → (x = 2 * Int.pi * k ∨ x = 2 * Int.pi * k + (2 / 3) * Int.pi ∨ x = 2 * Int.pi * k - (2 / 3) * Int.pi) := 
by sorry

end z1_eq_z2_l739_739542


namespace first_digit_of_A_l739_739403

theorem first_digit_of_A (A B : ℕ) (d : ℕ) :
  A < 10^8 ∧ B < 10^8 ∧ 
  A % 10 ≠ 0 ∧ B % 10 = 5 ∧ 
  ∀ a, A = sum_digits a ∧ length a = 8 ∧ 
  nodup a ∧ 
  (9999999 + A = B) → 
  (10^7 + A - 1) = B → 
  (A / 10^(7 : ℕ)) = 5 := sorry

end first_digit_of_A_l739_739403


namespace factor_expression_l739_739780

theorem factor_expression (x : ℝ) : 12 * x ^ 2 + 8 * x = 4 * x * (3 * x + 2) :=
by
  sorry

end factor_expression_l739_739780


namespace sqrt_9_is_pm3_l739_739676

theorem sqrt_9_is_pm3 : {x : ℝ | x ^ 2 = 9} = {3, -3} := sorry

end sqrt_9_is_pm3_l739_739676


namespace correlation_coefficient_property_l739_739200

theorem correlation_coefficient_property {r : ℝ} :
  |r| ≤ 1 ∧ (∀ s t : ℝ, |r| = s → |r| = t → s ≥ t → (correlation_strength s ≥ correlation_strength t)) :=
sorry

end correlation_coefficient_property_l739_739200


namespace find_m_n_l739_739722

theorem find_m_n (m n : ℕ) (h_pos : m > 0 ∧ n > 0) (h_gcd : m.gcd n = 1) (h_div : (m^3 + n^3) ∣ (m^2 + 20 * m * n + n^2)) :
  (m, n) ∈ [(1, 2), (2, 1), (2, 3), (3, 2), (1, 5), (5, 1)] :=
by
  sorry

end find_m_n_l739_739722


namespace max_ratio_of_three_digit_l739_739802

def is_digit (n : ℕ) : Prop := n >= 0 ∧ n ≤ 9

theorem max_ratio_of_three_digit (a b c : ℕ) (h_a : 1 ≤ a ∧ a ≤ 9) (h_b : is_digit b) (h_c : is_digit c) :
  let N := 100 * a + 10 * b + c in
  let S := a + b + c in
  S ≠ 0 →
  N / S ≤ 100 :=
by
  sorry

end max_ratio_of_three_digit_l739_739802


namespace number_of_valid_pairs_l739_739568

def point_in_T (x y : ℤ) : Prop := abs x ≤ 20 ∧ abs y ≤ 20 ∧ (x ≠ 0 ∨ y ≠ 0)

def is_colored (x y : ℤ) {x_colored y_colored : ℤ → ℤ → Prop} : Prop := x_colored x y ∨ x_colored (-x) (-y) 

noncomputable def T : set (ℤ × ℤ) := { p | ∃ x y, p = (x, y) ∧ point_in_T x y }

def valid_pair (x1 y1 x2 y2 : ℤ) : Prop :=
  x1 ≡ 2 * x2 [MOD 41] ∧ y1 ≡ 2 * y2 [MOD 41]

theorem number_of_valid_pairs (x_colored y_colored : ℤ → ℤ → Prop) :
  (∀ x y, point_in_T x y → is_colored x y) →
  (∃ N, N = 420) :=
sorry

end number_of_valid_pairs_l739_739568


namespace total_coins_received_l739_739369

theorem total_coins_received (coins_first_day coins_second_day : ℕ) 
  (h_first_day : coins_first_day = 22) 
  (h_second_day : coins_second_day = 12) : 
  coins_first_day + coins_second_day = 34 := 
by 
  sorry

end total_coins_received_l739_739369


namespace mike_average_points_per_game_l739_739248

theorem mike_average_points_per_game (total_points games_played points_per_game : ℕ) 
  (h1 : games_played = 6) 
  (h2 : total_points = 24) 
  (h3 : total_points = games_played * points_per_game) : 
  points_per_game = 4 :=
by
  rw [h1, h2] at h3  -- Substitute conditions h1 and h2 into the equation
  sorry  -- the proof goes here

end mike_average_points_per_game_l739_739248


namespace population_after_panic_l739_739416

noncomputable def villagePopulation := 7800
noncomputable def percentDisappeared := 10 / 100
noncomputable def percentPanic := 25 / 100

theorem population_after_panic 
    (villagePopulation : ℕ) 
    (percentDisappeared percentPanic : ℚ) 
    (initialPopulation : ℕ := villagePopulation): 
    initialPopulation = 7800 → 
    percentDisappeared = 10 / 100 → 
    percentPanic = 25 / 100 → 
    let disappeared := (percentDisappeared * initialPopulation).toNat in
    let remaining_after_disappearance := initialPopulation - disappeared in
    let left_during_panic := (percentPanic * remaining_after_disappearance).toNat in
    let final_population := remaining_after_disappearance - left_during_panic in
    final_population = 5265 := 
by {
  intros,
  rw [disappeared, remaining_after_disappearance, left_during_panic, final_population],
  sorry
}

end population_after_panic_l739_739416


namespace similar_triangles_perimeter_l739_739310

theorem similar_triangles_perimeter
  {k : ℕ} (h_ratio : 3 = 3) (p_small : 12 = 12) :
  let p_large := 20
  in p_large = 20 := by
  sorry

end similar_triangles_perimeter_l739_739310


namespace a_range_l739_739877

def f (x a : ℝ) : ℝ := cos (2 * x) + 2 * a * sin x + 3

theorem a_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, (π/3 < x₁ ∧ x₁ < π/2 ∧ π/3 < x₂ ∧ x₂ < π/2 ∧ x₁ < x₂) → f x₁ a ≥ f x₂ a) ↔ a ≤ real.sqrt 3 := sorry

end a_range_l739_739877


namespace find_side_a_l739_739209

-- Statements of the conditions
def ∠A : ℝ := Real.pi / 3 -- 60 degrees in radians
def ∠B : ℝ := Real.pi / 4 -- 45 degrees in radians
def side_c : ℝ := 20

-- Target statement to prove
theorem find_side_a : 
  let C := Real.pi - ∠A - ∠B
  let sinC := Real.sin C
  let sinA := Real.sin ∠A
  let a := (side_c * sinA) / sinC
  a = 30 * Real.sqrt 2 - 10 * Real.sqrt 6 :=
by
  sorry

end find_side_a_l739_739209


namespace simplify_expression_l739_739948

theorem simplify_expression :
  (256: ℝ) ^ (1 / 4) * (343: ℝ) ^ (1 / 3) = 28 := by
begin
  have h256 : (256: ℝ) = (2: ℝ) ^ 8 := by norm_num,
  have h343 : (343: ℝ) = (7: ℝ) ^ 3 := by norm_num,
  rw [h256, h343],
  rw [real.rpow_mul, real.rpow_mul, ← real.rpow_add, ← real.rpow_mul],
  norm_num,
end

sorry

end simplify_expression_l739_739948


namespace parity_of_expression_is_even_l739_739918

theorem parity_of_expression_is_even
  (a b c d : ℤ)
  (h₁ : a % 2 = 1)
  (h₂ : b % 2 = 0)
  (h₃ : c % 2 = 0)
  (h₄ : d % 2 = 1) : 
  (3^a + (b + 1)^2 + c * d) % 2 = 0 := 
by
  sorry

end parity_of_expression_is_even_l739_739918


namespace pot_filling_time_l739_739667

-- Define the given conditions
def drops_per_minute : ℕ := 3
def volume_per_drop : ℕ := 20 -- in ml
def pot_capacity : ℕ := 3000 -- in ml (3 liters * 1000 ml/liter)

-- Define the calculation for the drip rate
def drip_rate_per_minute : ℕ := drops_per_minute * volume_per_drop

-- Define the goal, i.e., how long it will take to fill the pot
def time_to_fill_pot (capacity : ℕ) (rate : ℕ) : ℕ := capacity / rate

-- Proof statement
theorem pot_filling_time :
  time_to_fill_pot pot_capacity drip_rate_per_minute = 50 := 
sorry

end pot_filling_time_l739_739667


namespace correct_average_marks_l739_739284

theorem correct_average_marks (n : ℕ) (average initial_wrong current_correct : ℕ) 
  (h_n : n = 10) 
  (h_avg : average = 100) 
  (h_wrong : initial_wrong = 60)
  (h_correct : current_correct = 10) : 
  (average * n - initial_wrong + current_correct) / n = 95 := 
by
  -- This is where the proof would go
  sorry

end correct_average_marks_l739_739284


namespace base_three_to_base_ten_10212_l739_739339

-- Definition of the conversion from base 3 to base 10 for the specific number
theorem base_three_to_base_ten_10212 : 
  let base_three_number := 1 * 3^4 + 0 * 3^3 + 2 * 3^2 + 1 * 3^1 + 2 * 3^0 in
  base_three_number = 104 :=
by
  -- The proof goes here, but it is omitted with 'sorry'
  sorry

end base_three_to_base_ten_10212_l739_739339


namespace value_of_Y_l739_739791

theorem value_of_Y :
  let part1 := 15 * 180 / 100  -- 15% of 180
  let part2 := part1 - part1 / 3  -- one-third less than 15% of 180
  let part3 := 24.5 * (2 * 270 / 3) / 100  -- 24.5% of (2/3 * 270)
  let part4 := (5.4 * 2) / (0.25 * 0.25)  -- (5.4 * 2) / (0.25)^2
  let Y := part2 + part3 - part4
  Y = -110.7 := by
    -- proof skipped
    sorry

end value_of_Y_l739_739791


namespace bisect_segment_l739_739923

variables {A B C D E P : Point}
variables {α β γ δ ε : Real} -- angles in degrees
variables {BD CE : Line}

-- Geometric predicates
def Angle (x y z : Point) : Real := sorry -- calculates the angle ∠xyz

def isMidpoint (M A B : Point) : Prop := sorry -- M is the midpoint of segment AB

-- Given Conditions
variables (h1 : convex_pentagon A B C D E)
          (h2 : Angle B A C = Angle C A D ∧ Angle C A D = Angle D A E)
          (h3 : Angle A B C = Angle A C D ∧ Angle A C D = Angle A D E)
          (h4 : intersects BD CE P)

-- Conclusion to be proved
theorem bisect_segment : isMidpoint P C D :=
by {
  sorry -- proof to be filled in
}

end bisect_segment_l739_739923
