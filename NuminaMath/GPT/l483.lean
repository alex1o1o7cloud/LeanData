import Data.Nat.Digits
import Mathlib
import Mathlib.Algebra.GroupPower.Lemmas
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Vector
import Mathlib.Analysis.Calculus.Pi
import Mathlib.Analysis.Complex.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Calculus.Integral
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Composition
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Cbrt
import Mathlib.Data.Set
import Mathlib.Data.Set.Basic
import Mathlib.LinearAlgebra.LinearMap
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Topology.Basic
import Mathlib.Topology.ContinuousFunction

namespace percent_time_in_meetings_l483_483769

theorem percent_time_in_meetings
  (work_day_minutes : ℕ := 8 * 60)
  (first_meeting_minutes : ℕ := 30)
  (second_meeting_minutes : ℕ := 3 * 30) :
  (first_meeting_minutes + second_meeting_minutes) / work_day_minutes * 100 = 25 :=
by
  -- sorry to skip the actual proof
  sorry

end percent_time_in_meetings_l483_483769


namespace matrix_M_properties_l483_483270

noncomputable def M : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![3, 1, 8], ![4, 6, -2], ![-9, -3, 5]]

def e_i : Fin 3 → Fin 3 → ℝ
| ⟨0, _⟩, ⟨0, _⟩ => 1
| ⟨1, _⟩, ⟨1, _⟩ => 1
| ⟨2, _⟩, ⟨2, _⟩ => 1
| _, _ => 0

def e_j : Fin 3 → Fin 3 → ℝ
| ⟨0, _⟩, ⟨1, _⟩ => 1
| ⟨1, _⟩, ⟨0, _⟩ => 1
| ⟨2, _⟩, ⟨2, _⟩ => 1
| _, _ => 0

def e_k : Fin 3 → Fin 3 → ℝ
| ⟨0, _⟩, ⟨2, _⟩ => 1
| ⟨1, _⟩, ⟨0, _⟩ => 1
| ⟨2, _⟩, ⟨0, _⟩ => 1
| _, _ => 0

theorem matrix_M_properties (λ : ℝ) :
  (M.mulVec ![1, 0, 0] = ![3, 4, -9]) ∧
  (M.mulVec ![0, 1, 0] = ![1, 6, -3]) ∧
  (M.mulVec ![0, 0, 1] = ![8, -2, 5]) ∧
  (M.mulVec (λ • ![0, 0, 1]) = λ • ![8, -2, 5]) :=
by
  -- Proof skipped
  sorry

end matrix_M_properties_l483_483270


namespace number_of_arrangements_l483_483105

def boys_and_girls_arrangement (students : List String) : Nat :=
  sorry  -- Here we would define the function to determine the number of arrangements

theorem number_of_arrangements {students : List String} (conditions : 2 boys and 3 girls ∧ boy A not at ends ∧ exactly two of the girls together) :
  boys_and_girls_arrangement students = 48 :=
sorry

end number_of_arrangements_l483_483105


namespace present_age_of_son_l483_483556

variable (S M : ℕ)

-- Conditions
def age_difference : Prop := M = S + 40
def age_relation_in_seven_years : Prop := M + 7 = 3 * (S + 7)

-- Theorem to prove
theorem present_age_of_son : age_difference S M → age_relation_in_seven_years S M → S = 13 := by
  sorry

end present_age_of_son_l483_483556


namespace triangle_incircle_ratio_l483_483546

theorem triangle_incircle_ratio
  (a b c : ℝ) (ha : a = 15) (hb : b = 12) (hc : c = 9)
  (r s : ℝ) (hr : r + s = c) (r_lt_s : r < s) :
  r / s = 1 / 2 :=
sorry

end triangle_incircle_ratio_l483_483546


namespace polynomial_square_b_value_l483_483812

theorem polynomial_square_b_value (a b : ℚ) (h : ∃ (p q : ℚ), x^4 + 3 * x^3 + x^2 + a * x + b = (x^2 + p * x + q)^2) : 
  b = 25/64 := 
by 
  -- Proof steps go here
  sorry

end polynomial_square_b_value_l483_483812


namespace series_sum_correct_l483_483206

noncomputable def sum_series : ℚ :=
  ∑ n in finset.range 150, 1 / ((3 * (n + 1) - 1) * (3 * (n + 1) + 1))

theorem series_sum_correct : sum_series = 225 / 904 :=
by
  sorry

end series_sum_correct_l483_483206


namespace find_c_of_odd_function_find_min_value_of_f_l483_483332

def f (x : ℝ) (c : ℝ) : ℝ := (x^2 + 1) / (x + c)

theorem find_c_of_odd_function : ∀ c : ℝ, (∀ x : ℝ, f (-x) c = -f x c) → c = 0 :=
by
  intro c h
  have h₁ : f (-1) c = -f 1 c := h 1
  have h₂ : f (-1) c = -(1^2 + 1) / (1 + c) := by simp [f]
  have h₃ : f 1 c = (1^2 + 1) / (1 + c) := by simp [f]
  have h₄ : f (-1) c = -f 1 c → -(1^2 + 1) / (1 + c) = -(1^2 + 1) / (1 + c) := by simp [h₁, h₂, h₃]
  have h₅ : -(1^2 + 1) / (1 + c) = ((-(1^2 + 1)) / (1 + c)) := by ring
  have h₆ : -(1^2 + 1) = -(1^2 + 1) → c = 0 := by
    simp [(h₄ h₅)]
  exact h₆

theorem find_min_value_of_f : min_value_on_set (λ x, f x 0) (set.Ici 2) = 5 / 2 :=
by
  sorry

end find_c_of_odd_function_find_min_value_of_f_l483_483332


namespace same_length_segments_l483_483008

open Finset

theorem same_length_segments
  (points : Finset (ℕ × ℕ))
  (h_points : points.card = 2017)
  (h_x_range : ∀ p ∈ points, 1 ≤ p.1 ∧ p.1 ≤ 2016)
  (h_y_range : ∀ p ∈ points, 1 ≤ p.2 ∧ p.2 ≤ 2016) :
  ∃ p₁ p₂ p₃ p₄ ∈ points, (p₁ ≠ p₂ ∧ p₃ ≠ p₄) ∧ (euclidean_dist p₁ p₂ = euclidean_dist p₃ p₄) := 
sorry

end same_length_segments_l483_483008


namespace factorial_divides_product_l483_483415

theorem factorial_divides_product (a b n : ℕ) (ha : 0 < a) (hb : 0 < b) (hn : 0 < n) :
  n.factorial ∣ b^(n-1) * (List.range n).map (λ k => a + k * b) .prod := sorry

end factorial_divides_product_l483_483415


namespace tan_11_25_form_l483_483203

theorem tan_11_25_form :
  ∃ a b c d : ℕ, a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ tan (11.25 * (Real.pi / 180)) = Real.sqrt a - Real.sqrt b + Real.sqrt c - d ∧ a + b + c + d = 8 :=
by
  sorry

end tan_11_25_form_l483_483203


namespace f_five_eq_eight_l483_483640

def f (x : ℕ) : ℕ := 
  if x ≥ 10 then x - 3 
  else f (f (x + 6))

theorem f_five_eq_eight : f 5 = 8 := 
by 
  sorry

end f_five_eq_eight_l483_483640


namespace find_c_and_area_l483_483689

noncomputable def f (x c : ℝ) : ℝ := x^3 - x^2 - x + c

theorem find_c_and_area :
  (∃ c : ℝ, ∀ x : ℝ, f x c = 0 → f x c = 1 → ∫ x in -1..1, f x 1 = 4 / 3) :=
by sorry

end find_c_and_area_l483_483689


namespace min_area_ratio_l483_483677

theorem min_area_ratio (A B C D E F : Point) (hABC : equilateral_triangle A B C) 
  (hD : D ∈ segment A B) (hE : E ∈ segment B C) (hF : F ∈ segment C A) 
  (hDEF : right_triangle D E F) (h_angle_DEF : angle D E F = 90) (h_angle_EDF : angle E D F = 30) :
  area_ratio S_triangle_DEF S_triangle_ABC = 3 / 14 := 
sorry

end min_area_ratio_l483_483677


namespace jacqueline_boxes_of_erasers_l483_483021

theorem jacqueline_boxes_of_erasers (total_erasers : ℕ) (erasers_per_box : ℕ) (total_erasers = 40) (erasers_per_box = 10) :
  total_erasers / erasers_per_box = 4 :=
by
  sorry

end jacqueline_boxes_of_erasers_l483_483021


namespace rectangle_area_l483_483894

theorem rectangle_area
    (w l : ℕ)
    (h₁ : 28 = 2 * (l + w))
    (h₂ : w = 6) : l * w = 48 :=
by
  sorry

end rectangle_area_l483_483894


namespace object_l483_483716

theorem object's_speed (distance_ft : ℝ) (time_sec : ℝ) (mile_in_ft : ℝ) :
  distance_ft = 300 →
  time_sec = 5 →
  mile_in_ft = 5280 →
  (distance_ft / mile_in_ft) / (time_sec / 3600) ≈ 40.909 :=
by
  intros h1 h2 h3
  sorry

end object_l483_483716


namespace smallest_arithmetic_mean_divisible_by_1111_l483_483853

/-- 
Given the product of nine consecutive natural numbers is divisible by 1111, 
prove that the smallest possible value of the arithmetic mean of these nine numbers is 97.
-/
theorem smallest_arithmetic_mean_divisible_by_1111 :
  ∃ n : ℕ, (∀ k : ℕ, k = n →  (∏ i in finset.range 9, k + i) % 1111 = 0) 
  ∧ (n ≥ 93) ∧ (n + 4 = 97) :=
sorry

end smallest_arithmetic_mean_divisible_by_1111_l483_483853


namespace min_area_ratio_of_inscribed_right_triangle_in_equilateral_l483_483674

theorem min_area_ratio_of_inscribed_right_triangle_in_equilateral (a b c d e f : ℝ) 
    (hDEF : (d = 0 ∧ e = 0 ∧ f = 0) ∨ ( (a,b,c) ≠ (0,0,0) ∧ ∠ DEF = 90 ∧ ∠ EDF = 30 )) 
    (hABC : (a,b,c) ∈ equilateral_triangle ) :
    ∃ DEF ABC : ℝ, min (frac_area DEF ABC) = 3 / 14 := 
sorry

end min_area_ratio_of_inscribed_right_triangle_in_equilateral_l483_483674


namespace n_greater_than_sqrt_p_sub_1_l483_483405

theorem n_greater_than_sqrt_p_sub_1 {p n : ℕ} (hp : Nat.Prime p) (hn : n ≥ 2) (hdiv : p ∣ (n^6 - 1)) : n > Nat.sqrt p - 1 := 
by
  sorry

end n_greater_than_sqrt_p_sub_1_l483_483405


namespace compute_xy_l483_483893

variable (x y : ℝ)

-- Conditions from the problem
def condition1 : Prop := x + y = 10
def condition2 : Prop := x^3 + y^3 = 172

-- Theorem statement to prove the answer
theorem compute_xy (h1 : condition1 x y) (h2 : condition2 x y) : x * y = 41.4 :=
sorry

end compute_xy_l483_483893


namespace exists_tangent_sphere_l483_483179

-- Definitions
structure Polyhedron :=
  (vertices : Set Point)
  (edges : Set (Point × Point))
  (is_convex : ConvexHull vertices = {p | p ∈ vertices})

structure Sphere :=
  (center : Point)
  (radius : ℝ)
  
def trisects_edge (S : Sphere) (P : Polyhedron) :=
  ∀ {X Y : Point}, ({X, Y} ∈ P.edges) →
    ∃ X1 Y1 : Point, (distance X X1 = distance X1 Y1) ∧
                     (distance Y1 Y = distance X Y / 3) ∧
                     (distance X1 Y1 = distance X Y / 3) ∧
                     X1 ≠ Y1 ∧
                     (X1, Y1 ∈ S)

-- The proof statement
theorem exists_tangent_sphere (P : Polyhedron) (S : Sphere) (h : trisects_edge S P) :
  ∃ S' : Sphere, ∀ {X Y : Point}, ({X, Y} ∈ P.edges) →
    (∃ X1 Y1 : Point, X1 ≠ Y1 ∧
                      X1 ∈ S' ∧
                      Y1 ∈ S' ∧
                      distance (S'.center) X1 = S'.radius ∧
                      distance (S'.center) Y1 = S'.radius) := 
sorry

end exists_tangent_sphere_l483_483179


namespace max_true_statements_l483_483038

-- Definitions and conditions from the problem
variables (a b c : ℝ)

def statement1 : Prop := (1 / a < 1 / b)
def statement2 : Prop := (a^2 > b^2)
def statement3 : Prop := (a < b)
def statement4 : Prop := (a < 0)
def statement5 : Prop := (b < 0)
def statement6 : Prop := (c < (a + b) / 2)

-- The proof problem statement
theorem max_true_statements : 
  ∃ (a b c : ℝ), set.count_true (set.of [statement1 a b, statement2 a b, statement3 a b, statement4 a b, statement5 a b, statement6 a b c]) = 4 :=
sorry

end max_true_statements_l483_483038


namespace sum_of_tens_and_ones_digits_of_7_pow_15_l483_483922

-- Definition for checking cyclical patterns of the digits
def cyclic_ones_7 (n : ℕ) : ℕ :=
  let ones_cycle := [7, 9, 3, 1]
  ones_cycle[(n % 4).toNat]

def cyclic_tens_7 (n : ℕ) : ℕ :=
  let tens_cycle := [0, 4, 4, 0]
  tens_cycle[(n % 4).toNat]

-- Main theorem statement
theorem sum_of_tens_and_ones_digits_of_7_pow_15 :
  cyclic_ones_7 15 + cyclic_tens_7 15 = 7 := by
  sorry

end sum_of_tens_and_ones_digits_of_7_pow_15_l483_483922


namespace skillful_hands_wire_cut_l483_483525

theorem skillful_hands_wire_cut :
  ∃ x : ℕ, (1000 = 15 * x) ∧ (1040 = 15 * x) ∧ x = 66 :=
by
  sorry

end skillful_hands_wire_cut_l483_483525


namespace initial_soldiers_l483_483733

-- Define the initial conditions and variables
variables (S : ℕ) (P : ℕ)

-- Condition 1: Each soldier consumes 3 kg/day and provisions last for 30 days
def provision_first_condition : Prop := P = S * 3 * 30

-- Condition 2: 528 more soldiers join, each consumes 2.5 kg/day and provisions last for 25 days
def provision_second_condition : Prop := P = (S + 528) * 25 * 2.5

-- Theorem: Prove the initial number of soldiers is 1200 under the given conditions
theorem initial_soldiers : provision_first_condition S P → provision_second_condition S P → S = 1200 :=
by {
  -- Skip the proof
  sorry 
}

end initial_soldiers_l483_483733


namespace line_intersects_circle_l483_483883

theorem line_intersects_circle (a : ℝ) : 
  let r := 2 * real.sqrt 2
  let d := abs(1 - a) / real.sqrt (1 + a^2)
  (d < r) → True := 
by
  let r := 2 * real.sqrt 2
  let d := abs(1 - a) / real.sqrt (1 + a^2)
  intro h
  trivial

end line_intersects_circle_l483_483883


namespace man_rate_in_still_water_l483_483935

theorem man_rate_in_still_water (V_m V_s : ℝ)
  (h1 : V_m + V_s = 16)
  (h2 : V_m - V_s = 8) :
  V_m = 12 :=
by
  have h_add : 2 * V_m = 24 :=
    by linarith
  have h_div : V_m = 24 / 2 :=
    by algebra
  have h_res : V_m = 12 :=
    by norm_num
  assumption

end man_rate_in_still_water_l483_483935


namespace largest_binomial_term_l483_483491

theorem largest_binomial_term (x : ℝ) : 
  let n := 6 in let a := 1 in let b := 2 in
  (n % 2 = 0) →
  ((a + b * x) ^ n).expand.coeff (n / 2) = 160 * x ^ 3 :=
by
  sorry

end largest_binomial_term_l483_483491


namespace probability_odd_even_draw_l483_483536

theorem probability_odd_even_draw :
  let balls := {1, 2, 3, 4, 5}
  let odd_balls := {1, 3, 5}
  let even_balls := {2, 4}
  let total_balls := 5
  let first_draw_odd := (odd_balls.card : ℚ) / total_balls
  let remaining_after_odd := total_balls - 1
  let second_draw_even := (even_balls.card : ℚ) / remaining_after_odd
  (first_draw_odd * second_draw_even = 3 / 10) :=
by
  intros
  rw [Set.card_image_of_injective _ (Set.pairwise_injective_of_fintype _), Set.to_finite.card,
      Set.card_image_of_injective _ (Set.pairwise_injective_of_fintype _), Set.to_finite.card]
  simp [first_draw_odd, second_draw_even, div_eq_mul_inv]
  norm_num
  sorry

end probability_odd_even_draw_l483_483536


namespace select_4_students_count_l483_483285

/-- There is a group of 10 students including A, B, C. 
If A is selected, then B must be selected.
If A is not selected, then C must be selected.
Prove that the number of ways to select 4 students for an activity is 84.
-/
theorem select_4_students_count : 
  let n := 10 in 
  let students := fin n in
  let A B C : students := 0, 1, 2 in
  let X := Finset.powersetLen 4 (Finset.univ : Finset students) in
  let S := {S ∈ X | ((A ∈ S ∧ B ∈ S) ∨ (A ∉ S ∧ C ∈ S))} in
  Finset.card S = 84 :=
by
  let n := 10
  let students := fin n
  let A B C : students := 0, 1, 2
  let X := Finset.powersetLen 4 (Finset.univ : Finset students)
  let S := {S ∈ X | ((A ∈ S ∧ B ∈ S) ∨ (A ∉ S ∧ C ∈ S))}
  have card_X : Finset.card X = 210 := by sorry
  have card_S : Finset.card S = 84 := by sorry
  exact card_S

end select_4_students_count_l483_483285


namespace inequality_solution_l483_483792

theorem inequality_solution (x : ℝ) :
  (x > -4 ∧ x < -5 / 3) ↔ 
  (2 * x + 3) / (3 * x + 5) > (4 * x + 1) / (x + 4) := 
sorry

end inequality_solution_l483_483792


namespace largest_angle_ABC_l483_483726

variable (m : ℝ) (h₀ : m > 0)
def side_a := 2 * m + 3
def side_b := m^2 + 2 * m
def side_c := m^3 + 3 * m + 3

-- Largest angle is opposite the longest side
-- Proof that the largest angle in △ABC is 120 degrees
theorem largest_angle_ABC :
  (angle_of_opposite_side_eq_max side_a side_b side_c h₀ = 120) :=
sorry

end largest_angle_ABC_l483_483726


namespace triangle_side_a_l483_483650

theorem triangle_side_a (a : ℝ) (h1 : 4 < a) (h2 : a < 10) : a = 8 :=
  by
  sorry

end triangle_side_a_l483_483650


namespace train_length_l483_483142

theorem train_length
  (L : ℝ)
  (same_direction : True)
  (length_equal : True)
  (speed_faster : 46) -- km/hr
  (speed_slower : 36) -- km/hr
  (time_to_pass : 108) -- seconds
  (relative_speed : 10 = 46 - 36)
  (relative_speed_ms : (10 * (5 / 18)) = (25 / 9)) :
  L = 150 :=
by
  sorry

end train_length_l483_483142


namespace ellipse_equation_fixed_point_line_l483_483298

-- Define the conditions as hypotheses
def eccentricity (e : ℝ) := e = (real.sqrt 2) / 2
def foci_coordinates (F1 F2 : ℝ × ℝ) (c : ℝ) := F1 = (-c, 0) ∧ F2 = (c, 0)
def fixed_point (P : ℝ × ℝ) := P = (2, real.sqrt 3)
def bisector_condition (F1 F2 P : ℝ × ℝ) := dist F1 F2 = dist P F2

-- Define the questions as Lean theorem statements
theorem ellipse_equation (e : ℝ) (F1 F2 P : ℝ × ℝ) (c a b : ℝ)
  (h_ecc : eccentricity e)
  (h_foci : foci_coordinates F1 F2 c)
  (h_P : fixed_point P)
  (h_bis : bisector_condition F1 F2 P) :
  (a = real.sqrt 2 * c) ∧ ((c = 1) ∧ a = real.sqrt 2) → b^2 = a^2 - c^2 → (∀ x y : ℝ, (x^2) / (2) + (y^2) = 1) :=
by sorry

theorem fixed_point_line (k : ℝ) (m : ℝ) (x₁ x₂ : ℝ)
  (h_line_intersect_ellipse : ∀ x y : ℝ, y = k * x + m → (x^2) / 2 + (y^2) = 1) :
  m = -2 * k → (∀ x y : ℝ, y = k * (x - 2)) → (x = 2 ∧ y = 0) :=
by sorry

end ellipse_equation_fixed_point_line_l483_483298


namespace smallest_arithmetic_mean_divisible_product_l483_483840

theorem smallest_arithmetic_mean_divisible_product :
  ∃ (n : ℕ), (∀ i : ℕ, i < 9 → Nat.gcd ((n + i) * 1111) 1111 = 1111) ∧ (n + 4 = 97) := 
  sorry

end smallest_arithmetic_mean_divisible_product_l483_483840


namespace min_period_2sin2x_sub_1_l483_483807

theorem min_period_2sin2x_sub_1 :
  ∀ x : ℝ, ∃ T > 0, (∀ x : ℝ, 2 * sin x ^ 2 - 1 = 2 * sin (x + T) ^ 2 - 1) ∧
    (∀ T2 > 0, (∀ x : ℝ, 2 * sin (x + T2) ^ 2 - 1 = 2 * sin x ^ 2 - 1) → T ≤ T2) :=
begin
  sorry
end

end min_period_2sin2x_sub_1_l483_483807


namespace proj_w_v_eq_2_5_proj_u_v_eq_2_5_l483_483336

variables (v w u : ℝ^3)
variables (v_norm : ∥v∥ = 5) (w_norm : ∥w∥ = 8) (u_norm : ∥u∥ = 6)
variables (v_dot_w : v • w = 20) (v_dot_u : v • u = 15)

noncomputable def proj_w_v : ℝ := ∥(v • w) / ∥w∥∥
noncomputable def proj_u_v : ℝ := ∥(v • u) / ∥u∥∥

theorem proj_w_v_eq_2_5 : proj_w_v v w = 2.5 := by
  sorry

theorem proj_u_v_eq_2_5 : proj_u_v v u = 2.5 := by
  sorry

end proj_w_v_eq_2_5_proj_u_v_eq_2_5_l483_483336


namespace min_value_sin_cos_l483_483635

theorem min_value_sin_cos (α : ℝ) (hα : α ∈ set.Ioo 0 (Real.pi / 2)) :
  ∃ x, x = 1 ∧ ∀ y, y = (sin(α) ^ 3) / (cos(α)) + (cos(α) ^ 3) / (sin(α)) → y ≥ 1 :=
by    sorry

end min_value_sin_cos_l483_483635


namespace tan_alpha_sub_60_l483_483305

theorem tan_alpha_sub_60 
  (alpha : ℝ) 
  (h : Real.tan alpha = 4 * Real.sin (420 * Real.pi / 180)) : 
  Real.tan (alpha - 60 * Real.pi / 180) = (Real.sqrt 3) / 7 :=
by sorry

end tan_alpha_sub_60_l483_483305


namespace game_probability_after_2025_rings_l483_483071

def initial_state : (ℕ × ℕ × ℕ) := (3, 3, 3)
def bell_rings (state : ℕ × ℕ × ℕ) (rings : ℕ) : Prop := sorry

theorem game_probability_after_2025_rings :
  bell_rings initial_state 2025 = (3, 3, 3) → 
  (∃ p : ℚ, p = 2/27) :=
sorry

end game_probability_after_2025_rings_l483_483071


namespace medians_concurrent_perpendicular_bisectors_concurrent_altitudes_concurrent_nine_point_circle_l483_483946

-- Define the basic structure - a triangle
structure Triangle (K : Type*) [Field K] :=
(A B C : K × K)

-- Define the midpoint function
def midpoint {K : Type*} [Field K] (p1 p2 : K × K) : K × K :=
((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Define the altitudes foot
def foot_of_altitude {K : Type*} [Field K] (t: Triangle K) (A : K × K) : K × K :=
-- Placeholder definition, typically requires reflection logic
sorry

-- Define the centroid (concurrence of medians)
def centroid {K : Type*} [Field K] (t: Triangle K) : K × K :=
((t.A.1 + t.B.1 + t.C.1) / 3, (t.A.2 + t.B.2 + t.C.2) / 3)

-- Define circumcenter
def circumcenter {K : Type*} [Field K] (t: Triangle K) : K × K :=
-- Placeholder definition, requires solving perpendicularly bisected lines intersection
sorry

-- Define orthocenter
def orthocenter {K : Type*} [Field K] (t: Triangle K) : K × K :=
-- Placeholder definition, typically requires other geometric constructions
sorry

-- Define Euler circle center
def euler_circle_center {K : Type*} [Field K] (O H : K × K) : K × K :=
midpoint O H

-- Define Euler circle
noncomputable def euler_circle_radius {K : Type*} [Field K] (R : K) : K :=
R / 2

-- The core statements we need to prove
theorem medians_concurrent {K : Type*} [Field K] (t: Triangle K) :
  ∃ G, 1 = 1 ∧ 1 = 1 := sorry

theorem perpendicular_bisectors_concurrent {K : Type*} [Field K] (t: Triangle K) :
  ∃ O, 1 = 1 ∧ 1 = 1 := sorry

theorem altitudes_concurrent {K : Type*} [Field K] (t: Triangle K) :
  ∃ H, 1 = 1 ∧ 1 = 1 := sorry

theorem nine_point_circle (K : Type*) [Field K] (t: Triangle K) :
  let A' := midpoint t.B t.C,
      B' := midpoint t.A t.C,
      C' := midpoint t.A t.B,
      H_A := foot_of_altitude t t.A,
      H_B := foot_of_altitude t t.B,
      H_C := foot_of_altitude t t.C,
      G := centroid t,
      O := circumcenter t,
      H := orthocenter t,
      P_A := midpoint H t.A,
      P_B := midpoint H t.B,
      P_C := midpoint H t.C,
      Omega := euler_circle_center O H,
      R := 1 -- Placeholder for circumradius
  in ∀ P, P ∈ {A', B', C', H_A, H_B, H_C, P_A, P_B, P_C} → P ∈ circle Omega (euler_circle_radius R) := sorry

end medians_concurrent_perpendicular_bisectors_concurrent_altitudes_concurrent_nine_point_circle_l483_483946


namespace profit_ratio_7_10_l483_483096

def profit_ratio (investment_p : ℕ) (investment_q : ℕ) (time_p : ℕ) (time_q : ℕ) : (ℕ × ℕ) :=
  (investment_p * time_p, investment_q * time_q)

theorem profit_ratio_7_10 (x : ℕ) :
  let investment_p := 7 * x 
  let investment_q := 5 * x
  let time_p := 2
  let time_q := 4 in
  profit_ratio investment_p investment_q time_p time_q = (7 * 2 * x, 5 * 4 * x) ∧
  prod.fst (7 * 2 * x, 5 * 4 * x) / gcd (7 * 2 * x) (5 * 4 * x) = 7 ∧
  prod.snd (7 * 2 * x, 5 * 4 * x) / gcd (7 * 2 * x) (5 * 4 * x) = 10 :=
by
  sorry

end profit_ratio_7_10_l483_483096


namespace sum_of_tens_and_ones_digits_of_7_pow_15_l483_483920

-- Definition for checking cyclical patterns of the digits
def cyclic_ones_7 (n : ℕ) : ℕ :=
  let ones_cycle := [7, 9, 3, 1]
  ones_cycle[(n % 4).toNat]

def cyclic_tens_7 (n : ℕ) : ℕ :=
  let tens_cycle := [0, 4, 4, 0]
  tens_cycle[(n % 4).toNat]

-- Main theorem statement
theorem sum_of_tens_and_ones_digits_of_7_pow_15 :
  cyclic_ones_7 15 + cyclic_tens_7 15 = 7 := by
  sorry

end sum_of_tens_and_ones_digits_of_7_pow_15_l483_483920


namespace probability_normal_distribution_l483_483670

theorem probability_normal_distribution (ξ : ℝ → ℝ) (δ : ℝ) 
  (H1 : ξ ∼ Normal 2 δ^2)
  (H2 : ∫ (x : ℝ) in -∞..3, ξ x = 0.8413) : 
  ∫ (x : ℝ) in -∞..1, ξ x = 0.1587 := 
sorry

end probability_normal_distribution_l483_483670


namespace university_students_problem_l483_483569

/-- Lean statement for the university students' problem -/
theorem university_students_problem :
  let H_min := (0.7 * 3000).ceil
  let H_max := (0.75 * 3000).floor
  let P_min := (0.4 * 3000).ceil
  let P_max := (0.5 * 3000).floor
  let m' := H_max + P_max - 3000
  let M' := H_min + P_min - 3000
  m' - M' = 450 :=
by
  let H_min := (0.7 * 3000).ceil
  let H_max := (0.75 * 3000).floor
  let P_min := (0.4 * 3000).ceil
  let P_max := (0.5 * 3000).floor
  let m' :=  H_max + P_max - 3000
  let M' := H_min + P_min - 3000
  have h1 : m' = 750 := sorry
  have h2 : M' = 300 := sorry
  exact h1 - h2

end university_students_problem_l483_483569


namespace second_rewind_time_l483_483572

noncomputable def viewing_time_before_last_uninterrupted_part : ℕ :=
  35 + 5 + 45 + 20

theorem second_rewind_time (total_time : ℕ) (added_time : ℕ) :
    total_time = 120 ∧ viewing_time_before_last_uninterrupted_part = 105 →
    added_time = total_time - 105 → added_time = 15 := by
  intro h
  unfold viewing_time_before_last_uninterrupted_part at h
  cases h with h_total h_view
  rw [h_total, h_view]
  intro h_added
  rw [h_added]
  rfl

end second_rewind_time_l483_483572


namespace select_pencils_erasers_cases_l483_483276

theorem select_pencils_erasers_cases (pencils erasers : ℕ) (hp : pencils = 3) (he : erasers = 2) : 
  pencils * erasers = 6 := by
  rw [hp, he]
  calc
    3 * 2 = 6 : by norm_num
  
  sorry

end select_pencils_erasers_cases_l483_483276


namespace expected_waiting_time_first_bite_l483_483236

-- Definitions and conditions as per the problem
def poisson_rate := 6  -- lambda value, bites per 5 minutes
def interval_minutes := 5
def interval_seconds := interval_minutes * 60
def expected_waiting_time_seconds := interval_seconds / poisson_rate

-- The theorem we want to prove
theorem expected_waiting_time_first_bite :
  expected_waiting_time_seconds = 50 := 
by
  let x := interval_seconds / poisson_rate
  have h : interval_seconds = 300 := by norm_num; rfl
  have h2 : x = 50 := by rw [h, interval_seconds]; norm_num
  exact h2

end expected_waiting_time_first_bite_l483_483236


namespace system_solution_l483_483075

theorem system_solution (x y : ℝ) 
    (hx1 : 0 < x) (hx2 : x ≠ 1) 
    (hy1 : 0 < y) (hy2 : y ≠ 1) 
    (h1 : log y x - log x y = 8 / 3)
    (h2 : x * y = 16) :
    (x = 8 ∧ y = 2) ∨ (x = 1 / 4 ∧ y = 64) :=
sorry

end system_solution_l483_483075


namespace probability_of_valid_region_l483_483559

-- Define the bounds for x and y
def in_bounds (x y : ℝ) : Prop := (0 ≤ x ∧ x ≤ 4) ∧ (0 ≤ y ∧ y ≤ 8)

-- Define the region where x + y <= 6
def valid_region (x y : ℝ) : Prop := x + y ≤ 6

-- Define the total area
def total_area : ℝ := 32

-- Define the valid area
def valid_area : ℝ := 12

-- Define the probability
def probability : ℝ := valid_area / total_area

theorem probability_of_valid_region :
  (∀ x y, in_bounds x y → valid_region x y) → probability = 3 / 8 := 
by
  sorry

end probability_of_valid_region_l483_483559


namespace chord_length_intersection_l483_483090

theorem chord_length_intersection :
  let l : set (ℝ × ℝ) := {p | p.1 - p.2 + 3 = 0}
  let C : set (ℝ × ℝ) := {p | (p.1 - 2)^2 + (p.2 - 3)^2 = 16}
  ∃ r ∈ ℝ, r = 2 * Real.sqrt 14 → ∃ p1 p2 ∈ l ∩ C, Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) = r
:= sorry

end chord_length_intersection_l483_483090


namespace expected_waiting_time_for_first_bite_l483_483214

noncomputable def average_waiting_time (λ : ℝ) : ℝ := 1 / λ

theorem expected_waiting_time_for_first_bite (bites_first_rod : ℝ) (bites_second_rod : ℝ) (total_time_minutes : ℝ) (total_time_seconds : ℝ) :
  bites_first_rod = 5 → 
  bites_second_rod = 1 → 
  total_time_minutes = 5 → 
  total_time_seconds = 300 → 
  average_waiting_time (bites_first_rod + bites_second_rod) * total_time_seconds = 50 :=
begin
  intros,
  sorry
end

end expected_waiting_time_for_first_bite_l483_483214


namespace complex_number_value_l483_483719

def z (a b : ℝ) : ℂ := a + b * complex.I

theorem complex_number_value (a b : ℝ) (h : (3 - z a b) * complex.I = 2 ) : 
  z a b = 3 + 2 * complex.I :=
by 
  sorry

end complex_number_value_l483_483719


namespace nine_consecutive_arithmetic_mean_divisible_1111_l483_483873

theorem nine_consecutive_arithmetic_mean_divisible_1111 {n : ℕ} (h1 : ∀ i : ℕ, 0 ≤ i ∧ i < 9 → Nat.Prime (n + i)) :
  ∃ n : ℕ, (∀ k : ℕ, 0 ≤ k ∧ k < 9 → (n + k) ∣ 1111) → (n + 4) = 97 := by
  sorry

end nine_consecutive_arithmetic_mean_divisible_1111_l483_483873


namespace problem_1_problem_2_l483_483327

def f (x a : ℝ) : ℝ := abs (2 * x - a) + abs (2 * x + 3)
def g (x : ℝ) : ℝ := abs (2 * x - 3) + 2

theorem problem_1 (x : ℝ) :
  abs (g x) < 5 → 0 < x ∧ x < 3 :=
sorry

theorem problem_2 (a : ℝ) :
  (∀ x1 : ℝ, ∃ x2 : ℝ, f x1 a = g x2) →
  (a ≥ -1 ∨ a ≤ -5) :=
sorry

end problem_1_problem_2_l483_483327


namespace solve_for_x_l483_483242

theorem solve_for_x : ∃ x : ℤ, 250957 + x^3 = 18432100 ∧ x = 263 :=
by
  use 263
  simp
  sorry

end solve_for_x_l483_483242


namespace rationalize_denominator_l483_483447

noncomputable def cube_root (x : ℝ) := x^(1/3)

theorem rationalize_denominator (a b : ℝ) (h : cube_root 27 = 3) : 
  1 / (cube_root 3 + cube_root 27) = (3 - cube_root 3) / (9 - 3 * cube_root 3) :=
by
  sorry

end rationalize_denominator_l483_483447


namespace length_of_parallel_line_closer_to_base_l483_483187

-- Given definitions and conditions
variables {α : Type} [LinearOrderedField α]

def triangle_base : α := 20
def first_line_divides_into_four_equal_areas (base : α) : α := base / 2
def second_line_divides_quarter_area_into_three (first_line : α) : α := first_line / (Real.sqrt 3)

-- State the theorem to be proved
theorem length_of_parallel_line_closer_to_base :
  let triangle_base := 20 : ℝ in
  let first_line := first_line_divides_into_four_equal_areas triangle_base in
  let second_line := second_line_divides_quarter_area_into_three first_line in
  second_line = 10 * (Real.sqrt 3) / 3 :=
by
  sorry

end length_of_parallel_line_closer_to_base_l483_483187


namespace estimated_prob_is_0_9_l483_483890

section GerminationProbability

-- Defining the experiment data
structure ExperimentData :=
  (totalSeeds : ℕ)
  (germinatedSeeds : ℕ)
  (germinationRate : ℝ)

def experiments : List ExperimentData := [
  ⟨100, 91, 0.91⟩, 
  ⟨400, 358, 0.895⟩, 
  ⟨800, 724, 0.905⟩,
  ⟨1400, 1264, 0.903⟩,
  ⟨3500, 3160, 0.903⟩,
  ⟨7000, 6400, 0.914⟩
]

-- Hypothesis based on the given problem's observation
def estimated_germination_probability (experiments : List ExperimentData) : ℝ :=
  /- Fictively calculating the stable germination rate here; however, logically we should use 
     some weighted average or similar statistical stability method. -/
  0.9  -- Rounded and concluded estimated value based on observation

theorem estimated_prob_is_0_9 :
  estimated_germination_probability experiments = 0.9 :=
  sorry

end GerminationProbability

end estimated_prob_is_0_9_l483_483890


namespace reflection_matrix_squared_identity_l483_483410

/-- Reflection matrix over the vector (3, 1) squares to the identity matrix. -/
theorem reflection_matrix_squared_identity (R : Matrix (Fin 2) (Fin 2) ℝ)
  (h : ∀ v : Vector (Fin 2) ℝ, R.mulVec v = v) : R.mulVec R = 1 :=
sorry

end reflection_matrix_squared_identity_l483_483410


namespace smallest_positive_four_digit_equivalent_to_5_mod_8_l483_483913

theorem smallest_positive_four_digit_equivalent_to_5_mod_8 : 
  ∃ (n : ℕ), n ≥ 1000 ∧ n % 8 = 5 ∧ n = 1005 :=
by
  sorry

end smallest_positive_four_digit_equivalent_to_5_mod_8_l483_483913


namespace standing_arrangements_l483_483751

theorem standing_arrangements : ∃ (arrangements : ℕ), arrangements = 2 :=
by
  -- Given that Jia, Yi, Bing, and Ding are four distinct people standing in a row
  -- We need to prove that there are exactly 2 different ways for them to stand such that Jia is not at the far left and Yi is not at the far right
  sorry

end standing_arrangements_l483_483751


namespace workers_appointment_l483_483887

theorem workers_appointment (F T V : ℕ) (hF : F = 5) (hT : T = 4) (hV : V = 2) : 
  let ways := (choose 5 4) * (choose 4 3) * (choose 2 1) +
              (choose 5 3) * (choose 4 4) * (choose 2 1) +
              (choose 5 4) * (choose 4 2) * (choose 2 2) +
              (choose 5 3) * (choose 4 3) * (choose 2 2) +
              (choose 5 3) * (choose 4 2) * (choose 2 2) in
  ways = 190 :=
by
  intros
  have h1 : (choose 5 4) * (choose 4 3) * (choose 2 1) = 40 := by sorry
  have h2 : (choose 5 3) * (choose 4 4) * (choose 2 1) = 20 := by sorry
  have h3 : (choose 5 4) * (choose 4 2) * (choose 2 2) = 30 := by sorry
  have h4 : (choose 5 3) * (choose 4 3) * (choose 2 2) = 40 := by sorry
  have h5 : (choose 5 3) * (choose 4 2) * (choose 2 2) = 60 := by sorry
  let ways := 40 + 20 + 30 + 40 + 60
  show ways = 190, by sorry

end workers_appointment_l483_483887


namespace sin_add_pi_div_two_eq_cos_l483_483280

theorem sin_add_pi_div_two_eq_cos (x : Real) : sin (x + π / 2) = cos x := 
by {
  sorry 
}

end sin_add_pi_div_two_eq_cos_l483_483280


namespace positive_x_condition_l483_483176

theorem positive_x_condition (x : ℝ) (h : x > 0 ∧ (0.01 * x * x = 9)) : x = 30 :=
sorry

end positive_x_condition_l483_483176


namespace arrange_six_lines_l483_483196

structure point := (x : ℝ) (y : ℝ)
structure line := (a b : point)

noncomputable def pointA : point := {x := 0, y := 0}
noncomputable def pointB : point := {x := 1, y := 1}
noncomputable def pointC : point := {x := 2, y := 0}
noncomputable def pointD : point := {x := 1, y := -1}
noncomputable def pointE : point := {x := 1, y := 0}

noncomputable def lineAB : line := {a := pointA, b := pointB}
noncomputable def lineBC : line := {a := pointB, b := pointC}
noncomputable def lineCD : line := {a := pointC, b := pointD}
noncomputable def lineDA : line := {a := pointD, b := pointA}
noncomputable def lineAC : line := {a := pointA, b := pointC}
noncomputable def lineBD : line := {a := pointB, b := pointD}

theorem arrange_six_lines :
  ∃ (lines : list line) (points : list point), lines.length = 6 ∧ points.length = 7 ∧
    (∀ l ∈ lines, ∃ p₁ p₂ p₃ ∈ points, l = line.mk p₁ p₂ ∨ l = line.mk p₁ p₃ ∨ l = line.mk p₂ p₃) :=
by
  sorry

end arrange_six_lines_l483_483196


namespace cyclic_ABCD_l483_483421

variable {Point : Type}
variable {Angle LineCircle : Type → Type}
variable {cyclicQuadrilateral : List (Point) → Prop}
variable {convexQuadrilateral : List (Point) → Prop}
variable {lineSegment : Point → Point → LineCircle Point}
variable {onSegment : Point → LineCircle Point → Prop}
variable {angle : Point → Point → Point → Angle Point}

theorem cyclic_ABCD (A B C D P Q E : Point)
  (h1 : convexQuadrilateral [A, B, C, D])
  (h2 : cyclicQuadrilateral [P, Q, D, A])
  (h3 : cyclicQuadrilateral [Q, P, B, C])
  (h4 : onSegment E (lineSegment P Q))
  (h5 : angle P A E = angle Q D E)
  (h6 : angle P B E = angle Q C E) :
  cyclicQuadrilateral [A, B, C, D] :=
  sorry

end cyclic_ABCD_l483_483421


namespace coordinates_of_P_respect_to_symmetric_y_axis_l483_483372

-- Definition of points in the Cartesian coordinate system
structure Point where
  x : ℝ
  y : ℝ

def symmetric_x_axis (p : Point) : Point :=
  { p with y := -p.y }

def symmetric_y_axis (p : Point) : Point :=
  { p with x := -p.x }

-- The given condition
def P_with_respect_to_symmetric_x_axis := Point.mk (-1) 2

-- The problem statement
theorem coordinates_of_P_respect_to_symmetric_y_axis :
    symmetric_y_axis (symmetric_x_axis P_with_respect_to_symmetric_x_axis) = Point.mk 1 (-2) :=
by
  sorry

end coordinates_of_P_respect_to_symmetric_y_axis_l483_483372


namespace min_nSn_value_l483_483490

noncomputable def Sn (n : ℕ) : ℚ := (1/3 : ℚ) * n^2 - (10/3 : ℚ) * n

theorem min_nSn_value :
  (Sn 10 = 0) ∧ (Sn 15 = 25) ∧ (∃ n : ℕ, ∃ (a : ℚ), a = n * Sn n ∧ a = -49) :=
by {
  split,
  { -- Sn 10 = 0
    calc Sn 10 = (1/3) * 10^2 - (10/3) * 10 : rfl
              ... = (1/3) * 100 - (10/3) * 10 : by simp
              ... = 100 / 3 - 100 / 3 : by norm_num
              ... = 0 : by norm_num },
  split,
  { -- Sn 15 = 25
    calc Sn 15 = (1/3) * 15^2 - (10/3) * 15 : rfl
              ... = (1/3) * 225 - (10/3) * 15 : by simp
              ... = 225 / 3 - 150 / 3 : by norm_num
              ... = 75 - 50 : by norm_num
              ... = 25 : by norm_num },
  { -- ∃ n : ℕ, ∃ (a : ℚ), a = n * Sn n ∧ a = -49
    use 7,
    use (7 * Sn 7),
    split,
    { refl },
    -- (7 * Sn 7) = -49
    calc 7 * Sn 7 = 7 * ((1/3) * 7^2 - (10/3) * 7) : rfl
                 ... = 7 * (49/3 - 70/3) : by simp
                 ... = 7 * (-21/3) : by norm_num
                 ... = 7 * (-7) : by norm_num
                 ... = -49 : by norm_num }
} 

end min_nSn_value_l483_483490


namespace task_completion_time_l483_483025

variable (x : Real) (y : Real)

theorem task_completion_time :
  (1 / 16) * y + (1 / 12) * x = 1 ∧ y + 5 = 8 → x = 3 ∧ y = 3 :=
  by {
    sorry 
  }

end task_completion_time_l483_483025


namespace unique_two_digit_perfect_square_divisible_by_5_l483_483709

-- Define the conditions
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

def two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

-- The statement to prove: there is exactly 1 two-digit perfect square that is divisible by 5
theorem unique_two_digit_perfect_square_divisible_by_5 :
  ∃! n : ℕ, is_perfect_square n ∧ two_digit n ∧ divisible_by_5 n :=
sorry

end unique_two_digit_perfect_square_divisible_by_5_l483_483709


namespace sum_of_digits_3plus4_pow_15_l483_483917

theorem sum_of_digits_3plus4_pow_15 : 
  let n := (3 + 4) ^ 15,
      ones_digit := n % 10,
      tens_digit := (n / 10) % 10 in
  ones_digit + tens_digit = 7 := by sorry

end sum_of_digits_3plus4_pow_15_l483_483917


namespace circle_intersects_y_axis_at_one_l483_483668

theorem circle_intersects_y_axis_at_one (a : ℝ) :
  ∃ b : ℝ, y = x^2 - a * x - 3 ∧
           (circle_passing_through A B (0, -3)).intersect_y_axis = (0, b) ∧
           b = 1 :=
by sorry

end circle_intersects_y_axis_at_one_l483_483668


namespace integrate_sin_plus_one_l483_483100

open Real

theorem integrate_sin_plus_one :
  ∫ x in -1..1, (sin x + 1) = 2 :=
by 
  sorry

end integrate_sin_plus_one_l483_483100


namespace rationalize_denominator_l483_483452

theorem rationalize_denominator : 
  let x := (1 : ℝ)
  let y := (3 : ℝ)
  let z := real.cbrt 3
  let w := real.cbrt 27
  (w = 3) →
  x / (z + w) = real.cbrt (9) / (3 * (real.cbrt (9) + 1)) := 
by
  intros _ h
  rw [h]
  sorry

end rationalize_denominator_l483_483452


namespace calculate_expression_l483_483202

theorem calculate_expression :
  (2^3 * 3 * 5) + (18 / 2) = 129 := by
  -- Proof skipped
  sorry

end calculate_expression_l483_483202


namespace cats_joined_l483_483966

theorem cats_joined (c : ℕ) (h : 1 + c + 2 * c + 6 * c = 37) : c = 4 :=
sorry

end cats_joined_l483_483966


namespace meeting_time_l483_483134

theorem meeting_time (track_length : ℕ) (speed_A_kmph speed_B_kmph : ℕ) (h_track_length : track_length = 600) (h_speed_A : speed_A_kmph = 30) (h_speed_B : speed_B_kmph = 60) :
  ∃ time_meeting : ℝ, time_meeting = (1.2 : ℝ) ∧
    let speed_A := (speed_A_kmph * 1000) / 60 in
    let speed_B := (speed_B_kmph * 1000) / 60 in
    let time_A := track_length / speed_A in
    let time_B := track_length / speed_B in
    ∃ k_A k_B : ℕ, time_meeting = k_A * time_A ∧ time_meeting = k_B * time_B :=
begin
  sorry
end

end meeting_time_l483_483134


namespace log_expression_equiv_l483_483934

noncomputable def log_simplified (a b : ℝ) : ℝ :=
  2 * real.sqrt (real.log a b) * (
    (real.sqrt ((1/4) * (2 + real.log a b + 1 / real.log a b))) - 
    (real.sqrt ((1/4) * (real.log a b - 1 + 1 / real.log a b - 1)))
  )

theorem log_expression_equiv (a b : ℝ) (h1 : 1 < a) (h2 : 1 < b) :
  log_simplified a b = 
  if 1 < a ∧ a ≤ b then 2 else 2 * real.log a b :=
by sorry

end log_expression_equiv_l483_483934


namespace rationalize_denominator_l483_483454

theorem rationalize_denominator : 
  let x := (1 : ℝ)
  let y := (3 : ℝ)
  let z := real.cbrt 3
  let w := real.cbrt 27
  (w = 3) →
  x / (z + w) = real.cbrt (9) / (3 * (real.cbrt (9) + 1)) := 
by
  intros _ h
  rw [h]
  sorry

end rationalize_denominator_l483_483454


namespace order_rectangles_l483_483373

noncomputable def rectangle := { (x1: ℝ) // x1 ≥ 0 } × { (y1: ℝ) // y1 ≥ 0 } × 
                               { (x2: ℝ) // x2 ≥ x1 } × { (y2: ℝ) // y2 ≥ y1 }

def is_below (a b : rectangle) : Prop :=
    ∃ g : ℝ, g < a.2.2.2 ∧ g > b.2.2.2

def is_right (a b : rectangle) : Prop :=
    ∃ g : ℝ, g > a.1.2 ∧ g < b.1.2

def preferable (a b : rectangle) : Prop :=
    is_right a b ∨ is_below a b

theorem order_rectangles (n : ℕ) (rects : fin n → rectangle) :
    ∃ (R : fin n → rectangle), ∀ i j : fin n, i < j → preferable (R j) (R i) :=
sorry

end order_rectangles_l483_483373


namespace irrational_sqrt3_l483_483513

theorem irrational_sqrt3 : 
  (∀ r : ℚ, r ≠ sqrt 3) ∧ 
  (∃ r : ℚ, r = 2 / 3) ∧ 
  (∃ r : ℚ, r = 1414 / 1000) ∧ 
  (∃ r : ℚ, r = sqrt 9) := 
by
  sorry

end irrational_sqrt3_l483_483513


namespace cosine_of_angle_of_triangle_l483_483302

theorem cosine_of_angle_of_triangle (P : ℝ × ℝ)
  (hP : P.1 ^ 2 / 25 + P.2 ^ 2 / 9 = 1)
  (area_triangle : 1/2 * ((P.1 + 4) * P.2 - (P.1 - 4) * P.2) = 6 * sqrt 3) :
  let F1 := (-4, 0), F2 := (4, 0) in
  ∃ (θ : ℝ), cos θ = 1/2 :=
by
  sorry

end cosine_of_angle_of_triangle_l483_483302


namespace maximize_CD_l483_483082

theorem maximize_CD (AB_length : ℝ) (h_AB : AB_length = 2)
                    (t : ℝ) (h_t : 0 ≤ t ∧ t < 1) :
  let r := (1 - t^2) / 4,
      x := (5 * t - t^3) / 4,
      CD := (t * (1 - t^2)) / 4 in
  (CD ≤ (1 / sqrt 3 * (1 - (1 / sqrt 3) ^ 2)) / 4) :=
begin
  sorry
end

end maximize_CD_l483_483082


namespace distance_from_center_of_tetrahedron_l483_483563

theorem distance_from_center_of_tetrahedron
  (T : Type) [RegularTetrahedron T]
  (volume_T : volume T = 1)
  (P : T)
  (scaled_T : T)
  (scale_factor : is_scaled_from P T scaled_T 2)
  (shared_volume : volume_overlap T scaled_T = 1 / 8) :
  distance P (centroid T) = (5 / 6) * (6 * sqrt 3)^(1 / 6) ∨ distance P (centroid T) = (6 * sqrt 3)^(1 / 6) / 2 :=
sorry

end distance_from_center_of_tetrahedron_l483_483563


namespace sum_of_array_l483_483155

-- Define the sequence value in terms of the row and column indices
def array_value (r c : Nat) : ℚ :=
  1 / ((8^r) * (4^c))

-- Define the infinite summation over all values in the array
def array_sum : ℚ :=
  ∑' r, ∑' c, array_value r c

theorem sum_of_array : array_sum = 32 / 21 := by
  sorry

end sum_of_array_l483_483155


namespace nina_age_l483_483428

theorem nina_age : ∀ (M L A N : ℕ), 
  (M = L - 5) → 
  (L = A + 6) → 
  (N = A + 2) → 
  (M = 16) → 
  N = 17 :=
by
  intros M L A N h1 h2 h3 h4
  sorry

end nina_age_l483_483428


namespace circumcircles_intersection_l483_483773

-- Define the given conditions using structures and predicates
structure Triangle (α : Type) [LinearOrder α] :=
(A B C : α)

structure ExternalTriangle (α : Type) [LinearOrder α] :=
(A' B' C' : α)

noncomputable def angle_sum_is_multiple_of_180 {α : Type} [LinearOrder α] (T : Triangle α) (ET : ExternalTriangle α) : Prop :=
  -- Assume we have a function to calculate the angles at vertices and their sum
  (angle T.A' T.B' T.C') + (angle T.B' T.C' T.A') + (angle T.C' T.A' T.B') % 180 = 0

-- Now we state the theorem
theorem circumcircles_intersection {α : Type} [LinearOrder α] (T : Triangle α) (ET : ExternalTriangle α)
  (h : angle_sum_is_multiple_of_180 T ET) :
  ∃ P : α, OnCircumcircle T.A' T.B' T.C' P ∧ OnCircumcircle T.B' T.C' T.A' P ∧ OnCircumcircle T.C' T.A' T.B' P := 
sorry

end circumcircles_intersection_l483_483773


namespace symmetric_lines_concur_on_circumcircle_l483_483171

open EuclideanGeometry

-- Definitions of the triangle and orthocenter condition
noncomputable def triangle (A B C : Point) := acute_angle (A B C)
noncomputable def orthocenter (A B C H : Point) := ∀ (AD BE CF : Line), is_altitude AD ∧ is_altitude BE ∧ is_altitude CF ∧ AD.intersect BE.intersect CF = some H

-- The given line passing through orthocenter
noncomputable def line_through_orthocenter (H : Point) (l : Line) := H ∈ l

-- Definitions of symmetric image conditions
noncomputable def symmetric_image (l : Line) (BC : Line) (la : Line) := reflection_over BC l = la
noncomputable def reflection_over (l1 l2: Line) : Line := sorry -- defines the reflection of a line over another

-- Main theorem statement
theorem symmetric_lines_concur_on_circumcircle
(triangle_ABC : triangle A B C)
(orthocenter_H : orthocenter A B C H)
(line_l : Line)
(line_through_H : line_through_orthocenter H line_l)
(symmetric_la : symmetric_image line_l (BC_line A B C) la)
(symmetric_lb : symmetric_image line_l (CA_line A B C) lb)
(symmetric_lc : symmetric_image line_l (AB_line A B C) lc): 
∃ O, O ∈ circumcircle A B C ∧ la ∩ lb ∩ lc = some O :=
sorry

end symmetric_lines_concur_on_circumcircle_l483_483171


namespace equivalent_expression_l483_483506

theorem equivalent_expression :
  (5+3) * (5^2 + 3^2) * (5^4 + 3^4) * (5^8 + 3^8) * (5^16 + 3^16) * 
  (5^32 + 3^32) * (5^64 + 3^64) = 5^128 - 3^128 := 
  sorry

end equivalent_expression_l483_483506


namespace minimum_actions_distinct_macaroni_l483_483951

-- Define the initial state
def initial_macaroni (n : Nat) : Nat := 100

-- Define what it means for an action to take place
-- An action is a function that takes a state and returns a new state
def action (s : Nat → Nat) (i : Nat) (recipients : List Nat) : Nat → Nat :=
  λ n =>
    if n = i then s n - 1
    else if n ∈ recipients then s n + 1
    else s n

-- Define the goal
theorem minimum_actions_distinct_macaroni :
  ∃ (actions : List (Nat → Nat → Nat → Nat)), -- list of actions
   -- apply each action in sequence to the initial state
   (List.foldl (λ s a => a s) initial_macaroni actions) ≠ 
   initial_macaroni ∧
   (∀ s, (List.foldl (λ s a => a s) initial_macaroni actions) s ≠ 
    (List.foldl (λ s a => a s) initial_macaroni actions) (s + 1)) ∧
   actions.length = 50 :=
sorry

end minimum_actions_distinct_macaroni_l483_483951


namespace arithmetic_sequence_sum_l483_483662

-- Definitions used in the conditions
variable (a : ℕ → ℕ)
variable (n : ℕ)
variable (a_seq : Prop)
-- Declaring the arithmetic sequence condition
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop := ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

noncomputable def a_5_is_2 : Prop := a 5 = 2

-- The statement we need to prove
theorem arithmetic_sequence_sum (a : ℕ → ℕ) (h_arith_seq : is_arithmetic_sequence a) (h_a5 : a 5 = 2) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 2 * 9 := by
sorry

end arithmetic_sequence_sum_l483_483662


namespace sin_double_angle_l483_483671

theorem sin_double_angle (α : ℝ) (h : cos α = - (√3) / 2 ∧ sin α = 1 / 2) : sin (2 * α) = - (√3) / 2 := 
by 
  sorry

end sin_double_angle_l483_483671


namespace smallest_arithmetic_mean_divisible_by_1111_l483_483852

/-- 
Given the product of nine consecutive natural numbers is divisible by 1111, 
prove that the smallest possible value of the arithmetic mean of these nine numbers is 97.
-/
theorem smallest_arithmetic_mean_divisible_by_1111 :
  ∃ n : ℕ, (∀ k : ℕ, k = n →  (∏ i in finset.range 9, k + i) % 1111 = 0) 
  ∧ (n ≥ 93) ∧ (n + 4 = 97) :=
sorry

end smallest_arithmetic_mean_divisible_by_1111_l483_483852


namespace michael_time_proof_l483_483431

-- Definitions based on the given conditions
def rate_father : ℝ := 4 -- rate in feet per hour
def time_father : ℝ := 400 -- time in hours
def depth_father := rate_father * time_father -- depth of father's hole
def depth_michael := 2 * depth_father - 400 -- depth of Michael's hole
def rate_michael : ℝ := rate_father -- same rate for Michael

-- Goal: Time taken by Michael to dig the hole
def time_michael := depth_michael / rate_michael

theorem michael_time_proof : time_michael = 700 :=
by
  unfold time_michael depth_michael rate_michael depth_father rate_father time_father
  simp
  norm_num
  sorry -- Proof steps omitted

end michael_time_proof_l483_483431


namespace rhombus_area_l483_483313

theorem rhombus_area (d1 d2 : ℝ) (h_d1 : d1 = 10) (h_d2 : d2 = 8) : 
  (1 / 2) * d1 * d2 = 40 :=
by
  rw [h_d1, h_d2]
  norm_num
  sorry

end rhombus_area_l483_483313


namespace smallest_arithmetic_mean_divisible_by_1111_l483_483833

noncomputable def nine_consecutive_numbers {n : ℕ} : list ℕ :=
  [n, n+1, n+2, n+3, n+4, n+5, n+6, n+7, n+8]

def divisible_by (x y : ℕ) : Prop := ∃ k : ℕ, x = k * y

def arithmetic_mean {l : list ℕ} (h_len : l.length = 9) : ℚ :=
  (l.sum : ℚ) / 9

theorem smallest_arithmetic_mean_divisible_by_1111 :
  ∃ n : ℕ, 
  divisible_by ((nine_consecutive_numbers n).prod) 1111 ∧ 
  arithmetic_mean (by simp [nine_consecutive_numbers_len]) = 97 :=
sorry

end smallest_arithmetic_mean_divisible_by_1111_l483_483833


namespace most_likely_event_is_C_l483_483549

open Classical

noncomputable def total_events : ℕ := 6 * 6

noncomputable def P_A : ℚ := 7 / 36
noncomputable def P_B : ℚ := 18 / 36
noncomputable def P_C : ℚ := 1
noncomputable def P_D : ℚ := 0

theorem most_likely_event_is_C :
  P_C > P_A ∧ P_C > P_B ∧ P_C > P_D := by
  sorry

end most_likely_event_is_C_l483_483549


namespace sum_first_2500_terms_eq_zero_l483_483986

theorem sum_first_2500_terms_eq_zero
  (b : ℕ → ℤ)
  (h1 : ∀ n ≥ 3, b n = b (n - 1) - b (n - 2))
  (h2 : (Finset.range 1800).sum b = 2023)
  (h3 : (Finset.range 2023).sum b = 1800) :
  (Finset.range 2500).sum b = 0 :=
sorry

end sum_first_2500_terms_eq_zero_l483_483986


namespace monkey_reaches_top_in_19_minutes_l483_483936

theorem monkey_reaches_top_in_19_minutes (pole_height : ℕ) (ascend_first_min : ℕ) (slip_every_alternate_min : ℕ) 
    (total_minutes : ℕ) (net_gain_two_min : ℕ) : 
    pole_height = 10 ∧ ascend_first_min = 2 ∧ slip_every_alternate_min = 1 ∧ net_gain_two_min = 1 ∧ total_minutes = 19 →
    (net_gain_two_min * (total_minutes - 1) / 2 + ascend_first_min = pole_height) := 
by
    intros
    sorry

end monkey_reaches_top_in_19_minutes_l483_483936


namespace find_matrix_l483_483274

theorem find_matrix (M : Matrix (Fin 2) (Fin 2) ℝ)
  (h : ∀ v : Matrix (Fin 2) (Fin 1) ℝ, M.mul_vec v = (-7 : ℝ) • v) :
  M = !![-7, 0; 0, -7] :=
by
  sorry

end find_matrix_l483_483274


namespace local_maximum_at_1_l483_483766

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - x^2

theorem local_maximum_at_1 : ∃ x : ℝ, x = 1 ∧ ∀ ε > 0, ∀ y : ℝ, (abs (y - x) < ε ∧ y ≠ x) → f y < f x :=
by
  let x := 1
  have hx : x = 1 := rfl
  have local_max_condition : ∀ ε > 0, ∀ y : ℝ, (abs (y - x) < ε ∧ y ≠ x) → f y < f x := 
    sorry
  use x
  exact ⟨hx, local_max_condition⟩

end local_maximum_at_1_l483_483766


namespace circumcircles_tangent_l483_483029

theorem circumcircles_tangent (A B C D O S : Point)
  (h1 : Trapezoid ABCD)
  (h2 : AD ∥ BC)
  (h3 : O = intersection BD AC)
  (h4 : S = second_intersection (circumcircle A O B) (circumcircle D O C)) :
  tangent (circumcircle A S D) (circumcircle B S C) :=
sorry

end circumcircles_tangent_l483_483029


namespace mathNotRebusOrLogic_l483_483990

open Set

variables (B : Type) [Fintype B]
variables (R M L : Set B) -- Sets of brainiacs who like each type of teaser
variables (brainiacsSurveyed : Set B) (neither : Set B)

-- Given conditions
def totalSurveyed : Fintype.card brainiacsSurveyed = 500 := sorry
def neitherTeasers : Fintype.card neither = 20 := sorry
def twiceAsManyRebus : Fintype.card R = 2 * Fintype.card M := sorry
def equalLogicMath : Fintype.card L = Fintype.card M := sorry
def bothRebusMath : Fintype.card (R ∩ M) = 72 := sorry
def bothRebusLogic : Fintype.card (R ∩ L) = 40 := sorry
def bothMathLogic : Fintype.card (M ∩ L) = 36 := sorry
def allThree : Fintype.card (R ∩ M ∩ L) = 10 := sorry

-- Prove the number of brainiacs who like math teasers but not rebus or logic teasers is 54
theorem mathNotRebusOrLogic :
  Fintype.card M - Fintype.card (R ∩ M) - Fintype.card (M ∩ L) + Fintype.card (R ∩ M ∩ L) = 54 := sorry

end mathNotRebusOrLogic_l483_483990


namespace probability_one_unit_apart_l483_483078

/-
We define the points on the 3x3 grid as a Finset.
We will define the total number of points and the property that two points are one unit apart.
-/

/-- A 3x3 grid with 10 points spaced around at intervals of one unit. -/
def points_on_grid : Finset (ℕ × ℕ) := { -- Here we define the set of 10 points
  (0, 0), (0, 2), (2, 0), (2, 2), -- Corners
  (1, 0), (1, 2), (0, 1), (2, 1), -- Mid-points of sides
  (1, 1) -- Center
}

-- Function to determine if two points are one unit apart
def one_unit_apart (a b : ℕ × ℕ) : Prop :=
  (abs (a.1 - b.1) + abs (a.2 - b.2) = 1)

/-- The probability calculation proof statement -/
theorem probability_one_unit_apart :
  (∃! (p1 p2 : (ℕ × ℕ)), ((p1 ∈ points_on_grid) ∧ (p2 ∈ points_on_grid) ∧ (one_unit_apart p1 p2))) →
  ∃! (ratio : ℚ), ratio = 16 / 45 :=
by
  sorry

end probability_one_unit_apart_l483_483078


namespace probability_both_asian_selected_probability_A1_but_not_B1_selected_l483_483186

noncomputable def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_both_asian_selected (A1 A2 A3 B1 B2 B3 : Prop) :
  let total_ways := choose 6 2
  let asian_ways := choose 3 2
  asian_ways / total_ways = 1 / 5 := by
  let total_ways := choose 6 2
  let asian_ways := choose 3 2
  sorry

theorem probability_A1_but_not_B1_selected (A1 A2 A3 B1 B2 B3 : Prop) :
  let total_ways := 9
  let valid_ways := 2
  valid_ways / total_ways = 2 / 9 := by
  let total_ways := 9
  let valid_ways := 2
  sorry

end probability_both_asian_selected_probability_A1_but_not_B1_selected_l483_483186


namespace integer_root_of_quadratic_eq_l483_483412

theorem integer_root_of_quadratic_eq (m : ℤ) (hm : ∃ x : ℤ, m * x^2 + 2 * (m - 5) * x + (m - 4) = 0) : m = -4 ∨ m = 4 ∨ m = -16 :=
sorry

end integer_root_of_quadratic_eq_l483_483412


namespace mixed_periodic_fraction_l483_483069

theorem mixed_periodic_fraction (n : ℤ) : 
  let s := (1 / (n : ℚ)) + (1 / ((n + 1) : ℚ)) + (1 / ((n + 2) : ℚ))
  in ∃ d : ℚ, ∃ f : list ℚ, ∃ r : list ℚ, rational.is_mixed_periodic_representation s d f r :=
by
  sorry

end mixed_periodic_fraction_l483_483069


namespace opposite_face_x_is_C_l483_483481

-- Defining the context of the problem
inductive Face
| x
| A
| B
| C
| D
| E

open Face

-- The proof problem
theorem opposite_face_x_is_C: (∃ cube : List Face, cube.length = 6 ∧ 
                                cube.nodup ∧
                                Face.x ∈ cube ∧ 
                                Face.A ∈ cube ∧ 
                                Face.B ∈ cube ∧ 
                                Face.C ∈ cube ∧ 
                                Face.D ∈ cube ∧ 
                                Face.E ∈ cube) 
                                → (∀ f ∈ cube, f ≠ Face.x → f = Face.C) :=
sorry

end opposite_face_x_is_C_l483_483481


namespace denomination_of_remaining_coins_l483_483495

/-
There are 324 coins total.
The total value of the coins is Rs. 70.
There are 220 coins of 20 paise each.
Find the denomination of the remaining coins.
-/

def total_coins := 324
def total_value := 7000 -- Rs. 70 converted into paise
def num_20_paise_coins := 220
def value_20_paise_coin := 20
  
theorem denomination_of_remaining_coins :
  let total_remaining_value := total_value - (num_20_paise_coins * value_20_paise_coin)
  let num_remaining_coins := total_coins - num_20_paise_coins
  num_remaining_coins > 0 →
  total_remaining_value / num_remaining_coins = 25 :=
by
  sorry

end denomination_of_remaining_coins_l483_483495


namespace length_PF_l483_483243

-- Define the parabola y^2 = 8x
def parabola (y x : ℝ) : Prop :=
  y^2 = 8 * x

-- Focus of the parabola
def focus : (ℝ × ℝ) := (2, 0)

-- Directrix of the parabola
def directrix (x : ℝ) : Prop :=
  x = -2

-- Point A is on the directrix and on line AF with slope -sqrt(3)
def pointA (x : ℝ) (y : ℝ) : Prop :=
  x = -2 ∧ y = -sqrt 3 * (x - 2)

-- Point P is on the parabola and on the vertical line through A
def pointP (x y : ℝ) : Prop :=
  parabola y x ∧ x = 6

-- Distance calculation between P and F
def distancePF (P F : ℝ × ℝ) : ℝ :=
  abs (P.1 - F.1)

theorem length_PF :
  ∀ (P F A : ℝ × ℝ),
    parabola (P.2) (P.1) ∧
    (A.1 = -2) ∧ (A.2 = 4 * sqrt 3) ∧
    (P.1 = 6) ∧
    (P.2 = A.2) ∧
    F = (2, 0) →
    distancePF P F = 8 := by
  sorry

end length_PF_l483_483243


namespace cos_four_theta_proof_l483_483347

noncomputable def cos_four_theta (θ : ℝ) : ℝ :=
  cos (4 * θ)

theorem cos_four_theta_proof : 
  ∀ (θ : ℝ), (complex.exp (θ * complex.I) = (1 + complex.I * real.sqrt 7) / 4) → cos_four_theta θ = 1 / 32 :=
by
  intros θ h
  sorry

end cos_four_theta_proof_l483_483347


namespace cylinder_volume_increase_l483_483955

noncomputable def cylinder_initial_volume (h r : ℝ) : ℝ :=
  Real.pi * r^2 * h

noncomputable def frustum_volume (m r R : ℝ) : ℝ :=
  (Real.pi * m / 3) * (R^2 + R * r + r^2)

noncomputable def percentage_increase (V_initial V_new : ℝ) : ℝ :=
  ((V_new - V_initial) / V_initial) * 100

theorem cylinder_volume_increase :
  let h := 20.0  -- height of the cylinder in cm
  let d := 10.0  -- diameter of the cylinder in cm
  let r := d / 2  -- radius of the base of the cylinder in cm
  let slant_h_initial := 10.0  -- initial slant height in cm
  let slant_h_new := 10.05  -- new slant height in cm after increase by 1 mm (0.1 cm)
  let a_diff := slant_h_new - slant_h_initial  -- the difference in slant heights
  let m := h / 2  -- height of each frustum in cm
  let r_new := Real.sqrt (slant_h_new ^ 2 - m ^ 2)\ / 2 + r -- calculate R
 
  let V_initial := cylinder_initial_volume h r
  let V_new := 2 * frustum_volume m r r_new
  percentage_increase V_initial V_new ≈ 21.36 :=
by
  sorry

end cylinder_volume_increase_l483_483955


namespace value_of_P_2017_l483_483053

theorem value_of_P_2017 (a b c : ℝ) (h_distinct: a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ 0 < a ∧ 0 < b ∧ 0 < c)
    (p : ℝ → ℝ) :
    (∀ x, p x = (c * (x - a) * (x - b) / ((c - a) * (c - b))) + (a * (x - b) * (x - c) / ((a - b) * (a - c))) + (b * (x - c) * (x - a) / ((b - c) * (b - a))) + 1) →
    p 2017 = 2 :=
sorry

end value_of_P_2017_l483_483053


namespace horizontal_asymptote_l483_483599

noncomputable def rational_function : ℝ → ℝ :=
  λ x, (15 * x^4 + 5 * x^3 + 7 * x^2 + 6 * x + 2) / 
       (5 * x^4 + 3 * x^3 + 10 * x^2 + 4 * x + 1)

theorem horizontal_asymptote :
  (∃ L : ℝ, ∀ ε > 0, ∃ M, ∀ x > M, |rational_function x - L| < ε) ∧ L = 3 :=
  sorry

end horizontal_asymptote_l483_483599


namespace find_c_m_l483_483817

-- Defining the quadratic function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x ^ 2 + b * x + c

-- Given points
def p1 := (quadratic a b c (-3)) = 5 / 2
def p2 := (quadratic a b c (-2)) = 4
def p3 := (quadratic a b c (-1)) = 9 / 2
def p4 := (quadratic a b c 0) = 4
def p5 := ∃ m : ℝ, (quadratic a b c 1) = m

-- Prove that c = 4 and m = 5 / 2
theorem find_c_m (a b : ℝ) (h : a ≠ 0) : 
  p1 → p2 → p3 → p4 → p5 → (c = 4 ∧ ∃ m, m = 5 / 2) :=
by
  intros h1 h2 h3 h4 h5
  have c_eq := h4
  show c = 4
  sorry
  exists 5 / 2
  sorry

end find_c_m_l483_483817


namespace problem_solution_l483_483758

theorem problem_solution (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b + c = 0) (h5 : a^2 + b^2 + c^2 = 3) :
  (a^3 + b^3 + c^3) / (a * b * c) = 3 := 
  sorry

end problem_solution_l483_483758


namespace mean_and_median_temperatures_l483_483079

theorem mean_and_median_temperatures (t : List ℤ) 
  (h : t = [-10, -5, -5, -7, 0, 7, 5]) :
  (t.sum / t.length = -15 / 7) ∧ (t.nth ((t.length - 1) / 2) = some -5) :=
by
  sorry

end mean_and_median_temperatures_l483_483079


namespace triangle_YAZ_is_right_l483_483390

-- Define the points and lengths in the problem
variables (X Y Z W A : Type)
variables [EuclideanSpace ℝ X] [EuclideanSpace ℝ Y] [EuclideanSpace ℝ Z]
variables [EuclideanSpace ℝ W] [EuclideanSpace ℝ A]

-- Define the geometric relationships and conditions
variables (YZ XZ : ℝ)
variables (hYZ_XZ : YZ = 2 * XZ)
variables (angleZXW angleZYX : ℝ)
variables (h_angleZXW_eq_angleZYX : angleZXW = angleZYX)
variables (XW : ℝ)
variables (angle_bisector : ℝ)

-- Assumption of intersection condition
variable (h_intersect_angle_bisector : True) -- Simplified representation

theorem triangle_YAZ_is_right :
  angle (Y :: A :: Z) = 90 :=
sorry

end triangle_YAZ_is_right_l483_483390


namespace days_b_worked_l483_483540

theorem days_b_worked (A_days B_days A_remaining_days : ℝ) (A_work_rate B_work_rate total_work : ℝ)
  (hA_rate : A_work_rate = 1 / A_days)
  (hB_rate : B_work_rate = 1 / B_days)
  (hA_days : A_days = 9)
  (hB_days : B_days = 15)
  (hA_remaining : A_remaining_days = 3)
  (h_total_work : ∀ x : ℝ, (x * B_work_rate + A_remaining_days * A_work_rate = total_work)) :
  ∃ x : ℝ, x = 10 :=
by
  sorry

end days_b_worked_l483_483540


namespace angle_between_DO_and_CE_is_l483_483145

variable (a : ℝ) -- side length of the tetrahedron

def point_S := (0, 0, 0 : ℝ × ℝ × ℝ)
def point_A := (a, 0, 0)
def point_B := (a/2, (a * Real.sqrt 3)/2, 0)
def point_C := (a/2, (a * Real.sqrt 3)/6, (a * Real.sqrt 6)/3)

-- Midpoint D of edge SA
def point_D := ((0 + a) / 2, (0 + 0) / 2, (0 + 0) / 2)

-- Centroid O of face ABC
def point_O := ((a + a/2 + a/2)/3, ((0 + (a * Real.sqrt 3) / 2 + (a * Real.sqrt 3) / 6) / 3),
                 (0 + 0 + (a * Real.sqrt 6) / 3) / 3)

-- Midpoint E of edge SB
def point_E := ((0 + a/2) / 2, (0 + (a * Real.sqrt 3) / 2) / 2, 0)

-- Vectors DO and CE
def vector_DO := ((2 * a) / 3 - a / 2, (a * Real.sqrt 3) / 3, (a * Real.sqrt 6) / 9)
def vector_CE := (a/4 - a/2, (a * Real.sqrt 3) / 4 - (a * Real.sqrt 3) / 6, 0 - (a * Real.sqrt 6) / 3)

-- Dot product of vectors DO and CE
def dot_product_DO_CE : ℝ :=
  let v_DO := vector_DO a
  let v_CE := vector_C_E a
  v_DO.1 * v_CE.1 + v_DO.2 * v_CE.2 + v_DO.3 * v_CE.3

-- Magnitudes of vectors DO and CE
def magnitude_DO : ℝ :=
  Real.sqrt (((vector_DO a).1 ^ 2) + ((vector_DO a).2 ^ 2) + ((vector_DO a).3 ^ 2))

def magnitude_CE : ℝ :=
  Real.sqrt (((vector_C_E a).1 ^ 2) + ((vector_C_E a).2 ^ 2) + ((vector_C_E a).3 ^ 2))

-- Angle θ between DO and CE
def angle_between_DO_CE : ℝ :=
  let dot := dot_product_DO_CE a
  let mag_DO := magnitude_DO a
  let mag_CE := magnitude_CE a
  Real.arccos (dot / (mag_DO * mag_CE))

theorem angle_between_DO_and_CE_is
  (a : ℝ) : angle_between_DO_CE a = Real.arccos(1 / 3) :=
  sorry

end angle_between_DO_and_CE_is_l483_483145


namespace acute_angle_at_10_50_l483_483901

def degrees_per_minute : ℕ := 6
def degrees_per_hour : ℕ := 30
def time_in_minutes (h m: ℕ) := h * 60 + m
def angle (h m: ℕ) : ℕ := 
  let minute_angle := m * degrees_per_minute
  let hour_angle := h * degrees_per_hour + (m * degrees_per_hour / 60)
  let raw_angle := abs (hour_angle - minute_angle)
  if raw_angle > 180 then 360 - raw_angle else raw_angle

theorem acute_angle_at_10_50 : angle 10 50 = 25 := 
by 
  sorry

end acute_angle_at_10_50_l483_483901


namespace expression_value_l483_483207

noncomputable def givenExpression : ℝ :=
  -2^2 + Real.sqrt 8 - 3 + 1/3

theorem expression_value : givenExpression = -20/3 + 2 * Real.sqrt 2 := 
by
  sorry

end expression_value_l483_483207


namespace min_area_ratio_of_equilateral_and_right_triangle_l483_483679

noncomputable def min_area_ratio (ABC : Triangle) (DEF : Triangle) :=
  (∃ (A B C : Point) (D E F : Point),
    Equilateral ABC ∧
    Right DEF ∧
    (on_side ABC A B C D E F) ∧
    (angle DEF = π / 2) ∧
    (angle EDF = π / 6) ∧
    min (S DEF / S ABC) = (3 / 14))

-- This statement asserts that given the conditions, the minimum value of the ratio of areas is 3/14
theorem min_area_ratio_of_equilateral_and_right_triangle (ABC DEF : Triangle) :
  min_area_ratio ABC DEF :=
sorry

end min_area_ratio_of_equilateral_and_right_triangle_l483_483679


namespace probability_f_geq_16_l483_483329

noncomputable def f (x : ℝ) : ℝ := 2^x

theorem probability_f_geq_16 (x0 : ℝ) (h0 : x0 ∈ Set.Icc 0 10) : 
  let prob := (10 - 4) / (10 - 0) in
  ∃ prob = 0.6 :=
by
  sorry

end probability_f_geq_16_l483_483329


namespace matrix_eigenvalue_neg7_l483_483272

theorem matrix_eigenvalue_neg7 (M : Matrix (Fin 2) (Fin 2) ℝ) :
  (∀ (v : Fin 2 → ℝ), M.mulVec v = -7 • v) →
  M = !![-7, 0; 0, -7] :=
by
  intro h
  -- proof goes here
  sorry

end matrix_eigenvalue_neg7_l483_483272


namespace composite_numbers_form_l483_483623

def is_composite (n : ℕ) : Prop := 
  ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ n = a * b

def f (n : ℕ) : ℕ := 
  let divisors := List.filter (λ d, n % d = 0) (List.range (n+1))
  divisors.nthLe 0 (by simp) + divisors.nthLe 1 (by simp) + divisors.nthLe 2 (by simp)

def g (n : ℕ) : ℕ := 
  let divisors := List.filter (λ d, n % d = 0) (List.range (n+1))
  divisors.nthLe (divisors.length - 1) (by simp) + divisors.nthLe (divisors.length - 2) (by simp)

theorem composite_numbers_form (n k : ℕ) (β : ℕ) (hk : 0 < k ∧ n = 2^(β + 2) * 3^β) :
  is_composite n ∧ g(n) = f(n)^k → ∃ β : ℕ, n = 2^(β + 2) * 3^β :=
begin
  sorry
end

end composite_numbers_form_l483_483623


namespace max_added_value_l483_483173

section
variables {t : ℝ} (h_t : 0 < t ∧ t ≤ 2)
noncomputable def f (x : ℝ) : ℝ := 4 * (1 - x) * x ^ 2

theorem max_added_value :
  ∃ (x_max y_max : ℝ), 
  (∀ x ∈ Ioo 0 (2 * t / (2 * t + 1)), f x ≤ y_max) ∧ 
  ((1 ≤ t ∧ t ≤ 2) → (x_max = 2 / 3 ∧ y_max = 16 / 27)) ∧
  ((0 < t ∧ t < 1) → (x_max = 2 * t / (2 * t + 1) ∧ y_max = 16 * t^2 / (2 * t + 1)^3)) :=
sorry
end

end max_added_value_l483_483173


namespace area_of_triangle_l483_483568

def line1 (x : ℝ) : ℝ := 2 * x + 1
def line2 (x : ℝ) : ℝ := -(1 / 2) * x + 3
def line3 (x : ℝ) : ℝ := 2

theorem area_of_triangle : (area (triangle_intersection_points line1 line2 line3)) = 0.45 :=
by
  -- Proof omitted
  sorry

end area_of_triangle_l483_483568


namespace total_people_on_hike_l483_483815

-- Definitions of the conditions
def n_cars : ℕ := 3
def n_people_per_car : ℕ := 4
def n_taxis : ℕ := 6
def n_people_per_taxi : ℕ := 6
def n_vans : ℕ := 2
def n_people_per_van : ℕ := 5

-- Statement of the problem
theorem total_people_on_hike : 
  n_cars * n_people_per_car + n_taxis * n_people_per_taxi + n_vans * n_people_per_van = 58 :=
by sorry

end total_people_on_hike_l483_483815


namespace min_area_ratio_of_inscribed_right_triangle_in_equilateral_l483_483675

theorem min_area_ratio_of_inscribed_right_triangle_in_equilateral (a b c d e f : ℝ) 
    (hDEF : (d = 0 ∧ e = 0 ∧ f = 0) ∨ ( (a,b,c) ≠ (0,0,0) ∧ ∠ DEF = 90 ∧ ∠ EDF = 30 )) 
    (hABC : (a,b,c) ∈ equilateral_triangle ) :
    ∃ DEF ABC : ℝ, min (frac_area DEF ABC) = 3 / 14 := 
sorry

end min_area_ratio_of_inscribed_right_triangle_in_equilateral_l483_483675


namespace original_number_l483_483364

theorem original_number (N m a b c : ℕ) (hN : N = 3306) 
  (h_eq : 3306 + m = 222 * (a + b + c)) 
  (hm_digits : m = 100 * a + 10 * b + c) 
  (h1 : a + b + c = 15) 
  (h2 : ∃ (a b c : ℕ), a + b + c = 15 ∧ 100 * a + 10 * b + c = 78): 
  100 * a + 10 * b + c = 753 := 
by sorry

end original_number_l483_483364


namespace same_units_digit_pages_count_l483_483557

theorem same_units_digit_pages_count :
  (∃ s : Finset ℕ, s.card = 6 ∧ (∀ x ∈ s, 1 ≤ x ∧ x ≤ 60 ∧ (x % 10 = (61 - x) % 10))) :=
by
  let s := {6, 16, 26, 36, 46, 56}.toFinset
  have h_card : s.card = 6 := by
    simp [s]
  have h_property : ∀ x ∈ s, 1 ≤ x ∧ x ≤ 60 ∧ (x % 10 = (61 - x) % 10) := by
    simp [s] 
    intros x hx
    fin_cases x,
    all_goals {simp}
  exact ⟨s, h_card, h_property⟩

end same_units_digit_pages_count_l483_483557


namespace smallest_arithmetic_mean_divisible_by_1111_l483_483825

theorem smallest_arithmetic_mean_divisible_by_1111 :
  ∃ (n : ℕ), (∀ i : ℕ, i < 9 → 1111 ∣ (n + i)) ∧ (n + 4 = 97) :=
by
  sorry

end smallest_arithmetic_mean_divisible_by_1111_l483_483825


namespace clock_angle_10_50_l483_483904

theorem clock_angle_10_50 :
  let hour := 10
  let minute := 50
  let hour_angle := hour * 30 + 30 * (minute / 60)
  let minute_angle := minute * 6
  let angle_between := |hour_angle - minute_angle| 
  hour = 10 → minute = 50 → angle_between = 25 :=
by
  intros
  let hour_angle := 10 * 30 + 30 * (50 / 60)
  let minute_angle := 50 * 6
  let angle_between := abs (hour_angle - minute_angle)
  have h1 : hour_angle = 325 := by norm_num
  have h2 : minute_angle = 300 := by norm_num
  have h3 : angle_between = abs (325 - 300) := by simp [h1, h2]
  have h4 : angle_between = 25 := by norm_num at h3
  exact (Eq.trans h4 rfl)

end clock_angle_10_50_l483_483904


namespace solve_for_x_l483_483791

theorem solve_for_x (log10_5 : ℝ ≈ 0.6990) (log10_625 : ℝ ≈ 2.796) : ∃ x : ℝ, 500^4 = 10^x ∧ x ≈ 10.796 := by
sorry

end solve_for_x_l483_483791


namespace smallest_arithmetic_mean_divisible_1111_l483_483842

theorem smallest_arithmetic_mean_divisible_1111 :
  ∃ n : ℕ, 93 ≤ n ∧ n + 4 = 97 ∧ (∀ i : ℕ, i ∈ finset.range 9 → (n + i) % 11 = 0 ∨ (n + i) % 101 = 0) :=
sorry

end smallest_arithmetic_mean_divisible_1111_l483_483842


namespace partition_count_l483_483403

noncomputable def count_partition (n : ℕ) : ℕ :=
  -- Function that counts the number of ways to partition n as per the given conditions
  n

theorem partition_count (n : ℕ) (h : n > 0) :
  count_partition n = n :=
sorry

end partition_count_l483_483403


namespace average_difference_is_7_l483_483483

/-- The differences between Mia's and Liam's study times for each day in one week -/
def daily_differences : List ℤ := [15, -5, 25, 0, -15, 20, 10]

/-- The number of days in a week -/
def number_of_days : ℕ := 7

/-- The total difference over the week -/
def total_difference : ℤ := daily_differences.sum

/-- The average difference per day -/
def average_difference_per_day : ℚ := total_difference / number_of_days

theorem average_difference_is_7 : average_difference_per_day = 7 := by 
  sorry

end average_difference_is_7_l483_483483


namespace expected_waiting_time_for_first_bite_l483_483217

noncomputable def average_waiting_time (λ : ℝ) : ℝ := 1 / λ

theorem expected_waiting_time_for_first_bite (bites_first_rod : ℝ) (bites_second_rod : ℝ) (total_time_minutes : ℝ) (total_time_seconds : ℝ) :
  bites_first_rod = 5 → 
  bites_second_rod = 1 → 
  total_time_minutes = 5 → 
  total_time_seconds = 300 → 
  average_waiting_time (bites_first_rod + bites_second_rod) * total_time_seconds = 50 :=
begin
  intros,
  sorry
end

end expected_waiting_time_for_first_bite_l483_483217


namespace correct_option_A_l483_483573

theorem correct_option_A :
  (sqrt 6 / sqrt 3 = sqrt 2) ∧ 
  ¬ (2 * sqrt 2 + 3 * sqrt 3 = 5 * sqrt 5) ∧ 
  ¬ (a ^ 6 / a ^ 3 = a ^ 2) ∧ 
  ¬ ((a ^ 3) ^ 2 = a ^ 5) :=
by
  sorry

end correct_option_A_l483_483573


namespace smallest_arithmetic_mean_divisible_1111_l483_483847

theorem smallest_arithmetic_mean_divisible_1111 :
  ∃ n : ℕ, 93 ≤ n ∧ n + 4 = 97 ∧ (∀ i : ℕ, i ∈ finset.range 9 → (n + i) % 11 = 0 ∨ (n + i) % 101 = 0) :=
sorry

end smallest_arithmetic_mean_divisible_1111_l483_483847


namespace intersection_is_planar_curve_l483_483070

-- Define the conical surfaces
def cone1 (a1 b1 c1 k : ℝ) (x y z : ℝ) : Prop :=
  (x - a1)^2 + (y - b1)^2 = k^2 * (z - c1)^2

def cone2 (a2 b2 c2 k : ℝ) (x y z : ℝ) : Prop :=
  (x - a2)^2 + (y - b2)^2 = k^2 * (z - c2)^2

-- Assumption that k = tan(α), same for both cones
variable {k : ℝ}

-- Assumption that the axes are parallel to the z-axis
variable {z_axis_parallel : Prop}

theorem intersection_is_planar_curve 
  (a1 b1 c1 a2 b2 c2 : ℝ) (h1 : cone1 a1 b1 c1 k) (h2 : cone2 a2 b2 c2 k) :
  ∃ (A B C D : ℝ), ∀ (x y z : ℝ), 
  A * x + B * y + C * z + D = 0 := sorry

end intersection_is_planar_curve_l483_483070


namespace min_area_ratio_l483_483676

theorem min_area_ratio (A B C D E F : Point) (hABC : equilateral_triangle A B C) 
  (hD : D ∈ segment A B) (hE : E ∈ segment B C) (hF : F ∈ segment C A) 
  (hDEF : right_triangle D E F) (h_angle_DEF : angle D E F = 90) (h_angle_EDF : angle E D F = 30) :
  area_ratio S_triangle_DEF S_triangle_ABC = 3 / 14 := 
sorry

end min_area_ratio_l483_483676


namespace units_digit_factorial_sum_l483_483279

theorem units_digit_factorial_sum :
  (1! + 2! + 3! + 4! + ∑ n in Finset.range (2011 - 5), (5 + n)! % 10) % 10 = 3 :=
by
  -- We will handle the details of the proof here.
  sorry

end units_digit_factorial_sum_l483_483279


namespace probability_sum_27_l483_483967

theorem probability_sum_27 :
  let die1_valid_faces := {i | 1 ≤ i ∧ i ≤ 18}.to_finset
  let die2_valid_faces := {i | 3 ≤ i ∧ i ≤ 20}.to_finset
  let valid_pairs := (die1_valid_faces × die2_valid_faces).filter (λ p, p.1 + p.2 = 27)
  let total_pairs := (finset.range 20).product (finset.range 20)
  (valid_pairs.card / total_pairs.card : ℚ) = 3 / 100 :=
by
  sorry

end probability_sum_27_l483_483967


namespace min_arithmetic_mean_of_consecutive_naturals_div_by_1111_l483_483862

theorem min_arithmetic_mean_of_consecutive_naturals_div_by_1111 :
  ∀ a : ℕ, (∃ k, (a * (a + 1) * (a + 2) * (a + 3) * (a + 4) * (a + 5) * (a + 6) * (a + 7) * (a + 8)) = 1111 * k) →
  (a + 4 ≥ 97) :=
sorry

end min_arithmetic_mean_of_consecutive_naturals_div_by_1111_l483_483862


namespace square_area_l483_483437

theorem square_area (P Q R S W X : ℝ) 
  (h_square : P = Q ∧ Q = R ∧ R = S ∧ S = P) 
  (h_W : W ∈ segment S P) 
  (h_X : X ∈ segment Q R) 
  (h_segments : dist Q W = 40 ∧ dist W X = 40 ∧ dist X S = 40) :
  ∃ a : ℝ, a = dist P Q ∧ a^2 = 14400 :=
by 
  sorry

end square_area_l483_483437


namespace expected_waiting_time_first_bite_l483_483237

-- Definitions and conditions as per the problem
def poisson_rate := 6  -- lambda value, bites per 5 minutes
def interval_minutes := 5
def interval_seconds := interval_minutes * 60
def expected_waiting_time_seconds := interval_seconds / poisson_rate

-- The theorem we want to prove
theorem expected_waiting_time_first_bite :
  expected_waiting_time_seconds = 50 := 
by
  let x := interval_seconds / poisson_rate
  have h : interval_seconds = 300 := by norm_num; rfl
  have h2 : x = 50 := by rw [h, interval_seconds]; norm_num
  exact h2

end expected_waiting_time_first_bite_l483_483237


namespace average_waiting_time_for_first_bite_l483_483219

/-- 
Let S be a period of 5 minutes (300 seconds).
- We have an average of 5 bites in 300 seconds on the first fishing rod.
- We have an average of 1 bite in 300 seconds on the second fishing rod.
- The total average number of bites on both rods during this period is 6 bites.
The bites occur independently and follow a Poisson process.

We aim to prove that the waiting time for the first bite, given these conditions, is 
expected to be 50 seconds.
-/
theorem average_waiting_time_for_first_bite :
  let S := 300 -- 5 minutes in seconds
  -- The average number of bites on the first and second rod in period S.
  let avg_bites1 := 5
  let avg_bites2 := 1
  let total_avg_bites := avg_bites1 + avg_bites2
  -- The rate parameter λ for the Poisson process is total_avg_bites / S.
  let λ := total_avg_bites / S
  -- The average waiting time for the first bite.
  1 / λ = 50 :=
by
  let S := 300
  let avg_bites1 := 5
  let avg_bites2 := 1
  let total_avg_bites := avg_bites1 + avg_bites2
  let λ := total_avg_bites / S
  -- convert λ to seconds to ensure unit consistency
  have hλ: λ = 6 / 300 := rfl
  -- The expected waiting time for the first bite is 1 / λ
  have h_waiting_time: 1 / λ = 300 / 6 := by
    rw [hλ, one_div, div_div_eq_mul]
    norm_num
  exact h_waiting_time

end average_waiting_time_for_first_bite_l483_483219


namespace gcd_of_sequence_l483_483419

theorem gcd_of_sequence (a : ℕ → ℤ) (x y p : ℤ) (hp : prime p) (h0 : a 0 = 0) (h1 : a 1 = 0)
  (h_rec : ∀ n, a (n + 2) = x * a (n + 1) + y * a n + 1) :
  gcd (a p) (a (p + 1)) = 1 ∨ gcd (a p) (a (p + 1)) > sqrt p :=
sorry

end gcd_of_sequence_l483_483419


namespace angle_between_vectors_is_90_degrees_l483_483422

open Real EuclideanSpace Finset

noncomputable theory

variables {ι : Type} [Fintype ι]

theorem angle_between_vectors_is_90_degrees
  (a b : EuclideanSpace ℝ ι)
  (h : ∥a + (2 : ℝ) • b∥ = (2 : ℝ) * ∥a∥) :
  angle (a + b) (b + (2 : ℝ) • a) = π / 2 :=
by
  sorry

end angle_between_vectors_is_90_degrees_l483_483422


namespace find_overlapping_area_l483_483976

-- Definitions based on conditions
def length_total : ℕ := 16
def length_strip1 : ℕ := 9
def length_strip2 : ℕ := 7
def area_only_strip1 : ℚ := 27
def area_only_strip2 : ℚ := 18

-- Widths are the same for both strips, hence areas are proportional to lengths
def area_ratio := (length_strip1 : ℚ) / (length_strip2 : ℚ)

-- The Lean statement to prove the question == answer
theorem find_overlapping_area : 
  ∃ S : ℚ, (area_only_strip1 + S) / (area_only_strip2 + S) = area_ratio ∧ 
              area_only_strip1 + S = area_only_strip1 + 13.5 := 
by 
  sorry

end find_overlapping_area_l483_483976


namespace A_max_minus_A_min_l483_483065

theorem A_max_minus_A_min : 
  ∃ A_min A_max, 
  (∃ a b : ℕ, A = (a + 3) / 12 ∧ A = 15 / (26 - b) ∧ a > 0 ∧ b > 0) ∧ ¬A ∈ ℤ ∧ 
  A_max = 15 ∧ A_min = 3 / 4 ∧ A_max - A_min = 57 / 4 := 
sorry

end A_max_minus_A_min_l483_483065


namespace start_A_can_give_B_l483_483734

-- Definitions of the given conditions
variable {Va Vb Vc : ℝ} -- speeds of A, B, and C
variable {Ta Tb Tc : ℝ} -- times taken by A, B, and C to complete the race

-- Condition 1: A gives C a 300 meters start in a kilometer race
def condition1 : Prop := Va * Ta = 1000 ∧ Vc * Tc = 700 ∧ Ta = Tc

-- Condition 2: B gives C a 176.47 meters start in a kilometer race
def condition2 : Prop := Vb * Tb = 1000 ∧ Vc * Tc = 823.53 ∧ Tb = Tc

-- The theorem stating that A can give B a 150 meters start in a kilometer race
theorem start_A_can_give_B (h1 : condition1) (h2 : condition2) : Va / Vb = 1.17647 → 1000 - 1000 / 1.17647 = 150 :=
by
  sorry

end start_A_can_give_B_l483_483734


namespace distance_speed_relationship_l483_483502

variables (a p q t b x : ℝ)
def distanceAB : ℝ := (q * (a * t + b)) / (q - p)

noncomputable def speed_second_object : ℝ := (a * p * t + b * q) / (t * (q - p))

noncomputable def speed_first_object : ℝ := (q * (a * t + b)) / (t * (q - p))

theorem distance_speed_relationship
  (h1 : ∀ t, (x + a) * t = distanceAB a p q t b)
  (h2 : ∀ x t, t * x = (p / q) * distanceAB a p q t b + b):
  speed_second_object a p q t b = (a * p * t + b * q) / (t * (q - p)) ∧
  speed_first_object a p q t b = (q * (a * t + b)) / (t * (q - p)) ∧
  distanceAB a p q t b = (q * (a * t + b)) / (q - p) :=
by
  sorry

end distance_speed_relationship_l483_483502


namespace simplify_and_rationalize_l483_483471

noncomputable def expression := 
  (Real.sqrt 8 / Real.sqrt 3) * 
  (Real.sqrt 25 / Real.sqrt 30) * 
  (Real.sqrt 16 / Real.sqrt 21)

theorem simplify_and_rationalize :
  expression = 4 * Real.sqrt 14 / 63 :=
by
  sorry

end simplify_and_rationalize_l483_483471


namespace opponent_final_score_l483_483103

theorem opponent_final_score (x : ℕ) (h : x + 29 = 39) : x = 10 :=
by {
  sorry
}

end opponent_final_score_l483_483103


namespace polio_probability_l483_483512

noncomputable def p : ℝ := 0.0001
noncomputable def n : ℝ := 1000
noncomputable def λ : ℝ := n * p

def P (k : ℕ) : ℝ := (λ^k * real.exp (-λ)) / (nat.factorial k)

theorem polio_probability :
  P 1 = 0.090484 ∧ P 2 = 0.004524 ∧ P 3 = 0.000151 ∧ P 4 = 0.00000377 :=
by
  sorry

end polio_probability_l483_483512


namespace divide_triangle_into_three_equal_parts_l483_483595

-- Define a triangle
structure Triangle :=
  (A B C : Point)

-- Define a point
structure Point :=
  (x y : ℝ)

-- Define equality of triangles (congruence)
def congruent (T1 T2 : Triangle) : Prop :=
  -- This is a simplified definition; in practice, we need to check the congruence conditions
  sorry

-- The problem statement: Given a triangle ABC
def divideTriangle (T : Triangle) : Prop :=
  let A := T.A in
  let B := T.B in
  let C := T.C in
  -- D and E are points on BC dividing it into three equal segments
  let D := Point ((2 * B.x + C.x) / 3) ((2 * B.y + C.y) / 3) in
  let E := Point ((B.x + 2 * C.x) / 3) ((B.y + 2 * C.y) / 3) in
  let T1 := Triangle A B D in
  let T2 := Triangle A D E in
  let T3 := Triangle A E C in
  congruent T1 T2 ∧ congruent T2 T3

-- The theorem to be proven
theorem divide_triangle_into_three_equal_parts (T : Triangle) :
  divideTriangle T :=
sorry

end divide_triangle_into_three_equal_parts_l483_483595


namespace integral_cos_from_0_to_pi_div_2_l483_483580

open Real

theorem integral_cos_from_0_to_pi_div_2 : (∫ x in 0..(pi/2), cos x) = 1 := 
by
  -- Proof goes here.
  sorry

end integral_cos_from_0_to_pi_div_2_l483_483580


namespace lisa_gift_price_l483_483624

noncomputable def price_of_gift (S_L C_M C_B C_F C_C : ℝ) : ℝ :=
  S_L + C_M + C_B + C_F + C_C + 800

theorem lisa_gift_price :
  ∀ (S_L : ℝ),
  let C_M := 7 / 15 * S_L in
  let C_B := 7 / 6 * C_M in
  let C_F := 3 / 5 * (C_M + C_B) in
  let C_C := 1 / 4 * C_F + 1 / 9 * S_L in
  S_L = 2000 →
  price_of_gift S_L C_M C_B C_F C_C = 6560.44 :=
begin
  intros,
  sorry
end

end lisa_gift_price_l483_483624


namespace inequality_solution_l483_483626

theorem inequality_solution (x : ℝ) : (-3 * x^2 - 9 * x - 6 ≥ -12) ↔ (-2 ≤ x ∧ x ≤ 1) := sorry

end inequality_solution_l483_483626


namespace nine_consecutive_arithmetic_mean_divisible_1111_l483_483867

theorem nine_consecutive_arithmetic_mean_divisible_1111 {n : ℕ} (h1 : ∀ i : ℕ, 0 ≤ i ∧ i < 9 → Nat.Prime (n + i)) :
  ∃ n : ℕ, (∀ k : ℕ, 0 ≤ k ∧ k < 9 → (n + k) ∣ 1111) → (n + 4) = 97 := by
  sorry

end nine_consecutive_arithmetic_mean_divisible_1111_l483_483867


namespace parallelogram_AB_length_l483_483374

theorem parallelogram_AB_length (a b : ℝ) (h₀ : AD = a) (h₁ : CD = b) 
  (h₂ : ∠D = 60) (h₃ : ∠B = 30) (h₄ : AB ∥ CD) : 
  AB = (2 + Real.sqrt 3) * b / Real.sqrt 3 := by
  sorry

end parallelogram_AB_length_l483_483374


namespace rhombus_properties_l483_483083

-- The statement to be proved: The area of the rhombus is 270 square units
-- and it is not a square, given the diagonals lengths.
theorem rhombus_properties
  (d1 d2 : ℕ)
  (h_d1 : d1 = 30)
  (h_d2 : d2 = 18) :
  (d1 * d2 / 2 = 270) ∧ (d1 ≠ d2) :=
by
  split
  { rw [h_d1, h_d2]
    norm_num }
  { rw [h_d1, h_d2]
    exact ne_of_eq h_d1 h_d2.symm }
  sorry

end rhombus_properties_l483_483083


namespace austin_pairs_of_shoes_l483_483198

theorem austin_pairs_of_shoes (S : ℕ) :
  0.45 * (S : ℝ) + 11 = S → S / 2 = 10 :=
by
  sorry

end austin_pairs_of_shoes_l483_483198


namespace sum_of_digits_3_plus_4_pow_15_l483_483925

theorem sum_of_digits_3_plus_4_pow_15 :
  let n := (3 + 4) ^ 15
  let last_two_digits := n % 100
  let tens_digit := last_two_digits / 10
  let ones_digit := last_two_digits % 10
  tens_digit + ones_digit = 7 :=
by sorry

end sum_of_digits_3_plus_4_pow_15_l483_483925


namespace find_eccentricity_l483_483292

-- Define the problem conditions
variables {a b : ℝ} (ha : a > b) (hb : b > 0)
variable (P : ℝ × ℝ)
def ellipse (P : ℝ × ℝ) : Prop := (P.1^2 / a^2) + (P.2^2 / b^2) = 1
variables (F1 F2 : ℝ × ℝ)
def orthogonal (P F1 F2 : ℝ × ℝ) : Prop := (P.1 - F1.1) * (P.1 - F2.1) + (P.2 - F1.2) * (P.2 - F2.2) = 0
def tan_angle (P F1 F2 : ℝ × ℝ) : Prop := real.tan (real.angle (P - F1) (F2 - F1)) = 1/2

-- Prove the eccentricity
theorem find_eccentricity {a b : ℝ} (ha : a > b) (hb : b > 0) 
  (P F1 F2 : ℝ × ℝ) (hP : ellipse P) 
  (h_orth : orthogonal P F1 F2) (h_tan : tan_angle P F1 F2) :
  (sqrt 5 / 3) = (* eccentricity calculation here *) := 
sorry

end find_eccentricity_l483_483292


namespace clock_angle_10_50_l483_483903

theorem clock_angle_10_50 :
  let hour := 10
  let minute := 50
  let hour_angle := hour * 30 + 30 * (minute / 60)
  let minute_angle := minute * 6
  let angle_between := |hour_angle - minute_angle| 
  hour = 10 → minute = 50 → angle_between = 25 :=
by
  intros
  let hour_angle := 10 * 30 + 30 * (50 / 60)
  let minute_angle := 50 * 6
  let angle_between := abs (hour_angle - minute_angle)
  have h1 : hour_angle = 325 := by norm_num
  have h2 : minute_angle = 300 := by norm_num
  have h3 : angle_between = abs (325 - 300) := by simp [h1, h2]
  have h4 : angle_between = 25 := by norm_num at h3
  exact (Eq.trans h4 rfl)

end clock_angle_10_50_l483_483903


namespace only_prime_in_sequence_of_47_l483_483345

theorem only_prime_in_sequence_of_47 (n : ℕ) :
  ∃! x : ℕ, x < n ∧ ∃ k : ℕ, x = 47 * (10^(2*k) + 10^(2*(k-1)) + ... + 1) ∧ prime x :=
by 
  sorry

end only_prime_in_sequence_of_47_l483_483345


namespace expected_voters_for_A_l483_483140

-- Define the conditions in Lean
def perc_Democrats : ℝ := 0.60
def perc_Republicans : ℝ := 0.40
def perc_Democrats_for_A : ℝ := 0.75
def perc_Republicans_for_A : ℝ := 0.30

-- Define the population size for simulation
def total_voters : ℕ := 100

-- Calculate the expected percentage of voters for candidate A
theorem expected_voters_for_A : 
  (0.75 * (0.60 * total_voters) + 0.3 * (0.40 * total_voters)) / total_voters * 100 = 57 :=
by
  -- Placeholder for the proof
  sorry

end expected_voters_for_A_l483_483140


namespace sum_of_digits_3_plus_4_pow_15_l483_483924

theorem sum_of_digits_3_plus_4_pow_15 :
  let n := (3 + 4) ^ 15
  let last_two_digits := n % 100
  let tens_digit := last_two_digits / 10
  let ones_digit := last_two_digits % 10
  tens_digit + ones_digit = 7 :=
by sorry

end sum_of_digits_3_plus_4_pow_15_l483_483924


namespace fraction_not_ripe_correct_l483_483359

-- Define the known conditions as variables
def total_apples : ℕ := 30
def too_small_fraction : ℚ := 1/6
def perfect_apples : ℕ := 15

-- Calculate the number of too small apples and the fraction of not ripe apples
def too_small_apples : ℕ := (too_small_fraction * total_apples).natAbs
def not_ripe_apples : ℕ := total_apples - too_small_apples - perfect_apples
def fraction_not_ripe : ℚ := not_ripe_apples / total_apples.toRat

-- The theorem to be proved
theorem fraction_not_ripe_correct : fraction_not_ripe = 1/3 := by
  sorry

end fraction_not_ripe_correct_l483_483359


namespace sum_b_n_formula_l483_483295

variable {a : ℕ → ℕ} -- Sequence a_n
variable {b : ℕ → ℚ} -- Sequence b_n
variable {S : ℕ → ℚ} -- Sum of the first n terms of b_n

-- Assuming condition 1: a_n = n
def a_n (n : ℕ) : ℕ := n

-- Assuming condition 2: b_n = 1 / (n * (n + 2))
def b_n (n : ℕ) : ℚ := 1 / (n * (n + 2))

-- Define the sum T_n
noncomputable def T_n (n : ℕ) : ℚ := ∑ i in Finset.range n, b_n (i + 1)

-- The theorem to prove
theorem sum_b_n_formula (n : ℕ) : T_n n = (3 / 4) - (2 * n + 3) / (2 * (n + 1) * (n + 2)) :=
  sorry

end sum_b_n_formula_l483_483295


namespace collinear_I_W_Z_l483_483046

theorem collinear_I_W_Z
  (A B C I U V X Y W Z : Point)
  (hA : IsIncenter A I)
  (hC_perpendicular : Perpendicular (line I C) (line I U))
  (hU : OnSegment U B C)
  (hV : OnArc V B C)
  (hU_parallel_A_I : Parallel (line U X) (line A I))
  (hV_parallel_A_I : Parallel (line V Y) (line A I))
  (hW_midpoint_A_X : IsMidpoint W A X)
  (hZ_midpoint_B_C : IsMidpoint Z B C)
  (hI_collinear_X_Y : Collinear3 I X Y) : Collinear3 I W Z :=  
sorry

end collinear_I_W_Z_l483_483046


namespace expected_waiting_time_correct_l483_483233

noncomputable def combined_average_bites_per_5_minutes := 6
def average_waiting_time_for_first_bite_in_seconds : ℝ := 50

theorem expected_waiting_time_correct :
  (1 / combined_average_bites_per_5_minutes) * 300 = average_waiting_time_for_first_bite_in_seconds :=
by
  sorry

end expected_waiting_time_correct_l483_483233


namespace route_down_distance_l483_483518

noncomputable def rate_up : ℝ := 3
noncomputable def time_up : ℝ := 2
noncomputable def time_down : ℝ := 2
noncomputable def rate_down := 1.5 * rate_up

theorem route_down_distance : rate_down * time_down = 9 := by
  sorry

end route_down_distance_l483_483518


namespace cut_without_cutting_dominos_l483_483534

-- Definition and Conditions
def chessboard : Type := fin 6 × fin 6
def domino := (chessboard × chessboard)

-- We assume each domino covers exactly two squares on the board, and the board is covered by 18 such dominoes
def cover (board : fin 6 → fin 6 → Prop) : Prop :=
  ∃ (dominoes : fin 18 → domino),
    ∀ (i : fin 18), 
      (fst (dominoes i)).fst.val < 6 ∧ (fst (dominoes i)).snd.val < 6 ∧
      (snd (dominoes i)).fst.val < 6 ∧ (snd (dominoes i)).snd.val < 6 ∧
      (board (fst (dominoes i)).fst (fst (dominoes i)).snd = true) ∧
      (board (snd (dominoes i)).fst (snd (dominoes i)).snd = true) ∧
      ((abs ((fst (dominoes i)).fst.val - (snd (dominoes i)).fst.val) + abs ((fst (dominoes i)).snd.val - (snd (dominoes i)).snd.val)) = 1)

-- The theorem statement
theorem cut_without_cutting_dominos : ∀ (board_cover : fin 6 → fin 6 → Prop),
  cover board_cover → 
  ∃ (x : fin 6) (dir : bool), 
    (dir = true → (∀ (i : fin 6), 
      (i.val < x.val → ∀ j : fin 6, board_cover i j = false) ∨ 
      (i.val ≥ x.val → ∀ j : fin 6, board_cover i j = false))) ∧
    (dir = false → (∀ (j : fin 6), 
      (j.val < x.val → ∀ i : fin 6, board_cover i j = false) ∨
      (j.val ≥ x.val → ∀ i : fin 6, board_cover i j = false))) := 
sorry

end cut_without_cutting_dominos_l483_483534


namespace leading_coefficient_of_g_is_3_l483_483488

variable (g : ℤ → ℤ)
variable (x : ℤ)

noncomputable def leading_coefficient_of_g := 
  ∀ (g : ℤ → ℤ), (∀ x : ℤ, g(x + 1) - g(x) = 6 * x + 6) → ∃ c : ℤ, leading_coeff g = c ∧ c = 3

theorem leading_coefficient_of_g_is_3 
  (hg : ∀ x : ℤ, g(x + 1) - g(x) = 6 * x + 6) :
  ∃ c : ℤ, leading_coeff g = c ∧ c = 3 :=
sorry

end leading_coefficient_of_g_is_3_l483_483488


namespace distance_to_origin_l483_483761

theorem distance_to_origin (i : ℂ) (hi : i.im = 1 ∧ i.re = 0) :
  let z := i / (i + 1)
  in complex.abs z = complex.abs ((1 / 2) + (1 / 2) * i) := by 
  have h1 : z = (1 / 2) + (1 / 2) * i := 
    by { field_simp [complex.add, complex.mul, complex.div, hi], done } 
  rw [h1]
  have h2 : complex.abs ((1 / 2) + (1 / 2) * i) = 
             complex.abs (1 / 2 * (1 + i)) := 
    by { congr' 1, ring } 
  rw [h2]
  exact complex.abs_mul (1 / 2) (1 + i)

end distance_to_origin_l483_483761


namespace sqrt_inequality_iff_l483_483504

variables (a b c : ℝ)
-- Given conditions: a > b > c and a + b + c = 0
variables (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0)

-- Prove that sqrt(b^2 - ac) < sqrt(3) * a <= provable if and only if (a - b) * (a - c) > 0
theorem sqrt_inequality_iff (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  (real.sqrt (b * b - a * c) < real.sqrt 3 * a) ↔ ((a - b) * (a - c) > 0) :=
sorry

end sqrt_inequality_iff_l483_483504


namespace cube_volume_is_125_l483_483802

-- Declare the side length variable
variable (s : ℝ)

-- Define condition: the space diagonal is 5 * sqrt 3
def is_space_diagonal_correct : Prop := s * (sqrt 3) = 5 * (sqrt 3)

-- Define the volume of a cube
def volume_of_cube : ℝ := s^3

-- Prove the volume is 125 cubic units given the conditions
theorem cube_volume_is_125 (h : is_space_diagonal_correct s) : volume_of_cube s = 125 := by
  sorry  -- Proof is omitted as per instructions

end cube_volume_is_125_l483_483802


namespace find_m_over_s_l483_483304

variables (m n p s : ℝ)

-- Given conditions as Lean definitions
def cond1 := m / n = 18
def cond2 := p / n = 2
def cond3 := p / s = 1 / 9

-- The goal statement to prove
theorem find_m_over_s (hc1 : cond1) (hc2 : cond2) (hc3 : cond3) : m / s = 1 / 2 :=
sorry

end find_m_over_s_l483_483304


namespace martian_base_conversion_l483_483535

def base_n_to_decimal (n : ℕ) (digits : List ℕ) : ℕ :=
digits.reverse.enum.map (λ ⟨i, d⟩ => d * n^i).sum

theorem martian_base_conversion : base_n_to_decimal 9 [4, 5, 2, 7] = 3346 := by
  sorry

end martian_base_conversion_l483_483535


namespace hiking_rate_l483_483516

theorem hiking_rate (rate_uphill: ℝ) (time_total: ℝ) (time_uphill: ℝ) (rate_downhill: ℝ) 
  (h1: rate_uphill = 4) (h2: time_total = 3) (h3: time_uphill = 1.2) : rate_downhill = 4.8 / (time_total - time_uphill) :=
by
  sorry

end hiking_rate_l483_483516


namespace nine_consecutive_arithmetic_mean_divisible_1111_l483_483871

theorem nine_consecutive_arithmetic_mean_divisible_1111 {n : ℕ} (h1 : ∀ i : ℕ, 0 ≤ i ∧ i < 9 → Nat.Prime (n + i)) :
  ∃ n : ℕ, (∀ k : ℕ, 0 ≤ k ∧ k < 9 → (n + k) ∣ 1111) → (n + 4) = 97 := by
  sorry

end nine_consecutive_arithmetic_mean_divisible_1111_l483_483871


namespace area_triangle_ADE_l483_483806

noncomputable def line1 := sorry -- Define the line y = -2x - 4
noncomputable def line2 := sorry -- Define the line y = -2x + 4
noncomputable def hyperbola1 := sorry -- Define the hyperbola y = -6 / x

-- Assume A, B are intersections of line1 and the hyperbola
def A := (-3, 2) : ℝ × ℝ
def B := (1, -6) : ℝ × ℝ

-- Assume D, E are intersections of line2 and the hyperbola
def D := (3, -2) : ℝ × ℝ
def E := (-1, 6) : ℝ × ℝ

-- Define the function to compute area of a triangle given three points
noncomputable def area_triangle (P Q R : ℝ × ℝ) : ℝ :=
  1/2 * abs (P.1*Q.2 + Q.1*R.2 + R.1*P.2 - P.2*Q.1 - Q.2*R.1 - R.2*P.1)

-- Problem statement in Lean: Prove that the area of triangle ADE is 16
theorem area_triangle_ADE :
  area_triangle A D E = 16 :=
sorry

end area_triangle_ADE_l483_483806


namespace isosceles_triangle_perimeter_l483_483492

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 9) (h2 : b = 4) (h3 : b < a + a) : a + a + b = 22 := by
  sorry

end isosceles_triangle_perimeter_l483_483492


namespace solve_for_y_l483_483256

theorem solve_for_y (x y : ℝ) :
  (3 * x^2 + 9 * x + 7 * y + 2 = 0) ∧ (3 * x + 2 * y + 4 = 0) → 4 * y^2 + 19 * y - 14 = 0 :=
by
  intro h
  cases h with h1 h2
  sorry

end solve_for_y_l483_483256


namespace system_of_equations_solutions_l483_483474

theorem system_of_equations_solutions (x y z : ℝ) :
  (x^2 - y^2 + z = 27 / (x * y)) ∧ 
  (y^2 - z^2 + x = 27 / (y * z)) ∧ 
  (z^2 - x^2 + y = 27 / (z * x)) ↔ 
  (x = 3 ∧ y = 3 ∧ z = 3) ∨
  (x = -3 ∧ y = -3 ∧ z = 3) ∨
  (x = -3 ∧ y = 3 ∧ z = -3) ∨
  (x = 3 ∧ y = -3 ∧ z = -3) :=
by 
  sorry

end system_of_equations_solutions_l483_483474


namespace evaluations_total_l483_483067

theorem evaluations_total :
    let class_A_students := 30
    let class_A_mc := 12
    let class_A_essay := 3
    let class_A_presentation := 1

    let class_B_students := 25
    let class_B_mc := 15
    let class_B_short_answer := 5
    let class_B_essay := 2

    let class_C_students := 35
    let class_C_mc := 10
    let class_C_essay := 3
    let class_C_presentation_groups := class_C_students / 5 -- groups of 5

    let class_D_students := 40
    let class_D_mc := 11
    let class_D_short_answer := 4
    let class_D_essay := 3

    let class_E_students := 20
    let class_E_mc := 14
    let class_E_short_answer := 5
    let class_E_essay := 2

    let total_mc := (class_A_students * class_A_mc) +
                    (class_B_students * class_B_mc) +
                    (class_C_students * class_C_mc) +
                    (class_D_students * class_D_mc) +
                    (class_E_students * class_E_mc)

    let total_short_answer := (class_B_students * class_B_short_answer) +
                              (class_D_students * class_D_short_answer) +
                              (class_E_students * class_E_short_answer)

    let total_essay := (class_A_students * class_A_essay) +
                       (class_B_students * class_B_essay) +
                       (class_C_students * class_C_essay) +
                       (class_D_students * class_D_essay) +
                       (class_E_students * class_E_essay)

    let total_presentation := (class_A_students * class_A_presentation) +
                              class_C_presentation_groups

    total_mc + total_short_answer + total_essay + total_presentation = 2632 := by
    sorry

end evaluations_total_l483_483067


namespace area_of_circle_with_radius_5_l483_483125

-- We define the radius of the circle
def radius : ℝ := 5

-- We define the expected area of the circle
def expected_area : ℝ := 25 * Real.pi

-- The theorem statement: The area of a circle with radius 5 meters is 25π square meters.
theorem area_of_circle_with_radius_5 : Real.pi * radius^2 = expected_area := by
  sorry

end area_of_circle_with_radius_5_l483_483125


namespace percentage_girls_cleared_l483_483952

-- Defining the conditions as given in the problem
def total_students := 400
def percentage_boys_cleared := 0.60
def percentage_total_cleared := 0.65
def girls_participated := 100 -- rounding off for practical purposes

-- Calculating number of boys
def boys_participated := total_students - girls_participated

-- Calculating number of boys who cleared the cut off
def boys_cleared := percentage_boys_cleared * boys_participated

-- Calculating total number of students who cleared the cut off
def total_cleared := percentage_total_cleared * total_students

-- Calculating number of girls who cleared the cut off
def girls_cleared := total_cleared - boys_cleared

-- Proving the percentage of girls who cleared the cut off is 80%
theorem percentage_girls_cleared : (girls_cleared / girls_participated) * 100 = 80 := by
  sorry

end percentage_girls_cleared_l483_483952


namespace sum_of_two_integers_is_22_l483_483884

noncomputable def a_and_b_sum_to_S : Prop :=
  ∃ (a b S : ℕ), 
    a + b = S ∧ 
    a^2 - b^2 = 44 ∧ 
    a * b = 120 ∧ 
    S = 22

theorem sum_of_two_integers_is_22 : a_and_b_sum_to_S :=
by {
  sorry
}

end sum_of_two_integers_is_22_l483_483884


namespace mike_worked_four_hours_l483_483394

-- Define the time to perform each task in minutes
def time_wash_car : ℕ := 10
def time_change_oil : ℕ := 15
def time_change_tires : ℕ := 30

-- Define the number of tasks Mike performed
def num_wash_cars : ℕ := 9
def num_change_oil : ℕ := 6
def num_change_tires : ℕ := 2

-- Define the total minutes Mike worked
def total_minutes_worked : ℕ :=
  (num_wash_cars * time_wash_car) +
  (num_change_oil * time_change_oil) +
  (num_change_tires * time_change_tires)

-- Define the conversion from minutes to hours
def total_hours_worked : ℕ := total_minutes_worked / 60

-- Formalize the proof statement
theorem mike_worked_four_hours :
  total_hours_worked = 4 :=
by
  sorry

end mike_worked_four_hours_l483_483394


namespace desk_arrangements_count_l483_483547

theorem desk_arrangements_count :
  let desk_arrangements (n : Nat) : Nat :=
    if n = 0 then
      1
    else
      (Nat.pow 2 n - 1) * (Nat.pow 2 n - 1) + 1
  in desk_arrangements 5 = 962 :=
by 
  sorry

end desk_arrangements_count_l483_483547


namespace smallest_four_digit_mod_8_l483_483915

theorem smallest_four_digit_mod_8 :
  ∃ x : ℕ, x ≥ 1000 ∧ x < 10000 ∧ x % 8 = 5 ∧ (∀ y : ℕ, y >= 1000 ∧ y < 10000 ∧ y % 8 = 5 → x ≤ y) :=
sorry

end smallest_four_digit_mod_8_l483_483915


namespace distinct_distribution_ways_l483_483259

theorem distinct_distribution_ways :
  ∃ (num_ways : ℕ), num_ways = 3 ∧ ∀ (positions : ℕ) (schools : ℕ),
    positions = 6 →
    schools = 3 →
    (∀ (distribution : Fin schools → ℕ),
      (∀ i, 1 ≤ distribution i) →
      (∀ i j, i ≠ j → distribution i ≠ distribution j) →
      ∑ i, distribution i = positions →
      distribution = distribution)
sorry

end distinct_distribution_ways_l483_483259


namespace isosceles_triangle_perimeter_l483_483652

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 6 ∨ a = 9) (h2 : b = 6 ∨ b = 9) (h : a ≠ b) : (a * 2 + b = 21 ∨ a * 2 + b = 24) :=
by
  sorry

end isosceles_triangle_perimeter_l483_483652


namespace geometry_problem_l483_483742

-- Definitions of the geometric objects and relationships in Lean
variables {ℝ : Type*} [nontrivial_ordered_comm_ring ℝ] 
variables (O1 O2 A B C P Q : EuclideanGeometry.Point ℝ)
variables (circle1 : EuclideanGeometry.Circle O1)
variables (circle2 : EuclideanGeometry.Circle O2)

-- Given problem conditions
variable (h1 : EuclideanGeometry.Intersect circle1 circle2)
variable (h2 : EuclideanGeometry.Tangent comm_line_AB circle1)
variable (h3 : EuclideanGeometry.PointOnCircle A circle1)
variable (h4 : EuclideanGeometry.PointOnCircle B circle2)
variable (h5 : EuclideanGeometry.Tangent comm_line_AC circle2)
variable (h6 : EuclideanGeometry.Tangent comm_line_BC circle1)
variable (h7 : EuclideanGeometry.Tangent line_CP circle1)
variable (h8 : EuclideanGeometry.Tangent line_CQ circle2)

-- Similar triangles assumption
variable (h_sim : EuclideanGeometry.SimilarTriangle (EuclideanGeometry.Triangle A P C) 
                                                   (EuclideanGeometry.Triangle B Q C))

-- Prove that if the triangles are similar, their corresponding lines are parallel
theorem geometry_problem : EuclideanGeometry.Parallel O1O2 PQ :=
by
  sorry

end geometry_problem_l483_483742


namespace ratio_of_inscribed_circle_segments_l483_483544

/-- A circle is inscribed in a triangle with side lengths 9, 12, and 15.
Let the segments of the side of length 9, made by a point of tangency, be r and s, with r < s.
Prove that the ratio r:s is 1:2. -/
theorem ratio_of_inscribed_circle_segments (r s : ℕ) (h : r < s) 
  (triangle_sides : r + s = 9) (point_of_tangency_15 : s + (12 - r) = 9) : 
  r / s = 1 / 2 := 
sorry

end ratio_of_inscribed_circle_segments_l483_483544


namespace smallest_arithmetic_mean_divisible_product_l483_483835

theorem smallest_arithmetic_mean_divisible_product :
  ∃ (n : ℕ), (∀ i : ℕ, i < 9 → Nat.gcd ((n + i) * 1111) 1111 = 1111) ∧ (n + 4 = 97) := 
  sorry

end smallest_arithmetic_mean_divisible_product_l483_483835


namespace equation_of_line_is_correct_l483_483314

-- Define the conditions
def point : ℝ × ℝ := (-2, 1)
def slope : ℝ := Real.pi / 2

-- Define the equation of the line with the conditions
def equation_of_line (x : ℝ) : Prop := x + 2 = 0

-- Theorem statement, to be proved later
theorem equation_of_line_is_correct :
  equation_of_line (-2) :=
by
  -- Here you can add the formal proof
  sorry

end equation_of_line_is_correct_l483_483314


namespace books_read_so_far_l483_483106

/-- There are 22 different books in the 'crazy silly school' series -/
def total_books : Nat := 22

/-- You still have to read 10 more books -/
def books_left_to_read : Nat := 10

theorem books_read_so_far :
  total_books - books_left_to_read = 12 :=
by
  sorry

end books_read_so_far_l483_483106


namespace find_mass_of_man_l483_483537

noncomputable def mass_of_man : ℝ :=
  let length : ℝ := 4
  let breadth : ℝ := 3
  let sunk_distance : ℝ := 0.04
  let density_of_water : ℝ := 999.1
  let local_gravity : ℝ := 9.75

  let area_of_boat := length * breadth
  let volume_displaced := area_of_boat * sunk_distance
  let mass_water_displaced := density_of_water * volume_displaced
  mass_water_displaced / local_gravity

theorem find_mass_of_man : abs (mass_of_man - 49.185) < 0.001 := by
  let length : ℝ := 4
  let breadth : ℝ := 3
  let sunk_distance : ℝ := 0.04
  let density_of_water : ℝ := 999.1
  let local_gravity : ℝ := 9.75

  let area_of_boat := length * breadth
  let volume_displaced := area_of_boat * sunk_distance
  let mass_water_displaced := density_of_water * volume_displaced
  let calculated_mass_of_man := mass_water_displaced / local_gravity
  have h : abs (calculated_mass_of_man - 49.185) < 0.001, by
    -- Calculation steps here
    sorry
  exact h

end find_mass_of_man_l483_483537


namespace perpendicular_planes_proof_l483_483704

noncomputable theory

variables {m n : Line} {α β : Plane}

def parallel_lines (l1 l2 : Line) : Prop := sorry
def line_in_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry
def parallel_planes (p1 p2 : Plane) : Prop := sorry
def perpendicular_planes (p1 p2 : Plane) : Prop := sorry

theorem perpendicular_planes_proof (h1 : parallel_lines m n)
                                  (h2 : line_in_plane m α)
                                  (h3 : perpendicular_line_plane n β) :
                                  perpendicular_planes α β := sorry

end perpendicular_planes_proof_l483_483704


namespace distance_ratio_proof_l483_483517

-- Define the conditions as hypotheses
variable (V : ℝ) -- Mr. Harris's walking speed in distance per hour
variable (D D' : ℝ) -- Distances to the store and your destination

def mr_harris_time := 2 -- Time taken by Mr. Harris to walk to the store (in hours)
noncomputable def your_time := 3 -- Time taken by you to walk to your destination (in hours)

-- You walk twice as fast as Mr. Harris
def your_speed := 2 * V

-- Mr. Harris covered D meters in 2 hours
def mr_harris_distance := V * mr_harris_time

-- You covered D' meters in 3 hours
noncomputable def your_distance := your_speed * your_time

theorem distance_ratio_proof :
  (\(D' : ℝ\), D' = your_distance) -> (\(D : ℝ\), D = mr_harris_distance) -> D' / D = 3 := by
  sorry -- proof goes here

end distance_ratio_proof_l483_483517


namespace a_seq_formula_T_seq_sum_l483_483316

-- Define the sequence $\{a_n\}$ with its conditions
def a_seq (n : ℕ) : ℕ := 
  if n = 1 then 1 else 4 * n - 3

-- Define the sum of the first $n$ terms of the sequence $\{a_n\}$
noncomputable def S_seq (n : ℕ) : ℕ := 
  if n = 1 then 1 else n * (2 * n - 1)

-- Define the sequence $\{b_n\}$
noncomputable def b_seq (n : ℕ) : ℚ :=
  (a_seq n : ℚ) / (2 ^ n : ℚ)

-- Define the sum of the first $n$ terms of the sequence $\{b_n\}$
noncomputable def T_seq (n : ℕ) : ℚ :=
  ∑ k in finset.range n, b_seq (k + 1)

-- State the theorem
theorem a_seq_formula (n : ℕ) (hn : n ≥ 1) : a_seq n = 4 * n - 3 := by
  sorry

theorem T_seq_sum (n : ℕ) (hn : n ≥ 1) : 
  T_seq n = 5 - (4 * n + 5) / (2 ^ n : ℚ) := by 
  sorry

end a_seq_formula_T_seq_sum_l483_483316


namespace area_of_overlap_l483_483982

theorem area_of_overlap 
  (len1 len2 : ℕ) (area_left only_left_area : ℚ) (area_right only_right_area : ℚ) (w : ℚ)
  (h_len1 : len1 = 9) (h_len2 : len2 = 7) (h_only_left_area : only_left_area = 27) 
  (h_only_right_area : only_right_area = 18) (h_w : w > 0)
  (h_area_left : area_left = only_left_area + (w * 1))
  (h_area_right : area_right = only_right_area + (w * 1))
  (h_ratio : (w * len1) / (w * len2) = 9 / 7) : 
  (13.5) :=
by
  sorry

end area_of_overlap_l483_483982


namespace product_of_possible_values_of_c_l483_483248

-- Definitions based on conditions
def g (c : ℝ) (x : ℝ) : ℝ := c / (3 * x - 5)

theorem product_of_possible_values_of_c :
  ( ∀ (c : ℝ), g c 3 = (λ y, g y (c+2)) (c) ) → 
  ∏ (c ∈ {x | x^2 + 7*x + 2 = 0 }) = 2 :=
by
  sorry

end product_of_possible_values_of_c_l483_483248


namespace correct_area_correct_sum_l483_483950

def VennDiagramArea : ℝ :=
  let radius := 1
  let area_circle := π * radius^2
  let sector_angle := π / 3
  let area_triangle := (sqrt 3 / 4) * radius^2
  let area_sector := (1 / 6) * π * radius^2
  let area_lens := 2 * (area_sector - area_triangle)
  let total_overlapping_area := 3 * (area_lens) / 2
  3 * area_circle - total_overlapping_area + area_triangle

theorem correct_area : 
  2 * π + 7 * sqrt 3 / 4 = VennDiagramArea :=
sorry

theorem correct_sum : 
  let a := 2
  let b := 1
  let c := 21
  a + b + c = 24 :=
rfl

end correct_area_correct_sum_l483_483950


namespace ping_pong_team_selection_l483_483240

theorem ping_pong_team_selection :
  ∃ (team_A team_B : Fin 1000 → Type) (players : Fin 10 → Fin 1000),
  (∀ i j, team_A i → team_B j → Bool) →
  (∀ b : Fin 1000, ∃ a : Fin 10, ∃ i : Fin 1000, players a = i ∧ ¬ team_B b (team_A i)) →
  true :=
by sorry

end ping_pong_team_selection_l483_483240


namespace absolute_value_diff_is_zero_l483_483611

-- Definitions of conditions
def A : ℕ := 1
def B : ℕ := 1

-- Expressing the computed absolute value and result
theorem absolute_value_diff_is_zero (A B : ℕ) (hA : A = 1) (hB : B = 1) : 
  abs (A - B) = 0 := by
  rw [hA, hB]
  exact abs_zero

#eval absolute_value_diff_is_zero A B rfl rfl -- This should evaluate to true confirming the theorem

end absolute_value_diff_is_zero_l483_483611


namespace midpoint_polar_coordinates_example_l483_483362
noncomputable def polar_midpoint (r1 θ1 r2 θ2 : ℝ) : ℝ × ℝ :=
  let A := (r1 * Real.cos θ1, r1 * Real.sin θ1)
  let B := (r2 * Real.cos θ2, r2 * Real.sin θ2)
  let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let r := Real.sqrt (M.1^2 + M.2^2)
  let θ := Real.atan2 M.2 M.1
  (r, θ)

theorem midpoint_polar_coordinates_example :
  polar_midpoint 6 (Real.pi / 4) 6 (3 * Real.pi / 4) = (3 * Real.sqrt 2, Real.pi / 2) :=
by
  sorry

end midpoint_polar_coordinates_example_l483_483362


namespace product_divisibility_l483_483290

theorem product_divisibility
  (k m n : ℕ)
  (hk : k < 0) (hm : m < 0) (hn : n < 0)
  (prime_cond : Nat.Prime (m + k + 1))
  (greater_cond : m + k + 1 > n + 1)
  (C : ℕ → ℕ := λ s, s * (s + 1)) :
  (∏ i in finset.range n, (C (m + i + 1) - C k)) % (∏ i in finset.range n, C (i + 1)) = 0 :=
by
  sorry

end product_divisibility_l483_483290


namespace multiplication_to_squares_l483_483753

theorem multiplication_to_squares :
  85 * 135 = 85^2 + 50^2 + 35^2 + 15^2 + 15^2 + 5^2 + 5^2 + 5^2 :=
by
  sorry

end multiplication_to_squares_l483_483753


namespace john_children_l483_483026

def total_notebooks (john_notebooks : ℕ) (wife_notebooks : ℕ) (children : ℕ) := 
  2 * children + 5 * children

theorem john_children (c : ℕ) (h : total_notebooks 2 5 c = 21) :
  c = 3 :=
sorry

end john_children_l483_483026


namespace valid_arrangements_count_l483_483001

-- Define the names of the individuals and the problem constraints
def names : List String := ["Wilma", "Paul", "Adam", "Betty", "Charlie", "D", "E", "F"]

-- The final number of valid arrangements
theorem valid_arrangements_count : 
  let total := (8!).toNat
  let WilmaPaulTogether := (7!).toNat * (2!).toNat
  let ABCtogether := (6!).toNat * (3!).toNat
  let bothTogether := (5!).toNat * (2!).toNat * (3!).toNat
  total - (WilmaPaulTogether + ABCtogether) + bothTogether = 25360 :=
by
  -- Decompose the problem into subcalculations
  let total := (8!).toNat
  let WilmaPaulTogether := (7!).toNat * (2!).toNat
  let ABCtogether := (6!).toNat * (3!).toNat
  let bothTogether := (5!).toNat * (2!).toNat * (3!).toNat
  sorry

end valid_arrangements_count_l483_483001


namespace largest_quantity_l483_483600

def A : ℝ := 2010 / 2009 + 2010 / 2011
def B : ℝ := 2010 / 2011 + 2012 / 2011
def C : ℝ := 2011 / 2010 + 2011 / 2012

theorem largest_quantity : A > B ∧ A > C := 
by {
  sorry
}

end largest_quantity_l483_483600


namespace smallest_arithmetic_mean_divisible_by_1111_l483_483820

theorem smallest_arithmetic_mean_divisible_by_1111 :
  ∃ (n : ℕ), (∀ i : ℕ, i < 9 → 1111 ∣ (n + i)) ∧ (n + 4 = 97) :=
by
  sorry

end smallest_arithmetic_mean_divisible_by_1111_l483_483820


namespace second_player_wins_l483_483435

theorem second_player_wins (n : ℕ) (h : 2 ≤ n) :
  ∃ (strategy : Π (points : set ℝ) (remaining_moves : ℕ), set ℝ),
    ∀ (player : ℕ) (points : set ℝ) (remaining_moves : ℕ),
      strategy points remaining_moves ∈ points ∧
      (∀ (point : ℝ), point ∈ points → obtuse_triangle_for_all_points (points \ {point}) → player = 1) :=
sorry

-- Definitions to outline the necessary concepts and conditions in the problem: 

def obtuse_triangle (a b c : ℝ) : Prop := 
  (* Define what it means for a triangle with vertices a, b, and c to be obtuse *)

def obtuse_triangle_for_all_points (points : set ℝ) : Prop :=
  ∀ (a b c : ℝ), a ∈ points → b ∈ points → c ∈ points → obtuse_triangle a b c

end second_player_wins_l483_483435


namespace average_speed_round_trip_l483_483505

def average_speed (d : ℝ) (v1 : ℝ) (v2 : ℝ) : ℝ := 
  let t1 := d / v1
  let t2 := d / v2
  let total_time := t1 + t2
  let total_distance := d * 2
  total_distance / total_time

theorem average_speed_round_trip : average_speed 1 40 60 = 48 := by
  sorry

end average_speed_round_trip_l483_483505


namespace total_skips_completed_l483_483458

def skips_completed (rate : ℕ) (time : ℕ) (time_unit : ℕ) : ℕ :=
  (rate / time_unit) * time

theorem total_skips_completed :
  let roberto_rate := 4200 in
  let roberto_time := 15 in
  let roberto_time_unit := 60 in

  let valerie_rate := 80 in
  let valerie_time := 10 in
  let valerie_time_unit := 1 in

  let lucas_rate := 150 in
  let lucas_time := 20 in
  let lucas_time_unit := 5 in

  skips_completed roberto_rate roberto_time roberto_time_unit +
  skips_completed valerie_rate valerie_time valerie_time_unit +
  skips_completed lucas_rate lucas_time lucas_time_unit = 2450 :=
by
  sorry

end total_skips_completed_l483_483458


namespace panda_bamboo_digestion_l483_483439

theorem panda_bamboo_digestion (h : 16 = 0.40 * x) : x = 40 :=
by sorry

end panda_bamboo_digestion_l483_483439


namespace find_m_and_n_l483_483533

-- Define the situation given in the problem.
def is_sum_of_proper_fractions_equal_5 (m n : ℕ) : Prop :=
  (∀ k ∈ finset.range m, 1 ≤ k ∧ k < m) →
  (∀ k ∈ finset.range(n), 1 ≤ k ∧ k < n) →
  let A := ((finset.range(m)).sum (λ k, k : ℚ)) / (2 * m) in
  let B := ((finset.range(n)).sum (λ k, k : ℚ)) / (2 * n) in
  (A * B = 5)

-- The main theorem which states the provided solution.
theorem find_m_and_n (m n : ℕ) (hm : m.prime) (hn : n.prime) (h : m < n) :
  is_sum_of_proper_fractions_equal_5 m n → (m = 3 ∧ n = 11) :=
by
  sorry

end find_m_and_n_l483_483533


namespace hall_breadth_is_12_l483_483551

/-- Given a hall with length 15 meters, if the sum of the areas of the floor and the ceiling 
    is equal to the sum of the areas of the four walls and the volume of the hall is 1200 
    cubic meters, then the breadth of the hall is 12 meters. -/
theorem hall_breadth_is_12 (b h : ℝ) (h1 : 15 * b * h = 1200)
  (h2 : 2 * (15 * b) = 2 * (15 * h) + 2 * (b * h)) : b = 12 :=
sorry

end hall_breadth_is_12_l483_483551


namespace area_of_circle_with_radius_5_l483_483124

-- We define the radius of the circle
def radius : ℝ := 5

-- We define the expected area of the circle
def expected_area : ℝ := 25 * Real.pi

-- The theorem statement: The area of a circle with radius 5 meters is 25π square meters.
theorem area_of_circle_with_radius_5 : Real.pi * radius^2 = expected_area := by
  sorry

end area_of_circle_with_radius_5_l483_483124


namespace square_mirror_side_length_l483_483181

noncomputable def side_length_of_mirror
  (width_wall : ℝ) 
  (length_wall : ℝ) 
  (area_mirror : ℝ) : ℝ :=
  Real.sqrt area_mirror

theorem square_mirror_side_length
  (width_wall : ℝ := 54)
  (length_wall : ℝ := 42.81481481481482) :
  (let area_wall := width_wall * length_wall,
       area_mirror := area_wall / 2,
       side_length := side_length_of_mirror width_wall length_wall area_mirror in
  side_length = 34) :=
by
  sorry

end square_mirror_side_length_l483_483181


namespace condition_for_equal_distances_l483_483508

variable {A B C D K L M N : Type}
variable [add_comm_group A] [vector_space ℝ A]
variable [add_comm_group B] [vector_space ℝ B]
variable [add_comm_group C] [vector_space ℝ C]
variable [add_comm_group D] [vector_space ℝ D]
variable [affine_space A A] [affine_space B B] [affine_space C C] [affine_space D D]

-- Define midpoints K, L, M, N of quadrilateral ABCD
def midpoint (x y : point) := ( x +ᵥ y) / 2

variable (ABCD: quadrilateral A B C D)
variable (K : point) (L : point) (M : point) (N : point)

-- K is the midpoint of AB etc
axiom K_is_midpoint : midpoint A B = K
axiom L_is_midpoint : midpoint B C = L
axiom M_is_midpoint : midpoint C D = M
axiom N_is_midpoint : midpoint D A = N

-- Quadrilateral KLMN formed by midpoints
def KLMN : quadrilateral K L M N := sorry

-- KLMN is a parallelogram
axiom KLMN_is_parallelogram : parallelogram K L M N

-- The quadrilateral KLMN is a parallelogram and sides are parallel to diagonals
axiom KLMN_parallels_diagonals :
  (is_parallel K L A C) ∧ (is_parallel L M B D) ∧ (is_parallel M N A C) ∧ (is_parallel N K B D)

-- Rectangular condition in terms of perpendicular diagonals
theorem condition_for_equal_distances :
  (ABCD: quadrilateral A B C D) → (distances_equal_midpoints K L M N) ↔ (diagonals_perpendicular A C B D) :=
sorry

end condition_for_equal_distances_l483_483508


namespace selling_price_correct_l483_483401

/-- Define the initial cost of the gaming PC. -/
def initial_pc_cost : ℝ := 1200

/-- Define the cost of the new video card. -/
def new_video_card_cost : ℝ := 500

/-- Define the total spending after selling the old card. -/
def total_spending : ℝ := 1400

/-- Define the selling price of the old card -/
def selling_price_of_old_card : ℝ := (initial_pc_cost + new_video_card_cost) - total_spending

/-- Prove that John sold the old card for $300. -/
theorem selling_price_correct : selling_price_of_old_card = 300 := by
  sorry

end selling_price_correct_l483_483401


namespace equidistant_points_count_l483_483380

theorem equidistant_points_count :
  ∀ (l1 l2 l3 : ℝ → ℝ), 
    (∃ p1 p2 : ℝ × ℝ, l1 = p1.1 ∧ l2 = p2.1 ∧ ∃ p : ℝ × ℝ, l2(p.fst) = l1(p.fst) ∧ ∃ d, ∀ x, ‖l3 x - l1 x‖ = d ∧ ∃ p1 p2, l2(p1.fst) = l2(p2.fst)) → 
    (∃ points : list (ℝ × ℝ), points.length = 2 ∧ ∀ p ∈ points, equidistant_from_lines p l1 l2 l3) :=
by sorry

noncomputable def equidistant_from_lines (p : ℝ × ℝ) (l1 l2 l3 : ℝ → ℝ) :=
  ∀ (d1 d2 d3 : ℝ), (d1 = ‖p.2 - l1(p.1)‖) ∧ (d2 = ‖p.2 - l2(p.1)‖) ∧ (d3 = ‖p.2 - l3(p.1)‖) ∧ (d1 = d2 ∧ d2 = d3 ∧ d1 = d3)


end equidistant_points_count_l483_483380


namespace median_of_sequence_is_142_l483_483382

-- Definition of the sequence where each number n (1 ≤ n ≤ 200) appears n times consecutively
def sequence : List ℕ :=
  List.bind (List.range' 1 200) (λ n, List.replicate n n)

-- Function to calculate the median of a given list of natural numbers
def median (l : List ℕ) : ℕ :=
  let sorted := l.qsort (≤)
  let len := sorted.length
  if len % 2 == 0 then
    (sorted.get! (len / 2 - 1) + sorted.get! (len / 2)) / 2
  else
    sorted.get! (len / 2)

-- Prove that the median of the sequence is 142
theorem median_of_sequence_is_142 : median sequence = 142 :=
by 
  -- The proof would go here (this is skipped per instructions).
  sorry

end median_of_sequence_is_142_l483_483382


namespace total_books_l483_483499

-- Define the number of books Tim has
def TimBooks : ℕ := 44

-- Define the number of books Sam has
def SamBooks : ℕ := 52

-- Statement to prove that the total number of books is 96
theorem total_books : TimBooks + SamBooks = 96 := by
  sorry

end total_books_l483_483499


namespace find_F_l483_483714

theorem find_F (F C : ℝ) (h1 : C = 35) (h2 : C = (7/12) * (F - 40)) : F = 100 :=
by
  sorry

end find_F_l483_483714


namespace smallest_arithmetic_mean_divisible_1111_l483_483849

theorem smallest_arithmetic_mean_divisible_1111 :
  ∃ n : ℕ, 93 ≤ n ∧ n + 4 = 97 ∧ (∀ i : ℕ, i ∈ finset.range 9 → (n + i) % 11 = 0 ∨ (n + i) % 101 = 0) :=
sorry

end smallest_arithmetic_mean_divisible_1111_l483_483849


namespace journey_duration_is_9_hours_l483_483113

noncomputable def journey_time : ℝ :=
  let d1 := 90 -- Distance traveled by Tom and Dick by car before Tom got off
  let d2 := 60 -- Distance Dick backtracked to pick up Harry
  let T := (d1 / 30) + ((120 - d1) / 5) -- Time taken for Tom's journey
  T

theorem journey_duration_is_9_hours : journey_time = 9 := 
by 
  sorry

end journey_duration_is_9_hours_l483_483113


namespace equation_of_line_B_T_l483_483318

theorem equation_of_line_B_T :
  let ellipse := ∀ x y, x^2 / 25 + y^2 / 9 = 1,
      A := (x1, y1),
      B := (4, 9/5),
      C := (x2, y2),
      F := (4, 0),
      distance := λ p1 p2, Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2),
      d_AF := distance A F,
      d_BF := distance B F,
      d_CF := distance C F,
      arithmetic_sequence := d_AF + d_CF = 2 * d_BF,
      intersect_x_axis := λ p1 p2, ∃ T, T.2 = 0 ∧ ∃ D, D.1 = (p1.1 + p2.1) / 2 ∧ D.2 = (p1.2 + p2.2) / 2 ∧ T.1 = slope_intercept_form(B, D) := -- intersection with x-axis
  intercept_line (proof_eq_line (intersect_x_axis A C)) ((4, 9/5), BT) := (25, -20, 64) :=
begin
  sorry,
end

end equation_of_line_B_T_l483_483318


namespace harriet_return_speed_l483_483132

theorem harriet_return_speed {t1 t2 t_total hours_to_minutes : ℝ}
  (h_speed_AV_to_BT : 110) 
  (h_total_time : t_total = 5)
  (h_time_AV_to_BT : t1 = 168 / hours_to_minutes)
  (h_t1_in_hours : t1 = 2.8)
  (h_distance_is_same : ∀ d, d = h_speed_AV_to_BT * t1)
  (h_time_BT_to_AV : t2 = t_total - t1) :
  let speed_BT_to_AV := (308 / t2 : ℝ) in
  speed_BT_to_AV = 140 :=
by
  sorry

end harriet_return_speed_l483_483132


namespace value_of_n_l483_483255

-- Define the eight-digit number in terms of its digits
def digits : List ℕ := [9, 6, 7, 3, n, 4, 3, 2]

-- Specify the condition for 9 divisibility
theorem value_of_n (n : ℕ) : (967300002 + n * 10000 + 43) % 9 = 0 ↔ n = 2 := by
  sorry

end value_of_n_l483_483255


namespace prime_divides_a_squared_plus_3_l483_483715

theorem prime_divides_a_squared_plus_3 (p a : ℕ) [Prime p] (h : p ∣ a^2 + 3) : 
  p = 2 ∨ p = 3 ∨ ∃ k : ℕ, p = 3 * k + 1 :=
by
  sorry 

end prime_divides_a_squared_plus_3_l483_483715


namespace half_vector_AB_l483_483637

-- Define vectors MA and MB
def MA : ℝ × ℝ := (-2, 4)
def MB : ℝ × ℝ := (2, 6)

-- Define the proof statement 
theorem half_vector_AB : (1 / 2 : ℝ) • (MB - MA) = (2, 1) :=
by sorry

end half_vector_AB_l483_483637


namespace moles_HCl_involved_l483_483617

-- Define the chemical elements as constants
constant NaHCO3 : Type
constant HCl : Type
constant NaCl : Type
constant H2O : Type
constant CO2 : Type

-- Define the number of moles of each substance involved in the reaction
constant moles_HCl : ℕ
constant moles_NaHCO3 : ℕ := 1
constant moles_CO2 : ℕ := 1

-- Define the chemical reaction
def reaction (a : ℕ) (b : ℕ) (c : ℕ) : Prop :=
  a = b ∧ b = c

-- Define the proof problem
theorem moles_HCl_involved : reaction moles_NaHCO3 moles_HCl moles_CO2 → moles_HCl = 1 :=
by
  sorry

end moles_HCl_involved_l483_483617


namespace ethan_siblings_product_l483_483604

theorem ethan_siblings_product
  (erins_sisters : ℕ)
  (erins_brothers : ℕ)
  (ethan_sisters : erins_sisters + 1)
  (ethan_brothers : erins_brothers - 1)
  (s : ethan_sisters = 5)
  (b : ethan_brothers = 6) :
  s * b = 30 := by
  sorry

end ethan_siblings_product_l483_483604


namespace smallest_arithmetic_mean_divisible_by_1111_l483_483850

/-- 
Given the product of nine consecutive natural numbers is divisible by 1111, 
prove that the smallest possible value of the arithmetic mean of these nine numbers is 97.
-/
theorem smallest_arithmetic_mean_divisible_by_1111 :
  ∃ n : ℕ, (∀ k : ℕ, k = n →  (∏ i in finset.range 9, k + i) % 1111 = 0) 
  ∧ (n ≥ 93) ∧ (n + 4 = 97) :=
sorry

end smallest_arithmetic_mean_divisible_by_1111_l483_483850


namespace percent_profit_l483_483988

theorem percent_profit (cost : ℝ) (markup_percent : ℝ) (discount_percent : ℝ) (final_profit_percent : ℝ)
  (h1 : cost = 50)
  (h2 : markup_percent = 30)
  (h3 : discount_percent = 10)
  (h4 : final_profit_percent = 17)
  : (markup_percent / 100 * cost - discount_percent / 100 * (cost + markup_percent / 100 * cost)) / cost * 100 = final_profit_percent := 
by
  sorry

end percent_profit_l483_483988


namespace find_new_basis_coordinates_l483_483682

variables {V : Type*} [inner_product_space ℝ V]

-- Conditions
variables (a b c m : V) 
variable (h_orthonormal : orthonormal_basis ℝ V ![a, b, c])
variable (h_coords_m : coordinates (basis.mk_of_vector_space ![a, b, c] h_orthonormal) m = ![1, 2, 3])

-- The translated proof problem
theorem find_new_basis_coordinates :
  (∀ (x y z : ℝ), m = x • (a + b) + y • (a - b) + z • c → ⟨x, y, z⟩ = ⟨3/2, -1/2, 3⟩) :=
sorry

end find_new_basis_coordinates_l483_483682


namespace count_non_decreasing_maps_l483_483810

theorem count_non_decreasing_maps : 
  let S := {1, 2, 3}
  let T := {1, 2, 3, 4, 5}
  let f : S → T
  in (∀ i j : S, i ≤ j → f i ≤ f j) →
     ∃ (n : ℕ), n = 35 :=
by
  sorry

end count_non_decreasing_maps_l483_483810


namespace find_x_given_y_inversely_varies_l483_483102

theorem find_x_given_y_inversely_varies (x y k : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0)
  (h_inv : x ^ 3 * y = k) (h_given : 2 ^ 3 * 8 = k) (h_y : y = 1000) : x = 0.4 :=
by
  have h_k : k = 64 := by
    rw [←h_given]
  rw [h_y] at h_inv
  rw [h_k] at h_inv
  have h_x_cubed : x^3 * 1000 = 64 := by
    rw [h_inv]
  have h_x_cubed_eq : x^3 = 0.064 := by
    linarith
  have h_x : x = 0.4 := by
    rw [real.eq_rpow_iff h_pos_x]
    norm_num at h_x_cubed_eq
    rw [←h_x_cubed_eq]
    norm_num
  exact h_x

end find_x_given_y_inversely_varies_l483_483102


namespace product_n_equals_7200_l483_483605

theorem product_n_equals_7200 :
  (3 - 1) * 3 * (3 + 1) * (3 + 2) * (3 + 3) * (3 ^ 2 + 1) = 7200 := by
  sorry

end product_n_equals_7200_l483_483605


namespace problem_statement_l483_483246

def A : ℝ := ∑' n in (Nat.filter (λ n, n % 3 ≠ 0)), (-1)^((n-1)/2) * (1/n^3)
def B : ℝ := ∑' n in (Nat.filter (λ n, n % 3 = 0)), (-1)^((3*n-1)/2) * (1/n^3)

theorem problem_statement : A / B = 28 := 
by {
  -- proof will be provided here
  sorry
}

end problem_statement_l483_483246


namespace length_of_paving_stone_l483_483541

def courtyard_length : ℝ := 30
def courtyard_width : ℝ := 16.5
def num_paving_stones : ℝ := 99
def width_of_paving_stone : ℝ := 2
def area_of_courtyard : ℝ := courtyard_length * courtyard_width := by sorry

theorem length_of_paving_stone :
  let L := (courtyard_length * courtyard_width) / (num_paving_stones * width_of_paving_stone)
  L = 2.5 :=
by sorry

end length_of_paving_stone_l483_483541


namespace cody_books_second_week_l483_483586

noncomputable def total_books := 54
noncomputable def books_first_week := 6
noncomputable def books_weeks_after_second := 9
noncomputable def total_weeks := 7

theorem cody_books_second_week :
  let b2 := total_books - (books_first_week + books_weeks_after_second * (total_weeks - 2))
  b2 = 3 :=
by
  sorry

end cody_books_second_week_l483_483586


namespace molecular_weight_calc_l483_483905

-- Definitions
def num_Al := 1
def num_O := 3
def num_H := 3
def weight_Al := 26.98
def weight_O := 16.00
def weight_H := 1.01

-- Theorem statement
theorem molecular_weight_calc :
  num_Al * weight_Al + num_O * weight_O + num_H * weight_H = 78.01 := 
  sorry

end molecular_weight_calc_l483_483905


namespace smallest_arithmetic_mean_divisible_product_l483_483834

theorem smallest_arithmetic_mean_divisible_product :
  ∃ (n : ℕ), (∀ i : ℕ, i < 9 → Nat.gcd ((n + i) * 1111) 1111 = 1111) ∧ (n + 4 = 97) := 
  sorry

end smallest_arithmetic_mean_divisible_product_l483_483834


namespace triangles_similar_l483_483782

theorem triangles_similar (a b c : ℝ) (h₁ : a + c = 2 * b) (h₂ : b + 2 * c = 5 * a) :
  (a / b = 5 / 7 ∧ b / c = 7 / 9 ∧ a / c = 5 / 9) → (∀ (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ), 
  (a₁ / b₁ = a / b ∧ b₁ / c₁ = b / c ∧ a₁ / c₁ = a / c) ↔ (a₁ / a₂ = b₁ / b₂ = c₁ / c₂)) := 
by
  sorry

end triangles_similar_l483_483782


namespace major_axis_length_l483_483959

def point := (ℝ, ℝ)

noncomputable def length_of_major_axis (f1 f2 : point) (tangent_x tangent_y : Bool) : ℝ :=
  if tangent_x && tangent_y && f1.1 = f2.1 && f1.2 - f2.2 = 2 * real.sqrt 8 then 
    8 
  else 
    0  -- Placeholder for the incorrect case

theorem major_axis_length : length_of_major_axis (5, -4 + real.sqrt 8) (5, -4 - real.sqrt 8) true true = 8 :=
by
  sorry

end major_axis_length_l483_483959


namespace first_successful_powered_flight_l483_483358

theorem first_successful_powered_flight :
  (∃ brothers, brothers = "Wright Brothers" ∧ achieved_flight_in_dec_1903 brothers "Flyer 1") ↔ ("Wright Brothers" = "Wright Brothers") :=
by
  apply iff.intro;
  intros h;
  sorry

end first_successful_powered_flight_l483_483358


namespace product_equals_zero_l483_483804

theorem product_equals_zero 
    (a : ℤ) (x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8 x_9 x_10 x_11 x_12 x_13 : ℤ) 
    (h : a = (1 + x_1) * (1 + x_2) * (1 + x_3) * (1 + x_4) * (1 + x_5) * (1 + x_6) * (1 + x_7) * (1 + x_8) * (1 + x_9) * (1 + x_10) * (1 + x_11) * (1 + x_12) * (1 + x_13) 
         ∧ a = (1 - x_1) * (1 - x_2) * (1 - x_3) * (1 - x_4) * (1 - x_5) * (1 - x_6) * (1 - x_7) * (1 - x_8) * (1 - x_9) * (1 - x_10) * (1 - x_11) * (1 - x_12) * (1 - x_{13})) :
    a * x_1 * x_2 * x_3 * x_4 * x_5 * x_6 * x_7 * x_8 * x_9 * x_10 * x_11 * x_12 * x_13 = 0 := 
sorry

end product_equals_zero_l483_483804


namespace no_intersection_pair_C_l483_483257

theorem no_intersection_pair_C :
  let y1 := fun x : ℝ => x
  let y2 := fun x : ℝ => x - 3
  ∀ x : ℝ, y1 x ≠ y2 x :=
by
  sorry

end no_intersection_pair_C_l483_483257


namespace correct_statement_l483_483930

-- Definitions of the conditions
def condition_A : Prop := ¬(Pie charts can clearly reflect the trend of things)
def condition_B : Prop := ¬(The service life of a certain model of electronic product is investigated comprehensively)
def condition_C : Prop := ¬(If the probability of winning a game is 1/5, then playing this game 5 times will definitely result in a win)
def condition_D : Prop := 
  let S_A² := 0.2
  let S_B² := 0.03
  (The average of two sets of data, Group A and Group B, is equal) → Group B is more stable than Group A

-- The main statement to prove
theorem correct_statement : (condition_A ∧ condition_B ∧ condition_C ∧ condition_D) → (correct answer = "D") :=
  by
    sorry -- Placeholder for the proof

end correct_statement_l483_483930


namespace distances_equality_l483_483296

open EuclideanGeometry

variable {A B C D M P : Point}

-- Given conditions
axiom trapezoid (h1 : is_trapezoid A B C D) : AD ∥ BC
axiom diagonal_intersection (h2 : M ∈ Line_ AC ∧ M ∈ Line_ BD)
axiom point_on_segment (h3 : P ∈ segment_ BC)
axiom angle_equality (h4 : ∠APM = ∠DPM)

-- Prove: distance from B to line DP = distance from C to line AP
theorem distances_equality (h1 : is_trapezoid A B C D) (h2 : M ∈ Line_ AC ∧ M ∈ Line_ BD)
  (h3 : P ∈ segment_ BC) (h4 : ∠APM = ∠DPM) : 
  distance B (Line_ DP) = distance C (Line_ AP) := 
sorry

end distances_equality_l483_483296


namespace jake_balloon_count_l483_483995

theorem jake_balloon_count :
  ∀ (initial_allan_balloons additional_allan_balloons : ℕ) (initial_allan_balloons = 2) (additional_allan_balloons = 3), 
  let total_allan_balloons := initial_allan_balloons + additional_allan_balloons in
  let jake_balloons := total_allan_balloons + 1 in
  jake_balloons = 6 :=
by
  intros
  unfold let total_allan_balloons jake_balloons
  sorry

end jake_balloon_count_l483_483995


namespace telephone_number_A_value_l483_483185

theorem telephone_number_A_value :
  ∃ A B C D E F G H I J : ℕ,
    A > B ∧ B > C ∧
    D > E ∧ E > F ∧
    G > H ∧ H > I ∧ I > J ∧
    (D = E + 1) ∧ (E = F + 1) ∧
    G + H + I + J = 20 ∧
    A + B + C = 15 ∧
    A = 8 := sorry

end telephone_number_A_value_l483_483185


namespace smallest_arithmetic_mean_l483_483881

noncomputable def S (n : ℕ) := (List.range' n 9).map Nat.ofNat

theorem smallest_arithmetic_mean (n : ℕ) (h1 : 93 ≤ n) (h2 : ∃ k ∈ S n, 11 ∣ k) (h3 : ∃ k ∈ S n, 101 ∣ k) : 
  (n + 4 = 97) := by
  sorry

end smallest_arithmetic_mean_l483_483881


namespace max_value_expr_l483_483072

noncomputable theory
open_locale classical

theorem max_value_expr
  (a b c : ℝ)
  (x₁ x₂ x₃ : ℝ)
  (λ : ℝ)
  (h_poly : ∀ (x : ℝ), x^3 + a * x^2 + b * x + c = 0)
  (h_lambda_pos : λ > 0)
  (h_roots : x_2 - x₁ = λ)
  (h_cond : x₃ > (1/2) * (x₁ + x₂)) :
  (∃ t, (t = \frac{3 * sqrt 3}{2}) ∧ ∀ v, v = \frac{2 * a^3 + 27 * c - 9 * a * b}{λ^3} -> v <= t) :=
sorry

end max_value_expr_l483_483072


namespace concurrency_DI_l483_483442

variable {A B C P Q : Type}
variables {I I' F F' E E' H H' G G' D D' : Point P Q}

-- Assuming definitions for three points to be concurrent if they meet at one point
def concurrent (a b c : Line) : Prop :=
  ∃ X : Point, X ∈ a ∧ X ∈ b ∧ X ∈ c

-- The conditions provided in the problem
variables [Geometry P Q]
variables (de fg hi : Line)
variables (di' e'f' g'h' : Line)

-- Assume the given conditions
axiom concurrent_de_fg_hi : concurrent de fg hi
axiom BI'_eq_CI : dist B I' = dist C I
axiom BF'_eq_CF : dist B F' = dist C F
axiom CE'_eq_AE : dist C E' = dist A E
axiom CH'_eq_AH : dist C H' = dist A H
axiom AD'_eq_BD : dist A D' = dist B D
axiom AG'_eq_BG : dist A G' = dist B G

-- Prove that D'I', E'F', and G'H' are concurrent
theorem concurrency_DI'_EI'_GH' : concurrent di' e'f' g'h' :=
sorry

end concurrency_DI_l483_483442


namespace angle_ADC_correct_l483_483014

noncomputable def angle_ADC_measure (A B C D : Type) [Triangle A B C] [Bisector A D] [Bisector D C] : Angle :=
  let α : Angle := 40   -- ∠BAC
  let β : Angle := 70   -- ∠ABC
  let γ : Angle := 180 - α - β -- ∠ACB because the sum of angles in triangle ABC is 180°
  let δ := γ / 2        -- ∠ACD = ∠BCD due to angle bisector DC
  let ε := 20           -- ∠BAD = ∠CAD due to angle bisector AD
  180 - (ε + δ)         -- ∠ADC is calculated using the angle sum property of triangle ADC

theorem angle_ADC_correct (A B C D : Type) [Triangle A B C] [Bisector A D] [Bisector D C]
  (h1 : angle_BAC A B C = 40) (h2 : angle_ABC A B C = 70)
  (h3 : angle_BAD A B = 20) (h4 : angle_DCA D C A = γ / 2) : 
  angle_ADC A D C = 125 := by
  let α := 40
  let β := 70
  let γ := 180 - α - β
  let δ := γ / 2
  let ε := 20
  let φ := 180 - (ε + δ)
  have h5 : φ = 125 := sorry
  exact h5

end angle_ADC_correct_l483_483014


namespace area_of_region_l483_483507

-- Define the equation as a predicate
def region (x y : ℝ) : Prop := x^2 + y^2 + 6*x = 2*y + 10

-- The proof statement
theorem area_of_region : (∃ (x y : ℝ), region x y) → ∃ A : ℝ, A = 20 * Real.pi :=
by 
  sorry

end area_of_region_l483_483507


namespace forty_percent_of_n_l483_483774

theorem forty_percent_of_n (N : ℝ) (h : (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 10) : 0.40 * N = 120 := by
  sorry

end forty_percent_of_n_l483_483774


namespace coat_total_selling_price_l483_483165

theorem coat_total_selling_price :
  let original_price := 120
  let discount_percent := 30
  let tax_percent := 8
  let discount_amount := (discount_percent / 100) * original_price
  let sale_price := original_price - discount_amount
  let tax_amount := (tax_percent / 100) * sale_price
  let total_selling_price := sale_price + tax_amount
  total_selling_price = 90.72 :=
by
  sorry

end coat_total_selling_price_l483_483165


namespace count_valid_points_l483_483985

-- Define the points C and D
def C : ℤ × ℤ := (2, 3)
def D : ℤ × ℤ := (-2, -3)

-- Define the condition for a point (x, y) to be on a valid path
def isValidPath (x y : ℤ) : Prop :=
  |(x - 2)| + |(x + 2)| + |(y - 3)| + |(y + 3)| ≤ 22

-- Prove that the number of points with integer coordinates on at least one valid path is 201
theorem count_valid_points : { p : ℤ × ℤ // isValidPath p.1 p.2 }.set.card = 201 :=
begin
  sorry
end

end count_valid_points_l483_483985


namespace simplify_trig_expr_find_tan_alpha_l483_483150

-- Problem 1: Simplify the expression
theorem simplify_trig_expr (α : ℝ) :
  (sin (π - α) * cos (π + α) * sin (π / 2 + α)) / (sin (-α) * sin (3 * π / 2 + α)) = -cos α :=
by
  sorry

-- Problem 2: Find tan α
theorem find_tan_alpha (α : ℝ) (h0 : π / 2 < α) (h1 : α < π) (h2 : sin (π - α) + cos α = 7 / 13) :
  tan α = -12 / 5 :=
by
  sorry

end simplify_trig_expr_find_tan_alpha_l483_483150


namespace ratio_of_ages_l483_483542

theorem ratio_of_ages (x m : ℕ) 
  (mother_current_age : ℕ := 41) 
  (daughter_current_age : ℕ := 23) 
  (age_diff : ℕ := mother_current_age - daughter_current_age) 
  (eq : (mother_current_age - x) = m * (daughter_current_age - x)) : 
  (41 - x) / (23 - x) = m :=
by
  -- Proof not required
  sorry

end ratio_of_ages_l483_483542


namespace probability_at_least_one_white_ball_l483_483104

theorem probability_at_least_one_white_ball
  (total_balls : ℕ)
  (white_balls : ℕ)
  (red_balls : ℕ)
  (total_combinations : ℕ)
  (combinations_at_least_one_white : ℕ)
  (probability : ℝ) :
  total_balls = 10 →
  white_balls = 8 →
  red_balls = 2 →
  total_combinations = Nat.choose 10 2 →
  combinations_at_least_one_white = Nat.choose 8 1 * Nat.choose 2 1 + Nat.choose 8 2 →
  probability = (combinations_at_least_one_white : ℝ) / (total_combinations : ℝ) →
  probability = 44 / 45 :=
by 
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3] at *
  rw [h4, h5] at *
  have h_total_combinations : total_combinations = 45 := by sorry
  have h_combinations_at_least_one_white : combinations_at_least_one_white = 44 := by sorry
  rw [h_total_combinations, h_combinations_at_least_one_white] at h6
  exact h6

end probability_at_least_one_white_ball_l483_483104


namespace num_factors_2310_l483_483343

theorem num_factors_2310 : 
  let n : ℕ := 2310 in
  number_of_factors n = 32 :=
by
  sorry

end num_factors_2310_l483_483343


namespace sum_of_naturals_l483_483266

theorem sum_of_naturals (n : ℕ) : (finset.range (n + 1)).sum id = n * (n + 1) / 2 := by
  sorry

end sum_of_naturals_l483_483266


namespace initial_clothing_count_l483_483189

theorem initial_clothing_count 
  (donated_first : ℕ) 
  (donated_second : ℕ) 
  (thrown_away : ℕ) 
  (remaining : ℕ) 
  (h1 : donated_first = 5) 
  (h2 : donated_second = 3 * donated_first) 
  (h3 : thrown_away = 15) 
  (h4 : remaining = 65) :
  donated_first + donated_second + thrown_away + remaining = 100 :=
by
  sorry

end initial_clothing_count_l483_483189


namespace intersection_with_xz_plane_l483_483277

-- Define the points through which the line passes
def point1 : ℝ × ℝ × ℝ := (2, 3, 1)
def point2 : ℝ × ℝ × ℝ := (6, 0, 7)

-- Define the direction vector of the line
def direction_vector (p1 p2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p2.1 - p1.1, p2.2 - p1.2, p2.3 - p1.3)

-- Parameterize the line based on a parameter t
def line_eq (p1 : ℝ × ℝ × ℝ) (d : ℝ × ℝ × ℝ) (t : ℝ) : ℝ × ℝ × ℝ :=
  (p1.1 + t * d.1, p1.2 + t * d.2, p1.3 + t * d.3)

-- Prove that the intersection point with the xz-plane is (6, 0, 7)
theorem intersection_with_xz_plane : 
  let d := direction_vector point1 point2 in
  let intersect_t := 1 in
  line_eq point1 d intersect_t = (6, 0, 7) :=
by 
  let d := direction_vector point1 point2
  let intersect_t := 1
  show line_eq point1 d intersect_t = (6, 0, 7)
  sorry

end intersection_with_xz_plane_l483_483277


namespace count_gcd_21_eq_7_l483_483282

theorem count_gcd_21_eq_7 : 
  {n : ℕ | 1 ≤ n ∧ n ≤ 100 ∧ Int.gcd 21 n = 7}.to_finset.card = 10 :=
by
  sorry

end count_gcd_21_eq_7_l483_483282


namespace sum_even_integers_102_to_200_l483_483141

theorem sum_even_integers_102_to_200 :
  let sum_first_50_even := 2550 in
  let n := 50 in
  let first_term := 102 in
  let last_term := 200 in
  let common_difference := 2 in
  let num_terms := (last_term - first_term) / common_difference + 1 in
  num_terms = n ∧ (n / 2) * (first_term + last_term) = 7550 :=
by 
  let sum_first_50_even := 2550
  let n := 50
  let first_term := 102
  let last_term := 200
  let common_difference := 2
  let num_terms := (last_term - first_term) / common_difference + 1
  have h1 : num_terms = n, sorry
  have h2 : (n / 2) * (first_term + last_term) = 7550, sorry
  exact ⟨h1, h2⟩

end sum_even_integers_102_to_200_l483_483141


namespace min_arithmetic_mean_of_consecutive_naturals_div_by_1111_l483_483864

theorem min_arithmetic_mean_of_consecutive_naturals_div_by_1111 :
  ∀ a : ℕ, (∃ k, (a * (a + 1) * (a + 2) * (a + 3) * (a + 4) * (a + 5) * (a + 6) * (a + 7) * (a + 8)) = 1111 * k) →
  (a + 4 ≥ 97) :=
sorry

end min_arithmetic_mean_of_consecutive_naturals_div_by_1111_l483_483864


namespace smallest_arithmetic_mean_divisible_by_1111_l483_483821

theorem smallest_arithmetic_mean_divisible_by_1111 :
  ∃ (n : ℕ), (∀ i : ℕ, i < 9 → 1111 ∣ (n + i)) ∧ (n + 4 = 97) :=
by
  sorry

end smallest_arithmetic_mean_divisible_by_1111_l483_483821


namespace find_B_find_area_l483_483357

variables {a b c : ℝ} {A B C : ℝ}

noncomputable def part_I := 
    2 * b * Real.sin B = (2 * a + c) * Real.sin A + (2 * c + a) * Real.sin C

noncomputable def part_II (b : ℝ) (A : ℝ) := 
    b = Real.sqrt 3 ∧ A = Real.pi / 4

theorem find_B (h : part_I) : B = 2 * Real.pi / 3 :=
sorry

theorem find_area (h1 : part_II b A) (h2 : B = 2 * Real.pi / 3) : 
    (1 / 2) * b * (Real.sqrt (6) - Real.sqrt (2)) / 2 * Real.sin (Real.pi / 4) = (3 - Real.sqrt (3)) / 4 :=
sorry  

end find_B_find_area_l483_483357


namespace tori_math_test_l483_483114

theorem tori_math_test (total_problems : ℕ) (arithmetic_problems algebra_problems geometry_problems : ℕ)
  (correct_arithmetic correct_algebra correct_geometry : ℕ)
  (pass_percentage : ℚ) (required_correct_for_pass : ℕ) :
  total_problems = 75 →
  arithmetic_problems = 10 →
  algebra_problems = 30 →
  geometry_problems = 35 →
  correct_arithmetic = nat.floor (0.7 * arithmetic_problems) →
  correct_algebra = nat.floor (0.4 * algebra_problems) →
  correct_geometry = nat.floor (0.6 * geometry_problems) →
  pass_percentage = 0.6 →
  required_correct_for_pass = nat.floor (pass_percentage * total_problems) →
  required_correct_for_pass - (correct_arithmetic + correct_algebra + correct_geometry) = 5 :=
sorry

end tori_math_test_l483_483114


namespace divisors_not_multiples_of_15_l483_483040

noncomputable def smallest_m : ℕ :=
  2^(21) * 3^(13) * 7^(15)

theorem divisors_not_multiples_of_15 :
  let m := smallest_m in
  (m % 2 = 0 ∧ 
   ∃ k : ℕ, (m / 2 = k ^ 2) ∧ 
   m % 3 = 0 ∧ 
   ∃ l : ℕ, (m / 3 = l ^ 3) ∧ 
   m % 7 = 0 ∧ 
   ∃ n : ℕ, (m / 7 = n ^ 7)) →
  (finset.univ.filter (λ d, (m % d = 0) ∧ d % 15 ≠ 0)).card = 833 :=
by {
  sorry
}

end divisors_not_multiples_of_15_l483_483040


namespace bryan_more_than_ben_l483_483201

theorem bryan_more_than_ben :
  let Bryan_candies := 50
  let Ben_candies := 20
  Bryan_candies - Ben_candies = 30 :=
by
  let Bryan_candies := 50
  let Ben_candies := 20
  sorry

end bryan_more_than_ben_l483_483201


namespace no_positive_c_with_rational_roots_l483_483051

theorem no_positive_c_with_rational_roots :
  let p := 2^24036583 - 1 in
  ∀ (c : ℕ), (∀ (a b : ℤ),
  (p^2 - 4 * (c : ℤ) = a^2 ∧ p^2 + 4 * (c : ℤ) = b^2) →
  ¬(∃ c' : ℕ, c' = c)) :=
by
  let p := 2^24036583 - 1
  intro c a b
  sorry

end no_positive_c_with_rational_roots_l483_483051


namespace geese_ratio_l483_483455

/-- Define the problem conditions --/

def lily_ducks := 20
def lily_geese := 10

def rayden_ducks : ℕ := 3 * lily_ducks
def total_lily_animals := lily_ducks + lily_geese
def total_rayden_animals := total_lily_animals + 70
def rayden_geese := total_rayden_animals - rayden_ducks

/-- Prove the desired ratio of the number of geese Rayden bought to the number of geese Lily bought --/
theorem geese_ratio : rayden_geese / lily_geese = 4 :=
sorry

end geese_ratio_l483_483455


namespace keystone_arch_angle_l483_483089

theorem keystone_arch_angle :
  ∀ (n : ℕ) (x : ℕ),
  n = 10 →
  ((360 / n / 2) * 2 = 162) →
  (180 - (162 / 2) = x) →
  x = 99 :=
by
  intros n x hn hab hbc
  rw [hn] at hab hbc
  simp at hab hbc
  exact hbc

end keystone_arch_angle_l483_483089


namespace marshmallow_total_l483_483706

def haley := 8
def michael := 3 * haley
def brandon := michael / 2
def sofia := 2 * (haley + brandon)
def lucas := (haley + michael + brandon) / 3 + (Int.floor (Real.sqrt sofia))

theorem marshmallow_total :
  haley + michael + brandon + sofia + lucas = 104 :=
by
  sorry

end marshmallow_total_l483_483706


namespace arithmetic_sequence_a6_l483_483738

theorem arithmetic_sequence_a6 (a : ℕ → ℤ) (h_arith : ∀ n, a (n+1) - a n = a 2 - a 1)
  (h_a1 : a 1 = 5) (h_a5 : a 5 = 1) : a 6 = 0 :=
by
  -- Definitions derived from conditions in the problem:
  -- 1. a : ℕ → ℤ : Sequence defined on ℕ with integer values.
  -- 2. h_arith : ∀ n, a (n+1) - a n = a 2 - a 1 : Arithmetic sequence property
  -- 3. h_a1 : a 1 = 5 : First term of the sequence is 5.
  -- 4. h_a5 : a 5 = 1 : Fifth term of the sequence is 1.
  sorry

end arithmetic_sequence_a6_l483_483738


namespace expected_waiting_time_for_first_bite_l483_483216

noncomputable def average_waiting_time (λ : ℝ) : ℝ := 1 / λ

theorem expected_waiting_time_for_first_bite (bites_first_rod : ℝ) (bites_second_rod : ℝ) (total_time_minutes : ℝ) (total_time_seconds : ℝ) :
  bites_first_rod = 5 → 
  bites_second_rod = 1 → 
  total_time_minutes = 5 → 
  total_time_seconds = 300 → 
  average_waiting_time (bites_first_rod + bites_second_rod) * total_time_seconds = 50 :=
begin
  intros,
  sorry
end

end expected_waiting_time_for_first_bite_l483_483216


namespace books_left_l483_483063

theorem books_left (original_books : ℝ) (given_away_books : ℝ) :
  original_books = 54.0 → given_away_books = 23.0 → (original_books - given_away_books) = 31.0 :=
by
  intros h_orig h_given
  rw [h_orig, h_given]
  norm_num
  sorry

end books_left_l483_483063


namespace Clarissa_needs_to_bring_photos_l483_483591

variable (Cristina John Sarah Clarissa Total_slots : ℕ)

def photo_album_problem := Cristina = 7 ∧ John = 10 ∧ Sarah = 9 ∧ Total_slots = 40 ∧
  (Clarissa + Cristina + John + Sarah = Total_slots)

theorem Clarissa_needs_to_bring_photos (h : photo_album_problem 7 10 9 14 40) : Clarissa = 14 := by
  cases h with _ h, cases h with _ h, cases h with _ h, cases h with _ h, cases h
  sorry

end Clarissa_needs_to_bring_photos_l483_483591


namespace proof_problem_l483_483356

variables (Books : Type) (Available : Books -> Prop)

def all_books_available : Prop := ∀ b : Books, Available b
def some_books_not_available : Prop := ∃ b : Books, ¬ Available b
def not_all_books_available : Prop := ¬ all_books_available Books Available

theorem proof_problem (h : ¬ all_books_available Books Available) : 
  some_books_not_available Books Available ∧ not_all_books_available Books Available :=
by 
  sorry

end proof_problem_l483_483356


namespace num_even_3digit_nums_lt_700_l483_483896

theorem num_even_3digit_nums_lt_700 
  (digits : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}) 
  (even_digits : Finset ℕ := {2, 4, 6}) 
  (h1 : ∀ n ∈ digits, n < 10)
  (h2 : 0 ∉ digits) : 
  ∃ n, n = 126 ∧ ∀ d, d ∈ digits → 
  (d < 10) ∧ ∀ u, u ∈ even_digits → 
  (u < 10) 
:=
  sorry

end num_even_3digit_nums_lt_700_l483_483896


namespace sum_of_digits_3plus4_pow_15_l483_483918

theorem sum_of_digits_3plus4_pow_15 : 
  let n := (3 + 4) ^ 15,
      ones_digit := n % 10,
      tens_digit := (n / 10) % 10 in
  ones_digit + tens_digit = 7 := by sorry

end sum_of_digits_3plus4_pow_15_l483_483918


namespace book_surface_area_l483_483057

variables (L : ℕ) (T : ℕ) (A1 : ℕ) (A2 : ℕ) (W : ℕ) (S : ℕ)

theorem book_surface_area (hL : L = 5) (hT : T = 2) 
                         (hA1 : A1 = L * W) (hA1_val : A1 = 50)
                         (hA2 : A2 = T * W) (hA2_val : A2 = 10) :
  S = 2 * A1 + A2 + 2 * (L * T) :=
sorry

end book_surface_area_l483_483057


namespace fractional_part_of_water_after_replacements_l483_483154

theorem fractional_part_of_water_after_replacements :
  let initial_volume : ℚ := 25
  let removed_volume : ℚ := 5
  let antifreeze_added : ℚ := 5
  let replacement_fraction : ℚ := (initial_volume - removed_volume) / initial_volume
  (replacement_fraction ^ 5 = (1024 / 3125)) :=
by
  let initial_volume : ℚ := 25
  let removed_volume : ℚ := 5
  let antifreeze_added : ℚ := 5
  let replacement_fraction : ℚ := (initial_volume - removed_volume) / initial_volume
  show (replacement_fraction ^ 5 = (1024 / 3125))
  sorry

end fractional_part_of_water_after_replacements_l483_483154


namespace rationalize_denominator_l483_483453

theorem rationalize_denominator : 
  let x := (1 : ℝ)
  let y := (3 : ℝ)
  let z := real.cbrt 3
  let w := real.cbrt 27
  (w = 3) →
  x / (z + w) = real.cbrt (9) / (3 * (real.cbrt (9) + 1)) := 
by
  intros _ h
  rw [h]
  sorry

end rationalize_denominator_l483_483453


namespace find_100th_non_square_cube_l483_483629

/-- A sequence of natural numbers with squares and cubes removed,
    and finding the 100th number in this sequence. -/
def is_square_or_cube (n : ℕ) : Prop :=
  (∃ (k : ℕ), n = k * k) ∨ (∃ (k : ℕ), n = k * k * k)

def num_not_square_or_cube (n : ℕ) : ℕ :=
  (list.range n).filter (λ x, ¬ is_square_or_cube x.succ).length

theorem find_100th_non_square_cube :
  ∃ k, num_not_square_or_cube k = 100 ∧ k = 112 := by
  sorry

end find_100th_non_square_cube_l483_483629


namespace price_of_filter_kit_l483_483539

theorem price_of_filter_kit :
  let total_price := 2 * 16.45 + 2 * 14.05 + 19.50 in
  let discount := 0.08 * total_price in
  total_price - discount = 74.06 :=
by
  let total_price := 2 * 16.45 + 2 * 14.05 + 19.50
  let discount := 0.08 * total_price
  have h1 : total_price = 80.50 := by sorry
  have h2 : discount = 6.44 := by sorry
  calc total_price - discount
       = 80.50 - 6.44 : by rw [h1, h2]
   ... = 74.06 : by norm_num

end price_of_filter_kit_l483_483539


namespace min_arithmetic_mean_of_consecutive_naturals_div_by_1111_l483_483861

theorem min_arithmetic_mean_of_consecutive_naturals_div_by_1111 :
  ∀ a : ℕ, (∃ k, (a * (a + 1) * (a + 2) * (a + 3) * (a + 4) * (a + 5) * (a + 6) * (a + 7) * (a + 8)) = 1111 * k) →
  (a + 4 ≥ 97) :=
sorry

end min_arithmetic_mean_of_consecutive_naturals_div_by_1111_l483_483861


namespace rearrange_portraits_l483_483479

-- Definitions for the conditions in the problem
def adjacent_swap (arr : list ℕ) (i : ℕ) : list ℕ :=
  if h : i < arr.length - 1 then
    arr.take i ++ [arr.get! (i + 1), arr.get! i] ++ arr.drop (i + 2)
  else
    arr

def valid_swap (arr : list ℕ) (i : ℕ) : bool :=
  if h : i < arr.length - 1 then
    abs (arr.get! i - arr.get! (i + 1)) > 1
  else
    false

def is_rotation (arr1 arr2 : list ℕ) : bool :=
  let n := arr1.length
  (list.range n).any (fun k => (list.drop k arr1 ++ list.take k arr1) = arr2)

-- The main theorem statement
theorem rearrange_portraits (initial_arrangement desired_arrangement : list ℕ) :
  (∀ i, valid_swap initial_arrangement i ∨ adjacent_swap initial_arrangement i = initial_arrangement) →
  is_rotation initial_arrangement desired_arrangement ∨
  (∃ i, valid_swap desired_arrangement i ∧ adjacent_swap desired_arrangement i = desired_arrangement) →
  ∃ steps, steps.length = initial_arrangement.length ∧
    (∀ j < steps.length, adjacent_swap (steps.get! j) j = steps.get! (j + 1)) ∧
    is_rotation (steps.get! (steps.length - 1)) desired_arrangement :=
sorry

end rearrange_portraits_l483_483479


namespace total_elephants_in_two_parks_l483_483809

theorem total_elephants_in_two_parks (n1 n2 : ℕ) (h1 : n1 = 70) (h2 : n2 = 3 * n1) : n1 + n2 = 280 := by
  sorry

end total_elephants_in_two_parks_l483_483809


namespace no_x_exists_l483_483086

theorem no_x_exists (x : ℂ) : sqrt (49 - 4 * x^2) + 7 = 0 → false :=
by
  sorry

end no_x_exists_l483_483086


namespace sum_of_last_three_coeffs_l483_483927

theorem sum_of_last_three_coeffs (a : ℝ) (h : a ≠ 0) : 
  let expr := (1 - 1 / a) ^ 8 in 
  let coeffs := (λ (n : ℕ), (binomial 8 n) * a ^ (8 - n) * (-1) ^ n) in
  coeffs 8 + coeffs 7 + coeffs 6 = 21 :=
by {
  sorry
}

end sum_of_last_three_coeffs_l483_483927


namespace hike_people_count_l483_483814

theorem hike_people_count :
  let cars := 3
  let taxis := 6
  let vans := 2
  let people_per_car := 4
  let people_per_taxi := 6
  let people_per_van := 5
  let total_people := (cars * people_per_car) + (taxis * people_per_taxi) + (vans * people_per_van)
  in total_people = 58 :=
by
  -- Proof steps will go here
  sorry

end hike_people_count_l483_483814


namespace bells_toll_together_l483_483954

noncomputable def LCM (a b : Nat) : Nat := (a * b) / (Nat.gcd a b)

theorem bells_toll_together :
  let intervals := [2, 4, 6, 8, 10, 12]
  let lcm := intervals.foldl LCM 1
  lcm = 120 →
  let duration := 30 * 60 -- 1800 seconds
  let tolls := duration / lcm
  tolls + 1 = 16 :=
by
  sorry

end bells_toll_together_l483_483954


namespace factorial_division_l483_483213

theorem factorial_division :
  15! / 14! = 15 := by
  sorry

end factorial_division_l483_483213


namespace problem_f_neg1_add_f_pos1_eq_zero_l483_483319

def f (x : ℝ) : ℝ := Math.sin x * Math.cos x

theorem problem_f_neg1_add_f_pos1_eq_zero : f (-1) + f 1 = 0 := by
  sorry

end problem_f_neg1_add_f_pos1_eq_zero_l483_483319


namespace find_T_when_S_max_l483_483944

noncomputable def S (a : Fin 9 → ℝ) : ℝ :=
  min (a 0) (a 1) + 2 * min (a 1) (a 2) + 3 * min (a 2) (a 3) +
  4 * min (a 3) (a 4) + 5 * min (a 4) (a 5) + 6 * min (a 5) (a 6) +
  7 * min (a 6) (a 7) + 8 * min (a 7) (a 8) + 9 * min (a 8) (a 0)

noncomputable def T (a : Fin 9 → ℝ) : ℝ :=
  max (a 0) (a 1) + 2 * max (a 1) (a 2) + 3 * max (a 2) (a 3) +
  4 * max (a 3) (a 4) + 5 * max (a 4) (a 5) + 6 * max (a 5) (a 6) +
  7 * max (a 6) (a 7) + 8 * max (a 7) (a 8) + 9 * max (a 8) (a 0)

theorem find_T_when_S_max (a : Fin 9 → ℝ) (h_sum : (∑ i, a i) = 1)
  (h_nonneg : ∀ i, 0 ≤ a i) : 
  S a = S0 → T a ∈ Set.Icc (36 / 5) (31 / 4) := 
sorry

end find_T_when_S_max_l483_483944


namespace angle_between_gradients_l483_483267

noncomputable def f (x y z : ℝ) : ℝ := Real.log (x^2 + y^2 + z^2)
noncomputable def g (x y z : ℝ) : ℝ := x * y + y * z + z * x
def M : ℝ × ℝ × ℝ := (1, 1, -1)

theorem angle_between_gradients :
  let grad_f := λ (x y z : ℝ), (2 * x / (x^2 + y^2 + z^2), 2 * y / (x^2 + y^2 + z^2), 2 * z / (x^2 + y^2 + z^2))
      grad_g := λ (x y z : ℝ), (y + z, x + z, y + x)
      ∇f_at_M := grad_f 1 1 (-1)
      ∇g_at_M := grad_g 1 1 (-1)
      dot_product := ∇f_at_M.1 * ∇g_at_M.1 + ∇f_at_M.2 * ∇g_at_M.2 + ∇f_at_M.3 * ∇g_at_M.3
      magnitude_grad_f := Real.sqrt (∇f_at_M.1^2 + ∇f_at_M.2^2 + ∇f_at_M.3^2)
      magnitude_grad_g := Real.sqrt (∇g_at_M.1^2 + ∇g_at_M.2^2 + ∇g_at_M.3^2)
      cos_alpha := dot_product / (magnitude_grad_f * magnitude_grad_g)
  in Real.arccos cos_alpha = Real.pi - Real.arccos (1 / Real.sqrt 3) :=
by sorry

end angle_between_gradients_l483_483267


namespace minimum_value_l483_483308

theorem minimum_value (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  ∃ (y : ℝ), y = (c / (a + b)) + (b / c) ∧ y ≥ (Real.sqrt 2) - (1 / 2) :=
sorry

end minimum_value_l483_483308


namespace sum_a_c_eq_l483_483639

theorem sum_a_c_eq
  (a b c d : ℝ)
  (h1 : a * b + a * c + b * c + b * d + c * d + a * d = 40)
  (h2 : b^2 + d^2 = 29) :
  a + c = 8.4 :=
by
  sorry

end sum_a_c_eq_l483_483639


namespace find_x0_of_f_eq_3_l483_483691

def f (x : ℝ) : ℝ :=
  if 0 ≤ x then 2 * x + 1 else |x|

theorem find_x0_of_f_eq_3 (x0 : ℝ) (h : f x0 = 3) : x0 = 1 ∨ x0 = -3 :=
by sorry

end find_x0_of_f_eq_3_l483_483691


namespace not_possible_to_tile_l483_483020

-- Define the specific type for cells in the figure, i.e., shaded and unshaded
inductive Cell : Type
| shaded : Cell
| unshaded : Cell

-- Define the arrangement of cells in the figure as a list of lists
def figure : List (List Cell) :=
[[Cell.unshaded, Cell.shaded, Cell.unshaded, Cell.shaded, Cell.unshaded],
 [Cell.shaded, Cell.unshaded, Cell.shaded, Cell.unshaded, Cell.shaded],
 [Cell.unshaded, Cell.shaded, Cell.unshaded, Cell.shaded, Cell.unshaded]]

-- Define the main theorem: it is impossible to tile the figure with 1x3 strips without overlaps and gaps
theorem not_possible_to_tile : (∃ (f : figure → bool) (conditions : Prop), 
  (∀ strip : List (List Cell), strip = (repeat (repeat Cell.shaded 3) 1) ∨ strip = (repeat (repeat Cell.unshaded 3) 1) → 
  covers_strip f strip → conditions) → false) := sorry

end not_possible_to_tile_l483_483020


namespace garden_bed_height_l483_483397

-- Definition for the side length of the square base and height
def side_length (x : ℕ) : ℕ := x
def height (x : ℕ) : ℕ := x + 4

-- Definition for the surface area of the rectangular prism
def surface_area (x : ℕ) : ℕ := 2 * x^2 + 4 * x * (x + 4)

-- Statement that if surface area is at least 110 square units, then the height is 8 units
theorem garden_bed_height (x : ℕ) (h : surface_area x ≥ 110) : height x = 8 :=
by sorry

end garden_bed_height_l483_483397


namespace simplify_cot_tan_l483_483468

theorem simplify_cot_tan :
  (Real.cot (Real.toRadians 20) + Real.tan (Real.toRadians 10) = Real.csc (Real.toRadians 20)) :=
by
  sorry

end simplify_cot_tan_l483_483468


namespace Barbier_theorem_for_delta_curves_l483_483120

def delta_curve (h : ℝ) : Type := sorry 
def can_rotate_freely_in_3gon (K : delta_curve h) : Prop := sorry
def length_of_curve (K : delta_curve h) : ℝ := sorry

theorem Barbier_theorem_for_delta_curves
  (K : delta_curve h)
  (h : ℝ)
  (H : can_rotate_freely_in_3gon K)
  : length_of_curve K = (2 * Real.pi * h) / 3 := 
sorry

end Barbier_theorem_for_delta_curves_l483_483120


namespace fourth_friend_payment_l483_483283

theorem fourth_friend_payment (a b c d : ℕ) 
  (h1 : a = (1 / 3) * (b + c + d)) 
  (h2 : b = (1 / 4) * (a + c + d)) 
  (h3 : c = (1 / 5) * (a + b + d))
  (h4 : a + b + c + d = 84) : 
  d = 40 := by
sorry

end fourth_friend_payment_l483_483283


namespace binom_eq_sum_l483_483303

theorem binom_eq_sum (x : ℕ) : (∃ x : ℕ, Nat.choose 7 x = 21) ∧ Nat.choose 7 x = Nat.choose 6 5 + Nat.choose 6 4 :=
by
  sorry

end binom_eq_sum_l483_483303


namespace friends_behind_Yuna_l483_483531

def total_friends : ℕ := 6
def friends_in_front_of_Yuna : ℕ := 2

theorem friends_behind_Yuna : total_friends - friends_in_front_of_Yuna = 4 :=
by
  -- Proof goes here
  sorry

end friends_behind_Yuna_l483_483531


namespace subset_B_of_A_l483_483657

def A : Set ℕ := {2, 0, 3}
def B : Set ℕ := {2, 3}

theorem subset_B_of_A : B ⊆ A :=
by
  sorry

end subset_B_of_A_l483_483657


namespace find_subsets_divisible_l483_483019

noncomputable def biggest_divisible_subsets (n : ℕ) (hn : n > 2) : ℕ :=
  let S := { s | ∃ (t : ℕ), t < n ∧ s = t + 1 }

theorem find_subsets_divisible (n : ℕ) (hn : n > 2) 
  (S : set ℕ) (hS : ∀ s ∈ S, ∃ t, t < n ∧ s = t + 1) :
  ∃ d, d = n - 1 ∧
  ∃ (A B C : set ℕ), A ⊆ S ∧ B ⊆ S ∧ C ⊆ S ∧
  A ≠ ∅ ∧ B ≠ ∅ ∧ C ≠ ∅ ∧
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
  (∑ x in A, x) % d = 0 ∧
  (∑ x in B, x) % d = 0 ∧
  (∑ x in C, x) % d = 0 :=
sorry

end find_subsets_divisible_l483_483019


namespace num_factors_2310_l483_483341

theorem num_factors_2310 : 
  let n := 2310
  let prime_factors := [(2, 1), (3, 1), (5, 1), (7, 1), (11, 1)]
  let distinct_factors := prime_factors.foldl (λ acc p, acc * (p.snd + 1)) 1
  distinct_factors = 32 := 
by
  -- The proof would generally be placed here
  sorry

end num_factors_2310_l483_483341


namespace smallest_four_digit_mod_8_l483_483916

theorem smallest_four_digit_mod_8 :
  ∃ x : ℕ, x ≥ 1000 ∧ x < 10000 ∧ x % 8 = 5 ∧ (∀ y : ℕ, y >= 1000 ∧ y < 10000 ∧ y % 8 = 5 → x ≤ y) :=
sorry

end smallest_four_digit_mod_8_l483_483916


namespace stratified_sampling_no_more_than_45_years_old_l483_483548

theorem stratified_sampling_no_more_than_45_years_old :
  ∀ (total_employees : ℕ) (young_employees : ℕ) (sample_size : ℕ) (drawing_ratio : ℚ),
    total_employees = 200 →
    young_employees = 120 →
    sample_size = 25 →
    drawing_ratio = 25 / 200 →
    young_employees * drawing_ratio = 15 :=
by
  intros total_employees young_employees sample_size drawing_ratio
  intros h_total_employees h_young_employees h_sample_size h_drawing_ratio
  rw [h_total_employees, h_young_employees, h_sample_size, h_drawing_ratio]
  norm_num
  sorry

end stratified_sampling_no_more_than_45_years_old_l483_483548


namespace Clarissa_photos_needed_l483_483593

theorem Clarissa_photos_needed :
  (7 + 10 + 9 <= 40) → 40 - (7 + 10 + 9) = 14 :=
by
  sorry

end Clarissa_photos_needed_l483_483593


namespace conjecture_f_l483_483764

noncomputable def f (n : ℕ) : ℚ :=
  (List.range n).map (λ k => 1 / (k + 1)).sum

lemma lemma_f_2 : f 2 = 3 / 2 :=
by sorry

lemma lemma_f_4 : f 4 > 2 :=
by sorry

lemma lemma_f_8 : f 8 > 5 / 2 :=
by sorry

lemma lemma_f_16 : f 16 > 3 :=
by sorry

lemma lemma_f_32 : f 32 > 7 / 2 :=
by sorry

theorem conjecture_f (n : ℕ) (hn : n ≥ 1) : f (2^n) ≥ (n + 2) / 2 :=
by sorry

end conjecture_f_l483_483764


namespace smallest_arithmetic_mean_l483_483876

noncomputable def S (n : ℕ) := (List.range' n 9).map Nat.ofNat

theorem smallest_arithmetic_mean (n : ℕ) (h1 : 93 ≤ n) (h2 : ∃ k ∈ S n, 11 ∣ k) (h3 : ∃ k ∈ S n, 101 ∣ k) : 
  (n + 4 = 97) := by
  sorry

end smallest_arithmetic_mean_l483_483876


namespace total_green_marbles_l483_483460

-- Conditions
def Sara_green_marbles : ℕ := 3
def Tom_green_marbles : ℕ := 4

-- Problem statement: proving the total number of green marbles
theorem total_green_marbles : Sara_green_marbles + Tom_green_marbles = 7 := by
  sorry

end total_green_marbles_l483_483460


namespace athlete_target_heart_rate_30_l483_483194

def target_heart_rate (age : ℕ) : ℤ :=
  let max_heart_rate := 225 - age
  let target_rate := (max_heart_rate * 75) / 100
  let adjusted_target_rate := target_rate + 2
  Int.round adjusted_target_rate

theorem athlete_target_heart_rate_30 : target_heart_rate 30 = 148 := 
  by 
    -- The proof steps are omitted
  sorry

end athlete_target_heart_rate_30_l483_483194


namespace wade_customers_l483_483899

theorem wade_customers (F : ℕ) (h1 : 2 * F + 6 * F + 72 = 296) : F = 28 := 
by 
  sorry

end wade_customers_l483_483899


namespace impossibility_of_unique_triangle_construction_l483_483119

theorem impossibility_of_unique_triangle_construction (a b : ℝ) (θ : ℝ) : 
  ¬ (∀ (T₁ T₂ : Triangle), 
    T₁.side1 = a → 
    T₁.side2 = b → 
    T₁.angleOpposite = θ → 
    T₂.side1 = a → 
    T₂.side2 = b → 
    T₂.angleOpposite = θ → 
    T₁ ≠ T₂) := 
sorry

end impossibility_of_unique_triangle_construction_l483_483119


namespace inequality_log_equality_log_l483_483783

theorem inequality_log (x : ℝ) (hx : x < 0 ∨ x > 0) :
  max 0 (Real.log (|x|)) ≥ 
  ((Real.sqrt 5 - 1) / (2 * Real.sqrt 5)) * Real.log (|x|) + 
  (1 / (2 * Real.sqrt 5)) * Real.log (|x^2 - 1|) + 
  (1 / 2) * Real.log ((Real.sqrt 5 + 1) / 2) := 
sorry

theorem equality_log (x : ℝ) :
  (max 0 (Real.log (|x|)) = 
  ((Real.sqrt 5 - 1) / (2 * Real.sqrt 5)) * Real.log (|x|) + 
  (1 / (2 * Real.sqrt 5)) * Real.log (|x^2 - 1|) + 
  (1 / 2) * Real.log ((Real.sqrt 5 + 1) / 2)) ↔ 
  (x = (Real.sqrt 5 + 1) / 2 ∨ x = (Real.sqrt 5 - 1) / 2 ∨ x = -(Real.sqrt 5 + 1) / 2 ∨ x = -(Real.sqrt 5 - 1) / 2) := 
sorry

end inequality_log_equality_log_l483_483783


namespace angle_A_minimum_a_l483_483388

variable {α : Type} [LinearOrderedField α]

-- Part 1: Prove A = π / 3 given the specific equation in triangle ABC
theorem angle_A (a b c : α) (cos : α → α)
  (h : b^2 * c * cos c + c^2 * b * cos b = a * b^2 + a * c^2 - a^3) :
  ∃ A : α, A = π / 3 :=
sorry

-- Part 2: Prove the minimum value of a is 1 when b + c = 2
theorem minimum_a (a b c : α) (h : b + c = 2) :
  ∃ a : α, a = 1 :=
sorry

end angle_A_minimum_a_l483_483388


namespace maximize_area_of_quadrilateral_l483_483648

variables {O : Type*} [metric_space O] [has_dist O]
variables (P : O) (a : ℝ) (x : ℝ) (E F : O)
variables (AC BD : set O)

-- Assume the following conditions from a)
-- Condition: Point P is inside the circle with center O
def point_in_circle (O P : O) (r : ℝ) : Prop :=
  dist O P < r

-- Condition: Chords AC and BD are mutually perpendicular
def mutually_perpendicular (AC BD : set O) : Prop :=
  ∃ a b c d : O, a ∈ AC ∧ b ∈ AC ∧ c ∈ BD ∧ d ∈ BD ∧
  dist a b = dist c d ∧
  dist (a + c) (b + d) = dist (a - c) (b - d)

-- Condition: E and F are midpoints of AC and BD respectively
def midpoint (AC : set O) (E : O) : Prop :=
  ∃ a b : O, a ∈ AC ∧ b ∈ AC ∧ dist a E = dist b E

-- Condition: distance from O to P is a
def dist_O_P (O P : O) (a : ℝ) : Prop :=
  dist O P = a

-- Condition: distance from O to F is x
def dist_O_F (O F : O) (x : ℝ) : Prop :=
  dist O F = x

-- Rewrite of the proof problem:
theorem maximize_area_of_quadrilateral 
  (r : ℝ) (h1 : point_in_circle O P r) 
  (h2 : mutually_perpendicular AC BD)
  (h3 : midpoint AC E) 
  (h4 : midpoint BD F) 
  (h5 : dist_O_P O P a) 
  (h6 : dist_O_F O F x) :
  x = a * real.sqrt 2 / 2 :=
sorry

end maximize_area_of_quadrilateral_l483_483648


namespace four_digit_number_exists_l483_483143

theorem four_digit_number_exists :
  ∃ (A B C D : ℕ), A = B / 3 ∧ C = A + B ∧ D = 3 * B ∧
  A ≠ 0 ∧ A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧
  (A * 1000 + B * 100 + C * 10 + D = 1349) :=
by
  sorry

end four_digit_number_exists_l483_483143


namespace ratio_female_to_male_l483_483360

theorem ratio_female_to_male (total_members : ℕ) (female_members : ℕ) (male_members : ℕ) 
  (h1 : total_members = 18) (h2 : female_members = 12) (h3 : male_members = total_members - female_members) : 
  (female_members : ℚ) / (male_members : ℚ) = 2 := 
by 
  sorry

end ratio_female_to_male_l483_483360


namespace digit_pairs_count_l483_483477

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

theorem digit_pairs_count :
  let pairs := (λ d e, is_digit d ∧ is_digit e ∧ (2:ℝ) + (d:ℝ) / 100 + (e:ℝ) / 10000 > 2.006)
  (Finset.univ.product Finset.univ).filter pairs |>.card = 99 :=
by
  sorry

end digit_pairs_count_l483_483477


namespace sum_not_divisible_l483_483444

theorem sum_not_divisible (n : ℕ) : ¬ (n + 2) ∣ (finset.range (n + 1)).sum (λ k, k ^ 1987) :=
sorry

end sum_not_divisible_l483_483444


namespace min_value_sequence_l483_483700

noncomputable def sequence (n : ℕ) : ℕ → ℤ
  | 0 => 15
  | n+1 => (λ (a : ℕ → ℤ), 
             have h : a n = (n+1)^2 - (n+1) + 15 from sorry
             h) sequence

def sequence_condition (n : ℕ) (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) - a n = 2 * n 

theorem min_value_sequence (a : ℕ → ℤ) 
  (h1 : a 1 = 15)   
  (h2 : sequence_condition 1 a) : 
  (∀ n > 0, (a n / n : ℚ) ≥ 27 / 4) :=
begin
  sorry
end

end min_value_sequence_l483_483700


namespace inequality_solution_l483_483610

theorem inequality_solution (x : ℚ) :
  (18 / 60) + abs (x - (21 / 60)) < (16 / 60) ↔ x ∈ Ioo (19 / 60) (23 / 60) :=
by
  sorry

end inequality_solution_l483_483610


namespace expected_waiting_time_l483_483225

/-- Consider a 5-minute interval. There are 5 bites on the first rod 
and 1 bite on the second rod in this interval. Therefore, the total average 
number of bites on both rods during these 5 minutes is 6. The expected waiting 
time for the first bite is 50 seconds. -/
theorem expected_waiting_time
    (average_bites_first_rod : ℝ)
    (average_bites_second_rod : ℝ)
    (total_interval_minutes : ℝ)
    (expected_waiting_time_seconds : ℝ) :
    average_bites_first_rod = 5 ∧
    average_bites_second_rod = 1 ∧
    total_interval_minutes = 5 →
    expected_waiting_time_seconds = 50 :=
by
  sorry

end expected_waiting_time_l483_483225


namespace find_tan_beta_and_sum_l483_483638

variables (α β : ℝ)

-- Conditions
def tan_α := tan α = sqrt 2
def cos_αβ := cos (α + β) = - sqrt 3 / 3
def interval_α := α ∈ Ioo 0 (π / 2)
def interval_β := β ∈ Ioo 0 (π / 2)

-- Proof that tan β and 2α + β have the given values
theorem find_tan_beta_and_sum (h1 : tan_α α) (h2 : cos_αβ α β) (h3 : interval_α α) (h4 : interval_β β) :
  ∃ β, tan β = 2 * sqrt 2 ∧ 2 * α + β = π :=
by 
  sorry

end find_tan_beta_and_sum_l483_483638


namespace dodecahedron_triangle_count_l483_483708

theorem dodecahedron_triangle_count : 
  ∃ (triangles : ℕ), triangles = 1140 ∧ (∃ (vertices : ℕ), vertices = 20 ∧ (20.choose 3) = 1140) :=
by
  sorry

end dodecahedron_triangle_count_l483_483708


namespace geometric_sequence_common_ratio_l483_483661

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : a 2 = 2)
  (h3 : a 5 = 1/4) :
  q = 1/2 :=
sorry

end geometric_sequence_common_ratio_l483_483661


namespace find_real_imaginary_parts_find_a_b_l483_483683

def complex_number_z : Complex := (1 - Complex.i) ^ 2 + 3 * (1 + Complex.i) / (2 - Complex.i)

theorem find_real_imaginary_parts :
  Complex.re (complex_number_z) = 1 ∧ Complex.im (complex_number_z) = 1 :=
by
  sorry

theorem find_a_b (a b : ℝ) (h : complex_number_z^2 + Complex.ofReal a * complex_number_z + Complex.ofReal b = 1 - Complex.i) :
  a = -3 ∧ b = 4 :=
by
  sorry

end find_real_imaginary_parts_find_a_b_l483_483683


namespace complex_sub_problem_l483_483900

def a : ℂ := 5 + complex.i
def b : ℂ := 2 - 3 * complex.i

theorem complex_sub_problem : a - 3 * b = 11 - 8 * complex.i := 
by {
  sorry
}

end complex_sub_problem_l483_483900


namespace max_cookies_Andy_could_have_eaten_l483_483118

theorem max_cookies_Andy_could_have_eaten (cookies : ℕ) (Andy Alexa : ℕ) 
  (h1 : cookies = 24) 
  (h2 : Alexa = k * Andy) 
  (h3 : k > 0) 
  (h4 : Andy + Alexa = cookies) 
  : Andy ≤ 12 := 
sorry

end max_cookies_Andy_could_have_eaten_l483_483118


namespace area_of_small_parallelograms_l483_483489

theorem area_of_small_parallelograms (m n : ℕ) (h_m : 0 < m) (h_n : 0 < n) :
  (1 : ℝ) / (m * n : ℝ) = 1 / (m * n) :=
by
  sorry

end area_of_small_parallelograms_l483_483489


namespace planes_through_skew_lines_are_infinite_l483_483717

-- Definitions of skew lines and conditions
def is_skew (a b : ℝ^3 → Prop) : Prop :=
  ¬ ∃ p, a p ∧ b p ∧ ¬ ∃ v, ∀ p, a p → b (p + v)

-- The actual theorem statement
theorem planes_through_skew_lines_are_infinite {a b : ℝ^3 → Prop} 
(h_skew : is_skew a b) : 
  ∃ (S : set (ℝ^3 → Prop)), 
  (∀ p : ℝ^3 → Prop, p ∈ S → (∀ x, a x → p x) ∧ (∀ v, b v → ¬ p v)) ∧ 
  set.infinite S :=
sorry

end planes_through_skew_lines_are_infinite_l483_483717


namespace calculate_total_area_l483_483241

theorem calculate_total_area :
  let height1 := 7
  let width1 := 6
  let width2 := 4
  let height2 := 5
  let height3 := 1
  let width3 := 2
  let width4 := 5
  let height4 := 6
  let area1 := width1 * height1
  let area2 := width2 * height2
  let area3 := height3 * width3
  let area4 := width4 * height4
  area1 + area2 + area3 + area4 = 94 := by
  sorry

end calculate_total_area_l483_483241


namespace factor_expression_l483_483263

theorem factor_expression (x : ℝ) : 
  5 * x * (x - 2) + 9 * (x - 2) - 4 * (x - 2) = 5 * (x - 2) * (x + 1) :=
by
  -- proof goes here
  sorry

end factor_expression_l483_483263


namespace possible_values_for_k_l483_483762

-- Definitions and conditions
variables {𝔸 : Type*} [inner_product_space ℝ 𝔸]
variables (a b c : 𝔸)
variables (k : ℝ)

-- Hypotheses
hypothesis h1 : ∥a∥ = 1
hypothesis h2 : ∥b∥ = 1
hypothesis h3 : ∥c∥ = 1
hypothesis h4 : inner a b = 0
hypothesis h5 : inner a c = 0
hypothesis h6 : real.angle b c = π / 3

-- Proof statement
theorem possible_values_for_k :
  (a = k • (b × c) → k = 2 * real.sqrt 3 / 3 ∨ k = -2 * real.sqrt 3 / 3) :=
sorry

end possible_values_for_k_l483_483762


namespace bus_full_people_could_not_take_l483_483538

-- Definitions of the given conditions
def bus_capacity : ℕ := 80
def first_pickup_people : ℕ := (3 / 5) * bus_capacity
def people_exit_at_second_pickup : ℕ := 25
def people_waiting_at_second_pickup : ℕ := 90

-- The Lean statement to prove the number of people who could not take the bus
theorem bus_full_people_could_not_take (h1 : bus_capacity = 80)
                                       (h2 : first_pickup_people = 48)
                                       (h3 : people_exit_at_second_pickup = 25)
                                       (h4 : people_waiting_at_second_pickup = 90) :
  90 - (80 - (48 - 25)) = 33 :=
by
  sorry

end bus_full_people_could_not_take_l483_483538


namespace min_value_log2_on_interval_minimum_value_f_is_zero_on_interval_minimum_value_of_log2_on_interval_l483_483092

def f (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem min_value_log2_on_interval : 
  ∀ x ∈ set.Icc 1 2, f x ≥ f 1 := 
by
  intro x hx
  unfold f
  sorry

theorem minimum_value_f_is_zero_on_interval :
  ∀ x ∈ set.Icc 1 2, f x ≥ 0 :=
by
  intro x hx
  unfold f
  sorry

example : f 1 = 0 := 
by
  unfold f
  sorry

example : f 2 > 0 :=
by
  unfold f
  sorry

theorem minimum_value_of_log2_on_interval :
  ∃ x ∈ set.Icc 1 2, f x = 0 :=
by
  use 1
  split
  · norm_num [set.Icc]
  · unfold f
    norm_num

end min_value_log2_on_interval_minimum_value_f_is_zero_on_interval_minimum_value_of_log2_on_interval_l483_483092


namespace probability_321_correct_l483_483497

def cards := {1, 2, 3}

-- Define the total number of permutations.
def total_permutations := Nat.factorial 3

-- Define the event "321".
def event_321 (order : List ℕ) : Prop := order = [3, 2, 1]

-- Define the probability of the event "321".
def probability_321 : ℚ := (1 / total_permutations : ℚ)

theorem probability_321_correct : probability_321 = 1 / 6 := by
  sorry

end probability_321_correct_l483_483497


namespace number_of_non_consecutive_triangles_l483_483796

theorem number_of_non_consecutive_triangles (P : Finset (Fin 10)) (hP : P.card = 10) :
  P.subset (Finset.univ : Finset (Fin 10)) → 
  ∃ n, n = 110 ∧
    ∀ (t : Finset (Fin 10)), t.card = 3 → (∀ x y ∈ t, x ≠ y → abs (x - y) ≠ 1) →
      Finset.card t = n :=
by sorry

end number_of_non_consecutive_triangles_l483_483796


namespace total_crayons_l483_483139

-- Define relevant conditions
def crayons_per_child : ℕ := 8
def number_of_children : ℕ := 7

-- Define the Lean statement to prove the total number of crayons
theorem total_crayons : crayons_per_child * number_of_children = 56 :=
by
  sorry

end total_crayons_l483_483139


namespace investment_years_l483_483080

noncomputable def P : ℝ := 4000
noncomputable def A : ℝ := 5324.000000000002
noncomputable def r : ℝ := 0.10
noncomputable def n : ℕ := 1

theorem investment_years : 
  ∃ t : ℝ, (A = P * (1 + r / n)^(n * t) ∧ (abs (t - 3) < 1)
  := sorry

end investment_years_l483_483080


namespace sum_abc_l483_483943

theorem sum_abc (a b c: ℝ) 
  (h1 : ∃ x: ℝ, x^2 + a * x + 1 = 0 ∧ x^2 + b * x + c = 0)
  (h2 : ∃ x: ℝ, x^2 + x + a = 0 ∧ x^2 + c * x + b = 0) :
  a + b + c = -3 := 
sorry

end sum_abc_l483_483943


namespace votes_tally_count_l483_483115

theorem votes_tally_count (m n : ℕ) (h : m > n) :
  (∑ x in finset.range (m+n), if x < m then 1 else -1).sum = m - n → 
  |valid_paths| = (m-n) * (nat.choose (m+n) m) / (m+n) :=
sorry

end votes_tally_count_l483_483115


namespace symmetry_condition_l483_483632

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * sin (x + π / 3)

theorem symmetry_condition (ϕ : ℝ) (hϕ : |ϕ| ≤ π / 2)
    (hxy: ∀ x : ℝ, f (x + ϕ) = f (-x + ϕ)) : ϕ = π / 6 :=
by
  -- Since the problem specifically asks for the statement only and not the proof steps,
  -- a "sorry" is used to skip the proof content.
  sorry

end symmetry_condition_l483_483632


namespace product_real_parts_l483_483801

theorem product_real_parts (x : ℂ) (y : ℂ) (hx : x^2 - 4 * x = -1 - 3 * complex.I) (hy : y^2 - 4 * y = -1 - 3 * complex.I) :
  (x.re * y.re) = (8 - sqrt 6 + sqrt 3) / 2 :=
sorry

end product_real_parts_l483_483801


namespace alternating_sum_eq_one_l483_483642

theorem alternating_sum_eq_one (n : ℕ) (a : ℕ → ℤ) (h1 : (1 + 2) ^ n = finset.sum finset.univ a)
    (h2 : (1 + 2 * (-1)) ^ n = finset.sum (finset.univ : finset (fin n)) (λ i, (-1) ^ i * a i))
    (h3 : finset.sum finset.univ a = 729) : finset.sum (finset.univ : finset (fin n)) (λ i, (-1) ^ i * a i) = 1 :=
by
  sorry

end alternating_sum_eq_one_l483_483642


namespace smallest_arithmetic_mean_l483_483879

noncomputable def S (n : ℕ) := (List.range' n 9).map Nat.ofNat

theorem smallest_arithmetic_mean (n : ℕ) (h1 : 93 ≤ n) (h2 : ∃ k ∈ S n, 11 ∣ k) (h3 : ∃ k ∈ S n, 101 ∣ k) : 
  (n + 4 = 97) := by
  sorry

end smallest_arithmetic_mean_l483_483879


namespace area_of_inscribed_square_in_ellipse_l483_483987

open Real

noncomputable def inscribed_square_area : ℝ := 32

theorem area_of_inscribed_square_in_ellipse :
  ∀ (x y : ℝ),
  (x^2 / 4 + y^2 / 8 = 1) →
  (x = t - t) ∧ (y = (t + t) / sqrt 2) ∧ 
  (t = sqrt 4) → inscribed_square_area = 32 :=
  sorry

end area_of_inscribed_square_in_ellipse_l483_483987


namespace find_common_ratio_l483_483239

theorem find_common_ratio (q : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : ∀ n, a n > 0) 
  (h2 : a 1 = 1) 
  (h3 : S 3 = 7)
  (hg : ∀ n, S n = ∑ i in finset.range n, a i)
  (hq : ∀ n, a (n + 1) = a n * q) : q = 2 := 
by 
  sorry

end find_common_ratio_l483_483239


namespace isosceles_triangle_perimeter_l483_483653

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 6 ∨ a = 9) (h2 : b = 6 ∨ b = 9) (h : a ≠ b) : (a * 2 + b = 21 ∨ a * 2 + b = 24) :=
by
  sorry

end isosceles_triangle_perimeter_l483_483653


namespace smallest_four_digit_mod_8_l483_483909

theorem smallest_four_digit_mod_8 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 8 = 5 ∧ (∀ m : ℕ, m >= 1000 ∧ m < 10000 ∧ m % 8 = 5 → n ≤ m) → n = 1005 :=
by
  sorry

end smallest_four_digit_mod_8_l483_483909


namespace inequality_k_m_l483_483417

theorem inequality_k_m (k m : ℕ) (hk : 0 < k) (hm : 0 < m) (hkm : k > m) (hdiv : (k^3 - m^3) ∣ k * m * (k^2 - m^2)) :
  (k - m)^3 > 3 * k * m := 
by sorry

end inequality_k_m_l483_483417


namespace arithmetic_and_geometric_sequences_l483_483306

def is_arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_sequence (a : ℕ → ℕ) : Prop :=
  ∃ C : ℤ, ∀ n : ℕ, a (n + 2) * a n = (a (n + 1))^2

def sequence_a (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  2 * n - 1

def sequence_b (b : ℕ → ℕ) (n : ℕ) : ℕ :=
  3^n

def log3 (x : ℝ) : ℝ :=
  (Real.log x) / (Real.log 3)

def sequence_c (a b : ℕ → ℕ) (c : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, c n = (a n)^2 + 8 * log3 (b n) / (a (n+1) * b n)

def sum_c (c : ℕ → ℝ) (M : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, M n = ∑ i in Finset.range n, c i

noncomputable def expected_sum (M : ℕ → ℝ) : ℕ → ℝ := 
  λ n, 2 - (n + 2) / (3^n)

theorem arithmetic_and_geometric_sequences (S₃ : ℕ) (a_2 a_5 a_14 : ℕ) (b_sum_formula : ℕ → ℝ) : 
  S₃ = 9 → -- First condition S₃ = 9
  is_arithmetic_sequence sequence_a 2 → -- Second condition on arithmetic sequence with common difference 2
  is_geometric_sequence (λ n, [a_2, a_5, a_14]!!n) → -- Third condition to form a geometric sequence
  b_sum_formula = (λ n, (3^(n+1) - 3) / 2) → -- Fourth condition on sum of sequence b
  sequence_a (λ n, 2 * n - 1) → -- Prove sequence a_n
  sequence_b (λ n, 3^n) → -- Prove sequence b_n
  ∀ c : (ℕ → ℝ), sum_c (sequence_c sequence_a sequence_b c) (expected_sum M) := -- Prove sum of sequence M_n
sorry

end arithmetic_and_geometric_sequences_l483_483306


namespace sequence_sum_l483_483249

theorem sequence_sum :
  (∑' k, (1 : ℝ) / (sequence_y k + 1)) = (1 : ℝ) / 151
  where
    sequence_y : ℕ → ℝ
    sequence_y 1 := 150
    sequence_y (n + 1) := (sequence_y n)^2 + 2 * (sequence_y n)
:=
sorry

end sequence_sum_l483_483249


namespace new_total_lines_is_240_l483_483436

-- Define the original number of lines, the increase, and the percentage increase
variables (L : ℝ) (increase : ℝ := 110) (percentage_increase : ℝ := 84.61538461538461 / 100)

-- The statement to prove
theorem new_total_lines_is_240 (h : increase = percentage_increase * L) : L + increase = 240 := sorry

end new_total_lines_is_240_l483_483436


namespace solution1_solution2_l483_483208

noncomputable def problem1 : ℝ :=
  real.sqrt 16 + 2 * real.sqrt 9 - real.cbrt 27

theorem solution1 : problem1 = 7 :=
by sorry

noncomputable def problem2 : ℝ :=
  abs (1 - real.sqrt 2) + real.sqrt 4 - real.cbrt (-8)

theorem solution2 : problem2 = real.sqrt 2 + 3 :=
by sorry

end solution1_solution2_l483_483208


namespace student_number_choice_l483_483989

theorem student_number_choice (x : ℤ) (h : 3 * x - 220 = 110) : x = 110 :=
by sorry

end student_number_choice_l483_483989


namespace michael_time_proof_l483_483432

-- Definitions based on the given conditions
def rate_father : ℝ := 4 -- rate in feet per hour
def time_father : ℝ := 400 -- time in hours
def depth_father := rate_father * time_father -- depth of father's hole
def depth_michael := 2 * depth_father - 400 -- depth of Michael's hole
def rate_michael : ℝ := rate_father -- same rate for Michael

-- Goal: Time taken by Michael to dig the hole
def time_michael := depth_michael / rate_michael

theorem michael_time_proof : time_michael = 700 :=
by
  unfold time_michael depth_michael rate_michael depth_father rate_father time_father
  simp
  norm_num
  sorry -- Proof steps omitted

end michael_time_proof_l483_483432


namespace first_group_men_l483_483475

theorem first_group_men (x : ℕ) (days1 days2 : ℝ) (men2 : ℕ) (h1 : days1 = 25) (h2 : days2 = 17.5) (h3 : men2 = 20) (h4 : x * days1 = men2 * days2) : x = 14 := 
by
  sorry

end first_group_men_l483_483475


namespace log_vs_ratio_l483_483712

theorem log_vs_ratio (x : ℝ) (hx : 0 < x) (h_small : x < 1) : 
  log (1 + x) > (x^2 / (1 + x)) :=
sorry

end log_vs_ratio_l483_483712


namespace ratio_of_area_of_shaded_square_l483_483971

theorem ratio_of_area_of_shaded_square 
  (large_square : Type) 
  (smaller_squares : Finset large_square) 
  (area_large_square : ℝ) 
  (area_smaller_square : ℝ) 
  (h_division : smaller_squares.card = 25)
  (h_equal_area : ∀ s ∈ smaller_squares, area_smaller_square = (area_large_square / 25))
  (shaded_square : Finset large_square)
  (h_shaded_sub : shaded_square ⊆ smaller_squares)
  (h_shaded_card : shaded_square.card = 5) :
  (5 * area_smaller_square) / area_large_square = 1 / 5 := 
by
  sorry

end ratio_of_area_of_shaded_square_l483_483971


namespace smallest_arithmetic_mean_l483_483880

noncomputable def S (n : ℕ) := (List.range' n 9).map Nat.ofNat

theorem smallest_arithmetic_mean (n : ℕ) (h1 : 93 ≤ n) (h2 : ∃ k ∈ S n, 11 ∣ k) (h3 : ∃ k ∈ S n, 101 ∣ k) : 
  (n + 4 = 97) := by
  sorry

end smallest_arithmetic_mean_l483_483880


namespace collinear_example_l483_483701

structure Vector2D where
  x : ℝ
  y : ℝ

def collinear (u v : Vector2D) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ v.x = k * u.x ∧ v.y = k * u.y

def a : Vector2D := ⟨1, 2⟩
def b : Vector2D := ⟨2, 4⟩

theorem collinear_example :
  collinear a b :=
by
  sorry

end collinear_example_l483_483701


namespace initial_marbles_l483_483583

variable (C_initial : ℕ)
variable (marbles_given : ℕ := 42)
variable (marbles_left : ℕ := 5)

theorem initial_marbles :
  C_initial = marbles_given + marbles_left :=
sorry

end initial_marbles_l483_483583


namespace expected_waiting_time_l483_483228

/-- Consider a 5-minute interval. There are 5 bites on the first rod 
and 1 bite on the second rod in this interval. Therefore, the total average 
number of bites on both rods during these 5 minutes is 6. The expected waiting 
time for the first bite is 50 seconds. -/
theorem expected_waiting_time
    (average_bites_first_rod : ℝ)
    (average_bites_second_rod : ℝ)
    (total_interval_minutes : ℝ)
    (expected_waiting_time_seconds : ℝ) :
    average_bites_first_rod = 5 ∧
    average_bites_second_rod = 1 ∧
    total_interval_minutes = 5 →
    expected_waiting_time_seconds = 50 :=
by
  sorry

end expected_waiting_time_l483_483228


namespace find_m_l483_483424

-- Definitions based on the conditions
def point_on_line (x y : ℝ) : Prop := x - 3 * y = 0

def z (m : ℝ) : ℂ := complex.mk (4^m - 1) (2^m + 1)

-- The main theorem we need to prove
theorem find_m (m : ℝ) (h : point_on_line (4^m - 1) (2^m + 1)) : m = 2 :=
sorry

end find_m_l483_483424


namespace students_play_neither_l483_483939

theorem students_play_neither (total_students : ℕ) (F : ℕ) (T : ℕ) (F_inter_T : ℕ) (N : ℕ) 
  (h1 : total_students = 40) 
  (h2 : F = 26) 
  (h3 : T = 20) 
  (h4 : F_inter_T = 17) 
  (h5 : N = total_students - (F + T - F_inter_T)) : 
  N = 11 :=
by
  rw [h1, h2, h3, h4, h5]
  exact rfl

end students_play_neither_l483_483939


namespace simplify_cot20_tan10_l483_483467

theorem simplify_cot20_tan10 :
  (Real.cot 20 + Real.tan 10 = Real.csc 20) :=
sorry

end simplify_cot20_tan10_l483_483467


namespace largest_black_square_circle_radius_l483_483889

-- Definition of the chessboard and its properties
def is_chessboard : Prop := ∃ (n : ℕ), n = 8 ∧ ∀ (x y : ℕ), (x < n ∧ y < n) → (x + y) % 2 = 0 ∨ (x + y) % 2 = 1

-- Definition of a black cell on the chessboard
def is_black_cell (x y : ℕ) : Prop := (x + y) % 2 = 0

-- Definition of a circle radius
def circle_radius (r : ℝ) : Prop := r = (Real.sqrt 10) / 2

-- The proof statement
theorem largest_black_square_circle_radius :
  is_chessboard → ∃ r : ℝ, circle_radius r :=
by
  intros,
  existsi (Real.sqrt 10) / 2,
  apply circle_radius,
  sorry

end largest_black_square_circle_radius_l483_483889


namespace sum_S_2018_l483_483381

noncomputable def a_n (n : ℕ) : ℝ := real.sin (2 * n * real.pi / 3) + real.sqrt 3 * real.cos (2 * n * real.pi / 3)

noncomputable def S_n (n : ℕ) : ℝ := (finset.range n).sum (λ i, a_n (i + 1))

theorem sum_S_2018 : S_n 2018 = -real.sqrt 3 :=
sorry

end sum_S_2018_l483_483381


namespace expected_waiting_time_for_first_bite_l483_483218

noncomputable def average_waiting_time (λ : ℝ) : ℝ := 1 / λ

theorem expected_waiting_time_for_first_bite (bites_first_rod : ℝ) (bites_second_rod : ℝ) (total_time_minutes : ℝ) (total_time_seconds : ℝ) :
  bites_first_rod = 5 → 
  bites_second_rod = 1 → 
  total_time_minutes = 5 → 
  total_time_seconds = 300 → 
  average_waiting_time (bites_first_rod + bites_second_rod) * total_time_seconds = 50 :=
begin
  intros,
  sorry
end

end expected_waiting_time_for_first_bite_l483_483218


namespace Peggy_bandages_needed_l483_483440

theorem Peggy_bandages_needed
  (initial_bandages : ℕ)
  (final_bandages : ℕ)
  (right_knee : ℕ)
  (left_knee : ℕ) :
  initial_bandages = 16 →
  final_bandages = 11 →
  right_knee = 3 →
  left_knee = initial_bandages - final_bandages - right_knee :=
begin
  intros hinitial hfinal hright,
  sorry
end

end Peggy_bandages_needed_l483_483440


namespace tangent_line_equation_l483_483307

noncomputable def f (x a : ℝ) : ℝ := x^2 * (x - a)
noncomputable def f' (x a : ℝ) : ℝ := 3 * x^2 - 2 * a * x

theorem tangent_line_equation
  (a : ℝ)
  (h : f'(1, a) = 3) :
  equation_of_tangent_line f (1, f(1, a)) = 3 * x - y - 2 := sorry

end tangent_line_equation_l483_483307


namespace quadrilateral_AB_length_l483_483293

theorem quadrilateral_AB_length
  (AD BC XY AB : ℝ)
  (H1 : AD = 16)
  (H2 : ∀ A B C D X Y, AD ∥ BC)
  (H3 : ∀ A B C D X Y, is_angle_bisector A X D C)
  (H4 : ∀ A B C D X, angle_AXC_90 (A X C))
  (H5 : CY = 13) :
  AB = 14.5 := by
  sorry

end quadrilateral_AB_length_l483_483293


namespace number_of_indeterminate_conditions_l483_483190

noncomputable def angle_sum (A B C : ℝ) : Prop := A + B + C = 180
noncomputable def condition1 (A B C : ℝ) : Prop := A + B = C
noncomputable def condition2 (A B C : ℝ) : Prop := A = C / 6 ∧ B = 2 * (C / 6)
noncomputable def condition3 (A B : ℝ) : Prop := A = 90 - B
noncomputable def condition4 (A B C : ℝ) : Prop := A = B ∧ B = C
noncomputable def condition5 (A B C : ℝ) : Prop := 2 * A = C ∧ 2 * B = C
noncomputable def is_right_triangle (C : ℝ) : Prop := C = 90

theorem number_of_indeterminate_conditions (A B C : ℝ) :
  (angle_sum A B C) →
  (condition1 A B C → is_right_triangle C) →
  (condition2 A B C → is_right_triangle C) →
  (condition3 A B → is_right_triangle C) →
  (condition4 A B C → ¬ is_right_triangle C) →
  (condition5 A B C → is_right_triangle C) →
  ∃ n, n = 1 :=
sorry

end number_of_indeterminate_conditions_l483_483190


namespace minimum_enclosing_sphere_radius_l483_483628

theorem minimum_enclosing_sphere_radius:
  ∀ (r: ℝ) (r = 1) (touches : ∀ i j, i ≠ j → dist (c i) (c j) = 2 * r),
  ∃ R: ℝ, R = (sqrt 6 + 2) / 2 :=
by
  sorry

end minimum_enclosing_sphere_radius_l483_483628


namespace value_of_expression_l483_483101

theorem value_of_expression (a : ℝ) 
  (h₀ : a ≠ 0) 
  (h₁ : 16 * a ≠ 0) 
  (h₂ : 64 ≠ 0)
  (h₃ : -32 ≠ 0) : 
  (1 / 16 * a^0 + (1 / (16 * a))^0 - 64^(-1/2) - (-32)^(-4/5) = 1) :=
by
  sorry

end value_of_expression_l483_483101


namespace dennis_floor_l483_483597

theorem dennis_floor :
  ∃ d c b f e: ℕ, 
  (d = c + 2) ∧ 
  (c = b + 1) ∧ 
  (c = f / 4) ∧ 
  (f = 16) ∧ 
  (e = d / 2) ∧ 
  (d = 6) :=
by
  sorry

end dennis_floor_l483_483597


namespace sequence_terms_are_integers_l483_483027

theorem sequence_terms_are_integers (a : ℕ → ℕ)
  (h0 : a 0 = 1) 
  (h1 : a 1 = 2) 
  (h_recurrence : ∀ n : ℕ, (n + 3) * a (n + 2) = (6 * n + 9) * a (n + 1) - n * a n) :
  ∀ n : ℕ, ∃ k : ℤ, a n = k := 
by
  -- Initialize the proof
  sorry

end sequence_terms_are_integers_l483_483027


namespace shares_difference_l483_483997

noncomputable def Faruk_share (V : ℕ) : ℕ := (3 * (V / 5))
noncomputable def Ranjith_share (V : ℕ) : ℕ := (7 * (V / 5))

theorem shares_difference {V : ℕ} (hV : V = 1500) : 
  Ranjith_share V - Faruk_share V = 1200 :=
by
  rw [Faruk_share, Ranjith_share]
  subst hV
  -- It's just a declaration of the problem and sorry is used to skip the proof.
  sorry

end shares_difference_l483_483997


namespace area_ratio_of_squares_l483_483882

theorem area_ratio_of_squares (R x y : ℝ) (hx : x^2 = (4/5) * R^2) (hy : y = R * Real.sqrt 2) :
  x^2 / y^2 = 2 / 5 :=
by sorry

end area_ratio_of_squares_l483_483882


namespace milkman_water_mixture_l483_483174

theorem milkman_water_mixture : 
  ∀ (total_milk liters_pure_milk : ℕ) (cost_pure_milk_per_liter profit within_total_milk total_cost_of_milk : ℝ),
  total_milk = 30 →
  liters_pure_milk = 20 →
  cost_pure_milk_per_liter = 18 →
  profit = 35 →
  within_total_milk = 395 / 18 →
  total_cost_of_milk = 360 →
  ∃ W : ℝ, W = ceiling (within_total_milk - liters_pure_milk) ∧ W = 2 := 
by
  sorry

end milkman_water_mixture_l483_483174


namespace expected_waiting_time_l483_483227

/-- Consider a 5-minute interval. There are 5 bites on the first rod 
and 1 bite on the second rod in this interval. Therefore, the total average 
number of bites on both rods during these 5 minutes is 6. The expected waiting 
time for the first bite is 50 seconds. -/
theorem expected_waiting_time
    (average_bites_first_rod : ℝ)
    (average_bites_second_rod : ℝ)
    (total_interval_minutes : ℝ)
    (expected_waiting_time_seconds : ℝ) :
    average_bites_first_rod = 5 ∧
    average_bites_second_rod = 1 ∧
    total_interval_minutes = 5 →
    expected_waiting_time_seconds = 50 :=
by
  sorry

end expected_waiting_time_l483_483227


namespace negation_of_prop_l483_483486

theorem negation_of_prop (P : Prop) :
  (¬ ∀ x > 0, x - 1 ≥ Real.log x) ↔ ∃ x > 0, x - 1 < Real.log x :=
by
  sorry

end negation_of_prop_l483_483486


namespace average_waiting_time_for_first_bite_l483_483223

/-- 
Let S be a period of 5 minutes (300 seconds).
- We have an average of 5 bites in 300 seconds on the first fishing rod.
- We have an average of 1 bite in 300 seconds on the second fishing rod.
- The total average number of bites on both rods during this period is 6 bites.
The bites occur independently and follow a Poisson process.

We aim to prove that the waiting time for the first bite, given these conditions, is 
expected to be 50 seconds.
-/
theorem average_waiting_time_for_first_bite :
  let S := 300 -- 5 minutes in seconds
  -- The average number of bites on the first and second rod in period S.
  let avg_bites1 := 5
  let avg_bites2 := 1
  let total_avg_bites := avg_bites1 + avg_bites2
  -- The rate parameter λ for the Poisson process is total_avg_bites / S.
  let λ := total_avg_bites / S
  -- The average waiting time for the first bite.
  1 / λ = 50 :=
by
  let S := 300
  let avg_bites1 := 5
  let avg_bites2 := 1
  let total_avg_bites := avg_bites1 + avg_bites2
  let λ := total_avg_bites / S
  -- convert λ to seconds to ensure unit consistency
  have hλ: λ = 6 / 300 := rfl
  -- The expected waiting time for the first bite is 1 / λ
  have h_waiting_time: 1 / λ = 300 / 6 := by
    rw [hλ, one_div, div_div_eq_mul]
    norm_num
  exact h_waiting_time

end average_waiting_time_for_first_bite_l483_483223


namespace teams_competing_l483_483197

theorem teams_competing
  (members_per_team : ℕ)
  (pairs_of_skates_per_member : ℕ)
  (sets_of_laces_per_pair : ℕ)
  (total_laces_given : ℕ)
  (h1 : members_per_team = 10)
  (h2 : pairs_of_skates_per_member = 2)
  (h3 : sets_of_laces_per_pair = 3)
  (h4 : total_laces_given = 240) :
  (total_laces_given / (members_per_team * (pairs_of_skates_per_member * sets_of_laces_per_pair))) = 4 :=
by
  rw [h1, h2, h3, h4]
  norm_num
  -- Here, norm_num simplifies the arithmetic operations, adhering to the given conditions.
  sorry

end teams_competing_l483_483197


namespace distributive_step_basis_l483_483147

-- Definitions based on conditions
def initial_expression (x y : ℝ) : ℝ := (x + y) / 2 - x

def expected_transformed_expression (x y : ℝ) : ℝ := (1 / 2) * x + (1 / 2) * y - x

-- The theorem based on the problem statement
theorem distributive_step_basis (x y : ℝ) : 
  initial_expression x y = expected_transformed_expression x y := 
by
  sorry

end distributive_step_basis_l483_483147


namespace melanie_plums_count_l483_483060

theorem melanie_plums_count (dan_plums sally_plums total_plums melanie_plums : ℕ)
    (h1 : dan_plums = 9)
    (h2 : sally_plums = 3)
    (h3 : total_plums = 16)
    (h4 : melanie_plums = total_plums - (dan_plums + sally_plums)) :
    melanie_plums = 4 := by
  -- Proof will be filled here
  sorry

end melanie_plums_count_l483_483060


namespace fraction_in_orange_tin_l483_483157

variables {C : ℕ} -- assume total number of cookies as a natural number

theorem fraction_in_orange_tin (h1 : 11 / 12 = (1 / 6) + (5 / 12) + w)
  (h2 : 1 - (11 / 12) = 1 / 12) :
  w = 1 / 3 :=
by
  sorry

end fraction_in_orange_tin_l483_483157


namespace collinear_of_set_l483_483047

-- Define the finite non-empty set of points in the plane.
def E : set point := sorry

-- Define the condition that any line passing through two points of E contains a third point of E.
def line_condition (E : set point) : Prop :=
∀ (p1 p2 : point), p1 ∈ E → p2 ∈ E → ∃ p3 ∈ E, p3 ≠ p1 ∧ p3 ≠ p2 ∧ collinear p1 p2 p3

-- Define the collinearity of points in E.
def collinear_set (E : set point) : Prop :=
∃ (l : line), ∀ (p ∈ E), p ∈ l

-- The theorem we want to prove.
theorem collinear_of_set (h₁ : set.finite E) (h₂ : E.nonempty) (h₃ : line_condition E) : collinear_set E :=
sorry

end collinear_of_set_l483_483047


namespace dresser_clothing_capacity_l483_483064

theorem dresser_clothing_capacity (pieces_per_drawer : ℕ) (number_of_drawers : ℕ) (total_pieces : ℕ) 
  (h1 : pieces_per_drawer = 5)
  (h2 : number_of_drawers = 8)
  (h3 : total_pieces = 40) :
  pieces_per_drawer * number_of_drawers = total_pieces :=
by {
  sorry
}

end dresser_clothing_capacity_l483_483064


namespace magnitude_of_vector_AB_l483_483438

   theorem magnitude_of_vector_AB :
     let A := (2 : ℝ, 1 : ℝ)
     let B := (-3 : ℝ, 2 : ℝ)
     let AB := (B.1 - A.1, B.2 - A.2)
     (real.sqrt (AB.1^2 + AB.2^2) = real.sqrt 26) :=
   by
     sorry
   
end magnitude_of_vector_AB_l483_483438


namespace sum_of_digits_3_plus_4_pow_15_l483_483923

theorem sum_of_digits_3_plus_4_pow_15 :
  let n := (3 + 4) ^ 15
  let last_two_digits := n % 100
  let tens_digit := last_two_digits / 10
  let ones_digit := last_two_digits % 10
  tens_digit + ones_digit = 7 :=
by sorry

end sum_of_digits_3_plus_4_pow_15_l483_483923


namespace cot_tan_simplified_l483_483462

theorem cot_tan_simplified :
  (Real.cot (Real.pi / 9) + Real.tan (Real.pi / 18) = Real.csc (Real.pi / 9)) :=
by
  sorry

end cot_tan_simplified_l483_483462


namespace min_value_fraction_l483_483275

theorem min_value_fraction (x : ℝ) (h : x > 9) : 
  (∃ y, y > 9 ∧ (∀ z, z > 9 → y ≤ (z^3 / (z - 9)))) ∧ (∀ z, z > 9 → (∃ w, w > 9 ∧ z^3 / (z - 9) = 325)) := 
  sorry

end min_value_fraction_l483_483275


namespace apple_and_cherry_pies_total_l483_483161

-- Given conditions state that:
def apple_pies : ℕ := 6
def cherry_pies : ℕ := 5

-- We aim to prove that the total number of apple and cherry pies is 11.
theorem apple_and_cherry_pies_total : apple_pies + cherry_pies = 11 := by
  sorry

end apple_and_cherry_pies_total_l483_483161


namespace average_increase_l483_483968

-- Definition of conditions
variables (A : ℝ) -- Average of the first 4 matches
variables (total_goals_5_matches : ℝ := 16) -- Total goals scored in 5 matches
variables (goals_5th_match : ℝ := 4) -- Goals scored in the 5th match

-- Total goals scored in the first 4 matches
def total_goals_4_matches : ℝ := 4 * A

-- Equation given by the problem
def equation := total_goals_4_matches + goals_5th_match = total_goals_5_matches

-- Needed to refer to the new and old averages
def new_average := total_goals_5_matches / 5
def old_average := A

-- Mathematical proof statement
theorem average_increase (h : equation) : (new_average - old_average) = 0.2 :=
by sorry

end average_increase_l483_483968


namespace area_of_common_region_l483_483182

theorem area_of_common_region (β : ℝ) (h1 : 0 < β ∧ β < π / 2) (h2 : Real.cos β = 3 / 5) :
  ∃ (area : ℝ), area = 4 / 9 := 
by 
  sorry

end area_of_common_region_l483_483182


namespace eccentricity_range_of_isosceles_right_triangle_l483_483665

theorem eccentricity_range_of_isosceles_right_triangle
  (a : ℝ) (e : ℝ)
  (ellipse_eq : ∀ (x y : ℝ), (x^2)/(a^2) + y^2 = 1)
  (h_a_gt_1 : a > 1)
  (B C : ℝ × ℝ)
  (isosceles_right_triangle : ∀ (A B C : ℝ × ℝ), ∃ k : ℝ, k > 0 ∧ 
    B = (-(2*k*a^2)/(1 + a^2*k^2), 0) ∧ 
    C = ((2*k*a^2)/(a^2 + k^2), 0) ∧ 
    (B.1^2 + B.2^2 = C.1^2 + C.2^2 + 1))
  (unique_solution : ∀ (k : ℝ), ∃! k', k' = 1)
  : 0 < e ∧ e ≤ (Real.sqrt 6) / 3 :=
sorry

end eccentricity_range_of_isosceles_right_triangle_l483_483665


namespace expected_waiting_time_correct_l483_483231

noncomputable def combined_average_bites_per_5_minutes := 6
def average_waiting_time_for_first_bite_in_seconds : ℝ := 50

theorem expected_waiting_time_correct :
  (1 / combined_average_bites_per_5_minutes) * 300 = average_waiting_time_for_first_bite_in_seconds :=
by
  sorry

end expected_waiting_time_correct_l483_483231


namespace arithmetic_sequence_find_m_l483_483058

theorem arithmetic_sequence_find_m (S : ℕ → ℤ) (m : ℕ)
  (h1 : S (m - 1) = -2)
  (h2 : S m = 0)
  (h3 : S (m + 1) = 3) :
  m = 5 :=
sorry

end arithmetic_sequence_find_m_l483_483058


namespace quadrilateral_ABCD_properties_l483_483740

structure quadrilateral :=
  (A B C D : Point)
  (AB : segment A B)
  (BC : segment B C)
  (CD : segment C D)
  (AD : segment A D)
  (∠B : angle B)
  (∠C : angle C)
  (AB_len : AB.length = 5)
  (BC_len : BC.length = 6)
  (CD_len : CD.length = 7)
  (∠B_value : ∠B = 130)
  (∠C_value : ∠C = 110)

def quadrilateral_property (q : quadrilateral) : Prop :=
  (q.area = 31.23) ∧ (q.perimeter = 24)

theorem quadrilateral_ABCD_properties :
  (exists q : quadrilateral, quadrilateral_property q) :=
by
  sorry

end quadrilateral_ABCD_properties_l483_483740


namespace sin_double_angle_l483_483007

noncomputable def r := Real.sqrt 5
noncomputable def sin_α := -2 / r
noncomputable def cos_α := 1 / r
noncomputable def sin_2α := 2 * sin_α * cos_α

theorem sin_double_angle (α : ℝ) :
  (∃ P : ℝ × ℝ, P = (1, -2) ∧ ∃ α : ℝ, true) → sin_2α = -4 / 5 :=
by
  sorry

end sin_double_angle_l483_483007


namespace min_area_ratio_of_equilateral_and_right_triangle_l483_483681

noncomputable def min_area_ratio (ABC : Triangle) (DEF : Triangle) :=
  (∃ (A B C : Point) (D E F : Point),
    Equilateral ABC ∧
    Right DEF ∧
    (on_side ABC A B C D E F) ∧
    (angle DEF = π / 2) ∧
    (angle EDF = π / 6) ∧
    min (S DEF / S ABC) = (3 / 14))

-- This statement asserts that given the conditions, the minimum value of the ratio of areas is 3/14
theorem min_area_ratio_of_equilateral_and_right_triangle (ABC DEF : Triangle) :
  min_area_ratio ABC DEF :=
sorry

end min_area_ratio_of_equilateral_and_right_triangle_l483_483681


namespace mode_and_median_of_donations_l483_483577

-- Define the dataset
def donations: List ℕ := List.replicate 3 20 ++ List.replicate 7 30 ++ List.replicate 5 35 ++ List.replicate 15 50 ++ List.replicate 10 100

-- Define what it means to be the mode
def is_mode (a: ℕ) (l: List ℕ) : Prop :=
  ∀ x ∈ l, l.count x ≤ l.count a

-- Define what it means to be the median
def is_median (a: ℕ) (l: List ℕ) : Prop :=
  let sorted_l := l.qsort (· ≤ ·)
  let n := l.length
  if n % 2 = 0 then a = (sorted_l.get! (n / 2 - 1) + sorted_l.get! (n / 2)) / 2
  else a = sorted_l.get! (n / 2)

-- The theorem to prove
theorem mode_and_median_of_donations :
    is_mode 50 donations ∧ is_median 50 donations :=
begin
  sorry
end

end mode_and_median_of_donations_l483_483577


namespace polar_equations_and_alpha_value_l483_483006

theorem polar_equations_and_alpha_value (theta: ℝ) (alpha: ℝ) 
  (0 < alpha ∧ alpha < (Real.pi / 2)) 
  (hx : ∀ x y, (x + 2 * y - 1 = 0) → (x = (cos theta) * (1 / (cos theta + 2 * sin theta))) 
  )
  (c_eq : ∀ (varphi : ℝ), (3 + 3 * cos varphi) = 6 * cos theta ∧ (3 * sin varphi) = 6 * sin theta)
  (OP : ℝ) (OQ : ℝ)
  (hOP : OP = 6 * cos alpha)
  (hOQ : OQ = 1 / |2 * cos alpha - sin alpha|)
  (hProduct : |OP * OQ| = 6):
  ∃ rho₁ rho₂, (rho₁ = 1 / (cos theta + 2 * sin theta) ∧ rho₂ = 6 * cos theta) ∧ (alpha = Real.pi / 4) :=
  sorry

end polar_equations_and_alpha_value_l483_483006


namespace max_area_BPC_l483_483385

noncomputable def triangle_area_max (AB BC CA : ℝ) (D : ℝ) : ℝ :=
  if h₁ : AB = 13 ∧ BC = 15 ∧ CA = 14 then
    112.5 - 56.25 * Real.sqrt 3
  else 0

theorem max_area_BPC : triangle_area_max 13 15 14 D = 112.5 - 56.25 * Real.sqrt 3 := by
  sorry

end max_area_BPC_l483_483385


namespace ellipse_eccentricity_l483_483685

theorem ellipse_eccentricity (a b c : ℝ) (h_eq : a * a = 16) (h_b : b * b = 12) (h_c : c * c = a * a - b * b) :
  c / a = 1 / 2 :=
by
  sorry

end ellipse_eccentricity_l483_483685


namespace TShapeGridSum_l483_483434

theorem TShapeGridSum (A B C D E F G H I : ℕ) (h1 : A ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) (h2 : B ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) (h3 : C ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) (h4 : D ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) (h5 : E ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) (h6 : F ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) (h7 : G ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) (h8 : H ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) (h9 : I ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9})
  (h_unique : list.nodup [A, B, C, D, E, F, G, H, I])
  (h_vertical : A + B + C + D + E = 31)
  (h_horizontal : F + C + G + H + I = 26) :
  A + B + C + D + E + F + G + H + I = 43 :=
by
  sorry

end TShapeGridSum_l483_483434


namespace hike_people_count_l483_483813

theorem hike_people_count :
  let cars := 3
  let taxis := 6
  let vans := 2
  let people_per_car := 4
  let people_per_taxi := 6
  let people_per_van := 5
  let total_people := (cars * people_per_car) + (taxis * people_per_taxi) + (vans * people_per_van)
  in total_people = 58 :=
by
  -- Proof steps will go here
  sorry

end hike_people_count_l483_483813


namespace f_1986_eq_one_l483_483039

def f : ℕ → ℤ := sorry

axiom f_def (a b : ℕ) : f (a + b) = f a + f b - 2 * f (a * b) + 1
axiom f_one : f 1 = 1

theorem f_1986_eq_one : f 1986 = 1 :=
sorry

end f_1986_eq_one_l483_483039


namespace marty_paint_combinations_l483_483427

theorem marty_paint_combinations : 
  let colors := {red, blue, green, yellow, black}
  let methods := {brush, roller, sponge}
  let valid_combinations := 
    (∀ c ∈ colors, c ≠ black → methods) ∪ ({brush, sponge} if black ∈ colors)
  Σ c in colors, #(valid_combinations c) = 14 :=
by
  sorry

end marty_paint_combinations_l483_483427


namespace find_f3_l483_483803

theorem find_f3
  (f : ℝ → ℝ)
  (h : ∀ x : ℝ, f(x) + 2 * f(2 - x) = 6 * x^2 - 4 * x + 1) :
  f 3 = -7 :=
sorry

end find_f3_l483_483803


namespace rectangle_area_invariance_l483_483363

theorem rectangle_area_invariance (θ : ℝ) :
  let vertices := [(-9, 1), (1, 1), (1, -8), (-9, -8)],
      original_length := (1 - (-9)),
      original_width := (1 - (-8)),
      original_area := original_length * original_width
  in original_area = 90 :=
by
  let vertices := [(-9, 1), (1, 1), (1, -8), (-9, -8)]
  let original_length := (1 - (-9))
  let original_width := (1 - (-8))
  let original_area := original_length * original_width
  exact sorry

end rectangle_area_invariance_l483_483363


namespace oranges_needed_l483_483059

theorem oranges_needed 
  (total_fruit_needed : ℕ := 12) 
  (apples : ℕ := 3) 
  (bananas : ℕ := 4) : 
  total_fruit_needed - (apples + bananas) = 5 :=
by 
  sorry

end oranges_needed_l483_483059


namespace cot_tan_simplified_l483_483463

theorem cot_tan_simplified :
  (Real.cot (Real.pi / 9) + Real.tan (Real.pi / 18) = Real.csc (Real.pi / 9)) :=
by
  sorry

end cot_tan_simplified_l483_483463


namespace hyperbola_eccentricity_l483_483645

theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (F : ℝ × ℝ) (A : ℝ × ℝ) (O : ℝ × ℝ) 
  (h3 : (F.1 - O.1)^2 / a^2 - (F.2 - O.2)^2 / b^2 = 1)
  (h4 : F = (sqrt(a^2 + b^2), 0)) -- assuming F is at (c,0)
  (h5 : A.1 = F.1 ∧ A.2 = (F.2 - A.2) * (b/a))
  (h6 : F ≠ O ∧ A ≠ O ∧ F ≠ A)
  (angle_cond : ∃ θ, θ = Real.arctan (b/a) ∧ θ = 2 * θ) :
  let e := Real.sqrt (1 + (b/a)^2) in 
  e = 2 * Real.sqrt 3 / 3 := sorry

end hyperbola_eccentricity_l483_483645


namespace sandy_books_from_second_shop_l483_483459

noncomputable def books_from_second_shop (books_first: ℕ) (cost_first: ℕ) (cost_second: ℕ) (avg_price: ℕ): ℕ :=
  let total_cost := cost_first + cost_second
  let total_books := books_first + (total_cost / avg_price) - books_first
  total_cost / avg_price - books_first

theorem sandy_books_from_second_shop :
  books_from_second_shop 65 1380 900 19 = 55 :=
by
  sorry

end sandy_books_from_second_shop_l483_483459


namespace average_waiting_time_for_first_bite_l483_483220

/-- 
Let S be a period of 5 minutes (300 seconds).
- We have an average of 5 bites in 300 seconds on the first fishing rod.
- We have an average of 1 bite in 300 seconds on the second fishing rod.
- The total average number of bites on both rods during this period is 6 bites.
The bites occur independently and follow a Poisson process.

We aim to prove that the waiting time for the first bite, given these conditions, is 
expected to be 50 seconds.
-/
theorem average_waiting_time_for_first_bite :
  let S := 300 -- 5 minutes in seconds
  -- The average number of bites on the first and second rod in period S.
  let avg_bites1 := 5
  let avg_bites2 := 1
  let total_avg_bites := avg_bites1 + avg_bites2
  -- The rate parameter λ for the Poisson process is total_avg_bites / S.
  let λ := total_avg_bites / S
  -- The average waiting time for the first bite.
  1 / λ = 50 :=
by
  let S := 300
  let avg_bites1 := 5
  let avg_bites2 := 1
  let total_avg_bites := avg_bites1 + avg_bites2
  let λ := total_avg_bites / S
  -- convert λ to seconds to ensure unit consistency
  have hλ: λ = 6 / 300 := rfl
  -- The expected waiting time for the first bite is 1 / λ
  have h_waiting_time: 1 / λ = 300 / 6 := by
    rw [hλ, one_div, div_div_eq_mul]
    norm_num
  exact h_waiting_time

end average_waiting_time_for_first_bite_l483_483220


namespace complex_expression_equality_l483_483620

open Complex

theorem complex_expression_equality (i : ℂ) (h : i^2 = -1) :
  (1 + i)^20 - (1 - i)^20 = 0 := 
sorry

end complex_expression_equality_l483_483620


namespace integral_solution_l483_483524

noncomputable def integral_problem : Prop :=
  ∫ (x : ℝ) in set.univ, (1 + x ^ (4 / 5)) ^ (1 / 3) / (x ^ 2 * x ^ (1 / 15)) = 
  -15 / 16 * (1 + x ^ (4 / 5)) ^ (4 / 3) / x ^ (16 / 18) + C

theorem integral_solution :
  integral_problem := sorry

end integral_solution_l483_483524


namespace smallest_n_l483_483786

theorem smallest_n (k t : ℤ) (hk : k ≡ 1 [MOD 7]) (ht : t > 0) : 
  ∃ (n : ℕ), n > 15 ∧ n ≡ 4 [MOD 6] ∧ n ≡ 3 [MOD 7] ∧ n = 10 + 42 * t :=
by {
  use 10 + 42 * t,
  split,
  {
    exact ht,
  },
  split,
  {
    unfold n,
    norm_num,
    exact hk,
  },
  {
    norm_num,
  }
}

end smallest_n_l483_483786


namespace totalCupsOfLiquid_l483_483582

def amountOfOil : ℝ := 0.17
def amountOfWater : ℝ := 1.17

theorem totalCupsOfLiquid : amountOfOil + amountOfWater = 1.34 := by
  sorry

end totalCupsOfLiquid_l483_483582


namespace problem_R_l483_483414

noncomputable def R (g S h : ℝ) : ℝ := g * S + h

theorem problem_R {g h : ℝ} (h_h : h = 6 - 4 * g) :
  R g 14 h = 56 :=
by
  sorry

end problem_R_l483_483414


namespace midpoint_of_ST_l483_483030

-- Definitions based on the given conditions
variables 
  (A B C D O M T S : Type)
  [AffineSpace ℝ A B C D O M T S]

-- Given conditions
variables 
  (convex_quadrilateral : ConvexQuadrilateral A B C D)
  (O_intersection : Intersection (Diagonals A B C D) O)
  (circle_OAD : Circumcircle O A D)
  (circle_OBC : Circumcircle O B C)
  (circle_OAB : Circumcircle O A B)
  (circle_OCD : Circumcircle O C D)
  (M_meet_OAD_OBC : MeetsCircumcircles O A D O B C O M)
  (T_meet_OAB : MeetsCircumcircle O M O A B T)
  (S_meet_OCD : MeetsCircumcircle O M O C D S)

-- Question to be proved
theorem midpoint_of_ST (h : Midpoint M S T) : True :=
sorry

end midpoint_of_ST_l483_483030


namespace area_of_union_of_seven_triangles_l483_483788

theorem area_of_union_of_seven_triangles :
  let s := 2
  let area_one_triangle := (sqrt 3 / 4) * s^2
  let total_area_without_overlaps := 7 * area_one_triangle
  let area_small_overlap := (sqrt 3 / 4) * (s / 2)^2
  let total_overlapping_area := 6 * area_small_overlap
  let net_area := total_area_without_overlaps - total_overlapping_area
  net_area = 11 * sqrt 3 / 2 :=
begin
  sorry
end

end area_of_union_of_seven_triangles_l483_483788


namespace sqrt_12_same_type_sqrt_3_l483_483192

-- We define that two square roots are of the same type if one is a multiple of the other
def same_type (a b : ℝ) : Prop := ∃ k : ℝ, b = k * a

-- We need to show that sqrt(12) is of the same type as sqrt(3), and check options
theorem sqrt_12_same_type_sqrt_3 : same_type (Real.sqrt 3) (Real.sqrt 12) ∧
  ¬ same_type (Real.sqrt 3) (Real.sqrt 8) ∧
  ¬ same_type (Real.sqrt 3) (Real.sqrt 18) ∧
  ¬ same_type (Real.sqrt 3) (Real.sqrt 6) :=
by
  sorry -- Proof is omitted


end sqrt_12_same_type_sqrt_3_l483_483192


namespace nine_consecutive_arithmetic_mean_divisible_1111_l483_483869

theorem nine_consecutive_arithmetic_mean_divisible_1111 {n : ℕ} (h1 : ∀ i : ℕ, 0 ≤ i ∧ i < 9 → Nat.Prime (n + i)) :
  ∃ n : ℕ, (∀ k : ℕ, 0 ≤ k ∧ k < 9 → (n + k) ∣ 1111) → (n + 4) = 97 := by
  sorry

end nine_consecutive_arithmetic_mean_divisible_1111_l483_483869


namespace jill_peaches_l483_483398

-- Definitions based on conditions in a
def Steven_has_peaches : ℕ := 19
def Steven_more_than_Jill : ℕ := 13

-- Statement to prove Jill's peaches
theorem jill_peaches : (Steven_has_peaches - Steven_more_than_Jill = 6) :=
by
  sorry

end jill_peaches_l483_483398


namespace enclosed_area_eq_two_l483_483612

noncomputable def enclosed_area : ℝ :=
  -∫ x in (2 * Real.pi / 3)..Real.pi, (Real.sin x - Real.sqrt 3 * Real.cos x)

theorem enclosed_area_eq_two : enclosed_area = 2 := 
  sorry

end enclosed_area_eq_two_l483_483612


namespace proof_B_proof_C_l483_483131

-- Definition for problem B
def satisfies_condition_B (x : ℝ) : Prop :=
  x < 1/3

def func_B (x : ℝ) : ℝ :=
  3 * x + (1 / (3 * x - 1))

-- Definition for problem C
def satisfies_conditions_C (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x + y + x * y = 3

def func_C (x y : ℝ) : ℝ :=
  x * y

-- Proof statements
theorem proof_B (x : ℝ) (h : satisfies_condition_B x) : ∀ y, func_B x ≤ -1 :=
  sorry

theorem proof_C (x y : ℝ) (h : satisfies_conditions_C x y) : func_C x y ≤ 1 :=
  sorry

end proof_B_proof_C_l483_483131


namespace mayoral_election_votes_l483_483735

theorem mayoral_election_votes (Y Z : ℕ) 
  (h1 : 22500 = Y + Y / 2) 
  (h2 : 15000 = Z - Z / 5 * 2)
  : Z = 25000 := 
  sorry

end mayoral_election_votes_l483_483735


namespace number_of_visits_at_least_students_l483_483793

-- Definitions
variables {α : Type} (students : Finset α) (groups : Finset (Finset α))
variable (n : ℕ) -- number of students
variable (m : ℕ) -- number of group visits

-- Conditions
def no_set_contains_all_elements : Prop :=
  ∀ g ∈ groups, ¬(students ⊆ g)

def each_pair_in_exactly_one_set : Prop :=
  ∀ (a b : α), a ≠ b → ∃! g ∈ groups, a ∈ g ∧ b ∈ g

def elements_appear_together_once : Prop :=
  ∀ g1 g2 ∈ groups, g1 ≠ g2 → ∀ a ∈ g1, ∀ b ∈ g1, (a = b) ∨ (a ∉ g2) ∨ (b ∉ g2)

-- Theorem statement
theorem number_of_visits_at_least_students (m_gt_one : m > 1)
  (students_card : students.card = n)
  (groups_card : groups.card = m)
  (cond1 : no_set_contains_all_elements students groups)
  (cond2 : each_pair_in_exactly_one_set students groups)
  (cond3 : elements_appear_together_once groups) :
  m ≥ n :=
sorry

end number_of_visits_at_least_students_l483_483793


namespace volume_pyramid_SPQR_l483_483779

variables (P Q R S : Type) [metric_space S]
variables (SP SQ SR : ℝ)

-- Define the conditions
axiom h1 : SP = 12
axiom h2 : SQ = 12
axiom h3 : SR = 8

axiom perp_SP_SQ : ∀ (S P Q : S), is_orthogonal S P Q
axiom perp_SQ_SR : ∀ (S Q R : S), is_orthogonal S Q R
axiom perp_SR_SP : ∀ (S R P : S), is_orthogonal S R P

-- Define the theorem to prove
theorem volume_pyramid_SPQR :
  \text{∃} (V : ℝ), V = (1 / 3) * (1 / 2) * SP * SQ * SR := 
  sorry

end volume_pyramid_SPQR_l483_483779


namespace mike_total_rose_bushes_l483_483061

-- Definitions based on the conditions
def costPerRoseBush : ℕ := 75
def costPerTigerToothAloe : ℕ := 100
def numberOfRoseBushesForFriend : ℕ := 2
def totalExpenseByMike : ℕ := 500
def numberOfTigerToothAloe : ℕ := 2

-- The total number of rose bushes Mike bought
noncomputable def totalNumberOfRoseBushes : ℕ :=
  let totalSpentOnAloes := numberOfTigerToothAloe * costPerTigerToothAloe
  let amountSpentOnRoseBushes := totalExpenseByMike - totalSpentOnAloes
  let numberOfRoseBushesForMike := amountSpentOnRoseBushes / costPerRoseBush
  numberOfRoseBushesForMike + numberOfRoseBushesForFriend

-- The theorem to prove
theorem mike_total_rose_bushes : totalNumberOfRoseBushes = 6 :=
  by
    sorry

end mike_total_rose_bushes_l483_483061


namespace books_per_box_l483_483898

theorem books_per_box (total_books : ℕ) (boxes : ℕ) (books_in_each_box : ℕ) 
  (h1 : total_books = 24) (h2 : boxes = 8) : books_in_each_box = 3 :=
by {
  have h3 : total_books / boxes = books_in_each_box := sorry,
  rw [h1, h2] at h3,
  exact h3,
  sorry
}

end books_per_box_l483_483898


namespace polygon_sides_l483_483963

theorem polygon_sides (n : ℕ) (h : (n-3) * 180 < 2008 ∧ 2008 < (n-1) * 180) : 
  n = 14 :=
sorry

end polygon_sides_l483_483963


namespace proof_problem1_proof_problem2_l483_483210

noncomputable def problem1 : ℝ := real.sqrt 16 + 2 * real.sqrt 9 - real.cbrt 27
noncomputable def problem2 : ℝ := abs (1 - real.sqrt 2) + real.sqrt 4 - real.cbrt (-8)

theorem proof_problem1 : problem1 = 7 := by
  sorry

theorem proof_problem2 : problem2 = real.sqrt 2 + 3 := by
  sorry

end proof_problem1_proof_problem2_l483_483210


namespace determine_constants_l483_483328

-- Define the function f based on given parameters and conditions
def f (a b c x : ℝ) := a * Real.pow b x + c

-- Mathematical conditions and the proof problem statement
theorem determine_constants : ∃ (a c : ℝ), (∀ (x : ℝ), 0 ≤ x → f a b c x ≥ -2) ∧ (∀ (x : ℝ), 0 ≤ x → f a b c x < 3) → a = -5 ∧ c = 3 :=
by 
  -- Given details and constraints on b
  let b : ℝ := sorry, -- Placeholder for the fact 0 < b < 1
  sorry -- Placeholder for the main proof logic

end determine_constants_l483_483328


namespace least_positive_four_digit_solution_l483_483615

theorem least_positive_four_digit_solution :
  ∃ x : ℕ, 1000 ≤ x ∧ x < 10000 ∧ 
    (5 * x) % 10 = 10 % 10 ∧ 
    (3 * x + 20) % 12 = 29 % 12 ∧ 
    (-3 * x + 2) % 30 = (2 * x) % 30 ∧ 
    x = 1002 :=
by
  sorry

end least_positive_four_digit_solution_l483_483615


namespace ellipse_eq_proof_slope_proof_l483_483941

variables (a b : ℝ) (e : ℝ) (P Q A B : ℝ × ℝ)

-- Given conditions
def ellipse_eq := ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1
def focal_distance := e = (√3)/2
def P_on_ellipse := P = (2, -1)
def a_greater_b := a > b ∧ b > 0
def parallel_line_PQ := P.2 = Q.2
def line_slope (p1 p2 : ℝ × ℝ) := (p2.2 - p1.2) / (p2.1 - p1.1)
def bisects_angle_P := ∀ (x1 y1 x2 y2 : ℝ), A = (x1, y1) ∧ B = (x2, y2) → 
                       (P.2 - A.2) / (P.1 - A.1) = -(P.2 - B.2) / (P.1 - B.1)

-- Prove parts
theorem ellipse_eq_proof : focal_distance a b e → ellipse_eq a b e P → P_on_ellipse P → a_greater_b a b → ellipse_eq a b := 
by sorry

theorem slope_proof : ellipse_eq a b → parallel_line_PQ P Q → bisects_angle_P P A B → 
                      ∃ k : ℝ, (line_slope A B) = k ∧ k = -1/2 := 
by sorry

end ellipse_eq_proof_slope_proof_l483_483941


namespace range_of_b_l483_483767

-- Definition of the function f
def f (x b : ℝ) : ℝ := 4 * x^3 + b * x + 1

-- The theorem to prove b = -3
theorem range_of_b (b : ℝ) : (∀ x : ℝ, x ∈ Icc (-1) 1 -> f x b ≥ 0) ↔ b = -3 :=
by
  sorry

end range_of_b_l483_483767


namespace smallest_arithmetic_mean_divisible_by_1111_l483_483857

/-- 
Given the product of nine consecutive natural numbers is divisible by 1111, 
prove that the smallest possible value of the arithmetic mean of these nine numbers is 97.
-/
theorem smallest_arithmetic_mean_divisible_by_1111 :
  ∃ n : ℕ, (∀ k : ℕ, k = n →  (∏ i in finset.range 9, k + i) % 1111 = 0) 
  ∧ (n ≥ 93) ∧ (n + 4 = 97) :=
sorry

end smallest_arithmetic_mean_divisible_by_1111_l483_483857


namespace find_valid_numbers_l483_483969

-- Define n as a four-digit number ABCD where A, B, C, and D are digits
def valid_digits : List ℕ := [1, 3, 5, 7, 9]

-- Examine all conditions and correct answers
theorem find_valid_numbers :
  {n | ∃ A B C D,
       n = 1000 * A + 100 * B + 10 * C + D ∧
       A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
       A ∈ valid_digits ∧ B ∈ valid_digits ∧ C ∈ valid_digits ∧ D ∈ valid_digits ∧
       n % A = 0 ∧ n % B = 0 ∧ n % C = 0 ∧ n % D = 0}
  = {1395, 1935, 3195, 3915, 9135, 9315} :=
by
  sorry

end find_valid_numbers_l483_483969


namespace find_overlapping_area_l483_483975

-- Definitions based on conditions
def length_total : ℕ := 16
def length_strip1 : ℕ := 9
def length_strip2 : ℕ := 7
def area_only_strip1 : ℚ := 27
def area_only_strip2 : ℚ := 18

-- Widths are the same for both strips, hence areas are proportional to lengths
def area_ratio := (length_strip1 : ℚ) / (length_strip2 : ℚ)

-- The Lean statement to prove the question == answer
theorem find_overlapping_area : 
  ∃ S : ℚ, (area_only_strip1 + S) / (area_only_strip2 + S) = area_ratio ∧ 
              area_only_strip1 + S = area_only_strip1 + 13.5 := 
by 
  sorry

end find_overlapping_area_l483_483975


namespace equation_of_perpendicular_line_l483_483949

theorem equation_of_perpendicular_line : 
  ∃ (a b c : ℝ), (∀ (x y : ℝ), 2 * x - y + 1 = 0 → x + 2 * y - 1 = 0) ∧ ∀ (x y : ℝ), point (1, 3) ∧ perpendicular_to_line x y (2 * x - y + 1) = 0 → x + 2 * y - 1 = 0 :=
begin
  sorry
end

end equation_of_perpendicular_line_l483_483949


namespace trajectory_and_product_of_slopes_constant_l483_483647

-- Given a circle
def on_circle (P : ℝ × ℝ) : Prop := P.1^2 + P.2^2 = 4

-- Define the midpoint condition for the line PQ
def midpoint (P Q M : ℝ × ℝ) : Prop := M.1 = (P.1 + Q.1) / 2 ∧ M.2 = (P.2 + Q.2) / 2

-- Specify trajectory C of point M
def trajectory (M : ℝ × ℝ) : Prop := M.1^2 / 4 + M.2^2 = 1 ∧ M.2 ≠ 0

-- Define the line intersection condition
def on_line (k : ℝ) (A B : ℝ × ℝ) : Prop := A.2 = k * A.1 ∧ B.2 = k * B.1

-- Problem statement: Proving both parts
theorem trajectory_and_product_of_slopes_constant
(P Q A B M : ℝ × ℝ) (k : ℝ) :
  on_circle P →
  midpoint P Q M →
  on_line k A B →
  trajectory M →
  (P ≠ Q) →
  (∃ K : ℝ, let K1 := (M.2 - A.2) / (M.1 - A.1),
                   K2 := (M.2 + A.2) / (M.1 + A.1)
               in K1 * K2 = K) :=
by
  sorry

end trajectory_and_product_of_slopes_constant_l483_483647


namespace rectangle_area_l483_483180

theorem rectangle_area (sq_area : ℕ) (rect_length_factor : ℕ) (h_sq_area : sq_area = 49) 
                       (h_rect_length_factor : rect_length_factor = 3) : 
  let sq_side := nat.sqrt sq_area in
  let rect_width := sq_side in
  let rect_length := rect_length_factor * rect_width in
  rect_width * rect_length = 147 :=
by
  -- assume given conditions
  have h_sq_side : sq_side = 7 := by sorry
  have h_rect_width : rect_width = 7 := by sorry
  have h_rect_length : rect_length = 21 := by sorry
  show rect_width * rect_length = 147 from by sorry

end rectangle_area_l483_483180


namespace max_C_inequality_l483_483616

theorem max_C_inequality :
  let C := ( Real.sqrt (13 + 16 * Real.sqrt 2) - 1) / 2 in
  ∀ (x y z : ℝ), 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z →
    x^3 + y^3 + z^3 + C * (x * y^2 + y * z^2 + z * x^2) ≥ (C + 1) * (x^2 * y + y^2 * z + z^2 * x) :=
by
  intros
  sorry

end max_C_inequality_l483_483616


namespace exists_parallel_line_intersecting_polygonal_line_at_least_101_times_l483_483748

theorem exists_parallel_line_intersecting_polygonal_line_at_least_101_times :
  ∃ l : ℝ, l < 1 ∧ 
  (∃ line : set (ℝ × ℝ), (line ⊆ {(x, y) | x = l ∨ y = l}) ∧
  ∀ P : set (ℝ × ℝ), 
      (∀ Q : ℝ × ℝ, Q ∈ P → (Q.1 = l ∨ Q.2 = l)) ∧
      (length_of_polygonal_line P (λ xy, xy ∈ square (0,0) (1,1)) ≥ 200) →
      ∃ k : ℕ, k ≥ 101 ∧ 
      (∃ points : fin k → (ℝ × ℝ), 
        (∀ i, i < k → 
          (points i ∈ line ∧ points i ∈ P) ∧
          (∀ j, j < i → points i ≠ points j)))) :=
by
  sorry

end exists_parallel_line_intersecting_polygonal_line_at_least_101_times_l483_483748


namespace tangent_line_at_1_range_of_a_l483_483325

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - Real.log x

theorem tangent_line_at_1 :
  let a := 1
  y = f a x at the point (1, f a 1) ⊢
  y = x := 
sorry

theorem range_of_a 
  (h : ∀ x ∈ Ioo 0 1, |f a x| ≥ 1) : 
  a ≥ Real.exp(1) / 2 := 
sorry

end tangent_line_at_1_range_of_a_l483_483325


namespace additional_hours_q_l483_483940

variable (P Q : ℝ)

theorem additional_hours_q (h1 : P = 1.5 * Q) 
                           (h2 : P = Q + 8) 
                           (h3 : 480 / P = 20):
  (480 / Q) - (480 / P) = 10 :=
by
  sorry

end additional_hours_q_l483_483940


namespace prime_sum_probability_l483_483099

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def spinner_a : Finset ℕ := {0, 2, 3}
def spinner_b : Finset ℕ := {1, 3, 5}

def prime_prob : ℚ :=
  let sums := Finset.product spinner_a spinner_b |>.image (λ ab, ab.1 + ab.2)
  let prime_sums := sums.filter is_prime
  (prime_sums.card : ℚ) / (sums.card : ℚ)

theorem prime_sum_probability : prime_prob = 5 / 9 := 
  sorry

end prime_sum_probability_l483_483099


namespace smallest_arithmetic_mean_l483_483874

noncomputable def S (n : ℕ) := (List.range' n 9).map Nat.ofNat

theorem smallest_arithmetic_mean (n : ℕ) (h1 : 93 ≤ n) (h2 : ∃ k ∈ S n, 11 ∣ k) (h3 : ∃ k ∈ S n, 101 ∣ k) : 
  (n + 4 = 97) := by
  sorry

end smallest_arithmetic_mean_l483_483874


namespace measure_of_angle_A_range_of_f_B_l483_483387

noncomputable def triangle_condition (a b c : ℝ) (A B C : ℝ) : Prop :=
  (a + c) * (Real.sin A - Real.sin C) = Real.sin B * (b - c)

theorem measure_of_angle_A (a b c A B C: ℝ) (h1: triangle_condition a b c A B C) : A = π / 3 :=
sorry

noncomputable def f (x : ℝ) : ℝ := √3 * Real.sin x * Real.cos x - (Real.cos x)^2 + 1/2

theorem range_of_f_B (B : ℝ) (A B C: ℝ) (h1: A = π / 3) (h2: 0 < B ∧ B < π - A) :
  -1/2 < f B ∧ f B ≤ 1 :=
sorry

end measure_of_angle_A_range_of_f_B_l483_483387


namespace trigonometric_identity_l483_483636

-- Define the problem conditions and formulas
variables (α : Real) (h : Real.cos (Real.pi / 6 + α) = Real.sqrt 3 / 3)

-- State the theorem
theorem trigonometric_identity : Real.cos (5 * Real.pi / 6 - α) = - (Real.sqrt 3 / 3) :=
by
  -- Placeholder for the proof
  sorry

end trigonometric_identity_l483_483636


namespace tangency_condition_l483_483703

-- Let f and g be linear functions with the same slope.
variables {a b A x : ℝ}

def f (x : ℝ) := a * x + b
def g (x : ℝ) := a * x + (b - 1)

theorem tangency_condition 
  (a_ne_zero : a ≠ 0) : 
  ((g x) ^ 2 = A * f x) → (A = 0 ∨ A = -4) :=
by {
  sorry -- Proof omitted.
}

end tangency_condition_l483_483703


namespace find_vertex_X_l483_483391

def midpoint (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((a.1 + b.1) / 2, (a.2 + b.2) / 2, (a.3 + b.3) / 2)

theorem find_vertex_X (M N P : ℝ × ℝ × ℝ) (X : ℝ × ℝ × ℝ)
  (hM : M = (3, 2, -3))
  (hN : N = (-1, 3, -5))
  (hP : P = (4, 0, 6))
  (hX : midpoint P M =  midpoint N X) :
  X = (8, -1, 8) := by
  sorry

end find_vertex_X_l483_483391


namespace polynomial_diff_operator_has_zero_l483_483052

variables {α : Type*} [LinearOrder α] [TopologicalSpace α] [OrderTopology α] [Nonempty α] [IsConnected α]
variables {n : ℕ}

-- Let p be a polynomial of degree n with leading coefficient 1 and all roots real
def p (x : ℝ) : Polynomial ℝ := sorry

-- Let f be an n-times differentiable function from [a, b] to ℝ with at least n + 1 distinct zeros
variables {a b : ℝ} {f : α → ℝ}
variable [DiffableOn f (Set.Ico a b) : ℕ)
variable distinct_zeros {f : (Set.Icc a b) → ℝ} at_least_n_plus_one_zeros : true

-- Show that p(D)f(x) has at least one zero on [a, b], where D is the differential operator
theorem polynomial_diff_operator_has_zero :
  ∃ x ∈ Set.Icc a b, (Polynomial.map (λ x, D) p).eval f x = 0 := sorry

end polynomial_diff_operator_has_zero_l483_483052


namespace area_PBC_less_half_probability_l483_483560

-- Definition of the problem
def area (base : ℝ) (height : ℝ) : ℝ := 0.5 * base * height

noncomputable def probability_area_PBC_half (A B C P : Point) (h : height P B C) : ℝ :=
if interior A B C P ∧ h < 0.5 * AC then 0.75 else 0.0

-- The theorem to prove
theorem area_PBC_less_half_probability (A B C P : Point) (h : height P B C) : 
  interior A B C P → 
  h < 0.5 * AC → 
  probability_area_PBC_half A B C P h = 0.75 :=
by 
  sorry

end area_PBC_less_half_probability_l483_483560


namespace smallest_arithmetic_mean_divisible_1111_l483_483845

theorem smallest_arithmetic_mean_divisible_1111 :
  ∃ n : ℕ, 93 ≤ n ∧ n + 4 = 97 ∧ (∀ i : ℕ, i ∈ finset.range 9 → (n + i) % 11 = 0 ∨ (n + i) % 101 = 0) :=
sorry

end smallest_arithmetic_mean_divisible_1111_l483_483845


namespace contribution_required_l483_483515

-- Definitions corresponding to the problem statement
def total_amount : ℝ := 2000
def number_of_friends : ℝ := 7
def your_contribution_factor : ℝ := 2

-- Prove that the amount each friend needs to raise is approximately 222.22
theorem contribution_required (x : ℝ) 
  (h : 9 * x = total_amount) :
  x = 2000 / 9 := 
  by sorry

end contribution_required_l483_483515


namespace smallest_difference_l483_483317

theorem smallest_difference : ∃ (a b c d e : ℕ), {a, b, c, d, e} = {1, 3, 7, 8, 9} ∧ 
                   a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
                   b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
                   c ≠ d ∧ c ≠ e ∧ 
                   d ≠ e ∧ 
                   100 * a + 10 * b + c - (10 * d + e) = 39 := 
sorry

end smallest_difference_l483_483317


namespace find_ones_placement_l483_483066

def cells := { (i : ℕ) // i < 25 }

def is_corner (i : cells) : Prop :=
  i = ⟨0, sorry⟩ ∨ i = ⟨4, sorry⟩ ∨ i = ⟨20, sorry⟩ ∨ i = ⟨24, sorry⟩

def is_edge (i : cells) : Prop :=
  ¬is_corner i ∧ (i.1 < 5 ∨ i.1 % 5 = 0 ∨ i.1 % 5 = 4 ∨ i.1 > 19)

def is_internal (i : cells) : Prop :=
  ¬is_corner i ∧ ¬is_edge i

axiom grid : cells → ℕ

axiom ones_and_zeroes : ∀ i : cells, grid i = 0 ∨ grid i = 1

axiom sixteen_ones : ∑ i : cells, grid i = 16

def sum_2x2 (i : cells) : ℕ :=
  if i.1 < 20 ∧ i.1 % 5 < 4 then 
    grid i + grid ⟨i.1 + 1, sorry⟩ + grid ⟨i.1 + 5, sorry⟩ + grid ⟨i.1 + 6, sorry⟩
  else
    0

axiom sum_2x2_subgrids : (∑ i, sum_2x2 ⟨i, sorry⟩) = 28

theorem find_ones_placement : 
  ∀ i : cells, (is_corner i ∨ is_edge i) → grid i = 1 :=
sorry

end find_ones_placement_l483_483066


namespace limit_of_I_k_diverges_l483_483606

noncomputable def I_k (k : ℝ) : ℝ := ∫ ∫ (x y : ℝ) in (0..k) × (0..k), (exp x - exp y) / (x - y)

theorem limit_of_I_k_diverges :
  filter.at_top.tendsto (λ k : ℝ, (exp (-k) * I_k k)) filter.at_top :=
sorry

end limit_of_I_k_diverges_l483_483606


namespace part_one_part_two_l483_483015

variable {A : Real}
variable {a b c : Real}
variable {m n : Vector Real}
variable {β γ : Real}

-- Conditions
axiom condition1 : m = Vector.mk (cos A) (sin A)
axiom condition2 : n = Vector.mk (cos A) (- sin A)
axiom condition3 : dot m n = 1 / 2
axiom condition4 : a = sqrt 5

-- (I) Proving the magnitude of angle A and the angle between vectors m and n
theorem part_one :
  A = π / 6 ∧ β = π / 3 := by
  sorry

-- (II) Proving the maximum area of triangle ABC
theorem part_two :
  area_max a b c A = (10 + 5 * sqrt 3) / 4 := by
  sorry

end part_one_part_two_l483_483015


namespace min_area_ratio_of_equilateral_and_right_triangle_l483_483680

noncomputable def min_area_ratio (ABC : Triangle) (DEF : Triangle) :=
  (∃ (A B C : Point) (D E F : Point),
    Equilateral ABC ∧
    Right DEF ∧
    (on_side ABC A B C D E F) ∧
    (angle DEF = π / 2) ∧
    (angle EDF = π / 6) ∧
    min (S DEF / S ABC) = (3 / 14))

-- This statement asserts that given the conditions, the minimum value of the ratio of areas is 3/14
theorem min_area_ratio_of_equilateral_and_right_triangle (ABC DEF : Triangle) :
  min_area_ratio ABC DEF :=
sorry

end min_area_ratio_of_equilateral_and_right_triangle_l483_483680


namespace find_k_l483_483664

variables {V : Type*} [inner_product_space ℝ V] 
(a b : V)
(k : ℝ)

open real

-- Conditions from the problem
axiom a_norm_eq_one : ∥a∥ = 1
axiom b_norm_eq_two : ∥b∥ = 2
axiom angle_a_b_sixty : real.angle a b = π / 3
axiom perp_condition : inner_product_space.orthogonal (k • a + b) b

-- The statement to prove
theorem find_k : k = -4 :=
sorry

end find_k_l483_483664


namespace nine_consecutive_arithmetic_mean_divisible_1111_l483_483870

theorem nine_consecutive_arithmetic_mean_divisible_1111 {n : ℕ} (h1 : ∀ i : ℕ, 0 ≤ i ∧ i < 9 → Nat.Prime (n + i)) :
  ∃ n : ℕ, (∀ k : ℕ, 0 ≤ k ∧ k < 9 → (n + k) ∣ 1111) → (n + 4) = 97 := by
  sorry

end nine_consecutive_arithmetic_mean_divisible_1111_l483_483870


namespace find_a_range_l483_483111

-- Definitions based on the conditions
def f (a : ℝ) (x : ℝ) : ℝ := (a - 1) * x + a * (Real.log (x - 2))

-- Main theorem statement
theorem find_a_range (a : ℝ) :
  (∀ x1 x2 : ℝ, 2 < x1 → 2 < x2 → f a x1 - f a x2 < -4) → a ≤ -3 :=
by
  sorry

end find_a_range_l483_483111


namespace max_sum_of_distinct_integers_l483_483012

theorem max_sum_of_distinct_integers (A B C : ℕ) (hABC_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C) (hProduct : A * B * C = 1638) :
  A + B + C ≤ 126 :=
sorry

end max_sum_of_distinct_integers_l483_483012


namespace total_people_on_hike_l483_483816

-- Definitions of the conditions
def n_cars : ℕ := 3
def n_people_per_car : ℕ := 4
def n_taxis : ℕ := 6
def n_people_per_taxi : ℕ := 6
def n_vans : ℕ := 2
def n_people_per_van : ℕ := 5

-- Statement of the problem
theorem total_people_on_hike : 
  n_cars * n_people_per_car + n_taxis * n_people_per_taxi + n_vans * n_people_per_van = 58 :=
by sorry

end total_people_on_hike_l483_483816


namespace find_m_if_f_is_even_l483_483687

theorem find_m_if_f_is_even 
  (f : ℝ → ℝ)
  (h : ∀ x : ℝ, f(x) = (m-1)*x^2 + (m-2)*x + (m^2 - 7m + 12))
  (even_f : ∀ x : ℝ, f(-x) = f(x)) :
  m = 2 :=
by
  sorry

end find_m_if_f_is_even_l483_483687


namespace particle_position_after_72_moves_l483_483558

def ω : ℂ := Complex.cis (Real.pi / 6)

def move (z : ℂ) : ℂ := ω * z + 6

def initial_position : ℂ := 3

noncomputable def position_after_moves (n : ℕ) : ℂ :=
  (Nat.iterate move n initial_position)

theorem particle_position_after_72_moves : position_after_moves 72 = 3 :=
by
  sorry

end particle_position_after_72_moves_l483_483558


namespace exists_10_element_subset_l483_483048

open Set

noncomputable def M : Set ℕ := { n | 1 ≤ n ∧ n ≤ 20 }

def is_9_element_subset (S : Set ℕ) : Prop :=
  S ⊆ M ∧ S.card = 9

def f : (Set ℕ) → ℕ := sorry -- Assume f is defined such that for all S, 1 ≤ f(S) ≤ 20

theorem exists_10_element_subset :
  ∃ T : Set ℕ, T ⊆ M ∧ T.card = 10 ∧ ∀ k ∈ T, f (T \ {k}) ≠ k :=
by
  sorry

end exists_10_element_subset_l483_483048


namespace sum_reciprocal_squares_l483_483553

open Real

theorem sum_reciprocal_squares (a : ℝ) (A B C D E F : ℝ)
    (square_ABCD : A = 0 ∧ B = a ∧ D = a ∧ C = a)
    (line_intersects : A = 0 ∧ E ≥ 0 ∧ E ≤ a ∧ F ≥ 0 ∧ F ≤ a) 
    (phi : ℝ) : 
    (cos phi * (a/cos phi))^2 + (sin phi * (a/sin phi))^2 = (1/a^2) := 
sorry 

end sum_reciprocal_squares_l483_483553


namespace total_amount_paid_l483_483970

theorem total_amount_paid (sets_of_drill_bits : ℕ) (cost_per_set : ℕ) (tax_rate : ℝ)
  (h1 : sets_of_drill_bits = 5) (h2 : cost_per_set = 6) (h3 : tax_rate = 0.10) : ℝ :=
  let total_cost_before_tax := sets_of_drill_bits * cost_per_set in
  let tax := total_cost_before_tax * tax_rate in
  let total_amount := total_cost_before_tax + tax in
  -- The final assertion
  total_amount = 33 := sorry

end total_amount_paid_l483_483970


namespace douglas_votes_in_county_X_l483_483366

theorem douglas_votes_in_county_X (V : ℝ) :
  (0.64 * (2 * V + V) - 0.4000000000000002 * V) / (2 * V) * 100 = 76 := by
sorry

end douglas_votes_in_county_X_l483_483366


namespace gcd_3pow600_minus_1_3pow612_minus_1_l483_483122

theorem gcd_3pow600_minus_1_3pow612_minus_1 :
  Nat.gcd (3^600 - 1) (3^612 - 1) = 531440 :=
by
  sorry

end gcd_3pow600_minus_1_3pow612_minus_1_l483_483122


namespace angle_C_max_triangle_area_l483_483727

variable (α : Type) [LinearOrderedField α] [RealField α]
variables (A B C a b c : α)

-- First part: Prove that C = 2π/3 given the condition
theorem angle_C (h1 : (2 * a + b) / c = cos (A + C) / cos C) :
  C = 2 * π / 3 := sorry

-- Second part: Given c = 2, find the values of a and b that maximize the area
theorem max_triangle_area (h2 : c = 2) (h3 : cos C = -1 / 2) :
  (a = b ∧ a = 2 * sqrt 3 / 3) :=
  sorry

end angle_C_max_triangle_area_l483_483727


namespace cyclic_quadrilateral_iff_condition_l483_483933

theorem cyclic_quadrilateral_iff_condition
  (α β γ δ : ℝ)
  (h : α + β + γ + δ = 2 * π) :
  (α * β + α * δ + γ * β + γ * δ = π^2) ↔ (α + γ = π ∧ β + δ = π) :=
by
  sorry

end cyclic_quadrilateral_iff_condition_l483_483933


namespace find_a_l483_483998

theorem find_a (a k : ℝ) (h1 : ∀ x, a * x^2 + 3 * x - k = 0 → x = 7) (h2 : k = 119) : a = 2 :=
by
  sorry

end find_a_l483_483998


namespace total_earnings_is_correct_l483_483768

def lloyd_normal_hours : ℝ := 7.5
def lloyd_rate : ℝ := 4.5
def lloyd_overtime_rate : ℝ := 2.0
def lloyd_hours_worked : ℝ := 10.5

def casey_normal_hours : ℝ := 8
def casey_rate : ℝ := 5
def casey_overtime_rate : ℝ := 1.5
def casey_hours_worked : ℝ := 9.5

def lloyd_earnings : ℝ := (lloyd_normal_hours * lloyd_rate) + ((lloyd_hours_worked - lloyd_normal_hours) * lloyd_rate * lloyd_overtime_rate)

def casey_earnings : ℝ := (casey_normal_hours * casey_rate) + ((casey_hours_worked - casey_normal_hours) * casey_rate * casey_overtime_rate)

def total_earnings : ℝ := lloyd_earnings + casey_earnings

theorem total_earnings_is_correct : total_earnings = 112 := by
  sorry

end total_earnings_is_correct_l483_483768


namespace f_monotonically_increasing_intervals_g_max_value_l483_483688

noncomputable def f (x : ℝ) := 2 * Real.cos x * (Real.sin x - Real.cos x) + 1
noncomputable def g (x : ℝ) := sqrt 2 * Real.sin (x / 2 + Real.pi / 4)

theorem f_monotonically_increasing_intervals :
    ∀ k : ℤ, ∀ x : ℝ, k * Real.pi - Real.pi / 8 ≤ x ∧ x ≤ k * Real.pi + 3 * Real.pi / 8 → 
    deriv f x > 0 := 
sorry

theorem g_max_value : 
    ∀ k : ℤ, ∃ x : ℝ, g x = sqrt 2 ∧ x = 2 * k * Real.pi + Real.pi / 4 :=
sorry

end f_monotonically_increasing_intervals_g_max_value_l483_483688


namespace card_sums_distinct_remainders_l483_483108

theorem card_sums_distinct_remainders (n : ℕ) (h : n ≥ 4) 
  (cards : Fin n → Fin n) (sums_distinct : ∀ i j : Fin n, i ≠ j → (cards i + cards ((i + 1) % n) + cards j + cards ((j + 1) % n)) % n ≠ (cards n.succ % n) % n) : 
  ∃ k, n = 2 * k :=
sorry

end card_sums_distinct_remainders_l483_483108


namespace square_area_dimensions_l483_483888

theorem square_area_dimensions (x : ℝ) (n : ℝ) : 
  (x^2 + (x + 12)^2 = 2120) → 
  (n = x + 12) → 
  (x = 26) → 
  (n = 38) := 
by
  sorry

end square_area_dimensions_l483_483888


namespace Michael_digging_time_l483_483430

theorem Michael_digging_time (father_rate : ℝ) (father_time : ℝ) (req_depth_diff : ℝ) : 
  (father_rate = 4) → 
  (father_time = 400) → 
  (req_depth_diff = 400) →
  (let father_depth := father_rate * father_time; 
       michael_depth := 2 * father_depth - req_depth_diff;
       michael_time := michael_depth / father_rate in michael_time = 700) :=
by
  intros hr_1 hr_2 hr_3
  have father_depth : ℝ := father_rate * father_time
  have michael_depth : ℝ := 2 * father_depth - req_depth_diff
  have michael_time : ℝ := michael_depth / father_rate
  sorry

end Michael_digging_time_l483_483430


namespace cyclic_quadrilateral_diagonals_l483_483644

variables (a b c d : ℝ)

theorem cyclic_quadrilateral_diagonals (h : Cyclic_Quadrilateral a b c d) :
  ∃ AC BD : ℝ, 
  AC = Real.sqrt ((a * c + b * d) * (a * d + b * c) / (a * b + c * d)) ∧ 
  BD = Real.sqrt ((a * c + b * d) * (a * b + c * d) / (a * d + b * c)) :=
begin
  sorry
end

end cyclic_quadrilateral_diagonals_l483_483644


namespace sine_double_angle_value_l483_483745

noncomputable def parametric_equations := 
  ∀ α t : ℝ, (x y : ℝ), 
  x = -1 + t * (Real.cos α) ∧ 
  y = -3 + t * (Real.sin α)

noncomputable def polar_equation := 
  ∀ θ : ℝ, (ρ : ℝ),
  ρ = 4 * (Real.cos θ)

theorem sine_double_angle_value (α : ℝ) (hα : α = Real.pi / 3) (hAB : ∀ t₁ t₂ : ℝ, |2| = sqrt ((6 * (Real.sin α + Real.cos α))^2 - 4 * 14)) : 
  Real.sin (2 * α) = 2 / 3 := sorry

end sine_double_angle_value_l483_483745


namespace problem_probability_l483_483034

noncomputable def Q (x : ℝ) : ℝ := x^2 - 4*x - 16

theorem problem_probability :
  let a := 1
  let b := 1
  let c := 1
  let d := 0
  let e := 14
  a + b + c + d + e = 17 ∧ 
  (∀ x, 6 ≤ x ∧ x ≤ 20 → ⌊(Q(x)).sqrt⌋ = (Q(⌊x⌋)).sqrt → x ∈ set.Ico 6 7) ∧
  (∀ x, 6 ≤ x → x < 7 → Q(x) < 1) :=
by
  sorry

end problem_probability_l483_483034


namespace triangle_perimeter_l483_483655

-- Define the triangle with sides a, b, c
structure Triangle :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

-- Define the predicate that checks if the triangle is isosceles
def isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.a = t.c ∨ t.b = t.c

-- Define the predicate that calculates the perimeter of the triangle
def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

-- State the problem
theorem triangle_perimeter : 
  ∃ (t : Triangle), isIsosceles t ∧ (    (t.a = 6 ∧ t.b = 9 ∧ perimeter t = 24)
                                       ∨ (t.b = 6 ∧ t.a = 9 ∧ perimeter t = 24)
                                       ∨ (t.c = 6 ∧ t.a = 9 ∧ perimeter t = 21)
                                       ∨ (t.a = 6 ∧ t.c = 9 ∧ perimeter t = 21)
                                       ∨ (t.b = 6 ∧ t.c = 9 ∧ perimeter t = 24)
                                       ∨ (t.c = 6 ∧ t.b = 9 ∧ perimeter t = 21)
                                    ) :=
sorry

end triangle_perimeter_l483_483655


namespace percentage_increase_l483_483024

theorem percentage_increase (D1 D2 : ℕ) (total_days : ℕ) (H1 : D1 = 4) (H2 : total_days = 9) (H3 : D1 + D2 = total_days) : 
  (D2 - D1) / D1 * 100 = 25 := 
sorry

end percentage_increase_l483_483024


namespace sum_series_l483_483619

theorem sum_series :
  (∑ k in Finset.range 2002, 1 / (k + 1) / (k + 2)) = 2002 / 2003 := 
by
  sorry

end sum_series_l483_483619


namespace monotonic_intervals_range_of_b_product_inequality_l483_483695

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x - Real.log (x + a)

theorem monotonic_intervals (a : ℝ) :
  (a ≥ 1 → ∀ x, x > -a → f x a > 0) ∧
  (a < 1 → (∀ x, x > 1 - a → f x a > 0) ∧ (∀ x, -a < x ∧ x < 1 - a → f x a < 0)) :=
sorry

theorem range_of_b (b : ℝ) :
  (∀ x, g x = x^2 - 3*x + Real.log x + b) → 
  (∀ x ∈ Icc (1/2 : ℝ) (2 : ℝ),  g' x = 2*x - 3 + 1/x) →
  ((5/4 + Real.log 2) ≤ b ∧ b < 2) :=
sorry

theorem product_inequality (n: ℕ) (hn: n ≥ 2) :
  (∏ i in Finset.range(n+1).filter(λ i, i > 1), (1 + 1/(i : ℝ)^2)) < Real.exp 1 :=
sorry

end monotonic_intervals_range_of_b_product_inequality_l483_483695


namespace students_arrangement_l483_483770

theorem students_arrangement {students : Finset ℕ} {activities : Finset ℕ}
  (h_students_count : students.card = 5)
  (h_activities_count : activities.card = 4)
  (h_artistic_count : ∃ artistic ∈ students, artistic.card = 2)
  (h_activities : activities = {0, 1, 2, 3})
  (h_conditions : ∀ student ∈ students, (∃! activity ∈ activities, participates_in(student, activity)) ∧
                 ∀ activity ∈ activities, 1 ≤ (participates_in.student ∩ {activity}).card)
  (h_artistic_performance : ∀ student ∈ artistic, participates_in(student, 0)) :
  ∃ arrangement_count : ℕ, arrangement_count = 78 :=
sorry

end students_arrangement_l483_483770


namespace cartesian_equation_of_curve_range_of_PA_PB_l483_483379

-- Definitions and conditions
def α_param := π / 4

def parametric_x (a α : ℝ) := a * Real.cos α
def parametric_y (b α : ℝ) := b * Real.sin α

def cartesian_equation (x y : ℝ) := (x^2 / 2) + y^2

-- Cartesian equation of the curve C
theorem cartesian_equation_of_curve :
  ∀ (a b : ℝ), (parametric_x a α_param = 1) → (parametric_y b α_param = sqrt 2 / 2) 
  → (a = sqrt 2) → (b = 1) → (cartesian_equation 1 (sqrt 2 / 2) = 1) := by
  intros a b Hxa Hby Ha Hb
  rw [parametric_x, parametric_y] at *
  sorry

-- Range of values for |PA| * |PB|
theorem range_of_PA_PB :
  ∀ (a b : ℝ), (cartesian_equation 0 (sqrt 2) = 1) → (∀ θ : ℝ, ∀ t : ℝ, 
  (1 + (Real.sin θ)^2) * t^2 + 4 * sqrt 2 * t * (Real.sin θ) + 2 = 0) 
  → ∃ (θ : ℝ), ((Real.sin θ)^2 > 1 / 3 → (1 <= 2 / (1 + (Real.sin θ)^2)) ∧ (2 / (1 + (Real.sin θ)^2) < 3 / 2)) := by
  intros a b Hc Hl
  -- The proof can be elaborated further based on provided constraints and calculations
  sorry

end cartesian_equation_of_curve_range_of_PA_PB_l483_483379


namespace nine_consecutive_arithmetic_mean_divisible_1111_l483_483868

theorem nine_consecutive_arithmetic_mean_divisible_1111 {n : ℕ} (h1 : ∀ i : ℕ, 0 ≤ i ∧ i < 9 → Nat.Prime (n + i)) :
  ∃ n : ℕ, (∀ k : ℕ, 0 ≤ k ∧ k < 9 → (n + k) ∣ 1111) → (n + 4) = 97 := by
  sorry

end nine_consecutive_arithmetic_mean_divisible_1111_l483_483868


namespace fraction_white_surface_area_l483_483965

-- Definitions based on the conditions
noncomputable def largerCubeEdge : ℕ := 4
noncomputable def totalSmallerCubes : ℕ := 64
noncomputable def whiteCubes : ℕ := 48
noncomputable def blackCubes : ℕ := 16
noncomputable def blackCubesAtCorners : ℕ := 16

-- Theorem statement for the proof problem
theorem fraction_white_surface_area :
  let total_surface_area := 6 * largerCubeEdge^2
  let black_faces_exposed := blackCubesAtCorners * 3
  let white_faces_exposed := total_surface_area - black_faces_exposed
  in (white_faces_exposed : ℚ) / total_surface_area = 3 / 4 :=
by
  -- This is a theorem placeholder to be proven.
  sorry

end fraction_white_surface_area_l483_483965


namespace car_is_late_l483_483158

variables (S D : ℝ)
variables (T_actual T_reduced : ℝ)

-- Conditions
def actual_time_condition : Prop := T_actual = 1
def reduced_speed_condition : Prop := T_reduced = D / ((4/5) * S)

-- Statement to prove
theorem car_is_late : (T_reduced - T_actual) * 60 = 15 :=
by
  assume S D : ℝ,
  assume T_actual = 1,
  assume reduced_speed_condition : T_reduced = D / ((4/5) * S),
  sorry

end car_is_late_l483_483158


namespace range_of_a_l483_483690

-- Define the function f
def f (x : ℝ) : ℝ :=
  if x ≤ 0 then -2017 * x ^ 2 + 2018 * x else real.log (x + 1)

-- Define the statement to prove
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f (1 - 2^x + a * 4^x) ≥ 0) → a ≥ 1 / 4 :=
begin
  sorry
end

end range_of_a_l483_483690


namespace rectangle_proof_l483_483370

theorem rectangle_proof
    (EFGH : Type*)
    [rect : parallelepiped EFGH]
    (M : point EFGH)
    (UV : line EFGH)
    (S : point EFGH)
    (EMH_angle : angle E M H = 90)
    (UV_perpendicular_FG : perpendicular UV FG)
    (FU_eq_UM : distance F U = distance U M)
    (ME : ℝ := 25)
    (EN : ℝ := 20)
    (MN : ℝ := 20) :
    (distance F M = 15) ∧ (distance N V = 5 * sqrt 7) := by
  sorry

end rectangle_proof_l483_483370


namespace problem_1_problem_2_problem_3_l483_483672

noncomputable def geometric_seq (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : Prop :=
a n = a 1 * q ^ (n - 1)

noncomputable def bn (a : ℕ → ℝ) (n : ℕ) : ℝ :=
∑ i in finset.range n, real.log (a (i + 1)) / real.log 2

noncomputable def cn (a : ℕ → ℝ) (b : ℕ → ℝ) (n : ℕ) : ℝ :=
(b n * a n) / n

theorem problem_1 (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n) 
  (h1 : a 1 + 2 * a 2 = 1) 
  (h2 : (a 3) ^ 2 = 4 * a 2 * a 6) :
  ∀ n, a n = 2 ^ -n :=
by sorry

theorem problem_2 (a : ℕ → ℝ) 
  (h_form : ∀ n, a n = 2 ^ -n) :
  ∀ n, ∑ i in finset.range n, 1 / (∑ j in finset.range (i + 1), real.log (a (j + 1)) / real.log 2) 
    = -2 * n / (n + 1) :=
by sorry

theorem problem_3 (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h_bn_form : ∀ n, b n = ∑ i in finset.range n, real.log (a (i + 1)) / real.log 2) 
  (h_an_form : ∀ n, a n = 2 ^ -n) 
  (h_cn_def : ∀ n, cn a b n = (b n * a n) / n) :
  ∀ n, ∑ i in finset.range n, cn a b (i + 1) = -2 + 2 ^ (-n+1) + (n / 2) * 2 ^ (-n) :=
by sorry

end problem_1_problem_2_problem_3_l483_483672


namespace arithmetic_mean_correct_l483_483601

-- Define the expressions
def expr1 (x : ℤ) := x + 12
def expr2 (y : ℤ) := y
def expr3 (x : ℤ) := 3 * x
def expr4 := 18
def expr5 (x : ℤ) := 3 * x + 6

-- The condition as a hypothesis
def condition (x y : ℤ) : Prop := (expr1 x + expr2 y + expr3 x + expr4 + expr5 x) / 5 = 30

-- The theorem to prove
theorem arithmetic_mean_correct : condition 6 72 :=
sorry

end arithmetic_mean_correct_l483_483601


namespace volume_pyramid_SPQR_l483_483778

variables (P Q R S : Type) [metric_space S]
variables (SP SQ SR : ℝ)

-- Define the conditions
axiom h1 : SP = 12
axiom h2 : SQ = 12
axiom h3 : SR = 8

axiom perp_SP_SQ : ∀ (S P Q : S), is_orthogonal S P Q
axiom perp_SQ_SR : ∀ (S Q R : S), is_orthogonal S Q R
axiom perp_SR_SP : ∀ (S R P : S), is_orthogonal S R P

-- Define the theorem to prove
theorem volume_pyramid_SPQR :
  \text{∃} (V : ℝ), V = (1 / 3) * (1 / 2) * SP * SQ * SR := 
  sorry

end volume_pyramid_SPQR_l483_483778


namespace exchange_cards_l483_483953

def redToGold (r : ℕ) : ℕ := r / 5 * 2
def goldToRedAndSilver (g : ℕ) : ℕ × ℕ := (g, g)

theorem exchange_cards : 
  ∀ (initialRed initialGold : ℕ), 
    initialRed = 3 → 
    initialGold = 3 → 
    let newGold := redToGold initialRed in
    let (redFromGold, silverFromGold) := goldToRedAndSilver (initialGold + newGold) in
    redFromGold + redFromGold = 6 →
    silverFromGold + redToGold redFromGold = 7 :=
begin
  intros initialRed initialGold hRed hGold,
  let newGold := redToGold initialRed,
  let (redFromGold, silverFromGold) := goldToRedAndSilver (initialGold + newGold),
  have hConv1 : redFromGold + redFromGold = 6 := sorry,
  have hConv2 : silverFromGold + redToGold redFromGold = 7 := sorry,
  exact hConv2
end

end exchange_cards_l483_483953


namespace businessmen_drink_neither_l483_483575

theorem businessmen_drink_neither : 
  ∀ (total coffee tea both : ℕ), 
    total = 30 → 
    coffee = 15 → 
    tea = 13 → 
    both = 8 → 
    total - (coffee - both + tea - both + both) = 10 := 
by 
  intros total coffee tea both h_total h_coffee h_tea h_both
  sorry

end businessmen_drink_neither_l483_483575


namespace tetrahedron_distance_PE_l483_483261

noncomputable def distance_PE (phi : ℝ) : ℝ :=
  1 / 2 * real.sqrt (5 - 2 * real.sqrt 6 * real.sin phi)

theorem tetrahedron_distance_PE (P A B C D E : Point) (phi : ℝ) (h : ∀(x y : Point), dist x y = 1) :
  -- Check if BD is the height of ABC
  let B := Point.mk 0 0 0 in
  let D := Point.mk 0 real.sqrt(3) / 2 0 in
  -- Check if the distance PE is correctly calculated
  dist PE (Point.mk 1 1 1) = distance_PE phi := sorry

end tetrahedron_distance_PE_l483_483261


namespace min_arithmetic_mean_of_consecutive_naturals_div_by_1111_l483_483865

theorem min_arithmetic_mean_of_consecutive_naturals_div_by_1111 :
  ∀ a : ℕ, (∃ k, (a * (a + 1) * (a + 2) * (a + 3) * (a + 4) * (a + 5) * (a + 6) * (a + 7) * (a + 8)) = 1111 * k) →
  (a + 4 ≥ 97) :=
sorry

end min_arithmetic_mean_of_consecutive_naturals_div_by_1111_l483_483865


namespace num_trees_after_planting_l483_483494

theorem num_trees_after_planting
    (current_trees : ℕ)
    (to_be_planted : ℕ) :
    current_trees = 25 →
    to_be_planted = 73 →
    current_trees + to_be_planted = 98 :=
by
  intros h1 h2
  rw [h1, h2]
  rfl

end num_trees_after_planting_l483_483494


namespace flower_pots_on_path_count_l483_483062

theorem flower_pots_on_path_count (L d : ℕ) (hL : L = 15) (hd : d = 3) : 
  (L / d) + 1 = 6 :=
by
  sorry

end flower_pots_on_path_count_l483_483062


namespace smallest_arithmetic_mean_divisible_by_1111_l483_483828

noncomputable def nine_consecutive_numbers {n : ℕ} : list ℕ :=
  [n, n+1, n+2, n+3, n+4, n+5, n+6, n+7, n+8]

def divisible_by (x y : ℕ) : Prop := ∃ k : ℕ, x = k * y

def arithmetic_mean {l : list ℕ} (h_len : l.length = 9) : ℚ :=
  (l.sum : ℚ) / 9

theorem smallest_arithmetic_mean_divisible_by_1111 :
  ∃ n : ℕ, 
  divisible_by ((nine_consecutive_numbers n).prod) 1111 ∧ 
  arithmetic_mean (by simp [nine_consecutive_numbers_len]) = 97 :=
sorry

end smallest_arithmetic_mean_divisible_by_1111_l483_483828


namespace leaked_before_fixing_l483_483999

def total_leaked_oil := 6206
def leaked_while_fixing := 3731

theorem leaked_before_fixing :
  total_leaked_oil - leaked_while_fixing = 2475 := by
  sorry

end leaked_before_fixing_l483_483999


namespace percentage_books_not_sold_l483_483752

theorem percentage_books_not_sold :
  let initial_stock := 1100
  let sold_monday := 75
  let sold_tuesday := 50
  let sold_wednesday := 64
  let sold_thursday := 78
  let sold_friday := 135
  let total_sold := sold_monday + sold_tuesday + sold_wednesday + sold_thursday + sold_friday
  let books_not_sold := initial_stock - total_sold
  let percentage_not_sold := (books_not_sold / initial_stock.to_float) * 100
  (Float.floor (percentage_not_sold * 100) / 100 = 63.45) :=
by
  let initial_stock := 1100
  let sold_monday := 75
  let sold_tuesday := 50
  let sold_wednesday := 64
  let sold_thursday := 78
  let sold_friday := 135
  let total_sold := sold_monday + sold_tuesday + sold_wednesday + sold_thursday + sold_friday
  have : total_sold = 402 := by norm_num
  let books_not_sold := initial_stock - total_sold
  have : books_not_sold = 698 := by norm_num
  let percentage_not_sold := (books_not_sold.to_float / initial_stock.to_float) * 100
  have : Float.floor (percentage_not_sold * 100) / 100 = 63.45 := sorry
  exact this

end percentage_books_not_sold_l483_483752


namespace complex_number_in_quadrant_I_l483_483684

-- Define the complex number and its transformation to coordinates
def z : ℂ := 1 - I

def transform_to_coordinates (z : ℂ) : ℝ × ℝ := (z.re, -z.im)

-- Define the property of being in Quadrant I
def in_quadrant_I (coords : ℝ × ℝ) : Prop :=
coords.1 > 0 ∧ coords.2 > 0

-- State the main theorem
theorem complex_number_in_quadrant_I : in_quadrant_I (transform_to_coordinates z) :=
by sorry

end complex_number_in_quadrant_I_l483_483684


namespace minimum_trains_needed_l483_483992

theorem minimum_trains_needed (n : ℕ) (h : 50 * n >= 645) : n = 13 :=
by
  sorry

end minimum_trains_needed_l483_483992


namespace expected_waiting_time_correct_l483_483230

noncomputable def combined_average_bites_per_5_minutes := 6
def average_waiting_time_for_first_bite_in_seconds : ℝ := 50

theorem expected_waiting_time_correct :
  (1 / combined_average_bites_per_5_minutes) * 300 = average_waiting_time_for_first_bite_in_seconds :=
by
  sorry

end expected_waiting_time_correct_l483_483230


namespace min_arithmetic_mean_of_consecutive_naturals_div_by_1111_l483_483860

theorem min_arithmetic_mean_of_consecutive_naturals_div_by_1111 :
  ∀ a : ℕ, (∃ k, (a * (a + 1) * (a + 2) * (a + 3) * (a + 4) * (a + 5) * (a + 6) * (a + 7) * (a + 8)) = 1111 * k) →
  (a + 4 ≥ 97) :=
sorry

end min_arithmetic_mean_of_consecutive_naturals_div_by_1111_l483_483860


namespace function_increasing_l483_483128

/-- Let \( f(x) = \sin(x + \varphi) \), where \( \varphi \) is an angle such that \( f \) attains its minimum value 
at \( x = \frac{\pi}{4} \). Prove that the function \( y = f\left(\frac{3\pi}{4} - x\right) \) is increasing in the interval \( \left(\frac{\pi}{2}, \pi\right) \). -/
theorem function_increasing (f : ℝ → ℝ) (ϕ : ℝ) (H : ∀ x, f x = sin (x + ϕ)) (H_min : f (π / 4) = -1) :
  ∀ x ∈ Ioo (π / 2) π, strict_mono (λ x, f (3 * π / 4 - x)) :=
sorry

end function_increasing_l483_483128


namespace trigonometric_simplification_l483_483790

theorem trigonometric_simplification :
  (sin (π / 180 * 15) + sin (π / 180 * 25) + sin (π / 180 * 35) + sin (π / 180 * 45) + sin (π / 180 * 55) + sin (π / 180 * 65) + sin (π / 180 * 75) + sin (π / 180 * 85)) /
  (cos (π / 180 * 10) * cos (π / 180 * 20) * cos (π / 180 * 30)) =
  16 * sin (π / 180 * 50) * cos (π / 180 * 20) / real.sqrt 3 :=
sorry

end trigonometric_simplification_l483_483790


namespace geometric_seq_a5_value_l483_483744

noncomputable theory
open_locale classical

theorem geometric_seq_a5_value
  (a r : ℝ)
  (h1 : a > 0)
  (h2 : r > 0)
  (h3 : (a * r^2) * (a * r^6) = 64) :
  a * r^4 = 8 :=
begin
  sorry
end

end geometric_seq_a5_value_l483_483744


namespace problem1_problem2_problem3_problem4_problem5_problem6_l483_483947

-- (1)
theorem problem1 : sqrt 16 + sqrt ((-3 : ℝ) ^ 2) + real.cbrt 27 = 10 := 
by sorry

-- (2)
theorem problem2 : (-1 : ℝ) ^ 2020 + sqrt 25 - sqrt ((1 - sqrt 2) ^ 2) + real.cbrt (-8) - sqrt ((-3) ^ 2) = - sqrt 2 := 
by sorry

-- (3)
theorem problem3 (x y : ℝ) (h1 : 3 * x + y = 3) (h2 : 3 * x - 2 * y = -6) : x = 0 ∧ y = 3 := 
by sorry

-- (4)
theorem problem4 (x y : ℝ) (h1 : x / 3 + (y + 1) / 2 = 3) (h2 : 2 * x - 3 * y = 9) : x = 6 ∧ y = 1 := 
by sorry

-- (5)
theorem problem5 (x : ℝ) (h : (1 - x) / 2 - 1 ≤ (2 * x - 6) / 3) : x ≥ 9 / 7 := 
by sorry

-- (6)
theorem problem6 (x : ℝ) (h1 : 4 * (x + 1) ≤ 7 * x + 10) (h2 : x - 5 < (x - 8) / 3) : -2 ≤ x ∧ x < 7 / 2 := 
by sorry

end problem1_problem2_problem3_problem4_problem5_problem6_l483_483947


namespace pyramid_volume_is_192_l483_483777

noncomputable def volume_of_pyramid 
  (P Q R S : Type) 
  [InnerProductSpace ℝ P] [InnerProductSpace ℝ Q] 
  [InnerProductSpace ℝ R] [InnerProductSpace ℝ S]
  (SP SQ : ℝ) 
  (SR PQ PR : ℝ) 
  (h_perpendicular_1 : InnerProductSpace.norm (SP) = 12) 
  (h_perpendicular_2 : InnerProductSpace.norm (SQ) = 12)
  (h_perpendicular_3 : InnerProductSpace.norm (SR) = 8)
  (h_v1 : ∀ (u : S) (v : P), orthogonal u v)
  (h_v2 : ∀ (u : S) (v : Q), orthogonal u v)
  (h_v3 : ∀ (u : S) (v : R), orthogonal u v) :
  ℝ := 
begin
  -- The volume of the pyramid SPQR
  let base_area := (1 / 2) * PQ * PR,
  let height := SR,
  let volume := (1 / 3) * base_area * height,
  volume
end

theorem pyramid_volume_is_192 : 
  volume_of_pyramid P Q R S 
                    (12) (12) (8) (12) (12) 
                    sorry sorry sorry sorry sorry sorry = 192 :=
sorry

end pyramid_volume_is_192_l483_483777


namespace gcd_gx_x_l483_483666

-- Condition: x is a multiple of 7263
def isMultipleOf7263 (x : ℕ) : Prop := ∃ k : ℕ, x = 7263 * k

-- Definition of g(x)
def g (x : ℕ) : ℕ := (3*x + 4) * (9*x + 5) * (17*x + 11) * (x + 17)

-- Statement to be proven
theorem gcd_gx_x (x : ℕ) (h : isMultipleOf7263 x) : Nat.gcd (g x) x = 1 := by
  sorry

end gcd_gx_x_l483_483666


namespace concert_attendance_difference_l483_483433

/-- Define the number of people attending the first concert. -/
def first_concert_attendance : ℕ := 65899

/-- Define the number of people attending the second concert. -/
def second_concert_attendance : ℕ := 66018

/-- The proof statement that the difference in attendance between the second and first concert is 119. -/
theorem concert_attendance_difference :
  (second_concert_attendance - first_concert_attendance = 119) := by
  sorry

end concert_attendance_difference_l483_483433


namespace minimum_value_of_f_l483_483093

-- Function definition
def f (x : ℝ) : ℝ := x^3 - 3 * x

-- Statement of the problem: that the minimum value of f(x) is -2
theorem minimum_value_of_f : ∃ x : ℝ, is_minimum_value f x (-2) := sorry

-- Definition of the is_minimum_value predicate (if not already defined in the library)
def is_minimum_value (f : ℝ → ℝ) (x min_val : ℝ) :=
  (∀ y : ℝ, f x ≤ f y) ∧ f x = min_val

end minimum_value_of_f_l483_483093


namespace triangle_AC_AA1_CC1_area_triangle_AA1_BB1_CC1_area_l483_483018

noncomputable def area_triangle_AC_AA1_CC1 : ℝ :=
  let ab := 4 in
  let bc := 4 in
  let ac := 2 in
  let aa1 := ab * ac / (ab + ac) in
  let cc1 := (sqrt (ab^2 - (ac / 2)^2)) / 2 in
  let ac1 := sqrt ((ac^2) - ((sqrt ((ab^2) - (ac/2)^2)) / 2)^2) in
  let area_acc1 := (1 / 2) * ac1 * cc1 in
  let s_amc := (ab / (ab + ac)) * area_acc1 in
  s_amc

noncomputable def area_triangle_AA1_BB1_CC1 : ℝ :=
  let ab := 4 in
  let bc := 4 in
  let ac := 2 in
  let bb1 := sqrt (ab^2 - (ac / 2)^2) in
  let cc1 := (sqrt (ab^2 - (ac / 2)^2)) / 2 in
  let ak := (2/3) * ac / 2 in
  let am := 1 / 2 * ac in
  let area_akb1 := (1 / 2) * bb1 * (ac / 2) in
  let area_kml := (2 / 3) * (1 / 2) * area_akb1 in
  area_kml

-- Proofs
theorem triangle_AC_AA1_CC1_area : 
  area_triangle_AC_AA1_CC1 = sqrt 15 / 10 := sorry

theorem triangle_AA1_BB1_CC1_area :
  area_triangle_AA1_BB1_CC1 = sqrt 15 / 30 := sorry

end triangle_AC_AA1_CC1_area_triangle_AA1_BB1_CC1_area_l483_483018


namespace average_waiting_time_for_first_bite_l483_483222

/-- 
Let S be a period of 5 minutes (300 seconds).
- We have an average of 5 bites in 300 seconds on the first fishing rod.
- We have an average of 1 bite in 300 seconds on the second fishing rod.
- The total average number of bites on both rods during this period is 6 bites.
The bites occur independently and follow a Poisson process.

We aim to prove that the waiting time for the first bite, given these conditions, is 
expected to be 50 seconds.
-/
theorem average_waiting_time_for_first_bite :
  let S := 300 -- 5 minutes in seconds
  -- The average number of bites on the first and second rod in period S.
  let avg_bites1 := 5
  let avg_bites2 := 1
  let total_avg_bites := avg_bites1 + avg_bites2
  -- The rate parameter λ for the Poisson process is total_avg_bites / S.
  let λ := total_avg_bites / S
  -- The average waiting time for the first bite.
  1 / λ = 50 :=
by
  let S := 300
  let avg_bites1 := 5
  let avg_bites2 := 1
  let total_avg_bites := avg_bites1 + avg_bites2
  let λ := total_avg_bites / S
  -- convert λ to seconds to ensure unit consistency
  have hλ: λ = 6 / 300 := rfl
  -- The expected waiting time for the first bite is 1 / λ
  have h_waiting_time: 1 / λ = 300 / 6 := by
    rw [hλ, one_div, div_div_eq_mul]
    norm_num
  exact h_waiting_time

end average_waiting_time_for_first_bite_l483_483222


namespace work_completion_time_l483_483519

noncomputable theory

-- Definitions for the conditions
def p_work_days : ℕ := 20
def q_work_days : ℕ := 12
def p_alone_days : ℕ := 4

-- The proved statement
theorem work_completion_time : p_work_days = 20 ∧ q_work_days = 12 ∧ p_alone_days = 4 → 10 = 10 :=
by
  intros h,
  cases h with hp hq,
  exact sorry

end work_completion_time_l483_483519


namespace identify_perfect_square_is_689_l483_483117

-- Definitions of the conditions
def natural_numbers (n : ℕ) : Prop := True -- All natural numbers are accepted
def digits_in_result (n m : ℕ) (d : ℕ) : Prop := (n * m) % 1000 = d

-- Theorem to be proved
theorem identify_perfect_square_is_689 (n : ℕ) :
  (∀ m, natural_numbers m → digits_in_result m m 689 ∨ digits_in_result m m 759) →
  ∃ m, natural_numbers m ∧ digits_in_result m m 689 :=
sorry

end identify_perfect_square_is_689_l483_483117


namespace infinite_geometric_sum_example_l483_483311

noncomputable def infinite_geometric_sum (a₁ q : ℝ) : ℝ :=
a₁ / (1 - q)

theorem infinite_geometric_sum_example :
  infinite_geometric_sum 18 (-1/2) = 12 := by
  sorry

end infinite_geometric_sum_example_l483_483311


namespace parallelepiped_has_12_edges_l483_483929

-- Definition of a parallelepiped
structure Parallelepiped where
  vertices : ℕ -- number of vertices
  edges : ℕ    -- number of edges
  faces : ℕ    -- number of faces
  face_type : ℕ → String -- each face type description

-- Parallelepiped example
def example_parallelepiped : Parallelepiped :=
  { vertices := 8,
    edges := 12,
    faces := 6,
    face_type := λ n, "parallelogram" }

-- Theorem: A parallelepiped has 12 edges
theorem parallelepiped_has_12_edges (p : Parallelepiped) : p.edges = 12 := by
  sorry

end parallelepiped_has_12_edges_l483_483929


namespace sequence_eventual_stationary_l483_483049

theorem sequence_eventual_stationary
  (N : ℕ) (hN : 0 < N)
  (a : ℕ → ℕ)
  (h_initial : ∀ i, 1 ≤ i → i ≤ N → ¬ (2^(N+1)) ∣ a i)
  (h_recurrence : ∀ n, n ≥ N + 1 → ∃ k, k ∈ finset.range n ∧ a (n+1) = 2 * a k ∧ (∀ j, j ∈ finset.range n → (a k % 2^n ≤ a j % 2^n))) :
  ∃ M, ∀ n, n ≥ M → a n = a M :=
begin
  sorry
end

end sequence_eventual_stationary_l483_483049


namespace determine_f_36_l483_483759

def strictly_increasing (f : ℕ → ℕ) : Prop :=
  ∀ n, f (n + 1) > f n

def multiplicative (f : ℕ → ℕ) : Prop :=
  ∀ m n, f (m * n) = f m * f n

def special_condition (f : ℕ → ℕ) : Prop :=
  ∀ m n, m > n → m^m = n^n → f m = n

theorem determine_f_36 (f : ℕ → ℕ)
  (H1: strictly_increasing f)
  (H2: multiplicative f)
  (H3: special_condition f)
  : f 36 = 1296 := 
sorry

end determine_f_36_l483_483759


namespace find_values_l483_483250

theorem find_values (x y z : ℝ) :
  (x + y + z = 1) →
  (x^2 * y + y^2 * z + z^2 * x = x * y^2 + y * z^2 + z * x^2) →
  (x^3 + y^2 + z = y^3 + z^2 + x) →
  ( (x = 1/3 ∧ y = 1/3 ∧ z = 1/3) ∨ 
    (x = 0 ∧ y = 0 ∧ z = 1) ∨
    (x = 2/3 ∧ y = -1/3 ∧ z = 2/3) ∨
    (x = 0 ∧ y = 1 ∧ z = 0) ∨
    (x = 1 ∧ y = 0 ∧ z = 0) ∨
    (x = -1 ∧ y = 1 ∧ z = 1) ) := 
sorry

end find_values_l483_483250


namespace digits_difference_divisible_by_three_l483_483084

theorem digits_difference_divisible_by_three (A B : ℕ) (h1 : A ≠ B) (h2 : A < 10) (h3 : B < 10) : (10 * A + B) - (10 * B + A) % 3 = 0 :=
begin
  sorry
end

end digits_difference_divisible_by_three_l483_483084


namespace sam_runs_12_l483_483707

variable (S : ℕ)

axiom harvey_runs_more : ∀ S, (∃ H, H = S + 8)
axiom total_miles : ∀ S H, S + H = 32

theorem sam_runs_12 : ∃ S, (harvey_runs_more S.1) ∧ (total_miles S.1 (S.1 + 8)) → S = 12 :=
by
  sorry

end sam_runs_12_l483_483707


namespace matrix_eigenvalue_neg7_l483_483271

theorem matrix_eigenvalue_neg7 (M : Matrix (Fin 2) (Fin 2) ℝ) :
  (∀ (v : Fin 2 → ℝ), M.mulVec v = -7 • v) →
  M = !![-7, 0; 0, -7] :=
by
  intro h
  -- proof goes here
  sorry

end matrix_eigenvalue_neg7_l483_483271


namespace area_triangle_N1N2N3_l483_483932

theorem area_triangle_N1N2N3 (K : ℝ)
  (h1 : ∀ A B C, ∃ D, ∃ E, ∃ F, ∃ N1, ∃ N2, ∃ N3,
    (CD_length : (D.1 - C.1)^2 + (D.2 - C.2)^2 = ((A.1 - C.1)^2 + (A.2 - C.2)^2) / 4) ∧
    (AE_length : (E.1 - A.1)^2 + (E.2 - A.2)^2 = ((B.1 - A.1)^2 + (B.2 - A.2)^2) / 4) ∧
    (BF_length : (F.1 - B.1)^2 + (F.2 - B.2)^2 = ((C.1 - B.1)^2 + (C.2 - B.2)^2) / 4) ∧
    N1 = intersect (A, D) (C, F) ∧
    N2 = intersect (B, E) (A, D) ∧
    N3 = intersect (B, E) (C, F)) :
  area (N1, N2, N3) = (1/4) * K :=
sorry

end area_triangle_N1N2N3_l483_483932


namespace probability_not_red_nor_white_l483_483367

noncomputable def num_red : ℕ := 2
noncomputable def num_white : ℕ := 3
noncomputable def num_yellow : ℕ := 5

noncomputable def total_balls : ℕ := num_red + num_white + num_yellow

theorem probability_not_red_nor_white : (num_yellow : ℝ) / (total_balls : ℝ) = 0.5 := by
  have h_total : total_balls = 10 := by
    unfold total_balls num_red num_white num_yellow; rfl
  have h_yellow : (num_yellow : ℝ) = 5 := by
    unfold num_yellow; exact rfl
  rw [h_total, h_yellow]
  norm_num
  sorry

end probability_not_red_nor_white_l483_483367


namespace molecular_weight_Dinitrogen_pentoxide_l483_483579

theorem molecular_weight_Dinitrogen_pentoxide :
  let atomic_weight_N := 14.01
  let atomic_weight_O := 16.00
  let molecular_formula := (2 * atomic_weight_N) + (5 * atomic_weight_O)
  molecular_formula = 108.02 :=
by
  sorry

end molecular_weight_Dinitrogen_pentoxide_l483_483579


namespace integral_abs_x_minus_1_l483_483262

open Real

noncomputable def evaluate_integral : ℝ :=
  ∫ x in -2..2, |x - 1|

theorem integral_abs_x_minus_1 : evaluate_integral = 5 :=
  sorry

end integral_abs_x_minus_1_l483_483262


namespace positive_diff_of_supplementary_angles_l483_483091

theorem positive_diff_of_supplementary_angles (x : ℝ) (h : 5 * x + 3 * x = 180) : 
  abs ((5 * x - 3 * x)) = 45 := by
  sorry

end positive_diff_of_supplementary_angles_l483_483091


namespace assemble_6x6_square_from_8_pieces_l483_483184

theorem assemble_6x6_square_from_8_pieces 
(piece_shapes : List (Set (ℕ × ℕ)))
(piece_8_cut : Set (ℕ × ℕ))
(h_piece_shapes : piece_shapes.length = 7)
(h_piece_8_is_part : ∃ (part1 part2 part3 : Set (ℕ × ℕ)),
  part1 ∪ part2 ∪ part3 = piece_8_cut ∧
  part1 ∈ piece_shapes ∧ 
  part2 ∈ piece_shapes ∧ 
  part3 ∈ piece_shapes)
: ∃ (final_square : Set (ℕ × ℕ)),
  (final_square.to_finset.card = 36) ∧ -- 6x6 square
  (final_square = piece_shapes.foldr (∪) ∅ ∪ piece_8_cut) :=
sorry

end assemble_6x6_square_from_8_pieces_l483_483184


namespace triangle_incircle_ratio_l483_483545

theorem triangle_incircle_ratio
  (a b c : ℝ) (ha : a = 15) (hb : b = 12) (hc : c = 9)
  (r s : ℝ) (hr : r + s = c) (r_lt_s : r < s) :
  r / s = 1 / 2 :=
sorry

end triangle_incircle_ratio_l483_483545


namespace count_very_interesting_quadruples_l483_483596

-- Definitions of the conditions as per the problem statement
def isVeryInteresting (a b c d : ℕ) : Prop :=
  1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 15 ∧ a + d > 2 * (b + c)

-- The theorem stating the number of very interesting quadruples.
theorem count_very_interesting_quadruples : (finset.univ.filter (λ (t : ℕ × ℕ × ℕ × ℕ), isVeryInteresting t.1 t.2.1 t.2.2.1 t.2.2.2)).card = 682 := 
  sorry

end count_very_interesting_quadruples_l483_483596


namespace rationalize_denominator_l483_483446

noncomputable def cube_root (x : ℝ) := x^(1/3)

theorem rationalize_denominator (a b : ℝ) (h : cube_root 27 = 3) : 
  1 / (cube_root 3 + cube_root 27) = (3 - cube_root 3) / (9 - 3 * cube_root 3) :=
by
  sorry

end rationalize_denominator_l483_483446


namespace minimum_c_is_1006_l483_483781

noncomputable def minimum_c (a b c : ℕ) (h1 : a < b) (h2 : b < c) :=
  2*x + y = 2010 ∧ y = |x - a| + |x - b| + |x - c| 

theorem minimum_c_is_1006 (a b c : ℕ) (h1 : a < b) (h2 : b < c) 
  (h3 : ∃! x, 2*x + (|x - a| + |x - b| + |x - c|) = 2010) :
  c = 1006 :=
sorry

end minimum_c_is_1006_l483_483781


namespace second_eq_value_l483_483351

variable (x y z w : ℝ)

theorem second_eq_value (h1 : 4 * x * z + y * w = 3) (h2 : (2 * x + y) * (2 * z + w) = 15) : 
  x * w + y * z = 6 :=
by
  sorry

end second_eq_value_l483_483351


namespace area_of_triangle_OAB_l483_483425

open Complex

noncomputable theory

def area_triangle_OAB (z1 z2 : ℂ) : ℝ :=
  (1 / 2) * (Complex.abs z1 * Complex.abs (z2 - z1))

theorem area_of_triangle_OAB (z1 z2 : ℂ) (h1 : Complex.abs z1 = 4)
  (h2 : 4 * z1^2 - 2 * z1 * z2 + z2^2 = 0) : area_triangle_OAB z1 z2 = 8 * Real.sqrt 3 := 
sorry

end area_of_triangle_OAB_l483_483425


namespace find_a_l483_483037

theorem find_a (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a > b) (h4 : a * b - a - b = 4) : a = 6 :=
sorry

end find_a_l483_483037


namespace vicente_rice_purchase_l483_483897

theorem vicente_rice_purchase :
  ∃ R : Nat, 2 * R + 3 * 5 = 25 ∧ R = 5 :=
by
  use 5
  split
  · calc
      2 * 5 + 3 * 5 = 10 + 15 : by rw [mul_add, mul_add, mul_comm, mul_comm]
      ...          = 25 : by rw [add_comm]
  · rfl

end vicente_rice_purchase_l483_483897


namespace performanceIncrease_l483_483376

def numberOfWinsCurrentSeason := 25
def numberOfLossesCurrentSeason := 8
def totalNumberOfGamesCurrentSeason := numberOfWinsCurrentSeason + numberOfLossesCurrentSeason

def numberOfWinsPreviousSeason := 18
def numberOfLossesPreviousSeason := 12
def totalNumberOfGamesPreviousSeason := numberOfWinsPreviousSeason + numberOfLossesPreviousSeason

def winningPercentage (wins : ℕ) (totalGames : ℕ) : ℝ :=
  ((wins : ℝ) / (totalGames : ℝ)) * 100

def currentSeasonWinningPercentage := winningPercentage numberOfWinsCurrentSeason totalNumberOfGamesCurrentSeason
def previousSeasonWinningPercentage := winningPercentage numberOfWinsPreviousSeason totalNumberOfGamesPreviousSeason

def performanceImprovement :=
  currentSeasonWinningPercentage - previousSeasonWinningPercentage

theorem performanceIncrease : performanceImprovement ≈ 15.76 :=
by sorry

end performanceIncrease_l483_483376


namespace value_euro_to_dollar_l483_483077

theorem value_euro_to_dollar (conversion_rate : ℝ) (diana_dollars : ℝ) (etienne_euros : ℝ) : 
  let etienne_dollars := etienne_euros * conversion_rate in
  (diana_dollars - etienne_dollars) / diana_dollars = 0.1 :=
  by
  let etienne_dollars := etienne_euros * conversion_rate
  sorry

end value_euro_to_dollar_l483_483077


namespace sum_squares_of_real_solutions_problem_sum_of_squares_l483_483278

theorem sum_squares_of_real_solutions (x : ℝ) (hx : x ^ 64 = 2 ^ 64) :
  x = 2 ∨ x = -2 :=
begin
  by_cases x = 2,
  { left,
    assumption, },
  { right,
    apply (neg_eq_iff_neg_eq).1,
    rw [← pow_eq_one_iff_mod_two_eq_zero] at hx,
    exact hx.symm, }
end

theorem problem_sum_of_squares :
  ∑ x in {x | x ^ 64 = 2 ^ 64}.to_finset, x^2 = 8 :=
begin
  apply finset.sum_const_nat,
  sorry,
end

end sum_squares_of_real_solutions_problem_sum_of_squares_l483_483278


namespace triangle_qr_length_l483_483384

-- Define the lengths of sides
def AB : ℝ := 13
def BC : ℝ := 12
def AC : ℝ := 5

-- Define the tangent circle property and intersection points
theorem triangle_qr_length (h1: AB = 13) (h2: BC = 12) (h3: AC = 5) 
  (h4: ∀ (P: ℝ), ∃ (A B C Q R: ℝ), (circle P A B) ∧ (tangent P BC) 
  ∧ (Q ≠ A) ∧ (R ≠ A) ∧ (intersects Q P AB) ∧ (intersects R P AC)) :
  QR = 26 := 
sorry

end triangle_qr_length_l483_483384


namespace probability_all_calls_same_probability_two_calls_for_A_l483_483112

theorem probability_all_calls_same (pA pB pC : ℚ) (hA : pA = 1/6) (hB : pB = 1/3) (hC : pC = 1/2) :
  (pA^3 + pB^3 + pC^3) = 1/6 :=
by
  sorry

theorem probability_two_calls_for_A (pA : ℚ) (hA : pA = 1/6) :
  (3 * (pA^2) * (5/6)) = 5/72 :=
by
  sorry

end probability_all_calls_same_probability_two_calls_for_A_l483_483112


namespace segment_area_l483_483204

theorem segment_area (d : ℝ) (θ : ℝ) (r := d / 2)
  (A_triangle := (1 / 2) * r^2 * Real.sin (θ * Real.pi / 180))
  (A_sector := (θ / 360) * Real.pi * r^2) :
  θ = 60 →
  d = 10 →
  A_sector - A_triangle = (100 * Real.pi - 75 * Real.sqrt 3) / 24 :=
by
  sorry

end segment_area_l483_483204


namespace can_guaranteedly_find_liar_l483_483110

noncomputable def canFindLiar : Prop :=
  ∃ (persons : Fin 10 → Nat), 
    (∀ i j, i ≠ j → persons i ≠ persons j) ∧ 
    (∃ liar_idx, ∀ M, liar_idx = i → (∃ (ask : (Fin 10 → Nat) → Fin 10 → Prop),
      (∀ i, 0 ≤ i ∧ i < 10 → ask persons i = M ↔ persons i ≠ M) 
      ∧ (∀ i j, ask persons i ≠ ask persons j)
      ∧ (ask persons liar_idx = M → persons liar_idx ≠ M))) 
    ∧ (∃ questions : Set (Fin 10 × Nat), questions.size ≤ 17)

theorem can_guaranteedly_find_liar : canFindLiar := by 
  sorry

end can_guaranteedly_find_liar_l483_483110


namespace duration_in_years_l483_483183

def simple_interest (P R T : ℝ) : ℝ := (P * R * T) / 100

variables (P R SI : ℝ)
variables (hP : P = 16065)
variables (hR : R = 5)
variables (hSI : SI = 4016.25)

theorem duration_in_years : ∃ T : ℝ, simple_interest P R T = SI ∧ T = 5 := by
  use 5
  rw [simple_interest, hP, hR, hSI]
  norm_num
  sorry

end duration_in_years_l483_483183


namespace rationalize_denominator_l483_483448

noncomputable def cube_root (x : ℝ) := x^(1/3)

theorem rationalize_denominator (a b : ℝ) (h : cube_root 27 = 3) : 
  1 / (cube_root 3 + cube_root 27) = (3 - cube_root 3) / (9 - 3 * cube_root 3) :=
by
  sorry

end rationalize_denominator_l483_483448


namespace sum_lambda_eq_one_l483_483408

theorem sum_lambda_eq_one
  (A B C D : Type)
  [AddCommGroup A] [VectorSpace ℝ A]
  (BC : Submodule ℝ A)
  (hD : D ∈ line_segment ℝ B C)
  (hBD : ∀ x ∈ BC, x = B - C → D = B + 1/3 • x) :
  ∃ λ1 λ2 : ℝ, 
  (segment ℝ A B D = λ1 • B + λ2 • C) ∧ (λ1 + λ2 = 1) := 
by
  sorry

end sum_lambda_eq_one_l483_483408


namespace radius_eq_l483_483286

-- Define the given conditions in the problem
variables (A M B C O : point)
variables (R : ℝ) -- Radius of the circle

-- Some basic geometric constructions and distances
variables (AM AC BC : ℝ)
variables (tangent secant : set point) -- The tangent and the secant lines
variables (dist : point → point → ℝ) -- The distance function between points
variables (circle : set point) (tangent_circle : subset tangent circle) (secant_circle : subset secant circle)

-- Specific given distances
axiom AM_eq : dist A M = 16
axiom AC_eq : dist A C = 32
axiom center_to_secant : dist O (Some_point_on_secant AC) = 5
axiom tangent_point_on_circle : M ∈ circle
axiom secant_points_on_circle : B ∈ circle ∧ C ∈ circle
axiom secant_length : dist B C = 24 -- derived from AC_eq and AB

-- The conclusion we need to reach
theorem radius_eq : R = 13 :=
sorry

end radius_eq_l483_483286


namespace smallest_N_for_301_l483_483121

def concatenation_pattern (N : ℕ) : String :=
  String.concat (List.map (λ k, toString k ++ toString (N - k + 1)) (List.range (N // 2 + 1)))

theorem smallest_N_for_301 :
  ∃ N : ℕ, (∀ n : ℕ, n < N → ¬"301".isIn (concatenation_pattern n)) ∧ "301".isIn (concatenation_pattern N) ∧ N = 38 :=
by
  sorry

end smallest_N_for_301_l483_483121


namespace smallest_arithmetic_mean_divisible_product_l483_483839

theorem smallest_arithmetic_mean_divisible_product :
  ∃ (n : ℕ), (∀ i : ℕ, i < 9 → Nat.gcd ((n + i) * 1111) 1111 = 1111) ∧ (n + 4 = 97) := 
  sorry

end smallest_arithmetic_mean_divisible_product_l483_483839


namespace prism_is_right_square_prism_l483_483799

variable (base : Set Point) (isSquare : base.is_square) (vertex : Point) (edges : Set LineSegment)
variable (isRhombus : base.is_rhombus) (mutuallyPerpendicular : ∀ e1 e2 ∈ edges, e1 ≠ e2 → e1.perpendicular_to e2)

theorem prism_is_right_square_prism (h_base_rhombus : isRhombus ∧ (∃ v ∈ base, ∀ e1 e2 e3 ∈ edges, mutuallyPerpendicular e1 e2 → mutuallyPerpendicular e2 e3 → mutuallyPerpendicular e3 e1)) 
  : isSquare ∧ (∀ lateral_face ∈ prism.lateral_faces, lateral_face.is_rectangle) :=
sorry

end prism_is_right_square_prism_l483_483799


namespace product_of_odd_integers_excluding_5_l483_483907

theorem product_of_odd_integers_excluding_5 :
  ∏ (i : ℕ) in (finset.filter (λ n, n % 2 = 1 ∧ n < 50 ∧ n % 5 ≠ 0) (finset.range 50)),
    i = 584358414459472720896 :=
by
sorry

end product_of_odd_integers_excluding_5_l483_483907


namespace f_not_even_or_odd_f_mono_increasing_in_l483_483330

section 

variable {a : ℝ} {x : ℝ} (hx : x ≠ 0)
def f (x : ℝ) : ℝ := x^2 + a / x

theorem f_not_even_or_odd (h : a ≠ 0) :
  ¬(∀ x : ℝ, f x = f (-x)) ∧ ¬(∀ x : ℝ, f x = -f (-x)) := sorry

theorem f_mono_increasing_in [2, +∞) :
  ∀ (x1 x2 : ℝ), (2 ≤ x1) → (2 ≤ x2) → (x1 < x2) → (f x1 < f x2) :=
by
  have a : ℝ := 1
  let f (x : ℝ) : ℝ := x^2 + a / x
  intros x1 x2 hx1 hx2 hlt
  sorry

end

end f_not_even_or_odd_f_mono_increasing_in_l483_483330


namespace red_points_count_l483_483418

open Set

-- Define the conditions
variables (k n : ℕ) (L : Set (Set ℝ × ℝ)) (I : Set (ℝ × ℝ)) (O : ℝ × ℝ)

-- Assume the conditions
-- Assume k and n are integers with 0 ≤ k ≤ n-2
axiom hk : 0 ≤ k ∧ k ≤ n-2

-- Assume L is a set of n lines such that no two are parallel and no three have a common point
axiom hL : ∀ (ℓ1 ℓ2 : Set (ℝ × ℝ)), ℓ1 ∈ L → ℓ2 ∈ L → ℓ1 ≠ ℓ2 → ¬ ∃ p q r, (p ∈ ℓ1 ∧ p ∈ ℓ2) ∧ (q ∈ ℓ1 ∧ q ∈ ℓ2) ∧ (r ∈ ℓ1 ∧ r ∈ ℓ2)

-- Assume I is the set of intersection points of lines in L
def I_def : Set (ℝ × ℝ) := { P | ∃ ℓ1 ℓ2, ℓ1 ∈ L ∧ ℓ2 ∈ L ∧ ℓ1 ≠ ℓ2 ∧ P ∈ ℓ1 ∧ P ∈ ℓ2 }

-- Assume O is a point not on any line in L
axiom hO : ∀ ℓ ∈ L, O ∉ ℓ

-- Define a point X in I is red if the open line segment OX intersects at most k lines in L
def is_red (X : ℝ × ℝ) : Prop := (λ X : ℝ × ℝ, ∃ m ≤ k, ∃ A, A ⊆ L ∧ m = size A) X

-- Prove that I contains at least (1/2)(k+1)(k+2) red points
theorem red_points_count : ∃ reds, reds ⊆ I_def L ∧ ∀ x ∈ reds, is_red k L O x ∧ size reds ≥ (1/2) * (k + 1) * (k + 2) :=
sorry

end red_points_count_l483_483418


namespace any_positive_integer_can_be_expressed_l483_483443

theorem any_positive_integer_can_be_expressed 
  (N : ℕ) (hN : 0 < N) : 
  ∃ (p q u v : ℤ), N = p * q + u * v ∧ (u - v = 2 * (p - q)) := 
sorry

end any_positive_integer_can_be_expressed_l483_483443


namespace A_share_correct_l483_483133

noncomputable def A_share_investment_months : ℤ := (20000 * 5) + (15000 * 7)
noncomputable def B_share_investment_months : ℤ := (20000 * 5) + (16000 * 7)
noncomputable def C_share_investment_months : ℤ := (20000 * 5) + (26000 * 7)
noncomputable def total_investment_months : ℤ := A_share_investment_months + B_share_investment_months + C_share_investment_months
noncomputable def total_profit : ℤ := 69900
noncomputable def A_share_ratio : ℚ := A_share_investment_months / total_investment_months
noncomputable def A_share_of_profit : ℤ := (A_share_ratio * total_profit).toInt

theorem A_share_correct : A_share_of_profit = 20500 := 
  by sorry

end A_share_correct_l483_483133


namespace minimum_abs_omega_l483_483697

def f (ω : ℝ) (x : ℝ) := Math.sin (ω * x + (Real.pi / 3)) + 2

theorem minimum_abs_omega (ω : ℝ) (x : ℝ) (n : ℤ) (h : ∀ x, f ω (x + (4 * Real.pi / 3)) = f ω x) :
  |ω| = 3 / 2 :=
by
  -- translation of conditions
  have period_condition : (4 * Real.pi / 3) = n * ((2 * Real.pi) / ω) := sorry
  -- conclusion for minimum ω
  have ω_condition : ω = n * 3 / 2 := sorry
  have ω_gt_zero : ω > 0 := sorry
  sorry

end minimum_abs_omega_l483_483697


namespace shoes_cost_percentage_increase_l483_483135

theorem shoes_cost_percentage_increase :
  let repair_cost := 14.50
  let repair_duration := 1.0
  let new_cost := 32.00
  let new_duration := 2.0
  let repair_avg_cost := repair_cost / repair_duration
  let new_avg_cost := new_cost / new_duration
  let cost_difference := new_avg_cost - repair_avg_cost
  let percentage_increase := (cost_difference / repair_avg_cost) * 100
  in percentage_increase = 10.34 :=
by
  sorry

end shoes_cost_percentage_increase_l483_483135


namespace cot_tan_simplified_l483_483464

theorem cot_tan_simplified :
  (Real.cot (Real.pi / 9) + Real.tan (Real.pi / 18) = Real.csc (Real.pi / 9)) :=
by
  sorry

end cot_tan_simplified_l483_483464


namespace find_a_l483_483326

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2

theorem find_a (a : ℝ) (h : deriv (f a) (-1) = 4) : a = 10 / 3 :=
by {
  sorry
}

end find_a_l483_483326


namespace smallest_arithmetic_mean_l483_483878

noncomputable def S (n : ℕ) := (List.range' n 9).map Nat.ofNat

theorem smallest_arithmetic_mean (n : ℕ) (h1 : 93 ≤ n) (h2 : ∃ k ∈ S n, 11 ∣ k) (h3 : ∃ k ∈ S n, 101 ∣ k) : 
  (n + 4 = 97) := by
  sorry

end smallest_arithmetic_mean_l483_483878


namespace initial_dogwood_trees_in_park_l483_483109

def num_added_trees := 5 + 4
def final_num_trees := 16
def initial_num_trees (x : ℕ) := x

theorem initial_dogwood_trees_in_park (x : ℕ) 
  (h1 : num_added_trees = 9) 
  (h2 : final_num_trees = 16) : 
  initial_num_trees x + num_added_trees = final_num_trees → 
  x = 7 := 
by 
  intro h3
  rw [initial_num_trees, num_added_trees] at h3
  linarith

end initial_dogwood_trees_in_park_l483_483109


namespace evaluate_expression_at_2_l483_483010

noncomputable def replace_and_evaluate (x : ℝ) : ℝ :=
  (3 * x - 2) / (-x + 6)

theorem evaluate_expression_at_2 :
  replace_and_evaluate 2 = -2 :=
by
  -- evaluation and computation would go here, skipped with sorry
  sorry

end evaluate_expression_at_2_l483_483010


namespace p_iff_q_l483_483729

variable {A B C a b c : ℝ}

def p : Prop := (a / Real.sin B = b / Real.sin C) ∧ (b / Real.sin C = c / Real.sin A)
def q : Prop := (A = 60) ∧ (B = 60) ∧ (C = 60)

theorem p_iff_q : p ↔ q := by
  sorry

end p_iff_q_l483_483729


namespace sum_of_first_100_terms_is_2653_l483_483649

def sequence (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n % 2 = 1 then 3
  else 1 + (n / 2 + 1) * 2

def sum_sequence (n : ℕ) : ℕ :=
  (Fin.range n).sum (λ i => sequence i)

theorem sum_of_first_100_terms_is_2653 : 
  (sum_sequence 100) = 2653 := sorry

end sum_of_first_100_terms_is_2653_l483_483649


namespace largest_inscribed_triangle_area_l483_483212

theorem largest_inscribed_triangle_area 
  (radius : ℝ) 
  (diameter : ℝ)
  (base : ℝ)
  (height : ℝ) 
  (area : ℝ)
  (h1 : radius = 10)
  (h2 : diameter = 2 * radius)
  (h3 : base = diameter)
  (h4 : height = radius) 
  (h5 : area = (1/2) * base * height) : 
  area  = 100 :=
by 
  have h_area := (1/2) * 20 * 10
  sorry

end largest_inscribed_triangle_area_l483_483212


namespace line_through_points_l483_483618

-- Define the points given in the conditions
def point1 : ℝ × ℝ := (-3, 7)
def point2 : ℝ × ℝ := (4, -2)

-- Define the slope m that we need to prove
def slope : ℝ := -9 / 7

-- Define the y-intercept b that we need to prove
def y_intercept : ℝ := 22 / 7

-- Define the slope-intercept equation based on m and b
def line_equation (x : ℝ) : ℝ := slope * x + y_intercept

-- The theorem to prove
theorem line_through_points :
  ∃ m b, (∀ (x : ℝ), y = m * x + b) ∧
    m = slope ∧ b = y_intercept :=
by
  use slope
  use y_intercept
  sorry

end line_through_points_l483_483618


namespace number_of_factors_of_prime_factors_8_sq_5_cubed_7_sq_l483_483339

theorem number_of_factors_of_prime_factors_8_sq_5_cubed_7_sq : 
  let n := (8 : Nat)^2 * 5^3 * 7^2
  in (∃ factors_count : Nat, factors_count = 84) :=
by
  let n := (8 : Nat)^2 * 5^3 * 7^2
  have prime_factorized_n : n = 2^6 * 5^3 * 7^2 := by
    -- Prime factorization logic (not detailed here)
    sorry

  -- Counting factors logic
  -- Total number of factors = (6 + 1) * (3 + 1) * (2 + 1) = 84
  let factors_count := 7 * 4 * 3
  use factors_count
  exact Eq.refl factors_count

end number_of_factors_of_prime_factors_8_sq_5_cubed_7_sq_l483_483339


namespace circles_intersect_l483_483253

-- Define the first circle with center and radius
def circle1_center := (-2 : ℝ, 0 : ℝ)
def circle1_radius := 1

-- Define the second circle with center and radius
def circle2_center := (2 : ℝ, 1 : ℝ)
def circle2_radius := 4

-- Calculate the distance between the two centers
def distance := Real.sqrt ((-2 - 2)^2 + (0 - 1)^2)

-- Positional relationship between the two circles
theorem circles_intersect :
  distance < circle1_radius + circle2_radius ∧ distance > abs (circle1_radius - circle2_radius) :=
by
  sorry

end circles_intersect_l483_483253


namespace rationalize_denominator_proof_l483_483450

def rationalize_denominator (cbrt : ℝ → ℝ) (a : ℝ) :=
  cbrt a = a^(1/3)

theorem rationalize_denominator_proof : 
  (rationalize_denominator (λ x, x ^ (1/3)) 27) →
  (rationalize_denominator (λ x, x ^ (1/3)) 9) →
  (1 / (3 ^ (1 / 3) + 3) = 9 ^ (1 / 3) / (3 + 9 * 3 ^ (1 / 3))) :=
by
  sorry

end rationalize_denominator_proof_l483_483450


namespace smallest_four_digit_mod_8_l483_483914

theorem smallest_four_digit_mod_8 :
  ∃ x : ℕ, x ≥ 1000 ∧ x < 10000 ∧ x % 8 = 5 ∧ (∀ y : ℕ, y >= 1000 ∧ y < 10000 ∧ y % 8 = 5 → x ≤ y) :=
sorry

end smallest_four_digit_mod_8_l483_483914


namespace sum_of_three_squares_81_l483_483368

-- Define a predicate to check if three numbers are perfect squares
def is_sum_of_three_squares (n a b c : ℕ) : Prop :=
  n = a*a + b*b + c*c

-- Main theorem: there are exactly 3 ways to write 81 as the sum of three positive perfect squares
theorem sum_of_three_squares_81 {a b c : ℕ} :
  (is_sum_of_three_squares 81 a b c) → (a > 0) → (b > 0) → (c > 0) →
  (∃ (a1 b1 c1 a2 b2 c2 a3 b3 c3 : ℕ),
   ((a = a1 ∧ b = b1 ∧ c = c1) ∨ (a = a2 ∧ b = b2 ∧ c = c2) ∨ (a = a3 ∧ b = b3 ∧ c = c3)) ∧
   (is_sum_of_three_squares 81 a1 b1 c1) ∧ (is_sum_of_three_squares 81 a2 b2 c2) ∧
   (is_sum_of_three_squares 81 a3 b3 c3) ∧ (a1 = 1 ∧ b1 = 4 ∧ c1 = 8 ∨ a1 = 4 ∧ b1 = 1 ∧ c1 = 8 ∨ a1 = 4 ∧ b1 = 8 ∧ c1 = 1 ∨ ...)
   sorry

end sum_of_three_squares_81_l483_483368


namespace expected_waiting_time_first_bite_l483_483234

-- Definitions and conditions as per the problem
def poisson_rate := 6  -- lambda value, bites per 5 minutes
def interval_minutes := 5
def interval_seconds := interval_minutes * 60
def expected_waiting_time_seconds := interval_seconds / poisson_rate

-- The theorem we want to prove
theorem expected_waiting_time_first_bite :
  expected_waiting_time_seconds = 50 := 
by
  let x := interval_seconds / poisson_rate
  have h : interval_seconds = 300 := by norm_num; rfl
  have h2 : x = 50 := by rw [h, interval_seconds]; norm_num
  exact h2

end expected_waiting_time_first_bite_l483_483234


namespace centroid_value_l483_483500

-- Define vertices P, Q, and R
def P : (ℝ × ℝ) := (-1, 3)
def Q : (ℝ × ℝ) := (2, 7)
def R : (ℝ × ℝ) := (4, 0)

-- Define the centroid of the triangle PQR
def centroid (P Q R : (ℝ × ℝ)) : (ℝ × ℝ) :=
  ((P.1 + Q.1 + R.1) / 3, (P.2 + Q.2 + R.2) / 3)

-- Calculate the centroid of the triangle with given vertices
def S := centroid P Q R

-- Final goal to prove
theorem centroid_value : 8 * S.1 + 3 * S.2 = 70 / 3 := by
  sorry

end centroid_value_l483_483500


namespace tangent_slope_at_1_l483_483321

noncomputable def f (x : ℝ) (f'1 : ℝ) : ℝ := x^2 + 2 * f'1 * Real.log x

noncomputable def f' (x : ℝ) (f'1 : ℝ) : ℝ := 2 * x + 2 * f'1 / x

theorem tangent_slope_at_1 : ∃ f'1 : ℝ, (f' 1 f'1) = -2 :=
begin
  use -2,
  sorry,
end

end tangent_slope_at_1_l483_483321


namespace strips_overlap_area_l483_483979

theorem strips_overlap_area 
  (length_total : ℝ) 
  (length_left : ℝ) 
  (length_right : ℝ) 
  (area_only_left : ℝ) 
  (area_only_right : ℝ) 
  (length_total_eq : length_total = 16) 
  (length_left_eq : length_left = 9) 
  (length_right_eq : length_right = 7) 
  (area_only_left_eq : area_only_left = 27) 
  (area_only_right_eq : area_only_right = 18) :
  ∃ S : ℝ, (27 + S) / (18 + S) = (9 / 7) ∧ 2 * S = 27 :=
begin
  use 13.5,
  split,
  {
    -- Show proportional relationship holds
    sorry
  },
  {
    -- Show 2 * S = 27
    sorry
  }
end

end strips_overlap_area_l483_483979


namespace babjis_height_less_by_20_percent_l483_483574

variable (B A : ℝ) (h : A = 1.25 * B)

theorem babjis_height_less_by_20_percent : ((A - B) / A) * 100 = 20 := by
  sorry

end babjis_height_less_by_20_percent_l483_483574


namespace parallelogram_angle_ratio_l483_483003

theorem parallelogram_angle_ratio (ABCD : Type*)
  [parallelogram ABCD] 
  (O : point) 
  (intersects_diag : diagonal_intersect_at ABCD O) 
  (alpha : angle) 
  (h1 : angle_AOB * 3 = angle_DBA)
  (h2 : angle_CAB * 3 = angle_DBC)
  (h3 : angle_ACB = r * angle_AOB) :
  r = 2 :=
by
  sorry

end parallelogram_angle_ratio_l483_483003


namespace MN_passes_through_midpoint_EF_l483_483741

/-
Given:
- A cyclic quadrilateral ABCD.
- M is the midpoint of AD.
- N is the foot of the perpendicular from M to BC.
- E is the foot of the perpendicular from A to BC.
- F is the foot of the perpendicular from D to BC.

Prove:
- MN passes through the midpoint of EF.
-/
theorem MN_passes_through_midpoint_EF
  (A B C D M N E F : ℝ) 
  (cyclic_quadrilateral : cyclic A B C D)
  (midpoint_M : M = (A + D) / 2)
  (foot_N : N = foot_of_perpendicular M B C)
  (foot_E : E = foot_of_perpendicular A B C)
  (foot_F : F = foot_of_perpendicular D B C) :
  Line.through M N → Line.passes_through_midpoint MN E F :=
sorry

end MN_passes_through_midpoint_EF_l483_483741


namespace distance_from_C_to_D_l483_483287

theorem distance_from_C_to_D :
  let south_displacement := 50 - 30,
      west_displacement := 80 - 40,
      distance := Real.sqrt (20^2 + 40^2)
  in
  distance = 20 * Real.sqrt 5 := by
  sorry

end distance_from_C_to_D_l483_483287


namespace cost_of_bread_l483_483402

theorem cost_of_bread :
  ∀ (total_money initial_leftover cost_of_milk cost_of_detergent cost_of_bananas cost_of_bread : ℝ),
  total_money = 20 →
  initial_leftover = 4 →
  cost_of_milk = 2 →
  cost_of_detergent = 9 →
  cost_of_bananas = 1.5 →
  total_money - initial_leftover = cost_of_milk + cost_of_detergent + cost_of_bananas + cost_of_bread →
  cost_of_bread = 3.5 :=
by
  intros total_money initial_leftover cost_of_milk cost_of_detergent cost_of_bananas cost_of_bread
  assume h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5] at h6
  -- skipped proof
  sorry

end cost_of_bread_l483_483402


namespace totalStudents_correct_l483_483478

-- Defining the initial number of classes, students per class, and new classes
def initialClasses : ℕ := 15
def studentsPerClass : ℕ := 20
def newClasses : ℕ := 5

-- Prove that the total number of students is 400
theorem totalStudents_correct : 
  initialClasses * studentsPerClass + newClasses * studentsPerClass = 400 := by
  sorry

end totalStudents_correct_l483_483478


namespace interest_earned_l483_483511

noncomputable def simple_interest (P R T : ℚ) : ℚ :=
  P * R * T

noncomputable def T_years : ℚ :=
  5 + (8 / 12) + (12 / 365)

def principal : ℚ := 30000
def rate : ℚ := 23.7 / 100

theorem interest_earned :
  simple_interest principal rate T_years = 40524 := by
  sorry

end interest_earned_l483_483511


namespace smallest_four_digit_mod_8_l483_483908

theorem smallest_four_digit_mod_8 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 8 = 5 ∧ (∀ m : ℕ, m >= 1000 ∧ m < 10000 ∧ m % 8 = 5 → n ≤ m) → n = 1005 :=
by
  sorry

end smallest_four_digit_mod_8_l483_483908


namespace part1_part2_l483_483514

noncomputable def probability_A_receives_one_red_envelope : ℚ :=
  sorry

theorem part1 (P_A1 : ℚ) (P_not_A1 : ℚ) (P_A2 : ℚ) (P_not_A2 : ℚ) :
  P_A1 = 1/3 ∧ P_not_A1 = 2/3 ∧ P_A2 = 1/3 ∧ P_not_A2 = 2/3 →
  probability_A_receives_one_red_envelope = 4/9 :=
sorry

noncomputable def probability_B_receives_at_least_10_yuan : ℚ :=
  sorry

theorem part2 (P_B1 : ℚ) (P_not_B1 : ℚ) (P_B2 : ℚ) (P_not_B2 : ℚ) (P_B3 : ℚ) (P_not_B3 : ℚ) :
  P_B1 = 1/3 ∧ P_not_B1 = 2/3 ∧ P_B2 = 1/3 ∧ P_not_B2 = 2/3 ∧ P_B3 = 1/3 ∧ P_not_B3 = 2/3 →
  probability_B_receives_at_least_10_yuan = 11/27 :=
sorry

end part1_part2_l483_483514


namespace angle_C_eq_pi_div_6_l483_483747

noncomputable def find_angle_C (a b c : ℝ) (cosC sinC : ℝ) : ℝ :=
by
  have b := a * (cosC + (Real.sqrt 3 / 3) * sinC)
  have a := Real.sqrt 3
  have c := 1
  sorry

theorem angle_C_eq_pi_div_6 :
  ∃ (C : ℝ), find_angle_C (Real.sqrt 3) (Real.sqrt 3 * (cos (π / 6) + (Real.sqrt 3 / 3) * sin (π / 6))) 1 (cos (π / 6)) (sin (π / 6)) = π / 6 :=
by sorry

end angle_C_eq_pi_div_6_l483_483747


namespace expected_waiting_time_correct_l483_483232

noncomputable def combined_average_bites_per_5_minutes := 6
def average_waiting_time_for_first_bite_in_seconds : ℝ := 50

theorem expected_waiting_time_correct :
  (1 / combined_average_bites_per_5_minutes) * 300 = average_waiting_time_for_first_bite_in_seconds :=
by
  sorry

end expected_waiting_time_correct_l483_483232


namespace raptors_points_l483_483167

theorem raptors_points (x y z : ℕ) (h1 : x + y + z = 48) (h2 : x - y = 18) :
  (z = 0 → y = 15) ∧
  (z = 12 → y = 9) ∧
  (z = 18 → y = 6) ∧
  (z = 30 → y = 0) :=
by sorry

end raptors_points_l483_483167


namespace contrapositive_ex_l483_483800

theorem contrapositive_ex (x y : ℝ)
  (h : x^2 + y^2 = 0 → x = 0 ∧ y = 0) :
  ¬ (x = 0 ∧ y = 0) → x^2 + y^2 ≠ 0 :=
by
  sorry

end contrapositive_ex_l483_483800


namespace parametric_to_ordinary_l483_483244

theorem parametric_to_ordinary (θ : ℝ) :
  let x := 2 + Real.sin θ ^ 2,
      y := Real.sin θ ^ 2 in
  y = x - 2 ∧ 2 ≤ x ∧ x ≤ 3 :=
by
  sorry

end parametric_to_ordinary_l483_483244


namespace expected_waiting_time_first_bite_l483_483238

-- Definitions and conditions as per the problem
def poisson_rate := 6  -- lambda value, bites per 5 minutes
def interval_minutes := 5
def interval_seconds := interval_minutes * 60
def expected_waiting_time_seconds := interval_seconds / poisson_rate

-- The theorem we want to prove
theorem expected_waiting_time_first_bite :
  expected_waiting_time_seconds = 50 := 
by
  let x := interval_seconds / poisson_rate
  have h : interval_seconds = 300 := by norm_num; rfl
  have h2 : x = 50 := by rw [h, interval_seconds]; norm_num
  exact h2

end expected_waiting_time_first_bite_l483_483238


namespace christine_wander_time_l483_483585

-- Definitions based on conditions
def distance : ℝ := 50.0
def speed : ℝ := 6.0

-- The statement to prove
theorem christine_wander_time : (distance / speed) = 8 + 20/60 :=
by
  sorry

end christine_wander_time_l483_483585


namespace neither_sufficient_nor_necessary_condition_l483_483713

theorem neither_sufficient_nor_necessary_condition (x : ℝ) :
  ¬ (∀ x ∈ ℝ, (x - 1) * (x + 3) < 0 → (x + 1) * (x - 3) < 0) ∧
  ¬ (∀ x ∈ ℝ, (x + 1) * (x - 3) < 0 → (x - 1) * (x + 3) < 0) := 
sorry

end neither_sufficient_nor_necessary_condition_l483_483713


namespace min_value_2a_plus_b_l483_483669

noncomputable theory

-- Given the conditions and the question, we write the Lean statement.
theorem min_value_2a_plus_b {a b : ℝ} (h1 : 0 < a) (h2 : 0 < b) (h3 : a^2 + 2 * a * b - 3 = 0) : 
  ∃ k, (2 * a + b = k) ∧ k ≥ 3 :=
sorry

end min_value_2a_plus_b_l483_483669


namespace sphere_radius_l483_483406

noncomputable def radius_of_sphere_centered_in_cube : ℝ :=
  2 - real.sqrt 2

theorem sphere_radius (r : ℝ) :
  ∃ r, 
    let A := (0, 0, 0) in
    let B := (1, 1, 1) in
    (∀ (x y z : ℝ), r = x ∧ r = y ∧ r = z) ∧ 
    (∀ (r : ℝ), sqrt ((r - r)^2 + (r - 1)^2 + (r - 1)^2) = r) ∧ 
    r = radius_of_sphere_centered_in_cube := 
begin
  use radius_of_sphere_centered_in_cube,
  sorry
end

end sphere_radius_l483_483406


namespace smallest_arithmetic_mean_divisible_product_l483_483837

theorem smallest_arithmetic_mean_divisible_product :
  ∃ (n : ℕ), (∀ i : ℕ, i < 9 → Nat.gcd ((n + i) * 1111) 1111 = 1111) ∧ (n + 4 = 97) := 
  sorry

end smallest_arithmetic_mean_divisible_product_l483_483837


namespace tangent_line_exponential_passing_through_origin_l483_483260

theorem tangent_line_exponential_passing_through_origin :
  ∃ (p : ℝ × ℝ) (m : ℝ), 
  (p = (1, Real.exp 1)) ∧ (m = Real.exp 1) ∧ 
  (∀ x : ℝ, x ≠ 1 → ¬ (∃ k : ℝ, k = (Real.exp x - 0) / (x - 0) ∧ k = Real.exp x)) :=
by 
  sorry

end tangent_line_exponential_passing_through_origin_l483_483260


namespace find_PR_l483_483728

open Real

-- Define points and lengths according to the conditions given.
variables (P Q R E G : Point)
variable (PR_length PQ_length : ℝ)

-- Define the lengths and conditions
axiom E_on_PQ : E ∈ PQ
axiom G_on_PR : G ∈ PR
axiom PQ_perpendicular_PR : perpendicular PQ PR
axiom PG_perpendicular_PR : perpendicular PG PR
axiom lengths_equal : QE = 3 ∧ EG = 3 ∧ GR = 3

-- The required length to find
noncomputable def PR := PR_length

-- Statement of the theorem
theorem find_PR : PR_length = 6 :=
sorry

end find_PR_l483_483728


namespace amazing_number_exists_l483_483509

theorem amazing_number_exists :
  ∃ x : ℝ, x = 1 ∧ x = 1 ∧ x = 1 := 
by
  use 1
  intro h
  split
  · exact h
  split
  · sorry
  sorry

end amazing_number_exists_l483_483509


namespace smallest_arithmetic_mean_divisible_by_1111_l483_483851

/-- 
Given the product of nine consecutive natural numbers is divisible by 1111, 
prove that the smallest possible value of the arithmetic mean of these nine numbers is 97.
-/
theorem smallest_arithmetic_mean_divisible_by_1111 :
  ∃ n : ℕ, (∀ k : ℕ, k = n →  (∏ i in finset.range 9, k + i) % 1111 = 0) 
  ∧ (n ≥ 93) ∧ (n + 4 = 97) :=
sorry

end smallest_arithmetic_mean_divisible_by_1111_l483_483851


namespace extremum_points_of_f_l483_483088

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then (x + 1)^3 * Real.exp (x + 1) - Real.exp 1
  else -((if -x < 0 then (-x + 1)^3 * Real.exp (-x + 1) - Real.exp 1 else 0))

theorem extremum_points_of_f : ∃! (a b : ℝ), 
  (∀ x < 0, f x = (x + 1)^3 * Real.exp (x + 1) - Real.exp 1) ∧ (f a = f b) ∧ a ≠ b :=
sorry

end extremum_points_of_f_l483_483088


namespace area_of_trapezoid_is_correct_length_of_BC_propertly_determined_l483_483081

namespace Geometry

-- Definitions based on the given conditions
def altitude : ℝ := 8
def AB : ℝ := 15
def CD : ℝ := 22

-- Given 
def area_of_trapezoid (h : ℝ) (a b : ℝ) : ℝ := (1 / 2) * (a + b) * h

-- Additional properties for rectangles and square roots
def square (x : ℝ) : ℝ := x * x
def AE : ℝ := Real.sqrt (square AB - square altitude)
def FD : ℝ := Real.sqrt (square CD - square altitude)

-- Prove the area and side lengths properties
theorem area_of_trapezoid_is_correct : 
  area_of_trapezoid altitude AB CD = 148 := by sorry

theorem length_of_BC_propertly_determined (BF : ℝ) : 
  (BF = altitude) → 
  (BC : ℝ) := by sorry

end Geometry

end area_of_trapezoid_is_correct_length_of_BC_propertly_determined_l483_483081


namespace strips_overlap_area_l483_483980

theorem strips_overlap_area 
  (length_total : ℝ) 
  (length_left : ℝ) 
  (length_right : ℝ) 
  (area_only_left : ℝ) 
  (area_only_right : ℝ) 
  (length_total_eq : length_total = 16) 
  (length_left_eq : length_left = 9) 
  (length_right_eq : length_right = 7) 
  (area_only_left_eq : area_only_left = 27) 
  (area_only_right_eq : area_only_right = 18) :
  ∃ S : ℝ, (27 + S) / (18 + S) = (9 / 7) ∧ 2 * S = 27 :=
begin
  use 13.5,
  split,
  {
    -- Show proportional relationship holds
    sorry
  },
  {
    -- Show 2 * S = 27
    sorry
  }
end

end strips_overlap_area_l483_483980


namespace pie_pre_cut_min_pieces_l483_483552

theorem pie_pre_cut_min_pieces (n : ℕ) : (∀ k ∈ {10, 11}, ∃ d : ℕ, n = k * d) → n = 20 :=
  sorry

end pie_pre_cut_min_pieces_l483_483552


namespace spherical_to_rectangular_coordinates_l483_483590

noncomputable
def convert_to_cartesian (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_coordinates :
  let ρ1 := 10
  let θ1 := Real.pi / 4
  let φ1 := Real.pi / 6
  let ρ2 := 15
  let θ2 := 5 * Real.pi / 4
  let φ2 := Real.pi / 3
  convert_to_cartesian ρ1 θ1 φ1 = (5 * Real.sqrt 2 / 2, 5 * Real.sqrt 2 / 2, 5 * Real.sqrt 3)
  ∧ convert_to_cartesian ρ2 θ2 φ2 = (-15 * Real.sqrt 6 / 4, -15 * Real.sqrt 6 / 4, 7.5) := 
by
  sorry

end spherical_to_rectangular_coordinates_l483_483590


namespace roots_sum_products_l483_483041

noncomputable def polynomial : Polynomial ℝ := 6 * Polynomial.X ^ 3 - 9 * Polynomial.X ^ 2 + 16 * Polynomial.X - 12

theorem roots_sum_products (p q r : ℝ) (h1 : Polynomial.root polynomial p)
  (h2 : Polynomial.root polynomial q) (h3 : Polynomial.root polynomial r) :
  p * q + p * r + q * r = 8 / 3 :=
sorry

end roots_sum_products_l483_483041


namespace four_pow_sub_divisible_iff_l483_483068

open Nat

theorem four_pow_sub_divisible_iff (m n k : ℕ) (h₁ : m > n) : 
  (3^(k + 1)) ∣ (4^m - 4^n) ↔ (3^k) ∣ (m - n) := 
by sorry

end four_pow_sub_divisible_iff_l483_483068


namespace distance_AD_between_38_and_39_l483_483441

noncomputable def A : ℝ × ℝ := (0, 0)
noncomputable def B : ℝ × ℝ := (20 * (real.cos (30 * real.pi / 180)), 0)
noncomputable def C : ℝ × ℝ := (20 * (real.cos (30 * real.pi / 180)), 20 * (real.sin (30 * real.pi / 180)))
noncomputable def D : ℝ × ℝ := 
  (C.1 + 40 * (real.cos (45 * real.pi / 180)),
   C.2 + 40 * (real.sin (45 * real.pi / 180)))

def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem distance_AD_between_38_and_39 : 38 < dist A D ∧ dist A D < 39 :=
  sorry

end distance_AD_between_38_and_39_l483_483441


namespace rationalize_denominator_proof_l483_483451

def rationalize_denominator (cbrt : ℝ → ℝ) (a : ℝ) :=
  cbrt a = a^(1/3)

theorem rationalize_denominator_proof : 
  (rationalize_denominator (λ x, x ^ (1/3)) 27) →
  (rationalize_denominator (λ x, x ^ (1/3)) 9) →
  (1 / (3 ^ (1 / 3) + 3) = 9 ^ (1 / 3) / (3 + 9 * 3 ^ (1 / 3))) :=
by
  sorry

end rationalize_denominator_proof_l483_483451


namespace cos_2pi_minus_alpha_l483_483667

noncomputable theory

open Real

-- Declare the conditions
variables {α : ℝ}
axiom h1 : α ∈ Ioo (π / 2) (3 * π / 2)
axiom h2 : tan α = - 12 / 5

-- Statement to prove
theorem cos_2pi_minus_alpha : cos (2 * π - α) = - 5 / 13 :=
by
  sorry

end cos_2pi_minus_alpha_l483_483667


namespace distinct_sums_impossible_l483_483392

-- Define the problem in Lean
theorem distinct_sums_impossible :
  ¬ ∃ (a : Fin 8 → Fin 8 → ℤ), 
    (∀ i, ∀ j, a i j ∈ {1, -1, 0}) ∧ 
    let row_sums := (Finset.univ : Finset (Fin 8)).image (fun i => ∑ j, a i j),
        col_sums := (Finset.univ : Finset (Fin 8)).image (fun j => ∑ i, a i j),
        diag1_sum := (∑ k, a k k),
        diag2_sum := (∑ k, a k (7 - k)) in
    row_sums ∪ col_sums ∪ {diag1_sum, diag2_sum} = Finset.univ.Image (fun (x : Fin 18) => (of_nat x.1 - 8) : ℤ) :=
sorry

end distinct_sums_impossible_l483_483392


namespace intercepts_of_line_l483_483622

theorem intercepts_of_line (x y : ℝ) 
  (h : 2 * x + 7 * y = 35) :
  (y = 5 → x = 0) ∧ (x = 17.5 → y = 0)  :=
by
  sorry

end intercepts_of_line_l483_483622


namespace five_distinct_real_roots_iff_l483_483760

def f (x c : ℝ) : ℝ := x^2 + 4 * x + c

def g (x c : ℝ) : ℝ := (f x c)^2 + 4 * (f x c) + c

theorem five_distinct_real_roots_iff (c : ℝ) :
  (∃ roots : finset ℝ, roots.card = 5 ∧ ∀ x ∈ roots, g x c = 0) ↔ (c = 0 ∨ c = 3) :=
by
  sorry

end five_distinct_real_roots_iff_l483_483760


namespace solve_hens_count_l483_483555

noncomputable def count_hens (H C : ℕ) : Prop :=
  H + C = 44 ∧ 2 * H + 4 * C = 128

theorem solve_hens_count : ∃ H C: ℕ, count_hens H C ∧ H = 24 :=
by
  use 24
  use 20
  split
  { unfold count_hens
    split
    { exact rfl }
    { norm_num } }
  exact rfl

end solve_hens_count_l483_483555


namespace solution_to_eq_diamondsuit_l483_483957

theorem solution_to_eq_diamondsuit (a b c : ℝ) (y : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0)
  (h₃ : ∀ a b c : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 → a ⊙ (b ⊙ c) = (a ⊙ b) * c)
  (h₄ : ∀ a : ℝ, a ≠ 0 → a ⊙ a = 1) :
  2024 ⊙ (8 ⊙ y) = 200 → y = 200 / 253 :=
by sorry

end solution_to_eq_diamondsuit_l483_483957


namespace sum_of_painted_segments_leq_0_5_l483_483772

noncomputable def segment_painting {α : Type*} [linear_ordered_field α] (painted_segments : set (set α)) :=
  ∀ (s ∈ painted_segments) (t ∈ painted_segments), 
    s ≠ t → ∀ x ∈ s, ∀ y ∈ t, abs (x - y) ≠ (0.1 : α)

theorem sum_of_painted_segments_leq_0_5 {α : Type*} [linear_ordered_field α] :
  ∀ (painted_segments : set (set α)),
    (∀ s ∈ painted_segments, ∃ x y : α, 0 ≤ x ∧ y ≤ 1 ∧ s = set.interval x y) →
    segment_painting painted_segments →
    ∑ s in painted_segments, measure_theory.measure (set.Icc (0 : α) 1) s ≤ (0.5 : α) :=
by
  sorry

end sum_of_painted_segments_leq_0_5_l483_483772


namespace temperature_problem_l483_483576

theorem temperature_problem
  (M L N : ℝ)
  (h1 : M = L + N)
  (h2 : M - 9 = M - 9)
  (h3 : L + 5 = L + 5)
  (h4 : abs (M - 9 - (L + 5)) = 1) :
  (N = 15 ∨ N = 13) → (N = 15 ∧ N = 13 → 15 * 13 = 195) :=
by
  sorry

end temperature_problem_l483_483576


namespace parallelogram_rational_relation_l483_483004

theorem parallelogram_rational_relation
  (ABCD : Type) [parallelogram ABCD]
  (O : Type) [intersection O (diagonal A C) (diagonal B D)]
  (theta : Real)
  (angle_CAB : Real) (angle_CAB_eq : angle_CAB = 3 * theta)
  (angle_DBC : Real) (angle_DBC_eq : angle_DBC = 3 * theta)
  (angle_DBA : Real) (angle_DBA_eq : angle_DBA = theta) 
  (angle_AOB : Real) (angle_AOB_eq : angle_AOB = 180 - 4 * theta)
  (angle_ACB : Real) (angle_ACB_eq : angle_ACB = 180 - 7 * theta) :
  r = 5 / 8 :=
sorry

end parallelogram_rational_relation_l483_483004


namespace water_volume_correct_l483_483984

-- Defining the given conditions
def river_depth : ℝ := 4
def river_width : ℝ := 22
def flow_rate_kmph : ℝ := 2

-- Conversion definitions
def flow_rate_mph : ℝ := flow_rate_kmph * 1000
def flow_rate_mpm : ℝ := flow_rate_mph / 60
def cross_sectional_area : ℝ := river_depth * river_width

-- The function to calculate the volume of water flowing per minute
def volume_per_minute : ℝ := cross_sectional_area * flow_rate_mpm

-- The proof problem statement
theorem water_volume_correct : volume_per_minute = 2933.04 := 
by
  unfold river_depth river_width flow_rate_kmph flow_rate_mph flow_rate_mpm cross_sectional_area volume_per_minute
  sorry

end water_volume_correct_l483_483984


namespace depletion_rate_per_annum_l483_483172

theorem depletion_rate_per_annum (initial_value final_value: ℝ) (years : ℕ) (r : ℝ)
  (h1 : initial_value = 1000)
  (h2 : final_value = 810)
  (h3 : years = 2)
  (h4 : final_value = initial_value * (1 - r) ^ years) : 
  r = 0.1 :=
by 
  subst h1
  subst h2
  subst h3
  rw [← h4]
  sorry

end depletion_rate_per_annum_l483_483172


namespace shirt_cost_l483_483938

theorem shirt_cost (J S : ℝ) (h1 : 3 * J + 2 * S = 69) (h2 : 2 * J + 3 * S = 86) : S = 24 :=
by
  sorry

end shirt_cost_l483_483938


namespace _l483_483043

noncomputable def sin_cos_ratio_theorem (u v : ℝ)
  (h1 : sin u / sin v = 4) 
  (h2 : cos u / cos v = 1/3) : 
  sin (2 * u) / sin (2 * v) + cos (2 * u) / cos (2 * v) = 19 / 381 :=
sorry

end _l483_483043


namespace distribution_plans_count_l483_483258

theorem distribution_plans_count (teachers classes : ℕ) (h_teachers : teachers = 5) (h_classes : classes = 4) :
    ∃ n, n = 4 ∧ (number_of_distribution_plans teachers classes) = n :=
by
  sorry

-- Placeholder for the function that calculates the number of distribution plans
noncomputable def number_of_distribution_plans (teachers classes : ℕ) : ℕ := sorry

end distribution_plans_count_l483_483258


namespace gdp_scientific_notation_l483_483377

theorem gdp_scientific_notation :
  (121 * 10^12 : ℝ) = 1.21 * 10^14 := by
  sorry

end gdp_scientific_notation_l483_483377


namespace sqrt5_plus_sqrt7_gt_1_plus_sqrt13_frac_sum_one_x_y_lt_3_l483_483948

-- Problem 1
theorem sqrt5_plus_sqrt7_gt_1_plus_sqrt13 : 
  sqrt 5 + sqrt 7 > 1 + sqrt 13 := 
by
  sorry

-- Problem 2
theorem frac_sum_one_x_y_lt_3 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y > 1) :
  (1 + x) / y < 3 ∨ (1 + y) / x < 3 := 
by
  sorry

end sqrt5_plus_sqrt7_gt_1_plus_sqrt13_frac_sum_one_x_y_lt_3_l483_483948


namespace placement_ways_2002_gon_l483_483369

theorem placement_ways_2002_gon :
  ∀ (n : ℕ), n = 2002 → (∃! (placement : fin n → fin n),
    (∀ i, abs (placement (fin.cast_add_one $ i % n) - placement (fin.cast_add_one $ (i + 1) % n)) ≤ 2) ∧
    (∃ p : perm (fin n), placement = p.to_fun)) → 2002 * 2 = 4004 :=
by 
  sorry

end placement_ways_2002_gon_l483_483369


namespace birthday_paradox_l483_483732

/-- 
In a class of 30 students, the probability that some two students share the same birthday
is more than 50%, assuming there are 365 days in a year.
-/
theorem birthday_paradox : 
  ∃ p : ℕ → ℝ, (∀ n, p n = (1 : ℝ) - ∏ i in finset.range n, (365 - i) / 365) ∧ p 30 > 0.5 := 
sorry

end birthday_paradox_l483_483732


namespace projection_AC_onto_AB_l483_483634

noncomputable def projection_vector
  (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
let AC := (C.1 - A.1, C.2 - A.2, C.3 - A.3) in
let AB := (B.1 - A.1, B.2 - A.2, B.3 - A.3) in
let dot_product := AC.1 * AB.1 + AC.2 * AB.2 + AC.3 * AB.3 in
let magnitude_squared := AB.1 * AB.1 + AB.2 * AB.2 + AB.3 * AB.3 in
(dot_product / magnitude_squared) * AB

theorem projection_AC_onto_AB
  (A B C : ℝ × ℝ × ℝ)
  (hA : A = (1, 1, 0))
  (hB : B = (0, 3, 0))
  (hC : C = (2, 2, 3)) :
  projection_vector A B C = (-1/5, 2/5, 0) :=
by
  sorry

end projection_AC_onto_AB_l483_483634


namespace length_of_sheet_l483_483974

theorem length_of_sheet (width_sheet margin_width area_picture : ℝ) 
  (h_width_sheet : width_sheet = 8.5) 
  (h_margin_width : margin_width = 1.5) 
  (h_area_picture : area_picture = 38.5) : 
  ∃ length_sheet : ℝ, length_sheet = 10 :=
by
  -- Definitions directly derived from the conditions:
  have width_picture : ℝ := width_sheet - 2 * margin_width,
  have h_width_picture : width_picture = 5.5 := by
    rw [h_width_sheet, h_margin_width],
    norm_num,

  have length_picture := area_picture / width_picture,
  have h_length_picture : length_picture = 7 := by
    rw [h_area_picture, h_width_picture],
    norm_num,

  -- Calculate the length of the sheet:
  have length_sheet := length_picture + 2 * margin_width,
  have h_length_sheet : length_sheet = 10 := by
    rw [h_length_picture, h_margin_width],
    norm_num,

  use length_sheet,
  exact h_length_sheet

end length_of_sheet_l483_483974


namespace Michael_digging_time_l483_483429

theorem Michael_digging_time (father_rate : ℝ) (father_time : ℝ) (req_depth_diff : ℝ) : 
  (father_rate = 4) → 
  (father_time = 400) → 
  (req_depth_diff = 400) →
  (let father_depth := father_rate * father_time; 
       michael_depth := 2 * father_depth - req_depth_diff;
       michael_time := michael_depth / father_rate in michael_time = 700) :=
by
  intros hr_1 hr_2 hr_3
  have father_depth : ℝ := father_rate * father_time
  have michael_depth : ℝ := 2 * father_depth - req_depth_diff
  have michael_time : ℝ := michael_depth / father_rate
  sorry

end Michael_digging_time_l483_483429


namespace point_in_first_quadrant_l483_483349

theorem point_in_first_quadrant (x y : ℝ) (h : |3 * x - 2 * y - 1| + real.sqrt (x + y - 2) = 0) : 0 < x ∧ 0 < y :=
by
  -- This is a placeholder for the proof.
  sorry

end point_in_first_quadrant_l483_483349


namespace fraction_studying_japanese_l483_483937

variable (J S : ℕ)
variable (hS : S = 3 * J)

def fraction_of_seniors_studying_japanese := (1 / 3 : ℚ) * S
def fraction_of_juniors_studying_japanese := (3 / 4 : ℚ) * J

def total_students := S + J

theorem fraction_studying_japanese (J S : ℕ) (hS : S = 3 * J) :
  ((1 / 3 : ℚ) * S + (3 / 4 : ℚ) * J) / (S + J) = 7 / 16 :=
by {
  -- proof to be filled in
  sorry
}

end fraction_studying_japanese_l483_483937


namespace altitude_length_l483_483567
noncomputable theory

-- Define the triangle with given median lengths and area
def triangle (ABC : Type) :=
  ∃ (median_5 median_9 : ℝ) (area : ℝ), median_5 = 5
  ∧ median_9 = 9 
  ∧ area = 4 * real.sqrt 21

-- Define the specific triangle with mentioned properties
def specific_triangle : Type :=
  { ABC // triangle ABC }

-- Prove that the length of the altitude from the vertex opposite the side bisected by the 5-inch median is 0.8√21
theorem altitude_length (t : specific_triangle) :
  ∃ (altitude : ℝ), altitude = 0.8 * real.sqrt 21 :=
begin
  sorry
end

end altitude_length_l483_483567


namespace find_lambda_l483_483659

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V) (λ : ℝ)

-- Conditions from the problem
def vectors_perp := inner_product_space.is_orthogonal a b
def magnitude_a := ∥a∥ = 2
def magnitude_b := ∥b∥ = 3
def perp_combination := inner_product_space.is_orthogonal (3 • a + 2 • b) (λ • a - b)

-- Problem statement to prove
theorem find_lambda 
  (h1 : vectors_perp) 
  (h2 : magnitude_a) 
  (h3 : magnitude_b) 
  (h4 : perp_combination) : 
  λ = 3 / 2 :=
begin
  sorry,
end

end find_lambda_l483_483659


namespace nine_consecutive_arithmetic_mean_divisible_1111_l483_483872

theorem nine_consecutive_arithmetic_mean_divisible_1111 {n : ℕ} (h1 : ∀ i : ℕ, 0 ≤ i ∧ i < 9 → Nat.Prime (n + i)) :
  ∃ n : ℕ, (∀ k : ℕ, 0 ≤ k ∧ k < 9 → (n + k) ∣ 1111) → (n + 4) = 97 := by
  sorry

end nine_consecutive_arithmetic_mean_divisible_1111_l483_483872


namespace find_overlapping_area_l483_483977

-- Definitions based on conditions
def length_total : ℕ := 16
def length_strip1 : ℕ := 9
def length_strip2 : ℕ := 7
def area_only_strip1 : ℚ := 27
def area_only_strip2 : ℚ := 18

-- Widths are the same for both strips, hence areas are proportional to lengths
def area_ratio := (length_strip1 : ℚ) / (length_strip2 : ℚ)

-- The Lean statement to prove the question == answer
theorem find_overlapping_area : 
  ∃ S : ℚ, (area_only_strip1 + S) / (area_only_strip2 + S) = area_ratio ∧ 
              area_only_strip1 + S = area_only_strip1 + 13.5 := 
by 
  sorry

end find_overlapping_area_l483_483977


namespace Clarissa_photos_needed_l483_483594

theorem Clarissa_photos_needed :
  (7 + 10 + 9 <= 40) → 40 - (7 + 10 + 9) = 14 :=
by
  sorry

end Clarissa_photos_needed_l483_483594


namespace nine_consecutive_arithmetic_mean_divisible_1111_l483_483866

theorem nine_consecutive_arithmetic_mean_divisible_1111 {n : ℕ} (h1 : ∀ i : ℕ, 0 ≤ i ∧ i < 9 → Nat.Prime (n + i)) :
  ∃ n : ℕ, (∀ k : ℕ, 0 ≤ k ∧ k < 9 → (n + k) ∣ 1111) → (n + 4) = 97 := by
  sorry

end nine_consecutive_arithmetic_mean_divisible_1111_l483_483866


namespace lucas_identity_785_l483_483749

def lucas : ℕ → ℤ
| 0     := 2
| 1     := 1
| (n+2) := lucas (n+1) + lucas n

theorem lucas_identity_785 :
  lucas 784 * lucas 786 - lucas 785 ^ 2 = 1 :=
by sorry

end lucas_identity_785_l483_483749


namespace sequence_eventually_constant_mod_l483_483073

theorem sequence_eventually_constant_mod (n : ℕ) (hn : n ≥ 1) :
  ∃ c, ∀ m ≥ c, a_m % n = a_c % n :=
by
  sorry

def a : ℕ → ℕ
| 0     := 2
| (n+1) := 2 ^ a n

end sequence_eventually_constant_mod_l483_483073


namespace complex_number_z_l483_483643

open Complex

theorem complex_number_z (a : ℝ) (z : ℂ) (hz : z = (4 + 2*𝐢) / (a + 𝐢))
  (hz_mag : abs z = Real.sqrt 10) : 
  z = 3 - 𝐢 ∨ z = -1 - 3*𝐢 :=
by sorry

end complex_number_z_l483_483643


namespace base_n_representation_l483_483404

theorem base_n_representation 
  (n : ℕ) 
  (hn : n > 0)
  (a b c : ℕ) 
  (ha : 0 ≤ a ∧ a < n)
  (hb : 0 ≤ b ∧ b < n) 
  (hc : 0 ≤ c ∧ c < n) 
  (h_digits_sum : a + b + c = 24)
  (h_base_repr : 1998 = a * n^2 + b * n + c) 
  : n = 15 ∨ n = 22 ∨ n = 43 :=
sorry

end base_n_representation_l483_483404


namespace value_G_at_4_l483_483200

def G (x : ℝ) : ℝ :=
  sqrt (abs (x - 1)) + (7 / real.pi) * atan (sqrt (abs (x - 2)))

theorem value_G_at_4 : G 4 = 6 :=
by
  sorry

end value_G_at_4_l483_483200


namespace yujin_wire_length_is_correct_l483_483754

def junhoe_wire_length : ℝ := 134.5
def multiplicative_factor : ℝ := 1.06
def yujin_wire_length (junhoe_length : ℝ) (factor : ℝ) : ℝ := junhoe_length * factor

theorem yujin_wire_length_is_correct : 
  yujin_wire_length junhoe_wire_length multiplicative_factor = 142.57 := 
by 
  sorry

end yujin_wire_length_is_correct_l483_483754


namespace max_abs_sum_l483_483423

theorem max_abs_sum (n : ℕ) (a : Fin n → ℕ) (h : ∀ i : Fin n, a i ∈ Finset.range (n + 1) ∧ ∀ i j, i ≠ j → a i ≠ a j) : 
  (∑ i, | a i - (i + 1) |) ≤ ⌊ n^2 / 2 ⌋ :=
sorry

end max_abs_sum_l483_483423


namespace smallest_arithmetic_mean_l483_483877

noncomputable def S (n : ℕ) := (List.range' n 9).map Nat.ofNat

theorem smallest_arithmetic_mean (n : ℕ) (h1 : 93 ≤ n) (h2 : ∃ k ∈ S n, 11 ∣ k) (h3 : ∃ k ∈ S n, 101 ∣ k) : 
  (n + 4 = 97) := by
  sorry

end smallest_arithmetic_mean_l483_483877


namespace centroid_of_A1B1C1_is_midpoint_of_PG_l483_483378

variables {A B C P A1 B1 C1 G M : Type*}

def is_centroid (G : Type*) (A B C : Type*) : Prop :=
  ∃ (α β γ : ℝ), α + β + γ = 1 ∧ (α • A + β • B + γ • C = G)

def is_median_parallel (P : Type*) (A1 B1 C1 A B C : Type*) : Prop :=
  -- Define the property of lines through P being parallel to the medians of triangle ABC and intersecting at A1, B1, and C1
  sorry

def is_midpoint (M : Type*) (P G : Type*) : Prop :=
  -- Define the property of M being the midpoint of segment PG
  sorry

theorem centroid_of_A1B1C1_is_midpoint_of_PG
  (h_median_parallel : is_median_parallel P A1 B1 C1 A B C)
  (h_centroid_ABC : is_centroid G A B C)
  : is_midpoint (is_centroid (A1 B1 C1) A1 B1 C1) (P, G) :=
begin
  sorry
end

end centroid_of_A1B1C1_is_midpoint_of_PG_l483_483378


namespace probability_product_divisible_by_10_l483_483562

noncomputable def prob_divisible_by_10 (n : ℕ) (hn : n > 1) : ℝ :=
  1 - (8 / 9) ^ n - (5 / 9) ^ n + (4 / 9) ^ n

theorem probability_product_divisible_by_10 {n : ℕ} (hn : n > 1) :
  prob_divisible_by_10 n hn = 1 - (8 / 9) ^ n - (5 / 9) ^ n + (4 / 9) ^ n :=
by
  intro n hn
  unfold prob_divisible_by_10
  sorry

end probability_product_divisible_by_10_l483_483562


namespace find_ordered_pairs_l483_483177

theorem find_ordered_pairs (a b x : ℕ) (h1 : b > a) (h2 : a + b = 15) (h3 : (a - 2 * x) * (b - 2 * x) = 2 * a * b / 3) :
  (a, b) = (8, 7) :=
by
  sorry

end find_ordered_pairs_l483_483177


namespace loads_of_laundry_l483_483532

theorem loads_of_laundry (families : ℕ) (days : ℕ) (adults_per_family : ℕ) (children_per_family : ℕ)
  (adult_towels_per_day : ℕ) (child_towels_per_day : ℕ) (initial_capacity : ℕ) (reduced_capacity : ℕ)
  (initial_days : ℕ) (remaining_days : ℕ) : 
  families = 7 → days = 12 → adults_per_family = 2 → children_per_family = 4 → 
  adult_towels_per_day = 2 → child_towels_per_day = 1 → initial_capacity = 8 → 
  reduced_capacity = 6 → initial_days = 6 → remaining_days = 6 → 
  (families * (adults_per_family * adult_towels_per_day + children_per_family * child_towels_per_day) * initial_days / initial_capacity) +
  (families * (adults_per_family * adult_towels_per_day + children_per_family * child_towels_per_day) * remaining_days / reduced_capacity) = 98 :=
by 
  intros _ _ _ _ _ _ _ _ _ _
  sorry

end loads_of_laundry_l483_483532


namespace paint_intensity_change_l483_483476

theorem paint_intensity_change (intensity_original : ℝ) (intensity_new : ℝ) (fraction_replaced : ℝ) 
  (h1 : intensity_original = 0.40) (h2 : intensity_new = 0.20) (h3 : fraction_replaced = 1) :
  intensity_new = 0.20 :=
by
  sorry

end paint_intensity_change_l483_483476


namespace intersection_of_A_and_B_l483_483300

def setA (x : ℝ) : Prop := x^2 < 4
def setB : Set ℝ := {0, 1}

theorem intersection_of_A_and_B :
  {x : ℝ | setA x} ∩ setB = setB := by
  sorry

end intersection_of_A_and_B_l483_483300


namespace probability_divisible_by_7_l483_483409

noncomputable def count_set_T : ℕ :=
  (Finset.range 50).choose 3

noncomputable def divisible_by_7_count : ℕ :=
  17 ^ 3

theorem probability_divisible_by_7 (p q : ℕ) (hpq : Nat.coprime p q) :
  let T := count_set_T
  let good := divisible_by_7_count
  p / q = good / T ∧ p = 4913 ∧ q = 19600 ∧ p + q = 24513 :=
sorry

end probability_divisible_by_7_l483_483409


namespace count_terminating_decimals_l483_483625

def terminates_decimal (m : ℕ) (d : ℕ) : Prop :=
  ∃ k : ℕ, d = 2^k * 5^k

theorem count_terminating_decimals (lower_bound upper_bound divisor : ℕ) :
  (∀ m, lower_bound ≤ m ∧ m ≤ upper_bound → terminates_decimal m divisor) → 
  Nat.floor (upper_bound / 49) = 10 :=
by
  sorry

#eval count_terminating_decimals 1 500 980  -- This should give 10

end count_terminating_decimals_l483_483625


namespace product_of_two_consecutive_even_numbers_is_divisible_by_8_l483_483138

theorem product_of_two_consecutive_even_numbers_is_divisible_by_8 (n : ℤ) : (4 * n * (n + 1)) % 8 = 0 :=
sorry

end product_of_two_consecutive_even_numbers_is_divisible_by_8_l483_483138


namespace area_of_overlap_l483_483983

theorem area_of_overlap 
  (len1 len2 : ℕ) (area_left only_left_area : ℚ) (area_right only_right_area : ℚ) (w : ℚ)
  (h_len1 : len1 = 9) (h_len2 : len2 = 7) (h_only_left_area : only_left_area = 27) 
  (h_only_right_area : only_right_area = 18) (h_w : w > 0)
  (h_area_left : area_left = only_left_area + (w * 1))
  (h_area_right : area_right = only_right_area + (w * 1))
  (h_ratio : (w * len1) / (w * len2) = 9 / 7) : 
  (13.5) :=
by
  sorry

end area_of_overlap_l483_483983


namespace extreme_value_at_half_sum_of_extremes_greater_than_zero_exp_n_choose_2_gt_factorial_l483_483324

noncomputable def f (a x : ℝ) : ℝ := ln (1 + a * x) - 2 * x / (x + 2)

theorem extreme_value_at_half :
  f 0.5 2 = ln 2 - 1 ∧ (∀ x, x ≠ 2 → f 0.5 x ≠ ln 2 - 1) :=
sorry

theorem sum_of_extremes_greater_than_zero {a : ℝ} (ha : 0.5 < a) (h₂a : a < 1) :
  let x₁ := 2 * sqrt (a * (1 - a)) / a,
      x₂ := -2 * sqrt (a * (1 - a)) / a in
  f a x₁ + f a x₂ > f a 0 :=
sorry

theorem exp_n_choose_2_gt_factorial (n : ℕ) (hn : 2 ≤ n) :
  exp (n * (n - 1) / 2) > n.fact :=
sorry

end extreme_value_at_half_sum_of_extremes_greater_than_zero_exp_n_choose_2_gt_factorial_l483_483324


namespace find_lengths_of_DE_and_HJ_l483_483739

noncomputable def lengths_consecutive_segments (BD DE EF FG GH HJ : ℝ) (BC : ℝ) : Prop :=
  BD = 5 ∧ EF = 11 ∧ FG = 7 ∧ GH = 3 ∧ BC = 29 ∧ BD + DE + EF + FG + GH + HJ = BC ∧ DE = HJ

theorem find_lengths_of_DE_and_HJ (x : ℝ) : lengths_consecutive_segments 5 x 11 7 3 x 29 → x = 1.5 :=
by
  intros h
  sorry

end find_lengths_of_DE_and_HJ_l483_483739


namespace expand_polynomial_l483_483608

theorem expand_polynomial (x : ℝ) : 
  3 * (x - 2) * (x^2 + x + 1) = 3 * x^3 - 3 * x^2 - 3 * x - 6 :=
by
  sorry

end expand_polynomial_l483_483608


namespace find_number_l483_483151

theorem find_number (x : ℤ) (h : (7 * (x + 10) / 5) - 5 = 44) : x = 25 :=
sorry

end find_number_l483_483151


namespace sqrt_expr_is_599_l483_483587

theorem sqrt_expr_is_599 : Real.sqrt ((26 * 25 * 24 * 23) + 1) = 599 := by
  sorry

end sqrt_expr_is_599_l483_483587


namespace polynomial_identity_l483_483480

theorem polynomial_identity (a b c : ℝ) : 
  a * (b - c)^3 + b * (c - a)^3 + c * (a - b)^3 = 
  (a - b) * (b - c) * (c - a) * (a + b + c) :=
sorry

end polynomial_identity_l483_483480


namespace min_area_ratio_of_inscribed_right_triangle_in_equilateral_l483_483673

theorem min_area_ratio_of_inscribed_right_triangle_in_equilateral (a b c d e f : ℝ) 
    (hDEF : (d = 0 ∧ e = 0 ∧ f = 0) ∨ ( (a,b,c) ≠ (0,0,0) ∧ ∠ DEF = 90 ∧ ∠ EDF = 30 )) 
    (hABC : (a,b,c) ∈ equilateral_triangle ) :
    ∃ DEF ABC : ℝ, min (frac_area DEF ABC) = 3 / 14 := 
sorry

end min_area_ratio_of_inscribed_right_triangle_in_equilateral_l483_483673


namespace simplify_cot20_tan10_l483_483466

theorem simplify_cot20_tan10 :
  (Real.cot 20 + Real.tan 10 = Real.csc 20) :=
sorry

end simplify_cot20_tan10_l483_483466


namespace simplify_cot_tan_l483_483470

theorem simplify_cot_tan :
  (Real.cot (Real.toRadians 20) + Real.tan (Real.toRadians 10) = Real.csc (Real.toRadians 20)) :=
by
  sorry

end simplify_cot_tan_l483_483470


namespace false_weight_l483_483168

theorem false_weight
    (gain_percentage : ℝ)
    (gain_percentage = 2.0408163265306145) :
    ∃ W : ℝ,
    (W = 1000 - (gain_percentage / 100) * 1000) :=
begin
    use 979.5918367346939,
    sorry
end

end false_weight_l483_483168


namespace intersecting_graphs_value_l483_483945

theorem intersecting_graphs_value (a b c d : ℝ) 
  (h1 : 5 = -|2 - a| + b) 
  (h2 : 3 = -|8 - a| + b) 
  (h3 : 5 = |2 - c| + d) 
  (h4 : 3 = |8 - c| + d) : 
  a + c = 10 :=
sorry

end intersecting_graphs_value_l483_483945


namespace cost_of_fencing_around_circular_field_l483_483614

noncomputable def cost_of_fencing 
  (d : ℝ) (rate : ℝ) : ℝ :=
  let pi := Real.pi in
  let circumference := pi * d in
  circumference * rate

theorem cost_of_fencing_around_circular_field :
  cost_of_fencing 31.5 2.25 ≈ 222.66 :=
by
  sorry

end cost_of_fencing_around_circular_field_l483_483614


namespace parabola_vertex_solution_l483_483972

theorem parabola_vertex_solution (
  (a b c : ℤ) -- Coefficients of the parabola equation
  (h k : ℤ) -- Coordinates of the vertex
  (px py : ℤ) -- Coordinates of the point it passes through
  (vertex_condition : h = 2 ∧ k = 1) -- Vertex condition
  (point_condition : px = 0 ∧ py = 5) -- Passes through point condition
  (parabola_eq_vertex : ∀ (x : ℤ), (a * (x - h)^2 + k) = (a * x^2 + b * x + c))
) : (a + b - c = -8) := by
  sorry

end parabola_vertex_solution_l483_483972


namespace trigonometric_ratio_l483_483045

open Real

theorem trigonometric_ratio (u v : ℝ) (h1 : sin u / sin v = 4) (h2 : cos u / cos v = 1 / 3) :
  (sin (2 * u) / sin (2 * v) + cos (2 * u) / cos (2 * v)) = 389 / 381 :=
by
  sorry

end trigonometric_ratio_l483_483045


namespace eve_stamp_collection_worth_l483_483607

def total_value_of_collection (stamps_value : ℕ) (num_stamps : ℕ) (set_size : ℕ) (set_value : ℕ) (bonus_per_set : ℕ) : ℕ :=
  let value_per_stamp := set_value / set_size
  let total_value := value_per_stamp * num_stamps
  let num_complete_sets := num_stamps / set_size
  let total_bonus := num_complete_sets * bonus_per_set
  total_value + total_bonus

theorem eve_stamp_collection_worth :
  total_value_of_collection 21 21 7 28 5 = 99 := by
  rfl

end eve_stamp_collection_worth_l483_483607


namespace desired_percentage_of_hydrocarbons_l483_483973

theorem desired_percentage_of_hydrocarbons
  (hc1 hc2 : ℝ) (v1 v2 : ℝ)
  (H1 : hc1 = 0.25) (H2 : hc2 = 0.75)
  (H3 : v2 = 30) (H4 : v1 + v2 = 50) :
  let pct_hydrocarbons := ((v2 * hc2) + (v1 * hc1)) / (v1 + v2) * 100 
  in pct_hydrocarbons = 55 := 
by
  -- The proof will go here
  sorry

end desired_percentage_of_hydrocarbons_l483_483973


namespace smallest_arithmetic_mean_divisible_product_l483_483836

theorem smallest_arithmetic_mean_divisible_product :
  ∃ (n : ℕ), (∀ i : ℕ, i < 9 → Nat.gcd ((n + i) * 1111) 1111 = 1111) ∧ (n + 4 = 97) := 
  sorry

end smallest_arithmetic_mean_divisible_product_l483_483836


namespace find_quadruples_l483_483265

def quadrupleSolution (a b c d : ℝ): Prop :=
  (a * (b + c) = b * (c + d) ∧ b * (c + d) = c * (d + a) ∧ c * (d + a) = d * (a + b))

def isSolution (a b c d : ℝ): Prop :=
  (a = 1 ∧ b = 0 ∧ c = 0 ∧ d = 0) ∨
  (a = 0 ∧ b = 1 ∧ c = 0 ∧ d = 0) ∨
  (a = 0 ∧ b = 0 ∧ c = 1 ∧ d = 0) ∨
  (a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 1) ∨
  (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1) ∨
  (a = 1 ∧ b = -1 ∧ c = 1 ∧ d = -1) ∨
  (a = 1 ∧ b = -1 + Real.sqrt 2 ∧ c = -1 ∧ d = 1 - Real.sqrt 2) ∨
  (a = 1 ∧ b = -1 - Real.sqrt 2 ∧ c = -1 ∧ d = 1 + Real.sqrt 2)

theorem find_quadruples (a b c d : ℝ) :
  quadrupleSolution a b c d ↔ isSolution a b c d :=
sorry

end find_quadruples_l483_483265


namespace smallest_arithmetic_mean_divisible_by_1111_l483_483829

noncomputable def nine_consecutive_numbers {n : ℕ} : list ℕ :=
  [n, n+1, n+2, n+3, n+4, n+5, n+6, n+7, n+8]

def divisible_by (x y : ℕ) : Prop := ∃ k : ℕ, x = k * y

def arithmetic_mean {l : list ℕ} (h_len : l.length = 9) : ℚ :=
  (l.sum : ℚ) / 9

theorem smallest_arithmetic_mean_divisible_by_1111 :
  ∃ n : ℕ, 
  divisible_by ((nine_consecutive_numbers n).prod) 1111 ∧ 
  arithmetic_mean (by simp [nine_consecutive_numbers_len]) = 97 :=
sorry

end smallest_arithmetic_mean_divisible_by_1111_l483_483829


namespace power_expression_eq_ten_l483_483581

theorem power_expression_eq_ten (a b c d: ℝ)
  (ha : a = 0.064)
  (hb : b = -7 / 8)
  (hc : c = 16)
  (hd : d = 0.25) :
  a^(-1 / 3) - b^0 + c^(3 / 4) + d^(1 / 2) = 10 := by
  sorry

end power_expression_eq_ten_l483_483581


namespace albert_runs_track_l483_483994

theorem albert_runs_track (x : ℕ) (track_distance : ℕ) (total_distance : ℕ) (additional_laps : ℕ) 
(h1 : track_distance = 9)
(h2 : total_distance = 99)
(h3 : additional_laps = 5)
(h4 : total_distance = track_distance * x + track_distance * additional_laps) :
x = 6 :=
by
  sorry

end albert_runs_track_l483_483994


namespace odd_function_inequality_l483_483056

-- Define f as an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem odd_function_inequality
  (f : ℝ → ℝ) (h1 : is_odd_function f)
  (a b : ℝ) (h2 : f a > f b) :
  f (-a) < f (-b) :=
by
  sorry

end odd_function_inequality_l483_483056


namespace exists_ints_sum_square_product_cube_l483_483445

theorem exists_ints_sum_square_product_cube (n : ℕ) (h_pos: 0 < n) : 
  ∃ (a : Fin n → ℕ), (∃ k1, (∑ i, a i) = k1^2) ∧ (∃ k2, (∏ i, a i) = k2^3) :=
sorry

end exists_ints_sum_square_product_cube_l483_483445


namespace significant_digits_of_square_side_l483_483811

theorem significant_digits_of_square_side (A : ℝ) (hA : A = 0.12321) : 
  ∃ s : ℝ, s * s = A ∧ num_significant_digits s = 5 :=
  sorry

end significant_digits_of_square_side_l483_483811


namespace variance_of_xi_l483_483054

noncomputable def d : ℝ := sorry
axiom nonzero_d : d ≠ 0

def arithmetic_seq (n : ℕ) : ℝ := x_1 + (n - 1) * d

def xi : ℝ → ℝ := sorry
axiom xi_values : ∀ n : ℕ, n ∈ (1 : ℕ) .. 9 → xi n = arithmetic_seq n

def variance (ξ : ℝ → ℝ) (mean : ℝ) : ℝ :=
  (1 / 9) * (Σ n in (finset.range 9).image (λ i, i + 1), (ξ n - mean) ^ 2)

theorem variance_of_xi :
  variance xi (arithmetic_seq 5) = (20 / 3) * d^2 :=
sorry

end variance_of_xi_l483_483054


namespace arnold_danny_age_l483_483195

theorem arnold_danny_age (x : ℕ) (h : (x + 1) * (x + 1) = x * x + 13) : x = 6 :=
by {
  sorry
}

end arnold_danny_age_l483_483195


namespace third_root_of_polynomial_l483_483503

theorem third_root_of_polynomial (a b : ℚ) : 
  (3 * b + a + 12 = 0) ∧ 
  (24 * a - 3 * b + 12 = 0) ∧ 
  (287 * a + 100 * b + 12 = 0) → 
  (∃ r : ℚ, r = -((\frac{205}{297}) - 1 + 3 - 4)) :=
by
  sorry

end third_root_of_polynomial_l483_483503


namespace largest_markers_package_l483_483603

theorem largest_markers_package (E T S : ℕ) (hE : E = 60) (hT : T = 36) (hS : S = 90) : 
  Nat.gcd (Nat.gcd E T) S = 6 :=
by
   rw [hE, hT, hS]
   -- This will replace E, T, and S with 60, 36, and 90 respectively
   -- Now we need to calculate the gcd
   have gcd_60_36 : Nat.gcd 60 36 = 12 := by sorry
   calc
     Nat.gcd (Nat.gcd 60 36) 90
         = Nat.gcd 12 90 : by rw gcd_60_36
         -- After these steps, you will get Nat.gcd 12 90
         -- Assuming we can calculate this directly
         = 6 : by sorry

end largest_markers_package_l483_483603


namespace smallest_positive_four_digit_equivalent_to_5_mod_8_l483_483911

theorem smallest_positive_four_digit_equivalent_to_5_mod_8 : 
  ∃ (n : ℕ), n ≥ 1000 ∧ n % 8 = 5 ∧ n = 1005 :=
by
  sorry

end smallest_positive_four_digit_equivalent_to_5_mod_8_l483_483911


namespace range_of_k_roots_for_neg_k_l483_483085

theorem range_of_k (k : ℝ) : (∃ x y : ℝ, x ≠ y ∧ (x^2 + (2*k + 1)*x + (k^2 - 1) = 0 ∧ y^2 + (2*k + 1)*y + (k^2 - 1) = 0)) ↔ k > -5 / 4 :=
by sorry

theorem roots_for_neg_k (k : ℤ) (h1 : k < 0) (h2 : k > -5 / 4) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + (2*k + 1)*x1 + (k^2 - 1) = 0 ∧ x2^2 + (2*k + 1)*x2 + (k^2 - 1) = 0 ∧ x1 = 0 ∧ x2 = 1)) :=
by sorry

end range_of_k_roots_for_neg_k_l483_483085


namespace expected_occur_two_consecutive_zeros_in_33_bits_string_l483_483294

theorem expected_occur_two_consecutive_zeros_in_33_bits_string :
  ∀ (s : string), s.length = 33 →
  (∀ i j, (string.get⟨i, h1⟩ = '0' ∧ string.get⟨i + 1, h2⟩ = '0')) →
  expected_value (occur_two_consecutive_zeros s) = 8 :=
by
  sorry

end expected_occur_two_consecutive_zeros_in_33_bits_string_l483_483294


namespace seashells_problem_l483_483426

theorem seashells_problem:
  let m := 0.5 in
  let k := 0.6 in
  let x := 2 in
  let y := 5 in
  let z := 9 in
  m * x + k * y = z → x + y = 7 :=
by
  intros
  sorry

end seashells_problem_l483_483426


namespace line_intersects_triangle_l483_483011

theorem line_intersects_triangle {A B C : Point} (triangle : Triangle A B C) (ℓ : Line) :
  ∃ (n : ℕ ∞), n = 0 ∨ n = 1 ∨ n = 2 ∨ n = ⊥ :=
sorry

end line_intersects_triangle_l483_483011


namespace smallest_arithmetic_mean_divisible_by_1111_l483_483832

noncomputable def nine_consecutive_numbers {n : ℕ} : list ℕ :=
  [n, n+1, n+2, n+3, n+4, n+5, n+6, n+7, n+8]

def divisible_by (x y : ℕ) : Prop := ∃ k : ℕ, x = k * y

def arithmetic_mean {l : list ℕ} (h_len : l.length = 9) : ℚ :=
  (l.sum : ℚ) / 9

theorem smallest_arithmetic_mean_divisible_by_1111 :
  ∃ n : ℕ, 
  divisible_by ((nine_consecutive_numbers n).prod) 1111 ∧ 
  arithmetic_mean (by simp [nine_consecutive_numbers_len]) = 97 :=
sorry

end smallest_arithmetic_mean_divisible_by_1111_l483_483832


namespace circle_area_is_correct_l483_483126

def radius : ℝ := 5

def area_of_circle (r : ℝ) : ℝ := π * r^2

theorem circle_area_is_correct : area_of_circle radius = 25 * π :=
by
  simp [area_of_circle, radius]
  sorry

end circle_area_is_correct_l483_483126


namespace trigonometric_ratio_l483_483044

open Real

theorem trigonometric_ratio (u v : ℝ) (h1 : sin u / sin v = 4) (h2 : cos u / cos v = 1 / 3) :
  (sin (2 * u) / sin (2 * v) + cos (2 * u) / cos (2 * v)) = 389 / 381 :=
by
  sorry

end trigonometric_ratio_l483_483044


namespace count_pairs_m_n_l483_483251

theorem count_pairs_m_n (m n : ℕ) : m > 0 → n > 0 → ∃! (k : ℕ), k = 67 ∧ (∀ m n : ℕ, 0 < m ∧ 0 < n → m^2 + 3 * n < 50 → (m, n).count = k) :=
by sorry

end count_pairs_m_n_l483_483251


namespace pi_approximation_l483_483630

-- Definition of the problem setting and the conditions
variables (n : ℕ) (x y : Fin n → ℝ)

/-- Each pair (x_i, y_i) is formed and lies within the interval [0,1] --/
def pairs_within_interval := ∀ i, 0 ≤ x i ∧ x i ≤ 1 ∧ 0 ≤ y i ∧ y i ≤ 1

/-- The pair (x_i, y_i) inside the unit circle --/
def pairs_in_unit_circle := ∀ i, (x i)^2 + (y i)^2 < 1

/-- m is the count of pairs inside the unit circle --/
noncomputable def m : ℕ := {
  count : ℕ // count = Finset.univ.filter (λ i => (x i)^2 + (y i)^2 < 1).card
}

/-- The approximation of π using the pairs in the unit circle --/
theorem pi_approximation (h1 : pairs_within_interval n x y)
                         (h2 : pairs_in_unit_circle n x y) :
                         (4 * m n x y / n : ℝ) = π :=
sorry

end pi_approximation_l483_483630


namespace min_arithmetic_mean_of_consecutive_naturals_div_by_1111_l483_483858

theorem min_arithmetic_mean_of_consecutive_naturals_div_by_1111 :
  ∀ a : ℕ, (∃ k, (a * (a + 1) * (a + 2) * (a + 3) * (a + 4) * (a + 5) * (a + 6) * (a + 7) * (a + 8)) = 1111 * k) →
  (a + 4 ≥ 97) :=
sorry

end min_arithmetic_mean_of_consecutive_naturals_div_by_1111_l483_483858


namespace range_of_m_l483_483722

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (x - m > 0) → (2*x + 1 > 3) → (x > 1)) → (m ≤ 1) :=
by
  intros h
  sorry

end range_of_m_l483_483722


namespace largest_divisor_l483_483928

theorem largest_divisor (A B : ℕ) (h : 24 = A * B + 4) : A ≤ 20 :=
sorry

end largest_divisor_l483_483928


namespace surrounding_circle_area_l483_483162

theorem surrounding_circle_area (R : ℝ) : 
  (∃ r : ℝ, r = R * (1 + Real.sqrt 2) ∧ ∃ S : ℝ, S = π * r^2) → 
  π * R^2 * (3 + 2 * Real.sqrt 2) = π * (R * (1 + Real.sqrt 2))^2 :=
by
  sorry

end surrounding_circle_area_l483_483162


namespace volume_of_open_box_l483_483136

-- Definitions based on the problem conditions.
def original_length : ℕ := 48
def original_width : ℕ := 36
def cut_side_length : ℕ := 7

-- The mathematical statement to prove.
theorem volume_of_open_box :
  let new_length := original_length - 2 * cut_side_length,
      new_width := original_width - 2 * cut_side_length,
      height := cut_side_length in
  new_length * new_width * height = 5236 :=
  by
    let new_length := original_length - 2 * cut_side_length
    let new_width := original_width - 2 * cut_side_length
    let height := cut_side_length
    sorry

end volume_of_open_box_l483_483136


namespace gcd_lcm_sum_l483_483794

-- Definitions based on the conditions
def G (a b : ℤ) : Polynomial ℤ := Polynomial.X^2 + (Polynomial.C a) * Polynomial.X + Polynomial.C b
def H (b c : ℤ) : Polynomial ℤ := Polynomial.X^2 + (Polynomial.C b) * Polynomial.X + Polynomial.C c

-- Statement of the theorem
theorem gcd_lcm_sum (a b c : ℤ) 
  (h1 : Polynomial.gcd (G a b) (H b c) = Polynomial.X + 1)
  (h2 : Polynomial.lcm (G a b) (H b c) = Polynomial.X^3 - 4 * Polynomial.X^2 + Polynomial.X + 6) :
  a + b + c = -6 :=
by
  -- Proof is omitted
  sorry

end gcd_lcm_sum_l483_483794


namespace train_pass_time_approx_18_seconds_l483_483137

noncomputable def train_pass_tree_time (train_length : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  train_length / train_speed_mps

theorem train_pass_time_approx_18_seconds :
  train_pass_tree_time 250 50 ≈ 18 :=
by
  sorry

end train_pass_time_approx_18_seconds_l483_483137


namespace inclination_angle_range_l483_483355

theorem inclination_angle_range (k : ℝ) (h : -1 ≤ k ∧ k ≤ sqrt 3) :
  ∃ α : ℝ, 0 ≤ α ∧ α < π ∧ (tan α = k) ∧ (α ∈ set.Icc 0 (π / 3) ∪ set.Icc (3 * π / 4) π) :=
sorry

end inclination_angle_range_l483_483355


namespace hyperbola_equation_l483_483310

theorem hyperbola_equation (a b : ℝ) (e : ℝ) :
  e = sqrt 5 →
  a > 0 →
  b > 0 →
  (a = 1) →
  (c = sqrt 5) →
  (c^2 = a^2 + b^2) →
  (x : ℝ) (y : ℝ) :
  x^2 - (y^2 / 4) = 1 := sorry

end hyperbola_equation_l483_483310


namespace find_length_SV_l483_483017

structure Triangle (P Q R : Type) :=
  (P : P)
  (Q : Q)
  (R : R)
  (PQ PR QR : Real)
  (S : Type)
  (PQ_ratio : PQ = 30)
  (PR_ratio : PR = 20)
  (QR_ratio : QR = 40)
  (PS PQ_division : Real)
  (S_divides_PQ : PS / PQ = 1 / 4)
  (T : Type)
  (PT PR_division : Real)
  (T_divides_PR : PT / PR = 1 / 3)
  (U : Type)
  (QU QR_division : Real)
  (U_divides_QR : QU / QR = 3 / 4)
  
noncomputable def intersection_length_of_angle_bisectors {P Q R S T U V : Type*}
  (triangle : Triangle P Q R)
  (H_S : triangle.PQ / (4 * triangle.S) = 30 / 4)
  (H_T : triangle.PQ / (3 * triangle.T) = 20 / 3)
  (H_U : triangle.PQ / (4 * triangle.U) = 40 / 4)
  (H_V : V → psr_angle_bisector_intersects_qsr_angle_bisector_at_V_within_triangle P Q R V S T U) :
  Real :=
  12

axiom psr_angle_bisector_intersects_qsr_angle_bisector_at_V_within_triangle
  {P Q R V S T U : Type*} : Prop

theorem find_length_SV
  (P Q R S T U V : Type*)
  (triangle : Triangle P Q R)
  (hS : triangle.PQ / (4 * triangle.S) = 30 / 4)
  (hT : triangle.PQ / (3 * triangle.T) = 20 / 3)
  (hU : triangle.PQ / (4 * triangle.U) = 40 / 4)
  (hV : psr_angle_bisector_intersects_qsr_angle_bisector_at_V_within_triangle P Q R V S T U) :
  intersection_length_of_angle_bisectors triangle hS hT hU hV = 12 :=
sorry

end find_length_SV_l483_483017


namespace max_min_values_area_closed_figure_l483_483320

def f (x : ℝ) : ℝ := (1 / 3) * x^3 - (1 / 2) * x^2 + 1

theorem max_min_values :
  (∀ x, x ≤ 0 ∨ x ≥ 1 → f x ≤ 1) ∧ (f 0 = 1) ∧ (f 1 = 5 / 6) := by
  sorry

theorem area_closed_figure :
  (∫ x in 0..(3 / 2), 1 - f x) = 9 / 64 := by
  sorry

end max_min_values_area_closed_figure_l483_483320


namespace expected_waiting_time_correct_l483_483229

noncomputable def combined_average_bites_per_5_minutes := 6
def average_waiting_time_for_first_bite_in_seconds : ℝ := 50

theorem expected_waiting_time_correct :
  (1 / combined_average_bites_per_5_minutes) * 300 = average_waiting_time_for_first_bite_in_seconds :=
by
  sorry

end expected_waiting_time_correct_l483_483229


namespace rancher_cows_l483_483561

theorem rancher_cows (H C : ℕ) (h1 : C = 5 * H) (h2 : C + H = 168) : C = 140 := by
  sorry

end rancher_cows_l483_483561


namespace lines_parallel_l483_483383

-- Definitions of the given conditions
variables (A B C L M P Q : Point)
variable (Ω Γ : Circle)

-- Assumptions based on the problem statement
variable (BL_is_angle_bisector : AngleBisector A B C L)
variable (M_on_CL : OnLineSegment M C L)
variable (P_on_CA : OnRay P C A)
variable (PB_tangent : Tangent PB Ω B)
variable (B_and_M_tangent : TangentsAtPoints B M Γ Q)

-- Statement of the problem to prove
theorem lines_parallel (h1 : BL_is_angle_bisector A B C L)
                      (h2 : M_on_CL M C L)
                      (h3 : P_on_CA P C A)
                      (h4 : PB_tangent PB Ω B)
                      (h5 : B_and_M_tangent B M Γ Q) :
                      Parallel PQ BL := sorry

end lines_parallel_l483_483383


namespace num_factors_2310_l483_483340

theorem num_factors_2310 : 
  let n := 2310
  let prime_factors := [(2, 1), (3, 1), (5, 1), (7, 1), (11, 1)]
  let distinct_factors := prime_factors.foldl (λ acc p, acc * (p.snd + 1)) 1
  distinct_factors = 32 := 
by
  -- The proof would generally be placed here
  sorry

end num_factors_2310_l483_483340


namespace smallest_arithmetic_mean_divisible_1111_l483_483848

theorem smallest_arithmetic_mean_divisible_1111 :
  ∃ n : ℕ, 93 ≤ n ∧ n + 4 = 97 ∧ (∀ i : ℕ, i ∈ finset.range 9 → (n + i) % 11 = 0 ∨ (n + i) % 101 = 0) :=
sorry

end smallest_arithmetic_mean_divisible_1111_l483_483848


namespace PAIB_cyclic_l483_483461

variables {A B C D I P : Point}
variables {alpha beta gamma delta : Real}

-- Definitions and Conditions
def incenter_of_quadrilateral (I : Point) (A B C D : Point) : Prop := 
  -- this represents that I is the incenter of quadrilateral ABCD
  -- you'll need a proper definition of what it means to be an incenter here
  sorry

def cyclic_quadrilateral (P A I B : Point) : Prop := 
  -- this represents that PAIB is a cyclic quadrilateral
  -- you'll need a proper definition of what it means to be cyclic here
  sorry

-- Given conditions as Lean definitions
axiom incenter_condition : incenter_of_quadrilateral I A B C D
axiom angle_IAP_PBI_supplementary : angle I A P + angle P B I = 180

-- The proof goal
theorem PAIB_cyclic : cyclic_quadrilateral P A I B :=
by { sorry }

end PAIB_cyclic_l483_483461


namespace equal_charge_at_250_l483_483566

/-- Define the monthly fee for Plan A --/
def planA_fee (x : ℕ) : ℝ :=
  0.4 * x + 50

/-- Define the monthly fee for Plan B --/
def planB_fee (x : ℕ) : ℝ :=
  0.6 * x

/-- Prove that the charges for Plan A and Plan B are equal when the call duration is 250 minutes --/
theorem equal_charge_at_250 : planA_fee 250 = planB_fee 250 :=
by
  sorry

end equal_charge_at_250_l483_483566


namespace max_intersections_l483_483602

theorem max_intersections 
  (X : Fin 8) (Y : Fin 6) :
  let segments := List.product (List.ofFn (fun _ => X)) (List.ofFn (fun _ => Y))
  let count_intersections := Nat.choose 8 2 * Nat.choose 6 2
  count_intersections = 420 :=
by
  sorry

end max_intersections_l483_483602


namespace problem_statement_l483_483164

structure StudentData where
  x : Fin 48 → ℝ -- x ranges from 1 to 48
  y : Fin 48 → ℝ -- heights for the 48 students
  z : Fin 48 → ℝ -- scores for the 48 students

def student_example : StudentData := {
  x := λ i, i.val + 1,
  y := λ i, [1.54, 1.56, 1.56, ..., 1.85, 1.85].get i.val sorry,
  z := λ i, [76, 65, 80, ..., 95, 80].get i.val sorry,
}

def is_function (A B : Type) (f : A → B) : Prop := ∀ a1 a2 : A, f a1 = f a2 → a1 = a2

theorem problem_statement (data : StudentData) :
  let n_true := (if is_function (Fin 48) ℝ data.y then 1 else 0)
                + (if is_function ℝ ℝ (λ h, data.z (Fin.mk h sorry)) then 1 else 0)
                + (if is_function ℝ (Fin 48) (λ s, data.x (Fin.mk s sorry)) then 1 else 0) in
  n_true = 1 := sorry

end problem_statement_l483_483164


namespace find_TU_squared_l483_483076

-- Defining square PQRS and its side length
variable (PQRS : Square)
variable (side_length : ℝ)
variable (side_length_eq : side_length = 15)

-- Points T and U and given distances
variable (T U : Point)
variable (QT RU PT SU : ℝ)
variable (QT_eq : QT = 7)
variable (RU_eq : RU = 7)
variable (PT_eq : PT = 10)
variable (SU_eq : SU = 10)

-- Lean 4 statement to prove
theorem find_TU_squared (PQRS_square : Square PQRS side_length_eq) :
  ∀ (QT RU PT SU : ℝ) (QT_eq : QT = 7) (RU_eq : RU = 7) (PT_eq : PT = 10) (SU_eq : SU = 10),
  TU^2 = 676 := by
  sorry

end find_TU_squared_l483_483076


namespace sale_in_fifth_month_l483_483169

theorem sale_in_fifth_month 
  (sale_month_1 : ℕ) (sale_month_2 : ℕ) (sale_month_3 : ℕ) (sale_month_4 : ℕ) 
  (sale_month_6 : ℕ) (average_sale : ℕ) 
  (h1 : sale_month_1 = 5266) (h2 : sale_month_2 = 5744) (h3 : sale_month_3 = 5864) 
  (h4 : sale_month_4 = 6122) (h6 : sale_month_6 = 4916) (h_avg : average_sale = 5750) :
  ∃ sale_month_5, sale_month_5 = 6588 :=
by
  sorry

end sale_in_fifth_month_l483_483169


namespace rpm_of_wheel_l483_483522

theorem rpm_of_wheel (radius : ℝ) (speed_kmh : ℝ) (π : ℝ) (speed_in_cm_min : ℝ) 
                      (circumference : ℝ) (rpm : ℝ) (h1 : radius = 140) 
                      (h2 : speed_kmh = 66) (h3 : π = 3.1416) 
                      (h4 : speed_in_cm_min = (speed_kmh * 100000) / 6) 
                      (h5 : circumference = 2 * π * radius) 
                      (h6 : rpm = speed_in_cm_min / circumference) : 
                      rpm ≈ 1250.14 :=
by
  sorry

end rpm_of_wheel_l483_483522


namespace f_increasing_l483_483457

def f (x : ℝ) : ℝ := Real.logBase (1/2) (1 - 2 * x)

theorem f_increasing: ∀ x₁ x₂ : ℝ, x₁ < x₂ → x₁ < (1/2) → x₂ < (1/2) → f(x₁) < f(x₂) :=
by
  sorry

end f_increasing_l483_483457


namespace area_FDBG_l483_483891

-- Given structures
structure Triangle :=
  (A B C : Type)
  (AB AC : ℝ)
  (area : ℝ)

structure Midpoint (P Q : Type) :=
  (M : Type)

-- Specific conditions
def ABC : Triangle := { A := ℝ, B := ℝ, C := ℝ, AB := 40, AC := 20, area := 160 }

def D : Midpoint ABC.ABC := { M := ℝ }
def E : Midpoint ABC.ABC := { M := ℝ }

def F : Type := ℝ
def G : Type := ℝ

-- Proof problem statement
theorem area_FDBG (ABC : Triangle) (D E : Midpoint ABC.ABC) (F G : Type) :
  ABC.AB = 40 → ABC.AC = 20 → ABC.area = 160 →
  ∃ (F : Type) (G : Type), 
    -- conditions about D, E, F, G and calculation steps are not included here
    -- directly, we state the final goal.
    area_FDBG = 106.6 :=
sorry

end area_FDBG_l483_483891


namespace sum_of_m_and_n_l483_483641

theorem sum_of_m_and_n (m n : ℝ) (h : m^2 + n^2 - 6 * m + 10 * n + 34 = 0) : m + n = -2 := 
sorry

end sum_of_m_and_n_l483_483641


namespace number_of_true_propositions_l483_483191

/-- Given four propositions regarding the properties of points and planes:
    1. If A ∈ α, B ∈ α, and C ∈ AB, then C ∈ α.
    2. If α ∩ β = l, b ⊆ α, c ⊆ β, and b ∩ c = A, then A ∈ l.
    3. If A, B, C ∈ α, A, B, C ∈ β and A, B, C are not collinear, then α coincides with β.
    4. Any four points that are not collinear with any three points must be coplanar.
    Prove that the number of true propositions is 3. -/
theorem number_of_true_propositions : 
  let prop1 := ∀ (α : set Point) (A B C : Point), A ∈ α ∧ B ∈ α ∧ C ∈ line_through A B → C ∈ α,
      prop2 := ∀ (α β : set Point) (l : set Point) (b c : set Point) (A : Point), α ∩ β = l ∧ b ⊆ α ∧ c ⊆ β ∧ b ∩ c = {A} → A ∈ l,
      prop3 := ∀ (α β : set Point) (A B C : Point), A ∈ α ∧ B ∈ α ∧ C ∈ α ∧ A ∈ β ∧ B ∈ β ∧ C ∈ β ∧ ¬ collinear {A, B, C} → α = β,
      prop4 := ∀ (points : set Point), ¬ collinear points ∧ ∃ (a b c d : Point), a ∈ points ∧ b ∈ points ∧ c ∈ points ∧ d ∈ points ∧ ¬ coplanar {a, b, c, d} → ¬ coplanar points in
  ([prop1, prop2, prop3, prop4].count true = 3) := 
sorry

end number_of_true_propositions_l483_483191


namespace smallest_three_digit_even_in_pascals_triangle_l483_483510

theorem smallest_three_digit_even_in_pascals_triangle : 
  (∃ n k : ℕ, n ≥ k ∧ binomial n k = 120) ∧
  (∀ m : ℕ, 100 ≤ m ∧ m < 120 → (∃ n k : ℕ, n ≥ k ∧ binomial n k = m) → ¬even m) :=
sorry

end smallest_three_digit_even_in_pascals_triangle_l483_483510


namespace acute_angle_at_10_50_l483_483902

def degrees_per_minute : ℕ := 6
def degrees_per_hour : ℕ := 30
def time_in_minutes (h m: ℕ) := h * 60 + m
def angle (h m: ℕ) : ℕ := 
  let minute_angle := m * degrees_per_minute
  let hour_angle := h * degrees_per_hour + (m * degrees_per_hour / 60)
  let raw_angle := abs (hour_angle - minute_angle)
  if raw_angle > 180 then 360 - raw_angle else raw_angle

theorem acute_angle_at_10_50 : angle 10 50 = 25 := 
by 
  sorry

end acute_angle_at_10_50_l483_483902


namespace distance_to_school_is_correct_l483_483885

-- Define the necessary constants, variables, and conditions
def distance_to_market : ℝ := 2
def total_weekly_mileage : ℝ := 44
def school_trip_miles (x : ℝ) : ℝ := 16 * x
def market_trip_miles : ℝ := 2 * distance_to_market
def total_trip_miles (x : ℝ) : ℝ := school_trip_miles x + market_trip_miles

-- Prove that the distance from Philip's house to the children's school is 2.5 miles
theorem distance_to_school_is_correct (x : ℝ) (h : total_trip_miles x = total_weekly_mileage) :
  x = 2.5 :=
by
  -- Insert necessary proof steps starting with the provided hypothesis
  sorry

end distance_to_school_is_correct_l483_483885


namespace simplify_cot_tan_l483_483469

theorem simplify_cot_tan :
  (Real.cot (Real.toRadians 20) + Real.tan (Real.toRadians 10) = Real.csc (Real.toRadians 20)) :=
by
  sorry

end simplify_cot_tan_l483_483469


namespace rhombus_area_l483_483520

theorem rhombus_area (d1 d2 : ℝ) (h_d1 : d1 = 25) (h_d2 : d2 = 50) :
  (d1 * d2) / 2 = 625 := 
by
  sorry

end rhombus_area_l483_483520


namespace constant_term_binomial_expansion_l483_483375

theorem constant_term_binomial_expansion :
  let n := 8
  let expr := (λ x : ℝ, (x^(1/3) - 2/x)^n)
  let fifth_term_is_greatest := true 
  ∃ const_term, const_term = 112 :=
sorry

end constant_term_binomial_expansion_l483_483375


namespace test_null_hypothesis_l483_483527

-- Definition of the problem conditions
def sample_size_1 : ℕ := 11
def sample_size_2 : ℕ := 14
def sample_variance_X : ℝ := 0.76
def sample_variance_Y : ℝ := 0.38
def alpha : ℝ := 0.05
def degrees_of_freedom_1 : ℕ := sample_size_1 - 1
def degrees_of_freedom_2 : ℕ := sample_size_2 - 1
def F_obs : ℝ := sample_variance_X / sample_variance_Y
def F_crit : ℝ := 2.67   -- From F-distribution table for df1 = 10, df2 = 13, alpha = 0.05

-- Theorem statement
theorem test_null_hypothesis : ¬ (F_obs > F_crit) := by
  unfold F_obs
  unfold F_crit
  linarith

end test_null_hypothesis_l483_483527


namespace ratio_HP_HA_l483_483393

-- Given Definitions
variables (A B C P Q H : Type)
variables (h1 : Triangle A B C) (h2 : AcuteTriangle A B C) (h3 : P ≠ Q)
variables (h4 : FootOfAltitudeFrom A H B C) (h5 : OnExtendedLine P A B) (h6 : OnExtendedLine Q A C)
variables (h7 : HP = HQ) (h8 : CyclicQuadrilateral B C P Q)

-- Required Ratio
theorem ratio_HP_HA : HP = HA := sorry

end ratio_HP_HA_l483_483393


namespace part1_part2_l483_483696

open Real

def f (x a : ℝ) : ℝ :=
  x^2 + a * x + 3

theorem part1 (x : ℝ) (h : x^2 - 4 * x + 3 < 0) :
  1 < x ∧ x < 3 :=
  sorry

theorem part2 (a : ℝ) (h : ∀ x, f x a > 0) :
  -2 * sqrt 3 < a ∧ a < 2 * sqrt 3 :=
  sorry

end part1_part2_l483_483696


namespace sin_cos_sum_l483_483660

theorem sin_cos_sum (θ : ℝ) (h : tan θ + (1 / tan θ) = 2) : sin θ + cos θ = √2 ∨ sin θ + cos θ = -√2 := 
  sorry

end sin_cos_sum_l483_483660


namespace inequality_proof_l483_483785

theorem inequality_proof (a b c d e f : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) (hf : 0 < f) :
  real.cbrt (abc / (a + b + d)) + real.cbrt (def / (c + e + f)) < real.cbrt ((a + b + d) * (c + e + f)) :=
sorry

end inequality_proof_l483_483785


namespace recurring_product_is_14_over_41_l483_483906

-- Conditions translating into Lean
def recurring_63_to_frac : ℚ := 63 / 99
def recurring_54_to_frac : ℚ := 54 / 99

theorem recurring_product_is_14_over_41 :
  recurring_63_to_frac * recurring_54_to_frac = 14 / 41 := 
by
  sorry

end recurring_product_is_14_over_41_l483_483906


namespace tangent_line_at_origin_l483_483757

theorem tangent_line_at_origin (a : ℝ) (f : ℝ → ℝ) (h1 : f = λ x, x^3 + a * x^2 + (a - 3) * x) 
    (h2 : ∀ x : ℝ, (fderiv ℝ f) (-x) = (fderiv ℝ f) x) : (fderiv ℝ f) 0 = -3 :=
by
  sorry

end tangent_line_at_origin_l483_483757


namespace range_of_a_l483_483656

theorem range_of_a (a : ℝ) :
  ¬ (∃ x : ℝ, (0 < x) ∧ (x + 1/x < a)) ↔ a ≤ 2 :=
by {
  sorry
}

end range_of_a_l483_483656


namespace cost_per_candy_bar_l483_483400

-- Define the conditions as hypotheses
variables (candy_bars_total : ℕ) (candy_bars_paid_by_dave : ℕ) (amount_paid_by_john : ℝ)
-- Assume the given values
axiom total_candy_bars : candy_bars_total = 20
axiom candy_bars_by_dave : candy_bars_paid_by_dave = 6
axiom paid_by_john : amount_paid_by_john = 21

-- Define the proof problem
theorem cost_per_candy_bar :
  (amount_paid_by_john / (candy_bars_total - candy_bars_paid_by_dave) = 1.50) :=
by
  sorry

end cost_per_candy_bar_l483_483400


namespace parabola_vertex_sum_l483_483598

theorem parabola_vertex_sum (p q r : ℝ) :
  (∃ (y : ℝ → ℝ), y = (λ x, px^2 + qx + r) ∧
  (∃ (vertex_x vertex_y : ℝ), vertex_x = -3 ∧ vertex_y = 7 ∧
  ∃ (symmetry : ℝ), symmetry = vertex_x ∧
  y (-6) = 4)) →
  p + q + r = 7 / 3 :=
by
  intros h
  sorry

end parabola_vertex_sum_l483_483598


namespace part1_part2_l483_483725

open Real

-- Declare the basic definitions and conditions used
variables {a b c : ℝ}
variables {A B C : ℝ}

-- Part 1: Prove that b = sqrt(2) given a = 2, c = sqrt(2), and C = π/4.
theorem part1 (a_val : a = 2) (c_val : c = sqrt 2) (C_val : C = π / 4) : b = sqrt 2 := by
  sorry

-- Part 2: Prove that the maximum value of the area of ΔABC is 1 given a = 2 and b + c = 2sqrt(2).
theorem part2 (a_val : a = 2) (b_c_sum : b + c = 2 * sqrt 2) : 
  let S := sqrt (b * c - 1) in 
  S ≤ 1 := by
  sorry

end part1_part2_l483_483725


namespace smallest_tan_B_l483_483013

def isRightTriangle (A B C : Type) [Add A] : Prop :=
∃ (AB BC AC : ℝ),
  AB = 24 ∧ BC = 10 ∧ AB^2 + BC^2 = AC^2

theorem smallest_tan_B :
  ∀ (A B C : Type) [Add A],  
    isRightTriangle A B C → 
    ∃ tanB : ℝ, tanB = (Real.sqrt 119) / 5 :=
by
  intros
  apply exists.intro
  sorry

end smallest_tan_B_l483_483013


namespace least_whole_number_subtracted_l483_483521

theorem least_whole_number_subtracted {x : ℕ} (h : 6 > x ∧ 7 > x) :
  (6 - x) / (7 - x : ℝ) < 16 / 21 -> x = 3 :=
by
  intros
  sorry

end least_whole_number_subtracted_l483_483521


namespace smallest_positive_four_digit_equivalent_to_5_mod_8_l483_483912

theorem smallest_positive_four_digit_equivalent_to_5_mod_8 : 
  ∃ (n : ℕ), n ≥ 1000 ∧ n % 8 = 5 ∧ n = 1005 :=
by
  sorry

end smallest_positive_four_digit_equivalent_to_5_mod_8_l483_483912


namespace find_B_l483_483129

theorem find_B (A B : Nat) (cond1 : 6 * 10 + 5 + B * 100 + 3 = 748)
  (cond2 : ∀ A, A ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
  (cond3 : 5 + 3 = 8)
  (cond4 : 6 * A + B = 7) : B = 1 := by
  sorry

end find_B_l483_483129


namespace complex_conjugate_solution_l483_483288

variable {Z : ℂ}

theorem complex_conjugate_solution (h : Z / (1 + complex.I) = complex.I) : conj Z = -1 - complex.I :=
  sorry

end complex_conjugate_solution_l483_483288


namespace matrix_multiplication_correct_l483_483588

variable (A B : Matrix (Fin 2) (Fin 2) ℤ)
variable (A_eq : A = ![![4, -2], ![-1, 5]])
variable (B_eq : B = ![![0, 3], ![2, -2]])

theorem matrix_multiplication_correct :
  A ⬝ B = ![![(-4 : ℤ), 16], ![10, -13]] :=
by
  sorry

end matrix_multiplication_correct_l483_483588


namespace pyramid_volume_is_192_l483_483776

noncomputable def volume_of_pyramid 
  (P Q R S : Type) 
  [InnerProductSpace ℝ P] [InnerProductSpace ℝ Q] 
  [InnerProductSpace ℝ R] [InnerProductSpace ℝ S]
  (SP SQ : ℝ) 
  (SR PQ PR : ℝ) 
  (h_perpendicular_1 : InnerProductSpace.norm (SP) = 12) 
  (h_perpendicular_2 : InnerProductSpace.norm (SQ) = 12)
  (h_perpendicular_3 : InnerProductSpace.norm (SR) = 8)
  (h_v1 : ∀ (u : S) (v : P), orthogonal u v)
  (h_v2 : ∀ (u : S) (v : Q), orthogonal u v)
  (h_v3 : ∀ (u : S) (v : R), orthogonal u v) :
  ℝ := 
begin
  -- The volume of the pyramid SPQR
  let base_area := (1 / 2) * PQ * PR,
  let height := SR,
  let volume := (1 / 3) * base_area * height,
  volume
end

theorem pyramid_volume_is_192 : 
  volume_of_pyramid P Q R S 
                    (12) (12) (8) (12) (12) 
                    sorry sorry sorry sorry sorry sorry = 192 :=
sorry

end pyramid_volume_is_192_l483_483776


namespace solution_l483_483694

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  exp x * (a * x + b) - x^2 - 4 * x

def tangent_point (f : ℝ → ℝ) :=
  (0, f 0)

def tangent_line (x : ℝ) : ℝ :=
  4 * x + 4

def deriv_f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  exp x * (a * x + a + b) - 2 * x - 4

theorem solution : 
  ∃ (a b : ℝ), 
  a = 4 ∧ b = 4 ∧ 
  ∀ (x : ℝ),
  (f 0 a b = 4 ∧ deriv_f 0 a b = 4) ∧
  (f x a b = 4 * exp x * (x + 1) - x^2 - 4 * x) ∧ 
  (
    (∃ (L : ℝ), L = -ln 2 ∧ 
      ((∀ (x : ℝ), x ∈ set.Ico (-∞ : ℝ) (-2 : ℝ) → deriv_f x a b > 0) ∧ 
      (∀ (x : ℝ), x ∈ set.Icc (-2) (-ln 2) → deriv_f x a b < 0) ∧ 
      (∀ (x : ℝ), x ∈ set.Ioi (-ln 2) → deriv_f x a b > 0)) ∧
      f (-2) a b = 4 * (1 - exp (-2))
    )
  ) :=
by
  sorry

end solution_l483_483694


namespace product_xy_zero_l483_483711

theorem product_xy_zero (x y : ℝ) (h1 : 5^x / 2^(x + y) = 25) (h2 : 4^(x + y) / 2^(5 * y) = 16) : x * y = 0 :=
by
  -- h1: 5^x / 2^(x + y) = 25
  -- h2: 4^(x + y) / 2^(5*y) = 16
  sorry

end product_xy_zero_l483_483711


namespace bobs_password_probability_l483_483578

theorem bobs_password_probability :
  let num_digits := 10 in
  let odd_digit_count := 5 in
  let positive_digit_count := 9 in
  let first_digit_odd_prob := odd_digit_count / num_digits in
  let two_letters_prob := 1 in
  let last_digit_positive_prob := positive_digit_count / num_digits in
  (first_digit_odd_prob * two_letters_prob * last_digit_positive_prob) = 9 / 20 :=
by
  let num_digits := 10
  let odd_digit_count := 5
  let positive_digit_count := 9
  let first_digit_odd_prob := (odd_digit_count : ℝ) / (num_digits : ℝ)
  let two_letters_prob := 1
  let last_digit_positive_prob := (positive_digit_count : ℝ) / (num_digits : ℝ)
  have first_digit_odd_prob_eq : first_digit_odd_prob = 1 / 2 :=
    by norm_num [first_digit_odd_prob, odd_digit_count, num_digits]
  have last_digit_positive_prob_eq : last_digit_positive_prob = 9 / 10 :=
    by norm_num [last_digit_positive_prob, positive_digit_count, num_digits]
  rw [first_digit_odd_prob_eq, last_digit_positive_prob_eq]
  calc
    (1 / 2) * 1 * (9 / 10) = (1 / 2) * (9 / 10) : by ring
                    ... = 9 / 20 : by ring


end bobs_password_probability_l483_483578


namespace bandits_gem_division_proof_l483_483942

theorem bandits_gem_division_proof (bandits : Fin 102 → ℕ × ℕ × ℕ)
  (h : ∀ b : Fin 102, let (r, s, e) := bandits b in r + s + e = 100) :
  (∃ b1 b2 : Fin 102, b1 ≠ b2 ∧ bandits b1 = bandits b2) ∨ 
  (∃ b1 b2 : Fin 102, bandits b1 ≠ bandits b2) :=
by
  sorry

end bandits_gem_division_proof_l483_483942


namespace min_dot_product_l483_483699

variables {a b : ℝ^2} -- Defining a and b as real planar vectors

theorem min_dot_product (h : ∥2 • a - b∥ ≤ 3) : ∃ (c : ℝ), c = -9/8 ∧ ∀ (a b : ℝ^2), ∥2 • a - b∥ ≤ 3 → a ⬝ b ≥ c :=
by {
  sorry
}

end min_dot_product_l483_483699


namespace smallest_arithmetic_mean_l483_483875

noncomputable def S (n : ℕ) := (List.range' n 9).map Nat.ofNat

theorem smallest_arithmetic_mean (n : ℕ) (h1 : 93 ≤ n) (h2 : ∃ k ∈ S n, 11 ∣ k) (h3 : ∃ k ∈ S n, 101 ∣ k) : 
  (n + 4 = 97) := by
  sorry

end smallest_arithmetic_mean_l483_483875


namespace parallelogram_angle_ratio_l483_483002

theorem parallelogram_angle_ratio (ABCD : Type*)
  [parallelogram ABCD] 
  (O : point) 
  (intersects_diag : diagonal_intersect_at ABCD O) 
  (alpha : angle) 
  (h1 : angle_AOB * 3 = angle_DBA)
  (h2 : angle_CAB * 3 = angle_DBC)
  (h3 : angle_ACB = r * angle_AOB) :
  r = 2 :=
by
  sorry

end parallelogram_angle_ratio_l483_483002


namespace increasing_interval_l483_483808

noncomputable def f : ℝ → ℝ := λ x, -x^2 + 2 * x - 2

theorem increasing_interval :
  ∀ x, ∀ y, (x < y ∧ y < 1) → f(x) ≤ f(y) :=
by
  sorry

end increasing_interval_l483_483808


namespace car_speed_is_80_l483_483159

theorem car_speed_is_80 : ∃ v : ℝ, (1 / v * 3600 = 45) ∧ (v = 80) :=
by
  sorry

end car_speed_is_80_l483_483159


namespace largest_value_is_E_l483_483931

theorem largest_value_is_E :
  let A := 3 + 1 + 2 + 9
  let B := 3 * 1 + 2 + 9
  let C := 3 + 1 * 2 + 9
  let D := 3 + 1 + 2 * 9
  let E := 3 * 1 * 2 * 9
  E > A ∧ E > B ∧ E > C ∧ E > D := 
by
  let A := 3 + 1 + 2 + 9
  let B := 3 * 1 + 2 + 9
  let C := 3 + 1 * 2 + 9
  let D := 3 + 1 + 2 * 9
  let E := 3 * 1 * 2 * 9
  sorry

end largest_value_is_E_l483_483931


namespace total_time_in_range_l483_483144

-- Definitions for the problem conditions
def section1 := 240 -- km
def section2 := 300 -- km
def section3 := 400 -- km

def speed1 := 40 -- km/h
def speed2 := 75 -- km/h
def speed3 := 80 -- km/h

-- The time it takes to cover a section at a certain speed
def time (distance : ℕ) (speed : ℕ) : ℕ := distance / speed

-- Total time to cover all sections with different speed assignments
def total_time (s1 s2 s3 v1 v2 v3 : ℕ) : ℕ :=
  time s1 v1 + time s2 v2 + time s3 v3

-- Prove that the total time is within the range [15, 17]
theorem total_time_in_range :
  (total_time section1 section2 section3 speed3 speed2 speed1 = 15) ∧
  (total_time section1 section2 section3 speed1 speed2 speed3 = 17) →
  ∃ (T : ℕ), 15 ≤ T ∧ T ≤ 17 :=
by
  intro h
  sorry

end total_time_in_range_l483_483144


namespace sine_angle_A_l483_483371

theorem sine_angle_A (A B C : Type) [HasAngle A B C] [RightTriangle A B C]
  (hC : ∠C = 90) (hAC : AC = 1) (hBC : BC = 3) : sin ∠A = 3 * sqrt 10 / 10 := by
  sorry

end sine_angle_A_l483_483371


namespace find_f2011_l483_483411

noncomputable def f : ℝ → ℝ :=
  sorry

axiom cond1 {x : ℝ} : f(x + 4) ≤ f(x) + 4
axiom cond2 {x : ℝ} : f(x + 2) ≥ f(x) + 2
axiom initial : f(1) = 0

theorem find_f2011 : f(2011) = 2010 :=
  sorry

end find_f2011_l483_483411


namespace find_e_l483_483050

noncomputable def f (x : ℝ) (c : ℝ) := 5 * x + 2 * c

noncomputable def g (x : ℝ) (c : ℝ) := c * x^2 + 3

noncomputable def fg (x : ℝ) (c : ℝ) := f (g x c) c

theorem find_e (c : ℝ) (e : ℝ) (h1 : f (g x c) c = 15 * x^2 + e) (h2 : 5 * c = 15) : e = 21 :=
by
  sorry

end find_e_l483_483050


namespace decreasing_function_a_range_l483_483693

theorem decreasing_function_a_range (a : ℝ) (f : ℝ → ℝ) :
  (∀ x y ∈ Ioo 0 1, x < y → f x > f y) →
  f = (λ x, log a (2 - a * x^2)) →
  (1 < a ∧ a ≤ 2) :=
by
  sorry

end decreasing_function_a_range_l483_483693


namespace Clever_not_Green_l483_483396

variables {Lizard : Type}
variables [DecidableEq Lizard] (Clever Green CanJump CanSwim : Lizard → Prop)

theorem Clever_not_Green (h1 : ∀ x, Clever x → CanJump x)
                        (h2 : ∀ x, Green x → ¬ CanSwim x)
                        (h3 : ∀ x, ¬ CanSwim x → ¬ CanJump x) :
  ∀ x, Clever x → ¬ Green x :=
by
  intro x hClever hGreen
  apply h3 x
  apply h2 x hGreen
  exact h1 x hClever

end Clever_not_Green_l483_483396


namespace valentines_cards_count_l483_483771

theorem valentines_cards_count (x y : ℕ) (h1 : x * y = x + y + 30) : x * y = 64 :=
by {
    sorry
}

end valentines_cards_count_l483_483771


namespace sum_of_first_fifty_terms_l483_483763

variable {a b : ℕ → ℝ}

axiom a_1 : a 1 = 15
axiom b_1 : b 1 = 45
axiom a_50_b_50 : a 50 + b 50 = 150

def sequence_sum (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, (a (i + 1) + b (i + 1))

theorem sum_of_first_fifty_terms : sequence_sum 50 = 5250 :=
by
  sorry

end sum_of_first_fifty_terms_l483_483763


namespace number_of_children_l483_483022

-- Definitions based on conditions
def numDogs : ℕ := 2
def numCats : ℕ := 1
def numLegsTotal : ℕ := 22
def numLegsDog : ℕ := 4
def numLegsCat : ℕ := 4
def numLegsHuman : ℕ := 2

-- Main theorem proving the number of children
theorem number_of_children :
  let totalPetLegs := (numDogs * numLegsDog) + (numCats * numLegsCat)
  let totalLegsAccounted := totalPetLegs + numLegsHuman
  let numLegsRemaining := numLegsTotal - totalLegsAccounted
  let numChildren := numLegsRemaining / numLegsHuman
  numChildren = 4 :=
by
  let totalPetLegs := (numDogs * numLegsDog) + (numCats * numLegsCat)
  let totalLegsAccounted := totalPetLegs + numLegsHuman
  let numLegsRemaining := numLegsTotal - totalLegsAccounted
  let numChildren := numLegsRemaining / numLegsHuman
  exact sorry

end number_of_children_l483_483022


namespace trajectory_of_P_l483_483702

open Real

def dist (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

definition point := ℝ × ℝ

theorem trajectory_of_P (F1 F2 P : point) (a : ℕ) :
  F1 = (-5, 0) → F2 = (5, 0) →
  (dist P F1 - dist P F2 = 2 * a) →
  (a = 3 → ∃ branch_of_hyperbola : set point, trajectory P branch_of_hyperbola) ∧
  (a = 5 → ∃ ray : set point, trajectory P ray) := 
by
  sorry

end trajectory_of_P_l483_483702


namespace sum_c_eq_S_n_l483_483334

def a : ℕ → ℕ
| 0     := 2
| (n+1) := 2 * (a n) + 2^(n+1)

def b (n : ℕ) : ℕ := a n / 2^n

def c (n : ℕ) : ℕ := a n - 1 / (b n * b (n + 1))

def S (n : ℕ) : ℕ := (n - 1) * 2^(n + 1) + 2 + 1 / (n + 1)

theorem sum_c_eq_S_n (n : ℕ) : (finset.range n).sum c = S n :=
sorry

end sum_c_eq_S_n_l483_483334


namespace slope_AB_is_one_l483_483299

def point := (ℝ × ℝ)

def slope (p1 p2: point): ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)

-- Given points A and B
def A : point := (1, 1)
def B : point := (2, 2)

-- To prove the slope is 1
theorem slope_AB_is_one (A B: point) (hA : A = (1, 1)) (hB : B = (2, 2)) : slope A B = 1 :=
by 
    rw [hA, hB]
    unfold slope
    norm_num
    sorry

end slope_AB_is_one_l483_483299


namespace roots_in_interval_l483_483322

noncomputable def f (a x : ℝ) : ℝ := 
  2 * Real.log x - x^2 + a

theorem roots_in_interval (a : ℝ) :
  (∃ x1 x2 ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1), x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0) ↔ 
  1 < a ∧ a ≤ 2 + 1 / Real.exp 2 :=
by
  sorry

end roots_in_interval_l483_483322


namespace proof_B_proof_C_l483_483130

-- Definition for problem B
def satisfies_condition_B (x : ℝ) : Prop :=
  x < 1/3

def func_B (x : ℝ) : ℝ :=
  3 * x + (1 / (3 * x - 1))

-- Definition for problem C
def satisfies_conditions_C (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x + y + x * y = 3

def func_C (x y : ℝ) : ℝ :=
  x * y

-- Proof statements
theorem proof_B (x : ℝ) (h : satisfies_condition_B x) : ∀ y, func_B x ≤ -1 :=
  sorry

theorem proof_C (x y : ℝ) (h : satisfies_conditions_C x y) : func_C x y ≤ 1 :=
  sorry

end proof_B_proof_C_l483_483130


namespace smallest_arithmetic_mean_divisible_1111_l483_483843

theorem smallest_arithmetic_mean_divisible_1111 :
  ∃ n : ℕ, 93 ≤ n ∧ n + 4 = 97 ∧ (∀ i : ℕ, i ∈ finset.range 9 → (n + i) % 11 = 0 ∨ (n + i) % 101 = 0) :=
sorry

end smallest_arithmetic_mean_divisible_1111_l483_483843


namespace opposite_of_2023_l483_483487

def opposite (n : Int) : Int := -n

theorem opposite_of_2023 : opposite 2023 = -2023 := by
  sorry

end opposite_of_2023_l483_483487


namespace clock_rings_l483_483631

theorem clock_rings (h1 : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 12 → n * 2 - 1 < 12) 
                    (h2 : ∀ m : ℕ, 1 ≤ m ∧ m ≤ 3 → m * 4 - 3 ≤ 12): 
  (nat.find (λ k, (h1 k ∨ h2 k)) = 10) :=
by 
  sorry

end clock_rings_l483_483631


namespace area_of_overlap_l483_483981

theorem area_of_overlap 
  (len1 len2 : ℕ) (area_left only_left_area : ℚ) (area_right only_right_area : ℚ) (w : ℚ)
  (h_len1 : len1 = 9) (h_len2 : len2 = 7) (h_only_left_area : only_left_area = 27) 
  (h_only_right_area : only_right_area = 18) (h_w : w > 0)
  (h_area_left : area_left = only_left_area + (w * 1))
  (h_area_right : area_right = only_right_area + (w * 1))
  (h_ratio : (w * len1) / (w * len2) = 9 / 7) : 
  (13.5) :=
by
  sorry

end area_of_overlap_l483_483981


namespace magnitude_of_c_l483_483335

noncomputable def a : ℝ × ℝ := (1, real.sqrt 3)
noncomputable def angle_ac_pi : ℝ := real.pi / 3
noncomputable def dot_product_ac : ℝ := 2

theorem magnitude_of_c
  (a : ℝ × ℝ)
  (angle_ac : ℝ)
  (dot_product_ac : ℝ) :
  real.sqrt (a.1^2 + a.2^2) = 2 →
  real.cos angle_ac = 0.5 →
  dot_product_ac = 2 →
  ∃ c : ℝ × ℝ, real.sqrt (c.1^2 + c.2^2) = 2 :=
by
  sorry

end magnitude_of_c_l483_483335


namespace students_at_end_l483_483000

def initial_students : ℝ := 42.0
def students_left : ℝ := 4.0
def students_transferred : ℝ := 10.0

theorem students_at_end : initial_students - students_left - students_transferred = 28.0 :=
by
  -- Proof omitted
  sorry

end students_at_end_l483_483000


namespace expected_waiting_time_l483_483224

/-- Consider a 5-minute interval. There are 5 bites on the first rod 
and 1 bite on the second rod in this interval. Therefore, the total average 
number of bites on both rods during these 5 minutes is 6. The expected waiting 
time for the first bite is 50 seconds. -/
theorem expected_waiting_time
    (average_bites_first_rod : ℝ)
    (average_bites_second_rod : ℝ)
    (total_interval_minutes : ℝ)
    (expected_waiting_time_seconds : ℝ) :
    average_bites_first_rod = 5 ∧
    average_bites_second_rod = 1 ∧
    total_interval_minutes = 5 →
    expected_waiting_time_seconds = 50 :=
by
  sorry

end expected_waiting_time_l483_483224


namespace sum_of_tens_and_ones_digits_of_7_pow_15_l483_483921

-- Definition for checking cyclical patterns of the digits
def cyclic_ones_7 (n : ℕ) : ℕ :=
  let ones_cycle := [7, 9, 3, 1]
  ones_cycle[(n % 4).toNat]

def cyclic_tens_7 (n : ℕ) : ℕ :=
  let tens_cycle := [0, 4, 4, 0]
  tens_cycle[(n % 4).toNat]

-- Main theorem statement
theorem sum_of_tens_and_ones_digits_of_7_pow_15 :
  cyclic_ones_7 15 + cyclic_tens_7 15 = 7 := by
  sorry

end sum_of_tens_and_ones_digits_of_7_pow_15_l483_483921


namespace exists_positive_integer_n_l483_483609

theorem exists_positive_integer_n :
  ∃ n : ℕ, (24 ∣ n) ∧ (9 < real.cbrt n) ∧ (real.cbrt n < 9.5) ∧ (n = 744) :=
by {
  use 744,
  split,
  { exact dvd.intro 31 rfl },
  split,
  { norm_num, },
  split,
  { norm_num, },
  { refl }
}

end exists_positive_integer_n_l483_483609


namespace solve_inequality_l483_483087

theorem solve_inequality:
  { x : ℝ | 
    (1 < x ∧ x < 2) 
    ∨ (x > Real.sqrt 10) 
  } = { x : ℝ | f x > 2 } 
  where
    f (x : ℝ) : ℝ :=
    if x < 2 then 2 * Real.exp (x - 1)
    else Real.log (x^2 - 1) / Real.log 3 := 
sorry

end solve_inequality_l483_483087


namespace value_of_x3_l483_483895

theorem value_of_x3 (
  x1 : ℝ,
  x2 : ℝ,
  x3 : ℝ,
  h_interval : 2 ≤ x1 ∧ x1 ≤ 4 ∧ 2 ≤ x2 ∧ x2 ≤ 4,
  h_better : x1 > x2,
  h_x1 : x1 = 2 + 0.618 * (4 - 2),
  h_x2 : x2 = 2 + (4 - x1)
) : x3 = 4 - 0.618 * (4 - x2) := by
  sorry

end value_of_x3_l483_483895


namespace triangle_perimeter_l483_483654

-- Define the triangle with sides a, b, c
structure Triangle :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

-- Define the predicate that checks if the triangle is isosceles
def isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.a = t.c ∨ t.b = t.c

-- Define the predicate that calculates the perimeter of the triangle
def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

-- State the problem
theorem triangle_perimeter : 
  ∃ (t : Triangle), isIsosceles t ∧ (    (t.a = 6 ∧ t.b = 9 ∧ perimeter t = 24)
                                       ∨ (t.b = 6 ∧ t.a = 9 ∧ perimeter t = 24)
                                       ∨ (t.c = 6 ∧ t.a = 9 ∧ perimeter t = 21)
                                       ∨ (t.a = 6 ∧ t.c = 9 ∧ perimeter t = 21)
                                       ∨ (t.b = 6 ∧ t.c = 9 ∧ perimeter t = 24)
                                       ∨ (t.c = 6 ∧ t.b = 9 ∧ perimeter t = 21)
                                    ) :=
sorry

end triangle_perimeter_l483_483654


namespace min_arithmetic_mean_of_consecutive_naturals_div_by_1111_l483_483859

theorem min_arithmetic_mean_of_consecutive_naturals_div_by_1111 :
  ∀ a : ℕ, (∃ k, (a * (a + 1) * (a + 2) * (a + 3) * (a + 4) * (a + 5) * (a + 6) * (a + 7) * (a + 8)) = 1111 * k) →
  (a + 4 ≥ 97) :=
sorry

end min_arithmetic_mean_of_consecutive_naturals_div_by_1111_l483_483859


namespace division_by_n_minus_1_squared_l483_483789

theorem division_by_n_minus_1_squared (n : ℕ) (h : n > 2) : (n ^ (n - 1) - 1) % ((n - 1) ^ 2) = 0 :=
sorry

end division_by_n_minus_1_squared_l483_483789


namespace simplify_cot20_tan10_l483_483465

theorem simplify_cot20_tan10 :
  (Real.cot 20 + Real.tan 10 = Real.csc 20) :=
sorry

end simplify_cot20_tan10_l483_483465


namespace smurf_team_count_l483_483152

def is_valid_team (team : Finset ℕ) : Prop :=
  ∀ smurf ∈ team, ∀ neighbor ∈ {smurf - 1, smurf + 1}, neighbor ∉ team

def count_valid_teams (n k : ℕ) : ℕ := 
  (Finset.range n).powerset.filter (λ s, s.card = k ∧ is_valid_team s).card

theorem smurf_team_count : count_valid_teams 12 5 = 36 := 
  by trivial sorry

end smurf_team_count_l483_483152


namespace solve_inequality_prove_inequality_l483_483530

open Real

-- Problem 1: Solve the inequality
theorem solve_inequality (x : ℝ) : (x - 1) / (2 * x + 1) ≤ 0 ↔ (-1 / 2) < x ∧ x ≤ 1 :=
sorry

-- Problem 2: Prove the inequality given positive a, b, and c
theorem prove_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a + b + c) * (1 / a + 1 / (b + c)) ≥ 4 :=
sorry

end solve_inequality_prove_inequality_l483_483530


namespace matrix_eigenvalue_problem_l483_483646

theorem matrix_eigenvalue_problem
  (a k : ℝ)
  (h1 : k ≠ 0)
  (h2 : ∃ λ : ℝ, (λ * k = a * k - k ∧ λ = 1))
  (h3 : (∃ (B : matrix (fin 2) (fin 2) ℝ), B * A = 1) 
        ∧ ∃ (v : vector ℝ 2), B = ![![1, -k], ![0, 1]] 
        ∧ v = ![3, 1] 
        ∧ B ⬝ v = ![1, 1]) :
  a + k = 3 :=
sorry

end matrix_eigenvalue_problem_l483_483646


namespace value_of_a_l483_483148

theorem value_of_a (a : ℝ) : 
  let A := { -1, 0, 1 }
  let B := { a + 1, 2 * a }
  A ∩ B = { 0 } → a = -1 :=
sorry

end value_of_a_l483_483148


namespace smallest_arithmetic_mean_divisible_product_l483_483841

theorem smallest_arithmetic_mean_divisible_product :
  ∃ (n : ℕ), (∀ i : ℕ, i < 9 → Nat.gcd ((n + i) * 1111) 1111 = 1111) ∧ (n + 4 = 97) := 
  sorry

end smallest_arithmetic_mean_divisible_product_l483_483841


namespace green_duck_percentage_is_correct_l483_483361

def smaller_pond_ducks : ℕ := 30
def larger_pond_ducks : ℕ := 50
def smaller_pond_green_percentage : ℚ := 0.20
def larger_pond_green_percentage : ℚ := 0.12

def smaller_pond_green_ducks : ℕ := (smaller_pond_green_percentage * smaller_pond_ducks).natValue
def larger_pond_green_ducks : ℕ := (larger_pond_green_percentage * larger_pond_ducks).natValue

def total_ducks : ℕ := smaller_pond_ducks + larger_pond_ducks
def total_green_ducks : ℕ := smaller_pond_green_ducks + larger_pond_green_ducks

def green_duck_percentage : ℚ := (total_green_ducks : ℚ) / (total_ducks : ℚ) * 100

theorem green_duck_percentage_is_correct : green_duck_percentage = 15 := by
  sorry

end green_duck_percentage_is_correct_l483_483361


namespace proof_problem1_proof_problem2_l483_483211

noncomputable def problem1 : ℝ := real.sqrt 16 + 2 * real.sqrt 9 - real.cbrt 27
noncomputable def problem2 : ℝ := abs (1 - real.sqrt 2) + real.sqrt 4 - real.cbrt (-8)

theorem proof_problem1 : problem1 = 7 := by
  sorry

theorem proof_problem2 : problem2 = real.sqrt 2 + 3 := by
  sorry

end proof_problem1_proof_problem2_l483_483211


namespace pentagon_angles_obtuse_l483_483166

theorem pentagon_angles_obtuse (P : Type*) [has_interior_angle P] [has_equal_sides P] 
  [convex_pentagon P] (H1 : ∀ (a : angle P), a < 120) :
  ∀ (a : angle P), a > 90 :=
sorry

end pentagon_angles_obtuse_l483_483166


namespace continuous_of_compact_and_connected_image_l483_483031

open Set Filter Topology

variable {n m : ℕ} (f : ℝ^n → ℝ^m)

theorem continuous_of_compact_and_connected_image (h1 : ∀ K : Set ℝ^n, IsCompact K → IsCompact (f '' K))
  (h2 : ∀ C : Set ℝ^n, IsConnected C → IsConnected (f '' C)) :
  Continuous f :=
begin
  sorry
end

end continuous_of_compact_and_connected_image_l483_483031


namespace tan_2x_parallel_f_monotonically_increasing_l483_483337

-- Definitions of vectors
def a (x : ℝ) : ℝ × ℝ := (sqrt 3 * sin x, 1)
def b (x : ℝ) : ℝ × ℝ := (cos x, 2)

-- Proof Problem 1:
theorem tan_2x_parallel (x : ℝ) (h : (∃ k : ℝ, a(x) = k • b(x))) : tan (2 * x) = 4 * sqrt 3 / 11 := by
  sorry

-- Definition of f(x) as in the condition
def f (x : ℝ) : ℝ := let a_vec := a x; let b_vec := b x in (a_vec.1 - b_vec.1) * b_vec.1 + (a_vec.2 - b_vec.2) * b_vec.2

-- Proof Problem 2:
theorem f_monotonically_increasing (k : ℤ) (x : ℝ) 
  (hx : -π/6 + ↑k*π ≤ x ∧ x ≤ π/3 + ↑k*π ) : 
  ∀ y, f y = f x :=
  sorry

end tan_2x_parallel_f_monotonically_increasing_l483_483337


namespace equiv_proof_l483_483482

-- Definitions
def u (x : ℝ) : ℝ := b * x
def v (x : ℝ) : ℝ := a * (x + 4) * (x - 1)

-- Conditions
variables (a b : ℝ)
variable h_asymptote : ∀ x, (x = -4 ∨ x = 1) → v x = 0
variable h_point : u 4 / v 4 = 2

-- Statement to prove
theorem equiv_proof : u 0 / v 0 = 0 := 
by
  sorry

end equiv_proof_l483_483482


namespace syrup_boxes_l483_483962

theorem syrup_boxes (soda_per_week : ℕ) (cost_per_box : ℕ) (total_cost_per_week : ℕ)
    (h_soda : soda_per_week = 180) (h_cost_box : cost_per_box = 40)
    (h_total_cost : total_cost_per_week = 240) :
    soda_per_week / (total_cost_per_week / cost_per_box) = 30 :=
by
  -- Definitions from conditions
  rw [h_soda, h_cost_box, h_total_cost] 
  norm_num
  -- Proof steps omitted
  sorry

end syrup_boxes_l483_483962


namespace triangle_ABC_length_BC_l483_483386

noncomputable theory
open scoped classical

variables (A B C : Type*) [euclidean_space A B C]
variables (AB AC BC BY CY x y : ℝ)

def triangle_ABC (AB AC x y : ℝ) :=
  AB = 74 ∧ AC = 88 ∧ ∃ y, y ∈ ℤ ∧ ∃ x, x ∈ ℤ ∧
    let BC := x + y in
    (88)^2 * y + (74)^2 * x = BC * (x * y + 88^2) ∧ BC = 104

theorem triangle_ABC_length_BC :
  triangle_ABC A B C AB AC x y →
  BC = 104 := by
  intro h
  rcases h with ⟨hAB, hAC, hyint, ⟨hy, hxyint, ⟨hx, hBC⟩, hsol⟩⟩
  rw [← hsol]

  sorry

end triangle_ABC_length_BC_l483_483386


namespace largest_prime_factor_of_1729_is_19_l483_483252

def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def largest_prime_factor (n : ℕ) (p : ℕ) := is_prime p ∧ p ∣ n ∧ ∀ q : ℕ, is_prime q ∧ q ∣ n → q ≤ p

theorem largest_prime_factor_of_1729_is_19 : largest_prime_factor 1729 19 :=
by
  sorry

end largest_prime_factor_of_1729_is_19_l483_483252


namespace problem1_l483_483529

theorem problem1 (a : ℝ) (x : ℝ) (h : a > 0) : |x - (1/a)| + |x + a| ≥ 2 :=
sorry

end problem1_l483_483529


namespace find_base_l483_483621

theorem find_base (m : ℕ) (h : m ≥ 8)
  (H : ∀ m, 7324_m + 6251_m = 15375_m) : m = 8 := 
sorry

end find_base_l483_483621


namespace calories_per_one_bar_l483_483496

variable (total_calories : ℕ) (num_bars : ℕ)
variable (calories_per_bar : ℕ)

-- Given conditions
axiom total_calories_given : total_calories = 15
axiom num_bars_given : num_bars = 5

-- Mathematical equivalent proof problem
theorem calories_per_one_bar :
  total_calories / num_bars = calories_per_bar →
  calories_per_bar = 3 :=
by
  sorry

end calories_per_one_bar_l483_483496


namespace add_base3_l483_483571

theorem add_base3 :
  ∀ (a b c d : ℕ),
    a = 2 ∧ b = 102 ∧ c = 1102 ∧ d = 11021 → (a + b + c + d) = 1221 := 
by
  intros a b c d h
  cases' h with h1 h2
  cases' h2 with h3 h4
  cases' h3 with h5 h6
  rw [h1, h6, h5, h4]
  sorry

end add_base3_l483_483571


namespace expected_waiting_time_l483_483226

/-- Consider a 5-minute interval. There are 5 bites on the first rod 
and 1 bite on the second rod in this interval. Therefore, the total average 
number of bites on both rods during these 5 minutes is 6. The expected waiting 
time for the first bite is 50 seconds. -/
theorem expected_waiting_time
    (average_bites_first_rod : ℝ)
    (average_bites_second_rod : ℝ)
    (total_interval_minutes : ℝ)
    (expected_waiting_time_seconds : ℝ) :
    average_bites_first_rod = 5 ∧
    average_bites_second_rod = 1 ∧
    total_interval_minutes = 5 →
    expected_waiting_time_seconds = 50 :=
by
  sorry

end expected_waiting_time_l483_483226


namespace area_comparison_l483_483032

-- Definitions based on the conditions:
def s1 := (26 + 26 + 30) / 2 -- Semiperimeter of the first triangle
def A := Real.sqrt (s1 * (s1 - 26) * (s1 - 26) * (s1 - 30))

def s2 := (26 + 26 + 50) / 2 -- Semiperimeter of the second triangle
def B := Real.sqrt (s2 * (s2 - 26) * (s2 - 26) * (s2 - 50))

-- The statement of the theorem to prove
theorem area_comparison : A > B := by
  sorry

end area_comparison_l483_483032


namespace roots_quadratic_sum_squares_l483_483352

theorem roots_quadratic_sum_squares :
  (∃ a b : ℝ, (∀ x : ℝ, x^2 - 4 * x + 4 = 0 → (x = a ∨ x = b)) ∧ a^2 + b^2 = 8) :=
by
  sorry

end roots_quadratic_sum_squares_l483_483352


namespace negation_is_all_odd_or_at_least_two_even_l483_483485

-- Define natural numbers a, b, and c.
variables {a b c : ℕ}

-- Define a predicate is_even which checks if a number is even.
def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

-- Define the statement that exactly one of the natural numbers a, b, and c is even.
def exactly_one_even (a b c : ℕ) : Prop :=
  (is_even a ∨ is_even b ∨ is_even c) ∧
  ¬ (is_even a ∧ is_even b) ∧
  ¬ (is_even a ∧ is_even c) ∧
  ¬ (is_even b ∧ is_even c)

-- Define the negation of the statement that exactly one of the natural numbers a, b, and c is even.
def negation_of_exactly_one_even (a b c : ℕ) : Prop :=
  ¬ exactly_one_even a b c

-- State that the negation of exactly one even number among a, b, c is equivalent to all being odd or at least two being even.
theorem negation_is_all_odd_or_at_least_two_even :
  negation_of_exactly_one_even a b c ↔ (¬ is_even a ∧ ¬ is_even b ∧ ¬ is_even c) ∨ (is_even a ∧ is_even b) ∨ (is_even a ∧ is_even c) ∨ (is_even b ∧ is_even c) :=
sorry

end negation_is_all_odd_or_at_least_two_even_l483_483485


namespace find_k_and_direction_l483_483705

open Vector

-- Define vectors a, b, c, and d
def a : ℝ × ℝ := (1, 0)
def b : ℝ × ℝ := (0, 1)
def c (k : ℝ) : ℝ × ℝ := (k * a.1 + b.1, k * a.2 + b.2)
def d : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)

-- Define parallel condition as scalar multiple of each other
def parallel (u v : ℝ × ℝ) : Prop :=
  ∃ λ : ℝ, u.1 = λ * v.1 ∧ u.2 = λ * v.2

theorem find_k_and_direction (k : ℝ) :
  parallel (c k) d → k = -1 ∧ (c k).1 = -(d.1) ∧ (c k).2 = -(d.2) :=
by
  intro h
  sorry

end find_k_and_direction_l483_483705


namespace balance_difference_l483_483584

noncomputable def cedric_balance (P_C : ℝ) (r_C : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P_C * (1 + r_C / n) ^ (n * t)

noncomputable def daniel_balance (P_D : ℝ) (r_D : ℝ) (t : ℕ) : ℝ :=
  P_D * (1 + r_D * t)

theorem balance_difference 
  (P_C P_D : ℝ) (r_C r_D : ℝ) (n t : ℕ) :
  P_C = 15000 → P_D = 15000 → r_C = 0.045 → r_D = 0.06 → n = 12 → t = 10 →
  abs (daniel_balance P_D r_D t - cedric_balance P_C r_C n t) = 413 :=
by {
  intros,
  sorry
}

end balance_difference_l483_483584


namespace sufficient_but_not_necessary_condition_l483_483523

-- Defining the conditions in Lean
variables {x : ℝ}

-- Define the conditions in lean
def absolute_condition := (|x| < 2)
def quadratic_condition := (x^2 - x - 6 < 0)

-- The main theorem we need to prove
theorem sufficient_but_not_necessary_condition : 
  absolute_condition → quadratic_condition ∧ ¬ (quadratic_condition → absolute_condition) :=
by
  sorry

end sufficient_but_not_necessary_condition_l483_483523


namespace flag_designs_count_l483_483245

theorem flag_designs_count :
  let colors := 3 in
  colors * colors * colors = 27 :=
by
  let colors := 3
  show colors * colors * colors = 27
  sorry

end flag_designs_count_l483_483245


namespace div_by_n_plus_2_iff_n_even_l483_483264

theorem div_by_n_plus_2_iff_n_even (n k : ℕ) (hn : 0 < n) (hk : 0 < k): 
  (n + 2) ∣ (∑ i in Finset.range (n + 1), i ^ (2 * k + 1)) ↔ (n % 2 = 0) := 
sorry

end div_by_n_plus_2_iff_n_even_l483_483264


namespace smallest_arithmetic_mean_divisible_by_1111_l483_483856

/-- 
Given the product of nine consecutive natural numbers is divisible by 1111, 
prove that the smallest possible value of the arithmetic mean of these nine numbers is 97.
-/
theorem smallest_arithmetic_mean_divisible_by_1111 :
  ∃ n : ℕ, (∀ k : ℕ, k = n →  (∏ i in finset.range 9, k + i) % 1111 = 0) 
  ∧ (n ≥ 93) ∧ (n + 4 = 97) :=
sorry

end smallest_arithmetic_mean_divisible_by_1111_l483_483856


namespace nathan_ate_100_gumballs_l483_483338

/-- Define the number of gumballs per package. -/
def gumballs_per_package : ℝ := 5.0

/-- Define the number of packages Nathan ate. -/
def number_of_packages : ℝ := 20.0

/-- Define the total number of gumballs Nathan ate. -/
def total_gumballs : ℝ := number_of_packages * gumballs_per_package

/-- Prove that Nathan ate 100.0 gumballs. -/
theorem nathan_ate_100_gumballs : total_gumballs = 100.0 :=
sorry

end nathan_ate_100_gumballs_l483_483338


namespace triangle_area_tangent_line_l483_483613

noncomputable def curve (x : ℝ) : ℝ := 2 * Real.log x 

noncomputable def deriv_curve (x : ℝ) : ℝ := 2 / x

def point : ℝ × ℝ := (Real.exp 2, 4)

theorem triangle_area_tangent_line : 
  let slope := deriv_curve point.1,
      intercept := point.2 - slope * point.1,
      tangent_line (x : ℝ) : ℝ := slope * x + intercept,
      area := (1 / 2) * tangent_line 0 * abs point.1
  in area = Real.exp 2 :=
by
  sorry

end triangle_area_tangent_line_l483_483613


namespace shaded_area_proof_l483_483564

noncomputable def total_shaded_area (side_length: ℝ) (large_square_ratio: ℝ) (small_square_ratio: ℝ): ℝ := 
  let S := side_length / large_square_ratio
  let T := S / small_square_ratio
  let large_square_area := S ^ 2
  let small_square_area := T ^ 2
  large_square_area + 12 * small_square_area

theorem shaded_area_proof
  (h1: ∀ side_length, side_length = 15)
  (h2: ∀ large_square_ratio, large_square_ratio = 5)
  (h3: ∀ small_square_ratio, small_square_ratio = 4)
  : total_shaded_area 15 5 4 = 15.75 :=
by
  sorry

end shaded_area_proof_l483_483564


namespace average_waiting_time_for_first_bite_l483_483221

/-- 
Let S be a period of 5 minutes (300 seconds).
- We have an average of 5 bites in 300 seconds on the first fishing rod.
- We have an average of 1 bite in 300 seconds on the second fishing rod.
- The total average number of bites on both rods during this period is 6 bites.
The bites occur independently and follow a Poisson process.

We aim to prove that the waiting time for the first bite, given these conditions, is 
expected to be 50 seconds.
-/
theorem average_waiting_time_for_first_bite :
  let S := 300 -- 5 minutes in seconds
  -- The average number of bites on the first and second rod in period S.
  let avg_bites1 := 5
  let avg_bites2 := 1
  let total_avg_bites := avg_bites1 + avg_bites2
  -- The rate parameter λ for the Poisson process is total_avg_bites / S.
  let λ := total_avg_bites / S
  -- The average waiting time for the first bite.
  1 / λ = 50 :=
by
  let S := 300
  let avg_bites1 := 5
  let avg_bites2 := 1
  let total_avg_bites := avg_bites1 + avg_bites2
  let λ := total_avg_bites / S
  -- convert λ to seconds to ensure unit consistency
  have hλ: λ = 6 / 300 := rfl
  -- The expected waiting time for the first bite is 1 / λ
  have h_waiting_time: 1 / λ = 300 / 6 := by
    rw [hλ, one_div, div_div_eq_mul]
    norm_num
  exact h_waiting_time

end average_waiting_time_for_first_bite_l483_483221


namespace Clarissa_needs_to_bring_photos_l483_483592

variable (Cristina John Sarah Clarissa Total_slots : ℕ)

def photo_album_problem := Cristina = 7 ∧ John = 10 ∧ Sarah = 9 ∧ Total_slots = 40 ∧
  (Clarissa + Cristina + John + Sarah = Total_slots)

theorem Clarissa_needs_to_bring_photos (h : photo_album_problem 7 10 9 14 40) : Clarissa = 14 := by
  cases h with _ h, cases h with _ h, cases h with _ h, cases h with _ h, cases h
  sorry

end Clarissa_needs_to_bring_photos_l483_483592


namespace proper_fraction_one_over_3x2_improper_fraction_conversion_improper_fraction_integer_values_improper_to_mixed_fraction_l483_483456

-- 1. Prove that the fraction 1 / 3x^2 is a proper fraction
theorem proper_fraction_one_over_3x2 (x : ℝ) (h : x ≠ 0) : 
  power 1 < power (3 * x^2) := sorry

-- 2. Convert 4a+1 / 2a-1 and find values of a such that 4a+1 / 2a-1 is an integer
theorem improper_fraction_conversion (a : ℝ) (h : a ≠ 1/2) : 
  (4 * a + 1) / (2 * a - 1) = 2 + 3 / (2 * a - 1) :=
sorry

theorem improper_fraction_integer_values (a : ℝ) : 
  ((4 * a + 1) / (2 * a - 1)).isInt -> ∃ x ∈ {1, 2, -1}, a = x :=
sorry

-- 3. Convert x^2 - 2x - 1 / x - 1 into a mixed fraction
theorem improper_to_mixed_fraction (x : ℝ) (h : x ≠ 1) :
  (x^2 - 2 * x - 1) / (x - 1) = x - 1 - 2 / (x - 1) :=
sorry

end proper_fraction_one_over_3x2_improper_fraction_conversion_improper_fraction_integer_values_improper_to_mixed_fraction_l483_483456


namespace problems_per_worksheet_l483_483188

theorem problems_per_worksheet (P : ℕ) (graded : ℕ) (remaining : ℕ) (total_worksheets : ℕ) (total_problems_remaining : ℕ) :
    graded = 5 →
    total_worksheets = 9 →
    total_problems_remaining = 16 →
    remaining = total_worksheets - graded →
    4 * P = total_problems_remaining →
    P = 4 :=
by
  intros h_graded h_worksheets h_problems h_remaining h_equation
  sorry

end problems_per_worksheet_l483_483188


namespace correct_list_prices_l483_483958

def cost_price_A : ℚ := 47.50
def cost_price_B : ℚ := 65
def cost_price_C : ℚ := 32

def profit_percentage_A : ℚ := 0.25
def profit_percentage_B : ℚ := 0.30
def profit_percentage_C : ℚ := 0.20

def discount_percentage_A : ℚ := 0.15
def discount_percentage_B : ℚ := 0.10
def discount_percentage_C : ℚ := 0.05

def list_price_A : ℚ := 50.47
def list_price_B : ℚ := 76.05
def list_price_C : ℚ := 36.48

theorem correct_list_prices :
  let profit_amount_A := cost_price_A * profit_percentage_A in
  let selling_price_before_discount_A := cost_price_A + profit_amount_A in
  let discount_amount_A := selling_price_before_discount_A * discount_percentage_A in
  let calculated_list_price_A := selling_price_before_discount_A - discount_amount_A in

  let profit_amount_B := cost_price_B * profit_percentage_B in
  let selling_price_before_discount_B := cost_price_B + profit_amount_B in
  let discount_amount_B := selling_price_before_discount_B * discount_percentage_B in
  let calculated_list_price_B := selling_price_before_discount_B - discount_amount_B in

  let profit_amount_C := cost_price_C * profit_percentage_C in
  let selling_price_before_discount_C := cost_price_C + profit_amount_C in
  let discount_amount_C := selling_price_before_discount_C * discount_percentage_C in
  let calculated_list_price_C := selling_price_before_discount_C - discount_amount_C in
  
  calculated_list_price_A ≈ list_price_A ∧
  calculated_list_price_B ≈ list_price_B ∧
  calculated_list_price_C ≈ list_price_C :=
by {
  sorry
}

end correct_list_prices_l483_483958


namespace staplers_left_is_correct_l483_483107

-- Define the initial conditions as constants
def initial_staplers : ℕ := 450
def stacie_reports : ℕ := 8 * 12 -- Stacie's reports in dozens converted to actual number
def jack_reports : ℕ := 9 * 12   -- Jack's reports in dozens converted to actual number
def laura_reports : ℕ := 50      -- Laura's individual reports

-- Define the stapler usage rates
def stacie_usage_rate : ℕ := 1                  -- Stacie's stapler usage rate (1 stapler per report)
def jack_usage_rate : ℕ := stacie_usage_rate / 2  -- Jack's stapler usage rate (half of Stacie's)
def laura_usage_rate : ℕ := stacie_usage_rate * 2 -- Laura's stapler usage rate (twice of Stacie's)

-- Define the usage calculations
def stacie_usage : ℕ := stacie_reports * stacie_usage_rate
def jack_usage : ℕ := jack_reports * jack_usage_rate
def laura_usage : ℕ := laura_reports * laura_usage_rate

-- Define total staplers used
def total_usage : ℕ := stacie_usage + jack_usage + laura_usage

-- Define the number of staplers left
def staplers_left : ℕ := initial_staplers - total_usage

-- Prove that the staplers left is 200
theorem staplers_left_is_correct : staplers_left = 200 := by
  unfold staplers_left initial_staplers total_usage stacie_usage jack_usage laura_usage
  unfold stacie_reports jack_reports laura_reports
  unfold stacie_usage_rate jack_usage_rate laura_usage_rate
  sorry   -- Place proof here

end staplers_left_is_correct_l483_483107


namespace parabola_chord_length_l483_483698

variable (p : ℝ)

theorem parabola_chord_length (h1 : 0 < p)
                              (h2 : ∀ x1 x2 y1 y2, y1 + y2 = 6)
                              (h3 : |-(x1 - x2)| = (6 + p ∨ 8)): p = 2 := sorry

end parabola_chord_length_l483_483698


namespace part1_part2_l483_483627

variable {m : ℝ}

def z (m : ℝ) : ℂ := complex.of_real (m * (m - 1)) + complex.I * (m - 1)

theorem part1 : (∃ m : ℝ, z m ∈ ℝ) ↔ m = 1 := by sorry

theorem part2 : (∃ m : ℝ, z m.re = 0 ∧ z m.im ≠ 0) ↔ m = 0 := by sorry

end part1_part2_l483_483627


namespace monotonicity_of_f_increasing_g_if_g_increasing_l483_483692

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := ln x - a / x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := f x a + a * x - 6 * ln x

theorem monotonicity_of_f
  (a : ℝ) (x : ℝ) (hx : 0 < x) :
  (0 ≤ a → ∀ x, 0 < x → 0 < (x + a) / x^2) ∧ 
  (a < 0 → ∀ x, 0 < x → (-a < x ↔ 0 < (x + a) / x^2) ∧ (x < -a ↔ (x + a) / x^2 < 0)) := sorry

theorem increasing_g_if_g_increasing
  (a : ℝ) :
  (∀ x : ℝ, 0 < x → (a * x^2 - 5 * x + a) / x^2 ≥ 0) → a ≥ 5 / 2 := sorry

end monotonicity_of_f_increasing_g_if_g_increasing_l483_483692


namespace collinear_A_S₁_S₂_l483_483407

open EuclideanGeometry

variables {A B C A' B' C' P Q R S₁ S₂ : Point} -- Defining the points

-- Given conditions
axiom midpoint_A' : midpoint A' B C
axiom midpoint_B' : midpoint B' C A
axiom midpoint_C' : midpoint C' A B
axiom foot_P : altitude_foot P A B C
axiom foot_Q : altitude_foot Q B C A
axiom foot_R : altitude_foot R C A B
axiom intersection_S₁ : ∃ S₁, intersects_at S₁ (line P Q) (line A' C') 
axiom intersection_S₂ : ∃ S₂, intersects_at S₂ (line P R) (line A' B') 

-- The theorem stating A, S₁, and S₂ are collinear
theorem collinear_A_S₁_S₂ : collinear {A, S₁, S₂} :=
sorry

end collinear_A_S₁_S₂_l483_483407


namespace probability_two_blue_gumballs_l483_483750

/-- Given:
  (1) A jar with pink and blue gumballs.
  (2) Each draw is independent (replaced each time).
  (3) Probability of drawing a pink gumball is 0.5714285714285714.
 Prove: The probability of drawing two blue gumballs in a row is approximately 0.1836734693877551. -/
theorem probability_two_blue_gumballs (P_P : ℝ) (P_P_eq : P_P = 0.5714285714285714) : 
  let P_B := 1 - P_P in
  (P_B * P_B ≈ 0.1836734693877551) :=
by
  sorry

end probability_two_blue_gumballs_l483_483750


namespace selling_price_calculation_l483_483193

-- Conditions
def CP : ℝ := 280
def profit_percentage : ℝ := 30
def profit := (profit_percentage / 100) * CP

-- Theorem statement
theorem selling_price_calculation : CP + profit = 364 := sorry

end selling_price_calculation_l483_483193


namespace train_speed_is_54_kph_l483_483156

-- Define the known quantities
def length_of_train : ℝ := 75  -- meters
def time_to_cross_pole : ℝ := 5  -- seconds

-- Conversion factor from m/s to km/hr
def conversion_factor : ℝ := 3.6 

-- Definition of speed calculation in m/s
def speed_in_mps : ℝ := length_of_train / time_to_cross_pole

-- Definition of speed calculation in km/hr
def speed_in_kph : ℝ := speed_in_mps * conversion_factor

-- The proof problem: Speed of the train in km/hr is 54 km/hr
theorem train_speed_is_54_kph : speed_in_kph = 54 := 
by 
  -- Proof is skipped
  sorry

end train_speed_is_54_kph_l483_483156


namespace inv_matrix_A_l483_483331

def A : Matrix (Fin 2) (Fin 2) ℚ :=
  ![ ![ -2, 1 ],
     ![ (3/2 : ℚ), -1/2 ] ]

def A_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  ![ ![ 1, 2 ],
     ![ 3, 4 ] ]

theorem inv_matrix_A : A⁻¹ = A_inv := by
  sorry

end inv_matrix_A_l483_483331


namespace norm_sum_eq_sqrt_45_div_7_l483_483035

  open Real

  variable (p q r : ℝ^3)

  def unit_vector (v : ℝ^3) : Prop := ∥v∥ = 1

  theorem norm_sum_eq_sqrt_45_div_7 :
    unit_vector p ∧ unit_vector q ∧ unit_vector r ∧
    p.dot q = -1/7 ∧ p.dot r = -1/7 ∧ q.dot r = -1/7 →
    ∥p + q + r∥ = sqrt (45 / 7) :=
  by
    sorry
  
end norm_sum_eq_sqrt_45_div_7_l483_483035


namespace min_max_product_l483_483413

-- Define the conditions
variables (x y : ℝ)

-- Define the expressions and conditions
def condition : Prop := 9 * x^2 + 15 * x * y + 8 * y^2 = 1

def quadratic_form (x y : ℝ) : ℝ := 3 * x^2 + 4 * x * y + 3 * y^2

-- Define m and M as the minimum and maximum values of the quadratic form

-- The main proof statement
theorem min_max_product (x y : ℝ) (h : condition x y) :
  let m := min (quadratic_form x y) in
  let M := max (quadratic_form x y) in
  m * M = 3 / 81 :=
sorry

end min_max_product_l483_483413


namespace new_computer_cost_l483_483399

-- Definition of the conditions
def cost_new_computer (C : ℝ) : Prop :=
  let cost_used_computers := 200 * 2 in
  let saving := 200 in
  C - saving = cost_used_computers

-- The theorem we want to prove
theorem new_computer_cost (C : ℝ) : C = 600 :=
begin
  unfold cost_new_computer, -- Using the definition we created
  assume h : cost_new_computer C,
  rw [<-h],
  norm_num,
end

end new_computer_cost_l483_483399


namespace smallest_arithmetic_mean_divisible_by_1111_l483_483824

theorem smallest_arithmetic_mean_divisible_by_1111 :
  ∃ (n : ℕ), (∀ i : ℕ, i < 9 → 1111 ∣ (n + i)) ∧ (n + 4 = 97) :=
by
  sorry

end smallest_arithmetic_mean_divisible_by_1111_l483_483824


namespace jacqueline_candy_multiple_l483_483284

theorem jacqueline_candy_multiple :
  let fred_candy := 12
  let uncle_bob_candy := fred_candy + 6
  let total_candy := fred_candy + uncle_bob_candy
  let jackie_candy := 120 / 0.40
  (jackie_candy / total_candy = 10) :=
by
  let fred_candy := 12
  let uncle_bob_candy := fred_candy + 6
  let total_candy := fred_candy + uncle_bob_candy
  let jackie_candy := 120 / 0.40
  show _ = _
  sorry

end jacqueline_candy_multiple_l483_483284


namespace find_x_for_given_y_l483_483493

theorem find_x_for_given_y (x y : ℝ) (h_pos : 0 < x ∧ 0 < y) (h_initial : x = 2 ∧ y = 8) (h_inverse : (2 ^ 3) * 8 = 128) :
  y = 1728 → x = (1 / (13.5) ^ (1 / 3)) :=
by
  sorry

end find_x_for_given_y_l483_483493


namespace smallest_arithmetic_mean_divisible_by_1111_l483_483830

noncomputable def nine_consecutive_numbers {n : ℕ} : list ℕ :=
  [n, n+1, n+2, n+3, n+4, n+5, n+6, n+7, n+8]

def divisible_by (x y : ℕ) : Prop := ∃ k : ℕ, x = k * y

def arithmetic_mean {l : list ℕ} (h_len : l.length = 9) : ℚ :=
  (l.sum : ℚ) / 9

theorem smallest_arithmetic_mean_divisible_by_1111 :
  ∃ n : ℕ, 
  divisible_by ((nine_consecutive_numbers n).prod) 1111 ∧ 
  arithmetic_mean (by simp [nine_consecutive_numbers_len]) = 97 :=
sorry

end smallest_arithmetic_mean_divisible_by_1111_l483_483830


namespace total_sum_is_120_rupees_l483_483160

/-- Definitions for the conditions given in the problem -/
def x : ℕ := sorry
def total_paisa_per_rupee := x + 65 + 40
def c_share_paisa := 4800

/-- Theorem stating the mathematically equivalent proof problem -/
theorem total_sum_is_120_rupees 
  (h1 : c_share_paisa = 4800)
  (h2 : total_paisa_per_rupee = x + 65 + 40)
  (h3 : C := 40 ) :
  S := 120 :=
sorry

end total_sum_is_120_rupees_l483_483160


namespace Yvettes_final_bill_l483_483395

namespace IceCreamShop

def sundae_price_Alicia : Real := 7.50
def sundae_price_Brant : Real := 10.00
def sundae_price_Josh : Real := 8.50
def sundae_price_Yvette : Real := 9.00
def tip_rate : Real := 0.20

theorem Yvettes_final_bill :
  let total_cost := sundae_price_Alicia + sundae_price_Brant + sundae_price_Josh + sundae_price_Yvette
  let tip := tip_rate * total_cost
  let final_bill := total_cost + tip
  final_bill = 42.00 :=
by
  -- calculations are skipped here
  sorry

end IceCreamShop

end Yvettes_final_bill_l483_483395


namespace ab_ac_product_eq_735_l483_483737

variable (A B C P Q X Y Z : Type)
variable [acute_triangle ABC : Triangle A B C]
variable [foot_perpendicular C P AB : Foot C A B]
variable [foot_perpendicular B Q AC : Foot B A C]
variable (circumcircle_ABC : Circle A B C)
variable [intersect_line_circle PQ circumcircle_ABC ⟨X⟩ ⟨Y⟩]
variable [XP_eq_15 : Distance X P = 15]
variable [PQ_eq_30 : Distance P Q = 30]
variable [QY_eq_20 : Distance Q Y = 20]
variable (another_circle : Circle P Q)
variable [intersect_another_circle_Z PQ another_circle P Z]
variable [PZ_eq_20 : Distance P Z = 20]

theorem ab_ac_product_eq_735 : AB * AC = 735 := sorry

end ab_ac_product_eq_735_l483_483737


namespace min_squared_dist_sum_l483_483146

theorem min_squared_dist_sum :
  let AB := 1
  let BC := 1
  let CD := 3
  let DE := 12
  ∀ (r : ℝ), (0 ≤ r ∧ r ≤ 4) →
    (r^2 + (r-1)^2 + (r-2)^2 + (r-4)^2 + (r-16)^2) ≥ 237 :=
begin
  sorry
end

end min_squared_dist_sum_l483_483146


namespace smallest_arithmetic_mean_divisible_by_1111_l483_483819

theorem smallest_arithmetic_mean_divisible_by_1111 :
  ∃ (n : ℕ), (∀ i : ℕ, i < 9 → 1111 ∣ (n + i)) ∧ (n + 4 = 97) :=
by
  sorry

end smallest_arithmetic_mean_divisible_by_1111_l483_483819


namespace range_of_m_l483_483720

theorem range_of_m (m : ℝ) (h1 : 0 < m) (h2 : ∀ x : ℝ, 0 < x ∧ x ≤ 1 → |m * x^3 - Real.log x| ≥ 1) : m ≥ (1 / 3) * Real.exp 2 :=
sorry

end range_of_m_l483_483720


namespace smallest_arithmetic_mean_divisible_by_1111_l483_483831

noncomputable def nine_consecutive_numbers {n : ℕ} : list ℕ :=
  [n, n+1, n+2, n+3, n+4, n+5, n+6, n+7, n+8]

def divisible_by (x y : ℕ) : Prop := ∃ k : ℕ, x = k * y

def arithmetic_mean {l : list ℕ} (h_len : l.length = 9) : ℚ :=
  (l.sum : ℚ) / 9

theorem smallest_arithmetic_mean_divisible_by_1111 :
  ∃ n : ℕ, 
  divisible_by ((nine_consecutive_numbers n).prod) 1111 ∧ 
  arithmetic_mean (by simp [nine_consecutive_numbers_len]) = 97 :=
sorry

end smallest_arithmetic_mean_divisible_by_1111_l483_483831


namespace unique_partition_no_primes_l483_483281

open Set

def C_oplus_C (C : Set ℕ) : Set ℕ :=
  {z | ∃ x y, x ∈ C ∧ y ∈ C ∧ x ≠ y ∧ z = x + y}

def is_partition (A B : Set ℕ) : Prop :=
  (A ∪ B = univ) ∧ (A ∩ B = ∅)

theorem unique_partition_no_primes (A B : Set ℕ) :
  (is_partition A B) ∧ (∀ x ∈ C_oplus_C A, ¬Nat.Prime x) ∧ (∀ x ∈ C_oplus_C B, ¬Nat.Prime x) ↔ 
    (A = { n | n % 2 = 1 }) ∧ (B = { n | n % 2 = 0 }) :=
sorry

end unique_partition_no_primes_l483_483281


namespace exists_set_satisfying_inequality_l483_483784

theorem exists_set_satisfying_inequality (n : ℕ) (h : 0 < n) : 
  ∃ S : Finset ℝ, S.card = n ∧ 
    (∀ (x y z : ℝ), x ∈ S → y ∈ S → z ∈ S → 
    (x ≠ y ∧ y ≠ z ∧ x ≠ z) → (|x - y| > (1 + 1 / n^1.6) * |x - z| ∨ 
                                 |x - z| > (1 + 1 / n^1.6) * |x - y|)) :=
sorry

end exists_set_satisfying_inequality_l483_483784


namespace solve_inequality_l483_483074

open Set Real

noncomputable def inequality_solution_set : Set ℝ :=
  {x | -2 ≤ x ∧ x ≤ 2} ∪ {6}

theorem solve_inequality (x : ℝ) :
  (x ^ 2 - 4) * (x - 6) ^ 2 ≤ 0 ↔ x ∈ inequality_solution_set := 
sorry

end solve_inequality_l483_483074


namespace tournament_games_needed_l483_483991

theorem tournament_games_needed (teams : ℕ) (h1 : teams = 19) (h2 : ∀ t, t > 1 → ∃ g, g = t - 1) : 
  (games_needed : ℕ) (h3 : games_needed = teams - 1) : 
  games_needed = 18 :=
by
  -- Leaving the proof as an exercise
  sorry

end tournament_games_needed_l483_483991


namespace slope_range_l483_483315

theorem slope_range (a : ℝ) (ha : a ∈ Set.Icc (Real.pi / 4) (Real.pi / 2)) :
  ∃ k : ℝ, k = Real.tan a ∧ k ∈ Set.Ici 1 :=
by {
  sorry
}

end slope_range_l483_483315


namespace rationalize_denominator_proof_l483_483449

def rationalize_denominator (cbrt : ℝ → ℝ) (a : ℝ) :=
  cbrt a = a^(1/3)

theorem rationalize_denominator_proof : 
  (rationalize_denominator (λ x, x ^ (1/3)) 27) →
  (rationalize_denominator (λ x, x ^ (1/3)) 9) →
  (1 / (3 ^ (1 / 3) + 3) = 9 ^ (1 / 3) / (3 + 9 * 3 ^ (1 / 3))) :=
by
  sorry

end rationalize_denominator_proof_l483_483449


namespace amount_of_tin_equals_44_l483_483153

-- Define the mass of alloy A and B
def mass_alloy_A : ℝ := 60
def mass_alloy_B : ℝ := 100

-- Define the ratio conditions
def ratio_lead_tin_A : ℝ × ℝ := (3, 2)
def ratio_tin_copper_B : ℝ × ℝ := (1, 4)

-- Define the amount of tin in alloy A and B given the ratios
def tin_in_alloy_A : ℝ := (ratio_lead_tin_A.2 / (ratio_lead_tin_A.1 + ratio_lead_tin_A.2)) * mass_alloy_A
def tin_in_alloy_B : ℝ := (ratio_tin_copper_B.1 / (ratio_tin_copper_B.1 + ratio_tin_copper_B.2)) * mass_alloy_B

-- The total amount of tin in the new alloy
def total_tin : ℝ := tin_in_alloy_A + tin_in_alloy_B

-- The goal theorem: total amount of tin is 44 kg
theorem amount_of_tin_equals_44 : total_tin = 44 :=
by
  unfold total_tin tin_in_alloy_A tin_in_alloy_B ratio_lead_tin_A ratio_tin_copper_B mass_alloy_A mass_alloy_B
  sorry

end amount_of_tin_equals_44_l483_483153


namespace mappings_Q_to_P_l483_483723

-- Define sets and their cardinalities
def Q : Set := {a, b, c}
def |Q| : Nat := 3

-- Given condition: 81 different mappings from P to Q
axiom card_P (P : Set) (h : (|Q|^|P|) = 81) : Type

-- Prove that there are 64 different mappings from Q to P
theorem mappings_Q_to_P (P : Set) (h : (|Q|^|P|) = 81) : (|P|^|Q|) = 64 :=
by
  sorry

end mappings_Q_to_P_l483_483723


namespace joan_needs_94_eggs_l483_483023

constant vanillaCakes : Nat
constant chocolateCakes : Nat
constant carrotCakes : Nat
constant eggsPerVanillaCake : Nat
constant eggsPerChocolateCake : Nat
constant eggsPerCarrotCake : Nat

def totalEggsNeeded : Nat :=
  (vanillaCakes * eggsPerVanillaCake) + (chocolateCakes * eggsPerChocolateCake) + (carrotCakes * eggsPerCarrotCake)

theorem joan_needs_94_eggs :
  vanillaCakes = 5 ∧ chocolateCakes = 4 ∧ carrotCakes = 3 ∧
  eggsPerVanillaCake = 8 ∧ eggsPerChocolateCake = 6 ∧ eggsPerCarrotCake = 10 →
  totalEggsNeeded = 94 := by
  sorry

end joan_needs_94_eggs_l483_483023


namespace math_problem_proof_l483_483651

noncomputable def ellipse_equation : Prop := 
  let e := (Real.sqrt 2) / 2
  ∃ (a b : ℝ), 0 < a ∧ a > b ∧ e = (Real.sqrt 2) / 2 ∧ 
    (∀ x y, (x^2) / (a^2) + (y^2) / (b^2) = 1 ↔ x^2 / 2 + y^2 = 1)

noncomputable def fixed_point_exist : Prop :=
  let S := (0, 1/3) 
  ∀ k : ℝ, ∃ A B : ℝ × ℝ, 
    let M := (0, 1)
    ( 
        (A.1, A.2) ∈ {P : ℝ × ℝ | (P.1^2) / 2 + P.2^2 = 1} ∧ 
        (B.1, B.2) ∈ {P : ℝ × ℝ | (P.1^2) / 2 + P.2^2 = 1} ∧ 
        (S.2 = k * S.1 - 1 / 3) ∧ 
        ((A.1 - M.1)^2 + (A.2 - M.2)^2) + ((B.1 - M.1)^2 + (B.2 - M.2)^2) = ((A.1 - B.1)^2 + (A.2 - M.2)^2) / 2)

theorem math_problem_proof : ellipse_equation ∧ fixed_point_exist := sorry

end math_problem_proof_l483_483651


namespace expected_waiting_time_first_bite_l483_483235

-- Definitions and conditions as per the problem
def poisson_rate := 6  -- lambda value, bites per 5 minutes
def interval_minutes := 5
def interval_seconds := interval_minutes * 60
def expected_waiting_time_seconds := interval_seconds / poisson_rate

-- The theorem we want to prove
theorem expected_waiting_time_first_bite :
  expected_waiting_time_seconds = 50 := 
by
  let x := interval_seconds / poisson_rate
  have h : interval_seconds = 300 := by norm_num; rfl
  have h2 : x = 50 := by rw [h, interval_seconds]; norm_num
  exact h2

end expected_waiting_time_first_bite_l483_483235


namespace lower_interest_rate_is_12_l483_483960

-- Define the conditions  
def sum_invested : ℝ := 2000
def time_period : ℝ := 2
def high_rate : ℝ := 0.18
def interest_difference : ℝ := 240

-- The interest when invested at the high rate
def interest_high_rate := sum_invested * high_rate * time_period

-- The interest when invested at the lower rate
def interest_low_rate (r : ℝ) := sum_invested * (r / 100) * time_period

-- The proof statement
theorem lower_interest_rate_is_12 :
  interest_high_rate = interest_low_rate 12 + interest_difference := 
  sorry

end lower_interest_rate_is_12_l483_483960


namespace expected_waiting_time_for_first_bite_l483_483215

noncomputable def average_waiting_time (λ : ℝ) : ℝ := 1 / λ

theorem expected_waiting_time_for_first_bite (bites_first_rod : ℝ) (bites_second_rod : ℝ) (total_time_minutes : ℝ) (total_time_seconds : ℝ) :
  bites_first_rod = 5 → 
  bites_second_rod = 1 → 
  total_time_minutes = 5 → 
  total_time_seconds = 300 → 
  average_waiting_time (bites_first_rod + bites_second_rod) * total_time_seconds = 50 :=
begin
  intros,
  sorry
end

end expected_waiting_time_for_first_bite_l483_483215


namespace gcd_lcm_sum_l483_483205

-- Definitions
def gcd_42_70 := Nat.gcd 42 70
def lcm_8_32 := Nat.lcm 8 32

-- Theorem statement
theorem gcd_lcm_sum : gcd_42_70 + lcm_8_32 = 46 := by
  sorry

end gcd_lcm_sum_l483_483205


namespace simplify_sqrt_expression_l483_483473

theorem simplify_sqrt_expression :
  sqrt 8 - sqrt 50 + sqrt 72 = 3 * sqrt 2 :=
sorry

end simplify_sqrt_expression_l483_483473


namespace circle_area_is_correct_l483_483127

def radius : ℝ := 5

def area_of_circle (r : ℝ) : ℝ := π * r^2

theorem circle_area_is_correct : area_of_circle radius = 25 * π :=
by
  simp [area_of_circle, radius]
  sorry

end circle_area_is_correct_l483_483127


namespace test_average_score_l483_483798

theorem test_average_score (A : ℝ) (h : 0.90 * A + 5 = 86) : A = 90 := 
by
  sorry

end test_average_score_l483_483798


namespace alternating_series_sum_l483_483926

theorem alternating_series_sum :
  let series := (List.range (5002)).map (λ n, if (n % 2 = 0) then -((n + 1) // 2 * 2) else ((n + 1) // 2 * 2 - 1))
  (series.sum = 2501) :=
by
  let series := (List.range (5002)).map (λ n, if (n % 2 = 0) then -((n + 1) // 2 * 2) else ((n + 1) // 2 * 2 - 1))
  sorry

end alternating_series_sum_l483_483926


namespace quadratic_function_properties_l483_483333

noncomputable def f (a b x : ℝ) := a * x^2 - b * x

theorem quadratic_function_properties (a b : ℝ)
  (h₁ : f a b 2 = 0)
  (h₂ : ∀ x : ℝ, (f a b x = x) → (b + 1 = 0) ∧ (2 * a - b = 0)) :
  (a = -1/2) ∧ (b = -1) ∧ (∀ x : ℝ, f a b x = (-1/2) * x^2 + x) ∧ (∀ x ∈ Icc(0:ℝ, 3:ℝ), f a b x ≤ 0.5) :=
by 
  sorry 

end quadratic_function_properties_l483_483333


namespace work_done_by_variable_force_l483_483795

noncomputable def F (x : ℝ) : ℝ := x^2 + 1

theorem work_done_by_variable_force : 
  ∫ x in 0..6, F x = 78 := 
by
  sorry

end work_done_by_variable_force_l483_483795


namespace CD_eq_CE_l483_483780

theorem CD_eq_CE {Point : Type*} [MetricSpace Point]
  (A B C D E : Point) (m : Set Point)
  (hAm : A ∈ m) (hBm : B ∈ m) (hCm : C ∈ m)
  (hDm : D ∉ m) (hEm : E ∉ m) 
  (hAD_AE : dist A D = dist A E)
  (hBD_BE : dist B D = dist B E) :
  dist C D = dist C E :=
sorry

end CD_eq_CE_l483_483780


namespace correct_statements_l483_483765

open Real

theorem correct_statements (m n : ℝ) (a : ℝ) :
  (¬ (m < n ∧ n < 0 → m^2 < n^2)) ∧ 
  (ma^2 < na^2 → m < n) ∧ 
  (¬ (m / n < a → m < n * a)) ∧ 
  (m < n ∧ n < 0 → n / m < 1) :=
by
  sorry

end correct_statements_l483_483765


namespace a_9_coefficient_l483_483633

theorem a_9_coefficient:
  (\(1 - x\)^10) = (\(\sum_{i : ℕ} (a_i * (1 + x)^i)\)  with bounds \(0 <= i <= 10)) -> 
  a_9 = -20 := by
  sorry

end a_9_coefficient_l483_483633


namespace numPalindromes24h_l483_483956

-- We define a structure for the 24-hour time to make sure it includes hours and minutes with leading zeros
structure Time24h where
  hour : Nat
  minute : Nat
  deriving Repr, BEq

-- An instance defining the validity of the time within 24-hour format
def Time24h.isValid (t : Time24h) : Bool := 
  t.hour < 24 ∧ t.minute < 60

-- A function to check if a given time is a palindrome
def isPalindrome (t : Time24h) : Bool :=
  let hourStr := if t.hour < 10 then "0" ++ toString t.hour else toString t.hour
  let minuteStr := if t.minute < 10 then "0" ++ toString t.minute else toString t.minute
  let timeStr := hourStr ++ minuteStr
  timeStr = String.rev timeStr

-- The theorem to be proved
theorem numPalindromes24h : 
  (Finset.filter (λ t, isPalindrome t ∧ t.isValid) 
    (Finset.product 
      (Finset.range 24) 
      (Finset.range 60))).card = 60 := by sorry

end numPalindromes24h_l483_483956


namespace smallest_arithmetic_mean_divisible_by_1111_l483_483826

noncomputable def nine_consecutive_numbers {n : ℕ} : list ℕ :=
  [n, n+1, n+2, n+3, n+4, n+5, n+6, n+7, n+8]

def divisible_by (x y : ℕ) : Prop := ∃ k : ℕ, x = k * y

def arithmetic_mean {l : list ℕ} (h_len : l.length = 9) : ℚ :=
  (l.sum : ℚ) / 9

theorem smallest_arithmetic_mean_divisible_by_1111 :
  ∃ n : ℕ, 
  divisible_by ((nine_consecutive_numbers n).prod) 1111 ∧ 
  arithmetic_mean (by simp [nine_consecutive_numbers_len]) = 97 :=
sorry

end smallest_arithmetic_mean_divisible_by_1111_l483_483826


namespace triangle_DEF_area_l483_483175

-- Definitions of the conditions as provided
variable (u_1 u_2 u_3 DEF : Type)
variable {Q : DEF}

-- The areas of the triangles u_1, u_2, and u_3
variable (area_u1 : ℝ) (area_u2 : ℝ) (area_u3 : ℝ)

-- Given areas of the smaller triangles
axiom area_u1_def : area_u1 = 16
axiom area_u2_def : area_u2 = 25
axiom area_u3_def : area_u3 = 64

-- The proof statement
theorem triangle_DEF_area (area_DEF : ℝ) (h : area_DEF = (289 : ℝ)) : area_DEF = 289 := 
by
  -- Given the areas of the smaller triangles, prove the area of triangle DEF
  rw [h]
  exact rfl

#eval triangle_DEF_area

end triangle_DEF_area_l483_483175


namespace woman_age_multiple_l483_483570

theorem woman_age_multiple (S : ℕ) (W : ℕ) (k : ℕ) 
  (h1 : S = 27)
  (h2 : W + S = 84)
  (h3 : W = k * S + 3) :
  k = 2 :=
by
  sorry

end woman_age_multiple_l483_483570


namespace problem_statement_l483_483028

variables (A B C D M I N : Type)
variables [CyclicQuadrilateral ABCD]
variables [EqualLengths AD BD]
variables [Intersection M AC BD]
variables [Incenter I (Triangle B C M)]
variables [SecondIntersection N AC (CircumscribedCircle (Triangle B M I))]

theorem problem_statement
  (h1 : CyclicQuadrilateral ABCD)
  (h2 : EqualLengths (Length AD) (Length BD))
  (h3 : Intersection M AC BD)
  (h4 : Incenter I (Triangle B C M))
  (h5 : SecondIntersection N AC (CircumscribedCircle (Triangle B M I))) :
  (Length AN) * (Length NC) = (Length CD) * (Length BN) :=
sorry

end problem_statement_l483_483028


namespace even_floor_of_coprime_odd_integers_l483_483756

theorem even_floor_of_coprime_odd_integers (m n : ℕ) (hm : Odd m) (hn : Odd n) (hmn : Nat.coprime m n) (hgt : m > 1 ∧ n > 1) :
  Even (Int.floor ((m ^ (Nat.totient n + 1) + n ^ (Nat.totient m + 1) : ℤ) / (m * n))) := 
sorry

end even_floor_of_coprime_odd_integers_l483_483756


namespace find_sum_l483_483097

theorem find_sum (a b : ℝ) (ha : a^3 - 3 * a^2 + 5 * a - 17 = 0) (hb : b^3 - 3 * b^2 + 5 * b + 11 = 0) :
  a + b = 2 :=
sorry

end find_sum_l483_483097


namespace winning_candidate_votes_percentage_l483_483365

theorem winning_candidate_votes_percentage (P : ℝ) 
    (majority : P/100 * 6000 - (6000 - P/100 * 6000) = 1200) : 
    P = 60 := 
by 
  sorry

end winning_candidate_votes_percentage_l483_483365


namespace common_ratio_geometric_series_l483_483098

theorem common_ratio_geometric_series {a r S : ℝ} (h₁ : S = (a / (1 - r))) (h₂ : (ar^4 / (1 - r)) = S / 64) (h₃ : S ≠ 0) : r = 1 / 2 :=
sorry

end common_ratio_geometric_series_l483_483098


namespace probability_green_ball_l483_483589

theorem probability_green_ball :
  let p_container_A := 1 / 3
  let p_container_B := 1 / 3
  let p_container_c := 1 / 3
  let p_green_A := 5 / 10
  let p_green_B := 3 / 10
  let p_green_C := 4 / 10
  let total_probability := p_container_A * p_green_A + p_container_B * p_green_B + p_container_C * p_green_C
  total_probability = 2 / 5 :=
by
  sorry

end probability_green_ball_l483_483589


namespace hyperbola_asymptotes_l483_483797

theorem hyperbola_asymptotes (x y : ℝ) :
  (∀ x y, (x^2 / 8) - (y^2 / 6) = 1) → (y = (√3 / 2) * x ∨ y = - (√3 / 2) * x) :=
by
  sorry

end hyperbola_asymptotes_l483_483797


namespace C_decreases_as_R_increases_l483_483686

variable (e n R r : ℝ) (h_e_pos : 0 < e) (h_n_pos : 0 < n) (h_r_pos : 0 < r)

def C (R : ℝ) : ℝ := e * n / (R + n * r)

theorem C_decreases_as_R_increases (h_R_incr : ∀ {R1 R2 : ℝ}, R1 < R2 → C e n R1 r > C e n R2 r) :
  ∀ R1 R2 : ℝ, R1 < R2 → C e n R1 r > C e n R2 r := by
  sorry

end C_decreases_as_R_increases_l483_483686


namespace gcd_140_396_is_4_l483_483123

def gcd_140_396 : ℕ := Nat.gcd 140 396

theorem gcd_140_396_is_4 : gcd_140_396 = 4 :=
by
  unfold gcd_140_396
  sorry

end gcd_140_396_is_4_l483_483123


namespace actual_miles_traveled_l483_483554

def skipped_digits_odometer (digits : List ℕ) : Prop :=
  digits = [0, 1, 2, 3, 6, 7, 8, 9]

theorem actual_miles_traveled (odometer_reading : String) (actual_miles : ℕ) :
  skipped_digits_odometer [0, 1, 2, 3, 6, 7, 8, 9] →
  odometer_reading = "000306" →
  actual_miles = 134 :=
by
  intros
  sorry

end actual_miles_traveled_l483_483554


namespace negation_of_universal_quantifier_l483_483094

-- Define the given proposition
def proposition_P (x : ℝ) : Prop := x > -1 → x < x + 1 → log (x + 1) < x

-- Define the negated proposition
def negated_proposition : Prop := ∃ x : ℝ, x > -1 ∧ log (x + 1) ≥ x

-- State the theorem to be proved
theorem negation_of_universal_quantifier : ¬ (∀ x : ℝ, proposition_P x) → negated_proposition
  sorry

end negation_of_universal_quantifier_l483_483094


namespace strips_overlap_area_l483_483978

theorem strips_overlap_area 
  (length_total : ℝ) 
  (length_left : ℝ) 
  (length_right : ℝ) 
  (area_only_left : ℝ) 
  (area_only_right : ℝ) 
  (length_total_eq : length_total = 16) 
  (length_left_eq : length_left = 9) 
  (length_right_eq : length_right = 7) 
  (area_only_left_eq : area_only_left = 27) 
  (area_only_right_eq : area_only_right = 18) :
  ∃ S : ℝ, (27 + S) / (18 + S) = (9 / 7) ∧ 2 * S = 27 :=
begin
  use 13.5,
  split,
  {
    -- Show proportional relationship holds
    sorry
  },
  {
    -- Show 2 * S = 27
    sorry
  }
end

end strips_overlap_area_l483_483978


namespace inequality_sin_cos_l483_483309

theorem inequality_sin_cos (n m : ℕ) (h1 : n > m) (x : ℝ) (hx : 0 < x ∧ x < real.pi / 2) : 
  2 * abs (real.sin x ^ n - real.cos x ^ n) ≤ 3 * abs (real.sin x ^ m - real.cos x ^ m) :=
by {
  sorry
}

end inequality_sin_cos_l483_483309


namespace sum_of_digits_3plus4_pow_15_l483_483919

theorem sum_of_digits_3plus4_pow_15 : 
  let n := (3 + 4) ^ 15,
      ones_digit := n % 10,
      tens_digit := (n / 10) % 10 in
  ones_digit + tens_digit = 7 := by sorry

end sum_of_digits_3plus4_pow_15_l483_483919


namespace smallest_arithmetic_mean_divisible_by_1111_l483_483818

theorem smallest_arithmetic_mean_divisible_by_1111 :
  ∃ (n : ℕ), (∀ i : ℕ, i < 9 → 1111 ∣ (n + i)) ∧ (n + 4 = 97) :=
by
  sorry

end smallest_arithmetic_mean_divisible_by_1111_l483_483818


namespace binom_divisibility_l483_483291

theorem binom_divisibility {p n : ℕ} (prime_condition : Prime (50 * p)) (n_ge_p : n ≥ p) :
  (nat.choose n p - n / p) % p = 0 :=
  sorry

end binom_divisibility_l483_483291


namespace find_k_of_quadratic_polynomial_l483_483721

variable (k : ℝ)

theorem find_k_of_quadratic_polynomial (h1 : (k - 2) = 0) (h2 : k ≠ 0) : k = 2 :=
by
  -- proof omitted
  sorry

end find_k_of_quadratic_polynomial_l483_483721


namespace smallest_arithmetic_mean_divisible_product_l483_483838

theorem smallest_arithmetic_mean_divisible_product :
  ∃ (n : ℕ), (∀ i : ℕ, i < 9 → Nat.gcd ((n + i) * 1111) 1111 = 1111) ∧ (n + 4 = 97) := 
  sorry

end smallest_arithmetic_mean_divisible_product_l483_483838


namespace cone_height_ratio_and_volume_l483_483178

-- Conditions
def original_circumference : ℝ := 20 * Real.pi
def original_height : ℝ := 40

-- Given the new volume condition
def new_volume : ℝ := 400 * Real.pi

-- The proof problem
theorem cone_height_ratio_and_volume (h₁ : original_circumference = 20 * Real.pi) (h₂ : original_height = 40) (h₃ : new_volume = 400 * Real.pi) :
  let r := 10 in
  let new_height := 12 in
  (new_height / original_height = 3 / 10) ∧ new_volume = 400 * Real.pi :=
by
  sorry

end cone_height_ratio_and_volume_l483_483178


namespace numberOfTriangles_l483_483346

-- Define the conditions of our problem
structure GridRectangle where
  width : ℕ
  height : ℕ

def hasDiagonals (rect : GridRectangle) : Prop :=
  true  -- By problem definition, every sub-rectangle has diagonals

-- Theorem statement which proves the number of triangles in the figure
theorem numberOfTriangles (rect : GridRectangle) (h : hasDiagonals rect) : rect.width = 4 → rect.height = 3 → number_of_triangles rect = 78 := 
by
  sorry

end numberOfTriangles_l483_483346


namespace polyhedron_volume_is_4_5_l483_483746

noncomputable def polyhedron_volume (A B C D : Point)
  (A1 B1 C1 D1 : Point) : Real :=
  let base_area := 1         -- unit square side lengths
  let height := 9
  let parallelepiped_volume := base_area * height
  parallelepiped_volume / 2

theorem polyhedron_volume_is_4_5 : 
  ∀ (A B C D A1 B1 C1 D1 : Point), 
    dist A (plane Point) = 0 ∧ dist B (plane Point) = 0 ∧ 
    dist C (plane Point) = 0 ∧ dist D (plane Point) = 0 ∧
    dist A1 (A + vector (0,0,3)) = 3 ∧ 
    dist B1 (B + vector (0,0,4)) = 4 ∧ 
    dist C1 (C + vector (0,0,6)) = 6 ∧ 
    dist D1 (D + vector (0,0,5)) = 5 -> 
  polyhedron_volume A B C D A1 B1 C1 D1 = 4.5 :=
by
  intros A B C D A1 B1 C1 D1 h
  sorry

end polyhedron_volume_is_4_5_l483_483746


namespace collinear_FDEG_l483_483743

theorem collinear_FDEG (A B C H M D E F G : Point) 
(orthocenter : is_orthocenter A B C H)
(tangent_BC : is_tangent (circle H) BC M)
(tangent_BD : is_tangent (line BD) (circle H) D)
(tangent_CE : is_tangent (line CE) (circle H) E)
(altitude_CF : is_altitude C F A B)
(altitude_BG : is_altitude B G A C) :
collinear F D E G :=
sorry

end collinear_FDEG_l483_483743


namespace buratino_bet_pierrot_bet_illegal_papa_carlo_bet_karabas_barabas_bet_l483_483526

-- Part (a)
theorem buratino_bet (a b c : ℕ) (h : a + b + c = 50) :
  5 * a >= 52 ∧ 4 * b >= 52 ∧ 2 * c >= 52 :=
  sorry

-- Part (b)
theorem pierrot_bet_illegal (a b c : ℕ) (h : a + b + c = 25) :
  ¬(5 * a >= 26 ∧ 4 * b >= 26 ∧ 2 * c >= 26) :=
  sorry

-- Part (c)
theorem papa_carlo_bet (S : ℕ) (h : S = 95) :
  ∃ a b c: ℕ, 5 * a >= S + 5 ∧ 4 * b >= S + 5 ∧ 2 * c >= S + 5 :=
  sorry

-- Part (d)
theorem karabas_barabas_bet (S : ℕ) :
  ¬∃ a b c: ℕ,  5 * a >= 1.06 * S ∧ 4 * b >= 1.06 * S ∧ 2 * c >= 1.06 * S :=
  sorry

end buratino_bet_pierrot_bet_illegal_papa_carlo_bet_karabas_barabas_bet_l483_483526


namespace propositions_correctness_l483_483055

noncomputable def is_l_increasing (f : ℝ → ℝ) (M : set ℝ) (l : ℝ) : Prop :=
∀ x ∈ M, x + l ∈ M ∧ f(x + l) ≥ f(x)

def prop1 : Prop :=
is_l_increasing (λ x, Real.log x / Real.log 2) {x : ℝ | 0 < x} 1 

def prop2 : Prop :=
is_l_increasing (λ x, Real.cos (2 * x)) set.univ Real.pi 

def prop3 : Prop :=
∀ (m : ℝ), m ∈ Icc (-1) (Real.pi) → ¬is_l_increasing (λ x, x^2) (Icc (-1) (Real.pi)) m

-- Main statement to be proved
theorem propositions_correctness : prop1 ∧ prop2 ∧ ¬prop3 :=
by
  sorry

end propositions_correctness_l483_483055


namespace smallest_arithmetic_mean_divisible_by_1111_l483_483855

/-- 
Given the product of nine consecutive natural numbers is divisible by 1111, 
prove that the smallest possible value of the arithmetic mean of these nine numbers is 97.
-/
theorem smallest_arithmetic_mean_divisible_by_1111 :
  ∃ n : ℕ, (∀ k : ℕ, k = n →  (∏ i in finset.range 9, k + i) % 1111 = 0) 
  ∧ (n ≥ 93) ∧ (n + 4 = 97) :=
sorry

end smallest_arithmetic_mean_divisible_by_1111_l483_483855


namespace number_of_correct_propositions_l483_483996

-- Define each of the propositions
def prop1 : Prop := ¬(∃ x : ℝ, x^2 - x > 0) = (∀ x : ℝ, x^2 - x ≤ 0)
def prop2 : Prop := ¬(∀ a b : ℝ, a > b → 2^a > 2^b - 1) = (∀ a b : ℝ, a ≤ b → 2^a ≤ 2^b - 1)
def prop3 : Prop := (∀ p q : Prop, (p ∨ q → p ∧ q)) = False
def prop4 : Prop := ∀ d : ℝ, (∀ a n : ℕ, n ≥ 1 → a + n * d = if (n = 1) then 2 else if (n = 3) then 2 + 3 * d else if (n = 4) then (2 + 3 * d) * d else 2 + n * d)

-- Define the statement we want to prove
theorem number_of_correct_propositions : (count (λ p : Prop, p = true) [prop1, prop2, prop3, prop4] = 1) :=
by
  sorry

end number_of_correct_propositions_l483_483996


namespace closest_integer_to_sum_l483_483269

noncomputable def S : ℝ := ∑ n in finset.range (5000 - 2 + 1), 1 / ((n + 2) ^ 2 - 1)

theorem closest_integer_to_sum : abs (2000 * S - 1500) < 0.5 :=
sorry

end closest_integer_to_sum_l483_483269


namespace infinite_unreachable_points_l483_483170

-- Definitions related to the problem
variable {α : Type*} [noncomputable_section]

-- Suppose the circle's circumference is normalized to 1.
-- α is the arc length between consecutive points of reflection.
def circle_circumference : ℝ := 1

-- Define reflection law (simplified for the purpose of the proof statement)
noncomputable def reflects (A B : ℝ) : Prop := sorry

-- Distance function on the circle's circumference (cyclic nature)
noncomputable def dist (A B : ℝ) : ℝ := (B - A) % circle_circumference

-- Theoretically reflective points creating equal arc lengths.
noncomputable def reflective_points (A : ℝ) : set ℝ :=
  {B : ℝ | ∃ n : ℕ, dist A B = n • α}

-- The main theorem statement to be proved
theorem infinite_unreachable_points (A : ℝ) :
  ∃ S : set ℝ, S.infinite ∧ ∀ B ∈ S, B ∉ reflective_points A :=
sorry

end infinite_unreachable_points_l483_483170


namespace suff_and_necc_l483_483301

variable (x : ℝ)

def A : Set ℝ := { x | x > 2 }
def B : Set ℝ := { x | x < 0 }
def C : Set ℝ := { x | x * (x - 2) > 0 }

theorem suff_and_necc : (x ∈ (A ∪ B)) ↔ (x ∈ C) := by
  sorry

end suff_and_necc_l483_483301


namespace chess_tournament_l483_483731

theorem chess_tournament (n k : ℕ)
  (hg : 2)
  (hs : 8)
  (hbk : k > 0):
  (∀ p : ℕ, p ≠ n → p ≠ 2 ∧ 8 + k * n = n * n + 3 * n − 14)
  → n = 7 ∨ n = 14 :=
sorry

end chess_tournament_l483_483731


namespace area_of_first_square_l483_483565

-- Definitions based on conditions
def perimeter_of_B := 16
def probability_not_in_B := 0.7538461538461538
def area_of_B := 16 -- Derived from perimeter calculation

-- Statement of the theorem
theorem area_of_first_square (A : ℝ) (h1 : perimeter_of_B = 16)
                             (h2 : probability_not_in_B = 0.7538461538461538):
  ((A - area_of_B) / A = probability_not_in_B) → A = 65 := 
by
  sorry

end area_of_first_square_l483_483565


namespace smallest_arithmetic_mean_divisible_by_1111_l483_483827

noncomputable def nine_consecutive_numbers {n : ℕ} : list ℕ :=
  [n, n+1, n+2, n+3, n+4, n+5, n+6, n+7, n+8]

def divisible_by (x y : ℕ) : Prop := ∃ k : ℕ, x = k * y

def arithmetic_mean {l : list ℕ} (h_len : l.length = 9) : ℚ :=
  (l.sum : ℚ) / 9

theorem smallest_arithmetic_mean_divisible_by_1111 :
  ∃ n : ℕ, 
  divisible_by ((nine_consecutive_numbers n).prod) 1111 ∧ 
  arithmetic_mean (by simp [nine_consecutive_numbers_len]) = 97 :=
sorry

end smallest_arithmetic_mean_divisible_by_1111_l483_483827


namespace ratio_sum_eq_l483_483420

variable {x y z : ℝ}

-- Conditions: 3x, 4y, 5z form a geometric sequence
def geom_sequence (x y z : ℝ) : Prop :=
  (∃ r : ℝ, 4 * y = 3 * x * r ∧ 5 * z = 4 * y * r)

-- Conditions: 1/x, 1/y, 1/z form an arithmetic sequence
def arith_sequence (x y z : ℝ) : Prop :=
  2 * x * z = y * z + x * y

-- Conclude: x/z + z/x = 34/15
theorem ratio_sum_eq (h1 : geom_sequence x y z) (h2 : arith_sequence x y z) : 
  (x / z + z / x) = (34 / 15) :=
sorry

end ratio_sum_eq_l483_483420


namespace range_of_a_l483_483354

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ∈ set.Icc 1 2 → |x + a| ≤ 2) ↔ a ∈ set.Icc (-3 : ℝ) 0 :=
by
  sorry

end range_of_a_l483_483354


namespace orthocenter_circumcenter_coincide_l483_483528

-- Definitions of points and triangle relations
variable {A B C : Point}
variable {A1 B1 C1 : Point}
variable (S I HA1B1C1 OIAIBC : Point)

noncomputable def incenter_medial_triangle (ABC : Triangle) : Point := S
noncomputable def symmetric_triangle (ABC : Triangle) (S : Point) : Triangle := ⟨A1, B1, C1⟩
noncomputable def orthocenter (T : Triangle) : Point := HA1B1C1
noncomputable def circumcenter_excentral_triangle (ABC : Triangle) : Point := OIAIBC

-- Main theorem
theorem orthocenter_circumcenter_coincide :
  let S := incenter_medial_triangle ⟨A, B, C⟩ in
  let A1B1C1 := symmetric_triangle ⟨A, B, C⟩ S in
  let HA1B1C1 := orthocenter A1B1C1 in
  let OIAIBC := circumcenter_excentral_triangle ⟨A, B, C⟩ in
  HA1B1C1 = OIAIBC := sorry

end orthocenter_circumcenter_coincide_l483_483528


namespace problem_l483_483247

noncomputable def f : ℝ → ℝ := sorry

theorem problem (x : ℝ) :
  (f (x + 2) + f x = 0) →
  (∀ x, f (-(x - 1)) = -f (x - 1)) →
  (
    (∀ e, ¬(e > 0 ∧ ∀ x, f (x + e) = f x)) ∧
    (∀ x, f (x + 1) = f (-x + 1)) ∧
    (¬(∀ x, f x = f (-x)))
  ) :=
by
  sorry

end problem_l483_483247


namespace eight_grade_students_condition_l483_483501

-- Define the problem variables and conditions
def number_of_participants (x : ℕ) : ℕ :=
x + 2

def total_games (x : ℕ) : ℕ :=
(x + 2) * (x + 1) / 2

def total_points (x : ℕ) : ℕ :=
total_games x

-- Given conditions setup
theorem eight_grade_students_condition (x y : ℕ) (hx : 2xy = x^2 + 3x - 14) :
x = 7 ∨ x = 14 :=
by
  sorry

end eight_grade_students_condition_l483_483501


namespace exists_x_in_open_interval_l483_483416

theorem exists_x_in_open_interval (n : ℕ) (a : Fin n → ℝ) (h : ∀ i, a i ∈ Icc 0 1) :
  ∃ x ∈ Ioo 0 1, ∑ i in Finset.univ, (1 / |x - a i|) ≤ 8 * n * ∑ i in Finset.range n, 1 / (2 * i + 1) :=
by
  sorry

end exists_x_in_open_interval_l483_483416


namespace angle_of_inclination_of_perpendicular_line_l483_483353

/-- Given that a line l is perpendicular to the horizontal line y = 1,
    we need to prove that the angle of inclination of line l is 90 degrees. -/
theorem angle_of_inclination_of_perpendicular_line (l : ℝ) (h : IsPerpendicular l) :
  angle_of_inclination l = 90 := 
by
  sorry -- proof not required

end angle_of_inclination_of_perpendicular_line_l483_483353


namespace red_pairs_count_l483_483736

def num_green_students : Nat := 63
def num_red_students : Nat := 69
def total_pairs : Nat := 66
def num_green_pairs : Nat := 27

theorem red_pairs_count : 
  (num_red_students - (num_green_students - num_green_pairs * 2)) / 2 = 30 := 
by sorry

end red_pairs_count_l483_483736


namespace smallest_arithmetic_mean_divisible_by_1111_l483_483822

theorem smallest_arithmetic_mean_divisible_by_1111 :
  ∃ (n : ℕ), (∀ i : ℕ, i < 9 → 1111 ∣ (n + i)) ∧ (n + 4 = 97) :=
by
  sorry

end smallest_arithmetic_mean_divisible_by_1111_l483_483822


namespace circle_area_difference_l483_483710

theorem circle_area_difference (r1 r2 : ℝ) (π : ℝ) (A1 A2 diff : ℝ) 
  (hr1 : r1 = 30)
  (hd2 : 2 * r2 = 30)
  (hA1 : A1 = π * r1^2)
  (hA2 : A2 = π * r2^2)
  (hdiff : diff = A1 - A2) :
  diff = 675 * π :=
by 
  sorry

end circle_area_difference_l483_483710


namespace find_a_l483_483663
-- Import the entire Mathlib to ensure all necessary primitives and theorems are available.

-- Define a constant equation representing the conditions.
def equation (x a : ℝ) := 3 * x + 2 * a

-- Define a theorem to prove the condition => result structure.
theorem find_a (h : equation 2 a = 0) : a = -3 :=
by sorry

end find_a_l483_483663


namespace f_at_neg_two_l483_483289

def is_odd_function (g : ℝ → ℝ) := ∀ x, g(-x) = -g(x)

theorem f_at_neg_two (f g : ℝ → ℝ) (h1 : ∀ x, f(x) = g(x) + 2) (h2 : is_odd_function(g)) (h3 : f(2) = 3) : f(-2) = 1 :=
by {
  sorry
}

end f_at_neg_two_l483_483289


namespace min_area_ratio_l483_483678

theorem min_area_ratio (A B C D E F : Point) (hABC : equilateral_triangle A B C) 
  (hD : D ∈ segment A B) (hE : E ∈ segment B C) (hF : F ∈ segment C A) 
  (hDEF : right_triangle D E F) (h_angle_DEF : angle D E F = 90) (h_angle_EDF : angle E D F = 30) :
  area_ratio S_triangle_DEF S_triangle_ABC = 3 / 14 := 
sorry

end min_area_ratio_l483_483678


namespace simplify_trig_expr_find_tan_alpha_l483_483149

-- Problem 1: Simplify the expression
theorem simplify_trig_expr (α : ℝ) :
  (sin (π - α) * cos (π + α) * sin (π / 2 + α)) / (sin (-α) * sin (3 * π / 2 + α)) = -cos α :=
by
  sorry

-- Problem 2: Find tan α
theorem find_tan_alpha (α : ℝ) (h0 : π / 2 < α) (h1 : α < π) (h2 : sin (π - α) + cos α = 7 / 13) :
  tan α = -12 / 5 :=
by
  sorry

end simplify_trig_expr_find_tan_alpha_l483_483149


namespace problem_part1_problem_part2_l483_483323

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  2 * (Real.sin (ω * x)) * (Real.cos (ω * x)) - 2 * Real.sqrt 3 * (Real.cos (ω * x))^2 + Real.sqrt 3

theorem problem_part1 (ω : ℝ) (k : ℤ) (x : ℝ) (hω : ω > 0) (hx : k * Real.pi - Real.pi / 12 ≤ x ∧ x ≤ k * Real.pi + 5 * Real.pi / 12) :
  monotone_on (f ω) (set.Icc (k * Real.pi - Real.pi / 12) (k * Real.pi + 5 * Real.pi / 12)) := sorry

theorem problem_part2 (A B C a b c : ℝ) (hC : 0 < C ∧ C < π / 2)
  (hfC : f 1 C = Real.sqrt 3) (hc : c = 3) (hB : Real.sin B = 2 * Real.sin A) :
  let area := 1 / 2 * a * b * Real.sin C in area = 3 * Real.sqrt 3 / 2 := sorry

end problem_part1_problem_part2_l483_483323


namespace locus_of_preposterous_midpoints_l483_483892

variables {A B X : Type} [MetricSpace ℝ X]

def preposterous_midpoint (A B X : X) : Prop :=
  ∃ (a1 a2 b1 b2 x1 x2 : ℝ), 
    a1 ≥ 0 ∧ a2 ≥ 0 ∧ b1 ≥ 0 ∧ b2 ≥ 0 ∧
    (A = (a1, a2) ∧ B = (b1, b2)) ∧
    (X = (x1, x2)) ∧
    (x1 = real.sqrt(a1 * b1)) ∧ (x2 = real.sqrt(a2 * b2))

theorem locus_of_preposterous_midpoints (A B : X) :
  ∀ X, preposterous_midpoint A B X ↔ 
  (dist A B / 2 ≥ dist (midpoint ℝ A B) X) ∧ 
  (X ≠ midpoint ℝ A B) :=
sorry

end locus_of_preposterous_midpoints_l483_483892


namespace ratio_of_inscribed_circle_segments_l483_483543

/-- A circle is inscribed in a triangle with side lengths 9, 12, and 15.
Let the segments of the side of length 9, made by a point of tangency, be r and s, with r < s.
Prove that the ratio r:s is 1:2. -/
theorem ratio_of_inscribed_circle_segments (r s : ℕ) (h : r < s) 
  (triangle_sides : r + s = 9) (point_of_tangency_15 : s + (12 - r) = 9) : 
  r / s = 1 / 2 := 
sorry

end ratio_of_inscribed_circle_segments_l483_483543


namespace fixed_point_trajectory_l483_483163

theorem fixed_point_trajectory (R : ℝ) :
  let r : ℝ := R / 2 in
  (∃ (K : ℝ × ℝ) (C₁ C₂ : ℝ × ℝ), 
    (circle K r).is_rolling_without_slipping (circle (0,0) R) K C₁ C₂) →
  (∃ (path : ℝ → ℝ × ℝ), path = λ θ, (0, θ*R) ∧ path ∈ diameter_path_set (circle (0,0) R)) :=
by 
  sorry

end fixed_point_trajectory_l483_483163


namespace parallelogram_rational_relation_l483_483005

theorem parallelogram_rational_relation
  (ABCD : Type) [parallelogram ABCD]
  (O : Type) [intersection O (diagonal A C) (diagonal B D)]
  (theta : Real)
  (angle_CAB : Real) (angle_CAB_eq : angle_CAB = 3 * theta)
  (angle_DBC : Real) (angle_DBC_eq : angle_DBC = 3 * theta)
  (angle_DBA : Real) (angle_DBA_eq : angle_DBA = theta) 
  (angle_AOB : Real) (angle_AOB_eq : angle_AOB = 180 - 4 * theta)
  (angle_ACB : Real) (angle_ACB_eq : angle_ACB = 180 - 7 * theta) :
  r = 5 / 8 :=
sorry

end parallelogram_rational_relation_l483_483005


namespace tank_capacity_correct_l483_483775

-- Define rates and times for each pipe
def rate_a : ℕ := 200 -- in liters per minute
def rate_b : ℕ := 50 -- in liters per minute
def rate_c : ℕ := 25 -- in liters per minute

def time_a : ℕ := 1 -- pipe A open time in minutes
def time_b : ℕ := 2 -- pipe B open time in minutes
def time_c : ℕ := 2 -- pipe C open time in minutes

def cycle_time : ℕ := time_a + time_b + time_c -- total time for one cycle in minutes
def total_time : ℕ := 40 -- total time to fill the tank in minutes

-- Net water added in one cycle
def net_water_in_cycle : ℕ :=
  (rate_a * time_a) + (rate_b * time_b) - (rate_c * time_c)

-- Number of cycles needed to fill the tank
def number_of_cycles : ℕ :=
  total_time / cycle_time

-- Total capacity of the tank
def tank_capacity : ℕ :=
  number_of_cycles * net_water_in_cycle

-- The hypothesis to prove
theorem tank_capacity_correct :
  tank_capacity = 2000 :=
  by
    sorry

end tank_capacity_correct_l483_483775


namespace smallest_arithmetic_mean_divisible_by_1111_l483_483823

theorem smallest_arithmetic_mean_divisible_by_1111 :
  ∃ (n : ℕ), (∀ i : ℕ, i < 9 → 1111 ∣ (n + i)) ∧ (n + 4 = 97) :=
by
  sorry

end smallest_arithmetic_mean_divisible_by_1111_l483_483823


namespace cistern_depth_l483_483961

theorem cistern_depth (h : ℝ) :
  (6 * 4 + 2 * (h * 6) + 2 * (h * 4) = 49) → (h = 1.25) :=
by
  sorry

end cistern_depth_l483_483961


namespace simplify_fraction_l483_483472

theorem simplify_fraction (m : ℝ) (h₁: m ≠ 0) (h₂: m ≠ 1): (m - 1) / m / ((m - 1) / (m * m)) = m := by
  sorry

end simplify_fraction_l483_483472


namespace proof_large_long_brown_dogs_l483_483730

open Finset

variable (D : Finset α) (L B La S : Finset α)

variables (hD : card D = 60)
variables (hL : card L = 35)
variables (hB : card B = 25)
variables (hNeitherLB : card (D \ (L ∪ B)) = 10)
variables (hLa : card La = 30)
variables (hS : card S = 30)
variables (hSmallBrown : card (S ∩ B) = 14)
variables (hLargeLongNotBrown : card (La ∩ L \ B) = 7)

theorem proof_large_long_brown_dogs : card (La ∩ L ∩ B) = 6 :=
  sorry

end proof_large_long_brown_dogs_l483_483730


namespace collinear_vectors_sum_l483_483036

theorem collinear_vectors_sum (x y : ℝ) (a b : ℝ × ℝ × ℝ) :
  a = (2 * x, 1, 3) →
  b = (1, -2 * y, 9) →
  ∃ λ : ℝ, a = (λ * 1, λ * (-2 * y), λ * 9) →
  x + y = -4 / 3 :=
by
  intros ha hb hcollinear
  cases ha
  cases hb
  -- Revealing the structure of tuples and defining the conditions
  have h1 : 2 * x = λ := by sorry
  have h2 : 1 = -2 * λ * y := by sorry
  have h3 : 3 = 9 * λ := by sorry
  -- Solving system and concluding with x + y = -4/3
  sorry

end collinear_vectors_sum_l483_483036


namespace chairs_difference_l483_483550

theorem chairs_difference : 
  ∀ (initial_chairs left_chairs : ℕ), 
  initial_chairs = 15 → 
  left_chairs = 3 → 
  initial_chairs - left_chairs = 12 := 
by
  intros initial_chairs left_chairs h1 h2
  rw [h1, h2]
  norm_num
  sorry

end chairs_difference_l483_483550


namespace baker_new_cakes_l483_483199

theorem baker_new_cakes :
  ∀ (initial_bought new_bought sold final : ℕ),
  initial_bought = 173 →
  sold = 86 →
  final = 190 →
  final = initial_bought + new_bought - sold →
  new_bought = 103 :=
by
  intros initial_bought new_bought sold final H_initial H_sold H_final H_eq
  sorry

end baker_new_cakes_l483_483199


namespace min_expression_value_l483_483718

theorem min_expression_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  ∃ x : ℝ, x = 5 ∧ ∀ y, (y = (b / (3 * a) + 3 / b)) → x ≤ y :=
by
  sorry

end min_expression_value_l483_483718


namespace inverse_f_l483_483484

noncomputable def f (x : ℝ) : ℝ := 2^(x + 1) - 1

theorem inverse_f (y : ℝ) (h : y > 1) : ∃ x > 0, f x = y ∧ (∀ x > 0, f x = y → x = log (y + 1) / log 2 - 1) := by
  sorry

end inverse_f_l483_483484


namespace g_inv_triple_three_l483_483886

def g : ℕ → ℕ
| 1 := 4
| 2 := 3
| 3 := 2
| 4 := 5
| 5 := 1
| _ := 0 -- Handling inputs outside the given domain for completeness

noncomputable def g_inv (y : ℕ) : ℕ :=
if h : ∃ x, g x = y then (classical.some h) else 0 -- Default to 0 if not found

theorem g_inv_triple_three : g_inv (g_inv (g_inv 3)) = 2 :=
by
  have : g 2 = 3 := rfl
  have : g_inv 3 = 2 := by
    simp [g, g_inv]
    exact if_pos ⟨2, rfl⟩
  rw [this]
  have : g 3 = 2 := rfl
  have : g_inv 2 = 3 := by
    simp [g, g_inv]
    exact if_pos ⟨3, rfl⟩
  rw [this]
  have : g 2 = 3 := rfl
  have : g_inv 3 = 2 := by
    simp [g, g_inv]
    exact if_pos ⟨2, rfl⟩
  rw [this]
  exact rfl

end g_inv_triple_three_l483_483886


namespace num_factors_2310_l483_483342

theorem num_factors_2310 : 
  let n : ℕ := 2310 in
  number_of_factors n = 32 :=
by
  sorry

end num_factors_2310_l483_483342


namespace two_digit_numbers_that_form_square_when_reversed_l483_483993

theorem two_digit_numbers_that_form_square_when_reversed :
  ∃ (S : set ℕ), S = {29, 38, 47, 56, 65, 74, 83, 92} ∧
  ∀ (n : ℕ), 10 ≤ n ∧ n < 100 →
  let tens := n / 10
      units := n % 10
      reversed := 10 * units + tens in
  (n + reversed) = k * k → n ∈ S
  where k : ℕ :=
sorry

end two_digit_numbers_that_form_square_when_reversed_l483_483993


namespace angle_CEF_is_45_l483_483009

theorem angle_CEF_is_45 (A B C D E F : Point) (h_square : square A B C D) 
  (h_F_on_BC : lies_on F B C) (h_equilateral_DEC : is_equilateral_triangle D E C) 
  (h_eb_eq_ef : dist E B = dist E F) : 
  measure_angle C E F = 45 :=
by sorry

end angle_CEF_is_45_l483_483009


namespace product_of_coordinates_of_D_l483_483658

theorem product_of_coordinates_of_D (D : ℝ × ℝ) (N : ℝ × ℝ) (C : ℝ × ℝ) 
  (hN : N = (4, 3)) (hC : C = (5, -1)) (midpoint : N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) :
  D.1 * D.2 = 21 :=
by
  sorry

end product_of_coordinates_of_D_l483_483658


namespace find_equation_of_ellipse_find_k1_k2_product_l483_483297

-- Given conditions
def is_ellipse (a b : ℝ) : Prop :=
  a > b ∧ b > 0

def focal_length (a b : ℝ) : Prop :=
  2 * real.sqrt (a^2 - b^2) = 2 * real.sqrt 3

def ellipse_passes_through_point (a b x y : ℝ) : Prop :=
  x = real.sqrt 3 ∧ y = 1/2 ∧ (x^2 / a^2 + y^2 / b^2 = 1)

def line_perpendicular_to_y_axis (k1 k2 : ℝ) : Prop :=
  ∀ (x y : ℝ), x = 2 - 8 * k1^2 / (1 + 4 * k1^2) → y = 4 * k1 / (1 + 4 * k1^2) →
               x = 2 - 8 * k2^2 / (1 + 4 * k2^2) → y = 4 * k2 / (1 + 4 * k2^2) →
               y = 0

-- To be proven
theorem find_equation_of_ellipse (a b : ℝ) (hab : is_ellipse a b)
  (hfocal : focal_length a b)
  (hpass : ellipse_passes_through_point a b (real.sqrt 3) (1 / 2)) :
  (4 : ℝ) = a^2 ∧ (1 : ℝ) = b^2 ∧ (∀ x y, (x^2 / 4 + y^2 = 1) ↔ (x^2 / a^2 + y^2 / b^2 = 1)) :=
sorry

theorem find_k1_k2_product (a b k1 k2 : ℝ) (hab : is_ellipse a b)
  (hfocal : focal_length a b)
  (hpass : ellipse_passes_through_point a b (real.sqrt 3) (1 / 2))
  (h_perpendicular : line_perpendicular_to_y_axis k1 k2) :
  k1 * k2 = 1 / 4 :=
sorry

end find_equation_of_ellipse_find_k1_k2_product_l483_483297


namespace triangle_CD_length_l483_483016

theorem triangle_CD_length 
  (A B C D : Type) [metric_space A] [add_group B] [module ℝ B] 
  (triangle_ABC : triangle A B C)
  (D_on_BC : D ∈ segment ℝ B C)
  (angle_BAC_eq_ADC : ∠ BAC = ∠ ADC)
  (AC_eq_8 : ∥ - B + C ∥ = 8)
  (BC_eq_16 : ∥ B - C ∥ = 16) : 
  ∥ D - C ∥ = 4 := 
sorry

end triangle_CD_length_l483_483016


namespace distinguishable_rearrangements_l483_483344

-- Define the conditions from the problem
def vowels := multiset.of_list ['O', 'I', 'U', 'O']
def consonants := multiset.of_list ['S', 'L', 'T', 'N', 'S', 'S']

-- Calculate the number of arrangements considering the indistinguishability of repeated elements
def arrangements_vowels : ℕ := (multiset.card vowels).factorial / (multiset.count 'O' vowels).factorial
def arrangements_consonants : ℕ := (multiset.card consonants).factorial / (multiset.count 'S' consonants).factorial

-- Prove the total number of distinguishable arrangements where vowels come first
theorem distinguishable_rearrangements : arrangements_vowels * arrangements_consonants = 1440 :=
by { have vowels_v : arrangements_vowels = 12 := by simp [vowels, arrangements_vowels],
     have consonants_c : arrangements_consonants = 120 := by simp [consonants, arrangements_consonants],
     rw [vowels_v, consonants_c],
     norm_num, sorry }

end distinguishable_rearrangements_l483_483344


namespace _l483_483042

noncomputable def sin_cos_ratio_theorem (u v : ℝ)
  (h1 : sin u / sin v = 4) 
  (h2 : cos u / cos v = 1/3) : 
  sin (2 * u) / sin (2 * v) + cos (2 * u) / cos (2 * v) = 19 / 381 :=
sorry

end _l483_483042


namespace triangle_is_obtuse_l483_483389

theorem triangle_is_obtuse {A B C : Type}
  [InnerProductSpace ℝ (EuclideanSpace ℝ (Fin 2))]
  (h1 : IsTriangle (A, B, C))
  (h2 : (A - B) ⬝ (C - B) > 0) : IsObtuse (A, B, C) :=
  sorry

end triangle_is_obtuse_l483_483389


namespace part1_part2_l483_483033

def set_A (a : ℝ) : set ℝ := {x | 2 - a ≤ x ∧ x ≤ 2 + a}
def set_B : set ℝ := {x | (x - 1) * (x - 4) ≥ 0}

theorem part1 (a : ℝ) (ha : a = 3) : 
  (set_A a) ∩ set_B = {x | -1 ≤ x ∧ x ≤ 1} ∪ {x | 4 ≤ x ∧ x ≤ 5} :=
by sorry

theorem part2 (a : ℝ) (ha : 0 < a) : 
  (set_A a ∩ set_B = ∅) → (0 < a ∧ a < 1) :=
by sorry

end part1_part2_l483_483033


namespace find_a_l483_483350

theorem find_a (x y a : ℝ) (h1 : 4 * x + y = 8) (h2 : 3 * x - 4 * y = 5) (h3 : a * x - 3 * y = 23) : 
  a = 12.141 :=
by
  sorry

end find_a_l483_483350


namespace smallest_arithmetic_mean_divisible_by_1111_l483_483854

/-- 
Given the product of nine consecutive natural numbers is divisible by 1111, 
prove that the smallest possible value of the arithmetic mean of these nine numbers is 97.
-/
theorem smallest_arithmetic_mean_divisible_by_1111 :
  ∃ n : ℕ, (∀ k : ℕ, k = n →  (∏ i in finset.range 9, k + i) % 1111 = 0) 
  ∧ (n ≥ 93) ∧ (n + 4 = 97) :=
sorry

end smallest_arithmetic_mean_divisible_by_1111_l483_483854


namespace tangent_line_equal_intercepts_l483_483268

theorem tangent_line_equal_intercepts
    (L : Real → Real → Prop)
    (circle : Real → Real → Prop := λ x y, x^2 + (y - 2)^2 = 2)
    (eq_intercepts : ∀ x y, L x y → x = y ∨ y = -x ∨ y = -x + 4)
    : ∀ x y, L x y ↔ ((L x y → circle x y) ∧ (L x y → (y = x ∨ y = -x ∨ y = -x + 4))) :=
by
  sorry

end tangent_line_equal_intercepts_l483_483268


namespace triangle_area_relation_l483_483724

-- Define the assumptions of the problem
variables (A B C H M : Type)
variable [MetricSpace A]
variable [MetricSpace B]
variable [MetricSpace C]
variable [MetricSpace H]
variable [MetricSpace M]
variable (AB : Real)
variable (angle_ABC : Real)
variable (area_CHM : Real)
variable (area_ABC : Real)

-- Assume the conditions given in the problem
theorem triangle_area_relation (h1 : angle_ABC = 60)
                               (h2 : isAltitude A B C H)
                               (h3 : isMedian A B C M)
                               (h4 : trisectsAngleAt C B A H M)
                               (h5 : area_CHM = K) :
  area_ABC = 4 * K := sorry

end triangle_area_relation_l483_483724


namespace find_matrix_l483_483273

theorem find_matrix (M : Matrix (Fin 2) (Fin 2) ℝ)
  (h : ∀ v : Matrix (Fin 2) (Fin 1) ℝ, M.mul_vec v = (-7 : ℝ) • v) :
  M = !![-7, 0; 0, -7] :=
by
  sorry

end find_matrix_l483_483273


namespace angle_CED_135_l483_483116

theorem angle_CED_135 (A B C D E : Point) (r : ℝ) 
  (h1 : dist A B = r)
  (h2 : dist B E = 2 * r) 
  (h3 : dist A E = r) 
  (h4 : dist A C = r) 
  (h5 : dist B D = 2 * r)
  (h6 : E ∈ circle A r)
  (h7 : E ∈ circle B (2 * r))
  (h8 : collinear {A, B, C})
  (h9 : collinear {A, B, D})
  (h10 : E ≠ A ∧ E ≠ B ∧ E ≠ C ∧ E ≠ D) :
  ∠ C E D = 135 := 
by
  sorry

end angle_CED_135_l483_483116


namespace solution1_solution2_l483_483209

noncomputable def problem1 : ℝ :=
  real.sqrt 16 + 2 * real.sqrt 9 - real.cbrt 27

theorem solution1 : problem1 = 7 :=
by sorry

noncomputable def problem2 : ℝ :=
  abs (1 - real.sqrt 2) + real.sqrt 4 - real.cbrt (-8)

theorem solution2 : problem2 = real.sqrt 2 + 3 :=
by sorry

end solution1_solution2_l483_483209


namespace smallest_arithmetic_mean_divisible_1111_l483_483846

theorem smallest_arithmetic_mean_divisible_1111 :
  ∃ n : ℕ, 93 ≤ n ∧ n + 4 = 97 ∧ (∀ i : ℕ, i ∈ finset.range 9 → (n + i) % 11 = 0 ∨ (n + i) % 101 = 0) :=
sorry

end smallest_arithmetic_mean_divisible_1111_l483_483846


namespace area_O₁O₂O₃O₄_le_one_l483_483755

-- Definition of the unit square and conditions
structure Point :=
(x : ℝ) (y : ℝ)

def unit_square := {A B C D : Point // 
  A.x = 0 ∧ A.y = 0 ∧ 
  B.x = 1 ∧ B.y = 0 ∧ 
  C.x = 1 ∧ C.y = 1 ∧ 
  D.x = 0 ∧ D.y = 1}

-- Given conditions
variable (A B C D E F G H : Point)
variable (O₁ O₂ O₃ O₄ : Point)
variable (ABC_square : unit_square)
variable (hE : ∠AEB = 90)
variable (hF : ∠BFC = 90)
variable (hG : ∠CGD = 90)
variable (hH : ∠DHA = 90)
variable (incenters : 
  (O₁ = incenter A E B) ∧ 
  (O₂ = incenter B F C) ∧ 
  (O₃ = incenter C G D) ∧ 
  (O₄ = incenter D H A))

noncomputable def area_quadrilateral (P Q R S : Point) : ℝ :=
  abs (0.5 * ((Q.x - P.x) * (R.y - P.y) - (R.x - P.x) * (Q.y - P.y)) + 
            ((R.x - P.x) * (S.y - P.y) - (S.x - P.x) * (R.y - P.y)))

theorem area_O₁O₂O₃O₄_le_one :
  area_quadrilateral O₁ O₂ O₃ O₄ ≤ 1 := 
sorry

end area_O₁O₂O₃O₄_le_one_l483_483755


namespace green_surface_fraction_l483_483964

-- Conditions definitions
def num_small_cubes : ℕ := 64
def num_green_cubes : ℕ := 14
def num_blue_cubes : ℕ := num_small_cubes - num_green_cubes
def side_length_large_cube : ℕ := 4
def side_length_small_cube : ℕ := 1
def surface_area_large_cube : ℕ := 6 * side_length_large_cube^2

-- The statement to be proved
theorem green_surface_fraction : (num_green_cubes + num_blue_cubes = num_small_cubes) →
  (num_green_cubes = 14 ∧ num_blue_cubes = 50) →
  (side_length_large_cube = 4) →
  (side_length_small_cube = 1) →
  ((6 * side_length_large_cube^2) = 96) →
  Σ_i (cond : green_cube_position i), (exposed_faces i = 1) →
  (exposed_green_surface_area : ℕ) →
  (fraction_green : ℚ) →
  (fraction_green = exposed_green_surface_area / surface_area_large_cube) →
  (fraction_green = 1 / 16) :=
by {
  -- Leave the proof part empty using sorry
  sorry
}

end green_surface_fraction_l483_483964


namespace sum_sequence_eq_l483_483254

-- Define the sequence
def sequence (n k : ℕ) : ℚ :=
  if k = 1 then 1 else (n - (k - 1)) / n

-- Define the sum of the sequence
def sum_sequence (n : ℕ) : ℚ :=
  1 + (Finset.sum (Finset.range (n - 1)) (λ k, (n - (k + 1)) / n))

-- Main theorem stating the problem
theorem sum_sequence_eq (n : ℕ) (hn : n > 0) : sum_sequence n = (n + 1) / 2 :=
by
  sorry

end sum_sequence_eq_l483_483254


namespace coloring_ways_l483_483805

theorem coloring_ways : 
  ∃ (A B : Finset ℕ) (hA : A ⊆ Finset.range 21) (hB : B ⊆ Finset.range 21),
  (∀ a ∈ A, (∀ b ∈ B, (a, b) ∈ Finset.disjoint_pairs)) ∧
  (∀ p : ℕ, p ∈ A ∨ p ∈ B) ∧
  ((∃ a ∈ A, true) ∧ (∃ b ∈ B, true)) ∧
  ∃ n : ℕ, n = 64 :=
sorry

end coloring_ways_l483_483805


namespace smallest_arithmetic_mean_divisible_1111_l483_483844

theorem smallest_arithmetic_mean_divisible_1111 :
  ∃ n : ℕ, 93 ≤ n ∧ n + 4 = 97 ∧ (∀ i : ℕ, i ∈ finset.range 9 → (n + i) % 11 = 0 ∨ (n + i) % 101 = 0) :=
sorry

end smallest_arithmetic_mean_divisible_1111_l483_483844


namespace range_of_a_l483_483312

theorem range_of_a {a : ℝ} (f : ℝ → ℝ) (h : f = λ x, (1/3 : ℝ) * x^3 - (a/2) * x^2 + x - 3) :
  (∀ x, deriv f x = x^2 - a * x + 1) ∧ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ deriv f x1 = 0 ∧ deriv f x2 = 0) →
  a ∈ set.Ioo (2 : ℝ) (⊤ : ℝ) ∪ set.Iio (-2 : ℝ) :=
begin
  sorry
end

end range_of_a_l483_483312


namespace min_arithmetic_mean_of_consecutive_naturals_div_by_1111_l483_483863

theorem min_arithmetic_mean_of_consecutive_naturals_div_by_1111 :
  ∀ a : ℕ, (∃ k, (a * (a + 1) * (a + 2) * (a + 3) * (a + 4) * (a + 5) * (a + 6) * (a + 7) * (a + 8)) = 1111 * k) →
  (a + 4 ≥ 97) :=
sorry

end min_arithmetic_mean_of_consecutive_naturals_div_by_1111_l483_483863


namespace smallest_four_digit_mod_8_l483_483910

theorem smallest_four_digit_mod_8 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 8 = 5 ∧ (∀ m : ℕ, m >= 1000 ∧ m < 10000 ∧ m % 8 = 5 → n ≤ m) → n = 1005 :=
by
  sorry

end smallest_four_digit_mod_8_l483_483910


namespace irrational_square_irrational_l483_483095

theorem irrational_square_irrational :
  ¬(∃ (x : ℝ), irrational x ∧ ∃ (y : ℚ), x^2 = y) ↔ ∀ (z : ℝ), irrational z → irrational (z^2) :=
begin
  sorry
end

end irrational_square_irrational_l483_483095


namespace distinguishable_large_triangle_count_l483_483498

theorem distinguishable_large_triangle_count :
  ∃ (num_colors : ℕ) (n : ℕ), num_colors = 7 ∧ n = (84 * num_colors) ∧ n = 588 :=
by
  have num_colors := 7
  have distinct_corners := 84
  have center_choices := num_colors
  let n := distinct_corners * center_choices
  use [num_colors, n]
  split
  · exact rfl
  split
  · unfold n
    exact rfl
  · unfold n
    exact rfl

end distinguishable_large_triangle_count_l483_483498


namespace no_solution_if_and_only_if_zero_l483_483348

theorem no_solution_if_and_only_if_zero (n : ℝ) :
  ¬(∃ (x y z : ℝ), 2 * n * x + y = 2 ∧ 3 * n * y + z = 3 ∧ x + 2 * n * z = 2) ↔ n = 0 := 
  by
  sorry

end no_solution_if_and_only_if_zero_l483_483348


namespace robert_teddy_total_spent_l483_483787

-- Definitions from conditions in a)
def pizza_price_per_box := 10
def pizza_quantity := 5
def pizza_discount_rate := 0.15
def soft_drink_price_per_can := 1.50
def soft_drink_quantity := 20
def hamburger_price_per := 3
def hamburger_quantity := 6
def hamburger_discount_rate := 0.10
def tax_rate := 0.20

-- Statement of the proof problem
theorem robert_teddy_total_spent : 
    let pizza_cost_without_discount := pizza_quantity * pizza_price_per_box,
        pizza_discount := if pizza_quantity > 3 then pizza_cost_without_discount * pizza_discount_rate else 0,
        pizza_cost_with_discount := pizza_cost_without_discount - pizza_discount,
        
        soft_drink_cost := soft_drink_quantity * soft_drink_price_per_can,
        
        hamburger_cost_without_discount := hamburger_quantity * hamburger_price_per,
        hamburger_discount := if hamburger_quantity > 5 then hamburger_cost_without_discount * hamburger_discount_rate else 0,
        hamburger_cost_with_discount := hamburger_cost_without_discount - hamburger_discount,
        
        total_cost_before_tax := pizza_cost_with_discount + soft_drink_cost + hamburger_cost_with_discount,
        tax := total_cost_before_tax * tax_rate,
        total_cost_including_tax := total_cost_before_tax + tax
    in 
    total_cost_including_tax = 106.44 :=
by
    -- Sorry to skip the actual proof
    sorry

end robert_teddy_total_spent_l483_483787
