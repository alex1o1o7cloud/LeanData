import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Trigonometry.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.Trigonometry.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Modular
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.GCD
import Mathlib.Data.Nat.Parity
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.AngleOrthogonal
import Mathlib.LinearAlgebra.Basic
import Mathlib.MeasureTheory.ProbabilityMassFunction
import Mathlib.NumberTheory.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Real.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Topology.Basic
import Real
import data.nat.prime

namespace solve_system_l800_800563

theorem solve_system (x y : ℝ) (h1 : 5 * x + y = 19) (h2 : x + 3 * y = 1) : 3 * x + 2 * y = 10 :=
by
  sorry

end solve_system_l800_800563


namespace equidistant_from_all_vertices_l800_800607

-- Let O be a point inside a convex polygon such that it forms isosceles triangles with every pair of its vertices.
-- Prove that O is equidistant from all vertices of the polygon.

variable {O : Type*} [EuclideanGeometry O]
variable {V : Type*} [VertexGeometry V] (polygon : ConvexPolygon V)
variable (O : Point O) (vertices : List (Point V))

def forms_isosceles_with_each_pair (O : Point O) (vertices : List (Point V)) : Prop :=
  ∀ (A B : Point V), A ∈ vertices → B ∈ vertices → A ≠ B → (dist O A = dist O B ∨ dist O A = dist A B ∨ dist O B = dist O A)

theorem equidistant_from_all_vertices
  (h1 : Inside O polygon)
  (h2 : forms_isosceles_with_each_pair O vertices) :
  ∀ (A B : Point V), A ∈ vertices → B ∈ vertices → dist O A = dist O B := sorry

end equidistant_from_all_vertices_l800_800607


namespace first_line_equation_l800_800033

theorem first_line_equation (x y : ℝ) :
  (∃ (y x: ℝ) (H_eq : y = x) (H_x_eq : x = -6) (H_area : (1/2) * 6 * 6 = 18), H_eq ∧ H_x_eq ∧ H_area) →
  y = x :=
by
  intros
  sorry

end first_line_equation_l800_800033


namespace determine_nature_of_a_and_b_l800_800179

variable (a b : ℂ)

theorem determine_nature_of_a_and_b (h1 : 2 * a^2 + a * b + 2 * b^2 = 0) 
                                   (h2 : a + 2 * b = 5) : 
                                   (a ∈ ℂ) ∧ (b ∈ ℂ) :=
by 
  sorry

end determine_nature_of_a_and_b_l800_800179


namespace cone_height_is_six_l800_800237

noncomputable def cylinder_volume (r h : ℝ) : ℝ :=
  π * r^2 * h

noncomputable def cone_volume (r h : ℝ) : ℝ :=
  (1/3) * π * r^2 * h

theorem cone_height_is_six :
  ∀ (r h : ℝ),
  (h = 6) →
  (cylinder_volume 2 6 = cone_volume r (r * sqrt 3)) →
  (r = 2 * sqrt 3) →
  (r * sqrt 3 = 6) :=
by
  intro r h h_cylinder
  intro vol_eq
  intro r_eq
  simp at *
  sorry

end cone_height_is_six_l800_800237


namespace tangent_line_slope_through_origin_l800_800103

theorem tangent_line_slope_through_origin :
  (∃ a : ℝ, (a^3 + a + 16 = (3 * a^2 + 1) * a ∧ a = 2)) →
  (3 * (2 : ℝ)^2 + 1 = 13) :=
by
  intro h
  -- Detailed proof goes here
  sorry

end tangent_line_slope_through_origin_l800_800103


namespace least_number_remainder_4_l800_800539

theorem least_number_remainder_4 : ∃ n, n % 7 = 4 ∧ n % 9 = 4 ∧ n % 12 = 4 ∧ n % 18 = 4 ∧ n = 256 :=
by
  use 256
  repeat' 
    split
    norm_num
  sorry

end least_number_remainder_4_l800_800539


namespace largest_valid_four_digit_number_l800_800062

-- Definition of the problem conditions
def is_valid_number (a b c d : ℕ) : Prop :=
  c = a + b ∧ d = b + c

-- Proposition that we need to prove
theorem largest_valid_four_digit_number : ∃ (a b c d : ℕ),
  is_valid_number a b c d ∧ a * 1000 + b * 100 + c * 10 + d = 9099 :=
by
  sorry

end largest_valid_four_digit_number_l800_800062


namespace route_length_l800_800513

theorem route_length (D : ℝ) (T : ℝ) 
  (hx : T = 400 / D) 
  (hy : 80 = (D / 5) * T) 
  (hz : 80 + (D / 4) * T = D) : 
  D = 180 :=
by
  sorry

end route_length_l800_800513


namespace correct_statement_l800_800091

def statement_A : Prop := ∃ x : ℝ, x^2 = 9 ∧ x = 3
def statement_B : Prop := ∀ y : ℝ, ∃ x : ℝ, x^3 = y
def statement_C : Prop := ∀ (T : Type) [triang T], ∀ (G C : Point T), is_centroid G T → (distance G C) = (distance G T) 
def statement_D : Prop := ∀ (t1 t2 : Triangle), (∃ (a b a' b' : ℝ), a ≠ b ∧ a' ≠ b' ∧ congruent_side t1 a b ∧ congruent_side t2 a' b' ∧ angle_between a b = angle_between a' b') → (t1 ≃ t2)

theorem correct_statement : statement_B :=
by sorry

end correct_statement_l800_800091


namespace relationship_p_q_no_linear_term_l800_800765

theorem relationship_p_q_no_linear_term (p q : ℝ) :
  (∀ x : ℝ, (x^2 - p * x + q) * (x - 3) = x^3 + (-p - 3) * x^2 + (3 * p + q) * x - 3 * q) 
  → (3 * p + q = 0) → (q + 3 * p = 0) :=
by
  intro h_expansion coeff_zero
  sorry

end relationship_p_q_no_linear_term_l800_800765


namespace angle_A_in_acute_triangle_ABC_l800_800884

theorem angle_A_in_acute_triangle_ABC (A B C O H : Type)
  [triangle : is_acute_triangle A B C]
  (circumcenter_O : is_circumcenter A B C O)
  (orthocenter_H : is_orthocenter A B C H)
  (equidistant : dist A O = dist A H) :
  angle A = 60 :=
sorry

end angle_A_in_acute_triangle_ABC_l800_800884


namespace line_AB_midpoint_P_cond_1_line_AB_midpoint_line_l800_800362

section Geometry

variables {x y : ℝ}

-- Definitions for the conditions
def ray_OA (x y : ℝ) : Prop := x - y = 0 ∧ 0 ≤ x
def ray_OB (x y : ℝ) : Prop := x + √3 * y = 0 ∧ 0 ≤ x
def passes_through_P (x y : ℝ) : Prop := y = (2 - √3) * (x - 1)

-- The first part: Proving the equation of line AB when the midpoint is P(1,0)
theorem line_AB_midpoint_P_cond_1 (A B : ℝ × ℝ) (x y : ℝ) (hx : A.1 - A.2 = 0 ∧ 0 ≤ A.1)
  (hy : A.1 + √3 * A.2 = 0 ∧ 0 ≤ A.1) 
  (midpoint := (A.1 + B.1, A.2 + B.2) / 2)
  (midpoint_P : midpoint = (1, 0)) :
  passes_through_P 1 0 ∧ passes_through_P A.1 A.2 ∧ passes_through_P B.1 B.2 :=
by
  sorry

-- The second part: Proving the equation of line AB when the midpoint lies on x - 2y = 0
theorem line_AB_midpoint_line (k : ℝ)
  (midpoint_cond : ((k / (k - 1) + √3 * k / (1 + √3 * k)) / 2) - 2 * ((k / (k - 1) - k / (1 + √3 * k)) / 2) = 0) 
  (line_eqAB : y = k * (x - 1)) :
  line_eqAB = 3 * x - (3 - √3) * y - 3 :=
by
  sorry

end Geometry

end line_AB_midpoint_P_cond_1_line_AB_midpoint_line_l800_800362


namespace fraction_calculation_l800_800968

theorem fraction_calculation :
  let a := (1 / 2) + (1 / 3)
  let b := (2 / 7) + (1 / 4)
  ((a / b) * (3 / 5)) = (14 / 15) :=
by
  sorry

end fraction_calculation_l800_800968


namespace probability_2x2_is_half_l800_800604

noncomputable def probability_2x2_between_0_and_half : ℝ :=
  let μ := MeasureTheory.Measure.dirac (Set.Icc (-1 : ℝ) 1) in
  μ.measure (λ x, 0 ≤ 2 * x ^ 2 ∧ 2 * x ^ 2 ≤ 1 / 2)

theorem probability_2x2_is_half :
  probability_2x2_between_0_and_half = 1 / 2 :=
by
  sorry

end probability_2x2_is_half_l800_800604


namespace samantha_laundromat_cost_l800_800162

-- Definitions of given conditions
def washer_cost : ℕ := 4
def dryer_cost_per_10_min : ℝ := 0.25
def num_washes : ℕ := 2
def num_dryers : ℕ := 3
def dryer_time : ℕ := 40

-- Calculate total cost
def washing_cost : ℝ := washer_cost * num_washes
def intervals_10min : ℕ := dryer_time / 10
def single_dryer_cost : ℝ := dryer_cost_per_10_min * intervals_10min
def total_drying_cost : ℝ := single_dryer_cost * num_dryers
def total_cost : ℝ := washing_cost + total_drying_cost

-- The statement to prove
theorem samantha_laundromat_cost : total_cost = 11 :=
by
  unfold washer_cost dryer_cost_per_10_min num_washes num_dryers dryer_time washing_cost intervals_10min single_dryer_cost total_drying_cost total_cost
  norm_num
  done

end samantha_laundromat_cost_l800_800162


namespace solve_x_l800_800954

variable {α : Real}
variable {x : Real}
variables (AB CD MN BK : Real)
variable (S_ABCD : Real)
variable (angle_A γ : Real)

-- Given conditions
axiom (H1 : AB = CD)
axiom (H2 : angle_A < π / 2) -- acute angle
axiom (H3 : AB = 2 * x)
axiom (H4 : BK = 2 * x * Real.sin α)
axiom (H5 : MN = x / Real.sin α)
axiom (H6 : S_ABCD = MN * BK)

-- Proving x = 15
theorem solve_x :
  (2 * x^2 = 450) → (x = 15) := by
  sorry

end solve_x_l800_800954


namespace tan_A_of_right_triangle_l800_800784

theorem tan_A_of_right_triangle (a b : ℝ) (A B C : Type) [Inhabited  A] [Inhabited B] 
  [Inhabited C]
  (h_right : ∠C = 90) (h_a : a = 3 * b) :
  tan A = 3 := 
  sorry

end tan_A_of_right_triangle_l800_800784


namespace train_passing_time_l800_800952

-- Condition Definitions
def train_length : ℤ := 385
def bridge_length : ℤ := 140
def train_speed_kmh : ℤ := 45

-- Conversion factor from km/h to m/s
def kmh_to_ms (speed_kmh : ℤ) : ℤ := (speed_kmh * 1000) / 3600

-- Given the conditions, we want to prove the time taken to pass the bridge
theorem train_passing_time :
  let total_distance := train_length + bridge_length,
      train_speed := kmh_to_ms train_speed_kmh,
      passing_time := total_distance / train_speed
  in passing_time = 42 := by
  -- introduction of variables
  let total_distance := train_length + bridge_length
  let train_speed := kmh_to_ms train_speed_kmh
  let passing_time := total_distance / train_speed
  -- placeholder for proof
  sorry

end train_passing_time_l800_800952


namespace specific_integers_exist_l800_800201

theorem specific_integers_exist (a b : ℕ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) :
  ¬ 7 ∣ (a * b * (a + b)) ∧ 7^7 ∣ (a + b)^7 - a^7 - b^7 :=
by
  use 18
  use 1
  split
  {
    sorry -- Proof that 7 does not divide 18 * 1 * (18 + 1)
  }
  {
    sorry -- Proof that 7^7 divides (18 + 1)^7 - 18^7 - 1^7
  }

end specific_integers_exist_l800_800201


namespace minimal_APR_bank_A_l800_800037

def nominal_interest_rate_A : Float := 0.05
def nominal_interest_rate_B : Float := 0.055
def nominal_interest_rate_C : Float := 0.06

def compounding_periods_A : ℕ := 4
def compounding_periods_B : ℕ := 2
def compounding_periods_C : ℕ := 12

def effective_annual_rate (nom_rate : Float) (n : ℕ) : Float :=
  (1 + nom_rate / n.toFloat)^n.toFloat - 1

def APR_A := effective_annual_rate nominal_interest_rate_A compounding_periods_A
def APR_B := effective_annual_rate nominal_interest_rate_B compounding_periods_B
def APR_C := effective_annual_rate nominal_interest_rate_C compounding_periods_C

theorem minimal_APR_bank_A :
  APR_A < APR_B ∧ APR_A < APR_C ∧ APR_A = 0.050945 :=
by
  sorry

end minimal_APR_bank_A_l800_800037


namespace cans_of_type_B_purchased_l800_800440

variable (T P R : ℕ)

-- Conditions
def cost_per_can_A : ℕ := P / T
def cost_per_can_B : ℕ := 2 * cost_per_can_A T P
def quarters_in_dollar : ℕ := 4

-- Question and proof target
theorem cans_of_type_B_purchased (T P R : ℕ) (hT : T > 0) (hP : P > 0) (hR : R > 0) :
  (4 * R) / (2 * P / T) = 2 * R * T / P :=
by
  sorry

end cans_of_type_B_purchased_l800_800440


namespace power_function_evaluation_l800_800255

theorem power_function_evaluation (f : ℝ → ℝ) (α : ℝ) (h : ∀ x, f x = x ^ α) (h_point : f 4 = 2) : f 16 = 4 :=
by
  sorry

end power_function_evaluation_l800_800255


namespace probability_of_correct_match_l800_800614

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def total_possible_arrangements : ℕ :=
  factorial 4

def correct_arrangements : ℕ :=
  1

def probability_correct_match : ℚ :=
  correct_arrangements / total_possible_arrangements

theorem probability_of_correct_match : probability_correct_match = 1 / 24 :=
by
  -- Proof is omitted
  sorry

end probability_of_correct_match_l800_800614


namespace area_face_ABC_l800_800240

-- Noncomputable theory to handle the given mathematical context involving noncomputable reals
noncomputable theory

-- Definition of the tetrahedron and necessary conditions
structure Tetrahedron :=
(A B C D : ℝ3) -- We consider points in 3-dimensional space

-- Conditions
structure Conditions where
  AC_perp_BD : (AC ⬝ BD = 0)
  AD_perp_BC : (AD ⬝ BC = 0)
  AB_eq_CD : dist A B = dist C D
  edges_touch_sphere : ∃ r : ℝ, ∀ p q ∈ {A, B, C, D}, dist p q < r ∧ dist p q > 0

-- The theorem statement
theorem area_face_ABC (T : Tetrahedron) (h : Conditions T) (r : ℝ) :
    ∃ A B C D : ℝ3, (h.AC_perp_BD) ∧ (h.AD_perp_BC) ∧ (h.AB_eq_CD) ∧ (h.edges_touch_sphere) ∧
    area_face (A B C) = 2 * r^2 * sqrt(3) := sorry

end area_face_ABC_l800_800240


namespace plane_through_points_eq_l800_800672

-- Define the points M, N, P
def M := (1, 2, 0)
def N := (1, -1, 2)
def P := (0, 1, -1)

-- Define the target plane equation
def target_plane_eq (x y z : ℝ) := 5 * x - 2 * y + 3 * z - 1 = 0

-- Main theorem statement
theorem plane_through_points_eq :
  ∀ (x y z : ℝ),
    (∃ A B C : ℝ,
      A * (x - 1) + B * (y - 2) + C * z = 0 ∧
      A * (1 - 1) + B * (-1 - 2) + C * (2 - 0) = 0 ∧
      A * (0 - 1) + B * (1 - 2) + C * (-1 - 0) = 0) →
    target_plane_eq x y z :=
by
  sorry

end plane_through_points_eq_l800_800672


namespace mass_of_man_l800_800931

theorem mass_of_man
  (L : ℝ) (B : ℝ) (h : ℝ)
  (ρ : ℝ) 
  (boat_length : L = 7) 
  (boat_breadth : B = 3) 
  (sink_height : h = 0.01) 
  (water_density : ρ = 1000) :
  let V := L * B * h in
  let m := ρ * V in
  m = 210 := 
by
  sorry

end mass_of_man_l800_800931


namespace parallel_vectors_implies_m_eq_neg1_l800_800294

theorem parallel_vectors_implies_m_eq_neg1 (m : ℝ) :
  let a := (m, -1)
  let b := (1, m + 2)
  a.1 * b.2 = a.2 * b.1 → m = -1 :=
by
  intro h
  sorry

end parallel_vectors_implies_m_eq_neg1_l800_800294


namespace math_olympiad_proof_l800_800350

theorem math_olympiad_proof (scores : Fin 20 → ℕ) 
  (h_diff : ∀ i j, i ≠ j → scores i ≠ scores j) 
  (h_sum : ∀ i j k, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k) : 
  ∀ i, scores i > 18 :=
by
  sorry

end math_olympiad_proof_l800_800350


namespace gcd_fx_x_l800_800248

-- Let x be an instance of ℤ
variable (x : ℤ)

-- Define that x is a multiple of 46200
def is_multiple_of_46200 := ∃ k : ℤ, x = 46200 * k

-- Define the function f(x) = (3x + 5)(5x + 3)(11x + 6)(x + 11)
def f (x : ℤ) := (3 * x + 5) * (5 * x + 3) * (11 * x + 6) * (x + 11)

-- The statement to prove
theorem gcd_fx_x (h : is_multiple_of_46200 x) : Int.gcd (f x) x = 990 := 
by
  -- Placeholder for the proof
  sorry

end gcd_fx_x_l800_800248


namespace min_red_marbles_is_120_l800_800119

noncomputable def min_red_marbles (w g r : ℕ) : Prop :=
  g ≥ (2 * w) / 3 ∧ g ≤ r / 4 ∧ w + g ≥ 72 ∧ r = 120

noncomputable def satisfies_conditions (w g r : ℕ) : Prop :=
  g ≥ (2 * w) / 3 ∧ g ≤ r / 4 ∧ w + g ≥ 72

theorem min_red_marbles_is_120 : ∃ w g r, satisfies_conditions w g r ∧ r = 120 :=
begin
  sorry
end

end min_red_marbles_is_120_l800_800119


namespace percentage_increase_in_weight_l800_800039

theorem percentage_increase_in_weight :
  ∀ (num_plates : ℕ) (weight_per_plate lowered_weight : ℝ),
    num_plates = 10 →
    weight_per_plate = 30 →
    lowered_weight = 360 →
    ((lowered_weight - num_plates * weight_per_plate) / (num_plates * weight_per_plate)) * 100 = 20 :=
by
  intros num_plates weight_per_plate lowered_weight h_num_plates h_weight_per_plate h_lowered_weight
  sorry

end percentage_increase_in_weight_l800_800039


namespace polar_to_rectangular_and_min_distance_l800_800714

theorem polar_to_rectangular_and_min_distance :
  (∀ (ρ θ : ℝ), ρ * sin (θ - π/4) = 2*√2 ↔ ∀ (x y : ℝ), (x = ρ * cos θ) ∧ (y = ρ * sin θ) → x - y + 4 = 0)
  ∧
  (∀ (x y : ℝ), (x^2 / 3 + y^2 / 9 = 1) → 
    ∀ (α : ℝ), P = (√3 * cos α, 3 * sin α) →
    min (|√3 * cos α - 3 * sin α + 4| / √2) = 2*√2 - √6) :=
by sorry

end polar_to_rectangular_and_min_distance_l800_800714


namespace sampling_methods_correct_l800_800890

def sampling_method_1 := "Systematic sampling"
def sampling_method_2 := "Stratified sampling"
def sampling_method_3 := "Simple random sampling"

def survey_method_1 := "To understand the situation of first-year high school students' mathematics learning, the school selects 2 students from each class for a discussion."
def survey_method_2 := "In a math competition, a certain class had 15 students scoring above 100, 35 students scoring between 90 and 100, and 10 students scoring below 90. Now, 12 students are selected for a discussion to understand the situation."
def survey_method_3 := "In a sports meeting, the staff fairly arranges tracks for 6 students participating in the 400m race."

theorem sampling_methods_correct :
  (survey_method_1 = "To understand the situation of first-year high school students' mathematics learning, the school selects 2 students from each class for a discussion." →
   sampling_method_1 = "Systematic sampling") ∧
  (survey_method_2 = "In a math competition, a certain class had 15 students scoring above 100, 35 students scoring between 90 and 100, and 10 students scoring below 90. Now, 12 students are selected for a discussion to understand the situation." →
   sampling_method_2 = "Stratified sampling") ∧
  (survey_method_3 = "In a sports meeting, the staff fairly arranges tracks for 6 students participating in the 400m race." →
   sampling_method_3 = "Simple random sampling") :=
by
  split;
  { intro h,
    try {exact rfl},
    sorry }

end sampling_methods_correct_l800_800890


namespace regular_pyramid_cannot_be_hexagonal_l800_800558

theorem regular_pyramid_cannot_be_hexagonal (n : ℕ) (h₁ : n = 6) (base_edge_length slant_height : ℝ) 
  (reg_pyramid : base_edge_length = slant_height) : false :=
by
  sorry

end regular_pyramid_cannot_be_hexagonal_l800_800558


namespace necessary_but_not_sufficient_l800_800244

variables (a b : Type) -- types to represent line entities
variables (intersect : a → b → Prop) -- predicate representing intersection
variables (skew : a → b → Prop) -- predicate representing skew lines

-- Proposition p: Lines a and b intersect
def p (a b : Type) [intersect a b] : Prop := intersect a b

-- Proposition q: Lines a and b are skew
def q (a b : Type) [skew a b] : Prop := skew a b

-- Theorem stating ¬p is necessary but not sufficient for q
theorem necessary_but_not_sufficient (a b : Type)
  (intersect : a → b → Prop)
  (skew : a → b → Prop)
  (np_impl_not_intersect : ¬(intersect a b) → True)
  (q_impl_not_intersect : skew a b → ¬(intersect a b)) :
  (¬(p a b [intersect]) → q a b [skew]) ∧ (q a b [skew] → ¬(p a b [intersect])) :=
by
  sorry

end necessary_but_not_sufficient_l800_800244


namespace total_volume_l800_800587

-- Definitions
def d : ℝ := 12
def r : ℝ := d / 2
def h : ℝ := 0.6 * d
def H : ℝ := 2 * h

def V_cone : ℝ := (Math.pi * r^2 * h) / 3
def V_cylinder : ℝ := Math.pi * r^2 * H

theorem total_volume : V_cone + V_cylinder = 604.8 * Math.pi := by
  -- proof goes here
  sorry

end total_volume_l800_800587


namespace largest_four_digit_number_with_property_l800_800073

theorem largest_four_digit_number_with_property :
  ∃ (a b c d : ℕ), (a = 9 ∧ b = 0 ∧ c = 9 ∧ d = 9 ∧ c = a + b ∧ d = b + c ∧ 1000 * a + 100 * b + 10 * c + d = 9099) :=
sorry

end largest_four_digit_number_with_property_l800_800073


namespace two_digit_number_perfect_cube_has_seven_values_l800_800618

def is_two_digit_number (N : ℕ) : Prop :=
  10 ≤ N ∧ N < 100

def digits_of_number (N : ℕ) : ℕ × ℕ :=
  (N / 10, N % 10)

def swapped_number (N : ℕ) : ℕ :=
  let (m, n) := digits_of_number N in 10 * n + m

def is_perfect_cube (k : ℤ) : Prop :=
  ∃ (n : ℤ), n^3 = k

def number_and_swapped_difference_is_perfect_cube (N : ℕ) : Prop :=
  let N' := swapped_number N in
  is_perfect_cube ((N : ℤ) - (N' : ℤ))

def valid_numbers (N : ℕ) : Prop :=
  is_two_digit_number N ∧ number_and_swapped_difference_is_perfect_cube N

theorem two_digit_number_perfect_cube_has_seven_values :
  ∃ (S : Finset ℕ), (∀ N, valid_numbers N ↔ N ∈ S) ∧ S.card = 7 :=
by
  sorry

end two_digit_number_perfect_cube_has_seven_values_l800_800618


namespace expression_equals_thirteen_l800_800904

-- Define the expression
def expression : ℤ :=
    8 + 15 / 3 - 4 * 2 + Nat.pow 2 3

-- State the theorem that proves the value of the expression
theorem expression_equals_thirteen : expression = 13 :=
by
  sorry

end expression_equals_thirteen_l800_800904


namespace part1_inequality_solution_part2_t_solution_l800_800267

open Real

def f (x : ℝ) : ℝ := abs x

theorem part1_inequality_solution :
  {x | f (2 * x) + f (x - 2) > 3} = {x | x < - (1 / 3) ∨ x > 1} := by
  sorry

def g (x : ℝ) (t : ℝ) : ℝ := 3 * f x - f (x - t)

theorem part2_t_solution (t : ℝ) (h : t ≠ 0) :
  (∫ x in -t/2..t/4, g x t) = 3 → t = 2 * sqrt 2 ∨ t = -2 * sqrt 2 := by
  sorry

end part1_inequality_solution_part2_t_solution_l800_800267


namespace jorkins_initial_money_l800_800556

def initial_pounds : ℕ := 19
def initial_shillings : ℕ := 18

theorem jorkins_initial_money (x y : ℕ) (h1 : 20 * x + y = 2 * (10 * y + x)) (h2 : y < 20) : 
  x = initial_pounds ∧ y = initial_shillings :=
by
  split
  { rw [<-nat.mul_div_cancel_left _ 19 (dec_trivial : 0 < 19), succ_eq_add_one, mul_add, mul_one] at h1,
    have h3 := nat.mul_left_inj (dec_trivial : 0 < 19),
    apply h3,
    linarith }
  { exact nat.div_eq_of_eq_mul_right (dec_trivial : 0 < 19) h.symm }

-- sorry

end jorkins_initial_money_l800_800556


namespace largest_four_digit_number_with_property_l800_800075

theorem largest_four_digit_number_with_property :
  ∃ (a b c d : ℕ), (a = 9 ∧ b = 0 ∧ c = 9 ∧ d = 9 ∧ c = a + b ∧ d = b + c ∧ 1000 * a + 100 * b + 10 * c + d = 9099) :=
sorry

end largest_four_digit_number_with_property_l800_800075


namespace largest_four_digit_number_prop_l800_800050

theorem largest_four_digit_number_prop :
  ∃ (a b c d : ℕ), a = 9 ∧ b = 0 ∧ c = 9 ∧ d = 9 ∧ (1000 * a + 100 * b + 10 * c + d = 9099) ∧ (c = a + b) ∧ (d = b + c) :=
by
  sorry

end largest_four_digit_number_prop_l800_800050


namespace positive_difference_of_probabilities_l800_800518

def binom (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def is_fair_coin : Prop := true -- Definition not needed, but included as a condition

def flips_eq_6 : ℕ := 6

def prob_exactly_heads (k : ℕ) (n : ℕ) := (binom n k) * ((1/2)^k) * ((1/2)^(n-k))

def prob4 := prob_exactly_heads 4 flips_eq_6
def prob6 := prob_exactly_heads 6 flips_eq_6

theorem positive_difference_of_probabilities :
  abs (prob4 - prob6) = 7 / 32 :=
by
  sorry

end positive_difference_of_probabilities_l800_800518


namespace vectors_perpendicular_vector_combination_l800_800732

def vector_a : ℝ × ℝ := (2, -1)
def vector_b : ℝ × ℝ := (-3, 2)
def vector_c : ℝ × ℝ := (1, 1)

-- Auxiliary definition of vector addition
def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)

-- Auxiliary definition of dot product
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := (v1.1 * v2.1 + v1.2 * v2.2)

-- Auxiliary definition of scalar multiplication
def scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

-- Proof that (vector_a + vector_b) is perpendicular to vector_c
theorem vectors_perpendicular : dot_product (vector_add vector_a vector_b) vector_c = 0 :=
by sorry

-- Proof that vector_c = 5 * vector_a + 3 * vector_b
theorem vector_combination : vector_c = vector_add (scalar_mul 5 vector_a) (scalar_mul 3 vector_b) :=
by sorry

end vectors_perpendicular_vector_combination_l800_800732


namespace concyclic_c_f_i_j_l800_800379

-- Definitions for points and circles
variables {A B C D E F I J K : Point} {ω : Circle}

-- Given conditions
axiom cyclic_quadrilateral (h1 : CyclicQuadrilateral A B C D) : circumcircle ω
axiom incenters (h2 : Incenter I (Triangle A B C)) (h3 : Incenter J (Triangle A C D)) (h4 : Incenter K (Triangle A B D))
axiom midpoint_arc_DB (h5 : Midpoint E (arc D B A ω))
axiom line_intersection_EK (h6 : Intersects (Line E K) ω F) (h7 : F ≠ E)

-- Theorem to be proved
theorem concyclic_c_f_i_j : is_concyclic C F I J :=
by
  sorry

end concyclic_c_f_i_j_l800_800379


namespace Seth_bought_20_cartons_of_ice_cream_l800_800835

-- Definitions from conditions
def ice_cream_cost_per_carton : ℕ := 6
def yogurt_cost_per_carton : ℕ := 1
def num_yogurt_cartons : ℕ := 2
def extra_amount_spent_on_ice_cream : ℕ := 118

-- Let x be the number of cartons of ice cream Seth bought
def num_ice_cream_cartons (x : ℕ) : Prop :=
  ice_cream_cost_per_carton * x = num_yogurt_cartons * yogurt_cost_per_carton + extra_amount_spent_on_ice_cream

-- The proof goal
theorem Seth_bought_20_cartons_of_ice_cream : num_ice_cream_cartons 20 :=
by
  unfold num_ice_cream_cartons
  unfold ice_cream_cost_per_carton yogurt_cost_per_carton num_yogurt_cartons extra_amount_spent_on_ice_cream
  sorry

end Seth_bought_20_cartons_of_ice_cream_l800_800835


namespace kalebs_restaurant_bill_l800_800633

theorem kalebs_restaurant_bill :
  let adults := 6
  let children := 2
  let adult_meal_cost := 6
  let children_meal_cost := 4
  let soda_cost := 2
  (adults * adult_meal_cost + children * children_meal_cost + (adults + children) * soda_cost) = 60 := 
by
  let adults := 6
  let children := 2
  let adult_meal_cost := 6
  let children_meal_cost := 4
  let soda_cost := 2
  calc 
    adults * adult_meal_cost + children * children_meal_cost + (adults + children) * soda_cost 
      = 6 * 6 + 2 * 4 + (6 + 2) * 2 : by rfl
    ... = 36 + 8 + 16 : by rfl
    ... = 60 : by rfl

end kalebs_restaurant_bill_l800_800633


namespace exponent_multiplication_l800_800029

theorem exponent_multiplication (a : ℝ) (h : a = 81) : 
  (a ^ 0.20 * a ^ 0.05 = 3) := 
by 
  sorry

end exponent_multiplication_l800_800029


namespace cubic_roots_inequalities_l800_800438

variable {a b : ℝ}

theorem cubic_roots_inequalities 
  (h1 : ∀ x : ℝ, ax^3 - x^2 + bx - 1 = 0 → (x > 0)) :
  0 < 3 * a * b ∧ 3 * a * b ≤ 1 ∧ b ≥ √3 :=
begin
  sorry
end

end cubic_roots_inequalities_l800_800438


namespace solve_for_y_l800_800683

theorem solve_for_y (x y : ℝ) (h : x - 2 = 4 * y + 3) : y = (x - 5) / 4 :=
by
  sorry

end solve_for_y_l800_800683


namespace inequality_geq_n_over_2_l800_800382

theorem inequality_geq_n_over_2 (n : ℕ) (x : Fin n → ℝ) (h1 : ∀ i, 0 < x i) (h2 : (∏ i, x i) = 1) :
  (∑ i, 1 / (x i * (x i + 1))) ≥ n / 2 :=
by
  sorry

end inequality_geq_n_over_2_l800_800382


namespace bullet_speed_difference_l800_800529

theorem bullet_speed_difference
  (horse_speed : ℝ := 20) 
  (bullet_speed : ℝ := 400) : 
  ((bullet_speed + horse_speed) - (bullet_speed - horse_speed) = 40) := by
  sorry

end bullet_speed_difference_l800_800529


namespace incorrect_statement_a_correct_statement_b_correct_statement_c_correct_statement_d_l800_800092

theorem incorrect_statement_a (h1 : ∀ x : ℤ, abs x ≥ 0)
  (h2 : ∃ y : ℤ, abs y = 0)
  (h3 : ∀ z : ℚ, abs z = 0 → z = 0)
  (h4 : ∀ p : ℚ, p > 0 ∨ p < 0 → p ≠ 0)
  : ¬ (∀ n : ℤ, abs n = 1 → n = 1) :=
by
  sorry

theorem correct_statement_b : ¬ (0 > 0) ∧ ¬ (0 < 0) :=
by
  sorry

theorem correct_statement_c (r : ℚ) : r ∈ ℤ ∨ r ∉ ℤ :=
by
  sorry

theorem correct_statement_d : abs (0 : ℤ) = 0 :=
by
  sorry

end incorrect_statement_a_correct_statement_b_correct_statement_c_correct_statement_d_l800_800092


namespace profitable_when_price_above_132_l800_800773

-- Transaction fee
def fee_rate := 7.5 / 1000

-- Selling price per share
def selling_price (x : ℝ) := (x + 2) * (1 - fee_rate)

-- Cost price per share
def cost_price (x : ℝ) := x * (1 + fee_rate)

-- Profit condition
theorem profitable_when_price_above_132 (x : ℝ) (h : x ≥ 132) :
  1000 * selling_price x - 1000 * cost_price x ≥ 0 :=
by
  sorry

end profitable_when_price_above_132_l800_800773


namespace necessary_not_sufficient_perpendicularity_l800_800686

variables (α β : Plane) (m : Line) 

-- We define the conditions from the problem.
axiom different_planes : α ≠ β
axiom line_in_plane : α.contains m
axiom planes_perpendicular : α ⊥ β

-- Now we express the theorem to be proven.
theorem necessary_not_sufficient_perpendicularity :
  ¬ ((α ⊥ β) → (m ⊥ β) ∧ ((m ⊥ β) → (α ⊥ β))) :=
sorry

end necessary_not_sufficient_perpendicularity_l800_800686


namespace bus_stops_for_15_minutes_per_hour_l800_800196

-- Define the conditions
def speed_without_stoppages := 64 -- in km/hr
def speed_with_stoppages := 48 -- in km/hr

-- Define the question in terms of Lean:
-- Prove that the bus stops for approximately 15 minutes per hour.
theorem bus_stops_for_15_minutes_per_hour :
  let speed_reduction := speed_without_stoppages - speed_with_stoppages
    km_per_minute := speed_without_stoppages / 60
    time_stopped := speed_reduction / km_per_minute
  in time_stopped ≈ 15 :=
by
  sorry

end bus_stops_for_15_minutes_per_hour_l800_800196


namespace part1_increasing_intervals_part2_cos2x0_l800_800273

noncomputable def f (x : ℝ) : ℝ := 
  sin x ^ 2 + 2 * sqrt 3 * sin x * cos x + sin (x + π / 4) * sin (x - π / 4)

theorem part1_increasing_intervals (k : ℤ) :
  ∀ x : ℝ, k * π - π / 6 ≤ x ∧ x ≤ k * π + π / 3 → 
           f' x ≥ 0 := 
sorry

theorem part2_cos2x0 (x₀ : ℝ) (h₀ : f x₀ = 0) (h1 : 0 ≤ x₀ ∧ x₀ ≤ π / 2) :
  cos (2 * x₀) = (3 * sqrt 15 + 1) / 8 := 
sorry

end part1_increasing_intervals_part2_cos2x0_l800_800273


namespace wealthiest_individuals_income_l800_800946

/--
Given the revised formula representing the number of individuals whose income surpasses \( x \) dollars, 
and that the number of individuals \( N \) is set to 500, 
prove that the minimum income of the wealthiest 500 individuals is 100 dollars.
-/

theorem wealthiest_individuals_income :
  ∃ x : ℝ, (x > 0) ∧ (500 = 5 * 10^7 * x^(-5/2)) ∧ (x = 10^2) :=
by
  use 10^2
  sorry

end wealthiest_individuals_income_l800_800946


namespace equilateral_triangle_if_bisectors_equal_l800_800781

theorem equilateral_triangle_if_bisectors_equal 
  (A B C D E I : Type) 
  [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E] [Inhabited I]
  (triangle : Triangle A B C)
  (bisector_B : Bisector B D A C)
  (bisector_C : Bisector C E A B)
  (intersection_I : Intersection I bisector_B bisector_C)
  (equal_segments : Segment I D = Segment I E) : 
  EquilateralTriangle A B C := 
sorry

end equilateral_triangle_if_bisectors_equal_l800_800781


namespace hyperbola_properties_l800_800939
-- Import the required math library

-- Define the given conditions
def foci_on_x_axis (h : ℝ → ℝ → Prop) : Prop := ∀ x y, h x y ↔ (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (x^2 / a^2) - (y^2 / b^2) = 1)
def angle_between_asymptotes (h : ℝ → ℝ → Prop) : Prop := 
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2 * b * a = sqrt(3) * (a^2 - b^2))
def focal_distance (h : ℝ → ℝ → Prop) (d : ℝ) : Prop := 
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^2 + b^2 = (d / 2)^2)

-- Define the statements to be proved
theorem hyperbola_properties :
  ∀ (h : ℝ → ℝ → Prop), 
    foci_on_x_axis h →
    angle_between_asymptotes h →
    focal_distance h 12 →
    h = (λ x y, (x^2 / 27) - (y^2 / 9) = 1) 
    ∨ h = (λ x y, (x^2 / 9) - (y^2 / 27) = 1) →
    (∃ e, e = 2 ∨ e = 2 * sqrt(3) / 3) :=
by
  sorry


end hyperbola_properties_l800_800939


namespace sum_of_ages_l800_800030

theorem sum_of_ages (youngest_age : ℕ) (interval : ℕ) (n : ℕ) : 
  youngest_age = 7 → interval = 3 → n = 5 → 
  (∑ i in finset.range n, youngest_age + i * interval) = 65 := 
by 
  intros h1 h2 h3 
  rw [h1, h2, h3] 
  norm_num 
  sorry

end sum_of_ages_l800_800030


namespace probability_students_from_different_grades_l800_800570

theorem probability_students_from_different_grades :
  let total_students := 4
  let first_grade_students := 2
  let second_grade_students := 2
  (2 from total_students are selected) ->
  (2 from total_students are from different grades) ->
  ℝ :=
by 
  sorry

end probability_students_from_different_grades_l800_800570


namespace sum_xor_floor_eq_l800_800671

theorem sum_xor_floor_eq :
  (∑ k in Finset.range (2^2014), k ⊕ (k / 2)) = 2^2013 * (2^2014 - 1) :=
sorry

end sum_xor_floor_eq_l800_800671


namespace angle_bisector_proportion_l800_800443

axiom isosceles_triangle (A B C : Type) [MetricSpace G] (Δ : Triangle A B C) : Prop :=
isosceles_triangle(angle A B C = 108)

def isosceles_triangle (A B C : Point) : Prop :=
isosceles_triangle.angle B = 108

theorem angle_bisector_proportion {A B C : Point} (Δ : Triangle A B C) (isosceles : isosceles_triangle A B C) 
  (AD BE : Point)
  (H_AD_is_bisector : is_angle_bisector A A B C AD)
  (H_BE_is_bisector : is_angle_bisector B B A C BE) :
  segment.length AD = 2 * segment.length BE :=
by sorry

end angle_bisector_proportion_l800_800443


namespace ratio_of_PC_to_PA_l800_800785

-- Define the square with coordinates
def square (A B C D : Point) (side : ℝ) :=
  A = (0, side) ∧ B = (side, side) ∧ C = (side, 0) ∧ D = (0, 0)

-- Define midpoint calculation
def midpoint (P Q : Point) : Point :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Define the line intersection point
def intersection (line1 line2 : ℝ × ℝ) : Point :=
  let x := line2.1
  let y := line1.1 * x + line1.2
  (x, y)

-- Define the distance calculation
def dist (P Q : Point) : ℝ :=
  sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Describe the problem conditions
theorem ratio_of_PC_to_PA (A B C D M N P : Point) (side : ℝ) :
  square A B C D side ∧ 
  side = 6 ∧
  M = midpoint C D ∧ 
  N = midpoint A B ∧
  P = intersection (line A N) (line B M) →
  (dist P C) / (dist P A) = 0 :=
by
  sorry

end ratio_of_PC_to_PA_l800_800785


namespace proof_problem_l800_800647

noncomputable def expr : ℝ := (3 - Real.pi)^0 - (1 / 3)^(-1) + abs (2 - Real.sqrt 8) + 2 * Real.cos (Real.pi / 4)

theorem proof_problem : expr = 3 * Real.sqrt 2 - 4 :=
by
  sorry

end proof_problem_l800_800647


namespace num_possible_outcomes_correct_l800_800941

noncomputable def num_possible_outcomes : Nat :=
  30

theorem num_possible_outcomes_correct (shots: List Bool) (hits: Nat) (consecutive_hits: Nat) : 
  shots.length = 8 ∧ hits = 4 ∧ consecutive_hits = 2 → List.count (shots) true = hits ∧ (∃ i, shots[i] = true ∧ shots[i+1] = true) → 
  List.count_by (λ s => s = true) (List.partition (λ n => ¬shots[n-1] ∧ ¬shots[n+1]) shots).1 = consecutive_hits → List.permutations shots = 30 := 
begin
  intros,
  sorry
end

end num_possible_outcomes_correct_l800_800941


namespace min_distance_ellipse_to_line_l800_800427

def ellipse (x y : ℝ) : Prop :=
  (x^2 / 9) + (y^2 / 4) = 1

def line (x y : ℝ) : Prop :=
  x + 2 * y - 10 = 0

theorem min_distance_ellipse_to_line :
  ∀ (M : ℝ × ℝ), ellipse M.1 M.2 → 
  (∃ (d : ℝ), d = sqrt 5 ∧ 
  (∀ (x y : ℝ), ellipse x y → 
    (abs ((x + 2 * y - 10) / sqrt 5) ≥ d))) :=
by
  sorry

end min_distance_ellipse_to_line_l800_800427


namespace probability_AC_lt_8_cm_is_046_l800_800626

noncomputable theory 

-- Define the points based on given distances and rotation angle.
def A : ℝ × ℝ := (0, -10)
def B : ℝ × ℝ := (0, 0)
def C (β : ℝ) : ℝ × ℝ := (6 * Math.cos β, 6 * Math.sin β)

-- Define circle equation centered at a point with a given radius.
def circle_eq (center : ℝ × ℝ) (r : ℝ) : (ℝ × ℝ) → Prop := 
  λ p, (p.1 - center.1)^2 + (p.2 - center.2)^2 = r^2

-- Define the probability calculation for the angle beta resulting in distance AC < 8.
def probability_AC_less_than_8 : ℝ :=
  ∫ β in (0:ℝ) .. (Real.arccos (4.8 / 6)), 1 / (Real.pi / 2)

-- Main theorem stating the probability AC < 8 is 0.46.
theorem probability_AC_lt_8_cm_is_046 : 
  (probability_AC_less_than_8 ≈ 0.46) := 
sorry

end probability_AC_lt_8_cm_is_046_l800_800626


namespace parabola_tangent_sequence_l800_800974

noncomputable def geom_seq_sum (a2 : ℕ) : ℕ :=
  a2 + a2 / 4 + a2 / 16

theorem parabola_tangent_sequence (a2 : ℕ) (h : a2 = 32) : geom_seq_sum a2 = 42 :=
by
  rw [h]
  norm_num
  sorry

end parabola_tangent_sequence_l800_800974


namespace geometric_progression_x_value_l800_800665

noncomputable def geometric_progression_solution (x : ℝ) : Prop :=
  let a := -30 + x
  let b := -10 + x
  let c := 40 + x
  b^2 = a * c

theorem geometric_progression_x_value :
  ∃ x : ℝ, geometric_progression_solution x ∧ x = 130 / 3 :=
by
  sorry

end geometric_progression_x_value_l800_800665


namespace triangle_angles_determined_l800_800706

theorem triangle_angles_determined
  (a b c : ℝ)
  (A B C : ℝ)
  (m : ℝ × ℝ := (sqrt 3, 1))
  (n : ℝ × ℝ := (Real.cos A, Real.sin A))
  (h1 : m.1 * n.1 + m.2 * n.2 = 0)
  (h2 : a * Real.cos B + b * Real.cos A = c * Real.sin C)
  (h3 : A + B + C = Real.pi) :
  A = Real.pi / 3 ∧ B = Real.pi / 6 :=
by 
  -- Proof would go here
  sorry

end triangle_angles_determined_l800_800706


namespace olympiad_scores_l800_800355

theorem olympiad_scores (scores : Fin 20 → ℕ) 
  (uniqueScores : ∀ i j, i ≠ j → scores i ≠ scores j)
  (less_than_sum_of_others : ∀ i j k, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k) :
  ∀ i, scores i > 18 := 
by sorry

end olympiad_scores_l800_800355


namespace triangle_construction_exists_l800_800243

-- Define the geometric elements and conditions

structure Point where
  x : ℝ
  y : ℝ

structure Vector where
  dx : ℝ
  dy : ℝ

structure Triangle where
  A B C : Point

-- Define the condition of a direction for a side
def direction (v : Vector) (p1 p2 : Point) : Prop := 
  (p2.x - p1.x) * v.dy = (p2.y - p1.y) * v.dx

-- Define the incircle property
def isIncircle (r : ℝ) (c : Point) (t : Triangle) : Prop := sorry  -- Assuming we define the necessary geometry

-- Define a triangle construction problem
def constructTriangle (A B : Point) (vB vC : Vector) (r : ℝ) : Triangle := sorry  -- Construction steps to be implemented

-- Define the theorem to be proved
theorem triangle_construction_exists 
  (A B : Point) 
  (vB vC : Vector) 
  (r : ℝ) 
  (h_direction : direction vB A B)
  (h_positive_r : r > 0) :
  ∃ (T : Triangle), 
    T.A = A ∧
    T.B = B ∧
    direction vC A T.C ∧
    isIncircle r (Point.mk ((A.x + B.x + T.C.x) / 3) ((A.y + B.y + T.C.y) / 3)) T :=
by 
  sorry

end triangle_construction_exists_l800_800243


namespace find_d_l800_800152

-- Defining the coordinates of the foci
def F1 := (4, 8)
def F2 := (d : ℝ × ℝ)
def P := (d', 0) -- P is the point where ellipse is tangent to the x-axis

-- Hypothesis based on conditions provided
def hypothesis (d : ℝ) :=
  F2 = (d, 8) ∧ (d + 4 = 2 * sqrt (((d - 4) / 2)^2 + 8^2))

-- The goal to prove
theorem find_d : ∃ d : ℝ, hypothesis d ∧ d = 30 :=
by 
  sorry

end find_d_l800_800152


namespace sale_price_60_l800_800801

theorem sale_price_60 (original_price : ℕ) (discount_percentage : ℝ) (sale_price : ℝ) 
  (h1 : original_price = 100) 
  (h2 : discount_percentage = 0.40) :
  sale_price = (original_price : ℝ) * (1 - discount_percentage) :=
by
  sorry

end sale_price_60_l800_800801


namespace sum_lt_2500_probability_l800_800970

open ProbabilityTheory

theorem sum_lt_2500_probability :
  let x : MeasureTheory.MeasureSpace ℝ := uniform (Icc 0 1000)
  let y : MeasureTheory.MeasureSpace ℝ := uniform (Icc 0 3000)
  P (λ (xy : ℝ × ℝ), xy.1 + xy.2 < 2500) = 1 / 4 :=
by
  -- uniform distribution properties and combination
  -- calculation will be required here
  sorry

end sum_lt_2500_probability_l800_800970


namespace trip_time_difference_l800_800108

theorem trip_time_difference
  (avg_speed : ℝ)
  (dist1 dist2 : ℝ)
  (h_avg_speed : avg_speed = 60)
  (h_dist1 : dist1 = 540)
  (h_dist2 : dist2 = 570) :
  ((dist2 - dist1) / avg_speed) * 60 = 30 := by
  sorry

end trip_time_difference_l800_800108


namespace find_a_values_l800_800318

-- Definitions related to the problem conditions
def circle : set (ℝ × ℝ) := 
  {p : ℝ × ℝ | let (x, y) := p in x^2 + y^2 - 2*x - 2*y - 2 = 0}

def line (a : ℝ) : set (ℝ × ℝ) := 
  {p : ℝ × ℝ | let (x, y) := p in x + y + a = 0}

-- Main problem statement
theorem find_a_values (a : ℝ) : 
  (∃ s ∈ circle, ∃ (p q r : (ℝ × ℝ)), 
  p ∈ circle ∧ q ∈ circle ∧ r ∈ circle ∧ 
  dist s (1, 1) = 1 ∧ dist (1, 1) (1, 1) = 1 ∧ 
  dist (1, 1) s = 1 ∧ 
  p ∈ (line a) ∧ q ∈ (line a) ∧ r ∈ (line a)) ↔ a = -2 + real.sqrt 2 ∨ a = -2 - real.sqrt 2 :=
sorry

end find_a_values_l800_800318


namespace intervals_of_monotonicity_range_of_a_for_zeros_l800_800269

open Real

noncomputable def f (x a : ℝ) : ℝ := (1/2) * x^2 - 3 * a * x + 2 * a^2 * log x

theorem intervals_of_monotonicity (a : ℝ) (ha : a ≠ 0) :
  (0 < a → ∀ x, (0 < x ∧ x < a → f x a < f (x + 1) a)
            ∧ (a < x ∧ x < 2 * a → f x a > f (x + 1) a)
            ∧ (2 * a < x → f x a < f (x + 1) a))
  ∧ (a < 0 → ∀ x, (0 < x → f x a < f (x + 1) a)) :=
sorry

theorem range_of_a_for_zeros (a x : ℝ) (ha : 0 < a) 
  (h1 : f a a > 0) (h2 : f (2 * a) a < 0) :
  e ^ (5 / 4) < a ∧ a < e ^ 2 / 2 :=
sorry

end intervals_of_monotonicity_range_of_a_for_zeros_l800_800269


namespace rhombus_perimeter_l800_800465

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 20) (h2 : d2 = 16) : 
  let side := (real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)) in 
  4 * side = 8 * real.sqrt 41 := by
  sorry

end rhombus_perimeter_l800_800465


namespace distinguishable_arrangements_l800_800891

theorem distinguishable_arrangements : 
    let jars := 2
        let total_balls := 17
        let red_balls := 8
        let blue_balls := 9
    (∀ balls_arrangement, 
        (balls_arrangement ∈ (list.range (red_balls + blue_balls)).permutations) → 
        ∃ jar1 jar2, 
            jar1 ∪ jar2 = balls_arrangement ∧ 
            (jar1 ≠ [] ∧ jar2 ≠ []) ∧ 
            (∀ b, b ∈ jar1 ∨ b ∈ jar2 → 
                ((b = blue_ball) → (¬ adjacent blue_ball blue_ball jar1) ∧
                (¬ adjacent blue_ball blue_ball jar2)))) → 
    M = 7 :=
sorry

end distinguishable_arrangements_l800_800891


namespace find_f_5pi_div_3_l800_800258

variable (f : ℝ → ℝ)

-- Define the conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_periodic_function (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem find_f_5pi_div_3
  (h_odd : is_odd_function f)
  (h_periodic : is_periodic_function f π)
  (h_def : ∀ x, 0 ≤ x → x ≤ π/2 → f x = Real.sin x) :
  f (5 * π / 3) = - (Real.sqrt 3 / 2) := by
  sorry

end find_f_5pi_div_3_l800_800258


namespace time_to_fill_cistern_l800_800504

def pipe_p_rate := (1: ℚ) / 10
def pipe_q_rate := (1: ℚ) / 15
def pipe_r_rate := - (1: ℚ) / 30
def combined_rate_p_q := pipe_p_rate + pipe_q_rate
def combined_rate_q_r := pipe_q_rate + pipe_r_rate
def initial_fill := 2 * combined_rate_p_q
def remaining_fill := 1 - initial_fill
def remaining_time := remaining_fill / combined_rate_q_r

theorem time_to_fill_cistern :
  remaining_time = 20 := by sorry

end time_to_fill_cistern_l800_800504


namespace ryan_spit_distance_l800_800641

def billy_distance : ℝ := 30

def madison_distance (b : ℝ) : ℝ := b + (20 / 100 * b)

def ryan_distance (m : ℝ) : ℝ := m - (50 / 100 * m)

theorem ryan_spit_distance : ryan_distance (madison_distance billy_distance) = 18 :=
by
  sorry

end ryan_spit_distance_l800_800641


namespace positive_solutions_to_cos_arctan_sin_arccos_eq_x_l800_800211

theorem positive_solutions_to_cos_arctan_sin_arccos_eq_x :
   ∃! x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ cos (arctan (sin (arccos x))) = x :=
sorry

end positive_solutions_to_cos_arctan_sin_arccos_eq_x_l800_800211


namespace largest_four_digit_number_with_property_l800_800076

theorem largest_four_digit_number_with_property :
  ∃ (a b c d : ℕ), (a = 9 ∧ b = 0 ∧ c = 9 ∧ d = 9 ∧ c = a + b ∧ d = b + c ∧ 1000 * a + 100 * b + 10 * c + d = 9099) :=
sorry

end largest_four_digit_number_with_property_l800_800076


namespace shifted_graph_sum_l800_800525

noncomputable def f (x : ℝ) : ℝ := 3 * x ^ 2 - 2 * x + 5

def shift_right (f : ℝ → ℝ) (h : ℝ) (x : ℝ) : ℝ := f (x - h)
def shift_up (f : ℝ → ℝ) (k : ℝ) (x : ℝ) : ℝ := f x + k

noncomputable def g (x : ℝ) : ℝ := shift_up (shift_right f 7) 3 x

theorem shifted_graph_sum : (∃ (a b c : ℝ), g x = a * x ^ 2 + b * x + c ∧ (a + b + c = 128)) :=
by
  sorry

end shifted_graph_sum_l800_800525


namespace distance_between_stations_l800_800512

theorem distance_between_stations 
  (distance_P_to_meeting : ℝ)
  (distance_Q_to_meeting : ℝ)
  (h1 : distance_P_to_meeting = 20 * 3)
  (h2 : distance_Q_to_meeting = 25 * 2)
  (h3 : distance_P_to_meeting + distance_Q_to_meeting = D) :
  D = 110 :=
by
  sorry

end distance_between_stations_l800_800512


namespace five_digit_palindromic_count_l800_800965

theorem five_digit_palindromic_count : 
  (∃ n : ℕ, n = (9 * 10 * 10) ∧ 
  (∀ a b c d e : ℕ, 
    a ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
    b ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
    c ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
    d = b ∧ e = a → 
    (n = 900))) :=
sorry

end five_digit_palindromic_count_l800_800965


namespace equation_has_real_solution_l800_800914

theorem equation_has_real_solution (m : ℝ) : ∃ x : ℝ, x^2 - m * x + m - 1 = 0 :=
by
  -- provide the hint that the discriminant (Δ) is (m - 2)^2
  have h : (m - 2)^2 ≥ 0 := by apply pow_two_nonneg
  sorry

end equation_has_real_solution_l800_800914


namespace rhombus_perimeter_l800_800460

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 20) (h2 : d2 = 16) :
  let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  in 4 * s = 8 * Real.sqrt 41 := 
by
  sorry

end rhombus_perimeter_l800_800460


namespace cone_radius_of_base_l800_800877

noncomputable def radius_of_base (CSA l : ℝ) : ℝ := 
  CSA / (Real.pi * l)

theorem cone_radius_of_base (CSA l r : ℝ) (h₁ : l = 20) (h₂ : CSA = 628.3185307179587) : 
  radius_of_base CSA l = 10 := by
  rw [h₁, h₂]
  -- sorry

end cone_radius_of_base_l800_800877


namespace train_cross_time_l800_800299

theorem train_cross_time (length_train : ℝ) (length_bridge : ℝ) (speed_kmph : ℝ) : 
  length_train = 100 →
  length_bridge = 150 →
  speed_kmph = 63 →
  (length_train + length_bridge) / (speed_kmph * (1000 / 3600)) = 14.29 :=
by
  sorry

end train_cross_time_l800_800299


namespace pyramid_height_l800_800251

theorem pyramid_height (lateral_edge : ℝ) (h : ℝ) (equilateral_angles : ℝ × ℝ × ℝ) (lateral_edge_length : lateral_edge = 3)
  (lateral_faces_are_equilateral : equilateral_angles = (60, 60, 60)) :
  h = 3 / 4 := by
  sorry

end pyramid_height_l800_800251


namespace ratio_of_dancers_l800_800035

theorem ratio_of_dancers (total_kids total_dancers slow_dance non_slow_dance : ℕ)
  (h1 : total_kids = 140)
  (h2 : slow_dance = 25)
  (h3 : non_slow_dance = 10)
  (h4 : total_dancers = slow_dance + non_slow_dance) :
  (total_dancers : ℚ) / total_kids = 1 / 4 :=
by
  sorry

end ratio_of_dancers_l800_800035


namespace find_x_l800_800754

noncomputable def A : (ℝ × ℝ × ℝ) := (-1, 2, 3)
noncomputable def B : (ℝ × ℝ × ℝ) := (2, -4, 1)
noncomputable def C (x : ℝ) : (ℝ × ℝ × ℝ) := (x, -1, -3)

noncomputable def vector_sub (p q : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1 - q.1, p.2 - q.2, p.3 - q.3)

noncomputable def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

theorem find_x : ∃ x : ℝ, dot_product (vector_sub B A) (vector_sub (C x) A) = 0 ∧ x = -11 :=
by
  -- Proof will be filled in here.
  sorry

end find_x_l800_800754


namespace largest_four_digit_number_property_l800_800066

theorem largest_four_digit_number_property : ∃ (a b c d : Nat), 
  (1000 * a + 100 * b + 10 * c + d = 9099) ∧ 
  (c = a + b) ∧ 
  (d = b + c) ∧ 
  1 ≤ a ∧ a ≤ 9 ∧ 
  0 ≤ b ∧ b ≤ 9 ∧ 
  0 ≤ c ∧ c ≤ 9 ∧ 
  0 ≤ d ∧ d ≤ 9 := 
sorry

end largest_four_digit_number_property_l800_800066


namespace subset_exact_distance_l800_800685

theorem subset_exact_distance {n : ℕ} (h : n > 14) (S : Finset (ℝ × ℝ))
  (hS : S.card = n) 
  (h_dist : ∀ p q ∈ S, p ≠ q → dist p q ≥ 1) :
  ∃ T ⊆ S, (∀ x y ∈ T, x ≠ y → dist x y ≥ sqrt 3) :=
sorry

end subset_exact_distance_l800_800685


namespace shopkeeper_bananas_l800_800130

noncomputable def number_of_bananas (num_oranges : ℕ) (pct_oranges_rotten : ℝ) (pct_bananas_rotten : ℝ) (pct_fruits_good_condition : ℝ) : ℕ :=
  let good_oranges := (1 - pct_oranges_rotten) * num_oranges
  let good_fruits := pct_fruits_good_condition * (num_oranges + B)
  let equation := good_oranges + (1 - pct_bananas_rotten) * B = good_fruits
  B

theorem shopkeeper_bananas (num_oranges : ℕ) (pct_oranges_rotten : ℝ) (pct_bananas_rotten : ℝ) (pct_fruits_good_condition : ℝ) (B : ℕ)
  (h1 : num_oranges = 600)
  (h2 : pct_oranges_rotten = 0.15)
  (h3 : pct_bananas_rotten = 0.04)
  (h4 : pct_fruits_good_condition = 0.894)
  : B = 400 :=
by sorry

end shopkeeper_bananas_l800_800130


namespace max_odd_integers_chosen_l800_800148

theorem max_odd_integers_chosen (a b c d e f g : ℕ) 
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f ∧ 0 < g)
  (h2 : a * b * c * d * e * f * g % 2 = 0) :
  (finset.card (finset.filter (λ x, x % 2 = 1) (finset.of_list [a, b, c, d, e, f, g])) ≤ 6) ∧ 
  (finset.card (finset.filter (λ x, x % 2 = 0) (finset.of_list [a, b, c, d, e, f, g])) ≥ 1) :=
by
  sorry

end max_odd_integers_chosen_l800_800148


namespace sum_of_three_numbers_l800_800007

theorem sum_of_three_numbers {a b c : ℝ} (h₁ : a ≤ b ∧ b ≤ c) (h₂ : b = 10)
  (h₃ : (a + b + c) / 3 = a + 20) (h₄ : (a + b + c) / 3 = c - 25) :
  a + b + c = 45 :=
by
  sorry

end sum_of_three_numbers_l800_800007


namespace probability_defective_first_lathe_overall_probability_defective_conditional_probability_second_lathe_conditional_probability_third_lathe_l800_800496

noncomputable def defect_rate_first_lathe : ℝ := 0.06
noncomputable def defect_rate_second_lathe : ℝ := 0.05
noncomputable def defect_rate_third_lathe : ℝ := 0.05
noncomputable def proportion_first_lathe : ℝ := 0.25
noncomputable def proportion_second_lathe : ℝ := 0.30
noncomputable def proportion_third_lathe : ℝ := 0.45

theorem probability_defective_first_lathe :
  defect_rate_first_lathe * proportion_first_lathe = 0.015 :=
by sorry

theorem overall_probability_defective :
  defect_rate_first_lathe * proportion_first_lathe +
  defect_rate_second_lathe * proportion_second_lathe +
  defect_rate_third_lathe * proportion_third_lathe = 0.0525 :=
by sorry

theorem conditional_probability_second_lathe :
  (defect_rate_second_lathe * proportion_second_lathe) /
  (defect_rate_first_lathe * proportion_first_lathe +
  defect_rate_second_lathe * proportion_second_lathe +
  defect_rate_third_lathe * proportion_third_lathe) = 2 / 7 :=
by sorry

theorem conditional_probability_third_lathe :
  (defect_rate_third_lathe * proportion_third_lathe) /
  (defect_rate_first_lathe * proportion_first_lathe +
  defect_rate_second_lathe * proportion_second_lathe +
  defect_rate_third_lathe * proportion_third_lathe) = 3 / 7 :=
by sorry

end probability_defective_first_lathe_overall_probability_defective_conditional_probability_second_lathe_conditional_probability_third_lathe_l800_800496


namespace binomial_sum_mod_prime_l800_800873

theorem binomial_sum_mod_prime (T : ℕ) (hT : T = ∑ k in Finset.range 65, Nat.choose 2024 k) : 
  T % 2027 = 1089 :=
by
  have h_prime : Nat.prime 2027 := by sorry -- Given that 2027 is prime
  have h := (2024 : ℤ) % 2027
  sorry -- The proof of the actual sum equivalences

end binomial_sum_mod_prime_l800_800873


namespace arithmetic_sequence_problem_l800_800395

-- Define the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- Use the given specific conditions
theorem arithmetic_sequence_problem 
  (a : ℕ → ℤ)
  (h1 : is_arithmetic_sequence a)
  (h2 : a 2 * a 3 = 21) : 
  a 1 * a 4 = -11 :=
sorry

end arithmetic_sequence_problem_l800_800395


namespace count_distinct_colored_cubes_l800_800935

-- Define colors and the cube structure
inductive Color 
| blue  
| red   
| green 

structure Cube :=
(faces : Fin 6 → Color)

-- The problem conditions about the cube
def valid_coloring (c : Cube) : Prop :=
∃ b r g : Fin 6 → Prop, 
  (∃ i, b i ∧ c.faces i = Color.blue) ∧
  (∀ i, b i → c.faces i = Color.blue) ∧
  (∃ j k l, r j ∧ r k ∧ r l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧ c.faces j = Color.red ∧ c.faces k = Color.red ∧ c.faces l = Color.red) ∧
  (∀ i, r i → c.faces i = Color.red) ∧
  (∃ m n, g m ∧ g n ∧ m ≠ n ∧ c.faces m = Color.green ∧ c.faces n = Color.green) ∧
  (∀ i, g i → c.faces i = Color.green) 

-- The equivalence condition for cubes under rotation will be ignored in this statement since it's complex
-- The goal is to show there are 3 distinct such cubes
theorem count_distinct_colored_cubes : 
  {c : Cube // valid_coloring c}.to_finset.card = 3 := 
sorry

end count_distinct_colored_cubes_l800_800935


namespace approximate_solution_l800_800964

-- Define the function f(x) = x^2 - 3x
def f (x : ℝ) : ℝ := x^2 - 3 * x 

-- Given conditions from the table
def table_values : List (ℝ × ℝ) := [
  (-1.13, 4.67),
  (-1.12, 4.61),
  (-1.11, 4.56),
  (-1.10, 4.51),
  (-1.09, 4.46),
  (-1.08, 4.41),
  (-1.07, 4.35)
]

-- Theorem to prove the approximate solution
theorem approximate_solution : ∃ x ∈ List.map Prod.fst table_values, f x = 4.6 → x ≈ -1.117 :=
sorry

end approximate_solution_l800_800964


namespace tangent_slope_is_four_l800_800021

-- Define the given curve and point
def curve (x : ℝ) : ℝ := 2 * x^2
def point : ℝ × ℝ := (1, 2)

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 4 * x

-- Define the tangent slope at the given point
def tangent_slope_at_point : ℝ := curve_derivative 1

-- Prove that the tangent slope at point (1, 2) is 4
theorem tangent_slope_is_four : tangent_slope_at_point = 4 :=
by
  -- We state that the slope at x = 1 is 4
  sorry

end tangent_slope_is_four_l800_800021


namespace rhombus_perimeter_l800_800466

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 20) (h2 : d2 = 16) : 
  let side := (real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)) in 
  4 * side = 8 * real.sqrt 41 := by
  sorry

end rhombus_perimeter_l800_800466


namespace hyperbola_equation_l800_800850

theorem hyperbola_equation (a b c : ℝ) (ha : 0 < a) (hb : 0 < b)
  (eccentricity : c = 2 * a)
  (asymptote_tangent : |a * b| / Real.sqrt (a^2 + b^2) = Real.sqrt (3) / 2) :
  (a = 1) ∧ (b = Real.sqrt 3) →
  ∃ a b : ℝ, (0 < a) ∧ (0 < b) ∧ (x^2 / a^2 - y^2 / b^2 = 1) →
  x^2 - y^2 / 3 = 1 :=
by
  intro a b ha hb eccentricity asymptote_tangent hab
  simp [eccentricity, asymptote_tangent] at hab
  use a, b
  split
  exact ha
  exact hb
  sorry

end hyperbola_equation_l800_800850


namespace pow_modulus_l800_800995

theorem pow_modulus : (5 ^ 2023) % 11 = 3 := by
  sorry

end pow_modulus_l800_800995


namespace ball_bounces_below_2_feet_l800_800928

theorem ball_bounces_below_2_feet :
  ∃ k : ℕ, 500 * (2 / 3 : ℝ) ^ k < 2 ∧ ∀ n < k, 500 * (2 / 3 : ℝ) ^ n ≥ 2 :=
by
  sorry

end ball_bounces_below_2_feet_l800_800928


namespace max_min_sequence_positions_l800_800236

noncomputable def a_n (n : ℕ) : ℝ :=
  if n = 0 then 0 else (n.to_real - Real.sqrt 98) / (n.to_real - Real.sqrt 99)

theorem max_min_sequence_positions :
  (∀ n ∈ Finset.range 20, n ≠ 0 → a_n n ≤ a_n 10) ∧ (∀ n ∈ Finset.range 20, n ≠ 0 → a_n n ≥ a_n 9) :=
by
  sorry

end max_min_sequence_positions_l800_800236


namespace k_plus_alpha_is_one_l800_800729

variable (f : ℝ → ℝ) (k α : ℝ)

-- Conditions from part a)
def power_function := ∀ x : ℝ, f x = k * x ^ α
def passes_through_point := f (1 / 2) = 2

-- Statement to be proven
theorem k_plus_alpha_is_one (h1 : power_function f k α) (h2 : passes_through_point f) : k + α = 1 :=
sorry

end k_plus_alpha_is_one_l800_800729


namespace evaluate_expression_l800_800966

theorem evaluate_expression : 3 + 5 * 2^3 - 4 / 2 + 7 * 3 = 62 := 
  by sorry

end evaluate_expression_l800_800966


namespace part1_part2_l800_800568

variable {A B : Type}
variables (profitA profitB : ℕ) -- Profit per car for models A and B in thousand yuan
variables (carsA carsB : ℕ) -- Number of cars for models A and B

-- Conditions as definitions
def condition1 : Prop := 3 * profitA + 2 * profitB = 34
def condition2 : Prop := profitA + 4 * profitB = 28
def condition3 : Prop := 16 * carsA + 14 * (30 - carsA) <= 440
def condition4 : Prop := 0.8 * carsA + 0.5 * (30 - carsA) >= 17.7

-- Proof statements
theorem part1 : condition1 ∧ condition2 → profitA = 8 ∧ profitB = 5 := sorry

theorem part2 : profitA = 8 ∧ profitB = 5 → condition3 ∧ condition4 → (carsA = 9 ∧ carsB = 21 ∨ carsA = 10 ∧ carsB = 20) := sorry

end part1_part2_l800_800568


namespace probability_different_grades_l800_800583

theorem probability_different_grades (A B : Type) [Fintype A] [Fintype B] (ha : Fintype.card A = 2) (hb : Fintype.card B = 2) :
  (∃ (s : Finset (A ⊕ B)), s.card = 2) →
  (Fintype.card (Finset (A ⊕ B)).filter (λ s, (∃ (a : A) (b : B), s = {sum.inl a, sum.inr b})) = 4) →
  (Fintype.card (Finset (A ⊕ B)).card-choose 2 = 6) →
  (Fintype.card (Finset (A ⊕ B)).filter (λ s, (∃ (a : A) (b : B), s = {sum.inl a, sum.inr b})) /
     Fintype.card (Finset (A ⊕ B)).card-choose 2 = 2 / 3) :=
sorry

end probability_different_grades_l800_800583


namespace ticket_price_correct_l800_800776

noncomputable def ticket_price (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 100 then
    0.5 * x
  else if 100 < x then
    0.4 * x + 10
  else
    0

theorem ticket_price_correct (x : ℝ) :
  (0 < x ∧ x ≤ 100 → ticket_price x = 0.5 * x) ∧
  (100 < x → ticket_price x = 0.4 * x + 10) :=
by
  split
  · intro h
    simp [ticket_price, h]
  · intro h
    simp [ticket_price, h, not_and]
    push_neg
    intro h₁
    linarith

end ticket_price_correct_l800_800776


namespace sequence_34th_term_l800_800286

theorem sequence_34th_term :
  let a : ℕ → ℚ := λ n, nat.rec_on n 1 (λ n an, an / (3 * an + 1)) in
  a 34 = 1/100 :=
by
  sorry

end sequence_34th_term_l800_800286


namespace cosine_difference_l800_800247

open Real

theorem cosine_difference (a : ℝ)
  (a_pos : 0 < a)
  (a_lt_pi_div_2 : a < π / 2)
  (tan_a : tan a = 2) :
  cos (a - π / 4) = 3 * sqrt 10 / 10 :=
by
  sorry

end cosine_difference_l800_800247


namespace probability_of_one_male_one_female_l800_800779

theorem probability_of_one_male_one_female :
  let total_students := 4
  let male_students := 1
  let female_students := 3
  let ways_to_select_2 := Nat.choose 4 2
  let ways_to_select_1_male_1_female := Nat.choose 3 1
  ∃ p, p = (ways_to_select_1_male_1_female * male_students) / ways_to_select_2 ∧ p = 1 / 2 := by
  sorry

end probability_of_one_male_one_female_l800_800779


namespace find_m_l800_800335

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, ∃ q : ℝ, (q > 0) ∧ (a n ≠ 0) ∧ (n ≥ 2) → a (n+1) * a (n-1) = 2 * a n

def product_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
∏ i in finset.range n, a i

def T_n (a : ℕ → ℝ) (n : ℕ) : ℝ := product_first_n_terms a n

theorem find_m (a : ℕ → ℝ) (m : ℕ) : 
  geometric_sequence a →
  T_n a (2*m - 1) = 512 →
  m = 5 :=
by
  sorry

end find_m_l800_800335


namespace sum_of_slopes_is_1_l800_800450

theorem sum_of_slopes_is_1 :
  ∀ (P Q R S : ℤ × ℤ), 
  P = (10, 50) ∧ S = (11, 51) ∧
  ¬ (∃ x, P.1 = x ∨ Q.1 = x ∨ R.1 = x ∨ S.1 = x) ∧
  ¬ (∃ y, P.2 = y ∨ Q.2 = y ∨ R.2 = y ∨ S.2 = y) ∧
  (∃ k l : ℤ, (Q.2 - P.2) * (R.2 - S.2) = k * l) 
  → (abs 1 + abs 0) = 1 := 
by
  sorry

end sum_of_slopes_is_1_l800_800450


namespace math_olympiad_proof_l800_800352

theorem math_olympiad_proof (scores : Fin 20 → ℕ) 
  (h_diff : ∀ i j, i ≠ j → scores i ≠ scores j) 
  (h_sum : ∀ i j k, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k) : 
  ∀ i, scores i > 18 :=
by
  sorry

end math_olympiad_proof_l800_800352


namespace cost_of_paving_l800_800097

def length : ℝ := 5.5
def width : ℝ := 3.75
def rate_per_sqm : ℝ := 1400
def expected_cost : ℝ := 28875

theorem cost_of_paving (l w r : ℝ) (h_l : l = length) (h_w : w = width) (h_r : r = rate_per_sqm) :
  (l * w * r) = expected_cost := by
  sorry

end cost_of_paving_l800_800097


namespace problem1_problem2_monotonicity_problem3_l800_800264

-- Definition for f(x)
def f (x : ℝ) : ℝ := 3^x - 1 / 3^|x|

-- Problem 1: If f(x) = 2, find x
theorem problem1 (x : ℝ) (h : f(x) = 2) : x = Real.log 3 (1 + Real.sqrt 2) :=
sorry

-- Problem 2: Determine the monotonicity of f(x) when x > 0
theorem problem2_monotonicity (x : ℝ) (h : x > 0) : monotone_on f (set.Ioi 0) :=
sorry

-- Problem 3: Find the range of m for the given inequality
theorem problem3 (t m : ℝ) (h_t : t ∈ set.Icc (1 / 2) 1) (h_ineq : ∀ t ∈ set.Icc (1 / 2) 1, 3^t * f(t) + m * f(t) ≥ 0) : m ≥ -4 :=
sorry

end problem1_problem2_monotonicity_problem3_l800_800264


namespace min_dose_for_effectiveness_max_effective_duration_l800_800606

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 4 then - (1 / 2) * x^2 + 2 * x + 8
  else if 4 < x ∧ x ≤ 16 then - (1 / 2) * x - (real.log x / real.log 2) + 12
  else 0

theorem min_dose_for_effectiveness (m : ℝ) (x : ℝ) (h_m : 0 < m) (h_x : 0 < x ∧ x ≤ 8) :
  (m ≥ 12 / 5) → (m * f x ≥ 12) := sorry

noncomputable def g (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 4 then -x^2 + 4 * x + 16
  else if 4 < x ∧ x ≤ 16 then -x - 2 * (real.log x / real.log 2) + 24
  else 0

theorem max_effective_duration (x : ℝ) (h_x : 0 < x ∧ x ≤ 16) :
  (g 6 ≥ 12) ∧ (g 7 < 12) → x ≤ 6 := sorry

end min_dose_for_effectiveness_max_effective_duration_l800_800606


namespace hyperbola_foci_coordinates_l800_800276

theorem hyperbola_foci_coordinates (a b : ℝ) (hy : a^2 = 16) (hx : b^2 = 9) :
    ∃ c : ℝ, c = sqrt (a^2 + b^2) ∧ (0, c) ∈ setOf (λ (x : ℝ × ℝ), x.2 = 5) ∧ (0, -c) ∈ setOf (λ (x : ℝ × ℝ), x.2 = -5) :=
by
  let c := sqrt (a^2 + b^2)
  use c
  split
  · exact rfl
  · split <;> simp [c, hy, hx]
  sorry

end hyperbola_foci_coordinates_l800_800276


namespace sum_bottom_row_is_18_l800_800189

-- Define the sets and sums needed
constant squares : Fin 9 → ℕ
constant G H I : ℕ 

axiom no_repeat : Function.Injective squares
axiom range_constraint : ∀ i, squares i ∈ finset.range 10
axiom sum_vertical_right : squares 2 + squares 5 + squares 8 = 32
axiom shared_square : squares 8 = 7
axiom sum_bottom : squares 6 = G ∧ squares 7 = H ∧ squares 8 = I

-- The theorem to be proved
theorem sum_bottom_row_is_18 : G + H + I = 18 :=
sorry

end sum_bottom_row_is_18_l800_800189


namespace intervals_of_monotonicity_range_of_a_for_zeros_l800_800268

open Real

noncomputable def f (x a : ℝ) : ℝ := (1/2) * x^2 - 3 * a * x + 2 * a^2 * log x

theorem intervals_of_monotonicity (a : ℝ) (ha : a ≠ 0) :
  (0 < a → ∀ x, (0 < x ∧ x < a → f x a < f (x + 1) a)
            ∧ (a < x ∧ x < 2 * a → f x a > f (x + 1) a)
            ∧ (2 * a < x → f x a < f (x + 1) a))
  ∧ (a < 0 → ∀ x, (0 < x → f x a < f (x + 1) a)) :=
sorry

theorem range_of_a_for_zeros (a x : ℝ) (ha : 0 < a) 
  (h1 : f a a > 0) (h2 : f (2 * a) a < 0) :
  e ^ (5 / 4) < a ∧ a < e ^ 2 / 2 :=
sorry

end intervals_of_monotonicity_range_of_a_for_zeros_l800_800268


namespace find_c_l800_800475

theorem find_c (x c : ℤ) (h1 : 3 * x + 9 = 0) (h2 : c * x - 5 = -11) : c = 2 := by
  have x_eq : x = -3 := by
    linarith
  subst x_eq
  have c_eq : c = 2 := by
    linarith
  exact c_eq

end find_c_l800_800475


namespace olympiad_scores_above_18_l800_800340

theorem olympiad_scores_above_18 
  (n : Nat) 
  (scores : Fin n → ℕ) 
  (h_diff_scores : ∀ i j : Fin n, i ≠ j → scores i ≠ scores j) 
  (h_score_sum : ∀ i j k : Fin n, i ≠ j ∧ i ≠ k ∧ j ≠ k → scores i < scores j + scores k) 
  (h_n : n = 20) : 
  ∀ i : Fin n, scores i > 18 := 
by 
  -- See the proof for the detailed steps.
  sorry

end olympiad_scores_above_18_l800_800340


namespace largest_four_digit_number_property_l800_800068

theorem largest_four_digit_number_property : ∃ (a b c d : Nat), 
  (1000 * a + 100 * b + 10 * c + d = 9099) ∧ 
  (c = a + b) ∧ 
  (d = b + c) ∧ 
  1 ≤ a ∧ a ≤ 9 ∧ 
  0 ≤ b ∧ b ≤ 9 ∧ 
  0 ≤ c ∧ c ≤ 9 ∧ 
  0 ≤ d ∧ d ≤ 9 := 
sorry

end largest_four_digit_number_property_l800_800068


namespace sin_of_2000_deg_l800_800302

theorem sin_of_2000_deg (a : ℝ) (h : Real.tan (160 * Real.pi / 180) = a) : 
  Real.sin (2000 * Real.pi / 180) = -a / Real.sqrt (1 + a^2) := 
by
  sorry

end sin_of_2000_deg_l800_800302


namespace future_value_of_investment_is_correct_l800_800918

theorem future_value_of_investment_is_correct :
  let P := 10000
  let r := 0.0396
  let n := 2
  let t := 2
  let A := P * (1 + r / n) ^ (n * t)
  A ≈ 10816.49 :=
by
  sorry

end future_value_of_investment_is_correct_l800_800918


namespace sequence_sum_l800_800793

theorem sequence_sum :
  let a : ℕ → ℕ := λ n, if n = 0 then 1 else if n = 1 then 2 else a (n - 2) + 1 + (-1)^n in
  let S_100 := ∑ i in finset.range 100, a i in
  S_100 = 2600 :=
by
  sorry

end sequence_sum_l800_800793


namespace alex_sandwich_count_l800_800143

-- Conditions: Alex has 12 kinds of lunch meat and 8 kinds of cheese
def lunch_meat := fin 12
def cheese := fin 8

-- The problem is to prove the total number of different sandwiches Alex can make is 528
theorem alex_sandwich_count : (nat.choose 12 2) * (nat.choose 8 1) = 528 := by
  sorry

end alex_sandwich_count_l800_800143


namespace smallest_positive_period_mono_increasing_interval_max_altitude_l800_800733

def f (x : ℝ) : ℝ := sqrt 3 * cos (2 * x) + 2 * sin ((3 / 2) * π + x) * sin (π - x)

theorem smallest_positive_period (x : ℝ) :
  ∃ T > 0, ∀ x ∈ ℝ, f (x + T) = f x ∧ T = π :=
sorry

theorem mono_increasing_interval (k : ℤ) :
  ∀ x ∈ ℝ, (k * π + 5 / 12 * π) ≤ x ∧ x ≤ (k * π + 11 / 12 * π) :=
sorry

def altitude_max (A : ℝ) (a b c : ℝ) (h : ℝ) :=
  a = 3 ∧ f A = -sqrt 3 ∧ A = π / 3 ∧ a * h ≤ b * c * sin A / 2

theorem max_altitude (a b c h : ℝ) :
  altitude_max (π / 3) a b c h → h = 3 * sqrt 3 / 2 :=
sorry

end smallest_positive_period_mono_increasing_interval_max_altitude_l800_800733


namespace tangent_line_at_origin_l800_800659

def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

-- Assuming natural calculus derivative definitions in Mathlib
theorem tangent_line_at_origin :
  let m := (Real.cos 0 - Real.sin 0)
  let c := f 0
  m = 1 ∧ c = 1 ∧ (∀ x y : ℝ, y = m * x + c → x - y + 1 = 0) :=
by
  sorry

end tangent_line_at_origin_l800_800659


namespace shopkeeper_gain_is_8_51_percent_l800_800591

def false_weight_A : ℝ := 950 / 1000
def false_weight_B : ℝ := 900 / 1000
def false_weight_C : ℝ := 920 / 1000

def cost_price_A : ℝ := 20
def cost_price_B : ℝ := 25
def cost_price_C : ℝ := 18

def amount_sold_A : ℝ := 10
def amount_sold_B : ℝ := 10
def amount_sold_C : ℝ := 10

noncomputable def shopkeeper_percentage_gain : ℝ :=
  let actual_weight_A := false_weight_A * amount_sold_A
  let actual_weight_B := false_weight_B * amount_sold_B
  let actual_weight_C := false_weight_C * amount_sold_C
  let total_cost_A := actual_weight_A * cost_price_A
  let total_cost_B := actual_weight_B * cost_price_B
  let total_cost_C := actual_weight_C * cost_price_C
  let total_cost_price := total_cost_A + total_cost_B + total_cost_C
  let selling_price_A := cost_price_A * amount_sold_A
  let selling_price_B := cost_price_B * amount_sold_B
  let selling_price_C := cost_price_C * amount_sold_C
  let total_selling_price := selling_price_A + selling_price_B + selling_price_C
  let gain := total_selling_price - total_cost_price
  (gain / total_cost_price) * 100

theorem shopkeeper_gain_is_8_51_percent : shopkeeper_percentage_gain ≈ 8.51 := by
  -- Proof goes here
  sorry

end shopkeeper_gain_is_8_51_percent_l800_800591


namespace balls_boxes_total_ways_one_box_empty_ways_two_boxes_empty_ways_l800_800497

theorem balls_boxes_total_ways : (number of ways to put 4 different balls into 4 different boxes) = 256 := 
sorry

theorem one_box_empty_ways : (number of ways to put 4 different balls into 4 different boxes with exactly one box empty) = 144 := 
sorry

theorem two_boxes_empty_ways : (number of ways to put 4 different balls into 4 different boxes with exactly two boxes empty) = 84 := 
sorry

end balls_boxes_total_ways_one_box_empty_ways_two_boxes_empty_ways_l800_800497


namespace collinear_points_l800_800627

variables (E F N M B D : Type*)
variables [loc : has_points_on a_line E F N M B D]

theorem collinear_points
(BC CD : E F)
(E_on_BC : E ∈ BC)
(F_on_CD : F ∈ CD)
(EN_perp_AF : ∠EN = 90)
(FM_perp_AE : ∠FM = 90)
(EAF_eq_45 : ∠EAF = 45)
: collinear B M N D := 
sorry

end collinear_points_l800_800627


namespace regular_polygon_center_zero_sum_l800_800807

open_locale big_operators

-- Defining the conditions
variables {n : ℕ} (A : fin n → ℝ × ℝ)
def length (P Q : ℝ × ℝ) : ℝ := ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2).sqrt
def vector_from (P Q : ℝ × ℝ) : ℝ × ℝ := (Q.1 - P.1, Q.2 - P.2)
def weighted_vector_sum (O : ℝ × ℝ) : (ℝ × ℝ) :=
  ∑ k in finset.range n, k * (vector_from O (A k)) / (length O (A k)) ^ 5

-- Stating the theorem
theorem regular_polygon_center_zero_sum
  (n : ℕ) (hn : n = 2017) (A : fin n → ℝ × ℝ) (O : ℝ × ℝ)
  (hA : ∀ k, k < n → A k ≠ O) (hr : is_regular_polygon A O) :
  weighted_vector_sum A O = (0,0) :=
sorry

end regular_polygon_center_zero_sum_l800_800807


namespace value_of_a_l800_800768

-- Definitions for the problem conditions
def line1 (a : ℝ) := λ x y : ℝ, a * x + 2 * y + 1 = 0
def line2 := λ x y : ℝ, x + 3 * y - 2 = 0

-- Mathematically equivalent proof problem in Lean
theorem value_of_a (a : ℝ) (h : ∀ x y : ℝ, (line1 a) x y ↔ (line2 x y)) : a = -6 :=
by {
  sorry
}

end value_of_a_l800_800768


namespace eval_expression_l800_800193

theorem eval_expression :
  ( (81 / 16) ^ (-1 / 4) + log (2: ℝ) (4 ^ 3 * 2 ^ 4) = 32 / 3 ) :=
by
  sorry

end eval_expression_l800_800193


namespace rhombus_perimeter_l800_800454

theorem rhombus_perimeter
  (d1 d2 : ℝ)
  (h1 : d1 = 20)
  (h2 : d2 = 16) :
  4 * (Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)) = 8 * Real.sqrt 41 := 
  sorry

end rhombus_perimeter_l800_800454


namespace car_speed_is_approximately_one_mph_l800_800109

noncomputable def distance_in_miles (yards : ℕ) : ℝ :=
  yards / 1760.0

noncomputable def time_in_hours (minutes : ℝ) : ℝ :=
  minutes / 60

noncomputable def speed_in_mph (yards : ℕ) (minutes : ℝ) : ℝ :=
  distance_in_miles yards / time_in_hours minutes

theorem car_speed_is_approximately_one_mph :
  (speed_in_mph 106 3.61) ≈ 1 :=
by
  sorry

end car_speed_is_approximately_one_mph_l800_800109


namespace cone_base_circumference_l800_800615

theorem cone_base_circumference (r : ℝ) (θ : ℝ) (h_r : r = 6) (h_θ : θ = 180) :
  let full_circle_circumference := 2 * Real.pi * r in
  let cone_base_circumference := (θ / 360) * full_circle_circumference in
  cone_base_circumference = 6 * Real.pi :=
by
  sorry

end cone_base_circumference_l800_800615


namespace base8_to_base10_problem_l800_800439

theorem base8_to_base10_problem (c d : ℕ) (h : 543 = 3*8^2 + c*8 + d) : (c * d) / 12 = 5 / 4 :=
by 
  sorry

end base8_to_base10_problem_l800_800439


namespace roots_real_distinct_roots_real_equal_roots_complex_roots_real_distinct_positive_roots_real_distinct_negative_roots_opposite_signs_l800_800921

variables (x y : ℝ)

def discriminant (x y : ℝ) : ℝ := 4 - 4 * y^2 - x^2

theorem roots_real_distinct : discriminant x y > 0 ↔ (x / 2) ^ 2 + y ^ 2 < 1 :=
by sorry

theorem roots_real_equal : discriminant x y = 0 ↔ (x / 2) ^ 2 + y ^ 2 = 1 :=
by sorry

theorem roots_complex : discriminant x y < 0 ↔ (x / 2) ^ 2 + y ^ 2 > 1 :=
by sorry

theorem roots_real_distinct_positive : 
  discriminant x y > 0 ∧ y > 0 ∧ (5 * y ^ 2 + x ^ 2 - 4 > 0) ↔ 
  (x / 2) ^ 2 + (y / sqrt (4/5)) ^ 2 > 1 ∧ y > 0 :=
by sorry

theorem roots_real_distinct_negative : 
  discriminant x y > 0 ∧ y < 0 ∧ (5 * y ^ 2 + x ^ 2 - 4 > 0) ↔ 
  (x / 2) ^ 2 + (y / sqrt (4/5)) ^ 2 > 1 ∧ y < 0 :=
by sorry

theorem roots_opposite_signs : 
  discriminant x y > 0 ∧ (5 * y ^ 2 + x ^ 2 - 4 < 0) ↔ 
  (x / 2) ^ 2 + (y / sqrt (4/5)) ^ 2 < 1 :=
by sorry

end roots_real_distinct_roots_real_equal_roots_complex_roots_real_distinct_positive_roots_real_distinct_negative_roots_opposite_signs_l800_800921


namespace krista_bank_balance_exceeds_5000_on_tuesday_l800_800804

-- Krista's deposit pattern definition
def krista_deposit (n : ℕ) : ℕ :=
  if n = 0 then 5
  else 5 * 2^n

-- Sum of deposits up to day n
def krista_total (n : ℕ) : ℕ :=
  (List.range (n + 1)).sum $ λ i, krista_deposit i

-- Main theorem stating that Krista's bank balance first exceeds 5000 cents on the tenth day, which is a Tuesday
theorem krista_bank_balance_exceeds_5000_on_tuesday : ∃ n : ℕ, 10 = n ∧ krista_total n > 5000 ∧ (n % 7 = 3) :=
by
  sorry

end krista_bank_balance_exceeds_5000_on_tuesday_l800_800804


namespace coloring_minimum_colors_l800_800540

theorem coloring_minimum_colors (n : ℕ) (h : n = 2016) : ∃ k : ℕ, k = 11 ∧
  (∀ (board : Matrix ℕ ℕ ℕ), 
    let color : ℕ → ℕ → ℕ := λ i j, if i = j then 0           -- Main diagonal colored '0'
                                      else if i < j then j - i -- Left of the diagonal
                                      else i - j               -- Right of the diagonal
    ∧ (∀ i < n, ∀ j < n, color i j = color j i)                -- Symmetry wrt diagonal
    ∧ (∀ i < n, ∀ j k, j ≠ k → color i j ≠ color i k)          -- Different colors in the row  
    ) :=
begin
  use 11,
  split,
  { refl },
  { intros board,
    let color := λ i j, if i = j then 0 else if i < j then j - i else i - j,
    split,
    { intros i hi j hj,
      exact (if i < j then rfl else rfl), },
    { intros i hi j k hjk,
      by_cases h : i = j,
      { rw h,
        exact hjk.elim, },
      { by_cases h' : j < k,
        { rw if_pos h' },
        { by_contradiction,
          have := nat.le_antisymm,
          contradiction } } } }
end

end coloring_minimum_colors_l800_800540


namespace series_value_l800_800494

theorem series_value : 
  let series := List.range' 0 51
                  |>.map (fun n => 100 - 2 * n)
  in series.sum = 50 := 
by
  sorry

end series_value_l800_800494


namespace hcl_mixture_problem_l800_800736

theorem hcl_mixture_problem:
  ∃ V : ℝ, 
  0.30 * (30.0 + V) = 0.10 * 30.0 + 0.60 * V ∧
  (30.0 + V) = 50.0 :=
by
  use 20.0
  split
  sorry
  sorry

end hcl_mixture_problem_l800_800736


namespace max_x2_minus_2xy_plus_y2_l800_800844

open Real

theorem max_x2_minus_2xy_plus_y2 (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : x^2 + 2 * x * y + y^2 = 9) :
  x^2 - 2 * x * y + y^2 ≤ 9 / 4 :=
begin
  sorry
end

end max_x2_minus_2xy_plus_y2_l800_800844


namespace abs_neg_five_l800_800846

theorem abs_neg_five : abs (-5) = 5 :=
by
  sorry

end abs_neg_five_l800_800846


namespace min_val_sin4_x_plus_tan4_x_l800_800210

theorem min_val_sin4_x_plus_tan4_x :
  ∃ x : ℝ, ∀ y : ℝ, sin y ^ 4 + tan y ^ 4 ≥ sin x ^ 4 + tan x ^ 4 ∧ sin x ^ 4 + tan x ^ 4 = 0 :=
by
  sorry

end min_val_sin4_x_plus_tan4_x_l800_800210


namespace triangle_circle_intersection_BL_eq_2_sqrt_14_l800_800896

theorem triangle_circle_intersection_BL_eq_2_sqrt_14
  (A B C L : Point)
  (h_AB : dist A B = 6)
  (h_BC : dist B C = 7)
  (h_CA : dist C A = 8)
  (ω1 : Circle)
  (h_ω1 : ω1.contains C ∧ ω1.isTangentAtLine AB B)
  (ω2 : Circle)
  (h_ω2 : ω2.contains A ∧ ω2.isTangentAtLine BC B)
  (h_L_intersection : L ≠ B ∧ L ∈ ω1 ∧ L ∈ ω2) :
  dist B L = 2 * Real.sqrt 14 := 
sorry

end triangle_circle_intersection_BL_eq_2_sqrt_14_l800_800896


namespace least_number_of_cans_l800_800592

theorem least_number_of_cans :
  let quantities := [139, 223, 179, 199, 173, 211, 131, 257] in
  quantities.sum = 1412 :=
by
  let quantities := [139, 223, 179, 199, 173, 211, 131, 257]
  have h : quantities.sum = 1412 := by sorry
  exact h

end least_number_of_cans_l800_800592


namespace olympiad_scores_above_18_l800_800338

theorem olympiad_scores_above_18 
  (n : Nat) 
  (scores : Fin n → ℕ) 
  (h_diff_scores : ∀ i j : Fin n, i ≠ j → scores i ≠ scores j) 
  (h_score_sum : ∀ i j k : Fin n, i ≠ j ∧ i ≠ k ∧ j ≠ k → scores i < scores j + scores k) 
  (h_n : n = 20) : 
  ∀ i : Fin n, scores i > 18 := 
by 
  -- See the proof for the detailed steps.
  sorry

end olympiad_scores_above_18_l800_800338


namespace discount_percentage_l800_800848

  variable (MP CP SP : ℝ)
  variable (D : ℝ)

  -- Definitions of the conditions
  def cost_price_condition : Prop := CP = 0.64 * MP
  def selling_price_gain_condition : Prop := SP = CP * 1.359375
  def selling_price_discount_condition : Prop := SP = MP * (1 - D / 100)
  
  -- The statement to be proven
  theorem discount_percentage (h1 : cost_price_condition) (h2 : selling_price_gain_condition) (h3 : selling_price_discount_condition) : D = 13.04 := sorry
  
end discount_percentage_l800_800848


namespace initial_mean_l800_800863

theorem initial_mean (M : ℝ) (n : ℕ) (observed_wrongly correct_wrongly : ℝ) (new_mean : ℝ) :
  n = 50 ∧ observed_wrongly = 23 ∧ correct_wrongly = 45 ∧ new_mean = 36.5 → M = 36.06 :=
by
  intro h
  cases h with h1 h2
  cases h2 with h3 h4
  cases h4 with h5 h6
  have sum_initial := n * M
  have diff := correct_wrongly - observed_wrongly
  have sum_corrected := sum_initial + diff
  have new_sum_corrected := n * new_mean
  have equation := sum_corrected = new_sum_corrected
  rw [←h3, ←h2, ←h1, ←h6] at equation
  have sum_initial_calculated := new_sum_corrected - diff
  have M_calculated_eq := sum_initial_calculated / n
  rw [←h5, ←h1] at M_calculated_eq
  -- Calculate manually to show final M == 36.06 which is the correct proof
  sorry

end initial_mean_l800_800863


namespace ball_reaches_less_than_2_feet_after_14_bounces_l800_800930

theorem ball_reaches_less_than_2_feet_after_14_bounces :
  ∀ (h₀ : ℝ) (r : ℝ), h₀ = 500 → r = 2 / 3 →
  ∃ (k : ℕ), k = 14 ∧ h₀ * r^k < 2 := by
  intros h₀ r h₀_eq r_eq
  use 14
  rw [h₀_eq, r_eq]
  norm_num
  apply lt_trans
    (norm_num [500 * (2 / 3)^14])
  norm_num [2]
  sorry -- Proof for the exact value comparison

end ball_reaches_less_than_2_feet_after_14_bounces_l800_800930


namespace total_painting_area_correct_l800_800107

def barn_width : ℝ := 12
def barn_length : ℝ := 15
def barn_height : ℝ := 6

def area_to_be_painted (width length height : ℝ) : ℝ := 
  2 * (width * height + length * height) + width * length

theorem total_painting_area_correct : area_to_be_painted barn_width barn_length barn_height = 828 := 
  by sorry

end total_painting_area_correct_l800_800107


namespace marble_probability_l800_800106

theorem marble_probability :
  let redMarbles := 4
  let blueMarbles := 6
  let totalMarbles := redMarbles + blueMarbles
  let firstRedProb := (redMarbles: ℝ) / totalMarbles
  let remainingMarbles := totalMarbles - 1
  let secondBlueProb := (blueMarbles: ℝ) / remainingMarbles
  let combinedProb := firstRedProb * secondBlueProb
  combinedProb = 4 / 15 :=
by
  sorry

end marble_probability_l800_800106


namespace quadratic_roots_l800_800387

variable (ω : ℂ) (h₁ : ω^9 = 1) (h₂ : ω ≠ 1)
def α : ℂ := ω + ω^3 + ω^6
def β : ℂ := ω^2 + ω^5 + ω^8

theorem quadratic_roots : (α ω h₁ h₂)^2 + 0 * (α ω h₁ h₂) + 0 = 0 ∧ (β ω h₁ h₂)^2 + 0 * (β ω h₁ h₂) + 0 = 0 := by
  sorry

end quadratic_roots_l800_800387


namespace min_AB_CD_l800_800279

theorem min_AB_CD {p : ℝ} (p_pos : p > 0) :
  ∀ (A B C D : ℝ × ℝ), on_parabola A B C D  &&
  mutually_perpendicular A B C D -> passing_through_origin A B C D -> 
  |AB|  + |CD | = 16 * p.
sorry

end min_AB_CD_l800_800279


namespace min_colors_2016x2016_l800_800546

def min_colors_needed (n : ℕ) : ℕ :=
  ⌈log 2 (n * n)⌉.natAbs

theorem min_colors_2016x2016 :
  min_colors_needed 2016 = 11 := 
by
  sorry

end min_colors_2016x2016_l800_800546


namespace locus_of_centers_l800_800041

noncomputable def locus_of_centers_of_triangle (circle1 circle2 : Circle) (A P : Point) (locus : Point -> Prop) : Prop :=
  ∀ (B C : Point), 
    (on_circle B circle1) →
    (on_circle C circle2) →
    (line_through P B) →
    (line_through P C) →
    (B ≠ P) →
    (C ≠ P) →
    on_circle (centroid_of_triangle A B C) locus ∧
    on_circle (circumcenter_of_triangle A B C) locus ∧
    on_circle (orthocenter_of_triangle A B C) locus ∧
    on_circle (incenter_of_triangle A B C) locus

theorem locus_of_centers (circle1 circle2 : Circle) (A P : Point) :
  locus_of_centers_of_triangle circle1 circle2 A P (λ Q, on_circle Q (circle_passing_through A)) :=
sorry

end locus_of_centers_l800_800041


namespace area_of_ABCS_l800_800126

section StarProof

variables (S : Type) [metric_space S] [has_area S]
variables (A B C D E F G H I J K L : point S)
variables (star : regular_six_pointed_star A B C D E F G H I J K L)
variables (area_equilateral_triangle : has_area.equilateral_triangle S 72)

theorem area_of_ABCS (ABCS : area (quadrilateral A B C S)) : area ABCS = 16 := 
sorry

end StarProof

end area_of_ABCS_l800_800126


namespace rhombus_perimeter_l800_800468

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 20) (h2 : d2 = 16) : ∃ p, p = 8 * Real.sqrt 41 := by
  let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  have h3 : s = 2 * Real.sqrt 41 := by sorry
  let p := 4 * s
  have h4 : p = 8 * Real.sqrt 41 := by sorry
  exact ⟨p, h4⟩

end rhombus_perimeter_l800_800468


namespace triangle_types_l800_800325

noncomputable def triangle1 := (5 : ℝ, 6 : ℝ, 7 : ℝ)
noncomputable def triangle2 := (5 : ℝ, 12 : ℝ, 13 : ℝ)
noncomputable def triangle3 := (2 : ℝ, 3 : ℝ, 4 : ℝ)

def cosine_rule (a b c : ℝ) : ℝ :=
  (a * a + b * b - c * c) / (2 * a * b)

def is_acute (a b c : ℝ) : Prop :=
  cosine_rule a b c > 0

def is_right (a b c : ℝ) : Prop :=
  cosine_rule a b c = 0

def is_obtuse (a b c : ℝ) : Prop :=
  cosine_rule a b c < 0

theorem triangle_types :
  is_acute (5 : ℝ) (6 : ℝ) (7 : ℝ) ∧
  is_right (5 : ℝ) (12 : ℝ) (13 : ℝ) ∧
  is_obtuse (2 : ℝ) (3 : ℝ) (4 : ℝ) :=
by
  sorry

end triangle_types_l800_800325


namespace clock_rotation_difference_l800_800955

-- Definitions for the problem conditions
def hour_rotation_per_hour : ℝ := -30
def minute_rotation_per_minute : ℝ := -6
def total_hours : ℝ := 3 + 35 / 60
def total_minutes : ℕ := 3 * 60 + 35

-- The Lean 4 theorem statement
theorem clock_rotation_difference :
  hour_rotation_per_hour * total_hours + 
  minute_rotation_per_minute * total_minutes = 1182.5 := by
sorry

end clock_rotation_difference_l800_800955


namespace find_c_l800_800327

theorem find_c (y c : ℝ) (h : y > 0) (h₂ : (8*y)/20 + (c*y)/10 = 0.7*y) : c = 6 :=
by
  sorry

end find_c_l800_800327


namespace range_of_a_l800_800017

noncomputable def a_n (a : ℝ) (n : ℕ) : ℝ :=
  if n <= 7 then (3 - a) * n - 3 else a ^ (n - 6)

def increasing_seq (a : ℝ) (n : ℕ) : Prop :=
  a_n a n < a_n a (n + 1)

theorem range_of_a (a : ℝ) :
  (∀ n, increasing_seq a n) ↔ (9 / 4 < a ∧ a < 3) :=
sorry

end range_of_a_l800_800017


namespace coloring_minimum_colors_l800_800543

theorem coloring_minimum_colors (n : ℕ) (h : n = 2016) : ∃ k : ℕ, k = 11 ∧
  (∀ (board : Matrix ℕ ℕ ℕ), 
    let color : ℕ → ℕ → ℕ := λ i j, if i = j then 0           -- Main diagonal colored '0'
                                      else if i < j then j - i -- Left of the diagonal
                                      else i - j               -- Right of the diagonal
    ∧ (∀ i < n, ∀ j < n, color i j = color j i)                -- Symmetry wrt diagonal
    ∧ (∀ i < n, ∀ j k, j ≠ k → color i j ≠ color i k)          -- Different colors in the row  
    ) :=
begin
  use 11,
  split,
  { refl },
  { intros board,
    let color := λ i j, if i = j then 0 else if i < j then j - i else i - j,
    split,
    { intros i hi j hj,
      exact (if i < j then rfl else rfl), },
    { intros i hi j k hjk,
      by_cases h : i = j,
      { rw h,
        exact hjk.elim, },
      { by_cases h' : j < k,
        { rw if_pos h' },
        { by_contradiction,
          have := nat.le_antisymm,
          contradiction } } } }
end

end coloring_minimum_colors_l800_800543


namespace major_premise_incorrect_l800_800976

-- Define the function and its derivative
def f (x : ℝ) := x^3
def f' (x : ℝ) := 3 * x^2

-- State the theorem with conditions and incorrect major premise
theorem major_premise_incorrect :
  (f' 0 = 0) ∧ (∀ x, f' x = 3 * x^2) ∧ (∃ x₀, f' x₀ = 0 ∧ ∀ ε > 0, ∃ δ > 0, (|x - x₀| < δ → f'(x₀) * f'(x) < 0 → x ≠ x₀) → f x₀ has no sign change) →
  ¬(∀ x₀, f' x₀ = 0 → (∃ ε > 0, ∀ x, |x - x₀| < ε → f x₀ = x₀))
:=
by
  sorry

end major_premise_incorrect_l800_800976


namespace martin_total_distance_l800_800410

-- Define the conditions
def total_trip_time : ℕ := 8
def first_half_speed : ℕ := 70
def second_half_speed : ℕ := 85
def half_trip_time : ℕ := total_trip_time / 2

-- Define the total distance traveled 
def total_distance : ℕ := (first_half_speed * half_trip_time) + (second_half_speed * half_trip_time)

-- Statement to prove
theorem martin_total_distance : total_distance = 620 :=
by
  -- This is a placeholder to represent that a proof is needed
  -- Actual proof steps are omitted as instructed
  sorry

end martin_total_distance_l800_800410


namespace problem_equivalent_l800_800306

open Real

theorem problem_equivalent (x : ℝ) (hx : x ∈ Ioo (exp (-1)) 1) :
  let a := log x
  let b := 2 * log x
  let c := (log x) ^ 3
  in b < a ∧ a < c :=
by
  let a := log x
  let b := 2 * log x
  let c := (log x) ^ 3
  sorry

end problem_equivalent_l800_800306


namespace no_positive_integers_q_t_satisfy_eq_l800_800762

theorem no_positive_integers_q_t_satisfy_eq (q t : ℕ) (hq : 0 < q) (ht : 0 < t) : qt + q + t ≠ 6 :=
sorry

end no_positive_integers_q_t_satisfy_eq_l800_800762


namespace side_length_square_is_9_l800_800034

-- Define the side length of the hexagon
def side_length_hexagon : ℝ := 6

-- Define the number of sides of the hexagon
def num_sides_hexagon : ℕ := 6

-- Calculate the perimeter of the hexagon
def perimeter_hexagon : ℝ := num_sides_hexagon * side_length_hexagon

-- Define the number of sides of the square
def num_sides_square : ℕ := 4

-- Define the side length of the square
def side_length_square : ℝ := perimeter_hexagon / num_sides_square

theorem side_length_square_is_9 :
  side_length_square = 9 :=
  by
    sorry

end side_length_square_is_9_l800_800034


namespace balls_prob_l800_800156

theorem balls_prob (N : ℕ) (h : (6/15) * (20/(20 + N)) + (9/15) * (N/(20 + N)) = 0.65) : N = 100 :=
sorry

end balls_prob_l800_800156


namespace problem1_problem2_l800_800169

-- Problem 1: Calculate (1)(√16)² - √25 + √((-2)²) == 13
theorem problem1 : (1 : ℝ) * (real.sqrt 16) ^ 2 - real.sqrt 25 + real.sqrt ((-2) ^ 2) = 13 := 
by 
  sorry

-- Problem 2: Calculate √(1/2) * √48 ÷ √(1/8) == 8√3
theorem problem2 : real.sqrt (1/2) * real.sqrt 48 / real.sqrt (1/8) = 8 * real.sqrt 3 := 
by 
  sorry

end problem1_problem2_l800_800169


namespace probability_of_sharing_education_of_love_probability_of_sharing_journey_and_education_of_love_l800_800894

def xiaoying_books := ["Journey to the West", "Romance of the Three Kingdoms", "How Steel Is Made", "Education of Love"]

def number_of_books := list.length xiaoying_books

theorem probability_of_sharing_education_of_love (h : number_of_books = 4) :
  (1 / 4) = 0.25 := by
  rw h
  simp
  norm_num

theorem probability_of_sharing_journey_and_education_of_love (h : number_of_books = 4) :
  (2 / (number_of_books * (number_of_books - 1))) = 1 / 6 := by
  rw h
  simp
  norm_num
  exact (by norm_num : 2 / 12 = 1 / 6)

end probability_of_sharing_education_of_love_probability_of_sharing_journey_and_education_of_love_l800_800894


namespace sum_of_roots_l800_800086

theorem sum_of_roots (x : ℝ) :
  (x^2 = 10 * x - 13) → ∃ s, s = 10 := 
by
  sorry

end sum_of_roots_l800_800086


namespace int_less_than_sqrt_23_l800_800423

theorem int_less_than_sqrt_23 : ∃ (n : ℤ), n < Real.sqrt 23 := by
  use 4
  have h : (4 : ℝ) < Real.sqrt 23 := by
    rw Real.sqrt_lt'_iff
    exact ⟨dec_trivial, dec_trivial⟩
  exact_mod_cast h

end int_less_than_sqrt_23_l800_800423


namespace track_team_children_l800_800502

/-- There were initially 18 girls and 15 boys on the track team.
    7 more girls joined the team, and 4 boys quit the team.
    The proof shows that the total number of children on the track team after the changes is 36. -/
theorem track_team_children (initial_girls initial_boys girls_joined boys_quit : ℕ)
  (h_initial_girls : initial_girls = 18)
  (h_initial_boys : initial_boys = 15)
  (h_girls_joined : girls_joined = 7)
  (h_boys_quit : boys_quit = 4) :
  initial_girls + girls_joined - boys_quit + initial_boys = 36 :=
by
  -- Placeholder to indicate the proof is omitted
  sorry

end track_team_children_l800_800502


namespace ball_bounces_below_2_feet_l800_800927

theorem ball_bounces_below_2_feet :
  ∃ k : ℕ, 500 * (2 / 3 : ℝ) ^ k < 2 ∧ ∀ n < k, 500 * (2 / 3 : ℝ) ^ n ≥ 2 :=
by
  sorry

end ball_bounces_below_2_feet_l800_800927


namespace express_y_in_terms_of_x_l800_800858

-- Defining the parameters and assumptions
variables (x y : ℝ)
variables (h : x * y = 30)

-- Stating the theorem
theorem express_y_in_terms_of_x (h : x * y = 30) : y = 30 / x :=
sorry

end express_y_in_terms_of_x_l800_800858


namespace eccentricity_range_of_ellipse_l800_800698

theorem eccentricity_range_of_ellipse 
  (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) 
  (P : ℝ × ℝ) (hP_ellipse : P.1^2 / a^2 + P.2^2 / b^2 = 1)
  (h_foci_relation : ∀(θ₁ θ₂ : ℝ), a / (Real.sin θ₁) = c / (Real.sin θ₂)) :
  ∃ (e : ℝ), e = c / a ∧ (Real.sqrt 2 - 1 < e ∧ e < 1) := 
sorry

end eccentricity_range_of_ellipse_l800_800698


namespace smallest_positive_period_of_f_is_pi_area_of_triangle_ABC_l800_800288

noncomputable def m (x : ℝ) : ℝ × ℝ := (2 * sin x, sqrt 3 * cos x ^ 2)
noncomputable def n (x : ℝ) : ℝ × ℝ := (cos x, 2)
noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2 - sqrt 3

theorem smallest_positive_period_of_f_is_pi : 
  ∀ x : ℝ, f(x + π) = f(x) :=
sorry

variables {A B C a b c : ℝ}
variables (area : ℝ)
variables (sin_B_sin_C : ℝ) (bc_product : ℝ)

axiom sin_B_C_cond : sin_B_sin_C = (13 * sqrt 3) / 14
axiom bc_condition : bc_product = 40
axiom length_a : a = 7
axiom angle_A : f((A / 2) - (π / 6)) = sqrt 3
axiom sinA : 2 * sin (A) = sqrt 3

theorem area_of_triangle_ABC :
  area = (1 / 2) * b * c * sin (A) :=
  sorry 

end smallest_positive_period_of_f_is_pi_area_of_triangle_ABC_l800_800288


namespace part1_monotonicity_part2_range_of_a_part3_inequality_l800_800266

-- Part 1
theorem part1_monotonicity (x : ℝ) : 
  let f := λ x : ℝ, x * exp x - exp x in 
  (x > 0 → (∃ ε > 0, ∀ y, 0 < y ∧ y < ε → (f x < f (x + y)))) ∧ 
  (x < 0 → (∃ ε > 0, ∀ y, 0 < y ∧ y < ε → (f x > f (x + y)))) :=
begin
  sorry
end

-- Part 2
theorem part2_range_of_a (a : ℝ) : 
  (∀ x > 0, x * exp (a * x) - exp x < -1) ↔ a ≤ (1/2 : ℝ) :=
begin
  sorry
end

-- Part 3
theorem part3_inequality (n : ℕ) (h : n > 0) : 
  (∑ k in finset.range (n + 1), 1 / real.sqrt (k^2 + k)) > real.log (n + 1) :=
begin
  sorry
end

end part1_monotonicity_part2_range_of_a_part3_inequality_l800_800266


namespace total_distance_traveled_l800_800411

def trip_duration : ℕ := 8
def speed_first_half : ℕ := 70
def speed_second_half : ℕ := 85
def time_each_half : ℕ := trip_duration / 2

theorem total_distance_traveled :
  let distance_first_half := time_each_half * speed_first_half
  let distance_second_half := time_each_half * speed_second_half
  let total_distance := distance_first_half + distance_second_half
  total_distance = 620 := by
  sorry

end total_distance_traveled_l800_800411


namespace tangent_line_condition_l800_800690

theorem tangent_line_condition :
  ∃ k : ℝ, (k = -4 / 3 ∨ k = 0) ∧ 
    (∀ x y : ℝ, (-4 / 3) * x - y + 3 = 0 ∨ x = -3) ∧ 
    (∀ x y : ℝ, (x + 2)^2 + y^2 = 1) ∧
    (|(-2) * k + 3k + 3|) / real.sqrt(k^2 + 1) = 1 :=
  sorry

end tangent_line_condition_l800_800690


namespace ferry_distance_l800_800593

theorem ferry_distance 
  (x : ℝ)
  (v_w : ℝ := 3)  -- speed of water flow in km/h
  (t_downstream : ℝ := 5)  -- time taken to travel downstream in hours
  (t_upstream : ℝ := 7)  -- time taken to travel upstream in hours
  (eqn : x / t_downstream - v_w = x / t_upstream + v_w) :
  x = 105 :=
sorry

end ferry_distance_l800_800593


namespace perfect_cubes_between_200_and_2000_l800_800741

noncomputable def count_perfect_cubes (a b : ℕ) : ℕ :=
  let lower_bound := (Nat.floor (Real.cbrt (a + 1))) + 1
  let upper_bound := Nat.floor (Real.cbrt b)
  upper_bound - lower_bound + 1

theorem perfect_cubes_between_200_and_2000 : count_perfect_cubes 200 2000 = 7 :=
by
  sorry

end perfect_cubes_between_200_and_2000_l800_800741


namespace probability_different_grades_l800_800581

theorem probability_different_grades (A B : Type) [Fintype A] [Fintype B] (ha : Fintype.card A = 2) (hb : Fintype.card B = 2) :
  (∃ (s : Finset (A ⊕ B)), s.card = 2) →
  (Fintype.card (Finset (A ⊕ B)).filter (λ s, (∃ (a : A) (b : B), s = {sum.inl a, sum.inr b})) = 4) →
  (Fintype.card (Finset (A ⊕ B)).card-choose 2 = 6) →
  (Fintype.card (Finset (A ⊕ B)).filter (λ s, (∃ (a : A) (b : B), s = {sum.inl a, sum.inr b})) /
     Fintype.card (Finset (A ⊕ B)).card-choose 2 = 2 / 3) :=
sorry

end probability_different_grades_l800_800581


namespace Roger_years_to_retire_l800_800217

noncomputable def Peter : ℕ := 12
noncomputable def Robert : ℕ := Peter - 4
noncomputable def Mike : ℕ := Robert - 2
noncomputable def Tom : ℕ := 2 * Robert
noncomputable def Roger : ℕ := Peter + Tom + Robert + Mike

theorem Roger_years_to_retire :
  Roger = 42 → 50 - Roger = 8 := by
sorry

end Roger_years_to_retire_l800_800217


namespace probability_is_one_third_l800_800770

-- Define the set S
def S : Set ℕ := {4, 5, 6, 9}

-- Define a function to check if two numbers' product is a multiple of 10
def is_multiple_of_10 (a b : ℕ) : Prop :=
  (a * b) % 10 = 0

-- Define the total number of combinations of choosing 2 numbers from S
def total_combinations : ℕ := (Finset.card (Finset.powersetLen 2 S.toFinset))

-- Define the set of valid pairs whose product is a multiple of 10
def valid_pairs : Finset (ℕ × ℕ) :=
  Finset.filter (λ pair, is_multiple_of_10 pair.1 pair.2)
  (Finset.powersetLen 2 S.toFinset).val.map (λ s, (s.val.head, s.val.tail.head))

-- Calculate the probability 
def probability : ℚ :=
  (valid_pairs.card : ℚ) / (total_combinations : ℚ)

-- The theorem to prove
theorem probability_is_one_third : probability = (1 : ℚ) / (3 : ℚ) := 
by sorry

end probability_is_one_third_l800_800770


namespace cube_has_two_regular_tetrahedrons_l800_800534

def cube_vertices : List (ℝ × ℝ × ℝ) := [
  (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
  (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)
]

def is_regular_tetrahedron (vertices: List (ℝ × ℝ × ℝ)) : Prop :=
  ∃ (a b c d: (ℝ × ℝ × ℝ)),
    List.perm vertices [a, b, c, d] ∧
    dist a b = dist a c ∧ dist a c = dist a d ∧
    dist b c = dist b d ∧ dist c d = dist a b

theorem cube_has_two_regular_tetrahedrons :
  ∃! (tetrahedrons: List (List (ℝ × ℝ × ℝ))),
    List.length tetrahedrons = 2 ∧
    ∀ t ∈ tetrahedrons, 
      List.subset t cube_vertices ∧ is_regular_tetrahedron t :=
sorry

end cube_has_two_regular_tetrahedrons_l800_800534


namespace cannot_determine_candies_l800_800833

theorem cannot_determine_candies (eggs total_eggs friends: ℕ) (candies: Type) :
  total_eggs = 16 ∧ friends = 8 ∧ (∀ friend, friend < friends → eggs / friends = 2) → 
  ∀ candies, ¬ false :=
by
  sorry

end cannot_determine_candies_l800_800833


namespace count_congruent_to_3_mod_8_in_300_l800_800738

theorem count_congruent_to_3_mod_8_in_300 : 
  {n : ℤ | 1 ≤ n ∧ n ≤ 300 ∧ n % 8 = 3}.card = 38 := 
by
  sorry

end count_congruent_to_3_mod_8_in_300_l800_800738


namespace triangle_inequality_inradius_l800_800431

theorem triangle_inequality_inradius 
  (A B C : Type) [Triangle A B C] (I : Incenter A B C) (r : Inradius A B C) :
  (∃ (AI BI CI : ℝ), AI ≥ (b + c) / a * r ∧ BI ≥ (a + c) / b * r ∧ CI ≥ (a + b) / c * r 
  → AI + BI + CI ≥ 6 * r) := 
sorry

end triangle_inequality_inradius_l800_800431


namespace arithmetic_sequence_identity_l800_800756

def arithmetic_sequence {α : Type*} [AddCommGroup α] (a : ℕ → α) : Prop :=
∀ m n : ℕ, a (m + n) = a m + a n

theorem arithmetic_sequence_identity {α : Type*} [AddCommGroup α] (a : ℕ → α)
  (h : arithmetic_sequence a) (m n p : ℕ) (hmnp : m ≠ n ∧ n ≠ p ∧ p ≠ m) :
  m * (a p - a n) + n * (a m - a p) + p * (a n - a m) = 0 :=
sorry

end arithmetic_sequence_identity_l800_800756


namespace horner_method_evaluation_l800_800899

def f (x : ℝ) := 0.5 * x^5 + 4 * x^4 + 0 * x^3 - 3 * x^2 + x - 1

theorem horner_method_evaluation : f 3 = 1 :=
by
  -- Placeholder for the proof
  sorry

end horner_method_evaluation_l800_800899


namespace weight_division_possible_l800_800031

theorem weight_division_possible :
  ∃ (A B C : Finset ℕ), A \cup B \cup C = {3, 4, 5, 6, 7, 8, 9, 10, 11} ∧
                         A ∩ B = ∅ ∧ B ∩ C = ∅ ∧ A ∩ C = ∅ ∧
                         A.sum id = 21 ∧ B.sum id = 21 ∧ C.sum id = 21 :=
by
  sorry

end weight_division_possible_l800_800031


namespace art_department_probability_l800_800573

theorem art_department_probability : 
  let students := {s1, s2, s3, s4} 
  let first_grade := {s1, s2}
  let second_grade := {s3, s4}
  let total_pairs := { (x, y) | x ∈ students ∧ y ∈ students ∧ x < y }.to_finset.card
  let diff_grade_pairs := { (x, y) | x ∈ first_grade ∧ y ∈ second_grade ∨ x ∈ second_grade ∧ y ∈ first_grade}.to_finset.card
  (diff_grade_pairs / total_pairs) = 2 / 3 := 
by 
  sorry

end art_department_probability_l800_800573


namespace section_is_parabola_l800_800446

-- Defining the conditions as Lean definitions.
def apex_angle (cone : Type) : ℝ := 90
def section_angle (cone : Type) : ℝ := 45

-- The formal statement of the problem in Lean 4.
theorem section_is_parabola (cone : Type) 
  (h_apex_angle: apex_angle cone = 90) 
  (h_section_angle: section_angle cone = 45) 
  : section_is_parabola cone :=
sorry

end section_is_parabola_l800_800446


namespace find_d_l800_800153

-- Defining the coordinates of the foci
def F1 := (4, 8)
def F2 := (d : ℝ × ℝ)
def P := (d', 0) -- P is the point where ellipse is tangent to the x-axis

-- Hypothesis based on conditions provided
def hypothesis (d : ℝ) :=
  F2 = (d, 8) ∧ (d + 4 = 2 * sqrt (((d - 4) / 2)^2 + 8^2))

-- The goal to prove
theorem find_d : ∃ d : ℝ, hypothesis d ∧ d = 30 :=
by 
  sorry

end find_d_l800_800153


namespace remainder_5_pow_2023_mod_11_l800_800999

theorem remainder_5_pow_2023_mod_11 : (5^2023) % 11 = 4 :=
by
  have h1 : 5^2 % 11 = 25 % 11 := sorry
  have h2 : 25 % 11 = 3 := sorry
  have h3 : (3^5) % 11 = 1 := sorry
  have h4 : 3^1011 % 11 = ((3^5)^202 * 3) % 11 := sorry
  have h5 : ((3^5)^202 * 3) % 11 = (1^202 * 3) % 11 := sorry
  have h6 : (1^202 * 3) % 11 = 3 % 11 := sorry
  have h7 : (5^2023) % 11 = (3 * 5) % 11 := sorry
  have h8 : (3 * 5) % 11 = 15 % 11 := sorry
  have h9 : 15 % 11 = 4 := sorry
  exact h9

end remainder_5_pow_2023_mod_11_l800_800999


namespace find_fraction_value_l800_800755

theorem find_fraction_value {m n r t : ℚ}
  (h1 : m / n = 5 / 2)
  (h2 : r / t = 7 / 5) :
  (2 * m * r - 3 * n * t) / (5 * n * t - 4 * m * r) = -4 / 9 :=
by
  sorry

end find_fraction_value_l800_800755


namespace inverse_square_variation_l800_800916

theorem inverse_square_variation (x y : ℝ) (h1 : y = 4) (h2 : x = 2) (k : ℝ) (h3 : 5 * y = k / x^2) :
  y = 1 := by
  have k_val : k = 80 := by
    calc k = 5 * 4 * (2 ^ 2) : by sorry
           ... = 80 : by norm_num
  have h4 : 5 * y = 80 / (4 ^ 2) := by sorry
  have h5 : 5 * y = 5 := by
    calc
      5 * y = 80 / 16 : by sorry
      ... = 5 : by norm_num
  show y = 1 from by
    calc
      y = (5 * y) / 5 : by sorry
      ... = 5 / 5 : by sorry
      ... = 1 : by norm_num

end inverse_square_variation_l800_800916


namespace width_of_carton_is_45_l800_800120

noncomputable def soap_box_carton_example : Prop :=
  ∃ (W : ℝ), 
    let length_carton := 30
    let height_carton := 60
    let total_boxes := 360
    let length_box := 7
    let height_box := 5
    let width_box := 6
    let num_length := length_carton / length_box
    let num_height := height_carton / height_box
    let num_width := W / width_box
    num_length.floor * num_height.floor * num_width.floor = total_boxes ∧ W = 45

theorem width_of_carton_is_45 : soap_box_carton_example :=
by
  sorry

end width_of_carton_is_45_l800_800120


namespace smallest_four_digit_palindromic_prime_l800_800865

def is_palindrome (n : Nat) : Prop :=
  n.toString = n.toString.reverse

theorem smallest_four_digit_palindromic_prime :
  (∀ n, is_palindrome n ∧ Nat.prime n → n < 1000 ∨ 1441 ≤ n) ∧
  is_palindrome 1441 ∧ Nat.prime 1441 :=
by sorry

end smallest_four_digit_palindromic_prime_l800_800865


namespace magnitude_of_combination_angle_between_combination_and_a_l800_800710
noncomputable theory

variables (a b : vector ℝ) 
variable θ : ℝ
variable magnitude_a magnitude_b : ℝ

-- the conditions
def angle_condition : θ = 5 * real.pi / 6 := sorry
def magnitude_condition_a : magnitude_a = 2 := sorry
def magnitude_condition_b : magnitude_b = real.sqrt 3 := sorry

-- the statements to prove
theorem magnitude_of_combination : 
  |3 * a + 2 * b| = 2 * real.sqrt 3 := sorry

theorem angle_between_combination_and_a :
  let comb := 3 * a + 2 * b in
  ∀ α : ℝ, α ∈ set.Icc 0 real.pi →
  real.cos α = (comb • a) / (|comb| * |a|) → α = real.pi / 6 := sorry

end magnitude_of_combination_angle_between_combination_and_a_l800_800710


namespace clock_angle_at_7_oclock_is_150_degrees_l800_800167

theorem clock_angle_at_7_oclock_is_150_degrees :
  ∀ (full_circle degrees_per_hour hours smalle_angle),
    full_circle = 360 → 
    hours = 12 → 
    degrees_per_hour = full_circle / hours →
    smalle_angle = 5 * degrees_per_hour →
    (smalle_angle = 150) := 
by
  intros full_circle degrees_per_horr hours smaller_angle
  intros h_full_circle h_hour_count h_angle_per_hour h_smaller_angle
  rw [h_full_circle] at h_angle_per_hour
  rw [h_hour_count] at h_angle_per_hour
  simp at h_angle_per_hour
  rw [h_smaller_angle]
  simp
  sorry

end clock_angle_at_7_oclock_is_150_degrees_l800_800167


namespace ab_calculation_l800_800303

noncomputable def triangle_area (a b : ℝ) : ℝ :=
  (1 / 2) * (4 / a) * (4 / b)

theorem ab_calculation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : triangle_area a b = 4) : a * b = 2 :=
by
  sorry

end ab_calculation_l800_800303


namespace joe_money_left_l800_800310

theorem joe_money_left (original_money : ℕ) (spent_chocolates_frac spent_fruits_frac : ℚ) 
  (h1 : original_money = 450) (h2 : spent_chocolates_frac = 1/9) (h3 : spent_fruits_frac = 2/5) : 
  original_money - (original_money * spent_chocolates_frac).to_nat - (original_money * spent_fruits_frac).to_nat = 220 := by
  sorry

end joe_money_left_l800_800310


namespace bus_stop_time_l800_800095

theorem bus_stop_time (speed_excluding_stops : ℝ) (speed_including_stops : ℝ) : 
  speed_excluding_stops = 50 ∧ speed_including_stops = 45 → 
  let stoppage_time := (speed_excluding_stops - speed_including_stops) / speed_excluding_stops * 60 in
  stoppage_time = 6 :=
by
  sorry

end bus_stop_time_l800_800095


namespace largest_valid_number_l800_800081

-- Define the conditions for the digits of the number
def valid_digits (a b c d : ℕ) : Prop :=
  c = a + b ∧ d = b + c

-- Prove that the number formed by digits 9, 0, 9, 9 is the largest valid 4-digit number
theorem largest_valid_number : ∃ a b c d, valid_digits a b c d ∧
  (a * 1000 + b * 100 + c * 10 + d = 9099) :=
begin
  use [9, 0, 9, 9],
  split,
  { -- Proof of valid digits condition
    split; refl },
  { -- Proof that the number is 9099
    refl }
end

end largest_valid_number_l800_800081


namespace pow_modulus_l800_800993

theorem pow_modulus : (5 ^ 2023) % 11 = 3 := by
  sorry

end pow_modulus_l800_800993


namespace number_of_outcomes_l800_800105

-- Define the conditions
def students : Nat := 4
def events : Nat := 3

-- Define the problem: number of possible outcomes for the champions
theorem number_of_outcomes : students ^ events = 64 :=
by sorry

end number_of_outcomes_l800_800105


namespace clock_overlap_24_hours_l800_800745

theorem clock_overlap_24_hours (hour_rotations : ℕ) (minute_rotations : ℕ) 
  (h_hour_rotations: hour_rotations = 2) 
  (h_minute_rotations: minute_rotations = 24) : 
  ∃ (overlaps : ℕ), overlaps = 22 := 
by 
  sorry

end clock_overlap_24_hours_l800_800745


namespace cost_relationship_l800_800159

variable {α : Type} [LinearOrderedField α]
variables (bananas_cost apples_cost pears_cost : α)

theorem cost_relationship :
  (5 * bananas_cost = 3 * apples_cost) →
  (10 * apples_cost = 6 * pears_cost) →
  (25 * bananas_cost = 9 * pears_cost) := by
  intros h1 h2
  sorry

end cost_relationship_l800_800159


namespace find_c_l800_800012

-- Defining vectors u and v
def u (c : ℝ) : ℝ × ℝ := (-4, c)
def v : ℝ × ℝ := (1, -2)

-- The projection vector formula on v
def proj_u_v (c : ℝ) : ℝ × ℝ :=
  let v_dot := (1 * 1 + (-2) * (-2)) in
  let u_dot_v := (-4) * 1 + c * (-2) in
  (u_dot_v / v_dot) • v

-- The main theorem we need to prove
theorem find_c (c : ℝ) : proj_u_v c = (7 / 5) • v → c = -11 / 2 := by
  sorry

end find_c_l800_800012


namespace find_a_b_min_f_at_1_l800_800723

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x^2 - b * real.log x

theorem find_a_b (a b : ℝ) (h : f 1 a b = 1) (h' : 2 * a - b = 0) : a = 1 ∧ b = 2 := sorry

theorem min_f_at_1 : ∀ x > 0, (f x 1 2) ≥ 1 := sorry

end find_a_b_min_f_at_1_l800_800723


namespace relationship_p_q_l800_800763

theorem relationship_p_q (p q : ℝ) : 
  (expand_poly : (x : ℝ) → ((x^2 - p * x + q) * (x - 3)) = (x * x * x + (-p - 3) * x * x + (3 * p + q) * x - 3 * q)) → 
  (linear_term_condition : ∀ x, expand_poly x → (3 * p + q = 0)) → 
  q + 3 * p = 0 :=
begin
  sorry
end

end relationship_p_q_l800_800763


namespace total_pencils_l800_800979

theorem total_pencils (pencils_per_child : ℕ) (children : ℕ) (hp : pencils_per_child = 2) (hc : children = 8) :
  pencils_per_child * children = 16 :=
by
  sorry

end total_pencils_l800_800979


namespace find_constants_a_b_l800_800204

variables (x a b : ℝ)

theorem find_constants_a_b (h : (x - a) / (x + b) = (x^2 - 45 * x + 504) / (x^2 + 66 * x - 1080)) :
  a + b = 48 :=
sorry

end find_constants_a_b_l800_800204


namespace report_word_count_l800_800139

theorem report_word_count (typed_in_30 : ℕ) (already_written : ℕ) (remaining_minutes : ℕ) : 
  (typed_in_30 = 300) → (already_written = 200) → (remaining_minutes = 80) → 
  (already_written + (typed_in_30 * remaining_minutes / 30) = 1000) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end report_word_count_l800_800139


namespace largest_prime_factor_sum_divisors_300_l800_800813

theorem largest_prime_factor_sum_divisors_300 :
  ∃ p : ℕ, prime p ∧ p = 31 ∧ p = Nat.max (Nat.factors (Nat.divisors_sum 300)) :=
sorry

end largest_prime_factor_sum_divisors_300_l800_800813


namespace largest_four_digit_number_prop_l800_800049

theorem largest_four_digit_number_prop :
  ∃ (a b c d : ℕ), a = 9 ∧ b = 0 ∧ c = 9 ∧ d = 9 ∧ (1000 * a + 100 * b + 10 * c + d = 9099) ∧ (c = a + b) ∧ (d = b + c) :=
by
  sorry

end largest_four_digit_number_prop_l800_800049


namespace quadratic_discriminant_l800_800205

def discriminant (a b c : ℚ) : ℚ :=
  b^2 - 4 * a * c

theorem quadratic_discriminant : discriminant 5 (5 + 1/2) (-2) = 281/4 := by
  sorry

end quadratic_discriminant_l800_800205


namespace problem1_problem2_l800_800702

open Real

noncomputable def alpha (hα : 0 < α ∧ α < π / 3) :=
  α

noncomputable def vec_a (hα : 0 < α ∧ α < π / 3) :=
  (sqrt 6 * sin (alpha hα), sqrt 2)

noncomputable def vec_b (hα : 0 < α ∧ α < π / 3) :=
  (1, cos (alpha hα) - sqrt 6 / 2)

theorem problem1 (hα : 0 < α ∧ α < π / 3) (h_orth : (sqrt 6 * sin (alpha hα)) + sqrt 2 * (cos (alpha hα) - sqrt 6 / 2) = 0) :
  tan (alpha hα + π / 6) = sqrt 15 / 5 :=
sorry

theorem problem2 (hα : 0 < α ∧ α < π / 3) (h_orth : (sqrt 6 * sin (alpha hα)) + sqrt 2 * (cos (alpha hα) - sqrt 6 / 2) = 0) :
  cos (2 * alpha hα + 7 * π / 12) = (sqrt 2 - sqrt 30) / 8 :=
sorry

end problem1_problem2_l800_800702


namespace regular_octagon_diagonal_length_l800_800610

theorem regular_octagon_diagonal_length {r : ℝ} (h₁ : r = 12) :
  ∃ AC, AC = sqrt (288 + 144 * sqrt 2) :=
by
  use sqrt (288 + 144 * sqrt 2)
  linarith

end regular_octagon_diagonal_length_l800_800610


namespace lines_concurrent_l800_800392

variables {A B C D X Y O M N : Type}

-- Define the distinct points on a line
axiom distinct_points_on_line : A ≠ B ∧ B ≠ C ∧ C ≠ D

-- Define the circles with diameters AC and BD intersecting at X and Y
axiom circles_with_diameters_intersect : 
  ∃ (circles : set (set (ℝ × ℝ))),
  ((∀ (P Q ∈ circles), P ≠ Q) ∧ (X ∈ circles) ∧ (Y ∈ circles))

-- Define an arbitrary point O on line XY but not on AD
axiom O_on_XY : O ≠ X ∧ O ≠ Y
axiom O_not_on_AD : O ≠ A ∧ O ≠ D

-- Define CO intersects the circle with diameter AC again at M
axiom CO_intersects_AC : M ∈ set_of (λ p, p ≠ C ∧ p ∈ (λ pq : ℝ × ℝ, pq.1 ^ 2 + pq.2 ^ 2 = (dist C p) ^ 2))

-- Define BO intersects the circle with diameter BD again at N
axiom BO_intersects_BD : N ∈ set_of (λ p, p ≠ B ∧ p ∈ (λ pq : ℝ × ℝ, pq.1 ^ 2 + pq.2 ^ 2 = (dist B p) ^ 2))

-- Prove lines AM, DN, and XY are concurrent
theorem lines_concurrent : 
  ∃ (P : Type), 
  ∃ (point : P), 
  ∃ (am : set (ℝ × ℝ)) (dn : set (ℝ × ℝ)) (xy : set (ℝ × ℝ)), 
  point ∈ am ∧ point ∈ dn ∧ point ∈ xy := 
sorry

end lines_concurrent_l800_800392


namespace chukchi_hut_location_l800_800984

theorem chukchi_hut_location :
  (∃ latitude, ∀ points,
    (points.1 = 0 ∧ points.2 = latitude ∧ points.3 = 0) →
    ((λ (latitude : ℝ), latitude - 10 = latitude) ∨
     (λ (longitude : ℝ), (longitude + 10) % (2 * π) = longitude))) →
  (latitude = π / 2) ∨ (latitude > -π / 2 ∧ ∃ n : ℕ, 10 / (2 * π * (cos latitude)) = n) :=
sorry

end chukchi_hut_location_l800_800984


namespace triangle_area_l800_800897

-- Definitions
variable (Point : Type) [EuclideanGeometry Point]
variable (A B F G H : Point)

noncomputable def sideAB : Real := 8
noncomputable def sideBF : Real := 8
noncomputable def sideBG : Real := 12

-- Conditions
axiom ABF_is_right_triangle : angle A B F = 90
axiom ABG_is_right_triangle : angle A B G = 90
axiom AB_eq_BF : dist A B = sideAB ∧ dist B F = sideBF
axiom AB_eq_BG : dist A B = sideBG ∧ dist B G = sideBG
axiom H_is_midpoint : dist B H = dist H F

-- Proof statement
theorem triangle_area : area (triangle A B H) = 16 := by
  sorry

end triangle_area_l800_800897


namespace exists_a_in_set_implies_a_eq_neg_one_l800_800226

theorem exists_a_in_set_implies_a_eq_neg_one (a : ℝ) (h : 1 ∈ ({a, a+1, a^2} : set ℝ)) : a = -1 :=
sorry

end exists_a_in_set_implies_a_eq_neg_one_l800_800226


namespace lemons_for_10_gallons_l800_800101

noncomputable def lemon_proportion : Prop :=
  ∃ x : ℝ, (36 / 48) = (x / 10) ∧ x = 7.5

theorem lemons_for_10_gallons : lemon_proportion :=
by
  sorry

end lemons_for_10_gallons_l800_800101


namespace pentagon_area_l800_800019

theorem pentagon_area (AB DE : ℝ) (H1 : AB = 1) (H2 : DE = 1) : 
  let pentagon_area := 1
  in pentagon_area = 1 :=
by
  sorry

end pentagon_area_l800_800019


namespace num_of_possible_values_of_a_l800_800429

theorem num_of_possible_values_of_a :
  ∃ a b c d : ℕ, a > b ∧ b > c ∧ c > d ∧ a + b + c + d = 2060 ∧ a^2 - b^2 + c^2 - d^2 = 1987 ∧ (count_a_values a b c d = 513) :=
by
  -- Proof details skipped
  sorry

end num_of_possible_values_of_a_l800_800429


namespace rhombus_perimeter_l800_800453

theorem rhombus_perimeter
  (d1 d2 : ℝ)
  (h1 : d1 = 20)
  (h2 : d2 = 16) :
  4 * (Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)) = 8 * Real.sqrt 41 := 
  sorry

end rhombus_perimeter_l800_800453


namespace quadratic_equal_roots_iff_l800_800978

theorem quadratic_equal_roots_iff (k : ℝ) : 
  (∀ x : ℝ, 3 * x^2 - k * x + 2 * x + 24 = 0 → x = 2 + 12 * real.sqrt 2 ∨ x = 2 - 12 * real.sqrt 2) ↔ 
  (k = 2 + 12 * real.sqrt 2 ∨ k = 2 - 12 * real.sqrt 2) := 
by sorry

end quadratic_equal_roots_iff_l800_800978


namespace binom_sum_mod_2027_l800_800867

theorem binom_sum_mod_2027 :
  let T := ∑ k in Finset.range 65, Nat.choose 2024 k
  T % 2027 = 1089 :=
by
  let T := ∑ k in Finset.range 65, Nat.choose 2024 k
  have h2027_prime : Nat.prime 2027 := by exact dec_trivial
  sorry -- This is the placeholder for the actual proof

end binom_sum_mod_2027_l800_800867


namespace remainders_and_minimal_x_l800_800908

theorem remainders_and_minimal_x (x : ℕ) (k : ℕ) :
  x = 285 * k + 31 →
  x % 17 = 14 ∧ x % 23 = 8 ∧ x % 19 = 12 ∧ x = 31 :=
by {
  intro hx,
  sorry
}

end remainders_and_minimal_x_l800_800908


namespace min_value_of_g_l800_800692

noncomputable def f (x : ℝ) : ℝ := -x^2 + 2*x + 15

noncomputable def g (x m : ℝ) : ℝ := (2 - 2 * m) * x - f(x)

theorem min_value_of_g (m : ℝ) :
  ∀ x ∈ Icc 0 2, g x m ≥ (if m ≤ 0 then -15 else if m < 2 then -m^2 - 15 else -4 * m - 11) :=
by sorry

end min_value_of_g_l800_800692


namespace integral_f_l800_800209

noncomputable def f (x : ℝ) : ℝ := (2 * x^3 + 2 * x^2 + 2 * x + 1) / ((x^2 + x + 1) * (x^2 + 1))

theorem integral_f :
  ∫ f(x) dx = (1/2) * Real.log (abs (x^2 + x + 1)) + (1/√3) * arctan ((2 * x + 1) / √3) +
              (1/2) * Real.log (abs (x^2 + 1)) + C :=
by
  sorry

end integral_f_l800_800209


namespace time_to_cover_escalator_l800_800625

def escalator_speed : ℝ := 12
def escalator_length : ℝ := 160
def person_speed : ℝ := 8

theorem time_to_cover_escalator :
  (escalator_length / (escalator_speed + person_speed)) = 8 := by
  sorry

end time_to_cover_escalator_l800_800625


namespace bryden_collection_value_l800_800117

-- Define the conditions
def face_value_half_dollar : ℝ := 0.5
def face_value_quarter : ℝ := 0.25
def num_half_dollars : ℕ := 5
def num_quarters : ℕ := 3
def multiplier : ℝ := 30

-- Define the problem statement as a theorem
theorem bryden_collection_value : 
  (multiplier * (num_half_dollars * face_value_half_dollar + num_quarters * face_value_quarter)) = 97.5 :=
by
  -- Proof is skipped since it's not required
  sorry

end bryden_collection_value_l800_800117


namespace range_of_t_l800_800257

variable {x y θ : ℝ}
variable (P : ℝ × ℝ)
variable (t : ℝ)
variable [noncomputable] (C : ℝ)

def circle := ∀ x y : ℝ, (x + 2)^2 + y^2 = 1
def tangent_length := ∀ x₀ y₀ : ℝ, (x₀ + 2)^2 + y₀^2 - 1
def distance_to_y_axis := ∀ x₀ : ℝ, |x₀|

def locus_condition (x₀ y₀ : ℝ) : Prop :=
  tangent_length x₀ y₀ = t^2 * (distance_to_y_axis x₀)

def locus_equation : Prop :=
  (1 - t^2) * x^2 + y^2 + 4 * x + 3 = 0

def ellipse_condition (t : ℝ) :=
  1 - t^2 > 0

def angle_condition (θ : ℝ) :=
  0 < θ < π

theorem range_of_t (θ : ℝ) (hθ : angle_condition θ)
  (ht : ellipse_condition t) : 0 < t ∧ t < Real.sin (θ / 2) := sorry

end range_of_t_l800_800257


namespace probability_is_two_thirds_l800_800578

-- Define the general framework and conditions
def total_students : ℕ := 4
def students_from_first_grade : ℕ := 2
def students_from_second_grade : ℕ := 2

-- Define the combinations for selecting 2 students out of 4
def total_ways_to_select_2_students : ℕ := Nat.choose total_students 2

-- Define the combinations for selecting 1 student from each grade
def ways_to_select_1_from_first : ℕ := Nat.choose students_from_first_grade 1
def ways_to_select_1_from_second : ℕ := Nat.choose students_from_second_grade 1
def favorable_ways : ℕ := ways_to_select_1_from_first * ways_to_select_1_from_second

-- The target probability calculation
noncomputable def probability_of_different_grades : ℚ :=
  favorable_ways / total_ways_to_select_2_students

-- The statement and proof requirement (proof is deferred with sorry)
theorem probability_is_two_thirds :
  probability_of_different_grades = 2 / 3 :=
by sorry

end probability_is_two_thirds_l800_800578


namespace largest_valid_number_l800_800082

-- Define the conditions for the digits of the number
def valid_digits (a b c d : ℕ) : Prop :=
  c = a + b ∧ d = b + c

-- Prove that the number formed by digits 9, 0, 9, 9 is the largest valid 4-digit number
theorem largest_valid_number : ∃ a b c d, valid_digits a b c d ∧
  (a * 1000 + b * 100 + c * 10 + d = 9099) :=
begin
  use [9, 0, 9, 9],
  split,
  { -- Proof of valid digits condition
    split; refl },
  { -- Proof that the number is 9099
    refl }
end

end largest_valid_number_l800_800082


namespace E_X_plus_Var_X_eq_l800_800329

noncomputable def probability_red : ℚ := 5 / 15

noncomputable def expectation_X : ℚ := 3 * probability_red

noncomputable def variance_X : ℚ := 3 * probability_red * (1 - probability_red)

noncomputable def E_X_plus_Var_X : ℚ := expectation_X + variance_X

theorem E_X_plus_Var_X_eq :
  E_X_plus_Var_X = 5 / 3 :=
by
  rw [E_X_plus_Var_X, expectation_X, variance_X, probability_red]
  simp
  sorry

end E_X_plus_Var_X_eq_l800_800329


namespace max_sqrt_sum_l800_800816

noncomputable def max_value (a b c : ℝ) : ℝ :=
  if h : a + b + c = 8 then (sqrt (3 * a + 2) + sqrt (3 * b + 2) + sqrt (3 * c + 2)) else 0

theorem max_sqrt_sum (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a + b + c = 8) : 
  max_value a b c = 3 * sqrt 10 :=
by
  sorry

end max_sqrt_sum_l800_800816


namespace barbi_weight_loss_duration_l800_800165

theorem barbi_weight_loss_duration :
  (∃ x : ℝ, 
    (∃ l_barbi l_luca : ℝ, 
      l_barbi = 1.5 * x ∧ 
      l_luca = 99 ∧ 
      l_luca = l_barbi + 81) ∧
    x = 12) :=
by
  sorry

end barbi_weight_loss_duration_l800_800165


namespace rhombus_perimeter_l800_800452

theorem rhombus_perimeter
  (d1 d2 : ℝ)
  (h1 : d1 = 20)
  (h2 : d2 = 16) :
  4 * (Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)) = 8 * Real.sqrt 41 := 
  sorry

end rhombus_perimeter_l800_800452


namespace problem_is_happy_number_512_l800_800314

/-- A number is a "happy number" if it is the square difference of two consecutive odd numbers. -/
def is_happy_number (x : ℕ) : Prop :=
  ∃ n : ℤ, x = 8 * n

/-- The number 512 is a "happy number". -/
theorem problem_is_happy_number_512 : is_happy_number 512 :=
  sorry

end problem_is_happy_number_512_l800_800314


namespace largest_four_digit_number_l800_800057

theorem largest_four_digit_number :
  ∃ a b c d : ℕ, 
    9 < 1000 * a + 100 * b + 10 * c + d ∧ 
    1000 * a + 100 * b + 10 * c + d < 10000 ∧ 
    c = a + b ∧ 
    d = b + c ∧ 
    1000 * a + 100 * b + 10 * c + d = 9099 :=
by {
  sorry
}

end largest_four_digit_number_l800_800057


namespace parallel_line_dividing_triangle_l800_800137

theorem parallel_line_dividing_triangle (base : ℝ) (length_parallel_line : ℝ) 
    (h_base : base = 24) 
    (h_parallel : (length_parallel_line / base)^2 = 1/2) : 
    length_parallel_line = 12 * Real.sqrt 2 :=
sorry

end parallel_line_dividing_triangle_l800_800137


namespace average_mpg_trip_l800_800637

theorem average_mpg_trip :
  let initial_odometer := 56300
  let initial_gasoline := 2
  let additional_gasoline_pre_trip := 8
  let odometer_during_trip := 56675
  let gasoline_during_trip := 18
  let final_odometer := 57200
  let gasoline_end_trip := 25
  let total_distance := final_odometer - initial_odometer
  let total_gasoline_used := additional_gasoline_pre_trip + gasoline_during_trip + gasoline_end_trip
  let average_mpg := total_distance.toFloat / total_gasoline_used.toFloat in
  Float.roundTo average_mpg 1 = 17.6 := by
  intros
  let initial_odometer := 56300
  let initial_gasoline := 2
  let additional_gasoline_pre_trip := 8
  let odometer_during_trip := 56675
  let gasoline_during_trip := 18
  let final_odometer := 57200
  let gasoline_end_trip := 25
  let total_distance := final_odometer - initial_odometer
  let total_gasoline_used := additional_gasoline_pre_trip + gasoline_during_trip + gasoline_end_trip
  let average_mpg := total_distance.toFloat / total_gasoline_used.toFloat
  have distance_eq : total_distance = 900 := rfl
  have gasoline_eq : total_gasoline_used = 51 := rfl
  have mpg_eq : average_mpg = 900.toFloat / 51.toFloat := by simp [average_mpg, total_distance, total_gasoline_used]
  have mpg_value : average_mpg = 17.647058823529413 := by simp [mpg_eq]; norm_num1
  have round_eq : Float.roundTo 17.647058823529413 1 = 17.6 := rfl
  show Float.roundTo average_mpg 1 = 17.6
  by rw [mpg_value, round_eq]

end average_mpg_trip_l800_800637


namespace inscribed_square_area_l800_800553

theorem inscribed_square_area (O : Point) (A B : Point) (r : ℝ) (AB_len : ℝ) (square_area : ℝ) :
  dist O A = r →
  dist O B = r →
  dist A B = AB_len →
  r = 5 →
  AB_len = 6 →
  square_area = 36 :=
by
  sorry

end inscribed_square_area_l800_800553


namespace smallest_positive_Sn_l800_800242

theorem smallest_positive_Sn
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (d : ℝ)
  (h1 : ∀ n : ℕ, S (n + 1) = S n + a (n + 1))
  (h2 : S 20 < 0)
  (h3 : S 19 > 0)
  (h4 : ∀ i j : ℕ, i ≤ j → a i > a j)
  (h5 : a 11 / a 10 < -1)
  (h6 : S (19 + 1) < S 19): ∃ n, n = 19 ∧ S n = 19 * a 10 ∧ S 20 = 10 * (a 10 + a 11)::= 0 := 
sorry

end smallest_positive_Sn_l800_800242


namespace two_x_plus_y_eq_12_l800_800917

-- Variables representing the prime numbers x and y
variables {x y : ℕ}

-- Definitions and conditions
def is_prime (n : ℕ) : Prop := Prime n
def lcm_eq (a b c : ℕ) : Prop := Nat.lcm a b = c

-- The theorem statement
theorem two_x_plus_y_eq_12 (h1 : lcm_eq x y 10) (h2 : is_prime x) (h3 : is_prime y) (h4 : x > y) :
    2 * x + y = 12 :=
sorry

end two_x_plus_y_eq_12_l800_800917


namespace total_cards_l800_800889

theorem total_cards (initial_cards additional_cards : ℕ) (h₁ : initial_cards = 9) (h₂ : additional_cards = 4) :
    initial_cards + additional_cards = 13 :=
by
  rw [h₁, h₂]
  exact Nat.add_comm 9 4

end total_cards_l800_800889


namespace scores_greater_than_18_l800_800343

theorem scores_greater_than_18 (scores : Fin 20 → ℝ) 
  (h_unique : Function.Injective scores)
  (h_sum : ∀ i j k : Fin 20, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k) :
  ∀ i : Fin 20, scores i > 18 := 
by
  sorry

end scores_greater_than_18_l800_800343


namespace largest_sum_of_products_of_groups_of_ten_l800_800419

theorem largest_sum_of_products_of_groups_of_ten :
  ∃ a : Fin 2009 → ℤ, (∀ i, a i = 1 ∨ a i = -1) ∧ (∃ i j : Fin 2009, i ≠ j ∧ a i ≠ a j) ∧
    (∑ i : Fin 2009, ∏ j in Finset.range 10, a ((i + j) % 2009) = 2005) :=
sorry

end largest_sum_of_products_of_groups_of_ten_l800_800419


namespace find_k_l800_800396

open Function

variable {α : Type*} [Field α]

def f (a b : α) : α → α := λ x, a * x + b

def f_iter (a b : α) : ℕ → (α → α)
| 0       := id
| (n + 1) := f a b ∘ f_iter n

theorem find_k (a b : α) (k : ℕ) (h₁ : 2 * a + b = -2) 
(h₂ : f_iter a b k = λ x, -243 * x + 244) : k = 5 :=
sorry

end find_k_l800_800396


namespace binomial_sum_mod_prime_l800_800874

theorem binomial_sum_mod_prime (T : ℕ) (hT : T = ∑ k in Finset.range 65, Nat.choose 2024 k) : 
  T % 2027 = 1089 :=
by
  have h_prime : Nat.prime 2027 := by sorry -- Given that 2027 is prime
  have h := (2024 : ℤ) % 2027
  sorry -- The proof of the actual sum equivalences

end binomial_sum_mod_prime_l800_800874


namespace curve_c_eccentricity_l800_800959

theorem curve_c_eccentricity :
  let e := (Real.sqrt 6) / 2
  let curveC : ∀ x y : ℝ, (x^2) / 4 - (y^2) / 2 = 1
  e = ∃ (a b : ℝ), (a^2 = 4) ∧ (b^2 = 2) ∧ (a^2 - b^2 = (c * e)^2) :=
by
  sorry

end curve_c_eccentricity_l800_800959


namespace value_of_sum_exponent_l800_800709

-- Define the given conditions
variables {a b : ℝ}
axiom symmetry_about_y_axis : (a, 3) = (4, b) ∨ (a, 3) = (-4, b)

-- The conjecture to be proved
theorem value_of_sum_exponent (h_symmetry : symmetry_about_y_axis) : (a + b) ^ 2008 = 1 := 
sorry

end value_of_sum_exponent_l800_800709


namespace olympiad_scores_l800_800357

theorem olympiad_scores (scores : Fin 20 → ℕ) 
  (uniqueScores : ∀ i j, i ≠ j → scores i ≠ scores j)
  (less_than_sum_of_others : ∀ i j k, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k) :
  ∀ i, scores i > 18 := 
by sorry

end olympiad_scores_l800_800357


namespace remainder_of_power_mod_l800_800997

theorem remainder_of_power_mod :
  (5^2023) % 11 = 4 :=
  by
    sorry

end remainder_of_power_mod_l800_800997


namespace count_congruent_to_3_mod_8_in_300_l800_800737

theorem count_congruent_to_3_mod_8_in_300 : 
  {n : ℤ | 1 ≤ n ∧ n ≤ 300 ∧ n % 8 = 3}.card = 38 := 
by
  sorry

end count_congruent_to_3_mod_8_in_300_l800_800737


namespace average_other_students_l800_800586

theorem average_other_students (total_students other_students : ℕ) (mean_score_first : ℕ) 
 (mean_score_class : ℕ) (mean_score_other : ℕ) (h1 : total_students = 20) (h2 : other_students = 10)
 (h3 : mean_score_first = 80) (h4 : mean_score_class = 70) :
 mean_score_other = 60 :=
by
  sorry

end average_other_students_l800_800586


namespace trigonometric_identity_verification_l800_800531

noncomputable def cos (x : ℝ) : ℝ := real.cos x
noncomputable def sin (x : ℝ) : ℝ := real.sin x

theorem trigonometric_identity_verification : 
  3.427 * cos (50 * real.pi / 180) + 
  8 * cos (200 * real.pi / 180) * cos (220 * real.pi / 180) * cos (80 * real.pi / 180) = 
  2 * (sin (65 * real.pi / 180))^2 := 
by sorry

/-- Additional helper conditions given in the original problem: -/

-- cos(200°) = -cos(20°)
lemma cos_200_eq_neg_cos_20 : cos (200 * real.pi / 180) = -cos (20 * real.pi / 180) :=
by sorry
  
-- cos(220°) = -cos(40°)
lemma cos_220_eq_neg_cos_40 : cos (220 * real.pi / 180) = -cos (40 * real.pi / 180) :=
by sorry

end trigonometric_identity_verification_l800_800531


namespace d_value_of_ellipse_tangent_l800_800154

/-- Given an ellipse tangent to both the x-axis and y-axis in the first quadrant, 
and having foci at (4,8) and (d,8), we prove that d equals 15. -/
theorem d_value_of_ellipse_tangent (d : ℝ) 
  (h : 2 * real.sqrt (((d - 4) / 2) ^ 2 + 64) = d + 4) : d = 15 :=
begin
  sorry
end

end d_value_of_ellipse_tangent_l800_800154


namespace log_domain_l800_800473

-- We define the logarithmic function condition
def log_domain_condition (x : ℝ) : Prop := 2 - x > 0

-- We state that the domain of the function log(2 - x) is the set of all x in ℝ such that 2 - x > 0, which translates to x < 2
theorem log_domain : {x : ℝ // log_domain_condition x} = set_of (λ x : ℝ, x < 2) :=
by { sorry }

end log_domain_l800_800473


namespace visible_shaded_area_l800_800775

open Real

/-- Conditions -/
def grid_total_area := 144
def circle_area (r : ℝ) := π * r ^ 2
def overlap_area (r : ℝ) (d : ℝ) := 2 * r ^ 2 * arccos (d / (2 * r)) - (d / 2) * sqrt (4 * r ^ 2 - d ^ 2)

/-- Given two circles each with radius 3 cm and the centers are 3 cm apart,
    the area of each circle is 9π and the overlap area is 6π. 

    The visible shaded area of the grid is grid_total_area - circle_area(3) + overlap_area. --/
theorem visible_shaded_area (r d : ℝ) (h1 : r = 3) (h2 : d = 3) :
  let C := 144
  let D := 6 in
  C - D * π = grid_total_area - 6 * π ∧ C + D = 150 :=
by
  sorry

end visible_shaded_area_l800_800775


namespace power_function_decreasing_l800_800283

theorem power_function_decreasing :
  ∀ (x : ℝ), (0 < x) → (∃ m : ℝ, m^2 - m - 1 = 1 ∧ y = (m^2 - m - 1) * x^(m^2 - 2m - 3) ∧
                 (∀ y : ℝ, y = x^(m^2 - 2m - 3) → ∀ (h : 0 < x),
                 x > y → x > x^(m^2 - 2m - 3))) → y = x^(-3) :=
by
  sorry

end power_function_decreasing_l800_800283


namespace trigonometric_sum_l800_800969

theorem trigonometric_sum :
  sin (-5 * Real.pi / 3) + cos (-5 * Real.pi / 4) + tan (-11 * Real.pi / 6) + cot (-4 * Real.pi / 3) = (Real.sqrt 3 - Real.sqrt 2) / 2 := 
by
  sorry

end trigonometric_sum_l800_800969


namespace find_line_bisecting_circle_l800_800005

theorem find_line_bisecting_circle :
  ∃ l : ℝ → ℝ, 
    (∀ x y : ℝ, (l x = y) ↔ (2 * x - y = 0)) ∧ 
    (∀ x y : ℝ, (x^2 + y^2 - 2 * x - 4 * y = 0) → (1, 2) on l) ∧ 
    (∀ x y : ℝ, (l x = y) → (x + 2 * y = 0) = False) :=
begin
  -- statement only, no proof
  sorry
end

end find_line_bisecting_circle_l800_800005


namespace blueberry_picking_relationship_l800_800910

theorem blueberry_picking_relationship (x : ℝ) (hx : x > 10) : 
  let y1 := 60 + 18 * x
  let y2 := 150 + 15 * x
  in y1 = 60 + 18 * x ∧ y2 = 150 + 15 * x := 
by {
  sorry
}

end blueberry_picking_relationship_l800_800910


namespace maximum_value_of_T_l800_800319

-- Definition of conditions
def point_moves_on_line_segment (P : ℝ × ℝ) : Prop :=
  ∃ x y : ℝ, P = (x, y) ∧ x + 2 * y = 4 ∧ x > 0 ∧ y > 0

-- Statement of the theorem
theorem maximum_value_of_T :
  ∃ (P : ℝ × ℝ), point_moves_on_line_segment P ∧
  let (x, y) := P in (log 2 x / log 2 2) + (log 2 y / log 2 2) = 1 :=
sorry

end maximum_value_of_T_l800_800319


namespace sequence_integer_l800_800434

variable (a : ℝ) (n : ℕ)

theorem sequence_integer (h : a + 1/a ∈ ℤ) (hn : 2 ≤ n ∧ n ≤ 7) : a^n + 1/a^n ∈ ℤ :=       
sorry

end sequence_integer_l800_800434


namespace number_divisible_by_396_l800_800837

theorem number_divisible_by_396 :
  ∀ digits : list ℕ, 
  (digits.length = 10 ∧ digits.nodup ∧ digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) →
  ∃ n : ℕ, 
  (fill_digits_in_number 3 4 1 0 8 2 40923 0 320 2 56 digits = n ∧ n % 396 = 0) :=
by
  sorry

end number_divisible_by_396_l800_800837


namespace parallelogram_area_is_15_l800_800823

noncomputable def distance (a b : ℝ × ℝ) : ℝ := 
  real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2)

noncomputable def parallelogram_area (P Q R S : ℝ × ℝ) : ℝ := 
  abs ((P.1 * Q.2 + Q.1 * R.2 + R.1 * S.2 + S.1 * P.2) 
       - (P.2 * Q.1 + Q.2 * R.1 + R.2 * S.1 + S.2 * P.1)) / 2

theorem parallelogram_area_is_15 :
  ∃ P Q R S W Z : ℝ × ℝ, 
  P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ R ∧ Q ≠ S ∧ R ≠ S ∧
  (W.1 = 0 ∧ W.2 = 0) ∧ 
  (P.1 = 5 ∧ P.2 = 0) ∧ 
  (Z.1 = 1 ∧ Z.2 = 0) ∧ 
  (R.1 = 5 ∧ R.2 = 0) ∧ 
  (distance P W = 5) ∧ 
  (distance W Z = 1) ∧ 
  (distance Z R = 4) ∧ 
  (P.1 = R.1 ∧ P.2 ≠ R.2 ∧ Z.2 = 0) ∧ 
  ((distance P R) = (distance Q S)) ∧ 
  (W.1 < Z.1) ∧ 
  (parallelogram_area P Q R S = 15) := sorry

end parallelogram_area_is_15_l800_800823


namespace overall_discount_correct_l800_800601

noncomputable def purchasing_cost : ℝ := 100

def item_data := {
  markup_percent: ℝ,
  profit_percent: ℝ
}

def items : list item_data := [
  {markup_percent := 50, profit_percent := 12.5},
  {markup_percent := 75, profit_percent := 22.5},
  {markup_percent := 100, profit_percent := 30},
  {markup_percent := 150, profit_percent := 50}
]

def calculate_discount (item : item_data) : ℝ :=
  let marked_price := purchasing_cost * (1 + item.markup_percent / 100) in
  let selling_price := purchasing_cost * (1 + item.profit_percent / 100) in
  marked_price - selling_price

def total_marked_price (items : list item_data) : ℝ :=
  items.sumBy (λ item, purchasing_cost * (1 + item.markup_percent / 100))

def total_discount (items : list item_data) : ℝ :=
  items.sumBy calculate_discount

def overall_discount_percentage (items : list item_data) : ℝ :=
  (total_discount items / total_marked_price items) * 100

theorem overall_discount_correct :
  overall_discount_percentage items ≈ 33.55 :=
sorry

end overall_discount_correct_l800_800601


namespace ratio_of_kids_in_morning_to_total_soccer_l800_800887

-- Define the known conditions
def total_kids_in_camp : ℕ := 2000
def kids_going_to_soccer_camp : ℕ := total_kids_in_camp / 2
def kids_going_to_soccer_camp_in_afternoon : ℕ := 750
def kids_going_to_soccer_camp_in_morning : ℕ := kids_going_to_soccer_camp - kids_going_to_soccer_camp_in_afternoon

-- Define the conclusion to be proven
theorem ratio_of_kids_in_morning_to_total_soccer :
  (kids_going_to_soccer_camp_in_morning : ℚ) / (kids_going_to_soccer_camp : ℚ) = 1 / 4 :=
by
  sorry

end ratio_of_kids_in_morning_to_total_soccer_l800_800887


namespace min_value_cubic_roots_l800_800261

theorem min_value_cubic_roots 
  (A B C : ℝ) 
  (α β γ : ℝ)
  (h1 : polynomial.eval (polynomial.C A * polynomial.X^2 + polynomial.C B * polynomial.X + polynomial.C C) α = 0)
  (h2 : polynomial.eval (polynomial.C A * polynomial.X^2 + polynomial.C B * polynomial.X + polynomial.C C) β = 0)
  (h3 : polynomial.eval (polynomial.C A * polynomial.X^2 + polynomial.C B * polynomial.X + polynomial.C C) γ = 0)
  (h4 : A = -(α + β + γ))
  (h5 : B = α*β + β*γ + γ*α)
  (h6 : C = -α*β*γ) : 
  ∃ θ : ℝ, θ = (|α| + |β| + |γ|) ∧ 
  \frac{1 + |A| + |B| + |C|}{|\alpha| + |\beta| + |\gamma|} ≥ \frac{³√|2|}{2} := 
sorry

end min_value_cubic_roots_l800_800261


namespace triangle_distance_equality_l800_800428

variables {A B C D E F : Type*} [MetricSpace A B C D E F]
variables (ABC : Triangle A B C) (D_pos : A ≠ D ∧ D ∈ lineThrough A B) (E_pos : A ≠ E ∧ E ∈ lineThrough A C)
variables (F_intersection : ∃ F, F = intersection (lineThrough B E) (lineThrough C D))
variables (dist_condition : dist A E + dist E F = dist A D + dist D F)

theorem triangle_distance_equality (h : AE + EF = AD + DF) :
  dist A C + dist C F = dist A B + dist B F :=
sorry

end triangle_distance_equality_l800_800428


namespace matrix_transformation_P_l800_800202

theorem matrix_transformation_P (N : Matrix (Fin 3) (Fin 3) ℝ) :
  ∃ (P : Matrix (Fin 3) (Fin 3) ℝ), (∀ (N : Matrix (Fin 3) (Fin 3) ℝ), 
  P ⬝ N = ⟨[(N 2 0), (N 2 1), (N 2 2)], 
           [3 * (N 1 0), 3 * (N 1 1), 3 * (N 1 2)],
           [(N 0 0), (N 0 1), (N 0 2)]⟩) ∧ 
  P = ⟨[(0 : ℝ), 0, 1], 
       [0, 3, 0],
       [1, 0, 0]⟩ :=
begin
  sorry
end

end matrix_transformation_P_l800_800202


namespace min_colors_2016x2016_l800_800544

def min_colors_needed (n : ℕ) : ℕ :=
  ⌈log 2 (n * n)⌉.natAbs

theorem min_colors_2016x2016 :
  min_colors_needed 2016 = 11 := 
by
  sorry

end min_colors_2016x2016_l800_800544


namespace b_50_value_l800_800695

open Nat

def b : ℕ → ℕ
| 0     := 0
| (n+1) := if n = 0 then 3 else 3 * (T n) + 1

def T : ℕ → ℕ
| 0     := 0
| (n+1) := T n + b (n+1)

theorem b_50_value : b 50 = 4^48 * 4 := by
  sorry

end b_50_value_l800_800695


namespace sign_of_slope_same_as_sign_of_correlation_l800_800845

theorem sign_of_slope_same_as_sign_of_correlation
  (x y : ℝ)  -- x, y are real numbers (representing the two variables)
  (r : ℝ)    -- r is the correlation coefficient
  (b a : ℝ)  -- b is the slope, a is the intercept
  (h : y = a + b * x) -- linear relationship condition
  (hr : r ≠ 0) -- r is nonzero (implying a true correlation)
  (hx : ∀ x, x ≠ 0) -- x is non-zero (for simplification)
  : sign b = sign r := sorry

end sign_of_slope_same_as_sign_of_correlation_l800_800845


namespace area_ratio_triangle_DME_ABC_l800_800393

theorem area_ratio_triangle_DME_ABC 
  (ABC : Triangle)
  (acute_ABC : ABC.is_acute)
  (M : Point)
  (midpoint_M : M.is_midpoint (ABC.BC))
  (AM_eq_BC : distance ABC.A M = distance ABC.BC)
  (D : Point)
  (E : Point)
  (D_bisector_AMB : D.is_angle_bisector (ABC.AMB))
  (E_bisector_AMC : E.is_angle_bisector (ABC.AMC)) :
  area (triangle D M E) / area (ABC) = 2 / 9 :=
sorry

end area_ratio_triangle_DME_ABC_l800_800393


namespace trajectory_of_P_max_triangle_area_l800_800559
-- Import necessary Lean libraries

-- Definitions and proof problem statement
noncomputable theory
open Real

-- Definitions for the conditions
def A : Point := (some appropriate definition)
def B : Point := (some appropriate definition)
def P : Point := (some appropriate definition)

-- Condition that |PA| + |PB| = 2
def condition (P : Point) : Prop := dist P A + dist P B = 2

-- Theorem stating the trajectory of point P is a circle with equation x^2 + y^2 = 1
theorem trajectory_of_P 
  (P : Point) 
  (h : condition P) : 
  ∃ x y, x^2 + y^2 = 1 := sorry

-- Definitions for the second condition and question
def line_l (k : ℝ) (Hk : k > 0) := { p : Point | p.y = k }

-- Theorem stating the maximum area of ΔBMN and the equation of line l at this time
theorem max_triangle_area 
  (k : ℝ) (Hk : k > 0) 
  (M N : Point) 
  (hM : M ∈ line_l k Hk)
  (hN : N ∈ line_l k Hk)
  (hP : condition P) 
  (hM_intersect : M ∈ trajectory_of_P)
  (hN_intersect : N ∈ trajectory_of_P): 
  ∃ (max_area_equ : ℝ) (line_equ : string), 
    max_area_equ = 1/2 ∧ line_equ = "y = 1/2" := sorry

end trajectory_of_P_max_triangle_area_l800_800559


namespace probability_different_grades_l800_800584

theorem probability_different_grades (A B : Type) [Fintype A] [Fintype B] (ha : Fintype.card A = 2) (hb : Fintype.card B = 2) :
  (∃ (s : Finset (A ⊕ B)), s.card = 2) →
  (Fintype.card (Finset (A ⊕ B)).filter (λ s, (∃ (a : A) (b : B), s = {sum.inl a, sum.inr b})) = 4) →
  (Fintype.card (Finset (A ⊕ B)).card-choose 2 = 6) →
  (Fintype.card (Finset (A ⊕ B)).filter (λ s, (∃ (a : A) (b : B), s = {sum.inl a, sum.inr b})) /
     Fintype.card (Finset (A ⊕ B)).card-choose 2 = 2 / 3) :=
sorry

end probability_different_grades_l800_800584


namespace arithmetic_geometric_sequence_l800_800713

theorem arithmetic_geometric_sequence (a b : ℝ)
  (h1 : 2 * a = 1 + b)
  (h2 : b^2 = a)
  (h3 : a ≠ b) : a = 1 / 4 :=
by
  sorry

end arithmetic_geometric_sequence_l800_800713


namespace Ryan_spit_distance_correct_l800_800644

-- Definitions of given conditions
def Billy_spit_distance : ℝ := 30
def Madison_spit_distance : ℝ := Billy_spit_distance * 1.20
def Ryan_spit_distance : ℝ := Madison_spit_distance * 0.50

-- Goal statement
theorem Ryan_spit_distance_correct : Ryan_spit_distance = 18 := by
  -- proof would go here
  sorry

end Ryan_spit_distance_correct_l800_800644


namespace sum_of_integers_sqrt_485_l800_800490

theorem sum_of_integers_sqrt_485 (x y : ℕ) (h1 : x^2 + y^2 = 245) (h2 : x * y = 120) : x + y = Real.sqrt 485 :=
sorry

end sum_of_integers_sqrt_485_l800_800490


namespace stratified_sampling_females_l800_800115

theorem stratified_sampling_females (total_students : ℕ) (male_students : ℕ) 
  (female_students : ℕ) (sample_size : ℕ) 
  (h1 : total_students = 45) 
  (h2 : male_students = 25) 
  (h3 : female_students = 20)
  (h4 : sample_size = 18) :
  let proportion_females := (female_students : ℚ) / total_students,
      females_in_sample := proportion_females * sample_size
  in females_in_sample = 8 :=
by
  sorry

end stratified_sampling_females_l800_800115


namespace largest_four_digit_number_l800_800055

theorem largest_four_digit_number :
  ∃ a b c d : ℕ, 
    9 < 1000 * a + 100 * b + 10 * c + d ∧ 
    1000 * a + 100 * b + 10 * c + d < 10000 ∧ 
    c = a + b ∧ 
    d = b + c ∧ 
    1000 * a + 100 * b + 10 * c + d = 9099 :=
by {
  sorry
}

end largest_four_digit_number_l800_800055


namespace gcd_lcm_sum_l800_800521

theorem gcd_lcm_sum : Nat.gcd 24 54 + Nat.lcm 40 10 = 46 := by
  sorry

end gcd_lcm_sum_l800_800521


namespace sum_b_k_l800_800876

def recurrence_relation (a_n a_n_plus_one : ℕ) (n : ℕ) :=
  ∃ b_n : ℤ, a_n + a_n_plus_one = -3 * n ∧ a_n * a_n_plus_one = b_n

def sum_b_k_equality (a : ℕ → ℕ) : ℕ → Prop
| 1 := a 1 = 1
| (n+1) := recurrence_relation (a n) (a (n+1)) n

theorem sum_b_k (a : ℕ → ℕ) (b : ℕ → ℤ) (k : ℕ) (h : ∀ n, 1 ≤ n → recurrence_relation (a n) (a (n+1)) n ∧ b n = a n * a (n+1)) :
  (∑ k in Finset.range 20, b (k + 1)) = 6385 :=
sorry

end sum_b_k_l800_800876


namespace relationship_p_q_l800_800764

theorem relationship_p_q (p q : ℝ) : 
  (expand_poly : (x : ℝ) → ((x^2 - p * x + q) * (x - 3)) = (x * x * x + (-p - 3) * x * x + (3 * p + q) * x - 3 * q)) → 
  (linear_term_condition : ∀ x, expand_poly x → (3 * p + q = 0)) → 
  q + 3 * p = 0 :=
begin
  sorry
end

end relationship_p_q_l800_800764


namespace bus_stops_for_15_minutes_per_hour_l800_800197

-- Define the conditions
def speed_without_stoppages := 64 -- in km/hr
def speed_with_stoppages := 48 -- in km/hr

-- Define the question in terms of Lean:
-- Prove that the bus stops for approximately 15 minutes per hour.
theorem bus_stops_for_15_minutes_per_hour :
  let speed_reduction := speed_without_stoppages - speed_with_stoppages
    km_per_minute := speed_without_stoppages / 60
    time_stopped := speed_reduction / km_per_minute
  in time_stopped ≈ 15 :=
by
  sorry

end bus_stops_for_15_minutes_per_hour_l800_800197


namespace value_of_y_plus_10_l800_800300

theorem value_of_y_plus_10 (x y : ℝ) (h1 : 3 * x = (3 / 4) * y) (h2 : x = 20) : y + 10 = 90 :=
by
  sorry

end value_of_y_plus_10_l800_800300


namespace twelve_divisible_by_three_l800_800495

theorem twelve_divisible_by_three (a b : ℕ) (h1 : a = 10) (h2 : b = 45) :
  (finset.filter (λ x, x % 3 = 0) (finset.range (b + 1))).card - (finset.filter (λ x, x % 3 = 0) (finset.range (a))).card = 12 :=
by 
  sorry

end twelve_divisible_by_three_l800_800495


namespace min_value_x_add_4_div_x_sub_1_l800_800234

theorem min_value_x_add_4_div_x_sub_1 (x : ℝ) (hx : x > 1) : 
  ∃ y, ∀ (z : ℝ), z > 1 → z + 4 / (z - 1) ≥ y ∧ (z = 3 → y = 5) := 
begin
  use 5,
  intros z hz,
  split,
  { have h1 : z - 1 > 0 := sub_pos.2 hz,
    have h2 : (z - 1 + 4 / (z - 1)) ≥ 2 * real.sqrt ((z - 1) * (4 / (z - 1))), from
      real.am_gm z (4 / (z - 1)) (h1).le (div_pos zero_lt_four h1).le,
    linarith, },
  { intro hz_eq_3,
    rw [hz_eq_3, sub_self 3, zero_add, div_self], norm_num, }
end

end min_value_x_add_4_div_x_sub_1_l800_800234


namespace golden_section_distance_l800_800962

theorem golden_section_distance (stage_length : ℝ) (golden_ratio : ℝ) :
  stage_length = 10 ∧
  golden_ratio = (1 + Real.sqrt 5) / 2 →
  ∃ d : ℝ, d = 10 * (3 - Real.sqrt 5) / 2 ∨ d = 10 * (1 + Real.sqrt 5) / 2 - 10 :=
by
  assume h
  sorry

end golden_section_distance_l800_800962


namespace chord_length_circle_l800_800857

theorem chord_length_circle {C : Type*} [EuclideanSpace C]
  (center : C) (radius : ℝ) (line_slope : ℝ) (line_point : C) :
  center = (2 : ℝ, 1 : ℝ) → radius = √2 →
  line_slope = 1 → line_point = (1 : ℝ, 1 : ℝ) →
  let d := |((2 : ℝ, 1 : ℝ) - (1 : ℝ, 1 : ℝ))| / √2 in
  2 * √(radius^2 - d^2) = √6 :=
by
  sorry

end chord_length_circle_l800_800857


namespace length_AB_is_16_l800_800727

open Real

def line (k : ℝ) : (ℝ × ℝ) → Prop := λ p, p.snd = k * (p.fst - 2)
def parabola : (ℝ × ℝ) → Prop := λ p, p.snd^2 = 8 * p.fst

def dot_product (u v : ℝ × ℝ) : ℝ := u.fst * v.fst + u.snd * v.snd

def length (p q : ℝ × ℝ) : ℝ := sqrt ((p.fst - q.fst) ^ 2 + (p.snd - q.snd) ^ 2)

def A (k x1 : ℝ) : ℝ × ℝ := (x1, k * (x1 - 2))
def B (k x2 : ℝ) : ℝ × ℝ := (x2, k * (x2 - 2))
def M : ℝ × ℝ := (-2, 4)

theorem length_AB_is_16 :
  ∀ (k x1 x2 : ℝ),
    parabola (A k x1) ∧ parabola (B k x2) ∧ line k (A k x1) ∧ line k (B k x2) ∧ 
    dot_product (A k x1 - M) (B k x2 - M) = 0 →
    length (A k x1) (B k x2) = 16 :=
by
  intros k x1 x2 h,
  sorry

end length_AB_is_16_l800_800727


namespace f_le_x_sq_l800_800400

variable (f : ℝ → ℝ)
variable {M : ℝ}

noncomputable def satisfies_conditions (f : ℝ → ℝ) (M : ℝ) : Prop :=
  (∀ x y, 0 ≤ x → 0 ≤ y → f(x) * f(y) ≤ y^2 * f(x / 2) + x^2 * f(y / 2)) ∧
  (M > 0) ∧ (∀ x, 0 ≤ x → x ≤ 1 → |f(x)| ≤ M)

theorem f_le_x_sq (f : ℝ → ℝ) (M : ℝ) (h : satisfies_conditions f M) :
  ∀ x, 0 ≤ x → f(x) ≤ x^2 :=
sorry

end f_le_x_sq_l800_800400


namespace conditions_on_a_b_c_l800_800235

noncomputable def odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
noncomputable def monotonic_on (f : ℝ → ℝ) (I : set ℝ) : Prop := ∀ x y ∈ I, x ≤ y → f x ≤ f y

theorem conditions_on_a_b_c 
  (a b c x : ℝ)
  (h₀ : odd (λ x => x^3 - a * x^2 - b * x + c))
  (h₁ : monotonic_on (λ x => x^3 - a * x^2 - b * x + c) (set.Ici 1)) :
  a = 0 ∧ c = 0 ∧ b ≤ 3 := 
sorry

end conditions_on_a_b_c_l800_800235


namespace solve_linear_system_l800_800402

theorem solve_linear_system :
  ∃ x y : ℤ, x + 9773 = 13200 ∧ 2 * x - 3 * y = 1544 ∧ x = 3427 ∧ y = 1770 := by
  sorry

end solve_linear_system_l800_800402


namespace bag_contains_fifteen_balls_l800_800565

theorem bag_contains_fifteen_balls 
  (r b : ℕ) 
  (h1 : r + b = 15) 
  (h2 : (r * (r - 1)) / 210 = 1 / 21) 
  : r = 4 := 
sorry

end bag_contains_fifteen_balls_l800_800565


namespace tickets_sold_l800_800617

theorem tickets_sold (student_tickets non_student_tickets student_ticket_price non_student_ticket_price total_revenue : ℕ)
  (h1 : student_ticket_price = 5)
  (h2 : non_student_ticket_price = 8)
  (h3 : total_revenue = 930)
  (h4 : student_tickets = 90)
  (h5 : non_student_tickets = 60) :
  student_tickets + non_student_tickets = 150 := 
by 
  sorry

end tickets_sold_l800_800617


namespace slope_bisector_of_acute_angle_l800_800444

theorem slope_bisector_of_acute_angle (m1 m2 : ℝ) (h1 : m1 = 2) (h2 : m2 = -2) :
  let k := (m1 + m2 + Real.sqrt (1 + m1^2 + m2^2)) / (1 - m1 * m2)
  in k = 3 / 5 :=
by
  sorry

end slope_bisector_of_acute_angle_l800_800444


namespace incenter_perpendicular_to_incenter_segment_l800_800628

open EuclideanGeometry

theorem incenter_perpendicular_to_incenter_segment
  (A B C D G I₁ I₂ I₃ : Point)
  (h_cyclic : is_cyclic_quadrilateral A B C D)
  (h_inter_G : intersect_diag_point AC BD G)
  (h_incenter_ADC : is_incenter I₁ (Triangle.mk A D C))
  (h_incenter_BDC : is_incenter I₂ (Triangle.mk B D C))
  (h_incenter_ABG : is_incenter I₃ (Triangle.mk A B G)) :
  is_perpendicular (Line.mk G I₃) (Line.mk I₁ I₂) :=
sorry

end incenter_perpendicular_to_incenter_segment_l800_800628


namespace investment_difference_l800_800799

/-- James and Maria's investment problem conditions -/
def james_initial_investment : ℝ := 60000
def james_interest_rate_per_period : ℝ := 0.025
def james_periods : ℝ := 6

def maria_initial_investment : ℝ := 70000
def maria_interest_rate_per_period : ℝ := 0.004167
def maria_periods : ℝ := 36

/-- The final amount of James's investment after the specified periods -/
def final_amount_james : ℝ :=
  james_initial_investment * (1 + james_interest_rate_per_period) ^ james_periods

/-- The final amount of Maria's investment after the specified periods -/
def final_amount_maria : ℝ :=
  maria_initial_investment * (1 + maria_interest_rate_per_period) ^ maria_periods

/-- Prove the difference in their final amounts is $12,408.92 -/
theorem investment_difference :
  final_amount_maria - final_amount_james = 12408.92 :=
sorry

end investment_difference_l800_800799


namespace find_c10_l800_800825

noncomputable def a : ℕ → ℕ
| 1     := 1
| 2     := 2
| (n+3) := a (n+2) + a (n+1) + 1

noncomputable def c : ℕ → ℕ
| 1     := 3
| 2     := 9
| (n+3) := c (n+2) * c (n+1)

theorem find_c10 : c 10 = 3 ^ a 10 :=
by
  have h_a10 : a 10 = 143 := by
    unfold a
    sorry

  have h_c10 : c 10 = 3 ^ 143 := by
    unfold c
    sorry

  exact h_c10

end find_c10_l800_800825


namespace largest_valid_four_digit_number_l800_800064

-- Definition of the problem conditions
def is_valid_number (a b c d : ℕ) : Prop :=
  c = a + b ∧ d = b + c

-- Proposition that we need to prove
theorem largest_valid_four_digit_number : ∃ (a b c d : ℕ),
  is_valid_number a b c d ∧ a * 1000 + b * 100 + c * 10 + d = 9099 :=
by
  sorry

end largest_valid_four_digit_number_l800_800064


namespace leadership_structuring_l800_800602

theorem leadership_structuring (total_members president_count vp_count dept_heads_per_vp: ℕ) 
  (h_mem : total_members = 13)
  (h_pres : president_count = 1)
  (h_vp : vp_count = 2)
  (h_dept : dept_heads_per_vp = 3)
  : (13 * (choose (total_members - president_count) vp_count) 
      * (choose (total_members - president_count - vp_count) dept_heads_per_vp) 
      * (choose (total_members - president_count - vp_count - dept_heads_per_vp) dept_heads_per_vp) 
      = 655920) :=
by sorry

end leadership_structuring_l800_800602


namespace positive_number_percent_l800_800943

theorem positive_number_percent (x : ℝ) (h : 0.01 * x^2 = 9) (hx : 0 < x) : x = 30 :=
sorry

end positive_number_percent_l800_800943


namespace rhombus_perimeter_l800_800458

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 20) (h2 : d2 = 16) :
  let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  in 4 * s = 8 * Real.sqrt 41 := 
by
  sorry

end rhombus_perimeter_l800_800458


namespace natural_numbers_between_sqrt_100_and_101_l800_800296

theorem natural_numbers_between_sqrt_100_and_101 :
  ∃ (n : ℕ), n = 200 ∧ (∀ k : ℕ, 100 < Real.sqrt k ∧ Real.sqrt k < 101 -> 10000 < k ∧ k < 10201) := 
by
  sorry

end natural_numbers_between_sqrt_100_and_101_l800_800296


namespace modular_inverse_expression_l800_800199

-- Definitions of the inverses as given in the conditions
def inv_7_mod_77 : ℤ := 11
def inv_13_mod_77 : ℤ := 6

-- The main theorem stating the equivalence
theorem modular_inverse_expression :
  (3 * inv_7_mod_77 + 9 * inv_13_mod_77) % 77 = 10 :=
by
  sorry

end modular_inverse_expression_l800_800199


namespace largest_four_digit_number_l800_800054

theorem largest_four_digit_number :
  ∃ a b c d : ℕ, 
    9 < 1000 * a + 100 * b + 10 * c + d ∧ 
    1000 * a + 100 * b + 10 * c + d < 10000 ∧ 
    c = a + b ∧ 
    d = b + c ∧ 
    1000 * a + 100 * b + 10 * c + d = 9099 :=
by {
  sorry
}

end largest_four_digit_number_l800_800054


namespace count_neither_multiples_of_2_nor_3_l800_800661

theorem count_neither_multiples_of_2_nor_3 : 
  let count_multiples (k n : ℕ) : ℕ := n / k
  let total_numbers := 100
  let multiples_of_2 := count_multiples 2 total_numbers
  let multiples_of_3 := count_multiples 3 total_numbers
  let multiples_of_6 := count_multiples 6 total_numbers
  let multiples_of_2_or_3 := multiples_of_2 + multiples_of_3 - multiples_of_6
  total_numbers - multiples_of_2_or_3 = 33 :=
by 
  sorry

end count_neither_multiples_of_2_nor_3_l800_800661


namespace sqrt_equation_solution_l800_800987

theorem sqrt_equation_solution (x : ℝ) (h : x > 4) :
  (sqrt (x - 4 * sqrt (x - 4)) + 2 = sqrt (x + 4 * sqrt (x - 4)) - 2) ↔ x ∈ set.Ici 8 :=
by 
  sorry

end sqrt_equation_solution_l800_800987


namespace tan_sum_eq_three_fourths_l800_800301

theorem tan_sum_eq_three_fourths (x y : ℝ) (h1 : sin x + sin y = 3 / 5) (h2 : cos x + cos y = 4 / 5) : tan x + tan y = 3 / 4 :=
by
  sorry

end tan_sum_eq_three_fourths_l800_800301


namespace range_of_a_l800_800719

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x < 1 then a - Real.exp x else x + 4 / x

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, f x a ≥ 4) : a ∈ set.Ici (Real.exp 1 + 4) :=
by
  have h_f_ge_4 : ∃ x : ℝ, f x a = 4 :=
    sorry  -- This is where the actual proof would be constructed.
  sorry  -- This is where the remaining proof would be.

end range_of_a_l800_800719


namespace problem_l800_800304

theorem problem (a b c : ℝ) (h1 : a = 3 ^ 0.1) (h2 : b = (Real.log 2) / (Real.log (1/3))) (h3 : c = (Real.log (1/3)) / (Real.log 2)) :
  c < b ∧ b < a :=
by
  sorry

end problem_l800_800304


namespace convert_to_base_k_l800_800678

noncomputable def base_k_eq (k : ℕ) : Prop :=
  4 * k + 4 = 36

theorem convert_to_base_k :
  ∃ k : ℕ, base_k_eq k ∧ (67 / k^2 % k^2 % k = 1 ∧ 67 / k % k = 0 ∧ 67 % k = 3) :=
sorry

end convert_to_base_k_l800_800678


namespace distance_between_points_l800_800207

open Real

def point := ℝ × ℝ

def distance (p1 p2 : point) : ℝ :=
  sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem distance_between_points :
  let p1 : point := (2, -7)
  let p2 : point := (-8, 4)
  distance p1 p2 = sqrt 221 :=
by
  sorry

end distance_between_points_l800_800207


namespace t_shaped_grid_sum_l800_800842

open Finset

theorem t_shaped_grid_sum :
  ∃ (a b c d e : ℕ), 
    a ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Finset ℕ) ∧
    b ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Finset ℕ) ∧
    c ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Finset ℕ) ∧
    d ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Finset ℕ) ∧
    e ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Finset ℕ) ∧
    (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧
    (b ≠ c) ∧ (b ≠ d) ∧ (b ≠ e) ∧
    (c ≠ d) ∧ (c ≠ e) ∧
    (d ≠ e) ∧
    a + b + c = 20 ∧
    d + e = 7 ∧
    (a + b + c + d + e + b) = 33 :=
sorry

end t_shaped_grid_sum_l800_800842


namespace third_year_students_in_school_l800_800613

theorem third_year_students_in_school
  (first_year_students : ℕ)
  (total_selected : ℕ)
  (second_year_selected : ℕ)
  (is_arithmetic_mean : ℕ → ℕ → ℕ → Prop)
  (third_year_students : ℕ)
  (third_year_students_selected : ℕ)
  (number_of_third_year_students_in_school : ℕ) :
  first_year_students = 720 →
  total_selected = 180 →
  second_year_selected = 40 →
  is_arithmetic_mean total_selected second_year_selected third_year_students →
  number_of_third_year_students_in_school = 960 :=
begin
  sorry
end

def is_arithmetic_mean (a b c : ℕ) : Prop := 
  a = (b + c) / 2

end third_year_students_in_school_l800_800613


namespace line_equation_standard_form_l800_800878

-- Define the given conditions
def slope : ℝ := -2
def y_intercept : ℝ := 4

-- Define the goal (equation of the line in standard form)
def line_equation (x y : ℝ) : Prop := 2 * x + y - 4 = 0

-- State the theorem to be proved
theorem line_equation_standard_form (x y : ℝ) :
  (slope = -2 ∧ y_intercept = 4) → line_equation x y :=
by
  intros h
  sorry

end line_equation_standard_form_l800_800878


namespace extreme_points_range_of_a_l800_800720

def f (a x : ℝ) := (Real.exp x) / x + a * (x - Real.log x)

theorem extreme_points_range_of_a (a : ℝ) :
  (∀ x ∈ Ioo (1/2:ℝ) 2,
    (∃ y ∈ Ioo (1/2:ℝ) 2, y ≠ x ∧ f a x = 0 ∧ f a y = 0)) ↔ (-2 * Real.sqrt Real.exp 1 < a ∧ a < -Real.exp 1) := 
sorry

end extreme_points_range_of_a_l800_800720


namespace village_population_origin_l800_800104

def initial_population (current_population : ℝ) (died_percentage : ℝ) (left_percentage : ℝ) : ℝ :=
  current_population / ((1 - left_percentage) * (1 - died_percentage))

theorem village_population_origin 
  (current_population : ℝ) 
  (died_percentage : ℝ) 
  (left_percentage : ℝ) 
  (final_population : ℝ) 
  (h1 : died_percentage = 0.1) 
  (h2 : left_percentage = 0.2) 
  (h3 : final_population = 4554) : 
  initial_population final_population died_percentage left_percentage = 6325 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end village_population_origin_l800_800104


namespace angle_BHC_in_triangle_l800_800796

-- Define the problem within Lean
theorem angle_BHC_in_triangle (A B C D E F H : Type) 
    (h_triangle: ∀ P Q R : Type, Type)
    (h_orthocenter: Orthocenter A B C H) 
    (h_altitudes: Altitudes A B C D E F H)
    (angle_ABC : angle A B C = 30)
    (angle_ACB : angle A C B = 70) :
    angle B H C = 100 :=
  sorry

end angle_BHC_in_triangle_l800_800796


namespace no_two_birch_adjacent_l800_800938

noncomputable def calculate_probability_no_birch_adjacent : ℚ := 
  let total_ways := fact 12
  let non_birch_ways := fact 7
  let birch_spaces := (finset.range 8).choose 5
  let birch_permutations := fact 5
  ((non_birch_ways * (birch_spaces * birch_permutations)) : ℚ) / total_ways

theorem no_two_birch_adjacent (m n : ℕ) (h : no_two_birch_adjacent_probability = m / n) :
  m + n = 106 :=
by
  -- provided that the probability is simplified to 7 / 99
  have : no_two_birch_adjacent_probability = 7 / 99 := sorry
  sorry

end no_two_birch_adjacent_l800_800938


namespace probability_is_two_thirds_l800_800577

-- Define the general framework and conditions
def total_students : ℕ := 4
def students_from_first_grade : ℕ := 2
def students_from_second_grade : ℕ := 2

-- Define the combinations for selecting 2 students out of 4
def total_ways_to_select_2_students : ℕ := Nat.choose total_students 2

-- Define the combinations for selecting 1 student from each grade
def ways_to_select_1_from_first : ℕ := Nat.choose students_from_first_grade 1
def ways_to_select_1_from_second : ℕ := Nat.choose students_from_second_grade 1
def favorable_ways : ℕ := ways_to_select_1_from_first * ways_to_select_1_from_second

-- The target probability calculation
noncomputable def probability_of_different_grades : ℚ :=
  favorable_ways / total_ways_to_select_2_students

-- The statement and proof requirement (proof is deferred with sorry)
theorem probability_is_two_thirds :
  probability_of_different_grades = 2 / 3 :=
by sorry

end probability_is_two_thirds_l800_800577


namespace problem_part_one_problem_part_two_problem_part_three_l800_800818

variable {f : ℝ → ℝ}

-- Condition: f is Cauchy additive.
axiom func_add : ∀ x y : ℝ, f(x + y) = f(x) + f(y)

-- Given condition for (3)
axiom func_inc : ∀ a b : ℝ, a < b → f(a) < f(b)
axiom f_one : f 1 = 1

@[simp]
theorem problem_part_one : f 0 = 0 := 
sorry

theorem problem_part_two : ∀ x : ℝ, f (-x) = -f x := 
sorry

theorem problem_part_three {a : ℝ} : f (2 * a) > f (a - 1) + 2 → a > 1 := 
sorry

end problem_part_one_problem_part_two_problem_part_three_l800_800818


namespace pow_modulus_l800_800994

theorem pow_modulus : (5 ^ 2023) % 11 = 3 := by
  sorry

end pow_modulus_l800_800994


namespace train_length_is_correct_l800_800136

def speed_km_per_hr : ℝ := 70
def time_s : ℝ := 36

def speed_m_per_s (speed_km_per_hr : ℝ) : ℝ :=
  speed_km_per_hr * 1000 / 3600

def distance (speed_m_per_s : ℝ) (time_s : ℝ) : ℝ :=
  speed_m_per_s * time_s

theorem train_length_is_correct :
  let speed_m_s := speed_m_per_s speed_km_per_hr
  let length_of_train := distance speed_m_s time_s
  abs (length_of_train - 699.84) < 0.01 :=
by
  let speed_m_s := speed_m_per_s speed_km_per_hr
  let length_of_train := distance speed_m_s time_s
  sorry

end train_length_is_correct_l800_800136


namespace determine_range_of_a_l800_800716

noncomputable def quadratic_inequality_with_three_integer_solutions (a : ℝ) : Prop :=
  ∃ (x1 x2 x3 : ℤ), 
    2 * (x1 : ℝ) ^ 2 - 17 * (x1 : ℝ) + a ≤ 0 ∧ 
    2 * (x2 : ℝ) ^ 2 - 17 * (x2 : ℝ) + a ≤ 0 ∧ 
    2 * (x3 : ℝ) ^ 2 - 17 * (x3 : ℝ) + a ≤ 0 ∧ 
    x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3

theorem determine_range_of_a (a : ℝ) (h : quadratic_inequality_with_three_integer_solutions a) : -33 ≤ a ∧ a < -30 :=
begin
  sorry
end

end determine_range_of_a_l800_800716


namespace prob_two_more_heads_than_tails_eq_210_1024_l800_800758

-- Let P be the probability of getting exactly two more heads than tails when flipping 10 coins.
def P (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) / (2^n : ℚ)

theorem prob_two_more_heads_than_tails_eq_210_1024 :
  P 10 6 = 210 / 1024 :=
by
  -- The steps leading to the proof are omitted and hence skipped
  sorry

end prob_two_more_heads_than_tails_eq_210_1024_l800_800758


namespace trigonometric_identity_l800_800677

variable (α : ℝ)

theorem trigonometric_identity (h1 : 0 < α ∧ α < π) (h2 : sin α * cos α = -1 / 2) : 
  (1 / (1 + sin α) + 1 / (1 + cos α) = 4) :=
sorry

end trigonometric_identity_l800_800677


namespace simplify_trig_expr_l800_800838

noncomputable def sin15 := Real.sin (Real.pi / 12)
noncomputable def sin30 := Real.sin (Real.pi / 6)
noncomputable def sin45 := Real.sin (Real.pi / 4)
noncomputable def sin60 := Real.sin (Real.pi / 3)
noncomputable def sin75 := Real.sin (5 * Real.pi / 12)
noncomputable def cos10 := Real.cos (Real.pi / 18)
noncomputable def cos20 := Real.cos (Real.pi / 9)
noncomputable def cos30 := Real.cos (Real.pi / 6)

theorem simplify_trig_expr :
  (sin15 + sin30 + sin45 + sin60 + sin75) / (cos10 * cos20 * cos30) = 5.128 :=
sorry

end simplify_trig_expr_l800_800838


namespace alternating_signs_sum_l800_800183

def sum_alternating_signs : ℕ → ℤ
| 0     := 0
| (n+1) := if ∃ m : ℕ, (n+1) = (m+1)^2 then -((n+1) : ℤ) + sum_alternating_signs n else (n+1 : ℤ) + sum_alternating_signs n

theorem alternating_signs_sum :
  sum_alternating_signs 100 = 275 :=
sorry

end alternating_signs_sum_l800_800183


namespace gg_eq_3_has_three_solutions_l800_800399

def g (x : ℝ) : ℝ :=
if x ≤ 1 then -x + 2 else 3*x - 7

theorem gg_eq_3_has_three_solutions :
  {x : ℝ | g (g x) = 3}.finite.to_finset.card = 3 := 
sorry

end gg_eq_3_has_three_solutions_l800_800399


namespace find_distance_between_parallel_sides_l800_800988

theorem find_distance_between_parallel_sides 
  (a b : ℝ) (A : ℝ) (h : ℝ) 
  (h_a : a = 20) 
  (h_b : b = 18) 
  (h_A : A = 228) 
  (area_formula : A = 1 / 2 * (a + b) * h) :
  h = 12 :=
by
  rw [h_a, h_b] at area_formula
  simp at area_formula
  have h_19 : 19 * h = 228 := by linarith
  rw ← h_19
  exact eq_of_eq_mul_right (by norm_num) (eq.symm (div_eq_iff (by norm_num)).mpr rfl)

end find_distance_between_parallel_sides_l800_800988


namespace sum_reciprocal_f_l800_800817

-- Define f such that f(n) = m if and only if (m - 1/2)^3 < n <= (m + 1/2)^3
def f (n : ℕ) : ℕ :=
  Nat.find (λ m, (m:ℚ - 1/2) ^ 3 < n ∧ n ≤ (m:ℚ + 1/2) ^ 3)

-- State the sum of reciprocals of f(k) from k = 1 to 3045 equals 319.04
theorem sum_reciprocal_f :
  ∑ k in Finset.range 3045, (1 / f (k + 1) : ℚ) = 319.04 := sorry

end sum_reciprocal_f_l800_800817


namespace intersection_eq_1_2_l800_800827

-- Define the set M
def M : Set ℝ := {y : ℝ | -2 ≤ y ∧ y ≤ 2}

-- Define the set N
def N : Set ℝ := {x : ℝ | 1 < x}

-- The intersection of M and N
def intersection : Set ℝ := { x : ℝ | 1 < x ∧ x ≤ 2 }

-- Our goal is to prove that M ∩ N = (1, 2]
theorem intersection_eq_1_2 : (M ∩ N) = (Set.Ioo 1 2) :=
by
  sorry

end intersection_eq_1_2_l800_800827


namespace find_a_l800_800557

theorem find_a (z a : ℂ) (h1 : ‖z‖ = 2) (h2 : (z - a)^2 = a) : a = 2 :=
sorry

end find_a_l800_800557


namespace value_of_f_90_l800_800477

def f : ℕ → ℕ
| n := if n ≥ 1000 then n - 3 else f (f (n + 7))

theorem value_of_f_90 : f 90 = 999 :=
sorry

end value_of_f_90_l800_800477


namespace apex_angle_of_cone_l800_800898

noncomputable def angle_at_apex_of_cone : ℝ :=
2 * Real.arcCot 3 -- α = ∠
or
2 * Real.arcCot (4 / 3) -- β = ∠

theorem apex_angle_of_cone (r : ℝ) (d : ℝ) (h₁ : r = 12) (h₂ : d = 13) :
  2 * Real.arcCot 3 = angle_at_apex_of_cone ∨ 2 * Real.arcCot (4 / 3) = angle_at_apex_of_cone :=
begin
  sorry
end

end apex_angle_of_cone_l800_800898


namespace area_comparison_l800_800806

-- Definitions based on conditions in the problem
variable {A B M C A' B' O : Type} -- Points on the circle and related points
variable [MetricSpace A] [MetricSpace B] [MetricSpace M]
variable [MetricSpace C] [MetricSpace A'] [MetricSpace B']
variable [MetricSpace O]

-- Given conditions
variable (is_on_circle : Circle O A)
variable (is_on_circle : Circle O B)
variable (midpoint_M : Midpoint A B M)
variable (orth_proj_C : OrthogonalProjection B (TangentLine A))
variable (tangent_intersect_A' : TangentLine M ∩ AC = {A'})
variable (tangent_intersect_B' : TangentLine M ∩ BC = {B'})

-- Given angle condition
variable (angle_condition : ∠BAC < π / 8)

-- Main theorem to prove
theorem area_comparison (h : ∠BAC < π / 8) :
  Area ABC < 2 * Area A'B'C :=
sorry

end area_comparison_l800_800806


namespace root_exists_in_interval_l800_800449

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem root_exists_in_interval : ∃ x ∈ set.Ioo (2 : ℝ) 3, f x = 0 :=
by
  have h_mono : StrictMonoOn f (set.Ioi 0) :=
    λ x y hx hy hxy, by
      have h_diff : deriv f x = 1 / x + 2 :=
        by
          rw [f, deriv_add, deriv_log', deriv_smul_const, deriv_sub]
          simp
      sorry
  have h_f2 : f 2 < 0 :=
    by
      rw [f]
      norm_num
      simp [Real.log_eq_zero]
  have h_f3 : 0 < f 3 :=
    by
      rw [f]
      norm_num
      simp [Real.log_eq_zero]
  have h_exists := IntermediateValueTheoremVector f 2 3 h_f2 h_f3
  sorry

end root_exists_in_interval_l800_800449


namespace tangent_line_at_zero_l800_800208

noncomputable def f (x : ℝ) : ℝ := Real.sin x + x - 1

theorem tangent_line_at_zero : ∀ x : ℝ, (tangent_line x = 2 * x - 1) :=
by
  sorry

end tangent_line_at_zero_l800_800208


namespace set_M_description_l800_800322

variable {M : Set ℝ}

theorem set_M_description
  (h1 : ∃ x ∈ M, x < 0)
  (h2 : ∀ x ∈ M, x < 3) :
  M = Iio 1 ∨ M = Ioo (-3) 3 := 
sorry

end set_M_description_l800_800322


namespace probability_is_two_thirds_l800_800579

-- Define the general framework and conditions
def total_students : ℕ := 4
def students_from_first_grade : ℕ := 2
def students_from_second_grade : ℕ := 2

-- Define the combinations for selecting 2 students out of 4
def total_ways_to_select_2_students : ℕ := Nat.choose total_students 2

-- Define the combinations for selecting 1 student from each grade
def ways_to_select_1_from_first : ℕ := Nat.choose students_from_first_grade 1
def ways_to_select_1_from_second : ℕ := Nat.choose students_from_second_grade 1
def favorable_ways : ℕ := ways_to_select_1_from_first * ways_to_select_1_from_second

-- The target probability calculation
noncomputable def probability_of_different_grades : ℚ :=
  favorable_ways / total_ways_to_select_2_students

-- The statement and proof requirement (proof is deferred with sorry)
theorem probability_is_two_thirds :
  probability_of_different_grades = 2 / 3 :=
by sorry

end probability_is_two_thirds_l800_800579


namespace revenue_from_full_price_tickets_l800_800611

def total_tickets : Nat := 180
def total_revenue : ℝ := 2400
def full_price (f p : ℝ) : ℝ := f * p
def half_price (h p : ℝ) : ℝ := h * (p / 2)
def tickets (f h : ℕ) := f + h = total_tickets
def revenue (f h p : ℝ) := full_price f p + half_price h p = total_revenue

theorem revenue_from_full_price_tickets 
  (f h : ℕ) (p : ℝ) (htickets : tickets f h) 
  (hrevenue : revenue f h p) : 
  full_price (f : ℝ) p = 300 := 
sorry

end revenue_from_full_price_tickets_l800_800611


namespace number_of_large_boats_is_five_l800_800111

noncomputable def large_boats_rented (students boats student_capacity_large student_capacity_small : ℕ) (all_boats_occupied num_of_boats : Prop) : ℕ :=
if h : all_boats_occupied ∧ num_of_boats ∧ students = 50 ∧ boats = 10 ∧ student_capacity_large = 6 ∧ student_capacity_small = 4 then 5 else 0

theorem number_of_large_boats_is_five (students boats : ℕ) (all_boats_occupied num_of_boats : Prop) (student_capacity_large student_capacity_small : ℕ) :
  students = 50 → boats = 10 → student_capacity_large = 6 → student_capacity_small = 4 → all_boats_occupied → num_of_boats → large_boats_rented students boats student_capacity_large student_capacity_small all_boats_occupied num_of_boats = 5 :=
begin
  intros h1 h2 h3 h4 h5 h6,
  unfold large_boats_rented,
  split_ifs,
  exact rfl,
  cases h,
  contradiction,
end

end number_of_large_boats_is_five_l800_800111


namespace vasya_incorrect_calculation_l800_800332

theorem vasya_incorrect_calculation (x y : ℤ) (h1 : x + y = 2021) (h2 : 10 * x + y = 2221) : ¬ ∃ k : ℤ, x = k :=
by {
    intro h,
    cases h with k hk,
    have h3 : 9 * k = 200,
    { rw ← hk at h1 h2,
      have h3 : 10 * k + y = 2221,
      { rw [hk] at h2, exact h2 },
      have h4 : k + y = 2021,
      { rw [hk] at h1, exact h1 },
      linarith, },
    have h4 : ¬ ∃ k : ℤ, 9 * k = 200,
    { intro k,
      cases k with z ez,
      obtain ⟨m, n⟩ := @int.eq_coe_or_neg_coe z,
      { exact ⟨n, ⟨m, rfl⟩⟩, },
      apply (ne_of_gt (by norm_num : 200 > int.abs 200)).symm,
      contrapose! k,
      simp, },
    exact h4 ⟨k, h3⟩, }

end vasya_incorrect_calculation_l800_800332


namespace regional_salary_condition_l800_800330

-- Definitions based on conditions
def capital_workers := 1000000
def capital_salary := 81
def province_workers := 9000000
def province_salary := 1

def total_country_salary := capital_workers * capital_salary + province_workers * province_salary
def total_capital_salary := capital_workers * capital_salary
def total_province_salary := province_workers * province_salary

-- The percentage of salary in each region
def capital_salary_percentage := total_capital_salary / total_country_salary
def province_salary_percentage := total_province_salary / total_country_salary

-- Lean statement for the proof problem
theorem regional_salary_condition :
  (capital_salary_percentage = 0.9) ∧ (province_salary_percentage = 0.1) →
  (∀ (n : ℕ) (hn : n ≤ capital_workers), n * capital_salary ≤ (total_capital_salary * 11 / 100)) ∧
  (∀ (n : ℕ) (hn : n ≤ province_workers), n * province_salary ≤ (total_province_salary * 11 / 100)) :=
by {
    -- proof steps here, this is just the statement
    sorry
}

end regional_salary_condition_l800_800330


namespace problem_part1_problem_part2_l800_800265

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 2

theorem problem_part1 (h : ∀ x, x ∈ set.Icc (-5 : ℝ) 5 → f (-1) x = x^2 - 2*x + 2) :
  (∃ x_min ∈ set.Icc (-5 : ℝ) 5, ∀ x ∈ set.Icc (-5 : ℝ) 5, f (-1) x_min <= f (-1) x) ∧
  (∃ x_max ∈ set.Icc (-5 : ℝ) 5, ∀ x ∈ set.Icc (-5 : ℝ) 5, f (-1) x_max >= f (-1) x) :=
by {
  sorry
}

theorem problem_part2 :
  (∀ (a : ℝ), (∀ x ∈ set.Icc (-5 : ℝ) 5, monotone_on (f a) (set.Icc (-5 : ℝ) 5)) ↔ a >= 5) :=
by {
  sorry
}

end problem_part1_problem_part2_l800_800265


namespace monotonic_intervals_max_min_values_l800_800232

noncomputable def F (x : ℝ) : ℝ :=
  ∫ 0 to x, (t ^ 2 + 2 * t - 8) dt

theorem monotonic_intervals (x : ℝ) (h : 0 < x) :
  (∀ (y : ℝ), 2 < y → y < x → F' y > 0) ∧ (∀ (y : ℝ), 0 < y → y < 2 → F' y < 0) :=
sorry

theorem max_min_values (h1 : 0 < 1) (h3 : 3 < 3) :
  max_ (F 1) (F 2) (F 3) = F 3 ∧ min_ (F 1) (F 2) (F 3) = F 2 :=
sorry

end monotonic_intervals_max_min_values_l800_800232


namespace line_a_does_not_intersect_plane_alpha_l800_800042

-- Definitions and conditions
variable (a b : ℝ^3) (α : set (ℝ^3))
variable (parallel : ∀ x ∈ ℝ^3, x ∈ a ∨ x ∉ b)  -- Express the condition a ∥ b
variable (subset_b_alpha : ∀ y ∈ b, y ∈ α)         -- Express the condition b ⊂ α

-- Statement to be proved
theorem line_a_does_not_intersect_plane_alpha :
  (∀ z ∈ a, z ∉ α) := sorry

end line_a_does_not_intersect_plane_alpha_l800_800042


namespace gazelle_top_speed_l800_800113

theorem gazelle_top_speed (cheetah_speed_mph : ℝ) (conversion_factor : ℝ) (catch_up_time_sec : ℕ) (initial_distance_feet : ℕ) 
    (cheetah_speed_fps := cheetah_speed_mph * conversion_factor)
    (cheetah_distance_feet := cheetah_speed_fps * catch_up_time_sec)
    (distance_covered_by_gazelle_feet := cheetah_distance_feet - initial_distance_feet)
    (gazelle_speed_fps := distance_covered_by_gazelle_feet / catch_up_time_sec)
    (gazelle_speed_mph := gazelle_speed_fps * 3600 / 5280) :
  cheetah_speed_mph = 60 → 
  conversion_factor = 1.5 → 
  catch_up_time_sec = 7 →
  initial_distance_feet = 210 →
  gazelle_speed_mph ≈ 40.91 := 
by 
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end gazelle_top_speed_l800_800113


namespace solution_set_of_inequality_l800_800489

theorem solution_set_of_inequality :
  { x : ℝ | -x^2 + 2*x + 3 ≥ 0 } = { x : ℝ | -1 ≤ x ∧ x ≤ 3 } :=
sorry

end solution_set_of_inequality_l800_800489


namespace decagon_perimeter_l800_800906

-- Define conditions
def is_regular_decagon (num_sides : ℕ) := num_sides = 10

def side_lengths (s1 s2 : ℕ) := 
  (s1 = 3 ∧ s2 = 3 ∧ s1 + s2 = 6) ∨ 
  (s1 = 4 ∧ s2 = 4 ∧ s1 + s2 = 8)

-- Theorem statement
theorem decagon_perimeter 
  (num_sides : ℕ) 
  (sides : fin 10 → ℕ)
  (h_num_sides : is_regular_decagon num_sides)
  (h_sides : ∃ (idx_3 : fin 8) (idx_4 : fin 2), 
               (∀ i, i ∈ idx_3 → sides i = 3) ∧ 
               (∀ i, i ∈ idx_4 → sides i = 4)) :
  ∑ i, sides i = 32 :=
  sorry

end decagon_perimeter_l800_800906


namespace martin_total_distance_l800_800408

-- Define the conditions
def total_trip_time : ℕ := 8
def first_half_speed : ℕ := 70
def second_half_speed : ℕ := 85
def half_trip_time : ℕ := total_trip_time / 2

-- Define the total distance traveled 
def total_distance : ℕ := (first_half_speed * half_trip_time) + (second_half_speed * half_trip_time)

-- Statement to prove
theorem martin_total_distance : total_distance = 620 :=
by
  -- This is a placeholder to represent that a proof is needed
  -- Actual proof steps are omitted as instructed
  sorry

end martin_total_distance_l800_800408


namespace type_C_cards_at_least_20_l800_800949

theorem type_C_cards_at_least_20 (x y z : ℕ) (h1 : x + y + z = 150) (h2 : 0.5 * x + y + 2.5 * z = 180) : z ≥ 20 :=
by
  sorry

end type_C_cards_at_least_20_l800_800949


namespace largest_fraction_added_l800_800595

def is_proper_fraction (f : ℚ) : Prop :=
  f.num < f.denom

theorem largest_fraction_added 
  (x : ℚ) 
  (h1 : is_proper_fraction (1 / 6 + x)) 
  (h2 : (1 / 6 + x).denom < 6) : 
  x = 19 / 30 := 
by 
  sorry

end largest_fraction_added_l800_800595


namespace max_hot_dogs_with_300_dollars_l800_800638

def num_hot_dogs (dollars : ℕ) 
  (cost_8 : ℚ) (count_8 : ℕ) 
  (cost_20 : ℚ) (count_20 : ℕ)
  (cost_250 : ℚ) (count_250 : ℕ) : ℕ :=
  sorry

theorem max_hot_dogs_with_300_dollars : 
  num_hot_dogs 300 1.55 8 3.05 20 22.95 250 = 3258 :=
sorry

end max_hot_dogs_with_300_dollars_l800_800638


namespace distance_between_points_eq_l800_800206

noncomputable def dist (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem distance_between_points_eq :
  dist 1 5 7 2 = 3 * Real.sqrt 5 :=
by
  sorry

end distance_between_points_eq_l800_800206


namespace percentage_given_to_close_friends_l800_800377

-- Definitions
def total_boxes : ℕ := 20
def pens_per_box : ℕ := 5
def total_pens : ℕ := total_boxes * pens_per_box
def pens_left_after_classmates : ℕ := 45

-- Proposition
theorem percentage_given_to_close_friends (P : ℝ) :
  total_boxes = 20 → pens_per_box = 5 → pens_left_after_classmates = 45 →
  (3 / 4) * (100 - P) = (pens_left_after_classmates : ℝ) →
  P = 40 :=
by
  intros h_total_boxes h_pens_per_box h_pens_left_after h_eq
  sorry

end percentage_given_to_close_friends_l800_800377


namespace difference_in_price_l800_800491

noncomputable def total_cost : ℝ := 70.93
noncomputable def pants_price : ℝ := 34.00

theorem difference_in_price (total_cost pants_price : ℝ) (h_total : total_cost = 70.93) (h_pants : pants_price = 34.00) :
  (total_cost - pants_price) - pants_price = 2.93 :=
by
  sorry

end difference_in_price_l800_800491


namespace blueberry_picking_l800_800911

-- Define the amounts y1 and y2 as a function of x
variable (x : ℝ)
def y1 : ℝ := 60 + 18 * x
def y2 : ℝ := 150 + 15 * x

-- State the theorem about the relationships given the condition 
theorem blueberry_picking (hx : x > 10) : 
  y1 x = 60 + 18 * x ∧ y2 x = 150 + 15 * x :=
by
  sorry

end blueberry_picking_l800_800911


namespace find_integers_xyz_l800_800200

theorem find_integers_xyz : 
  {p : ℤ × ℤ × ℤ // p.1 + p.2 + p.3 + p.1 * p.2 + p.2 * p.3 + p.3 * p.1 + p.1 * p.2 * p.3 = 2017} =
  { (0, 1, 1008), (0, 1008, 1), (1, 0, 1008), (1, 1008, 0), (1008, 0, 1), (1008, 1, 0) } :=
by
  sorry

end find_integers_xyz_l800_800200


namespace equilateral_triangles_less_bound_l800_800499

theorem equilateral_triangles_less_bound (n k : ℕ) (h1 : n > 3) (h2 : ∃ P : set.point_in_plane, P ⊆ convex_ngon_vertices n) (h3 : ∀ P ∈ P, ∃! T : equilateral_triangle, side_length T = 1 ∧ vertices T ⊆ P) : k < 2 * n / 3 :=
sorry

end equilateral_triangles_less_bound_l800_800499


namespace probability_sum_geq_9_l800_800505

def count_sum_geq_9 : ℕ :=
  (if 3 + 6 ≥ 9 then 1 else 0) + 
  (if 6 + 3 ≥ 9 then 1 else 0) +
  (if 4 + 5 ≥ 9 then 1 else 0) +
  (if 5 + 4 ≥ 9 then 1 else 0) +
  (if 4 + 6 ≥ 9 then 1 else 0) +
  (if 6 + 4 ≥ 9 then 1 else 0) +
  (if 5 + 5 ≥ 9 then 1 else 0) +
  (if 5 + 6 ≥ 9 then 1 else 0) +
  (if 6 + 5 ≥ 9 then 1 else 0) +
  (if 6 + 6 ≥ 9 then 1 else 0)

theorem probability_sum_geq_9 (n : ℕ) (m : ℕ) (p : ℚ) :
  n = 36 → 
  m = 10 → 
  p = 5 / 18 → 
  n = 6 * 6 →
  m = count_sum_geq_9 →
  p = m / n :=
by {
  intros,
  sorry
}

end probability_sum_geq_9_l800_800505


namespace larger_segment_on_side110_l800_800020

theorem larger_segment_on_side110 (a b c : ℕ) (h₁ : a = 40) (h₂ : b = 50) (h₃ : c = 110) (h₄ : a < c) (h₅ : b < c) :
  let shorter_segment := a ^ 2 - (((a ^ 2) - (c - (s := Nat)) ^ 2)) / c 
  let larger_segment := c - shorter_segment in
  larger_segment = 59 := sorry

end larger_segment_on_side110_l800_800020


namespace initial_mean_l800_800864

theorem initial_mean (M : ℝ) (n : ℕ) (observed_wrongly correct_wrongly : ℝ) (new_mean : ℝ) :
  n = 50 ∧ observed_wrongly = 23 ∧ correct_wrongly = 45 ∧ new_mean = 36.5 → M = 36.06 :=
by
  intro h
  cases h with h1 h2
  cases h2 with h3 h4
  cases h4 with h5 h6
  have sum_initial := n * M
  have diff := correct_wrongly - observed_wrongly
  have sum_corrected := sum_initial + diff
  have new_sum_corrected := n * new_mean
  have equation := sum_corrected = new_sum_corrected
  rw [←h3, ←h2, ←h1, ←h6] at equation
  have sum_initial_calculated := new_sum_corrected - diff
  have M_calculated_eq := sum_initial_calculated / n
  rw [←h5, ←h1] at M_calculated_eq
  -- Calculate manually to show final M == 36.06 which is the correct proof
  sorry

end initial_mean_l800_800864


namespace perpendicular_lines_parallel_l800_800623

-- Given conditions
variable (plane : Type) [nonempty plane]
variables (l1 l2 : plane → Prop) -- l1 and l2 are two lines in 'plane'

-- Predicate for line being perpendicular to a plane
def is_perpendicular_to_plane (l : plane → Prop) (p : plane) : Prop :=
  ∀ (v1 v2 : plane), v1 ≠ v2 → l v1 ∧ l v2 → ⟪v1, v2⟫ = 0

-- Definition of parallel lines
def are_parallel (l1 l2 : plane → Prop) : Prop :=
  ∃ (v1 v2 : plane), l1 v1 ∧ l2 v2 ∧ ∃ k : ℝ, v2 = k • v1

-- The theorem to be proved
theorem perpendicular_lines_parallel (p : plane)
  (h1 : is_perpendicular_to_plane l1 p)
  (h2 : is_perpendicular_to_plane l2 p)
  : are_parallel l1 l2 :=
sorry

end perpendicular_lines_parallel_l800_800623


namespace sum_of_coefficients_equals_28_l800_800645

def P (x : ℝ) : ℝ :=
  2 * (4 * x^8 - 5 * x^5 + 9 * x^3 - 6) + 8 * (x^6 - 4 * x^3 + 6)

theorem sum_of_coefficients_equals_28 : P 1 = 28 := by
  sorry

end sum_of_coefficients_equals_28_l800_800645


namespace sales_not_paint_or_brushes_l800_800481

theorem sales_not_paint_or_brushes (p b : ℝ) (h1 : p = 38) (h2 : b = 22) :
  let s := 100 - (p + b) in s = 40 := 
by
  sorry

end sales_not_paint_or_brushes_l800_800481


namespace general_term_sum_first_n_terms_l800_800285

-- Define the sequence using the given recursive relation and initial condition
def a : ℕ → ℕ
| 1       := 1
| (n + 1) := 2 * (a n) + 1

-- Prove the general term formula
theorem general_term (n : ℕ) (hn : n ≥ 1) : a n = 2^n - 1 := by
  induction n using Nat.strong_induction_on with n ih
  cases n with
  | zero => 
    -- Cases n < 1
    have: 0 < 1:= by norm_num
    contradiction
  | succ n =>
    cases n with
    | zero =>
      simp [a] -- a(1) = 1
      have : 2 ^ 1 - 1 = 1 := by norm_num
      assumption
    | succ m =>
      -- Use the hypothesis of induction
      have h: m.succ ≥ 1:= by norm_num
      calc
        a m.succ := 2 * (a m.succ.pred) + 1 := by simp [a]
               ... := 2 * (2^m.succ - 1) + 1 := by rw ih _ (Nat.lt_succ_self m.succ)
               ... :=  2^m.succ.succ - 1 := by ring

-- Prove the sum of the first n terms
theorem sum_first_n_terms (n : ℕ) : (∑ i in Finset.range n, a (i + 1)) = 2^(n + 1) - 2 - n := by
  induction n with
  | zero => simp -- S0 = 0
  | succ m ih =>
    simp only [Finset.sum_range_succ, ih, a]
    have hp: 1 < 2:= by norm_num
    rw [general_term (m + 1) hp,← pow_succ]
    ring

end general_term_sum_first_n_terms_l800_800285


namespace math_olympiad_proof_l800_800351

theorem math_olympiad_proof (scores : Fin 20 → ℕ) 
  (h_diff : ∀ i j, i ≠ j → scores i ≠ scores j) 
  (h_sum : ∀ i j k, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k) : 
  ∀ i, scores i > 18 :=
by
  sorry

end math_olympiad_proof_l800_800351


namespace count_divisible_by_4_3_5_l800_800744

theorem count_divisible_by_4_3_5 : 
  let count := (List.range' 1 300).filter (λ n, n % 4 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0),
  count.length = 4 := 
by {
  let count := (List.range' 1 300).filter (λ n, n % 4 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0),
  have h : count.length = 4 := sorry,
  exact h,
}

end count_divisible_by_4_3_5_l800_800744


namespace good_coloring_exists_l800_800902

-- Define what constitutes a "good" coloring of an m x n table.
def good_coloring (m n : ℕ) (coloring : (ℕ × ℕ) → ℕ) : Prop :=
  (m ≥ 5) ∧ (n ≥ 5) ∧ 
  -- condition 1: each cell has the same number of neighboring cells of the two other colors
  (∀ i j, (1 ≤ i ∧ i ≤ m) ∧ (1 ≤ j ∧ j ≤ n) → 
    ∀ c, c ≠ coloring (i, j) → 
      ((if i > 1 then (coloring (i-1, j) = c) else 0) +
       (if i < m then (coloring (i+1, j) = c) else 0) +
       (if j > 1 then (coloring (i, j-1) = c) else 0) +
       (if j < n then (coloring (i, j+1) = c) else 0) = 2)) ∧
  -- condition 2: each corner has no neighboring cells of its color
  (∀ (i, j) ∈ [(1,1), (1,n), (m,1), (m,n)], 
    (if i > 1 then coloring (i-1, j) ≠ coloring (i, j) else true) ∧
    (if i < m then coloring (i+1, j) ≠ coloring (i, j) else true) ∧
    (if j > 1 then coloring (i, j-1) ≠ coloring (i, j) else true) ∧
    (if j < n then coloring (i, j+1) ≠ coloring (i, j) else true))

theorem good_coloring_exists (m n : ℕ) (h_m : m ≥ 5) (h_n : n ≥ 5) :
  (∃ coloring : (ℕ × ℕ) → ℕ, good_coloring m n coloring) ↔
  (2 ∣ m ∧ 3 ∣ n) ∨ (2 ∣ n ∧ 3 ∣ m) :=
by
  sorry  -- this is where the proof would go

end good_coloring_exists_l800_800902


namespace circle_arc_and_circumference_l800_800171

theorem circle_arc_and_circumference (C_X : ℝ) (θ_YOZ : ℝ) (C_D : ℝ) (r_X r_D : ℝ) :
  C_X = 100 ∧ θ_YOZ = 150 ∧ r_X = 50 / π ∧ r_D = 25 / π ∧ C_D = 50 →
  (θ_YOZ / 360) * C_X = 500 / 12 ∧ 2 * π * r_D = C_D :=
by sorry

end circle_arc_and_circumference_l800_800171


namespace median_is_8_l800_800333

def xiao_jun_scores : List ℕ := [6, 7, 7, 7, 8, 8, 9, 9, 9, 10]

theorem median_is_8 (scores : List ℕ) (h : scores = xiao_jun_scores) : 
  List.median scores = 8 := 
by 
  sorry

end median_is_8_l800_800333


namespace quadrilateral_prism_volume_l800_800018

-- Defining the conditions and the target volume
variables {a α : ℝ}

-- The volume of the prism given the conditions
theorem quadrilateral_prism_volume (h1 : a > 0) (h2 : 0 < α ∧ α < π) :
  let V_p := (a^3 * (sqrt (2 * cos α))) / (2 * sin (α / 2))
  in V_p = (a^3 * (sqrt (2 * cos α))) / (2 * sin (α / 2)) :=
by
  sorry

end quadrilateral_prism_volume_l800_800018


namespace minimize_M_coordinates_l800_800260

-- Define point and coordinate system
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the coordinates of A and F
def A : Point := ⟨3, 2⟩
def F : Point := ⟨1/2, 0⟩

-- Define the parabola condition
def on_parabola (M : Point) : Prop :=
  M.y^2 = 2 * M.x

-- Define the distance function
def distance (P Q : Point) : ℝ :=
  real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

-- Define the minimization condition
def minimize_distance (M : Point) : Prop :=
  ∀N : Point, on_parabola N →
    distance M A + distance M F ≤
    distance N A + distance N F

-- Main theorem statement
theorem minimize_M_coordinates :
  ∃ (M : Point), on_parabola M ∧ minimize_distance M ∧ M = ⟨2, 2⟩ :=
begin
  sorry
end

end minimize_M_coordinates_l800_800260


namespace vojta_problem_proof_l800_800514

def is_palindrome (n : ℕ) : Prop :=
  -- check if n is a palindrome
  let s := n.digits 10 in s = s.reverse

def count_palindromic_numbers (seq : list ℕ) (n : ℕ) : ℕ :=
  (list.sublists' seq).filter (λ l, l.length = n ∧ is_palindrome (nat.of_digits 10 l)).length

def infinite_sequence (n : ℕ) : list ℕ :=
  list.replicate n 2 ++ list.replicate n 0 ++ list.replicate n 1 ++ list.replicate n 0

theorem vojta_problem_proof :
  let sequence := (list.replicate 100 2010).bind (λ n, [2, 0, 1, 0]) in
  count_palindromic_numbers sequence 4 = 0 ∧
  count_palindromic_numbers sequence 5 = 198 :=
by
  sorry

end vojta_problem_proof_l800_800514


namespace part1_part2_l800_800366

-- Define the conditions
def A_n (n : ℕ) (hn : 0 < n) : ℝ × ℝ := (0, 1 / n)
def B_n (n : ℕ) (hn : 0 < n) : ℝ × ℝ := let bn := 1 / (sqrt 3 * n) in (bn, sqrt 2 * bn)

-- Define the theorem to prove part 1
theorem part1 (n : ℕ) (hn : 0 < n) (hn1 : 0 < n + 1) :
  let an := (B_n n hn).1 * ((sqrt 3 * (B_n n hn).1 - 1/n) / (sqrt 3 * (B_n n hn).1 - sqrt 2 * (B_n n hn).1)) in
  let an1 := (B_n (n + 1) hn1).1 * ((sqrt 3 * (B_n (n + 1) hn1).1 - 1/(n + 1)) / (sqrt 3 * (B_n (n + 1) hn1).1 - sqrt 2 * (B_n (n + 1) hn1).1)) in
  an > an1 ∧ an1 > 4 :=
sorry

-- Define the theorem to prove part 2
theorem part2 : ∃ n0 : ℕ, 0 < n0 ∧ ∀ n : ℕ, n0 < n →
  ∑ i in (finset.range n).filter (λ k, 0 < k), (let b (m : ℕ) := 1 / (sqrt 3 * m) in b (i + 1) / b i) < n - 2004 :=
sorry

end part1_part2_l800_800366


namespace count_valid_polynomials_l800_800649

theorem count_valid_polynomials :
  ∃ n : ℕ, ∀ (b₀ b₁ b₂ b₃ b₄ b₅ b₆ : ℕ),
    b₀ ∈ ({0, 1, 2} : set ℕ) →
    b₁ ∈ ({0, 1, 2} : set ℕ) →
    b₂ ∈ ({0, 1, 2} : set ℕ) →
    b₃ ∈ ({0, 1, 2} : set ℕ) →
    b₄ ∈ ({0, 1, 2} : set ℕ) →
    b₅ ∈ ({0, 1, 2} : set ℕ) →
    b₆ ∈ ({0, 1, 2} : set ℕ) →
    let p := 1 + b₆ + b₅ + b₄ + b₃ + b₂ + b₁ + b₀,
        q := -126 + 64 * b₆ - 32 * b₅ + 16 * b₄ - 8 * b₃ + 4 * b₂ - 2 * b₁ + b₀ in
    (p = 0 ∧ q = 0) → n = ∃ (count : ℕ), count ∈ {1, 2, 3, ...}
:= sorry

end count_valid_polynomials_l800_800649


namespace students_scoring_120_or_more_l800_800780

noncomputable def num_students_over_120 (total_students : ℕ) (mean : ℝ) (std_dev_square : ℝ) 
  (prop_80_to_100 : ℝ) : ℕ :=
  if total_students = 1200 ∧ mean = 100 ∧ prop_80_to_100 = 1/3 
  then (1 / 6) * total_students
  else 0

theorem students_scoring_120_or_more 
  (total_students : ℕ)
  (mean : ℝ)
  (std_dev_square : ℝ)
  (prop_80_to_100 : ℝ)
  (h1 : total_students = 1200)
  (h2 : mean = 100)
  (h3 : prop_80_to_100 = 1 / 3) :
  num_students_over_120 total_students mean std_dev_square prop_80_to_100 = 200 :=
begin
  sorry,
end

end students_scoring_120_or_more_l800_800780


namespace dot_product_identity_l800_800441

variable {V : Type*} [inner_product_space ℝ V]

variables (a b c : V)
variables (h1 : inner_product a b = 5)
variables (h2 : inner_product a c = -2)
variables (h3 : inner_product b c = 9)

theorem dot_product_identity : inner_product b (5 • c - 3 • a) = 30 :=
by
  sorry

end dot_product_identity_l800_800441


namespace rhombus_perimeter_l800_800467

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 20) (h2 : d2 = 16) : ∃ p, p = 8 * Real.sqrt 41 := by
  let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  have h3 : s = 2 * Real.sqrt 41 := by sorry
  let p := 4 * s
  have h4 : p = 8 * Real.sqrt 41 := by sorry
  exact ⟨p, h4⟩

end rhombus_perimeter_l800_800467


namespace calculate_leakage_rate_l800_800803

variable (B : ℕ) (T : ℕ) (R : ℝ)

-- B represents the bucket's capacity in ounces, T represents time in hours, R represents the rate of leakage per hour in ounces per hour.

def leakage_rate (B : ℕ) (T : ℕ) (R : ℝ) : Prop :=
  (B = 36) ∧ (T = 12) ∧ (B / 2 = T * R)

theorem calculate_leakage_rate : leakage_rate 36 12 1.5 :=
by 
  simp [leakage_rate]
  sorry

end calculate_leakage_rate_l800_800803


namespace minimum_colors_2016_board_l800_800549

theorem minimum_colors_2016_board : ∃ k : ℕ, ∀ (board : Array (Array ℕ)), (∀ i j : ℕ, i < 2016 → j < 2016 → board[i][j] ∈ Fin k) ∧
(board[0][0] = 1) ∧
(∀ i : ℕ, i < 2016 → board[i][i] = 1) ∧
(∀ i j : ℕ, i < j → i < 2016 → j < 2016 → board[i][j] = board[j][i]) ∧
(∀ i j : ℕ, i < 2016 → j < 2016 → i ≠ j → board[i][j] ≠ board[i][2015-i]) → 
k = 11 :=
by
  sorry

end minimum_colors_2016_board_l800_800549


namespace sufficient_but_not_necessary_condition_l800_800229

theorem sufficient_but_not_necessary_condition
  (a b : ℝ) : (a - b) * a^2 < 0 → a < b :=
sorry

end sufficient_but_not_necessary_condition_l800_800229


namespace binary_representation_more_zeros_l800_800129

theorem binary_representation_more_zeros (n s : ℕ) 
  (hn : ∃ seq : list ℕ, (∀ x ∈ seq, Nat.binary_length x = 2013 ∧ (Nat.bit_count 0 x > Nat.bit_count 1 x)) ∧ seq.length = n ∧ seq.sum = s) :
  Nat.bit_count 0 (n + s) > Nat.bit_count 1 (n + s) :=
sorry

end binary_representation_more_zeros_l800_800129


namespace sum_of_possible_values_of_N_l800_800483

theorem sum_of_possible_values_of_N (N : ℤ) : 
  (N * (N - 8) = 16) -> (∃ a b, N^2 - 8 * N - 16 = 0 ∧ (a + b = 8)) :=
sorry

end sum_of_possible_values_of_N_l800_800483


namespace vertex_of_quadratic1_vertex_of_quadratic2_l800_800977

theorem vertex_of_quadratic1 :
  ∃ x y : ℝ, 
  (∀ x', 2 * x'^2 - 4 * x' - 1 = 2 * (x' - x)^2 + y) ∧ 
  (x = 1 ∧ y = -3) :=
by sorry

theorem vertex_of_quadratic2 :
  ∃ x y : ℝ, 
  (∀ x', -3 * x'^2 + 6 * x' - 2 = -3 * (x' - x)^2 + y) ∧ 
  (x = 1 ∧ y = 1) :=
by sorry

end vertex_of_quadratic1_vertex_of_quadratic2_l800_800977


namespace deer_meat_distribution_l800_800786

theorem deer_meat_distribution (a d : ℕ) (H1 : a = 100) :
  ∀ (Dafu Bugeng Zanbao Shangzao Gongshe : ℕ),
    Dafu = a - 2 * d →
    Bugeng = a - d →
    Zanbao = a →
    Shangzao = a + d →
    Gongshe = a + 2 * d →
    Dafu + Bugeng + Zanbao + Shangzao + Gongshe = 500 →
    Bugeng + Zanbao + Shangzao = 300 :=
by
  intros Dafu Bugeng Zanbao Shangzao Gongshe hDafu hBugeng hZanbao hShangzao hGongshe hSum
  sorry

end deer_meat_distribution_l800_800786


namespace evaluate_A_minus10_3_l800_800176

def A (x : ℝ) (m : ℕ) : ℝ :=
  if m = 0 then 1 else x * A (x - 1) (m - 1)

theorem evaluate_A_minus10_3 : A (-10) 3 = 1320 := 
  sorry

end evaluate_A_minus10_3_l800_800176


namespace number_of_elements_in_B_l800_800383

def A : Set ℕ := { x | ∃ (n : ℤ), x = n^2 }

def f (x : ℕ) : ℕ := x % 5

theorem number_of_elements_in_B : (∃ (B : Set ℕ), (∀ y ∈ B, ∃ x ∈ A, f x = y) ∧ B.card = 3) :=
sorry

end number_of_elements_in_B_l800_800383


namespace garrison_reinforcement_l800_800596

theorem garrison_reinforcement (x : ℕ) (h1 : ∀ (n m p : ℕ), n * m = p → x = n - m) :
  (150 * (31 - x) = 450 * 5) → x = 16 :=
by sorry

end garrison_reinforcement_l800_800596


namespace proof_problem_l800_800252

-- Define the probabilities
def P (X : ℕ → ℝ) (n : ℕ) : ℝ := X n

-- Given conditions
def m : ℝ := 0.3
def total_prob (P : ℕ → ℝ) : Prop := P 1 + P 2 + P 3 = 1

-- Probabilities distribution of X
def P_dist : ℕ → ℝ
| 1 := 0.3
| 2 := 0.3
| 3 := 0.1 + 0.3
| _ := 0

-- Expected value definition
def E_X (P : ℕ → ℝ) : ℝ := P 1 * 1 + P 2 * 2 + P 3 * 3

-- Proof problem
theorem proof_problem (P : ℕ → ℝ) (m : ℝ) (h : total_prob P) : 
  m = 0.3 ∧ E_X P = 2.1 := by
  sorry

end proof_problem_l800_800252


namespace prob_log2_between_0_and_2_l800_800432

noncomputable def geometric_probability_log2 (x : ℝ) : ℝ :=
if x ∈ set.Icc 1 4 then 1 / 6 else 0

theorem prob_log2_between_0_and_2 :
  ∫ x in set.Icc 0 6, geometric_probability_log2 x = 1 / 2 := by
sorry

end prob_log2_between_0_and_2_l800_800432


namespace investment_worth_l800_800841

noncomputable def initial_investment (total_earning : ℤ) : ℤ := total_earning / 2

noncomputable def current_worth (initial_investment total_earning : ℤ) : ℤ :=
  initial_investment + total_earning

theorem investment_worth (monthly_earning : ℤ) (months : ℤ) (earnings : ℤ)
  (h1 : monthly_earning * months = earnings)
  (h2 : earnings = 2 * initial_investment earnings) :
  current_worth (initial_investment earnings) earnings = 90 := 
by
  -- We proceed to show the current worth is $90
  -- Proof will be constructed here
  sorry
  
end investment_worth_l800_800841


namespace PQN_collinear_l800_800824

open_locale classical

variables {A B C M N P Q : Type*} [plane_geometry A B C M N P Q]

/-- Let M and N be the midpoints of the hypotenuse AB and the leg BC of the right triangle ABC, respectively.
    The excircle of triangle ACM touches side AM at point Q and line AC at point P.
    Prove that points P, Q, and N are collinear. -/
theorem PQN_collinear (h1 : midpoint A B M) 
                     (h2 : midpoint B C N) 
                     (h3 : excircle_touches ACM AM Q) 
                     (h4 : excircle_touches ACM AC P) : 
  collinear P Q N :=
sorry

end PQN_collinear_l800_800824


namespace largest_four_digit_number_prop_l800_800048

theorem largest_four_digit_number_prop :
  ∃ (a b c d : ℕ), a = 9 ∧ b = 0 ∧ c = 9 ∧ d = 9 ∧ (1000 * a + 100 * b + 10 * c + d = 9099) ∧ (c = a + b) ∧ (d = b + c) :=
by
  sorry

end largest_four_digit_number_prop_l800_800048


namespace sin_cos_sum_l800_800246

theorem sin_cos_sum (α : ℝ) (h1 : Real.sin (2 * α) = 3 / 4) (h2 : π < α) (h3 : α < 3 * π / 2) : 
  Real.sin α + Real.cos α = -Real.sqrt (7 / 4) := 
by
  sorry

end sin_cos_sum_l800_800246


namespace delta_value_l800_800749

theorem delta_value (Δ : ℤ) (h : 4 * (-3) = Δ - 3) : Δ = -9 :=
by {
  sorry
}

end delta_value_l800_800749


namespace playerB_hit_rate_playerA_probability_l800_800040

theorem playerB_hit_rate (p : ℝ) (h : (1 - p)^2 = 1/16) : p = 3/4 :=
sorry

theorem playerA_probability (hit_rate : ℝ) (h : hit_rate = 1/2) : 
  (1 - (1 - hit_rate)^2) = 3/4 :=
sorry

end playerB_hit_rate_playerA_probability_l800_800040


namespace Richard_walked_10_miles_third_day_l800_800832

def distance_to_NYC := 70
def day1 := 20
def day2 := (day1 / 2) - 6
def remaining_distance := 36
def day3 := 70 - (day1 + day2 + remaining_distance)

theorem Richard_walked_10_miles_third_day (h : day3 = 10) : day3 = 10 :=
by {
    sorry
}

end Richard_walked_10_miles_third_day_l800_800832


namespace remainder_of_power_mod_l800_800996

theorem remainder_of_power_mod :
  (5^2023) % 11 = 4 :=
  by
    sorry

end remainder_of_power_mod_l800_800996


namespace find_total_profit_l800_800619

variables (A B C total_investment profit : ℝ)

-- Given conditions
def total_investment_condition : Prop := A + B + C = 90000
def condition_A_B : Prop := A = B + 6000
def condition_B_C : Prop := B = C - 3000
def condition_A_share : Prop := (A / (A + B + C)) * profit = 3168

-- The proof goal
theorem find_total_profit (h1 : total_investment = 90000)
    (h2 : total_investment_condition)
    (h3 : condition_A_B)
    (h4 : condition_B_C)
    (h5 : condition_A_share) : profit = 8640 :=
sorry

end find_total_profit_l800_800619


namespace daughter_name_l800_800510

theorem daughter_name (c1 c2: String) (H: c2 = c1) (hConv: c1 = "Nina".succ):
  c2 = "Nina" :=
by
  sorry

end daughter_name_l800_800510


namespace int_less_than_sqrt_23_l800_800424

theorem int_less_than_sqrt_23 : ∃ (n : ℤ), n < Real.sqrt 23 := by
  use 4
  have h : (4 : ℝ) < Real.sqrt 23 := by
    rw Real.sqrt_lt'_iff
    exact ⟨dec_trivial, dec_trivial⟩
  exact_mod_cast h

end int_less_than_sqrt_23_l800_800424


namespace scores_greater_than_18_l800_800347

theorem scores_greater_than_18 (scores : Fin 20 → ℝ) 
  (h_unique : Function.Injective scores)
  (h_sum : ∀ i j k : Fin 20, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k) :
  ∀ i : Fin 20, scores i > 18 := 
by
  sorry

end scores_greater_than_18_l800_800347


namespace painting_problem_l800_800157

theorem painting_problem (t : ℝ) (h1 : 0 < t)
  (AndyRate : ℝ := 1 / 4)
  (BobRate : ℝ := 1 / 6)
  (CombinedRate : ℝ := AndyRate + BobRate)
  (actual_working_time := t - 2) :
  (CombinedRate * actual_working_time = 1) ↔ (CombinedRate * (t - 2) = 1) :=
by
  have h2 : CombinedRate = 5 / 12 := by norm_num
  rw [←h2]
  simp
  sorry

end painting_problem_l800_800157


namespace third_smallest_digit_number_l800_800913

theorem third_smallest_digit_number : 
  ∃ n : ℕ, (n < 1000) ∧ (n > 99) ∧ (n ≠ 100 * 4 + 10 * 0 + 9) ∧ (n ≠ 100 * 4 + 10 * 0 + 8) ∧ (n = 100 * 4 + 10 * 8 + 0) ∧ 
             (∀ x y z: ℕ, (x ∈ {0, 4, 8, 9} ∧ y ∈ {0, 4, 8, 9} ∧ z ∈ {0, 4, 8, 9} ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z) → 
             (100 * x + 10 * y + z) ≠ n) :=
by
  sorry

end third_smallest_digit_number_l800_800913


namespace fifth_number_in_10th_row_l800_800482

theorem fifth_number_in_10th_row : 
  ∀ (n : ℕ), (∃ (a : ℕ), ∀ (m : ℕ), 1 ≤ m ∧ m ≤ 10 → (m = 10 → a = 67)) :=
by
  sorry

end fifth_number_in_10th_row_l800_800482


namespace value_of_sum_of_squares_l800_800734

noncomputable def is_value_of_sum_of_squares (a b : ℝ) : Prop :=
  (a + b)^2 = 1 ∨ (a + b)^2 = 25

theorem value_of_sum_of_squares (a b : ℝ) (h1 : |a| = 3) (h2 : |b| = 2) (h3 : a < b) : 
  is_value_of_sum_of_squares a b :=
begin
  sorry,
end

end value_of_sum_of_squares_l800_800734


namespace ball_reaches_less_than_2_feet_after_14_bounces_l800_800929

theorem ball_reaches_less_than_2_feet_after_14_bounces :
  ∀ (h₀ : ℝ) (r : ℝ), h₀ = 500 → r = 2 / 3 →
  ∃ (k : ℕ), k = 14 ∧ h₀ * r^k < 2 := by
  intros h₀ r h₀_eq r_eq
  use 14
  rw [h₀_eq, r_eq]
  norm_num
  apply lt_trans
    (norm_num [500 * (2 / 3)^14])
  norm_num [2]
  sorry -- Proof for the exact value comparison

end ball_reaches_less_than_2_feet_after_14_bounces_l800_800929


namespace quadratic_to_vertex_form_l800_800650

theorem quadratic_to_vertex_form:
  ∀ (x : ℝ), (x^2 - 4 * x + 3 = (x - 2)^2 - 1) :=
by
  sorry

end quadratic_to_vertex_form_l800_800650


namespace alice_trajectory_length_is_correct_l800_800957

noncomputable def length_of_trajectory 
  (radius₀ : ℝ) (ratio : ℝ) (spin_rate : ℝ) (initial_time : ℝ) (time_elapsed : ℝ) : ℝ :=
  sorry -- We are skipping the complete implementation of this function

theorem alice_trajectory_length_is_correct 
  (radius₀ : ℝ := 5) (ratio : ℝ := 2/3) (spin_rate : ℝ := (2/3) * real.pi / 6) 
  (initial_time : ℝ := 0) (time_elapsed : ℝ := 12) :
  length_of_trajectory radius₀ ratio spin_rate initial_time time_elapsed = 18 * real.pi :=
begin
  sorry
end

end alice_trajectory_length_is_correct_l800_800957


namespace largest_valid_number_l800_800078

-- Define the conditions for the digits of the number
def valid_digits (a b c d : ℕ) : Prop :=
  c = a + b ∧ d = b + c

-- Prove that the number formed by digits 9, 0, 9, 9 is the largest valid 4-digit number
theorem largest_valid_number : ∃ a b c d, valid_digits a b c d ∧
  (a * 1000 + b * 100 + c * 10 + d = 9099) :=
begin
  use [9, 0, 9, 9],
  split,
  { -- Proof of valid digits condition
    split; refl },
  { -- Proof that the number is 9099
    refl }
end

end largest_valid_number_l800_800078


namespace power_function_at_16_l800_800253

theorem power_function_at_16 :
  ∃ (α : ℝ), (∀ x : ℝ, f x = x ^ α) ∧ (f 4 = 2) → (f 16 = 4) :=
by
  sorry

end power_function_at_16_l800_800253


namespace equation_of_line_l_equations_of_line_m_l800_800239

noncomputable def point_P : ℝ × ℝ := (-2, 5)
def slope_l : ℝ := -3/4

theorem equation_of_line_l :
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ a * -2 + b * 5 + c = 0 ∧ b ≠ 0 ∧ slope_l = -(a/b) ∧ 3*a + 4*b + c = 0 :=
sorry

def distance_between_lines := 3

theorem equations_of_line_m :
  ∃ (a b c1 c2 : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ a = 3 ∧ b = 4 ∧ (∀ c, c = c1 ∨ c = c2) ∧
  (c1 = 1 ∧ c2 = -29) :=
sorry

end equation_of_line_l_equations_of_line_m_l800_800239


namespace percentage_profit_first_bicycle_l800_800831

theorem percentage_profit_first_bicycle :
  ∃ (C1 C2 : ℝ), 
    (C1 + C2 = 1980) ∧ 
    (0.9 * C2 = 990) ∧ 
    (12.5 / 100 * C1 = (990 - C1) / C1 * 100) :=
by
  sorry

end percentage_profit_first_bicycle_l800_800831


namespace closest_point_to_A_in_plane_l800_800991

def point_in_plane_closest_to (A : ℝ × ℝ × ℝ) (a b c d : ℝ) : ℝ × ℝ × ℝ :=
  let t := 23 / 29 in
  (2 + 2 * t, -1 + 4 * t, 4 - 3 * t)

theorem closest_point_to_A_in_plane :
  let A := (2, -1, 4) in
  let plane_normal := (2, 4, -3) in
  let plane_const := 15 in
  point_in_plane_closest_to A 2 4 -3 15 = (78 / 29, 68 / 29, -19 / 29) :=
by
  sorry

end closest_point_to_A_in_plane_l800_800991


namespace simplify_vector_eq_l800_800840

-- Define points A, B, C
variables {A B C O : Type} [AddGroup A]

-- Define vector operations corresponding to overrightarrow.
variables (AB OC OB AC AO BO : A)

-- Conditions in Lean definitions
-- Assuming properties like vector addition and subtraction, and associative properties
def vector_eq : Prop := AB + OC - OB = AC

theorem simplify_vector_eq :
  AB + OC - OB = AC :=
by
  -- Proof steps go here
  sorry

end simplify_vector_eq_l800_800840


namespace exists_non_periodic_function_from_functional_eq_all_functions_periodic_from_functional_eq_l800_800561

-- Part (a)
theorem exists_non_periodic_function_from_functional_eq :
  ∃ (f : ℝ → ℝ), (∀ x : ℝ, f(x - 1) + f(x + 1) = sqrt 5 * f(x)) ∧ ¬∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f(x + T) = f(x) := 
sorry

-- Part (b)
theorem all_functions_periodic_from_functional_eq :
  ∀ (g : ℝ → ℝ), (∀ x : ℝ, g(x - 1) + g(x + 1) = sqrt 3 * g(x)) → ∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, g(x + T) = g(x) := 
sorry

end exists_non_periodic_function_from_functional_eq_all_functions_periodic_from_functional_eq_l800_800561


namespace initial_mean_of_observations_l800_800861

-- Definitions of the given conditions and proof of the correct initial mean
theorem initial_mean_of_observations 
  (M : ℝ) -- Mean of 50 observations
  (initial_sum := 50 * M) -- Initial sum of observations
  (wrong_observation : ℝ := 23) -- Wrong observation
  (correct_observation : ℝ := 45) -- Correct observation
  (understated_by := correct_observation - wrong_observation) -- Amount of understatement
  (correct_sum := initial_sum + understated_by) -- Corrected sum
  (corrected_mean : ℝ := 36.5) -- Corrected new mean
  (eq1 : correct_sum = 50 * corrected_mean) -- Equation from condition of corrected mean
  (eq2 : initial_sum = 50 * corrected_mean - understated_by) -- Restating in terms of initial sum
  : M = 36.06 := -- The initial mean of observations
  sorry -- Proof omitted

end initial_mean_of_observations_l800_800861


namespace looms_employed_l800_800134

def sales_value := 500000
def manufacturing_expenses := 150000
def establishment_charges := 75000
def profit_decrease := 5000

def profit_per_loom (L : ℕ) : ℕ := (sales_value / L) - (manufacturing_expenses / L)

theorem looms_employed (L : ℕ) (h : profit_per_loom L = profit_decrease) : L = 70 :=
by
  have h_eq : profit_per_loom L = (sales_value - manufacturing_expenses) / L := by
    sorry
  have profit_expression : profit_per_loom L = profit_decrease := by
    sorry
  have L_value : L = (sales_value - manufacturing_expenses) / profit_decrease := by
    sorry
  have L_is_70 : L = 70 := by
    sorry
  exact L_is_70

end looms_employed_l800_800134


namespace rhombus_perimeter_l800_800462

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 20) (h2 : d2 = 16) : 
  let side := (real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)) in 
  4 * side = 8 * real.sqrt 41 := by
  sorry

end rhombus_perimeter_l800_800462


namespace sum_of_solutions_l800_800085

theorem sum_of_solutions :
  let congruence := ∀ x : ℕ, 7 * (5 * x - 3) % 10 = 35 % 10
  let less_than_30 := ∀ x : ℕ, x ≤ 30 
  ∃! xs : List ℕ, (∀ x ∈ xs, congruence x ∧ less_than_30 x) ∧ xs.sum = 225 :=
by
  sorry

end sum_of_solutions_l800_800085


namespace part_a_part_b_l800_800404

-- Conditions for both parts
variables {A1 A2 A3 B1 B2 B3 P1 P2 P3 : Type} [geometry A1 A2 A3] [geometry B1 B2 B3] [line_intersections P1 P2 P3]

-- Part (a)
theorem part_a
  (circ1 : circumcircle (A1, A2, P3))
  (circ2 : circumcircle (A1, A3, P2))
  (circ3 : circumcircle (A2, A3, P1))
  (similarity_circle : similar_circle (A1, B1) (A2, B2) (A3, B3)) :
  ∃ V, ∀ v ∈ circ1, v ∈ circ2 ∧ v ∈ circ3 ∧ v ∈ similarity_circle := sorry

-- Part (b)
theorem part_b
  (O1 : center_of_rotational_homothety (A2, B2) (A3, B3))
  (O2 : center_of_rotational_homothety (A1, B1) (A3, B3))
  (O3 : center_of_rotational_homothety (A1, B1) (A2, B2))
  (similarity_circle : similar_circle (A1, B1) (A2, B2) (A3, B3)) :
  ∃ U, U ∈ inter (P1, O1) (P2, O2) (P3, O3) ∧ U ∈ similarity_circle :=
  sorry

end part_a_part_b_l800_800404


namespace log_equation_exponent_equation_l800_800972

-- Question 1: Translating the problem statement into Lean
theorem log_equation : 
  (log 2 24) + (log 10 (1 / 2)) - (log 3 (√27)) + (log 10 2) - (log 2 3) = 3 / 2 := 
by 
  sorry

-- Question 2: Translating the problem statement into Lean
theorem exponent_equation : 
  ((33 * real.sqrt 2) ^ 6) - (1 / 9) ^ (-3 / 2) - (-8) ^ 0 = 44 := 
by 
  sorry

end log_equation_exponent_equation_l800_800972


namespace combination_9_8_l800_800173

theorem combination_9_8 : nat.choose 9 8 = 9 := by
  sorry

end combination_9_8_l800_800173


namespace minimum_instantaneous_rate_of_change_l800_800616

def temperature (x : ℝ) : ℝ := (1 / 3) * x^3 - x^2 + 8

theorem minimum_instantaneous_rate_of_change : 
  ∃ x ∈ set.Icc (0:ℝ) 5, (deriv temperature x) = -1 :=
by
  sorry

end minimum_instantaneous_rate_of_change_l800_800616


namespace problem1_problem2_l800_800648

theorem problem1 : -3 + (-2) * 5 - (-3) = -10 :=
by
  sorry

theorem problem2 : -1^4 + ((-5)^2 - 3) / |(-2)| = 10 :=
by
  sorry

end problem1_problem2_l800_800648


namespace ant_returns_to_C_after_6_minutes_l800_800151

noncomputable def ant_probability : ℚ := 1 / 262144

theorem ant_returns_to_C_after_6_minutes :
  let C : char := 'C'
  let grid := list.ofFn (λ i, list.ofFn (λ j, abs (i - j)))
  let neighbors := 8
  let moves := 6
  let symmetry : bool := true in
  (probability_after_n_moves grid moves C symmetry) = ant_probability := sorry

end ant_returns_to_C_after_6_minutes_l800_800151


namespace distance_ratio_l800_800170

def initial_speed_A_kmh := 70
def initial_speed_B_kmh := 35
def acceleration_A := 3 -- m/s²
def acceleration_B := 1.5 -- m/s²
def travel_time_hours := 10

def kmh_to_ms (v: ℝ) : ℝ := v / 3.6
def seconds_in_hour := 3600
def travel_time_seconds := travel_time_hours * seconds_in_hour

def initial_speed_A := kmh_to_ms initial_speed_A_kmh
def initial_speed_B := kmh_to_ms initial_speed_B_kmh

def distance (v₀: ℝ) (a: ℝ) (t: ℝ) : ℝ := v₀ * t + 0.5 * a * t^2

def distance_A := distance initial_speed_A acceleration_A travel_time_seconds
def distance_B := distance initial_speed_B acceleration_B travel_time_seconds

def ratio (d_A: ℝ) (d_B: ℝ) : ℝ := d_A / d_B

theorem distance_ratio : ratio distance_A distance_B = 2 := by
  -- proof will be provided here
  sorry

end distance_ratio_l800_800170


namespace bouquets_sold_on_Monday_l800_800594

theorem bouquets_sold_on_Monday
  (tuesday_three_times_monday : ∀ (x : ℕ), bouquets_sold_Tuesday = 3 * x)
  (wednesday_third_of_tuesday : ∀ (bouquets_sold_Tuesday : ℕ), bouquets_sold_Wednesday = bouquets_sold_Tuesday / 3)
  (total_bouquets : bouquets_sold_Monday + bouquets_sold_Tuesday + bouquets_sold_Wednesday = 60)
  : bouquets_sold_Monday = 12 := 
sorry

end bouquets_sold_on_Monday_l800_800594


namespace troll_ratio_l800_800191

theorem troll_ratio 
  (B : ℕ)
  (h1 : 6 + B + (1 / 2 : ℚ) * B = 33) : 
  B / 6 = 3 :=
by
  sorry

end troll_ratio_l800_800191


namespace largest_valid_number_l800_800080

-- Define the conditions for the digits of the number
def valid_digits (a b c d : ℕ) : Prop :=
  c = a + b ∧ d = b + c

-- Prove that the number formed by digits 9, 0, 9, 9 is the largest valid 4-digit number
theorem largest_valid_number : ∃ a b c d, valid_digits a b c d ∧
  (a * 1000 + b * 100 + c * 10 + d = 9099) :=
begin
  use [9, 0, 9, 9],
  split,
  { -- Proof of valid digits condition
    split; refl },
  { -- Proof that the number is 9099
    refl }
end

end largest_valid_number_l800_800080


namespace andrew_paid_1428_to_shopkeeper_l800_800094

-- Given conditions
def rate_per_kg_grapes : ℕ := 98
def quantity_of_grapes : ℕ := 11
def rate_per_kg_mangoes : ℕ := 50
def quantity_of_mangoes : ℕ := 7

-- Definitions for costs
def cost_of_grapes : ℕ := rate_per_kg_grapes * quantity_of_grapes
def cost_of_mangoes : ℕ := rate_per_kg_mangoes * quantity_of_mangoes
def total_amount_paid : ℕ := cost_of_grapes + cost_of_mangoes

-- Theorem to prove the total amount paid
theorem andrew_paid_1428_to_shopkeeper : total_amount_paid = 1428 := by
  sorry

end andrew_paid_1428_to_shopkeeper_l800_800094


namespace biscuit_weight_not_qualified_l800_800881

theorem biscuit_weight_not_qualified :
  let standard_weight := 350
      deviation := 5
      acceptable_min := standard_weight - deviation
      acceptable_max := standard_weight + deviation
      weight_A := 348
      weight_B := 352
      weight_C := 358
      weight_D := 346
  in (weight_C < acceptable_min ∨ weight_C > acceptable_max) :=
by
  let standard_weight := 350
  let deviation := 5
  let acceptable_min := standard_weight - deviation
  let acceptable_max := standard_weight + deviation
  let weight_A := 348
  let weight_B := 352
  let weight_C := 358
  let weight_D := 346
  sorry

end biscuit_weight_not_qualified_l800_800881


namespace min_y_value_l800_800651

theorem min_y_value :
  ∃ c : ℝ, ∀ x : ℝ, (5 * x^2 + 20 * x + 25) >= c ∧ (∀ x : ℝ, (5 * x^2 + 20 * x + 25 = c) → x = -2) ∧ c = 5 :=
by
  sorry

end min_y_value_l800_800651


namespace increase_by_50_percent_l800_800924

def original : ℕ := 350
def increase_percent : ℕ := 50
def increased_number : ℕ := original * increase_percent / 100
def final_number : ℕ := original + increased_number

theorem increase_by_50_percent : final_number = 525 := 
by
  sorry

end increase_by_50_percent_l800_800924


namespace pencils_ratio_l800_800371

theorem pencils_ratio 
  (jeff_initial_pencils : ℕ)
  (vicki_initial_pencils : ℕ)
  (donated_ratio_jeff : ℕ)
  (donated_ratio_vicki : ℕ)
  (remaining_pencils : ℕ) 
  (h1 : jeff_initial_pencils = 300)
  (h2 : donated_ratio_jeff = 30)
  (h3 : donated_ratio_vicki = 3/4)
  (h4 : remaining_pencils = 360) 
  (h5 : jeff_initial_pencils - 0.30 * jeff_initial_pencils + vicki_initial_pencils * (1 - 3/4) = remaining_pencils):
  vicki_initial_pencils / jeff_initial_pencils = 2 :=
by sorry

end pencils_ratio_l800_800371


namespace annual_income_is_2000_l800_800203

-- Define the conditions as given in the problem
def investment_amount : ℝ := 6800
def dividend_rate : ℝ := 0.40
def price_per_share : ℝ := 136
def par_value : ℝ := 100

-- Calculate dividend per share
def dividend_per_share : ℝ := par_value * dividend_rate

-- Calculate number of shares purchased
def number_of_shares : ℝ := investment_amount / price_per_share

-- Calculate annual income
def calculate_annual_income : ℝ := dividend_per_share * number_of_shares

-- The proof problem to prove the annual income is $2000
theorem annual_income_is_2000 : calculate_annual_income = 2000 := by
  sorry

end annual_income_is_2000_l800_800203


namespace max_value_of_y_is_2_l800_800814

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a * x^2 + (a - 3) * x

theorem max_value_of_y_is_2 (a : ℝ) (h : ∀ x : ℝ, (3 * x^2 + 2 * a * x + (a - 3)) = (3 * x^2 - 2 * a * x + (a - 3))) : 
  ∃ x : ℝ, f a x = 2 :=
sorry

end max_value_of_y_is_2_l800_800814


namespace alex_sandwich_count_l800_800144

-- Conditions: Alex has 12 kinds of lunch meat and 8 kinds of cheese
def lunch_meat := fin 12
def cheese := fin 8

-- The problem is to prove the total number of different sandwiches Alex can make is 528
theorem alex_sandwich_count : (nat.choose 12 2) * (nat.choose 8 1) = 528 := by
  sorry

end alex_sandwich_count_l800_800144


namespace find_g_2021_l800_800985

noncomputable def g : ℝ → ℝ := sorry

theorem find_g_2021 (h : ∀ x y : ℝ, g(x - y) = 2021 * (g x + g y) - 2022 * x * y) :
  g 2021 = 2043231 :=
by sorry

end find_g_2021_l800_800985


namespace min_colors_2016x2016_l800_800547

def min_colors_needed (n : ℕ) : ℕ :=
  ⌈log 2 (n * n)⌉.natAbs

theorem min_colors_2016x2016 :
  min_colors_needed 2016 = 11 := 
by
  sorry

end min_colors_2016x2016_l800_800547


namespace section_is_parabola_l800_800445

-- Defining the conditions as Lean definitions.
def apex_angle (cone : Type) : ℝ := 90
def section_angle (cone : Type) : ℝ := 45

-- The formal statement of the problem in Lean 4.
theorem section_is_parabola (cone : Type) 
  (h_apex_angle: apex_angle cone = 90) 
  (h_section_angle: section_angle cone = 45) 
  : section_is_parabola cone :=
sorry

end section_is_parabola_l800_800445


namespace vectors_parallel_l800_800291

theorem vectors_parallel (m : ℝ) (a : ℝ × ℝ := (m, -1)) (b : ℝ × ℝ := (1, m + 2)) :
  (∃ k : ℝ, a = (k * b.1, k * b.2)) → m = -1 := by
  sorry

end vectors_parallel_l800_800291


namespace scores_greater_than_18_l800_800344

theorem scores_greater_than_18 (scores : Fin 20 → ℝ) 
  (h_unique : Function.Injective scores)
  (h_sum : ∀ i j k : Fin 20, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k) :
  ∀ i : Fin 20, scores i > 18 := 
by
  sorry

end scores_greater_than_18_l800_800344


namespace imag_part_fraction_imag_unit_l800_800855

theorem imag_part_fraction_imag_unit : 
  ∀ i : ℂ, i^2 = -1 → (im ((1 + i) / (1 - i)) = 1) :=
by
  intros
  have h : 1 - i ≠ 0, by
    simp [sub_ne_zero]
  have h1 : (1 + i) / (1 - i) = ((1 + i) * (1 + i)) / ((1 - i) * (1 + i)), by
    field_simp [h]
  rw h1
  norm_num
  rw [← div_mul_div, mul_self_conj, conj_i]
  simp
  sorry

end imag_part_fraction_imag_unit_l800_800855


namespace gcd_of_polynomials_l800_800669

theorem gcd_of_polynomials (n : ℕ) (h : n > 2^5) : gcd (n^3 + 5^2) (n + 6) = 1 :=
by sorry

end gcd_of_polynomials_l800_800669


namespace find_f_8_l800_800675

def f : ℝ → ℝ
| x := if x ≤ 5 then x - 5 * x^2 else f (x - 2)

theorem find_f_8 : f 8 = -76 :=
by sorry

end find_f_8_l800_800675


namespace parabola_tangent_circle_radius_l800_800282

theorem parabola_tangent_circle_radius :
  (∃ (t : ℝ), (8*t^2) = x ∧ (8*t) = y) →
  (∃ (m : ℝ), m = 1 ∧ ∃ (b : ℝ), b = 2 ∧ (y = m*x - b)) →
  (∃ (r : ℝ), 0 < r ∧ ∀ x y, (x - 4)^2 + y^2 = r^2 ∧ abs ((4*1 - 0*1 - 2) / sqrt (1^2 + (-1)^2)) = r) → 
  r = √2 :=
by {
  sorry
}

end parabola_tangent_circle_radius_l800_800282


namespace price_decrease_l800_800124

theorem price_decrease (original_price : ℝ) (h_original_pos : original_price > 0) :
  let increased_price := original_price * 1.10 in
  let decreased_price := increased_price * 0.90 in
  decreased_price < original_price :=
by
  sorry

end price_decrease_l800_800124


namespace find_y_l800_800025

open Nat

theorem find_y (y : ℕ) (h1 : sum_factors y = 10) (h2 : 2 ∣ y) : y = 6 := by
  sorry

/-- Helper function to compute the sum of all positive factors of an integer -/
def sum_factors (n : ℕ) : ℕ := (List.range' 1 (n + 1)).filter (λ d, d ∣ n).sum

end find_y_l800_800025


namespace shapes_cannot_form_rectangle_l800_800888

-- Definitions representing the 5 shapes and the rectangle
def shape1 := {squares : list (nat × nat) // squares.length = 4}
def shape2 := {squares : list (nat × nat) // squares.length = 4}
def shape3 := {squares : list (nat × nat) // squares.length = 4}
def shape4 := {squares : list (nat × nat) // squares.length = 4}
def shape5 := {squares : list (nat × nat) // squares.length = 4}

def rectangle := {squares : list (nat × nat) // squares.length = 20 ∧ (∀ s ∈ squares, s.1 < 4 ∧ s.2 < 5)}

theorem shapes_cannot_form_rectangle (s1 : shape1) (s2 : shape2) (s3 : shape3) (s4 : shape4) (s5 : shape5) :
  ∀ r : rectangle, ¬ (s1.squares ∪ s2.squares ∪ s3.squares ∪ s4.squares ∪ s5.squares = r.squares) :=
by
  sorry

end shapes_cannot_form_rectangle_l800_800888


namespace correct_propositions_l800_800797

section
variables {V : Type*} [inner_product_space ℝ V] 
variables {A B C : V} (AB AC BC : V)
def is_isosceles := (AB + AC) • (AB - AC) = 0 → ∥AB∥ = ∥AC∥
def is_right_angle := inner (AC) (AB) = 0

theorem correct_propositions 
  (H1 : AB - AC = BC)
  (H2 : AB + BC + (-(AB + BC)) = 0)
  (H3 : ∀ x y : V, (x + y) • (x - y) = 0 → ∥x∥ = ∥y∥)
  (H4 : ∀ x y z : V, inner x y = 0) :
  ② ∧ ③
  := by {
  exact sorry
}
end

end correct_propositions_l800_800797


namespace investment_y_l800_800099

theorem investment_y (x y z: ℕ) (investment_x investment_z: ℕ)
  (hx : investment_x = 5000)
  (hz : investment_z = 7000)
  (ratio_x ratio_y ratio_z: ℕ)
  (hratios : ratio_x = 2 ∧ ratio_y = 6 ∧ ratio_z = 7) :
  y = 15000 :=
by
  let part_value := investment_x / ratio_x
  have hpart_value_eval : part_value = 2500, sorry
  let expected_y := part_value * ratio_y
  have hexpected_y_eval : expected_y = 15000, sorry
  exact hexpected_y_eval

end investment_y_l800_800099


namespace sum_binom_2024_mod_2027_l800_800869

theorem sum_binom_2024_mod_2027 :
  let T := ∑ k in Finset.range 65, Nat.choose 2024 k
  2027.prime →
  T % 2027 = 1089 :=
by
  intros T hp
  sorry

end sum_binom_2024_mod_2027_l800_800869


namespace not_convex_path_O_l800_800150

-- Defining what it means for a polygon to be convex
def is_convex_polygon (n : ℕ) (A : fin n → ℝ × ℝ) : Prop :=
  sorry -- Using sorry since precise definition is not required.

-- Defining the circumcenter for a given triangle
def circumcenter (A B C : ℝ × ℝ) : ℝ × ℝ :=
  sorry -- Using sorry since precise definition is not required.

-- Define the main theorem that needs to be proved.
theorem not_convex_path_O {n : ℕ} (h : n ≥ 5)
  (A : fin n → ℝ × ℝ)
  (all_obtuse : ∀ i : fin n, 90 < angle (A i) (A (i + 1) % n) (A (i + 2) % n) < 180) :
  ¬is_convex_polygon n (λ i, circumcenter (A ((i + n - 1) % n)) (A i) (A ((i + 1) % n))) :=
sorry

end not_convex_path_O_l800_800150


namespace math_olympiad_proof_l800_800348

theorem math_olympiad_proof (scores : Fin 20 → ℕ) 
  (h_diff : ∀ i j, i ≠ j → scores i ≠ scores j) 
  (h_sum : ∀ i j k, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k) : 
  ∀ i, scores i > 18 :=
by
  sorry

end math_olympiad_proof_l800_800348


namespace quadrilateral_Q_area_l800_800174

def regularOctagon (P : ℝ → ℝ → Prop) := ∀ i : ℤ, 1 ≤ i ∧ i ≤ 8 →
  let s := 6 / (1 + Real.sqrt 2)
  in P (s * Real.cos (π * (2 * i - 1) / 8)) (s * Real.sin (π * (2 * i - 1) / 8))

def apothem (a : ℝ) : Prop := a = 3

def midpoint (P : ℝ → ℝ → Prop) (Q : ℝ → ℝ → Prop) : Prop :=
  ∀ i : ℤ, 1 ≤ i ∧ i ≤ 8 → 
  Q ((P i).1 + (P i + 1).1 / 2) ((P i).2 + (P i + 1).2 / 2)

theorem quadrilateral_Q_area (P Q : ℝ → ℝ → Prop) (a : ℝ) :
  regularOctagon P → apothem a → midpoint P Q →
  area (Q 1) (Q 2) (Q 3) (Q 4) = 9 * (3 - 2 * Real.sqrt 2) := 
by sorry

end quadrilateral_Q_area_l800_800174


namespace greatest_integer_part_expected_winnings_l800_800530

noncomputable def expected_winnings_one_envelope : ℝ := 500

noncomputable def expected_winnings_two_envelopes : ℝ := 625

noncomputable def expected_winnings_three_envelopes : ℝ := 695.3125

theorem greatest_integer_part_expected_winnings :
  ⌊expected_winnings_three_envelopes⌋ = 695 :=
by 
  sorry

end greatest_integer_part_expected_winnings_l800_800530


namespace quadratic_inequality_no_solution_l800_800769

theorem quadratic_inequality_no_solution (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 3 * x + a < 0) ↔ a ∈ Iio (-3 / 2) :=
by
  sorry

end quadratic_inequality_no_solution_l800_800769


namespace find_x_l800_800787

-- Define the angles as real numbers representing degrees.
variable (angle_SWR angle_WRU angle_x : ℝ)

-- Conditions given in the problem
def conditions (angle_SWR angle_WRU angle_x : ℝ) : Prop :=
  angle_SWR = 50 ∧ angle_WRU = 30 ∧ angle_SWR = angle_WRU + angle_x

-- Main theorem to prove that x = 20 given the conditions
theorem find_x (angle_SWR angle_WRU angle_x : ℝ) :
  conditions angle_SWR angle_WRU angle_x → angle_x = 20 := by
  sorry

end find_x_l800_800787


namespace find_original_radius_l800_800947

variable (r : ℝ) (π : ℝ := Real.pi)

def volume_sphere (R : ℝ) : ℝ := (4 / 3) * π * R^3
def volume_quarter_sphere (r : ℝ) : ℝ := (1 / 3) * π * r^3

theorem find_original_radius (r := 2 * Real.cbrt 4) : ∃ (R : ℝ), R = 2 :=
by
  sorry

end find_original_radius_l800_800947


namespace percent_increase_march_to_april_l800_800484

variables (P : ℝ) (X : ℝ)
def April_profit := P * (1 + X / 100)
def May_profit := April_profit * 0.8
def June_profit := May_profit * 1.5
def final_profit := P * 1.8000000000000003

theorem percent_increase_march_to_april :
  June_profit = final_profit → X = 50 :=
by
  intros h
  unfold June_profit final_profit at h
  sorry

end percent_increase_march_to_april_l800_800484


namespace five_digit_numbers_l800_800295

theorem five_digit_numbers (M : ℕ) (b : ℕ) (y : ℕ) :
  10000 ≤ M ∧ M ≤ 99999 ∧
  y = M % 10000 ∧ 
  M = 9 * y ∧
  (1 ≤ b ∧ b ≤ 9) ∧
  M = 10000 * b + y →
  ∃ b_vals : list ℕ, (∀ b' ∈ b_vals, 1 ≤ b' ∧ b' ≤ 7) ∧ list.length b_vals = 7 :=
by
  sorry

end five_digit_numbers_l800_800295


namespace intervals_of_monotonicity_range_of_a_for_three_zeros_l800_800271

    -- Define the function f(x) = 1/2 * x^2 - 3 * a * x + 2 * a^2 * ln(x)
    noncomputable def f (a x : ℝ) : ℝ := (1/2) * x^2 - 3 * a * x + 2 * a^2 * Real.log x

    -- Prove the intervals of monotonicity for f(x)
    theorem intervals_of_monotonicity (a : ℝ) (h : a ≠ 0) :
      (a > 0 → 
         monotone_on (f a) (Ioo 0 a) ∧
         antitone_on (f a) (Ioo a (2 * a)) ∧
         monotone_on (f a) (Ioi (2 * a))) ∧
      (a < 0 → 
         monotone_on (f a) (Ioi 0)) := 
    sorry

    -- Define the function's number of zeros and check the range of 'a' for having 3 zeros
    noncomputable def number_of_zeros (f : ℝ → ℝ) : ℕ := 
      -- A placeholder function representing the number of zeros of f
      sorry 

    theorem range_of_a_for_three_zeros :
      ∃ a : ℝ, (e^(5/4) < a ∧ a < (e^2 / 2)) ∧ number_of_zeros (f a) = 3 :=
    sorry
    
end intervals_of_monotonicity_range_of_a_for_three_zeros_l800_800271


namespace sequence_general_term_l800_800854

theorem sequence_general_term (n : ℕ) : 
  (2 * n - 1) / (2 ^ n) = a_n := 
sorry

end sequence_general_term_l800_800854


namespace is_happy_number_512_l800_800313

def happy_number (n : ℕ) : Prop := 
  ∃ k : ℕ, n = 8 * k

theorem is_happy_number_512 : happy_number 512 := 
  by
  unfold happy_number
  use 64
  sorry

end is_happy_number_512_l800_800313


namespace find_angle_DAB_l800_800384

variable (A B C D : Type)
variable [linear_ordered_field A] [linear_ordered_field B] [linear_ordered_field C] [linear_ordered_field D]

variables (AB : A) (DC : A) (angle_DCA : B) (angle_DAC : B) (angle_ABC : B)

noncomputable def angle_DAB : B :=
  if h1 : AB = DC ∧ angle_DCA = 24 ∧ angle_DAC = 31 ∧ angle_ABC = 55 then 63 else 0

theorem find_angle_DAB
  (AB : A) (DC : A) 
  (angle_DCA : B) (angle_DAC : B) (angle_ABC : B)
  (h : AB = DC)
  (h1 : angle_DCA = 24)
  (h2 : angle_DAC = 31)
  (h3 : angle_ABC = 55) :
  angle_DAB AB DC angle_DCA angle_DAC angle_ABC = 63 :=
  by {
    sorry
  }

end find_angle_DAB_l800_800384


namespace probability_of_multiples_l800_800622

noncomputable def multiples (n m : Nat) : Nat :=
  (1 to m).count (λ x => x % n = 0)

theorem probability_of_multiples : 
  let A := multiples 4 150
  let B := multiples 5 150
  let C := multiples 6 150
  let AB := multiples (Nat.lcm 4 5) 150
  let AC := multiples (Nat.lcm 4 6) 150
  let BC := multiples (Nat.lcm 5 6) 150
  let ABC := multiples (Nat.lcm 4 (Nat.lcm 5 6)) 150
  let total := 150
  let prob := (A + B + C - AB - AC - BC + ABC) / total
  prob = 7 / 15 :=
by
  sorry

end probability_of_multiples_l800_800622


namespace total_sheets_folded_l800_800620

theorem total_sheets_folded (initially_folded : ℕ) (additionally_folded : ℕ) (total_folded : ℕ) :
  initially_folded = 45 → additionally_folded = 18 → total_folded = initially_folded + additionally_folded → total_folded = 63 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3


end total_sheets_folded_l800_800620


namespace correct_average_of_10_numbers_l800_800919

theorem correct_average_of_10_numbers
    (avg : ℝ)
    (n : ℝ)
    (incorrect_num1 actual_num1 : ℝ)
    (incorrect_num2 actual_num2 : ℝ)
    (h_avg : avg = 40.2)
    (h_n : n = 10)
    (h_incorrect_num1 : incorrect_num1 = actual_num1 + 19)
    (h_incorrect_num2 : incorrect_num2 = 13)
    (h_actual_num2 : actual_num2 = 31) :
  (1 / n) * ((n * avg) - 19 + (actual_num2 - incorrect_num2)) = 40.1 :=
by
  -- Constants inferred from the conditions
  have h_sum := h_n * h_avg
  have h_corrected_sum := h_sum - 19 + (actual_num2 - incorrect_num2)
  have h_correct_avg := (1 / h_n) * h_corrected_sum
  -- The result
  exact h_correct_avg

end correct_average_of_10_numbers_l800_800919


namespace triangle_congruence_by_hl_l800_800526

theorem triangle_congruence_by_hl (A B C D E F : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]
  (BC EF AC DF : ℝ)
  (angleA angleD : ℝ)
  (h1 : BC = EF)
  (h2 : AC = DF)
  (h3 : angleA = 90)
  (h4 : angleD = 90) :
  congruent (triangle A B C) (triangle D E F) :=
by
  sorry

end triangle_congruence_by_hl_l800_800526


namespace polynomial_monomial_degree_l800_800715

theorem polynomial_monomial_degree :
  ∀ (m n : ℤ),
  (∀ {x y : ℤ}, (Polynomial : ℤ) ({coeff 1 / (5*x^(m+1)*y^(2) + x*y - 4*x^(3) + 1)}) = 6) →
  (∀ {x y : ℤ}, (Monomial : ℤ) ({coeff 1 / (8*x^(2*n)*y^(5-m)}) = 6) →
  (-m)^3 + 2*n = -23 :=
by
  intros m n h_poly h_mono
  sorry

end polynomial_monomial_degree_l800_800715


namespace residual_correctness_l800_800014

noncomputable def regression_equation (x : ℝ) : ℝ := 0.85 * x - 82.71

def residual (x actual_weight : ℝ) : ℝ := 
  let predicted_weight := regression_equation x
  actual_weight - predicted_weight

theorem residual_correctness : 
  residual 160 53 = -0.29 :=
by
  -- Place-holder for the proof
  sorry

end residual_correctness_l800_800014


namespace minimum_colors_2016_board_l800_800548

theorem minimum_colors_2016_board : ∃ k : ℕ, ∀ (board : Array (Array ℕ)), (∀ i j : ℕ, i < 2016 → j < 2016 → board[i][j] ∈ Fin k) ∧
(board[0][0] = 1) ∧
(∀ i : ℕ, i < 2016 → board[i][i] = 1) ∧
(∀ i j : ℕ, i < j → i < 2016 → j < 2016 → board[i][j] = board[j][i]) ∧
(∀ i j : ℕ, i < 2016 → j < 2016 → i ≠ j → board[i][j] ≠ board[i][2015-i]) → 
k = 11 :=
by
  sorry

end minimum_colors_2016_board_l800_800548


namespace total_distance_traveled_l800_800412

def trip_duration : ℕ := 8
def speed_first_half : ℕ := 70
def speed_second_half : ℕ := 85
def time_each_half : ℕ := trip_duration / 2

theorem total_distance_traveled :
  let distance_first_half := time_each_half * speed_first_half
  let distance_second_half := time_each_half * speed_second_half
  let total_distance := distance_first_half + distance_second_half
  total_distance = 620 := by
  sorry

end total_distance_traveled_l800_800412


namespace largest_valid_four_digit_number_l800_800060

-- Definition of the problem conditions
def is_valid_number (a b c d : ℕ) : Prop :=
  c = a + b ∧ d = b + c

-- Proposition that we need to prove
theorem largest_valid_four_digit_number : ∃ (a b c d : ℕ),
  is_valid_number a b c d ∧ a * 1000 + b * 100 + c * 10 + d = 9099 :=
by
  sorry

end largest_valid_four_digit_number_l800_800060


namespace murine_sum_mod_488_l800_800808

def is_k_murine {p : ℕ} (S : Type) [Fintype S] [DecidableEq S] (f : S → S) (k : ℕ) : Prop :=
  ∀ u v : S, (inner_product (f u) (f v)) % p = (inner_product u v) % p

def num_k_murine {p : ℕ} [Fact (Nat.Prime p)] (k : ℕ) : ℕ :=
  -- Function to count the k-murine functions (details omitted)
  sorry

def sum_k_murine_functions (p : ℕ) [Fact (Nat.Prime p)] : ℕ :=
  List.sum (List.map (num_k_murine p) (List.range (p+1)))

theorem murine_sum_mod_488 (p : ℕ) [Fact (Nat.Prime p)] (h : p = 491) :
  (sum_k_murine_functions p) % 488 = 18 := by
  sorry

end murine_sum_mod_488_l800_800808


namespace smallest_n_for_rotation_l800_800662

open Real Matrix

noncomputable def rot_60 := ![
  #[cos (60 * (π / 180)), -sin (60 * (π / 180))],
  #[sin (60 * (π / 180)), cos (60 * (π / 180))]
]

theorem smallest_n_for_rotation :
  ∃ n : ℕ, 0 < n ∧ rot_60 ^ n = (1 : Matrix ℝ (Fin 2) (Fin 2)) ∧ ∀ m : ℕ, 0 < m ∧ rot_60 ^ m = (1 : Matrix ℝ (Fin 2) (Fin 2)) → n ≤ m :=
begin
  -- Proof will go here
  sorry
end

end smallest_n_for_rotation_l800_800662


namespace trailing_zeroes_70_140_l800_800746

theorem trailing_zeroes_70_140 : 
  (trailing_zeroes (70! + 140!)) = 16 :=
sorry

end trailing_zeroes_70_140_l800_800746


namespace sequences_recurrence_relation_l800_800486

theorem sequences_recurrence_relation 
    (a : ℕ → ℕ) (b : ℕ → ℕ) (c : ℕ → ℕ)
    (h1 : a 1 = 1) (h2 : b 1 = 3) (h3 : c 1 = 2)
    (ha : ∀ i : ℕ, a (i + 1) = a i + c i - b i + 2)
    (hb : ∀ i : ℕ, b (i + 1) = (3 * c i - a i + 5) / 2)
    (hc : ∀ i : ℕ, c (i + 1) = 2 * a i + 2 * b i - 3) : 
    (∀ n, a n = 2^(n-1)) ∧ (∀ n, b n = 2^n + 1) ∧ (∀ n, c n = 3 * 2^(n-1) - 1) := 
sorry

end sequences_recurrence_relation_l800_800486


namespace ellipse_equation_and_line_through_foci_l800_800712

-- Given conditions
def F1 : ℝ × ℝ := (-2, 0)
def F2 : ℝ × ℝ := (2, 0)
def P : ℚ × ℚ := (5/2, -3/2)

-- Standard equation of the ellipse
def standard_equation_of_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Verification of the line equation
def perpendicular_property (O A B : ℝ × ℝ) : Prop :=
  (O.1 * A.1 + O.2 * A.2) = 0 ∧ (O.1 * B.1 + O.2 * B.2) = 0 

def line_equation (m : ℝ) (x y : ℝ) : Prop :=
  y = m * x + 2 * m

theorem ellipse_equation_and_line_through_foci
  (a b : ℝ) (m1 m2 : ℝ) :
  (standard_equation_of_ellipse a b 5/2 (-3/2) ∧
   a = sqrt (b^2 + 4)) →
  (line_equation (sqrt 15 / 15) 0 0 ∨ 
   line_equation (-sqrt 15 / 15) 0 0) :=
sorry

end ellipse_equation_and_line_through_foci_l800_800712


namespace trig_identity_l800_800102

variable (α : ℝ)

theorem trig_identity :
    (sin (6 * α) + sin (7 * α) + sin (8 * α) + sin (9 * α)) / (cos (6 * α) + cos (7 * α) + cos (8 * α) + cos (9 * α)) = 
    tan (15 * α / 2) :=
  sorry

end trig_identity_l800_800102


namespace find_T_values_l800_800554

variables {a : Fin 9 → ℝ}

def S (a : Fin 9 → ℝ) : ℝ :=
  ∑ i in Finset.range 9, (i + 1) * min (a i) (a ((i + 1) % 9))

def T (a : Fin 9 → ℝ) : ℝ :=
  ∑ i in Finset.range 9, (i + 1) * max (a i) (a ((i + 1) % 9))

theorem find_T_values
  (h_nonneg : ∀ i : Fin 9, 0 ≤ a i)
  (h_sum : ∑ i, a i = 1)
  (h_S_max : ∀ a', (∀ i, 0 ≤ a' i) → (∑ i, a' i = 1) → S a' ≤ S a):
  T a ∈ Set.Icc (21 / 5 : ℝ) (19 / 4 : ℝ) :=
sorry

end find_T_values_l800_800554


namespace horner_method_value_v2_value_of_v2_is_correct_l800_800046

variable (x : ℝ)
variable (v0 v1 v2 : ℝ)

-- Given conditions
def polynomial := 2 * x^5 - x^4 + 2 * x^2 + 5 * x + 3
def x_value := 3
def initial_v0 := 2
def initial_v1 := 5

theorem horner_method_value_v2 :
  v2 = initial_v1 * x_value := 
  by sorry

-- Prove that v2 = 15 given the conditions
theorem value_of_v2_is_correct :
  (v0 = initial_v0) → (v1 = initial_v1) → (x = x_value) → (v2 = initial_v1 * x_value) → v2 = 15 :=
  by
  intros h_v0 h_v1 h_x h_v2
  rw [h_v0, h_v1, h_x, h_v2]
  norm_num
  exact h_v2


end horner_method_value_v2_value_of_v2_is_correct_l800_800046


namespace total_bill_is_60_l800_800632

def num_adults := 6
def num_children := 2
def cost_adult := 6
def cost_child := 4
def cost_soda := 2

theorem total_bill_is_60 : num_adults * cost_adult + num_children * cost_child + (num_adults + num_children) * cost_soda = 60 := by
  sorry

end total_bill_is_60_l800_800632


namespace plane_equation_l800_800386

variable (x y z : ℝ)

def w : ℝ × ℝ × ℝ := (3, -3, 1)
def v : ℝ × ℝ × ℝ := (x, y, z)

def proj (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let num := u.1 * v.1 + u.2 * v.2 + u.3 * v.3
  let denom := v.1 * v.1 + v.2 * v.2 + v.3 * v.3
  let scale := num / denom
  (scale * v.1, scale * v.2, scale * v.3)

theorem plane_equation : proj v w = w → 3 * x - 3 * y + z - 19 = 0 :=
by
  sorry

end plane_equation_l800_800386


namespace sales_volume_relation_maximize_profit_l800_800038

-- Definition of the conditions given in the problem
def cost_price : ℝ := 40
def min_selling_price : ℝ := 45
def initial_selling_price : ℝ := 45
def initial_sales_volume : ℝ := 700
def sales_decrease_rate : ℝ := 20

-- Lean statement for part 1
theorem sales_volume_relation (x : ℝ) : 
  (45 ≤ x) →
  (y = 700 - 20 * (x - 45)) → 
  y = -20 * x + 1600 := sorry

-- Lean statement for part 2
theorem maximize_profit (x : ℝ) :
  (45 ≤ x) →
  (P = (x - 40) * (-20 * x + 1600)) →
  ∃ max_x max_P, max_x = 60 ∧ max_P = 8000 := sorry

end sales_volume_relation_maximize_profit_l800_800038


namespace is_happy_number_512_l800_800312

def happy_number (n : ℕ) : Prop := 
  ∃ k : ℕ, n = 8 * k

theorem is_happy_number_512 : happy_number 512 := 
  by
  unfold happy_number
  use 64
  sorry

end is_happy_number_512_l800_800312


namespace minimum_colors_2016_board_l800_800550

theorem minimum_colors_2016_board : ∃ k : ℕ, ∀ (board : Array (Array ℕ)), (∀ i j : ℕ, i < 2016 → j < 2016 → board[i][j] ∈ Fin k) ∧
(board[0][0] = 1) ∧
(∀ i : ℕ, i < 2016 → board[i][i] = 1) ∧
(∀ i j : ℕ, i < j → i < 2016 → j < 2016 → board[i][j] = board[j][i]) ∧
(∀ i j : ℕ, i < 2016 → j < 2016 → i ≠ j → board[i][j] ≠ board[i][2015-i]) → 
k = 11 :=
by
  sorry

end minimum_colors_2016_board_l800_800550


namespace rhombus_perimeter_l800_800464

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 20) (h2 : d2 = 16) : 
  let side := (real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)) in 
  4 * side = 8 * real.sqrt 41 := by
  sorry

end rhombus_perimeter_l800_800464


namespace largest_four_digit_number_prop_l800_800052

theorem largest_four_digit_number_prop :
  ∃ (a b c d : ℕ), a = 9 ∧ b = 0 ∧ c = 9 ∧ d = 9 ∧ (1000 * a + 100 * b + 10 * c + d = 9099) ∧ (c = a + b) ∧ (d = b + c) :=
by
  sorry

end largest_four_digit_number_prop_l800_800052


namespace largest_four_digit_number_l800_800058

theorem largest_four_digit_number :
  ∃ a b c d : ℕ, 
    9 < 1000 * a + 100 * b + 10 * c + d ∧ 
    1000 * a + 100 * b + 10 * c + d < 10000 ∧ 
    c = a + b ∧ 
    d = b + c ∧ 
    1000 * a + 100 * b + 10 * c + d = 9099 :=
by {
  sorry
}

end largest_four_digit_number_l800_800058


namespace max_value_bx_plus_a_l800_800218

variable (a b : ℝ)

theorem max_value_bx_plus_a (h : ∀ x, 0 ≤ x ∧ x ≤ 1 → |a * x + b| ≤ 1) :
  ∃ x, 0 ≤ x ∧ x ≤ 1 ∧ |b * x + a| = 2 :=
by
  -- Proof goes here
  sorry

end max_value_bx_plus_a_l800_800218


namespace solve_inequality_l800_800728

theorem solve_inequality 
  (k_0 k b m n : ℝ)
  (hM1 : -1 = k_0 * m + b) (hM2 : -1 = k^2 / m)
  (hN1 : 2 = k_0 * n + b) (hN2 : 2 = k^2 / n) :
  {x : ℝ | x^2 > k_0 * k^2 + b * x} = {x : ℝ | x < -1 ∨ x > 2} :=
  sorry

end solve_inequality_l800_800728


namespace degrees_divisibility_l800_800390

theorem degrees_divisibility
  {f g : Polynomial ℤ}
  (hf_coeffs : ∀ i, f.coeff i = 1 ∨ f.coeff i = 2022)
  (hg_coeffs : ∀ i, g.coeff i = 1 ∨ g.coeff i = 2022)
  (hdiv : f ∣ g) :
  (f.natDegree + 1) ∣ (g.natDegree + 1) :=
sorry

end degrees_divisibility_l800_800390


namespace find_common_ratio_of_geometric_sequence_l800_800711

-- Definitions based on conditions in a)
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Lean 4 statement for the proof problem
theorem find_common_ratio_of_geometric_sequence (q : ℝ) (a : ℕ → ℝ) 
    (h_pos : 0 < q) 
    (h_geometric : is_geometric_sequence a q) 
    (h_relation : a 3 * a 9 = 2 * (a 5)^2) : 
  q = real.sqrt 2 := 
sorry

end find_common_ratio_of_geometric_sequence_l800_800711


namespace probability_more_2s_than_5s_l800_800759

/-- If Greg rolls six fair six-sided dice, the probability that he rolls more 2's 
than 5's is 16710 / 46656. -/
theorem probability_more_2s_than_5s : 
  let outcomes := 6^6 in
  let equal_count := 4096 + 7680 + 1440 + 20 in
  let prob_equal := equal_count / outcomes in
  let desired_prob := (1 / 2) * (1 - prob_equal) in
  desired_prob = 16710 / 46656 :=
by
  sorry

end probability_more_2s_than_5s_l800_800759


namespace relationship_p_q_no_linear_term_l800_800766

theorem relationship_p_q_no_linear_term (p q : ℝ) :
  (∀ x : ℝ, (x^2 - p * x + q) * (x - 3) = x^3 + (-p - 3) * x^2 + (3 * p + q) * x - 3 * q) 
  → (3 * p + q = 0) → (q + 3 * p = 0) :=
by
  intro h_expansion coeff_zero
  sorry

end relationship_p_q_no_linear_term_l800_800766


namespace part_a_part_b_part_c_l800_800560

-- Part (a)
theorem part_a (m : ℤ) : (m^2 + 10) % (m - 2) = 0 ∧ (m^2 + 10) % (m + 4) = 0 ↔ m = -5 ∨ m = 9 := 
sorry

-- Part (b)
theorem part_b (n : ℤ) : ∃ m : ℤ, (m^2 + n^2 + 1) % (m - n + 1) = 0 ∧ (m^2 + n^2 + 1) % (m + n + 1) = 0 :=
sorry

-- Part (c)
theorem part_c (n : ℤ) : ∃ N : ℕ, ∀ m : ℤ, (m^2 + n^2 + 1) % (m - n + 1) = 0 ∧ (m^2 + n^2 + 1) % (m + n + 1) = 0 → m < N :=
sorry

end part_a_part_b_part_c_l800_800560


namespace pentagon_diagonals_sum_l800_800385

theorem pentagon_diagonals_sum (P Q R S T : Point) (hpq : distance P Q = 4) (hrs : distance R S = 4) (hqr : distance Q R = 9) (hst : distance S T = 9) (hpt : distance P T = 15) :
  let sum_diagonals := 3 * 15 + (15^2 - 81) / 4 + (15^2 - 16) / 9
  sum_diagonals = 104.222 :=
by
  sorry

end pentagon_diagonals_sum_l800_800385


namespace jelly_beans_remaining_l800_800892

-- Definitions based on the conditions
def total_jelly_beans : ℕ := 100
def total_children : ℕ := 40
def percentage_allowed : ℝ := 0.80
def jelly_beans_per_child : ℕ := 2

-- Translate the problem statement into a Lean proposition
theorem jelly_beans_remaining : total_jelly_beans - (nat.floor (percentage_allowed * total_children) * jelly_beans_per_child) = 36 :=
by
  sorry

end jelly_beans_remaining_l800_800892


namespace min_points_in_set_satisfying_circle_conditions_l800_800336

theorem min_points_in_set_satisfying_circle_conditions 
  (M : Set Point) 
  (C : ℕ → Set Point) 
  (hC1 : ∃ p ∈ M, C 1 = {p}) 
  (hC2 : ∃ p1 p2 ∈ M, C 2 = {p1, p2}) 
  (hC3 : ∃ p1 p2 p3 ∈ M, C 3 = {p1, p2, p3}) 
  (hC4 : ∃ p1 p2 p3 p4 ∈ M, C 4 = {p1, p2, p3, p4}) 
  (hC5 : ∃ p1 p2 p3 p4 p5 ∈ M, C 5 = {p1, p2, p3, p4, p5}) 
  (hC6 : ∃ p1 p2 p3 p4 p5 p6 ∈ M, C 6 = {p1, p2, p3, p4, p5, p6}) 
  (hC7 : ∃ p1 p2 p3 p4 p5 p6 p7 ∈ M, C 7 = {p1, p2, p3, p4, p5, p6, p7}) : 
  ∃ S ⊆ M, (∀ n, 1 ≤ n ∧ n ≤ 7 → ∃ s ⊆ S, C n = s ∧ |s| = n) ∧ |S| = 12 := 
by 
  sorry

end min_points_in_set_satisfying_circle_conditions_l800_800336


namespace journey_time_l800_800536

theorem journey_time
  (speed1 speed2 : ℝ)
  (distance total_time : ℝ)
  (h1 : speed1 = 40)
  (h2 : speed2 = 60)
  (h3 : distance = 240)
  (h4 : total_time = 5) :
  ∃ (t1 t2 : ℝ), (t1 + t2 = total_time) ∧ (speed1 * t1 + speed2 * t2 = distance) ∧ (t1 = 3) := 
by
  use (3 : ℝ), (2 : ℝ)
  simp [h1, h2, h3, h4]
  norm_num
  -- Additional steps to finish the proof would go here, but are omitted as per the requirements
  -- sorry

end journey_time_l800_800536


namespace problem1_problem2_problem3_l800_800259

-- Example Statement for the Math Problems.
section ProblemStatement

-- (I) Prove the range of values for \( m \)
theorem problem1 (m : ℝ) : (-m + (37 / 4) > 0) ↔ (m < (37 / 4)) := sorry

-- (II) Prove the symmetrical circle equation when circle \( C \) is tangent to line \( l \)
theorem problem2 : ( ∃ (a b : ℝ), a = 0 ∧ b = 7 / 2 ∧ group.circle (x^2 + (y - b)^2 = 1 / 8)) := sorry

-- (III) Prove the existence of m such that the circle with PQ as its diameter passes through origin
theorem problem3 : ( ∃ (m : ℝ), m = -3 / 2 ∧ group.circle {C: x^2 + y^2 + x - 6y + m + λ (x + y - 3) = 0} ) := sorry

end ProblemStatement

end problem1_problem2_problem3_l800_800259


namespace probability_different_bins_l800_800036

-- Definitions of conditions
def balls := (red, green, blue)
def probability_in_bin (k : ℕ) : ℝ := 3^(-k)

-- The theorem to prove
theorem probability_different_bins : 
  ∑ k in Finset.range(∞), (probability_in_bin k) ^ 3 = (1/26) → 1 - (1/26) = 25/26 :=
begin
  sorry
end

end probability_different_bins_l800_800036


namespace tom_slices_after_returns_and_eat_l800_800506

theorem tom_slices_after_returns_and_eat :
  (let total_slices := 5 * 6 in
   let jerry_slices := 18 in
   let tim_slices := 8 in
   let kate_slices := total_slices - jerry_slices - tim_slices in
   let jerry_returned := 1/3 * jerry_slices in
   let tim_returned := 1/2 * tim_slices in
   let total_returned := jerry_returned + tim_returned + kate_slices in
   let tom_ate := 3/5 * total_returned in
   let tom_left := total_returned - tom_ate in
   tom_left = 6) :=
sorry

end tom_slices_after_returns_and_eat_l800_800506


namespace min_races_needed_l800_800420

noncomputable def minimum_races (total_horses : ℕ) (max_race_horses : ℕ) : ℕ :=
  if total_horses ≤ max_race_horses then 1 else
  if total_horses % max_race_horses = 0 then total_horses / max_race_horses else total_horses / max_race_horses + 1

/-- We need to show that the minimum number of races required to find the top 3 fastest horses
    among 35 horses, where a maximum of 4 horses can race together at a time, is 10. -/
theorem min_races_needed : minimum_races 35 4 = 10 :=
  sorry

end min_races_needed_l800_800420


namespace rhombus_perimeter_l800_800455

theorem rhombus_perimeter
  (d1 d2 : ℝ)
  (h1 : d1 = 20)
  (h2 : d2 = 16) :
  4 * (Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)) = 8 * Real.sqrt 41 := 
  sorry

end rhombus_perimeter_l800_800455


namespace number_of_centroids_l800_800175

-- Define the vertices of the square
def squareVertices := [(0, 0), (20, 0), (20, 20), (0, 20)] : List (ℤ × ℤ)

-- Define the equally spaced points along the perimeter
def perimeter_points : List (ℤ × ℤ) := sorry -- Define all 80 points here (left as sorry for brevity)

-- Define a function to check if three points are non-collinear
def non_collinear (P Q R : ℤ × ℤ) : Prop :=
  ¬ (P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2) = 0)

-- Define the centroid calculation
def centroid (P Q R : ℤ × ℤ) : ℚ × ℚ :=
  ((P.1 + Q.1 + R.1) / 3, (P.2 + Q.2 + R.2) / 3)

-- Define the proportion calculation
def proportion (G : ℚ × ℚ) : ℚ × ℚ :=
  (3 * G.1, 3 * G.2)

-- Main theorem statement
theorem number_of_centroids : 
  (∃ (P Q R : ℤ × ℤ), P ∈ perimeter_points ∧ Q ∈ perimeter_points ∧ R ∈ perimeter_points ∧ 
                      non_collinear P Q R ∧ let G := centroid P Q R in 
                      let (m, n) := proportion G in
                      1 ≤ m ∧ m ≤ 59 ∧ 1 ≤ n ∧ n ≤ 59) = 3481 := sorry

end number_of_centroids_l800_800175


namespace geometric_sequence_common_ratio_l800_800704

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) (h : ∀ n, a (n + 1) = a n * q)
  (h0 : a 1 = 2) (h1 : a 4 = 1 / 4) : q = 1 / 2 :=
by
  sorry

end geometric_sequence_common_ratio_l800_800704


namespace initial_number_of_girls_l800_800673

theorem initial_number_of_girls (b g : ℕ) (h1 : b = 3 * (g - 20)) (h2 : 7 * (b - 54) = g - 20) : g = 39 :=
sorry

end initial_number_of_girls_l800_800673


namespace sum_binom_2024_mod_2027_l800_800871

theorem sum_binom_2024_mod_2027 :
  let T := ∑ k in Finset.range 65, Nat.choose 2024 k
  2027.prime →
  T % 2027 = 1089 :=
by
  intros T hp
  sorry

end sum_binom_2024_mod_2027_l800_800871


namespace minimal_abs_p_minus_q_l800_800757

theorem minimal_abs_p_minus_q (p q : ℕ) (h : p * q - 4 * p + 3 * q = 312) : ∃ (min_abs : ℕ), min_abs = 12 ∧ 
  ∀ (p q: ℕ), p * q - 4 * p + 3 * q = 312 → |p - q| ≥ min_abs :=
begin
  use 12,
  sorry -- proof to be completed
end

end minimal_abs_p_minus_q_l800_800757


namespace treasure_chest_age_l800_800936

theorem treasure_chest_age (n : ℕ) (h : n = 3 * 8^2 + 4 * 8^1 + 7 * 8^0) : n = 231 :=
by
  sorry

end treasure_chest_age_l800_800936


namespace intersection_of_cone_section_is_parabola_l800_800448

def cone_section_is_parabola (apex_angle section_angle : ℝ) : Prop :=
  (apex_angle = 90) → (section_angle = 45) → (section_shape = "parabola")

theorem intersection_of_cone_section_is_parabola :
  cone_section_is_parabola 90 45 :=
by
  intro h1
  intro h2
  sorry

end intersection_of_cone_section_is_parabola_l800_800448


namespace eval_f_log3_l800_800853

def f : ℝ → ℝ :=
  λ x, if x > 0 then (1/3)^x else
  if x < 0 then f (-x) else 1

theorem eval_f_log3 (x : ℝ) (hx : x = log 3 (1/6)) : f x = 1 / 6 := by
  sorry

end eval_f_log3_l800_800853


namespace convert_base10_to_base7_l800_800177

-- Definitions for powers and conditions
def n1 : ℕ := 7
def n2 : ℕ := n1 * n1
def n3 : ℕ := n2 * n1
def n4 : ℕ := n3 * n1

theorem convert_base10_to_base7 (n : ℕ) (h₁ : n = 395) : 
  ∃ a b c d : ℕ, 
    a * n3 + b * n2 + c * n1 + d = 395 ∧
    a < 7 ∧ b < 7 ∧ c < 7 ∧ d < 7 ∧
    a = 1 ∧ b = 1 ∧ c = 0 ∧ d = 3 :=
by { sorry }

end convert_base10_to_base7_l800_800177


namespace units_digit_2_pow_10_l800_800523

theorem units_digit_2_pow_10 : (2 ^ 10) % 10 = 4 := 
sorry

end units_digit_2_pow_10_l800_800523


namespace min_beacons_required_l800_800365

-- Definitions for rooms and beacons in the maze
constant Room : Type
constant Beacon : Type
constant distance : Room → Beacon → ℕ  -- Distance function

-- The maze has rooms and corridors 
constant Maze : Type
constant contains : Maze → Room → Prop
constant has_beacons : Maze → ℕ → Prop -- property check for number of beacons

-- A set of beacon locations
constant beacon_locations : set Beacon

-- Predicate for the robot being able to determine its position from beacons uniquely
constant unique_determination_from_beacons (m : Maze) (b_locs : set Beacon) : Prop 

-- The problem statement in Lean
theorem min_beacons_required (m : Maze) : 
  unique_determination_from_beacons m beacon_locations → 
  has_beacons m 3 ∧ ∀ n, n < 3 → ¬ unique_determination_from_beacons m (set.take n beacon_locations) :=
sorry

end min_beacons_required_l800_800365


namespace find_midpoint_E_l800_800367

variables {V : Type*} [InnerProductSpace ℝ V] 
variables {A B C A1 B1 C1 E : V}

def regular_triangular_prism (A B C A1 B1 C1 : V) : Prop :=
  -- Define the regularity conditions for the triangular prism here
  sorry  

def midpoint (X Y M : V) : Prop :=
  dist X M = dist M Y ∧ dist X Y = 2 * dist X M

theorem find_midpoint_E 
  (h_prism : regular_triangular_prism A B C A1 B1 C1)
  (h_length : dist A B = dist A A1)
  (h_45 : ∀ line_segment : seg V, line_segment = (A1, E) → 
    dihedral_angle A1 (plane_eq_of_points A1 E C) (plane_eq_of_points A1 B1 C1) = 45) :
  midpoint E B B1 :=
sorry


end find_midpoint_E_l800_800367


namespace coloring_minimum_colors_l800_800541

theorem coloring_minimum_colors (n : ℕ) (h : n = 2016) : ∃ k : ℕ, k = 11 ∧
  (∀ (board : Matrix ℕ ℕ ℕ), 
    let color : ℕ → ℕ → ℕ := λ i j, if i = j then 0           -- Main diagonal colored '0'
                                      else if i < j then j - i -- Left of the diagonal
                                      else i - j               -- Right of the diagonal
    ∧ (∀ i < n, ∀ j < n, color i j = color j i)                -- Symmetry wrt diagonal
    ∧ (∀ i < n, ∀ j k, j ≠ k → color i j ≠ color i k)          -- Different colors in the row  
    ) :=
begin
  use 11,
  split,
  { refl },
  { intros board,
    let color := λ i j, if i = j then 0 else if i < j then j - i else i - j,
    split,
    { intros i hi j hj,
      exact (if i < j then rfl else rfl), },
    { intros i hi j k hjk,
      by_cases h : i = j,
      { rw h,
        exact hjk.elim, },
      { by_cases h' : j < k,
        { rw if_pos h' },
        { by_contradiction,
          have := nat.le_antisymm,
          contradiction } } } }
end

end coloring_minimum_colors_l800_800541


namespace number_of_intersections_l800_800212

-- Definitions of the conditions
def sine_fn (x : ℝ) : ℝ := Real.sin x
def exp_fn (x : ℝ) : ℝ := (1 / 3) ^ x
def interval : Set ℝ := {x | 0 < x ∧ x < 100 * Real.pi}

-- Main theorem statement
theorem number_of_intersections : 
  ∃ n : ℕ, n = 50 ∧ 
  (∀ x ∈ interval, sine_fn x = exp_fn x ↔ x ∈ (Set.univ : Set ℝ).finite) :=
sorry

end number_of_intersections_l800_800212


namespace inequality_am_gm_l800_800430

theorem inequality_am_gm (n : ℕ) (h : 2 ≤ n) : 
  n * (Real.sqrt' n (n + 1) - 1) ≤ (∑ k in Finset.range n, 1/(k+1) : ℝ) ∧ (∑ k in Finset.range n, 1/(k+1) : ℝ) ≤ n - (n-1) / Real.sqrt (n-1) n := 
by 
  sorry

end inequality_am_gm_l800_800430


namespace num_palindromic_numbers_l800_800900

/--
Prove that the number of positive five-digit integers that are palindromes
using only the digits 1, 2, and 3 is 27.
-/
theorem num_palindromic_numbers : 
  let digits := {1, 2, 3}
  let count := 3 * 3 * 3
  count = 27 := 
by
  sorry

end num_palindromic_numbers_l800_800900


namespace vectors_parallel_l800_800292

theorem vectors_parallel (m : ℝ) (a : ℝ × ℝ := (m, -1)) (b : ℝ × ℝ := (1, m + 2)) :
  (∃ k : ℝ, a = (k * b.1, k * b.2)) → m = -1 := by
  sorry

end vectors_parallel_l800_800292


namespace best_fitting_model_l800_800524

theorem best_fitting_model (R2_Model1 R2_Model2 R2_Model3 R2_Model4 : ℝ)
    (h1 : R2_Model1 = 0.98)
    (h2 : R2_Model2 = 0.80)
    (h3 : R2_Model3 = 0.50)
    (h4 : R2_Model4 = 0.25) :
    R2_Model1 = max R2_Model1 (max R2_Model2 (max R2_Model3 R2_Model4)) :=
by
  have h_max : max R2_Model2 (max R2_Model3 R2_Model4) = R2_Model2,
  { sorry }
  have h_max_total : max R2_Model1 R2_Model2 = R2_Model1,
  { -- Since R2_Model1 has the highest value
    sorry }
  exact h_max_total

end best_fitting_model_l800_800524


namespace range_of_x_l800_800013

noncomputable def log_base (b x : ℝ) : ℝ :=
  real.log x / real.log b

theorem range_of_x (x : ℝ) :
  abs (log_base (1/2) x - (4 : ℂ)) ≥ abs (3 + (4 : ℂ)) →
  (0 < x ∧ x ≤ 1/8) ∨ (x ≥ 8) :=
by sorry

end range_of_x_l800_800013


namespace sqrt_fraction_simplification_l800_800922

theorem sqrt_fraction_simplification : 
  let a : ℝ := 8 
  let b : ℝ := 4 
  (8 = 2^3) → 
  (4 = 2^2) →
  Real.sqrt ((a^10 + b^10) / (a^4 + b^11)) = 16 := 
by 
  intros h1 h2
  rw [h1, h2]
  sorry

end sqrt_fraction_simplification_l800_800922


namespace min_lit_bulbs_l800_800224

theorem min_lit_bulbs (n : ℕ) (h : n ≥ 1) : 
  ∃ rows cols, (rows ⊆ Finset.range n) ∧ (cols ⊆ Finset.range n) ∧ 
  (∀ i j, (i ∈ rows ∧ j ∈ cols) ↔ (i + j) % 2 = 1) ∧ 
  rows.card * (n - cols.card) + cols.card * (n - rows.card) = 2 * n - 2 :=
by sorry

end min_lit_bulbs_l800_800224


namespace fair_coin_heads_before_tails_sum_m_n_l800_800391

-- Definition of the states and transitions according to the problem.
def prob_three_heads_before_two_tails : ℚ :=
  let q_1 := (1 : ℚ) / 2 in
  let r_1 := (1 : ℚ) / 4 in
  let q_2 := (1 : ℚ) / 2 + (1 : ℚ) / 2 * r_1 in
  let q_0 := (1 : ℚ) / 2 * q_1 + (1 : ℚ) / 2 * r_1 in
  q_0

-- Main theorem statement
theorem fair_coin_heads_before_tails_sum_m_n : let q := prob_three_heads_before_two_tails in
  ∃ m n : ℕ, (m : ℚ) / (n : ℚ) = q ∧ Nat.gcd m n = 1 ∧ m + n = 11 :=
begin
  sorry
end

end fair_coin_heads_before_tails_sum_m_n_l800_800391


namespace value_of_a_l800_800401

theorem value_of_a (U : Set ℕ) (M : Set ℕ) (a : ℕ) :
  U = {1, 3, 5, 7} →
  M = {1, |a - 5|} →
  compl U M = {5, 7} →
  (a = 2 ∨ a = 8) :=
by
  sorry

end value_of_a_l800_800401


namespace percentage_reduction_is_65_l800_800747

-- Define the initial conditions
def initial_volume : ℚ := 14
def initial_concentration : ℚ := 0.2
def added_water : ℚ := 26
def unchanged_alcohol : ℚ := initial_volume * initial_concentration
def new_total_volume : ℚ := initial_volume + added_water
def new_concentration : ℚ := unchanged_alcohol / new_total_volume

-- Define the percentage reduction in concentration
def percentage_reduction : ℚ := ((initial_concentration - new_concentration) / initial_concentration) * 100

-- The theorem to prove
theorem percentage_reduction_is_65 : percentage_reduction = 65 :=
by 
  unfold percentage_reduction
  unfold initial_concentration unchanged_alcohol new_total_volume new_concentration
  sorry

end percentage_reduction_is_65_l800_800747


namespace interest_rate_additional_investment_l800_800164

theorem interest_rate_additional_investment 
  (initial_investment : ℝ := 2200) 
  (initial_rate : ℝ := 0.05) 
  (total_income : ℝ := 1099.9999999999998) 
  (total_rate : ℝ := 0.06) : 
  ∃ r : ℝ, 
  r ≈ 0.0613 :=
by
  let annual_income_initial := initial_investment * initial_rate
  let total_investment a := initial_investment + a
  let total_annual_income a := total_rate * total_investment a
  let a := (total_income / total_rate - initial_investment)
  let r := (total_income - annual_income_initial) / a
  exact exists.intro r sorry

end interest_rate_additional_investment_l800_800164


namespace square_area_l800_800630

theorem square_area (A B C D G : Type) (D_G : ℝ) (h1 : DG = D_G) (h2 : D_G = 5) :
  let side_length : ℝ := 8 in
  let area := side_length * side_length in
  area = 64 :=
by
  sorry

end square_area_l800_800630


namespace largest_four_digit_number_l800_800056

theorem largest_four_digit_number :
  ∃ a b c d : ℕ, 
    9 < 1000 * a + 100 * b + 10 * c + d ∧ 
    1000 * a + 100 * b + 10 * c + d < 10000 ∧ 
    c = a + b ∧ 
    d = b + c ∧ 
    1000 * a + 100 * b + 10 * c + d = 9099 :=
by {
  sorry
}

end largest_four_digit_number_l800_800056


namespace sum_of_integers_between_neg10_and_5_l800_800967

theorem sum_of_integers_between_neg10_and_5 :
  (∑ i in Finset.Icc (-10 : ℤ) 5, i) = -40 := 
by
  sorry

end sum_of_integers_between_neg10_and_5_l800_800967


namespace expression_of_f_range_of_g_l800_800687

section
  variable (f : ℝ → ℝ)

  -- Given condition
  axiom f_condition : ∀ x : ℝ, f x + 2 * f (-x) = -3 * x - 6
  
  -- Goal 1: Find the analytical expression of f(x) as f(x) = 3x - 2
  theorem expression_of_f : ∀ x : ℝ, f x = 3 * x - 2 :=
  sorry

  -- Goal 2: Find the range of g(x) on [0, 3]
  noncomputable def g (x : ℝ) := x * f x

  theorem range_of_g : set.range (g ∘ coe : set.Icc 0 3 → ℝ) = set.Icc (-1/3 : ℝ) 21 :=
  sorry
end

end expression_of_f_range_of_g_l800_800687


namespace payment_to_C_l800_800532

def work_rate (days : ℕ) : ℚ := 1 / days

def total_payment : ℚ := 3360

def work_done (rate : ℚ) (days : ℕ) : ℚ := rate * days

-- Conditions
def person_A_work_rate := work_rate 6
def person_B_work_rate := work_rate 8
def combined_work_rate := person_A_work_rate + person_B_work_rate
def work_by_A_and_B_in_3_days := work_done combined_work_rate 3
def total_work : ℚ := 1
def work_done_by_C := total_work - work_by_A_and_B_in_3_days

-- Proof problem statement
theorem payment_to_C :
  (work_done_by_C / total_work) * total_payment = 420 := 
sorry

end payment_to_C_l800_800532


namespace olympiad_scores_above_18_l800_800339

theorem olympiad_scores_above_18 
  (n : Nat) 
  (scores : Fin n → ℕ) 
  (h_diff_scores : ∀ i j : Fin n, i ≠ j → scores i ≠ scores j) 
  (h_score_sum : ∀ i j k : Fin n, i ≠ j ∧ i ≠ k ∧ j ≠ k → scores i < scores j + scores k) 
  (h_n : n = 20) : 
  ∀ i : Fin n, scores i > 18 := 
by 
  -- See the proof for the detailed steps.
  sorry

end olympiad_scores_above_18_l800_800339


namespace binom_sum_mod_2027_l800_800866

theorem binom_sum_mod_2027 :
  let T := ∑ k in Finset.range 65, Nat.choose 2024 k
  T % 2027 = 1089 :=
by
  let T := ∑ k in Finset.range 65, Nat.choose 2024 k
  have h2027_prime : Nat.prime 2027 := by exact dec_trivial
  sorry -- This is the placeholder for the actual proof

end binom_sum_mod_2027_l800_800866


namespace seven_by_seven_grid_partition_l800_800925

theorem seven_by_seven_grid_partition : 
  ∀ (x y : ℕ), 4 * x + 3 * y = 49 ∧ x + y ≥ 16 → x = 1 :=
by sorry

end seven_by_seven_grid_partition_l800_800925


namespace difference_in_ages_is_nine_l800_800026

-- Representing the digits of Jean's birth year
variables {m c d u m' c' d' u' : ℕ}

-- Condition: The sum of the digits of their birth years are equal
def sum_digits_equal : Prop := m + c + d + u = m' + c' + d' + u'

-- Condition: The age of each of them starts with the same digit
def same_starting_digit_age (year : ℕ) : Prop := 
  let age_Jean := year - (1000 * m + 100 * c + 10 * d + u)
  let age_Jack := year - (1000 * m' + 100 * c' + 10 * d' + u')
  age_Jean / 10 = age_Jack / 10

-- Main theorem
theorem difference_in_ages_is_nine (year : ℕ) 
  (h1 : sum_digits_equal) 
  (h2 : same_starting_digit_age year) :
  abs ((year - (1000 * m + 100 * c + 10 * d + u)) - (year - (1000 * m' + 100 * c' + 10 * d' + u'))) = 9 := by
  sorry

end difference_in_ages_is_nine_l800_800026


namespace problem_statement_l800_800721

noncomputable def f (x : ℝ) : ℝ := 2 * sqrt 3 * sin x * cos x - 2 * cos x^2 - 1

theorem problem_statement :
  (∃ T : ℝ, (∀ x : ℝ, f(x + T) = f(x)) ∧ T > 0 ∧ (∀ T' : ℝ, (∀ x : ℝ, f(x + T') = f(x)) ∧ T' > 0 → T' ≥ T)) ∧
  (∃ y : ℝ, y < 0 ∧ (∀ x : ℝ, f(x) ≥ y)) ∧
  ∃ (a b c A B C : ℝ), (c = sqrt 3) ∧ (f C = 0) ∧ (sin B = 2 * sin A) ∧
  (a = 1) ∧ (b = 2) :=
sorry

end problem_statement_l800_800721


namespace largest_four_digit_number_property_l800_800070

theorem largest_four_digit_number_property : ∃ (a b c d : Nat), 
  (1000 * a + 100 * b + 10 * c + d = 9099) ∧ 
  (c = a + b) ∧ 
  (d = b + c) ∧ 
  1 ≤ a ∧ a ≤ 9 ∧ 
  0 ≤ b ∧ b ≤ 9 ∧ 
  0 ≤ c ∧ c ≤ 9 ∧ 
  0 ≤ d ∧ d ≤ 9 := 
sorry

end largest_four_digit_number_property_l800_800070


namespace trigonometric_identity_l800_800679

theorem trigonometric_identity
  (θ : ℝ) 
  (h_tan : Real.tan θ = 3) :
  (1 - Real.cos θ) / (Real.sin θ) - (Real.sin θ) / (1 + (Real.cos θ)^2) = (11 * Real.sqrt 10 - 101) / 33 := 
by
  sorry

end trigonometric_identity_l800_800679


namespace real_part_of_z_is_zero_l800_800676

def complex_num (a b : ℤ) : ℝ × ℝ := (a, b)

-- Define the complex division operation
def complex_div (z1 z2 : ℝ × ℝ) : ℝ × ℝ :=
    let (a, b) := z1
    let (c, d) := z2
    ((a*c + b*d) / (c^2 + d^2), (b*c - a*d) / (c^2 + d^2))

noncomputable def z := complex_div (complex_num 2 1) (complex_num (-2) 1)

theorem real_part_of_z_is_zero : (z.1 = 0) :=
sorry

end real_part_of_z_is_zero_l800_800676


namespace sum_of_prime_cool_numbers_l800_800608

def is_cool (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ n = a.factorial * b.factorial + 315

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem sum_of_prime_cool_numbers :
  ∑ n in (Finset.filter is_prime (Finset.filter is_cool (Finset.range (10^6)))), id n = 317 :=
by
  sorry

end sum_of_prime_cool_numbers_l800_800608


namespace van_aubels_theorem_l800_800380

theorem van_aubels_theorem 
  (A B C P D E F : Type) 
  [Add A B C P D E F]
  (AP PD AF FB AE EC : ℝ)
  (hAP: AP > 0) (hPD: PD > 0) (hAF: AF > 0) (hFB: FB > 0) (hAE: AE > 0) (hEC: EC > 0) :
  let k := ∀ (x y z : Prop), x ∈ y ∧ y ∩ z ∧ z ∈ x in 
  k P (interior (triangle A B C)) ∧  
    meet AP BC D ∧ meet BP CA E ∧ meet CP AB F → 
      AP / PD = AF / FB + AE / EC :=
sorry

end van_aubels_theorem_l800_800380


namespace rhombus_perimeter_l800_800459

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 20) (h2 : d2 = 16) :
  let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  in 4 * s = 8 * Real.sqrt 41 := 
by
  sorry

end rhombus_perimeter_l800_800459


namespace max_result_of_expr_l800_800508

theorem max_result_of_expr : 
  ∃ expr : Int, 
    (∃ expr1 expr2, 
      expr1 = 20 * 20 - 18 ∧ 
      expr2 = 19 * expr1 ∧ 
      expr = expr2) ∧ 
    ∀ other_expr, expr ≥ other_expr :=
begin
  sorry
end

end max_result_of_expr_l800_800508


namespace jacket_total_selling_price_l800_800598

theorem jacket_total_selling_price :
  let original_price := 120
  let discount_rate := 0.30
  let tax_rate := 0.08
  let processing_fee := 5
  let discounted_price := original_price * (1 - discount_rate)
  let tax := discounted_price * tax_rate
  let total_price := discounted_price + tax + processing_fee
  total_price = 95.72 := by
  sorry

end jacket_total_selling_price_l800_800598


namespace product_of_number_and_sum_of_digits_l800_800782

variable (n : ℕ)
variable (tens units : ℕ)

def is_valid_number (n : ℕ) (tens units : ℕ) : Prop :=
  n = tens * 10 + units ∧ units = tens + 4

theorem product_of_number_and_sum_of_digits :
  is_valid_number 26 2 6 →
  let sum_of_digits := 2 + 6 in
  let product := 26 * sum_of_digits in
  product = 208 :=
by
  intros
  let sum_of_digits := 2 + 6
  let product := 26 * sum_of_digits
  have h1: sum_of_digits = 8 := by norm_num
  have h2: product = 208 := by norm_num
  exact h2

end product_of_number_and_sum_of_digits_l800_800782


namespace constant_term_of_binomial_expansion_l800_800705

noncomputable def constant_in_binomial_expansion (a : ℝ) : ℝ := 
  if h : a = ∫ (x : ℝ) in (0)..(1), 2 * x 
  then ((1 : ℝ) - (a : ℝ)^(-1 : ℝ))^6
  else 0

theorem constant_term_of_binomial_expansion : 
  ∃ a : ℝ, (a = ∫ (x : ℝ) in (0)..(1), 2 * x) → constant_in_binomial_expansion a = (15 : ℝ) := sorry

end constant_term_of_binomial_expansion_l800_800705


namespace pr_bisects_angle_mpn_l800_800511

section GeometryProblem

variables {r1 r2 : ℝ} -- Radii of the circles
variables {Γ1 Γ2 : Type} -- Types representing the circles
variables [MetricSpace Γ1] [MetricSpace Γ2] -- Circles are within some metric space
variables (P R M N : Γ1) (O1 O2 : Γ2) -- Points on the circles

-- Conditions (We assume the necessary structures on circles)
-- Circles touch internally at point P
axiom touch_at_P : ∀ (γ1 γ2 : Γ1), touches γ1 γ2 P
-- Tangent parallel to diameter through P touches Γ1 at R
axiom tangent_at_R : ∀ (γ1 : Γ1), is_tangent P R γ1
-- Same tangent intersects Γ2 at M and N
axiom tangent_intersection : ∀ (γ2 : Γ2), intersects γ2 P M ∧ intersects γ2 P N

-- Question
theorem pr_bisects_angle_mpn :
  ∀ (P R M N : Γ1) (O1 O2 : Γ2),
    touches Γ1 Γ2 P → 
    is_tangent P R Γ1 →
    (intersects Γ2 P M ∧ intersects Γ2 P N) →
    angle_bisector P R (∠ M P N) :=
sorry

end GeometryProblem

end pr_bisects_angle_mpn_l800_800511


namespace interval_of_monotonic_increase_l800_800479

-- Define the function as f(x)
noncomputable def f (x : ℝ) : ℝ := log (1 / 2) (x^2 - 2 * x + 1)

-- Define the domain condition
def domain (x : ℝ) : Prop := x ≠ 1

-- Statement to prove the interval of monotonic increase
theorem interval_of_monotonic_increase : ∀ x : ℝ, domain x → x ∈ set.Ioo (-∞) 1 → strict_mono_incr_on f (set.Ioo (-∞) 1) :=
sorry

end interval_of_monotonic_increase_l800_800479


namespace ball_hits_ground_at_time_l800_800852

theorem ball_hits_ground_at_time :
  ∀ (t : ℝ), (-18 * t^2 + 30 * t + 60 = 0) ↔ (t = (5 + Real.sqrt 145) / 6) :=
sorry

end ball_hits_ground_at_time_l800_800852


namespace math_problem_1_l800_800697

-- Definitions for the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℕ) (a1 d : ℕ) : Prop :=
  a 22 - 3 * a 7 = 2  ∧ (a 1 = a1 ∧ ∀ n, a (n + 1) = a1 + n * d)

-- Geometric sequence condition
def geometric_sequence_condition (a : ℕ → ℕ) (a1 d : ℕ) (S : ℕ → ℕ) : Prop :=
  (1 / a 2) * (sqrt (S 2 - 3)) = S 3

-- General formula for the sequence
def general_formula (a_n : ℕ → ℕ) : Prop :=
  ∀ n, a_n n = 2 * n

-- Condition on the range of λ
def lambda_range_condition (T_n : ℕ → ℕ) (λ : ℝ) : Prop :=
  ∀ n > 0, 64 * T_n n < abs (3 * λ - 1) → (λ ≥ 2 ∨ λ ≤ -4/3)

-- Combining all conditions for full problem statement
theorem math_problem_1 (a : ℕ → ℕ) (S : ℕ → ℕ) (a1 d : ℕ) (b : ℕ → ℝ) (T : ℕ → ℝ) (λ : ℝ) :
  arithmetic_sequence a a1 d →
  geometric_sequence_condition a a1 d S →
  general_formula a →
  lambda_range_condition T λ :=
sorry

end math_problem_1_l800_800697


namespace consecutive_negatives_product_to_sum_l800_800011

theorem consecutive_negatives_product_to_sum :
  ∃ (n : ℤ), n * (n + 1) = 2184 ∧ n + (n + 1) = -95 :=
by {
  sorry
}

end consecutive_negatives_product_to_sum_l800_800011


namespace harry_calculator_incorrect_calc_l800_800436

theorem harry_calculator_incorrect_calc (switch: ℕ → ℕ) (correct_calc : ∀ a b : ℕ, a * b = b * a) :
  switch 1 = 3 ∧ switch 5 = 7 ∧ switch 9 = 5 →
  ¬((switch 1 * 100 + switch 5 * 10 + switch 9) *
    (switch 9 * 100 + switch 5 * 10 + switch 1) = 159 * 951) := 
begin
  intros h,
  cases h with h1 h2,
  cases h2 with h5 h9,
  -- Prove that with the given switches, the calculation gives a wrong result
  -- We need to show this step to diverge from 159*951 with hinted switch values as 3, 7 and 9
  sorry
end

end harry_calculator_incorrect_calc_l800_800436


namespace solution_set_of_equation_l800_800718

def f (x : ℝ) : ℝ :=
  if x < 0 then 1 else x^2 + 1

theorem solution_set_of_equation :
  {x : ℝ | f (1 - x^2) = f (2 * x)} = {x : ℝ | x ≤ -1 ∨ x = -1 + Real.sqrt 2} :=
sorry

end solution_set_of_equation_l800_800718


namespace find_single_digit_l800_800597

def isSingleDigit (n : ℕ) : Prop := n < 10

def repeatedDigitNumber (A : ℕ) : ℕ := 10 * A + A 

theorem find_single_digit (A : ℕ) (h1 : isSingleDigit A) (h2 : repeatedDigitNumber A + repeatedDigitNumber A = 132) : A = 6 :=
by
  sorry

end find_single_digit_l800_800597


namespace minimum_colors_2016_board_l800_800551

theorem minimum_colors_2016_board : ∃ k : ℕ, ∀ (board : Array (Array ℕ)), (∀ i j : ℕ, i < 2016 → j < 2016 → board[i][j] ∈ Fin k) ∧
(board[0][0] = 1) ∧
(∀ i : ℕ, i < 2016 → board[i][i] = 1) ∧
(∀ i j : ℕ, i < j → i < 2016 → j < 2016 → board[i][j] = board[j][i]) ∧
(∀ i j : ℕ, i < 2016 → j < 2016 → i ≠ j → board[i][j] ≠ board[i][2015-i]) → 
k = 11 :=
by
  sorry

end minimum_colors_2016_board_l800_800551


namespace roots_of_unity_cubic_l800_800180

noncomputable def countRootsOfUnityCubic (c d e : ℤ) : ℕ := sorry

theorem roots_of_unity_cubic :
  ∃ (z : ℂ) (n : ℕ), (z^n = 1) ∧ (∃ (c d e : ℤ), z^3 + c * z^2 + d * z + e = 0)
  ∧ countRootsOfUnityCubic c d e = 12 :=
sorry

end roots_of_unity_cubic_l800_800180


namespace pentagon_area_l800_800980

theorem pentagon_area (ABCDE : Type) [ConvexPentagon ABCDE]
  (h : ∀ (D1 D2 : Diagonal ABCDE), Area (cut_off_triangle D1 ABCDE) = 1) :
  Area ABCDE = (5 + Real.sqrt 5) / 2 := 
sorry

end pentagon_area_l800_800980


namespace remainder_2_pow_305_mod_9_l800_800519

theorem remainder_2_pow_305_mod_9 :
  2^305 % 9 = 5 :=
by sorry

end remainder_2_pow_305_mod_9_l800_800519


namespace remainder_of_power_mod_l800_800998

theorem remainder_of_power_mod :
  (5^2023) % 11 = 4 :=
  by
    sorry

end remainder_of_power_mod_l800_800998


namespace triangle_area_l800_800953

theorem triangle_area : 
  let L1 := λ x : ℝ, 2 * x + 1
  let L2 := λ x : ℝ, -x + 4
  let L3 := λ x : ℝ, -1
  let p1 := (1 : ℝ, 3 : ℝ)
  let p2 := (-1 : ℝ, -1 : ℝ)
  let p3 := (5 : ℝ, -1 : ℝ)
  let base := dist (p2.1, p2.2) (p3.1, p3.2)
  let height := abs (p1.2 - p2.2)
  base * height * 0.5 = 12.00 :=
by
  sorry

end triangle_area_l800_800953


namespace max_a_plus_b_cubed_plus_c_fourth_l800_800815

theorem max_a_plus_b_cubed_plus_c_fourth (a b c : ℕ) (h : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 2) :
  a + b^3 + c^4 ≤ 2 := sorry

end max_a_plus_b_cubed_plus_c_fourth_l800_800815


namespace longest_segment_squared_in_sector_l800_800933

-- Definitions of the conditions
def diameter : ℝ := 20
def radius : ℝ := diameter / 2
def sector_angle : ℝ := 90  -- degrees

theorem longest_segment_squared_in_sector (d : ℝ) (r : ℝ) (angle : ℝ) 
  (h1 : d = 20) (h2 : r = d / 2) (h3 : angle = 90) :
  let m := r * Real.sqrt 2 in m^2 = 200 :=
by
  -- Placeholder for the actual proof
  sorry

end longest_segment_squared_in_sector_l800_800933


namespace boat_speed_in_still_water_l800_800566

theorem boat_speed_in_still_water (D V_s t_down t_up : ℝ) (h_val : V_s = 3) (h_down : D = (15 + V_s) * t_down) (h_up : D = (15 - V_s) * t_up) : 15 = 15 :=
by
  have h1 : 15 = (D / 1 - V_s) := sorry
  have h2 : 15 = (D / 1.5 + V_s) := sorry
  sorry

end boat_speed_in_still_water_l800_800566


namespace num_students_third_school_l800_800503

variable (x : ℕ)

def num_students_condition := (2 * (x + 40) + (x + 40) + x = 920)

theorem num_students_third_school (h : num_students_condition x) : x = 200 :=
sorry

end num_students_third_school_l800_800503


namespace perpendicular_vectors_l800_800289

def vector (α : Type) := (α × α)

variables (t : ℝ)

def a : vector ℝ := (1, 2)
def b : vector ℝ := (-4, t)

def dot_product (v1 v2 : vector ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem perpendicular_vectors :
  dot_product (1, 2) (-4, t) = 0 → t = 2 :=
by
  sorry

end perpendicular_vectors_l800_800289


namespace laptop_price_reduction_total_price_reduction_laptop_total_discount_l800_800121

theorem laptop_price_reduction (P : ℝ) : 
  (0.8 * P) * 0.7 * 0.9 = 0.504 * P :=
by sorry

theorem total_price_reduction : 
  (1 - 0.504) * 100 = 49.6 :=
by sorry

-- Proof of the theorem from the given conditions
theorem laptop_total_discount :
  ∀ (P : ℝ), ((1 - 0.504) * 100 = 49.6) :=
begin
  intros P,
  have h1 : (0.8 : ℝ) * P = 0.8 * P := by simp,
  have h2 : (0.8 * P) * (0.7 : ℝ) = 0.56 * P := by ring,
  have h3 : (0.56 * P) * (0.9 : ℝ) = 0.504 * P := by ring,
  have h4 : 1 - 0.504 = 0.496 := by norm_num,
  have h5 : 0.496 * 100 = 49.6 := by norm_num,
  exact h5,
  sorry
end

end laptop_price_reduction_total_price_reduction_laptop_total_discount_l800_800121


namespace distances_product_equal_l800_800696

variable {A B C P : Type} 
variable {k : circle A B C} -- Circumscribed circle of triangle ABC
variable {d_a d_b d_c t_a t_b t_c : ℝ} -- Distances

-- Assumptions based on conditions
variable (P_is_on_k : k.contains P)
variable (d_a_def : per_dist P BC = d_a)
variable (d_b_def : per_dist P CA = d_b)
variable (d_c_def : per_dist P AB = d_c)
variable (t_a_def : tang_dist P A = t_a)
variable (t_b_def : tang_dist P B = t_b)
variable (t_c_def : tang_dist P C = t_c)

theorem distances_product_equal : d_a * d_b * d_c = t_a * t_b * t_c := 
sorry

end distances_product_equal_l800_800696


namespace find_smallest_c_l800_800398

/-- Let a₀, a₁, ... and b₀, b₁, ... be geometric sequences with common ratios rₐ and r_b, 
respectively, such that ∑ i=0 ∞ aᵢ = ∑ i=0 ∞ bᵢ = 1 and 
(∑ i=0 ∞ aᵢ²)(∑ i=0 ∞ bᵢ²) = ∑ i=0 ∞ aᵢbᵢ. Prove that a₀ < 4/3 -/
theorem find_smallest_c (r_a r_b : ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h1 : ∑' n, a n = 1)
  (h2 : ∑' n, b n = 1)
  (h3 : (∑' n, (a n)^2) * (∑' n, (b n)^2) = ∑' n, (a n) * (b n)) :
  a 0 < 4 / 3 := by
  sorry

end find_smallest_c_l800_800398


namespace Debby_spent_on_yoyo_l800_800636

theorem Debby_spent_on_yoyo 
  (hat_tickets stuffed_animal_tickets total_tickets : ℕ) 
  (h1 : hat_tickets = 2) 
  (h2 : stuffed_animal_tickets = 10) 
  (h3 : total_tickets = 14) 
  : ∃ yoyo_tickets : ℕ, hat_tickets + stuffed_animal_tickets + yoyo_tickets = total_tickets ∧ yoyo_tickets = 2 := 
by 
  sorry

end Debby_spent_on_yoyo_l800_800636


namespace initial_mean_of_observations_l800_800862

-- Definitions of the given conditions and proof of the correct initial mean
theorem initial_mean_of_observations 
  (M : ℝ) -- Mean of 50 observations
  (initial_sum := 50 * M) -- Initial sum of observations
  (wrong_observation : ℝ := 23) -- Wrong observation
  (correct_observation : ℝ := 45) -- Correct observation
  (understated_by := correct_observation - wrong_observation) -- Amount of understatement
  (correct_sum := initial_sum + understated_by) -- Corrected sum
  (corrected_mean : ℝ := 36.5) -- Corrected new mean
  (eq1 : correct_sum = 50 * corrected_mean) -- Equation from condition of corrected mean
  (eq2 : initial_sum = 50 * corrected_mean - understated_by) -- Restating in terms of initial sum
  : M = 36.06 := -- The initial mean of observations
  sorry -- Proof omitted

end initial_mean_of_observations_l800_800862


namespace part1_rect_eq_circle_part2_intersection_sum_l800_800792

-- Definitions based on conditions
def line_param (t : ℝ) : ℝ × ℝ := (3 - (sqrt 2) * t / 2, sqrt 5 - (sqrt 2) * t / 2)
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2 * sqrt 5 * y = 0
def pointP : (ℝ × ℝ) := (3, sqrt 5)

-- Proving
theorem part1_rect_eq_circle : ∀ (x y : ℝ), (∃ θ : ℝ, (x, y) = ((2 * sqrt 5 * sin θ) * cos θ, (2 * sqrt 5 * sin θ) * sin θ)) ↔ circle_eq x y := by
  sorry

theorem part2_intersection_sum (t1 t2 : ℝ) (h1: line_param t1 = (λ (x y : ℝ), circle_eq x y)) (h2: line_param t2 = (λ (x y : ℝ), circle_eq x y))
  (t1_pos : t1 > 0) (t2_pos : t2 > 0) :
  abs (1 / dist pointP (line_param t1) + 1 / dist pointP (line_param t2)) = 3 * sqrt 2 / 4 := by
  sorry

end part1_rect_eq_circle_part2_intersection_sum_l800_800792


namespace probability_all_red_chips_drawn_l800_800600

   -- Define the problem setup
   def magician_hat := {chips : list (bool × string) // 
     chips.count (λ c, c = (true, "red")) = 4 ∧ 
     chips.count (λ c, c = (false, "green")) = 3}

   def draw_chips (hat : magician_hat) : list (bool × string) :=
     hat.val.take 7  -- We draw all the chips

   def all_red_drawn_before_two_green (draws : list (bool × string)) : Prop :=
     let greens_drawn := draws.count (λ c, c = (false, "green")) in
     let reds_drawn := draws.count (λ c, c = (true, "red")) in
     reds_drawn = 4 ∧ greens_drawn < 2

   def probability_all_red (hat : magician_hat) : ℚ :=
     if all_red_drawn_before_two_green (draw_chips hat)
     then 1 / 7
     else 0

   -- Problem statement in Lean to prove the probability
   theorem probability_all_red_chips_drawn : 
     ∀ hat : magician_hat, probability_all_red hat = 1 / 7 :=
   by sorry
   
end probability_all_red_chips_drawn_l800_800600


namespace martin_total_distance_l800_800406

theorem martin_total_distance (T S1 S2 t : ℕ) (hT : T = 8) (hS1 : S1 = 70) (hS2 : S2 = 85) (ht : t = T / 2) : S1 * t + S2 * t = 620 := 
by
  sorry

end martin_total_distance_l800_800406


namespace volume_of_silver_l800_800112

open Real

-- Given conditions
def diameter_mm : ℝ := 1
def length_m : ℝ := 14.00563499208679

-- Conversion constants
def mm_to_cm : ℝ := 1 / 10
def m_to_cm : ℝ := 100

-- Converted values
def radius_cm : ℝ := (diameter_mm / 2) * mm_to_cm
def height_cm : ℝ := length_m * m_to_cm

def volume_wire_cm³ : ℝ := π * (radius_cm ^ 2) * height_cm

theorem volume_of_silver :
  volume_wire_cm³ = 11.001102244054492 :=
sorry

end volume_of_silver_l800_800112


namespace find_q_l800_800654

theorem find_q (q : ℚ) (h : 16^4 = (8^3) / 2 * 4^(12 * q - 3)) : q = 7 / 12 :=
by
  sorry

end find_q_l800_800654


namespace average_speed_of_trip_l800_800567

theorem average_speed_of_trip :
  let speed1 := 30
  let time1 := 5
  let speed2 := 42
  let time2 := 10
  let total_time := 15
  let distance1 := speed1 * time1
  let distance2 := speed2 * time2
  let total_distance := distance1 + distance2
  let average_speed := total_distance / total_time
  average_speed = 38 := 
by 
  sorry

end average_speed_of_trip_l800_800567


namespace problem_expression_l800_800168

theorem problem_expression :
  (∏ n in (range 11).map (λ i, i + 2), (1 - 1 / ((n : ℚ) * n))) + 1 - 1 / 2 = 25 / 24 :=
by
  sorry

end problem_expression_l800_800168


namespace trajectory_of_midpoint_l800_800761

-- Definitions based on the conditions identified in the problem
variables {x y x1 y1 : ℝ}

-- Condition that point P is on the curve y = 2x^2 + 1
def point_on_curve (x1 y1 : ℝ) : Prop :=
  y1 = 2 * x1^2 + 1

-- Definition of the midpoint M conditions
def midpoint_def (x y x1 y1 : ℝ) : Prop :=
  x = (x1 + 0) / 2 ∧ y = (y1 - 1) / 2

-- Final theorem statement to be proved
theorem trajectory_of_midpoint (x y x1 y1 : ℝ) :
  point_on_curve x1 y1 → midpoint_def x y x1 y1 → y = 4 * x^2 :=
sorry

end trajectory_of_midpoint_l800_800761


namespace area_swept_by_triangle_l800_800790

theorem area_swept_by_triangle (BC AB AD : ℝ) (speed time : ℝ)
  (h1 : BC = 6) (h2 : AB = 5) (h3 : AD = 4) (h4 : speed = 3) (h5 : time = 2) :
  let BD := real.sqrt (AB^2 - AD^2)
  let h := speed * time 
  let triangle_area := (1/2) * BC * AD
  let rectangle_area := BC * h
  let parallelogram_area := BC * BD
  triangle_area + rectangle_area + parallelogram_area = 66 := by
  sorry

end area_swept_by_triangle_l800_800790


namespace total_is_twenty_l800_800926

def num_blue := 5
def num_red := 7
def prob_red_or_white : ℚ := 0.75

noncomputable def total_marbles (T : ℕ) (W : ℕ) :=
  5 + 7 + W = T ∧ (7 + W) / T = prob_red_or_white

theorem total_is_twenty : ∃ (T : ℕ) (W : ℕ), total_marbles T W ∧ T = 20 :=
by
  sorry

end total_is_twenty_l800_800926


namespace find_angle_AMN_l800_800421

-- Given points A, B, C, L, K
variables {A B C L K M N : Type}

-- Given line segments
variables [LineSegment A B] [LineSegment B C] [LineSegment C A]
variables [LineSegment A L]
variables [LineSegment A B] [LineSegment C K] [LineSegment A C] [LineSegment B K]

-- Given angles
variables (angle_BAD angle_BCD angle_ABC angle_BAK angle_BKL angle_KBL : Real)
variables (angle_AMN : Real)

-- Initial conditions
variables (h1: on_bisector A L A B C)
variables (h2: ∠ B K L = 30)
variables (h3: ∠ K B L = 30)
variables (h4: intersection_point M A B C K)
variables (h5: intersection_point N A C B K)

-- Theorem stating the problem
theorem find_angle_AMN : ∠ A M N = 60 := sorry

end find_angle_AMN_l800_800421


namespace scores_greater_than_18_l800_800345

theorem scores_greater_than_18 (scores : Fin 20 → ℝ) 
  (h_unique : Function.Injective scores)
  (h_sum : ∀ i j k : Fin 20, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k) :
  ∀ i : Fin 20, scores i > 18 := 
by
  sorry

end scores_greater_than_18_l800_800345


namespace largest_four_digit_number_property_l800_800071

theorem largest_four_digit_number_property : ∃ (a b c d : Nat), 
  (1000 * a + 100 * b + 10 * c + d = 9099) ∧ 
  (c = a + b) ∧ 
  (d = b + c) ∧ 
  1 ≤ a ∧ a ≤ 9 ∧ 
  0 ≤ b ∧ b ≤ 9 ∧ 
  0 ≤ c ∧ c ≤ 9 ∧ 
  0 ≤ d ∧ d ≤ 9 := 
sorry

end largest_four_digit_number_property_l800_800071


namespace impossible_to_return_to_original_set_l800_800701

theorem impossible_to_return_to_original_set (S : Set ℝ) (hS : S.size ≥ 2) 
  (h : ∀ x ∈ S, x ≠ 0) :
  ∀ S', (S' ⊆ S ∧ S'.size = S.size ∧
    (∀ A B ∈ S', A ≠ B → S' = (S' \ {A, B}) ∪ { (A + B) / 2, (B - A) / 2 })) →
    S' ≠ S :=
by
  sorry

end impossible_to_return_to_original_set_l800_800701


namespace value_of_6_15_minus_z_star_l800_800535

def z_star (z : ℝ) : ℤ :=
  if (z < 2) then 0 else 2 * Int.floor (z / 2)

theorem value_of_6_15_minus_z_star : 
  6.15 - (z_star 6.15) = 0.15 :=
by
  sorry

end value_of_6_15_minus_z_star_l800_800535


namespace tangency_segment_ratio_l800_800488

-- Define the sides ratio condition
def sides_ratio (a b c : ℝ) : Prop := a / b = 3 / 4 ∧ a / c = 3 / 5 ∧ b / c = 4 / 5

-- Define the right triangle with sides in the ratio 5:4:3
def right_triangle (a b c : ℝ) : Prop := 
  sides_ratio a b c ∧ a^2 + b^2 = c^2

-- Define the tangency segments
def tangency_segments (a b c l₁ l₂ l₃ k₁ k₂ k₃ : ℝ) : Prop :=
  l₁ + l₂ = a ∧ l₃ = b - k₁ ∧ k₂ = c - k₃ ∧
  l₁ = k₃ ∧ l₂ = k₂

-- The main theorem statement
theorem tangency_segment_ratio : ∀ (a b c l₁ l₂ l₃ k₁ k₂ k₃ : ℝ),
  right_triangle a b c →
  tangency_segments a b c l₁ l₂ l₃ k₁ k₂ k₃ →
  (l₁ = 2 ∧ l₂ = 3) ∨ (l₁ = 3 ∧ l₂ = 2) :=
begin
  intros,
  sorry
end

end tangency_segment_ratio_l800_800488


namespace age_twice_in_two_years_l800_800940

-- conditions
def father_age (S : ℕ) : ℕ := S + 24
def present_son_age : ℕ := 22
def present_father_age : ℕ := father_age present_son_age

-- theorem statement
theorem age_twice_in_two_years (S M Y : ℕ) (h1 : S = present_son_age) (h2 : M = present_father_age) : 
  M + 2 = 2 * (S + 2) :=
by
  sorry

end age_twice_in_two_years_l800_800940


namespace max_odd_integers_chosen_l800_800149

theorem max_odd_integers_chosen (a b c d e f g : ℕ) 
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f ∧ 0 < g)
  (h2 : a * b * c * d * e * f * g % 2 = 0) :
  (finset.card (finset.filter (λ x, x % 2 = 1) (finset.of_list [a, b, c, d, e, f, g])) ≤ 6) ∧ 
  (finset.card (finset.filter (λ x, x % 2 = 0) (finset.of_list [a, b, c, d, e, f, g])) ≥ 1) :=
by
  sorry

end max_odd_integers_chosen_l800_800149


namespace smaller_rectangle_perimeter_l800_800125

def problem_conditions (a b : ℝ) : Prop :=
  2 * (a + b) = 96 ∧ 
  8 * b + 11 * a = 342 ∧
  a + b = 48 ∧ 
  (a * (b - 1) <= 0 ∧ b * (a - 1) <= 0 ∧ a > 0 ∧ b > 0)

theorem smaller_rectangle_perimeter (a b : ℝ) (hab : problem_conditions a b) :
  2 * (a / 12 + b / 9) = 9 :=
  sorry

end smaller_rectangle_perimeter_l800_800125


namespace greatest_value_of_squares_l800_800810

theorem greatest_value_of_squares 
    (a b c d : ℝ)
    (h1 : a + b = 16)
    (h2 : a * b + c + d = 81)
    (h3 : a * d + b * c = 168)
    (h4 : c * d = 100) : a^2 + b^2 + c^2 + d^2 ≤ 82 := 
sorriendo

end greatest_value_of_squares_l800_800810


namespace Linda_six_scores_arithmetic_mean_l800_800403

def scores : List ℕ := [87, 90, 85, 93, 89, 92]

def arithmeticMean (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

theorem Linda_six_scores_arithmetic_mean :
  arithmeticMean scores = 268 / 3 := 
by 
  -- sum of the scores
  have h1 : scores.sum = 536 := by norm_num [scores, List.sum]
  -- length of the scores list
  have h2 : (scores.length : ℚ) = 6 := by norm_num [scores.length]
  -- computing the mean
  rw [scores, h1, h2]
  norm_num 
  sorry

end Linda_six_scores_arithmetic_mean_l800_800403


namespace second_year_students_auto_control_l800_800789

def total_students : ℕ := 673
def proportion_second_year : ℝ := 0.80
def students_numeric_methods : ℕ := 250
def students_both_methods : ℕ := 134
def total_second_year_students : ℕ := (proportion_second_year * total_students).to_nat

theorem second_year_students_auto_control : total_second_year_students = 538 →
  ∀ (A : ℕ), A + students_numeric_methods - students_both_methods = total_second_year_students → A = 422 :=
by
  intros h1 A h2
  sorry

end second_year_students_auto_control_l800_800789


namespace scientific_notation_of_0_0000003_l800_800963

theorem scientific_notation_of_0_0000003 :
  0.0000003 = 3 * 10^(-7) :=
sorry

end scientific_notation_of_0_0000003_l800_800963


namespace hockey_league_games_l800_800777

noncomputable def total_regular_games (top5_internal: ℕ) (top5_external: ℕ) (rank6to10_internal: ℕ) (rank6to10_external: ℕ) (bottom5_internal: ℕ) : ℕ :=
  top5_internal + top5_external + rank6to10_internal + rank6to10_external + bottom5_internal

theorem hockey_league_games : 
  total_regular_games 
  (nat.choose 5 2 * 12) -- Internal games for top 5
  (5 * 10 * 8) -- External games for top 5
  (nat.choose 5 2 * 10) -- Internal games for rank 6 to 10
  (5 * 5 * 6) -- External games for rank 6 to 10
  (nat.choose 5 2 * 8) -- Internal games for bottom 5
  = 850 := by
  sorry

end hockey_league_games_l800_800777


namespace angle_C_side_c_length_l800_800249

-- Define the basic conditions of the triangle
variables (A B C : ℝ) (a b c : ℝ)
-- Define the vectors m and n
variables (m : ℝ × ℝ) (n : ℝ × ℝ)
-- Define the vector product condition
variable (dot_product_condition : m.1 * n.1 + m.2 * n.2 = Real.sin (2 * C))
-- Define the arithmetic sequence condition
variable (arithmetic_sequence : 2 * Real.sin C = Real.sin A + Real.sin B)
-- Define the vector product of CA and AB - AC
variable (vector_product_condition : ∀ CA AB AC : ℝ × ℝ, CA.1 * (AB.1 - AC.1) + CA.2 * (AB.2 - AC.2) = 18)

-- Prove the size of angle C
theorem angle_C (m n : ℝ × ℝ) (dot_product_condition : m.1 * n.1 + m.2 * n.2 = Real.sin (2 * C)) (A B C : ℝ) :
  Real.cos C = 1 / 2 → C = Real.pi / 3 :=
by
suffices h : Real.sin (2 * C) = Real.sin C; sorry

-- Prove the length of side c
theorem side_c_length (a b c : ℝ) (ab_length : a * b = 36) (A B C : ℝ) (arithmetic_sequence : 2 * Real.sin C = Real.sin A + Real.sin B) :
  2 * c = a + b → c = 6 :=
by 
suffices h : c^2 = (a + b)^2 - 3 * 36; sorry


end angle_C_side_c_length_l800_800249


namespace find_m_l800_800227

def vector_parallel (a b : ℝ × ℝ) : Prop := 
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem find_m
  (m : ℝ)
  (a : ℝ × ℝ := (m, 1))
  (b : ℝ × ℝ := (2, -1))
  (h : vector_parallel a (b.1 - a.1, b.2 - a.2)) :
  m = -2 :=
by
  sorry

end find_m_l800_800227


namespace tom_average_speed_l800_800507

noncomputable def average_speed (d1 d2 s1 s2 : ℝ) : ℝ :=
  let t1 := d1 / s1
  let t2 := d2 / s2
  let total_distance := d1 + d2
  let total_time := t1 + t2
  total_distance / total_time

theorem tom_average_speed :
  average_speed 30 50 30 50 = 40 := 
begin
  sorry
end

end tom_average_speed_l800_800507


namespace slope_of_line_l800_800879

theorem slope_of_line (x y : ℝ) : (y = 3 * x - 1) → (3) := 
by
  sorry

end slope_of_line_l800_800879


namespace parallel_lines_condition_l800_800287

theorem parallel_lines_condition (a : ℝ) :
    (∀ x y : ℝ, (a - 1) * x + y + 3 = 0) → 
    (∀ x y : ℝ, 2 * x + a * y + 1 = 0) → 
    (l_1 ∥ l_2) → a = 2 ∨ a = -1 := 
begin
  intros h1 h2 h_parallel,
  -- proof steps would go here
  sorry
end

end parallel_lines_condition_l800_800287


namespace time_without_walkway_l800_800942

/-- A person walks with and against the direction of a moving walkway.-/
def walking_times (vp vw : ℝ) (d : ℝ) : Prop :=
  (d = (vp + vw) * 60) ∧ 
  (d = (vp - vw) * 360) ∧
  ((d/vp) = 200 * (36/65))

/-- Given the conditions, prove the time taken remains consistent when the walkway stops.-/
theorem time_without_walkway
  (d : ℝ := 200)
  (time_with := 60)
  (time_against := 360)
  (vp vw : ℝ) 
  (cond : walking_times vp vw d) :
  (200 / vp) ≈ 110.77 :=
by
  sorry

end time_without_walkway_l800_800942


namespace ants_convex_polygon_no_complete_coverage_ants_non_convex_polygon_no_complete_coverage_l800_800552

theorem ants_convex_polygon_no_complete_coverage :
  ∀ (P : Type) [polygon P] (is_convex : convex P) (side_lengths : ∀ s ∈ sides P, length s > 1)
    (initial_positions: {p1 p2 // p1 ∈ perimeter P ∧ p2 ∈ perimeter P ∧ dist p1 p2 = 0.1}),
    ¬(∀ p ∈ perimeter P, p = p1 ∨ p = p2) :=
sorry

theorem ants_non_convex_polygon_no_complete_coverage :
  ∀ (P : Type) [polygon P] (initial_positions: {p1 p2 // p1 ∈ perimeter P ∧ p2 ∈ perimeter P ∧ dist p1 p2 = 0.1}),
    ¬(∀ p ∈ perimeter P, p = p1 ∨ p = p2) :=
sorry

end ants_convex_polygon_no_complete_coverage_ants_non_convex_polygon_no_complete_coverage_l800_800552


namespace mean_home_runs_correct_l800_800166

def mean_home_runs : ℚ :=
  let n_5 := 2
  let n_6 := 3
  let n_7 := 1
  let n_9 := 1
  let n_10 := 2
  let total_home_runs := n_5 * 5 + n_6 * 6 + n_7 * 7 + n_9 * 9 + n_10 * 10
  let total_players := n_5 + n_6 + n_7 + n_9 + n_10
  total_home_runs / total_players

theorem mean_home_runs_correct : mean_home_runs = 64 / 9 :=
by
  unfold mean_home_runs
  norm_num
  exact rfl

end mean_home_runs_correct_l800_800166


namespace spinner_prob_C_l800_800603

theorem spinner_prob_C :
  (∃ A B C D E : ℝ, A = 4 / 10 ∧ B = 1 / 5 ∧ C = D ∧ E = 2 * C ∧ A + B + C + D + E = 1) →
  ∃ C : ℝ, C = 1 / 10 :=
begin
  sorry
end

end spinner_prob_C_l800_800603


namespace derivative_of_ex_cosx_l800_800451

theorem derivative_of_ex_cosx : 
  ∀ (x : ℝ), deriv (λ x, exp x * cos x) x = exp x * (cos x - sin x) :=
by
  intro x
  simp [deriv]
  have h := deriv_mul
  -- further steps needed for complete proof
  sorry

end derivative_of_ex_cosx_l800_800451


namespace max_sum_of_digits_l800_800190

theorem max_sum_of_digits (num1 num2 num3 : ℕ) (h1 : num1 ≠ 0) (h2 : num2 ≠ 0) (h3 : num3 ≠ 0) (h4 : num1 ≠ num2) (h5 : num2 ≠ num3) 
(h6 : num1 ≠ num3) (h7 : digits_used : ∀ d, d ∈ (num1.digits 10) ∪ (num2.digits 10) ∪ (num3.digits 10) ↔ d ∈ finset.range 10) :
(num1 + num2 + num3) ≤ 9760 + 8531 + 7429 → 9762 ∈ set.mk [num1, num2, num3] :=
by
  sorry

end max_sum_of_digits_l800_800190


namespace largest_four_digit_number_with_property_l800_800074

theorem largest_four_digit_number_with_property :
  ∃ (a b c d : ℕ), (a = 9 ∧ b = 0 ∧ c = 9 ∧ d = 9 ∧ c = a + b ∧ d = b + c ∧ 1000 * a + 100 * b + 10 * c + d = 9099) :=
sorry

end largest_four_digit_number_with_property_l800_800074


namespace art_department_probability_l800_800575

theorem art_department_probability : 
  let students := {s1, s2, s3, s4} 
  let first_grade := {s1, s2}
  let second_grade := {s3, s4}
  let total_pairs := { (x, y) | x ∈ students ∧ y ∈ students ∧ x < y }.to_finset.card
  let diff_grade_pairs := { (x, y) | x ∈ first_grade ∧ y ∈ second_grade ∨ x ∈ second_grade ∧ y ∈ first_grade}.to_finset.card
  (diff_grade_pairs / total_pairs) = 2 / 3 := 
by 
  sorry

end art_department_probability_l800_800575


namespace greatest_prime_factor_of_247_l800_800516

theorem greatest_prime_factor_of_247:
  ( ∃ (p : ℕ), nat.prime p ∧ ∃ (q : ℕ), nat.prime q ∧ 247 = p * q ∧ p ≤ q  ∧ q = 19) :=
begin
  have fact247 : 247 = 13 * 19 := by norm_num,
  have prime13 : nat.prime 13 := by norm_num,
  have prime19 : nat.prime 19 := by norm_num,
  use 13,
  use 19,
  exact ⟨prime19, ⟨prime13, ⟨fact247, by linarith⟩⟩⟩,
end

end greatest_prime_factor_of_247_l800_800516


namespace perimeter_parallel_triangle_l800_800509

theorem perimeter_parallel_triangle (AB BC AC : ℝ) (lA lB lC : ℝ) 
  (hAB : AB = 120) (hBC : BC = 220) (hAC : AC = 180) 
  (h_parallel_A : lA = 55) (h_parallel_B : lB = 45) (h_parallel_C : lC = 15) : 
  let rA := lA / BC
  let rB := lB / AC
  let rC := lC / AB
  let factor := (rA + rB + rC) * (BC / 220)
  (factor * (AB + BC + AC) = 715) := 
begin
  sorry
end

end perimeter_parallel_triangle_l800_800509


namespace ones_digit_of_six_power_l800_800905

theorem ones_digit_of_six_power (n : ℕ) (hn : n ≥ 1) : (6 ^ n) % 10 = 6 :=
by
  sorry

example : (6 ^ 34) % 10 = 6 :=
by
  have h : 34 ≥ 1 := by norm_num
  exact ones_digit_of_six_power 34 h

end ones_digit_of_six_power_l800_800905


namespace probability_divisible_by_5_l800_800316

def is_five_digit_number (n : ℕ) : Prop :=
  n ≥ 10000 ∧ n < 100000

def sum_of_digits_is_40 (n : ℕ) : Prop :=
  (n.digits 10).sum = 40

def first_digit_is_even (n : ℕ) : Prop :=
  let d := (n.digits 10).reverse.head!
  in d % 2 = 0

def divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

theorem probability_divisible_by_5 :
  ∃ (s : finset ℕ), 
  (∀ n ∈ s, is_five_digit_number n) ∧ 
  (∀ n ∈ s, sum_of_digits_is_40 n) ∧ 
  (∀ n ∈ s, first_digit_is_even n) ∧ 
  (∃ (t : finset ℕ), 
     (∀ m ∈ t, m ∈ s) ∧ 
     (∀ m ∈ t, divisible_by_5 m) ∧ 
     (t.card * 13 = s.card * 4)) :=
begin
  sorry
end

end probability_divisible_by_5_l800_800316


namespace largest_valid_four_digit_number_l800_800065

-- Definition of the problem conditions
def is_valid_number (a b c d : ℕ) : Prop :=
  c = a + b ∧ d = b + c

-- Proposition that we need to prove
theorem largest_valid_four_digit_number : ∃ (a b c d : ℕ),
  is_valid_number a b c d ∧ a * 1000 + b * 100 + c * 10 + d = 9099 :=
by
  sorry

end largest_valid_four_digit_number_l800_800065


namespace unique_two_digit_integer_l800_800501

theorem unique_two_digit_integer (t : ℕ) (h : 11 * t % 100 = 36) (ht : 10 ≤ t ∧ t ≤ 99) : t = 76 :=
by
  sorry

end unique_two_digit_integer_l800_800501


namespace find_b3_l800_800875

variable {b : ℕ → ℚ}

-- Conditions
def b1 : Prop := b 1 = 23
def b10 : Prop := b 10 = 115
def arithmetic_mean (n : ℕ) : Prop := ∀ n ≥ 4, b n = (∑ i in Finset.range (n - 2), b (i + 1)) / (n - 2)
def b3_formula : Prop := b 3 = (23 + b 2) / 2

-- Goal
theorem find_b3 : b1 ∧ b10 ∧ arithmetic_mean → b3_formula → b 3 = 115 / 3 :=
by
  sorry

end find_b3_l800_800875


namespace area_of_triangle_l800_800772

theorem area_of_triangle
  (X Y Z U V : Type)
  (triangle : Triangle X Y Z)
  (median_XU : Median triangle X U)
  (median_YV : Median triangle Y V)
  (intersect_right_angle : ∠ XU YV = 90)
  (XU_length : length XU = 18)
  (YV_length : length YV = 24) :
  area triangle = 288 :=
sorry

end area_of_triangle_l800_800772


namespace f_90_eq_999_l800_800002

def f : ℕ → ℕ
| n := if n ≥ 1000 then n - 3 else f (f (n + 7))

theorem f_90_eq_999 : f 90 = 999 :=
sorry

end f_90_eq_999_l800_800002


namespace bridge_length_l800_800538

theorem bridge_length 
  (train_length : ℕ) 
  (speed_km_hr : ℕ) 
  (cross_time_sec : ℕ) 
  (conversion_factor_num : ℕ) 
  (conversion_factor_den : ℕ)
  (expected_length : ℕ) 
  (speed_m_s : ℕ := speed_km_hr * conversion_factor_num / conversion_factor_den)
  (total_distance : ℕ := speed_m_s * cross_time_sec) :
  train_length = 150 →
  speed_km_hr = 45 →
  cross_time_sec = 30 →
  conversion_factor_num = 1000 →
  conversion_factor_den = 3600 →
  expected_length = 225 →
  total_distance - train_length = expected_length :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end bridge_length_l800_800538


namespace min_AB_CD_l800_800278

theorem min_AB_CD {p : ℝ} (p_pos : p > 0) :
  ∀ (A B C D : ℝ × ℝ), on_parabola A B C D  &&
  mutually_perpendicular A B C D -> passing_through_origin A B C D -> 
  |AB|  + |CD | = 16 * p.
sorry

end min_AB_CD_l800_800278


namespace hyperbola_equation_is_correct_l800_800726

variables {x y m n : ℝ}

def hyperbola_equation (m n : ℝ) : ℝ := (x^2 / m^2 + y^2 / n^2)

def eccentricity (c a : ℝ) : ℝ := c / a

def focus_of_parabola : ℝ := 2

def hyperbola_focus (c : ℝ) : Prop := c = focus_of_parabola

def hyperbola_eccentricity (e c n : ℝ) : Prop := e = c / n

theorem hyperbola_equation_is_correct :
  ∀ (m n : ℝ), 
  hyperbola_focus 2 →
  hyperbola_eccentricity (2 * real.sqrt 3 / 3) 2 n →
  n = 3 →
  m = (-1) →
  (hyperbola_equation m n = 1) =
  (y^2 / 3 - x^2 = 1) :=
by sorry

end hyperbola_equation_is_correct_l800_800726


namespace number_of_valid_six_digit_numbers_l800_800220

def is_valid_number (n : Fin 1000000) : Prop :=
  let digits := [1,2,3,4,5,6]
  let even_digits := [2,4,6]
  let d_str := toDigits 10 n
  n.toDigits 10 == digits ∧ 
  d_str.head ≠ 1 ∧ 
  d_str.last ≠ 1 ∧ 
  (d_str.filter (λ x, x ∈ even_digits)).length = 3 ∧
  (adjacent_pairs d_str).count (λ pr, pr.1 ∈ even_digits ∧ pr.2 ∈ even_digits) = 1

def num_six_digit_numbers_satisfying_conditions : Nat :=
  (List.range 999999).count is_valid_number

theorem number_of_valid_six_digit_numbers : num_six_digit_numbers_satisfying_conditions = 288 :=
  sorry

end number_of_valid_six_digit_numbers_l800_800220


namespace sum_of_possible_values_B_l800_800182

theorem sum_of_possible_values_B :
  (∑ b in ({b | b ∈ Finset.range 10 ∧ (32 * 10 + b) % 8 = 0} : Finset ℕ), b) = 12 := 
sorry

end sum_of_possible_values_B_l800_800182


namespace range_of_a_l800_800681

noncomputable def f (a x : ℝ) : ℝ :=
  (Matrix.det ![
  ![a * x, x],
  ![-2, 2 * x]] : ℝ)

def g (x : ℝ) : ℝ :=
  (2 * x^2 + 1) / x

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 ∈ Set.Icc (1 : ℝ) (4 : ℝ), f a x1 ≤ g x2) →
  a ≤ - (1 / 6) :=
sorry

end range_of_a_l800_800681


namespace largest_four_digit_number_property_l800_800067

theorem largest_four_digit_number_property : ∃ (a b c d : Nat), 
  (1000 * a + 100 * b + 10 * c + d = 9099) ∧ 
  (c = a + b) ∧ 
  (d = b + c) ∧ 
  1 ≤ a ∧ a ≤ 9 ∧ 
  0 ≤ b ∧ b ≤ 9 ∧ 
  0 ≤ c ∧ c ≤ 9 ∧ 
  0 ≤ d ∧ d ≤ 9 := 
sorry

end largest_four_digit_number_property_l800_800067


namespace real_root_complex_solution_l800_800851

noncomputable def complex_number (a b : ℝ) := a + b * complex.i

theorem real_root_complex_solution :
  (∃ (a : ℝ), ∃ (b : ℝ), b ∈ set_of (λ x, (x^2 + (4 + complex.i) * x + (4 + a * complex.i) = 0))
    ∧ complex_number a b = 2 - 2 * complex.i) :=
begin
  sorry
end

end real_root_complex_solution_l800_800851


namespace line_intersection_l800_800989

theorem line_intersection : 
  ∃ (x y : ℚ), 
    8 * x - 5 * y = 10 ∧ 
    3 * x + 2 * y = 16 ∧ 
    x = 100 / 31 ∧ 
    y = 98 / 31 :=
by
  use 100 / 31
  use 98 / 31
  sorry

end line_intersection_l800_800989


namespace sandwich_combinations_l800_800141

theorem sandwich_combinations (meats cheeses : Finset ℕ) (h_meats : meats.card = 12) (h_cheeses : cheeses.card = 8) :
  (meats.card.choose 2) * cheeses.card.choose 1 = 528 :=
by
  simp only [Finset.card_choose, h_meats, h_cheeses]
  have h1 : 12.choose 2 = 66 := by norm_num
  have h2 : 8.choose 1 = 8 := by norm_num
  rw [h1, h2]
  norm_num
  exact rfl

end sandwich_combinations_l800_800141


namespace number_of_divisible_45_format_90_is_11_l800_800743

theorem number_of_divisible_45_format_90_is_11 :
  ∃! n, (∃ ab, n = 1000 * ab + 90 ∧ 1000 <= n ∧ n < 10000 ∧ ∃ k, ab = 9 * k ∧ 1 ≤ k ∧ k ≤ 11) :=
begin
  sorry
end

end number_of_divisible_45_format_90_is_11_l800_800743


namespace overlapping_triangle_area_l800_800693

/-- Given a rectangle with length 8 and width 4, folded along its diagonal, 
    the area of the overlapping part (grey triangle) is 10. --/
theorem overlapping_triangle_area : 
  let length := 8 
  let width := 4 
  let diagonal := (length^2 + width^2)^(1/2) 
  let base := (length^2 / (width^2 + length^2))^(1/2) * width 
  let height := width
  1 / 2 * base * height = 10 := by 
  sorry

end overlapping_triangle_area_l800_800693


namespace max_value_PA_dot_PB_div_cos_l800_800358

variables (A B C D P : Type)
variables (AB BC : ℝ) (α β : ℝ)

def is_rectangle (ABCD : Type) (A B C D : ABCD): Prop :=
  true -- Definition of rectangle can be further defined if necessary

def PA (P A : Type) : ℝ := -- Length PA can be defined in geometric terms
sorry

def PB (P B : Type) : ℝ := -- Length PB can be defined in geometric terms
sorry

def dot_product_PA_PB (PA PB : ℝ) : ℝ :=
sorry

def cos_sum_angles (α β : ℝ) : ℝ :=
Math.cos (α + β)

theorem max_value_PA_dot_PB_div_cos (h1 : is_rectangle ABCD A B C D)
  (h2 : AB = 3) (h3 : BC = 1)
  (h4 : ∃ (P : ABCD), P ∈ side_CD)
  (h5 : ∃ (α β : ℝ), angle P A B = α ∧ angle P B A = β):
  ∃ (max_val : ℝ), max_val = (dot_product_PA_PB (PA P A) (PB P B)) / (cos_sum_angles α β) ∧ max_val = 3 :=
sorry

end max_value_PA_dot_PB_div_cos_l800_800358


namespace largest_four_digit_number_l800_800059

theorem largest_four_digit_number :
  ∃ a b c d : ℕ, 
    9 < 1000 * a + 100 * b + 10 * c + d ∧ 
    1000 * a + 100 * b + 10 * c + d < 10000 ∧ 
    c = a + b ∧ 
    d = b + c ∧ 
    1000 * a + 100 * b + 10 * c + d = 9099 :=
by {
  sorry
}

end largest_four_digit_number_l800_800059


namespace olympiad_scores_l800_800353

theorem olympiad_scores (scores : Fin 20 → ℕ) 
  (uniqueScores : ∀ i j, i ≠ j → scores i ≠ scores j)
  (less_than_sum_of_others : ∀ i j k, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k) :
  ∀ i, scores i > 18 := 
by sorry

end olympiad_scores_l800_800353


namespace largest_four_digit_number_prop_l800_800053

theorem largest_four_digit_number_prop :
  ∃ (a b c d : ℕ), a = 9 ∧ b = 0 ∧ c = 9 ∧ d = 9 ∧ (1000 * a + 100 * b + 10 * c + d = 9099) ∧ (c = a + b) ∧ (d = b + c) :=
by
  sorry

end largest_four_digit_number_prop_l800_800053


namespace problem_I_problem_II_l800_800700

variable (x a m : ℝ)

theorem problem_I (h: ¬ (∃ r : ℝ, r^2 - 2*a*r + 2*a^2 - a - 6 = 0)) : 
  a < -2 ∨ a > 3 := by
  sorry

theorem problem_II (p : ∃ r : ℝ, r^2 - 2*a*r + 2*a^2 - a - 6 = 0) (q : m-1 ≤ a ∧ a ≤ m+3) :
  ∀ a : ℝ, -2 ≤ a ∧ a ≤ 3 → m ∈ [-1, 0] := by
  sorry

end problem_I_problem_II_l800_800700


namespace find_cost_price_l800_800015

variable (cost_price : ℝ)
variable (selling_price : ℝ := 200)
variable (discount_rate : ℝ := 0.1)
variable (profit_rate : ℝ := 0.2)

theorem find_cost_price:
  let discounted_price := selling_price * (1 - discount_rate)
  let cost_price_with_profit := cost_price * (1 + profit_rate)
  discounted_price = cost_price_with_profit → cost_price = 150 :=
begin
  sorry
end

end find_cost_price_l800_800015


namespace arithmetic_sequence_problem_l800_800027

theorem arithmetic_sequence_problem
  (a : ℕ → ℤ)  -- the arithmetic sequence
  (S : ℕ → ℤ)  -- the sum of the first n terms
  (m : ℕ)      -- the m in question
  (h1 : a (m - 1) + a (m + 1) - a m ^ 2 = 0)
  (h2 : S (2 * m - 1) = 18) :
  m = 5 := 
sorry

end arithmetic_sequence_problem_l800_800027


namespace largest_valid_four_digit_number_l800_800063

-- Definition of the problem conditions
def is_valid_number (a b c d : ℕ) : Prop :=
  c = a + b ∧ d = b + c

-- Proposition that we need to prove
theorem largest_valid_four_digit_number : ∃ (a b c d : ℕ),
  is_valid_number a b c d ∧ a * 1000 + b * 100 + c * 10 + d = 9099 :=
by
  sorry

end largest_valid_four_digit_number_l800_800063


namespace problem_statement_l800_800378

def g (a b c : ℝ) : ℝ :=
  if a + b + c ≤ 5 then (a * b - a * c + c) / (2 * a - c)
  else (a * b - b * c - c) / (-2 * b + c)

theorem problem_statement : g 3 2 0 + g 1 3 2 = 2.25 := 
by 
  -- We'll skip the proof here.
  sorry

end problem_statement_l800_800378


namespace derivative_of_x_log_x_l800_800658

noncomputable def y (x : ℝ) := x * Real.log x

theorem derivative_of_x_log_x (x : ℝ) (hx : 0 < x) :
  (deriv y x) = Real.log x + 1 :=
sorry

end derivative_of_x_log_x_l800_800658


namespace find_range_of_a_l800_800262

variables (a : ℝ)

def proposition_p : Prop :=
∀ x : ℝ, a * x^2 + a * x + 1 > 0

def proposition_q : Prop :=
∀ x : ℝ, x ≥ 1 →  8 * x - a ≥ 0

theorem find_range_of_a
  (h1 : proposition_p ∨ proposition_q)
  (h2 : ¬proposition_p) :
  a ≤ 0 ∨ 4 ≤ a ∧ a ≤ 8 :=
by {
  sorry
}

end find_range_of_a_l800_800262


namespace probability_students_from_different_grades_l800_800571

theorem probability_students_from_different_grades :
  let total_students := 4
  let first_grade_students := 2
  let second_grade_students := 2
  (2 from total_students are selected) ->
  (2 from total_students are from different grades) ->
  ℝ :=
by 
  sorry

end probability_students_from_different_grades_l800_800571


namespace range_of_m_l800_800682

variable (m : ℝ) (a : ℝ)

def p := ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 2 * x - 2 ≥ m^2 - 3 * m

def q := ∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ m ≤ a * x

theorem range_of_m (h₁ : ¬p m) (h₂ : ¬q m a) (h₃ : p m ∨ q m a) (ha : a = 1) :
  m ∈ set.Ioo 1 2 ∪ set.Iio 1 := sorry

end range_of_m_l800_800682


namespace comparison_l800_800680

noncomputable def a : ℝ := real.log 2 / real.log 3
noncomputable def b : ℝ := (real.log 2 + real.log 5) / (real.log 3 + real.log 5)
noncomputable def c : ℝ := real.sin (1 / 2)

theorem comparison (a b c : ℝ) 
  (ha : a = real.log 2 / real.log 3) 
  (hb : b = (real.log 2 + real.log 5) / (real.log 3 + real.log 5))
  (hc : c = real.sin (1 / 2)) : b > a ∧ a > c := by
sorry

end comparison_l800_800680


namespace triangle_area_l800_800795

noncomputable theory
open Real

variables (a b c : ℝ) (B : ℝ)

def area_of_triangle_ABC (B : ℝ) (a : ℝ) (c : ℝ) : ℝ :=
  0.5 * a * c * sin B

theorem triangle_area : 
  B = (60 : ℝ) * π / 180 → c = 3 → b = sqrt 7 → 
  (∃ a, (b * b = a * a + c * c - 2 * a * c * cos B) ∧ (area_of_triangle_ABC B a c = 3 * sqrt 3 / 4 ∨ area_of_triangle_ABC B a c = 3 * sqrt 3 / 2)) :=
by
  intros hB hc hb
  -- Proof can be completed here using further derivations and steps.
  sorry

end triangle_area_l800_800795


namespace sin_D_eq_1_l800_800360

-- Definitions of the triangle and angle conditions
variables {D E F : Type} [Field D]
variables [IsRightTriangle : RightTriangle DEF {angleD := 90}] (DE : Real) (EF : Real)

-- Given conditions
#check IsRightTriangle
def angD := (90 : Real)
def DE := (12 : Real)
def EF := (35 : Real)

-- The theorem to prove
theorem sin_D_eq_1 (h : angD = 90) : sin angD = 1 := by
  sorry

end sin_D_eq_1_l800_800360


namespace sum_of_areas_l800_800363

section
variables {p : ℝ} (h_pos : 0 ≤ p ∧ p ≤ 15)

theorem sum_of_areas (h_pos : 0 ≤ p ∧ p ≤ 15) : 
  let area_DOE := (1 / 2) * p * 15 in
  let OF := Real.sqrt ((5 - 0)^2 + (15 - 0)^2) in
  let area_EOF := (1 / 2) * 15 * OF in
  area_DOE + area_EOF = (15 * (p + 5 * Real.sqrt(10))) / 2 :=
begin
  sorry
end
end

end sum_of_areas_l800_800363


namespace repayment_difference_l800_800145

noncomputable def compounded_repayment (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

def simple_repayment (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r * t)

theorem repayment_difference {P : ℝ} {r1 r2 : ℝ} {n t : ℕ}
  (hP : P = 12000)
  (hr1 : r1 = 0.08)
  (hr2 : r2 = 0.10)
  (hn : n = 2)
  (ht : t = 12)
  (ht_half : t / 2 = 6) :
  abs ((simple_repayment P r2 t) - (compounded_repayment (compounded_repayment P r1 n (t / 2) / 2) r1 n (t / 2) + (compounded_repayment P r1 n (t / 2) / 2))) = 3901 :=
  sorry

end repayment_difference_l800_800145


namespace sum_binom_2024_mod_2027_l800_800870

theorem sum_binom_2024_mod_2027 :
  let T := ∑ k in Finset.range 65, Nat.choose 2024 k
  2027.prime →
  T % 2027 = 1089 :=
by
  intros T hp
  sorry

end sum_binom_2024_mod_2027_l800_800870


namespace power_subtraction_modulo_7_l800_800172

theorem power_subtraction_modulo_7 : 
    (47^1357 - 23^1357) % 7 = 3 :=
by {
  have h47 : 47 % 7 = 5, by norm_num,
  have h23 : 23 % 7 = 2, by norm_num,
  sorry
}

end power_subtraction_modulo_7_l800_800172


namespace find_special_integer_l800_800655

theorem find_special_integer :
  ∃ (n : ℕ), n > 0 ∧
    let a := n.digits 10 in
    a.length ≥ 2 ∧
    (let m := a.length - 1 in
     (a.head!.at 1) * 10^m + (a.head!.first).val * 10^(m-1) +
      (n.div10 (n.div10 n)) * 10^2 = 2 * n) :=
sorry

end find_special_integer_l800_800655


namespace parallel_lines_value_of_m_l800_800859

theorem parallel_lines_value_of_m (m : ℝ) 
  (h1 : ∀ x y : ℝ, x + m * y - 2 = 0 = (2 * x + (1 - m) * y + 2 = 0)) : 
  m = 1 / 3 :=
by {
  sorry
}

end parallel_lines_value_of_m_l800_800859


namespace total_bill_is_60_l800_800631

def num_adults := 6
def num_children := 2
def cost_adult := 6
def cost_child := 4
def cost_soda := 2

theorem total_bill_is_60 : num_adults * cost_adult + num_children * cost_child + (num_adults + num_children) * cost_soda = 60 := by
  sorry

end total_bill_is_60_l800_800631


namespace sequence_general_term_l800_800694

theorem sequence_general_term (a : ℕ → ℝ) (S : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ) :
  (a 2 = 4) →
  (∀ n : ℕ, n > 0 → a (n + 1) = a n + 2) →
  (∀ n : ℕ, S n = ∑ k in finset.range n, a (k + 1)) →
  (∀ n, b n = 1 / S n) →
  (∀ n, T n = ∑ k in finset.range n, b (k + 1)) →
  (∀ n, a n = 2 * n) ∧ (∀ n, T n = n / (n + 1)) :=
begin
  intros ha2 ha_step hS hb hT,
  sorry
end

end sequence_general_term_l800_800694


namespace linear_function_graph_not_in_second_quadrant_l800_800364

open Real

theorem linear_function_graph_not_in_second_quadrant 
  (k b : ℝ) (h1 : k > 0) (h2 : b < 0) :
  ¬ ∃ (x y : ℝ), x < 0 ∧ y > 0 ∧ y = k * x + b := 
sorry

end linear_function_graph_not_in_second_quadrant_l800_800364


namespace jack_total_books_is_541_l800_800798

-- Define the number of books in each section
def american_books : ℕ := 6 * 34
def british_books : ℕ := 8 * 29
def world_books : ℕ := 5 * 21

-- Define the total number of books based on the given sections
def total_books : ℕ := american_books + british_books + world_books

-- Prove that the total number of books is 541
theorem jack_total_books_is_541 : total_books = 541 :=
by
  sorry

end jack_total_books_is_541_l800_800798


namespace parallelogram_with_right_angle_is_rectangle_l800_800317

-- Definitions for the mathematical proof problem
def is_parallelogram (P : Type) [quadrilateral P] : Prop :=
  (∀ A B C D : P, parallel (segment A B) (segment C D) ∧ parallel (segment B C) (segment D A)) ∧
  (∀ A B C D : P, length (segment A B) = length (segment C D) ∧ length (segment B C) = length (segment D A))

def is_rectangle (P : Type) [quadrilateral P] : Prop :=
  is_parallelogram P ∧ (∀ A B C D : P, right_angle (angle A B C) ∧ right_angle (angle B C D) ∧ right_angle (angle C D A) ∧ right_angle (angle D A B))

-- The problem statement in Lean 4
theorem parallelogram_with_right_angle_is_rectangle
  (P : Type) [quadrilateral P] (A B C D : P)
  (parallelogram_P : is_parallelogram P)
  (right_angle_A : right_angle (angle A B C)) : is_rectangle P :=
by
  sorry

end parallelogram_with_right_angle_is_rectangle_l800_800317


namespace min_value_of_3x_plus_4y_l800_800307

open Real

theorem min_value_of_3x_plus_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 5 * x * y) : 3 * x + 4 * y ≥ 5 :=
sorry

end min_value_of_3x_plus_4y_l800_800307


namespace coloring_minimum_colors_l800_800542

theorem coloring_minimum_colors (n : ℕ) (h : n = 2016) : ∃ k : ℕ, k = 11 ∧
  (∀ (board : Matrix ℕ ℕ ℕ), 
    let color : ℕ → ℕ → ℕ := λ i j, if i = j then 0           -- Main diagonal colored '0'
                                      else if i < j then j - i -- Left of the diagonal
                                      else i - j               -- Right of the diagonal
    ∧ (∀ i < n, ∀ j < n, color i j = color j i)                -- Symmetry wrt diagonal
    ∧ (∀ i < n, ∀ j k, j ≠ k → color i j ≠ color i k)          -- Different colors in the row  
    ) :=
begin
  use 11,
  split,
  { refl },
  { intros board,
    let color := λ i j, if i = j then 0 else if i < j then j - i else i - j,
    split,
    { intros i hi j hj,
      exact (if i < j then rfl else rfl), },
    { intros i hi j k hjk,
      by_cases h : i = j,
      { rw h,
        exact hjk.elim, },
      { by_cases h' : j < k,
        { rw if_pos h' },
        { by_contradiction,
          have := nat.le_antisymm,
          contradiction } } } }
end

end coloring_minimum_colors_l800_800542


namespace grade_assignment_ways_l800_800944

/-- Define the number of students and the number of grade choices -/
def num_students : ℕ := 15
def num_grades : ℕ := 4

/-- Define the total number of ways to assign grades -/
def total_ways : ℕ := num_grades ^ num_students

/-- Prove that the total number of ways to assign grades is 4^15 -/
theorem grade_assignment_ways : total_ways = 1073741824 := by
  -- proof here
  sorry

end grade_assignment_ways_l800_800944


namespace hyperbola_standard_eq_and_line_eq_l800_800689

-- Problem Statement
theorem hyperbola_standard_eq_and_line_eq
  (F1 F2 : ℝ × ℝ) (e : ℝ) (M : ℝ × ℝ) (x y : ℝ)
  (hF1 : F1 = (-2, 0))
  (hF2 : F2 = (2, 0))
  (he : e = 2)
  (hM : M = (1, 3))
  : (∃ a b : ℝ, a = 1 ∧ b = sqrt 3 ∧ (x^2 / a^2) - (y^2 / b^2) = 1) ∧ (∃ l m : ℝ, l = 1 ∧ m = 2 ∧ y = l * x + m) :=
sorry

end hyperbola_standard_eq_and_line_eq_l800_800689


namespace total_distance_traveled_l800_800413

def trip_duration : ℕ := 8
def speed_first_half : ℕ := 70
def speed_second_half : ℕ := 85
def time_each_half : ℕ := trip_duration / 2

theorem total_distance_traveled :
  let distance_first_half := time_each_half * speed_first_half
  let distance_second_half := time_each_half * speed_second_half
  let total_distance := distance_first_half + distance_second_half
  total_distance = 620 := by
  sorry

end total_distance_traveled_l800_800413


namespace rhombus_perimeter_l800_800471

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 20) (h2 : d2 = 16) : ∃ p, p = 8 * Real.sqrt 41 := by
  let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  have h3 : s = 2 * Real.sqrt 41 := by sorry
  let p := 4 * s
  have h4 : p = 8 * Real.sqrt 41 := by sorry
  exact ⟨p, h4⟩

end rhombus_perimeter_l800_800471


namespace gaussian_guardians_points_l800_800492

theorem gaussian_guardians_points :
  let Daniel := 7
  let Curtis := 8
  let Sid := 2
  let Emily := 11
  let Kalyn := 6
  let Hyojeong := 12
  let Ty := 1
  let Winston := 7
  Daniel + Curtis + Sid + Emily + Kalyn + Hyojeong + Ty + Winston = 54 :=
by
  sorry

end gaussian_guardians_points_l800_800492


namespace least_whole_number_for_ratio_l800_800517

theorem least_whole_number_for_ratio :
  ∃ x : ℕ, (6 - x) * 21 < (7 - x) * 16 ∧ x = 3 :=
by
  sorry

end least_whole_number_for_ratio_l800_800517


namespace integer_solutions_count_is_9_l800_800219

noncomputable def integer_solutions_count : Nat :=
  { x : Int | x^2 < 9 * x + 6 }.card

theorem integer_solutions_count_is_9 : integer_solutions_count = 9 := by
  sorry

end integer_solutions_count_is_9_l800_800219


namespace samantha_total_cost_l800_800160

-- Defining the conditions in Lean
def washer_cost : ℕ := 4
def dryer_cost_per_10_min : ℕ := 25
def loads : ℕ := 2
def num_dryers : ℕ := 3
def dryer_time : ℕ := 40

-- Proving the total cost Samantha spends is $11
theorem samantha_total_cost : (loads * washer_cost + num_dryers * (dryer_time / 10 * dryer_cost_per_10_min)) = 1100 :=
by
  sorry

end samantha_total_cost_l800_800160


namespace ben_and_sara_tie_fraction_l800_800044

theorem ben_and_sara_tie_fraction (ben_wins sara_wins : ℚ) (h1 : ben_wins = 2 / 5) (h2 : sara_wins = 1 / 4) : 
  1 - (ben_wins + sara_wins) = 7 / 20 :=
by
  rw [h1, h2]
  norm_num

end ben_and_sara_tie_fraction_l800_800044


namespace coinCombinationCount_l800_800783

-- Definitions for the coin values and the target amount
def quarter := 25
def dime := 10
def nickel := 5
def penny := 1
def total := 400

-- Define a function counting the number of ways to reach the total using given coin values
def countWays : Nat := sorry -- placeholder for the actual computation

-- Theorem stating the problem statement
theorem coinCombinationCount (n : Nat) :
  countWays = n :=
sorry

end coinCombinationCount_l800_800783


namespace a1_value_b_arithmetic_seq_sum_reciprocal_lt_one_l800_800184

-- Let's define the sequences and conditions
variable {b : ℕ → ℝ}
variable {a : ℕ → ℝ}

-- Condition: b_{n+1} = a_{n+1} - 1
def b_def (n : ℕ) : Prop := b(n + 1) = a(n + 1) - 1

-- Prove: a_1 = 3
theorem a1_value : a 1 = 3 := sorry

-- Prove: sequence {b} is an arithmetic sequence
theorem b_arithmetic_seq (h : ∀ n, b_def n) : ∃ d, ∀ n, b (n + 1) - b n = d := sorry

-- Assume given general term formula for a_n (we need some way to capture the formula)
variable (d a1 : ℝ)

-- Prove: sum of reciprocals of {a_n} is less than 1
theorem sum_reciprocal_lt_one : ∀ n, ∑ i in range n, 1 / a i < 1 := sorry

end a1_value_b_arithmetic_seq_sum_reciprocal_lt_one_l800_800184


namespace length_of_hypotenuse_l800_800337

noncomputable def right_triangle_hypotenuse_length (AB AC : ℝ) (h₀ : AB = 1) (h₁ : AC = 2) (angle_ACB : Real.Angle) (h₂ : angle_ACB = Real.Angle.pi_div_two)
  (D E : ℝ) (hypotenuse_division : BC : ℝ) (h₃ : D = BC / 3) (h₄ : E = 2 * BC / 3)
  (x : ℝ) (h₅ : 0 < x ∧ x < Real.Angle.pi_div_two) (AD AE : ℝ) (h₆ : AD = Real.tan x) (h₇ : AE = Real.cot x) : Prop :=
  ∃ BC : ℝ, BC = Real.sqrt (AB^2 + AC^2)
  
theorem length_of_hypotenuse : right_triangle_hypotenuse_length 1 2 1 rfl Real.Angle.pi_div_two rfl (BC / 3) (2 * BC / 3) BC (0 < x ∧ x < Real.Angle.pi_div_two) (Real.tan x) (Real.cot x)    := 
begin
  use Real.sqrt(1^2 + 2^2),
  sorry
end

end length_of_hypotenuse_l800_800337


namespace rainfall_in_April_l800_800774

theorem rainfall_in_April (rainfall_March : ℝ) (rainfall_diff : ℝ) (h1 : rainfall_March = 0.81) (h2 : rainfall_diff = 0.35) : 
  (0.81 - 0.35 = 0.46) :=
by {
  rw [h1, h2],
  norm_num,
  sorry
}

end rainfall_in_April_l800_800774


namespace simplify_fraction_l800_800971

-- We start by defining the problem in Lean.
theorem simplify_fraction :
  (1722^2 - 1715^2) / (1730^2 - 1705^2) = 7 / 25 :=
by
  -- Begin proof sketch (proof is not required, so we put sorry here)
  sorry

end simplify_fraction_l800_800971


namespace area_change_factor_l800_800123

theorem area_change_factor (k b : ℝ) (hk : 0 < k) (hb : 0 < b) :
  let S1 := (b * b) / (2 * k)
  let S2 := (b * b) / (16 * k)
  S1 / S2 = 8 :=
by
  sorry

end area_change_factor_l800_800123


namespace puppies_sold_theorem_l800_800605

variable (initial_puppies : ℕ) (cages_used : ℕ) (puppies_per_cage : ℕ)

def puppies_sold (initial_puppies cages_used puppies_per_cage : ℕ) : ℕ :=
  initial_puppies - (cages_used * puppies_per_cage)
  
theorem puppies_sold_theorem (initial_puppies cages_used puppies_per_cage : ℕ) :
  initial_puppies = 18 → cages_used = 3 → puppies_per_cage = 5 → puppies_sold initial_puppies cages_used puppies_per_cage = 3 := by
  intro h_initial h_cages h_puppies_per_cage
  simp [puppies_sold, h_initial, h_cages, h_puppies_per_cage]
  sorry

end puppies_sold_theorem_l800_800605


namespace line_through_intersection_and_parallel_l800_800474

theorem line_through_intersection_and_parallel :
  (∃ (x y : ℝ), x + y = 9 ∧ 2 * x - y = 18 ∧ (3 * x - 2 * y - 27 = 0)) :=
begin
  sorry
end

end line_through_intersection_and_parallel_l800_800474


namespace question1_question2_l800_800272

-- Given function and conditions
def f (ω φ x : ℝ) := Real.sin (ω * x + φ)
axiom ω_pos : ω > 0
axiom φ_range : 0 ≤ φ ∧ φ ≤ π

-- Given properties
axiom f_even : ∀ x : ℝ, f ω φ x = f ω φ (-x)
axiom dist_adj_high_low : ∀ x1 x2 : ℝ, dist_adj_high_low_condition x1 x2

-- Prove the function and the simplified expression
theorem question1 : f 1 (π / 2) = Real.cos := sorry

theorem question2 (α : ℝ) (h : Real.sin α + f 1 (π / 2) α = 2/3) :
  (Real.sqrt 2 * Real.sin (2 * α - π / 4) + 1) /
  (1 + Real.tan α) = -5/9 := sorry

end question1_question2_l800_800272


namespace ratio_of_area_of_midpoint_region_to_square_l800_800043

theorem ratio_of_area_of_midpoint_region_to_square 
  (v : ℝ) 
  (hv : 0 < v) 
  (s : ℝ) 
  (hs : 0 < s) 
  (square_abc_side_length : 1) 
  (ABCD_path : ℝ → (ℝ × ℝ)) 
  (first_particle_start : (0, 0)) 
  (second_particle_start : (s, 0)) 
  (first_particle_velocity : v) 
  (second_particle_velocity : 2 * v) : 
  (∃ R : set (ℝ × ℝ), 
   let area_R := sorry, 
       area_square_ABC := s ^ 2 in 
   area_R / area_square_ABC = 1 / 8) :=
sorry

end ratio_of_area_of_midpoint_region_to_square_l800_800043


namespace measure_of_angle_A_length_of_side_c_l800_800328

variables {A B C : ℝ} {a b c : ℝ}

-- Given conditions
def condition1 (a b c : ℝ) (C : ℝ) : Prop :=
  2 * a * Real.cos C = 2 * b - c

def condition2 (a b : ℝ) : Prop :=
  a = Real.sqrt 21 ∧ b = 4

-- Theorem statements (proofs are omitted)
theorem measure_of_angle_A (h1: condition1 a b c C) : 
  A = Real.pi / 3 :=
sorry

theorem length_of_side_c (h1: condition1 a b c C) (h2: condition2 a b) : 
  c = 5 :=
sorry

end measure_of_angle_A_length_of_side_c_l800_800328


namespace volleyball_team_math_l800_800635

theorem volleyball_team_math 
  (total : ℕ) (physics_only: ℕ) (both_subjects : ℕ) (at_least_one : total = 25)
  (total_physics : physics_only + both_subjects = 10) (total_players : physics_only + both_subjects + (total - physics_only - both_subjects - total_physics) = total)
  : (total - physics_only = 21) :=
by
  have h1 : total = 25 := at_least_one
  have h2 : total_physics = 10, sorry
  have h3 : total - physics_only = 21, sorry
  exact h3

end volleyball_team_math_l800_800635


namespace triangle_ratio_l800_800368

theorem triangle_ratio (A B C a b c: ℝ) 
  (h1 : a = sqrt 2 * c)
  (h2 : (sqrt 2 * a - b) * tan B = b * tan C)
  (habc : a^2 + b^2 > c^2) : 
  b = c :=
by sorry

end triangle_ratio_l800_800368


namespace problem1_problem2_l800_800230

noncomputable def problem1_min_value (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : a + b + c = 1) : ℝ := 
(a + 1)^2 + 4 * b^2 + 9 * c^2

theorem problem1 (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : a + b + c = 1) : 
  problem1_min_value a b c h₀ h₁ h₂ h₃ = 144 / 49 := sorry

theorem problem2 (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : a + b + c = 1) :
  1 / (sqrt a + sqrt b) + 1 / (sqrt b + sqrt c) + 1 / (sqrt c + sqrt a) ≥ 3 * sqrt 3 / 2 := sorry

end problem1_problem2_l800_800230


namespace congruence_problem_l800_800843

theorem congruence_problem (x : ℤ)
  (h1 : 2 + x ≡ 9 [MOD 16])
  (h2 : 3 + x ≡ 8 [MOD 81])
  (h3 : 4 + x ≡ 27 [MOD 8]) :
  x ≡ 23 [MOD 24] :=
sorry

end congruence_problem_l800_800843


namespace same_answer_l800_800653

structure Person :=
(name : String)
(tellsTruth : Bool)

def Fedya : Person :=
{ name := "Fedya",
  tellsTruth := true }

def Vadim : Person :=
{ name := "Vadim",
  tellsTruth := false }

def question (p : Person) (q : String) : Bool :=
if p.tellsTruth then q = p.name else q ≠ p.name

theorem same_answer (q : String) :
  (question Fedya q = question Vadim q) :=
sorry

end same_answer_l800_800653


namespace beef_weight_loss_percentage_l800_800131

theorem beef_weight_loss_percentage :
  ∀ (weight_before weight_after weight_lost : ℝ),
  weight_before = 846.15 →
  weight_after = 550 →
  weight_lost = weight_before - weight_after →
  (weight_lost / weight_before) * 100 ≈ 34.99 :=
by
  intros weight_before weight_after weight_lost h_before h_after h_loss
  sorry

end beef_weight_loss_percentage_l800_800131


namespace JonahLychees_l800_800376

theorem JonahLychees :
  ∃ x : ℕ, (∀ n ∈ {3, 4, 5, 6, 7, 8}, (x % n = n - 1)) ∧ x = 839 :=
by
  sorry

end JonahLychees_l800_800376


namespace probability_factor_less_than_10_of_120_l800_800907

theorem probability_factor_less_than_10_of_120 : 
  let n := 120 in
  let factors := { x : ℕ | x > 0 ∧ x ∣ n } in
  let favorable_factors := { x | x ∈ factors ∧ x < 10 } in
  (favorable_factors.to_finset.card : ℚ) / (factors.to_finset.card : ℚ) = 7 / 16 :=
by
  sorry

end probability_factor_less_than_10_of_120_l800_800907


namespace max_earnings_l800_800414

section MaryEarnings

def regular_rate : ℝ := 10
def first_period_hours : ℕ := 40
def second_period_hours : ℕ := 10
def third_period_hours : ℕ := 10
def weekend_days : ℕ := 2
def weekend_bonus_per_day : ℝ := 50
def bonus_threshold_hours : ℕ := 55
def overtime_multiplier_second_period : ℝ := 0.25
def overtime_multiplier_third_period : ℝ := 0.5
def milestone_bonus : ℝ := 100

def regular_pay := regular_rate * first_period_hours
def second_period_pay := (regular_rate * (1 + overtime_multiplier_second_period)) * second_period_hours
def third_period_pay := (regular_rate * (1 + overtime_multiplier_third_period)) * third_period_hours
def weekend_bonus := weekend_days * weekend_bonus_per_day
def milestone_pay := milestone_bonus

def total_earnings := regular_pay + second_period_pay + third_period_pay + weekend_bonus + milestone_pay

theorem max_earnings : total_earnings = 875 := by
  sorry

end MaryEarnings

end max_earnings_l800_800414


namespace necessary_but_not_sufficient_l800_800100

theorem necessary_but_not_sufficient (x: ℝ) :
  (1 < x ∧ x < 4) → (1 < x ∧ x < 3) := by
sorry

end necessary_but_not_sufficient_l800_800100


namespace min_sum_length_perpendicular_chords_l800_800280

variables {p : ℝ} (h : p > 0)

def parabola (x y : ℝ) : Prop := y^2 = 4 * p * (x + p)

theorem min_sum_length_perpendicular_chords (h: p > 0) :
  ∃ (AB CD : ℝ), AB * CD = 1 → |AB| + |CD| = 16 * p := sorry

end min_sum_length_perpendicular_chords_l800_800280


namespace rhombus_perimeter_l800_800461

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 20) (h2 : d2 = 16) :
  let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  in 4 * s = 8 * Real.sqrt 41 := 
by
  sorry

end rhombus_perimeter_l800_800461


namespace ryan_spit_distance_l800_800642

def billy_distance : ℝ := 30

def madison_distance (b : ℝ) : ℝ := b + (20 / 100 * b)

def ryan_distance (m : ℝ) : ℝ := m - (50 / 100 * m)

theorem ryan_spit_distance : ryan_distance (madison_distance billy_distance) = 18 :=
by
  sorry

end ryan_spit_distance_l800_800642


namespace pentagon_perimeter_l800_800084

def point := (ℝ × ℝ)
def dist (p1 p2 : point) : ℝ := real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

variables (A B C D E : point)
variable h1 : dist A B = 1
variable h2 : dist B C = 1
variable h3 : dist C D = 1
variable h4 : dist D E = 1
variable h5 : dist A E = 2

theorem pentagon_perimeter :
  dist A B + dist B C + dist C D + dist D E + dist A E = 6 :=
sorry

end pentagon_perimeter_l800_800084


namespace masha_meets_mother_at_l800_800415

noncomputable def meets_masha_time : ℕ :=
  let usual_end_time := 780 -- 13:00 in minutes
  let early_end_time := 720 -- 12:00 in minutes
  let arrive_early := 12  -- 12 minutes in minutes
  let meeting_time := usual_end_time - 6 in -- The mother meets Masha 6 minutes before the usual end time
  meeting_time

theorem masha_meets_mother_at (usual_end_time early_end_time arrive_early : ℕ) :
  usual_end_time = 780 ∧ early_end_time = 720 ∧ arrive_early = 12 →
  meets_masha_time = 774 :=
begin
  intro h,
  simp [meets_masha_time],
  exact h,
end

end masha_meets_mother_at_l800_800415


namespace sin_value_l800_800753

theorem sin_value (A : ℝ) (h: Real.tan A + Real.cot A = 2) : Real.sin A = Real.sqrt 2 / 2 := 
sorry

end sin_value_l800_800753


namespace number_of_satisfying_n_l800_800809

def g (n : ℕ) : ℕ :=
if n % 2 = 1 then n^2 - 1 else n / 2

def satisfies_g (n : ℕ) : Prop :=
∃ m : ℕ, Nat.iterate g m n = 1

theorem number_of_satisfying_n : 
  { n | n ≥ 1 ∧ n ≤ 100 ∧ satisfies_g n }.finite.toFinset.card = 8 := by
  sorry

end number_of_satisfying_n_l800_800809


namespace count_congruent_3_mod_8_l800_800739

theorem count_congruent_3_mod_8 (n : ℕ) (h₁ : 1 ≤ n) (h₂ : n ≤ 300) :
  ∃ k : ℕ, (1 ≤ 8 * k + 3 ∧ 8 * k + 3 ≤ 300) ∧ n = 38 :=
by
  sorry

end count_congruent_3_mod_8_l800_800739


namespace maximize_profit_l800_800000

noncomputable def profit_function : ℝ → ℝ
| x := if 0 < x ∧ x < 40 then
         -10 * x^2 + 400 * x - 2000
       else
         -x - 10000 / x + 2500

theorem maximize_profit :
  ∃ x : ℝ, (profit_function 100 = 2300) ∧
           ∀ y : ℝ, y ≠ 100 → profit_function y < 2300 :=
begin
  sorry
end

end maximize_profit_l800_800000


namespace joe_money_left_l800_800308

def initial_pocket_money : ℝ := 450
def fraction_spent_on_chocolates : ℝ := 1 / 9
def fraction_spent_on_fruits : ℝ := 2 / 5
def amount_spent_on_chocolates : ℝ := fraction_spent_on_chocolates * initial_pocket_money
def amount_spent_on_fruits : ℝ := fraction_spent_on_fruits * initial_pocket_money
def amount_left (initial : ℝ) (spent_chocolates : ℝ) (spent_fruits : ℝ) : ℝ := 
  initial - spent_chocolates - spent_fruits

theorem joe_money_left : amount_left initial_pocket_money amount_spent_on_chocolates amount_spent_on_fruits = 220 :=
  by
    sorry

end joe_money_left_l800_800308


namespace power_function_at_16_l800_800254

theorem power_function_at_16 :
  ∃ (α : ℝ), (∀ x : ℝ, f x = x ^ α) ∧ (f 4 = 2) → (f 16 = 4) :=
by
  sorry

end power_function_at_16_l800_800254


namespace fraction_evaluation_l800_800087

theorem fraction_evaluation (a : ℝ) (h : a = 3) : (3 * a^(-2) + (a^(-3) / 3)) / (a^2) = 28 / 729 := by
  rw [h]
  simp
  sorry

end fraction_evaluation_l800_800087


namespace math_olympiad_proof_l800_800349

theorem math_olympiad_proof (scores : Fin 20 → ℕ) 
  (h_diff : ∀ i j, i ≠ j → scores i ≠ scores j) 
  (h_sum : ∀ i j k, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k) : 
  ∀ i, scores i > 18 :=
by
  sorry

end math_olympiad_proof_l800_800349


namespace reservoir_before_storm_fullness_l800_800093

variable (C : ℝ) (original_contents : ℝ := 200) (storm_deposit : ℝ := 120) (after_storm_fullness : ℝ := 0.80)

theorem reservoir_before_storm_fullness :
  C = 320 / 0.80 → (original_contents / C) * 100 = 50 :=
by
  intro hC
  rw [hC]
  have h_capacity : C = 400 := by
    calc
      C = 320 / 0.80 : hC
      ... =  400 : by norm_num
  have h_percentage : (original_contents / 400) * 100 = 50 := by norm_num
  exact h_percentage

end reservoir_before_storm_fullness_l800_800093


namespace min_lit_bulbs_l800_800223

theorem min_lit_bulbs (n : ℕ) (h : n ≥ 1) : 
  ∃ rows cols, (rows ⊆ Finset.range n) ∧ (cols ⊆ Finset.range n) ∧ 
  (∀ i j, (i ∈ rows ∧ j ∈ cols) ↔ (i + j) % 2 = 1) ∧ 
  rows.card * (n - cols.card) + cols.card * (n - rows.card) = 2 * n - 2 :=
by sorry

end min_lit_bulbs_l800_800223


namespace distance_from_B_to_orthocenter_of_BKH_l800_800674

variable {a b : ℝ}

theorem distance_from_B_to_orthocenter_of_BKH (H : ℝ) (K : ℝ) :
  let B := BKH in
  KH = a ∧ BD = b ∧ ∃ (B H K : ℝ), 
  B = H ∧ B = K ∧ H = orthocenter B K H :=
  distance B (orthocenter B K H) = sqrt (b^2 - a^2) :=
sorry

end distance_from_B_to_orthocenter_of_BKH_l800_800674


namespace production_days_l800_800562

theorem production_days (x : ℕ) (d : ℕ) (h1 : ∀ (p : ℕ), 20 * p = 2 * x → p = 4 * d ) (h2 : ∀ (q : ℕ), 5 * q = x → q = d/4) :
(d = 4) :=
begin
  rewrite h1,
  rewrite h2,
  linarith,
  sorry,
end

end production_days_l800_800562


namespace player_B_questions_l800_800923

theorem player_B_questions :
  ∀ (a b : ℕ → ℕ), (∀ i j, i ≠ j → a i + b j = a j + b i) →
  ∃ k, k = 11 := sorry

end player_B_questions_l800_800923


namespace minimum_distance_focus_to_circle_point_l800_800245

def focus_of_parabola : ℝ × ℝ := (1, 0)
def center_of_circle : ℝ × ℝ := (4, 4)
def radius_of_circle : ℝ := 4
def circle_equation (x y : ℝ) : Prop :=
  (x - 4)^2 + (y - 4)^2 = 16

theorem minimum_distance_focus_to_circle_point :
  ∃ P : ℝ × ℝ, circle_equation P.1 P.2 ∧ dist focus_of_parabola P = 5 :=
sorry

end minimum_distance_focus_to_circle_point_l800_800245


namespace abs_inequality_solution_l800_800024

theorem abs_inequality_solution :
  {x : ℝ | |x + 2| > 3} = {x : ℝ | x < -5} ∪ {x : ℝ | x > 1} :=
by
  sorry

end abs_inequality_solution_l800_800024


namespace find_x_l800_800950

theorem find_x (x : ℝ) (h : 40 * x - 138 = 102) : x = 6 :=
by 
  sorry

end find_x_l800_800950


namespace range_of_a_l800_800323

def proposition (a : ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → log x - (1/2) * x^2 - a < 0

theorem range_of_a (a : ℝ) : ¬ proposition a ↔ a ≤ -1/2 :=
by sorry

end range_of_a_l800_800323


namespace probability_of_zero_product_l800_800225

open Set

noncomputable def set := \{ -3, 0, 0, 4, 7, 8 \}

def total_combinations := Nat.choose 6 2

def favorable_outcomes := 9

def probability := favorable_outcomes / total_combinations

theorem probability_of_zero_product : probability = 3 / 5 := by
  sorry

end probability_of_zero_product_l800_800225


namespace number_of_lattice_points_l800_800297

theorem number_of_lattice_points :
  let pairs := {p : ℤ × ℤ | let a := p.1, let b := p.2 in a^2 + b^2 < 25 ∧ a^2 + b^2 < 10 * a ∧ a^2 + b^2 < 10 * b} in
  finset.card (finset.filter pairs finset.univ) = 8 :=
by
  sorry

end number_of_lattice_points_l800_800297


namespace sum_of_quarter_circles_approaches_circumference_l800_800932

theorem sum_of_quarter_circles_approaches_circumference (C : ℝ) (n : ℕ) (h : 0 < n) :
  (∑ i in finset.range(2 * n), (C / (2 * n))) = C :=
sorry

end sum_of_quarter_circles_approaches_circumference_l800_800932


namespace rhombus_perimeter_l800_800469

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 20) (h2 : d2 = 16) : ∃ p, p = 8 * Real.sqrt 41 := by
  let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  have h3 : s = 2 * Real.sqrt 41 := by sorry
  let p := 4 * s
  have h4 : p = 8 * Real.sqrt 41 := by sorry
  exact ⟨p, h4⟩

end rhombus_perimeter_l800_800469


namespace volume_increase_is_79_4_percent_l800_800480

noncomputable def original_volume (L B H : ℝ) : ℝ := L * B * H

noncomputable def new_volume (L B H : ℝ) : ℝ :=
  (L * 1.15) * (B * 1.30) * (H * 1.20)

noncomputable def volume_increase (L B H : ℝ) : ℝ :=
  new_volume L B H - original_volume L B H

theorem volume_increase_is_79_4_percent (L B H : ℝ) :
  volume_increase L B H = 0.794 * original_volume L B H := by
  sorry

end volume_increase_is_79_4_percent_l800_800480


namespace locus_of_point_Q_l800_800369

open Real EuclideanGeometry

-- Definitions of parameters and points
def sphere_center (a b c R : ℝ) : EuclideanSpace ℝ (fin 3) := ![a, b, c]

def point_P (a b c : ℝ) : EuclideanSpace ℝ (fin 3) := ![0, 0, 0]

def point_A (a b c R : ℝ) : EuclideanSpace ℝ (fin 3) := ![a + sqrt (R^2 - b^2 - c^2), 0, 0]

def point_B (a b c R : ℝ) : EuclideanSpace ℝ (fin 3) := ![0, b + sqrt (R^2 - a^2 - c^2), 0]

def point_C (a b c R : ℝ) : EuclideanSpace ℝ (fin 3) := ![0, 0, c + sqrt (R^2 - a^2 - b^2)]

def point_Q (a b c R : ℝ) : EuclideanSpace ℝ (fin 3) :=
  let A := point_A a b c R in
  let B := point_B a b c R in
  let C := point_C a b c R in
  A + B + C

-- The theorem stating the problem
theorem locus_of_point_Q (a b c R : ℝ) :
  ∃ r : ℝ, r = sqrt (3 * R^2 - 2 * norm (sphere_center a b c R - point_P a b c)) :=
sorry

end locus_of_point_Q_l800_800369


namespace trig_identity_in_triangle_l800_800771

noncomputable section

variables {a b c : ℝ} {A B C : ℝ}
-- Assume that a, b, c are side lengths of triangle ABC with respective angles A, B, C
-- One of the given conditions
def given_condition (a b : ℝ) (A B : ℝ) : Prop := a * cos A = b * sin B

-- The goal to prove
theorem trig_identity_in_triangle (h : given_condition a b A B) :
  sin A * cos A + cos B * cos B = 1 := sorry

end trig_identity_in_triangle_l800_800771


namespace martin_total_distance_l800_800405

theorem martin_total_distance (T S1 S2 t : ℕ) (hT : T = 8) (hS1 : S1 = 70) (hS2 : S2 = 85) (ht : t = T / 2) : S1 * t + S2 * t = 620 := 
by
  sorry

end martin_total_distance_l800_800405


namespace cistern_length_is_8_l800_800585

noncomputable def cistern_length : ℝ :=
let w := 4 in
let d := 1.25 in
let A := 62 in
let L := ((A - 2 * (d * w) - (d * w)) / (w + d)) in
L

theorem cistern_length_is_8 :
  cistern_length = 8 :=
by
  unfold cistern_length
  sorry

end cistern_length_is_8_l800_800585


namespace olympiad_scores_above_18_l800_800342

theorem olympiad_scores_above_18 
  (n : Nat) 
  (scores : Fin n → ℕ) 
  (h_diff_scores : ∀ i j : Fin n, i ≠ j → scores i ≠ scores j) 
  (h_score_sum : ∀ i j k : Fin n, i ≠ j ∧ i ≠ k ∧ j ≠ k → scores i < scores j + scores k) 
  (h_n : n = 20) : 
  ∀ i : Fin n, scores i > 18 := 
by 
  -- See the proof for the detailed steps.
  sorry

end olympiad_scores_above_18_l800_800342


namespace quadratic_inequality_solution_l800_800663

theorem quadratic_inequality_solution : 
  {x : ℝ | (x + 2) * (x - 3) < 0} = {x : ℝ | -2 < x ∧ x < 3} := 
by 
  -- Definitions of roots
  let x1 := -2
  let x2 := 3
  -- Conditions about the quadratic inequality and parabola's properties
  have h_roots : ∀ x, (x + 2) * (x - 3) = 0 → x = x1 ∨ x = x2 := sorry
  have h_parabola : ∀ x, (x + 2) * (x - 3) < 0 ↔ -2 < x ∧ x < 3 := sorry
  -- We need to show the solution set is exactly the interval where the parabola is below the x-axis
  assumption
  sorry

end quadratic_inequality_solution_l800_800663


namespace largest_four_digit_number_with_property_l800_800072

theorem largest_four_digit_number_with_property :
  ∃ (a b c d : ℕ), (a = 9 ∧ b = 0 ∧ c = 9 ∧ d = 9 ∧ c = a + b ∧ d = b + c ∧ 1000 * a + 100 * b + 10 * c + d = 9099) :=
sorry

end largest_four_digit_number_with_property_l800_800072


namespace sum_of_digits_y_coordinate_of_C_l800_800812

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_y_coordinate_of_C :
  ∃ (a b : ℝ), (a ≠ b) ∧
               ((A := (a, a^2) : ℝ × ℝ) ×
                (B := (b, b^2) : ℝ × ℝ) ×
                (C := ((a + b) / 2, ((a + b) / 2)^2) : ℝ × ℝ), 
                ∃ (abc_area : ℝ), abc_area = 504 ∧
                2 * abs (b - a) * ((a + b) / 2)^2 = 504 →
                sum_of_digits (nat_abs (256 : ℤ)) = 13) :=
by sorry

end sum_of_digits_y_coordinate_of_C_l800_800812


namespace range_of_f_l800_800992

noncomputable def arccot (x : ℝ) : ℝ := real.arccot x

noncomputable def f (x : ℝ) : ℝ := real.arcsin x + real.arccos x + arccot x

theorem range_of_f :
  set.range (λ x, f x) = set.Icc (3 * real.pi / 4) (5 * real.pi / 4) :=
by {
  sorry
}

end range_of_f_l800_800992


namespace olympiad_scores_l800_800354

theorem olympiad_scores (scores : Fin 20 → ℕ) 
  (uniqueScores : ∀ i j, i ≠ j → scores i ≠ scores j)
  (less_than_sum_of_others : ∀ i j k, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k) :
  ∀ i, scores i > 18 := 
by sorry

end olympiad_scores_l800_800354


namespace find_k_l800_800708

noncomputable theory

def circle (P : ℝ × ℝ) := P.1^2 + P.2^2 - 4 * P.1 - 4 * P.2 + 7 = 0
def line (Q : ℝ × ℝ) (k : ℝ) := Q.2 = k * Q.1
def min_dist (PQ : ℝ) := PQ = 2 * real.sqrt 2 - 1

theorem find_k (k : ℝ) (P Q : ℝ × ℝ) 
  (hP : circle P) 
  (hQ : line Q k) 
  (h_dist : min_dist (dist P Q)) : 
  k = -1 :=
by
  sorry

end find_k_l800_800708


namespace cos_sq_minus_sin_sq_15_degree_value_of_cos_sq_minus_sin_sq_15_degree_l800_800028

-- Define the angles and trigonometric identities
def angle := 15 -- degrees
def doubleAngle := 30 -- degrees

def cos_sq_minus_sin_sq (θ : ℝ) : ℝ := Real.cos θ ^ 2 - Real.sin θ ^ 2
def cosine (θ : ℝ) : ℝ := Real.cos θ
def expected_value := Real.sqrt 3 / 2

theorem cos_sq_minus_sin_sq_15_degree :
  cos_sq_minus_sin_sq (angle * Real.pi / 180) = cosine (doubleAngle * Real.pi / 180) := by
  sorry

theorem value_of_cos_sq_minus_sin_sq_15_degree :
  cos_sq_minus_sin_sq (angle * Real.pi / 180) = expected_value := by
  sorry

end cos_sq_minus_sin_sq_15_degree_value_of_cos_sq_minus_sin_sq_15_degree_l800_800028


namespace total_selection_methods_l800_800128

def num_courses_group_A := 3
def num_courses_group_B := 4
def total_courses_selected := 3

theorem total_selection_methods 
  (at_least_one_from_each : num_courses_group_A > 0 ∧ num_courses_group_B > 0)
  (total_courses : total_courses_selected = 3) :
  ∃ N, N = 30 :=
sorry

end total_selection_methods_l800_800128


namespace maoming_population_scientific_notation_l800_800009

-- Definitions for conditions
def is_scientific_notation (a : ℝ) (n : ℤ) (x : ℝ) : Prop :=
  x = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10

-- The main theorem to prove
theorem maoming_population_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), is_scientific_notation a n 6800000 ∧ a = 6.8 ∧ n = 6 :=
sorry

end maoming_population_scientific_notation_l800_800009


namespace wendy_accountant_years_l800_800515

def wendy_age : ℕ := 80
def accounting_related_percentage : ℚ := 0.5
def accounting_manager_years : ℕ := 15

def accounting_related_years : ℕ := accounting_related_percentage * wendy_age
def accountant_years : ℕ := accounting_related_years - accounting_manager_years

theorem wendy_accountant_years : accountant_years = 25 := by sorry

end wendy_accountant_years_l800_800515


namespace bus_stop_time_l800_800195

-- Define the conditions as constants
def speed_without_stoppages : ℝ := 64 / 60 -- in km/min
def speed_with_stoppages : ℝ := 48 / 60 -- in km/min
def distance_difference : ℝ := speed_without_stoppages - speed_with_stoppages

-- Definition of time stopped per hour based on conditions
def time_stopped_per_hour : ℝ := distance_difference / speed_without_stoppages * 60

-- Target proof
theorem bus_stop_time :
  time_stopped_per_hour = 15 :=
by
  sorry

end bus_stop_time_l800_800195


namespace marked_price_correct_l800_800639

noncomputable def marked_price (CP : ℝ) (first_discount second_discount : ℝ) (additional_profit : ℝ) (profit : ℝ) := 
  let SP := CP * (1 + profit)
  let FSP := SP * (1 + additional_profit)
  (FSP / ((1 - first_discount) * (1 - second_discount)))

-- Define the given conditions
def cost_price : ℝ := 47.50
def first_discount : ℝ := 0.05
def second_discount : ℝ := 0.10
def profit : ℝ := 0.30
def additional_profit : ℝ := 0.20

-- Define the statement we want to prove
theorem marked_price_correct : marked_price cost_price first_discount second_discount additional_profit profit ≈ 86.67 := sorry

end marked_price_correct_l800_800639


namespace min_of_four_l800_800960

theorem min_of_four (a b c d : ℝ) (h1 : a = 0) (h2 : b = -1/2) (h3 : c = -1) (h4 : d = Real.sqrt 2) : 
  min (min (min a b) c) d = -1 :=
by
  -- Given conditions
  rw [h1, h2, h3, h4]
  -- Intermediate min comparisons
  have h : min (min 0 (-1/2)) (-1) = -1,
  { norm_num },
  rw h,
  -- Final min comparison with √2
  have h' : min (-1) (Real.sqrt 2) = -1,
  { norm_num },
  exact h'

end min_of_four_l800_800960


namespace smallest_N_is_14_l800_800472

-- Definition of depicted number and cyclic arrangement
def depicted_number : Type := List (Fin 2) -- Depicted numbers are lists of digits (0 corresponds to 1, 1 corresponds to 2)

-- A condition representing the function that checks if a list contains all possible four-digit combinations
def contains_all_four_digit_combinations (arr: List (Fin 2)) : Prop :=
  ∀ (seq: List (Fin 2)), seq.length = 4 → seq ⊆ arr

-- The problem statement: find the smallest N where an arrangement contains all four-digit combinations
def smallest_N (N: Nat) (arr: List (Fin 2)) : Prop :=
  N = arr.length ∧ contains_all_four_digit_combinations arr

theorem smallest_N_is_14 : ∃ (N : Nat) (arr: List (Fin 2)), smallest_N N arr ∧ N = 14 :=
by
  -- Placeholder for the proof
  sorry

end smallest_N_is_14_l800_800472


namespace consumption_increased_by_27_91_percent_l800_800883
noncomputable def percentage_increase_in_consumption (T C : ℝ) : ℝ :=
  let new_tax_rate := 0.86 * T
  let new_revenue_effect := 1.1000000000000085
  let cons_percentage_increase (P : ℝ) := (new_tax_rate * (C * (1 + P))) = new_revenue_effect * (T * C)
  let P_solution := 0.2790697674418605
  if cons_percentage_increase P_solution then P_solution * 100 else 0

-- The statement we are proving
theorem consumption_increased_by_27_91_percent (T C : ℝ) (hT : 0 < T) (hC : 0 < C) :
  percentage_increase_in_consumption T C = 27.91 :=
by
  sorry

end consumption_increased_by_27_91_percent_l800_800883


namespace new_radius_of_circle_l800_800485

theorem new_radius_of_circle
  (r_1 : ℝ)
  (A_1 : ℝ := π * r_1^2)
  (r_2 : ℝ)
  (A_2 : ℝ := 0.64 * A_1) 
  (h1 : r_1 = 5) 
  (h2 : A_2 = π * r_2^2) : 
  r_2 = 4 :=
by 
  sorry

end new_radius_of_circle_l800_800485


namespace value_of_f_at_2pi_over_3_l800_800238

def y (a b c x : ℝ) : ℝ := a * Real.sin x + b * Real.cos x + c

def f (a b c x : ℝ) : ℝ := Real.cos (2 * x) + 3

theorem value_of_f_at_2pi_over_3 (a b c : ℝ)
  (h1 : y a b c (π / 4) = 4)
  (h2 : ∀ x, 2 ≤ y a b c x)
  (ha : a^2 + b^2 = 2)
  (h3 : c = 3) :
  f a b c (2 * π / 3) = 5 / 2 := by
  sorry

end value_of_f_at_2pi_over_3_l800_800238


namespace necessary_but_not_sufficient_condition_l800_800228

theorem necessary_but_not_sufficient_condition (a b : ℝ) :
  ((a > 2) ∧ (b > 2) → (a + b > 4)) ∧ ¬((a + b > 4) → (a > 2) ∧ (b > 2)) :=
by
  sorry

end necessary_but_not_sufficient_condition_l800_800228


namespace min_sum_length_perpendicular_chords_l800_800281

variables {p : ℝ} (h : p > 0)

def parabola (x y : ℝ) : Prop := y^2 = 4 * p * (x + p)

theorem min_sum_length_perpendicular_chords (h: p > 0) :
  ∃ (AB CD : ℝ), AB * CD = 1 → |AB| + |CD| = 16 * p := sorry

end min_sum_length_perpendicular_chords_l800_800281


namespace sugar_required_in_new_recipe_l800_800098

theorem sugar_required_in_new_recipe 
  (initial_ratio_flour_water_sugar : Nat → Nat → Nat → Prop)
  (new_ratio_flour_water : Nat → Nat → Prop)
  (new_ratio_flour_sugar : Rat → Nat → Prop)
  (cups_of_water : ℚ)
  (sugar_quantity : ℚ) :
  (initial_ratio_flour_water_sugar 11 5 2) →
  (new_ratio_flour_water 22 10) →
  (new_ratio_flour_sugar 5.5 1) →
  (cups_of_water = 7.5) →
  (sugar_quantity = 3) :=
begin
  sorry
end

end sugar_required_in_new_recipe_l800_800098


namespace positive_difference_jo_kate_l800_800375

def sum_n (n : ℕ) : ℕ := n * (n + 1) / 2

def round_to_nearest_5 (n : ℕ) : ℕ :=
  let r := n % 5
  if r = 0 then n
  else if r < 3 then n - r
  else n + (5 - r)

def kate_sum (n : ℕ) : ℕ :=
  (list.range n).map (λ x => round_to_nearest_5 (x + 1)).sum

theorem positive_difference_jo_kate : 
  |sum_n 100 - kate_sum 100| = 3550 :=
by
  sorry

end positive_difference_jo_kate_l800_800375


namespace rhombus_perimeter_l800_800456

theorem rhombus_perimeter
  (d1 d2 : ℝ)
  (h1 : d1 = 20)
  (h2 : d2 = 16) :
  4 * (Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)) = 8 * Real.sqrt 41 := 
  sorry

end rhombus_perimeter_l800_800456


namespace probability_is_two_thirds_l800_800580

-- Define the general framework and conditions
def total_students : ℕ := 4
def students_from_first_grade : ℕ := 2
def students_from_second_grade : ℕ := 2

-- Define the combinations for selecting 2 students out of 4
def total_ways_to_select_2_students : ℕ := Nat.choose total_students 2

-- Define the combinations for selecting 1 student from each grade
def ways_to_select_1_from_first : ℕ := Nat.choose students_from_first_grade 1
def ways_to_select_1_from_second : ℕ := Nat.choose students_from_second_grade 1
def favorable_ways : ℕ := ways_to_select_1_from_first * ways_to_select_1_from_second

-- The target probability calculation
noncomputable def probability_of_different_grades : ℚ :=
  favorable_ways / total_ways_to_select_2_students

-- The statement and proof requirement (proof is deferred with sorry)
theorem probability_is_two_thirds :
  probability_of_different_grades = 2 / 3 :=
by sorry

end probability_is_two_thirds_l800_800580


namespace art_department_probability_l800_800574

theorem art_department_probability : 
  let students := {s1, s2, s3, s4} 
  let first_grade := {s1, s2}
  let second_grade := {s3, s4}
  let total_pairs := { (x, y) | x ∈ students ∧ y ∈ students ∧ x < y }.to_finset.card
  let diff_grade_pairs := { (x, y) | x ∈ first_grade ∧ y ∈ second_grade ∨ x ∈ second_grade ∧ y ∈ first_grade}.to_finset.card
  (diff_grade_pairs / total_pairs) = 2 / 3 := 
by 
  sorry

end art_department_probability_l800_800574


namespace painting_equation_satisfied_l800_800187

variable (t : ℝ)

-- Define the painting rates
def rate_Doug : ℝ := 1 / 5
def rate_Dave : ℝ := 1 / 7
def combined_rate : ℝ := rate_Doug + rate_Dave

-- Define the total time including breaks
def total_time := t

-- Define the time spent actually painting
def painting_time := t - 2

-- State the theorem to be proved
theorem painting_equation_satisfied : combined_rate * painting_time = 1 := 
sorry

end painting_equation_satisfied_l800_800187


namespace quadrilateral_is_rhombus_l800_800609

/-- 
  A quadrilateral circumscribed around a circle with diagonals intersecting at the center 
  of the circle is a rhombus.
-/
theorem quadrilateral_is_rhombus (A B C D O : Point) 
  (h_circumscribed : isCircumscribedQuadrilateral A B C D O)
  (h_diagonals_intersect : diagonalsIntersectAtCenter A B C D O) : 
  isRhombus A B C D :=
sorry

end quadrilateral_is_rhombus_l800_800609


namespace sequence_sum_l800_800016

theorem sequence_sum (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = a 1 + a n + n) :
  ∑ k in finset.range 2016, 1 / (a (k + 1)) = 4032 / 2017 := 
sorry

end sequence_sum_l800_800016


namespace sandbox_area_l800_800127

def length : ℕ := 312
def width : ℕ := 146
def area : ℕ := 45552

theorem sandbox_area : length * width = area := by
  sorry

end sandbox_area_l800_800127


namespace martin_total_distance_l800_800407

theorem martin_total_distance (T S1 S2 t : ℕ) (hT : T = 8) (hS1 : S1 = 70) (hS2 : S2 = 85) (ht : t = T / 2) : S1 * t + S2 * t = 620 := 
by
  sorry

end martin_total_distance_l800_800407


namespace delta_value_l800_800751

theorem delta_value (Δ : ℤ) (h : 4 * -3 = Δ - 3) : Δ = -9 :=
sorry

end delta_value_l800_800751


namespace rectangle_angle_AMD_l800_800433

theorem rectangle_angle_AMD {A B C D M : Type} [rectangle A B C D] (hAB : AB = 8) (hBC : BC = 4)
  (hM : M ∈ segment A B) (hAngle : ∠A M D = ∠C M D) : ∠A M D = 45 :=
by
  -- The proof goes here
  sorry

end rectangle_angle_AMD_l800_800433


namespace collinearity_of_k_l_m_n_l800_800826

theorem collinearity_of_k_l_m_n
  (A B C D K L M N : Point)
  (on_edge_AB : K ∈ LineSegment A B)
  (on_edge_BC : L ∈ LineSegment B C)
  (on_edge_CD : M ∈ LineSegment C D)
  (on_edge_DA : N ∈ LineSegment D A)
  (coplanar : ∃ α : Plane, A ∈ α ∧ B ∈ α ∧ C ∈ α ∧ D ∈ α ∧ K ∈ α ∧ L ∈ α ∧ M ∈ α ∧ N ∈ α)
  (ratio_AN_AD_BL_BC : AN / AD = BL / BC):
  DM / MC = AK / KB := 
  sorry

end collinearity_of_k_l_m_n_l800_800826


namespace largest_valid_four_digit_number_l800_800061

-- Definition of the problem conditions
def is_valid_number (a b c d : ℕ) : Prop :=
  c = a + b ∧ d = b + c

-- Proposition that we need to prove
theorem largest_valid_four_digit_number : ∃ (a b c d : ℕ),
  is_valid_number a b c d ∧ a * 1000 + b * 100 + c * 10 + d = 9099 :=
by
  sorry

end largest_valid_four_digit_number_l800_800061


namespace water_evaporation_weight_l800_800915

noncomputable def initial_weight : ℝ := 200
noncomputable def initial_salt_concentration : ℝ := 0.05
noncomputable def final_salt_concentration : ℝ := 0.08

theorem water_evaporation_weight (W_final : ℝ) (evaporation_weight : ℝ) 
  (h1 : W_final = 10 / final_salt_concentration) 
  (h2 : evaporation_weight = initial_weight - W_final) : 
  evaporation_weight = 75 :=
by
  sorry

end water_evaporation_weight_l800_800915


namespace kalebs_restaurant_bill_l800_800634

theorem kalebs_restaurant_bill :
  let adults := 6
  let children := 2
  let adult_meal_cost := 6
  let children_meal_cost := 4
  let soda_cost := 2
  (adults * adult_meal_cost + children * children_meal_cost + (adults + children) * soda_cost) = 60 := 
by
  let adults := 6
  let children := 2
  let adult_meal_cost := 6
  let children_meal_cost := 4
  let soda_cost := 2
  calc 
    adults * adult_meal_cost + children * children_meal_cost + (adults + children) * soda_cost 
      = 6 * 6 + 2 * 4 + (6 + 2) * 2 : by rfl
    ... = 36 + 8 + 16 : by rfl
    ... = 60 : by rfl

end kalebs_restaurant_bill_l800_800634


namespace probability_second_student_is_boy_l800_800140

theorem probability_second_student_is_boy :
  let students := ["boy", "boy", "girl", "girl"]
  let permutations := students.permutations.filter (λ s, s.to_seq.nodup)
  let favorable := permutations.filter (λ s, s[1] == "boy")
  (favorable.length / permutations.length : ℚ) = 1 / 2 := by
  sorry

end probability_second_student_is_boy_l800_800140


namespace find_k_l800_800707

noncomputable def vec3 : Type := ℝ × ℝ × ℝ

def magnitude (v : vec3) : ℝ := ⟨λp, (p.1^2 + p.2^2 + p.3^2)⟩

def dot_product (v w : vec3) : ℝ := v.1*w.1 + v.2*w.2 + v.3*w.3

def sub_vec (v w : vec3) : vec3 := (v.1 - w.1, v.2 - w.2, v.3 - w.3)

theorem find_k (a b : vec3) (k : ℝ) (θ : ℝ) (h_a : magnitude a = 2) 
  (h_b : magnitude b = 1) (h_θ : θ = real.pi / 3) 
  (h_eq : magnitude (sub_vec a ⟨k*b⟩) = real.sqrt 3) : k = 1 :=
sorry

end find_k_l800_800707


namespace max_odd_integers_l800_800147

theorem max_odd_integers (l : List ℕ) (h_length : l.length = 7) (h_product_even : l.prod % 2 = 0) : 
  ∃ n : ℕ, n ≤ 7 ∧ (filter (λ x, x % 2 = 1) l).length = n ∧ n = 6 := 
by
  sorry

end max_odd_integers_l800_800147


namespace right_triangle_medians_AB_length_l800_800416

theorem right_triangle_medians_AB_length
  (A B C M N : Point)
  (h_ABC : right_triangle A B C)
  (hM : midpoint M B C)
  (hN : midpoint N A C)
  (hM_len : dist A M = 8)
  (hN_len : dist B N = 2 * Real.sqrt 14) :
  dist A B = 4 * Real.sqrt 6 :=
sorry

end right_triangle_medians_AB_length_l800_800416


namespace probability_students_from_different_grades_l800_800572

theorem probability_students_from_different_grades :
  let total_students := 4
  let first_grade_students := 2
  let second_grade_students := 2
  (2 from total_students are selected) ->
  (2 from total_students are from different grades) ->
  ℝ :=
by 
  sorry

end probability_students_from_different_grades_l800_800572


namespace number_and_sum_of_divisors_l800_800742

theorem number_and_sum_of_divisors (n : ℕ) (h : n = 90) : 
  (Nat.totient n = 12) ∧ (NumberTheory.sum_divisors n = 234) := by
  sorry

end number_and_sum_of_divisors_l800_800742


namespace legendre_formula_l800_800397

theorem legendre_formula (p : ℕ) (n : ℕ) (hp : Nat.Prime p) :
  Nat.factorialFactors p n = ∑ k in Finset.range (Nat.log n / Nat.log p + 1), n / p^k :=
sorry

end legendre_formula_l800_800397


namespace intersection_of_cone_section_is_parabola_l800_800447

def cone_section_is_parabola (apex_angle section_angle : ℝ) : Prop :=
  (apex_angle = 90) → (section_angle = 45) → (section_shape = "parabola")

theorem intersection_of_cone_section_is_parabola :
  cone_section_is_parabola 90 45 :=
by
  intro h1
  intro h2
  sorry

end intersection_of_cone_section_is_parabola_l800_800447


namespace smallest_n_for_P_lt_1_over_4020_l800_800981

def P (n : ℕ) : ℚ :=
  (List.range (n - 1)).map (λ k, (2 * (k + 1) : ℚ) / (2 * (k + 1) - 1)).prod * (1 / (2 * n + 1))

theorem smallest_n_for_P_lt_1_over_4020 : ∃ n : ℕ, (P n < 1 / 4020) ∧ (∀ m : ℕ, m < n → P m ≥ 1 / 4020) := 
begin
  sorry
end

end smallest_n_for_P_lt_1_over_4020_l800_800981


namespace rectangular_hyperbola_through_foci_l800_800699

noncomputable def ellipse_eq (a b : ℝ) : Prop :=
∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1

noncomputable def foci (a b : ℝ) : set (ℝ × ℝ) :=
{ (sqrt (a^2 - b^2), 0), (-sqrt (a^2 - b^2), 0)}

noncomputable def hyperbola_through_A'B'C'D' (a b : ℝ) (A' B' C' D' : ℝ × ℝ) : Prop :=
∀ x y : ℝ, (x * y) / (a * b) + (x^2 / a^2) - (y^2 / b^2) = (a^2 - b^2)

theorem rectangular_hyperbola_through_foci :
∀ (a b : ℝ) (A' B' C' D' : ℝ × ℝ),
ellipse_eq a b →
hyperbola_through_A'B'C'D' a b A' B' C' D' →
∀ f ∈ foci a b, hyperbola_through_A'B'C'D' a b A' B' C' D' f.snd f.fst :=
begin
  -- proof omitted, just statement
  sorry
end

end rectangular_hyperbola_through_foci_l800_800699


namespace xy_product_l800_800250

-- Define the proof problem with the conditions and required statement
theorem xy_product (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy_distinct : x ≠ y) (h : x + 3 / x = y + 3 / y) : x * y = 3 := 
  sorry

end xy_product_l800_800250


namespace mountaineers_arrangement_l800_800186
open BigOperators

-- Definition to state the number of combinations
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- The main statement translating our problem
theorem mountaineers_arrangement :
  (choose 4 2) * (choose 6 2) = 120 := by
  sorry

end mountaineers_arrangement_l800_800186


namespace land_remaining_is_correct_l800_800956

def lizzie_covered : ℕ := 250
def other_covered : ℕ := 265
def total_land : ℕ := 900
def land_remaining : ℕ := total_land - (lizzie_covered + other_covered)

theorem land_remaining_is_correct : land_remaining = 385 := 
by
  sorry

end land_remaining_is_correct_l800_800956


namespace parallel_vectors_implies_m_eq_neg1_l800_800293

theorem parallel_vectors_implies_m_eq_neg1 (m : ℝ) :
  let a := (m, -1)
  let b := (1, m + 2)
  a.1 * b.2 = a.2 * b.1 → m = -1 :=
by
  intro h
  sorry

end parallel_vectors_implies_m_eq_neg1_l800_800293


namespace rectangle_perimeter_l800_800856

variable (L W : ℝ)

-- Conditions
def width := 70
def length := (7 / 5) * width

-- Perimeter calculation and proof goal
def perimeter (L W : ℝ) := 2 * (L + W)

theorem rectangle_perimeter : perimeter (length) (width) = 336 := by
  sorry

end rectangle_perimeter_l800_800856


namespace max_days_group_students_l800_800188

theorem max_days_group_students (students : ℕ) (groups : ℕ) (group_size : ℕ)
  (total_groups : students = groups * group_size)
  (unique_pairs : ∀ (d1 d2 : ℕ) (g1 g2 : ℕ), d1 ≠ d2 → g1 ≠ g2 → 
                   ∀ (s1 s2 : ℕ), s1 ∈ g1 ↔ s2 ∈ g2 → false) :
  students = 289 → groups = 17 → group_size = 17 → 
  ∃ (days : ℕ), days = 18 :=
by {
  intros h_students h_groups h_group_size,
  use 18,
  sorry
}

end max_days_group_students_l800_800188


namespace joe_money_left_l800_800311

theorem joe_money_left (original_money : ℕ) (spent_chocolates_frac spent_fruits_frac : ℚ) 
  (h1 : original_money = 450) (h2 : spent_chocolates_frac = 1/9) (h3 : spent_fruits_frac = 2/5) : 
  original_money - (original_money * spent_chocolates_frac).to_nat - (original_money * spent_fruits_frac).to_nat = 220 := by
  sorry

end joe_money_left_l800_800311


namespace overall_percentage_gain_l800_800589

def false_weight_A : ℝ := 950 / 1000
def false_weight_B : ℝ := 900 / 1000
def false_weight_C : ℝ := 920 / 1000

def cost_price_A : ℝ := 20
def cost_price_B : ℝ := 25
def cost_price_C : ℝ := 18

def amount_sold : ℝ := 10

def total_actual_cost_price : ℝ := 
  (false_weight_A * amount_sold * cost_price_A) +
  (false_weight_B * amount_sold * cost_price_B) +
  (false_weight_C * amount_sold * cost_price_C)

def total_selling_price : ℝ := 
  (amount_sold * cost_price_A) +
  (amount_sold * cost_price_B) +
  (amount_sold * cost_price_C)

def total_gain : ℝ := total_selling_price - total_actual_cost_price

def percentage_gain : ℝ := (total_gain / total_actual_cost_price) * 100

theorem overall_percentage_gain : percentage_gain ≈ 8.5 := sorry

end overall_percentage_gain_l800_800589


namespace side_length_of_largest_square_l800_800118

theorem side_length_of_largest_square (A_cross : ℝ) (s : ℝ)
  (h1 : A_cross = 810) : s = 36 :=
  have h_large_squares : 2 * (s / 2)^2 = s^2 / 2 := by sorry
  have h_small_squares : 2 * (s / 4)^2 = s^2 / 8 := by sorry
  have h_combined_area : s^2 / 2 + s^2 / 8 = 810 := by sorry
  have h_final : 5 * s^2 / 8 = 810 := by sorry
  have h_s2 : s^2 = 1296 := by sorry
  have h_s : s = 36 := by sorry
  h_s

end side_length_of_largest_square_l800_800118


namespace base_8_sum_units_digit_l800_800215

section
  def digit_in_base (n : ℕ) (base : ℕ) (d : ℕ) : Prop :=
  ((n % base) = d)

theorem base_8_sum_units_digit :
  let n1 := 63
  let n2 := 74
  let base := 8
  (digit_in_base n1 base 3) →
  (digit_in_base n2 base 4) →
  digit_in_base (n1 + n2) base 7 :=
by
  intro h1 h2
  -- placeholder for the detailed proof
  sorry
end

end base_8_sum_units_digit_l800_800215


namespace triangle_area_is_correct_l800_800138

noncomputable def triangle_area_inscribed_circle (r : ℝ) (θ1 θ2 θ3 : ℝ) : ℝ := 
  (1 / 2) * r^2 * (Real.sin θ1 + Real.sin θ2 + Real.sin θ3)

theorem triangle_area_is_correct :
  triangle_area_inscribed_circle (18 / Real.pi) (Real.pi / 3) (2 * Real.pi / 3) Real.pi =
  162 * Real.sqrt 3 / (Real.pi^2) :=
by sorry

end triangle_area_is_correct_l800_800138


namespace second_yellow_ball_probability_l800_800032

-- Define conditions
def initial_white_balls := 5
def initial_yellow_balls := 3
def first_ball_is_yellow := true
def remaining_total_balls := initial_white_balls + initial_yellow_balls - 1

-- Define the expected probability
def expected_probability := 2 / remaining_total_balls

-- Define the number of yellow balls left after drawing the first yellow ball
def yellow_balls_left := initial_yellow_balls - 1

-- Define the probability calculation function
def probability_of_second_yellow (yellow_balls_left : ℕ) (remaining_balls : ℕ) : ℚ :=
  yellow_balls_left / remaining_balls

-- Theorem stating the problem
theorem second_yellow_ball_probability :
  yellow_balls_left = 2 →
  remaining_total_balls = 7 →
  probability_of_second_yellow yellow_balls_left remaining_total_balls = expected_probability :=
by {
  intros h1 h2,
  rw [h1, h2],
  exact rfl,
}

end second_yellow_ball_probability_l800_800032


namespace hyperbola_equation_distance_AB_l800_800274

-- Definition of the hyperbola C
def hyperbola (x y a b : ℝ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (x^2 / a^2 - y^2 / b^2 = 1)

-- Definition of conditions
structure hyperbola_conditions :=
(a : ℝ)
(b : ℝ)
(c : ℝ)
(a_pos : a > 0)
(b_pos : b > 0)
(real_axis_length : 2 * a = 2)
(right_focus : c = sqrt 5)
(c_relation : c^2 = a^2 + b^2)

-- Part 1: Prove the equation of the hyperbola
theorem hyperbola_equation
  (h : hyperbola_conditions) :
  hyperbola 1 2 1 2 :=
by {
  have a_eq : h.a = 1,
  by linarith [h.real_axis_length],
  have c_eq : h.c = sqrt 5 := h.right_focus,
  have b_sq_eq : h.b^2 = 4,
  by {
    rw [h.c_relation, a_eq],
    ring_nf at *,
    linarith,
  },
  have b_eq : h.b = 2,
  by linarith [b_sq_eq],
  exact ⟨h.a_pos, h.b_pos, by {
    rw [a_eq, b_eq],
    field_simp
  }⟩,
}

-- Part 2: Prove the distance between points A and B
theorem distance_AB
  (h : hyperbola_conditions) :
  ∃ (A B : ℝ × ℝ), ((A.1 - B.1)^2 + (A.2 - B.2)^2 = (4 * sqrt 14 / 3)^2) :=
by {
  sorry -- the actual proof is omitted.
}

end hyperbola_equation_distance_AB_l800_800274


namespace correct_statement_l800_800528

def isAcuteAngle (θ : ℝ) : Prop :=
  0 < θ ∧ θ < 90

def isFirstQuadrantAngle (θ : ℝ) : Prop :=
  ∃ (k : ℤ), (k * 360 : ℝ) < θ ∧ θ < (k * 360 : ℝ + 90)

theorem correct_statement :
  (∃ θ : ℝ, isAcuteAngle θ ∧ isFirstQuadrantAngle θ) :=
by
  sorry

end correct_statement_l800_800528


namespace distribute_pencils_equally_l800_800088

theorem distribute_pencils_equally :
  ( ∃ (totalPencils : ℕ), totalPencils = 6 * 8 + 4) →
  ((totalPencils : ℕ) : (totalPencils) % 4 = 0) →
  ∃ (pencilsPerPerson : ℕ), 
    totalPencils / 4 = pencilsPerPerson ∧ pencilsPerPerson = 13 :=
by
  sorry

end distribute_pencils_equally_l800_800088


namespace tournament_partition_l800_800951

theorem tournament_partition {T : Type} [fintype T] [decidable_eq T] 
  (k : ℕ) (h : fintype.card T = 2 * k) (no7cycles : ∀ (C : finset T), C.card = 7 → ¬is_cycle C) : 
  ∃ A B : finset T, A.card = k ∧ B.card = k ∧ ∀ G ∈ [A, B], ∀ (C : finset (finset.univ.filter (λ x, G x))), C.card = 3 → ¬is_cycle C :=
by
  sorry

end tournament_partition_l800_800951


namespace hundreds_digit_of_8_pow_2048_l800_800178

theorem hundreds_digit_of_8_pow_2048 : 
  (8^2048 % 1000) / 100 = 0 := 
by
  sorry

end hundreds_digit_of_8_pow_2048_l800_800178


namespace increasing_interval_of_cubic_plus_linear_l800_800003

theorem increasing_interval_of_cubic_plus_linear :
    ∀ x : ℝ, (deriv (λ x, x^3 + x) x > 0) → increasing_on (λ x, x^3 + x) (set.univ) :=
by
  intro x h
  sorry

end increasing_interval_of_cubic_plus_linear_l800_800003


namespace prime_divisible_by_5_l800_800010

/--
  Given the sequence of starting sums of prime numbers,
  prove that exactly one of the first 15 sums is both prime and divisible by 5.
-/

theorem prime_divisible_by_5 :
  let sums := [2, 2 + 3, 2 + 3 + 5, 2 + 3 + 5 + 7, 2 + 3 + 5 + 7 + 11, 2 + 3 + 5 + 7 + 11 + 13, 
               2 + 3 + 5 + 7 + 11 + 13 + 17, 2 + 3 + 5 + 7 + 11 + 13 + 17 + 19, 
               2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 + 23, 2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 + 23 + 29, 
               2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 + 23 + 29 + 31, 
               2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 + 23 + 29 + 31 + 37, 
               2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 + 23 + 29 + 31 + 37 + 41, 
               2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 + 23 + 29 + 31 + 37 + 41 + 43, 
               2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 + 23 + 29 + 31 + 37 + 41 + 43 + 47]
  in (sums.filter (λ n => n % 5 = 0 ∧ nat.prime n)).length = 1 := by
  sorry

end prime_divisible_by_5_l800_800010


namespace imaginary_part_of_z_l800_800717

-- Define the imaginary unit i
def i : ℂ := complex.I

-- Define the modulus of 1 - i
def modulus_1_minus_i : ℝ := complex.abs (1 - i)

-- Define the complex number z
def z : ℂ := modulus_1_minus_i * (i^2017)

-- State the theorem to prove the imaginary part of z is -sqrt(2)
theorem imaginary_part_of_z : complex.im z = -real.sqrt 2 := by
  sorry

end imaginary_part_of_z_l800_800717


namespace minimize_resistance_l800_800748

theorem minimize_resistance
  (a1 a2 a3 a4 a5 a6 : ℝ)
  (h1 : a1 > a2)
  (h2 : a2 > a3)
  (h3 : a3 > a4)
  (h4 : a4 > a5)
  (h5 : a5 > a6) :
  -- Assuming a component configuration Resistors_Configuration to complete the statement
  minimize_total_resistance a1 a2 a3 a4 a5 a6 :=
sorry

end minimize_resistance_l800_800748


namespace second_group_students_l800_800004

theorem second_group_students (S : ℕ) : 
    (1200 / 40) = 9 + S + 11 → S = 10 :=
by sorry

end second_group_students_l800_800004


namespace find_x_l800_800290

-- Define the vectors a and b
def a : ℝ × ℝ := (4, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 3)

-- Define the dot product function
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Condition for perpendicular vectors
def perpendicular (u v : ℝ × ℝ) : Prop :=
  dot_product u v = 0

theorem find_x (x : ℝ) (h : perpendicular a (b x)) : x = -3 / 2 :=
by
  sorry

end find_x_l800_800290


namespace arithmetic_sequence_third_term_l800_800324

theorem arithmetic_sequence_third_term (a d : ℤ) (h : a + (a + 4 * d) = 14) : a + 2 * d = 7 := by
  -- We assume the sum of the first and fifth term is 14 and prove that the third term is 7.
  sorry

end arithmetic_sequence_third_term_l800_800324


namespace rhombus_perimeter_l800_800463

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 20) (h2 : d2 = 16) : 
  let side := (real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)) in 
  4 * side = 8 * real.sqrt 41 := by
  sorry

end rhombus_perimeter_l800_800463


namespace cos_2alpha_zero_l800_800684

theorem cos_2alpha_zero (α : ℝ) (hα1 : 0 < α) (hα2 : α < Real.pi / 2) 
(h : Real.sin (2 * α) = Real.cos (Real.pi / 4 - α)) : 
  Real.cos (2 * α) = 0 :=
by
  sorry

end cos_2alpha_zero_l800_800684


namespace solve_eq_x_squared_floor_x_eq_3_l800_800435

theorem solve_eq_x_squared_floor_x_eq_3 (x : ℝ) (n : ℤ) (α : ℝ) 
  (h1 : x = n + α)
  (h2 : 0 ≤ α)
  (h3 : α < 1) 
  (h4 : x^2 - n = 3)
  : x = real.cbrt 4 :=
sorry

end solve_eq_x_squared_floor_x_eq_3_l800_800435


namespace geometric_problems_l800_800629

variables {A P Q M K L O X Y Z : Point}
variables {c l : Line}
variables [Circle c O]

-- Conditions
axiom tangent_AP : TangentLine c A P
axiom tangent_AQ : TangentLine c A Q
axiom midpoint_M : Midpoint M P Q
axiom secant_AKL : SecantLine A K L
axiom parallel_l_AQ : Parallel l AQ
axiom intersects_X : IntersectAt l QK X
axiom intersects_Y : IntersectAt l QP Y
axiom intersects_Z : IntersectAt l QL Z

-- Prove the required conditions:
theorem geometric_problems
  (h1 : PM^2 = KM * ML)
  (h2 : XY = YZ) : true :=
sorry

end geometric_problems_l800_800629


namespace radius_of_inner_circle_tangent_to_semicircles_l800_800133

noncomputable def square_side_length : ℝ := 4
def semicircle_radius : ℝ := square_side_length / 2.5

theorem radius_of_inner_circle_tangent_to_semicircles
  (r : ℝ)
  (h1 : 10 * semicircle_radius = 4)
  (h2 : semicircle_radius = 4 / 2.5)
  : r = (Real.sqrt 116) / 10 :=
sorry

end radius_of_inner_circle_tangent_to_semicircles_l800_800133


namespace exists_perfect_square_between_sums_l800_800158

def is_prime (n : ℕ) : Prop := n ≥ 2 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes : ℕ → ℕ
| 0       := 2
| (n + 1) := Nat.find (λ m, m > primes n ∧ is_prime m)

def S (n : ℕ) : ℕ := (Finset.range n).sum primes

theorem exists_perfect_square_between_sums (n : ℕ) : 
  ∃ a : ℕ, (∃ k : ℕ, a = k ^ 2) ∧ S n < a ∧ a < S (n + 1) :=
sorry

end exists_perfect_square_between_sums_l800_800158


namespace delta_value_l800_800750

theorem delta_value (Δ : ℤ) (h : 4 * (-3) = Δ - 3) : Δ = -9 :=
by {
  sorry
}

end delta_value_l800_800750


namespace domain_of_f_l800_800849

noncomputable def domain_of_log_function : set ℝ :=
{x : ℝ | 2 - x > 0}

theorem domain_of_f :
  ∀ x : ℝ, x ∈ domain_of_log_function ↔ x < 2 :=
by
  sorry

end domain_of_f_l800_800849


namespace total_cups_needed_l800_800945

theorem total_cups_needed (cereal_servings : ℝ) (milk_servings : ℝ) (nuts_servings : ℝ) 
  (cereal_cups_per_serving : ℝ) (milk_cups_per_serving : ℝ) (nuts_cups_per_serving : ℝ) : 
  cereal_servings = 18.0 ∧ milk_servings = 12.0 ∧ nuts_servings = 6.0 ∧ 
  cereal_cups_per_serving = 2.0 ∧ milk_cups_per_serving = 1.5 ∧ nuts_cups_per_serving = 0.5 → 
  (cereal_servings * cereal_cups_per_serving + milk_servings * milk_cups_per_serving + 
   nuts_servings * nuts_cups_per_serving) = 57.0 :=
by
  sorry

end total_cups_needed_l800_800945


namespace cos_equation_solutions_l800_800990

theorem cos_equation_solutions :
  (∃ (s : Finset ℝ), 
    (∀ x ∈ s, -π ≤ x ∧ x ≤ π ∧ 
      cos (8 * x) + cos(4 * x)^2 + cos(2 * x)^3 + cos(x)^4 = 0) ∧ 
    s.card = 16) :=
sorry

end cos_equation_solutions_l800_800990


namespace true_discount_correct_l800_800493

-- Define the conditions
def face_value : ℝ := 2240
def annual_interest_rate : ℝ := 0.16
def time_in_years : ℝ := 9 / 12

-- Define the present value function
def present_value (FV : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  FV / (1 + r * t)

-- Define the true discount function
def true_discount (FV : ℝ) (PV : ℝ) : ℝ :=
  FV - PV

-- Theorem stating the question and expected answer
theorem true_discount_correct :
  true_discount face_value (present_value face_value annual_interest_rate time_in_years) = 240 :=
by
  sorry

end true_discount_correct_l800_800493


namespace largest_valid_number_l800_800083

-- Define the conditions for the digits of the number
def valid_digits (a b c d : ℕ) : Prop :=
  c = a + b ∧ d = b + c

-- Prove that the number formed by digits 9, 0, 9, 9 is the largest valid 4-digit number
theorem largest_valid_number : ∃ a b c d, valid_digits a b c d ∧
  (a * 1000 + b * 100 + c * 10 + d = 9099) :=
begin
  use [9, 0, 9, 9],
  split,
  { -- Proof of valid digits condition
    split; refl },
  { -- Proof that the number is 9099
    refl }
end

end largest_valid_number_l800_800083


namespace part_one_part_two_l800_800670

noncomputable def a_n (n : ℕ) : ℝ :=
  if h : n > 0 then
    classical.some (exists_unique.exists (by {
      have := classical.some_spec (exists_real_root_of_polynomial (polynomial.mk [0, 1 / (↑n : ℝ), 0, 1])),
      rw [polynomial.eval, polynomial.mk] at this,
      exact this.1,
    }))
  else 0

def a_root_eq (n : ℕ) : Prop :=
  a_n n ^ 3 + a_n n / n = 1

theorem part_one (n : ℕ) (h : n > 0) :
  a_n (n + 1) > a_n n :=
sorry

theorem part_two (n : ℕ) (h : n > 0) :
  ∑ i in finset.range n, 1 / ((i+1) ^ 2 * a_n (i+1)) < a_n n :=
sorry

end part_one_part_two_l800_800670


namespace min_bulbs_lit_proof_l800_800221

-- Definitions to capture initial conditions and the problem
def bulb_state (n : ℕ) : Type := fin n → fin n → bool

def initial_state (n : ℕ) : bulb_state n := λ i j, false

-- Function to describe the state change after pressing a bulb
def press_bulb (n : ℕ) (state : bulb_state n) (i j : fin n) : bulb_state n :=
  λ x y, if x = i ∨ y = j then ¬ state x y else state x y

-- Function to calculate the minimum lit bulbs
def min_lit_bulbs (n : ℕ) (initial : bulb_state n) : ℕ :=
  2 * n - 2

-- The theorem statement
theorem min_bulbs_lit_proof (n : ℕ) (initial : bulb_state n) : ∃ seq : list (fin n × fin n), 
  (press_bulb n initial_state seq.head.1 seq.head.2).count true = 2 * n - 2 :=
sorry

end min_bulbs_lit_proof_l800_800221


namespace permutation_as_disjoint_cycles_l800_800821

noncomputable def A : Set ℕ := Set.univ ∩ { x | x ≤ 9 }

noncomputable def f : ℕ → ℕ 
| 1 := 3
| 2 := 4
| 3 := 7
| 4 := 6
| 5 := 9
| 6 := 2
| 7 := 1
| 8 := 8
| 9 := 5
| _ := 0  -- For inputs outside the set {1, 2, ..., 9}, choose a default value

theorem permutation_as_disjoint_cycles :
  f = permutations.cycle_of_list [1, 3, 7] * permutations.cycle_of_list [2, 4, 6] * permutations.cycle_of_list [5, 9] * permutations.cycle_of_list [8] := sorry

end permutation_as_disjoint_cycles_l800_800821


namespace triangle_congruence_condition_D_fails_l800_800958

theorem triangle_congruence_condition_D_fails {A B C A' B' C' : Type}
  (hA : ∠A = ∠A')
  (hB : ∠B = ∠B')
  (hC : ∠C = ∠C') : 
  ¬ (∠A = ∠A' ∧ ∠B = ∠B' ∧ ∠C = ∠C' → ΔABC ≌ ΔA'B'C') :=
sorry

end triangle_congruence_condition_D_fails_l800_800958


namespace right_triangle_tan_sin_l800_800359

theorem right_triangle_tan_sin (A B C : Type*) 
  (AB BC : ℝ)
  (h_AB : AB = 15)
  (h_BC : BC = 17)
  (h_angle : 90 = 90) : 
  ∃ AC : ℝ, 
  AC = (64:ℝ).sqrt ∧ 
  tan(atan(AC / AB)) = AC / AB ∧
  sin(asin(AC / BC)) = AC / BC :=
by {
  let AC := (64:ℝ).sqrt,
  use AC,
  simp [h_AB, h_BC],
  split,
  { exact rfl },
  split,
  { simp [Real.tan_eq_iff_tan_eq, Real.atan_eq_iff_implies AC, h_AB] },
  { simp [Real.sin_eq_iff_sin_eq, Real.asin_eq_iff_implies AC, h_BC] },
  sorry
}

end right_triangle_tan_sin_l800_800359


namespace largest_even_integer_of_product_2880_l800_800893

theorem largest_even_integer_of_product_2880 :
  ∃ n : ℤ, (n-2) * n * (n+2) = 2880 ∧ n + 2 = 22 := 
by {
  sorry
}

end largest_even_integer_of_product_2880_l800_800893


namespace max_real_part_sum_wk_l800_800820

noncomputable def z (k : ℕ) : ℂ := 8 * exp (0 : ℂ)
def poly_root_form (k : ℕ) : ℂ := 8 * complex.cos (2 * k * real.pi / 6) + 8 * complex.sin (2 * k * real.pi / 6) * complex.I

theorem max_real_part_sum_wk :
  ∃ (w : ℕ → ℂ), 
  (∀ k, w k = (poly_root_form k) ∨ w k = complex.I * (poly_root_form k)) ∧
  ∑ k in finset.range 6, w k.real = 8 + 8 * real.sqrt 3 :=
by
  sorry

end max_real_part_sum_wk_l800_800820


namespace values_of_a_l800_800920

noncomputable def odd_function_domain (a : ℝ) (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f(x) = x^a ∧ (-x)^a = -x^a

theorem values_of_a {a : ℝ} :
  (∃ f : ℝ → ℝ, odd_function_domain a f) ↔ a = 1 ∨ a = 3 :=
by
  sorry

end values_of_a_l800_800920


namespace total_spent_l800_800983

def price_almond_croissant : ℝ := 4.50
def price_salami_cheese_croissant : ℝ := 4.50
def price_plain_croissant : ℝ := 3.00
def price_focaccia : ℝ := 4.00
def price_latte : ℝ := 2.50
def num_lattes : ℕ := 2

theorem total_spent :
  price_almond_croissant + price_salami_cheese_croissant + price_plain_croissant +
  price_focaccia + (num_lattes * price_latte) = 21.00 := by
  sorry

end total_spent_l800_800983


namespace remaining_integers_in_T_after_removals_l800_800487

open Set

def T : Set ℕ := { n | n ∈ range 60 } ∪ { 60 }

def multiples (k : ℕ) : Set ℕ := { n | ∃ m, n = k * m }

def filtered_set : Set ℕ := T \ multiples 2 \ multiples 3 \ multiples 5

theorem remaining_integers_in_T_after_removals :
  card filtered_set = 16 :=
sorry

end remaining_integers_in_T_after_removals_l800_800487


namespace angle_ABF_eq_90_l800_800788

noncomputable def ellipse := 
  {a b : ℝ // a > b ∧ b > 0} 

theorem angle_ABF_eq_90 (a b : ℝ) (h : a > b ∧ b > 0) 
(eccentricity : (∃ c : ℝ, c/a = (Real.sqrt 5 - 1) / 2)) : 
  ∀ F A B : ℝ,
  ∃ (x y : ℝ) (eq : (x/a)^2 + (y/b)^2 = 1)
  (F A B : ℝ),
  angle F A B = 90 :=
by sorry

end angle_ABF_eq_90_l800_800788


namespace increasing_intervals_l800_800660

noncomputable def y (x : ℝ) : ℝ := Real.sqrt (Real.sin (π / 3 - 2 * x))

def is_increasing_interval (k : ℤ) : Set ℝ :=
  set.Icc (-π / 3 + k * π) (-π / 12 + k * π)

theorem increasing_intervals :
  ∀ k : ℤ, ∀ x1 x2 ∈ is_increasing_interval k, x1 < x2 → y x1 < y x2 := 
by
  intros k x1 x2 hx1 hx2 hlt
  sorry

end increasing_intervals_l800_800660


namespace count_valid_numbers_l800_800735

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def satisfies_conditions (n : ℕ) : Prop :=
  (n % 4 = 0) ∧ (n % 9 = 0) ∧ (n % 25 = 0)

theorem count_valid_numbers : {n : ℕ // is_four_digit n ∧ satisfies_conditions n}.card = 10 :=
  sorry

end count_valid_numbers_l800_800735


namespace positive_difference_jo_kate_l800_800374

def sum_n (n : ℕ) : ℕ := n * (n + 1) / 2

def round_to_nearest_5 (n : ℕ) : ℕ :=
  let r := n % 5
  if r = 0 then n
  else if r < 3 then n - r
  else n + (5 - r)

def kate_sum (n : ℕ) : ℕ :=
  (list.range n).map (λ x => round_to_nearest_5 (x + 1)).sum

theorem positive_difference_jo_kate : 
  |sum_n 100 - kate_sum 100| = 3550 :=
by
  sorry

end positive_difference_jo_kate_l800_800374


namespace binomial_sum_mod_prime_l800_800872

theorem binomial_sum_mod_prime (T : ℕ) (hT : T = ∑ k in Finset.range 65, Nat.choose 2024 k) : 
  T % 2027 = 1089 :=
by
  have h_prime : Nat.prime 2027 := by sorry -- Given that 2027 is prime
  have h := (2024 : ℤ) % 2027
  sorry -- The proof of the actual sum equivalences

end binomial_sum_mod_prime_l800_800872


namespace largest_four_digit_number_with_property_l800_800077

theorem largest_four_digit_number_with_property :
  ∃ (a b c d : ℕ), (a = 9 ∧ b = 0 ∧ c = 9 ∧ d = 9 ∧ c = a + b ∧ d = b + c ∧ 1000 * a + 100 * b + 10 * c + d = 9099) :=
sorry

end largest_four_digit_number_with_property_l800_800077


namespace find_k_l800_800668

variable (θ : ℝ)

theorem find_k
  (h : (sin θ + csc θ)^2 + (cos θ + sec θ)^2 = k + 3 * tan θ ^ 2 + 3 * cot θ ^ 2) :
  k = 6 :=
sorry

end find_k_l800_800668


namespace d_value_of_ellipse_tangent_l800_800155

/-- Given an ellipse tangent to both the x-axis and y-axis in the first quadrant, 
and having foci at (4,8) and (d,8), we prove that d equals 15. -/
theorem d_value_of_ellipse_tangent (d : ℝ) 
  (h : 2 * real.sqrt (((d - 4) / 2) ^ 2 + 64) = d + 4) : d = 15 :=
begin
  sorry
end

end d_value_of_ellipse_tangent_l800_800155


namespace final_price_wednesday_l800_800778

theorem final_price_wednesday :
  let coffee_price := 6
  let cheesecake_price := 10
  let sandwich_price := 8
  let coffee_discount := 0.25
  let cheesecake_discount_wednesday := 0.10
  let additional_discount := 3
  let sales_tax := 0.05
  let discounted_coffee_price := coffee_price - coffee_price * coffee_discount
  let discounted_cheesecake_price := cheesecake_price - cheesecake_price * cheesecake_discount_wednesday
  let total_price_before_additional_discount := discounted_coffee_price + discounted_cheesecake_price + sandwich_price
  let total_price_after_additional_discount := total_price_before_additional_discount - additional_discount
  let total_price_with_tax := total_price_after_additional_discount + total_price_after_additional_discount * sales_tax
  let final_price := total_price_with_tax.round
  final_price = 19.43 :=
by
  sorry

end final_price_wednesday_l800_800778


namespace base_eight_satisfies_conditions_l800_800667

theorem base_eight_satisfies_conditions (b : ℕ) (h : b > 7) :
  let val_123_b := b^2 + 2 * b + 3
      val_234_b := 2 * b^2 + 3 * b + 4
      val_357_b := 3 * b^2 + 5 * b + 7
  in val_123_b + val_234_b = val_357_b ∧ b = 8 :=
by
  sorry

end base_eight_satisfies_conditions_l800_800667


namespace king_can_inform_all_citizens_l800_800599

-- Define the problem domain.
def kingdom_area := 2 -- side length in kilometers

def messenger_start_time := 12 -- noon in 24-hour format

def citizen_speed := 3 -- speed in kilometers per hour

def ball_start_time := 19 -- 7 PM in 24-hour format

 -- Define the maximum time needed to inform all citizens using geometric series summation.
def total_time_to_inform_all_citizens : ℝ :=
  let s := kingdom_area * Real.sqrt 2
  let base_time := s / citizen_speed
  3 / (1 - (1 / 2))

-- The main theorem.
theorem king_can_inform_all_citizens :
  (total_time_to_inform_all_citizens ≤ 6) →
  (∀ t : ℝ, (messenger_start_time ≤ t) → (t ≤ ball_start_time - 1) → t ≠ messenger_start_time) :=
begin
  intro h,
  sorry
end

end king_can_inform_all_citizens_l800_800599


namespace interval_increasing_f_beta_eq_sqrt2_l800_800263

-- Define the function f
def f (x : ℝ) : ℝ := Real.sin(5 * Real.pi / 4 - x) - Real.cos(Real.pi / 4 + x)

-- Define the conditions for α and β
variables (α β : ℝ)
axiom cos_alpha_beta : Real.cos (α - β) = 3 / 5
axiom cos_alpha_plus_beta : Real.cos (α + β) = -3 / 5
axiom alpha_beta_range : 0 < α ∧ α < β ∧ β ≤ Real.pi / 2 

-- Define the theorem statements
theorem interval_increasing (k : ℤ) :
  monotoneOn f (Set.Icc (2 * k * Real.pi - Real.pi / 4) (2 * k * Real.pi + 3 * Real.pi / 4)) :=
sorry

theorem f_beta_eq_sqrt2 : f (Real.pi / 2) = Real.sqrt 2 :=
sorry

end interval_increasing_f_beta_eq_sqrt2_l800_800263


namespace savings_percentage_l800_800122

-- Define the problem setup and the requirements to prove the savings percentage

variables
  (P : ℝ)  -- Percentage of salary saved
  (S : ℝ := 6500)  -- Monthly salary
  (E : ℝ)  -- Original expenses
  (NewE : ℝ := 1.20 * E)  -- New expenses after 20% increase
  (NewSavings : ℝ := 260)  -- New savings after increase in expenses

-- Define the conditions based on the given problem
def conditions :=
  NewSavings = S - NewE ∧
  (P / 100) * S + E = S

-- Conjecture to prove
theorem savings_percentage : conditions → P = 20 :=
  sorry

end savings_percentage_l800_800122


namespace simplify_sequence_product_l800_800839

theorem simplify_sequence_product :
  (∏ n in Finset.range 1000, (3 * (n + 2)) / (3 * (n + 1))) = 1001 :=
by
  sorry

end simplify_sequence_product_l800_800839


namespace probability_different_grades_l800_800582

theorem probability_different_grades (A B : Type) [Fintype A] [Fintype B] (ha : Fintype.card A = 2) (hb : Fintype.card B = 2) :
  (∃ (s : Finset (A ⊕ B)), s.card = 2) →
  (Fintype.card (Finset (A ⊕ B)).filter (λ s, (∃ (a : A) (b : B), s = {sum.inl a, sum.inr b})) = 4) →
  (Fintype.card (Finset (A ⊕ B)).card-choose 2 = 6) →
  (Fintype.card (Finset (A ⊕ B)).filter (λ s, (∃ (a : A) (b : B), s = {sum.inl a, sum.inr b})) /
     Fintype.card (Finset (A ⊕ B)).card-choose 2 = 2 / 3) :=
sorry

end probability_different_grades_l800_800582


namespace jo_kate_difference_100_l800_800373

def sum_1_to_n (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def rounded_sum (n : ℕ) : ℕ :=
  let blocks := n / 5 in
  15 * blocks

theorem jo_kate_difference_100 :
  let jo_sum := sum_1_to_n 100
  let kate_sum := rounded_sum 100
  (jo_sum - kate_sum = 4750) :=
by
  let jo_sum := sum_1_to_n 100
  let kate_sum := rounded_sum 100
  show jo_sum - kate_sum = 4750
  sorry

end jo_kate_difference_100_l800_800373


namespace Juan_birth_year_proof_l800_800001

-- Let BTC_year(n) be the year of the nth BTC competition.
def BTC_year (n : ℕ) : ℕ :=
  1990 + (n - 1) * 2

-- Juan's birth year given his age and the BTC he participated in.
def Juan_birth_year (current_year : ℕ) (age : ℕ) : ℕ :=
  current_year - age

-- Main proof problem statement.
theorem Juan_birth_year_proof :
  (BTC_year 5 = 1998) →
  (Juan_birth_year 1998 14 = 1984) :=
by
  intros
  sorry

end Juan_birth_year_proof_l800_800001


namespace problem_is_happy_number_512_l800_800315

/-- A number is a "happy number" if it is the square difference of two consecutive odd numbers. -/
def is_happy_number (x : ℕ) : Prop :=
  ∃ n : ℤ, x = 8 * n

/-- The number 512 is a "happy number". -/
theorem problem_is_happy_number_512 : is_happy_number 512 :=
  sorry

end problem_is_happy_number_512_l800_800315


namespace mil_equals_one_fortieth_mm_l800_800047

-- The condition that one mil is equal to one thousandth of an inch
def mil_in_inch := 1 / 1000

-- The condition that an inch is about 2.5 cm
def inch_in_mm := 25

-- The problem statement in Lean 4 form
theorem mil_equals_one_fortieth_mm : (mil_in_inch * inch_in_mm = 1 / 40) :=
by
  sorry

end mil_equals_one_fortieth_mm_l800_800047


namespace problem_solution_l800_800090

-- Definitions
def arithmetic_seq (a : ℕ → ℝ) : Prop := ∃ d, ∀ n, a (n + 1) = a n + d
def geometric_seq (a : ℕ → ℝ) : Prop := ∃ r, ∀ n, a (n + 1) = a n * r

def sum_n (a : ℕ → ℝ) (n : ℕ) : ℝ := (Finset.range (n + 1)).sum a

-- Propositions
lemma cond_1 (a : ℕ → ℝ) (m n s t : ℕ) (hm : m > 0) (hn : n > 0) (hs : s > 0) (ht : t > 0) 
  (h_arith : arithmetic_seq a) (h_add_eq : a m + a n = a s + a t) : m + n = s + t := 
sorry  -- Condition 1 proof

lemma cond_2 (a : ℕ → ℝ) (n : ℕ) (h_arith : arithmetic_seq a) : 
  arithmetic_seq (λ k, sum_n a (k * n) - sum_n a ((k - 1) * n)) := 
sorry  -- Condition 2 proof

lemma cond_3 (a : ℕ → ℝ) (n : ℕ) (h_geom : geometric_seq a) : 
  geometric_seq (λ k, sum_n a (k * n) - sum_n a ((k - 1) * n)) := 
sorry  -- Condition 3 proof

lemma cond_4 (a : ℕ → ℝ) (S : ℕ → ℝ) (A B : ℝ) (q : ℝ) (h_geom : geometric_seq a)
  (hS : ∀ n, S n = A * (q ^ n) + B) : A + B = 0 := 
sorry  -- Condition 4 proof

-- The main theorem combining all conditions
theorem problem_solution :
  (¬ cond_1) ∧ cond_2 ∧ (¬ cond_3) ∧ cond_4 :=
sorry

end problem_solution_l800_800090


namespace exists_int_less_than_sqrt_twenty_three_l800_800425

theorem exists_int_less_than_sqrt_twenty_three : ∃ n : ℤ, n < Real.sqrt 23 := 
  sorry

end exists_int_less_than_sqrt_twenty_three_l800_800425


namespace largest_valid_number_l800_800079

-- Define the conditions for the digits of the number
def valid_digits (a b c d : ℕ) : Prop :=
  c = a + b ∧ d = b + c

-- Prove that the number formed by digits 9, 0, 9, 9 is the largest valid 4-digit number
theorem largest_valid_number : ∃ a b c d, valid_digits a b c d ∧
  (a * 1000 + b * 100 + c * 10 + d = 9099) :=
begin
  use [9, 0, 9, 9],
  split,
  { -- Proof of valid digits condition
    split; refl },
  { -- Proof that the number is 9099
    refl }
end

end largest_valid_number_l800_800079


namespace min_colors_2016x2016_l800_800545

def min_colors_needed (n : ℕ) : ℕ :=
  ⌈log 2 (n * n)⌉.natAbs

theorem min_colors_2016x2016 :
  min_colors_needed 2016 = 11 := 
by
  sorry

end min_colors_2016x2016_l800_800545


namespace complex_conjugate_solution_l800_800847

noncomputable def wz (z : ℂ) := (1 + 2 * complex.I) * complex.conj z

theorem complex_conjugate_solution :
  ∀ z : ℂ, wz z = 4 + 3 * complex.I → z = 2 + complex.I :=
by
  -- Proof goes here
  sorry

end complex_conjugate_solution_l800_800847


namespace probability_students_from_different_grades_l800_800569

theorem probability_students_from_different_grades :
  let total_students := 4
  let first_grade_students := 2
  let second_grade_students := 2
  (2 from total_students are selected) ->
  (2 from total_students are from different grades) ->
  ℝ :=
by 
  sorry

end probability_students_from_different_grades_l800_800569


namespace problem_solution_l800_800829

noncomputable def correlation_coefficient (sx2 : ℝ) (sy2 : ℝ) (sxy : ℝ) : ℝ :=
  sxy / (Real.sqrt (sx2 * sy2))

def mean (values : List ℝ) : ℝ :=
  (List.sum values) / (values.length : ℝ)

noncomputable def regression_slope (sxy : ℝ) (sx2 : ℝ) : ℝ :=
  sxy / sx2

noncomputable def regression_intercept (mean_x : ℝ) (mean_y : ℝ) (slope : ℝ) : ℝ :=
  mean_y - slope * mean_x

noncomputable def predict_y (slope : ℝ) (intercept : ℝ) (x : ℝ) : ℝ :=
  slope * x + intercept

theorem problem_solution :
  ∀ (x : List ℝ) (y : List ℝ) (n : ℕ)
    (sx2 : ℝ) (sy2 : ℝ) (sxy : ℝ)
    (mean_x_val mean_y_val : ℝ)
    (x10 : ℝ),
    sx2 = 28 →
    sy2 = 118 →
    sxy = 56 →
    mean_x x = mean_x_val →
    mean_y y = mean_y_val →
    x10 = 10 →
    let r := correlation_coefficient sx2 sy2 sxy in
    let slope := regression_slope sxy sx2 in
    let intercept := regression_intercept mean_x_val mean_y_val slope in
    let y_hat := predict_y slope intercept x10 in
    r ≈ 0.98 ∧
    slope = 2 ∧
    intercept = 9 ∧
    y_hat = 29 := by {
  intros,
  sorry
}

end problem_solution_l800_800829


namespace dot_product_of_unit_vectors_l800_800731

variables (a b : ℝ^3)
variables (h1 : ‖a‖ = 1) (h2 : ‖b‖ = 1)
variables (h3 : (a + (1/2) • b) ⬝ (a - 7 • b) = 0)

theorem dot_product_of_unit_vectors :
  a ⬝ b = -5 / 13 :=
sorry

end dot_product_of_unit_vectors_l800_800731


namespace ezekiel_painted_faces_l800_800198

noncomputable def cuboid_faces_painted (num_cuboids : ℕ) (faces_per_cuboid : ℕ) : ℕ :=
num_cuboids * faces_per_cuboid

theorem ezekiel_painted_faces :
  cuboid_faces_painted 8 6 = 48 := 
by
  sorry

end ezekiel_painted_faces_l800_800198


namespace length_of_platform_l800_800564

theorem length_of_platform (L : ℕ) (train_length : ℕ) (time_platform : ℕ) (time_pole : ℕ)
  (h1 : train_length = 300)
  (h2 : time_platform = 39)
  (h3 : time_pole = 18) :
  L = 350 :=
by 
  let train_speed := train_length / time_pole
  have h_speed : train_speed = 50 / 3, sorry
  have h_distance := train_speed * time_platform = train_length + L
  have h_result : 650 = 300 + L, sorry
  exact calc
    L = 650 - 300 : by sorry
      ... = 350 : by sorry

end length_of_platform_l800_800564


namespace percentage_within_one_standard_deviation_l800_800110

variable {α : Type*}
variables (m f : ℝ) 
variable {P : ℝ → α → Prop}

-- Symmetry about the mean: if x is within f of m, then -x is within f of m.
variable (symmetry : ∀ x, P (m + x) ↔ P (m - x))

-- 84% of the population is less than m + f
variable (distribution_condition : ∀ x, P (m + f x ))

-- Prove 68% lies within one standard deviation of the mean
theorem percentage_within_one_standard_deviation : 
  (∃ percentage_within_std_dev : ℝ, percentage_within_std_dev = 0.68) :=
begin
  sorry
end

end percentage_within_one_standard_deviation_l800_800110


namespace area_ratio_trapezoid_rectangle_l800_800794

theorem area_ratio_trapezoid_rectangle (r k : ℝ) (h1 : 0 < k) (h2 : k < r / 3) :
  let AB := 2 * real.sqrt (3 * r * k - 9 * k ^ 2),
      CD := 2 * real.sqrt (2 * r * k - 4 * k ^ 2),
      area_trapezoid := (AB + CD) * k / 2,
      area_rectangle := AB * k
  in (k → r / 3) →
     (area_trapezoid / area_rectangle) = 1 / 2 + 1 / real.sqrt 3 := by
  sorry

end area_ratio_trapezoid_rectangle_l800_800794


namespace hexagon_sides_equal_l800_800811

theorem hexagon_sides_equal
  (ABC : Triangle)
  (A1 B1 C1 : Point)  -- altitude feet
  (AA1 : Altitude ABC A A1)
  (BB1 : Altitude ABC B B1)
  (CC1 : Altitude ABC C C1)
  (O_A O_B O_C : Point)
  (insc_A : IncircleCenter ABC B1 C1 O_A)
  (insc_B : IncircleCenter ABC C1 A1 O_B)
  (insc_C : IncircleCenter ABC A1 B1 O_C)
  (T_A T_B T_C : Point)
  (tang_A : TangentPoint ABC T_A BC)
  (tang_B : TangentPoint ABC T_B CA)
  (tang_C : TangentPoint ABC T_C AB) :
  ∀ (T_A O_C T_B O_A T_C O_B : Segment), SegmentLength T_A O_C = SegmentLength T_B O_A := sorry

end hexagon_sides_equal_l800_800811


namespace smallest_time_for_horses_l800_800886
-- Import the necessary libraries

-- Definition for the problem statement in Lean
theorem smallest_time_for_horses :
  ∃ T > 0, (T = 72) ∧ ∃ horses : Finset ℕ, horses.card ≥ 8 ∧ ∀ k ∈ horses, T % k = 0 :=
begin
  sorry
end

end smallest_time_for_horses_l800_800886


namespace plane_eqn_correct_l800_800361

-- Define point P
def P := (2, 1, 1) : ℝ × ℝ × ℝ

-- Define the normal vectors of the given planes
def n1 := (1, -3, 1) : ℝ × ℝ × ℝ
def n2 := (3, -2, -2) : ℝ × ℝ × ℝ

-- Define the cross product of the vectors
def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2.2 * b.3 - a.3 * b.2.2,
   a.3 * b.1 - a.1 * b.3,
   a.1 * b.2.2 - a.2.2 * b.1)

-- Compute the normal vector of the plane
def n := cross_product n1 n2

-- Define the plane equation based on a plane normal and a point
def plane_equation (n : ℝ × ℝ × ℝ) (P : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ → ℝ :=
  λ Q : ℝ × ℝ × ℝ, n.1 * (Q.1 - P.1) + n.2.2 * (Q.2 - P.2) + n.3 * (Q.3 - P.3)

noncomputable def equation_of_plane : ℝ × ℝ × ℝ → ℝ :=
  plane_equation n P

-- Theorem statement: prove the equation of the plane is 8x + 5y + 7z - 28 = 0
theorem plane_eqn_correct :
  ∀ (x y z : ℝ), equation_of_plane (x, y, z) = 0 ↔ 8 * x + 5 * y + 7 * z - 28 = 0 :=
by
  sorry

end plane_eqn_correct_l800_800361


namespace dice_probability_l800_800937

/--
Given a single die roll, let event A be the event of rolling a number at least 5. 
In six independent trials, the probability of event A occurring at least 5 times 
is equal to 13/729.
-/
theorem dice_probability (n : ℕ) (h : n = 6) (p : ℕ → ℚ) (q : ℚ)
  (h1 : ∀ k, (k = 5 ∨ k = 6) → p k = 1/3)
  (h2 : ∀ k, ¬ (k = 5 ∨ k = 6) → p k = 2/3)
  (h3 : q = ∑ k in finset.range (n + 1), if k ≥ 5 then (nat.choose n k : ℚ) * (1/3) ^ k * (2/3) ^ (n - k) else 0) :
  q = 13 / 729 :=
by
  sorry

end dice_probability_l800_800937


namespace jo_kate_difference_100_l800_800372

def sum_1_to_n (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def rounded_sum (n : ℕ) : ℕ :=
  let blocks := n / 5 in
  15 * blocks

theorem jo_kate_difference_100 :
  let jo_sum := sum_1_to_n 100
  let kate_sum := rounded_sum 100
  (jo_sum - kate_sum = 4750) :=
by
  let jo_sum := sum_1_to_n 100
  let kate_sum := rounded_sum 100
  show jo_sum - kate_sum = 4750
  sorry

end jo_kate_difference_100_l800_800372


namespace overall_percentage_gain_l800_800588

def false_weight_A : ℝ := 950 / 1000
def false_weight_B : ℝ := 900 / 1000
def false_weight_C : ℝ := 920 / 1000

def cost_price_A : ℝ := 20
def cost_price_B : ℝ := 25
def cost_price_C : ℝ := 18

def amount_sold : ℝ := 10

def total_actual_cost_price : ℝ := 
  (false_weight_A * amount_sold * cost_price_A) +
  (false_weight_B * amount_sold * cost_price_B) +
  (false_weight_C * amount_sold * cost_price_C)

def total_selling_price : ℝ := 
  (amount_sold * cost_price_A) +
  (amount_sold * cost_price_B) +
  (amount_sold * cost_price_C)

def total_gain : ℝ := total_selling_price - total_actual_cost_price

def percentage_gain : ℝ := (total_gain / total_actual_cost_price) * 100

theorem overall_percentage_gain : percentage_gain ≈ 8.5 := sorry

end overall_percentage_gain_l800_800588


namespace oranges_in_bin_after_changes_l800_800948

def initial_oranges := 31
def thrown_away_oranges := 9
def new_oranges := 38

theorem oranges_in_bin_after_changes : 
  initial_oranges - thrown_away_oranges + new_oranges = 60 := by
  sorry

end oranges_in_bin_after_changes_l800_800948


namespace blueberry_picking_relationship_l800_800909

theorem blueberry_picking_relationship (x : ℝ) (hx : x > 10) : 
  let y1 := 60 + 18 * x
  let y2 := 150 + 15 * x
  in y1 = 60 + 18 * x ∧ y2 = 150 + 15 * x := 
by {
  sorry
}

end blueberry_picking_relationship_l800_800909


namespace max_odd_integers_l800_800146

theorem max_odd_integers (l : List ℕ) (h_length : l.length = 7) (h_product_even : l.prod % 2 = 0) : 
  ∃ n : ℕ, n ≤ 7 ∧ (filter (λ x, x % 2 = 1) l).length = n ∧ n = 6 := 
by
  sorry

end max_odd_integers_l800_800146


namespace polynomial_positive_roots_l800_800478

variable {R : Type*} [LinearOrderedField R] [CompleteLinearOrderedField R]

noncomputable def composed_poly (P : R → R) (k : ℕ) : R → R :=
  nat.rec_on k id (λ k' composed_poly_k', P ∘ composed_poly_k')

theorem polynomial_positive_roots
  {P : R → R}
  {n : ℕ}
  (h_real_coeffs: ∀ a : ℕ, a < n → (P (a.cast : R) ∈ R))
  (k m : ℕ)
  (h_nat_number_k : k ≥ 1)
  (h_real_roots : ∀ x : R, (composed_poly P m x = 0 → x > 0)) :
  ∃ x : R, P x = 0 ∧ x > 0 :=
sorry

end polynomial_positive_roots_l800_800478


namespace correct_parallelogram_is_rectangle_l800_800527

-- Definitions of conditions
def quadrilateral (A B C D : Type) : Prop := sorry
def adjacent_equal_sides (A B C D : Type) : Prop := sorry
def parallelogram (A B C D : Type) : Prop := sorry
def right_angle (A B C D : Type) : Prop := sorry
def perpendicular_diagonals (A B C D : Type) : Prop := sorry
def one_pair_parallel_sides (A B C D : Type) : Prop := sorry
def rectangle (A B C D : Type) : Prop := sorry

-- The proof problem statement
theorem correct_parallelogram_is_rectangle (A B C D : Type) : 
  (quadrilateral A B C D ∧ adjacent_equal_sides A B C D → False) →
  (parallelogram A B C D ∧ right_angle A B C D → rectangle A B C D) →
  (parallelogram A B C D ∧ perpendicular_diagonals A B C D → False) →
  (quadrilateral A B C D ∧ one_pair_parallel_sides A B C D → False) -/
  True
:= by
  intros
  sorry

end correct_parallelogram_is_rectangle_l800_800527


namespace solve_fraction_equation_l800_800666

def fraction_equation (x : ℝ) : Prop :=
  1 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) + 2 / (x - 1) = 5

theorem solve_fraction_equation (x : ℝ) (h1 : x ≠ -3) (h2 : x ≠ 1) :
  fraction_equation x → 
  x = (-11 + Real.sqrt 257) / 4 ∨ x = (-11 - Real.sqrt 257) / 4 :=
by
  sorry

end solve_fraction_equation_l800_800666


namespace Ryan_spit_distance_correct_l800_800643

-- Definitions of given conditions
def Billy_spit_distance : ℝ := 30
def Madison_spit_distance : ℝ := Billy_spit_distance * 1.20
def Ryan_spit_distance : ℝ := Madison_spit_distance * 0.50

-- Goal statement
theorem Ryan_spit_distance_correct : Ryan_spit_distance = 18 := by
  -- proof would go here
  sorry

end Ryan_spit_distance_correct_l800_800643


namespace minimum_value_2_pow_x_and_inverse_l800_800181

theorem minimum_value_2_pow_x_and_inverse (x : ℝ) : 
  2^x + 2^(-x) ≥ 2 ∧ (2^x + 2^(-x) = 2 ↔ x = 0) :=
by sorry

end minimum_value_2_pow_x_and_inverse_l800_800181


namespace find_a_l800_800640

theorem find_a (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_max_y : ∀ x, a * Real.sec (b * x) ≤ 4) (h_period : ∀ x, a * Real.sec (b * x) = a * Real.sec (b * (x + π))) : a = 4 :=
sorry

end find_a_l800_800640


namespace rhombus_perimeter_l800_800457

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 20) (h2 : d2 = 16) :
  let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  in 4 * s = 8 * Real.sqrt 41 := 
by
  sorry

end rhombus_perimeter_l800_800457


namespace students_standing_together_l800_800498

theorem students_standing_together (s : Finset ℕ) (h_size : s.card = 6) (a b : ℕ) (h_ab : a ∈ s ∧ b ∈ s) (h_ab_together : ∃ (l : List ℕ), l.length = 6 ∧ a :: b :: l = l):
  ∃ (arrangements : ℕ), arrangements = 240 := by
  sorry

end students_standing_together_l800_800498


namespace rhombus_perimeter_l800_800470

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 20) (h2 : d2 = 16) : ∃ p, p = 8 * Real.sqrt 41 := by
  let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  have h3 : s = 2 * Real.sqrt 41 := by sorry
  let p := 4 * s
  have h4 : p = 8 * Real.sqrt 41 := by sorry
  exact ⟨p, h4⟩

end rhombus_perimeter_l800_800470


namespace distance_from_Q_to_EG_l800_800437

variables {E F G H N Q: ℝ × ℝ}
variables {r1 r2: ℝ}

-- Conditions
def isSquare (E F G H : ℝ × ℝ) (sideLength : ℝ) : Prop :=
  E.2 = sideLength ∧ F.1 = sideLength ∧ G.1 = sideLength ∧ H.2 = 0 ∧ 
  E.1 = 0 ∧ F.2 = sideLength ∧ G.2 = 0 ∧ H.1 = 0

def midpoint (A B N : ℝ × ℝ) : Prop :=
  N.1 = (A.1 + B.1) / 2 ∧ N.2 = (A.2 + B.2) / 2

def circle (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : Prop :=
  (point.1 - center.1)^2 + (point.2 - center.2)^2 = radius^2

-- Proof of the distance from Q to line EG
theorem distance_from_Q_to_EG (sideLength : ℝ)
    (h_square : isSquare E F G H sideLength)
    (h_midpoint : midpoint G H N)
    (h_radius1 : r1 = 2.5)
    (h_radius2 : r2 = 5)
    (h_circle1 : circle N r1 Q)
    (h_circle2 : circle E r2 Q) :
    Q.2 = 2 :=
  sorry

end distance_from_Q_to_EG_l800_800437


namespace area_of_triangle_from_tangent_line_and_axes_l800_800657

noncomputable def curve (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

def tangent_line (x : ℝ) : ℝ := -3*x + 2

def triangle_area : ℝ := 1 / 2 * (2 / 3) * 2

theorem area_of_triangle_from_tangent_line_and_axes :
  triangle_area = 2 / 3 :=
by
  sorry

end area_of_triangle_from_tangent_line_and_axes_l800_800657


namespace lemon_pie_degrees_l800_800331

def total_students : ℕ := 45
def chocolate_pie_students : ℕ := 15
def apple_pie_students : ℕ := 10
def blueberry_pie_students : ℕ := 7
def cherry_and_lemon_students := total_students - (chocolate_pie_students + apple_pie_students + blueberry_pie_students)
def lemon_pie_students := cherry_and_lemon_students / 2

theorem lemon_pie_degrees (students_nonnegative : lemon_pie_students ≥ 0) (students_rounding : lemon_pie_students = 7) :
  (lemon_pie_students * 360 / total_students) = 56 := 
by
  -- Proof to be provided
  sorry

end lemon_pie_degrees_l800_800331


namespace ratio_of_shorter_to_longer_l800_800961

def ratio (a b : ℕ) : ℚ := a / b

theorem ratio_of_shorter_to_longer (total_length : ℕ) (shorter_length : ℕ) (longer_length : ℕ) :
  total_length = 80 →
  shorter_length = 30 →
  longer_length = total_length - shorter_length →
  ratio shorter_length longer_length = 3 / 5 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  unfold ratio
  norm_num
  sorry

end ratio_of_shorter_to_longer_l800_800961


namespace no_solution_equation_B_l800_800089

theorem no_solution_equation_B :
  ¬(∃ x : ℝ, (|2 * x| + 4 = 0)) ∧
  (∃ x : ℝ, (x - 3)^2 = 0) ∧
  (∃ x : ℝ, √(3 * x) - 1 = 0) ∧
  (∃ x : ℝ, √(-3 * x) - 3 = 0) ∧
  (∃ x : ℝ, |5 * x| - 6 = 0) := by
{
  sorry -- Proof is omitted
}

end no_solution_equation_B_l800_800089


namespace olympiad_scores_l800_800356

theorem olympiad_scores (scores : Fin 20 → ℕ) 
  (uniqueScores : ∀ i j, i ≠ j → scores i ≠ scores j)
  (less_than_sum_of_others : ∀ i j k, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k) :
  ∀ i, scores i > 18 := 
by sorry

end olympiad_scores_l800_800356


namespace inverse_proportion_second_fourth_l800_800830

-- Conditions:
def is_inverse_proportion (f : ℝ → ℝ) : Prop := ∃ (k : ℝ), f = (λ x, k / x)
def in_second_and_fourth_quadrants (f : ℝ → ℝ) : Prop := 
  ∀ (x : ℝ), x ≠ 0 → (x > 0 → f x < 0) ∧ (x < 0 → f x > 0)

-- The function we want to prove
def f (x : ℝ) := -3 / x

-- The statement including our conditions and what we need to prove
theorem inverse_proportion_second_fourth : is_inverse_proportion f ∧ in_second_and_fourth_quadrants f :=
by
  unfold is_inverse_proportion
  unfold in_second_and_fourth_quadrants
  sorry

end inverse_proportion_second_fourth_l800_800830


namespace train_time_to_pass_bridge_l800_800135

theorem train_time_to_pass_bridge
  (length_train : ℝ) (length_bridge : ℝ) (speed_kmph : ℝ)
  (h1 : length_train = 500) (h2 : length_bridge = 200) (h3 : speed_kmph = 72) :
  let speed_mps := speed_kmph * 1000 / 3600
  let total_distance := length_train + length_bridge
  let time := total_distance / speed_mps
  time = 35 :=
by
  sorry

end train_time_to_pass_bridge_l800_800135


namespace equation_of_parabola_from_point_area_of_triangle_OMN_l800_800277

-- Definition of the parabola and given conditions.
def parabola (p : ℝ) (p_pos : 0 < p) := ∀ x y, y^2 = 2 * p * x
def pointA := (1 : ℝ, -2 : ℝ)
def lineThroughFocus45Degrees (F : ℝ × ℝ) := λ y, y - F.snd = 1

-- Theorem statements
theorem equation_of_parabola_from_point :
  ∃ p : ℝ, 0 < p ∧ p = 2 ∧
    (parabola p (by norm_num) pointA.fst pointA.snd ∧
      ∀ y, y^2 = 4 * pointA.fst) :=
by
  existsi 2
  split
  norm_num
  split
  refl
  split
  sorry

theorem area_of_triangle_OMN :
  (∀ F: ℝ × ℝ, F = (1 : ℝ, 0 : ℝ) ∧
  ∀ (x1 y1 x2 y2 : ℝ), 
    (lineThroughFocus45Degrees F y1 = parabola 2 y1 x1) ∧
    (lineThroughFocus45Degrees F y2 = parabola 2 y2 x2) ∧
    (1/2) * |distance x1 y1 x2 y2| * |distance 0 0 F.fst F.snd| = 2 * real.sqrt 2) :=
by
  sorry

end equation_of_parabola_from_point_area_of_triangle_OMN_l800_800277


namespace sum_of_ages_l800_800370

-- Definitions based on conditions
variables (J S : ℝ) -- J and S are real numbers

-- First condition: Jane is five years older than Sarah
def jane_older_than_sarah := J = S + 5

-- Second condition: Nine years from now, Jane will be three times as old as Sarah was three years ago
def future_condition := J + 9 = 3 * (S - 3)

-- Conclusion to prove
theorem sum_of_ages (h1 : jane_older_than_sarah J S) (h2 : future_condition J S) : J + S = 28 :=
by
  sorry

end sum_of_ages_l800_800370


namespace power_function_evaluation_l800_800256

theorem power_function_evaluation (f : ℝ → ℝ) (α : ℝ) (h : ∀ x, f x = x ^ α) (h_point : f 4 = 2) : f 16 = 4 :=
by
  sorry

end power_function_evaluation_l800_800256


namespace quadratic_square_binomial_l800_800185

theorem quadratic_square_binomial (k : ℝ) : 
  (∃ a : ℝ, (x : ℝ) → x^2 - 20 * x + k = (x + a)^2) ↔ k = 100 := 
by
  sorry

end quadratic_square_binomial_l800_800185


namespace team_selection_l800_800860

theorem team_selection :
  let girls := 4
  let boys := 6
  choose 2 girls * choose 3 boys = 120 :=
by 
  sorry

end team_selection_l800_800860


namespace prob_iff_eq_l800_800730

noncomputable def A (m : ℝ) : Set ℝ := { x | x^2 + m * x + 2 ≥ 0 ∧ x ≥ 0 }
noncomputable def B (m : ℝ) : Set ℝ := { y | ∃ x, x ∈ A m ∧ y = Real.sqrt (x^2 + m * x + 2) }

theorem prob_iff_eq (m : ℝ) : (A m = { y | ∃ x, x ^ 2 + m * x + 2 = y ^ 2 ∧ x ≥ 0 } ↔ m = -2 * Real.sqrt 2) :=
by
  sorry

end prob_iff_eq_l800_800730


namespace combined_surface_area_composite_shape_l800_800882

-- Definitions of the given conditions
def surface_area_sphere (r : ℝ) : ℝ := 4 * π * r^2
def radius_original_hemisphere : ℝ := 10
def spherical_cap_height : ℝ := radius_original_hemisphere - 6
def radius_spherical_cap_base : ℝ := real.sqrt (radius_original_hemisphere^2 - spherical_cap_height^2)
def area_base_original_hemisphere : ℝ := π * radius_original_hemisphere^2

-- Problem Statement to prove the area of the composite shape
theorem combined_surface_area_composite_shape : 
  (2 * π * radius_spherical_cap_base * spherical_cap_height + π * 6^2) = 100 * π :=
by 
  -- placeholder for proof
  sorry

end combined_surface_area_composite_shape_l800_800882


namespace segment_length_tangent_circles_l800_800901

theorem segment_length_tangent_circles
  (r1 r2 : ℝ)
  (h1 : r1 > 0)
  (h2 : r2 > 0)
  (h3 : 7 - 4 * Real.sqrt 3 ≤ r1 / r2)
  (h4 : r1 / r2 ≤ 7 + 4 * Real.sqrt 3)
  :
  ∃ d : ℝ, d^2 = (1 / 12) * (14 * r1 * r2 - r1^2 - r2^2) :=
sorry

end segment_length_tangent_circles_l800_800901


namespace delta_value_l800_800752

theorem delta_value (Δ : ℤ) (h : 4 * -3 = Δ - 3) : Δ = -9 :=
sorry

end delta_value_l800_800752


namespace constant_function_no_monotonicity_l800_800321

theorem constant_function_no_monotonicity {f: ℝ → ℝ} (I : set ℝ) (hI : is_interval I) :
  (∀ x ∈ I, deriv f x = 0) → (∀ x ∈ I, ∀ y ∈ I, f x = f y) :=
by
  sorry

end constant_function_no_monotonicity_l800_800321


namespace solve_k_l800_800656

noncomputable def is_solution (f : ℤ → ℤ) (k : ℕ) : Prop :=
∀ (a b c : ℤ), (a + b + c = 0) → 
  (f(a) + f(b) + f(c) = (f(a - b) + f(b - c) + f(c - a)) / k)

theorem solve_k : { k : ℕ | ∃ f : ℤ → ℤ, is_solution f k ∧ ¬is_linear f } = {1, 3, 9} := sorry

end solve_k_l800_800656


namespace martin_total_distance_l800_800409

-- Define the conditions
def total_trip_time : ℕ := 8
def first_half_speed : ℕ := 70
def second_half_speed : ℕ := 85
def half_trip_time : ℕ := total_trip_time / 2

-- Define the total distance traveled 
def total_distance : ℕ := (first_half_speed * half_trip_time) + (second_half_speed * half_trip_time)

-- Statement to prove
theorem martin_total_distance : total_distance = 620 :=
by
  -- This is a placeholder to represent that a proof is needed
  -- Actual proof steps are omitted as instructed
  sorry

end martin_total_distance_l800_800409


namespace number_of_new_bricks_l800_800116

-- Definitions from conditions
def edge_length_original_brick : ℝ := 0.3
def edge_length_new_brick : ℝ := 0.5
def number_original_bricks : ℕ := 600

-- The classroom volume is unchanged, so we set up a proportion problem
-- Assuming the classroom is fully paved
theorem number_of_new_bricks :
  let volume_original_bricks := number_original_bricks * (edge_length_original_brick ^ 2)
  let volume_new_bricks := x * (edge_length_new_brick ^ 2)
  volume_original_bricks = volume_new_bricks → x = 216 := 
by
  sorry

end number_of_new_bricks_l800_800116


namespace range_m_triangle_sides_l800_800722

noncomputable def f (x m : ℝ) : ℝ := (cos x + m) / (cos x + 2)

theorem range_m_triangle_sides : 
  (∀ a b c : ℝ, let fa := f a m, let fb := f b m, let fc := f c m in 
    fa + fb > fc ∧ fb + fc > fa ∧ fc + fa > fb) ↔ m ∈ set.Ioo (7 / 5 : ℝ) 5 :=
sorry

end range_m_triangle_sides_l800_800722


namespace bus_stop_time_l800_800194

-- Define the conditions as constants
def speed_without_stoppages : ℝ := 64 / 60 -- in km/min
def speed_with_stoppages : ℝ := 48 / 60 -- in km/min
def distance_difference : ℝ := speed_without_stoppages - speed_with_stoppages

-- Definition of time stopped per hour based on conditions
def time_stopped_per_hour : ℝ := distance_difference / speed_without_stoppages * 60

-- Target proof
theorem bus_stop_time :
  time_stopped_per_hour = 15 :=
by
  sorry

end bus_stop_time_l800_800194


namespace joe_money_left_l800_800309

def initial_pocket_money : ℝ := 450
def fraction_spent_on_chocolates : ℝ := 1 / 9
def fraction_spent_on_fruits : ℝ := 2 / 5
def amount_spent_on_chocolates : ℝ := fraction_spent_on_chocolates * initial_pocket_money
def amount_spent_on_fruits : ℝ := fraction_spent_on_fruits * initial_pocket_money
def amount_left (initial : ℝ) (spent_chocolates : ℝ) (spent_fruits : ℝ) : ℝ := 
  initial - spent_chocolates - spent_fruits

theorem joe_money_left : amount_left initial_pocket_money amount_spent_on_chocolates amount_spent_on_fruits = 220 :=
  by
    sorry

end joe_money_left_l800_800309


namespace peaches_at_stand_l800_800417

-- Define the given conditions and the question as the theorem we need to prove
theorem peaches_at_stand (peaches_picked total_peaches initial_peaches : ℕ)
  (h1 : peaches_picked = 52)
  (h2 : total_peaches = 86) :
  initial_peaches = total_peaches - peaches_picked :=
begin
  -- The necessary proof is skipped with sorry
  sorry
end

end peaches_at_stand_l800_800417


namespace area_of_PQFT_l800_800624

-- Assume that a square ABCD with side length 1 and a point E inside the square are given.
constant A B C D E P Q F T : Prop
-- Assume ABCD forms a square
axiom square_ABCD : ∀ (A B C D : Prop), Prop
-- Assume E is inside the square
axiom E_inside_square : ∀ (A B C D E: Prop), Prop
-- Define that P, Q, F, T are points of intersection of medians of the respective triangles
axiom median_intersections : ∀ (B C D E A P Q F T: Prop), (intersection_medians_triangle B C E P) ∧ (intersection_medians_triangle C D E Q) ∧ (intersection_medians_triangle D A E F) ∧ (intersection_medians_triangle A B E T)

-- The proof to show the area of PQFT is 2/9.
theorem area_of_PQFT : area_quadrilateral P Q F T = 2 / 9 := sorry

end area_of_PQFT_l800_800624


namespace degree_f_x2_mul_g_x4_l800_800389

-- Define a polynomial f of degree 3
noncomputable def f : Polynomial ℝ := sorry

-- Define a polynomial g of degree 6
noncomputable def g : Polynomial ℝ := sorry

-- Define the degrees of f and g
axiom deg_f : degree f = 3
axiom deg_g : degree g = 6

-- State the theorem
theorem degree_f_x2_mul_g_x4 :
  degree (f.comp (Polynomial.C 1 - Polynomial.X^2) * (g.comp (Polynomial.C 1 - Polynomial.X^4))) = 30 :=
by
  -- skipping the proof
  sorry

end degree_f_x2_mul_g_x4_l800_800389


namespace probability_sum_divisible_by_three_l800_800326

open Set
open Finset

-- Defining the set of first nine prime numbers.
def first_nine_primes : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23}

-- Condition: Two distinct numbers are selected at random.
def distinct_pairs (s : Finset ℕ) : Finset (ℕ × ℕ) :=
  (s.product s).filter (λ p => p.1 < p.2)

-- Definition of pairs whose sum is divisible by 3.
def divisible_by_three_pairs (s : Finset ℕ) : Finset (ℕ × ℕ) :=
  (distinct_pairs s).filter (λ p => (p.1 + p.2) % 3 = 0)

-- Theorem statement.
theorem probability_sum_divisible_by_three :
  (divisible_by_three_pairs first_nine_primes).card.toRat /
    (distinct_pairs first_nine_primes).card.toRat = 2 / 9 :=
by
  sorry

end probability_sum_divisible_by_three_l800_800326


namespace largest_four_digit_number_property_l800_800069

theorem largest_four_digit_number_property : ∃ (a b c d : Nat), 
  (1000 * a + 100 * b + 10 * c + d = 9099) ∧ 
  (c = a + b) ∧ 
  (d = b + c) ∧ 
  1 ≤ a ∧ a ≤ 9 ∧ 
  0 ≤ b ∧ b ≤ 9 ∧ 
  0 ≤ c ∧ c ≤ 9 ∧ 
  0 ≤ d ∧ d ≤ 9 := 
sorry

end largest_four_digit_number_property_l800_800069


namespace inequality_abc_l800_800394

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : (a + b) * (b + c) * (c + a) = 1) :
    (a^2 / (1 + real.sqrt (b * c)) + b^2 / (1 + real.sqrt (c * a)) + c^2 / (1 + real.sqrt (a * b)) ≥ 1 / 2) :=
  sorry

end inequality_abc_l800_800394


namespace ellipse_eq_l800_800664

theorem ellipse_eq (foci_on_x_axis : Prop)
  (major_eq_3times_minor : Prop)
  (passes_through_A : Prop)
  (center_at_origin : Prop)
  (coord_axes_symmetry : Prop)
  (passes_through_P1 : Prop)
  (passes_through_P2 : Prop) :
  (foci_on_x_axis ∧ major_eq_3times_minor ∧ passes_through_A ∧ center_at_origin ∧ coord_axes_symmetry ∧ passes_through_P1 ∧ passes_through_P2) →
  ∃ a b : ℝ, a = 3 * b ∧ ellipse_eq x y a b := by
  sorry

-- We can define the conditions as follows
def foci_on_x_axis : Prop := true  -- Placeholder, complete with the actual condition
def major_eq_3times_minor : Prop := ∀ a b : ℝ, a = 3 * b
def passes_through_A : Prop := A = (3, 0)
def center_at_origin : Prop := center = (0, 0)
def coord_axes_symmetry : Prop := true  -- Placeholder, complete with the actual condition
def passes_through_P1 : Prop := P1 = (sqrt 6, 1)
def passes_through_P2 : Prop := P2 = (- sqrt 3, - sqrt 2)

-- Equation for ellipse
def ellipse_eq (x y a b : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

end ellipse_eq_l800_800664


namespace remainder_sum_first_20_div_9_l800_800520

theorem remainder_sum_first_20_div_9 :
  let S := (20 * 21) / 2
  in S % 9 = 3 :=
by
  let S := (20 * 21) / 2
  show S % 9 = 3
  sorry

end remainder_sum_first_20_div_9_l800_800520


namespace find_1993_star_1935_l800_800895

axiom star (x y : ℕ) : ℕ

axiom star_self {x : ℕ} : star x x = 0
axiom star_assoc {x y z : ℕ} : star x (star y z) = star x y + z

theorem find_1993_star_1935 : star 1993 1935 = 58 :=
by
  sorry

end find_1993_star_1935_l800_800895


namespace range_a_iff_l800_800231

variable (f : ℝ → ℝ) (a : ℝ)

def f_def (x : ℝ) : ℝ := a * (exp x) / x - x

theorem range_a_iff (x₁ x₂ : ℝ) (H₁ : 0 < x₁) (H₂ : 0 < x₂) (H₃ : x₁ < x₂) :
  (f x₁ / x₂ - f x₂ / x₁ < 0) ↔ a ≥ 2 / exp 1 :=
sorry

end range_a_iff_l800_800231


namespace distance_CD_l800_800822

def distance_cartesian (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

variable (φ1 φ2 : ℝ)
noncomputable def C : ℝ × ℝ := (5 * Real.cos φ1, 5 * Real.sin φ1)
noncomputable def D : ℝ × ℝ := (7 * Real.cos φ2, 7 * Real.sin φ2)

theorem distance_CD : 
  φ1 - φ2 = Real.pi / 3 → 
  distance_cartesian (C φ1).1 (C φ1).2 (D φ2).1 (D φ2).2 = Real.sqrt 39 :=
by
  sorry

end distance_CD_l800_800822


namespace problem1_condition1_problem2_condition2_l800_800214

noncomputable theory

def condition1_center_on_y_eq_x (m : ℝ) : Prop :=
  m = 4 ∨ m = 8

def condition2_center_coordinates (m : ℝ) (x y : ℝ) : Prop :=
  m = x ∧ m = y

def condition1_tangent_to_y_eq_6 (m : ℝ) (r : ℝ) : Prop :=
  |m - 6| = r

def condition2_through_points (D E F : ℝ) : Prop :=
  4 * D + 3 * E + F = -25 ∧
  5 * D + 2 * E + F = -29 ∧
  D + F = -1

theorem problem1_condition1 (m : ℝ) :
  condition1_center_on_y_eq_x m → 
  condition2_center_coordinates m m m → 
  condition1_tangent_to_y_eq_6 m 2 → 
  ((m = 4 ∧ ∀ x y, (x - 4) ^ 2 + (y - 4) ^ 2 = 4) ∨ 
   (m = 8 ∧ ∀ x y, (x - 8) ^ 2 + (y - 8) ^ 2 = 4)) :=
sorry

theorem problem2_condition2 (D E F : ℝ) :
  condition2_through_points D E F → 
  D = -6 ∧ E = -2 ∧ F = 5 ∧ ∀ x y, x^2 + y^2 - 6 * x - 2 * y + 5 = 0 :=
sorry

end problem1_condition1_problem2_condition2_l800_800214


namespace largest_four_digit_number_prop_l800_800051

theorem largest_four_digit_number_prop :
  ∃ (a b c d : ℕ), a = 9 ∧ b = 0 ∧ c = 9 ∧ d = 9 ∧ (1000 * a + 100 * b + 10 * c + d = 9099) ∧ (c = a + b) ∧ (d = b + c) :=
by
  sorry

end largest_four_digit_number_prop_l800_800051


namespace collinear_points_on_sphere_l800_800880

-- Definitions related to the geometric configuration
variable (α : Type*) [EuclideanSpace α ℝ]

-- Point Definitions
variables (A B C S A1 B1 C1 A2 B2 C2 : α)

-- Sphere Definitions
variable (ω Ω : Sphere α ℝ)

-- Hypotheses as per given conditions
variable (h1 : ω.contains S)
variable (h2 : ω.intersects (line_through S A) A1)
variable (h3 : ω.intersects (line_through S B) B1)
variable (h4 : ω.intersects (line_through S C) C1)
variable (h5 : Ω.circumscribes_pyramid (S A B C))
variable (h6 : ∃ (μ : Circle α ℝ), μ ⊂ Ω ∩ ω ∧ μ a_plane_parallel (plane A B C))
variable (h7 : A2.symmetric_wrt_midpoint (S A) A1)
variable (h8 : B2.symmetric_wrt_midpoint (S B) B1)
variable (h9 : C2.symmetric_wrt_midpoint (S C) C1)

-- Desired proof statement
theorem collinear_points_on_sphere :
  sphere_through [A, B, C, A2, B2, C2] :=
by
  sorry -- Proof is omitted

end collinear_points_on_sphere_l800_800880


namespace min_value_of_max_exp_l800_800903

theorem min_value_of_max_exp : 
  ∃ (x y : ℝ), (max (x^2 + |y|) ((x + 2)^2 + |y|) (x^2 + |y - 1|)) = 1.5 ∧ 
  ∀ (x y : ℝ), 1.5 ≤ max (x^2 + |y|) ((x + 2)^2 + |y|) (x^2 + |y - 1|) := 
by
  sorry

end min_value_of_max_exp_l800_800903


namespace sandwich_combinations_l800_800142

theorem sandwich_combinations (meats cheeses : Finset ℕ) (h_meats : meats.card = 12) (h_cheeses : cheeses.card = 8) :
  (meats.card.choose 2) * cheeses.card.choose 1 = 528 :=
by
  simp only [Finset.card_choose, h_meats, h_cheeses]
  have h1 : 12.choose 2 = 66 := by norm_num
  have h2 : 8.choose 1 = 8 := by norm_num
  rw [h1, h2]
  norm_num
  exact rfl

end sandwich_combinations_l800_800142


namespace largest_constant_for_interesting_sequences_l800_800934

noncomputable def is_interesting_sequence (z : ℕ → ℂ) : Prop :=
(z 1).abs = 1 ∧ ∀ n : ℕ, 0 < n → 4 * (z (n + 1))^2 + 2 * (z n) * (z (n + 1)) + (z n)^2 = 0

theorem largest_constant_for_interesting_sequences (z : ℕ → ℂ) (m : ℕ) (h : is_interesting_sequence z) (hm : 0 < m) : 
  abs (finset.sum (finset.range m) (λ n, z (n + 1))) ≥ real.sqrt 3 / 3 :=
sorry

end largest_constant_for_interesting_sequences_l800_800934


namespace sin_alpha_plus_beta_l800_800703

theorem sin_alpha_plus_beta (α β : ℝ) (h₁ : 0 < β ∧ β < π / 4 ∧ π / 4 < α ∧ α < 3 * π / 4)
    (h₂ : cos (π / 4 - α) = 3 / 5)
    (h₃ : sin (3 * π / 4 + β) = 5 / 13) :
    sin (α + β) = 56 / 65 := by
  sorry

end sin_alpha_plus_beta_l800_800703


namespace solve_hyperbola_eq_and_k_range_l800_800688

def ellipse := (x y : ℝ) → x^2 / 8 + y^2 / 4 = 1

def hyperbola := (x y : ℝ) → x^2 / 3 - y^2 = 1

def intersects_two_points (l : ℝ → ℝ) (C : ℝ → ℝ → Prop) :=
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧
  ∃ (y1 y2 : ℝ), y1 = l x1 ∧ y2 = l x2 ∧
  C x1 y1 ∧ C x2 y2

def dot_product_gt_two (O : ℝ × ℝ) (A B : ℝ × ℝ) :=
  let ⟨x1, y1⟩ := A in
  let ⟨x2, y2⟩ := B in
  (x1 * x2 + y1 * y2 > 2)

theorem solve_hyperbola_eq_and_k_range :
  (∀ (x y : ℝ), ellipse x y → hyperbola x y) ∧
  (∀ (k : ℝ), intersects_two_points (λ x, k * x + real.sqrt 2) hyperbola →
    (∀ (x1 x2 : ℝ), x1 ≠ x2 →
      (dot_product_gt_two (0, 0) (x1, k * x1 + real.sqrt 2) (x2, k * x2 + real.sqrt 2)) →
      k ∈ set.Ioo (-1 : ℝ) (-real.sqrt 3 / 3) ∨ k ∈ set.Ioo (real.sqrt 3 / 3) 1))
  :=
by sorry

end solve_hyperbola_eq_and_k_range_l800_800688


namespace measure_angle_DOA_l800_800791

/-- In the given figure, with \( O \) as the center of the circle and \( AB \parallel CD \), 
  the quadrilateral \(ADCB\) is an inscribed isosceles trapezoid with \(\angle BAD = \angle CBA = 63^\circ\). 
  Additionally, triangle \(DOA\) is isosceles with \(OA = OD\). Prove that the angle \(\angle DOA\) is \(54^\circ\). -/
theorem measure_angle_DOA :
  ∀ (O A B C D : Point) (h1 : center O (⊙O))
    (h2 : parallel AB CD) (h3 : inscribed_trapezoid A D C B (⊙O))
    (h4 : isosceles_trapezoid A D C B) (h5 : internal_angle A B 63 ∧ internal_angle C B 63)
    (h6 : isosceles_triangle O A D),
  measure_angle DOA = 54 :=
by
  sorry

end measure_angle_DOA_l800_800791


namespace time_to_watch_all_episodes_l800_800802

theorem time_to_watch_all_episodes 
    (n_seasons : ℕ) (episodes_per_season : ℕ) (last_season_extra_episodes : ℕ) (hours_per_episode : ℚ)
    (h1 : n_seasons = 9)
    (h2 : episodes_per_season = 22)
    (h3 : last_season_extra_episodes = 4)
    (h4 : hours_per_episode = 0.5) :
    n_seasons * episodes_per_season + (episodes_per_season + last_season_extra_episodes) * hours_per_episode = 112 :=
by
  sorry

end time_to_watch_all_episodes_l800_800802


namespace range_of_a_l800_800213

theorem range_of_a (a : ℝ) :  (5 - a > 0) ∧ (a - 2 > 0) ∧ (a - 2 ≠ 1) → (2 < a ∧ a < 3) ∨ (3 < a ∧ a < 5) :=
by
  intro h
  sorry

end range_of_a_l800_800213


namespace exists_int_less_than_sqrt_twenty_three_l800_800426

theorem exists_int_less_than_sqrt_twenty_three : ∃ n : ℤ, n < Real.sqrt 23 := 
  sorry

end exists_int_less_than_sqrt_twenty_three_l800_800426


namespace solution_set_ln_inequality_correct_l800_800023

noncomputable def solution_set_ln_inequality : set ℝ :=
  {x | ∀ (x : ℝ), 0 < x → (ln x)^2 + ln x < 0 → e⁻¹ < x ∧ x < 1}

theorem solution_set_ln_inequality_correct :
  { x : ℝ | 0 < x ∧ (ln x)^2 + ln x < 0 } = { x : ℝ | e⁻¹ < x ∧ x < 1 } :=
by sorry

end solution_set_ln_inequality_correct_l800_800023


namespace plane_equation_l800_800828

noncomputable def plane_L_intersection (a b c d : ℝ) (x y z : ℝ) : Prop :=
  a * x + b * y + c * z = d

theorem plane_equation
  (x y z : ℝ)
  (L1 : plane_L_intersection 1 1 2 4 x y z)
  (L2 : plane_L_intersection 2 -2 1 1 x y z)
  (P : plane_L_intersection 6 0 5 9 x y z)
  (dist : ℝ := (abs ((6 * 1) + (0 * 2) + (5 * 3) - 9)) / (sqrt ((6^2) + (0^2) + (5^2))) = 1) :
  ∃ (P : ℝ → ℝ → ℝ → Prop), P 6 0 5 9 x y z :=
  sorry

end plane_equation_l800_800828


namespace leap_day_2024_l800_800805

theorem leap_day_2024 (h : day_of_week 1996 2 29 = Thursday) : day_of_week 2024 2 29 = Thursday :=
sorry

end leap_day_2024_l800_800805


namespace hockey_league_teams_l800_800500

theorem hockey_league_teams (n : ℕ) (h : (n * (n - 1) * 10) / 2 = 1710) : n = 19 :=
by {
  sorry
}

end hockey_league_teams_l800_800500


namespace tan_squared_arccot_l800_800973

noncomputable def arccot (x : ℝ) : ℝ :=
if x = 0 then π / 2 else Real.arctan (1 / x)

theorem tan_squared_arccot (a b : ℝ) (ha : a = 3) (hb : b = 5) :
  (Real.tan (arccot (a / b)))^2 = 25 / 9 :=
by
  sorry

end tan_squared_arccot_l800_800973


namespace modulus_of_complex_main_l800_800982

variable (a b : Real)
hypothesis (h1 : a = 3)
hypothesis (h2 : b = -10 * Real.sqrt 3)

theorem modulus_of_complex :
  Complex.abs ⟨a, b⟩ = Real.sqrt (a^2 + b^2) := sorry

theorem main :
  Complex.abs ⟨3, -10 * Real.sqrt 3⟩ = Real.sqrt 309 := by
  have h1 : a = 3 := by rfl
  have h2 : b = -10 * Real.sqrt 3 := by rfl
  have hab : Complex.abs ⟨a, b⟩ = Real.sqrt (a^2 + b^2) := modulus_of_complex a b h1 h2
  show Complex.abs ⟨3, -10 * Real.sqrt 3⟩ = Real.sqrt 309 from
    by rw [h1, h2, hab, sq (a^2 + b^2)]

end modulus_of_complex_main_l800_800982


namespace rectangle_area_l800_800537

theorem rectangle_area :
  ∃ (b l : ℕ), l = 3 * b ∧ 2 * (l + b) = 104 ∧ l * b = 507 :=
begin
  sorry
end

end rectangle_area_l800_800537


namespace sum_of_primitive_roots_mod_11_l800_800691

def is_primitive_root (g : ℕ) (p : ℕ) : Prop :=
  ∀ k, 1 ≤ k < p → (g ^ k) % p ≠ 1

def primitive_roots_mod_p (n : ℕ) (p : ℕ) : List ℕ :=
  (List.range n).filter (λ x => is_primitive_root (x + 1) p)

theorem sum_of_primitive_roots_mod_11 :
  let p := 11
  let primitive_roots := primitive_roots_mod_p 10 p
  primitive_roots.sum = 23 := 
by
  sorry

end sum_of_primitive_roots_mod_11_l800_800691


namespace james_budget_spending_l800_800800

theorem james_budget_spending :
  let budget := 1000
  let food := 0.3 * budget
  let accommodation := 0.15 * budget
  let entertainment := 0.2 * budget
  let transportation := 0.1 * budget
  let clothes := 0.05 * budget
  let coursework_materials := budget - (food + accommodation + entertainment + transportation + clothes)
  let combined_percentage := 0.2 + 0.1 + (coursework_materials / budget)
  let combined_amount := entertainment + transportation + coursework_materials
  combined_percentage = 0.5 ∧ combined_amount = 500 :=
by {
  let budget := 1000
  let food := 0.3 * budget
  let accommodation := 0.15 * budget
  let entertainment := 0.2 * budget
  let transportation := 0.1 * budget
  let clothes := 0.05 * budget
  let coursework_materials := budget - (food + accommodation + entertainment + transportation + clothes)
  let combined_percentage := 0.2 + 0.1 + (coursework_materials / budget)
  let combined_amount := entertainment + transportation + coursework_materials
  have hp : combined_percentage = 0.5, from sorry,
  have ha : combined_amount = 500, from sorry,
  exact ⟨hp, ha⟩
}

end james_budget_spending_l800_800800


namespace blueberry_picking_l800_800912

-- Define the amounts y1 and y2 as a function of x
variable (x : ℝ)
def y1 : ℝ := 60 + 18 * x
def y2 : ℝ := 150 + 15 * x

-- State the theorem about the relationships given the condition 
theorem blueberry_picking (hx : x > 10) : 
  y1 x = 60 + 18 * x ∧ y2 x = 150 + 15 * x :=
by
  sorry

end blueberry_picking_l800_800912


namespace probability_all_white_is_correct_l800_800885

-- Define the total number of balls
def total_balls : ℕ := 25

-- Define the number of white balls
def white_balls : ℕ := 10

-- Define the number of black balls
def black_balls : ℕ := 15

-- Define the number of balls drawn
def balls_drawn : ℕ := 4

-- Define combination function
def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Total ways to choose 4 balls from 25
def total_ways : ℕ := C total_balls balls_drawn

-- Ways to choose 4 white balls from 10 white balls
def white_ways : ℕ := C white_balls balls_drawn

-- Probability that all 4 drawn balls are white
def prob_all_white : ℚ := white_ways / total_ways

theorem probability_all_white_is_correct :
  prob_all_white = (3 : ℚ) / 181 := by
  -- Proof statements go here
  sorry

end probability_all_white_is_correct_l800_800885


namespace sufficient_not_necessary_condition_for_q_l800_800233

theorem sufficient_not_necessary_condition_for_q (p q : Prop)
  (h_p : ∀ (m : ℝ), (∀ x y : ℝ, 0 < x → x < y → f(x) = x^2 + 2*m*x + 1 → f(y) = y^2 + 2*m*y + 1 → f(x) ≤ f(y)))
  (h_q : ∀ m : ℝ, m ≥ -5) :
  (∀ m : ℝ, p → q) ∧ (∃ m : ℝ, ¬ (q → p)) :=
by
  sorry

end sufficient_not_necessary_condition_for_q_l800_800233


namespace shopkeeper_gain_is_8_51_percent_l800_800590

def false_weight_A : ℝ := 950 / 1000
def false_weight_B : ℝ := 900 / 1000
def false_weight_C : ℝ := 920 / 1000

def cost_price_A : ℝ := 20
def cost_price_B : ℝ := 25
def cost_price_C : ℝ := 18

def amount_sold_A : ℝ := 10
def amount_sold_B : ℝ := 10
def amount_sold_C : ℝ := 10

noncomputable def shopkeeper_percentage_gain : ℝ :=
  let actual_weight_A := false_weight_A * amount_sold_A
  let actual_weight_B := false_weight_B * amount_sold_B
  let actual_weight_C := false_weight_C * amount_sold_C
  let total_cost_A := actual_weight_A * cost_price_A
  let total_cost_B := actual_weight_B * cost_price_B
  let total_cost_C := actual_weight_C * cost_price_C
  let total_cost_price := total_cost_A + total_cost_B + total_cost_C
  let selling_price_A := cost_price_A * amount_sold_A
  let selling_price_B := cost_price_B * amount_sold_B
  let selling_price_C := cost_price_C * amount_sold_C
  let total_selling_price := selling_price_A + selling_price_B + selling_price_C
  let gain := total_selling_price - total_cost_price
  (gain / total_cost_price) * 100

theorem shopkeeper_gain_is_8_51_percent : shopkeeper_percentage_gain ≈ 8.51 := by
  -- Proof goes here
  sorry

end shopkeeper_gain_is_8_51_percent_l800_800590


namespace samantha_total_cost_l800_800161

-- Defining the conditions in Lean
def washer_cost : ℕ := 4
def dryer_cost_per_10_min : ℕ := 25
def loads : ℕ := 2
def num_dryers : ℕ := 3
def dryer_time : ℕ := 40

-- Proving the total cost Samantha spends is $11
theorem samantha_total_cost : (loads * washer_cost + num_dryers * (dryer_time / 10 * dryer_cost_per_10_min)) = 1100 :=
by
  sorry

end samantha_total_cost_l800_800161


namespace ratio_of_areas_of_squares_l800_800114

theorem ratio_of_areas_of_squares (a : ℝ) : 
  let r := a / 2 in
  let b := a / Real.sqrt 2 in
  let A1 := a ^ 2 in
  let A2 := b ^ 2 in
  (A1 / A2) = 2 :=
by
  let r := a / 2
  let b := a / Real.sqrt 2
  let A1 := a ^ 2
  let A2 := b ^ 2
  sorry

end ratio_of_areas_of_squares_l800_800114


namespace compute_expression_l800_800305

noncomputable def c : ℝ := Real.log 8
noncomputable def d : ℝ := Real.log 25

theorem compute_expression : 5^(c / d) + 2^(d / c) = 2 * Real.sqrt 2 + 5^(2 / 3) :=
by
  sorry

end compute_expression_l800_800305


namespace olivia_final_distance_l800_800418

noncomputable def olivia_walk_distance_proof : ℝ :=
let meters_to_feet (m : ℝ) := m * 3.28084 in
let north_distance := meters_to_feet 15 in
let south_distance := meters_to_feet 15 + 48 in
let net_south := south_distance - north_distance in
let east_distance := 40 in
let distance := Math.sqrt (net_south ^ 2 + east_distance ^ 2) in
distance

theorem olivia_final_distance : round olivia_walk_distance_proof = 105 :=
by
  calc
    let north_distance := meters_to_feet 15
    let south_distance := meters_to_feet 15 + 48
    let net_south := south_distance - north_distance
    let east_distance := 40
    let distance := Math.sqrt (net_south ^ 2 + east_distance ^ 2)
    round distance = 105 := 
      sorry

end olivia_final_distance_l800_800418


namespace solve_congruence_l800_800022

theorem solve_congruence :
  ∃ a m : ℕ, (8 * (x : ℕ) + 1) % 12 = 5 % 12 ∧ m ≥ 2 ∧ a < m ∧ x ≡ a [MOD m] ∧ a + m = 5 :=
by
  sorry

end solve_congruence_l800_800022


namespace samantha_laundromat_cost_l800_800163

-- Definitions of given conditions
def washer_cost : ℕ := 4
def dryer_cost_per_10_min : ℝ := 0.25
def num_washes : ℕ := 2
def num_dryers : ℕ := 3
def dryer_time : ℕ := 40

-- Calculate total cost
def washing_cost : ℝ := washer_cost * num_washes
def intervals_10min : ℕ := dryer_time / 10
def single_dryer_cost : ℝ := dryer_cost_per_10_min * intervals_10min
def total_drying_cost : ℝ := single_dryer_cost * num_dryers
def total_cost : ℝ := washing_cost + total_drying_cost

-- The statement to prove
theorem samantha_laundromat_cost : total_cost = 11 :=
by
  unfold washer_cost dryer_cost_per_10_min num_washes num_dryers dryer_time washing_cost intervals_10min single_dryer_cost total_drying_cost total_cost
  norm_num
  done

end samantha_laundromat_cost_l800_800163


namespace quadratic_root_neg3_l800_800422

theorem quadratic_root_neg3 : ∃ x : ℝ, x^2 - 9 = 0 ∧ (x = -3) :=
by
  sorry

end quadratic_root_neg3_l800_800422


namespace two_by_two_square_exists_l800_800381

theorem two_by_two_square_exists {n : ℕ} (h : n ≥ 2) :
  ∃ (table : Fin n × Fin n → ℕ),
  (∀ i : Fin n, ∀ j : Fin n, table (i, j) ∈ Finset.range (n ^ 2 + 1)) ∧
  (∀ i : Fin n, ∀ j : Fin (n - 1), (table (i, j) - table (i, j + 1)).natAbs ≤ n) ∧
  (∀ i : Fin (n - 1), ∀ j : Fin n, (table (i, j) - table (i + 1, j)).natAbs ≤ n) ∧
  (∃ i j : Fin (n - 1), table (i, j) + table (i + 1, j + 1) = table (i, j + 1) + table (i + 1, j)) :=
by
  sorry

end two_by_two_square_exists_l800_800381


namespace money_left_l800_800621

variables (x n y : ℝ)

-- Total amount of money
-- x is the total amount of money Alex earned.
def total_money (x : ℝ) := x

-- Cost condition based on given problem
-- Alex uses one-fourth of his money to buy one-half of the video games.
def cost_condition (x n y : ℝ) := (1/4)*x = (1/2) * n * y

-- Cost of accessories
-- Alex also spends one-sixth of his money on gaming accessories.
def accessories_cost (x : ℝ) := (1/6) * x

-- Fraction of money left
-- Prove Alex has 1/3 of his money left
theorem money_left (x n y : ℝ) (hx1 : total_money x) (hx2 : cost_condition x n y) (hx3 : accessories_cost x) :
  x - (1/2)*x - (1/6)*x = (1/3) * x :=
by
  sorry

end money_left_l800_800621


namespace find_g2_l800_800476

theorem find_g2 (g : ℝ → ℝ) (h : ∀ x : ℝ, x ≠ 0 → g x - 3 * g (1 / x) = 3 ^ x) : 
  g 2 = (9 - 3 * Real.sqrt 3) / 8 := 
sorry

end find_g2_l800_800476


namespace evaluate_g_at_3_l800_800760

def g (x : ℝ) := 3 * x ^ 4 - 5 * x ^ 3 + 4 * x ^ 2 - 7 * x + 2

theorem evaluate_g_at_3 : g 3 = 125 :=
by
  -- Proof omitted for this exercise.
  sorry

end evaluate_g_at_3_l800_800760


namespace intervals_of_monotonicity_range_of_a_for_three_zeros_l800_800270

    -- Define the function f(x) = 1/2 * x^2 - 3 * a * x + 2 * a^2 * ln(x)
    noncomputable def f (a x : ℝ) : ℝ := (1/2) * x^2 - 3 * a * x + 2 * a^2 * Real.log x

    -- Prove the intervals of monotonicity for f(x)
    theorem intervals_of_monotonicity (a : ℝ) (h : a ≠ 0) :
      (a > 0 → 
         monotone_on (f a) (Ioo 0 a) ∧
         antitone_on (f a) (Ioo a (2 * a)) ∧
         monotone_on (f a) (Ioi (2 * a))) ∧
      (a < 0 → 
         monotone_on (f a) (Ioi 0)) := 
    sorry

    -- Define the function's number of zeros and check the range of 'a' for having 3 zeros
    noncomputable def number_of_zeros (f : ℝ → ℝ) : ℕ := 
      -- A placeholder function representing the number of zeros of f
      sorry 

    theorem range_of_a_for_three_zeros :
      ∃ a : ℝ, (e^(5/4) < a ∧ a < (e^2 / 2)) ∧ number_of_zeros (f a) = 3 :=
    sorry
    
end intervals_of_monotonicity_range_of_a_for_three_zeros_l800_800270


namespace find_units_digit_l800_800216

theorem find_units_digit : 
  (7^1993 + 5^1993) % 10 = 2 :=
by
  sorry

end find_units_digit_l800_800216


namespace count_congruent_3_mod_8_l800_800740

theorem count_congruent_3_mod_8 (n : ℕ) (h₁ : 1 ≤ n) (h₂ : n ≤ 300) :
  ∃ k : ℕ, (1 ≤ 8 * k + 3 ∧ 8 * k + 3 ≤ 300) ∧ n = 38 :=
by
  sorry

end count_congruent_3_mod_8_l800_800740


namespace part1_f_ge_0_part2_number_of_zeros_part2_number_of_zeros_case2_l800_800725

-- Part 1: Prove f(x) ≥ 0 when a = 1
noncomputable def f (x : ℝ) : ℝ := Real.exp x - x - 1

theorem part1_f_ge_0 : ∀ x : ℝ, f x ≥ 0 := sorry

-- Part 2: Discuss the number of zeros of the function f(x)
noncomputable def g (a x : ℝ) : ℝ := Real.exp x - a * x - 1

theorem part2_number_of_zeros (a : ℝ) : 
  (a ≤ 0 ∨ a = 1) → ∃! x : ℝ, g a x = 0 := sorry

theorem part2_number_of_zeros_case2 (a : ℝ) : 
  (0 < a ∧ a < 1) ∨ (a > 1) → ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ g a x1 = 0 ∧ g a x2 = 0 := sorry

end part1_f_ge_0_part2_number_of_zeros_part2_number_of_zeros_case2_l800_800725


namespace problem_1_problem_2_problem_3_l800_800724

variables {a b x m : ℝ}

-- Problem 1
noncomputable def f (x : ℝ) := a * x^3 + (2 - 3 * a) / 2 * x^2 + b * x
noncomputable def f' (x : ℝ) := 3 * a * x^2 + (2 - 3 * a) * x + b

-- Tangent condition at x = 2
axiom f_prime_at_two : f' 2 = 1
axiom f_at_two : f 2 = 8

-- Correct expression for f(x)
def correct_expression := -x^3 + 5 / 2 * x^2 + 3 * x

-- Problem 2
noncomputable def g (x : ℝ) := - x^3 + x^2 + x

-- Intersection condition
def intersections (m : ℝ) :=
  if -5 / 27 < m ∧ m < 1 then 3
  else if m = -5 / 27 ∨ m = 1 then 2
  else 1

-- Problem 3
axiom a_is_one : a = 1
axiom ln_x_leq_f'_x : ∀ x : ℝ, 0 < x → Real.log x ≤ 3 * x^2 - x + b

-- Condition for b
def valid_b (b : ℝ) := b ≥ -Real.log 2 - 1 / 4

-- Statements to prove
theorem problem_1 : f = correct_expression := sorry
theorem problem_2 : ∀ m : ℝ, intersections m := sorry
theorem problem_3 : valid_b b := sorry

end problem_1_problem_2_problem_3_l800_800724


namespace sum_nth_beginning_end_l800_800522

theorem sum_nth_beginning_end (n : ℕ) (F L : ℤ) (M : ℤ) 
  (consecutive : ℤ → ℤ) (median : M = 60) 
  (median_formula : M = (F + L) / 2) :
  n = n → F + L = 120 :=
by
  sorry

end sum_nth_beginning_end_l800_800522


namespace decreasing_interval_of_even_function_l800_800767

noncomputable def f (k : ℝ) : ℝ → ℝ := λ x, (k-2)*x^2 + (k-1)*x + 3

theorem decreasing_interval_of_even_function (k : ℝ) (h_even : ∀ x : ℝ, f k (-x) = f k x) :
  (k = 1) → (∀ x : ℝ, x > 0 → f k (real.max x (-x)) < f k (real.min x (-x))) :=
by
  intro h1 hx hx_pos
  sorry

end decreasing_interval_of_even_function_l800_800767


namespace art_department_probability_l800_800576

theorem art_department_probability : 
  let students := {s1, s2, s3, s4} 
  let first_grade := {s1, s2}
  let second_grade := {s3, s4}
  let total_pairs := { (x, y) | x ∈ students ∧ y ∈ students ∧ x < y }.to_finset.card
  let diff_grade_pairs := { (x, y) | x ∈ first_grade ∧ y ∈ second_grade ∨ x ∈ second_grade ∧ y ∈ first_grade}.to_finset.card
  (diff_grade_pairs / total_pairs) = 2 / 3 := 
by 
  sorry

end art_department_probability_l800_800576


namespace hyperbola_focus_distance_l800_800275

theorem hyperbola_focus_distance :
  ∀ (x y : ℝ), (x^2 / 4 - y^2 / 3 = 1) → ∀ (F₁ F₂ : ℝ × ℝ), ∃ P : ℝ × ℝ, dist P F₁ = 3 → dist P F₂ = 7 :=
by
  sorry

end hyperbola_focus_distance_l800_800275


namespace points_concyclic_l800_800241

-- Definitions: Acute triangle ABC with AB < AC
variables {A B C : Type*} [ordered_ring A]
variables {o : A} (A B C : point o)

-- Midpoints M and N of AB and AC respectively
variables (M : midpoint A B) (N : midpoint A C)

-- Foot of the altitude from A to BC
variables (D : foot_altitude A B C)

-- Point K on segment MN such that BK = CK
variables (K : point o) (MN : segment M N)
variables (hyp_K : (distance B K) = (distance C K))

-- Ray KD intersects the circumcircle Ω of triangle ABC at point Q
variables (Q : point o) (Ω : circle ABC)
variables (ray_KD : ray K D)
axiom intersects (Ω : circle ABC) (ray_KD : ray K D) : (ray_KD ∩ Ω = {Q})

-- To prove: points C, N, K, and Q are concyclic
theorem points_concyclic : cyclic_quad C N K Q :=
by
  sorry

end points_concyclic_l800_800241


namespace max_19_points_in_circle_l800_800298

noncomputable def max_points_in_circle (r : ℝ) (c : ℕ) (d : ℝ) : Prop :=
  ∀ (P : finset (ℝ × ℝ)),
  (∀ p ∈ P, (p.1^2 + p.2^2) ≤ r^2) →
  (0, 0) ∈ P →
  (∀ p1 p2 ∈ P, p1 ≠ p2 → (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 ≥ d^2) →
  P.card ≤ c

theorem max_19_points_in_circle :
  max_points_in_circle 2 19 1 := 
sorry

end max_19_points_in_circle_l800_800298


namespace min_bulbs_lit_proof_l800_800222

-- Definitions to capture initial conditions and the problem
def bulb_state (n : ℕ) : Type := fin n → fin n → bool

def initial_state (n : ℕ) : bulb_state n := λ i j, false

-- Function to describe the state change after pressing a bulb
def press_bulb (n : ℕ) (state : bulb_state n) (i j : fin n) : bulb_state n :=
  λ x y, if x = i ∨ y = j then ¬ state x y else state x y

-- Function to calculate the minimum lit bulbs
def min_lit_bulbs (n : ℕ) (initial : bulb_state n) : ℕ :=
  2 * n - 2

-- The theorem statement
theorem min_bulbs_lit_proof (n : ℕ) (initial : bulb_state n) : ∃ seq : list (fin n × fin n), 
  (press_bulb n initial_state seq.head.1 seq.head.2).count true = 2 * n - 2 :=
sorry

end min_bulbs_lit_proof_l800_800222


namespace more_than_10_numbers_sum_20_no_3_l800_800836

theorem more_than_10_numbers_sum_20_no_3: 
    ∃ (n : ℕ) (a : ℕ → ℕ), n > 10 
        ∧ (∑ i in Finset.range n, a i) = 20 
        ∧ (∀ i, a i ≠ 3) 
        ∧ (∀ i j, i ≤ j → j < n → (∑ k in Finset.Icc i j, a k) ≠ 3) :=
sorry

end more_than_10_numbers_sum_20_no_3_l800_800836


namespace n_squared_plus_3n_is_perfect_square_iff_l800_800986

theorem n_squared_plus_3n_is_perfect_square_iff (n : ℕ) : 
  ∃ k : ℕ, n^2 + 3 * n = k^2 ↔ n = 1 :=
by 
  sorry

end n_squared_plus_3n_is_perfect_square_iff_l800_800986


namespace distinct_symbols_count_l800_800334

theorem distinct_symbols_count :
  let components (n : ℕ) := 2^n in
  (components 1) + (components 2) + (components 3) + (components 4) + (components 5) = 62 :=
by
  sorry

end distinct_symbols_count_l800_800334


namespace augmented_wedge_volume_proof_l800_800132

open Real

noncomputable def sphere_radius (circumference : ℝ) : ℝ :=
  circumference / (2 * π)

noncomputable def sphere_volume (r : ℝ) : ℝ :=
  (4/3) * π * r^3

noncomputable def wedge_volume (volume_sphere : ℝ) (number_of_wedges : ℕ) : ℝ :=
  volume_sphere / number_of_wedges

noncomputable def augmented_wedge_volume (original_wedge_volume : ℝ) : ℝ :=
  2 * original_wedge_volume

theorem augmented_wedge_volume_proof (circumference : ℝ) (number_of_wedges : ℕ) 
  (volume : ℝ) (augmented_volume : ℝ) :
  circumference = 18 * π →
  number_of_wedges = 6 →
  volume = sphere_volume (sphere_radius circumference) →
  augmented_volume = augmented_wedge_volume (wedge_volume volume number_of_wedges) →
  augmented_volume = 324 * π :=
by
  intros h_circ h_wedges h_vol h_aug_vol
  -- This is where the proof steps would go
  sorry

end augmented_wedge_volume_proof_l800_800132


namespace found_bottle_caps_l800_800975

theorem found_bottle_caps (threw_away: ℕ) (current_total: ℕ) (extra_bottle_caps: ℕ) : threw_away = 35 → current_total = 22 → extra_bottle_caps = 1 → 
  ∃ found : ℕ, found = threw_away + extra_bottle_caps ∧ current_total = (throw_away - 35) + found :=
by
  intros h1 h2 h3
  use threw_away + extra_bottle_caps
  split
  sorry
  sorry

end found_bottle_caps_l800_800975


namespace probability_two_vertices_share_edge_l800_800045

theorem probability_two_vertices_share_edge (cube_vertices : Finset (Fin 8)) (connected_edges : ∀ (v : Fin 8), Finset (Fin 8))
  (h_cube_vertices : cube_vertices.card = 8)
  (h_connected_edges : ∀ v, (connected_edges v).card = 3) :
  let total_ways := (cube_vertices.card.choose 2)
  let favorable_outcomes := (cube_vertices.sum (λ v, (connected_edges v).card)) / 2
  let probability := (favorable_outcomes : ℚ) / total_ways
  probability = 3 / 7 :=
by
  sorry

end probability_two_vertices_share_edge_l800_800045


namespace sum_difference_l800_800834

def is_even (n : ℕ) : Prop := n % 2 = 0

def set_A : Finset ℕ := 
  (Finset.range 81).filter (λ x, 32 ≤ x ∧ x ≤ 80 ∧ is_even x)

def set_B : Finset ℕ := 
  (Finset.range 111).filter (λ x, 62 ≤ x ∧ x ≤ 110 ∧ is_even x)

def sum_set (s : Finset ℕ) : ℕ :=
  s.sum id

theorem sum_difference : sum_set set_B - sum_set set_A = 750 := by
  sorry

end sum_difference_l800_800834


namespace problem_statement_l800_800819

open Real

theorem problem_statement (x : Fin 50 → ℝ)
  (h1 : (∑ i, x i) = 1)
  (h2 : (∑ i, x i / (1 - x i)) = 1)
  (h3 : (∑ i, (x i)^2) = 1 / 2) :
  (∑ i, (x i)^2 / (1 - x i)) = 0 :=
sorry

end problem_statement_l800_800819


namespace binom_sum_mod_2027_l800_800868

theorem binom_sum_mod_2027 :
  let T := ∑ k in Finset.range 65, Nat.choose 2024 k
  T % 2027 = 1089 :=
by
  let T := ∑ k in Finset.range 65, Nat.choose 2024 k
  have h2027_prime : Nat.prime 2027 := by exact dec_trivial
  sorry -- This is the placeholder for the actual proof

end binom_sum_mod_2027_l800_800868


namespace exists_y_z_l800_800555

theorem exists_y_z (n : ℕ) (x : Fin n → ℝ) (hx : ∑ i, (x i) ^ 2 = 1) :
  ∃ y z : Fin n → ℝ, (∑ i, |y i| ≤ 1) ∧ (∀ i, |z i| ≤ 1) ∧ (∀ i, 2 * x i = y i + z i) :=
by
  sorry

end exists_y_z_l800_800555


namespace determine_common_ratio_l800_800388

noncomputable def geometric_sequence (q : ℝ) (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * q

def sum_of_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, S n = ∑ i in finset.range n, (a i)

theorem determine_common_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) 
  (h1 : geometric_sequence q a)
  (h2 : a 3 = 3 / 2)
  (h3 : S 3 = 9 / 2) :
  q = 1 ∨ q = -1 / 2 :=
sorry

end determine_common_ratio_l800_800388


namespace part_a_connected_3_rings_part_b_connected_10_rings_l800_800533

-- Define a predicate to describe the connection property of rings
def connected_rings (n : ℕ) : Prop :=
  ∃ R : set (set ℕ), (R.card = n) ∧
    (∀ (r1 r2 : set ℕ), r1 ∈ R → r2 ∈ R → r1 ≠ r2 → is_disjoint r1 r2) ∧
    (∀ r ∈ R, ∃ s ∈ R, ¬ is_disjoint r s) ∧
    (∀ (r : set ℕ), r ∈ R → (∃ s ∈ R, s ≠ r ∧ is_disjoint (r \ s) s))

-- Problem statement for part (a)
theorem part_a_connected_3_rings : connected_rings 3 :=
by
  sorry

-- Problem statement for part (b)
theorem part_b_connected_10_rings : connected_rings 10 :=
by
  sorry

end part_a_connected_3_rings_part_b_connected_10_rings_l800_800533


namespace fraction_evaluation_l800_800192

theorem fraction_evaluation (a b : ℝ) (h : a ≠ b) :
  (a ^ (-6) - b ^ (-6)) / (a ^ (-3) - b ^ (-3)) = a ^ (-1) + b ^ (-1) :=
by sorry

end fraction_evaluation_l800_800192


namespace sqrt_expression_identity_l800_800646

theorem sqrt_expression_identity :
  (Real.sqrt 3 + Real.sqrt 2) * (Real.sqrt 3 - Real.sqrt 2)^2 = Real.sqrt 3 - Real.sqrt 2 := 
by
  sorry

end sqrt_expression_identity_l800_800646


namespace part1_max_price_part2_min_sales_volume_l800_800442

-- Definitions for part 1
def original_price := 25
def original_sales_volume := 80000
def revenue (price : ℕ) (sales_volume : ℕ) : ℕ := price * sales_volume
def price_increase_effect (price_increase : ℕ) : ℕ := original_sales_volume - 2000 * price_increase
def total_revenue (price : ℕ) : ℕ := revenue price (price_increase_effect (price - original_price))

-- Definitions for part 2
def original_revenue : ℕ := revenue original_price original_sales_volume
def investment_cost (x : ℕ) : ℕ := (1/6 : ℚ) * (x^2 - 600)
def publicity_cost (x : ℕ) : ℕ := 50 + 2 * x
def total_investment_cost (x : ℕ) : ℕ := investment_cost x + publicity_cost x
def required_sales_volume (price : ℕ) (revenue_target : ℕ) : ℕ := revenue_target / price

-- Lean 4 statements
theorem part1_max_price : ∃ t, 25 ≤ t ∧ t ≤ 40 ∧ total_revenue t = 40 :=
  sorry

theorem part2_min_sales_volume (x : ℕ) : x > 25 → 
  ∃ (a : ℕ), required_sales_volume x (original_revenue + total_investment_cost x) ≥ 12 ∧ x = 30 :=
  sorry

end part1_max_price_part2_min_sales_volume_l800_800442


namespace minimum_teachers_to_cover_all_subjects_l800_800612

/- Define the problem conditions -/
def maths_teachers := 7
def physics_teachers := 6
def chemistry_teachers := 5
def max_subjects_per_teacher := 3

/- The proof statement -/
theorem minimum_teachers_to_cover_all_subjects : 
  (maths_teachers + physics_teachers + chemistry_teachers) / max_subjects_per_teacher = 7 :=
sorry

end minimum_teachers_to_cover_all_subjects_l800_800612


namespace mean_of_combined_set_l800_800006

theorem mean_of_combined_set
  (mean1 : ℕ → ℝ)
  (n1 : ℕ)
  (mean2 : ℕ → ℝ)
  (n2 : ℕ)
  (h1 : ∀ n1, mean1 n1 = 15)
  (h2 : ∀ n2, mean2 n2 = 26) :
  (n1 + n2) = 15 → 
  ((n1 * 15 + n2 * 26) / (n1 + n2)) = (313/15) :=
by
  sorry

end mean_of_combined_set_l800_800006


namespace scores_greater_than_18_l800_800346

theorem scores_greater_than_18 (scores : Fin 20 → ℝ) 
  (h_unique : Function.Injective scores)
  (h_sum : ∀ i j k : Fin 20, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k) :
  ∀ i : Fin 20, scores i > 18 := 
by
  sorry

end scores_greater_than_18_l800_800346


namespace evaluate_expression_l800_800652

theorem evaluate_expression (x y z : ℚ) (hx : x = 1 / 2) (hy : y = 1 / 3) (hz : z = 2) : 
  (x^3 * y^4 * z)^2 = 1 / 104976 :=
by 
  sorry

end evaluate_expression_l800_800652


namespace tennis_preference_percentage_l800_800008

-- Definitions based on the conditions
def total_students_north : ℕ := 1800
def percentage_tennis_north : ℝ := 0.20

def total_students_south : ℕ := 2200
def percentage_tennis_south : ℝ := 0.30

-- Calculate the number of students who prefer tennis at each school
def students_tennis_north : ℝ := total_students_north * percentage_tennis_north
def students_tennis_south : ℝ := total_students_south * percentage_tennis_south

-- Calculate the total number of students who prefer tennis in both schools
def total_students_tennis : ℝ := students_tennis_north + students_tennis_south

-- Calculate the combined total number of students in both schools
def total_students_combined : ℕ := total_students_north + total_students_south

-- Calculate the percentage of students who prefer tennis in both schools combined
def percentage_tennis_combined : ℝ := (total_students_tennis / total_students_combined) * 100

-- Proof statement
theorem tennis_preference_percentage : 
  percentage_tennis_combined ≈ 26.0 := 
by sorry

end tennis_preference_percentage_l800_800008


namespace olympiad_scores_above_18_l800_800341

theorem olympiad_scores_above_18 
  (n : Nat) 
  (scores : Fin n → ℕ) 
  (h_diff_scores : ∀ i j : Fin n, i ≠ j → scores i ≠ scores j) 
  (h_score_sum : ∀ i j k : Fin n, i ≠ j ∧ i ≠ k ∧ j ≠ k → scores i < scores j + scores k) 
  (h_n : n = 20) : 
  ∀ i : Fin n, scores i > 18 := 
by 
  -- See the proof for the detailed steps.
  sorry

end olympiad_scores_above_18_l800_800341


namespace square_field_area_l800_800096

-- Definitions from the problem conditions
def Joy_speed := 8 -- in km/hr
def Time_taken := 0.5 -- in hours
def Distance := Joy_speed * Time_taken -- Distance = Speed × Time
def Field_diagonal := Distance -- Diagonal of the square field

-- Pythagorean theorem application to find side length
def field_side_length := real.to_nnreal (sqrt (Field_diagonal ^ 2 / 2))
def field_area := field_side_length ^ 2

-- Proof statement
theorem square_field_area : field_area = 8 :=
by
  -- Calculations based on conditions
  sorry

end square_field_area_l800_800096


namespace true_proposition_l800_800284

open Real

-- Define the propositions p and q
def p : Prop := ∀ x : ℝ, 2^x < 3^x
def q : Prop := ∃ x₀ : ℝ, x₀^2 - 2*x₀ + 1 > 0

-- Define the statement to be proven
theorem true_proposition : (¬p) ∧ q :=
by
  -- Since the statement says we do not need to consider the solution steps and only provide the statement, we use sorry here.
  sorry

end true_proposition_l800_800284


namespace imaginary_part_of_z_l800_800320

theorem imaginary_part_of_z : 
  ∀ (z : ℂ), (1 + complex.i) * z = (1 - 2 * complex.i) → complex.im z = -3 / 2 :=
by 
  intros z h
  sorry

end imaginary_part_of_z_l800_800320
