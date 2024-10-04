import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.ArithmeticSequence
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Group.InjSurj
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Quadratic.Basic
import Mathlib.Analysis.Calculus.Fderiv
import Mathlib.Analysis.Limits
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Integrals
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Binomial
import Mathlib.Combinatorics.Burnside
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Init.Algebra
import Mathlib.LinearAlgebra.Matrix
import Mathlib.MeasureTheory.MeasureSpace
import Mathlib.Meta.Basic
import Mathlib.Probability.Distribution
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Sorry
import Mathlib.Topology.MetricSpace.Basic
import Mathlibimportant closed_roadsimportant (set1
import mathlib.data.real.nnreal

namespace projection_same_on_plane_l207_207600

noncomputable def projection (v n : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let d := (v.1 * n.1 + v.2 * n.2 + v.3 * n.3) / (n.1 * n.1 + n.2 * n.2 + n.3 * n.3)
  (v.1 - d * n.1, v.2 - d * n.2, v.3 - d * n.3)

theorem projection_same_on_plane :
  let a := (2, -2, 4)
  let b := (0, 4, 0)
  let n := (1, -1, 2)
  let pa := projection a n
  let pb := projection b n
  pa = (1/3, -1/3, 2/3) ∧ pa = pb :=
by
  sorry

end projection_same_on_plane_l207_207600


namespace circle_line_no_intersection_l207_207686

theorem circle_line_no_intersection :
  ∀ (r : ℝ), (r > 0) →
  circle_eq : (∀ x y : ℝ, (x + 5)^2 + y^2 = r^2) →
  line_eq : (∀ x y : ℝ, 3 * x + y + 5 = 0) →
  (¬ ∃ x y : ℝ, (x + 5)^2 + y^2 = r^2 ∧ 3 * x + y + 5 = 0) →
  0 < r ∧ r < real.sqrt 10 :=
by
  intros r hr circle_eq line_eq h_nc
  sorry

end circle_line_no_intersection_l207_207686


namespace ratio_nonupgraded_to_upgraded_l207_207556

-- Define the initial conditions and properties
variable (S : ℝ) (N : ℝ)
variable (h1 : ∀ N, N = S / 32)
variable (h2 : ∀ S, 0.25 * S = 0.25 * S)
variable (h3 : S > 0)

-- Define the theorem to show the required ratio
theorem ratio_nonupgraded_to_upgraded (h3 : 24 * N = 0.75 * S) : (N / (0.25 * S) = 1 / 8) :=
by
  sorry

end ratio_nonupgraded_to_upgraded_l207_207556


namespace shaded_area_of_squares_is_20_l207_207180

theorem shaded_area_of_squares_is_20 :
  ∀ (a b : ℝ), a = 2 → b = 6 → 
    (1/2) * a * a + (1/2) * b * b = 20 :=
by
  intros a b ha hb
  rw [ha, hb]
  sorry

end shaded_area_of_squares_is_20_l207_207180


namespace Petya_friends_l207_207852

-- Definitions from the conditions
def num_students : ℕ := 25
def possible_friend_counts : set ℕ := {n | n ≤ 24 ∧ n ≥ 0}
def distinct_friend_counts (counts : finset ℕ) : Prop :=
  counts.card = num_students ∧ ∀ (x ∈ counts), x ∈ possible_friend_counts ∧ ∀ (y ∈ counts), x ≠ y → (counts : set ℕ).erase x = {n | n ≠ y}

-- Given the distinct friend counts property holds for the class,
-- prove that Petya's number of friends is 12 or 13
theorem Petya_friends :
  ∃ (counts : finset ℕ), distinct_friend_counts counts ∧ (12 ∈ counts ∨ 13 ∈ counts) :=
sorry

end Petya_friends_l207_207852


namespace area_of_rectangular_field_l207_207417

theorem area_of_rectangular_field (W D : ℝ) (hW : W = 15) (hD : D = 17) :
  ∃ L : ℝ, (W * L = 120) ∧ D^2 = L^2 + W^2 :=
by 
  use 8
  sorry

end area_of_rectangular_field_l207_207417


namespace area_of_circle_l207_207916

theorem area_of_circle (x y : ℝ) :
  x^2 + y^2 - 4*x - 6*y = -3 →
  ∃ A : ℝ, A = 10 * Real.pi :=
by
  intro h
  sorry

end area_of_circle_l207_207916


namespace range_of_function_l207_207895

noncomputable def f (x : ℝ) : ℝ := x + Real.sqrt (1 - x)

theorem range_of_function : Set.Iic (5 / 4) = SetOf (y : ℝ) (∃ x : ℝ, f x = y) :=
by 
  sorry

end range_of_function_l207_207895


namespace simplify_and_evaluate_l207_207437

theorem simplify_and_evaluate :
  let x := -2 in
  (2 * x + 1) * (x - 2) - (2 - x) ^ 2 = -4 := by
  sorry

end simplify_and_evaluate_l207_207437


namespace parallelepiped_volume_l207_207875

theorem parallelepiped_volume (AB AD : ℝ) (angle_ABC : ℝ) (diagonal_condition : ℝ) :
  AB = 3 ∧ AD = 4 ∧ angle_ABC = 120 ∧ diagonal_condition = real.sqrt (3^2 + 4^2 - 2 * 3 * 4 * real.cos (120 * real.pi / 180))
  → ∃ V : ℝ, V = 36 * real.sqrt 2 := by
  sorry

end parallelepiped_volume_l207_207875


namespace find_speed_in_second_hour_l207_207897

-- Define the given conditions as hypotheses
def speed_in_first_hour : ℝ := 50
def average_speed : ℝ := 55
def total_time : ℝ := 2

-- Define a function that represents the speed in the second hour
def speed_second_hour (s2 : ℝ) := 
  (speed_in_first_hour + s2) / total_time = average_speed

-- The statement to prove: the speed in the second hour is 60 km/h
theorem find_speed_in_second_hour : speed_second_hour 60 :=
by sorry

end find_speed_in_second_hour_l207_207897


namespace ashley_champagne_bottles_l207_207581

theorem ashley_champagne_bottles (guests : ℕ) (glasses_per_guest : ℕ) (servings_per_bottle : ℕ) 
  (h1 : guests = 120) (h2 : glasses_per_guest = 2) (h3 : servings_per_bottle = 6) : 
  (guests * glasses_per_guest) / servings_per_bottle = 40 :=
by
  -- The proof will go here
  sorry

end ashley_champagne_bottles_l207_207581


namespace Moscow1964_27th_MMO_l207_207707

theorem Moscow1964_27th_MMO {a : ℤ} (h : ∀ k : ℤ, k ≠ 27 → ∃ m : ℤ, a - k^1964 = m * (27 - k)) : 
  a = 27^1964 :=
sorry

end Moscow1964_27th_MMO_l207_207707


namespace valid_sequences_count_l207_207373

noncomputable def number_of_valid_sequences
(strings : List (List Nat))
(ball_A_shot : Nat)
(ball_B_shot : Nat) : Nat := 144

theorem valid_sequences_count :
  let strings := [[1, 2], [3, 4, 5], [6, 7, 8, 9]];
  let ball_A := 1;  -- Assuming A is the first ball in the first string
  let ball_B := 3;  -- Assuming B is the first ball in the second string
  ball_A = 1 →
  ball_B = 3 →
  ball_A_shot = 5 →
  ball_B_shot = 6 →
  number_of_valid_sequences strings ball_A_shot ball_B_shot = 144 :=
by
  intros strings ball_A ball_B hA hB hAShot hBShot
  sorry

end valid_sequences_count_l207_207373


namespace compressor_stations_possible_distances_when_a_30_l207_207910

variables (a x y z : ℝ)
noncomputable def isValid (a : ℝ) : Prop :=
  0 < a ∧ a < 60

noncomputable def distances_valid (x y z a : ℝ) : Prop :=
  x = (240 - a) / 6 ∧
  y = (20 + (2 * a) / 3) ∧
  z = (20 + a / 6)

theorem compressor_stations_possible (a : ℝ) : isValid a ∧
  distances_valid ((240 - a) / 6) (20 + (2 * a) / 3) (20 + a / 6) a :=
sorry

theorem distances_when_a_30 : distances_valid 35 40 25 30 :=
begin
  norm_num, -- Apply numerical simplification
  split,
  { norm_num, },
  split,
  { norm_num, },
  { norm_num, }
end

end compressor_stations_possible_distances_when_a_30_l207_207910


namespace tangent_slope_at_A_l207_207101

theorem tangent_slope_at_A : 
  (∀ x : ℝ, y x = exp x + 1) → tangent_slope (y 0) = 1 :=
begin
  sorry
end

end tangent_slope_at_A_l207_207101


namespace trapezoid_length_KLMN_l207_207790

variables {K L M N P Q : Type}
variables (trapezoid KLMN : K L M N)
variable (KM : ℝ) (KP MQ LM MP : ℝ)
variables (perp1 : KP > 0) (perp2 : MQ > 0)
variables (equal1 : KM = 1) (equal2 : KP = MQ) (equal3 : LM = MP)

theorem trapezoid_length_KLMN
(equality_KM: KM = 1)
(equality_KP_MQ: KP = MQ)
(equality_LM_MP: LM = MP)
: LM = sqrt 2 := 
by sorry

end trapezoid_length_KLMN_l207_207790


namespace ratio_of_even_to_odd_divisors_l207_207837

def N : ℕ := 45 * 45 * 70 * 375

theorem ratio_of_even_to_odd_divisors :
  let even_div_sum := ∑ n in (finset.filter (λ d, d % 2 = 0) (finset.divisors N)), n,
      odd_div_sum := ∑ n in (finset.filter (λ d, ¬ d % 2 = 0) (finset.divisors N)), n
  in even_div_sum / odd_div_sum = 1 :=
by {
  -- proof goes here. Sorry added to skip the proof part.
  sorry
}

end ratio_of_even_to_odd_divisors_l207_207837


namespace divisibility_by_3_l207_207724

theorem divisibility_by_3 (a b c : ℤ) (h1 : c ≠ b)
    (h2 : ∃ x : ℂ, (a * x^2 + b * x + c = 0 ∧ (c - b) * x^2 + (c - a) * x + (a + b) = 0)) :
    3 ∣ (a + b + 2 * c) :=
by
  sorry

end divisibility_by_3_l207_207724


namespace simplify_and_evaluate_l207_207438

theorem simplify_and_evaluate :
  let x := -2 in
  (2 * x + 1) * (x - 2) - (2 - x) ^ 2 = -4 := by
  sorry

end simplify_and_evaluate_l207_207438


namespace proof_mt_eq_neg2_l207_207036

noncomputable def g : ℝ → ℝ := sorry

axiom g_condition1 : g 1 = 2
axiom g_condition2 : ∀ x y : ℝ, g (2 * x * y + g x) = x * g y + g x

theorem proof_mt_eq_neg2 :
  let vals := {g (-1 / 2) | true}.toFinset
  let m := vals.card
  let t := vals.sum id
  m * t = -2 :=
by
  let vals := {g (-1 / 2) | true}.toFinset
  let m := vals.card
  let t := vals.sum id
  have m_eq : m = 1 := sorry
  have t_eq : t = -2 := sorry
  rw [m_eq, t_eq]
  exact mul_one (-2)

end proof_mt_eq_neg2_l207_207036


namespace arrange_magnitudes_l207_207314

theorem arrange_magnitudes (x : ℝ) (hx : 0.8 < x ∧ x < 0.9) :
  let y := x^x
  let z := x^(x^x)
  x < z ∧ z < y := by
  sorry

end arrange_magnitudes_l207_207314


namespace factorize_expression_l207_207619

variable {R : Type} [CommRing R] (a x y : R)

theorem factorize_expression :
  a^2 * (x - y) + 9 * (y - x) = (x - y) * (a + 3) * (a - 3) :=
by
  sorry

end factorize_expression_l207_207619


namespace prove_u_div_p_l207_207747

theorem prove_u_div_p (p r s u : ℚ) 
  (h1 : p / r = 8)
  (h2 : s / r = 5)
  (h3 : s / u = 1 / 3) : 
  u / p = 15 / 8 := 
by 
  sorry

end prove_u_div_p_l207_207747


namespace find_x_plus_inv_x_l207_207704

theorem find_x_plus_inv_x (x : ℝ) (h : x^3 + 1/x^3 = 110) : x + 1/x = 5 := by
  sorry

end find_x_plus_inv_x_l207_207704


namespace line_intersects_circle_l207_207304

noncomputable def cartesian_circle_equation : Prop := 
  ∀ (x y : ℝ), ρ = 2 * cos θ → (x - 1) ^ 2 + y ^ 2 = 1

theorem line_intersects_circle (t : ℝ) (x y : ℝ) :
  (x = 1/2 + (real.sqrt 3)/2 * t) ∧ 
  (y = 1/2 + 1/2 * t) ∧ 
  ((x - 1) ^ 2 + y ^ 2 = 1) →
  ∃ (t1 t2 : ℝ), (t1 * t2 = -1/2) ∧ (abs (t1) ≠ abs (t)) ∧ (abs (t2) ≠ abs (t)) ∧ (|AP| * |AQ| = 1/2)
:= 
by
  sorry 

end line_intersects_circle_l207_207304


namespace general_term_sequence_a_l207_207727

noncomputable def sequence_a : ℕ → ℤ
| 0          := 0
| 1          := -1
| (n + 1 + 1) := 2 * (sequence_a (n + 1)) + (3 * (n + 1) - 1) * 3^(n + 2)

theorem general_term_sequence_a :
  ∀ (n : ℕ), sequence_a (n + 1) = 31 * 2^(n + 1) + (3 * (n + 1) - 10) * 3^(n + 2) :=
sorry

end general_term_sequence_a_l207_207727


namespace optimal_strategy_for_father_l207_207544

-- Define the individual players
inductive player
| Father 
| Mother 
| Son

open player

-- Define the probabilities of player defeating another
def prob_defeat (p1 p2 : player) : ℝ := sorry  -- These will be defined as per the problem's conditions.

-- Define the probability of father winning given the first matchups
def P_father_vs_mother : ℝ :=
  prob_defeat Father Mother * prob_defeat Father Son +
  prob_defeat Father Mother * prob_defeat Son Father * prob_defeat Mother Son * prob_defeat Father Mother +
  prob_defeat Mother Father * prob_defeat Son Mother * prob_defeat Father Son * prob_defeat Father Mother

def P_father_vs_son : ℝ :=
  prob_defeat Father Son * prob_defeat Father Mother +
  prob_defeat Father Son * prob_defeat Mother Father * prob_defeat Son Mother * prob_defeat Father Son +
  prob_defeat Son Father * prob_defeat Mother Son * prob_defeat Father Mother * prob_defeat Father Son

-- Define the optimality condition
theorem optimal_strategy_for_father :
  P_father_vs_mother > P_father_vs_son :=
sorry

end optimal_strategy_for_father_l207_207544


namespace find_x_l207_207311

theorem find_x (x : ℝ) (h1 : |x + 7| = 3) (h2 : x^2 + 2*x - 3 = 5) : x = -4 :=
by
  sorry

end find_x_l207_207311


namespace diagram_b_achievable_diagram_c_not_achievable_l207_207706

-- Definitions based on conditions provided.
structure Hexagon :=
(colored : Bool)

def flip (h : Hexagon) : Hexagon := { colored := !h.colored }

-- Operations considering adjacent hexagons
def flip_adjacent (h : Hexagon) (adjacents : List Hexagon) : List Hexagon :=
flip h :: adjacents.map flip

-- Initial and target configurations labeled as diagrams (b) and (c)
def initial_configuration : List Hexagon := 
-- Define some initial configuration (example)
[ Hexagon.mk true, Hexagon.mk false, Hexagon.mk true, Hexagon.mk false, Hexagon.mk true, Hexagon.mk false, Hexagon.mk true ]

def target_configuration_b : List Hexagon := 
-- Define target configuration for diagram (b)
[ Hexagon.mk false, Hexagon.mk true, Hexagon.mk false, Hexagon.mk true, Hexagon.mk false, Hexagon.mk true, Hexagon.mk false ]

def target_configuration_c : List Hexagon :=
-- Define target configuration for diagram (c)
[ Hexagon.mk false, Hexagon.mk true, Hexagon.mk true, Hexagon.mk false, Hexagon.mk false, Hexagon.mk true, Hexagon.mk true ]

-- Proving configurations
theorem diagram_b_achievable : 
    ∃ flips : List (Hexagon → List Hexagon), 
    ∃ start : List Hexagon, 
    start = initial_configuration ∧
    (flips.foldl (λ acc f => acc.concat_map f) start) = target_configuration_b :=
sorry

theorem diagram_c_not_achievable : 
    ¬ ∃ flips : List (Hexagon → List Hexagon), 
    ∃ start : List Hexagon, 
    start = initial_configuration ∧
    (flips.foldl (λ acc f => acc.concat_map f) start) = target_configuration_c :=
sorry

end diagram_b_achievable_diagram_c_not_achievable_l207_207706


namespace number_of_dividing_functions_number_of_dividing_functions_six_l207_207596

-- Part (a)
theorem number_of_dividing_functions (n : ℕ) (k : ℕ) (p : Fin k → ℕ) (hp : ∀ i j, p i ≠ p j ):  
  (∃ m : ℕ, n = (Finset.univ : Finset (Fin k)).prod p m) →
  ∃ r : ℕ, 
  (∀ f : Fin n → ℕ, 
      (∀ i : Fin n, ∃ j : (Fin k), f i = p j) → (Finset.univ : Finset (Fin n)).prod f ∣ n) →
  r = 2^(k * n) := 
sorry

-- Part (b)
theorem number_of_dividing_functions_six : 
  ∃ r : ℕ, 
  (∀ f : Fin 6 → ℕ, 
      (∀ x : Fin 6, f x ∈ {1, 2, 3, 4, 6, 9, 12, 18, 36}) → (Finset.univ : Finset (Fin 6)).prod f ∣ 36) →
  r = 580 := 
sorry

end number_of_dividing_functions_number_of_dividing_functions_six_l207_207596


namespace rachel_hw_diff_l207_207065

-- Definitions based on the conditions of the problem
def math_hw_pages := 15
def reading_hw_pages := 6

-- The statement we need to prove, including the conditions
theorem rachel_hw_diff : 
  math_hw_pages - reading_hw_pages = 9 := 
by
  sorry

end rachel_hw_diff_l207_207065


namespace boat_speed_in_still_water_l207_207154

theorem boat_speed_in_still_water (speed_of_stream time distance : ℝ) 
(speed_of_stream_eq : speed_of_stream = 5)
(time_eq : time = 7)
(distance_eq : distance = 147)
: ∃ Vb : ℝ, Vb = 16 :=
by
  have h := distance_eq
  rw [distance_eq, time_eq, speed_of_stream_eq] at h
  use 16
  sorry

end boat_speed_in_still_water_l207_207154


namespace original_plan_was_to_produce_125_sets_per_day_l207_207504

-- We state our conditions
def plans_to_complete_in_days : ℕ := 30
def produces_sets_per_day : ℕ := 150
def finishes_days_ahead_of_schedule : ℕ := 5

-- Calculations based on conditions
def actual_days_used : ℕ := plans_to_complete_in_days - finishes_days_ahead_of_schedule
def total_production : ℕ := produces_sets_per_day * actual_days_used
def original_planned_production_per_day : ℕ := total_production / plans_to_complete_in_days

-- Claim we want to prove
theorem original_plan_was_to_produce_125_sets_per_day :
  original_planned_production_per_day = 125 :=
by
  sorry

end original_plan_was_to_produce_125_sets_per_day_l207_207504


namespace distinct_cube_constructions_l207_207601

theorem distinct_cube_constructions :
  let group_rotations := 7
  let identity_fixed := binomial 8 5
  let edge_rotations_fixed := 0
  (identity_fixed + 6 * edge_rotations_fixed) / group_rotations = 8 :=
by 
  let group_rotations := 7
  let identity_fixed := binomial 8 5
  let edge_rotations_fixed := 0
  have identity_contrib : identity_fixed = 56 := rfl
  have edge_contribs : 6 * edge_rotations_fixed = 0 := rfl
  show (identity_fixed + 6 * edge_rotations_fixed) / group_rotations = 8
  calc (identity_fixed + 6 * edge_rotations_fixed) / group_rotations
    = (56 + 0) / 7 : by rw [identity_contrib, edge_contribs]
    ... = 8 : rfl

end distinct_cube_constructions_l207_207601


namespace persons_in_first_group_l207_207529

-- Definition of the conditions
def first_group_days := 12
def first_group_hours_per_day := 5
def second_group_persons := 30
def second_group_days := 19
def second_group_hours_per_day := 6

-- Prove the question == answer given the conditions
theorem persons_in_first_group :
  let P := (second_group_persons * second_group_days * second_group_hours_per_day) / (first_group_days * first_group_hours_per_day) in
  P = 57 :=
by
  -- The proof is omitted
  sorry

end persons_in_first_group_l207_207529


namespace problem1_problem2_problem3_l207_207395

section Problem1

def f (x : ℝ) : ℝ := Real.log (x + 1)

theorem problem1 (x : ℝ) (hx : 0 ≤ x) : f(x) ≥ x - x^2 := 
by {
  sorry
}

end Problem1

section Problem2

def g (x : ℝ) (a : ℝ) : ℝ := x * (x + a + 1) / (x + 1)

theorem problem2 (x : ℝ) (a : ℝ) (hx : 0 ≤ x) (ha : a ≤ 1) : f(x) + x ≥ g(x, a) := 
by {
  sorry
}

end Problem2

section Problem3

open Real

theorem problem3 (n : ℕ) (hn : 1 ≤ n) : log (n^2 + 3*n + 2) > (finset.range (n - 1)).sum (λ k, 1 / ((k + 1) * (k + 2) ^ 2)) :=
by {
  sorry
}

end Problem3

end problem1_problem2_problem3_l207_207395


namespace slope_of_line_l207_207920

theorem slope_of_line : 
  ∀ (x1 y1 x2 y2 : ℝ), 
  x1 = 1 → y1 = 3 → x2 = 6 → y2 = -7 → 
  (x1 ≠ x2) → ((y2 - y1) / (x2 - x1) = -2) :=
by
  intros x1 y1 x2 y2 hx1 hy1 hx2 hy2 hx1_ne_x2
  rw [hx1, hy1, hx2, hy2]
  sorry

end slope_of_line_l207_207920


namespace angle_PDA_eq_angle_QBA_l207_207416

variables {A B C D A1 C1 P Q : Point}
variables {parallelogram_ABCD : Parallelogram A B C D}
variables {A1_on_AB : A1 ∈ LineSegment A B}
variables {C1_on_BC : C1 ∈ LineSegment B C}
variables {P_intersection : IntersectLine AC1 CA1 P}
variables {Q_inside_ACD : InTriangle Q A C D}
variables {circumcircle_AA1P : Circumcircle A A1 P}
variables {circumcircle_CC1P : Circumcircle C C1 P}
variables {Q_on_circumcircles : Q ∈ circumcircle_AA1P ∧ Q ∈ circumcircle_CC1P}

theorem angle_PDA_eq_angle_QBA :
  ∠PDA = ∠QBA :=
begin
  sorry
end

end angle_PDA_eq_angle_QBA_l207_207416


namespace selena_taco_packages_l207_207434

-- Define the problem conditions
def tacos_per_package : ℕ := 4
def shells_per_package : ℕ := 6
def min_tacos : ℕ := 60
def min_shells : ℕ := 60

-- Lean statement to prove the smallest number of taco packages needed
theorem selena_taco_packages :
  ∃ n : ℕ, (n * tacos_per_package ≥ min_tacos) ∧ (∃ m : ℕ, (m * shells_per_package ≥ min_shells) ∧ (n * tacos_per_package = m * shells_per_package) ∧ n = 15) := 
by {
  sorry
}

end selena_taco_packages_l207_207434


namespace proposal_totals_and_difference_l207_207486

theorem proposal_totals_and_difference :
  ∀ (x y z : ℝ),
    (58 + 0.40 * x = 0.60 * x) →
    (72 + 0.45 * y = 0.55 * y) →
    (30 + 0.55 * z = 0.45 * z) →
    (x = 290) →
    (y = 720) →
    (z = 300) →
    (∃ d, d = abs (max x (max y z) - min x (min y z)) ∧ d = 430) :=
by
  intros x y z Hx Hy Hz Hx_sol Hy_sol Hz_sol
  have Hx' : x = 290 := Hx_sol
  have Hy' : y = 720 := Hy_sol
  have Hz' : z = 300 := Hz_sol
  use abs (max x (max y z) - min x (min y z))
  have H1 : max x (max y z) = 720 := by { sorry }
  have H2 : min x (min y z) = 290 := by { sorry }
  rw [H1, H2]
  have H3 : abs (720 - 290) = 430 := by { sorry }
  exact ⟨H3, rfl⟩

end proposal_totals_and_difference_l207_207486


namespace math_proof_l207_207458

noncomputable def problem_proof : Prop :=
  ∃ (x y : ℕ), 
    (log 2 x + 3 * log 2 (Nat.gcd x y) = 30) ∧
    (log 2 y + 3 * log 2 (Nat.lcm x y) = 270) ∧
    let p := x.factorization.toList.length in
    let q := y.factorization.toList.length in
      4 * p + 3 * q = 240

theorem math_proof : problem_proof :=
  sorry

end math_proof_l207_207458


namespace count_sixth_powers_lt_200_l207_207324

theorem count_sixth_powers_lt_200 : 
  {n : ℕ | n > 0 ∧ n < 200 ∧ (∃ k : ℕ, n = k^6)}.to_finset.card = 2 := 
by sorry

end count_sixth_powers_lt_200_l207_207324


namespace geometric_sequence_sum_l207_207227

theorem geometric_sequence_sum (a_1 q : ℝ) (n : ℕ) (Sn : ℝ) :
  Sn = (∑ i in range n, a_1 * q ^ i) →
  (q ≠ 1 → Sn = a_1 * (1 - q ^ n) / (1 - q)) ∧ (q = 1 → Sn = n * a_1) 
  :=
by
  intro hSn
  split
  . intro hq
    sorry
  . intro hq_eq_1
    sorry

end geometric_sequence_sum_l207_207227


namespace spears_per_sapling_l207_207406

/-- Given that a log can produce 9 spears and 6 saplings plus a log produce 27 spears,
prove that a single sapling can produce 3 spears (S = 3). -/
theorem spears_per_sapling (L S : ℕ) (hL : L = 9) (h: 6 * S + L = 27) : S = 3 :=
by
  sorry

end spears_per_sapling_l207_207406


namespace parallel_line_plane_l207_207348

variables {l m : Line} {α β : Plane}

-- Definitions of lines and planes being non-coincident
def non_coincident_lines := ∀ (l m : Line), l ≠ m
def non_coincident_planes := ∀ (α β : Plane), α ≠ β

-- Given conditions
axiom alpha_inter_beta_eq_m : α ∩ β = m
axiom l_not_in_alpha : l ⊈ α
axiom l_parallel_m : l ∥ m 

-- To prove
theorem parallel_line_plane : l ∥ α :=
by 
  sorry

end parallel_line_plane_l207_207348


namespace sum_fractions_a_sum_fractions_b_seven_distinct_reciprocal_sum_l207_207943

-- Part (a)
theorem sum_fractions_a : (1 / 2) + (1 / 3) + (1 / 6) = 1 := 
by 
  sorry

-- Part (b)
theorem sum_fractions_b : (1 / 12) + (1 / 18) + (1 / 36) = 1 / 6 := 
by 
  sorry

-- Part (c)
theorem seven_distinct_reciprocal_sum : 
  ∀ (a b c d e f g : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ 
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ 
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ 
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ 
    e ≠ f ∧ e ≠ g ∧ 
    f ≠ g ∧ 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧ g > 0 →
  (1 / a : ℚ) + (1 / b) + (1 / c) + (1 / d) + (1 / e) + (1 / f) + (1 / g) = 1 → 
  ({a, b, c, d, e, f, g} = {2, 3, 12, 18, 72, 108, 216}) :=
by 
  sorry

end sum_fractions_a_sum_fractions_b_seven_distinct_reciprocal_sum_l207_207943


namespace translation_down_by_3_l207_207002

noncomputable def translate_down (f : ℝ → ℝ) (d : ℝ) : (ℝ → ℝ) :=
  λ x, f x - d

def initial_function : ℝ → ℝ :=
  λ x, 3 * x + 2

theorem translation_down_by_3 :
  translate_down initial_function 3 = (λ x, 3 * x - 1) :=
by
  sorry

end translation_down_by_3_l207_207002


namespace cost_price_of_computer_table_l207_207096

theorem cost_price_of_computer_table (C SP : ℝ) (h1 : SP = 1.25 * C) (h2 : SP = 8340) :
  C = 6672 :=
by
  sorry

end cost_price_of_computer_table_l207_207096


namespace any_natural_representation_l207_207424

theorem any_natural_representation (n : ℕ) : ∃ (k : ℕ) (u v : Fin k → ℕ),
  (∀ i j, i < j → u i > u j) ∧
  (∀ i j, i < j → v i < v j ∧ v i ≥ 0) ∧
  (n = ∑ i, 3 ^ (u i * 2 ^ v i)) :=
sorry

end any_natural_representation_l207_207424


namespace function_range_l207_207106

theorem function_range (f : ℝ → ℝ) (s : Set ℝ) (h : s = Set.Ico (-5 : ℝ) 2) (h_f : ∀ x ∈ s, f x = 3 * x - 1) :
  Set.image f s = Set.Ico (-16 : ℝ) 5 :=
sorry

end function_range_l207_207106


namespace population_ratio_l207_207139

variables (Px Py Pz : ℕ)

theorem population_ratio (h1 : Py = 2 * Pz) (h2 : Px = 8 * Py) : Px / Pz = 16 :=
by
  sorry

end population_ratio_l207_207139


namespace sum_of_d_and_e_l207_207079

def original_number1 : ℕ := 835697
def original_number2 : ℕ := 934821
def displayed_sum : ℕ := 1867428

theorem sum_of_d_and_e (d e : ℕ) (h : (λ n1 n2 : ℕ, 
    (λ f1 f2, f1 + f2 = displayed_sum) 
    (nat.digits 10 (nat.replace_digit 10 d e n1)) 
    (nat.digits 10 (nat.replace_digit 10 d e n2))) 
    original_number1 original_number2) :
    d + e = 7 :=
sorry

end sum_of_d_and_e_l207_207079


namespace range_of_k_l207_207281

noncomputable def triangle_range (A B C : ℝ) (a b c k : ℝ) : Prop :=
  A + B + C = Real.pi ∧
  (B = Real.pi / 3) ∧       -- From arithmetic sequence and solving for B
  a^2 + c^2 = k * b^2 ∧
  (1 < k ∧ k <= 2)

theorem range_of_k (A B C a b c k : ℝ) :
  A + B + C = Real.pi →
  (B = Real.pi - (A + C)) →
  (B = Real.pi / 3) →
  a^2 + c^2 = k * b^2 →
  0 < A ∧ A < 2*Real.pi/3 →
  1 < k ∧ k <= 2 :=
by
  sorry

end range_of_k_l207_207281


namespace count_sixth_powers_below_200_l207_207329

noncomputable def is_sixth_power (n : ℕ) : Prop := ∃ z : ℕ, n = z^6

theorem count_sixth_powers_below_200 : 
  (finset.filter (λ n, n < 200) (finset.filter is_sixth_power (finset.range 200))).card = 2 := 
by 
  sorry

end count_sixth_powers_below_200_l207_207329


namespace cos_A_in_triangle_l207_207380

variable {a b c : ℝ}
variable {A B C : ℝ} [OrderedField ℝ]

theorem cos_A_in_triangle
  (h : 2 * a * Real.sin A = (2 * b - c) * Real.sin B + (2 * c - b) * Real.sin C) :
  Real.cos A = 1 / 2 :=
sorry

end cos_A_in_triangle_l207_207380


namespace exists_q_no_zeros_l207_207860

theorem exists_q_no_zeros (n : ℕ) (hn : n ≥ 0) : ∃ q : ℚ, 
  ∀ k : ℕ, decimal_repr (q * 2^n) k ≠ '0' :=
begin
  sorry
end

end exists_q_no_zeros_l207_207860


namespace oranges_sold_l207_207020

theorem oranges_sold (joan_oranges_picked : ℕ) (joan_oranges_left : ℕ) (alyssa_pears_picked : ℕ) (H1 : joan_oranges_picked = 37) (H2 : joan_oranges_left = 27) : 
  ∃ (oranges_sold : ℕ), oranges_sold = 10 := 
by 
  use joan_oranges_picked - joan_oranges_left 
  rw [H1, H2] 
  norm_num 
  done

end oranges_sold_l207_207020


namespace triangle_count_l207_207999

/-- Define points coordinate constraints and calculate the number of possible triangles. -/
theorem triangle_count (h : ∀ x y : ℕ, 31 * x + y = 2017) : 
  ∑ p in finset.Icc 0 65, ∑ q in finset.Icc 0 65, (p ≠ q) ∧ (p - q) % 2 = 0 = 1056 :=
begin
  sorry
end

end triangle_count_l207_207999


namespace point_on_diagonal_l207_207906

-- Definitions of point and quadrilateral
structure Point where
  x : ℝ
  y : ℝ

structure Quadrilateral where
  A B C D : Point

-- Definition of the condition: Four triangles having equal area
def equal_areas (P : Point) (Q : Quadrilateral) : Prop :=
  let area (P1 P2 P3 : Point) : ℝ :=
    0.5 * abs ((P1.x - P3.x) * (P2.y - P3.y) - (P2.x - P3.x) * (P1.y - P3.y))
  area P Q.A Q.B = area P Q.B Q.C ∧
  area P Q.B Q.C = area P Q.C Q.D ∧
  area P Q.C Q.D = area P Q.D Q.A

-- Proposition: Given the equal area condition, prove P lies on one of the diagonals.
theorem point_on_diagonal (Q : Quadrilateral) (P : Point) (h : equal_areas P Q) :
  ∃ X Y : Point, (X = Q.A ∧ Y = Q.C ∨ X = Q.B ∧ Y = Q.D) ∧
  (∃ k : ℝ, P.x = X.x + k * (Y.x - X.x) ∧ P.y = X.y + k * (Y.y - X.y)) :=
sorry -- Proof omitted

end point_on_diagonal_l207_207906


namespace range_of_alpha_l207_207278

noncomputable def f (x : ℝ) : ℝ := (1/2) * (x^3 - 1/x)

def derivative_f (x : ℝ) : ℝ := (1/2) * (3 * x^2 + 1 / x^2)

theorem range_of_alpha (α : ℝ) (x : ℝ) (h : ∀ y, f'(y) = derivative_f y) : α ∈ set.Ico (π / 3) (π / 2) :=
begin
  sorry
end

end range_of_alpha_l207_207278


namespace infinite_square_pairs_l207_207110

theorem infinite_square_pairs:
  ∃ (f g : ℕ → ℕ), (∀ k : ℕ, f k = (10^k - 1)^2 ∧ g k = (5 * 10^(k-1) - 1)^2 ∧
  let N := (f k) * 10^(2*k) + (g k) in ∃ m : ℕ, N = m^2) :=
sorry

end infinite_square_pairs_l207_207110


namespace number_of_baskets_l207_207389

-- Definitions based on the given conditions
def baskets := Nat
def crabs_per_basket : Nat := 4
def collections_per_week : Nat := 2
def price_per_crab : Nat := 3
def total_earnings : Nat := 72

-- The statement to prove
theorem number_of_baskets (B : baskets) :
  (total_earnings / price_per_crab) / crabs_per_basket = B → B = 6 := by
sorry

end number_of_baskets_l207_207389


namespace hyperbola_equation_l207_207722

theorem hyperbola_equation
  (a b c e : ℝ)
  (h₁ : e = 5 / 4)
  (h₂ : c = 5)
  (h₃ : a = 4)
  (h₄ : b = Real.sqrt (c^2 - a^2)) :
  C : x^2 / a^2 - y^2 / b^2 = 1 := 
by
  have h_eq : C = ∀ x y, x^2 / 16 - y^2 / 9 = 1,
    from sorry,
  exact h_eq

end hyperbola_equation_l207_207722


namespace smallest_four_digit_palindromic_prime_l207_207497

-- Definitions based on conditions
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def is_palindromic (n : ℕ) : Prop := n.to_string.reverse.to_nat = n
def is_prime (n : ℕ) : Prop := nat.prime n

-- Final theorem statement
theorem smallest_four_digit_palindromic_prime : ∃ n : ℕ, is_four_digit n ∧ is_palindromic n ∧ is_prime n ∧ ∀ m : ℕ, is_four_digit m ∧ is_palindromic m ∧ is_prime m → n ≤ m :=
  by
  existsi 1661
  -- Proof steps would go here
  sorry

end smallest_four_digit_palindromic_prime_l207_207497


namespace nine_digit_no_zero_remainders_unique_l207_207386

theorem nine_digit_no_zero_remainders_unique : 
  ¬ ∃ (n : ℕ), 
    (∀ d ∈ [1,2,3,4,5,6,7,8,9], d ∣ n ∧ n < 10^9 ∧ n ≥ 10^8) ∧ 
    (∀ i j ∈ [1,2,3,4,5,6,7,8,9], i ≠ j → n % i ≠ n % j) :=
by
  sorry

end nine_digit_no_zero_remainders_unique_l207_207386


namespace find_b_range_l207_207303

noncomputable def has_zero_point_in_interval (f : ℝ → ℝ) (a b : ℝ) (s : set ℝ) : Prop :=
∃ x ∈ s, f x = 0

theorem find_b_range {a b : ℝ} 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : a + a^2 = 6) 
  (h4 : has_zero_point_in_interval (λ x, a^x + log a x + b) a b (set.Ioo 1 2)) :
  -5 < b ∧ b < -2 :=
sorry

end find_b_range_l207_207303


namespace highest_score_not_necessarily_20_l207_207172

theorem highest_score_not_necessarily_20:
  ∃ (scores : List ℕ), 
  length scores = 16 ∧
  ∀ i < length scores, scores[i] ≤ 15 ∧
  scores.max ≤ 15 :=
begin
  sorry
end

end highest_score_not_necessarily_20_l207_207172


namespace monotonic_decreasing_interval_l207_207888

noncomputable def function_domain (x : ℝ) : Prop :=
  x^2 - 3 * x > 0

theorem monotonic_decreasing_interval :
  ∀ x : ℝ, (function_domain x) → 
    IsDecreasing (λ x, log x^2 - 3 * x / log (1 / 3)) on_set (3, +∞) := sorry

end monotonic_decreasing_interval_l207_207888


namespace perpendicular_distance_l207_207218

theorem perpendicular_distance (D A B C : ℝ×ℝ×ℝ)
  (d_pos : D = (0, 0, 0))
  (a_pos : A = (5, 0, 0))
  (b_pos : B = (0, 5, 0))
  (c_pos : C = (0, 0, 4)) :
  let distance := 3.5 in
  ∀ P : ℝ×ℝ×ℝ, 
  P.1*(5-0) + P.2*(5-0) + P.3*(4-0) = 0 → 
  (((D.1 - P.1)^2 + (D.2 - P.2)^2 + (D.3 - P.3)^2)^(1/2) = distance) :=
by
  sorry

end perpendicular_distance_l207_207218


namespace largest_prime_factor_12321_l207_207650

theorem largest_prime_factor_12321 : ∃ p, prime p ∧ (∀ q, prime q ∧ q ∣ 12321 → q ≤ p) ∧ p = 19 :=
by {
  sorry
}

end largest_prime_factor_12321_l207_207650


namespace star_11_comm_star_11_assoc_star_11_cancel_star_12_comm_star_12_assoc_star_12_cancel_l207_207786

-- Define a function f : Fin n → Fin n for n = 11
def f_11 : Fin 11 → ℤ
| ⟨1, _⟩ := 11
| ⟨2, _⟩ := 1
| ⟨3, _⟩ := 5
| ⟨4, _⟩ := 2
| ⟨5, _⟩ := 7
| ⟨6, _⟩ := 6
| ⟨7, _⟩ := 4
| ⟨8, _⟩ := 3
| ⟨9, _⟩ := 10
| ⟨10, _⟩ := 8
| ⟨0, _⟩ := 9  -- Note: ⟨0, _⟩ represents 11 in Fin 11 as Lean starts counting from 0.

-- Define the inverse function f⁻¹ for n = 11
def f_inv_11 : ℤ → Fin 11
| 11 := ⟨1, by decide⟩
| 1 := ⟨2, by decide⟩
| 5 := ⟨3, by decide⟩
| 2 := ⟨4, by decide⟩
| 7 := ⟨5, by decide⟩
| 6 := ⟨6, by decide⟩
| 4 := ⟨7, by decide⟩
| 3 := ⟨8, by decide⟩
| 10 := ⟨9, by decide⟩
| 8 := ⟨10, by decide⟩
| 9 := ⟨0, by decide⟩

-- Define the multiplication operation * for the set S₁₁
def star_11 (a b : Fin 11) : Fin 11 :=
  f_inv_11 ((f_11 a + f_11 b) % 11)

-- Repeat similar definitions for n = 12
def f_12 : Fin 12 → ℤ
| ⟨1, _⟩ := 12
| ⟨2, _⟩ := 1
| ⟨3, _⟩ := 4
| ⟨4, _⟩ := 2
| ⟨5, _⟩ := 9
| ⟨6, _⟩ := 5
| ⟨7, _⟩ := 7
| ⟨8, _⟩ := 3
| ⟨9, _⟩ := 8
| ⟨10, _⟩ := 10
| ⟨11, _⟩ := 11
| ⟨0, _⟩ := 6  -- Note: ⟨0, _⟩ represents 12 in Fin 12 as Lean starts counting from 0.

-- Define the inverse function f⁻¹ for n = 12
def f_inv_12 : ℤ → Fin 12
| 12 := ⟨1, by decide⟩
| 1 := ⟨2, by decide⟩
| 4 := ⟨3, by decide⟩
| 2 := ⟨4, by decide⟩
| 9 := ⟨5, by decide⟩
| 5 := ⟨6, by decide⟩
| 7 := ⟨7, by decide⟩
| 3 := ⟨8, by decide⟩
| 8 := ⟨9, by decide⟩
| 10 := ⟨10, by decide⟩
| 11 := ⟨11, by decide⟩
| 6 := ⟨0, by decide⟩

-- Define the multiplication operation * for the set S₁₂
def star_12 (a b : Fin 12) : Fin 12 :=
  f_inv_12 ((f_12 a + f_12 b) % 12)

-- Define and prove the properties for n = 11
theorem star_11_comm (a b : Fin 11) : star_11 a b = star_11 b a := by
  sorry

theorem star_11_assoc (a b c : Fin 11) : star_11 (star_11 a b) c = star_11 a (star_11 b c) := by
  sorry

theorem star_11_cancel (a b c : Fin 11) : star_11 a b = star_11 a c → b = c := by
  sorry

-- Define and prove the properties for n = 12
theorem star_12_comm (a b : Fin 12) : star_12 a b = star_12 b a := by
  sorry

theorem star_12_assoc (a b c : Fin 12) : star_12 (star_12 a b) c = star_12 a (star_12 b c) := by
  sorry

theorem star_12_cancel (a b c : Fin 12) : star_12 a b = star_12 a c → b = c := by
  sorry

end star_11_comm_star_11_assoc_star_11_cancel_star_12_comm_star_12_assoc_star_12_cancel_l207_207786


namespace determine_a_range_l207_207885

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≤ 0 then real.exp (a / 3 * x)
else (3 * real.log x) / x

theorem determine_a_range :
  ∃ a, ∀ x ∈ Icc (-3 : ℝ) 3, f x a ≤ 3 / real.exp 1 → a ≥ 1 - log 3 :=
sorry

end determine_a_range_l207_207885


namespace factorize_expression_l207_207618

variable {R : Type} [CommRing R] (a x y : R)

theorem factorize_expression :
  a^2 * (x - y) + 9 * (y - x) = (x - y) * (a + 3) * (a - 3) :=
by
  sorry

end factorize_expression_l207_207618


namespace interval_of_monotonic_increase_range_of_a_l207_207300

noncomputable def f (a x : ℝ) : ℝ := ln x - a * x + (1 - a) / x + 1

-- Interval of monotonic increase
theorem interval_of_monotonic_increase (a x: ℝ) (hx : x > 0) :
  (a <= 0 → (∀ x ∈ Ioi 1, 0 ≤ (deriv (f a)) x)) ∧
  (0 < a ∧ a < 1/2 → (∀ x ∈ Ioo 1 ((1 - a) / a), 0 ≤ (deriv (f a)) x)) ∧
  (a = 1/2 → ¬ ∃ x, 0 ≤ (deriv (f a)) x) ∧
  (1/2 < a ∧ a <= 1 → (∀ x ∈ Ioo ((1 - a) / a) 1, 0 ≤ (deriv (f a)) x)) ∧
  (a > 1 → (∀ x ∈ Ioo 0 1, 0 ≤ (deriv (f a)) x)) :=
by
  sorry

-- Range of a given the minimum value condition
theorem range_of_a (a : ℝ) (a_in : (1/3 : ℝ) < a ∧ a < 1) :
  (∀ t ∈ Icc 2 3, ∀ x ∈ Ioc 0 t, (f a t) ≤ (f a) x) →
  (2 * ln 2 - 1) ≤ a ∧ a < 1 :=
by
  sorry

end interval_of_monotonic_increase_range_of_a_l207_207300


namespace clock_hands_equal_angle_l207_207354

/-- 
Given that the hour, minute, and second hands of a clock are moving uniformly,
and the starting time is 1:00, within 1 minute, the number of times that one of the three hands
forms an equal angle with the other two is 4.
-/
theorem clock_hands_equal_angle :
  ∀ (h m s : ℝ), (uniform_motion h ∧ uniform_motion m ∧ uniform_motion s) →
  (start_time h m s = (1, 0, 0) ∧ time_window = 1) →
  equal_angle_count_within_minute h m s = 4 :=
sorry

end clock_hands_equal_angle_l207_207354


namespace sum_of_first_500_natural_numbers_l207_207518

theorem sum_of_first_500_natural_numbers : (Range 501).Sum = 125250 := by
  sorry

end sum_of_first_500_natural_numbers_l207_207518


namespace grade_received_no_more_than_twice_l207_207144

theorem grade_received_no_more_than_twice 
  (grades : Fin 17 → ℕ) 
  (h1 : ∀ i, grades i ∈ {2, 3, 4, 5}) 
  (h2 : (Finset.univ.sum grades) % 17 = 0) : 
  ∃ grade ∈ {2, 3, 4, 5}, (Finset.filter (λ i, grades i = grade) Finset.univ).card ≤ 2 := 
sorry

end grade_received_no_more_than_twice_l207_207144


namespace circle_area_isosceles_triangle_l207_207540

theorem circle_area_isosceles_triangle (a b c : ℝ) (h1 : a = 5) (h2 : b = 5) (h3 : c = 4) :
  let r := ((25 * Real.sqrt 21) / 42)
  in ∃ (O : Point) (r : ℝ), Circle O r ∧ r^2 * Real.pi = (13125 / 1764) * Real.pi := by
  sorry

end circle_area_isosceles_triangle_l207_207540


namespace greatest_positive_integer_k_l207_207396

theorem greatest_positive_integer_k (n : ℕ) (h1 : n ≥ 4) : 
  ∃ k : ℕ, k = ⌊(n - 1) / 3⌋ ∧ 
  ∃ a b c : ℕ, 1 ≤ a ∧ a ≤ n ∧ 1 ≤ b ∧ b ≤ n ∧ 1 ≤ c ∧ c ≤ n ∧ 
  a + b > c ∧ b + c > a ∧ c + a > b ∧ 
  b - a ≥ k ∧ c - b ≥ k :=
begin
  sorry
end

end greatest_positive_integer_k_l207_207396


namespace find_divisor_l207_207849

theorem find_divisor 
  (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) 
  (h_dividend : dividend = 190) (h_quotient : quotient = 9) (h_remainder : remainder = 1) :
  ∃ divisor : ℕ, dividend = divisor * quotient + remainder ∧ divisor = 21 := 
by
  sorry

end find_divisor_l207_207849


namespace angle_ADC_is_90_degrees_l207_207461

theorem angle_ADC_is_90_degrees
  (ABC_triangle : Type)
  (A B C A' B' C' D : ABC_triangle)
  (incircle_touches_A' : A' touches side BC)
  (incircle_touches_B' : B' touches side CA)
  (incircle_touches_C' : C' touches side AB)
  (A'C'_line_meet : line A' C' ∩ angle_bisector(A) = D) :
  ∠ADC = 90° := sorry

end angle_ADC_is_90_degrees_l207_207461


namespace smallest_five_digit_palindrome_divisible_by_7_l207_207493

theorem smallest_five_digit_palindrome_divisible_by_7 :
  ∃ (n : ℕ), 10000 ≤ n ∧ n < 100000 ∧ 
    (∃ (A B C : ℕ), A ≠ 0 ∧ n = 10001 * A + 1010 * B + 100 * C ∧ 2 * A + 5 * B + 4 * C ≡ 0 [MOD 7]) ∧ 
    ∀ m, (10000 ≤ m ∧ m < 100000 ∧ 
          (∃ (D E F : ℕ), D ≠ 0 ∧ m = 10001 * D + 1010 * E + 100 * F ∧ 2 * D + 5 * E + 4 * F ≡ 0 [MOD 7])) 
          → n ≤ m :=
exists.intro 10101 ⟨by norm_num,by norm_num,
  ⟨1,1,0,by norm_num, by norm_num, by norm_num⟩,
  λ m h, sorry
⟩

end smallest_five_digit_palindrome_divisible_by_7_l207_207493


namespace find_ax5_by5_l207_207347

variables (a b x y: ℝ)

theorem find_ax5_by5 (h1 : a * x + b * y = 5)
                      (h2 : a * x^2 + b * y^2 = 11)
                      (h3 : a * x^3 + b * y^3 = 24)
                      (h4 : a * x^4 + b * y^4 = 56) :
                      a * x^5 + b * y^5 = 180.36 :=
sorry

end find_ax5_by5_l207_207347


namespace b_alone_time_l207_207352

variable (a b c : Type)
variable [LinearOrderedField A B C]

theorem b_alone_time (work_rate_combined : (A + B + C = 1/4)) 
  (work_rate_a : (A = 1/12)) 
  (work_rate_c : (C = 1/9)) :
  B = 1/18 := 
sorry

end b_alone_time_l207_207352


namespace triangle_ABC_properties_l207_207761

theorem triangle_ABC_properties
  (a b : ℝ)
  (h1 : ∀ x, x^2 - 2*sqrt(3)*x + 2 = 0 → x = a ∨ x = b)
  (cos_sum : 2*cos(∡A + ∡B) = 1)
  (A B C : X) -- A, B, C are points forming the triangle ABC
  (BC : dist B C = a)
  (AC : dist A C = b) :
  (∡C = 120 ∧ dist A B = sqrt(10)) :=
by
  sorry

end triangle_ABC_properties_l207_207761


namespace champagne_bottles_needed_l207_207574

-- Define the initial conditions of the problem
def num_guests : ℕ := 120
def glasses_per_guest : ℕ := 2
def servings_per_bottle : ℕ := 6

-- The statement we need to prove
theorem champagne_bottles_needed : 
  (num_guests * glasses_per_guest) / servings_per_bottle = 40 := 
by
  sorry

end champagne_bottles_needed_l207_207574


namespace sequence_bound_l207_207692

/-- This definition states that given the initial conditions and recurrence relation
for a sequence of positive integers, the 2021st term is greater than 2^2019. -/
theorem sequence_bound (a : ℕ → ℕ) (h_initial : a 2 > a 1)
  (h_recurrence : ∀ n, a (n + 2) = 3 * a (n + 1) - 2 * a n) :
  a 2021 > 2 ^ 2019 :=
sorry

end sequence_bound_l207_207692


namespace solve_for_x_l207_207440

-- We define that the condition and what we need to prove.
theorem solve_for_x (x : ℝ) : (x + 7) / (x - 4) = (x - 3) / (x + 6) → x = -3 / 2 :=
by sorry

end solve_for_x_l207_207440


namespace inequality_proof_l207_207750

-- Conditions: a > b and c > d
variables {a b c d : ℝ}

-- The main statement to prove: d - a < c - b with given conditions
theorem inequality_proof (h1 : a > b) (h2 : c > d) : d - a < c - b := 
sorry

end inequality_proof_l207_207750


namespace number_exceeds_self_percentage_by_l207_207937

theorem number_exceeds_self_percentage_by (x : Real) : (x - 0.16 * x = 105) → x = 125 :=
by
  intro h
  calc x = 125 : sorry

end number_exceeds_self_percentage_by_l207_207937


namespace bricks_of_other_types_l207_207052

theorem bricks_of_other_types (A B total other: ℕ) (hA: A = 40) (hB: B = A / 2) (hTotal: total = 150) (hSum: total = A + B + other): 
  other = 90 :=
by sorry

end bricks_of_other_types_l207_207052


namespace original_inhabitants_l207_207193

theorem original_inhabitants (X : ℝ) 
  (h1 : 10 ≤ X) 
  (h2 : 0.9 * X * 0.75 + 0.225 * X * 0.15 = 5265) : 
  X = 7425 := 
sorry

end original_inhabitants_l207_207193


namespace trapezoid_LM_value_l207_207812

theorem trapezoid_LM_value (K L M N P Q : Type) 
  (d1 d2 : ℝ)
  (h1 : d1 = 1)
  (h2 : d2 = 1)
  (height_eq : KM = 1)
  (KN_eq_MQ : KN = MQ)
  (LM_eq_MP : LM = MP) :
  LM = 1 / real.sqrt (real.sqrt 2) :=
by 
  sorry

end trapezoid_LM_value_l207_207812


namespace derivative_of_f_l207_207988

noncomputable def f (x : ℝ) : ℝ := (Real.sin (1 / x)) ^ 3

theorem derivative_of_f (x : ℝ) (hx : x ≠ 0) : 
  deriv f x = - (3 / x ^ 2) * (Real.sin (1 / x)) ^ 2 * Real.cos (1 / x) :=
by
  sorry 

end derivative_of_f_l207_207988


namespace park_area_l207_207470

theorem park_area (l w : ℝ) (h1 : 2 * l + 2 * w = 80) (h2 : l = 3 * w) : l * w = 300 :=
sorry

end park_area_l207_207470


namespace sector_area_proof_area_of_sector_l207_207708

variable (l : ℝ) (α : ℝ) (r : ℝ) (S : ℝ)

-- Given conditions
axiom h1 : α = 1
axiom h2 : l = 2
axiom h3 : α = l / r

-- Goal to prove: The area of the sector
theorem sector_area : S = (1 / 2) * l * r := by
  sorry

-- Since we know the radius r and the α relationship
noncomputable def radius : ℝ := l / α

-- Given radius is r = 2
axiom h_radius : radius = 2

-- The key part of the proof
theorem proof_area_of_sector : S = 2 := by
  -- Using the given conditions and theorem sector_area
  have r_def : r = 2 := by
    sorry   -- Proof that r = 2 follows directly from the given axiom h_radius
  have sector_area_def : S = (1 / 2) * l * r := by
    apply sector_area
    -- The equation should hold directly because of the setup
  -- Substitute the given values
  rw [r_def, h2]
  calc S = (1/2) * 2 * 2 : sorry
   ... = 2 : by norm_num


end sector_area_proof_area_of_sector_l207_207708


namespace Bob_can_determine_polynomial_l207_207976

theorem Bob_can_determine_polynomial (P : ℤ[X]) (h : ∀ n ∈ P.coeffs, n ≥ 0) : 
  ∃ a b : ℤ, ∃ c : ℕ, a = 1 ∧ b = c + 1 ∧ (P.eval a = c ∧ P.eval b = P.eval (c + 1)) →
  ∀ Q : ℤ[X], (∀ n ∈ Q.coeffs, n ≥ 0) → (P.eval a = Q.eval a ∧ P.eval b = Q.eval b) → P = Q :=
by sorry

end Bob_can_determine_polynomial_l207_207976


namespace cos_alpha_sum_numerator_denominator_l207_207358

noncomputable def central_angles_and_cos_sum : Prop :=
∀ (α β : ℝ) (h_alpha_beta : α + β < real.pi)
  (h_cos_alpha_positive : 0 < real.cos α),
  let cos_half_alpha := 21 / 24 in
  let cos_alpha := 2 * (cos_half_alpha)^2 - 1 in
  (let (num, denom) := (17, 32) in
   num + denom = 49) ∧ (real.cos α = num / denom)

theorem cos_alpha_sum_numerator_denominator :
  central_angles_and_cos_sum :=
sorry

end cos_alpha_sum_numerator_denominator_l207_207358


namespace equation_of_plane_l207_207883

-- Define the coordinates of the given point
def point : ℝ × ℝ × ℝ := (15, -3, 6)

-- Define the condition that the point is the foot of the perpendicular from the origin
def on_plane (x y z : ℝ) : Prop :=
  ∃ (A B C D : ℤ), 
  A*x + B*y + C*z + D = 0 ∧ 
  point = (A, B, C) ∧ 
  A > 0 ∧ 
  Int.gcd (Int.gcd (Int.gcd A B) C) D = 1

-- The statement to be proven
theorem equation_of_plane : ∃ (A B C D : ℤ), 
  on_plane 15 (-3) 6 ∧
  A = 5 ∧ B = -1 ∧ C = 2 ∧ D = -90 :=
sorry

end equation_of_plane_l207_207883


namespace work_days_l207_207939

theorem work_days (Dx Dy : ℝ) (H1 : Dy = 45) (H2 : 8 / Dx + 36 / Dy = 1) : Dx = 40 :=
by
  sorry

end work_days_l207_207939


namespace champagne_bottles_needed_l207_207576

-- Define the initial conditions of the problem
def num_guests : ℕ := 120
def glasses_per_guest : ℕ := 2
def servings_per_bottle : ℕ := 6

-- The statement we need to prove
theorem champagne_bottles_needed : 
  (num_guests * glasses_per_guest) / servings_per_bottle = 40 := 
by
  sorry

end champagne_bottles_needed_l207_207576


namespace g_675_eq_42_l207_207037

noncomputable def g : ℕ → ℕ := sorry

axiom gxy : ∀ (x y : ℕ), g (x * y) = g x + g y
axiom g15 : g 15 = 18
axiom g45 : g 45 = 24

theorem g_675_eq_42 : g 675 = 42 :=
sorry

end g_675_eq_42_l207_207037


namespace angle_BPC_is_90_l207_207371

theorem angle_BPC_is_90
  (A B C D E P Q : Type)
  (BC : BC)
  (sides_ABCD_eq_6 : sides_ABCD_eq_6 A B C D)
  (isosceles_ABE : isosceles_ABE A B E)
  (angle_ABE_45 : angle_ABE_45 A B E)
  (intersect_BE_AC_at_P : intersect_BE_AC_at_P B E A C P)
  (perpendicular_PQ_BC : perpendicular_PQ_BC P Q B C)
  (PQ_eq_x : PQ_eq_x P Q x) :
  angle_BPC_is_90 B P C := 
begin 
  sorry 
end

end angle_BPC_is_90_l207_207371


namespace relationship_between_a_and_b_l207_207755

theorem relationship_between_a_and_b : 
  ∀ (a b : ℝ), (∀ x y : ℝ, (x-a)^2 + (y-b)^2 = b^2 + 1 → (x+1)^2 + (y+1)^2 = 4 → (2 + 2*a)*x + (2 + 2*b)*y - a^2 - 1 = 0) → a^2 + 2*a + 2*b + 5 = 0 :=
by
  intros a b hyp
  sorry

end relationship_between_a_and_b_l207_207755


namespace coeff_x2_expansion_l207_207007

theorem coeff_x2_expansion : 
  (let f (n : ℕ) := (1 + 1 : ℝ) + (1 + 1 : ℝ) ^ n + ∑ i in (2:ℕ)..9, (binom i 2)
  in f (10 - 1) = 120) :=
by
  sorry

end coeff_x2_expansion_l207_207007


namespace polynomial_factor_l207_207473

theorem polynomial_factor (a b : ℝ) : 
  (∃ c d : ℝ, (5 * c = a) ∧ (5 * d - 3 * c = b) ∧ (2 * c - 3 * d + 25 = 45) ∧ (2 * d - 15 = -18)) 
  → (a = 151.25 ∧ b = -98.25) :=
by
  sorry

end polynomial_factor_l207_207473


namespace perpendicular_lines_l207_207758

theorem perpendicular_lines (a : ℝ) : (x + 2*y + 1 = 0) ∧ (ax + y - 2 = 0) → a = -2 :=
by
  sorry

end perpendicular_lines_l207_207758


namespace sin_double_angle_values_l207_207745

theorem sin_double_angle_values (α : ℝ) (hα : 0 < α ∧ α < π) (h : 3 * (Real.cos α)^2 = Real.sin ((π / 4) - α)) :
  Real.sin (2 * α) = 1 ∨ Real.sin (2 * α) = -17 / 18 :=
by
  sorry

end sin_double_angle_values_l207_207745


namespace zhang_hua_success_probability_l207_207505

theorem zhang_hua_success_probability :
  ∃ (d1 d2 d3 : ℕ), (d1 = 0 ∧ d2 = 2 ∧ d3 = 8) ∧ 
                     (finset.card (finset.univ : finset (fin 6)) = 6) ∧
                     (1 / 6 = 1 / 6) := 
  sorry

end zhang_hua_success_probability_l207_207505


namespace volume_of_cylinder_l207_207709

theorem volume_of_cylinder (r h : ℝ) (hr : r = 1) (hh : h = 2) (A : r * h = 4) : (π * r^2 * h = 2 * π) :=
by
  sorry

end volume_of_cylinder_l207_207709


namespace prob_condition1_prob_condition2_l207_207714

noncomputable def f (x : ℝ) : ℝ :=
  sqrt 3 * sin (2 * x - π / 6)

theorem prob_condition1 (ω : ℝ) (φ : ℝ) (hω : ω = 2) (hφ : φ = -π / 6) :
  f x = sqrt 3 * sin (2 * x - π / 6) := by
  sorry

theorem prob_condition2 (α : ℝ) (hα1 : π / 6 < α) (hα2 : α < 2 * π / 3)
  (h : f (α / 2) = 4 * sqrt 3 / 5) :
  sin α = (4 * sqrt 3 + 3) / 10 := by
  sorry

end prob_condition1_prob_condition2_l207_207714


namespace rectangle_from_similar_isosceles_triangles_l207_207982

theorem rectangle_from_similar_isosceles_triangles :
  (∃ (T : Type) [triangle T], 
    ∃ (a b : T) [isosceles_triangle a b] 
    (angles : a.angles = {30, 30, 120} ∧ b.angles = {30, 30, 120}),
    ∃ (rectangle : Type), formed_rectangle rectangle):
  True :=
sorry

end rectangle_from_similar_isosceles_triangles_l207_207982


namespace cement_weight_in_pounds_l207_207946

theorem cement_weight_in_pounds :
  let kg_to_lb : ℝ := 1 / 0.45,
      cement_weight_kg : ℝ := 150
  in
  let cement_weight_lb := cement_weight_kg * kg_to_lb
  in
  (cement_weight_lb ≈ 333)  := -- Rounded to the nearest whole number
sorry

end cement_weight_in_pounds_l207_207946


namespace evaluate_expression_l207_207996

theorem evaluate_expression (x : ℝ) (hx : x > 0) : Real.root 4 ((x * Real.sqrt x)^2) = x^(3 / 4) :=
by sorry

end evaluate_expression_l207_207996


namespace cement_mixture_weight_l207_207157

theorem cement_mixture_weight :
  ∃ (W : ℕ), (1/3 : ℚ) * W + (1/4 : ℚ) * W + 10 = W ∧ W = 24 :=
by {
  use 24,
  split,
  { 
    norm_num,
    linarith
  },
  { 
    norm_num
  }
}

end cement_mixture_weight_l207_207157


namespace sum_of_200_terms_l207_207282

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)
variable (a1 a200 : ℝ)

-- Conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a n = a 0 + n * (a 1 - a 0)

def sum_of_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
∀ n, S n = (n * (a 1 + a n)) / 2

def collinearity_condition (a1 a200 : ℝ) : Prop :=
a1 + a200 = 1

-- Proof statement
theorem sum_of_200_terms (a : ℕ → ℝ) (S : ℕ → ℝ) (a1 a200 : ℝ) 
  (h_seq : arithmetic_sequence a)
  (h_sum : sum_of_first_n_terms S a)
  (h_collinear : collinearity_condition a1 a200) : 
  S 200 = 100 := 
sorry

end sum_of_200_terms_l207_207282


namespace champagne_bottles_needed_l207_207575

-- Define the initial conditions of the problem
def num_guests : ℕ := 120
def glasses_per_guest : ℕ := 2
def servings_per_bottle : ℕ := 6

-- The statement we need to prove
theorem champagne_bottles_needed : 
  (num_guests * glasses_per_guest) / servings_per_bottle = 40 := 
by
  sorry

end champagne_bottles_needed_l207_207575


namespace total_cost_of_paving_l207_207176

theorem total_cost_of_paving 
  (length : ℝ)
  (width : ℝ)
  (square_slab_side : ℝ)
  (square_slab_cost_per_sq_meter : ℝ)
  (rectangular_slab_length : ℝ)
  (rectangular_slab_width : ℝ)
  (rectangular_slab_cost_per_sq_meter : ℝ)
  (triangular_slab_height : ℝ)
  (triangular_slab_base : ℝ)
  (triangular_slab_cost_per_sq_meter : ℝ)
  (square_slab_percentage : ℝ)
  (rectangular_slab_percentage : ℝ)
  (triangular_slab_percentage : ℝ)
  : (length = 5.5) →
    (width = 3.75) →
    (square_slab_side = 1) →
    (square_slab_cost_per_sq_meter = 800) →
    (rectangular_slab_length = 1.5) →
    (rectangular_slab_width = 1) →
    (rectangular_slab_cost_per_sq_meter = 1000) →
    (triangular_slab_height = 1) →
    (triangular_slab_base = 1) →
    (triangular_slab_cost_per_sq_meter = 1200) →
    (square_slab_percentage = 0.40) →
    (rectangular_slab_percentage = 0.35) →
    (triangular_slab_percentage = 0.25) →
    let total_area := length * width in 
    let area_square_slab := square_slab_percentage * total_area in 
    let area_rectangular_slab := rectangular_slab_percentage * total_area in 
    let area_triangular_slab := triangular_slab_percentage * total_area in 
    let cost_square_slab := area_square_slab * square_slab_cost_per_sq_meter in 
    let cost_rectangular_slab := area_rectangular_slab * rectangular_slab_cost_per_sq_meter in 
    let cost_triangular_slab := area_triangular_slab * triangular_slab_cost_per_sq_meter in 
    let total_cost := cost_square_slab + cost_rectangular_slab + cost_triangular_slab in 
    total_cost = 20006.25 := 
by 
  intros; 
  simp [total_area, area_square_slab, area_rectangular_slab, area_triangular_slab, cost_square_slab, cost_rectangular_slab, cost_triangular_slab, total_cost]; 
  sorry

end total_cost_of_paving_l207_207176


namespace p_range_l207_207834

def h (x : ℝ) : ℝ := 2 * x + 3

def p (x : ℝ) : ℝ := h (h (h (h x)))

theorem p_range :
  ∀ x, -1 ≤ x ∧ x ≤ 3 → 29 ≤ p x ∧ p x ≤ 93 :=
by
  intros x hx
  sorry

end p_range_l207_207834


namespace concurrency_and_circumcircle_l207_207208

-- Given data defining intersection points and tangency conditions
variables {A B C G H U V T W D E F: Type} 
[A-excircle : tangent AB AC at G H]
[B-excircle : tangent AB BC at U V]
[C-excircle : tangent AC BC at T W]
[circumcircle : tangent_circle ABC at D E F]
{X Y Z : Type} 
[intersection_points : (UT GW HV) intersect at (X Y Z)]

-- The statement to be proven
theorem concurrency_and_circumcircle (A B C G H U V T W D E F X Y Z : Type)
   [A_excircle: tangent AB AC at G H]
   [B_excircle: tangent AB BC at U V]
   [C_excircle: tangent AC BC at T W]
   [circumcircle: tangent_circle ABC at D E F]
   [intersection_points: (UT GW HV) intersect at (X Y Z)] :
   concurrent (FZ XE YD) ∧ (intersection_point_of FZ XE YD) lies on circumcircle :=
by 
sorry

end concurrency_and_circumcircle_l207_207208


namespace calculate_total_students_l207_207360

/-- Define the number of students who like basketball, cricket, and soccer. -/
def likes_basketball : ℕ := 7
def likes_cricket : ℕ := 10
def likes_soccer : ℕ := 8
def likes_all_three : ℕ := 2
def likes_basketball_and_cricket : ℕ := 5
def likes_basketball_and_soccer : ℕ := 4
def likes_cricket_and_soccer : ℕ := 3

/-- Calculate the number of students who like at least one sport using the principle of inclusion-exclusion. -/
def students_who_like_at_least_one_sport (b c s bc bs cs bcs : ℕ) : ℕ :=
  b + c + s - (bc + bs + cs) + bcs

theorem calculate_total_students :
  students_who_like_at_least_one_sport likes_basketball likes_cricket likes_soccer 
    (likes_basketball_and_cricket - likes_all_three) 
    (likes_basketball_and_soccer - likes_all_three) 
    (likes_cricket_and_soccer - likes_all_three) 
    likes_all_three = 21 := 
by
  sorry

end calculate_total_students_l207_207360


namespace triangle_distance_sum_eq_8_l207_207187

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

noncomputable def sum_of_distances (A B C P : ℝ × ℝ) : ℝ :=
  let d1 := distance A.1 A.2 P.1 P.2
  let d2 := distance B.1 B.2 P.1 P.2
  let d3 := distance C.1 C.2 P.1 P.2
  d1 + d2 + d3

theorem triangle_distance_sum_eq_8 
  (A B C P : ℝ × ℝ)
  (hA : A = (0, 0))
  (hB : B = (10, -2))
  (hC : C = (7, 6))
  (hP : P = (6, 3)) :
  ∃ m n p : ℤ, 
  (sum_of_distances A B C P = (m : ℝ) + (n : ℝ) * real.sqrt (p : ℝ)) ∧ 
  (m + n + p = 8) :=
by
  sorry

end triangle_distance_sum_eq_8_l207_207187


namespace students_joined_l207_207984

-- Define the conditions
def students_10th_grade : ℕ := 150
def students_left_final_year : ℕ := 15
def students_end_final_year : ℕ := 165

-- Define the number of students who joined in the following year as X
def X : ℕ := students_end_final_year + students_left_final_year - students_10th_grade

-- Prove that the number of students who joined is 30
theorem students_joined : X = 30 := by
  have h1 : X = 165 + 15 - 150 := rfl
  rw [h1]
  norm_num
  sorry

end students_joined_l207_207984


namespace butterfly_problem_solved_l207_207490

variables {A B O E F' : Type} [ProjectiveSpace A B O E] [CrossRatio A B O E]

/-- Given definitions and conditions -/
variables 
  (symm_F'_F_O : symmetric_coefficient F' F O)
  (proj_trans-AB : ∀ (M Q : Type) (S: Circle), 
    projective_transform_AB_to_S_and_back AB S M Q = projective_transformation)
  (combined_transformation : CombinedTransformation A B O E B A F' O)
  (equal_cross_ratios : (A B O E) = (B A F' O))

/-- To prove: F' = E -/
theorem butterfly_problem_solved 
  : F' = E := 
sorry

end butterfly_problem_solved_l207_207490


namespace part_I_part_II_l207_207717

noncomputable def f (a x : ℝ) : ℝ := a - 1 / (2^x + 1)

theorem part_I (a : ℝ) : ∀ x : ℝ, (0 < (2^x * Real.log 2) / (2^x + 1)^2) :=
by
  sorry

theorem part_II (h : ∀ x : ℝ, f a x = -f a (-x)) : 
  a = (1:ℝ)/2 ∧ ∀ x : ℝ, -((1:ℝ)/2) < f (1/2) x ∧ f (1/2) x < (1:ℝ)/2 :=
by
  sorry

end part_I_part_II_l207_207717


namespace difference_area_circle_triangle_l207_207951

open Real

noncomputable def areaCircle (R : ℝ) : ℝ :=
  π * R^2

noncomputable def areaEquilateralTriangle (s : ℝ) : ℝ :=
  (s^2 * sqrt 3) / 4

theorem difference_area_circle_triangle : 
  let s := 12 in
  let R := 2 * (s / 2 * sqrt 3 / 3) in
  areaCircle R - areaEquilateralTriangle s = 144 * π - 36 * sqrt 3 :=
by 
  let s := 12
  let R := 2 * (s / 2 * sqrt 3 / 3)
  let areaCirc := areaCircle R
  let areaTri := areaEquilateralTriangle s
  calc
  areaCirc - areaTri = sorry

end difference_area_circle_triangle_l207_207951


namespace clock_store_sale_l207_207135

theorem clock_store_sale (C : ℝ) (h1 : C - 0.60 * C = 100) : 
  let buy_back_price := 0.60 * C in
  let second_selling_price := buy_back_price + 0.80 * buy_back_price in
  second_selling_price = 270 :=
by
  sorry

end clock_store_sale_l207_207135


namespace megan_initial_markers_l207_207409

theorem megan_initial_markers (gave : ℕ) (total : ℕ) (initial : ℕ) 
  (h1 : gave = 109) 
  (h2 : total = 326) 
  (h3 : initial + gave = total) : 
  initial = 217 := 
by 
  sorry

end megan_initial_markers_l207_207409


namespace find_distance_d_l207_207372

theorem find_distance_d (d : ℝ) (XR : ℝ) (YP : ℝ) (XZ : ℝ) (YZ : ℝ) (XY : ℝ) (h1 : XR = 3) (h2 : YP = 12) (h3 : XZ = 3 + d) (h4 : YZ = 12 + d) (h5 : XY = 15) (h6 : (XZ)^2 + (XY)^2 = (YZ)^2) : d = 5 :=
sorry

end find_distance_d_l207_207372


namespace greatest_number_in_set_S_l207_207935

theorem greatest_number_in_set_S (S : Set ℕ) (h1 : ∀ n ∈ S, n % 5 = 0) (h2 : |S| = 45) (h3 : ∃ a ∈ S, (∀ b ∈ S, a ≤ b)) (h4 : ∃ a ∈ S, a = 55) : ∃ b ∈ S, (∀ a ∈ S, a ≤ b) ∧ b = 275 :=
by {
     sorry
}

end greatest_number_in_set_S_l207_207935


namespace relationship_among_abc_l207_207263

noncomputable def a : ℝ := 2 ^ (-1 / 3)
noncomputable def b : ℝ := (2 ^ (Real.log 3 / Real.log 2)) ^ (-1 / 2)
noncomputable def c : ℝ := (1 / 4) * ∫ x in 0..Real.pi, Real.sin x

theorem relationship_among_abc : a > b ∧ b > c :=
by
  -- Definitions from conditions
  have a_val : a = 2 ^ (-1 / 3) := rfl
  have b_val : b = (2 ^ (Real.log 3 / Real.log 2)) ^ (-1 / 2) := rfl
  have c_val : c = (1 / 4) * ∫ x in 0..Real.pi, Real.sin x := rfl

  -- Create intermediate expressions for verification
  have a_simplified : a = 1 / Real.cbrt 2 := sorry
  have b_simplified : b = 1 / Real.sqrt 3 := sorry
  have c_simplified: c = 1 / 2 := sorry
  
  -- Final assertion
  sorry

end relationship_among_abc_l207_207263


namespace sin_390_eq_l207_207210

theorem sin_390_eq :
  sin (390 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end sin_390_eq_l207_207210


namespace maximize_ratio_l207_207604

noncomputable theory

def isosceles_triangle (α : ℝ) : Prop :=
  0 < α ∧ α < 90

def r (α : ℝ) : ℝ :=
  (Real.sin α) / (1 + Real.cos α)

def R (α : ℝ) (a : ℝ) : ℝ :=
  a / (2 * Real.sin α)

def ratio (α : ℝ) (a : ℝ) : ℝ :=
  (r α) / (R α a)

theorem maximize_ratio :
  ∀ (α : ℝ) (a : ℝ), isosceles_triangle α → α = 60 ∨ α = 45 → ratio α a = 4 * Real.sin (α / 2)^2 * Real.cos (α / 2)^2 :=
by
  intros α a hα hangle
  sorry

end maximize_ratio_l207_207604


namespace count_sixth_powers_below_200_l207_207328

noncomputable def is_sixth_power (n : ℕ) : Prop := ∃ z : ℕ, n = z^6

theorem count_sixth_powers_below_200 : 
  (finset.filter (λ n, n < 200) (finset.filter is_sixth_power (finset.range 200))).card = 2 := 
by 
  sorry

end count_sixth_powers_below_200_l207_207328


namespace length_of_CD_l207_207060

theorem length_of_CD (x y u v : ℝ) (R S C D : ℝ → ℝ)
  (h1 : 5 * x = 3 * y)
  (h2 : 7 * u = 4 * v)
  (h3 : u = x + 3)
  (h4 : v = y - 3)
  (h5 : C x + D y = 1) : 
  x + y = 264 :=
by
  sorry

end length_of_CD_l207_207060


namespace distance_after_second_sign_l207_207408

-- Define the known conditions
def total_distance_ridden : ℕ := 1000
def distance_to_first_sign : ℕ := 350
def distance_between_signs : ℕ := 375

-- The distance Matt rode after passing the second sign
theorem distance_after_second_sign :
  total_distance_ridden - (distance_to_first_sign + distance_between_signs) = 275 := by
  sorry

end distance_after_second_sign_l207_207408


namespace valid_permutations_count_l207_207320

theorem valid_permutations_count : 
  ∃ S : Finset ℕ, (∀ n ∈ S, 100 ≤ n ∧ n ≤ 999 ∧ (∃ m : ℕ, m ≤ factorial 3 ∧ some_permutation_multiple_of_9 n m)) ∧ S.card = 420 :=
begin
  sorry
end

-- Helper predicate to check if some permutation of the digits forms a multiple of 9
def some_permutation_multiple_of_9 (n : ℕ) (m : ℕ) : Prop := 
  m.digits.map (λ _, n.digits).perms.any (λ p, p.to_nat % 9 = 0)

end valid_permutations_count_l207_207320


namespace probability_1_girl_3_boys_max_value_n_l207_207114

-- Definitions
def total_students := 9
def boys := 5
def girls := 4
def select_count := 4
def total_combinations := Nat.choose total_students select_count

-- Definition for the probability of selecting 1 girl and 3 boys
def select_1_girl_3_boys :=
  let combinations_1_girl_3_boys := Nat.choose girls 1 * Nat.choose boys 3
  combinations_1_girl_3_boys / total_combinations.toRational

theorem probability_1_girl_3_boys : select_1_girl_3_boys = 20 / 63 :=
by
  sorry

-- Definition for P_n
def P (n : ℕ) :=
  let ways := ∑ k in Finset.range (n + 1), (Nat.choose boys (select_count - k) * Nat.choose girls k)
  ways.toRational / total_combinations.toRational

theorem max_value_n : ∃ n, P n ≥ 3 / 4 ∧ ∀ m, m > n → P m < 3 / 4 :=
by
  use 2
  sorry

end probability_1_girl_3_boys_max_value_n_l207_207114


namespace find_opposite_pair_l207_207563

def is_opposite (x y : ℤ) : Prop := x = -y

theorem find_opposite_pair :
  ¬is_opposite 4 4 ∧ ¬is_opposite 2 2 ∧ ¬is_opposite (-8) (-8) ∧ is_opposite 4 (-4) := 
by
  sorry

end find_opposite_pair_l207_207563


namespace eq_triangle_iff_eq_zero_eq_triangle_iff_square_equals_l207_207138

noncomputable def ε : ℂ := (1 / 2 : ℂ) + (complex.I * real.sqrt 3 / 2 : ℂ)

theorem eq_triangle_iff_eq_zero (a b c : ℂ) :
  (a + ε^2 * b + ε^4 * c = 0 ∨ a + ε^4 * b + ε^2 * c = 0) ↔ 
  ∃ (u v w : ℂ), u = a ∨ u = b ∨ u = c ∧ 
                  v = a ∨ v = b ∨ v = c ∧ 
                  w = a ∨ w = b ∨ w = c ∧ 
                  (u - v) = (ε * (v - w)) :=
sorry

theorem eq_triangle_iff_square_equals (a b c : ℂ) :
  (a + ε^2 * b + ε^4 * c = 0 ∨ a + ε^4 * b + ε^2 * c = 0) ↔
  a^2 + b^2 + c^2 = a * b + b * c + c * a :=
sorry

end eq_triangle_iff_eq_zero_eq_triangle_iff_square_equals_l207_207138


namespace percent_value_quarters_l207_207134

noncomputable def value_in_cents (dimes quarters nickels : ℕ) : ℕ := 
  (dimes * 10) + (quarters * 25) + (nickels * 5)

noncomputable def percent_in_quarters (quarters total_value : ℕ) : ℚ := 
  (quarters * 25 : ℚ) / total_value * 100

theorem percent_value_quarters 
  (h_dimes : ℕ := 80) 
  (h_quarters : ℕ := 30) 
  (h_nickels : ℕ := 40) 
  (h_total_value := value_in_cents h_dimes h_quarters h_nickels) : 
  percent_in_quarters h_quarters h_total_value = 42.86 :=
by sorry

end percent_value_quarters_l207_207134


namespace sqrt_sum_eq_seven_l207_207077

theorem sqrt_sum_eq_seven (y : ℝ) (h : sqrt (64 - y^2) - sqrt (36 - y^2) = 4) : 
  sqrt (64 - y^2) + sqrt (36 - y^2) = 7 :=
sorry

end sqrt_sum_eq_seven_l207_207077


namespace largest_prime_factor_12321_l207_207651

theorem largest_prime_factor_12321 : ∃ p, prime p ∧ (∀ q, prime q ∧ q ∣ 12321 → q ≤ p) ∧ p = 19 :=
by {
  sorry
}

end largest_prime_factor_12321_l207_207651


namespace work_rates_man_alone_6_days_l207_207173

variable (M W B : ℝ)

-- Define the conditions
def BoyWorkRate := B = 1 / 18
def WomanWorkRate := W = 1 / 36
def CombinedWorkRate := M + W + B = 1 / 4

-- Translate the problem into a Lean theorem statement
theorem work_rates_man_alone_6_days (BoyWorkRate : BoyWorkRate) (WomanWorkRate : WomanWorkRate) (CombinedWorkRate : CombinedWorkRate) : M = 1 / 6 :=
  by
  -- Proof is omitted
  sorry

end work_rates_man_alone_6_days_l207_207173


namespace age_of_b_l207_207080

variable (A B C : ℕ)

-- Conditions
def avg_abc : Prop := A + B + C = 78
def avg_ac : Prop := A + C = 58

-- Question: Prove that B = 20
theorem age_of_b (h1 : avg_abc A B C) (h2 : avg_ac A C) : B = 20 := 
by sorry

end age_of_b_l207_207080


namespace probability_unique_rolls_l207_207664

theorem probability_unique_rolls :
  let num_people := 5 in
  let die_faces := 6 in
  let total_ways := die_faces ^ num_people in
  let successful_ways := die_faces * (die_faces - 1) * (die_faces - 2) * (die_faces - 3) * (die_faces - 4) in
  let probability := (successful_ways : ℚ) / (total_ways : ℚ) in
  probability = 5 / 54 :=
by 
  sorry

end probability_unique_rolls_l207_207664


namespace committee_locks_keys_l207_207174

theorem committee_locks_keys :
  ∃ locks keys : ℕ,
  (locks = 126) ∧ (keys = 504) :=
begin
  -- Let the committee be composed of 9 professors
  let committee : set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9},

  -- Define the condition that at least 6 professors are needed to open the safe
  let min_professors := 6,

  -- Define the number of locks as the number of 5-combinations of committee members
  let num_locks := nat.choose 9 5,

  -- Define the total number of keys as four times the number of locks
  let total_keys := 4 * num_locks,

  -- Assert the number of locks and keys
  use num_locks,
  use total_keys,

  split,
  { -- Provide the number of locks
    rw [nat.choose_eq_factorial_div_factorial (9 - 5) 5],
    simp,
  },
  { -- Provide the total number of keys
    rw [mul_comm, nat.choose_eq_factorial_div_factorial (9 - 5) 5],
    simp,
  }
end

end committee_locks_keys_l207_207174


namespace solve_for_x_l207_207867

theorem solve_for_x (x : ℝ) (h : (x - 5)^4 = (1 / 16)⁻¹) : x = 7 :=
by
  sorry

end solve_for_x_l207_207867


namespace FE_tangent_to_Γ_l207_207824

variables {A B C D E F G H : Type}
variables [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D]
variables [DecidableEq E] [DecidableEq F] [DecidableEq G] [DecidableEq H]

-- Assume cyclic quadrilateral and intersection points
variable (ABCD_cyclic : CyclicQuadrilateral A B C D)
variable (E_def : intersection (line_through A C) (line_through B D) = E)
variable (F_def : intersection (line_through A D) (line_through B C) = F)

-- Assume midpoints
variable (G_def : midpoint A B = G)
variable (H_def : midpoint C D = H)

-- Circumcircle of triangle EGH
variable (Γ : Circle (triangle E G H))

-- Tangency proof statement
theorem FE_tangent_to_Γ :
  TangentLine (line_through F E) Γ :=
sorry

end FE_tangent_to_Γ_l207_207824


namespace fifth_power_last_digit_l207_207428

theorem fifth_power_last_digit (n : ℕ) : 
  (n % 10)^5 % 10 = n % 10 :=
by sorry

end fifth_power_last_digit_l207_207428


namespace trader_sold_bags_l207_207186

-- Define the conditions as constants
def initial_bags : ℕ := 55
def restocked_bags : ℕ := 132
def current_bags : ℕ := 164

-- Define a function to calculate the number of bags sold
def bags_sold (initial restocked current : ℕ) : ℕ :=
  initial + restocked - current

-- Statement of the proof problem
theorem trader_sold_bags : bags_sold initial_bags restocked_bags current_bags = 23 :=
by
  -- Proof is omitted
  sorry

end trader_sold_bags_l207_207186


namespace solve_for_x_l207_207866

theorem solve_for_x (x : ℝ) (h : (x - 5)^4 = (1 / 16)⁻¹) : x = 7 :=
by
  sorry

end solve_for_x_l207_207866


namespace number_of_pairs_l207_207671

theorem number_of_pairs (H : ∀ x y : ℕ , 0 < x → 0 < y → x < y → 2 * x * y / (x + y) = 4 ^ 15) :
  ∃ n : ℕ, n = 29 :=
by
  sorry

end number_of_pairs_l207_207671


namespace simplest_quadratic_radical_l207_207129

theorem simplest_quadratic_radical :
  ∀ (A B C D : ℝ),
  A = 1 →
  B = Real.sqrt 7 →
  C = Real.sqrt 12 →
  D = 1 / Real.sqrt 13 →
  ∃ simplest, simplest = B :=
by
  assume A B C D hA hB hC hD
  let simplest := B
  have h_simplest : simplest = B := by sorry
  exact ⟨simplest, h_simplest⟩

end simplest_quadratic_radical_l207_207129


namespace general_formula_of_sequence_a_sum_of_b_is_T_min_value_of_lambda_l207_207305

noncomputable def sequence_a (n : ℕ) : ℤ :=
if n = 1 then -3 else 2 * (n : ℤ) - 5

noncomputable def sequence_b (n : ℕ) : ℤ :=
2 ^ (sequence_a n) + 1

noncomputable def sum_S (n : ℕ) : ℤ :=
(n : ℤ) ^ 2 - 4 * (n : ℤ)

noncomputable def sum_T (n : ℕ) : ℤ :=
(1 / 24) * (4 ^ (n : ℤ) - 1) + (n : ℤ)

def min_lambda (n : ℕ) (a : ℕ → ℤ) : ℚ :=
∑ i in Finset.range n, 1 / (a i * a (i + 1))

theorem general_formula_of_sequence_a :
  ∀ n : ℕ, n > 0 → sequence_a n = if n = 1 then -3 else 2 * (n : ℤ) - 5 := sorry

theorem sum_of_b_is_T :
  ∀ n : ℕ, sum_T n = ∑ i in Finset.range n, sequence_b i := sorry

theorem min_value_of_lambda :
  ∀ n : ℕ, min_lambda n sequence_a ≤ (1 : ℚ) / 3 := sorry

end general_formula_of_sequence_a_sum_of_b_is_T_min_value_of_lambda_l207_207305


namespace form_five_squares_l207_207853

-- The conditions of the problem as premises
variables (initial_configuration : Set (ℕ × ℕ))               -- Initial positions of 12 matchsticks
          (final_configuration : Set (ℕ × ℕ))                 -- Final positions of matchsticks to form 5 squares
          (fixed_matchsticks : Set (ℕ × ℕ))                    -- Positions of 6 fixed matchsticks
          (movable_matchsticks : Set (ℕ × ℕ))                 -- Positions of 6 movable matchsticks

-- Condition to avoid duplication or free ends
variables (no_duplication : Prop)
          (no_free_ends : Prop)

-- Proof statement
theorem form_five_squares : ∃ rearranged_configuration, 
  rearranged_configuration = final_configuration ∧
  initial_configuration = fixed_matchsticks ∪ movable_matchsticks ∧
  no_duplication ∧
  no_free_ends :=
sorry -- Proof omitted.

end form_five_squares_l207_207853


namespace relationship_between_fx1_fx2_l207_207226

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := derivative f

theorem relationship_between_fx1_fx2
  (h_even : ∀ x : ℝ, f (x + 1) = f (-(x + 1)))
  (h_derivative : ∀ x : ℝ, (x - 1) * f' x < 0)
  (x1 x2 : ℝ)
  (h1 : x1 < x2)
  (h2 : x1 + x2 > 2) :
  f x1 > f x2 :=
sorry

end relationship_between_fx1_fx2_l207_207226


namespace range_of_a_iff_l207_207731

def cubic_inequality (x : ℝ) : Prop := x^3 + 3 * x^2 - x - 3 > 0

def quadratic_inequality (x a : ℝ) : Prop := x^2 - 2 * a * x - 1 ≤ 0

def integer_solution_condition (x : ℤ) (a : ℝ) : Prop := 
  x^3 + 3 * x^2 - x - 3 > 0 ∧ x^2 - 2 * a * x - 1 ≤ 0

def range_of_a (a : ℝ) : Prop := (3 / 4 : ℝ) ≤ a ∧ a < (4 / 3 : ℝ)

theorem range_of_a_iff : 
  (∃ x : ℤ, integer_solution_condition x a) ↔ range_of_a a := 
sorry

end range_of_a_iff_l207_207731


namespace true_proposition_l207_207148

theorem true_proposition:
  (∀ x : ℝ, 2^x > 0) ∧ ¬("x > 0" is a sufficient but not necessary condition for "x > 2") := 
begin
  sorry
end

end true_proposition_l207_207148


namespace sqrt_fraction_simplification_l207_207128

theorem sqrt_fraction_simplification (x : ℝ) (h : x < 0) : 
  sqrt (x / (1 - (x - 1) / x)) = -x :=
by
  sorry

end sqrt_fraction_simplification_l207_207128


namespace garden_area_increase_l207_207963

-- Definitions corresponding to the conditions
def length := 40
def width := 20
def original_perimeter := 2 * (length + width)

-- Definition of the correct answer calculation
def original_area := length * width
def side_length := original_perimeter / 4
def new_area := side_length * side_length
def area_increase := new_area - original_area

-- The statement to be proven
theorem garden_area_increase : area_increase = 100 :=
by sorry

end garden_area_increase_l207_207963


namespace jogger_speed_is_9_l207_207957

noncomputable def train_speed := 45 -- in kmph
noncomputable def gap := 240 -- in meters
noncomputable def train_length := 120 -- in meters
noncomputable def time_to_pass := 36 -- in seconds

noncomputable def jogger_speed : ℝ :=
  let relative_speed_mps := (gap + train_length) / time_to_pass in
  (relative_speed_mps * 18) / 5

theorem jogger_speed_is_9 : jogger_speed = 9 := by
  sorry

end jogger_speed_is_9_l207_207957


namespace range_of_m_l207_207840

noncomputable def ellipse_hyperbola_condition (m : ℝ) : Prop :=
  2 * m + 1 > 6 ∧ 2 * m - 1 < 6

theorem range_of_m (m : ℝ) (h : ellipse_hyperbola_condition m) : (5 / 2 < m) ∧ (m < 7 / 2) :=
begin
  cases h with h1 h2,
  split;
  { linarith },
end

end range_of_m_l207_207840


namespace A_has_winning_strategy_if_n_equals_9_l207_207481

noncomputable def has_winning_strategy_for_A (n : ℕ) := 
  ∃ (strategy : (ℕ → ℕ)), 
    (∀ (m : ℕ), 
      (1 ≤ strategy m ∧ strategy m ≤ 3) ∧ 
      (m ≤ n) → 
      strategy m ∈ {1, 2, 3}) 

theorem A_has_winning_strategy_if_n_equals_9 :
  has_winning_strategy_for_A 9 :=
sorry

end A_has_winning_strategy_if_n_equals_9_l207_207481


namespace tax_collected_from_village_l207_207243

-- Definitions according to the conditions in the problem
def MrWillamTax : ℝ := 500
def MrWillamPercentage : ℝ := 0.21701388888888893

-- The theorem to prove the total tax collected
theorem tax_collected_from_village : ∃ (total_collected : ℝ), MrWillamPercentage * total_collected = MrWillamTax ∧ total_collected = 2303.7037037037035 :=
sorry

end tax_collected_from_village_l207_207243


namespace negation_exists_negation_proposition_l207_207093

theorem negation_exists (P : ℝ → Prop) :
  (∃ x : ℝ, P x) ↔ ¬ (∀ x : ℝ, ¬ P x) :=
by sorry

theorem negation_proposition :
  (¬ ∃ x : ℝ, x^2 + 1 < 0) ↔ (∀ x : ℝ, x^2 + 1 ≥ 0) :=
by sorry

end negation_exists_negation_proposition_l207_207093


namespace Q_is_circumcenter_of_CDE_l207_207029

theorem Q_is_circumcenter_of_CDE {l1 l2 : Line} {O O1 O2 : Circle} 
  (M N : Point) (H1 : IsTangent O l1 M) (H2 : IsTangent O l2 N)
  (E : Point) (H3 : IsExternalTangent O1 O2 E)
  (A C : Point) (H4 : IsTangent O1 l1 A) (H5 : IsTangent O1 O C)
  (B D : Point) (H6 : IsTangent O2 l2 B) (H7 : IsTangent O2 O D)
  (Q : Point) (H8 : Intersects (Line.through A D) (Line.through B C) Q)
  (H9 : Parallel l1 l2) :
  IsCircumcenter Q (Triangle.mk C D E) :=
by
  sorry

end Q_is_circumcenter_of_CDE_l207_207029


namespace perimeter_triangle_series_limit_l207_207554

noncomputable def perimeter_sum_limit (a b : ℝ) : ℝ :=
  let h := real.sqrt (a^2 + b^2)
  let P n := (a + b + h) / 2^(n-1)
  series_sum (λ n, P n) -- Sum of the series of perimeters of all triangles.

theorem perimeter_triangle_series_limit (a b : ℝ) : 
  perimeter_sum_limit a b = 2 * (a + b + real.sqrt (a^2 + b^2)) := by
  sorry  -- The proof can be filled in here

end perimeter_triangle_series_limit_l207_207554


namespace abigail_boxes_proof_l207_207191

def total_cookies_per_person (grayson_cookies olivia_cookies abigail_cookies : ℕ) : ℕ :=
  grayson_cookies + olivia_cookies + abigail_cookies

def grayson_boxes := 3 / 4
def olivia_boxes := 3
def cookies_per_box := 48
def total_cookies := 276

theorem abigail_boxes_proof :
  let grayson_cookies := (grayson_boxes * cookies_per_box).toInt
  let olivia_cookies := (olivia_boxes * cookies_per_box)
  let collected_cookies := total_cookies_per_person grayson_cookies olivia_cookies
  let abigail_cookies := total_cookies - collected_cookies
  (abigail_cookies / cookies_per_box) = 2 :=
by
  sorry

end abigail_boxes_proof_l207_207191


namespace domain_of_f_l207_207880

noncomputable def f (x : ℝ) : ℝ := (sqrt (2 - x)) / (Real.log (x + 1))

theorem domain_of_f :
  ∀ x : ℝ, x > -1 ∧ x ≤ 2 ∧ x ≠ 0 ↔ x ∈ Set.Ioo (-1:ℝ) 0 ∪ Set.Ioc 0 2 := 
sorry

end domain_of_f_l207_207880


namespace saree_price_calculation_scarf_price_calculation_l207_207479

noncomputable def final_sale_price (initial_price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl (λ price discount, price - (price * discount / 100)) initial_price

def saree_initial_price : ℝ := 500
def saree_discounts : List ℝ := [10, 5, 8]
def saree_sale_price : ℝ := 393.30

def scarf_initial_price : ℝ := 350
def scarf_discounts : List ℝ := [12, 7, 5]
def scarf_sale_price : ℝ := 272.12

theorem saree_price_calculation : 
  final_sale_price saree_initial_price saree_discounts = saree_sale_price :=
by
  sorry

theorem scarf_price_calculation : 
  final_sale_price scarf_initial_price scarf_discounts = scarf_sale_price :=
by
  sorry

end saree_price_calculation_scarf_price_calculation_l207_207479


namespace Kylie_US_coins_left_l207_207022

-- Define the given conditions
def initial_US_coins : ℝ := 15
def Euro_coins : ℝ := 13
def Canadian_coins : ℝ := 8
def US_coins_given_to_Laura : ℝ := 21
def Euro_to_US_rate : ℝ := 1.18
def Canadian_to_US_rate : ℝ := 0.78

-- Define the conversions
def Euro_to_US : ℝ := Euro_coins * Euro_to_US_rate
def Canadian_to_US : ℝ := Canadian_coins * Canadian_to_US_rate
def total_US_before_giving : ℝ := initial_US_coins + Euro_to_US + Canadian_to_US
def US_left_with : ℝ := total_US_before_giving - US_coins_given_to_Laura

-- Statement of the problem to be proven
theorem Kylie_US_coins_left :
  US_left_with = 15.58 := by
  sorry

end Kylie_US_coins_left_l207_207022


namespace find_x_y_sum_l207_207265

theorem find_x_y_sum (x y : ℝ) (h1 : 4^x = 16^(y + 2)) (h2 : 27^y = 9^(x - 6)) :
  x + y = 16 :=
by
  sorry

end find_x_y_sum_l207_207265


namespace ian_remaining_money_l207_207342

def colin_payment := 20
def helen_payment := 2 * colin_payment
def benedict_payment := helen_payment / 2
def emma_initial_debt := 15
def emma_interest_rate := 0.10
def emma_total_payment := emma_initial_debt + emma_interest_rate * emma_initial_debt
def ava_initial_debt := 10
def ava_forgiven_debt := ava_initial_debt * 0.25
def ava_total_payment := ava_initial_debt - ava_forgiven_debt
def ian_total_winnings := 100
def total_debts := colin_payment + helen_payment + benedict_payment + emma_total_payment + ava_total_payment

theorem ian_remaining_money : ian_total_winnings - total_debts = -4 := by
  sorry

end ian_remaining_money_l207_207342


namespace more_tvs_sold_l207_207411

variable (T x : ℕ)

theorem more_tvs_sold (h1 : T + x = 327) (h2 : T + 3 * x = 477) : x = 75 := by
  sorry

end more_tvs_sold_l207_207411


namespace volume_frustum_fraction_l207_207972

-- Define the base edge and initial altitude of the pyramid.
def base_edge := 32 -- in inches
def altitude_original := 1 -- in feet

-- Define the fractional part representing the altitude of the smaller pyramid.
def altitude_fraction := 1/4

-- Define the volume of the original pyramid being V.
noncomputable def volume_original : ℝ := (1/3) * (base_edge ^ 2) * altitude_original

-- Define the volume of the smaller pyramid being removed.
noncomputable def volume_smaller : ℝ := (1/3) * ((altitude_fraction * base_edge) ^ 2) * (altitude_fraction * altitude_original)

-- We now state the proof
theorem volume_frustum_fraction : 
  (volume_original - volume_smaller) / volume_original = 63/64 :=
by
  sorry

end volume_frustum_fraction_l207_207972


namespace area_triangle_l207_207856

-- Define the points of the triangle and the given extensions
variables {A B C A1 B1 C1 : Type}
variables [Point A] [Point B] [Point C]
variables [Point A1] [Point B1] [Point C1]

-- Define the conditions
axiom AB_extension : (A B1 : vector) = 2 * (A B : vector)
axiom BC_extension : (B C1 : vector) = 2 * (B C : vector)
axiom CA_extension : (C A1 : vector) = 2 * (C A : vector)
variable {area_ABC : ℝ} -- The area of triangle ABC

-- Define the areas of the triangles in terms of the area of ABC
def area_ABC := A → B → C → ℝ
def area_A1B1C1 := A1 → B1 → C1 → ℝ

theorem area_triangle {S : ℝ} (h : area_ABC S) :
  area_A1B1C1 = 7 * S :=
sorry

end area_triangle_l207_207856


namespace common_point_of_sampling_methods_l207_207082

-- Define the properties of each sampling method as conditions
def simple_random_sampling (population : Type) : Prop :=
  -- Involves drawing each individual one by one from a population
  ∃ draw_one_by_one : population → bool, 
    ∀ p : population, draw_one_by_one p = true

def systematic_sampling (population : Type) : Prop :=
  -- Involves dividing the population into several parts according to certain rules
  ∃ divide_into_parts : list population → list (list population), 
    ∀ parts : list (list population), divide_into_parts parts = parts

def stratified_sampling (population : Type) : Prop :=
  -- Involves dividing the population into several layers before sampling
  ∃ divide_into_layers : list population → list (list population),
    ∀ layers : list (list population), divide_into_layers layers = layers

-- Prove that the chance of each individual being drawn during the sampling process is the same
theorem common_point_of_sampling_methods (population : Type) :
  simple_random_sampling population ∧ systematic_sampling population ∧ stratified_sampling population →
  ∃ each_individual_same_chance : population → bool,
    ∀ p : population, each_individual_same_chance p = true :=
begin
  intros h,
  sorry
end

end common_point_of_sampling_methods_l207_207082


namespace triangle_side_ratio_l207_207039

/-
Given:
- p, q, r are real numbers
- The midpoints of the sides of triangle ABC are known:
    - Midpoint of BC is (p, 0, 1)
    - Midpoint of AC is (0, q, 1)
    - Midpoint of AB is (0, 0, r)

Prove that the ratio of the sum of the squares of the sides to p^2 + q^2 + r^2 is
12 - 4r(2 - r) / (p^2 + q^2 + r^2).
-/
theorem triangle_side_ratio (p q r : ℝ) (A B C : ℝ×ℝ×ℝ)
  (hBC : (p, 0, 1) = ((A.1 + B.1) / 2, (A.2 + B.2) / 2, (A.3 + B.3) / 2))
  (hAC : (0, q, 1) = ((A.1 + C.1) / 2, (A.2 + C.2) / 2, (A.3 + C.3) / 2))
  (hAB : (0, 0, r) = ((B.1 + C.1) / 2, (B.2 + C.2) / 2, (B.3 + C.3) / 2)) :
  (AB Sqrd + AC Sqrd + BC Sqrd) / (p^2 + q^2 + r^2) = 12 - (4*r*(2 - r)) / (p^2 + q^2 + r^2) :=
by
  sorry

end triangle_side_ratio_l207_207039


namespace polynomial_roots_l207_207427

theorem polynomial_roots :
  ∃ (x : ℝ), x^3 - 3 * tan (π / 12) * x^2 - 3 * x + tan (π / 12) = 0 ∧ 
  ( ∃ (u v w : ℝ), 
      (u = tan (π / 36) ∧ v = tan (13 * π / 36) ∧ w = tan (25 * π / 36)) ∧
      (u^3 - 3 * tan (π / 12) * u^2 - 3 * u + tan (π / 12) = 0) ∧ 
      (v^3 - 3 * tan (π / 12) * v^2 - 3 * v + tan (π / 12) = 0) ∧ 
      (w^3 - 3 * tan (π / 12) * w^2 - 3 * w + tan (π / 12) = 0) 
    )
sorry

end polynomial_roots_l207_207427


namespace beta_max_success_ratio_l207_207364

-- Define Beta's score conditions
variables (a b c d : ℕ)
def beta_score_conditions :=
  (0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) ∧
  (a * 25 < b * 9) ∧
  (c * 25 < d * 17) ∧
  (b + d = 600)

-- Define Beta's success ratio
def beta_success_ratio :=
  (a + c) / 600

theorem beta_max_success_ratio :
  beta_score_conditions a b c d →
  beta_success_ratio a c ≤ 407 / 600 :=
sorry

end beta_max_success_ratio_l207_207364


namespace polynomial_degree_sum_of_coeffs_l207_207850

def P (x : ℤ) (n : ℤ) : ℤ :=
  (x^2 - x + 1)^n - (x^2 - x + 2)^n + (1 + x)^n + (2 - x)^n

variable {n : ℤ}
variable (hn : n > 2)

theorem polynomial_degree (P : ℤ → ℤ) (n : ℤ) (hn : n > 2) : 
  degree (λ x : ℤ, (x^2 - x + 1)^n - (x^2 - x + 2)^n + (1 + x)^n + (2 - x)^n) = 2 * n - 2 := 
sorry

theorem sum_of_coeffs (P : ℤ → ℤ) (n : ℤ) (a : ℕ → ℤ) (hn : n > 2) :
  (∀ x : ℤ, P x = ∑ i in finset.range (2n - 1), a i * x^i) → 
  ∑ k in finset.Ico 2 (2n-1), a k = 0 := 
sorry

end polynomial_degree_sum_of_coeffs_l207_207850


namespace area_of_intersection_l207_207779

-- Define the region M
def in_region_M (x y : ℝ) : Prop :=
  y ≥ 0 ∧ y ≤ x ∧ y ≤ 2 - x

-- Define the region N as it changes with t
def in_region_N (t x : ℝ) : Prop :=
  t ≤ x ∧ x ≤ t + 1 ∧ 0 ≤ t ∧ t ≤ 1

-- Define the function f(t) which represents the common area of M and N
noncomputable def f (t : ℝ) : ℝ :=
  -t^2 + t + 0.5

-- Prove that f(t) is correct given the above conditions
theorem area_of_intersection (t : ℝ) :
  (∀ x y : ℝ, in_region_M x y → in_region_N t x → y ≤ f t) →
  0 ≤ t ∧ t ≤ 1 →
  f t = -t^2 + t + 0.5 :=
by
  sorry

end area_of_intersection_l207_207779


namespace P_at_1_l207_207452

-- Definitions for the polynomial and conditions
def P (x : ℤ) : ℤ := a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4

axiom a_0_nonneg : a_0 ≥ 0
axiom a_0_less_100 : a_0 < 100

axiom a_1_nonneg : a_1 ≥ 0
axiom a_1_less_100 : a_1 < 100

axiom a_2_nonneg : a_2 ≥ 0
axiom a_2_less_100 : a_2 < 100

axiom a_3_nonneg : a_3 ≥ 0
axiom a_3_less_100 : a_3 < 100

axiom a_4_nonneg : a_4 ≥ 0
axiom a_4_less_100 : a_4 < 100

axiom P_at_10 : P 10 = 331633
axiom P_at_neg_10 : P (-10) = 273373

-- Proof statement
theorem P_at_1 : P 1 = 100 := by
  sorry

end P_at_1_l207_207452


namespace product_of_consecutive_natural_numbers_l207_207064

theorem product_of_consecutive_natural_numbers (n : ℕ) :
  let product := n * (n + 1) * (n + 2) * (n + 3) in
  (product % 1000 = 24)
  ∨ (product % 10 = 0 ∧ ((product / 10) % 100 % 4 = 0)) :=
by sorry

end product_of_consecutive_natural_numbers_l207_207064


namespace isosceles_O₁_O_O₂_equilateral_condition_O₁_O_O₂_l207_207015

-- Definition of triangle and circumcenters
structure Triangle :=
(A B C : Point)

structure Circumcenter (T : Triangle) :=
(O : Point) (is_circumcenter : ∀ P : Point, P ∈ T.A ∨ P ∈ T.B ∨ P ∈ T.C → dist P O = dist (opposite P {A, B, C}) O)

variables {A B C A' : Point}
variables (α : angle)

def angle_bisector (A : Point) (α : angle) (BC : line) : Point := A'

def circumcenter_ABC : Circumcenter ⟨A, B, C⟩ := ⟨O, sorry⟩
def circumcenter_ABA' : Circumcenter ⟨A, B, A'⟩ := ⟨O₁, sorry⟩
def circumcenter_ACA' : Circumcenter ⟨A, C, A'⟩ := ⟨O₂, sorry⟩

-- Statement to prove
theorem isosceles_O₁_O_O₂ 
  (angle_bisector_intersects : angle_bisector A α (line_through B C) = A')
  (O := circumcenter_ABC.O)
  (O₁ := circumcenter_ABA'.O)
  (O₂ := circumcenter_ACA'.O)
  : is_isosceles_triangle O O₁ O₂ := 
sorry

theorem equilateral_condition_O₁_O_O₂ 
  (angle_bisector_intersects : angle_bisector A α (line_through B C) = A')
  (O := circumcenter_ABC.O)
  (O₁ := circumcenter_ABA'.O)
  (O₂ := circumcenter_ACA'.O)
  : is_equilateral_triangle O O₁ O₂ ↔ α = 120 := 
sorry

end isosceles_O₁_O_O₂_equilateral_condition_O₁_O_O₂_l207_207015


namespace function_range_l207_207097

-- Define the function \( f(x) = 1 + 2\sqrt{x-1} \)
noncomputable def f (x : ℝ) : ℝ := 1 + 2 * real.sqrt (x - 1)

-- Define the range theorem
theorem function_range (x : ℝ) (hx : 1 ≤ x) : ∃ y, y = f x ∧ y ∈ set.Ici 1 :=
by
  sorry

end function_range_l207_207097


namespace smallest_four_digit_palindromic_prime_l207_207496

-- Definitions based on conditions
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def is_palindromic (n : ℕ) : Prop := n.to_string.reverse.to_nat = n
def is_prime (n : ℕ) : Prop := nat.prime n

-- Final theorem statement
theorem smallest_four_digit_palindromic_prime : ∃ n : ℕ, is_four_digit n ∧ is_palindromic n ∧ is_prime n ∧ ∀ m : ℕ, is_four_digit m ∧ is_palindromic m ∧ is_prime m → n ≤ m :=
  by
  existsi 1661
  -- Proof steps would go here
  sorry

end smallest_four_digit_palindromic_prime_l207_207496


namespace num_non_differentiable_points_l207_207225

def f (x : ℝ) : ℝ := max (Real.sin x) (Real.cos x)

theorem num_non_differentiable_points : 
  ∃ (S : Finset ℝ), (∀ x ∈ S, x ∈ Ioo (-2 * Real.pi) (2 * Real.pi) ∧ Real.sin x = Real.cos x) ∧ S.card = 4 := 
sorry

end num_non_differentiable_points_l207_207225


namespace sum_of_angles_is_990_l207_207608

noncomputable def z₁ : ℂ := 2 * complex.exp (complex.I * real.pi * (54 / 180))
noncomputable def z₂ : ℂ := 2 * complex.exp (complex.I * real.pi * (126 / 180))
noncomputable def z₃ : ℂ := 2 * complex.exp (complex.I * real.pi * (198 / 180))
noncomputable def z₄ : ℂ := 2 * complex.exp (complex.I * real.pi * (270 / 180))
noncomputable def z₅ : ℂ := 2 * complex.exp (complex.I * real.pi * (342 / 180))

theorem sum_of_angles_is_990 :
  (54 + 126 + 198 + 270 + 342 : ℝ) = 990 :=
by
  sorry

end sum_of_angles_is_990_l207_207608


namespace largest_root_in_range_l207_207221

-- Define the conditions for the equation parameters
variables (a0 a1 a2 : ℝ)
-- Define the conditions for the absolute value constraints
variables (h0 : |a0| < 2) (h1 : |a1| < 2) (h2 : |a2| < 2)

-- Define the equation
def cubic_equation (x : ℝ) : ℝ := x^3 + a2 * x^2 + a1 * x + a0

-- Define the property we want to prove about the largest positive root r
theorem largest_root_in_range :
  ∃ r > 0, (∃ x, cubic_equation a0 a1 a2 x = 0 ∧ r = x) ∧ (5 / 2 < r ∧ r < 3) :=
by sorry

end largest_root_in_range_l207_207221


namespace profit_percentage_is_correct_l207_207565

def CP : ℝ := 38
def SP : ℝ := 50
def discount_rate : ℝ := 0.05

noncomputable def profit_percentage (CP SP : ℝ) : ℝ :=
  let MP := SP / (1 - discount_rate) in
  let profit := SP - CP in
  (profit / CP) * 100

theorem profit_percentage_is_correct
  (hCP : CP = 38)
  (hSP : SP = 50)
  (h_discount : discount_rate = 0.05) :
  profit_percentage CP SP = 31.58 := by
  sorry

end profit_percentage_is_correct_l207_207565


namespace angle_BAC_of_circumscribed_triangle_l207_207530

theorem angle_BAC_of_circumscribed_triangle :
  ∀ {O A B C : Type} [distinct_points : ∀A B C, A ≠ B → B ≠ C → C ≠ A]
  (α_120 : ∠ AOB = 120)
  (β_90 : ∠ BOC = 90)
  (circumscribed : circle O ∼= \triangle ABC),
  ∠ BAC = 45 :=
by sorry

end angle_BAC_of_circumscribed_triangle_l207_207530


namespace stewart_farm_sheep_count_l207_207098

theorem stewart_farm_sheep_count
  (ratio : ℕ → ℕ → Prop)
  (S H : ℕ)
  (ratio_S_H : ratio S H)
  (one_sheep_seven_horses : ratio 1 7)
  (food_per_horse : ℕ)
  (total_food : ℕ)
  (food_per_horse_val : food_per_horse = 230)
  (total_food_val : total_food = 12880)
  (calc_horses : H = total_food / food_per_horse)
  (calc_sheep : S = H / 7) :
  S = 8 :=
by {
  /- Given the conditions, we need to show that S = 8 -/
  sorry
}

end stewart_farm_sheep_count_l207_207098


namespace probability_of_winning_correct_l207_207361

noncomputable def probability_of_winning (P_L : ℚ) (P_T : ℚ) : ℚ :=
  1 - (P_L + P_T)

theorem probability_of_winning_correct :
  probability_of_winning (3/7) (2/21) = 10/21 :=
by
  sorry

end probability_of_winning_correct_l207_207361


namespace x_squared_plus_y_squared_l207_207350

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : x - y = 12) (h2 : x * y = 9) : x^2 + y^2 = 162 :=
by
  sorry

end x_squared_plus_y_squared_l207_207350


namespace jake_has_8_more_peaches_l207_207019

variables (Jake Steven Jill Tom : ℕ)

def initial_peaches (Jill_initial : ℕ) (Steven_offset : ℕ) (Jake_offset : ℕ) (Tom_factor : ℕ) : Prop :=
  Steven = Jill_initial + Steven_offset ∧
  Jake = Steven - Jake_offset ∧
  Tom = Tom_factor * Steven

def peaches_after_exchange (Jake_initial Jill_initial : ℕ) (Jake_add Jill_add : ℕ) : Prop :=
  Jake + Jake_add - (Jill + Jill_add) = 8

theorem jake_has_8_more_peaches :
  initial_peaches Jake Steven Jill 87 18 5 2 →
  peaches_after_exchange Jake 100 Jill 87 5 10 :=
by
  intros,
  sorry

end jake_has_8_more_peaches_l207_207019


namespace equal_segments_on_rays_l207_207059

-- Define the structure of a triangle
structure Triangle (A B C : Type) :=
(point : A → B → C → Prop)

-- Define collinearity for three points
def collinear {A} [affine_space ℝ A] (P Q R : A) : Prop :=
affine_combination.line_ℝ.convex_space ℝ P Q R

-- Define the main problem statement
theorem equal_segments_on_rays (A B C P M N : Type)
  [affine_space ℝ A] [affine_space ℝ B] [affine_space ℝ C] [affine_space ℝ P]
  [affine_space ℝ M] [affine_space ℝ N] (T : Triangle A B C)
  (hP_on_AB : collinear P A B)
  (hM_on_CA : collinear M C A)
  (hN_on_CB : collinear N C B) :
  ∃ (line_PM: P → M) (line_PN: P → N), distance M A = distance N B :=
begin
  sorry
end

end equal_segments_on_rays_l207_207059


namespace isosceles_triangle_BC_length_l207_207000

theorem isosceles_triangle_BC_length :
  ∀ (A B C H : Type) 
  (d : dist A C = 5)
  (altitude : ∀ (H : Type), collinear A C H ∧ right_angle B H C ∧ segment_length Ah Hc = 2 * segment_length Hc C),
  dist B C = 5 * sqrt 6 / 3 :=
by
  sorry

end isosceles_triangle_BC_length_l207_207000


namespace Mark_soup_cans_l207_207845

theorem Mark_soup_cans 
    (bread_cost : ℕ) 
    (cereal_cost : ℕ) 
    (milk_cost : ℕ) 
    (total_money : ℕ) 
    (soup_cost_per_can : ℕ)
    (remaining_money_for_soup : ℕ) 
    (number_of_soup_cans : ℕ) : 
    bread_cost = 2 * 5 →
    cereal_cost = 2 * 3 →
    milk_cost = 2 * 4 →
    total_money = 4 * 10 →
    soup_cost_per_can = 2 →
    remaining_money_for_soup = total_money - (bread_cost + cereal_cost + milk_cost) →
    number_of_soup_cans = remaining_money_for_soup / soup_cost_per_can →
    number_of_soup_cans = 8 :=
by
  intros,
  sorry

end Mark_soup_cans_l207_207845


namespace number_of_houses_with_neither_feature_l207_207846

variable (T G P B : ℕ)

theorem number_of_houses_with_neither_feature 
  (hT : T = 90)
  (hG : G = 50)
  (hP : P = 40)
  (hB : B = 35) : 
  T - (G + P - B) = 35 := 
    by
      sorry

end number_of_houses_with_neither_feature_l207_207846


namespace average_temperature_problem_l207_207451

variable {T W Th F : ℝ}

theorem average_temperature_problem (h1 : (W + Th + 44) / 3 = 34) (h2 : T = 38) : 
  (T + W + Th) / 3 = 32 := by
  sorry

end average_temperature_problem_l207_207451


namespace find_f_neg15_l207_207269

def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = f x

def log2_function_defined_on_positive (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, x > 0 → f x = Real.log2 (x + 1)

theorem find_f_neg15 (f : ℝ → ℝ) 
  (h1 : is_even_function f) 
  (h2 : log2_function_defined_on_positive f) :
  f (-15) = 4 :=
sorry

end find_f_neg15_l207_207269


namespace trader_sold_bags_l207_207185

-- Define the conditions as constants
def initial_bags : ℕ := 55
def restocked_bags : ℕ := 132
def current_bags : ℕ := 164

-- Define a function to calculate the number of bags sold
def bags_sold (initial restocked current : ℕ) : ℕ :=
  initial + restocked - current

-- Statement of the proof problem
theorem trader_sold_bags : bags_sold initial_bags restocked_bags current_bags = 23 :=
by
  -- Proof is omitted
  sorry

end trader_sold_bags_l207_207185


namespace sum_first_9_zero_l207_207450

variable {a : ℕ → ℝ} -- a_1, a_2, ..., a_n is an arithmetic sequence with real numbers

-- Declaring the common difference
constant d : ℝ

-- Definitions based on the arithmetic sequence
def a_n (n : ℕ) := a 1 + (n - 1) * d

-- Given condition that a_2 + a_9 = a_6
axiom a2_a9_eq_a6 : a_n 2 + a_n 9 = a_n 6

-- Proposition to prove that the sum of the first 9 terms S_9 equals 0
theorem sum_first_9_zero : (∑ i in Finset.range 9, a_n (i + 1)) = 0 :=
sorry

end sum_first_9_zero_l207_207450


namespace trapezoid_LM_sqrt2_l207_207802

theorem trapezoid_LM_sqrt2 (K L M N P Q : Type*)
  (KM : ℝ)
  (KN MQ LM MP : ℝ)
  (h_KM : KM = 1)
  (h_KN_MQ : KN = MQ)
  (h_LM_MP : LM = MP) 
  (h_KP_1 : KN = 1) 
  (h_MQ_1 : MQ = 1) :
  LM = Real.sqrt 2 :=
by
  sorry

end trapezoid_LM_sqrt2_l207_207802


namespace arrange_plants_ways_l207_207207

theorem arrange_plants_ways :
  let basil_plants := 4
  let tomato_plants := 4
  let total_ways := (fact (basil_plants + 1)) * (fact tomato_plants)
  total_ways = 2880 :=
by
  sorry

end arrange_plants_ways_l207_207207


namespace trader_sold_23_bags_l207_207184

theorem trader_sold_23_bags
    (initial_stock : ℕ) (restocked : ℕ) (final_stock : ℕ) (x : ℕ)
    (h_initial : initial_stock = 55)
    (h_restocked : restocked = 132)
    (h_final : final_stock = 164)
    (h_equation : initial_stock - x + restocked = final_stock) :
    x = 23 :=
by
    -- Here will be the proof of the theorem
    sorry

end trader_sold_23_bags_l207_207184


namespace min_number_solutions_in_interval_l207_207519

noncomputable def f : ℝ → ℝ := sorry

theorem min_number_solutions_in_interval  (h_even : ∀ x : ℝ, f(-x) = f(x))
                                         (h_period : ∀ x : ℝ, f(x + 3) = f(x))
                                         (h_value : f 2 = 0) :
                                         ∃ (S : set ℝ), (∀ x ∈ S, f x = 0) ∧ S ⊆ set.Ioo 0 6 ∧ S.card = 4 :=
sorry

end min_number_solutions_in_interval_l207_207519


namespace solve_price_of_meat_l207_207004

def price_of_meat_per_ounce (x : ℕ) : Prop :=
  16 * x - 30 = 8 * x + 18

theorem solve_price_of_meat : ∃ x, price_of_meat_per_ounce x ∧ x = 6 :=
by
  sorry

end solve_price_of_meat_l207_207004


namespace new_selling_price_for_60_percent_profit_l207_207508

-- Definitions from the problem's conditions
variable (C : ℝ) -- The store's cost for the computer
variable (selling_price1 : ℝ := 2240) -- The given selling price which yields a 40% profit
variable (profit1 : ℝ := 0.40) -- The 40% profit

-- Definitions derived from the problem
def compute_cost (selling_price1 : ℝ) (profit1 : ℝ) : ℝ := 
  selling_price1 / (1 + profit1)

def compute_new_selling_price (C : ℝ) (profit2 : ℝ) : ℝ :=
  C * (1 + profit2)

-- Theorem stating the new selling price for 60% profit
theorem new_selling_price_for_60_percent_profit : 
  let C := compute_cost selling_price1 profit1 in
  compute_new_selling_price C 0.60 = 2560 := by
  sorry

end new_selling_price_for_60_percent_profit_l207_207508


namespace find_smallest_n_l207_207216

def vertices (n : ℤ) : ℤ × ℤ → ℤ × ℤ → ℤ × ℤ  :=
(n + i, 1), ((n + i)^2, 2n), ((n + i)^4, 4n^3 - 4n)

def triangle_area (n : ℤ) : ℤ :=
(1 / 2 * abs((n * 2n) + ((n^2 - 1 )* (4n^3 - 4n)) + ((n^4 - 2n^2 -3) * 1) - 
((1* n^2 -1)) -  ((2n) * (n^4 - 2n^2 - 3)) - (4n^3 - 4n)*n)

theorem find_smallest_n :
  ∃ n : ℤ, 0 < n ∧ (triangle_area n > 1000) := sorry

end find_smallest_n_l207_207216


namespace exists_positive_integer_n_l207_207516

theorem exists_positive_integer_n : ∃ n : ℕ, n = 10000 ∧ 1.001^n > 10 ∧ 0.999^n < 0.1 :=
by
  use 10000
  split
  · rfl
  · split
    · sorry
    · sorry

end exists_positive_integer_n_l207_207516


namespace decreasing_interval_ln_x_minus_x_squared_l207_207465

theorem decreasing_interval_ln_x_minus_x_squared :
  {x : ℝ | 0 < x ∧ (1 - 2 * x^2) / x < 0} = Ico (real.sqrt 2 / 2) (∞) := 
sorry

end decreasing_interval_ln_x_minus_x_squared_l207_207465


namespace cloth_ninth_day_l207_207876

variables (a₁ d : ℕ)

-- Conditions described in the problem
def condition1 := 7 * a₁ + 21 * d = 28
def condition2 := a₁ + d + a₁ + 4 * d + a₁ + 7 * d = 15

-- Conclusion we want to prove
def target := a₁ + 8 * d = 9

theorem cloth_ninth_day : condition1 ∧ condition2 → target := by
  sorry

end cloth_ninth_day_l207_207876


namespace triangle_perimeter_area_l207_207989

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def perimeter (A B C : ℝ × ℝ) : ℝ :=
  distance A B + distance B C + distance C A

def area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem triangle_perimeter_area :
  let A := (2, 3) 
  let B := (2, 9)
  let C := (6, 6)
  perimeter A B C = 16 ∧ area A B C = 12 :=
by
  let A := (2, 3)
  let B := (2, 9)
  let C := (6, 6)
  sorry

end triangle_perimeter_area_l207_207989


namespace word_count_with_vowel_l207_207740

theorem word_count_with_vowel : 
  let letters := ['A', 'B', 'C', 'D', 'E', 'F'],
      vowels := {'A', 'E'},
      len := 5
  in 
  let total_words := (letters.length)^len,
      total_vowelless_words := (letters.length - vowels.size)^len
  in 
  (total_words - total_vowelless_words = 6752) :=
by 
  let letters := ['A', 'B', 'C', 'D', 'E', 'F'];
  let vowels := {'A', 'E'};
  let len := 5;
  let total_words := letters.length ^ len;
  let total_vowelless_words := (letters.length - vowels.size) ^ len;
  have h1 : total_words = 6 ^ 5, by sorry,
  have h2 : total_vowelless_words = 4 ^ 5, by sorry,
  have h3 : 7776 - 1024 = 6752, from rfl,
  show total_words - total_vowelless_words = 6752, by 
  { rw [h1, h2, h3] }

end word_count_with_vowel_l207_207740


namespace oc_expression_l207_207162

theorem oc_expression
  (θ : ℝ)
  (s : ℝ)
  (c : ℝ)
  (r : ℝ)
  (sin_def : s = Real.sin θ)
  (cos_def : c = Real.cos θ)
  (radius_one : ∀ (O A : Point), dist O A = 1)
  (tangent_segment : ∀ (O A B : Point), is_tangent_segment AB OA A)
  (angle_conditions : ∀ (O A B C : Point), ∠ O B A = θ ∧ segment_bisector BC (∠ O B A) C) : 
  OC = (r * (c ^ 2 / 4)) / (s + (c ^ 2 / 4)) :=
by
  sorry

end oc_expression_l207_207162


namespace parallel_vectors_determine_t_l207_207841

theorem parallel_vectors_determine_t (t : ℝ) (h : (t, -6) = (k * -3, k * 2)) : t = 9 :=
by
  sorry

end parallel_vectors_determine_t_l207_207841


namespace factorization1_factorization2_l207_207617

-- Definitions for the first problem
def expr1 (x : ℝ) := 3 * x^2 - 12
def factorized_form1 (x : ℝ) := 3 * (x + 2) * (x - 2)

-- Theorem for the first problem
theorem factorization1 (x : ℝ) : expr1 x = factorized_form1 x :=
  sorry

-- Definitions for the second problem
def expr2 (a x y : ℝ) := a * x^2 - 4 * a * x * y + 4 * a * y^2
def factorized_form2 (a x y : ℝ) := a * (x - 2 * y) * (x - 2 * y)

-- Theorem for the second problem
theorem factorization2 (a x y : ℝ) : expr2 a x y = factorized_form2 a x y :=
  sorry

end factorization1_factorization2_l207_207617


namespace annual_interest_rate_is_correct_l207_207480

-- Definitions of the conditions
def true_discount : ℚ := 210
def bill_amount : ℚ := 1960
def time_period_years : ℚ := 3 / 4

-- The present value of the bill
def present_value : ℚ := bill_amount - true_discount

-- The formula for simple interest given principal, rate, and time
def simple_interest (P R T : ℚ) : ℚ :=
  P * R * T / 100

-- Proof statement
theorem annual_interest_rate_is_correct : 
  ∃ (R : ℚ), simple_interest present_value R time_period_years = true_discount ∧ R = 16 :=
by
  use 16
  sorry

end annual_interest_rate_is_correct_l207_207480


namespace proposition_2_proposition_3_proposition_4_l207_207734

-- Definitions of the lines and planes
variable {a b : Line} (α β : Plane)

-- Assuming the distinct lines and planes
axiom distinct_lines : a ≠ b
axiom distinct_planes : α ≠ β

-- Proposition ②: If a is perpendicular to α and a is perpendicular to β, then α is parallel to β.
theorem proposition_2 (h₁ : a ⟂ α) (h₂ : a ⟂ β) : α ∥ β :=
sorry

-- Proposition ③: If α is perpendicular to β, then there exists a plane γ such that γ is perpendicular to both α and β.
theorem proposition_3 (h : α ⟂ β) : ∃ γ : Plane, γ ⟂ α ∧ γ ⟂ β :=
sorry

-- Proposition ④: If α is perpendicular to β, then there exists a line l such that l is perpendicular to α and l is parallel to β.
theorem proposition_4 (h : α ⟂ β) : ∃ l : Line, l ⟂ α ∧ l ∥ β :=
sorry

end proposition_2_proposition_3_proposition_4_l207_207734


namespace d4_neither_necessary_nor_sufficient_l207_207003

def is_geometric_sequence (a1 a2 a3 : ℕ) : Prop :=
  a2 * a2 = a1 * a3

theorem d4_neither_necessary_nor_sufficient :
  ∀ (d : ℕ),
  let a1 := 2 in
  let a2 := a1 + d in
  let a3 := a1 + 2 * d in
  (d = 4 → ¬is_geometric_sequence a1 a2 a3) ∧ 
  (is_geometric_sequence a1 a2 a3 → d = 0) :=
by
  intros d a1 a2 a3
  sorry

end d4_neither_necessary_nor_sufficient_l207_207003


namespace binary_division_remainder_l207_207606

theorem binary_division_remainder : 
  let b := 0b101101011010
  let n := 8
  b % n = 2 
:= by 
  sorry

end binary_division_remainder_l207_207606


namespace smallest_set_of_points_l207_207969

-- Define the conditions as Lean definitions
def is_origin_symmetric (T : Set (ℝ × ℝ)) : Prop :=
  ∀ {a b : ℝ}, (a, b) ∈ T → (-a, -b) ∈ T

def is_x_axis_symmetric (T : Set (ℝ × ℝ)) : Prop :=
  ∀ {a b : ℝ}, (a, b) ∈ T → (a, -b) ∈ T

def is_y_axis_symmetric (T : Set (ℝ × ℝ)) : Prop :=
  ∀ {a b : ℝ}, (a, b) ∈ T → (-a, b) ∈ T

def is_y_eq_x_symmetric (T : Set (ℝ × ℝ)) : Prop :=
  ∀ {a b : ℝ}, (a, b) ∈ T → (b, a) ∈ T

def is_y_eq_neg_x_symmetric (T : Set (ℝ × ℝ)) : Prop :=
  ∀ {a b : ℝ}, (a, b) ∈ T → (-b, -a) ∈ T

def contains_point (T : Set (ℝ × ℝ)) (p : ℝ × ℝ) : Prop :=
  p ∈ T

-- State the equivalent proof problem
theorem smallest_set_of_points (T : Set (ℝ × ℝ)) : 
  is_origin_symmetric T ∧ 
  is_x_axis_symmetric T ∧ 
  is_y_axis_symmetric T ∧ 
  is_y_eq_x_symmetric T ∧ 
  is_y_eq_neg_x_symmetric T ∧ 
  contains_point T (1, 4) → 
  ∃ S ⊆ T, S.card = 8 :=
sorry

end smallest_set_of_points_l207_207969


namespace midpoint_coordinates_to_product_l207_207698

theorem midpoint_coordinates_to_product (x y : ℝ) 
  (h_midpoint_x : (1 + x) / 2 = 4 )
  (h_midpoint_y : (-2 + y) / 2 = 8) :
  x * y = 126 :=
by 
  have hx : 1 + x = 8 := by linarith,
  have hy : -2 + y = 16 := by linarith,
  sorry

end midpoint_coordinates_to_product_l207_207698


namespace smallest_n_for_donuts_l207_207232

-- Definition of the condition that 13n - 1 is divisible by 9
def is_divisible_by_9 (n : ℕ) : Prop := (13 * n - 1) % 9 = 0

-- The main theorem asserting the smallest non-negative n satisfying the condition is 7
theorem smallest_n_for_donuts : ∃ n : ℕ, is_divisible_by_9 n ∧ n = 7 :=
by {
  use 7,
  unfold is_divisible_by_9,
  norm_num,
  sorry
}

end smallest_n_for_donuts_l207_207232


namespace tablecloth_length_l207_207844

/--
Maria needs to order a custom tablecloth and must specify the length in centimeters. Given that
- there are 12 inches in a foot,
- exactly 30.48 centimeters in a foot,
- and the required table length is 60 inches,

prove that the length she should specify in centimeters is 152.4 cm.
-/
theorem tablecloth_length (inches_to_foot : ℝ) (cm_to_foot : ℝ) (table_length_in : ℝ) : 
  inches_to_foot = 12 → 
  cm_to_foot = 30.48 → 
  table_length_in = 60 → 
  table_length_in * (1 / inches_to_foot) * cm_to_foot = 152.4 := 
by 
  intros h1 h2 h3 
  rw [h1, h2, h3] 
  norm_num 
  sorry

end tablecloth_length_l207_207844


namespace determine_prices_city_plans_and_max_profit_l207_207447

noncomputable def type_a_price : ℝ := 25
noncomputable def type_b_price : ℝ := 10

def total_price_2A3B (a b : ℝ) := 2 * a + 3 * b
def total_price_3A2B (a b : ℝ) := 3 * a + 2 * b

theorem determine_prices :
  (∃ a b : ℝ, total_price_2A3B a b = 80 ∧ total_price_3A2B a b = 95) ∧
  total_price_2A3B type_a_price type_b_price = 80 ∧
  total_price_3A2B type_a_price type_b_price = 95 :=
begin
  split,
  { use [25, 10],
    split;
    linarith, },
  { split;
    linarith, }
end

noncomputable def profit_per_a : ℝ := 8000
noncomputable def profit_per_b : ℝ := 5000

def purchase_equation (m n : ℝ) := 25 * m + 10 * n = 200

def profit (m n : ℝ) := profit_per_a * m + profit_per_b * n

theorem city_plans_and_max_profit :
  (∃ m n : ℝ, purchase_equation m n ∧ m ∈ ℤ ∧ n ∈ ℤ) ∧
  (∃! (m n : ℝ), purchase_equation m n ∧ m ∈ ℤ ∧ n ∈ ℤ ∧ 
    profit m n = 91000) :=
sorry

end determine_prices_city_plans_and_max_profit_l207_207447


namespace mirror_area_l207_207021

theorem mirror_area
  (L_frame W_frame w: ℝ)
  (h1: L_frame = 100)
  (h2: W_frame = 50)
  (h3: w = 8) :
  let L_mirror := L_frame - 2 * w
  let W_mirror := W_frame - 2 * w
  (L_mirror * W_mirror = 2856) :=
by
  let L_mirror := 100 - 2 * 8
  let W_mirror := 50 - 2 * 8
  show L_mirror * W_mirror = 2856
  sorry

end mirror_area_l207_207021


namespace range_of_k_real_roots_l207_207725

theorem range_of_k_real_roots : ∀ (k : ℝ), (∃ x : ℝ, x^2 + 2 * x + k = 0) ↔ (k ≤ 1) :=
by
  intro k
  split
  { -- Assuming there exists a real root
    intro h
    have h_discriminant := calc 
      (2 : ℝ)^2 - 4 * 1 * k = 4 - 4 * k : by ring
    exact le_of_eq (sub_nonneg.mp h_discriminant)
  }
  { -- Assuming k ≤ 1
    intro h
    use (-1 + real.sqrt (1 - k))
    have h_discriminant := calc 
      (2 : ℝ)^2 - 4 * 1 * k = 4 - 4 * k : by ring,
    have real_root_iff_discriminant_nonneg := (sq_nonneg (k - 1)).symm.trans h_discriminant,
    use sorry -- The existence of real root can be substantiated here.
  }

end range_of_k_real_roots_l207_207725


namespace students_with_two_skills_l207_207613

theorem students_with_two_skills :
  ∀ (n_students n_chess n_puzzles n_code : ℕ),
  n_students = 120 →
  n_chess = n_students - 50 →
  n_puzzles = n_students - 75 →
  n_code = n_students - 40 →
  (n_chess + n_puzzles + n_code - n_students) = 75 :=
by 
  sorry

end students_with_two_skills_l207_207613


namespace arithmetic_mean_of_4_and_6_is_5_l207_207449

-- Define the arithmetic mean of two numbers
def arithmetic_mean (a b : ℕ) : ℕ :=
  (a + b) / 2

-- State the theorem to be proved
theorem arithmetic_mean_of_4_and_6_is_5 : arithmetic_mean 4 6 = 5 :=
by 
  simp [arithmetic_mean]
  rfl

end arithmetic_mean_of_4_and_6_is_5_l207_207449


namespace opening_price_l207_207209

theorem opening_price (closing_price: ℝ) (percent_increase: ℝ) (opening_price: ℝ) : 
  closing_price = 9  → 
  percent_increase = 0.125 → 
  closing_price = opening_price * (1 + percent_increase) → 
  opening_price = 8 := 
by
  intro h1 h2 h3
  have h4 : opening_price * 1.125 = 9, from h3.subst (h1.symm ▸ rfl)
  have h5 : opening_price = 9 / 1.125, from eq_div_of_mul_eq h4 rfl
  have h6 : opening_price = 8, from calc
    opening_price = 9 / 1.125 : by exact h5
    ... = 8 : by norm_num
  exact h6
  

end opening_price_l207_207209


namespace wendy_washing_loads_l207_207117

theorem wendy_washing_loads (shirts sweaters machine_capacity : ℕ) (total_clothes := shirts + sweaters) 
  (loads := total_clothes / machine_capacity) 
  (remainder := total_clothes % machine_capacity) 
  (h_shirts : shirts = 39) 
  (h_sweaters : sweaters = 33) 
  (h_machine_capacity : machine_capacity = 8) : loads = 9 ∧ remainder = 0 := 
by 
  sorry

end wendy_washing_loads_l207_207117


namespace arithmetic_sequence_equality_l207_207777

theorem arithmetic_sequence_equality {a b c : ℝ} (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) (a20 : a ≠ c) (a2012 : b ≠ c) 
(h₄ : ∀ (i : ℕ), ∃ d : ℝ, a_n = a + i * d) : 
  1992 * a * c - 1811 * b * c - 181 * a * b = 0 := 
by {
  sorry
}

end arithmetic_sequence_equality_l207_207777


namespace mean_score_calculation_l207_207410

variable (morning_mean noon_mean afternoon_mean : ℝ)
variable (morning_ratio noon_ratio afternoon_ratio : ℝ)

def mean_of_all_students (morning_mean noon_mean afternoon_mean : ℝ)
  (morning_ratio noon_ratio afternoon_ratio : ℝ) : ℝ :=
  let total_students := (morning_ratio + noon_ratio + afternoon_ratio)
  let total_score := (morning_mean * morning_ratio + noon_mean * noon_ratio + afternoon_mean * afternoon_ratio)
  total_score / total_students

theorem mean_score_calculation :
  morning_mean = 85 → noon_mean = 75 → afternoon_mean = 65 →
  morning_ratio = 3/5 → noon_ratio = 2/5 → afternoon_ratio = 4/5 →
  mean_of_all_students morning_mean noon_mean afternoon_mean morning_ratio noon_ratio afternoon_ratio = 74 :=
  by
    intros
    sorry

end mean_score_calculation_l207_207410


namespace find_LM_l207_207807

variables (K L M N P Q : Type)
variables (KL MN LM KN MQ MP KP KM : ℝ) 

-- Conditions
def trapezoid (K L M N : Type) : Prop := 
  KM = 1 ∧ 
  KP = 1 ∧
  MQ = 1 ∧
  KN = MQ ∧
  LM = MP

-- To Prove
theorem find_LM (h : trapezoid K L M N) : LM = sqrt 2 :=
by sorry

end find_LM_l207_207807


namespace ant_ways_to_reach_l207_207201

theorem ant_ways_to_reach : 
  let n := 4020
  let a := 2010
  let b := 1005
  (nat.choose n b)^2 = nat.choose n (n - b) * nat.choose n b := by
sorry

end ant_ways_to_reach_l207_207201


namespace product_a4_a5_a6_l207_207012

-- Define the sequence
def seq (n : ℕ) : ℕ → ℕ
| 0       := 1    -- Arbitrary initial condition, since its value is not specifically provided
| (n + 1) := 2 * seq n

-- Given condition: a_5 = 4
axiom a5_eq_4 : seq 5 = 4

-- Prove that a_4 * a_5 * a_6 = 64
theorem product_a4_a5_a6 : seq 4 * seq 5 * seq 6 = 64 :=
by
  sorry

end product_a4_a5_a6_l207_207012


namespace simplified_expression_term_count_l207_207459

def even_exponents_terms_count : ℕ :=
  let n := 2008
  let k := 1004
  Nat.choose (k + 2) 2

theorem simplified_expression_term_count :
  even_exponents_terms_count = 505815 :=
sorry

end simplified_expression_term_count_l207_207459


namespace proof_one_third_of_seven_times_nine_subtract_three_l207_207626

def one_third_of_seven_times_nine_subtract_three : ℕ :=
  let product := 7 * 9
  let one_third := product / 3
  one_third - 3

theorem proof_one_third_of_seven_times_nine_subtract_three : one_third_of_seven_times_nine_subtract_three = 18 := by
  sorry

end proof_one_third_of_seven_times_nine_subtract_three_l207_207626


namespace profit_percentage_is_correct_l207_207175

-- Defining the conditions
def cost_price (x : ℝ) : ℝ := 6 * x
def selling_price (x : ℝ) : ℝ := 8 * (2 * x)
def profit (x : ℝ) : ℝ := selling_price x - cost_price x
def profit_percentage (x : ℝ) : ℝ := (profit x / cost_price x) * 100

-- Proving the profit percentage is 166.67%
theorem profit_percentage_is_correct (x : ℝ) (hx : x > 0) : profit_percentage x = 166.67 := 
  by
  sorry

end profit_percentage_is_correct_l207_207175


namespace common_sale_days_in_august_l207_207444

/-- Define the days of August that are multiples of 3 -/
def clothing_store_sale_days : finset ℕ :=
  finset.filter (λ n, n % 3 = 0) (finset.range 31)

/-- Define the days of August that are multiples of 7 + 5 (toy store - every 7 days starting from 5) -/
def toy_store_sale_days : finset ℕ :=
  finset.filter (λ n, ∃ k, n = 5 + 7 * k) (finset.range 31)

/-- The number of common sale days of August for both stores is 2. -/
theorem common_sale_days_in_august :
  finset.card (clothing_store_sale_days ∩ toy_store_sale_days) = 2 :=
sorry

end common_sale_days_in_august_l207_207444


namespace sum_non_prime_between_50_and_60_eq_383_l207_207938

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def non_primes_between_50_and_60 : List ℕ :=
  [51, 52, 54, 55, 56, 57, 58]

def sum_non_primes_between_50_and_60 : ℕ :=
  non_primes_between_50_and_60.sum

theorem sum_non_prime_between_50_and_60_eq_383 :
  sum_non_primes_between_50_and_60 = 383 :=
by
  sorry

end sum_non_prime_between_50_and_60_eq_383_l207_207938


namespace product_N_l207_207584

theorem product_N (A D D1 A1 : ℤ) (N : ℤ) 
  (h1 : D = A - N)
  (h2 : D1 = D + 7)
  (h3 : A1 = A - 2)
  (h4 : |D1 - A1| = 8) : 
  N = 1 → N = 17 → N * 17 = 17 :=
by
  sorry

end product_N_l207_207584


namespace trapezoid_LM_sqrt2_l207_207801

theorem trapezoid_LM_sqrt2 (K L M N P Q : Type*)
  (KM : ℝ)
  (KN MQ LM MP : ℝ)
  (h_KM : KM = 1)
  (h_KN_MQ : KN = MQ)
  (h_LM_MP : LM = MP) 
  (h_KP_1 : KN = 1) 
  (h_MQ_1 : MQ = 1) :
  LM = Real.sqrt 2 :=
by
  sorry

end trapezoid_LM_sqrt2_l207_207801


namespace sin_cos_sum_l207_207682

theorem sin_cos_sum (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π)
  (h : Real.tan (θ + Real.pi / 4) = 1 / 7) : Real.sin θ + Real.cos θ = -1 / 5 := 
by
  sorry

end sin_cos_sum_l207_207682


namespace sequence_thirtieth_term_l207_207820

theorem sequence_thirtieth_term :
  ∀ (a : ℕ) (d : ℕ), a = 2 ∧ d = 2 → a + 29 * d = 60 :=
by
  intro a d
  rintro ⟨ha, hd⟩
  simp [ha, hd]
  sorry

end sequence_thirtieth_term_l207_207820


namespace zero_point_when_a_is_one_range_of_a_for_zero_point_l207_207716

def f (a x : ℝ) : ℝ := 2 * a * 4^x - 2^x - 1

theorem zero_point_when_a_is_one : f 1 0 = 0 := 
by sorry

theorem range_of_a_for_zero_point : (∃ x : ℝ, f a x = 0) → a > 0 :=
by sorry

end zero_point_when_a_is_one_range_of_a_for_zero_point_l207_207716


namespace circle_polar_eq_l207_207375

theorem circle_polar_eq :
  ∀ (ρ θ : ℝ), (∀ (c : ℝ), c = sqrt 2 → (ρ = 0 → θ = π)) → 
  ((ρ^2 + 2 * sqrt 2 * ρ * cos θ = 0) ↔ (ρ = -2 * sqrt 2 * cos θ)) :=
by
  sorry

end circle_polar_eq_l207_207375


namespace alice_walk_time_l207_207194

theorem alice_walk_time (bob_time : ℝ) 
  (bob_distance : ℝ) 
  (alice_distance1 : ℝ) 
  (alice_distance2 : ℝ) 
  (time_ratio : ℝ) 
  (expected_alice_time : ℝ) :
  bob_time = 36 →
  bob_distance = 6 →
  alice_distance1 = 4 →
  alice_distance2 = 7 →
  time_ratio = 1 / 3 →
  expected_alice_time = 21 →
  (expected_alice_time = alice_distance2 / (alice_distance1 / (bob_time * time_ratio))) := 
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h3, h5]
  have h_speed : ℝ := alice_distance1 / (bob_time * time_ratio)
  rw [h4, h6]
  linarith [h_speed]

end alice_walk_time_l207_207194


namespace intersection_of_A_and_B_l207_207309

def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℕ := {1, 2, 4, 6}

theorem intersection_of_A_and_B : A ∩ B = {1, 2, 4} := by
  sorry

end intersection_of_A_and_B_l207_207309


namespace quadratic_discriminant_l207_207721

theorem quadratic_discriminant (a b c : ℝ) (x₁ x₂ : ℝ) 
  (h_eq : x₁ = (-b - real.sqrt (b^2 - 4*a*c)) / (2*a))
  (h_eq2 : x₂ = (-b + real.sqrt (b^2 - 4*a*c)) / (2*a))
  (h_dist : x₂ - x₁ = 2) :
  (b^2 - 4*a*c) = 4 :=
by sorry

end quadratic_discriminant_l207_207721


namespace total_payment_l207_207430

theorem total_payment (rahul_days : ℕ) (rajesh_days : ℕ) (rahul_share : ℚ) (total_payment : ℚ) 
  (h_rahul_days : rahul_days = 3) 
  (h_rajesh_days : rajesh_days = 2) 
  (h_rahul_share : rahul_share = 42)
  (work_per_day_rahul : ℚ := 1 / rahul_days)
  (work_per_day_rajesh : ℚ := 1 / rajesh_days)
  (total_work_per_day : ℚ := work_per_day_rahul + work_per_day_rajesh) 
  (work_completed_together : total_work_per_day = 5 / 6)
  (h_proportion : rahul_share / work_per_day_rahul = total_payment / 1) :
  total_payment = 126 := 
by
  sorry

end total_payment_l207_207430


namespace relationship_between_trigonometric_functions_l207_207268

open Real

theorem relationship_between_trigonometric_functions :
  let a := cos 2
  let b := sin 3
  let c := tan 4
  in a < b ∧ b < c := by
  -- Definitions based on conditions
  let a := cos 2
  let b := sin 3
  let c := tan 4
  -- Sorry to skip the proof
  sorry

end relationship_between_trigonometric_functions_l207_207268


namespace length_of_BM_l207_207694

theorem length_of_BM (A B C M : Point) (x h d : ℝ)
  (h_iso : dist A B = dist A C)
  (h_perp : ∠ C A B = π / 2)
  (h_BC : dist B C = 2 * h)
  (h_CA : dist C A = d)
  (h_BM_MC : dist B M = dist M C)
  (h_BM : dist B M = x) :
  x = h := 
sorry

end length_of_BM_l207_207694


namespace find_angle_C_l207_207762

noncomputable def ABC_triangle (A B C a b c : ℝ) : Prop :=
b = c * Real.cos A + Real.sqrt 3 * a * Real.sin C

theorem find_angle_C (A B C a b c : ℝ) (h : ABC_triangle A B C a b c) :
  C = π / 6 :=
sorry

end find_angle_C_l207_207762


namespace expected_value_of_sum_l207_207334

variable {s : Finset ℕ} (h_s : s = {1, 2, 3, 4, 5, 6}) (n : ℕ) [IncRange : 1 ≤ n ∧ n ≤ 6] 

theorem expected_value_of_sum (hmarbles : s.card = 6) (hk : choose 3 6 = 20) : 
  (∑ x in (s.powerset.filter (λ xs, xs.card = 3)), xs.sum) / 20 = 10.5 :=
by
  sorry

end expected_value_of_sum_l207_207334


namespace statement_a_statement_c_statement_d_l207_207383

noncomputable def triangleABC (A B C a b c : ℝ) := 
  A + B + C = π ∧ a = b * sin A / sin B ∧ b = c * sin B / sin C

theorem statement_a (A B : ℝ) (h : A > B) : cos A < cos B :=
by sorry

theorem statement_c {A B C : ℝ} (h : cos A * cos B * cos C > 0) : 
  A > 0 ∧ A < π / 2 ∧ B > 0 ∧ B < π / 2 ∧ C > 0 ∧ C < π / 2 :=
by sorry

theorem statement_d {a b c : ℝ} {A B C : ℝ} 
  (h : a - c * cos B = a * cos C) : 
  a = b ∨ C = π / 2 :=
by sorry

end statement_a_statement_c_statement_d_l207_207383


namespace find_a8_l207_207047

noncomputable def geometric_sequence (a_1 q : ℝ) (n : ℕ) : ℝ := a_1 * q^(n-1)

noncomputable def sum_geom (a_1 q : ℝ) (n : ℕ) : ℝ := a_1 * (1 - q^n) / (1 - q)

theorem find_a8 (a_1 q a_2 a_5 a_8 : ℝ) (S : ℕ → ℝ) 
  (Hsum : ∀ n, S n = sum_geom a_1 q n)
  (H1 : 2 * S 9 = S 3 + S 6)
  (H2 : a_2 = geometric_sequence a_1 q 2)
  (H3 : a_5 = geometric_sequence a_1 q 5)
  (H4 : a_2 + a_5 = 4)
  (H5 : a_8 = geometric_sequence a_1 q 8) :
  a_8 = 2 :=
sorry

end find_a8_l207_207047


namespace arjun_starting_amount_l207_207570

theorem arjun_starting_amount (X : ℝ) (h1 : Anoop_investment = 4000) (h2 : Anoop_months = 6) (h3 : Arjun_months = 12) (h4 : (X * 12) = (4000 * 6)) :
  X = 2000 :=
sorry

end arjun_starting_amount_l207_207570


namespace inverse_of_A_cubed_l207_207748

noncomputable def A_inv : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![ -2,  3],
    ![  0,  1]]

theorem inverse_of_A_cubed :
  (A_inv ^ 3) = ![![ -8,  9],
                    ![  0,  1]] :=
by sorry

end inverse_of_A_cubed_l207_207748


namespace circle_area_isosceles_triangle_l207_207533

theorem circle_area_isosceles_triangle : 
  ∀ (A B C : Type) (AB AC : Type) (a b c : ℝ),
  a = 5 →
  b = 5 →
  c = 4 →
  isosceles_triangle A B C a b c →
  circle_passes_through_vertices A B C →
  ∃ (r : ℝ), 
    area_of_circle_passing_through_vertices A B C = (15625 * π) / 1764 :=
by intros A B C AB AC a b c ha hb hc ht hcirc
   sorry

end circle_area_isosceles_triangle_l207_207533


namespace boat_goes_6_km_upstream_l207_207775

variable (speed_in_still_water : ℕ) (distance_downstream : ℕ) (time_downstream : ℕ) (effective_speed_downstream : ℕ) (speed_of_stream : ℕ)

-- Given conditions
def condition1 : Prop := speed_in_still_water = 11
def condition2 : Prop := distance_downstream = 16
def condition3 : Prop := time_downstream = 1
def condition4 : Prop := effective_speed_downstream = speed_in_still_water + speed_of_stream
def condition5 : Prop := effective_speed_downstream = 16

-- Prove that the boat goes 6 km against the stream in one hour.
theorem boat_goes_6_km_upstream : speed_of_stream = 5 →
  11 - 5 = 6 :=
by
  intros
  sorry

end boat_goes_6_km_upstream_l207_207775


namespace sammy_scored_fewer_than_jen_l207_207985

theorem sammy_scored_fewer_than_jen : ∀ (Bryan Jen Sammy : ℕ), 
  Bryan = 20 → 
  Jen = Bryan + 10 → 
  Sammy = 35 - 7 → 
  (Jen - Sammy) = 2 := 
by 
  intros Bryan Jen Sammy hBryan hJen hSammy
  rw [hBryan, hJen, hSammy]
  simp
  sorry

end sammy_scored_fewer_than_jen_l207_207985


namespace K_3_15_10_eq_151_30_l207_207677

def K (a b c : ℕ) : ℚ := (a : ℚ) / b + (b : ℚ) / c + (c : ℚ) / a

theorem K_3_15_10_eq_151_30 : K 3 15 10 = 151 / 30 := 
by
  sorry

end K_3_15_10_eq_151_30_l207_207677


namespace number_is_125_l207_207934

/-- Let x be a real number such that the difference between x and 3/5 of x is 50. -/
def problem_statement (x : ℝ) : Prop :=
  x - (3 / 5) * x = 50

/-- Prove that the only number that satisfies the above condition is 125. -/
theorem number_is_125 (x : ℝ) (h : problem_statement x) : x = 125 :=
by
  sorry

end number_is_125_l207_207934


namespace initial_rate_per_bowl_is_correct_l207_207959

noncomputable def initial_rate_per_bowl 
  (initial_bowls : ℕ) 
  (sold_bowls : ℕ) 
  (selling_price_per_bowl : ℝ)
  (percentage_gain : ℝ) 
  : ℝ :=
let C := initial_bowls * (17 / (1 + percentage_gain / 100)) in 
C / initial_bowls

theorem initial_rate_per_bowl_is_correct : initial_rate_per_bowl 114 108 17 23.88663967611336 = 13 :=
by simp [initial_rate_per_bowl]; sorry

end initial_rate_per_bowl_is_correct_l207_207959


namespace area_of_circumcircle_of_isosceles_triangle_l207_207535

theorem area_of_circumcircle_of_isosceles_triangle :
  ∀ (r : ℝ) (π : ℝ), (∀ (a b c : ℝ)
  (h1 : a = 5) (h2 : b = 5) (h3 : c = 4),
  r = sqrt (a * b * (a + b + c) * (a + b - c)) / c →
  ∀ (area : ℝ), area = π * r ^ 2 →
  area = 13125 / 1764 * π) :=
  λ r π a b c h1 h2 h3 h_radius area h_area, sorry

end area_of_circumcircle_of_isosceles_triangle_l207_207535


namespace find_LM_l207_207805

variables (K L M N P Q : Type)
variables (KL MN LM KN MQ MP KP KM : ℝ) 

-- Conditions
def trapezoid (K L M N : Type) : Prop := 
  KM = 1 ∧ 
  KP = 1 ∧
  MQ = 1 ∧
  KN = MQ ∧
  LM = MP

-- To Prove
theorem find_LM (h : trapezoid K L M N) : LM = sqrt 2 :=
by sorry

end find_LM_l207_207805


namespace rhomboid_toothpicks_l207_207958

/-- 
Given:
- The rhomboid consists of two sections, each similar to half of a large equilateral triangle split along its height.
- The longest diagonal of the rhomboid contains 987 small equilateral triangles.
- The effective fact that each small equilateral triangle contributes on average 1.5 toothpicks due to shared sides.

Prove:
- The number of toothpicks required to construct the rhomboid is 1463598.
-/

-- Defining the number of small triangles along the base of the rhomboid
def base_triangles : ℕ := 987

-- Calculating the number of triangles in one section of the rhomboid
def triangles_in_section : ℕ := (base_triangles * (base_triangles + 1)) / 2

-- Calculating the total number of triangles in the rhomboid
def total_triangles : ℕ := 2 * triangles_in_section

-- Given the effective sides per triangle contributing to toothpicks is on average 1.5
def avg_sides_per_triangle : ℚ := 1.5

-- Calculating the total number of toothpicks required
def total_toothpicks : ℚ := avg_sides_per_triangle * total_triangles

theorem rhomboid_toothpicks (h : base_triangles = 987) : total_toothpicks = 1463598 := by
  sorry

end rhomboid_toothpicks_l207_207958


namespace min_value_of_expression_l207_207249

noncomputable def expression (x : ℝ) : ℝ := (15 - x) * (12 - x) * (15 + x) * (12 + x)

theorem min_value_of_expression :
  ∃ x : ℝ, (expression x) = -1640.25 :=
sorry

end min_value_of_expression_l207_207249


namespace zeros_of_f_l207_207898

def f (x : ℝ) : ℝ := x^3 - 2*x^2 - x + 2

theorem zeros_of_f : (f (-1) = 0) ∧ (f 1 = 0) ∧ (f 2 = 0) :=
by 
  -- Placeholder for the proof
  sorry

end zeros_of_f_l207_207898


namespace find_degree_measure_of_B_l207_207763

-- Definitions for the given conditions
variable (A B C : ℝ)
variable [Fact (0 < B)] [Fact (B < π)]

-- Main condition
axiom condition : sin B ^ 2 - sin C ^ 2 - sin A ^ 2 = sqrt 3 * sin A * sin C

-- Target degree measure of B
theorem find_degree_measure_of_B : B = (5 * π) / 6 := by
  sorry

end find_degree_measure_of_B_l207_207763


namespace parabola_kite_area_correct_l207_207467

noncomputable def parabolas_intersect_area_kite
  (a b : ℝ)
  (h1 : (∃ x, y = ax^2 - 4) ∧ (∃ x, y = 6 - bx^2))
  (h2 : (a ≠ 0 ∧ b ≠ 0) ∧ (4b = 6a))
  (h3 : 16 = (1 / 2) * 2 * sqrt (4 / a) * 10) : Prop :=
a + b = 3.90625

theorem parabola_kite_area_correct : 
  ∀ (a b : ℝ),
  (parabolas_intersect_area_kite a b (∃ x, y = ax^2 - 4 ∧ ∃ x, y = 6 - bx^2)
    ((a ≠ 0 ∧ b ≠ 0) ∧ (4b = 6a))
    (16 = (1 / 2) * 2 * (4 / a).sqrt * 10))
  -> a + b = 3.90625 := sorry

end parabola_kite_area_correct_l207_207467


namespace unit_circle_equation_l207_207787

variable (s d t : ℝ)

-- Define the conditions for the problem
def unit_circle_conditions := PQ_parallel_MN ∧ OR_perpendicular_MN ∧ PQ_eq_s ∧ MN_eq_d ∧ PM_eq_t

-- The main theorem statement
theorem unit_circle_equation : unit_circle_conditions s d t → d + s = 2 * t :=
sorry

end unit_circle_equation_l207_207787


namespace smallest_b_for_perfect_square_l207_207922

theorem smallest_b_for_perfect_square : ∃ b : ℕ, b > 5 ∧ (∃ n : ℕ, 4 * b + 5 = n^2) ∧ ∀ b' : ℕ, b' > 5 → (∃ n' : ℕ, 4 * b' + 5 = n'^2) → b ≤ b' :=
by { use 11, split, linarith, split, use 7, norm_num, intros b' hb' hb'n', rcases hb'n' with ⟨n', hn'⟩, linarith }

end smallest_b_for_perfect_square_l207_207922


namespace count_perfect_sixth_powers_less_200_l207_207323

noncomputable def countPerfectSixthPowersUnder(n : ℕ) : ℕ :=
  Nat.card { k : ℕ | ∃ x : ℕ, x > 0 ∧ x^6 = k ∧ k < n }

theorem count_perfect_sixth_powers_less_200 : countPerfectSixthPowersUnder(200) = 2 := by
  sorry

end count_perfect_sixth_powers_less_200_l207_207323


namespace triangles_with_positive_area_in_3x3_grid_l207_207331

-- Define the points in the grid
def points : List (ℤ × ℤ) := [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]

-- Definition to count triangles with positive area formed by these points
def num_positive_area_triangles (pts : List (ℤ × ℤ)) : ℕ :=
  let combinations := pts.combinations 3
  let is_triangle (p1 p2 p3 : (ℤ × ℤ)) : Bool :=
    ((p2.1 - p1.1) * (p3.2 - p1.2)) ≠ ((p3.1 - p1.1) * (p2.2 - p1.2))
  combinations.filter (λ [p1, p2, p3] => is_triangle p1 p2 p3).length

-- The theorem to prove the number of positive area triangles is 76
theorem triangles_with_positive_area_in_3x3_grid : num_positive_area_triangles points = 76 := by
  sorry

end triangles_with_positive_area_in_3x3_grid_l207_207331


namespace y_solution_l207_207623

noncomputable def find_y (y : ℝ) : Prop :=
  (y^2 - 9*y + 8) / (y - 1) + (3*y^2 + 16*y - 12) / (3*y - 2) = -3 ∧
  y ≠ 1 ∧
  y ≠ 2 / 3

theorem y_solution : ∃ y : ℝ, find_y y ∧ y = -1 / 2 :=
begin
  sorry
end

end y_solution_l207_207623


namespace sum_sequence_eq_5000_l207_207122

theorem sum_sequence_eq_5000 :
  let seq := (List.range' 1 10001).map (λ n, if n % 2 = 0 then n else -n)
  List.sum seq = 5000 :=
by
  sorry

end sum_sequence_eq_5000_l207_207122


namespace find_a_l207_207785

-- Define the parametric curves
def C1_x (t : ℝ) : ℝ := t + 1 / t
def C1_y (t : ℝ) : ℝ := t - 1 / t

def C2_x (a θ : ℝ) : ℝ := a * Real.cos θ
def C2_y (θ : ℝ) : ℝ := Real.sin θ

-- Define the focus of the ellipse C2
def focus_C2 (a : ℝ) : set (ℝ × ℝ) := {p | p = (Real.sqrt (a^2 - 1), 0) ∨ p = (-Real.sqrt (a^2 - 1), 0)}

-- Problem statement: Prove that if C1 passes through the focus of C2, then a = sqrt(5)
theorem find_a (a : ℝ) (h : a > 1) :
  (∃ t : ℝ, (C1_x t, C1_y t) ∈ focus_C2 a) → a = Real.sqrt 5 := 
sorry

end find_a_l207_207785


namespace problem_solution_l207_207980

noncomputable def count_valid_arrangements : ℕ :=
  let candidates := { s : Fin 6 → Fin 6 | 
    s 0 ≠ 0 ∧ 
    s 2 ≠ 2 ∧ 
    s 4 ≠ 4 ∧ 
    s 0 < s 2 ∧ 
    s 2 < s 4 ∧ 
    ∀ i, s i ∈ {0, 1, 2, 3, 4, 5}.erase s 0 ∧ 
    ∀ j, s i < s j → i < j
  }
  candidates.card

theorem problem_solution : count_valid_arrangements = 30 := 
  sorry

end problem_solution_l207_207980


namespace find_perp_line_eq_l207_207457

-- Line equation in the standard form
def line_eq (x y : ℝ) : Prop := 4 * x - 3 * y - 12 = 0

-- Equation of the required line that is perpendicular to the given line and has the same y-intercept
def perp_line_eq (x y : ℝ) : Prop := 3 * x + 4 * y + 16 = 0

theorem find_perp_line_eq (x y : ℝ) :
  (∃ k : ℝ, line_eq 0 k ∧ perp_line_eq 0 k) →
  (∃ a b c : ℝ, perp_line_eq a b) :=
by
  sorry

end find_perp_line_eq_l207_207457


namespace periodic_decimal_to_fraction_l207_207241

theorem periodic_decimal_to_fraction : (0.7 + 0.32 : ℝ) == (1013 / 990 : ℝ) := by
  sorry

end periodic_decimal_to_fraction_l207_207241


namespace find_BF_l207_207781

theorem find_BF (A B C D E F : Type) 
  [has_right_angle A C]
  [is_perpendicular (line DE) (line AC)]
  [is_perpendicular (line BF) (line AC)] 
  (AE DE CE AC AF CF BF : ℝ) 
  (h1 : AE = 4) 
  (h2 : DE = 6) 
  (h3 : CE = 8) 
  (h4 : AC = AE + CE) 
  (h5 : AC = AF + CF) 
  (h6 : AF / BF = 2 / 3) 
  (h7 : CF / BF = 4 / 3) 
  (h8 : AE + CE = 12) : 
  BF = 6 := 
sorry

end find_BF_l207_207781


namespace pure_imaginary_tan_l207_207343

theorem pure_imaginary_tan (θ : ℝ) (h : ∃ θ : ℝ, (sin θ - (3/5) = 0) ∧ (cos θ - (4/5) ≠ 0)) :
  tan θ = - (3/4) :=
sorry

end pure_imaginary_tan_l207_207343


namespace number_representation_correct_l207_207189

-- Conditions: 5 in both the tenths and hundredths places, 0 in remaining places.
def number : ℝ := 50.05

theorem number_representation_correct :
  number = 50.05 :=
by 
  -- The proof will show that the definition satisfies the condition.
  sorry

end number_representation_correct_l207_207189


namespace circle_tangent_radii_l207_207242

theorem circle_tangent_radii (a b c : ℝ) (A : ℝ) (p : ℝ)
  (r r_a r_b r_c : ℝ)
  (h1 : p = (a + b + c) / 2)
  (h2 : r = A / p)
  (h3 : r_a = A / (p - a))
  (h4 : r_b = A / (p - b))
  (h5 : r_c = A / (p - c))
  : 1 / r = 1 / r_a + 1 / r_b + 1 / r_c := 
  sorry

end circle_tangent_radii_l207_207242


namespace sequence_70th_term_is_630_l207_207896

def contains_digit_1 (n : ℕ) : Prop := 
  n.toDigits 10 |>.any (λ d, d = 1)

def multiple_of_3_with_1_digit (n : ℕ) : Prop := 
  n % 3 = 0 ∧ contains_digit_1 n

noncomputable def sequence_70th_term := 
  Nat.find (λ n, n = 630)

theorem sequence_70th_term_is_630 : sequence_70th_term = 630 := 
sorry

end sequence_70th_term_is_630_l207_207896


namespace correct_statement_B_l207_207130

/-- Define the diameter of a sphere -/
def diameter (d : ℝ) (s : Set (ℝ × ℝ × ℝ)) : Prop :=
∃ x y : ℝ × ℝ × ℝ, x ∈ s ∧ y ∈ s ∧ dist x y = d ∧ ∀ z ∈ s, dist x y ≥ dist x z ∧ dist x y ≥ dist z y

/-- Define that a line segment connects two points on the sphere's surface and passes through the center -/
def connects_diameter (center : ℝ × ℝ × ℝ) (radius : ℝ) (x y : ℝ × ℝ × ℝ) : Prop :=
dist center x = radius ∧ dist center y = radius ∧ (x + y) / 2 = center

/-- A sphere is the set of all points at a fixed distance from the center -/
def sphere (center : ℝ × ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ × ℝ) :=
{x | dist center x = radius}

theorem correct_statement_B (center : ℝ × ℝ × ℝ) (radius : ℝ) (x y : ℝ × ℝ × ℝ):
  (∀ (s : Set (ℝ × ℝ × ℝ)), sphere center radius = s → diameter (2 * radius) s)
  → connects_diameter center radius x y
  → (∃ d : ℝ, diameter d (sphere center radius)) := 
by
  intros
  sorry

end correct_statement_B_l207_207130


namespace molar_mass_of_compound_l207_207742

variable (total_weight : ℝ) (num_moles : ℝ)

theorem molar_mass_of_compound (h1 : total_weight = 2352) (h2 : num_moles = 8) :
    total_weight / num_moles = 294 :=
by
  rw [h1, h2]
  norm_num

end molar_mass_of_compound_l207_207742


namespace find_square_plot_area_l207_207124

noncomputable def side_length (cost_per_foot : ℝ) (total_cost : ℝ) : ℝ :=
  total_cost / (4 * cost_per_foot)

noncomputable def area_of_square_plot (cost_per_foot : ℝ) (total_cost : ℝ) : ℝ :=
  let s := side_length cost_per_foot total_cost
  s^2

theorem find_square_plot_area (cost_per_foot : ℝ) (total_cost : ℝ) :
  cost_per_foot = 58 → total_cost = 3944 → area_of_square_plot cost_per_foot total_cost = 289 :=
by
  intros h_cost h_total
  rw [area_of_square_plot, side_length]
  rw [h_cost, h_total]
  norm_num
  sorry

end find_square_plot_area_l207_207124


namespace symmetry_of_transformed_graphs_l207_207257

noncomputable def y_eq_f_x_symmetric_line (f : ℝ → ℝ) : Prop :=
∀ (x : ℝ), f (x - 19) = f (99 - x) ↔ x = 59

theorem symmetry_of_transformed_graphs (f : ℝ → ℝ) :
  y_eq_f_x_symmetric_line f :=
by {
  sorry
}

end symmetry_of_transformed_graphs_l207_207257


namespace range_of_a_l207_207315

noncomputable def P (a : ℝ) : Prop :=
∀ x : ℝ, a * x^2 + a * x + 1 > 0

noncomputable def Q (a : ℝ) : Prop :=
(∃ (x y : ℝ), (x^2 / a + y^2 / (a - 3) = 1)) ∧ ∀ (x y : ℝ), (x^2 / a + y^2 / (a - 3) = 1) → (a * (a - 3) < 0)

theorem range_of_a (a : ℝ) (h1 : P a ∨ Q a) (h2 : ¬ (P a ∧ Q a)) : a = 0 ∨ (3 ≤ a ∧ a < 4) := 
sorry

end range_of_a_l207_207315


namespace largest_angle_of_triangle_l207_207605

theorem largest_angle_of_triangle (a b: ℝ) (h1: a = 70) (h2: b = 80) : 
  ∃ (c: ℝ), c + a + b = 180 ∧ max a (max b c) = 80 :=
by
  use 30
  split
  sorry
  sorry

end largest_angle_of_triangle_l207_207605


namespace minimum_apples_l207_207155

theorem minimum_apples (n : ℕ) : 
  n % 4 = 1 ∧ n % 5 = 2 ∧ n % 9 = 7 → n = 97 := 
by 
  -- To be proved
  sorry

end minimum_apples_l207_207155


namespace gameFair_l207_207503

-- Definition of the condition
def isGameFair (A B : ℕ) : Prop :=
  (A % 2 = B % 2) → (1/2 : ℝ)

-- The theorem that needs to be proven
theorem gameFair : 
  ∀ (A B : ℕ), (isGameFair A B) → (1/2) :=
by sorry

end gameFair_l207_207503


namespace simple_interest_months_l207_207660

theorem simple_interest_months
    (P : ℝ) (R : ℝ) (SI : ℝ) (months : ℕ)
    (hP : P = 10000)
    (hR : R = 0.08)
    (hSI : SI = 800)
    (hMonths : months = 12) :
    (SI / (P * R)) * 12 = months :=
by
  rw [hP, hR, hSI, hMonths]
  norm_num
  sorry

end simple_interest_months_l207_207660


namespace compute_expression_l207_207593

theorem compute_expression : 6^2 + 2 * 5 - 4^2 = 30 :=
by sorry

end compute_expression_l207_207593


namespace triangle_area_l207_207768

variable (a b c k : ℝ)
variable (h1 : a = 2 * k)
variable (h2 : b = 3 * k)
variable (h3 : c = k * Real.sqrt 13)

theorem triangle_area (h_right_triangle : a^2 + b^2 = c^2) : 
  (1 / 2 * a * b) = 3 * k^2 := 
by 
  sorry

end triangle_area_l207_207768


namespace logarithm_problem_l207_207607

theorem logarithm_problem :
  (2017 ^ (1 / (Real.log 2017 / Real.log 2)) * 
   2017 ^ (1 / (Real.log 2017 / Real.log 4)) * 
   2017 ^ (1 / (Real.log 2017 / Real.log 8)) * 
   2017 ^ (1 / (Real.log 2017 / Real.log 16)) * 
   2017 ^ (1 / (Real.log 2017 / Real.log 32))) ^ (1 / 5) = 8 :=
by
  sorry

end logarithm_problem_l207_207607


namespace find_LM_l207_207803

variables (K L M N P Q : Type)
variables (KL MN LM KN MQ MP KP KM : ℝ) 

-- Conditions
def trapezoid (K L M N : Type) : Prop := 
  KM = 1 ∧ 
  KP = 1 ∧
  MQ = 1 ∧
  KN = MQ ∧
  LM = MP

-- To Prove
theorem find_LM (h : trapezoid K L M N) : LM = sqrt 2 :=
by sorry

end find_LM_l207_207803


namespace find_square_plot_area_l207_207123

noncomputable def side_length (cost_per_foot : ℝ) (total_cost : ℝ) : ℝ :=
  total_cost / (4 * cost_per_foot)

noncomputable def area_of_square_plot (cost_per_foot : ℝ) (total_cost : ℝ) : ℝ :=
  let s := side_length cost_per_foot total_cost
  s^2

theorem find_square_plot_area (cost_per_foot : ℝ) (total_cost : ℝ) :
  cost_per_foot = 58 → total_cost = 3944 → area_of_square_plot cost_per_foot total_cost = 289 :=
by
  intros h_cost h_total
  rw [area_of_square_plot, side_length]
  rw [h_cost, h_total]
  norm_num
  sorry

end find_square_plot_area_l207_207123


namespace inscribed_shapes_equal_area_l207_207017

theorem inscribed_shapes_equal_area (b h : ℝ) (h_circle : h > 0) : 
  (h * b = (1 / 2) * b * (1 - h / 2)) → (h = 2 / 5) :=
by
  intros h_eq
  have h_eq_prime : h * b = b / 2 - b * (h / 4) := by nlinarith
  replace h_eq_prime := (eq_div_iff (by linarith : b ≠ 0)).mp h_eq 
  have : h = (1/2) - (h / 4) := by linarith
  have : 4 * h = 2 - h := by linarith
  have : 5 * h = 2 := by linarith
  linarith

end inscribed_shapes_equal_area_l207_207017


namespace largest_prime_factor_of_12321_l207_207636

theorem largest_prime_factor_of_12321 : ∃ p : ℕ, prime p ∧ p = 43 ∧ (∀ q : ℕ, prime q ∧ q ∣ 12321 → q ≤ p) :=
by
  sorry

end largest_prime_factor_of_12321_l207_207636


namespace ratio_of_areas_l207_207203

noncomputable def side_length_triangle : ℝ := 10

def radius_inscribed_circle (s: ℝ) : ℝ := (s * Real.sqrt 3) / 6

def side_length_inscribed_square (R: ℝ) : ℝ := 2 * R

def radius_smallest_circle (a: ℝ) : ℝ := a / 2

def area_equilateral_triangle (s: ℝ) : ℝ := (Real.sqrt 3 / 4) * s^2

def area_circle (r: ℝ) : ℝ := Real.pi * r^2

theorem ratio_of_areas :
  let s := side_length_triangle
  let R := radius_inscribed_circle s
  let a := side_length_inscribed_square R
  let r := radius_smallest_circle a
  let A_triangle := area_equilateral_triangle s
  let A_circle := area_circle r in
  (A_circle / A_triangle) = Real.pi * Real.sqrt 3 :=
by
  sorry

end ratio_of_areas_l207_207203


namespace PyramidVolume_l207_207862

noncomputable def VolumeOfPyramid : ℝ :=
  let EF := 10
  let FG := 6
  let PF := 20
  let BaseArea := EF * FG
  let PE := Real.sqrt (PF^2 - EF^2)
  V := (1 / 3) * BaseArea * PE
  V

theorem PyramidVolume : 
  VolumeOfPyramid = 200 * Real.sqrt 3 := by
  sorry

end PyramidVolume_l207_207862


namespace AM_length_l207_207001

-- Definitions based on conditions:
def parallelogram (A B C D : Type*) : Prop :=
  AB = CD ∧ AD = BC ∧ (∃ M : Type*, midpoint A B M ∧ midpoint C D M)

def midpoint (A B M : Type*) : Prop :=
  distance A M = distance M B

def distance (A B : Type*) : Prop :=
  -- assume some distance function

variables {A B C D M : Type*}
variables (AB BC CD AD : ℝ)

-- Parallelogram conditions
axiom parallelogram_ABCD : parallelogram A B C D
axiom AB_eq_10 : distance A B = 10
axiom AB_eq_2BC : distance A B = 2 * distance B C
axiom M_midpoint_CD : midpoint C D M
axiom BM_eq_2AM : distance B M = 2 * distance A M

-- Lean 4 statement to prove that AM = 2 * sqrt(5)
theorem AM_length : distance A M = 2 * sqrt 5 :=
by
  sorry

end AM_length_l207_207001


namespace minimum_value_of_expr_l207_207398

open Real

-- Define the main function to be minimized
def expr (x y z : ℝ) : ℝ :=
  9 * z / (3 * x + 2 * y) + 9 * x / (2 * y + 3 * z) + 4 * y / (2 * x + z)

-- State the theorem
theorem minimum_value_of_expr (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  expr x y z ≥ 9 / 2 :=
sorry

end minimum_value_of_expr_l207_207398


namespace ratio_AB_F1F2_l207_207979

variables (p q : ℝ)

-- Conditions
def is_equilateral_triangle_inscribed (A B C : ℝ × ℝ) : Prop :=
  B = (0, q) ∧
  A.2 = C.2 ∧
  (∃ F1 F2 : ℝ × ℝ, F1.2 - F2.2 = 0 ∧ (B - A).dist (B - C) = F1.dist F2 = 2)

def ellipse_eq (p q : ℝ) (A : ℝ × ℝ) : Prop := (A.1 ^ 2 / p^2) + (A.2 ^ 2 / q^2) = 1

-- Goals
theorem ratio_AB_F1F2 (A B C F1 F2 : ℝ × ℝ) (h1 : is_equilateral_triangle_inscribed A B C) (h2 : ellipse_eq p q A)
  (h3 : ellipse_eq p q B) (h4 : ellipse_eq p q C) : 
  (B.dist C / 2) = 8 / 5 :=
sorry

end ratio_AB_F1F2_l207_207979


namespace length_of_BC_l207_207760

noncomputable def BC_length (AB AC : ℝ) (angle_ABC : ℝ) : ℝ :=
  if h : AB = 2 ∧ AC = Real.sqrt 7 ∧ angle_ABC = (2 * Real.pi / 3) then
    let t := 1 in
    t
  else
    0

theorem length_of_BC :
  ∀ (t : ℝ) (AB AC : ℝ) (angle_ABC : ℝ),
    AB = 2 →
    AC = Real.sqrt 7 →
    angle_ABC = (2 * Real.pi / 3) →
    t = BC_length AB AC angle_ABC →
    t = 1 := by
  intros t AB AC angle_ABC hAB hAC hAngle h
  rw [BC_length] at h
  split_ifs at h
  . cases h
    exact (show 1 = 1 by rfl)
  . contradiction
  sorry

end length_of_BC_l207_207760


namespace sum_of_distinct_prime_factors_of_252_l207_207928

theorem sum_of_distinct_prime_factors_of_252 : (∑ p in {2, 3, 7}, p) = 12 := by
  sorry

end sum_of_distinct_prime_factors_of_252_l207_207928


namespace basketball_scoring_l207_207947

theorem basketball_scoring : 
  ∃ n : ℕ, n = 17 ∧ 
  ∀ (x y z : ℕ), (x + y + z = 8) → 
  (∀ p, p = 3*x + 2*y + 1*z → 
    ∃ p_set : set ℕ, p ∈ p_set ∧ ((∀ (x y z : ℕ), (x + y + z = 8) → (3*x + 2*y + 1*z ∈ p_set)) ∧ p_set.card = n)) :=
sorry

end basketball_scoring_l207_207947


namespace number_mod_conditions_l207_207548

theorem number_mod_conditions :
  ∃ N, (N % 10 = 9) ∧ (N % 9 = 8) ∧ (N % 8 = 7) ∧ (N % 7 = 6) ∧
       (N % 6 = 5) ∧ (N % 5 = 4) ∧ (N % 4 = 3) ∧ (N % 3 = 2) ∧ (N % 2 = 1) ∧
       N = 2519 :=
by
  sorry

end number_mod_conditions_l207_207548


namespace tank_capacity_l207_207182

theorem tank_capacity (fill_rate drain_rate1 drain_rate2 : ℝ)
  (initial_fullness : ℝ) (time_to_fill : ℝ) (capacity_in_liters : ℝ) :
  fill_rate = 1 / 2 ∧
  drain_rate1 = 1 / 4 ∧
  drain_rate2 = 1 / 6 ∧ 
  initial_fullness = 1 / 2 ∧ 
  time_to_fill = 60 →
  capacity_in_liters = 10000 :=
by {
  sorry
}

end tank_capacity_l207_207182


namespace distance_between_circles_distance_between_curves_l207_207031

theorem distance_between_circles (C_1 C_2 : ℝ → ℝ → Prop) : 
  (∀ x y, C_1 x y ↔ x^2 + y^2 = 2) → 
  (∀ x y, C_2 x y ↔ (x - 3)^2 + (y - 3)^2 = 2) →
  d C_1 C_2 = sqrt 2 :=
by
  intros hC1 hC2
  sorry

theorem distance_between_curves (C_3 C_4 : ℝ → ℝ → Prop) : 
  (∀ x y, C_3 x y ↔ e^x - 2 * y = 0) →
  (∀ x y, C_4 x y ↔ log x + log 2 = y) →
  d C_3 C_4 = sqrt 2 * (1 - log 2) :=
by
  intros hC3 hC4
  sorry

end distance_between_circles_distance_between_curves_l207_207031


namespace probability_AB_not_consecutive_l207_207442

-- Definitions based on the given conditions:
def cars : List String := ["A", "B", "C", "D", "E"]
def total_arrangements := cars.permutations.length

-- Function to calculate the number of valid arrangements where A and B are not next to each other
def non_consecutive_AB_arrangements : Int := sorry

-- Lean theorem statement:
theorem probability_AB_not_consecutive :
  (non_consecutive_AB_arrangements.toFloat / total_arrangements.toFloat) = 3 / 5 := 
sorry

end probability_AB_not_consecutive_l207_207442


namespace sum_has_minimum_term_then_d_positive_Sn_positive_then_increasing_sequence_l207_207033

def is_sum_of_arithmetic_sequence (S : ℕ → ℚ) (a₁ d : ℚ) :=
  ∀ n : ℕ, S n = n * a₁ + (n * (n - 1) / 2) * d

theorem sum_has_minimum_term_then_d_positive
  {S : ℕ → ℚ} {d a₁ : ℚ} (h : d ≠ 0)
  (hS : is_sum_of_arithmetic_sequence S a₁ d)
  (h_min : ∃ n : ℕ, ∀ m : ℕ, S n ≤ S m) :
  d > 0 :=
sorry

theorem Sn_positive_then_increasing_sequence
  {S : ℕ → ℚ} {d a₁ : ℚ} (h : d ≠ 0)
  (hS : is_sum_of_arithmetic_sequence S a₁ d)
  (h_pos : ∀ n : ℕ, S n > 0) :
  (∀ n : ℕ, S n < S (n + 1)) :=
sorry

end sum_has_minimum_term_then_d_positive_Sn_positive_then_increasing_sequence_l207_207033


namespace angle_AHB_l207_207562

-- Definitions of the conditions
def angle_BAC : ℝ := 52
def angle_ABC : ℝ := 64
def angle_ACB := 180 - angle_BAC - angle_ABC

-- We assume the triangle ABC, its orthocenter H, and altitudes meet the above conditions.
theorem angle_AHB (H orthocenter : triangle) :
  angle_BAC = 52 ∧
  angle_ABC = 64 ∧
  angle_ACB = 180 - angle_BAC - angle_ABC →
  angle orthocenter A H B = 180 - angle_ACB :=
by
  intros h,
  sorry

end angle_AHB_l207_207562


namespace emissions_2019_range_of_m_l207_207069

-- Define the basic setup and conditions
def carbon_emissions (year: ℕ) (m: ℝ) : ℝ :=
  if year = 1 then 400 * 0.9 + m
  else (400 - 10 * m) * (0.9)^year + 10 * m

-- Prove total carbon emissions of City A in 2019
theorem emissions_2019 (m: ℝ) (hm: m > 0) : 
  carbon_emissions 2 m = 324 + 1.9 * m := 
sorry

-- Prove the range of values for m
theorem range_of_m (m: ℝ) (hm: m > 0) :
  (∀ n: ℕ, n > 0 → carbon_emissions n m ≤ 550) ↔ (m ∈ Ioc 0 55) :=
sorry

end emissions_2019_range_of_m_l207_207069


namespace probability_top_face_odd_is_137_over_252_l207_207057

noncomputable def probability_top_face_odd (faces : Fin 6 → ℕ) : ℚ :=
  let n := 1/6 * ((399/420) + (76/420) + (323/420) + (136/420) + (256/420) + (180/420))
  n

theorem probability_top_face_odd_is_137_over_252 :
  probability_top_face_odd (λ i : Fin 6, i.val + 1) = 137 / 252 :=
by 
  sorry

end probability_top_face_odd_is_137_over_252_l207_207057


namespace general_term_formula_l207_207730

noncomputable def seq (n : ℕ) : ℝ :=
  if n = 0 then 0
  else (seq n.succ - 1) = 3 * (seq n) + sqrt (8 * (seq n) ^ 2 + 1)

theorem general_term_formula :
  ∀ n : ℕ, seq n = (sqrt 2 / 8) * (3 + 2 * sqrt 2) ^ n - (sqrt 2 / 8) * (3 - 2 * sqrt 2) ^ n :=
by
  sorry

end general_term_formula_l207_207730


namespace number_of_paths_l207_207595

-- Define vertices A, B, D, and the adjacency relations in the cube.
variables (A B D : Type) [fintype A] [fintype B] [fintype D]

-- Define edge relationships
variable (edges : A -> B -> D -> Prop)

-- Conditions
variable (adjacent : ∀ v : A, fintype (ULift.{u} (fin 3)))
variable (path_condition : ∀ (a : A) (b : B) (d : D), ∃ x y : adjacent, edges a x y d)

-- Proposition: There are exactly 2 paths from A to B passing through D
theorem number_of_paths (edges : A -> B -> D -> Prop) : 
  (∃ (path : A -> B -> D -> Type), fintype (path A B D) ∧ (fintype.card (path A B D) = 2)) :=
  sorry

end number_of_paths_l207_207595


namespace cannot_make_it_in_time_l207_207385

theorem cannot_make_it_in_time (time_available : ℕ) (distance_to_station : ℕ) (v1 : ℕ) :
  time_available = 2 ∧ distance_to_station = 2 ∧ v1 = 30 → 
  ¬ ∃ v2, (time_available - (distance_to_station / v1)) * v2 ≥ 1 :=
by
  sorry

end cannot_make_it_in_time_l207_207385


namespace probability_of_selecting_ANGLE_letter_l207_207235

-- Define the set of letters in "GEOMETRY"
def GEOMETRY : set Char := {'G', 'E', 'O', 'M', 'T', 'R', 'Y'}

-- Define the set of letters in "ANGLE"
def ANGLE : set Char := {'A', 'N', 'G', 'L', 'E'}

-- The eight letters in "GEOMETRY" are placed in a bag
def letters_in_bag : list Char := ['G', 'E', 'O', 'M', 'E', 'T', 'R', 'Y']

-- Define the probability function
def prob_letter_in_ANGLE (letters_in_bag : list Char) (ANGLE : set Char) : ℚ :=
  let common_letters := list.filter (λ c, c ∈ ANGLE) letters_in_bag in
  common_letters.length / letters_in_bag.length

theorem probability_of_selecting_ANGLE_letter :
  prob_letter_in_ANGLE letters_in_bag ANGLE = 1 / 4 := by
  sorry

end probability_of_selecting_ANGLE_letter_l207_207235


namespace sum_of_elements_in_S_in_base_2_l207_207392

def is_valid_element (n : ℕ) : Prop :=
  (1 << 3 ≤ n) ∧ (n < 1 << 4)

def S : set ℕ := { n | is_valid_element n}

theorem sum_of_elements_in_S_in_base_2 :
  ∑ n in S.to_finset, n = 0b1011100 := by
  sorry

end sum_of_elements_in_S_in_base_2_l207_207392


namespace not_divisible_by_5_and_product_of_digits_l207_207196

theorem not_divisible_by_5_and_product_of_digits :
  (∀ n ∈ [3640, 3855, 3922, 4025, 4120], ¬ (n ≠ 3922 ∧ (n % 10 = 0 ∨ n % 10 = 5)))
  ∧ ((3922 / 100 % 10) * (3922 / 10 % 10) = 18) :=
by
  -- Condition 1: Every number except 3922 is divisible by 5
  have cond1 : ∀ n ∈ [3640, 3855, 3922, 4025, 4120], ¬ (n ≠ 3922 ∧ (n % 10 = 0 ∨ n % 10 = 5)) := by sorry,
  -- Condition 2: The product of the hundreds digit and the tens digit of 3922 is 18
  have cond2 : (3922 / 100 % 10) * (3922 / 10 % 10) = 18 := by sorry,
  exact ⟨cond1, cond2⟩

end not_divisible_by_5_and_product_of_digits_l207_207196


namespace largest_prime_factor_12321_l207_207653

theorem largest_prime_factor_12321 : ∃ p, prime p ∧ (∀ q, prime q ∧ q ∣ 12321 → q ≤ p) ∧ p = 19 :=
by {
  sorry
}

end largest_prime_factor_12321_l207_207653


namespace num_good_numbers_in_set_1_to_200_l207_207961

def is_good_number (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), n = 20 * a + 8 * b + 27 * c

def good_numbers (s : Finset ℕ) : Finset ℕ :=
  s.filter is_good_number

theorem num_good_numbers_in_set_1_to_200 : 
  (good_numbers (Finset.range 201)).card = 153 :=
by
  sorry

end num_good_numbers_in_set_1_to_200_l207_207961


namespace ant_path_count_l207_207199

theorem ant_path_count :
  let binom := Nat.choose 4020 1005 in
  ∃ f : Fin 2 → ℕ, 
  (f 0 = 2010 ∧ f 1 = 2010 ∧ (∑ x : Fin 4020, if x.mod 2 = 0 then 1 else -1) = 4020) →
  binom * binom = (Nat.choose 4020 1005) ^ 2 := 
by
  sorry

end ant_path_count_l207_207199


namespace trapezoid_LM_value_l207_207808

theorem trapezoid_LM_value (K L M N P Q : Type) 
  (d1 d2 : ℝ)
  (h1 : d1 = 1)
  (h2 : d2 = 1)
  (height_eq : KM = 1)
  (KN_eq_MQ : KN = MQ)
  (LM_eq_MP : LM = MP) :
  LM = 1 / real.sqrt (real.sqrt 2) :=
by 
  sorry

end trapezoid_LM_value_l207_207808


namespace part1_part2_l207_207402

variable {D : Type*} [InnerProductSpace ℝ D]
variables {a b : D} {lambda : ℝ} 
          (A B C : D)
          (x : D)
          (f : D → D)

-- Definitions/Conditions
def map_f (x : D) : D := lambda • x
def not_collinear (a b : D) : Prop := a ≠ 0 ∧ b ≠ 0 ∧ ¬(∃ k : ℝ, k ≠ 0 ∧ a = k • b)
def same_length (a b : D) : Prop := ∥a∥ = ∥b∥

theorem part1 (h1 : same_length a b) (h2 : not_collinear a b) (h3 : f = map_f) :
  inner (f a - f b) (a + b) = 0 := sorry

noncomputable def map_and_scale (x : D) : D := (2 : ℝ) • x -- Given that λ = 2

theorem part2 :
  let A := (1 : ℝ, 2 : ℝ) in
  let B := (3 : ℝ, 6 : ℝ) in
  let C := (4 : ℝ, 8 : ℝ) in
  f = map_and_scale →
  lambda = 2 := sorry

end part1_part2_l207_207402


namespace meaningful_sqrt_of_nonneg_l207_207756

theorem meaningful_sqrt_of_nonneg (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 2)) ↔ x ≥ 2 :=
by sorry

end meaningful_sqrt_of_nonneg_l207_207756


namespace first_non_divisible_number_l207_207483

theorem first_non_divisible_number :
  ∃ (n : ℕ), n = 7 ∧ (∀ (x : ℕ), 200 ≤ x ∧ x ≤ 300 → (¬ (x % 3 = 0) ∧ ¬ (x % 5 = 0)) → x % n ≠ 0) :=
begin
  sorry -- proof goes here
end

end first_non_divisible_number_l207_207483


namespace solve_power_tower_eq_four_l207_207456

noncomputable def infinite_power_tower (x : ℝ) : ℝ :=
classical.some (Zorn.zorn_partial_order₀ {y : ℝ | y = x ^ y})

theorem solve_power_tower_eq_four :
  infinite_power_tower (Real.sqrt 2) = 4 :=
sorry

end solve_power_tower_eq_four_l207_207456


namespace max_expr_value_l207_207656

def expr (x y : ℝ) :=
  (sqrt (3 - sqrt 2) * sin x - sqrt (2 * (1 + cos (2 * x))) - 1) *
  (3 + 2 * sqrt (7 - sqrt 2) * cos y - cos (2 * y))

theorem max_expr_value : ∃ x y : ℝ, abs (expr x y - 9) < 1 :=
by
  sorry

end max_expr_value_l207_207656


namespace ashley_champagne_bottles_l207_207580

theorem ashley_champagne_bottles (guests : ℕ) (glasses_per_guest : ℕ) (servings_per_bottle : ℕ) 
  (h1 : guests = 120) (h2 : glasses_per_guest = 2) (h3 : servings_per_bottle = 6) : 
  (guests * glasses_per_guest) / servings_per_bottle = 40 :=
by
  -- The proof will go here
  sorry

end ashley_champagne_bottles_l207_207580


namespace find_x_plus_inv_x_l207_207705

theorem find_x_plus_inv_x (x : ℝ) (h : x^3 + 1/x^3 = 110) : x + 1/x = 5 := by
  sorry

end find_x_plus_inv_x_l207_207705


namespace sum_of_perimeters_490_l207_207420

-- Definitions and conditions
def Point (ℝ : Type) := ℝ × ℝ
noncomputable def A : Point ℝ := (-15, 0)
noncomputable def B : Point ℝ := (-7, 0)
noncomputable def C : Point ℝ := (15, 0)
def AB : ℝ := 8
def BC : ℝ := 22
def AC : ℝ := 30

-- Problem statement
theorem sum_of_perimeters_490 :
  let D : Point ℝ, AD : ℝ, CD : ℝ, BD : ℝ in
  ∀ (x y : ℝ), point_of_line(D) -> AD = x ∧ CD = x → y^2 = x^2 - (AB)^2 ∧ y^2 = x^2 - (BC)^2 →
  let perimeters := 3 * AC + 2 * (106 + 55 + 39) in
  perimeters = 490 :=
by
  sorry

end sum_of_perimeters_490_l207_207420


namespace xy_eq_xb_iff_xy_parallel_de_l207_207836

theorem xy_eq_xb_iff_xy_parallel_de 
  (A B C D E : Point) (Γ : Circle)
  (hA : A ∈ Γ) (hB : B ∈ Γ) (hC : C ∈ Γ) 
  (hD : D ∈ Γ) (hE : E ∈ Γ) 
  (clockwise_order : clockwise_order Γ A B C D E)
  (Y : Point) (hCD_AE : line_through C D ∩ line_through A E = Y)
  (X : Point) (hAC_ext : X ∈ line_through A C)
  (hXB_tangent : tangent_to_circle X B Γ) :
  (XY = XB) ↔ (is_parallel XY DE) :=
sorry

end xy_eq_xb_iff_xy_parallel_de_l207_207836


namespace coins_in_bag_l207_207152

theorem coins_in_bag (n : ℕ) :
  let total_value := n * (1 + 0.5 + 0.25 + 0.10 + 0.05) in
  total_value = 555 → n = 292 :=
by {
  intro h,
  sorry
}

end coins_in_bag_l207_207152


namespace files_rem_nat_eq_two_l207_207146

-- Conditions
def initial_music_files : ℕ := 4
def initial_video_files : ℕ := 21
def files_deleted : ℕ := 23

-- Correct Answer
def files_remaining : ℕ := initial_music_files + initial_video_files - files_deleted

theorem files_rem_nat_eq_two : files_remaining = 2 := by
  sorry

end files_rem_nat_eq_two_l207_207146


namespace correct_statements_l207_207930

section

variables {k b a : ℝ}
variables {x y : ℝ}

-- Condition for Statement A
def line_passes_first_second_fourth_quadrant (k b : ℝ) : Prop :=
  ∀ x, (x * k + b > 0 ∧ x > 0) ∨ (x * k + b > 0 ∧ x < 0) ∨ (x * k + b < 0 ∧ x > 0)

-- Condition for Statement B
def line_passes_fixed_point (a : ℝ) : Prop :=
  ∀ (x y : ℝ), y = a * x - 3 * a + 2 → (x = 3 ∧ y = 2)

-- Condition for Statement C
def point_slope_form_correct (x y : ℝ) (m : ℝ) : Prop :=
  y + 1 = m * (x - 2)

-- Our proof goal
theorem correct_statements :
  (line_passes_first_second_fourth_quadrant k b → k < 0 ∧ b > 0) ∧
  (line_passes_fixed_point a → ∀ a, ∃ x y, y = a * x - 3 * a + 2 ∧ x = 3 ∧ y = 2) ∧
  (point_slope_form_correct 2 (-1) (-real.sqrt 3) → ∀ (x y : ℝ), y + 1 = -real.sqrt 3 * (x - 2)) :=
by { 
  sorry 
}

end

end correct_statements_l207_207930


namespace divisors_difference_l207_207666

theorem divisors_difference 
  (n : ℕ)
  (p_1 p_2 p_3 p_4 : ℕ) 
  (h_n : n = p_1 * p_2 * p_3 * p_4)
  (h_distinct_primes : nat.prime p_1 ∧ nat.prime p_2 ∧ nat.prime p_3 ∧ nat.prime p_4 ∧ p_1 ≠ p_2 ∧ p_1 ≠ p_3 ∧ p_1 ≠ p_4 ∧ p_2 ≠ p_3 ∧ p_2 ≠ p_4 ∧ p_3 ≠ p_4)
  (d : list ℕ)
  (h_d : d = (nat.divisors n).filter (λ x, x ≤ n))
  (h_sorted : d.sorted (≤))
  (h_length : d.length = 16)
  (h_lt : n < 1995) :
  (d.nth_le 8 sorry) - (d.nth_le 7 sorry) ≠ 22 :=
sorry

end divisors_difference_l207_207666


namespace largest_prime_factor_12321_l207_207643

theorem largest_prime_factor_12321 : 
  ∃ p : ℕ, prime p ∧ p ∣ 12321 ∧ ∀ q : ℕ, prime q ∧ q ∣ 12321 → q ≤ p :=
begin
  use 83,
  split,
  { -- Prove that 83 is a prime number
    sorry },
  split,
  { -- Prove that 83 divides 12321
    sorry },
  { -- Prove that any other prime factor of 12321 is less than or equal to 83
    sorry }
end

end largest_prime_factor_12321_l207_207643


namespace perpendicular_segments_product_equal_l207_207769

theorem perpendicular_segments_product_equal
  {A B C D E F : Type*}
  [metric_space A]
  [metric_space B]
  [metric_space C]
  [metric_space D]
  [metric_space E]
  [metric_space F]
  (h : is_right_angle ∠ B A C) 
  (D_on_hypotenuse : is_on_segment B C D) 
  (h_perpendicular : ∥D E∥ ⊥ BC ∧ ∥D F∥ ⊥ BC) :
  segment_length D B * segment_length D C = segment_length D E * segment_length D F :=
sorry

end perpendicular_segments_product_equal_l207_207769


namespace trapezoid_length_KLMN_l207_207789

variables {K L M N P Q : Type}
variables (trapezoid KLMN : K L M N)
variable (KM : ℝ) (KP MQ LM MP : ℝ)
variables (perp1 : KP > 0) (perp2 : MQ > 0)
variables (equal1 : KM = 1) (equal2 : KP = MQ) (equal3 : LM = MP)

theorem trapezoid_length_KLMN
(equality_KM: KM = 1)
(equality_KP_MQ: KP = MQ)
(equality_LM_MP: LM = MP)
: LM = sqrt 2 := 
by sorry

end trapezoid_length_KLMN_l207_207789


namespace sum_of_distinct_prime_factors_of_252_l207_207925

theorem sum_of_distinct_prime_factors_of_252 : (∑ p in {2, 3, 7}, p) = 12 :=
by {
  -- The prime factors of 252 are 2, 3, and 7
  have h1 : {2, 3, 7} ⊆ {p : ℕ | p.prime ∧ p ∣ 252},
  { simp, },
  -- Calculate the sum of these factors
  exact sorry
}

end sum_of_distinct_prime_factors_of_252_l207_207925


namespace sunny_lead_l207_207764

-- Define the given conditions as hypotheses
variables (h d : ℝ) (s w : ℝ)
    (H1 : ∀ t, t = 2 * h → (s * t) = 2 * h ∧ (w * t) = 2 * h - 2 * d)
    (H2 : ∀ t, (s * t) = 2 * h + 2 * d → (w * t) = 2 * h)

-- State the theorem we want to prove
theorem sunny_lead (h d : ℝ) (s w : ℝ) 
    (H1 : ∀ t, t = 2 * h → (s * t) = 2 * h ∧ (w * t) = 2 * h - 2 * d)
    (H2 : ∀ t, (s * t) = 2 * h + 2 * d → (w * t) = 2 * h) :
    ∃ distance_ahead_Sunny : ℝ, distance_ahead_Sunny = (2 * d^2) / h :=
sorry

end sunny_lead_l207_207764


namespace geom_seq_inc_condition_l207_207828

theorem geom_seq_inc_condition (a₁ a₂ q : ℝ) (h₁ : a₁ > 0) (h₂ : a₂ = a₁ * q) :
  (a₁^2 < a₂^2) ↔ 
  (∀ n m : ℕ, n < m → (a₁ * q^n) < (a₁ * q^m) ∨ ((a₁ * q^n) = (a₁ * q^m) ∧ q = 1)) :=
by
  sorry

end geom_seq_inc_condition_l207_207828


namespace sin_40_point_5_deg_l207_207229

theorem sin_40_point_5_deg :
  let θ := 81
  let cos_identity := ∀ α, cos (90 - α) = sin α
  let cos_81 := sin 9
  sin 40.5 = sqrt (2 + sqrt (2 + sqrt 2)) / 2 :=
by
  -- Define degrees in terms of radians
  let deg := 0.0174533 -- approximation of π/180 for degrees to radians conversion
  let sin_deg := λ x : ℝ, Real.sin (x * deg)
  let cos_deg := λ x : ℝ, Real.cos (x * deg)

  -- Define needed constants in terms of degrees
  have θ := 81
  have cos_identity := ∀ α, cos_deg (90 - α) = sin_deg α
  have cos_81 := sin_deg 9

  -- Then the goal is to show
  show sin_deg 40.5 = sqrt (2 + sqrt (2 + sqrt 2)) / 2
  sorry

end sin_40_point_5_deg_l207_207229


namespace soccer_league_games_l207_207870

/-- Prove the number of league games scheduled in the Big Eighteen Soccer League is 351. --/
theorem soccer_league_games (divisions : ℕ) (teams_per_division : ℕ) (intra_division_games : ℕ) (inter_division_games : ℕ) :
  divisions = 3 →
  teams_per_division = 6 →
  intra_division_games = 3 →
  inter_division_games = 2 →
  -- Calculating intra-division games
  let intra_division_game_count := (teams_per_division * (teams_per_division - 1) / 2) * intra_division_games * divisions in
  -- Calculating inter-division games
  let inter_division_game_count := (teams_per_division * teams_per_division * (divisions - 1)) * inter_division_games in
  -- Total games, making sure to divide the inter-division count by 2 because each game is counted twice
  let total_games := intra_division_game_count + (inter_division_game_count / 2) in
  total_games = 351 := 
by
  intros h1 h2 h3 h4
  simp [h1, h2, h3, h4]
  sorry

end soccer_league_games_l207_207870


namespace pebble_placement_count_l207_207217

theorem pebble_placement_count (n : ℕ) (h : 1 < n) :
  ∃ (f : fin n → fin n → bool), (∀ i j, f i j = tt → i ≠ j) ∧
  (∑ i in finset.range n, finset.univ.filter (λ j, f i j = tt).card = 1) ∧
  ∃ k, k = 2^n := sorry

end pebble_placement_count_l207_207217


namespace bottles_needed_l207_207577

-- Define specific values provided in conditions
def servings_per_guest : ℕ := 2
def number_of_guests : ℕ := 120
def servings_per_bottle : ℕ := 6

-- Define total servings needed
def total_servings : ℕ := servings_per_guest * number_of_guests

-- Define the number of bottles needed (as a proof statement)
theorem bottles_needed : total_servings / servings_per_bottle = 40 := by
  /-
    The proof will go here. For now we place a sorry to mark the place where
    a proof would be required. The statement should check the equivalence of 
    number of bottles needed being 40 given the total servings divided by 
    servings per bottle.
  -/
  sorry

end bottles_needed_l207_207577


namespace circles_intersect_tangent_m_form_l207_207589

noncomputable def sum_abc : ℕ := sorry

theorem circles_intersect_tangent_m_form :
  ∃ (a b c : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    (∀ p, prime p → ¬ (p^2 ∣ b)) ∧ 
    gcd a c = 1 ∧ 
    (∃ m, m = (a * (b.sqrt)) / c ∧
          a + b + c = 180) :=
begin
  sorry
end

end circles_intersect_tangent_m_form_l207_207589


namespace largest_multiple_l207_207918

theorem largest_multiple (a b limit : ℕ) (ha : a = 3) (hb : b = 5) (h_limit : limit = 800) : 
  ∃ (n : ℕ), (lcm a b) * n < limit ∧ (lcm a b) * (n + 1) ≥ limit ∧ (lcm a b) * n = 795 := 
by 
  sorry

end largest_multiple_l207_207918


namespace martha_apartment_number_l207_207407

def is_two_digit_number (n : ℕ) : Prop := n ≥ 10 ∧ n < 100
def is_prime (n : ℕ) : Prop := nat.prime n
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0
def has_digit_7 (n : ℕ) : Prop := n / 10 = 7 ∨ n % 10 = 7

theorem martha_apartment_number (n : ℕ) : 
  is_two_digit_number n ∧
  (is_prime n ∨ is_odd n ∨ is_divisible_by_3 n ∨ has_digit_7 n) ∧
  (if is_prime n then 1 else 0) +
  (if is_odd n then 1 else 0) +
  (if is_divisible_by_3 n then 1 else 0) +
  (if has_digit_7 n then 1 else 0) = 3 →
  (n / 10 = 5) :=
begin
  sorry
end

end martha_apartment_number_l207_207407


namespace not_congruent_triangles_l207_207132

-- Definitions based directly on conditions
def equilateral (T : Triangle) : Prop :=
  T.a = T.b ∧ T.b = T.c 

def similar (T1 T2 : Triangle) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ T1.a = k * T2.a ∧ T1.b = k * T2.b ∧ T1.c = k * T2.c

def congruent (T1 T2 : Triangle) : Prop :=
  T1.a = T2.a ∧ T1.b = T2.b ∧ T1.c = T2.c

-- Condition of triangle ABC
def triangle_ABC : Triangle := ⟨a1, b1, c1⟩
-- Condition of triangle DEF
def triangle_DEF : Triangle := ⟨a2, b2, c2⟩

-- Assuming equilateral conditions
axiom equilateral_ABC : equilateral triangle_ABC
axiom equilateral_DEF : equilateral triangle_DEF

-- Given triangle DEF has twice the side length of triangle ABC
axiom def_side_length : 2 * triangle_ABC.a = triangle_DEF.a

-- The proof problem reshaped as a statement in Lean
theorem not_congruent_triangles : ¬ congruent triangle_ABC triangle_DEF :=
by sorry

end not_congruent_triangles_l207_207132


namespace vectors_parallel_eq_l207_207738

-- Defining the problem
variables {m : ℝ}

-- Main statement
theorem vectors_parallel_eq (h : ∃ k : ℝ, (k ≠ 0) ∧ (k * 1 = m) ∧ (k * m = 2)) :
  m = Real.sqrt 2 ∨ m = -Real.sqrt 2 :=
sorry

end vectors_parallel_eq_l207_207738


namespace count_perfect_sixth_powers_less_200_l207_207321

noncomputable def countPerfectSixthPowersUnder(n : ℕ) : ℕ :=
  Nat.card { k : ℕ | ∃ x : ℕ, x > 0 ∧ x^6 = k ∧ k < n }

theorem count_perfect_sixth_powers_less_200 : countPerfectSixthPowersUnder(200) = 2 := by
  sorry

end count_perfect_sixth_powers_less_200_l207_207321


namespace even_function_a_value_l207_207044

noncomputable def f (x a : ℝ) : ℝ := (x + 1) * (2 * x + 3 * a)

theorem even_function_a_value (a : ℝ) : f(x, a) = (x + 1) * (2 * x + 3 * a) ∧ (∀ x : ℝ, f x a = f (-x) a) → a = - (2 / 3) := by
  sorry

end even_function_a_value_l207_207044


namespace worker_A_probability_worker_B_disqualification_l207_207942

noncomputable def problem1 (p_A : ℝ) : ℝ :=
  1 - (p_A ^ 3)

noncomputable def problem2 (p_B : ℝ) : ℝ :=
  let q_B := 1 - p_B in
  -- Placeholder for actual detailed formula involving p_B
  (q_B ^ 2) * (p_B ^ 2) +  -- Fill in the actual expression
  sorry

/-!
  Given:
  p_A : ℝ is the probability that Worker A passes a test.
  p_B : ℝ is the probability that Worker B passes a test.

  Prove:
  problem1 is Worker A's probability of failing at least once in three consecutive months.
  problem2 is Worker B's probability of being disqualified after exactly four tests.
-/

theorem worker_A_probability (p_A : ℝ) (hp_A : 0 ≤ p_A ∧ p_A ≤ 1) : 
  problem1 p_A = 1 - (p_A ^ 3) := by
  -- The proof would go here
  sorry

theorem worker_B_disqualification (p_B : ℝ) (hp_B : 0 ≤ p_B ∧ p_B ≤ 1) : 
  let q_B := 1 - p_B in
  problem2 p_B = (q_B ^ 2) * (p_B ^ 2) + -- Fill in actual details
  sorry

end worker_A_probability_worker_B_disqualification_l207_207942


namespace polynomial_identity_l207_207827

noncomputable def Q : ℝ → ℝ := λ x, -x^3 + 2*x^2 + x - 1

theorem polynomial_identity (Q0 Q1 Q3 Q4 : ℝ)
  (h1 : ∀ x, Q x = Q0 + Q1 * x + Q3 * x^2 + Q4 * x^3)
  (h2 : Q(-2) = 2)
  (h3 : Q(-1) = 3) :
  ∀ x, Q x = -x^3 + 2*x^2 + x - 1 := 
by
  sorry

end polynomial_identity_l207_207827


namespace sum_of_x_values_l207_207772

noncomputable def arithmetic_angles_triangle (x : ℝ) : Prop :=
  let α := 30 * Real.pi / 180
  let β := (30 + 40) * Real.pi / 180
  let γ := (30 + 80) * Real.pi / 180
  (x = 6) ∨ (x = 8) ∨ (x = (7 + Real.sqrt 36 + Real.sqrt 83))

theorem sum_of_x_values : ∀ x : ℝ, 
  arithmetic_angles_triangle x → 
  (∃ p q r : ℝ, x = p + Real.sqrt q + Real.sqrt r ∧ p = 7 ∧ q = 36 ∧ r = 83) := 
by
  sorry

end sum_of_x_values_l207_207772


namespace pyramid_volume_formula_l207_207081

noncomputable def pyramid_volume (a α β : ℝ) : ℝ :=
  (1/6) * a^3 * (Real.sin (α/2)) * (Real.tan β)

theorem pyramid_volume_formula (a α β : ℝ) :
  (base_is_isosceles_triangle : Prop) → (lateral_edges_inclined : Prop) → 
  pyramid_volume a α β = (1/6) * a^3 * (Real.sin (α/2)) * (Real.tan β) :=
by
  intros c1 c2
  exact sorry

end pyramid_volume_formula_l207_207081


namespace sequence_x_2022_l207_207674

theorem sequence_x_2022 :
  ∃ (x : ℕ → ℤ), x 1 = 1 ∧ x 2 = 1 ∧ x 3 = -1 ∧
  (∀ n, 4 ≤ n → x n = x (n-1) * x (n-3)) ∧ x 2022 = 1 := by
  sorry

end sequence_x_2022_l207_207674


namespace original_charge_rate_eq_l207_207083

-- Given conditions
variables (a b : ℝ)

-- Original problem statement and answer
theorem original_charge_rate_eq : 
  ∃ (x : ℝ), (0.8 * (x - a) = b) ∧ (x = a + 1.25 * b) :=
begin
  use a + 1.25 * b,
  split,
  { sorry },
  { refl }
end

end original_charge_rate_eq_l207_207083


namespace solution_to_sequence_l207_207680

theorem solution_to_sequence (a q : ℝ)
  (h1 : ∀ (n : ℕ), n >= 1 → a * q ^ (n - 1))
  (h2 : ∀ (b1 b2 b3 b4 : ℝ), b1 = a ∧ b2 = aq + 6 ∧ b3 = aq^2 + 3 ∧ b4 = aq^3 - 96 → (b2 - b1 = b3 - b2 ∧ b3 - b2 = b4 - b3)) :
  a = 1 ∧ q = 4 ∧ aq = 4 ∧ aq^2 = 16 ∧ aq^3 = 64 :=
by
  sorry

end solution_to_sequence_l207_207680


namespace cupcakes_left_at_home_correct_l207_207435

-- Definitions of the conditions
def total_cupcakes_baked : ℕ := 53
def boxes_given_away : ℕ := 17
def cupcakes_per_box : ℕ := 3

-- Calculate the total number of cupcakes given away
def total_cupcakes_given_away := boxes_given_away * cupcakes_per_box

-- Calculate the number of cupcakes left at home
def cupcakes_left_at_home := total_cupcakes_baked - total_cupcakes_given_away

-- Prove that the number of cupcakes left at home is 2
theorem cupcakes_left_at_home_correct : cupcakes_left_at_home = 2 := by
  sorry

end cupcakes_left_at_home_correct_l207_207435


namespace projection_correct_l207_207228

open Real EuclideanSpace

noncomputable def projection_onto_line (v: ℝ^3) (d: ℝ^3) : ℝ^3 :=
  let dot_prod := inner v d
  let mag_squared := inner d d
  (dot_prod / mag_squared) • d

theorem projection_correct : 
  let v := ![3, 6, -3]
  let d := ![1, -1/2, 1/2]
  projection_onto_line v d = ![-1, 1/2, -1/2] := by
  sorry

end projection_correct_l207_207228


namespace bijection_exists_l207_207028

variable {n : ℕ} (S : Set ℕ) (hS : S.card = n) (hn : 3 ≤ n)

noncomputable def exists_bijection (f : Fin n → ℕ) : Prop :=
  (∀ (i j k : Fin n), i < j ∧ j < k → f j ^ 2 ≠ f i * f k) ∧ Function.Bijective f

theorem bijection_exists :
  ∃ (f : Fin n → ℕ), exists_bijection S f :=
sorry

end bijection_exists_l207_207028


namespace trapezoid_length_KLMN_l207_207792

variables {K L M N P Q : Type}
variables (trapezoid KLMN : K L M N)
variable (KM : ℝ) (KP MQ LM MP : ℝ)
variables (perp1 : KP > 0) (perp2 : MQ > 0)
variables (equal1 : KM = 1) (equal2 : KP = MQ) (equal3 : LM = MP)

theorem trapezoid_length_KLMN
(equality_KM: KM = 1)
(equality_KP_MQ: KP = MQ)
(equality_LM_MP: LM = MP)
: LM = sqrt 2 := 
by sorry

end trapezoid_length_KLMN_l207_207792


namespace fifth_diagram_shaded_fraction_l207_207977

theorem fifth_diagram_shaded_fraction :
  ∀ (n : ℕ), n = 5 →
  let shaded_squares := (2 * n - 1) ^ 2,
      total_squares := (2 * n) ^ 2 in
  (n = 5) → (shaded_squares : ℚ) / total_squares = 81 / 100 :=
by
  intros n hn
  simp only [shaded_squares, total_squares]
  simp only [hn]
  sorry

end fifth_diagram_shaded_fraction_l207_207977


namespace smallest_b_eq_sqrt_11_div_30_l207_207661

theorem smallest_b_eq_sqrt_11_div_30 :
  ∃ b : ℝ, 
    (∃ b_pos : b > 0, 
      ∀ x : ℝ, 
        x > 0 → 
          (
            (9 * real.sqrt ((3 * b)^2 + 2^2) - 6 * b^2 - 4) / (real.sqrt (4 + 6 * b^2) + 5) = 3 
          ) → 
          x >= b
    ) 
    ∧ b = real.sqrt (11 / 30) :=
begin
  sorry
end

end smallest_b_eq_sqrt_11_div_30_l207_207661


namespace area_of_rectangular_park_l207_207472

theorem area_of_rectangular_park
  (l w : ℕ) 
  (h_perimeter : 2 * l + 2 * w = 80)
  (h_length : l = 3 * w) :
  l * w = 300 :=
sorry

end area_of_rectangular_park_l207_207472


namespace cat_mouse_positions_l207_207374

theorem cat_mouse_positions (n : ℕ) (h : n = 179) : 
  (cat_position n = bottom_right) ∧ (mouse_position n = right_middle) :=
sorry

end cat_mouse_positions_l207_207374


namespace evaluate_g_at_5_l207_207038

noncomputable def g (x : ℝ) : ℝ := 2 * x ^ 4 - 15 * x ^ 3 + 24 * x ^ 2 - 18 * x - 72

theorem evaluate_g_at_5 : g 5 = -7 := by
  sorry

end evaluate_g_at_5_l207_207038


namespace count_sixth_powers_lt_200_l207_207326

theorem count_sixth_powers_lt_200 : 
  {n : ℕ | n > 0 ∧ n < 200 ∧ (∃ k : ℕ, n = k^6)}.to_finset.card = 2 := 
by sorry

end count_sixth_powers_lt_200_l207_207326


namespace rick_total_clothes_ironed_l207_207067

def rick_ironing_pieces
  (shirts_per_hour : ℕ)
  (pants_per_hour : ℕ)
  (hours_shirts : ℕ)
  (hours_pants : ℕ) : ℕ :=
  (shirts_per_hour * hours_shirts) + (pants_per_hour * hours_pants)

theorem rick_total_clothes_ironed :
  rick_ironing_pieces 4 3 3 5 = 27 :=
by
  sorry

end rick_total_clothes_ironed_l207_207067


namespace variance_scaling_and_translation_l207_207279

variables (ξ : Type*) [variance_structure ξ]
variable (x : ξ) 

def D (ξ : Type*) [variance_structure ξ] : real :=
sorry  -- Add the actual definition if available

axiom var_xi : D ξ = 2

theorem variance_scaling_and_translation : D (2 * x + 3) = 8 :=
by
  rw [variance_scaling, var_xi]
  sorry

end variance_scaling_and_translation_l207_207279


namespace circle_area_isosceles_triangle_l207_207539

theorem circle_area_isosceles_triangle (a b c : ℝ) (h1 : a = 5) (h2 : b = 5) (h3 : c = 4) :
  let r := ((25 * Real.sqrt 21) / 42)
  in ∃ (O : Point) (r : ℝ), Circle O r ∧ r^2 * Real.pi = (13125 / 1764) * Real.pi := by
  sorry

end circle_area_isosceles_triangle_l207_207539


namespace ashok_total_subjects_l207_207583

theorem ashok_total_subjects 
  (n : ℕ)
  (avg_all_subjects : ℕ)
  (avg_first_five : ℕ)
  (marks_last : ℕ)
  (h1 : avg_all_subjects = 80)
  (h2 : avg_first_five = 74)
  (h3 : marks_last = 110)
  (h4 : n > 5)
  : n = 6 :=
by
  let T := avg_first_five * 5
  have eq1 : T + marks_last = avg_all_subjects * n, from sorry
  have eq2 : T = 74 * 5, from sorry
  have eq3 : 74 * 5 + 110 = 80 * n, from sorry
  have eq4 : 370 + 110 = 80 * n, from sorry
  have eq5 : 480 = 80 * n, from sorry
  have eq6 : n = 480 / 80, from sorry
  sorry

end ashok_total_subjects_l207_207583


namespace solve_for_x_l207_207446

-- Define the custom operation
def custom_mul (a b : ℝ) : ℝ := 4 * a - 2 * b

-- Main statement to prove
theorem solve_for_x : (∃ x : ℝ, custom_mul 3 (custom_mul 4 x) = 10) ↔ (x = 7.5) :=
by
  sorry

end solve_for_x_l207_207446


namespace number_of_B_students_l207_207363

-- Conditions
def prob_A (prob_B : ℝ) := 0.6 * prob_B
def prob_C (prob_B : ℝ) := 1.6 * prob_B
def prob_D (prob_B : ℝ) := 0.3 * prob_B

-- Total students
def total_students : ℝ := 50

-- Main theorem statement
theorem number_of_B_students (x : ℝ) (h1 : prob_A x + x + prob_C x + prob_D x = total_students) :
  x = 14 :=
  by
-- Proof skipped
  sorry

end number_of_B_students_l207_207363


namespace prove_avg_mark_of_batch3_l207_207484

noncomputable def avg_mark_of_batch3 (A1 A2 A3 : ℕ) (Marks1 Marks2 Marks3 : ℚ) : Prop :=
  A1 = 40 ∧ A2 = 50 ∧ A3 = 60 ∧ Marks1 = 45 ∧ Marks2 = 55 ∧ 
  (A1 * Marks1 + A2 * Marks2 + A3 * Marks3) / (A1 + A2 + A3) = 56.333333333333336 → 
  Marks3 = 65

theorem prove_avg_mark_of_batch3 : avg_mark_of_batch3 40 50 60 45 55 65 :=
by
  unfold avg_mark_of_batch3
  sorry

end prove_avg_mark_of_batch3_l207_207484


namespace determinant_of_projection_matrix_l207_207400

noncomputable def projection_matrix (a b : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  Matrix.of ![![a^2, a*b], ![a*b, b^2]] / (a^2 + b^2)

theorem determinant_of_projection_matrix :
  let vector := ![3, 5]
  let a := (3 : ℝ)
  let b := (5 : ℝ)
  let norm_sq := a^2 + b^2
  let Q := projection_matrix a b in
  Matrix.det Q = 0 :=
by
  let vector := ![3, 5]
  let a := (3 : ℝ)
  let b := (5 : ℝ)
  let norm_sq := a^2 + b^2
  let Q := projection_matrix a b
  have Q_is_projection_matrix : Q = Matrix.of ![![9 / 34, 15 / 34], ![15 / 34, 25 / 34]],
  sorry
  show Matrix.det Q = 0,
  sorry

end determinant_of_projection_matrix_l207_207400


namespace tripod_new_height_approximation_l207_207560

noncomputable def new_tripod_height : ℝ :=
  let leg_length := 6 in
  let initial_height := 5 in
  -- Conversion of angles from degrees to radians
  let angle_100 := Real.pi * 100 / 180 in
  let angle_65 := Real.pi * 65 / 180 in
  let angle_130 := Real.pi * 130 / 180 in
  -- Calculations
  let AB := leg_length * sin(angle_50) in
  let area_triangle_ABC := 1/2 * leg_length * leg_length * sin(angle_100) in
  let TD := 2 * area_triangle_ABC / AB in
  let TG := 2/3 * TD in
  TG

theorem tripod_new_height_approximation :
  (floor (5 + sqrt 130) : ℤ) = 5 :=
by
  sorry

end tripod_new_height_approximation_l207_207560


namespace harmonic_mean_pairs_count_l207_207669

theorem harmonic_mean_pairs_count :
  ∃ (s : Finset (ℕ × ℕ)), (∀ p ∈ s, p.1 < p.2 ∧ 2 * p.1 * p.2 = 4^15 * (p.1 + p.2)) ∧ s.card = 29 :=
sorry

end harmonic_mean_pairs_count_l207_207669


namespace necklaces_caught_l207_207233

noncomputable def total_necklaces_caught (boudreaux rhonda latch cecilia : ℕ) : ℕ :=
  boudreaux + rhonda + latch + cecilia

theorem necklaces_caught :
  ∃ (boudreaux rhonda latch cecilia : ℕ), 
    boudreaux = 12 ∧
    rhonda = boudreaux / 2 ∧
    latch = 3 * rhonda - 4 ∧
    cecilia = latch + 3 ∧
    total_necklaces_caught boudreaux rhonda latch cecilia = 49 ∧
    (total_necklaces_caught boudreaux rhonda latch cecilia) % 7 = 0 :=
by
  sorry

end necklaces_caught_l207_207233


namespace minimize_y_l207_207598

theorem minimize_y {a b x : ℝ} (min_x : x = (a + b) / 2) : 
  ∀ y : ℝ, y = 3 * (x - a) ^ 2 + 3 * (x - b) ^ 2 → x = (a + b) / 2 :=
begin
  sorry
end

end minimize_y_l207_207598


namespace number_of_solutions_l207_207251

theorem number_of_solutions : 
  (∃ x : ℝ, 8^(x^2 - 6*x + 9) = 1 ∧ x > 0) ∧ 
  (∀ x₁ x₂ : ℝ, 8^(x₁^2 - 6*x₁ + 9) = 1 ∧ x₁ > 0 ∧ 
  8^(x₂^2 - 6*x₂ + 9) = 1 ∧ x₂ > 0 → x₁ = x₂) :=
by
  sorry

end number_of_solutions_l207_207251


namespace ellipse_foci_y_axis_range_l207_207353

theorem ellipse_foci_y_axis_range (k : ℝ) :
  (∃ x y : ℝ, x^2 + k * y^2 = 4 ∧ (∃ c1 c2 : ℝ, y = 0 → c1^2 + c2^2 = 4)) ↔ 0 < k ∧ k < 1 :=
by
  sorry

end ellipse_foci_y_axis_range_l207_207353


namespace multiple_of_3804_l207_207423

theorem multiple_of_3804 (n : ℕ) (hn : 0 < n) : 
  ∃ k : ℕ, (n^3 - n) * (5^(8*n+4) + 3^(4*n+2)) = k * 3804 :=
by
  sorry

end multiple_of_3804_l207_207423


namespace greatest_a_solution_l207_207248

noncomputable def find_greatest_a : ℝ :=
  let a := sqrt ((5 + sqrt 10) / 2) in
  a

theorem greatest_a_solution :
  (5 * sqrt ((2 * find_greatest_a) ^ 2 + 1 ^ 2) - 4 * find_greatest_a ^ 2 - 1) / (sqrt (1 + 4 * find_greatest_a ^ 2) + 3) = 3 :=
by
  sorry

end greatest_a_solution_l207_207248


namespace circle_area_isosceles_triangle_l207_207531

theorem circle_area_isosceles_triangle : 
  ∀ (A B C : Type) (AB AC : Type) (a b c : ℝ),
  a = 5 →
  b = 5 →
  c = 4 →
  isosceles_triangle A B C a b c →
  circle_passes_through_vertices A B C →
  ∃ (r : ℝ), 
    area_of_circle_passing_through_vertices A B C = (15625 * π) / 1764 :=
by intros A B C AB AC a b c ha hb hc ht hcirc
   sorry

end circle_area_isosceles_triangle_l207_207531


namespace simplify_fraction_l207_207436

theorem simplify_fraction :
  (4^5 + 4^3) / (4^4 - 4^2 - 4) = 272 / 59 :=
by
  sorry

end simplify_fraction_l207_207436


namespace law_school_student_count_l207_207159

theorem law_school_student_count 
    (business_students : ℕ)
    (sibling_pairs : ℕ)
    (selection_probability : ℚ)
    (L : ℕ)
    (h1 : business_students = 500)
    (h2 : sibling_pairs = 30)
    (h3 : selection_probability = 7.500000000000001e-5) :
    L = 8000 :=
by
  sorry

end law_school_student_count_l207_207159


namespace angle_A_in_triangle_max_area_of_triangle_l207_207357

theorem angle_A_in_triangle (a b c : ℝ) (A B : ℝ) (h1 : cos A * (2 * c + b) + a * cos B = 0) (h2 : A ∈ (0, π)) : A = 2 * π / 3 :=
by
  sorry

theorem max_area_of_triangle (a b c : ℝ) (A : ℝ) (h1 : A = 2 * π / 3) (h2 : a = 4 * sqrt 3) (h3 : 48 = b^2 + c^2 + b * c) : 
  ½ * b * c * sin A ≤ 4 * sqrt 3 :=
by
  sorry

end angle_A_in_triangle_max_area_of_triangle_l207_207357


namespace garden_area_increase_l207_207962

-- Definitions corresponding to the conditions
def length := 40
def width := 20
def original_perimeter := 2 * (length + width)

-- Definition of the correct answer calculation
def original_area := length * width
def side_length := original_perimeter / 4
def new_area := side_length * side_length
def area_increase := new_area - original_area

-- The statement to be proven
theorem garden_area_increase : area_increase = 100 :=
by sorry

end garden_area_increase_l207_207962


namespace largest_prime_factor_of_12321_l207_207635

theorem largest_prime_factor_of_12321 : ∃ p : ℕ, prime p ∧ p = 43 ∧ (∀ q : ℕ, prime q ∧ q ∣ 12321 → q ≤ p) :=
by
  sorry

end largest_prime_factor_of_12321_l207_207635


namespace find_m_l207_207551

-- Define the interval
def interval := {x | 0 ≤ x ∧ x ≤ 2}

-- Define the event condition
def event_condition (x : ℝ) (m : ℝ) : Prop := 3 * x - m < 0

-- Define the probability given the interval
noncomputable def event_probability (m : ℝ) : ℝ :=
  (measure_theory.measure_space.volume {x ∈ interval | event_condition x m}).to_real / 2

-- The proof statement
theorem find_m : ∃ (m : ℝ), event_probability m = 1 / 6 := sorry

end find_m_l207_207551


namespace sum_of_alternating_sums_n_10_l207_207665

def alternating_sum (S : List ℕ) : ℕ :=
  (List.reverse (List.sort (≤) S)).foldr (λ x (acc: ℕ × ℕ) => (acc.2, acc.1 - x)) (0, 0) |>.fst

def all_subsets (n : ℕ) : List (List ℕ) :=
  List.range (n + 1) >>= List.subsets'

def total_alternating_sum (n : ℕ) : ℕ :=
  (all_subsets n).filter (λ S => S ≠ []).map alternating_sum |> List.sum

theorem sum_of_alternating_sums_n_10 :
  total_alternating_sum 10 = 5120 :=
by
  sorry

end sum_of_alternating_sums_n_10_l207_207665


namespace inequality_holds_l207_207062

theorem inequality_holds : ∀ (n : ℕ), (n - 1)^(n + 1) * (n + 1)^(n - 1) < n^(2 * n) :=
by sorry

end inequality_holds_l207_207062


namespace maia_client_requests_l207_207050

theorem maia_client_requests (x : ℕ) 
  (H1 : ∀ (n : ℕ), (n ≤ 5) → (client_requests n = x * n - 4 * n))
  (H2 : client_requests 5 = 10) : x = 6 := by 
  sorry

end maia_client_requests_l207_207050


namespace part1_part2_l207_207784

section Problem

-- Definitions
def m : ℝ × ℝ := (Real.sqrt 2 / 2, -Real.sqrt 2 / 2)
def n (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)
def x_set : Set ℝ := {x | 0 < x ∧ x < Real.pi / 2}

-- Part 1: Prove that if \overrightarrow{m} ⊥ \overrightarrow{n}, then \tan x = 1
theorem part1 (x : ℝ) (h₁ : x ∈ x_set) (h₂ : m.1 * (n x).1 + m.2 * (n x).2 = 0) : Real.tan x = 1 :=
  sorry

-- Part 2: Prove that if the angle between \overrightarrow{m} and \overrightarrow{n} is \(\frac{π}{3}\), then x = \(\frac{5π}{12}\)
theorem part2 (x : ℝ) (h₁ : x ∈ x_set) (h₂ : m.1 * (n x).1 + m.2 * (n x).2 = 1 / 2) : x = 5 * Real.pi / 12 :=
  sorry

end Problem

end part1_part2_l207_207784


namespace determinant_of_combined_transformation_l207_207034

theorem determinant_of_combined_transformation :
  let θ := 45 * Real.pi / 180
  let R : Matrix (Fin 2) (Fin 2) ℝ :=
    ![![Real.cos θ, -Real.sin θ], ![Real.sin θ, Real.cos θ]]
  let T : Matrix (Fin 2) (Fin 2) ℝ :=
    ![![2, 0], ![0, 2]]
  let S := T.mul R
  Matrix.det S = 4 := by
  sorry

end determinant_of_combined_transformation_l207_207034


namespace volume_of_ellipse_is_16pi_l207_207108

noncomputable def volume_of_ellipse_revolution : ℝ :=
  2 * ∫ x in 0..3, 4 * π * (1 - x^2 / 9)

theorem volume_of_ellipse_is_16pi :
  volume_of_ellipse_revolution = 16 * π :=
by
  -- Proof will go here
  sorry

end volume_of_ellipse_is_16pi_l207_207108


namespace solution_set_of_absolute_value_inequality_l207_207011

theorem solution_set_of_absolute_value_inequality (x : ℝ) :
  (||x - 2| - 1| ≤ 1) ↔ (0 ≤ x ∧ x ≤ 4) := by
sorry

end solution_set_of_absolute_value_inequality_l207_207011


namespace students_in_group_B_l207_207112

theorem students_in_group_B (B : ℕ) : 
  (let A := 20 in
   let forgot_A := 0.20 * 20 in
   let forgot_B := 0.15 * B in
   let total_students := 20 + B in
   let total_forgot := 0.16 * total_students in
   4 + 0.15 * B = 0.16 * (20 + B) →
   B = 80) :=
sorry

end students_in_group_B_l207_207112


namespace infinite_set_P_l207_207558

-- Define the condition as given in the problem
def has_property_P (P : Set ℕ) : Prop :=
  ∀ k : ℕ, k > 0 → (∀ p : ℕ, p.Prime → p ∣ k^3 + 6 → p ∈ P)

-- State the proof problem
theorem infinite_set_P (P : Set ℕ) (h : has_property_P P) : ∃ p : ℕ, p ∉ P → false :=
by
  -- The statement asserts that the set P described by has_property_P is infinite.
  sorry

end infinite_set_P_l207_207558


namespace number_of_slices_per_pizza_l207_207387

-- Given conditions as definitions in Lean 4
def total_pizzas := 2
def total_slices_per_pizza (S : ℕ) : ℕ := total_pizzas * S
def james_portion : ℚ := 2 / 3
def james_ate_slices (S : ℕ) : ℚ := james_portion * (total_slices_per_pizza S)
def james_ate_exactly := 8

-- The main theorem to prove
theorem number_of_slices_per_pizza (S : ℕ) (h : james_ate_slices S = james_ate_exactly) : S = 6 :=
sorry

end number_of_slices_per_pizza_l207_207387


namespace find_x_plus_inv_x_l207_207703

theorem find_x_plus_inv_x (x : ℝ) (h : x^3 + (1/x)^3 = 110) : x + (1/x) = 5 :=
sorry

end find_x_plus_inv_x_l207_207703


namespace unknown_cube_edge_length_l207_207454

noncomputable def volume_of_cube (a : ℝ) : ℝ := a ^ 3

theorem unknown_cube_edge_length :
  ∀ (a b c x : ℝ),
    volume_of_cube a = 512 ∧ 
    volume_of_cube b = 1000 ∧ 
    volume_of_cube c = 1728 ∧ 
    c^3 = a^3 + b^3 + x^3 → 
    x = 6 :=
by
  intro a b c x
  intro h
  have h1 : a^3 = 8^3 := h.1
  have h2 : b^3 = 10^3 := h.2.1
  have h3 : c^3 = 12^3 := h.2.2.1
  have h4 : 12^3 = 8^3 + 10^3 + x^3 := h.2.2.2
  sorry

end unknown_cube_edge_length_l207_207454


namespace limit_of_f_at_1_l207_207462

noncomputable def f (x : ℝ) : ℝ := (x^3 - 1) / (x - 1)

theorem limit_of_f_at_1 :
  filter.tendsto f (nhds 1) (nhds 3) :=
by
  sorry

end limit_of_f_at_1_l207_207462


namespace ending_number_of_range_l207_207899

theorem ending_number_of_range : 
  ∃ n, (∃ (count : ℤ), count = 13.5) ∧ 
       (∃ k, k ≥ 100 ∧ (k / 8) ∈ ℤ) ∧ 
       (n - k) / 8 = 13.5 → 
  n = 204 :=
by 
  sorry  -- skipping proof

end ending_number_of_range_l207_207899


namespace subset_a1_a3_is_5_nat_211_is_subset_a1_a2_a5_a7_a8_l207_207868

def E : Set (Fin 10) := {i | i < 10}

def subset_to_nat (s : Set (Fin 10)) : Nat :=
  s.to_finset.fold (λ acc i => acc + (2^(i.val))) 0 id

def nat_to_subset (k : Nat) : Set (Fin 10) :=
  {i | i < 10 ∧ (k / 2^i) % 2 = 1}

theorem subset_a1_a3_is_5 : subset_to_nat {1, 3} = 5 := 
  by 
    -- Example calculation: 2^0 + 2^2
    sorry

theorem nat_211_is_subset_a1_a2_a5_a7_a8 : nat_to_subset 211 = {1, 2, 5, 7, 8} :=
  by 
    -- Example calculation to verify
    sorry

end subset_a1_a3_is_5_nat_211_is_subset_a1_a2_a5_a7_a8_l207_207868


namespace find_even_and_monotonically_decreasing_function_l207_207502

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_monotonically_decreasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f x ≥ f y

noncomputable def f1 : ℝ → ℝ := λ x, (x - 2) ^ 2
noncomputable def f2 : ℝ → ℝ := λ x, real.log (abs x)
noncomputable def f3 : ℝ → ℝ := λ x, x * real.cos x
noncomputable def f4 : ℝ → ℝ := λ x, real.exp (-real.abs x)

theorem find_even_and_monotonically_decreasing_function :
  (is_even f1 ∧ is_monotonically_decreasing_on f1 (set.Ioi 0)) ∨
  (is_even f2 ∧ is_monotonically_decreasing_on f2 (set.Ioi 0)) ∨
  (is_even f3 ∧ is_monotonically_decreasing_on f3 (set.Ioi 0)) ∨
  (is_even f4 ∧ is_monotonically_decreasing_on f4 (set.Ioi 0)) ↔ 
  (is_even f4 ∧ is_monotonically_decreasing_on f4 (set.Ioi 0)) :=
by sorry

end find_even_and_monotonically_decreasing_function_l207_207502


namespace largest_prime_factor_of_12321_l207_207637

theorem largest_prime_factor_of_12321 : ∃ p : ℕ, prime p ∧ p = 43 ∧ (∀ q : ℕ, prime q ∧ q ∣ 12321 → q ≤ p) :=
by
  sorry

end largest_prime_factor_of_12321_l207_207637


namespace sean_whistles_l207_207433

def charles_whistles : ℕ := 128
def sean_more_whistles : ℕ := 95

theorem sean_whistles : charles_whistles + sean_more_whistles = 223 :=
by {
  sorry
}

end sean_whistles_l207_207433


namespace find_point_C_on_z_axis_l207_207377

noncomputable def point_c_condition (C : ℝ × ℝ × ℝ) (A B : ℝ × ℝ × ℝ) : Prop :=
  dist C A = dist C B

theorem find_point_C_on_z_axis :
  ∃ C : ℝ × ℝ × ℝ, C = (0, 0, 1) ∧ point_c_condition C (1, 0, 2) (1, 1, 1) :=
by
  use (0, 0, 1)
  simp [point_c_condition]
  sorry

end find_point_C_on_z_axis_l207_207377


namespace rabbit_excursion_time_l207_207102

theorem rabbit_excursion_time 
  (line_length : ℝ := 40) 
  (line_speed : ℝ := 3) 
  (rabbit_speed : ℝ := 5) : 
  -- The time calculated for the rabbit to return is 25 seconds
  (line_length / (rabbit_speed - line_speed) + line_length / (rabbit_speed + line_speed)) = 25 :=
by
  -- Placeholder for the proof, to be filled in with a detailed proof later
  sorry

end rabbit_excursion_time_l207_207102


namespace find_a_l207_207886

theorem find_a 
  (a : ℝ) 
  (h1 : ∀ x : ℝ, (x - 3) ^ 2 + 5 = a * x^2 + bx + c) 
  (h2 : (3, 5) = (3, a * 3 ^ 2 + b * 3 + c))
  (h3 : (-2, -20) = (-2, a * (-2)^2 + b * (-2) + c)) : a = -1 :=
by
  sorry

end find_a_l207_207886


namespace sulfuric_acid_moles_l207_207250

def reaction_equation (moles_SO₂ moles_H₂O₂ : ℕ) : ℕ :=
  if moles_SO₂ = moles_H₂O₂ then moles_SO₂ else 0

theorem sulfuric_acid_moles (moles_SO₂ moles_H₂O₂ : ℕ) (h1 : moles_SO₂ = 2) (h2 : moles_H₂O₂ = 2) :
  reaction_equation moles_SO₂ moles_H₂O₂ = 2 :=
by
  unfold reaction_equation
  rw [h1, h2]
  simp
  sorry

end sulfuric_acid_moles_l207_207250


namespace problem_bounds_l207_207032

noncomputable def subset_product_log_bounds : Prop :=
  let n := 2012
  let Q := ∏ (s : Finset (Fin n)) in s.card > 0, s.card
  let M := Real.log2 (Real.log2 Q)
  2014 < M ∧ M < 2016

theorem problem_bounds : subset_product_log_bounds :=
by
  sorry

end problem_bounds_l207_207032


namespace find_abc_intervals_of_monotonicity_l207_207302

def f (x : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) : ℝ := x^3 + a * x^2 + b * x + c
def g (x : ℝ) : ℝ := 12 * x - 4
def h (x : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) : ℝ := f x a b c - g x

theorem find_abc (a b c : ℝ) :
  f (-1) a b c = 0 ∧
  f 1 a b c = 8 ∧
  (deriv (λ x, f x a b c)) 1 = 12 → 
  a = 3 ∧ b = 3 ∧ c = 1 :=
sorry

theorem intervals_of_monotonicity (a b c : ℝ) :
  a = 3 ∧ b = 3 ∧ c = 1 → 
  {I : set ℝ | ∀ x ∈ I, (deriv (λ x, h x a b c)) x > 0} = {t | t < -3} ∪ {t | 1 < t} ∧
  {I : set ℝ | ∀ x ∈ I, (deriv (λ x, h x a b c)) x < 0} = {t | -3 < t ∧ t < 1} :=
sorry

end find_abc_intervals_of_monotonicity_l207_207302


namespace expected_value_sum_marbles_l207_207340

theorem expected_value_sum_marbles :
  let marbles := {1, 2, 3, 4, 5, 6} in
  let combinations := {S | S ⊆ marbles ∧ S.size = 3} in
  let summed_values := {sum S | S ∈ combinations} in
  let total_sum := summed_values.sum in
  let number_of_combinations := combinations.card in
  (total_sum: ℚ) / number_of_combinations = 10.5 :=
sorry

end expected_value_sum_marbles_l207_207340


namespace part_a_part_b_part_c_l207_207058

-- Define the problem conditions
variables (F1 F2 F3 : Type) 
variables (A1 B1 A2 B2 A3 B3 : Type)
variables (U V O : Type)
variables (D1 D2 D3 D1' D2' D3' : Type)
variables (triangle_similar : F1 -> F2 -> F3 -> Prop)
variables (segments_parallel : A1 -> B1 -> A2 -> B2 -> A3 -> B3 -> Prop)
variables (similarity_circle : F1 -> F2 -> F3 -> Type)

-- Part (a)
theorem part_a (F1_sim : triangle_similar F1 F2 F3)
(A1B1_par : segments_parallel A1 B1 A2 B2 A3 B3)
(U_on_circle : similarity_circle F1 F2 F3 U) :
  ∃ (U : Type), ∀ D1 D2 D3 O1 O2 O3, 
  (D1O1, D2O2, D3O3) meet at U → U_on_circle :=
sorry

-- Part (b)
theorem part_b (F1_sim : triangle_similar F1 F2 F3)
(A1B1_par : segments_parallel A1 B1 A2 B2 A3 B3)
(V_on_circle : similarity_circle F1 F2 F3 V) :
  ∃ (V : Type), (circumcircle A1 A2 D3, circumcircle A1 A3 D2, circumcircle A2 A3 D1) 
  intersect at V → V_on_circle :=
sorry

-- Part (c)
theorem part_c (F1_sim : triangle_similar F1 F2 F3)
(A1B1_par : segments_parallel A1 B1 A2 B2 A3 B3)
(O_on_circle : similarity_circle F1 F2 F3 O) :
  ∃ (O : Type), ∀ (D1' D2' D3' : Type), 
  triangle_similar D1 D2 D3 D1' D2' D3' → O_on_circle :=
sorry

end part_a_part_b_part_c_l207_207058


namespace function_odd_and_increasing_l207_207089

def f (x : ℝ) : ℝ := 3^x - 3^(-x)

theorem function_odd_and_increasing : 
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x y : ℝ, x < y → f x < f y) :=
by 
  sorry

end function_odd_and_increasing_l207_207089


namespace brick_weight_l207_207663

theorem brick_weight (b s : ℕ) (h1 : 5 * b = 4 * s) (h2 : 2 * s = 80) : b = 32 :=
by {
  sorry
}

end brick_weight_l207_207663


namespace exist_two_balancing_lines_l207_207521

def is_balancing_line (pts : List (ℝ × ℝ)) (n : ℕ) (l : Line) : Prop :=
  let (blues, reds) := (pts.filter (λ p => p.color = Color.Blue), 
                        pts.filter (λ p => p.color = Color.Red))
  let (half1, half2) := l.split_plane pts
  (∃ b ∈ blues, ∃ r ∈ reds, l.passes_through b r) ∧ 
  (half1.filter (λ p => p.color = Color.Blue)).length = (half1.filter (λ p => p.color = Color.Red)).length ∧ 
  (half2.filter (λ p => p.color = Color.Blue)).length = (half2.filter (λ p => p.color = Color.Red)).length

theorem exist_two_balancing_lines (n : ℕ) (h : n > 1) 
  (pts : List (ℝ × ℝ)) (h1 : pts.length = 2 * n) 
  (h2 : ∀ (p1 p2 p3 : (ℝ × ℝ)), p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 → 
    ¬ (collinear p1 p2 p3))
  (h3 : (pts.filter (λ p => p.color = Color.Blue)).length = n)
  (h4 : (pts.filter (λ p => p.color = Color.Red)).length = n) : 
  ∃ l1 l2 : Line, l1 ≠ l2 ∧ is_balancing_line pts n l1 ∧ is_balancing_line pts n l2 :=
sorry


end exist_two_balancing_lines_l207_207521


namespace expected_value_sum_of_three_marbles_l207_207338

-- Definitions
def marbles : List ℕ := [1, 2, 3, 4, 5, 6]
def combinations (n k : ℕ) : ℕ := Nat.choose n k

def sum_combinations (combs : List (List ℕ)) : ℕ :=
  combs.map List.sum |>.sum

-- Main theorem to prove
theorem expected_value_sum_of_three_marbles : 
  let combs := (marbles.combination 3) in
  combinations 6 3 = 20 → 
  sum_combinations combs = 210 →
  (sum_combinations combs) / (combinations 6 3) = 10.5 :=
by
  sorry

end expected_value_sum_of_three_marbles_l207_207338


namespace apples_on_tree_now_l207_207066

-- Definitions based on conditions
def initial_apples : ℕ := 11
def apples_picked : ℕ := 7
def new_apples : ℕ := 2

-- Theorem statement proving the final number of apples on the tree
theorem apples_on_tree_now : initial_apples - apples_picked + new_apples = 6 := 
by 
  sorry

end apples_on_tree_now_l207_207066


namespace geometric_sequence_formula_l207_207169

noncomputable theory

open_locale classical

theorem geometric_sequence_formula :
  ∃ (a : ℕ+ → ℝ), 
    (∀ n : ℕ+, a n > 0) ∧
    (∀ n : ℕ+, a (n + 1) > a n) ∧
    (∀ n : ℕ+, a (n + 1) + (∑ k in finset.range n, a k) < ∑ k in finset.range n, a k) ∧
    (∀ n : ℕ+, a n = - (1 / 2) ^ n) :=
sorry

end geometric_sequence_formula_l207_207169


namespace conveyor_belt_efficiencies_and_min_cost_l207_207573

theorem conveyor_belt_efficiencies_and_min_cost :
  ∃ (efficiency_B efficiency_A : ℝ),
    efficiency_A = 1.5 * efficiency_B ∧
    18000 / efficiency_B - 18000 / efficiency_A = 10 ∧
    efficiency_B = 600 ∧
    efficiency_A = 900 ∧
    ∃ (cost_A cost_B : ℝ),
      cost_A = 8 * 20 ∧
      cost_B = 6 * 30 ∧
      cost_A = 160 ∧
      cost_B = 180 ∧
      cost_A < cost_B :=
by
  sorry

end conveyor_belt_efficiencies_and_min_cost_l207_207573


namespace internal_angle_sine_l207_207345

theorem internal_angle_sine (α : ℝ) (h1 : α > 0 ∧ α < 180) (h2 : Real.sin (α * (Real.pi / 180)) = 1 / 2) : α = 30 ∨ α = 150 :=
sorry

end internal_angle_sine_l207_207345


namespace cyclic_iff_equal_diagonals_convex_l207_207941

structure Hexagon (α : Type*) :=
(A B C D E F : α)

structure ConvexHexagon (α : Type*) (H : Hexagon α) :=
(convex : ∀ (A B C D E F : α), -- details for convex property
 sorry)

structure PairwiseParallel (α : Type*) [inner_product_space ℝ α] (H : Hexagon α) :=
(p1 : inner_product_space.parallel H.A H.D)
(p2 : inner_product_space.parallel H.B H.E)
(p3 : inner_product_space.parallel H.C H.F)

structure EqualDiagonals (α : Type*) [inner_product_space ℝ α] (H : Hexagon α) :=
(eq1 : dist H.A H.D = dist H.B H.E)
(eq2 : dist H.B H.E = dist H.C H.F)

structure Cyclic (α : Type*) [metric_space α] (H : Hexagon α) :=
(inscribed : -- details for cyclic property
 sorry)

theorem cyclic_iff_equal_diagonals_convex {α : Type*} [metric_space α] [inner_product_space ℝ α] {H : Hexagon α}
  (convhex : ConvexHexagon α H) (ppar : PairwiseParallel α H) :
  (Cyclic α H) ↔ (EqualDiagonals α H) :=
sorry

end cyclic_iff_equal_diagonals_convex_l207_207941


namespace elsa_ends_with_145_marbles_l207_207237

theorem elsa_ends_with_145_marbles :
  let initial := 150
  let after_breakfast := initial - 7
  let after_lunch := after_breakfast - 57
  let after_afternoon := after_lunch + 25
  let after_evening := after_afternoon + 85
  let after_exchange := after_evening - 9 + 6
  let final := after_exchange - 48
  final = 145 := by
    sorry

end elsa_ends_with_145_marbles_l207_207237


namespace find_coefficient_c_l207_207230

theorem find_coefficient_c (c : ℚ) :
  (x : ℚ) → (P : ℚ → ℚ) → P x = x^4 + 3*x^3 + c*x^2 + 15*x + 20 → (P 3 = 0 → c = -227/9) :=
by
  sorry

end find_coefficient_c_l207_207230


namespace angle_between_is_pi_div_4_l207_207264

variables {E : Type*} [inner_product_space ℝ E]
variables (a b : E)

def is_unit_vector (v : E) : Prop := ∥v∥ = 1
def is_sqrt2_vector (v : E) : Prop := ∥v∥ = real.sqrt 2
def are_perpendicular (u v : E) : Prop := inner u v = 0
def angle_between (u v : E) : ℝ := real.arccos (inner u v / (∥u∥ * ∥v∥))

noncomputable theory

theorem angle_between_is_pi_div_4
  (ha : is_unit_vector a)
  (hb : is_sqrt2_vector b)
  (h_perp : are_perpendicular a (a - b)) :
  angle_between a b = real.pi / 4 :=
sorry

end angle_between_is_pi_div_4_l207_207264


namespace real_y_iff_x_interval_l207_207443

theorem real_y_iff_x_interval (x : ℝ) :
  (∃ y : ℝ, 3*y^2 + 2*x*y + x + 5 = 0) ↔ (x ≤ -3 ∨ x ≥ 5) :=
by
  sorry

end real_y_iff_x_interval_l207_207443


namespace sufficient_condition_for_min_value_not_necessary_condition_for_min_value_l207_207719

noncomputable def f (x b : ℝ) : ℝ := x^2 + b*x

theorem sufficient_condition_for_min_value (b : ℝ) : b < 0 → ∀ x, min (f (f x b) b) = min (f x b) :=
sorry

theorem not_necessary_condition_for_min_value (b : ℝ) : (b < 0) ∧ (∀ x, min (f (f x b) b) = min (f x b)) → b ≤ 0 ∨ b ≥ 2 := 
sorry

end sufficient_condition_for_min_value_not_necessary_condition_for_min_value_l207_207719


namespace number_of_tshirts_sold_l207_207871

theorem number_of_tshirts_sold 
    (original_price discounted_price revenue : ℕ)
    (discount : ℕ) 
    (no_of_tshirts: ℕ)
    (h1 : original_price = 51)
    (h2 : discount = 8)
    (h3 : discounted_price = original_price - discount)
    (h4 : revenue = 5590)
    (h5 : revenue = no_of_tshirts * discounted_price) : 
    no_of_tshirts = 130 :=
by
  sorry

end number_of_tshirts_sold_l207_207871


namespace sin_angle_DAE_l207_207366

noncomputable def equilateralTriangleABC : Type :=
  {A B C : Point}
  (hA_eq_10 : dist A B = 10)
  (hB_eq_10 : dist B C = 10)
  (hC_eq_10 : dist C A = 10)
  (hEquilateral : ∀ {X Y Z : Point}, equilateral A B C → equilateral X Y Z → dist X Y = dist Y Z)

def bisects (D E : Point) (B C : Point) :=
  D = midpoint B C

def sinAngleDAE (A D E : Point) : ℝ :=
  let area := triangleArea A D E
  let sideLengths := sideLengths A D E
  let (a, b, c) := sideLengths
  2 * area / (a * b)

theorem sin_angle_DAE
  {A B C D E : Point}
  (hEquilateral : equilateralTriangleABC A B C)
  (hBisect : bisects D E B C) :
  sinAngleDAE A D E = $\frac{\sqrt{3}}{18}$ := by
    sorry

end sin_angle_DAE_l207_207366


namespace largest_prime_factor_of_12321_l207_207634

theorem largest_prime_factor_of_12321 : ∃ p : ℕ, prime p ∧ p = 43 ∧ (∀ q : ℕ, prime q ∧ q ∣ 12321 → q ≤ p) :=
by
  sorry

end largest_prime_factor_of_12321_l207_207634


namespace trapezoid_LM_sqrt2_l207_207793

theorem trapezoid_LM_sqrt2 (K L M N P Q : Point) : 
  ∀ (h_trapezoid : is_trapezoid K L M N) 
     (diag_eq_height : distance K M = 1 ∧ height_trapezoid K L M N = 1) 
     (perp_KP_MQ : is_perpendicular(K P MN) ∧ is_perpendicular(M Q KL)) 
     (KN_MQ_eq : distance K N = distance M Q) 
     (LM_MP_eq : distance L M = distance M P), 
  distance L M = Real.sqrt 2 :=
by
  sorry

end trapezoid_LM_sqrt2_l207_207793


namespace find_m_plus_b_l207_207460

noncomputable def point_reflection (P Q : ℝ × ℝ) (m b : ℝ) : Prop :=
  let (x1, y1) := P in
  let (x2, y2) := Q in
  (x2 = x1 + 2 * ((1 / (1 + m^2)) * ((1 - m^2) * x1 + 2*m*y1 + 2*m*b - m*x1)) ∧ 
   y2 = y1 + 2 * ((1 / (1 + m^2)) * ((m^2 - 1) * y1 + 2*m*x1 + 2*b)))

theorem find_m_plus_b : 
  -- Conditions
  ∃ (m b : ℝ), point_reflection (-4, 2) (6, 0) m b ∧
  -- Conclusion
  m + b = 1 :=
sorry

end find_m_plus_b_l207_207460


namespace expected_value_sum_of_three_marbles_l207_207336

-- Definitions
def marbles : List ℕ := [1, 2, 3, 4, 5, 6]
def combinations (n k : ℕ) : ℕ := Nat.choose n k

def sum_combinations (combs : List (List ℕ)) : ℕ :=
  combs.map List.sum |>.sum

-- Main theorem to prove
theorem expected_value_sum_of_three_marbles : 
  let combs := (marbles.combination 3) in
  combinations 6 3 = 20 → 
  sum_combinations combs = 210 →
  (sum_combinations combs) / (combinations 6 3) = 10.5 :=
by
  sorry

end expected_value_sum_of_three_marbles_l207_207336


namespace olympiad_pen_problem_l207_207468

/-- There are 9 pens, and the following conditions hold:
1. Among any four pens, at least two belong to the same person.
2. Among any five pens, no more than three belong to the same person.
Then the number of students is 3, and each student owns 3 pens. -/
theorem olympiad_pen_problem :
  ∃ (students : ℕ) (pens : ℕ → ℕ), students = 3 ∧
  (∀ s, s < students → 2 ≤ pens s ∧ pens s ≤ 3) ∧
  (∀ pens_per_student, (∑ i in finset.range students, pens i) = 9 ) ∧
  (∀ (subset : fin N (set.range 9)), subset.card = 4 → ∃ (student : ℕ), student < students ∧ 2 ≤ ∑ i in subset, pens i) ∧
  (∀ (subset : fin N (set.range 9)), subset.card = 5 → ∀ (student : ℕ), student < students → (∑ i in subset, pens i) ≤ 3) := 
sorry

end olympiad_pen_problem_l207_207468


namespace half_cube_surface_area_correct_l207_207514

-- Define the surface area of the cube and the conditions of the cut
def cube_surface_area (a : ℝ) : ℝ := 6 * a^2

-- Define the retained surface area of each half-cube
def retained_surface_area (a : ℝ) : ℝ := 0.5 * cube_surface_area a

-- Define the side length of the hexagonal face and its area
def hexagon_side_length (a : ℝ) : ℝ := real.sqrt (2 * (a / 2)^2)
def hexagon_area (a : ℝ) : ℝ := (3 * real.sqrt 3 / 2) * (hexagon_side_length a)^2

-- Define the total surface area of each half-cube, incorporating the approximate value
def half_cube_surface_area (a : ℝ) : ℝ := retained_surface_area a + hexagon_area a

-- Convert the value to the nearest integer for the final result
noncomputable def approx_half_cube_surface_area (a : ℝ) : ℝ := real.round (half_cube_surface_area a)

theorem half_cube_surface_area_correct : approx_half_cube_surface_area 4 = 69 := 
by
  -- Proof omitted
  sorry

end half_cube_surface_area_correct_l207_207514


namespace root_in_interval_l207_207914

noncomputable def f (x : ℝ) : ℝ := Real.log x - 6 + 2 * x

theorem root_in_interval : ∃ c ∈ Set.Ioo 2 3, f c = 0 :=
begin
  -- Necessary definitions and conditions follow here
  have f2 : f 2 < 0 := by {
    calc f 2 = Real.log 2 - 6 + 4 : by rw [f]
    ... = Real.log 2 - 2
    ... < 0 : by linarith [Real.log_pos (by linarith)]
  },
  have f3 : f 3 > 0 := by {
    calc f 3 = Real.log 3 - 6 + 6 : by rw [f]
    ... = Real.log 3
    ... > 0 : by exact Real.log_pos (by linarith)
  },
  exact exists_Ioo_of_intermediate_value f2 f3 sorry,
end

end root_in_interval_l207_207914


namespace length_of_bridge_l207_207136

theorem length_of_bridge (t : ℝ) (s : ℝ) (d : ℝ) : 
  (t = 24 / 60) ∧ (s = 10) ∧ (d = s * t) → d = 4 := by
  sorry

end length_of_bridge_l207_207136


namespace number_of_true_propositions_is_three_l207_207299

-- Define the four propositions
def proposition1 : Prop := ¬∀(R : Type) (a b c d : R), true -- False statement for demonstration as rhombus diagonals equality is false
def proposition2 : Prop := ∀ (x y : ℝ), (x * y = 1) → ((1/x = y) ∧ (1/y = x))
def proposition3 : Prop := ∀ (A B : Type), (∃ (a b : A) (c d : B), a = c ∧ b = d) → false -- True statement for demonstration
def proposition4 : Prop := ∀ (a b : ℝ), ((a^2 ≤ b^2) → (a ≤ b))

-- Prove that exactly three of the above propositions are true
theorem number_of_true_propositions_is_three : 
  [proposition1, proposition2, proposition3, proposition4].count(λ p, p) = 3 := 
by 
  sorry

end number_of_true_propositions_is_three_l207_207299


namespace peaches_division_l207_207464

theorem peaches_division (n k r : ℕ) 
  (h₁ : 100 = n * k + 10)
  (h₂ : 1000 = n * k * 11 + r) :
  r = 10 :=
by sorry

end peaches_division_l207_207464


namespace pieces_remaining_l207_207485

theorem pieces_remaining (n : ℕ) (seq : Finₓ 3000 → ℕ) :
  n = 1333 →  ∀ i : Finₓ 1333, seq i = 407 :=
by {
  sorry
}

end pieces_remaining_l207_207485


namespace magnitude_of_u_converse_magnitude_of_u_l207_207271

open Complex

variables {a b z : ℂ}

theorem magnitude_of_u (h1: abs a ≠ abs b)
  (h2: abs z = 1) :
  abs ( (a + b * z) / (conj b + conj a * z) ) = 1 :=
by sorry

theorem converse_magnitude_of_u (h1: abs a ≠ abs b)
  (h2: abs ((a + b * z) / (conj b + conj a * z)) = 1) :
  abs z = 1 :=
by sorry

end magnitude_of_u_converse_magnitude_of_u_l207_207271


namespace zero_primes_divisible_by_46_l207_207330

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem zero_primes_divisible_by_46 : 
    ∀ p : ℕ, is_prime p → 46 ∣ p → p = 0 :=
by
  intro p hp h46d
  have h46 : 46 = 2 * 23 := by norm_num
  have hdiv2 : 2 ∣ p := dvd_of_dvd_mul_right (by norm_num : 2 ∣ 46)
  have hdiv23 : 23 ∣ p := dvd_of_dvd_mul_left (by norm_num : 23 ∣ 46)
  have hpdv2 : p = 2 := sorry
  have hpdv23 : p = 23 := sorry
  have h_incon : 2 ≠ 23 := by norm_num
  contradiction

end zero_primes_divisible_by_46_l207_207330


namespace geometric_seq_sum_of_seq_l207_207515

noncomputable def a_seq : ℕ → ℚ
| 0       := 1/4  -- for convenience, redefine a_1 as a_seq 0
| 1       := 3/4
| (n + 2) := 2 * a_seq (n + 1) - a_seq n

noncomputable def b_seq : ℕ → ℚ
| 0       := 11/12  -- for convenience, redefine b_1 as b_seq 0
| 1       := arbitrary ℚ  -- impose condition b_1 ≠ 1/4
| (n + 2) := (b_seq (n + 1) + n.succ) / 3

def S (n : ℕ) := ∑ k in Finset.range (n + 1), b_seq k

theorem geometric_seq (n : ℕ) (hn : n ≥ 2) :
  ∃ r : ℚ, (b_seq n - a_seq n) = r * (b_seq (n - 1) - a_seq (n - 1)) :=
sorry

theorem sum_of_seq (n : ℕ) :
  S n = (1/4 : ℚ) * n^2 + 2/3 - 2 / 3^n :=
sorry

end geometric_seq_sum_of_seq_l207_207515


namespace new_socks_bought_l207_207056

-- Given conditions:
def initial_socks : ℕ := 11
def socks_thrown_away : ℕ := 4
def final_socks : ℕ := 33

-- Theorem proof statement:
theorem new_socks_bought : (final_socks - (initial_socks - socks_thrown_away)) = 26 :=
by
  sorry

end new_socks_bought_l207_207056


namespace rotating_isosceles_trapezoid_results_l207_207432

-- Definitions for the problem conditions
def isosceles_trapezoid (a b : ℝ) := 
  ∃ h : ℝ, h > 0 ∧ a ≠ b ∧ a > 0 ∧ b > 0

def rotate_around_base (shape : Type) (axis : Type) : Type :=
sorry -- Detailed definition is omitted for the purpose of this example

-- The mathematically equivalent Lean statement for the problem
theorem rotating_isosceles_trapezoid_results (a b : ℝ) (h : a ≠ b) (ha : a > 0) (hb : b > 0) :
  let trapezoid := isosceles_trapezoid a b in
  rotate_around_base trapezoid (a, b) = "one cylinder and two cones" :=
sorry

end rotating_isosceles_trapezoid_results_l207_207432


namespace find_length_of_AB_l207_207767

open Real

theorem find_length_of_AB (A B C : ℝ) 
    (h1 : tan A = 3 / 4) 
    (h2 : B = 6) 
    (h3 : C = π / 2) : sqrt (B^2 + ((3/4) * B)^2) = 7.5 :=
by
  sorry

end find_length_of_AB_l207_207767


namespace proof_sin_sum_ineq_proof_sin_product_ineq_proof_cos_sum_double_ineq_proof_cos_square_sum_ineq_proof_cos_half_product_ineq_proof_cos_product_ineq_l207_207384

noncomputable def sin_sum_ineq (A B C : ℝ) (hABC : A + B + C = π) : Prop := 
  (Real.sin A + Real.sin B + Real.sin C) ≤ (3 / 2) * Real.sqrt 3

noncomputable def sin_product_ineq (A B C : ℝ) (hABC : A + B + C = π) : Prop := 
  (Real.sin A * Real.sin B * Real.sin C) ≤ (3 / 8) * Real.sqrt 3

noncomputable def cos_sum_double_ineq (A B C : ℝ) (hABC : A + B + C = π) : Prop := 
  (Real.cos (2 * A) + Real.cos (2 * B) + Real.cos (2 * C)) ≥ (-3 / 2)

noncomputable def cos_square_sum_ineq (A B C : ℝ) (hABC : A + B + C = π) : Prop := 
  (Real.cos A ^ 2 + Real.cos B ^ 2 + Real.cos C ^ 2) ≥ (3 / 4)

noncomputable def cos_half_product_ineq (A B C : ℝ) (hABC : A + B + C = π) : Prop := 
  (Real.cos (A / 2) * Real.cos (B / 2) * Real.cos (C / 2)) ≤ (3 / 8) * Real.sqrt 3

noncomputable def cos_product_ineq (A B C : ℝ) (hABC : A + B + C = π) : Prop := 
  (Real.cos A * Real.cos B * Real.cos C) ≤ (1 / 8)

theorem proof_sin_sum_ineq {A B C : ℝ} (hABC : A + B + C = π) : sin_sum_ineq A B C hABC := sorry

theorem proof_sin_product_ineq {A B C : ℝ} (hABC : A + B + C = π) : sin_product_ineq A B C hABC := sorry

theorem proof_cos_sum_double_ineq {A B C : ℝ} (hABC : A + B + C = π) : cos_sum_double_ineq A B C hABC := sorry

theorem proof_cos_square_sum_ineq {A B C : ℝ} (hABC : A + B + C = π) : cos_square_sum_ineq A B C hABC := sorry

theorem proof_cos_half_product_ineq {A B C : ℝ} (hABC : A + B + C = π) : cos_half_product_ineq A B C hABC := sorry

theorem proof_cos_product_ineq {A B C : ℝ} (hABC : A + B + C = π) : cos_product_ineq A B C hABC := sorry

end proof_sin_sum_ineq_proof_sin_product_ineq_proof_cos_sum_double_ineq_proof_cos_square_sum_ineq_proof_cos_half_product_ineq_proof_cos_product_ineq_l207_207384


namespace tickets_per_friend_l207_207055

-- Defining the conditions
def initial_tickets := 11
def remaining_tickets := 3
def friends := 4

-- Statement to prove
theorem tickets_per_friend (h_tickets_given : initial_tickets - remaining_tickets = 8) : (initial_tickets - remaining_tickets) / friends = 2 :=
by
  sorry

end tickets_per_friend_l207_207055


namespace find_k_l207_207153

variable (k : ℕ) (hk : k > 0)

theorem find_k (h : (24 - k) / (8 + k) = 1) : k = 8 :=
by sorry

end find_k_l207_207153


namespace equivalent_problem_l207_207936

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then x^2 else sorry

theorem equivalent_problem 
  (f_odd : ∀ x : ℝ, f (-x) = -f x)
  (f_periodic : ∀ x : ℝ, f (x + 2) = f x)
  (f_interval : ∀ x : ℝ, 0 < x ∧ x < 1 → f x = x^2)
  : f (-3/2) + f 1 = 3/4 :=
sorry

end equivalent_problem_l207_207936


namespace find_abcde_l207_207262

noncomputable def find_five_digit_number (a b c d e : ℕ) : ℕ :=
  10000 * a + 1000 * b + 100 * c + 10 * d + e

theorem find_abcde
  (a b c d e : ℕ)
  (h1 : 0 ≤ a ∧ a ≤ 9)
  (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : 0 ≤ c ∧ c ≤ 9)
  (h4 : 0 ≤ d ∧ d ≤ 9)
  (h5 : 0 ≤ e ∧ e ≤ 9)
  (h6 : a ≠ 0)
  (h7 : (10 * a + b + 10 * b + c) * (10 * b + c + 10 * c + d) * (10 * c + d + 10 * d + e) = 157605) :
  find_five_digit_number a b c d e = 12345 ∨ find_five_digit_number a b c d e = 21436 :=
sorry

end find_abcde_l207_207262


namespace volume_of_inscribed_tetrahedron_l207_207859

theorem volume_of_inscribed_tetrahedron (r h : ℝ) (V : ℝ) (tetrahedron_inscribed : Prop) 
  (cylinder_condition : π * r^2 * h = 1) 
  (inscribed : tetrahedron_inscribed → True) : 
  V ≤ 2 / (3 * π) :=
sorry

end volume_of_inscribed_tetrahedron_l207_207859


namespace abs_diff_m_n_l207_207855

theorem abs_diff_m_n : 
  ∀ (m n : ℝ), 
  ∃ (θ : ℝ), θ = π / 3 ∧
  (cos θ, m) ∈ (λ (x y : ℝ), y^2 + x^6 = 3 * x^2 * y + x) ∧
  (cos θ, n) ∈ (λ (x y : ℝ), y^2 + x^6 = 3 * x^2 * y + x) ∧
  |m - n| = 5 / 2 :=
begin
  sorry
end

end abs_diff_m_n_l207_207855


namespace largest_prime_factor_12321_l207_207648

theorem largest_prime_factor_12321 : 
  ∃ p : ℕ, p.prime ∧ p ∣ 12321 ∧ ∀ q : ℕ, q.prime ∧ q ∣ 12321 → q ≤ p :=
sorry

end largest_prime_factor_12321_l207_207648


namespace percentage_difference_l207_207510

theorem percentage_difference :
  let a1 := 0.12 * 24.2
  let a2 := 0.10 * 14.2
  a1 - a2 = 1.484 := 
by
  -- Definitions
  let a1 := 0.12 * 24.2
  let a2 := 0.10 * 14.2
  -- Proof body (skipped for this task)
  sorry

end percentage_difference_l207_207510


namespace largest_prime_factor_12321_l207_207647

theorem largest_prime_factor_12321 : 
  ∃ p : ℕ, p.prime ∧ p ∣ 12321 ∧ ∀ q : ℕ, q.prime ∧ q ∣ 12321 → q ≤ p :=
sorry

end largest_prime_factor_12321_l207_207647


namespace y_pow_x_eq_nine_l207_207752

theorem y_pow_x_eq_nine (x y : ℝ) (h : x^2 + y^2 - 4 * x + 6 * y + 13 = 0) : y^x = 9 := by
  sorry

end y_pow_x_eq_nine_l207_207752


namespace omega_value_monotonicity_f_l207_207713

noncomputable def f (ω x : ℝ) := 4 * cos (ω * x) * cos (ω * x + π / 3)

theorem omega_value (ω : ℝ) (hω : ω > 0) (hT : ∀ T > 0, (∀ x, f ω (x + T) = f ω x) ↔ T = π):
  ω = 1 :=
sorry

theorem monotonicity_f (ω : ℝ) (hω : ω = 1):
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ π / 3 → f 1 x > f 1 (x + ε x)) ∧
  (∀ x : ℝ, π / 3 ≤ x ∧ x ≤ 5 * π / 6 → f 1 x < f 1 (x + ε x)) :=
sorry

end omega_value_monotonicity_f_l207_207713


namespace coef_x9_expansion_correct_l207_207987

noncomputable theory

def coef_x9_expansion : ℚ :=
  let poly := (2 + 3 * x - 2 * x^3) ^ 5 in
  let target_term := x^9 in
  -- Calculation omitted, final result plugged directly
  1080

theorem coef_x9_expansion_correct : coef_x9_expansion = 1080 :=
sorry

end coef_x9_expansion_correct_l207_207987


namespace maximum_value_expression_l207_207655

theorem maximum_value_expression :
  (∀ x y : Real, 
     -10.4168 * (3 + 2 * Real.sqrt (7 - Real.sqrt 2) * Real.cos y - Real.cos (2 * y)) ≤ 
     (\(Real.sqrt (3 - Real.sqrt 2) * Real.sin x - Real.sqrt (2 * (1 + Real.cos (2 * x))) - 
       1) * (3 + 2 * Real.sqrt (7 - Real.sqrt 2) * Real.cos y - Real.cos (2 * y))) ≤ 9) :=
by
  sorry

end maximum_value_expression_l207_207655


namespace ashley_champagne_bottles_l207_207582

theorem ashley_champagne_bottles (guests : ℕ) (glasses_per_guest : ℕ) (servings_per_bottle : ℕ) 
  (h1 : guests = 120) (h2 : glasses_per_guest = 2) (h3 : servings_per_bottle = 6) : 
  (guests * glasses_per_guest) / servings_per_bottle = 40 :=
by
  -- The proof will go here
  sorry

end ashley_champagne_bottles_l207_207582


namespace problem_solution_l207_207940

noncomputable def proofProblem (a b c : ℝ) (f : ℝ → ℝ) (l m n p q r : ℝ) : Prop :=
  (a ≠ 0) ∧ (b ≠ 0) ∧ (c ≠ 0) ∧
  (∀ x, f x = (x - a) * (x - b) * (λ x, 0) + (p * x + l)) ∧
  (∀ x, f x = (x - b) * (x - c) * (λ x, 0) + (q * x + m)) ∧
  (∀ x, f x = (x - c) * (x - a) * (λ x, 0) + (r * x + n))

theorem problem_solution (a b c : ℝ) (f : ℝ → ℝ) (l m n p q r : ℝ) 
  (h : proofProblem a b c f l m n p q r) :
  l * (1/a - 1/b) + m * (1/b - 1/c) + n * (1/c - 1/a) = 0 :=
by 
  sorry

end problem_solution_l207_207940


namespace hexagon_covering_l207_207492

noncomputable def equilateral_triangle_area (s : ℝ) : ℝ := (sqrt 3 / 4) * (s * s)
noncomputable def hexagon_area (s : ℝ) : ℝ := (3 * sqrt 3 / 2) * (s * s)

theorem hexagon_covering :
  let side_length_small_triangle := 2
  let side_length_hexagon := 10
  let area_small_triangle := equilateral_triangle_area side_length_small_triangle
  let area_hexagon := hexagon_area side_length_hexagon
  area_hexagon / area_small_triangle = 150 :=
by
  sorry

end hexagon_covering_l207_207492


namespace min_value_of_xy_l207_207040

open real

noncomputable def min_value (x y : ℝ) : ℝ := x + y

theorem min_value_of_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 9/y = 1) :
  min_value x y = 16 :=
by
  sorry

end min_value_of_xy_l207_207040


namespace problem1_problem2_problem3_l207_207913

-- Problem 1
theorem problem1 (x : ℝ) (h : x^2 + x - 2 = 0) : x^2 + x + 2023 = 2025 := 
  sorry

-- Problem 2
theorem problem2 (a b : ℝ) (h : a + b = 5) : 2 * (a + b) - 4 * a - 4 * b + 21 = 11 := 
  sorry

-- Problem 3
theorem problem3 (a b : ℝ) (h1 : a^2 + 3 * a * b = 20) (h2 : b^2 + 5 * a * b = 8) : 2 * a^2 - b^2 + a * b = 32 := 
  sorry

end problem1_problem2_problem3_l207_207913


namespace find_p_plus_q_l207_207393

-- Definition of the set T
def T := {n : ℤ | ∃ j k : ℕ, 0 ≤ j ∧ j < k ∧ k ≤ 49 ∧ n = 2^j + 2^k}

-- Function to check divisibility by 11
def is_divisible_by_11 (n : ℤ) : Prop := n % 11 = 0

-- Definition of the probability p/q
def probability_divisible_by_11 (T : Set ℤ) : ℚ :=
  let total_numbers := T.card
  let divisible_numbers := (T.filter is_divisible_by_11).card
  (divisible_numbers : ℚ) / total_numbers

theorem find_p_plus_q (p q : ℕ) (h1 : nat.coprime p q)
  (h2 : probability_divisible_by_11 T = (p : ℚ) / q) :
  p + q = 54 :=
sorry

end find_p_plus_q_l207_207393


namespace find_x_l207_207749

theorem find_x (a x : ℝ) (ha : 1 < a) (hx : 0 < x)
  (h : (3 * x)^(Real.log 3 / Real.log a) - (4 * x)^(Real.log 4 / Real.log a) = 0) : 
  x = 1 / 4 := 
by 
  sorry

end find_x_l207_207749


namespace cos_arcsin_half_l207_207215

-- Defining the necessary conditions:
variable θ : ℝ
variable h_sin : Real.sin θ = 1/2
variable h_theta : θ = Real.arcsin (1/2)

-- The proof statement showing the relationship:
theorem cos_arcsin_half : Real.cos (Real.arcsin (1/2)) = (√3) / 2 :=
by
  sorry

end cos_arcsin_half_l207_207215


namespace number_of_subsets_of_M_l207_207094

-- Define the set M
def M : Set ℤ := {x | x - 2 = 0}

-- Define the proof statement
theorem number_of_subsets_of_M : M = {2} ∧ M.finite → 2^(M.toFinset.card) = 2 := by
  sorry

end number_of_subsets_of_M_l207_207094


namespace trapezoid_LM_sqrt2_l207_207800

theorem trapezoid_LM_sqrt2 (K L M N P Q : Type*)
  (KM : ℝ)
  (KN MQ LM MP : ℝ)
  (h_KM : KM = 1)
  (h_KN_MQ : KN = MQ)
  (h_LM_MP : LM = MP) 
  (h_KP_1 : KN = 1) 
  (h_MQ_1 : MQ = 1) :
  LM = Real.sqrt 2 :=
by
  sorry

end trapezoid_LM_sqrt2_l207_207800


namespace area_of_triangle_ABC_l207_207776

variables (A B C : Type) [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C]
variables (angle_BC_right : ∠C = π/2) (angle_BC_equal : ∠B = ∠C) (hypotenuse_AC_length : dist A C = 8 * sqrt 2)

-- noncomputable action_space
theorem area_of_triangle_ABC :
  (1/2) * abs (dist_vector_space.side_AB_BC 8 * dist_vector_space.side_AB_AB 8) = 32 :=
  sorry


end area_of_triangle_ABC_l207_207776


namespace proof_problem_l207_207270

noncomputable def condition_p (a : ℝ) : Prop :=
  ∃ x > 0, exp(x) - a * x < 1

def condition_q (a : ℝ) : Prop :=
  a > 2

theorem proof_problem (a : ℝ) :
  condition_q a → condition_p a ∧ ¬(condition_p a → condition_q a) :=
by 
  sorry

end proof_problem_l207_207270


namespace range_of_k_l207_207307

-- Define the set M
def M := {x : ℝ | -1 ≤ x ∧ x ≤ 7}

-- Define the set N based on k
def N (k : ℝ) := {x : ℝ | k + 1 ≤ x ∧ x ≤ 2 * k - 1}

-- The main statement to prove
theorem range_of_k (k : ℝ) : M ∩ N k = ∅ → 6 < k :=
by
  -- skipping the proof as instructed
  sorry

end range_of_k_l207_207307


namespace largest_prime_factor_12321_l207_207639

theorem largest_prime_factor_12321 : 
  ∃ p : ℕ, prime p ∧ p ∣ 12321 ∧ ∀ q : ℕ, prime q ∧ q ∣ 12321 → q ≤ p :=
begin
  use 83,
  split,
  { -- Prove that 83 is a prime number
    sorry },
  split,
  { -- Prove that 83 divides 12321
    sorry },
  { -- Prove that any other prime factor of 12321 is less than or equal to 83
    sorry }
end

end largest_prime_factor_12321_l207_207639


namespace area_of_rectangle_ABCD_l207_207367

-- Define the given configuration and conditions
variables (A B C D E : Point)
variable (rectangle_ABCD : Rectangle ABCD)
variable (C_bisector : AngleBisector ABCD C E)
variable (E_on_AB : OnLine E (Line AB))
variable (BE_length : Length BE = 8)
variable (AD_length : Length AD = 10)

-- Define the main proof problem
theorem area_of_rectangle_ABCD :
  (area_of_rectangle ABCD = 160) :=
sorry

end area_of_rectangle_ABCD_l207_207367


namespace circle_area_isosceles_triangle_l207_207542

theorem circle_area_isosceles_triangle (a b c : ℝ) (h1 : a = 5) (h2 : b = 5) (h3 : c = 4) :
  let r := ((25 * Real.sqrt 21) / 42)
  in ∃ (O : Point) (r : ℝ), Circle O r ∧ r^2 * Real.pi = (13125 / 1764) * Real.pi := by
  sorry

end circle_area_isosceles_triangle_l207_207542


namespace area_of_circumcircle_of_isosceles_triangle_l207_207538

theorem area_of_circumcircle_of_isosceles_triangle :
  ∀ (r : ℝ) (π : ℝ), (∀ (a b c : ℝ)
  (h1 : a = 5) (h2 : b = 5) (h3 : c = 4),
  r = sqrt (a * b * (a + b + c) * (a + b - c)) / c →
  ∀ (area : ℝ), area = π * r ^ 2 →
  area = 13125 / 1764 * π) :=
  λ r π a b c h1 h2 h3 h_radius area h_area, sorry

end area_of_circumcircle_of_isosceles_triangle_l207_207538


namespace geom_series_problem_l207_207009

variable {α : Type*} [Field α]

noncomputable def geom_series (a1 q : α) (n : ℕ) : α :=
a1 * (1 - q ^ n) / (1 - q)

theorem geom_series_problem :
  -- Part 1 conditions
  let q1 := (2 : α),
      S4 := (1 : α),
      a1_15 := (1 / 15 : α) in
  geom_series a1_15 q1 4 = S4 →
  geom_series a1_15 q1 8 = (17 : α) ∧

  -- Part 2 conditions
  let a1 := (8 : α),
      q2 := (1 / 2 : α) in
  a1 + a1 * q2 ^ 2 = (10 : α) →
  a1 * q2 ^ 3 + a1 * q2 ^ 5 = (5 / 4 : α) →
  a1 * q2 ^ 3 = (1 : α) ∧
  geom_series a1 q2 5 = (31 / 2 : α) :=
by
  intros h h1 h2
  sorry

end geom_series_problem_l207_207009


namespace decreasing_intervals_range_of_m_l207_207884

-- Define the given function f(x)
def f (x : ℝ) (ω : ℝ) (ϕ : ℝ) : ℝ := 
  sqrt 3 * (cos (ω * x + ϕ))^2 - cos (ω * x + ϕ) * sin (ω * x + ϕ + π / 3) - sqrt 3 / 4

-- Define the constraints on ω and ϕ
variable (ω ϕ : ℝ)
hypothesis h1 : 0 < ω
hypothesis h2 : 0 < ϕ ∧ ϕ < π / 2

-- Define the conditions for the problem
def g (x : ℝ) (m : ℝ) : ℝ :=
  f (x - 5 / 6) ω (ϕ) ^ 2 + 1 / 4 * f (x - 1 / 3) ω (ϕ) + m

-- Prove that for the given x, the intervals where f(x) is decreasing are
theorem decreasing_intervals (x : ℝ) : 
  0 ≤ x ∧ x ≤ 2 → (0 ≤ x ∧ x ≤ 1/6) ∨ (5/6 ≤ x ∧ x ≤ 2) :=
  sorry

-- Prove that the range of m for which g(x) has a root in [5/6, 3/2]
theorem range_of_m (x : ℝ) (m : ℝ) :
  5/6 ≤ x ∧ x ≤ 3/2 → g x m = 0 → -17/64 ≤ m ∧ m ≤ -1/8 :=
  sorry

end decreasing_intervals_range_of_m_l207_207884


namespace class_B_more_uniform_than_class_A_l207_207160

-- Definitions based on the given problem
def class_height_variance (class_name : String) : ℝ :=
  if class_name = "A" then 3.24 else if class_name = "B" then 1.63 else 0

-- The theorem statement proving that Class B has more uniform heights (smaller variance)
theorem class_B_more_uniform_than_class_A :
  class_height_variance "B" < class_height_variance "A" :=
by
  sorry

end class_B_more_uniform_than_class_A_l207_207160


namespace positive_number_eq_576_l207_207142

theorem positive_number_eq_576 (x : ℝ) (h : 0 < x) (h_eq : (2 / 3) * x = (25 / 216) * (1 / x)) : x = 5.76 := 
by 
  sorry

end positive_number_eq_576_l207_207142


namespace real_value_when_z_is_real_m_not_equal_1_when_z_is_imaginary_pure_imaginary_value_when_z_is_pure_imaginary_l207_207679

noncomputable def z (m : ℝ) : ℂ := complex.mk (m + 1) (m - 1)

theorem real_value_when_z_is_real (m : ℝ) :
  (complex.im (z m) = 0) → m = 1 :=
by sorry

theorem m_not_equal_1_when_z_is_imaginary (m : ℝ) :
  (complex.re (z m) ≠ 0 ∧ complex.im (z m) ≠ 0) → m ≠ 1 :=
by sorry

theorem pure_imaginary_value_when_z_is_pure_imaginary (m : ℝ) :
  (complex.re (z m) = 0 ∧ complex.im (z m) ≠ 0) → m = -1 :=
by sorry

end real_value_when_z_is_real_m_not_equal_1_when_z_is_imaginary_pure_imaginary_value_when_z_is_pure_imaginary_l207_207679


namespace statement_A_statement_D_l207_207275

-- Condition: Line m, and planes α and β.
variable (m : Line) (α β : Plane)

-- Statement A to be proven in Lean: If m ∥ α and α ∥ β, then m ∥ β.
theorem statement_A (h1 : m ∥ α) (h2 : α ∥ β) : m ∥ β :=
by
  sorry

-- Statement D to be proven in Lean: If m ⊥ α and α ⊥ β, then m ∥ β.
theorem statement_D (h1 : m ⊥ α) (h2 : α ⊥ β) : m ∥ β :=
by
  sorry

end statement_A_statement_D_l207_207275


namespace remainder_degrees_l207_207500

theorem remainder_degrees (f : Polynomial ℝ) :
  let divisor := (2 : ℚ) * X^3 - (5 : ℚ) * X^2 + (7 : ℚ) * X - 8
  ∃ r : Polynomial ℝ, r.degree < divisor.degree :=
begin
  sorry
end

end remainder_degrees_l207_207500


namespace triangle_height_from_area_and_base_l207_207873

theorem triangle_height_from_area_and_base (height base area : ℝ) 
  (h_area : area = 24.36) 
  (h_base : base = 8.4) 
  (h_formula : area = (base * height) / 2) : 
  height = 5.8 := 
by {
  -- Reduce the goal utilizing the provided conditions
  subst h_area, subst h_base, sorry
}

end triangle_height_from_area_and_base_l207_207873


namespace caleb_grandfather_age_l207_207992

theorem caleb_grandfather_age :
  let yellow_candles := 27
  let red_candles := 14
  let blue_candles := 38
  yellow_candles + red_candles + blue_candles = 79 :=
by
  sorry

end caleb_grandfather_age_l207_207992


namespace prove_pattern_example_l207_207317

noncomputable def pattern_example : Prop :=
  (1 * 9 + 2 = 11) ∧
  (12 * 9 + 3 = 111) ∧
  (123 * 9 + 4 = 1111) ∧
  (1234 * 9 + 5 = 11111) ∧
  (12345 * 9 + 6 = 111111) →
  (123456 * 9 + 7 = 1111111)

theorem prove_pattern_example : pattern_example := by
  sorry

end prove_pattern_example_l207_207317


namespace trapezoid_LM_sqrt2_l207_207799

theorem trapezoid_LM_sqrt2 (K L M N P Q : Type*)
  (KM : ℝ)
  (KN MQ LM MP : ℝ)
  (h_KM : KM = 1)
  (h_KN_MQ : KN = MQ)
  (h_LM_MP : LM = MP) 
  (h_KP_1 : KN = 1) 
  (h_MQ_1 : MQ = 1) :
  LM = Real.sqrt 2 :=
by
  sorry

end trapezoid_LM_sqrt2_l207_207799


namespace find_norm_b_l207_207732

noncomputable def a : ℝ × ℝ := (2, 4)
noncomputable def proj_a_on_b (b : ℝ × ℝ) : ℝ := 3
def distance_a_b : ℝ := 3 * Real.sqrt 3

theorem find_norm_b (b : ℝ × ℝ) (h1 : (Real.sqrt((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2)) = distance_a_b)
  (h2 : (a.1 * b.1 + a.2 * b.2) / Real.sqrt(b.1 ^ 2 + b.2 ^ 2) = proj_a_on_b b) :
  Real.sqrt (b.1 ^ 2 + b.2 ^ 2) = 7 :=
sorry

end find_norm_b_l207_207732


namespace number_of_pairs_l207_207670

theorem number_of_pairs (H : ∀ x y : ℕ , 0 < x → 0 < y → x < y → 2 * x * y / (x + y) = 4 ^ 15) :
  ∃ n : ℕ, n = 29 :=
by
  sorry

end number_of_pairs_l207_207670


namespace price_per_individual_bag_l207_207978

-- Define the problem conditions as Lean types and instances.
def num_students := 25
def num_vampire := 11
def num_pumpkin := 14
def pack_size := 5
def pack_price := 3
def total_budget := 17

-- Define the number of packs needed and individual bags remaining.
def num_vampire_packs := num_vampire / pack_size -- 11 / 5 = 2 packs
def num_pumpkin_packs := num_pumpkin / pack_size -- 14 / 5 = 2 packs

def remaining_budget := total_budget - (num_vampire_packs + num_pumpkin_packs) * pack_price
def num_individual_bags := (num_vampire % pack_size) + (num_pumpkin % pack_size) -- 1 + 4 = 5

-- Prove the price of each individual bag
theorem price_per_individual_bag : 
    remaining_budget / num_individual_bags = 1 :=
by 
  unfold num_vampire_packs 
  unfold num_pumpkin_packs 
  unfold remaining_budget 
  unfold num_individual_bags 
  sorry

end price_per_individual_bag_l207_207978


namespace trains_crossing_time_opposite_directions_l207_207489

noncomputable theory

-- Define the lengths and speeds of the trains
def speed1_kmph : ℝ := 60
def speed2_kmph : ℝ := 40
def conversion_factor : ℝ := 5 / 18
def speed_diff_mps := (speed1_kmph - speed2_kmph) * conversion_factor
def speed_sum_mps := (speed1_kmph + speed2_kmph) * conversion_factor
def time_same_dir_sec : ℝ := 50

-- Length of each train in meters
def length_each_train : ℝ := (speed_diff_mps * time_same_dir_sec) / 2

-- Distance they need to cover when running in opposite directions (2L)
def distance_opposite_dir : ℝ := 2 * length_each_train

-- Time to cross each other when running in opposite directions
def time_opposite_dir_sec : ℝ := distance_opposite_dir / speed_sum_mps

-- The main theorem
theorem trains_crossing_time_opposite_directions :
  time_opposite_dir_sec = 10 := by
  -- We state the proof here, but in this task we only need the statement including final answer
  sorry

end trains_crossing_time_opposite_directions_l207_207489


namespace sum_of_edges_l207_207929

-- Define the number of edges for a triangle and a rectangle
def edges_triangle : Nat := 3
def edges_rectangle : Nat := 4

-- The theorem states that the sum of the edges of a triangle and a rectangle is 7
theorem sum_of_edges : edges_triangle + edges_rectangle = 7 := 
by
  -- proof omitted
  sorry

end sum_of_edges_l207_207929


namespace sum_of_ages_l207_207975

-- Defining the variables representing ages.
variables (a b c : ℕ)

-- Conditions given in the problem.
def condition1 : Prop := a = 20 + 2 * (b + c)
def condition2 : Prop := a^2 = 1980 + 3 * (b + c)^2

-- The proof problem: Prove the sum of ages equals 68 given the conditions.
theorem sum_of_ages (h1 : condition1) (h2 : condition2) : a + b + c = 68 :=
sorry

end sum_of_ages_l207_207975


namespace midpoints_collinear_l207_207693

variable {α : Type*} [EuclideanSpace α]

-- Definitions for triangle ABC, point D, and points A1, B1, C1.
variables (A B C D A1 B1 C1 : α)

-- Definition of midpoints MA, MB, MC.
def midpoint (x y : α) : α := (x + y) / 2

def MA : α := midpoint A A1
def MB : α := midpoint B B1
def MC : α := midpoint C C1

-- Main theorem statement: The midpoints of segments AA1, BB1, CC1 are collinear.
theorem midpoints_collinear
  (hA1 : A1 = foot_of_perpendicular D (line_through B C))
  (hB1 : B1 = foot_of_perpendicular D (line_through A C))
  (hC1 : C1 = foot_of_perpendicular D (line_through A B)) :
  collinear {MA, MB, MC} :=
by sorry

end midpoints_collinear_l207_207693


namespace equidistant_points_l207_207163

theorem equidistant_points (r d1 d2 : ℝ) (d1_eq : d1 = r) (d2_eq : d2 = 6) : 
  ∃ p : ℝ, p = 2 := 
sorry

end equidistant_points_l207_207163


namespace necessity_and_insufficiency_for_p_q_l207_207683

def p (x : ℝ) : Prop := x > 1
def q (x : ℝ) : Prop := log (2^x) > 1

theorem necessity_and_insufficiency_for_p_q (x : ℝ) :
  (q x → p x) ∧ ¬(p x → q x) := by
  sorry

end necessity_and_insufficiency_for_p_q_l207_207683


namespace Lorraine_ate_one_brownie_l207_207404

theorem Lorraine_ate_one_brownie :
  ∀ (total initial children_fraction family_fraction remaining : ℕ), 
    initial = 16 →
    children_fraction = 25 →
    family_fraction = 50 →
    remaining = 5 →
    let children_ate := initial * children_fraction / 100 in
    let after_children := initial - children_ate in
    let family_ate := after_children * family_fraction / 100 in
    let after_family := after_children - family_ate in
    after_family - remaining = 1 :=
begin
  sorry
end

end Lorraine_ate_one_brownie_l207_207404


namespace div_by_20_l207_207063

theorem div_by_20 (n : ℕ) : 20 ∣ (9 ^ (8 * n + 4) - 7 ^ (8 * n + 4)) :=
  sorry

end div_by_20_l207_207063


namespace weeks_worked_l207_207615

-- Define a structure to encapsulate Everett's work conditions
structure WorkConditions where
  average_hours_weekday : ℕ -- average 5 hours per weekday
  average_hours_weekend : ℕ -- average 6 hours per weekend day
  total_hours : ℕ -- total 140 hours
  overtime_pay : ℕ -- 300 dollars in overtime pay
  normal_rate : ℕ -- 15 dollars per hour
  overtime_rate : ℕ -- 30 dollars per hour

-- Helper function to calculate hours worked per week
def hours_worked_per_week (conds : WorkConditions) : ℕ :=
  (conds.average_hours_weekday * 5) + (conds.average_hours_weekend * 2)

-- Everett's work conditions
def everett_conditions : WorkConditions :=
  { average_hours_weekday := 5,
    average_hours_weekend := 6,
    total_hours := 140,
    overtime_pay := 300,
    normal_rate := 15,
    overtime_rate := 30 }

-- Prove that Everett worked for approximately 4 weeks given the conditions
theorem weeks_worked (conds : WorkConditions) : ℕ :=
  conds.total_hours / (hours_worked_per_week conds)
  -- Since he can't work a fraction of a week, round to the nearest whole number
  -- This is interpreted in Lean as proof ≈ 4 weeks
  ≈ 4

#eval weeks_worked everett_conditions -- Should evaluate to true for the given conditions.

end weeks_worked_l207_207615


namespace length_of_EF_l207_207572

/-- Rectangle with properties as described in the problem. -/
structure Rectangle := 
  (A B C D E F : Point)
  (AB BC : ℝ)
  (side_condition: AB = 8 ∧ BC = 22)
  (point_condition: E ∈ segment A B ∧ F ∈ segment B C ∧ AE = EF ∧ EF = FD)

-- Define the theorem to prove the length of EF
theorem length_of_EF (r : Rectangle) : r.EF = 12 :=
by
  sorry

end length_of_EF_l207_207572


namespace expected_value_sum_of_three_marbles_l207_207337

-- Definitions
def marbles : List ℕ := [1, 2, 3, 4, 5, 6]
def combinations (n k : ℕ) : ℕ := Nat.choose n k

def sum_combinations (combs : List (List ℕ)) : ℕ :=
  combs.map List.sum |>.sum

-- Main theorem to prove
theorem expected_value_sum_of_three_marbles : 
  let combs := (marbles.combination 3) in
  combinations 6 3 = 20 → 
  sum_combinations combs = 210 →
  (sum_combinations combs) / (combinations 6 3) = 10.5 :=
by
  sorry

end expected_value_sum_of_three_marbles_l207_207337


namespace exists_triangle_labeled_0_1_2_l207_207429

-- Assuming we have some type Point and Triangle
-- And that triangles can be partitioned and vertices labeled.

variables {Point : Type} [DecidableEq Point] -- Points in the plane
variables {Triangle : Type} -- Triangles formed by points
variables partition : Triangle → Prop -- Partition of triangles

-- Assuming a function to get the labels of the vertices of a triangle
def vertices_labels (Δ : Triangle) : Finset ℕ := sorry

-- Main statement
theorem exists_triangle_labeled_0_1_2 (partition : Triangle → Prop) :
  (∃ (Δ : Triangle), partition Δ ∧ vertices_labels Δ = {0, 1, 2}) :=
sorry

end exists_triangle_labeled_0_1_2_l207_207429


namespace remainder_of_2013th_term_l207_207960

def fibonacci_sequence_mod (n : ℕ) (a₀ a₁: ℕ) : ℕ := 
  (λ i, if i = 0 then a₀ else if i = 1 then a₁ else fibonacci_sequence_mod (i - 1) a₀ a₁ + fibonacci_sequence_mod (i - 2) a₀ a₁) n

theorem remainder_of_2013th_term :
  let a₀ := 8 in
  let a₁ := 1 in
  let n := 2013 in
  let mod_val := 105 in
  fibonacci_sequence_mod n a₀ a₁ % mod_val = 16 :=
sorry

end remainder_of_2013th_term_l207_207960


namespace smallest_b_for_45_b_square_l207_207924

theorem smallest_b_for_45_b_square :
  ∃ b : ℕ, b > 5 ∧ ∃ n : ℕ, 4 * b + 5 = n^2 ∧ b = 11 :=
by
  sorry

end smallest_b_for_45_b_square_l207_207924


namespace range_of_t_l207_207720

noncomputable def f (x : ℝ) : ℝ := |x * Real.exp x|

def quadratic (t : ℝ) (x : ℝ) := (f x)^2 - 2 * t * (f x) + 3

theorem range_of_t (t : ℝ) :
  (∀ x : ℝ, (quadratic t x) = 0 → f x = t ∧ t ∈ Ioo (Real.sqrt 3) ((1 + 3 * Real.exp 2) / (2 * Real.exp 1))) :=
by
  sorry

end range_of_t_l207_207720


namespace value_of_tangent_function_at_pi_over_4_l207_207463

theorem value_of_tangent_function_at_pi_over_4 :
  ∀ (ω : ℝ), 0 < ω → 
  (∃ x₁ x₂ : ℝ, f x₁ = f x₂ ∧ f x₁ = π / 4 ∧ f x₂ = π / 4 ∧ abs(x₂ - x₁) = π / 4) →
  f(π / 4) = 0 :=
by {
  intros ω ω_pos intersects_at_two_points,
  let f := λ x, tan (ω * x),
  have := sorry,
  exact this,
}

end value_of_tangent_function_at_pi_over_4_l207_207463


namespace min_value_f_domain_l207_207658

def f (x y : ℝ) : ℝ := x*y / (x^2 + y^2)

theorem min_value_f_domain : 
  (∀ x y : ℝ, (1/2 ≤ x ∧ x ≤ 1) ∧ (2/5 ≤ y ∧ y ≤ 1/2) → 
  f x y ≥ 1/2 ∧ ∃ x y, (1/2 ≤ x ∧ x ≤ 1) ∧ (2/5 ≤ y ∧ y ≤ 1/2) ∧ f x y = 1/2) := 
by
  sorry

end min_value_f_domain_l207_207658


namespace num_candidates_l207_207509

theorem num_candidates (n : ℕ) (h : n * (n - 1) = 30) : n = 6 :=
sorry

end num_candidates_l207_207509


namespace expected_value_sum_marbles_l207_207339

theorem expected_value_sum_marbles :
  let marbles := {1, 2, 3, 4, 5, 6} in
  let combinations := {S | S ⊆ marbles ∧ S.size = 3} in
  let summed_values := {sum S | S ∈ combinations} in
  let total_sum := summed_values.sum in
  let number_of_combinations := combinations.card in
  (total_sum: ℚ) / number_of_combinations = 10.5 :=
sorry

end expected_value_sum_marbles_l207_207339


namespace complex_sum_zero_l207_207591

theorem complex_sum_zero (i : ℂ) (h : i^2 = -1) : 3 * (∑ k in finset.range 604, i ^ k) = 0 :=
by
  sorry

end complex_sum_zero_l207_207591


namespace total_rainfall_2000_l207_207379

theorem total_rainfall_2000 (avg_rainfall_jan_to_jun : ℕ) (rainfall_increase_jul_onwards : ℕ) :
  avg_rainfall_jan_to_jun = 30 → rainfall_increase_jul_onwards = 5 → 
  6 * avg_rainfall_jan_to_jun + 6 * (avg_rainfall_jan_to_jun + rainfall_increase_jul_onwards) = 390 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end total_rainfall_2000_l207_207379


namespace eval_expression_l207_207253

def numerator := ( (7 - 6.35) / 6.5 + 9.9) * (1 / 12.8)
def inner_denominator_1 := 1.2 / 36
def inner_denominator_2 := 6 / 5 / (1 / 4)
def inner_denominator_3 := 11 / 6
def d := inner_denominator_1 + inner_denominator_2 - inner_denominator_3
def denominator := d * 5 / 4
def expression := (numerator / denominator) / 0.125

theorem eval_expression : expression = 5 / 3 :=
by
  sorry

end eval_expression_l207_207253


namespace product_of_numbers_l207_207105

theorem product_of_numbers (x y z : ℤ) 
  (h1 : x + y + z = 30) 
  (h2 : x = 3 * ((y + z) - 2))
  (h3 : y = 4 * z - 1) : 
  x * y * z = 294 := 
  sorry

end product_of_numbers_l207_207105


namespace value_of_B_minus_3_plus_A_l207_207753

theorem value_of_B_minus_3_plus_A (A B : ℝ) (h : A + B = 5) : B - 3 + A = 2 :=
by 
  sorry

end value_of_B_minus_3_plus_A_l207_207753


namespace parallelogram_condition_l207_207041

theorem parallelogram_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (∀ (P : ℝ × ℝ), P ∈ {P | P.1^2 / a^2 + P.2^2 / b^2 = 1} → 
  ∃ (A B C : ℝ × ℝ), 
    is_vertex P A B C ∧ is_parallelogram P A B C ∧ 
    circumscribed_around A B C (λ p, p.1^2 + p.2^2 = 1) ∧
    inscribed_in (A, B, C, P) (λ p, p.1^2 / a^2 + p.2^2 / b^2 = 1)) ↔ 
  (1/a^2 + 1/b^2 = 1) := 
sorry

end parallelogram_condition_l207_207041


namespace sum_of_digits_of_9n_is_9_l207_207878

theorem sum_of_digits_of_9n_is_9 (n : ℕ) 
  (h : ∀ i j : ℕ, i < j → digit n i < digit n j) : 
  sum_of_digits (9 * n) = 9 :=
sorry

end sum_of_digits_of_9n_is_9_l207_207878


namespace number_of_diagonal_lengths_l207_207766

theorem number_of_diagonal_lengths :
  let A, B, C, D : Type:= sorry 
  let AB: ℕ := 7
  let BC: ℕ := 9
  let CD: ℕ := 14
  let DA: ℕ := 10
  let possible_diagonal_lengths := {x : ℕ | 5 ≤ x ∧ x ≤ 15}
  possible_diagonal_lengths.card = 11 :=
by
  sorry

end number_of_diagonal_lengths_l207_207766


namespace find_LM_l207_207806

variables (K L M N P Q : Type)
variables (KL MN LM KN MQ MP KP KM : ℝ) 

-- Conditions
def trapezoid (K L M N : Type) : Prop := 
  KM = 1 ∧ 
  KP = 1 ∧
  MQ = 1 ∧
  KN = MQ ∧
  LM = MP

-- To Prove
theorem find_LM (h : trapezoid K L M N) : LM = sqrt 2 :=
by sorry

end find_LM_l207_207806


namespace bugs_move_produces_empty_cell_l207_207774

theorem bugs_move_produces_empty_cell :
    ∀ (board : ℕ → ℕ → bool), -- board arrangement
    (∀ i j, i < 5 → j < 5 → board i j) → -- each cell has one bug initially
    (∀ i j, i < 5 → j < 5 → 
      ((i > 0 ∧ board (i - 1) j = false) ∨ 
       (i < 4 ∧ board (i + 1) j = false) ∨ 
       (j > 0 ∧ board i (j - 1) = false) ∨
       (j < 4 ∧ board i (j + 1) = false)) → 
    (∃ i j, i < 5 ∧ j < 5 ∧ board i j = false) → -- there is at least one empty cell after move
    ∃ i j, i < 5 ∧ j < 5 ∧ ¬ board i j := -- conclusion: there is at least one empty cell
sorry

end bugs_move_produces_empty_cell_l207_207774


namespace triangle_altitude_and_angle_bisector_iff_l207_207024

theorem triangle_altitude_and_angle_bisector_iff
  (A B C D P E F : Point)
  (h1 : ¬ (∠ABC = 90 ∨ ∠BCA = 90 ∨ ∠CAB = 90)) -- Triangle has no right angles
  (h2 : D ∈ Line(B, C)) -- D is on BC
  (h3 : E ∈ Perpendicular(D, Line(A, B))) -- E is foot of perpendicular from D to AB
  (h4 : F ∈ Perpendicular(D, Line(A, C))) -- F is foot of perpendicular from D to AC
  (h5 : P = intersection(PointOnLine(B, F), PointOnLine(C, E))) -- P is intersection of BF and CE
  : (Perpendicular(Line(A, P), Line(B, C)) ↔ AngleBisector(PointOnLine(A, D), ∠BAC)) :=
sorry

end triangle_altitude_and_angle_bisector_iff_l207_207024


namespace algebraic_expression_value_l207_207127

-- Definition of the conditions
def condition (a b : ℝ) : Prop :=
  a + b + 1 = 5

-- The proof problem statement
theorem algebraic_expression_value (a b : ℝ) (h : condition a b) : 
  (a * (-1)^3 + b * (-1) + 1 = -3) :=
by {
  intro a b,
  simp at *,
  sorry
}

end algebraic_expression_value_l207_207127


namespace trapezoid_LM_value_l207_207811

theorem trapezoid_LM_value (K L M N P Q : Type) 
  (d1 d2 : ℝ)
  (h1 : d1 = 1)
  (h2 : d2 = 1)
  (height_eq : KM = 1)
  (KN_eq_MQ : KN = MQ)
  (LM_eq_MP : LM = MP) :
  LM = 1 / real.sqrt (real.sqrt 2) :=
by 
  sorry

end trapezoid_LM_value_l207_207811


namespace find_valid_sets_l207_207244

noncomputable def satisfies_property (X : Set ℕ) : Prop :=
  ∀ (m n : ℕ), m ∈ X → n ∈ X → m < n → ∃ k ∈ X, n = m * k^2

noncomputable def is_valid_set (X : Set ℕ) : Prop :=
  ∃ m, m > 1 ∧ X = {m, m^3}

theorem find_valid_sets :
  ∀ X : Set ℕ, X.nonempty ∧ ∃ m n : ℕ, m ∈ X ∧ n ∈ X ∧ m < n
  → satisfies_property X
  → is_valid_set X :=
by
  intros X H1 H2
  sorry

end find_valid_sets_l207_207244


namespace projection_problem_l207_207550

noncomputable def vector_projection 
  (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2 in
  let magnitude_squared := v.1 * v.1 + v.2 * v.2 in
  let scalar := dot_product / magnitude_squared in
  (scalar * v.1, scalar * v.2)

theorem projection_problem :
  vector_projection (-3, 2) (2, -1) = (-16 / 5, 8 / 5) :=
by
  sorry

end projection_problem_l207_207550


namespace num_small_cubes_l207_207070

theorem num_small_cubes (L : ℝ) (h : L > 0) : 
  let V_large := L^3
  let V_small := (L / 4)^3
  V_large / V_small = 64 :=
by {
  -- Definitions
  let V_large := L^3
  let V_small := (L / 4)^3

  -- Proof goal
  have h1 : V_large = L^3 := rfl
  have h2 : V_small = (L / 4)^3 := rfl

  -- Calculations
  have h3 : V_large / V_small = (L^3) / (L^3 / 64) := by simp [h1, h2]
  have h4 : (L^3) / (L^3 / 64) = 64 := by field_simp; ring
  show V_large / V_small = 64, from h4
}

end num_small_cubes_l207_207070


namespace convert_to_rectangular_form_l207_207602

theorem convert_to_rectangular_form :
  (√3 * Complex.exp (13 * Real.pi * Complex.I / 6)) = (3 / 2 + (√3 / 2) * Complex.I) :=
by
  sorry

end convert_to_rectangular_form_l207_207602


namespace number_is_divisible_by_divisor_l207_207547

-- Defining the number after replacing y with 3
def number : ℕ := 7386038

-- Defining the divisor which we need to prove 
def divisor : ℕ := 7

-- Stating the property that 7386038 is divisible by 7
theorem number_is_divisible_by_divisor : number % divisor = 0 := by
  sorry

end number_is_divisible_by_divisor_l207_207547


namespace log_a_2020_l207_207728

noncomputable def a : ℕ → ℝ
| 0     := 1 / 2
| (n+1) := 2 * a n

theorem log_a_2020 : log 2 (a 2019) = 2018 :=
by
  have h₁ : a 1 = 1/2 := rfl
  have h₂ : a 2 = 2 * (1/2) := rfl
  have h₃ : a 2020 = 2 ^ 2018 := sorry
  rw [h₃, log_pow]
  norm_num

end log_a_2020_l207_207728


namespace count_sixth_powers_below_200_l207_207327

noncomputable def is_sixth_power (n : ℕ) : Prop := ∃ z : ℕ, n = z^6

theorem count_sixth_powers_below_200 : 
  (finset.filter (λ n, n < 200) (finset.filter is_sixth_power (finset.range 200))).card = 2 := 
by 
  sorry

end count_sixth_powers_below_200_l207_207327


namespace four_digit_numbers_with_8_or_3_l207_207318

theorem four_digit_numbers_with_8_or_3 :
  let total_four_digit_numbers := 9000
  let without_8_or_3_first := 7
  let without_8_or_3_rest := 8
  let numbers_without_8_or_3 := without_8_or_3_first * without_8_or_3_rest^3
  total_four_digit_numbers - numbers_without_8_or_3 = 5416 :=
by
  let total_four_digit_numbers := 9000
  let without_8_or_3_first := 7
  let without_8_or_3_rest := 8
  let numbers_without_8_or_3 := without_8_or_3_first * without_8_or_3_rest^3
  sorry

end four_digit_numbers_with_8_or_3_l207_207318


namespace distance_center_sphere_to_triangle_plane_l207_207970

/-- Problem Conditions -/
def r_sphere : ℝ := 10
def a_triang : ℝ := 13
def b_triang : ℝ := 13
def c_triang : ℝ := 20

/-- Theorem statement -/
theorem distance_center_sphere_to_triangle_plane (r_sphere a_triang b_triang c_triang : ℝ) 
  (tangent_sides : a_triang = 13 ∧ b_triang = 13 ∧ c_triang = 20 ∧ r_sphere = 10) :
  distance_from_center_to_plane r_sphere a_triang b_triang c_triang = 7 :=
sorry

end distance_center_sphere_to_triangle_plane_l207_207970


namespace probability_three_digit_multiple_of_3_l207_207525

theorem probability_three_digit_multiple_of_3 : 
  let digits := {1, 2, 3, 4, 5}
  let total_combinations := 60
  let valid_combinations := 24
  valid_combinations / total_combinations = 2 / 5 :=
by
  let digits := {1, 2, 3, 4, 5}
  let total_combinations := 60
  let valid_combinations := 24
  have h1 : valid_combinations / total_combinations = 2 / 5 := 
    by linarith
  exact h1


end probability_three_digit_multiple_of_3_l207_207525


namespace garden_enlargement_l207_207964

-- Define the problem conditions
def rect_length : ℝ := 40
def rect_width : ℝ := 20
def rect_area : ℝ := rect_length * rect_width
def rect_perimeter : ℝ := 2 * (rect_length + rect_width)
def square_side : ℝ := rect_perimeter / 4
def square_area : ℝ := square_side * square_side

-- State the theorem to be proved
theorem garden_enlargement : square_area - rect_area = 100 := by
  sorry

end garden_enlargement_l207_207964


namespace part_I_part_II_l207_207266

noncomputable def M : set ℝ := { x : ℝ | (2 * x - 2) / (x + 3) > 1 }
noncomputable def N (a : ℝ) : set ℝ := { x : ℝ | x^2 + (a - 8) * x - 8 * a ≤ 0 }

theorem part_I (a : ℝ) (x : ℝ) : 
  a = -6 → (x ∈ N a → x ∈ M) :=
by
  intro ha
  have hM : M = { x : ℝ | x < -3 ∨ x > 5 } := sorry
  have hN : N (-6) = { x : ℝ | 6 ≤ x ∧ x ≤ 8 } := sorry
  sorry

theorem part_II (a : ℝ) :
  (∀ x, x ∈ M → x ∈ N a) ↔ a < -5 :=
by
  have hM : M = { x : ℝ | x < -3 ∨ x > 5 } := sorry
  have hN : N a = { x : ℝ | (x - 8) * (x + a) ≤ 0 } := sorry
  sorry

end part_I_part_II_l207_207266


namespace common_difference_l207_207288

def arithmetic_sequence {α : Type} [Add α] [Mul α] (a : ℕ → α) (d : α) : Prop :=
∀ n m, a (n + 1) = a n + d

theorem common_difference (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 1 + a 7 = -2) (h2 : a 3 = 2)
  (h_seq : arithmetic_sequence a d) : d = -3 := by
  sorry

end common_difference_l207_207288


namespace problem_part_I_problem_part_II_l207_207778

variables (x y x0 y0: ℝ)

def point_A := (0, -1)
def line_B : Set (ℝ × ℝ) := { p | p.snd = -3 }
def point_O := (0, 0)

def vector_MA (x y : ℝ) : ℝ × ℝ := (-x, -1 - y)
def vector_MB (x y : ℝ) : ℝ × ℝ := (0, -3 - y)
def vector_AB (x : ℝ) : ℝ × ℝ := (x, -2)

def curve_C : Set (ℝ × ℝ) := { p | p.snd = 1/4 * p.fst^2 - 2 }

theorem problem_part_I (h : ∃ M : ℝ × ℝ, (∀ B : ℝ × ℝ, B ∈ line_B → ∥(vector_MA M.fst M.snd) ∥ * ∥(vector_AB B.fst).1∥ = ∥(vector_MB B.fst M.snd)∥ * ∥vector_AB B.fst∥ )) :
  ∀ M : ℝ × ℝ, M ∈ curve_C := sorry

theorem problem_part_II (P : ℝ × ℝ) (hP : P ∈ curve_C) (l : ℝ × ℝ → Prop)
  (hl : ∀ x0 : ℝ, l = λ x, (x0 * x.fst - 2 * x.snd + 2 * (1/4 * x0^2 - 2) - x0^2 = 0)) :
  ∃ d : ℝ, ∀ x0 : ℝ, d = (abs (2 * (1/4 * x0^2 - 2) - x0^2) / sqrt (4 + x0^2)) :=
begin
  sorry
end

end problem_part_I_problem_part_II_l207_207778


namespace circle_area_isosceles_triangle_l207_207541

theorem circle_area_isosceles_triangle (a b c : ℝ) (h1 : a = 5) (h2 : b = 5) (h3 : c = 4) :
  let r := ((25 * Real.sqrt 21) / 42)
  in ∃ (O : Point) (r : ℝ), Circle O r ∧ r^2 * Real.pi = (13125 / 1764) * Real.pi := by
  sorry

end circle_area_isosceles_triangle_l207_207541


namespace seating_problem_smallest_n_l207_207902

   theorem seating_problem_smallest_n (k : ℕ) (n : ℕ) (h1 : 2 ≤ k) (h2 : k < n)
     (h3 : 2 * (nat.factorial (n-1) / nat.factorial (n-k)) = 
            (nat.factorial n / (nat.factorial (k-2) * nat.factorial (n-k+2))) * nat.factorial (k-2)) :
     n = 12 :=
   sorry
   
end seating_problem_smallest_n_l207_207902


namespace percentage_within_one_standard_deviation_l207_207362

-- Define the constants
def m : ℝ := sorry     -- mean
def g : ℝ := sorry     -- standard deviation
def P : ℝ → ℝ := sorry -- cumulative distribution function

-- The condition that 84% of the distribution is less than m + g
def condition1 : Prop := P (m + g) = 0.84

-- The condition that the distribution is symmetric about the mean
def symmetric_distribution (P : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ x, P (m + (m - x)) = 1 - P x

-- The problem asks to prove that 68% of the distribution lies within one standard deviation of the mean
theorem percentage_within_one_standard_deviation 
  (h₁ : condition1)
  (h₂ : symmetric_distribution P m) : 
  P (m + g) - P (m - g) = 0.68 :=
sorry

end percentage_within_one_standard_deviation_l207_207362


namespace bottles_needed_l207_207578

-- Define specific values provided in conditions
def servings_per_guest : ℕ := 2
def number_of_guests : ℕ := 120
def servings_per_bottle : ℕ := 6

-- Define total servings needed
def total_servings : ℕ := servings_per_guest * number_of_guests

-- Define the number of bottles needed (as a proof statement)
theorem bottles_needed : total_servings / servings_per_bottle = 40 := by
  /-
    The proof will go here. For now we place a sorry to mark the place where
    a proof would be required. The statement should check the equivalence of 
    number of bottles needed being 40 given the total servings divided by 
    servings per bottle.
  -/
  sorry

end bottles_needed_l207_207578


namespace distance_between_circle_centers_l207_207879

open Real

theorem distance_between_circle_centers :
  let center1 := (1 / 2, 0)
  let center2 := (0, 1 / 2)
  dist center1 center2 = sqrt 2 / 2 :=
by
  sorry

end distance_between_circle_centers_l207_207879


namespace sin_double_angle_l207_207700

variable {α : Real}

theorem sin_double_angle (h : Real.tan α = 2) : Real.sin (2 * α) = 4 / 5 := 
sorry

end sin_double_angle_l207_207700


namespace triangle_is_isosceles_l207_207823

variable {α : Type*}
variables (A B C : α)
variables [Triangle α] (bisectors : Bisectors A B C) (I : α) [Incenter α I]
variable [Area α] 

theorem triangle_is_isosceles 
  (h : area (IA1 B) + area (IB1 C) + area (IC1 A) = (1 / 2) * area (A B C)) : 
  is_isosceles A B C :=
sorry

end triangle_is_isosceles_l207_207823


namespace measure_angle_between_vectors_value_of_m_l207_207147
-- Problem (1)


-- Variables for vectors and their properties
variables (a b : EuclideanSpace ℝ (Fin 3))
variables (ha : ∥a∥ = 3)
variables (hb : ∥b∥ = 2)
variables (dot_ab : a ⬝ b = -3)

-- Statement to be proved
theorem measure_angle_between_vectors :
  real.angle_between a b = 2 * real.pi / 3 :=
sorry

-- Problem (2)

-- Variables for m and vectors
variables {m : ℝ}
variables (a : EuclideanSpace ℝ (Fin 2)) (b : EuclideanSpace ℝ (Fin 2))
variables (Ha : a = ![(m-2), -3])
variables (Hb : b = ![-1, m])
variables (parallel : a • b = 0)

-- Statement to be proved
theorem value_of_m :
  m = 3 ∨ m = -1 :=
sorry

end measure_angle_between_vectors_value_of_m_l207_207147


namespace lamp_post_height_l207_207545

noncomputable def height_of_lamp_post (AC AD DE : ℝ) (E : AC = 4 ∧ AD = 3 ∧ DE = 1.6) : ℝ :=
  let DC := AC - AD in
  let ratio := DE / DC in
  let AB := ratio * AC in
  AB

theorem lamp_post_height : height_of_lamp_post 4 3 1.6 (by simp) = 6.4 :=
  sorry

end lamp_post_height_l207_207545


namespace sequence_sum_abs_eq_68_l207_207729

open Real

def a (n : ℕ) : ℤ := 2 * n - 5

def S (N : ℕ) : ℤ :=
  ∑ i in finset.range (N + 1), abs (a (i + 1))

theorem sequence_sum_abs_eq_68 : S 10 = 68 := by
  sorry

end sequence_sum_abs_eq_68_l207_207729


namespace auction_bidding_people_l207_207202

-- Definitions extracted from conditions
def price_initial := 15
def price_final := 65
def price_increment := 5
def bids_per_person := 5

-- Number of people involved in bidding war
def num_people (initial final increment bids_per_person : ℕ) : ℕ :=
  let total_increase := final - initial
  let total_bids := total_increase / increment
  total_bids / bids_per_person

-- Theorem to prove the problem statement
theorem auction_bidding_people :
  num_people price_initial price_final price_increment bids_per_person = 2 :=
by 
  simp [num_people, price_initial, price_final, price_increment, bids_per_person]
  sorry

end auction_bidding_people_l207_207202


namespace area_of_bounded_region_l207_207998

theorem area_of_bounded_region (a : ℝ) (h : a > 0) :
  let eq1 := (λ x y : ℝ, (x + 2 * a * y)^2 = 16 * a^2)
  let eq2 := (λ x y : ℝ, (2 * a * x - y)^2 = 4 * a^2)
  in area_of_region_bounded_by eq1 eq2 = 32 * a^2 / Real.sqrt (1 + 8 * a^2 + 16 * a^4) :=
sorry

end area_of_bounded_region_l207_207998


namespace correct_division_l207_207192

def divide_horses (total_horses : ℕ) :=
  let eldest_share := total_horses / 2
  let middle_share := total_horses / 4
  let youngest_share := total_horses / 8
  (eldest_share, middle_share, youngest_share)

theorem correct_division : 
  divide_horses 8 = (4, 2, 1) ∧ 4 + 2 + 1 = 7 :=
by
  unfold divide_horses
  split
  -- Calculation for shares
  {
    exact (nat.div_eq_of_lt zero_lt_two two_le_eight, 
           nat.div_eq_of_lt zero_lt_four four_le_eight, 
           nat.div_eq_of_lt zero_lt_eight eight_le_eight)
  }
  -- Summing up the shares
  {
    rfl
  }

end correct_division_l207_207192


namespace inequality_ab_equals_bc_l207_207858

-- Define the given conditions and state the theorem as per the proof problem
theorem inequality_ab_equals_bc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    a^b * b^c * c^a ≤ a^a * b^b * c^c :=
by
  sorry

end inequality_ab_equals_bc_l207_207858


namespace gold_common_difference_l207_207234

theorem gold_common_difference :
  (∃ (a : ℚ) (d : ℚ),
    let a1 := a + 9 * d,
        a2 := a + 8 * d,
        a3 := a + 7 * d,
        a4 := a + 6 * d,
        a5 := a + 5 * d,
        a6 := a + 4 * d,
        a7 := a + 3 * d,
        a8 := a + 2 * d,
        a9 := a + d,
        a10 := a in
    (a8 + a9 + a10 = 4) ∧
    (a1 + a2 + a3 + a4 + a5 + a6 + a7 = 3)) →
  ∃ d : ℚ, d = 7 / 78 :=
by
  intro h
  sorry

end gold_common_difference_l207_207234


namespace curves_have_equidistant_point_l207_207737

/-- Define the points A and B and the given curves in Lean. -/
structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨1, -2⟩
def B : Point := ⟨-4, -2⟩

/-- Define the curves as sets of points satisfying the given equations. -/
def curve₁ (p : Point) : Prop := 4 * p.x + 2 * p.y = 3
def curve₂ (p : Point) : Prop := p.x^2 + p.y^2 = 3
def curve₃ (p : Point) : Prop := p.x^2 + 2 * p.y^2 = 3
def curve₄ (p : Point) : Prop := p.x^2 - 2 * p.y = 3

/-- Define the condition that the distances from point P to A and B are equal. -/
def equidistant (p : Point) : Prop :=
  (sqrt ((p.x - A.x)^2 + (p.y - A.y)^2)) = (sqrt ((p.x - B.x)^2 + (p.y - B.y)^2))

/-- Prove that for each curve, there exists a point P that is equidistant to points A and B. -/
theorem curves_have_equidistant_point : 
  (∃ p, curve₁ p ∧ equidistant p) ∧ 
  (∃ p, curve₂ p ∧ equidistant p) ∧ 
  (∃ p, curve₃ p ∧ equidistant p) ∧ 
  (∃ p, curve₄ p ∧ equidistant p) :=
  by
    sorry

end curves_have_equidistant_point_l207_207737


namespace total_money_is_correct_l207_207023

-- Definitions based on the conditions
variables (X Y Z C : ℝ)

def initial_amount : ℝ := 74
def earned_amount := X + Y + Z
def spent_amount := C
def final_amount : ℝ := 86

theorem total_money_is_correct :
  initial_amount + earned_amount - spent_amount = final_amount := sorry

end total_money_is_correct_l207_207023


namespace problem1_part1_problem1_part2_l207_207520

theorem problem1_part1 (x : ℝ) :
  -real.sqrt 2 ≤ real.sin x + real.cos x ∧ real.sin x + real.cos x ≤ real.sqrt 2 ∧
  -real.sqrt 2 ≤ real.sin x - real.cos x ∧ real.sin x - real.cos x ≤ real.sqrt 2 := 
sorry

theorem problem1_part2 (x : ℝ) :
  1 ≤ abs (real.sin x) + abs (real.cos x) ∧ abs (real.sin x) + abs (real.cos x) ≤ real.sqrt 2 :=
sorry

end problem1_part1_problem1_part2_l207_207520


namespace c_months_value_l207_207932

   variable (a_horses : ℕ) (a_months : ℕ)
   variable (b_horses : ℕ) (b_months : ℕ) (b_payment_rs : ℕ)
   variable (c_horses : ℕ) (c_months : ℕ)
   variable (total_cost_rs : ℕ)

   -- Given conditions
   def a_horses_val : a_horses = 12 := rfl
   def a_months_val : a_months = 8 := rfl
   def b_horses_val : b_horses = 16 := rfl
   def b_months_val : b_months = 9 := rfl
   def b_payment_rs_val : b_payment_rs = 360 := rfl
   def total_cost_rs_val : total_cost_rs = 870 := rfl
   def c_horses_val : c_horses = 18 := rfl

   -- Proof problem
   theorem c_months_value :
     let cost_per_horse_month := b_payment_rs_val.val / (b_horses * b_months) in
     let cost_a := (a_horses * a_months) * cost_per_horse_month in
     let cost_b := (b_horses * b_months) * cost_per_horse_month in
     let cost_c := (c_horses * c_months) * cost_per_horse_month in
     total_cost_rs_val.val = cost_a + cost_b + cost_c →
     c_months = 6 :=
   by
     intros h
     sorry
   
end c_months_value_l207_207932


namespace sum_even_302_to_400_l207_207104

theorem sum_even_302_to_400 : (List.range' 302 (400 - 302 + 1)).filter (λ x, x % 2 = 0).sum = 17550 :=
by {
  let evens_in_range := (List.range' 302 (400 - 302 + 1)).filter (λ x, x % 2 = 0),
  have n : evens_in_range.length = 50,
  { -- calculate the number of even integers between 302 and 400
    unfold evens_in_range,
    rw [List.filter_length_eq],
    -- use arithmetic series properties to show length is 50
    sorry,
  },
  -- use the arithmetic series formula for sum
  have sum_eq : evens_in_range.sum = 25 * (302 + 400),
  { -- calculate the sum using the series formula
    unfold evens_in_range,
    rw [List.sum_eq_sum_map],
    -- properties of arithmetic sequences
    sorry,
  },
  exact sum_eq,
}

end sum_even_302_to_400_l207_207104


namespace adam_remaining_loads_l207_207974

theorem adam_remaining_loads (total_loads washed_by_noon : ℕ) (h : total_loads = 14 ∧ washed_by_noon = 8) : total_loads - washed_by_noon = 6 :=
by
  -- Extract values from the provided conditions
  cases h with
  | intro h1 h2 =>
    -- Substitute the given values
    rw [h1, h2]
    -- Simplify 14 - 8
    norm_num
    -- Conclude the proof
    rfl

end adam_remaining_loads_l207_207974


namespace evaluate_expression_l207_207258

-- Define the greatest integer function
def greatest_integer (x : ℝ) : ℤ := int.floor x

-- Define y
def y : ℝ := 2 / 3

-- The goal is to prove the given expression evaluates to 10.4 under the given conditions
theorem evaluate_expression : 
  greatest_integer 6.5 * greatest_integer y + (greatest_integer 2) * 7.2 + greatest_integer 8.4 - 6.0 = 10.4 :=
by sorry

end evaluate_expression_l207_207258


namespace variance_scaled_data_l207_207103

noncomputable def standard_deviation (data : List ℝ) : ℝ := 
  sorry -- Omit the proof of standard deviation calculation

noncomputable def variance (data : List ℝ) : ℝ := 
  (standard_deviation data) ^ 2

theorem variance_scaled_data
  (a : List ℝ)
  (h : standard_deviation a = 2) :
  variance (a.map(λ x, 2 * x)) = 16 := 
  sorry

end variance_scaled_data_l207_207103


namespace minimal_k_shirts_l207_207771

-- Define the conditions
def white_shirts : ℕ := 21
def purple_shirts : ℕ := 21

-- Define the problem
theorem minimal_k_shirts :
  ∃ (k : ℕ), k = 10 ∧ 
  (∀ (order : list ℕ), 
     order.length = 42 ∧ 
     (∀ (i : ℕ), i < 21 → (order.filter (λ x, x = i)).length = 1) ∧ 
     (∀ (i : ℕ), 21 ≤ i → i < 42 → (order.filter (λ x, x = i-21)).length = 1) → 
     ((∃ (w p: list ℕ), 
         (w.length = 21 - k) ∧ 
         (p.length = 21 - k) ∧ 
         (w ∈ order.prefixes) ∧ 
         (p ∈ order.suffixes) ∧ 
         (p.all (λ x, w.nth (x - (21 - k)))) ∧ 
         (w.all (λ x, p.nth (x - (21 - k))))))) :=
sorry

end minimal_k_shirts_l207_207771


namespace probability_of_double_l207_207954

/-- 
Each integer from 0 to 13 is paired with every other integer from 0 to 13 exactly once, forming a complete set of dominoes.
A double is a domino that has the same integer on both its squares.
Prove that the probability of a randomly selected domino being a double is 2/15.
-/
theorem probability_of_double :
  let total_pairs := ((14 * 14) - 14) / 2 + 14 in
  let total_doubles := 14 in
  (total_doubles : ℚ) / total_pairs = 2 / 15 :=
by
  let total_pairs := ((14 * 14) - 14) / 2 + 14
  let total_doubles := 14
  have total_pairs_eq : total_pairs = 105 := by sorry
  have total_doubles_eq : total_doubles = 14 := by sorry
  have h_frac : (14 : ℚ) / 105 = 2 / 15 := by sorry
  rw [← total_pairs_eq, ← total_doubles_eq, h_frac]

end probability_of_double_l207_207954


namespace distinct_numbers_in_union_of_sequences_l207_207030

-- Define the first sequence
def seq1 : ℕ → ℕ := λ k, 4 * k - 1

-- Define the second sequence
def seq2 : ℕ → ℕ := λ l, 10 * l

-- Define the first set A as the first 3000 terms of seq1
def A : Finset ℕ := Finset.image seq1 (Finset.range 3000)

-- Define the second set B as the first 3000 terms of seq2
def B : Finset ℕ := Finset.image seq2 (Finset.range 3000)

-- Define the union of sets A and B
def T : Finset ℕ := A ∪ B

-- Define the cardinality of the union set T
def T_cardinality : ℕ := Finset.card T

-- The proof statement that the number of distinct numbers in T is 5250
theorem distinct_numbers_in_union_of_sequences :
  T_cardinality = 5250 :=
by
  sorry

end distinct_numbers_in_union_of_sequences_l207_207030


namespace problem_statements_analysis_l207_207131

theorem problem_statements_analysis:
  (¬∀ a b : ℝ, a + b > a) ∧
  (¬∀ x : ℝ, |x| = -x → x < 0) ∧
  (¬∀ x y : ℝ, |x| = |y| → x = y) ∧
  (∀ p q : ℤ, q ≠ 0 → (p/q : ℚ)).
by
  -- Proof of theorems goes here, but omitted with sorry
  sorry

end problem_statements_analysis_l207_207131


namespace distance_after_time_l207_207973

-- Define the speeds of Adam and Simon
def speed_adam : ℝ := 10
def speed_simon : ℝ := 8

-- Define the distance they need to be apart
def target_distance : ℝ := 50

-- Define the number of hours after which they are 50 miles apart
def time : ℝ := 25 / Real.sqrt 41

-- Prove the distance between them is 50 miles after that time
theorem distance_after_time :
  Real.sqrt ((speed_adam * time)^2 + (speed_simon * time)^2) = target_distance :=
by
  sorry

end distance_after_time_l207_207973


namespace five_more_than_three_in_pages_l207_207236

def pages := (List.range 512).map (λ n => n + 1)

def count_digit (d : Nat) (n : Nat) : Nat :=
  if n = 0 then 0
  else if n % 10 = d then 1 + count_digit d (n / 10)
  else count_digit d (n / 10)

def total_digit_count (d : Nat) (l : List Nat) : Nat :=
  l.foldl (λ acc x => acc + count_digit d x) 0

theorem five_more_than_three_in_pages :
  total_digit_count 5 pages - total_digit_count 3 pages = 22 := 
by 
  sorry

end five_more_than_three_in_pages_l207_207236


namespace triangulated_square_even_degrees_l207_207971

theorem triangulated_square_even_degrees (V : Type) [Fintype V] [DecidableEq V] (E : Finset (Sym2 V)) :
  (∀ v ∈ V, ∃ (n : ℕ), 2 ∣ (E.card v)) → 
  ∃ (v₀ : V), ∀ v ∈ V, (∃ (e : Sym2 V), e ∈ E ∧ v ∈ e → 
  (E.card v) ≠ 2) :=
sorry

end triangulated_square_even_degrees_l207_207971


namespace common_root_rational_l207_207597

variable (a b c d e f g : ℚ) -- coefficient variables

def poly1 (x : ℚ) : ℚ := 90 * x^4 + a * x^3 + b * x^2 + c * x + 18

def poly2 (x : ℚ) : ℚ := 18 * x^5 + d * x^4 + e * x^3 + f * x^2 + g * x + 90

theorem common_root_rational (k : ℚ) (h1 : poly1 a b c k = 0) (h2 : poly2 d e f g k = 0) 
  (hn : k < 0) (hi : ∀ (m n : ℤ), k ≠ m / n) : k = -1/3 := sorry

end common_root_rational_l207_207597


namespace largest_prime_factor_of_12321_l207_207633

-- Definitions based on the given conditions
def n := 12321
def a := 111
def p₁ := 3
def p₂ := 37

-- Given conditions as hypotheses
theorem largest_prime_factor_of_12321 (h1 : n = a^2) (h2 : a = p₁ * p₂) (hp₁_prime : Prime p₁) (hp₂_prime : Prime p₂) :
  p₂ = 37 ∧ ∀ p, Prime p → p ∣ n → p ≤ 37 := 
by 
  sorry

end largest_prime_factor_of_12321_l207_207633


namespace pistachio_count_l207_207526

variable {P : ℕ} -- Total number of pistachios in the bag

theorem pistachio_count (h1: 0.95 * P ∈ ℕ)
                        (h2: 0.75 * (0.95 * P) = 57) : 
                        P = 80 :=
by
  sorry

end pistachio_count_l207_207526


namespace log_relation_l207_207699

variable (a b : ℝ)

theorem log_relation (h : real.log 3 a < real.log 3 b ∧ real.log 3 b < 0) : 0 < a ∧ a < b ∧ b < 1 :=
by
  sorry

end log_relation_l207_207699


namespace find_x_plus_inv_x_l207_207702

theorem find_x_plus_inv_x (x : ℝ) (h : x^3 + (1/x)^3 = 110) : x + (1/x) = 5 :=
sorry

end find_x_plus_inv_x_l207_207702


namespace largest_prime_factor_of_12321_l207_207629

-- Definitions based on the given conditions
def n := 12321
def a := 111
def p₁ := 3
def p₂ := 37

-- Given conditions as hypotheses
theorem largest_prime_factor_of_12321 (h1 : n = a^2) (h2 : a = p₁ * p₂) (hp₁_prime : Prime p₁) (hp₂_prime : Prime p₂) :
  p₂ = 37 ∧ ∀ p, Prime p → p ∣ n → p ≤ 37 := 
by 
  sorry

end largest_prime_factor_of_12321_l207_207629


namespace smallest_four_digit_palindromic_prime_is_1101_l207_207495

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_digits 10
  s = s.reverse

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem smallest_four_digit_palindromic_prime_is_1101 :
  ∀ n : ℕ, is_four_digit_number n → is_palindrome n → is_prime n → n ≥ 1101 := by
  sorry

end smallest_four_digit_palindromic_prime_is_1101_l207_207495


namespace sequence_x_2022_l207_207673

theorem sequence_x_2022 :
  ∃ (x : ℕ → ℤ), x 1 = 1 ∧ x 2 = 1 ∧ x 3 = -1 ∧
  (∀ n, 4 ≤ n → x n = x (n-1) * x (n-3)) ∧ x 2022 = 1 := by
  sorry

end sequence_x_2022_l207_207673


namespace trapezoid_length_KLMN_l207_207788

variables {K L M N P Q : Type}
variables (trapezoid KLMN : K L M N)
variable (KM : ℝ) (KP MQ LM MP : ℝ)
variables (perp1 : KP > 0) (perp2 : MQ > 0)
variables (equal1 : KM = 1) (equal2 : KP = MQ) (equal3 : LM = MP)

theorem trapezoid_length_KLMN
(equality_KM: KM = 1)
(equality_KP_MQ: KP = MQ)
(equality_LM_MP: LM = MP)
: LM = sqrt 2 := 
by sorry

end trapezoid_length_KLMN_l207_207788


namespace circle_through_points_and_center_on_line_l207_207247

theorem circle_through_points_and_center_on_line :
  ∃ (c : ℝ × ℝ) (r : ℝ), (c = (1, 1)) ∧ r = 2 ∧ ∀ (x y : ℝ), 
  (x - c.1)^2 + (y - c.2)^2 = r^2 ↔ (x - 1)^2 + (y - 1)^2 = 4 
  :=
begin
  sorry
end

end circle_through_points_and_center_on_line_l207_207247


namespace koala_fiber_intake_l207_207822

theorem koala_fiber_intake (x : ℝ) (h : 0.40 * x = 16) : x = 40 :=
by
  sorry

end koala_fiber_intake_l207_207822


namespace value_of_k_l207_207378

-- Define the conditions
variables {m n k : ℝ}
def line_eq1 : Prop := m = 2 * n + 5
def line_eq2 : Prop := m + 1 = 2 * (n + k) + 5

-- The theorem to prove
theorem value_of_k (h1 : line_eq1) (h2 : line_eq2) : k = 1 / 2 :=
sorry

end value_of_k_l207_207378


namespace largest_prime_factor_12321_l207_207652

theorem largest_prime_factor_12321 : ∃ p, prime p ∧ (∀ q, prime q ∧ q ∣ 12321 → q ≤ p) ∧ p = 19 :=
by {
  sorry
}

end largest_prime_factor_12321_l207_207652


namespace circle_area_isosceles_triangle_l207_207532

theorem circle_area_isosceles_triangle : 
  ∀ (A B C : Type) (AB AC : Type) (a b c : ℝ),
  a = 5 →
  b = 5 →
  c = 4 →
  isosceles_triangle A B C a b c →
  circle_passes_through_vertices A B C →
  ∃ (r : ℝ), 
    area_of_circle_passing_through_vertices A B C = (15625 * π) / 1764 :=
by intros A B C AB AC a b c ha hb hc ht hcirc
   sorry

end circle_area_isosceles_triangle_l207_207532


namespace half_difference_donation_l207_207842

def margoDonation : ℝ := 4300
def julieDonation : ℝ := 4700

theorem half_difference_donation : (julieDonation - margoDonation) / 2 = 200 := by
  sorry

end half_difference_donation_l207_207842


namespace find_s_l207_207701

theorem find_s (a b r1 r2 : ℝ) (h1 : r1 + r2 = -a) (h2 : r1 * r2 = b) :
    let new_root1 := (r1 + r2) * (r1 + r2)
    let new_root2 := (r1 * r2) * (r1 + r2)
    let s := b * a - a * a
    s = ab - a^2 :=
  by
    -- the proof goes here
    sorry

end find_s_l207_207701


namespace city_H_has_greatest_percentage_increase_l207_207894

def population_1990 := 
{
  F := 60000,
  G := 80000,
  H := 70000,
  I := 85000,
  J := 95000
}

def population_2000 := 
{
  F := 78000,
  G := 96000,
  H := 91000,
  I := 94500,
  J := 114000
}

def percentage_increase (pop_1990 pop_2000 : ℕ) : ℚ := 
  ((pop_2000 - pop_1990 : ℚ) / pop_1990) * 100

theorem city_H_has_greatest_percentage_increase :
  percentage_increase population_1990.H population_2000.H =
  max (percentage_increase population_1990.F population_2000.F)
      (max (percentage_increase population_1990.G population_2000.G)
          (max (percentage_increase population_1990.I population_2000.I)
              (percentage_increase population_1990.J population_2000.J))) := 
sorry

end city_H_has_greatest_percentage_increase_l207_207894


namespace alpha_beta_sum_l207_207903

theorem alpha_beta_sum (α β : ℝ) (h : ∀ x : ℝ, (x - α) / (x + β) = (x^2 - 102 * x + 2021) / (x^2 + 89 * x - 3960)) : α + β = 176 := by
  sorry

end alpha_beta_sum_l207_207903


namespace A_inter_B_l207_207308

open Set

variable {α : Type*} [LinearOrder α] [TopologicalSpace α] [OrderTopology α] [DenselyOrdered α]

def A : Set ℝ := {x | -3 < x ∧ x < 1}
def B : Set ℝ := {x | log x / log 2 < 1 ∧ x ≠ 0}

theorem A_inter_B :
  (A ∩ B) = {x : ℝ | (-2 < x ∧ x < 0) ∨ (0 < x ∧ x < 1)} :=
  sorry

end A_inter_B_l207_207308


namespace max_face_sum_is_14_l207_207882

-- Define a cube with numbered faces
def cube_faces : list nat := [1, 2, 3, 4, 5, 6]

-- The vertices in a cube are each formed by three intersecting faces
-- We need to consider the maximum sum of these three faces
def max_sum_of_three_faces (l : list nat) : nat :=
  max (6 + 5 + 3) (max (6 + 5 + 1) (max (6 + 3 + 1) (5 + 4 + 1)))

-- The statement we need to prove
theorem max_face_sum_is_14 : max_sum_of_three_faces cube_faces = 14 :=
  sorry

end max_face_sum_is_14_l207_207882


namespace largest_prime_factor_12321_l207_207646

theorem largest_prime_factor_12321 : 
  ∃ p : ℕ, p.prime ∧ p ∣ 12321 ∧ ∀ q : ℕ, q.prime ∧ q ∣ 12321 → q ≤ p :=
sorry

end largest_prime_factor_12321_l207_207646


namespace largest_prime_factor_12321_l207_207640

theorem largest_prime_factor_12321 : 
  ∃ p : ℕ, prime p ∧ p ∣ 12321 ∧ ∀ q : ℕ, prime q ∧ q ∣ 12321 → q ≤ p :=
begin
  use 83,
  split,
  { -- Prove that 83 is a prime number
    sorry },
  split,
  { -- Prove that 83 divides 12321
    sorry },
  { -- Prove that any other prime factor of 12321 is less than or equal to 83
    sorry }
end

end largest_prime_factor_12321_l207_207640


namespace men_wages_l207_207149

-- Conditions
variable (M W B : ℝ)
variable (h1 : 15 * M = W)
variable (h2 : W = 12 * B)
variable (h3 : 15 * M + W + B = 432)

-- Statement to prove
theorem men_wages : 15 * M = 144 :=
by
  sorry

end men_wages_l207_207149


namespace triangle_value_x_l207_207370

theorem triangle_value_x (PQ QR QS : ℝ) 
  (h1 : PQ = 10)
  (h2 : QR = x)
  (h3 : ∠QSR = ∠QRS)
  (h4 : ∠SPQ = 90)
  (h5 : ∠PQS = 60)
  (h6 : ∠PSQ = 30)
  (h7 : PQ^2 + QS^2 = QR^2) 
  : x = 20 :=
sorry

end triangle_value_x_l207_207370


namespace max_expr_value_l207_207657

def expr (x y : ℝ) :=
  (sqrt (3 - sqrt 2) * sin x - sqrt (2 * (1 + cos (2 * x))) - 1) *
  (3 + 2 * sqrt (7 - sqrt 2) * cos y - cos (2 * y))

theorem max_expr_value : ∃ x y : ℝ, abs (expr x y - 9) < 1 :=
by
  sorry

end max_expr_value_l207_207657


namespace faster_train_speed_l207_207143

noncomputable def speed_faster_train (d₁ d₂ : ℕ) (t : ℕ) (k : ℕ) : ℕ :=
  let v := (d₁ + d₂) / (3 * t) in
  k * v

theorem faster_train_speed
  (length_train1 length_train2 : ℕ) (crossing_time : ℕ) (speed_ratio : ℕ)
  (h_length : length_train1 = 150) (h_length2 : length_train2 = 150)
  (h_time : crossing_time = 8) (h_ratio : speed_ratio = 2) :
  speed_faster_train length_train1 length_train2 crossing_time speed_ratio = 25 := by
  sorry

end faster_train_speed_l207_207143


namespace general_formula_sequence_l207_207691

theorem general_formula_sequence (a : ℕ → ℤ)
  (h1 : a 1 = 3)
  (h_rec : ∀ n : ℕ, n > 0 → a (n + 1) = 4 * a n + 3) :
  ∀ n : ℕ, n > 0 → a n = 4^n - 1 :=
by 
  sorry

end general_formula_sequence_l207_207691


namespace construction_company_order_l207_207166

def concrete_weight : ℝ := 0.17
def bricks_weight : ℝ := 0.17
def stone_weight : ℝ := 0.5
def total_weight : ℝ := 0.84

theorem construction_company_order :
  concrete_weight + bricks_weight + stone_weight = total_weight :=
by
  -- The proof would go here but is omitted per instructions.
  sorry

end construction_company_order_l207_207166


namespace values_of_x_satisfying_sin_l207_207741

theorem values_of_x_satisfying_sin (x : ℝ) : 
  0 ≤ x ∧ x < 360 → sin x = -0.73 → ∃ y₁ y₂, x = y₁ ∨ x = y₂ ∧ y₁ ≠ y₂ ∧ 
  (0 ≤ y₁ ∧ y₁ < 360) ∧ (0 ≤ y₂ ∧ y₂ < 360) :=
by
  sorry

end values_of_x_satisfying_sin_l207_207741


namespace emily_can_see_emerson_l207_207238

theorem emily_can_see_emerson : 
  ∀ (emily_speed emerson_speed : ℝ) 
    (initial_distance final_distance : ℝ), 
  emily_speed = 15 → 
  emerson_speed = 9 → 
  initial_distance = 1 → 
  final_distance = 1 →
  (initial_distance / (emily_speed - emerson_speed) + final_distance / (emily_speed - emerson_speed)) * 60 = 20 :=
by
  intros emily_speed emerson_speed initial_distance final_distance
  sorry

end emily_can_see_emerson_l207_207238


namespace rectangles_perimeter_correct_l207_207431

-- Define a structure for rectangles
structure Rectangle where
  width : ℕ
  height : ℕ

-- Define the specific conditions for the pattern of rectangles
def rectangles_pattern (rect : Rectangle) (n : ℕ) : Prop :=
  rect.width = 4 ∧ rect.height = 2 ∧ n = 10

-- Define the perimeter calculation
def pattern_perimeter (rect : Rectangle) (n : ℕ) : ℕ :=
  if rectangles_pattern rect n then 84 else 0

-- The main proof statement
theorem rectangles_perimeter_correct (rect : Rectangle) (n : ℕ) (h : rectangles_pattern rect n) : pattern_perimeter rect n = 84 :=
by
  sorry

end rectangles_perimeter_correct_l207_207431


namespace function_defined_when_x_geq_3_l207_207476

variable (x : ℝ)

def f (x : ℝ) : ℝ := (Real.sqrt (x - 3)) / (x - 1)

theorem function_defined_when_x_geq_3 (x : ℝ) : (∃ y : ℝ, f x = y) ↔ x ≥ 3 :=
by
  sorry

end function_defined_when_x_geq_3_l207_207476


namespace painted_cube_probability_l207_207611

theorem painted_cube_probability :
  let faces := 6,
      color_choices := 2,
      total_arrangements := color_choices^faces,
      all_same_color := 2,
      five_faces_same := 12,
      four_faces_one_color_two_opposite := 6,
      suitable_arrangements := all_same_color + five_faces_same + four_faces_one_color_two_opposite
  in
  (suitable_arrangements.to_rat / total_arrangements.to_rat) = (5 / 16 : ℚ) :=
by
  sorry

end painted_cube_probability_l207_207611


namespace triangle_inequality_equivalence_l207_207813

theorem triangle_inequality_equivalence (A B C : Type) [RealizedTriangle A B C]
  (a b : ℝ) (ha hb : ℝ) (h_ha : ha = b * sin C) (h_hb : hb = a * sin C) :
  (a + ha ≥ b + hb) ↔ (a ≥ b) := 
by
  sorry

end triangle_inequality_equivalence_l207_207813


namespace trains_cross_each_other_time_l207_207512

theorem trains_cross_each_other_time :
  (L1 L2 : ℕ) (T1 T2 : ℕ) (H1 : L1 = 120) (H2 : L2 = 150) (H3 : T1 = 10) (H4 : T2 = 15) →
  let S1 := L1 / T1 in
  let S2 := L2 / T2 in
  let SR := S1 - S2 in
  let L := L1 + L2 in
  let T := L / SR in
  T = 135 :=
by
  -- Proof goes here
  sorry

end trains_cross_each_other_time_l207_207512


namespace last_digit_product_odd_27_to_89_l207_207475

theorem last_digit_product_odd_27_to_89 
  (all_odd: ∀ n, 27 ≤ n ∧ n ≤ 89 → n % 2 = 1) : 
  (∏ n in (finset.Icc 27 89), if n % 2 = 1 then n else 1) % 10 = 5 :=
sorry

end last_digit_product_odd_27_to_89_l207_207475


namespace number_of_white_balls_l207_207773

theorem number_of_white_balls (x : ℕ) : (3 : ℕ) + x = 12 → x = 9 :=
by
  intros h
  sorry

end number_of_white_balls_l207_207773


namespace two_color_sufficient_l207_207415

-- Definitions and conditions
def equilateral_triangle : Type := sorry -- placeholder for the type representing equilateral triangles on a plane
def adjacent (t1 t2 : equilateral_triangle) : Prop := sorry -- placeholder defining when two triangles share a side
def shares_point (t1 t2 : equilateral_triangle) : Prop := sorry -- placeholder defining when two triangles share a point

-- Problem statement transformed to a Lean theorem
theorem two_color_sufficient (triangles : set equilateral_triangle) :
  (∀ t1 t2 ∈ triangles, adjacent t1 t2 → t1 ≠ t2) →
  (∀ t1 t2 ∈ triangles, shares_point t1 t2 → t1 = t2 ∨ t1 ≠ t2) →
  ∃ (coloring : equilateral_triangle → ℕ), (∀ t1 t2 ∈ triangles, adjacent t1 t2 → coloring t1 ≠ coloring t2) ∧ (∀ t ∈ triangles, coloring t = 0 ∨ coloring t = 1) :=
sorry

end two_color_sufficient_l207_207415


namespace problem1_problem2_l207_207285

-- Define the sets P and Q
def set_P : Set ℝ := {x | 2 * x^2 - 5 * x - 3 < 0}
def set_Q (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 1}

-- Problem (1): P ∩ Q = Q implies a ∈ (-1/2, 2)
theorem problem1 (a : ℝ) : (set_Q a) ⊆ set_P → -1/2 < a ∧ a < 2 :=
by 
  sorry

-- Problem (2): P ∩ Q = ∅ implies a ∈ (-∞, -3/2] ∪ [3, ∞)
theorem problem2 (a : ℝ) : (set_Q a) ∩ set_P = ∅ → a ≤ -3/2 ∨ a ≥ 3 :=
by 
  sorry

end problem1_problem2_l207_207285


namespace largest_prime_factor_12321_l207_207642

theorem largest_prime_factor_12321 : 
  ∃ p : ℕ, prime p ∧ p ∣ 12321 ∧ ∀ q : ℕ, prime q ∧ q ∣ 12321 → q ≤ p :=
begin
  use 83,
  split,
  { -- Prove that 83 is a prime number
    sorry },
  split,
  { -- Prove that 83 divides 12321
    sorry },
  { -- Prove that any other prime factor of 12321 is less than or equal to 83
    sorry }
end

end largest_prime_factor_12321_l207_207642


namespace initial_quantity_of_gummy_worms_l207_207045

theorem initial_quantity_of_gummy_worms (x : ℕ) (h : x / 2^4 = 4) : x = 64 :=
sorry

end initial_quantity_of_gummy_worms_l207_207045


namespace range_of_B_l207_207356

theorem range_of_B (a b c : ℝ) (h : a + c = 2 * b) :
  ∃ B : ℝ, 0 < B ∧ B ≤ π / 3 ∧
  ∃ A C : ℝ, ∃ ha : a = c, 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ A + B + C = π :=
sorry

end range_of_B_l207_207356


namespace sum_of_distinct_prime_factors_of_252_l207_207926

theorem sum_of_distinct_prime_factors_of_252 : (∑ p in {2, 3, 7}, p) = 12 :=
by {
  -- The prime factors of 252 are 2, 3, and 7
  have h1 : {2, 3, 7} ⊆ {p : ℕ | p.prime ∧ p ∣ 252},
  { simp, },
  -- Calculate the sum of these factors
  exact sorry
}

end sum_of_distinct_prime_factors_of_252_l207_207926


namespace find_value_l207_207351

variable (y : ℝ) (Q : ℝ)
axiom condition : 5 * (3 * y + 7 * Real.pi) = Q

theorem find_value : 10 * (6 * y + 14 * Real.pi) = 4 * Q :=
by
  sorry

end find_value_l207_207351


namespace novice_experienced_parts_l207_207952

variables (x y : ℕ)

theorem novice_experienced_parts :
  (y - x = 30) ∧ (x + 2 * y = 180) :=
sorry

end novice_experienced_parts_l207_207952


namespace acute_triangle_integers_l207_207667

theorem acute_triangle_integers :
  let count := {x : ℤ | 8 < x ∧ x < 22 ∧ (x^2 < 7^2 + 15^2 ∨ 15^2 < 7^2 + x^2)}.card
  count = 3 :=
  sorry

end acute_triangle_integers_l207_207667


namespace max_profit_l207_207165

def annual_fixed_cost : ℝ := 100000
def cost_per_1000_items : ℝ := 27000
def revenue_per_1000 (x : ℝ) : ℝ := if 0 < x ∧ x ≤ 10 then 10.8 - x^2 / 30 else 108 / x - 1000 / (3 * x^2)
def profit (x : ℝ) : ℝ := if 0 < x ∧ x ≤ 10 then (8.1 * x - x^3 / 30 - 10) else (98 - 1000 / (3 * x) - 2.7 * x)

theorem max_profit (x : ℝ) : profit x ≤ profit 9 :=
sorry

end max_profit_l207_207165


namespace find_r_eq_23_lambda_lt_1_div_3_exists_polynomial_g_l207_207280
noncomputable def sequence_a (n : ℕ) (h : 0 < n) : ℝ := 
  if h : n = 1 then 2 else n * (n + 1)

def sum_S (r : ℝ) (n : ℕ) (h : 0 < n) : ℝ := 
  sequence_a n h * ((n : ℝ) / 3 + r)

def sequence_b (n : ℕ) (h : 0 < n) : ℝ :=
  (n : ℝ) / sequence_a n h

def sum_T (n : ℕ) (h : 0 < n) : ℝ :=
  (finset.range n).sum (λ i, sequence_b (i + 1) (nat.succ_pos i))

theorem find_r_eq_23 (r : ℝ) : r = 2 / 3 :=
sorry

theorem lambda_lt_1_div_3 (λ : ℝ) (n : ℕ) (h : 0 < n) : 
  λ < sum_T (2 * n) (nat.mul_pos (nat.succ_pos _) h) - sum_T n h :=
sorry

theorem exists_polynomial_g (n : ℕ) (h : 2 ≤ n) :
  ∃ g : ℕ → ℕ, (∀ i ≥ 1, T_i + 1) = T_n * (g n) - 1 :=
sorry

end find_r_eq_23_lambda_lt_1_div_3_exists_polynomial_g_l207_207280


namespace integral_curve_exists_l207_207145

theorem integral_curve_exists (C : ℝ) :
  ∀ (x y : ℝ), 
  y' = (x^2 + 2*x*y - 5*y^2) / (2*x^2 - 6*x*y) → 
  (2 * arctan (y / x) - log ((x^2 + y^2)^3 / abs (x^5)) = C) :=
by
  intros x y hy_eq
  sorry

end integral_curve_exists_l207_207145


namespace sum_x_coords_Q3_eq_1500_l207_207151

def Q1 : Type := list (ℝ × ℝ) -- 50-gon in the Cartesian plane represented by a list of 50 vertices

def Q2 (Q : Type) : Type := list (ℝ × ℝ) -- 50-gon formed by midpoints of sides of Q

def Q3 (Q : Type) : Type := list (ℝ × ℝ) -- 50-gon formed by midpoints of sides of Q2

def sum_x_coords (Q : list (ℝ × ℝ)) : ℝ := Q.foldl (λ acc (p : ℝ × ℝ), acc + p.fst) 0

theorem sum_x_coords_Q3_eq_1500 (Q1 : list (ℝ × ℝ)) (h1 : Q1.length = 50) 
  (h_sum : sum_x_coords Q1 = 1500) : 
  sum_x_coords (Q3 (Q2 Q1)) = 1500 := 
by
  -- proof is omitted
  sorry

end sum_x_coords_Q3_eq_1500_l207_207151


namespace min_max_dot_product_l207_207839

variable (a b : ℝ) (θ : ℝ)

def v : ℝ × ℝ := (a, b)
def u (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

theorem min_max_dot_product (hθ : 0 ≤ θ ∧ θ ≤ 2*Real.pi) :
  let dot_product := a * Real.cos θ + b * Real.sin θ in
  -Real.sqrt (a^2 + b^2) ≤ dot_product ∧ dot_product ≤ Real.sqrt (a^2 + b^2) :=
by
  sorry

end min_max_dot_product_l207_207839


namespace building_height_l207_207586

theorem building_height
  (h_bamboo : ℝ := 1.8)
  (s_bamboo : ℝ := 3)
  (s_building : ℝ := 35) :
  ∃ h_building : ℝ, (h_bamboo / s_bamboo) = (h_building / s_building) ∧ h_building = 21 := 
by
  use 21
  split
  · rw [← mul_eq_mul_right_iff, mul_comm, ← mul_assoc, mul_comm (h_building _)]
    norm_num
  · norm_num
  sorry

end building_height_l207_207586


namespace trapezoid_LM_sqrt2_l207_207794

theorem trapezoid_LM_sqrt2 (K L M N P Q : Point) : 
  ∀ (h_trapezoid : is_trapezoid K L M N) 
     (diag_eq_height : distance K M = 1 ∧ height_trapezoid K L M N = 1) 
     (perp_KP_MQ : is_perpendicular(K P MN) ∧ is_perpendicular(M Q KL)) 
     (KN_MQ_eq : distance K N = distance M Q) 
     (LM_MP_eq : distance L M = distance M P), 
  distance L M = Real.sqrt 2 :=
by
  sorry

end trapezoid_LM_sqrt2_l207_207794


namespace largest_prime_factor_12321_l207_207644

theorem largest_prime_factor_12321 : 
  ∃ p : ℕ, p.prime ∧ p ∣ 12321 ∧ ∀ q : ℕ, q.prime ∧ q ∣ 12321 → q ≤ p :=
sorry

end largest_prime_factor_12321_l207_207644


namespace m_mul_n_value_l207_207759

theorem m_mul_n_value :
  let m := (Finset.range (5+1)).sum (λ k, (-2)^k * binom 5 k)
  let n := ((Finset.range (6+1)).sum (λ k, (if (k = 5) then 1 else 0) * (-2)^k * binom 6 k))
             + ((Finset.range (6+1)).sum (λ k, (if (k = 2) then 1 else 0) * (-2)^k * binom 6 k) * (x^3).coeff 3)
  in m * n = 132 :=
by {
  let m := (Finset.range (5+1)).sum (λ k, (-2)^k * binom 5 k),
  let n := ((Finset.range (6+1)).sum (λ k, (if (k = 5) then (-2)^k * binom 6 k else 0)))
            + ((Finset.range (6+1)).sum (λ k, (if (k = 2) then (-2)^k * binom 6 k else 0)) * 1),
  have h : m = -1 := by sorry,
  have hn : n = -132 := by sorry,
  show m * n = 132,
  rw [h, hn],
  exact (by sorry : -1 * -132 = 132),
  sorry,
}

end m_mul_n_value_l207_207759


namespace area_of_circumcircle_of_isosceles_triangle_l207_207537

theorem area_of_circumcircle_of_isosceles_triangle :
  ∀ (r : ℝ) (π : ℝ), (∀ (a b c : ℝ)
  (h1 : a = 5) (h2 : b = 5) (h3 : c = 4),
  r = sqrt (a * b * (a + b + c) * (a + b - c)) / c →
  ∀ (area : ℝ), area = π * r ^ 2 →
  area = 13125 / 1764 * π) :=
  λ r π a b c h1 h2 h3 h_radius area h_area, sorry

end area_of_circumcircle_of_isosceles_triangle_l207_207537


namespace fraction_sum_equality_l207_207832

theorem fraction_sum_equality 
  (a b c x y z : ℝ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < x ∧ 0 < y ∧ 0 < z)
  (h_1 : a^2 + b^2 + c^2 = 49)
  (h_2 : x^2 + y^2 + z^2 = 64)
  (h_3 : ax + by + cz = 56) :
  (a + b + c) / (x + y + z) = 7 / 8 := 
by
  sorry

end fraction_sum_equality_l207_207832


namespace grasshopper_contradiction_l207_207487

noncomputable def can_reach_final_positions : Prop :=
  ∃ (jump_sequence : List (ℤ × ℤ × ℤ × ℤ)),
    jump_sequence.head = ((1, 0), (0, 0), (0, 1)) ∧
    jump_sequence.last = some ((0, 0), (-1, -1), (1, 1)) ∧
    ∀ (A B C A' B' C' : ℤ × ℤ) (P : List (ℤ × ℤ × ℤ × ℤ)),
      (P = ((A, B, C), (A', B', C')) ∧ P ∈ jump_sequence) →
      ∃ (X Y : ℤ × ℤ),
        (X ≠ Y ∧ (A' - A, B' - B, C' - C) = (X - A, Y - B, Y - C)) ∧
        ((A', B', C') = (X, Y, Y) ∨ (B', X, Y))  

theorem grasshopper_contradiction : ¬ can_reach_final_positions :=
sorry

end grasshopper_contradiction_l207_207487


namespace factorize_expression_l207_207620

variable (a x y : ℝ)

theorem factorize_expression : a^2 * (x - y) + 9 * (y - x) = (x - y) * (a + 3) * (a - 3) :=
by
  sorry

end factorize_expression_l207_207620


namespace sets_common_element_l207_207695

theorem sets_common_element {α : Type*} [decidable_eq α] (sets : fin 50 → finset α) (h : ∀ (T : finset (fin 50)), T.card = 30 → ∃ x, ∀ i ∈ T, x ∈ sets i) : 
  ∃ y, ∀ (i : fin 50), y ∈ sets i :=
by
  -- The proof needs to be filled in here.
  sorry

end sets_common_element_l207_207695


namespace number_of_households_using_both_brands_l207_207945

theorem number_of_households_using_both_brands (total_households neither onlyA : ℕ) (onlyB_factor both_brands : ℕ)
  (h_total : total_households = 240)
  (h_neither : neither = 80)
  (h_onlyA : onlyA = 60)
  (h_onlyB_factor : ∀ X, onlyB_factor = 3 * X)
  (h_both_brands : ∀ X, X = both_brands) :
  both_brands = 25 :=
by
  let X := both_brands
  have h_onlyB : 3X = 3 * X := by rw [h_onlyB_factor, h_both_brands]
  have h_eq : neither + onlyA + 3 * X + X = total_households := by simp [h_neither, h_onlyA, h_total]
  rw [h_total] at h_eq
  rw [h_neither, h_onlyA] at h_eq
  have h_simplified : 140 + 4 * X = 240 := by linarith
  have h_solve : 4 * X = 100 := by linarith
  have h_final : X = 25 := by linarith
  exact h_final

end number_of_households_using_both_brands_l207_207945


namespace area_of_triangle_MFO_l207_207627

-- Point definition
structure Point :=
(x : ℝ)
(y : ℝ)

-- Focus of the parabola
def focus : Point := ⟨real.sqrt 2, 0⟩

-- Origin
def origin : Point := ⟨0, 0⟩

-- Parabola condition
def on_parabola (p : Point) : Prop :=
  p.y^2 = 4 * real.sqrt 2 * p.x

-- Distance condition
def distance_to_focus (p : Point) : Prop :=
  real.sqrt ((p.x - focus.x)^2 + p.y^2) = 4 * real.sqrt 2

noncomputable def triangle_area (p q r : Point) : ℝ :=
  (1 / 2) * real.abs (p.x * (q.y - r.y) + q.x * (r.y - p.y) + r.x * (p.y - q.y))

-- Main theorem: area of the triangle MFO
theorem area_of_triangle_MFO (M : Point) 
  (h_parabola : on_parabola M)
  (h_distance : distance_to_focus M) :
  triangle_area origin focus M = 3 * real.sqrt 3 := 
by 
  sorry

end area_of_triangle_MFO_l207_207627


namespace area_of_rectangle_EFGH_l207_207368

-- Definitions for the conditions
def E : ℝ × ℝ := (-4, 10)
def F : ℝ × ℝ := (196, 58)
def G_x (x : ℝ) : ℝ × ℝ := (x, 62)

-- Main theorem statement
theorem area_of_rectangle_EFGH (x : ℤ) (h : G_x x = G_x x) : 
  let EF := ((F.1 - E.1) ^ 2 + (F.2 - E.2) ^ 2) ^ (1/2)
  let EG := ((G_x x).1 - E.1) ^ 2 + ((G_x x).2 - E.2) ^ 2) ^ (1/2)
  in (EF * EG) = 45938 := 
sorry

end area_of_rectangle_EFGH_l207_207368


namespace pyramid_volume_eq_l207_207594

noncomputable def volume_of_pyramid (base_length1 base_length2 height : ℝ) : ℝ :=
  (1 / 3) * base_length1 * base_length2 * height

theorem pyramid_volume_eq (base_length1 base_length2 height : ℝ) (h1 : base_length1 = 1) (h2 : base_length2 = 2) (h3 : height = 1) :
  volume_of_pyramid base_length1 base_length2 height = 2 / 3 := by
  sorry

end pyramid_volume_eq_l207_207594


namespace locus_of_harmonic_conjugate_is_a_straight_line_l207_207681

-- Point definitions on a secant intersecting a circle with arbitrary points C, B, D, and A

variables (A B D C O P Q : Point)

-- A line from a point A intersects a circle O at two distinct points B and D.
axiom secant_line : Line A → Circle O → Line B D

-- C is defined as the harmonic conjugate of B and D with respect to A.
axiom harmonic_conjugate : HarmonicConjugate A B D C

-- Theorem statement: Prove that the locus of harmonic conjugate C forms a straight line.
theorem locus_of_harmonic_conjugate_is_a_straight_line (A B D C : Point) (O : Circle) :
  secant_line A O → harmonic_conjugate A B D C → ∃ L : Line, ∀ A : Point, C ∈ L :=
sorry

end locus_of_harmonic_conjugate_is_a_straight_line_l207_207681


namespace first_player_can_win_l207_207912

theorem first_player_can_win (piles : List ℕ) (h : piles = [100]) : 
  ∃ w ∈ {true, false}, w = true :=
by
  -- conditions for the game
  -- 1. Two players take turns
  let player1 := true
  let player2 := false

  -- 2. Split piles logic
  -- We will use inductive game definitions.

  -- Initial pile condition
  have init_pile : ∃ p, p ∈ piles ∧ p = 100 := by
    use 100
    simp [piles, h]

  -- We assume further definitions and logic
  sorry

end first_player_can_win_l207_207912


namespace mutually_exclusive_not_contradictory_l207_207549

def hit_twice := ℕ  -- Representing the number of hits as natural numbers

def event_A (hits : hit_twice) : Prop := hits ≤ 1  -- Hit at most once
def event_B (hits : hit_twice) : Prop := 1 ≤ hits  -- Hit at least once
def event_C (hits : hit_twice) : Prop := 1 ≤ hits  -- Hit on the first try (at least once)
def event_D (hits : hit_twice) : Prop := hits = 0  -- Miss both times
def hit_exactly_once (hits : hit_twice) : Prop := hits = 1  -- Hit exactly once

theorem mutually_exclusive_not_contradictory :
  (∀ hits, event_D hits → ¬ hit_exactly_once hits) ∧ ¬ (∀ hits, event_D hits ∨ hit_exactly_once hits) :=
begin
  split,
  { intros hits hD hOnce,
    rw event_D at hD,
    rw hit_exactly_once at hOnce,
    contradiction },
  { intro h,
    have : event_D 1 := h 1,
    simp [event_D] at this,
    exact this }
end

end mutually_exclusive_not_contradictory_l207_207549


namespace banker_l207_207874

theorem banker's_discount (BD TD FV : ℝ) (hBD : BD = 18) (hTD : TD = 15) 
(h : BD = TD + (TD^2 / FV)) : FV = 75 := by
  sorry

end banker_l207_207874


namespace gcd_12345_6789_l207_207121

theorem gcd_12345_6789 : Int.gcd 12345 6789 = 3 :=
by
  sorry

end gcd_12345_6789_l207_207121


namespace reconstruct_square_from_octagon_l207_207179

theorem reconstruct_square_from_octagon
  (square: Type) [field square] -- Assuming the notion of a square in a Euclidean space
  (octagon: Type) [field octagon] -- Assuming the notion of an octagon in a Euclidean space
  (is_convex: Type → Prop) -- Convexity predicate
  (is_regular_octagon: octagon → Prop) -- Regular octagon predicate
  (dist_opposite_sides: octagon → square) -- Distance between opposite sides
  (oct: octagon) -- Our remaining octagon
  (pieces: ℕ) -- Number of pieces we cut the square into
  (is_convex_piece: ∀ i, i < pieces → is_convex i) -- All pieces are convex
  (five_pieces_lost: fin 5) -- Five pieces are lost
  : 
  ∃ (sq: square), -- There exists a square
    (is_square: square → Prop) -- Predicate to check if it is a square
    (is_square sq) -- The square we need to reconstruct
    (side_length: ∀ (sq: square), dist_opposite_sides oct = side_length sq) -- Distance equals side length of square
    (reconstructible: oct → square → Prop) -- Reconstruction predicate
    (reconstructible oct sq): -- The original question
    sorry

end reconstruct_square_from_octagon_l207_207179


namespace grain_distance_l207_207488

theorem grain_distance
    (d : ℝ) (v_church : ℝ) (v_cathedral : ℝ)
    (h_d : d = 400) (h_v_church : v_church = 20) (h_v_cathedral : v_cathedral = 25) :
    ∃ x : ℝ, x = 1600 / 9 ∧ v_church * x = v_cathedral * (d - x) :=
by
  sorry

end grain_distance_l207_207488


namespace car_a_travel_time_l207_207993

theorem car_a_travel_time :
  (CarA_speed : ℕ) (CarB_speed : ℕ) (CarB_time : ℕ) 
  (ratio : ℕ)
  (H1 : CarA_speed = 80)
  (H2 : CarB_speed = 100)
  (H3 : CarB_time = 2)
  (H4 : ratio = 2)
  : (CarA_time : ℕ) := 
    by sorry

end car_a_travel_time_l207_207993


namespace find_n_l207_207869

noncomputable def cube_probability_solid_color (num_cubes edge_length num_corner num_edge num_face_center num_center : ℕ)
  (corner_prob edge_prob face_center_prob center_prob : ℚ) : ℚ :=
  have total_corner_prob := corner_prob ^ num_corner
  have total_edge_prob := edge_prob ^ num_edge
  have total_face_center_prob := face_center_prob ^ num_face_center
  have total_center_prob := center_prob ^ num_center
  2 * (total_corner_prob * total_edge_prob * total_face_center_prob * total_center_prob)

theorem find_n : ∃ n : ℕ, cube_probability_solid_color 27 3 8 12 6 1
  (1/8) (1/4) (1/2) 1 = (1 / (2 : ℚ) ^ n) ∧ n = 53 := by
  use 53
  simp only [cube_probability_solid_color]
  sorry

end find_n_l207_207869


namespace total_paint_required_l207_207171

-- Definitions based on conditions
def num_statues : ℕ := 1000
def height_small_statue : ℝ := 3
def height_large_statue : ℝ := 9
def paint_large_statue : ℝ := 1  -- 1 pint of paint for 9 feet high statue

-- Theorem statement
theorem total_paint_required : 
  let surface_area_ratio := (height_small_statue / height_large_statue) ^ 2 in
  let paint_small_statue := paint_large_statue * surface_area_ratio in
  num_statues * paint_small_statue = 1000 / 9 :=
by
  sorry

end total_paint_required_l207_207171


namespace find_sin_theta_l207_207394

variables (a b c : ℝ^3)
variables (θ : ℝ)

axiom norm_a : ‖a‖ = 2
axiom norm_b : ‖b‖ = 4
axiom norm_c : ‖c‖ = 6
axiom cross_prod_eq : a × (a × b) = c

theorem find_sin_theta : sin θ = 3 / 8 :=
sorry

end find_sin_theta_l207_207394


namespace trapezoid_LM_value_l207_207809

theorem trapezoid_LM_value (K L M N P Q : Type) 
  (d1 d2 : ℝ)
  (h1 : d1 = 1)
  (h2 : d2 = 1)
  (height_eq : KM = 1)
  (KN_eq_MQ : KN = MQ)
  (LM_eq_MP : LM = MP) :
  LM = 1 / real.sqrt (real.sqrt 2) :=
by 
  sorry

end trapezoid_LM_value_l207_207809


namespace find_coordinates_of_P_l207_207422

structure Point where
  x : Int
  y : Int

def symmetric_origin (A B : Point) : Prop :=
  B.x = -A.x ∧ B.y = -A.y

def symmetric_y_axis (A B : Point) : Prop :=
  B.x = -A.x ∧ B.y = A.y

theorem find_coordinates_of_P :
  ∀ M N P : Point, 
  M = Point.mk (-4) 3 →
  symmetric_origin M N →
  symmetric_y_axis N P →
  P = Point.mk 4 3 := 
by 
  intros M N P hM hSymN hSymP
  sorry

end find_coordinates_of_P_l207_207422


namespace critical_point_at_one_function_nonnegative_exponential_inequality_l207_207301

noncomputable def f (a x : ℝ) : ℝ := (Real.log (1 + x)) - (a * x) / (1 + x)

theorem critical_point_at_one (a : ℝ) (h : a = 2) : 
  let f' x := (1 + x - a) / (1 + x)^2 
  in f'(1) = 0 := 
by 
  have h := h
  sorry

theorem function_nonnegative (a : ℝ) (h : 0 < a ∧ a ≤ 1) :
  ∀ x ∈ Set.Ici (0 : ℝ), f a x ≥ 0 :=
by
  have h := h
  sorry

theorem exponential_inequality :
  (2017 : ℝ)^2017 / (2016 : ℝ)^2017 > Real.exp 1 :=
by
  sorry

end critical_point_at_one_function_nonnegative_exponential_inequality_l207_207301


namespace ratio_seniors_to_juniors_l207_207585

variable (j s : ℕ)

-- Condition: \(\frac{3}{7}\) of the juniors participated is equal to \(\frac{6}{7}\) of the seniors participated
def participation_condition (j s : ℕ) : Prop :=
  3 * j = 6 * s

-- Theorem to be proved: the ratio of seniors to juniors is \( \frac{1}{2} \)
theorem ratio_seniors_to_juniors (j s : ℕ) (h : participation_condition j s) : s / j = 1 / 2 :=
  sorry

end ratio_seniors_to_juniors_l207_207585


namespace xiao_ming_should_choose_store_A_l207_207133

def storeB_cost (x : ℕ) : ℝ := 0.85 * x

def storeA_cost (x : ℕ) : ℝ :=
  if x ≤ 10 then x
  else 0.7 * x + 3

theorem xiao_ming_should_choose_store_A (x : ℕ) (h : x = 22) :
  storeA_cost x < storeB_cost x := by
  sorry

end xiao_ming_should_choose_store_A_l207_207133


namespace smallest_integer_cube_ends_in_580_l207_207662

theorem smallest_integer_cube_ends_in_580 :
  ∃ n : ℕ, (n > 0) ∧ (n^3 % 1000 = 580) ∧ (∀ m : ℕ, (m > 0) ∧ (m^3 % 1000 = 580) → n ≤ m) :=
by
  let n : ℕ := 36
  existsi n
  split
  norm_num
  split
  norm_num
  sorry

end smallest_integer_cube_ends_in_580_l207_207662


namespace playground_width_correct_l207_207919

-- Given conditions
variables (w : ℝ) -- width of the playground
variables (garden_width playground_length : ℝ)
variables (garden_perimeter : ℝ)
variables (garden_area playground_area : ℝ)

-- Conditions as definitions
def garden_width := 24
def playground_length := 16
def garden_perimeter := 64
def playground_area := playground_length * w
def garden_length : ℝ := 8 -- derived from perimeter condition
def garden_area := garden_width * garden_length

-- The theorem to prove
theorem playground_width_correct :
  garden_width = 24 ∧ playground_length = 16 ∧ garden_perimeter = 64 ∧ garden_area = playground_area → w = 12 :=
by
  intros hconds
  sorry

end playground_width_correct_l207_207919


namespace scientific_notation_correct_l207_207616

theorem scientific_notation_correct :
  scientific_notation 682000000 = (6.82, 8) :=
by
  sorry

end scientific_notation_correct_l207_207616


namespace second_part_sum_l207_207181

theorem second_part_sum :
  ∃ (x y : ℝ), 
  x + y = 2769 ∧ 
  (24 * x / 100) = (15 * y / 100) ∧ 
  y = 1704 :=
begin
  sorry
end

end second_part_sum_l207_207181


namespace heptagon_can_form_two_layered_quad_l207_207078

theorem heptagon_can_form_two_layered_quad (P : convex_poly) (h1 : result_folded_quad (P)) :
  P.sides = 7 → ∃ Q : convex_quad, Q.two_layered_convex = true :=
by sorry

end heptagon_can_form_two_layered_quad_l207_207078


namespace heart_beats_during_marathon_l207_207566

theorem heart_beats_during_marathon :
  (∃ h_per_min t1 t2 total_time,
    h_per_min = 140 ∧
    t1 = 15 * 6 ∧
    t2 = 15 * 5 ∧
    total_time = t1 + t2 ∧
    23100 = h_per_min * total_time) :=
  sorry

end heart_beats_during_marathon_l207_207566


namespace matrix_vector_product_l207_207592

theorem matrix_vector_product :
  let A := !![
    [3, -2],
    [-4, 5]
  ]
  let b := ![
    4, -2
  ]
  (A.mulVec b) = ![
    16, -26
  ] :=
by
  sorry

end matrix_vector_product_l207_207592


namespace coloring_exists_l207_207025

variable {X : Type*} (H : Set (Set X)) (κ : Cardinal)
variable [Nonempty (Ordinal.{u})] -- µ is strictly positive

theorem coloring_exists (hH : ∀ x ∈ X, {h ∈ H | x ∈ h}.size < κ) :
  ∃ f : X → κ, ∀ A ∈ H, ∃ x ∈ A, ∀ y ∈ A, x ≠ y → f x ≠ f y :=
sorry

end coloring_exists_l207_207025


namespace prob_y_intercept_gt_1_l207_207297

theorem prob_y_intercept_gt_1 (b : ℝ) (h : ∀ b, (-2 ≤ -b ∧ -b ≤ 3)) : 
  (measure_space.prob {b : ℝ | b > 1}) = 1 / 5 :=
sorry

end prob_y_intercept_gt_1_l207_207297


namespace opposite_of_2023_is_neg_2023_l207_207466

theorem opposite_of_2023_is_neg_2023 : (2023 + (-2023) = 0) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l207_207466


namespace digit_in_tens_place_of_smallest_even_number_is_five_l207_207453

theorem digit_in_tens_place_of_smallest_even_number_is_five :
  ∃ (n : ℕ), (n.to_digits = [1, 2, 3, 5, 8] ∨ n.to_digits = [1, 2, 3, 8, 5]) ∧ (n % 2 = 0 ∧ (n / 10) % 10 = 5) := sorry

end digit_in_tens_place_of_smallest_even_number_is_five_l207_207453


namespace cardinal_D_ge_cardinal_A_l207_207075

open Set

variables {n : ℕ}
variables {A : Set (Fin n → ℝ)}

def gamma (α β : Fin n → ℝ) : Fin n → ℝ :=
  fun i => abs (α i - β i)

def D (A : Set (Fin n → ℝ)) : Set (Fin n → ℝ) :=
  {γ | ∃ α β ∈ A, γ = gamma α β}

theorem cardinal_D_ge_cardinal_A (A : Set (Fin n → ℝ)) :
  A.Finite → (D A).card ≥ A.card :=
by
  intro hA
  sorry

end cardinal_D_ge_cardinal_A_l207_207075


namespace happy_numbers_transformation_l207_207255

-- Define what it means for a three-digit number to be a "happy number"
def is_happy_number (n : ℕ) : Prop :=
  let a := n / 100 in
  let b := (n % 100) / 10 in
  let c := n % 10 in
  a + b - c = 6

-- Define the transformation described in the problem
def transform_number (m : ℕ) : ℕ :=
  let a := m / 100 in
  let b := (m % 100) / 10 in
  let c := m % 10 in
  2 * c * 100 + a * 10 + b

-- Define the main theorem
theorem happy_numbers_transformation (m : ℕ) :
  is_happy_number m →
  is_happy_number (transform_number m) →
  m = 532 ∨ m = 464 :=
sorry

end happy_numbers_transformation_l207_207255


namespace c_alone_finishes_job_in_7_5_days_l207_207506

theorem c_alone_finishes_job_in_7_5_days (A B C : ℝ) (h1 : A + B = 1 / 15) (h2 : A + B + C = 1 / 5) :
  1 / C = 7.5 :=
by
  -- The proof is omitted
  sorry

end c_alone_finishes_job_in_7_5_days_l207_207506


namespace area_of_triangle_l207_207014

theorem area_of_triangle (a c : ℝ) (A : ℝ) (h_a : a = 2) (h_c : c = 2 * Real.sqrt 3) (h_A : A = Real.pi / 6) :
  ∃ (area : ℝ), area = 2 * Real.sqrt 3 ∨ area = Real.sqrt 3 :=
by
  sorry

end area_of_triangle_l207_207014


namespace angle_BMD_right_l207_207277

/-- Given a parallelogram ABCD and a point K such that AK = BD, if M is the midpoint of CK, 
then the angle ∠BMD is a right angle (90 degrees). -/
theorem angle_BMD_right {A B C D K M : Point} (h_parallelogram : parallelogram A B C D)
  (h_eq : dist A K = dist B D) (h_midpoint : midpoint C K M) :
  ∠ B M D = 90 := by
  sorry

end angle_BMD_right_l207_207277


namespace sequence_2010_is_2_l207_207100

noncomputable def sequence (n : ℕ) : ℝ :=
nat.rec_on n (1/2) (λ n a_n, 1 - 1/a_n)

theorem sequence_2010_is_2 : sequence 2010 = 2 := 
sorry

end sequence_2010_is_2_l207_207100


namespace mouse_jump_less_l207_207092

theorem mouse_jump_less (grasshopper_jump frog_extra mouse_extra : ℕ) :
  grasshopper_jump = 14 →
  frog_extra = 37 →
  mouse_extra = 21 →
  (grasshopper_jump + frog_extra) - (grasshopper_jump + mouse_extra) = 16 :=
by
  intros hg hf hm
  simp [hg, hf, hm]
  sorry

end mouse_jump_less_l207_207092


namespace largest_power_of_two_dividing_product_of_binomials_l207_207628

theorem largest_power_of_two_dividing_product_of_binomials :
  let f : ℕ → ℕ := λ k, (nat.choose (2 * k) k)
  let product := ∏ k in finset.range 65, f k
  ∃ n : ℕ, 2^n ∣ product ∧ n = 193 := 
sorry

end largest_power_of_two_dividing_product_of_binomials_l207_207628


namespace tank_capacity_l207_207933

-- Define the conditions
variable (C : ℝ)   -- Capacity of the tank in litres
variable (rate_outlet : ℝ := C / 5)  -- Outlet pipe rate in litres per hour
variable (rate_inlet : ℝ := 8 * 60)  -- Inlet pipe rate in litres per hour (8 lit/min)
variable (time_with_inlet : ℝ := 8)  -- Time to empty tank with both pipes in hours

-- Effective rate of emptying with both pipes open
def effective_rate : ℝ := rate_outlet - rate_inlet

-- Capacity condition when both pipes are open
axiom capacity_condition : rate_outlet - rate_inlet = C / time_with_inlet

-- Proof statement that the capacity is 1280 litres
theorem tank_capacity : C = 1280 := by
  -- conditions and the required relation between variables
  have h1 : rate_outlet = C / 5 := rfl
  have h2 : rate_inlet = 480 := rfl
  have h3 : effective_rate = C / time_with_inlet := rfl
  have h4 : C / 5 - 480 = C / 8 := capacity_condition
  -- solve by steps implied in the solution
  sorry

end tank_capacity_l207_207933


namespace lim_f_iterate_l207_207027

open MeasureTheory

noncomputable def f (x : ℝ) : ℝ := (1 + Real.cos (2 * Real.pi * x)) / 2

noncomputable def f_iterate (n : ℕ) : ℝ → ℝ :=
match n with
| 0 => id
| n + 1 => f ∘ f_iterate n

theorem lim_f_iterate (x : ℝ) (hx : x ∈ set.univ) :
  ∀ᶠ (x : ℝ) in (volume : measure ℝ).ae, tendsto (λ n, f_iterate n x) at_top (𝓝 1) :=
sorry

end lim_f_iterate_l207_207027


namespace amaya_movie_watching_time_l207_207195

theorem amaya_movie_watching_time :
  let t1 := 30 + 5
  let t2 := 20 + 7
  let t3 := 10 + 12
  let t4 := 15 + 8
  let t5 := 25 + 15
  let t6 := 15 + 10
  t1 + t2 + t3 + t4 + t5 + t6 = 172 :=
by
  sorry

end amaya_movie_watching_time_l207_207195


namespace ant_ways_to_reach_l207_207200

theorem ant_ways_to_reach : 
  let n := 4020
  let a := 2010
  let b := 1005
  (nat.choose n b)^2 = nat.choose n (n - b) * nat.choose n b := by
sorry

end ant_ways_to_reach_l207_207200


namespace area_of_square_plot_l207_207125

theorem area_of_square_plot (s : ℕ) (price_per_foot total_cost: ℕ)
  (h_price : price_per_foot = 58)
  (h_total_cost : total_cost = 3944) :
  (s * s = 289) :=
by
  sorry

end area_of_square_plot_l207_207125


namespace polynomial_irreducible_l207_207833

theorem polynomial_irreducible (n : ℤ) (hn : n > 1) :
  ¬ ∃ (g h : ℤ[X]),
    g.degree ≥ 1 ∧ 
    h.degree ≥ 1 ∧ 
    f = g * h :=
by 
  let f : ℤ[X] := X^n + 5*X^(n-1) + 3
  sorry

end polynomial_irreducible_l207_207833


namespace total_smaller_cubes_l207_207953

theorem total_smaller_cubes (n : ℕ) (painted: cube n) (cut_into_smaller_cubes: cube n → ℕ → cube (n /3)) (one_colorless_cube: ∃ c, colorless c ∧ c ∈ cube (n /3) ∧ n ≥ 3) :
    ∑ k in finset.range (n / 3), ∑ j in finset.range (n / 3), ∑ i in finset.range (n / 3), 1 = 27 := 
sorry

end total_smaller_cubes_l207_207953


namespace least_common_positive_period_l207_207220

theorem least_common_positive_period {f : ℝ → ℝ}
  (h : ∀ x, f(x + 6) + f(x - 6) = f(x)) : ∃ p > 0, (∀ x, f(x) = f(x + p)) ∧ (∀ q > 0, (∀ x, f(x) = f(x + q)) → p ≤ q) :=
by
  -- Proof skipped
  sorry

end least_common_positive_period_l207_207220


namespace maximum_value_a_l207_207678

def P (a : ℂ) (x : ℂ) : ℂ := (1 - 2 * x + a * x^2) ^ 8

noncomputable def coeff_x4 (a : ℂ) : ℂ :=
  ∑ n1 n2 n3 in finset.nat.antidiagonal 8,
    if n1 + n2 + n3 = 8 ∧ n2 + 2 * n3 = 4
    then (nat.choose 8 n1 * nat.choose (8 - n1) n2) * (-2) ^ n2 * a ^ n3 else 0

theorem maximum_value_a :
  ∃ a : ℂ, coeff_x4 a = -1540 ∧ a = -6 + complex.I * complex.sqrt 59 :=
by
  sorry

end maximum_value_a_l207_207678


namespace sin_B_sin_C_l207_207382

open Real

noncomputable def triangle_condition (A B C : ℝ) (a b c : ℝ) : Prop :=
  cos (2 * A) - 3 * cos (B + C) = 1 ∧
  (1 / 2) * b * c * sin A = 5 * sqrt 3 ∧
  b = 5

theorem sin_B_sin_C {A B C a b c : ℝ} (h : triangle_condition A B C a b c) :
  (sin B) * (sin C) = 5 / 7 := 
sorry

end sin_B_sin_C_l207_207382


namespace parabola_directrix_l207_207086

theorem parabola_directrix (x y : ℝ) : 
    (x^2 = (1/2) * y) -> (y = -1/8) :=
sorry

end parabola_directrix_l207_207086


namespace a2_plus_a3_eq_neg1_l207_207344

theorem a2_plus_a3_eq_neg1 (a : ℕ → ℤ) (x : ℤ) : 
  (2 + x) * (1 - x) ^ 6 = a(0) + a(1) * x + a(2) * x^2 + a(3) * x^3 + a(4) * x^4 + a(5) * x^5 + a(6) * x^6 + a(7) * x^7 →
  a(2) + a(3) = -1 := 
  by sorry

end a2_plus_a3_eq_neg1_l207_207344


namespace inverse_proportion_relationship_l207_207284

theorem inverse_proportion_relationship :
  ∀ (y₁ y₂ : ℝ), y₁ = -3/1 → y₂ = -3/2 → y₁ < y₂ :=
by
  -- Definitions corresponding to the conditions
  assume y₁ y₂,
  assume h₁ : y₁ = -3 / 1,
  assume h₂ : y₂ = -3 / 2,
  sorry

end inverse_proportion_relationship_l207_207284


namespace zachary_additional_money_needed_l207_207260

noncomputable def total_cost : ℝ := 3.756 + 2 * 2.498 + 11.856 + 4 * 1.329 + 7.834
noncomputable def zachary_money : ℝ := 24.042
noncomputable def money_needed : ℝ := total_cost - zachary_money

theorem zachary_additional_money_needed : money_needed = 9.716 := 
by 
  sorry

end zachary_additional_money_needed_l207_207260


namespace max_path_length_rect_3x5_l207_207524

-- Define the basic structure and conditions of the rectangle and paths
variables {R : Type} -- Define the type R for rectangle vertices
-- Define a 3x5 rectangle
def rectangle_3x5 : Prop := ∃ v : R -> ℕ × ℕ, 
  v 'A = (0, 0) ∧ v 'B = (0, 5) ∧ v 'C = (3, 5) ∧ v 'D = (3, 0)

-- Define paths along unit squares without retreading edges
def path (start end : R) (length : ℕ) : Prop := 
  ∃ edges : list (R × R), 
    nodup edges ∧ -- no repeated edges
    (∀ (v1 v2 : R), (v1, v2) ∈ edges → v1 ≠ v2) ∧ 
    length = edges.length

-- Define two opposite vertices
variables (A C : R) 

-- Maximum path length without retreading edge is
theorem max_path_length_rect_3x5 
  (h : rectangle_3x5) 
  (h_path : path A C 30) : 
  path A C 30 := sorry

end max_path_length_rect_3x5_l207_207524


namespace cube_division_points_on_sphere_l207_207610

theorem cube_division_points_on_sphere (a : ℝ) (h : a > 0) : 
  ∃ (S : Sphere) (P : Point → Prop), 
  (∀ (q : Point), P q → q ∈ S) ∧ 
  (let r := (a * Real.sqrt 19) / 6 in
   4 * Real.pi * r^2 = (19 * Real.pi * a^2) / 9) := 
by sorry

end cube_division_points_on_sphere_l207_207610


namespace x2022_equals_1_l207_207676

noncomputable def sequence (n : ℕ) : ℤ :=
if n = 1 then 1 else
if n = 2 then 1 else
if n = 3 then -1 else
sequence (n-1) * sequence (n-3)

theorem x2022_equals_1 : sequence 2022 = 1 :=
sorry

end x2022_equals_1_l207_207676


namespace strategic_important_l207_207005

-- Definitions required based directly on conditions in a)
variables {Bases : Type} {Roads : Type} 
variable [fintype Bases]
variable [decidable_eq Bases]

def important (closed_roads : set Roads) (is_connected : (Bases → Bases → set Roads)) : Prop :=
∃ b1 b2 : Bases, b1 ≠ b2 ∧ (is_connected b1 b2).inter closed_roads = ∅ 

def strategic (closed_roads : set Roads) (is_connected : (Bases → Bases → set Roads)) : Prop :=
  ∀ closed_roads' ⊂ closed_roads, ¬important closed_roads' is_connected

-- The solution statement based on c)
theorem strategic_important 
  (is_connected : (Bases → Bases → set Roads)) (set1 set2 : set Roads)
  (h1 : strategic set1 is_connected) (h2 : strategic set2 is_connected)
  (disjoint_sets : set1 ∩ set2 = ∅) :
sorry

end strategic_important_l207_207005


namespace probability_of_pink_gumball_l207_207818

-- Definitions based on conditions from part a)
def probability_color (color : Type) : ℝ := 1 / 5

-- Given conditions
axiom prob_sequence : probability_color Green * probability_color Blue * probability_color Blue = 9 / 343

-- Proof statement
theorem probability_of_pink_gumball : probability_color Pink = 1 / 5 :=
by
  -- Normally, a proof would be provided here
  sorry

end probability_of_pink_gumball_l207_207818


namespace coprime_sum_floors_eq_l207_207425

theorem coprime_sum_floors_eq :
  ∀ (p q : ℕ), Nat.coprime p q →
    (∑ k in Finset.range (q-1), ⌊ (k+1 : ℝ) * (p : ℝ) / (q.R) ⌋ = (p-1)*(q-1)/2) ∧
    (∑ k in Finset.range (p-1), ⌊ (k+1 : ℝ) * (q : ℝ) / (p : ℝ) ⌋ = (p-1)*(q-1)/2) :=
by sorry

end coprime_sum_floors_eq_l207_207425


namespace trapezoid_LM_value_l207_207810

theorem trapezoid_LM_value (K L M N P Q : Type) 
  (d1 d2 : ℝ)
  (h1 : d1 = 1)
  (h2 : d2 = 1)
  (height_eq : KM = 1)
  (KN_eq_MQ : KN = MQ)
  (LM_eq_MP : LM = MP) :
  LM = 1 / real.sqrt (real.sqrt 2) :=
by 
  sorry

end trapezoid_LM_value_l207_207810


namespace ant_path_count_l207_207198

theorem ant_path_count :
  let binom := Nat.choose 4020 1005 in
  ∃ f : Fin 2 → ℕ, 
  (f 0 = 2010 ∧ f 1 = 2010 ∧ (∑ x : Fin 4020, if x.mod 2 = 0 then 1 else -1) = 4020) →
  binom * binom = (Nat.choose 4020 1005) ^ 2 := 
by
  sorry

end ant_path_count_l207_207198


namespace relationship_among_abc_l207_207290

noncomputable def a : ℝ := 6 ^ 0.7
noncomputable def b : ℝ := 0.7 ^ 6
noncomputable def c : ℝ := Real.log 6 / Real.log 0.7

theorem relationship_among_abc : c < b ∧ b < a :=
by
  sorry

end relationship_among_abc_l207_207290


namespace exists_integers_sum_zero_l207_207231

open nat

theorem exists_integers_sum_zero (a b c : ℤ) : 
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 
  a + b + c = 0 ∧ 
  ∃ k : ℕ, a^13 + b^13 + c^13 = k^2 := 
sorry

end exists_integers_sum_zero_l207_207231


namespace purely_imaginary_product_l207_207624

open Complex

theorem purely_imaginary_product (x : ℝ) : 
  (x^4 + x^3 + x^2 + 2*x - 2 = 0) → 
  (∃ y : ℝ, (x + Complex.i) * ((x^2 + 1) + Complex.i) * ((x^2 + 2) + Complex.i) = Complex.i * y) :=
begin
  sorry
end

end purely_imaginary_product_l207_207624


namespace problem1_problem2_problem3_problem4_l207_207365

variable {α : Type*} [AddGroup α] [Module ℤ α]

/-- Arithmetic sequence properties -/
def arithmetic_seq (a : ℕ → α) := ∃ d, ∀ n, a (n + 1) - a n = d

/-- Sum of the first n terms of the sequence -/
def sum_seq (a : ℕ → α) : ℕ → α
  | 0       => 0
  | (n + 1) => sum_seq a n + a n

variable (a : ℕ → ℤ)

-- First statement
theorem problem1 (h : arithmetic_seq a) (h₁ : a 4 + a 14 = 2) : sum_seq a 17 = 17 := by
  sorry

-- Second statement
theorem problem2 (h : arithmetic_seq a) (h₂ : a 11 = 10) : sum_seq a 21 = 210 := by
  sorry

-- Third statement
theorem problem3 (h : arithmetic_seq a) (h₃ : sum_seq a 11 = 55) : a 6 = 5 := by
  sorry

-- Fourth statement
theorem problem4 (h : arithmetic_seq a) (h₄ : sum_seq a 8 = 100) (h₅ : sum_seq a 16 = 392) : sum_seq a 24 = 976 := by
  sorry

end problem1_problem2_problem3_problem4_l207_207365


namespace game_ends_in_16_rounds_l207_207955

-- Definitions for the problem
def initial_tokens : List ℕ := [17, 16, 15, 14]

def tokens_after_rounds (tokens : List ℕ) : List ℕ :=
  let max_tokens := tokens.maximum.getD 0
  tokens.map (fun t => 
    if t = max_tokens then t - 4 else t + 1
  )

def game_ends (tokens : List ℕ) : Prop :=
  tokens.any (· = 0)

-- Theorem stating the main problem
theorem game_ends_in_16_rounds : ∃ n, n = 16 ∧ game_ends (iterate tokens_after_rounds n initial_tokens) := sorry

end game_ends_in_16_rounds_l207_207955


namespace oblique_asymptote_eq_l207_207120

noncomputable def oblique_asymptote (f : ℚ(x)) : ℚ(x) :=
  ((3 * x^2 + 4 * x + 5) / (x + 4))

theorem oblique_asymptote_eq : oblique_asymptote (3 * x^2 + 4 * x + 5) = 3 * x - 8 := by
  sorry

end oblique_asymptote_eq_l207_207120


namespace first_month_sale_l207_207170

theorem first_month_sale 
(sale_2 sale_3 sale_4 sale_5 sale_6 : ℕ)
(avg_sale : ℕ) 
(h_avg: avg_sale = 6500)
(h_sale2: sale_2 = 6927)
(h_sale3: sale_3 = 6855)
(h_sale4: sale_4 = 7230)
(h_sale5: sale_5 = 6562)
(h_sale6: sale_6 = 4791)
: sale_1 = 6635 := by
  sorry

end first_month_sale_l207_207170


namespace maximum_value_expression_l207_207654

theorem maximum_value_expression :
  (∀ x y : Real, 
     -10.4168 * (3 + 2 * Real.sqrt (7 - Real.sqrt 2) * Real.cos y - Real.cos (2 * y)) ≤ 
     (\(Real.sqrt (3 - Real.sqrt 2) * Real.sin x - Real.sqrt (2 * (1 + Real.cos (2 * x))) - 
       1) * (3 + 2 * Real.sqrt (7 - Real.sqrt 2) * Real.cos y - Real.cos (2 * y))) ≤ 9) :=
by
  sorry

end maximum_value_expression_l207_207654


namespace minimum_of_f_l207_207715

noncomputable def f (x : ℝ) : ℝ := 2^x + 1 / (2^x - 1)

theorem minimum_of_f : ∃ (x : ℝ) (hx : x > 0), (∀ y > 0, f y ≥ 3) ∧ f x = 3 :=
by
  use 1
  split
  { intro y hy
    sorry }
  { exact rfl }

end minimum_of_f_l207_207715


namespace number_of_rational_solutions_l207_207890

namespace RationalSolutions

def system_of_equations (x y z : ℚ) : Prop :=
  x + y + z = 0 ∧
  xyz + z = 0 ∧
  xy + yz + xz + y = 0

theorem number_of_rational_solutions : 
  (∃ x y z : ℚ, system_of_equations x y z) ∧ 
  ∀ (x₁ y₁ z₁) (x₂ y₂ z₂ : ℚ), 
    system_of_equations x₁ y₁ z₁ →
    system_of_equations x₂ y₂ z₂ →
    (x₁ = x₂ ∧ y₁ = y₂ ∧ z₁ = z₂) ↔ 
    (x₁, y₁, z₁) = (0, 0, 0) ∨ (x₁, y₁, z₁) = (-1, 1, 0) ∨ 
    (x₂, y₂, z₂) = (0, 0, 0) ∨ (x₂, y₂, z₂) = (-1, 1, 0)
:=
begin
  sorry
end

end RationalSolutions

end number_of_rational_solutions_l207_207890


namespace symmetry_probability_l207_207178

theorem symmetry_probability (n : ℕ) (h : n = 11) (P : ℕ) (hp : P = 6) :
  let total_points := n * n - 1 in
  let symmetric_points := 4 * (2 * (n - 1)) in
  let probability := symmetric_points / total_points in
  probability = 1 / 3 :=
by
  -- Mathematical definitions
  let n := 11
  let total_points := n * n - 1
  let symmetric_points := 4 * (2 * (n - 1))
  let probability := symmetric_points / total_points
  -- Given conditions
  have h_n : n = 11 := rfl
  have h_total_points : total_points = 120 := by norm_num
  have h_symmetric_points : symmetric_points = 40 := by norm_num
  -- Calculating the probability
  suffices : probability = 1 / 3, exact this
  calc
    probability = 40 / 120 : by { simp [total_points, symmetric_points], }
             ... = 1 / 3   : by norm_num

end symmetry_probability_l207_207178


namespace correct_calculation_l207_207744

theorem correct_calculation (x : ℕ) (h : x - 749 = 280) : x + 479 = 1508 := 
by
  -- We perform the necessary operations to establish the desired proof.
  have hx : x = 280 + 749 := by linarith,
  rw [hx],
  linarith

end correct_calculation_l207_207744


namespace heather_final_blocks_l207_207739

def heather_initial_blocks : ℝ := 86.0
def jose_shared_blocks : ℝ := 41.0

theorem heather_final_blocks : heather_initial_blocks + jose_shared_blocks = 127.0 :=
by
  sorry

end heather_final_blocks_l207_207739


namespace greatest_possible_avg_speed_l207_207981

theorem greatest_possible_avg_speed (initial_odometer : ℕ) (max_speed : ℕ) (time_hours : ℕ) (max_distance : ℕ) (target_palindrome : ℕ) :
  initial_odometer = 12321 →
  max_speed = 80 →
  time_hours = 4 →
  (target_palindrome = 12421 ∨ target_palindrome = 12521 ∨ target_palindrome = 12621 ∨ target_palindrome = 12721 ∨ target_palindrome = 12821 ∨ target_palindrome = 12921 ∨ target_palindrome = 13031) →
  target_palindrome - initial_odometer ≤ max_distance →
  max_distance = 300 →
  target_palindrome = 12621 →
  time_hours = 4 →
  target_palindrome - initial_odometer = 300 →
  (target_palindrome - initial_odometer) / time_hours = 75 :=
by 
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end greatest_possible_avg_speed_l207_207981


namespace circus_ticket_total_cost_l207_207413

def number_of_adults := 2
def number_of_children := 5
def price_per_adult := 44.00
def price_per_child := 28.00
def total_number_of_tickets := number_of_adults + number_of_children
def discount_threshold := 6
def discount_rate := 0.10

def total_cost_without_discount := (number_of_adults * price_per_adult) + (number_of_children * price_per_child)
def discount_amount := if total_number_of_tickets > discount_threshold then total_cost_without_discount * discount_rate else 0.0
def total_cost_with_discount := total_cost_without_discount - discount_amount

theorem circus_ticket_total_cost :
  total_cost_with_discount = 205.20 :=
by
  sorry

end circus_ticket_total_cost_l207_207413


namespace sum_of_triangles_l207_207445

def triangle (a b c : ℕ) : ℕ := a * b + c

theorem sum_of_triangles :
  triangle 3 2 5 + triangle 4 1 7 = 22 :=
by
  sorry

end sum_of_triangles_l207_207445


namespace angelina_speed_l207_207205

theorem angelina_speed (v : ℝ) (h₁ : ∀ t : ℝ, t = 100 / v) (h₂ : ∀ t : ℝ, t = 180 / (2 * v)) 
  (h₃ : ∀ d t : ℝ, 100 / v - 40 = 180 / (2 * v)) : 
  2 * v = 1 / 2 :=
by
  sorry

end angelina_speed_l207_207205


namespace repetend_fraction_7_19_l207_207246

theorem repetend_fraction_7_19 : decimal_repetend (7 / 19) = 368421052631578947 := by
  sorry

end repetend_fraction_7_19_l207_207246


namespace remaining_seconds_l207_207061

theorem remaining_seconds (s x : ℝ) (start_time duration : ℝ) (angle_hour_hand angle_minute_hand : ℝ) 
  (H1 : start_time = 0) 
  (H2 : duration = 3600) 
  (H3 : angle_hour_hand = x) 
  (H4 : angle_minute_hand = 360 - x) 
  (H5 : s = 120 * x) 
  (H6 : s = 10 * (360 - x)) 
  : duration - s.to_nat = 277 := 
by
  sorry

end remaining_seconds_l207_207061


namespace soap_bubble_radius_l207_207177

/-- Given a spherical soap bubble that divides into two equal hemispheres, 
    each having a radius of 6 * (2 ^ (1 / 3)) cm, 
    show that the radius of the original bubble is also 6 * (2 ^ (1 / 3)) cm. -/
theorem soap_bubble_radius (r : ℝ) (R : ℝ) (π : ℝ) 
  (h_r : r = 6 * (2 ^ (1 / 3)))
  (h_volume_eq : (4 / 3) * π * R^3 = (4 / 3) * π * r^3) : 
  R = 6 * (2 ^ (1 / 3)) :=
by
  sorry

end soap_bubble_radius_l207_207177


namespace smallest_integer_k_l207_207498

theorem smallest_integer_k (some_exponent : ℤ) (k : ℤ) (h : k = 6) : 64 ^ k > 4 ^ some_exponent → 18 > some_exponent :=
by sorry

end smallest_integer_k_l207_207498


namespace gain_percent_correct_l207_207511

theorem gain_percent_correct (C S : ℝ) (h : 50 * C = 28 * S) : 
  ( (S - C) / C ) * 100 = 1100 / 14 :=
by
  sorry

end gain_percent_correct_l207_207511


namespace anthony_success_rate_increase_l207_207206

theorem anthony_success_rate_increase :
  let initial_made := 7
      initial_attempts := 20
      additional_attempts := 28
      additional_success_rate := 3 / 4
      initial_rate := initial_made / initial_attempts
      additional_made := additional_success_rate * additional_attempts
      total_attempts := initial_attempts + additional_attempts
      total_made := initial_made + additional_made
      overall_rate := total_made / total_attempts
      improvement := overall_rate - initial_rate
      improvement_percentage := improvement * 100
  in improvement_percentage ≈ 23 :=
by
  sorry

end anthony_success_rate_increase_l207_207206


namespace trihedral_angle_inequality_l207_207478

structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

structure Angle (A B C : Point) :=
  (value : ℝ)  -- Placeholder for actual angle calculation

def trihedral_angle_sum (S A B C : Point) : ℝ :=
  (Angle S A B).value + (Angle S B C).value + (Angle S C A).value

theorem trihedral_angle_inequality
  (S A B C C' : Point)
  (h : C' = C) -- Placeholder for defining "SC' lies inside the trihedral angle SABC"
  : trihedral_angle_sum S A B C > trihedral_angle_sum S A B C' :=
sorry

end trihedral_angle_inequality_l207_207478


namespace max_coefficient_term_l207_207295

theorem max_coefficient_term (n : ℕ) (h : 5 ≤ n) (hn : (3 * x + 1 / x) = 16) :
  max_coefficient_term ((3 * x) + (1 / x)) 16 = 9 :=
sorry

end max_coefficient_term_l207_207295


namespace bricks_of_other_types_l207_207051

theorem bricks_of_other_types (A B total other: ℕ) (hA: A = 40) (hB: B = A / 2) (hTotal: total = 150) (hSum: total = A + B + other): 
  other = 90 :=
by sorry

end bricks_of_other_types_l207_207051


namespace coefficient_of_x6_in_q_squared_l207_207751

def q (x : ℝ) : ℝ := x^5 - 4 * x^3 + 3

theorem coefficient_of_x6_in_q_squared (x : ℝ) :
  let q_squared := q x * q x in
  (polynomial.coeff q_squared 6 = 16) :=
begin
  sorry
end

end coefficient_of_x6_in_q_squared_l207_207751


namespace largest_prime_factor_12321_l207_207641

theorem largest_prime_factor_12321 : 
  ∃ p : ℕ, prime p ∧ p ∣ 12321 ∧ ∀ q : ℕ, prime q ∧ q ∣ 12321 → q ≤ p :=
begin
  use 83,
  split,
  { -- Prove that 83 is a prime number
    sorry },
  split,
  { -- Prove that 83 divides 12321
    sorry },
  { -- Prove that any other prime factor of 12321 is less than or equal to 83
    sorry }
end

end largest_prime_factor_12321_l207_207641


namespace area_of_circumcircle_of_isosceles_triangle_l207_207536

theorem area_of_circumcircle_of_isosceles_triangle :
  ∀ (r : ℝ) (π : ℝ), (∀ (a b c : ℝ)
  (h1 : a = 5) (h2 : b = 5) (h3 : c = 4),
  r = sqrt (a * b * (a + b + c) * (a + b - c)) / c →
  ∀ (area : ℝ), area = π * r ^ 2 →
  area = 13125 / 1764 * π) :=
  λ r π a b c h1 h2 h3 h_radius area h_area, sorry

end area_of_circumcircle_of_isosceles_triangle_l207_207536


namespace local_minimum_conditions_l207_207296

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x - (1 / 2) * x^2 + b * x

theorem local_minimum_conditions (a b : ℝ) (h_min : ∃ x > 0, ∀ y > 0, f a b x ≤ f a b y) : 
  a < 0 ∧ b > 0 := 
begin
  sorry
end

end local_minimum_conditions_l207_207296


namespace jessica_balloon_count_l207_207388

theorem jessica_balloon_count :
  (∀ (joan_initial_balloon_count sally_popped_balloon_count total_balloon_count: ℕ),
  joan_initial_balloon_count = 9 →
  sally_popped_balloon_count = 5 →
  total_balloon_count = 6 →
  ∃ (jessica_balloon_count: ℕ),
    jessica_balloon_count = total_balloon_count - (joan_initial_balloon_count - sally_popped_balloon_count) →
    jessica_balloon_count = 2) :=
by
  intros joan_initial_balloon_count sally_popped_balloon_count total_balloon_count j1 j2 t1
  use total_balloon_count - (joan_initial_balloon_count - sally_popped_balloon_count)
  sorry

end jessica_balloon_count_l207_207388


namespace general_formula_exists_find_m_l207_207048

-- Define the geometric sequence and conditions
def sequence (a₁ q : ℕ) (n : ℕ) : ℕ := a₁ * q^(n-1)
def condition1 (a₁ q : ℕ) : Prop := a₁ + a₁ * q = 4
def condition2 (a₁ q : ℕ) : Prop := a₁ * q^2 - a₁ = 8

-- The formula to be proven
def formula (a₁ q : ℕ) : (ℕ → ℕ) := 
  λ n, 3^(n - 1)

-- Main theorem for the general formula
theorem general_formula_exists (a₁ q : ℕ) 
  (h1 : condition1 a₁ q) (h2 : condition2 a₁ q) : 
  sequence a₁ q = formula a₁ q := sorry

-- Define the sum of the logarithmic sequence
def log_sequence_sum (n : ℕ) : ℕ := 
  n * (n - 1) / 2

-- The summation condition for the sequence
def summation_condition (m : ℕ) : Prop :=
  log_sequence_sum m + log_sequence_sum (m + 1) =
  log_sequence_sum (m + 3)

-- Main theorem for the value of m
theorem find_m (m : ℕ) : summation_condition m → m = 6 := sorry

end general_formula_exists_find_m_l207_207048


namespace complex_magnitude_l207_207294

theorem complex_magnitude (z : ℂ) (h : (3 - 4 * complex.I) * z = 4 + 3 * complex.I) : complex.abs z = 1 :=
by
  sorry

end complex_magnitude_l207_207294


namespace find_error_pages_l207_207891

noncomputable def correct_sum (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem find_error_pages (p q : ℕ) (n : ℕ) 
(h1 : correct_sum n = 4005) 
(h2 : p + q = 25) 
(h3 : p ≠ q) 
(h4 : 1 ≤ p ∧ p ≤ n) 
(h5 : 1 ≤ q ∧ q ≤ n) :
p = 12 ∧ q = 13 :=
by {
  have h0 : correct_sum 89 = 4005 := by norm_num,
  have h12 : p + q = 25 := by assumption,
  sorry
}

end find_error_pages_l207_207891


namespace find_c_plus_d_l207_207421

-- Define the square WXYZ with points Q, O3, O4
variables {W X Y Z Q O3 O4 : Point}
variables {c d : ℕ}

-- Assume WZ is length 10 and angle O3QO4 is 90 degrees
axiom WZ_length : dist W Z = 10
axiom angle_O3QO4_right : angle O3 Q O4 = 90

-- Assume that O3 and O4 are circumcenters of triangles WZQ and CYQ
def is_circumcenter (O : Point) (A B C : Triangle) : Prop := 
  is_perp_bisector O A B ∧ is_perp_bisector O B C ∧ is_perp_bisector O C A

axiom O3_is_circumcenter_WZQ : is_circumcenter O3 W Z Q
axiom O4_is_circumcenter_CYQ : is_circumcenter O4 C Y Q

-- Assume Q lies on diagonal AC with WQ > CQ
axiom point_Q_on_AC : lies_on Q (line_through A C) ∧ dist W Q > dist C Q

-- Given the conditions, prove that WQ = sqrt(c) + sqrt(d) and c + d = 75
theorem find_c_plus_d :
  ∃ (c d : ℕ), dist W Q = sqrt c + sqrt d ∧ c + d = 75 :=
sorry

end find_c_plus_d_l207_207421


namespace trapezoid_LM_sqrt2_l207_207798

theorem trapezoid_LM_sqrt2 (K L M N P Q : Type*)
  (KM : ℝ)
  (KN MQ LM MP : ℝ)
  (h_KM : KM = 1)
  (h_KN_MQ : KN = MQ)
  (h_LM_MP : LM = MP) 
  (h_KP_1 : KN = 1) 
  (h_MQ_1 : MQ = 1) :
  LM = Real.sqrt 2 :=
by
  sorry

end trapezoid_LM_sqrt2_l207_207798


namespace inkTransfer_equivalence_l207_207224

noncomputable def cupInk (m a : ℝ) (h : 0 < a ∧ a < m) : Prop :=
( let A_blue := (m * a) / (m + a),
      B_red := (m * a) / (m + a)
  in A_blue = B_red )

theorem inkTransfer_equivalence (m a : ℝ) (h : 0 < a ∧ a < m) : cupInk m a h :=
begin
  sorry
end

end inkTransfer_equivalence_l207_207224


namespace percentage_meat_fish_l207_207817

-- Definitions for the costs of the various items in Janet's grocery list.
def broccoli_cost : ℕ := 3 * 4
def orange_cost : ℕ := 3 * 75 / 100
def cabbage_cost : ℕ := 375 / 100
def bacon_cost : ℕ := 3
def chicken_cost : ℕ := 2 * 3
def tilapia_cost : ℕ := 5
def steak_cost : ℕ := 8
def apple_cost : ℕ := 5 * 150 / 100
def yogurt_cost : ℕ := 6
def milk_cost : ℕ := 350 / 100

-- Definition of the total cost of meat and fish.
def meat_fish_cost : ℕ := bacon_cost + chicken_cost + tilapia_cost + steak_cost

-- Definition of the total cost of all groceries purchased by Janet.
def total_cost : ℕ := broccoli_cost + orange_cost + cabbage_cost +
                                bacon_cost + chicken_cost + tilapia_cost + steak_cost + 
                                apple_cost + yogurt_cost + milk_cost

-- Definition to calculate the percentage of budget spent on meat and fish.
def percentage_spent : ℕ := ((meat_fish_cost.toFloat / total_cost.toFloat) * 100).round

-- Prove that the percentage of Janet's grocery budget spent on meat and fish is 39%.
theorem percentage_meat_fish : percentage_spent = 39 := by
  sorry

end percentage_meat_fish_l207_207817


namespace tan_alpha_plus_beta_eq_4_max_b_plus_c_norm_eq_4sqrt2_a_parallel_b_l207_207733

noncomputable def vec_a (α : ℝ) : ℝ × ℝ := (4 * Real.cos α, Real.sin α)
noncomputable def vec_b (β : ℝ) : ℝ × ℝ := (Real.sin β, 4 * Real.cos β)
noncomputable def vec_c (β : ℝ) : ℝ × ℝ := (Real.cos β, -4 * Real.sin β)

-- Problem 1
theorem tan_alpha_plus_beta_eq_4 (α β : ℝ) 
  (h : (vec_a α).1 * (vec_b β).1 + (vec_a α).2 * (vec_b β).2 = 
       2 * ((vec_a α).1 * (vec_c β).1 + (vec_a α).2 * (vec_c β).2)) : 
  Real.tan (α + β) = 4 := 
sorry

-- Problem 2
theorem max_b_plus_c_norm_eq_4sqrt2 (β : ℝ) :
  ∃ x, ∀ β, |vec_b β + vec_c β| ≤ x ∧ (x = 4 * Real.sqrt 2) :=
sorry

-- Problem 3
theorem a_parallel_b (α β : ℝ) (h : Real.tan α * Real.tan β = 16) :
  let a := vec_a α
  let b := vec_b β
  a.2 / b.2 = a.1 / b.1 :=
sorry

end tan_alpha_plus_beta_eq_4_max_b_plus_c_norm_eq_4sqrt2_a_parallel_b_l207_207733


namespace min_value_abs_sum_min_value_when_x_in_range_final_min_value_l207_207399

noncomputable def alpha_beta_roots : ℝ × ℝ :=
  let roots := Polynomial.roots ⟨1, -6, 5⟩ in
  have h : roots.length = 2 := by sorry
  ⟨roots[0], roots[1]⟩

def min_abs_sum (x : ℝ) (α β : ℝ) : ℝ := |x - α| + |x - β|

theorem min_value_abs_sum : ∀ x : ℝ, 
  let (α, β) := alpha_beta_roots in
  min_abs_sum x α β ≥ 4 :=
by 
  sorry

theorem min_value_when_x_in_range :  ∃ x : ℝ, 
  let (α, β) := alpha_beta_roots in 
  1 ≤ x ∧ x ≤ 5 ∧ min_abs_sum x α β = 4 :=
by 
  sorry

theorem final_min_value : 
  let (α, β) := alpha_beta_roots in
  ∃ m : ℝ, (∀ x : ℝ, min_abs_sum x α β ≥ m) ∧ (∃ x : ℝ, min_abs_sum x α β = m) ∧ m = 4 :=
by 
  existsi 4
  split
  { intro x
    exact min_value_abs_sum x }
  split
  { exact ⟨2, by
      let (α, β) := alpha_beta_roots
      exact min_value_when_x_in_range⟩ }
  { refl }

end min_value_abs_sum_min_value_when_x_in_range_final_min_value_l207_207399


namespace complex_number_a_l207_207298

theorem complex_number_a (a : ℂ) (ha : a - 2 * complex.I = 2 + complex.I) :
  (a = 1) ↔ ∃ b : ℂ, (a - 2 * complex.I) / (2 + complex.I) = complex.I * b :=
begin
  sorry
end

end complex_number_a_l207_207298


namespace endangered_species_count_l207_207116

section BirdsSanctuary

-- Define the given conditions
def pairs_per_species : ℕ := 7
def total_pairs : ℕ := 203

-- Define the result to be proved
theorem endangered_species_count : total_pairs / pairs_per_species = 29 := by
  sorry

end BirdsSanctuary

end endangered_species_count_l207_207116


namespace bisecting_sequence_length_l207_207276

-- Define the bisecting sequence of points on a line segment
variable (a : ℝ) -- Length of the segment AB
variable (n : ℕ) -- Number of bisections

-- Prove the length of AA_n in terms of a and n, given the bisecting properties
theorem bisecting_sequence_length : AA_n a n = (1 / 2)^n * a :=
sorry

end bisecting_sequence_length_l207_207276


namespace sum_first_9_terms_l207_207283

-- Define the arithmetic sequence {a_n}
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions of the problem
variables {a : ℕ → ℝ} (h_seq : arithmetic_sequence a) (h_sum : a 2 + a 6 = 8)

-- Sum of the first 9 terms
def S_9 := ∑ i in finset.range 9, a i

-- The theorem statement
theorem sum_first_9_terms : S_9 = 36 :=
by sorry

end sum_first_9_terms_l207_207283


namespace largest_prime_factor_of_12321_l207_207630

-- Definitions based on the given conditions
def n := 12321
def a := 111
def p₁ := 3
def p₂ := 37

-- Given conditions as hypotheses
theorem largest_prime_factor_of_12321 (h1 : n = a^2) (h2 : a = p₁ * p₂) (hp₁_prime : Prime p₁) (hp₂_prime : Prime p₂) :
  p₂ = 37 ∧ ∀ p, Prime p → p ∣ n → p ≤ 37 := 
by 
  sorry

end largest_prime_factor_of_12321_l207_207630


namespace exists_unique_triangle_l207_207222

-- Define the structure and properties
structure Triangle (A B C D : Type) :=
(len_AD BD CD : ℝ)
(angle_bisector : ∀ (A B C D : Type), True) -- Placeholder for the angle bisector property

-- State the theorem for the existence of the triangle given the properties
theorem exists_unique_triangle (A B C D : Type) (len_AD BD CD : ℝ) 
  (h_AD : len_AD > 0) (h_BD : len_BD > 0) (h_CD : len_CD > 0) :
  ∃ (Δ : Triangle A B C D), (Δ.len_AD = len_AD ∧ Δ.len_BD = len_BD ∧ Δ.len_CD = len_CD) ∧
    (Δ.angle_bisector A B C D) := 
sorry

end exists_unique_triangle_l207_207222


namespace point_in_second_quadrant_of_third_quadrant_l207_207746

-- Define the conditions
def is_third_quadrant (α : ℝ) : Prop :=
  π < α ∧ α < 3 * π / 2

def sin_negative (α : ℝ) : Prop :=
  sin α < 0

def tan_positive (α : ℝ) : Prop :=
  tan α > 0

-- Define the point P and the quadrant it lies in
def point_P (α : ℝ) : Prop :=
  ∃ (x y : ℝ), x = sin α ∧ y = tan α

def second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

-- Main theorem to prove
theorem point_in_second_quadrant_of_third_quadrant (α : ℝ) (h1 : is_third_quadrant α) (h2 : sin_negative α) (h3 : tan_positive α) :
  ∃ x y, point_P α ∧ second_quadrant x y :=
sorry

end point_in_second_quadrant_of_third_quadrant_l207_207746


namespace trader_sold_23_bags_l207_207183

theorem trader_sold_23_bags
    (initial_stock : ℕ) (restocked : ℕ) (final_stock : ℕ) (x : ℕ)
    (h_initial : initial_stock = 55)
    (h_restocked : restocked = 132)
    (h_final : final_stock = 164)
    (h_equation : initial_stock - x + restocked = final_stock) :
    x = 23 :=
by
    -- Here will be the proof of the theorem
    sorry

end trader_sold_23_bags_l207_207183


namespace coordinates_after_movement_l207_207085

theorem coordinates_after_movement :
  let A_initial := (1, 0)
  let move_right := 2
  let move_down := 3
  let A_final := (fst A_initial + move_right, snd A_initial - move_down)
  A_final = (3, -3) :=
by
  let A_initial := (1, 0)
  let move_right := 2
  let move_down := 3
  let A_final := (fst A_initial + move_right, snd A_initial - move_down)
  show A_final = (3, -3) from
    sorry

end coordinates_after_movement_l207_207085


namespace harmonic_mean_pairs_count_l207_207668

theorem harmonic_mean_pairs_count :
  ∃ (s : Finset (ℕ × ℕ)), (∀ p ∈ s, p.1 < p.2 ∧ 2 * p.1 * p.2 = 4^15 * (p.1 + p.2)) ∧ s.card = 29 :=
sorry

end harmonic_mean_pairs_count_l207_207668


namespace move_line_upwards_l207_207087

theorem move_line_upwards (x y : ℝ) :
  (y = -x + 1) → (y + 5 = -x + 6) :=
by
  intro h
  sorry

end move_line_upwards_l207_207087


namespace polar_equation_solution_l207_207893

theorem polar_equation_solution (x y : ℝ) : 
  (∃ (θ : ℝ), θ ∈ set.Ico 0 (2 * Real.pi) ∧ ∃ (ρ : ℝ), ρ ≥ 0 ∧ 
  (ρ * Real.cos θ = x) ∧ (ρ * Real.sin θ = y) ∧ 
  (Real.atan2 (√3 / 3) 1 = θ)) ↔ θ = π / 6 ∨ θ = 7 * π / 6 := 
by sorry

end polar_equation_solution_l207_207893


namespace sum_of_distinct_prime_factors_of_252_l207_207927

theorem sum_of_distinct_prime_factors_of_252 : (∑ p in {2, 3, 7}, p) = 12 := by
  sorry

end sum_of_distinct_prime_factors_of_252_l207_207927


namespace projection_of_e_in_direction_of_a_l207_207313

variable (a : ℝ × ℝ) (e : ℝ × ℝ)
variable (ha : a = (1, real.sqrt 3))
variable (he_unit : e.1^2 + e.2^2 = 1)
variable (proj_a_e : (1 * e.1 + real.sqrt 3 * e.2) / real.sqrt (1^2 + (real.sqrt 3)^2) = -real.sqrt 2)

theorem projection_of_e_in_direction_of_a :
  ((e.1 * 1 + e.2 * real.sqrt 3) / real.sqrt (1^2 + (real.sqrt 3)^2)) = -real.sqrt 2 / 2 :=
by
  sorry

end projection_of_e_in_direction_of_a_l207_207313


namespace range_of_a_tangent_intersection_ineq_l207_207757

-- Definitions based on the conditions
variables {a x x1 x2 x3 y3 : ℝ}

-- Function definition and its properties
def f (x : ℝ) : ℝ := a * log x - (1 / 2) * x^2 + a + 1 / 2
def f_has_two_zeros (x1 x2 : ℝ) (h : x1 < x2) : Prop := f x1 = 0 ∧ f x2 = 0

-- Problem (1): Proving the range of values for a
theorem range_of_a (hx: f_has_two_zeros x1 x2 (by linarith)) : 0 < a :=
sorry

-- Problem (2): Proving the inequality involving x1, x2, x3
theorem tangent_intersection_ineq (hx: f_has_two_zeros x1 x2 (by linarith)) (hy: y3 = 0) :
  2 * x3 < x1 + x2 :=
sorry

end range_of_a_tangent_intersection_ineq_l207_207757


namespace selling_price_is_80000_l207_207861

-- Given the conditions of the problem
def purchasePrice : ℕ := 45000
def repairCosts : ℕ := 12000
def profitPercent : ℚ := 40.35 / 100

-- Total cost calculation
def totalCost := purchasePrice + repairCosts

-- Profit calculation
def profit := profitPercent * totalCost

-- Selling price calculation
def sellingPrice := totalCost + profit

-- Statement of the proof problem
theorem selling_price_is_80000 : round sellingPrice = 80000 := by
  sorry

end selling_price_is_80000_l207_207861


namespace population_approx_16000_in_2060_l207_207240

def population (n : ℕ) : ℕ := 500 * (4 ^ (n / 30))

theorem population_approx_16000_in_2060 :
  population 60 ≈ 16000 :=
sorry

end population_approx_16000_in_2060_l207_207240


namespace percent_increase_from_first_to_fourth_l207_207567

-- Define the side lengths of the triangles
def side_length_triangle1 : ℝ := 3
def ratio : ℝ := 1.6

-- Calculate the side lengths of subsequent triangles
def side_length_triangle2 : ℝ := side_length_triangle1 * ratio
def side_length_triangle3 : ℝ := side_length_triangle2 * ratio
def side_length_triangle4 : ℝ := side_length_triangle3 * ratio

-- Calculate the perimeters of the triangles
def perimeter_triangle1 : ℝ := 3 * side_length_triangle1
def perimeter_triangle4 : ℝ := 3 * side_length_triangle4

-- Calculate the percent increase in the perimeter
def percent_increase : ℝ := ((perimeter_triangle4 - perimeter_triangle1) / perimeter_triangle1) * 100

-- The statement to prove
theorem percent_increase_from_first_to_fourth : percent_increase = 309.6 := by
  sorry

end percent_increase_from_first_to_fourth_l207_207567


namespace minimum_value_is_six_l207_207397

noncomputable def minimum_value_expression (x y z : ℝ) : ℝ :=
  (x^2 + y^2) / (x + y) + (x^2 + z^2) / (x + z) + (y^2 + z^2) / (y + z)

theorem minimum_value_is_six
  (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x + y + z = 9) (h2 : y = 2 * x) :
  minimum_value_expression x y z = 6 :=
by
  sorry

end minimum_value_is_six_l207_207397


namespace sum_lent_correct_l207_207528

noncomputable section

-- Define the principal amount (sum lent)
def P : ℝ := 4464.29

-- Define the interest rate per annum
def R : ℝ := 12.0

-- Define the time period in years
def T : ℝ := 12.0

-- Define the interest after 12 years (using the initial conditions and results)
def I : ℝ := 1.44 * P

-- Define the interest given as "2500 less than double the sum lent" condition
def I_condition : ℝ := 2 * P - 2500

-- Theorem stating the sum lent is the given value P
theorem sum_lent_correct : P = 4464.29 :=
by
  -- Placeholder for the proof
  sorry

end sum_lent_correct_l207_207528


namespace expected_value_of_sum_l207_207335

variable {s : Finset ℕ} (h_s : s = {1, 2, 3, 4, 5, 6}) (n : ℕ) [IncRange : 1 ≤ n ∧ n ≤ 6] 

theorem expected_value_of_sum (hmarbles : s.card = 6) (hk : choose 3 6 = 20) : 
  (∑ x in (s.powerset.filter (λ xs, xs.card = 3)), xs.sum) / 20 = 10.5 :=
by
  sorry

end expected_value_of_sum_l207_207335


namespace neither_sufficient_nor_necessary_l207_207830

variable (a b : ℝ)

theorem neither_sufficient_nor_necessary (a b : ℝ) : 
  (¬(a > b → a^2 > b^2)) ∧ (¬(a^2 > b^2 → a > b)) := 
begin
  sorry
end

end neither_sufficient_nor_necessary_l207_207830


namespace toothpicks_needed_for_structure_l207_207956

def total_small_triangles(base_triangles: ℕ) : ℕ :=
  (base_triangles * (base_triangles + 1)) / 2

def total_toothpicks_without_boundary(triangles: ℕ) : ℕ :=
  3 * triangles

def shared_toothpicks(toothpicks: ℕ) : ℕ :=
  toothpicks / 2

def boundary_toothpicks(base_triangles: ℕ) : ℕ :=
  3 * base_triangles + 3

def total_toothpicks(base_triangles: ℕ) : ℕ :=
  let triangles := total_small_triangles(base_triangles)
  let toothpicks := total_toothpicks_without_boundary(triangles)
  let shared := shared_toothpicks(toothpicks)
  let boundary := boundary_toothpicks(base_triangles)
  shared + boundary

theorem toothpicks_needed_for_structure (base_triangles: ℕ) (h : base_triangles = 101) : 
  total_toothpicks base_triangles = 8032 := by
  sorry

end toothpicks_needed_for_structure_l207_207956


namespace expression_I_expression_II_l207_207213

theorem expression_I : (1 / 4 : ℝ) ^ (-1 / 2 : ℝ) + (8 : ℝ) ^ 2 ^ (1 / 3 : ℝ) - 625 ^ (1 / 4 : ℝ) = 1 := 
by {
  sorry,
}

theorem expression_II : 
  (Real.logBase 2 125 + Real.logBase 4 25 + Real.logBase 8 5) * 
  (Real.logBase 5 2 + Real.logBase 25 4 + Real.logBase 125 8) = 13 :=
by {
  sorry,
}

end expression_I_expression_II_l207_207213


namespace moving_circle_passes_through_focus_l207_207546

open Real

def parabola (x y : ℝ) : Prop := y^2 = 8 * x
def directrix (x : ℝ) : Prop := x + 2 = 0
def focus : ℝ × ℝ := (2, 0)

theorem moving_circle_passes_through_focus :
  (∃ c : ℝ × ℝ, (parabola c.1 c.2) ∧ (∀ r : ℝ, dist (c.1, c.2) (-2, r) = dist (c.1, c.2) focus)) →
    (2, 0) ∈ 
sorry

end moving_circle_passes_through_focus_l207_207546


namespace graph_symmetry_y_axis_l207_207346

theorem graph_symmetry_y_axis (a b : ℝ) (ha : a ≠ 1) (hb : b ≠ 1)
  (hlog : log a + log b = 0) :
  ∀ x : ℝ, f x = a * x ∧ g x = b * x → f(-x) = g x :=
by
  sorry

end graph_symmetry_y_axis_l207_207346


namespace value_of_a_l207_207881

theorem value_of_a (x : ℝ) (h1 : x^2 - 3*|x| + 2 = 0 → x = 1 ∨ x = -1 ∨ x = 2 ∨ x = -2) : 
  (∀ x, x^4 - a*x^2 + 4 = 0 → x = 1 ∨ x = -1 ∨ x = 2 ∨ x = -2) → a = 5 :=
begin
  sorry
end

end value_of_a_l207_207881


namespace number_of_adults_l207_207950

theorem number_of_adults (A C S : ℕ) (h1 : C = A - 35) (h2 : S = 2 * C) (h3 : A + C + S = 127) : A = 58 :=
by
  sorry

end number_of_adults_l207_207950


namespace count_sixth_powers_lt_200_l207_207325

theorem count_sixth_powers_lt_200 : 
  {n : ℕ | n > 0 ∧ n < 200 ∧ (∃ k : ℕ, n = k^6)}.to_finset.card = 2 := 
by sorry

end count_sixth_powers_lt_200_l207_207325


namespace impossible_arithmetic_mean_l207_207848

-- Define the infinite tape and a function to represent the placement of integers
def infinite_tape : Type := ℤ → ℤ

-- Arithmetic mean of two integers a and b
def arithmetic_mean (a b : ℤ) : ℤ := (a + b) / 2

-- Proposition that states there does not exist an infinite tape
-- arrangement such that the arithmetic mean does not lie between.
theorem impossible_arithmetic_mean :
  ¬ ∃ (tape_position : infinite_tape), ∀ (a b : ℤ),
    let m := arithmetic_mean (tape_position a) (tape_position b) in
    (tape_position a < m ∨ m < tape_position b) := 
sorry

end impossible_arithmetic_mean_l207_207848


namespace factorize_expression_l207_207621

variable (a x y : ℝ)

theorem factorize_expression : a^2 * (x - y) + 9 * (y - x) = (x - y) * (a + 3) * (a - 3) :=
by
  sorry

end factorize_expression_l207_207621


namespace no_possible_arrangement_of_balloons_l207_207109

/-- 
  There are 10 balloons hanging in a row: blue and green. This statement proves that it is impossible 
  to arrange 10 balloons such that between every two blue balloons, there is an even number of 
  balloons and between every two green balloons, there is an odd number of balloons.
--/

theorem no_possible_arrangement_of_balloons :
  ¬ (∃ (color : Fin 10 → Bool), 
    (∀ i j, i < j ∧ color i = color j ∧ color i = tt → (j - i - 1) % 2 = 0) ∧
    (∀ i j, i < j ∧ color i = color j ∧ color i = ff → (j - i - 1) % 2 = 1)) :=
by
  sorry

end no_possible_arrangement_of_balloons_l207_207109


namespace original_number_is_repeating_decimal_l207_207948

theorem original_number_is_repeating_decimal :
  ∃ N : ℚ, (N * 10 ^ 28) % 10^30 = 15 ∧ N * 5 = 0.7894736842105263 ∧ 
  (N = 3 / 19) :=
sorry

end original_number_is_repeating_decimal_l207_207948


namespace smallest_four_digit_palindromic_prime_is_1101_l207_207494

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_digits 10
  s = s.reverse

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem smallest_four_digit_palindromic_prime_is_1101 :
  ∀ n : ℕ, is_four_digit_number n → is_palindrome n → is_prime n → n ≥ 1101 := by
  sorry

end smallest_four_digit_palindromic_prime_is_1101_l207_207494


namespace maximize_risk_adjusted_returns_l207_207568

/-- Definition of the Sharpe ratio -/
def sharpe_ratio (expected_return risk_free_rate std_deviation : ℝ) : ℝ :=
  (expected_return - risk_free_rate) / std_deviation

/-- Given data for the stocks and market -/
def stock_A_expected_return := 0.10
def stock_A_std_deviation := 0.15

def stock_B_expected_return := 0.08
def stock_B_std_deviation := 0.10

def stock_C_expected_return := 0.06
def stock_C_std_deviation := 0.18

def risk_free_rate := 0.02

/-- Assertion that Stock B has the highest Sharpe ratio -/
theorem maximize_risk_adjusted_returns : 
  sharpe_ratio stock_B_expected_return risk_free_rate stock_B_std_deviation = 
  max (sharpe_ratio stock_A_expected_return risk_free_rate stock_A_std_deviation) 
      (max (sharpe_ratio stock_B_expected_return risk_free_rate stock_B_std_deviation) 
           (sharpe_ratio stock_C_expected_return risk_free_rate stock_C_std_deviation)) :=
by sorry

end maximize_risk_adjusted_returns_l207_207568


namespace find_inverse_square_sum_and_major_axis_range_l207_207455

variables {a b x y : ℝ}

noncomputable def ellipse_line_intersect_perpendicular : Prop :=
  (a > b ∧ b > 0) ∧
  (let P := (x, y),
       Q := (2 - x, 2 - y),
       OP_perp_OQ := (x * (2 - x) + y * (2 - y) = 0) in
  OP_perp_OQ ∧
  (∃ P Q, (P = (x, y ∧ Q = (2 - x, 2 - y) ∧
    (x * (2 - x) + y * (2 - y) = 0) ∧
    (x^2 / a^2 + y^2 / b^2 = 1) ∧
    (2^2 / a^2 + (2-2)^2 / b^2 = 1))))

theorem find_inverse_square_sum_and_major_axis_range :
  ellipse_line_intersect_perpendicular →
  ( ∃ (a b : ℝ), a > b ∧ b > 0 ∧ 
    (\frac{1}{a^2} + \frac{1}{b^2} = \frac{1}{2}) ∧ 
    (\frac{\sqrt{3}}{3} ≤ √(1 - (b / a)^2) ∧ √(1 - (b / a)^2) ≤ \frac{\sqrt{6}}{3}) ∧ 
    (2 * a) ∈ set.Icc (2 * real.sqrt 5) (4 * real.sqrt 2)) :=
sorry

end find_inverse_square_sum_and_major_axis_range_l207_207455


namespace correct_judgment_l207_207272

-- Definitions of planes and lines
variable (α β : Type) -- Assuming types for planes
variable (a b : Type) -- Assuming types for lines

-- Conditions for Propositions
variable [IsPlane α] [IsPlane β]
variable [IsLine a] [IsLine b]

-- Propositions
def proposition1 := (b ⊂ α) ∧ (a ⊄ α) → (a ∥ b) ↔ (a ∥ α)
def proposition2 := (a ⊂ α) ∧ (b ⊂ α) → (α ∥ β) ↔ ((α ∥ β) ∧ (b ∥ β))

-- Proof of correctness for the propositions
theorem correct_judgment : proposition1 α β a b = true ∧ proposition2 α β a b = false :=
by
  -- Placeholder for the proof
  sorry

end correct_judgment_l207_207272


namespace num_dogs_correct_l207_207482

-- Definitions based on conditions
def total_animals : ℕ := 17
def number_of_cats : ℕ := 8

-- Definition based on required proof
def number_of_dogs : ℕ := total_animals - number_of_cats

-- Proof statement
theorem num_dogs_correct : number_of_dogs = 9 :=
by
  sorry

end num_dogs_correct_l207_207482


namespace hexagon_internal_angles_equal_l207_207688

theorem hexagon_internal_angles_equal
  (hexagon : ConvexHexagon)
  (h1 : ∀ (side1 side2 : Segment), distance (midpoint side1) (midpoint side2) = (sqrt 3 / 2) * (length side1 + length side2)) :
  ∀ (angle : InternalAngle hexagon), angle = π/3 :=
by
  sorry

end hexagon_internal_angles_equal_l207_207688


namespace intersection_distance_product_l207_207369

theorem intersection_distance_product (θ t : ℝ) :
  (∀ (x y : ℝ), ((4 * cos θ)^2 + (4 * sin θ)^2 = 16) ∧ ((x = 1 + (sqrt 3 / 2) * t) → (y = 2 + (1 / 2) * t))) →
  (∃ A B : (ℝ × ℝ), line_intersects_circle (1, 2) (π / 6) (4 * cos θ, 4 * sin θ) A B ∧ |dist (1, 2) A * dist (1, 2) B| = 11) :=
sorry

end intersection_distance_product_l207_207369


namespace num_zeros_sin_pi_cos_in_interval_l207_207095

-- Define the function f
def f (x : ℝ) : ℝ := Real.sin (π * Real.cos x)

-- State the theorem to prove the number of zeros of f(x) in the interval [0, 2π]
theorem num_zeros_sin_pi_cos_in_interval :
  ∃ (n : ℕ), n = 4 ∧
  (∀ y ∈ Icc 0 (2 * π), f y = 0 → (y = 0 ∨ y = π/2 ∨ y = π ∨ y = 3 * π/2)) :=
sorry

end num_zeros_sin_pi_cos_in_interval_l207_207095


namespace incenter_circumcircle_equality_l207_207571

open Geometry

variable {α : Type} [Field α] [Metric α]

/-- Prove that for a triangle's incenter's intersections with its sides and circumcircle,
    the products of the segments are equal. -/
theorem incenter_circumcircle_equality
  (A B C I D E F M N P : Point α)
  (circumcircle : Circle α)
  (h1 : circumcircle.is_circumcircle_of (triangle A B C))
  (h2 : I.is_incenter_of (triangle A B C))
  (h3 : line_through A I ∩ B C = D)
  (h4 : line_through B I ∩ A C = E)
  (h5 : line_through C I ∩ A B = F)
  (h6 : line_through A I ∩ circumcircle = M)
  (h7 : line_through B I ∩ circumcircle = N)
  (h8 : line_through C I ∩ circumcircle = P) :
  (distance A M * distance I D = distance B N * distance I E) ∧
  (distance B N * distance I E = distance C P * distance I F) :=
by
  sorry

end incenter_circumcircle_equality_l207_207571


namespace james_bike_ride_l207_207816

variable {D P : ℝ}

theorem james_bike_ride :
  (∃ D P, 3 * D + (18 + 18 * 0.25) = 55.5 ∧ (18 = D * (1 + P / 100))) → P = 20 := by
  sorry

end james_bike_ride_l207_207816


namespace correct_statement_l207_207931

def degree (term : String) : ℕ :=
  if term = "1/2πx^2" then 2
  else if term = "-4x^2y" then 3
  else 0

def coefficient (term : String) : ℤ :=
  if term = "-4x^2y" then -4
  else if term = "3(x+y)" then 3
  else 0

def is_monomial (term : String) : Bool :=
  if term = "8" then true
  else false

theorem correct_statement : 
  (degree "1/2πx^2" ≠ 3) ∧ 
  (coefficient "-4x^2y" ≠ 4) ∧ 
  (is_monomial "8" = true) ∧ 
  (coefficient "3(x+y)" ≠ 3) := 
by
  sorry

end correct_statement_l207_207931


namespace fresh_fruit_sold_l207_207072

variable (total_fruit frozen_fruit : ℕ)

theorem fresh_fruit_sold (h1 : total_fruit = 9792) (h2 : frozen_fruit = 3513) : 
  total_fruit - frozen_fruit = 6279 :=
by sorry

end fresh_fruit_sold_l207_207072


namespace problem_statement_l207_207291

-- Define the condition as a predicate
def condition (α : ℝ) : Prop :=
  sin α * (1 / cos α) * sqrt((1 / (sin α)^2) - 1) = -1

-- Define the proposition stating that alpha is in Quadrant II or IV
def in_II_or_IV (α : ℝ) : Prop :=
  (π / 2 < α ∧ α < π) ∨ (3 * π / 2 < α ∧ α < 2 * π)

-- The theorem statement asserting that the condition implies the proposition
theorem problem_statement (α : ℝ) (h : condition α) : in_II_or_IV α :=
by
  -- sorry to skip the proof
  sorry

end problem_statement_l207_207291


namespace range_of_f_l207_207252

def f (x : ℝ) : ℝ := sin x ^ 4 - sin x * cos x + cos x ^ 4

theorem range_of_f : ∀ y : ℝ, y ∈ set.range f ↔ 0 ≤ y ∧ y ≤ 9 / 8 :=
by
  sorry

end range_of_f_l207_207252


namespace continuous_g_l207_207825

noncomputable def f (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3)

noncomputable def g (x : ℝ) : ℝ := min (f x) (3 * x^2 - 10 * x + 8)

theorem continuous_g : ¬ ∃ x : ℝ, (is_discontinuity g x) :=
sorry

end continuous_g_l207_207825


namespace solve_equation_l207_207073

theorem solve_equation :
  ∃ x : ℝ, (x + 2) / 4 - (2 * x - 3) / 6 = 2 ∧ x = -12 :=
by
  sorry

end solve_equation_l207_207073


namespace area_of_square_plot_l207_207126

theorem area_of_square_plot (s : ℕ) (price_per_foot total_cost: ℕ)
  (h_price : price_per_foot = 58)
  (h_total_cost : total_cost = 3944) :
  (s * s = 289) :=
by
  sorry

end area_of_square_plot_l207_207126


namespace smallest_possible_value_of_sum_of_cubes_l207_207076

noncomputable theory

open Complex

theorem smallest_possible_value_of_sum_of_cubes 
  (a b : ℂ) (h1 : abs (a + b) = 2) (h2 : abs (a^2 + b^2) = 8) : 
  abs (a^3 + b^3) = 20 :=
sorry

end smallest_possible_value_of_sum_of_cubes_l207_207076


namespace area_of_rectangular_field_l207_207418

theorem area_of_rectangular_field (W D : ℝ) (hW : W = 15) (hD : D = 17) :
  ∃ L : ℝ, (W * L = 120) ∧ D^2 = L^2 + W^2 :=
by 
  use 8
  sorry

end area_of_rectangular_field_l207_207418


namespace find_abcde_l207_207261

noncomputable def find_five_digit_number (a b c d e : ℕ) : ℕ :=
  10000 * a + 1000 * b + 100 * c + 10 * d + e

theorem find_abcde
  (a b c d e : ℕ)
  (h1 : 0 ≤ a ∧ a ≤ 9)
  (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : 0 ≤ c ∧ c ≤ 9)
  (h4 : 0 ≤ d ∧ d ≤ 9)
  (h5 : 0 ≤ e ∧ e ≤ 9)
  (h6 : a ≠ 0)
  (h7 : (10 * a + b + 10 * b + c) * (10 * b + c + 10 * c + d) * (10 * c + d + 10 * d + e) = 157605) :
  find_five_digit_number a b c d e = 12345 ∨ find_five_digit_number a b c d e = 21436 :=
sorry

end find_abcde_l207_207261


namespace triangle_divisible_into_n_similar_triangles_l207_207188

theorem triangle_divisible_into_n_similar_triangles 
    (triangle_divisible_into_three_similar : ∀ Δ : Triangle, ∃ Δ1 Δ2 Δ3 : Triangle, similar Δ Δ1 ∧ similar Δ Δ2 ∧ similar Δ Δ3) 
    (n : ℕ) :
    ∃ Δs : Fin n → Triangle, (∀ i : Fin n, similar Δs i Δ) :=
sorry

end triangle_divisible_into_n_similar_triangles_l207_207188


namespace Cylinder_views_are_A_l207_207088

open Classical

noncomputable def Cylinder.Views : Type :=
  { front : String, side : String, top : String }

noncomputable def givenCylinder : Cylinder.Views :=
  { front := "Rectangle", side := "Rectangle", top := "Circle" }

theorem Cylinder_views_are_A :
  givenCylinder = { front := "Rectangle", side := "Rectangle", top := "Circle" } :=
by
  sorry

end Cylinder_views_are_A_l207_207088


namespace trucks_and_goods_l207_207167

variable (x : ℕ) -- Number of trucks
variable (goods : ℕ) -- Total tons of goods

-- Conditions
def condition1 : Prop := goods = 3 * x + 5
def condition2 : Prop := goods = 4 * (x - 5)

theorem trucks_and_goods (h1 : condition1 x goods) (h2 : condition2 x goods) : x = 25 ∧ goods = 80 :=
by
  sorry

end trucks_and_goods_l207_207167


namespace surface_area_of_sphere_l207_207690

/-- Given a right prism with all vertices on a sphere, a height of 4, and a volume of 64,
    the surface area of this sphere is 48π -/
theorem surface_area_of_sphere (h : ℝ) (V : ℝ) (S : ℝ) :
  h = 4 → V = 64 → S = 48 * Real.pi := by
  sorry

end surface_area_of_sphere_l207_207690


namespace roots_of_equation_l207_207625

theorem roots_of_equation (a b c d x : ℝ) (h1 : a + d = 2015) (h2 : b + c = 2015) (h3 : a ≠ c) : 
  (x-a)*(x-b) = (x-c)*(x-d) → x = 1007.5 :=
begin
  sorry
end

end roots_of_equation_l207_207625


namespace factorize_x2_add_2x_sub_3_l207_207219

theorem factorize_x2_add_2x_sub_3 :
  (x^2 + 2 * x - 3) = (x + 3) * (x - 1) :=
by
  sorry

end factorize_x2_add_2x_sub_3_l207_207219


namespace quadruple_never_repeats_original_l207_207689

noncomputable def quadruple_transformation (a b c d : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  (a * b, b * c, c * d, d * a)

theorem quadruple_never_repeats_original (a b c d : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) :
  ∀ n m : ℕ, (n ≠ m) → quadruple_transformation^[n] (a, b, c, d) ≠ quadruple_transformation^[m] (a, b, c, d) :=
  sorry

end quadruple_never_repeats_original_l207_207689


namespace one_pair_parallel_and_equal_in_parallelogram_one_pair_parallel_and_equal_in_quadrilateral_diagonals_perpendicular_in_rhombus_diagonals_perpendicular_in_quadrilateral_count_correct_statements_problem_correct_statements_count_l207_207197

def is_parallelogram (Q : Type) [qu : quadrilateral Q] : Prop :=
  ∀ one_pair_parallel one_pair_equal, true -- Simplified for illustration
def is_rhombus (R : Type) [qu : quadrilateral R] : Prop :=
  ∀ diagonals_perpendicular, true -- Simplified for illustration

constant Q1 : Type
constant Q2 : Type
constant Q3 : Type
constant Q4 : Type

instance : quadrilateral Q1 := sorry
instance : quadrilateral Q2 := sorry
instance : quadrilateral Q3 := sorry
instance : quadrilateral Q4 := sorry

theorem one_pair_parallel_and_equal_in_parallelogram (q : Q1) [is_parallelogram Q1] : 
  ∃ one_pair_parallel one_pair_equal, true :=
sorry 

theorem one_pair_parallel_and_equal_in_quadrilateral (q : Q2) :
  ¬ ( ∃ one_pair_parallel one_pair_equal, is_parallelogram Q2) :=
sorry

theorem diagonals_perpendicular_in_rhombus (r : Q3) [is_rhombus Q3] :
  ∃ diagonals_perpendicular, true :=
sorry

theorem diagonals_perpendicular_in_quadrilateral (r : Q4) : 
  ¬ (∃ diagonals_perpendicular, is_rhombus Q4) :=
sorry 

theorem count_correct_statements : nat :=
  nat.succ (nat.succ 0) -- 2

theorem problem_correct_statements_count :
  (∀ (q1 : Q1), is_parallelogram Q1 → true) ∧ 
  (¬ (∀ (q2 : Q2), true → is_parallelogram Q2)) ∧ 
  (∀ (r : Q3), is_rhombus Q3 → true) ∧ 
  (¬ (∀ (r : Q4), true → is_rhombus Q4)) →
  count_correct_statements = 2 :=
sorry

end one_pair_parallel_and_equal_in_parallelogram_one_pair_parallel_and_equal_in_quadrilateral_diagonals_perpendicular_in_rhombus_diagonals_perpendicular_in_quadrilateral_count_correct_statements_problem_correct_statements_count_l207_207197


namespace simplify_expression_l207_207865

theorem simplify_expression : (1 / (1 + Real.sqrt 3) * 1 / (1 + Real.sqrt 3)) = 1 - Real.sqrt 3 / 2 :=
by
  sorry

end simplify_expression_l207_207865


namespace min_lines_geq_l207_207826

variable {f : ℕ → Type} -- Define a function f that takes a natural number as input and returns a type

-- Define M to be a function that takes f and returns the minimum number of lines in a program
-- that computes f, where each line computes either a disjunction or a conjunction
constant M : (ℕ → Type) → ℕ

-- Main theorem statement from the problem
theorem min_lines_geq : ∀ n, n ≥ 4 → M (f n) ≥ M (f (n - 2)) + 3 := 
by 
  intro n h
  sorry

end min_lines_geq_l207_207826


namespace division_implies_equality_l207_207564

theorem division_implies_equality (x y b : ℝ) (hb : b ≠ 0) :
  (x / b = y / b) → (x = y) :=
begin
  intro h,
  apply eq_of_mul_eq_mul_right hb,
  rw [←mul_assoc, h, mul_assoc],
end

end division_implies_equality_l207_207564


namespace sequence_divisibility_l207_207390

-- Define the sequence using the given conditions
def sequence (k : ℕ) (h : Even k) (n : ℕ) : ℕ :=
  match n with
  | 0       => 1
  | n + 1   => k^(sequence k h n) + 1

-- Define the statement to be proved
theorem sequence_divisibility (k : ℕ) (h : Even k) (n : ℕ) (hn : n ≥ 2) :
  (sequence k h n)^2 ∣ (sequence k h (n-1) * sequence k h (n+1)) :=
sorry

end sequence_divisibility_l207_207390


namespace children_playing_tennis_l207_207359

theorem children_playing_tennis
  (Total : ℕ) (S : ℕ) (N : ℕ) (B : ℕ) (T : ℕ) 
  (hTotal : Total = 38) (hS : S = 21) (hN : N = 10) (hB : B = 12) :
  T = 38 - 21 + 12 - 10 :=
by
  sorry

end children_playing_tennis_l207_207359


namespace dice_probability_l207_207911

open ProbabilityTheory

noncomputable def prob_two_ones_and_one_six : ℝ :=
  let n := 12
  let k := 2
  let p := 1 / 6
  let q := 5 / 6
  let combinations := Nat.choose n k
  let prob_two_ones := combinations * p^k * q^(n - k)
  let prob_at_least_one_six := 1 - q^10
  prob_two_ones * prob_at_least_one_six

theorem dice_probability :
  prob_two_ones_and_one_six = 0.049 :=
sorry

end dice_probability_l207_207911


namespace range_of_f_l207_207090

def f (x : ℝ) : ℝ := x^2 - 4*x + 5

theorem range_of_f : set.range (λ x, f x) = set.Icc 1 10 := 
by 
    sorry 

end range_of_f_l207_207090


namespace count_squares_below_line_l207_207887

theorem count_squares_below_line :
  let line_eq := λ x y : ℕ, 8 * x + 200 * y = 1600
  let first_quadrant := λ x y : ℕ, x ≥ 0 ∧ y ≥ 0
  ∃ count : ℕ, 
    count = (sorry : ℕ) → -- this is where the number calculation would be relevant
    count = 697 := 
by 
  sorry

end count_squares_below_line_l207_207887


namespace min_blocks_to_remove_l207_207513

theorem min_blocks_to_remove (n : ℕ) (h : n = 59) : 
  ∃ (k : ℕ), k = 32 ∧ (∃ m, n = m^3 + k ∧ m^3 ≤ n) :=
by {
  sorry
}

end min_blocks_to_remove_l207_207513


namespace sum_of_powers_of_i_l207_207991

theorem sum_of_powers_of_i : 
  (∑ k in finset.range 4004, (k + 1) * (complex.I^ (k + 1))) = -2002 + 2002 * complex.I :=
by
  sorry

end sum_of_powers_of_i_l207_207991


namespace largest_prime_factor_of_12321_l207_207631

-- Definitions based on the given conditions
def n := 12321
def a := 111
def p₁ := 3
def p₂ := 37

-- Given conditions as hypotheses
theorem largest_prime_factor_of_12321 (h1 : n = a^2) (h2 : a = p₁ * p₂) (hp₁_prime : Prime p₁) (hp₂_prime : Prime p₂) :
  p₂ = 37 ∧ ∀ p, Prime p → p ∣ n → p ≤ 37 := 
by 
  sorry

end largest_prime_factor_of_12321_l207_207631


namespace greatest_k_find_m_l207_207391

-- Define the initial conditions and function
def flip_cards (n : ℕ) (initial_config : Fin n → Bool) : ℕ → ℕ
| 0 => initial_config.1.count_eq tt
| (t + 1) =>
    let flipped := List.map (λ (b : Bool), bnot b) (initial_config.toList (take t + 1))
    in flipped.count_eq tt + (initial_config.toList (drop t + 1)).count_eq tt

-- Part (a): Prove the greatest integer k for at least k distinct values
theorem greatest_k (n : ℕ) (initial_config : Fin n → Bool) : 2 ≤ (List.range (n + 1)).countp (λ t, (flip_cards n initial_config t) ∈ (List.range (n + 1))) :=
sorry

-- Part (b): Find all positive integers m such that for each initial card configuration, there exists an index r such that s_r = m
theorem find_m (n : ℕ) (initial_config : Fin n → Bool) (m : ℕ) :
  (m = n / 2) ∨ (m = (n + 1) / 2) →
  ∃ r < n + 1, flip_cards n initial_config r = m :=
sorry

end greatest_k_find_m_l207_207391


namespace percentage_increase_of_machine_b_l207_207049

theorem percentage_increase_of_machine_b 
    (total_sprockets : ℕ)
    (rate_machine_a : ℕ)
    (time_difference : ℕ)
    (rate_machine_a_approx : rate_machine_a = 4)
    (total_sprockets_eq : total_sprockets = 440)
    (time_difference_eq : time_difference = 10) :
    let time_machine_a := total_sprockets / rate_machine_a in
    let time_machine_b := time_machine_a - time_difference in
    let rate_machine_b := total_sprockets / time_machine_b in
    (rate_machine_b - rate_machine_a) / rate_machine_a * 100 = 10 :=
by
  sorry

end percentage_increase_of_machine_b_l207_207049


namespace range_of_m_l207_207712

-- Define the conditions based on the problem statement
def equation (x m : ℝ) : Prop := (2 * x + m) = (x - 1)

-- The goal is to prove that if there exists a positive solution x to the equation, then m < -1
theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, equation x m ∧ x > 0) → m < -1 :=
by
  sorry

end range_of_m_l207_207712


namespace trapezoid_LM_sqrt2_l207_207797

theorem trapezoid_LM_sqrt2 (K L M N P Q : Point) : 
  ∀ (h_trapezoid : is_trapezoid K L M N) 
     (diag_eq_height : distance K M = 1 ∧ height_trapezoid K L M N = 1) 
     (perp_KP_MQ : is_perpendicular(K P MN) ∧ is_perpendicular(M Q KL)) 
     (KN_MQ_eq : distance K N = distance M Q) 
     (LM_MP_eq : distance L M = distance M P), 
  distance L M = Real.sqrt 2 :=
by
  sorry

end trapezoid_LM_sqrt2_l207_207797


namespace geometric_sequence_properties_l207_207008

theorem geometric_sequence_properties (a : ℕ → ℕ) (q : ℕ)
  (h1 : a 2 - a 1 = 2)
  (h2 : 2 * a 2 = (3*a 1 + a 3) / 2) :
  a 1 = 1 ∧ q = 3 ∧ ∀ n, (S n = (3^n - 1) / 2) :=
sorry

end geometric_sequence_properties_l207_207008


namespace paula_paint_cans_needed_l207_207851

-- Let's define the initial conditions and required computations in Lean.
def initial_rooms : ℕ := 48
def cans_lost : ℕ := 4
def remaining_rooms : ℕ := 36
def large_rooms_to_paint : ℕ := 8
def normal_rooms_to_paint : ℕ := 20
def paint_per_large_room : ℕ := 2 -- as each large room requires twice as much paint

-- Define a function to compute the number of cans required.
def cans_needed (initial_rooms remaining_rooms large_rooms_to_paint normal_rooms_to_paint paint_per_large_room : ℕ) : ℕ :=
  let rooms_lost := initial_rooms - remaining_rooms
  let cans_per_room := rooms_lost / cans_lost
  let total_room_equivalents := large_rooms_to_paint * paint_per_large_room + normal_rooms_to_paint
  total_room_equivalents / cans_per_room

theorem paula_paint_cans_needed : cans_needed initial_rooms remaining_rooms large_rooms_to_paint normal_rooms_to_paint paint_per_large_room = 12 :=
by
  -- The proof would go here
  sorry

end paula_paint_cans_needed_l207_207851


namespace sum_series_equals_half_l207_207994

theorem sum_series_equals_half :
  ∑' n, 1 / (n * (n+1) * (n+2)) = 1 / 2 :=
sorry

end sum_series_equals_half_l207_207994


namespace seat_arrangement_l207_207901

theorem seat_arrangement (seats : ℕ) (people : ℕ) (min_empty_between : ℕ) : 
  seats = 9 ∧ people = 3 ∧ min_empty_between = 2 → 
  ∃ ways : ℕ, ways = 60 :=
by
  intro h
  sorry

end seat_arrangement_l207_207901


namespace x_squared_plus_y_squared_l207_207349

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : x - y = 12) (h2 : x * y = 9) : x^2 + y^2 = 162 :=
by
  sorry

end x_squared_plus_y_squared_l207_207349


namespace reflection_problem_l207_207552

theorem reflection_problem (v₁ v₂ w : ℝ × ℝ)
  (h₁ : v₁ = (3, 2))
  (h₂ : v₂ = (1, 6))
  (h₃ : w = (2, -1)) :
  let mp := ((fst v₁ + fst v₂) / 2, (snd v₁ + snd v₂) / 2) in
  let line_vec := (fst mp - fst v₁, snd mp - snd v₁) in
  let proj_w := ((w.1 * line_vec.1 + w.2 * line_vec.2) / (line_vec.1 ^ 2 + line_vec.2 ^ 2)) * line_vec in
  let reflection := (2 * proj_w.1 - w.1, 2 * proj_w.2 - w.2) in
  reflection = (-2 / 5, -11 / 5) :=
by
  sorry

end reflection_problem_l207_207552


namespace find_min_max_sum_l207_207831

noncomputable def sum_min_max_expression (a b c d : ℝ) : ℝ :=
  4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4)

theorem find_min_max_sum :
  ∀ a b c d : ℝ,
  a + b + c + d = 6 →
  a^2 + b^2 + c^2 + d^2 = 12 →
  let m := Real.Inf {x | ∃ (a b c d : ℝ), a + b + c + d = 6 ∧ a^2 + b^2 + c^2 + d^2 = 12 ∧ x = sum_min_max_expression a b c d} in
  let M := Real.Sup {x | ∃ (a b c d : ℝ), a + b + c + d = 6 ∧ a^2 + b^2 + c^2 + d^2 = 12 ∧ x = sum_min_max_expression a b c d} in
  m + M = 84 :=
by
  intros
  sorry

end find_min_max_sum_l207_207831


namespace scientific_notation_43300000_l207_207889

theorem scientific_notation_43300000 : 43300000 = 4.33 * 10^7 :=
by
  sorry

end scientific_notation_43300000_l207_207889


namespace intersection_M_N_l207_207312

-- Define the universe U
def U : Set ℤ := {-2, -1, 0, 1, 2}

-- Define the set M based on the condition x^2 <= x
def M : Set ℤ := {x ∈ U | x^2 ≤ x}

-- Define the set N based on the condition x^3 - 3x^2 + 2x = 0
def N : Set ℤ := {x ∈ U | x^3 - 3*x^2 + 2*x = 0}

-- State the theorem to be proven
theorem intersection_M_N : M ∩ N = {0, 1} :=
by
  sorry

end intersection_M_N_l207_207312


namespace perimeter_triangle_ABC_l207_207006

-- Definitions from the conditions
noncomputable def radius := 2
noncomputable def distance_between_centers := 2 * radius

-- The theorem statement that we need to prove
theorem perimeter_triangle_ABC (P Q R : Point) (A B C : Point) 
  (hPQ_tangent_AB : tangent_point P Q = B)
  (hPR_tangent_AC : tangent_point P R = C)
  (hQR_tangent_BC : tangent_point Q R = C)
  (hP_center : distance A P = radius ∧ distance C P = radius)
  (hQ_center : distance A Q = radius ∧ distance B Q = radius)
  (hR_center : distance C R = radius ∧ distance B R = radius)
  (hTrianglesSimilar : is_similar_triangle P Q R A B C) :
  perimeter A B C = 24 := 
sorry

end perimeter_triangle_ABC_l207_207006


namespace range_of_a_l207_207310

theorem range_of_a (x y a : ℝ): 
  (x + 3 * y = 3 - a) ∧ (2 * x + y = 1 + 3 * a) ∧ (x + y > 3 * a + 4) ↔ (a < -3 / 2) :=
sorry

end range_of_a_l207_207310


namespace exist_a_b_monotonicity_l207_207718

noncomputable def f (a b x : ℝ) : ℝ := a*x^2 - b*log(x)
noncomputable def g (a b x : ℝ) : ℝ := f a b x - x^2 + (x - 1)

theorem exist_a_b_monotonicity 
    (h₁ : ∃ (a b : ℝ), f a b 1 = 1 ∧ (2 * a - b = 0)) 
    (h₂ : ∀ x, 0 < x ∧ x ≤ 1): 
    ∃ a : ℝ, ∀ x : ℝ, 0 < x ∧ x ≤ 1 
    → ((a ≤ 0 ∧ g a 4 x < 0) 
    ∨ (0 < a ∧ a ≤ 4 ∧ g a 4 x < 0) 
    ∨ (a > 4 ∧ g a 4 x < g a 4 (4/a) ≤ 0)) :=
begin
  sorry
end

end exist_a_b_monotonicity_l207_207718


namespace jeff_stars_l207_207239

noncomputable def eric_stars : ℕ := 4
noncomputable def chad_initial_stars : ℕ := 2 * eric_stars
noncomputable def chad_stars_after_sale : ℕ := chad_initial_stars - 2
noncomputable def total_stars : ℕ := 16
noncomputable def stars_eric_and_chad : ℕ := eric_stars + chad_stars_after_sale

theorem jeff_stars :
  total_stars - stars_eric_and_chad = 6 := 
by 
  sorry

end jeff_stars_l207_207239


namespace largest_prime_factor_of_12321_l207_207638

theorem largest_prime_factor_of_12321 : ∃ p : ℕ, prime p ∧ p = 43 ∧ (∀ q : ℕ, prime q ∧ q ∣ 12321 → q ≤ p) :=
by
  sorry

end largest_prime_factor_of_12321_l207_207638


namespace system_solution_correct_l207_207711

theorem system_solution_correct (b : ℝ) : (∃ x y : ℝ, (y = 3 * x - 5) ∧ (y = 2 * x + b) ∧ (x = 1) ∧ (y = -2)) ↔ b = -4 :=
by
  sorry

end system_solution_correct_l207_207711


namespace total_adults_wearing_hats_l207_207414

theorem total_adults_wearing_hats (total_adults : ℕ) (men_percentage : ℝ) (men_hats_percentage : ℝ) 
  (women_hats_percentage : ℝ) (total_men_wearing_hats : ℕ) (total_women_wearing_hats : ℕ) : 
  (total_adults = 1200) ∧ (men_percentage = 0.60) ∧ (men_hats_percentage = 0.15) 
  ∧ (women_hats_percentage = 0.10)
     → total_men_wearing_hats + total_women_wearing_hats = 156 :=
by
  -- Definitions
  let total_men := total_adults * men_percentage
  let total_women := total_adults - total_men
  let men_wearing_hats := total_men * men_hats_percentage
  let women_wearing_hats := total_women * women_hats_percentage
  sorry

end total_adults_wearing_hats_l207_207414


namespace place_balls_in_boxes_l207_207419

-- Definitions based on conditions in the problem
def num_balls : ℕ := 5
def num_boxes : ℕ := 4

-- Statement of the problem
theorem place_balls_in_boxes : ∃ (ways : ℕ), (ways = 240) :=
by {
  have balls := num_balls,
  have boxes := num_boxes,
  sorry
}

end place_balls_in_boxes_l207_207419


namespace ellipse_equation_l207_207293

open Real

-- Definitions based directly on given conditions
def is_focus (F : ℝ × ℝ) (x : ℝ) (y : ℝ) (a : ℝ) (b : ℝ) : Prop :=
  let c := sqrt (a^2 - b^2)
  F = (c, 0)

def is_eccentricity (e : ℝ) (a : ℝ) : Prop :=
  e = c / a

def standard_equation (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧
  ∀ x y, (x^2 / a^2 + y^2 / b^2 = 1)

-- Lean statement that ties all the conditions to the correct answer
theorem ellipse_equation : (∃ F e a b, is_focus F 1 0 a b ∧ is_eccentricity e a ∧ standard_equation a b) →
  standard_equation 2 (sqrt 3) :=
begin
  sorry
end

end ellipse_equation_l207_207293


namespace tens_digit_of_9_pow_2010_is_0_l207_207917

theorem tens_digit_of_9_pow_2010_is_0 : 
  ∀ (n : ℕ),
  (n % 10 = 0) → 
  (let cycle := [9, 81, 29, 61, 49, 41, 69, 21, 89, 1] in
  let two_digits := cycle[((n % 10 + 9) % 10)] in
  (two_digits / 10).to_nat % 10 = 0) := 
by
  intros n h_mod
  let cycle := [9, 81, 29, 61, 49, 41, 69, 21, 89, 1]
  let two_digits := cycle[((n % 10 + 9) % 10)]
  sorry

end tens_digit_of_9_pow_2010_is_0_l207_207917


namespace trapezoid_LM_sqrt2_l207_207795

theorem trapezoid_LM_sqrt2 (K L M N P Q : Point) : 
  ∀ (h_trapezoid : is_trapezoid K L M N) 
     (diag_eq_height : distance K M = 1 ∧ height_trapezoid K L M N = 1) 
     (perp_KP_MQ : is_perpendicular(K P MN) ∧ is_perpendicular(M Q KL)) 
     (KN_MQ_eq : distance K N = distance M Q) 
     (LM_MP_eq : distance L M = distance M P), 
  distance L M = Real.sqrt 2 :=
by
  sorry

end trapezoid_LM_sqrt2_l207_207795


namespace sandwiches_difference_l207_207412

theorem sandwiches_difference :
  let monday_lunch := 3
  let monday_dinner := 2 * monday_lunch
  let monday_total := monday_lunch + monday_dinner

  let tuesday_lunch := 4
  let tuesday_dinner := tuesday_lunch / 2
  let tuesday_total := tuesday_lunch + tuesday_dinner

  let wednesday_lunch := 2 * tuesday_lunch
  let wednesday_dinner := 3 * tuesday_lunch
  let wednesday_total := wednesday_lunch + wednesday_dinner

  let total_mw := monday_total + tuesday_total + wednesday_total

  let thursday_lunch := 3 * 2
  let thursday_dinner := 5
  let thursday_total := thursday_lunch + thursday_dinner

  total_mw - thursday_total = 24 :=
by
  sorry

end sandwiches_difference_l207_207412


namespace problem_arithmetic_seq_problem_sum_problem_geometric_seq_l207_207815

noncomputable def a_n : ℕ → ℝ := λ n, if n = 0 then 1 else 2 * n - 1
noncomputable def S_n (n : ℕ) : ℝ := (a_n n * a_n n + 1) / 4
noncomputable def b_n (n : ℕ) : ℝ := a_n n / 2^n
noncomputable def T_n (n : ℕ) : ℝ := ∑ i in finset.range n, b_n (i + 1)
noncomputable def R_n (n : ℕ) (λ : ℝ) : ℝ := (T_n n + λ) / a_n (n + 2)

theorem problem_arithmetic_seq (n : ℕ) :
  S_n n = (1 / 4) * (a_n n + 1) ^ 2 :=
sorry

theorem problem_sum (n : ℕ) :
  T_n n = 3 - (2 * n - 3) / 2^n :=
sorry

theorem problem_geometric_seq (n : ℕ) :
  ∃ λ : ℝ, (∀ n > 0, R_n n λ / R_n (n + 1) λ = constant_ratio) :=
sorry

end problem_arithmetic_seq_problem_sum_problem_geometric_seq_l207_207815


namespace resistor_problem_l207_207474

theorem resistor_problem (R : ℝ)
  (initial_resistance : ℝ := 3 * R)
  (parallel_resistance : ℝ := R / 3)
  (resistance_change : ℝ := initial_resistance - parallel_resistance)
  (condition : resistance_change = 10) : 
  R = 3.75 := by
  sorry

end resistor_problem_l207_207474


namespace union_A_B_eq_l207_207401

noncomputable def A (a : ℝ) : Set ℝ := {5, log 2 (a + 3)}
def B (a b : ℝ) : Set ℝ := {a, b}

theorem union_A_B_eq {a b : ℝ} (h_intersect : A a ∩ B a b = {1}) : 
  A (-1) ∪ B (-1) 1 = {-1, 1, 5} :=
by
  sorry

end union_A_B_eq_l207_207401


namespace first_term_exceeds_2016_l207_207907

def sequence : ℕ → ℕ
| 0     := 12
| 1     := 19
| (n+2) := if (sequence n + sequence (n+1)) % 2 = 1 
           then sequence n + sequence (n+1) 
           else |sequence n - sequence (n+1)|

theorem first_term_exceeds_2016 : ∃ n, sequence n > 2016 ∧ n = 504 :=
by {
  sorry
}

end first_term_exceeds_2016_l207_207907


namespace marbles_lost_l207_207908

def initial_marbles := 8
def current_marbles := 6

theorem marbles_lost : initial_marbles - current_marbles = 2 :=
by
  sorry

end marbles_lost_l207_207908


namespace least_stamps_to_make_60_cents_l207_207587

theorem least_stamps_to_make_60_cents : 
  ∃ x y : ℕ, (5 * x + 6 * y = 60) ∧ (x + y = 10) :=
begin
  sorry
end

end least_stamps_to_make_60_cents_l207_207587


namespace problem_1_expression_problem_2_expression_l207_207212

section Problem1
variable (α : Real)

-- Conditions for Problem 1
def sin_alpha_eq_root : (5 * (!\sin α)^2 - 7 * !\sin α - 6 = 0) :=
sorry -- (Assume some particular α satisfies this condition)

def alpha_in_third_quadrant : (π ≤ α ∧ α ≤ 3 * π / 2) :=
sorry -- (Assume α is in the third quadrant)

-- Statement to prove for Problem 1
theorem problem_1_expression :
  sin_alpha_eq_root α →
  alpha_in_third_quadrant α →
  ( sin (-α - 3/2*π) * cos (3/2*π - α) /
    (cos (π/2 - α) * sin(π/2 + α)) ) *
  (tan (π - α))^2 = -9/16 :=
sorry
end Problem1

section Problem2

-- Conditions for Problem 2
def sin_40_deg : Real := sin (40 * π / 180)
def cos_40_deg : Real := cos (40 * π / 180)
def sin_50_deg : Real := sin (50 * π / 180)

-- Statement to prove for Problem 2
theorem problem_2_expression :
  (sqrt (1 - 2 * sin_40_deg * cos_40_deg) /
   (cos_40_deg - sqrt (1 - sin_50_deg^2))) = 1 :=
sorry

end Problem2

end problem_1_expression_problem_2_expression_l207_207212


namespace probability_xi_greater_than_2_l207_207355

noncomputable def normalDist := MeasureTheory.ProbabilityDistribution.normal 0 σ^2

variables (ξ : ℝ) (σ : ℝ)
  [fact (σ > 0)]

theorem probability_xi_greater_than_2 :
  MeasureTheory.Probability.mass (λ x, ξ > 2) normalDist = 0.1 :=
by
  have h1 : MeasureTheory.Probability.mass (λ x, -2 < ξ ≤ 0) normalDist = 0.4 := sorry
  have h2 : MeasureTheory.Probability.mass (λ x, 0 < ξ ≤ 2) normalDist = 0.4 := sorry
  have h3 : MeasureTheory.Probability.mass (λ x, -2 ≤ ξ ≤ 2) normalDist = 0.8 := sorry
  have h4 : MeasureTheory.Probability.mass (λ x, |ξ| > 2) normalDist = 0.2 := sorry
  have h5 : MeasureTheory.Probability.mass (λ x, ξ > 2) normalDist = 0.1 := sorry
  exact h5

end probability_xi_greater_than_2_l207_207355


namespace garden_enlargement_l207_207965

-- Define the problem conditions
def rect_length : ℝ := 40
def rect_width : ℝ := 20
def rect_area : ℝ := rect_length * rect_width
def rect_perimeter : ℝ := 2 * (rect_length + rect_width)
def square_side : ℝ := rect_perimeter / 4
def square_area : ℝ := square_side * square_side

-- State the theorem to be proved
theorem garden_enlargement : square_area - rect_area = 100 := by
  sorry

end garden_enlargement_l207_207965


namespace equilateral_triangle_area_l207_207499

theorem equilateral_triangle_area (d : ℝ) :
  let ABC : Type := ℝ × ℝ × ℝ in
  let C := (d, 0) in
  let A := (0, d) in
  let B := (d, 0) in
  let area := d^2 * (Real.pi / 6 + Real.sqrt 3 / 2) in
  sorry -- This is where the proof would be

end equilateral_triangle_area_l207_207499


namespace constant_term_in_binomial_expansion_l207_207084

theorem constant_term_in_binomial_expansion :
  let x := λ r : ℕ, (nat.choose 6 r) * (-2)^r in
  x 2 = 60 :=
by
  sorry

end constant_term_in_binomial_expansion_l207_207084


namespace eight_digit_not_perfect_square_l207_207426

theorem eight_digit_not_perfect_square : ∀ x : ℕ, 0 ≤ x ∧ x ≤ 9999 → ¬ ∃ y : ℤ, (99990000 + x) = y * y := 
by
  intros x hx
  intro h
  obtain ⟨y, hy⟩ := h
  sorry

end eight_digit_not_perfect_square_l207_207426


namespace chickens_in_coop_l207_207111

theorem chickens_in_coop (C : ℕ)
  (H1 : ∃ C : ℕ, ∀ R : ℕ, R = 2 * C)
  (H2 : ∃ R : ℕ, ∀ F : ℕ, F = 2 * R - 4)
  (H3 : ∃ F : ℕ, F = 52) :
  C = 14 :=
by sorry

end chickens_in_coop_l207_207111


namespace expected_value_of_sum_l207_207333

variable {s : Finset ℕ} (h_s : s = {1, 2, 3, 4, 5, 6}) (n : ℕ) [IncRange : 1 ≤ n ∧ n ≤ 6] 

theorem expected_value_of_sum (hmarbles : s.card = 6) (hk : choose 3 6 = 20) : 
  (∑ x in (s.powerset.filter (λ xs, xs.card = 3)), xs.sum) / 20 = 10.5 :=
by
  sorry

end expected_value_of_sum_l207_207333


namespace eq1_eq2_eq2_neq_eq3_l207_207569

theorem eq1_eq2 (x : ℝ) (h : x ≠ 3) : 
  ( -x - 2)/(x - 3) = (x + 1)/(x - 3) ↔ -x - 2 = x + 1 := 
by 
  sorry

theorem eq2_neq_eq3 (x : ℝ) : 
  (-x - 2) = (x + 1) ≠ (-x - 2)*(x - 3) = (x + 1)*(x - 3) :=
by 
  sorry

end eq1_eq2_eq2_neq_eq3_l207_207569


namespace triangle_with_consecutive_sides_and_angle_property_l207_207814

theorem triangle_with_consecutive_sides_and_angle_property :
  ∃ (a b c : ℕ), (b = a + 1) ∧ (c = b + 1) ∧
    (∃ (α β γ : ℝ), 2 * α = γ ∧
      (a * a + b * b = c * c + 2 * a * b * α.cos) ∧
      (b * b + c * c = a * a + 2 * b * c * β.cos) ∧
      (c * c + a * a = b * b + 2 * c * a * γ.cos) ∧
      (a = 4) ∧ (b = 5) ∧ (c = 6) ∧
      (γ.cos = 1 / 8)) :=
sorry

end triangle_with_consecutive_sides_and_angle_property_l207_207814


namespace alcohol_mixture_percentage_l207_207439

/-- 
Given:
- Solution x is 10 percent alcohol by volume.
- Solution y is 30 percent alcohol by volume.
- 900 milliliters of solution y and 300 milliliters of solution x are mixed.
Prove that the resulting solution has 25% alcohol by volume.
-/
theorem alcohol_mixture_percentage :
  let volume_x := 300 -- in milliliters
  let volume_y := 900 -- in milliliters
  let percent_x := 0.10 -- percentage of alcohol in solution x
  let percent_y := 0.30 -- percentage of alcohol in solution y
  let volume_alcohol_x := percent_x * volume_x
  let volume_alcohol_y := percent_y * volume_y
  let total_volume := volume_x + volume_y
  let total_volume_alcohol := volume_alcohol_x + volume_alcohol_y
  let result_percent := (total_volume_alcohol / total_volume) * 100
  in result_percent = 25 :=
by
  sorry

end alcohol_mixture_percentage_l207_207439


namespace sum_of_reciprocals_l207_207829

open Polynomial

variable (a b c : ℂ) (h1: (a - 2) * (b - 2) * (c - 2) = 0)

theorem sum_of_reciprocals : 
(∃ a b c : ℂ, (X^3 - 2 * X + 4).isRoot a ∧ (X^3 - 2 * X + 4).isRoot b ∧ (X^3 - 2 * X + 4).isRoot c ∧ 
∑ x in [{a}, {b}, {c}], (1 / (x - 2)) = -5/4) :=
begin
  sorry
end

end sum_of_reciprocals_l207_207829


namespace numbers_must_be_equal_l207_207685

theorem numbers_must_be_equal
  (n : ℕ) (nums : Fin n → ℕ)
  (hn_pos : n = 99)
  (hbound : ∀ i, nums i < 100)
  (hdiv : ∀ (s : Finset (Fin n)) (hs : 2 ≤ s.card), ¬ 100 ∣ s.sum nums) :
  ∀ i j, nums i = nums j := 
sorry

end numbers_must_be_equal_l207_207685


namespace trapezoid_area_sum_l207_207559

theorem trapezoid_area_sum :
  let r_1 := 56
      r_2 := 126
      r_3 := 0
      n_1 := 320
      n_2 := 240
  in ⌊r_1 + r_2 + r_3 + n_1 + n_2⌋ = 742 :=
by
  -- Assuming the correctness of the given areas and integer forms,
  -- we compute the sum:
  let r_1 := 56
  let r_2 := 126
  let r_3 := 0
  let n_1 := 320
  let n_2 := 240
  have sum : r_1 + r_2 + r_3 + n_1 + n_2 = 742 := by rfl
  exact sum ▸ by rfl

end trapezoid_area_sum_l207_207559


namespace min_visible_sum_l207_207523

theorem min_visible_sum (n : ℕ) (h_n : n = 4) :
  let corner_dice := 8
  let edge_dice := 24
  let face_center_dice := 24
  let internal_dice := (n^3 - corner_dice - edge_dice - face_center_dice)
  in
  (∀ (d : ℕ), d ∈ ({1, 2, 3, 4, 5, 6}) → d = 7 - d) →
  (min_visible_sum := (corner_dice * 6 + edge_dice * 3 + face_center_dice * 1)) →
  min_visible_sum = 144 :=
by
  -- We assume the user will complete the proof here.
  sorry

end min_visible_sum_l207_207523


namespace largest_possible_value_of_abs_z_l207_207035

variables {a b c d z : ℂ} {k : ℝ}

/-- Given conditions - each definition is needed as a condition in Lean 4 -/
def condition_1 (a d : ℂ) : Prop := |a| = |d| ∧ |d| > 0
def condition_2 (b d : ℂ) (k : ℝ) : Prop := b = k * d
def condition_3 (c d : ℂ) (k : ℝ) : Prop := c = k^2 * d
def condition_4 (a b c z : ℂ) : Prop := a * z^2 + b * z + c = 0

/-- Prove that given the conditions, the largest possible value of |z| is the expected result -/
theorem largest_possible_value_of_abs_z (h1 : condition_1 a d)
                                        (h2 : condition_2 b d k)
                                        (h3 : condition_3 c d k)
                                        (h4 : condition_4 a b c z) :
  |z| ≤ (k^3 + real.sqrt (k^6 + 4 * k^3)) / 2 :=
sorry

end largest_possible_value_of_abs_z_l207_207035


namespace incenter_proof_l207_207783

noncomputable def is_incenter_of_triangle (P Q R I : Point) : Prop := 
∃ I, 
  angle_bisector P Q R I ∧ 
  dist I Q = dist I R ∧ 
  dist I Q = dist I (seg_intersection (perpendicular bisector P Q) (perpendicular bisector Q R))

structure IncenterProofProblem where
  (O O' A B C D M N I : Point)
  (circleO : InscribedQuadrilateral O (A, B, C, D))
  (internal_tangent : AreTangentInternally O O')
  (tangentBC_MC : TangentAtPoint BC O' M)
  (tangentCD_ND : TangentAtPoint CD O' N)
  (bisectorBAD_intersection : AngleBisector BAD MN I)

theorem incenter_proof (problem : IncenterProofProblem) : is_incenter_of_triangle B C D I :=
sorry

end incenter_proof_l207_207783


namespace sum_sequence_l207_207376

def a : ℕ+ → ℤ
def S : ℕ → ℤ

axiom a_1 : a 1 = 2
axiom a_seq : ∀ n : ℕ+, a n + a (n + 1) = 1
axiom S_def : ∀ n, S n = ∑ i in finset.range n, a ⟨i.succ, by linarith⟩

theorem sum_sequence :
  S 2007 - 2 * S 2006 + S 2005 = 3 :=
sorry

end sum_sequence_l207_207376


namespace work_completion_time_l207_207156

theorem work_completion_time (d : ℚ) : 
  (∀ (A B : ℚ), A = 30 ∧ B = 55 → d ≈ 330 / 17) :=
by
  intros A B h
  cases h with hA hB
  have A_work_rate : ℚ := 1 / A
  have B_work_rate : ℚ := 1 / B
  have combined_work_rate : ℚ := A_work_rate + B_work_rate
  have := calc 
    1 / combined_work_rate 
      = d : by  
        linarith 
        sorry 
  exact this

end work_completion_time_l207_207156


namespace tea_mixture_price_l207_207016

theorem tea_mixture_price :
  ∀ (price_A price_B : ℝ) (ratio_A ratio_B : ℝ),
  price_A = 65 →
  price_B = 70 →
  ratio_A = 1 →
  ratio_B = 1 →
  (price_A * ratio_A + price_B * ratio_B) / (ratio_A + ratio_B) = 67.5 :=
by
  intros price_A price_B ratio_A ratio_B h1 h2 h3 h4
  sorry

end tea_mixture_price_l207_207016


namespace park_area_l207_207469

theorem park_area (l w : ℝ) (h1 : 2 * l + 2 * w = 80) (h2 : l = 3 * w) : l * w = 300 :=
sorry

end park_area_l207_207469


namespace total_balls_l207_207821

theorem total_balls (jungkook_balls : ℕ) (yoongi_balls : ℕ) (h1 : jungkook_balls = 3) (h2 : yoongi_balls = 4) : 
  jungkook_balls + yoongi_balls = 7 :=
by
  -- This is a placeholder for the proof
  sorry

end total_balls_l207_207821


namespace j_identical_to_inverse_l207_207983

def h : ℝ → ℝ := λ x, (x - 5) / (x - 4)

-- Condition 1: Graph of \( h(x) \) is symmetric with respect to the line \( y = x + 3 \).
def symmetric_about_line (f : ℝ → ℝ) (m c : ℝ) : Prop :=
  ∀ x, f (m - x + c) = m - f x + c

axiom h_symmetric : symmetric_about_line h 1 3

-- Question: For what value of \( b \) is \( j(x) = h(x + b) \) identical to its inverse \( j^{-1}(x) \)?
def j (b : ℝ) : ℝ → ℝ := λ x, h (x + b)

def j_inverse (b : ℝ) (x : ℝ) : ℝ := j b x

-- Prove that \( j(x) \) is identical to \( j^{-1}(x) \) given the conditions
theorem j_identical_to_inverse (b : ℝ) :
  (∀ x, j (-b) (j b x) = x) :=
by
  sorry

end j_identical_to_inverse_l207_207983


namespace polar_to_rectangular_conversion_l207_207223

noncomputable def r : ℝ := 3 * Real.sqrt 2
noncomputable def theta : ℝ := Real.pi / 4

def polarToRectangular (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem polar_to_rectangular_conversion :
  polarToRectangular r theta = (3, 3) :=
by
  sorry

end polar_to_rectangular_conversion_l207_207223


namespace shaded_hexagons_are_balanced_l207_207491

-- Definitions and conditions from the problem
def is_balanced (a b c : ℕ) : Prop :=
  (a = b ∧ b = c) ∨ (a ≠ b ∧ b ≠ c ∧ a ≠ c)

def hexagon_grid_balanced (grid : ℕ × ℕ → ℕ) : Prop :=
  ∀ (i j : ℕ),
  (i % 2 = 0 ∧ grid (i, j) = grid (i, j + 1) ∧ grid (i, j + 1) = grid (i + 1, j + 1))
  ∨ (grid (i, j) ≠ grid (i, j + 1) ∧ grid (i, j + 1) ≠ grid (i + 1, j + 1) ∧ grid (i, j) ≠ grid (i + 1, j + 1))
  ∨ (i % 2 ≠ 0 ∧ grid (i, j) = grid (i - 1, j) ∧ grid (i - 1, j) = grid (i - 1, j + 1))
  ∨ (grid (i, j) ≠ grid (i - 1, j) ∧ grid (i - 1, j) ≠ grid (i - 1, j + 1) ∧ grid (i, j) ≠ grid (i - 1, j + 1))

theorem shaded_hexagons_are_balanced (grid : ℕ × ℕ → ℕ) (h_balanced : hexagon_grid_balanced grid) :
  is_balanced (grid (1, 1)) (grid (1, 10)) (grid (10, 10)) :=
sorry

end shaded_hexagons_are_balanced_l207_207491


namespace first_super_lucky_year_after_2000_l207_207190

def is_super_lucky_year (year : ℕ) : Prop :=
  ∃ (m1 m2 d1 d2 : ℕ), 
    m1 ≠ m2 ∧ d1 ≠ d2 ∧ 
    m1 * d1 = year % 100 ∧ m2 * d2 = year % 100 ∧ 
    1 ≤ m1 ∧ m1 ≤ 12 ∧ 1 ≤ d1 ∧ d1 ≤ 31 ∧
    1 ≤ m2 ∧ m2 ≤ 12 ∧ 1 ≤ d2 ∧ d2 ≤ 31

theorem first_super_lucky_year_after_2000 : ∃ year, 2000 < year ∧ is_super_lucky_year year ∧
  (∀ y, 2000 < y → y < year → ¬ is_super_lucky_year y) :=
begin
  use 2004,
  split,
  { norm_num },
  split,
  { -- Proof that 2004 is a super lucky year
    use [1, 4, 4, 1],
    split,
    { norm_num },
    split,
    { norm_num },
    split,
    { norm_num },
    split,
    { norm_num },
    split,
    { norm_num },
    split,
    { norm_num },
    split,
    { norm_num },
    split,
    { norm_num },
    { norm_num }
  },
  {
    -- Proof that there is no super lucky year before 2004
    intros y hy1 hy2,
    -- This proof would involve checking each year individually,
    -- which cannot be easily expressed in Lean without computational support.
    sorry
  }
end

end first_super_lucky_year_after_2000_l207_207190


namespace mike_bricks_l207_207053

theorem mike_bricks (total_bricks bricks_A bricks_B bricks_other: ℕ) 
  (h1 : bricks_A = 40) 
  (h2 : bricks_B = bricks_A / 2)
  (h3 : total_bricks = 150) 
  (h4 : total_bricks = bricks_A + bricks_B + bricks_other) : bricks_other = 90 := 
by 
  sorry

end mike_bricks_l207_207053


namespace degree_of_polynomial_degree_of_exp_polynomial_l207_207119

def f (x : ℝ) : ℝ := 2 * x ^ 3 - 5 * x + 7

theorem degree_of_polynomial : polynomial.degree (polynomial.X^3 * 2 + polynomial.X * -5 + 7) = 3 :=
sorry

theorem degree_of_exp_polynomial : polynomial.degree (polynomial.expand 10 (polynomial.X^3 * 2 + polynomial.X * -5 + 7)) = 30 :=
sorry

end degree_of_polynomial_degree_of_exp_polynomial_l207_207119


namespace periodic_function_period_4_explicit_formula_in_interval_l207_207710

noncomputable def f : ℝ → ℝ := sorry -- Definition based on given conditions

axiom odd_function (f : ℝ → ℝ) : ∀ x, f(-x) = -f(x)
axiom symmetry_about_x1 (f : ℝ → ℝ) : ∀ x, f(2 - x) = f(x)
axiom value_in_interval (f : ℝ → ℝ) : ∀ x, 0 < x ∧ x ≤ 1 → f(x) = sqrt x

theorem periodic_function_period_4 (f : ℝ → ℝ) 
  (odd : ∀ x, f(-x) = -f(x))
  (symmetry : ∀ x, f(2 - x) = f(x))
  (interval_value : ∀ x, 0 < x ∧ x ≤ 1 → f(x) = sqrt x) : 
  ∀ x, f(x) = f(x + 4) :=
sorry

theorem explicit_formula_in_interval (f : ℝ → ℝ) 
  (odd : ∀ x, f(-x) = -f(x))
  (symmetry : ∀ x, f(2 - x) = f(x))
  (interval_value : ∀ x, 0 < x ∧ x ≤ 1 → f(x) = sqrt x) : 
  ∀ x, -5 ≤ x ∧ x ≤ -4 → f(x) = -sqrt(-(x + 4)) :=
sorry

end periodic_function_period_4_explicit_formula_in_interval_l207_207710


namespace calculation_l207_207986

theorem calculation :
  5 * 399 + 4 * 399 + 3 * 399 + 397 = 5185 :=
  by
    sorry

end calculation_l207_207986


namespace tangents_intersect_on_AC_l207_207967

variables {A B C O I B' : Point}
variables (ABC_circumcircle : Circle) (ABC_incircle : Circle)

-- Assume necessary conditions
axiom ABC_Scalene : IsScaleneTriangle A B C
axiom ABC_Inscribed : IsInscribedInCircle ABC_circumcircle A B C
axiom ABC_Circumscribed : IsCircumscribedAroundCircle ABC_incircle A B C
axiom B'_Symmetric : SymmetricWithRespectToLine O I B B'
axiom B'_WithinAngle : WithinAngle A B I B'

theorem tangents_intersect_on_AC :
  TangentsIntersectOnLine (CircumcircleOfTriangle B' B I) B' I A C :=
sorry

end tangents_intersect_on_AC_l207_207967


namespace ratio_distances_l207_207553

variables (A B C D P : Point)
variables (distance : Point → Plane → ℝ) (distance_line : Point → Line → ℝ)
variables (s S : ℝ)

-- Predicate to check if a set of points form a regular tetrahedron
def is_regular_tetrahedron (A B C D : Point) : Prop := sorry

-- Predicate to check if P is the centroid of the tetrahedron
def is_centroid (P A B C D : Point) : Prop := sorry

-- Definitions of the planes containing the faces of the tetrahedron
def plane (X Y Z : Point) : Plane := sorry

-- Definitions of the lines containing the edges of the tetrahedron
def line (X Y : Point) : Line := sorry

axiom distance_planes : 
  s = distance P (plane D A B) + distance P (plane D B C) + distance P (plane D C A)

axiom distance_lines : 
  S = distance_line P (line A B) + distance_line P (line B C) + distance_line P (line C A)

theorem ratio_distances :
  is_regular_tetrahedron A B C D →
  is_centroid P A B C D →
  s / S = √2 / 2 :=
by
  sorry -- proof goes here

end ratio_distances_l207_207553


namespace angle_sum_l207_207381

theorem angle_sum (A B C : ℝ) (h1 : A > 0) (h2 : B > 0) (h3 : C > 0) (h_triangle : A + B + C = 180) (h_complement : 180 - C = 130) :
  A + B = 130 :=
by
  sorry

end angle_sum_l207_207381


namespace hyperbola_equation_l207_207723

theorem hyperbola_equation 
    (a b : ℝ) 
    (C : ∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1) 
    (e : ℝ) 
    (h_ecc : e = 5/4) 
    (h_focus : 5 = real.sqrt (a^2 + b^2))
    : (∀ x y : ℝ, (x^2/16 - y^2/9) = 1) :=
by
  sorry

end hyperbola_equation_l207_207723


namespace number_of_apps_needed_l207_207403

-- Definitions based on conditions
variable (cost_per_app : ℕ) (total_money : ℕ) (remaining_money : ℕ)

-- Assume the conditions given
axiom cost_app_eq : cost_per_app = 4
axiom total_money_eq : total_money = 66
axiom remaining_money_eq : remaining_money = 6

-- The goal is to determine the number of apps Lidia needs to buy
theorem number_of_apps_needed (n : ℕ) (h : total_money - remaining_money = cost_per_app * n) :
  n = 15 :=
by
  sorry

end number_of_apps_needed_l207_207403


namespace standard_deviation_calculation_l207_207448

theorem standard_deviation_calculation : 
  let mean := 16.2 
  let stddev := 2.3 
  mean - 2 * stddev = 11.6 :=
by
  sorry

end standard_deviation_calculation_l207_207448


namespace percentage_x_equals_twenty_percent_of_487_50_is_65_l207_207949

theorem percentage_x_equals_twenty_percent_of_487_50_is_65
    (x : ℝ)
    (hx : x = 150)
    (y : ℝ)
    (hy : y = 487.50) :
    (∃ (P : ℝ), P * x = 0.20 * y ∧ P * 100 = 65) :=
by
  sorry

end percentage_x_equals_twenty_percent_of_487_50_is_65_l207_207949


namespace most_likely_outcomes_l207_207254

-- Define the probability function
noncomputable def probability (n : ℕ) (k : ℕ) : ℚ := 
  (Nat.choose n k) * (1/2)^n

-- Define the different outcomes
def outcome_A : ℚ := (1/2)^5
def outcome_B : ℚ := (1/2)^5
def outcome_C : ℚ := probability 5 3
def outcome_D : ℚ := 2 * probability 5 1

-- State the theorem
theorem most_likely_outcomes :
 (outcome_C = 5/16) ∧ (outcome_D = 5/16) ∧ (outcome_A = 1/32) ∧ (outcome_B = 1/32) := 
by
  sorry

end most_likely_outcomes_l207_207254


namespace expected_value_sum_marbles_l207_207341

theorem expected_value_sum_marbles :
  let marbles := {1, 2, 3, 4, 5, 6} in
  let combinations := {S | S ⊆ marbles ∧ S.size = 3} in
  let summed_values := {sum S | S ∈ combinations} in
  let total_sum := summed_values.sum in
  let number_of_combinations := combinations.card in
  (total_sum: ℚ) / number_of_combinations = 10.5 :=
sorry

end expected_value_sum_marbles_l207_207341


namespace decompose_even_odd_l207_207838

variable {R : Type*} [LinearOrderedField R]

def f (x : R) : R

def g (f : R → R) (x : R) : R := (f x + f (-x)) / 2

def h (f : R → R) (x : R) : R := (f x - f (-x)) / 2

theorem decompose_even_odd
  (f : R → R)
  (hf : ∀ x, f x = g f x + h f x)
  : (∀ x, g f x = g f (-x)) ∧ (∀ x, h f x = -h f x) :=
by
  sorry

end decompose_even_odd_l207_207838


namespace minimize_sum_of_squares_distances_plane_minimize_sum_of_squares_distances_space_l207_207137

noncomputable def sum_of_squares_distances {V : Type*} [InnerProductSpace ℝ V] (O : V) (points : Finset V) : ℝ :=
  points.sum (λ P, ∥O - P∥^2)

theorem minimize_sum_of_squares_distances_plane {V : Type*} [InnerProductSpace ℝ V] 
  (A B C D : V) : ∃ O, 
  O = (1 / 4 : ℝ) • (A + B + C + D) ∧ 
  ∀ M, sum_of_squares_distances O {A, B, C, D} ≤ sum_of_squares_distances M {A, B, C, D} := sorry

theorem minimize_sum_of_squares_distances_space {V : Type*} [InnerProductSpace ℝ V] 
  (A B C D : V) [FiniteDimensional ℝ V] : ∃ O, 
  O = (1 / 4 : ℝ) • (A + B + C + D) ∧ 
  ∀ M, sum_of_squares_distances O {A, B, C, D} ≤ sum_of_squares_distances M {A, B, C, D} := sorry

end minimize_sum_of_squares_distances_plane_minimize_sum_of_squares_distances_space_l207_207137


namespace product_mn_l207_207872

-- Definitions of the conditions
def radius : ℝ := 7
def BC : ℝ := 8
def M_bisects_CD : Prop := true  -- We assume the definition that M bisects CD
def unique_bisection_of_CD : Prop := true  -- Assume CD is uniquely bisected by BC

-- Definition for sine being rational
def sin_rational (α : ℝ) : Prop := ∃ (m n : ℕ), gcd m n = 1 ∧ real.sin α = (m : ℝ) / (n : ℝ)

-- Sine of the central angle of minor arc CB
def central_angle_CB : ℝ := real.arcsin ((BC / 2) / radius)

-- Main theorem statement
theorem product_mn (m n : ℕ) (h_gcd : nat.gcd m n = 1) (h_sin : real.sin (central_angle_CB) = (m : ℝ) / (n : ℝ)) : m * n = 28 := 
by 
  sorry

end product_mn_l207_207872


namespace area_of_rectangular_park_l207_207471

theorem area_of_rectangular_park
  (l w : ℕ) 
  (h_perimeter : 2 * l + 2 * w = 80)
  (h_length : l = 3 * w) :
  l * w = 300 :=
sorry

end area_of_rectangular_park_l207_207471


namespace total_squares_in_6x6_grid_l207_207614

theorem total_squares_in_6x6_grid : 
  let n := 6
  sum (λ k, (n - k) ^ 2) (range n) = 55 :=
by
  let n := 6
  have h1 : (range 6).map (λ k, (6 - k) ^ 2) = [25, 16, 9, 4, 1], by sorry
  have h2 : list.sum [25, 16, 9, 4, 1] = 55, by sorry
  exact h2

-- This adds assumptions and sums the number of squares from 1x1 to 5x5 in a 6x6 grid.

end total_squares_in_6x6_grid_l207_207614


namespace cost_price_to_selling_price_ratio_l207_207477

variable (CP SP : ℝ)
variable (profit_percent : ℝ)

theorem cost_price_to_selling_price_ratio
  (h1 : profit_percent = 0.25)
  (h2 : SP = (1 + profit_percent) * CP) :
  (CP / SP) = 4 / 5 := by
  sorry

end cost_price_to_selling_price_ratio_l207_207477


namespace volume_of_regular_square_pyramid_l207_207292

theorem volume_of_regular_square_pyramid (a : ℝ) (h : ℝ) (s : ℝ) (a_eq : a = 2)
  (h_eq : h = √2)
  (s_eq : s = 4) :
  1/3 * s * h = 4 * √2 / 3 :=
by 
  rw [a_eq, h_eq, s_eq]
  exact sorry

end volume_of_regular_square_pyramid_l207_207292


namespace star_operation_l207_207259

def new_op (a b : ℝ) : ℝ :=
  a^2 + b^2 - a * b

theorem star_operation (x y : ℝ) : 
  new_op (x + 2 * y) (y + 3 * x) = 7 * x^2 + 3 * y^2 + 3 * (x * y) :=
by
  sorry

end star_operation_l207_207259


namespace trapezoid_LM_sqrt2_l207_207796

theorem trapezoid_LM_sqrt2 (K L M N P Q : Point) : 
  ∀ (h_trapezoid : is_trapezoid K L M N) 
     (diag_eq_height : distance K M = 1 ∧ height_trapezoid K L M N = 1) 
     (perp_KP_MQ : is_perpendicular(K P MN) ∧ is_perpendicular(M Q KL)) 
     (KN_MQ_eq : distance K N = distance M Q) 
     (LM_MP_eq : distance L M = distance M P), 
  distance L M = Real.sqrt 2 :=
by
  sorry

end trapezoid_LM_sqrt2_l207_207796


namespace ice_cream_stackings_l207_207068

theorem ice_cream_stackings :
  let scoops := ["vanilla", "chocolate", "strawberry", "cherry", "pistachio"]
  in (scoops.length).factorial = 120 :=
by
  sorry

end ice_cream_stackings_l207_207068


namespace parabola_slope_l207_207286

theorem parabola_slope (p : ℝ) (A B R : ℝ × ℝ) (FA FB : ℝ) :
  R = (2, 1) →
  (R.1 + R.1) / 2 = 2 →
  (R.2 + R.2) / 2 = 1 →
  R = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  A.2 ^ 2 = 2 * p * A.1 →
  B.2 ^ 2 = 2 * p * B.1 →
  |FA| + |FB| = 5 →
  let F := (p / 2, 0) in 
  let k := (A.2 - B.2) / (A.1 - B.1) in 
  k = 1 :=
begin
  sorry
end

end parabola_slope_l207_207286


namespace product_of_specific_integers_less_than_100_is_square_l207_207211

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def has_three_divisors (n : ℕ) : Prop :=
  ∃ p, is_prime p ∧ n = p * p

theorem product_of_specific_integers_less_than_100_is_square :
  let values := [4, 9, 25, 49] in
  ((values.prod = 44100) ∧ ∃ k, k * k = values.prod) :=
by
  let values := [4, 9, 25, 49]
  have product := values.prod
  have h1 : product = 4 * 9 * 25 * 49 := by sorry
  have h2 : product = 44100 := by sorry
  have h3 : ∃ k, k * k = 44100 := by sorry
  exact ⟨h2, h3⟩

end product_of_specific_integers_less_than_100_is_square_l207_207211


namespace average_score_of_group_l207_207765

theorem average_score_of_group 
  (scores_6_avg_90 : ∀ (i : ℕ), 0 ≤ i ∧ i < 6 → ℕ, scores_6_avg_90 i = 90)
  (scores_4_avg_80 : ∀ (i : ℕ), 0 ≤ i ∧ i < 4 → ℕ, scores_4_avg_80 i = 80) :
  86 = (∑ i in finset.range 6, scores_6_avg_90 i + ∑ i in finset.range 4, scores_4_avg_80 i) / 10 := 
by
  sorry

end average_score_of_group_l207_207765


namespace complex_number_satisfying_iz_eq_1_is_neg_i_l207_207687

theorem complex_number_satisfying_iz_eq_1_is_neg_i (z : ℂ) (h : (complex.I * z) = 1) : z = -complex.I :=
by
  sorry

end complex_number_satisfying_iz_eq_1_is_neg_i_l207_207687


namespace dice_probability_l207_207612

theorem dice_probability :
  let total_outcomes := 6^4 in
  let success_one_pair := 6 * (Nat.choose 4 2) * 5 * 4 in
  let success_two_pairs := (Nat.choose 6 2) * (4! / (2! * 2!)) in
  let total_success := success_one_pair + success_two_pairs in
  total_success / total_outcomes = 5 / 8 :=
by
  let total_outcomes := 6^4
  let success_one_pair := 6 * (Nat.choose 4 2) * 5 * 4
  let success_two_pairs := (Nat.choose 6 2) * (4! / (2! * 2!))
  let total_success := success_one_pair + success_two_pairs
  have h1 : total_outcomes = 1296 := by norm_num
  have h2 : success_one_pair = 720 := by norm_num
  have h3 : success_two_pairs = 90 := by norm_num
  have h4 : total_success = 810 := by norm_num
  have h5 : (total_success / total_outcomes) = (5 / 8) := by norm_num
  exact h5

end dice_probability_l207_207612


namespace good_set_exists_element_removal_l207_207557

-- Defining the property of a 'good' set
def is_good (C : Set ℕ) : Prop :=
  ∀ k : ℤ, ∃ (a b : ℕ), a ≠ b ∧ a ∈ C ∧ b ∈ C ∧ Int.gcd (a + Nat.ofInt k) (b + Nat.ofInt k) > 1

theorem good_set_exists_element_removal (C : Set ℕ)
  (hC_good : is_good C)
  (hC_sum : C.toFinset.sum id = 2003) :
  ∃ c ∈ C, is_good (C \ {c}) :=
sorry

end good_set_exists_element_removal_l207_207557


namespace evaluate_expression_l207_207256

theorem evaluate_expression (x : ℝ) : x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3*x^3 - 5*x^2 + 12*x + 2 := by
  sorry

end evaluate_expression_l207_207256


namespace angle_sum_is_ninety_degrees_l207_207043

-- Define the geometrical constructs
variables {Point : Type} {Line : Type} {Plane : Type}
variable [has_angle Line Plane]

-- Assume the conditions given
variables (p : Line) (pi : Plane)
variable (hp : is_perpendicular p pi)

-- The definition of angle between a line and a plane/hplane
noncomputable def angle (l : Line) (π : Plane) : ℝ := sorry
noncomputable def angle (l : Line) (m : Line) : ℝ := sorry

-- Now, the theorem statement
theorem angle_sum_is_ninety_degrees (l : Line) : 
  angle l pi + angle l p = 90 := sorry

end angle_sum_is_ninety_degrees_l207_207043


namespace walking_rate_on_escalator_l207_207204

theorem walking_rate_on_escalator (v : ℝ)
  (escalator_speed : ℝ := 12)
  (escalator_length : ℝ := 160)
  (time_taken : ℝ := 8)
  (distance_eq : escalator_length = (v + escalator_speed) * time_taken) :
  v = 8 :=
by
  sorry

end walking_rate_on_escalator_l207_207204


namespace x2022_equals_1_l207_207675

noncomputable def sequence (n : ℕ) : ℤ :=
if n = 1 then 1 else
if n = 2 then 1 else
if n = 3 then -1 else
sequence (n-1) * sequence (n-3)

theorem x2022_equals_1 : sequence 2022 = 1 :=
sorry

end x2022_equals_1_l207_207675


namespace range_of_a_l207_207091

noncomputable def f : ℝ → ℝ := sorry
variable (f_even : ∀ x : ℝ, f x = f (-x))
variable (f_increasing : ∀ x y : ℝ, x ≤ y ∧ y ≤ 0 → f x ≤ f y)
variable (a : ℝ) (h : f a ≤ f 2)

theorem range_of_a (f_even : ∀ x : ℝ, f x = f (-x))
                   (f_increasing : ∀ x y : ℝ, x ≤ y ∧ y ≤ 0 → f x ≤ f y)
                   (h : f a ≤ f 2) :
                   a ≤ -2 ∨ a ≥ 2 :=
sorry

end range_of_a_l207_207091


namespace map_distance_cm_l207_207847

/-- The number of centimeters measured on a map given:
1. On a map, 2 inches represent 30 miles.
2. 1 inch is 2.54 centimeters.
3. The actual distance is approximately 224.40944881889763 miles.
-/
theorem map_distance_cm (miles_to_inches : ℝ) (inches_to_cm : ℝ) (measured_miles : ℝ) : 
  miles_to_inches = 30 / 2 → 
  inches_to_cm = 2.54 → 
  measured_miles = 224.40944881889763 → 
  ∃ cm : ℝ, cm ≈ 38.00524934383202 :=
by
  sorry

end map_distance_cm_l207_207847


namespace problem_statement_l207_207696

variable (x : ℝ) (x₀ : ℝ)

def p : Prop := ∀ x > 0, x + 4 / x ≥ 4

def q : Prop := ∃ x₀ ∈ Set.Ioi (0 : ℝ), 2 * x₀ = 1 / 2

theorem problem_statement : p ∧ ¬q :=
by
  sorry

end problem_statement_l207_207696


namespace largest_prime_factor_12321_l207_207649

theorem largest_prime_factor_12321 : ∃ p, prime p ∧ (∀ q, prime q ∧ q ∣ 12321 → q ≤ p) ∧ p = 19 :=
by {
  sorry
}

end largest_prime_factor_12321_l207_207649


namespace ivanushka_strategy_l207_207780

def source_label := ℕ

def can_survive_from (source1 source2 : source_label) : Prop := 
  source1 < source2

def is_deadly_except (person : String) (source : source_label) : Prop := 
  source = 10 ∧ person ≠ "Koschei"

def duel_outcome (koschei_drink ivan_drink : source_label) : Prop :=
  (koschei_drink = 10 → True) ∧ (ivan_drink = 1 ∧ koschei_drink = 10 → True)

theorem ivanushka_strategy:
  ∀ (koschei_drink ivan_drink : source_label),
    koschei_drink = 10 → ivan_drink = 1 → 
    duel_outcome koschei_drink ivan_drink :=
by
  intros koschei_drink ivan_drink
  intros h_koschei_drink h_ivan_drink
  unfold duel_outcome
  split
  exact True.intro
  intro h_combined
  exact True.intro

#check ivanushka_strategy -- ensure it type checks correctly

end ivanushka_strategy_l207_207780


namespace encode_MATEMATIKA_l207_207555

variable (encode : String → String)

/-- Conditions -/
def encode_ROBOT : encode "ROBOT" = "3112131233" := by sorry

def encode_KROKODIL_and_BEGEMOT_identical : encode "KROKODIL" = encode "BEGEMOT" := by sorry

def letters_encoded_with_123 : ∀ c : Char, encode (Char.toString c) ∈ {"1", "2", "3", "11", "12", "13", "21", "22", "23", "31", "32", "33"} := by sorry

/-- Proof goal -/
theorem encode_MATEMATIKA : encode "MATEMATIKA" = "2232331122323323132" := by
  exact sorry

end encode_MATEMATIKA_l207_207555


namespace monic_polynomial_roots_l207_207835

theorem monic_polynomial_roots (r1 r2 r3 : ℝ) (h : ∀ x : ℝ, x^3 - 4*x^2 + 5 = 0 ↔ x = r1 ∨ x = r2 ∨ x = r3) :
  ∀ x : ℝ, x^3 - 12*x^2 + 135 = 0 ↔ x = 3*r1 ∨ x = 3*r2 ∨ x = 3*r3 :=
by
  sorry

end monic_polynomial_roots_l207_207835


namespace probability_two_blue_l207_207527

-- Define the conditions of the problem
def total_balls : ℕ := 15
def blue_balls : ℕ := 5
def red_balls : ℕ := 10
def balls_drawn : ℕ := 6
def blue_needed : ℕ := 2
def red_needed : ℕ := 4

-- Calculate the total number of ways to choose 6 balls out of 15
def total_outcomes : ℕ := Nat.choose total_balls balls_drawn

-- Calculate the number of favorable outcomes (2 blue, 4 red)
def blue_combinations : ℕ := Nat.choose blue_balls blue_needed
def red_combinations : ℕ := Nat.choose red_balls red_needed
def favorable_outcomes : ℕ := blue_combinations * red_combinations

-- Calculate the probability of 2 blue balls out of 6 drawn
def probability : ℚ := favorable_outcomes /. total_outcomes

theorem probability_two_blue :
  probability = 2100 /. 5005 := by
  sorry

end probability_two_blue_l207_207527


namespace solve_rational_equation_solve_quadratic_equation_l207_207441

-- Statement for the first equation
theorem solve_rational_equation (x : ℝ) (h : x ≠ 1) : 
  (x / (x - 1) + 2 / (1 - x) = 2) → (x = 0) :=
by intro h1; sorry

-- Statement for the second equation
theorem solve_quadratic_equation (x : ℝ) : 
  (2 * x^2 + 6 * x - 3 = 0) → (x = 1/2 ∨ x = -3) :=
by intro h1; sorry

end solve_rational_equation_solve_quadratic_equation_l207_207441


namespace largest_prime_factor_12321_l207_207645

theorem largest_prime_factor_12321 : 
  ∃ p : ℕ, p.prime ∧ p ∣ 12321 ∧ ∀ q : ℕ, q.prime ∧ q ∣ 12321 → q ≤ p :=
sorry

end largest_prime_factor_12321_l207_207645


namespace smallest_b_for_perfect_square_l207_207921

theorem smallest_b_for_perfect_square : ∃ b : ℕ, b > 5 ∧ (∃ n : ℕ, 4 * b + 5 = n^2) ∧ ∀ b' : ℕ, b' > 5 → (∃ n' : ℕ, 4 * b' + 5 = n'^2) → b ≤ b' :=
by { use 11, split, linarith, split, use 7, norm_num, intros b' hb' hb'n', rcases hb'n' with ⟨n', hn'⟩, linarith }

end smallest_b_for_perfect_square_l207_207921


namespace kira_total_time_l207_207904

noncomputable def total_travel_time (travel_times: List ℕ) : ℕ :=
  travel_times.sum

noncomputable def total_break_time (break_times: List ℕ) : ℕ :=
  break_times.sum

noncomputable def total_time (travel_times: List ℕ) (break_times: List ℕ) : ℕ :=
  total_travel_time travel_times + total_break_time break_times

def kira_conditions : Prop :=
  let travel_times := [180, 120, 90, 240, 60, 150] -- In minutes
  let break_times := [45, 30, 15]
  total_time travel_times break_times = 930

theorem kira_total_time : kira_conditions :=
by
  let travel_times := [180, 120, 90, 240, 60, 150] -- In minutes
  let break_times := [45, 30, 15]
  have travel_sum := total_travel_time travel_times
  have break_sum := total_break_time break_times
  have tt := total_time travel_times break_times
  calc
    tt = travel_sum + break_sum : by rfl
    ... = 930 : by sorry -- Sum calculation can be verified here

end kira_total_time_l207_207904


namespace PQ_passes_through_fixed_point_l207_207274

noncomputable def fixed_point (r a : ℝ) (h : a ≠ 0) : ℝ × ℝ :=
  (r^2 / a, 0)

theorem PQ_passes_through_fixed_point (r a : ℝ) (h : a ≠ 0) :
  ∃ P Q : ℝ × ℝ, 
    let l := (a : ℝ)
    ∧ let M := (a, 0)
    ∧ let A₁ := (r, 0)
    ∧ let A₂ := (-r, 0)
    ∧ M.1 = l
    ∧ P ≠ Q 
    ∧ (P.1 - M.1) * (A₂.2 - M.2) - (P.2 - M.2) * (A₂.1 - M.1) = 0
    ∧ (Q.1 - M.1) * (A₁.2 - M.2) - (Q.2 - M.2) * (A₁.1 - M.1) = 0
    ∧ (Q.2 - P.2) / (Q.1 - P.1) = (M.2 - P.2) / (M.1 - P.1)
    → (let PQ := (Q.2 - P.2) * (fixed_point r a h).1 + (P.1 - Q.1) * (fixed_point r a h).2 + P.1 * Q.2 - Q.1 * P.2 
       ∧ PQ = 0) :=
sorry

end PQ_passes_through_fixed_point_l207_207274


namespace inequality_solution_l207_207074

noncomputable def log_b (b x : ℝ) := Real.log x / Real.log b

noncomputable def lhs (x : ℝ) := 
  log_b 5 250 + ((4 - (log_b 5 2) ^ 2) / (2 + log_b 5 2))

noncomputable def rhs (x : ℝ) := 
  125 ^ (log_b 5 x) ^ 2 - 24 * x ^ (log_b 5 x)

theorem inequality_solution (x : ℝ) : 
  (lhs x <= rhs x) ↔ (0 < x ∧ x ≤ 1/5) ∨ (5 ≤ x) := 
sorry

end inequality_solution_l207_207074


namespace problem1_problem2_l207_207316

-- Part (Ⅰ)
noncomputable def vector_m (x : ℝ) : ℝ × ℝ :=
  (Real.sqrt 2 * Real.cos (x / 4), 2 * Real.cos (x / 4))

noncomputable def vector_n (x : ℝ) : ℝ × ℝ :=
  (Real.sqrt 2 * Real.cos (x / 4), Real.sqrt 3 * Real.sin (x / 4))

noncomputable def f (x : ℝ) : ℝ :=
  vector_m x.1 * vector_n x.1 + vector_m x.2 * vector_n x.2

theorem problem1 (α : ℝ) (h : f α = 2) : Real.cos (α + Real.pi / 3) = 1 / 2 := 
sorry

-- Part (Ⅱ)
variables {A B C : ℝ} {a b c : ℝ}

theorem problem2  (h₁ : (2 * a - b) * Real.cos C = c * Real.cos B) :
(2 < f A) ∧ (f A < 3) :=
sorry

end problem1_problem2_l207_207316


namespace avg_people_per_hour_l207_207010

theorem avg_people_per_hour (num_people : ℕ) (days : ℕ) (hours_per_day : ℕ) : 
  num_people = 4000 → days = 5 → hours_per_day = 24 → 
  Nat.round ((num_people : ℝ) / (hours_per_day * days : ℝ)) = 33 :=
by
  sorry

end avg_people_per_hour_l207_207010


namespace rational_numbers_countable_l207_207018

theorem rational_numbers_countable : ∃ (f : ℚ → ℕ), Function.Bijective f :=
by
  sorry

end rational_numbers_countable_l207_207018


namespace final_price_correct_l207_207561

open BigOperators

-- Define the constants used in the problem
def original_price : ℝ := 500
def first_discount : ℝ := 0.25
def second_discount : ℝ := 0.10
def state_tax : ℝ := 0.05

-- Define the calculation steps
def price_after_first_discount : ℝ := original_price * (1 - first_discount)
def price_after_second_discount : ℝ := price_after_first_discount * (1 - second_discount)
def final_price : ℝ := price_after_second_discount * (1 + state_tax)

-- Prove that the final price is 354.375
theorem final_price_correct : final_price = 354.375 :=
by
  sorry

end final_price_correct_l207_207561


namespace sample_division_l207_207770

theorem sample_division :
  ∀ (capacity max min class_width : ℕ), capacity = 60 → max = 123 → min = 41 → class_width = 10 →
  nat.ceil ((max - min) / class_width) = 9 :=
by
  intros capacity max min class_width hc hmax hmin hcw
  rw [hc, hmax, hmin, hcw]
  sorry

end sample_division_l207_207770


namespace polar_r_eq_3_is_circle_l207_207245

theorem polar_r_eq_3_is_circle :
  ∀ θ : ℝ, ∃ x y : ℝ, (x, y) = (3 * Real.cos θ, 3 * Real.sin θ) ∧ x^2 + y^2 = 9 :=
by
  sorry

end polar_r_eq_3_is_circle_l207_207245


namespace smallest_b_for_45_b_square_l207_207923

theorem smallest_b_for_45_b_square :
  ∃ b : ℕ, b > 5 ∧ ∃ n : ℕ, 4 * b + 5 = n^2 ∧ b = 11 :=
by
  sorry

end smallest_b_for_45_b_square_l207_207923


namespace sum_of_y_l207_207113

variables {x_1 x_2 x_3 x_4 x_5 y_1 y_2 y_3 y_4 y_5 : ℝ}

-- Conditions from the problem
def conditions (h1 : x_1 + x_2 + x_3 + x_4 + x_5 = 150) 
               (h2 : ∀ i, i ∈ [1, 2, 3, 4, 5] → 
                     let x := [x_1, x_2, x_3, x_4, x_5], y := [y_1, y_2, y_3, y_4, y_5] in
                     y.nth i = some (0.67 * x.nth i + 54.9)) : Prop :=
  true

-- Statement of the theorem to prove
theorem sum_of_y (h1 : x_1 + x_2 + x_3 + x_4 + x_5 = 150)
                 (h2 : ∀ i, i ∈ [1, 2, 3, 4, 5] → 
                     let x := [x_1, x_2, x_3, x_4, x_5], y := [y_1, y_2, y_3, y_4, y_5] in
                     y.nth i = some (0.67 * x.nth i + 54.9)) :
  y_1 + y_2 + y_3 + y_4 + y_5 = 375 :=
sorry

end sum_of_y_l207_207113


namespace tens_place_of_smallest_number_l207_207659

theorem tens_place_of_smallest_number (d1 d2 d3 : ℕ) (h1 : d1 = 0 ∧ d2 = 5 ∧ d3 = 8) :
  let smallest_number := if d1 = 0 then d2 * 100 + d1 * 10 + d3 else min (d1 * 100 + d2 * 10 + d3) (d1 * 100 + d3 * 10 + d2)
  in (smallest_number / 10) % 10 = 0 :=
by
  sorry

end tens_place_of_smallest_number_l207_207659


namespace factorial_sum_mod_26_l207_207990

theorem factorial_sum_mod_26 :
  (∑ n in Finset.range 26, n.factorial) % 26 = 0 :=
by
  sorry

end factorial_sum_mod_26_l207_207990


namespace bottles_needed_l207_207579

-- Define specific values provided in conditions
def servings_per_guest : ℕ := 2
def number_of_guests : ℕ := 120
def servings_per_bottle : ℕ := 6

-- Define total servings needed
def total_servings : ℕ := servings_per_guest * number_of_guests

-- Define the number of bottles needed (as a proof statement)
theorem bottles_needed : total_servings / servings_per_bottle = 40 := by
  /-
    The proof will go here. For now we place a sorry to mark the place where
    a proof would be required. The statement should check the equivalence of 
    number of bottles needed being 40 given the total servings divided by 
    servings per bottle.
  -/
  sorry

end bottles_needed_l207_207579


namespace rectangular_field_area_l207_207140

theorem rectangular_field_area (a b c : ℕ) (h1 : a = 15) (h2 : c = 17)
  (h3 : a * a + b * b = c * c) : a * b = 120 := by
  sorry

end rectangular_field_area_l207_207140


namespace largest_n_for_positive_sum_l207_207289

theorem largest_n_for_positive_sum (a : ℕ → ℝ) (n : ℕ) 
  (h_arith : ∀ n, a (n+1) = a n + d) 
  (h_a1_pos : a 1 > 0) 
  (h_sum_pos : a 2015 + a 2016 > 0) 
  (h_prod_neg : a 2015 * a 2016 < 0) :
  ∃ n, n = 4030 ∧ (∑ i in range (n+1), a i) > 0 
      ∧ (∀ m, m > 4030 → (∑ i in range (m+1), a i) ≤ 0) := 
sorry

end largest_n_for_positive_sum_l207_207289


namespace silver_volume_l207_207161
noncomputable def wire_volume : ℝ :=
  let r := 0.0005 in  -- with diameter 1 mm, radius in meters
  let h := 56.02253996834716 in 
  let V := Real.pi * r^2 * h in
  V

theorem silver_volume : 
  wire_volume * (1000^3) = 44.017701934265 := 
by
  sorry

end silver_volume_l207_207161


namespace real_roots_range_real_roots_specific_value_l207_207726

-- Part 1
theorem real_roots_range (a b m : ℝ) (h_eq : a ≠ 0) (h_discriminant : b^2 - 4 * a * m ≥ 0) :
  m ≤ (b^2) / (4 * a) :=
sorry

-- Part 2
theorem real_roots_specific_value (x1 x2 m : ℝ) (h_sum : x1 + x2 = 4) (h_product : x1 * x2 = m)
  (h_condition : x1^2 + x2^2 + (x1 * x2)^2 = 40) (h_range : m ≤ 4) :
  m = -4 :=
sorry

end real_roots_range_real_roots_specific_value_l207_207726


namespace probability_of_A_l207_207141

theorem probability_of_A :
  ∀ (P : set (set ℝ) → ℝ) (A B : set ℝ),
  (independent P [A, B]) →
  (0 < P A) →
  (P A = 2 * P B) →
  (P (A ∪ B) = 5 * P (A ∩ B)) →
  P A = 1 / 2 := by
  sorry

end probability_of_A_l207_207141


namespace actual_average_speed_l207_207507

theorem actual_average_speed (v t : ℝ) (h1 : v > 0) (h2: t > 0) (h3 : (t / (t - (1 / 4) * t)) = ((v + 12) / v)) : v = 36 :=
by
  sorry

end actual_average_speed_l207_207507


namespace arithmetic_seq_sum_l207_207997

-- Define a sequence and sum of the first n terms
def arithmetic_sequence (a : ℕ → ℝ) := ∀ n : ℕ, ∃ d : ℝ, a (n + 1) = a 1 + n * d
def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) := (n / 2) * (a 1 + a n)

-- Given conditions and definition for S5
theorem arithmetic_seq_sum (a : ℕ → ℝ) (d : ℝ) (h : a 1 + a 3 + a 5 = 3) :
  sum_first_n_terms a 5 = 5 :=
by
  -- Here we skip the proof steps and only define the problem statement
  sorry

end arithmetic_seq_sum_l207_207997


namespace watermelon_prices_l207_207909

noncomputable def morning_price := 3.75
noncomputable def afternoon_price := 1.25

variables {x y z : ℕ} {m n : ℝ}

axiom H1 : ∀ m, ∀ n, ∀ x, (m > n) → (x < 10) → (m * x + n * (10 - x) = 35) → False
axiom H2 : ∀ m, ∀ n, ∀ y, (m > n) → (y < 16) → (m * y + n * (16 - y) = 35) → False
axiom H3 : ∀ m, ∀ n, ∀ z, (m > n) → (z < 26) → (m * z + n * (26 - z) = 35) → False

theorem watermelon_prices :
  ∃ x y z m n, (m > n) ∧ (x < 10) ∧ (y < 16) ∧ (z < 26) ∧
    (m = morning_price) ∧ (n = afternoon_price) ∧
    (m * x + n * (10 - x) = 35) ∧ 
    (m * y + n * (16 - y) = 35) ∧
    (m * z + n * (26 - z) = 35) :=
begin
  -- Proof is omitted
  sorry
end

end watermelon_prices_l207_207909


namespace g_2025_divisors_count_l207_207672

noncomputable def g (m : ℕ) : ℕ :=
2^m * 5^m

theorem g_2025_divisors_count : 
  g(2025).num_divisors = 4104676 := 
sorry

end g_2025_divisors_count_l207_207672


namespace black_to_white_area_ratio_l207_207968

noncomputable def radius1 : ℝ := 2
noncomputable def radius2 : ℝ := 4
noncomputable def radius3 : ℝ := 6
noncomputable def radius4 : ℝ := 8
noncomputable def radius5 : ℝ := 10

noncomputable def area (r : ℝ) : ℝ := Real.pi * r^2

noncomputable def black_area : ℝ :=
  area radius1 + (area radius3 - area radius2) + (area radius5 - area radius4)

noncomputable def white_area : ℝ :=
  (area radius2 - area radius1) + (area radius4 - area radius3)

theorem black_to_white_area_ratio :
  black_area / white_area = 3 / 2 := by
  sorry

end black_to_white_area_ratio_l207_207968


namespace lemons_for_lemonade_l207_207517

theorem lemons_for_lemonade (lemons gallons : ℝ) (h1 : lemons = 30) (h2 : gallons = 40) (constant_ratio : lemons / gallons = 3 / 4) : 
  (x : ℝ) (gallons_needed : ℝ) (h3 : gallons_needed = 10) : x = 7.5 :=
by
  sorry

end lemons_for_lemonade_l207_207517


namespace find_a_value_l207_207735

def line1 (a : ℝ) (x y : ℝ) : ℝ := a * x + (a + 2) * y + 1
def line2 (a : ℝ) (x y : ℝ) : ℝ := a * x - y + 2

-- Define what it means for two lines to be not parallel
def not_parallel (a : ℝ) : Prop :=
  ∀ x y : ℝ, (line1 a x y ≠ 0 ∧ line2 a x y ≠ 0)

theorem find_a_value (a : ℝ) (h : not_parallel a) : a = 0 ∨ a = -3 :=
  sorry

end find_a_value_l207_207735


namespace circle_area_isosceles_triangle_l207_207534

theorem circle_area_isosceles_triangle : 
  ∀ (A B C : Type) (AB AC : Type) (a b c : ℝ),
  a = 5 →
  b = 5 →
  c = 4 →
  isosceles_triangle A B C a b c →
  circle_passes_through_vertices A B C →
  ∃ (r : ℝ), 
    area_of_circle_passing_through_vertices A B C = (15625 * π) / 1764 :=
by intros A B C AB AC a b c ha hb hc ht hcirc
   sorry

end circle_area_isosceles_triangle_l207_207534


namespace blackRhinoCount_correct_l207_207150

noncomputable def numberOfBlackRhinos : ℕ :=
  let whiteRhinoCount := 7
  let whiteRhinoWeight := 5100
  let blackRhinoWeightInTons := 1
  let totalWeight := 51700
  let oneTonInPounds := 2000
  let totalWhiteRhinoWeight := whiteRhinoCount * whiteRhinoWeight
  let totalBlackRhinoWeight := totalWeight - totalWhiteRhinoWeight
  totalBlackRhinoWeight / (blackRhinoWeightInTons * oneTonInPounds)

theorem blackRhinoCount_correct : numberOfBlackRhinos = 8 := by
  sorry

end blackRhinoCount_correct_l207_207150


namespace integral_solution_l207_207588

noncomputable def integral_problem : ℝ :=
  ∫ x in -1..1, 2 * real.sqrt(1 - x^2) - real.sin x

theorem integral_solution : integral_problem = real.pi := by
  sorry

end integral_solution_l207_207588


namespace find_mn_l207_207013

variable (OA OB OC : EuclideanSpace ℝ (Fin 3))
variable (AOC BOC : ℝ)

axiom length_OA : ‖OA‖ = 2
axiom length_OB : ‖OB‖ = 2
axiom length_OC : ‖OC‖ = 2 * Real.sqrt 3
axiom tan_angle_AOC : Real.tan AOC = 3 * Real.sqrt 3
axiom angle_BOC : BOC = Real.pi / 3

theorem find_mn : ∃ m n : ℝ, OC = m • OA + n • OB ∧ m = 5 / 3 ∧ n = 2 * Real.sqrt 3 := by
  sorry

end find_mn_l207_207013


namespace stratified_sampling_correct_l207_207522

def total_employees : ℕ := 150
def senior_titles : ℕ := 15
def intermediate_titles : ℕ := 45
def general_staff : ℕ := 90
def sample_size : ℕ := 30

def stratified_sampling (total_employees senior_titles intermediate_titles general_staff sample_size : ℕ) : (ℕ × ℕ × ℕ) :=
  (senior_titles * sample_size / total_employees, 
   intermediate_titles * sample_size / total_employees, 
   general_staff * sample_size / total_employees)

theorem stratified_sampling_correct :
  stratified_sampling total_employees senior_titles intermediate_titles general_staff sample_size = (3, 9, 18) :=
  by sorry

end stratified_sampling_correct_l207_207522


namespace parabola_focus_line_intersection_l207_207287

theorem parabola_focus_line_intersection :
  let F := (1, 0)
  let parabola (x y : ℝ) := y^2 = 4 * x
  let line (x y : ℝ) := y = sqrt(3) * (x - 1)
  let A_x := 3
  let B_x := 1 / 3
  let A := (A_x, sqrt(3) * (A_x - 1))
  let B := (B_x, sqrt(3) * (B_x - 1))
  dist_sq (1, 0) A - dist_sq (1, 0) B = 128 / 9 := sorry

def dist_sq (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

end parabola_focus_line_intersection_l207_207287


namespace count_integers_200_to_250_l207_207319

theorem count_integers_200_to_250 :
  ∃ (count : ℕ), count = 11 ∧
    ∀ n, 200 ≤ n ∧ n < 250 →
      (let digits := [n / 100, (n / 10) % 10, n % 10] in
       (digits.nth 0 = some 2) ∧
       (digits.nth 0 ≠ digits.nth 1) ∧
       (digits.nth 1 ≠ digits.nth 2) ∧
       (digits.nth 0 < digits.nth 1) ∧
       (digits.nth 1 < digits.nth 2)) →
      (n = 234 ∨ n = 235 ∨ n = 236 ∨ n = 237 ∨ n = 238 ∨ n = 239 ∨
       n = 245 ∨ n = 246 ∨ n = 247 ∨ n = 248 ∨ n = 249).
Proof
  sorry

end count_integers_200_to_250_l207_207319


namespace problem_correct_l207_207590

theorem problem_correct (x : ℝ) : 
  14 * ((150 / 3) + (35 / 7) + (16 / 32) + x) = 777 + 14 * x := 
by
  sorry

end problem_correct_l207_207590


namespace max_PM_minus_PN_l207_207854

def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 15 = 1

def circle1 (x y : ℝ) : Prop :=
  (x + 4)^2 + y^2 = 4

def circle2 (x y : ℝ) : Prop :=
  (x - 4)^2 + y^2 = 4

theorem max_PM_minus_PN (P M N : ℝ × ℝ) (hP : hyperbola P.1 P.2) 
    (hM : circle1 M.1 M.2) (hN : circle2 N.1 N.2) :
    let dF1 := (P.1 + 4)^2 + P.2^2
    let dF2 := (P.1 - 4)^2 + P.2^2
    max_PM_minus_PN = sqrt dF1 + 2 - (sqrt dF2 - 2) ≤ 6 := 
sorry 

end max_PM_minus_PN_l207_207854


namespace ratio_XA_XY_l207_207782

-- Define conditions
variables (ABCD_sq XYZ_tr : Type) [IsSquare ABCD_sq] [IsTriangle XYZ_tr]
variable (area_ABCD : ℝ)
variable (area_XYZ : ℝ)
variable (XA XY : ℝ)

-- Area condition
axiom area_condition : area_ABCD = (7 / 32) * area_XYZ
-- Square area relation
axiom square_area_relation : area_ABCD = XA * XA
-- Triangle area expression
axiom triangle_area_relation : area_XYZ = XY * XY -- Simplified area for a specific right triangle configuration

-- Proof goal: the ratio between XA and XY is either 1/8 or 7/8
theorem ratio_XA_XY : XA / XY = 1 / 8 ∨ XA / XY = 7 / 8 := by
  sorry

end ratio_XA_XY_l207_207782


namespace contradiction_example_l207_207501

theorem contradiction_example (a b c : ℕ) : (¬ (a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0)) → (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1) :=
by
  sorry

end contradiction_example_l207_207501


namespace major_premise_wrong_l207_207905

theorem major_premise_wrong (f : ℝ → ℝ) (h_diff : ∀ x, DifferentiableAt ℝ f x) (h_ext : ∀ x₀, f.deriv x₀ = 0 → ¬ (∀ ε > 0 , ∃ δ > 0, (∀ x, abs (x - x₀) < δ → f.deriv x * (x - x₀) > 0) ∨ (∀ x, abs (x - x₀) < δ → f.deriv x * (x - x₀) < 0)))
  (h_f_x3 : ∀ x, f x = x ^ 3) : 
  ¬ (∀ x₀ : ℝ, f.deriv x₀ = 0 → ∃ x : ℝ, f x = x₀) :=
by
  sorry

end major_premise_wrong_l207_207905


namespace area_difference_of_circles_l207_207332

theorem area_difference_of_circles 
  (r1 : ℝ) (t : ℝ) (r2 : ℝ) (h1 : r1 = 12) (h2 : t = 2) (h3 : r2 = 10) :
  real.pi * ((r1 + t)^2 - r2^2) = 96 * real.pi := by
  sorry

end area_difference_of_circles_l207_207332


namespace water_added_l207_207158

theorem water_added (W : ℝ) : 
  let initial_volume : ℝ := 11
  let initial_percentage : ℝ := 0.16
  let final_percentage : ℝ := 7.333333333333333 / 100
  let amount_of_alcohol : ℝ := initial_percentage * initial_volume
  let new_volume : ℝ := initial_volume + W in
  amount_of_alcohol = final_percentage * new_volume →
  W ≈ 13 :=
by 
  -- The exact proof steps should go here
  sorry

end water_added_l207_207158


namespace club_officer_election_l207_207164

theorem club_officer_election:
  let n := 30 -- number of members
  let k := 3 -- number of officers
  let special := {Alice, Bob, Charlie} -- special members
  let others := 27 -- number of other members
  let case1 := others * (others - 1) * (others - 2)
  let pairs := 3 * 2
  let case2 := pairs * others * 3
  let case3 := 3!
  case1 + case2 + case3 = 18042 :=
by sorry

end club_officer_election_l207_207164


namespace quadratic_expression_with_factor_l207_207944

-- Define that (x + 3) is a factor of a quadratic expression, and m = 2.
theorem quadratic_expression_with_factor (x c : ℝ) : 
  let m := 2 in 
  ∃ f : ℝ → ℝ, (∀ x, f(x) = m * (x + 3) * (x + c)) :=
by
  sorry

end quadratic_expression_with_factor_l207_207944


namespace brownie_to_bess_ratio_l207_207622

-- Define daily milk production
def bess_daily_milk : ℕ := 2
def daisy_daily_milk : ℕ := bess_daily_milk + 1

-- Calculate weekly milk production
def bess_weekly_milk : ℕ := bess_daily_milk * 7
def daisy_weekly_milk : ℕ := daisy_daily_milk * 7

-- Given total weekly milk production
def total_weekly_milk : ℕ := 77
def combined_bess_daisy_weekly_milk : ℕ := bess_weekly_milk + daisy_weekly_milk
def brownie_weekly_milk : ℕ := total_weekly_milk - combined_bess_daisy_weekly_milk

-- Main proof statement
theorem brownie_to_bess_ratio : brownie_weekly_milk / bess_weekly_milk = 3 :=
by
  -- Skip the proof
  sorry

end brownie_to_bess_ratio_l207_207622


namespace circle_area_from_equation_l207_207915

theorem circle_area_from_equation :
  (∀ x y : ℝ, x^2 + y^2 - 4 * x + 6 * y = -9) →
  ∃ (r : ℝ), (r = 2) ∧
    (∃ (A : ℝ), A = π * r^2 ∧ A = 4 * π) :=
by {
  -- Conditions included as hypothesis
  sorry -- Proof to be provided here
}

end circle_area_from_equation_l207_207915


namespace wheels_in_garage_l207_207900

theorem wheels_in_garage :
  let bicycles := 9
  let cars := 16
  let single_axle_trailers := 5
  let double_axle_trailers := 3
  let wheels_per_bicycle := 2
  let wheels_per_car := 4
  let wheels_per_single_axle_trailer := 2
  let wheels_per_double_axle_trailer := 4
  let total_wheels := bicycles * wheels_per_bicycle + cars * wheels_per_car + single_axle_trailers * wheels_per_single_axle_trailer + double_axle_trailers * wheels_per_double_axle_trailer
  total_wheels = 104 := by
  sorry

end wheels_in_garage_l207_207900


namespace area_ADP_l207_207877

variable (S_ABP S_BCP S_CDP : ℝ)

theorem area_ADP (h1: S_ABP > 0) (h2: S_BCP > 0) (h3: S_CDP > 0) :
    ∃ (S_ADP : ℝ), S_ADP = (S_ABP * S_CDP) / S_BCP :=
by
  use (S_ABP * S_CDP) / S_BCP
  sorry

end area_ADP_l207_207877


namespace trapezoid_length_KLMN_l207_207791

variables {K L M N P Q : Type}
variables (trapezoid KLMN : K L M N)
variable (KM : ℝ) (KP MQ LM MP : ℝ)
variables (perp1 : KP > 0) (perp2 : MQ > 0)
variables (equal1 : KM = 1) (equal2 : KP = MQ) (equal3 : LM = MP)

theorem trapezoid_length_KLMN
(equality_KM: KM = 1)
(equality_KP_MQ: KP = MQ)
(equality_LM_MP: LM = MP)
: LM = sqrt 2 := 
by sorry

end trapezoid_length_KLMN_l207_207791


namespace polyhedron_with_all_square_faces_is_cube_l207_207609

-- Definition of a polyhedron, convex polyhedron, and properties
structure Polyhedron :=
  (faces : Set (Set (ℝ × ℝ × ℝ)))
  (is_polyhedron : true)  -- Simplified definition; actual definition would be complex

def ConvexPolyhedron (P : Polyhedron) : Prop :=
  ∀ (p1 p2 : ℝ × ℝ × ℝ), 
    p1 ∈ P.faces → p2 ∈ P.faces → 
    (∀ t ∈ Set.Icc 0 1, t * p1 + (1 - t) * p2 ∈ P.faces)

def AllFacesAreSquares (P : Polyhedron) : Prop :=
  ∀ f ∈ P.faces, ∃ a, ∃ b, f = { (x, y, z) | x = a ∧ y = b ∧ z = 0 }

def ThreeFacesPerVertex (P : Polyhedron) : Prop :=
  ∀ v ∈ (⋃ (f ∈ P.faces), f), (∃! v.1, ∃! v.2, (v, v.1) ∈ v.2)

-- Problem: Proving existence
theorem polyhedron_with_all_square_faces_is_cube (P : Polyhedron) :
  ConvexPolyhedron P → AllFacesAreSquares P → ThreeFacesPerVertex P → P = cube :=
sorry

end polyhedron_with_all_square_faces_is_cube_l207_207609


namespace reduced_price_is_correct_l207_207966

-- Definitions for the conditions in the problem
def original_price_per_dozen (P : ℝ) : Prop :=
∀ (X : ℝ), X * P = 40.00001

def reduced_price_per_dozen (P R : ℝ) : Prop :=
R = 0.60 * P

def bananas_purchased_additional (P R : ℝ) : Prop :=
∀ (X Y : ℝ), (Y = X + (64 / 12)) → (X * P = Y * R) 

-- Assertion of the proof problem
theorem reduced_price_is_correct : 
  ∃ (R : ℝ), 
  (∀ P, original_price_per_dozen P ∧ reduced_price_per_dozen P R ∧ bananas_purchased_additional P R) → 
  R = 3.00000075 := 
by sorry

end reduced_price_is_correct_l207_207966


namespace volume_of_water_in_cylinder_l207_207543

-- Define the conditions
constant radius : ℝ := 5
constant height : ℝ := 10
constant water_depth : ℝ := 3

-- Define the theorem
theorem volume_of_water_in_cylinder :
  let r := radius,
      h := height,
      d := water_depth
  in (∃ V : ℝ, V = 116.377 * real.pi - 120) :=
begin
  sorry
end

end volume_of_water_in_cylinder_l207_207543


namespace sum_of_series_eq_6_div_7_l207_207599

noncomputable def infinite_series : ℕ → ℚ
| n => if even n then (3 ^ (n/2)) / (2 ^ (n/2 + 1)) else -(3 ^ (n/2)) / (2 ^ (n/2 + 1))

theorem sum_of_series_eq_6_div_7 : 
  let S := (∑' n, infinite_series n) in 
  S = (6 / 7) := 
  by
  sorry

end sum_of_series_eq_6_div_7_l207_207599


namespace david_marks_in_english_l207_207603

variable (E : ℕ)
variable (marks_in_math : ℕ := 98)
variable (marks_in_physics : ℕ := 99)
variable (marks_in_chemistry : ℕ := 100)
variable (marks_in_biology : ℕ := 98)
variable (average_marks : ℚ := 98.2)
variable (num_subjects : ℕ := 5)

theorem david_marks_in_english 
  (H1 : average_marks = (E + marks_in_math + marks_in_physics + marks_in_chemistry + marks_in_biology) / num_subjects) :
  E = 96 :=
sorry

end david_marks_in_english_l207_207603


namespace union_of_sets_l207_207697

-- We first define the sets A and B using the given conditions.
def setA : Set ℝ := { x | x^2 - 2 * x < 0 }
def setB : Set ℝ := { y | ∃ x : ℝ, y = real.log _2 (2 - x^2) }

-- Now we assert the theorem stating that the union of these sets equals (-∞, 2).
theorem union_of_sets :
  setA ∪ setB = { z | z < 2 } := 
sorry

end union_of_sets_l207_207697


namespace luisa_trip_l207_207405

noncomputable def additional_miles (d1: ℝ) (s1: ℝ) (s2: ℝ) (desired_avg_speed: ℝ) : ℝ := 
  let t1 := d1 / s1
  let t := (d1 * (desired_avg_speed - s1)) / (s2 * (s1 - desired_avg_speed))
  s2 * t

theorem luisa_trip :
  additional_miles 18 36 60 45 = 18 :=
by
  sorry

end luisa_trip_l207_207405


namespace find_LM_l207_207804

variables (K L M N P Q : Type)
variables (KL MN LM KN MQ MP KP KM : ℝ) 

-- Conditions
def trapezoid (K L M N : Type) : Prop := 
  KM = 1 ∧ 
  KP = 1 ∧
  MQ = 1 ∧
  KN = MQ ∧
  LM = MP

-- To Prove
theorem find_LM (h : trapezoid K L M N) : LM = sqrt 2 :=
by sorry

end find_LM_l207_207804


namespace factor_equivalence_l207_207214

noncomputable def given_expression (x : ℝ) :=
  (3 * x^3 + 70 * x^2 - 5) - (-4 * x^3 + 2 * x^2 - 5)

noncomputable def target_form (x : ℝ) :=
  7 * x^2 * (x + 68 / 7)

theorem factor_equivalence (x : ℝ) : given_expression x = target_form x :=
by
  sorry

end factor_equivalence_l207_207214


namespace half_difference_donation_l207_207843

def margoDonation : ℝ := 4300
def julieDonation : ℝ := 4700

theorem half_difference_donation : (julieDonation - margoDonation) / 2 = 200 := by
  sorry

end half_difference_donation_l207_207843


namespace problem_l207_207026

open Real

theorem problem {a : ℝ} (ha : 0 < a) :
  e^a < (λ x, (∑ s in finset.range (nat.succ x), ((a + s) / x) ^ x)) at_top < e^(a + 1) :=
begin
  sorry
end

end problem_l207_207026


namespace pairing_property_l207_207857

open Set

theorem pairing_property (n : ℕ) (h_odd : Odd n) (h_ge3 : 3 ≤ n) :
  ∃ (points : Fin 2n → ℝ × ℝ), 
  (∀ i j : Fin 2n, i ≠ j → points i ≠ points j) ∧ 
  ¬ ∃ l : (Fin 2n → Prop), ∀ i, points i ∈ l → (∀ i j : Fin 2n, i ≠ j → points i ≠ points j) ∧
  (∀ (pairs : Fin n → Fin 2n × Fin 2n), 
    ∀ (i j : Fin n), i ≠ j →
    ∃ k : Fin 2n, 
    (points (pairs i).fst, points (pairs i).snd) ∈ l ∧ 
    (points (pairs j).fst, points (pairs j).snd) ∈ l ∧ 
    k = i → (points k).fst = (pairs i).fst ∧ (points k).snd = (pairs i).snd) :=
sorry

end pairing_property_l207_207857


namespace estimate_pi_l207_207863

theorem estimate_pi (n m : ℕ) (h_n : n = 100) (h_m : m = 31) :
    let area_total := 1
    let area_obtuse_triangle := π / 4 - 1 / 2
    let estimation := (m : ℝ) / (n : ℝ) * area_total
    estimation = area_obtuse_triangle → π = 81 / 25 :=
by 
    intros n m h_n h_m
    let area_total := 1
    let area_obtuse_triangle := π / 4 - 1 / 2
    let estimation := (m : ℝ) / (n : ℝ) * area_total
    have h1 : estimation = area_obtuse_triangle,
    { sorry }
    have h2 : π / 4 - 1 / 2 = 31 / 100,
    { sorry }
    rw [h_n, h_m, h2] at h1,
    sorry

end estimate_pi_l207_207863


namespace value_range_l207_207107

-- Define the function y
def y (x : ℝ) : ℝ := (Real.cos x) ^ 2 + (Real.sqrt 3) * (Real.sin x) * (Real.cos x)

-- Define the interval
def interval : Set ℝ := { x | -Real.pi / 6 ≤ x ∧ x ≤ Real.pi / 4 }

-- Define the expected value range
def expected_range : Set ℝ := { y | 0 ≤ y ∧ y ≤ 3 / 2 }

theorem value_range (h : ∀ x ∈ interval, y x ∈ expected_range) : 
  ∀ x ∈ interval, (y x) ∈ expected_range :=
sorry

end value_range_l207_207107


namespace number_of_solutions_for_z_l207_207168

def f (z : ℂ) : ℂ := complex.I * complex.conj z

theorem number_of_solutions_for_z :
  {z : ℂ | complex.abs z = 5 ∧ f z = z}.to_finset.card = 2 :=
by sorry

end number_of_solutions_for_z_l207_207168


namespace simplify_expression_l207_207071

def expression (p : ℝ) : ℝ :=
  ((7 * p + 4) - 3 * p * 3) * 2 + (5 - 2 / 4) * (4 * p - 6)

theorem simplify_expression (p : ℝ) : expression p = 14 * p - 19 :=
by
  sorry

end simplify_expression_l207_207071


namespace number_of_true_propositions_is_four_l207_207736

-- Definitions given in the problem
variables (α β γ : Type) (a b : Type)

-- Conditions in the problem
variable [perpendicular α γ] 
variable [perpendicular β γ]
variable [intersection α γ = a] 
variable [intersection β γ = b]

-- Propositions extracted from the problem
def prop1 : Prop := (perpendicular a b) → (perpendicular α β)
def prop2 : Prop := (parallel α b) → (parallel α β)
def prop3 : Prop := (perpendicular α β) → (perpendicular a b)
def prop4 : Prop := (parallel α β) → (parallel a b)

-- Main statement: The number of true propositions is 4
theorem number_of_true_propositions_is_four : 
  (prop1 α β γ a b) ∧ (prop2 α β γ a b) ∧ (prop3 α β γ a b) ∧ (prop4 α β γ a b) :=
sorry

end number_of_true_propositions_is_four_l207_207736


namespace log_geometric_and_a_n_formula_sum_first_n_terms_l207_207306

variable (a : ℕ → ℝ)
variable (b : ℕ → ℝ)
variable (S : ℕ → ℝ)

-- conditions for sequence a_n
axiom a₁_eq : a 1 = 2
axiom a_recur : ∀ n : ℕ, 0 < n → a (n + 1) = (a n)^2 + 2 * (a n)

-- the first problem: prove that {log(1 + a_n)} is geometric and find the general term for a_n
theorem log_geometric_and_a_n_formula (n : ℕ) (h : 0 < n) :
  ∃ c r, ( ∀ k : ℕ, 0 < k ∧ k ≤ n → log (1 + a k) = c * r^k ) ∧ a n = 3^(2^(n-1)) - 1 := sorry

-- conditions for sequence b_n
axiom b_def : ∀ n : ℕ, 0 < n → b n = 1 / (a n) + 1 / (a n + 2)

-- the second problem: find the sum of the first n terms S_n of the sequence b_n
theorem sum_first_n_terms (n : ℕ) (h : 0 < n) :
  S n = 2 * (1 / (a 1) - 1 / (a (n + 1))) ∧ S n = 1 - 2 / (3^(2^n) - 1) := sorry

end log_geometric_and_a_n_formula_sum_first_n_terms_l207_207306


namespace largest_prime_factor_of_12321_l207_207632

-- Definitions based on the given conditions
def n := 12321
def a := 111
def p₁ := 3
def p₂ := 37

-- Given conditions as hypotheses
theorem largest_prime_factor_of_12321 (h1 : n = a^2) (h2 : a = p₁ * p₂) (hp₁_prime : Prime p₁) (hp₂_prime : Prime p₂) :
  p₂ = 37 ∧ ∀ p, Prime p → p ∣ n → p ≤ 37 := 
by 
  sorry

end largest_prime_factor_of_12321_l207_207632


namespace complement_intersection_M_N_l207_207046

def M : Set ℝ := {x | x < 3}
def N : Set ℝ := {x | x > -1}
def U : Set ℝ := Set.univ

theorem complement_intersection_M_N :
  U \ (M ∩ N) = {x | x ≤ -1} ∪ {x | x ≥ 3} :=
by
  sorry

end complement_intersection_M_N_l207_207046


namespace factorial_division_l207_207995

theorem factorial_division : (52! / 50!) = 2652 := by
  sorry

end factorial_division_l207_207995


namespace digit_B_for_divisibility_by_9_l207_207118

theorem digit_B_for_divisibility_by_9 :
  ∃! (B : ℕ), B < 10 ∧ (5 + B + B + 3) % 9 = 0 :=
by
  sorry

end digit_B_for_divisibility_by_9_l207_207118


namespace max_car_washes_l207_207819

-- Definitions of variables and conditions
def normal_price : ℝ := 15
def budget : ℝ := 250

-- Definitions of prices at each discount tier
def price_90_percent : ℝ := normal_price * 0.90
def price_80_percent : ℝ := normal_price * 0.80
def price_70_percent : ℝ := normal_price * 0.70

-- Definitions of maximum car washes Jim can buy at each tier based on budget
def max_car_washes_90 : ℕ := (budget / price_90_percent).toNat
def max_car_washes_80 : ℕ := (budget / price_80_percent).toNat
def max_car_washes_70 : ℕ := (budget / price_70_percent).toNat

-- Proof statement: The maximum number of car washes Jim can purchase is 23
theorem max_car_washes : max_car_washes_70 = 23 := sorry

end max_car_washes_l207_207819


namespace parabola_equation_min_area_triangle_l207_207273

variables {x y a b p k x1 x2 : ℝ}
variables {A B P : ℝ × ℝ}

-- The equation of the circle
def circle (a b : ℝ) : Prop := (x - a)^2 + (y - b)^2 = 9 / 4

-- The center of the circle lies on the parabola
def on_parabola (a b p : ℝ) : Prop := (p > 0) ∧ (a^2 = 2 * p * b)

-- Circle passing through the origin
def passes_through_origin (a b : ℝ) : Prop := (a^2 + b^2 = 9 / 4)

-- Circle tangent to the directrix
def tangent_directrix (b p : ℝ) : Prop := (b + p / 2 = 3 / 2)

-- Prove the equation of the parabola is x^2 = 4y
theorem parabola_equation (a b p : ℝ) (hp : on_parabola a b p) (hc : circle a b)
  (hpo : passes_through_origin a b) (htd : tangent_directrix b p) : p = 2 := 
sorry

-- Equation for a line passing through the focus of the parabola
def line_through_focus (k : ℝ) : Prop := y = k*x + 1

-- Two tangents to the parabola at points A, B intersect at point P
def tangents_intersect (x1 x2 k : ℝ) (P : ℝ × ℝ) : Prop := 
  P = (2 * k, -1)

-- Minimum area of triangle PAB and the equation of the line l in this case
theorem min_area_triangle (k : ℝ) : 
  k = 0 → ∃ (S : ℝ) (l : ℝ → ℝ), (S = 4 ∧ ∀ x, l x = 1) :=
sorry

end parabola_equation_min_area_triangle_l207_207273


namespace diophantine_infinite_solutions_l207_207042

theorem diophantine_infinite_solutions
  (l m n : ℕ) (h_l_positive : l > 0) (h_m_positive : m > 0) (h_n_positive : n > 0)
  (h_gcd_lm_n : gcd (l * m) n = 1) (h_gcd_ln_m : gcd (l * n) m = 1) (h_gcd_mn_l : gcd (m * n) l = 1)
  : ∃ x y z : ℕ, (x > 0 ∧ y > 0 ∧ z > 0 ∧ (x ^ l + y ^ m = z ^ n)) ∧ (∀ a b c : ℕ, (a > 0 ∧ b > 0 ∧ c > 0 ∧ (a ^ l + b ^ m = c ^ n)) → ∀ d : ℕ, d > 0 → ∃ e f g : ℕ, (e > 0 ∧ f > 0 ∧ g > 0 ∧ (e ^ l + f ^ m = g ^ n))) :=
sorry

end diophantine_infinite_solutions_l207_207042


namespace value_of_expression_l207_207754

theorem value_of_expression
  (a b c : ℝ)
  (h1 : |a - b| = 1)
  (h2 : |b - c| = 1)
  (h3 : |c - a| = 2)
  (h4 : a * b * c = 60) :
  (a / (b * c) + b / (c * a) + c / (a * b) - 1 / a - 1 / b - 1 / c) = 1 / 10 :=
sorry

end value_of_expression_l207_207754


namespace sally_students_are_30_l207_207864

-- Define the conditions given in the problem
def school_money : ℕ := 320
def book_cost : ℕ := 12
def sally_money : ℕ := 40
def total_students : ℕ := 30

-- Define the total amount Sally can spend on books
def total_amount_available : ℕ := school_money + sally_money

-- The total cost of books for S students
def total_cost (S : ℕ) : ℕ := book_cost * S

-- The main theorem stating that S students will cost the same as the amount Sally can spend
theorem sally_students_are_30 : total_cost 30 = total_amount_available :=
by
  sorry

end sally_students_are_30_l207_207864


namespace H_on_segment_DE_l207_207892

noncomputable def circumcenter (A B C : Point) : Point := sorry -- Detail the definition as per required

structure Triangle (Ω : Type*) :=
(A B C : Ω)
(ac : AcuteTriangle A B C) -- Indicating that ABC is an acute triangle

structure Point (Ω : Type*) :=
(x y : Ω)

structure Circumcircle (Ω : Type*) :=
(circle : Circle)
(point_on_circle : ∀ (X : Point Ω), X ∈ circle.radius)

def orthocenter_of_triangle {Ω : Type*} (T : Triangle Ω) : Point Ω := sorry

def segment (Ω : Type*) (a b : Point Ω) : Set (Point Ω) := sorry

theorem H_on_segment_DE 
  {Ω : Type*} 
  (A B C D E H : Point Ω)
  (T : Triangle Ω) 
  (circumcircle : Circumcircle Ω)
  (ortho : orthocenter_of_triangle T = H)
  (dist_AD : distance A D = distance B C)
  (dist_AE : distance A E = distance B C)
  (eq : distance A H ^ 2 = distance B H ^ 2 + distance C H ^ 2) :
  H ∈ segment Ω D E :=
sorry

end H_on_segment_DE_l207_207892


namespace mike_bricks_l207_207054

theorem mike_bricks (total_bricks bricks_A bricks_B bricks_other: ℕ) 
  (h1 : bricks_A = 40) 
  (h2 : bricks_B = bricks_A / 2)
  (h3 : total_bricks = 150) 
  (h4 : total_bricks = bricks_A + bricks_B + bricks_other) : bricks_other = 90 := 
by 
  sorry

end mike_bricks_l207_207054


namespace equilateral_triangle_cover_points_l207_207684

theorem equilateral_triangle_cover_points
    (points : Fin 1990 → (ℝ × ℝ))
    (side_length_square : ℝ)
    (side_length_triangle : ℝ)
    (points_in_square : ∀ i, 0 ≤ (points i).1 ∧ (points i).1 ≤ side_length_square ∧ 0 ≤ (points i).2 ∧ (points i).2 ≤ side_length_square)
    (side_length_square = 12)
    (side_length_triangle = 11) :
  ∃ (triangle_points : Finₙ 498 → (ℝ × ℝ)), ∀ i, ∃ j, triangle_points i = points j :=
sorry

end equilateral_triangle_cover_points_l207_207684


namespace count_perfect_sixth_powers_less_200_l207_207322

noncomputable def countPerfectSixthPowersUnder(n : ℕ) : ℕ :=
  Nat.card { k : ℕ | ∃ x : ℕ, x > 0 ∧ x^6 = k ∧ k < n }

theorem count_perfect_sixth_powers_less_200 : countPerfectSixthPowersUnder(200) = 2 := by
  sorry

end count_perfect_sixth_powers_less_200_l207_207322


namespace correct_propositions_count_l207_207267

-- Define the necessary types and predicates
variables {Plane Line : Type} (α β : Plane) (m n : Line)

-- Define the conditions
axiom prop1 (h1 : m ∥ n) (h2 : m ⊥ α) : n ⊥ α
axiom prop2 (h1 : m ∥ α) (h2 : ∃ l, α ∩ β = l ∧ l = n) : m ∥ n
axiom prop3 (h1 : m ⊥ α) (h2 : n ⊥ β) (h3 : α ⊥ β) : m ⊥ n
axiom prop4 (h1 : n ⊂ α) (h2 : m ⊂ β) (h3 : α ∥ β) : m ∥ n

-- Statement about number of correct propositions
theorem correct_propositions_count : 
  (∃ h1 h2, prop1 h1 h2) ∧ ¬ (∃ h1 h2, prop2 h1 h2) ∧ ¬ (∃ h1 h2 h3, prop3 h1 h2 h3) ∧ ¬ (∃ h1 h2 h3, prop4 h1 h2 h3) := 
sorry

end correct_propositions_count_l207_207267


namespace allowance_spent_on_games_l207_207743

theorem allowance_spent_on_games (total_money : ℝ) 
  (fraction_books fraction_snacks fraction_music : ℝ) :
  total_money = 50 → fraction_books = 1/4 → fraction_snacks = 1/5 → fraction_music = 2/5
  → total_money - (fraction_books * total_money + fraction_snacks * total_money + fraction_music * total_money) = 7.5 :=
by 
  intros htm hfb hfs hfm
  rw [htm, hfb, hfs, hfm]
  have hbooks : 50 * (1 / 4) = 12.5 := by norm_num
  have hsnacks : 50 * (1 / 5) = 10 := by norm_num
  have hmusic : 50 * (2 / 5) = 20 := by norm_num
  calc
    50 - (12.5 + 10 + 20) = 50 - 42.5 := by congr; exact (add_assoc 12.5 10 20).symm
    ... = 7.5 := by norm_num

end allowance_spent_on_games_l207_207743


namespace first_discount_percentage_l207_207099

theorem first_discount_percentage (original_price : ℝ) (final_price : ℝ) (additional_discount : ℝ) (first_discount : ℝ) : 
  original_price = 600 → final_price = 513 → additional_discount = 0.05 →
  600 * (1 - first_discount / 100) * (1 - 0.05) = 513 →
  first_discount = 10 :=
by
  intros h1 h2 h3 h4
  sorry

end first_discount_percentage_l207_207099


namespace walnut_swap_exists_l207_207115

theorem walnut_swap_exists (n : ℕ) (h_n : n = 2021) :
  ∃ k : ℕ, k < n ∧ ∃ a b : ℕ, a < k ∧ k < b :=
by
  sorry

end walnut_swap_exists_l207_207115
